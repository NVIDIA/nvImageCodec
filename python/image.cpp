/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "image.h"

#include <iostream>

#include <dlpack/dlpack.h>

#include "dlpack_utils.h"
#include "error_handling.h"
#include "type_utils.h"

namespace nvimgcodec {

Image::Image(nvimgcodecInstance_t instance, nvimgcodecImageInfo_t* image_info)
    : instance_(instance)
{
    initBuffer(image_info);

    nvimgcodecImage_t image;
    CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance, &image, image_info));
    image_ =
        std::shared_ptr<std::remove_pointer<nvimgcodecImage_t>::type>(image, [](nvimgcodecImage_t image) { nvimgcodecImageDestroy(image); });
    dlpack_tensor_ = std::make_shared<DLPackTensor>(*image_info, img_buffer_);
    if (image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
        initCudaArrayInterface(*image_info);
        initCudaEventForDLPack();
    } else if (image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        initArrayInterface(*image_info);
    }
}

void Image::initBuffer(nvimgcodecImageInfo_t* image_info)
{
    if (image_info->buffer == nullptr) {
        if (image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
            initDeviceBuffer(image_info);
        } else if (image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            initHostBuffer(image_info);
        } else {
            throw std::runtime_error("Unsupported buffer type.");
        }
    }
}

void Image::initDeviceBuffer(nvimgcodecImageInfo_t* image_info)
{
    unsigned char* buffer;
    CHECK_CUDA(cudaMallocAsync((void**)&buffer, image_info->buffer_size, image_info->cuda_stream));
    auto cuda_stream = image_info->cuda_stream;
    img_buffer_ = std::shared_ptr<unsigned char>(buffer, [cuda_stream](unsigned char* buffer) { cudaFreeAsync(buffer, cuda_stream); });
    image_info->buffer = buffer;
}

void Image::initHostBuffer(nvimgcodecImageInfo_t* image_info)
{
    unsigned char* buffer;
    CHECK_CUDA(cudaMallocHost((void**)&buffer, image_info->buffer_size));
    img_host_buffer_ = std::shared_ptr<unsigned char>(buffer, [](unsigned char* buffer) { cudaFreeHost(buffer); });
    image_info->buffer = buffer;
}

void Image::initDLPack(nvimgcodecImageInfo_t* image_info, py::capsule cap)
{
    if (auto* tensor = static_cast<DLManagedTensor*>(cap.get_pointer())) {
        check_cuda_buffer(tensor->dl_tensor.data);
        dlpack_tensor_ = std::make_shared<DLPackTensor>(tensor);
        // signal that producer don't have to call tensor's deleter, consumer will do it instead
        cap.set_name("used_dltensor");
        dlpack_tensor_->getImageInfo(image_info);
    } else {
        throw std::runtime_error("Unsupported dlpack PyCapsule object.");
    }
}

void Image::initImageInfoFromInterfaceDict(const py::dict& iface, nvimgcodecImageInfo_t* image_info)
{
    std::vector<long> vshape;
    py::tuple shape = iface["shape"].cast<py::tuple>();
    for (auto& o : shape) {
        vshape.push_back(o.cast<long>());
    }
    if (vshape.size() < 3) {
        throw std::runtime_error("Unexpected number of dimensions");
    }

    std::vector<int> vstrides;
    if (iface.contains("strides")) {
        py::object strides = iface["strides"];
        if (!strides.is(py::none())) {
            strides = strides.cast<py::tuple>();
            for (auto& o : strides) {
                vstrides.push_back(o.cast<int>());
            }
        }
    }

    bool is_interleaved = true; //TODO detect interleaved if we have HWC layout

    if (is_interleaved) {
        image_info->num_planes = 1;
        image_info->plane_info[0].height = vshape[0];
        image_info->plane_info[0].width = vshape[1];
        image_info->plane_info[0].num_channels = vshape[2];
    } else {
        image_info->num_planes = vshape[0];
        image_info->plane_info[0].height = vshape[1];
        image_info->plane_info[0].width = vshape[2];
        image_info->plane_info[0].num_channels = 1;
    }

    std::string typestr = iface["typestr"].cast<std::string>();
    auto sample_type = type_from_format_str(typestr);

    int bytes_per_element = sample_type_to_bytes_per_element(sample_type);

    image_info->color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    image_info->sample_format = is_interleaved ? NVIMGCODEC_SAMPLEFORMAT_I_RGB : NVIMGCODEC_SAMPLEFORMAT_P_RGB;
    image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_444;

    int pitch_in_bytes = vstrides.size() > 1 ? (is_interleaved ? vstrides[0] : vstrides[1])
                                             : image_info->plane_info[0].width * image_info->plane_info[0].num_channels * bytes_per_element;
    size_t buffer_size = 0;
    for (size_t c = 0; c < image_info->num_planes; c++) {
        image_info->plane_info[c].width = image_info->plane_info[0].width;
        image_info->plane_info[c].height = image_info->plane_info[0].height;
        image_info->plane_info[c].row_stride = pitch_in_bytes;
        image_info->plane_info[c].sample_type = sample_type;
        image_info->plane_info[c].num_channels = image_info->plane_info[0].num_channels;
        buffer_size += image_info->plane_info[c].row_stride * image_info->plane_info[0].height;
    }
    py::tuple tdata = iface["data"].cast<py::tuple>();
    void* buffer = PyLong_AsVoidPtr(tdata[0].ptr());
    image_info->buffer = buffer;
    image_info->buffer_size = buffer_size;
}

Image::Image(nvimgcodecInstance_t instance, PyObject* o, intptr_t cuda_stream)
    : instance_(instance)
{
    if (!o) {
        throw std::runtime_error("Object cannot be None");
    }
    py::object tmp = py::reinterpret_borrow<py::object>(o);
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    image_info.cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    if (py::isinstance<py::capsule>(tmp)) {
        py::capsule cap = tmp.cast<py::capsule>();
        initDLPack(&image_info, cap);
    } else if (hasattr(tmp, "__cuda_array_interface__")) {
        py::dict iface = tmp.attr("__cuda_array_interface__").cast<py::dict>();

        if (!iface.contains("shape") || !iface.contains("typestr") || !iface.contains("data") || !iface.contains("version")) {
            throw std::runtime_error("Unsupported __cuda_array_interface__ with missing field(s)");
        }

        int version = iface["version"].cast<int>();
        if (version < 2) {
            throw std::runtime_error("Unsupported __cuda_array_interface__ with version < 2");
        }
        initImageInfoFromInterfaceDict(iface, &image_info);
        std::optional<intptr_t> stream =
            version >= 3 && iface.contains("stream") ? iface["stream"].cast<std::optional<intptr_t>>() : std::optional<intptr_t>();

        if (stream.has_value()) {
            if (*stream == 0) {
                throw std::runtime_error("Invalid for stream to be 0");
            } else {
                image_info.cuda_stream = reinterpret_cast<cudaStream_t>(*stream);
            }
        }
        check_cuda_buffer(image_info.buffer);
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        dlpack_tensor_ = std::make_shared<DLPackTensor>(image_info, img_buffer_);
    } else if (hasattr(tmp, "__array_interface__")) {
        py::dict iface = tmp.attr("__array_interface__").cast<py::dict>();

        if (!iface.contains("shape") || !iface.contains("typestr") || !iface.contains("data") || !iface.contains("version")) {
            throw std::runtime_error("Unsupported __array_interface__ with missing field(s)");
        }

        int version = iface["version"].cast<int>();
        if (version < 2) {
            throw std::runtime_error("Unsupported __array_interface__ with version < 2");
        }

        initImageInfoFromInterfaceDict(iface, &image_info);
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        dlpack_tensor_ = std::make_shared<DLPackTensor>(image_info, img_buffer_);
    } else if (hasattr(tmp, "__dlpack__")) {
        // Quickly check if we support the device
        if (hasattr(tmp, "__dlpack_device__")) {
            py::tuple dlpack_device = tmp.attr("__dlpack_device__")().cast<py::tuple>();
            auto dev_type = static_cast<DLDeviceType>(dlpack_device[0].cast<int>());
            if (!is_cuda_accessible(dev_type)) {
                throw std::runtime_error("Unsupported device in DLTensor. Only CUDA-accessible memory buffers can be wrapped");
            }
        }
        py::object py_cuda_stream = cuda_stream ? py::int_((intptr_t)(cuda_stream)) : py::int_(1);
        py::capsule cap = tmp.attr("__dlpack__")(py_cuda_stream).cast<py::capsule>();
        initDLPack(&image_info, cap);
    } else {
        throw std::runtime_error("Object does not support neither __cuda_array_interface__ nor __dlpack__");
    }
    nvimgcodecImage_t image;
    CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance, &image, &image_info));
    image_ =
        std::shared_ptr<std::remove_pointer<nvimgcodecImage_t>::type>(image, [](nvimgcodecImage_t image) { nvimgcodecImageDestroy(image); });
    if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
        initCudaArrayInterface(image_info);
        initCudaEventForDLPack();
    } else if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        initArrayInterface(image_info);
    }
}

void Image::initInterfaceDictFromImageInfo(const nvimgcodecImageInfo_t& image_info, py::dict* d)
{
    std::string format = format_str_from_type(image_info.plane_info[0].sample_type);
    bool is_interleaved = is_sample_format_interleaved(image_info.sample_format) || image_info.num_planes == 1;
    int bytes_per_element = sample_type_to_bytes_per_element(image_info.plane_info[0].sample_type);
    py::tuple strides_tuple = is_interleaved ? py::make_tuple(image_info.plane_info[0].row_stride,
                                                   image_info.plane_info[0].num_channels * bytes_per_element, bytes_per_element)
                                             : py::make_tuple(image_info.plane_info[0].row_stride * image_info.plane_info[0].height,
                                                   image_info.plane_info[0].row_stride, bytes_per_element);

    py::tuple shape_tuple =
        is_interleaved
            ? py::make_tuple(image_info.plane_info[0].height, image_info.plane_info[0].width, image_info.plane_info[0].num_channels)
            : py::make_tuple(image_info.num_planes, image_info.plane_info[0].height, image_info.plane_info[0].width);

    py::object strides = is_interleaved ? py::object(py::none()) : py::object(strides_tuple);

    (*d)["shape"] = shape_tuple;
    (*d)["strides"] = strides;
    (*d)["typestr"] = format;
    (*d)["data"] = py::make_tuple(py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(image_info.buffer)), false);
    (*d)["version"] = 3;
}

void Image::initArrayInterface(const nvimgcodecImageInfo_t& image_info)
{
    try {
        initInterfaceDictFromImageInfo(image_info, &array_interface_);
    } catch (...) {
        throw std::runtime_error("Unable to initialize __array_interface__");
    }
}

void Image::initCudaArrayInterface(const nvimgcodecImageInfo_t& image_info)
{
    try {
        initInterfaceDictFromImageInfo(image_info, &cuda_array_interface_);
        py::object stream = image_info.cuda_stream ? py::int_((intptr_t)(image_info.cuda_stream)) : py::int_(1);
        cuda_array_interface_["stream"] = stream;
    } catch (...) {
        throw std::runtime_error("Unable to initialize __cuda_array_interface__");
    }
}

void Image::initCudaEventForDLPack()
{
    if (!dlpack_cuda_event_) {
        cudaEvent_t event;
        CHECK_CUDA(cudaEventCreate(&event));
        dlpack_cuda_event_ = std::shared_ptr<std::remove_pointer<cudaEvent_t>::type>(event, [](cudaEvent_t e) { cudaEventDestroy(e); });
    }
}

int Image::getWidth() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    return image_info.plane_info[0].width;
}
int Image::getHeight() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    return image_info.plane_info[0].height;
}

int Image::getNdim() const
{
    py::tuple shape_tuple;
    if (cuda_array_interface_.contains("shape")) {
        shape_tuple = cuda_array_interface_["shape"];
    } else if (array_interface_.contains("shape")) {
        shape_tuple = array_interface_["shape"];
    } 

    return shape_tuple.size();
}


py::dict Image::array_interface() const
{
    return array_interface_;
}

py::dict Image::cuda_interface() const
{
    return cuda_array_interface_;
}
py::object Image::shape() const
{
    return cuda_array_interface_["shape"];
}

py::object Image::dtype() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    std::string format = format_str_from_type(image_info.plane_info[0].sample_type);
    return py::dtype(format);
}

int Image::precision() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    return image_info.plane_info[0].precision;
}

nvimgcodecImageBufferKind_t Image::getBufferKind() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    return image_info.buffer_kind;
}

nvimgcodecImage_t Image::getNvImgCdcsImage() const
{
    return image_.get();
}

py::capsule Image::dlpack(py::object stream_obj) const
{
    py::capsule cap = dlpack_tensor_->getPyCapsule();
    if (std::string(cap.name()) != "dltensor") {
        throw std::runtime_error(
            "Could not get DLTensor capsules. It can be consumed only once, so you might have already constructed a tensor from it once.");
    }

    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);

    // Add synchronisation
    std::optional<intptr_t> stream = stream_obj.cast<std::optional<intptr_t>>();
    intptr_t stream_value = stream.has_value() ? *stream : 0;
    static constexpr intptr_t kDoNotSync = -1; // if provided stream is -1, no stream order should be established;
    if (stream_value != kDoNotSync) {
        // the consumer stream should wait for the work on Image stream
        auto consumer_stream = reinterpret_cast<cudaStream_t>(stream_value);
        if (consumer_stream != image_info.cuda_stream) {
            CHECK_CUDA(cudaEventRecord(dlpack_cuda_event_.get(), image_info.cuda_stream));
            CHECK_CUDA(cudaStreamWaitEvent(consumer_stream, dlpack_cuda_event_.get()));
        }
    }
    return cap;
}

const py::tuple Image::getDlpackDevice() const
{
    return py::make_tuple(
        py::int_(static_cast<int>((*dlpack_tensor_)->device.device_type)), py::int_(static_cast<int>((*dlpack_tensor_)->device.device_id)));
}

py::object Image::cpu()
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);

    if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
        nvimgcodecImageInfo_t cpu_image_info(image_info);
        cpu_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        cpu_image_info.buffer = nullptr;

        auto image = Image(instance_, &cpu_image_info);
        CHECK_CUDA(cudaMemcpyAsync(
            cpu_image_info.buffer, image_info.buffer, image_info.buffer_size, cudaMemcpyDeviceToHost, image_info.cuda_stream));
        CHECK_CUDA(cudaStreamSynchronize(image_info.cuda_stream));

        return py::cast(image);
    } else if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        return py::cast(this);
    } else {
        return py::none();
    }
}

py::object Image::cuda(bool synchronize)
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);

    if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        nvimgcodecImageInfo_t cuda_image_info(image_info);
        cuda_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        cuda_image_info.buffer = nullptr;
        auto image = Image(instance_, &cuda_image_info);

        CHECK_CUDA(cudaMemcpyAsync(
            cuda_image_info.buffer, image_info.buffer, image_info.buffer_size, cudaMemcpyHostToDevice, cuda_image_info.cuda_stream));
        if (synchronize)
            CHECK_CUDA(cudaStreamSynchronize(cuda_image_info.cuda_stream));

        return py::cast(image);
    } else if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
        return py::cast(this);
    } else {
        return py::none();
    }
}

void Image::exportToPython(py::module& m)
{
    py::class_<Image>(m, "Image", "Class which wraps buffer with pixels. It can be decoded pixels or pixels to encode.")
        .def_property_readonly("__array_interface__", &Image::array_interface,
            R"pbdoc(
            TODO
            The array interchange interface compatible with Numba v0.39.0 or later (see 
            `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ for details)
            )pbdoc")
        .def_property_readonly("__cuda_array_interface__", &Image::cuda_interface,
            R"pbdoc(
            The CUDA array interchange interface compatible with Numba v0.39.0 or later (see 
            `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ for details)
            )pbdoc")
        .def_property_readonly("shape", &Image::shape)
        .def_property_readonly("width", &Image::getWidth)
        .def_property_readonly("height", &Image::getHeight)
        .def_property_readonly("ndim", &Image::getNdim)
        .def_property_readonly("dtype", &Image::dtype)
        .def_property_readonly("precision", &Image::precision, R"pbdoc(Maximum number of significant bits in data type. Value 0 
        means that precision is equal to data type bit depth)pbdoc")
        .def_property_readonly("buffer_kind", &Image::getBufferKind, R"pbdoc(Buffer kind in which image data is stored.)pbdoc")
        .def("__dlpack__", &Image::dlpack, "stream"_a = py::none(), "Export the image as a DLPack tensor")
        .def("__dlpack_device__", &Image::getDlpackDevice, "Get the device associated with the buffer")
        .def("to_dlpack", &Image::dlpack,
            R"pbdoc(
            Export the image with zero-copy conversion to a DLPack tensor. 
            
            Args:
                cuda_stream: An optional cudaStream_t represented as a Python integer, 
                             upon which synchronization must take place in created Image.

            Returns:
                DLPack tensor which is encapsulated in a PyCapsule object.
            )pbdoc",
            "cuda_stream"_a = py::none())
        .def("cpu", &Image::cpu,
            R"pbdoc(
            Returns a copy of this image in CPU memory. If this image is already in CPU memory, 
            than no copy is performed and the original object is returned. 
            
            Returns:
                Image object with content in CPU memory or None if copy could not be done.
            )pbdoc")
        .def("cuda", &Image::cuda,
            R"pbdoc(
            Returns a copy of this image in device memory. If this image is already in device memory, 
            than no copy is performed and the original object is returned.  
            
            Args:
                synchronize: If True (by default) it blocks and waits for copy from host to device to be finished, 
                             else not synchronization is executed and further synchronization needs to be done using
                             cuda stream provided by e.g. \_\_cuda_array_interface\_\_. 

            Returns:
                Image object with content in device memory or None if copy could not be done.
            )pbdoc",
            "synchronize"_a = true);
}

} // namespace nvimgcodec
