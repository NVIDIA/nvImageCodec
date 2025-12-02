/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <math.h>
#include <sys/types.h>
#include <filesystem>
#include <fstream>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <dirent.h>

#include <cuda_runtime_api.h>
#include <cvcuda/OpCustomCrop.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpResize.hpp>

#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <nvcv_adapter.hpp>

#include "command_line_params.h"

namespace fs = std::filesystem;

/**
 * @brief Crop and Resize sample app.
 *
 * The Crop and Resize is a simple pipeline which demonstrates usage of
 * CVCuda Tensor along with a few operators.
 *
 * Input Batch Tensor -> Crop -> Resize -> WriteImage
 * 
 * Compatibility: CV-CUDA v0.4.0 Beta
 * 
 */

inline void CheckCudaError(cudaError_t code, const char* file, const int line)
{
    if (code != cudaSuccess) {
        const char* errorMessage = cudaGetErrorString(code);
        const std::string message = "CUDA error returned at " + std::string(file) + ":" + std::to_string(line) +
                                    ", Error code: " + std::to_string(code) + " (" + std::string(errorMessage) + ")";
        throw std::runtime_error(message);
    }
}

#define CHECK_CUDA_ERROR(val)                      \
    {                                              \
        CheckCudaError((val), __FILE__, __LINE__); \
    }

double wtime(void)
{
    struct timespec tp;
    int rv = clock_gettime(CLOCK_MONOTONIC, &tp);

    if (rv)
        return 0;

    return tp.tv_nsec / 1.0E+9 + (double)tp.tv_sec;
}

uint32_t verbosity2severity(int verbose)
{
    uint32_t result = 0;
    if (verbose >= 1)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL | NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR;
    if (verbose >= 2)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING;
    if (verbose >= 3)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO;
    if (verbose >= 4)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG;
    if (verbose >= 5)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE;

    return result;
}

inline size_t sample_type_to_bytes_per_element(nvimgcodecSampleDataType_t sample_type)
{
    return static_cast<unsigned int>(sample_type) >> (8 + 3);
}

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char>> FileData;

int collect_input_files(const std::string& sInputPath, std::vector<std::string>& filelist)
{
    int error_code = 1;
    struct stat s;

    if (stat(sInputPath.c_str(), &s) == 0) {
        if (s.st_mode & S_IFREG) {
            filelist.push_back(sInputPath);
        } else if (s.st_mode & S_IFDIR) {
            // processing each file in directory
            DIR* dir_handle;
            struct dirent* dir;
            dir_handle = opendir(sInputPath.c_str());
            std::vector<std::string> filenames;
            if (dir_handle) {
                error_code = 0;
                while ((dir = readdir(dir_handle)) != NULL) {
                    if (dir->d_type == DT_REG) {
                        std::string sFileName = sInputPath + dir->d_name;
                        filelist.push_back(sFileName);
                    } else if (dir->d_type == DT_DIR) {
                        std::string sname = dir->d_name;
                        if (sname != "." && sname != "..") {
                            collect_input_files(sInputPath + sname + "/", filelist);
                        }
                    }
                }
                closedir(dir_handle);
            } else {
                std::cout << "Cannot open input directory: " << sInputPath << std::endl;
                return error_code;
            }
        } else {
            std::cout << "Cannot open input: " << sInputPath << std::endl;
            return error_code;
        }
    } else {
        std::cout << "Cannot find input path " << sInputPath << std::endl;
        return error_code;
    }

    return 0;
}

int decode_one_image(nvimgcodecInstance_t instance, const CommandLineParams& params, const FileNames& image_names,
    nvimgcodecSampleFormat_t out_format, NVCVImageData* image_data, NVCVTensorData* tensor_data, cudaStream_t& stream)
{
    int result = EXIT_SUCCESS;
    nvimgcodecCodeStream_t code_stream = nullptr;
    nvimgcodecCodeStreamCreateFromFile(instance, &code_stream, image_names[0].c_str());
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecCodeStreamGetImageInfo(code_stream, &image_info);

    // Prepare decode parameters
    nvimgcodecDecodeParams_t decode_params{};
    decode_params.apply_exif_orientation= 1;
    int bytes_per_element = sample_type_to_bytes_per_element(image_info.plane_info[0].sample_type);

    // Preparing output image_info
    image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
    image_info.sample_format = out_format;
    size_t buffer_size = 0;
    if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB) {
        image_info.num_planes = 1;
        image_info.plane_info[0].num_channels = 3;
        image_info.plane_info[0].row_stride = image_info.plane_info[0].width * bytes_per_element * image_info.plane_info[0].num_channels;
        image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        buffer_size = image_info.plane_info[0].row_stride * image_info.plane_info[0].height * image_info.num_planes;
    } else if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) {
        size_t row_stride = image_info.plane_info[0].width * bytes_per_element;
        image_info.num_planes = 3;
        buffer_size = row_stride * image_info.plane_info[0].height * image_info.num_planes;
        for (auto p = 0; p < image_info.num_planes; ++p) {
            image_info.plane_info[p].height = image_info.plane_info[0].height;
            image_info.plane_info[p].width = image_info.plane_info[0].width;
            image_info.plane_info[p].row_stride = row_stride;
            image_info.plane_info[p].num_channels = 1;
            image_info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        }
    }
    image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;

    CHECK_CUDA_ERROR(cudaMallocAsync(&image_info.buffer, buffer_size, stream));

    nvimgcodecImage_t image = nullptr;
    nvimgcodecImageCreate(instance, &image, &image_info);

    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
    nvimgcodecDecoder_t decoder = nullptr;
    nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr);

    nvimgcodecFuture_t future;
    nvimgcodecDecoderDecode(decoder, &code_stream, &image, 1, &decode_params, &future);

    nvimgcodecProcessingStatus_t decode_status;
    size_t size;
    nvimgcodecFutureGetProcessingStatus(future, &decode_status, &size);
    if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during decoding" << std::endl;
        result = EXIT_FAILURE;
    }

    nvimgcodecFutureDestroy(future);

    nvimgcodecImageGetImageInfo(image, &image_info);
    nvimgcodec::adapter::nvcv::ImageInfo2ImageData(image_data, image_info);
    nvimgcodec::adapter::nvcv::ImageInfo2TensorData(tensor_data, image_info);

    nvimgcodecImageDestroy(image);
    nvimgcodecDecoderDestroy(decoder);
    nvimgcodecCodeStreamDestroy(code_stream);

    return result;
}

void fill_encode_params(const CommandLineParams& params, fs::path output_path, nvimgcodecEncodeParams_t* encode_params,
    nvimgcodecJpeg2kEncodeParams_t* jpeg2k_encode_params, nvimgcodecJpegEncodeParams_t* jpeg_encode_params,
    nvimgcodecJpegImageInfo_t* jpeg_image_info)
{
    encode_params->struct_type = NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS;
    encode_params->quality_type = params.quality_type;
    encode_params->quality_value = params.quality_value;

    //codec sepcific encode params
    if (params.output_codec == "jpeg2k") {
        jpeg2k_encode_params->struct_type = NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
        jpeg2k_encode_params->stream_type =
            output_path.extension().string() == ".jp2" ? NVIMGCODEC_JPEG2K_STREAM_JP2 : NVIMGCODEC_JPEG2K_STREAM_J2K;
        jpeg2k_encode_params->code_block_w = params.code_block_w;
        jpeg2k_encode_params->code_block_h = params.code_block_h;
        jpeg2k_encode_params->prog_order = params.jpeg2k_prog_order;
        jpeg2k_encode_params->num_resolutions = params.num_decomps + 1;
        jpeg2k_encode_params->mct_mode = params.enc_color_trans ? 1 : 0;
        encode_params->struct_next = jpeg2k_encode_params;
    } else if (params.output_codec == "jpeg") {
        jpeg_encode_params->struct_type = NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS;
        jpeg_image_info->encoding = params.jpeg_encoding;
        jpeg_encode_params->optimized_huffman = params.optimized_huffman;
        encode_params->struct_next = jpeg_encode_params;
    }
}

int encode_one_image(nvimgcodecInstance_t instance, const CommandLineParams& params, const NVCVTensorData& tensor_data, fs::path output_path,
    cudaStream_t& stream)
{
    int result = EXIT_SUCCESS;

    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodec::adapter::nvcv::TensorData2ImageInfo(&image_info, tensor_data);
    if (0) {
        std::cout << "Input image info: " << std::endl;
        std::cout << "\t - width:" << image_info.plane_info[0].width << std::endl;
        std::cout << "\t - height:" << image_info.plane_info[0].height << std::endl;
        std::cout << "\t - num_planes:" << image_info.num_planes << std::endl;
        std::cout << "\t - num_channels:" << image_info.plane_info[0].num_channels << std::endl;
        std::cout << "\t - sample_format:" << image_info.sample_format << std::endl;
        std::cout << "\t - row_stride:" << image_info.plane_info[0].row_stride << std::endl;
        std::cout << "\t - codec:" << params.output_codec.data() << std::endl;
    }

    nvimgcodecImage_t image = nullptr;
    nvimgcodecImageCreate(instance, &image, &image_info);

    nvimgcodecJpegImageInfo_t out_jpeg_image_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
    nvimgcodecEncodeParams_t encode_params{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
    nvimgcodecJpeg2kEncodeParams_t jpeg2k_encode_params{NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, sizeof(nvimgcodecJpeg2kEncodeParams_t), 0};
    nvimgcodecJpegEncodeParams_t jpeg_encode_params{NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, sizeof(nvimgcodecJpegEncodeParams_t), 0};
    fill_encode_params(params, output_path, &encode_params, &jpeg2k_encode_params, &jpeg_encode_params, &out_jpeg_image_info);

    nvimgcodecImageInfo_t out_image_info(image_info);
    strcpy(out_image_info.codec_name, params.output_codec.c_str());
    out_image_info.struct_next = &out_jpeg_image_info;
    if (params.enc_color_trans && params.output_codec != "jpeg2k") {// For jpeg2k, we use MCT mode to convert color spec
        out_image_info.color_spec = NVIMGCODEC_COLORSPEC_SYCC;
    }
    nvimgcodecCodeStream_t code_stream = nullptr;
    nvimgcodecCodeStreamCreateToFile(instance, &code_stream, output_path.string().c_str(), &out_image_info);


    nvimgcodecEncoder_t encoder = nullptr;
    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
    nvimgcodecEncoderCreate(instance, &encoder, &exec_params, nullptr);

    nvimgcodecFuture_t future;
    nvimgcodecEncoderEncode(encoder, &image, &code_stream, 1, &encode_params, &future);

    nvimgcodecProcessingStatus_t encode_status;
    size_t status_size;
    nvimgcodecFutureGetProcessingStatus(future, &encode_status, &status_size);
    if (encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during encoding" << std::endl;
        result = EXIT_FAILURE;
    }
    nvimgcodecFutureDestroy(future);
    nvimgcodecEncoderDestroy(encoder);
    nvimgcodecImageDestroy(image);
    nvimgcodecCodeStreamDestroy(code_stream);

    return result;
}

int process_one_image(nvimgcodecInstance_t instance, fs::path input_path, fs::path output_path, const CommandLineParams& params)
{
    // tag: Scan input directory and collect source images
    FileNames image_names;
    collect_input_files(input_path.string(), image_names);

    // tag: Create the cuda stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // tag: Image Loading
    // nvImageCodec is used to load the image

    NVCVImageData image_data;
    NVCVTensorData tensor_data;
    decode_one_image(instance, params, image_names, NVIMGCODEC_SAMPLEFORMAT_P_RGB, &image_data, &tensor_data, stream);
    if (0) { // Example for image creation
        nvcv::ImageDataStridedCuda imageDataStridedCuda(image_data);
        NVCVImageHandle img_handle = nullptr;
        nvcvImageWrapDataConstruct(&imageDataStridedCuda.cdata(), nullptr, nullptr, &img_handle);
        nvcv::Image inImage(std::move(img_handle));
        NVCVTensorHandle handle;
        nvcvTensorWrapImageConstruct(inImage.handle(), &handle);
        nvcv::Tensor decodedTensor(std::move(handle));
    }

    nvcv::TensorDataStridedCuda tensorDataStridedCuda(tensor_data);
    NVCVTensorHandle handle;
    nvcvTensorWrapDataConstruct(&tensorDataStridedCuda.cdata(), nullptr, nullptr, &handle);
    nvcv::Tensor decodedTensor(std::move(handle));

    // tag: Create operator input tensor in RGB interleaved format
    nvcv::Tensor inTensor(1, {static_cast<int>(tensor_data.shape[3]), static_cast<int>(tensor_data.shape[2])}, nvcv::FMT_RGB8);

    // tag: Convert form planar to interleaved
    cvcuda::Reformat reformatOp;
    reformatOp(stream, decodedTensor, inTensor);

    // tag: The input buffer is now ready to be used by the operators

    // Set parameters for Crop and Resize
    // ROI dimensions to crop in the input image
    int cropX = 50;
    int cropY = 100;
    int cropWidth = 320;
    int cropHeight = 240;

    // Set the resize dimensions
    int resizeWidth = 800;
    int resizeHeight = 600;

    //  Initialize the CVCUDA ROI struct
    NVCVRectI crpRect = {cropX, cropY, cropWidth, cropHeight};

    // tag: Allocate Tensors for Crop and Resize

    // Create a CVCUDA Tensor based on the crop window size.
    nvcv::Tensor cropTensor(1, {cropWidth, cropHeight}, nvcv::FMT_RGB8);
    // Create a CVCUDA Tensor based on resize dimensions
    nvcv::Tensor resizedTensor(1, {resizeWidth, resizeHeight}, nvcv::FMT_RGB8);

#ifdef PROFILE_SAMPLE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    // tag: Initialize operators for Crop and Resize
    cvcuda::CustomCrop cropOp;
    cvcuda::Resize resizeOp;

    // tag: Executes the CustomCrop operation on the given cuda stream
    cropOp(stream, inTensor, cropTensor, crpRect);
    //cropOp(stream, decodedTensor, cropTensor, crpRect);

    // Resize operator can now be enqueued into the same stream
    resizeOp(stream, cropTensor, resizedTensor, NVCV_INTERP_LINEAR);

    // tag: Profile section
#ifdef PROFILE_SAMPLE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float operatorms = 0;
    cudaEventElapsedTime(&operatorms, start, stop);
    std::cout << "Time for Crop and Resize : " << operatorms << " ms" << std::endl;
#endif

    // tag: Create output tensor in planar RGB as currently all codecs (except nvtiff) supports this format
    bool is_tiff = strcmp(params.output_codec.c_str(), "tiff") == 0;
    nvcv::Tensor outTensor(1, {resizeWidth, resizeHeight}, is_tiff ? nvcv::FMT_RGB8 : nvcv::FMT_RGB8p);

    // tag: Reformat interleaved to planar
    reformatOp(stream, resizedTensor, outTensor);

    // tag: Copy the buffer to CPU and write resized image into output file
    NVCVTensorData out_tensor_data = outTensor.exportData().cdata();
    encode_one_image(instance, params, out_tensor_data, output_path, stream);

    // tag: Clean up
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}

void list_cuda_devices(int num_devices)
{
    for (int i = 0; i < num_devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    }
}

int main(int argc, const char* argv[])
{
    int exit_code = EXIT_SUCCESS;
    CommandLineParams params;
    int status = process_commandline_params(argc, argv, &params);
    if (status != -1) {
        return status;
    }
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (params.list_cuda_devices) {
        list_cuda_devices(num_devices);
    }

    if (params.device_id < num_devices) {
        cudaSetDevice(params.device_id);
    } else {
        std::cerr << "Error: Wrong device id #" << params.device_id << std::endl;
        list_cuda_devices(num_devices);
        return EXIT_FAILURE;
    }

    nvimgcodecProperties_t properties{NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES, sizeof(nvimgcodecProperties_t), 0};
    nvimgcodecGetProperties(&properties);
    std::cout << "nvImageCodec version: " << NVIMGCODEC_STREAM_VER(properties.version) << std::endl;
    std::cout << " - CUDA Runtime version: " << properties.cudart_version / 1000 << "." << (properties.cudart_version % 1000) / 10
              << std::endl;
    cudaDeviceProp props;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);
    std::cout << "Using GPU: " << props.name << " with Compute Capability " << props.major << "." << props.minor << std::endl;

    nvimgcodecInstance_t instance;
    nvimgcodecInstanceCreateInfo_t instance_create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    instance_create_info.load_builtin_modules = 1;
    instance_create_info.load_extension_modules = 1;
    instance_create_info.create_debug_messenger = 1;
    instance_create_info.message_severity = verbosity2severity(params.verbose);
    instance_create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;

    nvimgcodecInstanceCreate(&instance, &instance_create_info);

    fs::path exe_path(argv[0]);
    fs::path input_path = fs::absolute(exe_path).parent_path() / fs::path(params.input);
    fs::path output_path = fs::absolute(exe_path).parent_path() / fs::path(params.output);

    if (fs::is_directory(input_path)) {
        // exit_code = process_images(instance, input_path, output_path, params);
    } else {
        exit_code = process_one_image(instance, input_path, output_path, params);
    }
    nvimgcodecInstanceDestroy(instance);

    return exit_code;
}
