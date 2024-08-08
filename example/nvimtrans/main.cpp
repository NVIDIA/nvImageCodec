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


#include <cuda_runtime_api.h>
#include <nvimgcodec.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include "command_line_params.h"


#ifdef NVIMGCODEC_REGISTER_EXT_API
    #include <extensions/nvpnm/nvpnm_ext.h>
#endif

namespace fs = std::filesystem;

#if defined(_WIN32)
    #include <windows.h>
#endif

#define CHECK_CUDA(call)                                                                                          \
    {                                                                                                             \
        cudaError_t _e = (call);                                                                                  \
        if (_e != cudaSuccess) {                                                                                  \
            std::cerr << "CUDA Runtime failure: '#" << std::to_string(_e) << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                                  \
        }                                                                                                         \
    }

#define CHECK_NVIMGCODEC(call)                                   \
    {                                                           \
        nvimgcodecStatus_t _e = (call);                          \
        if (_e != NVIMGCODEC_STATUS_SUCCESS) {                   \
            std::stringstream _error;                           \
            _error << "nvImageCodec failure: '#" << std::to_string(_e) << "'"; \
            throw std::runtime_error(_error.str());             \
        }                                                       \
    }

double wtime(void)
{
#if defined(_WIN32)
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    } else {
        return (double)GetTickCount() / 1000.0;
    }
#else
    struct timespec tp;
    int rv = clock_gettime(CLOCK_MONOTONIC, &tp);

    if (rv)
        return 0;

    return tp.tv_nsec / 1.0E+9 + (double)tp.tv_sec;

#endif
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
    return static_cast<unsigned int>(sample_type)>> (8+3);
}

void fill_encode_params(const CommandLineParams& params, fs::path output_path, nvimgcodecEncodeParams_t* encode_params,
    nvimgcodecJpeg2kEncodeParams_t* jpeg2k_encode_params, nvimgcodecJpegEncodeParams_t* jpeg_encode_params,
    nvimgcodecJpegImageInfo_t* jpeg_image_info)
{
    encode_params->struct_type = NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS;
    encode_params->quality = params.quality;
    encode_params->target_psnr = params.target_psnr;

    //codec sepcific encode params
    if (params.output_codec == "jpeg2k") {
        jpeg2k_encode_params->struct_type = NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
        jpeg2k_encode_params->stream_type =
            output_path.extension().string() == ".jp2" ? NVIMGCODEC_JPEG2K_STREAM_JP2 : NVIMGCODEC_JPEG2K_STREAM_J2K;
        jpeg2k_encode_params->code_block_w = params.code_block_w;
        jpeg2k_encode_params->code_block_h = params.code_block_h;
        jpeg2k_encode_params->irreversible = !params.reversible;
        jpeg2k_encode_params->prog_order = params.jpeg2k_prog_order;
        jpeg2k_encode_params->num_resolutions = params.num_decomps;
        encode_params->struct_next = jpeg2k_encode_params;
    } else if (params.output_codec == "jpeg") {
        jpeg_encode_params->struct_type = NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS;
        jpeg_image_info->encoding = params.jpeg_encoding;
        jpeg_encode_params->optimized_huffman = params.optimized_huffman;
        encode_params->struct_next = jpeg_encode_params;
    }
}

void generate_backends_without_hw_gpu(std::vector<nvimgcodecBackend_t>& nvimgcds_backends) {
    nvimgcds_backends.resize(3);
    nvimgcds_backends[0].kind = NVIMGCODEC_BACKEND_KIND_CPU_ONLY;
    nvimgcds_backends[0].params = {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f};
    nvimgcds_backends[1].kind = NVIMGCODEC_BACKEND_KIND_GPU_ONLY;
    nvimgcds_backends[1].params = {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f};
    nvimgcds_backends[2].kind = NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU;
    nvimgcds_backends[2].params = {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f};
}

int process_one_image(nvimgcodecInstance_t instance, fs::path input_path, fs::path output_path, const CommandLineParams& params)
{
    std::cout << "Loading " << input_path.string() << " file" << std::endl;
    nvimgcodecCodeStream_t code_stream;
    std::ifstream file(input_path.string(), std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
    try {
        if (0) {
            CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateFromFile(instance, &code_stream, input_path.string().c_str()));
        } else {
            CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateFromHostMem(instance, &code_stream, buffer.data(), buffer.size()));
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: Unsupported  file format" << std::endl;
        return EXIT_FAILURE;
    }
    std::unique_ptr<std::remove_pointer<nvimgcodecCodeStream_t>::type, decltype(&nvimgcodecCodeStreamDestroy)> code_stream_raii(
            code_stream, &nvimgcodecCodeStreamDestroy);

    double parse_time = wtime();
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecJpegImageInfo_t jpeg_image_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
    image_info.struct_next = &jpeg_image_info;
    nvimgcodecCodeStreamGetImageInfo(code_stream, &image_info);
    parse_time = wtime() - parse_time;

    std::cout << "Input image info: " << std::endl;
    std::cout << "\t - width:" << image_info.plane_info[0].width << std::endl;
    std::cout << "\t - height:" << image_info.plane_info[0].height << std::endl;
    std::cout << "\t - components:" << image_info.num_planes << std::endl;
    std::cout << "\t - codec:" << image_info.codec_name << std::endl;
    if (jpeg_image_info.encoding != NVIMGCODEC_JPEG_ENCODING_UNKNOWN) {
        std::cout << "\t - jpeg encoding: 0x" << std::hex << jpeg_image_info.encoding << std::endl;
    }

    // Prepare decode parameters
    nvimgcodecDecodeParams_t decode_params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
    decode_params.apply_exif_orientation = !params.ignore_orientation;
    int bytes_per_element = sample_type_to_bytes_per_element(image_info.plane_info[0].sample_type);
    // Preparing output image_info
    if (jpeg_image_info.encoding == NVIMGCODEC_JPEG_ENCODING_LOSSLESS_HUFFMAN) {
        image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;        
        image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
    } else {
        image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
    }
        
    bool swap_wh = decode_params.apply_exif_orientation && ((image_info.orientation.rotated / 90) % 2);
    if (swap_wh) {
        std::swap(image_info.plane_info[0].height, image_info.plane_info[0].width);
    }

    size_t device_pitch_in_bytes = image_info.plane_info[0].width * bytes_per_element;

    for (uint32_t c = 0; c < image_info.num_planes; ++c) {
        image_info.plane_info[c].height = image_info.plane_info[0].height;
        image_info.plane_info[c].width = image_info.plane_info[0].width;
        image_info.plane_info[c].row_stride = device_pitch_in_bytes;
    }

    image_info.buffer_size = image_info.plane_info[0].row_stride * image_info.plane_info[0].height * image_info.num_planes;
    image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    CHECK_CUDA(cudaMalloc(&image_info.buffer, image_info.buffer_size));

    nvimgcodecImage_t image;
    nvimgcodecImageCreate(instance, &image, &image_info);
    std::unique_ptr<std::remove_pointer<nvimgcodecImage_t>::type, decltype(&nvimgcodecImageDestroy)> image_raii(
            image, &nvimgcodecImageDestroy);

    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;

    // vector must exist when we call nvimgcodecDecoderCreate() and nvimgcodecEncoderCreate()
    std::vector<nvimgcodecBackend_t> nvimgcds_backends;
    if (params.skip_hw_gpu_backend) {
        generate_backends_without_hw_gpu(nvimgcds_backends);

        exec_params.num_backends = nvimgcds_backends.size();
        exec_params.backends = nvimgcds_backends.data();
    }

    nvimgcodecDecoder_t decoder;
    std::string dec_options{":fancy_upsampling=0"};
    nvimgcodecDecoderCreate(instance, &decoder, &exec_params, dec_options.c_str());
    std::unique_ptr<std::remove_pointer<nvimgcodecDecoder_t>::type, decltype(&nvimgcodecDecoderDestroy)> decoder_raii(
            decoder, &nvimgcodecDecoderDestroy);

    // warm up
    for (int warmup_iter = 0; warmup_iter < params.warmup; warmup_iter++) {
        if (warmup_iter == 0) {
            std::cout << "Warmup started!" << std::endl;
        }
        nvimgcodecFuture_t future;
        nvimgcodecDecoderDecode(decoder, &code_stream, &image, 1, &decode_params, &future);
        nvimgcodecProcessingStatus_t decode_status;
        size_t status_size;
        nvimgcodecFutureGetProcessingStatus(future, &decode_status, &status_size);
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            std::cerr << "Error: Something went wrong during warmup decoding" << std::endl;
        }
        nvimgcodecFutureDestroy(future);

        if (warmup_iter == (params.warmup - 1)) {
            std::cout << "Warmup done!" << std::endl;
        }
    }

    nvimgcodecFuture_t decode_future;
    double decode_time = wtime();
    nvimgcodecDecoderDecode(decoder, &code_stream, &image, 1, &decode_params, &decode_future);
    std::unique_ptr<std::remove_pointer<nvimgcodecFuture_t>::type, decltype(&nvimgcodecFutureDestroy)> decode_future_raii(
            decode_future, &nvimgcodecFutureDestroy);

    size_t status_size;
    nvimgcodecProcessingStatus_t decode_status;
    nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
    CHECK_CUDA(cudaDeviceSynchronize()); // makes GPU wait until all decoding is finished
    decode_time = wtime() - decode_time; // record wall clock time on CPU (now includes GPU time as well)
    if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during decoding - processing status: " << decode_status << std::endl;
    }

    std::cout << "Saving to " << output_path.string() << " file" << std::endl;    

    nvimgcodecJpegImageInfo_t out_jpeg_image_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
    nvimgcodecEncodeParams_t encode_params{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
    nvimgcodecJpeg2kEncodeParams_t jpeg2k_encode_params{NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, sizeof(nvimgcodecJpeg2kEncodeParams_t), 0};
    nvimgcodecJpegEncodeParams_t jpeg_encode_params{NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, sizeof(nvimgcodecJpegEncodeParams_t), 0};
    fill_encode_params(params, output_path, &encode_params, &jpeg2k_encode_params, &jpeg_encode_params, &out_jpeg_image_info);

    nvimgcodecImageInfo_t out_image_info(image_info);
    out_image_info.chroma_subsampling = params.chroma_subsampling;
    out_image_info.struct_next = &out_jpeg_image_info;
    out_jpeg_image_info.struct_next = image_info.struct_next;
    strcpy(out_image_info.codec_name, params.output_codec.c_str());
    if (params.enc_color_trans) {
        out_image_info.color_spec = NVIMGCODEC_COLORSPEC_SYCC;
    }

    nvimgcodecCodeStream_t output_code_stream;
    nvimgcodecCodeStreamCreateToFile(
        instance, &output_code_stream, output_path.string().c_str(), &out_image_info);
    std::unique_ptr<std::remove_pointer<nvimgcodecCodeStream_t>::type, decltype(&nvimgcodecCodeStreamDestroy)> output_code_stream_raii(
            output_code_stream, &nvimgcodecCodeStreamDestroy);

    nvimgcodecEncoder_t encoder;
    nvimgcodecEncoderCreate(instance, &encoder, &exec_params, nullptr);
    std::unique_ptr<std::remove_pointer<nvimgcodecEncoder_t>::type, decltype(&nvimgcodecEncoderDestroy)> encoder_raii(
            encoder, &nvimgcodecEncoderDestroy);

    nvimgcodecFuture_t encode_future;
    double encode_time = wtime();
    nvimgcodecEncoderEncode(encoder, &image, &output_code_stream, 1, &encode_params, &encode_future);
    std::unique_ptr<std::remove_pointer<nvimgcodecFuture_t>::type, decltype(&nvimgcodecFutureDestroy)> encode_future_raii(
            encode_future, &nvimgcodecFutureDestroy);


    nvimgcodecProcessingStatus_t encode_status;
    nvimgcodecFutureGetProcessingStatus(encode_future, &encode_status, &status_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    encode_time = wtime() - encode_time;
    if (encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during encoding" << std::endl;
    }

    double total_time = parse_time + decode_time + encode_time;
    std::cout << "Total time spent on transcoding: " << total_time << std::endl;
    std::cout << " - time spent on parsing: " << parse_time << std::endl;
    std::cout << " - time spent on decoding: " << decode_time << std::endl;
    std::cout << " - time spent on encoding (including writting): " << encode_time << std::endl;

    CHECK_CUDA(cudaFree(image_info.buffer));

    return EXIT_SUCCESS;
}

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char>> FileData;

int collect_input_files(const std::string& sInputPath, std::vector<std::string>& filelist)
{

    if (fs::is_regular_file(sInputPath)) {
        filelist.push_back(sInputPath);
    } else if (fs::is_directory(sInputPath)) {
        fs::recursive_directory_iterator iter(sInputPath);
        for (auto& p : iter) {
            if (fs::is_regular_file(p)) {
                filelist.push_back(p.path().string());
            }
        }
    } else {
        std::cout << "unable to open input" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int read_next_batch(FileNames& image_names, int batch_size, FileNames::iterator& cur_iter, FileData& raw_data, std::vector<size_t>& raw_len,
    FileNames& current_names, bool verbose)
{
    int counter = 0;

    while (counter < batch_size) {
        if (cur_iter == image_names.end()) {
            if (verbose) {
                std::cerr << "Image list is too short to fill the batch, adding files "
                             "from the beginning of the image list"
                          << std::endl;
            }
            cur_iter = image_names.begin();
        }

        if (image_names.size() == 0) {
            std::cerr << "No valid images left in the input list, exit" << std::endl;
            return EXIT_FAILURE;
        }

        // Read an image from disk.
        std::ifstream input(cur_iter->c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open())) {
            std::cerr << "Cannot open image: " << *cur_iter << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }

        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (raw_data[counter].size() < static_cast<size_t>(file_size)) {
            raw_data[counter].resize(file_size);
        }
        if (!input.read(raw_data[counter].data(), file_size)) {
            std::cerr << "Cannot read from file: " << *cur_iter << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }
        raw_len[counter] = file_size;

        current_names[counter] = *cur_iter;

        counter++;
        cur_iter++;
    }
    return EXIT_SUCCESS;
}

constexpr int MAX_NUM_COMPONENTS = 4;
struct ImageBuffer
{
    unsigned char* data;
    size_t size;
    size_t pitch_in_bytes;
};

int prepare_decode_resources(nvimgcodecInstance_t instance, FileData& file_data, std::vector<size_t>& file_len,
    std::vector<ImageBuffer>& ibuf, FileNames& current_names, nvimgcodecDecoder_t& decoder, bool& is_host_output, bool& is_device_output,
    std::vector<nvimgcodecCodeStream_t>& code_streams, std::vector<nvimgcodecImage_t>& images, const nvimgcodecDecodeParams_t& decode_params,
    double& parse_time, bool skip_hw_gpu_backend)
{
    parse_time = 0;
    for (uint32_t i = 0; i < file_data.size(); i++) {
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateFromHostMem(instance, &code_streams[i], (unsigned char*)file_data[i].data(), file_len[i]));

        double time = wtime();
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamGetImageInfo(code_streams[i], &image_info));
        parse_time += wtime() - time;

        if (image_info.num_planes > MAX_NUM_COMPONENTS) {
            std::cout << "Num Components > " << MAX_NUM_COMPONENTS << "not supported by this sample" << std::endl;
            return EXIT_FAILURE;
        }

        for (uint32_t c = 0; c < image_info.num_planes; ++c) {
            if (image_info.plane_info[c].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                std::cout << "Data type #" << image_info.plane_info[c].sample_type << " not supported by this sample" << std::endl;
                return EXIT_FAILURE;
            }
        }

        int bytes_per_element = sample_type_to_bytes_per_element(image_info.plane_info[0].sample_type);

        //Decode to format
        image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;

        bool swap_wh = decode_params.apply_exif_orientation && ((image_info.orientation.rotated / 90) % 2);
        if (swap_wh) {
            std::swap(image_info.plane_info[0].height, image_info.plane_info[0].width);
        }

        size_t device_pitch_in_bytes = image_info.plane_info[0].width * bytes_per_element;

        for (uint32_t c = 0; c < image_info.num_planes; ++c) {
            image_info.plane_info[c].height = image_info.plane_info[0].height;
            image_info.plane_info[c].width = image_info.plane_info[0].width;
            image_info.plane_info[c].row_stride = device_pitch_in_bytes;
        }

        image_info.buffer_size = image_info.plane_info[0].row_stride * image_info.plane_info[0].height * image_info.num_planes;
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;

        if (image_info.buffer_size > ibuf[i].size) {
            if (ibuf[i].data) {
                CHECK_CUDA(cudaFree(ibuf[i].data));
            }
            CHECK_CUDA(cudaMalloc((void**)&ibuf[i].data, image_info.buffer_size));
        }
        
        ibuf[i].pitch_in_bytes = device_pitch_in_bytes;
        ibuf[i].size = image_info.buffer_size;
        image_info.buffer = ibuf[i].data;

        CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance, &images[i], &image_info));

        if (decoder == nullptr) {
            std::string dec_options{":fancy_upsampling=0"};
            nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
            exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;

            // vector must exist when we call nvimgcodecDecoderCreate()
            std::vector<nvimgcodecBackend_t> nvimgcds_backends;
            if (skip_hw_gpu_backend) {
                generate_backends_without_hw_gpu(nvimgcds_backends);

                exec_params.num_backends = nvimgcds_backends.size();
                exec_params.backends = nvimgcds_backends.data();
            }

            CHECK_NVIMGCODEC(nvimgcodecDecoderCreate(instance, &decoder, &exec_params, dec_options.c_str()));
        }
    }
    return EXIT_SUCCESS;
}

static std::map<std::string, std::string> codec2ext = {
    {"bmp", ".bmp"}, {"jpeg2k", ".j2k"}, {"tiff", ".tiff"}, {"jpeg", ".jpg"}, {"pnm", ".ppm"}};

int prepare_encode_resources(nvimgcodecInstance_t instance, FileNames& current_names, nvimgcodecEncoder_t& encoder, bool& is_host_input,
    bool& is_device_input, std::vector<nvimgcodecCodeStream_t>& out_code_streams, std::vector<nvimgcodecImage_t>& images,
    const nvimgcodecEncodeParams_t& encode_params, const CommandLineParams& params, nvimgcodecJpegImageInfo_t& jpeg_image_info,
    fs::path output_path, bool skip_hw_gpu_backend)
{
    for (uint32_t i = 0; i < current_names.size(); i++) {
        fs::path filename = fs::path(current_names[i]).filename();
        //TODO extension can depend on image format e.g. ppm, pgm
        std::string ext = "___";
        auto it = codec2ext.find(params.output_codec);
        if (it != codec2ext.end()) {
            ext = it->second;
        }

        filename = filename.replace_extension(ext);
        fs::path output_filename = output_path / filename;

        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        nvimgcodecImageGetImageInfo(images[i], &image_info);
        nvimgcodecImageInfo_t out_image_info(image_info);
        out_image_info.chroma_subsampling = params.chroma_subsampling;
        out_image_info.struct_next = reinterpret_cast<void*>(&jpeg_image_info);
        jpeg_image_info.struct_next = image_info.struct_next;
        strcpy(out_image_info.codec_name, params.output_codec.c_str());
        if (params.enc_color_trans) {
            out_image_info.color_spec = NVIMGCODEC_COLORSPEC_SYCC;
        }

        CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateToFile(
            instance, &out_code_streams[i], output_filename.string().c_str(), &out_image_info));

        if (encoder == nullptr) {
            const char* options = nullptr;
            nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
            exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;

            // vector must exist when we call nvimgcodecEncoderCreate()
            std::vector<nvimgcodecBackend_t> nvimgcds_backends;
            if (skip_hw_gpu_backend) {
                generate_backends_without_hw_gpu(nvimgcds_backends);

                exec_params.num_backends = nvimgcds_backends.size();
                exec_params.backends = nvimgcds_backends.data();
            }

            CHECK_NVIMGCODEC(nvimgcodecEncoderCreate(instance, &encoder, &exec_params, options));
        }
    }
    return EXIT_SUCCESS;
}

int process_images(nvimgcodecInstance_t instance, fs::path input_path, fs::path output_path, const CommandLineParams& params)
{
    int ret = EXIT_SUCCESS;
    double time = wtime();
    FileNames image_names;
    collect_input_files(input_path.string(), image_names);
    int total_images = params.total_images;
    if (total_images == -1) {
        total_images = image_names.size();
    }
    // vector for storing raw files and file lengths
    FileData file_data(params.batch_size);
    std::vector<size_t> file_len(params.batch_size);

    FileNames current_names(params.batch_size);
    // we wrap over image files to process total_images of files
    FileNames::iterator file_iter = image_names.begin();

    std::vector<ImageBuffer> image_buffers(params.batch_size);

    nvimgcodecDecoder_t decoder = nullptr;
    nvimgcodecEncoder_t encoder = nullptr;

    std::vector<nvimgcodecCodeStream_t> in_code_streams(params.batch_size);
    std::vector<nvimgcodecCodeStream_t> out_code_streams(params.batch_size);
    std::vector<nvimgcodecImage_t> images(params.batch_size);
    nvimgcodecDecodeParams_t decode_params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
    decode_params.apply_exif_orientation = !params.ignore_orientation;

    nvimgcodecJpegImageInfo_t jpeg_image_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
    nvimgcodecEncodeParams_t encode_params{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
    nvimgcodecJpeg2kEncodeParams_t jpeg2k_encode_params{NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, sizeof(nvimgcodecJpeg2kEncodeParams_t), 0};
    nvimgcodecJpegEncodeParams_t jpeg_encode_params{NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, sizeof(nvimgcodecJpegEncodeParams_t), 0};
    fill_encode_params(params, output_path, &encode_params, &jpeg2k_encode_params, &jpeg_encode_params, &jpeg_image_info);

    bool is_host_output = false;
    bool is_device_output = false;
    bool is_host_input = false;
    bool is_device_input = false;

    cudaEvent_t startEvent = NULL;
    cudaEvent_t stopEvent = NULL;

    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

    float loopTime = 0;

    int total_processed = 0;

    double total_processing_time = 0;
    double total_reading_time = 0;
    double total_parse_time = 0;
    double total_decode_time = 0;
    double total_encode_time = 0;
    double total_time = 0;

    int warmup = 0;
    while (total_processed < total_images && ret == EXIT_SUCCESS) {
        images.resize(params.batch_size);
        in_code_streams.resize(params.batch_size);
        out_code_streams.resize(params.batch_size);

        double start_reading_time = wtime();
        ret = read_next_batch(image_names, params.batch_size, file_iter, file_data, file_len, current_names, params.verbose);
        double reading_time = wtime() - start_reading_time;

        double parse_time = 0;
        ret = prepare_decode_resources(instance, file_data, file_len, image_buffers, current_names, decoder, is_host_output,
            is_device_output, in_code_streams, images, decode_params, parse_time, params.skip_hw_gpu_backend);

        std::cout << "." << std::flush;

        nvimgcodecFuture_t decode_future;
        double start_decoding_time = wtime();
        CHECK_NVIMGCODEC(
            nvimgcodecDecoderDecode(decoder, in_code_streams.data(), images.data(), params.batch_size, &decode_params, &decode_future));
        std::unique_ptr<std::remove_pointer<nvimgcodecDecoder_t>::type, decltype(&nvimgcodecDecoderDestroy)> decoder_raii(
            decoder, &nvimgcodecDecoderDestroy);
        std::unique_ptr<std::remove_pointer<nvimgcodecFuture_t>::type, decltype(&nvimgcodecFutureDestroy)> decode_future_raii(
            decode_future, &nvimgcodecFutureDestroy);

        size_t status_size;
        nvimgcodecFutureGetProcessingStatus(decode_future, nullptr, &status_size);
        CHECK_CUDA(cudaDeviceSynchronize());
        double decode_time = wtime() - start_decoding_time;

        std::vector<nvimgcodecProcessingStatus_t> decode_status(status_size);
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status[0], &status_size);
        for (int i = 0; i < decode_status.size(); ++i) {
            if (decode_status[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                std::cerr << "Error: Something went wrong during decoding image #" << i << " it will not be encoded" << std::endl;
                if (images[i])
                    nvimgcodecImageDestroy(images[i]);
                images.erase(images.begin() + i);
                if (out_code_streams[i])
                    nvimgcodecCodeStreamDestroy(out_code_streams[i]);
                out_code_streams.erase(out_code_streams.begin() + i);
                --i;
            }
        }

        ret = prepare_encode_resources(instance, current_names, encoder, is_host_input, is_device_input, out_code_streams, images,
            encode_params, params, jpeg_image_info, output_path, params.skip_hw_gpu_backend);
        std::unique_ptr<std::remove_pointer<nvimgcodecEncoder_t>::type, decltype(&nvimgcodecEncoderDestroy)> encoder_raii(
            encoder, &nvimgcodecEncoderDestroy);

        nvimgcodecFuture_t encode_future;
        double start_encoding_time = wtime();
        CHECK_NVIMGCODEC(nvimgcodecEncoderEncode(
            encoder, images.data(), out_code_streams.data(), out_code_streams.size(), &encode_params, &encode_future));
        std::unique_ptr<std::remove_pointer<nvimgcodecFuture_t>::type, decltype(&nvimgcodecFutureDestroy)> encode_future_raii(
            encode_future, &nvimgcodecFutureDestroy);

        nvimgcodecFutureGetProcessingStatus(encode_future, nullptr, &status_size);
        CHECK_CUDA(cudaDeviceSynchronize());
        double encode_time = wtime() - start_encoding_time;
        std::vector<nvimgcodecProcessingStatus_t> encode_status(status_size);
        nvimgcodecFutureGetProcessingStatus(encode_future, &encode_status[0], &status_size);
        for (int i = 0; i < encode_status.size(); ++i) {
            if (encode_status[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                std::cerr << "Error: Something went wrong during encoding image #" << i << " it will not be saved" << std::endl;
            }
        }

        if (warmup < params.warmup) {
            warmup++;
        } else {
            total_processed += params.batch_size;
            total_reading_time += reading_time;
            total_parse_time += parse_time;
            total_decode_time += decode_time;
            total_encode_time += encode_time;
            total_processing_time += parse_time + decode_time + encode_time;
        }

        for (auto& cs : in_code_streams) {
            nvimgcodecCodeStreamDestroy(cs);
        }
        for (auto& cs : out_code_streams) {
            nvimgcodecCodeStreamDestroy(cs);
        }
        for (auto& img : images) {
            nvimgcodecImageDestroy(img);
        }
    }

    for (int i = 0; i < params.batch_size; ++i) {
        if (image_buffers[i].data) {
            CHECK_CUDA(cudaFree(image_buffers[i].data));
        }
    }

    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    total_time += wtime() - time;
    std::cout << std::endl;
    std::cout << "Total images: " << total_images << std::endl;
    std::cout << "Total transcoding time: " << total_time << std::endl;
    std::cout << "Avg transcoding time per image: " << total_time / total_images << std::endl;
    std::cout << "Avg transcode speed  (in images per sec): " << total_images / total_time << std::endl;
    std::cout << "Avg transcode time per batch: " << total_time / ((total_images + params.batch_size - 1) / params.batch_size) << std::endl;

    std::cout << "Total reading time: " << total_reading_time << std::endl;
    std::cout << "Avg reading time per image: " << total_reading_time / total_images << std::endl;
    std::cout << "Avg reading speed  (in images per sec): " << total_images / total_reading_time << std::endl;
    std::cout << "Avg reading time per batch: " << total_reading_time / ((total_images + params.batch_size - 1) / params.batch_size)
              << std::endl;

    std::cout << "Total parsing time: " << total_parse_time << std::endl;
    std::cout << "Avg parsing time per image: " << total_parse_time / total_images << std::endl;
    std::cout << "Avg parsing speed  (in images per sec): " << total_images / total_parse_time << std::endl;
    std::cout << "Avg parsing time per batch: " << total_parse_time / ((total_images + params.batch_size - 1) / params.batch_size)
              << std::endl;

    std::cout << "Total decoding time: " << total_decode_time << std::endl;
    std::cout << "Avg decoding time per image: " << total_decode_time / total_images << std::endl;
    std::cout << "Avg decoding speed  (in images per sec): " << total_images / total_decode_time << std::endl;
    std::cout << "Avg decoding time per batch: " << total_decode_time / ((total_images + params.batch_size - 1) / params.batch_size)
              << std::endl;

    std::cout << "Total encoding time: " << total_encode_time << std::endl;
    std::cout << "Avg encoding time per image: " << total_encode_time / total_images << std::endl;
    std::cout << "Avg encoding speed  (in images per sec): " << total_images / total_encode_time << std::endl;
    std::cout << "Avg encoding time per batch: " << total_encode_time / ((total_images + params.batch_size - 1) / params.batch_size)
              << std::endl;

    return ret;
}

void list_cuda_devices(int num_devices)
{
    for (int i = 0; i < num_devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
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
    std::cout << " - extension API version: " << NVIMGCODEC_STREAM_VER(properties.ext_api_version) << std::endl;
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

#ifdef NVIMGCODEC_REGISTER_EXT_API
    nvimgcodecExtension_t pnm_extension{nullptr};
    nvimgcodecExtensionDesc_t pnm_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
    get_nvpnm_extension_desc(&pnm_extension_desc);
    nvimgcodecExtensionCreate(instance, &pnm_extension, &pnm_extension_desc);
#endif

    fs::path exe_path(argv[0]);
    fs::path input_path = fs::absolute(exe_path).parent_path() / fs::path(params.input);
    fs::path output_path = fs::absolute(exe_path).parent_path() / fs::path(params.output);

    if (fs::is_directory(input_path)) {
        exit_code = process_images(instance, input_path, output_path, params);
    } else {
        exit_code = process_one_image(instance, input_path, output_path, params);
    }

#ifdef NVIMGCODEC_REGISTER_EXT_API
    if (pnm_extension) {
        nvimgcodecExtensionDestroy(pnm_extension);
    }
#endif
    
    nvimgcodecInstanceDestroy(instance);

    return exit_code;
}
