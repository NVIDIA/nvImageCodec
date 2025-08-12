/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <nvimgcodec.h>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

struct CommandLineParams
{
    std::string input;
    std::string output;
    std::string output_codec;
    int warmup;
    int batch_size;
    int total_images;
    int device_id;
    int verbose;
    nvimgcodecQualityType_t quality_type;
    float quality_value;
    bool write_output;
    int num_decomps;
    int code_block_w;
    int code_block_h;
    bool enc_color_trans;
    bool optimized_huffman;
    bool ignore_orientation;
    nvimgcodecJpegEncoding_t jpeg_encoding;
    nvimgcodecJpeg2kProgOrder_t jpeg2k_prog_order;
    nvimgcodecChromaSubsampling_t chroma_subsampling;
    bool list_cuda_devices;
    bool skip_hw_gpu_backend;
};

int find_param_index(const char** argv, int argc, const char* parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++) {
        if (strncmp(argv[i], parm, 100) == 0) {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1) {
        return index;
    } else {
        std::cout << "Error, parameter " << parm << " has been specified more than once, exiting\n" << std::endl;
        return -1;
    }

    return -1;
}

int process_commandline_params(int argc, const char* argv[], CommandLineParams* params)
{
    static std::map<std::string, std::string> ext2codec = {{".bmp", "bmp"}, {".j2c", "jpeg2k"}, {".j2k", "jpeg2k"}, {".jp2", "jpeg2k"},
        {".tiff", "tiff"}, {".tif", "tiff"}, {".jpg", "jpeg"}, {".jpeg", "jpeg"}, {".ppm", "pnm"}, {".pgm", "pnm"}, {".pbm", "pnm"},
        {".webp", "webp"}};

    int pidx;
    if ((pidx = find_param_index(argv, argc, "-h")) != -1 || (pidx = find_param_index(argv, argc, "--help")) != -1) {
        std::cout << "Usage: " << argv[0] << " [decoding options]"
                  << " -i <input> "
                  << "[encoding options]"
                  << " -o <output> " << std::endl;
        std::cout << std::endl;
        std::cout << "General options: " << std::endl;
        std::cout << "  -h --help\t\t: show help" << std::endl;
        std::cout << "  -v --verbose\t\t: verbosity level from 0 to 5 (default 1)" << std::endl;
        std::cout << "  -w\t\t\t: warmup iterations (default 0)" << std::endl;
        std::cout << "  -l\t\t\t: List cuda devices" << std::endl;
        std::cout << "  -b --batch_size\t: Batch size (default 1)" << std::endl;
        std::cout << "  -d --device\t: Cuda device (default 0)" << std::endl;
        std::cout << std::endl;
        std::cout << "  --ignore_orientation\t: Ignore EXFIF orientation (default false)" << std::endl;
        std::cout << "  -i  --input\t\t: Path to single image or directory" << std::endl;
        std::cout << std::endl;
        std::cout << "Encoding options: " << std::endl;
        std::cout << "  -c --output_codec\t: Output codec (default bmp)" << std::endl;
        std::cout << "  --quality_type\t\t: Quality type: DEFAULT, LOSSLESS, QUALITY, QUANTIZATION_STEP, PSNR, SIZE_RATIO (default DEFAULT)" << std::endl;
        std::cout << "  --quality_value\t\t: Quality value, ignored for DEFAULT type option (default 0)" << std::endl;
        std::cout << "  --chroma_subsampling\t: Chroma subsampling (default 444)" << std::endl;
        std::cout << "  --enc_color_trans\t: Encoding color transfrom. For true transform RGB "
                     "color images to YUV (default false)"
                  << std::endl;
        std::cout << "  --num_decomps\t\t: number of wavelet transform decompositions levels (default 5)" << std::endl;
        std::cout << "  --optimized_huffman\t: For false non-optimized Huffman will be used. Otherwise "
                     "optimized version will be used. (default false)."
                  << std::endl;
        std::cout << "  --jpeg_encoding\t: Corresponds to the JPEG marker"
                     " baseline_dct, progressive_dct (default "
                     "baseline_dct)."
                  << std::endl;
        std::cout << "  --jpeg2k_prog_order\t: Jpeg2000 progression order: LRCP, RLCP, RPCL, PCRL, CPRL (default RPCL)" << std::endl;
        std::cout << "  -o  --output\t\t: File or directory to write decoded image using <output_codec>" << std::endl;
        std::cout << "  --skip_hw_gpu_backend\t: Skip hardware gpu backend (default false)" << std::endl;

        return EXIT_SUCCESS;
    }
    params->warmup = 0;
    if ((pidx = find_param_index(argv, argc, "-w")) != -1) {
        params->warmup = static_cast<int>(strtod(argv[pidx + 1], NULL));
    }

    params->verbose = 1;
    if (((pidx = find_param_index(argv, argc, "--verbose")) != -1) || ((pidx = find_param_index(argv, argc, "-v")) != -1)) {
        params->verbose = static_cast<int>(strtod(argv[pidx + 1], NULL));
    }

    params->input = "./";
    if ((pidx = find_param_index(argv, argc, "-i")) != -1 || (pidx = find_param_index(argv, argc, "--input")) != -1) {
        params->input = argv[pidx + 1];
    } else {
        std::cout << "Please specify input directory with encoded images" << std::endl;
        return EXIT_FAILURE;
    }

    params->ignore_orientation = false;
    if ((pidx = find_param_index(argv, argc, "--ignore_orientation")) != -1) {
        params->ignore_orientation = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->quality_type = NVIMGCODEC_QUALITY_TYPE_DEFAULT;
    if ((pidx = find_param_index(argv, argc, "--quality_type")) != -1) {
        std::string quailty_type = argv[pidx + 1];
        if (quailty_type == "DEFAULT") {
            params->quality_type = NVIMGCODEC_QUALITY_TYPE_DEFAULT;
        } else if (quailty_type == "LOSSLESS") {
            params->quality_type = NVIMGCODEC_QUALITY_TYPE_LOSSLESS;
        } else if (quailty_type == "QUALITY") {
            params->quality_type = NVIMGCODEC_QUALITY_TYPE_QUALITY;
        } else if (quailty_type == "QUANTIZATION_STEP") {
            params->quality_type = NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP;
        } else if (quailty_type == "PSNR") {
            params->quality_type = NVIMGCODEC_QUALITY_TYPE_PSNR;
        } else if (quailty_type == "SIZE_RATIO") {
            params->quality_type = NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO;
        } else {
            std::cout << "Unrecognized quality type: " << quailty_type << std::endl;
            return EXIT_FAILURE;
        }
    }

    params->quality_value = 0;
    if ((pidx = find_param_index(argv, argc, "--quality_value")) != -1) {
        params->quality_value = static_cast<float>(strtod(argv[pidx + 1], NULL));
    }

    params->write_output = false;
    if ((pidx = find_param_index(argv, argc, "-o")) != -1 || (pidx = find_param_index(argv, argc, "--output")) != -1) {
        params->output = argv[pidx + 1];
    }

    params->output_codec = "bmp";
    fs::path file_path(params->output);
    if (file_path.has_extension()) {
        std::string extension = file_path.extension().string();
        auto it = ext2codec.find(extension);
        if (it != ext2codec.end()) {
            params->output_codec = it->second;
        }
    }
    if ((pidx = find_param_index(argv, argc, "-c")) != -1 || (pidx = find_param_index(argv, argc, "--output_codec")) != -1) {
        params->output_codec = argv[pidx + 1];
    }

    params->num_decomps = 5;
    if ((pidx = find_param_index(argv, argc, "--num_decomps")) != -1) {
        params->num_decomps = atoi(argv[pidx + 1]);
    }

    params->code_block_w = 64;
    params->code_block_h = 64;
    if ((pidx = find_param_index(argv, argc, "--block_size")) != -1) {
        params->code_block_h = atoi(argv[pidx + 1]);
        params->code_block_w = atoi(argv[pidx + 2]);
    }

    params->enc_color_trans = false;
    if ((pidx = find_param_index(argv, argc, "--enc_color_trans")) != -1) {
        params->enc_color_trans = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->optimized_huffman = false;
    if ((pidx = find_param_index(argv, argc, "--optimized_huffman")) != -1) {
        params->optimized_huffman = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->jpeg_encoding = NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT;
    if ((pidx = find_param_index(argv, argc, "--jpeg_encoding")) != -1) {
        if (strcmp(argv[pidx + 1], "baseline_dct") == 0) {
            params->jpeg_encoding = NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT;
        } else if (strcmp(argv[pidx + 1], "progressive_dct") == 0) {
            params->jpeg_encoding = NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;
        } else {
            std::cout << "Unknown jpeg encoding type: " << argv[pidx + 1] << std::endl;
        }
    }
    params->chroma_subsampling = NVIMGCODEC_SAMPLING_444;
    if ((pidx = find_param_index(argv, argc, "--chroma_subsampling")) != -1) {
        std::map<std::string, nvimgcodecChromaSubsampling_t> str2Css = {{"444", NVIMGCODEC_SAMPLING_444}, {"420", NVIMGCODEC_SAMPLING_420},
            {"440", NVIMGCODEC_SAMPLING_440}, {"422", NVIMGCODEC_SAMPLING_422}, {"411", NVIMGCODEC_SAMPLING_411},
            {"410", NVIMGCODEC_SAMPLING_410}, {"gray", NVIMGCODEC_SAMPLING_GRAY}, {"410v", NVIMGCODEC_SAMPLING_410V}};
        auto it = str2Css.find(argv[pidx + 1]);
        if (it != str2Css.end()) {
            params->chroma_subsampling = it->second;
        } else {
            std::cout << "Unknown chroma subsampling type: " << argv[pidx + 1] << std::endl;
        }
    }
    params->jpeg2k_prog_order = NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL;
    if ((pidx = find_param_index(argv, argc, "--jpeg2k_prog_order")) != -1) {
        std::map<std::string, nvimgcodecJpeg2kProgOrder_t> str2Css = {{"LRCP", NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP},
            {"RLCP", NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP}, {"RPCL", NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL},
            {"PCRL", NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL}, {"CPRL", NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL}};
        auto it = str2Css.find(argv[pidx + 1]);
        if (it != str2Css.end()) {
            params->jpeg2k_prog_order = it->second;
        } else {
            std::cout << "Unknown progression order type: " << argv[pidx + 1] << std::endl;
        }
    }

    params->batch_size = 1;
    if ((pidx = find_param_index(argv, argc, "-b")) != -1 || (pidx = find_param_index(argv, argc, "--batch_size")) != -1) {
        params->batch_size = std::atoi(argv[pidx + 1]);
    }

    params->total_images = -1;
    if ((pidx = find_param_index(argv, argc, "-t")) != -1) {
        params->total_images = std::atoi(argv[pidx + 1]);
    }

    params->list_cuda_devices = find_param_index(argv, argc, "-l") != -1;

    params->skip_hw_gpu_backend = false;
    if ((pidx = find_param_index(argv, argc, "--skip_hw_gpu_backend")) != -1) {
        params->skip_hw_gpu_backend = strcmp(argv[pidx + 1], "true") == 0;
    }


    params->device_id = 0;
    if ((pidx = find_param_index(argv, argc, "-d")) != -1 || (pidx = find_param_index(argv, argc, "--device")) != -1) {
        params->device_id = std::atoi(argv[pidx + 1]);
    }
    return -1;
}
