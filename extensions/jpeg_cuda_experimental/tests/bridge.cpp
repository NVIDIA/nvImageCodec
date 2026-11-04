// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
// Test-only C ABI: no Torch or Python dependency in the decoder component.
#include <exception>
#include <string>
#include "decoder.h"
using namespace nvimgcodec::jpeg_cuda_experimental;
static thread_local std::string last_error;
extern "C"
{
    const char* jce_last_error()
    {
        return last_error.c_str();
    }
    int jce_create(int device, void** output)
    {
        try {
            *output = new Decoder(device);
            return 0;
        } catch (const std::exception& e) {
            last_error = e.what();
            return -1;
        }
    }
    void jce_destroy(void* decoder)
    {
        delete static_cast<Decoder*>(decoder);
    }
    int jce_probe(const uint8_t* bytes, size_t size, int* width, int* height)
    {
        auto info = Probe(bytes, size);
        *width = info.width;
        *height = info.height;
        return static_cast<int>(info.status);
    }
    int jce_decode(void* decoder, const uint8_t* bytes, size_t size, uint8_t* buffer, size_t row_stride, size_t plane_stride, int width,
        int height, void* stream)
    {
        try {
            Output output;
            output.width = width;
            output.height = height;
            for (int c = 0; c < 3; ++c) {
                output.plane[c] = buffer + c * plane_stride;
                output.row_stride[c] = row_stride;
            }
            return static_cast<int>(static_cast<Decoder*>(decoder)->Decode(bytes, size, output, static_cast<cudaStream_t>(stream)));
        } catch (const std::exception& e) {
            last_error = e.what();
            return -1;
        }
    }
    int jce_wait(void* decoder)
    {
        try {
            return static_cast<int>(static_cast<Decoder*>(decoder)->Wait());
        } catch (const std::exception& e) {
            last_error = e.what();
            return -1;
        }
    }
}
