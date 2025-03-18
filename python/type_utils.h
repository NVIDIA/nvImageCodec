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

#pragma once

#include <nvimgcodec.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace nvimgcodec {

inline size_t sample_type_to_bytes_per_element(nvimgcodecSampleDataType_t sample_type)
{
    //Shift by 8 since 8..15 bits represents type bitdepth,  then shift by 3 to convert to # bytes 
    return static_cast<unsigned int>(sample_type) >> (8 + 3);
}

inline bool is_sample_format_interleaved(nvimgcodecSampleFormat_t sample_format)
{
    //First bit of sample format says if this is interleaved or not  
    return static_cast<int>(sample_format) % 2 == 0 ;
}

inline std::string format_str_from_type(nvimgcodecSampleDataType_t type)
{
    switch (type) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
        return "|i1";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
        return "|u1";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
        return "<i2";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
        return "<u2";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT32:
        return "<i4";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32:
        return "<u4";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT64:
        return "<i8";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64:
        return "<u8";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16:
        return "<f2";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
        return "<f4";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64:
        return "<f8";
    default:
        break;
    }
    return "";
}

inline nvimgcodecSampleDataType_t type_from_format_str(const std::string& typestr)
{
    pybind11::ssize_t itemsize = py::dtype(typestr).itemsize();
    if (itemsize == 1) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT8;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    } else if (itemsize == 2) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16;
    } else if (itemsize == 4) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT32;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32;
    } else if (itemsize == 8) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT64;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64;
    }
    return NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN;
}

inline std::string dtype_to_str(const py::dtype& t)
{
    if (t.itemsize() == 1) {
        if (t.kind() == 'i')
            return "int8";
        if (t.kind() == 'u')
            return "uint8";
    } else if (t.itemsize() == 2) {
        if (t.kind() == 'i')
            return "int16";
        if (t.kind() == 'u')
            return "uint16";
        if (t.kind() == 'f')
            return "float16";
    } else if (t.itemsize() == 4) {
        if (t.kind() == 'i')
            return "int32";
        if (t.kind() == 'u')
            return "uint32";
        if (t.kind() == 'f')
            return "float32";
    } else if (t.itemsize() == 8) {
        if (t.kind() == 'i')
            return "int64";
        if (t.kind() == 'u')
            return "uint64";
        if (t.kind() == 'f')
            return "float64";
    }
    return "unknown type";
}

inline bool is_c_style_contiguous(const py::dict& iface)
{
     if (!iface.contains("strides")) {
        return true; // Assumed None which is for packed arrays
    }
    py::object strides = iface["strides"];  
    if (strides.is(py::none())) {
        return true;
    } else {
        py::tuple t_strides = strides.cast<py::tuple>();
        std::string type_str =  iface["typestr"].cast<std::string>();
        pybind11::ssize_t item_size = py::dtype(type_str).itemsize();

        py::tuple shape = iface["shape"].cast<py::tuple>();
        size_t stride_in_bytes = item_size;
        for (int i = shape.size() - 1; i >=0; --i) {
            if (t_strides[i].cast<size_t>() != stride_in_bytes) {
                return false;
            }
            stride_in_bytes *= shape[i].cast<size_t>();
        }
        return true;
    }

    return false;
} 

} // namespace nvimgcodec