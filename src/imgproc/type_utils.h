/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <type_traits>
#include "nvimgcodec.h"

namespace nvimgcodec {

constexpr size_t TypeSize(nvimgcodecSampleDataType_t type)
{
    return static_cast<size_t>(type) >> (8 + 3);
}

constexpr bool IsFloatingPoint(nvimgcodecSampleDataType_t type)
{
    switch (type) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64:
        return true;
    default:
        return false;
    }
}

constexpr bool IsIntegral(nvimgcodecSampleDataType_t type)
{
    switch (type) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT32:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT64:
        return true;
    default:
        return false;
    }
}

constexpr bool IsSigned(nvimgcodecSampleDataType_t type)
{
    switch (type) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT32:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT64:
        return true;
    default:
        return false;
    }
}

constexpr bool IsUnsigned(nvimgcodecSampleDataType_t type)
{
    switch (type) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64:
        return true;
    default:
        return false;
    }
}

inline int PositiveBits(nvimgcodecSampleDataType_t dtype)
{
    assert(IsIntegral(dtype));
    int positive_bits = 8 * TypeSize(dtype) - IsSigned(dtype);
    return positive_bits;
}

/**
 * @brief Expected maximum value for a given type
 */
inline double MaxValue(nvimgcodecSampleDataType_t dtype)
{
    if (!IsIntegral(dtype))
        return 1.0;
    return (uint64_t(1) << PositiveBits(dtype)) - 1;
}

inline int ActualPrecision(int precision, nvimgcodecSampleDataType_t dtype)
{
    assert(precision <= PositiveBits(dtype));
    return precision == 0 ? PositiveBits(dtype) : precision;
}

/**
 * @brief Whether given precision needs scaling to use the full width of the type
 */
inline bool NeedDynamicRangeScaling(int precision, nvimgcodecSampleDataType_t dtype)
{
    return PositiveBits(dtype) != ActualPrecision(precision, dtype);
}

/**
 * @brief Whether given precision needs scaling to convert from one precision/dtype to another
 */
inline bool NeedDynamicRangeScaling(
    int out_precision, nvimgcodecSampleDataType_t out_dtype, int in_precision, nvimgcodecSampleDataType_t in_dtype)
{
    if (out_dtype == in_dtype && ActualPrecision(out_precision, out_dtype) == ActualPrecision(in_precision, in_dtype))
        return false;
    return NeedDynamicRangeScaling(in_precision, in_dtype) || NeedDynamicRangeScaling(out_precision, out_dtype);
}

/**
 * @brief Dynamic range multiplier to apply when precision is lower than the
 *        width of the data type
 */
inline double DynamicRangeMultiplier(int precision, nvimgcodecSampleDataType_t dtype)
{
    double input_max_value = (uint64_t(1) << ActualPrecision(precision, dtype)) - 1;
    return MaxValue(dtype) / input_max_value;
}

/**
 * @brief Dynamic range multiplier to apply when precision is lower than the
 *        width of the data type on either the input or output data type
 */
inline double DynamicRangeMultiplier(
    int out_precision, nvimgcodecSampleDataType_t out_dtype, int in_precision, nvimgcodecSampleDataType_t in_dtype)
{
    return DynamicRangeMultiplier(in_precision, in_dtype) / DynamicRangeMultiplier(out_precision, out_dtype);
}

template <nvimgcodecSampleDataType_t id>
struct id2type_helper;

/**
 * @brief Compile-time mapping from a type to nvimgcodecSampleDataType_t
 *
 * @note If your compiler complains, that "Use of class template `type2id`
 * requires template arguments", include `static_swtich.h` is your file.
 */
template <typename data_type>
struct type2id;

/**
 * @brief Compile-time mapping from nvimgcodecSampleDataType_t to a type
 */
template <nvimgcodecSampleDataType_t id>
using id2type = typename id2type_helper<id>::type;

#define NVIMGCODEC_STATIC_TYPE_MAPPING(data_type, id)\
template <>\
struct type2id<data_type> : std::integral_constant<nvimgcodecSampleDataType_t, id> {};\
template <>\
struct id2type_helper<id> { using type = data_type; };


NVIMGCODEC_STATIC_TYPE_MAPPING(int8_t, NVIMGCODEC_SAMPLE_DATA_TYPE_INT8);
NVIMGCODEC_STATIC_TYPE_MAPPING(uint8_t, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8);
NVIMGCODEC_STATIC_TYPE_MAPPING(int16_t, NVIMGCODEC_SAMPLE_DATA_TYPE_INT16);
NVIMGCODEC_STATIC_TYPE_MAPPING(uint16_t, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16);
NVIMGCODEC_STATIC_TYPE_MAPPING(int32_t, NVIMGCODEC_SAMPLE_DATA_TYPE_INT32);
NVIMGCODEC_STATIC_TYPE_MAPPING(uint32_t, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32);
NVIMGCODEC_STATIC_TYPE_MAPPING(int64_t, NVIMGCODEC_SAMPLE_DATA_TYPE_INT64);
NVIMGCODEC_STATIC_TYPE_MAPPING(uint64_t, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64);
// NVIMGCODEC_STATIC_TYPE_MAPPING(float16, NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16);
NVIMGCODEC_STATIC_TYPE_MAPPING(float, NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32);
NVIMGCODEC_STATIC_TYPE_MAPPING(double, NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64);

}  // namespace nvimgcodec