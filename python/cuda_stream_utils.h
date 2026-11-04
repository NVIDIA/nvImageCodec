/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

namespace nvimgcodec {

namespace py = pybind11;

// Result of coercing a Python cuda_stream argument.
// - specified: false when the argument was omitted or None (unspecified).
// - handle: opaque cudaStream_t bits as intptr_t (0 = default/null stream)
// - owner: non-none only when the handle was obtained via __cuda_stream__();
//   holding this reference keeps the foreign Python stream object alive
//   (CuPy Stream.from_external / CUDA Stream Protocol pattern).
struct CudaStreamArg {
    intptr_t handle = 0;
    py::object owner;
    bool specified = false;
};

// Accept None (unspecified), int handles, or objects implementing the CUDA
// Stream Protocol (__cuda_stream__() -> (version, stream_ptr)). Protocol
// version must be 0. Unspecified means handle 0 unless a wrapping API has
// another source (CUDA Array Interface ``stream``).
inline CudaStreamArg as_cuda_stream_arg(py::handle obj)
{
    CudaStreamArg arg;
    if (obj.is_none()) {
        return arg;
    }
    arg.specified = true;

    // Prefer the protocol so the owner is pinned even for wrapper types
    // that also expose an integer handle.
    if (py::hasattr(obj, "__cuda_stream__")) {
        py::object meth = obj.attr("__cuda_stream__");
        if (!PyCallable_Check(meth.ptr())) {
            throw py::type_error(
                "cuda_stream.__cuda_stream__ must be a callable method returning "
                "a 2-tuple (version, stream_ptr)");
        }
        py::object result = meth();
        if (!py::isinstance<py::tuple>(result) || py::len(result) != 2) {
            throw py::type_error(
                "cuda_stream.__cuda_stream__() must return a 2-tuple (version, stream_ptr)");
        }
        py::tuple tup = result.cast<py::tuple>();
        int version = 0;
        try {
            version = tup[0].cast<int>();
        } catch (const py::cast_error&) {
            throw py::type_error(
                "cuda_stream.__cuda_stream__() version must be an integer");
        }
        if (version != 0) {
            throw std::invalid_argument(
                "Unsupported __cuda_stream__ protocol version " + std::to_string(version) +
                " (only version 0 is supported)");
        }
        try {
            arg.handle = tup[1].cast<intptr_t>();
        } catch (const py::cast_error&) {
            throw py::type_error(
                "cuda_stream.__cuda_stream__() stream_ptr must be an integer handle");
        }
        arg.owner = py::reinterpret_borrow<py::object>(obj);
        return arg;
    }

    try {
        arg.handle = obj.cast<intptr_t>();
        return arg;
    } catch (const py::cast_error&) {
        throw py::type_error(
            "cuda_stream must be an int (cudaStream_t handle) or an object "
            "implementing __cuda_stream__()");
    }
}

} // namespace nvimgcodec

// Allow pybind11 to accept CudaStreamArg parameters directly from Python.
// The signature is int | object | None. None/omitted is unspecified.
namespace pybind11 {
namespace detail {
template <>
struct type_caster<nvimgcodec::CudaStreamArg> {
public:
    PYBIND11_TYPE_CASTER(nvimgcodec::CudaStreamArg, const_name("int | object | None"));

    bool load(handle src, bool /* convert */)
    {
        // Ignore convert: there are no competing CudaStreamArg overloads, and
        // throwing from as_cuda_stream_arg keeps TypeError/ValueError specific
        // instead of a generic overload mismatch. None is unspecified
        // (specified=false, handle 0), not an error.
        value = nvimgcodec::as_cuda_stream_arg(src);
        return true;
    }

    // Unused: CudaStreamArg is inbound-only and is never returned to Python.
    static handle cast(const nvimgcodec::CudaStreamArg& src, return_value_policy /* policy */, handle /* parent */)
    {
        return pybind11::int_(src.handle).release();
    }
};
} // namespace detail
} // namespace pybind11
