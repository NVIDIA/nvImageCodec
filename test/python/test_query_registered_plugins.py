# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvidia import nvimgcodec


def test_extension_ids_and_codec_names():
    extension_ids = nvimgcodec.get_extension_ids()
    assert extension_ids == nvimgcodec.get_extension_ids()
    assert "nvimgcodec_builtin_parsers" in extension_ids
    assert all(isinstance(plugin_id, str) and plugin_id for plugin_id in extension_ids)

    codec_names = nvimgcodec.get_codec_names()
    assert codec_names == nvimgcodec.get_codec_names()
    assert {"bmp", "jpeg", "jpeg2k", "png", "pnm", "tiff", "webp"} <= set(
        codec_names
    )


def test_parser_ids_and_codec_filter():
    parser_ids = nvimgcodec.get_parser_ids(codec_name="jpeg")
    assert "jpeg_parser" in parser_ids
    assert nvimgcodec.get_parser_ids(codec_name="not-a-codec") == []


def test_decoder_id_filters_are_stable_subsets():
    all_jpeg = nvimgcodec.get_decoder_ids(codec_name="jpeg")
    assert all_jpeg == nvimgcodec.get_decoder_ids(codec_name="jpeg")

    for backend_kind in (
        nvimgcodec.BackendKind.CPU_ONLY,
        nvimgcodec.BackendKind.GPU_ONLY,
        nvimgcodec.BackendKind.HYBRID_CPU_GPU,
        nvimgcodec.BackendKind.HW_GPU_ONLY,
    ):
        filtered = nvimgcodec.get_decoder_ids(
            codec_name="jpeg", backend_kind=backend_kind
        )
        assert filtered == nvimgcodec.get_decoder_ids(
            codec_name="jpeg", backend_kind=backend_kind
        )
        assert set(filtered) <= set(all_jpeg)

    assert nvimgcodec.get_decoder_ids(codec_name="not-a-codec") == []


def test_encoder_id_filters_are_stable_subsets():
    all_jpeg = nvimgcodec.get_encoder_ids(codec_name="jpeg")
    assert all_jpeg == nvimgcodec.get_encoder_ids(codec_name="jpeg")

    for backend_kind in (
        nvimgcodec.BackendKind.CPU_ONLY,
        nvimgcodec.BackendKind.GPU_ONLY,
        nvimgcodec.BackendKind.HYBRID_CPU_GPU,
        nvimgcodec.BackendKind.HW_GPU_ONLY,
    ):
        filtered = nvimgcodec.get_encoder_ids(
            codec_name="jpeg", backend_kind=backend_kind
        )
        assert filtered == nvimgcodec.get_encoder_ids(
            codec_name="jpeg", backend_kind=backend_kind
        )
        assert set(filtered) <= set(all_jpeg)

    assert nvimgcodec.get_encoder_ids(codec_name="not-a-codec") == []
