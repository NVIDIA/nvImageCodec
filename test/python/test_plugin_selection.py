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

import pytest

from nvidia import nvimgcodec


CPU_BACKENDS = [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)]


def _first_id(get_ids, backend_kind):
    for codec_name in nvimgcodec.get_codec_names():
        plugin_ids = get_ids(codec_name=codec_name, backend_kind=backend_kind)
        if plugin_ids:
            return codec_name, plugin_ids[0]
    pytest.skip(f"No registered plugin for backend {backend_kind}")


@pytest.mark.parametrize(
    "constructor,argument_name",
    [
        (nvimgcodec.Decoder, "decoder_ids"),
        (nvimgcodec.Encoder, "encoder_ids"),
    ],
)
def test_selection_argument_shape_validation(constructor, argument_name):
    kwargs = {"device_id": nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY}

    with pytest.raises(TypeError):
        constructor(**kwargs, **{argument_name: "not-a-sequence"})
    with pytest.raises(TypeError):
        constructor(**kwargs, **{argument_name: {"unordered"}})
    with pytest.raises(TypeError):
        constructor(**kwargs, **{argument_name: (plugin_id for plugin_id in ["generated"])})
    with pytest.raises(TypeError):
        constructor(**kwargs, **{argument_name: b"bytes"})
    with pytest.raises(TypeError):
        constructor(**kwargs, **{argument_name: {1: ["plugin"]}})
    with pytest.raises(ValueError):
        constructor(**kwargs, **{argument_name: []})
    with pytest.raises(ValueError):
        constructor(**kwargs, **{argument_name: {}})
    with pytest.raises(ValueError):
        constructor(**kwargs, **{argument_name: ["duplicate", "duplicate"]})
    with pytest.raises(ValueError):
        constructor(**kwargs, **{argument_name: ["plugin\0suffix"]})
    with pytest.raises(ValueError):
        constructor(**kwargs, **{argument_name: {"codec\0suffix": ["plugin"]}})


def test_unknown_and_wrong_role_ids_are_rejected():
    with pytest.raises(ValueError):
        nvimgcodec.Decoder(
            device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY,
            decoder_ids=["not_a_registered_decoder"],
        )
    with pytest.raises(ValueError):
        nvimgcodec.Encoder(
            device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY,
            encoder_ids=["not_a_registered_encoder"],
        )

    encoder_ids = nvimgcodec.get_encoder_ids()
    if encoder_ids:
        with pytest.raises(ValueError):
            nvimgcodec.Decoder(
                device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY,
                decoder_ids=[encoder_ids[0]],
            )


def test_decoder_accepts_flat_and_mapping_selections():
    codec_name, decoder_id = _first_id(
        nvimgcodec.get_decoder_ids, nvimgcodec.BackendKind.CPU_ONLY
    )

    nvimgcodec.Decoder(
        device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY,
        backends=CPU_BACKENDS,
        decoder_ids=[decoder_id],
    )
    nvimgcodec.Decoder(
        device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY,
        backends=CPU_BACKENDS,
        decoder_ids={codec_name: [decoder_id]},
    )


def test_encoder_accepts_flat_and_mapping_selections():
    codec_name, encoder_id = _first_id(
        nvimgcodec.get_encoder_ids, nvimgcodec.BackendKind.CPU_ONLY
    )

    nvimgcodec.Encoder(
        device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY,
        backends=CPU_BACKENDS,
        encoder_ids=[encoder_id],
    )
    nvimgcodec.Encoder(
        device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY,
        backends=CPU_BACKENDS,
        encoder_ids={codec_name: [encoder_id]},
    )


def test_backend_filter_overrides_selected_decoder(capfd):
    _, cpu_decoder_id = _first_id(
        nvimgcodec.get_decoder_ids, nvimgcodec.BackendKind.CPU_ONLY
    )
    _, gpu_decoder_id = _first_id(
        nvimgcodec.get_decoder_ids, nvimgcodec.BackendKind.GPU_ONLY
    )

    nvimgcodec.Decoder(
        device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY,
        backends=CPU_BACKENDS,
        decoder_ids=[gpu_decoder_id, cpu_decoder_id],
    )
    captured = capfd.readouterr()
    assert gpu_decoder_id in captured.err
    assert "excluded by the backend configuration" in captured.err
