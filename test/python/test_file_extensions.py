# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
import textwrap

import nvidia.nvimgcodec as nvimgcodec


def run_in_fresh_process(source):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_get_codec_names_with_file_extension_filter():
    names = nvimgcodec.get_codec_names(file_extension=".jpg")
    assert names[0] == "jpeg"
    assert nvimgcodec.get_codec_names(file_extension="JPEG")[0] == "jpeg"
    assert nvimgcodec.get_codec_names(file_extension=".unknown") == []


def test_get_file_extensions_with_optional_codec_filter():
    exts = nvimgcodec.get_file_extensions(codec_name="jpeg")
    assert exts == [".jpg", ".jpeg"]
    assert nvimgcodec.get_file_extensions(codec_name="jpeg2k")[0] == ".jp2"
    assert nvimgcodec.get_file_extensions(codec_name="no_such_codec") == []

    all_extensions = nvimgcodec.get_file_extensions()
    assert ".jpg" in all_extensions
    assert ".jp2" in all_extensions
    assert len(all_extensions) == len(set(all_extensions))


def test_register_and_unregister_file_extension():
    run_in_fresh_process(
        """
        from nvidia import nvimgcodec

        codec_name = "python_file_extension_only_codec"
        file_extension = ".nvimgcodec_python_test"
        assert codec_name not in nvimgcodec.get_codec_names()
        assert nvimgcodec.get_codec_names(file_extension=file_extension) == []

        nvimgcodec.register_file_extension(codec_name, file_extension)
        try:
            assert codec_name in nvimgcodec.get_codec_names()
            assert file_extension in nvimgcodec.get_file_extensions()
            assert nvimgcodec.get_file_extensions(codec_name=codec_name) == [file_extension]
            assert nvimgcodec.get_codec_names(file_extension=file_extension) == [codec_name]
            assert nvimgcodec.get_parser_ids(codec_name=codec_name) == []
            assert nvimgcodec.get_decoder_ids(codec_name=codec_name) == []
            assert nvimgcodec.get_encoder_ids(codec_name=codec_name) == []
        finally:
            nvimgcodec.unregister_file_extension(codec_name, file_extension)

        assert codec_name not in nvimgcodec.get_codec_names()
        assert file_extension not in nvimgcodec.get_file_extensions()
        assert nvimgcodec.get_codec_names(file_extension=file_extension) == []
        """
    )


def test_register_file_extension_can_change_default_codec():
    run_in_fresh_process(
        """
        from nvidia import nvimgcodec

        file_extension = ".nvimgcodec_python_default_test"
        nvimgcodec.register_file_extension("jpeg", file_extension)
        nvimgcodec.register_file_extension("jpeg2k", file_extension, is_default=True)
        try:
            assert nvimgcodec.get_codec_names(file_extension=file_extension) == ["jpeg2k", "jpeg"]
        finally:
            nvimgcodec.unregister_file_extension("jpeg2k", file_extension)
            nvimgcodec.unregister_file_extension("jpeg", file_extension)
        assert nvimgcodec.get_codec_names(file_extension=file_extension) == []
        """
    )


def test_register_file_extension_rejects_active_encoder():
    run_in_fresh_process(
        """
        from nvidia import nvimgcodec

        codec_name = "python_late_registered_codec"
        file_extension = ".nvimgcodec_late"
        cpu_backends = [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)]
        encoder = nvimgcodec.Encoder(
            device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY,
            backends=cpu_backends,
        )
        try:
            nvimgcodec.register_file_extension(codec_name, file_extension)
        except RuntimeError as error:
            assert "encoders" in str(error)
        else:
            raise AssertionError("registration unexpectedly succeeded")

        del encoder
        nvimgcodec.register_file_extension(codec_name, file_extension)
        nvimgcodec.unregister_file_extension(codec_name, file_extension)
        """
    )
