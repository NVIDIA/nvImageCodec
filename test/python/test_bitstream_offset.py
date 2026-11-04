# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Tests for TIFF IFD offset traversal functionality.

This module tests:
1. subifd_offsets property for accessing SubIFD byte offsets
2. bitstream_offset parameter for CodeStream creation and get_sub_code_stream
3. ifd_offset and next_ifd_offset for IFD-chain traversal
4. Python validation when ambiguous TIFF views are applied to streams
5. Decoding images via SubIFD offset
"""

from __future__ import annotations
import gc
import os
import struct
import numpy as np
from nvidia import nvimgcodec
import pytest as t
from utils import img_dir_path, is_nvcomp_supported


# Test image with SubIFD (thumbnail)
CAT_WITH_THUMBNAIL = "tiff/cat_with_thumbnail.tiff"
CAT_MAIN_WIDTH, CAT_MAIN_HEIGHT = 720, 720
CAT_THUMB_WIDTH, CAT_THUMB_HEIGHT = 180, 180

# Multi-page TIFF for IFD traversal tests
MULTI_PAGE_TIFF = "tiff/multi_page.tif"

# JPEG for non-TIFF tests
JPEG_PATH = "jpeg/padlock-406986_640_420.jpg"

def chained_ifd_tiff(num_ifds=1, *, tail="cycle"):
    """Build a classic little-endian TIFF with `num_ifds` minimal 16x16 IFDs
    chained head-to-tail. The final IFD's next pointer either loops back to the
    first IFD (tail="cycle") or points past EOF (tail="oob"), so walking past the
    last IFD fails while reaching any index in [0, num_ifds) succeeds."""
    HEADER_SIZE = 8
    IFD_SIZE = 2 + 4 * 12 + 4  # entry count + 4 entries + next-IFD offset

    def entry(tag, typ, value):
        return struct.pack("<HHII", tag, typ, 1, value)  # tag, type, count=1, inline value

    data = bytearray(struct.pack("<2sHI", b"II", 42, HEADER_SIZE))  # header -> first IFD at offset 8
    for i in range(num_ifds):
        this_off = HEADER_SIZE + i * IFD_SIZE
        if i < num_ifds - 1:
            next_off = this_off + IFD_SIZE
        elif tail == "cycle":
            next_off = HEADER_SIZE
        else:  # "oob": past end of file
            next_off = HEADER_SIZE + num_ifds * IFD_SIZE + 4096
        data += struct.pack("<H", 4)   # 4 directory entries
        data += entry(256, 4, 16)      # ImageWidth      (LONG)  = 16
        data += entry(257, 4, 16)      # ImageLength     (LONG)  = 16
        data += entry(258, 3, 8)       # BitsPerSample   (SHORT) = 8
        data += entry(277, 3, 1)       # SamplesPerPixel (SHORT) = 1
        data += struct.pack("<I", next_off)
    return bytes(data)


def cyclic_root_ifd_tiff():
    return chained_ifd_tiff(1, tail="cycle")


class TestSubIFDOffsetsProperty:
    """Tests for the subifd_offsets property on CodeStream."""

    def test_subifd_offsets_returns_list_for_tiff_with_subifd(self):
        """Test that subifd_offsets returns correct offsets for TIFF with SubIFD."""
        fpath = os.path.join(img_dir_path, CAT_WITH_THUMBNAIL)
        cs = nvimgcodec.CodeStream(fpath)

        offsets = cs.subifd_offsets
        assert isinstance(offsets, list), f"subifd_offsets should return list, got {type(offsets)}"
        assert len(offsets) == 1, f"Expected 1 SubIFD offset, got {len(offsets)}"
        assert offsets[0] > 0, f"SubIFD offset should be positive, got {offsets[0]}"

    def test_subifd_offsets_empty_for_tiff_without_subifd(self):
        """Test that subifd_offsets returns empty list for TIFF without SubIFD."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        offsets = cs.subifd_offsets
        assert isinstance(offsets, list), f"subifd_offsets should return list, got {type(offsets)}"
        assert len(offsets) == 0, f"Expected empty list for TIFF without SubIFD, got {offsets}"

    def test_subifd_offsets_empty_for_non_tiff(self):
        """Test that subifd_offsets returns empty list for non-TIFF files."""
        fpath = os.path.join(img_dir_path, JPEG_PATH)
        cs = nvimgcodec.CodeStream(fpath)

        offsets = cs.subifd_offsets
        assert isinstance(offsets, list), f"subifd_offsets should return list, got {type(offsets)}"
        assert len(offsets) == 0, f"Expected empty list for JPEG, got {offsets}"

    def test_subifd_offset_can_be_used_to_access_thumbnail(self):
        """Test that SubIFD offset from property can be used to create CodeStream for thumbnail."""
        fpath = os.path.join(img_dir_path, CAT_WITH_THUMBNAIL)
        cs = nvimgcodec.CodeStream(fpath)

        # Verify main image dimensions
        assert cs.width == CAT_MAIN_WIDTH, f"Main width should be {CAT_MAIN_WIDTH}, got {cs.width}"
        assert cs.height == CAT_MAIN_HEIGHT, f"Main height should be {CAT_MAIN_HEIGHT}, got {cs.height}"

        offsets = cs.subifd_offsets
        assert len(offsets) > 0, "Test file should have SubIFD"

        # Access thumbnail using the offset
        thumb_cs = nvimgcodec.CodeStream(fpath, bitstream_offset=offsets[0])

        # Verify thumbnail dimensions
        assert thumb_cs.width == CAT_THUMB_WIDTH, f"Thumbnail width should be {CAT_THUMB_WIDTH}, got {thumb_cs.width}"
        assert thumb_cs.height == CAT_THUMB_HEIGHT, f"Thumbnail height should be {CAT_THUMB_HEIGHT}, got {thumb_cs.height}"

    def test_subifd_offset_via_get_sub_code_stream(self):
        """Test that SubIFD offset works with get_sub_code_stream."""
        fpath = os.path.join(img_dir_path, CAT_WITH_THUMBNAIL)
        cs = nvimgcodec.CodeStream(fpath)

        offsets = cs.subifd_offsets
        assert len(offsets) > 0, "Test file should have SubIFD"

        # Access via get_sub_code_stream
        thumb_cs = cs.get_sub_code_stream(bitstream_offset=offsets[0])

        # Should get same dimensions as top-level CodeStream with offset
        thumb_direct = nvimgcodec.CodeStream(fpath, bitstream_offset=offsets[0])
        assert thumb_cs.width == thumb_direct.width
        assert thumb_cs.height == thumb_direct.height

    def test_subifd_offsets_on_sub_code_stream(self):
        """Test that subifd_offsets works on sub-code streams too."""
        fpath = os.path.join(img_dir_path, CAT_WITH_THUMBNAIL)
        cs = nvimgcodec.CodeStream(fpath)

        # Get sub-code stream for main image
        main_cs = cs.get_sub_code_stream(image_idx=0)
        offsets = main_cs.subifd_offsets

        assert isinstance(offsets, list)
        assert len(offsets) == 1, "Main image should have 1 SubIFD"

    def test_thumbnail_has_no_subifds(self):
        """Test that thumbnail (SubIFD) itself has no nested SubIFDs."""
        fpath = os.path.join(img_dir_path, CAT_WITH_THUMBNAIL)
        cs = nvimgcodec.CodeStream(fpath)

        offsets = cs.subifd_offsets
        assert len(offsets) > 0, "Test file should have SubIFD"

        # Access thumbnail
        thumb_cs = nvimgcodec.CodeStream(fpath, bitstream_offset=offsets[0])

        # Thumbnail should not have nested SubIFDs
        thumb_offsets = thumb_cs.subifd_offsets
        assert len(thumb_offsets) == 0, "Thumbnail should not have nested SubIFDs"


class TestSubIFDDecoding:
    """Tests for decoding images via SubIFD offset."""

    @t.mark.parametrize("backends", [
        [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)],
        [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)]
    ])
    def test_decode_subifd_thumbnail(self, backends):
        """Test decoding thumbnail image from SubIFD offset."""

        fpath = os.path.join(img_dir_path, CAT_WITH_THUMBNAIL)
        decoder = nvimgcodec.Decoder(backends=backends)
        cs = nvimgcodec.CodeStream(fpath)

        # Verify main image dimensions
        assert cs.width == CAT_MAIN_WIDTH, f"Main width should be {CAT_MAIN_WIDTH}, got {cs.width}"
        assert cs.height == CAT_MAIN_HEIGHT, f"Main height should be {CAT_MAIN_HEIGHT}, got {cs.height}"

        # Get SubIFD offset using subifd_offsets property
        offsets = cs.subifd_offsets
        assert len(offsets) > 0, "Test file should have SubIFD"

        # Create sub-code stream at SubIFD offset
        thumb_cs = cs.get_sub_code_stream(bitstream_offset=offsets[0])

        # Verify thumbnail dimensions
        assert thumb_cs.width == CAT_THUMB_WIDTH, f"Thumbnail width should be {CAT_THUMB_WIDTH}, got {thumb_cs.width}"
        assert thumb_cs.height == CAT_THUMB_HEIGHT, f"Thumbnail height should be {CAT_THUMB_HEIGHT}, got {thumb_cs.height}"

        # Decode the thumbnail
        thumb_img = decoder.decode(thumb_cs)
        assert thumb_img is not None, "Failed to decode thumbnail"

        # Verify decoded image shape matches expected dimensions
        img_np = thumb_img.cpu()
        assert img_np.shape[0] == CAT_THUMB_HEIGHT, f"Decoded height should be {CAT_THUMB_HEIGHT}, got {img_np.shape[0]}"
        assert img_np.shape[1] == CAT_THUMB_WIDTH, f"Decoded width should be {CAT_THUMB_WIDTH}, got {img_np.shape[1]}"


class TestBitstreamOffsetInheritance:
    """Tests for bitstream_offset inheritance in nested sub-code streams."""

    @t.mark.parametrize("backends", [
        [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)],
        [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
    ])
    def test_bitstream_offset_inherited_from_parent(self, backends):
        """Test that bitstream_offset is inherited when creating sub-code stream from parent."""

        fpath = os.path.join(img_dir_path, CAT_WITH_THUMBNAIL)
        decoder = nvimgcodec.Decoder(backends=backends)
        cs = nvimgcodec.CodeStream(fpath)

        # Get SubIFD offset
        offsets = cs.subifd_offsets
        assert len(offsets) > 0, "Test file should have SubIFD"

        # Create parent sub-code stream with bitstream_offset
        parent_cs = cs.get_sub_code_stream(bitstream_offset=offsets[0])

        # Verify parent is at thumbnail dimensions
        assert parent_cs.width == CAT_THUMB_WIDTH, f"Parent width should be {CAT_THUMB_WIDTH}, got {parent_cs.width}"
        assert parent_cs.height == CAT_THUMB_HEIGHT, f"Parent height should be {CAT_THUMB_HEIGHT}, got {parent_cs.height}"

        # Create child sub-code stream from parent (should inherit bitstream_offset)
        child_cs = parent_cs.get_sub_code_stream(image_idx=0)

        # Child should have same thumbnail dimensions
        assert child_cs.width == CAT_THUMB_WIDTH, f"Child width should be {CAT_THUMB_WIDTH}, got {child_cs.width}"
        assert child_cs.height == CAT_THUMB_HEIGHT, f"Child height should be {CAT_THUMB_HEIGHT}, got {child_cs.height}"

        # Decode from child should work and produce same result as decoding from parent
        parent_img = decoder.decode(parent_cs).cpu()
        child_img = decoder.decode(child_cs).cpu()

        np.testing.assert_array_equal(parent_img, child_img,
            err_msg="Child sub-code stream should decode same image as parent")

    def test_inherited_bitstream_offset_rejects_nonzero_image_idx(self):
        """Test that inherited bitstream_offset cannot be combined with nonzero image_idx."""
        fpath = os.path.join(img_dir_path, CAT_WITH_THUMBNAIL)
        cs = nvimgcodec.CodeStream(fpath)

        offsets = cs.subifd_offsets
        assert len(offsets) > 0, "Test file should have SubIFD"

        parent_cs = cs.get_sub_code_stream(bitstream_offset=offsets[0])

        with t.raises(RuntimeError, match="Cannot apply nonzero image_idx to a sub code stream"):
            parent_cs.get_sub_code_stream(image_idx=1)

    def test_bitstream_offset_not_inherited_when_explicitly_set(self):
        """Test that explicit bitstream_offset overrides parent's offset."""
        fpath = os.path.join(img_dir_path, CAT_WITH_THUMBNAIL)
        cs = nvimgcodec.CodeStream(fpath)

        # Verify main image dimensions
        assert cs.width == CAT_MAIN_WIDTH, f"Main width should be {CAT_MAIN_WIDTH}, got {cs.width}"
        assert cs.height == CAT_MAIN_HEIGHT, f"Main height should be {CAT_MAIN_HEIGHT}, got {cs.height}"

        # Get SubIFD offset
        offsets = cs.subifd_offsets
        assert len(offsets) > 0, "Test file should have SubIFD"

        # Create parent at SubIFD offset (thumbnail)
        parent_cs = cs.get_sub_code_stream(bitstream_offset=offsets[0])

        # Verify parent is at thumbnail dimensions
        assert parent_cs.width == CAT_THUMB_WIDTH, f"Thumbnail width should be {CAT_THUMB_WIDTH}, got {parent_cs.width}"
        assert parent_cs.height == CAT_THUMB_HEIGHT, f"Thumbnail height should be {CAT_THUMB_HEIGHT}, got {parent_cs.height}"


class TestIfdTraversal:
    """Tests for TIFF IFD-chain traversal via ifd_offset and next_ifd_offset."""

    def test_root_reports_current_and_next_ifd_offsets(self):
        """Test that root CodeStream exposes the first and second top-level IFD offsets."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images
        assert total_images > 2, f"Test file should have >2 images, got {total_images}"

        assert isinstance(cs.ifd_offset, int)
        assert cs.ifd_offset > 0
        assert isinstance(cs.next_ifd_offset, int)
        assert cs.next_ifd_offset > 0
        assert cs.next_ifd_offset != cs.ifd_offset

    def test_offset_selected_stream_is_single_image_and_reports_next_ifd(self):
        """Test that a CodeStream selected by IFD offset represents exactly that image."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        second_offset = cs.next_ifd_offset
        assert second_offset is not None, "Test file should have a second IFD"

        second_by_offset = cs.get_sub_code_stream(bitstream_offset=second_offset)
        second_by_index = cs.get_sub_code_stream(image_idx=1)

        assert second_by_offset.num_images == 1
        assert second_by_offset.ifd_offset == second_offset
        assert second_by_offset.width == second_by_index.width
        assert second_by_offset.height == second_by_index.height

    def test_image_idx_substream_is_single_image_and_reports_next_ifd(self):
        """Test that image_idx selection also produces a one-image view."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        root = nvimgcodec.CodeStream(fpath)

        second_by_index = root.get_sub_code_stream(image_idx=1)

        assert second_by_index.num_images == 1
        assert second_by_index.ifd_offset is not None
        assert second_by_index.next_ifd_offset is not None

    def test_page_zero_substream_rejects_nested_nonzero_image_idx(self):
        """Test that any image_idx-selected substream remains a one-image view."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        root = nvimgcodec.CodeStream(fpath)

        page0 = root.get_sub_code_stream(image_idx=0)

        with t.raises(RuntimeError):
            page0.get_sub_code_stream(image_idx=1)

    def test_next_ifd_offset_chain_visits_all_root_images(self):
        """Test that walking next_ifd_offset visits each top-level image once."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        root = nvimgcodec.CodeStream(fpath)

        total_images = root.num_images
        assert total_images >= 4, "Test file should have >=4 images"

        visited_offsets = []
        current = root
        while True:
            visited_offsets.append(current.ifd_offset)
            next_offset = current.next_ifd_offset
            if next_offset is None:
                break
            current = root.get_sub_code_stream(bitstream_offset=next_offset)

        assert len(visited_offsets) == total_images
        assert len(set(visited_offsets)) == total_images

    def test_ifd_traversal_accesses_same_pages_as_image_idx(self):
        """Test that offset traversal and image_idx traversal select the same IFDs."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        root = nvimgcodec.CodeStream(fpath)

        total_images = root.num_images
        assert total_images >= 4, "Test file should have >=4 images"

        current = root
        for image_idx in range(total_images):
            by_index = root.get_sub_code_stream(image_idx=image_idx)

            assert current.width == by_index.width
            assert current.height == by_index.height
            assert current.ifd_offset == by_index.ifd_offset

            next_offset = current.next_ifd_offset
            if next_offset is None:
                break
            current = root.get_sub_code_stream(bitstream_offset=next_offset)

    def test_next_ifd_offset_none_on_last_page(self):
        """Test that next_ifd_offset is None for the last IFD in the root chain."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        root = nvimgcodec.CodeStream(fpath)

        current = root
        while True:
            next_offset = current.next_ifd_offset
            if next_offset is None:
                break
            current = root.get_sub_code_stream(bitstream_offset=next_offset)

        assert current.next_ifd_offset is None


class TestTopLevelCodeStreamParameters:
    """Tests for bitstream_offset at CodeStream creation."""

    def test_codestream_bitstream_offset_parameter(self):
        """Test that CodeStream accepts bitstream_offset at creation."""
        fpath = os.path.join(img_dir_path, CAT_WITH_THUMBNAIL)
        cs = nvimgcodec.CodeStream(fpath)

        # Verify main image dimensions
        assert cs.width == CAT_MAIN_WIDTH, f"Main width should be {CAT_MAIN_WIDTH}, got {cs.width}"
        assert cs.height == CAT_MAIN_HEIGHT, f"Main height should be {CAT_MAIN_HEIGHT}, got {cs.height}"

        offsets = cs.subifd_offsets
        assert len(offsets) > 0, "Test file should have SubIFD"

        # Create CodeStream with offset directly
        thumb_cs = nvimgcodec.CodeStream(fpath, bitstream_offset=offsets[0])

        # Verify thumbnail dimensions
        assert thumb_cs.width == CAT_THUMB_WIDTH, f"Thumbnail width should be {CAT_THUMB_WIDTH}, got {thumb_cs.width}"
        assert thumb_cs.height == CAT_THUMB_HEIGHT, f"Thumbnail height should be {CAT_THUMB_HEIGHT}, got {thumb_cs.height}"

class TestCodeStreamViewBitstreamOffset:
    """Tests for CodeStreamView bitstream_offset parameter."""

    def test_code_stream_view_bitstream_offset(self):
        """Test that CodeStreamView properly stores bitstream_offset."""
        view = nvimgcodec.CodeStreamView(image_idx=0, bitstream_offset=1000)

        assert view.image_idx == 0
        assert view.bitstream_offset == 1000

    def test_code_stream_view_stores_image_idx_with_bitstream_offset(self):
        """CodeStreamView is a value object; ambiguity is rejected when used."""
        view = nvimgcodec.CodeStreamView(image_idx=2, bitstream_offset=5000)

        assert view.image_idx == 2
        assert view.bitstream_offset == 5000

    def test_code_stream_view_accepts_bitstream_offset_zero(self):
        """Explicit bitstream_offset=0 is the default offset."""
        view = nvimgcodec.CodeStreamView(bitstream_offset=0)

        assert view.image_idx == 0
        assert view.bitstream_offset == 0

    def test_code_stream_view_accepts_bitstream_offset_none(self):
        """Test that None remains the Python spelling for the default offset."""
        view = nvimgcodec.CodeStreamView(bitstream_offset=None)

        assert view.image_idx == 0
        assert view.bitstream_offset == 0


class TestErrorHandling:
    """Tests for error handling in TIFF IFD traversal."""

    def test_invalid_bitstream_offset_raises_error(self):
        """Test that invalid bitstream_offset raises an error."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)

        # Very large offset that's definitely invalid
        invalid_offset = 999999999999

        # Should raise an error when parsing
        with t.raises(RuntimeError):
            cs = nvimgcodec.CodeStream(fpath, bitstream_offset=invalid_offset)
            # Force parsing by accessing a property
            _ = cs.width

    def test_bitstream_offset_at_actual_ifd(self):
        """Test bitstream_offset pointing to an actual IFD offset."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        second_ifd_offset = cs.next_ifd_offset
        assert second_ifd_offset is not None, "Should have second IFD"

        # Create CodeStream at that offset - should work
        cs_at_offset = nvimgcodec.CodeStream(fpath, bitstream_offset=second_ifd_offset)
        assert cs_at_offset.width > 0, "Should successfully parse at valid IFD offset"
        assert cs_at_offset.height > 0, "Should successfully parse at valid IFD offset"

    def test_direct_bitstream_offset_reports_next_ifd_offset(self):
        """Test that direct offset streams can continue root IFD traversal."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        root = nvimgcodec.CodeStream(fpath)
        second_ifd_offset = root.next_ifd_offset
        assert second_ifd_offset is not None, "Test file should have a second IFD"

        second = nvimgcodec.CodeStream(fpath, bitstream_offset=second_ifd_offset)

        assert second.ifd_offset == second_ifd_offset
        assert second.next_ifd_offset is not None
        assert second.next_ifd_offset != second.ifd_offset

    def test_codestream_accepts_bitstream_offset_zero_as_default(self):
        """Explicit bitstream_offset=0 is accepted as the default view."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        root = nvimgcodec.CodeStream(fpath)
        explicit_default = nvimgcodec.CodeStream(fpath, bitstream_offset=0)

        assert explicit_default.width == root.width
        assert explicit_default.height == root.height

    def test_get_sub_code_stream_rejects_image_idx_with_bitstream_offset(self):
        """Test that applying image_idx plus an explicit IFD offset is rejected."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        with t.raises(RuntimeError):
            cs.get_sub_code_stream(image_idx=1, bitstream_offset=cs.next_ifd_offset)

    def test_get_sub_code_stream_rejects_ambiguous_code_stream_view(self):
        """Ambiguous TIFF selection is rejected when a CodeStreamView is applied."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)
        view = nvimgcodec.CodeStreamView(image_idx=1, bitstream_offset=cs.next_ifd_offset)

        with t.raises(RuntimeError):
            cs.get_sub_code_stream(view)

    def test_get_sub_code_stream_accepts_bitstream_offset_zero_as_default(self):
        """Explicit bitstream_offset=0 is accepted as the default child view."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)
        sub = cs.get_sub_code_stream(bitstream_offset=0)

        assert sub.width == cs.width
        assert sub.height == cs.height

    def test_get_sub_code_stream_view_keeps_parent_alive(self):
        """Regression test for a pybind11 keep_alive crash.

        get_sub_code_stream() is overloaded: one form takes (image_idx, region,
        bitstream_offset), the other takes a single CodeStreamView. Calling the
        CodeStreamView form first makes pybind11's overload resolution try (and
        reject) the scalar-args form, since a CodeStreamView does not bind to
        image_idx: size_t. On pybind11 3.1.0 this rejection previously segfaulted
        the process, because both overloads carried a declarative
        py::keep_alive<0, 1>() policy and 3.1.0's dispatcher ran that policy's
        postcall hook even on the rejected overload, dereferencing the internal
        PYBIND11_TRY_NEXT_OVERLOAD sentinel instead of a real object. The fix
        applies keep_alive manually inside each overload's body instead of via
        the declarative policy, so this call must not crash on any pybind11
        version, and the manual keep_alive must still do its job: the parent
        CodeStream's ``bytes`` buffer must stay alive for as long as the sub
        code stream returned here is alive, even after the parent's only Python
        reference is dropped and collected.
        """
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        with open(fpath, "rb") as f:
            data = f.read()

        # Constructed from `bytes` (rather than a file path) so the parent
        # CodeStream owns a Python object holding the encoded data - the case
        # the keep_alive policy exists to protect, per the binding's own comment.
        cs = nvimgcodec.CodeStream(data)
        second_ifd_offset = cs.next_ifd_offset
        assert second_ifd_offset is not None, "Test file should have a second IFD"
        view = nvimgcodec.CodeStreamView(bitstream_offset=second_ifd_offset)

        sub = cs.get_sub_code_stream(view)

        # Drop both the parent and this test's own reference to the raw buffer.
        # If `data` stayed alive here, its refcount alone would keep the buffer
        # valid even with a broken keep_alive, and the assertion below would
        # pass regardless of whether the fix actually works. `sub` must be the
        # only thing left keeping the buffer alive.
        del cs
        del data
        gc.collect()

        # `sub` must still be valid: keep_alive must have kept the parent (and
        # therefore its `bytes` buffer) alive for as long as `sub` references it.
        assert sub.width > 0
        assert sub.height > 0

    def test_cyclic_root_ifd_chain_num_images_fails(self):
        """A malformed TIFF root-chain cycle makes num_images fail (cycle detection)
        rather than looping forever."""
        cs = nvimgcodec.CodeStream(cyclic_root_ifd_tiff())
        with t.raises(RuntimeError):
            _ = cs.num_images

    def test_root_view_does_not_walk_full_root_ifd_chain(self):
        """Root view is known locally and should not need exact root num_images."""
        assert nvimgcodec.CodeStream(cyclic_root_ifd_tiff()).view is None

    def test_root_first_image_metadata_does_not_walk_full_root_ifd_chain(self):
        """First-image metadata should not need exact root num_images."""
        cs = nvimgcodec.CodeStream(cyclic_root_ifd_tiff())

        assert cs.width == 16
        assert cs.height == 16
        with t.raises(RuntimeError):
            _ = cs.num_images

    def test_offset_view_first_image_is_lazy_on_cyclic_root(self):
        """A one-image view parses only its IFD, so it succeeds on a cyclic root
        where the full-chain num_images cannot."""
        root = nvimgcodec.CodeStream(cyclic_root_ifd_tiff())
        first = root.get_sub_code_stream(0)

        assert first.num_images == 1
        assert first.width == 16
        assert first.height == 16
        assert first.ifd_offset == 8
        # next_ifd_offset is the raw stored pointer (here the self-cycle):
        # it is read, not followed, so it does not trip cycle detection.
        assert first.next_ifd_offset == first.ifd_offset

        # Walking the full chain is the operation that fails on the cycle.
        with t.raises(RuntimeError):
            _ = root.num_images

    def test_root_offset_fields_walk_full_root_ifd_chain(self):
        """Root-level ifd_offset/next_ifd_offset/subifd_offsets go through the full
        root walk (to report the true num_images), unlike width/height, so they fail
        on a cyclic root."""
        # width/height stay lazy (resolved through an image-0 view) and succeed.
        assert nvimgcodec.CodeStream(cyclic_root_ifd_tiff()).width == 16

        # The offset/SubIFD properties on the root walk the chain and fail.
        for attr in ("ifd_offset", "next_ifd_offset", "subifd_offsets"):
            root = nvimgcodec.CodeStream(cyclic_root_ifd_tiff())
            with t.raises(RuntimeError):
                _ = getattr(root, attr)

    def test_image_idx_parses_only_up_to_requested_index(self):
        """image_idx=k walks only to IFD k. With a cycle past the last valid IFD,
        reaching any index in [0, num_ifds) succeeds, while num_images (the full
        walk) and an out-of-range index follow the cyclic tail and fail."""
        num_ifds = 3
        root = nvimgcodec.CodeStream(chained_ifd_tiff(num_ifds, tail="cycle"))

        # Selecting any valid index parses only up to that IFD and succeeds,
        # without following the (cyclic) tail after the last requested IFD.
        for k in range(num_ifds):
            view = root.get_sub_code_stream(image_idx=k)
            assert view.num_images == 1
            assert view.width == 16 and view.height == 16

        # Going past the last valid IFD follows the cyclic tail and fails.
        with t.raises(RuntimeError):
            _ = root.get_sub_code_stream(image_idx=num_ifds).width
        with t.raises(RuntimeError):
            _ = root.num_images


def test_non_tiff_invalid_image_idx_fails_at_substream_creation():
    """Image-indexed substream creation should reject invalid single-image indexes."""
    fpath = os.path.join(img_dir_path, JPEG_PATH)
    cs = nvimgcodec.CodeStream(fpath)

    with t.raises(RuntimeError):
        cs.get_sub_code_stream(image_idx=1)
