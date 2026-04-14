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
Tests for bitstream_offset and pagination functionality.

This module tests:
1. subifd_offsets property for accessing SubIFD byte offsets
2. bitstream_offset parameter for CodeStream creation and get_sub_code_stream
3. limit_images parameter for pagination
4. next_bitstream_offset for pagination continuation
5. Decoding images via SubIFD offset
"""

from __future__ import annotations
import os
import numpy as np
from nvidia import nvimgcodec
import pytest as t
from utils import img_dir_path, is_nvcomp_supported


# Test image with SubIFD (thumbnail)
CAT_WITH_THUMBNAIL = "tiff/cat_with_thumbnail.tiff"
CAT_MAIN_WIDTH, CAT_MAIN_HEIGHT = 720, 720
CAT_THUMB_WIDTH, CAT_THUMB_HEIGHT = 180, 180

# Multi-page TIFF for pagination tests
MULTI_PAGE_TIFF = "tiff/multi_page.tif"

# JPEG for non-TIFF tests
JPEG_PATH = "jpeg/padlock-406986_640_420.jpg"


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


class TestPagination:
    """Tests for pagination with limit_images and next_bitstream_offset."""

    def test_limit_images_restricts_num_images(self):
        """Test that limit_images restricts the number of images in code stream."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images
        assert total_images > 2, f"Test file should have >2 images, got {total_images}"

        # Create sub-code stream with limit
        limit = 2
        limited_cs = cs.get_sub_code_stream(limit_images=limit)

        assert limited_cs.num_images == limit, \
            f"Limited code stream should have {limit} images, got {limited_cs.num_images}"

    def test_next_bitstream_offset_for_pagination(self):
        """Test that next_bitstream_offset allows continuing pagination."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images
        assert total_images >= 4, f"Test file should have >=4 images, got {total_images}"

        # Get first batch
        batch_size = 2
        batch1_cs = cs.get_sub_code_stream(limit_images=batch_size)

        assert batch1_cs.num_images == batch_size

        # Get next offset
        next_offset = batch1_cs.next_bitstream_offset
        assert next_offset is not None, "next_bitstream_offset should not be None when more images exist"

        # Get second batch starting from next offset
        batch2_cs = cs.get_sub_code_stream(bitstream_offset=next_offset, limit_images=batch_size)

        assert batch2_cs.num_images == batch_size, \
            f"Second batch should have {batch_size} images, got {batch2_cs.num_images}"

    def test_pagination_accesses_different_pages(self):
        """Test that pagination actually accesses different pages by comparing dimensions."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images
        assert total_images >= 4, f"Test file should have >=4 images"

        # Get first page dimensions
        page0 = cs.get_sub_code_stream(image_idx=0)
        page0_dims = (page0.width, page0.height)

        # Get third page dimensions (page 3 has different size in multi_page.tif)
        page3 = cs.get_sub_code_stream(image_idx=3)
        page3_dims = (page3.width, page3.height)

        # Verify pages have different dimensions (multi_page.tif has varied sizes)
        assert page0_dims != page3_dims, \
            f"Test file should have pages with different dimensions, got {page0_dims} and {page3_dims}"

        # Now test pagination - get batch starting at page 3
        batch1 = cs.get_sub_code_stream(limit_images=3)
        next_offset = batch1.next_bitstream_offset
        assert next_offset is not None, "Should have more pages"

        # Get batch starting at offset (should be page 3)
        batch2 = cs.get_sub_code_stream(bitstream_offset=next_offset, limit_images=1)
        batch2_first = batch2.get_sub_code_stream(image_idx=0)

        # Batch2's first page should have page3's dimensions
        assert batch2_first.width == page3_dims[0], "Pagination should access page 3"
        assert batch2_first.height == page3_dims[1], "Pagination should access page 3"


class TestPaginationEdgeCases:
    """Tests for pagination edge cases."""

    def test_limit_images_one(self):
        """Test pagination with limit_images=1."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images
        assert total_images > 1, "Test file should have >1 images"

        limited_cs = cs.get_sub_code_stream(limit_images=1)
        assert limited_cs.num_images == 1, \
            f"limit_images=1 should return 1 image, got {limited_cs.num_images}"

        # Should have next offset since there are more images
        assert limited_cs.next_bitstream_offset is not None, \
            "next_bitstream_offset should not be None when more images exist"

    def test_limit_images_greater_than_total(self):
        """Test that limit_images greater than total returns all images."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images
        large_limit = total_images + 100

        limited_cs = cs.get_sub_code_stream(limit_images=large_limit)
        assert limited_cs.num_images == total_images, \
            f"limit_images > total should return all {total_images} images, got {limited_cs.num_images}"

        # No more images, so next_bitstream_offset should be None
        assert limited_cs.next_bitstream_offset is None, \
            "next_bitstream_offset should be None when no more images exist"

    def test_limit_images_zero_means_no_limit(self):
        """Test that limit_images=0 returns all images (no limit)."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images

        # limit_images=0 should mean no limit
        unlimited_cs = cs.get_sub_code_stream(limit_images=0)
        assert unlimited_cs.num_images == total_images, \
            f"limit_images=0 should return all {total_images} images, got {unlimited_cs.num_images}"

    def test_next_bitstream_offset_none_on_last_page(self):
        """Test that next_bitstream_offset is None when on last page."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images

        # Get all images - should have no next offset
        all_cs = cs.get_sub_code_stream(limit_images=total_images)
        assert all_cs.next_bitstream_offset is None, \
            "next_bitstream_offset should be None when all images are included"

    def test_pagination_chain_visits_all_images(self):
        """Test that chaining pagination visits all images exactly once."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images
        assert total_images >= 4, "Test file should have >=4 images"

        batch_size = 2
        visited_count = 0
        current_offset = None

        # Chain through all batches
        while True:
            if current_offset is None:
                batch_cs = cs.get_sub_code_stream(limit_images=batch_size)
            else:
                batch_cs = cs.get_sub_code_stream(bitstream_offset=current_offset, limit_images=batch_size)

            visited_count += batch_cs.num_images
            current_offset = batch_cs.next_bitstream_offset

            if current_offset is None:
                break

        assert visited_count == total_images, \
            f"Pagination should visit all {total_images} images, visited {visited_count}"

    def test_pagination_last_batch_partial(self):
        """Test pagination with odd number of images in last batch."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images
        # Choose batch size that doesn't divide evenly
        batch_size = 3
        expected_last_batch = total_images % batch_size
        if expected_last_batch == 0:
            expected_last_batch = batch_size

        # Navigate to last batch
        current_offset = None
        while True:
            if current_offset is None:
                batch_cs = cs.get_sub_code_stream(limit_images=batch_size)
            else:
                batch_cs = cs.get_sub_code_stream(bitstream_offset=current_offset, limit_images=batch_size)

            next_offset = batch_cs.next_bitstream_offset
            if next_offset is None:
                # This is the last batch
                last_batch_count = batch_cs.num_images
                break
            current_offset = next_offset

        assert last_batch_count == expected_last_batch, \
            f"Last batch should have {expected_last_batch} images, got {last_batch_count}"

    def test_different_limit_images_on_same_stream(self):
        """Test that changing limit_images on the same file invalidates the
        decoder's parsed-stream cache and produces correct results."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        decoder = nvimgcodec.Decoder()

        # First decode with limit_images=2
        cs1 = nvimgcodec.CodeStream(fpath, limit_images=2)
        assert cs1.num_images == 2, \
            f"First code stream should have 2 images, got {cs1.num_images}"
        imgs1 = decoder.decode(cs1)

        # Second decode of the same file with a different limit_images
        cs2 = nvimgcodec.CodeStream(fpath, limit_images=4)
        assert cs2.num_images == 4, \
            f"Second code stream should have 4 images, got {cs2.num_images}"
        imgs2 = decoder.decode(cs2)

        # Third decode with no limit (all images)
        cs3 = nvimgcodec.CodeStream(fpath)
        total_images = cs3.num_images
        assert total_images > 4, "Test file should have >4 images"
        imgs3 = decoder.decode(cs3)


class TestTopLevelCodeStreamParameters:
    """Tests for bitstream_offset and limit_images at CodeStream creation."""

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

    def test_codestream_limit_images_parameter(self):
        """Test that CodeStream accepts limit_images at creation."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)

        # Without limit
        cs_all = nvimgcodec.CodeStream(fpath)
        total_images = cs_all.num_images
        assert total_images > 2, "Test file should have >2 images"

        # With limit
        cs_limited = nvimgcodec.CodeStream(fpath, limit_images=2)
        assert cs_limited.num_images == 2, \
            f"CodeStream with limit_images=2 should have 2 images, got {cs_limited.num_images}"

    def test_codestream_combined_parameters(self):
        """Test CodeStream with both bitstream_offset and limit_images."""
        fpath = os.path.join(img_dir_path, MULTI_PAGE_TIFF)
        cs = nvimgcodec.CodeStream(fpath)

        total_images = cs.num_images
        assert total_images >= 4, "Test file should have >=4 images"

        # Get offset to second page
        batch1 = cs.get_sub_code_stream(limit_images=1)
        second_page_offset = batch1.next_bitstream_offset
        assert second_page_offset is not None, "Should have offset to second page"

        # Create CodeStream starting at second page, limited to 2 images
        cs_partial = nvimgcodec.CodeStream(fpath, bitstream_offset=second_page_offset, limit_images=2)
        assert cs_partial.num_images == 2, \
            f"Expected 2 images, got {cs_partial.num_images}"

        # Verify dimensions match second page (not first page)
        second_page_cs = cs.get_sub_code_stream(image_idx=1)

        # The partial CodeStream's first image should match original second page
        partial_first = cs_partial.get_sub_code_stream(image_idx=0)
        assert partial_first.width == second_page_cs.width, \
            "First image of partial should match second page width"
        assert partial_first.height == second_page_cs.height, \
            "First image of partial should match second page height"


class TestCodeStreamViewBitstreamOffset:
    """Tests for CodeStreamView bitstream_offset parameter."""

    def test_code_stream_view_bitstream_offset(self):
        """Test that CodeStreamView properly stores bitstream_offset."""
        view = nvimgcodec.CodeStreamView(image_idx=0, bitstream_offset=1000)

        assert view.image_idx == 0
        assert view.bitstream_offset == 1000

    def test_code_stream_view_limit_images(self):
        """Test that CodeStreamView properly stores limit_images."""
        view = nvimgcodec.CodeStreamView(image_idx=0, limit_images=5)

        assert view.image_idx == 0
        assert view.limit_images == 5

    def test_code_stream_view_combined_params(self):
        """Test that CodeStreamView properly stores all parameters."""
        view = nvimgcodec.CodeStreamView(
            image_idx=2,
            bitstream_offset=5000,
            limit_images=10
        )

        assert view.image_idx == 2
        assert view.bitstream_offset == 5000
        assert view.limit_images == 10


class TestErrorHandling:
    """Tests for error handling in pagination."""

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

        # Get the offset to the second page (a valid IFD offset)
        batch1 = cs.get_sub_code_stream(limit_images=1)
        second_ifd_offset = batch1.next_bitstream_offset
        assert second_ifd_offset is not None, "Should have second IFD"

        # Create CodeStream at that offset - should work
        cs_at_offset = nvimgcodec.CodeStream(fpath, bitstream_offset=second_ifd_offset)
        assert cs_at_offset.width > 0, "Should successfully parse at valid IFD offset"
        assert cs_at_offset.height > 0, "Should successfully parse at valid IFD offset"
