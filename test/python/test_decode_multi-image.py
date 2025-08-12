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


from __future__ import annotations
import os
import numpy as np
from nvidia import nvimgcodec
import pytest as t
from utils import img_dir_path
import nvjpeg_test_speedup
from utils import *

multi_page_tiff_sample={
        'multipage_file':"tiff/multi_page.tif",  # this multi page tiff consists of all files which are listed in next tuple and are used as references
        'included_pages' : ("bmp/cat-111793_640.bmp", 
         "bmp/cat-111793_640_grayscale.bmp", 
         "jpeg/padlock-406986_640_444.jpg", 
         "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",
         "tiff/cat-300572_640_grayscale.tiff",
         "jpeg2k/cat-1046544_640-16bit.jp2",
         "png/with_alpha_16bit/4ch16bpp.png")}


@t.mark.parametrize("file_sample", [multi_page_tiff_sample])
@t.mark.parametrize("indices", [
    (0, 1, 2, 3, 4, 5, 6),
    (0, 2, 4, 6,),
    (1, 3, 5,),
    (0, ),
    (6,)
])
def test_decode_selected_images(file_sample, indices):
    decoder = nvimgcodec.Decoder(options=":fancy_upsampling=1")
    
    fpath = os.path.join(img_dir_path, file_sample['multipage_file'])

    ref_imgs = [get_opencv_reference(os.path.join(img_dir_path, file_sample['included_pages'][i])) for i in indices]
    
    cs = nvimgcodec.CodeStream(fpath)
    assert(cs.num_images == len(file_sample['included_pages']))
    
    for i in range(len(indices)):
        scs = cs.getSubCodeStream(indices[i])
        img = decoder.decode(scs)
        np.testing.assert_allclose(ref_imgs[i], img.cpu(), atol=4)

@t.mark.parametrize("file_sample", [multi_page_tiff_sample])
@t.mark.parametrize("indices", [
    (0, 1, 2, 3, 4, 5, 6),
    (0, 2, 4, 6,),
    (1, 3, 5,),
    (0, ),
    (6,)
])
def test_decode_selected_images_in_batch(file_sample, indices):
    decoder = nvimgcodec.Decoder(options=":fancy_upsampling=1")
    
    fpath = os.path.join(img_dir_path, file_sample['multipage_file'])

    ref_imgs = [get_opencv_reference(os.path.join(img_dir_path, file_sample['included_pages'][i])) for i in indices]

    cs = nvimgcodec.CodeStream(fpath)
    assert(cs.num_images == len(file_sample['included_pages']))
    
    scs = []
    for i in range(len(indices)):
        scs.append(cs.getSubCodeStream(indices[i]))
        
    imgs = decoder.decode(scs)
            
    for i in range(len(indices)):
        np.testing.assert_allclose(ref_imgs[i], imgs[i].cpu(), atol=4)

@t.mark.parametrize("file_sample", [multi_page_tiff_sample])
@t.mark.parametrize("image_id_and_roi", [
    (0, (10, 20, 100, 200)),
    (1, (20, 30, 150, 250)), 
    (2, (30, 40, 200, 300)),
    (3, (40, 50, 60, 80)),
    (4, (50, 60, 300, 400)),
    (5, (60, 70, 350, 450)),
    (6, (70, 80, 90, 100)),
])
def test_decode_selected_image_with_roi(file_sample, image_id_and_roi):
    decoder = nvimgcodec.Decoder(options=":fancy_upsampling=1")
    fpath = os.path.join(img_dir_path, file_sample['multipage_file'])
    code_stream = nvimgcodec.CodeStream(fpath)

    roi = nvimgcodec.Region(image_id_and_roi[1][0], image_id_and_roi[1][1], image_id_and_roi[1][2], image_id_and_roi[1][3])

    ref_img = get_opencv_reference(os.path.join(img_dir_path, file_sample['included_pages'][image_id_and_roi[0]]))
    ref_img_np = np.array(ref_img)[roi.start[0]:roi.end[0], roi.start[1]:roi.end[1]]

    scs = code_stream.getSubCodeStream(image_id_and_roi[0], region = roi)

    img_roi = decoder.decode(scs).cpu()

    np.testing.assert_allclose(img_roi, ref_img_np, atol=4)

@t.mark.parametrize("file_sample", [multi_page_tiff_sample])
@t.mark.parametrize("image_id_and_roi_batch", [
    (
        (0, (10, 20, 100, 200)),
        (1, (20, 30, 150, 250)), 
        (2, (30, 40, 200, 300)),
        (3, (40, 50, 60, 80)),
    ),
    (
        (4, (50, 60, 300, 400)),
        (5, (60, 70, 350, 450)),
        (6, (70, 80, 90, 100)),
    )
])
def test_decode_batch_roi(file_sample, image_id_and_roi_batch):
    decoder = nvimgcodec.Decoder(options=":fancy_upsampling=1")
    fpath = os.path.join(img_dir_path, file_sample['multipage_file'])
    code_stream = nvimgcodec.CodeStream(fpath)

    scs = []
    ref_imgs_np = []
    for image_id_and_roi in image_id_and_roi_batch:
        roi = nvimgcodec.Region(image_id_and_roi[1][0], image_id_and_roi[1][1], image_id_and_roi[1][2], image_id_and_roi[1][3])

        ref_img = get_opencv_reference(os.path.join(img_dir_path, file_sample['included_pages'][image_id_and_roi[0]]))
        ref_imgs_np.append(np.array(ref_img)[roi.start[0]:roi.end[0], roi.start[1]:roi.end[1]])

        scs.append(code_stream.getSubCodeStream(image_id_and_roi[0], region = roi))

    imgs_roi = decoder.decode(scs)
    for ref_img, test_img,  in zip(ref_imgs_np, imgs_roi):
        np.testing.assert_allclose(ref_img, test_img.cpu(), atol=4)

@t.mark.parametrize("file_sample", [
    "tiff/multi_page.tif", 
])
def test_decode_image_out_of_range(file_sample):
    fpath = os.path.join(img_dir_path, file_sample)

    cs = nvimgcodec.CodeStream(fpath)

    with t.raises(Exception) as excinfo:
        scs = cs.getSubCodeStream(cs.num_images)
    assert (isinstance(excinfo.value, RuntimeError) )
    assert (str(excinfo.value) == f"Image index #{cs.num_images} out of range (0, {cs.num_images - 1})")
    
    with t.raises(Exception) as excinfo:
        scs = cs.getSubCodeStream(-1)
    assert (isinstance(excinfo.value, TypeError) )
    error_msg = "getSubCodeStream(): incompatible function arguments"
    assert (str(excinfo.value)[:len(error_msg)] == error_msg)

@t.mark.parametrize("file_sample", [
    "tiff/multi_page.tif"
])
def test_decode_sub_code_stream_num_images_always_one(file_sample):
    fpath = os.path.join(img_dir_path, file_sample)
    code_stream = nvimgcodec.CodeStream(fpath)

    for i in range(code_stream.num_images):
        scs = code_stream.getSubCodeStream(i)
        assert(scs.num_images == 1)
        
