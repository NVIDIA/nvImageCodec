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
import sys
import pytest as t

from nvidia import nvimgcodec

from utils import *

@t.mark.parametrize(
    "test_case",
    [
        {"input_img_file": "tiff/Alex_2016-01-14_1300Z_(Geotiff).tif",
            "num_images": 1,
            "images": [{
                "num_metadata": 1,
                "metadata": [
                    {"kind": nvimgcodec.MetadataKind.GEO, 
                    "format": nvimgcodec.MetadataFormat.JSON,
                    "expected_buffer": '{"MODEL_PIXEL_SCALE":[0.00259611,0.0022483,0],'
                    '"MODEL_TIEPOINT":[0,0,0,-35.489,39.1935,0],'
                    '"GT_MODEL_TYPE":2,"GT_RASTER_TYPE":1,"GEODETIC_CRS":4326,'
                    '"GEODETIC_CITATION":"WGS 84","GEOG_ANGULAR_UNITS":9102,'
                    '"ELLIPSOID_SEMI_MAJOR_AXIS":6.37814e+06,"ELLIPSOID_INV_FLATTENING":298.257}'
                    }
                ]
            }]
        },
        {"input_img_file": "tiff/JP2K-33003-1.svs", 
            "num_images": 6,
            "images": [
                {
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_APERIO, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": "Aperio Image Library v10.0.50\r\n16000x17597 [0,100 15374x17497] (256x256) J2K/YUV16 Q=70|AppMag = 40|StripeWidth = 1000|ScanScope ID = SS1283|Filename = 6797|Title = univ missouri 07.15.09|Date = 07/16/09|Time = 18:15:06|User = 93d70f65-3b32-4072-ba6a-bd6785a781be|MPP = 0.2498|Left = 39.010742|Top = 14.299895|LineCameraSkew = -0.003035|LineAreaXOffset = 0.000000|LineAreaYOffset = 0.000000|Focus Offset = -0.001000|DSR ID = homer|ImageID = 6797|OriginalWidth = 16000|Originalheight = 17597|Filtered = 3|ICC Profile = ScanScope v1"
                    }],
                },
                {
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_APERIO, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                         "expected_buffer": "Aperio Image Library v10.0.50\n15374x17497 -> 674x768 - |AppMag = 40|StripeWidth = 1000|ScanScope ID = SS1283|Filename = 6797|Title = univ missouri 07.15.09|Date = 07/16/09|Time = 18:15:06|User = 93d70f65-3b32-4072-ba6a-bd6785a781be|MPP = 0.2498|Left = 39.010742|Top = 14.299895|LineCameraSkew = -0.003035|LineAreaXOffset = 0.000000|LineAreaYOffset = 0.000000|Focus Offset = -0.001000|DSR ID = homer|ImageID = 6797|OriginalWidth = 16000|Originalheight = 17597|Filtered = 3|ICC Profile = ScanScope v1"
                    }],
                },
                {
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_APERIO, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": "Aperio Image Library v10.0.50\r\n16000x17597 [0,100 15374x17497] (256x256) -> 3843x4374 J2K/YUV16 Q=70"
                                            }],
                },
                {
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_APERIO, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": "Aperio Image Library v10.0.50\r\n16000x17597 [0,100 15374x17497] (256x256) -> 1921x2187 J2K/YUV16 Q=70"
                
                    }],
                },
                {
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_APERIO, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": "Aperio Image Library v10.0.50\nlabel 415x422"
                    }],
                },
                {
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_APERIO, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": "Aperio Image Library v10.0.50\nmacro 1280x421"
                    }],
                }
            ]
        },
        {"input_img_file": "tiff/Philips-1.tiff", 
            "num_images": 8,
            "images": [
                {
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_PHILIPS, 
                        "format": nvimgcodec.MetadataFormat.XML,
                        "expected_buffer": '<?xml version="1.0" encoding="UTF-8" ?>\n<DataObject ObjectType="DPUfsImport">\n\t<Attribute Name="DICOM_MANUFACTURER" Group="0x0008" Element="0x0070" PMSVR="IString">Hamamatsu</Attribute>\n\t<Attribute Name="PIM_DP_SCANNED_IMAGES" Group="0x301D" Element="0x1003" PMSVR="IDataObjectArray">\n\t\t<Array>\n\t\t\t<DataObject ObjectType="DPScannedImage">\n\t\t\t\t<Attribute Name="PIM_DP_IMAGE_TYPE" Group="0x301D" Element="0x1004" PMSVR="IString">WSI</Attribute>\n\t\t\t\t<Attribute Name="UFS_IMAGE_PIXEL_TRANSFORMATION_METHOD" Group="0x301D" Element="0x2013" PMSVR="IString">0</Attribute>\n\t\t\t\t<Attribute Name="DICOM_BITS_ALLOCATED" Group="0x0028" Element="0x0100" PMSVR="IUInt16">8</Attribute>\n\t\t\t\t<Attribute Name="DICOM_BITS_STORED" Group="0x0028" Element="0x0101" PMSVR="IUInt16">8</Attribute>\n\t\t\t\t<Attribute Name="DICOM_DERIVATION_DESCRIPTION" Group="0x0008" Element="0x2111" PMSVR="IString">tiff-useBigTIFF=1-useRgb=0-levels=10003,10002,10000,10001-processing=0-q80-sourceFilename=&quot;T14-03469_3311940 - 2015-12-09 17.29.29.ndpi&quot;</Attribute>\n\t\t\t\t<Attribute Name="DICOM_HIGH_BIT" Group="0x0028" Element="0x0102" PMSVR="IUInt16">7</Attribute>\n\t\t\t\t<Attribute Name="DICOM_LOSSY_IMAGE_COMPRESSION" Group="0x0028" Element="0x2110" PMSVR="IString">01</Attribute>\n\t\t\t\t<Attribute Name="DICOM_LOSSY_IMAGE_COMPRESSION_METHOD" Group="0x0028" Element="0x2114" PMSVR="IStringArray">&quot;PHILIPS_TIFF_1_0&quot;</Attribute>\n\t\t\t\t<Attribute Name="DICOM_LOSSY_IMAGE_COMPRESSION_RATIO" Group="0x0028" Element="0x2112" PMSVR="IDoubleArray">&quot;3&quot;</Attribute>\n\t\t\t\t<Attribute Name="DICOM_PHOTOMETRIC_INTERPRETATION" Group="0x0028" Element="0x0004" PMSVR="IString">RGB</Attribute>\n\t\t\t\t<Attribute Name="DICOM_PIXEL_REPRESENTATION" Group="0x0028" Element="0x0103" PMSVR="IUInt16">0</Attribute>\n\t\t\t\t<Attribute Name="DICOM_PIXEL_SPACING" Group="0x0028" Element="0x0030" PMSVR="IDoubleArray">&quot;0.000226891&quot; &quot;0.000226907&quot;</Attribute>\n\t\t\t\t<Attribute Name="DICOM_PLANAR_CONFIGURATION" Group="0x0028" Element="0x0006" PMSVR="IUInt16">0</Attribute>\n\t\t\t\t<Attribute Name="DICOM_SAMPLES_PER_PIXEL" Group="0x0028" Element="0x0002" PMSVR="IUInt16">3</Attribute>\n\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE" Group="0x1001" Element="0x8B01" PMSVR="IDataObjectArray">\n\t\t\t\t\t<Array>\n\t\t\t\t\t\t<DataObject ObjectType="PixelDataRepresentation">\n\t\t\t\t\t\t\t<Attribute Name="DICOM_PIXEL_SPACING" Group="0x0028" Element="0x0030" PMSVR="IDoubleArray">&quot;0.000227273&quot; &quot;0.000227273&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_DP_PIXEL_DATA_REPRESENTATION_POSITION" Group="0x101D" Element="0x100B" PMSVR="IDoubleArray">&quot;0&quot; &quot;0&quot; &quot;0&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_COLUMNS" Group="0x2001" Element="0x115E" PMSVR="IUInt32">45056</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_NUMBER" Group="0x1001" Element="0x8B02" PMSVR="IUInt16">0</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_ROWS" Group="0x2001" Element="0x115D" PMSVR="IUInt32">35840</Attribute>\n\t\t\t\t\t\t</DataObject>\n\t\t\t\t\t\t<DataObject ObjectType="PixelDataRepresentation">\n\t\t\t\t\t\t\t<Attribute Name="DICOM_PIXEL_SPACING" Group="0x0028" Element="0x0030" PMSVR="IDoubleArray">&quot;0.000454545&quot; &quot;0.000454545&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_DP_PIXEL_DATA_REPRESENTATION_POSITION" Group="0x101D" Element="0x100B" PMSVR="IDoubleArray">&quot;0&quot; &quot;0&quot; &quot;0&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_COLUMNS" Group="0x2001" Element="0x115E" PMSVR="IUInt32">22528</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_NUMBER" Group="0x1001" Element="0x8B02" PMSVR="IUInt16">1</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_ROWS" Group="0x2001" Element="0x115D" PMSVR="IUInt32">17920</Attribute>\n\t\t\t\t\t\t</DataObject>\n\t\t\t\t\t\t<DataObject ObjectType="PixelDataRepresentation">\n\t\t\t\t\t\t\t<Attribute Name="DICOM_PIXEL_SPACING" Group="0x0028" Element="0x0030" PMSVR="IDoubleArray">&quot;0.000909091&quot; &quot;0.000909091&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_DP_PIXEL_DATA_REPRESENTATION_POSITION" Group="0x101D" Element="0x100B" PMSVR="IDoubleArray">&quot;0&quot; &quot;0&quot; &quot;0&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_COLUMNS" Group="0x2001" Element="0x115E" PMSVR="IUInt32">11264</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_NUMBER" Group="0x1001" Element="0x8B02" PMSVR="IUInt16">2</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_ROWS" Group="0x2001" Element="0x115D" PMSVR="IUInt32">9216</Attribute>\n\t\t\t\t\t\t</DataObject>\n\t\t\t\t\t\t<DataObject ObjectType="PixelDataRepresentation">\n\t\t\t\t\t\t\t<Attribute Name="DICOM_PIXEL_SPACING" Group="0x0028" Element="0x0030" PMSVR="IDoubleArray">&quot;0.00181818&quot; &quot;0.00181818&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_DP_PIXEL_DATA_REPRESENTATION_POSITION" Group="0x101D" Element="0x100B" PMSVR="IDoubleArray">&quot;0&quot; &quot;0&quot; &quot;0&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_COLUMNS" Group="0x2001" Element="0x115E" PMSVR="IUInt32">5632</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_NUMBER" Group="0x1001" Element="0x8B02" PMSVR="IUInt16">3</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_ROWS" Group="0x2001" Element="0x115D" PMSVR="IUInt32">4608</Attribute>\n\t\t\t\t\t\t</DataObject>\n\t\t\t\t\t\t<DataObject ObjectType="PixelDataRepresentation">\n\t\t\t\t\t\t\t<Attribute Name="DICOM_PIXEL_SPACING" Group="0x0028" Element="0x0030" PMSVR="IDoubleArray">&quot;0.00363636&quot; &quot;0.00363636&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_DP_PIXEL_DATA_REPRESENTATION_POSITION" Group="0x101D" Element="0x100B" PMSVR="IDoubleArray">&quot;0&quot; &quot;0&quot; &quot;0&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_COLUMNS" Group="0x2001" Element="0x115E" PMSVR="IUInt32">3072</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_NUMBER" Group="0x1001" Element="0x8B02" PMSVR="IUInt16">4</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_ROWS" Group="0x2001" Element="0x115D" PMSVR="IUInt32">2560</Attribute>\n\t\t\t\t\t\t</DataObject>\n\t\t\t\t\t\t<DataObject ObjectType="PixelDataRepresentation">\n\t\t\t\t\t\t\t<Attribute Name="DICOM_PIXEL_SPACING" Group="0x0028" Element="0x0030" PMSVR="IDoubleArray">&quot;0.00727273&quot; &quot;0.00727273&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_DP_PIXEL_DATA_REPRESENTATION_POSITION" Group="0x101D" Element="0x100B" PMSVR="IDoubleArray">&quot;0&quot; &quot;0&quot; &quot;0&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_COLUMNS" Group="0x2001" Element="0x115E" PMSVR="IUInt32">1536</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_NUMBER" Group="0x1001" Element="0x8B02" PMSVR="IUInt16">5</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_ROWS" Group="0x2001" Element="0x115D" PMSVR="IUInt32">1536</Attribute>\n\t\t\t\t\t\t</DataObject>\n\t\t\t\t\t\t<DataObject ObjectType="PixelDataRepresentation">\n\t\t\t\t\t\t\t<Attribute Name="DICOM_PIXEL_SPACING" Group="0x0028" Element="0x0030" PMSVR="IDoubleArray">&quot;0.0145455&quot; &quot;0.0145455&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_DP_PIXEL_DATA_REPRESENTATION_POSITION" Group="0x101D" Element="0x100B" PMSVR="IDoubleArray">&quot;0&quot; &quot;0&quot; &quot;0&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_COLUMNS" Group="0x2001" Element="0x115E" PMSVR="IUInt32">1024</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_NUMBER" Group="0x1001" Element="0x8B02" PMSVR="IUInt16">6</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_ROWS" Group="0x2001" Element="0x115D" PMSVR="IUInt32">1024</Attribute>\n\t\t\t\t\t\t</DataObject>\n\t\t\t\t\t\t<DataObject ObjectType="PixelDataRepresentation">\n\t\t\t\t\t\t\t<Attribute Name="DICOM_PIXEL_SPACING" Group="0x0028" Element="0x0030" PMSVR="IDoubleArray">&quot;0.0290909&quot; &quot;0.0290909&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_DP_PIXEL_DATA_REPRESENTATION_POSITION" Group="0x101D" Element="0x100B" PMSVR="IDoubleArray">&quot;0&quot; &quot;0&quot; &quot;0&quot;</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_COLUMNS" Group="0x2001" Element="0x115E" PMSVR="IUInt32">512</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_NUMBER" Group="0x1001" Element="0x8B02" PMSVR="IUInt16">7</Attribute>\n\t\t\t\t\t\t\t<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_ROWS" Group="0x2001" Element="0x115D" PMSVR="IUInt32">512</Attribute>\n\t\t\t\t\t\t</DataObject>\n\t\t\t\t\t</Array>\n\t\t\t\t</Attribute>\n\t\t\t\t<Attribute Name="PIM_DP_IMAGE_COLUMNS" Group="0x301D" Element="0x1007" PMSVR="IUInt32">45056</Attribute>\n\t\t\t\t<Attribute Name="PIM_DP_IMAGE_ROWS" Group="0x301D" Element="0x1006" PMSVR="IUInt32">35840</Attribute>\n\t\t\t\t<Attribute Name="PIM_DP_SOURCE_FILE" Group="0x301D" Element="0x1000" PMSVR="IString">%FILENAME%</Attribute>\n\t\t\t</DataObject>\n\t\t</Array>\n\t</Attribute>\n\t<Attribute Name="PIM_DP_UFS_BARCODE" Group="0x301D" Element="0x1002" PMSVR="IString">MzMxMTk0MA==</Attribute>\n\t<Attribute Name="PIM_DP_UFS_INTERFACE_VERSION" Group="0x301D" Element="0x1001" PMSVR="IString">3.0</Attribute>\n\t<Attribute Name="DICOM_SOFTWARE_VERSIONS" Group="0x0018" Element="0x1020" PMSVR="IStringArray">&quot;4.0.3&quot;</Attribute>\n</DataObject>\n'
                    }]
                },
                {
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_PHILIPS, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": 'level=1 mag=22 quality=80'
                    }]
                },
                {
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_PHILIPS, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": 'level=2 mag=11 quality=80'
                    }]
                },{
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_PHILIPS, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": 'level=3 mag=5.5 quality=80'
                    }]
                },{
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_PHILIPS, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": 'level=4 mag=2.75 quality=80'
                    }]
                },{
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_PHILIPS, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": 'level=5 mag=1.375 quality=80'
                    }]
                },{
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_PHILIPS, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": 'level=6 mag=0.6875 quality=80'
                    }]
                },{
                    "num_metadata": 1,
                    "metadata": [{
                        "kind": nvimgcodec.MetadataKind.MED_PHILIPS, 
                        "format": nvimgcodec.MetadataFormat.RAW,
                        "expected_buffer": 'level=7 mag=0.34375 quality=80'
                    }]
                },
                
            ]
        },
    ]
)
def test_metadata_read_from_tiff(test_case):
    input_img_path = os.path.join(img_dir_path, test_case["input_img_file"])
    cs = nvimgcodec.CodeStream(input_img_path)
    assert cs.num_images == test_case["num_images"]
    decoder = nvimgcodec.Decoder()
    for i in range(cs.num_images):
        sub_cs = cs.getSubCodeStream(i)
        metadata = decoder.getMetadata(sub_cs)
        assert len(metadata) == test_case["images"][i]["num_metadata"]
        for j in range(len(metadata)):
            assert metadata[j].kind == test_case["images"][i]["metadata"][j]["kind"]
            assert metadata[j].format == test_case["images"][i]["metadata"][j]["format"]
            assert metadata[j].buffer.decode('utf-8') == test_case["images"][i]["metadata"][j]["expected_buffer"]

@t.mark.parametrize("file_name", [
    "tiff/multi_page.tif", 
    "tiff/cat-300572_640.tiff"])
def test_metadata_read_from_code_stream_without_metadata_return_empty_list(file_name):
    input_img_path = os.path.join(img_dir_path, file_name)
    cs = nvimgcodec.CodeStream(input_img_path)
    decoder = nvimgcodec.Decoder()
    metadata = decoder.getMetadata(cs)
    assert len(metadata) == 0

@t.mark.parametrize("file_name", [
    "jpeg/padlock-406986_640_410.jpg",
    "jpeg2k/cat-111793_640.jp2",
    "bmp/cat-111793_640.bmp",
    "pnm/cat-1245673_640.pgm",
    "webp/lossy/cat-3113513_640.webp",
    "png/with_alpha_16bit/4ch16bpp.png",
    ])
def test_metadata_read_for_codec_with_unsupported_metadata_throws(file_name):
    input_img_path = os.path.join(img_dir_path, file_name)
    cs = nvimgcodec.CodeStream(input_img_path)
    decoder = nvimgcodec.Decoder()
    
    with t.raises(Exception) as excinfo:
        decoder.getMetadata(cs)
    
    assert (isinstance(excinfo.value, RuntimeError) )
    assert (str(excinfo.value) == "nvImageCodec failure: '#9'")
        