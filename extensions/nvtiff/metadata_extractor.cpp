/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "metadata_extractor.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <algorithm>
#include <exception>
#include <utility>
#include "error_handling.h"
#include "log.h"


#ifdef NDEBUG
#define ENABLE_TEST_METADATA_EXTRACTOR 0
#else
#define ENABLE_TEST_METADATA_EXTRACTOR 0 // set to 1 for testing
#endif


namespace {
    template<typename T>
    void get_key_numerical_value(nvtiffStream_t* nvtiff_stream, nvtiffGeoKey_t key, uint32_t index, uint32_t count, std::stringstream& ss) 
    {
        if (count > 1) {
            ss << "[";
        }
        for (uint32_t j = 0; j < count; j++) {
            if (j > 0) {
                ss << ",";
            }
            T value;
            if constexpr (std::is_same<T, unsigned short>::value) {
                XM_CHECK_NVTIFF(nvtiffStreamGetGeoKeySHORT(*nvtiff_stream, key, &value, index, 1));
            } else if constexpr (std::is_same<T, double>::value) {
                XM_CHECK_NVTIFF(nvtiffStreamGetGeoKeyDOUBLE(*nvtiff_stream, key, &value, index, 1));
            }
            ss << value;
        }
        if (count > 1) {
            ss << "]";
        }
    }
}
namespace nvtiff {


class BaseMetadataExtractor : public MetadataExtractor::IMetadataSetExtractor
{
  public:
    BaseMetadataExtractor(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, nvtiffStream_t* nvtiff_stream)
        : framework_(framework)
        , plugin_id_(plugin_id)
        , nvtiff_stream_(nvtiff_stream)
        , current_image_idx_(0)
        , cache_valid_(false)
    {}
    ~BaseMetadataExtractor() = default;
    bool extract(const nvimgcodecCodeStreamDesc_t* code_stream) override = 0;
    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t* metadata, int index) override
    {
        try {
            // Get current code stream info
            nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
            nvimgcodecStatus_t ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not get code stream info.");
                return NVIMGCODEC_STATUS_INTERNAL_ERROR;
            }
            
            uint32_t image_idx = codestream_info.code_stream_view ? codestream_info.code_stream_view->image_idx : 0;
            
            // Check if image_idx has changed since last extraction
            if (!cache_valid_ || current_image_idx_ != image_idx) {
                // Re-extract metadata for the new image_idx
                if (!extract(code_stream)) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not extract metadata.");
                    return NVIMGCODEC_STATUS_INTERNAL_ERROR;
                }
                current_image_idx_ = image_idx;
                cache_valid_ = true;
            }
            
            metadata->kind = NVIMGCODEC_METADATA_KIND_UNKNOWN;
            metadata->format = NVIMGCODEC_METADATA_FORMAT_UNKNOWN;
            if (!metadata->buffer) {
                if (metadata->buffer_size == 0) {
                    metadata->buffer_size = metadata_str.size();
                } else {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Invalid parameter. It is not possible to set buffer_size to 0.");
                    return NVIMGCODEC_STATUS_INVALID_PARAMETER;
                }
            } else {
                if (metadata->buffer_size < metadata_str.size()) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Invalid parameter. buffer_size is less than the size of the metadata.");
                    return NVIMGCODEC_STATUS_INVALID_PARAMETER;
                }
                memcpy(metadata->buffer, metadata_str.c_str(), metadata_str.size());
            }
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Error in BaseMetadataExtractor::getMetadata: " << e.what());
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  protected:
    const nvimgcodecFrameworkDesc_t* framework_;
    const char* plugin_id_;
    nvtiffStream_t* nvtiff_stream_;
    std::string metadata_str;
    uint32_t current_image_idx_;
    bool cache_valid_;
};
#if ENABLE_TEST_METADATA_EXTRACTOR
class TestMetadataExtractor : public BaseMetadataExtractor
{
  public:
    TestMetadataExtractor(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, nvtiffStream_t* nvtiff_stream)
        : BaseMetadataExtractor(framework, plugin_id, nvtiff_stream)
    {}
    ~TestMetadataExtractor() = default;

    bool extract(const nvimgcodecCodeStreamDesc_t* code_stream) override
    {
        metadata_str = metadata_test_;
        return true;
    }

  private:
    const std::string metadata_test_ = "tes_key_0:test_value_0|tes_key_1:test_value_1|tes_key_2:test_value_2";
};
#endif
class GeoMetadataExtractor : public BaseMetadataExtractor
{
  public:
    GeoMetadataExtractor(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, nvtiffStream_t* nvtiff_stream)
        : BaseMetadataExtractor(framework, plugin_id, nvtiff_stream)
    {}
    ~GeoMetadataExtractor() = default;
    bool extract(const nvimgcodecCodeStreamDesc_t* code_stream) override;
    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t* metadata, int index) override
    {
        auto ret = BaseMetadataExtractor::getMetadata(code_stream, metadata, index);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not get metadata.");
            return ret;
        }
        metadata->kind = NVIMGCODEC_METADATA_KIND_GEO;
        metadata->format = NVIMGCODEC_METADATA_FORMAT_JSON;
        return ret;
    }
};

class AperioMetadataExtractor : public BaseMetadataExtractor
{
  public:
    AperioMetadataExtractor(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, nvtiffStream_t* nvtiff_stream)
        : BaseMetadataExtractor(framework, plugin_id, nvtiff_stream)
    {}
    ~AperioMetadataExtractor() = default;
    bool extract(const nvimgcodecCodeStreamDesc_t* code_stream) override;
    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t* metadata, int index) override
    {
        auto ret = BaseMetadataExtractor::getMetadata(code_stream, metadata, index);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not get metadata.");
            return ret;
        }
        metadata->kind = NVIMGCODEC_METADATA_KIND_MED_APERIO;
        metadata->format = NVIMGCODEC_METADATA_FORMAT_RAW;
        return ret;
    }
};

class PhilipsMetadataExtractor : public BaseMetadataExtractor
{
  public:
    PhilipsMetadataExtractor(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, nvtiffStream_t* nvtiff_stream)
        : BaseMetadataExtractor(framework, plugin_id, nvtiff_stream)
    {}
    ~PhilipsMetadataExtractor() = default;
    bool extract(const nvimgcodecCodeStreamDesc_t* code_stream) override;
    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t* metadata, int index) override
    {
        auto ret = BaseMetadataExtractor::getMetadata(code_stream, metadata, index);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not get metadata.");
            return ret;
        }
        metadata->kind = NVIMGCODEC_METADATA_KIND_MED_PHILIPS;
        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not get code stream info.");
            return ret;
        }
        
        uint32_t image_id = codestream_info.code_stream_view ? codestream_info.code_stream_view->image_idx : 0;
        if (image_id == 0) {
            metadata->format = NVIMGCODEC_METADATA_FORMAT_XML;
        } else {
            metadata->format = NVIMGCODEC_METADATA_FORMAT_RAW;
        }
        return ret;
    }
};
// clang-format off


enum class TiffTag : uint16_t {
    ImageDescription = 270, // info about image
    Software = 305 // name & release
};

static std::map<nvtiffGeoKey_t, std::string> geokey2string = {
    { NVTIFF_GEOKEY_GT_MODEL_TYPE                , "GT_MODEL_TYPE"},
    { NVTIFF_GEOKEY_GT_RASTER_TYPE               , "GT_RASTER_TYPE"},
    { NVTIFF_GEOKEY_GT_CITATION                  , "GT_CITATION"},
    { NVTIFF_GEOKEY_GEODETIC_CRS                 , "GEODETIC_CRS"},
    { NVTIFF_GEOKEY_GEODETIC_CITATION            , "GEODETIC_CITATION"},
    { NVTIFF_GEOKEY_GEODETIC_DATUM               , "GEODETIC_DATUM"},
    { NVTIFF_GEOKEY_PRIME_MERIDIAN               , "PRIME_MERIDIAN"},
    { NVTIFF_GEOKEY_GEOG_LINEAR_UNITS            , "GEOG_LINEAR_UNITS"},
    { NVTIFF_GEOKEY_GEOG_LINEAR_UNIT_SIZE        , "GEOG_LINEAR_UNIT_SIZE"},
    { NVTIFF_GEOKEY_GEOG_ANGULAR_UNITS           , "GEOG_ANGULAR_UNITS"},
    { NVTIFF_GEOKEY_GEOG_ANGULAR_UNIT_SIZE       , "GEOG_ANGULAR_UNIT_SIZE"},
    { NVTIFF_GEOKEY_ELLIPSOID                    , "ELLIPSOID"},
    { NVTIFF_GEOKEY_ELLIPSOID_SEMI_MAJOR_AXIS    , "ELLIPSOID_SEMI_MAJOR_AXIS"},
    { NVTIFF_GEOKEY_ELLIPSOID_SEMI_MINOR_AXIS    , "ELLIPSOID_SEMI_MINOR_AXIS"},
    { NVTIFF_GEOKEY_ELLIPSOID_INV_FLATTENING     , "ELLIPSOID_INV_FLATTENING"},
    { NVTIFF_GEOKEY_GEOG_AZIMUTH_UNITS           , "GEOG_AZIMUTH_UNITS"},
    { NVTIFF_GEOKEY_PRIME_MERIDIAN_LONG          , "PRIME_MERIDIAN_LONG"},
    { NVTIFF_GEOKEY_PROJECTED_CRS                , "PROJECTED_CRS"},
    { NVTIFF_GEOKEY_PROJECTED_CITATION           , "PROJECTED_CITATION"},
    { NVTIFF_GEOKEY_PROJECTION                   , "PROJECTION"},
    { NVTIFF_GEOKEY_PROJ_METHOD                  , "PROJ_METHOD"},
    { NVTIFF_GEOKEY_PROJ_LINEAR_UNITS            , "PROJ_LINEAR_UNITS"},
    { NVTIFF_GEOKEY_PROJ_LINEAR_UNIT_SIZE        , "PROJ_LINEAR_UNIT_SIZE"},
    { NVTIFF_GEOKEY_PROJ_STD_PARALLEL1           , "PROJ_STD_PARALLEL1"},
    { NVTIFF_GEOKEY_PROJ_STD_PARALLEL            , "PROJ_STD_PARALLEL"},
    { NVTIFF_GEOKEY_PROJ_STD_PARALLEL2           , "PROJ_STD_PARALLEL2"},
    { NVTIFF_GEOKEY_PROJ_NAT_ORIGIN_LONG         , "PROJ_NAT_ORIGIN_LONG"},
    { NVTIFF_GEOKEY_PROJ_ORIGIN_LONG             , "PROJ_ORIGIN_LONG"},
    { NVTIFF_GEOKEY_PROJ_NAT_ORIGIN_LAT          , "PROJ_NAT_ORIGIN_LAT"},
    { NVTIFF_GEOKEY_PROJ_ORIGIN_LAT              , "PROJ_ORIGIN_LAT"},
    { NVTIFF_GEOKEY_PROJ_FALSE_EASTING           , "PROJ_FALSE_EASTING"},
    { NVTIFF_GEOKEY_PROJ_FALSE_NORTHING          , "PROJ_FALSE_NORTHING"},
    { NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_LONG       , "PROJ_FALSE_ORIGIN_LONG"},
    { NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_LAT        , "PROJ_FALSE_ORIGIN_LAT"},
    { NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_EASTING    , "PROJ_FALSE_ORIGIN_EASTING"},
    { NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_NORTHING   , "PROJ_FALSE_ORIGIN_NORTHING"},
    { NVTIFF_GEOKEY_PROJ_CENTER_LONG             , "PROJ_CENTER_LONG"},
    { NVTIFF_GEOKEY_PROJ_CENTER_LAT              , "PROJ_CENTER_LAT"},
    { NVTIFF_GEOKEY_PROJ_CENTER_EASTING          , "PROJ_CENTER_EASTING"},
    { NVTIFF_GEOKEY_PROJ_CENTER_NORTHING         , "PROJ_CENTER_NORTHING"},
    { NVTIFF_GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN     , "PROJ_SCALE_AT_NAT_ORIGIN"},
    { NVTIFF_GEOKEY_PROJ_SCALE_AT_ORIGIN         , "PROJ_SCALE_AT_ORIGIN"},
    { NVTIFF_GEOKEY_PROJ_SCALE_AT_CENTER         , "PROJ_SCALE_AT_CENTER"},
    { NVTIFF_GEOKEY_PROJ_AZIMUTH_ANGLE           , "PROJ_AZIMUTH_ANGLE"},
    { NVTIFF_GEOKEY_PROJ_STRAIGHT_VERT_POLE_LONG , "PROJ_STRAIGHT_VERT_POLE_LONG"},
    { NVTIFF_GEOKEY_VERTICAL                     , "VERTICAL" },
    { NVTIFF_GEOKEY_VERTICAL_CITATION            , "VERTICAL_CITATION" },
    { NVTIFF_GEOKEY_VERTICAL_DATUM               , "VERTICAL_DATUM" },
    { NVTIFF_GEOKEY_VERTICAL_UNITS               , "VERTICAL_UNITS" },
    { NVTIFF_GEOKEY_BASE                         , "BASE" },
    { NVTIFF_GEOKEY_END                          , "END" }};

static std::map<nvtiffTag_t, std::string> tag2string = {
    {NVTIFF_TAG_MODEL_PIXEL_SCALE, "MODEL_PIXEL_SCALE"},
    {NVTIFF_TAG_MODEL_TIE_POINT, "MODEL_TIEPOINT"},
    {NVTIFF_TAG_MODEL_TRANSFORMATION, "MODEL_TRANSFORMATION"}
};
// clang-format on
bool GeoMetadataExtractor::extract(const nvimgcodecCodeStreamDesc_t* code_stream)
{
    std::stringstream ss;
    metadata_str.clear();
    bool has_data = false;
    try {
        ss << "{";

        constexpr std::array<nvtiffTag_t, 3> geo_tags = {
            NVTIFF_TAG_MODEL_PIXEL_SCALE, NVTIFF_TAG_MODEL_TIE_POINT, NVTIFF_TAG_MODEL_TRANSFORMATION};
        for (auto& tag : geo_tags) {
            nvtiffTagDataType_t tag_type;
            uint32_t count = 0, size = 0;
            nvtiffStatus_t status = nvtiffStreamGetTagInfo(*nvtiff_stream_, 0, tag, &tag_type, &size, &count);
            if (status == NVTIFF_STATUS_SUCCESS) {
                if (tag_type != NVTIFF_TAG_TYPE_DOUBLE) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Tiff tag type is not supported by the sample. Skipping metadata extraction.");
                    continue;
                }
                if (has_data) {
                    ss << ",";
                }
                std::vector<double> values(count);
                XM_CHECK_NVTIFF(nvtiffStreamGetTagValue(*nvtiff_stream_, 0, tag, (void*)values.data(), count));
                ss << "\"" << tag2string[tag] << "\":";
                if (count > 1) {
                    ss << "[";
                }
                for (size_t i = 0; i < values.size(); ++i) {
                    if (i > 0) {
                        ss << ",";
                    }
                    ss << values[i];
                }
                if (count > 1) {
                    ss << "]";
                }
                has_data = true;
            } 
        }

        uint32_t num_keys = 0;
        nvtiffStatus_t status = nvtiffStreamGetNumberOfGeoKeys(*nvtiff_stream_, nullptr, &num_keys);
        if (status == NVTIFF_STATUS_SUCCESS) {
            std::vector<nvtiffGeoKey_t> keys(num_keys);
            XM_CHECK_NVTIFF(nvtiffStreamGetNumberOfGeoKeys(*nvtiff_stream_, keys.data(), &num_keys));
            for (uint32_t i = 0; i < num_keys; i++) {
                uint32_t size = 0;
                uint32_t count = 0;
                nvtiffGeoKeyDataType_t type = NVTIFF_GEOKEY_TYPE_UNKNOWN;
                XM_CHECK_NVTIFF(nvtiffStreamGetGeoKeyInfo(*nvtiff_stream_, keys[i], &size, &count, &type));
                if (has_data) {
                    ss << ",";
                }
                if (geokey2string.find(keys[i]) == geokey2string.end()) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Tiff GeoKey " << keys[i] << " is not supported by the sample. Skipping it.");
                    continue;
                }
                ss << "\"" << geokey2string[keys[i]] << "\":";
                switch (type) {
                case NVTIFF_GEOKEY_TYPE_ASCII: {
                    if (count > 0) {
                        std::string value(count - 1, '\0');
                        XM_CHECK_NVTIFF(nvtiffStreamGetGeoKeyASCII(*nvtiff_stream_, keys[i], value.data(), count));
                        ss << "\"" << value << "\"";
                    }
                    break;
                }
                case NVTIFF_GEOKEY_TYPE_SHORT: {
                    get_key_numerical_value<unsigned short>(nvtiff_stream_, keys[i], 0, count, ss);
                    break;
                }
                case NVTIFF_GEOKEY_TYPE_DOUBLE: {
                    get_key_numerical_value<double>(nvtiff_stream_, keys[i], 0, count, ss);
                    break;
                }
                default:{
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Tiff GeoKey :" << keys[i] << "  type :" << type << " is not supported by the sample. Skipping it.");
                    continue;
                }
                }
                has_data = true;
            }
        }
        ss << "}";
        metadata_str = ss.str();
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Error in GeoMetadataExtractor::extract: " << e.what());
        return false;
    }
    return has_data;
}

bool AperioMetadataExtractor::extract(const nvimgcodecCodeStreamDesc_t* code_stream)
{
    std::stringstream ss;
    metadata_str.clear();
    try {
        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        nvimgcodecStatus_t ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not get code stream info.");
            return false;
        }
        
        uint32_t image_id = codestream_info.code_stream_view ? codestream_info.code_stream_view->image_idx : 0;
        
        nvtiffTagDataType_t tag_type;
        uint32_t count = 0, size = 0;
        nvtiffStatus_t status = nvtiffStreamGetTagInfoGeneric(*nvtiff_stream_, image_id, static_cast<uint16_t>(TiffTag::ImageDescription), &tag_type, &size, &count);
        if (status == NVTIFF_STATUS_SUCCESS) {
            if (tag_type != NVTIFF_TAG_TYPE_ASCII) {
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Tiff ImageDescription tag type is not ASCII as expected. Skipping metadata extraction.");
                return false;
            }
            std::string value(count - 1, '\0');
            XM_CHECK_NVTIFF(nvtiffStreamGetTagValueGeneric(*nvtiff_stream_, image_id, static_cast<uint16_t>(TiffTag::ImageDescription), (void*)value.data(), count));
            std::string value_lower_case(value);
            std::transform(value_lower_case.begin(), value_lower_case.end(), value_lower_case.begin(), ::tolower);
            if (value_lower_case.find("aperio") != std::string::npos) {
                ss << value;
            }
        }
        metadata_str = ss.str();
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Error in AperioMetadataExtractor::extract: " << e.what());
        return false;
    }
    bool has_data = metadata_str.size() > 0;
    return has_data;
}

bool PhilipsMetadataExtractor::extract(const nvimgcodecCodeStreamDesc_t* code_stream)
{
    std::stringstream ss;
    metadata_str.clear();
    try {
        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        nvimgcodecStatus_t ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not get code stream info.");
            return false;
        }
        
        uint32_t image_id = codestream_info.code_stream_view ? codestream_info.code_stream_view->image_idx : 0;
        
        nvtiffTagDataType_t tag_type;
        uint32_t count = 0, size = 0;
        nvtiffStatus_t status = nvtiffStreamGetTagInfoGeneric(*nvtiff_stream_, image_id, static_cast<uint16_t>(TiffTag::Software), &tag_type, &size, &count);
        if (status == NVTIFF_STATUS_SUCCESS) {

            if (tag_type != NVTIFF_TAG_TYPE_ASCII) {
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Tiff Software tag type is not ASCII as expected. Skipping metadata extraction.");
                return false;
            }
            std::string value(count - 1, '\0');
            XM_CHECK_NVTIFF(nvtiffStreamGetTagValueGeneric(*nvtiff_stream_, image_id, static_cast<uint16_t>(TiffTag::Software), (void*)value.data(), count));
            std::string value_lower_case(value);
            std::transform(value_lower_case.begin(), value_lower_case.end(), value_lower_case.begin(), ::tolower);
            if (value_lower_case.find("philips") != std::string::npos) {
                status = nvtiffStreamGetTagInfoGeneric(*nvtiff_stream_, image_id, static_cast<uint16_t>(TiffTag::ImageDescription), &tag_type, &size, &count);
                if (status != NVTIFF_STATUS_SUCCESS || tag_type != NVTIFF_TAG_TYPE_ASCII) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Tiff ImageDescription tag type is not ASCII as expected. Skipping metadata extraction.");
                    return false;
                }
                std::string value(count - 1, '\0');
                XM_CHECK_NVTIFF(nvtiffStreamGetTagValueGeneric(*nvtiff_stream_, image_id, static_cast<uint16_t>(TiffTag::ImageDescription), (void*)value.data(), count));
                ss << value;
            } else {
                return false;
            }
        }
        metadata_str = ss.str();
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Error in PhilipsMetadataExtractor::extract: " << e.what());
        return false;
    }
    bool has_data = metadata_str.size() > 0;
    return has_data;
}

MetadataExtractor::MetadataExtractor(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, nvtiffStream_t* nvtiff_stream)
    : framework_(framework)
    , plugin_id_(plugin_id)
    , nvtiff_stream_(nvtiff_stream)
{
#if ENABLE_TEST_METADATA_EXTRACTOR
    metadata_set_extractors_.emplace_back(std::make_unique<TestMetadataExtractor>(framework_, plugin_id_, nvtiff_stream_));
#endif
    metadata_set_extractors_.emplace_back(std::make_unique<GeoMetadataExtractor>(framework_, plugin_id_, nvtiff_stream_));
    metadata_set_extractors_.emplace_back(std::make_unique<AperioMetadataExtractor>(framework_, plugin_id_, nvtiff_stream_));
    metadata_set_extractors_.emplace_back(std::make_unique<PhilipsMetadataExtractor>(framework_, plugin_id_, nvtiff_stream_));
}

MetadataExtractor::MetadataExtractor(MetadataExtractor&& other) noexcept
    : framework_(other.framework_)
    , plugin_id_(other.plugin_id_)
    , nvtiff_stream_(other.nvtiff_stream_)
    , active_extractors_(std::move(other.active_extractors_))
{
    metadata_set_extractors_ = std::move(other.metadata_set_extractors_);
    active_extractors_ = std::move(other.active_extractors_);
    other.framework_ = nullptr;
    other.plugin_id_ = nullptr;
    other.nvtiff_stream_ = nullptr;
}

MetadataExtractor& MetadataExtractor::operator=(MetadataExtractor&& other) noexcept
{
    if (this != &other) {
        framework_ = other.framework_;
        plugin_id_ = other.plugin_id_;
        nvtiff_stream_ = other.nvtiff_stream_;
        metadata_set_extractors_ = std::move(other.metadata_set_extractors_);
        active_extractors_ = std::move(other.active_extractors_);
        other.framework_ = nullptr;
        other.plugin_id_ = nullptr;
        other.nvtiff_stream_ = nullptr;
    }
    return *this;
}

size_t MetadataExtractor::getMetadataCount(const nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        // Stream should already be parsed by the decoder's ensureStreamParsed function
        // We just need to extract metadata from all extractors
        active_extractors_.clear();
        for (auto& extractor : metadata_set_extractors_) {
            if (extractor->extract(code_stream)) {
                active_extractors_.push_back(extractor.get());
            }
        }
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Error in getMetadataCount: " << e.what());
        return 0;
    }

    return active_extractors_.size();
}

nvimgcodecStatus_t MetadataExtractor::getMetadata(
    const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t* metadata, int index)
{
    if (index < 0 || static_cast<size_t>(index) >= active_extractors_.size()) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return active_extractors_[index]->getMetadata(code_stream, metadata, index);
}

} // namespace nvtiff