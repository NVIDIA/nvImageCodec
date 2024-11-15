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

#include <nvimgcodec.h>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "code_stream.h"
#include "codec_registry.h"
#include "imgproc/exception.h"
#include "file_ext_codec.h"
#include "icodec.h"
#include "iimage_decoder.h"
#include "iimage_encoder.h"
#include "image.h"
#include "image_generic_decoder.h"
#include "image_generic_encoder.h"
#include "log.h"
#include "nvimgcodec_director.h"
#include "nvimgcodec_type_utils.h"
#include "plugin_framework.h"
#include "processing_results.h"

namespace fs = std::filesystem;

using namespace nvimgcodec;

__inline__ nvimgcodecStatus_t getCAPICode(Status status)
{
    nvimgcodecStatus_t code = NVIMGCODEC_STATUS_SUCCESS;
    switch (status) {
    case STATUS_OK:
        code = NVIMGCODEC_STATUS_SUCCESS;
        break;
    case NOT_VALID_FORMAT_STATUS:
    case PARSE_STATUS:
    case BAD_FORMAT_STATUS:
        code = NVIMGCODEC_STATUS_BAD_CODESTREAM;
        break;
    case UNSUPPORTED_FORMAT_STATUS:
        code = NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
        break;
    case CUDA_CALL_ERROR:
        code = NVIMGCODEC_STATUS_EXECUTION_FAILED;
        break;
    case ALLOCATION_ERROR:
        code = NVIMGCODEC_STATUS_ALLOCATOR_FAILURE;
        break;
    case INTERNAL_ERROR:
        code = NVIMGCODEC_STATUS_INTERNAL_ERROR;
        break;
    case INVALID_PARAMETER:
        code = NVIMGCODEC_STATUS_INVALID_PARAMETER;
        break;
    default:
        code = NVIMGCODEC_STATUS_INTERNAL_ERROR;
        break;
    }
    return code;
}

#ifndef NDEBUG
    #define VERBOSE_ERRORS
#endif

#define NVIMGCODECAPI_TRY try

#ifndef VERBOSE_ERRORS
    #define NVIMGCODECAPI_CATCH(a)                                                   \
        catch (const Exception& e)                                                   \
        {                                                                            \
            a = getCAPICode(e.status());                                             \
        }                                                                            \
        catch (const std::exception& e)                                              \
        {                                                                            \
            NVIMGCODEC_LOG_ERROR(Logger::get_default(), e.what());                   \
            a = NVIMGCODEC_STATUS_INTERNAL_ERROR;                                    \
        }                                                                            \
        catch (...)                                                                  \
        {                                                                            \
            NVIMGCODEC_LOG_ERROR(Logger::get_default(), "Unknown NVIMGCODEC error"); \
            a = NVIMGCODEC_STATUS_INTERNAL_ERROR;                                    \
        }
#else
    #define NVIMGCODECAPI_CATCH(a)                                                                                                  \
        catch (const Exception& e)                                                                                                  \
        {                                                                                                                           \
            NVIMGCODEC_LOG_ERROR(Logger::get_default(),                                                                             \
                "Error status: " << e.status() << " Where: " << e.where() << " Message: " << e.message() << " What: " << e.what()); \
            a = getCAPICode(e.status());                                                                                            \
        }                                                                                                                           \
        catch (const std::exception& e)                                                                                             \
        {                                                                                                                           \
            NVIMGCODEC_LOG_ERROR(Logger::get_default(), e.what());                                                                  \
            a = NVIMGCODEC_STATUS_INTERNAL_ERROR;                                                                                   \
        }                                                                                                                           \
        catch (...)                                                                                                                 \
        {                                                                                                                           \
            NVIMGCODEC_LOG_ERROR(Logger::get_default(), "Unknown NVIMGCODEC error");                                                \
            a = NVIMGCODEC_STATUS_INTERNAL_ERROR;                                                                                   \
        }
#endif

struct nvimgcodecInstance
{
    nvimgcodecInstance(const nvimgcodecInstanceCreateInfo_t* create_info)
        : director_(create_info)
    {
    }
    NvImgCodecDirector director_;
};

struct nvimgcodecFuture
{
    ProcessingResultsPromise::FutureImpl handle_;
};

struct nvimgcodecDecoder
{
    nvimgcodecInstance_t instance_;
    std::unique_ptr<ImageGenericDecoder> image_decoder_;
};

struct nvimgcodecEncoder
{
    nvimgcodecInstance_t instance_;
    std::unique_ptr<ImageGenericEncoder> image_encoder_;
};

struct nvimgcodecDebugMessenger
{
    nvimgcodecInstance_t instance_;
    nvimgcodecDebugMessenger(const nvimgcodecDebugMessengerDesc_t* desc)
        : debug_messenger_(desc)
    {
    }
    DebugMessenger debug_messenger_;
};

struct nvimgcodecExtension
{
    nvimgcodecInstance_t nvimgcodec_instance_;
    nvimgcodecExtension_t extension_ext_handle_;
};

struct nvimgcodecCodeStream
{
    nvimgcodecInstance_t nvimgcodec_instance_;
    std::unique_ptr<CodeStream> code_stream_;
};

struct nvimgcodecImage
{
    nvimgcodecInstance_t nvimgcodec_instance_;
    Image image_;
};

nvimgcodecStatus_t nvimgcodecGetProperties(nvimgcodecProperties_t* properties)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(properties);
            if (properties->struct_type != NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
            properties->version = NVIMGCODEC_VER;
            properties->ext_api_version = NVIMGCODEC_EXT_API_VER;
            properties->cudart_version = CUDART_VERSION;
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecInstanceCreate(nvimgcodecInstance_t* instance, const nvimgcodecInstanceCreateInfo_t* create_info)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    nvimgcodecInstance_t nvimgcodec = nullptr;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(instance);
            CHECK_NULL(create_info);
            nvimgcodec = new nvimgcodecInstance(create_info);
            *instance = nvimgcodec;
        }
    NVIMGCODECAPI_CATCH(ret)

    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        if (nvimgcodec) {
            delete nvimgcodec;
        }
    }

    return ret;
}

nvimgcodecStatus_t nvimgcodecInstanceDestroy(nvimgcodecInstance_t instance)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(instance)
            delete instance;
        }
    NVIMGCODECAPI_CATCH(ret)

    return ret;
}

nvimgcodecStatus_t nvimgcodecExtensionCreate(
    nvimgcodecInstance_t instance, nvimgcodecExtension_t* extension, nvimgcodecExtensionDesc_t* extension_desc)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(extension_desc)
            nvimgcodecExtension_t extension_ext_handle;
            ret = instance->director_.plugin_framework_.registerExtension(&extension_ext_handle, extension_desc);
            if (ret == NVIMGCODEC_STATUS_SUCCESS) {
                *extension = new nvimgcodecExtension();
                (*extension)->nvimgcodec_instance_ = instance;
                (*extension)->extension_ext_handle_ = extension_ext_handle;
            }
        }
    NVIMGCODECAPI_CATCH(ret)

    return ret;
}

nvimgcodecStatus_t nvimgcodecExtensionDestroy(nvimgcodecExtension_t extension)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(extension)

            ret = extension->nvimgcodec_instance_->director_.plugin_framework_.unregisterExtension(extension->extension_ext_handle_);
            delete extension;
        }
    NVIMGCODECAPI_CATCH(ret)

    return ret;
}



static nvimgcodecStatus_t nvimgcodecStreamCreate(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(instance);
            CHECK_NULL(code_stream);
            *code_stream = new nvimgcodecCodeStream();
            (*code_stream)->code_stream_ = instance->director_.createCodeStream();
            (*code_stream)->nvimgcodec_instance_ = instance;
   
        }
    NVIMGCODECAPI_CATCH(ret)

    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        if (*code_stream) {
            delete *code_stream;
        }
    }
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamCreateFromFile(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, const char* file_name)
{
    nvimgcodecStatus_t ret = nvimgcodecStreamCreate(instance, code_stream);

    NVIMGCODECAPI_TRY
        {
            if (ret == NVIMGCODEC_STATUS_SUCCESS) {
                (*code_stream)->code_stream_->parseFromFile(std::string(file_name));
            }
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamCreateFromHostMem(
    nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, const unsigned char* data, size_t size)
{
    nvimgcodecStatus_t ret = nvimgcodecStreamCreate(instance, code_stream);

    NVIMGCODECAPI_TRY
        {
            if (ret == NVIMGCODEC_STATUS_SUCCESS) {
                (*code_stream)->code_stream_->parseFromMem(data, size);
            }
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamCreateToFile(
    nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, const char* file_name, const nvimgcodecImageInfo_t* image_info)
{
    nvimgcodecStatus_t ret = nvimgcodecStreamCreate(instance, code_stream);
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(code_stream)
            CHECK_NULL(file_name)
            if (ret == NVIMGCODEC_STATUS_SUCCESS) {
                (*code_stream)->code_stream_->setOutputToFile(file_name);
                (*code_stream)->code_stream_->setImageInfo(image_info);
            }
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamCreateToHostMem(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, void* ctx,
    nvimgcodecResizeBufferFunc_t get_buffer_func, const nvimgcodecImageInfo_t* image_info)
{
    nvimgcodecStatus_t ret = nvimgcodecStreamCreate(instance, code_stream);
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(code_stream)
            CHECK_NULL(image_info)
            CHECK_NULL(get_buffer_func)
            if (ret == NVIMGCODEC_STATUS_SUCCESS) {
                (*code_stream)->code_stream_->setOutputToHostMem(ctx, get_buffer_func);
                (*code_stream)->code_stream_->setImageInfo(image_info);
            }
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamDestroy(nvimgcodecCodeStream_t code_stream)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(code_stream)
            delete code_stream;
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamGetImageInfo(nvimgcodecCodeStream_t code_stream, nvimgcodecImageInfo_t* image_info)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(code_stream)
            CHECK_NULL(image_info)
            return code_stream->code_stream_->getImageInfo(image_info);
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDecoderCreate(
    nvimgcodecInstance_t instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;

    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(decoder)
            CHECK_NULL(exec_params)
            std::unique_ptr<ImageGenericDecoder> image_decoder =
                instance->director_.createGenericDecoder(exec_params, options);
            *decoder = new nvimgcodecDecoder();
            (*decoder)->image_decoder_ = std::move(image_decoder);
            (*decoder)->instance_ = instance;
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecDecoderDestroy(nvimgcodecDecoder_t decoder)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(decoder)
            delete decoder;
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDecoderCanDecode(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStream_t* streams,
    const nvimgcodecImage_t* images, int batch_size, const nvimgcodecDecodeParams_t* params, nvimgcodecProcessingStatus_t* processing_status,
    int force_format)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(streams)
            CHECK_NULL(images)
            CHECK_NULL(params)

            if (batch_size <= 0)
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;

            std::vector<nvimgcodec::ICodeStream*> internal_code_streams;
            std::vector<nvimgcodec::IImage*> internal_images;

            for (int i = 0; i < batch_size; ++i) {
                internal_code_streams.push_back(streams[i]->code_stream_.get());
                internal_images.push_back(&images[i]->image_);
            }

            decoder->image_decoder_->canDecode(internal_code_streams, internal_images, params, processing_status, force_format);
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecDecoderDecode(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStream_t* streams, const nvimgcodecImage_t* images,
    int batch_size, const nvimgcodecDecodeParams_t* params, nvimgcodecFuture_t* future)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(streams)
            CHECK_NULL(images)
            CHECK_NULL(params)
            CHECK_NULL(future)

            if (batch_size <= 0)
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;

            std::vector<nvimgcodec::ICodeStream*> internal_code_streams;
            std::vector<nvimgcodec::IImage*> internal_images;

            for (int i = 0; i < batch_size; ++i) {
                internal_code_streams.push_back(streams[i]->code_stream_.get());
                internal_images.push_back(&images[i]->image_);
            }
            *future = new nvimgcodecFuture();

            (*future)->handle_ = std::move(decoder->image_decoder_->decode(internal_code_streams, internal_images, params));
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecImageCreate(nvimgcodecInstance_t instance, nvimgcodecImage_t* image, const nvimgcodecImageInfo_t* image_info)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;

    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(image)
            CHECK_NULL(instance)
            CHECK_NULL(image_info)
            CHECK_NULL(image_info->buffer)
            if (image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_UNKNOWN ||
                image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED) {
                NVIMGCODEC_LOG_ERROR(Logger::get_default(), "Unknown or unsupported buffer kind");
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }

            *image = new nvimgcodecImage();
            (*image)->image_.setImageInfo(image_info);
            (*image)->nvimgcodec_instance_ = instance;
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecImageDestroy(nvimgcodecImage_t image)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(image)
            delete image;
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecImageGetImageInfo(nvimgcodecImage_t image, nvimgcodecImageInfo_t* image_info)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(image)
            image->image_.getImageInfo(image_info);
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecEncoderCreate(nvimgcodecInstance_t instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;

    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(encoder)
            CHECK_NULL(exec_params)
            std::unique_ptr<ImageGenericEncoder> image_encoder =
                instance->director_.createGenericEncoder(exec_params, options);
            *encoder = new nvimgcodecEncoder();
            (*encoder)->image_encoder_ = std::move(image_encoder);
            (*encoder)->instance_ = instance;
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecEncoderDestroy(nvimgcodecEncoder_t encoder)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(encoder)
            delete encoder;
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecEncoderCanEncode(nvimgcodecEncoder_t encoder, const nvimgcodecImage_t* images,
    const nvimgcodecCodeStream_t* streams, int batch_size, const nvimgcodecEncodeParams_t* params,
    nvimgcodecProcessingStatus_t* processing_status, int force_format)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;

    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(streams)
            CHECK_NULL(images)
            CHECK_NULL(params)

            std::vector<nvimgcodec::ICodeStream*> internal_code_streams;
            std::vector<nvimgcodec::IImage*> internal_images;

            for (int i = 0; i < batch_size; ++i) {
                internal_code_streams.push_back(streams[i]->code_stream_.get());
                internal_images.push_back(&images[i]->image_);
            }

            encoder->image_encoder_->canEncode(internal_images, internal_code_streams, params, processing_status, force_format);
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecEncoderEncode(nvimgcodecEncoder_t encoder, const nvimgcodecImage_t* images, const nvimgcodecCodeStream_t* streams,
    int batch_size, const nvimgcodecEncodeParams_t* params, nvimgcodecFuture_t* future)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;

    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(streams)
            CHECK_NULL(images)
            CHECK_NULL(params)
            CHECK_NULL(future)

            std::vector<nvimgcodec::ICodeStream*> internal_code_streams;
            std::vector<nvimgcodec::IImage*> internal_images;

            for (int i = 0; i < batch_size; ++i) {
                internal_code_streams.push_back(streams[i]->code_stream_.get());
                internal_images.push_back(&images[i]->image_);
            }

            *future = new nvimgcodecFuture();

            (*future)->handle_ = std::move(encoder->image_encoder_->encode(internal_images, internal_code_streams, params));
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecDebugMessengerCreate(
    nvimgcodecInstance_t instance, nvimgcodecDebugMessenger_t* dbgMessenger, const nvimgcodecDebugMessengerDesc_t* messengerDesc)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(messengerDesc)

            *dbgMessenger = new nvimgcodecDebugMessenger(messengerDesc);
            (*dbgMessenger)->instance_ = instance;
            instance->director_.registerDebugMessenger(&(*dbgMessenger)->debug_messenger_);
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecDebugMessengerDestroy(nvimgcodecDebugMessenger_t dbgMessenger)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(dbgMessenger)
            dbgMessenger->instance_->director_.unregisterDebugMessenger(&dbgMessenger->debug_messenger_);
            delete dbgMessenger;
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecFutureWaitForAll(nvimgcodecFuture_t future)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(future)
            future->handle_.wait();
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecFutureDestroy(nvimgcodecFuture_t future)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(future)
            delete future;
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecFutureGetProcessingStatus(nvimgcodecFuture_t future, nvimgcodecProcessingStatus_t* processing_status, size_t* size)
{
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY
        {
            CHECK_NULL(future)
            CHECK_NULL(size)
            auto results = future->handle_.get();
            *size = results.size();
            if (processing_status) {
                auto ptr = processing_status;
                for (auto r : results)
                    *ptr++ = r;
            }
        }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}
