import numpy as np
import pytest as t
from nvidia import nvimgcodec

backends_cpu_only = [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]
backends_gpu_only = [nvimgcodec.Backend(nvimgcodec.GPU_ONLY), nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU)]
default_image_shape = (480, 640, 3)

def encode_decode(extension, backends, dtype, shape, max_mean_diff=None, encode_params=None, decode_params=None):
    encoder = nvimgcodec.Encoder(backends=backends)
    decoder = nvimgcodec.Decoder(backends=backends)

    image = np.random.randint(0, np.iinfo(dtype).max, shape, dtype)
    encoded = encoder.encode(image, extension, encode_params)
    decoded = np.asarray(decoder.decode(encoded, decode_params).cpu())

    assert decoded.dtype == dtype

    mean_diff = np.abs(image.astype(np.int32) - decoded.astype(np.int32)).mean()

    if max_mean_diff is None:
        assert mean_diff == 0.0
    else:
        assert mean_diff > 0 and mean_diff / np.iinfo(dtype).max < max_mean_diff

def encode_decode_lossless(extension, backends, dtype, shape):
    assert extension != "jpeg", "jpeg is always lossless"

    if extension == "jpeg2k":
        encode_params = nvimgcodec.EncodeParams(jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(reversible=True))
    elif extension == "webp":
        encode_params = nvimgcodec.EncodeParams(quality=101)
    else:
        encode_params = None

    if dtype != np.uint8:
        decode_params = nvimgcodec.DecodeParams(allow_any_depth=True)
    else:
        decode_params = None

    encode_decode(extension, backends, dtype, shape, max_mean_diff=None,
                    encode_params=encode_params, decode_params=decode_params)

def encode_decode_lossy(extension, backends, dtype, shape):
    assert extension == "jpeg" or extension == "jpeg2k" or extension == "webp"

    if dtype != np.uint8:
        decode_params = nvimgcodec.DecodeParams(allow_any_depth=True)
    else:
        decode_params = None

    if extension == "webp":
        max_mean_diff = 0.2 #webp is not that great with random images
    else:
        max_mean_diff = 0.02

    encode_decode(extension, backends, dtype, shape, max_mean_diff=max_mean_diff,
                    decode_params=decode_params)

@t.mark.parametrize("extension", ["png", "bmp", "jpeg2k", "pnm", "tiff", "webp"])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_uint8_lossless(extension, backends):
    encode_decode_lossless(extension, backends, np.uint8, default_image_shape)

@t.mark.parametrize("extension", ["png", "jpeg2k", "pnm", "tiff"])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_uint16_lossless(extension, backends):
    encode_decode_lossless(extension, backends, np.uint16, default_image_shape)

@t.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_only_gpu_lossless(dtype):
    encode_decode_lossless("jpeg2k", backends_gpu_only, dtype, default_image_shape)

@t.mark.parametrize("extension,dtype", [
    ("jpeg", np.uint8),
    ("jpeg2k", np.uint8),
    ("jpeg2k", np.uint16),
    ("webp", np.uint8),
])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_lossy(extension, dtype, backends):
    encode_decode_lossy(extension, backends, dtype, default_image_shape)

@t.mark.parametrize("extension,dtype", [
    ("jpeg", np.uint8),
    ("jpeg2k", np.uint8),
    ("jpeg2k", np.uint16),
])
def test_only_gpu_lossy(extension, dtype):
    encode_decode_lossy("jpeg2k", backends_gpu_only, dtype, default_image_shape)
