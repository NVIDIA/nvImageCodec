
from nvidia import nvimgcodec
from utils import img_dir_path
import os

# create dummy decoder, which is alive for all tests and speedups creation of nvJPEG extension
# make sure utils are imported in each test that is using nvJPEG to see the speedup
dummy = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.HW_GPU_ONLY)])
try:
    # decode an image to load HW JPEG decoder
    dummy.decode(os.path.join(img_dir_path, "jpeg/padlock-406986_640_420.jpg"))
except: # may fail, if image is not present, but we just need to load decoder, so we don't care
    pass
