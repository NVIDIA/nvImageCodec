# Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
"""Optional GPU contract test; Torch/torchvision are references, not decoder dependencies."""
import argparse
import ctypes
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.io import decode_jpeg, ImageReadMode


class Native:
    def __init__(self, library, device):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.library.jce_last_error.restype = ctypes.c_char_p
        self.library.jce_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self.library.jce_destroy.argtypes = [ctypes.c_void_p]
        self.library.jce_probe.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.library.jce_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                           ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.library.jce_wait.argtypes = [ctypes.c_void_p]
        self.handle = ctypes.c_void_p()
        self.check(self.library.jce_create(device, ctypes.byref(self.handle)))

    def check(self, result):
        if result:
            raise RuntimeError(f'native status {result}: {self.library.jce_last_error().decode()}')

    def decode(self, encoded, device, padding=7):
        source = ctypes.create_string_buffer(encoded)
        width, height = ctypes.c_int(), ctypes.c_int()
        self.check(self.library.jce_probe(source, len(encoded), ctypes.byref(width), ctypes.byref(height)))
        backing = torch.full((3, height.value, width.value + padding), 203, dtype=torch.uint8, device=device)
        output = backing[:, :, :width.value]
        self.check(self.library.jce_decode(self.handle, source, len(encoded), output.data_ptr(), output.stride(1),
                                           output.stride(0), width.value, height.value,
                                           torch.cuda.current_stream(device).cuda_stream))
        self.check(self.library.jce_wait(self.handle))
        if padding:
            assert torch.all(backing[:, :, width.value:] == 203).item(), 'row padding overwritten'
        return output

    def close(self):
        if self.handle:
            self.library.jce_destroy(self.handle)
            self.handle = ctypes.c_void_p()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda', args.gpu)
    native = Native(args.library, args.gpu)
    cases = []
    rng = np.random.default_rng(2026)
    for height, width in [(1, 1), (7, 9), (8, 8), (17, 31), (64, 65), (257, 128)]:
        for gray in (False, True):
            for quality in (30, 85, 100):
                shape = (height, width) if gray else (height, width, 3)
                image = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality, subsampling=0)
                cases.append((f'generated-{height}x{width}-gray{gray}-q{quality}', buffer.getvalue()))
    if args.images:
        cases += [(path.name, path.read_bytes()) for path in sorted(args.images.glob('*.jpg'))[:args.limit]]
    checked = skipped = 0
    try:
        for name, encoded in cases:
            header_width, header_height = ctypes.c_int(), ctypes.c_int()
            source = ctypes.create_string_buffer(encoded)
            status = native.library.jce_probe(source, len(encoded), ctypes.byref(header_width), ctypes.byref(header_height))
            if status:
                if name.startswith('generated-'):
                    raise RuntimeError(f'generated image rejected: {name}, {status}')
                skipped += 1
                continue
            reference = decode_jpeg(torch.frombuffer(bytearray(encoded), dtype=torch.uint8),
                                    mode=ImageReadMode.RGB, device=device)
            output = native.decode(encoded, device)
            assert torch.equal(reference, output), (name, (reference.int() - output.int()).abs().max().item())
            checked += 1
            last_supported = encoded
        # A caller-supplied stream and a tightly packed destination exercise a
        # second allocation/stream path with the same decoder's resource reuse.
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            encoded = last_supported
            output = native.decode(encoded, device, padding=0)
        reference = decode_jpeg(torch.frombuffer(bytearray(encoded), dtype=torch.uint8),
                                mode=ImageReadMode.RGB, device=device)
        assert torch.equal(reference, output)
        result = {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device),
                  'checked': checked, 'skipped_outside_initial_scope': skipped,
                  'bit_identical': checked, 'padded_rows': True, 'alternate_stream': True}
        print(json.dumps(result))
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + '\n')
    finally:
        native.close()


if __name__ == '__main__':
    main()
