# DICOM Compression Tool

A command-line tool for compressing DICOM images using nvImageCodec with hardware-accelerated encoding and decoding. Optimize storage and I/O performance for medical imaging workflows with HTJ2K lossless compression and multi-frame conversion.

## Features

- **HTJ2K Compression**: Convert DICOM files to High Throughput JPEG 2000 (HTJ2K) format with lossless compression (60-70% typical compression ratios)
- **Multi-frame Conversion**: Convert legacy single-frame DICOM series to enhanced multi-frame DICOM format
- **Hardware Acceleration**: Uses NVIDIA nvImageCodec for GPU-accelerated processing
- **Batch Processing**: Efficient batch processing optimized for large datasets and network storage (S3, NFS)
- **Basic Offset Table**: Automatically adds Basic Offset Table (BOT) for O(1) frame access
- **Network Storage Optimization**: File discovery optimized for S3/NFS with minimal stat calls using `os.scandir`

## Prerequisites

- **nvImageCodec** - NVIDIA's hardware-accelerated image codec library
- **pydicom** >= 3.0.0
- **numpy**
- **highdicom** (optional, for enhanced multi-frame conversion)

## Usage

### Basic Commands

```bash
# Show help
./nvdicom.py --help

# Show help for compress command
./nvdicom.py compress --help

# Enable verbose logging (input_dir can be a directory containing DICOM files)
./nvdicom.py --verbose compress /path/to/input_dir /path/to/output_dir --htj2k
```

### Compress DICOM with HTJ2K

Convert DICOM files to HTJ2K (High Throughput JPEG 2000) lossless compression:

```bash
# Basic HTJ2K compression (keeps original file structure)
./nvdicom.py compress /path/to/input_dir /path/to/output_dir --htj2k

# HTJ2K compression with custom parameters
./nvdicom.py compress /path/to/input_dir /path/to/output_dir \
    --htj2k \
    --htj2k-num-resolutions 8 \
    --htj2k-code-block-size 128 128 \
    --htj2k-progression-order RLCP \
    --batch-size 512
```

### Convert to Multi-Frame Enhanced DICOM

Combine a series of single-frame DICOM files (all from the same imaging series and with matching dimensions/modality) into a single enhanced multi-frame DICOM file.

**Note:** You need multiple DICOM files from the same series (typically all slices of a CT, MR, or similar study) to combine them into one multi-frame DICOM. The tool discovers and sorts all DICOMs in the input directory by SeriesInstanceUID, groups them appropriately, and outputs one enhanced DICOM file per input series. If input files have inconsistent image sizes or other required attributes, those files will not be converted, and only copied to the output directory.

```bash
# Convert to uncompressed multi-frame (legacy format to enhanced format)
./nvdicom.py compress /path/to/input_dir /path/to/output_dir --multiframe

# Convert to multi-frame AND compress with HTJ2K (recommended)
./nvdicom.py compress /path/to/input_dir /path/to/output_dir --multiframe --htj2k

# Multi-frame with custom HTJ2K parameters
./nvdicom.py compress /path/to/input_dir /path/to/output_dir \
    --multiframe \
    --htj2k \
    --htj2k-num-resolutions 8 \
    --htj2k-progression-order RPCL
```

### Command Options

**Main Options:**
- `--htj2k`: Enable HTJ2K lossless compression
- `--multiframe`: Convert single-frame series to multi-frame enhanced DICOM
- `--verbose` or `-v`: Enable verbose logging

**HTJ2K Parameters** (only apply when `--htj2k` is used):
- `--htj2k-num-resolutions N`: Number of wavelet decomposition levels (default: 6)
  - Higher values = better compression but slower encoding
- `--htj2k-code-block-size HEIGHT WIDTH`: Code block size (default: 64 64)
  - Must be powers of 2. Common values: 32, 64, 128
- `--htj2k-progression-order ORDER`: Progression order (default: RPCL)
  - Options: LRCP, RLCP, RPCL, PCRL, CPRL (see details below)
- `--batch-size N`: Maximum batch size for GPU processing (default: 256)
  - Increase for larger GPUs (512 or 1024) to improve throughput

## Examples

### Example 1: HTJ2K compression only

Transcode existing DICOM files to HTJ2K without changing file organization:

```bash
./nvdicom.py compress ./dicom_input ./dicom_htj2k_output --htj2k
```

**Result**: Each input DICOM file → corresponding HTJ2K compressed output file

### Example 2: Multi-frame conversion with HTJ2K (recommended)

Convert single-frame series to multi-frame AND compress with HTJ2K:

```bash
./nvdicom.py compress ./single_frame_series ./multiframe_output --multiframe --htj2k
```

**Result**: Multiple single-frame files per series → single multi-frame HTJ2K file per series

### Example 3: Advanced HTJ2K with custom parameters

```bash
./nvdicom.py --verbose compress ./input ./output \
    --multiframe \
    --htj2k \
    --htj2k-num-resolutions 8 \
    --htj2k-code-block-size 128 128 \
    --batch-size 512 \
    --htj2k-progression-order RLCP
```

## HTJ2K Progression Orders

The progression order determines how the compressed data is organized in the bitstream:

- **LRCP**: Layer-Resolution-Component-Position (quality scalability)
  - Transfer Syntax: `1.2.840.10008.1.2.4.201` (HTJ2K Lossless Only)
- **RLCP**: Resolution-Layer-Component-Position (resolution scalability)
  - Transfer Syntax: `1.2.840.10008.1.2.4.201` (HTJ2K Lossless Only)
- **RPCL**: Resolution-Position-Component-Layer (progressive by resolution) - **Default**
  - Transfer Syntax: `1.2.840.10008.1.2.4.202` (HTJ2K with RPCL Options)
- **PCRL**: Position-Component-Resolution-Layer (progressive by spatial area)
  - Transfer Syntax: `1.2.840.10008.1.2.4.201` (HTJ2K Lossless Only)
- **CPRL**: Component-Position-Resolution-Layer (component scalability)
  - Transfer Syntax: `1.2.840.10008.1.2.4.201` (HTJ2K Lossless Only)

## Technical Details

### HTJ2K (High Throughput JPEG 2000)

HTJ2K is a faster variant of JPEG 2000 optimized for medical imaging:
- **Lossless compression**: 60-70% typical compression ratios (file size reduced to 30-40% of original)
- **Hardware acceleration**: GPU-accelerated encoding/decoding via NVIDIA nvImageCodec
- **Maintains image quality**: Bit-perfect lossless compression for medical applications
- **Faster than JPEG 2000**: Optimized block coding for higher throughput
- **DICOM compliant**: Uses standard DICOM Transfer Syntax UIDs

### Multi-Frame Enhanced DICOM

The tool converts legacy single-frame DICOM series to enhanced multi-frame format:
- **Single file per series**: Instead of hundreds of individual files
- **Standards-compliant**: Uses `highdicom` library for DICOM Part 3 compliance
- **Supported modalities**: CT, MR, PT
- **Unsupported modalities**: MG, US, XA (transcoded or copied without conversion)
- **Preserves metadata**: SeriesInstanceUID can be preserved or regenerated
- **Better I/O performance**: Fewer files = faster network operations and backups

### Supported Transfer Syntaxes

**Input (will be transcoded):**
- HTJ2K (High-Throughput JPEG 2000) - `1.2.840.10008.1.2.4.201`, `1.2.840.10008.1.2.4.202`
- JPEG 2000 (lossless and lossy) - `1.2.840.10008.1.2.4.90`, `1.2.840.10008.1.2.4.91`
- JPEG (baseline, extended, lossless) - `1.2.840.10008.1.2.4.50`, `1.2.840.10008.1.2.4.51`, etc.
- Uncompressed - Explicit/Implicit VR Little/Big Endian

**Output:**
- HTJ2K lossless compression (when `--htj2k` flag is used)
- Uncompressed Explicit VR Little Endian (for multi-frame without `--htj2k`)

### Network Storage Optimization

File discovery has been optimized for network storage (S3, NFS, etc.):
- Uses `os.scandir()` instead of `os.walk()` to minimize stat calls
- Reduced network round-trips for faster directory traversal
- Validates DICOM files by reading minimal header (checks for "DICM" magic number at offset 128)
- Significant performance improvement when working with cloud storage

## Performance Tips

1. **Batch Size**: Increase `--batch-size` for larger GPUs
   - Default: 256 frames
   - Increase for higher-end GPUs
   - Lower values reduce memory usage
2. **GPU Memory**: Monitor with `nvidia-smi` to tune batch size
3. **Storage Savings**: HTJ2K typically reduces file size to 30-40% of original (60-70% compression)
4. **Multi-frame + HTJ2K**: Combine both for maximum benefit:
   - Fewer files (better I/O)
   - Smaller size (better storage efficiency)

## Testing

The tool includes comprehensive test suites:
- `test_convert_htj2k.py` - Tests for HTJ2K transcoding
- `test_convert_multiframe.py` - Tests for multi-frame conversion
- `test_compress_midi_b_dataset.py` - Tests for MIDI-B dataset compression

Run tests with:
```bash
pytest -v test/python/example/tools/dicom/
```

## Troubleshooting

### Import Errors
If you see `ImportError: nvimgcodec not found`:
```bash
pip install nvidia-nvimgcodec-cu12[all]
```
or 
```bash
pip install nvidia-nvimgcodec-cu13[all]
```

### Memory Issues
If GPU runs out of memory:
- Reduce `--batch-size` (try 128 or 64)
- Process smaller directories at a time

### Network Storage Performance
The tool tries to minimize the number of reads. However, one could try to copy the files to temporary
local storage before the conversion.

## License

SPDX-License-Identifier: Apache-2.0

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

