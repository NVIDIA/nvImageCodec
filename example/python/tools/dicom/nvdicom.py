#!/usr/bin/env python3
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
DICOM Conversion Tool

This tool provides utilities for converting DICOM images using nvImageCodec.
"""
import logging
import os
import hashlib
from pathlib import Path
import pydicom

from nvidia.nvimgcodec.tools.dicom.convert_htj2k import transcode_datasets_to_htj2k
from nvidia.nvimgcodec.tools.dicom.convert_multiframe import convert_to_enhanced_dicom
from nvidia.nvimgcodec.tools.dicom.dicom_utils import (
    DicomSeriesScanner,
    DicomFileLoader,
)

logger = logging.getLogger(__name__)

def _save_dataset(dataset, output_dir):
    # Generate output filename using study/series/instance info
    study_uid = getattr(dataset, "StudyInstanceUID", "unknown")
    series_num = str(getattr(dataset, "SeriesNumber", 1))
    instance_num = str(getattr(dataset, "InstanceNumber", 1))
    sop_class = str(getattr(dataset, "SOPClassUID", ""))
    sop_instance = str(getattr(dataset, "SOPInstanceUID", ""))

    # Create short hash for uniqueness
    hash_input = (sop_class + sop_instance).encode("utf-8")
    hash_val = hashlib.sha256(hash_input).hexdigest()[:8]

    filename = f"{study_uid}-{series_num}-{instance_num}-{hash_val}.dcm"
    output_path = os.path.join(output_dir, filename)

    logger.info(f"Saving: {output_path}")
    dataset.save_as(output_path, enforce_file_format=False)

def main():
    """Main CLI entry point for DICOM conversion tools."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="DICOM Conversion Tool - Transcode DICOM files using nvImageCodec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcode DICOM to HTJ2K, convert single-frame series to multi-frame and compress with HTJ2K:
  %(prog)s compress /path/to/input_dir /path/to/output_dir --multiframe --htj2k

  # Transcode DICOM to HTJ2K, but keep single-frame series:
  %(prog)s compress /path/to/input_dir /path/to/output_dir --htj2k

  # Convert single-frame series to multi-frame, but do not compress with HTJ2K:
  %(prog)s compress /path/to/input_dir /path/to/output_dir --multiframe
        """,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Conversion command")

    # Compress command
    compress_parser = subparsers.add_parser(
        "compress",
        help="Compress DICOM files, by converting single-frame series to multi-frame and compressing with HTJ2K",
    )
    compress_parser.add_argument(
        "input_dir", 
        type=str, help="Input directory containing DICOM files"
    )
    compress_parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=None,
        help="Output directory (default: creates a temporary directory)",
    )
    compress_parser.add_argument(
        "--htj2k",
        action="store_true",
        help="Compress with HTJ2K",
    )
    compress_parser.add_argument(
        "--multiframe",
        action="store_true",
        help="Convert single-frame series to multi-frame",
    )
    compress_parser.add_argument(
        "--htj2k-num-resolutions",
        type=int,
        default=6,
        help="Number of wavelet decomposition levels (default: 6)",
    )
    compress_parser.add_argument(
        "--htj2k-code-block-size",
        type=int,
        nargs=2,
        default=[64, 64],
        metavar=("HEIGHT", "WIDTH"),
        help="Code block size as height width (default: 64 64)",
    )
    compress_parser.add_argument(
        "--htj2k-progression-order",
        type=str,
        default="RPCL",
        choices=["LRCP", "RLCP", "RPCL", "PCRL", "CPRL"],
        help="Progression order for HTJ2K encoding (default: RPCL)",
    )
    compress_parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Maximum batch size for processing (default: 256)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Check if command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "compress":
            # Validate input directory
            if not os.path.isdir(args.input_dir):
                logger.error(f"Input directory does not exist: {args.input_dir}")
                sys.exit(1)

            # Setup output directory
            if args.output_dir is None:
                import tempfile
                args.output_dir = tempfile.mkdtemp(prefix="dicom_compress_")
            os.makedirs(args.output_dir, exist_ok=True)

            # Scan input directory and group by series
            if args.multiframe:
                scanner = DicomSeriesScanner(args.input_dir)
                scanner.scan()

                logger.info(f"Found {scanner.get_series_count()} series to compress")

                # HTJ2K transfer syntax
                transfer_syntax = "1.2.840.10008.1.2.4.202" if args.htj2k else None

                # Convert each series
                for series in scanner.iter_series():
                    logger.info(
                        f"Converting series {series.series_uid}: "
                        f"{series.modality}, {len(series)} files"
                    )

                    # Load datasets for this series
                    series_datasets = [[pydicom.dcmread(f) for f in series.files]]

                    # Convert to enhanced multi-frame
                    enhanced_datasets = convert_to_enhanced_dicom(
                        series_datasets=series_datasets,
                        transfer_syntax_uid=transfer_syntax,
                        num_resolutions=args.htj2k_num_resolutions,
                        code_block_size=tuple(args.htj2k_code_block_size),
                        progression_order=args.htj2k_progression_order,
                    )
                    for dataset in enhanced_datasets:
                        _save_dataset(dataset, args.output_dir)
            
            elif args.htj2k:
                file_loader = DicomFileLoader(args.input_dir, args.output_dir, args.batch_size)
                for batch_input, batch_output in file_loader:
                    datasets = [pydicom.dcmread(f) for f in batch_input]
                    enhanced_datasets = transcode_datasets_to_htj2k(
                        datasets=datasets,
                        num_resolutions=args.htj2k_num_resolutions,
                        code_block_size=tuple(args.htj2k_code_block_size),
                        progression_order=args.htj2k_progression_order,
                        max_batch_size=args.batch_size,
                        skip_transfer_syntaxes=["1.2.840.10008.1.2.4.201", "1.2.840.10008.1.2.4.202"],
                    )
                    for dataset in enhanced_datasets:
                        _save_dataset(dataset, args.output_dir)
            else:
                parser.error("At least one of --multiframe or --htj2k must be specified")

            logger.info(
                f"\n✓ Multi-frame conversion complete! Output directory: {args.output_dir}"
            )

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(
            "Please ensure nvidia-nvimgcodec and other dependencies are installed"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
