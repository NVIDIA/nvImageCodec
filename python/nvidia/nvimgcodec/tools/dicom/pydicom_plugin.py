# SPDX-FileCopyrightText: Copyright (c) 2025-2026 MONAI Consortium
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This decoder plugin for nvimgcodec <https://github.com/NVIDIA/nvImageCodec> decompresses
encoded Pixel Data for the following transfer syntaxes:
    JPEGBaseline8Bit, 1.2.840.10008.1.2.4.50, JPEG Baseline (Process 1)
    JPEGLossless, 1.2.840.10008.1.2.4.57, JPEG Lossless, Non-Hierarchical (Process 14)
    JPEGLosslessSV1, 1.2.840.10008.1.2.4.70, JPEG Lossless, Non-Hierarchical, First-Order Prediction
    JPEG2000Lossless, 1.2.840.10008.1.2.4.90, JPEG 2000 Image Compression (Lossless Only)
    JPEG2000, 1.2.840.10008.1.2.4.91, JPEG 2000 Image Compression
    HTJ2KLossless, 1.2.840.10008.1.2.4.201, HTJ2K Image Compression (Lossless Only)
    HTJ2KLosslessRPCL, 1.2.840.10008.1.2.4.202, HTJ2K with RPCL Options Image Compression (Lossless Only)
    HTJ2K, 1.2.840.10008.1.2.4.203, HTJ2K Image Compression

There are two ways to add a custom decoding plugin to pydicom:
1. Using the pixel_data_handlers backend, though pydicom.pixel_data_handlers module is deprecated
   and will be removed in v4.0.
2. Using the pixels backend by adding a decoder plugin to existing decoders with the add_plugin method,
   see https://pydicom.github.io/pydicom/stable/guides/decoding/decoder_plugins.html

It is noted that pydicom.dataset.Dataset.pixel_array changed in version 3.0 where the backend used for
pixel data decoding changed from the pixel_data_handlers module to the pixels module.

So, this implementation uses the pixels backend.

Plugin Requirements:
A custom decoding plugin must implement three objects within the same module:
 - A function named is_available with the following signature:
    def is_available(uid: pydicom.uid.UID) -> bool:
      Where uid is the Transfer Syntax UID for the corresponding decoder as a UID
 - A dict named DECODER_DEPENDENCIES with the type dict[pydicom.uid.UID, tuple[str, ...], such as:
    DECODER_DEPENDENCIES = {JPEG2000Lossless: ('numpy', 'pillow', 'imagecodecs'),}
      This will be used to provide the user with a list of dependencies required by the plugin.
 - A function that performs the decoding with the following function signature as in Github repo:
    def _decode_frame(src: bytes, runner: DecodeRunner) -> bytearray | bytes
      src is a single frame’s worth of raw compressed data to be decoded, and
      runner is a DecodeRunner instance that manages the decoding process.

Adding plugins to a Decoder:
Additional plugins can be added to an existing decoder with the add_plugin() method
 ```python
  from pydicom.pixels.decoders import RLELosslessDecoder
  RLELosslessDecoder.add_plugin(
    'my_decoder',  the plugin's label
    ('my_package.decoders', 'my_decoder_func')  the import paths
  )
 ```
"""

import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
from pydicom.pixels.common import PhotometricInterpretation as PI  # noqa: N817
from pydicom.pixels.common import (
    RunnerBase,
)
from pydicom.pixels.decoders import (
    HTJ2KDecoder,
    HTJ2KLosslessDecoder,
    HTJ2KLosslessRPCLDecoder,
    JPEG2000Decoder,
    JPEG2000LosslessDecoder,
    JPEGBaseline8BitDecoder,
    JPEGLosslessDecoder,
    JPEGLosslessSV1Decoder,
)
from pydicom.pixels.decoders.base import DecodeRunner
from pydicom.uid import UID, JPEG2000TransferSyntaxes

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from nvidia import nvimgcodec
    try:
        version_str = getattr(nvimgcodec, "__version__", "0")
        nvimgcodec_version = tuple(int(s) for s in version_str.split(".") if s.isdigit())
    except (AttributeError, ValueError, TypeError):
        nvimgcodec_version = (0,)
except ImportError:
    nvimgcodec = None

# nvimgcodec minimum version required and the label for this decoder plugin.
NVIMGCODEC_MIN_VERSION = "0.8.0"
NVIMGCODEC_MIN_VERSION_TUPLE = tuple(int(x) for x in NVIMGCODEC_MIN_VERSION.split("."))
NVIMGCODEC_PLUGIN_LABEL = "0.8.0+nvimgcodec"  # to be sorted to first in ascending order of plugins

# Supported decoder classes of the corresponding transfer syntaxes by this decoder plugin.
SUPPORTED_DECODER_CLASSES = [
    JPEGBaseline8BitDecoder,  # 1.2.840.10008.1.2.4.50, JPEG Baseline (Process 1)
    JPEGLosslessDecoder,  # 1.2.840.10008.1.2.4.57, JPEG Lossless, Non-Hierarchical (Process 14)
    JPEGLosslessSV1Decoder,  # 1.2.840.10008.1.2.4.70, JPEG Lossless, Non-Hierarchical, First-Order Prediction
    JPEG2000LosslessDecoder,  # 1.2.840.10008.1.2.4.90, JPEG 2000 Image Compression (Lossless Only)
    JPEG2000Decoder,  # 1.2.840.10008.1.2.4.91, JPEG 2000 Image Compression
    HTJ2KLosslessDecoder,  # 1.2.840.10008.1.2.4.201, HTJ2K Image Compression (Lossless Only)
    HTJ2KLosslessRPCLDecoder,  # 1.2.840.10008.1.2.4.202, HTJ2K with RPCL Options Image Compression (Lossless Only)
    HTJ2KDecoder,  # 1.2.840.10008.1.2.4.203, HTJ2K Image Compression
]

SUPPORTED_TRANSFER_SYNTAXES: Iterable[UID] = [x.UID for x in SUPPORTED_DECODER_CLASSES]

_logger = logging.getLogger(__name__)


# Lazy singleton for nvimgcodec decoder; initialized on first use
# Decode params are created per-decode based on image characteristics
if nvimgcodec:
    _NVIMGCODEC_DECODER: Any = None
else:  # pragma: no cover - nvimgcodec not installed
    _NVIMGCODEC_DECODER = None

# Required for decoder plugin
DECODER_DEPENDENCIES = {
    x: ("numpy", "cupy", f"nvidia-nvimgcodec-cuXX[all]>=0.8.0 (where XX is the CUDA version: 12, 13)")
    for x in SUPPORTED_TRANSFER_SYNTAXES
}

DEFAULT_PI_NAME = "nvimgcodec_default_photometric_interpretation"


# Required for decoder plugin
def is_available(uid: UID) -> bool:
    """Return ``True`` if a pixel data decoder for ``uid`` is available.

    Args:
        uid (UID): The transfer syntax UID to check.

    Returns:
        bool: ``True`` if a pixel data decoder for ``uid`` is available,
        ``False`` otherwise.
    """

    _logger.debug(f"Checking if CUDA and nvimgcodec available for transfer syntax: {uid}")

    if uid not in SUPPORTED_TRANSFER_SYNTAXES:
        _logger.debug(f"Transfer syntax {uid} not supported by nvimgcodec.")
        return False

    return True


# Required for decoder plugin
def _decode_frame(src: bytes, runner: DecodeRunner) -> bytearray | bytes:
    """Return the decoded image data in `src` as a :class:`bytearray` or :class:`bytes`."""
    tsyntax = runner.transfer_syntax
    _logger.debug(f"transfer_syntax: {tsyntax}")

    if not is_available(tsyntax):
        raise ValueError(f"Transfer syntax {tsyntax} not supported; see details in the debug log.")

    # runner.set_frame_option(runner.index, "decoding_plugin", NVIMGCODEC_PLUGIN_LABEL)  # type: ignore[attr-defined]
    # in pydicom v3.1.0 can use the above call, but do we want to limit to this plugin?
    is_jpeg2k = tsyntax in JPEG2000TransferSyntaxes
    samples_per_pixel = runner.samples_per_pixel
    photometric_interpretation = runner.photometric_interpretation

    # --- JPEG 2000: Precision/Bit depth ---
    if is_jpeg2k:
        precision, bits_allocated = _jpeg2k_precision_bits(runner)
        # runner.set_frame_option(runner.index, "bits_allocated", bits_allocated)  # type: ignore[attr-defined]
        # in pydicom v3.1.0 can use the above call
        runner.set_option("bits_allocated", bits_allocated)
        _logger.debug(f"Set bits_allocated to {bits_allocated} for J2K precision {precision}")

    # Check if RGB conversion requested (following Pillow decoder logic)
    convert_to_rgb = (
        samples_per_pixel > 1 and runner.get_option("as_rgb", False) and "YBR" in photometric_interpretation
    )

    decoder = _get_decoder_resources()
    params = _get_decode_params(runner)
    decoded_data = decoder.decode(src, params=params)
    if decoded_data is None:
        raise RuntimeError("Decoding failed: decoder.decode() returned None")
    decoded_data = decoded_data.cpu()
    np_surface = np.ascontiguousarray(np.asarray(decoded_data))

    # Update photometric interpretation if we converted to RGB, or JPEG 2000 YBR*
    if convert_to_rgb or photometric_interpretation in (PI.YBR_ICT, PI.YBR_RCT):
        # runner.set_frame_option(runner.index, "photometric_interpretation", PI.RGB)  # type: ignore[attr-defined]
        # in pydicom v3.1.0 can use the above call
        runner.set_option("photometric_interpretation", PI.RGB)
        _logger.debug(
            "Set photometric_interpretation to RGB after conversion"
            if convert_to_rgb
            else f"Set photometric_interpretation to RGB for {photometric_interpretation}"
        )

    return np_surface.tobytes()


def _get_decoder_resources() -> Any:
    """Return cached nvimgcodec decoder (parameters are created per decode)."""
    global _NVIMGCODEC_DECODER

    if _NVIMGCODEC_DECODER is None:
        _NVIMGCODEC_DECODER = nvimgcodec.Decoder(options=":fancy_upsampling=1")

    return _NVIMGCODEC_DECODER


def _get_decode_params(runner: RunnerBase) -> Any:
    """Create decode parameters based on DICOM image characteristics.

    Mimics the behavior of pydicom's Pillow decoder:
    - By default, keeps JPEG data in YCbCr format (no conversion)
    - If as_rgb option is True and photometric interpretation is YBR*, converts to RGB

    This matches the logic in pydicom.pixels.decoders.pillow._decode_frame()

    Args:
        runner: The DecodeRunner or RunnerBase instance with access to DICOM metadata.

    Returns:
        nvimgcodec.DecodeParams: Configured decode parameters.
    """
    # Access DICOM metadata from the runner
    samples_per_pixel = runner.samples_per_pixel
    photometric_interpretation = runner.get_option(DEFAULT_PI_NAME, runner.photometric_interpretation)

    # Cache the original photometric interpretation before any conversions.
    # This is needed because we may modify the PI after decoding (e.g., YBR -> RGB),
    # but the decoder needs to know the original PI to decide on color transformations.
    # Subsequent frames in multi-frame images can then reference the original PI.
    if runner.get_option(DEFAULT_PI_NAME, None) is None:
        runner.set_option(DEFAULT_PI_NAME, photometric_interpretation)

    transfer_syntax = runner.transfer_syntax
    as_rgb = runner.get_option("as_rgb", False)
    force_rgb = runner.get_option("force_rgb", False)
    force_ybr = runner.get_option("force_ybr", False)

    _logger.debug("DecodeRunner options:")
    _logger.debug(f"transfer_syntax: {transfer_syntax}")
    _logger.debug(f"photometric_interpretation: {photometric_interpretation}")
    _logger.debug(f"samples_per_pixel: {samples_per_pixel}")
    _logger.debug(f"as_rgb: {as_rgb}")
    _logger.debug(f"force_rgb: {force_rgb}")
    _logger.debug(f"force_ybr: {force_ybr}")

    # Default: keep color space unchanged
    color_spec = nvimgcodec.ColorSpec.UNCHANGED

    # For multi-sample (color) images, check if RGB conversion is requested
    if samples_per_pixel > 1:
        # JPEG 2000 color transformations are always returned as RGB (matches Pillow)
        if photometric_interpretation in (PI.YBR_ICT, PI.YBR_RCT):
            color_spec = nvimgcodec.ColorSpec.SRGB
            _logger.debug(
                f"Using RGB color spec for JPEG 2000 color transformation " f"(PI: {photometric_interpretation})"
            )
        elif transfer_syntax in (JPEGBaseline8BitDecoder.UID):
            # approach is similar to pylibjpeg from pydicom - for ybr full and 422 it needs conversion from ycbcr to rgb
            # for any other PI it just skips color conversion (ignoring what is inside jpeg header)
            if photometric_interpretation in (PI.YBR_FULL, PI.YBR_FULL_422):
                # we want to apply ycbcr -> rgb conversion
                color_spec = nvimgcodec.ColorSpec.SRGB
            else:
                # ignore color conversion as image should already be in rgb or grayscale (but jpeg header may contain wrong data)
                color_spec = nvimgcodec.ColorSpec.SYCC
        else:
            # Check the as_rgb option - same as Pillow decoder
            convert_to_rgb = as_rgb or (force_rgb and "YBR" in photometric_interpretation)

            if convert_to_rgb:
                # Convert YCbCr to RGB as requested
                color_spec = nvimgcodec.ColorSpec.SRGB
                _logger.debug(f"Using RGB color spec (as_rgb=True, PI: {photometric_interpretation})")
            else:
                # Keep YCbCr unchanged - matches Pillow's image.draft("YCbCr") behavior
                _logger.debug(
                    f"Using UNCHANGED color spec to preserve YCbCr " f"(as_rgb=False, PI: {photometric_interpretation})"
                )
    else:
        # Grayscale image - keep unchanged
        _logger.debug(
            f"Using UNCHANGED color spec for grayscale image (samples_per_pixel: {samples_per_pixel},"
            f" PI: {photometric_interpretation}, transfer_syntax: {transfer_syntax})"
        )

    return nvimgcodec.DecodeParams(
        allow_any_depth=True,
        color_spec=color_spec,
    )


def _jpeg2k_precision_bits(runner: DecodeRunner) -> tuple[int, int]:
    # precision = runner.get_frame_option(runner.index, "j2k_precision", runner.bits_stored)  # type: ignore[attr-defined]
    # in pydicom v3.1.0 can use the above call
    precision = runner.get_option("j2k_precision", runner.bits_stored)
    if 0 < precision <= 8:
        return precision, 8
    elif 8 < precision <= 16:
        if runner.samples_per_pixel > 1:
            _logger.warning(
                f"JPEG 2000 with {precision}-bit multi-sample data may have precision issues with some decoders"
            )
        return precision, 16
    else:
        raise ValueError(f"Only 'Bits Stored' values up to 16 are supported, got {precision}")


# Helper functions for an application to register/unregister this decoder plugin with Pydicom at application startup.


def register(module_path: Optional[str] = None) -> bool:
    """Register the nvImageCodec decoder plugin with pydicom.

    This enables GPU-accelerated DICOM decoding via nvImageCodec for pydicom's pixel_array
    functionality. Once registered, pydicom will automatically use nvImageCodec for supported
    transfer syntaxes.

    The Decoder class does not support sorting the plugins and uses the order in which plugins were added.
    Furthermore, the properties of ``available_plugins`` returns sorted labels only but not the Callables or
    their module and function names, and the function ``remove_plugin`` only returns a boolean.
    So there is no way to remove the available plugins before adding them back after this plugin is added.

    For now, have to access the ``private`` property ``_available`` of the Decoder class to sort the available
    plugins and make sure this custom plugin is the first in the sorted list by its label. It is known that the
    first plugin in the default list is always ``gdcm`` for the supported decoder classes, so label name needs
    to be lexicographically less than ``gdcm`` to be the first in the sorted list.

    Example:
        >>> from nvidia.nvimgcodec.tools.dicom import pydicom_plugin
        >>> import pydicom
        >>> 
        >>> # Register the plugin
        >>> pydicom_plugin.register()
        >>> 
        >>> # Now use pydicom as normal - nvImageCodec will be used automatically
        >>> ds = pydicom.dcmread("my_dicom.dcm")
        >>> pixels = ds.pixel_array  # Decoded with GPU acceleration!

    Args:
        module_path (Optional[str]): The importable module path for this plugin.
            When ``None`` or ``"__main__"``, search the loaded modules for an entry whose ``__file__`` resolves
            to the current file, e.g. module paths that start with ``nvidia.nvimgcodec.tools.dicom``.

    Returns:
        bool: ``True`` if the decoder plugin is registered successfully, ``False`` otherwise.
    """
    try:
        func_name = getattr(_decode_frame, "__name__", None)
    except NameError:
        _logger.error("Decoder function `_decode_frame` not found.")
        return False

    if module_path is None:
        module_path = _find_module_path(__name__)
    else:
        # Double check if the module path exists and if it is the same as the one for the callable origin.
        module_path_found, func_name_found = _get_callable_origin(_decode_frame)  # get the func's module path.
        if module_path_found:
            if module_path.casefold() != module_path_found.casefold():
                _logger.warning(f"Module path {module_path} does not match {module_path_found} for decoder plugin.")
        else:
            _logger.error(f"Module path {module_path} not found for decoder plugin.")
            return False

        if func_name != func_name_found:
            _logger.warning(
                f"Function {func_name_found} in {module_path_found} instead of {func_name} used for decoder plugin."
            )

    for decoder_class in SUPPORTED_DECODER_CLASSES:
        if NVIMGCODEC_PLUGIN_LABEL in decoder_class.available_plugins:
            _logger.debug(f"{NVIMGCODEC_PLUGIN_LABEL} already registered for transfer syntax {decoder_class.UID}.")
            continue

        decoder_class.add_plugin(NVIMGCODEC_PLUGIN_LABEL, (module_path, str(func_name)))
        _logger.debug(
            f"Added plugin for transfer syntax {decoder_class.UID}: "
            f"{NVIMGCODEC_PLUGIN_LABEL} with {func_name} in module path {module_path}."
        )

        # Need to sort the plugins to make sure the custom plugin is the first in items() of
        # the decoder class search for the plugin to be used.
        decoder_class._available = dict(sorted(decoder_class._available.items(), key=lambda item: item[0]))
        _logger.debug(f"Sorted plugins for transfer syntax {decoder_class.UID}: {decoder_class._available}")

    _logger.info(f"NVIDIA nvImageCodec decoder plugin registered with {len(SUPPORTED_DECODER_CLASSES)} decoder classes.")

    return True


def unregister() -> bool:
    """Unregister the nvImageCodec decoder plugin from pydicom.
    
    Example:
        >>> from nvidia.nvimgcodec.tools.dicom import pydicom_plugin
        >>> pydicom_plugin.unregister()
    
    Returns:
        bool: ``True`` if unregistration successful, ``False`` otherwise.
    """

    for decoder_class in SUPPORTED_DECODER_CLASSES:
        if NVIMGCODEC_PLUGIN_LABEL in decoder_class.available_plugins:
            decoder_class.remove_plugin(NVIMGCODEC_PLUGIN_LABEL)
        _logger.debug(f"Unregistered plugin for transfer syntax {decoder_class.UID}: {NVIMGCODEC_PLUGIN_LABEL}")
    _logger.info(f"Unregistered plugin {NVIMGCODEC_PLUGIN_LABEL} for all supported transfer syntaxes.")

    return True

def _find_module_path(module_name: str | None) -> str:
    """Return the importable module path for *module_name* file.

    When *module_name* is ``None`` or ``"__main__"``, search the loaded modules
    for an entry whose ``__file__`` resolves to the current file.

    When *module_name* is provided and not ``"__main__"``, validate it exists in
    loaded modules and corresponds to the current file, returning it if valid.

    When used in MONAI, likely in module paths ``monai.deploy.operators`` or ``monai.data``.
    """

    current_file = Path(__file__).resolve()

    # If a specific module name is provided (not None or "__main__"), validate it
    if module_name and module_name != "__main__":
        module = sys.modules.get(module_name)
        if module:
            module_file = getattr(module, "__file__", None)
            if module_file:
                try:
                    if Path(module_file).resolve() == current_file:
                        return module_name
                    else:
                        _logger.warning(f"Module {module_name} found but its file path does not match current file.")
                except (OSError, RuntimeError):
                    _logger.warning(f"Could not resolve file path for module {module_name}.")
            else:
                _logger.warning(f"Module {module_name} has no __file__ attribute.")
        else:
            _logger.warning(f"Module {module_name} not found in loaded modules.")
        # Fall through to search for the correct module

    # Search for modules that correspond to the current file
    candidates: list[str] = []

    for name, module in sys.modules.items():
        if not name or name == "__main__":
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            if Path(module_file).resolve() == current_file:
                candidates.append(name)
        except (OSError, RuntimeError):
            continue

    # Prefer nvimagecodec module paths over other candidates
    preferred_prefixes = ("nvidia.nvimgcodec.tools.dicom", "nvidia.nvimgcodec.tools", "nvidia.nvimgcodec")
    for prefix in preferred_prefixes:
        for name in candidates:
            if name.startswith(prefix):
                return name

    if candidates:
        # deterministic fallback
        return sorted(candidates)[0]

    return __name__


def _get_callable_origin(obj: Callable[..., Any]) -> tuple[str | None, str | None]:
    """Return the importable module path and attribute(function) name for *obj*.

    Can be used to get the importable module path and func name of existing callables.

    Args:
        obj: Callable retrieved via :func:`getattr` or similar.

    Returns:
        tuple[str | None, str | None]: ``(module_path, attr_name)``; each element
        is ``None`` if it cannot be determined. When both values are available,
        the same callable can be re-imported using
        :func:`importlib.import_module` followed by :func:`getattr`.
    """

    if not callable(obj):
        return None, None

    target = inspect.unwrap(obj)
    attr_name = getattr(target, "__name__", None)
    module = inspect.getmodule(target)
    module_path = getattr(module, "__name__", None)

    # If the callable is defined in a different module, find the attribute name in the module.
    if module_path and attr_name:
        module_obj = sys.modules.get(module_path)
        if module_obj and getattr(module_obj, attr_name, None) is not target:
            for name in dir(module_obj):
                try:
                    if getattr(module_obj, name) is target:
                        attr_name = name
                        break
                except AttributeError:
                    continue

    return module_path, attr_name
