## nvImageCodec OSS Contribution Rules

The nvImageCodec project is develop internally at NVIDIA. The source code of nvImageCodec are made available to the public regularly with each release. External contributions are welcome with the reservation that you must agree to the Developer Certificate of Origin presented below and certify that contribution is under the Apache License Version 2.0.

All enhancement, bugfix, or change requests must begin with the creation of a [nvImageCodec Issue Request](https://github.com/nvidia/nvImageCodec/issues).

### Optional dependency pinning policy

nvImageCodec declares the NVIDIA codec libraries it can use (nvJPEG2000, nvTIFF, nvCOMP) as
optional "extra" dependencies of the Python wheel. Their version ranges are pinned according to
the maturity of each library:

* **Pre-1.0 libraries** (currently nvJPEG2000 and nvTIFF) are pinned to the **minor** version,
  e.g. `>=0.12.0,<0.13`. Their API surface is still evolving and a minor release may add or change
  APIs we rely on, so a minor pin keeps the declared dependencies in step with what we build and
  test against.
* **Post-1.0 libraries** (currently nvCOMP) are pinned to the **major** version, e.g. `>=6,<7`,
  relying on semantic versioning for compatibility within a major.

When a dependency reaches 1.0, relax its pin from minor to major.

The pins live in the following places and must be kept in sync:

* `python/CMakeLists.txt` — `EXTRAS_REQUIRE`, the wheel's `extras_require` metadata (one block per
  CUDA major version and for the Tegra flavor).
* `test/requirements_*.txt` — the test requirements.
* `conda/recipes/libnvimgcodec/meta.yaml` — the conda dependencies. Note that nvJPEG2000 and
  nvTIFF are currently vendored into the package rather than taken from conda, so only nvCOMP is
  pinned here today; restore the pins on `libnvjpeg2k-dev`/`libnvtiff-dev` when those deps come
  back. The conda-forge feedstock carries its own copy of the recipe and must be updated there.
* The [installation guide](https://docs.nvidia.com/cuda/nvimagecodec/installation.html) of the
  nvImageCodec documentation, which lists the user-facing supported version ranges.

Bump the lower bounds whenever the version we build and test against changes.

### Developer Certificate of Origin

```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

  (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

  (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

  (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
