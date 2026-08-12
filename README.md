# IPDE — Image Precision Data Extractor

IPDE extracts depth images, HDR gain maps, portrait/semantic mattes, and other
HEIF auxiliary images without display-oriented processing. The project has two
parts:

- a Python command-line extractor built on `pillow-heif >= 1.5.0`;
- a native Qt 6 desktop GUI that runs the extractor with structured JSON I/O.

## Precision model

IPDE preserves the values produced by the HEIF decoder. It does not apply gamma
correction, tone mapping, color enhancement, range stretching, or normalization.

There is an important distinction between *decoded-sample preservation* and the
original scene:

- HEVC auxiliary images may have been encoded lossily by the camera. No decoder
  can reconstruct information that the original encoding discarded.
- A depth plane may store uniform depth, inverse depth, disparity, or a nonlinear
  representation. The numeric plane alone is not necessarily distance in metres.
  IPDE preserves the depth-representation metadata in the JSON manifest.
- `pillow-heif` can expand 10/12-bit codes into a 16-bit display range. IPDE
  explicitly disables that behavior with `hdr_to_16bit=False`; a decoded 12-bit
  value of `1` remains `1`, not `16`.
- `pillow-heif 1.5` inventories high-bit auxiliary items but rejects decoding
  them. For those items only, IPDE calls the same libheif bundled in the wheel,
  requests 10/12-bit output in unshifted `uint16` storage, verifies every code is
  within the source range, and records this fallback in the manifest. No 8-bit
  intermediary is created.

Every extractable plane gets two outputs by default:

1. A lossless exchange file: 8/16-bit PNG for unsigned integer planes, or
   losslessly compressed OpenEXR for `float16`/`float32` planes.
2. A NumPy `.npy` file containing the exact dtype, shape, byte order, and decoded
   sample bits.

A `<source>_aux_manifest.json` records source/output SHA-256 hashes, dimensions,
dtype, source bit depth, decoder settings, depth semantics, auxiliary URNs, and
raw per-asset metadata blocks. PNG and EXR files are read back and compared before
they are committed. Outputs are written through same-directory temporary files
and existing files are never replaced unless `--overwrite` is requested.

## Requirements

- macOS with Python 3.14 at
  `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14`
- Qt 6.11.1 at `/opt/Qt/6.11.1/macos`
- CMake 3.24+

Create the local environment:

```sh
make setup
```

This also installs an `ipde` console command in `.venv/bin`.

The core dependencies are NumPy, [pillow-heif](https://pillow-heif.readthedocs.io/),
and the official [OpenEXR Python module](https://openexr.com/en/latest/python.html).
OpenEXR is only exercised when a floating-point plane is encountered, but it is
installed by `make setup` so that such a plane can never be silently downgraded.

## Command line

Inspect a file without writing anything:

```sh
.venv/bin/python ipde_extract.py --inspect portrait.heic
```

Extract beside the source image:

```sh
.venv/bin/python ipde_extract.py portrait.heic
```

Extract several files to a chosen directory:

```sh
.venv/bin/python ipde_extract.py --output-dir /path/to/output photo1.heic photo2.heic
```

Useful options:

```text
--inspect       Inventory and decode auxiliary planes without writing files
--json          Emit one machine-readable JSON object per input file
--overwrite     Atomically replace colliding output files
--no-npy        Omit exact-array companions (PNG/EXR remain verified and lossless)
```

The process returns nonzero if any input fails. In JSON mode, errors are emitted
as JSON objects as well, which is what the GUI consumes.

## Qt GUI

Build and launch against the supplied Qt installation:

```sh
make gui
```

Files may be added with the picker or drag-and-drop. IPDE inventories them first,
showing every depth, auxiliary, and alpha plane with its dimensions, dtype, and
source bit depth. Extraction runs one source at a time to keep peak memory use
bounded. The output folder may be left blank to write beside each source.

Build without launching:

```sh
make build
```

The app bundle is `build/IPDE.app`. The Python extractor and package are copied
into the bundle resources, while the configured Python runtime remains external.

## Validation

```sh
make test
make smoke
```

The test suite covers exact 8/16-bit PNG round trips, exact NPY round trips,
floating-point EXR round trips when OpenEXR is installed, metadata preservation,
collision refusal, auxiliary enumeration, and CLI behavior.

## Output naming

For a typical Apple portrait named `IMG_0001.HEIC`, outputs look like:

```text
IMG_0001_depth.png
IMG_0001_depth.npy
IMG_0001_hdr_gain_map.png
IMG_0001_hdr_gain_map.npy
IMG_0001_hdr_gain_map_metadata0.xmp
IMG_0001_aux_manifest.json
```

Indices are added when a container has multiple top-level images or repeated
auxiliary types. Auxiliary metadata sidecars preserve their original bytes; the
manifest also contains a base64 copy and SHA-256 digest.
