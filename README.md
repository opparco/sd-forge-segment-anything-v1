# sd-forge-segment-anything-v1

Segment Anything v1 extension for Stable Diffusion WebUI Forge and Forge-based distributions.

This extension adds a point-based SAM v1 mask generator tab. You can place foreground and background points on a reference image, generate a mask, adjust basic mask postprocessing, and export the result to Forge's inpaint canvas.

## Features

- SAM v1 checkpoint selection from `models/SAM`
- Automatic model type detection for `vit_b`, `vit_l`, and `vit_h` checkpoint filenames
- Foreground/background point prompts on the reference image
- Automatic mask preview generation after placing points
- Mask postprocessing with dilate, erode, and selectable operation order
- Parameter import/export as JSON
- Export generated masks to the Forge inpaint canvas
- CPU, CUDA, and auto device selection

## Requirements

- Stable Diffusion WebUI Forge or a Forge-based distribution
- Python environment used by the WebUI
- A SAM v1 checkpoint (`.pth` or `.pt`)

The extension installs the `segment-anything` Python package through `install.py` unless the WebUI is started with `--skip-install`.

## Installation

Clone this repository into your WebUI `extensions` directory:

```bash
git clone https://github.com/opparco/sd-forge-segment-anything-v1.git extensions/sd-forge-segment-anything-v1
```

Restart the WebUI after installation.

## Model Setup

Download SAM v1 checkpoints from the official Segment Anything repository:

```text
https://github.com/facebookresearch/segment-anything
```

Place SAM v1 checkpoint files in:

```text
models/SAM
```

Supported checkpoint filename hints:

- `vit_b`
- `vit_l`
- `vit_h`

If the model type cannot be inferred from the filename, select the model type explicitly in the UI.

## Usage

1. Open the `Segment Anything v1` tab.
2. Load a reference image.
3. Select a SAM checkpoint.
4. Left-click the image to add foreground points.
5. Right-click the image to add background points.
6. Adjust postprocessing if needed.
7. Use `Export to inpaint` to send the reference image and generated mask to Forge inpaint.

## Postprocessing

The `Postprocess` panel provides:

- `Dilate mask`: expands the generated mask.
- `Erode mask`: shrinks the generated mask.
- `Order`: chooses whether to run `Dilate then erode` or `Erode then dilate`.

The default order is `Dilate then erode`, matching the original behavior.

## Parameter JSON

The `Parameters` panel can save and load:

- checkpoint selection
- model type
- device
- multimask setting
- postprocess settings
- foreground/background points

This is useful when repeating the same mask workflow across sessions.

## Notes

- This extension targets SAM v1 only. It does not provide SAM 2 support.
- The inpaint export path depends on Forge's canvas implementation.
- Large SAM checkpoints may require significant VRAM when using CUDA.
