from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path

import gradio as gr
import cv2
import numpy as np
import torch
from PIL import Image

from modules import paths, script_callbacks

TITLE = "Segment Anything v1"
SAM_MODEL_DIR = Path(paths.models_path) / "SAM"


class CachedPredictor:
    def __init__(self, predictor: object) -> None:
        self.predictor = predictor
        self.image_hash: str | None = None


_predictor_cache: dict[tuple[str, str, str], CachedPredictor] = {}
_model_mapping: dict[str, str] = {}


def _scan_models() -> dict[str, str]:
    found: dict[str, str] = {}
    if not SAM_MODEL_DIR.exists():
        return found
    for path in sorted([*SAM_MODEL_DIR.glob("*.pth"), *SAM_MODEL_DIR.glob("*.pt")]):
        found[path.name] = str(path)
    return found


def refresh_models() -> gr.Dropdown:
    global _model_mapping
    _model_mapping = _scan_models()
    choices = list(_model_mapping)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _resolve_model_path(model_choice: str | None) -> Path:
    if model_choice and model_choice in _model_mapping:
        return Path(_model_mapping[model_choice])
    if model_choice:
        return Path(model_choice)
    raise FileNotFoundError(f"No SAM checkpoint selected. Put SAM v1 checkpoints in {SAM_MODEL_DIR}")


def _infer_model_type(model_type: str, checkpoint: Path) -> str:
    if model_type != "auto":
        return model_type
    name = checkpoint.name.lower()
    if "vit_h" in name:
        return "vit_h"
    if "vit_l" in name:
        return "vit_l"
    if "vit_b" in name:
        return "vit_b"
    raise ValueError("Could not infer SAM model type from filename. Select vit_b, vit_l, or vit_h explicitly.")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _image_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(str(rgb.size).encode("utf-8"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _get_predictor(checkpoint: Path, model_type: str, device: str) -> CachedPredictor:
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as exc:
        raise RuntimeError("segment-anything is not installed. Restart Forge so the extension install.py can run, or install it manually.") from exc

    key = (str(checkpoint.resolve()), model_type, device)
    cached = _predictor_cache.get(key)
    if cached is not None:
        return cached

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    cached = CachedPredictor(predictor=SamPredictor(sam))
    _predictor_cache[key] = cached
    return cached


def _parse_points(points_json: str, labels_json: str) -> tuple[np.ndarray, np.ndarray]:
    points = json.loads(points_json or "[]")
    labels = json.loads(labels_json or "[]")
    if len(points) != len(labels):
        raise ValueError("input_points and input_labels lengths differ.")
    if not points:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _postprocess_mask(mask: np.ndarray, dilate_pixels: int, erode_pixels: int) -> np.ndarray:
    processed = mask.astype(np.uint8)
    if dilate_pixels > 0:
        kernel_size = dilate_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        processed = cv2.dilate(processed, kernel, iterations=1)
    if erode_pixels > 0:
        kernel_size = erode_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        processed = cv2.erode(processed, kernel, iterations=1)
    return processed.astype(bool)


def _overlay_image(image: Image.Image, mask: np.ndarray, points: np.ndarray, labels: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_alpha = Image.fromarray((mask.astype(np.uint8) * 120), mode="L")
    color = Image.new("RGBA", base.size, (56, 189, 248, 0))
    color.putalpha(mask_alpha)
    overlay.alpha_composite(color)

    arr = np.asarray(overlay).copy()
    h, w = mask.shape
    for (x, y), label in zip(points.astype(int), labels.astype(int), strict=False):
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        radius = max(4, int(min(w, h) * 0.008))
        yy, xx = np.ogrid[:h, :w]
        dot = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        arr[dot] = (34, 197, 94, 255) if label == 1 else (239, 68, 68, 255)
    overlay = Image.fromarray(arr, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def clear_points() -> tuple[str, str, None, None, str]:
    return "[]", "[]", None, None, "Cleared points."


def generate_mask(
    image: Image.Image | None,
    points_json: str,
    labels_json: str,
    model_choice: str | None,
    model_type: str,
    device: str,
    multimask_output: bool,
    dilate_pixels: int,
    erode_pixels: int,
) -> tuple[Image.Image | None, Image.Image | None, str]:
    if image is None:
        return None, None, "Load a reference image first."

    try:
        points, labels = _parse_points(points_json, labels_json)
        if len(points) == 0:
            return image.convert("RGB"), None, "Add at least one foreground/background point."

        checkpoint = _resolve_model_path(model_choice)
        if not checkpoint.exists():
            return None, None, f"Model not found: {checkpoint}"

        resolved_type = _infer_model_type(model_type, checkpoint)
        resolved_device = _resolve_device(device)
        image = image.convert("RGB")

        cached = _get_predictor(checkpoint, resolved_type, resolved_device)
        current_hash = _image_hash(image)
        if cached.image_hash != current_hash:
            cached.predictor.set_image(np.asarray(image))
            cached.image_hash = current_hash

        masks, scores, _ = cached.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output,
        )
        best_index = int(np.argmax(scores))
        mask = _postprocess_mask(masks[best_index], int(dilate_pixels), int(erode_pixels))

        preview = _overlay_image(image, mask, points, labels)
        status = (
            f"Generated mask from {len(points)} point(s). Score: {float(scores[best_index]):.4f}"
            f"\nPostprocess: dilate {int(dilate_pixels)} px, erode {int(erode_pixels)} px."
        )
        return preview, _mask_image(mask), status
    except Exception as exc:
        traceback.print_exc()
        return None, None, f"Error: {exc}"


def on_ui_tabs():
    refresh_models()

    with gr.Blocks() as tab:
        with gr.Row():
            with gr.Column(scale=1):
                ref_image = gr.Image(label="Reference image", type="pil", elem_id="segment_anything_ref_image")
                input_points = gr.Textbox(label="input_points", value="[]", elem_id="segment_anything_input_points")
                input_labels = gr.Textbox(label="input_labels", value="[]", elem_id="segment_anything_input_labels")
                with gr.Row():
                    clear_button = gr.Button("Clear points", elem_id="segment_anything_clear_points")
                    generate_button = gr.Button("Generate mask", variant="primary", elem_id="segment_anything_generate_mask")
                    auto_generate_button = gr.Button(
                        "Auto generate mask",
                        elem_id="segment_anything_auto_generate_mask",
                        elem_classes=["segment-anything-hidden-control"],
                    )

            with gr.Column(scale=1):
                preview = gr.Image(label="Segment preview", type="pil")
                mask = gr.Image(label="Mask", type="pil")
                status = gr.Textbox(label="Status", interactive=False, lines=3)

        with gr.Accordion("SAM model", open=True):
            with gr.Row():
                model_choice = gr.Dropdown(choices=list(_model_mapping), value=next(iter(_model_mapping), None), label="Checkpoint")
                model_type = gr.Dropdown(choices=["auto", "vit_b", "vit_l", "vit_h"], value="auto", label="Model type")
                device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="auto", label="Device", allow_custom_value=True)
            with gr.Row():
                refresh_button = gr.Button("Refresh models")
                multimask_output = gr.Checkbox(label="Use SAM multimask output", value=True)

        with gr.Accordion("Postprocess", open=False):
            with gr.Row():
                dilate_pixels = gr.Slider(
                    minimum=0,
                    maximum=128,
                    value=0,
                    step=1,
                    label="Dilate mask",
                )
                erode_pixels = gr.Slider(
                    minimum=0,
                    maximum=128,
                    value=0,
                    step=1,
                    label="Erode mask",
                )

        mask_inputs = [
            ref_image,
            input_points,
            input_labels,
            model_choice,
            model_type,
            device,
            multimask_output,
            dilate_pixels,
            erode_pixels,
        ]
        generate_button.click(generate_mask, inputs=mask_inputs, outputs=[preview, mask, status])
        auto_generate_button.click(generate_mask, inputs=mask_inputs, outputs=[preview, mask, status])
        clear_button.click(clear_points, outputs=[input_points, input_labels, preview, mask, status])
        refresh_button.click(refresh_models, outputs=[model_choice])

    return ((tab, TITLE, "segment_anything_v1"),)


script_callbacks.on_ui_tabs(on_ui_tabs)
