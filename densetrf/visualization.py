"""Save paired training outputs without another forward pass or any target-label access."""
import colorsys
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F


CHOLEC_COLORS = [(127, 127, 127), (210, 140, 140), (255, 114, 114), (231, 70, 156),
                (186, 183, 75), (170, 255, 0), (255, 85, 0), (255, 0, 0),
                (255, 255, 0), (169, 255, 184), (255, 160, 165), (0, 50, 128), (111, 74, 0)]


def palette(count, semantic=False):
    if semantic and count == 13:
        return np.asarray(CHOLEC_COLORS, dtype=np.uint8)
    return np.asarray([tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(
        (index * 0.61803398875) % 1, 0.7, 0.95)) for index in range(count)], dtype=np.uint8)


def panel(image, title, size=224):
    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    canvas = Image.new("RGB", (size, size + 28), (24, 24, 24))
    canvas.paste(image.convert("RGB").resize((size, size), Image.Resampling.NEAREST), (0, 28))
    ImageDraw.Draw(canvas).text((6, 7), title, fill="white")
    return canvas


def grid(panels, columns):
    width, height = panels[0].size
    canvas = Image.new("RGB", (columns * width, math.ceil(len(panels) / columns) * height), (24, 24, 24))
    for index, item in enumerate(panels):
        canvas.paste(item, ((index % columns) * width, (index // columns) * height))
    return canvas


def save_latest_visuals(directory, images, output, *, masks=None, sample_ids=None,
                        color_order="RGB", class_names=None, step=0, max_samples=4):
    """Capture this iteration's local forward, before its update/merge.

    Summary uses argmax colors for readability. Binary per-class comparisons and
    raw arrays preserve the independent BCE channels and soft slot probabilities.
    """
    if max_samples <= 0:
        return
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    count = min(len(images), max_samples)
    inputs = images[:count].detach().cpu().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    if color_order == "BGR":
        inputs = inputs[..., ::-1].copy()
    elif color_order != "RGB":
        raise ValueError("Unsupported visualization color order")
    slots = output["masks"][:count].detach().float().cpu()
    side = math.isqrt(slots.shape[-1])
    if side * side != slots.shape[-1]:
        raise ValueError("Expected a square slot-mask patch grid")
    slots = F.interpolate(slots.reshape(count, slots.shape[1], side, side),
                          size=images.shape[-2:], mode="bilinear", align_corners=False).numpy()
    probabilities = output["logits"][:count].detach().float().sigmoid().cpu().numpy() if "logits" in output else None
    truth = masks[:count].detach().float().cpu().numpy() if masks is not None else None
    slot_colors = palette(slots.shape[1])
    semantic_count = probabilities.shape[1] if probabilities is not None else (truth.shape[1] if truth is not None else 0)
    class_colors = palette(semantic_count, semantic=True)
    names = list(class_names or [f"class {i}" for i in range(semantic_count)])
    metadata = {"step": step, "iteration": step + 1, "snapshot": "local forward before optimizer update and server merge",
                "has_ground_truth": truth is not None, "has_prediction": probabilities is not None,
                "prediction_preview": "argmax of sigmoid probabilities; see per_class.png for independent BCE channels",
                "slot_preview": "argmax across resized soft slot masks; slot colors are not semantic classes",
                "classes": names, "class_colors": class_colors.tolist(), "slot_colors": slot_colors.tolist(), "samples": []}
    # Publish only a completed snapshot; the directory is a managed, replaceable output.
    with tempfile.TemporaryDirectory(prefix=".visuals-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        summary_rows = []
        for index in range(count):
            sample_dir = staging / f"sample_{index:03d}"
            sample_dir.mkdir()
            sample_id = str(sample_ids[index]) if sample_ids is not None else str(index)
            metadata["samples"].append({"id": sample_id, "directory": sample_dir.name})
            Image.fromarray(inputs[index]).save(sample_dir / "input.png")
            slot_rgb = slot_colors[slots[index].argmax(axis=0)]
            Image.fromarray(slot_rgb).save(sample_dir / "slots.png")
            row = [panel(inputs[index], f"Input {index} / iteration {step + 1}")]
            arrays = {"slot_probabilities": slots[index]}
            if probabilities is not None:
                prediction = class_colors[probabilities[index].argmax(axis=0)]
                Image.fromarray(prediction).save(sample_dir / "prediction.png")
                row.append(panel(prediction, "Prediction (argmax)"))
                arrays["prediction_probabilities"] = probabilities[index]
            if truth is not None:
                gt = class_colors[truth[index].argmax(axis=0)]
                gt[truth[index].sum(axis=0) == 0] = 0
                Image.fromarray(gt).save(sample_dir / "ground_truth.png")
                row.append(panel(gt, "Ground truth"))
                arrays["ground_truth"] = truth[index].astype(np.uint8)
            row.append(panel(slot_rgb, "Slot assignment"))
            comparison = grid(row, len(row))
            comparison.save(sample_dir / "comparison.png")
            summary_rows.append(comparison)
            heatmaps = [panel((slot * 255).round().astype(np.uint8), f"Slot {slot_id}", size=128)
                        for slot_id, slot in enumerate(slots[index])]
            grid(heatmaps, 3).save(sample_dir / "slot_probabilities.png")
            if probabilities is not None:
                channels = []
                for channel in range(semantic_count):
                    title = f"{channel}: {names[channel]}" if channel < len(names) else str(channel)
                    channels.append(panel(inputs[index], title, size=160))
                    predicted = (probabilities[index, channel] >= 0.5).astype(np.uint8) * 255
                    channels.append(panel(predicted, "Prediction >= 0.5", size=160))
                    if truth is not None:
                        channels.append(panel((truth[index, channel] * 255).astype(np.uint8), "Ground truth", size=160))
                grid(channels, 3 if truth is not None else 2).save(sample_dir / "per_class.png")
            np.savez_compressed(sample_dir / "masks.npz", **arrays)
        grid(summary_rows, 1).save(staging / "summary.png")
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        previous = directory.with_name(directory.name + ".previous")
        if previous.exists():
            shutil.rmtree(previous)
        if directory.exists():
            directory.rename(previous)
        try:
            staging.rename(directory)
        except Exception:
            if previous.exists():
                previous.rename(directory)
            raise
        if previous.exists():
            shutil.rmtree(previous)
