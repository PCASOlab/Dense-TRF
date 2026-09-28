"""Export a small CholecSeg8k demo from trusted legacy clip pickles.

Mirrors the train/eval configurations' sorted 5%/95% split explicitly.
Evaluation-source images are exported for unlabeled adaptation only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

CLASSES = ["Black Background", "Abdominal Wall", "Liver", "Gastrointestinal Tract",
           "Fat", "Grasper", "Connective Tissue", "Blood", "Cystic Duct",
           "L-hook Electrocautery", "Gallbladder", "Hepatic Vein", "Liver Ligament"]
COLORS = [(127, 127, 127), (210, 140, 140), (255, 114, 114), (231, 70, 156),
          (186, 183, 75), (170, 255, 0), (255, 85, 0), (255, 0, 0),
          (255, 255, 0), (169, 255, 184), (255, 160, 165), (0, 50, 128), (111, 74, 0)]


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if np.__version__.split(".")[0] == "1":
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def export(source, output, count=4, fraction=0.05):
    files = sorted(Path(source).glob("*.pkl"))
    boundary = int(len(files) * fraction)
    pools = {"supervised": files[:boundary], "unlabeled": files[boundary:]}
    if count <= 0 or any(len(pool) < count for pool in pools.values()):
        raise ValueError("Need enough clips in both partitions for the requested sample count")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"source_dataset": "CholecSeg8k", "source_clip_count": len(files),
                  "training_clip_count": boundary, "adaptation_clip_count": len(files) - boundary,
                  "split": "sorted clip names: first 5% supervised, remaining 95% adaptation",
                  "train_config": "working_para/working_dir_root_train_cholec_super.py",
                  "adaptation_source_config": "working_para/working_dir_root_eval_cholec_super.py",
                  "frame_index": 0, "samples": {}}
    hashes = set()
    for role, pool in pools.items():
        entries = []
        for index, clip_index in enumerate(np.linspace(0, len(pool) - 1, count, dtype=int)):
            path = pool[clip_index]
            with path.open("rb") as stream:
                sample = LegacyUnpickler(stream).load()
            frames = np.asarray(sample["frames"])
            if frames.ndim != 4 or frames.shape[0] != 3 or frames.dtype != np.uint8:
                raise ValueError(f"Expected uint8 3xTxHxW frames: {path.name}")
            # Legacy cv2.imread stored BGR; PNGs must be RGB for normal viewing.
            bgr = frames[:, 0].transpose(1, 2, 0)
            rgb = bgr[..., ::-1].copy()
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError("Duplicate image across chosen sample frames")
            hashes.add(digest)
            name = f"{role}_{index:03d}"
            Image.fromarray(rgb).save(output / f"{name}.png")
            entry = {"id": name, "image": f"{name}.png"}
            if role == "supervised":
                masks = np.asarray(sample["labels"])[:, 0]
                if masks.shape != (13, *bgr.shape[:2]) or not np.isin(masks, [0, 1]).all():
                    raise ValueError("Expected 13 binary CholecSeg8k mask channels")
                np.savez_compressed(output / f"{name}_masks.npz", masks=masks.astype(np.uint8))
                entry["mask"] = f"{name}_masks.npz"
                # RGB preview is for humans; training uses all original binary channels.
                preview = np.asarray(COLORS, dtype=np.uint8)[masks.argmax(axis=0)]
                preview[masks.sum(axis=0) == 0] = 0
                Image.fromarray(preview).save(output / f"{name}_mask_preview.png")
            # Do not read or export the adaptation examples' label arrays.
            entries.append(entry)
            provenance["samples"][name] = {"source_clip": path.name, "frame": 0, "image_sha256": digest}
        manifest = {"color_order": "BGR", "classes": CLASSES, "samples": entries}
        (output / f"{role}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", default=Path("examples/cholec"), type=Path)
    parser.add_argument("--count", default=4, type=int)
    args = parser.parse_args()
    export(args.source, args.output, args.count)


if __name__ == "__main__":
    main()
