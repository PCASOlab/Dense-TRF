import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from densetrf.visualization import save_latest_visuals


class VisualizationTests(unittest.TestCase):
    def test_paired_outputs_preserve_colors_masks_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "latest_visuals"
            images = torch.zeros(1, 3, 8, 8)
            images[:, 2] = 255  # BGR red
            masks = torch.zeros(1, 2, 8, 8)
            masks[:, 0, :4] = 1
            masks[:, 1, 4:] = 1
            logits = torch.ones_like(masks)  # both BCE classes exceed the threshold
            slots = torch.tensor([[[1., 1., 0., 0.], [0., 0., 1., 1.]]])
            before = torch.get_rng_state().clone()
            save_latest_visuals(root, images, {"masks": slots, "logits": logits}, masks=masks,
                                color_order="BGR", sample_ids=["case-a"], class_names=["a", "b"], step=7)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            metadata = json.loads((root / "metadata.json").read_text())
            self.assertEqual(metadata["iteration"], 8)
            self.assertEqual(metadata["samples"][0]["id"], "case-a")
            with Image.open(root / "sample_000/input.png") as image:
                np.testing.assert_array_equal(np.asarray(image)[0, 0], [255, 0, 0])
            with np.load(root / "sample_000/masks.npz") as data:
                np.testing.assert_array_equal(data["ground_truth"], masks[0].numpy())
                np.testing.assert_allclose(data["prediction_probabilities"], logits[0].sigmoid().numpy())
                np.testing.assert_allclose(data["slot_probabilities"].sum(0), 1)
            self.assertTrue((root / "summary.png").is_file())
            self.assertTrue((root / "sample_000/per_class.png").is_file())

    def test_unlabeled_snapshot_has_no_gt_or_task_prediction_and_replaces_old_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "latest_visuals"
            images = torch.zeros(2, 3, 8, 8)
            slots = torch.full((2, 2, 4), 0.5)
            save_latest_visuals(root, images, {"masks": slots}, sample_ids=["a", "b"], step=0)
            self.assertTrue((root / "sample_001/input.png").is_file())
            save_latest_visuals(root, images[:1], {"masks": slots[:1]}, sample_ids=["last"], step=1)
            self.assertFalse((root / "sample_001").exists())
            self.assertFalse((root / "sample_000/ground_truth.png").exists())
            self.assertFalse((root / "sample_000/prediction.png").exists())
            with np.load(root / "sample_000/masks.npz") as data:
                self.assertEqual(data.files, ["slot_probabilities"])
            self.assertEqual(json.loads((root / "metadata.json").read_text())["step"], 1)


if __name__ == "__main__":
    unittest.main()
