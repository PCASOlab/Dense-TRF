import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import yaml

from densetrf.data import ImageDataset


class CholecExamplesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.data = self.root / "examples/cholec"

    def test_disjoint_sources_and_adaptation_has_no_annotations(self):
        provenance = json.loads((self.data / "provenance.json").read_text())
        supervised = json.loads((self.data / "supervised.json").read_text())["samples"]
        unlabeled = json.loads((self.data / "unlabeled.json").read_text())["samples"]
        self.assertEqual((len(supervised), len(unlabeled)), (4, 4))
        self.assertEqual(provenance["training_clip_count"], 5)
        train_clips = {provenance["samples"][entry["id"]]["source_clip"] for entry in supervised}
        target_clips = {provenance["samples"][entry["id"]]["source_clip"] for entry in unlabeled}
        self.assertFalse(train_clips & target_clips)
        for entry in supervised:
            self.assertLess(int(provenance["samples"][entry["id"]]["source_clip"][5:11]), 5)
        for entry in unlabeled:
            self.assertGreaterEqual(int(provenance["samples"][entry["id"]]["source_clip"][5:11]), 5)
            self.assertEqual(set(entry), {"id", "image"})
        self.assertFalse(list(self.data.glob("unlabeled*.npz")))
        self.assertFalse((self.data / "evaluation.json").exists())
        hashes = [entry["image_sha256"] for entry in provenance["samples"].values()]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_13_class_masks_and_legacy_color_order(self):
        spec = json.loads((self.data / "supervised.json").read_text())
        dataset = ImageDataset(self.data / "supervised.json", labeled=True, size=256, num_classes=13)
        self.assertEqual(len(spec["classes"]), 13)
        for index, entry in enumerate(spec["samples"]):
            loaded = dataset[index]
            with Image.open(self.data / entry["image"]) as image:
                rgb = np.array(image)
            np.testing.assert_array_equal(loaded["image"].numpy(), rgb[..., ::-1].transpose(2, 0, 1))
            with np.load(self.data / entry["mask"], allow_pickle=False) as masks:
                np.testing.assert_array_equal(loaded["mask"].numpy(), masks["masks"])
            self.assertEqual(tuple(loaded["mask"].shape), (13, 256, 256))
        config = yaml.safe_load((self.root / "configs/demo.yaml").read_text())
        self.assertEqual(config["num_classes"], 13)
        self.assertEqual(config["image_size"], 224)
        self.assertEqual(config["unlabeled"]["manifest"], "examples/cholec/unlabeled.json")


if __name__ == "__main__":
    unittest.main()
