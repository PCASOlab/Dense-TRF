import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import unittest
import tempfile
import torch

from densetrf.checkpoints import load_base, atomic_save
from densetrf.data import ImageDataset
from densetrf.federation import average_states, serve
from densetrf.models.network import DenseTRF
from densetrf.training import objective

torch.set_num_threads(2)

def test_unlabeled_loader_never_opens_masks(tmp_path):
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "image.png")
    manifest = tmp_path / "data.json"
    manifest.write_text(json.dumps({"samples": [{"id": "x", "image": "image.png", "mask": "does-not-exist.npz"}]}))
    batch = ImageDataset(manifest, labeled=False)[0]
    assert "mask" not in batch
    with unittest.TestCase().assertRaises(FileNotFoundError):
        ImageDataset(manifest, labeled=True)[0]


def test_averaging_exact_and_rejects_invalid_updates():
    left = {name: {"weight": torch.tensor([1., 3.]), "buffer": torch.tensor(2)} for name in DenseTRF.components}
    right = {name: {"weight": torch.tensor([3., 7.]), "buffer": torch.tensor(2)} for name in DenseTRF.components}
    averaged = average_states([left, right])
    for component in DenseTRF.components:
        torch.testing.assert_close(averaged[component]["weight"], torch.tensor([2., 5.]))
        assert averaged[component]["buffer"].dtype == torch.int64
    with unittest.TestCase().assertRaises(ValueError):
        average_states([left])
    right["encoder"]["weight"][0] = float("nan")
    with unittest.TestCase().assertRaises(ValueError):
        average_states([left, right])


def test_supervised_loss_schedule_and_unsupervised_loss():
    config = {"reconstruction_weight": 0.1, "reconstruction_until": 400}
    reconstruction = torch.tensor(2., requires_grad=True)
    logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
    output = {"reconstruction_loss": reconstruction, "logits": logits}
    labels = torch.ones_like(logits)
    joint, _ = objective(output, labels, 399, config)
    joint.backward()
    unittest.TestCase().assertAlmostEqual(reconstruction.grad.item(), 0.1)
    reconstruction.grad = None
    supervised, _ = objective(output, labels, 400, config)
    supervised.backward()
    assert reconstruction.grad is None
    unsupervised, _ = objective(output, None, 999, config)
    assert unsupervised is reconstruction


def pretrained():
    directory = os.environ.get("DENSETRF_TEST_BASE")
    if not directory:
        raise unittest.SkipTest("Set DENSETRF_TEST_BASE to run real-checkpoint tests")
    model = DenseTRF(2)
    load_base(model, directory)
    return model


def test_real_checkpoint_forward_and_gradients(pretrained):
    model = pretrained
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(torch.zeros(1, 3, 128, 128), head_detach=True)
    assert output["masks"].shape == (1, 9, 196)
    torch.testing.assert_close(output["masks"].sum(1), torch.ones(1, 196))
    torch.nn.functional.binary_cross_entropy_with_logits(output["logits"], torch.zeros_like(output["logits"])).backward()
    assert model.dense_head.net[0].weight.grad is not None
    assert model.adapter.fusion_layer[0].weight.grad is None
    model.zero_grad(set_to_none=True)
    output = model(torch.zeros(1, 3, 128, 128))
    output["reconstruction_loss"].backward()
    assert model.adapter.fusion_layer[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert not hasattr(model, "model_s")
    assert len(model.encoder.encoders) == 1


def test_shared_load_preserves_local_head_and_encoder_aliases(pretrained):
    model = pretrained
    before = model.dense_head.net[0].weight.detach().clone()
    shared = model.shared_state()
    model.load_shared(shared)
    torch.testing.assert_close(before, model.dense_head.net[0].weight)
    state = shared["encoder"]
    assert state["encoders.0.dinov3.cls_token"].data_ptr() == state["encoders.0.image_encoder.cls_token"].data_ptr()


class CoreTests(unittest.TestCase):
    def test_unlabeled(self):
        with tempfile.TemporaryDirectory() as directory:
            test_unlabeled_loader_never_opens_masks(Path(directory))

    def test_average(self):
        test_averaging_exact_and_rejects_invalid_updates()

    def test_losses(self):
        test_supervised_loss_schedule_and_unsupervised_loss()

class ServerTests(unittest.TestCase):
    def test_fresh_updates_and_wrong_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(json.dumps({"id": "run-a", "participants": ["supervised", "unlabeled"], "rounds": 1}))
            state = {name: {"weight": torch.tensor([1.])} for name in DenseTRF.components}
            packet = {"run_id": "run-a", "round": 0, "role": "supervised", "shared": state}
            atomic_save(packet, root / "supervised/000000.pt")
            with self.assertRaises(TimeoutError):
                serve(root, timeout=0.01)
            self.assertFalse((root / "global/000000.pt").exists())
            self.assertFalse((root / "server.lock").exists())
            packet.update(role="unlabeled", round=1)
            packet["shared"] = {name: {"weight": torch.tensor([3.])} for name in DenseTRF.components}
            atomic_save(packet, root / "unlabeled/000000.pt")
            with self.assertRaises(ValueError):
                serve(root, timeout=0.01)
            packet["round"] = 0
            atomic_save(packet, root / "unlabeled/000000.pt")
            serve(root, timeout=0.01)
            result = torch.load(root / "global/000000.pt", weights_only=True)
            self.assertEqual(result["run_id"], "run-a")
            self.assertNotIn("head", result["shared"])
            torch.testing.assert_close(result["shared"]["adapter"]["weight"], torch.tensor([2.]))


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = pretrained()

    def test_gradients(self):
        test_real_checkpoint_forward_and_gradients(self.model)

    def test_shared_state(self):
        test_shared_load_preserves_local_head_and_encoder_aliases(self.model)

if __name__ == "__main__":
    unittest.main()
