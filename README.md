# DenseTRF

**DenseTRF: Texture-Aware Unsupervised Representation Adaptation for Surgical Scene Dense Prediction** — MICCAI 2026.

[Paper](https://arxiv.org/abs/2605.11265) · [Pretrained base model](https://huggingface.co/GuiqiuLiao/DenseTRF_base)

This standalone release contains the DINOv3 ViT-B/16 feature extractor, projection adapter, nine-slot attention model, feature reconstruction decoder, dense prediction head, and two-client adaptation workflow. It has no dependency on the original research repository and constructs only one representation model per client.

## Method and released checkpoint

1. Pretrain a representation with feature reconstruction on a broad Cholec/Thoracic/POEM image pool.
2. Initialize **both** adaptation clients from that same pretrained base.
3. Train a supervised task client with BCE + reconstruction (later BCE only), and an unlabeled target client with reconstruction only.
4. Merge their shared representation parameters with equal weights and load the result into both clients. The dense head remains local to the supervised client.

The base-pretraining client does **not** participate in adaptation merging. The checkpoint is a representation initializer; it does not contain a trained segmentation head. See [reproduction notes](docs/reproduction.md) for the inspected research settings, differences from the paper, and release-specific changes.

## Installation

Python 3.10 or newer is required. Install a PyTorch build appropriate for your CPU/CUDA environment first, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m densetrf --help
```

PyTorch, NumPy, Pillow, PyYAML, and huggingface-hub are the runtime dependencies. A minimal DINOv3 inference implementation is included with its original license. No separate foundation-model repository or weight download is needed.

## Download the base model

```bash
python -m densetrf download --output checkpoints/base
python -m densetrf verify --base checkpoints/base
```

The download is pinned to Hugging Face revision `56d71ef4f540def964c1df893edf9875cda17b58`. All five files are checked against SHA-256 hashes; the total download is about **355 MiB**. Alternatively, place `encoder.pth`, `adapter.pth`, `initializer.pth`, `processor.pth`, and `decoder.pth` together in a local directory and pass it with `--base`.

See [model manifest](docs/model_manifest.md). Downloaded checkpoints and experiment outputs are Git-ignored.

## Run the small demo

From the repository root:

```bash
python scripts/run_demo.py --base checkpoints/base --run outputs/demo
```

This starts one server and two CPU clients, performs two updates per client and one merge, and saves the supervised checkpoint. The example pool contains **four labeled CholecSeg8k training frames and four unlabeled target frames**, using the original 13 mask channels. Training examples come from the first 5% of sorted source clips; target examples come from the remaining 95%, following the supplied Cholec train/eval configurations. **Evaluation-source images are used only for adaptation**: no target masks or default scored evaluation are included. See [example provenance and data terms](examples/cholec/README.md). This short demo verifies execution, not segmentation quality. Use a new `--run` directory for every experiment; existing experiments are never overwritten.

For GPUs:

```bash
python scripts/run_demo.py --base checkpoints/base --run outputs/demo_gpu \
  --supervised-device cuda:0 --unlabeled-device cuda:1
```

The processes communicate using a shared local filesystem. They can share one GPU if sufficient memory is available. The demo writes approximately 2 GiB of full model snapshots; longer runs retain immutable round snapshots for inspection. Outputs can be removed after retaining the desired `supervised/latest.pt` and configuration.

### Latest training images and masks

Each client updates `outputs/<run>/<role>/latest_visuals/` during training:

- `summary.png`: paired input, predicted semantic mask, GT mask, and slot assignment for the supervised client. The unlabeled client shows input and slots only.
- `sample_000/` (and further samples): `input.png`, `prediction.png`, `ground_truth.png`, `slots.png`, `comparison.png`, `slot_probabilities.png` (all nine soft masks), and `per_class.png` (input/prediction/GT for each semantic class). Prediction/GT files are omitted when unavailable.
- `masks.npz` in each sample directory: soft slot masks, prediction probabilities, and available GT channels. These retain the independent BCE channels; the colored summary uses argmax for readability.
- `metadata.json`: sample IDs, zero-based step, one-based iteration, class names, and color legends.

Snapshots use the actual local forward pass from that iteration, before its optimizer update and server merge; no extra forward or random slot sampling is performed. They therefore need not match a subsequent merged checkpoint. The folder is replaced with the newest snapshot, including at the final iteration. `visualization_every` defaults to 1 (set to 0 to disable); `visualization_max_samples` defaults to 4 per batch. The unlabeled client never loads target GT for visualization. Base pretraining also writes inputs and slots under its output directory.

## Train on your own data

Create manifests following [the data format](docs/data_format.md), then edit a copy of `configs/thoracic_adaptation.yaml`. Paths in YAML are relative to the directory where you invoke the command. The initialization command resolves them before saving the run configuration.

```bash
python -m densetrf init-run --config configs/thoracic_adaptation.yaml \
  --base checkpoints/base --run outputs/thoracic_run1
```

Run each of these in a separate terminal:

```bash
python -m densetrf server --run outputs/thoracic_run1
python -m densetrf train --run outputs/thoracic_run1 --role supervised --device cuda:0
python -m densetrf train --run outputs/thoracic_run1 --role unlabeled --device cuda:1
```

The server requires exactly these two named roles and one fresh update from each in every round. All components, including the encoder, are merged; the dense head is excluded. Each process exits after the configured number of steps. A failed process causes peers to time out; restart in a new run directory. Automatic resume is not implemented.

```bash
python -m densetrf evaluate --checkpoint outputs/thoracic_run1/supervised/latest.pt \
  --manifest data/thoracic/evaluation.json --output outputs/thoracic_run1/evaluation \
  --device cuda:0
```

Evaluation writes one binary PNG per class and image, plus global-pixel IoU/Dice metrics. It uses sigmoid thresholding, includes every supplied class channel, and excludes absent-in-both channels from the mean. These metrics are not interchangeable with every paper's per-image averaging convention.

## Optional base pretraining

Provide a broad-image manifest and adjust `configs/base_pretraining.yaml`:

```bash
python -m densetrf pretrain --config configs/base_pretraining.yaml \
  --encoder checkpoints/base/encoder.pth --output outputs/new_base --device cuda:0
```

This initializes fresh adapter/slot/decoder modules and uses the supplied foundation encoder. It does not claim to reproduce the historical base checkpoint without its full data and training history. A locally trained base can initialize an experiment with `init-run --custom-base --base outputs/new_base`; this opts out of the published-file hash check, while retaining strict tensor loading.

## Tests

```bash
python -m unittest discover -s tests -v
DENSETRF_TEST_BASE=checkpoints/base python -m unittest discover -s tests -v
```

The second command also checks the real checkpoint, gradient routing, slot masks, and preservation of the local head. See [validation](docs/validation.md) for the tested environment and numerical comparison against the research implementation.

## Data and licensing

The default CholecSeg8k examples are included under their original CC BY-NC-SA 4.0 terms, with attribution and preprocessing provenance in [examples/cholec/README.md](examples/cholec/README.md). Training masks have 13 channels; adaptation masks are omitted. Synthetic examples remain available as an optional execution test through `configs/demo_synthetic.yaml`. Previously staged private Thoracic examples remain Git-ignored. See [examples](examples/README.md).

Original release code is MIT licensed. VideoSAUR-derived components retain their MIT attribution; vendored DINOv3 code and DINOv3-derived encoder weights retain the DINOv3 terms. The MIT license for the release code does not replace those terms. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{liao2026densetrf,
  title={DenseTRF: Texture-Aware Unsupervised Representation Adaptation for Surgical Scene Dense Prediction},
  author={Liao, Guiqiu and Jogan, Matjaž and Hashimoto, Daniel A.},
  journal={arXiv preprint arXiv:2605.11265},
  year={2026}
}
```
