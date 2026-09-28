# Pretrained model

- Repository: https://huggingface.co/GuiqiuLiao/DenseTRF_base
- Pinned revision: `56d71ef4f540def964c1df893edf9875cda17b58`
- Provenance: author's client-11 backup `base_proj_mix_cho_tho_poem`.
- Architecture: DINOv3 ViT-B/16, 9 slots, 256-dimensional projection/slots, 196 patches, 768-dimensional reconstruction.
- Scope: base representation, no trained task head and no optimizer state.

| File | Bytes | SHA-256 |
|---|---:|---|
| adapter.pth | 6309760 | c215de10e0d15439dfed4ff864fabc5e474c29ba488fcb08c40072035172549f |
| decoder.pth | 17005493 | 50837d605f19266f8f3b75f055e39731222c7b9d963e56c865bced12e431ddd8 |
| encoder.pth | 346313266 | a781ed8c10f608a2a05ed25c432b90b4a48756e647c3b38af446586fa3fc0520 |
| initializer.pth | 3632 | a4324b975b31c35fe1287fe3274f264a967ed1d9c1c9a759185a6226fc55cb17 |
| processor.pth | 2374035 | eb0ea4e51885246cae2d99b0f2431bc80067d19234d57a62a86732b2277c5567 |

These hashes match both the local backup and Hugging Face LFS metadata. The runtime source of truth is `densetrf/base_manifest.json`. Updating the model requires updating the pinned revision and hashes intentionally. `status.txt` from the historical client is not downloaded or used.

All necessary base-model weights are already uploaded. The encoder file includes the DINOv3 backbone, so no other foundation checkpoint is required. The duplicate `dinov3`/`image_encoder` keys refer to the same backbone in the original wrapper; loading preserves this alias and allocates one backbone. Unused projection parameters inside the wrapper are retained for strict checkpoint compatibility.

If a future ready-to-use segmentation model is released, place it in the ignored `upload_artifacts/` directory first, with its task class order, shared representation, dense head, preprocessing, provenance, and evaluation settings. Such a task model is optional for this release.

The encoder incorporates DINOv3 materials. Keep the upstream terms with it; the Hugging Face repository's MIT tag alone does not describe every bundled component's license. A suggested expanded model card and upstream license are staged in `upload_artifacts/` for the owner to review and upload.
