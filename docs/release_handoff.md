# Release handoff

`Opensource/` is the standalone repository root. The research project outside it was not changed by the extraction. The public files occupy approximately **2.63 MiB**, including the 1.4 MiB paper, CholecSeg8k examples, and optional synthetic fixtures. Downloaded model weights, private Thoracic samples, caches, build products, and local upload staging are excluded by `.gitignore`.

## Files to review

- `README.md`: installation, download, demo, training, evaluation, citation.
- `docs/reproduction.md`: actual two-client procedure and differences from the paper/research scripts.
- `docs/model_manifest.md`: pinned model revision and hashes.
- `LICENSE` and `THIRD_PARTY_NOTICES.md`: MIT for original release code; upstream terms preserved separately.
- `examples/private_thoracic/`: ten locally staged clinical examples, currently ignored. Select public redistribution terms before adding them to Git. The runnable public demo uses the publicly included CholecSeg8k examples under their original data terms.
- `upload_artifacts/huggingface_README.md` and `upload_artifacts/DINOv3_LICENSE.md`: proposed metadata/license additions for the existing Hugging Face model repository. No additional model weights need uploading.

The local sample-source mapping in `upload_artifacts/private_sample_sources.json` contains original paths and must remain private.

## GitHub publication

The destination `https://github.com/PCASOlab/Dense-TRF` already has commit history. Preserve it by cloning into a new directory and copying the release files there, honoring this folder's `.gitignore`. Do not force-push or import the parent research repository's history.

One way to do this after reviewing the release is:

```bash
git clone https://github.com/PCASOlab/Dense-TRF.git /path/to/new/Dense-TRF
python /path/to/Opensource/scripts/export_release.py --list
python /path/to/Opensource/scripts/export_release.py --output /path/to/new/Dense-TRF
cd /path/to/new/Dense-TRF
git status --short
git add .
git diff --cached --stat
# Review the staged paths before committing and pushing.
git commit -m "Release standalone DenseTRF training and pretrained model loader"
git push origin main
```

The exporter uses Git's actual ignore rules, leaves the destination `.git` directory intact, and replaces matching release files (including the starter README). It does not delete unrelated destination files. The model weights stay on Hugging Face. GitHub publication uses a separate checkout so that the research repository and its history remain unchanged. Hugging Face metadata changes are a separate manual step.
