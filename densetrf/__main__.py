import argparse
import json
from pathlib import Path

import torch
import yaml


def configuration(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    for key in ("steps", "image_size"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if "merge_every" in config and config["merge_every"] <= 0:
        raise ValueError("merge_every must be positive")
    # Paths are relative to the invoking directory, normally the repository root.
    for role in ("supervised", "unlabeled", "base"):
        if role in config:
            manifest = Path(config[role]["manifest"]).resolve()
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            config[role]["manifest"] = str(manifest)
    return config


def main():
    parser = argparse.ArgumentParser(description="DenseTRF: pretrained slots and two-branch target adaptation")
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads per process")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify the published base checkpoint")
    download.add_argument("--output", default="checkpoints/base")
    verify = commands.add_parser("verify", help="Verify all public base checkpoint hashes")
    verify.add_argument("--base", default="checkpoints/base")
    initialize = commands.add_parser("init-run", help="Initialize both clients from one verified base checkpoint")
    initialize.add_argument("--config", required=True)
    initialize.add_argument("--base", default="checkpoints/base")
    initialize.add_argument("--run", required=True)
    initialize.add_argument("--custom-base", action="store_true", help="Use a locally trained base; strict tensor loading still applies")
    server = commands.add_parser("server", help="Merge fresh updates from exactly the two named clients")
    server.add_argument("--run", required=True)
    server.add_argument("--timeout", type=float, default=3600)
    client = commands.add_parser("train", help="Run one supervised or unlabeled client")
    client.add_argument("--role", choices=["supervised", "unlabeled"], required=True)
    client.add_argument("--run", required=True)
    client.add_argument("--device", default="cpu")
    client.add_argument("--timeout", type=float, default=3600)
    pretrain = commands.add_parser("pretrain", help="Train fresh slot modules with a supplied foundation encoder")
    pretrain.add_argument("--config", required=True)
    pretrain.add_argument("--encoder", default="checkpoints/base/encoder.pth")
    pretrain.add_argument("--output", required=True)
    pretrain.add_argument("--device", default="cpu")
    evaluation = commands.add_parser("evaluate", help="Evaluate a checkpoint containing a trained dense head")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "download":
        from .checkpoints import download_base
        print(download_base(args.output))
    elif args.command == "verify":
        from .checkpoints import verify_base
        verify_base(args.base)
        print("All five checkpoint files verified")
    elif args.command == "init-run":
        from .federation import initialize_run
        print(json.dumps(initialize_run(configuration(args.config), args.run, args.base, verify=not args.custom_base), indent=2))
    elif args.command == "server":
        from .federation import serve
        serve(args.run, args.timeout)
    elif args.command == "train":
        from .training import train_client
        print(train_client(args.run, args.role, args.device, args.timeout))
    elif args.command == "pretrain":
        from .training import pretrain
        pretrain(configuration(args.config), args.encoder, args.output, args.device)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(args.checkpoint, args.manifest, args.output, args.device, seed=args.seed)


if __name__ == "__main__":
    main()
