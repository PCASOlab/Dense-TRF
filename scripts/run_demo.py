"""Launch a bounded, fail-fast two-client demo on CPU or specified GPUs."""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="checkpoints/base")
    parser.add_argument("--run", default="outputs/demo")
    parser.add_argument("--supervised-device", default="cpu")
    parser.add_argument("--unlabeled-device", default="cpu")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--evaluation-manifest", help="Optional separate labeled evaluation set; omitted by default")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    prefix = [sys.executable, "-m", "densetrf"]
    subprocess.run(prefix + ["init-run", "--config", args.config, "--base", args.base, "--run", args.run], check=True)
    commands = [["server", "--run", args.run, "--timeout", str(args.timeout)]]
    for role, device in [("supervised", args.supervised_device), ("unlabeled", args.unlabeled_device)]:
        commands.append(["train", "--run", args.run, "--role", role, "--device", device, "--timeout", str(args.timeout)])
    processes = []
    try:
        processes = [subprocess.Popen(prefix + command) for command in commands]
        deadline = time.monotonic() + args.timeout
        while any(process.poll() is None for process in processes):
            for process in processes:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"Demo process failed with exit code {process.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("Demo exceeded its time limit")
            time.sleep(0.2)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("A demo process failed")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    checkpoint = Path(args.run) / "supervised/latest.pt"
    if args.evaluation_manifest:
        subprocess.run(prefix + ["evaluate", "--checkpoint", str(checkpoint),
                                "--manifest", args.evaluation_manifest, "--output", str(Path(args.run) / "evaluation"),
                                "--device", args.supervised_device], check=True)
    else:
        print(f"Demo complete. Supervised checkpoint: {checkpoint}")
        print("No scored evaluation: target examples are used only for unlabeled adaptation.")


if __name__ == "__main__":
    main()
