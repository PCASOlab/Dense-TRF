"""Copy the Git-visible release files into a checkout, preserving its .git directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def public_files(root):
    # A temporary index avoids the parent research repository's Opensource/ ignore rule.
    # The actual release tree is read-only during enumeration.
    with tempfile.TemporaryDirectory(prefix="densetrf-export-index-") as temporary:
        subprocess.run(["git", "init", "-q", temporary], check=True)
        listing = subprocess.check_output([
            "git", f"--git-dir={temporary}/.git", f"--work-tree={root}",
            "ls-files", "--others", "--exclude-standard", "-z",
        ])
    return [Path(name) for name in listing.decode().split("\0") if name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Destination checkout; matching release files are replaced")
    parser.add_argument("--list", action="store_true", help="List public files and total size without copying")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = public_files(root)
    if not args.list and args.output is None:
        parser.error("Specify --output or --list")
    destination = args.output.resolve() if args.output else None
    if destination is not None and (destination == root or root in destination.parents):
        parser.error("Destination must be outside the release source directory")
    for relative in files:
        if args.list:
            print(relative)
        elif destination is not None:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
    size = sum((root / relative).stat().st_size for relative in files)
    print(f"{len(files)} public files, {size} bytes ({size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
