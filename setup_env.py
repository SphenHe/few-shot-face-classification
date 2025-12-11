#!/usr/bin/env python3
"""Environment bootstrapper for few-shot face classification."""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

BASE_REQUIREMENTS = [
    ("facenet_pytorch", "facenet-pytorch>=2.5.2"),
    ("PIL", "Pillow>=9.0"),
    ("matplotlib", "matplotlib>=3.5"),
    ("sklearn", "scikit-learn>=1.0"),
    ("tqdm", "tqdm>=4.60"),
    ("numpy", "numpy>=1.21"),
]

TORCH_IMPORTS = ("torch", "torchvision")


def _run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def _is_installed(mod_name: str) -> bool:
    try:
        importlib.import_module(mod_name)
        return True
    except ImportError:
        return False


def _ensure_package(mod_name: str, spec: str, pip_args: list[str]) -> None:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"OK {mod_name} {ver}")
        return
    except ImportError:
        print(f"Installing {spec}...")
    _run([sys.executable, "-m", "pip", "install", spec, *pip_args])


def _ensure_torch(torch_spec: str, index_url: str | None) -> None:
    missing = [m for m in TORCH_IMPORTS if not _is_installed(m)]
    if not missing:
        versions = []
        for m in TORCH_IMPORTS:
            ver = getattr(importlib.import_module(m), "__version__", "unknown")
            versions.append(f"{m} {ver}")
        print("OK torch stack", " | ".join(versions))
        return

    cmd = [sys.executable, "-m", "pip", "install", *torch_spec.split()]
    if index_url:
        cmd += ["--index-url", index_url]
    print("Installing torch stack...")
    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install runtime dependencies and the local package.")
    parser.add_argument(
        "--torch-spec",
        default="torch torchvision",
        help="Pip spec used when installing torch (space separated).",
    )
    parser.add_argument(
        "--torch-index-url",
        default=None,
        help="Optional index URL for torch wheels (e.g. https://download.pytorch.org/whl/cu130).",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Pass --upgrade to pip installs.",
    )
    parser.add_argument(
        "--no-editable",
        action="store_true",
        help="Skip installing this repo in editable mode.",
    )
    args = parser.parse_args()

    if sys.version_info < (3, 8):
        sys.exit("Python 3.8+ is required.")
    print(f"Python {sys.version.split()[0]}")

    pip_args = ["--upgrade"] if args.upgrade else []

    _ensure_torch(args.torch_spec, args.torch_index_url)
    for mod_name, spec in BASE_REQUIREMENTS:
        _ensure_package(mod_name, spec, pip_args)

    if not args.no_editable:
        repo_root = Path(__file__).resolve().parent
        print("Installing project in editable mode...")
        _run([sys.executable, "-m", "pip", "install", "-e", str(repo_root), *pip_args])

    print("Environment ready.")


if __name__ == "__main__":
    main()
