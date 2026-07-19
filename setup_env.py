#!/usr/bin/env python3
"""Create a Conda or venv environment and install this project."""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union


def _run(cmd):
    print(">>", shlex.join(cmd))
    subprocess.check_call(cmd)


def _is_conda_env(conda: str, env_name: str) -> bool:
    result = subprocess.run(
        [conda, "env", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    return any(line.split() and line.split()[0] == env_name for line in result.stdout.splitlines())


def _install_with_python(
    python: Union[Path, str],
    repo_root: Path,
    upgrade: bool,
    extra_index_url: Optional[str],
) -> None:
    install_cmd = [str(python), "-m", "pip", "install"]
    if upgrade:
        install_cmd.append("--upgrade")
    if extra_index_url:
        install_cmd += ["--extra-index-url", extra_index_url]

    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    _run([*install_cmd, "-e", str(repo_root)])


def _setup_conda(
    env_name: str,
    repo_root: Path,
    upgrade: bool,
    extra_index_url: Optional[str],
) -> None:
    conda = shutil.which("conda")
    if conda is None:
        sys.exit("conda was not found. Install Conda or run: python3 setup_env.py --env-manager venv")

    env_file = repo_root / "environment.yml"
    if not _is_conda_env(conda, env_name):
        _run([conda, "env", "create", "-n", env_name, "-f", str(env_file)])
    else:
        _run([conda, "env", "update", "-n", env_name, "-f", str(env_file), "--prune"])

    pip_cmd = [conda, "run", "-n", env_name, "python", "-m", "pip"]
    _run([*pip_cmd, "install", "--upgrade", "pip"])

    install_cmd = [*pip_cmd, "install", "-e", str(repo_root)]
    if upgrade:
        install_cmd.append("--upgrade")
    if extra_index_url:
        install_cmd += ["--extra-index-url", extra_index_url]
    _run(install_cmd)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _activation_hint(venv_dir: Path) -> str:
    if os.name == "nt":
        return f"{venv_dir}\\Scripts\\Activate.ps1"
    return f"source {venv_dir}/bin/activate"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up a ready-to-use local Python environment.")
    parser.add_argument(
        "--env-manager",
        choices=["auto", "conda", "venv", "current"],
        default="auto",
        help="Environment manager to use. auto prefers conda and falls back to venv.",
    )
    parser.add_argument(
        "--conda-env",
        default="few-shot-face-classification",
        help="Conda environment name.",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=Path(".venv"),
        help="Virtual environment directory.",
    )
    parser.add_argument(
        "--extra-index-url",
        default=None,
        help="Optional extra package index, useful for custom PyTorch wheels.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade dependencies while installing.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    env_manager = args.env_manager

    if env_manager == "auto":
        env_manager = "conda" if shutil.which("conda") else "venv"

    if env_manager == "conda":
        _setup_conda(args.conda_env, repo_root, args.upgrade, args.extra_index_url)
        print("Environment ready.")
        print(f"Activate it with: conda activate {args.conda_env}")
        return

    if sys.version_info < (3, 8) or sys.version_info >= (3, 15):
        sys.exit("Python >=3.8 and <3.15 is required for venv/current installs.")

    python = Path(sys.executable)
    if env_manager == "venv":
        if not args.venv.exists():
            _run([str(python), "-m", "venv", str(args.venv)])
        python = _venv_python(args.venv)

    _install_with_python(python, repo_root, args.upgrade, args.extra_index_url)

    print("Environment ready.")
    if env_manager == "venv":
        print(f"Activate it with: {_activation_hint(args.venv)}")


if __name__ == "__main__":
    main()
