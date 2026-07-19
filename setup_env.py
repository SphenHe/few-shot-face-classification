#!/usr/bin/env python3
"""Create a Conda or venv environment and install this project."""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Union


FACENET_WHEEL_URL = (
    "http://166.111.17.106:23333/interchange%20station/20900012/"
    "facenet_pytorch-3.0.0-py3-none-any.whl"
)
FACENET_ARTIFACT_URL = (
    "https://api.github.com/repos/SphenHe/facenet-pytorch/actions/"
    "artifacts/8441941129/zip"
)
FACENET_SOURCE_URL = (
    "git+https://github.com/SphenHe/facenet-pytorch.git@"
    "f26ffeb58782e86cf872664b4a01baa7ce110d77"
)
FACENET_WHEEL_NAME = "facenet_pytorch-3.0.0-py3-none-any.whl"


def _run(cmd):
    print(">>", shlex.join(cmd))
    subprocess.check_call(cmd)


def _download(url: str, destination: Path, token: Optional[str] = None) -> None:
    """Download a URL atomically, leaving no partial destination on failure."""
    headers = {"User-Agent": "few-shot-face-classification-setup"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/vnd.github+json"
    request = urllib.request.Request(url, headers=headers)
    partial = destination.with_name(f".{destination.name}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _github_artifact_wheel(download_dir: Path) -> Path:
    """Download and extract the pinned GitHub Actions wheel artifact."""
    archive = download_dir / "facenet-pytorch-wheel.zip"
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if shutil.which("gh"):
        _run(
            [
                "gh",
                "run",
                "download",
                "29685412234",
                "--repo",
                "SphenHe/facenet-pytorch",
                "--name",
                "facenet-pytorch-wheel",
                "--dir",
                str(download_dir),
            ]
        )
    else:
        if not token:
            raise RuntimeError("GitHub artifact download requires gh login, GH_TOKEN, or GITHUB_TOKEN")
        _download(FACENET_ARTIFACT_URL, archive, token=token)
        with zipfile.ZipFile(archive) as artifact:
            member = next(
                (item for item in artifact.infolist() if Path(item.filename).name == FACENET_WHEEL_NAME),
                None,
            )
            if member is None:
                raise RuntimeError(f"{FACENET_WHEEL_NAME} was not found in the GitHub artifact")
            wheel = download_dir / FACENET_WHEEL_NAME
            with artifact.open(member) as source, wheel.open("wb") as destination:
                shutil.copyfileobj(source, destination)

    wheel = download_dir / FACENET_WHEEL_NAME
    if not wheel.is_file():
        raise RuntimeError(f"GitHub artifact did not produce {FACENET_WHEEL_NAME}")
    _validate_wheel(wheel)
    return wheel


def _validate_wheel(wheel: Path) -> None:
    """Reject truncated files and error pages before handing a wheel to pip."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"Corrupt wheel downloaded: {wheel}")
            if not any(name.endswith(".dist-info/METADATA") for name in archive.namelist()):
                raise RuntimeError(f"Downloaded file is not a Python wheel: {wheel}")
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"Downloaded file is not a valid wheel: {wheel}") from error


def _resolve_facenet_install(local_wheel: Optional[Path], download_dir: Path) -> Union[Path, str]:
    """Resolve facenet install source: local/internal, Actions artifact, then source."""
    if local_wheel:
        return local_wheel

    wheel = download_dir / FACENET_WHEEL_NAME
    print(f"Downloading facenet-pytorch wheel from preferred mirror: {FACENET_WHEEL_URL}")
    try:
        _download(FACENET_WHEEL_URL, wheel)
        _validate_wheel(wheel)
        return wheel
    except Exception as error:
        print(f"Preferred wheel mirror failed: {error}", file=sys.stderr)

    print("Trying GitHub Actions artifact facenet-pytorch-wheel...")
    try:
        return _github_artifact_wheel(download_dir)
    except Exception as error:
        print(f"GitHub Actions artifact failed: {error}", file=sys.stderr)

    print("Falling back to building facenet-pytorch from the pinned GitHub source.")
    return FACENET_SOURCE_URL


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
    facenet_install: Union[Path, str],
) -> None:
    install_cmd = [str(python), "-m", "pip", "install"]
    if upgrade:
        install_cmd.append("--upgrade")
    if extra_index_url:
        install_cmd += ["--extra-index-url", extra_index_url]

    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    _run([*install_cmd, str(facenet_install)])
    _run([*install_cmd, "-e", str(repo_root)])


def _setup_conda(
    env_name: str,
    repo_root: Path,
    upgrade: bool,
    extra_index_url: Optional[str],
    facenet_install: Union[Path, str],
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

    install_cmd = [*pip_cmd, "install"]
    if upgrade:
        install_cmd.append("--upgrade")
    if extra_index_url:
        install_cmd += ["--extra-index-url", extra_index_url]
    _run([*install_cmd, str(facenet_install)])
    _run([*install_cmd, "-e", str(repo_root)])


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
    parser.add_argument(
        "--facenet-wheel",
        type=Path,
        default=None,
        help="Install the prebuilt facenet-pytorch 3.0.0 wheel before this project.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    env_manager = args.env_manager
    facenet_wheel = args.facenet_wheel
    if facenet_wheel:
        facenet_wheel = facenet_wheel.expanduser().resolve()
        if not facenet_wheel.is_file() or facenet_wheel.suffix != ".whl":
            sys.exit(f"facenet wheel not found or not a .whl file: {facenet_wheel}")
        try:
            _validate_wheel(facenet_wheel)
        except RuntimeError as error:
            sys.exit(str(error))

    if env_manager == "auto":
        env_manager = "conda" if shutil.which("conda") else "venv"

    try:
        with tempfile.TemporaryDirectory(prefix="facenet-pytorch-") as temporary:
            facenet_install = _resolve_facenet_install(facenet_wheel, Path(temporary))

            if env_manager == "conda":
                _setup_conda(
                    args.conda_env,
                    repo_root,
                    args.upgrade,
                    args.extra_index_url,
                    facenet_install,
                )
                print("Environment ready.")
                print(f"Activate it with: conda activate {args.conda_env}")
                return

            if sys.version_info < (3, 9) or sys.version_info >= (3, 15):
                sys.exit("Python >=3.9 and <3.15 is required for venv/current installs.")

            python = Path(sys.executable)
            if env_manager == "venv":
                if not args.venv.exists():
                    _run([str(python), "-m", "venv", str(args.venv)])
                python = _venv_python(args.venv)

            _install_with_python(
                python,
                repo_root,
                args.upgrade,
                args.extra_index_url,
                facenet_install,
            )
    except (OSError, RuntimeError, subprocess.CalledProcessError, urllib.error.URLError) as error:
        sys.exit(f"Environment setup failed: {error}")

    print("Environment ready.")
    if env_manager == "venv":
        print(f"Activate it with: {_activation_hint(args.venv)}")


if __name__ == "__main__":
    main()
