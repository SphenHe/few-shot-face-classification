"""Embedding cache helpers."""
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from few_shot_face_classification.data import get_im_paths
from few_shot_face_classification.embed import embed_paths, embed_folder


_CACHE_VERSION = 2
_FileMeta = Dict[str, Dict[str, Any]]
_Logger = Optional[Callable[[str], None]]


def _relative_path(path: Path, folder: Path) -> str:
    return path.relative_to(folder).as_posix()


def _file_metadata(folder: Path) -> _FileMeta:
    files = {}
    for path in get_im_paths(folder):
        stat = path.stat()
        files[_relative_path(path, folder)] = {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
    return files


def _restore_paths(raw_paths: List[Any], folder: Path) -> List[Path]:
    return [folder / Path(path) for path in raw_paths]


def _current_cached_files(
        data: Dict[str, Any],
        current_files: _FileMeta,
        cache_file: Path,
        folder: Path,
) -> Tuple[set, set]:
    """Return current file paths that can be reused and paths that need embedding."""
    cached_paths = {Path(path).as_posix() for path in data.get("paths", [])}
    current_paths = set(current_files)

    cached_files = data.get("files")
    if cached_files is not None:
        cached_current = {
            path
            for path, meta in current_files.items()
            if path in cached_paths and cached_files.get(path) == meta
        }
        return cached_current, current_paths - cached_current

    cache_mtime = cache_file.stat().st_mtime
    cached_current = {
        path
        for path in current_paths
        if path in cached_paths and (folder / path).stat().st_mtime < cache_mtime
    }
    return cached_current, current_paths - cached_current


def _filter_embeddings(
        paths: List[Path],
        embeddings: List[Any],
        folder: Path,
        keep_paths: set,
) -> Tuple[List[Path], List[Any]]:
    filtered_paths, filtered_embeddings = [], []
    for path, embedding in zip(paths, embeddings):
        if _relative_path(path, folder) in keep_paths:
            filtered_paths.append(path)
            filtered_embeddings.append(embedding)
    return filtered_paths, filtered_embeddings


def save_embeddings_cache(
        cache_file: Path,
        labeled_folder: Path,
        labeled_paths: List[Path],
        labeled_embeddings: List[Any],
        log: _Logger = None,
) -> None:
    """Persist embeddings with the current labeled file inventory."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        rel_paths = [_relative_path(path, labeled_folder) for path in labeled_paths]
        with open(cache_file, "wb") as f:
            pickle.dump({
                "version": _CACHE_VERSION,
                "paths": rel_paths,
                "embeddings": labeled_embeddings,
                "files": _file_metadata(labeled_folder),
            }, f)
        if log is not None:
            log(f"Saved embeddings to cache: {cache_file}")
    except Exception as exc:
        if log is not None:
            log(f"Warning: Failed to save cache: {exc}")


def load_or_create_embeddings(
        labeled_folder: Path,
        batch_size: int = 32,
        cache_file: Optional[Path] = None,
        use_cache: bool = True,
        device: str = "cpu",
        num_workers: Optional[int] = None,
        log: _Logger = None,
) -> Tuple[List[Path], List[Any]]:
    """Backward-compatible alias for load_or_build_embeddings_cache."""
    return load_or_build_embeddings_cache(
        labeled_folder=labeled_folder,
        batch_size=batch_size,
        cache_file=cache_file,
        use_cache=use_cache,
        device=device,
        num_workers=num_workers,
        log=log,
    )


def build_embeddings_cache(
        labeled_folder: Path,
        cache_file: Optional[Path] = None,
        batch_size: int = 32,
        device: str = "cpu",
        num_workers: Optional[int] = None,
        log: _Logger = None,
) -> Tuple[List[Path], List[Any]]:
    """Build labeled embeddings and optionally persist them to cache."""
    labeled_paths, labeled_embeddings = embed_folder(
        labeled_folder,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    if cache_file is not None:
        save_embeddings_cache(Path(cache_file), Path(labeled_folder), labeled_paths, labeled_embeddings, log=log)
    return labeled_paths, labeled_embeddings


def load_or_build_embeddings_cache(
        labeled_folder: Path,
        batch_size: int = 32,
        cache_file: Optional[Path] = None,
        use_cache: bool = True,
        device: str = "cpu",
        num_workers: Optional[int] = None,
        log: _Logger = None,
) -> Tuple[List[Path], List[Any]]:
    """Load cached embeddings when possible, otherwise build the shared cache."""
    if not use_cache or cache_file is None:
        return build_embeddings_cache(
            labeled_folder,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            log=log,
        )

    labeled_folder = Path(labeled_folder)
    cache_file = Path(cache_file)

    if cache_file.exists():
        if log is not None:
            log(f"Loading embeddings from cache: {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            paths = _restore_paths(data["paths"], labeled_folder)
            embeddings = data["embeddings"]
            if len(paths) != len(embeddings):
                raise ValueError("Cached paths and embeddings have different lengths")

            current_files = _file_metadata(labeled_folder)
            cached_current, pending = _current_cached_files(data, current_files, cache_file, labeled_folder)
            paths, embeddings = _filter_embeddings(paths, embeddings, labeled_folder, cached_current)
            if not pending:
                if log is not None:
                    log(f"Loaded {len(embeddings)} cached embeddings")
                return paths, embeddings

            pending_paths = [labeled_folder / path for path in sorted(pending)]
            if log is not None:
                log(f"Embedding {len(pending_paths)} new or changed labeled images...")
            new_paths, new_embeddings = embed_paths(
                pending_paths,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
            )
            paths.extend(new_paths)
            embeddings.extend(new_embeddings)
            save_embeddings_cache(cache_file, labeled_folder, paths, embeddings, log=log)
            return paths, embeddings
        except Exception as exc:
            if log is not None:
                log(f"Failed to load cache: {exc}")
                log("Re-processing labeled images...")

    return build_embeddings_cache(
        labeled_folder,
        cache_file=cache_file,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        log=log,
    )
