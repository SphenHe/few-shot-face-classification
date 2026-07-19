"""Roster, confirmation, and persistence helpers for live attendance."""

from __future__ import annotations

import csv
import os
import re
import shutil
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple


PRESENT = "present"
ABSENT = "absent"
_ROSTER_HEADERS = ("姓名", "name")
_LABEL_PATTERN = re.compile(r"^(.+)_([0-9]+)$")
_STATE_FIELDS = ("name", "status", "checkin_time", "method", "updated_at")
_EVENT_FIELDS = ("timestamp", "name", "old_status", "new_status", "method")


class RosterError(ValueError):
    """Raised when a roster cannot be used safely."""


def load_roster(path: Path) -> List[str]:
    """Load unique names from a UTF-8 CSV with a ``姓名`` or ``name`` column."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = [header.strip() for header in (reader.fieldnames or []) if header]
            name_header = next((header for header in _ROSTER_HEADERS if header in headers), None)
            if name_header is None:
                raise RosterError("名单 CSV 必须包含“姓名”或“name”列")

            names: List[str] = []
            seen: Set[str] = set()
            duplicates: Set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                name = (row.get(name_header) or "").strip()
                if not name:
                    raise RosterError(f"名单第 {line_number} 行姓名为空")
                if name in seen:
                    duplicates.add(name)
                else:
                    seen.add(name)
                    names.append(name)
    except UnicodeDecodeError as exc:
        raise RosterError("名单 CSV 必须使用 UTF-8 或 UTF-8 BOM 编码") from exc
    except OSError as exc:
        raise RosterError(f"无法读取名单 CSV：{path}") from exc

    if duplicates:
        raise RosterError(f"名单中存在重复姓名：{', '.join(sorted(duplicates))}")
    if not names:
        raise RosterError("名单 CSV 中没有学生")
    return names


def parse_label_name(path: Path) -> Optional[str]:
    """Return the class from a strict ``姓名_序号`` reference-image name."""
    match = _LABEL_PATTERN.fullmatch(Path(path).stem.strip())
    if match is None:
        return None
    return match.group(1).strip() or None


def filter_labeled_embeddings(
    paths: Sequence[Path],
    embeddings: Sequence[Any],
    roster: Iterable[str],
) -> Tuple[List[Path], List[Any], List[str]]:
    """Keep valid roster/negative samples and report ignored reference images."""
    roster_names = set(roster)
    valid_paths: List[Path] = []
    valid_embeddings: List[Any] = []
    warnings: List[str] = []

    for path, embedding in zip(paths, embeddings):
        label = parse_label_name(path)
        if label is None:
            warnings.append(f"忽略命名不规范的标注照片：{Path(path).name}")
            continue
        if label.lower() != "none" and label not in roster_names:
            warnings.append(f"忽略名单外标注照片：{Path(path).name}")
            continue
        valid_paths.append(Path(path))
        valid_embeddings.append(embedding)

    return valid_paths, valid_embeddings, warnings


class RollingConfirmation:
    """Confirm identities after enough hits in a rolling inference window."""

    def __init__(self, hits: int = 3, window: int = 5) -> None:
        if hits < 1 or window < 1 or hits > window:
            raise ValueError("hits/window must satisfy 1 <= hits <= window")
        self.hits = hits
        self.window = window
        self._history: Dict[str, Deque[bool]] = {}

    def update(self, detected: Iterable[str], eligible: Iterable[str]) -> Set[str]:
        detected_names = set(detected)
        eligible_names = set(eligible)
        for stale_name in self._history.keys() - eligible_names:
            del self._history[stale_name]

        confirmed: Set[str] = set()
        for name in eligible_names:
            history = self._history.setdefault(name, deque(maxlen=self.window))
            history.append(name in detected_names)
            if len(history) >= self.hits and sum(history) >= self.hits:
                confirmed.add(name)
        for name in confirmed:
            self._history.pop(name, None)
        return confirmed

    def reset(self) -> None:
        self._history.clear()


@dataclass
class AttendanceRecord:
    name: str
    status: str = ABSENT
    checkin_time: str = ""
    method: str = ""
    updated_at: str = ""

    def as_row(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "checkin_time": self.checkin_time,
            "method": self.method,
            "updated_at": self.updated_at,
        }


class AttendanceSession:
    """Maintain and persist the single current attendance session."""

    def __init__(self, roster: Sequence[str], directory: Path) -> None:
        self.roster = list(roster)
        self.directory = Path(directory)
        self.current_file = self.directory / "current.csv"
        self.events_file = self.directory / "events.csv"
        self.archive_dir = self.directory / "archive"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records = self._load_current()
        self._persist_current()

    def _load_current(self) -> Dict[str, AttendanceRecord]:
        restored: Dict[str, AttendanceRecord] = {}
        if self.current_file.exists():
            try:
                with self.current_file.open("r", encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        name = (row.get("name") or "").strip()
                        status = row.get("status")
                        if name in self.roster and status in (PRESENT, ABSENT):
                            restored[name] = AttendanceRecord(
                                name=name,
                                status=status,
                                checkin_time=row.get("checkin_time") or "",
                                method=row.get("method") or "",
                                updated_at=row.get("updated_at") or "",
                            )
            except (OSError, csv.Error):
                restored = {}
        return {name: restored.get(name, AttendanceRecord(name=name)) for name in self.roster}

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def set_present(self, name: str, present: bool, method: str) -> bool:
        if name not in self.records:
            raise KeyError(name)
        record = self.records[name]
        new_status = PRESENT if present else ABSENT
        if record.status == new_status:
            return False

        timestamp = self._now()
        old_status = record.status
        record.status = new_status
        record.checkin_time = timestamp if present else ""
        record.method = method
        record.updated_at = timestamp
        self._persist_current()
        self._append_event(timestamp, name, old_status, new_status, method)
        return True

    def toggle_manual(self, name: str) -> bool:
        return self.set_present(name, self.records[name].status != PRESENT, "manual")

    def present_names(self) -> Set[str]:
        return {name for name, record in self.records.items() if record.status == PRESENT}

    def _persist_current(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temp_name = ""
        try:
            with NamedTemporaryFile(
                "w", encoding="utf-8-sig", newline="", dir=self.directory, delete=False
            ) as handle:
                temp_name = handle.name
                writer = csv.DictWriter(handle, fieldnames=_STATE_FIELDS)
                writer.writeheader()
                writer.writerows(self.records[name].as_row() for name in self.roster)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.current_file)
        finally:
            if temp_name and Path(temp_name).exists():
                Path(temp_name).unlink()

    def _append_event(
        self, timestamp: str, name: str, old_status: str, new_status: str, method: str
    ) -> None:
        write_header = not self.events_file.exists() or self.events_file.stat().st_size == 0
        with self.events_file.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_EVENT_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": timestamp,
                "name": name,
                "old_status": old_status,
                "new_status": new_status,
                "method": method,
            })
            handle.flush()
            os.fsync(handle.fileno())

    def reset_and_archive(self) -> Path:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        destination = self.archive_dir / timestamp
        counter = 1
        while destination.exists():
            destination = self.archive_dir / f"{timestamp}_{counter}"
            counter += 1
        destination.mkdir(parents=True, exist_ok=False)

        for source in (self.current_file, self.events_file):
            if source.exists():
                shutil.move(str(source), str(destination / source.name))
        self.records = {name: AttendanceRecord(name=name) for name in self.roster}
        self._persist_current()
        return destination
