"""Fast file and corpus hashing for change detection."""

from __future__ import annotations

import hashlib
import pathlib
from enum import StrEnum
from typing import Literal


class HashMode(StrEnum):
    MTIME = "mtime"  # fast: (path, size, mtime) — good enough for local files
    HASH = "hash"  # thorough: xxhash of content
    ALWAYS = "always"  # always treat as changed — returns unique value per call


def file_hash(path: str, mode: HashMode | str = HashMode.MTIME) -> str:
    """Return a change-detection hash for a single file."""
    p = pathlib.Path(path)
    if mode == HashMode.ALWAYS:
        import time

        return f"always-{time.monotonic_ns()}"
    if mode == HashMode.MTIME:
        stat = p.stat()
        raw = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
    # HashMode.HASH — content-based
    try:
        import xxhash

        hasher = xxhash.xxh64()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return str(hasher.hexdigest())[:16]
    except ImportError:
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                md5.update(chunk)
        return md5.hexdigest()[:16]


def corpus_version(
    corpus_path: str,
    *,
    mode: Literal["mtime", "hash", "always"] = "mtime",
    include_patterns: list[str] | None = None,
) -> str:
    """Return a short hex string that changes when any file in the corpus changes."""
    root = pathlib.Path(corpus_path)
    hashes: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        # skip .sleuth dir itself
        try:
            p.relative_to(root / ".sleuth")
            continue
        except ValueError:
            pass
        hashes.append(file_hash(str(p), mode=HashMode(mode)))
    combined = "|".join(hashes)
    return hashlib.md5(combined.encode()).hexdigest()[:16]
