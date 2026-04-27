"""Async ripgrep wrapper for CodeSearch."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("sleuth.backends.codesearch.ripgrep")


@dataclass(frozen=True, slots=True)
class RipgrepHit:
    path: Path
    line_number: int  # 1-based
    line_text: str


async def run_ripgrep(
    pattern: str,
    root: Path,
    *,
    glob: str | None = None,  # e.g. "*.py" to restrict file types
    fixed_strings: bool = False,  # pass -F for literal (non-regex) queries
    max_count: int | None = None,  # per-file hit cap
) -> AsyncIterator[RipgrepHit]:
    """Yield RipgrepHit objects for every match of *pattern* under *root*.

    Uses ``rg --json`` so output is machine-readable.  Respects .gitignore
    automatically (ripgrep default behaviour).  Binary files are skipped by
    ripgrep silently.
    """
    cmd: list[str] = [
        "rg",
        "--json",
        "--line-number",
        "--no-require-git",  # respect .gitignore even outside a git repo
    ]
    if fixed_strings:
        cmd.append("--fixed-strings")
    if glob:
        cmd.extend(["--glob", glob])
    if max_count is not None:
        cmd.extend(["--max-count", str(max_count)])
    cmd.extend(["--", pattern, str(root)])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None

    async for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("rg: non-JSON line skipped: %s", line[:80])
            continue

        if obj.get("type") != "match":
            continue

        data = obj["data"]
        path = Path(data["path"]["text"])
        line_number: int = data["line_number"]
        line_text: str = data["lines"]["text"].rstrip("\n")
        yield RipgrepHit(path=path, line_number=line_number, line_text=line_text)

    await proc.wait()
    if proc.returncode not in (0, 1):  # 1 = no matches, which is fine
        stderr = (await proc.stderr.read()).decode(errors="replace") if proc.stderr else ""
        logger.warning("rg exited %d: %s", proc.returncode, stderr[:200])
