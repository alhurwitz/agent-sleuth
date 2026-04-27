#!/usr/bin/env python3
"""
Update tests/perf/baselines/develop.json with fresh measurements.

Usage (run on develop branch after all perf tests pass):
    uv run python scripts/perf-baseline.py

The script runs the perf suite with --benchmark-json, extracts median (p50)
and p95 per benchmark, and writes them to tests/perf/baselines/develop.json.

Commit the updated baseline file to lock in new expected values.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASELINES_PATH = Path(__file__).parent.parent / "tests" / "perf" / "baselines" / "develop.json"
TMP_JSON = Path("/tmp/perf-baseline-tmp.json")


def main() -> int:
    print("Running perf suite (5 rounds per benchmark)...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "-m",
            "perf",
            "--benchmark-json",
            str(TMP_JSON),
            "--benchmark-min-rounds=5",
            "-q",
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        print(
            "ERROR: perf suite failed. Fix failures before updating baseline.",
            file=sys.stderr,
        )
        return 1

    if not TMP_JSON.exists():
        print("ERROR: benchmark JSON not written.", file=sys.stderr)
        return 1

    raw = json.loads(TMP_JSON.read_text())
    baselines: dict[str, dict[str, float]] = {}

    for bench in raw.get("benchmarks", []):
        name: str = bench["name"]
        stats: dict[str, object] = bench["stats"]
        # pytest-benchmark stats keys: mean, median, min, max, stddev, iqr, ops,
        # rounds, iterations. We use median (p50) and compute p95 from the rounds list.
        rounds_data: list[float] = bench.get("stats", {}).get("data", [])

        p50 = float(stats.get("median", stats.get("mean", 0.0)))  # type: ignore[arg-type]
        if rounds_data:
            sorted_data = sorted(rounds_data)
            idx_95 = int(len(sorted_data) * 0.95)
            p95 = sorted_data[min(idx_95, len(sorted_data) - 1)]
        else:
            p95 = float(stats.get("max", p50))  # type: ignore[arg-type]

        # Convert seconds → ms
        baselines[name] = {
            "median_ms": round(p50 * 1000, 2),
            "p95_ms": round(p95 * 1000, 2),
        }

    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_PATH.write_text(json.dumps(baselines, indent=2) + "\n")
    print(f"Baseline written to {BASELINES_PATH}")
    for name, vals in baselines.items():
        print(f"  {name}: median={vals['median_ms']}ms  p95={vals['p95_ms']}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
