#!/usr/bin/env python3
"""Check deterministic generation for a fixed set of seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.mujoco_wam.generator import generate_dataset  # noqa: E402


def digest(root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"dataset_summary.json"}:
            hasher.update(str(path.relative_to(root)).encode("utf-8"))
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        first = args.output_root / f"seed_{seed}_a"
        second = args.output_root / f"seed_{seed}_b"
        for target in (first, second):
            if target.exists():
                shutil.rmtree(target)
            generate_dataset(
                output_root=target,
                families=("free_fall_projectile",),
                trajectories_per_family=1,
                master_seed=seed,
                render_video=False,
                limit=1,
            )
        first_digest = digest(first)
        second_digest = digest(second)
        records.append({"seed": seed, "first": first_digest, "second": second_digest, "match": first_digest == second_digest})
    report = {"seeds": records, "valid": all(row["match"] for row in records)}
    (args.output_root / "repro_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
