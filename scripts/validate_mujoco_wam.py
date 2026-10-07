#!/usr/bin/env python3
"""Validate a MuJoCo-WAM dataset root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.mujoco_wam.validate import validate_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--require-analytic-sanity",
        "--require-analytic-families",
        dest="require_analytic_sanity",
        action="store_true",
        help="also run free-fall, spring, and pendulum sanity checks",
    )
    args = parser.parse_args()
    report = validate_dataset(
        args.dataset_root,
        analytic_sanity=args.require_analytic_sanity,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
