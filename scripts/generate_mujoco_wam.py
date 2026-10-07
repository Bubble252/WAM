#!/usr/bin/env python3
"""Generate MuJoCo-WAM v0.1 data.

The script is intentionally single-process for the first reproducible smoke
run.  Remote jobs may shard by family or by ``--limit`` after the one-seed
sanity run passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.mujoco_wam.generator import DEFAULT_CONFIG, generate_dataset  # noqa: E402
from data.mujoco_wam.scenarios import SCENARIO_FAMILIES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--families", nargs="*", default=list(SCENARIO_FAMILIES))
    parser.add_argument("--trajectories-per-family", type=int, default=300)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_CONFIG["master_seed"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    summary = generate_dataset(
        output_root=args.output_root,
        families=args.families,
        trajectories_per_family=args.trajectories_per_family,
        master_seed=args.master_seed,
        render_video=not args.no_video,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
