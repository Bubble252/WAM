"""Validation checks for MuJoCo-WAM manifests and state artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np


REQUIRED_STATE_KEYS = {
    "time",
    "qpos",
    "qvel",
    "qacc",
    "act",
    "ctrl",
    "body_xpos",
    "body_xquat",
    "body_linear_velocity",
    "body_angular_velocity",
    "contact_geom_ids",
    "contact_positions",
    "contact_normals",
    "contact_distances",
    "contact_forces",
    "contact_frictions",
    "contact_valid",
    "contact_count",
    "qfrc_constraint",
    "qfrc_passive",
    "qfrc_actuator",
    "qfrc_applied",
    "xfrc_applied",
    "kinetic_energy",
    "potential_energy",
    "total_energy",
    "linear_momentum",
    "angular_momentum",
    "external_work",
    "M_state",
    "M_conservative",
    "M_momentum",
}


def _finite(name: str, array: np.ndarray) -> List[str]:
    if not np.all(np.isfinite(array)):
        return [f"{name}: contains non-finite values"]
    return []


def validate_entry(root: Path, entry: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    trajectory_id = str(entry.get("trajectory_id", "<missing>"))
    for key in ("mjcf", "state", "events", "camera"):
        rel = entry.get("paths", {}).get(key)
        if not rel or not (root / rel).exists():
            errors.append(f"{trajectory_id}: missing {key} artifact")
    state_path = root / entry.get("paths", {}).get("state", "")
    if not state_path.is_file():
        return errors
    with np.load(state_path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = REQUIRED_STATE_KEYS - keys
        errors.extend(f"{trajectory_id}: missing state key {key}" for key in sorted(missing))
        for key in sorted(REQUIRED_STATE_KEYS & keys):
            errors.extend(_finite(f"{trajectory_id}:{key}", data[key]))
        if "time" in keys:
            time = data["time"]
            if len(time) < 2 or np.any(np.diff(time) < 0):
                errors.append(f"{trajectory_id}: time is not monotonic")
        if "body_xquat" in keys:
            norms = np.linalg.norm(data["body_xquat"], axis=-1)
            if np.max(np.abs(norms[:, 1:] - 1.0)) > 5e-3:
                errors.append(f"{trajectory_id}: body quaternion norm drift")
        if "contact_valid" in keys and "contact_count" in keys:
            counts = data["contact_valid"].sum(axis=1)
            if np.any(counts != data["contact_count"]):
                errors.append(f"{trajectory_id}: contact count mismatch")
    event_path = root / entry.get("paths", {}).get("events", "")
    if event_path.exists():
        try:
            json.loads(event_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{trajectory_id}: invalid events JSON: {exc}")
    return errors


def validate_dataset(root: Path) -> Dict[str, Any]:
    root = Path(root)
    manifest_dir = root / "manifests"
    entries: List[Mapping[str, Any]] = []
    errors: List[str] = []
    for manifest_path in sorted(manifest_dir.glob("*.jsonl")):
        with manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{manifest_path}:{line_number}: invalid JSON: {exc}")
                    continue
                entries.append(entry)
                errors.extend(validate_entry(root, entry))
    split_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    for entry in entries:
        split_counts[str(entry.get("split"))] = split_counts.get(str(entry.get("split")), 0) + 1
        family_counts[str(entry.get("family"))] = family_counts.get(str(entry.get("family")), 0) + 1
    report = {
        "dataset_root": str(root),
        "entries": len(entries),
        "split_counts": split_counts,
        "family_counts": family_counts,
        "error_count": len(errors),
        "errors": errors[:200],
        "valid": not errors,
    }
    (root / "dataset_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# MuJoCo-WAM v0.1 validation",
        "",
        f"- Entries: {report['entries']}",
        f"- Valid: `{report['valid']}`",
        f"- Errors: {report['error_count']}",
        f"- Splits: `{json.dumps(split_counts, sort_keys=True)}`",
        f"- Families: `{json.dumps(family_counts, sort_keys=True)}`",
        "",
    ]
    if errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in errors[:200])
    else:
        lines.extend(["No schema, finite-value, monotonic-time, quaternion, contact-count, or JSON errors were found.", ""])
    (root / "dataset_validation.md").write_text("\n".join(lines), encoding="utf-8")
    return report
