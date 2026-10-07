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


def _first_entry(
    entries: Iterable[Mapping[str, Any]],
    family: str,
    predicate: Any | None = None,
) -> Mapping[str, Any] | None:
    candidates = [
        entry for entry in entries
        if entry.get("family") == family and (predicate is None or predicate(entry))
    ]
    return sorted(candidates, key=lambda row: str(row.get("trajectory_id", "")))[0] if candidates else None


def _body_index(data: Mapping[str, np.ndarray], name: str) -> int:
    names = [str(value) for value in np.asarray(data["body_names"]).tolist()]
    try:
        return names.index(name)
    except ValueError as exc:
        raise KeyError(name) from exc


def analytic_sanity_checks(
    root: Path,
    entries: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Run deterministic sanity checks against simple analytic dynamics.

    These checks are deliberately conservative.  They are not a substitute for
    the simulator's full validation; they only catch broken initialization,
    gravity, spring, or pendulum wiring before a large batch is accepted.
    """

    entries = list(entries)
    checks: Dict[str, Any] = {}
    errors: List[str] = []

    # Free fall: compare the pre-contact projectile against x(t)=x0+v0*t+1/2*g*t^2.
    free_entry = _first_entry(entries, "free_fall_projectile")
    if free_entry is None:
        errors.append("analytic free-fall check: no free_fall_projectile entry")
    else:
        with np.load(root / str(free_entry["paths"]["state"]), allow_pickle=False) as data:
            body_id = _body_index(data, "target")
            params = free_entry.get("parameters", {})
            x0 = np.asarray(params.get("initial_position", data["body_xpos"][0, body_id]), dtype=np.float64)
            v0 = np.asarray(params.get("initial_velocity", data["qvel"][0, :3]), dtype=np.float64)
            gravity = np.asarray(data["gravity"], dtype=np.float64)
            time = np.asarray(data["time"], dtype=np.float64)
            contact = np.asarray(data["contact_count"]) > 0
            first_contact = float(time[np.flatnonzero(contact)[0]]) if np.any(contact) else float(time[-1])
            eligible = (time <= min(first_contact, 0.5)) & ~contact
            prediction = x0[None, :] + v0[None, :] * time[:, None] + 0.5 * gravity[None, :] * time[:, None] ** 2
            error = np.linalg.norm(data["body_xpos"][:, body_id, :] - prediction, axis=-1)
            max_error = float(np.max(error[eligible])) if np.any(eligible) else float("inf")
            checks["free_fall_projectile"] = {
                "trajectory_id": free_entry["trajectory_id"],
                "eligible_frames": int(np.count_nonzero(eligible)),
                "max_position_error_m": max_error,
                "tolerance_m": 0.03,
                "passed": bool(np.any(eligible) and max_error <= 0.03),
            }
            if not checks["free_fall_projectile"]["passed"]:
                errors.append(
                    "analytic free-fall check failed: "
                    f"max error {max_error:.6g} m > 0.03 m"
                )

    # Spring: an undamped slide joint should follow the harmonic oscillator.
    spring_entry = _first_entry(
        entries,
        "spring_oscillator",
        lambda row: float(row.get("parameters", {}).get("damping", 1.0)) <= 1e-12,
    )
    if spring_entry is None:
        errors.append("analytic spring check: no undamped spring_oscillator entry")
    else:
        with np.load(root / str(spring_entry["paths"]["state"]), allow_pickle=False) as data:
            body_id = _body_index(data, "target")
            params = spring_entry.get("parameters", {})
            stiffness = float(params.get("stiffness", 0.0))
            mass = float(np.asarray(data["body_mass"])[body_id])
            time = np.asarray(data["time"], dtype=np.float64)
            position = np.asarray(data["qpos"][:, 0], dtype=np.float64)
            velocity = float(np.asarray(data["qvel"])[0, 0])
            omega = float(np.sqrt(max(stiffness, 0.0) / max(mass, 1e-12)))
            if omega > 0.0:
                prediction = (
                    position[0] * np.cos(omega * time)
                    + velocity / omega * np.sin(omega * time)
                )
                eligible = time <= min(8.0, float(time[-1]))
                max_error = float(np.max(np.abs(position[eligible] - prediction[eligible])))
            else:
                max_error = float("inf")
                eligible = np.zeros_like(time, dtype=bool)
            checks["spring_oscillator"] = {
                "trajectory_id": spring_entry["trajectory_id"],
                "eligible_frames": int(np.count_nonzero(eligible)),
                "max_position_error_m": max_error,
                "tolerance_m": 0.12,
                "passed": bool(np.any(eligible) and max_error <= 0.12),
            }
            if not checks["spring_oscillator"]["passed"]:
                errors.append(
                    "analytic spring check failed: "
                    f"max error {max_error:.6g} m > 0.12 m"
                )

    # Pendulum: for a no-damping simple pendulum, energy should remain bounded.
    pendulum_entry = _first_entry(
        entries,
        "simple_double_pendulum",
        lambda row: float(row.get("parameters", {}).get("damping", 1.0)) <= 1e-12,
    )
    if pendulum_entry is None:
        errors.append("analytic pendulum check: no undamped pendulum entry")
    else:
        with np.load(root / str(pendulum_entry["paths"]["state"]), allow_pickle=False) as data:
            energy = np.asarray(data["total_energy"], dtype=np.float64)
            mask = np.asarray(data["M_conservative"], dtype=bool)
            eligible = mask & (np.asarray(data["time"]) <= 8.0)
            values = energy[eligible]
            if values.size:
                scale = max(abs(float(values[0])), 1e-6)
                relative_range = float((np.max(values) - np.min(values)) / scale)
            else:
                relative_range = float("inf")
            check_name = "simple_pendulum" if not pendulum_entry.get("parameters", {}).get("double") else "double_pendulum"
            checks[check_name] = {
                "trajectory_id": pendulum_entry["trajectory_id"],
                "eligible_frames": int(values.size),
                "relative_energy_range": relative_range,
                "tolerance": 0.20,
                "passed": bool(values.size >= 10 and relative_range <= 0.20),
            }
            if not checks[check_name]["passed"]:
                errors.append(
                    "analytic pendulum check failed: "
                    f"relative energy range {relative_range:.6g} > 0.20"
                )

    return {"checks": checks, "errors": errors, "valid": not errors}


def validate_dataset(root: Path, *, analytic_sanity: bool = False) -> Dict[str, Any]:
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
    if analytic_sanity:
        analytic = analytic_sanity_checks(root, entries)
        report["analytic_sanity"] = analytic
        errors.extend(analytic["errors"])
        report["errors"] = errors[:200]
        report["error_count"] = len(errors)
        report["valid"] = not errors
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
