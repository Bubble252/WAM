"""Deterministic MuJoCo-WAM v0.1 trajectory generator."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .scenarios import SCENARIO_FAMILIES, Scenario, build_scenario


try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised on the target server
    raise RuntimeError(
        "MuJoCo is required. Install it in the remote WAM environment before running."
    ) from exc


DEFAULT_CONFIG: Dict[str, Any] = {
    "duration_seconds": 32.0,
    "sim_timestep": 1.0 / 240.0,
    "state_fps": 30,
    "video_fps": 8,
    "render_width": 128,
    "render_height": 128,
    "max_contacts": 16,
    "trajectories_per_family": 300,
    "master_seed": 20261007,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def split_for_index(index: int) -> str:
    if index < 200:
        return "train"
    if index < 250:
        return "val"
    if index < 275:
        return "test_interpolation"
    return "test_ood"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _angular_velocity(previous: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
    relative = _quat_multiply(current, _quat_conjugate(previous))
    if relative[0] < 0:
        relative = -relative
    vector = relative[1:]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(norm, max(float(relative[0]), 1e-12))
    return vector / norm * (angle / dt)


def _joint_initialise(model: Any, data: Any, init: Mapping[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for joint_name, (qpos, qvel) in init.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Initial state refers to missing joint {joint_name}")
        qpos_addr = int(model.jnt_qposadr[joint_id])
        dof_addr = int(model.jnt_dofadr[joint_id])
        nq = int(model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE and 7 or
                 model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL and 4 or 1)
        nv = int(model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE and 6 or
                 model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL and 3 or 1)
        data.qpos[qpos_addr:qpos_addr + nq] = np.asarray(qpos, dtype=np.float64)[:nq]
        data.qvel[dof_addr:dof_addr + nv] = np.asarray(qvel, dtype=np.float64)[:nv]
    if model.nu:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _body_names(model: Any) -> List[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        for body_id in range(model.nbody)
    ]


def _geom_names(model: Any) -> List[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        for geom_id in range(model.ngeom)
    ]


def _camera_metadata(model: Any) -> Dict[str, Any]:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")
    if camera_id < 0:
        return {}
    return {
        "name": "main",
        "id": int(camera_id),
        "pos": model.cam_pos[camera_id].copy(),
        "quat": model.cam_quat[camera_id].copy(),
        "fovy": float(model.cam_fovy[camera_id]),
        "resolution": [128, 128],
        "mode": "trackcom",
    }


def _full_mass_matrix(model: Any, data: Any) -> np.ndarray:
    matrix = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, matrix, data.qM)
    return matrix


def _energy_and_momentum(
    model: Any,
    data: Any,
    body_linear_velocity: np.ndarray,
    body_angular_velocity: np.ndarray,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    mass_matrix = _full_mass_matrix(model, data)
    kinetic = float(0.5 * data.qvel.dot(mass_matrix.dot(data.qvel)))
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    potential = float(
        sum(
            -model.body_mass[body_id] * np.dot(gravity, data.xpos[body_id])
            for body_id in range(1, model.nbody)
        )
    )
    linear = np.zeros(3, dtype=np.float64)
    angular = np.zeros(3, dtype=np.float64)
    for body_id in range(1, model.nbody):
        mass = float(model.body_mass[body_id])
        velocity = body_linear_velocity[body_id]
        linear += mass * velocity
        angular += np.cross(data.xpos[body_id], mass * velocity)
        angular += model.body_inertia[body_id] * body_angular_velocity[body_id]
    return kinetic, potential, kinetic + potential, linear, angular


def _contact_snapshot(model: Any, data: Any, max_contacts: int) -> Dict[str, np.ndarray]:
    geom_ids = np.full((max_contacts, 2), -1, dtype=np.int32)
    positions = np.zeros((max_contacts, 3), dtype=np.float32)
    normals = np.zeros((max_contacts, 3), dtype=np.float32)
    distances = np.zeros(max_contacts, dtype=np.float32)
    forces = np.zeros((max_contacts, 6), dtype=np.float32)
    frictions = np.zeros((max_contacts, 3), dtype=np.float32)
    valid = np.zeros(max_contacts, dtype=np.uint8)
    count = min(int(data.ncon), max_contacts)
    for index in range(count):
        contact = data.contact[index]
        geom_ids[index] = (int(contact.geom1), int(contact.geom2))
        positions[index] = contact.pos
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        normals[index] = frame[0]
        distances[index] = float(contact.dist)
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        forces[index] = force
        frictions[index] = model.geom_friction[contact.geom1]
        valid[index] = 1
    return {
        "geom_ids": geom_ids,
        "positions": positions,
        "normals": normals,
        "distances": distances,
        "forces": forces,
        "frictions": frictions,
        "valid": valid,
        "count": np.asarray(count, dtype=np.int32),
    }


def _event_records(
    *,
    times: np.ndarray,
    body_names: Sequence[str],
    geom_names: Sequence[str],
    body_xpos: np.ndarray,
    body_linear_velocity: np.ndarray,
    contact_valid: np.ndarray,
    contact_geom_ids: np.ndarray,
    qvel: np.ndarray,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    previous_pairs: set[Tuple[int, int]] = set()
    speed = np.linalg.norm(body_linear_velocity[:, 1:, :], axis=-1)
    for frame in range(len(times)):
        pairs: set[Tuple[int, int]] = set()
        for slot in range(contact_valid.shape[1]):
            if not contact_valid[frame, slot]:
                continue
            pair = tuple(int(v) for v in contact_geom_ids[frame, slot])
            pairs.add(pair)
            if pair not in previous_pairs:
                geom_a, geom_b = pair
                records.append(
                    {
                        "type": "collision",
                        "frame": frame,
                        "time": float(times[frame]),
                        "object_ids": [
                            geom_names[geom_a] if 0 <= geom_a < len(geom_names) else str(geom_a),
                            geom_names[geom_b] if 0 <= geom_b < len(geom_names) else str(geom_b),
                        ],
                        "before": None,
                        "after": {
                            "contact": True,
                            "position": body_xpos[frame].tolist(),
                        },
                    }
                )
        if pairs and frame > 0:
            if np.any(speed[frame] < 0.03) and np.any(speed[frame - 1] > 0.15):
                records.append(
                    {
                        "type": "stop",
                        "frame": frame,
                        "time": float(times[frame]),
                        "object_ids": body_names[1:],
                        "before": {"speed": speed[frame - 1].tolist()},
                        "after": {"speed": speed[frame].tolist()},
                    }
                )
        if frame > 0 and np.any(np.abs(qvel[frame]) < 0.02) and np.any(np.abs(qvel[frame - 1]) > 0.15):
            records.append(
                {
                    "type": "velocity_zero_crossing",
                    "frame": frame,
                    "time": float(times[frame]),
                    "object_ids": body_names[1:],
                    "before": {"qvel": qvel[frame - 1].tolist()},
                    "after": {"qvel": qvel[frame].tolist()},
                }
            )
        previous_pairs = pairs
    return records


def _token_alignment(
    *,
    duration_seconds: float,
    video_fps: int,
    token_count: int = 256,
    tokens_per_block: int = 32,
) -> Dict[str, Any]:
    num_frames = int(round(duration_seconds * video_fps)) + 1
    num_blocks = token_count // tokens_per_block
    frame_edges = np.linspace(0, num_frames - 1, num_blocks + 1, dtype=int)
    blocks = []
    for block_id in range(num_blocks):
        blocks.append(
            {
                "block_id": block_id,
                "token_start": block_id * tokens_per_block,
                "token_end": (block_id + 1) * tokens_per_block,
                "frame_start": int(frame_edges[block_id]),
                "frame_end": int(frame_edges[block_id + 1]),
                "time_start": float(frame_edges[block_id] / video_fps),
                "time_end": float(frame_edges[block_id + 1] / video_fps),
            }
        )
    return {
        "token_count": token_count,
        "tokens_per_block": tokens_per_block,
        "num_blocks": num_blocks,
        "video_fps": video_fps,
        "num_frames": num_frames,
        "blocks": blocks,
        "token_values_available": False,
    }


def generate_trajectory(
    *,
    family: str,
    index: int,
    output_root: Path,
    master_seed: int,
    config: Mapping[str, Any] | None = None,
    render_video: bool = True,
) -> Dict[str, Any]:
    settings = dict(DEFAULT_CONFIG)
    if config:
        settings.update(config)
    split = split_for_index(index)
    trajectory_id = f"{family}_{index:04d}"
    seed = int(master_seed + index * 1009 + SCENARIO_FAMILIES.index(family) * 7919)
    rng = np.random.default_rng(seed)
    camera = tuple(float(v) for v in (
        rng.uniform(4.0, 6.0),
        rng.uniform(-4.5, -2.5),
        rng.uniform(2.7, 4.5),
    ))
    scenario = build_scenario(family, rng, split=split, camera=camera)
    model = mujoco.MjModel.from_xml_string(scenario.xml)
    data = mujoco.MjData(model)
    _joint_initialise(model, data, scenario.joint_init)

    state_step = max(1, int(round(1.0 / (settings["sim_timestep"] * settings["state_fps"]))))
    video_step = max(1, int(round(1.0 / (settings["sim_timestep"] * settings["video_fps"]))))
    sim_steps = int(round(settings["duration_seconds"] / settings["sim_timestep"]))
    state_indices = list(range(0, sim_steps + 1, state_step))
    if state_indices[-1] != sim_steps:
        state_indices.append(sim_steps)
    video_indices = set(range(0, sim_steps + 1, video_step))

    body_names = _body_names(model)
    geom_names = _geom_names(model)
    n_state = len(state_indices)
    nbody = model.nbody
    max_contacts = int(settings["max_contacts"])
    arrays: Dict[str, np.ndarray] = {
        "time": np.zeros(n_state, dtype=np.float64),
        "qpos": np.zeros((n_state, model.nq), dtype=np.float64),
        "qvel": np.zeros((n_state, model.nv), dtype=np.float64),
        "qacc": np.zeros((n_state, model.nv), dtype=np.float64),
        "act": np.zeros((n_state, model.na), dtype=np.float64),
        "ctrl": np.zeros((n_state, model.nu), dtype=np.float64),
        "body_xpos": np.zeros((n_state, nbody, 3), dtype=np.float64),
        "body_xquat": np.zeros((n_state, nbody, 4), dtype=np.float64),
        "body_linear_velocity": np.zeros((n_state, nbody, 3), dtype=np.float64),
        "body_angular_velocity": np.zeros((n_state, nbody, 3), dtype=np.float64),
        "contact_geom_ids": np.full((n_state, max_contacts, 2), -1, dtype=np.int32),
        "contact_positions": np.zeros((n_state, max_contacts, 3), dtype=np.float32),
        "contact_normals": np.zeros((n_state, max_contacts, 3), dtype=np.float32),
        "contact_distances": np.zeros((n_state, max_contacts), dtype=np.float32),
        "contact_forces": np.zeros((n_state, max_contacts, 6), dtype=np.float32),
        "contact_frictions": np.zeros((n_state, max_contacts, 3), dtype=np.float32),
        "contact_valid": np.zeros((n_state, max_contacts), dtype=np.uint8),
        "contact_count": np.zeros(n_state, dtype=np.int32),
        "qfrc_constraint": np.zeros((n_state, model.nv), dtype=np.float64),
        "qfrc_passive": np.zeros((n_state, model.nv), dtype=np.float64),
        "qfrc_actuator": np.zeros((n_state, model.nv), dtype=np.float64),
        "qfrc_applied": np.zeros((n_state, model.nv), dtype=np.float64),
        "xfrc_applied": np.zeros((n_state, nbody, 6), dtype=np.float64),
        "kinetic_energy": np.zeros(n_state, dtype=np.float64),
        "potential_energy": np.zeros(n_state, dtype=np.float64),
        "total_energy": np.zeros(n_state, dtype=np.float64),
        "linear_momentum": np.zeros((n_state, 3), dtype=np.float64),
        "angular_momentum": np.zeros((n_state, 3), dtype=np.float64),
        "external_work": np.zeros(n_state, dtype=np.float64),
        "M_state": np.ones(n_state, dtype=np.uint8),
        "M_conservative": np.full(n_state, int(scenario.conservative_candidate), dtype=np.uint8),
        "M_momentum": np.full(n_state, int(scenario.momentum_candidate), dtype=np.uint8),
    }

    previous_xpos = np.zeros((nbody, 3), dtype=np.float64)
    previous_xquat = np.zeros((nbody, 4), dtype=np.float64)
    previous_time = 0.0
    state_cursor = 0
    rendered_frames: List[np.ndarray] = []
    renderer = None
    if render_video:
        renderer = mujoco.Renderer(
            model,
            height=int(settings["render_height"]),
            width=int(settings["render_width"]),
        )

    for sim_step in range(sim_steps + 1):
        if sim_step in state_indices:
            data_time = float(data.time)
            xpos = np.asarray(data.xpos, dtype=np.float64).copy()
            xquat = np.asarray(data.xquat, dtype=np.float64).copy()
            dt = data_time - previous_time
            if state_cursor == 0 or dt <= 0:
                linear = np.zeros((nbody, 3), dtype=np.float64)
                angular = np.zeros((nbody, 3), dtype=np.float64)
            else:
                linear = (xpos - previous_xpos) / dt
                angular = np.zeros((nbody, 3), dtype=np.float64)
                for body_id in range(1, nbody):
                    angular[body_id] = _angular_velocity(
                        previous_xquat[body_id], xquat[body_id], dt
                    )
            contacts = _contact_snapshot(model, data, max_contacts)
            kinetic, potential, total, linear_momentum, angular_momentum = _energy_and_momentum(
                model, data, linear, angular
            )
            arrays["time"][state_cursor] = data_time
            arrays["qpos"][state_cursor] = data.qpos
            arrays["qvel"][state_cursor] = data.qvel
            arrays["qacc"][state_cursor] = data.qacc
            if model.na:
                arrays["act"][state_cursor] = data.act
            if model.nu:
                arrays["ctrl"][state_cursor] = data.ctrl
            arrays["body_xpos"][state_cursor] = xpos
            arrays["body_xquat"][state_cursor] = xquat
            arrays["body_linear_velocity"][state_cursor] = linear
            arrays["body_angular_velocity"][state_cursor] = angular
            for key in ("geom_ids", "positions", "normals", "distances", "forces", "frictions", "valid"):
                target_key = {
                    "geom_ids": "contact_geom_ids",
                    "positions": "contact_positions",
                    "normals": "contact_normals",
                    "distances": "contact_distances",
                    "forces": "contact_forces",
                    "frictions": "contact_frictions",
                    "valid": "contact_valid",
                }[key]
                arrays[target_key][state_cursor] = contacts[key]
            arrays["contact_count"][state_cursor] = contacts["count"]
            arrays["qfrc_constraint"][state_cursor] = data.qfrc_constraint
            arrays["qfrc_passive"][state_cursor] = data.qfrc_passive
            arrays["qfrc_actuator"][state_cursor] = data.qfrc_actuator
            arrays["qfrc_applied"][state_cursor] = data.qfrc_applied
            arrays["xfrc_applied"][state_cursor] = data.xfrc_applied
            arrays["kinetic_energy"][state_cursor] = kinetic
            arrays["potential_energy"][state_cursor] = potential
            arrays["total_energy"][state_cursor] = total
            arrays["linear_momentum"][state_cursor] = linear_momentum
            arrays["angular_momentum"][state_cursor] = angular_momentum
            if state_cursor:
                work = float(np.dot(data.qfrc_applied, data.qvel) * dt)
                arrays["external_work"][state_cursor] = arrays["external_work"][state_cursor - 1] + work
            previous_xpos = xpos
            previous_xquat = xquat
            previous_time = data_time
            state_cursor += 1
        if renderer is not None and sim_step in video_indices:
            renderer.update_scene(data, camera="main")
            rendered_frames.append(np.asarray(renderer.render()).copy())
        if sim_step < sim_steps:
            mujoco.mj_step(model, data)

    output_root = Path(output_root)
    for directory in (
        output_root / "state",
        output_root / "events",
        output_root / "cameras",
        output_root / "mjcf" / family,
        output_root / "video_8fps",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    mjcf_path = output_root / "mjcf" / family / f"{trajectory_id}.xml"
    mjcf_path.write_text(scenario.xml, encoding="utf-8")
    state_path = output_root / "state" / f"{trajectory_id}.npz"
    np.savez_compressed(
        state_path,
        **arrays,
        body_names=np.asarray(body_names),
        geom_names=np.asarray(geom_names),
        body_mass=np.asarray(model.body_mass),
        body_inertia=np.asarray(model.body_inertia),
        geom_friction=np.asarray(model.geom_friction),
        gravity=np.asarray(model.opt.gravity),
        timestep=np.asarray(settings["sim_timestep"]),
        state_fps=np.asarray(settings["state_fps"]),
        video_fps=np.asarray(settings["video_fps"]),
    )
    event_records = _event_records(
        times=arrays["time"],
        body_names=body_names,
        geom_names=geom_names,
        body_xpos=arrays["body_xpos"],
        body_linear_velocity=arrays["body_linear_velocity"],
        contact_valid=arrays["contact_valid"],
        contact_geom_ids=arrays["contact_geom_ids"],
        qvel=arrays["qvel"],
    )
    event_path = output_root / "events" / f"{trajectory_id}.json"
    event_path.write_text(
        json.dumps(
            {
                "trajectory_id": trajectory_id,
                "body_names": body_names,
                "geom_names": geom_names,
                "events": _jsonable(event_records),
                "contact_sampling": "state_fps",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    camera_path = output_root / "cameras" / f"{trajectory_id}.json"
    camera_path.write_text(
        json.dumps(
            {
                "trajectory_id": trajectory_id,
                "camera": _jsonable(_camera_metadata(model)),
                "render_width": int(settings["render_width"]),
                "render_height": int(settings["render_height"]),
                "video_fps": int(settings["video_fps"]),
                "camera_seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    video_path = output_root / "video_8fps" / f"{trajectory_id}.mp4"
    if renderer is not None:
        try:
            import imageio.v2 as imageio

            with imageio.get_writer(video_path, fps=int(settings["video_fps"]), codec="libx264", quality=7) as writer:
                for frame in rendered_frames:
                    writer.append_data(frame)
        except Exception:
            if video_path.exists():
                video_path.unlink()
            raise

    alignment = _token_alignment(
        duration_seconds=float(settings["duration_seconds"]),
        video_fps=int(settings["video_fps"]),
    )
    entry = {
        "trajectory_id": trajectory_id,
        "family": family,
        "template_id": scenario.template_id,
        "difficulty": scenario.difficulty,
        "split": split,
        "seed": seed,
        "duration_seconds": float(settings["duration_seconds"]),
        "sim_timestep": float(settings["sim_timestep"]),
        "state_fps": int(settings["state_fps"]),
        "video_fps": int(settings["video_fps"]),
        "num_state_frames": int(n_state),
        "num_video_frames": int(len(rendered_frames)),
        "max_contacts": max_contacts,
        "conservative_candidate": bool(scenario.conservative_candidate),
        "momentum_candidate": bool(scenario.momentum_candidate),
        "body_roles": _jsonable(scenario.body_roles),
        "parameters": _jsonable(scenario.parameters),
        "paths": {
            "mjcf": str(mjcf_path.relative_to(output_root)),
            "state": str(state_path.relative_to(output_root)),
            "events": str(event_path.relative_to(output_root)),
            "camera": str(camera_path.relative_to(output_root)),
            "video": str(video_path.relative_to(output_root)) if video_path.exists() else None,
        },
        "token_alignment": alignment,
        "sha256": {
            "mjcf": _hash_file(mjcf_path),
            "state": _hash_file(state_path),
            "events": _hash_file(event_path),
            "camera": _hash_file(camera_path),
            "video": _hash_file(video_path) if video_path.exists() else None,
        },
    }
    return entry


def write_manifest(entries: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["split"]), []).append(entry)
    manifest_dir = Path(output_root) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, split_entries in grouped.items():
        path = manifest_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for entry in sorted(split_entries, key=lambda row: str(row["trajectory_id"])):
                handle.write(json.dumps(_jsonable(entry), sort_keys=True) + "\n")


def generate_dataset(
    *,
    output_root: Path,
    families: Sequence[str] = SCENARIO_FAMILIES,
    trajectories_per_family: int = 300,
    master_seed: int = 20261007,
    config: Mapping[str, Any] | None = None,
    render_video: bool = True,
    limit: int | None = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    entries: List[Mapping[str, Any]] = []
    completed = 0
    for family in families:
        for index in range(trajectories_per_family):
            if limit is not None and completed >= limit:
                break
            entry = generate_trajectory(
                family=family,
                index=index,
                output_root=output_root,
                master_seed=master_seed,
                config=config,
                render_video=render_video,
            )
            entries.append(entry)
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "trajectory_complete",
                        "trajectory_id": entry["trajectory_id"],
                        "count": completed,
                        "family": family,
                        "split": entry["split"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if limit is not None and completed >= limit:
            break
    write_manifest(entries, output_root)
    summary = {
        "dataset": "mujoco_wam_v0",
        "version": "0.1",
        "master_seed": master_seed,
        "families": list(families),
        "trajectories_per_family_requested": trajectories_per_family,
        "trajectories_generated": len(entries),
        "splits": {
            split: sum(1 for entry in entries if entry["split"] == split)
            for split in ("train", "val", "test_interpolation", "test_ood")
        },
        "render_video": render_video,
        "config": _jsonable({**DEFAULT_CONFIG, **(config or {})}),
        "remote_generation_required": True,
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
