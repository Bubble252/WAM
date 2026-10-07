"""Small, deterministic MuJoCo scene templates for WAM v0.1.

The generator deliberately keeps the scene XML human-readable.  Each
trajectory receives its own parameterized XML file so that a manifest can
replay the exact simulator configuration without relying on hidden Python
state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np


SCENARIO_FAMILIES = (
    "free_fall_projectile",
    "elastic_bounce",
    "inelastic_collision",
    "sliding_friction",
    "rolling_body",
    "spring_oscillator",
    "simple_double_pendulum",
    "ramp_obstacle",
    "stack_topple",
    "collision_chain",
    "hinge_lever",
    "compositional_scene",
)


@dataclass(frozen=True)
class Scenario:
    family: str
    template_id: str
    xml: str
    joint_init: Mapping[str, Tuple[np.ndarray, np.ndarray]]
    body_roles: Mapping[str, str]
    parameters: Mapping[str, Any]
    conservative_candidate: bool
    momentum_candidate: bool
    difficulty: str


def _fmt(values: Any) -> str:
    if isinstance(values, (tuple, list, np.ndarray)):
        return " ".join(f"{float(v):.8g}" for v in values)
    return f"{float(values):.8g}"


def _header(
    name: str,
    *,
    timestep: float,
    gravity: Tuple[float, float, float],
    camera_pos: Tuple[float, float, float],
    integrator: str = "implicitfast",
) -> str:
    return f"""<mujoco model="{name}">
  <compiler angle="radian" inertiafromgeom="true" coordinate="local"/>
  <option timestep="{timestep:.12g}" gravity="{_fmt(gravity)}"
          integrator="{integrator}" iterations="50" tolerance="1e-8"/>
  <size njmax="2000" nconmax="2000"/>
  <visual>
    <global offwidth="128" offheight="128"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <material name="floor_mat" rgba="0.18 0.20 0.24 1"/>
    <material name="red_mat" rgba="0.85 0.18 0.12 1"/>
    <material name="blue_mat" rgba="0.12 0.34 0.88 1"/>
    <material name="green_mat" rgba="0.18 0.68 0.28 1"/>
    <material name="yellow_mat" rgba="0.92 0.70 0.12 1"/>
    <material name="purple_mat" rgba="0.55 0.22 0.75 1"/>
  </asset>
  <worldbody>
    <camera name="main" mode="trackcom" pos="{_fmt(camera_pos)}"/>
"""


def _footer() -> str:
    return """  </worldbody>
</mujoco>
"""


def _floor(*, friction: float = 0.7, z: float = 0.0) -> str:
    return (
        f'    <geom name="floor" type="plane" pos="0 0 {z:.8g}" '
        f'size="10 10 0.1" friction="{friction:.8g} 0.005 0.0001" '
        'material="floor_mat"/>\n'
    )


def _free_body(
    name: str,
    pos: Tuple[float, float, float],
    geom: str,
) -> str:
    return (
        f'    <body name="{name}" pos="{_fmt(pos)}">\n'
        f'      <freejoint name="{name}_free"/>\n'
        f"      {geom}\n"
        "    </body>\n"
    )


def _free_init(
    pos: Tuple[float, float, float],
    vel: Tuple[float, float, float],
    angular: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray((*pos, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    qvel = np.asarray((*vel, *angular), dtype=np.float64)
    return qpos, qvel


def _camera(rng: np.random.Generator, *, distance: float = 5.0) -> Tuple[float, float, float]:
    azimuth = rng.uniform(-20.0, 20.0)
    x = distance * math.cos(math.radians(azimuth))
    y = -distance * math.sin(math.radians(azimuth)) - distance * 0.7
    z = rng.uniform(2.8, 4.2)
    return (float(x), float(y), float(z))


def _common_parameters(
    rng: np.random.Generator,
    *,
    split: str,
    friction: Tuple[float, float] = (0.0, 0.8),
) -> Dict[str, float]:
    wide = split == "test_ood"
    mass_lo, mass_hi = (0.45, 2.8) if wide else (0.75, 1.4)
    friction_lo, friction_hi = friction
    if wide:
        friction_hi = max(friction_hi, 1.25)
    return {
        "mass_scale": float(rng.uniform(mass_lo, mass_hi)),
        "friction": float(rng.uniform(friction_lo, friction_hi)),
        "gravity": float(rng.uniform(8.5, 11.5) if wide else 9.81),
    }


def _free_fall_projectile(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.0, 0.1))
    x = rng.uniform(-1.0, 1.0)
    y = rng.uniform(-0.5, 0.5)
    z = rng.uniform(2.2, 4.0)
    vel = (rng.uniform(0.5, 2.2), rng.uniform(-0.8, 0.8), rng.uniform(0.2, 1.5))
    xml = (
        _header("free_fall_projectile", timestep=1.0 / 240.0,
                gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
        + _floor(friction=0.05)
        + _free_body(
            "target",
            (x, y, z),
            f'<geom name="target_geom" type="sphere" size="0.22" '
            f'mass="{p["mass_scale"]:.8g}" material="red_mat"/>',
        )
        + _footer()
    )
    return Scenario(
        "free_fall_projectile", "sphere_v0", xml,
        {"target_free": _free_init((x, y, z), vel)},
        {"target": "projectile"},
        {**p, "initial_position": [x, y, z], "initial_velocity": list(vel)},
        True, False, "basic",
    )


def _elastic_bounce(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.0, 0.08))
    x, y, z = rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3), rng.uniform(1.2, 3.0)
    vz = rng.uniform(-1.0, -0.1)
    xml = (
        _header("elastic_bounce", timestep=1.0 / 240.0,
                gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
        + _floor(friction=0.02)
        + _free_body(
            "target",
            (x, y, z),
            f'<geom name="target_geom" type="sphere" size="0.28" '
            f'mass="{p["mass_scale"]:.8g}" friction="0.01 0.001 0.0001" '
            'solref="0.002 1" material="blue_mat"/>',
        )
        + _footer()
    )
    return Scenario(
        "elastic_bounce", "sphere_floor_v0", xml,
        {"target_free": _free_init((x, y, z), (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), vz))},
        {"target": "bouncing_body"},
        {**p, "contact_model": "low_friction_sphere_floor"},
        True, False, "basic",
    )


def _inelastic_collision(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.0, 0.05))
    m1 = p["mass_scale"]
    m2 = float(rng.uniform(0.5, 2.8) if split == "test_ood" else rng.uniform(0.8, 1.8))
    z = rng.uniform(0.45, 0.7)
    xml = (
        _header("inelastic_collision", timestep=1.0 / 240.0,
                gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
        + _floor(friction=0.03)
        + _free_body(
            "target_a", (-1.15, 0.0, z),
            f'<geom name="target_a_geom" type="sphere" size="0.32" mass="{m1:.8g}" '
            'friction="0.01 0.001 0.0001" material="green_mat"/>',
        )
        + _free_body(
            "target_b", (1.15, 0.0, z),
            f'<geom name="target_b_geom" type="sphere" size="0.38" mass="{m2:.8g}" '
            'friction="0.01 0.001 0.0001" material="yellow_mat"/>',
        )
        + _footer()
    )
    return Scenario(
        "inelastic_collision", "two_spheres_v0", xml,
        {
            "target_a_free": _free_init((-1.15, 0.0, z), (rng.uniform(1.5, 2.8), 0.0, 0.0)),
            "target_b_free": _free_init((1.15, 0.0, z), (rng.uniform(-0.5, 0.0), 0.0, 0.0)),
        },
        {"target_a": "collision_body_a", "target_b": "collision_body_b"},
        {**p, "mass_a": m1, "mass_b": m2},
        False, True, "contact",
    )


def _sliding_friction(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.15, 0.9))
    x, y = rng.uniform(-1.5, 1.5), rng.uniform(-0.4, 0.4)
    z = 0.32
    xml = (
        _header("sliding_friction", timestep=1.0 / 240.0,
                gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
        + _floor(friction=p["friction"])
        + _free_body(
            "target", (x, y, z),
            f'<geom name="target_geom" type="box" size="0.32 0.22 0.20" '
            f'mass="{p["mass_scale"]:.8g}" friction="{p["friction"]:.8g} 0.01 0.001" '
            'material="red_mat"/>',
        )
        + _footer()
    )
    return Scenario(
        "sliding_friction", "box_floor_v0", xml,
        {"target_free": _free_init((x, y, z), (rng.uniform(1.0, 3.0), rng.uniform(-0.2, 0.2), 0.0))},
        {"target": "sliding_body"},
        {**p, "damping": 0.0},
        False, False, "contact",
    )


def _rolling_body(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.2, 0.9))
    x, y, z = rng.uniform(-1.2, 1.2), rng.uniform(-0.3, 0.3), 0.36
    xml = (
        _header("rolling_body", timestep=1.0 / 240.0,
                gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
        + _floor(friction=p["friction"])
        + _free_body(
            "target", (x, y, z),
            f'<geom name="target_geom" type="cylinder" size="0.28 0.32" '
            f'quat="0.7071068 0 0.7071068 0" mass="{p["mass_scale"]:.8g}" '
            f'friction="{p["friction"]:.8g} 0.02 0.01" material="purple_mat"/>',
        )
        + _footer()
    )
    speed = rng.uniform(0.7, 2.2)
    return Scenario(
        "rolling_body", "cylinder_floor_v0", xml,
        {"target_free": _free_init((x, y, z), (speed, 0.0, 0.0), (0.0, rng.uniform(-8.0, -2.0), 0.0))},
        {"target": "rolling_body"},
        {**p, "damping": 0.0},
        False, False, "contact",
    )


def _spring_oscillator(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.0, 0.0))
    stiffness = float(rng.uniform(8.0, 35.0) if split == "test_ood" else rng.uniform(12.0, 25.0))
    damping = 0.0 if split != "test_ood" else float(rng.uniform(0.0, 0.08))
    xml = (
        _header("spring_oscillator", timestep=1.0 / 240.0,
                gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
        + _floor(friction=0.0)
        + f'''    <body name="target" pos="0 0 0.45">
      <joint name="target_slide" type="slide" axis="1 0 0"
             stiffness="{stiffness:.8g}" damping="{damping:.8g}"/>
      <geom name="target_geom" type="box" size="0.25 0.25 0.25"
            mass="{p["mass_scale"]:.8g}" material="green_mat"/>
    </body>
'''
        + _footer()
    )
    return Scenario(
        "spring_oscillator", "slide_spring_v0", xml,
        {"target_slide": (np.asarray([rng.uniform(-0.55, 0.55)]), np.asarray([rng.uniform(-0.2, 0.2)]))},
        {"target": "spring_mass"},
        {**p, "stiffness": stiffness, "damping": damping},
        damping == 0.0, False, "basic",
    )


def _simple_double_pendulum(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.0, 0.0))
    double = bool(rng.integers(0, 2))
    damping = 0.0 if split != "test_ood" else float(rng.uniform(0.0, 0.05))
    if double:
        xml = (
            _header("double_pendulum", timestep=1.0 / 240.0,
                    gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
            + f'''    <body name="pendulum_a" pos="0 0 2.2">
      <joint name="pendulum_a_hinge" type="hinge" axis="0 1 0" damping="{damping:.8g}"/>
      <geom name="pendulum_a_geom" type="capsule" fromto="0 0 0 0 0 -1.0"
            size="0.12" mass="{p["mass_scale"]:.8g}" material="blue_mat"/>
      <body name="pendulum_b" pos="0 0 -1.0">
        <joint name="pendulum_b_hinge" type="hinge" axis="0 1 0" damping="{damping:.8g}"/>
        <geom name="pendulum_b_geom" type="capsule" fromto="0 0 0 0 0 -0.9"
              size="0.11" mass="{p["mass_scale"] * 0.8:.8g}" material="yellow_mat"/>
      </body>
    </body>
'''
            + _footer()
        )
        joints = {
            "pendulum_a_hinge": (np.asarray([rng.uniform(-0.8, 0.8)]), np.asarray([rng.uniform(-0.2, 0.2)])),
            "pendulum_b_hinge": (np.asarray([rng.uniform(-1.1, 1.1)]), np.asarray([rng.uniform(-0.2, 0.2)])),
        }
    else:
        xml = (
            _header("simple_pendulum", timestep=1.0 / 240.0,
                    gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
            + f'''    <body name="pendulum" pos="0 0 2.2">
      <joint name="pendulum_hinge" type="hinge" axis="0 1 0" damping="{damping:.8g}"/>
      <geom name="pendulum_geom" type="capsule" fromto="0 0 0 0 0 -1.4"
            size="0.13" mass="{p["mass_scale"]:.8g}" material="blue_mat"/>
    </body>
'''
            + _footer()
        )
        joints = {
            "pendulum_hinge": (np.asarray([rng.uniform(-1.0, 1.0)]), np.asarray([rng.uniform(-0.4, 0.4)])),
        }
    return Scenario(
        "simple_double_pendulum", "double" if double else "simple", xml, joints,
        {"pendulum_a": "pendulum_link", "pendulum_b": "pendulum_link", "pendulum": "pendulum_link"},
        {**p, "damping": damping, "double": double},
        damping == 0.0, False, "basic",
    )


def _ramp_obstacle(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.1, 0.8))
    angle = float(rng.uniform(-0.35, 0.35))
    half = 1.8
    quat = (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)
    x, y, z = -2.0, rng.uniform(-0.25, 0.25), 1.1
    xml = (
        _header("ramp_obstacle", timestep=1.0 / 240.0,
                gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
        + _floor(friction=p["friction"])
        + f'    <geom name="ramp" type="box" pos="0 0 {half * 0.18:.8g}" '
          f'size="{half} 1.2 0.18" quat="{_fmt(quat)}" '
          f'friction="{p["friction"]:.8g} 0.01 0.001" material="floor_mat"/>\n'
        + _free_body(
            "target", (x, y, z),
            f'<geom name="target_geom" type="sphere" size="0.22" '
            f'mass="{p["mass_scale"]:.8g}" material="red_mat"/>',
        )
        + _footer()
    )
    return Scenario(
        "ramp_obstacle", "ramp_sphere_v0", xml,
        {"target_free": _free_init((x, y, z), (rng.uniform(1.0, 2.8), 0.0, rng.uniform(-0.2, 0.4)))},
        {"target": "ramp_object"},
        {**p, "ramp_angle": angle},
        False, False, "contact",
    )


def _stack_topple(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.25, 0.9))
    masses = [p["mass_scale"] * float(rng.uniform(0.7, 1.3)) for _ in range(3)]
    top_x = float(rng.uniform(-0.15, 0.15))
    xml = _header(
        "stack_topple", timestep=1.0 / 240.0,
        gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera,
    ) + _floor(friction=p["friction"])
    xml += _free_body("box_0", (0.0, 0.0, 0.28),
                      f'<geom name="box_0_geom" type="box" size="0.42 0.42 0.28" mass="{masses[0]:.8g}" material="blue_mat"/>')
    xml += _free_body("box_1", (0.0, 0.0, 0.84),
                      f'<geom name="box_1_geom" type="box" size="0.38 0.38 0.28" mass="{masses[1]:.8g}" material="green_mat"/>')
    xml += _free_body("box_2", (top_x, 0.0, 1.40),
                      f'<geom name="box_2_geom" type="box" size="0.34 0.34 0.28" mass="{masses[2]:.8g}" material="red_mat"/>')
    xml += _footer()
    return Scenario(
        "stack_topple", "three_box_v0", xml,
        {
            "box_0_free": _free_init((0.0, 0.0, 0.28), (0.0, 0.0, 0.0)),
            "box_1_free": _free_init((0.0, 0.0, 0.84), (0.0, 0.0, 0.0)),
            "box_2_free": _free_init((top_x, 0.0, 1.40), (rng.uniform(-0.5, 0.5), 0.0, 0.0)),
        },
        {"box_0": "support", "box_1": "support", "box_2": "top"},
        {**p, "masses": masses},
        False, False, "compositional",
    )


def _collision_chain(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.0, 0.2))
    xml = (
        _header("collision_chain", timestep=1.0 / 240.0,
                gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
        + _floor(friction=0.08)
    )
    joints: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, x in enumerate((-1.35, 0.0, 1.35)):
        name = f"ball_{i}"
        material = ("blue_mat", "green_mat", "yellow_mat")[i]
        xml += _free_body(
            name, (x, 0.0, 0.35),
            f'<geom name="{name}_geom" type="sphere" size="0.30" '
            f'mass="{p["mass_scale"] * (0.8 + 0.2 * i):.8g}" '
            f'friction="0.03 0.001 0.0001" material="{material}"/>',
        )
        vx = 2.4 if i == 0 else 0.0
        joints[f"{name}_free"] = _free_init((x, 0.0, 0.35), (vx, 0.0, 0.0))
    xml += _footer()
    return Scenario(
        "collision_chain", "three_spheres_v0", xml, joints,
        {"ball_0": "chain_body", "ball_1": "chain_body", "ball_2": "chain_body"},
        p, False, False, "compositional",
    )


def _hinge_lever(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.0, 0.0))
    damping = 0.0 if split != "test_ood" else float(rng.uniform(0.0, 0.08))
    xml = (
        _header("hinge_lever", timestep=1.0 / 240.0,
                gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
        + f'''    <body name="lever" pos="0 0 1.35">
      <joint name="lever_hinge" type="hinge" axis="0 1 0" damping="{damping:.8g}"/>
      <geom name="lever_geom" type="box" fromto="0 0 0 1.8 0 0"
            size="0.12" mass="{p["mass_scale"]:.8g}" material="purple_mat"/>
      <geom name="tip_geom" type="sphere" pos="1.8 0 0" size="0.18"
            mass="{p["mass_scale"] * 0.4:.8g}" material="yellow_mat"/>
    </body>
'''
        + _footer()
    )
    return Scenario(
        "hinge_lever", "hinge_bar_v0", xml,
        {"lever_hinge": (np.asarray([rng.uniform(-0.7, 0.7)]), np.asarray([rng.uniform(-1.2, 1.2)]))},
        {"lever": "hinged_body"},
        {**p, "damping": damping},
        damping == 0.0, False, "basic",
    )


def _compositional_scene(
    rng: np.random.Generator,
    split: str,
    camera: Tuple[float, float, float],
) -> Scenario:
    p = _common_parameters(rng, split=split, friction=(0.15, 0.8))
    xml = (
        _header("compositional_scene", timestep=1.0 / 240.0,
                gravity=(0.0, 0.0, -p["gravity"]), camera_pos=camera)
        + _floor(friction=p["friction"])
        + '    <geom name="obstacle" type="box" pos="0.4 0 0.4" size="0.4 0.8 0.4" material="floor_mat"/>\n'
        + _free_body(
            "ball", (-1.8, 0.0, 0.55),
            f'<geom name="ball_geom" type="sphere" size="0.24" mass="{p["mass_scale"]:.8g}" '
            f'friction="{p["friction"]:.8g} 0.01 0.001" material="red_mat"/>',
        )
        + _free_body(
            "box", (1.4, 0.0, 0.28),
            f'<geom name="box_geom" type="box" size="0.30 0.30 0.28" mass="{p["mass_scale"] * 1.4:.8g}" '
            f'friction="{p["friction"]:.8g} 0.01 0.001" material="blue_mat"/>',
        )
        + _footer()
    )
    return Scenario(
        "compositional_scene", "obstacle_ball_box_v0", xml,
        {
            "ball_free": _free_init((-1.8, 0.0, 0.55), (rng.uniform(1.4, 3.0), 0.0, rng.uniform(0.0, 0.8))),
            "box_free": _free_init((1.4, 0.0, 0.28), (0.0, 0.0, 0.0)),
        },
        {"ball": "moving_body", "box": "obstacle_body"},
        p, False, False, "compositional",
    )


_BUILDERS = {
    "free_fall_projectile": _free_fall_projectile,
    "elastic_bounce": _elastic_bounce,
    "inelastic_collision": _inelastic_collision,
    "sliding_friction": _sliding_friction,
    "rolling_body": _rolling_body,
    "spring_oscillator": _spring_oscillator,
    "simple_double_pendulum": _simple_double_pendulum,
    "ramp_obstacle": _ramp_obstacle,
    "stack_topple": _stack_topple,
    "collision_chain": _collision_chain,
    "hinge_lever": _hinge_lever,
    "compositional_scene": _compositional_scene,
}


def build_scenario(
    family: str,
    rng: np.random.Generator,
    *,
    split: str,
    camera: Tuple[float, float, float] | None = None,
) -> Scenario:
    if family not in _BUILDERS:
        raise KeyError(f"Unknown scenario family: {family}")
    if camera is None:
        camera = _camera(rng)
    return _BUILDERS[family](rng, split, camera)
