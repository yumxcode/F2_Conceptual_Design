"""The concept specification: defaults, YAML loading and validation.

A concept is a nested mapping. Every key has a default here; a YAML file states
only what it changes. Unknown keys raise, so a typo cannot silently fall back
to a default.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

#: Every key a concept may state, with its default. Units are SI unless the key
#: says otherwise. Ratios are fractions of standing height unless noted.
DEFAULTS: dict = {
    "name": "concept",
    "body": {
        "height_m": 1.50,            # standing height, sole to crown
        "leg_ratio": 0.45,           # (thigh + shank) / height: hip joint -> ankle joint
        "thigh_fraction": 0.50,      # thigh / (thigh + shank)
        "ankle_height_ratio": 0.045,  # ankle joint above the sole
        "head_neck_ratio": 0.18,     # shoulder joint -> crown
        "waist_fraction": 0.25,      # waist joint above the hips, fraction of hip -> shoulder
        "upper_arm_ratio": 0.17,
        "forearm_ratio": 0.14,       # elbow -> wrist
        "hand_ratio": 0.07,          # wrist -> grasp point
        "foot_length_ratio": 0.15,
        "foot_width_ratio": 0.055,
        "heel_fraction": 0.25,       # share of the foot behind the ankle
        "hip_width_ratio": 0.13,     # between the hip joint centres
        "shoulder_width_ratio": 0.26,
    },
    "dof": {
        "waist": ["yaw"],            # any of yaw, roll, pitch
        "wrist": 1,                  # 0..3 per arm
    },
    "mechanism": {
        "ankle": "parallel",         # serial | parallel (two motors share pitch+roll)
        "ankle_motor_location": "knee",  # ankle | knee (remote, linkage-driven)
        "knee_motor_location": "knee",   # knee | hip (remote, linkage-driven)
    },
    "mass": {
        "structural_scale": 1.0,     # multiplies every structural trend mass
        "trunk_shell_fraction": 0.17,  # trunk frame, covers, wiring, boards / total mass
        "head_fraction": 0.018,      # head shell / total mass (population median)
        "end_effector_kg": 0.6,      # each hand / gripper
        "compute_kg": 1.5,
        "sensors_kg": 1.0,           # cameras, lidar, IMU (in the head)
        "battery_height_fraction": 0.45,  # battery centre along hip -> shoulder
    },
    "payload": {
        "hands_total_kg": 5.0,       # carried with both hands (squat_lift, arm_lift, carry)
    },
    "electrical": {
        "bus_voltage_V": 48.0,
        "compute_power_W": 100.0,
        "sensor_power_W": 30.0,
        "drive_efficiency": 0.85,    # motor + gearbox + inverter, positive work
        "regen_efficiency": 0.0,     # share of negative work returned to the pack
        "copper_loss_W_per_kg": 30.0,  # Joule loss of an actuator at its continuous torque, per kg
        "battery_Wh_per_kg": 160.0,  # pack level
        "battery_Wh_per_L": 300.0,
        "battery_max_C": 5.0,        # continuous discharge
        "battery_reserve": 0.2,      # unusable share (depth of discharge, ageing)
        "runtime_h": 2.0,
    },
    "mission": [                     # how the runtime is spent; shares sum to 1
        {"task": "walk", "share": 0.6},
        {"task": "stand", "share": 0.3},
        {"task": "stairs", "share": 0.05},
        {"task": "squat_lift", "share": 0.05},
    ],
    "tasks": {
        "stand": {"enabled": True, "knee_bend_deg": 15.0, "duration_s": 2.0,
                  "hold_payload": False},
        "walk": {"enabled": True, "speed": 1.0, "max_speed": 1.5,
                 "step_length_ratio": None,     # step length / leg; None = Froude scaling
                 "swing_height_ratio": 0.05,    # foot clearance / height
                 "max_leg_extension": 0.92,     # hip-ankle / leg at heel strike (all gaits)
                 "step_width_ratio": None},     # None = hip width
        "stairs": {"enabled": True, "step_height_m": 0.15, "tread_m": 0.28,
                   "step_time_s": None},        # None = scaled with sqrt(leg/g)
        "squat_lift": {"enabled": True, "depth": 0.55,    # lowest hip height / leg
                       "duration_s": 1.2, "carry_payload": True},
        "stoop_lift": {"enabled": True, "lean_deg": 80.0, "knee_bend_deg": 20.0,
                       "duration_s": 1.5, "carry_payload": True},
        "run": {"enabled": True, "speed": 2.0, "duty_factor": 0.35,
                "touchdown_extension": 0.95},   # hip-ankle / leg at touchdown
        "arm_lift": {"enabled": True, "duration_s": 1.0},
        "turn": {"enabled": True, "yaw_rate_rad_s": 2.0, "foot_friction": 0.6},
    },
    "balance": {
        # Share of the way to the toe / sole edge the ankle alone must be able to
        # push the CoP with the full weight on one foot. 1.0 = full authority;
        # published humanoids install about half (benchmark), relying on the hips.
        "ankle_cop_authority": 0.5,
    },
    "actuators": {
        "mode": "trend",             # trend (continuous sizing) | catalog (real parts)
        "torque_safety_factor": 1.3,
        "speed_safety_factor": 1.15,
        "thermal_safety_factor": 1.2,
        "voltage_limit_factor": 2.0,  # torque-speed envelope; 1 = catalogue straight line
        "frontier": "p90",           # p90 | max: how close to the best catalogued density
        "peak_to_continuous": {"QDD": 3.0, "MidGear": 2.5, "Harmonic": 2.0, "default": 2.5},
        "categories": ["QDD", "MidGear", "Harmonic"],
        "extra_catalog": None,       # YAML/CSV of your own suppliers' parts
        # Agility floor: every joint must also move this fast while carrying
        # `speed_floor_load` of its peak torque (balance recovery, side steps,
        # arm swings) -- motions the reference tasks are too gentle to ask for.
        "speed_floor_rad_s": {"hip_pitch": 8.0, "hip_roll": 6.0, "hip_yaw": 6.0,
                              "knee": 10.0, "ankle_pitch": 8.0, "ankle_roll": 6.0,
                              "ankle_drive": 8.0, "waist_yaw": 4.0, "waist_roll": 3.0,
                              "waist_pitch": 3.0, "shoulder_pitch": 6.0,
                              "shoulder_roll": 6.0, "shoulder_yaw": 6.0, "elbow": 6.0,
                              "wrist_yaw": 6.0, "wrist_pitch": 6.0, "wrist_roll": 6.0},
        "speed_floor_load": 0.3,
        "groups": {},                # name -> [joint, ...]: one part number for the group
        "top_n": 3,
    },
    "convergence": {"max_iter": 40, "tol": 3e-3, "max_mass_kg": 400.0},
}

JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
          "waist_yaw", "waist_roll", "waist_pitch",
          "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
          "wrist_yaw", "wrist_pitch", "wrist_roll")

#: Joints whose actuators are physically per side (counted twice).
PAIRED = {"hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
          "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
          "wrist_yaw", "wrist_pitch", "wrist_roll"}

TASK_NAMES = tuple(DEFAULTS["tasks"])

_OPEN_KEYS = {("actuators", "groups"), ("actuators", "peak_to_continuous"),
              ("actuators", "speed_floor_rad_s")}


def _merge(base: Any, over: Any, path: tuple = ()) -> Any:
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            if k not in base and path not in _OPEN_KEYS:
                where = ".".join(path + (k,))
                raise KeyError(f"unknown concept key {where!r}")
            out[k] = _merge(base[k], v, path + (k,)) if k in base else v
        return out
    return copy.deepcopy(over)


def make_spec(overrides: Optional[dict] = None) -> dict:
    """Defaults merged with ``overrides`` (validated)."""
    spec = _merge(copy.deepcopy(DEFAULTS), overrides or {})
    validate(spec)
    return spec


def load_spec(path: str | Path, overrides: Optional[dict] = None) -> dict:
    """Read a concept YAML; ``overrides`` (dotted or nested) are applied on top."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.pop("sweep", None)
    spec = _merge(copy.deepcopy(DEFAULTS), data)
    if overrides:
        spec = _merge(spec, overrides)
    validate(spec)
    return spec


def set_dotted(spec: dict, dotted: str, value: Any) -> dict:
    """Copy of ``spec`` with ``a.b.c = value``."""
    out = copy.deepcopy(spec)
    node = out
    keys = dotted.split(".")
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"unknown concept key {dotted!r}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"unknown concept key {dotted!r}")
    node[keys[-1]] = value
    return out


def get_dotted(spec: dict, dotted: str) -> Any:
    node = spec
    for k in dotted.split("."):
        node = node[k]
    return node


def validate(spec: dict) -> None:
    b = spec["body"]
    if not 0.4 <= b["height_m"] <= 2.5:
        raise ValueError(f"body.height_m={b['height_m']} is outside 0.4-2.5 m")
    if not 0.25 <= b["leg_ratio"] <= 0.62:
        raise ValueError(f"body.leg_ratio={b['leg_ratio']} is outside 0.25-0.62")
    if not 0.3 <= b["thigh_fraction"] <= 0.7:
        raise ValueError("body.thigh_fraction must be within 0.3-0.7")
    trunk = 1.0 - b["leg_ratio"] - b["ankle_height_ratio"] - b["head_neck_ratio"]
    if trunk < 0.15:
        raise ValueError(
            f"hip-to-shoulder is only {trunk:.2f} of the height: lower leg_ratio "
            f"or head_neck_ratio")
    for w in spec["dof"]["waist"]:
        if w not in ("yaw", "roll", "pitch"):
            raise ValueError(f"dof.waist: unknown axis {w!r}")
    if spec["dof"]["wrist"] not in (0, 1, 2, 3):
        raise ValueError("dof.wrist must be 0-3")
    m = spec["mechanism"]
    if m["ankle"] not in ("serial", "parallel"):
        raise ValueError("mechanism.ankle must be serial or parallel")
    if m["ankle_motor_location"] not in ("ankle", "knee"):
        raise ValueError("mechanism.ankle_motor_location must be ankle or knee")
    if m["knee_motor_location"] not in ("knee", "hip"):
        raise ValueError("mechanism.knee_motor_location must be knee or hip")
    if spec["actuators"]["mode"] not in ("trend", "catalog"):
        raise ValueError("actuators.mode must be trend or catalog")
    shares = sum(float(x["share"]) for x in spec["mission"])
    if abs(shares - 1.0) > 1e-6:
        raise ValueError(f"mission shares sum to {shares:.3f}, not 1")
    for x in spec["mission"]:
        if x["task"] not in TASK_NAMES:
            raise ValueError(f"mission: unknown task {x['task']!r}")


def active_joints(spec: dict) -> list[str]:
    """Joints this concept actually has."""
    wrist = ["wrist_yaw", "wrist_pitch", "wrist_roll"][: spec["dof"]["wrist"]]
    waist = [f"waist_{a}" for a in spec["dof"]["waist"]]
    legs = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
    arms = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"] + wrist
    return legs + waist + arms


def dof_count(spec: dict) -> int:
    return sum(2 if j in PAIRED else 1 for j in active_joints(spec))
