"""Published humanoids from Draft's robot survey, as yardsticks.

``datasets/robot_descriptions/data/`` holds, for every surveyed robot, its total
mass, joint-frame span, leg length and the effort/velocity limits written in
its URDF/MJCF. The screening uses them two ways:

* scatter plots: where a concept sits among real robots (mass vs height,
  knee torque vs mass);
* :func:`benchmark`: rebuild each robot's height and leg ratio as a concept,
  close its mass loop, and compare predicted mass and joint torques with what
  the vendor built. That is how far to trust the screening's numbers.

URDF limits are what the vendor *declared*, often the actuator's peak and
sometimes a software limit; treat them as approximate.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Optional

from ..paths import repo_root

#: z_span (ankle -> highest joint frame) as a share of standing height, for the
#: default proportions (ankle 0.045 H, shoulder-to-crown 0.18 H).
Z_SPAN_SHARE = 1.0 - 0.045 - 0.18


@dataclass
class RefRobot:
    key: str
    name: str
    maker: str
    mass_kg: float
    z_span_m: float
    leg_m: float
    dof: int
    limits: dict          # joint class -> (effort Nm, velocity rad/s or None)

    @property
    def height_est_m(self) -> float:
        return self.z_span_m / Z_SPAN_SHARE


_PATTERNS = {
    "knee": r"knee",
    "hip_pitch": r"hip.*pitch|hip_y(?!aw)|hip_flex",
    "hip_roll": r"hip.*roll",
    "hip_yaw": r"hip.*yaw",
    "ankle_pitch": r"ankle.*pitch|ankle_y",
    "ankle_roll": r"ankle.*roll|ankle_x",
    "shoulder_pitch": r"shoulder.*pitch",
    "elbow": r"elbow",
}


def _finite(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and 0 < x < 1e5 else None


def load_reference(require_span: bool = True) -> list[RefRobot]:
    """Humanoids from the survey (one row per robot; MJCF duplicates dropped)."""
    data = repo_root() / "datasets" / "robot_descriptions" / "data"
    segs = json.loads((data / "segments.json").read_text())
    rows = json.loads((data / "segment_rows.json").read_text())
    span = {r["robot"]: r for r in rows["per_robot"]}
    out = []
    for s in segs:
        if "humanoid" not in s.get("tags", []):
            continue
        r = span.get(s["key"])
        if require_span and (not r or not r.get("z_span_m") or not r.get("leg_length_m")):
            continue
        joints = [j for seg in s["segments"] for j in seg["joints"]]
        limits = {}
        for cls, pat in _PATTERNS.items():
            eff = [_finite(j.get("effort")) for j in joints if re.search(pat, j["name"], re.I)]
            vel = [_finite(j.get("velocity")) for j in joints if re.search(pat, j["name"], re.I)]
            eff = [e for e in eff if e]
            vel = [v for v in vel if v]
            if eff:
                limits[cls] = (max(eff), max(vel) if vel else None)
        out.append(RefRobot(
            key=s["key"], name=s.get("robot", s["key"]), maker=s.get("maker", ""),
            mass_kg=float(s["total_mass_kg"]),
            z_span_m=float(r["z_span_m"]) if r else 0.0,
            leg_m=float(r["leg_length_m"]) if r else 0.0,
            dof=int(s.get("n_dof", 0)), limits=limits))
    return out


def concept_overrides(ref: RefRobot, scale_tasks: bool = True) -> Optional[dict]:
    """A concept with this robot's height and leg ratio (None if out of range).

    With ``scale_tasks`` the duty cycle is made dynamically similar across sizes
    (speeds at fixed Froude numbers, payload a fixed share of the population
    mass, risers no higher than a fifth of the leg), so a 1 m robot is not asked
    to do what a 1.8 m robot does.
    """
    H = ref.height_est_m
    if not 0.9 <= H <= 2.3:
        return None
    lr = ref.leg_m / H
    if not 0.3 <= lr <= 0.6:
        return None
    ov: dict = {"name": f"ref-{ref.key}", "body": {"height_m": round(H, 3),
                                                     "leg_ratio": round(lr, 3)}}
    if scale_tasks:
        from ..trends import link_fits
        g, leg = 9.81, ref.leg_m
        mv = link_fits("humanoid").mass_vs_height
        m_trend = mv["coef"] * ref.z_span_m ** mv["exp"]
        ov["tasks"] = {
            "walk": {"speed": round(math.sqrt(0.15 * g * leg), 2),
                     "max_speed": round(math.sqrt(0.30 * g * leg), 2)},
            "run": {"speed": round(math.sqrt(0.60 * g * leg), 2)},
            "stairs": {"step_height_m": round(min(0.15, 0.2 * leg), 3)},
        }
        ov["payload"] = {"hands_total_kg": round(0.08 * m_trend, 1)}
    return ov


def benchmark(base: dict, evaluate_fn, refs: Optional[list] = None,
              scale_tasks: bool = True) -> list[dict]:
    """Close the mass loop for each reference robot's proportions; compare."""
    from .spec import _merge
    rows = []
    for ref in refs or load_reference():
        ov = concept_overrides(ref, scale_tasks)
        if ov is None:
            continue
        spec = _merge(base, ov)
        res = evaluate_fn(spec)
        row = {"robot": ref.name, "maker": ref.maker, "height_est_m": round(ref.height_est_m, 2),
               "walk_m_s": spec["tasks"]["walk"]["speed"], "run_m_s": spec["tasks"]["run"]["speed"],
               "payload_kg": spec["payload"]["hands_total_kg"],
               "leg_ratio": ov["body"]["leg_ratio"], "dof_real": ref.dof,
               "mass_real_kg": round(ref.mass_kg, 1),
               "mass_pred_kg": round(res.mass, 1) if res.closed else math.inf,
               "closed": res.closed}
        row["mass_ratio"] = round(row["mass_pred_kg"] / ref.mass_kg, 2) if res.closed else math.inf
        peaks = res.joint_peaks()
        for cls in ("knee", "hip_pitch", "ankle_pitch", "hip_roll", "shoulder_pitch"):
            real = ref.limits.get(cls, (None, None))[0]
            pred = peaks.get(cls, {}).get("tau")
            row[f"{cls}_real_Nm"] = real
            row[f"{cls}_pred_Nm"] = round(pred, 1) if pred is not None else None
            row[f"{cls}_ratio"] = round(pred / real, 2) if pred and real else None
        rows.append(row)
    return rows
