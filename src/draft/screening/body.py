"""Geometry and mass model of a humanoid concept, and its planar projections.

Lengths come from standing height and a handful of ratios. Masses come from
three sources, kept apart so the report can show where each kilogram is:

* **actuators** — one mass and one reflected inertia per actuator slot, supplied
  by the caller (trend-sized or a catalogue part);
* **structure** — limb members priced by Draft's fitted structural trends
  (``LinkTrends``, mass vs member length), scaled by ``mass.structural_scale``;
* **trunk** — a shell that is a fixed share of total mass (frame, covers,
  wiring, boards), plus the explicit battery, compute, sensors and head.

Because the trunk shell is a share of the total, the total is solved, not
summed: ``M = Σ(other) / (1 - shell - head)``.

The planar trees are rooted at the stance ankle (the foot is welded to the
ground). Body frames point along each segment from the joint nearest the root
to the next one, so a child joint always sits at ``(length, 0)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..trends import link_fits
from .planar import PlanarBody, PlanarTree, lump, rod_inertia
from .spec import absolute_arm, active_joints

_LINKS = None


def _link_trends():
    global _LINKS
    if _LINKS is None:
        _LINKS = link_fits("humanoid")
    return _LINKS


def structural_mass(segment: str, length_m: float) -> float:
    """Structural (non-actuator) mass of one member from Draft's humanoid trend.

    Uses ``LinkTrends.structural_mass``: the fitted ``m = c·L^e``, with mass per
    metre held at the edge of the fitted length range and clamped to p10-p90.
    """
    out = _link_trends().structural_mass(segment, length_m)
    if out is None:
        raise KeyError(f"no structural trend for {segment!r}")
    return float(out["mass_kg"])


@dataclass
class Geometry:
    """Every length the models use (metres)."""
    height: float
    ankle_h: float
    thigh: float
    shank: float
    hip_to_shoulder: float
    waist: float              # hip -> waist joint
    head_neck: float
    upper_arm: float
    forearm: float
    hand: float
    foot_length: float
    foot_width: float
    heel: float
    hip_width: float
    shoulder_width: float
    arm_note: str = ""        # set when the arm length came from reach constraints

    @property
    def arm(self) -> float:
        """Shoulder joint -> grasp point."""
        return self.upper_arm + self.forearm + self.hand

    @property
    def leg(self) -> float:
        return self.thigh + self.shank

    @property
    def toe(self) -> float:
        return self.foot_length - self.heel

    @property
    def z_span(self) -> float:
        """Ankle to shoulder: Draft's population size metric (joint-frame span)."""
        return self.leg + self.hip_to_shoulder

    @property
    def hip_height_straight(self) -> float:
        return self.ankle_h + self.leg


def geometry(spec: dict) -> Geometry:
    b = spec["body"]
    H = float(b["height_m"])
    leg = b["leg_ratio"] * H
    ankle_h = b["ankle_height_ratio"] * H
    head_neck = b["head_neck_ratio"] * H
    h2s = H - ankle_h - leg - head_neck
    foot = b["foot_length_ratio"] * H
    geo = Geometry(
        height=H, ankle_h=ankle_h,
        thigh=leg * b["thigh_fraction"], shank=leg * (1 - b["thigh_fraction"]),
        hip_to_shoulder=h2s, waist=h2s * b["waist_fraction"], head_neck=head_neck,
        upper_arm=b["upper_arm_ratio"] * H, forearm=b["forearm_ratio"] * H,
        hand=b["hand_ratio"] * H,
        foot_length=foot, foot_width=b["foot_width_ratio"] * H,
        heel=foot * b["heel_fraction"],
        hip_width=b["hip_width_ratio"] * H,
        shoulder_width=b["shoulder_width_ratio"] * H,
    )
    if spec["reach"]["arm_from_reach"]:
        # function-level import: reach.py builds on Geometry
        from .reach import any_constraint, arm_from_reach
        if absolute_arm(spec):
            geo.arm_note = "已给定绝对臂长，未按够取约束推算"
        elif any_constraint(spec):
            geo.upper_arm, geo.forearm, geo.hand, geo.arm_note = arm_from_reach(geo, spec)
    return geo


#: Actuator slots: where each motor physically sits. ``parallel`` ankles replace
#: ankle_pitch/ankle_roll by two identical ``ankle_drive`` motors per leg.
def actuator_slots(spec: dict) -> list[str]:
    slots = []
    for j in active_joints(spec):
        if spec["mechanism"]["ankle"] == "parallel" and j in ("ankle_pitch", "ankle_roll"):
            if "ankle_drive" not in slots:
                slots.append("ankle_drive")
            continue
        slots.append(j)
    return slots


@dataclass
class MassModel:
    """Every mass item of the concept (kg), and the solved total."""
    total: float
    actuators: dict            # slot -> mass of ONE actuator
    actuator_count: dict       # slot -> how many on the robot
    structure: dict            # member -> mass of ONE member
    trunk: dict                # item -> mass
    payload_hands: float
    armature: dict             # slot -> reflected inertia (kg m^2)

    def breakdown(self) -> dict:
        act = sum(self.actuators[s] * self.actuator_count[s] for s in self.actuators)
        struct = (2 * (self.structure["thigh"] + self.structure["shank"]
                       + self.structure["foot"] + 2 * self.structure["hip_link"])
                  + 2 * (self.structure["upper_arm"] + self.structure["forearm"]
                         + 2 * self.structure["shoulder_link"]))
        return {
            "actuators": act,
            "limb_structure": struct,
            "end_effectors": 2 * self.trunk["end_effector_each"],
            "trunk_shell": self.trunk["shell"],
            "head": self.trunk["head"],
            "battery": self.trunk["battery"],
            "compute": self.trunk["compute"],
            "sensors": self.trunk["sensors"],
        }


def mass_model(spec: dict, geo: Geometry, act_mass: dict, act_armature: dict,
               battery_kg: float) -> MassModel:
    """Solve the total mass for given actuator masses and battery."""
    m = spec["mass"]
    s = m["structural_scale"]
    struct = {
        "thigh": s * structural_mass("thigh", geo.thigh),
        "shank": s * structural_mass("shank", geo.shank),
        "foot": s * structural_mass("foot", geo.foot_length),
        "hip_link": s * structural_mass("hip_link", 0.35 * geo.hip_width),
        "upper_arm": s * structural_mass("upper_arm", geo.upper_arm),
        "forearm": s * structural_mass("forearm", geo.forearm),
        "shoulder_link": s * structural_mass("shoulder_link", 0.25 * geo.upper_arm),
    }
    slots = actuator_slots(spec)
    count = {}
    for sl in slots:
        if sl == "ankle_drive":
            count[sl] = 4
        elif sl.startswith("waist"):
            count[sl] = 1
        else:
            count[sl] = 2
    acts = {sl: float(act_mass[sl]) for sl in slots}
    act_total = sum(acts[sl] * count[sl] for sl in slots)
    struct_total = (2 * (struct["thigh"] + struct["shank"] + struct["foot"]
                         + 2 * struct["hip_link"])
                    + 2 * (struct["upper_arm"] + struct["forearm"]
                           + 2 * struct["shoulder_link"]))
    fixed = (act_total + struct_total + 2 * m["end_effector_kg"]
             + m["compute_kg"] + m["sensors_kg"] + battery_kg)
    share = m["trunk_shell_fraction"] + m["head_fraction"]
    if share >= 0.9:
        raise ValueError("trunk_shell_fraction + head_fraction must stay below 0.9")
    total = fixed / (1.0 - share)
    trunk = {
        "shell": m["trunk_shell_fraction"] * total,
        "head": m["head_fraction"] * total,
        "battery": battery_kg,
        "compute": m["compute_kg"],
        "sensors": m["sensors_kg"],
        "end_effector_each": m["end_effector_kg"],
    }
    return MassModel(total=total, actuators=acts, actuator_count=count,
                     structure=struct, trunk=trunk,
                     payload_hands=spec["payload"]["hands_total_kg"],
                     armature={sl: float(act_armature.get(sl, 0.0)) for sl in slots})


@dataclass
class Concept:
    """Geometry + masses, with builders for the planar models."""
    spec: dict
    geo: Geometry
    mass: MassModel
    notes: list = field(default_factory=list)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _act(self, slot: str) -> float:
        return self.mass.actuators.get(slot, 0.0)

    def _arm(self, slot: str) -> float:
        return self.mass.armature.get(slot, 0.0)

    def ankle_motors_mass(self) -> float:
        """Mass of the ankle actuators of ONE leg."""
        if "ankle_drive" in self.mass.actuators:
            return 2 * self._act("ankle_drive")
        return self._act("ankle_pitch") + self._act("ankle_roll")

    def _ankle_armature(self, axis: str) -> float:
        if "ankle_drive" in self.mass.armature:
            # two motors act on each axis through the linkage
            return 2 * self._arm("ankle_drive")
        return self._arm(f"ankle_{axis}")

    def _leg_parts(self):
        """Mass items of one leg in physical coordinates (distance below the hip)."""
        g, mm = self.geo, self.mass
        mech = self.spec["mechanism"]
        knee_at_hip = mech["knee_motor_location"] == "hip"
        ankle_at_knee = mech["ankle_motor_location"] == "knee"
        thigh = [
            (mm.structure["thigh"], 0.40 * g.thigh),
            (2 * mm.structure["hip_link"], 0.03 * g.thigh),
            (self._act("hip_roll"), 0.0),
            (self._act("hip_yaw"), 0.12 * g.thigh),
            (self._act("knee"), (0.10 if knee_at_hip else 1.0) * g.thigh),
        ]
        shank = [
            (mm.structure["shank"], 0.43 * g.shank),
            (self.ankle_motors_mass(), (0.18 if ankle_at_knee else 1.0) * g.shank),
        ]
        return thigh, shank

    def _upper_parts(self, scale: float):
        """Pelvis and torso items: (mass, distance above the hip along the trunk)."""
        g, mm = self.geo, self.mass
        t = mm.trunk
        L = g.hip_to_shoulder
        waist_mot = sum(self._act(f"waist_{a}") for a in ("yaw", "roll", "pitch"))
        pelvis = [
            (2 * self._act("hip_pitch"), 0.0),
            (0.4 * t["shell"], 0.45 * g.waist),
            (waist_mot, g.waist),
        ]
        torso = [
            (0.6 * t["shell"], 0.55 * L),
            (t["battery"], self.spec["mass"]["battery_height_fraction"] * L),
            (t["compute"], 0.70 * L),
            (t["head"] + t["sensors"], L + 0.55 * g.head_neck),
            (2 * self._act("shoulder_pitch"), L),
        ]
        pelvis = [(m * scale, z) for m, z in pelvis]
        torso = [(m * scale, z) for m, z in torso]
        return pelvis, torso

    def _arm_parts(self, scale: float, payload: float):
        """Both arms lumped (sagittal): upper arm and forearm+hand items."""
        g, mm = self.geo, self.mass
        n_wrist = self.spec["dof"]["wrist"]
        wrist_mot = sum(self._act(w) for w in ("wrist_yaw", "wrist_pitch", "wrist_roll")[:n_wrist])
        upper = [
            (mm.structure["upper_arm"], 0.45 * g.upper_arm),
            (2 * mm.structure["shoulder_link"], 0.05 * g.upper_arm),
            (self._act("shoulder_roll"), 0.0),
            (self._act("shoulder_yaw"), 0.20 * g.upper_arm),
            (self._act("elbow"), g.upper_arm),
        ]
        fore = [
            (mm.structure["forearm"], 0.43 * g.forearm),
            (wrist_mot, 0.85 * g.forearm),
            (mm.trunk["end_effector_each"], g.forearm + 0.5 * g.hand),
        ]
        upper = [(2 * m * scale, z) for m, z in upper]
        fore = [(2 * m * scale, z) for m, z in fore]
        if payload > 0:
            fore.append((payload * scale, g.forearm + g.hand))
        return upper, fore

    @staticmethod
    def _rigid(items, radius: float = 0.04):
        """Lump items lying on a segment's axis (local x) into (m, com, I)."""
        pts = [(m, (x, 0.0), 0.0) for m, x in items if m > 0]
        if not pts:
            return 0.0, (0.0, 0.0), 0.0
        m, com, inertia = lump(pts)
        # members are not points: add each structural item's own spread
        inertia += sum(rod_inertia(mi, 2 * radius) for mi, _ in items)
        return m, (float(com[0]), float(com[1])), inertia

    # ── sagittal model ──────────────────────────────────────────────────────
    def sagittal(self, *, double_support: bool, payload: float = 0.0) -> PlanarTree:
        """Sagittal model rooted at the stance ankle.

        Single support carries the whole upper body and the swing leg; double
        support is the symmetric half-robot (one leg, half of everything above).
        Arms are lumped (both arms move together); their torques are halved by
        the caller.
        """
        g = self.geo
        scale = 0.5 if double_support else 1.0
        thigh, shank = self._leg_parts()
        # Stance leg runs ankle -> knee -> hip: flip the physical coordinates.
        st_shank = [(m, g.shank - z) for m, z in shank]
        st_thigh = [(m, g.thigh - z) for m, z in thigh]
        pelvis, torso = self._upper_parts(scale)
        torso_local = [(m, z - g.waist) for m, z in torso]
        upper, fore = self._arm_parts(scale, payload)

        def body(name, parent, jpos, joint, items, armature=0.0, radius=0.04):
            m, com, inertia = self._rigid(items, radius)
            return PlanarBody(name, parent, jpos, joint, m, com, inertia, armature)

        waist_joint = "waist_pitch" if "pitch" in self.spec["dof"]["waist"] else None
        depth = 0.09 * g.height
        bodies = [
            body("st_shank", -1, (0.0, 0.0), "ankle_pitch", st_shank,
                 self._ankle_armature("pitch")),
            body("st_thigh", 0, (g.shank, 0.0), "knee", st_thigh, self._arm("knee")),
            body("pelvis", 1, (g.thigh, 0.0), "hip_pitch", pelvis,
                 self._arm("hip_pitch"), radius=depth),
            body("torso", 2, (g.waist, 0.0), waist_joint, torso_local,
                 self._arm("waist_pitch") if waist_joint else 0.0, radius=depth),
            body("upper_arm", 3, (g.hip_to_shoulder - g.waist, 0.0), "shoulder_pitch",
                 upper, 2 * self._arm("shoulder_pitch")),
            body("forearm", 4, (g.upper_arm, 0.0), "elbow", fore, 2 * self._arm("elbow")),
        ]
        if not double_support:
            foot_items = [(self.mass.structure["foot"], (0.5 * g.foot_length - g.heel, -0.5 * g.ankle_h), 0.0)]
            if self.spec["mechanism"]["ankle"] == "serial" and \
                    self.spec["mechanism"]["ankle_motor_location"] == "ankle":
                foot_items.append((self._act("ankle_roll"), (0.0, -0.3 * g.ankle_h), 0.0))
            fm, fc, fi = lump(foot_items)
            fi += rod_inertia(fm, g.foot_length)
            bodies += [
                body("sw_thigh", 2, (0.0, 0.0), "hip_pitch_swing", thigh, self._arm("hip_pitch")),
                body("sw_shank", 6, (g.thigh, 0.0), "knee_swing", shank, self._arm("knee")),
                PlanarBody("sw_foot", 7, (g.shank, 0.0), "ankle_pitch_swing", fm,
                           (float(fc[0]), float(fc[1])), fi, self._ankle_armature("pitch")),
            ]
        return PlanarTree(bodies)

    # ── frontal model ───────────────────────────────────────────────────────
    def frontal(self, hip_height: float, swing_leg_length: float) -> PlanarTree:
        """Frontal single-support model (y lateral, z up), rooted at the stance ankle.

        The stance leg is one rigid body ankle -> hip. The pelvis frame points
        from the stance hip to the swing hip; everything above the hips is lumped
        on it (or on a torso body when the waist has a roll joint).
        """
        g, mm = self.geo, self.mass
        thigh, shank = self._leg_parts()
        L = hip_height - g.ankle_h            # vertical ankle -> hip span
        leg_items = ([(m, L - z * L / g.leg) for m, z in thigh]
                     + [(m, L - (g.thigh + z) * L / g.leg) for m, z in shank])
        pelvis, torso = self._upper_parts(1.0)
        upper, fore = self._arm_parts(1.0, 0.0)
        arms_m = sum(m for m, _ in upper) + sum(m for m, _ in fore)
        arms_z = g.hip_to_shoulder - 0.45 * (g.upper_arm + g.forearm)
        w = g.hip_width
        has_roll = "roll" in self.spec["dof"]["waist"]
        # pelvis: a = along the hip axis (toward the swing hip), b = up
        pel_pts = [(m, (w / 2, z), 0.0) for m, z in pelvis]
        pel_pts[0] = (pelvis[0][0] / 2, (0.0, 0.0), 0.0)        # stance hip-pitch motor
        pel_pts.append((pelvis[0][0] / 2, (w, 0.0), 0.0))       # swing hip-pitch motor
        up_pts = [(m, (w / 2, z), 0.0) for m, z in torso] + [(arms_m, (w / 2, arms_z), 0.0)]
        bodies = []
        lm, lc, li = self._rigid(leg_items)
        bodies.append(PlanarBody("st_leg", -1, (0.0, 0.0), "ankle_roll", lm, lc, li,
                                 self._ankle_armature("roll")))
        sw_items = [(m, z * swing_leg_length / g.leg) for m, z in thigh + [(m, g.thigh + z) for m, z in shank]]
        sw_items.append((mm.structure["foot"], swing_leg_length))
        sm, sc, si = self._rigid(sw_items)
        if has_roll:
            pm, pc, pi = lump(pel_pts)
            pi += rod_inertia(pm, w)
            tm, tc, ti = lump([(m, (r[1] - g.waist, r[0] - w / 2), 0.0) for m, r, _ in up_pts])
            ti += rod_inertia(tm, g.hip_to_shoulder)
            bodies += [
                PlanarBody("pelvis", 0, (L, 0.0), "hip_roll", pm, tuple(pc), pi, self._arm("hip_roll")),
                PlanarBody("torso", 1, (w / 2, g.waist), "waist_roll", tm, tuple(tc), ti,
                           self._arm("waist_roll")),
                PlanarBody("sw_leg", 1, (w, 0.0), "hip_roll_swing", sm, sc, si, self._arm("hip_roll")),
            ]
        else:
            pm, pc, pi = lump(pel_pts + up_pts)
            pi += rod_inertia(pm, g.hip_to_shoulder)
            bodies += [
                PlanarBody("pelvis", 0, (L, 0.0), "hip_roll", pm, tuple(pc), pi, self._arm("hip_roll")),
                PlanarBody("sw_leg", 1, (w, 0.0), "hip_roll_swing", sm, sc, si, self._arm("hip_roll")),
            ]
        return PlanarTree(bodies)

    # ── lumped quantities used by quasi-static checks ───────────────────────
    def upper_body_yaw_inertia(self) -> float:
        """Yaw inertia of everything above the waist (arms out to the side)."""
        g = self.geo
        _, torso = self._upper_parts(1.0)
        upper, fore = self._arm_parts(1.0, self.mass.payload_hands)
        m_t = sum(m for m, _ in torso)
        m_a = sum(m for m, _ in upper) + sum(m for m, _ in fore)
        r_t = 0.3 * g.shoulder_width
        return m_t * r_t ** 2 + m_a * (0.5 * g.shoulder_width) ** 2

    def forearm_hand_mass(self) -> tuple[float, float]:
        """(mass, COM distance from the elbow) of ONE forearm + hand + its payload share."""
        _, fore = self._arm_parts(0.5, self.mass.payload_hands)   # lumped x2 -> one arm
        m = sum(x for x, _ in fore)
        c = sum(x * z for x, z in fore) / max(m, 1e-9)
        return m, c

    def arm_mass(self) -> tuple[float, float]:
        """(mass, COM distance from the shoulder) of ONE arm with its payload share."""
        upper, fore = self._arm_parts(0.5, self.mass.payload_hands)
        items = upper + [(m, self.geo.upper_arm + z) for m, z in fore]
        m = sum(x for x, _ in items)
        return m, sum(x * z for x, z in items) / max(m, 1e-9)


def build_concept(spec: dict, act_mass: dict, act_armature: dict,
                  battery_kg: float) -> Concept:
    geo = geometry(spec)
    return Concept(spec, geo, mass_model(spec, geo, act_mass, act_armature, battery_kg))


def stand_hip_height(geo: Geometry, knee_bend_deg: float) -> float:
    """Hip height with the knee bent by ``knee_bend_deg`` (foot flat)."""
    k = math.radians(knee_bend_deg)
    d = math.sqrt(geo.thigh ** 2 + geo.shank ** 2 + 2 * geo.thigh * geo.shank * math.cos(k))
    return geo.ankle_h + d
