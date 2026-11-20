"""Planar rigid-body inverse dynamics (recursive Newton-Euler on a tree).

The screening models are planar: a sagittal (x forward, z up) or frontal
(y lateral, z up) projection of the robot, rooted at the stance foot, which is
welded to the ground. Every body is described by the *absolute* angle of its
frame, so the caller never has to track relative-angle sign conventions: a
joint's angle is the difference between its body's angle and its parent's.

Conventions (2-D, coordinates ``(h, z)`` with ``h`` the horizontal axis):

* angles are counter-clockwise when ``h`` points right and ``z`` up;
* ``perp(r) = (-r_z, r_h)`` so that ``v = ω·perp(r)``;
* ``cross(a, b) = a_h·b_z - a_z·b_h`` is the counter-clockwise moment.

Gravity enters as an upward acceleration of the ground (the usual RNEA trick),
so the returned torques are what the actuators must supply, including gravity.
Rotor inertia reflected through the gearbox (``armature = N²·J``) adds
``armature·q̈`` to a joint's torque.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

G = 9.81


def lump(points: Sequence[tuple[float, tuple[float, float], float]]
         ) -> tuple[float, np.ndarray, float]:
    """Combine point/rigid masses ``(m, (h, z), I_own)`` into one rigid body.

    Returns total mass, centre of mass and the inertia about that centre.
    """
    m = float(sum(p[0] for p in points))
    if m <= 0:
        return 0.0, np.zeros(2), 0.0
    com = sum(p[0] * np.asarray(p[1], float) for p in points) / m
    inertia = 0.0
    for mi, ri, Ii in points:
        d = np.asarray(ri, float) - com
        inertia += Ii + mi * float(d @ d)
    return m, com, inertia


def rod_inertia(m: float, length: float, radius: float = 0.0) -> float:
    """Inertia of a slender cylinder about its centre, perpendicular to its axis."""
    return m * (length ** 2 / 12.0 + radius ** 2 / 4.0)


@dataclass
class PlanarBody:
    """One rigid body of a planar tree.

    ``joint_pos`` is the joint origin in the parent's frame (in the world for a
    child of the ground). ``joint`` names the actuated joint between this body
    and its parent; ``None`` welds the body to its parent (the caller then
    supplies an absolute angle that moves with the parent).
    """
    name: str
    parent: int
    joint_pos: tuple[float, float]
    joint: Optional[str]
    mass: float
    com: tuple[float, float]
    inertia: float
    armature: float = 0.0
    tags: dict = field(default_factory=dict)


def _rot(phi: np.ndarray, v: Sequence[float]) -> np.ndarray:
    """Rotate a body-frame vector by angle(s) ``phi`` -> shape (T, 2)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]], axis=-1)


def _perp(r: np.ndarray) -> np.ndarray:
    return np.stack([-r[..., 1], r[..., 0]], axis=-1)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@dataclass
class IDResult:
    """Joint torques and kinematics of one inverse-dynamics pass (arrays over time)."""
    tau: dict                 # joint name -> (T,)
    qd: dict                  # joint name -> (T,)  relative joint speed
    qdd: dict                 # joint name -> (T,)
    origin: dict              # body name -> (T, 2) frame origin in the world
    com_world: np.ndarray     # (T, 2) whole-body centre of mass
    ground_force: np.ndarray  # (T, 2) force the ground exerts on the tree
    ground_moment: np.ndarray  # (T,) moment the ground exerts, about the world origin


class PlanarTree:
    """A tree of :class:`PlanarBody`, listed parents-first."""

    def __init__(self, bodies: Sequence[PlanarBody]):
        self.bodies = list(bodies)
        for i, b in enumerate(self.bodies):
            if b.parent >= i:
                raise ValueError(f"{b.name}: parent {b.parent} must precede it")
        self.index = {b.name: i for i, b in enumerate(self.bodies)}
        self.children: list[list[int]] = [[] for _ in self.bodies]
        self.roots: list[int] = []
        for i, b in enumerate(self.bodies):
            (self.children[b.parent] if b.parent >= 0 else self.roots).append(i)

    @property
    def total_mass(self) -> float:
        return sum(b.mass for b in self.bodies)

    def inverse_dynamics(self, phi: dict, dphi: dict, ddphi: dict,
                         gravity: float = G) -> IDResult:
        """Joint torques for prescribed absolute angles and their derivatives.

        ``phi``, ``dphi``, ``ddphi`` map body name -> array (T,).
        """
        nb = len(self.bodies)
        T = len(next(iter(phi.values())))
        zero = np.zeros(T)
        P = [None] * nb        # origin positions (T,2)
        A = [None] * nb        # origin accelerations (incl. gravity)
        Ac = [None] * nb       # COM accelerations
        Cw = [None] * nb       # COM offset in world orientation
        W = [None] * nb
        Al = [None] * nb
        base_acc = np.tile([0.0, gravity], (T, 1))
        for i, b in enumerate(self.bodies):
            ph, w, al = phi[b.name], dphi[b.name], ddphi[b.name]
            if b.parent < 0:
                r = np.tile(np.asarray(b.joint_pos, float), (T, 1))
                P[i] = r
                A[i] = base_acc.copy()
            else:
                p = b.parent
                r = _rot(phi[self.bodies[p].name], b.joint_pos)
                P[i] = P[p] + r
                A[i] = (A[p] + Al[p][:, None] * _perp(r)
                        - (W[p] ** 2)[:, None] * r)
            W[i], Al[i] = w, al
            c = _rot(ph, b.com)
            Cw[i] = c
            Ac[i] = A[i] + al[:, None] * _perp(c) - (w ** 2)[:, None] * c

        F = [None] * nb    # force on body i from its parent, at the joint
        N = [None] * nb    # moment on body i from its parent, about the joint
        tau, qd, qdd = {}, {}, {}
        for i in reversed(range(nb)):
            b = self.bodies[i]
            f = b.mass * Ac[i]
            n = b.inertia * Al[i] + _cross(Cw[i], f)
            for j in self.children[i]:
                rj = P[j] - P[i]
                f = f + F[j]
                n = n + N[j] + _cross(rj, F[j])
            F[i], N[i] = f, n
            if b.joint:
                if b.parent >= 0:
                    pn = self.bodies[b.parent].name
                    rel_d = dphi[b.name] - dphi[pn]
                    rel_dd = ddphi[b.name] - ddphi[pn]
                else:
                    rel_d, rel_dd = dphi[b.name], ddphi[b.name]
                t = n + b.armature * rel_dd
                # Two bodies may share an actuated joint name (left/right lumped);
                # keep the larger magnitude sample-wise is wrong, so refuse.
                if b.joint in tau:
                    raise ValueError(f"joint {b.joint!r} appears twice in one tree")
                tau[b.joint], qd[b.joint], qdd[b.joint] = t, rel_d, rel_dd

        gf = sum((F[i] for i in self.roots), np.zeros((T, 2)))
        gm = sum((N[i] + _cross(P[i], F[i]) for i in self.roots), zero.copy())
        # Whole-body COM
        mtot = self.total_mass
        com = sum(self.bodies[i].mass * (P[i] + Cw[i]) for i in range(nb)) / max(mtot, 1e-12)
        origin = {b.name: P[i] for i, b in enumerate(self.bodies)}
        # F at the roots includes gravity (base acceleration); it is the ground force.
        return IDResult(tau=tau, qd=qd, qdd=qdd, origin=origin, com_world=com,
                        ground_force=gf, ground_moment=gm)


def static_com(tree: PlanarTree, angles: dict) -> np.ndarray:
    """Whole-body centre of mass for one pose (absolute angles, scalars)."""
    phi = {k: np.array([float(v)]) for k, v in angles.items()}
    zero = {k: np.zeros(1) for k in angles}
    return tree.inverse_dynamics(phi, zero, zero).com_world[0]


def derivatives(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second time derivatives of a sampled signal (2nd-order accurate)."""
    d = np.gradient(x, t, edge_order=2)
    dd = np.gradient(d, t, edge_order=2)
    return d, dd


def angle_series(t: np.ndarray, angles: dict) -> tuple[dict, dict, dict]:
    """Unwrap absolute angles and differentiate them -> (phi, dphi, ddphi)."""
    phi, dphi, ddphi = {}, {}, {}
    for k, v in angles.items():
        u = np.unwrap(np.asarray(v, float))
        d, dd = derivatives(t, u)
        phi[k], dphi[k], ddphi[k] = u, d, dd
    return phi, dphi, ddphi
