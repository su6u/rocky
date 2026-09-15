"""Kinematically constrained rigid registration (numpy only).

The action-figure parts are registered to the one-piece statue so the sculpted pose
can be expressed as joint rotations of the real kit hardware:
  * torso: free 6-DOF rigid transform,
  * limb segment A: rotation about the torso ball-stud centre (3-DOF),
  * limb segment B: rotation about the elbow hinge axis (1-DOF, both insertions tested).
Point-to-plane Gauss-Newton with robust (Tukey) weights; outliers are expected because
kit hardware (studs, pins) and neighbouring statue geometry are not shared surfaces.
"""
from __future__ import annotations

import numpy as np

from meshio import RadiusGrid


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / th)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)


def axis_angle(axis, ang):
    a = np.asarray(axis, float)
    return exp_so3(a / np.linalg.norm(a) * ang)


def rot_between(a, b):
    """Minimal rotation taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if c < -0.999999:
        orth = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        return axis_angle(orth, np.pi)
    return np.eye(3) + skew(v) + skew(v) @ skew(v) / (1 + c)


def random_rotations(n, seed=0):
    """Uniform random rotations via unit quaternions."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def tukey(r, c):
    w = (1 - (r / c) ** 2) ** 2
    return np.where(np.abs(r) < c, w, 0.0)


def inlier_score(grid: RadiusGrid, pts, normals=None, tol=0.8):
    idx, d = grid.query(pts)
    ok = (idx >= 0) & (d < tol)
    if normals is not None:
        ok &= (normals * grid.n[np.maximum(idx, 0)]).sum(1) > 0.5
    return ok.mean()


def icp(grid: RadiusGrid, src, src_n, R, t, mode="free", pivot=None, axis=None, iters=40, cutoff=1.5,
        min_cutoff=0.25):
    """Point-to-plane ICP.

    mode 'free'  : x' = R x + t                       (6 dof: w, dt)
    mode 'pivot' : x' = R (x - c0) + pivot, R updated  (3 dof: w) — src must be given relative to c0
    mode 'hinge' : x' = Rot(axis, th) R0 x + t0, th    (1 dof) plus optional axial slide (1 dof)
    For 'pivot' and 'hinge' the caller supplies src already expressed so that R,t compose correctly.
    """
    R = R.copy(); t = t.copy()
    c = cutoff
    for it in range(iters):
        x = src @ R.T + t
        nx = src_n @ R.T
        idx, d = grid.query(x)
        ok = idx >= 0
        y = grid.p[np.maximum(idx, 0)]; m = grid.n[np.maximum(idx, 0)]
        ok &= (nx * m).sum(1) > 0.3
        r = ((x - y) * m).sum(1)
        w = tukey(r, c) * ok * tukey(d, 2.5 * c)
        if w.sum() < 10:
            break
        if mode == "free":
            J = np.c_[np.cross(x - t.mean() * 0, m) * 0 + np.cross(x, m), m]
            H = (J * w[:, None]).T @ J; g = (J * w[:, None]).T @ r
            step = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
            dR = exp_so3(step[:3])
            R = dR @ R; t = dR @ t + step[3:]
        elif mode == "pivot":
            rel = x - pivot
            J = np.cross(rel, m)
            H = (J * w[:, None]).T @ J; g = (J * w[:, None]).T @ r
            step = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
            dR = exp_so3(step)
            R = dR @ R; t = dR @ (t - pivot) + pivot
        elif mode == "hinge":
            rel = x - pivot
            J = np.c_[(np.cross(axis, rel) * m).sum(1), (m @ axis)]
            H = (J * w[:, None]).T @ J; g = (J * w[:, None]).T @ r
            step = -np.linalg.solve(H + 1e-9 * np.eye(2), g)
            dR = axis_angle(axis, step[0])
            R = dR @ R; t = dR @ (t - pivot) + pivot + axis * step[1]
        if it > 5:
            c = max(min_cutoff, c * 0.85)
    x = src @ R.T + t
    idx, d = grid.query(x)
    ok = idx >= 0
    rms = float(np.sqrt(np.mean(d[ok & (d < 0.5)] ** 2))) if np.any(ok & (d < 0.5)) else np.inf
    return R, t, dict(rms_inliers=rms, inlier_frac=float(np.mean(ok & (d < 0.5))), mean_abs=float(np.mean(d[ok])) if ok.any() else np.inf)
