"""RANSAC detection of the kit's joint hardware: spheres (ball studs / sockets)
and cylinders (hinge pins / holes), refined by nonlinear least squares.

Uses oriented points (position + unit normal). Sign of the normal relative to the
primitive's centre/axis separates convex hardware (studs, pins: outward normals)
from concave hardware (sockets, holes: inward normals).
"""
from __future__ import annotations

import numpy as np


def _closest_between_lines(p1, d1, p2, d2):
    """Midpoint of the shortest segment between lines p1+t d1 and p2+s d2 (batched)."""
    w = p1 - p2
    a = (d1 * d1).sum(-1); b = (d1 * d2).sum(-1); c = (d2 * d2).sum(-1)
    d = (d1 * w).sum(-1); e = (d2 * w).sum(-1)
    den = a * c - b * b
    ok = den > 1e-9
    den = np.where(ok, den, 1.0)
    t = (b * e - c * d) / den
    s = (a * e - b * d) / den
    return 0.5 * ((p1 + t[:, None] * d1) + (p2 + s[:, None] * d2)), ok


def _local_partners(p, idx, i1, local, rng):
    """For each i1 pick a random partner from the same voxel (edge `local`) among idx."""
    key = np.floor(p[idx] / local).astype(np.int64)
    lin = (key[:, 0] * 73856093) ^ (key[:, 1] * 19349663) ^ (key[:, 2] * 83492791)
    order = np.argsort(lin, kind="stable")
    lin_sorted = lin[order]
    uniq, start, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
    pos_of = np.empty(len(idx), np.int64)
    pos_of[order] = np.arange(len(idx))
    where = np.searchsorted(uniq, lin_sorted)
    grp = where[pos_of]  # group id per element of idx
    # map i1 (global ids) to positions in idx
    lookup = -np.ones(len(p), np.int64); lookup[idx] = np.arange(len(idx))
    g = grp[lookup[i1]]
    j = start[g] + (rng.random(len(i1)) * counts[g]).astype(np.int64)
    return idx[order[j]]


def fit_sphere_lsq(p):
    """Algebraic sphere fit |p|^2 = 2 c.p + (r^2 - |c|^2), then Gauss-Newton on geometric error."""
    A = np.c_[2 * p, np.ones(len(p))]
    sol, *_ = np.linalg.lstsq(A, (p * p).sum(1), rcond=None)
    c = sol[:3]; r = np.sqrt(max(sol[3] + c @ c, 1e-12))
    for _ in range(20):
        dv = p - c; dist = np.linalg.norm(dv, axis=1); u = dv / dist[:, None]
        res = dist - r
        J = np.c_[-u, -np.ones(len(p))]
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c = c + step[:3]; r = r + step[3]
        if np.linalg.norm(step) < 1e-10:
            break
    res = np.linalg.norm(p - c, axis=1) - r
    return c, float(r), float(np.sqrt((res ** 2).mean()))


def ransac_spheres(p, n, rmin, rmax, sign, tol=0.06, ang_deg=20.0, iters=6000, min_inliers=150,
                   max_models=8, seed=0, local=3.0):
    """Detect spheres whose normals point outward (sign=+1) or inward (sign=-1)."""
    rng = np.random.default_rng(seed)
    cos_tol = np.cos(np.radians(ang_deg))
    remaining = np.ones(len(p), bool)
    models = []
    for _ in range(max_models):
        idx = np.flatnonzero(remaining)
        if len(idx) < min_inliers:
            break
        i1 = idx[rng.integers(len(idx), size=iters)]
        # second point from the same small voxel (spatially coherent sampling)
        i2 = _local_partners(p, idx, i1, local, rng)
        c, ok = _closest_between_lines(p[i1], n[i1], p[i2], n[i2])
        r1 = np.linalg.norm(p[i1] - c, axis=1); r2 = np.linalg.norm(p[i2] - c, axis=1)
        good = ok & (np.abs(r1 - r2) < tol) & (r1 > rmin) & (r1 < rmax)
        good &= sign * ((p[i1] - c) * n[i1]).sum(1) / np.maximum(r1, 1e-9) > cos_tol
        good &= sign * ((p[i2] - c) * n[i2]).sum(1) / np.maximum(r2, 1e-9) > cos_tol
        cand = np.flatnonzero(good)
        if len(cand) == 0:
            break
        best, best_count = None, 0
        sub = p[idx]; subn = n[idx]
        for k in cand[:400]:
            cc, rr = c[k], 0.5 * (r1[k] + r2[k])
            dv = sub - cc; dist = np.linalg.norm(dv, axis=1)
            inl = (np.abs(dist - rr) < tol) & (sign * (dv * subn).sum(1) / np.maximum(dist, 1e-9) > cos_tol)
            cnt = int(inl.sum())
            if cnt > best_count:
                best, best_count = (cc, rr), cnt
        if best is None or best_count < min_inliers:
            break
        cc, rr = best
        dv = p - cc; dist = np.linalg.norm(dv, axis=1)
        inl = remaining & (np.abs(dist - rr) < tol * 2) & (sign * (dv * n).sum(1) / np.maximum(dist, 1e-9) > cos_tol)
        c_ref, r_ref, rms = fit_sphere_lsq(p[inl])
        dv = p - c_ref; dist = np.linalg.norm(dv, axis=1)
        inl = remaining & (np.abs(dist - r_ref) < tol) & (sign * (dv * n).sum(1) / np.maximum(dist, 1e-9) > cos_tol)
        c_ref, r_ref, rms = fit_sphere_lsq(p[inl])
        # angular coverage of inliers: fraction of 26 directions hit
        u = (p[inl] - c_ref) / r_ref
        models.append(dict(center=c_ref, radius=r_ref, rms=rms, inliers=int(inl.sum()), dirs=u))
        remaining &= ~(np.linalg.norm(p - c_ref, axis=1) < r_ref + 3 * tol)
    return models


def fit_cylinder_lsq(p, axis, center):
    """Refine an infinite cylinder (axis direction a, point c, radius r) by Gauss-Newton."""
    a = axis / np.linalg.norm(axis)
    # parametrize orientation by small rotation, point by 2D offset in plane orthogonal to a
    c = center - a * ((center - p.mean(0)) @ a)
    def radial(c, a):
        w = p - c
        w_perp = w - np.outer(w @ a, a)
        return np.linalg.norm(w_perp, axis=1), w_perp
    dist, _ = radial(c, a); r = dist.mean()
    for _ in range(30):
        # orthonormal basis
        t1 = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); t1 /= np.linalg.norm(t1)
        t2 = np.cross(a, t1)
        def resid(x):
            aa = a + x[0] * t1 + x[1] * t2; aa /= np.linalg.norm(aa)
            cc = c + x[2] * t1 + x[3] * t2
            d, _ = radial(cc, aa)
            return d - (r + x[4])
        x0 = np.zeros(5); r0 = resid(x0)
        J = np.empty((len(p), 5)); eps = 1e-6
        for k in range(5):
            dx = np.zeros(5); dx[k] = eps
            J[:, k] = (resid(dx) - r0) / eps
        step, *_ = np.linalg.lstsq(J, -r0, rcond=None)
        a = a + step[0] * t1 + step[1] * t2; a /= np.linalg.norm(a)
        c = c + step[2] * t1 + step[3] * t2; r = r + step[4]
        c = c - a * ((c - p.mean(0)) @ a)
        if np.linalg.norm(step) < 1e-10:
            break
    d, _ = radial(c, a)
    res = d - r
    return a, c, float(r), float(np.sqrt((res ** 2).mean()))


def ransac_cylinders(p, n, rmin, rmax, sign, tol=0.05, ang_deg=15.0, iters=8000, min_inliers=120,
                     max_models=6, seed=0, local=4.0):
    rng = np.random.default_rng(seed)
    cos_tol = np.cos(np.radians(ang_deg))
    remaining = np.ones(len(p), bool)
    models = []
    for _ in range(max_models):
        idx = np.flatnonzero(remaining)
        if len(idx) < min_inliers:
            break
        i1 = idx[rng.integers(len(idx), size=iters)]
        i2 = _local_partners(p, idx, i1, local, rng)
        a = np.cross(n[i1], n[i2]); an = np.linalg.norm(a, axis=1)
        ok = an > 0.25
        a = a / np.maximum(an, 1e-9)[:, None]
        # project into plane orthogonal to axis and intersect normal lines
        def proj(x):
            return x - (x * a).sum(1)[:, None] * a
        q1, q2 = proj(p[i1]), proj(p[i2]); m1, m2 = proj(n[i1]), proj(n[i2])
        c, ok2 = _closest_between_lines(q1, m1, q2, m2)
        r1 = np.linalg.norm(q1 - c, axis=1); r2 = np.linalg.norm(q2 - c, axis=1)
        good = ok & ok2 & (np.abs(r1 - r2) < tol) & (r1 > rmin) & (r1 < rmax)
        good &= sign * ((q1 - c) * n[i1]).sum(1) / np.maximum(r1, 1e-9) > cos_tol
        good &= sign * ((q2 - c) * n[i2]).sum(1) / np.maximum(r2, 1e-9) > cos_tol
        cand = np.flatnonzero(good)
        if len(cand) == 0:
            break
        best, best_count = None, 0
        sub = p[idx]; subn = n[idx]
        for k in cand[:500]:
            aa, cc, rr = a[k], c[k], 0.5 * (r1[k] + r2[k])
            w = sub - cc; wp = w - np.outer(w @ aa, aa); dist = np.linalg.norm(wp, axis=1)
            inl = (np.abs(dist - rr) < tol) & (sign * (wp * subn).sum(1) / np.maximum(dist, 1e-9) > cos_tol)
            inl &= np.abs(subn @ aa) < 0.3
            # keep only the axial cluster around the sampled point (avoid far coaxial junk)
            t = w @ aa; t0 = (p[i1[k]] - cc) @ aa
            inl &= np.abs(t - t0) < 8.0
            cnt = int(inl.sum())
            if cnt > best_count:
                best, best_count = (aa, cc, rr, t0), cnt
        if best is None or best_count < min_inliers:
            break
        aa, cc, rr, t0 = best
        w = p - cc; wp = w - np.outer(w @ aa, aa); dist = np.linalg.norm(wp, axis=1)
        inl = remaining & (np.abs(dist - rr) < tol) & (sign * (wp * n).sum(1) / np.maximum(dist, 1e-9) > cos_tol)
        inl &= (np.abs(n @ aa) < 0.3) & (np.abs(w @ aa - t0) < 8.0)
        a_ref, c_ref, r_ref, rms = fit_cylinder_lsq(p[inl], aa, cc)
        t = (p[inl] - c_ref) @ a_ref
        models.append(dict(axis=a_ref, point=c_ref + a_ref * 0.5 * (t.min() + t.max()), radius=r_ref, rms=rms,
                           inliers=int(inl.sum()), length=float(t.max() - t.min())))
        w = p - c_ref; wp = w - np.outer(w @ a_ref, a_ref)
        remaining &= ~((np.linalg.norm(wp, axis=1) < r_ref + 0.6) & (np.abs(w @ a_ref - 0.5 * (t.min() + t.max())) < 0.5 * (t.max() - t.min()) + 0.6))
    return models
