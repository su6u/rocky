"""Exact rigid-body mass properties of triangle meshes (numpy).

Solid:  divergence theorem over a closed, consistently oriented surface
        (D. Eberly, "Polyhedral Mass Properties (Revisited)", Geometric Tools, 2002/2009).
Shell:  area integrals of a thin surface of uniform areal density (thickness t ≪ feature size),
        exact per triangle via the second-moment formula for a triangle.
Printed part = walls (shell, thickness t) + sparse infill (solid scaled by infill fraction over the
        remaining interior volume). This is the standard first-order FDM mass model; it ignores
        top/bottom skin differences and seam overlaps (a few percent).
"""
from __future__ import annotations

import numpy as np


def solid_properties(v: np.ndarray, f: np.ndarray, density: float = 1.0):
    """Return mass, centroid (3,), inertia about centroid (3,3) for a closed mesh of uniform density."""
    x0, x1, x2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    d = np.cross(x1 - x0, x2 - x0)
    def subexpr(w0, w1, w2):
        t0 = w0 + w1; f1 = t0 + w2
        t1 = w0 * w0; t2 = t1 + w1 * t0
        f2 = t2 + w2 * f1
        f3 = w0 * t1 + w1 * t2 + w2 * f2
        g0 = f2 + w0 * (f1 + w0); g1 = f2 + w1 * (f1 + w1); g2 = f2 + w2 * (f1 + w2)
        return f1, f2, f3, g0, g1, g2
    fx = subexpr(x0[:, 0], x1[:, 0], x2[:, 0])
    fy = subexpr(x0[:, 1], x1[:, 1], x2[:, 1])
    fz = subexpr(x0[:, 2], x1[:, 2], x2[:, 2])
    intg = np.zeros(10)
    intg[0] = (d[:, 0] * fx[0]).sum()
    intg[1] = (d[:, 0] * fx[1]).sum(); intg[2] = (d[:, 1] * fy[1]).sum(); intg[3] = (d[:, 2] * fz[1]).sum()
    intg[4] = (d[:, 0] * fx[2]).sum(); intg[5] = (d[:, 1] * fy[2]).sum(); intg[6] = (d[:, 2] * fz[2]).sum()
    intg[7] = (d[:, 0] * (x0[:, 1] * fx[3] + x1[:, 1] * fx[4] + x2[:, 1] * fx[5])).sum()
    intg[8] = (d[:, 1] * (x0[:, 2] * fy[3] + x1[:, 2] * fy[4] + x2[:, 2] * fy[5])).sum()
    intg[9] = (d[:, 2] * (x0[:, 0] * fz[3] + x1[:, 0] * fz[4] + x2[:, 0] * fz[5])).sum()
    intg *= np.array([1 / 6, 1 / 24, 1 / 24, 1 / 24, 1 / 60, 1 / 60, 1 / 60, 1 / 120, 1 / 120, 1 / 120])
    vol = intg[0]
    c = intg[1:4] / vol
    Ixx = intg[5] + intg[6] - vol * (c[1] ** 2 + c[2] ** 2)
    Iyy = intg[4] + intg[6] - vol * (c[2] ** 2 + c[0] ** 2)
    Izz = intg[4] + intg[5] - vol * (c[0] ** 2 + c[1] ** 2)
    Ixy = -(intg[7] - vol * c[0] * c[1])
    Iyz = -(intg[8] - vol * c[1] * c[2])
    Ixz = -(intg[9] - vol * c[2] * c[0])
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    return density * vol, c, density * I, vol


def shell_properties(v: np.ndarray, f: np.ndarray, areal_density: float = 1.0):
    """Mass, centroid, inertia about centroid of a thin shell (uniform mass per area)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    A = area.sum()
    cent = (area[:, None] * (a + b + c) / 3).sum(0) / A
    # second moment of each triangle about the origin: ∫ x xᵀ dA = A/12 (Σ vi viᵀ + (Σ vi)(Σ vi)ᵀ)
    s = a + b + c
    S = (np.einsum('ni,nj->nij', a, a) + np.einsum('ni,nj->nij', b, b) + np.einsum('ni,nj->nij', c, c)
         + np.einsum('ni,nj->nij', s, s)) * (area / 12)[:, None, None]
    C = S.sum(0) - A * np.outer(cent, cent)           # central second moment
    I = np.trace(C) * np.eye(3) - C
    return areal_density * A, cent, areal_density * I, A


def combine(parts):
    """Combine [(m, c, I_about_c), ...] into one rigid body."""
    m = sum(p[0] for p in parts)
    c = sum(p[0] * np.asarray(p[1]) for p in parts) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in parts:
        d = np.asarray(ci) - c
        I += Ii + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return m, c, I


def printed_part(v, f, scale_m_per_unit, rho=1240.0, wall=0.0024, infill=0.15):
    """FDM-printed part at `scale` (mesh units → metres): walls of thickness `wall` + infill fraction."""
    vs = v * scale_m_per_unit
    ms, cs, Is, A = shell_properties(vs, f, rho * wall)
    msol, csol, Isol, V = solid_properties(vs, f, 1.0)
    interior = max(V - A * wall, 0.0)
    k = interior / V if V > 0 else 0.0
    mi = rho * infill * interior
    # approximate the interior as the solid scaled in density (inertia scales with mass, same centroid)
    parts = [(ms, cs, Is), (mi, csol, Isol * (mi / msol if msol > 0 else 0.0) if msol > 0 else np.zeros((3, 3)))]
    m, c, I = combine(parts)
    return dict(mass=m, com=c, inertia=I, area=A, volume=V, shell_mass=ms, infill_mass=mi)


if __name__ == "__main__":
    # self-test against closed forms
    # unit cube [0,1]^3 as 12 triangles
    V = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], float)
    F = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    m, c, I, vol = solid_properties(V, F, 1.0)
    assert abs(vol - 1) < 1e-12 and np.allclose(c, 0.5) and np.allclose(I, np.eye(3) / 6), (vol, c, I)
    ms, cs, Is, A = shell_properties(V, F, 1.0)
    # thin cubic shell of side 1, area 6: I = (5/18) M a² … per axis: 4 faces at distance, 2 faces centred → (2·(1/12+1/12)… ) = 5/3 total/6
    assert abs(A - 6) < 1e-12 and np.allclose(cs, 0.5) and np.allclose(Is, np.eye(3) * (5 / 3) / 1), (A, Is)
    # icosphere-like sphere check
    phi = (1 + 5 ** 0.5) / 2
    ico = np.array([[-1,phi,0],[1,phi,0],[-1,-phi,0],[1,-phi,0],[0,-1,phi],[0,1,phi],[0,-1,-phi],[0,1,-phi],[phi,0,-1],[phi,0,1],[-phi,0,-1],[-phi,0,1]], float)
    icf = np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]])
    verts = ico / np.linalg.norm(ico, axis=1)[:, None]; faces = icf
    for _ in range(5):
        edge = {}; newf = []; vl = list(verts)
        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in edge:
                p = (vl[i] + vl[j]) / 2; vl.append(p / np.linalg.norm(p)); edge[key] = len(vl) - 1
            return edge[key]
        for a, b, c2 in faces:
            ab, bc, ca = mid(a, b), mid(b, c2), mid(c2, a)
            newf += [[a, ab, ca], [b, bc, ab], [c2, ca, bc], [ab, bc, ca]]
        verts = np.array(vl); faces = np.array(newf)
    m, c, I, vol = solid_properties(verts, faces, 1.0)
    print('sphere volume %.5f (4π/3=%.5f)  Ixx/(m r²) %.5f (0.4)' % (vol, 4 * np.pi / 3, I[0, 0] / m))
    ms, cs, Is, A = shell_properties(verts, faces, 1.0)
    print('spherical shell area %.5f (4π=%.5f)  Ixx/(m r²) %.5f (2/3=%.5f)' % (A, 4 * np.pi, Is[0, 0] / ms, 2 / 3))
    print('massprops self-test OK')
