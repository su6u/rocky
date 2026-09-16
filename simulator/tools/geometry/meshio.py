"""Minimal, dependency-light mesh utilities (numpy only).

Binary STL loading with exact vertex welding, area-weighted vertex normals,
area-uniform surface sampling, and a radius-limited nearest-neighbour grid.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

STL_DTYPE = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def load_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices float64 [V,3], faces int64 [F,3]) with identical corners welded."""
    raw = Path(path).read_bytes()
    count = struct.unpack("<I", raw[80:84])[0]
    if 84 + 50 * count != len(raw):
        raise ValueError(f"{path}: not a binary STL with {count} triangles")
    tri = np.frombuffer(raw, dtype=STL_DTYPE, offset=84, count=count)["v"].astype(np.float64)
    verts, inverse = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    # Drop degenerate faces (two identical corners after welding).
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return verts, faces[ok]


def face_normals(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit face normals and face areas."""
    c = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    a = np.linalg.norm(c, axis=1)
    return c / np.maximum(a, 1e-300)[:, None], 0.5 * a


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    c = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # area-weighted
    n = np.zeros_like(v)
    for j in range(3):
        np.add.at(n, f[:, j], c)
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]


def sample_surface(v: np.ndarray, f: np.ndarray, count: int, seed: int = 0):
    """Area-uniform random surface samples with their face normals."""
    rng = np.random.default_rng(seed)
    fn, area = face_normals(v, f)
    idx = rng.choice(len(f), size=count, p=area / area.sum())
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    a, b, c = v[f[idx, 0]], v[f[idx, 1]], v[f[idx, 2]]
    p = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    return p, fn[idx]


class RadiusGrid:
    """Exact nearest neighbour among points within `radius` (uniform hash grid).

    Points are voxel-thinned so every cell holds at most `per_cell` points; queries
    inspect the 27 surrounding cells. Queries with no point within the cell
    neighbourhood return index -1.
    """

    def __init__(self, points: np.ndarray, normals: np.ndarray, radius: float, per_cell: int = 8):
        self.h = float(radius)
        sub = self.h / round(per_cell ** (1 / 3))
        key = np.floor(points / sub).astype(np.int64)
        _, keep = np.unique(key, axis=0, return_index=True)
        self.p = points[keep]
        self.n = normals[keep]
        cell = np.floor(self.p / self.h).astype(np.int64)
        self.origin = cell.min(0) - 1
        cell -= self.origin
        self.shape = cell.max(0) + 2
        lin = (cell[:, 0] * self.shape[1] + cell[:, 1]) * self.shape[2] + cell[:, 2]
        order = np.argsort(lin, kind="stable")
        lin = lin[order]
        self.p, self.n = self.p[order], self.n[order]
        uniq, start, counts = np.unique(lin, return_index=True, return_counts=True)
        self.cap = int(counts.max())
        table = -np.ones((len(uniq), self.cap), dtype=np.int64)
        rank = np.arange(len(lin)) - np.repeat(start, counts)
        table[np.repeat(np.arange(len(uniq)), counts), rank] = np.arange(len(lin))
        self.keys, self.table = uniq, table

    def query(self, q: np.ndarray):
        cell = np.floor(q / self.h).astype(np.int64) - self.origin
        offs = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 3)
        nb = cell[:, None, :] + offs[None]
        inside = np.all((nb >= 0) & (nb < self.shape), axis=2)
        lin = (nb[..., 0] * self.shape[1] + nb[..., 1]) * self.shape[2] + nb[..., 2]
        pos = np.searchsorted(self.keys, lin)
        pos = np.clip(pos, 0, len(self.keys) - 1)
        hit = inside & (self.keys[pos] == lin)
        cand = np.where(hit[..., None], self.table[pos], -1).reshape(len(q), -1)
        valid = cand >= 0
        d2 = np.where(valid, ((self.p[np.maximum(cand, 0)] - q[:, None, :]) ** 2).sum(-1), np.inf)
        j = np.argmin(d2, axis=1)
        best = cand[np.arange(len(q)), j]
        dist = np.sqrt(d2[np.arange(len(q)), j])
        best[~np.isfinite(dist)] = -1
        return best, dist
