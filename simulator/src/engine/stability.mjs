// Stability measures for multi-legged support. All inputs in world coordinates.
//
//  SSM   static stability margin (McGhee & Frank 1968): signed distance from the COM's projection along
//        gravity to the nearest support-polygon edge. Valid only for quasi-static motion.
//  ZMP   for coplanar contacts the zero-moment point equals the centre of pressure of the measured
//        normal forces (Vukobratović & Borovac 2004). ZMP margin = distance of CoP to the polygon edge;
//        the polygon uses contacts that actually carry load, not the planned stance set.
//  NESM  normalised energy stability margin (Hirose et al. 1998): the smallest rise of the COM needed to
//        tip about any support edge. Derivation: rotating the COM about edge axis â (through p) moves it on
//        a circle of radius ρ = ‖r⊥‖ in the plane ⊥ â; the highest point of that circle lies ρ·cosψ above the
//        axis, ψ = inclination of â, so h = ρ (cosψ − ê₁·û) with ê₁ = r⊥/ρ and û = −ĝ.
//  FASM  force–angle stability measure (Papadopoulos & Rey 1996): for each tip axis, the angle between the
//        net destabilising force (gravity, inertia, disturbances) projected ⊥ axis and the axis-to-COM normal,
//        weighted by ‖d‖‖f‖. Negative ⇒ tip-over in progress; captures dynamic effects that SSM misses.
//  CP    instantaneous capture point of the linear inverted pendulum (Pratt et al. 2006):
//        ξ = c + ċ / ω₀, ω₀ = √(g / z_c). Outside the polygon ⇒ the robot cannot stop without stepping.

export function hull2d(pts) {
  const p = pts.map((q, i) => [q[0], q[1], i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of p) { while (lower.length > 1 && cross(lower.at(-2), lower.at(-1), q) <= 1e-15) lower.pop(); lower.push(q); }
  for (const q of [...p].reverse()) { while (upper.length > 1 && cross(upper.at(-2), upper.at(-1), q) <= 1e-15) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1)); // counter-clockwise, entries carry original index
}

// Signed distance from point to CCW polygon boundary (positive inside).
export function polygonMargin(poly, x, y) {
  if (poly.length < 3) return -Infinity;
  let m = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const ex = b[0] - a[0], ey = b[1] - a[1], L = Math.hypot(ex, ey) || 1;
    m = Math.min(m, (ex * (y - a[1]) - ey * (x - a[0])) / L);
  }
  return m;
}

// Chebyshev centre of a CCW convex polygon: point maximising the minimum edge distance.
// max r s.t. nₖ·x + r ≤ dₖ  — a 3-variable LP; the optimum lies at the intersection of three edge
// constraints (or two, for degenerate strips), so exhaustive enumeration of triples is exact for n ≤ 6.
export function chebyshevCenter(poly) {
  const n = poly.length; if (n < 3) return null;
  const E = poly.map((a, i) => { const b = poly[(i + 1) % n], ex = b[0] - a[0], ey = b[1] - a[1], L = Math.hypot(ex, ey); const nx = ey / L, ny = -ex / L; return [nx, ny, nx * a[0] + ny * a[1]]; });
  // inward margin of x: min_k (d_k − n_k·x) with outward normals n_k (CCW ⇒ (ey,−ex) is outward)
  const sols = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
    const A = [E[i], E[j], E[k]];
    const det = (a) => a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    const M = A.map((e) => [e[0], e[1], 1]), D = det(M);
    if (Math.abs(D) < 1e-14) continue;
    const rhs = A.map((e) => e[2]);
    const sub = (col) => M.map((row, r) => row.map((v, c) => (c === col ? rhs[r] : v)));
    const x = det(sub(0)) / D, y = det(sub(1)) / D, r = det(sub(2)) / D;
    if (r > 0 && E.every((e) => e[0] * x + e[1] * y + r <= e[2] + 1e-9)) sols.push([x, y, r]);
  }
  if (!sols.length) return null;
  // The optimal set is convex; when two edges are parallel it is a segment. Average its vertices so the
  // chosen centre is unique and central (enumeration returns the segment's end points).
  const rmax = Math.max(...sols.map((q) => q[2]));
  const top = sols.filter((q) => q[2] >= rmax - 1e-9);
  const best = [top.reduce((a, q) => a + q[0], 0) / top.length, top.reduce((a, q) => a + q[1], 0) / top.length, rmax];
  return best;
}

export function stabilityReport({ contacts, com, comVel, comAcc, mass, gravity = [0, 0, -9.81], minForce = 0.5, external = [0, 0, 0] }) {
  const loaded = contacts.filter((c) => c.fn > minForce);
  const pts = loaded.map((c) => c.point);
  const g = Math.hypot(...gravity), up = gravity.map((x) => -x / g);
  const out = { loadedFeet: loaded.length, polygon: [], ssm: -Infinity, zmp: null, zmpMargin: -Infinity, nesm: -Infinity, fasm: -Infinity, capture: null, captureMargin: -Infinity };
  if (pts.length < 3) return out;
  const hull = hull2d(pts);
  out.polygon = hull.map((h) => [h[0], h[1]]);
  // project along gravity onto the mean support height (flat or tilted ground)
  const zg = pts.reduce((s, p) => s + p[2], 0) / pts.length;
  const t = (com[2] - zg) / up[2];
  const cx = com[0] - up[0] * t, cy = com[1] - up[1] * t;
  out.comProjection = [cx, cy, zg];
  out.ssm = polygonMargin(out.polygon, cx, cy);
  const F = loaded.reduce((s, c) => s + c.fn, 0);
  const zx = loaded.reduce((s, c) => s + c.fn * c.point[0], 0) / F, zy = loaded.reduce((s, c) => s + c.fn * c.point[1], 0) / F;
  out.zmp = [zx, zy, zg]; out.zmpMargin = polygonMargin(out.polygon, zx, zy);
  // NESM
  let nesm = Infinity, fasm = Infinity;
  const fr = [mass * gravity[0] - mass * (comAcc?.[0] ?? 0) + external[0], mass * gravity[1] - mass * (comAcc?.[1] ?? 0) + external[1], mass * gravity[2] - mass * (comAcc?.[2] ?? 0) + external[2]];
  for (let i = 0; i < hull.length; i++) {
    const a = pts[hull[i][2]], b = pts[hull[(i + 1) % hull.length][2]];
    let ax = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]; const al = Math.hypot(...ax); ax = ax.map((x) => x / al);
    const r = [com[0] - a[0], com[1] - a[1], com[2] - a[2]], ra = r[0] * ax[0] + r[1] * ax[1] + r[2] * ax[2];
    const rp = [r[0] - ra * ax[0], r[1] - ra * ax[1], r[2] - ra * ax[2]], rho = Math.hypot(...rp);
    const e1u = (rp[0] * up[0] + rp[1] * up[1] + rp[2] * up[2]) / rho;
    const au = ax[0] * up[0] + ax[1] * up[1] + ax[2] * up[2];
    const h = rho * (Math.sqrt(Math.max(0, 1 - au * au)) - e1u);
    nesm = Math.min(nesm, h);
    // FASM: f⊥ = (I − â âᵀ) f_r ; l = r⊥ ; θ = angle(f⊥, −l) signed by whether f⊥ points toward the polygon side
    const fa = fr[0] * ax[0] + fr[1] * ax[1] + fr[2] * ax[2];
    const fp = [fr[0] - fa * ax[0], fr[1] - fa * ax[1], fr[2] - fa * ax[2]], fpl = Math.hypot(...fp);
    if (fpl > 1e-9) {
      // θ = angle between f⊥ and −r⊥ (COM → axis); σ = +1 if the moment of f⊥ about â restores (rotates COM inward)
      const cosang = -(rp[0] * fp[0] + rp[1] * fp[1] + rp[2] * fp[2]) / (rho * fpl);
      const theta = Math.acos(Math.max(-1, Math.min(1, cosang)));
      const mom = (rp[1] * fp[2] - rp[2] * fp[1]) * ax[0] + (rp[2] * fp[0] - rp[0] * fp[2]) * ax[1] + (rp[0] * fp[1] - rp[1] * fp[0]) * ax[2];
      const sigma = mom <= 0 ? 1 : -1;
      fasm = Math.min(fasm, sigma * theta * fpl / (mass * g));
    }
  }
  out.nesm = nesm; out.fasm = fasm;
  if (comVel) {
    const z = Math.max(1e-3, com[2] - zg), w0 = Math.sqrt(g / z);
    out.capture = [com[0] + comVel[0] / w0, com[1] + comVel[1] / w0, zg];
    out.captureMargin = polygonMargin(out.polygon, out.capture[0], out.capture[1]);
  }
  return out;
}
