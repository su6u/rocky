// Limb inverse kinematics on the true (skewed) joint axes of the kit.
//
// The elbow hinge of the official figure is not guaranteed to be parallel to the shoulder pitch axis, so a
// closed-form planar 2-link solution would be wrong. We solve  min ‖p(θ) − p*‖²  over the 3 servo angles with
// damped least squares (Levenberg–Marquardt):  Δθ = Jᵀ(J Jᵀ + λ² I)⁻¹ e,  J_k = â_k × (p − o_k),
// clamped to joint limits, warm-started from the previous solution. λ grows near singular configurations
// (straight elbow), which bounds joint velocity instead of producing the explosive steps of a pure inverse.
import { axisAngleMat, mat3mul, mat3vec, cross3 } from '../engine/linalg.mjs';

export class LimbIK {
  // joints: [{R0, r0, axis, lo, hi}] from the torso frame outwards; foot: point in the last joint's body frame.
  constructor(joints, foot) { this.joints = joints; this.foot = Float64Array.from(foot); this.theta = new Float64Array(joints.length); }
  fk(theta) {
    let R = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), p = new Float64Array(3);
    const origins = [], axes = [];
    for (let k = 0; k < this.joints.length; k++) {
      const J = this.joints[k];
      const o = mat3vec(R, J.r0); p = new Float64Array([p[0] + o[0], p[1] + o[1], p[2] + o[2]]);
      R = mat3mul(mat3mul(R, J.R0), axisAngleMat(J.axis, theta[k]));
      origins.push(p); axes.push(mat3vec(R, J.axis));
    }
    const f = mat3vec(R, this.foot);
    return { foot: [p[0] + f[0], p[1] + f[1], p[2] + f[2]], origins, axes, R };
  }
  // Knee-up analytic seed (law of cosines in the limb plane). The straight limb θ = 0 is a singular
  // configuration where the bend direction is ambiguous; DLS started there can converge to the elbow-up
  // branch and stall on joint limits. The seed selects the spider branch; DLS then absorbs axis skew/offsets.
  seed(target) {
    const [J0, , J2] = this.joints, F = J0.R0;
    const d = [target[0] - J0.r0[0], target[1] - J0.r0[1], target[2] - J0.r0[2]];
    const dl = [F[0] * d[0] + F[3] * d[1] + F[6] * d[2], F[1] * d[0] + F[4] * d[1] + F[7] * d[2], F[2] * d[0] + F[5] * d[1] + F[8] * d[2]];
    const A = Math.hypot(J2.r0[0], J2.r0[1], J2.r0[2]), B = Math.hypot(this.foot[0], this.foot[1], this.foot[2]);
    const yaw = Math.atan2(dl[1], dl[0]);
    const rho = Math.hypot(dl[0], dl[1]), D = Math.min(A + B - 1e-6, Math.max(Math.abs(A - B) + 1e-6, Math.hypot(rho, dl[2])));
    const alpha = Math.atan2(dl[2], rho);
    const beta = Math.acos(Math.max(-1, Math.min(1, (A * A + D * D - B * B) / (2 * A * D))));
    const gamma = Math.acos(Math.max(-1, Math.min(1, (A * A + B * B - D * D) / (2 * A * B))));
    const th = [yaw, alpha + beta, -(Math.PI - gamma)];
    for (let k = 0; k < 3; k++) th[k] = Math.max(this.joints[k].lo ?? -Math.PI, Math.min(this.joints[k].hi ?? Math.PI, th[k]));
    this.theta.set(th);
    return th;
  }
  solve(target, { iters = 30, tol = 1e-6, lambda0 = 0.002, reseed = 0.02 } = {}) {
    if (!this.seeded) { this.seed(target); this.seeded = true; }
    const th = Float64Array.from(this.theta), n = th.length;
    let err = Infinity, lam = lambda0;
    for (let it = 0; it < iters; it++) {
      const { foot, origins, axes } = this.fk(th);
      const e = [target[0] - foot[0], target[1] - foot[1], target[2] - foot[2]];
      err = Math.hypot(...e);
      if (err < tol) break;
      const J = axes.map((a, k) => cross3(a, [foot[0] - origins[k][0], foot[1] - origins[k][1], foot[2] - origins[k][2]]));
      // A = J Jᵀ + λ² I (3×3), solve A y = e, Δθ = Jᵀ y
      const A = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) { let s = 0; for (let k = 0; k < n; k++) s += J[k][r] * J[k][c]; A[3 * r + c] = s + (r === c ? lam * lam : 0); }
      const y = solve3(A, e);
      let step = 0;
      for (let k = 0; k < n; k++) {
        const d = J[k][0] * y[0] + J[k][1] * y[1] + J[k][2] * y[2];
        const nt = Math.max(this.joints[k].lo ?? -Math.PI, Math.min(this.joints[k].hi ?? Math.PI, th[k] + d));
        step = Math.max(step, Math.abs(nt - th[k])); th[k] = nt;
      }
      // adapt damping: shrink when progressing, grow near singularity (tiny det)
      const det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
      lam = det < 1e-12 ? Math.min(lam * 2, 0.05) : Math.max(lam * 0.7, lambda0);
      if (step < 1e-9) break;
    }
    this.theta.set(th);
    const res = this.fk(th).foot;
    const error = Math.hypot(target[0] - res[0], target[1] - res[1], target[2] - res[2]);
    if (error > 0.03 && !this._retry) { this._retry = true; this.seed(target); const r = this.solve(target, { iters: iters * 2, tol, lambda0, reseed: Infinity }); this._retry = false; return r.error < error ? r : { theta: Array.from(th), error }; }
    return { theta: Array.from(th), error };
  }
}

// Feasibility probe for planners: solves without disturbing the warm start used by the running controller.
LimbIK.prototype.probe = function probe(target) {
  const theta = Float64Array.from(this.theta), seeded = this.seeded;
  const r = this.solve(target);
  this.theta.set(theta); this.seeded = seeded;
  return r;
};

function solve3(A, b) {
  const [a, bb, c, d, e, f, g, h, i] = A;
  const det = a * (e * i - f * h) - bb * (d * i - f * g) + c * (d * h - e * g);
  const inv = [(e * i - f * h), -(bb * i - c * h), (bb * f - c * e), -(d * i - f * g), (a * i - c * g), -(a * f - c * d), (d * h - e * g), -(a * h - bb * g), (a * e - bb * d)];
  return [0, 1, 2].map((r) => (inv[3 * r] * b[0] + inv[3 * r + 1] * b[1] + inv[3 * r + 2] * b[2]) / det);
}
