// Floating-base articulated rigid-body dynamics (Featherstone, "Rigid Body Dynamics Algorithms", 2008).
//
// Conventions
//   q = [p_world(3), quat_wxyz(4), θ_1 … θ_n]            (nq = 7 + n)
//   v = [ω_base(3), v_base(3), θ̇_1 … θ̇_n]  base twist in BASE coordinates (Featherstone order)
//   Body i (i ≥ 1) hangs from parent λ(i) by a revolute joint: X_body→parent = R0_i · Rot(axis_i, θ_i),
//   joint origin r0_i expressed in parent coordinates. Spatial vectors are [angular; linear].
//
// Equations of motion:  M(q) v̇ + c(q, v) = τ + Jᵀ f
//   M  – composite rigid-body algorithm (CRBA) + reflected actuator inertia (armature) on the diagonal
//   c  – recursive Newton–Euler (RNEA) with v̇ = 0; gravity enters as a fictitious base acceleration.
import {
  cross3, mat3mul, mat3vec, mat3Tvec, axisAngleMat, quatToMat,
} from './linalg.mjs';

export class ArticulatedModel {
  constructor({ bodies, gravity = [0, 0, -9.81] }) {
    // bodies[0] is the floating base. Each body: {name, parent, R0(9), r0(3), axis(3), mass, com(3), Icom(9),
    //   armature, damping, qmin, qmax}
    this.bodies = bodies;
    this.nb = bodies.length;
    this.nj = this.nb - 1;
    this.nv = 6 + this.nj;
    this.nq = 7 + this.nj;
    this.gravity = Float64Array.from(gravity);
    this.parent = Int32Array.from(bodies.map((b) => b.parent ?? -1));
    for (let i = 1; i < this.nb; i++) if (!(this.parent[i] >= 0 && this.parent[i] < i)) throw new Error('bodies must be topologically ordered');
    this.dof = Int32Array.from(bodies.map((_, i) => (i === 0 ? 0 : 5 + i))); // body i joint → v index
    // ancestor chains for Jacobians (joint bodies only, excluding base)
    this.chain = bodies.map((_, i) => { const c = []; for (let j = i; j > 0; j = this.parent[j]) c.push(j); return c; });
    this.armature = new Float64Array(this.nv);
    this.damping = new Float64Array(this.nv);
    for (let i = 1; i < this.nb; i++) {
      this.armature[this.dof[i]] = bodies[i].armature ?? 0;
      this.damping[this.dof[i]] = bodies[i].damping ?? 0;
    }
  }
  totalMass() { return this.bodies.reduce((s, b) => s + b.mass, 0); }
}

// Parallel-axis tensor D(d) = (d·d) 1 − d dᵀ (row-major 9)
function addParallel(I, m, d) {
  const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  I[0] += m * (dd - d[0] * d[0]); I[1] -= m * d[0] * d[1]; I[2] -= m * d[0] * d[2];
  I[3] -= m * d[1] * d[0]; I[4] += m * (dd - d[1] * d[1]); I[5] -= m * d[1] * d[2];
  I[6] -= m * d[2] * d[0]; I[7] -= m * d[2] * d[1]; I[8] += m * (dd - d[2] * d[2]);
}

export class Kinematics {
  constructor(model) {
    const nb = model.nb;
    this.model = model;
    this.Rp = Array.from({ length: nb }, () => new Float64Array(9));   // body→parent rotation
    this.Rw = Array.from({ length: nb }, () => new Float64Array(9));   // body→world rotation
    this.pw = Array.from({ length: nb }, () => new Float64Array(3));   // body origin, world
    this.vel = Array.from({ length: nb }, () => new Float64Array(6));  // spatial velocity, body coords
    this.axisW = Array.from({ length: nb }, () => new Float64Array(3)); // joint axis, world
    this._A = new Float64Array(9);
  }
  // Forward kinematics (positions + velocities).
  update(q, v) {
    const m = this.model, B = m.bodies;
    quatToMat([q[3], q[4], q[5], q[6]], this.Rw[0]);
    this.pw[0].set([q[0], q[1], q[2]]);
    if (v) this.vel[0].set(v.subarray(0, 6));
    for (let i = 1; i < m.nb; i++) {
      const p = m.parent[i], b = B[i];
      axisAngleMat(b.axis, q[7 + i - 1], this._A);
      mat3mul(b.R0, this._A, this.Rp[i]);
      mat3mul(this.Rw[p], this.Rp[i], this.Rw[i]);
      const t = mat3vec(this.Rw[p], b.r0);
      this.pw[i][0] = this.pw[p][0] + t[0]; this.pw[i][1] = this.pw[p][1] + t[1]; this.pw[i][2] = this.pw[p][2] + t[2];
      mat3vec(this.Rw[i], b.axis, this.axisW[i]);
      if (v) {
        // v_i = X_i v_p + S θ̇ ; X_i: E = Rpᵀ, translation r0 (parent coords)
        const vp = this.vel[p], vi = this.vel[i], E = this.Rp[i];
        const wp = [vp[0], vp[1], vp[2]], lp = [vp[3], vp[4], vp[5]];
        const rxw = cross3(b.r0, wp);
        const w = mat3Tvec(E, wp), l = mat3Tvec(E, [lp[0] - rxw[0], lp[1] - rxw[1], lp[2] - rxw[2]]);
        const qd = v[5 + i];
        vi[0] = w[0] + b.axis[0] * qd; vi[1] = w[1] + b.axis[1] * qd; vi[2] = w[2] + b.axis[2] * qd;
        vi[3] = l[0]; vi[4] = l[1]; vi[5] = l[2];
      }
    }
  }
  // World position of a point given in body coordinates.
  pointWorld(i, local, out = new Float64Array(3)) {
    mat3vec(this.Rw[i], local, out);
    out[0] += this.pw[i][0]; out[1] += this.pw[i][1]; out[2] += this.pw[i][2];
    return out;
  }
  // World linear velocity of a world point p rigidly attached to body i.
  pointVelocity(i, p, out = new Float64Array(3)) {
    const R = this.Rw[i], vi = this.vel[i];
    const w = mat3vec(R, [vi[0], vi[1], vi[2]]), l = mat3vec(R, [vi[3], vi[4], vi[5]]);
    const d = [p[0] - this.pw[i][0], p[1] - this.pw[i][1], p[2] - this.pw[i][2]];
    const c = cross3(w, d);
    out[0] = l[0] + c[0]; out[1] = l[1] + c[1]; out[2] = l[2] + c[2];
    return out;
  }
  // Row of the linear point Jacobian along world direction dir: (dir·∂v_p/∂v). Adds `sign`×row into out.
  pointJacobianDir(i, p, dir, out, sign = 1) {
    const m = this.model, R0 = this.Rw[0], p0 = this.pw[0];
    // base: v_p = R0 v_b − [p−p0]× R0 ω_b  →  dir·v_p = (R0ᵀ dir)·v_b + (R0ᵀ((p−p0)×dir))·ω_b
    const d = [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]];
    const a = mat3Tvec(R0, cross3(d, dir)), l = mat3Tvec(R0, dir);
    out[0] += sign * a[0]; out[1] += sign * a[1]; out[2] += sign * a[2];
    out[3] += sign * l[0]; out[4] += sign * l[1]; out[5] += sign * l[2];
    for (const j of m.chain[i]) {
      const ax = this.axisW[j], o = this.pw[j];
      const col = cross3(ax, [p[0] - o[0], p[1] - o[1], p[2] - o[2]]);
      out[m.dof[j]] += sign * (col[0] * dir[0] + col[1] * dir[1] + col[2] * dir[2]);
    }
    return out;
  }
  // Whole-body centre of mass (world) and total mass.
  com(out = new Float64Array(3)) {
    const B = this.model.bodies; let M = 0; out.fill(0);
    for (let i = 0; i < B.length; i++) {
      if (!B[i].mass) continue;
      const c = this.pointWorld(i, B[i].com);
      out[0] += B[i].mass * c[0]; out[1] += B[i].mass * c[1]; out[2] += B[i].mass * c[2]; M += B[i].mass;
    }
    out[0] /= M; out[1] /= M; out[2] /= M;
    return out;
  }
  // Linear and angular momentum about the world origin (world coordinates), and kinetic energy.
  momentum() {
    const B = this.model.bodies; const P = [0, 0, 0], L = [0, 0, 0]; let T = 0;
    for (let i = 0; i < B.length; i++) {
      const b = B[i]; if (!b.mass) continue;
      const R = this.Rw[i], vi = this.vel[i];
      const w = [vi[0], vi[1], vi[2]];
      const vc = [vi[3] + (w[1] * b.com[2] - w[2] * b.com[1]), vi[4] + (w[2] * b.com[0] - w[0] * b.com[2]), vi[5] + (w[0] * b.com[1] - w[1] * b.com[0])];
      const Iw = mat3vec(b.Icom, w);
      T += 0.5 * b.mass * (vc[0] ** 2 + vc[1] ** 2 + vc[2] ** 2) + 0.5 * (w[0] * Iw[0] + w[1] * Iw[1] + w[2] * Iw[2]);
      const vcw = mat3vec(R, vc), Iww = mat3vec(R, Iw), cw = this.pointWorld(i, b.com);
      P[0] += b.mass * vcw[0]; P[1] += b.mass * vcw[1]; P[2] += b.mass * vcw[2];
      const rxp = cross3(cw, [b.mass * vcw[0], b.mass * vcw[1], b.mass * vcw[2]]);
      L[0] += rxp[0] + Iww[0]; L[1] += rxp[1] + Iww[1]; L[2] += rxp[2] + Iww[2];
    }
    return { P, L, T };
  }
}

// Spatial inertia (m, c, Ic) applied to motion [w; l] → force [n; f]
function inertiaApply(m, c, Ic, w, l, out) {
  const wxc = [w[1] * c[2] - w[2] * c[1], w[2] * c[0] - w[0] * c[2], w[0] * c[1] - w[1] * c[0]];
  const f = [m * (l[0] + wxc[0]), m * (l[1] + wxc[1]), m * (l[2] + wxc[2])];
  const Iw = mat3vec(Ic, w), cxf = cross3(c, f);
  out[0] = Iw[0] + cxf[0]; out[1] = Iw[1] + cxf[1]; out[2] = Iw[2] + cxf[2];
  out[3] = f[0]; out[4] = f[1]; out[5] = f[2];
  return out;
}

export class Dynamics {
  constructor(model) {
    const nb = model.nb, nv = model.nv;
    this.model = model;
    this.M = new Float64Array(nv * nv);
    this.c = new Float64Array(nv);
    this.cm = new Float64Array(nb); this.cc = Array.from({ length: nb }, () => new Float64Array(3)); this.cI = Array.from({ length: nb }, () => new Float64Array(9));
    this.acc = Array.from({ length: nb }, () => new Float64Array(6));
    this.frc = Array.from({ length: nb }, () => new Float64Array(6));
  }
  // Composite rigid-body algorithm. Fills this.M (nv×nv, symmetric) including armature.
  massMatrix(kin) {
    const m = this.model, B = m.bodies, nv = m.nv, M = this.M;
    M.fill(0);
    for (let i = 0; i < m.nb; i++) { this.cm[i] = B[i].mass; this.cc[i].set(B[i].com); this.cI[i].set(B[i].Icom); }
    for (let i = m.nb - 1; i > 0; i--) {
      const p = m.parent[i], Rp = kin.Rp[i];
      const mi = this.cm[i]; if (mi <= 0) continue;
      // child composite → parent coords
      const cp = mat3vec(Rp, this.cc[i]); cp[0] += B[i].r0[0]; cp[1] += B[i].r0[1]; cp[2] += B[i].r0[2];
      const Ip = mat3mul(mat3mul(Rp, this.cI[i]), [Rp[0], Rp[3], Rp[6], Rp[1], Rp[4], Rp[7], Rp[2], Rp[5], Rp[8]]);
      const mp = this.cm[p], mt = mp + mi;
      const cnew = [(mp * this.cc[p][0] + mi * cp[0]) / mt, (mp * this.cc[p][1] + mi * cp[1]) / mt, (mp * this.cc[p][2] + mi * cp[2]) / mt];
      const I = this.cI[p];
      addParallel(I, mp, [this.cc[p][0] - cnew[0], this.cc[p][1] - cnew[1], this.cc[p][2] - cnew[2]]);
      for (let k = 0; k < 9; k++) I[k] += Ip[k];
      addParallel(I, mi, [cp[0] - cnew[0], cp[1] - cnew[1], cp[2] - cnew[2]]);
      this.cm[p] = mt; this.cc[p].set(cnew);
    }
    const F = new Float64Array(6), tmp = new Float64Array(6);
    for (let i = 1; i < m.nb; i++) {
      const k = m.dof[i], ax = B[i].axis;
      inertiaApply(this.cm[i], this.cc[i], this.cI[i], ax, [0, 0, 0], F);
      M[k * nv + k] = ax[0] * F[0] + ax[1] * F[1] + ax[2] * F[2];
      let j = i;
      while (true) {
        // F ← X_jᵀ F  (child j coords → parent coords)
        const Rp = kin.Rp[j], r = B[j].r0;
        const n = mat3vec(Rp, [F[0], F[1], F[2]]), f = mat3vec(Rp, [F[3], F[4], F[5]]);
        const rxf = cross3(r, f);
        tmp[0] = n[0] + rxf[0]; tmp[1] = n[1] + rxf[1]; tmp[2] = n[2] + rxf[2]; tmp[3] = f[0]; tmp[4] = f[1]; tmp[5] = f[2];
        F.set(tmp);
        j = m.parent[j];
        if (j === 0) { for (let r6 = 0; r6 < 6; r6++) { M[r6 * nv + k] = F[r6]; M[k * nv + r6] = F[r6]; } break; }
        const kj = m.dof[j], axj = B[j].axis;
        const h = axj[0] * F[0] + axj[1] * F[1] + axj[2] * F[2];
        M[kj * nv + k] = h; M[k * nv + kj] = h;
      }
    }
    // base block: spatial inertia of the whole robot about the base origin
    const mt = this.cm[0], c = this.cc[0], I = Float64Array.from(this.cI[0]);
    addParallel(I, mt, c);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) M[a * nv + b] = I[3 * a + b];
    const cx = [0, -c[2], c[1], c[2], 0, -c[0], -c[1], c[0], 0];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { M[a * nv + 3 + b] = mt * cx[3 * a + b]; M[(3 + b) * nv + a] = mt * cx[3 * a + b]; }
    for (let a = 0; a < 3; a++) M[(3 + a) * nv + 3 + a] = mt;
    for (let k = 6; k < nv; k++) M[k * nv + k] += m.armature[k];
    return M;
  }
  // Recursive Newton–Euler: returns τ = M(q) vdot + c(q,v) (generalized forces), gravity included.
  // With vdot = null computes the bias vector c.
  inverseDynamics(kin, v, vdot, out = this.c, gravity = this.model.gravity) {
    const m = this.model, B = m.bodies;
    const R0 = kin.Rw[0];
    const g = mat3Tvec(R0, [-gravity[0], -gravity[1], -gravity[2]]);
    const a0 = this.acc[0];
    a0[0] = 0; a0[1] = 0; a0[2] = 0; a0[3] = g[0]; a0[4] = g[1]; a0[5] = g[2];
    if (vdot) for (let k = 0; k < 6; k++) a0[k] += vdot[k];
    const f = new Float64Array(6);
    const netForce = (i) => {
      const b = B[i], vi = kin.vel[i], ai = this.acc[i];
      const w = [vi[0], vi[1], vi[2]], l = [vi[3], vi[4], vi[5]];
      const Ia = inertiaApply(b.mass, b.com, b.Icom, [ai[0], ai[1], ai[2]], [ai[3], ai[4], ai[5]], new Float64Array(6));
      const Iv = inertiaApply(b.mass, b.com, b.Icom, w, l, new Float64Array(6));
      // crf(v) Iv = [w×n + l×f ; w×f]
      const wxn = cross3(w, [Iv[0], Iv[1], Iv[2]]), lxf = cross3(l, [Iv[3], Iv[4], Iv[5]]), wxf = cross3(w, [Iv[3], Iv[4], Iv[5]]);
      const fr = this.frc[i];
      fr[0] = Ia[0] + wxn[0] + lxf[0]; fr[1] = Ia[1] + wxn[1] + lxf[1]; fr[2] = Ia[2] + wxn[2] + lxf[2];
      fr[3] = Ia[3] + wxf[0]; fr[4] = Ia[4] + wxf[1]; fr[5] = Ia[5] + wxf[2];
    };
    netForce(0);
    for (let i = 1; i < m.nb; i++) {
      const p = m.parent[i], b = B[i], E = kin.Rp[i], ap = this.acc[p], ai = this.acc[i], vi = kin.vel[i];
      const wp = [ap[0], ap[1], ap[2]], rxw = cross3(b.r0, wp);
      const w = mat3Tvec(E, wp), l = mat3Tvec(E, [ap[3] - rxw[0], ap[4] - rxw[1], ap[5] - rxw[2]]);
      const qd = v[5 + i], qdd = vdot ? vdot[5 + i] : 0;
      // crm(v_i) S θ̇ = [ω_i × s θ̇ ; l_i × s θ̇]
      const s = [b.axis[0] * qd, b.axis[1] * qd, b.axis[2] * qd];
      const wxs = cross3([vi[0], vi[1], vi[2]], s), lxs = cross3([vi[3], vi[4], vi[5]], s);
      ai[0] = w[0] + b.axis[0] * qdd + wxs[0]; ai[1] = w[1] + b.axis[1] * qdd + wxs[1]; ai[2] = w[2] + b.axis[2] * qdd + wxs[2];
      ai[3] = l[0] + lxs[0]; ai[4] = l[1] + lxs[1]; ai[5] = l[2] + lxs[2];
      netForce(i);
    }
    out.fill(0);
    for (let i = m.nb - 1; i > 0; i--) {
      const b = B[i], fr = this.frc[i], k = m.dof[i];
      out[k] = b.axis[0] * fr[0] + b.axis[1] * fr[1] + b.axis[2] * fr[2] + (vdot ? m.armature[k] * vdot[k] : 0);
      const p = m.parent[i], Rp = kin.Rp[i];
      const n = mat3vec(Rp, [fr[0], fr[1], fr[2]]), ff = mat3vec(Rp, [fr[3], fr[4], fr[5]]);
      const rxf = cross3(b.r0, ff), fp = this.frc[p];
      fp[0] += n[0] + rxf[0]; fp[1] += n[1] + rxf[1]; fp[2] += n[2] + rxf[2]; fp[3] += ff[0]; fp[4] += ff[1]; fp[5] += ff[2];
    }
    for (let k = 0; k < 6; k++) out[k] = this.frc[0][k];
    return out;
  }
}
