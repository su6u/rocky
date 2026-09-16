// Time stepping with contact, friction, self-collision, joint limits, backlash and servo actuators.
//
// Discretisation (semi-implicit Euler with implicit damping, velocity-level constraints):
//   (M + h D) v⁺ = M v + h (τ − c) + Jᵀ λ,      q⁺ = q ⊕ h v⁺
// with D = diag(back-EMF + viscous + passive damping). Constraint impulses λ solve a mixed
// complementarity problem by projected Gauss–Seidel (PGS) on the Delassus operator W = J (M + hD)⁻¹ Jᵀ:
//   contact normal   0 ≤ λ_n ⊥ (v_n⁺ + γ λ_n − b) ≥ 0      (implicit Hunt–Crossley pad)
//   Coulomb friction ‖λ_t‖ ≤ μ λ_n, v_t⁺ → 0 while inside the disk (maximal dissipation when sliding)
//   servo friction   |λ_f| ≤ h τ_f^max  with v_joint⁺ → 0 while inside (exact stiction)
//   limits/backlash  unilateral rows with implicit stiffness and damping
// Hunt–Crossley pad: F = k δ^{3/2} (1 + α δ̇). Linearised implicitly around δ each step:
//   F⁺ ≈ k δ^{3/2} − (h k_t + c_t) v_n⁺,  k_t = 1.5 k δ^{1/2}, c_t = α k δ^{3/2}
//   ⇒ γ = 1 / (h (h k_t + c_t)),  b = k δ^{3/2} / (h k_t + c_t)
// Separated contacts within a margin are speculative rows: v_n⁺ ≥ −gap/h (no tunnelling, no force).
// Self-collision: collision spheres on different limbs, and lower limbs against the carapace, contact each other
// with the same pad law; the relative-velocity Jacobian is J_a(p)·n − J_b(p)·n.
//
// Memory: constraint rows, the Delassus matrix and PGS buffers are pooled and reused between steps, so a step makes
// almost no garbage (the browser's collector otherwise causes frame-time spikes).
import { Kinematics, Dynamics } from './dynamics.mjs';
import { cholesky, cholSolve, quatIntegrateBody } from './linalg.mjs';
import { Terrain } from './terrain.mjs';
export { Terrain } from './terrain.mjs';

export const ROW = { NORMAL: 1, UNILATERAL: 2, TANGENT1: 3, TANGENT2: 4, BOX: 5 };
const KEY = { ground: 1, box: 2, self: 3, limit: 4, friction: 5 };

class RowPool {
  constructor(nv) { this.nv = nv; this.rows = []; this.count = 0; }
  reset() { this.count = 0; }
  next() {
    let R = this.rows[this.count];
    if (!R) { R = { J: new Float64Array(this.nv), nz: new Int32Array(this.nv), nnz: 0 }; this.rows.push(R); }
    else R.J.fill(0);
    R.nnz = 0; R.type = 0; R.key = 0; R.gamma = 0; R.bias = 0; R.mu = 0; R.bound = 0; R.normalRow = -1; R.kind = null; R.dof = -1; R.side = 0;
    this.count++;
    return R;
  }
}

export class World {
  constructor(model, { dt = 0.001, terrain = new Terrain(), pad = { k: 3.0e5, alpha: 0.6, margin: 0.004 }, iterations = 40, tolerance = 0, fixedBase = false } = {}) {
    this.model = model; this.dt = dt; this.terrain = terrain; this.pad = pad; this.iterations = iterations; this.fixedBase = fixedBase;
    this.tolerance = tolerance;   // PGS stops when the largest impulse change falls below this (N·s); 0 = run all iterations
    this.kin = new Kinematics(model); this.dyn = new Dynamics(model);
    const nv = model.nv;
    this.q = new Float64Array(model.nq); this.q[3] = 1; // identity quaternion [w,x,y,z] at q[3..6]
    this.v = new Float64Array(nv);
    this.time = 0;
    this.Mhat = new Float64Array(nv * nv); this.L = new Float64Array(nv * nv);
    this.tau = new Float64Array(nv); this.D = new Float64Array(nv); this.rhs = new Float64Array(nv); this.vu = new Float64Array(nv);
    this.contactShapes = [];     // {body, local:[3], radius, kind:'foot'|'shell', padK?, name, limb?, group?}
    this.selfPairs = [];         // [shapeA, shapeB, k, mu] pairs allowed to collide (see enableSelfCollision)
    this.actuators = [];         // {servo, dof, backlashDof, body, limb, joint}
    this.limits = [];            // {dof, lo, hi, k, zeta, inertia}
    this.warm = new Map();
    this.externalForces = [];    // {body, local, force(world), until}
    this.battery = null;
    this.contacts = []; this.stats = { iterations: 0, residual: 0, rows: 0 };
    this.lastVdot = new Float64Array(nv);
    this.frictionTorque = new Float64Array(nv);
    this.limitHits = [];
    this.pool = new RowPool(nv);
    this.buf = { cap: 0, lambda: null, Y: null, W: null, u: null, vel: null };
    this._p = new Float64Array(3); this._c = new Float64Array(3); this._c2 = new Float64Array(3); this._dir = new Float64Array(3);
    this._rowExt = new Float64Array(nv);
  }

  addExternalForce(body, local, forceWorld, duration) { this.externalForces.push({ body, local, force: forceWorld, until: this.time + duration }); }

  // Allow collisions between collision spheres of different kinematic groups (limbs, carapace), except pairs that
  // already overlap in the reference configuration (the shoulder sockets sit inside the carapace by construction).
  enableSelfCollision({ k, mu = 0.3, exclude = () => false } = {}) {
    const kin = this.kin, S = this.contactShapes;
    kin.update(this.q, this.v);
    const centers = S.map((s) => kin.pointWorld(s.body, s.local));
    this.selfPairs = [];
    for (let a = 0; a < S.length; a++) for (let b = a + 1; b < S.length; b++) {
      const A = S[a], B = S[b];
      if (A.group === undefined || B.group === undefined || A.group === B.group || exclude(A, B)) continue;
      const d = Math.hypot(centers[a][0] - centers[b][0], centers[a][1] - centers[b][1], centers[a][2] - centers[b][2]);
      if (d < A.radius + B.radius + 0.002) continue;
      const Rstar = (A.radius * B.radius) / (A.radius + B.radius);
      this.selfPairs.push([a, b, k(Rstar), mu]);
    }
    return this.selfPairs.length;
  }

  step() {
    const m = this.model, nv = m.nv, h = this.dt, kin = this.kin, dyn = this.dyn;
    const q = this.q, v = this.v;
    kin.update(q, v);
    const M = dyn.massMatrix(kin);
    const c = dyn.inverseDynamics(kin, v, null);
    const tau = this.tau; tau.fill(0);
    const D = this.D; D.set(m.damping);
    const vBus = this.battery ? this.battery.voltage : 1e9;
    // Servo drive torques (explicit) and back-EMF + viscous damping (implicit)
    for (const a of this.actuators) {
      const enc = q[7 + a.dof - 6] + (a.backlashDof >= 0 ? q[7 + a.backlashDof - 6] : 0);
      const { tauExplicit, damping } = a.servo.computeDrive(this.time, h, enc, v[a.dof], vBus);
      tau[a.dof] += tauExplicit;
      D[a.dof] += damping + (a.servo.p.friction.viscous ?? 0);
    }
    // External point forces (disturbances): τ += Jᵀ F
    if (this.externalForces.length) {
      this.externalForces = this.externalForces.filter((e) => e.until > this.time);
      const row = this._rowExt, dir = this._dir;
      for (const e of this.externalForces) {
        const p = kin.pointWorld(e.body, e.local, this._p);
        for (let ax = 0; ax < 3; ax++) {
          if (!e.force[ax]) continue;
          row.fill(0); dir[0] = 0; dir[1] = 0; dir[2] = 0; dir[ax] = 1;
          kin.pointJacobianDir(e.body, p, dir, row);
          for (let k = 0; k < nv; k++) tau[k] += row[k] * e.force[ax];
        }
      }
    }
    // M̂ = M + hD ; rhs = h(τ − c − D v)
    const Mh = this.Mhat; Mh.set(M);
    for (let k = 0; k < nv; k++) { Mh[k * nv + k] += h * D[k]; this.rhs[k] = h * (tau[k] - c[k] - D[k] * v[k]); }
    if (this.fixedBase) {
      for (let a = 0; a < 6; a++) { for (let k = 0; k < nv; k++) { Mh[a * nv + k] = 0; Mh[k * nv + a] = 0; } Mh[a * nv + a] = 1; this.rhs[a] = 0; v[a] = 0; }
    }
    this.L.set(Mh);
    if (!cholesky(this.L, nv)) throw new Error('mass matrix not positive definite');
    this.vu.set(this.rhs); cholSolve(this.L, nv, this.vu);
    for (let k = 0; k < nv; k++) this.vu[k] += v[k];

    // ---- constraint rows ----
    this.buildRows();
    const rows = this.pool.rows, nr = this.pool.count;
    this.stats.rows = nr;
    const lambda = this.solveConstraints(rows, nr);
    // accelerations for actuator bookkeeping
    for (let k = 0; k < nv; k++) this.lastVdot[k] = (this.vu[k] - v[k]) / h;
    v.set(this.vu);
    kin.update(q, v);                                   // post-impulse velocities for slip diagnostics
    this.recordConstraintForces(rows, nr, lambda, h);
    // integrate configuration
    const R0 = kin.Rw[0];
    const vx = v[3], vy = v[4], vz = v[5];
    q[0] += h * (R0[0] * vx + R0[1] * vy + R0[2] * vz);
    q[1] += h * (R0[3] * vx + R0[4] * vy + R0[5] * vz);
    q[2] += h * (R0[6] * vx + R0[7] * vy + R0[8] * vz);
    const qn = quatIntegrateBody([q[3], q[4], q[5], q[6]], [v[0], v[1], v[2]], h);
    q[3] = qn[0]; q[4] = qn[1]; q[5] = qn[2]; q[6] = qn[3];
    for (let k = 6; k < nv; k++) q[k + 1] += h * v[k];
    // actuator + battery bookkeeping
    let busCurrent = 0;
    for (const a of this.actuators) {
      a.servo.finish(h, v[a.dof], this.lastVdot[a.dof], this.frictionTorque[a.dof], vBus);
      busCurrent += Math.max(0, a.servo.supplyCurrent ?? 0);
    }
    if (this.battery) this.battery.update(busCurrent + (this.quiescentCurrent ?? 0), h);   // + controller electronics
    this.time += h;
  }

  // Projected Gauss–Seidel on W = J M̂⁻¹ Jᵀ with warm start. Updates this.vu to v⁺ and returns λ (impulses).
  solveConstraints(rows, nr) {
    const nv = this.model.nv, B = this.buf;
    if (B.cap < nr) {
      const cap = Math.max(nr, 2 * B.cap, 32);
      Object.assign(B, { cap, lambda: new Float64Array(cap), Y: new Float64Array(cap * nv), W: new Float64Array(cap * cap), u: new Float64Array(cap), vel: new Float64Array(cap) });
    }
    const { lambda, Y, W, u, vel } = B;
    this.stats.iterations = 0; this.stats.residual = 0;
    if (!nr) return lambda;
    // Y = M̂⁻¹ Jᵀ (columns), W = J Y, u = J v_u
    for (let r = 0; r < nr; r++) { Y.set(rows[r].J, r * nv); cholSolve(this.L, nv, Y, r * nv, 1); }
    for (let r = 0; r < nr; r++) {
      const J = rows[r].J, nz = rows[r].nz, nnz = rows[r].nnz;
      let s = 0; for (let i = 0; i < nnz; i++) { const k = nz[i]; s += J[k] * this.vu[k]; }
      u[r] = s;
      for (let s2 = r; s2 < nr; s2++) {
        let w = 0; const off = s2 * nv;
        for (let i = 0; i < nnz; i++) { const k = nz[i]; w += J[k] * Y[off + k]; }
        W[r * nr + s2] = w; W[s2 * nr + r] = w;
      }
    }
    for (let r = 0; r < nr; r++) lambda[r] = this.warm.get(rows[r].key) ?? 0;
    for (let r = 0; r < nr; r++) { let s = u[r]; const off = r * nr; for (let s2 = 0; s2 < nr; s2++) s += W[off + s2] * lambda[s2]; vel[r] = s; }
    const apply = (r, d) => { if (d === 0) return; lambda[r] += d; for (let s2 = 0; s2 < nr; s2++) vel[s2] += W[s2 * nr + r] * d; };
    let it = 0;
    for (it = 0; it < this.iterations; it++) {
      let change = 0;
      for (let r = 0; r < nr; r++) {
        const R = rows[r];
        if (R.type === ROW.NORMAL || R.type === ROW.UNILATERAL) {
          const denom = W[r * nr + r] + R.gamma;
          const nl = Math.max(0, lambda[r] - (vel[r] + R.gamma * lambda[r] - R.bias) / denom);
          const d = nl - lambda[r]; if (Math.abs(d) > change) change = Math.abs(d); apply(r, d);
        } else if (R.type === ROW.TANGENT1) {
          // Exact Coulomb block: min ½λᵀAλ + λᵀe  s.t. ‖λ‖ ≤ μλ_n  (A = W_tt + γI, e = slip velocity without this impulse).
          // Outside the disk the KKT point is λ = −(A + νI)⁻¹e with ν ≥ 0 solving ‖λ(ν)‖ = μλ_n, which makes λ
          // anti-parallel to the post-impulse slip (maximal dissipation). Solved by Newton on the secular equation.
          const r2 = r + 1, rad = Math.max(0, lambda[R.normalRow] * R.mu);
          const a = W[r * nr + r] + R.gamma, b = W[r * nr + r2], d = W[r2 * nr + r2] + R.gamma;
          const e1 = vel[r] + R.gamma * lambda[r] - (a * lambda[r] + b * lambda[r2]);
          const e2 = vel[r2] + R.gamma * lambda[r2] - (b * lambda[r] + d * lambda[r2]);
          coulombBlock(a, b, d, e1, e2, rad, this._c);
          const d1 = this._c[0] - lambda[r], d2 = this._c[1] - lambda[r2];
          change = Math.max(change, Math.abs(d1), Math.abs(d2));
          apply(r, d1); apply(r2, d2);
          r++;
        } else if (R.type === ROW.BOX) {
          const denom = W[r * nr + r] + R.gamma;
          const nl = Math.max(-R.bound, Math.min(R.bound, lambda[r] - (vel[r] + R.gamma * lambda[r]) / denom));
          const d = nl - lambda[r]; if (Math.abs(d) > change) change = Math.abs(d); apply(r, d);
        }
      }
      if (change <= this.tolerance) { it++; break; }
    }
    this.stats.iterations = it;
    // residual: normal complementarity (velocity units, m/s or rad/s)
    let res = 0;
    for (let r = 0; r < nr; r++) {
      const R = rows[r];
      if (R.type === ROW.NORMAL || R.type === ROW.UNILATERAL) { const w = vel[r] + R.gamma * lambda[r] - R.bias; res = Math.max(res, lambda[r] > 1e-12 ? Math.abs(w) : Math.max(0, -w)); }
    }
    this.stats.residual = res;
    this.warm.clear();
    for (let r = 0; r < nr; r++) if (lambda[r] !== 0) this.warm.set(rows[r].key, lambda[r]);
    // v⁺ = v_u + Y λ
    const vnew = this.vu;
    for (let r = 0; r < nr; r++) { const l = lambda[r]; if (!l) continue; const off = r * nv; for (let k = 0; k < nv; k++) vnew[k] += Y[off + k] * l; }
    return lambda;
  }

  // Record the non-zero pattern of a finished row (and drop base columns for a fixed base).
  finalize(R) {
    if (this.fixedBase) R.J.fill(0, 0, 6);
    let n = 0; const J = R.J;
    for (let k = 0; k < J.length; k++) if (J[k] !== 0) R.nz[n++] = k;
    R.nnz = n;
    return R;
  }

  buildRows() {
    const kin = this.kin, h = this.dt, v = this.v, T = this.terrain, S = this.contactShapes;
    this.pool.reset();
    this.contacts = [];
    // --- ground plane and box obstacles ---
    const n = T.planeNormal();
    const t1 = [Math.cos(T.slope), 0, Math.sin(T.slope)], t2 = [0, 1, 0];
    const centers = this._centers ??= [];
    for (let si = 0; si < S.length; si++) {
      const shape = S[si];
      const cw = centers[si] ??= new Float64Array(3);
      kin.pointWorld(shape.body, shape.local, cw);
      for (let bi = 0; bi < T.boxes.length; bi++) this.boxContact(shape, si, cw, T.boxes[bi], bi);
      const dist = cw[0] * n[0] + cw[1] * n[1] + cw[2] * n[2] - shape.radius; // signed gap along the plane normal
      if (dist > 0) continue;
      const off = shape.radius + dist * 0.5;
      const p = [cw[0] - n[0] * off, cw[1] - n[1] * off, cw[2] - n[2] * off];
      this.addContactRows(shape.body, -1, si, KEY.ground, si, p, n, t1, t2, dist, T.muAt(p[0], p[1]), shape.padK ?? this.pad.k, T.alpha ?? this.pad.alpha, false, shape);
    }
    // --- self-collision between spheres of different limbs / carapace ---
    const margin = this.pad.margin;
    for (let i = 0; i < this.selfPairs.length; i++) {
      const [a, b, k, mu] = this.selfPairs[i];
      const A = S[a], B = S[b], ca = centers[a], cb = centers[b];
      const dx = ca[0] - cb[0], dy = ca[1] - cb[1], dz = ca[2] - cb[2];
      const reach = A.radius + B.radius + margin;
      if (dx * dx + dy * dy + dz * dz > reach * reach) continue;
      const d = Math.hypot(dx, dy, dz);
      if (d < 1e-9) continue;
      const gap = d - A.radius - B.radius;
      const nn = [dx / d, dy / d, dz / d];                                   // from B toward A
      const off = B.radius + gap * 0.5;
      const p = [cb[0] + nn[0] * off, cb[1] + nn[1] * off, cb[2] + nn[2] * off];
      const [u1, u2] = tangentBasis(nn);
      this.addContactRows(A.body, B.body, a, KEY.self, i, p, nn, u1, u2, gap, mu, k, T.alpha ?? this.pad.alpha, gap > 0, A, B);
    }
    // --- joint limits ---
    for (const L of this.limits) {
      const qk = this.q[L.dof + 1], vk = v[L.dof];
      for (let side = 0; side < 2; side++) {
        const bound = side ? L.hi : L.lo, sign = side ? -1 : 1;
        const gap = sign * (qk - bound); // ≥ 0 inside range
        if (gap > Math.max(0.02, Math.abs(vk) * h * 2)) continue;
        const R = this.pool.next(); R.J[L.dof] = sign; this.finalize(R);
        R.type = ROW.UNILATERAL; R.key = KEY.limit * 1e6 + L.dof * 2 + side; R.kind = 'limit'; R.dof = L.dof; R.side = side;
        if (gap >= 0) { R.gamma = 0; R.bias = -gap / h; }
        else { const kk = L.k, cdamp = 2 * L.zeta * Math.sqrt(kk * (L.inertia ?? 0.02)); R.gamma = 1 / (h * (h * kk + cdamp)); R.bias = (kk * -gap) / (h * kk + cdamp); }
      }
    }
    // --- servo gearbox friction (box rows); budgets from the BAM model using last step's torques. The gear train
    //     keeps its friction when a servo is unpowered or failed. ---
    for (const a of this.actuators) {
      const budget = a.servo.frictionBudget(v[a.dof]);
      const R = this.pool.next(); R.J[a.dof] = 1; this.finalize(R);
      R.type = ROW.BOX; R.bound = h * budget; R.gamma = 1e-9; R.key = KEY.friction * 1e6 + a.dof; R.kind = 'friction'; R.dof = a.dof;
    }
  }

  // Normal + two tangent rows for a contact between body a and body b (b = −1: the static environment).
  addContactRows(bodyA, bodyB, shape, keyKind, keyIndex, p, n, t1, t2, dist, mu, k, alpha, speculative, shapeA, shapeB = null) {
    const kin = this.kin, h = this.dt, pool = this.pool;
    const jac = (R, dir) => { kin.pointJacobianDir(bodyA, p, dir, R.J, 1); if (bodyB >= 0) kin.pointJacobianDir(bodyB, p, dir, R.J, -1); this.finalize(R); };
    const nrow = pool.count;
    const Rn = pool.next(); jac(Rn, n);
    Rn.type = ROW.NORMAL; Rn.key = (keyKind * 1e5 + keyIndex) * 4;
    const pen = -dist;
    if (speculative) { Rn.gamma = 0; Rn.bias = -dist / h; }   // separated but closing: stop at the surface, no force
    else {
      // implicit Hunt–Crossley pad, linearised about the current penetration (floor avoids γ→∞ at first touch)
      const d = Math.max(pen, 0), d15 = Math.pow(d, 1.5), kt = 1.5 * k * Math.sqrt(Math.max(d, 1e-7));
      const ct = alpha * k * d15;
      Rn.gamma = 1 / (h * (h * kt + ct)); Rn.bias = (k * d15) / (h * kt + ct);
    }
    const Rt1 = pool.next(); jac(Rt1, t1);
    const Rt2 = pool.next(); jac(Rt2, t2);
    Rt1.type = ROW.TANGENT1; Rt2.type = ROW.TANGENT2;
    Rt1.normalRow = nrow; Rt1.mu = mu; Rt1.gamma = 1e-7; Rt2.gamma = 1e-7;
    Rt1.key = Rn.key + 1; Rt2.key = Rn.key + 2;
    this.contacts.push({ shape, body: bodyA, otherBody: bodyB, point: p, normal: n, t1, t2, rowN: nrow, pen, mu,
      kind: bodyB >= 0 ? 'self' : shapeA.kind, name: shapeA.name, other: shapeB ? shapeB.name : null, env: keyKind === KEY.box ? 'box' : bodyB >= 0 ? 'self' : 'ground' });
  }

  boxContact(S, si, c, box, bi) {
    // Axis-aligned box obstacle {min:[3], max:[3]}; sphere–box closest point, or the nearest face when the centre is inside.
    let cp = [0, 1, 2].map((i) => Math.max(box.min[i], Math.min(box.max[i], c[i])));
    let d = [c[0] - cp[0], c[1] - cp[1], c[2] - cp[2]];
    let len = Math.hypot(d[0], d[1], d[2]), nrm, gap;
    if (len > 1e-9) {
      gap = len - S.radius;
      if (gap > this.pad.margin) return;
      nrm = [d[0] / len, d[1] / len, d[2] / len];
    } else {
      // centre inside the box: push out through the closest face
      let best = Infinity, axis = 2, sgn = 1;
      for (let i = 0; i < 3; i++) {
        const lo = c[i] - box.min[i], hi = box.max[i] - c[i];
        if (lo < best) { best = lo; axis = i; sgn = -1; }
        if (hi < best) { best = hi; axis = i; sgn = 1; }
      }
      nrm = [0, 0, 0]; nrm[axis] = sgn;
      cp = [...c]; cp[axis] = sgn > 0 ? box.max[axis] : box.min[axis];
      gap = -best - S.radius;
    }
    const [u1, u2] = tangentBasis(nrm);
    this.addContactRows(S.body, -1, si, KEY.box, bi * 1000 + si, cp, nrm, u1, u2, gap, box.mu ?? this.terrain.mu, S.padK ?? this.pad.k, this.terrain.alpha ?? this.pad.alpha, gap > 0, S);
  }

  recordConstraintForces(rows, nr, lambda, h) {
    for (const ci of this.contacts) {
      const ln = lambda[ci.rowN] / h, lt1 = lambda[ci.rowN + 1] / h, lt2 = lambda[ci.rowN + 2] / h;
      ci.fn = ln; ci.ft = [lt1, lt2];
      ci.force = [ci.normal[0] * ln + ci.t1[0] * lt1 + ci.t2[0] * lt2, ci.normal[1] * ln + ci.t1[1] * lt1 + ci.t2[1] * lt2, ci.normal[2] * ln + ci.t1[2] * lt1 + ci.t2[2] * lt2];
      ci.frictionUse = ln > 1e-9 ? Math.hypot(lt1, lt2) / (ci.mu * ln) : 0;
      const pv = this.kin.pointVelocity(ci.body, ci.point);
      if (ci.otherBody >= 0) { const pb = this.kin.pointVelocity(ci.otherBody, ci.point); pv[0] -= pb[0]; pv[1] -= pb[1]; pv[2] -= pb[2]; }
      const vn = pv[0] * ci.normal[0] + pv[1] * ci.normal[1] + pv[2] * ci.normal[2];
      ci.slip = Math.hypot(pv[0] - ci.normal[0] * vn, pv[1] - ci.normal[1] * vn, pv[2] - ci.normal[2] * vn);
    }
    this.frictionTorque.fill(0);
    this.limitHits = [];
    for (let r = 0; r < nr; r++) {
      const R = rows[r];
      if (R.kind === 'friction') this.frictionTorque[R.dof] = lambda[r] / h;
      else if (R.kind === 'limit' && lambda[r] > 0) this.limitHits.push({ kind: 'limit', dof: R.dof, side: R.side ? 'hi' : 'lo' });
    }
  }
}

function tangentBasis(n) {
  const t1 = Math.abs(n[2]) < 0.9 ? [n[1], -n[0], 0] : [0, n[2], -n[1]];   // n × ẑ or n × x̂ (unnormalised)
  const l = Math.hypot(t1[0], t1[1], t1[2]); t1[0] /= l; t1[1] /= l; t1[2] /= l;
  const t2 = [n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]];
  return [t1, t2];
}

// Minimiser of ½xᵀAx + xᵀe over the disk ‖x‖ ≤ rad, A = [[a,b],[b,d]] symmetric positive definite.
export function coulombBlock(a, b, d, e1, e2, rad, out = [0, 0]) {
  if (rad <= 0) { out[0] = 0; out[1] = 0; return out; }
  const det = a * d - b * b;
  const x1 = -(d * e1 - b * e2) / det, x2 = -(a * e2 - b * e1) / det;
  if (x1 * x1 + x2 * x2 <= rad * rad) { out[0] = x1; out[1] = x2; return out; }
  // eigen-decomposition of A
  const tr = a + d, disc = Math.sqrt(Math.max(0, 0.25 * (a - d) * (a - d) + b * b));
  const l1 = 0.5 * tr + disc, l2 = 0.5 * tr - disc;
  let c = 1, s = 0;
  if (Math.abs(b) > 1e-300) { const vx = l1 - d, vy = b, nn = Math.hypot(vx, vy); c = vx / nn; s = vy / nn; } else if (d > a) { c = 0; s = 1; }
  const b1 = c * e1 + s * e2, b2 = -s * e1 + c * e2;       // e in the eigenbasis (u1 = [c,s], u2 = [−s,c])
  // φ(ν) = b1²/(l1+ν)² + b2²/(l2+ν)² − rad² is convex and decreasing for ν > −l2; Newton on 1/‖x‖ is robust.
  let nu = Math.max(0, Math.hypot(b1, b2) / rad - l2);
  for (let k = 0; k < 60; k++) {
    const p1 = b1 / (l1 + nu), p2 = b2 / (l2 + nu);
    const nrm = Math.hypot(p1, p2);
    const dn = -((p1 * p1) / (l1 + nu) + (p2 * p2) / (l2 + nu)) / nrm;   // d‖x‖/dν
    const f = 1 / nrm - 1 / rad, df = -dn / (nrm * nrm);
    const step = f / df;
    nu = Math.max(0, nu - step);
    if (Math.abs(step) < 1e-14 * (1 + nu)) break;
  }
  const y1 = -b1 / (l1 + nu), y2 = -b2 / (l2 + nu);
  out[0] = c * y1 - s * y2; out[1] = s * y1 + c * y2;
  return out;
}
