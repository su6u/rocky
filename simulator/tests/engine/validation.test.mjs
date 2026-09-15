// Analytic validation of the dynamics engine: every check compares the simulator against a closed-form result.
import test from 'node:test';
import assert from 'node:assert/strict';
import { ArticulatedModel } from '../../src/engine/dynamics.mjs';
import { World } from '../../src/engine/world.mjs';
import { Terrain } from '../../src/engine/terrain.mjs';
import { boxInertia, sphereInertia, I3 } from '../../src/engine/inertia.mjs';
import { axisAngleMat, matToQuat } from '../../src/engine/linalg.mjs';
import { Servo } from '../../src/hardware/servo.mjs';
import { Battery } from '../../src/hardware/battery.mjs';
import { SERVO_PRESETS } from '../../src/hardware/presets.mjs';

const g = 9.81;
const ball = (m, r) => new ArticulatedModel({ bodies: [{ name: 'ball', parent: -1, mass: m, com: [0, 0, 0], Icom: sphereInertia(m, r) }] });

test('free fall matches z0 − ½gt² within the semi-implicit Euler bound', () => {
  const w = new World(ball(1, 0.05), { dt: 0.0005 });
  w.q.set([0, 0, 1.0, 1, 0, 0, 0]);
  w.contactShapes.push({ body: 0, local: [0, 0, 0], radius: 0.05, kind: 'shell' });
  const T = 0.3;
  while (w.time < T - 1e-12) w.step();
  const exact = 1.0 - 0.5 * g * T * T, bound = 0.5 * g * w.dt * T + 1e-9;
  assert.ok(Math.abs(w.q[2] - exact) < bound, `z=${w.q[2]} exact=${exact} bound=${bound}`);
});

test('Hertz pad equilibrium penetration δ = (mg/k)^(2/3)', () => {
  const m = 2.0, k = 4.0e6;
  const w = new World(ball(m, 0.03), { dt: 0.001, pad: { k, alpha: 2.0, margin: 0.004 } });
  w.q.set([0, 0, 0.03, 1, 0, 0, 0]);
  w.contactShapes.push({ body: 0, local: [0, 0, 0], radius: 0.03, kind: 'foot' });
  for (let i = 0; i < 4000; i++) w.step();
  const pen = 0.03 - w.q[2], exact = Math.pow((m * g) / k, 2 / 3), fn = w.contacts[0]?.fn ?? 0;
  assert.ok(Math.abs(pen - exact) / exact < 0.02, `δ=${pen * 1e6} µm exact=${exact * 1e6} µm`);
  assert.ok(Math.abs(fn - m * g) / (m * g) < 0.005, `Fn=${fn} N mg=${m * g} N`);
});

function inclineBlock(deg, mu) {
  const m = 1.5;
  const model = new ArticulatedModel({ bodies: [{ name: 'block', parent: -1, mass: m, com: [0, 0, 0], Icom: boxInertia(m, 0.2, 0.2, 0.05) }] });
  const w = new World(model, { dt: 0.001, terrain: new Terrain({ slopeDeg: deg, mu }), pad: { k: 5e5, alpha: 3, margin: 0.004 } });
  const th = (deg * Math.PI) / 180, nrm = [-Math.sin(th), 0, Math.cos(th)];
  const q = matToQuat(axisAngleMat([0, 1, 0], -th));
  const h0 = 0.0201;
  w.q.set([nrm[0] * h0, nrm[1] * h0, nrm[2] * h0, q[0], q[1], q[2], q[3]]);
  for (const [x, y] of [[0.08, 0.08], [0.08, -0.08], [-0.08, 0.08], [-0.08, -0.08]]) w.contactShapes.push({ body: 0, local: [x, y, -0.005], radius: 0.015, kind: 'foot' });
  for (let i = 0; i < 400; i++) w.step();
  return { w, th };
}

test('Coulomb incline below atan μ: the block sticks', () => {
  const { w } = inclineBlock(25, 0.6);
  const x0 = w.q[0], z0 = w.q[2];
  for (let i = 0; i < 600; i++) w.step();
  const creep = Math.hypot(w.q[0] - x0, w.q[2] - z0);
  assert.ok(creep < 2e-4, `creep ${creep * 1000} mm`);
});

test('Coulomb incline above atan μ: slides with a = g(sinθ − μcosθ)', () => {
  const mu = 0.6, { w, th } = inclineBlock(40, mu);
  for (let i = 0; i < 600; i++) w.step();
  const speed = () => { w.kin.update(w.q, w.v); const R = w.kin.Rw[0]; return R[0] * w.v[3] + R[1] * w.v[4] + R[2] * w.v[5]; };
  const v1 = speed();
  for (let i = 0; i < 300; i++) w.step();
  const a = -(speed() - v1) / (300 * w.dt) / Math.cos(th), exact = g * (Math.sin(th) - mu * Math.cos(th));
  assert.ok(Math.abs(a - exact) / exact < 0.005, `a=${a} exact=${exact}`);
});

test('compound pendulum small-amplitude period, including reflected armature', () => {
  const m = 0.4, L = 0.3, d = L / 2, arm = 0.002, Ic = boxInertia(m, L, 0.02, 0.02);
  const model = new ArticulatedModel({ bodies: [
    { name: 'base', parent: -1, mass: 1, com: [0, 0, 0], Icom: I3() },
    { name: 'link', parent: 0, R0: I3(), r0: [0, 0, 0], axis: [0, 1, 0], mass: m, com: [d, 0, 0], Icom: Ic, armature: arm },
  ] });
  const w = new World(model, { dt: 0.0005, fixedBase: true });
  w.q[7] = Math.PI / 2 + 0.05;
  let last = null, prev = w.q[7] - Math.PI / 2;
  const periods = [];
  for (let i = 0; i < 20000; i++) {
    w.step();
    const cur = w.q[7] - Math.PI / 2;
    if (prev < 0 && cur >= 0) { const tc = w.time - (w.dt * cur) / (cur - prev); if (last !== null) periods.push(tc - last); last = tc; }
    prev = cur;
  }
  const Tm = periods.reduce((a, b) => a + b, 0) / periods.length;
  const exact = 2 * Math.PI * Math.sqrt((Ic[4] + m * d * d + arm) / (m * g * d));
  assert.ok(Math.abs(Tm - exact) / exact < 0.002, `T=${Tm} exact=${exact}`);
});

test('double pendulum energy error converges at first order in h', () => {
  const m1 = 0.5, m2 = 0.3, L = 0.25;
  const run = (h) => {
    const model = new ArticulatedModel({ bodies: [
      { name: 'base', parent: -1, mass: 1, com: [0, 0, 0], Icom: I3() },
      { name: 'l1', parent: 0, R0: I3(), r0: [0, 0, 0], axis: [0, 1, 0], mass: m1, com: [L / 2, 0, 0], Icom: boxInertia(m1, L, 0.02, 0.02) },
      { name: 'l2', parent: 1, R0: I3(), r0: [L, 0, 0], axis: [0, 1, 0], mass: m2, com: [L / 2, 0, 0], Icom: boxInertia(m2, L, 0.02, 0.02) },
    ] });
    const w = new World(model, { dt: h, fixedBase: true });
    w.q[7] = 0.3; w.q[8] = 1.1;
    const energy = () => {
      w.kin.update(w.q, w.v);
      let V = 0;
      for (let i = 0; i < model.nb; i++) V += model.bodies[i].mass * g * w.kin.pointWorld(i, model.bodies[i].com)[2];
      return w.kin.momentum().T + V;
    };
    const E0 = energy();
    let maxDev = 0;
    for (let i = 0, N = Math.round(2.0 / h); i < N; i++) { w.step(); maxDev = Math.max(maxDev, Math.abs(energy() - E0)); }
    return maxDev;
  };
  const e1 = run(4e-4), e2 = run(2e-4), e3 = run(1e-4), scale = (m1 + m2) * g * L;
  assert.ok(Math.log2(e1 / e2) > 0.8 && Math.log2(e2 / e3) > 0.8, `orders ${Math.log2(e1 / e2)}, ${Math.log2(e2 / e3)}`);
  assert.ok(e3 / scale < 3e-3, `ΔE at 0.1 ms = ${(100 * e3) / scale} %`);
});

test('free-floating momentum drift converges at first order in h', () => {
  const run = (h) => {
    const model = new ArticulatedModel({ gravity: [0, 0, 0], bodies: [
      { name: 'base', parent: -1, mass: 2, com: [0, 0, 0], Icom: boxInertia(2, 0.3, 0.3, 0.1) },
      { name: 'a', parent: 0, R0: I3(), r0: [0.15, 0, 0], axis: [0, 0, 1], mass: 0.3, com: [0.1, 0, 0], Icom: boxInertia(0.3, 0.2, 0.03, 0.03) },
      { name: 'b', parent: 1, R0: I3(), r0: [0.2, 0, 0], axis: [0, 1, 0], mass: 0.2, com: [0.1, 0, 0], Icom: boxInertia(0.2, 0.2, 0.03, 0.03) },
    ] });
    const w = new World(model, { dt: h });
    w.v.set([0.1, -0.2, 0.3, 0.05, 0.02, -0.01, 1.5, -2.0]);
    w.kin.update(w.q, w.v);
    const m0 = w.kin.momentum();
    for (let i = 0, N = Math.round(1.0 / h); i < N; i++) w.step();
    w.kin.update(w.q, w.v);
    const m1 = w.kin.momentum();
    return [Math.hypot(...m1.P.map((x, i) => x - m0.P[i])) / Math.hypot(...m0.P), Math.hypot(...m1.L.map((x, i) => x - m0.L[i])) / Math.hypot(...m0.L)];
  };
  const [p1, l1] = run(1e-3), [p2, l2] = run(5e-4), [p3, l3] = run(2.5e-4);
  assert.ok(Math.log2(p1 / p2) > 0.8 && Math.log2(l2 / l3) > 0.8, `P ${p1} ${p2}; L ${l1} ${l2} ${l3}`);
  assert.ok(p3 < 2e-3 && l3 < 1e-3, `P ${p3} L ${l3}`);
});

function servoBench({ lockAt = null } = {}) {
  const p = { ...SERVO_PRESETS.sts3215_7v4, friction: { model: 'm1', base: 0, viscous: 0 }, maxVelocity: 1e3, commandDelay: 0 };
  const model = new ArticulatedModel({ gravity: [0, 0, 0], bodies: [
    { name: 'base', parent: -1, mass: 1, com: [0, 0, 0], Icom: I3() },
    { name: 'horn', parent: 0, R0: I3(), r0: [0, 0, 0], axis: [0, 0, 1], mass: 0.01, com: [0, 0, 0], Icom: sphereInertia(0.01, 0.01), armature: p.armature },
  ] });
  const w = new World(model, { dt: 0.0005, fixedBase: true });
  w.battery = new Battery({ cells: 2, rInternal: 0 });
  w.battery.ocv = () => 7.4; w.battery.update(0, 0);
  w.quiescentCurrent = 0;
  const s = new Servo(p, { protection: false });
  s.command(1000, 0);
  w.actuators.push({ servo: s, dof: 6, backlashDof: -1 });
  if (lockAt !== null) w.limits.push({ dof: 6, lo: -10, hi: lockAt, k: 1e4, zeta: 1, inertia: p.armature });
  return { w, s, p, U: p.dutyMax * 7.4 };
}

test('servo no-load speed ω = duty·V/k_t (ideal, frictionless)', () => {
  const { w, p, U } = servoBench();
  for (let i = 0; i < 6000; i++) w.step();
  const exact = U / p.kt;
  assert.ok(Math.abs(w.v[6] - exact) / exact < 0.002, `ω=${w.v[6]} exact=${exact}`);
});

test('servo stall torque τ = k_t·duty·V/R(T) at the winding temperature', () => {
  const { w, s, p, U } = servoBench({ lockAt: 0.1 });
  for (let i = 0; i < 4000; i++) w.step();
  const exact = (p.kt * U) / s.resistance();
  assert.ok(Math.abs(s.tauMotor - exact) / exact < 0.002, `τ=${s.tauMotor} exact=${exact}`);
});
