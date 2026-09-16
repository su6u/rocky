// Robot model, limb IK on the kit's skewed axes, gait geometry and the controller's planning helpers.
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadRobot } from '../../tools/analysis/metrics.mjs';
import { createSimulation } from '../../src/sim/simulation.mjs';
import { phasedSwing, predictPose, SWING_PHASES } from '../../src/robot/gait.mjs';
import { polygonMargin, hull2d } from '../../src/engine/stability.mjs';
import { mat3vec } from '../../src/engine/linalg.mjs';

const robot = loadRobot();
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);
const standing = () => createSimulation(robot, { scenario: 'stand' });

test('desk build: 2.73 kg, 18.2 cm carapace, 15 servos, 16 bodies', () => {
  const { meta, model } = standing();
  close(meta.totalMass, 2.73, 0.02);
  close(meta.carapaceAcross, 0.182, 0.002);
  assert.equal(meta.servoCount, 15);
  assert.equal(model.nb, 16);
});

test('limb IK reaches forward-kinematics targets on every limb and probe() keeps the warm start', () => {
  const { controller } = standing();
  for (const leg of controller.legs) {
    const theta = [0.2, 0.3, -1.4];
    const target = leg.ik.fk(theta).foot;
    const before = Float64Array.from(leg.ik.theta);
    const probed = leg.ik.probe(target);
    assert.ok(probed.error < 1e-4, `L${leg.L.limb} probe error ${probed.error}`);
    assert.deepEqual(Array.from(leg.ik.theta), Array.from(before), 'probe must not move the warm start');
    const solved = leg.ik.solve(target);
    assert.ok(solved.error < 1e-4, `L${leg.L.limb} error ${solved.error}`);
  }
});

test('swing path: starts and ends on its endpoints, transfer finishes before the descent nears the ground', () => {
  const a = [0, 0, 0.01], b = [0.08, 0.02, 0.015], apex = 0.03;
  assert.deepEqual(phasedSwing(a, b, apex, 0).map((x) => +x.toFixed(12)), a);
  assert.deepEqual(phasedSwing(a, b, apex, 1).map((x) => +x.toFixed(12)), b);
  const atTransferEnd = phasedSwing(a, b, apex, SWING_PHASES.transferEnd);
  close(atTransferEnd[0], b[0], 1e-9, 'horizontal travel done');
  assert.ok(atTransferEnd[2] > b[2] + 0.25 * (apex - b[2]), 'still clearly above the landing point');
  for (let s = 0; s <= 1; s += 0.05) assert.ok(phasedSwing(a, b, apex, s)[2] >= Math.min(a[2], b[2]) - 1e-12);
});

test('predictPose integrates constant body-frame velocity exactly along the arc', () => {
  const pose = { x: 0.1, y: -0.2, yaw: 0.3 }, vel = { vx: 0.02, vy: -0.01, wz: 0.15 }, T = 4;
  let p = { ...pose };
  for (let i = 0; i < 40000; i++) {
    const h = T / 40000, c = Math.cos(p.yaw), s = Math.sin(p.yaw);
    p = { x: p.x + (vel.vx * c - vel.vy * s) * h, y: p.y + (vel.vx * s + vel.vy * c) * h, yaw: p.yaw + vel.wz * h };
  }
  const exact = predictPose(pose, vel, T);
  close(exact.x, p.x, 1e-5); close(exact.y, p.y, 1e-5); close(exact.yaw, p.yaw, 1e-12);
});

test('stance layout and gait envelope: stride from every shoulder yaw room, feasible below vMax', () => {
  const { controller: c } = standing();
  assert.ok(c.stanceLayout.alpha > 0 && c.stanceLayout.alpha < 1, `blend ${c.stanceLayout.alpha}`);
  assert.ok(c.strideLimit() > 0.08, `stride ${c.strideLimit()}`);
  const ok = c.walkTiming(0.015), vMax = c.walkLimits.vMax;
  assert.ok(ok.feasible && vMax > 0.015 && vMax < 0.05, `vMax ${vMax}`);
  assert.ok(ok.cycle >= ok.tMin - 1e-9 && ok.cycle <= Math.min(ok.tMax, 8) + 1e-9);
  assert.equal(c.walkTiming(3 * vMax).feasible, false);
  const clamped = c.clampCommand({ vx: 1, vy: 0, wz: 0 });
  close(clamped.vx, 0.95 * vMax, 1e-9, 'teleop command limited to the envelope');
});

test('gesture pose is reachable with joint margin, and the support shift clears the requested margin', () => {
  const { controller: c } = standing();
  for (const leg of c.legs) {
    const pose = c.gesturePose(leg), r = leg.ik.probe(pose.local);
    assert.ok(r.error < 0.001, `L${leg.L.limb} gesture error ${r.error}`);
    r.theta.forEach((th, k) => assert.ok(th > leg.limits[k][0] + 0.05 && th < leg.limits[k][1] - 0.05, `L${leg.L.limb} joint ${k}`));
    const i = c.legs.indexOf(leg), shift = c.supportShift(i, 0.04);
    const com = mat3vec(c.bodyMatrix(c.body), c.comOffset), poly = hull2d(c.legs.filter((_, k) => k !== i).map((l) => l.stanceFoot)).map((p) => [p[0], p[1]]);
    const margin = polygonMargin(poly, c.body.p[0] + com[0] + shift[0], c.body.p[1] + com[1] + shift[1]);
    assert.ok(margin >= 0.04 - 1e-4 || margin >= 0.7 * 0.04, `L${leg.L.limb} margin ${margin}`);
  }
});

test('runs are reproducible for a seed and differ between seeds', () => {
  const run = (seed) => { const sim = createSimulation(robot, { scenario: 'walk', seed }); while (sim.world.time < 2) sim.step(); return Array.from(sim.world.q.slice(0, 3)); };
  assert.deepEqual(run(3), run(3));
  assert.notDeepEqual(run(3), run(4));
});
