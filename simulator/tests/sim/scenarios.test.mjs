// Scenario regression: every scenario builds and runs, and the behaviours keep the stability and slip performance
// measured when the controller was validated (2026-09-15). Thresholds leave headroom above the measured values.
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadRobot, runScenario, SCENARIOS } from '../../tools/analysis/metrics.mjs';

const robot = loadRobot();

test('every scenario builds and runs two seconds without errors or divergence', () => {
  for (const key of Object.keys(SCENARIOS)) {
    const m = runScenario(robot, key, { duration: 2 });
    assert.equal(m.error, null, `${key}: ${m.error}`);
  }
});

test('walk: stays on ≥ 3 feet with positive static margin, little slip, and covers the commanded distance', () => {
  const m = runScenario(robot, 'walk', { duration: 20 });
  assert.equal(m.error, null);
  assert.ok(m.minLoadedFeet >= 3, `feet ${m.minLoadedFeet}`);
  assert.ok(m.minSsm > 0.01, `SSM ${m.minSsm}`);
  assert.ok(m.slipFraction5 < 0.04, `slip>5 mm/s ${m.slipFraction5}`);
  assert.ok(m.maxSlip < 0.09, `max slip ${m.maxSlip}`);
  assert.ok(m.distance > 0.25 && m.distance < 0.4, `distance ${m.distance}`);
  assert.equal(m.shellHits.length, 0);
  assert.ok(m.maxIkError < 0.002, `IK ${m.maxIkError}`);
});

test('turn in place: rotates without reach errors or significant slip', () => {
  const m = runScenario(robot, 'turn', { duration: 16 });
  assert.equal(m.error, null);
  assert.ok(Math.abs(m.yaw) > 0.7, `yaw ${m.yaw}`);
  assert.ok(m.maxIkError < 0.005, `IK ${m.maxIkError}`);
  assert.ok(m.slip30 < 15, `slip samples ${m.slip30}`);
  assert.ok(m.minSsm > 0, `SSM ${m.minSsm}`);
});

test('step up 20 mm: climbs onto the block without tipping or touching the shell', () => {
  const m = runScenario(robot, 'step', { duration: 40 });
  assert.equal(m.error, null);
  assert.ok(m.sim.world.q[2] > 0.155, `body height ${m.sim.world.q[2]}`);
  assert.ok(m.minSsm > 0, `SSM ${m.minSsm}`);
  assert.equal(m.shellHits.length, 0);
  assert.ok(m.maxSlip < 0.12, `max slip ${m.maxSlip}`);
});

test('fist bump: four feet carry the robot while limb 1 is raised', () => {
  const m = runScenario(robot, 'fistbump', { duration: 8 });
  assert.ok(m.minLoadedFeet >= 4, `feet ${m.minLoadedFeet}`);
  assert.ok(m.minSsm > 0.03, `SSM ${m.minSsm}`);
  assert.ok(m.maxIkError < 0.001, `IK ${m.maxIkError}`);
});

test('servo failure: detected within 0.2 s, weight moved off the limb, no collapse', () => {
  const m = runScenario(robot, 'servofail', { duration: 10 });
  const failure = m.events.find((e) => e.kind === 'failure'), fault = m.events.find((e) => e.kind === 'fault');
  assert.ok(failure && fault && fault.t - failure.t <= 0.2, JSON.stringify(m.events));
  assert.ok(m.minSsm > 0.02, `SSM ${m.minSsm}`);
  assert.equal(m.shellHits.length, 0);
});

test('canon mass on hobby servos collapses (overload protection trips, carapace on the floor)', () => {
  const m = runScenario(robot, 'canon', { duration: 6 });
  assert.ok(m.shellHits.includes('carapace'));
  assert.ok(m.servos.some((s) => s.trips.includes('overload')));
});

test('drive: keyboard commands walk the robot and queued actions run once the feet are down', () => {
  const script = [[0.5, { vx: 0.02 }], [6, { vx: 0 }], [6.2, ['fistbump']]];
  let k = 0;
  const behaviours = new Set();
  const m = runScenario(robot, 'drive', {
    duration: 20,
    drive: (sim, t) => {
      while (k < script.length && t >= script[k][0]) { const s = script[k++][1]; if (Array.isArray(s)) sim.action(s[0], s[1] ?? {}); else sim.command(s); }
      behaviours.add(sim.controller.behavior);
    },
  });
  assert.equal(m.error, null);
  assert.ok(m.distance > 0.04, `distance ${m.distance}`);
  assert.ok(behaviours.has('fistbump'), [...behaviours].join(','));
  assert.equal(m.sim.controller.teleop.queue.length, 0);
});
