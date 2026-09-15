// Headless scenario runner and metric collector shared by the analysis CLIs and the regression tests.
// Runs the same engine, controller and servo models as the browser, without rendering.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createSimulation, measure, SCENARIOS } from '../../src/sim/simulation.mjs';

export { SCENARIOS };

export function loadRobot() {
  return JSON.parse(readFileSync(fileURLToPath(new URL('../../assets/robot.json', import.meta.url)), 'utf8'));
}

const quantile = (values, q) => {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))];
};

// Run one scenario for `duration` seconds (default: the scenario's own, capped by `maxDuration`).
// `drive(sim, t)` is called every physics step for scripted teleoperation.
export function runScenario(robot, key, { duration, maxDuration = 60, sim: simOptions = {}, drive = null, sampleEvery = 5 } = {}) {
  const spec = SCENARIOS[key];
  if (!spec) throw new Error(`unknown scenario ${key}`);
  const T = duration ?? Math.min(spec.duration ?? maxDuration, maxDuration);
  const t0 = performance.now();
  const sim = createSimulation(robot, { scenario: key, ...simOptions });
  const { world, meta, controller: ctrl } = sim;
  const W = meta.totalMass * 9.81, settle = (spec.settle ?? 0) + 0.5;
  const m = {
    scenario: key, label: spec.label, duration: T, totalMass: meta.totalMass, servo: meta.servo.label, steps: 0, iterations: 0,
    minLoadedFeet: 9, minSsm: Infinity, minCaptureMargin: Infinity, loadedSamples: 0, slip5: 0, slip30: 0, maxSlip: 0,
    shellHits: new Set(), selfHits: new Set(), maxIkError: 0, minBusVoltage: Infinity, touchdowns: 0, error: null,
  };
  const joints = new Map(world.actuators.map((a) => [a, { name: `L${a.limb} ${a.joint}`, limb: a.limb, joint: a.joint, taus: [], peak: 0, peakCurrent: 0, peakCase: 0, peakWinding: 0, speedPeak: 0 }]));
  const start = { x: world.q[0], y: world.q[1] };
  const seenTouchdown = new Map();
  try {
    while (world.time < T - 1e-9) {
      if (drive) drive(sim, world.time);
      sim.step();
      m.steps++; m.iterations += world.stats.iterations;
      if (!Number.isFinite(world.q[0] + world.q[1] + world.q[2] + world.q[3])) throw new Error('numerical divergence');
      for (const [a, j] of joints) {
        const s = a.servo;
        j.peak = Math.max(j.peak, Math.abs(s.tauMotor)); j.peakCurrent = Math.max(j.peakCurrent, Math.abs(s.I));
        j.peakCase = Math.max(j.peakCase, s.T); j.peakWinding = Math.max(j.peakWinding, s.Tw);
        j.speedPeak = Math.max(j.speedPeak, Math.abs(world.v[a.dof]));
      }
      if (m.steps % sampleEvery) continue;
      for (const [a, j] of joints) j.taus.push(Math.abs(a.servo.tauMotor));
      for (const c of world.contacts) {
        if (c.kind === 'foot' && c.fn > 0.5) {
          m.loadedSamples++;
          if (c.slip > 0.005) m.slip5++;
          if (c.slip > 0.03) m.slip30++;
          m.maxSlip = Math.max(m.maxSlip, c.slip);
        } else if (c.kind === 'shell' && c.fn > 0.05 * W) m.shellHits.add(c.name);
        else if (c.kind === 'self' && c.fn > 0.02 * W) m.selfHits.add(`${c.name}–${c.other}`);
      }
      for (const leg of ctrl.legs) {
        m.maxIkError = Math.max(m.maxIkError, leg.ikError ?? 0);
        if (leg.lastTouchdown && seenTouchdown.get(leg) !== leg.lastTouchdown) { seenTouchdown.set(leg, leg.lastTouchdown); m.touchdowns++; }
      }
      if (m.steps % (sampleEvery * 4) === 0) {
        const r = measure(sim);
        m.minBusVoltage = Math.min(m.minBusVoltage, r.battery.v);
        if (world.time > settle) {
          m.minLoadedFeet = Math.min(m.minLoadedFeet, r.stab.loadedFeet);
          if (Number.isFinite(r.stab.ssm)) m.minSsm = Math.min(m.minSsm, r.stab.ssm);
          if (Number.isFinite(r.stab.captureMargin)) m.minCaptureMargin = Math.min(m.minCaptureMargin, r.stab.captureMargin);
        }
      }
    }
  } catch (err) {
    m.error = `${err.message} at t = ${world.time.toFixed(3)} s`;
  }
  const rated = meta.servo.ratedTorqueDatasheet, stall = meta.servo.stallTorqueDatasheet;
  m.msPerStep = (performance.now() - t0) / Math.max(1, m.steps);
  m.realTimeFactor = world.time / ((performance.now() - t0) / 1000);
  m.meanIterations = m.iterations / Math.max(1, m.steps);
  m.slipFraction5 = m.loadedSamples ? m.slip5 / m.loadedSamples : 0;
  m.shellHits = [...m.shellHits]; m.selfHits = [...m.selfHits];
  m.distance = Math.hypot(world.q[0] - start.x, world.q[1] - start.y);
  m.yaw = 2 * Math.atan2(world.q[6], world.q[3]);
  m.battery = { minVoltage: m.minBusVoltage, soc: world.battery.soc, energyJ: world.battery.energyJ };
  m.servos = [...joints.entries()].map(([a, j]) => ({
    name: j.name, limb: j.limb, joint: j.joint, peak: j.peak, rms: a.servo.rmsTorque(), p95: quantile(j.taus, 0.95),
    peakCurrent: j.peakCurrent, peakCase: j.peakCase, peakWinding: j.peakWinding, speedPeak: j.speedPeak,
    timeAboveRated: a.servo.timeAboveRated, energyJ: a.servo.energyElec, trips: a.servo.trips.map((t) => t.kind),
    ratedShareRms: a.servo.rmsTorque() / rated, stallSharePeak: j.peak / stall,
  }));
  m.byJoint = Object.fromEntries(['yaw', 'pitch', 'elbow'].map((jn) => {
    const list = m.servos.filter((s) => s.joint === jn);
    return [jn, { peak: Math.max(...list.map((s) => s.peak)), rms: Math.max(...list.map((s) => s.rms)), p95: Math.max(...list.map((s) => s.p95)), peakCase: Math.max(...list.map((s) => s.peakCase)), timeAboveRated: Math.max(...list.map((s) => s.timeAboveRated)) }];
  }));
  m.events = sim.events.map((e) => ({ t: +e.t.toFixed(3), kind: e.kind, text: e.text }));
  m.eventCounts = m.events.reduce((acc, e) => ((acc[e.kind] = (acc[e.kind] ?? 0) + 1), acc), {});
  delete m.slip5; delete m.iterations; delete m.minBusVoltage;
  for (const k of ['minSsm', 'minCaptureMargin']) if (!Number.isFinite(m[k])) m[k] = null;
  m.sim = sim;
  return m;
}

export function formatRow(m) {
  const mm = (x) => (x == null ? '—' : (x * 1000).toFixed(0));
  const j = m.byJoint;
  return [
    m.scenario.padEnd(10), `${m.duration}s`.padStart(5), `${m.realTimeFactor.toFixed(1)}×`.padStart(6),
    `feet≥${m.minLoadedFeet === 9 ? '—' : m.minLoadedFeet}`, `SSM ${mm(m.minSsm)}mm`.padEnd(11),
    `slip>5 ${(100 * m.slipFraction5).toFixed(1)}%`.padEnd(13), `>30 ${m.slip30}`.padEnd(7), `max ${mm(m.maxSlip)}mm/s`.padEnd(13),
    `τpk y${j.yaw.peak.toFixed(2)} p${j.pitch.peak.toFixed(2)} e${j.elbow.peak.toFixed(2)}`,
    `rms p${j.pitch.rms.toFixed(2)} e${j.elbow.rms.toFixed(2)}`, `Tc ${Math.max(j.yaw.peakCase, j.pitch.peakCase, j.elbow.peakCase).toFixed(1)}°C`,
    `V ${m.battery.minVoltage.toFixed(2)}`, `ik ${mm(m.maxIkError)}mm`, `d ${mm(m.distance)}mm`,
    m.shellHits.length ? `shell[${m.shellHits}]` : '', m.selfHits.length ? `self[${m.selfHits}]` : '',
    Object.keys(m.eventCounts).length ? JSON.stringify(m.eventCounts) : '', m.error ? `ERROR ${m.error}` : '',
  ].filter(Boolean).join('  ');
}
