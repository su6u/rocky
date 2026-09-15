// Servo model: two-node thermal network, Feetech firmware protections, preset consistency with datasheets.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Servo, frictionBudget } from '../../src/hardware/servo.mjs';
import { SERVO_PRESETS } from '../../src/hardware/presets.mjs';
import { Battery } from '../../src/hardware/battery.mjs';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);
const P = SERVO_PRESETS.sts3215_7v4;

// Drive a servo's bookkeeping directly with a fixed duty and a locked output (no motion). Repeated calls continue
// from where the previous one stopped.
function holdAt(servo, { duty, seconds, vBus = 8.0, dt = 0.01 }) {
  const goal = duty / (servo.p.kp * servo.p.errorGain);
  if (servo.targetSmooth === null) servo.targetSmooth = goal;
  servo.targetCmd = goal;
  servo.clock ??= 0;
  for (let n = Math.round(seconds / dt), i = 0; i < n; i++, servo.clock += dt) {
    servo.computeDrive(servo.clock, dt, 0, 0, vBus);
    servo.finish(dt, 0, 0, 0, vBus);
  }
}

test('thermal network settles at T_c = T_amb + P·R_ca and T_w = T_c + P_cu·R_wc', () => {
  const s = new Servo(P, { protection: false, ambient: 25 });
  holdAt(s, { duty: 0.3, seconds: 4000, dt: 0.05 });
  const Pcu = s.copperLoss, Pq = 8.0 * P.thermal.quiescentCurrent, th = P.thermal;
  close(s.T, 25 + (Pcu + Pq) * th.Rca, 0.05, 'case');
  close(s.Tw, s.T + Pcu * th.Rwc, 0.05, 'winding');
  assert.ok(s.Tw > s.T && s.T > 25);
});

test('enclosure factor raises the case-to-ambient resistance', () => {
  const open = new Servo(P, { protection: false }), closed = new Servo(P, { protection: false, enclosure: 1.5 });
  holdAt(open, { duty: 0.3, seconds: 3000, dt: 0.05 });
  holdAt(closed, { duty: 0.3, seconds: 3000, dt: 0.05 });
  assert.ok(closed.T - 25 > 1.35 * (open.T - 25), `${closed.T} vs ${open.T}`);
});

test('overload protection: 80 % duty for 2 s limits output to 20 %, a new goal releases it', () => {
  const s = new Servo(P);
  holdAt(s, { duty: 0.9, seconds: 1.9 });
  assert.equal(s.protected, false, 'not before 2 s');
  holdAt(s, { duty: 0.9, seconds: 0.2 });
  assert.equal(s.protected, true, 'after 2 s of saturation');
  s.computeDrive(s.clock, 0.01, 0, 0, 8);
  close(Math.abs(s.duty), 0.2 * P.dutyMax, 1e-9, 'protective torque');
  s.command(s.protectGoal + 0.1, s.clock, 0); s.computeDrive(s.clock + 0.1, 0.01, 0, 0, 8);
  assert.equal(s.protected, false, 'released by a new goal');
});

test('overheat protection releases torque at the board limit and refuses to re-enable while hot', () => {
  const s = new Servo(P, { ambient: 25 });
  s.T = P.thermal.Tmax + 0.5; s.Tw = s.T + 10;
  s.finish(0.001, 0, 0, 0, 8);
  assert.equal(s.unload, 'overheat');
  s.computeDrive(0, 0.001, 0, 0, 8);
  assert.equal(s.duty, 0, 'no drive while unloaded');
  assert.equal(s.enableTorque(), false, 'still above the limit');
  s.T = P.thermal.Tmax - 12;
  assert.equal(s.enableTorque(), true);
  assert.equal(s.unload, null);
});

test('a failed servo keeps its gearbox friction budget (stiction at rest)', () => {
  const f = P.friction;
  const budget = frictionBudget(f, 0, 0, 0.05);
  assert.ok(budget > 0.05, `holds a 0.05 N·m load: ${budget}`);
  assert.ok(frictionBudget(f, 0, 0, 0.5) < 0.5, 'but a 0.5 N·m load back-drives it');
});

test('presets: required fields, and the 12 V STS3215 matches its datasheet', () => {
  for (const [key, p] of Object.entries(SERVO_PRESETS)) {
    for (const field of ['label', 'source', 'kt', 'R', 'armature', 'friction', 'thermal', 'stallTorqueDatasheet', 'ratedTorqueDatasheet', 'vMin']) assert.ok(p[field] !== undefined, `${key}.${field}`);
    for (const field of ['Rwc', 'Cw', 'Rca', 'Cc', 'Tmax']) assert.ok(p.thermal[field] > 0, `${key}.thermal.${field}`);
  }
  const p = SERVO_PRESETS.sts3215_12v, b = SERVO_PRESETS.sts3215_7v4;
  close((p.dutyMax * 12) / p.kt, (45 * 2 * Math.PI) / 60, 0.2, 'ideal no-load speed vs 45 rpm (friction lowers the real one)');
  close(((p.kt * 12) / p.R) / ((b.kt * 7.4) / b.R), 30 / 19.5, 1e-9, 'stall torque ratio to the 7.4 V model = datasheet ratio');
});

test('battery: open-circuit voltage falls with charge and the bus sags under current', () => {
  const full = new Battery({ cells: 2, soc: 1 }), low = new Battery({ cells: 2, soc: 0.1 });
  assert.ok(full.ocv() > low.ocv());
  close(full.ocv(), 8.4, 1e-9);
  const ocv = full.ocv();
  full.update(5, 0.001);
  close(full.voltage, ocv - 5 * full.rInternal, 1e-9);
});
