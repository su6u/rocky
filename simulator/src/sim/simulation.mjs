// Scenario assembly: physical Rocky robot + world + electrical system + controller, plus failure cases.
import { buildRocky } from '../robot/model.mjs';
import { RockyController } from '../robot/controller.mjs';
import { World } from '../engine/world.mjs';
import { Terrain } from '../engine/terrain.mjs';
import { stabilityReport } from '../engine/stability.mjs';
import { Servo } from '../hardware/servo.mjs';
import { Battery } from '../hardware/battery.mjs';
import { SERVO_PRESETS } from '../hardware/presets.mjs';
import { MATERIALS, hertzK } from '../hardware/materials.mjs';
import { SCENARIOS } from './scenarios.mjs';

const CONTROLLER_POWER_W = 2.5;    // single-board computer, IMU, bus adapter and foot-sensor ADC (estimate)
const SHELL_SOFTENING = 0.02;      // collision spheres are coarse stand-ins for thin printed shells

// Environment overrides from the viewer ({mu, slope, speed, soc, ambient}); unset keys keep the scenario's own values.
function withEnv(spec, env) {
  const S = { ...spec, params: { ...(spec.params || {}) }, battery: spec.battery ? { ...spec.battery } : undefined };
  if (env.mu != null) S.mu = env.mu;
  if (env.slope != null) S.slope = env.slope;
  if (env.speed != null && S.params.speed != null) S.params.speed = env.speed;
  if (env.soc != null) S.battery = { ...(S.battery || {}), soc: env.soc };
  if (env.ambient != null) S.ambient = env.ambient;
  return S;
}

export function createSimulation(robotJson, { scenario = 'stand', design = {}, dt = 0.001, backlash = false, controller = {}, servoPreset, env = {}, seed = 1, selfCollision = true } = {}) {
  const S = withEnv(SCENARIOS[scenario], env);
  const d = { ...design, ...(S.design || {}) };
  if (d.scale === 'canon') d.scale = robotJson.canon.printToCanonScale / 1000;
  if (backlash) d.backlash = true;
  if (servoPreset) d.servo = servoPreset;
  if (scenario === 'fistbump') d.hand = '1-B';
  const { model, meta } = buildRocky(robotJson, d);
  const s = meta.design.scale, ls = s / 0.0045;
  const scaled = (list = []) => list.map((b) => ({ ...b, min: b.min.map((x) => x * ls), max: b.max.map((x) => x * ls) }));
  const terrain = new Terrain({ slopeDeg: S.slope ?? 0, mu: S.mu ?? 0.85, restitutionAlpha: 1.2, boxes: scaled(S.environment?.boxes), patches: scaled(S.environment?.patches) });
  const world = new World(model, { dt, terrain, pad: { k: 4e6, alpha: 1.2, margin: 0.004 * ls }, iterations: 60, tolerance: 1e-7 });
  // electrical system: the pack matches the servo voltage class (2S for 7.4 V servos, 3S for 12 V)
  const cells = meta.servo.cells ?? 2;
  world.battery = new Battery({ cells, capacityAh: meta.design.battery.capacityAh, rInternal: meta.design.battery.rInternal * (cells / 2), soc: 0.95, ...(S.battery || {}) });
  world.quiescentCurrent = CONTROLLER_POWER_W / (cells * 3.7);
  // actuators, limits
  const ambient = S.ambient ?? 25;
  for (let i = 1; i < model.nb; i++) {
    const b = model.bodies[i];
    if (b.servo) {
      const backlashBody = model.bodies.findIndex((bb) => bb.backlashOf === i);
      // yaw servos sit inside the closed carapace: poorer convection (×1.5 case-to-ambient resistance)
      const servo = new Servo(meta.servo, { name: b.name, ambient, enclosure: b.joint === 'yaw' ? 1.5 : 1 });
      world.actuators.push({ servo, dof: model.dof[i], backlashDof: backlashBody > 0 ? model.dof[backlashBody] : -1, body: i, limb: b.limb, joint: b.joint });
      world.limits.push({ dof: model.dof[i], lo: b.lo, hi: b.hi, k: 200 * ls ** 3, zeta: 0.7, inertia: meta.servo.armature });
    }
    if (b.backlashOf) {
      const half = 0.5 * meta.servo.backlashRad;
      world.limits.push({ dof: model.dof[i], lo: -half, hi: half, k: 300 * ls ** 3, zeta: 0.3, inertia: 1e-4 });
    }
  }
  // contact shapes: feet (TPU pad or PLA), limb shells, carapace. group = limb number (0 = carapace) for self-collision.
  const footMat = MATERIALS[S.footMaterial ?? 'tpu95a'], pla = MATERIALS.pla;
  for (const L of meta.limbs) {
    world.contactShapes.push({ body: L.footBody, local: L.foot.local, radius: L.foot.radius, kind: 'foot', name: `foot${L.limb}`, limb: L.limb, group: L.limb, padK: hertzK(footMat.E, footMat.nu, L.foot.radius) });
    for (const c of L.collisionA) world.contactShapes.push({ body: L.meshHolderPitch, local: c.center, radius: c.radius, kind: 'shell', name: `A${L.limb}`, limb: L.limb, group: L.limb, segment: 'A', padK: hertzK(pla.E, pla.nu, c.radius) * SHELL_SOFTENING });
    for (const c of L.collisionB.slice(0, 3)) world.contactShapes.push({ body: L.footBody, local: c.center, radius: c.radius, kind: 'shell', name: `B${L.limb}`, limb: L.limb, group: L.limb, segment: 'B', padK: hertzK(pla.E, pla.nu, c.radius) * SHELL_SOFTENING });
  }
  for (const c of robotJson.torso.collision) world.contactShapes.push({ body: 0, local: c.center.map((x) => x * s), radius: c.radius * s, kind: 'shell', name: 'carapace', group: 0, padK: hertzK(pla.E, pla.nu, c.radius * s) * SHELL_SOFTENING });
  const ctrl = new RockyController(world, meta, { seed, ...(S.controller || {}), ...controller });
  if (S.start === 'lying') {
    ctrl.initialPose(world);
    const lowest = Math.min(...robotJson.torso.collision.map((c) => (c.center[2] - c.radius) * s));
    const rest = -lowest + 0.002 * ls;
    world.q[2] = rest; ctrl.body.p[2] = rest;
    for (const leg of ctrl.legs) { leg.stanceFoot = [leg.stanceFoot[0], leg.stanceFoot[1], ctrl.footZ(leg.stanceFoot[0], leg.stanceFoot[1], leg.L.foot.radius)]; ctrl.solveLeg(leg, leg.stanceFoot, ctrl.body); }
    for (const leg of ctrl.legs) { world.q[7 + leg.L.yaw - 1] = leg.angles[0]; world.q[7 + leg.L.pitch - 1] = leg.angles[1]; world.q[7 + leg.L.elbow - 1] = leg.angles[2]; }
    for (const a of world.actuators) { a.servo.targetSmooth = null; a.servo.targetCmd = world.q[a.dof + 1]; }
    ctrl.setBehavior('standup', { restHeight: rest });
  } else {
    ctrl.initialPose(world);
    if (S.drop) { world.q[2] += S.drop * ls; ctrl.body.p[2] = ctrl.opts.bodyHeight; }
    ctrl.setBehavior('stand');
  }
  // PLA against PLA: both bodies deform, E* = E / (2(1 − ν²)) on the effective radius R* = R₁R₂/(R₁ + R₂)
  const pairs = selfCollision ? world.enableSelfCollision({ k: (Rstar) => (4 / 3) * (pla.E / (2 * (1 - pla.nu * pla.nu))) * Math.sqrt(Rstar) * SHELL_SOFTENING, mu: 0.3 }) : 0;
  const sim = { world, model, meta, controller: ctrl, scenario, spec: S, events: [], started: false, selfPairs: pairs };
  sim.step = () => stepScenario(sim);
  sim.command = (cmd) => ctrl.setCommand(cmd);
  sim.action = (name, params) => ctrl.requestAction(name, params);
  return sim;
}

function stepScenario(sim) {
  const { world, controller: ctrl, spec: S } = sim;
  const t = world.time;
  if (!sim.started && S.behavior !== 'stand' && S.start !== 'lying' && t >= (S.settle ?? 0)) { ctrl.setBehavior(S.behavior, S.params || {}); sim.started = true; }
  if (S.push && !sim.pushed && t >= S.push.at) {
    const F = S.push.fraction * sim.meta.totalMass * 9.81;
    world.addExternalForce(0, [0, 0, 0], [0, F, 0], S.push.duration); sim.pushed = true;
    sim.events.push({ t, kind: 'push', text: `Lateral push ${F.toFixed(1)} N for ${S.push.duration} s` });
  }
  if (S.failServo && !sim.failed && t >= S.failServo.at) {
    const act = world.actuators.find((a) => a.limb === S.failServo.limb && a.joint === S.failServo.joint);
    if (act) { act.servo.failed = true; sim.failed = true; sim.events.push({ t, kind: 'failure', text: `L${S.failServo.limb} ${S.failServo.joint} servo unpowered` }); }
  }
  ctrl.update();
  if (ctrl.events.length) for (const e of ctrl.events.splice(0)) sim.events.push(e);
  world.step();
}

// Lateral shove on the carapace, as a fraction of body weight, along heading `angle` (rad, world frame).
export function shove(sim, { fraction = 0.3, duration = 0.15, angle = Math.PI / 2 } = {}) {
  const F = fraction * sim.meta.totalMass * 9.81;
  sim.world.addExternalForce(0, [0, 0, 0], [F * Math.cos(angle), F * Math.sin(angle), 0], duration);
  sim.events.push({ t: sim.world.time, kind: 'push', text: `Manual shove ${F.toFixed(1)} N for ${duration} s` });
}

export function measure(sim) {
  const { world, meta } = sim, kin = world.kin;
  const com = kin.com();
  const { P } = kin.momentum();
  const M = meta.totalMass;
  const comVel = [P[0] / M, P[1] / M, P[2] / M];
  const g = [0, 0, -9.81];
  const feet = world.contacts.filter((c) => c.kind === 'foot');
  // COM acceleration by finite difference over the interval since the previous measurement (callers may
  // sample slower than the physics rate; dividing by world.dt would then overstate the inertial force).
  const interval = sim.lastMeasureT != null ? world.time - sim.lastMeasureT : 0;
  const comAcc = sim.lastComVel && interval > 0 ? comVel.map((v, i) => (v - sim.lastComVel[i]) / interval) : [0, 0, 0];
  const stab = stabilityReport({ contacts: feet, com, comVel, comAcc, mass: M, gravity: g });
  sim.lastComVel = comVel; sim.lastMeasureT = world.time;
  const servos = world.actuators.map((a) => ({ name: a.servo.name, limb: a.limb, joint: a.joint, tau: a.servo.tauMotor, current: a.servo.I, temp: a.servo.T, winding: a.servo.Tw, duty: a.servo.duty, protected: a.servo.protected, unload: a.servo.unload, failed: a.servo.failed, q: world.q[a.dof + 1], target: a.servo.targetSmooth }));
  const shellHits = world.contacts.filter((c) => c.kind === 'shell' && c.fn > 0.05 * M * 9.81);
  const selfHits = world.contacts.filter((c) => c.kind === 'self' && c.fn > 0.02 * M * 9.81);
  const slip = Math.max(0, ...feet.filter((c) => c.fn > 0.5).map((c) => c.slip));
  return { t: world.time, com, comVel, stab, servos, battery: { v: world.battery.voltage, i: world.battery.current, soc: world.battery.soc }, feet, slip, shellHits: shellHits.map((c) => c.name), selfHits: selfHits.map((c) => `${c.name}–${c.other}`), normal: feet.reduce((s, c) => s + c.fn, 0), weight: M * 9.81, solver: { ...world.stats } };
}
export { SCENARIOS, SERVO_PRESETS };
