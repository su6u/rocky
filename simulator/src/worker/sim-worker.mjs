// Physics worker. Owns the simulation (src/sim), advances it against the wall clock at a chosen playback rate, and
// streams snapshots to the viewer. Keeping the 1 kHz engine off the main thread lets the renderer hold its frame
// rate while the solver works. Message protocol: ./protocol.mjs.
import { createSimulation, measure, shove, SCENARIOS, SERVO_PRESETS } from '../sim/simulation.mjs';
import { MSG, ACTIONS, XF_STRIDE } from './protocol.mjs';

const FRAME_MS = 1000 / 60;         // snapshot cadence
const BUDGET_MS = 12;               // physics compute allowed per tick
const SLIP_EVENT = 0.005;           // m/s, loaded-foot slip worth reporting

let robot = null, sim = null, config = null;
let playing = false, rate = 1, timer = null;
let wallAnchor = 0, simAnchor = 0;
let rtf = { value: 0, t0: 0, s0: 0 };
let flags = {};

const post = (msg) => self.postMessage(msg);

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    switch (msg.type) {
      case MSG.INIT: {
        robot = await fetch(msg.robotUrl).then((r) => { if (!r.ok) throw new Error(`robot description: HTTP ${r.status}`); return r.json(); });
        const servos = Object.fromEntries(Object.entries(SERVO_PRESETS).map(([k, p]) => [k, { label: p.label, source: p.source, stall: p.stallTorqueDatasheet, rated: p.ratedTorqueDatasheet, maxVelocity: p.maxVelocity, mass: p.mass, ideal: !!p.ideal }]));
        post({ type: MSG.READY, scenarios: SCENARIOS, servos, statuePose: robot.statuePose, statueUp: robot.torso.statueUp });
        break;
      }
      case MSG.LOAD: load(msg.config); break;
      case MSG.PLAY: setPlaying(true); break;
      case MSG.PAUSE: setPlaying(false); break;
      case MSG.RATE: rate = msg.value; anchor(); break;
      case MSG.STEP: {
        if (!sim) return;
        setPlaying(false);
        const end = sim.world.time + msg.seconds;
        while (sim.world.time < end - 1e-9) if (!advance()) break;
        post({ type: MSG.FRAME, frame: snapshot() });
        break;
      }
      case MSG.SHOVE: if (sim) shove(sim, msg.options); break;
      case MSG.COMMAND: if (sim) sim.command({ vx: +msg.vx || 0, vy: +msg.vy || 0, wz: +msg.wz || 0, height: Number.isFinite(msg.height) ? msg.height : undefined }); break;
      case MSG.ACTION: if (sim && ACTIONS.includes(msg.name)) sim.action(msg.name, msg.params || {}); break;
      default: break;
    }
  } catch (err) {
    fail(err);
  }
};

function load(cfg) {
  setPlaying(false);
  config = cfg;
  const t0 = performance.now();
  sim = createSimulation(robot, {
    scenario: cfg.scenario,
    servoPreset: cfg.servoPreset,
    backlash: !!cfg.backlash,
    design: cfg.hand ? { hand: cfg.hand } : {},
    env: cfg.env || {},
    seed: cfg.seed ?? 1,
  });
  sim.world.kin.update(sim.world.q, sim.world.v);
  flags = { protected: new Set(), overheated: new Set(), shell: new Set(), self: new Set(), complete: false, eventCursor: 0 };
  rtf = { value: 0, t0: performance.now(), s0: 0 };
  post({ type: MSG.LOADED, info: describe(), frame: snapshot(), buildMs: performance.now() - t0 });
}

function setPlaying(on) {
  playing = on && !!sim && !flags.complete;
  if (timer) { clearTimeout(timer); timer = null; }
  if (playing) { anchor(); timer = setTimeout(tick, 0); }
  post({ type: MSG.PLAYING, value: playing });
}

function anchor() {
  wallAnchor = performance.now();
  simAnchor = sim ? sim.world.time : 0;
  rtf.t0 = wallAnchor; rtf.s0 = simAnchor;
}

// One physics step with failure containment. Returns false when the run must stop.
function advance() {
  try {
    sim.step();
  } catch (err) {
    fail(err);
    return false;
  }
  const q = sim.world.q;
  if (!Number.isFinite(q[0] + q[1] + q[2] + q[3])) { fail(new Error('numerical divergence (non-finite state)')); return false; }
  const duration = sim.spec.duration;
  if (duration && sim.world.time >= duration - 1e-9) {
    flags.complete = true;
    sim.events.push({ t: sim.world.time, kind: 'complete', text: `Scenario complete (${duration} s)` });
    return false;
  }
  return true;
}

function tick() {
  timer = null;
  if (!playing || !sim) return;
  const start = performance.now();
  const maxSpeed = rate === 'max';
  const target = maxSpeed ? Infinity : simAnchor + ((start - wallAnchor) / 1000) * rate;
  let running = true;
  while (sim.world.time < target - 1e-9 && performance.now() - start < BUDGET_MS) {
    if (!advance()) { running = false; break; }
  }
  // Could not keep up: drop the debt instead of spiralling, and report the achieved real-time factor.
  if (!maxSpeed && sim.world.time < target - 0.05) { wallAnchor = performance.now(); simAnchor = sim.world.time; }
  const now = performance.now();
  if (now - rtf.t0 > 500) { rtf.value = (sim.world.time - rtf.s0) / ((now - rtf.t0) / 1000); rtf.t0 = now; rtf.s0 = sim.world.time; }
  post({ type: MSG.FRAME, frame: snapshot() });
  if (!running) { setPlaying(false); return; }
  timer = setTimeout(tick, Math.max(0, FRAME_MS - (performance.now() - start)));
}

function fail(err) {
  playing = false;
  if (timer) { clearTimeout(timer); timer = null; }
  post({ type: MSG.ERROR, message: err.message || String(err), t: sim ? sim.world.time : 0 });
  post({ type: MSG.PLAYING, value: false });
}

// Static description of the loaded robot and scenario for the renderer and UI.
function describe() {
  const { model, meta, world, controller: ctrl, spec } = sim;
  const bodies = model.bodies.map((b, i) => ({ index: i, name: b.name, parent: b.parent, part: b.part || null, partToBody: b.partToBody || null, joint: b.joint || null, limb: b.limb || null, servo: !!b.servo, lo: b.lo, hi: b.hi }));
  const walking = ['walk', 'turn', 'teleop'].includes(spec.behavior);
  ctrl.walkTiming(walking ? spec.params?.speed ?? 0 : 0, 0, walking ? spec.params?.turn ?? 0 : 0);
  const rMax = Math.max(...ctrl.legs.map((l) => l.radius));
  return {
    scenario: sim.scenario,
    spec: { label: spec.label, group: spec.group, description: spec.description, duration: spec.duration, behavior: spec.behavior, fast: !!spec.fast, interactive: !!spec.interactive, speed: spec.params?.speed ?? null, mu: world.terrain.mu, slope: spec.slope ?? 0, ambient: spec.ambient ?? 25, payload: meta.payloadMass },
    config,
    scale: meta.design.scale,
    lengthScale: meta.design.scale / 0.0045,
    totalMass: meta.totalMass,
    structuralMass: meta.structuralMass,
    servoMassTotal: meta.servoMassTotal,
    carapaceAcross: meta.carapaceAcross,
    weight: meta.totalMass * 9.81,
    hand: meta.design.hand,
    servo: { key: config.servoPreset || meta.design.servo, label: meta.servo.label, source: meta.servo.source, stall: meta.servo.stallTorqueDatasheet, rated: meta.servo.ratedTorqueDatasheet, maxVelocity: meta.servo.maxVelocity, Tmax: meta.servo.thermal.Tmax, vMin: meta.servo.vMin, backlashDeg: (meta.servo.backlashRad * 180) / Math.PI, ideal: !!meta.servo.ideal },
    battery: { cells: world.battery.cells, capacityAh: world.battery.capacityAh, rInternal: world.battery.rInternal },
    bodies,
    limbs: meta.limbs.map((L) => ({ limb: L.limb, yaw: L.yaw, pitch: L.pitch, elbow: L.elbow, footBody: L.footBody, footLocal: L.foot.local, footRadius: L.foot.radius, bPart: L.bPart, lengthA: L.lengthA, lengthB: L.lengthB })),
    shapes: world.contactShapes.map((c) => ({ body: c.body, local: Array.from(c.local), radius: c.radius, kind: c.kind, name: c.name })),
    actuators: world.actuators.map((a) => ({ name: a.servo.name, limb: a.limb, joint: a.joint, body: a.body })),
    terrain: world.terrain.describe(),
    selfCollisionPairs: sim.selfPairs,
    gait: { order: ctrl.gait.order.map((i) => i + 1), plannedMargin: ctrl.gait.plannedMargin, stanceYawDeviation: ctrl.stanceYawDeviation, layout: ctrl.stanceLayout, limits: ctrl.walkLimits || null, rMax },
    controller: { rate: ctrl.opts.rate, busDelay: ctrl.opts.busDelay, bodyHeight: ctrl.opts.bodyHeight, stepHeight: ctrl.opts.stepHeight, forceBalance: ctrl.opts.forceBalance },
    dt: world.dt,
  };
}

function snapshot() {
  const { world, model, meta, controller: ctrl } = sim, kin = world.kin;
  kin.update(world.q, world.v);
  const m = measure(sim);
  const nb = model.nb;
  const xf = new Float64Array(nb * XF_STRIDE);
  for (let i = 0; i < nb; i++) { xf.set(kin.Rw[i], i * XF_STRIDE); xf.set(kin.pw[i], i * XF_STRIDE + 9); xf.set(kin.axisW[i], i * XF_STRIDE + 12); }
  const plan = ctrl.lastPlan;
  const limbs = meta.limbs.map((L, k) => {
    const leg = ctrl.legs[k];
    return {
      shoulder: Array.from(kin.pw[L.yaw]), elbow: Array.from(kin.pw[L.elbow]), foot: Array.from(kin.pointWorld(L.footBody, L.foot.local)),
      target: leg.targetFoot ? Array.from(leg.targetFoot) : null, planned: leg.contact !== false, reach: leg.reachFraction ?? 0, ik: leg.ikError ?? 0,
      load: leg.loadMeasured ?? 0, loadTarget: leg.loadTarget ?? 0,
    };
  });
  const contacts = world.contacts.map((c) => ({ kind: c.kind, name: c.name, other: c.other, env: c.env, p: Array.from(c.point), f: c.force ? Array.from(c.force) : [0, 0, 0], fn: c.fn ?? 0, use: c.frictionUse ?? 0, slip: c.slip ?? 0 }));
  const servos = world.actuators.map((a) => {
    const s = a.servo;
    return { limb: a.limb, joint: a.joint, tau: s.tauMotor, current: s.I, temp: s.T, winding: s.Tw, duty: s.duty, protected: s.protected, failed: s.failed, unload: s.unload, hot: s.unload === 'overheat', q: world.q[a.dof + 1], target: s.targetSmooth, peak: s.peakTorque, rms: s.rmsTorque(), velocity: world.v[a.dof] };
  });
  const loco = ctrl.loco, [w, , , z] = [world.q[3], world.q[4], world.q[5], world.q[6]];
  return {
    t: world.time,
    xf,
    com: Array.from(m.com), comVel: m.comVel,
    stab: { loadedFeet: m.stab.loadedFeet, polygon: m.stab.polygon, ssm: fin(m.stab.ssm), zmp: m.stab.zmp, zmpMargin: fin(m.stab.zmpMargin), nesm: fin(m.stab.nesm), fasm: fin(m.stab.fasm), capture: m.stab.capture, captureMargin: fin(m.stab.captureMargin), comProjection: m.stab.comProjection || null },
    normal: m.normal, weight: m.weight, slip: m.slip, shellHits: m.shellHits, selfHits: m.selfHits,
    limbs, contacts, servos,
    battery: m.battery,
    solver: { rows: world.stats.rows, iterations: world.stats.iterations, residual: world.stats.residual },
    body: { z: world.q[2], speed: Math.hypot(m.comVel[0], m.comVel[1]), yaw: 2 * Math.atan2(z, w) },
    drive: { command: { ...loco.cmd }, velocity: { ...loco.vel }, idle: loco.idle, height: ctrl.teleop.height, queued: ctrl.teleop.queue.length, vMax: ctrl.walkLimits?.vMax ?? 0 },
    swingLeg: plan?.swingLeg ?? -1,
    behavior: ctrl.behavior,
    faults: [...ctrl.faults.entries()].map(([name, f]) => ({ name, kind: f.kind })),
    rtf: rtf.value,
    complete: flags.complete,
    events: collectEvents(m),
  };
}

const fin = (x) => (Number.isFinite(x) ? x : null);

// Turn state changes into one-off events (scenario and controller events are forwarded as they appear).
function collectEvents(m) {
  const out = [];
  const t = sim.world.time;
  while (flags.eventCursor < sim.events.length) out.push(sim.events[flags.eventCursor++]);
  for (const a of sim.world.actuators) {
    const s = a.servo, id = `L${a.limb} ${a.joint}`;
    if (s.protected && !flags.protected.has(id)) { flags.protected.add(id); out.push({ t, kind: 'protect', text: `${id} overload protection: output limited to 20 % until a new goal` }); }
    if (!s.protected) flags.protected.delete(id);
    if (s.unload === 'overheat' && !flags.overheated.has(id)) { flags.overheated.add(id); out.push({ t, kind: 'thermal', text: `${id} board sensor above ${s.p.thermal.Tmax} °C: firmware released torque (winding ${s.Tw.toFixed(0)} °C)` }); }
    if (s.unload !== 'overheat') flags.overheated.delete(id);
  }
  for (const name of m.shellHits) if (!flags.shell.has(name)) { flags.shell.add(name); out.push({ t, kind: 'contact', text: name === 'carapace' ? 'Carapace touched the ground' : `Shell of ${name.replace(/^([AB])(\d)$/, 'limb $2 segment $1')} touched the ground` }); }
  for (const pair of m.selfHits) if (!flags.self.has(pair)) { flags.self.add(pair); out.push({ t, kind: 'contact', text: `Self-collision: ${pair.replace(/([AB])(\d)/g, 'limb $2$1').replace('foot', 'foot ')}` }); }
  // Conditions that can flicker (slip, tipping, brown-out) re-arm only after staying clear for 2 s.
  const edge = (key, active, text, kind) => {
    const f = flags[key] ??= { on: false, clearSince: -Infinity };
    if (active) {
      if (!f.on && t - f.clearSince >= 2) out.push({ t, kind, text });
      f.on = true;
    } else if (f.on) { f.on = false; f.clearSince = t; }
  };
  edge('slip', m.slip > SLIP_EVENT, `Foot slipping ${(m.slip * 1000).toFixed(1)} mm/s`, 'slip');
  edge('tip', m.stab.loadedFeet >= 3 && Number.isFinite(m.stab.ssm) && m.stab.ssm < 0, 'COM projection left the support polygon', 'tip');
  edge('brownout', m.battery.v < sim.meta.servo.vMin, `Bus brown-out: ${m.battery.v.toFixed(2)} V < servo minimum ${sim.meta.servo.vMin} V`, 'power');
  return out;
}
