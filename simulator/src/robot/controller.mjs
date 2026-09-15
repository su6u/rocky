// Rocky behaviour controller. Runs at the servo-bus rate: reads only what a physical robot can sense (encoders, IMU,
// foot switches, servo status), plans a body pose and foot targets, solves limb IK on the true kit axes and writes
// position goals to the simulated servos with transport delay.
//
// Behaviours: stand (breathing idle), walk / turn (constant command), teleop (live velocity command + queued actions),
// fist bump and wave (gestures on one limb), startle, stand-up, sit, and limp (fault reaction).
import { LimbIK } from './ik.mjs';
import { Locomotion } from './locomotion.mjs';
import { Estimator, rpyMatrix } from './estimator.mjs';
import { planWaveOrder, smooth5 } from './gait.mjs';
import { hull2d, chebyshevCenter, polygonMargin } from '../engine/stability.mjs';
import { mat3vec, mat3Tvec, axisAngleMat } from '../engine/linalg.mjs';
import { createRng } from '../engine/random.mjs';

export { ENCODER_COUNTS } from './estimator.mjs';
const TAU = Math.PI * 2;
const rotZ = (a) => axisAngleMat([0, 0, 1], a);
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// Gesture timelines (seconds): shift weight, raise, act, lower, shift back.
const GESTURE = { shift: 1.2, raise: 1.2, bump: 1.0, wave: 2.4, lower: 1.2 };

export class RockyController {
  constructor(world, meta, opts = {}) {
    this.world = world; this.meta = meta; this.model = world.model;
    this.opts = {
      rate: 50, busDelay: 0.006, bodyHeight: null, stanceReach: 0.74, stepHeight: null, speed: 0.018,
      // sagCompensation stays off: the 1:345 gear train's stiction carries most static load, so a joint pre-load
      // makes the motor fight its own gearbox (5× the current for < 0.4 mm less sag at this build)
      sagCompensation: false, forceBalance: true, attitudeFeedback: true, legOdometry: false, terrainFollow: 0.5, imuNoiseDeg: 0.15,
      faultReaction: true, seed: 1, ...opts,
    };
    const s = meta.design.scale / 0.0045;             // geometric similarity factor relative to the desk design
    this.lengthScale = s;
    this.opts.bodyHeight ??= 0.145 * s;
    this.opts.stepHeight ??= 0.015 * s;              // clearance above the highest point under the swing path
    this.rng = opts.rng ?? createRng(this.opts.seed);
    this.legs = meta.limbs.map((L) => this.makeLeg(L));
    this.setupStance();
    this.measureHomeJoints();
    this.gait = planWaveOrder(this.legs.map((l) => l.home));
    this.estimator = new Estimator(world, this.model, this.legs, { rng: this.rng, imuNoiseDeg: this.opts.imuNoiseDeg });
    this.loco = new Locomotion(this);
    this.actuatorByBody = new Map(world.actuators.map((a) => [a.body, a]));
    this.nextTick = 0; this.nextHealth = 0; this.behavior = 'stand'; this.t0 = 0; this.params = {}; this.phaseState = {};
    this.body = { p: [0, 0, this.opts.bodyHeight], yaw: 0, roll: 0, pitch: 0 };
    this.bodyStart = { ...this.body, p: [...this.body.p] };
    this.attitudeI = [0, 0];
    this.groundRef = 0;
    this.events = [];                                  // {t, kind, text}; forwarded by the simulation
    this.faults = new Map();                           // 'L3 pitch' → {kind, t}
    this.teleop = { cmd: { vx: 0, vy: 0, wz: 0 }, height: 0, queue: [], active: null };
    this.heightOffset = 0;                             // body-height request followed at ≤ 20 mm/s (a step drops the body)
    this.comOffset = this.nominalComOffset();
  }

  makeLeg(L) {
    const B = this.model.bodies;
    const j = (idx) => ({ R0: B[idx].R0, r0: B[idx].r0, axis: B[idx].axis, lo: B[idx].lo, hi: B[idx].hi });
    const ik = new LimbIK([j(L.yaw), j(L.pitch), j(L.elbow)], L.foot.local);
    return { L, ik, home: null, foot: [0, 0, 0], stanceFoot: [0, 0, 0], contact: true, angles: [0, 0, 0], limits: [j(L.yaw), j(L.pitch), j(L.elbow)].map((x) => [x.lo, x.hi]) };
  }

  // Neutral footholds. The kit's shoulder studs are irregular (≈0°, 64°, 152°, 226°, 270°). Feet spaced at a uniform 72°
  // give the best-shaped support polygon but start some shoulders near their yaw limit (up to 0.48 rad), which halves
  // the stride; feet straight out along each shoulder waste polygon margin. The layout blends the two,
  //   φ_i = φ_i^uniform + α (φ_i^radial − φ_i^uniform),
  // and α is chosen to maximise the stride every leg's shoulder-yaw room allows while keeping ≥ 95 % of the uniform
  // layout's wave-gait support margin. The radius puts each foot at `stanceReach` of the limb's length.
  setupStance() {
    const radial = this.legs.map((l) => Math.atan2(l.L.shoulder[1], l.L.shoulder[0]));
    const n = radial.length, order = [...radial.keys()].sort((a, b) => ((radial[a] + TAU) % TAU) - ((radial[b] + TAU) % TAU));
    const wrap = (a) => ((((a + Math.PI) % TAU) + TAU) % TAU) - Math.PI;
    let best = null;
    for (let th0 = -Math.PI; th0 < Math.PI; th0 += 0.0025) {
      let worst = 0;
      order.forEach((li, k) => { worst = Math.max(worst, Math.abs(wrap(th0 + (TAU * k) / n - radial[li]))); });
      if (!best || worst < best.worst) best = { th0, worst };
    }
    const uniform = [];
    order.forEach((li, k) => { uniform[li] = best.th0 + (TAU * k) / n; });
    const place = (alpha) => this.legs.forEach((leg, i) => {
      const L = leg.L, phi = uniform[i] + alpha * wrap(radial[i] - uniform[i]);
      const reach = (L.lengthA + L.lengthB) * this.opts.stanceReach;
      const drop = this.opts.bodyHeight + L.shoulder[2] - L.foot.radius;
      const horiz = Math.sqrt(Math.max(reach * reach - drop * drop, 0.0004));
      leg.azimuth = phi; leg.radius = Math.hypot(L.shoulder[0], L.shoulder[1]) + horiz;
      leg.home = [leg.radius * Math.cos(phi), leg.radius * Math.sin(phi)];
    });
    const candidates = [];
    for (let alpha = 0; alpha <= 1.0001; alpha += 0.05) {
      place(alpha);
      this.measureHomeJoints();
      candidates.push({ alpha, stride: this.strideLimit(), margin: planWaveOrder(this.legs.map((l) => l.home)).plannedMargin });
    }
    const floor = 0.95 * candidates[0].margin;
    const chosen = candidates.filter((c) => c.margin >= floor).reduce((a, b) => (b.stride > a.stride ? b : a));
    place(chosen.alpha);
    this.stanceLayout = { alpha: chosen.alpha, strideMax: chosen.stride, margin: chosen.margin };
    this.stanceYawDeviation = Math.max(...this.legs.map((l, i) => Math.abs(wrap(l.azimuth - radial[i]))));
  }

  // Joint angles and the shoulder-to-foot lever at each leg's neutral foothold. The stride a leg allows is set by its own
  // shoulder yaw room around this pose: seen from a shoulder 7–10 cm off the body centre, a 0.24 rad azimuth deviation
  // of the foothold becomes up to ≈0.36 rad of shoulder yaw.
  measureHomeJoints() {
    for (const leg of this.legs) {
      const r = leg.L.foot.radius, local = [leg.home[0], leg.home[1], r - this.opts.bodyHeight];
      const res = leg.ik.probe(local);
      leg.homeAngles = res.theta;
      leg.homeLever = Math.hypot(leg.home[0] - leg.L.shoulder[0], leg.home[1] - leg.L.shoulder[1]);
    }
  }

  // Longest stride every leg allows: ±S/2 around its neutral foothold with 0.1 rad of shoulder-yaw margin.
  strideLimit() {
    return Math.min(...this.legs.map((l) => {
      const yaw = l.homeAngles[0], room = Math.max(0.05, Math.min(l.limits[0][1] - yaw, yaw - l.limits[0][0]) - 0.1);
      return 2 * l.homeLever * Math.tan(room);
    }));
  }

  // Whole-robot COM in the torso frame at the nominal stance (the controller's own mass model).
  nominalComOffset() {
    const kin = this.world.kin;
    kin.update(this.world.q, null);
    const c = kin.com(), R = kin.Rw[0], p = kin.pw[0];
    return mat3Tvec(R, [c[0] - p[0], c[1] - p[1], c[2] - p[2]]);
  }

  // ---------------------------------------------------------------- terrain queries (what a mapped floor provides)
  // A body-mounted depth sensor sees the floor relative to the robot, so odometry drift (stance slips add up to
  // centimetres over a walk) never misplaces a foot relative to an obstacle. Modelled at 5 Hz as a measurement of where
  // the controller's own world frame sits in the real one, with 1 mm and 0.2° noise; every terrain query goes through it.
  perceive() {
    const q = this.world.q, b = this.body, g = this.rng.normal;
    const yawTrue = Math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
    this.map = { cx: b.p[0], cy: b.p[1], dx: q[0] - b.p[0] + 0.001 * g(), dy: q[1] - b.p[1] + 0.001 * g(), dyaw: yawTrue - (b.yaw ?? 0) + 0.0035 * g() };
  }
  // Vertical leg odometry: how far the real body sits below the planned one (servo compliance, and the lag while the
  // body climbs onto a step). Swing feet are aimed lower by this amount so they meet the ground on schedule instead of
  // landing early and fast. Only the vertical component is used; horizontal estimates jump between frames at liftoff.
  measureHeightLag(dt) {
    const stance = this.legs.map((leg) => ({ leg, foot: leg.stanceFoot })).filter(({ leg }) => leg.contact !== false);
    const est = this.estimator.bodyPose(stance);
    if (!est) return;
    const lag = clamp(this.body.p[2] - est.p[2], -0.015 * this.lengthScale, 0.015 * this.lengthScale);
    this.heightLag = (this.heightLag ?? 0) + (lag - (this.heightLag ?? 0)) * (1 - Math.exp(-dt / 0.1));
    this.bodyEstimate = est;
  }
  // (a 3 mm deadband leaves the steady millimetre of sag on flat ground to the touchdown search, which handles it better)
  swingHeightCorrection(clock) {
    const lag = this.heightLag ?? 0, dead = 0.003 * this.lengthScale;
    return Math.sign(lag) * Math.max(0, Math.abs(lag) - dead) * smooth5(clock / 0.35);
  }

  toTerrain(x, y) {
    const m = this.map;
    if (!m) return [x, y];
    const c = Math.cos(m.dyaw), s = Math.sin(m.dyaw), rx = x - m.cx, ry = y - m.cy;
    return [m.cx + m.dx + c * rx - s * ry, m.cy + m.dy + s * rx + c * ry];
  }
  terrainHeight(x, y, margin = 0) { const p = this.toTerrain(x, y); return this.world.terrain.heightAt(p[0], p[1], margin); }
  terrainEdge(x, y) { const p = this.toTerrain(x, y); return this.world.terrain.edgeDistance(p[0], p[1]); }
  terrainMaxAlong(a, b, margin = 0) { return this.world.terrain.maxHeightAlong(this.toTerrain(a[0], a[1]), this.toTerrain(b[0], b[1]), margin); }
  groundZ(x, y = 0) { return this.terrainHeight(x, y); }
  footZ(x, y, radius) { return this.terrainHeight(x, y, radius) + radius / Math.cos(this.world.terrain.slope); }
  stepHeight() { return this.world.terrain.boxes.length ? this.opts.stepHeight * 1.4 : this.opts.stepHeight; }
  // Body attitude that follows a planar slope partially: nose up uphill, lean into a side slope.
  terrainAttitude(yaw) {
    const g = Math.tan(this.world.terrain.slope), k = this.opts.terrainFollow;
    return { roll: k * Math.atan(-g * Math.sin(yaw)), pitch: -k * Math.atan(g * Math.cos(yaw)) };
  }
  updateGroundReference(dt) {
    const stance = this.legs.filter((l) => l.contact !== false);
    if (!stance.length) return this.groundRef;
    const slope = this.world.terrain.slope;
    const target = stance.reduce((s, l) => s + l.stanceFoot[2] - l.L.foot.radius / Math.cos(slope), 0) / stance.length;
    this.groundRef += (target - this.groundRef) * (1 - Math.exp(-dt / 0.4));
    return this.groundRef;
  }

  // ---------------------------------------------------------------- kinematics helpers
  bodyMatrix(body) { return rpyMatrix(body.roll ?? 0, body.pitch ?? 0, body.yaw ?? 0); }
  solveLeg(leg, footWorld, body) {
    const R = body.R ?? this.bodyMatrix(body);
    const local = mat3Tvec(R, [footWorld[0] - body.p[0], footWorld[1] - body.p[1], footWorld[2] - body.p[2]]);
    // project unreachable targets onto 97 % of the limb reach (sphere about the shoulder)
    const sh = leg.L.shoulder, rel = [local[0] - sh[0], local[1] - sh[1], local[2] - sh[2]], dist = Math.hypot(...rel);
    const reachMax = 0.97 * (leg.L.lengthA + leg.L.lengthB);
    if (dist > reachMax) for (let k = 0; k < 3; k++) local[k] = sh[k] + (rel[k] * reachMax) / dist;
    leg.reachFraction = dist / (leg.L.lengthA + leg.L.lengthB);
    const r = leg.ik.solve(local);
    leg.ikError = r.error; leg.angles = r.theta;
    return r.theta;
  }
  // The IMU integral trims the commanded attitude away from the target attitude to cancel leg compliance (on a 15° slope
  // by ≈3–4°). Legs are commanded in that trimmed frame, so a point on the real ground must be mapped into it before a
  // swing foot is sent there: p' = p_b + R_cmd R_targetᵀ (p − p_b). Without this the uphill feet of a slope land ≈17 mm
  // short of the ground.
  toCommandFrame(p) {
    const b = this.body, yaw = b.yaw ?? 0, att = this.terrainAttitude(yaw);
    const Rt = rpyMatrix(att.roll, att.pitch, yaw), Rc = rpyMatrix(att.roll + 0.6 * this.attitudeI[0], att.pitch + 0.6 * this.attitudeI[1], yaw);
    const d = mat3Tvec(Rt, [p[0] - b.p[0], p[1] - b.p[1], p[2] - b.p[2]]), w = mat3vec(Rc, d);
    return [b.p[0] + w[0], b.p[1] + w[1], b.p[2] + w[2]];
  }

  // Can this leg put its foot at xy (on the ground) with the body at `pose`, with joint-limit margin to spare?
  footReachable(leg, xy, pose, { margin = 0.06, maxReach = 0.9 } = {}) {
    const r = leg.L.foot.radius, z = this.footZ(xy[0], xy[1], r);
    const att = this.terrainAttitude(pose.yaw);
    const bodyZ = this.groundRef + this.opts.bodyHeight + this.heightOffset;
    const R = rpyMatrix(att.roll, att.pitch, pose.yaw);
    const local = mat3Tvec(R, [xy[0] - pose.x, xy[1] - pose.y, z - bodyZ]);
    const sh = leg.L.shoulder;
    if (Math.hypot(local[0] - sh[0], local[1] - sh[1], local[2] - sh[2]) > maxReach * (leg.L.lengthA + leg.L.lengthB)) return false;
    const res = leg.ik.probe(local);
    if (res.error > 0.0015 * this.lengthScale) return false;
    return res.theta.every((th, k) => th > leg.limits[k][0] + margin && th < leg.limits[k][1] - margin);
  }

  // Place the robot in a standing configuration with feet on the ground (used by scenario setup).
  initialPose(world, x = 0, y = 0, yaw = 0) {
    this.groundRef = this.groundZ(x, y);
    this.body = { p: [x, y, this.opts.bodyHeight + this.groundRef], yaw, roll: 0, pitch: 0 };
    world.q.fill(0); world.q[3] = 1; world.v.fill(0);
    world.q[0] = x; world.q[1] = y; world.q[2] = this.body.p[2];
    world.q[3] = Math.cos(yaw / 2); world.q[6] = Math.sin(yaw / 2);
    for (const leg of this.legs) {
      const h = mat3vec(rotZ(yaw), [leg.home[0], leg.home[1], 0]);
      leg.foot = [x + h[0], y + h[1], this.footZ(x + h[0], y + h[1], leg.L.foot.radius)];
      leg.stanceFoot = [...leg.foot];
      const th = this.solveLeg(leg, leg.foot, this.body);
      this.setJoint(world, leg.L.yaw, th[0]); this.setJoint(world, leg.L.pitch, th[1]); this.setJoint(world, leg.L.elbow, th[2]);
    }
    for (const a of world.actuators) { a.servo.targetSmooth = null; a.servo.targetCmd = world.q[a.dof + 1]; a.servo.queue = []; }
    this.bodyStart = { ...this.body, p: [...this.body.p] };
  }
  setJoint(world, bodyIndex, angle) { world.q[7 + bodyIndex - 1] = angle; }

  // ---------------------------------------------------------------- commands
  setBehavior(name, params = {}) {
    this.behavior = name; this.params = params; this.t0 = this.world.time; this.phaseState = {};
    for (const leg of this.legs) leg.contact = true;
    this.bodyStart = { ...this.body, p: [...this.body.p] };
    if (name === 'walk' || name === 'turn' || name === 'teleop') {
      this.loco.reset(this.world.time, { x: this.body.p[0], y: this.body.p[1], yaw: this.body.yaw });
      const wanted = name === 'walk' ? { vx: params.speed ?? this.opts.speed, vy: params.lateral ?? 0, wz: params.turn ?? 0 }
        : name === 'turn' ? { vx: 0, vy: 0, wz: params.turn ?? 0.08 } : this.teleop.cmd;
      const cmd = this.clampCommand(wanted);
      if (name !== 'teleop' && Math.abs(cmd.vx - wanted.vx) + Math.abs(cmd.wz - wanted.wz) > 1e-9) {
        this.events.push({ t: this.world.time, kind: 'gait', text: `Command limited to the gait envelope: ${(Math.hypot(cmd.vx, cmd.vy) * 1000).toFixed(1)} mm/s, ${cmd.wz.toFixed(3)} rad/s` });
      }
      this.loco.setCommand(cmd);
    }
  }
  // Teleoperation: body-frame velocity (m/s, rad/s) and a body-height offset (m). Speeds are limited to the gait's
  // feasible envelope at this build, so a keyboard can never ask for a stride the servos cannot deliver.
  setCommand({ vx = 0, vy = 0, wz = 0, height = this.teleop.height } = {}) {
    const s = this.lengthScale;
    this.teleop.height = clamp(height, -0.035 * s, 0.015 * s);
    this.teleop.cmd = { vx, vy, wz };
    if (this.behavior === 'teleop' && !this.teleop.stopping) this.loco.setCommand(this.clampCommand(this.teleop.cmd));
  }
  clampCommand({ vx, vy, wz }) {
    const lim = this.walkTiming(vx, vy, wz), rMax = Math.max(...this.legs.map((l) => l.radius));
    const demand = Math.hypot(vx, vy) + Math.abs(wz) * rMax, cap = 0.95 * lim.vMax;
    const k = demand > cap ? cap / demand : 1;
    return { vx: vx * k, vy: vy * k, wz: wz * k };
  }
  requestAction(name, params = {}) {
    // sit is a toggle: a second press (or `stand`) gets up again
    if (name === 'stand' || (name === 'sit' && this.behavior === 'sit')) { if (this.behavior === 'sit') this.pendingStand = true; return; }
    if (this.teleop.queue.length < 4) this.teleop.queue.push({ name, params });
  }

  // ---------------------------------------------------------------- control loop
  update() {
    const world = this.world;
    if (world.time + 1e-12 < this.nextTick) return;
    const dt = 1 / this.opts.rate;
    this.nextTick += dt;
    if (world.time >= this.nextHealth) { this.nextHealth += 0.1; this.monitorHealth(); }
    if (world.time >= (this.nextPerception ?? 0)) { this.nextPerception = world.time + 0.2; this.perceive(); }
    this.measureHeightLag(dt);
    this.updateGroundReference(dt);
    const hRate = 0.02 * this.lengthScale * dt;
    this.heightOffset += clamp(this.teleop.height - this.heightOffset, -hRate, hRate);
    const plan = this.plan(world.time - this.t0);
    const imu = this.estimator.imu();
    if (this.opts.attitudeFeedback && plan.level !== false) {
      // integral trim with a slow leak (τ = 20 s) so stiction-induced dead zones cannot wind it up
      const leak = 1 - dt / 20;
      this.attitudeI[0] = clamp(leak * this.attitudeI[0] + (plan.body.roll - imu.roll) * dt, -0.15, 0.15);
      this.attitudeI[1] = clamp(leak * this.attitudeI[1] + (plan.body.pitch - imu.pitch) * dt, -0.15, 0.15);
      plan.body.roll += 0.6 * this.attitudeI[0];
      plan.body.pitch += 0.6 * this.attitudeI[1];
    }
    this.body = plan.body;
    plan.body.R = this.bodyMatrix(plan.body);
    // leg odometry + IMU: where the body really is, so swing feet land where planned despite stance-leg sag
    const stance = this.legs.map((leg, i) => ({ leg, foot: leg.stanceFoot, i })).filter(({ leg }) => leg.contact !== false);
    this.estimate = this.opts.legOdometry ? this.estimator.bodyPose(stance, imu) : null;
    if (this.opts.forceBalance) this.balanceLoads(plan, dt);
    const goals = [], dtc = dt, maxRate = 0.8 * this.meta.servo.maxVelocity;
    for (let i = 0; i < this.legs.length; i++) {
      const leg = this.legs[i], f = plan.feet[i];
      const stanceLeg = leg.contact !== false;
      leg.targetFoot = stanceLeg && leg.loadOffset ? [f[0], f[1], f[2] + leg.loadOffset] : f;
      const frame = !stanceLeg && this.estimate ? this.estimate : plan.body;
      const th = this.solveLeg(leg, leg.targetFoot, frame);
      // rate-limit goals: a branch change or unreachable target must not command a joint slew beyond the servo
      if (leg.lastGoal) for (let k = 0; k < 3; k++) th[k] = leg.lastGoal[k] + clamp(th[k] - leg.lastGoal[k], -maxRate * dtc, maxRate * dtc);
      goals.push(th);
    }
    if (this.opts.sagCompensation) this.compensateSag(goals, plan); else for (const leg of this.legs) leg.sagOffset = [0, 0, 0];
    for (let i = 0; i < this.legs.length; i++) {
      const leg = this.legs[i], th = goals[i];
      leg.lastGoal = [...th];
      [leg.L.yaw, leg.L.pitch, leg.L.elbow].forEach((b, k) => {
        const act = this.actuatorByBody.get(b);
        // pre-load offsets change when a leg lifts or lands; low-pass them (τ = 80 ms) so the joint does not twitch
        leg.sagSmooth ??= [0, 0, 0];
        leg.sagSmooth[k] += ((leg.sagOffset?.[k] ?? 0) - leg.sagSmooth[k]) * (1 - Math.exp(-dtc / 0.08));
        if (act) act.servo.command(th[k] + leg.sagSmooth[k], world.time, this.opts.busDelay);
      });
    }
    for (const e of this.loco.log.splice(0)) this.events.push(e);
    this.lastPlan = plan;
  }

  // Load balancing with the pad force sensors. Five (or four) position-controlled stance legs are statically
  // indeterminate: millimetre mismatches from servo sag, backlash and encoder steps decide which feet carry the robot,
  // and a nearly unloaded foot slides as soon as the body sways. Desired vertical loads come from the controller's
  // mass model (vertical-only minimum norm with moment balance about the planned COM):
  //   min Σ F_i²  s.t.  Σ F_i = m g,  Σ F_i (x_i − c_x) = 0,  Σ F_i (y_i − c_y) = 0
  // The leg about to lift is removed from the distribution over the pre-lift phase, so its load hands over smoothly.
  // Each stance foot target is raised when overloaded and lowered when underloaded (integral action, zero-mean so the
  // body height is unchanged):  δz_i += k (F_i − F_i*) dt,  |δz_i| ≤ 4 mm.
  balanceLoads(plan, dt) {
    const s = this.lengthScale, W = this.meta.totalMass * 9.81, k = 0.004 / s, lim = 0.004 * s;
    const legs = this.legs, stance = legs.map((l, i) => i).filter((i) => legs[i].contact !== false);
    const R = plan.body.R ?? this.bodyMatrix(plan.body), com = mat3vec(R, this.comOffset);
    com[0] += plan.body.p[0]; com[1] += plan.body.p[1];
    const slot = this.behavior === 'walk' || this.behavior === 'turn' || this.behavior === 'teleop' ? this.loco.slot : null;
    let handover = -1, beta = 0;
    if (plan.handover) ({ leg: handover, beta } = plan.handover);   // gestures and fault reactions declare their own
    else if (slot && !slot.done) {
      // hand the load over during the last 0.2 s before liftoff: a foot unloaded any earlier just slides as the body sways
      handover = slot.leg; beta = smooth5(1 - (slot.tLift - this.world.time) / Math.min(0.2, slot.tLift - slot.t0));
    }
    const distribute = (ids) => {
      if (ids.length < 3) return null;
      const A = [ids.map(() => 1), ids.map((i) => legs[i].stanceFoot[0] - com[0]), ids.map((i) => legs[i].stanceFoot[1] - com[1])];
      const AAt = [0, 1, 2].map((r) => [0, 1, 2].map((c2) => A[r].reduce((sum, _, j) => sum + A[r][j] * A[c2][j], 0) + (r === c2 ? 1e-9 : 0)));
      const y = solveDense(AAt, [W, 0, 0]);
      return new Map(ids.map((i, j) => [i, Math.max(0, A[0][j] * y[0] + A[1][j] * y[1] + A[2][j] * y[2])]));
    };
    const all = distribute(stance), without = handover >= 0 ? distribute(stance.filter((i) => i !== handover)) : null;
    for (const i of stance) {
      const leg = legs[i], target = without ? (1 - beta) * (all?.get(i) ?? 0) + beta * (without.get(i) ?? 0) : all?.get(i) ?? 0;
      const measured = this.estimator.footForce(i, dt);
      leg.loadTarget = target; leg.loadMeasured = measured;
      leg.loadOffset = clamp((leg.loadOffset ?? 0) + k * (measured - target) * dt, -lim, lim);
    }
    // Keep only the part of the offsets that redistributes load: remove their best-fit plane a + b·x + c·y, which would
    // raise or tilt the body and fight the height and attitude loops (with three stance feet nothing is left to adjust).
    if (stance.length >= 4 && this.opts.balanceDecouple !== false) {
      const pts = stance.map((i) => [1, legs[i].stanceFoot[0] - com[0], legs[i].stanceFoot[1] - com[1]]);
      const AtA = [0, 1, 2].map((r) => [0, 1, 2].map((q) => pts.reduce((sum, p) => sum + p[r] * p[q], 0) + (r === q ? 1e-12 : 0)));
      const coef = solveDense(AtA, [0, 1, 2].map((r) => stance.reduce((sum, i, j) => sum + pts[j][r] * legs[i].loadOffset, 0)));
      stance.forEach((i, j) => { legs[i].loadOffset -= coef[0] + coef[1] * pts[j][1] + coef[2] * pts[j][2]; });
    } else if (stance.length < 4) for (const i of stance) legs[i].loadOffset = 0;
    for (let i = 0; i < legs.length; i++) {
      const leg = legs[i];
      if (leg.contact === false) { this.estimator.footForce(i, dt); leg.loadOffset = 0; leg.loadTarget = 0; }
    }
  }

  // Servo P-loops have finite stiffness k = (k_t/R)·V·k_p·g_e, so a loaded stance leg sits lower than commanded.
  // Feedforward pre-load from a model force distribution (no measured forces):
  //   min Σ‖f_i‖²  s.t.  Σ f_i = −m g,  Σ (p_i − c) × f_i = 0        f = Aᵀ (A Aᵀ)⁻¹ b
  // Joint pre-load τ_j = −â_j · ((p − o_j) × f), goal offset = τ_j / k (limb FK from the controller's own model).
  // Measured contact forces would close a positive-feedback loop in this statically indeterminate stance.
  compensateSag(goals, plan) {
    const M = this.meta.totalMass, g = 9.81;
    for (const leg of this.legs) leg.sagOffset = [0, 0, 0];
    const stance = this.legs.map((l, i) => ({ l, i })).filter(({ l }) => l.contact !== false);
    if (stance.length < 3) return;
    const body = plan.body, R = body.R, com = mat3vec(R, this.comOffset);
    com[0] += body.p[0]; com[1] += body.p[1]; com[2] += body.p[2];
    const n = stance.length, A = Array.from({ length: 6 }, () => new Float64Array(3 * n));
    stance.forEach(({ l }, k) => {
      const f = l.targetFoot ?? l.stanceFoot, r = [f[0] - com[0], f[1] - com[1], f[2] - com[2]];
      for (let a = 0; a < 3; a++) A[a][3 * k + a] = 1;
      A[3][3 * k + 1] = -r[2]; A[3][3 * k + 2] = r[1];
      A[4][3 * k + 0] = r[2]; A[4][3 * k + 2] = -r[0];
      A[5][3 * k + 0] = -r[1]; A[5][3 * k + 1] = r[0];
    });
    const b = [0, 0, M * g, 0, 0, 0];
    const AAt = Array.from({ length: 6 }, (_, r) => Array.from({ length: 6 }, (_, c) => { let s2 = 0; for (let k = 0; k < 3 * n; k++) s2 += A[r][k] * A[c][k]; return s2 + (r === c ? 1e-9 : 0); }));
    const y = solveDense(AAt, b);
    const V = this.world.battery ? this.world.battery.voltage : this.meta.servo.vNominal;
    stance.forEach(({ l, i }, k) => {
      const f = [0, 1, 2].map((a) => A.reduce((s2, row, r) => s2 + row[3 * k + a] * y[r], 0));
      const p = l.targetFoot ?? l.stanceFoot, fk = l.ik.fk(goals[i]);
      for (let j = 0; j < 3; j++) {
        const ax = mat3vec(R, fk.axes[j]), o = mat3vec(R, fk.origins[j]);
        o[0] += body.p[0]; o[1] += body.p[1]; o[2] += body.p[2];
        const r = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
        const tau = ax[0] * (r[1] * f[2] - r[2] * f[1]) + ax[1] * (r[2] * f[0] - r[0] * f[2]) + ax[2] * (r[0] * f[1] - r[1] * f[0]);
        const act = this.actuatorByBody.get([l.L.yaw, l.L.pitch, l.L.elbow][j]);
        if (!act) continue;
        const sp = act.servo.p, kStiff = (sp.kt / act.servo.resistance()) * V * sp.kp * sp.errorGain;
        l.sagOffset[j] = clamp(-tau / kStiff, -0.12, 0.12);
      }
    });
  }

  // Bus status poll (10 Hz): a real controller sees missing replies, overload and overheat error bits.
  monitorHealth() {
    const t = this.world.time;
    for (const a of this.world.actuators) {
      const s = a.servo, key = `L${a.limb} ${a.joint}`;
      const kind = s.failed ? 'no response' : s.unload === 'overheat' ? 'overheat' : s.unload === 'overcurrent' ? 'overcurrent' : s.protected ? 'overload' : null;
      const known = this.faults.get(key);
      if (kind && (!known || known.kind !== kind)) {
        this.faults.set(key, { kind, t, limb: a.limb });
        this.events.push({ t, kind: 'fault', text: `Controller: ${key} reports ${kind}` });
        if (this.opts.faultReaction && kind !== 'overload') this.beginLimp(a.limb);
      } else if (!kind && known) {
        this.faults.delete(key);
        this.events.push({ t, kind: 'recovered', text: `Controller: ${key} back to normal` });
      }
      // overheat recovery: the firmware only re-enables torque once the board is below its limit; wait for 10 K margin
      if (s.unload === 'overheat' && s.T < s.p.thermal.Tmax - 10 && s.enableTorque()) this.events.push({ t, kind: 'recovered', text: `Controller: ${key} re-enabled after cooling to ${s.T.toFixed(0)} °C` });
    }
  }
  beginLimp(limb) {
    if (this.behavior === 'limp' || this.behavior === 'standup') return;
    if (this.behavior === 'press' && limb === (this.params.limb ?? 1)) return;   // the press behaviour rests its own limb
    const leg = this.legs.findIndex((l) => l.L.limb === limb);
    if (['walk', 'turn', 'teleop'].includes(this.behavior)) { this.loco.setCommand({ vx: 0, vy: 0, wz: 0 }); this.pendingLimp = leg; return; }
    this.setBehavior('limp', { leg });
  }

  // ---------------------------------------------------------------- behaviours
  plan(t) {
    switch (this.behavior) {
      case 'walk': case 'turn': case 'teleop': return this.planLocomotion(t);
      case 'fistbump': return this.planGesture(t, 'bump');
      case 'wave': return this.planGesture(t, 'wave');
      case 'startle': return this.planStartle(t);
      case 'standup': return this.planStandUp(t);
      case 'sit': return this.planSit(t);
      case 'limp': return this.planLimp(t);
      case 'press': return this.planPress(t);
      default: return this.planStand(t);
    }
  }
  standFeet() { return this.legs.map((l) => l.stanceFoot); }
  breathing(t) {
    const s = this.lengthScale;
    return { dx: 0.003 * s * Math.sin(0.7 * t), dy: 0.002 * s * Math.sin(0.43 * t + 1), dz: 0.004 * s * Math.sin(1.3 * t), roll: 0.02 * Math.sin(0.5 * t), pitch: 0.015 * Math.sin(0.8 * t + 0.5) };
  }
  planStand(t) {
    // Rocky "never stands still": slow breathing heave and sway (puppeteers kept the rod puppet alive)
    const b = this.bodyStart, br = this.breathing(t), att = this.terrainAttitude(b.yaw);
    const body = { p: [b.p[0] + br.dx, b.p[1] + br.dy, this.opts.bodyHeight + this.groundRef + this.heightOffset + br.dz], yaw: b.yaw, roll: att.roll + br.roll, pitch: att.pitch + br.pitch };
    return { body, feet: this.standFeet(), swingLeg: -1 };
  }
  planLocomotion(t) {
    const tele = this.behavior === 'teleop';
    if (this.pendingLimp !== undefined && this.loco.idle) { const leg = this.pendingLimp; this.pendingLimp = undefined; this.setBehavior('limp', { leg }); return this.planLimp(0); }
    if (tele && this.teleop.queue.length) {
      if (!this.loco.idle || this.loco.moving()) { this.teleop.stopping = true; this.loco.setCommand({ vx: 0, vy: 0, wz: 0 }); }
      else {
        const action = this.teleop.queue.shift();
        this.teleop.stopping = false;
        this.setBehavior(action.name, { ...action.params, returnTo: 'teleop' });
        return this.plan(0);
      }
    }
    const L = this.loco.update(this.world.time);
    this.legs.forEach((l, i) => { l.contact = i !== L.swingLeg; });
    const att = this.terrainAttitude(L.pose.yaw);
    const br = L.idle && !this.loco.moving() ? this.breathing(t) : { dx: 0, dy: 0, dz: 0, roll: 0, pitch: 0 };
    const body = {
      p: [L.pose.x + L.sway[0] + br.dx, L.pose.y + L.sway[1] + br.dy, this.groundRef + this.opts.bodyHeight + this.heightOffset + br.dz],
      yaw: L.pose.yaw, roll: att.roll + br.roll, pitch: att.pitch + br.pitch,
    };
    return { body, feet: L.feet, swingLeg: L.swingLeg, idle: L.idle, vel: L.vel, command: { ...this.loco.cmd } };
  }
  // Smallest body shift that puts the COM a safe margin inside the support polygon of every foot except `exclude`
  // (searched along the line to that polygon's Chebyshev centre). Moving all the way to the centre is rarely needed
  // and costs reach on every other leg.
  supportShift(exclude, margin = 0.04 * this.lengthScale) {
    const b = this.body, c = mat3vec(this.bodyMatrix(b), this.comOffset), com = [b.p[0] + c[0], b.p[1] + c[1]];
    const poly = hull2d(this.legs.filter((_, k) => k !== exclude).map((l) => [l.stanceFoot[0], l.stanceFoot[1]])).map((q) => [q[0], q[1]]);
    const cc = chebyshevCenter(poly);
    if (!cc) return [0, 0];
    const want = Math.min(margin, 0.75 * cc[2]);
    if (polygonMargin(poly, com[0], com[1]) >= want) return [0, 0];
    let lo = 0, hi = 1;
    for (let k = 0; k < 24; k++) {
      const mid = 0.5 * (lo + hi);
      if (polygonMargin(poly, com[0] + (cc[0] - com[0]) * mid, com[1] + (cc[1] - com[1]) * mid) >= want) hi = mid; else lo = mid;
    }
    return [(cc[0] - com[0]) * hi, (cc[1] - com[1]) * hi];
  }

  // Finish a timed action; in teleop mode hand control back to the keyboard.
  finishAction() {
    if (this.params.returnTo === 'teleop') this.setBehavior('teleop');
    else this.setBehavior('stand');
  }

  // Gesture on one limb: shift the weight onto the other four, raise the limb to a reachable pose in front of its
  // shoulder (bump: push along the limb direction; wave: swing the shoulder yaw), lower, shift back.
  planGesture(t, kind) {
    const limb = kind === 'bump' ? 1 : this.params.limb ?? 1;
    const i = this.legs.findIndex((l) => l.L.limb === limb), leg = this.legs[i], s = this.lengthScale, b0 = this.bodyStart;
    const st = this.phaseState;
    if (!st.pose) {
      st.shift = this.supportShift(i, 0.045 * s);        // raising a limb also moves the COM: keep 45 mm in hand
      st.pose = this.gesturePose(leg);
      st.act = kind === 'bump' ? GESTURE.bump : GESTURE.wave;
    }
    const tRaise = GESTURE.shift, tAct = tRaise + GESTURE.raise, tLower = tAct + st.act, tBack = tLower + GESTURE.lower, tEnd = tBack + GESTURE.shift;
    if (t >= tEnd) { this.finishAction(); return this.plan(0); }
    const u = smooth5(t / GESTURE.shift) * (1 - smooth5((t - tBack) / GESTURE.shift));
    const body = { p: [b0.p[0] + st.shift[0] * u, b0.p[1] + st.shift[1] * u, this.groundRef + this.opts.bodyHeight + this.heightOffset - 0.01 * s * u], yaw: b0.yaw, roll: 0, pitch: kind === 'bump' ? -0.06 * u : 0 };
    const raise = smooth5((t - tRaise) / GESTURE.raise) * (1 - smooth5((t - tLower) / GESTURE.lower));
    const act = Math.max(0, Math.min(1, (t - tAct) / st.act));
    const R = this.bodyMatrix(body);
    let local = [...st.pose.local];
    if (kind === 'bump') { const k = 0.025 * s * Math.sin(Math.PI * act); local = local.map((x, c) => x + st.pose.dir[c] * k); }
    else if (act > 0 && act < 1) {
      const sh = leg.L.shoulder, ang = 0.35 * Math.sin(2 * TAU * act) * Math.sin(Math.PI * act), c = Math.cos(ang), sn = Math.sin(ang);
      const rel = [local[0] - sh[0], local[1] - sh[1]];
      local = [sh[0] + c * rel[0] - sn * rel[1], sh[1] + sn * rel[0] + c * rel[1], local[2]];
    }
    const w = mat3vec(R, local), raised = [w[0] + body.p[0], w[1] + body.p[1], w[2] + body.p[2]];
    const feet = this.standFeet().map((f) => [...f]), sf = leg.stanceFoot;
    feet[i] = [sf[0] + (raised[0] - sf[0]) * raise, sf[1] + (raised[1] - sf[1]) * raise, sf[2] + (raised[2] - sf[2]) * raise];
    leg.contact = raise < 1e-3;
    // hand the limb's load to the other four during the 0.3 s before it lifts, and take it back after it lands
    const handover = smooth5((t - (tRaise - 0.3)) / 0.3) * (1 - smooth5((t - tBack) / 0.3));
    return { body, feet, swingLeg: raise > 1e-3 ? i : -1, handover: { leg: i, beta: handover } };
  }
  // A raised pose in front of the shoulder that the true (skewed) limb can reach with margin: search reach and
  // elevation instead of assuming a planar two-link arm.
  gesturePose(leg) {
    const sh = leg.L.shoulder, az = Math.atan2(sh[1], sh[0]), len = leg.L.lengthA + leg.L.lengthB;
    const dirH = [Math.cos(az), Math.sin(az)];
    for (const reach of [0.74, 0.8, 0.68, 0.86]) for (const elevDeg of [25, 18, 32, 10]) {
      const e = (elevDeg * Math.PI) / 180, D = reach * len;
      const dir = [dirH[0] * Math.cos(e), dirH[1] * Math.cos(e), Math.sin(e)];
      const local = [sh[0] + dir[0] * D, sh[1] + dir[1] * D, sh[2] + dir[2] * D];
      const ext = local.map((x, c) => x + dir[c] * 0.025 * this.lengthScale);
      const ok = (p) => { const r = leg.ik.probe(p); return r.error < 0.001 * this.lengthScale && r.theta.every((th, k) => th > leg.limits[k][0] + 0.08 && th < leg.limits[k][1] - 0.08); };
      if (ok(local) && ok(ext)) return { local, dir, reach, elevDeg };
    }
    const e = 0.3, D = 0.7 * len;
    return { local: [sh[0] + dirH[0] * D * Math.cos(e), sh[1] + dirH[1] * D * Math.cos(e), sh[2] + D * Math.sin(e)], dir: [dirH[0], dirH[1], 0], reach: 0.7, elevDeg: 17 };
  }
  planStartle(t) {
    const b0 = this.bodyStart, s = this.lengthScale;
    if (t > 4 && this.params.returnTo) { this.finishAction(); return this.plan(0); }
    const k = t < 0.5 ? 0 : t < 0.68 ? smooth5((t - 0.5) / 0.18) : Math.exp(-(t - 0.68) * 1.6);
    const body = { p: [b0.p[0] - 0.012 * s * k, b0.p[1], this.opts.bodyHeight + this.groundRef - 0.03 * s * k], yaw: b0.yaw, roll: 0.05 * k, pitch: 0.12 * k };
    return { body, feet: this.standFeet(), swingLeg: -1 };
  }
  planStandUp(t) {
    // body starts resting on the carapace; feet planted wide, then the body is lifted over 2.5 s
    const b0 = this.bodyStart, lie = this.params.restHeight ?? b0.p[2];
    if (t > 3.8 && this.params.returnTo) { this.finishAction(); return this.plan(0); }
    const u = smooth5((t - 0.8) / 2.5);
    const top = this.opts.bodyHeight + this.groundRef + this.heightOffset;
    return { body: { p: [b0.p[0], b0.p[1], lie + (top - lie) * u], yaw: b0.yaw, roll: 0, pitch: 0 }, feet: this.standFeet(), swingLeg: -1, level: u > 0.2 };
  }
  planSit(t) {
    if (this.pendingStand) {
      this.pendingStand = false;
      const returnTo = this.params.returnTo;
      this.setBehavior('standup', { restHeight: this.body.p[2], returnTo });
      return this.plan(0);
    }
    const b0 = this.bodyStart, low = this.params.restHeight ?? this.opts.bodyHeight * 0.55;
    if (this.params.returnTo && this.params.hold === false && t > 3) { this.finishAction(); return this.plan(0); }
    const u = smooth5(t / 2.5);
    return { body: { p: [b0.p[0], b0.p[1], this.groundRef + this.opts.bodyHeight + (low - this.opts.bodyHeight) * u], yaw: b0.yaw, roll: 0, pitch: 0 }, feet: this.standFeet(), swingLeg: -1 };
  }
  // Push one limb against an obstacle it cannot move (a stuck hatch). The fist is driven `depth` past the obstacle's
  // face, so the servos stall against it at well under the 80 % duty that trips Feetech overload protection. That is
  // the classic way hobby servos cook: full stall current, no protection. When the board sensor reaches 70 °C the
  // firmware releases torque; the controller sees the error bit, retracts the limb, rests it until the servo has cooled
  // and been re-enabled, and pushes again.
  planPress(t) {
    const limb = this.params.limb ?? 1, i = this.legs.findIndex((l) => l.L.limb === limb), leg = this.legs[i];
    const s = this.lengthScale, b0 = this.bodyStart, st = this.phaseState;
    if (!st.pose) {
      st.shift = this.supportShift(i, 0.05 * s);
      st.pose = this.gesturePose(leg);
      st.mode = 'raise'; st.since = t; st.push = 0; st.rounds = 0;
      // distance from the raised fist to the obstacle face along the limb direction (terrain perception)
      const R = this.bodyMatrix({ ...b0, roll: 0, pitch: 0 }), p0 = mat3vec(R, st.pose.local), dir = mat3vec(R, st.pose.dir);
      const start = [b0.p[0] + p0[0], b0.p[1] + p0[1], b0.p[2] + p0[2]];
      let reach = 0;
      for (let d = 0; d < 0.2 * s; d += 0.001 * s) { if (this.terrainHeight(start[0] + dir[0] * d, start[1] + dir[1] * d) > start[2]) { reach = d; break; } }
      st.travel = (reach || 0.03 * s) - leg.L.foot.radius + (this.params.depth ?? 0.03 * s);
    }
    const faulty = [...this.faults.values()].some((f) => f.limb === limb && (f.kind === 'overheat' || f.kind === 'overcurrent'));
    const age = t - st.since;
    const next = (mode, text) => { st.mode = mode; st.since = t; if (text) this.events.push({ t: this.world.time, kind: mode === 'rest' ? 'thermal' : 'gait', text }); };
    let raise = 0, push = 0;
    switch (st.mode) {
      case 'raise': raise = smooth5(age / GESTURE.raise); if (age > GESTURE.raise + 0.2) next('press', `Limb ${limb} pushing on the hatch (${(st.travel * 1000).toFixed(0)} mm stroke, blocked after ${((st.travel - (this.params.depth ?? 0.03 * s)) * 1000).toFixed(0)} mm)`); break;
      case 'press': raise = 1; push = smooth5(age / 1.5); if (faulty) next('retract', `Limb ${limb} lost torque while pushing: retracting to let the servo cool`); break;
      case 'retract': raise = 1 - smooth5(age / GESTURE.lower); push = 1 - smooth5(age / 0.6); if (age > GESTURE.lower + 0.2) next('rest'); break;
      case 'rest': raise = 0; if (!faulty && age > 2) { st.rounds++; next('raise', `Limb ${limb} cooled and re-enabled: pushing again (attempt ${st.rounds + 1})`); } break;
      default: break;
    }
    const u = smooth5(t / GESTURE.shift);
    const body = { p: [b0.p[0] + st.shift[0] * u, b0.p[1] + st.shift[1] * u, this.groundRef + this.opts.bodyHeight + this.heightOffset - 0.01 * s * u], yaw: b0.yaw, roll: 0, pitch: 0 };
    const R = this.bodyMatrix(body), local = st.pose.local.map((x, k) => x + st.pose.dir[k] * st.travel * push);
    const w = mat3vec(R, local), raised = [w[0] + body.p[0], w[1] + body.p[1], w[2] + body.p[2]];
    const feet = this.standFeet().map((f) => [...f]), sf = leg.stanceFoot;
    feet[i] = [sf[0] + (raised[0] - sf[0]) * raise, sf[1] + (raised[1] - sf[1]) * raise, sf[2] + (raised[2] - sf[2]) * raise];
    leg.contact = raise < 1e-3;
    const handover = st.mode === 'rest' ? 0 : st.mode === 'raise' ? smooth5(age / 0.3) : 1;
    return { body, feet, swingLeg: raise > 1e-3 ? i : -1, handover: { leg: i, beta: handover } };
  }

  // Fault reaction: move the COM over the four healthy feet (Chebyshev centre of their polygon) and lower the body.
  planLimp(t) {
    const b0 = this.bodyStart, s = this.lengthScale, st = this.phaseState, i = this.params.leg ?? 0;
    if (!st.shift) {
      st.shift = this.supportShift(i, 0.04 * s);
      this.events.push({ t: this.world.time, kind: 'fault', text: `Controller: shifting weight off limb ${this.legs[i].L.limb} (${(Math.hypot(...st.shift) * 1000).toFixed(0)} mm, COM ≥ 40 mm inside the four healthy feet)` });
    }
    const u = smooth5(t / 1.0);
    const body = { p: [b0.p[0] + st.shift[0] * u, b0.p[1] + st.shift[1] * u, this.groundRef + this.opts.bodyHeight - 0.012 * s * u], yaw: b0.yaw, roll: 0, pitch: 0 };
    return { body, feet: this.standFeet(), swingLeg: -1, handover: { leg: i, beta: u } };
  }

  // ---------------------------------------------------------------- gait envelope
  // Feasible cycle-time window T for the wave gait at this build: n limbs, swing share φ of a slot, a slow touchdown
  // approach of t_a seconds, and a rise/transfer taking 35 %/68 % of the remaining swing time t_s = φT/n − t_a.
  // With u the mean foot speed a servo slewing at 60 % of its velocity limit sustains on the shoulder-to-foot lever ρ
  // (quintic peak/mean ratio 1.875):
  //   stride (yaw workspace)   v·T·(1 − φ/n) ≤ S_max = 2ρ·tan(yaw room)
  //   rise   (servo speed)     h ≤ 0.35·t_s·u
  //   transfer (servo speed)   v·(T − t_x) ≤ u·t_x,   t_x = 0.68·t_s  (foot speed relative to the moving shoulder)
  // The largest feasible v is found by bisection; the cycle is the geometric mean of the window.
  walkTiming(vx = 0, vy = 0, wz = 0) {
    const n = this.gait.n, o = this.loco?.opts ?? { swingStart: 0.25, swingEnd: 0.97, approachHeight: 0.004, approachSpeed: 0.04 };
    const phi = o.swingEnd - o.swingStart, h = this.stepHeight(), ta = o.approachHeight / o.approachSpeed;
    const strideMax = this.strideLimit();
    const rho = Math.min(...this.legs.map((l) => l.homeLever));
    const u = (0.6 * this.meta.servo.maxVelocity * rho) / 1.875;
    const rMax = Math.max(...this.legs.map((l) => l.radius));
    const window = (v) => {
      const tLift = (n * ((h + o.approachHeight) / (0.35 * u) + ta)) / phi;
      const k = (u + v) * 0.68 * (phi / n) - v;                   // transfer: T·k ≥ (u + v)·0.68·t_a
      const tTransfer = v <= 1e-9 ? 0 : k > 0 ? ((u + v) * 0.68 * ta) / k : Infinity;
      const lo = Math.max(tLift, tTransfer, 2.0);
      const hi = v > 1e-6 ? strideMax / (v * (1 - phi / n)) : 8;
      return { lo, hi, ok: lo <= hi };
    };
    let a = 0, b = u;
    for (let i = 0; i < 40; i++) { const m = 0.5 * (a + b); if (window(m).ok) a = m; else b = m; }
    const vMax = a, speed = Math.hypot(vx, vy) + Math.abs(wz) * rMax, W = window(speed);
    const hi = Math.min(W.hi, 8), feasible = W.ok && speed <= vMax + 1e-9;
    const cycle = feasible ? clamp(Math.sqrt(W.lo * hi), W.lo, hi) : Math.min(W.lo, 8);
    this.walkLimits = { strideMax, uMax: u, tMin: W.lo, tMax: W.hi, vMax, feasible, cycle, stepHeight: h };
    return { cycle, slot: cycle / n, strideMax, stride: speed * cycle, vMax, tMin: W.lo, tMax: W.hi, feasible };
  }
}

function solveDense(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((row, i) => row[n] / row[i]);
}
