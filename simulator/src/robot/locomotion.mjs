// Velocity-commanded, statically stable pentapod wave gait (one limb swings at a time).
//
// The body follows a reference pose integrated from a smoothed velocity command (acceleration-limited, so a new command
// never jerks the carapace). Each slot of the gait:
//   1. shift   the body sways toward the Chebyshev centre of the four feet that stay down (largest inscribed circle
//              of the next support polygon), blended with a raised cosine so the sway velocity is continuous;
//   2. swing   the free limb lifts vertically, transfers at clearance above the highest terrain along its path, and
//              descends vertically to a few millimetres above the expected ground;
//   3. land    it then approaches at a slow constant speed until the pad's force sensor fires (touchdown search).
//              The 1:345 gear train reflects ≈1–2 kg of rotor inertia to each foot, and the servo's P loop is
//              underdamped (ζ ≈ 0.4) and trails its goal by ≈30 ms. A trajectory that merely ends at the ground still
//              arrives at ≈0.25 m/s: a 50 N impact that bounces the carapace and unloads three feet. The foot is planted
//              where it actually touched, so terrain height errors cannot build up as pre-load in the stance legs.
// Touchdown ends on the pad sensor, not on the clock; the slot simply lasts until the foot is down.
// Footholds use Raibert-style placement: the neutral stance position ("home") at the body pose predicted for the
// middle of the coming stance phase. Each foothold is checked with the true limb IK at touchdown and at liftoff and
// pulled back toward the neutral point until both are reachable with joint-limit margin.
import { hull2d, chebyshevCenter } from '../engine/stability.mjs';
import { phasedSwing, predictPose, SWING_PHASES } from './gait.mjs';

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

export class Locomotion {
  constructor(ctrl, opts = {}) {
    const s = ctrl.lengthScale;
    this.ctrl = ctrl;
    this.opts = {
      swingStart: 0.25, swingEnd: 0.97,
      accel: 0.05 * s, yawAccel: 0.25,              // m/s², rad/s²
      settleTolerance: 0.008 * s,                   // feet farther than this from neutral trigger settling steps
      approachHeight: 0.004 * s, approachSpeed: 0.04 * s, searchMax: 0.025 * s, servoLag: 0.03,
      swayGain: 0.35, swayCap: 0.012 * s, swayBlend: 1.0,   // gentle sway: a fast one drags lightly loaded feet
      ...opts,
    };
    this.cmd = { vx: 0, vy: 0, wz: 0 };
    this.vel = { vx: 0, vy: 0, wz: 0 };
    this.pose = { x: 0, y: 0, yaw: 0 };
    this.slot = null; this.k = 0; this.lastT = 0;
    this.sway = [0, 0];
    this.log = [];                                  // {t, kind, text} for the event stream
  }

  reset(t, pose) {
    this.pose = { x: pose.x, y: pose.y, yaw: pose.yaw };
    this.vel = { vx: 0, vy: 0, wz: 0 };
    this.cmd = { vx: 0, vy: 0, wz: 0 };
    this.slot = null; this.lastT = t; this.sway = [0, 0];
  }

  setCommand({ vx = 0, vy = 0, wz = 0 }) { this.cmd = { vx, vy, wz }; }
  commanded() { return Math.abs(this.cmd.vx) + Math.abs(this.cmd.vy) + Math.abs(this.cmd.wz) > 1e-6; }
  moving() { return this.commanded() || Math.abs(this.vel.vx) + Math.abs(this.vel.vy) + Math.abs(this.vel.wz) > 1e-5; }
  get idle() { return !this.slot; }

  // Body pose τ seconds ahead, with the velocity still ramping toward the command at the acceleration limits.
  predict(tau) {
    let pose = { ...this.pose };
    const vel = { ...this.vel }, N = Math.max(1, Math.ceil(tau / 0.05)), h = tau / N, o = this.opts;
    for (let i = 0; i < N; i++) {
      vel.vx += clamp(this.cmd.vx - vel.vx, -o.accel * h, o.accel * h);
      vel.vy += clamp(this.cmd.vy - vel.vy, -o.accel * h, o.accel * h);
      vel.wz += clamp(this.cmd.wz - vel.wz, -o.yawAccel * h, o.yawAccel * h);
      pose = predictPose(pose, vel, h);
    }
    return pose;
  }

  neutral(leg, pose) {
    const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw), h = leg.home;
    return [pose.x + c * h[0] - s * h[1], pose.y + s * h[0] + c * h[1]];
  }

  footOffset(i) {
    const leg = this.ctrl.legs[i], n = this.neutral(leg, this.pose);
    return Math.hypot(leg.stanceFoot[0] - n[0], leg.stanceFoot[1] - n[1]);
  }

  update(t) {
    const dt = Math.max(0, t - this.lastT), o = this.opts;
    this.lastT = t;
    this.vel.vx += clamp(this.cmd.vx - this.vel.vx, -o.accel * dt, o.accel * dt);
    this.vel.vy += clamp(this.cmd.vy - this.vel.vy, -o.accel * dt, o.accel * dt);
    this.vel.wz += clamp(this.cmd.wz - this.vel.wz, -o.yawAccel * dt, o.yawAccel * dt);
    this.pose = predictPose(this.pose, this.vel, dt);
    const legs = this.ctrl.legs;
    if (!this.slot) this.beginSlot(t);
    let swingLeg = -1;
    const feet = legs.map((l) => l.stanceFoot);
    const S = this.slot;
    if (S) {
      const u = (t - S.t0) / S.T;
      const blend = u < o.swayBlend ? 0.5 - 0.5 * Math.cos((Math.PI * u) / o.swayBlend) : 1;
      this.sway = [S.swayFrom[0] + (S.swayTo[0] - S.swayFrom[0]) * blend, S.swayFrom[1] + (S.swayTo[1] - S.swayFrom[1]) * blend];
      if (t >= S.tLift && !S.done) {
        const leg = legs[S.leg], est = this.ctrl.estimator;
        const dts = Math.max(0, t - (S.lastT ?? S.tLift));
        S.lastT = t;
        let foot;
        if (S.clock < 1) {
          S.clock = Math.min(1, S.clock + dts / S.phasedTime);
          if (S.clock >= 1) S.tApproach = t;
          foot = phasedSwing(S.liftoff, S.pre, S.apexZ, S.clock);
        } else {
          const z = S.pre[2] - o.approachSpeed * (t - S.tApproach);
          foot = [S.pre[0], S.pre[1], Math.max(z, S.touchdown[2] - o.searchMax)];
          if (z <= S.touchdown[2] - o.searchMax) S.exhausted = true;
        }
        foot[2] -= this.ctrl.swingHeightCorrection(S.clock);     // the real body sits lower than planned
        // listen for touchdown from the start of the descent: servo sag or a higher floor can make the foot land early,
        // and a leg that keeps following its descent after touching pushes the carapace up and unloads two other feet
        const sensing = S.clock >= SWING_PHASES.descentStart && est.footContact(S.leg);
        if (sensing || S.exhausted) {
          S.done = true;
          // the servo trails its goal by ≈ v·lag; plant the foot where it is, not where the goal already went
          if (sensing && S.clock >= 1) foot[2] = Math.min(S.pre[2], foot[2] + o.approachSpeed * o.servoLag);
          leg.stanceFoot = foot;
          feet[S.leg] = foot;                        // (the array was filled before the swing update)
          leg.lastTouchdown = { t, planned: S.touchdown, actual: foot, sensed: sensing };
          if (!sensing) this.log.push({ t, kind: 'terrain', text: `L${leg.L.limb} found no ground within ${(o.searchMax * 1000).toFixed(0)} mm below the planned foothold` });
        } else { swingLeg = S.leg; feet[S.leg] = foot; }
      }
      if (u >= 1 && S.done) { this.k++; this.slot = null; this.beginSlot(t); }
    }
    return { pose: { ...this.pose }, vel: { ...this.vel }, sway: [...this.sway], feet, swingLeg, idle: !this.slot, slot: this.slot };
  }

  beginSlot(t) {
    const ctrl = this.ctrl, o = this.opts, legs = ctrl.legs, n = legs.length;
    let leg = -1;
    if (this.moving()) this.settleSteps = 0;
    const settling = !this.moving() && (this.settleSteps ?? 0) < n;   // at most one settling cycle
    for (let tries = 0; tries < n; tries++) {
      const cand = ctrl.gait.order[this.k % n];
      if (this.moving() || (settling && this.footOffset(cand) > o.settleTolerance)) { leg = cand; break; }
      this.k++;
    }
    if (leg < 0) { this.sway = [this.sway[0] * 0.98, this.sway[1] * 0.98]; return; }
    if (!this.moving()) this.settleSteps = (this.settleSteps ?? 0) + 1;
    const timing = ctrl.walkTiming(this.cmd.vx, this.cmd.vy, this.cmd.wz);
    if (!this.moving()) timing.slot = timing.tMin / n;              // settling steps use the quickest safe timing
    const T = timing.slot, L = legs[leg], swingTime = (o.swingEnd - o.swingStart) * T;
    const approachTime = Math.min(0.4 * swingTime, o.approachHeight / o.approachSpeed);
    const tTouch = o.swingEnd * T, stance = (n - (o.swingEnd - o.swingStart)) * T;
    const poseTouch = this.predict(tTouch), poseMid = this.predict(tTouch + stance / 2), poseLift = this.predict(tTouch + stance);
    const target = this.neutral(L, poseMid), fallback = this.neutral(L, poseTouch);
    const reachable = (xy) => ctrl.footReachable(L, xy, poseTouch) && ctrl.footReachable(L, xy, poseLift);
    let xy = target;
    if (!reachable(target)) {
      let lo = 0, hi = 1;
      for (let i = 0; i < 7; i++) {
        const mid = 0.5 * (lo + hi), cand = [fallback[0] + (target[0] - fallback[0]) * mid, fallback[1] + (target[1] - fallback[1]) * mid];
        if (reachable(cand)) lo = mid; else hi = mid;
      }
      xy = [fallback[0] + (target[0] - fallback[0]) * lo, fallback[1] + (target[1] - fallback[1]) * lo];
      if (lo < 0.999) this.log.push({ t, kind: 'gait', text: `L${L.L.limb} step shortened to ${(lo * 100).toFixed(0)} % to stay inside its joint limits` });
    }
    const r = L.L.foot.radius, terrain = ctrl.world.terrain;
    // clearance a round foot needs from a block edge: its radius plus the servos' horizontal tracking error
    const edgeClear = Math.max(2.5 * r, r + 0.015 * ctrl.lengthScale);
    if (terrain.boxes.length && ctrl.terrainEdge(xy[0], xy[1]) < edgeClear) {
      // keep the foot clearly on or clearly off a block: slide the foothold along the stride until it is clear
      const dir = [xy[0] - L.stanceFoot[0], xy[1] - L.stanceFoot[1]], len = Math.hypot(dir[0], dir[1]) || 1;
      for (const k of [1, -1, 2, -2, 3, -3]) {
        const cand = [xy[0] + (dir[0] / len) * k * edgeClear, xy[1] + (dir[1] / len) * k * edgeClear];
        if (ctrl.terrainEdge(cand[0], cand[1]) >= edgeClear && reachable(cand)) { xy = cand; break; }
      }
    }
    const ground = [xy[0], xy[1], ctrl.footZ(xy[0], xy[1], r)];
    // only the height of the attitude-trimmed frame matters for landing; keeping x, y exact keeps footholds on neutral
    const trim = ctrl.toCommandFrame(ground)[2] - ground[2], touchdown = [ground[0], ground[1], ground[2] + trim];
    const liftoff = [...L.stanceFoot];
    const terrainTop = ctrl.terrainMaxAlong(liftoff, ground, edgeClear) + r / Math.cos(terrain.slope) + Math.max(0, trim);
    const apexZ = Math.max(liftoff[2], touchdown[2], terrainTop) + ctrl.stepHeight();
    const support = legs.map((l, i) => (i === leg ? null : [l.stanceFoot[0], l.stanceFoot[1]])).filter(Boolean);
    const cc = chebyshevCenter(hull2d(support).map((q) => [q[0], q[1]]));
    const pc = this.predict(0.5 * T);
    const swayTo = cc ? [clamp((cc[0] - pc.x) * o.swayGain, -o.swayCap, o.swayCap), clamp((cc[1] - pc.y) * o.swayGain, -o.swayCap, o.swayCap)] : [0, 0];
    const pre = [touchdown[0], touchdown[1], touchdown[2] + approachTime * o.approachSpeed];
    const tLift = t + o.swingStart * T;
    this.slot = { k: this.k, leg, t0: t, T, tLift, tApproach: Infinity, phasedTime: swingTime - approachTime, clock: 0, liftoff, touchdown, pre, apexZ: Math.max(apexZ, pre[2] + 0.002 * ctrl.lengthScale), swayFrom: [...this.sway], swayTo, done: false, exhausted: false, timing };
  }
}
