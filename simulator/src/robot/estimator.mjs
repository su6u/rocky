// State estimation from the sensors a physical Rocky would carry, so the controller never reads the simulator's truth:
//   • 12-bit magnetic encoders in every servo (quantised joint angles; backlash sits after the encoder)
//   • a 6-axis IMU in the carapace (fused roll/pitch with noise, gyro heading)
//   • force-sensitive resistors in the TPU foot pads (binary contact with hysteresis)
// Body pose = IMU attitude + leg odometry. With the stance feet fixed where they touched down, the body position that
// best explains the measured leg geometry is the least-squares solution
//   p̂ = mean_i ( f_i^w − R̂ · FK_i(θ̂_i) )
// which removes the servo sag that makes a purely planned body pose sit lower than commanded.
import { mat3mul, axisAngleMat, mat3vec } from '../engine/linalg.mjs';

export const ENCODER_COUNTS = 4096;
const TAU = Math.PI * 2;

export function rpyMatrix(roll, pitch, yaw) {
  return mat3mul(axisAngleMat([0, 0, 1], yaw), mat3mul(axisAngleMat([0, 1, 0], pitch), axisAngleMat([1, 0, 0], roll)));
}

export class Estimator {
  constructor(world, model, legs, { rng, imuNoiseDeg = 0.15, contactOn = 1.2, contactOff = 0.4, forceRange = 30, smoothing = 0.04 } = {}) {
    Object.assign(this, { world, model, legs, rng, imuNoise: (imuNoiseDeg * Math.PI) / 180, contactOn, contactOff, forceRange, smoothing });
    this.contact = legs.map(() => false);
    this.forceFiltered = legs.map(() => 0);
    this.actuatorByDof = new Map(world.actuators.map((a) => [a.dof, a]));
    this.poseFiltered = null;
  }
  encoder(bodyIndex) {
    const w = this.world, dof = this.model.dof[bodyIndex], act = this.actuatorByDof.get(dof);
    const q = w.q[dof + 1] + (act && act.backlashDof >= 0 ? w.q[act.backlashDof + 1] : 0);
    const step = TAU / ENCODER_COUNTS;
    return Math.round(q / step) * step;
  }
  encoders(leg) { return [this.encoder(leg.L.yaw), this.encoder(leg.L.pitch), this.encoder(leg.L.elbow)]; }
  imu() {
    const [w, x, y, z] = [this.world.q[3], this.world.q[4], this.world.q[5], this.world.q[6]];
    const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
    const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))));
    const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    const n = this.imuNoise, g = this.rng ? this.rng.normal : () => 0;
    return { roll: roll + n * g(), pitch: pitch + n * g(), yaw: yaw + 0.3 * n * g() };
  }
  trueFootForce(i) {
    const name = `foot${this.legs[i].L.limb}`;
    let fn = 0;
    for (const c of this.world.contacts) if (c.name === name && c.kind === 'foot') fn += c.fn ?? 0;
    return fn;
  }
  // Foot switch in the pad, sampled at the control rate, with hysteresis against chatter.
  footContact(i) {
    const fn = this.trueFootForce(i);
    this.contact[i] = this.contact[i] ? fn > this.contactOff : fn > this.contactOn;
    return this.contact[i];
  }
  // The same pad read as an analog force (FSR behind the TPU sole): ±5 % gain error per sample, 0.05 N noise,
  // saturation at the sensor range, and a one-pole 40 ms filter. Call once per control tick.
  footForce(i, dt) {
    const g = this.rng ? this.rng.normal : () => 0;
    const raw = Math.min(this.forceRange, Math.max(0, this.trueFootForce(i) * (1 + 0.05 * g()) + 0.05 * g()));
    this.forceFiltered[i] += (raw - this.forceFiltered[i]) * (1 - Math.exp(-dt / this.smoothing));
    return this.forceFiltered[i];
  }
  // stance: [{ leg, foot: world position where that foot is planted }]; returns null with fewer than 3 feet.
  bodyPose(stance, imu = this.imu()) {
    if (stance.length < 3) return null;
    const R = rpyMatrix(imu.roll, imu.pitch, imu.yaw), p = [0, 0, 0];
    for (const { leg, foot } of stance) {
      const f = mat3vec(R, leg.ik.fk(this.encoders(leg)).foot);
      p[0] += foot[0] - f[0]; p[1] += foot[1] - f[1]; p[2] += foot[2] - f[2];
    }
    const k = 1 / stance.length, pose = { p: [p[0] * k, p[1] * k, p[2] * k], roll: imu.roll, pitch: imu.pitch, yaw: imu.yaw, R };
    // one-pole smoothing (≈50 ms at the 50 Hz bus rate) removes IMU and encoder-quantisation jitter from swing targets
    const f = this.poseFiltered;
    if (f && Math.hypot(f.p[0] - pose.p[0], f.p[1] - pose.p[1], f.p[2] - pose.p[2]) < 0.02) {
      const a = 0.33;
      for (const key of ['roll', 'pitch', 'yaw']) pose[key] = f[key] + (pose[key] - f[key]) * a;
      pose.p = pose.p.map((x, c) => f.p[c] + (x - f.p[c]) * a);
      pose.R = rpyMatrix(pose.roll, pose.pitch, pose.yaw);
    }
    this.poseFiltered = pose;
    return pose;
  }
}
