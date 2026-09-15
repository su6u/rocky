// Smooth rendering between physics snapshots. The worker posts frames on its own ≈60 Hz timer, which is not locked to
// the display, so drawing the newest frame directly micro-stutters. The viewer renders one snapshot interval behind
// real time and blends the two frames that bracket that moment: slerp for rotations, lerp for positions.
import { matToQuat, quatToMat } from '../engine/linalg.mjs';
import { XF_STRIDE } from '../worker/protocol.mjs';

export class FrameInterpolator {
  constructor() { this.prev = null; this.curr = null; this.prevAt = 0; this.currAt = 0; this.out = null; this.interval = 1000 / 60; }
  reset() { this.prev = null; this.curr = null; }
  push(frame, now = performance.now()) {
    // a jump backwards in sim time (restart) or a long pause makes blending meaningless
    if (this.curr && (frame.t < this.curr.t || now - this.currAt > 250)) this.prev = null;
    else { this.prev = this.curr; this.prevAt = this.currAt; }
    this.curr = frame; this.currAt = now;
    if (this.prev) this.interval += (Math.min(100, this.currAt - this.prevAt) - this.interval) * 0.1;
  }
  // Frame to draw at wall time `now`: the newest one with its transforms blended toward it.
  sample(now = performance.now()) {
    const a = this.prev, b = this.curr;
    if (!b) return null;
    if (!a || a.xf.length !== b.xf.length) return b;
    const alpha = Math.max(0, Math.min(1, (now - this.currAt) / this.interval));
    if (alpha >= 1) return b;
    if (!this.out || this.out.length !== b.xf.length) this.out = new Float64Array(b.xf.length);
    blendTransforms(a.xf, b.xf, alpha, this.out);
    return { ...b, xf: this.out };
  }
}

const qa = new Float64Array(4), qb = new Float64Array(4), R = new Float64Array(9);
export function blendTransforms(A, B, t, out) {
  for (let o = 0; o < B.length; o += XF_STRIDE) {
    qa.set(matToQuat(A.subarray(o, o + 9))); qb.set(matToQuat(B.subarray(o, o + 9)));
    slerp(qa, qb, t, qa);
    quatToMat(qa, R); out.set(R, o);
    for (let k = 9; k < XF_STRIDE; k++) out[o + k] = A[o + k] + (B[o + k] - A[o + k]) * t;
  }
  return out;
}

function slerp(a, b, t, out) {
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3], sign = 1;
  if (dot < 0) { dot = -dot; sign = -1; }                          // take the short way round
  let wa = 1 - t, wb = t * sign;
  if (dot < 0.9995) { const th = Math.acos(dot), s = Math.sin(th); wa = Math.sin((1 - t) * th) / s; wb = (Math.sin(t * th) / s) * sign; }
  for (let k = 0; k < 4; k++) out[k] = wa * a[k] + wb * b[k];
  const n = Math.hypot(out[0], out[1], out[2], out[3]);
  for (let k = 0; k < 4; k++) out[k] /= n;
  return out;
}
