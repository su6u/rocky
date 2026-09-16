// Gait geometry for a statically stable pentapod: swing order, swing-foot trajectories and timing helpers.
import { hull2d, polygonMargin, chebyshevCenter } from '../engine/stability.mjs';

// Quintic smoothstep: 0 → 1 with zero velocity and acceleration at both ends.
export const smooth5 = (u) => { u = Math.max(0, Math.min(1, u)); return u * u * u * (10 + u * (-15 + 6 * u)); };

// Swing order for a wave gait (one limb swings at a time), chosen once by exhaustive search over the cyclic orders:
// maximise the worst support-polygon margin (Chebyshev radius of the four stance feet) while the body travels a stride
// along +x. The kit's irregular shoulder layout makes the best order non-obvious.
export function planWaveOrder(homes, stride = 0.08) {
  const n = homes.length, idx = [...Array(n).keys()], perms = [];
  const permute = (arr, l) => {
    if (l === arr.length) { if (arr[0] === 0) perms.push([...arr]); return; }
    for (let i = l; i < arr.length; i++) { [arr[l], arr[i]] = [arr[i], arr[l]]; permute(arr, l + 1); [arr[l], arr[i]] = [arr[i], arr[l]]; }
  };
  permute(idx, 0);
  let best = null;
  for (const order of perms) {
    let worst = Infinity;
    for (let s = 0; s < n; s++) {
      const swing = order[s];
      const pts = homes.map((h, i) => {
        const slotsSinceTouchdown = (((s - order.indexOf(i)) % n) + n) % n; // 0 = just landed
        return [h[0] + stride * (0.5 - slotsSinceTouchdown / (n - 1)), h[1]];
      }).filter((_, i) => i !== swing);
      const hull = hull2d(pts).map((p) => [p[0], p[1]]);
      const cc = chebyshevCenter(hull);
      worst = Math.min(worst, cc ? cc[2] : polygonMargin(hull, 0, 0));
    }
    if (!best || worst > best.margin) best = { order, margin: worst };
  }
  return { n, order: best.order, plannedMargin: best.margin };
}

// Swing-foot path shaped for a servo with ≈30 ms lag and an underdamped (ζ ≈ 0.4, ≈5 Hz) position loop: every segment
// lasts several servo time constants and the vertical motion is slow where the foot meets the ground.
//   s ∈ [0, 0.35]    rise to the apex (quintic)
//   s ∈ [0.12, 0.8]  horizontal transfer (quintic: it starts gently while the foot is already rising and is finished
//                    before the descent reaches the ground, so an early contact from servo sag cannot scuff)
//   s ∈ [0.5, 1]     descend to `end`, which the planner puts a few millimetres above the expected ground; the final
//                    approach from there is a slow constant-speed search driven by the pad sensor (see locomotion.mjs)
export const SWING_PHASES = { riseEnd: 0.35, transferStart: 0.12, transferEnd: 0.8, descentStart: 0.5 };
export function phasedSwing(start, end, apexZ, s, out = [0, 0, 0]) {
  const P = SWING_PHASES;
  const hx = smooth5((s - P.transferStart) / (P.transferEnd - P.transferStart));
  const up = smooth5(s / P.riseEnd), down = smooth5((s - P.descentStart) / (1 - P.descentStart));
  out[0] = start[0] + (end[0] - start[0]) * hx;
  out[1] = start[1] + (end[1] - start[1]) * hx;
  out[2] = start[2] + (apexZ - start[2]) * up + (end[2] - apexZ) * down;
  return out;
}

// Body pose after τ seconds of constant body-frame velocity (vx, vy, wz), integrated exactly along the arc.
export function predictPose(pose, vel, tau) {
  const { vx, vy, wz } = vel, psi0 = pose.yaw, psi1 = psi0 + wz * tau;
  if (Math.abs(wz) < 1e-9) {
    const c = Math.cos(psi0), s = Math.sin(psi0);
    return { x: pose.x + (vx * c - vy * s) * tau, y: pose.y + (vx * s + vy * c) * tau, yaw: psi1 };
  }
  const S = (Math.sin(psi1) - Math.sin(psi0)) / wz, C = (Math.cos(psi0) - Math.cos(psi1)) / wz;
  return { x: pose.x + vx * S - vy * C, y: pose.y + vx * C + vy * S, yaw: psi1 };
}
