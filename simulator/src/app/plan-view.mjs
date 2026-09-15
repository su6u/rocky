// 2D kinematics pane: top (x–y) or side (x–z) projection of the live physics state. Everything drawn comes from
// the simulator snapshot: joint positions, measured contacts, the loaded support polygon, COM, ZMP and capture point.
const COLORS = {
  gridMajor: '#403b3b', grid: '#302c2c', ground: '#8f8985', body: '#4c4747', bodyEdge: '#8d8783',
  stance: '#f4f0eb', swing: '#c8bfba', planned: '#aaa4a0', failed: '#ff8e82', support: 'rgba(244, 240, 235, 0.09)', supportEdge: '#c4bdb8',
  com: '#fffdf9', zmp: '#d8c49f', capture: '#e8a69d', label: '#aaa4a0', force: '#ddd7d2', friction: '#d8c49f',
};

export class PlanView {
  constructor(canvas) {
    this.canvas = canvas;
    this.mode = 'top';
    this.info = null; this.outline = null; this.center = null;
  }

  setMode(mode) { this.mode = mode; this.center = null; }

  // outline: body-frame torso silhouette points [[x, y, z], …] in metres (computed once from the part mesh).
  setInfo(info, outline) { this.info = info; this.outline = outline; this.center = null; }

  draw(frame) {
    const { canvas } = this;
    const rect = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.round(rect.width * dpr), H = Math.round(rect.height * dpr);
    if (!W || !H) return;
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    const c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, rect.width, rect.height);
    const info = this.info;
    if (!info || !frame) return;
    const w = rect.width, h = rect.height, top = this.mode === 'top';
    const span = info.carapaceAcross, ls = info.lengthScale;
    const ppm = Math.min(w / (3.5 * span), (h - 96) / (top ? 3.3 * span : 2.4 * span));
    const slope = info.terrain.slope, tanS = Math.tan(slope);
    // camera follows the body smoothly
    const want = top ? [frame.com[0], frame.com[1]] : [frame.com[0], 0.55 * span + tanS * frame.com[0]];
    this.center = this.center ? [this.center[0] + (want[0] - this.center[0]) * 0.15, this.center[1] + (want[1] - this.center[1]) * 0.15] : want;
    const cx = w / 2, cy = top ? h * 0.52 : h * 0.56;
    const X = (p) => cx + (p[0] - this.center[0]) * ppm;
    const Y = (p) => cy - ((top ? p[1] : p[2]) - this.center[1]) * ppm;
    const P = (p) => [X(p), Y(p)];

    // grid (world-fixed), 5 cm at desk scale
    const step = 0.05 * ls;
    c.lineWidth = 1;
    const x0 = this.center[0] - w / 2 / ppm, x1 = this.center[0] + w / 2 / ppm;
    const y0 = this.center[1] - h / 2 / ppm, y1 = this.center[1] + h / 2 / ppm;
    for (let gx = Math.floor(x0 / step) * step; gx <= x1; gx += step) {
      const sx = cx + (gx - this.center[0]) * ppm, major = Math.abs(Math.round(gx / step) % 4) === 0;
      c.strokeStyle = major ? COLORS.gridMajor : COLORS.grid; c.beginPath(); c.moveTo(sx, 0); c.lineTo(sx, h); c.stroke();
    }
    for (let gy = Math.floor(y0 / step) * step; gy <= y1; gy += step) {
      const sy = cy - (gy - this.center[1]) * ppm, major = Math.abs(Math.round(gy / step) % 4) === 0;
      c.strokeStyle = major ? COLORS.gridMajor : COLORS.grid; c.beginPath(); c.moveTo(0, sy); c.lineTo(w, sy); c.stroke();
    }

    // terrain: slippery patches (top view) and blocks (both views), exactly the shapes the solver collides with
    const T = info.terrain || {};
    if (top) for (const q of T.patches ?? []) {
      c.fillStyle = 'rgba(160, 196, 214, 0.10)'; c.strokeStyle = 'rgba(160, 196, 214, 0.45)'; c.setLineDash([2, 3]);
      c.fillRect(X([q.min[0]]), Y([0, q.max[1]]), (q.max[0] - q.min[0]) * ppm, (q.max[1] - q.min[1]) * ppm);
      c.strokeRect(X([q.min[0]]), Y([0, q.max[1]]), (q.max[0] - q.min[0]) * ppm, (q.max[1] - q.min[1]) * ppm); c.setLineDash([]);
      c.fillStyle = 'rgba(180, 210, 224, 0.75)'; c.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      c.fillText(q.label || `μ ${q.mu}`, X([q.min[0]]) + 6, Y([0, q.max[1]]) + 14);
    }
    for (const b of T.boxes ?? []) {
      const x0b = X([b.min[0]]), w = (b.max[0] - b.min[0]) * ppm;
      const yTop = top ? Y([0, b.max[1]]) : Y([0, 0, b.max[2]]), h = top ? (b.max[1] - b.min[1]) * ppm : (b.max[2] - b.min[2]) * ppm;
      c.fillStyle = 'rgba(128, 122, 112, 0.28)'; c.fillRect(x0b, yTop, w, h);
      c.strokeStyle = 'rgba(200, 190, 175, 0.7)'; c.lineWidth = 1.2; c.strokeRect(x0b, yTop, w, h);
      c.fillStyle = 'rgba(214, 204, 190, 0.85)'; c.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      c.fillText(b.label || `${((b.max[2] - b.min[2]) * 1000).toFixed(0)} mm`, x0b + 5, yTop + (top ? 13 : -5));
    }

    const perFoot = info.weight / 5;
    const footLoad = (limb) => frame.contacts.filter((k) => k.kind === 'foot' && k.name === `foot${limb}`).reduce((s, k) => s + k.fn, 0);
    const limbFailed = (limb) => frame.servos.some((s) => s.limb === limb && (s.failed || s.protected));

    if (!top) {
      // ground line (tilted plane z = x·tanθ)
      c.strokeStyle = COLORS.ground; c.lineWidth = 1.5;
      c.beginPath(); c.moveTo(0, Y([x0, 0, tanS * x0])); c.lineTo(w, Y([x1, 0, tanS * x1])); c.stroke();
      c.fillStyle = 'rgba(111, 143, 125, 0.07)';
      c.beginPath(); c.moveTo(0, Y([x0, 0, tanS * x0])); c.lineTo(w, Y([x1, 0, tanS * x1])); c.lineTo(w, h); c.lineTo(0, h); c.fill();
    }

    // support polygon from loaded feet
    const poly = frame.stab.polygon || [];
    if (top && poly.length >= 3) {
      c.beginPath(); poly.forEach((q, i) => (i ? c.lineTo(X(q), Y(q)) : c.moveTo(X(q), Y(q)))); c.closePath();
      c.fillStyle = COLORS.support; c.fill();
      c.setLineDash([5, 5]); c.strokeStyle = COLORS.supportEdge; c.lineWidth = 1.2; c.stroke(); c.setLineDash([]);
    }

    // carapace silhouette
    const R = frame.xf.subarray(0, 9), o = [frame.xf[9], frame.xf[10], frame.xf[11]];
    if (this.outline) {
      const pts = this.outline[top ? 'top' : 'side'].map((q) => {
        const wpt = [R[0] * q[0] + R[1] * q[1] + R[2] * q[2] + o[0], R[3] * q[0] + R[4] * q[1] + R[5] * q[2] + o[1], R[6] * q[0] + R[7] * q[1] + R[8] * q[2] + o[2]];
        return P(wpt);
      });
      c.beginPath(); pts.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.closePath();
    } else {
      c.beginPath(); c.arc(X(o), Y(o), 0.5 * span * ppm, 0, Math.PI * 2);
    }
    c.fillStyle = 'rgba(47, 79, 69, 0.55)'; c.fill(); c.strokeStyle = COLORS.bodyEdge; c.lineWidth = 1.2; c.stroke();

    // limbs
    info.limbs.forEach((L, i) => {
      const s = frame.limbs[i], load = footLoad(L.limb), loaded = load > 0.05 * perFoot;
      const color = limbFailed(L.limb) ? COLORS.failed : loaded ? COLORS.stance : s.planned ? COLORS.planned : COLORS.swing;
      const a = P(s.shoulder), b = P(s.elbow), f = P(s.foot);
      c.lineCap = 'round'; c.lineJoin = 'round';
      c.beginPath(); c.moveTo(...a); c.lineTo(...b); c.lineTo(...f);
      c.strokeStyle = '#1a1818'; c.lineWidth = 7; c.stroke();
      c.strokeStyle = color; c.lineWidth = 2; c.stroke();
      for (const q of [a, b]) { c.beginPath(); c.arc(q[0], q[1], 3.2, 0, Math.PI * 2); c.fillStyle = '#1a1818'; c.fill(); c.strokeStyle = color; c.lineWidth = 1.3; c.stroke(); }
      const r = Math.max(3, L.footRadius * ppm);
      c.beginPath(); c.arc(f[0], f[1], r, 0, Math.PI * 2);
      c.fillStyle = loaded ? color : '#1a1818'; c.globalAlpha = loaded ? 0.85 : 1; c.fill(); c.globalAlpha = 1;
      c.strokeStyle = color; c.lineWidth = 1.3; c.stroke();
      if (!s.planned && s.target) {
        const t = P(s.target);
        c.setLineDash([3, 3]); c.beginPath(); c.arc(t[0], t[1], r + 3, 0, Math.PI * 2); c.strokeStyle = COLORS.swing; c.stroke(); c.setLineDash([]);
      }
      // label outward from the body
      const dx = f[0] - X(o), dy = f[1] - Y(o), dl = Math.hypot(dx, dy) || 1;
      c.fillStyle = color; c.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      c.fillText(`L${L.limb}`, clamp(f[0] + (dx / dl) * 16 - 7, 6, w - 26), clamp(f[1] + (top ? (dy / dl) * 16 : 18) + 3, 14, h - 8));
      // side view: ground reaction force at the foot (normal up, tangential along x)
      if (!top && load > 0.02 * perFoot) {
        const fx = frame.contacts.filter((k) => k.kind === 'foot' && k.name === `foot${L.limb}`).reduce((acc, k) => [acc[0] + k.f[0], acc[1] + k.f[2]], [0, 0]);
        const k = (0.45 * span * ppm) / perFoot;
        arrow(c, f[0], f[1] + r, f[0] + fx[0] * k, f[1] + r - fx[1] * k, COLORS.force);
      }
    });

    // shell contacts (anything other than feet touching the ground)
    for (const k of frame.contacts) {
      if (k.kind !== 'shell' || k.fn < 0.01 * perFoot) continue;
      const q = P(k.p); c.beginPath(); c.arc(q[0], q[1], 5, 0, Math.PI * 2); c.strokeStyle = COLORS.failed; c.lineWidth = 2; c.stroke();
    }

    // COM, ZMP, capture point
    const com = frame.com;
    const comP = top ? P(frame.stab.comProjection || com) : P(com);
    if (!top) { const g = P([com[0], com[1], tanS * com[0]]); c.setLineDash([2, 4]); c.strokeStyle = COLORS.com; c.globalAlpha = 0.5; c.beginPath(); c.moveTo(...comP); c.lineTo(...g); c.stroke(); c.setLineDash([]); c.globalAlpha = 1; }
    if (frame.stab.capture) {
      const cp = P(top ? frame.stab.capture : [frame.stab.capture[0], 0, tanS * frame.stab.capture[0]]);
      c.strokeStyle = COLORS.capture; c.lineWidth = 1.2; c.beginPath(); c.moveTo(...(top ? comP : P([com[0], 0, tanS * com[0]]))); c.lineTo(...cp); c.stroke();
      c.beginPath(); c.arc(cp[0], cp[1], 5, 0, Math.PI * 2); c.stroke();
    }
    if (frame.stab.zmp) {
      const z = P(top ? frame.stab.zmp : [frame.stab.zmp[0], 0, tanS * frame.stab.zmp[0]]);
      c.fillStyle = COLORS.zmp; c.beginPath(); c.moveTo(z[0], z[1] - 5); c.lineTo(z[0] + 5, z[1]); c.lineTo(z[0], z[1] + 5); c.lineTo(z[0] - 5, z[1]); c.closePath(); c.fill();
    }
    c.strokeStyle = COLORS.com; c.lineWidth = 1.6;
    c.beginPath(); c.arc(comP[0], comP[1], 7, 0, Math.PI * 2); c.moveTo(comP[0] - 11, comP[1]); c.lineTo(comP[0] + 11, comP[1]); c.moveTo(comP[0], comP[1] - 11); c.lineTo(comP[0], comP[1] + 11); c.stroke();

    // axis legend
    c.fillStyle = COLORS.label; c.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    c.fillText(top ? 'x →   y ↑   (top)' : 'x →   z ↑   (side, looking +y)', 16, h - 16);
  }
}

// Silhouettes of the torso part in its body frame (metres), from the baked/STL mesh: convex hulls in x–y and x–z.
export function torsoOutline(geometry, partToBody, scale) {
  const pos = geometry.attributes.position, P = partToBody.R.flat(), t = partToBody.t;
  const pts = [];
  for (let i = 0; i < pos.count; i += 7) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    pts.push([(P[0] * x + P[1] * y + P[2] * z + t[0]) * scale, (P[3] * x + P[4] * y + P[5] * z + t[1]) * scale, (P[6] * x + P[7] * y + P[8] * z + t[2]) * scale]);
  }
  const zMid = pts.reduce((s, p) => s + p[2], 0) / pts.length;
  return {
    top: hull(pts, 0, 1).map((p) => [p[0], p[1], zMid]),
    side: hull(pts, 0, 2).map((p) => [p[0], 0, p[2]]),
  };
}

function hull(points, a, b) {
  const p = [...points].sort((u, v) => u[a] - v[a] || u[b] - v[b]);
  const cross = (o, u, v) => (u[a] - o[a]) * (v[b] - o[b]) - (u[b] - o[b]) * (v[a] - o[a]);
  const lower = [], upper = [];
  for (const q of p) { while (lower.length > 1 && cross(lower.at(-2), lower.at(-1), q) <= 0) lower.pop(); lower.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (upper.length > 1 && cross(upper.at(-2), upper.at(-1), q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function arrow(c, x0, y0, x1, y1, color) {
  const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
  if (L < 2) return;
  const ux = dx / L, uy = dy / L, hl = Math.min(8, 0.35 * L);
  c.strokeStyle = color; c.lineWidth = 1.4;
  c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1);
  c.moveTo(x1, y1); c.lineTo(x1 - ux * hl - uy * hl * 0.45, y1 - uy * hl + ux * hl * 0.45);
  c.moveTo(x1, y1); c.lineTo(x1 - ux * hl + uy * hl * 0.45, y1 - uy * hl - ux * hl * 0.45);
  c.stroke();
}

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
