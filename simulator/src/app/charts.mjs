// Small rolling time-series charts on canvas (no dependencies).
export class Chart {
  // series: [{ key, color, dash?, width? }] · window: seconds shown · floor/ceiling: optional fixed y bounds
  constructor(canvas, { series, window = 12, floor = null, ceiling = null, minRange = 1, format = (v) => v.toFixed(0) }) {
    Object.assign(this, { canvas, series, window, floor, ceiling, minRange, format });
    this.t = []; this.data = Object.fromEntries(series.map((s) => [s.key, []]));
  }
  reset() { this.t.length = 0; for (const k in this.data) this.data[k].length = 0; }
  push(t, values) {
    if (this.t.length && t < this.t.at(-1)) this.reset();
    this.t.push(t);
    for (const s of this.series) this.data[s.key].push(Number.isFinite(values[s.key]) ? values[s.key] : NaN);
    const cut = t - this.window * 1.1;
    let drop = 0; while (drop < this.t.length && this.t[drop] < cut) drop++;
    if (drop) { this.t.splice(0, drop); for (const k in this.data) this.data[k].splice(0, drop); }
  }
  draw() {
    const { canvas } = this;
    const rect = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.round(rect.width * dpr), H = Math.round(rect.height * dpr);
    if (!W || !H) return;
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    const c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    c.clearRect(0, 0, w, h);
    if (this.t.length < 2) return;
    const tEnd = this.t.at(-1), tStart = Math.max(this.t[0], tEnd - this.window);
    let lo = Infinity, hi = -Infinity;
    for (const s of this.series) for (let i = 0; i < this.t.length; i++) { if (this.t[i] < tStart) continue; const v = this.data[s.key][i]; if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
    if (!Number.isFinite(lo)) return;
    if (this.floor !== null) lo = Math.min(lo, this.floor);
    if (this.ceiling !== null) hi = Math.max(hi, this.ceiling);
    if (hi - lo < this.minRange) { const m = (hi + lo) / 2; lo = m - this.minRange / 2; hi = m + this.minRange / 2; }
    const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
    const left = 34, right = 4, topPad = 6, bottom = 6;
    const X = (t) => left + ((t - tStart) / Math.max(1e-6, tEnd - tStart)) * (w - left - right);
    const Y = (v) => topPad + (1 - (v - lo) / (hi - lo)) * (h - topPad - bottom);
    c.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
    c.lineWidth = 1;
    for (let j = 0; j <= 2; j++) {
      const v = lo + ((hi - lo) * (2 - j)) / 2, y = Y(v);
      c.strokeStyle = '#403b3b'; c.beginPath(); c.moveTo(left, y); c.lineTo(w - right, y); c.stroke();
      c.fillStyle = '#99918d'; c.fillText(this.format(v), 0, y + 3);
    }
    if (lo < 0 && hi > 0) { c.strokeStyle = '#5a5451'; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(left, Y(0)); c.lineTo(w - right, Y(0)); c.stroke(); c.setLineDash([]); }
    for (const s of this.series) {
      c.strokeStyle = s.color; c.lineWidth = s.width ?? 1.5; c.setLineDash(s.dash ?? []);
      c.beginPath();
      let pen = false;
      for (let i = 0; i < this.t.length; i++) {
        if (this.t[i] < tStart) continue;
        const v = this.data[s.key][i];
        if (!Number.isFinite(v)) { pen = false; continue; }
        const x = X(this.t[i]), y = Y(v);
        if (pen) c.lineTo(x, y); else { c.moveTo(x, y); pen = true; }
      }
      c.stroke();
    }
    c.setLineDash([]);
  }
}
