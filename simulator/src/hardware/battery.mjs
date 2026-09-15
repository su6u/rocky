export class Battery {
  // Lumped LiPo pack: open-circuit voltage vs state of charge, series resistance (cells + wiring + connectors).
  constructor({ cells = 2, capacityAh = 2.2, rInternal = 0.04, soc = 1.0 } = {}) {
    Object.assign(this, { cells, capacityAh, rInternal, soc });
    this.current = 0; this.voltage = this.ocv(); this.energyJ = 0; this.minVoltage = this.voltage;
  }
  ocv() {
    // Typical LiPo discharge curve per cell (piecewise-linear fit to common vendor curves; estimate).
    const s = Math.max(0, Math.min(1, this.soc));
    const pts = [[0, 3.3], [0.05, 3.55], [0.1, 3.68], [0.3, 3.78], [0.5, 3.85], [0.7, 3.95], [0.9, 4.1], [1, 4.2]];
    for (let i = 1; i < pts.length; i++) if (s <= pts[i][0]) { const [a, va] = pts[i - 1], [b, vb] = pts[i]; return this.cells * (va + (vb - va) * (s - a) / (b - a)); }
    return this.cells * 4.2;
  }
  update(currentA, dt) {
    this.current = currentA;
    this.voltage = Math.max(0, this.ocv() - this.rInternal * currentA);
    this.soc -= (currentA * dt) / (3600 * this.capacityAh);
    this.energyJ += this.voltage * currentA * dt;
    this.minVoltage = Math.min(this.minVoltage, this.voltage);
  }
}
