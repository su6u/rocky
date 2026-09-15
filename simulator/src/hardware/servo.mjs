// Servo actuator physics: electromechanics, gearbox friction, heating and the servo's own firmware.
//
// Electrical/mechanical model (drive/brake H-bridge, output-referred constants), after
// Duclusaud & Passault, "Extended Friction Models for the Physics Simulation of Servo Actuators",
// ICRA 2025 (arXiv:2410.08650) and the open BAM implementation (github.com/Rhoban/bam):
//
//   duty  = clip(k_p · g_e · (θ*_s − θ_enc), ±duty_max)          firmware P position loop
//   U     = duty · V_bus                                          PWM-averaged armature voltage
//   I     = (U − k_t θ̇_m) / R(T_w)                               armature current (L ≈ 0 at these time scales)
//   τ_m   = k_t I = (k_t/R) U − (k_t²/R) θ̇_m                    back-EMF acts as viscous damping
//   J_m θ̈_m = τ_m + τ_f − b θ̇_m + τ_e                            armature J_m = N² J_rotor, b viscous
//
// Friction is a torque budget |τ_f| ≤ τ_f^max(θ̇, τ_m, τ_e) (BAM M1–M6), enforced as a box-constrained velocity
// row in the contact solver: exact stiction instead of sign(θ̇) chatter. The gear train keeps this friction when the
// servo is unpowered, so a failed servo is back-driven only by loads above its stiction budget.
//
// Extensions (documented here, not part of BAM):
//   • copper resistance drift  R(T_w) = R₂₅ (1 + α_Cu (T_w − 25 °C)), α_Cu = 0.00393 K⁻¹
//   • two-node thermal network (winding → case → ambient):
//         C_w Ṫ_w = I² R(T_w) − (T_w − T_c)/R_wc
//         C_c Ṫ_c = (T_w − T_c)/R_wc + |τ_f θ̇| + b θ̇² + V_bus I_q − (T_c − T_amb)/(R_ca · k_enclosure)
//     The firmware's temperature register reads the driver board, i.e. the case node T_c; its 70 °C limit applies
//     there. Calibration: see presets.mjs (Robo9 bench test of the STS3215).
//   • firmware protections, as in the Feetech STS memory table:
//         overload     |duty| ≥ 80 % of max duty for 2 s → output limited to 20 % until a new goal arrives
//         overheat     T_c > max temperature (70 °C)     → torque released (unload) until re-enabled below the limit
//         overcurrent  |I| > protection current for 2 s  → torque released (only where the datasheet states it)
//   • command transport delay and the firmware's internal target velocity limit (BAM)
//   • supply-voltage dependence (battery sag computed by the world each step)

export const ALPHA_CU = 0.00393;
const DEG = Math.PI / 180;

// BAM friction torque budget. dth: motor-side velocity, tm: motor torque, te: external (load) torque.
export function frictionBudget(fr, dth, tm, te) {
  const S = fr.vStribeck ? Math.exp(-Math.pow(Math.abs(dth / fr.vStribeck), fr.alpha)) : 0;
  switch (fr.model) {
    case 'm1': return fr.base;
    case 'm2': return fr.base + S * fr.stribeck;
    case 'm3': return fr.base + fr.load * Math.abs(te - tm);
    case 'm4': { const g = Math.abs(te - tm); return fr.base + fr.load * g + S * (fr.stribeck + fr.loadStribeck * g); }
    case 'm5':
    case 'm6': {
      const g = Math.abs(te * fr.loadExternal - tm * fr.loadMotor);
      const gs = Math.abs(te * fr.loadExternalStribeck - tm * fr.loadMotorStribeck);
      let f = fr.base + g + S * (fr.stribeck + gs);
      if (fr.model === 'm6' && Math.sign(te) !== Math.sign(tm)) {
        const quad = Math.abs(te) < Math.abs(tm) ? fr.quadExternal * te * te : fr.quadMotor * tm * tm;
        f += S * quad;
      }
      return f;
    }
    default: return fr.base;
  }
}

export class Servo {
  // enclosure: multiplier on the case-to-ambient thermal resistance (servos inside the closed carapace run hotter).
  constructor(preset, { name = '', ambient = 25, protection = true, enclosure = 1 } = {}) {
    this.p = preset; this.name = name;
    this.ambient = ambient; this.enclosure = enclosure;
    this.T = ambient; this.Tw = ambient;                 // case/sensor and winding temperatures (°C)
    this.protectionEnabled = protection && !preset.ideal;
    this.overloadTimer = 0; this.protected = false; this.protectGoal = 0;
    this.overcurrentTimer = 0; this.unload = null;       // 'overheat' | 'overcurrent' | null (torque released by firmware)
    this.failed = false; this.torqueEnabled = true; this.overheated = false;
    this.targetCmd = 0; this.targetSmooth = null; this.queue = [];
    this.duty = 0; this.U = 0; this.I = 0; this.tauMotor = 0; this.tauExt = 0; this.tauFriction = 0;
    this.powerElec = 0; this.powerMech = 0; this.copperLoss = 0; this.frictionLoss = 0; this.supplyCurrent = 0;
    this.energyElec = 0; this.peakTorque = 0; this.peakCurrent = 0; this.peakTemp = ambient;
    this.sumSqTorque = 0; this.sumTime = 0; this.timeAboveRated = 0; this.trips = [];
  }
  resistance() { return this.p.R * (1 + ALPHA_CU * (this.Tw - (this.p.Tref ?? 25))); }
  get powered() { return this.torqueEnabled && !this.failed && !this.unload; }
  // Controller writes a new goal; it becomes visible to the firmware after the transport delay.
  command(target, time, extraDelay = 0) { this.queue.push([time + this.p.commandDelay + extraDelay, target]); }
  // Supervisor request to re-enable torque after a firmware unload. The firmware refuses while still over temperature.
  enableTorque() {
    if (this.unload === 'overheat' && this.T > this.p.thermal.Tmax) return false;
    this.unload = null; this.overcurrentTimer = 0;
    return true;
  }
  // Called every physics step. Returns {tauExplicit, damping} for the servo (motor-side) DOF.
  computeDrive(time, dt, thetaEnc, dthMotor, vBus) {
    const p = this.p;
    while (this.queue.length && this.queue[0][0] <= time) {
      const goal = this.queue.shift()[1];
      // Feetech overload protection is released by a new goal command (changes below 1° are treated as the same goal).
      if (this.protected && Math.abs(goal - this.protectGoal) > (p.protectionReleaseDeg ?? 1) * DEG) { this.protected = false; this.overloadTimer = 0; }
      this.targetCmd = goal;
    }
    if (this.targetSmooth === null) this.targetSmooth = thetaEnc;
    const maxStep = p.maxVelocity * dt;
    this.targetSmooth += Math.max(-maxStep, Math.min(maxStep, this.targetCmd - this.targetSmooth));
    const R = this.resistance(), kt = p.kt;
    let dutyMax = p.dutyMax;
    if (this.protected) dutyMax *= p.protectiveFraction ?? 0.2;
    let duty = p.kp * p.errorGain * (this.targetSmooth - thetaEnc);
    duty = Math.max(-dutyMax, Math.min(dutyMax, duty));
    const powered = this.powered && vBus >= p.vMin;
    if (!powered) duty = 0;
    this.duty = duty; this.U = duty * vBus;
    // Drive/brake bridge: when unpowered the windings are left open (coast), so there is no back-EMF braking.
    return { tauExplicit: powered ? (kt * this.U) / R : 0, damping: powered ? (kt * kt) / R : 0 };
  }
  // After the step: bookkeeping with the resolved motion. tauFriction is the solver's gearbox friction torque.
  finish(dt, dthMotor, accMotor, tauFriction, vBus) {
    const p = this.p, R = this.resistance(), kt = p.kt, th = p.thermal;
    const powered = this.powered && vBus >= p.vMin;
    this.I = powered ? (this.U - kt * dthMotor) / R : 0;
    this.tauMotor = kt * this.I;
    this.tauFriction = tauFriction;
    const b = p.friction.viscous ?? 0;
    // Load torque acting on the output as seen by the gearbox: J_m θ̈ = τ_m + τ_f − b θ̇ + τ_e
    this.tauExt = p.armature * accMotor - this.tauMotor - tauFriction + b * dthMotor;
    this.copperLoss = this.I * this.I * R;
    this.frictionLoss = Math.abs(tauFriction * dthMotor) + b * dthMotor * dthMotor;
    this.powerElec = this.U * this.I;                              // negative when back-driven (regeneration)
    this.powerMech = this.tauMotor * dthMotor;
    const quiescent = vBus >= p.vMin ? (th.quiescentCurrent ?? 0.02) : 0;
    this.supplyCurrent = p.ideal ? 0 : this.duty * this.I + quiescent;   // PWM-averaged bus current + logic
    this.energyElec += Math.max(0, this.powerElec) * dt;
    // two-node thermal network (explicit Euler is stable: the fastest time constant R_wc·C_w ≈ 20 s ≫ dt)
    const qwc = (this.Tw - this.T) / th.Rwc;
    const Rca = th.Rca * this.enclosure;
    this.Tw += (dt * (this.copperLoss - qwc)) / th.Cw;
    this.T += (dt * (qwc + this.frictionLoss + vBus * quiescent - (this.T - this.ambient) / Rca)) / th.Cc;
    const absT = Math.abs(this.tauMotor);
    this.peakTorque = Math.max(this.peakTorque, absT); this.peakCurrent = Math.max(this.peakCurrent, Math.abs(this.I));
    this.peakTemp = Math.max(this.peakTemp, this.T);
    this.sumSqTorque += this.tauMotor * this.tauMotor * dt; this.sumTime += dt;
    if (absT > p.ratedTorqueDatasheet) this.timeAboveRated += dt;
    if (!this.protectionEnabled) return;
    // overload: the firmware's "present load" is the PWM duty; sustained saturation means the servo is blocked
    if (!this.protected && Math.abs(this.duty) >= (p.overloadFraction ?? 0.8) * p.dutyMax) {
      this.overloadTimer += dt;
      if (this.overloadTimer >= (p.protectionTime ?? 2.0)) { this.protected = true; this.protectGoal = this.targetCmd; this.trips.push({ kind: 'overload', t: this.sumTime }); }
    } else if (!this.protected) this.overloadTimer = 0;
    if (p.protectionCurrent && !this.unload) {
      if (Math.abs(this.I) > p.protectionCurrent) {
        this.overcurrentTimer += dt;
        if (this.overcurrentTimer >= (p.protectionTime ?? 2.0)) { this.unload = 'overcurrent'; this.trips.push({ kind: 'overcurrent', t: this.sumTime }); }
      } else this.overcurrentTimer = 0;
    }
    if (this.T > th.Tmax && this.unload !== 'overheat') { this.unload = 'overheat'; this.overheated = true; this.trips.push({ kind: 'overheat', t: this.sumTime }); }
  }
  rmsTorque() { return this.sumTime > 0 ? Math.sqrt(this.sumSqTorque / this.sumTime) : 0; }
  // Gearbox friction torque budget for the solver's box row, from last step's motor and load torques.
  frictionBudget(dthMotor) { return frictionBudget(this.p.friction, dthMotor, this.tauMotor, this.tauExt); }
}
