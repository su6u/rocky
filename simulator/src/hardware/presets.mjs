// Actuator catalogue: identified or datasheet-derived servo parameters, output-referred (after the gearbox).
// Every preset states its provenance in `source`; values that were not identified on a bench say so.
//
// Thermal calibration shared by the STS3215 housing (45.2 × 24.7 × 35 mm, 55 g)
//   Robo9 bench test of the STS3215 (robonine.com, 2025): a static 15 kg·cm hold (1.47 N·m) settles at 48 °C
//   after 10 min and stays there for > 1 h (ΔT ≈ 23 K at 25 °C ambient). Lifted into position, the BAM M6
//   friction budget makes the motor supply τ_load + τ_f ≈ 2.2 N·m, i.e. I ≈ 0.92 A at k_t = 2.39 N·m/A (12 V
//   winding) and ≈ 5.6 W of copper loss at R(60 °C) ≈ 6.6 Ω, plus ≈ 0.24 W of logic: R_ca ≈ 4 K/W on the bench
//   jig, which conducts heat away through the horn and screws. Mounted in a printed PLA limb that path is mostly
//   gone, and convection plus radiation from the 66 cm² case alone gives ≈ 10 K/W, so 6 K/W is used (±40 %).
//   "Stable after 10 min" gives a case time constant ≈ 200 s → C_c ≈ 35 J/K. The winding node (R_wc 3 K/W,
//   C_w 7 J/K, τ ≈ 20 s) is a typical small DC motor value, not measured. Consistency check: the model draws
//   0.62 A at 10 kg·cm, matching Robo9's "well below 1 A". Continuous ±90° cycling reached 71 °C after ≈110 min
//   on the bench (not fitted).
const STS3215_THERMAL = { Rwc: 3.0, Cw: 7, Rca: 6.0, Cc: 35, Tmax: 70, quiescentCurrent: 0.02 };
const STS3215_FRICTION_M6 = {
  model: 'm6', base: 0.0533134815909817, stribeck: 4.257322248151969e-05, viscous: 0.028160537042911545,
  loadMotor: 0.044912012282132555, loadExternal: 0.24355483356616028, loadMotorStribeck: 0.00012448975632695385,
  loadExternalStribeck: 0.1422905635963302, quadMotor: 0.007594943296670792, quadExternal: 0.00340408727365082,
  vStribeck: 0.371136775424206, alpha: 9.93094944677977,
};
// Feetech STS memory-table defaults: overload torque 80 %, protection time 2 s, protective torque 20 %, max temperature 70 °C.
const FEETECH_PROTECTION = { overloadFraction: 0.8, protectionTime: 2.0, protectiveFraction: 0.2, protectionReleaseDeg: 1 };
const KGCM = 0.0980665;

export const SERVO_PRESETS = {
  sts3215_7v4: {
    label: 'Feetech STS3215 (7.4 V, 1:345)',
    source: 'BAM identification feetech_sts3215_7_4V m6 (kt, R, armature, friction, delay, velocity limit); datasheet mass, stall 19.5 kg·cm and rated 5 kg·cm; thermal network calibrated to the Robo9 bench test',
    vNominal: 7.4, kp: 32, errorGain: 0.166 * 1.1616979929275297, dutyMax: 0.97,
    kt: 1.2753202369175398, R: 2.753421214375944, armature: 0.021564290129722483,
    maxVelocity: 5.095754044445978, commandDelay: 0.0049807526194816515,
    friction: STS3215_FRICTION_M6,
    mass: 0.055, stallTorqueDatasheet: 19.5 * KGCM, ratedTorqueDatasheet: 5 * KGCM, stallCurrent: 2.4,
    backlashRad: 0.87 * Math.PI / 180,           // measured by Robo9; the datasheet claims ≤ 0.5°
    thermal: STS3215_THERMAL, ...FEETECH_PROTECTION,
    vMin: 4.0, vMax: 8.4, cells: 2,
  },
  sts3215_12v: {
    label: 'Feetech STS3215 (12 V, 1:345)',
    source: 'Datasheet 30 kg·cm stall / 10 kg·cm rated / 45 rpm at 12 V, 2 A overcurrent limit. k_t is scaled from the 7.4 V BAM identification by the no-load speed per volt (×1.874). R is set so the stall torque keeps the datasheet ratio to the 7.4 V model (the identified 7.4 V model is 18 % stronger than its own datasheet, and so is this one: ≈35 kg·cm, as Robo9 measured at 12 V). That gives 2.1 A at stall against the datasheet 2.7 A; the datasheet torque and current do not agree. Same gearbox friction, armature and housing — NOT identified',
    vNominal: 12, kp: 32, errorGain: 0.166 * 1.1616979929275297, dutyMax: 0.97,
    kt: 1.2753202369175398 * (12 / 7.4) * (52 / 45), R: 2.753421214375944 * (12 / 7.4) * (52 / 45) * (12 / 7.4) / (30 / 19.5), armature: 0.021564290129722483,
    maxVelocity: 5.095754044445978, commandDelay: 0.0049807526194816515,
    friction: STS3215_FRICTION_M6,
    mass: 0.055, stallTorqueDatasheet: 30 * KGCM, ratedTorqueDatasheet: 10 * KGCM, stallCurrent: 2.7,
    backlashRad: 0.87 * Math.PI / 180,
    thermal: STS3215_THERMAL, ...FEETECH_PROTECTION, protectionCurrent: 2.0,
    vMin: 4.0, vMax: 14.0, cells: 3,
  },
  sts3250_12v: {
    label: 'Feetech STS3250 (12 V, 1:345)',
    source: 'Datasheet-derived (50 kg·cm stall, 4.2 A stall, 0.133 s/60°); friction scaled from the STS3215 identification — NOT identified',
    vNominal: 12, kp: 32, errorGain: 0.166 * 1.16, dutyMax: 0.97,
    kt: 12 / (Math.PI / 3 / 0.133) * 0.97, R: 12 / 4.2, armature: 0.021564290129722483 * 1.6,
    maxVelocity: 7.0, commandDelay: 0.005,
    friction: { model: 'm4', base: 0.0533 * 2.5, stribeck: 0.0111 * 2.5, viscous: 0.0338 * 1.6, load: 0.2079, loadStribeck: 4.9e-5, vStribeck: 0.4432, alpha: 9.99 },
    mass: 0.0745, stallTorqueDatasheet: 50 * KGCM, ratedTorqueDatasheet: 50 * KGCM * 0.26, stallCurrent: 4.2,
    backlashRad: 0.7 * Math.PI / 180,
    thermal: { Rwc: 2.5, Cw: 10, Rca: 5.0, Cc: 50, Tmax: 70, quiescentCurrent: 0.02 }, ...FEETECH_PROTECTION,
    vMin: 7.0, vMax: 12.6, cells: 3,
  },
  xm540_w270: {
    label: 'Dynamixel XM540-W270 (12 V, 1:272.5)',
    source: 'Datasheet-derived (10.6 N·m / 4.4 A stall @12 V, 30 rpm no-load, 165 g, 0.25° backlash); friction scaled from MX-106 identification — NOT identified',
    vNominal: 12, kp: 32, errorGain: 0.158, dutyMax: 0.9625,
    kt: 12 / (30 * 2 * Math.PI / 60) * 0.96, R: 12 / 4.4, armature: 0.02457397334336701 * 2.2,
    maxVelocity: 3.0, commandDelay: 0.005,
    friction: { model: 'm4', base: 0.05, stribeck: 0.10491060771568866 * 1.3, viscous: 0.05096082506010955 * 1.3, load: 0.009150334028953775, loadStribeck: 0.22235912179095094, vStribeck: 1.6232949773728853, alpha: 1.323485668999259 },
    mass: 0.165, stallTorqueDatasheet: 10.6, ratedTorqueDatasheet: 10.6 * 0.3, stallCurrent: 4.4,
    backlashRad: 0.25 * Math.PI / 180,
    thermal: { Rwc: 1.8, Cw: 25, Rca: 3.5, Cc: 110, Tmax: 80, quiescentCurrent: 0.04 }, overloadFraction: 0.8, protectionTime: 2.0, protectiveFraction: 0.2, protectionReleaseDeg: 1,
    vMin: 10, vMax: 14.8, cells: 3,
  },
  ideal: {
    label: 'Ideal joint (demand measurement)',
    source: 'Not hardware: a near-rigid, frictionless, delay-free position servo with 740 N·m of authority, used to measure the torque a trajectory demands',
    ideal: true,
    vNominal: 7.4, kp: 32, errorGain: 0.166 * 1.1616979929275297, dutyMax: 1.0,
    kt: 1.0, R: 0.01, armature: 0.0005, maxVelocity: 100, commandDelay: 0,
    friction: { model: 'm1', base: 0, viscous: 0 },
    mass: 0.055, stallTorqueDatasheet: 740, ratedTorqueDatasheet: 740, stallCurrent: 0,
    backlashRad: 0, thermal: { Rwc: 1, Cw: 1e9, Rca: 1, Cc: 1e9, Tmax: 1e9, quiescentCurrent: 0 },
    vMin: 0, vMax: 1e9, cells: 2,
  },
};
