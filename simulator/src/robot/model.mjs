// Build a simulatable Rocky robot from assets/robot.json (official kit geometry, print millimetres).
//
// Kinematic design (per limb, 3 actuated DOF):
//   shoulder yaw   – axis: torso normal through the kit ball centre
//   shoulder pitch – axis: horizontal, ⊥ radial, through the ball centre (positive lifts the limb)
//   elbow          – axis: the kit's pin/hole hinge (measured, including its skew)
// The ball joint's third rotation (roll about the upper limb) is not needed to place a point foot, so it is
// fixed in the robot. Optional passive backlash DOFs sit in series with each servo (encoder on the output side).
//
// Mass model: FDM-printed stone shell (walls + sparse infill, from exact surface/volume integrals of the kit
// meshes) + servo point masses + battery/electronics, or a canon override that rescales densities to 168 kg.
import { ArticulatedModel } from '../engine/dynamics.mjs';
import { SERVO_PRESETS } from '../hardware/presets.mjs';

const T = (R) => [R[0][0], R[1][0], R[2][0], R[0][1], R[1][1], R[2][1], R[0][2], R[1][2], R[2][2]];
const flat = (R) => [R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2]];
const scaleI = (I, k) => Float64Array.from(flat(I).map((x) => x * k));

export const DEFAULT_DESIGN = {
  scale: 0.0045,                  // metres per print-mm → carapace 18.2 cm across (desk companion)
  servo: 'sts3215_7v4',
  pla: { density: 1240, wall: 0.0024, infill: 0.15 },
  battery: { cells: 2, capacityAh: 2.2, mass: 0.125, rInternal: 0.035 },
  electronics: 0.09,              // controller, IMU, bus adapter, wiring (kg)
  elbowServoAt: 'elbow',          // 'elbow' (direct drive) | 'shoulder' (linkage/belt)
  backlash: false,
  hand: '1-B',                    // fist for walking; '1-C' open hand
  canonMass: null,                // e.g. 168 → rescale structural density to reach this total mass
  payload: null,                  // {mass (kg), height (m above the carapace centre)}: a load strapped on top of the carapace
  limits: { yaw: [-0.75, 0.75], pitch: [-1.35, 1.45], elbow: [-2.7, 1.7] },
};

export function buildRocky(robot, design = {}) {
  const D = { ...DEFAULT_DESIGN, ...design, pla: { ...DEFAULT_DESIGN.pla, ...(design.pla || {}) }, limits: { ...DEFAULT_DESIGN.limits, ...(design.limits || {}) } };
  const s = D.scale, servo = SERVO_PRESETS[D.servo];
  const bodies = [], meta = { limbs: [], design: D, servo, bodiesByName: {} };
  const massOf = (rec) => {
    const shellM = rec.shell.mass_per_density * D.pla.density * D.pla.wall * s * s;   // area·ρ·t
    const V = rec.solid.measure * s ** 3, A = rec.shell.measure * s * s;
    const interior = Math.max(V - A * D.pla.wall, 0);
    const infM = D.pla.density * D.pla.infill * interior;
    const Ishell = rec.shell.inertia.map((row) => row.map((x) => x * D.pla.density * D.pla.wall * s ** 4));
    const solidUnitMass = rec.solid.mass_per_density * s ** 3;
    const Iinf = rec.solid.inertia.map((row) => row.map((x) => x * s ** 5 * (solidUnitMass > 0 ? infM / solidUnitMass : 0)));
    const cS = rec.shell.com.map((x) => x * s), cI = rec.solid.com.map((x) => x * s);
    return combine([{ m: shellM, c: cS, I: Ishell }, { m: infM, c: cI, I: Iinf }]);
  };
  const add = (b) => { bodies.push(b); meta.bodiesByName[b.name] = bodies.length - 1; return bodies.length - 1; };
  // ---- torso ----
  const torso = massOf(robot.torso);
  const extras = [
    { m: D.battery.mass, c: [torso.c[0], torso.c[1], torso.c[2] - 0.15 * robot.torso.carapaceThickness_mm * s], I: boxI(D.battery.mass, 0.07, 0.035, 0.02) },
    { m: D.electronics, c: [torso.c[0], torso.c[1], torso.c[2] + 0.1 * robot.torso.carapaceThickness_mm * s], I: boxI(D.electronics, 0.06, 0.05, 0.02) },
  ];
  if (D.payload?.mass) {
    const top = torso.c[2] + 0.5 * robot.torso.carapaceThickness_mm * s + (D.payload.height ?? 0.02);
    extras.push({ m: D.payload.mass, c: [torso.c[0], torso.c[1], top], I: boxI(D.payload.mass, 0.08, 0.08, 0.04) });
  }
  const servoBox = boxI(servo.mass, 0.045, 0.025, 0.035);
  for (const L of robot.limbs) {
    const c = L.shoulder.map((x) => x * s), r = L.radial;
    extras.push({ m: servo.mass, c: [c[0] - r[0] * 0.012, c[1] - r[1] * 0.012, c[2]], I: servoBox }); // yaw servo inside carapace
  }
  const torsoAll = combine([torso, ...extras]);
  const baseIndex = add({ name: 'torso', parent: -1, mass: torsoAll.m, com: torsoAll.c, Icom: Float64Array.from(flat(torsoAll.I)), part: 'torso', partToBody: robot.torso.partToBody });
  const limbsStructural = [];
  for (const L of robot.limbs) {
    const i = L.limb, c = L.shoulder.map((x) => x * s);
    const F = [[L.radial[0], -L.radial[1], 0], [L.radial[1], L.radial[0], 0], [0, 0, 1]]; // columns: radial, tangent, z
    const lim = D.limits;
    const pushJoint = (name, parent, R0, r0, axis, mass, com, I, extra = {}) => add({ name, parent, R0: Float64Array.from(R0), r0: Float64Array.from(r0), axis: Float64Array.from(axis), mass, com: Float64Array.from(com), Icom: Float64Array.from(I), ...extra });
    const zero9 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    // yaw
    let parent = pushJoint(`L${i}.yaw`, baseIndex, flat(F), c, [0, 0, 1], 0, [0, 0, 0], zero9, { servo: true, joint: 'yaw', limb: i, armature: servo.armature, lo: lim.yaw[0], hi: lim.yaw[1] });
    const yawIdx = parent;
    if (D.backlash) parent = pushJoint(`L${i}.yaw.backlash`, parent, flat(I3), [0, 0, 0], [0, 0, 1], 0, [0, 0, 0], zero9, { backlashOf: yawIdx });
    // pitch servo mass rides on the yaw stage (at the ball centre)
    const pitchServo = { m: servo.mass, c: [0, 0, 0], I: servoBox };
    // pitch (carries limb A + pitch servo + optionally elbow servo)
    const A = L.A, Am = massOf(A);
    const hinge = A.hinge.map((x) => x * s);
    const elbowServo = D.elbowServoAt === 'elbow' ? { m: servo.mass, c: hinge, I: servoBox } : { m: servo.mass, c: hinge.map((x) => x * 0.2), I: servoBox };
    const Aall = combine([Am, pitchServo, elbowServo]);
    parent = pushJoint(`L${i}.pitch`, parent, flat(I3), [0, 0, 0], [0, -1, 0], Aall.m, Aall.c, flat(Aall.I), { servo: true, joint: 'pitch', limb: i, armature: servo.armature, lo: lim.pitch[0], hi: lim.pitch[1], part: A.part, partToBody: A.partToBody });
    const pitchIdx = parent;
    let meshHolderPitch = pitchIdx;
    if (D.backlash) {
      // move limb mass to the output side of the backlash
      bodies[pitchIdx].mass = 0; bodies[pitchIdx].com = new Float64Array(3); bodies[pitchIdx].Icom = new Float64Array(9);
      parent = pushJoint(`L${i}.pitch.backlash`, parent, flat(I3), [0, 0, 0], [0, -1, 0], Aall.m, Aall.c, flat(Aall.I), { backlashOf: pitchIdx, part: A.part, partToBody: A.partToBody });
      meshHolderPitch = parent; delete bodies[pitchIdx].part;
    }
    // elbow (carries limb B)
    const bName = i === 1 ? D.hand : `${i}-B`;
    const B = L.B[bName], Bm = massOf(B);
    const H = A.elbowFrame; // columns x,y,z in A coords
    parent = pushJoint(`L${i}.elbow`, parent, flat(H), hinge, [0, 1, 0], Bm.m, Bm.c, flat(Bm.I), { servo: true, joint: 'elbow', limb: i, armature: servo.armature, lo: lim.elbow[0], hi: lim.elbow[1], part: B.part, partToBody: B.partToBody });
    const elbowIdx = parent;
    let footBody = elbowIdx;
    if (D.backlash) {
      bodies[elbowIdx].mass = 0; bodies[elbowIdx].com = new Float64Array(3); bodies[elbowIdx].Icom = new Float64Array(9); delete bodies[elbowIdx].part;
      parent = pushJoint(`L${i}.elbow.backlash`, parent, flat(I3), [0, 0, 0], [0, 1, 0], Bm.m, Bm.c, flat(Bm.I), { backlashOf: elbowIdx, part: B.part, partToBody: B.partToBody });
      footBody = parent;
    }
    limbsStructural.push(Am.m + Bm.m);
    meta.limbs.push({
      limb: i, yaw: yawIdx, pitch: pitchIdx, elbow: elbowIdx, footBody, meshHolderPitch,
      foot: { local: B.foot.center.map((x) => x * s), radius: B.foot.radius * s },
      shoulder: c, radial: L.radial, hinge, statue: L.statue, statueElbow: B.statueElbow, bPart: bName,
      collisionA: A.collision.map((q) => ({ center: q.center.map((x) => x * s), radius: q.radius * s })),
      collisionB: B.collision.map((q) => ({ center: q.center.map((x) => x * s), radius: q.radius * s })),
      lengthA: Math.hypot(...hinge), lengthB: Math.hypot(...B.foot.center) * s,
    });
  }
  // Optional canon mass: scale every structural body so the total reaches the target (servo masses unchanged).
  let model = new ArticulatedModel({ bodies });
  const total = model.totalMass();
  if (D.canonMass) {
    const k = D.canonMass / total;
    for (const b of bodies) { b.mass *= k; for (let j = 0; j < 9; j++) b.Icom[j] *= k; }
    model = new ArticulatedModel({ bodies });
  }
  meta.totalMass = model.totalMass();
  meta.payloadMass = D.payload?.mass ?? 0;
  meta.structuralMass = torso.m + limbsStructural.reduce((a, b) => a + b, 0);
  meta.servoCount = robot.limbs.length * 3;
  meta.servoMassTotal = meta.servoCount * servo.mass;
  meta.carapaceAcross = robot.torso.carapaceWidthMax_mm * s;
  meta.canonScale = robot.canon.printToCanonScale / 1000;
  return { model, meta };
}

const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
function boxI(m, a, b, c) { return [[m * (b * b + c * c) / 12, 0, 0], [0, m * (a * a + c * c) / 12, 0], [0, 0, m * (a * a + b * b) / 12]]; }
function combine(parts) {
  const m = parts.reduce((a, p) => a + p.m, 0);
  const c = [0, 1, 2].map((k) => parts.reduce((a, p) => a + p.m * p.c[k], 0) / m);
  const I = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of parts) {
    const d = [p.c[0] - c[0], p.c[1] - c[1], p.c[2] - c[2]], dd = d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
    for (let r = 0; r < 3; r++) for (let q = 0; q < 3; q++) I[r][q] += p.I[r][q] + p.m * ((r === q ? dd : 0) - d[r] * d[q]);
  }
  return { m, c, I };
}
export { combine, flat, T };
