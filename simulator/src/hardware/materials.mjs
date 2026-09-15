// Contact materials of the printed robot and the Hertz stiffness of a sphere on a rigid floor.
//   k = (4/3) E* √R,  E* = E / (1 − ν²)       (Johnson, Contact Mechanics, §4.2)
export const hertzK = (E, nu, R) => (4 / 3) * (E / (1 - nu * nu)) * Math.sqrt(R);

export const MATERIALS = {
  tpu95a: { E: 26e6, nu: 0.45, label: 'TPU 95A foot pad' },
  pla: { E: 3.5e9, nu: 0.36, label: 'PLA shell' },
};
