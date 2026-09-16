// Closed-form rigid-body inertia tensors (about the centre of mass, principal frame) and helpers.
export const boxInertia = (m, sx, sy, sz) => new Float64Array([m * (sy * sy + sz * sz) / 12, 0, 0, 0, m * (sx * sx + sz * sz) / 12, 0, 0, 0, m * (sx * sx + sy * sy) / 12]);
export const sphereInertia = (m, r) => { const I = 0.4 * m * r * r; return new Float64Array([I, 0, 0, 0, I, 0, 0, 0, I]); };
export const rodInertia = (m, L, axis = 0) => { const I = new Float64Array(9), j = m * L * L / 12; for (let k = 0; k < 3; k++) I[4 * k] = k === axis ? 1e-12 : j; return I; };
export const I3 = () => new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
