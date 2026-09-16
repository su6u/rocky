// Small dense linear algebra on Float64Array. Matrices are row-major, n×n unless noted.
// Kept dependency-free so the same code runs in the browser and in Node tests.

export const vec3 = (x = 0, y = 0, z = 0) => new Float64Array([x, y, z]);
export const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm3 = (a) => Math.hypot(a[0], a[1], a[2]);
export function cross3(a, b, out = new Float64Array(3)) {
  const x = a[1] * b[2] - a[2] * b[1], y = a[2] * b[0] - a[0] * b[2], z = a[0] * b[1] - a[1] * b[0];
  out[0] = x; out[1] = y; out[2] = z; return out;
}
export function add3(a, b, out = new Float64Array(3)) { out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2]; return out; }
export function sub3(a, b, out = new Float64Array(3)) { out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2]; return out; }
export function scale3(a, s, out = new Float64Array(3)) { out[0] = a[0] * s; out[1] = a[1] * s; out[2] = a[2] * s; return out; }
export function normalize3(a, out = new Float64Array(3)) { const n = norm3(a) || 1; return scale3(a, 1 / n, out); }

// 3×3 rotation matrices stored row-major in length-9 arrays.
export const mat3I = () => new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
// Aliasing-safe (out may be A or B) without a temporary allocation.
export function mat3mul(A, B, out = new Float64Array(9)) {
  const a0 = A[0], a1 = A[1], a2 = A[2], a3 = A[3], a4 = A[4], a5 = A[5], a6 = A[6], a7 = A[7], a8 = A[8];
  const b0 = B[0], b1 = B[1], b2 = B[2], b3 = B[3], b4 = B[4], b5 = B[5], b6 = B[6], b7 = B[7], b8 = B[8];
  out[0] = a0 * b0 + a1 * b3 + a2 * b6; out[1] = a0 * b1 + a1 * b4 + a2 * b7; out[2] = a0 * b2 + a1 * b5 + a2 * b8;
  out[3] = a3 * b0 + a4 * b3 + a5 * b6; out[4] = a3 * b1 + a4 * b4 + a5 * b7; out[5] = a3 * b2 + a4 * b5 + a5 * b8;
  out[6] = a6 * b0 + a7 * b3 + a8 * b6; out[7] = a6 * b1 + a7 * b4 + a8 * b7; out[8] = a6 * b2 + a7 * b5 + a8 * b8;
  return out;
}
export function mat3T(A, out = new Float64Array(9)) {
  const a1 = A[1], a2 = A[2], a5 = A[5];
  out[0] = A[0]; out[1] = A[3]; out[2] = A[6]; out[3] = a1; out[4] = A[4]; out[5] = A[7]; out[6] = a2; out[7] = a5; out[8] = A[8];
  return out;
}
export function mat3vec(A, v, out = new Float64Array(3)) {
  const x = A[0] * v[0] + A[1] * v[1] + A[2] * v[2], y = A[3] * v[0] + A[4] * v[1] + A[5] * v[2], z = A[6] * v[0] + A[7] * v[1] + A[8] * v[2];
  out[0] = x; out[1] = y; out[2] = z; return out;
}
export function mat3Tvec(A, v, out = new Float64Array(3)) {
  const x = A[0] * v[0] + A[3] * v[1] + A[6] * v[2], y = A[1] * v[0] + A[4] * v[1] + A[7] * v[2], z = A[2] * v[0] + A[5] * v[1] + A[8] * v[2];
  out[0] = x; out[1] = y; out[2] = z; return out;
}
// Rodrigues: rotation by angle about unit axis.
export function axisAngleMat(axis, ang, out = new Float64Array(9)) {
  const [x, y, z] = axis, c = Math.cos(ang), s = Math.sin(ang), C = 1 - c;
  out[0] = c + x * x * C; out[1] = x * y * C - z * s; out[2] = x * z * C + y * s;
  out[3] = y * x * C + z * s; out[4] = c + y * y * C; out[5] = y * z * C - x * s;
  out[6] = z * x * C - y * s; out[7] = z * y * C + x * s; out[8] = c + z * z * C;
  return out;
}

// Unit quaternions [w, x, y, z]; body→world rotation.
export function quatToMat(q, out = new Float64Array(9)) {
  const [w, x, y, z] = q;
  out[0] = 1 - 2 * (y * y + z * z); out[1] = 2 * (x * y - z * w); out[2] = 2 * (x * z + y * w);
  out[3] = 2 * (x * y + z * w); out[4] = 1 - 2 * (x * x + z * z); out[5] = 2 * (y * z - x * w);
  out[6] = 2 * (x * z - y * w); out[7] = 2 * (y * z + x * w); out[8] = 1 - 2 * (x * x + y * y);
  return out;
}
export function quatMul(a, b, out = new Float64Array(4)) {
  const w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  const x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  const y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  const z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
  out[0] = w; out[1] = x; out[2] = y; out[3] = z; return out;
}
// Exact exponential map: q ⊗ exp(ω h / 2) for body-frame angular velocity ω.
export function quatIntegrateBody(q, w, h, out = new Float64Array(4)) {
  const th = Math.hypot(w[0], w[1], w[2]) * h;
  let dq;
  if (th < 1e-12) dq = [1, 0.5 * w[0] * h, 0.5 * w[1] * h, 0.5 * w[2] * h];
  else { const s = Math.sin(th / 2) / (th / h); dq = [Math.cos(th / 2), w[0] * s, w[1] * s, w[2] * s]; }
  quatMul(q, dq, out);
  const n = Math.hypot(out[0], out[1], out[2], out[3]);
  for (let i = 0; i < 4; i++) out[i] /= n;
  return out;
}
export function matToQuat(R) {
  const t = R[0] + R[4] + R[8];
  let w, x, y, z;
  if (t > 0) { const s = Math.sqrt(t + 1) * 2; w = s / 4; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s; }
  else if (R[0] > R[4] && R[0] > R[8]) { const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2; w = (R[7] - R[5]) / s; x = s / 4; y = (R[3] + R[1]) / s; z = (R[2] + R[6]) / s; }
  else if (R[4] > R[8]) { const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2; w = (R[2] - R[6]) / s; x = (R[3] + R[1]) / s; y = s / 4; z = (R[7] + R[5]) / s; }
  else { const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2; w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[7] + R[5]) / s; z = s / 4; }
  const q = new Float64Array([w, x, y, z]); const n = Math.hypot(w, x, y, z);
  for (let i = 0; i < 4; i++) q[i] /= n * (w < 0 ? -1 : 1);
  return q;
}

// In-place Cholesky factorization A = L Lᵀ (lower triangle of A overwritten). Returns false if not SPD.
export function cholesky(A, n) {
  for (let j = 0; j < n; j++) {
    let d = A[j * n + j];
    for (let k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
    if (!(d > 0)) return false;
    const ljj = Math.sqrt(d);
    A[j * n + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s / ljj;
    }
  }
  return true;
}
// Solve L Lᵀ x = b in place (b overwritten with x); stride allows column vectors in a flat array.
export function cholSolve(L, n, b, off = 0, stride = 1) {
  for (let i = 0; i < n; i++) {
    let s = b[off + i * stride];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * b[off + k * stride];
    b[off + i * stride] = s / L[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[off + i * stride];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * b[off + k * stride];
    b[off + i * stride] = s / L[i * n + i];
  }
  return b;
}

// Symmetric 3×3 eigen-decomposition (Jacobi). Returns {values, vectors(row-major columns = eigvecs)}.
export function symEig3(S) {
  const a = Float64Array.from(S), V = mat3I();
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = Math.abs(a[1]) + Math.abs(a[2]) + Math.abs(a[5]);
    if (off < 1e-15) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = a[3 * p + q];
      if (Math.abs(apq) < 1e-18) continue;
      const theta = (a[3 * q + q] - a[3 * p + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[3 * k + p], akq = a[3 * k + q];
        a[3 * k + p] = c * akp - s * akq; a[3 * k + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[3 * p + k], aqk = a[3 * q + k];
        a[3 * p + k] = c * apk - s * aqk; a[3 * q + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[3 * k + p], vkq = V[3 * k + q];
        V[3 * k + p] = c * vkp - s * vkq; V[3 * k + q] = s * vkp + c * vkq;
      }
    }
  }
  return { values: [a[0], a[4], a[8]], vectors: V };
}
