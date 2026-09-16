// Seeded pseudo-random numbers so a run can be reproduced exactly (sensor noise, disturbances).
// sfc32 generator (Doty-Humphrey, PractRand-tested) and Marsaglia's polar method for normal deviates.
export function createRng(seed = 1) {
  let a = 0x9e3779b9, b = 0x243f6a88, c = 0xb7e15162, d = (seed >>> 0) ^ 0x5bd1e995;
  const next = () => {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 16; i++) next();
  let spare = null;
  const normal = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u, v, r;
    do { u = next() * 2 - 1; v = next() * 2 - 1; r = u * u + v * v; } while (r >= 1 || r === 0);
    const m = Math.sqrt((-2 * Math.log(r)) / r);
    spare = v * m;
    return u * m;
  };
  return { next, normal };
}
