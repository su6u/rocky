// Terrain queries, support-polygon geometry, the exact Coulomb block, and the seeded random generator.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Terrain } from '../../src/engine/terrain.mjs';
import { hull2d, polygonMargin, chebyshevCenter } from '../../src/engine/stability.mjs';
import { coulombBlock } from '../../src/engine/world.mjs';
import { createRng } from '../../src/engine/random.mjs';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);

test('terrain: plane height, blocks with foot margin, edge distance and friction patches', () => {
  const T = new Terrain({ slopeDeg: 0, mu: 0.8, boxes: [{ min: [0.3, -0.1, 0], max: [0.5, 0.1, 0.02], mu: 0.5 }], patches: [{ min: [-1, -1], max: [0, 0], mu: 0.2 }] });
  close(T.heightAt(0, 0), 0, 1e-12);
  close(T.heightAt(0.4, 0), 0.02, 1e-12, 'on the block');
  close(T.heightAt(0.295, 0), 0, 1e-12, 'just before the block');
  close(T.heightAt(0.295, 0, 0.01), 0.02, 1e-12, 'a 10 mm foot 5 mm before the edge stands on it');
  close(T.edgeDistance(0.4, 0), 0.1, 1e-12, 'centre of a 200 mm wide block');
  close(T.edgeDistance(0.25, 0), 0.05, 1e-12, 'outside, in front of the face');
  close(T.muAt(-0.5, -0.5), 0.2, 1e-12); close(T.muAt(0.5, 0.5), 0.8, 1e-12);
  close(T.maxHeightAlong([0, 0], [0.6, 0]), 0.02, 1e-12, 'a path across the block');
  const slope = new Terrain({ slopeDeg: 15 });
  close(slope.heightAt(0.2, 0), 0.2 * Math.tan((15 * Math.PI) / 180), 1e-12);
});

test('stability geometry: hull, signed margin and Chebyshev centre of known polygons', () => {
  const square = hull2d([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]).map((p) => [p[0], p[1]]);
  assert.equal(square.length, 4, 'interior point removed');
  close(polygonMargin(square, 0.5, 0.5), 0.5, 1e-12);
  close(polygonMargin(square, 1.2, 0.5), -0.2, 1e-12, 'outside is negative');
  const [cx, cy, r] = chebyshevCenter(square);
  close(cx, 0.5, 1e-9); close(cy, 0.5, 1e-9); close(r, 0.5, 1e-9);
  // regular pentagon of circumradius 1: inradius cos(36°)
  const pent = hull2d([...Array(5)].map((_, k) => [Math.cos((2 * Math.PI * k) / 5), Math.sin((2 * Math.PI * k) / 5)])).map((p) => [p[0], p[1]]);
  const c = chebyshevCenter(pent);
  close(c[2], Math.cos(Math.PI / 5), 1e-9, 'pentagon inradius');
  close(Math.hypot(c[0], c[1]), 0, 1e-9, 'pentagon centre');
});

test('Coulomb block: sticks inside the friction disk, slides anti-parallel to slip on its boundary', () => {
  // A = I, e = slip velocity: unconstrained minimiser x = −e
  const inside = coulombBlock(1, 0, 1, 0.3, -0.4, 1);
  close(inside[0], -0.3, 1e-12); close(inside[1], 0.4, 1e-12);
  const out = coulombBlock(2, 0.3, 1, 3, -1, 0.5);
  close(Math.hypot(out[0], out[1]), 0.5, 1e-9, 'impulse on the disk boundary');
  // KKT: (A + νI) x = −e with ν ≥ 0
  const a = 2, b = 0.3, d = 1, e1 = 3, e2 = -1;
  const nu = -(a * out[0] + b * out[1] + e1) / out[0];
  assert.ok(nu >= 0, `multiplier ${nu}`);
  close(-(b * out[0] + d * out[1] + e2) / out[1], nu, 1e-6, 'same multiplier in both rows');
});

test('seeded random numbers are reproducible and normally distributed', () => {
  const a = createRng(7), b = createRng(7), c = createRng(8);
  const seqA = [...Array(5)].map(() => a.next()), seqB = [...Array(5)].map(() => b.next()), seqC = [...Array(5)].map(() => c.next());
  assert.deepEqual(seqA, seqB);
  assert.notDeepEqual(seqA, seqC);
  const g = createRng(1), n = 20000;
  let sum = 0, sq = 0;
  for (let i = 0; i < n; i++) { const x = g.normal(); sum += x; sq += x * x; }
  close(sum / n, 0, 0.03, 'mean'); close(Math.sqrt(sq / n), 1, 0.03, 'std');
});
