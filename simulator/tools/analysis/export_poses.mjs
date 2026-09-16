// Export rigid-part world poses (metres) from the simulator for offline rendering.
// Usage: node tools/analysis/export_poses.mjs <out.json>
import fs from 'node:fs';
import { createSimulation } from '../../src/sim/simulation.mjs';

const robot = JSON.parse(fs.readFileSync(new URL('../../assets/robot.json', import.meta.url), 'utf8'));
const out = { scale: null, frames: {} };

function partPoses(sim) {
  const { world, model } = sim, kin = world.kin;
  kin.update(world.q, world.v);
  const poses = {};
  for (let i = 0; i < model.nb; i++) {
    const b = model.bodies[i];
    if (!b.part) continue;
    const Rw = kin.Rw[i], pw = kin.pw[i];
    const R = [[Rw[0], Rw[1], Rw[2]], [Rw[3], Rw[4], Rw[5]], [Rw[6], Rw[7], Rw[8]]];
    // x_world = Rw (s·(P x_part + p) ) + pw  with partToBody (P, p) in print mm
    poses[b.part] = { R, t: Array.from(pw), partToBody: b.partToBody };
  }
  const feet = world.contacts.filter((c) => c.kind === 'foot').map((c) => ({ p: c.point, fn: c.fn }));
  return { parts: poses, feet, time: world.time };
}

// standing + walking
const walk = createSimulation(robot, { scenario: 'walk' });
out.scale = walk.meta.design.scale;
out.meta = { totalMass: walk.meta.totalMass, carapaceAcross: walk.meta.carapaceAcross, servo: walk.meta.servo.label };
const times = [0.9, 4.0, 5.6, 7.2, 8.8, 10.4, 12.0];
let k = 0;
while (walk.world.time < times.at(-1) + 1e-9) {
  walk.step();
  if (k < times.length && walk.world.time >= times[k] - 1e-9) { out.frames[k === 0 ? 'stand' : `walk${k}`] = partPoses(walk); k++; }
}
// fist bump (hand 1-B), sampled at the top of the raise
const fb = createSimulation(robot, { scenario: 'fistbump' });
while (fb.world.time < 3.9) fb.step();
out.frames.fistbump = partPoses(fb);
fs.writeFileSync(process.argv[2], JSON.stringify(out));
console.log('frames', Object.keys(out.frames).join(', '), 'scale', out.scale);
