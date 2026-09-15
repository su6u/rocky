// The physics worker's message protocol, run in Node with a mocked worker global and fetch.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { MSG } from '../../src/worker/protocol.mjs';

const robotPath = fileURLToPath(new URL('../../assets/robot.json', import.meta.url));

async function startWorker() {
  const posted = [];
  globalThis.self = { postMessage: (m) => posted.push(m) };
  globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => JSON.parse(readFileSync(robotPath, 'utf8')) });
  await import(`../../src/worker/sim-worker.mjs?${Date.now()}`);
  const send = async (msg) => { await self.onmessage({ data: msg }); };
  const last = (type) => [...posted].reverse().find((m) => m.type === type);
  return { posted, send, last };
}

test('init → ready, load → loaded with terrain and gait envelope, step → frame, drive commands', async () => {
  const w = await startWorker();
  await w.send({ type: MSG.INIT, robotUrl: 'robot.json' });
  const ready = w.last(MSG.READY);
  assert.ok(ready?.scenarios?.drive && ready.servos.sts3215_7v4 && ready.servos.sts3215_12v, 'ready lists scenarios and servos');

  await w.send({ type: MSG.LOAD, config: { scenario: 'drive', servoPreset: 'sts3215_7v4', env: {} } });
  const loaded = w.last(MSG.LOADED);
  assert.ok(loaded, 'loaded');
  assert.equal(loaded.info.spec.interactive, true);
  assert.equal(loaded.info.terrain.boxes.length, 2);
  assert.ok(loaded.info.gait.limits.vMax > 0.01);
  assert.equal(loaded.info.actuators.length, 15);
  assert.equal(loaded.frame.xf.length, loaded.info.bodies.length * 15);

  await w.send({ type: MSG.STEP, seconds: 0.6 });
  await w.send({ type: MSG.COMMAND, vx: 0.015, vy: 0, wz: 0, height: 0 });
  await w.send({ type: MSG.STEP, seconds: 1.0 });
  const frame = w.last(MSG.FRAME).frame;
  assert.ok(frame.t > 1.5, `time ${frame.t}`);
  assert.ok(frame.drive.command.vx > 0.01, 'command reached the controller');
  assert.equal(frame.servos.length, 15);
  assert.ok(Array.isArray(frame.events));

  await w.send({ type: MSG.ACTION, name: 'fistbump', params: {} });
  await w.send({ type: MSG.ACTION, name: 'not-an-action', params: {} });
  await w.send({ type: MSG.STEP, seconds: 0.02 });
  assert.equal(w.last(MSG.FRAME).frame.drive.queued, 1, 'only the valid action was queued');
  assert.equal(w.posted.filter((m) => m.type === MSG.ERROR).length, 0, 'no errors');
});
