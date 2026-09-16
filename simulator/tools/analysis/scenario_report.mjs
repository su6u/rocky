// Scenario benchmark: runs scenarios headless and prints stability, slip, servo load, thermal and power metrics.
//   node tools/analysis/scenario_report.mjs                       every scenario (open-ended ones capped at 30 s)
//   node tools/analysis/scenario_report.mjs walk turn --max 20    selected scenarios, capped duration
//   node tools/analysis/scenario_report.mjs --servo sts3215_12v --json reports/12v.json
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { loadRobot, runScenario, formatRow, SCENARIOS } from './metrics.mjs';

const args = process.argv.slice(2);
const flag = (name) => { const i = args.indexOf(name); if (i < 0) return undefined; const v = args[i + 1]; args.splice(i, 2); return v; };
const maxDuration = +(flag('--max') ?? 30);
const servoPreset = flag('--servo');
const jsonOut = flag('--json');
const keys = args.length ? args : Object.keys(SCENARIOS);
const robot = loadRobot();
const results = [];
for (const key of keys) {
  const m = runScenario(robot, key, { maxDuration, sim: servoPreset ? { servoPreset } : {} });
  console.log(formatRow(m));
  delete m.sim;
  results.push(m);
}
if (jsonOut) {
  mkdirSync(dirname(jsonOut), { recursive: true });
  writeFileSync(jsonOut, JSON.stringify({ generatedAt: new Date().toISOString(), servoPreset: servoPreset ?? 'default', maxDuration, results }, null, 1));
  console.log(`wrote ${jsonOut}`);
}
