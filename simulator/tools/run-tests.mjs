// Portable test entry point: collects tests/**/*.test.mjs and runs them with Node's built-in test runner.
// (`node --test` only expands glob patterns from Node 21, and Node 22 treats a directory argument as a file.)
//   node tools/run-tests.mjs            every suite
//   node tools/run-tests.mjs robot sim  only these folders under tests/
import { readdirSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const folders = process.argv.slice(2);
const starts = folders.length ? folders.map((f) => join(root, 'tests', f)) : [join(root, 'tests')];
const files = [];
const walk = (dir) => {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) walk(path);
    else if (entry.name.endsWith('.test.mjs')) files.push(path);
  }
};
starts.forEach(walk);
if (!files.length) { console.error('no test files found'); process.exit(1); }
const run = spawnSync(process.execPath, ['--test', ...files.sort()], { stdio: 'inherit' });
process.exit(run.status ?? 1);
