// Rocky motion laboratory: wires the physics worker, the 3D view, the 2D kinematics pane, keyboard driving and the
// telemetry UI.
import { RockyView } from '../render/view.mjs';
import { loadManifest } from '../render/assets.mjs';
import { FrameInterpolator } from '../render/interpolate.mjs';
import { MSG } from '../worker/protocol.mjs';
import { PlanView, torsoOutline } from './plan-view.mjs';
import { Chart } from './charts.mjs';
import { KeyboardTeleop } from './teleop.mjs';

const $ = (id) => document.getElementById(id);
const JOINTS = ['yaw', 'pitch', 'elbow'];
const ASSET_BASE = 'assets/';
const CAMERA_MODES = ['orbit', 'chase', 'top'];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const narrowScreen = window.matchMedia('(max-width: 860px)');

const state = {
  scenarios: null, servos: null, statuePose: null, statueUp: null,
  scenario: 'stand', info: null, frame: null, playing: false, ready: false, partsReady: false,
  statue: false, error: null, events: [], records: [], lastRecordT: -1, manifest: null, outline: null, outlineScale: null,
};

const view = new RockyView($('viewport'));
view.reducedMotion = reducedMotion.matches;
const plan = new PlanView($('plan'));
const interp = new FrameInterpolator();
const charts = {
  force: new Chart($('chartForce'), { series: [{ key: 'weight', color: '#77706c', dash: [3, 4], width: 1 }, { key: 'normal', color: '#f4f0eb' }], minRange: 4 }),
  stab: new Chart($('chartStab'), { series: [{ key: 'ssm', color: '#f4f0eb' }, { key: 'zmp', color: '#d8c49f', width: 1.2 }, { key: 'capture', color: '#e8a69d', width: 1.2 }], floor: 0, minRange: 20 }),
  servo: new Chart($('chartServo'), { series: [{ key: 'load', color: '#d8c49f' }, { key: 'heat', color: '#e8a69d', width: 1.2 }], floor: 0, minRange: 10 }),
};

const worker = new Worker(new URL('../worker/sim-worker.mjs', import.meta.url), { type: 'module' });
worker.onmessage = (e) => handleWorker(e.data);
worker.onerror = (e) => showError(`Physics worker failed: ${e.message || 'could not start (serve the folder over http, not file://)'}`);
const send = (msg) => worker.postMessage(msg);

const teleop = new KeyboardTeleop({
  send: (command) => send({ type: MSG.COMMAND, ...command }),
  onAction: (name, params) => { send({ type: MSG.ACTION, name, params }); flashAction(name, params); },
  isActive: () => !!state.info?.spec.interactive && !state.statue && state.playing,
});

boot();

async function boot() {
  if (narrowScreen.matches) { $('shadows').checked = false; view.setShadows(false); }   // phones: performance first
  send({ type: MSG.INIT, robotUrl: new URL('assets/robot.json', document.baseURI).href });
  wireControls();
  state.manifest = await loadManifest(ASSET_BASE + 'manifest.json');
  try {
    await view.loadParts({
      manifest: state.manifest, base: ASSET_BASE, tier: $('tier').value, compression: $('compression').value === 'ktx2',
      onProgress: (done, total, name, baked) => {
        $('loadBar').style.width = `${(100 * done) / total}%`;
        $('loadText').textContent = `Loading parts ${done}/${total} · ${name}${baked ? '' : ' (unbaked: STL fallback)'}`;
      },
    });
  } catch (err) {
    showError(`Could not load Rocky's geometry: ${err.message}`);
    return;
  }
  state.partsReady = true;
  $('loading').hidden = true;
  describeAssets();
  if (state.info) applyInfo(state.info);
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ worker messages
function handleWorker(msg) {
  if (msg.type === MSG.READY) {
    Object.assign(state, { scenarios: msg.scenarios, servos: msg.servos, statuePose: msg.statuePose, statueUp: msg.statueUp, ready: true });
    buildScenarioChips();
    $('servo').innerHTML = Object.entries(msg.servos).map(([k, s]) => `<option value="${k}">${s.label}</option>`).join('');
    const initial = new URLSearchParams(location.search).get('scenario');
    selectScenario(initial && msg.scenarios[initial] ? initial : 'stand');
  } else if (msg.type === MSG.LOADED) {
    state.info = msg.info; state.frame = msg.frame;
    state.events = []; state.records = []; state.lastRecordT = -1; state.error = null;
    Object.values(charts).forEach((c) => c.reset());
    interp.reset(); teleop.reset();
    $('errorBox').hidden = true;
    renderEvents();
    if (state.partsReady) applyInfo(msg.info);
    ingestFrame(msg.frame);
    for (const id of ['play', 'restart', 'step', 'shove', 'export', 'apply', 'defaults']) $(id).disabled = false;
    if (state.autoplay !== false) send({ type: MSG.PLAY });
  } else if (msg.type === MSG.FRAME) {
    ingestFrame(msg.frame);
  } else if (msg.type === MSG.PLAYING) {
    state.playing = msg.value;
    $('play').textContent = msg.value ? 'Pause' : 'Play';
    if (!msg.value) teleop.release();
  } else if (msg.type === MSG.ERROR) {
    state.error = msg.message;
    showError(`Simulation stopped at t = ${msg.t.toFixed(3)} s: ${msg.message}`);
  }
}

function ingestFrame(f) {
  state.frame = f;
  interp.push(f);
  const info = state.info;
  if (!info) return;
  for (const e of f.events) state.events.unshift(e);
  if (f.events.length) { renderEvents(); toast(f.events.at(-1)); }
  const stall = info.servo.stall;
  const peak = Math.max(...f.servos.map((s) => Math.abs(s.tau)));
  const hottest = Math.max(...f.servos.map((s) => s.temp));
  charts.force.push(f.t, { normal: f.normal, weight: f.weight });
  charts.stab.push(f.t, { ssm: mm(f.stab.ssm), zmp: mm(f.stab.zmpMargin), capture: mm(f.stab.captureMargin) });
  charts.servo.push(f.t, { load: (100 * peak) / stall, heat: hottest - (info.spec.ambient ?? 25) });
  if (f.t - state.lastRecordT >= 0.05 - 1e-9) {
    state.lastRecordT = f.t;
    state.records.push({
      t: +f.t.toFixed(4), com: f.com.map((x) => +x.toFixed(5)), normal: +f.normal.toFixed(3), loadedFeet: f.stab.loadedFeet,
      ssm: f.stab.ssm, zmpMargin: f.stab.zmpMargin, captureMargin: f.stab.captureMargin, nesm: f.stab.nesm, slip: f.slip,
      battery: f.battery, drive: f.drive, behavior: f.behavior,
      servos: f.servos.map((s) => ({ limb: s.limb, joint: s.joint, tau: +s.tau.toFixed(4), current: +s.current.toFixed(4), temp: +s.temp.toFixed(3), winding: +s.winding.toFixed(3), q: +s.q.toFixed(5), protected: s.protected, unload: s.unload, failed: s.failed })),
    });
    if (state.records.length > 40000) state.records.shift();
  }
}

function applyInfo(info) {
  if (state.statue) toggleStatue(false);
  view.setInfo(info);
  const torso = view.parts.get('torso');
  if (torso && state.outlineScale !== info.scale) {
    state.outline = torsoOutline(torso.geometry, info.bodies[0].partToBody, info.scale);
    state.outlineScale = info.scale;
  }
  plan.setInfo(info, state.outline);
  const spec = info.spec;
  $('scenarioText').textContent = spec.description;
  $('servoLabel').textContent = info.servo.label;
  $('servoNote').textContent = `|τ| in N·m · bar = share of datasheet stall torque (${info.servo.stall.toFixed(2)} N·m) · board sensor °C (firmware limit ${info.servo.Tmax} °C)`;
  const L = info.gait.limits;
  $('walkNote').textContent = L
    ? `Gait envelope at this build: cycle ${L.tMin.toFixed(1)}–${Math.min(L.tMax, 8).toFixed(1)} s, stride ≤ ${(L.strideMax * 100).toFixed(1)} cm, speed ≤ ${(L.vMax * 1000).toFixed(0)} mm/s${L.feasible ? '' : ' · requested speed was limited'}. Swing order ${info.gait.order.join('→')}.`
    : `Swing order ${info.gait.order.join('→')} · control ${info.controller.rate} Hz · bus delay ${(info.controller.busDelay * 1000).toFixed(0)} ms`;
  $('assetStats').textContent = `Loaded build: ${info.totalMass.toFixed(2)} kg (structure ${info.structuralMass.toFixed(2)} kg, servos ${info.servoMassTotal.toFixed(2)} kg${spec.payload ? `, payload ${spec.payload.toFixed(2)} kg` : ''}), carapace ${(info.carapaceAcross * 100).toFixed(1)} cm, scale ${info.scale.toFixed(4)} m per print-mm, ${info.bodies.length} bodies, ${info.actuators.length} servos, ${info.selfCollisionPairs} self-collision sphere pairs.`;
  buildServoRows(info);
  syncEnvControls();
  // keyboard driving
  const interactive = !!spec.interactive;
  document.body.classList.toggle('driving', interactive);
  $('hud').hidden = !interactive;
  teleop.setLimits({ vMax: L?.vMax, rMax: info.gait.rMax, lengthScale: info.lengthScale });
  if (interactive && view.cameraMode === 'orbit') setCameraMode('chase');
  $('timeline').classList.toggle('live', !spec.duration);
}

// ------------------------------------------------------------------ scenario + build controls
function buildScenarioChips() {
  const make = (group, host) => {
    host.innerHTML = '';
    for (const [key, s] of Object.entries(state.scenarios)) {
      if (s.group !== group) continue;
      const b = document.createElement('button');
      b.textContent = s.label; b.dataset.scenario = key; b.dataset.group = group;
      if (s.interactive) b.classList.add('interactive');
      b.onclick = () => selectScenario(key);
      host.append(b);
    }
  };
  make('behaviour', $('behaviours'));
  make('failure', $('failures'));
}

function scenarioDefaults(key) {
  const s = state.scenarios[key];
  return { mu: s.mu ?? 0.85, slope: s.slope ?? 0, speed: s.params?.speed ?? null, soc: s.battery?.soc ?? 0.95 };
}

function selectScenario(key) {
  state.scenario = key;
  document.querySelectorAll('[data-scenario]').forEach((b) => { b.classList.toggle('on', b.dataset.scenario === key); b.setAttribute('aria-pressed', b.dataset.scenario === key); });
  const spec = state.scenarios[key];
  setEnvControls(scenarioDefaults(key));
  const rate = spec.fast ? 'max' : '1';
  $('rate').value = rate;
  send({ type: MSG.RATE, value: spec.fast ? 'max' : 1 });
  const url = new URL(location.href); url.searchParams.set('scenario', key); history.replaceState(null, '', url);
  loadScenario();
}

function loadScenario() {
  if (!state.ready) return;
  const d = scenarioDefaults(state.scenario), env = readEnvControls();
  const overrides = {};
  for (const k of ['mu', 'slope', 'soc']) if (Math.abs(env[k] - d[k]) > 1e-9) overrides[k] = env[k];
  if (d.speed != null && Math.abs(env.speed - d.speed) > 1e-9) overrides.speed = env.speed;
  state.autoplay = true;
  send({ type: MSG.LOAD, config: { scenario: state.scenario, servoPreset: $('servo').value, hand: $('hand').value, backlash: $('backlash').checked, env: overrides } });
}

function setEnvControls(v) {
  $('mu').value = v.mu; $('slope').value = v.slope; $('soc').value = v.soc;
  $('speed').disabled = v.speed == null;
  if (v.speed != null) $('speed').value = v.speed;
  syncEnvControls();
}
function readEnvControls() { return { mu: +$('mu').value, slope: +$('slope').value, speed: +$('speed').value, soc: +$('soc').value }; }
function syncEnvControls() {
  const v = readEnvControls();
  $('muOut').textContent = v.mu.toFixed(2);
  $('slopeOut').textContent = `${v.slope.toFixed(1)}°`;
  $('speedOut').textContent = $('speed').disabled ? 'n/a' : `${(v.speed * 1000).toFixed(0)} mm/s`;
  $('socOut').textContent = `${Math.round(v.soc * 100)} %`;
  for (const id of ['mu', 'slope', 'speed', 'soc']) {
    const input = $(id);
    const progress = 100 * (+input.value - +input.min) / (+input.max - +input.min);
    input.style.setProperty('--range-progress', `${progress}%`);
  }
}

function setCameraMode(mode) {
  view.setCameraMode(mode);
  document.querySelectorAll('[data-cam]').forEach((b) => { b.classList.toggle('on', b.dataset.cam === mode); b.setAttribute('aria-pressed', b.dataset.cam === mode); });
}

function shove() {
  // push from the camera's side toward the robot, so "shove" means what the viewer sees
  const theta = view.cameraMode === 'orbit' ? view.orbit.theta : view.chase.theta ?? view.orbit.theta;
  const angle = Math.atan2(Math.cos(theta), Math.sin(theta)) + Math.PI;
  send({ type: MSG.SHOVE, options: { fraction: 0.3, duration: 0.15, angle } });
}

function wireControls() {
  $('play').onclick = () => send({ type: state.playing ? MSG.PAUSE : MSG.PLAY });
  $('restart').onclick = () => { toggleStatue(false); loadScenario(); };
  $('step').onclick = () => { toggleStatue(false); send({ type: MSG.STEP, seconds: 0.02 }); };
  $('rate').onchange = (e) => send({ type: MSG.RATE, value: e.target.value === 'max' ? 'max' : +e.target.value });
  $('shove').onclick = shove;
  $('apply').onclick = () => { toggleStatue(false); loadScenario(); };
  $('defaults').onclick = () => { setEnvControls(scenarioDefaults(state.scenario)); loadScenario(); };
  for (const id of ['mu', 'slope', 'speed', 'soc']) $(id).oninput = () => syncEnvControls();
  document.querySelectorAll('[data-plan]').forEach((b) => b.onclick = () => {
    document.querySelectorAll('[data-plan]').forEach((x) => x.classList.toggle('on', x === b));
    plan.setMode(b.dataset.plan);
  });
  document.querySelectorAll('[data-mode]').forEach((b) => b.onclick = () => {
    document.querySelectorAll('[data-mode]').forEach((x) => x.classList.toggle('on', x === b));
    view.setMode(b.dataset.mode);
  });
  document.querySelectorAll('[data-cam]').forEach((b) => b.onclick = () => setCameraMode(b.dataset.cam));
  document.querySelectorAll('[data-action]').forEach((b) => b.onclick = () => {
    const [name, limb] = b.dataset.action.split(':');
    if (name === 'shove') shove();
    else { send({ type: MSG.ACTION, name, params: limb ? { limb: +limb } : {} }); flashAction(name, limb ? { limb: +limb } : {}); }
  });
  $('drive').onclick = () => selectScenario('drive');
  $('follow').onchange = (e) => view.setFollow(e.target.checked);
  $('resetCam').onclick = () => view.resetCamera();
  $('fullscreen').onclick = toggleFullscreen;
  $('help').onclick = () => toggleHelp(true);
  $('helpClose').onclick = () => toggleHelp(false);
  $('shadows').onchange = (e) => view.setShadows(e.target.checked);
  $('grid').onchange = (e) => view.setGrid(e.target.checked);
  $('statue').onclick = () => toggleStatue(!state.statue);
  const switchTextures = async () => {
    if (!state.partsReady) return;
    const tier = $('tier').value, compression = $('compression').value === 'ktx2';
    $('loading').hidden = false; $('loadBar').style.width = '0%'; $('loadText').textContent = `Loading ${tier} textures…`;
    try {
      await view.setTextures({ tier, compression }, state.manifest, ASSET_BASE, (done, total, name) => { $('loadBar').style.width = `${(100 * done) / total}%`; $('loadText').textContent = `Loading ${tier} textures ${done}/${total} · ${name}`; });
    } catch (err) {
      showError(`Texture switch failed: ${err.message}${tier === 'high' ? ' (the high tier is optional and may not be installed; rebuild it with tools/textures/encode_web_textures.py)' : ''}`);
      $('tier').value = view.tier;
    }
    $('loading').hidden = true;
    describeAssets();
  };
  $('tier').onchange = switchTextures;
  $('compression').onchange = switchTextures;
  $('export').onclick = exportRun;
  const stage = $('view');
  document.addEventListener('fullscreenchange', () => {
    const on = document.fullscreenElement === stage;
    stage.classList.toggle('is-fullscreen', on);
    $('fullscreen').textContent = on ? 'Exit full screen' : 'Full screen';
    $('fullscreen').setAttribute('aria-pressed', on);
    if (on) stage.focus();
  });
  reducedMotion.addEventListener?.('change', (e) => { view.reducedMotion = e.matches; });
  window.addEventListener('keydown', (e) => {
    if (e.target?.closest?.('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.code === 'Escape' && !$('helpDialog').hidden) { toggleHelp(false); return; }
    if (e.code === 'Escape' && $('view').classList.contains('is-maximized')) { setMaximized(false); return; }
    if (e.repeat) return;
    if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
    else if (e.code === 'KeyR') $('restart').click();
    else if (e.code === 'KeyF') toggleFullscreen();
    else if (e.code === 'KeyV') setCameraMode(CAMERA_MODES[(CAMERA_MODES.indexOf(view.cameraMode) + 1) % CAMERA_MODES.length]);
    else if (e.code === 'KeyG') shove();
    else if (e.code === 'KeyH' || e.key === '?') toggleHelp($('helpDialog').hidden);
  });
}

// Full screen for the 3D view. Embedded browsers (app webviews, some iframes) accept the request but never enter full
// screen; after 400 ms without a fullscreenchange the view is maximised inside the page instead.
function toggleFullscreen() {
  const stage = $('view');
  if (document.fullscreenElement) { document.exitFullscreen?.(); return; }
  if (stage.classList.contains('is-maximized')) { setMaximized(false); return; }
  const request = stage.requestFullscreen ?? stage.webkitRequestFullscreen;
  const fallback = () => { if (!document.fullscreenElement) setMaximized(true); };
  try {
    const pending = request?.call(stage);
    pending?.catch?.(fallback);
    setTimeout(fallback, 400);
  } catch { fallback(); }
}
function setMaximized(on) {
  const stage = $('view');
  stage.classList.toggle('is-maximized', on);
  document.body.classList.toggle('stage-maximized', on);
  $('fullscreen').textContent = on ? 'Exit full screen' : 'Full screen';
  $('fullscreen').setAttribute('aria-pressed', on);
  if (on) stage.focus();
}

function toggleHelp(open) {
  $('helpDialog').hidden = !open;
  if (open) $('helpClose').focus();
}

function toggleStatue(on) {
  if (on === state.statue) return;
  if (on && (!state.partsReady || !state.info)) return;
  state.statue = on;
  if (on) {
    send({ type: MSG.PAUSE });
    view.setStatue({ pose: state.statuePose, up: state.statueUp, scale: state.info.scale });
    $('statue').textContent = 'Back to simulation';
    $('scenarioText').textContent = 'The supplied sculpture pose, rebuilt from the registered kit parts (physics paused).';
  } else {
    view.setStatue(null);
    $('statue').textContent = 'Sculpture pose';
    if (state.info) $('scenarioText').textContent = state.info.spec.description;
  }
}

function describeAssets() {
  const s = view.partStats();
  const fmt = s.compressedMaps === s.maps && s.maps ? `${s.gpuFormat}` : s.compressedMaps ? `${s.compressedMaps}/${s.maps} maps ${s.gpuFormat}` : 'RGBA8';
  $('partInfo').textContent = `${s.baked.length}/12 kit parts baked · ${(s.triangles / 1000).toFixed(0)}k triangles · ${view.tier} textures ≈${s.textureMB.toFixed(0)} MB GPU (${fmt})${s.fallback.length ? ` · STL fallback: ${s.fallback.join(', ')}` : ''}`;
  if (!s.gpuFormat) $('compression').disabled = true;
}

// ------------------------------------------------------------------ rendering + UI refresh
let lastUi = 0, lastCharts = 0, lastFrameAt = performance.now(), lastA11y = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - lastFrameAt) / 1000);
  lastFrameAt = now;
  teleop.update(dt);
  const f = interp.sample(now) ?? state.frame;
  if (f) view.setFrame(f);
  view.render();
  if (state.frame && state.info && !state.statue) plan.draw(state.frame);
  if (now - lastUi > 100 && state.frame && state.info) { updateReadouts(state.frame, state.info); lastUi = now; }
  if (now - lastCharts > 66) { Object.values(charts).forEach((c) => c.draw()); lastCharts = now; }
  if (now - lastA11y > 2000 && state.frame && state.info) { describeCharts(state.frame, state.info); lastA11y = now; }
  requestAnimationFrame(loop);
}

function updateReadouts(f, info) {
  const minutes = Math.floor(f.t / 60);
  $('clock').textContent = `${String(minutes).padStart(2, '0')}:${(f.t - minutes * 60).toFixed(3).padStart(6, '0')}`;
  $('rtf').textContent = state.playing && f.rtf > 0 ? `${f.rtf.toFixed(2)}× real time` : state.playing ? '' : 'paused';
  const dur = info.spec.duration;
  $('progress').style.width = dur ? `${Math.min(100, (100 * f.t) / dur)}%` : '100%';
  $('timeText').textContent = dur ? `${f.t.toFixed(1)} / ${dur} s` : `${f.t.toFixed(1)} s · live`;
  $('behaviorText').textContent = `${f.behavior}${f.swingLeg >= 0 ? ` · L${f.swingLeg + 1} swing` : ''}`;
  $('fps').textContent = `${view.perf.fps.toFixed(0)} fps · ${view.pixelRatio.toFixed(2)}× pixels`;

  const tone = statusTone(f, info);
  $('statusPill').dataset.tone = tone[0]; $('statusPill').textContent = tone[1];

  const stab = f.stab;
  setMetric('mSupport', `${f.normal.toFixed(1)} N`, `weight ${f.weight.toFixed(1)} N · ${stab.loadedFeet} loaded feet`, stab.loadedFeet < 3 ? 'bad' : null);
  setMetric('mSsm', fmtMm(stab.ssm), `ZMP margin ${fmtMm(stab.zmpMargin)}`, stab.ssm == null ? 'bad' : stab.ssm < 0 ? 'bad' : stab.ssm < 0.01 * info.lengthScale ? 'warn' : null);
  setMetric('mCapture', fmtMm(stab.captureMargin), `NESM ${fmtMm(stab.nesm)}`, stab.captureMargin != null && stab.captureMargin < 0 ? 'warn' : null);
  const ground = info.terrain.boxes?.length ? 0 : Math.tan(info.terrain.slope) * f.com[0];
  setMetric('mBody', `${((f.body.z - ground) * 1000).toFixed(0)} mm`, `speed ${(f.body.speed * 1000).toFixed(1)} mm/s`);
  let peak = f.servos[0];
  for (const s of f.servos) if (Math.abs(s.tau) > Math.abs(peak.tau)) peak = s;
  const loadPct = (100 * Math.abs(peak.tau)) / info.servo.stall;
  setMetric('mLoad', `${loadPct.toFixed(0)} %`, `L${peak.limb} ${peak.joint} ${Math.abs(peak.tau).toFixed(2)} N·m`, loadPct > 100 ? 'bad' : loadPct > 60 ? 'warn' : null);
  let hot = f.servos[0];
  for (const s of f.servos) if (s.temp > hot.temp) hot = s;
  setMetric('mTemp', `${hot.temp.toFixed(1)} °C`, `L${hot.limb} ${hot.joint} · winding ${hot.winding.toFixed(0)} °C · limit ${info.servo.Tmax} °C`, hot.unload === 'overheat' || hot.temp > info.servo.Tmax ? 'bad' : hot.temp > info.servo.Tmax - 15 ? 'warn' : null);
  setMetric('mBus', `${f.battery.v.toFixed(2)} V`, `${f.battery.i.toFixed(2)} A · charge ${(f.battery.soc * 100).toFixed(1)} %`, f.battery.v < info.servo.vMin ? 'bad' : f.battery.v < info.servo.vMin + 1 ? 'warn' : null);
  const feet = f.contacts.filter((c) => c.kind === 'foot' && c.fn > 0.05 * (info.weight / 5));
  const use = feet.length ? Math.max(...feet.map((c) => c.use)) : 0;
  setMetric('mSlip', `${(f.slip * 1000).toFixed(1)} mm/s`, `friction use ${(use * 100).toFixed(0)} %`, f.slip > 0.02 ? 'warn' : null);

  updateServoRows(f, info);
  $('solver').textContent = `${f.solver.rows} constraint rows · PGS ${f.solver.iterations} it · residual ${f.solver.residual.toExponential(1)} m/s · dt ${(info.dt * 1000).toFixed(0)} ms · control ${info.controller.rate} Hz, bus delay ${(info.controller.busDelay * 1000).toFixed(0)} ms`;
  if (info.spec.interactive) updateHud(f, info, tone);
}

// Heads-up display for driving: commanded vs actual speed and turn rate, height, margins, power and heat.
function updateHud(f, info, tone) {
  const d = f.drive, vMax = Math.max(1e-6, info.gait.limits?.vMax ?? 0.02), wMax = vMax / info.gait.rMax;
  const yaw = f.body.yaw, c = Math.cos(yaw), s = Math.sin(yaw);
  // the COM speed swings within every gait cycle (sway, swinging limb); show a ≈1 s average of travel speed
  const raw = [c * f.comVel[0] + s * f.comVel[1], -s * f.comVel[0] + c * f.comVel[1]], k = 0.1;
  state.hudVel = state.hudVel ? state.hudVel.map((v, i) => v + (raw[i] - v) * k) : raw;
  const vBody = state.hudVel;
  const bar = (id, value, max) => {
    const v = Math.max(-1, Math.min(1, value / max)), el = $(id);
    el.style.left = `${v >= 0 ? 50 : 50 + 50 * v}%`; el.style.width = `${50 * Math.abs(v)}%`;
  };
  bar('hudCmdX', d.command.vx, vMax); bar('hudActX', vBody[0], vMax);
  bar('hudCmdY', d.command.vy, vMax); bar('hudActY', vBody[1], vMax);
  bar('hudCmdW', d.command.wz, wMax); bar('hudActW', d.velocity.wz, wMax);
  $('hudSpeed').textContent = `${(Math.hypot(vBody[0], vBody[1]) * 1000).toFixed(1)} mm/s`;
  $('hudLimit').textContent = `envelope ${(vMax * 1000).toFixed(0)} mm/s`;
  $('hudHeight').textContent = `${d.height >= 0 ? '+' : ''}${(d.height * 1000).toFixed(0)} mm`;
  $('hudState').textContent = tone[1];
  $('hudState').dataset.tone = tone[0];
  $('hudMargin').textContent = `margin ${fmtMm(f.stab.ssm)} · ${f.stab.loadedFeet} feet`;
  const hot = f.servos.reduce((a, b) => (b.temp > a.temp ? b : a));
  $('hudPower').textContent = `${f.battery.v.toFixed(2)} V · ${f.battery.i.toFixed(2)} A · ${hot.temp.toFixed(0)} °C L${hot.limb} ${hot.joint}`;
  $('hudBehavior').textContent = d.queued ? `${f.behavior} · ${d.queued} queued` : f.behavior === 'teleop' ? (d.idle ? 'standing' : 'walking') : f.behavior;
}

function flashAction(name, params) {
  const label = name === 'wave' ? `Wave limb ${params.limb}` : name === 'fistbump' ? 'Fist bump' : name === 'sit' ? 'Sit / stand' : name === 'startle' ? 'Startle' : name;
  toast({ kind: 'push', text: `${label} requested (starts once all feet are planted)` });
}

let toastTimer = null;
function toast(e) {
  if (!e || !document.body.classList.contains('driving')) return;
  const el = $('hudToast');
  el.textContent = e.text; el.dataset.kind = e.kind; el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3500);
}

// Text alternatives for the canvases, refreshed every two seconds for screen readers.
function describeCharts(f, info) {
  $('chartForce').setAttribute('aria-label', `Ground reaction ${f.normal.toFixed(1)} newtons against a weight of ${f.weight.toFixed(1)} newtons`);
  $('chartStab').setAttribute('aria-label', `Static stability margin ${fmtMm(f.stab.ssm)}, ZMP margin ${fmtMm(f.stab.zmpMargin)}, capture margin ${fmtMm(f.stab.captureMargin)}`);
  const peak = Math.max(...f.servos.map((s) => Math.abs(s.tau))), hot = Math.max(...f.servos.map((s) => s.temp));
  $('chartServo').setAttribute('aria-label', `Peak servo torque ${((100 * peak) / info.servo.stall).toFixed(0)} percent of stall, hottest servo ${hot.toFixed(1)} degrees Celsius`);
  $('plan').setAttribute('aria-label', `Top view: ${f.stab.loadedFeet} feet loaded, behaviour ${f.behavior}, body speed ${(f.body.speed * 1000).toFixed(1)} millimetres per second`);
}

function statusTone(f, info) {
  if (state.error) return ['bad', 'Solver stopped'];
  if (state.statue) return ['done', 'Sculpture pose'];
  if (f.complete) return ['done', 'Scenario complete'];
  if (f.servos.some((s) => s.failed)) return ['bad', 'Servo failed'];
  if (f.servos.some((s) => s.unload === 'overheat')) return ['bad', 'Servo overheated'];
  if (f.shellHits.includes('carapace') && f.behavior !== 'sit') return ['bad', 'Carapace on the ground'];
  if (f.battery.v < info.servo.vMin) return ['bad', 'Bus brown-out'];
  if (f.stab.loadedFeet < 3 && f.behavior !== 'sit') return ['bad', 'Unsupported'];
  if (f.stab.ssm != null && f.stab.ssm < 0) return ['bad', 'Tipping'];
  if (f.servos.some((s) => s.protected)) return ['warn', 'Overload protection'];
  if (f.selfHits?.length) return ['warn', 'Limbs colliding'];
  if (f.shellHits.length) return ['warn', f.behavior === 'sit' ? 'Sitting' : 'Shell contact'];
  if (f.stab.captureMargin != null && f.stab.captureMargin < 0) return ['warn', 'Capture point outside'];
  if (f.slip > 0.02) return ['warn', 'Foot slipping'];
  return state.playing ? ['ok', 'Within limits'] : ['idle', 'Paused'];
}

function buildServoRows(info) {
  const limbs = [...new Set(info.actuators.map((a) => a.limb))];
  $('servoRows').innerHTML = limbs.map((limb) => `<tr><th scope="row">L${limb}</th>${JOINTS.map((j) => `<td><div class="cell" id="sv-${limb}-${j}"><b>—</b><em>—</em><i></i></div></td>`).join('')}</tr>`).join('');
}

function updateServoRows(f, info) {
  for (const s of f.servos) {
    const cell = document.getElementById(`sv-${s.limb}-${s.joint}`);
    if (!cell) continue;
    const share = Math.abs(s.tau) / info.servo.stall;
    cell.querySelector('b').textContent = Math.abs(s.tau).toFixed(2);
    cell.querySelector('em').textContent = `${s.temp.toFixed(1)}°`;
    cell.style.setProperty('--load', `${Math.min(100, share * 100).toFixed(0)}%`);
    cell.dataset.state = s.failed || s.unload ? 'failed' : s.protected ? 'protect' : share > 0.8 ? 'over' : Math.abs(s.tau) > info.servo.rated ? 'high' : 'ok';
    cell.dataset.hot = s.temp > info.servo.Tmax - 15 ? '1' : '0';
    cell.title = `L${s.limb} ${s.joint}: τ ${s.tau.toFixed(3)} N·m (peak ${s.peak.toFixed(2)}, rms ${s.rms.toFixed(2)}), I ${s.current.toFixed(2)} A, duty ${(s.duty * 100).toFixed(0)} %, board ${s.temp.toFixed(1)} °C, winding ${s.winding.toFixed(1)} °C, θ ${((s.q * 180) / Math.PI).toFixed(1)}°${s.failed ? ' · FAILED' : s.unload ? ` · torque released (${s.unload})` : s.protected ? ' · overload protection' : ''}`;
  }
}

function renderEvents() {
  const list = $('events');
  if (!state.events.length) { list.innerHTML = '<li class="empty">No events yet</li>'; return; }
  list.innerHTML = state.events.slice(0, 40).map((e) => `<li data-kind="${e.kind}"><time>${e.t.toFixed(2)} s</time><span>${escapeHtml(e.text)}</span></li>`).join('');
}

function setMetric(id, value, sub, tone = null) {
  const el = $(id);
  el.textContent = value;
  if (sub !== undefined) $(`${id}Sub`).textContent = sub;
  if (tone) el.parentElement.dataset.tone = tone; else delete el.parentElement.dataset.tone;
}

function exportRun() {
  if (!state.info) return;
  const bundle = {
    format: 'rocky-motion-lab/3',
    exportedAt: new Date().toISOString(),
    scenario: state.info.scenario, spec: state.info.spec, config: state.info.config,
    robot: { totalMass: state.info.totalMass, carapaceAcross: state.info.carapaceAcross, scale: state.info.scale, servo: state.info.servo, battery: state.info.battery, gait: state.info.gait, controller: state.info.controller, terrain: state.info.terrain },
    sampleRate: 20,
    events: [...state.events].reverse(),
    samples: state.records,
  };
  const blob = new Blob([JSON.stringify(bundle)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `rocky-${state.info.scenario}-${state.frame ? state.frame.t.toFixed(1) : 0}s.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function showError(text) {
  const box = $('errorBox');
  box.textContent = text; box.hidden = false;
  $('statusPill').dataset.tone = 'bad'; $('statusPill').textContent = 'Error';
  if (!state.partsReady) $('loadText').textContent = text;
}

const mm = (x) => (x == null ? NaN : x * 1000);
const fmtMm = (x) => (x == null ? '—' : `${(x * 1000).toFixed(1)} mm`);
const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
