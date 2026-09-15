// Keyboard teleoperation. Held keys become a body-frame velocity command scaled to the gait envelope the worker
// reports for this build, so the keyboard asks for what the robot can deliver and the physics decides what happens.
// Physical key codes (event.code) keep WASD in place on AZERTY, Dvorak and other layouts.
export const DRIVE_KEYS = {
  forward: ['KeyW', 'ArrowUp'], back: ['KeyS', 'ArrowDown'],
  left: ['KeyA'], right: ['KeyD'],
  turnLeft: ['KeyQ', 'ArrowLeft'], turnRight: ['KeyE', 'ArrowRight'],
  lower: ['KeyZ'], raise: ['KeyX'],
  hurry: ['ShiftLeft', 'ShiftRight'],
};
const HEIGHT_RATE = 0.02;                          // m/s of body-height change while Z/X is held
const HEIGHT_RANGE = [-0.035, 0.015];              // m, matches the controller's clamp at desk scale
const SEND_INTERVAL = 0.25;                        // s between keep-alive commands while keys are held

export class KeyboardTeleop {
  // send(command) posts {vx, vy, wz, height} to the worker; onAction(name, params) posts discrete actions.
  constructor({ send, onAction, isActive }) {
    Object.assign(this, { send, onAction, isActive });
    this.held = new Set();
    this.height = 0; this.last = null; this.lastSent = 0; this.limits = { vMax: 0.02, rMax: 0.24, lengthScale: 1 };
    this.command = { vx: 0, vy: 0, wz: 0, height: 0 };
    this.onKeyDown = this.onKeyDown.bind(this); this.onKeyUp = this.onKeyUp.bind(this); this.release = this.release.bind(this);
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.release);
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.release(); });
  }
  setLimits({ vMax, rMax, lengthScale = 1 }) {
    this.limits = { vMax: vMax || 0.02, rMax: rMax || 0.24, lengthScale };
    this.height = Math.max(HEIGHT_RANGE[0] * lengthScale, Math.min(HEIGHT_RANGE[1] * lengthScale, this.height));
  }
  reset() { this.held.clear(); this.height = 0; this.last = null; this.command = { vx: 0, vy: 0, wz: 0, height: 0 }; }
  release() { if (!this.held.size) return; this.held.clear(); this.flush(true); }
  isDriveKey(code) { return Object.values(DRIVE_KEYS).some((list) => list.includes(code)); }
  down(name) { return DRIVE_KEYS[name].some((code) => this.held.has(code)) ? 1 : 0; }

  onKeyDown(e) {
    if (!this.isActive() || e.metaKey || e.ctrlKey || e.altKey || e.target?.closest?.('input, select, textarea')) return;
    if (this.isDriveKey(e.code)) {
      e.preventDefault();                          // arrows must not scroll the page while driving
      this.held.add(e.code);
      return;
    }
    if (e.repeat) return;
    const digit = /^Digit([1-5])$/.exec(e.code);
    if (e.code === 'KeyB') this.onAction('fistbump', {});
    else if (digit) this.onAction('wave', { limb: +digit[1] });
    else if (e.code === 'KeyC') this.onAction('sit', {});
    else if (e.code === 'KeyT') this.onAction('startle', {});
  }
  onKeyUp(e) { if (this.held.delete(e.code)) this.flush(false); }

  // Called every animation frame with the frame time; sends when the command changes or as a keep-alive.
  update(dt) {
    if (!this.isActive()) return;
    const L = this.limits, s = L.lengthScale;
    const dh = (this.down('raise') - this.down('lower')) * HEIGHT_RATE * s * dt;
    this.height = Math.max(HEIGHT_RANGE[0] * s, Math.min(HEIGHT_RANGE[1] * s, this.height + dh));
    const scale = this.down('hurry') ? 1 : 0.6;
    const fwd = this.down('forward') - this.down('back'), side = this.down('left') - this.down('right'), turn = this.down('turnLeft') - this.down('turnRight');
    // share the envelope between translation and rotation when both are held (the controller clamps again)
    const n = Math.max(1, Math.hypot(fwd, side) + Math.abs(turn));
    this.command = {
      vx: (fwd / n) * L.vMax * scale, vy: (side / n) * L.vMax * scale, wz: (turn / n) * (L.vMax / L.rMax) * scale, height: this.height,
    };
    this.flush(false);
  }
  flush(force) {
    const c = this.command, now = performance.now() / 1000;
    const changed = !this.last || Math.abs(c.vx - this.last.vx) + Math.abs(c.vy - this.last.vy) + Math.abs(c.wz - this.last.wz) > 1e-6 || Math.abs(c.height - this.last.height) > 5e-4;
    const moving = Math.abs(c.vx) + Math.abs(c.vy) + Math.abs(c.wz) > 0;
    if (force || changed || (moving && now - this.lastSent > SEND_INTERVAL)) {
      if (force) this.command = { ...c, vx: 0, vy: 0, wz: 0 };
      this.send({ ...this.command });
      this.last = { ...this.command }; this.lastSent = now;
    }
  }
}
