// Messages between the viewer (main thread) and the physics worker. Both sides import these names so a typo fails
// loudly instead of silently dropping a message.
//
// viewer → worker
//   init     {robotUrl}                         fetch the robot description, reply READY
//   load     {config: {scenario, servoPreset, hand, backlash, env, seed}}   build a simulation, reply LOADED
//   play / pause                                start or stop advancing against the wall clock
//   rate     {value: number | 'max'}            playback rate (sim seconds per wall second)
//   step     {seconds}                          advance a fixed amount while paused, reply FRAME
//   shove    {options: {fraction, duration, angle}}   horizontal push on the carapace
//   command  {vx, vy, wz, height}               teleoperation: body-frame velocity (m/s, rad/s), height offset (m)
//   action   {name, params}                     teleoperation: fistbump | wave {limb} | startle | sit | stand
// worker → viewer
//   ready    {scenarios, servos, statuePose, statueUp}
//   loaded   {info, frame, buildMs}            static description of the robot, scenario, terrain and gait envelope
//   frame    {frame}                            ≈60 Hz snapshot: body transforms, contacts, stability, servos, events
//   playing  {value}
//   error    {message, t}
export const MSG = Object.freeze({
  INIT: 'init', LOAD: 'load', PLAY: 'play', PAUSE: 'pause', RATE: 'rate', STEP: 'step', SHOVE: 'shove', COMMAND: 'command', ACTION: 'action',
  READY: 'ready', LOADED: 'loaded', FRAME: 'frame', PLAYING: 'playing', ERROR: 'error',
});

export const ACTIONS = Object.freeze(['fistbump', 'wave', 'startle', 'sit', 'stand']);

// Layout of the per-body transform block in a frame: world rotation (row-major 9), origin (3), joint axis (3).
export const XF_STRIDE = 15;
