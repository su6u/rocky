// Scenario catalogue: behaviours and failure cases, with their environment and build overrides.
//
// Fields: behavior (controller behaviour), params, settle (s before the behaviour starts), duration (s; null = open
// ended), environment ({boxes, patches} on the floor, desk-scale metres), mu, slope, footMaterial, battery,
// controller (controller options), design (robot build), ambient (°C), push, failServo, drop, start ('lying').

// Keyboard playground: a 15 mm step, a 30 mm block to climb or bump into, and a polished tile patch.
export const PLAYGROUND = {
  boxes: [
    { min: [0.34, -0.3, 0], max: [0.8, 0.3, 0.015], label: 'Step 15 mm' },
    { min: [-0.75, 0.38, 0], max: [-0.38, 0.95, 0.03], label: 'Block 30 mm' },
  ],
  patches: [{ min: [-0.65, -1.0], max: [0.05, -0.42], mu: 0.22, label: 'Polished tile μ 0.22' }],
};

export const SCENARIOS = {
  stand: { label: 'Rest', group: 'behaviour', description: 'Quiet stance with Rocky\'s constant micro-motion (breathing heave and sway).', behavior: 'stand', duration: 12 },
  walk: { label: 'Walk', group: 'behaviour', description: 'Pentapod wave gait: one limb swings at a time while the body sways toward the next support polygon\'s Chebyshev centre. Feet lift and land vertically, the pad sensors end each touchdown, and leg odometry places the swing foot where the body really is.', behavior: 'walk', params: { speed: 0.018 }, settle: 1.0, duration: 30 },
  turn: { label: 'Turn in place', group: 'behaviour', description: 'Wave gait with rotational foot placement (yaw rate 0.08 rad/s). Footholds are checked against every shoulder\'s yaw limit and shortened when needed.', behavior: 'turn', params: { turn: 0.08 }, settle: 1.0, duration: 25 },
  fistbump: { label: 'Fist bump', group: 'behaviour', description: 'Shift weight onto limbs 2–5, raise limb 1 (fist hand 1-B) to a pose its skewed joints can actually reach, bump, return.', behavior: 'fistbump', settle: 1.0, duration: 8 },
  startle: { label: 'Startle', group: 'behaviour', description: 'Fast recoil (0.18 s crouch and lean) followed by a slow exponential recovery — peak dynamic servo load.', behavior: 'startle', settle: 1.0, duration: 6 },
  standup: { label: 'Stand up', group: 'behaviour', description: 'From resting on the carapace to full height. Usually the worst-case servo torque event of the day.', behavior: 'standup', start: 'lying', duration: 6 },
  drive: { label: 'Drive (keyboard)', group: 'behaviour', description: 'You drive: W/S forward and back, A/D sideways, Q/E turn, Z/X lower and raise the body, B fist bump, 1–5 wave a limb. Commands go through the same gait, servos and physics as every other scenario, on a floor with a 15 mm step, a 30 mm block and a slippery tile.', behavior: 'teleop', environment: PLAYGROUND, settle: 0.5, duration: null, interactive: true },

  push: { label: 'Push recovery', group: 'failure', description: 'Lateral shove on the carapace (30 % of body weight for 0.15 s) while standing.', behavior: 'stand', push: { at: 2.0, fraction: 0.3, duration: 0.15 }, duration: 8 },
  slippery: { label: 'Slippery floor', group: 'failure', description: 'Walking on polished tile with hard PLA feet (μ ≈ 0.2): watch foot slip and friction-cone saturation.', behavior: 'walk', params: { speed: 0.018 }, mu: 0.2, footMaterial: 'pla', settle: 1.0, duration: 20 },
  slope: { label: 'Slope 15°', group: 'failure', description: 'Walking uphill on a 15° ramp. The body pitches with half the slope, IMU feedback trims the attitude, and friction and tipping margins shrink.', behavior: 'walk', params: { speed: 0.012 }, slope: 15, settle: 1.5, duration: 20 },
  step: { label: 'Step up 20 mm', group: 'failure', description: 'Walking onto a 20 mm block. Swing feet clear the edge, touch down on whatever height the pad sensor finds, and the body height follows the stance feet.', behavior: 'walk', params: { speed: 0.016 }, environment: { boxes: [{ min: [0.32, -0.6, 0], max: [1.2, 0.6, 0.02], label: 'Block 20 mm' }] }, settle: 1.0, duration: 40 },
  lowbattery: { label: 'Low battery', group: 'failure', description: '2S LiPo at 8 % charge with aged cells (R = 0.12 Ω): bus sags under load, torque and speed capacity fall.', behavior: 'walk', params: { speed: 0.018 }, battery: { soc: 0.08, rInternal: 0.12 }, settle: 1.0, duration: 20 },
  servofail: { label: 'Servo failure', group: 'failure', description: 'Limb 3 shoulder-pitch servo loses power at t = 3 s. Its gear train still has friction, so the limb sags under load rather than dropping; the controller sees the missing bus reply within 100 ms and moves the body over the four healthy feet.', behavior: 'stand', failServo: { limb: 3, joint: 'pitch', at: 3.0 }, duration: 10 },
  thermal: { label: 'Overheating', group: 'failure', description: 'Rocky pushes limb 1 against a stuck hatch for ten minutes. The fist is driven 12 mm into the hatch, so its elbow servo stalls at about 68 % duty: full stall current, yet under the 80 % that trips overload protection — the usual way hobby servos burn out. The board sensors climb along the calibrated two-node thermal model; at 70 °C the firmware releases torque, the controller retracts the limb, waits until the servo has cooled and been re-enabled, and pushes again.', behavior: 'press', params: { limb: 1, depth: 0.012 }, environment: { boxes: [{ min: [0.29, -0.09, 0], max: [0.35, 0.09, 0.4], label: 'Stuck hatch', mu: 0.6 }] }, settle: 1.0, duration: 600, fast: true },
  latency: { label: 'Bus latency', group: 'failure', description: 'Walking with a 20 Hz control loop and 60 ms transport delay (slow USB-serial adapter).', behavior: 'walk', params: { speed: 0.018 }, controller: { rate: 20, busDelay: 0.06 }, settle: 1.0, duration: 20 },
  canon: { label: 'Canon mass', group: 'failure', description: 'Film scale (18 in carapace) with the book\'s 168 kg on the same hobby servos — expected to collapse.', behavior: 'stand', design: { scale: 'canon', canonMass: 168 }, duration: 6 },
  drop: { label: 'Drop 8 cm', group: 'failure', description: 'Released 8 cm above the floor in standing pose: impact forces on feet and gearboxes.', behavior: 'stand', drop: 0.08, duration: 4 },
};
