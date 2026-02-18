// =============================================
// SUPER CV PORTFOLIO - Creative JavaScript
// Particles, Typing, Custom Cursor, Animations
// =============================================

// =============================================
// FEATURE FLAGS (v2 Redesign)
// Set to false to disable new features
// =============================================
const FEATURE_FLAGS = {
  ENABLE_SHOWCASE: true,        // New showcase section between hero and about
  ENABLE_GOLD_THEME: true,      // Gold + white theme (experimental)
  DISABLE_PARTICLES: true,      // Remove particle canvas background
  DISABLE_CUSTOM_CURSOR: true,  // Use system cursor instead
  SIMPLIFIED_HERO: true,        // Reduced hero height, no rings
  SIMPLIFIED_SKILLS: true       // Text-based skills, no progress bars
};

// Flagship project IDs for showcase section
const FLAGSHIP_PROJECT_IDS = ['naamika', 'asap', 'xr-teleop', 'rl-training-center', 'rc-uav', 'atv'];

// Password-protected project IDs (content hidden until password entered)
// Default list — overridden by localStorage if user has customized
const DEFAULT_PROTECTED_IDS = ['naamika', 'xr-teleop', 'surgical-robot'];
let PROTECTED_PROJECT_IDS = JSON.parse(localStorage.getItem('protectedProjectIds')) || [...DEFAULT_PROTECTED_IDS];
// SHA-256 hash of the access password (change hash when changing password)
// Current password: "recruiter2026"
const PROTECTED_HASH = '9f2feb701519cf3605ae8075e865857b97b17927bbd5045e33e3fb498a43b040';
let protectedUnlocked = false;

async function checkProtectedPassword(input) {
  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  return hashHex === PROTECTED_HASH;
}

function isProjectProtected(projectId) {
  return PROTECTED_PROJECT_IDS.includes(projectId) && !protectedUnlocked;
}

function saveProtectedIds() {
  localStorage.setItem('protectedProjectIds', JSON.stringify(PROTECTED_PROJECT_IDS));
}

function showProtectionSettings() {
  if (!protectedUnlocked) {
    showPasswordModal(() => showProtectionSettings());
    return;
  }

  const existing = document.getElementById('protectionSettingsModal');
  if (existing) existing.remove();

  const allProjects = [...projects, ...ongoingProjects];

  const modal = document.createElement('div');
  modal.id = 'protectionSettingsModal';
  modal.className = 'pw-modal-overlay';
  modal.innerHTML = `
    <div class="pw-modal-card" style="max-width:460px;text-align:left">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
        <div class="pw-modal-icon" style="margin:0;width:36px;height:36px;flex-shrink:0">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:18px;height:18px">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
          </svg>
        </div>
        <h3 style="margin:0;font-size:1.05rem">Protection Settings</h3>
      </div>
      <p style="color:var(--text-tertiary);font-size:0.8rem;margin:0 0 16px">Select which projects require password to view:</p>
      <div id="protectionCheckboxes" style="max-height:350px;overflow-y:auto;margin-bottom:16px">
        ${allProjects.map(p => `
          <label style="display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;cursor:pointer;transition:background 0.15s;margin-bottom:2px"
            onmouseenter="this.style.background='var(--surface-hover)'" onmouseleave="this.style.background='transparent'">
            <input type="checkbox" value="${p.id}" ${PROTECTED_PROJECT_IDS.includes(p.id) ? 'checked' : ''}
              style="width:16px;height:16px;accent-color:var(--gold-primary);cursor:pointer">
            <span style="color:var(--text-primary);font-size:0.85rem;flex:1">${p.shortTitle || p.title}</span>
            <span style="color:var(--text-tertiary);font-size:0.7rem">${p.category}</span>
          </label>
        `).join('')}
      </div>
      <div class="pw-modal-actions">
        <button id="protSettingsCancel" class="pw-btn-cancel">Cancel</button>
        <button id="protSettingsSave" class="pw-btn-unlock">Save</button>
      </div>
    </div>`;
  document.body.appendChild(modal);

  document.getElementById('protSettingsSave').addEventListener('click', () => {
    const checked = modal.querySelectorAll('input[type="checkbox"]:checked');
    PROTECTED_PROJECT_IDS = Array.from(checked).map(cb => cb.value);
    saveProtectedIds();
    modal.remove();
    // Re-render to update lock badges
    renderProjectsCatalog();
    renderShowcase();
  });

  document.getElementById('protSettingsCancel').addEventListener('click', () => modal.remove());
  modal.addEventListener('click', e => { if (e.target === modal) modal.remove(); });
}

function showPasswordModal(onSuccess) {
  const existing = document.getElementById('passwordModal');
  if (existing) existing.remove();

  const modal = document.createElement('div');
  modal.id = 'passwordModal';
  modal.className = 'pw-modal-overlay';
  modal.innerHTML = `
    <div class="pw-modal-card">
      <div class="pw-modal-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
          <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
        </svg>
      </div>
      <h3>Protected Content</h3>
      <p class="pw-subtitle">This project contains confidential material.<br>Enter the access password to view.</p>
      <input type="password" id="protectedPwInput" placeholder="Enter password" autocomplete="off">
      <div id="pwError" class="pw-error">Incorrect password</div>
      <div class="pw-modal-actions">
        <button id="pwCancel" class="pw-btn-cancel">Cancel</button>
        <button id="pwSubmit" class="pw-btn-unlock">Unlock</button>
      </div>
    </div>`;
  document.body.appendChild(modal);

  const input = document.getElementById('protectedPwInput');
  const submit = document.getElementById('pwSubmit');
  const cancel = document.getElementById('pwCancel');
  const error = document.getElementById('pwError');

  input.focus();

  async function tryUnlock() {
    const ok = await checkProtectedPassword(input.value);
    if (ok) {
      protectedUnlocked = true;
      modal.remove();
      // Re-render to remove lock badges
      renderProjectsCatalog();
      renderShowcase();
      if (onSuccess) onSuccess();
    } else {
      error.style.display = 'block';
      input.value = '';
      input.focus();
    }
  }

  submit.addEventListener('click', tryUnlock);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') tryUnlock(); });
  cancel.addEventListener('click', () => modal.remove());
  modal.addEventListener('click', e => { if (e.target === modal) modal.remove(); });
}

// Ongoing Projects (separate from completed projects)
const ongoingProjects = [
  {
    id: 'rl-training-center',
    title: 'Isaac Lab RL Training Environment for Precision Arm Reaching',
    shortTitle: 'RL Training Center',
    category: 'Reinforcement Learning',
    status: 'Active Development',
    progress: '~70% Complete',
    description: 'Custom Isaac Lab environment for training PPO-based arm reaching policies on the Unitree G1. 3-phase curriculum learning with distance-based action scaling for precision manipulation.',
    currentWork: 'Extending to bimanual coordination and integrating with arm control GUI',
    tags: ['Isaac Lab', 'RSL-RL', 'PPO', 'PhysX', 'PyTorch', 'Curriculum Learning'],
    thumbnail: 'assets/projects/rl-training-center/img1.png',
    images: [
      'assets/projects/rl-training-center/img1.png',
      'assets/projects/rl-training-center/img2.png',
      'assets/projects/rl-training-center/img3.png',
      'assets/projects/rl-training-center/img4.png',
      'assets/projects/rl-training-center/img5.png',
      'assets/projects/rl-training-center/img6.png',
      'assets/projects/rl-training-center/img7.png'
    ],
    videos: [],
    fullDescription: `
      Custom reinforcement learning environment built on NVIDIA Isaac Lab (Isaac Sim) for training
      precision arm reaching policies on the Unitree G1 humanoid. Uses RSL-RL PPO with a 3-phase
      curriculum that progressively tightens success thresholds from 20mm → 10mm → 5mm.

      <strong>Environment Design (Isaac Lab):</strong>
      • Isaac Lab (Isaac Sim 4.x + PhysX GPU) — NOT Isaac Gym
      • 27-dim observation space: joint pos (7) + joint vel (7) + hand pos (3) + target pos (3) + error vec (3) + distance (1) + prev action (7) - 4
      • 7-dim action space: delta joint angles for single arm
      • Distance-based action scaling: actions shrink as hand approaches target
      • Episode length: 200 steps with auto-reset on success or timeout

      <strong>3-Phase Curriculum:</strong>
      • Phase 1 (0–2000 iter): Success < 20mm, wide workspace, large action scale
      • Phase 2 (2000–5000 iter): Success < 10mm, refined targets, medium scale
      • Phase 3 (5000+ iter): Success < 5mm, precision reaching, small action scale
      • Automatic phase advancement based on moving average success rate > 80%

      <strong>Reward Function:</strong>
      • Dense: -distance_to_target (continuous gradient signal)
      • Sparse: +10 bonus on reaching within threshold
      • Penalties: joint velocity (-0.01), joint acceleration (-0.001), action magnitude (-0.005)
      • Alive bonus: +0.1 per step (encourages exploration)

      <strong>Training Stack:</strong>
      • RSL-RL PPO (NOT Stable-Baselines3): optimized for GPU-parallel sim
      • MLP policy: 256 → 256 → 7 with ELU activations
      • Training: ~2hrs on RTX 3090 for convergence
      • Export: PyTorch checkpoint → loaded directly in arm control GUI
    `,
    technicalDeepDive: {
      sections: [
        {
          title: "Isaac Lab Environment Architecture",
          content: `
            <p>Built on <strong>NVIDIA Isaac Lab</strong> (Isaac Sim 4.x), NOT Isaac Gym. Uses PhysX GPU pipeline for physics with articulation-based robot model:</p>
            <div class="code-block"><code>Environment Config:
  Framework: Isaac Lab (omni.isaac.lab)
  Physics: PhysX GPU pipeline
  Robot: Unitree G1 (URDF → USD articulation)
  Action type: delta joint positions (7-DOF arm)

Observation Space (27-dim):
  joint_positions (7): current arm angles
  joint_velocities (7): current arm velocities
  hand_position (3): end-effector XYZ
  target_position (3): goal XYZ
  error_vector (3): target - hand
  distance (1): ‖error‖
  prev_action (7): last commanded deltas
  (minus 4 for internal processing)

Action Space (7-dim):
  delta_joints: clipped to [-action_scale, +action_scale]
  action_scale: varies by curriculum phase</code></div>
          `
        },
        {
          title: "3-Phase Curriculum Learning",
          content: `
            <p>Progressive difficulty scaling automatically advances as the policy improves:</p>
            <div class="code-block"><code>Phase 1 — Coarse Reaching (0–2000 iter):
  Success threshold: 20mm
  Action scale: 0.05 rad (large corrections OK)
  Workspace: full arm range
  Target: learn general reaching behavior

Phase 2 — Medium Precision (2000–5000 iter):
  Success threshold: 10mm
  Action scale: 0.03 rad (moderate)
  Workspace: refined region
  Target: consistent sub-cm reaching

Phase 3 — Fine Precision (5000+ iter):
  Success threshold: 5mm
  Action scale: 0.01 rad (tiny corrections)
  Distance-based scaling activates
  Target: sub-5mm terminal accuracy

Advancement Criterion:
  Moving average success rate > 80% over 500 episodes
  Automatic phase transition (no manual intervention)</code></div>
          `
        },
        {
          title: "Distance-Based Action Scaling",
          content: `
            <p>Key innovation: action magnitude automatically scales with distance to target, enabling both fast gross motion and precise fine adjustments:</p>
            <div class="code-block"><code>scale_factor = clamp(distance / d_max, 0.1, 1.0)
effective_action = raw_action × action_scale × scale_factor

Effect:
  Far (>10cm): Full action range → fast approach
  Medium (3–10cm): Reduced actions → controlled approach
  Close (<3cm): Minimal actions → precision adjustment

This prevents overshoot near the target without
sacrificing speed during gross motion.</code></div>
            <p>Combined with the curriculum, this produces policies that achieve ~3cm accuracy in Phase 1, ~1cm in Phase 2, and ~5mm in Phase 3.</p>
          `
        },
        {
          title: "RSL-RL PPO Training",
          content: `
            <p>Uses <strong>RSL-RL</strong> (from ETH Zürich / Legged Robotics), optimized for GPU-parallel simulation — NOT Stable-Baselines3:</p>
            <div class="code-block"><code>PPO Config:
  Algorithm: RSL-RL ActorCritic PPO
  Policy MLP: [256, 256] + ELU activations
  Value MLP: [256, 256] + ELU activations
  γ (discount): 0.99
  λ (GAE): 0.95
  clip_range: 0.2
  entropy_coef: 0.005
  learning_rate: 3e-4
  mini_batches: 4
  num_epochs: 5

Training Performance:
  ~2 hours on RTX 3090 (5000 iterations)
  Convergence: reward plateau + >80% success rate
  Checkpoint: model_{iter}.pt every 100 iterations</code></div>
          `
        }
      ]
    }
  }
];

// Project Data
const projects = [
  {
    id: 'naamika',
    title: 'Naamika: Humanoid Arm Manipulation & Control System',
    shortTitle: 'Naamika Humanoid',
    category: 'Robotics Integration',
    flagship: true,
    protected: true,
    problem: 'Precise dual-arm manipulation on a bipedal humanoid requires safe, real-time control across simulation and hardware with zero-jitter motion',
    outcome: 'Production 500Hz orchestrator with PPO-based IK, hot-swappable sim/real backends, ArUco vision tracking, and safety-critical motion control on Unitree G1',
    oneLiner: 'Single-authority 500Hz control loop → PPO IK solver → arm_sdk DDS → Unitree G1 with ArUco-guided reaching',
    description: `
      Production arm manipulation system for the Unitree G1 humanoid. Single-authority 500Hz orchestrator with PPO-based IK, hot-swappable sim/real backends, and ArUco vision-guided reaching. Deploys on real hardware via arm_sdk/DDS with safety-critical motion control.
    `,
    tags: ['Unitree G1', 'PPO', 'MuJoCo', 'PyQt6', 'arm_sdk', 'DDS', 'ArUco'],
    images: [
      'assets/projects/naamika/img1.jpg',
      'assets/projects/naamika/img2.jpg',
      'assets/projects/naamika/img3.jpg'
    ],
    videos: [
      { src: 'assets/projects/naamika/video1.mp4', title: 'Autonomous Navigation' },
      { src: 'assets/projects/naamika/video2.mp4', title: 'Locomotion Demo' },
      { src: 'assets/projects/naamika/video3.mp4', title: 'Voice Command Response' }
    ],
    thumbnail: 'assets/projects/naamika/img1.jpg',
    technicalDeepDive: {
      sections: [
        {
          title: "Single-Authority Orchestrator Architecture",
          content: `
            <p>The system enforces a strict <strong>single-writer rule</strong>: only the <code>ControlOrchestrator</code> can issue motor commands. GUI, policies, and cameras are advisory only.</p>
            <div class="code-block"><code>Architecture (RULEBOOK-enforced):
MainWindow (PyQt6) ─── signals ──→ ControlOrchestrator (500Hz)
PolicyController ──── callbacks ──→      │
ArUco Camera ──────── targets ───→      │
                                         ↓
                              ┌─── PlantAdapter ───┐
                              │                    │
                         SimPlant              RobotPlant
                         (MuJoCo)             (arm_sdk/DDS)

Threading Model:
  Main thread: Qt event loop (display only)
  Control thread: 500Hz deterministic loop
  Policy thread: PPO inference (~50Hz)
  Render thread: MuJoCo 30 FPS</code></div>
            <p>Mode exclusivity: exactly one mode active at any time. Emergency stop overrides from any thread. Simulation becomes <strong>passive shadow</strong> (display only) when real robot is active.</p>
          `
        },
        {
          title: "PPO Policy as IK Solver",
          content: `
            <p>Instead of analytical IK or Jacobian methods, arm reaching uses a <strong>trained PPO neural network</strong> that implicitly handles joint limits, collision, and smooth motion:</p>
            <div class="code-block"><code>Phase 1 — Internal IK Solve (no robot motion):
  1. Seed MuJoCo env with current robot joint state
  2. Set target position in environment
  3. Run policy.predict(obs) for up to 200 steps
  4. Check convergence: ‖hand_pos - target‖ < 3cm
  5. Extract final joint positions as IK solution

Phase 2 — Smoothstep Blend (robot moves):
  Duration: 3.0s at 50Hz (150 steps)
  Interpolation: t² × (3 - 2t) ease-in-out
  Output: blended joints → orchestrator → plant

Observation (41-dim):
  joint_pos (7) + joint_vel (7) + hand_xyz (3)
  + goal_xyz (3) + delta (3) + step_frac (1)

Action: 7-dim delta angles, clipped [-0.05, 0.05]
Model: ppo_ftp_LEFT/RIGHT_HAND_final.zip (173 KB each)</code></div>
          `
        },
        {
          title: "Plant Abstraction & Hot-Swap",
          content: `
            <p>Unified <code>PlantInterface</code> protocol enables seamless switching between simulation and real hardware:</p>
            <div class="code-block"><code>class PlantInterface(Protocol):
    def send_command(positions: Dict[int, float]) -> bool
    def get_measured_positions() -> Dict[int, float]
    def is_connected() -> bool
    def emergency_stop() -> None
    def get_name() -> str  # 'simulation' or 'robot'

SimPlantAdapter:
  - Wraps MuJoCo widget (qpos read/write)
  - set_active(False) → passive shadow mode
  - No physics stepping when robot is active

RobotPlantAdapter:
  - Wraps arm_sdk via CycloneDDS (LowCmd/LowState)
  - PD control: Kp=50, Kd=1.0 (arms), Kp=200, Kd=5.0 (waist)
  - 29-DOF: arms (14) + waist (3) + legs (12, read-only)

Switching: Sim → Robot
  1. sim_adapter.set_active(False)  # passive shadow
  2. orchestrator.use_robot_plant()
  3. State transfer: current joints seed new plant</code></div>
          `
        },
        {
          title: "Safety-Critical Motion Control",
          content: `
            <p>Multiple safety layers prevent unsafe motion on the 29-DOF humanoid:</p>
            <div class="code-block"><code>Safety Stack:
  1. Jump Detection: |target - current| > 1.5 rad → REJECT
  2. Velocity Limiting: 500Hz pre-computed deltas
     Manual: 1.0 rad/s → 0.002 rad/cycle
     Policy: 2.0 rad/s → 0.004 rad/cycle
  3. Position Error Monitor: |error| > 0.3 rad → auto-DAMP
  4. Emergency DAMP Sequence:
     a) Set _control_running = False (atomic)
     b) Wait for control thread exit (timeout 1s)
     c) Clear all pending commands
     d) Release weight gradually over 1s (50 steps)
     e) Send DAMP to high-level FSM

Boot FSM (safe startup):
  ZERO_TORQUE → [2s] → DAMP → [2s] → STAND_UP → [8s]
  → READY → [3s] → 10s hold → ARM CONTROL ACTIVE

Invariants (from RULEBOOK.md):
  ✓ Single control loop (500Hz)
  ✓ Single command writer (orchestrator only)
  ✓ GUI cannot move hardware directly
  ✓ Sim and robot never active simultaneously</code></div>
          `
        },
        {
          title: "ArUco Vision-Guided Reaching",
          content: `
            <p>End-to-end pipeline from camera detection to arm motion:</p>
            <div class="code-block"><code>Pipeline:
  G1 RealSense D435 → TCP socket (5555) → PC
    ↓
  ArUco 4x4_50 detection (15mm markers)
    ↓
  Camera → Robot frame transform
  (uses pelvis height + cam_pitch for dynamic transform)
    ↓
  Target lock (remote A button press)
    ↓
  PPO policy IK (Phase 1: solve internally)
    ↓
  Smoothstep blend (Phase 2: smooth motion, 3s)
    ↓
  Orchestrator → arm_sdk → G1 motors

Verified Results:
  Final reaching error: ~2.8 cm
  Tested with real RealSense on G1
  MuJoCo visualization confirmed smooth motion</code></div>
          `
        }
      ],
      metrics: [
        { value: "500 Hz", label: "Control Rate" },
        { value: "2ms", label: "Loop Period" },
        { value: "29-DOF", label: "Robot Joints" },
        { value: "~3cm", label: "Reach Accuracy" }
      ],
      references: [
        { title: "Unitree SDK2", url: "https://github.com/unitreerobotics/unitree_sdk2" },
        { title: "MuJoCo Docs", url: "https://mujoco.readthedocs.io/en/stable/" },
        { title: "Stable-Baselines3", url: "https://stable-baselines3.readthedocs.io/" },
        { title: "ArUco OpenCV", url: "https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html" }
      ]
    }
  },
  {
    id: 'asap',
    title: 'ASAP: Whole-Body Agile Skills via Sim-to-Real Transfer',
    shortTitle: 'ASAP Motion RL',
    category: 'Reinforcement Learning',
    flagship: true,
    problem: 'Training locomotion and whole-body motion policies in simulation that transfer to real 29-DOF humanoid hardware',
    outcome: 'Deployed CR7 celebration, kicks, jumps, and locomotion policies on Unitree G1 with delta action model for motor dynamics',
    oneLiner: 'PPO + SMPL retargeting in 4096 IsaacGym envs → ONNX export → 50Hz Sim2Real on G1',
    description: `
      Whole-body agile skills on the Unitree G1 via the ASAP framework (RSS 2025). PPO trained across 4096 IsaacGym envs with SMPL motion retargeting, domain randomization, and a delta action model for sim-to-real bridging. Deployed CR7 celebrations, kicks, jumps, and locomotion at 50Hz on real hardware.
    `,
    tags: ['Sim2Real', 'IsaacGym', 'PPO', 'SMPL', 'PyTorch', 'ONNX', 'Hydra'],
    images: [
      'assets/projects/asap/img1.gif',
      'assets/projects/asap/img2.gif',
      'assets/projects/asap/img3.gif',
      'assets/projects/asap/img4.gif'
    ],
    videos: [
      { src: 'assets/projects/asap/video1.mp4', title: 'Initial Training Stage' },
      { src: 'assets/projects/asap/video2.mp4', title: 'Intermediate' }
    ],
    thumbnail: 'assets/projects/asap/img1.gif',
    technicalDeepDive: {
      sections: [
        {
          title: "HumanoidVerse Multi-Simulator Training",
          content: `
            <p>Training uses the <strong>HumanoidVerse</strong> framework supporting multiple GPU-accelerated simulators. Primary training runs on IsaacGym Preview 4 with 4096 parallel environments:</p>
            <div class="code-block"><code>Simulator Support:
  IsaacGym Preview 4: Primary (4096 envs, PhysX GPU)
  IsaacSim 4.2:       Alternative backend
  Genesis 0.2.1:      Experimental support

IsaacGym Config (from Hydra):
  num_envs: 4096
  sim_device: cuda:0
  physics_engine: PhysX (GPU pipeline)
  dt: 0.005s (200Hz physics)
  substeps: 2
  decimation: 4 → 50Hz policy output

Observation (~453-dim actor):
  Phase-based motion tracking with 4-frame history
  DOF positions (29) + velocities (29) per frame
  Base ang_vel (3) + projected gravity (3)
  Target motion: root_pos, root_rot, DOF targets</code></div>
            <p>Hydra config system (OmegaConf) enables composable experiment management — mix robot, task, motion, and training configs via YAML overrides.</p>
          `
        },
        {
          title: "SMPL Motion Retargeting Pipeline",
          content: `
            <p>Human motions from the AMASS dataset are retargeted to G1 joint space via a two-stage SMPL fitting pipeline:</p>
            <div class="code-block"><code>Stage 1 — Shape Fitting (fit_smpl_shape.py):
  SMPL body model → Optimize β shape params
  Loss: ‖skeleton_lengths(SMPL) - skeleton_lengths(G1)‖²
  Output: G1-proportioned SMPL template

Stage 2 — Motion Fitting (fit_smpl_motion.py):
  AMASS .npz → SMPL pose sequence
  Per-frame: Optimize G1 joint angles to match
    SMPL joint positions (position matching)
    SMPL joint rotations (orientation matching)
  Output: .pkl with pose_quat_global, DOF positions/velocities

Retargeted Motions:
  CR7 Siuuu celebration, side jumps, forward jumps
  Kicks (soccer), basketball dribble, walking gaits
  From CMU, KIT, ACCAD motion capture subsets</code></div>
            <p>29-DOF G1 is annealed to 23-DOF during training by locking wrist/finger joints, reducing action space while preserving whole-body expressiveness.</p>
          `
        },
        {
          title: "PPO Training with Domain Randomization",
          content: `
            <p><strong>PPO with GAE</strong> (λ=0.95) optimizes motion tracking reward across 4096 environments simultaneously:</p>
            <div class="code-block"><code>PPO Config:
  γ (discount): 0.99
  λ (GAE): 0.95
  clip_range: 0.2
  entropy_coef: 0.01
  learning_rate: adaptive (KL target 0.01)
  epochs_per_rollout: 5
  horizon: 24 steps/env
  batch_size: 4096 × 24 = 98304 transitions

Domain Randomization:
  Link mass: ±20%
  PD gains: ±25% (Kp and Kd)
  Friction: 0.5–1.25 (ground contact)
  Push perturbations: every 5–10s, up to 1.0 m/s
  Base COM offset: ±10cm (XYZ)
  Control delay: 0–2 steps (0–90ms)
  Torque RFI: randomization limit 0.1</code></div>
            <p>Reward combines motion tracking (joint pos/vel matching), energy penalty, and alive bonus. Phase variable synchronizes policy output with reference motion timing.</p>
          `
        },
        {
          title: "Delta Action Model (Sim2Real Bridge)",
          content: `
            <p>A supervised neural network learns motor command corrections to bridge the sim-to-real gap, compensating for effects not captured in simulation:</p>
            <div class="code-block"><code>Delta Action Model Architecture:
  Input: target joint commands (from policy)
  Network: Linear(in, 256) → ELU → Linear(256, 256) → ELU → Linear(256, out)
  Output: corrected motor commands

Training:
  Data: Open-loop motor trajectories on real G1
  5000 parallel envs for data collection
  Supervised loss: ‖predicted_pos - measured_pos‖²
  Compensates for:
    - Motor friction & backlash
    - Thermal drift
    - Cable stiffness
    - Gearbox nonlinearities</code></div>
            <p>The delta model runs inline during deployment, sitting between the policy output and the motor commands — adding learned corrections at each timestep.</p>
          `
        },
        {
          title: "Sim2Real Deployment (50Hz)",
          content: `
            <p>Trained policies export to ONNX for deployment on the G1's onboard computer. The inference pipeline runs at <strong>50Hz</strong> (20ms loop):</p>
            <div class="code-block"><code>Deployment (newton_controller.py):
  1. Read joint encoders + IMU → state vector
  2. Construct observation (with 4-frame history)
  3. ONNX Runtime inference → action (23-dim)
  4. Delta action model correction
  5. action_scale × action + default_angles → targets
  6. PD control: Kp/Kd per-joint → torque
  7. Send via Unitree SDK2 / CycloneDDS

Deployed Skills:
  - CR7 Siuuu celebration (full body)
  - Forward/side jumps
  - Soccer kicks (left/right)
  - Walking locomotion (decoupled lower body)

Validation:
  MuJoCo sim2sim check before real deployment
  deploy_mujoco.py → visual gait verification</code></div>
            <p>Decoupled architecture: lower body runs locomotion policy independently while upper body executes motion imitation skills. Keyboard/joystick triggers skill selection.</p>
          `
        }
      ],
      metrics: [
        { value: "4096", label: "Parallel Envs" },
        { value: "50Hz", label: "Control Rate" },
        { value: "23-DOF", label: "Action Space" },
        { value: "~453-dim", label: "Actor Obs" }
      ],
      references: [
        { title: "ASAP Framework (RSS 2025)", url: "https://agility.csail.mit.edu/asap/" },
        { title: "IsaacGym Preview 4", url: "https://arxiv.org/abs/2108.10470" },
        { title: "SMPL Body Model", url: "https://smpl.is.tue.mpg.de/" },
        { title: "AMASS Motion Database", url: "https://amass.is.tue.mpg.de/" },
        { title: "PPO Algorithm", url: "https://arxiv.org/abs/1707.06347" }
      ]
    }
  },
  {
    id: 'rc-uav',
    title: 'RC Aircraft & Multi-Rotor UAV Systems',
    shortTitle: 'RC UAV',
    category: 'Aerial Systems',
    problem: 'Building flight-capable UAVs from scratch with stable control characteristics',
    outcome: 'Successful flight tests for both fixed-wing and quadcopter platforms',
    oneLiner: 'Custom fixed-wing aircraft and FPV quadcopter with tuned flight controllers',
    description: `
      Custom-built fixed-wing RC aircraft and FPV racing quadcopter, from airframe design through successful flight testing. Fixed-wing uses foam-board construction with 3-axis servo control; quadcopter runs Betaflight on an F4 FC with GPS position hold and 5.8GHz FPV.
    `,
    tags: ['UAV', 'Aerodynamics', 'Betaflight', 'FPV', 'Arduino', 'RC'],
    images: [
      'assets/projects/rc-uav/img1.jpg',
      'assets/projects/rc-uav/img2.jpg',
      'assets/projects/rc-uav/img3.jpg',
      'assets/projects/rc-uav/img4.jpg',
      'assets/projects/rc-uav/img5.jpg',
      'assets/projects/rc-uav/img6.jpg'
    ],
    videos: [
      { src: 'assets/projects/rc-uav/video1.mp4', title: 'Ground Maneuvering Test' },
      { src: 'assets/projects/rc-uav/video2.mp4', title: 'Flight Test' }
    ],
    thumbnail: 'assets/projects/rc-uav/img5.jpg',
    technicalDeepDive: {
      sections: [
        {
          title: "Fixed-Wing Aircraft Design",
          content: `
            <p>Custom-designed RC aircraft built from scratch with hand-cut foam board construction:</p>
            <div class="code-block"><code>Airframe Specs:
  Material: Foam board (5mm depron)
  Wingspan: ~1200mm
  Wing type: Flat-bottom airfoil (beginner-friendly)
  Control: 3-channel (aileron, elevator, rudder)

Propulsion:
  Motor: 2212 1000KV brushless outrunner
  ESC: 30A BLHeli
  Propeller: 10×4.7 (10" diameter, 4.7" pitch)
  Battery: 3S LiPo 2200mAh (11.1V)
  Thrust-to-weight ratio: >1.3

Servos: SG90 9g micro servos × 3
  Aileron: ±20° deflection
  Elevator: ±25° deflection
  Rudder: ±30° deflection</code></div>
          `
        },
        {
          title: "FPV Quadcopter Build",
          content: `
            <p>Carbon fiber racing quadcopter with GPS and FPV camera system:</p>
            <div class="code-block"><code>Frame: 250mm carbon fiber (X configuration)
  Weight: ~450g all-up

Electronics:
  FC: STM32 F4 running Betaflight 4.x
  ESC: 4-in-1 30A BLHeli_S
  Motors: 2205 2300KV brushless
  Props: 5×4.5 tri-blade

FPV System:
  Camera: CMOS 1200TVL (2.1mm lens, 150° FOV)
  VTX: 5.8GHz 600mW (48 channels)
  Antenna: RHCP cloverleaf (Tx) + patch (Rx)

GPS Module:
  Ublox M8N with compass
  Position hold mode
  Return-to-home failsafe
  Battery: 4S LiPo 1500mAh (14.8V)</code></div>
          `
        },
        {
          title: "Flight Controller & PID Tuning",
          content: `
            <p>Betaflight firmware configuration and PID tuning process for stable flight:</p>
            <div class="code-block"><code>Betaflight Configuration:
  Firmware: 4.x (STM32 F405)
  PID Loop: 8kHz gyro, 4kHz PID
  ESC Protocol: DShot600

PID Tuning (per axis):
  Roll:  P=45, I=80, D=30
  Pitch: P=47, I=85, D=32
  Yaw:   P=35, I=90, D=0

Flight Modes:
  Acro: Full manual (rate mode)
  Angle: Self-leveling (for beginners)
  GPS Hold: Position lock (M8N)
  RTH: Return-to-home on signal loss

Safety:
  Failsafe: RTH at 500m, land at 1km
  Low voltage alarm: 3.5V/cell
  ESC calibration: all-at-once method</code></div>
          `
        }
      ],
      metrics: [
        { value: "1200mm", label: "Fixed-Wing Span" },
        { value: "250mm", label: "Quad Frame" },
        { value: "8kHz", label: "Gyro Rate" },
        { value: "5.8GHz", label: "FPV Video" }
      ]
    }
  },
  {
    id: 'arm-control-gui',
    title: 'Humanoid Arm Control GUI with MuJoCo & PPO IK',
    shortTitle: 'Arm Control GUI',
    category: 'Robotics Integration',
    problem: 'Need unified interface to test arm policies in simulation before deploying to real robot hardware',
    outcome: 'Hot-swappable sim/real backend with 500Hz orchestrator, PPO-based IK, and 30fps MuJoCo viz',
    oneLiner: 'PyQt6 GUI → 500Hz orchestrator → PPO IK solver → PlantAdapter (MuJoCo ↔ arm_sdk/DDS)',
    description: `
      Unified PyQt6 interface for Unitree G1 dual-arm manipulation. 500Hz single-authority orchestrator with PPO-based IK, direct joint control, ArUco vision targeting, and hot-swappable MuJoCo sim / real robot backends via PlantAdapter abstraction.
    `,
    tags: ['PyQt6', 'MuJoCo', 'PPO', 'arm_sdk', 'DDS', 'ArUco'],
    images: [
      'assets/projects/arm-control-gui/screenshot1.png',
      'assets/projects/arm-control-gui/screenshot2.png',
      'assets/projects/arm-control-gui/screenshot3.png',
      'assets/projects/arm-control-gui/screenshot4.png'
    ],
    videos: [
      { src: 'assets/projects/arm-control-gui/demo_video.mp4', title: 'GUI Demo with MuJoCo Simulation' }
    ],
    thumbnail: 'assets/projects/arm-control-gui/screenshot2.png',
    technicalDeepDive: {
      sections: [
        {
          title: "Single-Authority Orchestrator",
          content: `
            <p>Strict <strong>single-writer rule</strong>: only the <code>ControlOrchestrator</code> issues motor commands. GUI, policies, and cameras are advisory only:</p>
            <div class="code-block"><code>Architecture:
MainWindow (PyQt6) ─── signals ──→ ControlOrchestrator (500Hz)
PolicyController ──── callbacks ──→      │
ArUco Camera ──────── targets ───→      │
                                         ↓
                              ┌─── PlantAdapter ───┐
                              │                    │
                         SimPlant              RobotPlant
                         (MuJoCo)             (arm_sdk/DDS)

Threading Model:
  Main thread: Qt event loop (display only)
  Control thread: 500Hz deterministic loop (2ms)
  Policy thread: PPO inference (~50Hz)
  Render thread: MuJoCo 30 FPS offscreen</code></div>
            <p>Mode exclusivity: exactly one mode active. Emergency stop overrides from any thread.</p>
          `
        },
        {
          title: "MuJoCo Visualization Pipeline",
          content: `
            <p>Real-time physics visualization using MuJoCo 3.x offscreen rendering with double-buffered anti-flicker:</p>
            <div class="code-block"><code>Render Pipeline (30Hz):
1. Acquire qpos_lock (thread safety via RLock)
2. renderer.update_scene(data, camera)
3. pixels = renderer.render() → numpy RGB
4. np.copy(pixels) inside lock (isolation)
5. QImage from buffer → QPixmap
6. Scale to widget size (bilinear)
7. setPixmap() on QLabel

Anti-Flickering:
  - Double-buffering via pixel copy
  - Re-entrant render prevention
  - Pending render flag for batching
  - Last-good-frame fallback on error

Camera: orbit (drag), pan (right-drag), zoom (scroll)</code></div>
          `
        },
        {
          title: "PlantAdapter Hot-Swap",
          content: `
            <p>Unified <code>PlantInterface</code> protocol enables seamless switching between simulation and real hardware:</p>
            <div class="code-block"><code>class PlantInterface(Protocol):
    def send_command(positions: Dict[int, float]) -> bool
    def get_measured_positions() -> Dict[int, float]
    def is_connected() -> bool
    def emergency_stop() -> None

SimPlantAdapter:
  - MuJoCo data.qpos read/write
  - set_active(False) → passive shadow mode
  - No physics stepping when robot is active

RobotPlantAdapter:
  - arm_sdk via CycloneDDS (LowCmd/LowState)
  - PD control: Kp=50, Kd=1.0 (arms)
  - 29-DOF: arms (14) + waist (3) + legs (12, read-only)

Switching Sim → Robot:
  1. sim_adapter.set_active(False)  # passive
  2. orchestrator.use_robot_plant()
  3. State transfer: joints seed new plant</code></div>
          `
        },
        {
          title: "PPO Policy as IK Solver",
          content: `
            <p>Arm reaching uses a <strong>trained PPO neural network</strong> instead of analytical IK or Jacobian methods:</p>
            <div class="code-block"><code>Phase 1 — Internal IK Solve (no robot motion):
  1. Seed MuJoCo env with current joint state
  2. Set target position in environment
  3. Run policy.predict(obs) for up to 200 steps
  4. Check convergence: ‖hand - target‖ < 3cm
  5. Extract final joint positions as IK solution

Phase 2 — Smoothstep Blend (robot moves):
  Duration: 3.0s at 50Hz (150 steps)
  Interpolation: t² × (3 - 2t) ease-in-out

Observation (41-dim):
  joint_pos (7) + joint_vel (7) + hand_xyz (3)
  + goal_xyz (3) + delta (3) + step_frac (1)

Action: 7-dim delta angles, clipped [-0.05, 0.05]
Model: ppo_ftp_LEFT/RIGHT_HAND_final.zip (173KB)</code></div>
          `
        },
        {
          title: "Safety & Thread Synchronization",
          content: `
            <p>Multiple safety layers prevent unsafe motion. Critical sections protected by <code>threading.RLock</code>:</p>
            <div class="code-block"><code>Safety Stack:
  1. Jump Detection: |target - current| > 1.5 rad → REJECT
  2. Velocity Limiting: 500Hz pre-computed deltas
     Manual: 1.0 rad/s → 0.002 rad/cycle
     Policy: 2.0 rad/s → 0.004 rad/cycle
  3. Position Error Monitor: |error| > 0.3 rad → auto-DAMP
  4. Emergency DAMP: gradual weight release over 1s

Thread Safety (RLock):
  qpos_lock protects MuJoCo data.qpos array
  Non-blocking render: _pending_render flag
  Lock order: control → render (no deadlock)</code></div>
          `
        }
      ],
      metrics: [
        { value: "500Hz", label: "Control Loop" },
        { value: "30 FPS", label: "Render Rate" },
        { value: "29-DOF", label: "Robot Model" },
        { value: "~3cm", label: "IK Accuracy" }
      ],
      references: [
        { title: "MuJoCo Python", url: "https://mujoco.readthedocs.io/en/stable/python.html" },
        { title: "Stable-Baselines3", url: "https://stable-baselines3.readthedocs.io/" },
        { title: "Unitree SDK2", url: "https://github.com/unitreerobotics/unitree_sdk2" }
      ]
    }
  },
  {
    id: 'hand-target',
    title: 'Multi-Method Object Detection & 6-DOF Pose Estimation',
    shortTitle: 'Hand Target',
    category: 'Computer Vision',
    problem: 'Robot arms need stable, accurate 6-DOF target poses for manipulation — single detection method insufficient',
    outcome: 'Sub-centimeter accuracy with 3 detection methods (ArUco + HSV + YOLOv8), median+Kalman temporal smoothing',
    oneLiner: 'RealSense RGB-D → ArUco / HSV / YOLOv8 detection → Median + Kalman smoothing → TCP socket to robot',
    description: `
      Real-time 6-DOF pose estimation for robot arm manipulation with three interchangeable detection backends: ArUco markers, HSV color segmentation, and custom-trained YOLOv8. Median + Kalman temporal smoothing produces jitter-free trajectories streamed via TCP to the robot controller.
    `,
    tags: ['RealSense', 'ArUco', 'YOLOv8', 'Kalman Filter', 'OpenCV', 'Python'],
    images: [],
    videos: [],
    thumbnail: null,
    hideGallery: true,
    technicalDeepDive: {
      sections: [
        {
          title: "Multi-Method Detection Architecture",
          content: `
            <p>Three interchangeable detection backends share a common output interface — position + optional orientation in camera frame:</p>
            <div class="code-block"><code>Detection Methods:
  1. ArUco (aruco_detector.py):
     cv2.aruco.detectMarkers() → corners, ids
     cv2.aruco.estimatePoseSingleMarkers() → rvec, tvec
     Full 6-DOF pose from marker geometry

  2. HSV Color (color_detector.py):
     cv2.cvtColor(BGR→HSV) → inRange threshold
     cv2.findContours() → largest contour centroid
     Depth lookup → 3D position (no orientation)

  3. YOLOv8 (yolo_detector.py):
     Ultralytics YOLOv8 inference on RGB frame
     Bounding box center → depth lookup → 3D position
     Custom trained on 200+ labeled images

Common Output:
  { position: [x, y, z], orientation: [qw, qx, qy, qz],
    confidence: float, method: str }</code></div>
          `
        },
        {
          title: "ArUco Marker System",
          content: `
            <p>Primary detection for precise 6-DOF pose. Multiple 15mm ArUco markers (4×4_50 dictionary) arranged in strip patterns for cylindrical objects:</p>
            <div class="code-block"><code>ArUco Pipeline:
  1. detectMarkers() → corners, ids
  2. For each marker:
     - estimatePoseSingleMarkers(15mm) → rvec, tvec
     - Rodrigues(rvec) → rotation matrix → quaternion
  3. Position: centroid of all marker tvecs
  4. Rotation: quaternion averaging (SLERP-based)

Strip Generator (strip_generator.py):
  Input: marker IDs, count, spacing
  Output: A4-printable PDF strip for wrapping
  Marker size: 15mm (configurable)

Outlier Rejection:
  Markers with ‖tvec - median_tvec‖ > 2σ discarded
  Quaternion dot product < 0.9 → inconsistent → reject</code></div>
          `
        },
        {
          title: "Temporal Smoothing: Median + Kalman Filter",
          content: `
            <p>Raw detections are noisy. The pipeline uses <strong>median filter + Kalman filter</strong> (NOT exponential moving average) for stable robot-friendly output:</p>
            <div class="code-block"><code>Smoothing Pipeline:
  Raw detection → Median filter (window=5)
                → Kalman filter (state estimation)
                → Output to robot controller

Median Filter:
  Sliding window of last 5 detections
  Removes outlier spikes (robust to noise)
  Applied independently to x, y, z

Kalman Filter:
  State: [x, y, z, vx, vy, vz]
  Prediction: constant velocity model
  Measurement: median-filtered position
  Benefits: velocity prediction fills dropped frames

Quaternion Flip Prevention:
  if np.dot(q_new, q_prev) < 0:
      q_new = -q_new  # Shorter SLERP path</code></div>
            <p>Output rate: 30Hz (matching camera). Kalman prediction bridges gaps when detection momentarily fails.</p>
          `
        },
        {
          title: "YOLOv8 Custom Training",
          content: `
            <p>For objects without markers, a custom-trained YOLOv8 model provides bounding-box detection with depth-based 3D position:</p>
            <div class="code-block"><code>Dataset:
  200+ labeled images of target objects
  Annotations: bounding boxes (YOLO format)
  Augmentation: flip, rotate, brightness, blur
  Split: 80% train, 20% val

Training:
  Model: YOLOv8n (nano — fast inference)
  Framework: Ultralytics
  Epochs: 100
  Input: 640×640 RGB

Inference Pipeline:
  1. YOLO predict on RGB frame → bbox + confidence
  2. bbox center → pixel coordinates (u, v)
  3. depth_frame.get_distance(u, v) → Z meters
  4. Deproject: X = (u - cx) * Z / fx
  5. Output: 3D position (no orientation)</code></div>
            <p>Fallback chain: ArUco (best) → YOLOv8 (good) → HSV color (basic). Automatic method selection based on detection confidence.</p>
          `
        }
      ],
      metrics: [
        { value: "30 FPS", label: "Processing Rate" },
        { value: "< 1cm", label: "Position Accuracy" },
        { value: "3", label: "Detection Methods" },
        { value: "200+", label: "YOLO Training Images" }
      ],
      references: [
        { title: "ArUco Library", url: "https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html" },
        { title: "RealSense SDK", url: "https://github.com/IntelRealSense/librealsense" },
        { title: "Ultralytics YOLOv8", url: "https://docs.ultralytics.com/" },
        { title: "Kalman Filtering", url: "https://www.kalmanfilter.net/" }
      ]
    }
  },
  {
    id: 'g1-isaac-training',
    title: 'Isaac Lab Teleoperation & Data Collection for G1',
    shortTitle: 'Isaac Lab Teleop',
    category: 'Simulation & Teleoperation',
    problem: 'Need high-fidelity simulation environment for G1 teleoperation testing and imitation learning data collection',
    outcome: 'Isaac Lab environment with multi-modal observation (87-dim joints + camera + IMU), DDS teleoperation, and episode recording',
    oneLiner: 'Isaac Lab (Isaac Sim) → G1 articulation model → DDS teleoperation → HDF5 episode recording',
    description: `
      Isaac Lab (Isaac Sim 4.x) environment for the Unitree G1, built for teleoperation testing and imitation learning data collection. Multi-modal observations (87-dim joints + camera + IMU), DDS-based teleop using the same topics as the real robot, and HDF5 episode recording for downstream learning.
    `,
    tags: ['Isaac Lab', 'Isaac Sim', 'CycloneDDS', 'Teleoperation', 'PyTorch', 'HDF5'],
    images: [],
    videos: [],
    thumbnail: null,
    hideGallery: true,
    github: 'https://github.com/bhaveshbakshi633/G1_Isaac_training',
    technicalDeepDive: {
      sections: [
        {
          title: "Isaac Lab Environment Architecture",
          content: `
            <p>Built on <strong>Isaac Lab</strong> (omni.isaac.lab) — the successor to Isaac Gym. Uses articulation-based physics with GPU pipeline, NOT the deprecated Isaac Gym task framework:</p>
            <div class="code-block"><code>Environment Config:
  Framework: Isaac Lab (Isaac Sim 4.x)
  Physics: PhysX GPU (articulation-based)
  num_envs: 1 (teleoperation mode)
  Robot: G1 URDF → USD articulation
  Step: dt=0.005s (200Hz physics)

Key Difference from Isaac Gym:
  Isaac Gym: 4096 parallel envs, RL-focused
  Isaac Lab: Manager-based, modular, sim+real
  This project: Single-env teleoperation + data collection

Manager Architecture:
  ObservationManager → multi-modal obs
  ActionManager → joint position targets
  EventManager → reset, recording triggers
  TerminationManager → episode boundaries</code></div>
          `
        },
        {
          title: "Multi-Modal Observation Space",
          content: `
            <p>Rich observation combining proprioception, vision, and vestibular data:</p>
            <div class="code-block"><code>Joint States (87-dim):
  - Joint positions (29): all actuated DOFs
  - Joint velocities (29): angular rates
  - Joint efforts (29): torque feedback

Camera Observations:
  - RGB: 640×480 @ 30Hz
  - Depth: 640×480 @ 30Hz (aligned)
  - Camera intrinsics: fx, fy, cx, cy

IMU Data:
  - Angular velocity (3): gyroscope
  - Linear acceleration (3): accelerometer
  - Orientation quaternion (4): from filter

Total: 87 (joints) + image tensors + 10 (IMU)</code></div>
            <p>Observations mirror what's available on the real G1 — ensuring policies trained in sim can directly transfer.</p>
          `
        },
        {
          title: "DDS Teleoperation Interface",
          content: `
            <p>Same CycloneDDS topics as the real robot, enabling seamless switching between sim and hardware:</p>
            <div class="code-block"><code>DDS Topics (shared with real robot):
  rt/lowcmd   → Joint position commands (input)
  rt/lowstate → Joint state feedback (output)
  rt/inspire/cmd → Hand commands (optional)

Domain Isolation:
  Domain 0: Physical robot (production)
  Domain 1: Isaac Lab simulation (testing)

Teleoperation Flow:
  XR Headset → teleop_hand_and_arm.py
    → CycloneDDS (Domain 1) → Isaac Lab env
    → Render + physics step
    → State feedback → DDS → teleop display

Benefits:
  - Test teleop code without real robot
  - Validate IK solutions in physics sim
  - Record demonstrations for imitation learning</code></div>
          `
        },
        {
          title: "Episode Recording for Imitation Learning",
          content: `
            <p>HDF5-based episode recording captures multi-modal demonstrations for downstream learning:</p>
            <div class="code-block"><code>Recording Format (HDF5):
episode_0/
├── observations/
│   ├── joint_positions    [T × 29]
│   ├── joint_velocities   [T × 29]
│   ├── camera_rgb         [T × 480 × 640 × 3]
│   └── camera_depth       [T × 480 × 640]
├── actions/               [T × action_dim]
├── timestamps/            [T]
└── metadata/
    ├── robot_type: "g1_29dof"
    ├── task: "reaching" / "manipulation"
    └── success: bool

Controls:
  's' = start/stop recording
  'r' = reset episode
  'q' = quit and save</code></div>
            <p>Downstream use: behavior cloning, GAIL, or DAgger-style iterative refinement. HDF5 format supports efficient random access for training dataloaders.</p>
          `
        }
      ],
      metrics: [
        { value: "87-dim", label: "Joint Obs" },
        { value: "200Hz", label: "Physics Rate" },
        { value: "1", label: "Num Envs" },
        { value: "HDF5", label: "Recording Format" }
      ],
      references: [
        { title: "Isaac Lab Docs", url: "https://isaac-sim.github.io/IsaacLab/" },
        { title: "Isaac Sim", url: "https://developer.nvidia.com/isaac-sim" },
        { title: "CycloneDDS", url: "https://cyclonedds.io/" },
        { title: "Unitree SDK2", url: "https://github.com/unitreerobotics/unitree_sdk2" }
      ]
    }
  },
  {
    id: 'gesture-detection',
    title: 'Real-Time Hand Gesture Recognition System',
    shortTitle: 'Gesture Detection',
    category: 'Computer Vision',
    problem: 'Touchless robot control through natural hand gestures',
    outcome: '95% accuracy gesture classification at 30+ fps on CPU',
    oneLiner: 'MediaPipe 21-landmark tracking with MLP classifier for robot control',
    description: `
      Real-time hand gesture recognition for human-robot interaction using MediaPipe 21-landmark tracking and a custom MLP classifier. Achieves 95% accuracy at 30+ FPS on CPU with 8 gesture classes, published as ROS2 commands for robot control.
    `,
    tags: ['MediaPipe', 'Gesture Recognition', 'Deep Learning', 'ROS2', 'OpenCV'],
    images: [],
    videos: [],
    thumbnail: null,
    hideGallery: true,
    github: 'https://github.com/bhaveshbakshi633/gesture_detection',
    technicalDeepDive: {
      sections: [
        {
          title: "MediaPipe Hand Landmark Detection",
          content: `
            <p>Google's <strong>MediaPipe Hands</strong> provides real-time 21-point hand skeleton detection using a two-stage pipeline:</p>
            <div class="code-block"><code>Stage 1: Palm Detection (BlazePalm)
  - Single-shot detector on full frame
  - Outputs bounding box + 7 palm keypoints
  - Runs only when hand not tracked

Stage 2: Hand Landmark (BlazePose)
  - Regression network on cropped ROI
  - 21 3D landmarks (x, y, z)
  - Runs every frame on tracked hand</code></div>
            <p>The pipeline achieves 30+ FPS on CPU by using the palm detector sparingly and tracking landmarks between frames.</p>
          `
        },
        {
          title: "Feature Extraction for Classification",
          content: `
            <p>Raw landmarks are converted to rotation/scale-invariant features:</p>
            <div class="code-block"><code>Features (per hand):
1. Finger Angles (5):
   - Angle between MCP-PIP-TIP for each finger
   - Normalized to [0, 1] (0=extended, 1=curled)

2. Finger Distances (5):
   - Fingertip to palm center distance
   - Normalized by hand size

3. Palm Orientation (2):
   - Pitch and roll from wrist-MCP vectors
   - Invariant to camera position

4. Inter-finger Angles (4):
   - Spread between adjacent fingers

Total: 16 features per hand</code></div>
            <p>Normalization by palm size ensures invariance to distance from camera.</p>
          `
        },
        {
          title: "Gesture Classification Model",
          content: `
            <p>A lightweight MLP classifier maps features to gesture classes:</p>
            <div class="code-block"><code>Architecture:
  Input: 16 features
  Hidden: 64 → ReLU → 32 → ReLU
  Output: 8 classes (softmax)

Gestures:
  - POINT: Index extended, others curled
  - GRAB: All fingers curled (fist)
  - RELEASE: All fingers extended (open palm)
  - THUMBS_UP: Thumb up, fist
  - WAVE: Open palm, motion detected
  - STOP: Palm facing camera
  - OK: Thumb-index circle
  - NONE: No clear gesture</code></div>
            <p>Training data collected via custom annotation tool - 500 samples per gesture with augmentation (rotation, scale, noise).</p>
          `
        },
        {
          title: "Temporal Smoothing & Debouncing",
          content: `
            <p>To prevent flickering predictions, temporal filtering is applied:</p>
            <div class="formula">Confidence smoothing: EMA with α=0.3
Gesture trigger: Same class for 5 consecutive frames
Cooldown: 500ms after trigger (prevents repeated fires)</div>
            <p><strong>State Machine:</strong> IDLE → DETECTED → TRIGGERED → COOLDOWN → IDLE. Only transitions to TRIGGERED if confidence > 0.85 for 5 frames.</p>
          `
        },
        {
          title: "Robot Integration",
          content: `
            <p>Detected gestures publish to ROS2 topics for robot control:</p>
            <div class="code-block"><code>ros2 topic: /gesture/detected (std_msgs/String)
Payload: {"gesture": "POINT", "confidence": 0.92, "hand": "right"}

Gesture → Action Mapping (configurable):
  POINT → Move to pointed direction
  GRAB → Close gripper
  RELEASE → Open gripper
  STOP → Emergency halt
  THUMBS_UP → Confirm action</code></div>
            <p>Visual feedback overlays skeleton and gesture label on camera stream for debugging.</p>
          `
        }
      ],
      metrics: [
        { value: "30+ FPS", label: "Processing Rate" },
        { value: "95%", label: "Accuracy" },
        { value: "21", label: "Landmarks" },
        { value: "8", label: "Gesture Classes" }
      ],
      references: [
        { title: "MediaPipe Hands", url: "https://google.github.io/mediapipe/solutions/hands.html" },
        { title: "BlazePalm Paper", url: "https://arxiv.org/abs/2006.10214" },
        { title: "Hand Gesture Recognition Survey", url: "https://arxiv.org/abs/1901.00925" }
      ]
    }
  },
  {
    id: 'availsure',
    title: 'AvailSure: Home Services Booking Platform',
    shortTitle: 'AvailSure App',
    category: 'Software Development',
    problem: 'Connect users with verified service providers with real-time tracking',
    outcome: 'Full-stack app with sub-second booking and 95%+ location accuracy',
    oneLiner: 'React Native + Node.js with real-time Socket.io tracking',
    description: `
      Full-stack home services booking app built with React Native and Node.js. Features real-time provider tracking via Socket.io, MongoDB geospatial queries for nearby matches, and offline-first architecture with sync queue.
    `,
    tags: ['React Native', 'Node.js', 'MongoDB', 'Socket.io', 'Google Maps'],
    images: [],
    videos: [],
    thumbnail: null,
    hideGallery: true,
    github: 'https://github.com/bhaveshbakshi633/AvailSure',
    technicalDeepDive: {
      sections: [
        {
          title: "React Native Architecture",
          content: `
            <p>Cross-platform mobile app built with <strong>React Native</strong> sharing 90%+ code between iOS and Android:</p>
            <div class="code-block"><code>Project Structure:
src/
├── components/     # Reusable UI components
├── screens/        # Screen-level containers
├── navigation/     # React Navigation stack
├── services/       # API & business logic
├── store/          # Redux state management
└── utils/          # Helpers & constants

Key Dependencies:
  - react-navigation v6 (stack + tabs)
  - redux-toolkit + redux-persist
  - react-native-maps (Google Maps)
  - socket.io-client (real-time)</code></div>
          `
        },
        {
          title: "Node.js Backend API",
          content: `
            <p>RESTful API built with Express.js following MVC pattern:</p>
            <div class="code-block"><code>API Endpoints:
  POST /auth/register      # User registration
  POST /auth/login         # JWT authentication
  GET  /services           # Browse service catalog
  POST /bookings           # Create new booking
  GET  /bookings/:id       # Booking details + status
  PUT  /bookings/:id/status # Provider status update
  GET  /providers/nearby   # Geospatial query

Middleware Stack:
  1. helmet (security headers)
  2. cors (cross-origin)
  3. express-rate-limit (DDoS protection)
  4. jwt-verify (authentication)
  5. express-validator (input validation)</code></div>
          `
        },
        {
          title: "MongoDB Data Modeling",
          content: `
            <p>NoSQL schema designed for geospatial queries and real-time updates:</p>
            <div class="code-block"><code>User Schema:
  { _id, name, email, phone, location: {
      type: "Point", coordinates: [lng, lat]
    }, role: "user"|"provider" }

Booking Schema:
  { _id, userId, providerId, serviceType,
    status: "pending"|"accepted"|"in_progress"|"completed",
    scheduledTime, location, price,
    tracking: [{ lat, lng, timestamp }] }

Indexes:
  - location: "2dsphere" (geospatial)
  - { status: 1, scheduledTime: 1 } (queries)
  - { providerId: 1, status: 1 } (provider dashboard)</code></div>
            <p><strong>Geospatial Query:</strong> <code>$nearSphere</code> finds providers within radius, sorted by distance.</p>
          `
        },
        {
          title: "Real-Time Tracking with Socket.io",
          content: `
            <p>Bidirectional WebSocket connection enables live location updates:</p>
            <div class="code-block"><code>// Provider app emits location every 5s
socket.emit('location:update', {
  bookingId: '...',
  lat: 28.4595,
  lng: 77.0266
});

// User app receives updates
socket.on('provider:location', (data) => {
  updateMapMarker(data.lat, data.lng);
  updateETA(data.eta);
});

// Server broadcasts to booking room
io.to(\`booking:\${bookingId}\`).emit('provider:location', data);</code></div>
            <p>Rooms isolate updates to relevant users. Heartbeat every 30s detects disconnections.</p>
          `
        },
        {
          title: "Offline-First Architecture",
          content: `
            <p>App remains functional without network using local persistence:</p>
            <div class="formula">AsyncStorage: User preferences, auth tokens
Redux-Persist: App state snapshot
Queue: Pending API calls (synced on reconnect)</div>
            <p><strong>Sync Strategy:</strong> On network restore, queued actions replay in order. Conflict resolution uses server timestamp as source of truth. Background fetch updates booking status.</p>
          `
        }
      ],
      metrics: [
        { value: "< 1s", label: "Booking Time" },
        { value: "95%+", label: "Location Accuracy" },
        { value: "90%", label: "Code Sharing" },
        { value: "5s", label: "Update Interval" }
      ],
      references: [
        { title: "React Native Docs", url: "https://reactnative.dev/docs/getting-started" },
        { title: "Socket.io Docs", url: "https://socket.io/docs/v4/" },
        { title: "MongoDB Geospatial", url: "https://www.mongodb.com/docs/manual/geospatial-queries/" }
      ]
    }
  },
  {
    id: 'xr-teleop',
    title: 'XR Teleoperation for Humanoid Robots',
    shortTitle: 'XR Teleop',
    category: 'Robotics Integration',
    protected: true,
    problem: 'Real-time bilateral teleoperation of 29-DOF humanoid with dexterous manipulation',
    outcome: '30Hz control loop, <50ms LAN latency, dual-arm IK + 5 end-effector types',
    oneLiner: 'Quest 3 → Pinocchio IK → CycloneDDS → Unitree G1/H1 with dexterous hands',
    description: `
      Production XR teleoperation for Unitree G1/H1 humanoids using Quest 3 and other headsets. Real-time dual-arm IK via Pinocchio + CasADi + IPOPT, DexPilot hand retargeting for 5 end-effector types, and a safety state machine — all at 30Hz with <50ms LAN latency over CycloneDDS.
    `,
    tags: ['Pinocchio', 'CasADi', 'Quest 3', 'CycloneDDS', 'DexPilot', 'Isaac Lab'],
    images: [],
    videos: [
      { src: 'assets/projects/xr-teleop/demo.mp4', title: 'XR Teleoperation Demo' }
    ],
    thumbnail: null,
    hideGallery: false,
    technicalDeepDive: {
      sections: [
        {
          title: "System Architecture",
          content: `
            <p>Three-tier architecture: XR headset → Host PC → Robot, connected via WebSocket + DDS:</p>
            <div class="code-block"><code>Quest 3 (OpenXR)          Host PC (Python)              Robot (G1/H1)
  Hand/Head Tracking    teleop_hand_and_arm.py         DDS Domain 0/1
    ↓ Vuer.js             ├─ Arm IK (Pinocchio+CasADi)   ├─ rt/lowcmd (arms)
  WebSocket :8012         ├─ Hand Retarget (DexPilot)     ├─ rt/inspire/cmd (hands)
    ↓ HTTPS/WSS           ├─ Safe Teleop (state machine)  ├─ rt/lowstate (feedback)
  LocalTunnel/ngrok       └─ Camera (teleimager)          └─ Motor Controllers

Submodules:
  televuer   → XR input capture (WebSocket/WebRTC)
  teleimager → Camera streaming (ZMQ/WebRTC)
  dex-retargeting → DexPilot finger IK library</code></div>
            <p>119 Python source files. Entry point: <code>teleop_hand_and_arm.py</code>. Supports 4 robot variants (G1 29/23-DOF, H1, H1_2) and 5 end-effector types.</p>
          `
        },
        {
          title: "Dual-Arm Inverse Kinematics",
          content: `
            <p>Real-time nonlinear IK using Pinocchio for kinematics, CasADi for symbolic autodiff, and IPOPT as the NLP solver:</p>
            <div class="code-block"><code>Optimization Problem (per arm, 7-DOF):
  minimize: 50·‖p_target - FK(q)‖² + ‖R_target - R(q)‖²
            + 0.02·‖q - q_neutral‖² + 0.1·‖q - q_prev‖²
  subject to: q_min ≤ q ≤ q_max

IPOPT Config:
  max_iter: 30, tol: 1e-4, warm_start: enabled
  Solve time: 5-10ms (warm-started)

Model: g1_29_model_cache.pkl (Pinocchio binary)
  Locks 27 joints (legs, waist, fingers) → reduced 14-DOF arm problem
  Frame targets: L_ee, R_ee (wrist + 0.05m offset)</code></div>
            <p>Weighted moving filter (α = [0.4, 0.3, 0.2, 0.1]) smooths joint trajectories. Null-space biases toward elbow-down configurations.</p>
          `
        },
        {
          title: "DexPilot Hand Retargeting",
          content: `
            <p>Maps 25-DOF OpenXR hand skeleton to robot finger commands via optimization-based retargeting:</p>
            <div class="code-block"><code>Pipeline:
  OpenXR 25 joints/hand → Extract fingertip positions
    ↓ DexPilot Optimizer
  Constraints: Robot URDF model, joint limits, collision avoidance
    ↓ Output: joint commands per hand
  Low-pass filter (α=0.2) → Normalize to motor range
    ↓ Publish via DDS (rt/inspire/cmd)

End-Effector Support:
  Inspire DFX/FTP: 6 motors/hand (4 proximal + 2 thumb)
  Dex3:            7 motors/hand (3 per finger: prox/mid/dist)
  Dex1:            2 motors/hand (simple gripper, linear map)
  BrainCo:         6 motors/hand

Config (YAML):
  type: DexPilot | vector
  scaling_factor: 1.20
  low_pass_alpha: 0.2</code></div>
            <p>Resolves 25 human DOF → 6-7 robot DOF via constrained optimization with collision avoidance.</p>
          `
        },
        {
          title: "Control Loop & Latency",
          content: `
            <p>30Hz main loop (~33ms per iteration) with warm-started IK:</p>
            <div class="code-block"><code>Loop Cycle (33ms budget):
  1. Receive XR pose from Vuer (WebSocket)        ~2ms
  2. Transform OpenXR → Robot frame                ~1ms
  3. Scale arms (human 0.60m → robot 0.75m)        ~1ms
  4. Solve dual-arm IK via IPOPT (warm-start)    5-10ms
  5. Check collision + joint limits                ~1ms
  6. Publish LowCmd to DDS (rt/lowcmd)             ~1ms
  7. Retarget hands via DexPilot                  1-2ms
  8. Publish hand commands (rt/inspire/cmd)         ~1ms
  9. Encode + send camera frame                   5-8ms

Total Latency:
  LAN:      30-50ms  (direct WiFi 6)
  Internet: 100-250ms (via LocalTunnel/ngrok)</code></div>
            <p>Camera: RealSense D435I @ 480×640, JPEG quality 60%, streamed via ZMQ (tcp://172.16.2.242:55555) or WebRTC (port 60001).</p>
          `
        },
        {
          title: "Safety & Deployment",
          content: `
            <p>Safety state machine prevents unsafe transitions. DDS domain isolation separates real robot from simulation:</p>
            <div class="code-block"><code>State Machine (safe_teleop.py):
  IDLE → ALIGNING (5s timeout) → TRACKING ↔ FAULT
                                    ↓          ↓
                                 EXITING ← RECOVERING

Safety Parameters:
  Alignment threshold: 0.05 rad (~3°)
  Fault trigger: 0.5 rad error sustained 30 frames
  Joint limit scale: 80% of full range
  Alignment velocity: 0.3 rad/s (slow ramp)

DDS Domain Isolation:
  Domain 0: Physical robot (G1, H1)
  Domain 1: Isaac Lab simulation
  Prevents accidental cross-contamination

Keyboard Controls:
  r = start tracking, q = exit, c = clear fault, s = record episode</code></div>
            <p>Episode recording via HDF5 (h5py) for imitation learning data collection. Deployed on Ubuntu 22.04 with NVIDIA RTX A2000.</p>
          `
        }
      ],
      metrics: [
        { value: "30 Hz", label: "Control Loop" },
        { value: "< 50ms", label: "LAN Latency" },
        { value: "14-DOF", label: "Dual-Arm IK" },
        { value: "5", label: "End-Effector Types" }
      ],
      references: [
        { title: "Pinocchio Docs", url: "https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/" },
        { title: "CasADi", url: "https://web.casadi.org/" },
        { title: "DexPilot Paper", url: "https://arxiv.org/abs/1910.03135" },
        { title: "Unitree XR Teleop", url: "https://github.com/unitreerobotics/xr_teleoperate" }
      ]
    }
  },
  {
    id: 'atv',
    title: 'All-Terrain Vehicle (ATV) Chassis Design',
    shortTitle: 'ATV Chassis',
    category: 'Mechanical Design',
    problem: 'Design lightweight yet rigid chassis meeting BAJA SAE safety standards',
    outcome: '38kg frame with 2.1+ FOS, 25% cost reduction through DFM optimization',
    oneLiner: 'FEA-optimized 4130 chromoly space frame for BAJA SAE',
    description: `
      BAJA SAE ATV chassis design using AISI 4130 chromoly space frame. FEA-optimized in Ansys for 2.5G bump, braking, and cornering loads with 2.1+ FOS. 38kg bare frame weight achieved through DFM optimization with 25% cost reduction.
    `,
    tags: ['SolidWorks', 'FEA', 'Ansys', 'BAJA SAE', 'Manufacturing'],
    images: ['assets/projects/atv/img1.jpg', 'assets/projects/atv/img2.jpg', 'assets/projects/atv/img3.jpg', 'assets/projects/atv/img4.png'],
    video: 'assets/projects/atv/video.mp4',
    thumbnail: 'assets/projects/atv/img1.jpg',
    technicalDeepDive: {
      sections: [
        {
          title: "Space Frame Design & Material Selection",
          content: `
            <p>Tubular space frame chassis designed to BAJA SAE rule specifications with AISI 4130 chromoly steel:</p>
            <div class="code-block"><code>Chassis Specifications:
  Material: AISI 4130 chromoly steel
  Primary tubes: 1" OD × 0.065" wall
  Secondary: 1" OD × 0.049" wall
  Bracing: 3/4" OD × 0.049" wall
  Weight: 38 kg bare frame

Design Requirements (BAJA SAE):
  - Roll cage: overhead protection with 6" clearance
  - Side impact: lateral members at hip height
  - Front impact: energy-absorbing crumple zone
  - Rear firewall: engine compartment isolation
  - Driver ergonomics: 5th-95th percentile fit

Triangulation:
  All bays fully triangulated for torsional rigidity
  Target: >3000 Nm/deg torsional stiffness
  Node-to-node member connections (no floating joints)</code></div>
          `
        },
        {
          title: "FEA Analysis (Ansys Workbench)",
          content: `
            <p>Comprehensive structural analysis covering static, dynamic, and fatigue loading scenarios:</p>
            <div class="code-block"><code>Load Cases:
  1. Bump (2.5G vertical):
     Max stress: 280 MPa (yield: 460 MPa)
     FOS: 1.64 → reinforced to 2.1

  2. Braking (1.5G longitudinal):
     Critical: front suspension mounts
     FOS: 2.3 (after reinforcement)

  3. Cornering (1.2G lateral):
     Critical: roll cage / side impact members
     FOS: 2.5

  4. Rollover (combined 2G):
     Dynamic impact simulation
     Energy absorption in crumple zones
     Roll cage deformation < 25mm

  5. Fatigue (10⁶ cycles):
     Stress concentration at weld toes
     Reinforcement gussets added where needed

Mesh: BEAM188 elements, 2mm element size
Solver: Ansys Mechanical APDL</code></div>
          `
        },
        {
          title: "Manufacturing & Assembly",
          content: `
            <p>Production process optimized for dimensional accuracy and cost reduction:</p>
            <div class="code-block"><code>Fabrication:
  Cutting: Tube notcher + band saw
  Bending: Manual tube bender (schedule)
  Welding: TIG (GTAW) with ER80S-D2 filler
  Joint type: Full-penetration butt welds

Assembly Jig:
  Steel fixture table (leveled ±0.5mm)
  Locating pins at all node points
  Dimensional tolerance: ±1mm
  Assembly sequence: RHD → LHD → cross

DFM Optimization (25% cost reduction):
  - Standardized tube sizes (3 diameters only)
  - Minimized bend count (straight members preferred)
  - Common notch angles (30°, 45°, 60°, 90°)
  - Batch cutting schedule for tube efficiency

QC: CMM measurement of critical dimensions
    Weld inspection: visual + dye penetrant</code></div>
          `
        }
      ],
      metrics: [
        { value: "38 kg", label: "Frame Weight" },
        { value: "2.1+", label: "Min FOS" },
        { value: "25%", label: "Cost Reduction" },
        { value: "±1mm", label: "Tolerance" }
      ]
    }
  },
  {
    id: 'surgical-robot',
    title: 'ASAP Sim2Real Deployment & Skill Execution on G1',
    shortTitle: 'Sim2Real Deploy',
    category: 'Robotics Deployment',
    protected: true,
    problem: 'Deploying RL-trained whole-body skills from simulation to real Unitree G1 hardware with stable execution',
    outcome: 'Real-time 50Hz skill execution on G1: CR7 celebration, kicks, jumps, and locomotion via ONNX + delta action model',
    oneLiner: 'ONNX policy + delta action model → 50Hz newton controller → Unitree SDK2 → G1 whole-body skills',
    description: `
      Sim-to-real deployment framework for ASAP-trained policies on the Unitree G1. Newton controller runs ONNX policies at 50Hz with inline delta action model correction via Unitree SDK2. Deploys CR7 celebrations, kicks, jumps, and locomotion after MuJoCo sim2sim validation.
    `,
    tags: ['Sim2Real', 'ONNX', 'Unitree SDK2', 'CycloneDDS', 'MuJoCo', 'Python'],
    images: [],
    videos: [],
    thumbnail: null,
    hideGallery: true,
    technicalDeepDive: {
      sections: [
        {
          title: "Newton Controller Architecture",
          content: `
            <p>The deployment controller (<code>newton_controller.py</code>) runs the trained policy on real hardware at 50Hz:</p>
            <div class="code-block"><code>Deployment Loop (20ms / 50Hz):
  1. Read joint encoders + IMU from G1
  2. Construct observation vector:
     - joint_positions (23): current DOF angles
     - joint_velocities (23): angular rates
     - base_ang_vel (3): from IMU gyroscope
     - projected_gravity (3): orientation ref
     - target_motion: from skill reference
     - 4-frame history (~453-dim total)
  3. ONNX Runtime inference → action (23-dim)
  4. Delta action model correction
  5. target = action_scale × action + default_angles
  6. PD torque: τ = Kp(target - q) - Kd(dq)
  7. Send torques via Unitree SDK2 / CycloneDDS

Timing:
  Policy inference: ~5ms (ONNX Runtime, CPU)
  Delta model: ~1ms
  Total loop: <15ms (within 20ms budget)</code></div>
          `
        },
        {
          title: "Delta Action Model Integration",
          content: `
            <p>The delta action model corrects policy outputs to compensate for motor dynamics not captured in simulation:</p>
            <div class="code-block"><code>Delta Model:
  Input: raw policy action (23-dim)
  Network: Linear(23, 256) → ELU
           → Linear(256, 256) → ELU
           → Linear(256, 23)
  Output: corrected motor commands

Compensates for:
  - Motor friction & backlash
  - Gearbox nonlinearities
  - Cable routing stiffness
  - Thermal drift

corrected_action = raw_action + delta_model(raw_action)
The model adds learned corrections at each timestep.</code></div>
            <p>Trained offline on open-loop motor trajectory data collected from the real G1. Supervised loss minimizes position prediction error.</p>
          `
        },
        {
          title: "Sim2Sim Validation Pipeline",
          content: `
            <p>Before deploying to real hardware, every policy must pass sim2sim validation in MuJoCo:</p>
            <div class="code-block"><code>Validation (deploy_mujoco.py):
  1. Load exported policy (ONNX or .pt)
  2. Initialize MuJoCo with G1 model
  3. Run policy in MuJoCo viewer
  4. Visual checks:
     - Gait stability on flat ground
     - Motion quality matches training
     - No PhysX-specific artifacts
     - Recovery from small perturbations

MuJoCo Viewer Controls:
  Arrow keys: velocity commands
  Space: trigger skill
  R: reset to default pose
  Q: quit

Catches:
  - Policies exploiting PhysX contact bugs
  - Unstable standing after skill execution
  - Excessive joint velocities or torques</code></div>
          `
        },
        {
          title: "Skill Execution & Triggering",
          content: `
            <p>Multiple whole-body skills deployed as separate ONNX policies with keyboard/joystick triggering:</p>
            <div class="code-block"><code>Deployed Skills:
  CR7 Siuuu:     Full-body celebration (jump + arms up)
  Forward Jump:  Bipedal forward leap (~30cm)
  Side Jump:     Lateral hop (left/right)
  Soccer Kick:   Single-leg kick (L/R selectable)
  Walk:          Velocity-commanded locomotion

Execution Architecture:
  Locomotion policy: always-on (lower body)
  Skill policy: triggered overlay (full body)

  When skill triggered:
    1. Blend from locomotion → skill (5 frames)
    2. Execute skill motion (variable duration)
    3. Blend back to locomotion (10 frames)
    4. Recovery: stand stabilization phase

Input: Keyboard mapping or joystick buttons
  W/A/S/D: velocity commands
  1-5: trigger skills
  ESC: emergency stop</code></div>
          `
        }
      ],
      metrics: [
        { value: "50Hz", label: "Control Rate" },
        { value: "23-DOF", label: "Action Space" },
        { value: "~5ms", label: "Inference Time" },
        { value: "5+", label: "Deployed Skills" }
      ],
      references: [
        { title: "ASAP Framework", url: "https://agility.csail.mit.edu/asap/" },
        { title: "Unitree SDK2", url: "https://github.com/unitreerobotics/unitree_sdk2" },
        { title: "ONNX Runtime", url: "https://onnxruntime.ai/" },
        { title: "MuJoCo Viewer", url: "https://mujoco.readthedocs.io/en/stable/python.html#viewer" }
      ]
    }
  },
  {
    id: 'chess-board',
    title: 'Automated Chess Board System',
    shortTitle: 'Auto Chess Board',
    category: 'Embedded Systems',
    problem: 'Physical chess board that moves pieces autonomously for remote/AI gameplay',
    outcome: 'Working XY gantry system with piece detection and network play',
    oneLiner: 'XY gantry with electromagnets and custom chess engine',
    description: `
      Created an interactive chessboard featuring electromagnetic piece movement, computer vision
      for move detection, and network connectivity for remote gameplay.

      The system combines mechanical precision with intelligent software algorithms. Uses XY gantry
      system with electromagnets to physically move chess pieces, Reed switches for piece detection,
      and a custom chess engine for AI gameplay.
    `,
    tags: ['Arduino', 'Stepper Motors', 'Reed Switches', 'Chess Engine'],
    images: [],
    videos: [],
    thumbnail: null,
    hideGallery: true,
    technicalDeepDive: {
      sections: [
        {
          title: "XY Gantry & Electromagnetic Actuation",
          content: `
            <p>Pieces are moved physically using an XY gantry system beneath the board with electromagnetic piece capture:</p>
            <div class="code-block"><code>Mechanical System:
  Motion: CoreXY belt-driven gantry
  Motors: NEMA 17 stepper × 2 (X, Y axes)
  Driver: A4988 stepper driver (1/16 microstepping)
  Resolution: 0.1mm per microstep
  Speed: ~50mm/s max travel

Piece Capture:
  Electromagnet: 12V DC, 5kg holding force
  Mounted on gantry carriage (below board)
  Pieces: steel base inserts for magnetic grip
  Sequence: engage → lift → translate → lower → release

Board Surface:
  Material: 3mm MDF with vinyl overlay
  Square size: 50mm × 50mm
  Total board: 400mm × 400mm</code></div>
          `
        },
        {
          title: "Piece Detection with Reed Switches",
          content: `
            <p>64 reed switches (one per square) detect piece presence for move validation:</p>
            <div class="code-block"><code>Detection Grid:
  Sensors: Reed switch × 64 (normally open)
  Scanning: 8×8 matrix with row/column multiplexing
  Controller: Arduino Mega (enough digital pins)
  Scan rate: Full board read in ~5ms

Move Detection:
  1. Scan board → 8×8 binary occupancy matrix
  2. Compare to previous state
  3. Detect: piece lifted (1→0) → piece placed (0→1)
  4. Validate against legal moves
  5. If illegal: LED indicator + buzzer alert

Multiplexing:
  8 row lines (output) + 8 column lines (input)
  Sequential row activation with column read
  Pull-down resistors on column inputs</code></div>
          `
        },
        {
          title: "Chess Engine & Network Play",
          content: `
            <p>Integrated chess engine for AI gameplay with optional network connectivity for remote matches:</p>
            <div class="code-block"><code>Chess Engine:
  Algorithm: Minimax with alpha-beta pruning
  Depth: 4-6 ply (adjustable difficulty)
  Evaluation: Material + position tables
  Opening book: 500+ common openings

Game Modes:
  Human vs Human (local): Reed switch detection
  Human vs AI: Engine calculates + gantry moves
  Human vs Human (remote): WiFi/Bluetooth

Communication:
  Protocol: Serial (Arduino ↔ PC) or WiFi (ESP32)
  Move format: UCI notation (e.g., e2e4)
  State sync: FEN string broadcast

Display:
  LCD screen: move history, clock, player info
  LED ring per square: highlight legal moves
  Buzzer: check/checkmate/illegal move alerts</code></div>
          `
        }
      ],
      metrics: [
        { value: "64", label: "Reed Switches" },
        { value: "0.1mm", label: "Gantry Resolution" },
        { value: "6-ply", label: "AI Depth" },
        { value: "5ms", label: "Board Scan Time" }
      ]
    }
  },
  {
    id: 'ir-remote',
    title: 'Smart IR Control Network',
    shortTitle: 'Smart IR Hub',
    category: 'Embedded Systems',
    problem: 'Unified control for all IR devices without multiple remotes',
    outcome: 'Learning remote that clones any IR signal with smartphone app control',
    oneLiner: 'ESP32-based IR learning system with protocol detection',
    description: `
      Designed a microcontroller-based infrared remote system capable of device identification,
      learning, and selective control.

      Features intelligent protocol detection that can learn and replicate IR signals from any
      device. Includes a centralized home automation management system with smartphone app control.
    `,
    tags: ['ESP32', 'IR Protocols', 'MQTT', 'Arduino', 'Smartphone App'],
    images: [],
    videos: [],
    thumbnail: null,
    hideGallery: true,
    technicalDeepDive: {
      sections: [
        {
          title: "IR Protocol Detection & Learning",
          content: `
            <p>Automatic protocol identification and signal learning from any IR remote control:</p>
            <div class="code-block"><code>Supported Protocols:
  NEC: 32-bit (most common — TVs, set-top boxes)
  RC5/RC6: Philips protocol (toggle bit handling)
  Sony SIRC: 12/15/20-bit variants
  Samsung: 32-bit proprietary
  Raw: Timing capture for unknown protocols

Learning Process:
  1. IR receiver (TSOP38238, 38kHz) captures signal
  2. Timing analysis: mark/space durations
  3. Protocol auto-detection from timing patterns
  4. Decode to command code (address + data)
  5. Store in EEPROM/Flash (persistent)

Raw Capture (fallback for unknown protocols):
  Records exact timing sequence (µs precision)
  Replay via IR LED with same timing
  Handles protocols not in library</code></div>
          `
        },
        {
          title: "ESP32 Hardware Architecture",
          content: `
            <p>Microcontroller-based hub with WiFi connectivity for smartphone control:</p>
            <div class="code-block"><code>Hardware:
  MCU: ESP32 (dual-core, WiFi + BLE)
  IR Receiver: TSOP38238 (38kHz, AGC)
  IR Transmitter: 940nm LED × 3 (wide coverage)
  Transistor: NPN driver for LED current
  Range: ~8m (multi-LED array, 120° spread)

Memory:
  Flash: Learned commands (up to 200 codes)
  SPIFFS: Device profiles and configuration
  NVS: WiFi credentials and pairing data

Power:
  USB 5V input or 3.7V LiPo battery
  Deep sleep: <10µA (wake on BLE/button)
  Active: ~120mA (WiFi connected)</code></div>
          `
        },
        {
          title: "Smartphone App & Home Automation",
          content: `
            <p>Centralized control interface replacing all physical remotes:</p>
            <div class="code-block"><code>App Features:
  Device management: Add/remove/rename devices
  Room-based grouping: Living room, bedroom, etc.
  Custom buttons: Map any learned code to UI button
  Scenes: "Movie mode" = dim lights + TV on + soundbar
  Scheduling: Timer-based automation (e.g., AC off at 2AM)

Communication:
  Local: ESP32 WiFi AP or mDNS discovery
  Protocol: HTTP REST API or MQTT
  Latency: <100ms button press to IR emission

MQTT Integration (optional):
  Broker: Mosquitto (local Raspberry Pi)
  Topics: home/{room}/{device}/command
  Enables: Google Home / Alexa voice control
  Payload: JSON { "action": "power", "device": "tv" }</code></div>
          `
        }
      ],
      metrics: [
        { value: "5+", label: "IR Protocols" },
        { value: "200+", label: "Stored Codes" },
        { value: "8m", label: "IR Range" },
        { value: "<100ms", label: "Response Time" }
      ]
    }
  }
];

// DOM Elements
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const sidebarClose = document.getElementById('sidebarClose');
const hamburger = document.getElementById('hamburger');
const projectsCatalog = document.getElementById('projectsCatalog');
const contentWrapper = document.getElementById('contentWrapper');
const projectDetailView = document.getElementById('projectDetailView');
const projectDetail = document.getElementById('projectDetail');
const backBtn = document.getElementById('backBtn');
const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightboxImage');
const lightboxClose = document.getElementById('lightboxClose');
const lightboxPrev = document.getElementById('lightboxPrev');
const lightboxNext = document.getElementById('lightboxNext');
const lightboxCounter = document.getElementById('lightboxCounter');
const cursor = document.getElementById('cursor');
const cursorFollower = document.getElementById('cursorFollower');

// State
let currentProject = null;
let currentImageIndex = 0;
let currentImages = [];
let savedScrollPosition = 0;
let mouseX = 0, mouseY = 0;
let cursorX = 0, cursorY = 0;

// =============================================
// PARTICLE BACKGROUND
// =============================================
function initParticles() {
  if (FEATURE_FLAGS.DISABLE_PARTICLES) return;

  const canvas = document.getElementById('particleCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let particles = [];

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  class Particle {
    constructor() {
      this.reset();
    }

    reset() {
      this.x = Math.random() * canvas.width;
      this.y = Math.random() * canvas.height;
      this.size = Math.random() * 2 + 0.5;
      this.speedX = (Math.random() - 0.5) * 0.5;
      this.speedY = (Math.random() - 0.5) * 0.5;
      this.opacity = Math.random() * 0.5 + 0.2;
    }

    update() {
      this.x += this.speedX;
      this.y += this.speedY;

      if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
      if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
    }

    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 240, 255, ${this.opacity})`;
      ctx.fill();
    }
  }

  function init() {
    resize();
    particles = [];
    const particleCount = Math.min(100, Math.floor((canvas.width * canvas.height) / 15000));
    for (let i = 0; i < particleCount; i++) {
      particles.push(new Particle());
    }
  }

  function drawLines() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 150) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0, 240, 255, ${0.1 * (1 - distance / 150)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  }

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    particles.forEach(particle => {
      particle.update();
      particle.draw();
    });

    drawLines();
    requestAnimationFrame(animate);
  }

  window.addEventListener('resize', init);
  init();
  animate();
}

// =============================================
// CUSTOM CURSOR
// =============================================
function initCursor() {
  if (FEATURE_FLAGS.DISABLE_CUSTOM_CURSOR) return;
  if (!cursor || !cursorFollower || window.innerWidth < 1024) return;

  // Cursor position - dono ko same position pe rakhna hai
  let targetX = 0, targetY = 0;
  let followerX = 0, followerY = 0;

  document.addEventListener('mousemove', (e) => {
    targetX = e.clientX;
    targetY = e.clientY;
  });

  // Click effect - hide circle for 100ms
  document.addEventListener('mousedown', () => {
    cursorFollower.style.opacity = '0';
    setTimeout(() => {
      cursorFollower.style.opacity = '1';
    }, 100);
  });

  // Hide cursor when leaving window
  document.addEventListener('mouseleave', () => {
    cursor.style.opacity = '0';
    cursorFollower.style.opacity = '0';
  });

  document.addEventListener('mouseenter', () => {
    cursor.style.opacity = '1';
    cursorFollower.style.opacity = '1';
  });

  function animateCursor() {
    // Follower follows near-instantly (0.85 = minimal lag)
    followerX += (targetX - followerX) * 0.85;
    followerY += (targetY - followerY) * 0.85;

    // Dot follows mouse directly (instant)
    cursor.style.left = targetX + 'px';
    cursor.style.top = targetY + 'px';

    // Circle follows with smooth delay
    cursorFollower.style.left = followerX + 'px';
    cursorFollower.style.top = followerY + 'px';

    requestAnimationFrame(animateCursor);
  }

  animateCursor();

  // Hover effect on interactive elements
  const interactiveElements = document.querySelectorAll('a, button, .skill-card, .contact-card, .stat-card, .gallery-item');
  interactiveElements.forEach(el => {
    el.addEventListener('mouseenter', () => cursorFollower.classList.add('hover'));
    el.addEventListener('mouseleave', () => cursorFollower.classList.remove('hover'));
  });
}

// =============================================
// TYPING EFFECT
// =============================================
function initTyping() {
  const typingElement = document.getElementById('typingText');
  if (!typingElement) return;

  const phrases = [
    'Robotics Systems Engineer',
    'RL & Sim2Real Specialist',
    'Control Systems Integrator',
    'I Ship Robots That Work'
  ];

  let phraseIndex = 0;
  let charIndex = 0;
  let isDeleting = false;
  let typingSpeed = 100;

  function type() {
    const currentPhrase = phrases[phraseIndex];

    if (isDeleting) {
      typingElement.textContent = currentPhrase.substring(0, charIndex - 1);
      charIndex--;
      typingSpeed = 50;
    } else {
      typingElement.textContent = currentPhrase.substring(0, charIndex + 1);
      charIndex++;
      typingSpeed = 100;
    }

    if (!isDeleting && charIndex === currentPhrase.length) {
      typingSpeed = 2000; // Pause at end
      isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
      isDeleting = false;
      phraseIndex = (phraseIndex + 1) % phrases.length;
      typingSpeed = 500; // Pause before new phrase
    }

    setTimeout(type, typingSpeed);
  }

  setTimeout(type, 1000);
}

// =============================================
// NUMBER COUNTER ANIMATION
// =============================================
function initCounters() {
  const counters = document.querySelectorAll('.stat-number[data-count]');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const target = entry.target;
        const count = parseInt(target.dataset.count);
        animateCounter(target, count);
        observer.unobserve(target);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(counter => observer.observe(counter));
}

function animateCounter(element, target) {
  let current = 0;
  const duration = 2000;
  const increment = target / (duration / 16);

  function update() {
    current += increment;
    if (current < target) {
      element.textContent = Math.floor(current);
      requestAnimationFrame(update);
    } else {
      element.textContent = target;
    }
  }

  update();
}

// =============================================
// CHARACTER REVEAL ANIMATION
// =============================================
function initCharReveal() {
  const chars = document.querySelectorAll('.char');
  chars.forEach((char, index) => {
    char.style.setProperty('--char-index', index);
  });
}

// =============================================
// SIDEBAR FUNCTIONS
// =============================================
function toggleSidebar() {
  sidebar.classList.toggle('open');
  sidebarOverlay.classList.toggle('active');
  hamburger.classList.toggle('active');
  document.body.style.overflow = sidebar.classList.contains('open') ? 'hidden' : '';
}

function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('active');
  hamburger.classList.remove('active');
  document.body.style.overflow = '';
}

// =============================================
// PROJECT CATALOG RENDER
// =============================================
function renderProjectsCatalog() {
  const catalog = document.getElementById('projectsCatalog');
  if (!catalog) return;

  // Group projects by category
  const categories = {};
  projects.forEach(project => {
    const cat = project.category || 'Other';
    if (!categories[cat]) categories[cat] = [];
    categories[cat].push(project);
  });

  // Category order
  const categoryOrder = [
    'Robotics Integration',
    'Reinforcement Learning',
    'Computer Vision',
    'Simulation & Teleoperation',
    'Robotics Deployment',
    'Aerial Systems',
    'Mechanical Design',
    'Embedded Systems',
    'Software Development'
  ];

  let html = '';
  categoryOrder.forEach(name => {
    if (!categories[name]) return;
    html += `
      <div class="project-category collapsed">
        <button class="category-header" aria-expanded="false">
          <h3 class="category-title">${name}</h3>
          <div class="category-header-right">
            <span class="category-count">${categories[name].length} ${categories[name].length === 1 ? 'project' : 'projects'}</span>
            <svg class="category-chevron" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="6,9 12,15 18,9"></polyline>
            </svg>
          </div>
        </button>
        <div class="category-projects">
          ${categories[name].map(project => `
            <a href="#" class="project-card ${project.flagship ? 'flagship' : ''} ${isProjectProtected(project.id) ? 'protected' : ''}" data-project-id="${project.id}">
              ${project.flagship ? '<span class="flagship-badge">Featured</span>' : ''}
              ${isProjectProtected(project.id) ? '<span class="protected-badge">Confidential</span>' : ''}
              <div class="project-card-content">
                <h4 class="project-card-title">${project.shortTitle}</h4>
                <p class="project-card-oneliner">${project.oneLiner || ''}</p>
                <div class="project-card-tags">
                  ${project.tags.slice(0, 3).map(tag => `<span class="card-tag">${tag}</span>`).join('')}
                </div>
              </div>
              <div class="project-card-arrow">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                  <polyline points="12,5 19,12 12,19"></polyline>
                </svg>
              </div>
            </a>
          `).join('')}
        </div>
      </div>
    `;
  });

  catalog.innerHTML = html;

  // Add click handlers for project cards
  document.querySelectorAll('.project-card').forEach(card => {
    card.addEventListener('click', (e) => {
      e.preventDefault();
      showProjectDetail(card.dataset.projectId);
    });
  });

  // Add click handlers for collapsible category headers
  document.querySelectorAll('.category-header').forEach(header => {
    header.addEventListener('click', () => {
      const category = header.closest('.project-category');
      const isCollapsed = category.classList.contains('collapsed');
      category.classList.toggle('collapsed');
      header.setAttribute('aria-expanded', isCollapsed ? 'true' : 'false');
    });
  });
}

// =============================================
// ONGOING PROJECTS RENDER
// =============================================
function renderOngoingProjects() {
  const container = document.getElementById('ongoingProjects');
  if (!container || ongoingProjects.length === 0) return;

  let html = '';
  ongoingProjects.forEach(project => {
    html += `
      <div class="ongoing-card" data-project-id="${project.id}">
        <div class="ongoing-card-thumbnail">
          <img src="${project.thumbnail}" alt="${project.title}" loading="lazy" />
        </div>
        <div class="ongoing-card-content">
          <span class="ongoing-card-badge">${project.status}</span>
          <h4 class="ongoing-card-title">${project.title}</h4>
          <p class="ongoing-card-desc">${project.description}</p>
          <p class="ongoing-card-status">${project.progress} — ${project.currentWork}</p>
          <div class="ongoing-card-tags">
            ${project.tags.map(tag => `<span class="ongoing-card-tag">${tag}</span>`).join('')}
          </div>
        </div>
        <div class="ongoing-card-arrow">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="5" y1="12" x2="19" y2="12"></line>
            <polyline points="12,5 19,12 12,19"></polyline>
          </svg>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;

  // Add click handlers for ongoing project cards
  document.querySelectorAll('.ongoing-card').forEach(card => {
    card.addEventListener('click', () => {
      showOngoingProjectDetail(card.dataset.projectId);
    });
  });
}

// Show ongoing project detail
function showOngoingProjectDetail(projectId) {
  const project = ongoingProjects.find(p => p.id === projectId);
  if (!project) return;

  // Password protection check
  if (isProjectProtected(projectId)) {
    showPasswordModal(() => showOngoingProjectDetail(projectId));
    return;
  }

  savedScrollPosition = window.scrollY;
  currentProject = project;
  currentImages = project.images || [];
  currentImageIndex = 0;

  const detail = document.getElementById('projectDetail');
  detail.innerHTML = `
    <header class="project-detail-header">
      <div class="project-title-row">
        <h1 class="project-detail-title">${project.title}</h1>
        <div class="ongoing-badge-large">
          <span class="ongoing-dot"></span>
          ${project.status}
        </div>
      </div>
      <div class="project-detail-tags">
        ${project.tags.map(tag => `<span class="project-tag">${tag}</span>`).join('')}
      </div>
    </header>

    <div class="project-detail-description">
      ${project.fullDescription || project.description}
    </div>

    ${currentImages.length > 0 ? `
      <div class="project-gallery">
        <h3 class="gallery-title">Development Progress</h3>
        <div class="gallery-grid">
          ${currentImages.map((img, i) => `
            <div class="gallery-item" data-index="${i}">
              <img src="${img}" alt="${project.title} screenshot ${i + 1}" loading="lazy" />
            </div>
          `).join('')}
        </div>
      </div>
    ` : ''}

    ${project.technicalDeepDive ? `
      <div class="tech-deepdive-section expanded">
        <div class="tech-deepdive-header">
          <div class="tech-deepdive-badge">
            <span>⚡</span>
            <span class="deepdive-pulse"></span>
          </div>
          <div class="tech-deepdive-title">
            <h3>Technical Details</h3>
            <p>Architecture and implementation notes</p>
          </div>
        </div>
        <div class="tech-deepdive-content">
          ${project.technicalDeepDive.sections.map((section, i) => `
            <div class="tech-section">
              <div class="tech-section-header">
                <span class="tech-section-number">${i + 1}</span>
                <span class="tech-section-title">${section.title}</span>
              </div>
              <div class="tech-section-content">
                ${section.content}
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    ` : ''}
  `;

  // Show detail view
  contentWrapper.classList.add('hidden');
  projectDetailView.classList.add('active');
  window.scrollTo(0, 0);

  // Add gallery click handlers
  document.querySelectorAll('.gallery-item').forEach(item => {
    item.addEventListener('click', () => openLightbox(parseInt(item.dataset.index)));
  });
}


// =============================================
// SHOWCASE SECTION (Flagship Projects)
// =============================================
function renderShowcase() {
  if (!FEATURE_FLAGS.ENABLE_SHOWCASE) return;

  const carousel = document.getElementById('showcaseCarousel');
  if (!carousel) return;

  // Filter flagship projects by ID list (check both projects and ongoingProjects)
  const allProjects = [...projects, ...ongoingProjects];
  const flagshipProjects = FLAGSHIP_PROJECT_IDS
    .map(id => allProjects.find(p => p.id === id))
    .filter(p => p);

  if (flagshipProjects.length === 0) return;

  // Generate card HTML for a project
  const generateCard = (project) => {
    const hasVideo = project.videos && project.videos.length > 0;
    const hasImages = project.images && project.images.length > 0;
    const thumbnailSrc = project.thumbnail || (hasImages ? project.images[0] : '');
    const isOngoing = ongoingProjects.some(p => p.id === project.id);

    // Slideshow interval (2.5s for images)
    const slideInterval = project.slideInterval || 2500;

    // Video projects: show video. Image-only: slideshow with arrows
    const mediaItems = [];
    if (hasVideo) {
      mediaItems.push({ type: 'video', src: project.videos[0].src });
    } else if (hasImages) {
      project.images.slice(0, 5).forEach(img => mediaItems.push({ type: 'image', src: img }));
    }

    // Slideshow only for image-only projects with multiple images
    const showSlideshow = !hasVideo && mediaItems.length > 1;

    return `
      <article class="showcase-card ${isOngoing ? 'ongoing' : ''} ${isProjectProtected(project.id) ? 'protected' : ''}" data-project-id="${project.id}" tabindex="0" role="button" aria-label="View ${project.shortTitle || project.title} project">
        ${isProjectProtected(project.id) ? `
          <div class="showcase-confidential-overlay">
            <div class="confidential-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
              </svg>
            </div>
            <span class="confidential-label">Confidential</span>
            <span class="confidential-sub">Requires permission to view</span>
          </div>
        ` : ''}
        <div class="showcase-card-media" data-media='${JSON.stringify(mediaItems)}' data-interval="${slideInterval}" data-slideshow="${showSlideshow}">
          ${hasVideo ? `
            <video
              class="showcase-video showcase-media-item active"
              src="${project.videos[0].src}"
              poster="${thumbnailSrc}"
              muted
              loop
              playsinline
              preload="metadata"
              autoplay
            ></video>
          ` : hasImages ? `
            <img class="showcase-media-item active" src="${project.images[0]}" alt="${project.title}" loading="lazy" />
          ` : `
            <div class="showcase-media-placeholder">
              <span>${project.shortTitle || project.title}</span>
            </div>
          `}
          ${showSlideshow ? `
            <button class="showcase-nav showcase-nav-prev" aria-label="Previous">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="15,18 9,12 15,6"></polyline>
              </svg>
            </button>
            <button class="showcase-nav showcase-nav-next" aria-label="Next">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="9,6 15,12 9,18"></polyline>
              </svg>
            </button>
            <div class="showcase-media-dots">
              ${mediaItems.map((_, i) => `<span class="showcase-dot ${i === 0 ? 'active' : ''}" data-index="${i}"></span>`).join('')}
            </div>
          ` : ''}
        </div>
        <div class="showcase-card-body">
          <span class="showcase-card-category">${project.category}${isOngoing ? ' <span class="ongoing-indicator">In Progress</span>' : ''}</span>
          <h3 class="showcase-card-title">${project.shortTitle || project.title}</h3>
          <p class="showcase-card-description">${project.outcome || project.oneLiner || project.description?.slice(0, 100) || ''}</p>
          <div class="showcase-card-tags">
            ${(project.tags || []).slice(0, 4).map(tag => `<span class="showcase-tag">${tag}</span>`).join('')}
          </div>
        </div>
      </article>
    `;
  };

  // Render cards - duplicate for seamless infinite scroll
  const cardsHtml = flagshipProjects.map(generateCard).join('');
  carousel.innerHTML = cardsHtml + cardsHtml; // Duplicate for seamless loop

  // Click and keyboard handlers - open project detail
  carousel.querySelectorAll('.showcase-card').forEach(card => {
    card.addEventListener('click', () => {
      showProjectDetail(card.dataset.projectId);
    });

    // Keyboard accessibility - Enter or Space to activate
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        showProjectDetail(card.dataset.projectId);
      }
    });
  });

  // Check for reduced motion preference
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Initialize media slideshow for each card
  carousel.querySelectorAll('.showcase-card-media').forEach(mediaContainer => {
    const mediaItems = JSON.parse(mediaContainer.dataset.media || '[]');
    const interval = parseInt(mediaContainer.dataset.interval) || 2500;
    const isSlideshow = mediaContainer.dataset.slideshow === 'true';

    // Video cards: just play video continuously (no slideshow)
    if (!isSlideshow) {
      const video = mediaContainer.querySelector('video');
      if (video && !prefersReducedMotion) {
        video.play().catch(() => {});
      }
      return;
    }

    // Image slideshow only
    if (mediaItems.length <= 1) {
      return;
    }

    let currentIndex = 0;
    let slideshowTimer = null;
    let isPaused = false;
    const card = mediaContainer.closest('.showcase-card');
    const dots = mediaContainer.querySelectorAll('.showcase-dot');

    // Function to go to specific slide
    const goToSlide = (index) => {
      const oldIndex = currentIndex;
      currentIndex = ((index % mediaItems.length) + mediaItems.length) % mediaItems.length; // Handle negative

      const currentItem = mediaItems[currentIndex];
      const existingMedia = mediaContainer.querySelector('.showcase-media-item');

      // Fade out current
      if (existingMedia) {
        existingMedia.classList.remove('active');
        existingMedia.classList.add('fading');
      }

      // After fade out, replace content
      setTimeout(() => {
        // Remove old media
        const oldMedia = mediaContainer.querySelector('.showcase-media-item.fading');
        if (oldMedia) oldMedia.remove();

        // Create new media element
        let newEl;
        if (currentItem.type === 'video') {
          newEl = document.createElement('video');
          newEl.className = 'showcase-video showcase-media-item';
          newEl.src = currentItem.src;
          newEl.muted = true;
          newEl.loop = true;
          newEl.playsInline = true;
          newEl.autoplay = true;
          if (!prefersReducedMotion) {
            newEl.play().catch(() => {});
          }
        } else {
          newEl = document.createElement('img');
          newEl.className = 'showcase-media-item';
          newEl.src = currentItem.src;
          newEl.alt = 'Project media';
          newEl.loading = 'lazy';
        }

        // Insert before nav buttons
        const navPrev = mediaContainer.querySelector('.showcase-nav-prev');
        if (navPrev) {
          mediaContainer.insertBefore(newEl, navPrev);
        } else {
          mediaContainer.appendChild(newEl);
        }

        // Trigger fade in
        requestAnimationFrame(() => {
          newEl.classList.add('active');
        });

        // Update dots
        dots.forEach((dot, i) => {
          dot.classList.toggle('active', i === currentIndex);
        });
      }, 450); // Match CSS transition duration
    };

    // Auto advance slideshow
    const startSlideshow = () => {
      if (slideshowTimer) clearInterval(slideshowTimer);
      slideshowTimer = setInterval(() => {
        if (!isPaused && !document.hidden) {
          goToSlide(currentIndex + 1);
        }
      }, interval);
    };

    // Navigation button handlers
    const prevBtn = mediaContainer.querySelector('.showcase-nav-prev');
    const nextBtn = mediaContainer.querySelector('.showcase-nav-next');

    if (prevBtn) {
      prevBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        goToSlide(currentIndex - 1);
        startSlideshow(); // Reset timer after manual nav
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        goToSlide(currentIndex + 1);
        startSlideshow(); // Reset timer after manual nav
      });
    }

    // Dot click handlers
    dots.forEach(dot => {
      dot.addEventListener('click', (e) => {
        e.stopPropagation();
        const index = parseInt(dot.dataset.index);
        goToSlide(index);
        startSlideshow(); // Reset timer after manual nav
      });
    });

    // Pause on hover (but keep showing media)
    card.addEventListener('mouseenter', () => {
      isPaused = true;
    });

    card.addEventListener('mouseleave', () => {
      isPaused = false;
    });

    // Start the slideshow
    if (!prefersReducedMotion) {
      startSlideshow();
    }

    // Ensure first video plays
    const firstVideo = mediaContainer.querySelector('video');
    if (firstVideo && !prefersReducedMotion) {
      firstVideo.play().catch(() => {});
    }
  });

  // Video autoplay with IntersectionObserver (for visibility-based playback)
  if ('IntersectionObserver' in window && !prefersReducedMotion) {
    const videoObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const video = entry.target;
        if (entry.isIntersecting) {
          video.play().catch(() => {});
        } else {
          video.pause();
        }
      });
    }, { threshold: 0.1 }); // Lower threshold for earlier trigger

    carousel.querySelectorAll('.showcase-video').forEach(video => {
      // Try to play immediately if visible
      video.play().catch(() => {});
      videoObserver.observe(video);
    });
  }
}


// =============================================
// PROJECT DETAIL VIEW
// =============================================
function showProjectDetail(projectId) {
  // Check both projects and ongoingProjects
  let project = projects.find(p => p.id === projectId);
  if (!project) {
    project = ongoingProjects.find(p => p.id === projectId);
  }
  if (!project) return;

  // Password protection check
  if (isProjectProtected(projectId)) {
    showPasswordModal(() => showProjectDetail(projectId));
    return;
  }

  currentProject = project;
  closeSidebar();

  document.querySelectorAll('.project-toggle').forEach(t => {
    t.classList.remove('active');
    if (t.dataset.projectId === projectId) t.classList.add('active');
  });

  projectDetail.innerHTML = `
    <div class="project-detail-header">
      <div class="project-title-row">
        <h1 class="project-detail-title">${project.title}</h1>
        ${project.technicalDeepDive ? `
          <button class="tech-deepdive-btn" onclick="toggleTechDeepDive(this)">
            <span class="deepdive-icon">⚡</span>
            <span>Technical Deep Dive</span>
            <svg class="deepdive-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="6,9 12,15 18,9"></polyline>
            </svg>
          </button>
        ` : ''}
      </div>
      ${project.oneLiner ? `<p class="project-detail-oneliner">${project.oneLiner}</p>` : ''}
      <div class="project-detail-tags">
        ${project.tags.map(tag => `<span class="project-tag">${tag}</span>`).join('')}
      </div>
    </div>

    <div class="project-detail-description">
      ${project.description.trim().split('\n\n').map(p => {
        const trimmed = p.trim();
        if (trimmed.includes('•')) {
          const lines = trimmed.split('\n').map(l => l.trim());
          const items = lines.filter(l => l.startsWith('•')).map(l => `<li>${l.substring(1).trim()}</li>`).join('');
          const intro = lines.filter(l => !l.startsWith('•')).join(' ');
          return intro ? `<p>${intro}</p><ul class="tech-list">${items}</ul>` : `<ul class="tech-list">${items}</ul>`;
        }
        return `<p>${trimmed}</p>`;
      }).join('')}
    </div>

    ${project.technicalDeepDive?.metrics ? `
      <div class="project-metrics-grid">
        ${project.technicalDeepDive.metrics.map(m => `
          <div class="project-metric-card">
            <span class="metric-value">${m.value}</span>
            <span class="metric-label">${m.label}</span>
          </div>
        `).join('')}
      </div>
    ` : ''}

    ${project.images?.length && !project.hideGallery ? `
      <div class="project-gallery">
        <h3 class="gallery-title">Project Gallery</h3>
        <div class="gallery-grid">
          ${project.images.map((img, i) => `
            <div class="gallery-item" data-index="${i}">
              <img src="${img}" alt="${project.title} - Image ${i + 1}" onerror="this.parentElement.style.display='none'">
            </div>
          `).join('')}
        </div>
      </div>
    ` : ''}

    ${project.videos?.length ? `
      <div class="project-videos">
        <h3 class="video-title">Project Videos</h3>
        <div class="videos-grid">
          ${project.videos.map((vid, i) => `
            <div class="video-card">
              <div class="video-container">
                <video class="project-video-player" autoplay muted loop playsinline preload="auto" title="Hover to unmute, click for fullscreen">
                  <source src="${vid.src}" type="video/mp4">
                  Your browser does not support video.
                </video>
              </div>
              ${vid.title ? `<p class="video-label">${vid.title}</p>` : ''}
            </div>
          `).join('')}
        </div>
      </div>
    ` : project.video ? `
      <div class="project-video">
        <h3 class="video-title">Project Video</h3>
        <div class="video-container">
          <video class="project-video-player" muted loop playsinline preload="metadata" title="Hover to unmute, click for fullscreen">
            <source src="${project.video}" type="video/mp4">
            Your browser does not support video.
          </video>
        </div>
      </div>
    ` : ''}

    ${project.technicalDeepDive ? `
      <div class="tech-deepdive-section" id="techDeepDive">
        <div class="tech-deepdive-header">
          <div class="tech-deepdive-badge">
            <span class="deepdive-pulse"></span>
            <span>⚡</span>
          </div>
          <div class="tech-deepdive-title">
            <h3>Technical Deep Dive</h3>
            <p>Architecture, Implementation & Under-the-Hood Details</p>
          </div>
        </div>
        <div class="tech-deepdive-content">
          ${project.technicalDeepDive.sections.map((section, i) => `
            <div class="tech-section">
              <div class="tech-section-header">
                <span class="tech-section-number">${i + 1}</span>
                <span class="tech-section-title">${section.title}</span>
              </div>
              <div class="tech-section-content">
                ${section.content}
              </div>
            </div>
          `).join('')}

          ${project.technicalDeepDive.references ? `
            <div class="tech-references">
              <h5>References & Resources</h5>
              <div class="reference-list">
                ${project.technicalDeepDive.references.map(ref => `
                  <a href="${ref.url}" target="_blank" class="reference-link">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                      <polyline points="15 3 21 3 21 9"></polyline>
                      <line x1="10" y1="14" x2="21" y2="3"></line>
                    </svg>
                    ${ref.title}
                  </a>
                `).join('')}
              </div>
            </div>
          ` : ''}
        </div>
      </div>
    ` : ''}
  `;

  document.querySelectorAll('.gallery-item').forEach(item => {
    item.addEventListener('click', () => {
      currentImages = project.images;
      openLightbox(parseInt(item.dataset.index));
    });
  });

  // Video controls: hover to unmute, click for fullscreen
  document.querySelectorAll('.project-video-player').forEach(video => {
    // Hover to unmute
    video.addEventListener('mouseenter', () => {
      video.muted = false;
    });
    video.addEventListener('mouseleave', () => {
      video.muted = true;
    });
    // Click for fullscreen with controls
    video.addEventListener('click', () => {
      if (video.requestFullscreen) {
        video.controls = true;
        video.requestFullscreen();
      } else if (video.webkitRequestFullscreen) {
        video.controls = true;
        video.webkitRequestFullscreen();
      }
    });
    // Remove controls when exiting fullscreen
    document.addEventListener('fullscreenchange', () => {
      if (!document.fullscreenElement) {
        video.controls = false;
      }
    });
  });

  // Save scroll position before showing detail
  savedScrollPosition = window.scrollY;

  contentWrapper.classList.add('hidden');
  projectDetailView.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'instant' });

  if (window.innerWidth < 1024) closeSidebar();
}

function hideProjectDetail() {
  contentWrapper.classList.remove('hidden');
  projectDetailView.classList.remove('active');
  currentProject = null;
  document.querySelectorAll('.project-toggle').forEach(t => t.classList.remove('active'));

  // Restore scroll position
  window.scrollTo({ top: savedScrollPosition, behavior: 'instant' });
}

// =============================================
// LIGHTBOX
// =============================================
function openLightbox(index) {
  currentImageIndex = index;
  updateLightboxImage();
  lightbox.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeLightboxModal() {
  lightbox.classList.remove('active');
  document.body.style.overflow = '';
}

function updateLightboxImage() {
  if (!currentImages.length) return;
  lightboxImage.src = currentImages[currentImageIndex];
  lightboxCounter.textContent = `${currentImageIndex + 1} / ${currentImages.length}`;
}

function nextImage() {
  currentImageIndex = (currentImageIndex + 1) % currentImages.length;
  updateLightboxImage();
}

function prevImage() {
  currentImageIndex = (currentImageIndex - 1 + currentImages.length) % currentImages.length;
  updateLightboxImage();
}

// =============================================
// NAVIGATION
// =============================================
function handleNavigation(e) {
  const link = e.target.closest('.nav-link');
  if (!link) return;

  e.preventDefault();
  if (projectDetailView.classList.contains('active')) hideProjectDetail();

  const targetId = link.getAttribute('href').slice(1);
  const target = document.getElementById(targetId);
  if (target) target.scrollIntoView({ behavior: 'smooth' });

  closeSidebar();
}

// =============================================
// SCROLL REVEAL SYSTEM (Apple-inspired)
// =============================================
function initScrollAnimations() {
  // Check for reduced motion preference
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (prefersReducedMotion) {
    // Just show everything immediately
    document.querySelectorAll('.scroll-reveal').forEach(el => {
      el.classList.add('revealed');
    });
    document.querySelectorAll('.stagger-children').forEach(el => {
      el.classList.add('revealed');
    });
    return;
  }

  // Main scroll reveal observer
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        // Optionally unobserve after revealing (one-time animation)
        // revealObserver.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.15,
    rootMargin: '0px 0px -50px 0px'
  });

  // Add scroll-reveal class to key elements
  const revealElements = [
    '.section-header',
    '.about-content',
    '.skill-card',
    '.project-category',
    '.ongoing-card',
    '.contact-intro',
    '.contact-icons-row'
  ];

  revealElements.forEach(selector => {
    document.querySelectorAll(selector).forEach(el => {
      el.classList.add('scroll-reveal');
      revealObserver.observe(el);
    });
  });

  // Stagger children for grids
  document.querySelectorAll('.skills-grid').forEach(grid => {
    grid.classList.add('stagger-children');
    revealObserver.observe(grid);
  });

  // Section fade-in (backwards compat)
  document.querySelectorAll('.section:not(.hero-section)').forEach(section => {
    revealObserver.observe(section);
    section.classList.add('scroll-reveal');
  });
}

// =============================================
// TAB VISIBILITY (Pause animations when hidden)
// =============================================
function initTabVisibility() {
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      document.body.classList.add('tab-hidden');
      // Pause all videos
      document.querySelectorAll('video').forEach(v => v.pause());
    } else {
      document.body.classList.remove('tab-hidden');
    }
  });
}

// =============================================
// CAROUSEL INFINITE SCROLL (JS-based)
// =============================================
function initCarouselInteraction() {
  const carousel = document.getElementById('showcaseCarousel');
  if (!carousel) return;

  // Remove CSS animation - we'll use JS
  carousel.style.animation = 'none';

  let position = 0;
  let isPaused = false;
  let speed = window.innerWidth <= 768 ? 1.5 : 1; // Faster on mobile

  // Get the width of half the carousel (original content before duplication)
  const getHalfWidth = () => {
    const cards = carousel.querySelectorAll('.showcase-card');
    const halfCount = cards.length / 2;
    let width = 0;
    for (let i = 0; i < halfCount; i++) {
      width += cards[i].offsetWidth + 24; // card width + gap
    }
    return width;
  };

  let halfWidth = getHalfWidth();

  // Recalculate on resize
  window.addEventListener('resize', () => {
    halfWidth = getHalfWidth();
    speed = window.innerWidth <= 768 ? 1.5 : 1;
  });

  // Animation loop
  function animate() {
    if (!isPaused) {
      position += speed;

      // Reset seamlessly when we've scrolled past the first half
      if (position >= halfWidth) {
        position = 0;
      }

      carousel.style.transform = `translateX(-${position}px)`;
    }
    requestAnimationFrame(animate);
  }

  animate();

  // Pause on hover (desktop only)
  const isMobile = window.innerWidth <= 768;
  if (!isMobile) {
    carousel.addEventListener('mouseenter', () => {
      isPaused = true;
    });

    carousel.addEventListener('mouseleave', () => {
      isPaused = false;
    });
  }
}

// =============================================
// INIT
// =============================================
document.addEventListener('DOMContentLoaded', () => {
  // Apply feature flag body classes
  if (FEATURE_FLAGS.SIMPLIFIED_HERO) {
    document.body.classList.add('simplified-hero');
  }
  if (FEATURE_FLAGS.DISABLE_PARTICLES) {
    document.body.classList.add('no-particles');
  }
  if (FEATURE_FLAGS.DISABLE_CUSTOM_CURSOR) {
    document.body.classList.add('no-custom-cursor');
  }
  if (FEATURE_FLAGS.SIMPLIFIED_SKILLS) {
    document.body.classList.add('simplified-skills');
  }
  if (FEATURE_FLAGS.ENABLE_GOLD_THEME) {
    document.body.classList.add('gold-theme');
  }

  // Initialize features
  initParticles();
  initCursor();
  initTyping();
  initCounters();
  initCharReveal();
  initScrollAnimations();
  initTabVisibility();
  renderProjectsCatalog();
  renderOngoingProjects();
  renderShowcase();
  initCarouselInteraction();

  // Event listeners
  hamburger?.addEventListener('click', toggleSidebar);
  sidebarClose?.addEventListener('click', closeSidebar);
  sidebarOverlay?.addEventListener('click', closeSidebar);

  backBtn?.addEventListener('click', hideProjectDetail);
  lightboxClose?.addEventListener('click', closeLightboxModal);
  lightboxPrev?.addEventListener('click', prevImage);
  lightboxNext?.addEventListener('click', nextImage);

  lightbox?.addEventListener('click', (e) => {
    if (e.target === lightbox) closeLightboxModal();
  });

  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', handleNavigation);
  });

  // Copy button functionality
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const textToCopy = btn.dataset.copy;
      navigator.clipboard.writeText(textToCopy).then(() => {
        btn.classList.add('copied');
        setTimeout(() => btn.classList.remove('copied'), 1500);
      });
    });
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (lightbox?.classList.contains('active')) {
      if (e.key === 'Escape') closeLightboxModal();
      if (e.key === 'ArrowRight') nextImage();
      if (e.key === 'ArrowLeft') prevImage();
    } else if (e.key === 'Escape' && sidebar?.classList.contains('open')) {
      closeSidebar();
    }
  });

  // Resize handler
  window.addEventListener('resize', () => {
    if (window.innerWidth >= 1024) {
      sidebarOverlay?.classList.remove('active');
      hamburger?.classList.remove('active');
      document.body.style.overflow = '';
    }
    initCursor();
  });
});

// Technical Deep Dive toggle function
function toggleTechDeepDive(btn) {
  const section = document.getElementById('techDeepDive');
  const isExpanded = section.classList.contains('expanded');

  if (isExpanded) {
    section.classList.remove('expanded');
    btn.classList.remove('active');
  } else {
    section.classList.add('expanded');
    btn.classList.add('active');
    // Smooth scroll to section after a small delay
    setTimeout(() => {
      section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }
}

// Make it globally accessible
window.toggleTechDeepDive = toggleTechDeepDive;

// Export for external use
window.portfolioProjects = projects;
window.addProject = (data) => { projects.push(data); renderProjectsCatalog(); };
