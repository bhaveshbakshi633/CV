// projects.js — portfolio project data
// curated and restructured for capability-based navigation

export const projectData = {
  flagship: [
    {
      id: 'naamika',
      title: 'Naamika: Humanoid Arm Manipulation & Control System',
      shortTitle: 'Naamika Humanoid',
      category: 'Robotics Integration',
      protected: true,
      problem: 'Precise dual-arm manipulation on a bipedal humanoid requires safe, real-time control across simulation and hardware with zero-jitter motion',
      outcome: 'Production 500Hz orchestrator with PPO-based IK, hot-swappable sim/real backends, ArUco vision tracking, and safety-critical motion control on Unitree G1',
      oneLiner: 'Single-authority 500Hz control loop \u2192 PPO IK solver \u2192 arm_sdk DDS \u2192 Unitree G1 with ArUco-guided reaching',
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
      tier: 'flagship',
      capability: 'real-time-control',
      ongoing: false,
      heroLine: "Trained a neural IK solver and deployed it at 500Hz on a real humanoid \u2014 sim, safety, and hardware in one loop.",
      technicalDeepDive: {
        sections: [
          {
            title: "Single-Authority Orchestrator Architecture",
            content: `
            <p>The system enforces a strict <strong>single-writer rule</strong>: only the <code>ControlOrchestrator</code> can issue motor commands. GUI, policies, and cameras are advisory only.</p>
            <div class="code-block"><code>Architecture (RULEBOOK-enforced):
MainWindow (PyQt6) \u2500\u2500\u2500 signals \u2500\u2500\u2192 ControlOrchestrator (500Hz)
PolicyController \u2500\u2500\u2500\u2500 callbacks \u2500\u2500\u2192      \u2502
ArUco Camera \u2500\u2500\u2500\u2500\u2500\u2500 targets \u2500\u2500\u2500\u2192      \u2502
                                         \u2193
                              \u250c\u2500\u2500\u2500 PlantAdapter \u2500\u2500\u2500\u2510
                              \u2502                    \u2502
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
            <div class="code-block"><code>Phase 1 \u2014 Internal IK Solve (no robot motion):
  1. Seed MuJoCo env with current robot joint state
  2. Set target position in environment
  3. Run policy.predict(obs) for up to 200 steps
  4. Check convergence: \u2016hand_pos - target\u2016 < 3cm
  5. Extract final joint positions as IK solution

Phase 2 \u2014 Smoothstep Blend (robot moves):
  Duration: 3.0s at 50Hz (150 steps)
  Interpolation: t\u00b2 \u00d7 (3 - 2t) ease-in-out
  Output: blended joints \u2192 orchestrator \u2192 plant

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
  - set_active(False) \u2192 passive shadow mode
  - No physics stepping when robot is active

RobotPlantAdapter:
  - Wraps arm_sdk via CycloneDDS (LowCmd/LowState)
  - PD control: Kp=50, Kd=1.0 (arms), Kp=200, Kd=5.0 (waist)
  - 29-DOF: arms (14) + waist (3) + legs (12, read-only)

Switching: Sim \u2192 Robot
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
  1. Jump Detection: |target - current| > 1.5 rad \u2192 REJECT
  2. Velocity Limiting: 500Hz pre-computed deltas
     Manual: 1.0 rad/s \u2192 0.002 rad/cycle
     Policy: 2.0 rad/s \u2192 0.004 rad/cycle
  3. Position Error Monitor: |error| > 0.3 rad \u2192 auto-DAMP
  4. Emergency DAMP Sequence:
     a) Set _control_running = False (atomic)
     b) Wait for control thread exit (timeout 1s)
     c) Clear all pending commands
     d) Release weight gradually over 1s (50 steps)
     e) Send DAMP to high-level FSM

Boot FSM (safe startup):
  ZERO_TORQUE \u2192 [2s] \u2192 DAMP \u2192 [2s] \u2192 STAND_UP \u2192 [8s]
  \u2192 READY \u2192 [3s] \u2192 10s hold \u2192 ARM CONTROL ACTIVE

Invariants (from RULEBOOK.md):
  \u2713 Single control loop (500Hz)
  \u2713 Single command writer (orchestrator only)
  \u2713 GUI cannot move hardware directly
  \u2713 Sim and robot never active simultaneously</code></div>
          `
          },
          {
            title: "ArUco Vision-Guided Reaching",
            content: `
            <p>End-to-end pipeline from camera detection to arm motion:</p>
            <div class="code-block"><code>Pipeline:
  G1 RealSense D435 \u2192 TCP socket (5555) \u2192 PC
    \u2193
  ArUco 4x4_50 detection (15mm markers)
    \u2193
  Camera \u2192 Robot frame transform
  (uses pelvis height + cam_pitch for dynamic transform)
    \u2193
  Target lock (remote A button press)
    \u2193
  PPO policy IK (Phase 1: solve internally)
    \u2193
  Smoothstep blend (Phase 2: smooth motion, 3s)
    \u2193
  Orchestrator \u2192 arm_sdk \u2192 G1 motors

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
      problem: 'Training locomotion and whole-body motion policies in simulation that transfer to real 29-DOF humanoid hardware',
      outcome: 'Deployed CR7 celebration, kicks, jumps, and locomotion policies on Unitree G1 with delta action model for motor dynamics',
      oneLiner: 'PPO + SMPL retargeting in 4096 IsaacGym envs \u2192 ONNX export \u2192 50Hz Sim2Real on G1',
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
      tier: 'flagship',
      capability: 'learning-systems',
      ongoing: false,
      heroLine: "Taught a humanoid to celebrate, kick, and walk \u2014 4096 parallel worlds in sim, one real humanoid at 50Hz.",
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
  decimation: 4 \u2192 50Hz policy output

Observation (~453-dim actor):
  Phase-based motion tracking with 4-frame history
  DOF positions (29) + velocities (29) per frame
  Base ang_vel (3) + projected gravity (3)
  Target motion: root_pos, root_rot, DOF targets</code></div>
            <p>Hydra config system (OmegaConf) enables composable experiment management \u2014 mix robot, task, motion, and training configs via YAML overrides.</p>
          `
          },
          {
            title: "SMPL Motion Retargeting Pipeline",
            content: `
            <p>Human motions from the AMASS dataset are retargeted to G1 joint space via a two-stage SMPL fitting pipeline:</p>
            <div class="code-block"><code>Stage 1 \u2014 Shape Fitting (fit_smpl_shape.py):
  SMPL body model \u2192 Optimize \u03b2 shape params
  Loss: \u2016skeleton_lengths(SMPL) - skeleton_lengths(G1)\u2016\u00b2
  Output: G1-proportioned SMPL template

Stage 2 \u2014 Motion Fitting (fit_smpl_motion.py):
  AMASS .npz \u2192 SMPL pose sequence
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
            <p><strong>PPO with GAE</strong> (\u03bb=0.95) optimizes motion tracking reward across 4096 environments simultaneously:</p>
            <div class="code-block"><code>PPO Config:
  \u03b3 (discount): 0.99
  \u03bb (GAE): 0.95
  clip_range: 0.2
  entropy_coef: 0.01
  learning_rate: adaptive (KL target 0.01)
  epochs_per_rollout: 5
  horizon: 24 steps/env
  batch_size: 4096 \u00d7 24 = 98304 transitions

Domain Randomization:
  Link mass: \u00b120%
  PD gains: \u00b125% (Kp and Kd)
  Friction: 0.5\u20131.25 (ground contact)
  Push perturbations: every 5\u201310s, up to 1.0 m/s
  Base COM offset: \u00b110cm (XYZ)
  Control delay: 0\u20132 steps (0\u201390ms)
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
  Network: Linear(in, 256) \u2192 ELU \u2192 Linear(256, 256) \u2192 ELU \u2192 Linear(256, out)
  Output: corrected motor commands

Training:
  Data: Open-loop motor trajectories on real G1
  5000 parallel envs for data collection
  Supervised loss: \u2016predicted_pos - measured_pos\u2016\u00b2
  Compensates for:
    - Motor friction & backlash
    - Thermal drift
    - Cable stiffness
    - Gearbox nonlinearities</code></div>
            <p>The delta model runs inline during deployment, sitting between the policy output and the motor commands \u2014 adding learned corrections at each timestep.</p>
          `
          },
          {
            title: "Sim2Real Deployment (50Hz)",
            content: `
            <p>Trained policies export to ONNX for deployment on the G1's onboard computer. The inference pipeline runs at <strong>50Hz</strong> (20ms loop):</p>
            <div class="code-block"><code>Deployment (newton_controller.py):
  1. Read joint encoders + IMU \u2192 state vector
  2. Construct observation (with 4-frame history)
  3. ONNX Runtime inference \u2192 action (23-dim)
  4. Delta action model correction
  5. action_scale \u00d7 action + default_angles \u2192 targets
  6. PD control: Kp/Kd per-joint \u2192 torque
  7. Send via Unitree SDK2 / CycloneDDS

Deployed Skills:
  - CR7 Siuuu celebration (full body)
  - Forward/side jumps
  - Soccer kicks (left/right)
  - Walking locomotion (decoupled lower body)

Validation:
  MuJoCo sim2sim check before real deployment
  deploy_mujoco.py \u2192 visual gait verification</code></div>
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
      id: 'xr-teleop',
      title: 'XR Teleoperation for Humanoid Robots',
      shortTitle: 'XR Teleop',
      category: 'Robotics Integration',
      protected: true,
      problem: 'Real-time bilateral teleoperation of 29-DOF humanoid with dexterous manipulation',
      outcome: '30Hz control loop, <50ms LAN latency, dual-arm IK + 5 end-effector types',
      oneLiner: 'Quest 3 \u2192 Pinocchio IK \u2192 CycloneDDS \u2192 Unitree G1/H1 with dexterous hands',
      description: `
      Production XR teleoperation for Unitree G1/H1 humanoids using Quest 3 and other headsets. Real-time dual-arm IK via Pinocchio + CasADi + IPOPT, DexPilot hand retargeting for 5 end-effector types, and a safety state machine \u2014 all at 30Hz with <50ms LAN latency over CycloneDDS.
    `,
      tags: ['Pinocchio', 'CasADi', 'Quest 3', 'CycloneDDS', 'DexPilot', 'Isaac Lab'],
      images: [],
      videos: [
        { src: 'assets/projects/xr-teleop/demo.mp4', title: 'XR Teleoperation Demo' }
      ],
      thumbnail: null,
      hideGallery: false,
      tier: 'flagship',
      capability: 'human-robot-interaction',
      ongoing: false,
      heroLine: "Put on a VR headset, moved my hands \u2014 a humanoid 10 feet away moved its. Under 50ms.",
      technicalDeepDive: {
        sections: [
          {
            title: "System Architecture",
            content: `
            <p>Three-tier architecture: XR headset \u2192 Host PC \u2192 Robot, connected via WebSocket + DDS:</p>
            <div class="code-block"><code>Quest 3 (OpenXR)          Host PC (Python)              Robot (G1/H1)
  Hand/Head Tracking    teleop_hand_and_arm.py         DDS Domain 0/1
    \u2193 Vuer.js             \u251c\u2500 Arm IK (Pinocchio+CasADi)   \u251c\u2500 rt/lowcmd (arms)
  WebSocket :8012         \u251c\u2500 Hand Retarget (DexPilot)     \u251c\u2500 rt/inspire/cmd (hands)
    \u2193 HTTPS/WSS           \u251c\u2500 Safe Teleop (state machine)  \u251c\u2500 rt/lowstate (feedback)
  LocalTunnel/ngrok       \u2514\u2500 Camera (teleimager)          \u2514\u2500 Motor Controllers

Submodules:
  televuer   \u2192 XR input capture (WebSocket/WebRTC)
  teleimager \u2192 Camera streaming (ZMQ/WebRTC)
  dex-retargeting \u2192 DexPilot finger IK library</code></div>
            <p>119 Python source files. Entry point: <code>teleop_hand_and_arm.py</code>. Supports 4 robot variants (G1 29/23-DOF, H1, H1_2) and 5 end-effector types.</p>
          `
          },
          {
            title: "Dual-Arm Inverse Kinematics",
            content: `
            <p>Real-time nonlinear IK using Pinocchio for kinematics, CasADi for symbolic autodiff, and IPOPT as the NLP solver:</p>
            <div class="code-block"><code>Optimization Problem (per arm, 7-DOF):
  minimize: 50\u00b7\u2016p_target - FK(q)\u2016\u00b2 + \u2016R_target - R(q)\u2016\u00b2
            + 0.02\u00b7\u2016q - q_neutral\u2016\u00b2 + 0.1\u00b7\u2016q - q_prev\u2016\u00b2
  subject to: q_min \u2264 q \u2264 q_max

IPOPT Config:
  max_iter: 30, tol: 1e-4, warm_start: enabled
  Solve time: 5-10ms (warm-started)

Model: g1_29_model_cache.pkl (Pinocchio binary)
  Locks 27 joints (legs, waist, fingers) \u2192 reduced 14-DOF arm problem
  Frame targets: L_ee, R_ee (wrist + 0.05m offset)</code></div>
            <p>Weighted moving filter (\u03b1 = [0.4, 0.3, 0.2, 0.1]) smooths joint trajectories. Null-space biases toward elbow-down configurations.</p>
          `
          },
          {
            title: "DexPilot Hand Retargeting",
            content: `
            <p>Maps 25-DOF OpenXR hand skeleton to robot finger commands via optimization-based retargeting:</p>
            <div class="code-block"><code>Pipeline:
  OpenXR 25 joints/hand \u2192 Extract fingertip positions
    \u2193 DexPilot Optimizer
  Constraints: Robot URDF model, joint limits, collision avoidance
    \u2193 Output: joint commands per hand
  Low-pass filter (\u03b1=0.2) \u2192 Normalize to motor range
    \u2193 Publish via DDS (rt/inspire/cmd)

End-Effector Support:
  Inspire DFX/FTP: 6 motors/hand (4 proximal + 2 thumb)
  Dex3:            7 motors/hand (3 per finger: prox/mid/dist)
  Dex1:            2 motors/hand (simple gripper, linear map)
  BrainCo:         6 motors/hand

Config (YAML):
  type: DexPilot | vector
  scaling_factor: 1.20
  low_pass_alpha: 0.2</code></div>
            <p>Resolves 25 human DOF \u2192 6-7 robot DOF via constrained optimization with collision avoidance.</p>
          `
          },
          {
            title: "Control Loop & Latency",
            content: `
            <p>30Hz main loop (~33ms per iteration) with warm-started IK:</p>
            <div class="code-block"><code>Loop Cycle (33ms budget):
  1. Receive XR pose from Vuer (WebSocket)        ~2ms
  2. Transform OpenXR \u2192 Robot frame                ~1ms
  3. Scale arms (human 0.60m \u2192 robot 0.75m)        ~1ms
  4. Solve dual-arm IK via IPOPT (warm-start)    5-10ms
  5. Check collision + joint limits                ~1ms
  6. Publish LowCmd to DDS (rt/lowcmd)             ~1ms
  7. Retarget hands via DexPilot                  1-2ms
  8. Publish hand commands (rt/inspire/cmd)         ~1ms
  9. Encode + send camera frame                   5-8ms

Total Latency:
  LAN:      30-50ms  (direct WiFi 6)
  Internet: 100-250ms (via LocalTunnel/ngrok)</code></div>
            <p>Camera: RealSense D435I @ 480\u00d7640, JPEG quality 60%, streamed via ZMQ (tcp://172.16.2.242:55555) or WebRTC (port 60001).</p>
          `
          },
          {
            title: "Safety & Deployment",
            content: `
            <p>Safety state machine prevents unsafe transitions. DDS domain isolation separates real robot from simulation:</p>
            <div class="code-block"><code>State Machine (safe_teleop.py):
  IDLE \u2192 ALIGNING (5s timeout) \u2192 TRACKING \u2194 FAULT
                                    \u2193          \u2193
                                 EXITING \u2190 RECOVERING

Safety Parameters:
  Alignment threshold: 0.05 rad (~3\u00b0)
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
      id: 'rl-training-center',
      title: 'Isaac Lab RL Training Environment for Precision Arm Reaching',
      shortTitle: 'RL Training Center',
      category: 'Reinforcement Learning',
      status: 'Active Development',
      progress: '~70% Complete',
      description: 'Custom Isaac Lab environment for training PPO-based arm reaching policies on the Unitree G1. 3-phase curriculum learning with distance-based action scaling for precision manipulation.',
      currentWork: 'Extending to bimanual coordination and integrating with arm control GUI',
      tags: ['Isaac Lab', 'RSL-RL', 'PPO', 'PhysX', 'PyTorch', 'Curriculum Learning'],
      thumbnail: 'assets/projects/rl-training-center/img1.jpg',
      images: [
        'assets/projects/rl-training-center/img1.jpg',
        'assets/projects/rl-training-center/img2.jpg',
        'assets/projects/rl-training-center/img3.jpg',
        'assets/projects/rl-training-center/img4.jpg',
        'assets/projects/rl-training-center/img5.jpg',
        'assets/projects/rl-training-center/img6.jpg',
        'assets/projects/rl-training-center/img7.jpg'
      ],
      videos: [],
      tier: 'flagship',
      capability: 'learning-systems',
      ongoing: true,
      heroLine: "Custom Isaac Lab curriculum that teaches a robot arm to reach within 5mm from scratch \u2014 no demonstrations.",
      fullDescription: `
      Custom reinforcement learning environment built on NVIDIA Isaac Lab (Isaac Sim) for training
      precision arm reaching policies on the Unitree G1 humanoid. Uses RSL-RL PPO with a 3-phase
      curriculum that progressively tightens success thresholds from 20mm \u2192 10mm \u2192 5mm.

      <strong>Environment Design (Isaac Lab):</strong>
      \u2022 Isaac Lab (Isaac Sim 4.x + PhysX GPU) \u2014 NOT Isaac Gym
      \u2022 27-dim observation space: joint pos (7) + joint vel (7) + hand pos (3) + target pos (3) + error vec (3) + distance (1) + prev action (7) - 4
      \u2022 7-dim action space: delta joint angles for single arm
      \u2022 Distance-based action scaling: actions shrink as hand approaches target
      \u2022 Episode length: 200 steps with auto-reset on success or timeout

      <strong>3-Phase Curriculum:</strong>
      \u2022 Phase 1 (0\u20132000 iter): Success < 20mm, wide workspace, large action scale
      \u2022 Phase 2 (2000\u20135000 iter): Success < 10mm, refined targets, medium scale
      \u2022 Phase 3 (5000+ iter): Success < 5mm, precision reaching, small action scale
      \u2022 Automatic phase advancement based on moving average success rate > 80%

      <strong>Reward Function:</strong>
      \u2022 Dense: -distance_to_target (continuous gradient signal)
      \u2022 Sparse: +10 bonus on reaching within threshold
      \u2022 Penalties: joint velocity (-0.01), joint acceleration (-0.001), action magnitude (-0.005)
      \u2022 Alive bonus: +0.1 per step (encourages exploration)

      <strong>Training Stack:</strong>
      \u2022 RSL-RL PPO (NOT Stable-Baselines3): optimized for GPU-parallel sim
      \u2022 MLP policy: 256 \u2192 256 \u2192 7 with ELU activations
      \u2022 Training: ~2hrs on RTX 3090 for convergence
      \u2022 Export: PyTorch checkpoint \u2192 loaded directly in arm control GUI
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
  Robot: Unitree G1 (URDF \u2192 USD articulation)
  Action type: delta joint positions (7-DOF arm)

Observation Space (27-dim):
  joint_positions (7): current arm angles
  joint_velocities (7): current arm velocities
  hand_position (3): end-effector XYZ
  target_position (3): goal XYZ
  error_vector (3): target - hand
  distance (1): \u2016error\u2016
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
            <div class="code-block"><code>Phase 1 \u2014 Coarse Reaching (0\u20132000 iter):
  Success threshold: 20mm
  Action scale: 0.05 rad (large corrections OK)
  Workspace: full arm range
  Target: learn general reaching behavior

Phase 2 \u2014 Medium Precision (2000\u20135000 iter):
  Success threshold: 10mm
  Action scale: 0.03 rad (moderate)
  Workspace: refined region
  Target: consistent sub-cm reaching

Phase 3 \u2014 Fine Precision (5000+ iter):
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
effective_action = raw_action \u00d7 action_scale \u00d7 scale_factor

Effect:
  Far (>10cm): Full action range \u2192 fast approach
  Medium (3\u201310cm): Reduced actions \u2192 controlled approach
  Close (<3cm): Minimal actions \u2192 precision adjustment

This prevents overshoot near the target without
sacrificing speed during gross motion.</code></div>
            <p>Combined with the curriculum, this produces policies that achieve ~3cm accuracy in Phase 1, ~1cm in Phase 2, and ~5mm in Phase 3.</p>
          `
          },
          {
            title: "RSL-RL PPO Training",
            content: `
            <p>Uses <strong>RSL-RL</strong> (from ETH Z\u00fcrich / Legged Robotics), optimized for GPU-parallel simulation \u2014 NOT Stable-Baselines3:</p>
            <div class="code-block"><code>PPO Config:
  Algorithm: RSL-RL ActorCritic PPO
  Policy MLP: [256, 256] + ELU activations
  Value MLP: [256, 256] + ELU activations
  \u03b3 (discount): 0.99
  \u03bb (GAE): 0.95
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
  ],
  supporting: [
    {
      id: 'arm-control-gui',
      title: 'Humanoid Arm Control GUI with MuJoCo & PPO IK',
      shortTitle: 'Arm Control GUI',
      category: 'Robotics Integration',
      problem: 'Need unified interface to test arm policies in simulation before deploying to real robot hardware',
      outcome: 'Hot-swappable sim/real backend with 500Hz orchestrator, PPO-based IK, and 30fps MuJoCo viz',
      oneLiner: 'PyQt6 GUI \u2192 500Hz orchestrator \u2192 PPO IK solver \u2192 PlantAdapter (MuJoCo \u2194 arm_sdk/DDS)',
      description: `
      Unified PyQt6 interface for Unitree G1 dual-arm manipulation. 500Hz single-authority orchestrator with PPO-based IK, direct joint control, ArUco vision targeting, and hot-swappable MuJoCo sim / real robot backends via PlantAdapter abstraction.
    `,
      tags: ['PyQt6', 'MuJoCo', 'PPO', 'arm_sdk', 'DDS', 'ArUco'],
      images: [
        'assets/projects/arm-control-gui/screenshot1.jpg',
        'assets/projects/arm-control-gui/screenshot2.jpg',
        'assets/projects/arm-control-gui/screenshot3.jpg',
        'assets/projects/arm-control-gui/screenshot4.jpg'
      ],
      videos: [
        { src: 'assets/projects/arm-control-gui/demo_video.mp4', title: 'GUI Demo with MuJoCo Simulation' }
      ],
      thumbnail: 'assets/projects/arm-control-gui/screenshot2.jpg',
      tier: 'supporting',
      capability: 'real-time-control',
      ongoing: false,
      heroLine: "One GUI, two backends \u2014 switch from simulation to real robot mid-session without restarting.",
      technicalDeepDive: {
        sections: [
          {
            title: "Single-Authority Orchestrator",
            content: `
            <p>Strict <strong>single-writer rule</strong>: only the <code>ControlOrchestrator</code> issues motor commands. GUI, policies, and cameras are advisory only:</p>
            <div class="code-block"><code>Architecture:
MainWindow (PyQt6) \u2500\u2500\u2500 signals \u2500\u2500\u2192 ControlOrchestrator (500Hz)
PolicyController \u2500\u2500\u2500\u2500 callbacks \u2500\u2500\u2192      \u2502
ArUco Camera \u2500\u2500\u2500\u2500\u2500\u2500 targets \u2500\u2500\u2500\u2192      \u2502
                                         \u2193
                              \u250c\u2500\u2500\u2500 PlantAdapter \u2500\u2500\u2500\u2510
                              \u2502                    \u2502
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
3. pixels = renderer.render() \u2192 numpy RGB
4. np.copy(pixels) inside lock (isolation)
5. QImage from buffer \u2192 QPixmap
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
  - set_active(False) \u2192 passive shadow mode
  - No physics stepping when robot is active

RobotPlantAdapter:
  - arm_sdk via CycloneDDS (LowCmd/LowState)
  - PD control: Kp=50, Kd=1.0 (arms)
  - 29-DOF: arms (14) + waist (3) + legs (12, read-only)

Switching Sim \u2192 Robot:
  1. sim_adapter.set_active(False)  # passive
  2. orchestrator.use_robot_plant()
  3. State transfer: joints seed new plant</code></div>
          `
          },
          {
            title: "PPO Policy as IK Solver",
            content: `
            <p>Arm reaching uses a <strong>trained PPO neural network</strong> instead of analytical IK or Jacobian methods:</p>
            <div class="code-block"><code>Phase 1 \u2014 Internal IK Solve (no robot motion):
  1. Seed MuJoCo env with current joint state
  2. Set target position in environment
  3. Run policy.predict(obs) for up to 200 steps
  4. Check convergence: \u2016hand - target\u2016 < 3cm
  5. Extract final joint positions as IK solution

Phase 2 \u2014 Smoothstep Blend (robot moves):
  Duration: 3.0s at 50Hz (150 steps)
  Interpolation: t\u00b2 \u00d7 (3 - 2t) ease-in-out

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
  1. Jump Detection: |target - current| > 1.5 rad \u2192 REJECT
  2. Velocity Limiting: 500Hz pre-computed deltas
     Manual: 1.0 rad/s \u2192 0.002 rad/cycle
     Policy: 2.0 rad/s \u2192 0.004 rad/cycle
  3. Position Error Monitor: |error| > 0.3 rad \u2192 auto-DAMP
  4. Emergency DAMP: gradual weight release over 1s

Thread Safety (RLock):
  qpos_lock protects MuJoCo data.qpos array
  Non-blocking render: _pending_render flag
  Lock order: control \u2192 render (no deadlock)</code></div>
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
      id: 'sim2real-deploy',
      title: 'ASAP Sim2Real Deployment & Skill Execution on G1',
      shortTitle: 'Sim2Real Deploy',
      category: 'Robotics Deployment',
      protected: true,
      problem: 'Deploying RL-trained whole-body skills from simulation to real Unitree G1 hardware with stable execution',
      outcome: 'Real-time 50Hz skill execution on G1: CR7 celebration, kicks, jumps, and locomotion via ONNX + delta action model',
      oneLiner: 'ONNX policy + delta action model \u2192 50Hz newton controller \u2192 Unitree SDK2 \u2192 G1 whole-body skills',
      description: `
      Sim-to-real deployment framework for ASAP-trained policies on the Unitree G1. Newton controller runs ONNX policies at 50Hz with inline delta action model correction via Unitree SDK2. Deploys CR7 celebrations, kicks, jumps, and locomotion after MuJoCo sim2sim validation.
    `,
      tags: ['Sim2Real', 'ONNX', 'Unitree SDK2', 'CycloneDDS', 'MuJoCo', 'Python'],
      images: [],
      videos: [],
      thumbnail: null,
      hideGallery: true,
      tier: 'supporting',
      capability: 'human-robot-interaction',
      ongoing: false,
      heroLine: "ONNX policy running at 50Hz on real hardware. No ROS. No middleware. Just inference and actuation.",
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
  3. ONNX Runtime inference \u2192 action (23-dim)
  4. Delta action model correction
  5. target = action_scale \u00d7 action + default_angles
  6. PD torque: \u03c4 = Kp(target - q) - Kd(dq)
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
  Network: Linear(23, 256) \u2192 ELU
           \u2192 Linear(256, 256) \u2192 ELU
           \u2192 Linear(256, 23)
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
    1. Blend from locomotion \u2192 skill (5 frames)
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
    }
  ],
  archived: [
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
      tier: 'archived',
      capability: 'hardware-build',
      ongoing: false,
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
  Propeller: 10\u00d74.7 (10" diameter, 4.7" pitch)
  Battery: 3S LiPo 2200mAh (11.1V)
  Thrust-to-weight ratio: >1.3

Servos: SG90 9g micro servos \u00d7 3
  Aileron: \u00b120\u00b0 deflection
  Elevator: \u00b125\u00b0 deflection
  Rudder: \u00b130\u00b0 deflection</code></div>
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
  Props: 5\u00d74.5 tri-blade

FPV System:
  Camera: CMOS 1200TVL (2.1mm lens, 150\u00b0 FOV)
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
      images: ['assets/projects/atv/img1.jpg', 'assets/projects/atv/img2.jpg', 'assets/projects/atv/img3.jpg', 'assets/projects/atv/img4.jpg'],
      video: 'assets/projects/atv/video.mp4',
      thumbnail: 'assets/projects/atv/img1.jpg',
      tier: 'archived',
      capability: 'hardware-build',
      ongoing: false,
      technicalDeepDive: {
        sections: [
          {
            title: "Space Frame Design & Material Selection",
            content: `
            <p>Tubular space frame chassis designed to BAJA SAE rule specifications with AISI 4130 chromoly steel:</p>
            <div class="code-block"><code>Chassis Specifications:
  Material: AISI 4130 chromoly steel
  Primary tubes: 1" OD \u00d7 0.065" wall
  Secondary: 1" OD \u00d7 0.049" wall
  Bracing: 3/4" OD \u00d7 0.049" wall
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
     FOS: 1.64 \u2192 reinforced to 2.1

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

  5. Fatigue (10\u2076 cycles):
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
  Steel fixture table (leveled \u00b10.5mm)
  Locating pins at all node points
  Dimensional tolerance: \u00b11mm
  Assembly sequence: RHD \u2192 LHD \u2192 cross

DFM Optimization (25% cost reduction):
  - Standardized tube sizes (3 diameters only)
  - Minimized bend count (straight members preferred)
  - Common notch angles (30\u00b0, 45\u00b0, 60\u00b0, 90\u00b0)
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
          { value: "\u00b11mm", label: "Tolerance" }
        ]
      }
    },
    {
      id: 'hand-target',
      title: 'Multi-Method Object Detection & 6-DOF Pose Estimation',
      shortTitle: 'Hand Target',
      category: 'Computer Vision',
      problem: 'Robot arms need stable, accurate 6-DOF target poses for manipulation \u2014 single detection method insufficient',
      outcome: 'Sub-centimeter accuracy with 3 detection methods (ArUco + HSV + YOLOv8), median+Kalman temporal smoothing',
      oneLiner: 'RealSense RGB-D \u2192 ArUco / HSV / YOLOv8 detection \u2192 Median + Kalman smoothing \u2192 TCP socket to robot',
      description: `
      Real-time 6-DOF pose estimation for robot arm manipulation with three interchangeable detection backends: ArUco markers, HSV color segmentation, and custom-trained YOLOv8. Median + Kalman temporal smoothing produces jitter-free trajectories streamed via TCP to the robot controller.
    `,
      tags: ['RealSense', 'ArUco', 'YOLOv8', 'Kalman Filter', 'OpenCV', 'Python'],
      images: [],
      videos: [],
      thumbnail: null,
      hideGallery: true,
      tier: 'archived',
      capability: 'real-time-control',
      ongoing: false,
      technicalDeepDive: {
        sections: [
          {
            title: "Multi-Method Detection Architecture",
            content: `
            <p>Three interchangeable detection backends share a common output interface \u2014 position + optional orientation in camera frame:</p>
            <div class="code-block"><code>Detection Methods:
  1. ArUco (aruco_detector.py):
     cv2.aruco.detectMarkers() \u2192 corners, ids
     cv2.aruco.estimatePoseSingleMarkers() \u2192 rvec, tvec
     Full 6-DOF pose from marker geometry

  2. HSV Color (color_detector.py):
     cv2.cvtColor(BGR\u2192HSV) \u2192 inRange threshold
     cv2.findContours() \u2192 largest contour centroid
     Depth lookup \u2192 3D position (no orientation)

  3. YOLOv8 (yolo_detector.py):
     Ultralytics YOLOv8 inference on RGB frame
     Bounding box center \u2192 depth lookup \u2192 3D position
     Custom trained on 200+ labeled images

Common Output:
  { position: [x, y, z], orientation: [qw, qx, qy, qz],
    confidence: float, method: str }</code></div>
          `
          },
          {
            title: "ArUco Marker System",
            content: `
            <p>Primary detection for precise 6-DOF pose. Multiple 15mm ArUco markers (4\u00d74_50 dictionary) arranged in strip patterns for cylindrical objects:</p>
            <div class="code-block"><code>ArUco Pipeline:
  1. detectMarkers() \u2192 corners, ids
  2. For each marker:
     - estimatePoseSingleMarkers(15mm) \u2192 rvec, tvec
     - Rodrigues(rvec) \u2192 rotation matrix \u2192 quaternion
  3. Position: centroid of all marker tvecs
  4. Rotation: quaternion averaging (SLERP-based)

Strip Generator (strip_generator.py):
  Input: marker IDs, count, spacing
  Output: A4-printable PDF strip for wrapping
  Marker size: 15mm (configurable)

Outlier Rejection:
  Markers with \u2016tvec - median_tvec\u2016 > 2\u03c3 discarded
  Quaternion dot product < 0.9 \u2192 inconsistent \u2192 reject</code></div>
          `
          },
          {
            title: "Temporal Smoothing: Median + Kalman Filter",
            content: `
            <p>Raw detections are noisy. The pipeline uses <strong>median filter + Kalman filter</strong> (NOT exponential moving average) for stable robot-friendly output:</p>
            <div class="code-block"><code>Smoothing Pipeline:
  Raw detection \u2192 Median filter (window=5)
                \u2192 Kalman filter (state estimation)
                \u2192 Output to robot controller

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
  Model: YOLOv8n (nano \u2014 fast inference)
  Framework: Ultralytics
  Epochs: 100
  Input: 640\u00d7640 RGB

Inference Pipeline:
  1. YOLO predict on RGB frame \u2192 bbox + confidence
  2. bbox center \u2192 pixel coordinates (u, v)
  3. depth_frame.get_distance(u, v) \u2192 Z meters
  4. Deproject: X = (u - cx) * Z / fx
  5. Output: 3D position (no orientation)</code></div>
            <p>Fallback chain: ArUco (best) \u2192 YOLOv8 (good) \u2192 HSV color (basic). Automatic method selection based on detection confidence.</p>
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
    }
  ]
};
