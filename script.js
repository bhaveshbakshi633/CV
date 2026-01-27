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
const FLAGSHIP_PROJECT_IDS = ['naamika', 'asap', 'rl-training-center', 'rc-uav', 'atv'];

// Ongoing Projects (separate from completed projects)
const ongoingProjects = [
  {
    id: 'rl-training-center',
    title: 'Universal RL Training Center',
    shortTitle: 'RL Training Center',
    category: 'Reinforcement Learning',
    status: 'Active Development',
    progress: '~70% Complete',
    description: 'End-to-end GUI for training RL agents on any robot. Joint visualization, reward design, curriculum training, domain randomization — all in one PyQt6 application.',
    currentWork: 'Building model comparison tools and Isaac Lab integration',
    tags: ['PyQt6', 'MuJoCo', 'PPO', 'Stable-Baselines3', 'URDF'],
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
      Comprehensive PyQt6 application for end-to-end reinforcement learning workflow on robotic systems.
      Designed to eliminate the fragmented tooling problem in robotics RL research.

      <strong>Completed Features (12 Tabs):</strong>
      • Joint Explorer: Interactive MuJoCo 3D visualization with real-time joint manipulation
      • Reward Designer: Visual reward function composition with live preview
      • Training Dashboard: Real-time loss curves, episode rewards, and TensorBoard integration
      • URDF to MJCF Converter: One-click conversion with mesh/joint/actuator mapping
      • Start Pose Editor: Save/load pose library for consistent training initialization
      • Hardware Profiler: Auto-detect GPU/CPU and recommend batch sizes
      • Domain Randomization: Mass, friction, noise, delays — all configurable per-parameter
      • Curriculum Builder: Define staged difficulty progression

      <strong>In Progress:</strong>
      • Model Testing Framework: Load and compare trained policies with quantitative metrics
      • Isaac Lab Backend: GPU-accelerated parallel training
      • Model Library: Pre-trained policies for common robots

      <strong>Tech Stack:</strong> PyQt6, MuJoCo, Stable-Baselines3, ONNX export, TensorBoard
    `,
    technicalDeepDive: {
      sections: [
        {
          title: "Architecture",
          content: `
            <p>Built on <strong>PyQt6</strong> with a modular dock-based layout. Each feature is an independent widget that can be rearranged, floated, or tabbed.</p>
            <p>MuJoCo rendering uses <code>mujoco.Renderer</code> with offscreen framebuffer, converted to QPixmap for display. Achieves 60fps on integrated graphics.</p>
          `
        },
        {
          title: "Reward Designer",
          content: `
            <p>Visual reward function builder with <strong>drag-and-drop components</strong>:</p>
            <p>Distance to target, orientation alignment, velocity tracking, energy penalty, collision penalty, joint limit penalty.</p>
            <p>Each component has configurable weight, and the combined reward function is previewed as Python code.</p>
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
    title: 'Naamika: Voice-Controlled Humanoid with Whole Body Control',
    shortTitle: 'Naamika Humanoid',
    category: 'Robotics Integration',
    flagship: true,
    problem: 'Humanoid robots need seamless integration of locomotion, manipulation, navigation, and natural interaction',
    outcome: 'Deployed 29-DOF bipedal humanoid with voice control, SLAM navigation, arm manipulation, and RL-trained gestures',
    oneLiner: 'Full-stack humanoid system: WBC, SLAM, voice AI, and gesture control on real hardware',
    description: `
      Complete humanoid robot system deployed on a 29-DOF bipedal platform. Integrates whole-body control,
      autonomous navigation, arm manipulation, and voice-driven AI interaction into a single production system.

      <strong>Whole Body Control:</strong>
      • 29-DOF joint control: legs (12), arms (14), waist (3)
      • RL-trained locomotion policies (PPO, Isaac Gym) for walking, turning, gestures
      • Real-time arm trajectory control with inverse kinematics
      • Coordinated upper-lower body motion during locomotion
      • 500Hz control loop via Unitree SDK2 (CycloneDDS)

      <strong>Navigation & SLAM:</strong>
      • Real-time SLAM with waypoint persistence
      • Collision-aware path planning via costmap
      • Time-based locomotion control (velocity commands)
      • Indoor localization and map management

      <strong>Voice & AI Integration:</strong>
      • Whisper STT + Ollama LLM (llama3.1:8b) for natural interaction
      • RAG pipeline with FAISS for domain knowledge
      • C++ safety gatekeeper with 22-action whitelist
      • 50ms emergency stop fast-path
      • Neural TTS for speech output

      <strong>System Architecture:</strong>
      • Distributed ROS2 nodes across multiple machines
      • Flask orchestrator coordinating voice, motion, and navigation
      • WebSocket pub/sub for real-time state sync
    `,
    tags: ['Humanoid', 'WBC', 'SLAM', 'ROS2', 'LLM', 'Python', 'C++'],
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
    hideGallery: true,
    technicalDeepDive: {
      sections: [
        {
          title: "Voice Activity Detection Pipeline",
          content: `
            <p>The VAD system uses <strong>Silero VAD</strong> (ONNX runtime) for real-time speech detection at 16kHz sample rate. Audio chunks arrive via WebSocket from the browser's MediaRecorder API in 16-bit PCM format.</p>
            <div class="code-block"><code>VADConfig:
  sample_rate: 16000
  frame_size: 512 samples (32ms)
  threshold: 0.5
  silence_timeout: 0.5s
  min_speech_duration: 0.25s</code></div>
            <p>The pipeline uses a state machine: <code>IDLE → SPEECH_DETECTED → RECORDING → SILENCE_DETECTED → PROCESSING</code>. The 0.5s silence timeout allows natural pauses without premature cutoff.</p>
          `
        },
        {
          title: "Hybrid RAG Architecture",
          content: `
            <p>Unlike pure RAG systems that lose conversation context, NAAMIKA implements a <strong>Hybrid RAG</strong> approach that maintains both conversation memory and knowledge retrieval simultaneously.</p>
            <div class="formula">Prompt = SystemPrompt + CriticalFacts + ConversationHistory[-10:] + RetrievedContext[top-5] + UserQuery</div>
            <p><strong>Vector Store:</strong> FAISS with HuggingFace sentence-transformers (<code>all-MiniLM-L6-v2</code>, 384-dim embeddings). Chunk size: 512 tokens with 50-token overlap using RecursiveCharacterTextSplitter.</p>
            <p><strong>Anti-Hallucination:</strong> Critical facts (exact values for frequently-asked questions) are injected into EVERY prompt to prevent LLM fabrication. This includes specific dates, numbers, and named entities that LLMs commonly hallucinate.</p>
          `
        },
        {
          title: "Intent Classification & Safety Gating",
          content: `
            <p>Before LLM inference, the <code>IntentReasoner</code> module classifies input into three categories:</p>
            <p><strong>1. ACTION:</strong> Direct robot commands (22 whitelisted actions) - bypasses LLM, goes directly to robot API.</p>
            <p><strong>2. CONVERSATION:</strong> General chat - processed by LLM with RAG augmentation.</p>
            <p><strong>3. QUERY:</strong> Questions about specific topics - uses RAG retrieval.</p>
            <div class="code-block"><code>Risk Levels:
  LOW: wave, shake_hand, hug → Immediate execution
  MEDIUM: forward, backward, turn → Requires confirmation
  HIGH: init, standup, damp → Strict confirmation + timeout</code></div>
            <p>The <strong>STOP fast-path</strong> bypasses all processing - keywords like "stop/halt/freeze/ruk/bas" trigger immediate DAMP state within 50ms through direct ROS2 topic publishing.</p>
          `
        },
        {
          title: "Parallel TTS Pipeline",
          content: `
            <p>To minimize perceived latency, TTS generation and playback run in parallel threads with a producer-consumer queue:</p>
            <div class="code-block"><code>┌─────────────────────┐     ┌─────────────────────┐
│  TTSGenerator       │     │  AudioPlayer        │
│  (Thread 1)         │     │  (Thread 2)         │
│                     │     │                     │
│  text → TTS → mp3 ────────→ queue.get() → play │
│                     │     │                     │
│  Generates chunk N+1│     │  Plays chunk N      │
│  WHILE N plays      │     │  SIMULTANEOUSLY     │
└─────────────────────┘     └─────────────────────┘</code></div>
            <p>First chunk: 50 chars (quick response). Subsequent chunks: 150 chars (better prosody). Backend options: Edge TTS (neural), Chatterbox (custom voice), pyttsx3 (offline).</p>
          `
        },
        {
          title: "Robot Communication Layer",
          content: `
            <p>Commands flow through a gatekeeper architecture before reaching the robot:</p>
            <p><code>Brain (Python) → HTTP POST → Action Gatekeeper (C++/ROS2) → G1 Orchestrator → Unitree SDK2 (CycloneDDS)</code></p>
            <p>The C++ gatekeeper maintains a <strong>whitelist</strong> of 22 approved actions and performs semantic veto (question words in input block action execution). Time-based locomotion commands auto-stop after configurable duration (2s forward/back, 1.5s turns).</p>
          `
        }
      ],
      metrics: [
        { value: "< 200ms", label: "Voice → Response" },
        { value: "22", label: "Whitelisted Actions" },
        { value: "50ms", label: "STOP Latency" },
        { value: "384-dim", label: "Embedding Size" }
      ],
      references: [
        { title: "Silero VAD", url: "https://github.com/snakers4/silero-vad" },
        { title: "LangChain RAG", url: "https://python.langchain.com/docs/tutorials/rag/" },
        { title: "Ollama API", url: "https://ollama.ai/docs/api" },
        { title: "Unitree SDK2", url: "https://github.com/unitreerobotics/unitree_sdk2" }
      ]
    }
  },
  {
    id: 'asap',
    title: 'Bipedal Motion Learning with ASAP Framework',
    shortTitle: 'ASAP Motion RL',
    category: 'Reinforcement Learning',
    problem: 'Training locomotion policies in simulation that actually transfer to real robots',
    outcome: 'Deployed walking, kicking, and gesture policies with <5% sim-to-real performance gap',
    oneLiner: 'Sim2Real locomotion using PPO in 4096 parallel Isaac Gym environments',
    description: `
      Deployed whole-body agile motion policies on a 23-DOF bipedal humanoid robot using the
      open-source ASAP (Aligning Simulation And real-world Physics) framework. Achieved successful
      sim-to-real transfer of dynamic locomotion and expressive gesture skills.

      <strong>Training Pipeline:</strong>
      • Multi-simulator setup: IsaacGym (4096 parallel envs), MuJoCo, Genesis
      • PPO with GAE (λ=0.95) and adaptive learning rate scheduling
      • Phase-based motion tracking with delta action residual learning
      • Domain randomization: mass (±15%), friction (0.5-1.2), motor strength (±10%)

      <strong>Motion Retargeting:</strong>
      • SMPL human mesh → robot joint mapping via inverse kinematics
      • AMASS motion capture dataset (40+ hours of human motion)
      • Temporal alignment with DTW for motion synchronization
      • Joint limit constraints and self-collision avoidance

      <strong>Sim2Real Transfer:</strong>
      • ONNX model export for 500Hz inference on robot
      • Actuator network for motor dynamics compensation
      • State estimation with EKF for base velocity
      • Real-time policy deployment via Unitree SDK2 (CycloneDDS)

      <strong>Results:</strong> Successfully deployed kicks, jumps, expressive gestures,
      and stable locomotion gaits on physical hardware with <5% performance degradation.
    `,
    tags: ['Sim2Real', 'IsaacGym', 'PPO', 'Motion Retargeting', 'PyTorch', 'ONNX'],
    images: [],
    videos: [
      { src: 'assets/projects/asap/img1.gif', title: 'Trained Action 1' },
      { src: 'assets/projects/asap/img2.gif', title: 'Trained Action 2' },
      { src: 'assets/projects/asap/img3.gif', title: 'Trained Action 3' },
      { src: 'assets/projects/asap/img4.gif', title: 'Trained Action 4' },
      { src: 'assets/projects/asap/video1.mp4', title: 'Initial Training Stage' },
      { src: 'assets/projects/asap/video2.mp4', title: 'Intermediate' }
    ],
    thumbnail: 'assets/projects/asap/img1.gif',
    hideGallery: true,
    technicalDeepDive: {
      sections: [
        {
          title: "Massively Parallel Simulation",
          content: `
            <p>Training leverages <strong>NVIDIA Isaac Gym</strong> to simulate 4096 robot instances in parallel on a single GPU. Each environment runs independently with its own physics state, terrain, and randomization parameters.</p>
            <div class="code-block"><code>IsaacGym Config:
  num_envs: 4096
  sim_device: cuda:0
  physics_engine: PhysX (GPU)
  dt: 0.005s (200Hz physics)
  substeps: 2
  decimation: 4 (50Hz policy)</code></div>
            <p>GPU-accelerated PhysX handles rigid body dynamics, contact resolution, and joint constraints. Batch tensor operations via PyTorch eliminate CPU-GPU data transfer bottlenecks.</p>
          `
        },
        {
          title: "PPO with Asymmetric Actor-Critic",
          content: `
            <p>Policy optimization uses <strong>Proximal Policy Optimization (PPO)</strong> with Generalized Advantage Estimation (GAE). The architecture employs asymmetric observations:</p>
            <div class="formula">Actor obs (47-dim): joint pos/vel, base orientation, commands, phase
Critic obs (50-dim): Actor obs + privileged info (contact forces, terrain height)</div>
            <p>The critic has access to ground-truth simulation state unavailable on real hardware, enabling better value estimation during training while the actor learns from realistic observations only.</p>
            <div class="code-block"><code>PPO Hyperparameters:
  γ (discount): 0.99
  λ (GAE): 0.95
  clip_range: 0.2
  entropy_coef: 0.01
  learning_rate: 1e-3 → 1e-5 (cosine decay)
  batch_size: 4096 * 24 steps</code></div>
          `
        },
        {
          title: "Domain Randomization",
          content: `
            <p>To bridge the sim-to-real gap, extensive randomization is applied during training:</p>
            <p><strong>Dynamics:</strong> Base mass (±3kg), friction coefficients (0.1-1.25), motor strength scaling (±10%), joint damping variation.</p>
            <p><strong>Observations:</strong> Gaussian noise on joint encoders, IMU bias drift, latency injection (0-40ms).</p>
            <p><strong>External Forces:</strong> Random pushes every 5s with up to 1.5 m/s velocity perturbation to train recovery behaviors.</p>
            <p>The policy learns a robust manifold that generalizes across the randomization distribution, making it resilient to real-world modeling errors.</p>
          `
        },
        {
          title: "Reward Engineering",
          content: `
            <p>Multi-objective reward function balances task performance with motion quality:</p>
            <div class="code-block"><code>Reward Components:
  tracking_lin_vel: 1.0   # Match commanded velocity
  tracking_ang_vel: 0.5   # Match commanded yaw rate
  lin_vel_z: -2.0         # Penalize vertical bounce
  orientation: -1.0       # Keep torso upright
  base_height: -10.0      # Maintain target height (0.78m)
  dof_acc: -2.5e-7        # Smooth joint motion
  action_rate: -0.01      # Penalize jerky commands
  alive: 0.15             # Survival bonus</code></div>
            <p>Curriculum learning progressively increases terrain difficulty and command ranges as the policy improves.</p>
          `
        },
        {
          title: "Sim2Real Deployment",
          content: `
            <p>Trained policies export to ONNX format for deployment on the robot's onboard computer. The inference pipeline runs at 500Hz:</p>
            <div class="code-block"><code>Real Robot Loop (2ms):
  1. Read joint encoders + IMU → state vector
  2. ONNX Runtime inference → action (12-dim)
  3. action_scale * action + default_angles → target positions
  4. PD controller → torque commands
  5. Send to motor drivers via CycloneDDS</code></div>
            <p><strong>Actuator Network:</strong> A learned model compensates for motor dynamics (friction, backlash, thermal effects) not captured in simulation.</p>
          `
        }
      ],
      metrics: [
        { value: "4096", label: "Parallel Envs" },
        { value: "500Hz", label: "Control Rate" },
        { value: "< 5%", label: "Sim2Real Gap" },
        { value: "2ms", label: "Inference Time" }
      ],
      references: [
        { title: "Isaac Gym Paper", url: "https://arxiv.org/abs/2108.10470" },
        { title: "PPO Algorithm", url: "https://arxiv.org/abs/1707.06347" },
        { title: "Unitree RL Gym", url: "https://github.com/unitreerobotics/unitree_rl_gym" },
        { title: "ASAP Framework", url: "https://agility.csail.mit.edu/asap/" }
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
      Designed and built multiple unmanned aerial vehicles from scratch, including fixed-wing aircraft
      and multi-rotor quadcopters. Projects span from concept design to successful flight testing.

      <strong>Fixed-Wing Aircraft:</strong>
      • Custom foam-board airframe with optimized aerodynamic profile
      • SG90 servo actuators for aileron, elevator, and rudder control surfaces
      • 2212 brushless outrunner motor with 30A ESC for propulsion
      • 2.4GHz RC transmitter with 6-channel receiver integration
      • Successful flight tests with stable handling characteristics

      <strong>FPV Racing Quadcopter:</strong>
      • Carbon fiber frame (250mm class) for high strength-to-weight ratio
      • F4 flight controller running Betaflight firmware with PID tuning
      • GPS module for position hold and return-to-home functionality
      • 4S LiPo battery with XT60 connector for high discharge rates
      • FPV camera system with 5.8GHz video transmitter

      <strong>Technical Stack:</strong> Betaflight, Arduino, ESC calibration, PID tuning,
      LiPo battery management, radio protocol configuration, thrust-to-weight optimization.
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
    thumbnail: 'assets/projects/rc-uav/img5.jpg'
  },
  {
    id: 'arm-control-gui',
    title: 'Robot Arm Control GUI with MuJoCo',
    shortTitle: 'Arm Control GUI',
    category: 'Robotics Integration',
    problem: 'Need unified interface to test arm policies in simulation before deploying to real robot',
    outcome: 'Hot-swappable sim/real backend with 30fps MuJoCo viz and 2ms policy inference',
    oneLiner: 'PyQt6 interface with MuJoCo visualization and real-time arm control',
    description: `
      Real-time control interface for bipedal humanoid robot arm manipulation. Integrates MuJoCo physics
      simulation, RL-based policy inference, and live camera feedback in a unified PyQt6 application.

      <strong>Architecture:</strong>
      • PyQt6 GUI with QThread-based async orchestration
      • MuJoCo renderer at 30Hz with double-buffered QImage pipeline
      • ROS2 bridge for robot state subscription (500Hz joint feedback)
      • Modular plant abstraction: simulation ↔ real hardware hot-swap

      <strong>Control Modes:</strong>
      • Direct joint control: 14-DOF arms (7 per arm) with real-time slider feedback
      • Policy inference: PPO-trained ONNX model for end-effector tracking
      • Hand target IK: 6-DOF pose input → joint trajectory via damped least-squares
      • Gesture playback: Pre-recorded motion sequences with interpolation

      <strong>Simulation Stack:</strong>
      • MuJoCo 3.x with 60-DOF humanoid model (floating base + 29 actuated joints)
      • Thread-safe qpos access with RLock for render/control synchronization
      • Camera orbit controls: azimuth/elevation drag, scroll zoom, pan

      <strong>Policy Controller:</strong>
      • Stable-Baselines3 PPO with MLP policy (256×256 hidden layers)
      • Observation: joint positions + velocities + target pose (41-dim)
      • Action: delta joint positions normalized to [-1, 1]
      • Inference: ~2ms latency on CPU (ONNX Runtime)

      <strong>Integration:</strong> Unitree SDK2 via CycloneDDS for real robot deployment,
      RealSense D435 depth camera for visual feedback, config-driven joint limits and gains.
    `,
    tags: ['PyQt6', 'MuJoCo', 'PPO', 'ROS2', 'IK', 'Unitree SDK'],
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
          title: "PyQt6 Application Architecture",
          content: `
            <p>Multi-threaded GUI application with clear separation between UI, control logic, and hardware interfaces:</p>
            <div class="code-block"><code>Architecture:
MainWindow (PyQt6)
├── MuJoCoWidget      # 3D visualization
├── RobotPanel        # Joint sliders & status
├── CameraWidget      # RealSense feed
└── Orchestrator      # Control logic (QThread)
    ├── SimPlantAdapter   # MuJoCo backend
    └── RealPlantAdapter  # Hardware backend

Threading Model:
  - Main thread: UI rendering
  - Control loop: 50Hz QThread
  - Camera feed: Async callback
  - MuJoCo render: Timer-based (30Hz)</code></div>
          `
        },
        {
          title: "MuJoCo Visualization Pipeline",
          content: `
            <p>Real-time physics visualization using MuJoCo's offscreen rendering:</p>
            <div class="code-block"><code>Render Pipeline (30Hz):
1. Acquire qpos_lock (thread safety)
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
  - Last-good-frame fallback on error</code></div>
            <p>Camera controls: left-drag (orbit), right-drag (pan), scroll (zoom), 'R' key (reset view).</p>
          `
        },
        {
          title: "Plant Abstraction Layer",
          content: `
            <p>Unified interface allows hot-swapping between simulation and real hardware:</p>
            <div class="code-block"><code>class PlantAdapter(ABC):
    @abstractmethod
    def get_joint_positions(self) -> Dict[int, float]

    @abstractmethod
    def set_joint_positions(self, positions: Dict[int, float])

    @abstractmethod
    def get_end_effector_pose(self, arm: str) -> Tuple[np.ndarray, np.ndarray]

Implementations:
  SimPlantAdapter:  MuJoCo data.qpos read/write
  RealPlantAdapter: ROS2 joint_states + SDK commands</code></div>
            <p>Switching plant triggers state transfer - current joint positions seed the new plant for seamless transition.</p>
          `
        },
        {
          title: "Policy Controller Integration",
          content: `
            <p>PPO-trained reaching policies loaded via ONNX for end-effector control:</p>
            <div class="code-block"><code>Policy Inference Loop:
1. target_pose = camera_tracker.get_target()
2. current_state = plant.get_observation()
   - joint_pos (7): normalized
   - joint_vel (7): scaled
   - ee_pos (3): current end-effector
   - target_pos (3): goal position

3. obs = concatenate([current_state, target_pose])
4. action = onnx_session.run(obs)  # 7-dim
5. target_joints = action * scale + default
6. plant.set_joint_positions(target_joints)

Inference: ~2ms CPU (ONNX Runtime)</code></div>
          `
        },
        {
          title: "Thread Safety & Synchronization",
          content: `
            <p>Critical sections protected by <code>threading.RLock</code> to prevent race conditions:</p>
            <div class="code-block"><code>Protected Resources:
  qpos_lock: MuJoCo data.qpos array
    - Read: get_joint_positions()
    - Write: set_joint_positions()
    - Render: update_scene() + render()

Lock Acquisition Order:
  1. Control loop acquires for state read
  2. Sets new targets
  3. Releases before sleep
  4. Render acquires during paint event

RLock (reentrant): Same thread can acquire
multiple times without deadlock.</code></div>
            <p>Non-blocking render: If lock held, sets <code>_pending_render = True</code> and returns immediately.</p>
          `
        }
      ],
      metrics: [
        { value: "30 FPS", label: "Render Rate" },
        { value: "50Hz", label: "Control Loop" },
        { value: "60-DOF", label: "Robot Model" },
        { value: "2ms", label: "Policy Inference" }
      ],
      references: [
        { title: "MuJoCo Python", url: "https://mujoco.readthedocs.io/en/stable/python.html" },
        { title: "PyQt6 Threading", url: "https://doc.qt.io/qtforpython-6/overviews/thread-basics.html" },
        { title: "ONNX Runtime", url: "https://onnxruntime.ai/docs/" }
      ]
    }
  },
  {
    id: 'hand-target',
    title: '6-DOF Object Pose Estimation for Robot Manipulation',
    shortTitle: 'Pose Estimation',
    category: 'Computer Vision',
    problem: 'Robot arms need stable, accurate 6-DOF target poses for manipulation tasks',
    outcome: 'Sub-centimeter position accuracy at 1m range with 30fps processing',
    oneLiner: 'RealSense RGB-D with ArUco markers and quaternion averaging',
    description: `
      Real-time 3D pose estimation pipeline for robot manipulation tasks. Combines ArUco marker
      tracking with point cloud processing to provide stable 6-DOF target poses for arm control.

      <strong>Detection Pipeline:</strong>
      • Intel RealSense D435 RGB-D streaming (640×480 @ 30fps)
      • Aligned depth + color frames for accurate 3D reconstruction
      • Multi-method detection: HSV color segmentation, contour analysis, ArUco markers
      • Statistical outlier removal + voxel downsampling for point cloud cleanup

      <strong>Pose Estimation:</strong>
      • Position: Point cloud centroid in camera frame (meters)
      • Orientation: PCA-based principal axis extraction (rotation matrix)
      • Multi-marker fusion: Quaternion averaging for robust rotation
      • Temporal smoothing: Exponential moving average for robot-friendly trajectories

      <strong>ArUco Marker System:</strong>
      • 4×4 dictionary (50 markers), 15mm marker size
      • Strip layout for cylindrical object wrapping
      • Custom A4 printable marker strip generator
      • Sub-centimeter position accuracy at 1m range

      <strong>Integration:</strong>
      • TCP socket streaming to robot controller
      • JSON pose output: position, euler angles, confidence
      • Headless mode for embedded deployment
      • 50+ FPS processing on laptop GPU
    `,
    tags: ['RealSense', 'ArUco', 'Point Cloud', 'PCA', 'OpenCV', 'Python'],
    images: [],
    videos: [],
    thumbnail: null,
    hideGallery: true,
    technicalDeepDive: {
      sections: [
        {
          title: "RGB-D Stream Processing",
          content: `
            <p>Intel RealSense D435 provides synchronized RGB (640×480) and depth frames at 30fps. The pipeline uses <strong>aligned depth</strong> - depth pixels are reprojected to match RGB pixel coordinates using camera intrinsics.</p>
            <div class="code-block"><code>RealSense Config:
  color: 640x480 @ 30fps (RGB8)
  depth: 640x480 @ 30fps (Z16)
  depth_units: 0.001m (1mm precision)
  align_to: color

Camera Intrinsics (D435):
  fx, fy: 607.0, 606.8
  cx, cy: 320.0, 240.0</code></div>
            <p>Frames arrive via <code>pyrealsense2</code> in a callback-based pipeline. Temporal filtering with persistence smooths depth noise.</p>
          `
        },
        {
          title: "Multi-Marker ArUco Detection",
          content: `
            <p>For robust pose estimation on cylindrical objects, multiple 15mm ArUco markers (4×4 dictionary) are arranged in a strip pattern. The detection pipeline:</p>
            <div class="code-block"><code>1. cv2.aruco.detectMarkers() → corners, ids
2. For each marker:
   - cv2.aruco.estimatePoseSingleMarkers() → rvec, tvec
   - Convert rvec to rotation matrix (Rodrigues)
   - Convert rotation to quaternion
3. Position: Centroid of all marker tvecs
4. Rotation: Quaternion averaging (SLERP-based)</code></div>
            <p>Quaternion averaging prevents gimbal lock and handles the discontinuity when markers wrap around the cylinder. Outlier rejection filters markers with inconsistent poses.</p>
          `
        },
        {
          title: "Temporal Smoothing for Robot Control",
          content: `
            <p>Raw pose estimates are noisy due to marker detection jitter. Heavy smoothing makes the output suitable for robot arm control:</p>
            <div class="formula">Position: EMA with α=0.15, history=30 frames
Rotation: Quaternion SLERP, history=25 frames
Output rate: 30Hz (matches camera)</div>
            <p><strong>Quaternion Flip Prevention:</strong> Quaternions q and -q represent the same rotation. The algorithm tracks the previous quaternion and flips the sign if the dot product is negative, ensuring smooth interpolation.</p>
            <div class="code-block"><code>if np.dot(q_new, q_prev) < 0:
    q_new = -q_new  # Flip to shorter path</code></div>
          `
        },
        {
          title: "Point Cloud Pose Estimation",
          content: `
            <p>Alternative method using depth-based 3D reconstruction for textureless objects:</p>
            <p><strong>1. Segmentation:</strong> HSV color thresholding or largest contour detection to create binary mask.</p>
            <p><strong>2. Deprojection:</strong> Masked pixels → 3D points using camera intrinsics: <code>X = (u - cx) * Z / fx</code></p>
            <p><strong>3. Filtering:</strong> Statistical outlier removal (mean + 2σ), voxel downsampling (5mm grid).</p>
            <p><strong>4. PCA:</strong> Principal Component Analysis extracts orientation - largest eigenvector = object's primary axis (stem direction for bouquet).</p>
            <div class="formula">Position = centroid(point_cloud)
Orientation = eigenvectors(covariance_matrix)</div>
          `
        }
      ],
      metrics: [
        { value: "30 FPS", label: "Processing Rate" },
        { value: "< 1cm", label: "Position Accuracy" },
        { value: "15mm", label: "Marker Size" },
        { value: "1m", label: "Working Range" }
      ],
      references: [
        { title: "ArUco Library", url: "https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html" },
        { title: "RealSense SDK", url: "https://github.com/IntelRealSense/librealsense" },
        { title: "Quaternion Averaging", url: "https://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf" }
      ]
    }
  },
  {
    id: 'g1-isaac-training',
    title: 'Legged Robot RL Training in Isaac Gym',
    shortTitle: 'Isaac Gym RL',
    category: 'Reinforcement Learning',
    problem: 'Training robust locomotion policies that generalize across terrain and disturbances',
    outcome: 'Stable walking at 1.5 m/s with automatic recovery from perturbations',
    oneLiner: 'Custom RL environment with terrain curriculum and domain randomization',
    description: `
      Custom reinforcement learning environment for training locomotion and manipulation policies
      on Unitree G1 humanoid robot using NVIDIA Isaac Gym massively parallel simulation.

      <strong>Environment Setup:</strong>
      • Isaac Gym with 4096 parallel environments on single GPU
      • Custom G1 URDF with accurate joint limits and collision meshes
      • Terrain curriculum: flat → slopes → stairs → rough
      • Domain randomization: mass, friction, motor strength, latency

      <strong>Policy Training:</strong>
      • PPO with GAE (λ=0.95) and clipped surrogate objective
      • Asymmetric actor-critic: privileged critic with ground truth state
      • Action space: target joint positions (delta from default pose)
      • Observation: joint pos/vel, base orientation, commands, phase

      <strong>Reward Design:</strong>
      • Tracking rewards: linear/angular velocity matching
      • Regularization: action smoothness, joint limits, torque penalty
      • Survival bonus with early termination on fall
      • Curriculum-based difficulty scaling

      <strong>Sim2Real:</strong>
      • ONNX export for 500Hz inference on robot
      • Actuator network for motor dynamics compensation
      • EKF-based state estimation for base velocity
      • Tested gaits: walk, trot, pace, bound
    `,
    tags: ['Isaac Gym', 'PPO', 'Sim2Real', 'URDF', 'PyTorch', 'ONNX'],
    images: [],
    videos: [],
    thumbnail: null,
    hideGallery: true,
    github: 'https://github.com/bhaveshbakshi633/G1_Isaac_training',
    technicalDeepDive: {
      sections: [
        {
          title: "G1 Robot Model Configuration",
          content: `
            <p>The G1 humanoid is modeled as a 12-DOF system for locomotion training (legs only, excluding arms). URDF defines the kinematic chain, collision meshes, and joint limits.</p>
            <div class="code-block"><code>Joint Configuration (12 DOF):
Left Leg:  hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll
Right Leg: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll

Default Pose (radians):
  hip_pitch: -0.1
  knee: 0.3
  ankle_pitch: -0.2
  others: 0.0

Standing Height: 0.8m</code></div>
            <p>Joint PD gains are tuned per-joint: hips use Kp=100, knees Kp=150, ankles Kp=40. Action scale of 0.25 limits maximum deviation from default pose.</p>
          `
        },
        {
          title: "Observation & Action Spaces",
          content: `
            <p>The policy observes a 47-dimensional state vector and outputs 12-dimensional joint position targets:</p>
            <div class="code-block"><code>Observations (47-dim):
  - Joint positions (12): normalized by limits
  - Joint velocities (12): scaled
  - Base angular velocity (3): from IMU
  - Projected gravity (3): orientation reference
  - Commands (3): vx, vy, yaw_rate
  - Last actions (12): for smoothness
  - Phase variables (2): sin/cos gait phase

Privileged Obs (critic only, +3):
  - Contact forces
  - Terrain height samples

Actions (12-dim):
  target_pos = action * 0.25 + default_angles</code></div>
          `
        },
        {
          title: "Reward Function Design",
          content: `
            <p>Multi-objective reward balances velocity tracking, stability, and energy efficiency:</p>
            <div class="code-block"><code>Positive Rewards:
  tracking_lin_vel (1.0): exp(-||v_cmd - v_actual||²)
  tracking_ang_vel (0.5): exp(-|ω_cmd - ω_actual|²)
  alive (0.15): survival bonus per step
  contact (0.18): reward proper foot contacts

Penalty Terms:
  lin_vel_z (-2.0): vertical velocity = bouncing
  orientation (-1.0): torso tilt from upright
  base_height (-10.0): deviation from 0.78m target
  dof_acc (-2.5e-7): joint acceleration smoothness
  action_rate (-0.01): jerky command changes
  hip_pos (-1.0): discourage splayed hips</code></div>
            <p>Termination occurs on pelvis contact (fall) or after 1000 steps. Early termination accelerates learning of stable behaviors.</p>
          `
        },
        {
          title: "Training Pipeline",
          content: `
            <p>Training runs for 1000-3000 iterations using the PPO implementation from <code>rsl_rl</code>:</p>
            <div class="code-block"><code>python legged_gym/scripts/train.py --task=g1 \\
  --num_envs=4096 \\
  --headless \\
  --max_iterations=2000

Output: logs/g1/&lt;timestamp&gt;/model_&lt;iter&gt;.pt</code></div>
            <p><strong>Curriculum:</strong> Command ranges expand as policy improves. Initial: vx∈[-0.5,0.5]. Final: vx∈[-1.5,1.5] m/s.</p>
            <p><strong>Checkpointing:</strong> Models saved every 50 iterations. Best model selected by average reward over 100 episodes.</p>
          `
        },
        {
          title: "Sim2Sim Validation (MuJoCo)",
          content: `
            <p>Before real deployment, policies are validated in MuJoCo to ensure they're not overfitting to Isaac Gym's physics:</p>
            <div class="code-block"><code>python deploy/deploy_mujoco/deploy_mujoco.py g1

# Loads exported policy and runs in MuJoCo viewer
# Checks for:
# - Gait stability on flat ground
# - Velocity tracking accuracy
# - Recovery from perturbations</code></div>
            <p>Policies that fail Sim2Sim typically have issues with contact dynamics or exploit PhysX-specific artifacts.</p>
          `
        }
      ],
      metrics: [
        { value: "12 DOF", label: "Action Space" },
        { value: "47-dim", label: "Observation" },
        { value: "~2hr", label: "Training Time" },
        { value: "1.5 m/s", label: "Max Velocity" }
      ],
      references: [
        { title: "Unitree RL Gym", url: "https://github.com/unitreerobotics/unitree_rl_gym" },
        { title: "RSL RL Library", url: "https://github.com/leggedrobotics/rsl_rl" },
        { title: "Legged Gym", url: "https://github.com/leggedrobotics/legged_gym" }
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
      Deep learning-based hand gesture recognition for human-robot interaction. Enables intuitive
      control of robotic systems through natural hand gestures captured via RGB camera.

      <strong>Detection Pipeline:</strong>
      • MediaPipe Hands for 21-point hand landmark detection
      • Real-time tracking at 30+ FPS on CPU
      • Multi-hand support with handedness classification
      • Robust to partial occlusion and varying lighting

      <strong>Gesture Classification:</strong>
      • Custom gesture vocabulary: point, grab, release, wave, thumbs up/down
      • Feature extraction: finger angles, palm orientation, landmark distances
      • Lightweight MLP classifier trained on collected dataset
      • Temporal smoothing to filter spurious predictions

      <strong>Robot Integration:</strong>
      • ROS2 topic publishing for gesture commands
      • Configurable gesture-to-action mapping
      • Cooldown timer to prevent repeated triggers
      • Visual feedback overlay on camera stream

      <strong>Applications:</strong>
      • Humanoid robot control via gestures
      • Touchless UI interaction
      • Sign language recognition prototype
      • Safety stop gesture for emergency halt
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
      Full-stack mobile application for booking and managing household maintenance services.
      Connects users with verified service providers for plumbing, electrical, cleaning, and more.

      <strong>User Features:</strong>
      • Service catalog with categories and search
      • Real-time booking with slot availability
      • Live tracking of service provider location
      • In-app cost estimation before booking
      • Rating and review system

      <strong>Service Provider Features:</strong>
      • Job request notifications and acceptance
      • Route optimization for multiple bookings
      • Earnings dashboard and payout tracking
      • Profile verification and badge system

      <strong>Technical Stack:</strong>
      • Frontend: React Native (iOS + Android)
      • Backend: Node.js + Express REST API
      • Database: MongoDB with geospatial indexing
      • Real-time: Socket.io for live tracking
      • Maps: Google Maps API for routing and ETA

      <strong>Key Metrics:</strong>
      • Sub-second booking confirmation
      • 95%+ location accuracy for tracking
      • Offline-first architecture with sync queue
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
    title: 'XR Teleoperation for Robot Arms',
    shortTitle: 'XR Teleop',
    category: 'Robotics Integration',
    problem: 'Intuitive robot arm control through immersive hand tracking',
    outcome: 'Sub-50ms end-to-end latency with 6-DOF hand-to-robot mapping',
    oneLiner: 'Quest 3 hand tracking to robot IK via CycloneDDS',
    description: `
      Immersive teleoperation system enabling real-time control of bipedal humanoid robots through
      Extended Reality (XR) devices. Operator hand/head tracking mapped to robot end-effectors
      with sub-100ms latency for intuitive manipulation tasks.

      <strong>XR Integration:</strong>
      • Meta Quest 3 / Apple Vision Pro / PICO 4 Ultra support
      • 6-DOF hand tracking via controller or hand-tracking SDK
      • Head pose streaming for robot head/torso orientation
      • Passthrough AR mode for immersive operator view

      <strong>Robot Control Pipeline:</strong>
      • Unitree SDK2 (CycloneDDS) for low-latency command streaming
      • Inverse kinematics solver for Cartesian → joint space mapping
      • Configurable workspace scaling and dead-zone filtering
      • Support for G1 (23/29 DOF), H1, H1_2 humanoid variants

      <strong>Dexterous Manipulation:</strong>
      • Multi-finger hand support: Dex3-1, Inspire, BrainCo hands
      • Per-finger joint mapping from XR hand skeleton
      • Grasp detection and force feedback visualization
      • Custom wrist/ring mount for hand tracker attachment

      <strong>Hardware Setup:</strong>
      • RealSense head-mounted camera for operator FPV stream
      • WiFi 6 router for high-bandwidth XR ↔ robot communication
      • ArUco marker-based workspace calibration
      • Custom 3D-printed mounts for camera and trackers
    `,
    tags: ['XR', 'Quest 3', 'Teleoperation', 'IK', 'CycloneDDS', 'RealSense'],
    images: [],
    videos: [],
    thumbnail: null,
    hideGallery: true,
    technicalDeepDive: {
      sections: [
        {
          title: "XR Device Integration",
          content: `
            <p>The system supports multiple XR headsets through their respective SDKs:</p>
            <div class="code-block"><code>Supported Devices:
  - Meta Quest 3: OpenXR + Hand Tracking API
  - Apple Vision Pro: ARKit + visionOS SDK
  - PICO 4 Ultra: PICO SDK (Enterprise)

Tracking Data (per hand):
  - Wrist pose: position (m) + quaternion
  - 21 finger joints: relative transforms
  - Pinch strength: 0-1 per finger
  - Hand confidence: tracking quality</code></div>
            <p>Quest 3 hand tracking runs at 60Hz with sub-centimeter accuracy. Controller fallback available for precise manipulation.</p>
          `
        },
        {
          title: "Coordinate Frame Transforms",
          content: `
            <p>XR tracking coordinates must be transformed to robot frame:</p>
            <div class="formula">T_robot = T_calibration × T_scale × T_xr_hand

Where:
  T_calibration: Workspace alignment (set via ArUco)
  T_scale: Motion scaling (2:1 to 5:1)
  T_xr_hand: Raw XR tracking pose</div>
            <p><strong>Workspace Mapping:</strong> Operator's reachable volume maps to robot's workspace. Configurable dead-zone at boundaries prevents accidental limit violations.</p>
          `
        },
        {
          title: "Inverse Kinematics Solver",
          content: `
            <p>Cartesian end-effector targets converted to joint positions via damped least-squares IK:</p>
            <div class="code-block"><code>IK Algorithm (Damped Least Squares):
  Δq = J^T × (J × J^T + λ²I)^(-1) × Δx

  J: Jacobian matrix (6×7 for 7-DOF arm)
  λ: Damping factor (0.1) - prevents singularities
  Δx: Task-space error (position + orientation)
  Δq: Joint-space update

Iteration: 5-10 steps to converge
Rate: 100Hz IK solve → 50Hz command output</code></div>
            <p>Null-space optimization biases solutions toward comfortable joint configurations (elbow down, wrist neutral).</p>
          `
        },
        {
          title: "Dexterous Hand Mapping",
          content: `
            <p>XR finger skeleton maps to robot dexterous hands (Dex3-1, Inspire):</p>
            <div class="code-block"><code>Finger Mapping (per finger):
  XR MCP angle → Robot proximal joint
  XR PIP angle → Robot intermediate joint
  XR DIP angle → Robot distal joint

Special Handling:
  - Thumb opposition: Separate abduction channel
  - Grasp detection: All fingertips within threshold
  - Force feedback: Vibration on contact (Quest)</code></div>
            <p>Retargeting accounts for different hand proportions between human operator and robot end-effector.</p>
          `
        },
        {
          title: "Communication Pipeline",
          content: `
            <p>Low-latency data flow from XR headset to robot actuators:</p>
            <div class="code-block"><code>XR App (Quest)
    ↓ UDP (WiFi 6, 5GHz)
Control PC (Python)
    ↓ CycloneDDS (Unitree SDK2)
Robot (G1/H1)
    ↓ EtherCAT
Motor Drivers

Latency Budget:
  XR tracking: 10ms
  Network: 5-20ms
  IK solve: 5ms
  Robot control: 2ms
  Total: < 50ms end-to-end</code></div>
            <p>UDP preferred over TCP for lower latency. Packet loss handled by predictor extrapolating from velocity.</p>
          `
        }
      ],
      metrics: [
        { value: "< 50ms", label: "End-to-End Latency" },
        { value: "60Hz", label: "Tracking Rate" },
        { value: "7-DOF", label: "Per Arm" },
        { value: "21", label: "Finger Joints" }
      ],
      references: [
        { title: "OpenXR Spec", url: "https://www.khronos.org/openxr/" },
        { title: "Quest Hand Tracking", url: "https://developer.oculus.com/documentation/native/android/mobile-hand-tracking/" },
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
      Complete mechanical design and structural analysis of a high-performance ATV chassis
      for the BAJA SAE competition. Optimized for durability, weight, and manufacturability.

      <strong>Structural Design:</strong>
      • Tubular space frame chassis with AISI 4130 chromoly steel tubing
      • Roll cage geometry optimized for driver safety (FIA standards)
      • Triangulated members for torsional rigidity (>3000 Nm/deg)
      • Suspension mounting points with adjustable geometry

      <strong>FEA Analysis (Ansys):</strong>
      • Static analysis: 2.5G bump, 1.5G braking, 1.2G cornering loads
      • Dynamic impact simulation for rollover scenarios
      • Fatigue life prediction (>10⁶ cycles target)
      • Stress concentrations identified and reinforced
      • Final FOS: 2.1 (critical members), 3.0+ (non-critical)

      <strong>Manufacturing:</strong>
      • TIG welding with full-penetration joints
      • Jig-based assembly for dimensional accuracy (±1mm)
      • 25% cost reduction through DFM optimization
      • Weight: 38kg bare frame (target: <40kg)

      <strong>Tools:</strong> SolidWorks (CAD), Ansys Workbench (FEA), AutoCAD (drawings).
    `,
    tags: ['SolidWorks', 'FEA', 'Ansys', 'BAJA SAE', 'Manufacturing'],
    images: ['assets/projects/atv/img1.jpg', 'assets/projects/atv/img2.jpg', 'assets/projects/atv/img3.jpg', 'assets/projects/atv/img4.png'],
    video: 'assets/projects/atv/video.mp4',
    thumbnail: 'assets/projects/atv/img1.jpg'
  },
  {
    id: 'surgical-robot',
    title: 'Surgical Robotics Assistant',
    shortTitle: 'Surgical Robot',
    category: 'Medical Robotics',
    problem: 'Precision manipulation for minimally invasive surgical procedures',
    outcome: 'Sub-millimeter accuracy with redundant safety interlocks',
    oneLiner: 'Precision manipulator with redundant safety systems',
    description: `
      Developed modular robotic systems for precision surgical applications. The project focused on
      creating reliable, accurate robotic assistance for minimally invasive procedures.

      Features include advanced motion control with sub-millimeter precision, real-time feedback
      systems for surgeon guidance, and intuitive operator interfaces. The system was designed with
      safety as the primary concern, incorporating multiple redundant safety mechanisms.
    `,
    tags: ['Robotics', 'Medical', 'Control Systems', 'Python'],
    images: ['assets/projects/surgical-robot/img1.jpg', 'assets/projects/surgical-robot/img2.jpg'],
    video: 'assets/projects/surgical-robot/video.mp4',
    thumbnail: 'assets/projects/surgical-robot/img1.jpg'
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
    tags: ['Automation', 'Computer Vision', 'IoT', 'Arduino'],
    images: ['assets/projects/chess-board/img1.jpg', 'assets/projects/chess-board/img2.jpg', 'assets/projects/chess-board/img3.jpg'],
    video: 'assets/projects/chess-board/video.mp4',
    thumbnail: 'assets/projects/chess-board/img1.jpg'
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
    tags: ['Arduino', 'IoT', 'Home Automation', 'ESP32'],
    images: ['assets/projects/ir-remote/img1.jpg', 'assets/projects/ir-remote/img2.jpg'],
    video: 'assets/projects/ir-remote/video.mp4',
    thumbnail: 'assets/projects/ir-remote/img1.jpg'
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
    'Aerial Systems',
    'Medical Robotics',
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
            <a href="#" class="project-card ${project.flagship ? 'flagship' : ''}" data-project-id="${project.id}">
              ${project.flagship ? '<span class="flagship-badge">Featured</span>' : ''}
              <div class="project-card-content">
                <h4 class="project-card-title">${project.shortTitle}</h4>
                <p class="project-card-problem"><span class="label">Problem:</span> ${project.problem || ''}</p>
                <p class="project-card-outcome"><span class="label">Outcome:</span> ${project.outcome || ''}</p>
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

    // Faster slideshow interval (2.5s default, images only - videos play continuously)
    const slideInterval = project.slideInterval || 2500;

    // Only images for slideshow (videos play on loop, not rotated)
    // For RC-UAV: only show video, no images in slideshow
    const mediaItems = [];
    if (hasVideo) {
      // Video projects: just show video (loops continuously)
      mediaItems.push({ type: 'video', src: project.videos[0].src });
    } else if (hasImages) {
      // Image-only projects: create slideshow
      project.images.slice(0, 5).forEach(img => mediaItems.push({ type: 'image', src: img }));
    }

    const showSlideshow = !hasVideo && mediaItems.length > 1;

    return `
      <article class="showcase-card ${isOngoing ? 'ongoing' : ''}" data-project-id="${project.id}" tabindex="0" role="button" aria-label="View ${project.shortTitle || project.title} project">
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
    }, { threshold: 0.3 });

    carousel.querySelectorAll('.showcase-video').forEach(video => {
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
      <div class="project-detail-tags">
        ${project.tags.map(tag => `<span class="project-tag">${tag}</span>`).join('')}
      </div>
    </div>

    <div class="project-detail-description">
      ${project.description.trim().split('\n\n').map(p => {
        const trimmed = p.trim();
        // Check if paragraph contains bullet points
        if (trimmed.includes('•')) {
          const lines = trimmed.split('\n').map(l => l.trim());
          const items = lines.filter(l => l.startsWith('•')).map(l => `<li>${l.substring(1).trim()}</li>`).join('');
          const intro = lines.filter(l => !l.startsWith('•')).join(' ');
          return intro ? `<p>${intro}</p><ul class="tech-list">${items}</ul>` : `<ul class="tech-list">${items}</ul>`;
        }
        return `<p>${trimmed}</p>`;
      }).join('')}
    </div>

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

          ${project.technicalDeepDive.metrics ? `
            <div class="metrics-grid">
              ${project.technicalDeepDive.metrics.map(m => `
                <div class="metric-card">
                  <div class="metric-value">${m.value}</div>
                  <div class="metric-label">${m.label}</div>
                </div>
              `).join('')}
            </div>
          ` : ''}

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
// CAROUSEL INTERACTION (Pause on hover/touch)
// =============================================
function initCarouselInteraction() {
  const carousel = document.getElementById('showcaseCarousel');
  if (!carousel) return;

  let hasInteracted = false;

  // Pause on hover
  carousel.addEventListener('mouseenter', () => {
    carousel.classList.add('paused');
  });

  carousel.addEventListener('mouseleave', () => {
    if (!hasInteracted) {
      carousel.classList.remove('paused');
    }
  });

  // Pause permanently after first click/interaction
  carousel.addEventListener('click', () => {
    hasInteracted = true;
    carousel.classList.add('paused');
  });

  // Touch support
  carousel.addEventListener('touchstart', () => {
    carousel.classList.add('paused');
  }, { passive: true });
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
