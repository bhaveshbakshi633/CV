// ============================================================
// Cart-Pole Balancer — REINFORCE policy gradient se pole balance karna seekhega
// Classic control problem hai ye — RL ka hello world samajh le
// Poore bugs fix kiye: visibility, manual mode, tanh, DPR, etc.
// ============================================================

// yahi main entry point hai — container dhundho, canvas banao, training shuru karo
export function initCartPole() {
  const container = document.getElementById('cartpoleContainer');
  if (!container) {
    console.warn('cartpoleContainer nahi mila bhai, cart-pole skip kar rahe hain');
    return;
  }

  // --- Physics constants ---
  // standard cart-pole parameters — OpenAI Gym se inspired hai
  const GRAVITY = 9.81;
  const CART_MASS = 1.0;
  const POLE_MASS = 0.1;
  const TOTAL_MASS = CART_MASS + POLE_MASS;
  const POLE_HALF_LEN = 0.5; // actual physics length
  const POLE_MASS_LENGTH = POLE_MASS * POLE_HALF_LEN;
  const FORCE_MAG = 20.0; // push force magnitude (RL ke liye — zyada force = zyada authority)
  const MANUAL_FORCE_MIN = 6.0; // manual mode: minimum force (tap/initial)
  const MANUAL_FORCE_MAX = 25.0; // manual mode: max force (hold karne pe ramp up)
  const MANUAL_RAMP_MS = 400; // itne ms mein min se max force tak pahunchega
  const DT = 0.02; // euler integration timestep
  const MAX_STEPS = 750; // 15 seconds at DT=0.02 — ye objective hai, 15 sec balance = solved
  const TRACK_LIMIT = 4.0; // cart itna door ja sakta hai (pehle 2.4 tha — ab zyada space)
  const ANGLE_LIMIT = 45 * Math.PI / 180; // 45 degrees — bahut margin, human playable

  // --- Canvas dimensions (CSS pixels mein, DPR se multiply nahi) ---
  const MAIN_HEIGHT = 300;
  const GRAPH_HEIGHT = 80;

  // --- Neural Network parameters ---
  // tanh activation, 4 inputs, 16 hidden (pehle 8 tha — too small), 2 outputs
  const INPUT_SIZE = 4;
  const HIDDEN_SIZE = 32; // 16 se badhaaya — wider state space handle karne ke liye
  const OUTPUT_SIZE = 2;
  const LEARNING_RATE = 0.005; // batch gradient + shaped reward ke saath 0.005 stable hai

  // --- Training state ---
  let episode = 0;
  let bestReward = 0;
  let rewardHistory = []; // last 50 episodes ka reward track karenge
  const MAX_HISTORY = 50;

  // current episode ka data — REINFORCE ke liye chahiye
  let episodeLog = []; // {state, action} store karenge har step ka

  // environment state
  let cartX = 0, cartVel = 0, poleAngle = 0, poleAngVel = 0;
  let stepCount = 0;
  let episodeDone = false;

  // speed control — 1x, 5x, 20x steps per frame
  let speedMultiplier = 1;
  let manualMode = false;
  // real-time control: track which keys are currently HELD
  // -1 = no key held (no force), 0 = left held, 1 = right held
  let manualAction = -1;
  let leftHeld = false, rightHeld = false;
  let holdStartTime = 0; // variable acceleration — jab se key hold kiya

  // animation state — properly managed with IntersectionObserver
  let animationId = null;
  let isVisible = false;

  // --- Neural Network weights ---
  // Xavier initialization — random nahi, proper scale se initialize kar
  let W1, b1, W2, b2;

  // saved log probabilities aur rewards — policy gradient ke liye
  // SIRF RL mode mein accumulate honge, manual mein nahi (bug #6 fix)
  let savedLogProbs = [];
  let savedRewards = [];

  // --- Weight initialization function ---
  function initWeights() {
    // Xavier/Glorot initialization — variance = 2/(fan_in + fan_out)
    const scale1 = Math.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE));
    const scale2 = Math.sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE));

    W1 = Array.from({ length: INPUT_SIZE }, () =>
      Array.from({ length: HIDDEN_SIZE }, () => (Math.random() * 2 - 1) * scale1)
    );
    b1 = new Array(HIDDEN_SIZE).fill(0);
    W2 = Array.from({ length: HIDDEN_SIZE }, () =>
      Array.from({ length: OUTPUT_SIZE }, () => (Math.random() * 2 - 1) * scale2)
    );
    b2 = new Array(OUTPUT_SIZE).fill(0);
  }

  // --- tanh activation --- (sigmoid hataya, vanishing gradient deta tha — bug #3)
  function tanh(x) {
    // overflow protection
    if (x > 20) return 1;
    if (x < -20) return -1;
    const e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }

  // --- Softmax — output probabilities ke liye ---
  function softmax(arr) {
    const maxVal = Math.max(...arr);
    const exps = arr.map(v => Math.exp(v - maxVal)); // numerical stability ke liye max ghata
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
  }

  // --- Forward pass — state se action probability nikal ---
  function forward(state) {
    // hidden layer: tanh(W1^T * state + b1)
    const hidden = new Array(HIDDEN_SIZE);
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      let sum = b1[j];
      for (let i = 0; i < INPUT_SIZE; i++) {
        sum += state[i] * W1[i][j];
      }
      hidden[j] = tanh(sum);
    }

    // output layer: softmax(W2^T * hidden + b2)
    const logits = new Array(OUTPUT_SIZE);
    for (let j = 0; j < OUTPUT_SIZE; j++) {
      let sum = b2[j];
      for (let i = 0; i < HIDDEN_SIZE; i++) {
        sum += hidden[i] * W2[i][j];
      }
      logits[j] = sum;
    }

    const probs = softmax(logits);
    return { hidden, logits, probs };
  }

  // --- Action sample kar probability se ---
  function selectAction(state) {
    const { hidden, probs } = forward(state);
    // probability se action choose kar — exploration ke liye zaroori hai
    const r = Math.random();
    const action = r < probs[0] ? 0 : 1;
    // log probability store kar — gradient update mein chahiye
    const logProb = Math.log(probs[action] + 1e-8);

    return { action, logProb, hidden, probs };
  }

  // --- State normalization — sabko [-1, 1] range mein laao ---
  // bina normalize ke cartX (~4) aur poleAngle (~0.7) mein 5x scale difference hai
  // gradient unbalanced ho jaata hai — network ek feature zyada sunega
  function normalizeState(state) {
    return [
      state[0] / TRACK_LIMIT,     // cartX: [-4, 4] → [-1, 1]
      state[1] / 5.0,             // cartVel: roughly [-5, 5] → [-1, 1]
      state[2] / ANGLE_LIMIT,     // poleAngle: [-0.785, 0.785] → [-1, 1]
      state[3] / 5.0              // poleAngVel: roughly [-5, 5] → [-1, 1]
    ];
  }

  // --- REINFORCE update — BATCH gradient accumulation ---
  // PEHLE KAISE THA (GALAT): har timestep pe weights update kar rahe the LOOP ke andar
  //   → t=0 pe weights change → t=1 pe forward pass GALAT weights se → gradient noise
  //   → 2000 steps mein weights completely destroy ho jaate the
  // AB KAISE HAI (SAHI): saare gradients pehle accumulate karo, fir EK BAAR apply karo
  function updatePolicy() {
    if (savedLogProbs.length === 0) return;

    const gamma = 0.99;
    const T = savedLogProbs.length;
    const returns = new Array(T);
    let R = 0;
    for (let t = T - 1; t >= 0; t--) {
      R = savedRewards[t] + gamma * R;
      returns[t] = R;
    }

    // normalize returns — mean=0, std=1
    const mean = returns.reduce((a, b) => a + b, 0) / T;
    const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / T;
    const std = Math.sqrt(variance) + 1e-8;
    for (let i = 0; i < T; i++) {
      returns[i] = (returns[i] - mean) / std;
    }

    // gradient accumulators — ZERO initialize
    const gW1 = Array.from({ length: INPUT_SIZE }, () => new Array(HIDDEN_SIZE).fill(0));
    const gb1 = new Array(HIDDEN_SIZE).fill(0);
    const gW2 = Array.from({ length: HIDDEN_SIZE }, () => new Array(OUTPUT_SIZE).fill(0));
    const gb2a = new Array(OUTPUT_SIZE).fill(0);

    // SAARE timesteps ka gradient accumulate kar — weights TOUCH NAHI KARNA
    for (let t = 0; t < T; t++) {
      const state = episodeLog[t].state;
      const action = episodeLog[t].action;
      const advantage = returns[t];

      // forward pass — weights UNCHANGED hain, sab consistent hai
      const { hidden, probs } = forward(state);
      const dLogits = probs.map((p, i) => (i === action ? 1 - p : -p));

      // W2 gradient accumulate
      for (let i = 0; i < HIDDEN_SIZE; i++) {
        for (let j = 0; j < OUTPUT_SIZE; j++) {
          gW2[i][j] += advantage * hidden[i] * dLogits[j];
        }
      }
      for (let j = 0; j < OUTPUT_SIZE; j++) {
        gb2a[j] += advantage * dLogits[j];
      }

      // backprop to hidden — tanh derivative: 1 - h^2
      const dHidden = new Array(HIDDEN_SIZE).fill(0);
      for (let i = 0; i < HIDDEN_SIZE; i++) {
        for (let j = 0; j < OUTPUT_SIZE; j++) {
          dHidden[i] += dLogits[j] * W2[i][j];
        }
        dHidden[i] *= (1 - hidden[i] * hidden[i]);
      }

      // W1 gradient accumulate
      for (let i = 0; i < INPUT_SIZE; i++) {
        for (let j = 0; j < HIDDEN_SIZE; j++) {
          gW1[i][j] += advantage * state[i] * dHidden[j];
        }
      }
      for (let j = 0; j < HIDDEN_SIZE; j++) {
        gb1[j] += advantage * dHidden[j];
      }
    }

    // gradient clipping — norm bahut bada ho toh clip kar, nahi toh weights explode karenge
    let gradNorm = 0;
    for (let i = 0; i < INPUT_SIZE; i++)
      for (let j = 0; j < HIDDEN_SIZE; j++) gradNorm += gW1[i][j] * gW1[i][j];
    for (let j = 0; j < HIDDEN_SIZE; j++) gradNorm += gb1[j] * gb1[j];
    for (let i = 0; i < HIDDEN_SIZE; i++)
      for (let j = 0; j < OUTPUT_SIZE; j++) gradNorm += gW2[i][j] * gW2[i][j];
    for (let j = 0; j < OUTPUT_SIZE; j++) gradNorm += gb2a[j] * gb2a[j];
    gradNorm = Math.sqrt(gradNorm);

    const maxNorm = 5.0;
    const clipScale = gradNorm > maxNorm ? maxNorm / gradNorm : 1.0;

    // EK SINGLE BATCH UPDATE — yahi sahi tarika hai REINFORCE ka
    const lr = LEARNING_RATE * clipScale;
    for (let i = 0; i < INPUT_SIZE; i++)
      for (let j = 0; j < HIDDEN_SIZE; j++) W1[i][j] += lr * gW1[i][j];
    for (let j = 0; j < HIDDEN_SIZE; j++) b1[j] += lr * gb1[j];
    for (let i = 0; i < HIDDEN_SIZE; i++)
      for (let j = 0; j < OUTPUT_SIZE; j++) W2[i][j] += lr * gW2[i][j];
    for (let j = 0; j < OUTPUT_SIZE; j++) b2[j] += lr * gb2a[j];
  }

  // --- Environment reset ---
  function resetEnv() {
    if (manualMode) {
      // manual mode — thoda zyada tilt se shuru, interesting ho
      cartX = (Math.random() - 0.5) * 0.3;
      cartVel = (Math.random() - 0.5) * 0.2;
      poleAngle = (Math.random() - 0.5) * 0.4; // ~±11.5 degrees
      poleAngVel = (Math.random() - 0.5) * 0.3;
    } else {
      // RL mode — chhoti initial conditions, nahi toh seekh nahi paayega
      cartX = (Math.random() - 0.5) * 0.1;
      cartVel = (Math.random() - 0.5) * 0.1;
      poleAngle = (Math.random() - 0.5) * 0.1; // ~±2.8 degrees
      poleAngVel = (Math.random() - 0.5) * 0.1;
    }
    stepCount = 0;
    episodeDone = false;
    // ye sirf episode ke shuru mein clear hote hain
    savedLogProbs = [];
    savedRewards = [];
    episodeLog = [];
  }

  // --- Physics step — Euler integration ---
  // action: 0 = left, 1 = right, -1 = no force (manual mode mein)
  // forceOverride: manual mode mein variable force pass karte hain
  function physicsStep(action, forceOverride) {
    let force;
    if (action === -1) {
      force = 0; // no key held — koi force nahi
    } else if (forceOverride !== undefined) {
      force = action === 1 ? forceOverride : -forceOverride;
    } else {
      force = action === 1 ? FORCE_MAG : -FORCE_MAG;
    }

    const cosA = Math.cos(poleAngle);
    const sinA = Math.sin(poleAngle);

    // cart-pole dynamics equations — Lagrangian mechanics se derive ki hain
    const temp = (force + POLE_MASS_LENGTH * poleAngVel * poleAngVel * sinA) / TOTAL_MASS;
    const angAcc = (GRAVITY * sinA - cosA * temp) /
      (POLE_HALF_LEN * (4.0 / 3.0 - POLE_MASS * cosA * cosA / TOTAL_MASS));
    const cartAcc = temp - POLE_MASS_LENGTH * angAcc * cosA / TOTAL_MASS;

    // Euler integration
    cartX += cartVel * DT;
    cartVel += cartAcc * DT;
    poleAngle += poleAngVel * DT;
    poleAngVel += angAcc * DT;

    stepCount++;

    // episode khatam check kar — limit cross ki ya steps khatam
    if (Math.abs(cartX) > TRACK_LIMIT || Math.abs(poleAngle) > ANGLE_LIMIT || stepCount >= MAX_STEPS) {
      episodeDone = true;
    }

    // SHAPED REWARD — sirf +1/0 se REINFORCE bahut slow seekhta hai
    // angle upright + position center = high reward, marne pe penalty
    if (episodeDone && stepCount < MAX_STEPS) {
      return -2.0; // penalty for dying — strong signal "ye galat tha"
    }
    const angleReward = 1.0 - Math.abs(poleAngle) / ANGLE_LIMIT;
    const posReward = 1.0 - Math.abs(cartX) / TRACK_LIMIT;
    return angleReward * 0.5 + posReward * 0.5;
  }

  // --- DOM structure banate hain ---
  // BUG #5 FIX: container ke existing children (header, description) ko RAKH
  // naye elements APPEND kar, wipe mat kar

  // main canvas — cart pole yahan dikhega
  const mainCanvas = document.createElement('canvas');
  mainCanvas.style.cssText = [
    'width:100%',
    'height:' + MAIN_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px',
    'cursor:default',
    'background:transparent',
  ].join(';');
  container.appendChild(mainCanvas);

  // graph canvas — reward history ka line chart
  const graphCanvas = document.createElement('canvas');
  graphCanvas.style.cssText = [
    'width:100%',
    'height:' + GRAPH_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px',
    'margin-top:8px',
    'background:transparent',
  ].join(';');
  container.appendChild(graphCanvas);

  // stats bar — episode, reward, best dikhao
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'justify-content:center',
    'gap:24px',
    'margin-top:8px',
    'font-family:monospace',
    'font-size:13px',
    'color:#b0b0b0',
  ].join(';');
  container.appendChild(statsDiv);

  // controls bar — buttons aur selectors
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Button helper ---
  function makeButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'background:rgba(74,158,255,0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(74,158,255,0.25)',
      'border-radius:6px',
      'padding:6px 14px',
      'font-size:12px',
      'font-family:monospace',
      'cursor:pointer',
      'transition:all 0.2s',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(74,158,255,0.25)';
      btn.style.color = '#ffffff';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(74,158,255,0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // reset button — naye sirey se training shuru, weights bhi reinitialize
  makeButton('Reset Training', () => {
    initWeights();
    episode = 0;
    bestReward = 0;
    rewardHistory = [];
    resetEnv();
    updateStats();
  });

  // speed selector label
  const speedLabel = document.createElement('span');
  speedLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace;';
  speedLabel.textContent = 'Speed:';
  controlsDiv.appendChild(speedLabel);

  // speed dropdown — 1x, 5x, 20x
  const speedSelect = document.createElement('select');
  speedSelect.style.cssText = [
    'background:rgba(74,158,255,0.1)',
    'color:#b0b0b0',
    'border:1px solid rgba(74,158,255,0.25)',
    'border-radius:6px',
    'padding:4px 8px',
    'font-size:12px',
    'font-family:monospace',
    'cursor:pointer',
  ].join(';');
  [1, 5, 20].forEach(s => {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s + 'x';
    opt.style.cssText = 'background:#1a1a2e;color:#b0b0b0;';
    speedSelect.appendChild(opt);
  });
  speedSelect.addEventListener('change', () => {
    speedMultiplier = parseInt(speedSelect.value);
  });
  controlsDiv.appendChild(speedSelect);

  // manual mode toggle button
  const manualBtn = makeButton('Manual Mode: OFF', () => {
    manualMode = !manualMode;
    manualBtn.textContent = 'Manual Mode: ' + (manualMode ? 'ON' : 'OFF');
    if (manualMode) {
      manualBtn.style.borderColor = 'rgba(74,158,255,0.6)';
      manualBtn.style.color = '#4a9eff';
      // reset held state — clean start
      leftHeld = false;
      rightHeld = false;
      manualAction = -1;
      holdStartTime = 0;
      savedLogProbs = [];
      savedRewards = [];
      episodeLog = [];
    } else {
      manualBtn.style.borderColor = 'rgba(74,158,255,0.25)';
      manualBtn.style.color = '#b0b0b0';
      leftHeld = false;
      rightHeld = false;
      manualAction = -1;
      resetEnv();
    }
  });

  // keyboard controls — REAL-TIME hold: keydown start, keyup stop
  // jab tak key dabaya tab tak force, chhodha toh zero force
  function updateManualAction() {
    const prevAction = manualAction;
    if (leftHeld && !rightHeld) manualAction = 0;
    else if (rightHeld && !leftHeld) manualAction = 1;
    else manualAction = -1; // no key or both keys = no force
    // agar direction badla toh hold timer reset kar
    if (manualAction !== prevAction && manualAction !== -1) {
      holdStartTime = performance.now();
    }
  }

  document.addEventListener('keydown', (e) => {
    if (!manualMode || !isVisible) return;
    if (e.key === 'ArrowLeft') { leftHeld = true; e.preventDefault(); updateManualAction(); }
    if (e.key === 'ArrowRight') { rightHeld = true; e.preventDefault(); updateManualAction(); }
  });
  document.addEventListener('keyup', (e) => {
    if (!manualMode) return;
    if (e.key === 'ArrowLeft') { leftHeld = false; updateManualAction(); }
    if (e.key === 'ArrowRight') { rightHeld = false; updateManualAction(); }
  });

  // --- Canvas pe touch/click — mobile users ke liye ---
  // left half = LEFT, right half = RIGHT
  // mousedown/touchstart = start push, mouseup/touchend = stop push
  function handleCanvasDown(e) {
    if (!manualMode) return;
    e.preventDefault();
    const rect = mainCanvas.getBoundingClientRect();
    let clientX;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
    } else {
      clientX = e.clientX;
    }
    const relX = clientX - rect.left;
    if (relX < rect.width / 2) {
      leftHeld = true;
    } else {
      rightHeld = true;
    }
    updateManualAction();
  }
  function handleCanvasUp(e) {
    if (!manualMode) return;
    // sab release kar do
    leftHeld = false;
    rightHeld = false;
    updateManualAction();
  }
  mainCanvas.addEventListener('mousedown', handleCanvasDown);
  mainCanvas.addEventListener('mouseup', handleCanvasUp);
  mainCanvas.addEventListener('mouseleave', handleCanvasUp);
  mainCanvas.addEventListener('touchstart', handleCanvasDown, { passive: false });
  mainCanvas.addEventListener('touchend', handleCanvasUp);
  mainCanvas.addEventListener('touchcancel', handleCanvasUp);

  // --- Canvas resize handling ---
  // BUG #7 FIX: setTransform use karenge, har jagah manual dpr multiply nahi
  function resizeCanvases() {
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const dpr = window.devicePixelRatio || 1;

    // canvas buffer size = CSS size * DPR (sharp rendering ke liye)
    mainCanvas.width = w * dpr;
    mainCanvas.height = MAIN_HEIGHT * dpr;
    graphCanvas.width = w * dpr;
    graphCanvas.height = GRAPH_HEIGHT * dpr;

    // CSS display size — ye actual pixels nahi, layout pixels hai
    mainCanvas.style.width = w + 'px';
    graphCanvas.style.width = w + 'px';
  }

  // --- Main canvas rendering ---
  function drawMain() {
    const ctx = mainCanvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    // BUG #7 FIX: setTransform se ek baar scale set kar do
    // ab saari drawing CSS pixels mein hogi — dpr se manually multiply nahi karna padega
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // CSS pixel dimensions mein kaam karenge
    const w = mainCanvas.width / dpr;
    const h = MAIN_HEIGHT;

    ctx.clearRect(0, 0, w, h);

    // coordinate system: center bottom mein origin rakh, scale kar
    const centerX = w / 2;
    const groundY = h * 0.8; // zameen ka level
    const scale = (w / 2) / (TRACK_LIMIT * 1.3); // track ko canvas mein fit kar

    // track/rail draw kar — neeche ek subtle line
    ctx.strokeStyle = 'rgba(74,158,255,0.15)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX - TRACK_LIMIT * scale, groundY);
    ctx.lineTo(centerX + TRACK_LIMIT * scale, groundY);
    ctx.stroke();

    // track limits — dono taraf chhoti vertical lines
    ctx.strokeStyle = 'rgba(255,100,100,0.3)';
    ctx.lineWidth = 1.5;
    [-TRACK_LIMIT, TRACK_LIMIT].forEach(lim => {
      const x = centerX + lim * scale;
      ctx.beginPath();
      ctx.moveTo(x, groundY - 10);
      ctx.lineTo(x, groundY + 10);
      ctx.stroke();
    });

    // cart draw kar — blue rectangle with rounded corners
    const cartW = 60;
    const cartH = 30;
    const cartScreenX = centerX + cartX * scale;
    const cartScreenY = groundY;

    ctx.fillStyle = 'rgba(74,158,255,0.85)';
    ctx.strokeStyle = 'rgba(74,158,255,0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(cartScreenX - cartW / 2, cartScreenY - cartH, cartW, cartH, 4);
    ctx.fill();
    ctx.stroke();

    // wheels — do chhote circles neeche
    ctx.fillStyle = 'rgba(74,158,255,0.5)';
    const wheelR = 5;
    [-15, 15].forEach(offset => {
      ctx.beginPath();
      ctx.arc(cartScreenX + offset, cartScreenY, wheelR, 0, Math.PI * 2);
      ctx.fill();
    });

    // pole draw kar — cart ke top center se
    const polePixelLen = 120;
    const poleStartX = cartScreenX;
    const poleStartY = cartScreenY - cartH;
    const poleEndX = poleStartX + Math.sin(poleAngle) * polePixelLen;
    const poleEndY = poleStartY - Math.cos(poleAngle) * polePixelLen;

    // pole shadow for depth effect
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(poleStartX, poleStartY);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();

    // actual pole — color changes based on tilt (green=stable, red=falling)
    const tiltRatio = Math.abs(poleAngle) / ANGLE_LIMIT;
    const poleR = Math.round(255 * Math.min(1, tiltRatio * 2));
    const poleG = Math.round(255 * Math.max(0, 1 - tiltRatio * 2));
    ctx.strokeStyle = 'rgba(' + poleR + ',' + poleG + ',100,0.95)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(poleStartX, poleStartY);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();

    // pivot point — chhota circle jahan pole cart se juda hai
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.beginPath();
    ctx.arc(poleStartX, poleStartY, 4, 0, Math.PI * 2);
    ctx.fill();

    // pole tip — glow effect
    ctx.fillStyle = 'rgba(' + poleR + ',' + poleG + ',100,0.6)';
    ctx.beginPath();
    ctx.arc(poleEndX, poleEndY, 3, 0, Math.PI * 2);
    ctx.fill();

    // status text — manual mode mein instruction dikhao
    ctx.font = '11px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.6)';
    ctx.textAlign = 'left';
    if (manualMode) {
      ctx.fillText('Hold arrow keys / tap & hold to push', 10, 20);
      // force indicator — dikhao kitni force lag rahi hai
      if (manualAction !== -1) {
        const holdDuration = performance.now() - holdStartTime;
        const ramp = Math.min(1, holdDuration / MANUAL_RAMP_MS);
        const forcePercent = Math.round((MANUAL_FORCE_MIN + (MANUAL_FORCE_MAX - MANUAL_FORCE_MIN) * ramp) / MANUAL_FORCE_MAX * 100);
        ctx.fillStyle = 'rgba(74,158,255,0.5)';
        ctx.fillText('Force: ' + forcePercent + '%', 10, 36);
      }
    } else {
      ctx.fillText('RL training — goal: 15s balance', 10, 20);
    }

    // on-canvas buttons — manual mode mein LEFT aur RIGHT buttons dikhao
    // mobile users ke liye touch targets
    if (manualMode) {
      const btnW = 80;
      const btnH = 32;
      const btnY = h - 15 - btnH; // canvas ke neeche rakh
      const btnLeftX = 15;
      const btnRightX = w - 15 - btnW;

      // LEFT button — highlight jab held
      ctx.fillStyle = leftHeld ? 'rgba(74,158,255,0.45)' : 'rgba(74,158,255,0.12)';
      ctx.strokeStyle = leftHeld ? 'rgba(74,158,255,0.7)' : 'rgba(74,158,255,0.4)';
      ctx.lineWidth = leftHeld ? 2 : 1;
      ctx.beginPath();
      ctx.roundRect(btnLeftX, btnY, btnW, btnH, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = leftHeld ? '#ffffff' : 'rgba(176,176,176,0.7)';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('\u2190 LEFT', btnLeftX + btnW / 2, btnY + btnH / 2 + 4);

      // RIGHT button
      ctx.fillStyle = rightHeld ? 'rgba(74,158,255,0.45)' : 'rgba(74,158,255,0.12)';
      ctx.strokeStyle = rightHeld ? 'rgba(74,158,255,0.7)' : 'rgba(74,158,255,0.4)';
      ctx.lineWidth = rightHeld ? 2 : 1;
      ctx.beginPath();
      ctx.roundRect(btnRightX, btnY, btnW, btnH, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = rightHeld ? '#ffffff' : 'rgba(176,176,176,0.7)';
      ctx.textAlign = 'center';
      ctx.fillText('RIGHT \u2192', btnRightX + btnW / 2, btnY + btnH / 2 + 4);
    }

    // transform reset kar — next frame ke liye clean slate
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  // --- Graph canvas — reward history plot ---
  function drawGraph() {
    const ctx = graphCanvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    // BUG #7 FIX: setTransform se DPR handle — CSS pixels mein draw karenge
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = graphCanvas.width / dpr;
    const h = GRAPH_HEIGHT;

    ctx.clearRect(0, 0, w, h);

    if (rewardHistory.length < 2) {
      ctx.font = '11px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.4)';
      ctx.textAlign = 'center';
      ctx.fillText('Balance time \u2014 training shuru hone do...', w / 2, h / 2);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      return;
    }

    const padL = 40;
    const padR = 10;
    const padT = 10;
    const padB = 15;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // y-axis range — 0 se MAX_STEPS tak
    const maxReward = MAX_STEPS;

    // grid lines — subtle
    ctx.strokeStyle = 'rgba(74,158,255,0.08)';
    ctx.lineWidth = 1;
    for (let y = 0; y <= 4; y++) {
      const yy = padT + plotH * (1 - y / 4);
      ctx.beginPath();
      ctx.moveTo(padL, yy);
      ctx.lineTo(padL + plotW, yy);
      ctx.stroke();
    }

    // y-axis labels — time in seconds
    ctx.font = '9px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.4)';
    ctx.textAlign = 'right';
    [0, 187, 375, 562, 750].forEach(v => {
      const yy = padT + plotH * (1 - v / maxReward);
      ctx.fillText((v * DT).toFixed(0) + 's', padL - 5, yy + 3);
    });

    // reward line draw kar
    ctx.strokeStyle = 'rgba(74,158,255,0.8)';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    const n = rewardHistory.length;
    for (let i = 0; i < n; i++) {
      const x = padL + (i / (Math.max(n - 1, 1))) * plotW;
      const y = padT + plotH * (1 - rewardHistory[i] / maxReward);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // fill under the curve — gradient effect
    const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
    grad.addColorStop(0, 'rgba(74,158,255,0.15)');
    grad.addColorStop(1, 'rgba(74,158,255,0.0)');
    ctx.fillStyle = grad;
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.lineTo(padL, padT + plotH);
    ctx.closePath();
    ctx.fill();

    // current point highlight — last data point pe dot
    if (n > 0) {
      const lastX = padL + ((n - 1) / Math.max(n - 1, 1)) * plotW;
      const lastY = padT + plotH * (1 - rewardHistory[n - 1] / maxReward);
      ctx.fillStyle = 'rgba(74,158,255,0.9)';
      ctx.beginPath();
      ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // transform reset
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  // --- Stats update — time-based display ---
  function stepsToTime(steps) {
    return (steps * DT).toFixed(1) + 's';
  }

  function updateStats() {
    const lastSteps = rewardHistory.length > 0 ? rewardHistory[rewardHistory.length - 1] : 0;

    while (statsDiv.firstChild) statsDiv.removeChild(statsDiv.firstChild);

    function addStat(label, value, color) {
      const span = document.createElement('span');
      span.textContent = label;
      const valSpan = document.createElement('span');
      valSpan.style.color = color;
      valSpan.textContent = value;
      span.appendChild(valSpan);
      statsDiv.appendChild(span);
    }

    addStat('Episode: ', episode, '#4a9eff');
    addStat('Time: ', stepsToTime(lastSteps), '#4a9eff');
    addStat('Best: ', stepsToTime(bestReward), bestReward >= MAX_STEPS ? '#4aff8f' : '#4a9eff');
    if (bestReward >= MAX_STEPS) {
      addStat('', ' SOLVED!', '#4aff8f');
    }
  }

  // --- Training loop — ek episode ka ek step ---
  function trainingStep() {
    if (episodeDone) {
      // episode khatam — policy update aur naya episode
      if (!manualMode) {
        updatePolicy();
      }

      const totalReward = stepCount;
      rewardHistory.push(totalReward);
      if (rewardHistory.length > MAX_HISTORY) rewardHistory.shift();
      if (totalReward > bestReward) bestReward = totalReward;
      episode++;

      updateStats();
      resetEnv();
      return;
    }

    const rawState = [cartX, cartVel, poleAngle, poleAngVel];
    const state = normalizeState(rawState);

    if (manualMode) {
      // REAL-TIME manual control — hold key = push, release = no force
      const action = manualAction; // -1 (none), 0 (left), 1 (right)
      if (action === -1) {
        // koi key nahi dabaya — zero force, pole apne physics pe depend karega
        physicsStep(-1);
      } else {
        // variable acceleration — jitna zyada hold karo utna strong push
        const holdDuration = performance.now() - holdStartTime;
        const ramp = Math.min(1, holdDuration / MANUAL_RAMP_MS);
        const currentForce = MANUAL_FORCE_MIN + (MANUAL_FORCE_MAX - MANUAL_FORCE_MIN) * ramp;
        physicsStep(action, currentForce);
      }
    } else {
      // RL mode — neural network se action lo
      const result = selectAction(state);
      const action = result.action;

      savedLogProbs.push(result.logProb);
      episodeLog.push({ state: state.slice(), action });

      const reward = physicsStep(action);
      savedRewards.push(reward);
    }
  }

  // --- Main animation loop ---
  // BUG #1 FIX: isVisible false hone pe requestAnimationFrame CALL HI NAHI KARNA
  // pehle !isVisible pe bhi rAF call hota tha — CPU waste
  function animate() {
    // agar visible nahi hai toh rAF schedule mat kar — IntersectionObserver wapas start karega
    if (!isVisible) {
      animationId = null;
      return;
    }

    // speed multiplier ke hisaab se multiple steps chala ek frame mein
    for (let i = 0; i < speedMultiplier; i++) {
      trainingStep();
    }

    drawMain();
    drawGraph();

    animationId = requestAnimationFrame(animate);
  }

  // animation start/stop helpers — visibility ke saath sync
  function startAnimation() {
    if (animationId === null) {
      animationId = requestAnimationFrame(animate);
    }
  }

  function stopAnimation() {
    if (animationId !== null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // --- IntersectionObserver — sirf visible hone pe animate kar, CPU bach jaayega ---
  // BUG #1 FIX: visible hone pe start, invisible hone pe stop — no unnecessary rAF calls
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        const wasVisible = isVisible;
        isVisible = entry.isIntersecting;
        if (isVisible && !wasVisible) {
          // abhi visible hua — animation shuru kar
          startAnimation();
        } else if (!isVisible && wasVisible) {
          // abhi invisible hua — animation rok de
          stopAnimation();
        }
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);

  // --- Initialization — sab setup karke animation shuru kar ---
  initWeights();
  resetEnv();
  resizeCanvases();
  updateStats();

  // resize pe canvas update kar — responsive rehna chahiye
  window.addEventListener('resize', resizeCanvases);

  // animation start kar — agar visible hai tabhi chalegi, nahi toh observer handle karega
  startAnimation();
}
