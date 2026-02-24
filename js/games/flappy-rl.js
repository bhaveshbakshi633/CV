// ============================================================
// Flappy Bird + REINFORCE Policy Gradient Agent
// Bird udna seekhega pipes ke beech se — live training dikhega
// Neural network visualization, reward plot, manual play mode sab hai
// ============================================================

// yahi main entry point hai — container dhundho, canvas banao, training shuru karo
export function initFlappyRL() {
  const container = document.getElementById('flappyRLContainer');
  if (!container) {
    console.warn('flappyRLContainer nahi mila bhai, Flappy RL skip kar rahe hain');
    return;
  }

  // purane children saaf kar — fresh start
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const ACCENT_RGB = '74,158,255';

  // --- Game constants ---
  const GRAVITY = 0.4;           // neeche kheenchne waali force
  const FLAP_VELOCITY = -6.5;    // flap karne pe kitna upar jaaye
  const PIPE_SPEED = 2.0;        // pipes kitni tez left move karein
  const PIPE_WIDTH = 45;         // pipe ki width pixels mein
  const PIPE_SPACING = 180;      // do pipes ke beech horizontal distance
  const BIRD_RADIUS = 10;        // bird ka size
  const GROUND_HEIGHT = 40;      // zameen ki height
  const BIRD_X_FRAC = 0.2;      // bird canvas ke 20% pe hoga (left side)

  // --- Neural Network constants ---
  const INPUT_SIZE = 5;          // bird_y, bird_vy, dist_pipe, gap_top, gap_bottom
  const HIDDEN_SIZE = 16;        // hidden layer neurons
  const OUTPUT_SIZE = 1;         // flap probability (sigmoid output)
  const GAMMA = 0.99;            // discount factor for returns
  const MAX_GRAD_NORM = 5.0;     // gradient clipping threshold

  // --- State variables ---
  let canvasW = 0;
  let dpr = 1;
  let isVisible = false;
  let animationId = null;

  // training parameters — sliders se control honge
  let trainSpeed = 5;            // kitne steps ek frame mein (training mode)
  let learningRate = 0.01;       // SGD learning rate
  let pipeGap = 120;             // pipes ke beech ka gap

  // game state
  let birdY = 0;                 // bird ki vertical position
  let birdVY = 0;                // bird ki vertical velocity
  let pipes = [];                // active pipes ka array
  let score = 0;                 // current episode ka score (pipes passed)
  let frameCount = 0;            // frames alive in current episode
  let gameOver = false;

  // training state
  let episode = 0;
  let bestScore = 0;
  let rewardHistory = [];        // last 100 episodes ka score
  const MAX_HISTORY = 100;

  // REINFORCE buffers — episode ke baad update ke liye
  let savedStates = [];          // har step ka state
  let savedActions = [];         // har step ka action (0 or 1)
  let savedRewards = [];         // har step ka reward

  // mode control: 'train' | 'watch' | 'play'
  let mode = 'train';

  // neural network weights — Xavier init honge
  let W1, b1, W2, b2;

  // last forward pass ke activations — visualization ke liye
  let lastHidden = new Array(HIDDEN_SIZE).fill(0);
  let lastInputs = new Array(INPUT_SIZE).fill(0);
  let lastOutput = 0.5;

  // split canvas layout — left game, right stats
  const GAME_FRAC = 0.65;       // 65% game area

  // manual play ke liye flap request
  let flapRequested = false;

  // --- Canvas banao ---
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:default',
    'background:#0a0a1a',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls container ---
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

  // --- Helper: slider banao ---
  function createSlider(label, min, max, value, step, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = [
      'display:flex',
      'align-items:center',
      'gap:6px',
      'font-family:"JetBrains Mono",monospace',
      'font-size:11px',
      'color:#888',
    ].join(';');

    const lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.whiteSpace = 'nowrap';
    wrap.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.value = value;
    slider.step = step;
    slider.style.cssText = 'width:80px;accent-color:' + ACCENT + ';cursor:pointer;';
    wrap.appendChild(slider);

    const valSpan = document.createElement('span');
    valSpan.textContent = step >= 1 ? Number(value).toFixed(0) : Number(value).toFixed(3);
    valSpan.style.cssText = 'min-width:32px;text-align:right;color:' + ACCENT + ';';
    wrap.appendChild(valSpan);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valSpan.textContent = step >= 1 ? v.toFixed(0) : v.toFixed(3);
      onChange(v);
    });

    controlsDiv.appendChild(wrap);
    return { wrap, slider, valSpan };
  }

  // --- Helper: button banao ---
  function makeButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'border-radius:6px',
      'padding:6px 14px',
      'font-size:12px',
      'font-family:"JetBrains Mono",monospace',
      'cursor:pointer',
      'transition:all 0.2s',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#ffffff';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // --- Controls banao ---
  createSlider('Speed', 1, 20, trainSpeed, 1, v => { trainSpeed = v; });
  createSlider('LR', 0.001, 0.05, learningRate, 0.001, v => { learningRate = v; });
  createSlider('Gap', 80, 160, pipeGap, 5, v => { pipeGap = v; });

  // mode buttons — highlight active mode
  let trainBtn, watchBtn, playBtn;

  function updateModeButtons() {
    // sab buttons ka style reset karo, active waale ko highlight
    [trainBtn, watchBtn, playBtn].forEach(btn => {
      btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#b0b0b0';
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
    });
    const activeBtn = mode === 'train' ? trainBtn : mode === 'watch' ? watchBtn : playBtn;
    activeBtn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.7)';
    activeBtn.style.color = ACCENT;
    activeBtn.style.background = 'rgba(' + ACCENT_RGB + ',0.2)';
  }

  trainBtn = makeButton('Train', () => {
    mode = 'train';
    updateModeButtons();
    resetGame();
  });
  watchBtn = makeButton('AI Play', () => {
    mode = 'watch';
    updateModeButtons();
    resetGame();
  });
  playBtn = makeButton('Play', () => {
    mode = 'play';
    updateModeButtons();
    resetGame();
  });
  makeButton('Reset Weights', () => {
    initWeights();
    episode = 0;
    bestScore = 0;
    rewardHistory = [];
    resetGame();
  });

  updateModeButtons();

  // ============================================================
  // NEURAL NETWORK
  // ============================================================

  // --- Xavier Weight Initialization ---
  function initWeights() {
    const scale1 = Math.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE));
    const scale2 = Math.sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE));

    // W1: INPUT_SIZE x HIDDEN_SIZE
    W1 = Array.from({ length: INPUT_SIZE }, () =>
      Array.from({ length: HIDDEN_SIZE }, () => (Math.random() * 2 - 1) * scale1)
    );
    b1 = new Array(HIDDEN_SIZE).fill(0);

    // W2: HIDDEN_SIZE x OUTPUT_SIZE
    W2 = Array.from({ length: HIDDEN_SIZE }, () =>
      Array.from({ length: OUTPUT_SIZE }, () => (Math.random() * 2 - 1) * scale2)
    );
    b2 = new Array(OUTPUT_SIZE).fill(0);
  }

  // --- tanh — hidden layer activation ---
  function tanh(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    const e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }

  // --- sigmoid — output layer, probability [0, 1] chahiye ---
  function sigmoid(x) {
    if (x > 20) return 1;
    if (x < -20) return 0;
    return 1.0 / (1.0 + Math.exp(-x));
  }

  // --- Forward pass — state se flap probability nikaal ---
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

    // output layer: sigmoid(W2^T * hidden + b2) — flap probability
    let outSum = b2[0];
    for (let i = 0; i < HIDDEN_SIZE; i++) {
      outSum += hidden[i] * W2[i][0];
    }
    const flapProb = sigmoid(outSum);

    // visualization ke liye store kar — har forward pass pe update
    lastHidden = hidden;
    lastInputs = state.slice();
    lastOutput = flapProb;

    return { hidden, flapProb };
  }

  // --- Action select kar — Bernoulli sampling ---
  function selectAction(state) {
    const { hidden, flapProb } = forward(state);
    // Bernoulli sample — random < p toh flap, nahi toh mat
    const action = Math.random() < flapProb ? 1 : 0;
    return { action, flapProb, hidden };
  }

  // ============================================================
  // STATE NORMALIZATION
  // ============================================================

  // --- Game state normalize kar — sabko [-1, 1] range mein lao ---
  function getState() {
    const gameW = canvasW * GAME_FRAC;
    const playH = CANVAS_HEIGHT - GROUND_HEIGHT;
    const birdX = gameW * BIRD_X_FRAC;

    // sabse nazdeeki pipe dhundh jo bird se aage hai
    let nextPipe = null;
    for (let i = 0; i < pipes.length; i++) {
      if (pipes[i].x + PIPE_WIDTH > birdX) {
        nextPipe = pipes[i];
        break;
      }
    }

    // agar koi pipe nahi hai — default: door, center gap
    if (!nextPipe) {
      return [
        (birdY / playH) * 2 - 1,
        birdVY / 10.0,
        1.0,
        0.0,
        0.0,
      ];
    }

    // normalized state vector — 5 features
    const distX = (nextPipe.x - birdX) / gameW;
    const gapTopY = (nextPipe.gapY / playH) * 2 - 1;
    const gapBotY = ((nextPipe.gapY + pipeGap) / playH) * 2 - 1;

    return [
      (birdY / playH) * 2 - 1,     // bird_y: [-1, 1]
      birdVY / 10.0,                // bird_vy: roughly [-1, 1]
      distX * 2 - 1,               // dist: [-1, 1] (close to far)
      gapTopY,                      // gap top: [-1, 1]
      gapBotY,                      // gap bottom: [-1, 1]
    ];
  }

  // ============================================================
  // REINFORCE POLICY GRADIENT UPDATE
  // ============================================================

  // episode khatam hone ke baad — BATCH gradient update
  function updatePolicy() {
    const T = savedStates.length;
    if (T === 0) return;

    // discounted returns compute kar — end se start tak jaao
    const returns = new Array(T);
    let R = 0;
    for (let t = T - 1; t >= 0; t--) {
      R = savedRewards[t] + GAMMA * R;
      returns[t] = R;
    }

    // returns normalize kar — mean 0, std 1 banao
    let mean = 0;
    for (let i = 0; i < T; i++) mean += returns[i];
    mean /= T;
    let variance = 0;
    for (let i = 0; i < T; i++) variance += (returns[i] - mean) * (returns[i] - mean);
    variance /= T;
    const std = Math.sqrt(variance) + 1e-8;
    for (let i = 0; i < T; i++) {
      returns[i] = (returns[i] - mean) / std;
    }

    // gradient accumulators — sab zero se shuru
    const gW1 = Array.from({ length: INPUT_SIZE }, () => new Array(HIDDEN_SIZE).fill(0));
    const gb1 = new Array(HIDDEN_SIZE).fill(0);
    const gW2 = Array.from({ length: HIDDEN_SIZE }, () => new Array(OUTPUT_SIZE).fill(0));
    const gb2a = new Array(OUTPUT_SIZE).fill(0);

    // har timestep ka gradient accumulate kar — weights mat chhuuna loop mein
    for (let t = 0; t < T; t++) {
      const state = savedStates[t];
      const action = savedActions[t];
      const advantage = returns[t];

      // forward pass — current (unchanged) weights se
      const { hidden, flapProb } = forward(state);

      // Bernoulli log-likelihood ka gradient: dL/dz = action - p
      // agar flap kiya (action=1): gradient = 1-p → probability badhao
      // agar nahi kiya (action=0): gradient = 0-p = -p → probability ghatao
      const dOut = action - flapProb;

      // W2 gradient accumulate: hidden_i * dOut * advantage
      for (let i = 0; i < HIDDEN_SIZE; i++) {
        gW2[i][0] += advantage * hidden[i] * dOut;
      }
      gb2a[0] += advantage * dOut;

      // backprop to hidden — tanh derivative: 1 - h^2
      const dHidden = new Array(HIDDEN_SIZE);
      for (let i = 0; i < HIDDEN_SIZE; i++) {
        dHidden[i] = dOut * W2[i][0] * (1 - hidden[i] * hidden[i]);
      }

      // W1 gradient accumulate: state_i * dHidden_j * advantage
      for (let i = 0; i < INPUT_SIZE; i++) {
        for (let j = 0; j < HIDDEN_SIZE; j++) {
          gW1[i][j] += advantage * state[i] * dHidden[j];
        }
      }
      for (let j = 0; j < HIDDEN_SIZE; j++) {
        gb1[j] += advantage * dHidden[j];
      }
    }

    // gradient clipping — exploding gradient se bacho
    let gradNorm = 0;
    for (let i = 0; i < INPUT_SIZE; i++)
      for (let j = 0; j < HIDDEN_SIZE; j++) gradNorm += gW1[i][j] * gW1[i][j];
    for (let j = 0; j < HIDDEN_SIZE; j++) gradNorm += gb1[j] * gb1[j];
    for (let i = 0; i < HIDDEN_SIZE; i++)
      gradNorm += gW2[i][0] * gW2[i][0];
    gradNorm += gb2a[0] * gb2a[0];
    gradNorm = Math.sqrt(gradNorm);

    const clipScale = gradNorm > MAX_GRAD_NORM ? MAX_GRAD_NORM / gradNorm : 1.0;

    // EK SINGLE BATCH UPDATE — weights sirf yahan change honge
    const lr = learningRate * clipScale;
    for (let i = 0; i < INPUT_SIZE; i++)
      for (let j = 0; j < HIDDEN_SIZE; j++) W1[i][j] += lr * gW1[i][j];
    for (let j = 0; j < HIDDEN_SIZE; j++) b1[j] += lr * gb1[j];
    for (let i = 0; i < HIDDEN_SIZE; i++)
      W2[i][0] += lr * gW2[i][0];
    b2[0] += lr * gb2a[0];
  }

  // ============================================================
  // GAME MECHANICS
  // ============================================================

  // --- Game reset — naya episode shuru ---
  function resetGame() {
    const playH = CANVAS_HEIGHT - GROUND_HEIGHT;
    birdY = playH * 0.5;         // center mein shuru
    birdVY = 0;
    pipes = [];
    score = 0;
    frameCount = 0;
    gameOver = false;
    flapRequested = false;

    // REINFORCE buffers clear kar
    savedStates = [];
    savedActions = [];
    savedRewards = [];

    // pehla pipe thoda door se spawn kar — start mein breathing room de
    spawnPipe(canvasW * GAME_FRAC * 0.8);
  }

  // --- Pipe spawn karo ---
  function spawnPipe(startX) {
    const playH = CANVAS_HEIGHT - GROUND_HEIGHT;
    // gap randomly place kar — top/bottom se margin rakh taaki impossible na ho
    const margin = 50;
    const minGapY = margin;
    const maxGapY = playH - pipeGap - margin;
    // agar gap bahut chhota hai toh clamp kar
    const gapY = maxGapY > minGapY
      ? minGapY + Math.random() * (maxGapY - minGapY)
      : playH / 2 - pipeGap / 2;

    pipes.push({
      x: startX !== undefined ? startX : canvasW * GAME_FRAC,
      gapY: gapY,
      scored: false,
    });
  }

  // --- Game step — ek frame ka physics + collision ---
  // returns: true agar pipe score hua is step mein
  function gameStep(doFlap) {
    if (gameOver) return false;

    const gameW = canvasW * GAME_FRAC;
    const playH = CANVAS_HEIGHT - GROUND_HEIGHT;
    const birdX = gameW * BIRD_X_FRAC;

    // bird physics — gravity lagao, flap se upar jaao
    if (doFlap) {
      birdVY = FLAP_VELOCITY;
    }
    birdVY += GRAVITY;
    birdY += birdVY;

    // pipes left move karo
    for (let i = 0; i < pipes.length; i++) {
      pipes[i].x -= PIPE_SPEED;
    }

    // purane pipes hatao — screen se bahar nikal gaye
    while (pipes.length > 0 && pipes[0].x + PIPE_WIDTH < 0) {
      pipes.shift();
    }

    // naye pipes spawn karo — spacing maintain karo
    if (pipes.length === 0 || pipes[pipes.length - 1].x < gameW - PIPE_SPACING) {
      spawnPipe();
    }

    // score check — kya bird ne koi pipe cross kiya?
    let justScored = false;
    for (let i = 0; i < pipes.length; i++) {
      if (!pipes[i].scored && pipes[i].x + PIPE_WIDTH < birdX) {
        pipes[i].scored = true;
        score++;
        justScored = true;
      }
    }

    // collision detection — ground/ceiling
    if (birdY - BIRD_RADIUS < 0 || birdY + BIRD_RADIUS > playH) {
      gameOver = true;
      return justScored;
    }

    // collision — pipes se takrao
    for (let i = 0; i < pipes.length; i++) {
      const p = pipes[i];
      // horizontally overlap check
      if (birdX + BIRD_RADIUS > p.x && birdX - BIRD_RADIUS < p.x + PIPE_WIDTH) {
        // top pipe hit — bird gap ke upar hai
        if (birdY - BIRD_RADIUS < p.gapY) {
          gameOver = true;
          return justScored;
        }
        // bottom pipe hit — bird gap ke neeche hai
        if (birdY + BIRD_RADIUS > p.gapY + pipeGap) {
          gameOver = true;
          return justScored;
        }
      }
    }

    frameCount++;
    return justScored;
  }

  // --- Ek training/AI step — state→action→step→reward ---
  function runOneStep() {
    // agar game over hai toh episode close kar
    if (gameOver) {
      if (mode === 'train') {
        updatePolicy();
      }
      rewardHistory.push(score);
      if (rewardHistory.length > MAX_HISTORY) rewardHistory.shift();
      if (score > bestScore) bestScore = score;
      episode++;
      resetGame();
      return;
    }

    // play mode mein AI kuch nahi karega — user handle karega
    if (mode === 'play') return;

    // state nikal, AI se action lo
    const state = getState();
    const { action, flapProb } = selectAction(state);
    const doFlap = action === 1;

    // REINFORCE buffer mein store kar (sirf train mode)
    if (mode === 'train') {
      savedStates.push(state);
      savedActions.push(action);
    }

    // game step chala — score change track kar
    const justScored = gameStep(doFlap);

    // reward assign kar (sirf train mode)
    if (mode === 'train') {
      let reward = 1.0;             // +1 per frame alive — survival bonus
      if (justScored) {
        reward += 10.0;              // +10 pipe pass kiya — strong positive signal
      }
      if (gameOver) {
        reward = -10.0;              // -10 death penalty — galat kiya toh saza
      }
      savedRewards.push(reward);
    }
  }

  // ============================================================
  // RENDERING
  // ============================================================

  // --- dark sky gradient background ---
  function drawBackground(gw, gh) {
    const grad = ctx.createLinearGradient(0, 0, 0, gh);
    grad.addColorStop(0, '#0a0a2e');     // dark blue top
    grad.addColorStop(0.6, '#0f1538');   // mid
    grad.addColorStop(1, '#0a1020');     // darker bottom
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, gw, gh);
  }

  // --- zameen draw kar — brown strip with grass ---
  function drawGround(gw) {
    const groundY = CANVAS_HEIGHT - GROUND_HEIGHT;
    ctx.fillStyle = '#2a1f14';
    ctx.fillRect(0, groundY, gw, GROUND_HEIGHT);
    // grass line — hari patti upar
    ctx.fillStyle = '#3a6b35';
    ctx.fillRect(0, groundY, gw, 3);
    // darker edge — neeche
    ctx.fillStyle = '#1a1208';
    ctx.fillRect(0, groundY + GROUND_HEIGHT - 2, gw, 2);
  }

  // --- pipes draw kar — green/teal rectangles with caps ---
  function drawPipes(gw) {
    for (let i = 0; i < pipes.length; i++) {
      const p = pipes[i];

      // --- Top pipe — upar se gapY tak ---
      const topH = p.gapY;
      if (topH > 0) {
        // pipe body
        ctx.fillStyle = '#1a7a5a';
        ctx.fillRect(p.x, 0, PIPE_WIDTH, topH);
        // darker border
        ctx.strokeStyle = '#0d4a38';
        ctx.lineWidth = 2;
        ctx.strokeRect(p.x, 0, PIPE_WIDTH, topH);
        // pipe cap — thoda wider, neeche
        ctx.fillStyle = '#22916a';
        ctx.fillRect(p.x - 3, topH - 18, PIPE_WIDTH + 6, 18);
        ctx.strokeStyle = '#0d4a38';
        ctx.strokeRect(p.x - 3, topH - 18, PIPE_WIDTH + 6, 18);
        // highlight strip — depth effect
        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        ctx.fillRect(p.x + 4, 0, 6, Math.max(0, topH - 18));
      }

      // --- Bottom pipe — gapY + pipeGap se neeche ---
      const playH = CANVAS_HEIGHT - GROUND_HEIGHT;
      const botY = p.gapY + pipeGap;
      const botH = playH - botY;
      if (botH > 0) {
        ctx.fillStyle = '#1a7a5a';
        ctx.fillRect(p.x, botY, PIPE_WIDTH, botH);
        ctx.strokeStyle = '#0d4a38';
        ctx.lineWidth = 2;
        ctx.strokeRect(p.x, botY, PIPE_WIDTH, botH);
        // pipe cap — upar
        ctx.fillStyle = '#22916a';
        ctx.fillRect(p.x - 3, botY, PIPE_WIDTH + 6, 18);
        ctx.strokeStyle = '#0d4a38';
        ctx.strokeRect(p.x - 3, botY, PIPE_WIDTH + 6, 18);
        // highlight
        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        ctx.fillRect(p.x + 4, botY + 18, 6, Math.max(0, botH - 18));
      }
    }
  }

  // --- bird draw kar — cute circle with eye, wing, beak ---
  function drawBird(gw) {
    const birdX = gw * BIRD_X_FRAC;

    // body — yellow-orange circle
    ctx.fillStyle = '#ffcc33';
    ctx.beginPath();
    ctx.arc(birdX, birdY, BIRD_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#cc8800';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // wing — choti ellipse, flapping animation
    const wingAngle = Math.sin(frameCount * 0.3) * 0.3;
    ctx.fillStyle = '#ffaa00';
    ctx.beginPath();
    ctx.ellipse(birdX - 4, birdY + 2, 7, 4, wingAngle, 0, Math.PI * 2);
    ctx.fill();

    // eye — white circle with black pupil
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(birdX + 4, birdY - 3, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#000000';
    ctx.beginPath();
    ctx.arc(birdX + 5, birdY - 3, 1.8, 0, Math.PI * 2);
    ctx.fill();

    // beak — chhota orange triangle
    ctx.fillStyle = '#ff6633';
    ctx.beginPath();
    ctx.moveTo(birdX + BIRD_RADIUS, birdY - 1);
    ctx.lineTo(birdX + BIRD_RADIUS + 6, birdY + 2);
    ctx.lineTo(birdX + BIRD_RADIUS, birdY + 4);
    ctx.closePath();
    ctx.fill();
  }

  // --- score display — game area mein bada number ---
  function drawScore(gw) {
    ctx.font = 'bold 28px "JetBrains Mono",monospace';
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.textAlign = 'center';
    ctx.fillText('' + score, gw / 2, 40);

    // mode indicator neeche
    ctx.font = '11px "JetBrains Mono",monospace';
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.6)';
    if (mode === 'train') {
      ctx.fillText('TRAINING (Ep ' + episode + ')', gw / 2, 58);
    } else if (mode === 'watch') {
      ctx.fillText('AI PLAYING', gw / 2, 58);
    } else {
      ctx.fillText('YOUR TURN \u2014 Space/Tap to flap', gw / 2, 58);
    }
  }

  // --- Stats panel — right 35% mein stats, plot, network viz ---
  function drawStatsPanel(gw) {
    const sx = gw + 1;
    const sw = canvasW - gw;
    const padX = 12;
    const padY = 12;

    // panel background — thoda alag shade
    ctx.fillStyle = '#080818';
    ctx.fillRect(sx, 0, sw, CANVAS_HEIGHT);

    // divider line — game aur stats ke beech
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sx, 0);
    ctx.lineTo(sx, CANVAS_HEIGHT);
    ctx.stroke();

    let y = padY + 4;
    const leftX = sx + padX;
    const rightX = sx + sw - padX;

    // --- Stats text rows ---
    ctx.font = '11px "JetBrains Mono",monospace';

    // Episode
    ctx.textAlign = 'left';
    ctx.fillStyle = '#888';
    ctx.fillText('Episode', leftX, y);
    ctx.textAlign = 'right';
    ctx.fillStyle = ACCENT;
    ctx.fillText('' + episode, rightX, y);
    y += 18;

    // Current Score
    ctx.textAlign = 'left';
    ctx.fillStyle = '#888';
    ctx.fillText('Score', leftX, y);
    ctx.textAlign = 'right';
    ctx.fillStyle = ACCENT;
    ctx.fillText('' + score, rightX, y);
    y += 18;

    // Best Score
    ctx.textAlign = 'left';
    ctx.fillStyle = '#888';
    ctx.fillText('Best', leftX, y);
    ctx.textAlign = 'right';
    ctx.fillStyle = bestScore > 5 ? '#4aff8f' : ACCENT;
    ctx.fillText('' + bestScore, rightX, y);
    y += 18;

    // Alive frames
    ctx.textAlign = 'left';
    ctx.fillStyle = '#888';
    ctx.fillText('Alive', leftX, y);
    ctx.textAlign = 'right';
    ctx.fillStyle = ACCENT;
    ctx.fillText('' + frameCount, rightX, y);
    y += 24;

    // --- Reward Plot — score history ka chart ---
    ctx.textAlign = 'left';
    ctx.fillStyle = '#666';
    ctx.font = '10px "JetBrains Mono",monospace';
    ctx.fillText('Score History (last ' + MAX_HISTORY + ')', leftX, y);
    y += 8;

    const plotX = leftX;
    const plotW = sw - padX * 2;
    const plotH = 70;
    const plotY = y;

    // plot background box
    ctx.fillStyle = 'rgba(255,255,255,0.03)';
    ctx.fillRect(plotX, plotY, plotW, plotH);
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.15)';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(plotX, plotY, plotW, plotH);

    if (rewardHistory.length >= 2) {
      const maxR = Math.max(1, Math.max(...rewardHistory));
      const n = rewardHistory.length;

      // thin bars — individual episodes
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.2)';
      const barW = Math.max(1, plotW / n);
      for (let i = 0; i < n; i++) {
        const bh = (rewardHistory[i] / maxR) * plotH;
        ctx.fillRect(plotX + i * barW, plotY + plotH - bh, Math.max(0.5, barW - 0.5), bh);
      }

      // rolling average line — smooth trend
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const windowSize = Math.min(10, Math.floor(n / 2));
      for (let i = 0; i < n; i++) {
        let sum = 0, count = 0;
        for (let j = Math.max(0, i - windowSize); j <= i; j++) {
          sum += rewardHistory[j];
          count++;
        }
        const avg = sum / count;
        const px = plotX + (i / Math.max(n - 1, 1)) * plotW;
        const py = plotY + plotH - (avg / maxR) * plotH;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // max value label
      ctx.font = '9px "JetBrains Mono",monospace';
      ctx.fillStyle = '#555';
      ctx.textAlign = 'right';
      ctx.fillText('' + maxR, plotX + plotW, plotY + 10);
    } else {
      // abhi data nahi hai
      ctx.font = '10px "JetBrains Mono",monospace';
      ctx.fillStyle = '#444';
      ctx.textAlign = 'center';
      ctx.fillText('training shuru hone do...', sx + sw / 2, plotY + plotH / 2 + 4);
    }

    y += plotH + 16;

    // --- Neural Network Visualization ---
    ctx.textAlign = 'left';
    ctx.fillStyle = '#666';
    ctx.font = '10px "JetBrains Mono",monospace';
    ctx.fillText('Policy Network', leftX, y);
    y += 10;

    drawNetworkViz(leftX, y, plotW, Math.min(140, CANVAS_HEIGHT - y - 10));
  }

  // --- Network visualization — 3-layer network diagram ---
  function drawNetworkViz(nx, ny, nw, nh) {
    if (nh < 30 || nw < 50) return; // bahut chhota space hai, skip

    // 3 columns: input(5) -> hidden(subset) -> output(1)
    const layerX = [nx + 20, nx + nw / 2, nx + nw - 20];

    // input layer — 5 nodes
    const inputLabels = ['y', 'vy', 'dx', 'gt', 'gb'];
    const inputY = [];
    const inputSpacing = Math.min(24, (nh - 10) / Math.max(INPUT_SIZE - 1, 1));
    const inputStartY = ny + (nh - inputSpacing * (INPUT_SIZE - 1)) / 2;
    for (let i = 0; i < INPUT_SIZE; i++) {
      inputY.push(inputStartY + i * inputSpacing);
    }

    // hidden layer — max 8 dikhao, warna cluttered lagega
    const maxHiddenShow = 8;
    const hiddenShow = Math.min(HIDDEN_SIZE, maxHiddenShow);
    const hiddenY = [];
    const hiddenSpacing = Math.min(16, (nh - 10) / Math.max(hiddenShow - 1, 1));
    const hiddenStartY = ny + (nh - hiddenSpacing * (hiddenShow - 1)) / 2;
    for (let i = 0; i < hiddenShow; i++) {
      hiddenY.push(hiddenStartY + i * hiddenSpacing);
    }

    // output — 1 node, center mein
    const outputY = [ny + nh / 2];

    // --- Connections draw kar (pehle, taaki nodes upar aayein) ---

    // input -> hidden
    for (let i = 0; i < INPUT_SIZE; i++) {
      for (let j = 0; j < hiddenShow; j++) {
        const w = W1[i][j];
        const mag = Math.min(Math.abs(w) * 2, 1);
        const alpha = mag * 0.5;
        if (alpha < 0.05) continue; // tiny weights skip
        ctx.strokeStyle = w > 0
          ? 'rgba(' + ACCENT_RGB + ',' + alpha.toFixed(2) + ')'
          : 'rgba(255,80,80,' + alpha.toFixed(2) + ')';
        ctx.lineWidth = mag * 2;
        ctx.beginPath();
        ctx.moveTo(layerX[0], inputY[i]);
        ctx.lineTo(layerX[1], hiddenY[j]);
        ctx.stroke();
      }
    }

    // hidden -> output
    for (let j = 0; j < hiddenShow; j++) {
      const w = W2[j][0];
      const mag = Math.min(Math.abs(w) * 2, 1);
      const alpha = mag * 0.6;
      if (alpha < 0.05) continue;
      ctx.strokeStyle = w > 0
        ? 'rgba(' + ACCENT_RGB + ',' + alpha.toFixed(2) + ')'
        : 'rgba(255,80,80,' + alpha.toFixed(2) + ')';
      ctx.lineWidth = mag * 2.5;
      ctx.beginPath();
      ctx.moveTo(layerX[1], hiddenY[j]);
      ctx.lineTo(layerX[2], outputY[0]);
      ctx.stroke();
    }

    // --- Nodes draw kar ---
    const nodeR = 5;

    // input nodes — brightness = activation value ki magnitude
    for (let i = 0; i < INPUT_SIZE; i++) {
      const val = lastInputs[i];
      const brightness = Math.min(1, Math.abs(val));
      ctx.fillStyle = val >= 0
        ? 'rgba(' + ACCENT_RGB + ',' + (0.3 + brightness * 0.7).toFixed(2) + ')'
        : 'rgba(255,130,80,' + (0.3 + brightness * 0.7).toFixed(2) + ')';
      ctx.beginPath();
      ctx.arc(layerX[0], inputY[i], nodeR, 0, Math.PI * 2);
      ctx.fill();

      // label — left side mein
      ctx.font = '8px "JetBrains Mono",monospace';
      ctx.fillStyle = '#555';
      ctx.textAlign = 'right';
      ctx.fillText(inputLabels[i], layerX[0] - nodeR - 3, inputY[i] + 3);
    }

    // hidden nodes — subset dikhao
    for (let j = 0; j < hiddenShow; j++) {
      const val = lastHidden[j];
      const brightness = Math.min(1, Math.abs(val));
      ctx.fillStyle = val >= 0
        ? 'rgba(' + ACCENT_RGB + ',' + (0.2 + brightness * 0.8).toFixed(2) + ')'
        : 'rgba(255,130,80,' + (0.2 + brightness * 0.8).toFixed(2) + ')';
      ctx.beginPath();
      ctx.arc(layerX[1], hiddenY[j], nodeR - 1, 0, Math.PI * 2);
      ctx.fill();
    }

    // agar zyada hidden nodes hain toh "..." dikhao
    if (HIDDEN_SIZE > maxHiddenShow) {
      ctx.fillStyle = '#555';
      ctx.font = '10px "JetBrains Mono",monospace';
      ctx.textAlign = 'center';
      ctx.fillText('...', layerX[1], hiddenY[hiddenShow - 1] + 16);
    }

    // output node — bada, flap probability ke hisaab se bright
    const outBrightness = lastOutput;
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',' + (0.3 + outBrightness * 0.7).toFixed(2) + ')';
    ctx.beginPath();
    ctx.arc(layerX[2], outputY[0], nodeR + 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // output label — "flap" aur percentage
    ctx.font = '8px "JetBrains Mono",monospace';
    ctx.fillStyle = '#888';
    ctx.textAlign = 'left';
    ctx.fillText('flap', layerX[2] + nodeR + 5, outputY[0] + 3);
    ctx.fillStyle = ACCENT;
    ctx.fillText((lastOutput * 100).toFixed(0) + '%', layerX[2] + nodeR + 5, outputY[0] + 14);
  }

  // --- Main render function — full canvas draw ---
  function render() {
    const currentDpr = window.devicePixelRatio || 1;
    ctx.setTransform(currentDpr, 0, 0, currentDpr, 0, 0);

    const gw = canvasW * GAME_FRAC;

    // game area clip — left portion mein hi draw kar
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, gw, CANVAS_HEIGHT);
    ctx.clip();

    drawBackground(gw, CANVAS_HEIGHT);
    drawGround(gw);
    drawPipes(gw);
    drawBird(gw);
    drawScore(gw);

    // game over overlay — sirf watch/play mode mein dikhao
    if (gameOver && (mode === 'play' || mode === 'watch')) {
      ctx.font = 'bold 20px "JetBrains Mono",monospace';
      ctx.fillStyle = 'rgba(255,80,80,0.8)';
      ctx.textAlign = 'center';
      ctx.fillText('GAME OVER', gw / 2, CANVAS_HEIGHT / 2);
      ctx.font = '12px "JetBrains Mono",monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      if (mode === 'play') {
        ctx.fillText('Space/Tap to restart', gw / 2, CANVAS_HEIGHT / 2 + 22);
      }
    }

    ctx.restore();

    // stats panel — right side mein
    drawStatsPanel(gw);

    // transform reset kar dena
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  // ============================================================
  // CANVAS & INPUT HANDLING
  // ============================================================

  // --- Canvas resize — DPR aware ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;

    canvas.width = Math.floor(containerWidth * dpr);
    canvas.height = Math.floor(CANVAS_HEIGHT * dpr);
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';
  }

  // --- Manual play input — Space/click/tap se flap ---
  function handleFlap() {
    if (mode !== 'play') return;
    if (gameOver) {
      resetGame();
      return;
    }
    flapRequested = true;
  }

  // keyboard — Space ya ArrowUp
  document.addEventListener('keydown', (e) => {
    if (!isVisible) return;
    if (e.code === 'Space' || e.key === 'ArrowUp') {
      e.preventDefault();
      handleFlap();
    }
  });

  // canvas click/tap
  canvas.addEventListener('click', (e) => {
    e.preventDefault();
    handleFlap();
  });
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    handleFlap();
  }, { passive: false });

  // ============================================================
  // ANIMATION LOOP
  // ============================================================

  function loop() {
    // lab pause: sirf active sim animate hoga
    if (window.__labPaused && window.__labPaused !== container.id) {
      animationId = null;
      return;
    }

    if (!isVisible) {
      animationId = null;
      return;
    }

    // mode ke hisaab se steps chalaao
    if (mode === 'train') {
      // fast training — multiple steps per frame
      for (let i = 0; i < trainSpeed; i++) {
        runOneStep();
      }
    } else if (mode === 'watch') {
      // AI normal speed — ek step per frame
      runOneStep();
    } else if (mode === 'play') {
      // manual mode — user ka flap request process kar
      if (!gameOver) {
        gameStep(flapRequested);
        flapRequested = false;
      }
    }

    render();
    animationId = requestAnimationFrame(loop);
  }

  // --- Start/Stop helpers ---
  function startAnimation() {
    if (isVisible && !animationId) {
      loop();
    }
  }

  function stopAnimation() {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // --- IntersectionObserver — sirf visible hone pe animate, CPU bachao ---
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        const wasVisible = isVisible;
        isVisible = entry.isIntersecting;
        if (isVisible && !wasVisible) {
          startAnimation();
        } else if (!isVisible && wasVisible) {
          stopAnimation();
        }
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);

  // lab:resume — jab dusri sim pause hata de toh wapas chalu kar
  document.addEventListener('lab:resume', () => {
    if (isVisible && !animationId) loop();
  });

  // tab visibility — tab switch pe band/chalu
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) {
        isVisible = true;
        startAnimation();
      }
    }
  });

  // resize pe canvas update karo
  window.addEventListener('resize', () => {
    resizeCanvas();
  });

  // ============================================================
  // INITIALIZATION — sab setup karke training shuru
  // ============================================================
  initWeights();
  resizeCanvas();
  resetGame();
  startAnimation();
}
