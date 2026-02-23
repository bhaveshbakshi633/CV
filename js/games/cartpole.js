// ============================================================
// Cart-Pole Balancer — REINFORCE policy gradient se pole balance karna seekhega
// Classic control problem hai ye — RL ka hello world samajh le
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
  const FORCE_MAG = 10.0; // push force magnitude
  const DT = 0.02; // euler integration timestep
  const MAX_STEPS = 200; // ek episode mein max steps
  const TRACK_LIMIT = 2.4; // cart itna door ja sakta hai
  const ANGLE_LIMIT = 12 * Math.PI / 180; // pole itna tilt hua toh game over

  // --- Canvas dimensions ---
  const MAIN_HEIGHT = 300;
  const GRAPH_HEIGHT = 80;

  // --- Neural Network parameters ---
  // chhota sa network — 4 inputs, 8 hidden, 2 outputs
  const INPUT_SIZE = 4;
  const HIDDEN_SIZE = 8;
  const OUTPUT_SIZE = 2;
  const LEARNING_RATE = 0.01;

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

  // speed control
  let speedMultiplier = 1;
  let manualMode = false;
  let manualAction = -1; // -1 = no action, 0 = left, 1 = right

  // animation state
  let animationId = null;
  let isVisible = false;

  // --- Neural Network weights ---
  // Xavier initialization — random nahi, proper scale se initialize kar
  let W1, b1, W2, b2;

  // saved log probabilities aur rewards — policy gradient ke liye
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

  // --- Sigmoid activation ---
  function sigmoid(x) {
    // overflow protection — bada negative aaya toh 0 return kar
    if (x < -500) return 0;
    if (x > 500) return 1;
    return 1.0 / (1.0 + Math.exp(-x));
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
    // hidden layer: sigmoid(W1^T * state + b1)
    const hidden = new Array(HIDDEN_SIZE);
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      let sum = b1[j];
      for (let i = 0; i < INPUT_SIZE; i++) {
        sum += state[i] * W1[i][j];
      }
      hidden[j] = sigmoid(sum);
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

  // --- REINFORCE update — episode khatam hone pe weights update kar ---
  function updatePolicy() {
    if (savedLogProbs.length === 0) return;

    // returns calculate kar — discount factor 0.99
    const gamma = 0.99;
    const returns = new Array(savedLogProbs.length);
    let R = 0;
    for (let t = savedLogProbs.length - 1; t >= 0; t--) {
      R = savedRewards[t] + gamma * R;
      returns[t] = R;
    }

    // normalize kar returns ko — training stable rehti hai
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const std = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length) + 1e-8;
    for (let i = 0; i < returns.length; i++) {
      returns[i] = (returns[i] - mean) / std;
    }

    // ab har step ke liye gradient calculate kar aur weights update kar
    // REINFORCE: delta_W = lr * return_t * grad(log_pi(a|s))
    for (let t = 0; t < savedLogProbs.length; t++) {
      const state = episodeLog[t].state;
      const action = episodeLog[t].action;
      const advantage = returns[t];

      // forward pass fir se kar — hidden activations chahiye gradient ke liye
      const { hidden, probs } = forward(state);

      // gradient of log probability w.r.t. logits
      // softmax + cross entropy ka gradient: (one_hot - probs)
      const dLogits = probs.map((p, i) => (i === action ? 1 - p : -p));

      // scale by advantage aur learning rate
      const lr = LEARNING_RATE;

      // W2 update: hidden^T * dLogits * advantage
      for (let i = 0; i < HIDDEN_SIZE; i++) {
        for (let j = 0; j < OUTPUT_SIZE; j++) {
          W2[i][j] += lr * advantage * hidden[i] * dLogits[j];
        }
      }
      // b2 update
      for (let j = 0; j < OUTPUT_SIZE; j++) {
        b2[j] += lr * advantage * dLogits[j];
      }

      // backprop to hidden layer
      const dHidden = new Array(HIDDEN_SIZE).fill(0);
      for (let i = 0; i < HIDDEN_SIZE; i++) {
        for (let j = 0; j < OUTPUT_SIZE; j++) {
          dHidden[i] += dLogits[j] * W2[i][j];
        }
        // sigmoid derivative: h * (1 - h)
        dHidden[i] *= hidden[i] * (1 - hidden[i]);
      }

      // W1 update
      for (let i = 0; i < INPUT_SIZE; i++) {
        for (let j = 0; j < HIDDEN_SIZE; j++) {
          W1[i][j] += lr * advantage * state[i] * dHidden[j];
        }
      }
      // b1 update
      for (let j = 0; j < HIDDEN_SIZE; j++) {
        b1[j] += lr * advantage * dHidden[j];
      }
    }
  }

  // --- Environment reset ---
  function resetEnv() {
    // chhoti random initial conditions — bilkul center se shuru nahi karna
    cartX = (Math.random() - 0.5) * 0.1;
    cartVel = (Math.random() - 0.5) * 0.1;
    poleAngle = (Math.random() - 0.5) * 0.1;
    poleAngVel = (Math.random() - 0.5) * 0.1;
    stepCount = 0;
    episodeDone = false;
    savedLogProbs = [];
    savedRewards = [];
    episodeLog = [];
  }

  // --- Physics step — Euler integration ---
  function physicsStep(action) {
    const force = action === 1 ? FORCE_MAG : -FORCE_MAG;
    const cosA = Math.cos(poleAngle);
    const sinA = Math.sin(poleAngle);

    // cart-pole dynamics equations — Lagrangian mechanics se derive ki hain
    const temp = (force + POLE_MASS_LENGTH * poleAngVel * poleAngVel * sinA) / TOTAL_MASS;
    const angAcc = (GRAVITY * sinA - cosA * temp) /
      (POLE_HALF_LEN * (4.0 / 3.0 - POLE_MASS * cosA * cosA / TOTAL_MASS));
    const cartAcc = temp - POLE_MASS_LENGTH * angAcc * cosA / TOTAL_MASS;

    // Euler integration — simple but works for this
    cartX += cartVel * DT;
    cartVel += cartAcc * DT;
    poleAngle += poleAngVel * DT;
    poleAngVel += angAcc * DT;

    stepCount++;

    // episode khatam check kar — limit cross ki ya steps khatam
    if (Math.abs(cartX) > TRACK_LIMIT || Math.abs(poleAngle) > ANGLE_LIMIT || stepCount >= MAX_STEPS) {
      episodeDone = true;
    }

    return episodeDone ? 0 : 1; // reward: 1 har step alive rehne pe
  }

  // --- DOM structure banate hain ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

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

  // reset button — naye sirey se training shuru
  makeButton('Reset Training', () => {
    initWeights();
    episode = 0;
    bestReward = 0;
    rewardHistory = [];
    resetEnv();
  });

  // speed selector
  const speedLabel = document.createElement('span');
  speedLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace;';
  speedLabel.textContent = 'Speed:';
  controlsDiv.appendChild(speedLabel);

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

  // manual mode toggle
  const manualBtn = makeButton('Manual Mode: OFF', () => {
    manualMode = !manualMode;
    manualBtn.textContent = 'Manual Mode: ' + (manualMode ? 'ON' : 'OFF');
    if (manualMode) {
      manualBtn.style.borderColor = 'rgba(74,158,255,0.6)';
      manualBtn.style.color = '#4a9eff';
    } else {
      manualBtn.style.borderColor = 'rgba(74,158,255,0.25)';
      manualBtn.style.color = '#b0b0b0';
    }
  });

  // keyboard controls — manual mode ke liye
  document.addEventListener('keydown', (e) => {
    if (!manualMode || !isVisible) return;
    if (e.key === 'ArrowLeft') { manualAction = 0; e.preventDefault(); }
    if (e.key === 'ArrowRight') { manualAction = 1; e.preventDefault(); }
  });
  document.addEventListener('keyup', (e) => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') manualAction = -1;
  });

  // --- Canvas resize handling ---
  function resizeCanvases() {
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const dpr = window.devicePixelRatio || 1;

    mainCanvas.width = w * dpr;
    mainCanvas.height = MAIN_HEIGHT * dpr;
    graphCanvas.width = w * dpr;
    graphCanvas.height = GRAPH_HEIGHT * dpr;

    mainCanvas.style.width = w + 'px';
    graphCanvas.style.width = w + 'px';
  }

  // --- Main canvas rendering ---
  function drawMain() {
    const ctx = mainCanvas.getContext('2d');
    const w = mainCanvas.width;
    const h = mainCanvas.height;
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, w, h);

    // coordinate system: center bottom mein origin rakh, scale kar
    const centerX = w / 2;
    const groundY = h * 0.8; // zameen ka level
    const scale = (w / 2) / (TRACK_LIMIT * 1.3); // track ko canvas mein fit kar

    // track/rail draw kar — neeche ek subtle line
    ctx.strokeStyle = 'rgba(74,158,255,0.15)';
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.moveTo(centerX - TRACK_LIMIT * scale, groundY);
    ctx.lineTo(centerX + TRACK_LIMIT * scale, groundY);
    ctx.stroke();

    // track limits — dono taraf chhoti vertical lines
    ctx.strokeStyle = 'rgba(255,100,100,0.3)';
    ctx.lineWidth = 1.5 * dpr;
    [-TRACK_LIMIT, TRACK_LIMIT].forEach(lim => {
      const x = centerX + lim * scale;
      ctx.beginPath();
      ctx.moveTo(x, groundY - 10 * dpr);
      ctx.lineTo(x, groundY + 10 * dpr);
      ctx.stroke();
    });

    // cart draw kar — blue rectangle
    const cartW = 60 * dpr;
    const cartH = 30 * dpr;
    const cartScreenX = centerX + cartX * scale;
    const cartScreenY = groundY;

    ctx.fillStyle = 'rgba(74,158,255,0.85)';
    ctx.strokeStyle = 'rgba(74,158,255,0.4)';
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    ctx.roundRect(cartScreenX - cartW / 2, cartScreenY - cartH, cartW, cartH, 4 * dpr);
    ctx.fill();
    ctx.stroke();

    // wheels — do chhote circles
    ctx.fillStyle = 'rgba(74,158,255,0.5)';
    const wheelR = 5 * dpr;
    [-15, 15].forEach(offset => {
      ctx.beginPath();
      ctx.arc(cartScreenX + offset * dpr, cartScreenY, wheelR, 0, Math.PI * 2);
      ctx.fill();
    });

    // pole draw kar — cart ke top center se
    const polePixelLen = 120 * dpr;
    const poleStartX = cartScreenX;
    const poleStartY = cartScreenY - cartH;
    const poleEndX = poleStartX + Math.sin(poleAngle) * polePixelLen;
    const poleEndY = poleStartY - Math.cos(poleAngle) * polePixelLen;

    // pole shadow for depth effect
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 8 * dpr;
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
    ctx.lineWidth = 4 * dpr;
    ctx.beginPath();
    ctx.moveTo(poleStartX, poleStartY);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();

    // pivot point — chhota circle jahan pole cart se juda hai
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.beginPath();
    ctx.arc(poleStartX, poleStartY, 4 * dpr, 0, Math.PI * 2);
    ctx.fill();

    // pole tip — glow effect
    ctx.fillStyle = 'rgba(' + poleR + ',' + poleG + ',100,0.6)';
    ctx.beginPath();
    ctx.arc(poleEndX, poleEndY, 3 * dpr, 0, Math.PI * 2);
    ctx.fill();

    // status text — manual mode mein instruction dikhao
    ctx.font = (11 * dpr) + 'px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.6)';
    ctx.textAlign = 'left';
    if (manualMode) {
      ctx.fillText('Arrow keys se control kar', 10 * dpr, 20 * dpr);
    } else {
      ctx.fillText('RL agent training...', 10 * dpr, 20 * dpr);
    }
  }

  // --- Graph canvas — reward history plot ---
  function drawGraph() {
    const ctx = graphCanvas.getContext('2d');
    const w = graphCanvas.width;
    const h = graphCanvas.height;
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, w, h);

    if (rewardHistory.length < 2) {
      ctx.font = (11 * dpr) + 'px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.4)';
      ctx.textAlign = 'center';
      ctx.fillText('Reward graph — training shuru hone do...', w / 2, h / 2);
      return;
    }

    const padL = 40 * dpr;
    const padR = 10 * dpr;
    const padT = 10 * dpr;
    const padB = 15 * dpr;
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

    // y-axis labels
    ctx.font = (9 * dpr) + 'px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.4)';
    ctx.textAlign = 'right';
    [0, 50, 100, 150, 200].forEach(v => {
      const yy = padT + plotH * (1 - v / maxReward);
      ctx.fillText(v.toString(), padL - 5 * dpr, yy + 3 * dpr);
    });

    // reward line draw kar
    ctx.strokeStyle = 'rgba(74,158,255,0.8)';
    ctx.lineWidth = 2 * dpr;
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
      ctx.arc(lastX, lastY, 3 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // --- Stats update — DOM safe way, no innerHTML ---
  function updateStats() {
    const lastReward = rewardHistory.length > 0 ? rewardHistory[rewardHistory.length - 1] : 0;

    // pehle saaf kar
    while (statsDiv.firstChild) statsDiv.removeChild(statsDiv.firstChild);

    // helper — label:value span pair bana
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
    addStat('Reward: ', lastReward, '#4a9eff');
    addStat('Best: ', bestReward, '#4aff8f');
  }

  // --- Training loop — ek episode ka ek step ---
  function trainingStep() {
    if (episodeDone) {
      // episode khatam — policy update kar aur naya episode shuru kar
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

    // current state bana — ye 4 values neural net ko milenge
    const state = [cartX, cartVel, poleAngle, poleAngVel];

    let action;
    if (manualMode) {
      // manual mode — arrow keys se control
      action = manualAction === 0 ? 0 : 1;
    } else {
      // neural network se action lo
      const result = selectAction(state);
      action = result.action;

      // REINFORCE ke liye data store kar
      savedLogProbs.push(result.logProb);
      episodeLog.push({ state: state.slice(), action });
    }

    // physics step chala — reward milega
    const reward = physicsStep(action);
    savedRewards.push(reward);
  }

  // --- Main animation loop ---
  function animate() {
    if (!isVisible) {
      animationId = requestAnimationFrame(animate);
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

  // --- IntersectionObserver — sirf visible hone pe animate kar, CPU bach jaayega ---
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        isVisible = entry.isIntersecting;
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

  // animation start kar — ab training chalegi
  animate();
}
