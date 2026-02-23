// =============================================================
// Gridworld Q-Learning Demo — Robotics Portfolio ke liye
// Agent seekhta hai grid mein navigate karna using Q-Learning
// =============================================================

const GRID_SIZE = 6;
const NUM_LAVA = 5;
const ACTIONS = ['up', 'down', 'left', 'right'];
const ACTION_DELTAS = {
  up: [0, -1],
  down: [0, 1],
  left: [-1, 0],
  right: [1, 0],
};

// Q-Learning hyperparameters — ye sab tuning ke baad set kiye hain
const ALPHA = 0.1;         // learning rate — zyada kiya toh oscillate karega
const GAMMA = 0.95;        // discount factor — future reward kitna important hai
const EPSILON_START = 0.3; // exploration shuru mein zyada — duniya explore kar
const EPSILON_MIN = 0.05;  // minimum exploration — thoda random rakh warna stuck ho jaayega
const EPSILON_DECAY = 0.9995; // har step pe epsilon kam hota hai

// Rewards — simple rakh rahe hain
const REWARD_GOAL = 10;
const REWARD_LAVA = -10;
const REWARD_STEP = -0.1;  // har step ka penalty — taki shortest path seekhe

// Rendering constants
const ARROW_CHARS = { up: '\u2191', down: '\u2193', left: '\u2190', right: '\u2192' };
const SPEED_OPTIONS = [1, 5, 20];

// ===================== Q-Table Class =====================
// State-action values store karta hai — agent ka "brain" samajh le
class QTable {
  constructor() {
    this.table = {};
  }

  // state key bana — "x,y" format mein
  _key(x, y) {
    return `${x},${y}`;
  }

  // Q-value nikal — agar pehle nahi dekha toh 0 return kar
  get(x, y, action) {
    const key = this._key(x, y);
    if (!this.table[key]) return 0;
    return this.table[key][action] || 0;
  }

  // Q-value set kar — yahi toh learning hai BC
  set(x, y, action, value) {
    const key = this._key(x, y);
    if (!this.table[key]) {
      this.table[key] = {};
    }
    this.table[key][action] = value;
  }

  // best action nikal kisi state ke liye — greedy policy
  bestAction(x, y) {
    let bestA = ACTIONS[0];
    let bestVal = -Infinity;
    for (const a of ACTIONS) {
      const v = this.get(x, y, a);
      if (v > bestVal) {
        bestVal = v;
        bestA = a;
      }
    }
    return bestA;
  }

  // max Q-value nikal — Bellman equation mein chahiye
  maxQ(x, y) {
    let maxVal = -Infinity;
    for (const a of ACTIONS) {
      const v = this.get(x, y, a);
      if (v > maxVal) maxVal = v;
    }
    return maxVal === -Infinity ? 0 : maxVal;
  }

  // saaf kar do — nayi duniya, naya brain
  clear() {
    this.table = {};
  }
}

// ===================== Gridworld Environment =====================
class GridworldEnv {
  constructor() {
    this.goal = { x: 5, y: 0 }; // top-right corner pe goal rakh
    this.lava = [];
    this.agentX = 0;
    this.agentY = 5; // bottom-left se shuru
    this._randomizeLava();
  }

  // lava cells random jagah rakh — goal aur starting position chhodke
  _randomizeLava() {
    this.lava = [];
    const occupied = new Set();
    occupied.add(`${this.goal.x},${this.goal.y}`);
    occupied.add(`0,5`); // default agent start

    while (this.lava.length < NUM_LAVA) {
      const x = Math.floor(Math.random() * GRID_SIZE);
      const y = Math.floor(Math.random() * GRID_SIZE);
      const key = `${x},${y}`;
      if (!occupied.has(key)) {
        occupied.add(key);
        this.lava.push({ x, y });
      }
    }
  }

  // check kar ye cell lava hai ya nahi
  isLava(x, y) {
    return this.lava.some(l => l.x === x && l.y === y);
  }

  // check kar ye cell goal hai ya nahi
  isGoal(x, y) {
    return x === this.goal.x && y === this.goal.y;
  }

  // agent ko kisi cell pe place kar — click se
  placeAgent(x, y) {
    if (!this.isLava(x, y) && !this.isGoal(x, y)) {
      this.agentX = x;
      this.agentY = y;
    }
  }

  // ek step le — action do, new state + reward lo
  // boundary check bhi — wall se takraaya toh same jagah reh
  step(action) {
    const [dx, dy] = ACTION_DELTAS[action];
    let nx = this.agentX + dx;
    let ny = this.agentY + dy;

    // boundary check — grid ke bahar nahi jaane denge
    if (nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE) {
      nx = this.agentX;
      ny = this.agentY;
    }

    this.agentX = nx;
    this.agentY = ny;

    // reward calculate kar
    if (this.isGoal(nx, ny)) {
      return { reward: REWARD_GOAL, done: true };
    }
    if (this.isLava(nx, ny)) {
      return { reward: REWARD_LAVA, done: true };
    }
    return { reward: REWARD_STEP, done: false };
  }

  // agent ko random empty cell pe reset kar — naya episode shuru
  resetAgent() {
    let x, y;
    do {
      x = Math.floor(Math.random() * GRID_SIZE);
      y = Math.floor(Math.random() * GRID_SIZE);
    } while (this.isLava(x, y) || this.isGoal(x, y));
    this.agentX = x;
    this.agentY = y;
  }

  // poora environment reset kar — lava bhi nayi jagah
  fullReset() {
    this._randomizeLava();
    this.resetAgent();
  }
}

// ===================== Renderer =====================
// Canvas pe sab draw karta hai — grid, Q-values, arrows, agent, sab
class GridworldRenderer {
  constructor(canvas, env, qTable) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.env = env;
    this.qTable = qTable;

    // cell size calculate kar canvas size se
    this.cellW = 0;
    this.cellH = 0;
    this.offsetX = 0;
    this.offsetY = 0;
    this._recalcLayout();
  }

  // layout recalculate kar — resize pe bhi kaam aayega
  _recalcLayout() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    // square cells chahiye — chhota dimension use kar
    const maxCellW = Math.floor(w / GRID_SIZE);
    const maxCellH = Math.floor(h / GRID_SIZE);
    const cellSize = Math.min(maxCellW, maxCellH);
    this.cellW = cellSize;
    this.cellH = cellSize;
    // center kar grid ko canvas mein
    this.offsetX = Math.floor((w - cellSize * GRID_SIZE) / 2);
    this.offsetY = Math.floor((h - cellSize * GRID_SIZE) / 2);
  }

  // Q-value ko color mein map kar — red(low) -> yellow(mid) -> green(high)
  _qColor(qVal) {
    // Q-values ko [-10, 10] range mein clamp kar
    const clamped = Math.max(-10, Math.min(10, qVal));
    // normalize to [0, 1]
    const t = (clamped + 10) / 20;

    let r, g, b;
    if (t < 0.5) {
      // red -> yellow
      const s = t / 0.5;
      r = 180 + Math.floor(40 * s);
      g = Math.floor(180 * s);
      b = 40;
    } else {
      // yellow -> green
      const s = (t - 0.5) / 0.5;
      r = 220 - Math.floor(180 * s);
      g = 180 + Math.floor(40 * s);
      b = 40 + Math.floor(30 * s);
    }

    // dark theme — alpha low rakh taki subtle lage
    return `rgba(${r},${g},${b},0.25)`;
  }

  // pixel coords se grid coords nikal — click handling ke liye
  pixelToGrid(px, py) {
    const gx = Math.floor((px - this.offsetX) / this.cellW);
    const gy = Math.floor((py - this.offsetY) / this.cellH);
    if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
      return { x: gx, y: gy };
    }
    return null;
  }

  // poora frame draw kar
  draw() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // canvas saaf kar — dark background
    ctx.clearRect(0, 0, w, h);

    const cw = this.cellW;
    const ch = this.cellH;
    const ox = this.offsetX;
    const oy = this.offsetY;

    // har cell draw kar
    for (let x = 0; x < GRID_SIZE; x++) {
      for (let y = 0; y < GRID_SIZE; y++) {
        const px = ox + x * cw;
        const py = oy + y * ch;

        // cell background — Q-value based color
        if (this.env.isGoal(x, y)) {
          // goal cell — bright green
          ctx.fillStyle = 'rgba(0, 230, 118, 0.35)';
        } else if (this.env.isLava(x, y)) {
          // lava cell — angry red
          ctx.fillStyle = 'rgba(255, 61, 61, 0.3)';
        } else {
          // normal cell — Q-value se color
          const maxQ = this.qTable.maxQ(x, y);
          ctx.fillStyle = this._qColor(maxQ);
        }
        ctx.fillRect(px + 1, py + 1, cw - 2, ch - 2);

        // grid lines — subtle white
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 1;
        ctx.strokeRect(px, py, cw, ch);

        // cell content draw kar
        if (this.env.isGoal(x, y)) {
          // goal pe star draw kar
          this._drawStar(px + cw / 2, py + ch / 2, Math.min(cw, ch) * 0.3);
        } else if (this.env.isLava(x, y)) {
          // lava pe fire emoji / X draw kar
          ctx.fillStyle = '#ff3d3d';
          ctx.font = `bold ${Math.floor(ch * 0.45)}px JetBrains Mono, monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('\u2716', px + cw / 2, py + ch / 2);
        } else {
          // normal cell — best action ka arrow dikhao
          const bestA = this.qTable.bestAction(x, y);
          const maxQ = this.qTable.maxQ(x, y);

          // sirf arrow dikhao agar kuch seekha hai agent ne
          if (maxQ !== 0) {
            // arrow opacity Q magnitude se scale kar
            const intensity = Math.min(1, Math.abs(maxQ) / 5);
            ctx.fillStyle = `rgba(255,255,255,${0.15 + intensity * 0.65})`;
            ctx.font = `${Math.floor(ch * 0.35)}px JetBrains Mono, monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(ARROW_CHARS[bestA], px + cw / 2, py + ch / 2);
          }
        }
      }
    }

    // agent draw kar — circle with glow
    this._drawAgent();
  }

  // star draw kar — goal cell ke liye
  _drawStar(cx, cy, r) {
    const ctx = this.ctx;
    const spikes = 5;
    const outerR = r;
    const innerR = r * 0.45;
    ctx.beginPath();
    for (let i = 0; i < spikes * 2; i++) {
      const angle = (i * Math.PI) / spikes - Math.PI / 2;
      const radius = i % 2 === 0 ? outerR : innerR;
      const sx = cx + Math.cos(angle) * radius;
      const sy = cy + Math.sin(angle) * radius;
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.closePath();
    ctx.fillStyle = '#00e676';
    ctx.fill();
    // thoda glow effect
    ctx.shadowColor = '#00e676';
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  // agent draw kar — ye blue circle hai with glow
  _drawAgent() {
    const ctx = this.ctx;
    const cw = this.cellW;
    const ch = this.cellH;
    const cx = this.offsetX + this.env.agentX * cw + cw / 2;
    const cy = this.offsetY + this.env.agentY * ch + ch / 2;
    const r = Math.min(cw, ch) * 0.3;

    // glow effect — agent dikhna chahiye clearly
    ctx.shadowColor = '#448aff';
    ctx.shadowBlur = 12;

    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = '#448aff';
    ctx.fill();

    // inner highlight — 3D look ke liye
    ctx.beginPath();
    ctx.arc(cx - r * 0.2, cy - r * 0.2, r * 0.4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.fill();

    ctx.shadowBlur = 0;
  }
}

// ===================== Main Controller =====================
// Ye sab orchestrate karta hai — env, Q-table, renderer, controls
export function initGridworld() {
  const container = document.getElementById('gridworldContainer');
  if (!container) {
    // container nahi mila — kuch mat kar, silently return
    console.warn('gridworldContainer nahi mila DOM mein');
    return;
  }

  // -------- DOM elements bana --------

  // wrapper — controls + canvas saath mein
  const wrapper = document.createElement('div');
  wrapper.style.cssText = `
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    width: 100%;
    font-family: 'JetBrains Mono', monospace;
  `;

  // controls bar — top pe
  const controlsBar = document.createElement('div');
  controlsBar.style.cssText = `
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    width: 100%;
    padding: 0 4px;
  `;

  // button style helper
  const btnStyle = `
    background: rgba(255,255,255,0.08);
    color: #e0e0e0;
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 6px;
    padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    user-select: none;
    white-space: nowrap;
  `;

  // Reset button
  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset';
  resetBtn.style.cssText = btnStyle;
  resetBtn.addEventListener('mouseenter', () => {
    resetBtn.style.background = 'rgba(255,61,61,0.2)';
    resetBtn.style.borderColor = 'rgba(255,61,61,0.4)';
  });
  resetBtn.addEventListener('mouseleave', () => {
    resetBtn.style.background = 'rgba(255,255,255,0.08)';
    resetBtn.style.borderColor = 'rgba(255,255,255,0.15)';
  });

  // Speed toggle button
  const speedBtn = document.createElement('button');
  speedBtn.style.cssText = btnStyle;
  speedBtn.addEventListener('mouseenter', () => {
    speedBtn.style.background = 'rgba(68,138,255,0.2)';
    speedBtn.style.borderColor = 'rgba(68,138,255,0.4)';
  });
  speedBtn.addEventListener('mouseleave', () => {
    speedBtn.style.background = 'rgba(255,255,255,0.08)';
    speedBtn.style.borderColor = 'rgba(255,255,255,0.15)';
  });

  // Stats display — episode count + reward
  const statsDisplay = document.createElement('span');
  statsDisplay.style.cssText = `
    color: rgba(255,255,255,0.6);
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
  `;

  controlsBar.appendChild(resetBtn);
  controlsBar.appendChild(speedBtn);
  controlsBar.appendChild(statsDisplay);

  // Canvas — yahi pe sab dikhaayenge
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `
    display: block;
    max-width: 100%;
    border-radius: 8px;
    cursor: pointer;
  `;

  wrapper.appendChild(controlsBar);
  wrapper.appendChild(canvas);
  container.appendChild(wrapper);

  // -------- Canvas sizing --------
  // container width se canvas size decide kar
  function resizeCanvas() {
    const containerWidth = container.clientWidth;
    const maxHeight = 320;
    // square cells chahiye — width aur height mein se chhota le
    const size = Math.min(containerWidth, maxHeight);
    // high-DPI support — crisp rendering ke liye
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
    if (renderer) {
      renderer.cellW = 0; // force recalc
      renderer._recalcLayout();
    }
  }

  // -------- Init game state --------
  const env = new GridworldEnv();
  const qTable = new QTable();
  const renderer = new GridworldRenderer(canvas, env, qTable);

  let epsilon = EPSILON_START;
  let episodeCount = 0;
  let currentEpisodeReward = 0;
  let totalStepsInEpisode = 0;
  let speedIndex = 0; // 0 = 1x, 1 = 5x, 2 = 20x
  let isVisible = false;
  let animFrameId = null;
  let lastStepTime = 0;

  // speed update kar
  function updateSpeedLabel() {
    speedBtn.textContent = `Speed: ${SPEED_OPTIONS[speedIndex]}x`;
  }
  updateSpeedLabel();

  // stats update kar
  function updateStats() {
    statsDisplay.textContent = `Ep: ${episodeCount} | R: ${currentEpisodeReward.toFixed(1)}`;
  }
  updateStats();

  // -------- Q-Learning step --------
  // ye hai core logic — ek step lena, Q-table update karna
  function qLearningStep() {
    // epsilon-greedy action select kar
    let action;
    if (Math.random() < epsilon) {
      // explore — random action le
      action = ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
    } else {
      // exploit — best action le Q-table se
      action = qTable.bestAction(env.agentX, env.agentY);
    }

    // current state yaad rakh — update ke liye chahiye
    const prevX = env.agentX;
    const prevY = env.agentY;

    // step le environment mein
    const { reward, done } = env.step(action);

    // Q-table update kar — Bellman equation
    // Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_Q(s') - Q(s,a))
    const oldQ = qTable.get(prevX, prevY, action);
    const nextMaxQ = done ? 0 : qTable.maxQ(env.agentX, env.agentY);
    const newQ = oldQ + ALPHA * (reward + GAMMA * nextMaxQ - oldQ);
    qTable.set(prevX, prevY, action, newQ);

    // episode tracking
    currentEpisodeReward += reward;
    totalStepsInEpisode++;

    // agar episode khatam ho gaya
    if (done || totalStepsInEpisode > 100) {
      // 100 steps se zyada matlab agent lost hai — episode khatam kar
      episodeCount++;
      currentEpisodeReward = 0;
      totalStepsInEpisode = 0;
      env.resetAgent();

      // epsilon decay kar — dhire dhire kam explore kar
      epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
    }
  }

  // -------- Animation loop --------
  // requestAnimationFrame se smooth rendering + step timing
  function gameLoop(timestamp) {
    if (!isVisible) {
      animFrameId = null;
      return;
    }

    // kitne steps lene hain is frame mein — speed ke hisaab se
    const speed = SPEED_OPTIONS[speedIndex];
    // base rate: ~8 steps per second at 1x
    const stepInterval = 1000 / (8 * speed);

    if (timestamp - lastStepTime >= stepInterval) {
      // zyada speed pe multiple steps le per frame — but render ek baar
      const stepsPerTick = speed >= 20 ? 4 : 1;
      for (let i = 0; i < stepsPerTick; i++) {
        qLearningStep();
      }
      lastStepTime = timestamp;
      updateStats();
    }

    // render har frame pe — smooth 60fps
    renderer._recalcLayout();
    renderer.draw();

    animFrameId = requestAnimationFrame(gameLoop);
  }

  function startLoop() {
    if (!animFrameId && isVisible) {
      lastStepTime = performance.now();
      animFrameId = requestAnimationFrame(gameLoop);
    }
  }

  function stopLoop() {
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  }

  // -------- Event handlers --------

  // canvas click — agent place kar us cell pe
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = (canvas.width / (window.devicePixelRatio || 1)) / rect.width;
    const scaleY = (canvas.height / (window.devicePixelRatio || 1)) / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;
    const cell = renderer.pixelToGrid(px, py);
    if (cell) {
      env.placeAgent(cell.x, cell.y);
      // episode reset kar — naye position se seekhe
      currentEpisodeReward = 0;
      totalStepsInEpisode = 0;
    }
  });

  // reset button — sab nayi duniya, nayi Q-table
  resetBtn.addEventListener('click', () => {
    env.fullReset();
    qTable.clear();
    epsilon = EPSILON_START;
    episodeCount = 0;
    currentEpisodeReward = 0;
    totalStepsInEpisode = 0;
    updateStats();
  });

  // speed toggle — cycle through 1x, 5x, 20x
  speedBtn.addEventListener('click', () => {
    speedIndex = (speedIndex + 1) % SPEED_OPTIONS.length;
    updateSpeedLabel();
  });

  // resize handle kar — responsive banane ke liye
  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas();
  });
  resizeObserver.observe(container);

  // -------- IntersectionObserver — sirf visible hone pe run kar --------
  // performance ke liye zaroori hai — background mein CPU waste mat kar
  const intersectionObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          isVisible = true;
          startLoop();
        } else {
          isVisible = false;
          stopLoop();
        }
      }
    },
    { threshold: 0.1 } // 10% dikhna chahiye start karne ke liye
  );
  intersectionObserver.observe(container);

  // initial setup
  resizeCanvas();
  // ek baar draw kar — taaki blank canvas na dikhe jab tak visible nahi hota
  renderer._recalcLayout();
  renderer.draw();
}
