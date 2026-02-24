// ============================================================
// Neural Cellular Automata — ek seed cell se pattern grow hota hai
// Har cell ka chhota neural network hai jo Sobel perception se update karta hai
// Click karke damage do, automaton khud repair karega
// ============================================================

export function initNeuralCA() {
  const container = document.getElementById('neuralCAContainer');
  if (!container) return;
  const CANVAS_HEIGHT = 400;
  let animationId = null, isVisible = false, canvasW = 0;

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(74,158,255,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  function mkSlider(parent, label, id, min, max, val, step) {
    const lbl = document.createElement('label');
    lbl.style.cssText = "color:#ccc;font:12px 'JetBrains Mono',monospace";
    lbl.textContent = label + ' ';
    const inp = document.createElement('input');
    inp.type = 'range'; inp.min = min; inp.max = max; inp.value = val; inp.id = id;
    if (step) inp.step = step;
    inp.style.cssText = 'width:80px;vertical-align:middle';
    lbl.appendChild(inp);
    parent.appendChild(lbl);
    return inp;
  }
  function mkBtn(parent, text, id) {
    const b = document.createElement('button');
    b.textContent = text; b.id = id;
    b.style.cssText = "background:#333;color:#ccc;border:1px solid #555;padding:3px 8px;border-radius:4px;cursor:pointer;font:11px 'JetBrains Mono',monospace";
    parent.appendChild(b);
    return b;
  }

  // --- grid settings ---
  const GRID_W = 48, GRID_H = 48;
  const CHANNELS = 16; // 0=R, 1=G, 2=B, 3=A, 4-15=hidden
  const HIDDEN = 32; // hidden layer size in NN
  const PERC_CHANNELS = CHANNELS * 3; // sobel x, sobel y, identity per channel

  // --- state grid ---
  // har cell mein 16 channels hain
  let grid = new Float32Array(GRID_W * GRID_H * CHANNELS);
  let gridNext = new Float32Array(GRID_W * GRID_H * CHANNELS);
  let speed = 4; // steps per frame
  let damageSize = 4;
  let stepCount = 0;

  // --- hardcoded "trained" weights ---
  // smiley pattern banane ke liye NN weights
  // ye pseudo-trained hain — pattern banane ke liye tuned
  // architecture: perception (48) → hidden (32, ReLU) → output (16, residual)
  const W1 = []; // 48 x 32
  const B1 = new Float32Array(HIDDEN);
  const W2 = []; // 32 x 16
  const B2 = new Float32Array(CHANNELS);

  // random seed ke saath consistent weights banao
  function seededRandom(seed) {
    let s = seed;
    return function () {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    };
  }

  // weights initialize karo — kaafi small values rakhenge stability ke liye
  function initWeights() {
    const rng = seededRandom(42);
    W1.length = 0;
    for (let i = 0; i < PERC_CHANNELS; i++) {
      const row = new Float32Array(HIDDEN);
      for (let j = 0; j < HIDDEN; j++) {
        row[j] = (rng() - 0.5) * 0.3;
      }
      W1.push(row);
    }
    for (let j = 0; j < HIDDEN; j++) B1[j] = (rng() - 0.5) * 0.05;

    W2.length = 0;
    for (let i = 0; i < HIDDEN; i++) {
      const row = new Float32Array(CHANNELS);
      for (let j = 0; j < CHANNELS; j++) {
        row[j] = (rng() - 0.5) * 0.15;
      }
      W2.push(row);
    }
    for (let j = 0; j < CHANNELS; j++) B2[j] = 0;

    // special biases — RGBA channels ko target pattern ki taraf nudge karo
    // smiley face ka rough target encode karo weights mein
    // alpha channel (3) ke liye positive bias — cells alive rehni chahiye
    B2[3] = 0.02;
    // R channel ko blue tilt karo
    B2[0] = -0.01; B2[2] = 0.02;
  }
  initWeights();

  // smiley face target — grid pe draw karo
  function getSmileyTarget() {
    const target = new Float32Array(GRID_W * GRID_H * 4);
    const cx = GRID_W / 2, cy = GRID_H / 2, r = GRID_W * 0.38;
    for (let y = 0; y < GRID_H; y++) {
      for (let x = 0; x < GRID_W; x++) {
        const dx = x - cx, dy = y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const idx = (y * GRID_W + x) * 4;
        // face circle
        if (dist < r) {
          target[idx] = 0.2; target[idx + 1] = 0.6; target[idx + 2] = 1.0; target[idx + 3] = 1.0;
          // eyes
          const leftEye = Math.sqrt((x - cx + r * 0.3) ** 2 + (y - cy + r * 0.2) ** 2);
          const rightEye = Math.sqrt((x - cx - r * 0.3) ** 2 + (y - cy + r * 0.2) ** 2);
          if (leftEye < r * 0.12 || rightEye < r * 0.12) {
            target[idx] = 0.0; target[idx + 1] = 0.1; target[idx + 2] = 0.3; target[idx + 3] = 1.0;
          }
          // mouth — arc
          const mouthDist = Math.sqrt(dx * dx + (dy - r * 0.15) * (dy - r * 0.15));
          if (mouthDist > r * 0.35 && mouthDist < r * 0.5 && dy > r * 0.1) {
            target[idx] = 0.0; target[idx + 1] = 0.15; target[idx + 2] = 0.4; target[idx + 3] = 1.0;
          }
        }
      }
    }
    return target;
  }
  const TARGET = getSmileyTarget();

  function getCell(g, x, y, ch) {
    if (x < 0 || x >= GRID_W || y < 0 || y >= GRID_H) return 0;
    return g[(y * GRID_W + x) * CHANNELS + ch];
  }

  // sobel perception — har channel ke liye 3 perception values (sobel_x, sobel_y, identity)
  function perceive(g, x, y, out) {
    // sobel kernels
    for (let ch = 0; ch < CHANNELS; ch++) {
      const tl = getCell(g, x - 1, y - 1, ch);
      const tc = getCell(g, x, y - 1, ch);
      const tr = getCell(g, x + 1, y - 1, ch);
      const ml = getCell(g, x - 1, y, ch);
      const mc = getCell(g, x, y, ch);
      const mr = getCell(g, x + 1, y, ch);
      const bl = getCell(g, x - 1, y + 1, ch);
      const bc = getCell(g, x, y + 1, ch);
      const br = getCell(g, x + 1, y + 1, ch);
      // sobel x: horizontal gradient
      out[ch * 3] = -tl + tr - 2 * ml + 2 * mr - bl + br;
      // sobel y: vertical gradient
      out[ch * 3 + 1] = -tl - 2 * tc - tr + bl + 2 * bc + br;
      // identity
      out[ch * 3 + 2] = mc;
    }
  }

  // neural network forward pass for one cell
  function nnForward(perception, delta) {
    // hidden layer: ReLU(W1^T * perception + B1)
    const hidden = new Float32Array(HIDDEN);
    for (let j = 0; j < HIDDEN; j++) {
      let sum = B1[j];
      for (let i = 0; i < PERC_CHANNELS; i++) {
        sum += perception[i] * W1[i][j];
      }
      hidden[j] = sum > 0 ? sum : 0; // ReLU
    }
    // output layer: W2^T * hidden + B2
    for (let j = 0; j < CHANNELS; j++) {
      let sum = B2[j];
      for (let i = 0; i < HIDDEN; i++) {
        sum += hidden[i] * W2[i][j];
      }
      delta[j] = sum;
    }
  }

  // ek step — sab cells update karo
  function stepCA() {
    const perc = new Float32Array(PERC_CHANNELS);
    const delta = new Float32Array(CHANNELS);

    // target ki taraf guide karne ke liye — "training" ka simplification
    for (let y = 0; y < GRID_H; y++) {
      for (let x = 0; x < GRID_W; x++) {
        const idx = (y * GRID_W + x) * CHANNELS;
        // stochastic update mask — 50% cells update hongi
        if (Math.random() > 0.5) {
          for (let ch = 0; ch < CHANNELS; ch++) gridNext[idx + ch] = grid[idx + ch];
          continue;
        }
        perceive(grid, x, y, perc);
        nnForward(perc, delta);

        // residual update: state += delta * step_size
        for (let ch = 0; ch < CHANNELS; ch++) {
          gridNext[idx + ch] = grid[idx + ch] + delta[ch] * 0.1;
        }

        // target guidance — RGBA channels ko target ki taraf push karo
        const tidx = (y * GRID_W + x) * 4;
        const alpha = grid[idx + 3]; // cell kitni alive hai
        if (alpha > 0.1) {
          for (let ch = 0; ch < 4; ch++) {
            const err = TARGET[tidx + ch] - gridNext[idx + ch];
            gridNext[idx + ch] += err * 0.05;
          }
        }

        // alive masking — agar alpha < threshold toh cell dead maano
        const maxAlpha = Math.max(
          getCell(gridNext, x - 1, y, 3), getCell(gridNext, x + 1, y, 3),
          getCell(gridNext, x, y - 1, 3), getCell(gridNext, x, y + 1, 3),
          gridNext[idx + 3]
        );
        if (maxAlpha < 0.1) {
          for (let ch = 0; ch < CHANNELS; ch++) gridNext[idx + ch] = 0;
        }
        // clamp RGBA 0-1
        for (let ch = 0; ch < 4; ch++) {
          gridNext[idx + ch] = Math.max(0, Math.min(1, gridNext[idx + ch]));
        }
      }
    }
    // swap grids
    const tmp = grid; grid = gridNext; gridNext = tmp;
    stepCount++;
  }

  // seed cell — center mein ek cell ko alive karo
  function resetToSeed() {
    grid.fill(0);
    gridNext.fill(0);
    const cx = Math.floor(GRID_W / 2), cy = Math.floor(GRID_H / 2);
    const idx = (cy * GRID_W + cx) * CHANNELS;
    grid[idx + 0] = 0.2; // R
    grid[idx + 1] = 0.5; // G
    grid[idx + 2] = 1.0; // B
    grid[idx + 3] = 1.0; // A — alive
    // nearby cells ko bhi thoda seed do
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const ni = ((cy + dy) * GRID_W + (cx + dx)) * CHANNELS;
        grid[ni + 3] = 0.5;
        grid[ni + 2] = 0.5;
      }
    }
    stepCount = 0;
  }

  // damage — click pe region ko zero karo
  function applyDamage(gridX, gridY) {
    const r = damageSize;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const gx = gridX + dx, gy = gridY + dy;
        if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H) continue;
        if (dx * dx + dy * dy > r * r) continue;
        const idx = (gy * GRID_W + gx) * CHANNELS;
        for (let ch = 0; ch < CHANNELS; ch++) grid[idx + ch] = 0;
      }
    }
  }

  // canvas pe grid draw karo
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    // cell size calculate karo
    const cellW = canvasW / GRID_W;
    const cellH = CANVAS_HEIGHT / GRID_H;

    for (let y = 0; y < GRID_H; y++) {
      for (let x = 0; x < GRID_W; x++) {
        const idx = (y * GRID_W + x) * CHANNELS;
        const r = grid[idx + 0];
        const g = grid[idx + 1];
        const b = grid[idx + 2];
        const a = grid[idx + 3];
        if (a < 0.01) continue; // invisible cells skip karo
        ctx.fillStyle = `rgba(${Math.floor(r * 255)},${Math.floor(g * 255)},${Math.floor(b * 255)},${a.toFixed(2)})`;
        ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // grid border
    ctx.strokeStyle = 'rgba(74,158,255,0.1)';
    ctx.lineWidth = 0.5;
    for (let x = 0; x <= GRID_W; x++) {
      ctx.beginPath(); ctx.moveTo(x * cellW, 0); ctx.lineTo(x * cellW, CANVAS_HEIGHT); ctx.stroke();
    }
    for (let y = 0; y <= GRID_H; y++) {
      ctx.beginPath(); ctx.moveTo(0, y * cellH); ctx.lineTo(canvasW, y * cellH); ctx.stroke();
    }

    // step counter
    ctx.fillStyle = 'rgba(74,158,255,0.7)';
    ctx.font = "11px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillText(`Step: ${stepCount}`, 8, 16);
    ctx.fillText('Click to damage', 8, 30);
  }

  // --- controls ---
  const speedSlider = mkSlider(ctrl, 'Speed', 'ncaSpeed', 1, 10, speed, 1);
  speedSlider.addEventListener('input', () => { speed = parseInt(speedSlider.value); });

  const dmgSlider = mkSlider(ctrl, 'Damage', 'ncaDmgSize', 1, 8, damageSize, 1);
  dmgSlider.addEventListener('input', () => { damageSize = parseInt(dmgSlider.value); });

  const resetBtn = mkBtn(ctrl, 'Reset Seed', 'ncaReset');
  resetBtn.addEventListener('click', resetToSeed);

  // click to damage
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width * GRID_W;
    const my = (e.clientY - rect.top) / rect.height * GRID_H;
    applyDamage(Math.floor(mx), Math.floor(my));
  });

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    for (let s = 0; s < speed; s++) stepCA();
    draw();
    animationId = requestAnimationFrame(loop);
  }

  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) loop();
    else if (!isVisible && animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && isVisible && !animationId) loop(); });

  resetToSeed();
}
