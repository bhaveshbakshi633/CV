// ============================================================
// Diffusion Model — Forward & Reverse Process ka visual demo
// Data → Noise (forward) aur Noise → Data (reverse denoising)
// MLP seekhta hai noise predict karna — step by step dekhna
// ============================================================

export function initDiffusionModel() {
  const container = document.getElementById('diffusionModelContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
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

  // controls
  const btnSmiley = mkBtn(ctrl, 'Smiley', 'dm-smiley');
  const btnRing = mkBtn(ctrl, 'Ring', 'dm-ring');
  const timeSlider = mkSlider(ctrl, 't:', 'dm-time', 0, 49, 0, 1);
  const btnForward = mkBtn(ctrl, 'Forward ▶', 'dm-fwd');
  const btnReverse = mkBtn(ctrl, 'Reverse ◀', 'dm-rev');
  const btnTrain = mkBtn(ctrl, 'Train Denoiser', 'dm-train');
  const speedSlider = mkSlider(ctrl, 'Speed:', 'dm-speed', 1, 5, 2, 1);
  const btnReset = mkBtn(ctrl, 'Reset', 'dm-reset');
  const infoLbl = document.createElement('span');
  infoLbl.style.cssText = "color:#888;font:11px 'JetBrains Mono',monospace;margin-left:8px";
  ctrl.appendChild(infoLbl);

  // --- State ---
  const T = 50;            // total diffusion timesteps
  const N_POINTS = 200;    // kitne points
  let datasetType = 'smiley';
  let originalData = [];   // clean data [{x, y}]
  let currentPts = [];     // current state of points
  let noisyStates = [];    // precomputed noisy states for each timestep
  let currentT = 0;        // current timestep (0=clean, T-1=noise)
  let mode = 'idle';       // 'forward', 'reverse', 'idle'
  let denoiserNet = null;  // trained MLP for denoising
  let trainEpochs = 0;

  // noise schedule — linear beta from small to large
  const betas = [];
  for (let t = 0; t < T; t++) {
    betas.push(0.0001 + (0.02 - 0.0001) * t / (T - 1));
  }
  // alpha_bar cumulative product — for direct sampling at timestep t
  const alphaBar = [];
  let cumProd = 1;
  for (let t = 0; t < T; t++) {
    cumProd *= (1 - betas[t]);
    alphaBar.push(cumProd);
  }

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // --- Dataset generators ---
  function genSmiley() {
    const pts = [];
    // aankhein
    for (let i = 0; i < 30; i++) {
      pts.push({ x: -0.35 + randn() * 0.06, y: 0.3 + randn() * 0.06 });
      pts.push({ x: 0.35 + randn() * 0.06, y: 0.3 + randn() * 0.06 });
    }
    // muh — smile arc
    for (let i = 0; i < 80; i++) {
      const a = -0.8 + (i / 80) * 1.6;
      const curve = -0.3 - 0.15 * Math.cos(a * Math.PI / 1.6);
      pts.push({ x: a * 0.5 + randn() * 0.03, y: curve + randn() * 0.03 });
    }
    // face outline — remaining count pehle capture karo, nahi toh loop shrink hoga
    const remaining = N_POINTS - pts.length;
    for (let i = 0; i < remaining; i++) {
      const a = (i / remaining) * Math.PI * 2;
      const r = 0.7 + randn() * 0.03;
      pts.push({ x: Math.cos(a) * r, y: Math.sin(a) * r });
    }
    return pts.slice(0, N_POINTS);
  }

  function genRing() {
    const pts = [];
    for (let i = 0; i < N_POINTS; i++) {
      const a = (i / N_POINTS) * Math.PI * 2;
      const r = 0.6 + randn() * 0.04;
      pts.push({ x: Math.cos(a) * r, y: Math.sin(a) * r });
    }
    return pts;
  }

  // --- Forward diffusion: x_t = √α̅_t · x_0 + √(1-α̅_t) · ε ---
  function computeNoisyStates() {
    noisyStates = [originalData.map(p => ({ x: p.x, y: p.y }))];
    for (let t = 1; t < T; t++) {
      const ab = alphaBar[t];
      const sqrtAb = Math.sqrt(ab);
      const sqrtOneMinusAb = Math.sqrt(1 - ab);
      const pts = [];
      for (let i = 0; i < N_POINTS; i++) {
        pts.push({
          x: sqrtAb * originalData[i].x + sqrtOneMinusAb * randn(),
          y: sqrtAb * originalData[i].y + sqrtOneMinusAb * randn()
        });
      }
      noisyStates.push(pts);
    }
  }

  // --- Simple denoiser MLP: [x, y, t/T] → [noise_x, noise_y] ---
  // 3 → 32 → 32 → 2 with tanh activations
  function initDenoiser() {
    const scale1 = Math.sqrt(2 / 35);
    const scale2 = Math.sqrt(2 / 64);
    return {
      w1: Array.from({ length: 3 }, () => Array.from({ length: 32 }, () => randn() * scale1)),
      b1: new Array(32).fill(0),
      w2: Array.from({ length: 32 }, () => Array.from({ length: 32 }, () => randn() * scale2)),
      b2: new Array(32).fill(0),
      w3: Array.from({ length: 32 }, () => Array.from({ length: 2 }, () => randn() * scale2)),
      b3: new Array(2).fill(0)
    };
  }

  function forwardMLP(net, input) {
    // layer 1
    const z1 = new Array(32).fill(0);
    for (let j = 0; j < 32; j++) {
      let s = net.b1[j];
      for (let i = 0; i < 3; i++) s += input[i] * net.w1[i][j];
      z1[j] = s;
    }
    const a1 = z1.map(v => Math.tanh(v));
    // layer 2
    const z2 = new Array(32).fill(0);
    for (let j = 0; j < 32; j++) {
      let s = net.b2[j];
      for (let i = 0; i < 32; i++) s += a1[i] * net.w2[i][j];
      z2[j] = s;
    }
    const a2 = z2.map(v => Math.tanh(v));
    // output
    const out = new Array(2).fill(0);
    for (let j = 0; j < 2; j++) {
      let s = net.b3[j];
      for (let i = 0; i < 32; i++) s += a2[i] * net.w3[i][j];
      out[j] = s;
    }
    return { z1, a1, z2, a2, out };
  }

  // train one mini-batch — noise prediction objective
  function trainDenoiserStep(net, lr) {
    const batchSize = 32;
    for (let b = 0; b < batchSize; b++) {
      // random timestep aur random point
      const t = Math.floor(Math.random() * (T - 1)) + 1;
      const idx = Math.floor(Math.random() * N_POINTS);
      const x0 = originalData[idx];
      const ab = alphaBar[t];
      const epsX = randn(), epsY = randn();
      // noisy point
      const xt = {
        x: Math.sqrt(ab) * x0.x + Math.sqrt(1 - ab) * epsX,
        y: Math.sqrt(ab) * x0.y + Math.sqrt(1 - ab) * epsY
      };
      const input = [xt.x, xt.y, t / T];
      const fwd = forwardMLP(net, input);
      // loss gradient: d/d_out (pred - eps)^2
      const dOut = [fwd.out[0] - epsX, fwd.out[1] - epsY];
      // simple backprop + update
      backpropMLP(net, fwd, input, dOut, lr);
    }
  }

  function backpropMLP(net, fwd, input, dOut, lr) {
    // output layer gradient
    const da2 = new Array(32).fill(0);
    for (let i = 0; i < 32; i++) {
      for (let j = 0; j < 2; j++) {
        da2[i] += dOut[j] * net.w3[i][j];
        net.w3[i][j] -= lr * fwd.a2[i] * dOut[j];
      }
    }
    for (let j = 0; j < 2; j++) net.b3[j] -= lr * dOut[j];

    // layer 2
    const dz2 = da2.map((d, i) => d * (1 - fwd.a2[i] * fwd.a2[i]));
    const da1 = new Array(32).fill(0);
    for (let i = 0; i < 32; i++) {
      for (let j = 0; j < 32; j++) {
        da1[i] += dz2[j] * net.w2[i][j];
        net.w2[i][j] -= lr * fwd.a1[i] * dz2[j];
      }
    }
    for (let j = 0; j < 32; j++) net.b2[j] -= lr * dz2[j];

    // layer 1
    const dz1 = da1.map((d, i) => d * (1 - fwd.a1[i] * fwd.a1[i]));
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 32; j++) {
        net.w1[i][j] -= lr * input[i] * dz1[j];
      }
    }
    for (let j = 0; j < 32; j++) net.b1[j] -= lr * dz1[j];
  }

  // --- Reverse diffusion step (DDPM-style) ---
  function reverseStep() {
    if (currentT <= 0) { mode = 'idle'; return; }
    const t = currentT;
    const beta = betas[t];
    const ab = alphaBar[t];
    const abPrev = t > 0 ? alphaBar[t - 1] : 1;

    for (let i = 0; i < N_POINTS; i++) {
      const input = [currentPts[i].x, currentPts[i].y, t / T];
      const fwd = forwardMLP(denoiserNet, input);
      const predNoise = fwd.out;
      // x_{t-1} = (1/√α_t)(x_t - β_t/√(1-α̅_t) · ε_θ) + σ_t·z
      const sqrtAlpha = Math.sqrt(1 - beta);
      const coeff = beta / Math.sqrt(1 - ab);
      currentPts[i].x = (currentPts[i].x - coeff * predNoise[0]) / sqrtAlpha;
      currentPts[i].y = (currentPts[i].y - coeff * predNoise[1]) / sqrtAlpha;
      // stochastic noise add (except at t=1)
      if (t > 1) {
        const sigma = Math.sqrt(beta);
        currentPts[i].x += sigma * randn() * 0.5;
        currentPts[i].y += sigma * randn() * 0.5;
      }
    }
    currentT--;
    timeSlider.value = currentT;
  }

  // --- Coordinate mapping ---
  function toCanvasX(dx) { return canvasW / 2 + dx * canvasW * 0.35; }
  function toCanvasY(dy) { return CANVAS_HEIGHT / 2 - dy * CANVAS_HEIGHT * 0.35; }

  // --- Render ---
  function render() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    const pts = currentPts;
    if (!pts || pts.length === 0) return;

    // points draw karo
    const alpha = Math.max(0.3, 1 - currentT / T * 0.7);
    ctx.fillStyle = `rgba(74,158,255,${alpha})`;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // score arrows dikhao — denoising direction (agar denoiser trained hai)
    if (denoiserNet && currentT > 0) {
      ctx.strokeStyle = 'rgba(255,140,66,0.3)';
      ctx.lineWidth = 1;
      const arrowStep = Math.max(1, Math.floor(N_POINTS / 40)); // har 5th point ka arrow
      for (let i = 0; i < N_POINTS; i += arrowStep) {
        const input = [pts[i].x, pts[i].y, currentT / T];
        const fwd = forwardMLP(denoiserNet, input);
        // arrow shows direction opposite to predicted noise (= towards data)
        const ax = toCanvasX(pts[i].x);
        const ay = toCanvasY(pts[i].y);
        const dx = -fwd.out[0] * 0.15;
        const dy = fwd.out[1] * 0.15;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax + dx * canvasW * 0.35, ay + dy * CANVAS_HEIGHT * 0.35);
        ctx.stroke();
      }
    }

    // timestep indicator
    ctx.font = "bold 13px 'JetBrains Mono', monospace";
    ctx.fillStyle = '#ccc';
    ctx.textAlign = 'left';
    ctx.fillText(`t = ${currentT}/${T - 1}`, 10, 25);

    // progress bar
    const barW = canvasW - 20;
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fillRect(10, CANVAS_HEIGHT - 15, barW, 8);
    ctx.fillStyle = ACCENT;
    ctx.fillRect(10, CANVAS_HEIGHT - 15, barW * (currentT / (T - 1)), 8);

    ctx.fillStyle = '#666';
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillText('clean data', 10, CANVAS_HEIGHT - 20);
    ctx.textAlign = 'right';
    ctx.fillText('pure noise', canvasW - 10, CANVAS_HEIGHT - 20);
    ctx.textAlign = 'left';
  }

  // --- Set timestep from slider or forward/reverse ---
  function setTimestep(t) {
    currentT = Math.max(0, Math.min(T - 1, t));
    if (noisyStates.length > currentT) {
      currentPts = noisyStates[currentT].map(p => ({ x: p.x, y: p.y }));
    }
    timeSlider.value = currentT;
    infoLbl.textContent = `t=${currentT} | Trained: ${trainEpochs} epochs`;
  }

  let animFrameCount = 0;

  function resetAll() {
    if (datasetType === 'smiley') originalData = genSmiley();
    else originalData = genRing();
    computeNoisyStates();
    denoiserNet = initDenoiser();
    trainEpochs = 0;
    setTimestep(0);
    mode = 'idle';
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    animFrameCount++;
    const speed = parseInt(speedSlider.value);
    if (animFrameCount % (6 - speed) === 0) {
      if (mode === 'forward') {
        if (currentT < T - 1) {
          setTimestep(currentT + 1);
        } else {
          mode = 'idle';
        }
      } else if (mode === 'reverse') {
        if (currentT > 0 && denoiserNet) {
          reverseStep();
          infoLbl.textContent = `Reverse t=${currentT} | Trained: ${trainEpochs} epochs`;
        } else {
          mode = 'idle';
        }
      }
    }
    render();
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

  // --- Events ---
  btnSmiley.addEventListener('click', () => { datasetType = 'smiley'; resetAll(); });
  btnRing.addEventListener('click', () => { datasetType = 'ring'; resetAll(); });
  timeSlider.addEventListener('input', () => { mode = 'idle'; setTimestep(parseInt(timeSlider.value)); });
  btnForward.addEventListener('click', () => {
    mode = 'forward';
    if (currentT >= T - 1) setTimestep(0);
    if (isVisible && !animationId) loop();
  });
  btnReverse.addEventListener('click', () => {
    if (!denoiserNet) { infoLbl.textContent = 'Pehle Train Denoiser daba!'; return; }
    mode = 'reverse';
    // reverse ke liye — agar pehle se noise pe nahi hai toh wahan le jao
    if (currentT < T - 1) {
      setTimestep(T - 1);
      // fresh noise se shuru karo
      for (let i = 0; i < N_POINTS; i++) {
        currentPts[i] = { x: randn(), y: randn() };
      }
    }
    if (isVisible && !animationId) loop();
  });
  btnTrain.addEventListener('click', () => {
    // 500 training steps batch mein chala do
    infoLbl.textContent = 'Training...';
    for (let i = 0; i < 500; i++) {
      trainDenoiserStep(denoiserNet, 0.003);
    }
    trainEpochs += 500;
    infoLbl.textContent = `Trained: ${trainEpochs} steps — ab Reverse try karo!`;
  });
  btnReset.addEventListener('click', resetAll);

  // --- Init ---
  resetAll();
  render();
}
