// ============================================================
// 2D GAN Lab — Generator vs Discriminator ka live battle
// Generator random noise se 2D points banata hai, Discriminator
// real vs fake distinguish karta hai — dono saath mein improve hote hain
// ============================================================

// yahi entry point hai — GAN ka poora tamasha yahan se shuru hota hai
export function initGanLab() {
  const container = document.getElementById('ganLabContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const GEN_COLOR = '#ff8c42';  // generator ke points — orange
  const REAL_COLOR = '#4a9eff'; // real data ke points — blue
  let animationId = null, isVisible = false, canvasW = 0;

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(74,158,255,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls ---
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

  // dataset buttons
  const btnCircle = mkBtn(ctrl, 'Circle', 'gl-circle');
  const btnBlobs = mkBtn(ctrl, 'Two Blobs', 'gl-blobs');
  const btnSpiral = mkBtn(ctrl, 'Spiral', 'gl-spiral');
  const lrSlider = mkSlider(ctrl, 'LR:', 'gl-lr', 0.001, 0.1, 0.03, 0.001);
  const btnStep = mkBtn(ctrl, 'Train Step', 'gl-step');
  const btnAuto = mkBtn(ctrl, 'Auto ▶', 'gl-auto');
  const btnReset = mkBtn(ctrl, 'Reset', 'gl-reset');

  // info label
  const infoLbl = document.createElement('span');
  infoLbl.style.cssText = "color:#888;font:11px 'JetBrains Mono',monospace;margin-left:8px";
  infoLbl.textContent = 'Epoch: 0';
  ctrl.appendChild(infoLbl);

  // --- Simple MLP implementation (manual backprop) ---
  // Gaussian random number — Box-Muller transform
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // tanh aur uska derivative
  function tanh(x) { return Math.tanh(x); }
  function dtanh(x) { const t = Math.tanh(x); return 1 - t * t; }

  // sigmoid — discriminator ka output
  function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x)))); }

  // weight matrix banao — Xavier initialization
  function initWeights(rows, cols) {
    const scale = Math.sqrt(2 / (rows + cols));
    const w = [];
    for (let i = 0; i < rows; i++) {
      w[i] = [];
      for (let j = 0; j < cols; j++) w[i][j] = randn() * scale;
    }
    return w;
  }
  function initBias(n) { return new Array(n).fill(0); }

  // MLP: 2 hidden layers, 16 neurons each, tanh activation
  // Generator: 2→16→16→2 (z_dim=2 → output_dim=2)
  // Discriminator: 2→16→16→1 (input_dim=2 → sigmoid)
  let gen = {}, disc = {};
  let epoch = 0, autoTrain = false;
  let datasetType = 'circle';
  let realData = [];
  const N_REAL = 200;    // kitne real data points
  const N_GEN = 200;     // kitne generated points dikhane hain
  const BATCH = 64;      // training batch size

  function initNetwork(inputDim, outputDim) {
    return {
      w1: initWeights(inputDim, 16), b1: initBias(16),
      w2: initWeights(16, 16), b2: initBias(16),
      w3: initWeights(16, outputDim), b3: initBias(outputDim)
    };
  }

  // forward pass — returns all layer outputs for backprop
  function forward(net, x, finalSigmoid) {
    // layer 1
    const z1 = new Array(16).fill(0);
    for (let j = 0; j < 16; j++) {
      let s = net.b1[j];
      for (let i = 0; i < x.length; i++) s += x[i] * net.w1[i][j];
      z1[j] = s;
    }
    const a1 = z1.map(tanh);

    // layer 2
    const z2 = new Array(16).fill(0);
    for (let j = 0; j < 16; j++) {
      let s = net.b2[j];
      for (let i = 0; i < 16; i++) s += a1[i] * net.w2[i][j];
      z2[j] = s;
    }
    const a2 = z2.map(tanh);

    // output layer
    const outDim = net.w3[0].length;
    const z3 = new Array(outDim).fill(0);
    for (let j = 0; j < outDim; j++) {
      let s = net.b3[j];
      for (let i = 0; i < 16; i++) s += a2[i] * net.w3[i][j];
      z3[j] = s;
    }
    const a3 = finalSigmoid ? z3.map(sigmoid) : z3.slice();
    return { x, z1, a1, z2, a2, z3, a3 };
  }

  // backprop for discriminator — binary cross entropy loss
  // dL/dz = sigmoid(z) - label, ye already logit gradient hai
  // isSigmoid=false rakhna hai kyunki dOut already pre-activation gradient hai
  function trainDisc(realBatch, fakeBatch, lr) {
    // real examples ke liye — label=1
    for (let b = 0; b < realBatch.length; b++) {
      const fwd = forward(disc, realBatch[b], true);
      const pred = fwd.a3[0];
      // dL/dz = pred - 1 (sigmoid + BCE combined gradient for label=1)
      const dOut = pred - 1;
      applyGradients(disc, fwd, [dOut], lr, false);
    }
    // fake examples ke liye — label=0
    for (let b = 0; b < fakeBatch.length; b++) {
      const fwd = forward(disc, fakeBatch[b], true);
      const pred = fwd.a3[0];
      // dL/dz = pred - 0 = pred (sigmoid + BCE combined gradient for label=0)
      const dOut = pred;
      applyGradients(disc, fwd, [dOut], lr, false);
    }
  }

  // backprop for generator — fool the discriminator
  function trainGen(batch, lr) {
    for (let b = 0; b < batch.length; b++) {
      const z = [randn() * 0.5, randn() * 0.5];
      const gFwd = forward(gen, z, false);
      const fakePoint = gFwd.a3;
      const dFwd = forward(disc, fakePoint, true);
      const pred = dFwd.a3[0];

      // generator chahta hai ki disc "real" bole — label=1 for generator loss
      const dDiscOut = pred - 1;
      // pehle disc ke through backprop karke fakePoint ka gradient nikalo
      const dFakePoint = backpropToInput(disc, dFwd, [dDiscOut]);
      // ab ye gradient generator ke output layer ka gradient hai
      applyGradients(gen, gFwd, dFakePoint, lr, false);
    }
  }

  // gradient compute + apply — ek hi function mein
  function applyGradients(net, fwd, dOutput, lr, isSigmoid) {
    const outDim = net.w3[0].length;
    // output layer gradient
    const dz3 = new Array(outDim);
    for (let j = 0; j < outDim; j++) {
      if (isSigmoid) {
        const s = sigmoid(fwd.z3[j]);
        dz3[j] = dOutput[j] * s * (1 - s);
      } else {
        dz3[j] = dOutput[j];
      }
    }
    // w3, b3 update
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < outDim; j++) {
        net.w3[i][j] -= lr * fwd.a2[i] * dz3[j];
      }
    }
    for (let j = 0; j < outDim; j++) net.b3[j] -= lr * dz3[j];

    // layer 2 gradient
    const da2 = new Array(16).fill(0);
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < outDim; j++) da2[i] += dz3[j] * net.w3[i][j];
    }
    const dz2 = da2.map((d, i) => d * dtanh(fwd.z2[i]));
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < 16; j++) net.w2[i][j] -= lr * fwd.a1[i] * dz2[j];
    }
    for (let j = 0; j < 16; j++) net.b2[j] -= lr * dz2[j];

    // layer 1 gradient
    const da1 = new Array(16).fill(0);
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < 16; j++) da1[i] += dz2[j] * net.w2[i][j];
    }
    const dz1 = da1.map((d, i) => d * dtanh(fwd.z1[i]));
    for (let i = 0; i < fwd.x.length; i++) {
      for (let j = 0; j < 16; j++) net.w1[i][j] -= lr * fwd.x[i] * dz1[j];
    }
    for (let j = 0; j < 16; j++) net.b1[j] -= lr * dz1[j];
  }

  // discriminator ke through backprop — input tak gradient laao
  // dOutput already logit gradient hai (sigmoid + BCE ka combined), double sigmoid mat lagao
  function backpropToInput(net, fwd, dOutput) {
    const outDim = net.w3[0].length;
    const dz3 = new Array(outDim);
    for (let j = 0; j < outDim; j++) {
      dz3[j] = dOutput[j]; // already dL/dz3 hai, sigmoid derivative nahi chahiye
    }
    const da2 = new Array(16).fill(0);
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < outDim; j++) da2[i] += dz3[j] * net.w3[i][j];
    }
    const dz2 = da2.map((d, i) => d * dtanh(fwd.z2[i]));
    const da1 = new Array(16).fill(0);
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < 16; j++) da1[i] += dz2[j] * net.w2[i][j];
    }
    const dz1 = da1.map((d, i) => d * dtanh(fwd.z1[i]));
    // input tak gradient
    const dInput = new Array(fwd.x.length).fill(0);
    for (let i = 0; i < fwd.x.length; i++) {
      for (let j = 0; j < 16; j++) dInput[i] += dz1[j] * net.w1[i][j];
    }
    return dInput;
  }

  // --- Dataset generators ---
  function genCircle(n) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + randn() * 0.05;
      const r = 0.7 + randn() * 0.05;
      pts.push([Math.cos(a) * r, Math.sin(a) * r]);
    }
    return pts;
  }
  function genBlobs(n) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const cx = i < n / 2 ? -0.5 : 0.5;
      const cy = i < n / 2 ? -0.3 : 0.3;
      pts.push([cx + randn() * 0.15, cy + randn() * 0.15]);
    }
    return pts;
  }
  function genSpiral(n) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const t = (i / n) * 3 * Math.PI;
      const r = 0.1 + t * 0.08;
      const sign = i < n / 2 ? 1 : -1;
      pts.push([Math.cos(t * sign) * r + randn() * 0.03, Math.sin(t * sign) * r + randn() * 0.03]);
    }
    return pts;
  }

  function generateData() {
    if (datasetType === 'circle') realData = genCircle(N_REAL);
    else if (datasetType === 'blobs') realData = genBlobs(N_REAL);
    else realData = genSpiral(N_REAL);
  }

  function resetAll() {
    gen = initNetwork(2, 2);
    disc = initNetwork(2, 1);
    epoch = 0;
    autoTrain = false;
    btnAuto.textContent = 'Auto ▶';
    generateData();
    infoLbl.textContent = 'Epoch: 0';
  }

  // --- Coordinate mapping (data space [-1.5,1.5] → canvas) ---
  function toCanvasX(dx) { return (dx + 1.5) / 3 * canvasW; }
  function toCanvasY(dy) { return (1.5 - dy) / 3 * CANVAS_HEIGHT; }

  // --- Training step ---
  function trainStep() {
    const lr = parseFloat(lrSlider.value);
    // batch banao
    const realBatch = [], fakeBatch = [];
    for (let i = 0; i < BATCH; i++) {
      realBatch.push(realData[Math.floor(Math.random() * realData.length)]);
      const z = [randn() * 0.5, randn() * 0.5];
      const gFwd = forward(gen, z, false);
      fakeBatch.push(gFwd.a3);
    }
    // discriminator ko train karo — real vs fake
    trainDisc(realBatch, fakeBatch, lr);
    // generator ko train karo — discriminator ko fool karo
    // genBatch nahi chahiye — trainGen khud z sample karta hai
    trainGen(new Array(BATCH), lr * 2);
    epoch++;
    infoLbl.textContent = 'Epoch: ' + epoch;
  }

  // --- Render ---
  function render() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // discriminator heatmap — har 8px pe disc evaluate karo
    const step = 8;
    for (let px = 0; px < canvasW; px += step) {
      for (let py = 0; py < CANVAS_HEIGHT; py += step) {
        const dx = (px / canvasW) * 3 - 1.5;
        const dy = 1.5 - (py / CANVAS_HEIGHT) * 3;
        const fwd = forward(disc, [dx, dy], true);
        const p = fwd.a3[0]; // 1=real, 0=fake
        // real=blue tint, fake=orange tint
        const r = Math.floor(40 + (1 - p) * 60);
        const g = Math.floor(40 + p * 30);
        const b = Math.floor(40 + p * 80);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(px, py, step, step);
      }
    }

    // real data points — blue dots
    ctx.fillStyle = REAL_COLOR;
    for (const pt of realData) {
      ctx.beginPath();
      ctx.arc(toCanvasX(pt[0]), toCanvasY(pt[1]), 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // generated points — orange dots
    ctx.fillStyle = GEN_COLOR;
    for (let i = 0; i < N_GEN; i++) {
      const z = [randn() * 0.5, randn() * 0.5];
      const fwd = forward(gen, z, false);
      ctx.beginPath();
      ctx.arc(toCanvasX(fwd.a3[0]), toCanvasY(fwd.a3[1]), 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // legend
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = REAL_COLOR;
    ctx.fillRect(10, 10, 10, 10);
    ctx.fillStyle = '#ccc';
    ctx.fillText('Real', 24, 19);
    ctx.fillStyle = GEN_COLOR;
    ctx.fillRect(10, 26, 10, 10);
    ctx.fillStyle = '#ccc';
    ctx.fillText('Generated', 24, 35);
    ctx.fillStyle = '#888';
    ctx.fillText('BG = Discriminator confidence', 10, 52);
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
    if (autoTrain) trainStep();
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

  // --- Event listeners ---
  btnCircle.addEventListener('click', () => { datasetType = 'circle'; resetAll(); });
  btnBlobs.addEventListener('click', () => { datasetType = 'blobs'; resetAll(); });
  btnSpiral.addEventListener('click', () => { datasetType = 'spiral'; resetAll(); });
  btnStep.addEventListener('click', () => { trainStep(); render(); });
  btnAuto.addEventListener('click', () => {
    autoTrain = !autoTrain;
    btnAuto.textContent = autoTrain ? 'Pause ⏸' : 'Auto ▶';
    if (autoTrain && isVisible && !animationId) loop();
  });
  btnReset.addEventListener('click', resetAll);

  // --- Init ---
  resetAll();
  render();
}
