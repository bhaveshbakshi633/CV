// ============================================================
// Self-Organizing Map (Kohonen) — 2D grid jo data ki topology seekhta hai
// Neuron mesh dekhna data pe mold hota hua — satisfying visualization
// ============================================================

export function initSOM() {
  const container = document.getElementById('somContainer');
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
  const btnClusters = mkBtn(ctrl, 'Clusters', 'som-clusters');
  const btnRing = mkBtn(ctrl, 'Ring', 'som-ring');
  const btnUniform = mkBtn(ctrl, 'Uniform', 'som-uniform');
  const lrSlider = mkSlider(ctrl, 'LR:', 'som-lr', 0.01, 1.0, 0.5, 0.01);
  const radiusSlider = mkSlider(ctrl, 'Radius:', 'som-radius', 0.5, 8, 4, 0.5);
  const btnStep = mkBtn(ctrl, 'Step', 'som-step');
  const btnAuto = mkBtn(ctrl, 'Auto ▶', 'som-auto');
  const btnReset = mkBtn(ctrl, 'Reset', 'som-reset');
  const infoLbl = document.createElement('span');
  infoLbl.style.cssText = "color:#888;font:11px 'JetBrains Mono',monospace;margin-left:8px";
  ctrl.appendChild(infoLbl);

  // --- State ---
  const GRID_SIZE = 15; // 15x15 neuron grid
  let neurons = [];      // neurons[i][j] = {x, y} — weight vector (2D position)
  let data = [];         // data points [{x, y}]
  let datasetType = 'clusters';
  let autoRun = false;
  let iteration = 0;
  let initialLR = 0.5;
  let initialRadius = 4;

  // data space [0, 1] → canvas mapping
  const PAD = 20;

  function toCanvasX(dx) { return PAD + dx * (canvasW - 2 * PAD); }
  function toCanvasY(dy) { return PAD + dy * (CANVAS_HEIGHT - 2 * PAD); }

  // --- Dataset generators ---
  function genClusters() {
    const pts = [];
    const centers = [[0.25, 0.25], [0.75, 0.25], [0.5, 0.75], [0.25, 0.7]];
    for (let i = 0; i < 200; i++) {
      const c = centers[i % centers.length];
      pts.push({ x: c[0] + randn() * 0.06, y: c[1] + randn() * 0.06 });
    }
    return pts;
  }
  function genRing() {
    const pts = [];
    for (let i = 0; i < 200; i++) {
      const angle = (i / 200) * Math.PI * 2;
      const r = 0.3 + randn() * 0.03;
      pts.push({ x: 0.5 + Math.cos(angle) * r, y: 0.5 + Math.sin(angle) * r });
    }
    return pts;
  }
  function genUniform() {
    const pts = [];
    for (let i = 0; i < 200; i++) {
      pts.push({ x: 0.1 + Math.random() * 0.8, y: 0.1 + Math.random() * 0.8 });
    }
    return pts;
  }

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // neurons initialize — center mein chhoti si grid se shuru
  function initNeurons() {
    neurons = [];
    for (let i = 0; i < GRID_SIZE; i++) {
      neurons[i] = [];
      for (let j = 0; j < GRID_SIZE; j++) {
        // center ke aas paas chhota sa grid — bada hota jaayega
        neurons[i][j] = {
          x: 0.4 + (i / GRID_SIZE) * 0.2,
          y: 0.4 + (j / GRID_SIZE) * 0.2
        };
      }
    }
  }

  function generateData() {
    if (datasetType === 'clusters') data = genClusters();
    else if (datasetType === 'ring') data = genRing();
    else data = genUniform();
  }

  // --- SOM training step ---
  // BMU dhundho, neighborhood update karo
  function somStep() {
    // random data point uthao
    const pt = data[Math.floor(Math.random() * data.length)];

    // Best Matching Unit (BMU) dhundho — sabse nazdeek neuron
    let bestI = 0, bestJ = 0, bestDist = Infinity;
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        const dx = neurons[i][j].x - pt.x;
        const dy = neurons[i][j].y - pt.y;
        const d = dx * dx + dy * dy;
        if (d < bestDist) { bestDist = d; bestI = i; bestJ = j; }
      }
    }

    // learning rate aur radius decay karo iterations ke saath
    const decay = Math.exp(-iteration / 1000);
    const lr = parseFloat(lrSlider.value) * decay;
    const radius = parseFloat(radiusSlider.value) * decay;
    const radiusSq = radius * radius;

    // neighborhood update — BMU ke paas ke neurons ko data ki taraf kheencho
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        // grid distance (not weight distance)
        const gridDist = (i - bestI) * (i - bestI) + (j - bestJ) * (j - bestJ);
        if (gridDist > radiusSq * 4) continue; // optimization — door wale skip karo

        // neighborhood function — Gaussian
        const h = Math.exp(-gridDist / (2 * radiusSq));
        // weight update: w_i ← w_i + α·h(i,c)·(x - w_i)
        neurons[i][j].x += lr * h * (pt.x - neurons[i][j].x);
        neurons[i][j].y += lr * h * (pt.y - neurons[i][j].y);
      }
    }
    iteration++;
    infoLbl.textContent = `Iter: ${iteration} | LR: ${lr.toFixed(3)} | R: ${radius.toFixed(1)}`;
  }

  // --- Render ---
  function render() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // data points — blue dots
    ctx.fillStyle = 'rgba(74,158,255,0.4)';
    for (const pt of data) {
      ctx.beginPath();
      ctx.arc(toCanvasX(pt.x), toCanvasY(pt.y), 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // neuron grid mesh — connections draw karo
    ctx.strokeStyle = 'rgba(255,140,66,0.5)';
    ctx.lineWidth = 1;

    // horizontal connections
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE - 1; j++) {
        ctx.beginPath();
        ctx.moveTo(toCanvasX(neurons[i][j].x), toCanvasY(neurons[i][j].y));
        ctx.lineTo(toCanvasX(neurons[i][j + 1].x), toCanvasY(neurons[i][j + 1].y));
        ctx.stroke();
      }
    }
    // vertical connections
    for (let j = 0; j < GRID_SIZE; j++) {
      for (let i = 0; i < GRID_SIZE - 1; i++) {
        ctx.beginPath();
        ctx.moveTo(toCanvasX(neurons[i][j].x), toCanvasY(neurons[i][j].y));
        ctx.lineTo(toCanvasX(neurons[i + 1][j].x), toCanvasY(neurons[i + 1][j].y));
        ctx.stroke();
      }
    }

    // neuron dots — orange
    ctx.fillStyle = '#ff8c42';
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        ctx.beginPath();
        ctx.arc(toCanvasX(neurons[i][j].x), toCanvasY(neurons[i][j].y), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // legend
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = 'rgba(74,158,255,0.8)';
    ctx.fillRect(canvasW - 120, 10, 8, 8);
    ctx.fillStyle = '#aaa';
    ctx.fillText('Data', canvasW - 108, 18);
    ctx.fillStyle = '#ff8c42';
    ctx.fillRect(canvasW - 120, 24, 8, 8);
    ctx.fillStyle = '#aaa';
    ctx.fillText('Neurons', canvasW - 108, 32);
  }

  function resetAll() {
    initNeurons();
    generateData();
    iteration = 0;
    autoRun = false;
    btnAuto.textContent = 'Auto ▶';
    infoLbl.textContent = 'Iter: 0';
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

  // steps per frame — auto mode mein fast chalao
  const STEPS_PER_FRAME = 10;

  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (autoRun) {
      for (let i = 0; i < STEPS_PER_FRAME; i++) somStep();
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
  btnClusters.addEventListener('click', () => { datasetType = 'clusters'; resetAll(); });
  btnRing.addEventListener('click', () => { datasetType = 'ring'; resetAll(); });
  btnUniform.addEventListener('click', () => { datasetType = 'uniform'; resetAll(); });
  btnStep.addEventListener('click', () => { for (let i = 0; i < STEPS_PER_FRAME; i++) somStep(); render(); });
  btnAuto.addEventListener('click', () => {
    autoRun = !autoRun;
    btnAuto.textContent = autoRun ? 'Pause ⏸' : 'Auto ▶';
    if (autoRun && isVisible && !animationId) loop();
  });
  btnReset.addEventListener('click', resetAll);

  resetAll();
  render();
}
