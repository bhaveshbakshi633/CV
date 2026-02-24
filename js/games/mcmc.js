// ============================================================
// MCMC — Metropolis-Hastings Sampling Visualization
// 2D distributions se sampling — random walk, accept/reject,
// trail aur marginal histograms sab dikhega
// ============================================================

export function initMCMC() {
  const container = document.getElementById('mcmcContainer');
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
  const btnBanana = mkBtn(ctrl, 'Banana', 'mc-banana');
  const btnDonut = mkBtn(ctrl, 'Donut', 'mc-donut');
  const btnBimodal = mkBtn(ctrl, 'Bimodal', 'mc-bimodal');
  const stepSlider = mkSlider(ctrl, 'Step:', 'mc-step', 0.05, 2.0, 0.4, 0.05);
  const btnAuto = mkBtn(ctrl, 'Auto ▶', 'mc-auto');
  const btnClear = mkBtn(ctrl, 'Clear', 'mc-clear');
  const infoLbl = document.createElement('span');
  infoLbl.style.cssText = "color:#888;font:11px 'JetBrains Mono',monospace;margin-left:8px";
  ctrl.appendChild(infoLbl);

  // --- State ---
  let distType = 'banana';
  let samples = [];        // accepted samples [{x, y}]
  let currentPos = { x: 0, y: 0 };
  let trail = [];          // last 100 positions for trail
  let autoRun = false;
  let totalProposed = 0, totalAccepted = 0;

  // coordinate range — data space [-4, 4]
  const RANGE = 4;
  const HIST_BINS = 40;
  const MARGIN_X = 50; // right margin for y-histogram
  const MARGIN_Y = 50; // bottom margin for x-histogram
  // plot area dimensions (computed in resize)
  let plotW = 0, plotH = 0;

  // --- Target distributions (unnormalized log-density) ---
  function logDensity(x, y) {
    if (distType === 'banana') {
      // banana/correlated distribution — Rosenbrock-like
      const a = 1, b = 5;
      return -0.5 * ((a - x) * (a - x) + b * (y - x * x) * (y - x * x)) * 0.3;
    } else if (distType === 'donut') {
      // donut — ring shape
      const r = Math.sqrt(x * x + y * y);
      return -2 * (r - 2) * (r - 2);
    } else {
      // bimodal — do Gaussian blobs
      const g1 = -0.5 * ((x - 1.5) * (x - 1.5) + (y - 1.5) * (y - 1.5)) * 2;
      const g2 = -0.5 * ((x + 1.5) * (x + 1.5) + (y + 1.5) * (y + 1.5)) * 2;
      // log-sum-exp trick for numerical stability
      const m = Math.max(g1, g2);
      return m + Math.log(Math.exp(g1 - m) + Math.exp(g2 - m));
    }
  }

  // density value (for heatmap)
  function density(x, y) {
    return Math.exp(logDensity(x, y));
  }

  // --- Metropolis-Hastings step ---
  function mhStep() {
    const stepSize = parseFloat(stepSlider.value);
    // propose new position — Gaussian random walk
    const propX = currentPos.x + randn() * stepSize;
    const propY = currentPos.y + randn() * stepSize;

    // acceptance ratio (log space mein)
    const logAlpha = logDensity(propX, propY) - logDensity(currentPos.x, currentPos.y);
    totalProposed++;

    if (Math.log(Math.random()) < logAlpha) {
      // accept!
      currentPos = { x: propX, y: propY };
      totalAccepted++;
    }
    // current position add karo (accepted ya rejected, position wahi rehta)
    samples.push({ x: currentPos.x, y: currentPos.y });
    trail.push({ x: currentPos.x, y: currentPos.y });
    if (trail.length > 80) trail.shift();

    const rate = totalProposed > 0 ? (totalAccepted / totalProposed * 100).toFixed(1) : '0';
    infoLbl.textContent = `Samples: ${samples.length} | Accept: ${rate}%`;
  }

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // --- Coordinate transforms (data → plot area on canvas) ---
  function toPlotX(dx) { return ((dx + RANGE) / (2 * RANGE)) * plotW; }
  function toPlotY(dy) { return ((RANGE - dy) / (2 * RANGE)) * plotH; }

  // --- Render ---
  function render() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    plotW = canvasW - MARGIN_X;
    plotH = CANVAS_HEIGHT - MARGIN_Y;

    // heatmap background — target distribution dikhao
    const hStep = 6;
    let maxD = 0;
    // pehle max density nikalo normalization ke liye
    for (let px = 0; px < plotW; px += hStep * 3) {
      for (let py = 0; py < plotH; py += hStep * 3) {
        const dx = (px / plotW) * 2 * RANGE - RANGE;
        const dy = RANGE - (py / plotH) * 2 * RANGE;
        const d = density(dx, dy);
        if (d > maxD) maxD = d;
      }
    }
    if (maxD === 0) maxD = 1;
    for (let px = 0; px < plotW; px += hStep) {
      for (let py = 0; py < plotH; py += hStep) {
        const dx = (px / plotW) * 2 * RANGE - RANGE;
        const dy = RANGE - (py / plotH) * 2 * RANGE;
        const d = density(dx, dy) / maxD;
        const intensity = Math.floor(d * 40);
        ctx.fillStyle = `rgba(74,158,255,${intensity / 255})`;
        ctx.fillRect(px, py, hStep, hStep);
      }
    }

    // plot border
    ctx.strokeStyle = 'rgba(74,158,255,0.2)';
    ctx.strokeRect(0, 0, plotW, plotH);

    // accepted samples — scatter plot
    ctx.fillStyle = 'rgba(74,158,255,0.35)';
    for (const s of samples) {
      const sx = toPlotX(s.x), sy = toPlotY(s.y);
      if (sx >= 0 && sx <= plotW && sy >= 0 && sy <= plotH) {
        ctx.beginPath();
        ctx.arc(sx, sy, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // trail — last N positions connected
    if (trail.length > 1) {
      ctx.strokeStyle = 'rgba(255,140,66,0.6)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(toPlotX(trail[0].x), toPlotY(trail[0].y));
      for (let i = 1; i < trail.length; i++) {
        ctx.lineTo(toPlotX(trail[i].x), toPlotY(trail[i].y));
      }
      ctx.stroke();
    }

    // current position — bada dot
    ctx.fillStyle = '#ff8c42';
    ctx.beginPath();
    ctx.arc(toPlotX(currentPos.x), toPlotY(currentPos.y), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // --- Marginal histograms ---
    // X-axis histogram (bottom)
    const xHist = new Array(HIST_BINS).fill(0);
    const yHist = new Array(HIST_BINS).fill(0);
    for (const s of samples) {
      const bx = Math.floor(((s.x + RANGE) / (2 * RANGE)) * HIST_BINS);
      const by = Math.floor(((s.y + RANGE) / (2 * RANGE)) * HIST_BINS);
      if (bx >= 0 && bx < HIST_BINS) xHist[bx]++;
      if (by >= 0 && by < HIST_BINS) yHist[by]++;
    }
    const maxXH = Math.max(...xHist, 1);
    const maxYH = Math.max(...yHist, 1);
    const barW = plotW / HIST_BINS;

    // x-histogram neeche
    ctx.fillStyle = 'rgba(74,158,255,0.5)';
    for (let i = 0; i < HIST_BINS; i++) {
      const h = (xHist[i] / maxXH) * (MARGIN_Y - 5);
      ctx.fillRect(i * barW, plotH + MARGIN_Y - 2 - h, barW - 1, h);
    }

    // y-histogram right side
    const barH = plotH / HIST_BINS;
    for (let i = 0; i < HIST_BINS; i++) {
      const w = (yHist[i] / maxYH) * (MARGIN_X - 5);
      ctx.fillRect(plotW + 2, i * barH, w, barH - 1);
    }
  }

  function clearSamples() {
    samples = [];
    trail = [];
    currentPos = { x: 0, y: 0 };
    totalProposed = 0;
    totalAccepted = 0;
    infoLbl.textContent = 'Samples: 0';
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    plotW = canvasW - MARGIN_X;
    plotH = CANVAS_HEIGHT - MARGIN_Y;
  }
  resize();
  window.addEventListener('resize', resize);

  // --- Animation loop ---
  let stepsPerFrame = 5;
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (autoRun) {
      for (let i = 0; i < stepsPerFrame; i++) mhStep();
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

  // --- Event listeners ---
  btnBanana.addEventListener('click', () => { distType = 'banana'; clearSamples(); });
  btnDonut.addEventListener('click', () => { distType = 'donut'; clearSamples(); });
  btnBimodal.addEventListener('click', () => { distType = 'bimodal'; clearSamples(); });
  btnAuto.addEventListener('click', () => {
    autoRun = !autoRun;
    btnAuto.textContent = autoRun ? 'Pause ⏸' : 'Auto ▶';
    if (autoRun && isVisible && !animationId) loop();
  });
  btnClear.addEventListener('click', clearSamples);

  clearSamples();
  render();
}
