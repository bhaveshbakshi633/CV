// ============================================================
// Bayesian Optimization — Gaussian Process ke saath 1D function optimize karo
// Sample leke uncertainty kam dekho, EI acquisition function se next point chuno
// ============================================================

// yahi entry point hai — GP fit karo, EI dikhao, sequentially optimize karo
export function initBayesianOpt() {
  const container = document.getElementById('bayesianOptContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  // canvas ka 70% upper portion GP ke liye, 30% neeche EI ke liye
  const GP_HEIGHT = CANVAS_HEIGHT * 0.68;
  const EI_HEIGHT = CANVAS_HEIGHT * 0.28;
  const EI_TOP = CANVAS_HEIGHT * 0.72;

  // target functions — user choose kar sakta hai
  const FUNCTIONS = {
    sinx: { name: 'sin(x)*x', fn: x => Math.sin(x) * x, range: [0, 12] },
    bumpy: { name: 'Bumpy', fn: x => Math.sin(3 * x) * Math.cos(x) + 0.5 * Math.sin(7 * x), range: [0, 8] },
    multimodal: { name: 'Multi-peak', fn: x => Math.sin(x) + Math.sin(3 * x) * 0.5 + Math.cos(5 * x) * 0.3, range: [0, 10] },
  };

  let animationId = null, isVisible = false, canvasW = 0;
  let currentFn = 'sinx';
  let observations = [];    // [{x, y}] — sampled points
  let lengthScale = 1.0;    // RBF kernel ka length scale
  let noiseVar = 0.01;      // observation noise

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(74,158,255,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  // helpers
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

  // function selector dropdown
  const selWrap = document.createElement('label');
  selWrap.style.cssText = "color:#ccc;font:12px 'JetBrains Mono',monospace";
  selWrap.textContent = 'f(x) ';
  const sel = document.createElement('select');
  sel.id = 'boFnSel';
  sel.style.cssText = "background:#222;color:#ccc;border:1px solid #555;padding:2px 6px;border-radius:4px;font:11px 'JetBrains Mono',monospace";
  Object.keys(FUNCTIONS).forEach(k => {
    const opt = document.createElement('option');
    opt.value = k; opt.textContent = FUNCTIONS[k].name;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => { currentFn = sel.value; observations = []; draw(); });
  selWrap.appendChild(sel);
  ctrl.appendChild(selWrap);

  // length scale slider
  const lsSlider = mkSlider(ctrl, 'Length Scale', 'boLS', 0.3, 3, lengthScale, 0.1);
  const lsVal = document.createElement('span');
  lsVal.style.cssText = "color:#4a9eff;font:11px 'JetBrains Mono',monospace;min-width:24px";
  lsVal.textContent = lengthScale.toFixed(1);
  ctrl.appendChild(lsVal);
  lsSlider.addEventListener('input', () => { lengthScale = +lsSlider.value; lsVal.textContent = lengthScale.toFixed(1); draw(); });

  // action buttons
  const sampleBtn = mkBtn(ctrl, 'Sample Next', 'boSample');
  sampleBtn.style.background = 'rgba(74,158,255,0.2)';
  sampleBtn.style.borderColor = ACCENT;
  sampleBtn.addEventListener('click', sampleNext);

  mkBtn(ctrl, 'Sample 5x', 'boSample5').addEventListener('click', () => { for (let i = 0; i < 5; i++) sampleNext(); });
  mkBtn(ctrl, 'Reset', 'boReset').addEventListener('click', () => { observations = []; draw(); });

  // stats
  const stats = document.createElement('div');
  stats.style.cssText = "font:11px 'JetBrains Mono',monospace;color:#888;margin-top:6px;";
  container.appendChild(stats);

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- RBF kernel — Gaussian kernel function ---
  function kernel(x1, x2) {
    const d = x1 - x2;
    return Math.exp(-0.5 * d * d / (lengthScale * lengthScale));
  }

  // --- matrix operations — GP ke liye zaroori hai ---
  // Cholesky decomposition — lower triangular L nikalo jaise K = L * L^T
  function cholesky(A) {
    const n = A.length;
    const L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
        if (i === j) {
          const val = A[i][i] - sum;
          L[i][j] = Math.sqrt(Math.max(1e-10, val));
        } else {
          L[i][j] = (A[i][j] - sum) / L[j][j];
        }
      }
    }
    return L;
  }

  // forward substitution — L * x = b solve karo
  function forwardSolve(L, b) {
    const n = L.length;
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < i; j++) sum += L[i][j] * x[j];
      x[i] = (b[i] - sum) / L[i][i];
    }
    return x;
  }

  // backward substitution — L^T * x = b solve karo
  function backwardSolve(L, b) {
    const n = L.length;
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = 0;
      for (let j = i + 1; j < n; j++) sum += L[j][i] * x[j];
      x[i] = (b[i] - sum) / L[i][i];
    }
    return x;
  }

  // --- GP predict karo — mean aur variance at test points ---
  function gpPredict(testX) {
    const n = observations.length;
    if (n === 0) {
      // prior — zero mean, unit variance
      return testX.map(() => ({ mean: 0, var: 1 }));
    }

    // kernel matrix K banao (n x n)
    const K = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => kernel(observations[i].x, observations[j].x) + (i === j ? noiseVar : 0))
    );

    // cholesky decomposition — stable inverse ke liye
    const L = cholesky(K);
    const y = observations.map(o => o.y);

    // alpha = K^-1 * y = L^T \ (L \ y)
    const alpha = backwardSolve(L, forwardSolve(L, y));

    return testX.map(xt => {
      // k* vector — test point aur observations ke beech kernel values
      const kStar = observations.map(o => kernel(xt, o.x));
      // mean = k*^T * alpha
      let mean = 0;
      for (let i = 0; i < n; i++) mean += kStar[i] * alpha[i];

      // variance = k** - k*^T * K^-1 * k* = k** - v^T * v where L*v = k*
      const v = forwardSolve(L, kStar);
      let varReduction = 0;
      for (let i = 0; i < n; i++) varReduction += v[i] * v[i];
      const variance = Math.max(0, kernel(xt, xt) - varReduction);

      return { mean, var: variance };
    });
  }

  // --- Expected Improvement (EI) — acquisition function ---
  function expectedImprovement(testX, predictions) {
    // best observation tak ka y dhundho
    const bestY = observations.length > 0 ? Math.max(...observations.map(o => o.y)) : -Infinity;

    return predictions.map((p, i) => {
      const sigma = Math.sqrt(Math.max(1e-10, p.var));
      if (sigma < 1e-8) return 0;
      const z = (p.mean - bestY) / sigma;
      // EI = (mean - best) * Phi(z) + sigma * phi(z)
      // Phi = CDF, phi = PDF of standard normal
      const phi = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
      const Phi = 0.5 * (1 + erf(z / Math.sqrt(2)));
      return (p.mean - bestY) * Phi + sigma * phi;
    });
  }

  // --- error function approximation — EI ke liye chahiye ---
  function erf(x) {
    // Abramowitz and Stegun approximation
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
    const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const sign = x >= 0 ? 1 : -1;
    const t = 1 / (1 + p * Math.abs(x));
    const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }

  // --- next sample pick karo — EI maximum pe ---
  function sampleNext() {
    const fn = FUNCTIONS[currentFn];
    const range = fn.range;

    if (observations.length === 0) {
      // pehla sample random jagah se lo
      const x = range[0] + Math.random() * (range[1] - range[0]);
      const y = fn.fn(x) + (Math.random() - 0.5) * 0.1;
      observations.push({ x, y });
      draw();
      return;
    }

    // test points pe GP predict karo
    const nTest = 200;
    const testX = Array.from({ length: nTest }, (_, i) => range[0] + (i / (nTest - 1)) * (range[1] - range[0]));
    const preds = gpPredict(testX);
    const ei = expectedImprovement(testX, preds);

    // maximum EI dhundho
    let maxEI = -1, bestX = testX[0];
    ei.forEach((val, i) => { if (val > maxEI) { maxEI = val; bestX = testX[i]; } });

    // us point pe sample lo
    const y = fn.fn(bestX) + (Math.random() - 0.5) * 0.1;
    observations.push({ x: bestX, y });
    draw();
  }

  // --- x value ko pixel x mein convert karo ---
  function xToPixel(x) {
    const range = FUNCTIONS[currentFn].range;
    return 40 + (x - range[0]) / (range[1] - range[0]) * (canvasW - 60);
  }
  function yToPixel(y, minY, maxY, top, height) {
    return top + height - (y - minY) / (maxY - minY) * height;
  }

  // --- main draw ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    const fn = FUNCTIONS[currentFn];
    const range = fn.range;

    // test points generate karo
    const nTest = 300;
    const testX = Array.from({ length: nTest }, (_, i) => range[0] + (i / (nTest - 1)) * (range[1] - range[0]));

    // true function evaluate karo
    const trueY = testX.map(x => fn.fn(x));

    // GP predictions
    const preds = gpPredict(testX);
    const gpMean = preds.map(p => p.mean);
    const gpStd = preds.map(p => Math.sqrt(Math.max(0, p.var)));

    // EI values
    const ei = observations.length > 0 ? expectedImprovement(testX, preds) : testX.map(() => 0);

    // y range compute karo — true function + GP mean + confidence band
    const allY = [...trueY, ...gpMean.map((m, i) => m + 2 * gpStd[i]), ...gpMean.map((m, i) => m - 2 * gpStd[i])];
    let minY = Math.min(...allY) - 0.5;
    let maxY = Math.max(...allY) + 0.5;
    if (observations.length > 0) {
      minY = Math.min(minY, ...observations.map(o => o.y) ) - 0.5;
      maxY = Math.max(maxY, ...observations.map(o => o.y)) + 0.5;
    }

    // --- GP plot (upper 68%) ---
    const gpTop = 15;
    const gpH = GP_HEIGHT - 25;

    // grid lines — subtle background
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const yy = gpTop + (gpH * i) / 4;
      ctx.beginPath(); ctx.moveTo(40, yy); ctx.lineTo(canvasW - 20, yy); ctx.stroke();
    }

    // confidence band — shaded area (2 sigma)
    if (observations.length > 0) {
      ctx.fillStyle = 'rgba(74,158,255,0.1)';
      ctx.beginPath();
      // upper band
      for (let i = 0; i < nTest; i++) {
        const px = xToPixel(testX[i]);
        const py = yToPixel(gpMean[i] + 2 * gpStd[i], minY, maxY, gpTop, gpH);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      // lower band (reverse)
      for (let i = nTest - 1; i >= 0; i--) {
        const px = xToPixel(testX[i]);
        const py = yToPixel(gpMean[i] - 2 * gpStd[i], minY, maxY, gpTop, gpH);
        ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();

      // 1-sigma band bhi dikhao — thoda darker
      ctx.fillStyle = 'rgba(74,158,255,0.08)';
      ctx.beginPath();
      for (let i = 0; i < nTest; i++) {
        const px = xToPixel(testX[i]);
        const py = yToPixel(gpMean[i] + gpStd[i], minY, maxY, gpTop, gpH);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      for (let i = nTest - 1; i >= 0; i--) {
        const px = xToPixel(testX[i]);
        const py = yToPixel(gpMean[i] - gpStd[i], minY, maxY, gpTop, gpH);
        ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
    }

    // true function — dashed line
    ctx.strokeStyle = '#888';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    testX.forEach((x, i) => {
      const px = xToPixel(x);
      const py = yToPixel(trueY[i], minY, maxY, gpTop, gpH);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // GP mean — solid blue line
    if (observations.length > 0) {
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.beginPath();
      testX.forEach((x, i) => {
        const px = xToPixel(x);
        const py = yToPixel(gpMean[i], minY, maxY, gpTop, gpH);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    // observations — dots with glow
    observations.forEach((o, idx) => {
      const px = xToPixel(o.x);
      const py = yToPixel(o.y, minY, maxY, gpTop, gpH);
      // glow
      ctx.fillStyle = 'rgba(74,158,255,0.3)';
      ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.fill();
      // dot
      ctx.fillStyle = idx === observations.length - 1 ? '#f59e0b' : ACCENT;
      ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.stroke();
      // sequence number
      ctx.fillStyle = '#888';
      ctx.font = "8px 'JetBrains Mono',monospace";
      ctx.textAlign = 'center';
      ctx.fillText(idx + 1, px, py - 8);
    });

    // legend
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    // true function legend
    ctx.setLineDash([4, 4]); ctx.strokeStyle = '#888'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(50, gpTop + 5); ctx.lineTo(70, gpTop + 5); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#888'; ctx.fillText('True f(x)', 74, gpTop + 9);
    // GP mean legend
    if (observations.length > 0) {
      ctx.strokeStyle = ACCENT; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(140, gpTop + 5); ctx.lineTo(160, gpTop + 5); ctx.stroke();
      ctx.fillStyle = ACCENT; ctx.fillText('GP Mean', 164, gpTop + 9);
    }

    // --- separator line ---
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(40, EI_TOP - 5); ctx.lineTo(canvasW - 20, EI_TOP - 5); ctx.stroke();

    // --- EI plot (lower 28%) ---
    ctx.fillStyle = '#4a9eff';
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillText('Expected Improvement', 45, EI_TOP + 5);

    if (observations.length > 0) {
      const maxEI = Math.max(...ei, 1e-10);
      // EI bar chart / filled area
      ctx.fillStyle = 'rgba(74,158,255,0.2)';
      ctx.beginPath();
      ctx.moveTo(xToPixel(testX[0]), EI_TOP + EI_HEIGHT);
      testX.forEach((x, i) => {
        const px = xToPixel(x);
        const py = EI_TOP + EI_HEIGHT - (ei[i] / maxEI) * (EI_HEIGHT - 15);
        ctx.lineTo(px, py);
      });
      ctx.lineTo(xToPixel(testX[nTest - 1]), EI_TOP + EI_HEIGHT);
      ctx.closePath();
      ctx.fill();

      // EI curve outline
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      testX.forEach((x, i) => {
        const px = xToPixel(x);
        const py = EI_TOP + EI_HEIGHT - (ei[i] / maxEI) * (EI_HEIGHT - 15);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();

      // max EI point pe marker
      let maxEIIdx = 0;
      ei.forEach((v, i) => { if (v > ei[maxEIIdx]) maxEIIdx = i; });
      const mpx = xToPixel(testX[maxEIIdx]);
      const mpy = EI_TOP + EI_HEIGHT - (ei[maxEIIdx] / maxEI) * (EI_HEIGHT - 15);
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath(); ctx.arc(mpx, mpy, 4, 0, Math.PI * 2); ctx.fill();
      // vertical line from EI max to GP plot
      ctx.strokeStyle = 'rgba(245,158,11,0.3)';
      ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(mpx, mpy); ctx.lineTo(mpx, gpTop); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#f59e0b';
      ctx.font = "9px 'JetBrains Mono',monospace";
      ctx.textAlign = 'center';
      ctx.fillText('next', mpx, mpy - 6);
    }

    // hint text
    if (observations.length === 0) {
      ctx.font = "13px 'JetBrains Mono',monospace";
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.textAlign = 'center';
      ctx.fillText('Click "Sample Next" to start optimization', canvasW / 2, CANVAS_HEIGHT / 2);
    }

    // stats
    if (observations.length > 0) {
      const best = observations.reduce((a, b) => a.y > b.y ? a : b);
      stats.textContent = `Samples: ${observations.length}  |  Best y: ${best.y.toFixed(3)} at x=${best.x.toFixed(3)}  |  Length scale: ${lengthScale.toFixed(1)}`;
    } else {
      stats.textContent = 'No observations yet — start sampling!';
    }
  }

  // animation loop — har frame pe redraw taaki resize ke baad canvas sahi dikhe
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    draw();
    animationId = requestAnimationFrame(loop);
  }

  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible) { draw(); if (!animationId) loop(); }
    else if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && isVisible && !animationId) loop(); });

  draw();
}
