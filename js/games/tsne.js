// ============================================================
// t-SNE — High-Dimensional Data Visualization
// 10D clusters ko 2D mein project karo, dekhna points kaise
// apne apne group mein settle hote hain — dimensionality reduction ka magic
// ============================================================

export function initTSNE() {
  const container = document.getElementById('tsneContainer');
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
  const perpSlider = mkSlider(ctrl, 'Perplexity:', 'ts-perp', 5, 50, 20, 1);
  const lrSlider = mkSlider(ctrl, 'LR:', 'ts-lr', 1, 200, 50, 1);
  const btnAuto = mkBtn(ctrl, 'Auto ▶', 'ts-auto');
  const btnStep = mkBtn(ctrl, 'Step', 'ts-step');
  const btnRestart = mkBtn(ctrl, 'Restart', 'ts-restart');
  const infoLbl = document.createElement('span');
  infoLbl.style.cssText = "color:#888;font:11px 'JetBrains Mono',monospace;margin-left:8px";
  ctrl.appendChild(infoLbl);

  // --- State ---
  const N_CLUSTERS = 5;
  const PTS_PER_CLUSTER = 50;
  const N = N_CLUSTERS * PTS_PER_CLUSTER; // 250 total points
  const DIMS = 10; // high-dimensional space
  // 5 cluster colors — distinct colors for each group
  const COLORS = ['#4a9eff', '#ef4444', '#4ade80', '#f59e0b', '#a855f7'];

  let highD = [];       // high-dimensional data [N][DIMS]
  let labels = [];      // cluster labels [N]
  let Y = [];           // 2D positions [N][2]
  let gains = [];       // adaptive learning rate gains
  let prevDY = [];      // previous gradients (momentum)
  let P = [];           // joint probability matrix (sparse — as flat array)
  let autoRun = false;
  let iteration = 0;

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // --- Generate high-D data: 5 clusters in 10D ---
  function generateHighDData() {
    highD = [];
    labels = [];
    for (let c = 0; c < N_CLUSTERS; c++) {
      // har cluster ka center random jagah pe
      const center = [];
      for (let d = 0; d < DIMS; d++) center.push(randn() * 5);
      for (let i = 0; i < PTS_PER_CLUSTER; i++) {
        const pt = [];
        for (let d = 0; d < DIMS; d++) pt.push(center[d] + randn() * 0.8);
        highD.push(pt);
        labels.push(c);
      }
    }
  }

  // --- Pairwise squared distances in high-D ---
  function computeDistances(data) {
    const n = data.length;
    const D = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let sum = 0;
        for (let d = 0; d < data[i].length; d++) {
          const diff = data[i][d] - data[j][d];
          sum += diff * diff;
        }
        D[i * n + j] = sum;
        D[j * n + i] = sum;
      }
    }
    return D;
  }

  // --- Compute pairwise affinities P (with binary search for sigma) ---
  function computeP() {
    const perplexity = parseFloat(perpSlider.value);
    const targetEntropy = Math.log(perplexity);
    const Dist = computeDistances(highD);

    // P matrix as flat array
    P = new Float64Array(N * N);

    // har point ke liye sigma find karo via binary search
    for (let i = 0; i < N; i++) {
      let lo = 0.001, hi = 100, mid = 1;
      for (let iter = 0; iter < 50; iter++) {
        mid = (lo + hi) / 2;
        // conditional probability compute karo
        let sumP = 0;
        for (let j = 0; j < N; j++) {
          if (j === i) continue;
          P[i * N + j] = Math.exp(-Dist[i * N + j] / (2 * mid * mid));
          sumP += P[i * N + j];
        }
        if (sumP === 0) sumP = 1e-10;
        // normalize
        let entropy = 0;
        for (let j = 0; j < N; j++) {
          if (j === i) continue;
          P[i * N + j] /= sumP;
          if (P[i * N + j] > 1e-10) entropy -= P[i * N + j] * Math.log(P[i * N + j]);
        }
        // binary search — entropy match karna hai perplexity se
        if (entropy > targetEntropy) hi = mid;
        else lo = mid;
      }
    }
    // symmetrize: P_ij = (P_i|j + P_j|i) / (2N)
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const sym = (P[i * N + j] + P[j * N + i]) / (2 * N);
        // early exaggeration — pehle 100 iterations mein P ko 4x karo
        P[i * N + j] = sym;
        P[j * N + i] = sym;
      }
    }
  }

  // --- Initialize 2D positions randomly ---
  function initPositions() {
    Y = [];
    gains = [];
    prevDY = [];
    for (let i = 0; i < N; i++) {
      Y.push([randn() * 0.01, randn() * 0.01]);
      gains.push([1, 1]);
      prevDY.push([0, 0]);
    }
    iteration = 0;
  }

  // --- t-SNE gradient descent step ---
  function tsneStep() {
    const lr = parseFloat(lrSlider.value);
    const momentum = iteration < 250 ? 0.5 : 0.8;

    // early exaggeration — pehle 100 iterations mein P ko exaggerate karo
    const exag = iteration < 100 ? 4.0 : 1.0;

    // pairwise distances in 2D (Student-t distribution)
    const Qnum = new Float64Array(N * N);
    let sumQ = 0;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = Y[i][0] - Y[j][0];
        const dy = Y[i][1] - Y[j][1];
        const d = 1 / (1 + dx * dx + dy * dy); // Student-t with 1 dof
        Qnum[i * N + j] = d;
        Qnum[j * N + i] = d;
        sumQ += 2 * d;
      }
    }
    if (sumQ === 0) sumQ = 1e-10;

    // gradient compute karo
    const dY = [];
    for (let i = 0; i < N; i++) dY.push([0, 0]);

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const qij = Qnum[i * N + j] / sumQ;
        const pij = P[i * N + j] * exag;
        const mult = 4 * (pij - qij) * Qnum[i * N + j];
        dY[i][0] += mult * (Y[i][0] - Y[j][0]);
        dY[i][1] += mult * (Y[i][1] - Y[j][1]);
      }
    }

    // adaptive gains + momentum update
    for (let i = 0; i < N; i++) {
      for (let d = 0; d < 2; d++) {
        // gain adjust — agar sign same hai toh slow karo, different toh speed up
        const sameSign = (dY[i][d] > 0) === (prevDY[i][d] > 0);
        gains[i][d] = sameSign ? gains[i][d] * 0.8 : gains[i][d] + 0.2;
        if (gains[i][d] < 0.01) gains[i][d] = 0.01;

        prevDY[i][d] = momentum * prevDY[i][d] - lr * gains[i][d] * dY[i][d];
        Y[i][d] += prevDY[i][d];
      }
    }

    // center karo — mean subtract
    let meanX = 0, meanY = 0;
    for (let i = 0; i < N; i++) { meanX += Y[i][0]; meanY += Y[i][1]; }
    meanX /= N; meanY /= N;
    for (let i = 0; i < N; i++) { Y[i][0] -= meanX; Y[i][1] -= meanY; }

    iteration++;
    infoLbl.textContent = `Iter: ${iteration}`;
  }

  // --- Render ---
  function render() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    if (Y.length === 0) return;

    // auto-scale — bounding box dhundho
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < N; i++) {
      if (Y[i][0] < minX) minX = Y[i][0];
      if (Y[i][0] > maxX) maxX = Y[i][0];
      if (Y[i][1] < minY) minY = Y[i][1];
      if (Y[i][1] > maxY) maxY = Y[i][1];
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale = Math.max(rangeX, rangeY);
    const PAD = 30;

    function toX(v) { return PAD + ((v - minX) / scale) * (canvasW - 2 * PAD); }
    function toY(v) { return PAD + ((v - minY) / scale) * (CANVAS_HEIGHT - 2 * PAD); }

    // points draw karo — cluster color ke saath
    for (let i = 0; i < N; i++) {
      const px = toX(Y[i][0]);
      const py = toY(Y[i][1]);
      ctx.fillStyle = COLORS[labels[i]];
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1.0;

    // legend — cluster colors
    ctx.font = "11px 'JetBrains Mono', monospace";
    for (let c = 0; c < N_CLUSTERS; c++) {
      ctx.fillStyle = COLORS[c];
      ctx.fillRect(10, 10 + c * 16, 10, 10);
      ctx.fillStyle = '#aaa';
      ctx.fillText('Cluster ' + (c + 1), 24, 19 + c * 16);
    }

    ctx.fillStyle = '#666';
    ctx.fillText(`${N} points in ${DIMS}D → 2D`, canvasW - 180, 18);
  }

  function restart() {
    generateHighDData();
    computeP();
    initPositions();
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

  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (autoRun) tsneStep();
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
  btnAuto.addEventListener('click', () => {
    autoRun = !autoRun;
    btnAuto.textContent = autoRun ? 'Pause ⏸' : 'Auto ▶';
    if (autoRun && isVisible && !animationId) loop();
  });
  btnStep.addEventListener('click', () => { tsneStep(); render(); });
  btnRestart.addEventListener('click', restart);
  perpSlider.addEventListener('change', () => {
    // perplexity badli — P recompute karna padega
    computeP();
    initPositions();
    infoLbl.textContent = 'Perplexity changed — recomputed P';
  });

  // --- Init ---
  restart();
  render();
}
