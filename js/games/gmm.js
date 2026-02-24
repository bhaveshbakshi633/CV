// ============================================================
// Gaussian Mixture Model — EM algorithm visualization
// Click se data points daalo, K Gaussians fit karke dekho
// E-step soft assignments, M-step update means/covariances
// Ellipses breathe karti hain — organic feel aata hai
// ============================================================

export function initGMM() {
  const container = document.getElementById('gmmContainer');
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

  // cluster colors — distinct hsl colors
  const COLORS = [
    { h: 210, s: 90, l: 60 }, // neela — #4a9eff types
    { h: 350, s: 85, l: 60 }, // laal
    { h: 140, s: 75, l: 50 }, // hara
    { h: 40, s: 95, l: 55 },  // peela
    { h: 280, s: 80, l: 65 }, // purple
  ];

  function clrStr(idx, a) {
    const c = COLORS[idx % COLORS.length];
    return a !== undefined ? `hsla(${c.h},${c.s}%,${c.l}%,${a})` : `hsl(${c.h},${c.s}%,${c.l}%)`;
  }

  // --- state ---
  let K = 3;
  let points = []; // [{x, y, resp: []}] — resp = responsibilities per cluster
  let means = []; // [{x, y}]
  let covs = []; // [{xx, xy, yy}] — covariance matrices (2x2 symmetric)
  let weights = []; // mixing weights pi_k
  let emIter = 0;
  let autoRun = false;
  let autoTimer = null;
  let converged = false;

  // Gaussian PDF — bivariate
  function gaussPDF(px, py, mx, my, cov) {
    const dx = px - mx, dy = py - my;
    const det = cov.xx * cov.yy - cov.xy * cov.xy;
    if (det <= 1e-10) return 1e-30;
    const invDet = 1 / det;
    // inverse covariance
    const ixx = cov.yy * invDet;
    const iyy = cov.xx * invDet;
    const ixy = -cov.xy * invDet;
    const exponent = -0.5 * (dx * dx * ixx + 2 * dx * dy * ixy + dy * dy * iyy);
    const norm = 1 / (2 * Math.PI * Math.sqrt(det));
    return norm * Math.exp(exponent);
  }

  // initialize GMM — K random means from data, identity covariances
  function initGMM_params() {
    if (points.length < K) return;
    means = [];
    covs = [];
    weights = [];
    emIter = 0;
    converged = false;

    // K random data points ko means banao
    const shuffled = [...points].sort(() => Math.random() - 0.5);
    for (let k = 0; k < K; k++) {
      means.push({ x: shuffled[k].x, y: shuffled[k].y });
      covs.push({ xx: 2000, xy: 0, yy: 2000 }); // badi initial covariance
      weights.push(1 / K);
    }

    // initial responsibilities clear karo
    for (const p of points) p.resp = new Array(K).fill(1 / K);
  }

  // E-step — har point ke liye responsibilities compute karo
  function eStep() {
    for (const p of points) {
      let totalProb = 0;
      const probs = [];
      for (let k = 0; k < K; k++) {
        const prob = weights[k] * gaussPDF(p.x, p.y, means[k].x, means[k].y, covs[k]);
        probs.push(prob);
        totalProb += prob;
      }
      // normalize — responsibilities sum to 1
      if (totalProb > 0) {
        p.resp = probs.map(pr => pr / totalProb);
      } else {
        p.resp = new Array(K).fill(1 / K);
      }
    }
  }

  // M-step — means, covariances, weights update karo
  function mStep() {
    for (let k = 0; k < K; k++) {
      // effective count — kitne points is cluster mein (soft)
      let Nk = 0;
      for (const p of points) Nk += p.resp[k];
      if (Nk < 1e-6) {
        // dead cluster — random point pe reset karo
        const rp = points[Math.floor(Math.random() * points.length)];
        means[k] = { x: rp.x, y: rp.y };
        covs[k] = { xx: 2000, xy: 0, yy: 2000 };
        weights[k] = 1 / K;
        continue;
      }

      // mean update
      let mx = 0, my = 0;
      for (const p of points) {
        mx += p.resp[k] * p.x;
        my += p.resp[k] * p.y;
      }
      means[k].x = mx / Nk;
      means[k].y = my / Nk;

      // covariance update
      let cxx = 0, cxy = 0, cyy = 0;
      for (const p of points) {
        const dx = p.x - means[k].x;
        const dy = p.y - means[k].y;
        cxx += p.resp[k] * dx * dx;
        cxy += p.resp[k] * dx * dy;
        cyy += p.resp[k] * dy * dy;
      }
      covs[k].xx = cxx / Nk + 1; // +1 regularization — singularity se bachao
      covs[k].xy = cxy / Nk;
      covs[k].yy = cyy / Nk + 1;

      // weight update
      weights[k] = Nk / points.length;
    }
  }

  // ek EM step chala
  function emStep() {
    if (points.length < K || means.length === 0) return;
    if (converged) return;

    const oldMeans = means.map(m => ({ x: m.x, y: m.y }));
    eStep();
    mStep();
    emIter++;

    // convergence check — means kitna hile
    let totalDelta = 0;
    for (let k = 0; k < K; k++) {
      const dx = means[k].x - oldMeans[k].x;
      const dy = means[k].y - oldMeans[k].y;
      totalDelta += Math.sqrt(dx * dx + dy * dy);
    }
    if (totalDelta < 0.1 && emIter > 5) converged = true;
  }

  // ellipse draw karo — covariance matrix se eigenvalues/vectors nikalo
  function drawEllipse(mx, my, cov, color, sigma) {
    // eigenvalue decomposition of 2x2 covariance matrix
    const a = cov.xx, b = cov.xy, d = cov.yy;
    const trace = a + d;
    const det = a * d - b * b;
    const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
    const lambda1 = trace / 2 + disc;
    const lambda2 = Math.max(0.1, trace / 2 - disc);

    // eigenvector direction — rotation angle
    const angle = b !== 0 ? Math.atan2(lambda1 - a, b) : (a >= d ? 0 : Math.PI / 2);

    // semi-axes — sigma * sqrt(eigenvalue)
    const rx = sigma * Math.sqrt(lambda1);
    const ry = sigma * Math.sqrt(lambda2);

    ctx.save();
    ctx.translate(mx, my);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = sigma === 1 ? 2 : 1;
    ctx.stroke();
    ctx.restore();
  }

  // dominant cluster nikal point ke liye
  function dominantCluster(p) {
    if (!p.resp || p.resp.length === 0) return 0;
    let maxR = 0, maxK = 0;
    for (let k = 0; k < p.resp.length; k++) {
      if (p.resp[k] > maxR) { maxR = p.resp[k]; maxK = k; }
    }
    return maxK;
  }

  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // gaussian ellipses draw karo — pehle background mein
    if (means.length > 0) {
      for (let k = 0; k < K; k++) {
        // 2-sigma ellipse — halki fill
        drawEllipse(means[k].x, means[k].y, covs[k], clrStr(k, 0.15), 2);
        // 1-sigma ellipse — bright
        drawEllipse(means[k].x, means[k].y, covs[k], clrStr(k, 0.5), 1);
      }
    }

    // data points draw karo — dominant cluster ke color mein
    for (const p of points) {
      const k = dominantCluster(p);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
      if (means.length > 0) {
        // responsibility ke hisaab se alpha set karo
        const alpha = 0.3 + p.resp[k] * 0.7;
        ctx.fillStyle = clrStr(k, alpha);
      } else {
        ctx.fillStyle = 'rgba(200,200,200,0.5)';
      }
      ctx.fill();
    }

    // means draw karo — bade X markers
    for (let k = 0; k < means.length; k++) {
      const mx = means[k].x, my = means[k].y;
      ctx.save();
      ctx.shadowColor = clrStr(k, 1);
      ctx.shadowBlur = 8;
      ctx.strokeStyle = clrStr(k, 1);
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      // X shape
      ctx.beginPath();
      ctx.moveTo(mx - 6, my - 6); ctx.lineTo(mx + 6, my + 6);
      ctx.moveTo(mx + 6, my - 6); ctx.lineTo(mx - 6, my + 6);
      ctx.stroke();
      ctx.restore();
    }

    // info text
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = "11px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    const status = converged ? 'Converged' : (means.length > 0 ? 'Running' : 'Click to add points');
    ctx.fillText(`EM Iter: ${emIter}  |  K: ${K}  |  Points: ${points.length}  |  ${status}`, 10, 18);

    if (points.length === 0) {
      ctx.fillStyle = 'rgba(74,158,255,0.3)';
      ctx.font = "13px 'JetBrains Mono',monospace";
      ctx.textAlign = 'center';
      ctx.fillText('Click to add data points, ya Generate dabao', canvasW / 2, CANVAS_HEIGHT / 2);
    }

    // legend — weights dikhao
    if (means.length > 0) {
      for (let k = 0; k < K; k++) {
        const ly = 35 + k * 16;
        ctx.fillStyle = clrStr(k, 0.8);
        ctx.beginPath();
        ctx.arc(16, ly, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = "10px 'JetBrains Mono',monospace";
        ctx.textAlign = 'left';
        ctx.fillText(`w=${weights[k].toFixed(2)}`, 26, ly + 4);
      }
    }
  }

  // random data generate karo — K blobs
  function generateRandomData() {
    points = [];
    const numPerCluster = 30 + Math.floor(Math.random() * 20);
    for (let k = 0; k < K; k++) {
      const cx = 80 + Math.random() * (canvasW - 160);
      const cy = 60 + Math.random() * (CANVAS_HEIGHT - 120);
      const spread = 30 + Math.random() * 40;
      // random rotation for each blob
      const angle = Math.random() * Math.PI;
      const scaleX = spread * (0.5 + Math.random());
      const scaleY = spread * (0.5 + Math.random());
      for (let i = 0; i < numPerCluster; i++) {
        // Box-Muller transform — Gaussian random
        const u1 = Math.random(), u2 = Math.random();
        const z1 = Math.sqrt(-2 * Math.log(u1 || 0.001)) * Math.cos(2 * Math.PI * u2);
        const z2 = Math.sqrt(-2 * Math.log(u1 || 0.001)) * Math.sin(2 * Math.PI * u2);
        // rotate
        const rx = z1 * scaleX * Math.cos(angle) - z2 * scaleY * Math.sin(angle);
        const ry = z1 * scaleX * Math.sin(angle) + z2 * scaleY * Math.cos(angle);
        const px = Math.max(5, Math.min(canvasW - 5, cx + rx));
        const py = Math.max(5, Math.min(CANVAS_HEIGHT - 5, cy + ry));
        points.push({ x: px, y: py, resp: new Array(K).fill(1 / K) });
      }
    }
  }

  // click to add point
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvasW / rect.width);
    const my = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height);
    points.push({ x: mx, y: my, resp: new Array(K).fill(1 / K) });
    // agar EM chal raha tha toh naye point ko bhi assign karo
    if (means.length > 0) {
      converged = false;
    }
  });

  // --- controls ---
  // K selector buttons
  const kLabel = document.createElement('span');
  kLabel.style.cssText = "color:#ccc;font:12px 'JetBrains Mono',monospace";
  kLabel.textContent = 'K: ';
  ctrl.appendChild(kLabel);

  for (let k = 2; k <= 5; k++) {
    const btn = mkBtn(ctrl, String(k), 'gmmK' + k);
    if (k === K) { btn.style.background = '#4a9eff'; btn.style.color = '#111'; }
    btn.addEventListener('click', () => {
      K = k;
      // reset button styles
      for (let j = 2; j <= 5; j++) {
        const b = document.getElementById('gmmK' + j);
        if (b) { b.style.background = j === K ? '#4a9eff' : '#333'; b.style.color = j === K ? '#111' : '#ccc'; }
      }
      // re-init agar points hain
      if (points.length >= K) {
        // resp arrays resize karo
        for (const p of points) p.resp = new Array(K).fill(1 / K);
        initGMM_params();
      }
    });
  }

  const stepBtn = mkBtn(ctrl, 'Step EM', 'gmmStep');
  stepBtn.addEventListener('click', () => {
    if (points.length < K) return;
    if (means.length === 0) initGMM_params();
    emStep();
  });

  const autoBtn = mkBtn(ctrl, 'Auto Run', 'gmmAuto');
  autoBtn.addEventListener('click', () => {
    autoRun = !autoRun;
    autoBtn.textContent = autoRun ? 'Stop' : 'Auto Run';
    autoBtn.style.background = autoRun ? '#4a9eff' : '#333';
    autoBtn.style.color = autoRun ? '#111' : '#ccc';
    if (autoRun && points.length >= K && means.length === 0) initGMM_params();
  });

  const genBtn = mkBtn(ctrl, 'Generate', 'gmmGen');
  genBtn.addEventListener('click', () => {
    generateRandomData();
    means = []; covs = []; weights = [];
    emIter = 0; converged = false; autoRun = false;
    autoBtn.textContent = 'Auto Run';
    autoBtn.style.background = '#333';
    autoBtn.style.color = '#ccc';
  });

  const resetBtn = mkBtn(ctrl, 'Reset', 'gmmReset');
  resetBtn.addEventListener('click', () => {
    points = []; means = []; covs = []; weights = [];
    emIter = 0; converged = false; autoRun = false;
    autoBtn.textContent = 'Auto Run';
    autoBtn.style.background = '#333';
    autoBtn.style.color = '#ccc';
  });

  // --- main loop ---
  let frameCount = 0;
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    frameCount++;
    // auto-run mein har 10 frames pe ek EM step — slow animation
    if (autoRun && frameCount % 10 === 0 && !converged) {
      if (means.length === 0 && points.length >= K) initGMM_params();
      emStep();
    }

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
}
