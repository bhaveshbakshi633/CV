// ============================================================
// Quantum Tunneling — 1D Schrödinger equation, Crank-Nicolson method
// Gaussian wave packet barrier se guzarta hai — Thomas algorithm se solve
// ============================================================

export function initQuantumTunneling() {
  const container = document.getElementById('quantumTunnelingContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  let animationId = null, isVisible = false, canvasW = 0;

  // grid parameters
  const N = 800;
  const dx = 1.0 / N;
  const dt = 0.5 * dx * dx; // stability ke liye chhota dt

  // physics parameters
  let barrierWidth = 0.04;
  let barrierHeight = 150;
  let k0 = 200; // wave packet ka momentum
  let barrierCenter = 0.5;

  // wave function — complex (real + imaginary parts)
  let psiR = new Float64Array(N);
  let psiI = new Float64Array(N);
  let potential = new Float64Array(N);

  // Crank-Nicolson tridiagonal coefficients
  let alpha = dt / (4 * dx * dx);

  // --- DOM banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(245,158,11,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  function createSlider(label, min, max, step, val, onChange) {
    const w = document.createElement('div');
    w.style.cssText = 'display:flex;align-items:center;gap:5px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#6b6b6b;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    lbl.textContent = label;
    w.appendChild(lbl);
    const sl = document.createElement('input');
    sl.type = 'range'; sl.min = String(min); sl.max = String(max); sl.step = String(step); sl.value = String(val);
    sl.style.cssText = 'width:70px;height:4px;accent-color:#f59e0b;cursor:pointer;';
    w.appendChild(sl);
    const vl = document.createElement('span');
    vl.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:32px;";
    vl.textContent = Number(val).toFixed(step < 1 ? 2 : 0);
    w.appendChild(vl);
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      vl.textContent = v.toFixed(step < 1 ? 2 : 0);
      onChange(v);
    });
    ctrl.appendChild(w);
  }

  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = "padding:5px 12px;font-size:11px;border-radius:6px;cursor:pointer;background:rgba(245,158,11,0.1);color:#b0b0b0;border:1px solid rgba(245,158,11,0.25);font-family:'JetBrains Mono',monospace;transition:all 0.2s ease;";
    btn.addEventListener('mouseenter', () => { btn.style.background = 'rgba(245,158,11,0.25)'; btn.style.color = '#e0e0e0'; });
    btn.addEventListener('mouseleave', () => { btn.style.background = 'rgba(245,158,11,0.1)'; btn.style.color = '#b0b0b0'; });
    btn.addEventListener('click', onClick);
    ctrl.appendChild(btn);
  }

  createSlider('width', 0.01, 0.1, 0.005, barrierWidth, (v) => { barrierWidth = v; initSim(); });
  createSlider('height', 0, 400, 10, barrierHeight, (v) => { barrierHeight = v; initSim(); });
  createSlider('k\u2080', 50, 500, 10, k0, (v) => { k0 = v; initSim(); });
  createButton('Reset', initSim);

  // --- potential barrier setup ---
  function setupPotential() {
    potential.fill(0);
    const left = Math.floor((barrierCenter - barrierWidth / 2) * N);
    const right = Math.floor((barrierCenter + barrierWidth / 2) * N);
    for (let i = Math.max(0, left); i < Math.min(N, right); i++) {
      potential[i] = barrierHeight;
    }
  }

  // --- Gaussian wave packet initialize karo ---
  function initSim() {
    setupPotential();
    const x0 = 0.25; // starting position — baayein taraf se aayega
    const sigma = 0.03; // packet width
    // psi(x) = A * exp(-(x-x0)^2 / (2*sigma^2)) * exp(i*k0*x)
    let norm = 0;
    for (let i = 0; i < N; i++) {
      const x = i * dx;
      const gauss = Math.exp(-((x - x0) * (x - x0)) / (2 * sigma * sigma));
      psiR[i] = gauss * Math.cos(k0 * x);
      psiI[i] = gauss * Math.sin(k0 * x);
      norm += psiR[i] * psiR[i] + psiI[i] * psiI[i];
    }
    // normalize karo
    norm = Math.sqrt(norm * dx);
    for (let i = 0; i < N; i++) {
      psiR[i] /= norm;
      psiI[i] /= norm;
    }
    alpha = dt / (4 * dx * dx);
  }

  // --- Crank-Nicolson step: Thomas algorithm se solve ---
  // (I + i*H*dt/2) * psi_new = (I - i*H*dt/2) * psi_old
  // H = -d²/dx² + V(x), finite difference mein tridiagonal hai
  function cnStep() {
    // tridiagonal solve dono real/imag ke liye coupled hai
    // Real part update: (using imaginary part)
    // -alpha * psi_new_R[i-1] + (1 + 2*alpha + V*dt/2) * psi_new_R[i] - alpha * psi_new_R[i+1] = RHS
    // RHS involves psi_I (current imaginary part)

    // Pehle imaginary part advance karo using real part
    const a = new Float64Array(N);
    const b = new Float64Array(N);
    const c = new Float64Array(N);
    const d = new Float64Array(N);

    // Step 1: advance psiI by dt/2 using psiR
    for (let i = 1; i < N - 1; i++) {
      const vi = potential[i] * dt * 0.5;
      a[i] = alpha;
      b[i] = -(1 + 2 * alpha + vi);
      c[i] = alpha;
      // RHS: (I - i*H*dt/2) acting on psiI — means use psiR for source
      d[i] = -alpha * psiR[i - 1] + (1 - 2 * alpha - vi) * psiR[i] - alpha * psiR[i + 1];
    }
    b[0] = -1; c[0] = 0; d[0] = 0;
    a[N - 1] = 0; b[N - 1] = -1; d[N - 1] = 0;

    // Thomas algorithm — forward sweep
    const cp = new Float64Array(N);
    const dp = new Float64Array(N);
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (let i = 1; i < N; i++) {
      const m = b[i] - a[i] * cp[i - 1];
      cp[i] = c[i] / m;
      dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    // back substitution — naya psiI nikalo
    const newI = new Float64Array(N);
    newI[N - 1] = dp[N - 1];
    for (let i = N - 2; i >= 0; i--) {
      newI[i] = dp[i] - cp[i] * newI[i + 1];
    }

    // Step 2: advance psiR by dt/2 using updated psiI
    for (let i = 1; i < N - 1; i++) {
      const vi = potential[i] * dt * 0.5;
      a[i] = -alpha;
      b[i] = 1 + 2 * alpha + vi;
      c[i] = -alpha;
      d[i] = alpha * newI[i - 1] + (1 - 2 * alpha - vi) * newI[i] + alpha * newI[i + 1];
    }
    b[0] = 1; c[0] = 0; d[0] = 0;
    a[N - 1] = 0; b[N - 1] = 1; d[N - 1] = 0;

    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (let i = 1; i < N; i++) {
      const m = b[i] - a[i] * cp[i - 1];
      cp[i] = c[i] / m;
      dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    const newR = new Float64Array(N);
    newR[N - 1] = dp[N - 1];
    for (let i = N - 2; i >= 0; i--) {
      newR[i] = dp[i] - cp[i] * newR[i + 1];
    }

    psiR.set(newR);
    psiI.set(newI);
  }

  // --- resize ---
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- draw ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    const h = CANVAS_HEIGHT;
    const baseY = h * 0.65; // probability density ka zero level
    const scale = h * 0.55; // kitna bada dikhana hai

    // grid lines — subtle
    ctx.strokeStyle = 'rgba(245,158,11,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 8; i++) {
      const y = (i / 8) * h;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvasW, y); ctx.stroke();
    }

    // barrier draw karo — red rectangle
    const bLeft = Math.floor((barrierCenter - barrierWidth / 2) * canvasW);
    const bRight = Math.floor((barrierCenter + barrierWidth / 2) * canvasW);
    const bHeight = Math.min(barrierHeight / 400 * h * 0.5, h * 0.5);
    ctx.fillStyle = 'rgba(239,68,68,0.25)';
    ctx.fillRect(bLeft, baseY - bHeight, bRight - bLeft, bHeight);
    ctx.strokeStyle = 'rgba(239,68,68,0.6)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(bLeft, baseY - bHeight, bRight - bLeft, bHeight);

    // barrier label
    ctx.font = "9px 'JetBrains Mono',monospace";
    ctx.fillStyle = 'rgba(239,68,68,0.5)';
    ctx.textAlign = 'center';
    ctx.fillText('V=' + barrierHeight, (bLeft + bRight) / 2, baseY - bHeight - 5);

    // |psi|^2 draw karo — filled cyan area
    ctx.beginPath();
    ctx.moveTo(0, baseY);
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * canvasW;
      const prob = psiR[i] * psiR[i] + psiI[i] * psiI[i];
      const y = baseY - prob * scale;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(canvasW, baseY);
    ctx.closePath();

    // gradient fill — cyan se transparent
    const grad = ctx.createLinearGradient(0, baseY - scale * 0.3, 0, baseY);
    grad.addColorStop(0, 'rgba(34,211,238,0.5)');
    grad.addColorStop(1, 'rgba(34,211,238,0.05)');
    ctx.fillStyle = grad;
    ctx.fill();

    // outline bhi draw karo — brighter
    ctx.beginPath();
    ctx.moveTo(0, baseY);
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * canvasW;
      const prob = psiR[i] * psiR[i] + psiI[i] * psiI[i];
      ctx.lineTo(x, baseY - prob * scale);
    }
    ctx.strokeStyle = 'rgba(34,211,238,0.8)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // real part bhi faintly dikhao — phase information ke liye
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * canvasW;
      const y = baseY - psiR[i] * scale * 0.3;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(245,158,11,0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // baseline
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, baseY); ctx.lineTo(canvasW, baseY); ctx.stroke();

    // probability left/right dikhao
    let probLeft = 0, probRight = 0;
    const mid = Math.floor(barrierCenter * N);
    for (let i = 0; i < N; i++) {
      const p = (psiR[i] * psiR[i] + psiI[i] * psiI[i]) * dx;
      if (i < mid) probLeft += p;
      else probRight += p;
    }

    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(34,211,238,0.5)';
    ctx.fillText('P(left)=' + probLeft.toFixed(3), 8, 20);
    ctx.textAlign = 'right';
    ctx.fillText('P(right)=' + probRight.toFixed(3), canvasW - 8, 20);

    // title
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.fillText('QUANTUM TUNNELING', 8, h - 10);
  }

  // --- init ---
  initSim();

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    // multiple physics steps per frame — speed ke liye
    for (let s = 0; s < 8; s++) {
      cnStep();
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
