// ============================================================
// Kalman Filter Tracker — noisy measurements se true position track karo
// Blue = truth, Red = noisy sensors, Green = Kalman estimate with uncertainty ellipse
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, Kalman chalao
export function initKalman() {
  const container = document.getElementById('kalmanContainer');
  if (!container) {
    console.warn('kalmanContainer nahi mila bhai, Kalman tracker skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 300;
  const DOT_RADIUS = 4;
  const TRAIL_MAX = 200;            // trail mein kitne points rakhne hain
  const FIGURE8_SPEED = 0.008;      // figure-8 kitni tezi se chale
  const FIGURE8_SCALE_X = 0.32;     // X axis pe kitna stretch ho
  const FIGURE8_SCALE_Y = 0.35;     // Y axis pe kitna stretch ho

  // --- State variables ---
  let canvasW = 0, canvasH = 0;
  let animationId = null;
  let isVisible = false;
  let time = 0;

  // true position state
  let trueX = 0, trueY = 0;
  let trueVx = 0, trueVy = 0;
  let truePrevX = 0, truePrevY = 0;

  // trails store karne ke liye
  let trueTrail = [];               // [{x, y}, ...]
  let measTrail = [];               // [{x, y}, ...]
  let estTrail = [];                // [{x, y}, ...]

  // Kalman filter state — [x, y, vx, vy]
  let X = [0, 0, 0, 0];            // state estimate
  // covariance matrix P — 4x4 identity se shuru
  let P = [
    [100, 0, 0, 0],
    [0, 100, 0, 0],
    [0, 0, 10, 0],
    [0, 0, 0, 10],
  ];

  // tunable parameters — sliders se change honge
  let processNoise = 0.5;           // Q scalar
  let measurementNoise = 10;        // R scalar
  let measurementRate = 10;         // Hz — kitni baar measurement aaye per second
  let lastMeasTime = 0;             // last measurement ka time
  let frameCount = 0;

  // stats
  let posError = 0;
  let velError = 0;
  let traceP = 0;

  // --- DOM structure banate hain ---
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(59,130,246,0.2)',
    'border-radius:8px',
    'cursor:default',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls + stats section
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:12px',
    'align-items:center',
    'justify-content:space-between',
  ].join(';');
  container.appendChild(controlsDiv);

  // sliders ka container
  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:14px;flex:1;min-width:280px;';
  controlsDiv.appendChild(slidersDiv);

  // stats display
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = 'display:flex;gap:16px;font-family:monospace;font-size:11px;color:#888;flex-wrap:wrap;';
  controlsDiv.appendChild(statsDiv);

  // --- Slider banane ka helper ---
  function createSlider(label, min, max, step, defaultVal, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = 'color:#b0b0b0;font-size:12px;font-weight:600;min-width:18px;font-family:monospace;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText = 'width:80px;height:4px;accent-color:rgba(59,130,246,0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText = 'color:#b0b0b0;font-size:11px;min-width:28px;font-family:monospace;';
    valueEl.textContent = parseFloat(defaultVal).toFixed(1);
    wrapper.appendChild(valueEl);

    // slider change hone pe update karo
    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = val.toFixed(1);
      onChange(val);
    });

    slidersDiv.appendChild(wrapper);
    return { slider, valueEl };
  }

  // teen sliders — Q, R, measurement rate
  createSlider('Q', 0.01, 5, 0.01, processNoise, (v) => { processNoise = v; });
  createSlider('R', 1, 50, 0.5, measurementNoise, (v) => { measurementNoise = v; });
  createSlider('Hz', 1, 30, 1, measurementRate, (v) => { measurementRate = v; });

  // stats elements
  const statPosErr = document.createElement('span');
  const statVelErr = document.createElement('span');
  const statTrace = document.createElement('span');
  statsDiv.appendChild(statPosErr);
  statsDiv.appendChild(statVelErr);
  statsDiv.appendChild(statTrace);

  // --- Canvas resize ---
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvasW = rect.width;
    canvasH = rect.height;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // --- Gaussian random number — Box-Muller transform ---
  // measurement noise ke liye chahiye — proper Gaussian distribution
  function gaussRandom(mean, sigma) {
    let u1 = Math.random();
    let u2 = Math.random();
    // zero pe log nahi le sakte, safety check
    while (u1 === 0) u1 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + sigma * z;
  }

  // --- Matrix operations — Kalman ke liye chahiye ---
  // 4x4 matrix multiply
  function matMul(A, B) {
    const n = A.length;
    const m = B[0].length;
    const p = B.length;
    const C = [];
    for (let i = 0; i < n; i++) {
      C[i] = [];
      for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let k = 0; k < p; k++) {
          sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
      }
    }
    return C;
  }

  // matrix transpose
  function matTranspose(A) {
    const rows = A.length;
    const cols = A[0].length;
    const T = [];
    for (let j = 0; j < cols; j++) {
      T[j] = [];
      for (let i = 0; i < rows; i++) {
        T[j][i] = A[i][j];
      }
    }
    return T;
  }

  // matrix add
  function matAdd(A, B) {
    return A.map((row, i) => row.map((val, j) => val + B[i][j]));
  }

  // matrix subtract
  function matSub(A, B) {
    return A.map((row, i) => row.map((val, j) => val - B[i][j]));
  }

  // matrix scale
  function matScale(A, s) {
    return A.map(row => row.map(val => val * s));
  }

  // 2x2 matrix inverse — Kalman gain calculate karne ke liye
  // S = H*P*H' + R hai 2x2, toh simple formula use kar sakte hain
  function mat2x2Inv(M) {
    const a = M[0][0], b = M[0][1], c = M[1][0], d = M[1][1];
    const det = a * d - b * c;
    if (Math.abs(det) < 1e-10) return [[1, 0], [0, 1]]; // singular hai toh identity de do
    return [
      [d / det, -b / det],
      [-c / det, a / det]
    ];
  }

  // identity matrix banao
  function eye(n) {
    const I = [];
    for (let i = 0; i < n; i++) {
      I[i] = [];
      for (let j = 0; j < n; j++) {
        I[i][j] = i === j ? 1 : 0;
      }
    }
    return I;
  }

  // --- True position update — figure-8 path pe chale ---
  function updateTruePosition() {
    truePrevX = trueX;
    truePrevY = trueY;

    time += FIGURE8_SPEED;

    // figure-8 (lemniscate of Bernoulli) — parametric form
    // x = sin(t), y = sin(t)*cos(t) wala pattern
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const scaleX = canvasW * FIGURE8_SCALE_X;
    const scaleY = canvasH * FIGURE8_SCALE_Y;

    trueX = cx + scaleX * Math.sin(time);
    trueY = cy + scaleY * Math.sin(time) * Math.cos(time);

    // velocity estimate — finite difference
    trueVx = trueX - truePrevX;
    trueVy = trueY - truePrevY;

    // trail mein add kar
    trueTrail.push({ x: trueX, y: trueY });
    if (trueTrail.length > TRAIL_MAX) trueTrail.shift();
  }

  // --- Kalman Filter prediction step ---
  // constant velocity model: x(k+1) = x(k) + vx*dt, y(k+1) = y(k) + vy*dt
  function kalmanPredict() {
    const dt = 1; // frame-based dt

    // state transition matrix F — constant velocity model
    // [x]     [1 0 dt 0] [x]
    // [y]   = [0 1 0 dt] [y]
    // [vx]    [0 0 1  0] [vx]
    // [vy]    [0 0 0  1] [vy]
    const F = [
      [1, 0, dt, 0],
      [0, 1, 0, dt],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];

    // process noise Q — kitna model pe trust hai
    const q = processNoise;
    const Q = [
      [q * dt * dt / 2, 0, q * dt, 0],
      [0, q * dt * dt / 2, 0, q * dt],
      [q * dt, 0, q, 0],
      [0, q * dt, 0, q],
    ];

    // predict state: X = F * X
    const newX = [
      F[0][0] * X[0] + F[0][2] * X[2],
      F[1][1] * X[1] + F[1][3] * X[3],
      X[2],
      X[3],
    ];
    X = newX;

    // predict covariance: P = F * P * F' + Q
    const FP = matMul(F, P);
    const FT = matTranspose(F);
    P = matAdd(matMul(FP, FT), Q);
  }

  // --- Kalman Filter update step — measurement aaya toh state correct karo ---
  function kalmanUpdate(measX, measY) {
    // measurement matrix H — sirf position measure kar rahe hain, velocity nahi
    // z = H * x, H = [[1,0,0,0],[0,1,0,0]]
    const H = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ];

    // measurement noise R
    const r = measurementNoise;
    const R = [
      [r, 0],
      [0, r],
    ];

    // innovation: y = z - H*x
    const z = [measX, measY];
    const Hx = [
      H[0][0] * X[0] + H[0][1] * X[1] + H[0][2] * X[2] + H[0][3] * X[3],
      H[1][0] * X[0] + H[1][1] * X[1] + H[1][2] * X[2] + H[1][3] * X[3],
    ];
    const innovation = [z[0] - Hx[0], z[1] - Hx[1]];

    // S = H * P * H' + R — innovation covariance (2x2)
    const HT = matTranspose(H);
    const HP = matMul(H, P);       // 2x4
    const HPHT = matMul(HP, HT);   // 2x2
    const S = matAdd(HPHT, R);      // 2x2

    // Kalman gain: K = P * H' * S^(-1)   (4x2)
    const Sinv = mat2x2Inv(S);
    const PHT = matMul(P, HT);     // 4x2
    const K = matMul(PHT, Sinv);   // 4x2

    // update state: X = X + K * innovation
    X = [
      X[0] + K[0][0] * innovation[0] + K[0][1] * innovation[1],
      X[1] + K[1][0] * innovation[0] + K[1][1] * innovation[1],
      X[2] + K[2][0] * innovation[0] + K[2][1] * innovation[1],
      X[3] + K[3][0] * innovation[0] + K[3][1] * innovation[1],
    ];

    // update covariance: P = (I - K*H) * P
    const KH = matMul(K, H);       // 4x4
    const IKH = matSub(eye(4), KH); // 4x4
    P = matMul(IKH, P);
  }

  // --- Drawing functions ---

  // trail draw kar — points ka fading path
  function drawTrail(trail, color, dotMode) {
    if (trail.length < 2) return;

    if (dotMode) {
      // dots mode — measurement points ke liye
      for (let i = 0; i < trail.length; i++) {
        const alpha = 0.15 + 0.6 * (i / trail.length);
        ctx.beginPath();
        ctx.arc(trail[i].x, trail[i].y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = color.replace('ALPHA', alpha.toFixed(2));
        ctx.fill();
      }
    } else {
      // line mode — smooth path
      ctx.beginPath();
      ctx.moveTo(trail[0].x, trail[0].y);
      for (let i = 1; i < trail.length; i++) {
        const alpha = 0.1 + 0.7 * (i / trail.length);
        ctx.strokeStyle = color.replace('ALPHA', alpha.toFixed(2));
        ctx.lineWidth = 1.5;
        ctx.lineTo(trail[i].x, trail[i].y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(trail[i].x, trail[i].y);
      }
    }
  }

  // uncertainty ellipse draw kar — 2-sigma confidence region
  function drawUncertaintyEllipse(x, y) {
    // P matrix se position uncertainty nikaal — top-left 2x2 block
    const px = P[0][0];
    const py = P[1][1];
    const pxy = P[0][1];

    // eigenvalues nikaal 2x2 matrix ke — ellipse ke semi-axes honge
    const trace = px + py;
    const det = px * py - pxy * pxy;
    const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
    const lambda1 = Math.max(0.1, trace / 2 + disc);
    const lambda2 = Math.max(0.1, trace / 2 - disc);

    // 2-sigma = 2 * sqrt(eigenvalue)
    const rx = 2 * Math.sqrt(lambda1);
    const ry = 2 * Math.sqrt(lambda2);

    // rotation angle — eigenvector direction
    const angle = 0.5 * Math.atan2(2 * pxy, px - py);

    // cap kar do — bahut bada nahi dikhana
    const maxR = 150;
    const drawRx = Math.min(rx, maxR);
    const drawRy = Math.min(ry, maxR);

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.ellipse(0, 0, drawRx, drawRy, 0, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(16,185,129,0.35)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.fillStyle = 'rgba(16,185,129,0.06)';
    ctx.fill();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // legend draw kar — corner mein
  function drawLegend() {
    const lx = 12;
    const ly = 16;
    const gap = 16;
    const dotR = 4;

    ctx.font = '11px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    // true — blue
    ctx.beginPath();
    ctx.arc(lx, ly, dotR, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(59,130,246,0.8)';
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText('True', lx + 10, ly);

    // measurement — red
    ctx.beginPath();
    ctx.arc(lx, ly + gap, dotR, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(239,68,68,0.8)';
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText('Measured', lx + 10, ly + gap);

    // estimate — green
    ctx.beginPath();
    ctx.arc(lx, ly + gap * 2, dotR, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16,185,129,0.8)';
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText('Estimate', lx + 10, ly + gap * 2);
  }

  // --- Main render function ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // trails draw kar
    drawTrail(trueTrail, 'rgba(59,130,246,ALPHA)', false);
    drawTrail(measTrail, 'rgba(239,68,68,ALPHA)', true);
    drawTrail(estTrail, 'rgba(16,185,129,ALPHA)', false);

    // uncertainty ellipse — estimate ke around
    if (X[0] !== 0 || X[1] !== 0) {
      drawUncertaintyEllipse(X[0], X[1]);
    }

    // true position — blue dot
    ctx.beginPath();
    ctx.arc(trueX, trueY, DOT_RADIUS + 1, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(59,130,246,0.9)';
    ctx.fill();
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Kalman estimate — green dot (thoda bada)
    if (X[0] !== 0 || X[1] !== 0) {
      ctx.beginPath();
      ctx.arc(X[0], X[1], DOT_RADIUS + 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(16,185,129,0.9)';
      ctx.fill();
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // legend
    drawLegend();
  }

  // --- Main animation loop ---
  function animate() {
    animationId = requestAnimationFrame(animate);
    if (!isVisible) return;

    frameCount++;

    // true position update kar — figure-8 pe chale
    updateTruePosition();

    // Kalman predict step — har frame pe
    kalmanPredict();

    // measurement check — rate ke hisaab se
    const measInterval = Math.max(1, Math.round(60 / measurementRate));
    if (frameCount % measInterval === 0) {
      // noisy measurement generate kar — true position + Gaussian noise
      const sigma = Math.sqrt(measurementNoise);
      const measX = gaussRandom(trueX, sigma);
      const measY = gaussRandom(trueY, sigma);

      // measurement trail mein add kar
      measTrail.push({ x: measX, y: measY });
      if (measTrail.length > TRAIL_MAX) measTrail.shift();

      // Kalman update — measurement incorporate kar
      kalmanUpdate(measX, measY);
    }

    // estimate trail mein add kar
    estTrail.push({ x: X[0], y: X[1] });
    if (estTrail.length > TRAIL_MAX) estTrail.shift();

    // stats calculate kar
    const dx = trueX - X[0];
    const dy = trueY - X[1];
    posError = Math.sqrt(dx * dx + dy * dy);
    const dvx = trueVx - X[2];
    const dvy = trueVy - X[3];
    velError = Math.sqrt(dvx * dvx + dvy * dvy);
    traceP = P[0][0] + P[1][1] + P[2][2] + P[3][3];

    // stats display update kar
    statPosErr.textContent = 'Pos err: ' + posError.toFixed(1) + 'px';
    statPosErr.style.color = posError < 10 ? '#10b981' : posError < 25 ? '#f59e0b' : '#ef4444';
    statVelErr.textContent = 'Vel err: ' + velError.toFixed(2);
    statTrace.textContent = 'Tr(P): ' + traceP.toFixed(1);

    // draw karo sab
    draw();
  }

  // --- IntersectionObserver — sirf visible hone pe animate kar ---
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      isVisible = entry.isIntersecting;
      if (isVisible && !animationId) {
        resizeCanvas();
        // initial state set kar — canvas center pe shuru kar
        trueX = canvasW / 2;
        trueY = canvasH / 2;
        X = [trueX, trueY, 0, 0];
        animate();
      }
    });
  }, { threshold: 0.1 });
  observer.observe(container);

  // resize listener
  window.addEventListener('resize', () => {
    if (isVisible) resizeCanvas();
  });

  // initial setup
  resizeCanvas();
  trueX = canvasW / 2;
  trueY = canvasH / 2;
  X = [trueX, trueY, 0, 0];
  animate();
}
