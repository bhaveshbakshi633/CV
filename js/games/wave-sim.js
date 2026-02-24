// ============================================================
// 1D Wave Equation Simulator — Finite Difference Method
// String pe click/drag karo, wave propagate hota dikhega real-time
// Boundary conditions, damping, presets — full physics demo
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, wave simulate karo
export function initWaveSim() {
  const container = document.getElementById('waveSimContainer');
  if (!container) {
    console.warn('waveSimContainer nahi mila bhai, wave sim skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 300;
  const N = 400; // string ke points — finite difference grid
  const SUBSTEPS = 8; // har frame mein itne physics steps chalao — stability ke liye
  const TWO_PI = Math.PI * 2;

  // --- Physics State ---
  // u[i] = current displacement, uPrev[i] = previous time step
  let u = new Float64Array(N);
  let uPrev = new Float64Array(N);
  let uVel = new Float64Array(N); // velocity tracking — color/thickness ke liye

  // --- Tunable Parameters ---
  let waveSpeed = 1.8; // c — propagation speed
  let damping = 0.998; // energy dissipation factor (1 = no damping)
  let tension = 1.0; // effective tension multiplier

  // boundary condition: 'fixed', 'free', 'absorbing'
  let boundaryMode = 'fixed';

  // --- Interaction State ---
  let isDragging = false;
  let dragIdx = -1; // konse index pe drag ho raha hai
  let dragY = 0; // drag displacement
  let prevDragIdx = -1; // smooth interpolation ke liye

  // --- Animation State ---
  let canvasW = 0;
  let dpr = 1;
  let animationId = null;
  let isVisible = false;
  let isPaused = false;

  // --- DOM Structure ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — string yahan render hogi
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(245,158,11,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls container
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Helper: button banao ---
  function createButton(text, onClick, parent) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(245,158,11,0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(245,158,11,0.25)',
      "font-family:'JetBrains Mono',monospace",
      'transition:all 0.2s ease',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(245,158,11,0.25)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn.dataset.active) {
        btn.style.background = 'rgba(245,158,11,0.1)';
        btn.style.color = '#b0b0b0';
      }
    });
    btn.addEventListener('click', onClick);
    (parent || controlsDiv).appendChild(btn);
    return btn;
  }

  function setButtonActive(btn, active) {
    btn.dataset.active = active ? '1' : '';
    if (active) {
      btn.style.background = 'rgba(245,158,11,0.35)';
      btn.style.color = '#e0e0e0';
      btn.style.borderColor = 'rgba(245,158,11,0.5)';
    } else {
      btn.style.background = 'rgba(245,158,11,0.1)';
      btn.style.color = '#b0b0b0';
      btn.style.borderColor = 'rgba(245,158,11,0.25)';
    }
  }

  // --- Helper: slider banao ---
  function createSlider(label, min, max, step, defaultVal, onChange, parent) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText =
      "color:#b0b0b0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:18px;";
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText =
      'width:70px;height:4px;accent-color:rgba(245,158,11,0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText =
      "color:#b0b0b0;font-size:11px;min-width:32px;font-family:'JetBrains Mono',monospace;";
    valueEl.textContent = parseFloat(defaultVal).toFixed(2);
    wrapper.appendChild(valueEl);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = val.toFixed(2);
      onChange(val);
    });

    (parent || controlsDiv).appendChild(wrapper);
    return { slider, valueEl, wrapper };
  }

  // --- Sliders ---
  createSlider('c', 0.2, 4.0, 0.1, waveSpeed, (v) => {
    waveSpeed = v;
  });
  createSlider('d', 0.98, 1.0, 0.001, damping, (v) => {
    damping = v;
  });
  createSlider('T', 0.1, 3.0, 0.1, tension, (v) => {
    tension = v;
  });

  // separator
  const sep1 = document.createElement('span');
  sep1.style.cssText = 'color:rgba(245,158,11,0.2);font-size:14px;';
  sep1.textContent = '|';
  controlsDiv.appendChild(sep1);

  // --- Boundary Condition Buttons ---
  const bcFixed = createButton('Fixed', () => setBoundary('fixed'));
  const bcFree = createButton('Free', () => setBoundary('free'));
  const bcAbsorb = createButton('Absorb', () => setBoundary('absorbing'));
  const bcButtons = [bcFixed, bcFree, bcAbsorb];
  setButtonActive(bcFixed, true); // default fixed

  function setBoundary(mode) {
    boundaryMode = mode;
    bcButtons.forEach((b) => setButtonActive(b, false));
    if (mode === 'fixed') setButtonActive(bcFixed, true);
    else if (mode === 'free') setButtonActive(bcFree, true);
    else setButtonActive(bcAbsorb, true);
  }

  // separator
  const sep2 = document.createElement('span');
  sep2.style.cssText = 'color:rgba(245,158,11,0.2);font-size:14px;';
  sep2.textContent = '|';
  controlsDiv.appendChild(sep2);

  // --- Preset Buttons ---
  createButton('Pulse', () => applyPreset('pulse'));
  createButton('Sine', () => applyPreset('sine'));
  createButton('Gauss', () => applyPreset('gaussian'));
  createButton('Pluck', () => applyPreset('pluck'));

  // separator
  const sep3 = document.createElement('span');
  sep3.style.cssText = 'color:rgba(245,158,11,0.2);font-size:14px;';
  sep3.textContent = '|';
  controlsDiv.appendChild(sep3);

  // --- Reset Button ---
  createButton('Reset', resetString);

  // --- Canvas Sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvas.width = containerWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- String Reset ---
  function resetString() {
    u.fill(0);
    uPrev.fill(0);
    uVel.fill(0);
  }

  // --- Preset Disturbances ---
  function applyPreset(type) {
    resetString();

    const mid = Math.floor(N / 2);
    const amp = 0.35; // normalized amplitude — canvas height ke relative baad mein scale hoga

    switch (type) {
      case 'pulse': {
        // sharp pulse — center pe ek chhota Gaussian bump
        const width = N * 0.03;
        for (let i = 0; i < N; i++) {
          const dist = (i - mid) / width;
          u[i] = amp * Math.exp(-dist * dist);
        }
        // previous step same rakh — pulse apni jagah se propagate karega
        uPrev.set(u);
        break;
      }

      case 'sine': {
        // standing wave seed — 2 cycles fit honge string mein
        const cycles = 2;
        for (let i = 0; i < N; i++) {
          u[i] = amp * 0.7 * Math.sin((TWO_PI * cycles * i) / N);
        }
        uPrev.set(u);
        break;
      }

      case 'gaussian': {
        // wide Gaussian — center se smoothly phailta hua
        const sigma = N * 0.08;
        for (let i = 0; i < N; i++) {
          const dist = (i - mid) / sigma;
          u[i] = amp * Math.exp(-0.5 * dist * dist);
        }
        // velocity de do — ek direction mein travel karega
        // iske liye previous position thodi shift kar do
        const shiftSigma = N * 0.085;
        for (let i = 0; i < N; i++) {
          const dist = (i - mid - 2) / shiftSigma;
          uPrev[i] = amp * Math.exp(-0.5 * dist * dist);
        }
        break;
      }

      case 'pluck': {
        // triangular pluck — guitar string jaisa
        const pluckPos = Math.floor(N * 0.35); // ek taraf se pluck karo
        for (let i = 0; i < N; i++) {
          if (i <= pluckPos) {
            u[i] = amp * (i / pluckPos);
          } else {
            u[i] = amp * ((N - 1 - i) / (N - 1 - pluckPos));
          }
        }
        uPrev.set(u);
        break;
      }
    }

    // endpoints enforce karo agar fixed boundary hai
    if (boundaryMode === 'fixed') {
      u[0] = 0;
      u[N - 1] = 0;
      uPrev[0] = 0;
      uPrev[N - 1] = 0;
    }
  }

  // --- Physics Step ---
  // 1D wave equation: u_tt = c^2 * u_xx
  // central difference: u_new[i] = 2*u[i] - u_prev[i] + alpha*(u[i+1] - 2*u[i] + u[i-1])
  // alpha = (c * dt / dx)^2 — Courant number squared
  function physicsStep() {
    // dx = 1 (grid spacing normalized), dt chosen for stability
    // Courant condition: c * dt / dx <= 1
    // dt = dx / (c_max * safety_factor)
    const cEff = waveSpeed * Math.sqrt(tension);
    const dt = 0.5 / Math.max(cEff, 0.1); // Courant-safe timestep
    const dx = 1.0;
    const alpha = (cEff * dt / dx) * (cEff * dt / dx);

    // clamp alpha for numerical stability — CFL condition
    const safeAlpha = Math.min(alpha, 0.95);

    const uNew = new Float64Array(N);

    // interior points — central difference scheme
    for (let i = 1; i < N - 1; i++) {
      // spatial laplacian: u_xx = u[i+1] - 2*u[i] + u[i-1]
      const laplacian = u[i + 1] - 2 * u[i] + u[i - 1];
      // time stepping: u_new = 2*u - u_prev + alpha*laplacian
      uNew[i] = 2 * u[i] - uPrev[i] + safeAlpha * laplacian;
      // damping lagao — energy slowly decay hogi
      uNew[i] *= damping;
    }

    // --- Boundary Conditions ---
    switch (boundaryMode) {
      case 'fixed':
        // dono endpoints zero pe fix — string ke dono siron pe nodes
        uNew[0] = 0;
        uNew[N - 1] = 0;
        break;

      case 'free':
        // free boundary — derivative zero (Neumann condition)
        // ghost point method: u[-1] = u[1], u[N] = u[N-2]
        uNew[0] = uNew[1];
        uNew[N - 1] = uNew[N - 2];
        break;

      case 'absorbing':
        // absorbing boundary — wave bahar nikal jaaye bina reflect hue
        // one-sided finite difference: u_new[0] = u[1] + (dt*c - dx)/(dt*c + dx) * (u_new[1] - u[0])
        {
          const r = (cEff * dt - dx) / (cEff * dt + dx);
          uNew[0] = u[1] + r * (uNew[1] - u[0]);
          uNew[N - 1] = u[N - 2] + r * (uNew[N - 2] - u[N - 1]);
        }
        break;
    }

    // agar drag ho raha hai toh waha pe displacement force karo
    if (isDragging && dragIdx >= 0 && dragIdx < N) {
      // drag point aur uske aas paas ke points ko smoothly deform karo
      const brushWidth = Math.max(3, Math.floor(N * 0.02));
      for (let i = dragIdx - brushWidth; i <= dragIdx + brushWidth; i++) {
        if (i >= 0 && i < N) {
          const dist = Math.abs(i - dragIdx) / brushWidth;
          const weight = Math.max(0, 1 - dist * dist); // quadratic falloff
          uNew[i] = uNew[i] * (1 - weight) + dragY * weight;
        }
      }
    }

    // velocity calculate karo — rendering ke liye chahiye
    for (let i = 0; i < N; i++) {
      uVel[i] = uNew[i] - u[i];
    }

    // shift arrays — current becomes previous, new becomes current
    uPrev.set(u);
    u.set(uNew);
  }

  // --- Mouse/Touch Interaction ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  function posToIndex(x) {
    // canvas x position se string index mein convert karo
    // thoda padding rakh dono taraf endpoints ke liye
    const pad = canvasW * 0.04;
    const stringW = canvasW - 2 * pad;
    const t = (x - pad) / stringW;
    return Math.max(0, Math.min(N - 1, Math.round(t * (N - 1))));
  }

  function posToDisplacement(y) {
    // canvas y position se normalized displacement mein convert karo
    // center = 0, top = +0.5, bottom = -0.5
    const centerY = CANVAS_HEIGHT / 2;
    return -(y - centerY) / (CANVAS_HEIGHT * 0.8);
  }

  function handleDragStart(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    isDragging = true;
    dragIdx = posToIndex(pos.x);
    dragY = posToDisplacement(pos.y);
    prevDragIdx = dragIdx;
  }

  function handleDragMove(e) {
    if (!isDragging) return;
    e.preventDefault();
    const pos = getCanvasPos(e);
    const newIdx = posToIndex(pos.x);
    dragY = posToDisplacement(pos.y);

    // agar mouse tez move hua toh beech ke points bhi fill karo — gaps na aayein
    if (prevDragIdx >= 0 && Math.abs(newIdx - prevDragIdx) > 1) {
      const step = newIdx > prevDragIdx ? 1 : -1;
      for (let i = prevDragIdx; i !== newIdx; i += step) {
        if (i >= 0 && i < N) {
          const brushWidth = Math.max(3, Math.floor(N * 0.02));
          for (let j = i - brushWidth; j <= i + brushWidth; j++) {
            if (j >= 0 && j < N) {
              const dist = Math.abs(j - i) / brushWidth;
              const weight = Math.max(0, 1 - dist * dist);
              u[j] = u[j] * (1 - weight * 0.5) + dragY * weight * 0.5;
              uPrev[j] = u[j]; // velocity zero karo drag point pe
            }
          }
        }
      }
    }

    dragIdx = newIdx;
    prevDragIdx = newIdx;
  }

  function handleDragEnd() {
    isDragging = false;
    dragIdx = -1;
    prevDragIdx = -1;
  }

  // mouse events
  canvas.addEventListener('mousedown', handleDragStart);
  canvas.addEventListener('mousemove', handleDragMove);
  canvas.addEventListener('mouseup', handleDragEnd);
  canvas.addEventListener('mouseleave', handleDragEnd);

  // touch events — mobile support
  canvas.addEventListener('touchstart', handleDragStart, { passive: false });
  canvas.addEventListener('touchmove', handleDragMove, { passive: false });
  canvas.addEventListener('touchend', handleDragEnd);
  canvas.addEventListener('touchcancel', handleDragEnd);

  // --- Rendering ---
  function draw() {
    const w = canvasW;
    const h = CANVAS_HEIGHT;
    ctx.clearRect(0, 0, w, h);

    // --- Background Grid ---
    drawGrid(w, h);

    // --- String Rendering ---
    drawString(w, h);

    // --- Fixed Endpoint Dots ---
    drawEndpoints(w, h);

    // --- Drag Indicator ---
    if (isDragging && dragIdx >= 0) {
      drawDragIndicator(w, h);
    }

    // --- Labels ---
    drawLabels(w, h);
  }

  function drawGrid(w, h) {
    ctx.strokeStyle = 'rgba(245,158,11,0.04)';
    ctx.lineWidth = 1;

    // horizontal lines
    const hLines = 8;
    for (let i = 0; i <= hLines; i++) {
      const y = (i / hLines) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // vertical lines
    const vLines = 16;
    for (let i = 0; i <= vLines; i++) {
      const x = (i / vLines) * w;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // center line (zero displacement) — thoda bright
    ctx.strokeStyle = 'rgba(245,158,11,0.1)';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
  }

  function indexToCanvasX(idx, w) {
    const pad = w * 0.04;
    const stringW = w - 2 * pad;
    return pad + (idx / (N - 1)) * stringW;
  }

  function displacementToCanvasY(disp, h) {
    // displacement ko canvas Y mein convert karo
    // scale factor — kitna bada dikhna chahiye
    const scale = h * 0.7;
    return h / 2 - disp * scale;
  }

  function drawString(w, h) {
    // max displacement dhundho — color scaling ke liye
    let maxDisp = 0;
    for (let i = 0; i < N; i++) {
      const absD = Math.abs(u[i]);
      if (absD > maxDisp) maxDisp = absD;
    }
    maxDisp = Math.max(maxDisp, 0.01); // zero divide se bachao

    // --- Smooth curve with quadratic bezier, segmented for color gradient ---
    // har 2-3 points pe ek segment banao — smooth + colored
    const segLen = 2; // kitne grid points per bezier segment
    const points = [];

    // pehle sab points calculate kar lo
    for (let i = 0; i < N; i++) {
      points.push({
        x: indexToCanvasX(i, w),
        y: displacementToCanvasY(u[i], h),
        disp: u[i],
        vel: uVel[i],
      });
    }

    // glow effect ke liye pehle ek blurred wide stroke
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length - 1; i += segLen) {
      const curr = points[i];
      const next = points[Math.min(i + segLen, points.length - 1)];
      const cpx = curr.x;
      const cpy = curr.y;
      const ex = (curr.x + next.x) / 2;
      const ey = (curr.y + next.y) / 2;
      ctx.quadraticCurveTo(cpx, cpy, ex, ey);
    }
    // last point tak line karo
    const lastPt = points[points.length - 1];
    ctx.lineTo(lastPt.x, lastPt.y);
    ctx.strokeStyle = 'rgba(245,158,11,0.12)';
    ctx.lineWidth = 6;
    ctx.shadowColor = 'rgba(245,158,11,0.15)';
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.restore();

    // --- Main string draw — displacement-based color segments ---
    // har chhote segment ko alag color do based on displacement
    for (let i = 0; i < points.length - 1; i++) {
      const p1 = points[i];
      const p2 = points[i + 1];

      // midpoint displacement for color
      const midDisp = (p1.disp + p2.disp) / 2;
      const normalizedDisp = midDisp / maxDisp; // -1 to 1

      // color: blue (negative) -> white (zero) -> red (positive)
      const color = displacementToColor(normalizedDisp);

      // thickness: base + velocity-based variation
      const midVel = Math.abs((p1.vel + p2.vel) / 2);
      const velThickness = Math.min(midVel * 15, 2.5);
      const thickness = 1.8 + velThickness;

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);

      // smooth interpolation — midpoint bezier use karo
      if (i < points.length - 2) {
        const p3 = points[i + 2];
        const mx = (p2.x + p3.x) / 2;
        const my = (p2.y + p3.y) / 2;
        ctx.quadraticCurveTo(p2.x, p2.y, mx, my);
      } else {
        ctx.lineTo(p2.x, p2.y);
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = thickness;
      ctx.lineCap = 'round';
      ctx.stroke();
    }
  }

  function displacementToColor(normalized) {
    // normalized: -1 to 1
    // negative = blue (#3b82f6), zero = white/amber-ish, positive = red (#ef4444)
    const t = Math.max(-1, Math.min(1, normalized));

    let r, g, b, a;

    if (t < 0) {
      // blue to white transition
      const s = -t; // 0 to 1 (0 = white, 1 = blue)
      r = Math.round(245 - s * 186); // 245 -> 59
      g = Math.round(200 - s * 70); // 200 -> 130
      b = Math.round(220 + s * 36); // 220 -> 246
      a = 0.6 + s * 0.3;
    } else {
      // white to red transition
      const s = t; // 0 to 1 (0 = white, 1 = red)
      r = Math.round(245 - s * 6); // 245 -> 239
      g = Math.round(200 - s * 132); // 200 -> 68
      b = Math.round(220 - s * 152); // 220 -> 68
      a = 0.6 + s * 0.3;
    }

    return 'rgba(' + r + ',' + g + ',' + b + ',' + a.toFixed(2) + ')';
  }

  function drawEndpoints(w, h) {
    const pad = w * 0.04;
    const leftX = pad;
    const rightX = w - pad;
    const leftY = displacementToCanvasY(u[0], h);
    const rightY = displacementToCanvasY(u[N - 1], h);
    const radius = 5;

    // endpoint style depends on boundary mode
    let fillColor, strokeColor;
    switch (boundaryMode) {
      case 'fixed':
        fillColor = 'rgba(245,158,11,0.8)';
        strokeColor = '#f59e0b';
        break;
      case 'free':
        fillColor = 'rgba(16,185,129,0.7)';
        strokeColor = '#10b981';
        break;
      case 'absorbing':
        fillColor = 'rgba(139,92,246,0.7)';
        strokeColor = '#8b5cf6';
        break;
      default:
        fillColor = 'rgba(245,158,11,0.8)';
        strokeColor = '#f59e0b';
    }

    // left endpoint
    ctx.beginPath();
    ctx.arc(leftX, leftY, radius, 0, TWO_PI);
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // right endpoint
    ctx.beginPath();
    ctx.arc(rightX, rightY, radius, 0, TWO_PI);
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawDragIndicator(w, h) {
    const x = indexToCanvasX(dragIdx, w);
    const y = displacementToCanvasY(dragY, h);

    // vertical dashed line — drag position dikhao
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = 'rgba(245,158,11,0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // drag point circle
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, TWO_PI);
    ctx.fillStyle = 'rgba(245,158,11,0.3)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(245,158,11,0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawLabels(w, h) {
    ctx.font = "9px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';

    // title label
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.fillText('1D WAVE EQUATION', 8, 14);

    // boundary mode indicator
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(245,158,11,0.35)';
    ctx.fillText('BC: ' + boundaryMode.toUpperCase(), w - 8, 14);

    // energy indicator — total kinetic + potential energy dikhao
    let totalEnergy = 0;
    for (let i = 0; i < N; i++) {
      // kinetic energy ~ 0.5 * v^2
      totalEnergy += 0.5 * uVel[i] * uVel[i];
      // potential energy ~ 0.5 * (du/dx)^2 — strain energy
      if (i < N - 1) {
        const strain = u[i + 1] - u[i];
        totalEnergy += 0.5 * strain * strain;
      }
    }

    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.25)';
    ctx.fillText('E: ' + totalEnergy.toFixed(4), 8, h - 8);

    // hint — agar string flat hai
    if (totalEnergy < 0.0001 && !isDragging) {
      ctx.font = "11px 'JetBrains Mono',monospace";
      ctx.fillStyle = 'rgba(176,176,176,0.2)';
      ctx.textAlign = 'center';
      ctx.fillText('click & drag the string or try a preset', w / 2, h / 2 + 4);
    }
  }

  // --- Animation Loop ---
  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    // multiple substeps per frame — physics stability ke liye
    if (!isPaused) {
      for (let s = 0; s < SUBSTEPS; s++) {
        physicsStep();
      }
    }

    draw();
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf visible hone pe animate karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    resizeCanvas();
    animationId = requestAnimationFrame(animate);
  }

  function stopAnimation() {
    isVisible = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          startAnimation();
        } else {
          stopAnimation();
        }
      });
    },
    { threshold: 0.1 }
  );

  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) animate(); });

  // tab switch pe bhi pause karo — battery bachao
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // initial state — ek Gaussian pulse se shuru karo taaki kuch dikhta rahe
  applyPreset('gaussian');
}
