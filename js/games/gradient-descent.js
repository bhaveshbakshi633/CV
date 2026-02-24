// ============================================================
// Gradient Descent Optimizer Racing — loss landscape pe optimizers ki race
// Click karke starting point do, 4 optimizers ko ladte dekho
// ============================================================

// yahi entry point hai — container dhundho, heatmap banao, optimizers daudo
export function initGradientDescent() {
  const container = document.getElementById('gradientDescentContainer');
  if (!container) {
    console.warn('gradientDescentContainer nahi mila bhai, gradient descent skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const ACCENT_RGB = '74,158,255';
  const BG_DARK = '#1a1a2e';
  const TRAIL_MAX = 2000;     // itne steps ki trail rakhenge max
  const BALL_RADIUS = 7;      // optimizer ball ka radius
  const GLOW_RADIUS = 14;     // glow effect radius
  const EPS = 1e-8;           // adam/rmsprop ke liye epsilon

  // optimizer colors — har ek ki apni pehchaan
  const OPT_COLORS = {
    sgd:      { hex: '#ef4444', rgb: '239,68,68' },
    momentum: { hex: '#f59e0b', rgb: '245,158,11' },
    rmsprop:  { hex: '#10b981', rgb: '16,185,129' },
    adam:     { hex: '#4a9eff', rgb: '74,158,255' },
  };

  // heatmap color scale — low loss se high loss tak
  // dark blue → cyan → green → yellow → red
  const HEATMAP_STOPS = [
    { t: 0.0, r: 10,  g: 20,  b: 80  },  // gahri neeli
    { t: 0.15, r: 15,  g: 50, b: 140  },  // neeli
    { t: 0.3, r: 20, g: 120, b: 180 },    // cyan types
    { t: 0.45, r: 16, g: 185, b: 129 },   // hari
    { t: 0.6, r: 120, g: 200, b: 40 },    // hara-peela
    { t: 0.7, r: 220, g: 200, b: 30 },    // peela
    { t: 0.85, r: 240, g: 130, b: 20 },   // orange
    { t: 1.0, r: 200, g: 40,  b: 40  },   // laal
  ];

  // --- Loss Functions ---
  // har function: { fn, gradX, gradY, rangeX, rangeY, name, minima }
  const LANDSCAPES = {
    beale: {
      name: 'Beale',
      fn: (x, y) => {
        const a = 1.5 - x + x * y;
        const b = 2.25 - x + x * y * y;
        const c = 2.625 - x + x * y * y * y;
        return a * a + b * b + c * c;
      },
      // analytic gradients — haath se derive kiye hain
      gradX: (x, y) => {
        const a = 1.5 - x + x * y;
        const b = 2.25 - x + x * y * y;
        const c = 2.625 - x + x * y * y * y;
        return 2 * a * (-1 + y) + 2 * b * (-1 + y * y) + 2 * c * (-1 + y * y * y);
      },
      gradY: (x, y) => {
        const a = 1.5 - x + x * y;
        const b = 2.25 - x + x * y * y;
        const c = 2.625 - x + x * y * y * y;
        return 2 * a * x + 2 * b * (2 * x * y) + 2 * c * (3 * x * y * y);
      },
      rangeX: [-4.5, 4.5],
      rangeY: [-4.5, 4.5],
      minima: [{ x: 3, y: 0.5 }],
    },
    rosenbrock: {
      name: 'Rosenbrock',
      fn: (x, y) => {
        const a = 1 - x;
        const b = y - x * x;
        return a * a + 100 * b * b;
      },
      // classic banana valley — gradient steep hai sides pe
      gradX: (x, y) => {
        return -2 * (1 - x) + 100 * 2 * (y - x * x) * (-2 * x);
      },
      gradY: (x, y) => {
        return 100 * 2 * (y - x * x);
      },
      rangeX: [-2, 2],
      rangeY: [-1, 3],
      minima: [{ x: 1, y: 1 }],
    },
    himmelblau: {
      name: 'Himmelblau',
      fn: (x, y) => {
        const a = x * x + y - 11;
        const b = x + y * y - 7;
        return a * a + b * b;
      },
      // 4 minima hain — optimizer kahan jaata hai depends on start
      gradX: (x, y) => {
        const a = x * x + y - 11;
        const b = x + y * y - 7;
        return 2 * a * (2 * x) + 2 * b;
      },
      gradY: (x, y) => {
        const a = x * x + y - 11;
        const b = x + y * y - 7;
        return 2 * a + 2 * b * (2 * y);
      },
      rangeX: [-5, 5],
      rangeY: [-5, 5],
      minima: [
        { x: 3, y: 2 },
        { x: -2.805118, y: 3.131312 },
        { x: -3.779310, y: -3.283186 },
        { x: 3.584428, y: -1.848126 },
      ],
    },
    rastrigin: {
      name: 'Rastrigin',
      fn: (x, y) => {
        return 20 + x * x - 10 * Math.cos(2 * Math.PI * x) +
               y * y - 10 * Math.cos(2 * Math.PI * y);
      },
      // bahut saare local minima — optimizer phasega yahan
      gradX: (x, y) => {
        return 2 * x + 10 * 2 * Math.PI * Math.sin(2 * Math.PI * x);
      },
      gradY: (x, y) => {
        return 2 * y + 10 * 2 * Math.PI * Math.sin(2 * Math.PI * y);
      },
      rangeX: [-5.12, 5.12],
      rangeY: [-5.12, 5.12],
      minima: [{ x: 0, y: 0 }],
    },
  };

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let currentLandscape = 'beale';
  let learningRate = 0.01;
  let stepsPerFrame = 5;
  let animationId = null;
  let isVisible = false;

  // heatmap cache — sirf landscape change ya resize pe rebuild karo
  let heatmapDirty = true;
  let heatmapImageData = null;

  // optimizer states
  let optimizers = {};
  let activeOptimizers = { sgd: true, momentum: true, rmsprop: true, adam: true };
  let hasStarted = false; // kya user ne click kiya hai?

  // --- DOM banao — pehle saaf karo container ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:' + BG_DARK,
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'align-items:center',
    'gap:12px',
    'margin-top:10px',
    'padding:10px 14px',
    'background:rgba(26,26,46,0.7)',
    'border:1px solid rgba(74,158,255,0.12)',
    'border-radius:8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:#ccc',
  ].join(';');
  container.appendChild(controlsDiv);

  // label + element helper
  function makeLabel(text) {
    const lbl = document.createElement('span');
    lbl.textContent = text;
    lbl.style.cssText = 'color:#888;font-size:11px;margin-right:4px;';
    return lbl;
  }

  // landscape dropdown
  const landscapeWrap = document.createElement('div');
  landscapeWrap.style.cssText = 'display:flex;align-items:center;gap:4px;';
  landscapeWrap.appendChild(makeLabel('Landscape'));
  const landscapeSelect = document.createElement('select');
  landscapeSelect.style.cssText = [
    'background:#0a0a1a',
    'color:#ccc',
    'border:1px solid rgba(74,158,255,0.2)',
    'border-radius:4px',
    'padding:4px 8px',
    'font-family:inherit',
    'font-size:12px',
    'cursor:pointer',
    'outline:none',
  ].join(';');
  Object.keys(LANDSCAPES).forEach(key => {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = LANDSCAPES[key].name;
    landscapeSelect.appendChild(opt);
  });
  landscapeSelect.value = currentLandscape;
  landscapeSelect.addEventListener('change', () => {
    currentLandscape = landscapeSelect.value;
    heatmapDirty = true;
    resetOptimizers();
    draw();
  });
  landscapeWrap.appendChild(landscapeSelect);
  controlsDiv.appendChild(landscapeWrap);

  // learning rate slider
  const lrWrap = document.createElement('div');
  lrWrap.style.cssText = 'display:flex;align-items:center;gap:4px;';
  lrWrap.appendChild(makeLabel('LR'));
  const lrSlider = document.createElement('input');
  lrSlider.type = 'range';
  lrSlider.min = '-4';   // 10^-4 = 0.0001
  lrSlider.max = '-1';   // 10^-1 = 0.1
  lrSlider.step = '0.1';
  lrSlider.value = Math.log10(learningRate).toString();
  lrSlider.style.cssText = 'width:80px;accent-color:' + ACCENT + ';cursor:pointer;';
  const lrLabel = document.createElement('span');
  lrLabel.textContent = learningRate.toFixed(4);
  lrLabel.style.cssText = 'color:' + ACCENT + ';min-width:50px;font-size:11px;';
  lrSlider.addEventListener('input', () => {
    learningRate = Math.pow(10, parseFloat(lrSlider.value));
    lrLabel.textContent = learningRate < 0.001
      ? learningRate.toExponential(1)
      : learningRate.toFixed(4);
  });
  lrWrap.appendChild(lrSlider);
  lrWrap.appendChild(lrLabel);
  controlsDiv.appendChild(lrWrap);

  // speed slider
  const speedWrap = document.createElement('div');
  speedWrap.style.cssText = 'display:flex;align-items:center;gap:4px;';
  speedWrap.appendChild(makeLabel('Speed'));
  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '20';
  speedSlider.value = stepsPerFrame.toString();
  speedSlider.style.cssText = 'width:60px;accent-color:' + ACCENT + ';cursor:pointer;';
  const speedLabel = document.createElement('span');
  speedLabel.textContent = stepsPerFrame + 'x';
  speedLabel.style.cssText = 'color:' + ACCENT + ';font-size:11px;min-width:24px;';
  speedSlider.addEventListener('input', () => {
    stepsPerFrame = parseInt(speedSlider.value);
    speedLabel.textContent = stepsPerFrame + 'x';
  });
  speedWrap.appendChild(speedSlider);
  speedWrap.appendChild(speedLabel);
  controlsDiv.appendChild(speedWrap);

  // optimizer checkboxes
  const optWrap = document.createElement('div');
  optWrap.style.cssText = 'display:flex;align-items:center;gap:8px;flex-wrap:wrap;';
  const optNames = ['sgd', 'momentum', 'rmsprop', 'adam'];
  const optLabels = ['SGD', 'Momentum', 'RMSProp', 'Adam'];
  optNames.forEach((key, i) => {
    const lbl = document.createElement('label');
    lbl.style.cssText = 'display:flex;align-items:center;gap:3px;cursor:pointer;color:' + OPT_COLORS[key].hex + ';font-size:11px;';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.style.cssText = 'accent-color:' + OPT_COLORS[key].hex + ';cursor:pointer;';
    cb.addEventListener('change', () => {
      activeOptimizers[key] = cb.checked;
      draw();
    });
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(optLabels[i]));
    optWrap.appendChild(lbl);
  });
  controlsDiv.appendChild(optWrap);

  // reset button
  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset';
  resetBtn.style.cssText = [
    'background:rgba(74,158,255,0.12)',
    'color:' + ACCENT,
    'border:1px solid rgba(74,158,255,0.25)',
    'border-radius:4px',
    'padding:4px 12px',
    'font-family:inherit',
    'font-size:12px',
    'cursor:pointer',
    'transition:background 0.2s',
  ].join(';');
  resetBtn.addEventListener('mouseenter', () => { resetBtn.style.background = 'rgba(74,158,255,0.25)'; });
  resetBtn.addEventListener('mouseleave', () => { resetBtn.style.background = 'rgba(74,158,255,0.12)'; });
  resetBtn.addEventListener('click', () => {
    resetOptimizers();
    draw();
  });
  controlsDiv.appendChild(resetBtn);

  // --- Stats panel ---
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:8px',
    'padding:8px 14px',
    'background:rgba(26,26,46,0.5)',
    'border:1px solid rgba(74,158,255,0.08)',
    'border-radius:8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:#888',
    'min-height:28px',
    'align-items:center',
  ].join(';');
  container.appendChild(statsDiv);

  // --- Utility functions ---

  // heatmap color interpolation — t 0 se 1 ke beech
  function heatmapColor(t) {
    t = Math.max(0, Math.min(1, t));
    // sahi stop dhundho
    let i = 0;
    while (i < HEATMAP_STOPS.length - 2 && HEATMAP_STOPS[i + 1].t < t) i++;
    const s0 = HEATMAP_STOPS[i];
    const s1 = HEATMAP_STOPS[i + 1];
    const frac = (t - s0.t) / (s1.t - s0.t);
    return {
      r: Math.round(s0.r + (s1.r - s0.r) * frac),
      g: Math.round(s0.g + (s1.g - s0.g) * frac),
      b: Math.round(s0.b + (s1.b - s0.b) * frac),
    };
  }

  // world coords (landscape range) se canvas pixel coords mein convert karo
  function worldToCanvas(wx, wy) {
    const ls = LANDSCAPES[currentLandscape];
    const rx = ls.rangeX, ry = ls.rangeY;
    const px = (wx - rx[0]) / (rx[1] - rx[0]) * canvasW;
    const py = (1 - (wy - ry[0]) / (ry[1] - ry[0])) * canvasH; // Y ulta hai
    return { x: px, y: py };
  }

  // canvas pixel coords se world coords mein
  function canvasToWorld(cx, cy) {
    const ls = LANDSCAPES[currentLandscape];
    const rx = ls.rangeX, ry = ls.rangeY;
    const wx = rx[0] + (cx / canvasW) * (rx[1] - rx[0]);
    const wy = ry[0] + (1 - cy / canvasH) * (ry[1] - ry[0]); // Y ulta
    return { x: wx, y: wy };
  }

  // --- Heatmap rendering ---
  function buildHeatmap() {
    const ls = LANDSCAPES[currentLandscape];
    const fn = ls.fn;
    const rx = ls.rangeX, ry = ls.rangeY;

    // pehle loss range figure out karo for normalization
    // sample karke min/max dhundho
    const samples = 80;
    let minLoss = Infinity, maxLoss = -Infinity;
    for (let i = 0; i <= samples; i++) {
      for (let j = 0; j <= samples; j++) {
        const wx = rx[0] + (i / samples) * (rx[1] - rx[0]);
        const wy = ry[0] + (j / samples) * (ry[1] - ry[0]);
        const v = fn(wx, wy);
        if (isFinite(v)) {
          if (v < minLoss) minLoss = v;
          if (v > maxLoss) maxLoss = v;
        }
      }
    }

    // log scale use karo — better contrast milta hai
    const logMin = Math.log(minLoss + 1);
    const logMax = Math.log(maxLoss + 1);
    const logRange = logMax - logMin || 1;

    // pixel level rendering — ImageData use karo speed ke liye
    const w = Math.ceil(canvasW);
    const h = Math.ceil(canvasH);
    if (w <= 0 || h <= 0) return;

    heatmapImageData = ctx.createImageData(w, h);
    const data = heatmapImageData.data;

    // contour levels precompute karo
    // log scale mein evenly spaced contours
    const numContours = 20;
    const contourLevels = [];
    for (let c = 0; c < numContours; c++) {
      const logVal = logMin + (c / (numContours - 1)) * logRange;
      contourLevels.push(Math.exp(logVal) - 1);
    }

    // har pixel ke liye loss calculate karo
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const wx = rx[0] + (px / w) * (rx[1] - rx[0]);
        const wy = ry[0] + (1 - py / h) * (ry[1] - ry[0]);
        const v = fn(wx, wy);
        const logV = Math.log(v + 1);
        const t = (logV - logMin) / logRange;

        const col = heatmapColor(t);
        const idx = (py * w + px) * 4;

        // contour line detection — check karo kya ye pixel contour ke paas hai
        let isContour = false;
        if (px > 0 && py > 0) {
          // left aur top pixel ka loss bhi dekho
          const wxL = rx[0] + ((px - 1) / w) * (rx[1] - rx[0]);
          const vL = fn(wxL, wy);
          const wyT = ry[0] + (1 - (py - 1) / h) * (ry[1] - ry[0]);
          const vT = fn(wx, wyT);

          for (let c = 0; c < contourLevels.length; c++) {
            const cl = contourLevels[c];
            // agar current aur neighbor ke beech contour level aata hai
            if ((v - cl) * (vL - cl) < 0 || (v - cl) * (vT - cl) < 0) {
              isContour = true;
              break;
            }
          }
        }

        if (isContour) {
          // contour lines — halki white lines
          data[idx] = Math.min(255, col.r + 40);
          data[idx + 1] = Math.min(255, col.g + 40);
          data[idx + 2] = Math.min(255, col.b + 40);
          data[idx + 3] = 255;
        } else {
          data[idx] = col.r;
          data[idx + 1] = col.g;
          data[idx + 2] = col.b;
          data[idx + 3] = 255;
        }
      }
    }

    heatmapDirty = false;
  }

  // --- Optimizer initialization ---
  function initOptimizer(key, startX, startY) {
    const state = {
      x: startX,
      y: startY,
      trail: [{ x: startX, y: startY }],
      steps: 0,
      loss: LANDSCAPES[currentLandscape].fn(startX, startY),
    };

    // har optimizer ka apna internal state hota hai
    if (key === 'momentum') {
      state.vx = 0; // velocity x
      state.vy = 0; // velocity y
    } else if (key === 'rmsprop') {
      state.sx = 0; // running avg of squared gradient x
      state.sy = 0; // running avg of squared gradient y
    } else if (key === 'adam') {
      state.mx = 0; // first moment x
      state.my = 0; // first moment y
      state.vx = 0; // second moment x
      state.vy = 0; // second moment y
      state.t = 0;  // timestep counter
    }

    return state;
  }

  function resetOptimizers() {
    hasStarted = false;
    optimizers = {};
    updateStats();
  }

  function startOptimizers(wx, wy) {
    hasStarted = true;
    optimizers = {};
    optNames.forEach(key => {
      optimizers[key] = initOptimizer(key, wx, wy);
    });
    updateStats();
  }

  // --- Optimizer step ---
  // ek step advance karo — gradient compute karo aur position update karo
  function stepOptimizer(key) {
    const state = optimizers[key];
    if (!state) return;

    const ls = LANDSCAPES[currentLandscape];
    const gx = ls.gradX(state.x, state.y);
    const gy = ls.gradY(state.x, state.y);

    // gradient clipping — bahut bada gradient ho toh limit karo
    const gNorm = Math.sqrt(gx * gx + gy * gy);
    const clipThreshold = 100;
    let cgx = gx, cgy = gy;
    if (gNorm > clipThreshold) {
      cgx = gx * clipThreshold / gNorm;
      cgy = gy * clipThreshold / gNorm;
    }

    const lr = learningRate;

    if (key === 'sgd') {
      // vanilla SGD — simple hai boss
      state.x -= lr * cgx;
      state.y -= lr * cgy;

    } else if (key === 'momentum') {
      // momentum — velocity accumulate hoti hai, ball ki tarah
      const beta = 0.9;
      state.vx = beta * state.vx + cgx;
      state.vy = beta * state.vy + cgy;
      state.x -= lr * state.vx;
      state.y -= lr * state.vy;

    } else if (key === 'rmsprop') {
      // RMSProp — adaptive LR per parameter
      const gamma = 0.9;
      state.sx = gamma * state.sx + (1 - gamma) * cgx * cgx;
      state.sy = gamma * state.sy + (1 - gamma) * cgy * cgy;
      state.x -= lr * cgx / (Math.sqrt(state.sx) + EPS);
      state.y -= lr * cgy / (Math.sqrt(state.sy) + EPS);

    } else if (key === 'adam') {
      // Adam — best of both worlds, momentum + adaptive LR
      const beta1 = 0.9, beta2 = 0.999;
      state.t += 1;
      state.mx = beta1 * state.mx + (1 - beta1) * cgx;
      state.my = beta1 * state.my + (1 - beta1) * cgy;
      state.vx = beta2 * state.vx + (1 - beta2) * cgx * cgx;
      state.vy = beta2 * state.vy + (1 - beta2) * cgy * cgy;
      // bias correction — shuru mein moments zero se start hote hain toh compensate karo
      const mxHat = state.mx / (1 - Math.pow(beta1, state.t));
      const myHat = state.my / (1 - Math.pow(beta1, state.t));
      const vxHat = state.vx / (1 - Math.pow(beta2, state.t));
      const vyHat = state.vy / (1 - Math.pow(beta2, state.t));
      state.x -= lr * mxHat / (Math.sqrt(vxHat) + EPS);
      state.y -= lr * myHat / (Math.sqrt(vyHat) + EPS);
    }

    // range mein clamp karo — bahar mat nikalne do
    const rx = ls.rangeX, ry = ls.rangeY;
    state.x = Math.max(rx[0], Math.min(rx[1], state.x));
    state.y = Math.max(ry[0], Math.min(ry[1], state.y));

    state.loss = ls.fn(state.x, state.y);
    state.steps += 1;

    // trail mein add karo
    state.trail.push({ x: state.x, y: state.y });
    if (state.trail.length > TRAIL_MAX) {
      state.trail.shift(); // purani trail hata do
    }
  }

  // --- Stats update ---
  // DOM safely clear karo bina innerHTML ke
  function clearElement(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function updateStats() {
    clearElement(statsDiv);

    if (!hasStarted) {
      const hint = document.createElement('span');
      hint.textContent = 'Click on landscape to drop optimizers';
      hint.style.cssText = 'color:#666;font-style:italic;';
      statsDiv.appendChild(hint);
      return;
    }

    // sabse kam loss wala dhundho — usko highlight karenge
    let minLossKey = null;
    let minLossVal = Infinity;
    optNames.forEach(key => {
      if (activeOptimizers[key] && optimizers[key]) {
        if (optimizers[key].loss < minLossVal) {
          minLossVal = optimizers[key].loss;
          minLossKey = key;
        }
      }
    });

    optNames.forEach(key => {
      if (!activeOptimizers[key] || !optimizers[key]) return;
      const state = optimizers[key];
      const isWinner = key === minLossKey;

      const item = document.createElement('div');
      item.style.cssText = [
        'display:flex',
        'align-items:center',
        'gap:6px',
        'padding:3px 8px',
        'border-radius:4px',
        'background:' + (isWinner ? 'rgba(' + OPT_COLORS[key].rgb + ',0.12)' : 'transparent'),
        'border:1px solid ' + (isWinner ? 'rgba(' + OPT_COLORS[key].rgb + ',0.3)' : 'transparent'),
        'transition:all 0.3s',
      ].join(';');

      // colored dot
      const dot = document.createElement('span');
      dot.style.cssText = [
        'width:8px',
        'height:8px',
        'border-radius:50%',
        'background:' + OPT_COLORS[key].hex,
        'flex-shrink:0',
        isWinner ? 'box-shadow:0 0 6px ' + OPT_COLORS[key].hex : '',
      ].join(';');
      item.appendChild(dot);

      // name + stats
      const txt = document.createElement('span');
      txt.style.cssText = 'color:' + OPT_COLORS[key].hex + ';' + (isWinner ? 'font-weight:bold;' : '');
      const lossStr = state.loss < 0.001
        ? state.loss.toExponential(2)
        : state.loss < 100
          ? state.loss.toFixed(3)
          : state.loss.toFixed(1);
      txt.textContent = optLabels[optNames.indexOf(key)] + ': ' + lossStr + ' (' + state.steps + ' steps)';
      item.appendChild(txt);

      statsDiv.appendChild(item);
    });
  }

  // --- Canvas resize ---
  function resize() {
    dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    canvasW = w;
    canvasH = CANVAS_HEIGHT;
    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    heatmapDirty = true;
  }
  resize();
  window.addEventListener('resize', () => {
    resize();
    draw();
  });

  // --- Drawing ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // heatmap render karo
    if (heatmapDirty || !heatmapImageData) {
      buildHeatmap();
    }
    if (heatmapImageData) {
      // temporary canvas pe ImageData put karo, fir main pe draw karo
      // kyunki putImageData DPR respect nahi karta
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = heatmapImageData.width;
      tmpCanvas.height = heatmapImageData.height;
      const tmpCtx = tmpCanvas.getContext('2d');
      tmpCtx.putImageData(heatmapImageData, 0, 0);
      ctx.drawImage(tmpCanvas, 0, 0, canvasW, canvasH);
    }

    // minima markers draw karo — star marker lagao
    const ls = LANDSCAPES[currentLandscape];
    ls.minima.forEach(m => {
      const p = worldToCanvas(m.x, m.y);
      ctx.save();
      ctx.translate(p.x, p.y);
      // star shape — 5 pointed
      const outerR = 8, innerR = 4;
      ctx.beginPath();
      for (let i = 0; i < 10; i++) {
        const r = i % 2 === 0 ? outerR : innerR;
        const angle = (Math.PI / 2) + (i * Math.PI / 5);
        const sx = Math.cos(angle) * r;
        const sy = -Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    });

    // agar started nahi hai toh instruction dikhao
    if (!hasStarted) {
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillRect(canvasW / 2 - 140, canvasH / 2 - 18, 280, 36);
      ctx.strokeStyle = 'rgba(74,158,255,0.3)';
      ctx.lineWidth = 1;
      ctx.strokeRect(canvasW / 2 - 140, canvasH / 2 - 18, 280, 36);
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = '#aaa';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Click anywhere to drop optimizers', canvasW / 2, canvasH / 2);
      ctx.restore();
      return;
    }

    // optimizer trails aur balls draw karo
    optNames.forEach(key => {
      if (!activeOptimizers[key] || !optimizers[key]) return;
      const state = optimizers[key];
      const color = OPT_COLORS[key];
      const trail = state.trail;
      if (trail.length < 2) return;

      // trail draw karo — fading effect ke saath
      ctx.save();
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // trail ko segments mein draw karo with fading alpha
      const len = trail.length;
      for (let i = 1; i < len; i++) {
        const alpha = 0.1 + 0.7 * (i / len); // puraane points halke, naye bright
        ctx.strokeStyle = 'rgba(' + color.rgb + ',' + alpha.toFixed(2) + ')';
        ctx.beginPath();
        const p0 = worldToCanvas(trail[i - 1].x, trail[i - 1].y);
        const p1 = worldToCanvas(trail[i].x, trail[i].y);
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      }
      ctx.restore();

      // current position pe ball draw karo with glow
      const curr = worldToCanvas(state.x, state.y);
      ctx.save();

      // glow effect — radial gradient
      const glow = ctx.createRadialGradient(curr.x, curr.y, 0, curr.x, curr.y, GLOW_RADIUS);
      glow.addColorStop(0, 'rgba(' + color.rgb + ',0.5)');
      glow.addColorStop(1, 'rgba(' + color.rgb + ',0)');
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(curr.x, curr.y, GLOW_RADIUS, 0, Math.PI * 2);
      ctx.fill();

      // main ball
      ctx.fillStyle = color.hex;
      ctx.strokeStyle = 'rgba(255,255,255,0.7)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(curr.x, curr.y, BALL_RADIUS, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.restore();
    });
  }

  // --- Animation loop ---
  function loop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    if (hasStarted) {
      // har active optimizer ko N steps advance karo
      for (let s = 0; s < stepsPerFrame; s++) {
        optNames.forEach(key => {
          if (activeOptimizers[key] && optimizers[key]) {
            stepOptimizer(key);
          }
        });
      }
      updateStats();
    }

    draw();
    animationId = requestAnimationFrame(loop);
  }

  // --- Click handler — optimizer drop karo ---
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const world = canvasToWorld(cx, cy);
    startOptimizers(world.x, world.y);
    draw();
  });

  // --- Mouse hover — coordinates dikhao cursor ke paas ---
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    // tooltip update karo
    const world = canvasToWorld(cx, cy);
    const ls = LANDSCAPES[currentLandscape];
    const loss = ls.fn(world.x, world.y);
    const lossStr = loss < 0.01
      ? loss.toExponential(2)
      : loss < 100
        ? loss.toFixed(2)
        : loss.toFixed(0);
    canvas.title = 'x=' + world.x.toFixed(2) + ' y=' + world.y.toFixed(2) + ' loss=' + lossStr;
  });

  // --- Visibility observer — jab dikhega tab animate karo ---
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) loop();
    else if (!isVisible && animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }, { threshold: 0.1 });
  obs.observe(container);
  // lab resume: restart loop when focus released
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });

  // initial draw — heatmap toh dikhe at least
  draw();
}
