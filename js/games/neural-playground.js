// ============================================================
// Neural Network Playground — 2D binary classification visualizer
// Decision boundary heatmap ke saath real-time training dikhega
// Click karke data points add kar, network seekhega classify karna
// ============================================================

// yahi main entry point hai — container dhundho, canvas banao, network banao
export function initNeuralPlayground() {
  const container = document.getElementById('neuralPlaygroundContainer');
  if (!container) {
    console.warn('neuralPlaygroundContainer nahi mila bhai, neural playground skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 350;
  const HEATMAP_RES = 5; // heatmap pixel resolution — har 5 CSS pixels pe ek sample
  const HEATMAP_RENDER_INTERVAL = 3; // har 3 frames pe heatmap update kar
  const DATA_RADIUS = 6; // data point ka radius
  const ACCENT = '#4a9eff'; // blue accent color
  const ACCENT_RGB = '74,158,255';
  const ORANGE = '#ff6b35'; // class 1 ka color
  const ORANGE_RGB = '255,107,53';
  const BLUE = '#4a9eff'; // class 0 ka color
  const BLUE_RGB = '74,158,255';

  // --- Network config ---
  // architecture: hidden layers ki sizes — default [4, 4]
  let hiddenLayers = [4, 4];
  let learningRate = 0.3;
  let stepsPerFrame = 5;
  let isTraining = false;

  // --- Data points ---
  // har point: { x: [-1,1], y: [-1,1], cls: 0 or 1 }
  let dataPoints = [];

  // --- Training state ---
  let epoch = 0;
  let currentLoss = 0;
  let frameCount = 0;

  // --- Animation state ---
  let animationId = null;
  let isVisible = false;

  // --- Network weights ---
  // layers[i] = { W: 2D array, b: 1D array }
  // W[j][k] = weight from input j to output k of that layer
  let layers = [];

  // ===================== Neural Network =====================

  // Xavier initialization — variance = 2/(fan_in + fan_out)
  function initNetwork() {
    layers = [];
    const sizes = [2, ...hiddenLayers, 1]; // input 2, hidden layers, output 1

    for (let l = 0; l < sizes.length - 1; l++) {
      const fanIn = sizes[l];
      const fanOut = sizes[l + 1];
      const scale = Math.sqrt(2.0 / (fanIn + fanOut));

      const W = [];
      for (let i = 0; i < fanIn; i++) {
        const row = [];
        for (let j = 0; j < fanOut; j++) {
          row.push((Math.random() * 2 - 1) * scale);
        }
        W.push(row);
      }

      const b = new Array(fanOut).fill(0);
      layers.push({ W, b });
    }

    epoch = 0;
    currentLoss = 0;
  }

  // tanh activation — hidden layers ke liye
  function tanh(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    const e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }

  // sigmoid activation — output layer ke liye
  function sigmoid(x) {
    if (x > 20) return 1;
    if (x < -20) return 0;
    return 1.0 / (1.0 + Math.exp(-x));
  }

  // forward pass — input se output tak activations nikal
  // returns array of activations for each layer (including input)
  function forward(input) {
    const activations = [input]; // layer 0 = input

    let current = input;
    for (let l = 0; l < layers.length; l++) {
      const { W, b } = layers[l];
      const fanOut = b.length;
      const fanIn = current.length;
      const next = new Array(fanOut);

      for (let j = 0; j < fanOut; j++) {
        let sum = b[j];
        for (let i = 0; i < fanIn; i++) {
          sum += current[i] * W[i][j];
        }
        // last layer = sigmoid, baaki = tanh
        next[j] = (l === layers.length - 1) ? sigmoid(sum) : tanh(sum);
      }

      activations.push(next);
      current = next;
    }

    return activations;
  }

  // single point predict kar — sirf output chahiye
  function predict(x, y) {
    const activations = forward([x, y]);
    return activations[activations.length - 1][0];
  }

  // backpropagation + SGD update — ek data point pe
  function trainOnPoint(point) {
    const { x, y, cls } = point;
    const activations = forward([x, y]);
    const output = activations[activations.length - 1][0];

    // binary cross-entropy loss ka derivative: dL/dOutput = (output - target)
    // sigmoid output layer ke liye: dL/dz = output - target (convenient form)
    // yahi chain rule ka starting point hai

    // delta for each layer — back se front jaayenge
    const numLayers = layers.length;
    const deltas = new Array(numLayers);

    // output layer delta — sigmoid + BCE ka combined derivative
    const outputError = output - cls; // dL/dz for output neuron
    deltas[numLayers - 1] = [outputError];

    // hidden layers — backpropagate karo
    for (let l = numLayers - 2; l >= 0; l--) {
      const { W: W_next } = layers[l + 1];
      const nextDelta = deltas[l + 1];
      const layerAct = activations[l + 1]; // is layer ki activations
      const fanOut = layerAct.length;
      const delta = new Array(fanOut);

      for (let j = 0; j < fanOut; j++) {
        // error propagate from next layer
        let err = 0;
        for (let k = 0; k < nextDelta.length; k++) {
          err += nextDelta[k] * W_next[j][k];
        }
        // tanh derivative: 1 - h^2
        const h = layerAct[j];
        delta[j] = err * (1 - h * h);
      }

      deltas[l] = delta;
    }

    // weights update — SGD
    for (let l = 0; l < numLayers; l++) {
      const { W, b } = layers[l];
      const prevAct = activations[l]; // previous layer ki activations (input to this layer)
      const delta = deltas[l];
      const fanIn = prevAct.length;
      const fanOut = delta.length;

      for (let i = 0; i < fanIn; i++) {
        for (let j = 0; j < fanOut; j++) {
          W[i][j] -= learningRate * delta[j] * prevAct[i];
        }
      }
      for (let j = 0; j < fanOut; j++) {
        b[j] -= learningRate * delta[j];
      }
    }
  }

  // ek epoch — saare data points pe ek baar train
  function trainEpoch() {
    if (dataPoints.length === 0) return;

    // data shuffle kar — stochastic gradient descent ke liye zaroori
    const shuffled = [...dataPoints];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    // har point pe train kar
    for (const point of shuffled) {
      trainOnPoint(point);
    }

    // loss calculate kar — binary cross-entropy
    let totalLoss = 0;
    for (const point of dataPoints) {
      const out = predict(point.x, point.y);
      const clamped = Math.max(1e-7, Math.min(1 - 1e-7, out));
      if (point.cls === 1) {
        totalLoss -= Math.log(clamped);
      } else {
        totalLoss -= Math.log(1 - clamped);
      }
    }
    currentLoss = totalLoss / dataPoints.length;
    epoch++;
  }

  // ===================== Preset Datasets =====================
  // sab [-1, 1] range mein generate honge

  function generateCircle(n) {
    const points = [];
    for (let i = 0; i < n; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random();
      const x = Math.cos(angle) * r;
      const y = Math.sin(angle) * r;
      // center ke paas = class 0 (blue), bahar = class 1 (orange)
      const cls = r > 0.5 ? 1 : 0;
      points.push({ x: x * 0.9, y: y * 0.9, cls });
    }
    return points;
  }

  function generateXOR(n) {
    const points = [];
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2 - 1;
      const y = Math.random() * 2 - 1;
      // XOR: top-left & bottom-right = class 1, baaki = class 0
      const cls = (x * y > 0) ? 0 : 1;
      points.push({ x: x * 0.85, y: y * 0.85, cls });
    }
    return points;
  }

  function generateSpiral(n) {
    const points = [];
    const half = Math.floor(n / 2);
    for (let cls = 0; cls < 2; cls++) {
      for (let i = 0; i < half; i++) {
        const t = (i / half) * 1.5 * Math.PI + (cls * Math.PI);
        const r = (i / half) * 0.8 + 0.1;
        const noise = (Math.random() - 0.5) * 0.15;
        const x = r * Math.cos(t) + noise;
        const y = r * Math.sin(t) + noise;
        // clamp to [-1, 1]
        points.push({
          x: Math.max(-1, Math.min(1, x)),
          y: Math.max(-1, Math.min(1, y)),
          cls,
        });
      }
    }
    return points;
  }

  function generateLinear(n) {
    const points = [];
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2 - 1;
      const y = Math.random() * 2 - 1;
      // simple diagonal split + noise
      const cls = (x + y + (Math.random() - 0.5) * 0.3) > 0 ? 1 : 0;
      points.push({ x: x * 0.85, y: y * 0.85, cls });
    }
    return points;
  }

  // ===================== DOM Setup =====================
  // pehle container saaf karo
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.style.cssText = 'width:100%;position:relative;';

  // --- Stats display — epoch aur loss dikhayega ---
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'justify-content:space-between',
    'align-items:center',
    'margin-bottom:8px',
    'padding:0 2px',
  ].join(';');
  container.appendChild(statsDiv);

  const epochLabel = document.createElement('span');
  epochLabel.style.cssText = 'color:#b0b0b0;font-size:13px;font-family:"JetBrains Mono",monospace;';
  epochLabel.textContent = 'Epoch: 0';
  statsDiv.appendChild(epochLabel);

  const lossLabel = document.createElement('span');
  lossLabel.style.cssText = 'color:#b0b0b0;font-size:13px;font-family:"JetBrains Mono",monospace;';
  lossLabel.textContent = 'Loss: —';
  statsDiv.appendChild(lossLabel);

  // --- Main canvas — heatmap + data points ---
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);

  // offscreen canvas — heatmap low resolution mein render karke scale up karenge
  const heatCanvas = document.createElement('canvas');
  const heatCtx = heatCanvas.getContext('2d');

  // --- Controls container ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:12px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Action buttons: Train/Pause, Reset, Clear ---
  const actionsDiv = document.createElement('div');
  actionsDiv.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;';
  controlsDiv.appendChild(actionsDiv);

  // button banane ka helper — consistent styling ke liye
  function createButton(text, onClick, isAccent) {
    const btn = document.createElement('button');
    btn.textContent = text;
    const bgBase = isAccent
      ? `rgba(${ACCENT_RGB},0.15)`
      : `rgba(${ACCENT_RGB},0.06)`;
    const bgHover = isAccent
      ? `rgba(${ACCENT_RGB},0.3)`
      : `rgba(${ACCENT_RGB},0.15)`;
    const borderColor = isAccent
      ? `rgba(${ACCENT_RGB},0.4)`
      : `rgba(${ACCENT_RGB},0.2)`;

    btn.style.cssText = [
      'padding:5px 14px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:' + bgBase,
      'color:#b0b0b0',
      'border:1px solid ' + borderColor,
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
      'white-space:nowrap',
    ].join(';');

    btn.addEventListener('mouseenter', () => {
      btn.style.background = bgHover;
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = bgBase;
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    return btn;
  }

  // Train / Pause button
  const trainBtn = createButton('Train', () => {
    isTraining = !isTraining;
    trainBtn.textContent = isTraining ? 'Pause' : 'Train';
  }, true);
  actionsDiv.appendChild(trainBtn);

  // Reset network button — weights reinitialize, epoch reset
  const resetBtn = createButton('Reset', () => {
    isTraining = false;
    trainBtn.textContent = 'Train';
    initNetwork();
    updateStats();
  }, false);
  actionsDiv.appendChild(resetBtn);

  // Clear data button — saara data hata do
  const clearBtn = createButton('Clear Data', () => {
    dataPoints = [];
    isTraining = false;
    trainBtn.textContent = 'Train';
    initNetwork();
    updateStats();
  }, false);
  actionsDiv.appendChild(clearBtn);

  // --- Preset dataset buttons ---
  const presetsDiv = document.createElement('div');
  presetsDiv.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;';
  controlsDiv.appendChild(presetsDiv);

  const presets = [
    { name: 'Circle', fn: () => generateCircle(120) },
    { name: 'XOR', fn: () => generateXOR(120) },
    { name: 'Spiral', fn: () => generateSpiral(120) },
    { name: 'Linear', fn: () => generateLinear(100) },
  ];

  presets.forEach((preset) => {
    const btn = createButton(preset.name, () => {
      dataPoints = preset.fn();
      isTraining = false;
      trainBtn.textContent = 'Train';
      initNetwork();
      updateStats();
    }, false);
    presetsDiv.appendChild(btn);
  });

  // --- Sliders row ---
  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:16px',
    'margin-top:8px',
    'align-items:center',
    'width:100%',
  ].join(';');
  container.appendChild(slidersDiv);

  // slider banane ka helper
  function createSlider(label, min, max, step, defaultVal, formatFn, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText = 'width:80px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText = 'color:#b0b0b0;font-size:12px;min-width:28px;font-family:"JetBrains Mono",monospace;';
    valueEl.textContent = formatFn(defaultVal);
    wrapper.appendChild(valueEl);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = formatFn(val);
      onChange(val);
    });

    slidersDiv.appendChild(wrapper);
    return { slider, valueEl };
  }

  // learning rate slider
  createSlider('LR', 0.01, 1.0, 0.01, learningRate,
    (v) => v.toFixed(2),
    (v) => { learningRate = v; }
  );

  // speed slider — steps per frame
  const speedSteps = [1, 5, 10, 50];
  const speedSlider = createSlider('Speed', 0, 3, 1, 1,
    (v) => speedSteps[Math.round(v)] + 'x',
    (v) => { stepsPerFrame = speedSteps[Math.round(v)]; }
  );

  // --- Architecture buttons row ---
  const archDiv = document.createElement('div');
  archDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:6px',
    'margin-top:8px',
    'align-items:center',
    'width:100%',
  ].join(';');
  container.appendChild(archDiv);

  const archLabel = document.createElement('span');
  archLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:"JetBrains Mono",monospace;margin-right:4px;';
  archLabel.textContent = 'Architecture:';
  archDiv.appendChild(archLabel);

  // architecture options — different hidden layer configs
  const architectures = [
    { name: '[4]', layers: [4] },
    { name: '[6]', layers: [6] },
    { name: '[4,4]', layers: [4, 4] },
    { name: '[6,4]', layers: [6, 4] },
    { name: '[8,6]', layers: [8, 6] },
    { name: '[6,4,2]', layers: [6, 4, 2] },
    { name: '[8,6,4]', layers: [8, 6, 4] },
  ];

  let activeArchBtn = null;

  architectures.forEach((arch) => {
    const btn = document.createElement('button');
    btn.textContent = arch.name;
    const isDefault = JSON.stringify(arch.layers) === JSON.stringify(hiddenLayers);

    function styleBtn(active) {
      btn.style.cssText = [
        'padding:4px 10px',
        'font-size:11px',
        'border-radius:5px',
        'cursor:pointer',
        'background:' + (active ? `rgba(${ACCENT_RGB},0.25)` : `rgba(${ACCENT_RGB},0.06)`),
        'color:' + (active ? '#e0e0e0' : '#808080'),
        'border:1px solid ' + (active ? `rgba(${ACCENT_RGB},0.5)` : `rgba(${ACCENT_RGB},0.15)`),
        'font-family:"JetBrains Mono",monospace',
        'transition:all 0.2s ease',
      ].join(';');
    }

    styleBtn(isDefault);
    if (isDefault) activeArchBtn = btn;

    btn.addEventListener('mouseenter', () => {
      if (btn !== activeArchBtn) {
        btn.style.background = `rgba(${ACCENT_RGB},0.15)`;
        btn.style.color = '#b0b0b0';
      }
    });
    btn.addEventListener('mouseleave', () => {
      if (btn !== activeArchBtn) {
        styleBtn(false);
      }
    });

    btn.addEventListener('click', () => {
      // purana active button deactivate kar
      if (activeArchBtn) {
        const prevBtn = activeArchBtn;
        activeArchBtn = null;
        // re-style as inactive
        prevBtn.style.background = `rgba(${ACCENT_RGB},0.06)`;
        prevBtn.style.color = '#808080';
        prevBtn.style.borderColor = `rgba(${ACCENT_RGB},0.15)`;
      }

      hiddenLayers = [...arch.layers];
      activeArchBtn = btn;
      styleBtn(true);

      // network reset kar nayi architecture ke saath
      isTraining = false;
      trainBtn.textContent = 'Train';
      initNetwork();
      updateStats();
    });

    archDiv.appendChild(btn);
  });

  // --- Click hint ---
  const hintDiv = document.createElement('div');
  hintDiv.style.cssText = [
    'color:#606060',
    'font-size:11px',
    'font-family:"JetBrains Mono",monospace',
    'margin-top:6px',
    'text-align:center',
  ].join(';');
  hintDiv.textContent = 'Left click = class 0 (blue) · Right/Shift+click = class 1 (orange)';
  container.appendChild(hintDiv);

  // ===================== Canvas Sizing =====================
  // DPR handle karna zaroori hai — retina display pe blur nahi hona chahiye

  let cssWidth = 0;
  let cssHeight = CANVAS_HEIGHT;
  let dpr = 1;

  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    cssWidth = container.clientWidth;
    cssHeight = CANVAS_HEIGHT;

    // main canvas
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
    canvas.style.width = cssWidth + 'px';
    canvas.style.height = cssHeight + 'px';

    // heatmap offscreen canvas — low resolution
    const heatW = Math.ceil(cssWidth / HEATMAP_RES);
    const heatH = Math.ceil(cssHeight / HEATMAP_RES);
    heatCanvas.width = heatW;
    heatCanvas.height = heatH;
  }

  resizeCanvas();

  // ===================== Mouse/Touch Events =====================
  // left click = class 0 (blue), right click / shift+click = class 1 (orange)

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    // pixel position ko [-1, 1] range mein convert kar
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const x = (px / rect.width) * 2 - 1;
    const y = (py / rect.height) * 2 - 1;
    return { x, y };
  }

  // right click prevent karo — warna context menu aa jaayega
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
  });

  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const { x, y } = getCanvasCoords(e);
    // right click ya shift+click = class 1 (orange)
    const cls = (e.button === 2 || e.shiftKey) ? 1 : 0;
    dataPoints.push({ x, y, cls });
  });

  // touch pe bhi kaam kare — mobile support
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const { x, y } = getCanvasCoords(e);
    // touch = class 0 by default
    dataPoints.push({ x, y, cls: 0 });
  }, { passive: false });

  // ===================== Stats Update =====================

  function updateStats() {
    epochLabel.textContent = 'Epoch: ' + epoch;
    if (epoch === 0 || dataPoints.length === 0) {
      lossLabel.textContent = 'Loss: —';
    } else {
      lossLabel.textContent = 'Loss: ' + currentLoss.toFixed(4);
    }
  }

  // ===================== Heatmap Rendering =====================
  // low resolution mein network output calculate karke color banao
  // fir main canvas pe scaled draw kar do — performance hack

  // color interpolation — prediction value (0 to 1) se color banao
  // 0 = blue (class 0), 1 = orange (class 1)
  // smooth gradient with nice dark tones
  function predToColor(pred) {
    // blue = [74, 158, 255], orange = [255, 107, 53]
    // dark center (pred ~0.5) for uncertainty
    // lerp between colors with darkness at boundary

    // confidence = kitna sure hai network
    const confidence = Math.abs(pred - 0.5) * 2; // 0 at boundary, 1 at extremes

    // base alpha — confidence zyada toh color zyada saturated
    // minimum brightness rakh taaki fully dark na ho
    const alpha = 0.15 + confidence * 0.55;

    let r, g, b;
    if (pred < 0.5) {
      // blue side — class 0
      const t = 1 - pred * 2; // 1 at pred=0, 0 at pred=0.5
      r = 74 * t;
      g = 158 * t;
      b = 255 * t;
    } else {
      // orange side — class 1
      const t = (pred - 0.5) * 2; // 0 at pred=0.5, 1 at pred=1
      r = 255 * t;
      g = 107 * t;
      b = 53 * t;
    }

    return { r: Math.round(r), g: Math.round(g), b: Math.round(b), a: alpha };
  }

  // heatmap render karo offscreen canvas pe
  function renderHeatmap() {
    const heatW = heatCanvas.width;
    const heatH = heatCanvas.height;
    const imageData = heatCtx.createImageData(heatW, heatH);
    const data = imageData.data;

    for (let py = 0; py < heatH; py++) {
      // y coordinate: canvas top = -1, bottom = +1
      const y = (py / (heatH - 1)) * 2 - 1;

      for (let px = 0; px < heatW; px++) {
        // x coordinate: canvas left = -1, right = +1
        const x = (px / (heatW - 1)) * 2 - 1;

        const pred = predict(x, y);
        const color = predToColor(pred);

        const idx = (py * heatW + px) * 4;
        data[idx] = color.r;
        data[idx + 1] = color.g;
        data[idx + 2] = color.b;
        data[idx + 3] = Math.round(color.a * 255);
      }
    }

    heatCtx.putImageData(imageData, 0, 0);
  }

  // ===================== Main Render =====================

  function render() {
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // canvas saaf kar
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    // heatmap draw kar — scaled up from low-res offscreen canvas
    // smoothing enable kar taaki gradient smooth dikhe
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(heatCanvas, 0, 0, cssWidth, cssHeight);

    // subtle grid lines — reference ke liye
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    // center crosshair
    ctx.beginPath();
    ctx.moveTo(cssWidth / 2, 0);
    ctx.lineTo(cssWidth / 2, cssHeight);
    ctx.moveTo(0, cssHeight / 2);
    ctx.lineTo(cssWidth, cssHeight / 2);
    ctx.stroke();

    // decision boundary line — pred = 0.5 ke paas subtle contour
    // ye expensive hai toh har frame nahi karenge, heatmap se implicit hai

    // data points draw kar
    for (const point of dataPoints) {
      // [-1, 1] se canvas coords mein convert kar
      const px = (point.x + 1) / 2 * cssWidth;
      const py = (point.y + 1) / 2 * cssHeight;

      // outer ring
      ctx.beginPath();
      ctx.arc(px, py, DATA_RADIUS + 1.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fill();

      // inner filled circle
      ctx.beginPath();
      ctx.arc(px, py, DATA_RADIUS, 0, Math.PI * 2);
      if (point.cls === 0) {
        ctx.fillStyle = BLUE;
        ctx.strokeStyle = 'rgba(74,158,255,0.8)';
      } else {
        ctx.fillStyle = ORANGE;
        ctx.strokeStyle = 'rgba(255,107,53,0.8)';
      }
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // inner highlight — glass effect ke liye
      ctx.beginPath();
      ctx.arc(px - 1.5, py - 1.5, DATA_RADIUS * 0.4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.fill();
    }

    // agar koi data nahi hai toh hint text dikhao canvas pe
    if (dataPoints.length === 0) {
      ctx.fillStyle = 'rgba(240,240,240,0.15)';
      ctx.font = '14px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Click to add data points', cssWidth / 2, cssHeight / 2 - 10);
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText('or choose a preset dataset below', cssWidth / 2, cssHeight / 2 + 12);
      ctx.textAlign = 'start';
    }
  }

  // ===================== Animation Loop =====================

  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    // agar visible nahi hai toh rAF schedule mat kar — IntersectionObserver wapas start karega
    if (!isVisible) {
      animationId = null;
      return;
    }

    // training chal rahi hai toh steps chala
    if (isTraining && dataPoints.length > 0) {
      for (let i = 0; i < stepsPerFrame; i++) {
        trainEpoch();
      }
      updateStats();
    }

    // heatmap har HEATMAP_RENDER_INTERVAL frames pe update kar — performance
    frameCount++;
    if (frameCount % HEATMAP_RENDER_INTERVAL === 0 || frameCount <= 1) {
      renderHeatmap();
    }

    render();

    animationId = requestAnimationFrame(animate);
  }

  function startAnimation() {
    if (animationId === null) {
      frameCount = 0;
      animationId = requestAnimationFrame(animate);
    }
  }

  function stopAnimation() {
    if (animationId !== null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // ===================== IntersectionObserver =====================
  // sirf visible hone pe animate kar — CPU bach jaayega

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        const wasVisible = isVisible;
        isVisible = entry.isIntersecting;
        if (isVisible && !wasVisible) {
          // abhi visible hua — animation shuru kar
          startAnimation();
        } else if (!isVisible && wasVisible) {
          // abhi invisible hua — animation rok de
          stopAnimation();
        }
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) animate(); });

  // ===================== Initialization =====================
  initNetwork();
  resizeCanvas();
  renderHeatmap(); // initial heatmap render — random weights ka visualization
  updateStats();

  // resize pe canvas update kar
  window.addEventListener('resize', () => {
    resizeCanvas();
    renderHeatmap();
  });

  // animation start kar — agar visible hai tabhi chalegi
  startAnimation();
}
