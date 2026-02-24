// ============================================================
// Interactive Backpropagation Visualizer
// Forward pass, backward pass, gradient descent — sab step by step dikhega
// XOR dataset pe 2→3→1 network train hota hai — educational demo hai ye
// ============================================================

// yahi main entry point hai — container dhundho, canvas banao, backprop dikhao
export function initBackprop() {
  const container = document.getElementById('backpropContainer');
  if (!container) {
    console.warn('backpropContainer nahi mila bhai, backprop visualizer skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 380;
  const ACCENT = '#4a9eff';
  const POS_COLOR = '#4a9eff';   // positive weight = blue
  const NEG_COLOR = '#ef4444';   // negative weight = red
  const NODE_RADIUS = 22;
  const LAYER_LABELS = ['Input', 'Hidden (tanh)', 'Output (\u03C3)'];
  const ANIM_DELAY = 120;        // ms delay between layers during animation

  // XOR dataset — classic non-linear problem, 4 examples
  const XOR_DATA = [
    { input: [0, 0], target: 0 },
    { input: [0, 1], target: 1 },
    { input: [1, 0], target: 1 },
    { input: [1, 1], target: 0 },
  ];

  // --- Network architecture: 2 → 3 → 1 ---
  const LAYER_SIZES = [2, 3, 1];

  // --- State variables ---
  let canvasW = 0, canvasH = 0;
  let animationId = null;
  let isVisible = false;

  // network weights aur biases
  let W1, b1, W2, b2; // W1: 2x3, b1: 3, W2: 3x1, b2: 1

  // current activations aur gradients — visualization ke liye store karte hain
  let activations = [null, null, null]; // har layer ki activations
  let preActivations = [null, null];    // hidden aur output ka pre-activation (z)
  let gradients = {
    dW1: null, db1: null, dW2: null, db2: null,
    dHidden: null, dOutput: null,
  };

  // training state
  let currentExample = 0;
  let learningRate = 0.5;
  let loss = null;
  let output = null;
  let trainStep = 0;
  let lossHistory = [];
  const MAX_LOSS_HISTORY = 100;

  // animation state — kaunsa phase chal raha hai
  // 'idle' | 'forward' | 'backward' | 'update'
  let phase = 'idle';
  let animLayerIdx = -1;     // animation mein kaunsi layer highlight ho rahi hai
  let animTimer = null;
  let autoTrain = false;
  let autoTrainTimer = null;
  let showGradients = false; // backward ke baad gradients dikhne chahiye

  // node positions — canvas pe kahan draw karne hain
  let nodePositions = [];    // [layer][node] = {x, y}

  // --- DOM structure banate hain ---
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — network yahan draw hoga
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.2)',
    'border-radius:8px',
    'cursor:default',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // info bar — input, target, output, loss dikhega (safe DOM construction)
  const infoBar = document.createElement('div');
  infoBar.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:16px',
    'margin-top:10px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:#b0b0b0',
    'align-items:center',
  ].join(';');
  container.appendChild(infoBar);

  // controls section — buttons + slider
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Info bar item banane ka helper (DOM-safe, no innerHTML) ---
  function createInfoItem(labelText, valueText) {
    const wrapper = document.createElement('span');
    wrapper.style.cssText = 'display:inline-flex;gap:4px;';
    const label = document.createElement('span');
    label.style.cssText = 'color:' + ACCENT + ';';
    label.textContent = labelText;
    const value = document.createElement('span');
    value.textContent = valueText;
    wrapper.appendChild(label);
    wrapper.appendChild(value);
    return wrapper;
  }

  function createInfoSeparator() {
    const sep = document.createElement('span');
    sep.style.cssText = 'color:#444;';
    sep.textContent = '|';
    return sep;
  }

  // --- Button banane ka helper ---
  function createBtn(label, onClick, highlight) {
    const btn = document.createElement('button');
    btn.textContent = label;
    btn.style.cssText = [
      'background:' + (highlight ? ACCENT : 'rgba(255,255,255,0.06)'),
      'color:' + (highlight ? '#000' : '#ccc'),
      'border:1px solid ' + (highlight ? ACCENT : 'rgba(255,255,255,0.12)'),
      'padding:6px 14px',
      'border-radius:6px',
      'cursor:pointer',
      'font-family:"JetBrains Mono",monospace',
      'font-size:12px',
      'font-weight:600',
      'transition:all 0.15s',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = highlight ? '#6bb3ff' : 'rgba(255,255,255,0.12)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = highlight ? ACCENT : 'rgba(255,255,255,0.06)';
    });
    btn.addEventListener('click', onClick);
    return btn;
  }

  // --- Buttons --- (onclick will be set after helper functions are defined)
  const btnForward = createBtn('\u25B6 Forward', () => {}, true);
  controlsDiv.appendChild(btnForward);

  const btnBackward = createBtn('\u25C0 Backward', () => {}, false);
  controlsDiv.appendChild(btnBackward);

  const btnUpdate = createBtn('\u21BB Update Weights', () => {}, false);
  controlsDiv.appendChild(btnUpdate);

  const btnAuto = createBtn('\u27F3 Auto Train', () => {}, false);
  controlsDiv.appendChild(btnAuto);

  const btnNextEx = createBtn('Next Example', () => {}, false);
  controlsDiv.appendChild(btnNextEx);

  const btnReset = createBtn('Reset Weights', () => {}, false);
  controlsDiv.appendChild(btnReset);

  // learning rate slider
  const lrWrapper = document.createElement('div');
  lrWrapper.style.cssText = 'display:flex;align-items:center;gap:6px;margin-left:8px;';
  const lrLabel = document.createElement('span');
  lrLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
  lrLabel.textContent = 'lr:';
  lrWrapper.appendChild(lrLabel);
  const lrSlider = document.createElement('input');
  lrSlider.type = 'range';
  lrSlider.min = '0.01';
  lrSlider.max = '2.0';
  lrSlider.step = '0.01';
  lrSlider.value = String(learningRate);
  lrSlider.style.cssText = 'width:80px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
  lrWrapper.appendChild(lrSlider);
  const lrValue = document.createElement('span');
  lrValue.style.cssText = 'color:#ccc;font-size:12px;font-family:"JetBrains Mono",monospace;min-width:32px;';
  lrValue.textContent = learningRate.toFixed(2);
  lrWrapper.appendChild(lrValue);
  controlsDiv.appendChild(lrWrapper);

  lrSlider.addEventListener('input', () => {
    learningRate = parseFloat(lrSlider.value);
    lrValue.textContent = learningRate.toFixed(2);
  });

  // --- Activation functions ---
  function tanhFn(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    const e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }

  function tanhDeriv(tanhVal) {
    // d/dx tanh(x) = 1 - tanh(x)^2
    return 1 - tanhVal * tanhVal;
  }

  function sigmoid(x) {
    if (x > 20) return 1;
    if (x < -20) return 0;
    return 1 / (1 + Math.exp(-x));
  }

  // --- Weight initialization --- Xavier/Glorot
  function initWeights() {
    const scale1 = Math.sqrt(2.0 / (LAYER_SIZES[0] + LAYER_SIZES[1]));
    const scale2 = Math.sqrt(2.0 / (LAYER_SIZES[1] + LAYER_SIZES[2]));

    // W1: 2x3 — input se hidden
    W1 = Array.from({ length: LAYER_SIZES[0] }, () =>
      Array.from({ length: LAYER_SIZES[1] }, () => (Math.random() * 2 - 1) * scale1)
    );
    b1 = new Array(LAYER_SIZES[1]).fill(0).map(() => (Math.random() * 2 - 1) * 0.1);

    // W2: 3x1 — hidden se output
    W2 = Array.from({ length: LAYER_SIZES[1] }, () =>
      Array.from({ length: LAYER_SIZES[2] }, () => (Math.random() * 2 - 1) * scale2)
    );
    b2 = new Array(LAYER_SIZES[2]).fill(0).map(() => (Math.random() * 2 - 1) * 0.1);
  }

  // --- Forward pass — compute activations layer by layer ---
  function computeForward() {
    const example = XOR_DATA[currentExample];
    const inp = example.input;

    // layer 0: input values as-is
    activations[0] = [...inp];

    // layer 1: hidden = tanh(W1^T * input + b1)
    const z1 = new Array(LAYER_SIZES[1]);
    const a1 = new Array(LAYER_SIZES[1]);
    for (let j = 0; j < LAYER_SIZES[1]; j++) {
      let sum = b1[j];
      for (let i = 0; i < LAYER_SIZES[0]; i++) {
        sum += inp[i] * W1[i][j];
      }
      z1[j] = sum;
      a1[j] = tanhFn(sum);
    }
    preActivations[0] = z1;
    activations[1] = a1;

    // layer 2: output = sigmoid(W2^T * hidden + b2)
    const z2 = new Array(LAYER_SIZES[2]);
    const a2 = new Array(LAYER_SIZES[2]);
    for (let j = 0; j < LAYER_SIZES[2]; j++) {
      let sum = b2[j];
      for (let i = 0; i < LAYER_SIZES[1]; i++) {
        sum += a1[i] * W2[i][j];
      }
      z2[j] = sum;
      a2[j] = sigmoid(sum);
    }
    preActivations[1] = z2;
    activations[2] = a2;

    output = a2[0];
    const target = example.target;
    // binary cross-entropy loss — standard classification loss
    loss = -(target * Math.log(output + 1e-8) + (1 - target) * Math.log(1 - output + 1e-8));
  }

  // --- Backward pass — gradients compute karo ---
  function computeBackward() {
    const example = XOR_DATA[currentExample];
    const target = example.target;
    const a0 = activations[0]; // input
    const a1 = activations[1]; // hidden activations
    const a2 = activations[2]; // output activations

    // output layer gradient: dL/da2 * da2/dz2
    // binary cross-entropy + sigmoid simplify: dL/dz2 = a2 - target
    // ye chain rule ka magic hai — bahut clean formula nikalta hai
    const dz2 = [a2[0] - target];
    gradients.dOutput = [...dz2];

    // W2 gradients: dL/dW2[i][j] = a1[i] * dz2[j]
    gradients.dW2 = Array.from({ length: LAYER_SIZES[1] }, (_, i) =>
      Array.from({ length: LAYER_SIZES[2] }, (_, j) => a1[i] * dz2[j])
    );
    gradients.db2 = [...dz2];

    // hidden layer gradient — backprop through W2 and tanh
    const dHidden = new Array(LAYER_SIZES[1]).fill(0);
    for (let i = 0; i < LAYER_SIZES[1]; i++) {
      for (let j = 0; j < LAYER_SIZES[2]; j++) {
        dHidden[i] += dz2[j] * W2[i][j];
      }
      // tanh derivative: 1 - tanh(z)^2
      dHidden[i] *= tanhDeriv(a1[i]);
    }
    gradients.dHidden = [...dHidden];

    // W1 gradients: dL/dW1[i][j] = a0[i] * dHidden[j]
    gradients.dW1 = Array.from({ length: LAYER_SIZES[0] }, (_, i) =>
      Array.from({ length: LAYER_SIZES[1] }, (_, j) => a0[i] * dHidden[j])
    );
    gradients.db1 = [...dHidden];
  }

  // --- Weight update — gradient descent step ---
  function applyGradients() {
    // W2 update: W2 -= lr * dW2
    for (let i = 0; i < LAYER_SIZES[1]; i++) {
      for (let j = 0; j < LAYER_SIZES[2]; j++) {
        W2[i][j] -= learningRate * gradients.dW2[i][j];
      }
    }
    for (let j = 0; j < LAYER_SIZES[2]; j++) {
      b2[j] -= learningRate * gradients.db2[j];
    }

    // W1 update: W1 -= lr * dW1
    for (let i = 0; i < LAYER_SIZES[0]; i++) {
      for (let j = 0; j < LAYER_SIZES[1]; j++) {
        W1[i][j] -= learningRate * gradients.dW1[i][j];
      }
    }
    for (let j = 0; j < LAYER_SIZES[1]; j++) {
      b1[j] -= learningRate * gradients.db1[j];
    }

    trainStep++;

    // average loss compute kar saare examples pe — overall progress dikhane ke liye
    let avgLoss = 0;
    for (let e = 0; e < XOR_DATA.length; e++) {
      const inp = XOR_DATA[e].input;
      const tgt = XOR_DATA[e].target;
      // quick forward pass — main activations disturb nahi karna
      const h = new Array(LAYER_SIZES[1]);
      for (let j = 0; j < LAYER_SIZES[1]; j++) {
        let s = b1[j];
        for (let i = 0; i < LAYER_SIZES[0]; i++) s += inp[i] * W1[i][j];
        h[j] = tanhFn(s);
      }
      let o = b2[0];
      for (let i = 0; i < LAYER_SIZES[1]; i++) o += h[i] * W2[i][0];
      o = sigmoid(o);
      avgLoss += -(tgt * Math.log(o + 1e-8) + (1 - tgt) * Math.log(1 - o + 1e-8));
    }
    avgLoss /= XOR_DATA.length;
    lossHistory.push(avgLoss);
    if (lossHistory.length > MAX_LOSS_HISTORY) lossHistory.shift();
  }

  // --- Auto train ---
  function stopAutoTrain() {
    autoTrain = false;
    if (autoTrainTimer) {
      clearTimeout(autoTrainTimer);
      autoTrainTimer = null;
    }
    if (animTimer) {
      clearTimeout(animTimer);
      animTimer = null;
    }
    btnAuto.style.background = 'rgba(255,255,255,0.06)';
    btnAuto.style.borderColor = 'rgba(255,255,255,0.12)';
    btnAuto.style.color = '#ccc';
    btnAuto.textContent = '\u27F3 Auto Train';
  }

  function autoStep() {
    if (!autoTrain || !isVisible) return;
    // cycle through all 4 XOR examples
    currentExample = trainStep % XOR_DATA.length;
    computeForward();
    computeBackward();
    applyGradients();
    computeForward(); // recompute with updated weights
    showGradients = false;
    phase = 'idle';
    animLayerIdx = -1;
    fullDraw();
    updateInfo();
    autoTrainTimer = setTimeout(autoStep, 60);
  }

  // --- Reset ---
  function resetWeights() {
    stopAutoTrain();
    initWeights();
    trainStep = 0;
    lossHistory = [];
    loss = null;
    output = null;
    activations = [null, null, null];
    preActivations = [null, null];
    gradients = { dW1: null, db1: null, dW2: null, db2: null, dHidden: null, dOutput: null };
    showGradients = false;
    phase = 'idle';
    animLayerIdx = -1;
  }

  // --- Node position computation ---
  function computeNodePositions() {
    nodePositions = [];
    const numLayers = LAYER_SIZES.length;
    // network takes left ~58% of canvas, right side for graphs/table
    const graphAreaWidth = canvasW * 0.50;
    const leftMargin = canvasW * 0.08;
    const layerSpacing = graphAreaWidth / (numLayers - 1);

    for (let l = 0; l < numLayers; l++) {
      const layerNodes = [];
      const n = LAYER_SIZES[l];
      const totalHeight = (n - 1) * 70;
      // vertically centered, nudged up a bit for labels below
      const startY = (canvasH * 0.52) - totalHeight / 2;

      for (let i = 0; i < n; i++) {
        layerNodes.push({
          x: leftMargin + l * layerSpacing,
          y: startY + i * 70,
        });
      }
      nodePositions.push(layerNodes);
    }
  }

  // --- Canvas resize ---
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvasW = rect.width;
    canvasH = rect.height;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    computeNodePositions();
  }

  // --- Info bar update (safe DOM, no innerHTML) ---
  function updateInfo() {
    // clear existing children
    while (infoBar.firstChild) {
      infoBar.removeChild(infoBar.firstChild);
    }

    const example = XOR_DATA[currentExample];

    infoBar.appendChild(createInfoItem('Input:', ' [' + example.input.join(', ') + ']'));
    infoBar.appendChild(createInfoSeparator());
    infoBar.appendChild(createInfoItem('Target:', ' ' + example.target));
    infoBar.appendChild(createInfoSeparator());

    if (output !== null) {
      infoBar.appendChild(createInfoItem('Output:', ' ' + output.toFixed(4)));
      infoBar.appendChild(createInfoSeparator());
    }
    if (loss !== null) {
      infoBar.appendChild(createInfoItem('Loss:', ' ' + loss.toFixed(4)));
      infoBar.appendChild(createInfoSeparator());
    }

    infoBar.appendChild(createInfoItem('Step:', ' ' + trainStep));
    infoBar.appendChild(createInfoSeparator());
    infoBar.appendChild(createInfoItem('Example:', ' ' + (currentExample + 1) + '/4'));
  }

  // --- Rounded rect helper ---
  function roundRect(context, x, y, w, h, r) {
    context.beginPath();
    context.moveTo(x + r, y);
    context.lineTo(x + w - r, y);
    context.quadraticCurveTo(x + w, y, x + w, y + r);
    context.lineTo(x + w, y + h - r);
    context.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    context.lineTo(x + r, y + h);
    context.quadraticCurveTo(x, y + h, x, y + h - r);
    context.lineTo(x, y + r);
    context.quadraticCurveTo(x, y, x + r, y);
    context.closePath();
  }

  // subscript digits helper
  function subscript(n) {
    const subs = '\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089';
    return String(n).split('').map(c => subs[parseInt(c)]).join('');
  }

  // === DRAWING FUNCTIONS ===

  // --- Draw connections (weights) between layers ---
  function drawConnections() {
    drawLayerConnections(0, 1, W1, gradients.dW1);
    drawLayerConnections(1, 2, W2, gradients.dW2);
  }

  function drawLayerConnections(fromLayer, toLayer, weights, gradWeights) {
    if (!weights) return;
    const fromNodes = nodePositions[fromLayer];
    const toNodes = nodePositions[toLayer];

    // max |weight| for thickness normalization
    let maxW = 0;
    for (let i = 0; i < fromNodes.length; i++) {
      for (let j = 0; j < toNodes.length; j++) {
        maxW = Math.max(maxW, Math.abs(weights[i][j]));
      }
    }
    if (maxW < 0.001) maxW = 1;

    for (let i = 0; i < fromNodes.length; i++) {
      for (let j = 0; j < toNodes.length; j++) {
        const w = weights[i][j];
        const absW = Math.abs(w);
        const thickness = 1 + (absW / maxW) * 4;
        const alpha = 0.25 + (absW / maxW) * 0.65;

        // highlight logic — forward: left to right, backward: right to left
        let isHighlighted = false;
        if (phase === 'forward' && animLayerIdx >= toLayer) {
          isHighlighted = true;
        } else if (phase === 'backward' && animLayerIdx >= 0 && animLayerIdx <= fromLayer) {
          isHighlighted = true;
        }

        const color = w >= 0 ? POS_COLOR : NEG_COLOR;
        const drawAlpha = isHighlighted ? Math.min(alpha + 0.3, 1.0) : alpha;

        ctx.beginPath();
        ctx.moveTo(fromNodes[i].x, fromNodes[i].y);
        ctx.lineTo(toNodes[j].x, toNodes[j].y);
        ctx.strokeStyle = color;
        ctx.globalAlpha = drawAlpha;
        ctx.lineWidth = thickness;
        ctx.stroke();
        ctx.globalAlpha = 1;

        // weight value label — midpoint pe, stagger so they don't overlap
        const mx = (fromNodes[i].x + toNodes[j].x) / 2;
        const my = (fromNodes[i].y + toNodes[j].y) / 2;
        const offsetX = (j - (toNodes.length - 1) / 2) * 4;
        const offsetY = (i - (fromNodes.length - 1) / 2) * 10;

        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        if (showGradients && gradWeights) {
          // gradient value dikhao — backward mode mein
          const gVal = gradWeights[i][j];
          const gColor = gVal >= 0 ? '#ff9966' : '#66ccff';
          ctx.fillStyle = gColor;
          ctx.fillText('\u2207' + gVal.toFixed(2), mx + offsetX, my + offsetY - 6);
          ctx.fillStyle = '#666';
          ctx.fillText(w.toFixed(2), mx + offsetX, my + offsetY + 6);
        } else {
          ctx.fillStyle = '#888';
          ctx.fillText(w.toFixed(2), mx + offsetX, my + offsetY);
        }
      }
    }
  }

  // --- Draw nodes ---
  function drawNodes() {
    for (let l = 0; l < LAYER_SIZES.length; l++) {
      for (let i = 0; i < LAYER_SIZES[l]; i++) {
        const pos = nodePositions[l][i];
        const act = activations[l] ? activations[l][i] : null;

        // node is "active" if animation has reached this layer
        let isActive = false;
        if (phase === 'forward' && animLayerIdx >= l) {
          isActive = true;
        } else if (phase === 'backward' && animLayerIdx >= 0 && animLayerIdx <= l) {
          isActive = true;
        } else if (phase === 'idle' && activations[l]) {
          isActive = true;
        }

        // brightness based on activation — darker=0, brighter=1
        let brightness = 0.15;
        if (isActive && act !== null) {
          let normAct;
          if (l === 1) {
            // hidden layer — tanh output [-1, 1] ko [0, 1] mein map kar
            normAct = (act + 1) / 2;
          } else {
            // input/output — already [0, 1] range
            normAct = Math.max(0, Math.min(1, act));
          }
          brightness = 0.15 + normAct * 0.85;
        }

        // glow effect for active nodes
        if (isActive) {
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, NODE_RADIUS + 6, 0, Math.PI * 2);
          const gradient = ctx.createRadialGradient(
            pos.x, pos.y, NODE_RADIUS,
            pos.x, pos.y, NODE_RADIUS + 10
          );
          gradient.addColorStop(0, 'rgba(74,158,255,' + (brightness * 0.3) + ')');
          gradient.addColorStop(1, 'rgba(74,158,255,0)');
          ctx.fillStyle = gradient;
          ctx.fill();
        }

        // node circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, NODE_RADIUS, 0, Math.PI * 2);

        // fill — brightness determines intensity of accent color
        const r = Math.round(74 * brightness);
        const g = Math.round(158 * brightness);
        const b = Math.round(255 * brightness);
        ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
        ctx.fill();

        // border
        ctx.strokeStyle = isActive ? ACCENT : 'rgba(74,158,255,0.3)';
        ctx.lineWidth = isActive ? 2 : 1;
        ctx.stroke();

        // activation value inside node
        ctx.font = 'bold 11px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (isActive && act !== null) {
          ctx.fillStyle = brightness > 0.5 ? '#000' : '#fff';
          ctx.fillText(act.toFixed(2), pos.x, pos.y);
        } else {
          ctx.fillStyle = '#555';
          ctx.fillText('\u2014', pos.x, pos.y);
        }

        // node label below — x₁, x₂ for input, h₁...h₃ for hidden, y-hat for output
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillStyle = '#666';
        ctx.textBaseline = 'top';
        let label;
        if (l === 0) label = 'x' + subscript(i + 1);
        else if (l === 1) label = 'h' + subscript(i + 1);
        else label = 'y\u0302'; // y-hat
        ctx.fillText(label, pos.x, pos.y + NODE_RADIUS + 4);

        // gradient values below node — backward mode mein
        if (showGradients && l > 0) {
          let gradVal = null;
          if (l === 1 && gradients.dHidden) gradVal = gradients.dHidden[i];
          if (l === 2 && gradients.dOutput) gradVal = gradients.dOutput[i];

          if (gradVal !== null) {
            const gColor = gradVal >= 0 ? '#ff9966' : '#66ccff';
            ctx.font = '9px "JetBrains Mono", monospace';
            ctx.fillStyle = gColor;
            ctx.fillText('\u2202L=' + gradVal.toFixed(3), pos.x, pos.y + NODE_RADIUS + 18);
          }
        }
      }
    }
  }

  // --- Layer labels above each column ---
  function drawLayerLabels() {
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';

    for (let l = 0; l < LAYER_SIZES.length; l++) {
      const x = nodePositions[l][0].x;
      const topY = nodePositions[l][0].y - NODE_RADIUS - 24;
      ctx.fillStyle = '#777';
      ctx.fillText(LAYER_LABELS[l], x, topY);
    }
  }

  // --- Loss graph (right side of canvas, upper) ---
  function drawLossGraph() {
    if (lossHistory.length < 2) return;

    const gx = canvasW * 0.63;
    const gy = 16;
    const gw = canvasW * 0.33;
    const gh = canvasH * 0.38;

    // background
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.strokeStyle = 'rgba(74,158,255,0.15)';
    ctx.lineWidth = 1;
    roundRect(ctx, gx, gy, gw, gh, 6);
    ctx.fill();
    ctx.stroke();

    // title
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#888';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Avg Loss', gx + 8, gy + 6);

    // loss value
    const currentLoss = lossHistory[lossHistory.length - 1];
    ctx.fillStyle = ACCENT;
    ctx.textAlign = 'right';
    ctx.fillText(currentLoss.toFixed(4), gx + gw - 8, gy + 6);

    // plot area
    const padX = 8;
    const padTop = 24;
    const padBottom = 6;
    const plotX = gx + padX;
    const plotY = gy + padTop;
    const plotW = gw - padX * 2;
    const plotH = gh - padTop - padBottom;

    // scale — min/max for y-axis
    let minL = Infinity, maxL = -Infinity;
    for (const l of lossHistory) {
      if (l < minL) minL = l;
      if (l > maxL) maxL = l;
    }
    if (maxL - minL < 0.01) {
      maxL = minL + 0.5;
    }

    // draw loss curve
    ctx.beginPath();
    for (let i = 0; i < lossHistory.length; i++) {
      const px = plotX + (i / (lossHistory.length - 1)) * plotW;
      const py = plotY + plotH - ((lossHistory[i] - minL) / (maxL - minL)) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // gradient fill under curve
    const grad = ctx.createLinearGradient(0, plotY, 0, plotY + plotH);
    grad.addColorStop(0, 'rgba(74,158,255,0.15)');
    grad.addColorStop(1, 'rgba(74,158,255,0)');
    ctx.lineTo(plotX + plotW, plotY + plotH);
    ctx.lineTo(plotX, plotY + plotH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // y-axis labels
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.fillStyle = '#555';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(maxL.toFixed(2), plotX - 2, plotY);
    ctx.textBaseline = 'bottom';
    ctx.fillText(minL.toFixed(2), plotX - 2, plotY + plotH);
  }

  // --- XOR truth table (right side, lower) ---
  function drawTruthTable() {
    const tx = canvasW * 0.63;
    const ty = canvasH * 0.46;
    const tw = canvasW * 0.33;
    const th = canvasH * 0.48;

    // background
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.strokeStyle = 'rgba(74,158,255,0.15)';
    ctx.lineWidth = 1;
    roundRect(ctx, tx, ty, tw, th, 6);
    ctx.fill();
    ctx.stroke();

    // title
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#888';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('XOR Truth Table', tx + 8, ty + 6);

    // column headers
    const colW = tw / 4;
    const headerY = ty + 24;
    ctx.fillStyle = '#777';
    ctx.textAlign = 'center';
    const headers = ['x\u2081', 'x\u2082', 'tgt', 'out'];
    for (let c = 0; c < 4; c++) {
      ctx.fillText(headers[c], tx + colW * c + colW / 2, headerY);
    }

    // separator line
    ctx.beginPath();
    ctx.moveTo(tx + 8, headerY + 16);
    ctx.lineTo(tx + tw - 8, headerY + 16);
    ctx.strokeStyle = 'rgba(74,158,255,0.1)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // rows — one for each XOR example
    for (let e = 0; e < XOR_DATA.length; e++) {
      const rowY = headerY + 22 + e * 28;
      const isActive = e === currentExample;

      // highlight current example row
      if (isActive) {
        ctx.fillStyle = 'rgba(74,158,255,0.1)';
        roundRect(ctx, tx + 4, rowY - 4, tw - 8, 22, 3);
        ctx.fill();
      }

      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      // input values
      ctx.fillStyle = isActive ? '#ddd' : '#666';
      ctx.fillText(String(XOR_DATA[e].input[0]), tx + colW * 0 + colW / 2, rowY);
      ctx.fillText(String(XOR_DATA[e].input[1]), tx + colW * 1 + colW / 2, rowY);

      // target
      ctx.fillStyle = isActive ? ACCENT : '#666';
      ctx.fillText(String(XOR_DATA[e].target), tx + colW * 2 + colW / 2, rowY);

      // prediction — quick forward for this example (doesn't disturb main state)
      const inp = XOR_DATA[e].input;
      const h = new Array(LAYER_SIZES[1]);
      for (let j = 0; j < LAYER_SIZES[1]; j++) {
        let s = b1[j];
        for (let ii = 0; ii < LAYER_SIZES[0]; ii++) s += inp[ii] * W1[ii][j];
        h[j] = tanhFn(s);
      }
      let o = b2[0];
      for (let ii = 0; ii < LAYER_SIZES[1]; ii++) o += h[ii] * W2[ii][0];
      o = sigmoid(o);

      // color by correctness — green if close, yellow if mid, red if far
      const err = Math.abs(o - XOR_DATA[e].target);
      const correctColor = err < 0.2 ? '#22c55e' : err < 0.4 ? '#f59e0b' : '#ef4444';
      ctx.fillStyle = isActive ? correctColor : '#555';
      ctx.fillText(o.toFixed(2), tx + colW * 3 + colW / 2, rowY);
    }
  }

  // --- Phase indicator (top left) ---
  function drawPhaseIndicator() {
    let label = '';
    let color = '#555';
    if (phase === 'forward') {
      label = 'FORWARD \u2192';
      color = '#22c55e';
    } else if (phase === 'backward') {
      label = '\u2190 BACKWARD';
      color = '#f59e0b';
    } else if (autoTrain) {
      label = 'AUTO TRAINING';
      color = '#ef4444';
    }

    if (label) {
      ctx.font = 'bold 11px "JetBrains Mono", monospace';
      ctx.fillStyle = color;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(label, 12, 12);
    }

    // step counter
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#555';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText('Step: ' + trainStep, canvasW * 0.56, 12);
  }

  // --- Main draw function (combines everything) ---
  function fullDraw() {
    ctx.clearRect(0, 0, canvasW, canvasH);
    if (nodePositions.length === 0) return;
    drawConnections();
    drawNodes();
    drawLayerLabels();
    drawLossGraph();
    drawTruthTable();
    drawPhaseIndicator();
  }

  // --- Forward animation — layers light up left to right ---
  function animateForwardStep() {
    animLayerIdx++;
    if (animLayerIdx >= LAYER_SIZES.length) {
      phase = 'idle';
      animLayerIdx = -1;
      fullDraw();
      updateInfo();
      return;
    }
    fullDraw();
    animTimer = setTimeout(animateForwardStep, ANIM_DELAY);
  }

  // --- Backward animation — layers light up right to left ---
  function animateBackwardStep() {
    animLayerIdx--;
    if (animLayerIdx < 0) {
      phase = 'idle';
      animLayerIdx = -1;
      fullDraw();
      updateInfo();
      return;
    }
    fullDraw();
    animTimer = setTimeout(animateBackwardStep, ANIM_DELAY);
  }

  // --- Idle render loop — redraws continuously when visible ---
  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }
    fullDraw();
    animationId = requestAnimationFrame(animate);
  }

  // --- Wire up button click handlers ---
  btnForward.onclick = () => {
    if (phase === 'forward') return;
    stopAutoTrain();
    phase = 'forward';
    showGradients = false;
    computeForward();
    animLayerIdx = -1;
    animateForwardStep();
  };

  btnBackward.onclick = () => {
    if (phase === 'backward') return;
    stopAutoTrain();
    // forward chahiye pehle — agar nahi hua toh kar lo silently
    if (!activations[2]) computeForward();
    phase = 'backward';
    computeBackward();
    showGradients = true;
    animLayerIdx = LAYER_SIZES.length;
    animateBackwardStep();
  };

  btnUpdate.onclick = () => {
    stopAutoTrain();
    if (!gradients.dW1) {
      // backward nahi hua — pehle forward+backward kar
      computeForward();
      computeBackward();
      showGradients = true;
    }
    applyGradients();
    computeForward(); // show updated output
    showGradients = false;
    phase = 'idle';
    animLayerIdx = -1;
    fullDraw();
    updateInfo();
  };

  btnAuto.onclick = () => {
    if (autoTrain) {
      stopAutoTrain();
    } else {
      autoTrain = true;
      btnAuto.style.background = '#ef4444';
      btnAuto.style.borderColor = '#ef4444';
      btnAuto.style.color = '#fff';
      btnAuto.textContent = '\u25A0 Stop';
      autoStep();
    }
  };

  btnNextEx.onclick = () => {
    stopAutoTrain();
    currentExample = (currentExample + 1) % XOR_DATA.length;
    computeForward();
    showGradients = false;
    phase = 'idle';
    animLayerIdx = -1;
    fullDraw();
    updateInfo();
  };

  btnReset.onclick = () => {
    resetWeights();
    fullDraw();
    updateInfo();
  };

  // --- IntersectionObserver — sirf visible hone pe animate kar ---
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      isVisible = entry.isIntersecting;
      if (isVisible && !animationId) {
        resizeCanvas();
        fullDraw();
        updateInfo();
        animate();
      }
    });
  }, { threshold: 0.1 });
  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) animate(); });

  // resize listener
  window.addEventListener('resize', () => {
    if (isVisible) {
      resizeCanvas();
      fullDraw();
    }
  });

  // --- Initial setup ---
  initWeights();
  resizeCanvas();
  fullDraw();
  updateInfo();
}
