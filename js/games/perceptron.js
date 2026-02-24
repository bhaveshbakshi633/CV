// ============================================================
// Perceptron & MLP Interactive Demo — XOR wala aha moment dikhana hai
// Single perceptron linear boundary vs MLP curved boundary
// Key insight: Perceptron can't solve XOR, MLP can — ye sikhana hai
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, perceptron chalao
export function initPerceptron() {
  const container = document.getElementById('perceptronContainer');
  if (!container) {
    console.warn('perceptronContainer nahi mila bhai, perceptron demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 350;
  const POINT_RADIUS = 6;
  const HEATMAP_RES = 40; // MLP heatmap resolution — zyada kiya toh slow ho jaayega
  const ACCENT = '#4a9eff';
  const POS_COLOR = '#4a9eff'; // blue = positive class
  const NEG_COLOR = '#ff4a6a'; // red = negative class
  const FONT = "'JetBrains Mono',monospace";

  // --- State variables ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let animationId = null;
  let isVisible = false;

  // data points — {x, y, label} jahan x,y normalized [-1,1] mein, label = +1/-1
  let dataPoints = [];

  // mode: 'perceptron' ya 'mlp'
  let mode = 'perceptron';

  // training state
  let isTraining = false;
  let epoch = 0;
  let accuracy = 0;
  let trainSpeed = 10; // har frame mein kitne training steps chalayein

  // perceptron weights — w1*x + w2*y + bias
  let pw1 = 0, pw2 = 0, pBias = 0;
  let pLR = 0.1; // perceptron learning rate

  // MLP weights — input(2) → hidden(6, tanh) → output(1, sigmoid)
  const HIDDEN_SIZE = 6;
  let mlpW1 = []; // [2][HIDDEN_SIZE]
  let mlpB1 = []; // [HIDDEN_SIZE]
  let mlpW2 = []; // [HIDDEN_SIZE]
  let mlpB2 = 0;  // scalar
  let mlpLR = 0.5; // MLP learning rate — zyada chahiye warna slow converge hoga

  // xor fail message dikhana hai ya nahi
  let showXorFail = false;

  // heatmap cache — har frame recalculate karna expensive hai
  let heatmapDirty = true;
  let heatmapImageData = null;

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — data points aur decision boundary yahan dikhega
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
  const ctx = canvas.getContext('2d');

  // stats bar — epoch, accuracy, weights dikhao
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'justify-content:center',
    'gap:16px',
    'margin-top:8px',
    'font-family:' + FONT,
    'font-size:12px',
    'color:#888',
    'min-height:20px',
  ].join(';');
  container.appendChild(statsDiv);

  // weights display — numerically update hota dikhega
  const weightsDiv = document.createElement('div');
  weightsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'justify-content:center',
    'gap:12px',
    'margin-top:4px',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#666',
    'min-height:18px',
  ].join(';');
  container.appendChild(weightsDiv);

  // XOR fail message container
  const xorMsgDiv = document.createElement('div');
  xorMsgDiv.style.cssText = [
    'text-align:center',
    'font-family:' + FONT,
    'font-size:13px',
    'color:#ff4a6a',
    'margin-top:6px',
    'min-height:20px',
    'transition:opacity 0.3s',
    'opacity:0',
  ].join(';');
  container.appendChild(xorMsgDiv);

  // controls bar — buttons, mode toggle, presets
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // presets bar — alag row mein data presets
  const presetsDiv = document.createElement('div');
  presetsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:6px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(presetsDiv);

  // learning rate slider row
  const lrDiv = document.createElement('div');
  lrDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:8px',
    'align-items:center',
    'justify-content:center',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#888',
  ].join(';');
  container.appendChild(lrDiv);

  // --- Button helper — consistent styling ---
  function makeButton(parentEl, text, onClick, highlight) {
    const btn = document.createElement('button');
    btn.textContent = text;
    const baseBg = highlight ? 'rgba(74,158,255,0.2)' : 'rgba(74,158,255,0.08)';
    const baseColor = highlight ? '#4a9eff' : '#b0b0b0';
    const baseBorder = highlight ? 'rgba(74,158,255,0.5)' : 'rgba(74,158,255,0.2)';
    btn.style.cssText = [
      'background:' + baseBg,
      'color:' + baseColor,
      'border:1px solid ' + baseBorder,
      'border-radius:6px',
      'padding:5px 12px',
      'font-size:11px',
      'font-family:' + FONT,
      'cursor:pointer',
      'transition:all 0.2s',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(74,158,255,0.25)';
      btn.style.color = '#fff';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = baseBg;
      btn.style.color = baseColor;
    });
    btn.addEventListener('click', onClick);
    parentEl.appendChild(btn);
    return btn;
  }

  // --- Safe DOM stat helper — no innerHTML, XSS safe ---
  function makeStatSpan(label, value, color) {
    const span = document.createElement('span');
    span.appendChild(document.createTextNode(label));
    const valSpan = document.createElement('span');
    valSpan.style.color = color;
    valSpan.textContent = value;
    span.appendChild(valSpan);
    return span;
  }

  function makeWeightSpan(label, value, color) {
    const span = document.createElement('span');
    span.appendChild(document.createTextNode(label + '='));
    const valSpan = document.createElement('span');
    valSpan.style.color = color;
    valSpan.textContent = value;
    span.appendChild(valSpan);
    return span;
  }

  // --- Perceptron functions ---

  // perceptron predict — w1*x + w2*y + bias, sign se class
  function perceptronPredict(x, y) {
    return pw1 * x + pw2 * y + pBias;
  }

  function perceptronClass(x, y) {
    return perceptronPredict(x, y) >= 0 ? 1 : -1;
  }

  // perceptron learning rule — agar galat classify kiya toh weights update kar
  function perceptronUpdate(x, y, label) {
    const pred = perceptronClass(x, y);
    if (pred !== label) {
      // w += lr * label * x (classic perceptron update)
      pw1 += pLR * label * x;
      pw2 += pLR * label * y;
      pBias += pLR * label;
      return true; // weight change hua
    }
    return false; // sahi tha, kuch nahi badla
  }

  // --- MLP functions ---

  // tanh activation
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
    return 1 / (1 + Math.exp(-x));
  }

  // MLP forward pass — input [x, y] → hidden (tanh) → output (sigmoid)
  function mlpForward(x, y) {
    const input = [x, y];
    // hidden layer
    const hidden = new Array(HIDDEN_SIZE);
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      let sum = mlpB1[j];
      for (let i = 0; i < 2; i++) {
        sum += input[i] * mlpW1[i][j];
      }
      hidden[j] = tanh(sum);
    }
    // output layer — single neuron, sigmoid
    let outSum = mlpB2;
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      outSum += hidden[j] * mlpW2[j];
    }
    const output = sigmoid(outSum);
    return { hidden, output };
  }

  // MLP predict class — output > 0.5 → +1, else -1
  function mlpClass(x, y) {
    const { output } = mlpForward(x, y);
    return output >= 0.5 ? 1 : -1;
  }

  // MLP backprop — single sample update (SGD)
  function mlpBackprop(x, y, label) {
    const input = [x, y];
    // target: label +1 → 1.0, label -1 → 0.0
    const target = label === 1 ? 1.0 : 0.0;

    // --- forward pass ---
    const hidden = new Array(HIDDEN_SIZE);
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      let sum = mlpB1[j];
      for (let i = 0; i < 2; i++) {
        sum += input[i] * mlpW1[i][j];
      }
      hidden[j] = tanh(sum);
    }
    let outSum = mlpB2;
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      outSum += hidden[j] * mlpW2[j];
    }
    const output = sigmoid(outSum);

    // --- backward pass ---
    // output layer error: dL/dout = output - target (cross-entropy + sigmoid = simplified)
    const dOut = output - target;

    // W2 gradients — pehle compute, fir apply (order matters for correct backprop)
    const savedW2 = mlpW2.slice(); // W2 copy rakho backprop ke liye
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      mlpW2[j] -= mlpLR * dOut * hidden[j];
    }
    mlpB2 -= mlpLR * dOut;

    // hidden layer error — backpropagate through saved W2
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      const dHidden = dOut * savedW2[j] * (1 - hidden[j] * hidden[j]); // tanh derivative: 1 - h^2
      for (let i = 0; i < 2; i++) {
        mlpW1[i][j] -= mlpLR * dHidden * input[i];
      }
      mlpB1[j] -= mlpLR * dHidden;
    }
  }

  // --- Weight initialization ---
  function initPerceptronWeights() {
    pw1 = (Math.random() - 0.5) * 0.5;
    pw2 = (Math.random() - 0.5) * 0.5;
    pBias = (Math.random() - 0.5) * 0.1;
  }

  function initMLPWeights() {
    // Xavier initialization — variance = 2/(fan_in + fan_out)
    const scale1 = Math.sqrt(2.0 / (2 + HIDDEN_SIZE));
    const scale2 = Math.sqrt(2.0 / (HIDDEN_SIZE + 1));
    mlpW1 = Array.from({ length: 2 }, () =>
      Array.from({ length: HIDDEN_SIZE }, () => (Math.random() * 2 - 1) * scale1)
    );
    mlpB1 = new Array(HIDDEN_SIZE).fill(0);
    mlpW2 = Array.from({ length: HIDDEN_SIZE }, () => (Math.random() * 2 - 1) * scale2);
    mlpB2 = 0;
  }

  function resetWeights() {
    if (mode === 'perceptron') {
      initPerceptronWeights();
    } else {
      initMLPWeights();
    }
    epoch = 0;
    accuracy = 0;
    heatmapDirty = true;
    showXorFail = false;
    xorMsgDiv.style.opacity = '0';
  }

  // --- Accuracy calculate kar ---
  function computeAccuracy() {
    if (dataPoints.length === 0) return 0;
    let correct = 0;
    for (const p of dataPoints) {
      const pred = mode === 'perceptron' ? perceptronClass(p.x, p.y) : mlpClass(p.x, p.y);
      if (pred === p.label) correct++;
    }
    return correct / dataPoints.length;
  }

  // --- XOR detection — check karo ki data XOR pattern jaisa hai ---
  function isXORLikeData() {
    if (dataPoints.length < 4) return false;
    let q = [0, 0, 0, 0]; // quadrant counts
    let ql = [0, 0, 0, 0]; // label sums per quadrant
    for (const p of dataPoints) {
      const qi = (p.x >= 0 ? 0 : 2) + (p.y >= 0 ? 0 : 1);
      q[qi]++;
      ql[qi] += p.label;
    }
    // sab quadrants mein points hone chahiye
    const hasAllQuadrants = q[0] > 0 && q[1] > 0 && q[2] > 0 && q[3] > 0;
    if (!hasAllQuadrants) return false;
    const s0 = Math.sign(ql[0]);
    const s1 = Math.sign(ql[1]);
    const s2 = Math.sign(ql[2]);
    const s3 = Math.sign(ql[3]);
    // diagonal quadrants same label, adjacent different — ye XOR hai
    return (s0 === s3 && s1 === s2 && s0 !== s1);
  }

  // --- Training step — ek epoch chalao ---
  function trainStep() {
    if (dataPoints.length === 0) return;

    if (mode === 'perceptron') {
      // perceptron: ek full pass over all data points
      // shuffle order for better convergence
      const indices = dataPoints.map((_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      for (const idx of indices) {
        const p = dataPoints[idx];
        perceptronUpdate(p.x, p.y, p.label);
      }
      epoch++;
      accuracy = computeAccuracy();
      heatmapDirty = true;

      // XOR fail detection — agar perceptron mode mein 50+ epochs aur accuracy < 80%
      if (epoch > 50 && accuracy < 0.8 && isXORLikeData()) {
        showXorFail = true;
        xorMsgDiv.textContent = "Can't solve! Perceptron fails on XOR \u2192 Try MLP mode";
        xorMsgDiv.style.opacity = '1';
      }
    } else {
      // MLP: ek epoch = saare data points pe ek baar backprop (shuffled SGD)
      const indices = dataPoints.map((_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      for (const idx of indices) {
        const p = dataPoints[idx];
        mlpBackprop(p.x, p.y, p.label);
      }
      epoch++;
      accuracy = computeAccuracy();
      heatmapDirty = true;

      // MLP ne XOR solve kar liya — celebration message
      if (showXorFail && accuracy > 0.95) {
        showXorFail = false;
        xorMsgDiv.textContent = 'MLP solved it! Non-linear boundary FTW';
        xorMsgDiv.style.color = '#4aff8f';
        xorMsgDiv.style.opacity = '1';
        setTimeout(() => {
          xorMsgDiv.style.opacity = '0';
          setTimeout(() => { xorMsgDiv.style.color = '#ff4a6a'; }, 300);
        }, 3000);
      }
    }
  }

  // --- Coordinate conversion: canvas pixel <-> normalized [-1,1] ---
  function canvasToNorm(px, py) {
    const pad = 30;
    const plotW = canvasW - 2 * pad;
    const plotH = canvasH - 2 * pad;
    const nx = ((px - pad) / plotW) * 2 - 1;
    const ny = 1 - ((py - pad) / plotH) * 2; // y flipped — upar positive
    return { x: Math.max(-1, Math.min(1, nx)), y: Math.max(-1, Math.min(1, ny)) };
  }

  function normToCanvas(nx, ny) {
    const pad = 30;
    const plotW = canvasW - 2 * pad;
    const plotH = canvasH - 2 * pad;
    const px = pad + ((nx + 1) / 2) * plotW;
    const py = pad + ((1 - ny) / 2) * plotH;
    return { px, py };
  }

  // --- Heatmap generate karo — decision boundary visualize karne ke liye ---
  function generateHeatmap() {
    if (!heatmapDirty) return;
    heatmapDirty = false;

    const pad = 30;
    const plotW = canvasW - 2 * pad;
    const plotH = canvasH - 2 * pad;

    if (plotW <= 0 || plotH <= 0) return;

    // offscreen canvas pe heatmap banao
    const offCanvas = document.createElement('canvas');
    offCanvas.width = HEATMAP_RES;
    offCanvas.height = HEATMAP_RES;
    const offCtx = offCanvas.getContext('2d');
    const imgData = offCtx.createImageData(HEATMAP_RES, HEATMAP_RES);
    const data = imgData.data;

    for (let gy = 0; gy < HEATMAP_RES; gy++) {
      for (let gx = 0; gx < HEATMAP_RES; gx++) {
        // grid cell ka center → normalized coordinate
        const nx = (gx / (HEATMAP_RES - 1)) * 2 - 1;
        const ny = 1 - (gy / (HEATMAP_RES - 1)) * 2;

        let confidence;
        if (mode === 'perceptron') {
          // perceptron: distance from boundary sigmoid se smooth 0-1 mein
          const raw = perceptronPredict(nx, ny);
          confidence = sigmoid(raw * 3);
        } else {
          // MLP: direct sigmoid output
          const { output } = mlpForward(nx, ny);
          confidence = output;
        }

        const idx = (gy * HEATMAP_RES + gx) * 4;
        // blue (positive) ↔ red (negative) gradient
        const r = Math.round(255 * (1 - confidence) * 0.9 + 30);
        const g = Math.round(40 + confidence * 20);
        const b = Math.round(255 * confidence * 0.9 + 30);
        const a = Math.round(50 + Math.abs(confidence - 0.5) * 2 * 80);
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = a;
      }
    }
    offCtx.putImageData(imgData, 0, 0);
    heatmapImageData = offCanvas;
  }

  // --- Main canvas rendering ---
  function draw() {
    dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    canvasW = w;
    canvasH = h;

    ctx.clearRect(0, 0, w, h);

    const pad = 30;
    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;

    // --- Background grid — subtle ---
    ctx.strokeStyle = 'rgba(74,158,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = pad + (i / 4) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, pad + plotH);
      ctx.stroke();
    }
    for (let i = 0; i <= 4; i++) {
      const y = pad + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(pad + plotW, y);
      ctx.stroke();
    }

    // axes — center pe cross
    ctx.strokeStyle = 'rgba(74,158,255,0.12)';
    ctx.lineWidth = 1;
    const cx = pad + plotW / 2;
    const cy = pad + plotH / 2;
    ctx.beginPath();
    ctx.moveTo(cx, pad);
    ctx.lineTo(cx, pad + plotH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad, cy);
    ctx.lineTo(pad + plotW, cy);
    ctx.stroke();

    // axis labels
    ctx.font = '9px ' + FONT;
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('-1', pad, pad + plotH + 4);
    ctx.fillText('0', cx, pad + plotH + 4);
    ctx.fillText('1', pad + plotW, pad + plotH + 4);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('1', pad - 6, pad);
    ctx.fillText('0', pad - 6, cy);
    ctx.fillText('-1', pad - 6, pad + plotH);

    // --- Heatmap / Decision boundary ---
    if (dataPoints.length > 0) {
      generateHeatmap();
      if (heatmapImageData) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(heatmapImageData, pad, pad, plotW, plotH);
      }

      // perceptron mode: explicit decision boundary line draw kar
      if (mode === 'perceptron' && (Math.abs(pw1) > 1e-8 || Math.abs(pw2) > 1e-8)) {
        // w1*x + w2*y + bias = 0
        ctx.strokeStyle = 'rgba(255,255,255,0.6)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();

        if (Math.abs(pw2) > 1e-8) {
          const x0 = -1, x1 = 1;
          const y0 = -(pw1 * x0 + pBias) / pw2;
          const y1 = -(pw1 * x1 + pBias) / pw2;
          const p0 = normToCanvas(x0, y0);
          const p1 = normToCanvas(x1, y1);
          ctx.moveTo(p0.px, p0.py);
          ctx.lineTo(p1.px, p1.py);
        } else {
          const xLine = -pBias / pw1;
          const p0 = normToCanvas(xLine, -1);
          const p1 = normToCanvas(xLine, 1);
          ctx.moveTo(p0.px, p0.py);
          ctx.lineTo(p1.px, p1.py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // MLP mode: marching squares se decision boundary contour draw kar
      if (mode === 'mlp') {
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        const res = 60;
        for (let gy = 0; gy < res; gy++) {
          for (let gx = 0; gx < res; gx++) {
            const nx0 = (gx / res) * 2 - 1;
            const ny0 = 1 - (gy / res) * 2;
            const nx1 = ((gx + 1) / res) * 2 - 1;
            const ny1 = 1 - ((gy + 1) / res) * 2;

            const v00 = mlpForward(nx0, ny0).output;
            const v10 = mlpForward(nx1, ny0).output;
            const v01 = mlpForward(nx0, ny1).output;
            const v11 = mlpForward(nx1, ny1).output;

            const threshold = 0.5;
            const b00 = v00 >= threshold ? 1 : 0;
            const b10 = v10 >= threshold ? 1 : 0;
            const b01 = v01 >= threshold ? 1 : 0;
            const b11 = v11 >= threshold ? 1 : 0;

            const cellCase = b00 + b10 + b01 + b11;
            if (cellCase === 0 || cellCase === 4) continue;

            // edge crossings dhundho — linear interpolation se
            const edges = [];
            if (b00 !== b10) {
              const t = (threshold - v00) / (v10 - v00);
              edges.push({ nx: nx0 + t * (nx1 - nx0), ny: ny0 });
            }
            if (b01 !== b11) {
              const t = (threshold - v01) / (v11 - v01);
              edges.push({ nx: nx0 + t * (nx1 - nx0), ny: ny1 });
            }
            if (b00 !== b01) {
              const t = (threshold - v00) / (v01 - v00);
              edges.push({ nx: nx0, ny: ny0 + t * (ny1 - ny0) });
            }
            if (b10 !== b11) {
              const t = (threshold - v10) / (v11 - v10);
              edges.push({ nx: nx1, ny: ny0 + t * (ny1 - ny0) });
            }

            if (edges.length >= 2) {
              const p0 = normToCanvas(edges[0].nx, edges[0].ny);
              const p1 = normToCanvas(edges[1].nx, edges[1].ny);
              ctx.beginPath();
              ctx.moveTo(p0.px, p0.py);
              ctx.lineTo(p1.px, p1.py);
              ctx.stroke();
            }
          }
        }
        ctx.setLineDash([]);
      }
    }

    // --- Data points draw kar ---
    for (const p of dataPoints) {
      const { px, py } = normToCanvas(p.x, p.y);
      const color = p.label === 1 ? POS_COLOR : NEG_COLOR;

      // hex color to RGB
      const r = parseInt(color.slice(1, 3), 16);
      const g = parseInt(color.slice(3, 5), 16);
      const b = parseInt(color.slice(5, 7), 16);

      // outer glow
      ctx.beginPath();
      ctx.arc(px, py, POINT_RADIUS + 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.15)';
      ctx.fill();

      // main circle
      ctx.beginPath();
      ctx.arc(px, py, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // border
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // label indicator — + ya - text
      ctx.font = 'bold 10px ' + FONT;
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(p.label === 1 ? '+' : '\u2212', px, py);
    }

    // --- Instructions text ---
    ctx.font = '10px ' + FONT;
    ctx.fillStyle = 'rgba(176,176,176,0.4)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    if (dataPoints.length === 0) {
      ctx.fillText('Click to add +1 (blue) \u2022 Shift+click for -1 (red)', pad + 4, pad + 4);
    }

    // mode indicator — top right pe
    ctx.textAlign = 'right';
    ctx.fillStyle = mode === 'perceptron' ? 'rgba(74,158,255,0.5)' : 'rgba(180,120,255,0.5)';
    ctx.font = '11px ' + FONT;
    ctx.fillText(mode === 'perceptron' ? 'PERCEPTRON' : 'MLP (' + HIDDEN_SIZE + ' hidden)', w - pad - 4, pad + 4);

    // transform reset
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  // --- Stats update — safe DOM methods, no innerHTML ---
  function updateStats() {
    while (statsDiv.firstChild) statsDiv.removeChild(statsDiv.firstChild);

    statsDiv.appendChild(makeStatSpan('Epoch: ', String(epoch), ACCENT));
    const accText = dataPoints.length > 0 ? (accuracy * 100).toFixed(1) + '%' : '--';
    const accColor = accuracy >= 0.95 ? '#4aff8f' : ACCENT;
    statsDiv.appendChild(makeStatSpan('Accuracy: ', accText, accColor));
    statsDiv.appendChild(makeStatSpan('Points: ', String(dataPoints.length), ACCENT));
    const modeColor = mode === 'perceptron' ? ACCENT : '#b478ff';
    const modeText = mode === 'perceptron' ? 'Perceptron' : 'MLP';
    statsDiv.appendChild(makeStatSpan('Mode: ', modeText, modeColor));
  }

  function updateWeightsDisplay() {
    while (weightsDiv.firstChild) weightsDiv.removeChild(weightsDiv.firstChild);

    if (mode === 'perceptron') {
      const items = [
        { label: 'w\u2081', val: pw1 },
        { label: 'w\u2082', val: pw2 },
        { label: 'bias', val: pBias },
      ];
      for (const item of items) {
        const valColor = item.val >= 0 ? 'rgba(74,158,255,0.7)' : 'rgba(255,74,106,0.7)';
        weightsDiv.appendChild(makeWeightSpan(item.label, item.val.toFixed(3), valColor));
      }
    } else {
      // MLP: summary stats — too many weights to show individually
      let totalW = 0, maxW = 0;
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < HIDDEN_SIZE; j++) {
          totalW += Math.abs(mlpW1[i][j]);
          maxW = Math.max(maxW, Math.abs(mlpW1[i][j]));
        }
      }
      for (let j = 0; j < HIDDEN_SIZE; j++) {
        totalW += Math.abs(mlpW2[j]);
        maxW = Math.max(maxW, Math.abs(mlpW2[j]));
      }
      const nWeights = 2 * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE + 1;
      const purpleColor = 'rgba(180,120,255,0.7)';
      weightsDiv.appendChild(makeWeightSpan('params', String(nWeights), purpleColor));
      weightsDiv.appendChild(makeWeightSpan('|w|avg', (totalW / nWeights).toFixed(3), purpleColor));
      weightsDiv.appendChild(makeWeightSpan('max|w|', maxW.toFixed(3), purpleColor));
    }
  }

  // --- Canvas click handling — data points add kar ---
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const norm = canvasToNorm(px, py);

    // boundary check
    if (norm.x < -1 || norm.x > 1 || norm.y < -1 || norm.y > 1) return;

    const label = e.shiftKey ? -1 : 1;
    dataPoints.push({ x: norm.x, y: norm.y, label });
    heatmapDirty = true;
    accuracy = computeAccuracy();
    updateStats();
    updateWeightsDisplay();
  });

  // --- Canvas resize ---
  function resizeCanvas() {
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = w + 'px';
    canvasW = w;
    canvasH = CANVAS_HEIGHT;
    heatmapDirty = true;
  }

  // --- Data presets ---
  function setPreset(name) {
    dataPoints = [];
    showXorFail = false;
    xorMsgDiv.style.opacity = '0';

    if (name === 'linear') {
      // linearly separable — two bands, tilted line ke dono taraf
      for (let i = 0; i < 15; i++) {
        const x = (Math.random() - 0.5) * 1.6;
        const y = x * 0.6 + 0.25 + (Math.random() - 0.5) * 0.3;
        dataPoints.push({ x: Math.max(-1, Math.min(1, x)), y: Math.max(-1, Math.min(1, y)), label: 1 });
      }
      for (let i = 0; i < 15; i++) {
        const x = (Math.random() - 0.5) * 1.6;
        const y = x * 0.6 - 0.25 + (Math.random() - 0.5) * 0.3;
        dataPoints.push({ x: Math.max(-1, Math.min(1, x)), y: Math.max(-1, Math.min(1, y)), label: -1 });
      }
    } else if (name === 'xor') {
      // XOR — 4 clusters, diagonal corners same class
      const spread = 0.15;
      const dist = 0.55;
      const corners = [
        { cx: -dist, cy:  dist, label:  1 },
        { cx:  dist, cy: -dist, label:  1 },
        { cx:  dist, cy:  dist, label: -1 },
        { cx: -dist, cy: -dist, label: -1 },
      ];
      for (const c of corners) {
        for (let i = 0; i < 6; i++) {
          const x = c.cx + (Math.random() - 0.5) * spread;
          const y = c.cy + (Math.random() - 0.5) * spread;
          dataPoints.push({ x, y, label: c.label });
        }
      }
    } else if (name === 'circle') {
      // circle pattern — inner ring positive, outer ring negative
      for (let i = 0; i < 20; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = 0.15 + Math.random() * 0.15;
        dataPoints.push({ x: Math.cos(angle) * r, y: Math.sin(angle) * r, label: 1 });
      }
      for (let i = 0; i < 25; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = 0.55 + Math.random() * 0.25;
        dataPoints.push({
          x: Math.max(-1, Math.min(1, Math.cos(angle) * r)),
          y: Math.max(-1, Math.min(1, Math.sin(angle) * r)),
          label: -1,
        });
      }
    } else if (name === 'clusters') {
      // 4 clusters — top row blue, bottom row red (linearly separable)
      const positions = [
        { cx: -0.5, cy:  0.5, label:  1 },
        { cx:  0.5, cy:  0.5, label:  1 },
        { cx: -0.5, cy: -0.5, label: -1 },
        { cx:  0.5, cy: -0.5, label: -1 },
      ];
      for (const p of positions) {
        for (let i = 0; i < 8; i++) {
          const x = p.cx + (Math.random() - 0.5) * 0.35;
          const y = p.cy + (Math.random() - 0.5) * 0.35;
          dataPoints.push({ x: Math.max(-1, Math.min(1, x)), y: Math.max(-1, Math.min(1, y)), label: p.label });
        }
      }
    }

    resetWeights();
    accuracy = computeAccuracy();
    updateStats();
    updateWeightsDisplay();
  }

  // --- Controls banao ---

  // Train / Pause toggle
  const trainBtn = makeButton(controlsDiv, 'Train', () => {
    isTraining = !isTraining;
    trainBtn.textContent = isTraining ? 'Pause' : 'Train';
    if (isTraining) {
      trainBtn.style.borderColor = 'rgba(74,255,143,0.5)';
      trainBtn.style.color = '#4aff8f';
    } else {
      trainBtn.style.borderColor = 'rgba(74,158,255,0.2)';
      trainBtn.style.color = '#b0b0b0';
    }
  });

  // Step — ek epoch manual chala
  makeButton(controlsDiv, 'Step', () => {
    if (dataPoints.length === 0) return;
    trainStep();
    updateStats();
    updateWeightsDisplay();
  });

  // Mode toggle — Perceptron ↔ MLP
  const modeBtn = makeButton(controlsDiv, 'Mode: Perceptron', () => {
    mode = mode === 'perceptron' ? 'mlp' : 'perceptron';
    modeBtn.textContent = 'Mode: ' + (mode === 'perceptron' ? 'Perceptron' : 'MLP');
    if (mode === 'mlp') {
      modeBtn.style.borderColor = 'rgba(180,120,255,0.5)';
      modeBtn.style.color = '#b478ff';
    } else {
      modeBtn.style.borderColor = 'rgba(74,158,255,0.2)';
      modeBtn.style.color = '#b0b0b0';
    }
    resetWeights();
    updateLRSlider();
    updateStats();
    updateWeightsDisplay();
  }, false);

  // Reset weights
  makeButton(controlsDiv, 'Reset Weights', () => {
    resetWeights();
    updateStats();
    updateWeightsDisplay();
  });

  // Clear data
  makeButton(controlsDiv, 'Clear Data', () => {
    dataPoints = [];
    resetWeights();
    isTraining = false;
    trainBtn.textContent = 'Train';
    trainBtn.style.borderColor = 'rgba(74,158,255,0.2)';
    trainBtn.style.color = '#b0b0b0';
    showXorFail = false;
    xorMsgDiv.style.opacity = '0';
    updateStats();
    updateWeightsDisplay();
  });

  // --- Preset buttons ---
  const presetLabel = document.createElement('span');
  presetLabel.style.cssText = 'color:#666;font-size:11px;font-family:' + FONT + ';';
  presetLabel.textContent = 'Presets:';
  presetsDiv.appendChild(presetLabel);

  makeButton(presetsDiv, 'Linear', () => setPreset('linear'));
  makeButton(presetsDiv, 'XOR', () => setPreset('xor'), true);
  makeButton(presetsDiv, 'Circle', () => setPreset('circle'));
  makeButton(presetsDiv, 'Clusters', () => setPreset('clusters'));

  // --- Learning rate slider ---
  const lrTextLabel = document.createElement('span');
  lrTextLabel.textContent = 'Learning Rate:';
  lrDiv.appendChild(lrTextLabel);

  const lrSlider = document.createElement('input');
  lrSlider.type = 'range';
  lrSlider.min = '-3';
  lrSlider.max = '1';
  lrSlider.step = '0.1';
  lrSlider.value = '-1'; // 10^-1 = 0.1
  lrSlider.style.cssText = 'width:120px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
  lrDiv.appendChild(lrSlider);

  const lrValSpan = document.createElement('span');
  lrValSpan.style.color = ACCENT;
  lrValSpan.textContent = '0.100';
  lrDiv.appendChild(lrValSpan);

  function updateLRSlider() {
    const currentLR = mode === 'perceptron' ? pLR : mlpLR;
    const logVal = Math.log10(currentLR);
    lrSlider.value = logVal.toFixed(1);
    lrValSpan.textContent = currentLR.toFixed(3);
  }

  lrSlider.addEventListener('input', () => {
    const lr = Math.pow(10, parseFloat(lrSlider.value));
    if (mode === 'perceptron') {
      pLR = lr;
    } else {
      mlpLR = lr;
    }
    lrValSpan.textContent = lr.toFixed(3);
  });

  // speed label + slider
  const speedLabel = document.createElement('span');
  speedLabel.textContent = 'Speed:';
  speedLabel.style.marginLeft = '16px';
  lrDiv.appendChild(speedLabel);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '50';
  speedSlider.step = '1';
  speedSlider.value = '10';
  speedSlider.style.cssText = 'width:80px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
  lrDiv.appendChild(speedSlider);

  const speedValSpan = document.createElement('span');
  speedValSpan.style.color = ACCENT;
  speedValSpan.textContent = '10';
  lrDiv.appendChild(speedValSpan);

  speedSlider.addEventListener('input', () => {
    trainSpeed = parseInt(speedSlider.value);
    speedValSpan.textContent = String(trainSpeed);
  });

  // --- Main animation loop ---
  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    // training steps chala agar training on hai
    if (isTraining && dataPoints.length > 0) {
      for (let i = 0; i < trainSpeed; i++) {
        trainStep();
      }
      updateStats();
      updateWeightsDisplay();
    }

    draw();
    animationId = requestAnimationFrame(animate);
  }

  function startAnimation() {
    if (animationId === null) {
      animationId = requestAnimationFrame(animate);
    }
  }

  function stopAnimation() {
    if (animationId !== null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // --- IntersectionObserver — sirf visible hone pe animate, CPU bach jaayega ---
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        const wasVisible = isVisible;
        isVisible = entry.isIntersecting;
        if (isVisible && !wasVisible) {
          resizeCanvas();
          startAnimation();
        } else if (!isVisible && wasVisible) {
          stopAnimation();
        }
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) animate(); });

  // --- Initialization — sab setup karke shuru kar ---
  initPerceptronWeights();
  initMLPWeights();
  resizeCanvas();
  updateStats();
  updateWeightsDisplay();
  updateLRSlider();

  // resize pe canvas update — responsive rehna chahiye
  window.addEventListener('resize', () => {
    resizeCanvas();
    heatmapDirty = true;
  });

  // animation start kar
  startAnimation();
}
