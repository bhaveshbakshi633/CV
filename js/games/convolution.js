// ============================================================
// CNN Convolution Kernel Visualizer
// Dekho kaise CNN images ko process karta hai — kernel sliding, edge detection, blur sab
// 3-panel layout: source → kernel → output, animated sliding window ke saath
// ============================================================

// yahi main entry point hai — container dhundho, canvas banao, convolution dikhao
export function initConvolution() {
  const container = document.getElementById('convolutionContainer');
  if (!container) {
    console.warn('convolutionContainer nahi mila bhai, convolution visualizer skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const NEG_COLOR = '#ef4444';
  const ZERO_COLOR = '#555';
  const FONT = "'JetBrains Mono',monospace";
  // source image ki resolution — individual pixels dikhne chahiye
  const IMG_W = 80;
  const IMG_H = 60;

  // --- Preset kernels — CNN ke classic filters ---
  const KERNEL_PRESETS = {
    'Identity': { size: 3, values: [0,0,0, 0,1,0, 0,0,0], divisor: 1 },
    'Sobel X': { size: 3, values: [-1,0,1, -2,0,2, -1,0,1], divisor: 1 },
    'Sobel Y': { size: 3, values: [-1,-2,-1, 0,0,0, 1,2,1], divisor: 1 },
    'Laplacian': { size: 3, values: [0,-1,0, -1,4,-1, 0,-1,0], divisor: 1 },
    'Gaussian Blur': { size: 3, values: [1,2,1, 2,4,2, 1,2,1], divisor: 16 },
    'Sharpen': { size: 3, values: [0,-1,0, -1,5,-1, 0,-1,0], divisor: 1 },
    'Emboss': { size: 3, values: [-2,-1,0, -1,1,1, 0,1,2], divisor: 1 },
    'Box Blur': { size: 3, values: [1,1,1, 1,1,1, 1,1,1], divisor: 9 },
  };

  // --- State variables ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let animationId = null;
  let isVisible = false;

  // source image data — flat array [IMG_W * IMG_H], grayscale 0-255
  let sourcePixels = new Uint8Array(IMG_W * IMG_H);
  // output image data — convolution result
  let outputPixels = new Uint8Array(IMG_W * IMG_H);

  // kernel state
  let kernelSize = 3; // 3 ya 5
  let kernelValues = [0,0,0, 0,1,0, 0,0,0]; // flat array, kernelSize x kernelSize
  let kernelDivisor = 1;
  let autoNormalize = false;
  let currentPreset = 'Sobel X';
  let currentSource = 'Shapes';

  // animation state — sliding window
  let animating = false;
  let animSpeed = 4; // kitne pixels skip kare per frame
  let animX = 0, animY = 0; // current kernel position
  let animDone = false;
  // partial output during animation — sirf animate wale pixels filled
  let animOutputPixels = null;

  // multi-kernel pipeline
  let pipeline = []; // [{values, size, divisor, preset}]
  let pipelineResults = []; // intermediate Uint8Arrays

  // paint mode state
  let isPainting = false;
  let paintRadius = 3;

  // animate button reference — controls section mein assign hoga
  let animBtnEl = null;

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — 3-panel layout yahan draw hoga
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.2)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // info bar — kernel info, pixel count, etc
  const infoBar = document.createElement('div');
  infoBar.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:16px',
    'margin-top:8px',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#888',
    'align-items:center',
  ].join(';');
  container.appendChild(infoBar);

  // kernel editor container — editable grid + normalize checkbox
  const kernelEditorDiv = document.createElement('div');
  kernelEditorDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:10px',
    'align-items:flex-start',
    'justify-content:center',
  ].join(';');
  container.appendChild(kernelEditorDiv);

  // controls row 1 — source, preset, kernel size, animate
  const controlsRow1 = document.createElement('div');
  controlsRow1.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsRow1);

  // controls row 2 — pipeline, speed, reset
  const controlsRow2 = document.createElement('div');
  controlsRow2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:6px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsRow2);

  // --- Helper: button banao consistent style mein ---
  function makeButton(parentEl, text, onClick, highlight) {
    const btn = document.createElement('button');
    btn.textContent = text;
    const baseBg = highlight ? 'rgba(74,158,255,0.2)' : 'rgba(74,158,255,0.08)';
    const baseColor = highlight ? ACCENT : '#b0b0b0';
    const baseBorder = highlight ? 'rgba(74,158,255,0.5)' : 'rgba(74,158,255,0.25)';
    btn.style.cssText = [
      'background:' + baseBg,
      'color:' + baseColor,
      'border:1px solid ' + baseBorder,
      'border-radius:6px',
      'padding:5px 12px',
      'font-size:11px',
      'font-family:' + FONT,
      'cursor:pointer',
      'transition:all 0.15s',
      'white-space:nowrap',
      'user-select:none',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(74,158,255,0.25)';
      btn.style.color = '#fff';
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn._active) {
        btn.style.background = baseBg;
        btn.style.color = baseColor;
      }
    });
    btn.addEventListener('click', onClick);
    parentEl.appendChild(btn);
    return btn;
  }

  // --- Helper: dropdown (select) banao ---
  function makeSelect(parentEl, label, options, onChange) {
    if (label) {
      const lbl = document.createElement('span');
      lbl.textContent = label;
      lbl.style.cssText = 'color:#888;font-size:11px;font-family:' + FONT + ';margin-right:4px;';
      parentEl.appendChild(lbl);
    }
    const sel = document.createElement('select');
    sel.style.cssText = [
      'background:rgba(10,10,20,0.8)',
      'color:#d0d0d0',
      'border:1px solid rgba(74,158,255,0.25)',
      'border-radius:6px',
      'padding:4px 8px',
      'font-size:11px',
      'font-family:' + FONT,
      'cursor:pointer',
      'outline:none',
    ].join(';');
    options.forEach(opt => {
      const o = document.createElement('option');
      o.value = opt;
      o.textContent = opt;
      sel.appendChild(o);
    });
    sel.addEventListener('change', () => onChange(sel.value));
    parentEl.appendChild(sel);
    return sel;
  }

  // --- Helper: separator ---
  function makeSep(parentEl) {
    const sep = document.createElement('span');
    sep.style.cssText = 'width:1px;height:18px;background:rgba(255,255,255,0.1);margin:0 4px;';
    parentEl.appendChild(sep);
  }

  // ============================================================
  // SOURCE IMAGE GENERATORS — koi external file nahi chahiye
  // ============================================================

  // shapes — black bg pe white geometric shapes
  function generateShapes() {
    const px = sourcePixels;
    px.fill(0);
    const cx = IMG_W / 2, cy = IMG_H / 2;

    // circle — center mein
    const r = Math.min(IMG_W, IMG_H) * 0.2;
    for (let y = 0; y < IMG_H; y++) {
      for (let x = 0; x < IMG_W; x++) {
        const dx = x - cx, dy = y - cy;
        if (dx * dx + dy * dy <= r * r) {
          px[y * IMG_W + x] = 255;
        }
      }
    }

    // square — left side
    const sq = Math.floor(r * 0.9);
    const sqX = Math.floor(IMG_W * 0.18);
    const sqY = Math.floor(IMG_H * 0.2);
    for (let y = sqY; y < sqY + sq * 2 && y < IMG_H; y++) {
      for (let x = sqX; x < sqX + sq * 2 && x < IMG_W; x++) {
        px[y * IMG_W + x] = 255;
      }
    }

    // triangle — right side
    const triCx = Math.floor(IMG_W * 0.78);
    const triCy = Math.floor(IMG_H * 0.5);
    const triH = Math.floor(r * 1.5);
    for (let row = 0; row < triH; row++) {
      const halfW = Math.floor((row / triH) * triH * 0.6);
      const yy = triCy - triH / 2 + row;
      if (yy < 0 || yy >= IMG_H) continue;
      for (let dx = -halfW; dx <= halfW; dx++) {
        const xx = triCx + dx;
        if (xx >= 0 && xx < IMG_W) {
          px[Math.floor(yy) * IMG_W + xx] = 255;
        }
      }
    }

    // cross — bottom area
    const crossCx = Math.floor(IMG_W * 0.5);
    const crossCy = Math.floor(IMG_H * 0.82);
    const arm = Math.floor(r * 0.6);
    const thick = 3;
    for (let dy = -arm; dy <= arm; dy++) {
      for (let dx = -thick; dx <= thick; dx++) {
        const yy = crossCy + dy, xx = crossCx + dx;
        if (yy >= 0 && yy < IMG_H && xx >= 0 && xx < IMG_W) px[yy * IMG_W + xx] = 255;
        const yy2 = crossCy + dx, xx2 = crossCx + dy;
        if (yy2 >= 0 && yy2 < IMG_H && xx2 >= 0 && xx2 < IMG_W) px[yy2 * IMG_W + xx2] = 255;
      }
    }
  }

  // gradient — smooth gradient with sharp transitions
  function generateGradient() {
    const px = sourcePixels;
    for (let y = 0; y < IMG_H; y++) {
      for (let x = 0; x < IMG_W; x++) {
        // horizontal gradient with step jumps
        let v = (x / IMG_W) * 255;
        // sharp transitions add karo — CNN edge detection ke liye interesting
        if (x > IMG_W * 0.3 && x < IMG_W * 0.35) v = 255;
        if (x > IMG_W * 0.6 && x < IMG_W * 0.65) v = 0;
        // vertical sine modulation — thoda variation
        v += Math.sin(y / IMG_H * Math.PI * 4) * 30;
        px[y * IMG_W + x] = Math.max(0, Math.min(255, Math.round(v)));
      }
    }
  }

  // checkerboard — alternating black white squares
  function generateCheckerboard() {
    const px = sourcePixels;
    const sq = 8; // har square 8 pixels ka
    for (let y = 0; y < IMG_H; y++) {
      for (let x = 0; x < IMG_W; x++) {
        const cx = Math.floor(x / sq);
        const cy = Math.floor(y / sq);
        px[y * IMG_W + x] = (cx + cy) % 2 === 0 ? 220 : 30;
      }
    }
  }

  // noise — random grayscale
  function generateNoise() {
    const px = sourcePixels;
    for (let i = 0; i < px.length; i++) {
      px[i] = Math.floor(Math.random() * 256);
    }
  }

  // paint — blank canvas, user draw karega
  function generatePaint() {
    sourcePixels.fill(0);
  }

  // source generate karo based on selection
  function generateSource(type) {
    currentSource = type;
    if (type === 'Shapes') generateShapes();
    else if (type === 'Gradient') generateGradient();
    else if (type === 'Checkerboard') generateCheckerboard();
    else if (type === 'Noise') generateNoise();
    else if (type === 'Paint') generatePaint();
  }

  // ============================================================
  // CONVOLUTION ALGORITHM — ye actual CNN convolution hai
  // ============================================================

  // single convolution apply karo — source se output mein
  function applyConvolution(src, kVals, kSize, kDiv) {
    const out = new Uint8Array(IMG_W * IMG_H);
    const half = Math.floor(kSize / 2);
    const divisor = kDiv || 1;

    for (let y = 0; y < IMG_H; y++) {
      for (let x = 0; x < IMG_W; x++) {
        let sum = 0;
        for (let ky = 0; ky < kSize; ky++) {
          for (let kx = 0; kx < kSize; kx++) {
            // source pixel position — mirror padding for borders
            let sy = y + ky - half;
            let sx = x + kx - half;
            // mirror padding — border pe reflect kar do
            if (sy < 0) sy = -sy;
            if (sy >= IMG_H) sy = 2 * IMG_H - sy - 2;
            if (sx < 0) sx = -sx;
            if (sx >= IMG_W) sx = 2 * IMG_W - sx - 2;
            // clamp just in case
            sy = Math.max(0, Math.min(IMG_H - 1, sy));
            sx = Math.max(0, Math.min(IMG_W - 1, sx));

            sum += src[sy * IMG_W + sx] * kVals[ky * kSize + kx];
          }
        }
        // divide aur clamp
        out[y * IMG_W + x] = Math.max(0, Math.min(255, Math.round(sum / divisor)));
      }
    }
    return out;
  }

  // full convolution run karo (single kernel ya pipeline)
  function runFullConvolution() {
    if (pipeline.length === 0) {
      // single kernel
      outputPixels = applyConvolution(sourcePixels, kernelValues, kernelSize, getEffectiveDivisor());
      pipelineResults = [];
    } else {
      // pipeline mode — sequentially apply
      pipelineResults = [];
      let current = sourcePixels;
      for (let i = 0; i < pipeline.length; i++) {
        const k = pipeline[i];
        const result = applyConvolution(current, k.values, k.size, k.divisor);
        pipelineResults.push(result);
        current = result;
      }
      // last pipeline output = final output
      outputPixels = current;
    }
  }

  // effective divisor — auto-normalize ya manual
  function getEffectiveDivisor() {
    if (autoNormalize) {
      let sum = 0;
      for (let i = 0; i < kernelValues.length; i++) sum += kernelValues[i];
      return sum !== 0 ? Math.abs(sum) : 1;
    }
    return kernelDivisor;
  }

  // ============================================================
  // KERNEL EDITOR — editable grid of values
  // ============================================================

  let kernelInputs = []; // DOM input references

  function buildKernelEditor() {
    // saaf karo pehle
    while (kernelEditorDiv.firstChild) kernelEditorDiv.removeChild(kernelEditorDiv.firstChild);
    kernelInputs = [];

    // left side — kernel grid
    const gridWrapper = document.createElement('div');
    gridWrapper.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:4px;';

    const gridLabel = document.createElement('div');
    gridLabel.textContent = 'Kernel ' + kernelSize + '\u00d7' + kernelSize;
    gridLabel.style.cssText = 'color:#888;font-size:11px;font-family:' + FONT + ';margin-bottom:2px;';
    gridWrapper.appendChild(gridLabel);

    const grid = document.createElement('div');
    grid.style.cssText = [
      'display:grid',
      'grid-template-columns:repeat(' + kernelSize + ',1fr)',
      'gap:2px',
    ].join(';');

    for (let i = 0; i < kernelSize * kernelSize; i++) {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.value = kernelValues[i] || 0;
      inp.step = 'any';
      const val = kernelValues[i] || 0;
      const bgColor = val > 0 ? 'rgba(74,158,255,0.15)' : val < 0 ? 'rgba(239,68,68,0.15)' : 'rgba(80,80,80,0.15)';
      inp.style.cssText = [
        'width:42px',
        'height:30px',
        'text-align:center',
        'font-size:11px',
        'font-family:' + FONT,
        'background:' + bgColor,
        'color:#d0d0d0',
        'border:1px solid rgba(74,158,255,0.2)',
        'border-radius:4px',
        'outline:none',
        'padding:0',
        '-moz-appearance:textfield',
      ].join(';');
      // index capture karo closure mein
      const idx = i;
      inp.addEventListener('change', () => {
        kernelValues[idx] = parseFloat(inp.value) || 0;
        updateKernelInputColors();
        onKernelChange();
      });
      inp.addEventListener('focus', () => {
        inp.style.borderColor = ACCENT;
      });
      inp.addEventListener('blur', () => {
        inp.style.borderColor = 'rgba(74,158,255,0.2)';
      });
      grid.appendChild(inp);
      kernelInputs.push(inp);
    }
    gridWrapper.appendChild(grid);

    // divisor display
    const divRow = document.createElement('div');
    divRow.style.cssText = 'display:flex;align-items:center;gap:6px;margin-top:4px;';
    const divLabel = document.createElement('span');
    divLabel.textContent = '\u00f7';
    divLabel.style.cssText = 'color:#888;font-size:13px;font-family:' + FONT + ';';
    divRow.appendChild(divLabel);

    const divInput = document.createElement('input');
    divInput.type = 'number';
    divInput.value = kernelDivisor;
    divInput.step = 'any';
    divInput.style.cssText = [
      'width:48px',
      'height:24px',
      'text-align:center',
      'font-size:11px',
      'font-family:' + FONT,
      'background:rgba(40,40,60,0.8)',
      'color:#d0d0d0',
      'border:1px solid rgba(74,158,255,0.2)',
      'border-radius:4px',
      'outline:none',
      'padding:0',
    ].join(';');
    divInput.addEventListener('change', () => {
      kernelDivisor = parseFloat(divInput.value) || 1;
      onKernelChange();
    });
    divRow.appendChild(divInput);
    gridWrapper.appendChild(divRow);

    // auto-normalize checkbox
    const normRow = document.createElement('label');
    normRow.style.cssText = 'display:flex;align-items:center;gap:4px;margin-top:2px;cursor:pointer;';
    const normCb = document.createElement('input');
    normCb.type = 'checkbox';
    normCb.checked = autoNormalize;
    normCb.style.cssText = 'cursor:pointer;accent-color:' + ACCENT + ';';
    normCb.addEventListener('change', () => {
      autoNormalize = normCb.checked;
      onKernelChange();
    });
    normRow.appendChild(normCb);
    const normLabel = document.createElement('span');
    normLabel.textContent = 'Auto-normalize';
    normLabel.style.cssText = 'color:#888;font-size:10px;font-family:' + FONT + ';';
    normRow.appendChild(normLabel);
    gridWrapper.appendChild(normRow);

    kernelEditorDiv.appendChild(gridWrapper);
  }

  // kernel input colors update karo — positive blue, negative red, zero gray
  function updateKernelInputColors() {
    kernelInputs.forEach((inp, i) => {
      const val = kernelValues[i] || 0;
      const bgColor = val > 0 ? 'rgba(74,158,255,0.15)' : val < 0 ? 'rgba(239,68,68,0.15)' : 'rgba(80,80,80,0.15)';
      inp.style.background = bgColor;
    });
  }

  // kernel values inputs mein sync karo
  function syncKernelToInputs() {
    kernelInputs.forEach((inp, i) => {
      inp.value = kernelValues[i] || 0;
    });
    updateKernelInputColors();
  }

  // jab kernel change ho — re-apply convolution
  function onKernelChange() {
    stopAnimation();
    runFullConvolution();
    updateInfoBar();
    requestDraw();
  }

  // preset load karo
  function loadPreset(name) {
    const preset = KERNEL_PRESETS[name];
    if (!preset) return;
    currentPreset = name;
    kernelSize = preset.size;
    kernelValues = preset.values.slice();
    kernelDivisor = preset.divisor;
    // kernel size badli toh editor rebuild karo
    buildKernelEditor();
    syncKernelToInputs();
    onKernelChange();
  }

  // kernel size change karo — values reset ya pad/trim
  function changeKernelSize(newSize) {
    if (newSize === kernelSize) return;
    const oldSize = kernelSize;
    const oldVals = kernelValues.slice();
    kernelSize = newSize;
    kernelValues = new Array(newSize * newSize).fill(0);
    // center mein identity rakh do
    const center = Math.floor(newSize / 2);
    kernelValues[center * newSize + center] = 1;
    kernelDivisor = 1;
    buildKernelEditor();
    syncKernelToInputs();
    onKernelChange();
  }

  // ============================================================
  // ANIMATION — sliding window
  // ============================================================

  function startAnimation() {
    if (animating) return;
    animating = true;
    animX = 0;
    animY = 0;
    animDone = false;
    animOutputPixels = new Uint8Array(IMG_W * IMG_H);
    animOutputPixels.fill(0);
    animBtnEl.textContent = 'Stop';
    animBtnEl._active = true;
    animBtnEl.style.background = 'rgba(74,158,255,0.35)';
    animBtnEl.style.borderColor = ACCENT;
    animBtnEl.style.color = '#fff';
    if (!animationId) loop();
  }

  function stopAnimation() {
    if (!animating) return;
    animating = false;
    animDone = false;
    animOutputPixels = null;
    animBtnEl.textContent = 'Animate';
    animBtnEl._active = false;
    animBtnEl.style.background = 'rgba(74,158,255,0.08)';
    animBtnEl.style.borderColor = 'rgba(74,158,255,0.25)';
    animBtnEl.style.color = '#b0b0b0';
  }

  function toggleAnimation() {
    if (animating) stopAnimation();
    else startAnimation();
    requestDraw();
  }

  // animation ka ek step — kernel aage badhao
  function animStep() {
    if (!animating || animDone) return;
    const half = Math.floor(kernelSize / 2);
    const kVals = pipeline.length > 0 ? pipeline[0].values : kernelValues;
    const kSize = pipeline.length > 0 ? pipeline[0].size : kernelSize;
    const kDiv = pipeline.length > 0 ? pipeline[0].divisor : getEffectiveDivisor();
    const src = sourcePixels;
    const halfK = Math.floor(kSize / 2);

    // speed ke hisaab se kitne pixels process kare
    for (let s = 0; s < animSpeed; s++) {
      if (animX >= IMG_W) {
        animX = 0;
        animY++;
      }
      if (animY >= IMG_H) {
        // done — ab full convolution dikha do
        animDone = true;
        // agar pipeline hai toh remaining kernels apply karo
        if (pipeline.length > 1) {
          let current = animOutputPixels;
          for (let i = 1; i < pipeline.length; i++) {
            current = applyConvolution(current, pipeline[i].values, pipeline[i].size, pipeline[i].divisor);
          }
          outputPixels = current;
        } else {
          outputPixels = animOutputPixels;
        }
        stopAnimation();
        return;
      }

      // is pixel ke liye convolution compute karo
      let sum = 0;
      for (let ky = 0; ky < kSize; ky++) {
        for (let kx = 0; kx < kSize; kx++) {
          let sy = animY + ky - halfK;
          let sx = animX + kx - halfK;
          if (sy < 0) sy = -sy;
          if (sy >= IMG_H) sy = 2 * IMG_H - sy - 2;
          if (sx < 0) sx = -sx;
          if (sx >= IMG_W) sx = 2 * IMG_W - sx - 2;
          sy = Math.max(0, Math.min(IMG_H - 1, sy));
          sx = Math.max(0, Math.min(IMG_W - 1, sx));
          sum += src[sy * IMG_W + sx] * kVals[ky * kSize + kx];
        }
      }
      animOutputPixels[animY * IMG_W + animX] = Math.max(0, Math.min(255, Math.round(sum / kDiv)));
      animX++;
    }
  }

  // ============================================================
  // MULTI-KERNEL PIPELINE
  // ============================================================

  function addKernelToPipeline() {
    if (pipeline.length >= 3) return; // max 3 kernels
    // current kernel pipeline mein add karo
    pipeline.push({
      values: kernelValues.slice(),
      size: kernelSize,
      divisor: getEffectiveDivisor(),
      preset: currentPreset,
    });
    updatePipelineDisplay();
    runFullConvolution();
    requestDraw();
  }

  function removeLastKernel() {
    if (pipeline.length === 0) return;
    pipeline.pop();
    pipelineResults.pop();
    updatePipelineDisplay();
    runFullConvolution();
    requestDraw();
  }

  let pipelineInfoEl = null;

  function updatePipelineDisplay() {
    if (pipelineInfoEl) {
      if (pipeline.length === 0) {
        pipelineInfoEl.textContent = '';
      } else {
        const names = pipeline.map((k, i) => 'K' + (i + 1) + '(' + k.preset + ')');
        pipelineInfoEl.textContent = 'Pipeline: ' + names.join(' \u2192 ');
      }
    }
  }

  // ============================================================
  // CANVAS DRAWING — 3-panel layout
  // ============================================================

  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvasH = CANVAS_HEIGHT;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // pixel array ko canvas pe draw karo — scaled up
  function drawPixelImage(pixels, x, y, w, h) {
    // har pixel ek rectangle — pixelated look chahiye
    const pxW = w / IMG_W;
    const pxH = h / IMG_H;
    for (let py = 0; py < IMG_H; py++) {
      for (let px = 0; px < IMG_W; px++) {
        const v = pixels[py * IMG_W + px];
        ctx.fillStyle = 'rgb(' + v + ',' + v + ',' + v + ')';
        ctx.fillRect(x + px * pxW, y + py * pxH, pxW + 0.5, pxH + 0.5);
      }
    }
  }

  // pixel grid overlay — faint lines dikhao
  function drawPixelGrid(x, y, w, h) {
    const pxW = w / IMG_W;
    const pxH = h / IMG_H;
    // sirf agar pixels kaafi bade hain tabhi grid dikhao
    if (pxW < 3) return;
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let i = 0; i <= IMG_W; i++) {
      const xx = x + i * pxW;
      ctx.moveTo(xx, y);
      ctx.lineTo(xx, y + h);
    }
    for (let j = 0; j <= IMG_H; j++) {
      const yy = y + j * pxH;
      ctx.moveTo(x, yy);
      ctx.lineTo(x + w, yy);
    }
    ctx.stroke();
  }

  // kernel matrix canvas pe draw karo — center panel mein
  function drawKernelMatrix(cx, cy, cellSize) {
    const kVals = pipeline.length > 0 && animating ? pipeline[0].values : kernelValues;
    const kSize = pipeline.length > 0 && animating ? pipeline[0].size : kernelSize;
    const totalW = kSize * cellSize;
    const startX = cx - totalW / 2;
    const startY = cy - totalW / 2;

    for (let ky = 0; ky < kSize; ky++) {
      for (let kx = 0; kx < kSize; kx++) {
        const val = kVals[ky * kSize + kx];
        const xx = startX + kx * cellSize;
        const yy = startY + ky * cellSize;

        // background — color based on value
        if (val > 0) {
          ctx.fillStyle = 'rgba(74,158,255,' + Math.min(0.5, Math.abs(val) * 0.1) + ')';
        } else if (val < 0) {
          ctx.fillStyle = 'rgba(239,68,68,' + Math.min(0.5, Math.abs(val) * 0.1) + ')';
        } else {
          ctx.fillStyle = 'rgba(80,80,80,0.2)';
        }
        ctx.fillRect(xx, yy, cellSize, cellSize);

        // border
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.strokeRect(xx, yy, cellSize, cellSize);

        // value text
        ctx.fillStyle = val > 0 ? ACCENT : val < 0 ? NEG_COLOR : ZERO_COLOR;
        ctx.font = '10px ' + FONT;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const displayVal = Number.isInteger(val) ? String(val) : val.toFixed(1);
        ctx.fillText(displayVal, xx + cellSize / 2, yy + cellSize / 2);
      }
    }

    // "Kernel" label
    ctx.fillStyle = '#888';
    ctx.font = '10px ' + FONT;
    ctx.textAlign = 'center';
    ctx.fillText('Kernel', cx, startY - 10);
  }

  // animated sliding window highlight draw karo
  function drawSlidingWindow(imgX, imgY, imgW, imgH) {
    if (!animating || animDone) return;
    const pxW = imgW / IMG_W;
    const pxH = imgH / IMG_H;
    const kSize = pipeline.length > 0 ? pipeline[0].size : kernelSize;
    const half = Math.floor(kSize / 2);

    // kernel window highlight on source
    const wx = imgX + (animX - half) * pxW;
    const wy = imgY + (animY - half) * pxH;
    const ww = kSize * pxW;
    const wh = kSize * pxH;

    // yellow-ish highlight border
    ctx.strokeStyle = '#ffcc00';
    ctx.lineWidth = 2;
    ctx.strokeRect(wx, wy, ww, wh);

    // semi-transparent overlay inside the window
    ctx.fillStyle = 'rgba(255,204,0,0.08)';
    ctx.fillRect(wx, wy, ww, wh);

    // current pixel marker — bright dot
    const cpx = imgX + animX * pxW + pxW / 2;
    const cpy = imgY + animY * pxH + pxH / 2;
    ctx.beginPath();
    ctx.arc(cpx, cpy, Math.max(2, pxW * 0.4), 0, Math.PI * 2);
    ctx.fillStyle = '#ffcc00';
    ctx.fill();
  }

  // connection arrows — source se kernel, kernel se output
  function drawArrows(srcEndX, kernelCx, outStartX, midY) {
    ctx.strokeStyle = 'rgba(74,158,255,0.3)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);

    // source → kernel
    const gap = 8;
    ctx.beginPath();
    ctx.moveTo(srcEndX + gap, midY);
    ctx.lineTo(kernelCx - 30, midY);
    ctx.stroke();
    // arrowhead
    drawArrowHead(kernelCx - 30, midY, 6, 0);

    // kernel → output
    ctx.beginPath();
    ctx.moveTo(kernelCx + 30, midY);
    ctx.lineTo(outStartX - gap, midY);
    ctx.stroke();
    drawArrowHead(outStartX - gap, midY, 6, 0);

    ctx.setLineDash([]);
  }

  function drawArrowHead(x, y, size, angle) {
    ctx.fillStyle = 'rgba(74,158,255,0.5)';
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - size, y - size * 0.5);
    ctx.lineTo(x - size, y + size * 0.5);
    ctx.closePath();
    ctx.fill();
  }

  // progress bar for animation
  function drawAnimProgress(x, y, w) {
    if (!animating && !animDone) return;
    const total = IMG_W * IMG_H;
    const done = animDone ? total : (animY * IMG_W + animX);
    const frac = done / total;

    ctx.fillStyle = 'rgba(40,40,60,0.5)';
    ctx.fillRect(x, y, w, 4);
    ctx.fillStyle = ACCENT;
    ctx.fillRect(x, y, w * frac, 4);
  }

  // main draw function
  function draw() {
    // clear
    ctx.clearRect(0, 0, canvasW, canvasH);

    // layout calculations
    const padding = 12;
    const imgAreaW = canvasW * 0.35;
    const kernelAreaW = canvasW * 0.15;
    // remaining = canvasW * 0.15 for arrows/gaps

    // image aspect ratio maintain karo
    const imgDisplayW = imgAreaW - padding * 2;
    const imgDisplayH = imgDisplayW * (IMG_H / IMG_W);
    const imgTop = (canvasH - imgDisplayH) / 2;

    // source image — left panel
    const srcX = padding;
    const srcY = Math.max(padding + 15, imgTop);
    const srcW = imgDisplayW;
    const srcH = Math.min(imgDisplayH, canvasH - srcY - padding - 10);

    // label
    ctx.fillStyle = '#888';
    ctx.font = '11px ' + FONT;
    ctx.textAlign = 'center';
    ctx.fillText('Source (' + currentSource + ')', srcX + srcW / 2, srcY - 5);

    drawPixelImage(sourcePixels, srcX, srcY, srcW, srcH);
    drawPixelGrid(srcX, srcY, srcW, srcH);

    // animated sliding window on source
    if (animating && !animDone) {
      drawSlidingWindow(srcX, srcY, srcW, srcH);
    }

    // output image — right panel
    const outX = canvasW - padding - imgDisplayW;
    const outY = srcY;
    const outW = srcW;
    const outH = srcH;

    // label
    ctx.fillStyle = '#888';
    ctx.font = '11px ' + FONT;
    ctx.textAlign = 'center';
    const outLabel = pipeline.length > 0 ? 'Output (Pipeline)' : 'Output';
    ctx.fillText(outLabel, outX + outW / 2, outY - 5);

    // animation ke time partial output dikhao
    const displayOutput = animating && animOutputPixels ? animOutputPixels : outputPixels;
    drawPixelImage(displayOutput, outX, outY, outW, outH);
    drawPixelGrid(outX, outY, outW, outH);

    // kernel matrix — center
    const kernelCx = canvasW / 2;
    const kernelCy = canvasH / 2;
    const kSize = pipeline.length > 0 && animating ? pipeline[0].size : kernelSize;
    const cellSize = Math.min(32, (kernelAreaW - 20) / kSize);
    drawKernelMatrix(kernelCx, kernelCy, cellSize);

    // arrows — source → kernel → output
    drawArrows(srcX + srcW, kernelCx, outX, kernelCy);

    // animation progress bar
    if (animating) {
      drawAnimProgress(srcX, srcY + srcH + 4, srcW);
    }

    // pipeline intermediate results dikhao — agar hain toh
    if (pipeline.length > 1 && pipelineResults.length > 0 && !animating) {
      // small thumbnails neeche dikhao
      const thumbH = 30;
      const thumbW = thumbH * (IMG_W / IMG_H);
      const startTX = canvasW / 2 - ((pipeline.length - 1) * (thumbW + 8)) / 2;
      const thumbY = canvasH - thumbH - 8;

      ctx.fillStyle = '#666';
      ctx.font = '9px ' + FONT;
      ctx.textAlign = 'center';
      ctx.fillText('Intermediate results:', canvasW / 2, thumbY - 6);

      for (let i = 0; i < pipelineResults.length - 1; i++) {
        const tx = startTX + i * (thumbW + 8);
        drawPixelImage(pipelineResults[i], tx, thumbY, thumbW, thumbH);
        ctx.strokeStyle = 'rgba(74,158,255,0.3)';
        ctx.lineWidth = 1;
        ctx.strokeRect(tx, thumbY, thumbW, thumbH);
        ctx.fillStyle = '#666';
        ctx.font = '8px ' + FONT;
        ctx.textAlign = 'center';
        ctx.fillText('K' + (i + 1), tx + thumbW / 2, thumbY + thumbH + 9);
      }
    }

    // convolution value computation display during animation
    if (animating && !animDone) {
      drawConvComputation(kernelCx, kernelCy + kSize * cellSize / 2 + 20);
    }
  }

  // animation ke time — current computation values dikhao
  function drawConvComputation(cx, y) {
    const kSize = pipeline.length > 0 ? pipeline[0].size : kernelSize;
    const kVals = pipeline.length > 0 ? pipeline[0].values : kernelValues;
    const kDiv = pipeline.length > 0 ? pipeline[0].divisor : getEffectiveDivisor();
    const half = Math.floor(kSize / 2);

    // current pixel ki computation
    let sum = 0;
    let parts = [];
    for (let ky = 0; ky < kSize; ky++) {
      for (let kx = 0; kx < kSize; kx++) {
        let sy = animY + ky - half;
        let sx = animX + kx - half;
        if (sy < 0) sy = -sy;
        if (sy >= IMG_H) sy = 2 * IMG_H - sy - 2;
        if (sx < 0) sx = -sx;
        if (sx >= IMG_W) sx = 2 * IMG_W - sx - 2;
        sy = Math.max(0, Math.min(IMG_H - 1, sy));
        sx = Math.max(0, Math.min(IMG_W - 1, sx));
        const pv = sourcePixels[sy * IMG_W + sx];
        const kv = kVals[ky * kSize + kx];
        sum += pv * kv;
        if (kv !== 0) parts.push(pv + '\u00d7' + kv);
      }
    }
    const result = Math.max(0, Math.min(255, Math.round(sum / kDiv)));

    ctx.fillStyle = '#888';
    ctx.font = '9px ' + FONT;
    ctx.textAlign = 'center';

    // sum display — truncate agar bahut lamba ho
    let sumText = 'sum=' + sum;
    if (kDiv !== 1) sumText += '/' + kDiv;
    sumText += ' \u2192 ' + result;
    ctx.fillText(sumText, cx, y);

    // position display
    ctx.fillStyle = '#666';
    ctx.fillText('pixel(' + animX + ',' + animY + ')', cx, y + 13);
  }

  // info bar update
  function updateInfoBar() {
    while (infoBar.firstChild) infoBar.removeChild(infoBar.firstChild);

    const items = [
      ['Kernel: ', currentPreset, ACCENT],
      ['Size: ', kernelSize + '\u00d7' + kernelSize, '#d0d0d0'],
      ['Divisor: ', String(getEffectiveDivisor()), '#d0d0d0'],
      ['Image: ', IMG_W + '\u00d7' + IMG_H, '#d0d0d0'],
    ];
    if (pipeline.length > 0) {
      items.push(['Pipeline: ', pipeline.length + ' kernels', ACCENT]);
    }

    items.forEach(([label, value, color]) => {
      const span = document.createElement('span');
      span.appendChild(document.createTextNode(label));
      const valSpan = document.createElement('span');
      valSpan.style.color = color;
      valSpan.textContent = value;
      span.appendChild(valSpan);
      infoBar.appendChild(span);
    });
  }

  // requestDraw — ek baar draw schedule karo
  let drawScheduled = false;
  function requestDraw() {
    if (drawScheduled) return;
    drawScheduled = true;
    requestAnimationFrame(() => {
      drawScheduled = false;
      draw();
    });
  }

  // ============================================================
  // PAINT MODE — canvas pe click karke draw karo
  // ============================================================

  function handleCanvasMouseDown(e) {
    if (currentSource !== 'Paint') return;
    isPainting = true;
    paintOnCanvas(e);
  }

  function handleCanvasMouseMove(e) {
    if (!isPainting) return;
    paintOnCanvas(e);
  }

  function handleCanvasMouseUp() {
    if (!isPainting) return;
    isPainting = false;
    // paint ke baad convolution re-apply
    runFullConvolution();
    requestDraw();
  }

  function paintOnCanvas(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // check if mouse is in source image area
    const padding = 12;
    const imgAreaW = canvasW * 0.35;
    const imgDisplayW = imgAreaW - padding * 2;
    const imgDisplayH = imgDisplayW * (IMG_H / IMG_W);
    const imgTop = (canvasH - imgDisplayH) / 2;
    const srcX = padding;
    const srcY = Math.max(padding + 15, imgTop);
    const srcW = imgDisplayW;
    const srcH = Math.min(imgDisplayH, canvasH - srcY - padding - 10);

    if (mx < srcX || mx > srcX + srcW || my < srcY || my > srcY + srcH) return;

    // mouse position ko pixel coordinates mein convert karo
    const px = Math.floor((mx - srcX) / srcW * IMG_W);
    const py = Math.floor((my - srcY) / srcH * IMG_H);

    // paint radius ke andar sab pixels white kar do
    for (let dy = -paintRadius; dy <= paintRadius; dy++) {
      for (let dx = -paintRadius; dx <= paintRadius; dx++) {
        if (dx * dx + dy * dy <= paintRadius * paintRadius) {
          const xx = px + dx;
          const yy = py + dy;
          if (xx >= 0 && xx < IMG_W && yy >= 0 && yy < IMG_H) {
            sourcePixels[yy * IMG_W + xx] = 255;
          }
        }
      }
    }
    requestDraw();
  }

  canvas.addEventListener('mousedown', handleCanvasMouseDown);
  canvas.addEventListener('mousemove', handleCanvasMouseMove);
  canvas.addEventListener('mouseup', handleCanvasMouseUp);
  canvas.addEventListener('mouseleave', handleCanvasMouseUp);

  // touch support for paint
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (currentSource !== 'Paint') return;
    isPainting = true;
    const touch = e.touches[0];
    paintOnCanvas(touch);
  }, { passive: false });
  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isPainting) return;
    const touch = e.touches[0];
    paintOnCanvas(touch);
  }, { passive: false });
  canvas.addEventListener('touchend', () => {
    handleCanvasMouseUp();
  });

  // ============================================================
  // CONTROLS BUILD KARO
  // ============================================================

  // Row 1: Source dropdown, Kernel preset dropdown, Kernel size toggle, Animate, Apply
  const sourceOptions = ['Shapes', 'Gradient', 'Checkerboard', 'Noise', 'Paint'];
  const srcSelect = makeSelect(controlsRow1, 'Source:', sourceOptions, (val) => {
    generateSource(val);
    if (val === 'Paint') {
      canvas.style.cursor = 'crosshair';
    } else {
      canvas.style.cursor = 'default';
    }
    stopAnimation();
    runFullConvolution();
    requestDraw();
  });
  srcSelect.value = 'Shapes';

  makeSep(controlsRow1);

  // preset dropdown
  const presetNames = Object.keys(KERNEL_PRESETS);
  const presetSelect = makeSelect(controlsRow1, 'Kernel:', presetNames, (val) => {
    loadPreset(val);
    presetSelect.value = val;
  });
  presetSelect.value = 'Sobel X';

  makeSep(controlsRow1);

  // kernel size toggle — 3x3 / 5x5
  const sizeLabel = document.createElement('span');
  sizeLabel.textContent = 'Size:';
  sizeLabel.style.cssText = 'color:#888;font-size:11px;font-family:' + FONT + ';';
  controlsRow1.appendChild(sizeLabel);

  const size3Btn = makeButton(controlsRow1, '3\u00d73', () => {
    changeKernelSize(3);
    updateSizeButtons();
  });
  const size5Btn = makeButton(controlsRow1, '5\u00d75', () => {
    changeKernelSize(5);
    updateSizeButtons();
  });

  function updateSizeButtons() {
    [size3Btn, size5Btn].forEach(btn => {
      const isActive = (btn === size3Btn && kernelSize === 3) || (btn === size5Btn && kernelSize === 5);
      btn._active = isActive;
      btn.style.background = isActive ? 'rgba(74,158,255,0.35)' : 'rgba(74,158,255,0.08)';
      btn.style.borderColor = isActive ? ACCENT : 'rgba(74,158,255,0.25)';
      btn.style.color = isActive ? '#fff' : '#b0b0b0';
    });
  }
  updateSizeButtons();

  makeSep(controlsRow1);

  // animate toggle
  animBtnEl = makeButton(controlsRow1, 'Animate', toggleAnimation);

  // apply button — instant full convolution
  makeButton(controlsRow1, 'Apply', () => {
    stopAnimation();
    runFullConvolution();
    requestDraw();
  }, true);

  // Row 2: Speed slider, Add/Remove kernel, Pipeline info, Reset
  const speedLabel = document.createElement('span');
  speedLabel.textContent = 'Speed:';
  speedLabel.style.cssText = 'color:#888;font-size:11px;font-family:' + FONT + ';';
  controlsRow2.appendChild(speedLabel);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '40';
  speedSlider.value = String(animSpeed);
  speedSlider.style.cssText = [
    'width:80px',
    'accent-color:' + ACCENT,
    'cursor:pointer',
    'height:16px',
  ].join(';');
  speedSlider.addEventListener('input', () => {
    animSpeed = parseInt(speedSlider.value);
    speedValLabel.textContent = animSpeed + 'px';
  });
  controlsRow2.appendChild(speedSlider);

  const speedValLabel = document.createElement('span');
  speedValLabel.textContent = animSpeed + 'px';
  speedValLabel.style.cssText = 'color:#888;font-size:10px;font-family:' + FONT + ';min-width:28px;';
  controlsRow2.appendChild(speedValLabel);

  makeSep(controlsRow2);

  // pipeline controls
  makeButton(controlsRow2, '+ Kernel', () => {
    addKernelToPipeline();
  });
  makeButton(controlsRow2, '- Kernel', () => {
    removeLastKernel();
  });

  makeSep(controlsRow2);

  // pipeline info
  pipelineInfoEl = document.createElement('span');
  pipelineInfoEl.style.cssText = 'color:' + ACCENT + ';font-size:10px;font-family:' + FONT + ';max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
  controlsRow2.appendChild(pipelineInfoEl);

  // reset button
  makeButton(controlsRow2, 'Reset', () => {
    stopAnimation();
    pipeline = [];
    pipelineResults = [];
    updatePipelineDisplay();
    currentPreset = 'Sobel X';
    presetSelect.value = 'Sobel X';
    loadPreset('Sobel X');
    generateSource('Shapes');
    srcSelect.value = 'Shapes';
    kernelSize = 3;
    updateSizeButtons();
    buildKernelEditor();
    syncKernelToInputs();
    runFullConvolution();
    updateInfoBar();
    requestDraw();
  });

  // ============================================================
  // ANIMATION LOOP — requestAnimationFrame
  // ============================================================

  function loop() {
    // lab pause check — sirf active sim animate kare
    if (window.__labPaused && window.__labPaused !== container.id) {
      animationId = null;
      return;
    }
    if (!isVisible) {
      animationId = null;
      return;
    }

    if (animating && !animDone) {
      animStep();
      draw();
      animationId = requestAnimationFrame(loop);
    } else {
      // animation nahi chal rahi — loop band karo, sirf draw karo
      draw();
      animationId = null;
    }
  }

  function startLoop() {
    if (animationId === null) {
      animationId = requestAnimationFrame(loop);
    }
  }

  function stopLoop() {
    if (animationId !== null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // ============================================================
  // INTERSECTION OBSERVER — sirf visible hone pe resources use kar
  // ============================================================

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        const wasVisible = isVisible;
        isVisible = entry.isIntersecting;
        if (isVisible && !wasVisible) {
          resizeCanvas();
          if (animating) startLoop();
          else requestDraw();
        } else if (!isVisible && wasVisible) {
          stopLoop();
        }
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);

  // lab:resume event — jab pause hataye koi doosri sim ne
  document.addEventListener('lab:resume', () => {
    if (isVisible && !animationId) {
      if (animating) startLoop();
      else requestDraw();
    }
  });

  // tab visibility — tab hide hone pe stop, show pe resume
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopLoop();
    } else if (isVisible) {
      if (animating) startLoop();
      else requestDraw();
    }
  });

  // resize handler — responsive
  window.addEventListener('resize', () => {
    resizeCanvas();
    requestDraw();
  });

  // ============================================================
  // INITIALIZATION — sab setup karke shuru kar
  // ============================================================

  resizeCanvas();
  generateSource('Shapes');
  loadPreset('Sobel X');
  buildKernelEditor();
  syncKernelToInputs();
  runFullConvolution();
  updateInfoBar();
  updatePipelineDisplay();
  updateSizeButtons();
  requestDraw();
}
