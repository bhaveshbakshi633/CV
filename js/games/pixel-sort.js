// ============================================================
// Pixel Sort — Glitch Art Generator
// Pixels ko brightness/hue/saturation se sort karke
// beautiful "melting" glitch effects banao
// Source images generate karo, sort karo, animate karo
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, pixels sort karo
export function initPixelSort() {
  const container = document.getElementById('pixelSortContainer');
  if (!container) {
    console.warn('pixelSortContainer nahi mila bhai, pixel sort skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';

  // sort criteria options
  const SORT_MODES = ['Brightness', 'Hue', 'Saturation', 'Red', 'Green', 'Blue'];
  const DIRECTIONS = ['Horizontal', 'Vertical', 'Both'];
  const SOURCES = ['Gradient', 'Circles', 'Noise', 'Bars'];

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let animationId = null;
  let isVisible = false;

  // sort parameters
  let currentSource = 'Gradient';
  let sortMode = 'Brightness';
  let direction = 'Horizontal';
  let threshold = 80;
  let animating = false;

  // image data — original aur working copy
  let originalImageData = null;  // source image ka backup
  let workingImageData = null;   // is pe sort hoga
  let sortInProgress = false;    // kya abhi sort chal raha hai
  let sortRow = 0;               // current row/col being sorted
  let sortPhase = 0;             // 0=horizontal, 1=vertical (for Both)
  let animThreshold = 0;         // animate mode ke liye oscillating threshold
  let animDirection = 1;         // threshold oscillation direction

  // uploaded image support
  let uploadedImage = null;

  // --- DOM structure banao ---
  // pehle existing children preserve karo
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // canvas — sorted image yahan dikhega
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'background:rgba(2,2,8,0.5)',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  // --- Controls container ---
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

  // row 2 — sliders aur toggle
  const controlsDiv2 = document.createElement('div');
  controlsDiv2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:8px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv2);

  // --- Helper: select dropdown banao ---
  function createSelect(label, options, selected, parent, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = [
      'display:flex',
      'align-items:center',
      'gap:5px',
      'font-family:"JetBrains Mono",monospace',
      'font-size:11px',
      'color:#888',
    ].join(';');

    const lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.whiteSpace = 'nowrap';
    wrap.appendChild(lbl);

    const sel = document.createElement('select');
    sel.style.cssText = [
      'padding:4px 8px',
      'font-size:11px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:"JetBrains Mono",monospace',
    ].join(';');
    options.forEach((opt) => {
      const o = document.createElement('option');
      o.value = opt;
      o.textContent = opt;
      o.style.background = '#1a1a2e';
      if (opt === selected) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('change', () => onChange(sel.value));
    wrap.appendChild(sel);
    parent.appendChild(wrap);
    return sel;
  }

  // --- Helper: button banao ---
  function createButton(text, parent, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:11px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
      'white-space:nowrap',
      'user-select:none',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
        btn.style.color = '#e0e0e0';
      }
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
        btn.style.color = '#b0b0b0';
      }
    });
    btn.addEventListener('click', onClick);
    parent.appendChild(btn);
    return btn;
  }

  function setButtonActive(btn, active) {
    btn._active = active;
    if (active) {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.35)';
      btn.style.color = ACCENT;
      btn.style.borderColor = ACCENT;
    } else {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      btn.style.color = '#b0b0b0';
      btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
    }
  }

  // --- Helper: slider banao ---
  function createSlider(label, min, max, value, step, parent, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = [
      'display:flex',
      'align-items:center',
      'gap:6px',
      'font-family:"JetBrains Mono",monospace',
      'font-size:11px',
      'color:#888',
    ].join(';');

    const lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.whiteSpace = 'nowrap';
    wrap.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.value = value;
    slider.step = step;
    slider.style.cssText = [
      'width:90px',
      'accent-color:' + ACCENT,
      'cursor:pointer',
    ].join(';');

    const valSpan = document.createElement('span');
    valSpan.textContent = value;
    valSpan.style.cssText = 'min-width:28px;text-align:right;color:' + ACCENT + ';font-size:10px;';

    slider.addEventListener('input', () => {
      valSpan.textContent = slider.value;
      onChange(Number(slider.value));
    });

    wrap.appendChild(slider);
    wrap.appendChild(valSpan);
    parent.appendChild(wrap);
    return { slider, valSpan };
  }

  // --- Controls Row 1: Source, Sort By, Direction ---
  const sourceSel = createSelect('Source', SOURCES, currentSource, controlsDiv, (val) => {
    currentSource = val;
    generateSourceImage();
    resetSort();
  });

  createSelect('Sort by', SORT_MODES, sortMode, controlsDiv, (val) => {
    sortMode = val;
  });

  createSelect('Direction', DIRECTIONS, direction, controlsDiv, (val) => {
    direction = val;
  });

  // --- File upload button ---
  const uploadWrap = document.createElement('div');
  uploadWrap.style.cssText = [
    'display:flex',
    'align-items:center',
    'gap:5px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
  ].join(';');

  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/*';
  fileInput.style.display = 'none';

  const uploadBtn = createButton('Upload', uploadWrap, () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        uploadedImage = img;
        currentSource = 'Upload';
        generateSourceImage();
        resetSort();
      };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  });

  uploadWrap.appendChild(fileInput);
  controlsDiv.appendChild(uploadWrap);

  // --- Controls Row 2: Threshold, Sort, Animate, Reset ---
  const thresholdSlider = createSlider('Threshold', 0, 255, threshold, 1, controlsDiv2, (val) => {
    threshold = val;
  });

  // sort button — ek baar sort trigger karo
  const sortBtn = createButton('Sort', controlsDiv2, () => {
    if (sortInProgress) return;
    startSort();
  });

  // animate toggle — continuously sort with oscillating threshold
  const animBtn = createButton('Animate', controlsDiv2, () => {
    animating = !animating;
    setButtonActive(animBtn, animating);
    animBtn.textContent = animating ? 'Animating' : 'Animate';
    if (animating) {
      animThreshold = threshold;
      animDirection = 1;
      // reset to original for clean animation
      restoreOriginal();
      startSort();
    } else {
      sortInProgress = false;
    }
  });

  // reset button — original image restore karo
  createButton('Reset', controlsDiv2, () => {
    animating = false;
    setButtonActive(animBtn, false);
    animBtn.textContent = 'Animate';
    sortInProgress = false;
    restoreOriginal();
  });

  // --- Canvas sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = Math.floor(containerWidth * dpr);
    canvas.height = Math.floor(CANVAS_HEIGHT * dpr);
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // source image regenerate karo naye size ke liye
    generateSourceImage();
    resetSort();
  }

  // --- Color utility functions ---
  // RGB se brightness nikalo (luminance formula)
  function getBrightness(r, g, b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
  }

  // RGB se HSL nikalo — hue/saturation sort ke liye
  function rgbToHsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const l = (max + min) / 2;
    let h = 0, s = 0;

    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
        case g: h = ((b - r) / d + 2) / 6; break;
        case b: h = ((r - g) / d + 4) / 6; break;
      }
    }

    return { h: h * 360, s: s * 100, l: l * 100 };
  }

  // sort criterion ke hisaab se pixel ka value nikalo
  function getSortValue(r, g, b) {
    switch (sortMode) {
      case 'Brightness': return getBrightness(r, g, b);
      case 'Hue': return rgbToHsl(r, g, b).h;
      case 'Saturation': return rgbToHsl(r, g, b).s;
      case 'Red': return r;
      case 'Green': return g;
      case 'Blue': return b;
      default: return getBrightness(r, g, b);
    }
  }

  // --- Perlin noise implementation (simplex-ish, lightweight) ---
  // gradient vectors for 2D noise
  const GRAD = [[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]];
  let perm = new Uint8Array(512);

  // permutation table initialize karo — random shuffle
  function initNoise() {
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    // Fisher-Yates shuffle
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = p[i]; p[i] = p[j]; p[j] = tmp;
    }
    for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  }
  initNoise();

  // 2D Perlin noise — smooth random values 0-1
  function noise2D(x, y) {
    // grid cell coordinates
    const xi = Math.floor(x) & 255;
    const yi = Math.floor(y) & 255;
    // fractional part
    const xf = x - Math.floor(x);
    const yf = y - Math.floor(y);
    // fade curves — smooth interpolation ke liye
    const u = xf * xf * xf * (xf * (xf * 6 - 15) + 10);
    const v = yf * yf * yf * (yf * (yf * 6 - 15) + 10);

    // hash coordinates to gradient indices
    const aa = perm[perm[xi] + yi] & 7;
    const ab = perm[perm[xi] + yi + 1] & 7;
    const ba = perm[perm[xi + 1] + yi] & 7;
    const bb = perm[perm[xi + 1] + yi + 1] & 7;

    // dot products with gradient vectors
    const g00 = GRAD[aa][0] * xf + GRAD[aa][1] * yf;
    const g10 = GRAD[ba][0] * (xf - 1) + GRAD[ba][1] * yf;
    const g01 = GRAD[ab][0] * xf + GRAD[ab][1] * (yf - 1);
    const g11 = GRAD[bb][0] * (xf - 1) + GRAD[bb][1] * (yf - 1);

    // bilinear interpolation
    const x1 = g00 + u * (g10 - g00);
    const x2 = g01 + u * (g11 - g01);
    const val = x1 + v * (x2 - x1);

    // normalize -1..1 to 0..1
    return (val + 1) * 0.5;
  }

  // fractal brownian motion — multiple octaves of noise, zyada detail
  function fbm(x, y, octaves) {
    let val = 0, amp = 0.5, freq = 1;
    for (let i = 0; i < octaves; i++) {
      val += amp * noise2D(x * freq, y * freq);
      amp *= 0.5;
      freq *= 2;
    }
    return val;
  }

  // --- Source image generators ---
  // har ek function canvas pe seedha draw karta hai

  // rainbow gradient with noise — classic test pattern
  function generateGradient() {
    const w = canvasW;
    const h = canvasH;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;

        // hue position — horizontal sweep with some noise
        const nx = fbm(x * 0.008, y * 0.008, 4);
        const hue = ((x / w) * 360 + nx * 120) % 360;
        // brightness varies vertically with noise
        const bright = 0.4 + 0.5 * (y / h) + 0.1 * fbm(x * 0.012 + 5, y * 0.012 + 5, 3);

        // HSL to RGB conversion (simplified)
        const s = 0.8 + 0.2 * fbm(x * 0.005 + 10, y * 0.005, 2);
        const rgb = hslToRgb(hue / 360, s, Math.min(1, Math.max(0, bright)));

        data[idx] = rgb[0];
        data[idx + 1] = rgb[1];
        data[idx + 2] = rgb[2];
        data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // overlapping colored circles on dark background
  function generateCircles() {
    // dark background
    ctx.fillStyle = '#0a0a14';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // bunch of random colored circles — overlapping, different sizes
    const numCircles = 25 + Math.floor(Math.random() * 15);
    for (let i = 0; i < numCircles; i++) {
      const x = Math.random() * canvasW;
      const y = Math.random() * canvasH;
      const r = 20 + Math.random() * Math.min(canvasW, canvasH) * 0.2;

      // vibrant random hue
      const hue = Math.random() * 360;
      const sat = 60 + Math.random() * 40;
      const light = 30 + Math.random() * 40;

      // radial gradient — center bright, edges dark
      const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, 'hsla(' + hue + ',' + sat + '%,' + (light + 20) + '%,0.9)');
      grad.addColorStop(0.5, 'hsla(' + hue + ',' + sat + '%,' + light + '%,0.6)');
      grad.addColorStop(1, 'hsla(' + hue + ',' + (sat * 0.5) + '%,' + (light * 0.3) + '%,0)');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Perlin noise colored field — psychedelic hues
  function generateNoise() {
    const w = canvasW;
    const h = canvasH;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    // noise offset har baar random — different patterns har baar
    const ox = Math.random() * 100;
    const oy = Math.random() * 100;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;

        // multiple noise layers for hue, sat, brightness
        const n1 = fbm(x * 0.006 + ox, y * 0.006 + oy, 5);
        const n2 = fbm(x * 0.01 + ox + 50, y * 0.01 + oy + 50, 3);
        const n3 = fbm(x * 0.004 + ox + 100, y * 0.004 + oy + 100, 4);

        const hue = (n1 * 360 + n3 * 180) % 360;
        const sat = 0.5 + 0.5 * n2;
        const light = 0.15 + 0.6 * n3;

        const rgb = hslToRgb(hue / 360, sat, light);
        data[idx] = rgb[0];
        data[idx + 1] = rgb[1];
        data[idx + 2] = rgb[2];
        data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // vertical colored bars — varying width and brightness
  function generateBars() {
    const w = canvasW;
    const h = canvasH;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    // random bar widths — kuch thin, kuch wide
    let barX = 0;
    const bars = [];
    while (barX < w) {
      const barW = 4 + Math.floor(Math.random() * 30);
      const hue = Math.random() * 360;
      const baseSat = 50 + Math.random() * 50;
      bars.push({ x: barX, w: barW, hue, sat: baseSat });
      barX += barW;
    }

    for (let y = 0; y < h; y++) {
      // brightness varies along height — gradient feel
      const yFactor = 0.2 + 0.8 * (1 - Math.abs(y / h - 0.5) * 2);
      // thoda noise bhi daalo brightness mein
      const yNoise = fbm(y * 0.02, 0.5, 2) * 0.3;

      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;

        // konsa bar hai ye pixel
        let bar = bars[0];
        for (let b = 0; b < bars.length; b++) {
          if (x >= bars[b].x && x < bars[b].x + bars[b].w) {
            bar = bars[b];
            break;
          }
        }

        const light = Math.min(0.85, Math.max(0.05, yFactor * 0.5 + yNoise));
        const rgb = hslToRgb(bar.hue / 360, bar.sat / 100, light);

        data[idx] = rgb[0];
        data[idx + 1] = rgb[1];
        data[idx + 2] = rgb[2];
        data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // uploaded image draw karo — canvas pe fit karke
  function generateUpload() {
    if (!uploadedImage) {
      generateGradient(); // fallback agar image nahi hai
      return;
    }

    ctx.fillStyle = '#0a0a14';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // aspect ratio maintain karke canvas mein fit karo
    const imgW = uploadedImage.width;
    const imgH = uploadedImage.height;
    const scale = Math.min(canvasW / imgW, canvasH / imgH);
    const drawW = imgW * scale;
    const drawH = imgH * scale;
    const offX = (canvasW - drawW) / 2;
    const offY = (canvasH - drawH) / 2;

    ctx.drawImage(uploadedImage, offX, offY, drawW, drawH);
  }

  // HSL to RGB conversion — standard algorithm
  function hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) {
      r = g = b = l;
    } else {
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
      };
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1/3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1/3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }

  // --- Source image generate karo based on selection ---
  function generateSourceImage() {
    if (canvasW <= 0 || canvasH <= 0) return;

    switch (currentSource) {
      case 'Gradient': generateGradient(); break;
      case 'Circles': generateCircles(); break;
      case 'Noise': generateNoise(); break;
      case 'Bars': generateBars(); break;
      case 'Upload': generateUpload(); break;
      default: generateGradient();
    }

    // original save karo — reset ke liye
    originalImageData = ctx.getImageData(0, 0, canvasW, canvasH);
    // working copy banao
    workingImageData = ctx.getImageData(0, 0, canvasW, canvasH);
  }

  // original image restore karo
  function restoreOriginal() {
    if (!originalImageData) return;
    // fresh copy banao original se
    workingImageData = new ImageData(
      new Uint8ClampedArray(originalImageData.data),
      originalImageData.width,
      originalImageData.height
    );
    ctx.putImageData(workingImageData, 0, 0);
    sortInProgress = false;
    sortRow = 0;
    sortPhase = 0;
  }

  // --- Pixel sorting algorithm ---
  // ek row (ya column) ko sort karo — span detection + sort within spans

  // horizontal row sort — left to right scan
  function sortRow_h(imgData, y, thresh) {
    const w = imgData.width;
    const data = imgData.data;
    let x = 0;

    while (x < w) {
      // span dhundho — consecutive pixels jahan brightness > threshold
      const spanStart = x;
      const span = [];

      while (x < w) {
        const idx = (y * w + x) * 4;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
        const bright = getBrightness(r, g, b);

        if (bright > thresh) {
          span.push({ r, g, b, a: data[idx + 3], val: getSortValue(r, g, b) });
          x++;
        } else {
          // brightness threshold se neeche — span khatam
          break;
        }
      }

      // agar span mila toh sort karo
      if (span.length > 1) {
        span.sort((a, b) => a.val - b.val);
        // sorted pixels wapas likh do
        for (let i = 0; i < span.length; i++) {
          const px = spanStart + i;
          const idx = (y * w + px) * 4;
          data[idx] = span[i].r;
          data[idx + 1] = span[i].g;
          data[idx + 2] = span[i].b;
          data[idx + 3] = span[i].a;
        }
      }

      // agar span nahi mila (pixel below threshold), aage badho
      if (span.length === 0) x++;
    }
  }

  // vertical column sort — top to bottom scan
  function sortCol_v(imgData, x, thresh) {
    const w = imgData.width;
    const h = imgData.height;
    const data = imgData.data;
    let y = 0;

    while (y < h) {
      const spanStart = y;
      const span = [];

      while (y < h) {
        const idx = (y * w + x) * 4;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
        const bright = getBrightness(r, g, b);

        if (bright > thresh) {
          span.push({ r, g, b, a: data[idx + 3], val: getSortValue(r, g, b) });
          y++;
        } else {
          break;
        }
      }

      if (span.length > 1) {
        span.sort((a, b) => a.val - b.val);
        for (let i = 0; i < span.length; i++) {
          const py = spanStart + i;
          const idx = (py * w + x) * 4;
          data[idx] = span[i].r;
          data[idx + 1] = span[i].g;
          data[idx + 2] = span[i].b;
          data[idx + 3] = span[i].a;
        }
      }

      if (span.length === 0) y++;
    }
  }

  // --- Sort process management ---
  // progressive sort — kuch rows/cols per frame, wave effect ke liye

  function startSort() {
    if (!workingImageData || !originalImageData) return;

    // fresh copy se shuru karo agar animate nahi ho raha
    if (!animating) {
      workingImageData = new ImageData(
        new Uint8ClampedArray(originalImageData.data),
        originalImageData.width,
        originalImageData.height
      );
    }

    sortRow = 0;
    sortPhase = 0;
    sortInProgress = true;
  }

  function resetSort() {
    sortInProgress = false;
    sortRow = 0;
    sortPhase = 0;
  }

  // ek frame mein kitni rows/cols sort karein — wave speed
  // zyada rows = fast sort, kam rows = slow dramatic sweep
  function getSortBatchSize() {
    // canvas height ke hisaab se adjust kar — ~2-3 seconds mein pura sort ho
    return Math.max(2, Math.floor(Math.max(canvasW, canvasH) / 60));
  }

  // ek frame ka sort step — progressive
  function sortStep() {
    if (!sortInProgress || !workingImageData) return;

    const w = workingImageData.width;
    const h = workingImageData.height;
    const batch = getSortBatchSize();
    const currentThresh = animating ? animThreshold : threshold;

    if (direction === 'Horizontal' || (direction === 'Both' && sortPhase === 0)) {
      // horizontal sort — row by row
      const endRow = Math.min(sortRow + batch, h);
      for (let y = sortRow; y < endRow; y++) {
        sortRow_h(workingImageData, y, currentThresh);
      }
      sortRow = endRow;

      if (sortRow >= h) {
        if (direction === 'Both') {
          // horizontal done, ab vertical shuru karo
          sortPhase = 1;
          sortRow = 0;
        } else {
          // sort complete
          sortInProgress = false;
          if (animating) {
            // animate mode — threshold oscillate karo, dubara sort shuru karo
            nextAnimCycle();
          }
        }
      }
    } else if (direction === 'Vertical' || (direction === 'Both' && sortPhase === 1)) {
      // vertical sort — column by column
      const endCol = Math.min(sortRow + batch, w);
      for (let x = sortRow; x < endCol; x++) {
        sortCol_v(workingImageData, x, currentThresh);
      }
      sortRow = endCol;

      if (sortRow >= w) {
        sortInProgress = false;
        if (animating) {
          nextAnimCycle();
        }
      }
    }

    // sorted image render karo
    ctx.putImageData(workingImageData, 0, 0);
  }

  // animate mode ka next cycle — threshold change karo, dubara sort shuru
  function nextAnimCycle() {
    // threshold oscillate karo — up and down
    animThreshold += animDirection * 15;
    if (animThreshold >= 220) {
      animThreshold = 220;
      animDirection = -1;
    } else if (animThreshold <= 20) {
      animThreshold = 20;
      animDirection = 1;
    }

    // threshold slider bhi update karo — visual feedback
    thresholdSlider.slider.value = Math.round(animThreshold);
    thresholdSlider.valSpan.textContent = Math.round(animThreshold);

    // fresh copy se dubara sort shuru karo
    workingImageData = new ImageData(
      new Uint8ClampedArray(originalImageData.data),
      originalImageData.width,
      originalImageData.height
    );
    sortRow = 0;
    sortPhase = 0;
    sortInProgress = true;
  }

  // --- Animation loop ---
  function loop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    // agar sort chal raha hai toh har frame pe kuch rows sort karo
    if (sortInProgress) {
      sortStep();
    }

    animationId = requestAnimationFrame(loop);
  }

  function startLoop() {
    if (!animationId && isVisible) {
      animationId = requestAnimationFrame(loop);
    }
  }

  function stopLoop() {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // --- Resize ---
  resizeCanvas();
  window.addEventListener('resize', () => {
    resizeCanvas();
  });

  // --- IntersectionObserver — sirf visible hone pe CPU use karo ---
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          isVisible = true;
          startLoop();
        } else {
          isVisible = false;
          stopLoop();
        }
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });

  // tab switch pe pause — background mein CPU waste mat karo
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      isVisible = false;
      stopLoop();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) {
        isVisible = true;
        startLoop();
      }
    }
  });
}
