// ============================================================
// Mandelbrot / Julia Set Explorer — fractal duniya mein ghoom
// Click se zoom karo, hover pe Julia set dekho, color schemes badlo
// Smooth coloring + escape-time algorithm — beautiful gradients
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, fractals render karo
export function initMandelbrot() {
  const container = document.getElementById('mandelbrotContainer');
  if (!container) {
    console.warn('mandelbrotContainer nahi mila bhai, Mandelbrot demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 380;
  const ACCENT = '#a78bfa'; // purple accent
  const ACCENT_RGB = '167,139,250';
  const DEFAULT_CENTER_X = -0.5;
  const DEFAULT_CENTER_Y = 0;
  const DEFAULT_RANGE_X = 3; // total width of view (-2 to 1)
  const DEFAULT_RANGE_Y = 2.4; // total height of view (-1.2 to 1.2)
  const ZOOM_FACTOR = 3;
  const PREVIEW_SCALE = 0.25; // low-res preview — 1/4 resolution

  // --- Color scheme definitions ---
  // har scheme mein stops hain — position (0-1) pe RGB value
  const COLOR_SCHEMES = {
    fire: {
      name: 'Fire',
      stops: [
        [0.0, 0, 0, 0],
        [0.05, 40, 0, 10],
        [0.15, 120, 10, 0],
        [0.35, 220, 80, 0],
        [0.55, 255, 180, 20],
        [0.75, 255, 245, 120],
        [0.9, 255, 255, 220],
        [1.0, 255, 255, 255],
      ],
    },
    ocean: {
      name: 'Ocean',
      stops: [
        [0.0, 0, 0, 0],
        [0.05, 0, 8, 30],
        [0.15, 0, 30, 80],
        [0.3, 0, 80, 160],
        [0.5, 20, 150, 220],
        [0.65, 80, 200, 255],
        [0.8, 180, 235, 255],
        [0.95, 240, 250, 255],
        [1.0, 255, 255, 255],
      ],
    },
    neon: {
      name: 'Neon',
      stops: [
        [0.0, 0, 0, 0],
        [0.05, 20, 0, 40],
        [0.12, 80, 0, 120],
        [0.25, 167, 50, 250],
        [0.4, 255, 50, 200],
        [0.55, 255, 100, 80],
        [0.7, 50, 255, 150],
        [0.85, 20, 180, 255],
        [0.95, 167, 139, 250],
        [1.0, 255, 255, 255],
      ],
    },
    grayscale: {
      name: 'Grayscale',
      stops: [
        [0.0, 0, 0, 0],
        [0.15, 30, 30, 30],
        [0.3, 70, 70, 70],
        [0.5, 130, 130, 130],
        [0.7, 190, 190, 190],
        [0.85, 230, 230, 230],
        [1.0, 255, 255, 255],
      ],
    },
  };

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;

  // view state — current center aur range
  let centerX = DEFAULT_CENTER_X;
  let centerY = DEFAULT_CENTER_Y;
  let rangeX = DEFAULT_RANGE_X;
  let rangeY = DEFAULT_RANGE_Y;
  let maxIter = 150;
  let colorScheme = 'neon';

  // Julia set mode
  let juliaMode = false;
  let juliaCx = -0.7;
  let juliaCy = 0.27;

  // mouse state
  let mouseX = -1, mouseY = -1;
  let mouseInCanvas = false;

  // rendering state
  let renderPending = false;
  let needsRender = true;

  // cached fractal image — overlay ke liye re-render avoid karne ke liye
  let cachedFractalCanvas = null;

  // animation / visibility
  let animationId = null;
  let isVisible = false;

  // color lookup table — precompute karte hain for speed
  let colorLUT = null;
  const LUT_SIZE = 2048;

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — fractal yahan dikhega
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:rgba(2,2,8,0.5)',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // info bar — coordinates aur zoom level dikhayega
  const infoBar = document.createElement('div');
  infoBar.style.cssText = [
    'display:flex',
    'justify-content:space-between',
    'align-items:center',
    'margin-top:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:rgba(' + ACCENT_RGB + ',0.6)',
    'min-height:18px',
    'flex-wrap:wrap',
    'gap:4px 12px',
  ].join(';');
  container.appendChild(infoBar);

  const coordsSpan = document.createElement('span');
  coordsSpan.textContent = 'Move mouse over fractal';
  infoBar.appendChild(coordsSpan);

  const zoomSpan = document.createElement('span');
  zoomSpan.textContent = 'Zoom: 1.0x';
  infoBar.appendChild(zoomSpan);

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
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
        btn.style.color = '#b0b0b0';
      }
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
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
  function createSlider(label, min, max, value, step, onChange) {
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
      'width:80px',
      'accent-color:' + ACCENT,
      'cursor:pointer',
    ].join(';');

    const valSpan = document.createElement('span');
    valSpan.textContent = value;
    valSpan.style.cssText = 'min-width:28px;text-align:right;color:' + ACCENT + ';';

    slider.addEventListener('input', () => {
      valSpan.textContent = slider.value;
      onChange(Number(slider.value));
    });

    wrap.appendChild(slider);
    wrap.appendChild(valSpan);
    controlsDiv.appendChild(wrap);
    return { slider, valSpan };
  }

  // --- Helper: select dropdown banao ---
  function createSelect(options, selected, onChange) {
    const sel = document.createElement('select');
    sel.style.cssText = [
      'padding:5px 8px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:"JetBrains Mono",monospace',
    ].join(';');
    options.forEach((opt) => {
      const o = document.createElement('option');
      o.value = opt.value;
      o.textContent = opt.label;
      o.style.background = '#1a1a2e';
      if (opt.value === selected) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('change', () => onChange(sel.value));
    controlsDiv.appendChild(sel);
    return sel;
  }

  // --- Controls banao ---

  // Reset button
  createButton('Reset View', () => {
    centerX = DEFAULT_CENTER_X;
    centerY = DEFAULT_CENTER_Y;
    rangeX = DEFAULT_RANGE_X;
    rangeY = DEFAULT_RANGE_Y;
    juliaMode = false;
    setButtonActive(juliaBtn, false);
    juliaBtn.textContent = 'Julia Mode';
    scheduleRender();
  });

  // Julia mode toggle
  const juliaBtn = createButton('Julia Mode', () => {
    juliaMode = !juliaMode;
    setButtonActive(juliaBtn, juliaMode);
    juliaBtn.textContent = juliaMode ? 'Julia Mode ON' : 'Julia Mode';
    if (juliaMode) {
      // Julia mode on — current mouse position ko c bana do, ya default rakh do
      // center reset karo Julia ke liye
      centerX = 0;
      centerY = 0;
      rangeX = DEFAULT_RANGE_X;
      rangeY = DEFAULT_RANGE_Y;
    } else {
      // Mandelbrot pe waapas ja
      centerX = DEFAULT_CENTER_X;
      centerY = DEFAULT_CENTER_Y;
      rangeX = DEFAULT_RANGE_X;
      rangeY = DEFAULT_RANGE_Y;
    }
    scheduleRender();
  });

  // Iterations slider
  createSlider('Iter', 50, 500, maxIter, 10, (val) => {
    maxIter = val;
    buildColorLUT();
    scheduleRender();
  });

  // Color scheme selector
  createSelect(
    Object.keys(COLOR_SCHEMES).map((k) => ({
      value: k,
      label: COLOR_SCHEMES[k].name,
    })),
    colorScheme,
    (val) => {
      colorScheme = val;
      buildColorLUT();
      scheduleRender();
    }
  );

  // --- Canvas sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = containerWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scheduleRender();
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Color LUT banao — precomputed gradient for fast pixel coloring ---
  function buildColorLUT() {
    const scheme = COLOR_SCHEMES[colorScheme];
    if (!scheme) return;
    colorLUT = new Uint8Array(LUT_SIZE * 3);

    for (let i = 0; i < LUT_SIZE; i++) {
      // t = 0 to 1, cyclic pattern for deep zooms
      const t = i / LUT_SIZE;
      const rgb = interpolateScheme(scheme.stops, t);
      colorLUT[i * 3] = rgb[0];
      colorLUT[i * 3 + 1] = rgb[1];
      colorLUT[i * 3 + 2] = rgb[2];
    }
  }

  // linear interpolation between color stops
  function interpolateScheme(stops, t) {
    // t ko clamp karo 0-1 range mein
    t = Math.max(0, Math.min(1, t));

    // sahi stops dhundho
    let i = 0;
    while (i < stops.length - 1 && stops[i + 1][0] < t) i++;
    if (i >= stops.length - 1) {
      return [stops[stops.length - 1][1], stops[stops.length - 1][2], stops[stops.length - 1][3]];
    }

    const s0 = stops[i];
    const s1 = stops[i + 1];
    const f = (t - s0[0]) / (s1[0] - s0[0]);
    // smooth interpolation — cosine based for nicer gradients
    const smoothF = (1 - Math.cos(f * Math.PI)) / 2;

    return [
      Math.round(s0[1] + (s1[1] - s0[1]) * smoothF),
      Math.round(s0[2] + (s1[2] - s0[2]) * smoothF),
      Math.round(s0[3] + (s1[3] - s0[3]) * smoothF),
    ];
  }

  buildColorLUT();

  // --- Fractal math ---
  // pixel coordinates se complex plane coordinates mein convert karo
  function pixelToComplex(px, py) {
    // canvas ke pixels ko complex number mein map karo
    const aspect = canvasW / canvasH;
    const rX = rangeX;
    const rY = rangeX / aspect; // aspect ratio maintain karo
    const re = centerX + (px / canvasW - 0.5) * rX;
    const im = centerY + (py / canvasH - 0.5) * rY;
    return [re, im];
  }

  // Mandelbrot iteration — smooth coloring ke saath
  // returns: iteration count (fractional for smooth), ya -1 agar set ke andar hai
  function mandelbrotIter(cRe, cIm, maxN) {
    let zRe = 0, zIm = 0;
    let zRe2 = 0, zIm2 = 0;

    for (let n = 0; n < maxN; n++) {
      zIm = 2 * zRe * zIm + cIm;
      zRe = zRe2 - zIm2 + cRe;
      zRe2 = zRe * zRe;
      zIm2 = zIm * zIm;

      // escape check — |z|^2 > 4 (|z| > 2)
      // bailout thoda zyada rakhte hain for smoother coloring
      if (zRe2 + zIm2 > 256) {
        // smooth iteration count — fractional part add karo
        // formula: n + 1 - log(log(|z|)) / log(2)
        const logZn = Math.log(zRe2 + zIm2) / 2;
        const nu = Math.log(logZn / Math.LN2) / Math.LN2;
        return n + 1 - nu;
      }
    }

    // set ke andar — black
    return -1;
  }

  // Julia iteration — same formula but c fixed hai, z = pixel position
  function juliaIter(zRe, zIm, cRe, cIm, maxN) {
    let zRe2 = zRe * zRe;
    let zIm2 = zIm * zIm;

    for (let n = 0; n < maxN; n++) {
      if (zRe2 + zIm2 > 256) {
        const logZn = Math.log(zRe2 + zIm2) / 2;
        const nu = Math.log(logZn / Math.LN2) / Math.LN2;
        return n + 1 - nu;
      }
      zIm = 2 * zRe * zIm + cIm;
      zRe = zRe2 - zIm2 + cRe;
      zRe2 = zRe * zRe;
      zIm2 = zIm * zIm;
    }

    return -1;
  }

  // --- Rendering ---
  // iteration count se color nikalo — LUT use karo
  function iterToColor(iter) {
    if (iter < 0) return [0, 0, 0]; // set ke andar — black

    // smooth cyclic mapping through the LUT
    // log scale use karo for nice distribution at deep zooms
    const t = (Math.log(iter + 1) / Math.log(maxIter + 1));
    // cycle through LUT multiple times for more color variation
    const lutIdx = Math.floor((t * 3.5 % 1) * (LUT_SIZE - 1));
    const idx = Math.max(0, Math.min(LUT_SIZE - 1, lutIdx)) * 3;

    return [colorLUT[idx], colorLUT[idx + 1], colorLUT[idx + 2]];
  }

  // fractal render karo — pehle low-res, fir full-res
  function renderFractal(lowRes) {
    const scale = lowRes ? PREVIEW_SCALE : 1;
    // actual pixel dimensions (DPR account karna hai for full-res, but for perf use CSS dims)
    const w = Math.ceil(canvasW * scale);
    const h = Math.ceil(canvasH * scale);

    if (w <= 0 || h <= 0) return;

    const imageData = ctx.createImageData(w, h);
    const data = imageData.data;

    const aspect = canvasW / canvasH;
    const rX = rangeX;
    const rY = rangeX / aspect;
    const startRe = centerX - rX / 2;
    const startIm = centerY - rY / 2;
    const stepRe = rX / w;
    const stepIm = rY / h;

    const isJulia = juliaMode;
    const jCx = juliaCx;
    const jCy = juliaCy;
    const maxN = maxIter;

    for (let py = 0; py < h; py++) {
      const im = startIm + py * stepIm;
      for (let px = 0; px < w; px++) {
        const re = startRe + px * stepRe;

        let iter;
        if (isJulia) {
          iter = juliaIter(re, im, jCx, jCy, maxN);
        } else {
          iter = mandelbrotIter(re, im, maxN);
        }

        const color = iterToColor(iter);
        const idx = (py * w + px) * 4;
        data[idx] = color[0];
        data[idx + 1] = color[1];
        data[idx + 2] = color[2];
        data[idx + 3] = 255;
      }
    }

    // canvas pe draw karo — scale up agar low-res hai
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = w;
    tmpCanvas.height = h;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.putImageData(imageData, 0, 0);

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = !lowRes; // pixelated preview, smooth full-res
    ctx.drawImage(tmpCanvas, 0, 0, canvas.width, canvas.height);
    ctx.restore();
    ctx.imageSmoothingEnabled = true;

    // full-res render cache karo — overlay ke liye
    if (!lowRes) {
      cachedFractalCanvas = document.createElement('canvas');
      cachedFractalCanvas.width = canvas.width;
      cachedFractalCanvas.height = canvas.height;
      const cacheCtx = cachedFractalCanvas.getContext('2d');
      cacheCtx.drawImage(tmpCanvas, 0, 0, canvas.width, canvas.height);
    }
  }

  // crosshair draw karo mouse position pe
  function drawCrosshair() {
    if (!mouseInCanvas || mouseX < 0 || mouseY < 0) return;

    ctx.save();
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    // vertical line
    ctx.beginPath();
    ctx.moveTo(mouseX, 0);
    ctx.lineTo(mouseX, canvasH);
    ctx.stroke();

    // horizontal line
    ctx.beginPath();
    ctx.moveTo(0, mouseY);
    ctx.lineTo(canvasW, mouseY);
    ctx.stroke();

    ctx.setLineDash([]);

    // center dot
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, 3, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  // julia preview — chhota sa Julia set dikhao corner mein based on mouse position
  function drawJuliaPreview() {
    if (!mouseInCanvas || juliaMode) return;

    const previewSize = Math.min(100, Math.floor(canvasW * 0.18));
    if (previewSize < 40) return;

    const margin = 8;
    const px0 = canvasW - previewSize - margin;
    const py0 = margin;

    // mouse position se c parameter nikalo
    const [cRe, cIm] = pixelToComplex(mouseX, mouseY);

    // Julia set render karo chhote resolution pe
    const s = previewSize;
    const imageData = ctx.createImageData(s, s);
    const data = imageData.data;
    const jRange = 3.2; // Julia set ka view range
    const halfRange = jRange / 2;
    const step = jRange / s;
    // Julia ke liye thode kam iterations — preview hai, speed chahiye
    const previewIter = Math.min(maxIter, 80);

    for (let py = 0; py < s; py++) {
      const im = -halfRange + py * step;
      for (let px = 0; px < s; px++) {
        const re = -halfRange + px * step;
        const iter = juliaIter(re, im, cRe, cIm, previewIter);
        const color = iterToColor(iter);
        const idx = (py * s + px) * 4;
        data[idx] = color[0];
        data[idx + 1] = color[1];
        data[idx + 2] = color[2];
        data[idx + 3] = 255;
      }
    }

    // preview box draw karo
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = s;
    tmpCanvas.height = s;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.putImageData(imageData, 0, 0);

    // border draw karo
    ctx.save();
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(px0 - 1, py0 - 1, previewSize + 2, previewSize + 2);

    // preview image
    ctx.drawImage(tmpCanvas, px0, py0, previewSize, previewSize);

    // label
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.7)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText('Julia Preview', px0, py0 + previewSize + 11);

    // c value dikhao
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
    ctx.fillText('c = ' + cRe.toFixed(3) + ' + ' + cIm.toFixed(3) + 'i', px0, py0 + previewSize + 22);

    ctx.restore();
  }

  // cached fractal restore karo — overlay redraw ke liye fractal dubara render nahi karna padega
  function restoreFromCache() {
    if (!cachedFractalCanvas) return false;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(cachedFractalCanvas, 0, 0, canvas.width, canvas.height);
    ctx.restore();
    return true;
  }

  // --- Render scheduling — pehle preview, fir full ---
  function scheduleRender() {
    needsRender = true;
    // frame schedule karo agar visible hai
    if (isVisible) scheduleFrame();
  }

  function doRender() {
    if (!isVisible) return;
    if (!needsRender) return;
    needsRender = false;

    // pehle low-res preview — instant feedback
    renderFractal(true);
    drawOverlay();

    // fir full-res render schedule karo
    // timeout se karo taaki UI block na ho zoom ke waqt
    if (renderPending) return;
    renderPending = true;

    requestAnimationFrame(() => {
      renderFractal(false);
      drawOverlay();
      renderPending = false;
    });
  }

  // overlay draw karo — crosshair + Julia preview
  function drawOverlay() {
    drawCrosshair();
    drawJuliaPreview();
  }

  // --- Info bar update ---
  function updateInfo() {
    if (mouseInCanvas && mouseX >= 0 && mouseY >= 0) {
      const [re, im] = pixelToComplex(mouseX, mouseY);
      const sign = im >= 0 ? '+' : '';
      coordsSpan.textContent = (juliaMode ? 'Julia | z = ' : 'c = ') + re.toFixed(6) + ' ' + sign + im.toFixed(6) + 'i';
    } else {
      coordsSpan.textContent = juliaMode ? 'Julia Set Mode — click to set c' : 'Click to zoom in';
    }

    const zoomLevel = DEFAULT_RANGE_X / rangeX;
    if (zoomLevel >= 1000000) {
      zoomSpan.textContent = 'Zoom: ' + (zoomLevel / 1000000).toFixed(1) + 'Mx';
    } else if (zoomLevel >= 1000) {
      zoomSpan.textContent = 'Zoom: ' + (zoomLevel / 1000).toFixed(1) + 'Kx';
    } else {
      zoomSpan.textContent = 'Zoom: ' + zoomLevel.toFixed(1) + 'x';
    }
  }

  // --- Mouse events ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  // mousemove throttle — Julia preview heavy hai, har frame pe nahi karna
  let overlayRafId = null;

  canvas.addEventListener('mousemove', (e) => {
    const pos = getCanvasPos(e);
    mouseX = pos.x;
    mouseY = pos.y;
    mouseInCanvas = true;
    updateInfo();

    // throttled overlay redraw — rAF se ek frame mein ek baar
    if (!overlayRafId && !needsRender && !renderPending && cachedFractalCanvas) {
      overlayRafId = requestAnimationFrame(() => {
        overlayRafId = null;
        if (cachedFractalCanvas) {
          restoreFromCache();
          drawOverlay();
        }
      });
    }
  });

  canvas.addEventListener('mouseleave', () => {
    mouseInCanvas = false;
    mouseX = -1;
    mouseY = -1;
    updateInfo();
    // overlay hata do — clean fractal dikhao cache se
    if (!needsRender && !renderPending && cachedFractalCanvas) {
      restoreFromCache();
    }
  });

  // click — zoom in ya Julia c set karo
  canvas.addEventListener('click', (e) => {
    const pos = getCanvasPos(e);
    const [re, im] = pixelToComplex(pos.x, pos.y);

    if (juliaMode) {
      // Julia mode mein click se c set karo
      juliaCx = re;
      juliaCy = im;
    } else {
      // zoom in — center ko click position pe le jao
      if (e.shiftKey) {
        // shift+click = zoom out
        centerX = re;
        centerY = im;
        rangeX *= ZOOM_FACTOR;
        rangeY *= ZOOM_FACTOR;
      } else {
        // normal click = zoom in
        centerX = re;
        centerY = im;
        rangeX /= ZOOM_FACTOR;
        rangeY /= ZOOM_FACTOR;
      }
    }
    scheduleRender();
  });

  // right click — zoom out
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const [re, im] = pixelToComplex(pos.x, pos.y);
    centerX = re;
    centerY = im;
    rangeX *= ZOOM_FACTOR;
    rangeY *= ZOOM_FACTOR;
    scheduleRender();
  });

  // touch support — tap to zoom
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    mouseX = pos.x;
    mouseY = pos.y;
    mouseInCanvas = true;
  });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (mouseX < 0) return;
    const [re, im] = pixelToComplex(mouseX, mouseY);

    if (juliaMode) {
      juliaCx = re;
      juliaCy = im;
    } else {
      centerX = re;
      centerY = im;
      rangeX /= ZOOM_FACTOR;
      rangeY /= ZOOM_FACTOR;
    }
    mouseInCanvas = false;
    scheduleRender();
  });

  // --- Render on demand — fractal mein continuous animation nahi chahiye ---
  // jab bhi scheduleRender hoga, doRender call hoga via rAF
  function scheduleFrame() {
    if (animationId || !isVisible) return;
    animationId = requestAnimationFrame(() => {
      animationId = null;
      doRender();
      updateInfo();
    });
  }

  // --- IntersectionObserver — sirf jab dikhe tab render karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    needsRender = true;
    scheduleFrame();
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

  // tab switch pe pause
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });
}
