// ============================================================
// Strange Attractor Point Art — mathematical beauty iterated millions of times
// Clifford, De Jong, Svensson, Bedhead attractors
// Density accumulation + log-scale coloring = professional generative art
// ============================================================

// yahi se sab shuru hota hai — container dhundho, density buffer banao, points iterate karo
export function initStrangeAttractors() {
  const container = document.getElementById('strangeAttractorsContainer');
  if (!container) {
    console.warn('strangeAttractorsContainer nahi mila bhai, Strange Attractors skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';

  // progressive rendering — pehle thode points, fir aur zyada
  const INITIAL_POINTS = 200000;    // pehle frame mein itne points
  const POINTS_PER_FRAME = 500000;  // baad ke frames mein add karo
  const MAX_TOTAL_POINTS = 2000000; // itne ke baad ruk jao

  // --- Attractor definitions ---
  // har attractor ka apna formula, default params, aur param ranges
  const ATTRACTORS = {
    clifford: {
      name: 'Clifford',
      params: { a: -1.4, b: 1.6, c: 1.0, d: 0.7 },
      ranges: { a: [-3, 3], b: [-3, 3], c: [-3, 3], d: [-3, 3] },
      // clifford formula: x' = sin(a*y) + c*cos(a*x), y' = sin(b*x) + d*cos(b*y)
      iterate(x, y, p) {
        return [
          Math.sin(p.a * y) + p.c * Math.cos(p.a * x),
          Math.sin(p.b * x) + p.d * Math.cos(p.b * y)
        ];
      }
    },
    dejong: {
      name: 'De Jong',
      params: { a: 1.4, b: -2.3, c: 2.4, d: -2.1 },
      ranges: { a: [-3, 3], b: [-3, 3], c: [-3, 3], d: [-3, 3] },
      // de jong formula: x' = sin(a*y) - cos(b*x), y' = sin(c*x) - cos(d*y)
      iterate(x, y, p) {
        return [
          Math.sin(p.a * y) - Math.cos(p.b * x),
          Math.sin(p.c * x) - Math.cos(p.d * y)
        ];
      }
    },
    svensson: {
      name: 'Svensson',
      params: { a: 1.5, b: -1.8, c: 1.6, d: 0.9 },
      ranges: { a: [-3, 3], b: [-3, 3], c: [-3, 3], d: [-3, 3] },
      // svensson formula: x' = d*sin(a*x) - sin(b*y), y' = c*cos(a*x) + cos(b*y)
      iterate(x, y, p) {
        return [
          p.d * Math.sin(p.a * x) - Math.sin(p.b * y),
          p.c * Math.cos(p.a * x) + Math.cos(p.b * y)
        ];
      }
    },
    bedhead: {
      name: 'Bedhead',
      params: { a: 0.06, b: 0.98 },
      ranges: { a: [-1, 1], b: [0.01, 2] },
      // bedhead formula: x' = sin(x*y/b)*y + cos(a*x - y), y' = x + sin(y)/b
      iterate(x, y, p) {
        return [
          Math.sin(x * y / p.b) * y + Math.cos(p.a * x - y),
          x + Math.sin(y) / p.b
        ];
      }
    }
  };

  // --- Color scheme definitions ---
  // har scheme mein gradient stops hain — density 0-1 ke beech interpolate hoga
  const COLOR_SCHEMES = {
    fire: {
      name: 'Fire',
      stops: [
        [0.0, 0, 0, 0],
        [0.1, 50, 5, 0],
        [0.25, 150, 20, 0],
        [0.4, 220, 80, 0],
        [0.6, 255, 160, 20],
        [0.8, 255, 230, 80],
        [0.95, 255, 255, 200],
        [1.0, 255, 255, 255]
      ]
    },
    ocean: {
      name: 'Ocean',
      stops: [
        [0.0, 0, 0, 0],
        [0.1, 0, 10, 40],
        [0.25, 0, 40, 100],
        [0.4, 0, 90, 170],
        [0.6, 30, 160, 230],
        [0.8, 100, 210, 255],
        [0.95, 200, 240, 255],
        [1.0, 255, 255, 255]
      ]
    },
    neon: {
      name: 'Neon',
      stops: [
        [0.0, 0, 0, 0],
        [0.08, 20, 0, 50],
        [0.2, 80, 0, 140],
        [0.35, 167, 40, 250],
        [0.5, 255, 60, 200],
        [0.65, 255, 100, 150],
        [0.8, 255, 160, 200],
        [0.95, 255, 220, 255],
        [1.0, 255, 255, 255]
      ]
    },
    viridis: {
      name: 'Viridis',
      stops: [
        [0.0, 0, 0, 0],
        [0.1, 68, 1, 84],
        [0.25, 59, 28, 140],
        [0.4, 33, 85, 141],
        [0.55, 32, 145, 140],
        [0.7, 53, 183, 121],
        [0.85, 144, 215, 67],
        [1.0, 253, 231, 37]
      ]
    }
  };

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;

  // current attractor aur params
  let currentAttractorKey = 'clifford';
  let params = { ...ATTRACTORS.clifford.params };
  let colorSchemeKey = 'fire';

  // density buffer — pixel pe kitne points gire, woh count karo
  let densityBuffer = null;
  let bufferW = 0, bufferH = 0;

  // iteration state — progressive rendering ke liye
  let totalPointsRendered = 0;
  let iterX = 0.1, iterY = 0.1; // current iteration position
  let maxDensity = 0;           // normalization ke liye

  // bounds tracking — auto-scale ke liye
  let boundsMinX = Infinity, boundsMaxX = -Infinity;
  let boundsMinY = Infinity, boundsMaxY = -Infinity;
  let boundsComputed = false;

  // animation / visibility
  let animationId = null;
  let isVisible = false;
  let renderComplete = false; // jab max points ho jaayein

  // color LUT — precomputed gradient for speed
  const LUT_SIZE = 1024;
  let colorLUT = null;

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — attractor yahan dikhega
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'background:#000000'
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // info bar — current params dikhayega
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
    'gap:4px 12px'
  ].join(';');
  container.appendChild(infoBar);

  const paramsSpan = document.createElement('span');
  paramsSpan.textContent = '';
  infoBar.appendChild(paramsSpan);

  const pointsSpan = document.createElement('span');
  pointsSpan.textContent = 'Points: 0';
  infoBar.appendChild(pointsSpan);

  // controls container
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center'
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
      'white-space:nowrap'
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // --- Helper: slider banao with label ---
  function createSlider(label, min, max, value, step, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = [
      'display:flex',
      'align-items:center',
      'gap:6px',
      'font-family:"JetBrains Mono",monospace',
      'font-size:11px',
      'color:#888'
    ].join(';');

    const lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.cssText = 'white-space:nowrap;min-width:12px;';
    wrap.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.value = value;
    slider.step = step;
    slider.style.cssText = 'width:80px;accent-color:' + ACCENT + ';cursor:pointer;';

    const valSpan = document.createElement('span');
    valSpan.textContent = Number(value).toFixed(2);
    valSpan.style.cssText = 'min-width:38px;text-align:right;color:' + ACCENT + ';font-size:11px;';

    slider.addEventListener('input', () => {
      valSpan.textContent = Number(slider.value).toFixed(2);
      onChange(Number(slider.value));
    });

    wrap.appendChild(slider);
    wrap.appendChild(valSpan);
    controlsDiv.appendChild(wrap);
    return { wrap, slider, valSpan };
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
      'font-family:"JetBrains Mono",monospace'
    ].join(';');
    options.forEach(opt => {
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

  // attractor type selector
  const attractorSelect = createSelect(
    Object.keys(ATTRACTORS).map(k => ({ value: k, label: ATTRACTORS[k].name })),
    currentAttractorKey,
    val => {
      currentAttractorKey = val;
      // naye attractor ke defaults laga do
      params = { ...ATTRACTORS[val].params };
      rebuildSliders();
      resetAndRestart();
    }
  );

  // color scheme selector
  createSelect(
    Object.keys(COLOR_SCHEMES).map(k => ({ value: k, label: COLOR_SCHEMES[k].name })),
    colorSchemeKey,
    val => {
      colorSchemeKey = val;
      buildColorLUT();
      // density buffer wahi hai, sirf redraw karo
      renderDensityToCanvas();
    }
  );

  // randomize button — interesting random params generate karo
  createButton('Randomize', () => {
    const attractor = ATTRACTORS[currentAttractorKey];
    const newParams = {};
    const keys = Object.keys(attractor.ranges);
    keys.forEach(k => {
      const [lo, hi] = attractor.ranges[k];
      // random value range mein — slightly biased towards interesting ranges
      newParams[k] = lo + Math.random() * (hi - lo);
      // round to 2 decimal places — cleaner values
      newParams[k] = Math.round(newParams[k] * 100) / 100;
    });
    params = newParams;
    rebuildSliders();
    resetAndRestart();
  });

  // param sliders — dynamic based on current attractor
  let paramSliders = {};
  const slidersContainer = document.createElement('div');
  slidersContainer.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:8px',
    'align-items:center',
    'width:100%'
  ].join(';');
  container.appendChild(slidersContainer);

  // sliders rebuild karo jab attractor change ho
  function rebuildSliders() {
    // purane sliders hata do
    while (slidersContainer.firstChild) slidersContainer.removeChild(slidersContainer.firstChild);
    paramSliders = {};

    const attractor = ATTRACTORS[currentAttractorKey];
    const keys = Object.keys(attractor.ranges);

    keys.forEach(k => {
      const [lo, hi] = attractor.ranges[k];
      const wrap = document.createElement('div');
      wrap.style.cssText = [
        'display:flex',
        'align-items:center',
        'gap:6px',
        'font-family:"JetBrains Mono",monospace',
        'font-size:11px',
        'color:#888'
      ].join(';');

      const lbl = document.createElement('span');
      lbl.textContent = k;
      lbl.style.cssText = 'white-space:nowrap;min-width:12px;font-weight:bold;color:' + ACCENT + ';';
      wrap.appendChild(lbl);

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = lo;
      slider.max = hi;
      slider.value = params[k];
      slider.step = 0.01;
      slider.style.cssText = 'width:90px;accent-color:' + ACCENT + ';cursor:pointer;';

      const valSpan = document.createElement('span');
      valSpan.textContent = params[k].toFixed(2);
      valSpan.style.cssText = 'min-width:42px;text-align:right;color:' + ACCENT + ';font-size:11px;';

      slider.addEventListener('input', () => {
        const val = Number(slider.value);
        valSpan.textContent = val.toFixed(2);
        params[k] = val;
        resetAndRestart();
      });

      wrap.appendChild(slider);
      wrap.appendChild(valSpan);
      slidersContainer.appendChild(wrap);
      paramSliders[k] = { slider, valSpan };
    });

    updateInfo();
  }

  // --- Canvas sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    canvasW = w;
    canvasH = CANVAS_HEIGHT;

    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.height = CANVAS_HEIGHT + 'px';

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // density buffer bhi resize karo — canvas ke CSS size pe based
    initDensityBuffer();
    resetAndRestart();
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Color LUT banao — precomputed gradient for pixel coloring ---
  function buildColorLUT() {
    const scheme = COLOR_SCHEMES[colorSchemeKey];
    if (!scheme) return;
    colorLUT = new Uint8Array(LUT_SIZE * 3);

    for (let i = 0; i < LUT_SIZE; i++) {
      const t = i / (LUT_SIZE - 1);
      const rgb = interpolateScheme(scheme.stops, t);
      colorLUT[i * 3] = rgb[0];
      colorLUT[i * 3 + 1] = rgb[1];
      colorLUT[i * 3 + 2] = rgb[2];
    }
  }

  // linear interpolation between color stops — cosine smoothing for nice gradients
  function interpolateScheme(stops, t) {
    t = Math.max(0, Math.min(1, t));

    let i = 0;
    while (i < stops.length - 1 && stops[i + 1][0] < t) i++;
    if (i >= stops.length - 1) {
      return [stops[stops.length - 1][1], stops[stops.length - 1][2], stops[stops.length - 1][3]];
    }

    const s0 = stops[i];
    const s1 = stops[i + 1];
    const f = (t - s0[0]) / (s1[0] - s0[0]);
    // cosine interpolation — smoother than linear
    const smoothF = (1 - Math.cos(f * Math.PI)) / 2;

    return [
      Math.round(s0[1] + (s1[1] - s0[1]) * smoothF),
      Math.round(s0[2] + (s1[2] - s0[2]) * smoothF),
      Math.round(s0[3] + (s1[3] - s0[3]) * smoothF)
    ];
  }

  buildColorLUT();

  // --- Density buffer management ---
  function initDensityBuffer() {
    // CSS pixel dimensions use karo — DPR wala resolution overkill hai density ke liye
    bufferW = canvasW;
    bufferH = canvasH;
    if (bufferW <= 0 || bufferH <= 0) return;
    densityBuffer = new Uint32Array(bufferW * bufferH);
  }

  function clearDensityBuffer() {
    if (densityBuffer) densityBuffer.fill(0);
    maxDensity = 0;
    totalPointsRendered = 0;
    renderComplete = false;
  }

  // --- Bounds computation ---
  // pehle kuch points iterate karke dekho ki attractor ka extent kya hai
  // taaki canvas pe sahi se fit ho sake
  function computeBounds() {
    const attractor = ATTRACTORS[currentAttractorKey];
    let x = 0.1, y = 0.1;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    // 50K points iterate karke bounds estimate karo
    // pehle 1000 skip karo — transient hata do
    for (let i = 0; i < 1000; i++) {
      const next = attractor.iterate(x, y, params);
      x = next[0];
      y = next[1];
      // NaN/Infinity check — kuch params se attractor diverge kar sakta hai
      if (!isFinite(x) || !isFinite(y)) {
        x = 0.1;
        y = 0.1;
      }
    }

    for (let i = 0; i < 50000; i++) {
      const next = attractor.iterate(x, y, params);
      x = next[0];
      y = next[1];
      if (!isFinite(x) || !isFinite(y)) {
        x = 0.1;
        y = 0.1;
        continue;
      }
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    // thoda padding add karo — edges pe points clip na ho
    const padX = (maxX - minX) * 0.05;
    const padY = (maxY - minY) * 0.05;
    boundsMinX = minX - padX;
    boundsMaxX = maxX + padX;
    boundsMinY = minY - padY;
    boundsMaxY = maxY + padY;

    // aspect ratio fix karo — canvas ka aspect maintain ho
    const boundsW = boundsMaxX - boundsMinX;
    const boundsH = boundsMaxY - boundsMinY;
    const canvasAspect = bufferW / bufferH;
    const boundsAspect = boundsW / boundsH;

    if (boundsAspect > canvasAspect) {
      // bounds zyada wide hain — height badhao
      const newH = boundsW / canvasAspect;
      const diff = (newH - boundsH) / 2;
      boundsMinY -= diff;
      boundsMaxY += diff;
    } else {
      // bounds zyada tall hain — width badhao
      const newW = boundsH * canvasAspect;
      const diff = (newW - boundsW) / 2;
      boundsMinX -= diff;
      boundsMaxX += diff;
    }

    boundsComputed = true;

    // iteration state reset karo — bounds wale warmup points use karo as starting point
    iterX = x;
    iterY = y;
  }

  // --- Point iteration — density buffer mein accumulate karo ---
  function iteratePoints(count) {
    if (!densityBuffer || bufferW <= 0 || bufferH <= 0) return;

    const attractor = ATTRACTORS[currentAttractorKey];
    const scaleX = (bufferW - 1) / (boundsMaxX - boundsMinX);
    const scaleY = (bufferH - 1) / (boundsMaxY - boundsMinY);
    let x = iterX, y = iterY;
    let localMax = maxDensity;

    for (let i = 0; i < count; i++) {
      const next = attractor.iterate(x, y, params);
      x = next[0];
      y = next[1];

      // divergence check — kuch param combos se blow up ho sakta hai
      if (!isFinite(x) || !isFinite(y)) {
        x = 0.1 + Math.random() * 0.01;
        y = 0.1 + Math.random() * 0.01;
        continue;
      }

      // canvas coordinates mein map karo
      const px = Math.floor((x - boundsMinX) * scaleX);
      const py = Math.floor((y - boundsMinY) * scaleY);

      // bounds ke andar hai toh density increment karo
      if (px >= 0 && px < bufferW && py >= 0 && py < bufferH) {
        const idx = py * bufferW + px;
        densityBuffer[idx]++;
        if (densityBuffer[idx] > localMax) {
          localMax = densityBuffer[idx];
        }
      }
    }

    // state save karo next call ke liye
    iterX = x;
    iterY = y;
    maxDensity = localMax;
    totalPointsRendered += count;
  }

  // --- Density buffer ko canvas pe render karo ---
  function renderDensityToCanvas() {
    if (!densityBuffer || maxDensity === 0 || bufferW <= 0 || bufferH <= 0) {
      // kuch bhi nahi hai toh black canvas dikhao
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvasW, canvasH);
      return;
    }

    // ImageData banao — canvas ke CSS size pe (DPR handle ctx transform karega)
    const imageData = ctx.createImageData(bufferW, bufferH);
    const data = imageData.data;

    // log scale normalization — ye CRITICAL hai visual quality ke liye
    // linear scale mein high density areas sab white ho jaate hain aur low density invisible
    // log scale se contrast bahut accha aata hai
    const logMax = Math.log(maxDensity + 1);

    for (let i = 0; i < bufferW * bufferH; i++) {
      const density = densityBuffer[i];
      if (density === 0) {
        // background — pure black
        const idx = i * 4;
        data[idx] = 0;
        data[idx + 1] = 0;
        data[idx + 2] = 0;
        data[idx + 3] = 255;
        continue;
      }

      // log scale mapping — 0 to 1 range mein
      const t = Math.log(density + 1) / logMax;

      // LUT se color nikalo
      const lutIdx = Math.min(LUT_SIZE - 1, Math.floor(t * (LUT_SIZE - 1)));
      const ci = lutIdx * 3;
      const idx = i * 4;
      data[idx] = colorLUT[ci];
      data[idx + 1] = colorLUT[ci + 1];
      data[idx + 2] = colorLUT[ci + 2];
      data[idx + 3] = 255;
    }

    // canvas pe draw karo — temp canvas use karo scaling ke liye
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = bufferW;
    tmpCanvas.height = bufferH;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.putImageData(imageData, 0, 0);

    // main canvas pe scale karke draw karo
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false; // crisp pixels — density art mein smoothing nahi chahiye
    ctx.drawImage(tmpCanvas, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  }

  // --- Reset aur restart — jab params change ho ---
  function resetAndRestart() {
    clearDensityBuffer();
    boundsComputed = false;
    renderComplete = false;
    updateInfo();

    // agar visible hai toh restart karo
    if (isVisible && !animationId) {
      loop();
    }
  }

  // --- Info bar update ---
  function updateInfo() {
    // current params dikhao
    const keys = Object.keys(params);
    const paramStr = keys.map(k => k + '=' + params[k].toFixed(2)).join('  ');
    paramsSpan.textContent = ATTRACTORS[currentAttractorKey].name + '  |  ' + paramStr;

    // points count dikhao
    if (totalPointsRendered >= 1000000) {
      pointsSpan.textContent = 'Points: ' + (totalPointsRendered / 1000000).toFixed(1) + 'M';
    } else if (totalPointsRendered >= 1000) {
      pointsSpan.textContent = 'Points: ' + (totalPointsRendered / 1000).toFixed(0) + 'K';
    } else {
      pointsSpan.textContent = 'Points: ' + totalPointsRendered;
    }

    if (renderComplete) {
      pointsSpan.textContent += ' (complete)';
    }
  }

  // --- Main animation loop ---
  function loop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    // agar render complete hai toh bas ruk jao — kaam ho gaya
    if (renderComplete) {
      animationId = null;
      return;
    }

    // pehle bounds compute karo agar nahi kiye
    if (!boundsComputed) {
      computeBounds();
    }

    // kitne points add karne hain is frame mein
    const pointsThisFrame = totalPointsRendered === 0 ? INITIAL_POINTS : POINTS_PER_FRAME;
    const remaining = MAX_TOTAL_POINTS - totalPointsRendered;
    const toRender = Math.min(pointsThisFrame, remaining);

    if (toRender > 0) {
      iteratePoints(toRender);
    }

    // canvas pe render karo
    renderDensityToCanvas();
    updateInfo();

    // check karo ki sab ho gaya ya nahi
    if (totalPointsRendered >= MAX_TOTAL_POINTS) {
      renderComplete = true;
      updateInfo();
      animationId = null;
      return;
    }

    animationId = requestAnimationFrame(loop);
  }

  // --- Build initial sliders ---
  rebuildSliders();

  // --- IntersectionObserver — sirf jab dikhe tab render karo ---
  const observer = new IntersectionObserver(
    ([entry]) => {
      isVisible = entry.isIntersecting;
      if (isVisible && !animationId && !renderComplete) {
        loop();
      } else if (!isVisible && animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    },
    { threshold: 0.1 }
  );

  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });

  // tab switch pe pause/resume
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView && !animationId && !renderComplete) {
        isVisible = true;
        loop();
      }
    }
  });
}
