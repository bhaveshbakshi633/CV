// ============================================================
// Fourier Series Drawing Tool — Draw karo, DFT lagao, epicycles dekho
// User freehand draw karta hai, fir rotating circles us shape ko trace karte hain
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, Fourier chalao
export function initFourierDraw() {
  const container = document.getElementById('fourierDrawContainer');
  if (!container) {
    console.warn('fourierDrawContainer nahi mila bhai, Fourier draw demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const RESAMPLE_COUNT = 200; // drawn path ko itne points mein resample karenge
  const TWO_PI = Math.PI * 2;
  const ACCENT = '#22d3ee'; // cyan accent color
  const ACCENT_RGB = '34,211,238';

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;

  // mode — DRAW ya PLAY
  let mode = 'DRAW'; // 'DRAW' | 'PLAY'

  // drawing state — raw points jo user draw karta hai
  let rawPoints = []; // [{x, y}]
  let isDrawing = false;

  // DFT state — computed coefficients
  let dftCoeffs = []; // [{re, im, freq, amp, phase}] — sorted by amplitude
  let numCircles = 0; // kitne circles dikhane hain — slider se control
  let maxCircles = 0; // total available circles

  // animation state
  let time = 0; // 0 to TWO_PI — ek full cycle
  let speed = 1.0; // speed multiplier
  let tracedPath = []; // reconstructed path points [{x, y}]
  let animationId = null;
  let isVisible = false;

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — drawing aur epicycles dono yahan
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);

  // controls section — sliders + buttons
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
      'padding:6px 14px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      "font-family:'JetBrains Mono',monospace",
      'transition:all 0.2s ease',
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

  // --- Helper: slider banao ---
  function createSlider(label, min, max, step, defaultVal, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = "color:#b0b0b0;font-size:12px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
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
    valueEl.style.cssText = "color:#b0b0b0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:28px;";
    valueEl.textContent = defaultVal;
    wrapper.appendChild(valueEl);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = Number.isInteger(val) ? val : val.toFixed(1);
      onChange(val);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, valueEl, wrapper };
  }

  // --- Clear / Redraw button ---
  const clearBtn = createButton('Clear / Redraw', () => {
    // DRAW mode pe wapas jao
    mode = 'DRAW';
    rawPoints = [];
    dftCoeffs = [];
    tracedPath = [];
    time = 0;
    numCircles = 0;
    maxCircles = 0;
    // sliders hide karo
    circleSliderObj.wrapper.style.display = 'none';
    canvas.style.cursor = 'crosshair';
  });

  // --- Preset buttons ---
  const presetSep = document.createElement('span');
  presetSep.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.2);font-size:14px;';
  presetSep.textContent = '|';
  controlsDiv.appendChild(presetSep);

  const presetLabel = document.createElement('span');
  presetLabel.style.cssText = "color:#b0b0b0;font-size:12px;font-family:'JetBrains Mono',monospace;";
  presetLabel.textContent = 'Presets:';
  controlsDiv.appendChild(presetLabel);

  // preset shapes define karo — points generate karte hain
  function generatePresetCircle() {
    const pts = [];
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const r = Math.min(canvasW, canvasH) * 0.3;
    for (let i = 0; i < 200; i++) {
      const angle = (i / 200) * TWO_PI;
      pts.push({ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
    }
    return pts;
  }

  function generatePresetSquare() {
    const pts = [];
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const half = Math.min(canvasW, canvasH) * 0.25;
    const perSide = 50; // har side pe 50 points
    // top side — left to right
    for (let i = 0; i < perSide; i++) {
      pts.push({ x: cx - half + (2 * half * i) / perSide, y: cy - half });
    }
    // right side — top to bottom
    for (let i = 0; i < perSide; i++) {
      pts.push({ x: cx + half, y: cy - half + (2 * half * i) / perSide });
    }
    // bottom side — right to left
    for (let i = 0; i < perSide; i++) {
      pts.push({ x: cx + half - (2 * half * i) / perSide, y: cy + half });
    }
    // left side — bottom to top
    for (let i = 0; i < perSide; i++) {
      pts.push({ x: cx - half, y: cy + half - (2 * half * i) / perSide });
    }
    return pts;
  }

  function generatePresetStar() {
    const pts = [];
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const outerR = Math.min(canvasW, canvasH) * 0.3;
    const innerR = outerR * 0.4;
    const points = 5; // 5-pointed star
    const totalVertices = points * 2;
    // pehle vertices banao
    const vertices = [];
    for (let i = 0; i < totalVertices; i++) {
      // -PI/2 se shuru taaki top point upar ho
      const angle = (i / totalVertices) * TWO_PI - Math.PI / 2;
      const r = i % 2 === 0 ? outerR : innerR;
      vertices.push({ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
    }
    // vertices ke beech interpolate karo — smooth path ke liye
    const ptsPerEdge = 20;
    for (let i = 0; i < totalVertices; i++) {
      const from = vertices[i];
      const to = vertices[(i + 1) % totalVertices];
      for (let j = 0; j < ptsPerEdge; j++) {
        const t = j / ptsPerEdge;
        pts.push({
          x: from.x + (to.x - from.x) * t,
          y: from.y + (to.y - from.y) * t,
        });
      }
    }
    return pts;
  }

  function generatePresetHeart() {
    const pts = [];
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const scale = Math.min(canvasW, canvasH) * 0.018;
    for (let i = 0; i < 200; i++) {
      const t = (i / 200) * TWO_PI;
      // parametric heart curve — classic formula
      const x = 16 * Math.pow(Math.sin(t), 3);
      const y = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t));
      pts.push({ x: cx + x * scale, y: cy + y * scale });
    }
    return pts;
  }

  function loadPreset(generator) {
    rawPoints = generator();
    startPlayMode();
  }

  createButton('Circle', () => loadPreset(generatePresetCircle));
  createButton('Square', () => loadPreset(generatePresetSquare));
  createButton('Star', () => loadPreset(generatePresetStar));
  createButton('Heart', () => loadPreset(generatePresetHeart));

  // --- Separator ---
  const sliderSep = document.createElement('span');
  sliderSep.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.2);font-size:14px;';
  sliderSep.textContent = '|';
  controlsDiv.appendChild(sliderSep);

  // --- Circles slider — kitne frequency components use karne hain ---
  const circleSliderObj = createSlider('Circles:', 1, RESAMPLE_COUNT, 1, RESAMPLE_COUNT, (val) => {
    numCircles = Math.round(val);
  });
  // initially hidden — jab tak draw nahi kiya tab tak dikhaane ka matlab nahi
  circleSliderObj.wrapper.style.display = 'none';

  // --- Speed slider ---
  const speedSliderObj = createSlider('Speed:', 0.1, 3.0, 0.1, 1.0, (val) => {
    speed = val;
  });

  // --- Canvas sizing — retina DPR handle ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = containerWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // ============================================================
  // MATH: Resample path to evenly spaced points
  // drawn path mein points bahut uneven hote hain — fast movement = sparse,
  // slow movement = dense. DFT ke liye even spacing chahiye.
  // ============================================================
  function resamplePath(points, targetCount) {
    if (points.length < 2) return points;

    // pehle cumulative distances nikal — har point tak ka total distance
    const cumDist = [0];
    for (let i = 1; i < points.length; i++) {
      const dx = points[i].x - points[i - 1].x;
      const dy = points[i].y - points[i - 1].y;
      cumDist.push(cumDist[i - 1] + Math.sqrt(dx * dx + dy * dy));
    }

    const totalLength = cumDist[cumDist.length - 1];
    if (totalLength === 0) return points.slice(0, targetCount);

    // har resampled point ko equal arc length pe rakh
    const resampled = [];
    let segIdx = 0;

    for (let i = 0; i < targetCount; i++) {
      const targetDist = (i / targetCount) * totalLength;

      // sahi segment dhundho — jismein targetDist aata hai
      while (segIdx < points.length - 2 && cumDist[segIdx + 1] < targetDist) {
        segIdx++;
      }

      // segment ke andar interpolate karo
      const segStart = cumDist[segIdx];
      const segEnd = cumDist[segIdx + 1];
      const segLen = segEnd - segStart;
      const t = segLen > 0 ? (targetDist - segStart) / segLen : 0;

      resampled.push({
        x: points[segIdx].x + (points[segIdx + 1].x - points[segIdx].x) * t,
        y: points[segIdx].y + (points[segIdx + 1].y - points[segIdx].y) * t,
      });
    }

    return resampled;
  }

  // ============================================================
  // MATH: Discrete Fourier Transform
  // Complex DFT — input points ko complex numbers maan ke transform karo
  // x = real part, y = imaginary part
  // ============================================================
  function computeDFT(points) {
    const N = points.length;
    const coeffs = [];

    for (let k = 0; k < N; k++) {
      let re = 0;
      let im = 0;

      for (let n = 0; n < N; n++) {
        // e^(-i * 2pi * k * n / N) = cos(...) - i*sin(...)
        const angle = (TWO_PI * k * n) / N;
        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);

        // complex multiplication: (x + iy) * (cos - i*sin)
        // = x*cos + y*sin + i*(y*cos - x*sin)
        re += points[n].x * cosA + points[n].y * sinA;
        im += points[n].y * cosA - points[n].x * sinA;
      }

      // normalize by N
      re /= N;
      im /= N;

      const amp = Math.sqrt(re * re + im * im);
      const phase = Math.atan2(im, re);

      coeffs.push({
        re: re,
        im: im,
        freq: k,
        amp: amp,
        phase: phase,
      });
    }

    // amplitude ke hisaab se sort karo — sabse bada pehle
    // ye visually accha lagta hai — important frequencies pehle dikhti hain
    coeffs.sort((a, b) => b.amp - a.amp);

    return coeffs;
  }

  // ============================================================
  // EPICYCLE: Given time t, epicycles trace karke final point nikal
  // Har coefficient ek rotating circle hai
  // Returns: {x, y, circles: [{cx, cy, radius}]}
  // ============================================================
  function computeEpicycles(t, coefficients, count) {
    let x = 0;
    let y = 0;
    const circles = [];
    const n = Math.min(count, coefficients.length);

    for (let i = 0; i < n; i++) {
      const c = coefficients[i];
      const prevX = x;
      const prevY = y;

      // har circle ka angle = freq * t + initial phase
      const angle = c.freq * t + c.phase;

      // naya point — circle ka center + radius * rotation
      x += c.amp * Math.cos(angle);
      y += c.amp * Math.sin(angle);

      circles.push({
        cx: prevX,
        cy: prevY,
        radius: c.amp,
        endX: x,
        endY: y,
      });
    }

    return { x, y, circles };
  }

  // ============================================================
  // DRAWING MODE: Mouse/touch se path capture karo
  // ============================================================
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  canvas.addEventListener('mousedown', (e) => {
    if (mode !== 'DRAW') return;
    isDrawing = true;
    rawPoints = [];
    const pos = getCanvasPos(e);
    rawPoints.push(pos);
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing || mode !== 'DRAW') return;
    const pos = getCanvasPos(e);
    rawPoints.push(pos);
  });

  canvas.addEventListener('mouseup', () => {
    if (!isDrawing || mode !== 'DRAW') return;
    isDrawing = false;
    // agar kuch draw kiya hai toh play mode shuru karo
    if (rawPoints.length > 10) {
      startPlayMode();
    }
  });

  canvas.addEventListener('mouseleave', () => {
    if (!isDrawing || mode !== 'DRAW') return;
    isDrawing = false;
    if (rawPoints.length > 10) {
      startPlayMode();
    }
  });

  // touch support — mobile pe bhi kaam kare
  canvas.addEventListener('touchstart', (e) => {
    if (mode !== 'DRAW') return;
    e.preventDefault();
    isDrawing = true;
    rawPoints = [];
    const pos = getCanvasPos(e);
    rawPoints.push(pos);
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    if (!isDrawing || mode !== 'DRAW') return;
    e.preventDefault();
    const pos = getCanvasPos(e);
    rawPoints.push(pos);
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    if (!isDrawing || mode !== 'DRAW') return;
    e.preventDefault();
    isDrawing = false;
    if (rawPoints.length > 10) {
      startPlayMode();
    }
  }, { passive: false });

  // ============================================================
  // PLAY MODE: DFT compute karo aur animation shuru karo
  // ============================================================
  function startPlayMode() {
    // path ko resample karo — evenly spaced points
    const resampled = resamplePath(rawPoints, RESAMPLE_COUNT);

    // center of mass nikal ke path ko center karo
    // taaki epicycles canvas ke center se shuru hon
    let cx = 0, cy = 0;
    for (const p of resampled) {
      cx += p.x;
      cy += p.y;
    }
    cx /= resampled.length;
    cy /= resampled.length;

    // center offset store karo — rendering ke time add karenge
    const centered = resampled.map(p => ({
      x: p.x - cx,
      y: p.y - cy,
    }));

    // DFT compute karo
    dftCoeffs = computeDFT(centered);
    maxCircles = dftCoeffs.length;
    numCircles = maxCircles;

    // circle slider update karo
    circleSliderObj.slider.max = maxCircles;
    circleSliderObj.slider.value = maxCircles;
    circleSliderObj.valueEl.textContent = maxCircles;
    circleSliderObj.wrapper.style.display = 'flex';

    // animation state reset
    time = 0;
    tracedPath = [];
    mode = 'PLAY';
    canvas.style.cursor = 'default';

    // offset store karo — epicycles ko canvas ke center mein draw karenge
    // nahi, original center mein rakhte hain taaki shape sahi jagah dikhe
    epicycleOffsetX = cx;
    epicycleOffsetY = cy;
  }

  // epicycle center offset — jahan pe shape originally drawn thi
  let epicycleOffsetX = 0;
  let epicycleOffsetY = 0;

  // ============================================================
  // RENDERING: Canvas pe sab draw karo
  // ============================================================
  function drawScene() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (mode === 'DRAW') {
      drawDrawMode(ctx);
    } else {
      drawPlayMode(ctx);
    }
  }

  // --- DRAW mode rendering ---
  function drawDrawMode(ctx) {
    // "Draw something!" text — jab kuch nahi draw kiya hai
    if (rawPoints.length === 0 && !isDrawing) {
      ctx.font = "16px 'JetBrains Mono',monospace";
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
      ctx.textAlign = 'center';
      ctx.fillText('Draw something!', canvasW / 2, canvasH / 2 - 10);

      ctx.font = "11px 'JetBrains Mono',monospace";
      ctx.fillStyle = 'rgba(176,176,176,0.3)';
      ctx.fillText('freehand draw with mouse or touch', canvasW / 2, canvasH / 2 + 15);
      ctx.fillText('or try a preset shape below', canvasW / 2, canvasH / 2 + 32);
      return;
    }

    // user ka drawn path dikhao — real-time jab draw kar raha hai
    if (rawPoints.length > 1) {
      ctx.beginPath();
      ctx.moveTo(rawPoints[0].x, rawPoints[0].y);
      for (let i = 1; i < rawPoints.length; i++) {
        ctx.lineTo(rawPoints[i].x, rawPoints[i].y);
      }
      ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.6)';
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();
    }

    // drawing indicator
    if (isDrawing) {
      ctx.font = "10px 'JetBrains Mono',monospace";
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
      ctx.textAlign = 'right';
      ctx.fillText('drawing... (' + rawPoints.length + ' pts)', canvasW - 12, 18);
    }
  }

  // --- PLAY mode rendering — epicycles + traced path ---
  function drawPlayMode(ctx) {
    if (dftCoeffs.length === 0) return;

    // epicycles compute karo current time pe
    const result = computeEpicycles(time, dftCoeffs, numCircles);

    // offset add karo — original drawing position pe render karo
    const offsetX = epicycleOffsetX;
    const offsetY = epicycleOffsetY;

    // --- Traced path draw karo pehle (background mein) ---
    if (tracedPath.length > 1) {
      ctx.beginPath();
      ctx.moveTo(tracedPath[0].x + offsetX, tracedPath[0].y + offsetY);
      for (let i = 1; i < tracedPath.length; i++) {
        ctx.lineTo(tracedPath[i].x + offsetX, tracedPath[i].y + offsetY);
      }

      // glow effect — bright cyan traced path
      ctx.shadowColor = 'rgba(' + ACCENT_RGB + ',0.4)';
      ctx.shadowBlur = 8;
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
    }

    // --- Circles draw karo (epicycles) ---
    for (let i = 0; i < result.circles.length; i++) {
      const c = result.circles[i];

      // bahut chhote circles skip karo — dikhte bhi nahi
      if (c.radius < 0.5) continue;

      // circle ka border — semi-transparent
      ctx.beginPath();
      ctx.arc(c.cx + offsetX, c.cy + offsetY, c.radius, 0, TWO_PI);
      ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',' + Math.max(0.04, 0.15 - i * 0.001) + ')';
      ctx.lineWidth = 1;
      ctx.stroke();

      // semi-transparent fill — subtle
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',' + Math.max(0.01, 0.04 - i * 0.0005) + ')';
      ctx.fill();

      // center se end tak line — connecting rod
      ctx.beginPath();
      ctx.moveTo(c.cx + offsetX, c.cy + offsetY);
      ctx.lineTo(c.endX + offsetX, c.endY + offsetY);
      ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',' + Math.max(0.08, 0.3 - i * 0.003) + ')';
      ctx.lineWidth = 1;
      ctx.stroke();

      // center dot — chhota sa
      if (i < 20) {
        ctx.beginPath();
        ctx.arc(c.cx + offsetX, c.cy + offsetY, 1.5, 0, TWO_PI);
        ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',' + Math.max(0.1, 0.4 - i * 0.02) + ')';
        ctx.fill();
      }
    }

    // --- Tip point — jahan pe pen hai abhi ---
    const tipX = result.x + offsetX;
    const tipY = result.y + offsetY;

    // bright dot at tip
    ctx.beginPath();
    ctx.arc(tipX, tipY, 4, 0, TWO_PI);
    ctx.fillStyle = ACCENT;
    ctx.shadowColor = ACCENT;
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // tip se last circle tak line — optional, looks nice
    if (result.circles.length > 0) {
      const lastCircle = result.circles[result.circles.length - 1];
      if (lastCircle.radius >= 0.5) {
        ctx.beginPath();
        ctx.moveTo(lastCircle.endX + offsetX, lastCircle.endY + offsetY);
        ctx.lineTo(tipX, tipY);
        ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // --- Info text ---
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.textAlign = 'left';
    ctx.fillText('FOURIER EPICYCLES', 10, 18);

    ctx.textAlign = 'right';
    ctx.fillText(numCircles + '/' + maxCircles + ' circles', canvasW - 12, 18);

    // progress bar — kitna cycle complete hua
    const progress = time / TWO_PI;
    const barW = 60;
    const barH = 3;
    const barX = canvasW - 12 - barW;
    const barY = 26;
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.1)';
    ctx.fillRect(barX, barY, barW, barH);
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
    ctx.fillRect(barX, barY, barW * progress, barH);
  }

  // ============================================================
  // ANIMATION LOOP
  // ============================================================
  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    if (mode === 'PLAY' && dftCoeffs.length > 0) {
      // time advance karo
      const dt = (TWO_PI / dftCoeffs.length) * speed;
      time += dt;

      // current point trace karo
      const result = computeEpicycles(time, dftCoeffs, numCircles);
      tracedPath.push({ x: result.x, y: result.y });

      // ek full cycle ho gaya toh traced path reset karo
      // nahi toh purane aur naye points overlap karenge
      if (time >= TWO_PI) {
        time = 0;
        tracedPath = [];
      }
    }

    drawScene();
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf jab dikhe tab animate karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
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

  // tab switch pe pause — battery bachao
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
