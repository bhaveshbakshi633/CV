// ============================================================
// Bezier Curve Editor — de Casteljau algorithm ka interactive demo
// Control points drag karo, construction lines dekho, samjho Bezier kaise banta hai
// Supports 2-8 control points, animated t parameter, preset curves
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, curves draw karo
export function initBezier() {
  const container = document.getElementById('bezierContainer');
  if (!container) {
    console.warn('bezierContainer nahi mila bhai, Bezier demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 380;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';
  const CURVE_SAMPLES = 200; // itne points se curve draw hoga — smooth dikhega
  const GRID_SPACING = 40; // grid lines ka gap
  const CONTROL_RADIUS = 8; // control point circle ka radius
  const HIT_RADIUS = 18; // click detection ke liye thoda bada area
  const ANIM_SPEED = 0.004; // t parameter ki animation speed per frame

  // de Casteljau construction ke har level ka color — rainbow vibes
  const LEVEL_COLORS = [
    'rgba(250,204,21,0.7)',   // level 0 — yellow (control polygon ke beech)
    'rgba(52,211,153,0.65)',  // level 1 — green
    'rgba(96,165,250,0.6)',   // level 2 — blue
    'rgba(244,114,182,0.55)', // level 3 — pink
    'rgba(251,146,60,0.5)',   // level 4 — orange
    'rgba(167,139,250,0.5)',  // level 5 — purple (accent match)
  ];

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;

  // control points — default cubic bezier (4 points, acchi shape)
  let controlPoints = [];

  // t parameter — 0 se 1 ke beech, construction yahi pe dikhegi
  let tParam = 0.5;

  // animation state — t ko 0 se 1 aur wapas sweep karta hai
  let animating = false;
  let animDirection = 1; // +1 = forward, -1 = backward

  // drag state
  let draggingIdx = -1; // konsa point pakda hai, -1 = koi nahi
  let isDragging = false;

  // visibility aur animation loop
  let animationId = null;
  let isVisible = false;

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:default',
    'background:rgba(2,2,8,0.5)',
    'touch-action:none',
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

  // --- Helper: slider banao ---
  function createSlider(label, min, max, step, value, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#b0b0b0;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.style.cssText = 'width:90px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
    wrapper.appendChild(slider);

    const valSpan = document.createElement('span');
    valSpan.style.cssText = "color:#e0e0e0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:30px;";
    valSpan.textContent = Number(value).toFixed(step < 0.01 ? 3 : (step < 1 ? 2 : 0));
    wrapper.appendChild(valSpan);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      const decimals = step < 0.01 ? 3 : (step < 1 ? 2 : 0);
      valSpan.textContent = v.toFixed(decimals);
      onChange(v);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, valSpan };
  }

  // --- Helper: button banao ---
  function createButton(text, onClick) {
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
      "font-family:'JetBrains Mono',monospace",
      'transition:all 0.2s ease',
      'white-space:nowrap',
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

  // --- Helper: dropdown (select) banao presets ke liye ---
  function createSelect(label, options, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#b0b0b0;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const sel = document.createElement('select');
    sel.style.cssText = [
      'padding:4px 8px',
      'font-size:11px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      "font-family:'JetBrains Mono',monospace",
    ].join(';');

    options.forEach((opt) => {
      const o = document.createElement('option');
      o.value = opt.value;
      o.textContent = opt.label;
      o.style.cssText = 'background:#1a1a2e;color:#b0b0b0;';
      sel.appendChild(o);
    });

    sel.addEventListener('change', () => { onChange(sel.value); });
    wrapper.appendChild(sel);
    controlsDiv.appendChild(wrapper);
    return sel;
  }

  // --- Controls banao ---

  // t parameter slider — de Casteljau construction yahi pe dikhti hai
  const tSlider = createSlider('t', 0, 1, 0.01, tParam, (v) => {
    tParam = v;
  });

  // animate toggle button
  let animBtn = null;
  animBtn = createButton('Animate', () => {
    animating = !animating;
    animBtn.textContent = animating ? 'Stop' : 'Animate';
    animBtn.style.background = animating
      ? 'rgba(' + ACCENT_RGB + ',0.3)'
      : 'rgba(' + ACCENT_RGB + ',0.1)';
  });

  // add control point
  createButton('+ Point', () => {
    if (controlPoints.length >= 8) return; // max 8 points
    // naya point last do points ke beech mein daalo
    const n = controlPoints.length;
    if (n < 2) {
      // agar somehow 1 ya 0 points hain, center mein daal do
      controlPoints.push({ x: canvasW / 2, y: canvasH / 2 });
    } else {
      // last do points ka midpoint
      const p1 = controlPoints[n - 2];
      const p2 = controlPoints[n - 1];
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      // naya point second-to-last position pe insert karo
      controlPoints.splice(n - 1, 0, mid);
    }
    updateInfoText();
  });

  // remove last point
  createButton('- Point', () => {
    if (controlPoints.length <= 2) return; // minimum 2 points chahiye
    controlPoints.pop();
    updateInfoText();
  });

  // reset button — default cubic pe le aao
  createButton('Reset', () => {
    setDefaultPoints();
    tParam = 0.5;
    tSlider.slider.value = '0.5';
    tSlider.valSpan.textContent = '0.50';
    animating = false;
    animDirection = 1;
    animBtn.textContent = 'Animate';
    animBtn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
    updateInfoText();
  });

  // preset curves dropdown
  createSelect('Preset', [
    { value: 'default', label: 'Cubic' },
    { value: 'scurve', label: 'S-Curve' },
    { value: 'loop', label: 'Loop' },
    { value: 'heart', label: 'Heart' },
    { value: 'wave', label: 'Wave' },
  ], (val) => {
    applyPreset(val);
    updateInfoText();
  });

  // info text — kitne points hain
  const infoSpan = document.createElement('span');
  infoSpan.style.cssText = "color:#6b6b6b;font-size:10px;font-family:'JetBrains Mono',monospace;margin-left:auto;";
  controlsDiv.appendChild(infoSpan);

  function updateInfoText() {
    const n = controlPoints.length;
    const names = ['', 'linear', 'linear', 'quadratic', 'cubic', 'quartic', 'quintic', 'sextic', 'septic'];
    const name = names[n] || ('degree ' + (n - 1));
    infoSpan.textContent = n + ' pts \u2022 ' + name;
  }

  // --- Canvas sizing — DPR aware ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    const oldW = canvasW;
    const oldH = canvasH;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = containerWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // agar pehle se points hain toh proportionally scale karo
    if (oldW > 0 && oldH > 0 && controlPoints.length > 0) {
      const scaleX = canvasW / oldW;
      const scaleY = canvasH / oldH;
      for (let i = 0; i < controlPoints.length; i++) {
        controlPoints[i].x *= scaleX;
        controlPoints[i].y *= scaleY;
      }
    }
  }

  // --- Default control points — acchi cubic bezier shape ---
  function setDefaultPoints() {
    const padX = canvasW * 0.12;
    const padY = canvasH * 0.15;
    controlPoints = [
      { x: padX, y: canvasH - padY },
      { x: canvasW * 0.33, y: padY },
      { x: canvasW * 0.67, y: canvasH - padY },
      { x: canvasW - padX, y: padY },
    ];
  }

  // --- Preset curves — har preset mein normalized (0-1) coordinates, fir canvas pe scale ---
  function applyPreset(name) {
    const padX = canvasW * 0.08;
    const padY = canvasH * 0.08;
    const w = canvasW - 2 * padX;
    const h = canvasH - 2 * padY;

    // helper — normalized coords (0-1) se canvas coords mein convert
    function pt(nx, ny) {
      return { x: padX + nx * w, y: padY + ny * h };
    }

    switch (name) {
      case 'scurve':
        // S-shape — 4 points, classic ease-in-out wali shape
        controlPoints = [
          pt(0, 0.85),
          pt(0.35, 0.85),
          pt(0.65, 0.15),
          pt(1, 0.15),
        ];
        break;

      case 'loop':
        // loop — 5 points jo ek loop banate hain, self-intersecting curve
        controlPoints = [
          pt(0.1, 0.7),
          pt(0.9, 0.1),
          pt(0.1, 0.1),
          pt(0.9, 0.7),
          pt(0.5, 0.9),
        ];
        break;

      case 'heart':
        // heart shape — 6 points
        controlPoints = [
          pt(0.5, 0.9),
          pt(0.1, 0.55),
          pt(0.05, 0.2),
          pt(0.5, 0.35),
          pt(0.95, 0.2),
          pt(0.9, 0.55),
        ];
        break;

      case 'wave':
        // wave — 7 points, sinusoidal-ish
        controlPoints = [
          pt(0.0, 0.5),
          pt(0.15, 0.1),
          pt(0.3, 0.9),
          pt(0.5, 0.1),
          pt(0.7, 0.9),
          pt(0.85, 0.1),
          pt(1.0, 0.5),
        ];
        break;

      default:
        // default cubic
        setDefaultPoints();
        break;
    }
  }

  // --- Canvas size set karo aur default points banao ---
  resizeCanvas();
  setDefaultPoints();
  updateInfoText();
  window.addEventListener('resize', () => {
    resizeCanvas();
    updateInfoText();
  });

  // ============================================================
  // MATH: de Casteljau's algorithm — Bezier curve ka dil
  // Recursively do points ke beech lerp karta hai
  // Har level pe ek kam point hota hai, jab tak ek point na reh jaaye
  // ============================================================

  // linear interpolation — do points ke beech t pe position
  function lerp(a, b, t) {
    return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
  }

  // de Casteljau — ek specific t pe saare intermediate levels calculate karo
  // returns: array of levels, har level mein us stage ke points
  // level 0 = original control points
  // level 1 = adjacent pairs ka lerp
  // last level = single point (curve pe)
  function deCasteljau(points, t) {
    const levels = [points.slice()]; // level 0 copy karo

    let current = points.slice();
    while (current.length > 1) {
      const next = [];
      for (let i = 0; i < current.length - 1; i++) {
        next.push(lerp(current[i], current[i + 1], t));
      }
      levels.push(next);
      current = next;
    }

    return levels;
  }

  // curve ko sample karo — bahut saare t values pe evaluate karke points banao
  function sampleCurve(points, numSamples) {
    const samples = [];
    for (let i = 0; i <= numSamples; i++) {
      const t = i / numSamples;
      const levels = deCasteljau(points, t);
      const lastLevel = levels[levels.length - 1];
      samples.push(lastLevel[0]);
    }
    return samples;
  }

  // ============================================================
  // DRAWING — grid, curve, construction, control points
  // ============================================================

  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // 1. subtle grid background
    drawGrid();

    // 2. control polygon — dotted lines connecting control points
    drawControlPolygon();

    // 3. final bezier curve — smooth solid line
    drawBezierCurve();

    // 4. de Casteljau construction at current t — educational visualization
    drawConstruction();

    // 5. control points — draggable circles, sabse upar
    drawControlPoints();

    // 6. info overlay — corner mein
    drawOverlay();
  }

  // --- Subtle grid background ---
  function drawGrid() {
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;

    // vertical lines
    for (let x = GRID_SPACING; x < canvasW; x += GRID_SPACING) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvasH);
      ctx.stroke();
    }

    // horizontal lines
    for (let y = GRID_SPACING; y < canvasH; y += GRID_SPACING) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasW, y);
      ctx.stroke();
    }
  }

  // --- Control polygon — dotted lines ---
  function drawControlPolygon() {
    if (controlPoints.length < 2) return;

    ctx.beginPath();
    ctx.moveTo(controlPoints[0].x, controlPoints[0].y);
    for (let i = 1; i < controlPoints.length; i++) {
      ctx.lineTo(controlPoints[i].x, controlPoints[i].y);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // --- Bezier curve — smooth solid line ---
  function drawBezierCurve() {
    if (controlPoints.length < 2) return;

    const samples = sampleCurve(controlPoints, CURVE_SAMPLES);

    ctx.beginPath();
    ctx.moveTo(samples[0].x, samples[0].y);
    for (let i = 1; i < samples.length; i++) {
      ctx.lineTo(samples[i].x, samples[i].y);
    }

    // bright accent color curve — thodi glow bhi
    ctx.shadowColor = ACCENT;
    ctx.shadowBlur = 6;
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }

  // --- De Casteljau construction visualization — KEY FEATURE ---
  // har level ke beech ki lines aur points dikhao
  function drawConstruction() {
    if (controlPoints.length < 2) return;

    const levels = deCasteljau(controlPoints, tParam);

    // level 1 se shuru karo (level 0 toh control polygon hai, wo already draw ho gaya)
    for (let lvl = 1; lvl < levels.length; lvl++) {
      const points = levels[lvl];
      const colorIdx = Math.min(lvl - 1, LEVEL_COLORS.length - 1);
      const color = LEVEL_COLORS[colorIdx];

      // is level ke points ke beech lines draw karo
      if (points.length > 1) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        // line width reduce hoti jaaye har level pe
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(0.8, 2 - lvl * 0.3);
        ctx.stroke();
      }

      // is level ke points draw karo — chhote dots
      const dotRadius = Math.max(2.5, 4.5 - lvl * 0.5);
      for (let i = 0; i < points.length; i++) {
        ctx.beginPath();
        ctx.arc(points[i].x, points[i].y, dotRadius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }

    // final point — curve pe wala, bright aur bada, glow ke saath
    const finalLevel = levels[levels.length - 1];
    if (finalLevel.length === 1) {
      const fp = finalLevel[0];

      // glow
      ctx.shadowColor = ACCENT;
      ctx.shadowBlur = 14;

      // outer ring
      ctx.beginPath();
      ctx.arc(fp.x, fp.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = ACCENT;
      ctx.fill();

      // bright inner core
      ctx.beginPath();
      ctx.arc(fp.x, fp.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();

      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;

      // t value label — final point ke paas
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.8)';
      ctx.textAlign = 'left';
      ctx.fillText('t=' + tParam.toFixed(2), fp.x + 12, fp.y - 8);
    }
  }

  // --- Control points — draggable circles ---
  function drawControlPoints() {
    for (let i = 0; i < controlPoints.length; i++) {
      const p = controlPoints[i];
      const isActive = (draggingIdx === i);

      // outer circle — border
      ctx.beginPath();
      ctx.arc(p.x, p.y, CONTROL_RADIUS, 0, Math.PI * 2);

      if (isActive) {
        // dragging — bright glow
        ctx.shadowColor = ACCENT;
        ctx.shadowBlur = 12;
        ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.9)';
      } else {
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.6)';
      }
      ctx.fill();

      // border ring
      ctx.beginPath();
      ctx.arc(p.x, p.y, CONTROL_RADIUS, 0, Math.PI * 2);
      ctx.strokeStyle = isActive ? '#ffffff' : 'rgba(255,255,255,0.5)';
      ctx.lineWidth = isActive ? 2 : 1.5;
      ctx.stroke();

      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;

      // point number label — chhota index dikhao
      ctx.font = "bold 9px 'JetBrains Mono', monospace";
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('P' + i, p.x, p.y);
      ctx.textBaseline = 'alphabetic';
    }
  }

  // --- Overlay info — degree aur helpful text ---
  function drawOverlay() {
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.25)';

    const n = controlPoints.length;
    ctx.fillText('degree ' + (n - 1) + ' \u2022 ' + n + ' points', 10, 16);

    // legend — construction level colors
    if (n > 2) {
      let legendY = 32;
      for (let lvl = 0; lvl < Math.min(n - 1, LEVEL_COLORS.length); lvl++) {
        ctx.fillStyle = LEVEL_COLORS[lvl];
        ctx.fillRect(10, legendY - 5, 10, 3);
        ctx.fillStyle = 'rgba(176,176,176,0.2)';
        ctx.fillText('L' + (lvl + 1), 24, legendY);
        legendY += 14;
      }
    }

    // hint text
    if (!isDragging && !animating) {
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillStyle = 'rgba(176,176,176,0.2)';
      ctx.textAlign = 'center';
      ctx.fillText('drag points \u2022 adjust t \u2022 animate to see construction', canvasW / 2, canvasH - 10);
    }
  }

  // ============================================================
  // INTERACTION: Mouse/touch se control points drag karo
  // ============================================================

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvasW / rect.width),
      y: (clientY - rect.top) * (canvasH / rect.height),
    };
  }

  // konsa control point hit hua — sabse paas wala return karo
  function hitTest(mx, my) {
    let closest = -1;
    let closestDist = HIT_RADIUS;

    for (let i = 0; i < controlPoints.length; i++) {
      const dx = mx - controlPoints[i].x;
      const dy = my - controlPoints[i].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < closestDist) {
        closestDist = dist;
        closest = i;
      }
    }

    return closest;
  }

  function onPointerDown(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const hit = hitTest(pos.x, pos.y);

    if (hit >= 0) {
      isDragging = true;
      draggingIdx = hit;
      canvas.style.cursor = 'grabbing';
    }
  }

  function onPointerMove(e) {
    const pos = getCanvasPos(e);

    if (isDragging && draggingIdx >= 0) {
      e.preventDefault();
      // clamp to canvas bounds — point bahar nahi jaana chahiye
      controlPoints[draggingIdx].x = Math.max(4, Math.min(canvasW - 4, pos.x));
      controlPoints[draggingIdx].y = Math.max(4, Math.min(canvasH - 4, pos.y));
    } else {
      // hover cursor change
      const hit = hitTest(pos.x, pos.y);
      canvas.style.cursor = hit >= 0 ? 'grab' : 'default';
    }
  }

  function onPointerUp() {
    if (isDragging) {
      isDragging = false;
      draggingIdx = -1;
      canvas.style.cursor = 'default';
    }
  }

  // mouse events
  canvas.addEventListener('mousedown', onPointerDown);
  canvas.addEventListener('mousemove', onPointerMove);
  canvas.addEventListener('mouseup', onPointerUp);
  canvas.addEventListener('mouseleave', onPointerUp);

  // touch events
  canvas.addEventListener('touchstart', (e) => { onPointerDown(e); }, { passive: false });
  canvas.addEventListener('touchmove', (e) => { onPointerMove(e); }, { passive: false });
  canvas.addEventListener('touchend', onPointerUp, { passive: false });
  canvas.addEventListener('touchcancel', onPointerUp, { passive: false });

  // ============================================================
  // ANIMATION LOOP
  // ============================================================

  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    // t parameter animation — sweep 0 to 1 to 0 loop
    if (animating) {
      tParam += ANIM_SPEED * animDirection;

      // bounce at endpoints
      if (tParam >= 1) {
        tParam = 1;
        animDirection = -1;
      } else if (tParam <= 0) {
        tParam = 0;
        animDirection = 1;
      }

      // slider ko bhi sync karo
      tSlider.slider.value = String(tParam);
      tSlider.valSpan.textContent = tParam.toFixed(2);
    }

    draw();
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — off-screen hone pe pause karo ---
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
}
