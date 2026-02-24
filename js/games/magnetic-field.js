// ============================================================
// Magnetic Dipole Field Visualization — bar magnets place karo, field lines dekho
// Dipole field B(r) = (μ₀/4π) * [3(m·r̂)r̂ - m] / r³
// RK4 integration se field lines, compass grid, heatmap — full EM demo
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, magnetic field visualize karo
export function initMagneticField() {
  const container = document.getElementById('magneticFieldContainer');
  if (!container) {
    console.warn('magneticFieldContainer nahi mila bhai, magnetic field demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const MU_FACTOR = 800; // scaled μ₀/4π — visual effect ke liye
  const FIELD_LINE_STEPS = 600; // max steps per field line
  const FIELD_LINE_STEP_SIZE = 2.0; // har step mein kitna aage badhna hai
  const MAGNET_W = 60; // magnet ki width (length along axis)
  const MAGNET_H = 22; // magnet ki height
  const POLE_SEP = MAGNET_W * 0.4; // N aur S pole ke beech distance
  const SNAP_DIST = 35; // drag/interact ke liye proximity threshold
  const MAX_MAGNETS = 6; // max kitne magnets allow hain
  const COMPASS_COLS = 15; // compass grid columns
  const COMPASS_ROWS = 10; // compass grid rows
  const HEATMAP_RES = 8; // heatmap pixel resolution
  const ACCENT = '#f59e0b'; // amber accent — physics wali feel
  const BG_DARK = '#0a0a0a';
  const BG_PANEL = '#1a1a2e';

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;

  // magnets ka list — {x, y, angle, moment}
  // angle radians mein — 0 = right facing, N pole right side
  let magnets = [];

  // toggles
  let showFieldLines = true;
  let showCompass = true;
  let showHeatmap = false;
  let linesPerMagnet = 12;

  // interaction state
  let dragIndex = -1;
  let dragOffsetX = 0, dragOffsetY = 0;
  let isDragging = false;

  // animation state
  let animationId = null;
  let isVisible = false;
  let needsRedraw = true; // dirty flag — sirf redraw jab kuch badla ho

  // offscreen canvas heatmap ke liye
  let heatCanvas = null;
  let heatCtx = null;

  // --- DOM Structure ---
  // purane children hata do, fresh start
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — field yahan render hoga
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(245,158,11,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:' + BG_DARK,
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls — dark theme, inline CSS, JetBrains Mono ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
    'padding:10px',
    'background:' + BG_PANEL,
    'border-radius:6px',
    'border:1px solid rgba(245,158,11,0.15)',
  ].join(';');
  container.appendChild(controlsDiv);

  // info bar — instructions dikhane ke liye
  const infoBar = document.createElement('div');
  infoBar.style.cssText = [
    'margin-top:6px',
    'padding:6px 10px',
    'font-size:11px',
    'font-family:"JetBrains Mono",monospace',
    'color:#8892a4',
    'background:rgba(26,26,46,0.5)',
    'border-radius:4px',
    'border:1px solid rgba(245,158,11,0.08)',
  ].join(';');
  infoBar.textContent = 'Click = place magnet | Drag = move | Scroll on magnet = rotate | Dbl-click = flip | Right-click = delete';
  container.appendChild(infoBar);

  // --- Control Helpers ---

  // button banane ka helper
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'background:rgba(245,158,11,0.1)',
      'color:' + ACCENT,
      'border:1px solid rgba(245,158,11,0.3)',
      'border-radius:4px',
      'padding:4px 12px',
      'cursor:pointer',
      'font-family:"JetBrains Mono",monospace',
      'font-size:12px',
      'transition:all 0.2s ease',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(245,158,11,0.25)';
      btn.style.color = '#fbbf24';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(245,158,11,0.1)';
      btn.style.color = ACCENT;
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // toggle button — on/off state maintain kare
  function createToggle(label, initial, onChange) {
    const btn = document.createElement('button');
    let active = initial;

    function updateStyle() {
      btn.style.cssText = [
        'padding:4px 12px',
        'font-size:12px',
        'border-radius:4px',
        'cursor:pointer',
        'font-family:"JetBrains Mono",monospace',
        'transition:all 0.2s ease',
        'background:' + (active ? 'rgba(245,158,11,0.25)' : 'rgba(245,158,11,0.05)'),
        'color:' + (active ? '#fbbf24' : '#6b7280'),
        'border:1px solid ' + (active ? 'rgba(245,158,11,0.5)' : 'rgba(245,158,11,0.15)'),
      ].join(';');
      btn.textContent = label + (active ? ' ON' : ' OFF');
    }

    updateStyle();
    btn.addEventListener('click', () => {
      active = !active;
      updateStyle();
      onChange(active);
    });
    controlsDiv.appendChild(btn);
    return btn;
  }

  // --- Controls populate karo ---

  // toggles
  createToggle('Lines', showFieldLines, (v) => { showFieldLines = v; needsRedraw = true; });
  createToggle('Compass', showCompass, (v) => { showCompass = v; needsRedraw = true; });
  createToggle('Heatmap', showHeatmap, (v) => { showHeatmap = v; needsRedraw = true; });

  // line density slider
  const densityWrap = document.createElement('div');
  densityWrap.style.cssText = 'display:flex;align-items:center;gap:6px;';

  const densityLabel = document.createElement('span');
  densityLabel.style.cssText = 'color:#8892a4;font-size:11px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
  densityLabel.textContent = 'Lines: ' + linesPerMagnet;

  const densitySlider = document.createElement('input');
  densitySlider.type = 'range';
  densitySlider.min = '6';
  densitySlider.max = '24';
  densitySlider.step = '2';
  densitySlider.value = String(linesPerMagnet);
  densitySlider.style.cssText = 'width:80px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
  densitySlider.addEventListener('input', () => {
    linesPerMagnet = parseInt(densitySlider.value);
    densityLabel.textContent = 'Lines: ' + linesPerMagnet;
    needsRedraw = true;
  });

  densityWrap.appendChild(densityLabel);
  densityWrap.appendChild(densitySlider);
  controlsDiv.appendChild(densityWrap);

  // clear button
  createButton('Clear All', () => {
    magnets = [];
    needsRedraw = true;
  });

  // presets dropdown
  const presetSelect = document.createElement('select');
  presetSelect.style.cssText = [
    'padding:4px 10px',
    'font-size:12px',
    'border-radius:4px',
    'cursor:pointer',
    'background:rgba(245,158,11,0.1)',
    'color:#8892a4',
    'border:1px solid rgba(245,158,11,0.25)',
    'font-family:"JetBrains Mono",monospace',
  ].join(';');

  const presets = ['Presets...', 'Attraction', 'Repulsion', 'Quadrupole'];
  presets.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    opt.style.background = '#1a1a1a';
    opt.style.color = '#b0b0b0';
    presetSelect.appendChild(opt);
  });

  presetSelect.addEventListener('change', () => {
    loadPreset(presetSelect.value);
    presetSelect.value = 'Presets...';
  });
  controlsDiv.appendChild(presetSelect);

  // ============================
  // DPR-AWARE RESIZE
  // ============================
  function resize() {
    dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    canvasW = w;
    canvasH = CANVAS_HEIGHT;
    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // heatmap offscreen canvas bhi resize karo
    const hW = Math.ceil(canvasW / HEATMAP_RES);
    const hH = Math.ceil(canvasH / HEATMAP_RES);
    heatCanvas = document.createElement('canvas');
    heatCanvas.width = hW;
    heatCanvas.height = hH;
    heatCtx = heatCanvas.getContext('2d');

    needsRedraw = true;
  }
  resize();
  window.addEventListener('resize', resize);

  // ============================
  // PHYSICS — Magnetic Dipole Field
  // ============================

  // ek single magnet ke N aur S pole positions nikalo
  // magnet center (x,y) pe hai, angle radians mein
  function getPoles(mag) {
    const cosA = Math.cos(mag.angle);
    const sinA = Math.sin(mag.angle);
    const halfSep = POLE_SEP;
    // N pole = magnet ke angle direction mein, S pole = opposite
    return {
      nx: mag.x + cosA * halfSep,
      ny: mag.y + sinA * halfSep,
      sx: mag.x - cosA * halfSep,
      sy: mag.y - sinA * halfSep,
    };
  }

  // magnetic field kisi bhi point (px, py) pe — sab magnets ka superposition
  // har magnet ko 2 point poles se model kar rahe hain (monopole approximation)
  // B = MU_FACTOR * q_m * r_hat / r^2 for each pole
  // N pole = +q_m, S pole = -q_m
  function magneticField(px, py) {
    let bx = 0, by = 0;

    for (let i = 0; i < magnets.length; i++) {
      const mag = magnets[i];
      const poles = getPoles(mag);
      const strength = mag.moment;

      // N pole contribution (positive source — field bahar jaata hai)
      let dx = px - poles.nx;
      let dy = py - poles.ny;
      let r2 = dx * dx + dy * dy;
      if (r2 < 9) r2 = 9; // singularity se bachao
      let r = Math.sqrt(r2);
      let r3 = r2 * r;
      let factor = MU_FACTOR * strength / r3;
      bx += factor * dx;
      by += factor * dy;

      // S pole contribution (negative sink — field andar jaata hai)
      dx = px - poles.sx;
      dy = py - poles.sy;
      r2 = dx * dx + dy * dy;
      if (r2 < 9) r2 = 9;
      r = Math.sqrt(r2);
      r3 = r2 * r;
      factor = MU_FACTOR * strength / r3;
      bx -= factor * dx;
      by -= factor * dy;
    }

    return [bx, by];
  }

  // field magnitude — heatmap aur compass brightness ke liye
  function fieldMagnitude(px, py) {
    const [bx, by] = magneticField(px, py);
    return Math.sqrt(bx * bx + by * by);
  }

  // ============================
  // FIELD LINE TRACING — RK4 Integration
  // ============================

  // RK4 se ek step lo B field ke direction mein
  function rk4Step(x, y, h) {
    const [k1x, k1y] = magneticField(x, y);
    const m1 = Math.sqrt(k1x * k1x + k1y * k1y);
    if (m1 < 0.001) return null; // field bahut weak, ruk jao
    const d1x = k1x / m1, d1y = k1y / m1;

    const [k2x, k2y] = magneticField(x + 0.5 * h * d1x, y + 0.5 * h * d1y);
    const m2 = Math.sqrt(k2x * k2x + k2y * k2y);
    if (m2 < 0.001) return null;
    const d2x = k2x / m2, d2y = k2y / m2;

    const [k3x, k3y] = magneticField(x + 0.5 * h * d2x, y + 0.5 * h * d2y);
    const m3 = Math.sqrt(k3x * k3x + k3y * k3y);
    if (m3 < 0.001) return null;
    const d3x = k3x / m3, d3y = k3y / m3;

    const [k4x, k4y] = magneticField(x + h * d3x, y + h * d3y);
    const m4 = Math.sqrt(k4x * k4x + k4y * k4y);
    if (m4 < 0.001) return null;
    const d4x = k4x / m4, d4y = k4y / m4;

    // RK4 weighted average — classic formula
    const nx = x + (h / 6) * (d1x + 2 * d2x + 2 * d3x + d4x);
    const ny = y + (h / 6) * (d1y + 2 * d2y + 2 * d3y + d4y);
    return [nx, ny];
  }

  // ek complete field line trace karo start point se
  // direction: +1 = N se bahar, -1 = S ki taraf
  function traceFieldLine(startX, startY, direction) {
    const points = [[startX, startY]];
    let x = startX, y = startY;
    const h = FIELD_LINE_STEP_SIZE * direction;

    for (let step = 0; step < FIELD_LINE_STEPS; step++) {
      const result = rk4Step(x, y, h);
      if (!result) break;

      x = result[0];
      y = result[1];

      // boundary check — canvas ke bahar nikla toh ruk jao
      if (x < -20 || x > canvasW + 20 || y < -20 || y > canvasH + 20) {
        points.push([x, y]);
        break;
      }

      // kisi S pole ke paas pahuncha toh ruk jao (agar forward trace hai)
      // ya N pole ke paas (agar reverse trace hai)
      let hitPole = false;
      for (let i = 0; i < magnets.length; i++) {
        const poles = getPoles(magnets[i]);
        // forward direction mein S pole pe terminate
        const targetX = direction > 0 ? poles.sx : poles.nx;
        const targetY = direction > 0 ? poles.sy : poles.ny;
        const dx = x - targetX;
        const dy = y - targetY;
        if (dx * dx + dy * dy < 64) { // 8px radius
          hitPole = true;
          break;
        }
      }

      points.push([x, y]);
      if (hitPole) break;
    }

    return points;
  }

  // ============================
  // PRESETS — ready-made magnet configurations
  // ============================
  function loadPreset(name) {
    const cx = canvasW / 2;
    const cy = canvasH / 2;

    switch (name) {
      case 'Attraction':
        // 2 magnets — N S facing each other (attract karenge)
        magnets = [
          { x: cx - 80, y: cy, angle: 0, moment: 1 },       // N right side
          { x: cx + 80, y: cy, angle: Math.PI, moment: 1 },  // N left side (facing each other)
        ];
        break;
      case 'Repulsion':
        // 2 magnets — N N facing each other (repel karenge)
        magnets = [
          { x: cx - 80, y: cy, angle: 0, moment: 1 },  // N right
          { x: cx + 80, y: cy, angle: 0, moment: 1 },   // N right (same direction)
        ];
        break;
      case 'Quadrupole':
        // 4 magnets in square — alternating orientation
        magnets = [
          { x: cx - 70, y: cy - 70, angle: -Math.PI / 4, moment: 1 },
          { x: cx + 70, y: cy - 70, angle: -3 * Math.PI / 4, moment: 1 },
          { x: cx + 70, y: cy + 70, angle: 3 * Math.PI / 4, moment: 1 },
          { x: cx - 70, y: cy + 70, angle: Math.PI / 4, moment: 1 },
        ];
        break;
      default:
        return;
    }
    needsRedraw = true;
  }

  // ============================
  // RENDERING
  // ============================

  function draw() {
    // pura canvas saaf karo
    ctx.clearRect(0, 0, canvasW, canvasH);

    // background — dark solid
    ctx.fillStyle = BG_DARK;
    ctx.fillRect(0, 0, canvasW, canvasH);

    if (magnets.length === 0) {
      // koi magnet nahi hai toh hint dikhao
      ctx.fillStyle = 'rgba(136,146,164,0.4)';
      ctx.font = '14px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Click to place magnets (max 6)', canvasW / 2, canvasH / 2);
      ctx.textAlign = 'start';
      return;
    }

    // heatmap sabse pehle — background layer
    if (showHeatmap) drawHeatmap();

    // compass grid
    if (showCompass) drawCompassGrid();

    // field lines
    if (showFieldLines) drawFieldLines();

    // magnets sabse upar — foreground
    drawMagnets();
  }

  // field lines draw karo — glowing white/cyan curves with arrowheads
  function drawFieldLines() {
    for (let i = 0; i < magnets.length; i++) {
      const mag = magnets[i];
      const poles = getPoles(mag);
      const numLines = linesPerMagnet;

      // N pole se lines start karo — equally spaced angles pe
      for (let j = 0; j < numLines; j++) {
        const angle = (j / numLines) * Math.PI * 2;
        const startX = poles.nx + Math.cos(angle) * 8;
        const startY = poles.ny + Math.sin(angle) * 8;

        // forward trace — N se S ki taraf
        const fwdPoints = traceFieldLine(startX, startY, 1);

        // reverse trace — N se bahar (magnet ke andar se loop complete karne ke liye)
        const revStartX = poles.nx + Math.cos(angle + Math.PI) * 8;
        const revStartY = poles.ny + Math.sin(angle + Math.PI) * 8;

        // dono directions mein trace kar ke full line banao
        drawFieldLinePoints(fwdPoints, 1);
      }
    }
  }

  // field line points draw karo with glow effect aur arrowheads
  function drawFieldLinePoints(points, direction) {
    if (points.length < 2) return;

    // main line — glowing cyan/white
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let p = 1; p < points.length; p++) {
      ctx.lineTo(points[p][0], points[p][1]);
    }
    ctx.strokeStyle = 'rgba(180,230,255,0.5)';
    ctx.lineWidth = 1.3;
    ctx.stroke();

    // glow layer — thodi moti aur transparent
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let p = 1; p < points.length; p++) {
      ctx.lineTo(points[p][0], points[p][1]);
    }
    ctx.strokeStyle = 'rgba(100,200,255,0.15)';
    ctx.lineWidth = 3;
    ctx.stroke();

    // arrowheads — har 60 points pe ek arrow
    const arrowInterval = 60;
    for (let p = arrowInterval; p < points.length - 1; p += arrowInterval) {
      const ax = points[p][0];
      const ay = points[p][1];
      const bx = points[p + 1][0];
      const by = points[p + 1][1];
      const dx = bx - ax;
      const dy = by - ay;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len < 0.5) continue;

      const ux = dx / len;
      const uy = dy / len;
      const headLen = 5;
      const headWidth = 3;

      // perpendicular direction
      const px = -uy;
      const py = ux;

      ctx.beginPath();
      ctx.moveTo(ax + ux * headLen, ay + uy * headLen);
      ctx.lineTo(ax - px * headWidth, ay - py * headWidth);
      ctx.lineTo(ax + px * headWidth, ay + py * headWidth);
      ctx.closePath();
      ctx.fillStyle = 'rgba(180,230,255,0.7)';
      ctx.fill();
    }
  }

  // compass grid draw karo — small arrows jo local B direction dikhayein
  function drawCompassGrid() {
    const spacingX = canvasW / (COMPASS_COLS + 1);
    const spacingY = canvasH / (COMPASS_ROWS + 1);
    const needleLen = Math.min(spacingX, spacingY) * 0.35;

    // max field magnitude dhundho — color scaling ke liye
    let maxMag = 0.1;
    for (let row = 1; row <= COMPASS_ROWS; row++) {
      for (let col = 1; col <= COMPASS_COLS; col++) {
        const x = col * spacingX;
        const y = row * spacingY;
        const mag = fieldMagnitude(x, y);
        if (mag > maxMag && mag < 10000) maxMag = mag;
      }
    }

    for (let row = 1; row <= COMPASS_ROWS; row++) {
      for (let col = 1; col <= COMPASS_COLS; col++) {
        const x = col * spacingX;
        const y = row * spacingY;

        // kisi magnet ke bahut paas ho toh skip karo
        let tooClose = false;
        for (let m = 0; m < magnets.length; m++) {
          const dx = x - magnets[m].x;
          const dy = y - magnets[m].y;
          if (dx * dx + dy * dy < 900) { tooClose = true; break; }
        }
        if (tooClose) continue;

        const [bx, by] = magneticField(x, y);
        const mag = Math.sqrt(bx * bx + by * by);
        if (mag < 0.001) continue;

        // direction angle
        const angle = Math.atan2(by, bx);

        // brightness — stronger field = brighter needle
        const brightness = Math.min(1, mag / maxMag);
        const alpha = 0.15 + brightness * 0.65;

        // needle ko triangular arrow ki tarah draw karo
        const tipX = x + Math.cos(angle) * needleLen;
        const tipY = y + Math.sin(angle) * needleLen;
        const tailX = x - Math.cos(angle) * needleLen * 0.5;
        const tailY = y - Math.sin(angle) * needleLen * 0.5;

        // perpendicular — width ke liye
        const perpX = -Math.sin(angle);
        const perpY = Math.cos(angle);
        const halfW = 2;

        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tailX + perpX * halfW, tailY + perpY * halfW);
        ctx.lineTo(tailX - perpX * halfW, tailY - perpY * halfW);
        ctx.closePath();

        // color — amber with variable brightness
        const r = Math.floor(245 * brightness + 100 * (1 - brightness));
        const g = Math.floor(158 * brightness + 80 * (1 - brightness));
        const b = Math.floor(11 * brightness + 60 * (1 - brightness));
        ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
        ctx.fill();
      }
    }
  }

  // heatmap draw karo — field strength ka color map
  function drawHeatmap() {
    if (!heatCanvas || !heatCtx) return;

    const hW = heatCanvas.width;
    const hH = heatCanvas.height;
    const imgData = heatCtx.createImageData(hW, hH);
    const data = imgData.data;

    // max field magnitude dhundho — normalization ke liye
    let maxMag = 0.1;
    for (let hy = 0; hy < hH; hy++) {
      for (let hx = 0; hx < hW; hx++) {
        const px = (hx + 0.5) * HEATMAP_RES;
        const py = (hy + 0.5) * HEATMAP_RES;
        const mag = fieldMagnitude(px, py);
        if (mag > maxMag && mag < 50000) maxMag = mag;
      }
    }

    // har pixel ka color set karo
    for (let hy = 0; hy < hH; hy++) {
      for (let hx = 0; hx < hW; hx++) {
        const px = (hx + 0.5) * HEATMAP_RES;
        const py = (hy + 0.5) * HEATMAP_RES;
        const mag = fieldMagnitude(px, py);

        // normalize — log scale better lagta hai large range ke liye
        const t = Math.min(1, Math.log(1 + mag) / Math.log(1 + maxMag));

        // color map: dark blue → cyan → yellow → red
        let r, g, b;
        if (t < 0.25) {
          // dark blue → blue
          const s = t / 0.25;
          r = Math.floor(5 + s * 15);
          g = Math.floor(5 + s * 30);
          b = Math.floor(40 + s * 100);
        } else if (t < 0.5) {
          // blue → cyan
          const s = (t - 0.25) / 0.25;
          r = Math.floor(20 + s * 10);
          g = Math.floor(35 + s * 180);
          b = Math.floor(140 + s * 60);
        } else if (t < 0.75) {
          // cyan → yellow
          const s = (t - 0.5) / 0.25;
          r = Math.floor(30 + s * 220);
          g = Math.floor(215 - s * 20);
          b = Math.floor(200 - s * 180);
        } else {
          // yellow → red
          const s = (t - 0.75) / 0.25;
          r = Math.floor(250);
          g = Math.floor(195 - s * 150);
          b = Math.floor(20 - s * 15);
        }

        const idx = (hy * hW + hx) * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 100; // semi-transparent — background layer hai
      }
    }

    heatCtx.putImageData(imgData, 0, 0);

    // offscreen canvas ko main canvas pe draw karo — scaled up
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heatCanvas, 0, 0, canvasW, canvasH);
  }

  // magnets draw karo — rounded rectangles with N/S labels, gradient fill
  function drawMagnets() {
    for (let i = 0; i < magnets.length; i++) {
      const mag = magnets[i];
      const poles = getPoles(mag);

      ctx.save();
      ctx.translate(mag.x, mag.y);
      ctx.rotate(mag.angle);

      const halfW = MAGNET_W / 2;
      const halfH = MAGNET_H / 2;
      const radius = 5; // rounded corners

      // glow effect — magnet ke around subtle glow
      const glowGrad = ctx.createRadialGradient(0, 0, halfW * 0.5, 0, 0, halfW * 1.8);
      glowGrad.addColorStop(0, 'rgba(245,158,11,0.1)');
      glowGrad.addColorStop(1, 'rgba(245,158,11,0)');
      ctx.fillStyle = glowGrad;
      ctx.fillRect(-halfW * 2, -halfH * 2, halfW * 4, halfH * 4);

      // magnet body — left half S (blue), right half N (red)
      // gradient: blue → red (S → N along angle direction)
      const bodyGrad = ctx.createLinearGradient(-halfW, 0, halfW, 0);
      bodyGrad.addColorStop(0, '#2563eb');   // S pole — blue
      bodyGrad.addColorStop(0.45, '#1e40af');
      bodyGrad.addColorStop(0.55, '#b91c1c');
      bodyGrad.addColorStop(1, '#ef4444');    // N pole — red

      // rounded rectangle path
      ctx.beginPath();
      ctx.moveTo(-halfW + radius, -halfH);
      ctx.lineTo(halfW - radius, -halfH);
      ctx.arcTo(halfW, -halfH, halfW, -halfH + radius, radius);
      ctx.lineTo(halfW, halfH - radius);
      ctx.arcTo(halfW, halfH, halfW - radius, halfH, radius);
      ctx.lineTo(-halfW + radius, halfH);
      ctx.arcTo(-halfW, halfH, -halfW, halfH - radius, radius);
      ctx.lineTo(-halfW, -halfH + radius);
      ctx.arcTo(-halfW, -halfH, -halfW + radius, -halfH, radius);
      ctx.closePath();

      ctx.fillStyle = bodyGrad;
      ctx.fill();

      // border
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // center divider line — N aur S ke beech
      ctx.beginPath();
      ctx.moveTo(0, -halfH);
      ctx.lineTo(0, halfH);
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // N label — right side (angle direction)
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('N', halfW * 0.5, 0);

      // S label — left side (opposite direction)
      ctx.fillText('S', -halfW * 0.5, 0);

      ctx.restore();
    }

    // text alignment reset
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  }

  // ============================
  // INTERACTION — mouse/touch events
  // ============================

  // canvas pe mouse position nikalo — CSS coordinates mein
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    return [
      e.clientX - rect.left,
      e.clientY - rect.top,
    ];
  }

  // closest magnet dhundho mouse ke paas
  function findMagnetAt(mx, my) {
    for (let i = 0; i < magnets.length; i++) {
      const mag = magnets[i];
      const dx = mx - mag.x;
      const dy = my - mag.y;
      if (dx * dx + dy * dy < SNAP_DIST * SNAP_DIST) return i;
    }
    return -1;
  }

  // mousedown — drag start ya new magnet place karo
  canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return; // sirf left click
    e.preventDefault();
    const [mx, my] = getCanvasPos(e);

    const idx = findMagnetAt(mx, my);

    if (idx >= 0) {
      // existing magnet pe click — drag start karo
      dragIndex = idx;
      dragOffsetX = magnets[idx].x - mx;
      dragOffsetY = magnets[idx].y - my;
      isDragging = false;
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    const [mx, my] = getCanvasPos(e);

    if (dragIndex >= 0) {
      isDragging = true;
      magnets[dragIndex].x = mx + dragOffsetX;
      magnets[dragIndex].y = my + dragOffsetY;
      needsRedraw = true;
    }
  });

  canvas.addEventListener('mouseup', (e) => {
    if (e.button !== 0) { dragIndex = -1; isDragging = false; return; }
    const [mx, my] = getCanvasPos(e);

    if (dragIndex < 0 && !isDragging) {
      // empty space pe click — new magnet place karo
      if (magnets.length < MAX_MAGNETS) {
        magnets.push({ x: mx, y: my, angle: 0, moment: 1 });
        needsRedraw = true;
      }
    }

    dragIndex = -1;
    isDragging = false;
    needsRedraw = true;
  });

  // scroll wheel — magnet rotate karo
  canvas.addEventListener('wheel', (e) => {
    const [mx, my] = getCanvasPos(e);
    const idx = findMagnetAt(mx, my);
    if (idx >= 0) {
      e.preventDefault();
      // scroll up = clockwise, scroll down = counter-clockwise
      magnets[idx].angle += e.deltaY > 0 ? 0.15 : -0.15;
      needsRedraw = true;
    }
  }, { passive: false });

  // double-click — polarity flip karo (180° rotate)
  canvas.addEventListener('dblclick', (e) => {
    e.preventDefault();
    const [mx, my] = getCanvasPos(e);
    const idx = findMagnetAt(mx, my);
    if (idx >= 0) {
      // 180° flip — N aur S swap ho jaayenge
      magnets[idx].angle += Math.PI;
      needsRedraw = true;
    }
  });

  // right-click — magnet delete karo
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const [mx, my] = getCanvasPos(e);
    const idx = findMagnetAt(mx, my);
    if (idx >= 0) {
      magnets.splice(idx, 1);
      needsRedraw = true;
    }
  });

  // --- Touch support — mobile pe bhi kaam kare ---
  let touchId = null;
  let touchStartTime = 0;
  let touchMoved = false;
  let touchStartIdx = -1;

  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (touchId !== null) return;
    const touch = e.changedTouches[0];
    touchId = touch.identifier;
    touchStartTime = performance.now();
    touchMoved = false;

    const rect = canvas.getBoundingClientRect();
    const mx = touch.clientX - rect.left;
    const my = touch.clientY - rect.top;

    const idx = findMagnetAt(mx, my);
    touchStartIdx = idx;
    if (idx >= 0) {
      dragIndex = idx;
      dragOffsetX = magnets[idx].x - mx;
      dragOffsetY = magnets[idx].y - my;
      isDragging = false;
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    for (let i = 0; i < e.changedTouches.length; i++) {
      const touch = e.changedTouches[i];
      if (touch.identifier !== touchId) continue;

      touchMoved = true;
      const rect = canvas.getBoundingClientRect();
      const mx = touch.clientX - rect.left;
      const my = touch.clientY - rect.top;

      if (dragIndex >= 0) {
        isDragging = true;
        magnets[dragIndex].x = mx + dragOffsetX;
        magnets[dragIndex].y = my + dragOffsetY;
        needsRedraw = true;
      }
    }
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    for (let i = 0; i < e.changedTouches.length; i++) {
      const touch = e.changedTouches[i];
      if (touch.identifier !== touchId) continue;

      const rect = canvas.getBoundingClientRect();
      const mx = touch.clientX - rect.left;
      const my = touch.clientY - rect.top;

      if (!touchMoved && touchStartIdx < 0) {
        // tap on empty space — new magnet
        if (magnets.length < MAX_MAGNETS) {
          magnets.push({ x: mx, y: my, angle: 0, moment: 1 });
          needsRedraw = true;
        }
      }

      // long press on magnet = flip polarity
      if (!touchMoved && touchStartIdx >= 0) {
        const holdTime = performance.now() - touchStartTime;
        if (holdTime > 500 && touchStartIdx < magnets.length) {
          magnets[touchStartIdx].angle += Math.PI;
          needsRedraw = true;
        }
      }

      dragIndex = -1;
      isDragging = false;
      touchId = null;
      touchStartIdx = -1;
      needsRedraw = true;
    }
  }, { passive: false });

  // ============================
  // ANIMATION LOOP — requestAnimationFrame
  // ============================

  function loop() {
    // lab pause: sirf active sim animate kare
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    // sirf redraw karo jab kuch change hua ho — performance ke liye
    if (needsRedraw) {
      draw();
      needsRedraw = false;
    }

    animationId = requestAnimationFrame(loop);
  }

  // ============================
  // INTERSECTION OBSERVER — sirf visible hone pe animate karo
  // ============================
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) {
      resize();
      needsRedraw = true;
      loop();
    } else if (!isVisible && animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }, { threshold: 0.1 });
  obs.observe(container);

  // lab resume: restart loop jab focus wapas aaye
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });

  // tab visibility: pause jab tab hidden ho
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    } else if (!document.hidden && isVisible && !animationId) {
      needsRedraw = true;
      loop();
    }
  });

  // --- Attraction preset se start karo — empty canvas boring lagta hai ---
  loadPreset('Attraction');
  needsRedraw = true;
}
