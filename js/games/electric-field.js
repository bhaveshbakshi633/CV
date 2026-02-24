// ============================================================
// Interactive Electric Field Visualizer — charges place karo, field lines dekho
// Coulomb's law se E field, RK4 integration, equipotential contours
// Marching squares + field line tracing — full EM demo
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, electric field visualize karo
export function initElectricField() {
  const container = document.getElementById('electricFieldContainer');
  if (!container) {
    console.warn('electricFieldContainer nahi mila bhai, electric field demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const K_COULOMB = 500; // scaled Coulomb constant — visual effect ke liye
  const FIELD_LINE_STEPS = 800; // max steps per field line
  const FIELD_LINE_STEP_SIZE = 2.5; // har step mein kitna aage badhna hai
  const LINES_PER_CHARGE = 16; // har positive charge se kitni lines niklengi
  const CHARGE_RADIUS = 18; // charge circle ka radius
  const SNAP_DIST = 22; // drag/remove ke liye proximity threshold
  const POTENTIAL_GRID_X = 120; // equipotential grid resolution
  const POTENTIAL_GRID_Y = 80;
  const VECTOR_GRID_SPACING = 35; // vector field arrows ka spacing
  const ARROW_MAX_LEN = 14; // arrow ki max length
  const ACCENT = '#f59e0b'; // amber accent
  const BG_DARK = '#1a1a2e';

  // equipotential contour levels — positive aur negative dono
  const CONTOUR_LEVELS = [-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50];

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  // charges ka list — {x, y, q} where q > 0 positive, q < 0 negative
  let charges = [];
  let chargeMagnitude = 1; // slider se control hoga

  // toggles
  let showFieldLines = true;
  let showEquipotentials = true;
  let showVectorField = false;

  // interaction state
  let dragIndex = -1; // kaunsa charge drag ho raha hai
  let dragOffsetX = 0, dragOffsetY = 0;
  let isDragging = false;
  let mouseX = 0, mouseY = 0;

  // animation state
  let animationId = null;
  let isVisible = false;
  let needsRedraw = true; // dirty flag — sirf redraw karo jab kuch badla ho

  // --- DOM Structure ---
  // existing children preserve karo (game-header, game-desc, details wagairah)
  const existingChildren = Array.from(container.children);
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
    'background:rgba(2,2,8,0.5)',
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
    'background:' + BG_DARK,
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
  infoBar.textContent = 'Click = +charge | Shift+Click = -charge | Drag to move | Right-click or Double-click to remove';
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
  createToggle('Equip.', showEquipotentials, (v) => { showEquipotentials = v; needsRedraw = true; });
  createToggle('Vectors', showVectorField, (v) => { showVectorField = v; needsRedraw = true; });

  // clear button
  createButton('Clear All', () => {
    charges = [];
    needsRedraw = true;
  });

  // charge magnitude slider
  const magWrap = document.createElement('div');
  magWrap.style.cssText = 'display:flex;align-items:center;gap:6px;';

  const magLabel = document.createElement('span');
  magLabel.style.cssText = 'color:#8892a4;font-size:11px;font-family:"JetBrains Mono",monospace;';
  magLabel.textContent = 'Q: 1';

  const magSlider = document.createElement('input');
  magSlider.type = 'range';
  magSlider.min = '1';
  magSlider.max = '5';
  magSlider.step = '1';
  magSlider.value = '1';
  magSlider.style.cssText = 'width:70px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
  magSlider.addEventListener('input', () => {
    chargeMagnitude = parseInt(magSlider.value);
    magLabel.textContent = 'Q: ' + chargeMagnitude;
  });

  magWrap.appendChild(magLabel);
  magWrap.appendChild(magSlider);
  controlsDiv.appendChild(magWrap);

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

  const presets = ['Presets...', 'Dipole', 'Quadrupole', 'Parallel Plates'];
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
    needsRedraw = true;
  }
  resize();
  window.addEventListener('resize', resize);

  // ============================
  // PHYSICS — Electric Field & Potential
  // ============================

  // kisi bhi point pe electric field calculate karo (Ex, Ey)
  // E = k * q * r_hat / r^2, sab charges ka sum
  function electricField(px, py) {
    let ex = 0, ey = 0;
    for (let i = 0; i < charges.length; i++) {
      const c = charges[i];
      const dx = px - c.x;
      const dy = py - c.y;
      const r2 = dx * dx + dy * dy;
      // singularity se bachne ke liye minimum distance rakh
      if (r2 < 4) continue;
      const r = Math.sqrt(r2);
      const r3 = r2 * r;
      // E = kQ / r^2, direction = r_hat
      const factor = K_COULOMB * c.q / r3;
      ex += factor * dx;
      ey += factor * dy;
    }
    return [ex, ey];
  }

  // kisi bhi point pe electric potential calculate karo
  // V = sum( k * q / r )
  function potential(px, py) {
    let v = 0;
    for (let i = 0; i < charges.length; i++) {
      const c = charges[i];
      const dx = px - c.x;
      const dy = py - c.y;
      const r = Math.sqrt(dx * dx + dy * dy);
      if (r < 2) return Infinity; // charge ke exactly upar infinite potential
      v += K_COULOMB * c.q / r;
    }
    return v;
  }

  // ============================
  // FIELD LINE TRACING — RK4 Integration
  // ============================

  // RK4 se ek step lo E field ke direction mein
  function rk4Step(x, y, h) {
    const [k1x, k1y] = electricField(x, y);
    const m1 = Math.sqrt(k1x * k1x + k1y * k1y);
    if (m1 < 0.001) return null; // field bahut weak hai, ruk jao
    const d1x = k1x / m1, d1y = k1y / m1;

    const [k2x, k2y] = electricField(x + 0.5 * h * d1x, y + 0.5 * h * d1y);
    const m2 = Math.sqrt(k2x * k2x + k2y * k2y);
    if (m2 < 0.001) return null;
    const d2x = k2x / m2, d2y = k2y / m2;

    const [k3x, k3y] = electricField(x + 0.5 * h * d2x, y + 0.5 * h * d2y);
    const m3 = Math.sqrt(k3x * k3x + k3y * k3y);
    if (m3 < 0.001) return null;
    const d3x = k3x / m3, d3y = k3y / m3;

    const [k4x, k4y] = electricField(x + h * d3x, y + h * d3y);
    const m4 = Math.sqrt(k4x * k4x + k4y * k4y);
    if (m4 < 0.001) return null;
    const d4x = k4x / m4, d4y = k4y / m4;

    // RK4 weighted average
    const nx = x + (h / 6) * (d1x + 2 * d2x + 2 * d3x + d4x);
    const ny = y + (h / 6) * (d1y + 2 * d2y + 2 * d3y + d4y);
    return [nx, ny];
  }

  // ek complete field line trace karo start point se
  // direction: +1 = field ke saath, -1 = field ke against
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
      if (x < -10 || x > canvasW + 10 || y < -10 || y > canvasH + 10) {
        points.push([x, y]);
        break;
      }

      // kisi negative charge ke paas pahuncha toh ruk jao
      let hitCharge = false;
      for (let i = 0; i < charges.length; i++) {
        const c = charges[i];
        // sirf opposite sign charge pe terminate karo
        if (c.q * direction < 0) {
          const dx = x - c.x;
          const dy = y - c.y;
          if (dx * dx + dy * dy < CHARGE_RADIUS * CHARGE_RADIUS) {
            hitCharge = true;
            break;
          }
        }
      }

      points.push([x, y]);
      if (hitCharge) break;
    }

    return points;
  }

  // ============================
  // MARCHING SQUARES — Equipotential Contours
  // ============================

  // marching squares se ek specific contour level ki lines nikalo
  function marchingSquaresContour(potGrid, gridW, gridH, cellW, cellH, level) {
    const segments = [];

    // har cell ke 4 corners ka potential compare karo level se
    for (let gy = 0; gy < gridH - 1; gy++) {
      for (let gx = 0; gx < gridW - 1; gx++) {
        // corners: top-left, top-right, bottom-right, bottom-left
        const tl = potGrid[gy * gridW + gx];
        const tr = potGrid[gy * gridW + gx + 1];
        const br = potGrid[(gy + 1) * gridW + gx + 1];
        const bl = potGrid[(gy + 1) * gridW + gx];

        // infinite values skip karo — charge ke bilkul paas
        if (!isFinite(tl) || !isFinite(tr) || !isFinite(br) || !isFinite(bl)) continue;

        // marching squares case — 4-bit index
        let caseIndex = 0;
        if (tl > level) caseIndex |= 1;
        if (tr > level) caseIndex |= 2;
        if (br > level) caseIndex |= 4;
        if (bl > level) caseIndex |= 8;

        // 0 ya 15 = koi contour nahi
        if (caseIndex === 0 || caseIndex === 15) continue;

        // linear interpolation se edge pe exact crossing point nikalo
        const x0 = gx * cellW;
        const y0 = gy * cellH;
        const x1 = (gx + 1) * cellW;
        const y1 = (gy + 1) * cellH;

        // edges: top, right, bottom, left
        // interpolation factor t = (level - v1) / (v2 - v1)
        function interpTop() {
          const t = (level - tl) / (tr - tl);
          return [x0 + t * cellW, y0];
        }
        function interpRight() {
          const t = (level - tr) / (br - tr);
          return [x1, y0 + t * cellH];
        }
        function interpBottom() {
          const t = (level - bl) / (br - bl);
          return [x0 + t * cellW, y1];
        }
        function interpLeft() {
          const t = (level - tl) / (bl - tl);
          return [x0, y0 + t * cellH];
        }

        // marching squares lookup table — har case ke liye segments
        const segs = marchingSquaresLUT(caseIndex, interpTop, interpRight, interpBottom, interpLeft);
        for (let s = 0; s < segs.length; s++) {
          segments.push(segs[s]);
        }
      }
    }

    return segments;
  }

  // marching squares lookup — case se segments nikalo
  function marchingSquaresLUT(c, top, right, bottom, left) {
    // classic 16-case LUT — har case 0/1/2 line segments deta hai
    switch (c) {
      case 1:  return [[left(), top()]];
      case 2:  return [[top(), right()]];
      case 3:  return [[left(), right()]];
      case 4:  return [[right(), bottom()]];
      case 5:  return [[left(), top()], [right(), bottom()]]; // saddle
      case 6:  return [[top(), bottom()]];
      case 7:  return [[left(), bottom()]];
      case 8:  return [[bottom(), left()]];
      case 9:  return [[bottom(), top()]];
      case 10: return [[top(), right()], [bottom(), left()]]; // saddle
      case 11: return [[bottom(), right()]];
      case 12: return [[right(), left()]];
      case 13: return [[right(), top()]];
      case 14: return [[top(), left()]];
      default: return [];
    }
  }

  // ============================
  // PRESETS — ready-made charge configurations
  // ============================
  function loadPreset(name) {
    const cx = canvasW / 2;
    const cy = canvasH / 2;

    switch (name) {
      case 'Dipole':
        charges = [
          { x: cx - 80, y: cy, q: 1 },
          { x: cx + 80, y: cy, q: -1 },
        ];
        break;
      case 'Quadrupole':
        charges = [
          { x: cx - 70, y: cy - 70, q: 1 },
          { x: cx + 70, y: cy - 70, q: -1 },
          { x: cx + 70, y: cy + 70, q: 1 },
          { x: cx - 70, y: cy + 70, q: -1 },
        ];
        break;
      case 'Parallel Plates':
        // 2 rows of charges — ek positive, ek negative
        charges = [];
        const plateLen = 5;
        const spacing = 40;
        const plateGap = 140;
        for (let i = 0; i < plateLen; i++) {
          const yOff = (i - (plateLen - 1) / 2) * spacing;
          charges.push({ x: cx - plateGap / 2, y: cy + yOff, q: 1 });
          charges.push({ x: cx + plateGap / 2, y: cy + yOff, q: -1 });
        }
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

    // background gradient — subtle dark effect
    const bgGrad = ctx.createLinearGradient(0, 0, 0, canvasH);
    bgGrad.addColorStop(0, 'rgba(2,2,12,0.9)');
    bgGrad.addColorStop(1, 'rgba(5,5,20,0.9)');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, canvasW, canvasH);

    if (charges.length === 0) {
      // koi charge nahi hai toh hint dikhao
      ctx.fillStyle = 'rgba(136,146,164,0.4)';
      ctx.font = '14px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Click to place charges', canvasW / 2, canvasH / 2);
      ctx.textAlign = 'start';
      return;
    }

    // vector field sabse pehle draw karo — background layer
    if (showVectorField) drawVectorField();

    // equipotential contours
    if (showEquipotentials) drawEquipotentials();

    // field lines
    if (showFieldLines) drawFieldLines();

    // charges sabse upar — foreground
    drawCharges();
  }

  // field lines draw karo — amber gradient
  function drawFieldLines() {
    for (let i = 0; i < charges.length; i++) {
      const c = charges[i];
      // sirf positive charges se lines start karo
      // agar koi positive charge nahi hai toh negative se reverse trace karo
      if (c.q <= 0) continue;

      const absQ = Math.abs(c.q);
      const numLines = LINES_PER_CHARGE * absQ;

      for (let j = 0; j < numLines; j++) {
        const angle = (j / numLines) * Math.PI * 2;
        const startX = c.x + Math.cos(angle) * (CHARGE_RADIUS + 2);
        const startY = c.y + Math.sin(angle) * (CHARGE_RADIUS + 2);

        const points = traceFieldLine(startX, startY, 1);
        if (points.length < 2) continue;

        // gradient ke saath draw karo — amber se fade
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        for (let p = 1; p < points.length; p++) {
          ctx.lineTo(points[p][0], points[p][1]);
        }
        ctx.strokeStyle = 'rgba(245,158,11,0.5)';
        ctx.lineWidth = 1.2;
        ctx.stroke();

        // fading trail effect — end ki taraf dim karo
        if (points.length > 4) {
          const fadeStart = Math.floor(points.length * 0.6);
          for (let p = fadeStart; p < points.length - 1; p++) {
            const alpha = 0.5 * (1 - (p - fadeStart) / (points.length - fadeStart));
            ctx.beginPath();
            ctx.moveTo(points[p][0], points[p][1]);
            ctx.lineTo(points[p + 1][0], points[p + 1][1]);
            ctx.strokeStyle = 'rgba(245,158,11,' + Math.max(0.05, alpha) + ')';
            ctx.lineWidth = 1.2;
            ctx.stroke();
          }
        }
      }
    }

    // agar sirf negative charges hain toh unse reverse trace karo
    const hasPositive = charges.some(c => c.q > 0);
    if (!hasPositive) {
      for (let i = 0; i < charges.length; i++) {
        const c = charges[i];
        if (c.q >= 0) continue;

        const absQ = Math.abs(c.q);
        const numLines = LINES_PER_CHARGE * absQ;

        for (let j = 0; j < numLines; j++) {
          const angle = (j / numLines) * Math.PI * 2;
          const startX = c.x + Math.cos(angle) * (CHARGE_RADIUS + 2);
          const startY = c.y + Math.sin(angle) * (CHARGE_RADIUS + 2);

          // reverse direction se trace karo
          const points = traceFieldLine(startX, startY, -1);
          if (points.length < 2) continue;

          ctx.beginPath();
          ctx.moveTo(points[0][0], points[0][1]);
          for (let p = 1; p < points.length; p++) {
            ctx.lineTo(points[p][0], points[p][1]);
          }
          ctx.strokeStyle = 'rgba(245,158,11,0.4)';
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
      }
    }
  }

  // equipotential contours draw karo — marching squares se
  function drawEquipotentials() {
    const gridW = POTENTIAL_GRID_X;
    const gridH = POTENTIAL_GRID_Y;
    const cellW = canvasW / (gridW - 1);
    const cellH = canvasH / (gridH - 1);

    // potential grid compute karo
    const potGrid = new Float64Array(gridW * gridH);
    for (let gy = 0; gy < gridH; gy++) {
      for (let gx = 0; gx < gridW; gx++) {
        const px = gx * cellW;
        const py = gy * cellH;
        potGrid[gy * gridW + gx] = potential(px, py);
      }
    }

    // har contour level ke liye marching squares chalao
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 0.8;

    for (let li = 0; li < CONTOUR_LEVELS.length; li++) {
      const level = CONTOUR_LEVELS[li];
      const segments = marchingSquaresContour(potGrid, gridW, gridH, cellW, cellH, level);

      if (segments.length === 0) continue;

      // positive potential = warm cyan, negative = cool cyan
      const alpha = Math.max(0.15, 0.4 - Math.abs(level) * 0.005);
      ctx.strokeStyle = level > 0
        ? 'rgba(34,211,238,' + alpha + ')'
        : 'rgba(96,165,250,' + alpha + ')';

      ctx.beginPath();
      for (let s = 0; s < segments.length; s++) {
        const seg = segments[s];
        ctx.moveTo(seg[0][0], seg[0][1]);
        ctx.lineTo(seg[1][0], seg[1][1]);
      }
      ctx.stroke();
    }

    ctx.setLineDash([]);
  }

  // vector field arrows draw karo — grid pe small arrows
  function drawVectorField() {
    const spacing = VECTOR_GRID_SPACING;

    for (let x = spacing; x < canvasW; x += spacing) {
      for (let y = spacing; y < canvasH; y += spacing) {
        const [ex, ey] = electricField(x, y);
        const mag = Math.sqrt(ex * ex + ey * ey);
        if (mag < 0.01) continue;

        // direction unit vector
        const dx = ex / mag;
        const dy = ey / mag;

        // arrow length — magnitude ke log proportional
        const len = Math.min(ARROW_MAX_LEN, Math.log(1 + mag) * 3);

        // alpha bhi magnitude pe depend kare — weak field dim dikhao
        const alpha = Math.min(0.6, Math.max(0.08, mag * 0.02));

        const tipX = x + dx * len;
        const tipY = y + dy * len;

        // arrow line
        ctx.beginPath();
        ctx.moveTo(x - dx * len * 0.3, y - dy * len * 0.3);
        ctx.lineTo(tipX, tipY);
        ctx.strokeStyle = 'rgba(245,158,11,' + alpha + ')';
        ctx.lineWidth = 1;
        ctx.stroke();

        // arrowhead — chhota triangle
        const headLen = Math.min(4, len * 0.4);
        const angle = Math.atan2(dy, dx);
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(
          tipX - headLen * Math.cos(angle - 0.5),
          tipY - headLen * Math.sin(angle - 0.5)
        );
        ctx.lineTo(
          tipX - headLen * Math.cos(angle + 0.5),
          tipY - headLen * Math.sin(angle + 0.5)
        );
        ctx.closePath();
        ctx.fillStyle = 'rgba(245,158,11,' + alpha + ')';
        ctx.fill();
      }
    }
  }

  // charges draw karo — circle + symbol + glow
  function drawCharges() {
    for (let i = 0; i < charges.length; i++) {
      const c = charges[i];
      const isPositive = c.q > 0;
      const absQ = Math.abs(c.q);
      const r = CHARGE_RADIUS + (absQ - 1) * 3; // bade charge = bada circle

      // glow effect — radial gradient
      const glow = ctx.createRadialGradient(c.x, c.y, r * 0.5, c.x, c.y, r * 3);
      if (isPositive) {
        glow.addColorStop(0, 'rgba(239,68,68,0.3)');
        glow.addColorStop(1, 'rgba(239,68,68,0)');
      } else {
        glow.addColorStop(0, 'rgba(59,130,246,0.3)');
        glow.addColorStop(1, 'rgba(59,130,246,0)');
      }
      ctx.beginPath();
      ctx.arc(c.x, c.y, r * 3, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      // main circle
      ctx.beginPath();
      ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
      ctx.fillStyle = isPositive ? 'rgba(239,68,68,0.85)' : 'rgba(59,130,246,0.85)';
      ctx.fill();
      ctx.strokeStyle = isPositive ? '#fca5a5' : '#93c5fd';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // + ya - symbol
      ctx.fillStyle = '#fff';
      ctx.font = 'bold ' + (14 + absQ * 2) + 'px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(isPositive ? '+' : '\u2212', c.x, c.y);

      // agar magnitude > 1 hai toh number bhi dikhao
      if (absQ > 1) {
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.fillText(absQ.toString(), c.x, c.y + r + 10);
      }
    }

    // text alignment reset karo
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

  // closest charge dhundho mouse ke paas
  function findChargeAt(mx, my) {
    for (let i = 0; i < charges.length; i++) {
      const c = charges[i];
      const dx = mx - c.x;
      const dy = my - c.y;
      if (dx * dx + dy * dy < SNAP_DIST * SNAP_DIST) return i;
    }
    return -1;
  }

  // mousedown — drag start ya new charge place karo
  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const [mx, my] = getCanvasPos(e);
    mouseX = mx; mouseY = my;

    const idx = findChargeAt(mx, my);

    if (idx >= 0) {
      // existing charge pe click — drag start karo
      dragIndex = idx;
      dragOffsetX = charges[idx].x - mx;
      dragOffsetY = charges[idx].y - my;
      isDragging = false; // abhi sirf click hai, drag confirm nahi hua
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    const [mx, my] = getCanvasPos(e);
    mouseX = mx; mouseY = my;

    if (dragIndex >= 0) {
      isDragging = true;
      charges[dragIndex].x = mx + dragOffsetX;
      charges[dragIndex].y = my + dragOffsetY;
      needsRedraw = true;
    }
  });

  canvas.addEventListener('mouseup', (e) => {
    const [mx, my] = getCanvasPos(e);

    if (dragIndex >= 0 && !isDragging) {
      // click tha drag nahi — kuch nahi karo, charge wahi rahega
      // (placement alag se handle hoga agar click empty space pe ho)
    }

    if (dragIndex < 0 && !isDragging) {
      // empty space pe click — new charge place karo
      const isNegative = e.shiftKey;
      const q = isNegative ? -chargeMagnitude : chargeMagnitude;
      charges.push({ x: mx, y: my, q: q });
      needsRedraw = true;
    }

    dragIndex = -1;
    isDragging = false;
    needsRedraw = true;
  });

  // right-click se charge remove karo
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const [mx, my] = getCanvasPos(e);
    const idx = findChargeAt(mx, my);
    if (idx >= 0) {
      charges.splice(idx, 1);
      needsRedraw = true;
    }
  });

  // double-click se bhi remove
  canvas.addEventListener('dblclick', (e) => {
    e.preventDefault();
    const [mx, my] = getCanvasPos(e);
    const idx = findChargeAt(mx, my);
    if (idx >= 0) {
      charges.splice(idx, 1);
      needsRedraw = true;
    }
  });

  // --- Touch support — mobile pe bhi kaam kare ---
  let touchId = null;
  let touchStartTime = 0;
  let touchMoved = false;

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

    const idx = findChargeAt(mx, my);
    if (idx >= 0) {
      dragIndex = idx;
      dragOffsetX = charges[idx].x - mx;
      dragOffsetY = charges[idx].y - my;
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
        charges[dragIndex].x = mx + dragOffsetX;
        charges[dragIndex].y = my + dragOffsetY;
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

      // agar touch move nahi hua toh new charge place karo
      if (!touchMoved && dragIndex < 0) {
        // long press = negative charge, short tap = positive
        const holdTime = performance.now() - touchStartTime;
        const q = holdTime > 500 ? -chargeMagnitude : chargeMagnitude;
        charges.push({ x: mx, y: my, q: q });
        needsRedraw = true;
      }

      // long press + no drag on existing charge = remove
      if (!touchMoved && dragIndex >= 0) {
        const holdTime = performance.now() - touchStartTime;
        if (holdTime > 600) {
          charges.splice(dragIndex, 1);
          needsRedraw = true;
        }
      }

      dragIndex = -1;
      isDragging = false;
      touchId = null;
      needsRedraw = true;
    }
  }, { passive: false });

  // ============================
  // ANIMATION LOOP — requestAnimationFrame
  // ============================

  function loop() {
    // lab pause: only active sim animates
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
  // lab resume: restart loop when focus released
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });

  // --- Dipole preset se start karo — empty canvas boring lagta hai ---
  loadPreset('Dipole');
  needsRedraw = true;
}
