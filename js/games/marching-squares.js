// ============================================================
// Marching Squares Contour Visualizer — scalar field se contour lines nikaal
// Threshold oscillate karo, contours ko dance karte dekho
// 16 cases, linear interpolation, ambiguous case handling
// ============================================================

// yahi function export hoga — container dhundho, canvas banao, marching squares chalao
export function initMarchingSquares() {
  const container = document.getElementById('marchingSquaresContainer');
  if (!container) {
    console.warn('marchingSquaresContainer nahi mila bhai, marching squares skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';
  const BG_COLOR = '#0a0a0a';

  // resolution presets — cols x rows
  const RESOLUTIONS = {
    'Coarse': { cols: 20, rows: 15 },
    'Medium': { cols: 40, rows: 30 },
    'Fine':   { cols: 80, rows: 60 },
  };

  // --- Perlin noise implementation — simple 2D ---
  // permutation table banao — classic approach
  const PERM = new Uint8Array(512);
  const GRAD = [
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1],
  ];
  (function initPerm() {
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    // Fisher-Yates shuffle
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = p[i]; p[i] = p[j]; p[j] = tmp;
    }
    for (let i = 0; i < 512; i++) PERM[i] = p[i & 255];
  })();

  function fade(t) { return t * t * t * (t * (t * 6 - 15) + 10); }
  function lerp(a, b, t) { return a + t * (b - a); }

  function perlin2D(x, y) {
    const xi = Math.floor(x) & 255;
    const yi = Math.floor(y) & 255;
    const xf = x - Math.floor(x);
    const yf = y - Math.floor(y);
    const u = fade(xf);
    const v = fade(yf);

    const aa = PERM[PERM[xi] + yi];
    const ab = PERM[PERM[xi] + yi + 1];
    const ba = PERM[PERM[xi + 1] + yi];
    const bb = PERM[PERM[xi + 1] + yi + 1];

    function dot(hash, fx, fy) {
      const g = GRAD[hash & 7];
      return g[0] * fx + g[1] * fy;
    }

    const x1 = lerp(dot(aa, xf, yf), dot(ba, xf - 1, yf), u);
    const x2 = lerp(dot(ab, xf, yf - 1), dot(bb, xf - 1, yf - 1), u);
    return lerp(x1, x2, v);
  }

  // --- Marching Squares lookup table ---
  // har case ke liye line segments — edge indices
  // edges: 0=top, 1=right, 2=bottom, 3=left
  // edge midpoint interpolation se exact position milegi
  const EDGE_TABLE = [
    [],              // 0:  0000 — sab neeche
    [[3, 2]],        // 1:  0001 — bottom-left
    [[2, 1]],        // 2:  0010 — bottom-right
    [[3, 1]],        // 3:  0011 — bottom dono
    [[0, 1]],        // 4:  0100 — top-right
    [[0, 1], [2, 3]],// 5:  0101 — saddle (ambiguous)
    [[0, 2]],        // 6:  0110 — right dono
    [[0, 3]],        // 7:  0111 — sirf top-left neeche
    [[3, 0]],        // 8:  1000 — top-left
    [[0, 2]],        // 9:  1001 — left dono
    [[3, 0], [1, 2]],// 10: 1010 — saddle (ambiguous)
    [[0, 1]],        // 11: 1011 — sirf top-right neeche
    [[3, 1]],        // 12: 1100 — top dono
    [[2, 1]],        // 13: 1101 — sirf bottom-right neeche
    [[3, 2]],        // 14: 1110 — sirf bottom-left neeche
    [],              // 15: 1111 — sab upar
  ];

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let animationId = null;
  let isVisible = false;

  // grid settings
  let gridCols = 40;
  let gridRows = 30;
  let threshold = 0.0;
  let fieldType = 'sine';   // sine, circles, perlin, paint
  let animateThreshold = true;

  // display toggles
  let showGrid = true;
  let showCases = false;
  let showValues = false;
  let showBackground = true;

  // scalar field data — (gridCols+1) x (gridRows+1) vertices
  let scalarField = null; // Float32Array
  let paintField = null;  // Float32Array — custom paint mode ka field

  // animation time
  let startTime = performance.now();

  // paint mode state
  let isPainting = false;
  let paintRadius = 3; // grid cells mein radius
  let paintValue = 1;  // +1 raise, -1 lower

  // --- DOM structure banao ---
  const existingChildren = Array.from(container.children);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(167,139,250,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:' + BG_COLOR,
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // label factory
  function createLabel(text) {
    const lbl = document.createElement('span');
    lbl.textContent = text;
    lbl.style.cssText = 'color:#888;font-size:11px;font-family:"JetBrains Mono",monospace;';
    controlsDiv.appendChild(lbl);
    return lbl;
  }

  // select factory — dark themed dropdown
  function createSelect(options, defaultVal, onChange) {
    const sel = document.createElement('select');
    sel.style.cssText = [
      'padding:4px 8px',
      'font-size:11px',
      'border-radius:6px',
      'border:1px solid rgba(167,139,250,0.25)',
      'background:rgba(167,139,250,0.08)',
      'color:#d0d0d0',
      'cursor:pointer',
      'font-family:"JetBrains Mono",monospace',
      'outline:none',
    ].join(';');
    options.forEach(opt => {
      const o = document.createElement('option');
      o.value = opt.value;
      o.textContent = opt.label;
      o.style.background = '#1a1a2e';
      o.style.color = '#d0d0d0';
      if (opt.value === defaultVal) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('change', () => onChange(sel.value));
    controlsDiv.appendChild(sel);
    return sel;
  }

  // toggle button factory
  function createToggle(text, initial, onToggle) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn._active = initial;
    function updateStyle() {
      if (btn._active) {
        btn.style.cssText = [
          'padding:5px 12px',
          'font-size:11px',
          'border-radius:6px',
          'border:1px solid ' + ACCENT,
          'background:rgba(167,139,250,0.3)',
          'color:#fff',
          'cursor:pointer',
          'font-family:"JetBrains Mono",monospace',
          'user-select:none',
          'white-space:nowrap',
        ].join(';');
      } else {
        btn.style.cssText = [
          'padding:5px 12px',
          'font-size:11px',
          'border-radius:6px',
          'border:1px solid rgba(167,139,250,0.25)',
          'background:rgba(167,139,250,0.08)',
          'color:#d0d0d0',
          'cursor:pointer',
          'font-family:"JetBrains Mono",monospace',
          'user-select:none',
          'white-space:nowrap',
        ].join(';');
      }
    }
    updateStyle();
    btn.addEventListener('click', () => {
      btn._active = !btn._active;
      updateStyle();
      onToggle(btn._active);
    });
    controlsDiv.appendChild(btn);
    return btn;
  }

  // separator
  function addSep() {
    const sep = document.createElement('span');
    sep.style.cssText = 'width:1px;height:18px;background:rgba(255,255,255,0.1);margin:0 4px;';
    controlsDiv.appendChild(sep);
  }

  // --- Build controls ---
  createLabel('Field:');
  const fieldSelect = createSelect([
    { value: 'sine', label: 'Sine Waves' },
    { value: 'circles', label: 'Circles' },
    { value: 'perlin', label: 'Perlin Noise' },
    { value: 'paint', label: 'Custom Paint' },
  ], 'sine', (val) => {
    fieldType = val;
    if (val === 'paint') {
      initPaintField();
      canvas.style.cursor = 'pointer';
    } else {
      canvas.style.cursor = 'crosshair';
    }
    generateField();
    requestDraw();
  });

  addSep();

  // threshold slider
  createLabel('Threshold:');
  const threshSlider = document.createElement('input');
  threshSlider.type = 'range';
  threshSlider.min = '-1.0';
  threshSlider.max = '1.0';
  threshSlider.step = '0.01';
  threshSlider.value = '0';
  threshSlider.style.cssText = [
    'width:100px',
    'accent-color:' + ACCENT,
    'cursor:pointer',
  ].join(';');
  controlsDiv.appendChild(threshSlider);

  const threshLabel = document.createElement('span');
  threshLabel.style.cssText = 'color:#d0d0d0;font-size:11px;font-family:"JetBrains Mono",monospace;min-width:38px;';
  threshLabel.textContent = '0.00';
  controlsDiv.appendChild(threshLabel);

  threshSlider.addEventListener('input', () => {
    threshold = parseFloat(threshSlider.value);
    threshLabel.textContent = threshold.toFixed(2);
    // manual slider move kare toh animate off kar do
    if (animateThreshold) {
      animateThreshold = false;
      animateBtn._active = false;
      animateBtn.click(); // toggle fire karega but already false so re-sync
      animateBtn._active = false;
      updateAnimBtnStyle();
    }
    requestDraw();
  });

  addSep();

  // animate threshold toggle
  const animateBtn = createToggle('Animate', true, (active) => {
    animateThreshold = active;
    if (active) {
      startTime = performance.now() - Math.asin(Math.max(-1, Math.min(1, threshold))) * 1000 / (Math.PI * 0.5);
      if (isVisible && !animationId) loop();
    }
  });

  function updateAnimBtnStyle() {
    // re-apply style from _active state
    animateBtn.click();
    animateBtn.click();
  }

  addSep();

  // show toggles
  createToggle('Grid', true, (v) => { showGrid = v; requestDraw(); });
  createToggle('Cases', false, (v) => { showCases = v; requestDraw(); });
  createToggle('Values', false, (v) => { showValues = v; requestDraw(); });
  createToggle('Background', true, (v) => { showBackground = v; requestDraw(); });

  addSep();

  // resolution dropdown
  createLabel('Res:');
  createSelect([
    { value: 'Coarse', label: 'Coarse' },
    { value: 'Medium', label: 'Medium' },
    { value: 'Fine', label: 'Fine' },
  ], 'Medium', (val) => {
    gridCols = RESOLUTIONS[val].cols;
    gridRows = RESOLUTIONS[val].rows;
    if (fieldType === 'paint') initPaintField();
    generateField();
    requestDraw();
  });

  // --- Canvas resize ---
  function resize() {
    dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    canvasW = w;
    canvasH = CANVAS_HEIGHT;
    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // --- Scalar field generation ---
  // field mein (cols+1) * (rows+1) vertices hain
  function generateField() {
    const vw = gridCols + 1;
    const vh = gridRows + 1;
    scalarField = new Float32Array(vw * vh);

    // cell size canvas coordinates mein
    const cellW = canvasW / gridCols;
    const cellH = canvasH / gridRows;

    if (fieldType === 'sine') {
      // sin(x/50) * cos(y/50) — circular interference pattern
      for (let gy = 0; gy < vh; gy++) {
        for (let gx = 0; gx < vw; gx++) {
          const px = gx * cellW;
          const py = gy * cellH;
          scalarField[gy * vw + gx] = Math.sin(px / 50) * Math.cos(py / 50);
        }
      }
    } else if (fieldType === 'circles') {
      // sum of signed distance functions from several circles
      const circles = [
        { cx: canvasW * 0.3, cy: canvasH * 0.4, r: 80 },
        { cx: canvasW * 0.7, cy: canvasH * 0.3, r: 60 },
        { cx: canvasW * 0.5, cy: canvasH * 0.7, r: 70 },
        { cx: canvasW * 0.2, cy: canvasH * 0.8, r: 45 },
        { cx: canvasW * 0.8, cy: canvasH * 0.7, r: 55 },
      ];
      for (let gy = 0; gy < vh; gy++) {
        for (let gx = 0; gx < vw; gx++) {
          const px = gx * cellW;
          const py = gy * cellH;
          let val = 0;
          for (const c of circles) {
            const dx = px - c.cx;
            const dy = py - c.cy;
            const dist = Math.sqrt(dx * dx + dy * dy);
            // signed distance — negative inside, positive outside
            // invert kar rahe hain taaki inside positive ho
            val += Math.max(0, 1.0 - dist / c.r);
          }
          // normalize to roughly -1 to 1 range
          scalarField[gy * vw + gx] = val - 0.5;
        }
      }
    } else if (fieldType === 'perlin') {
      // smooth noise terrain — multi-octave
      const scale = 0.04;
      for (let gy = 0; gy < vh; gy++) {
        for (let gx = 0; gx < vw; gx++) {
          const px = gx * cellW;
          const py = gy * cellH;
          // 2 octaves of perlin
          let val = perlin2D(px * scale, py * scale) * 0.7;
          val += perlin2D(px * scale * 2, py * scale * 2) * 0.3;
          scalarField[gy * vw + gx] = val;
        }
      }
    } else if (fieldType === 'paint') {
      // paint field se copy karo
      const pvw = gridCols + 1;
      const pvh = gridRows + 1;
      for (let i = 0; i < pvw * pvh; i++) {
        scalarField[i] = paintField && i < paintField.length ? paintField[i] : 0;
      }
    }
  }

  // paint field initialize karo — sab zero
  function initPaintField() {
    const vw = gridCols + 1;
    const vh = gridRows + 1;
    paintField = new Float32Array(vw * vh);
  }

  // paint brush — mouse position ke around field values badlo
  function paintAt(canvasX, canvasY, raise) {
    if (fieldType !== 'paint') return;
    const vw = gridCols + 1;
    const vh = gridRows + 1;
    const cellW = canvasW / gridCols;
    const cellH = canvasH / gridRows;

    // canvas position se grid position
    const gx = canvasX / cellW;
    const gy = canvasY / cellH;

    const r = paintRadius;
    const strength = 0.15; // har stroke mein kitna change ho

    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > r) continue;

        const ix = Math.round(gx + dx);
        const iy = Math.round(gy + dy);
        if (ix < 0 || ix >= vw || iy < 0 || iy >= vh) continue;

        // gaussian-ish falloff
        const falloff = 1.0 - dist / r;
        const delta = strength * falloff * (raise ? 1 : -1);
        const idx = iy * vw + ix;
        paintField[idx] = Math.max(-1, Math.min(1, paintField[idx] + delta));
      }
    }

    // paint field se scalar field update karo
    for (let i = 0; i < vw * vh; i++) {
      scalarField[i] = paintField[i];
    }
  }

  // --- Marching Squares core ---
  // edge pe interpolation — exact contour crossing point dhundho
  function interpolateEdge(v1, v2, t) {
    // v1 aur v2 ke beech mein jahan value = threshold ho
    if (Math.abs(v2 - v1) < 1e-10) return 0.5;
    return (t - v1) / (v2 - v1);
  }

  // ek cell ke contour segments nikaal — returns line segments in canvas coords
  function getCellContours(gx, gy, cellW, cellH, thresh) {
    const vw = gridCols + 1;

    // 4 corners ki values — TL, TR, BR, BL order
    const tl = scalarField[gy * vw + gx];
    const tr = scalarField[gy * vw + gx + 1];
    const br = scalarField[(gy + 1) * vw + gx + 1];
    const bl = scalarField[(gy + 1) * vw + gx];

    // case index — 4 bit, TL=8, TR=4, BR=2, BL=1
    let caseIdx = 0;
    if (tl >= thresh) caseIdx |= 8;
    if (tr >= thresh) caseIdx |= 4;
    if (br >= thresh) caseIdx |= 2;
    if (bl >= thresh) caseIdx |= 1;

    // edge table se segments lo
    let edges = EDGE_TABLE[caseIdx];

    // ambiguous cases (5 and 10) — center value se decide karo
    if (caseIdx === 5 || caseIdx === 10) {
      const center = (tl + tr + br + bl) * 0.25;
      if (caseIdx === 5) {
        // case 5: TL below, TR above, BR below, BL above
        // agar center above hai toh connect differently
        if (center >= thresh) {
          edges = [[0, 3], [1, 2]]; // horizontal-ish connection
        }
        // default table entry already has [[0,1],[2,3]]
      } else {
        // case 10: TL above, TR below, BR above, BL below
        if (center >= thresh) {
          edges = [[0, 1], [2, 3]]; // swap connection
        }
        // default table entry already has [[3,0],[1,2]]
      }
    }

    if (edges.length === 0) return { segments: [], caseIdx };

    // cell ka top-left corner canvas mein
    const x0 = gx * cellW;
    const y0 = gy * cellH;

    // edge interpolation points — edge index se position nikaal
    // edge 0 = top (TL->TR), edge 1 = right (TR->BR)
    // edge 2 = bottom (BL->BR), edge 3 = left (TL->BL)
    function edgePoint(edgeIdx) {
      let t;
      switch (edgeIdx) {
        case 0: // top edge — TL to TR
          t = interpolateEdge(tl, tr, thresh);
          return { x: x0 + t * cellW, y: y0 };
        case 1: // right edge — TR to BR
          t = interpolateEdge(tr, br, thresh);
          return { x: x0 + cellW, y: y0 + t * cellH };
        case 2: // bottom edge — BL to BR
          t = interpolateEdge(bl, br, thresh);
          return { x: x0 + t * cellW, y: y0 + cellH };
        case 3: // left edge — TL to BL
          t = interpolateEdge(tl, bl, thresh);
          return { x: x0, y: y0 + t * cellH };
        default:
          return { x: x0, y: y0 };
      }
    }

    const segments = [];
    for (const [e1, e2] of edges) {
      segments.push([edgePoint(e1), edgePoint(e2)]);
    }

    return { segments, caseIdx };
  }

  // --- Color mapping — scalar value to RGB ---
  function valueToColor(val) {
    // -1 to 1 range ko 0-1 mein map kar
    const t = (val + 1) * 0.5;
    const clamped = Math.max(0, Math.min(1, t));

    // blue (low) -> cyan -> green -> yellow -> red (high)
    let r, g, b;
    if (clamped < 0.25) {
      const s = clamped / 0.25;
      r = 10; g = Math.floor(30 + s * 100); b = Math.floor(120 + s * 60);
    } else if (clamped < 0.5) {
      const s = (clamped - 0.25) / 0.25;
      r = Math.floor(10 + s * 60); g = Math.floor(130 + s * 60); b = Math.floor(180 - s * 80);
    } else if (clamped < 0.75) {
      const s = (clamped - 0.5) / 0.25;
      r = Math.floor(70 + s * 130); g = Math.floor(190 - s * 30); b = Math.floor(100 - s * 70);
    } else {
      const s = (clamped - 0.75) / 0.25;
      r = Math.floor(200 + s * 55); g = Math.floor(160 - s * 120); b = Math.floor(30 - s * 20);
    }
    return { r, g, b };
  }

  // --- Drawing ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, canvasW, canvasH);

    if (!scalarField || canvasW <= 0) return;

    const cellW = canvasW / gridCols;
    const cellH = canvasH / gridRows;
    const vw = gridCols + 1;
    const vh = gridRows + 1;

    // Layer 1: background color-mapped scalar field
    if (showBackground) {
      // har cell ko color karo based on average vertex value
      for (let gy = 0; gy < gridRows; gy++) {
        for (let gx = 0; gx < gridCols; gx++) {
          const tl = scalarField[gy * vw + gx];
          const tr = scalarField[gy * vw + gx + 1];
          const br = scalarField[(gy + 1) * vw + gx + 1];
          const bl = scalarField[(gy + 1) * vw + gx];
          const avg = (tl + tr + br + bl) * 0.25;
          const c = valueToColor(avg);
          ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},0.4)`;
          ctx.fillRect(gx * cellW, gy * cellH, cellW + 0.5, cellH + 0.5);
        }
      }
    }

    // Layer 2: grid lines
    if (showGrid) {
      ctx.strokeStyle = 'rgba(255,255,255,0.07)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      // vertical lines
      for (let gx = 0; gx <= gridCols; gx++) {
        const x = gx * cellW;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvasH);
      }
      // horizontal lines
      for (let gy = 0; gy <= gridRows; gy++) {
        const y = gy * cellH;
        ctx.moveTo(0, y);
        ctx.lineTo(canvasW, y);
      }
      ctx.stroke();
    }

    // Layer 3: vertex markers — threshold ke hisaab se color
    // sirf jab cells zyada bade ho (coarse/medium) tab dikha
    if (showGrid && gridCols <= 40) {
      for (let gy = 0; gy < vh; gy++) {
        for (let gx = 0; gx < vw; gx++) {
          const val = scalarField[gy * vw + gx];
          const above = val >= threshold;
          const x = gx * cellW;
          const y = gy * cellH;

          ctx.beginPath();
          ctx.arc(x, y, above ? 2.5 : 1.5, 0, Math.PI * 2);
          ctx.fillStyle = above
            ? 'rgba(167,139,250,0.8)'
            : 'rgba(100,100,100,0.5)';
          ctx.fill();
        }
      }
    }

    // Layer 4: case numbers — har cell mein case index likho
    if (showCases && gridCols <= 40) {
      ctx.font = (cellW > 20 ? 9 : 7) + 'px "JetBrains Mono",monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let gy = 0; gy < gridRows; gy++) {
        for (let gx = 0; gx < gridCols; gx++) {
          const { caseIdx } = getCellContours(gx, gy, cellW, cellH, threshold);
          const cx = gx * cellW + cellW * 0.5;
          const cy = gy * cellH + cellH * 0.5;
          // edge cases bright, middle cases dim
          const isBoundary = caseIdx > 0 && caseIdx < 15;
          ctx.fillStyle = isBoundary
            ? 'rgba(255,255,255,0.6)'
            : 'rgba(255,255,255,0.15)';
          ctx.fillText(caseIdx.toString(), cx, cy);
        }
      }
    }

    // Layer 4b: vertex values — har vertex pe value likho
    if (showValues && gridCols <= 40) {
      ctx.font = (cellW > 20 ? 8 : 6) + 'px "JetBrains Mono",monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      for (let gy = 0; gy < vh; gy++) {
        for (let gx = 0; gx < vw; gx++) {
          const val = scalarField[gy * vw + gx];
          const x = gx * cellW;
          const y = gy * cellH - 4;
          ctx.fillStyle = val >= threshold
            ? 'rgba(167,139,250,0.6)'
            : 'rgba(150,150,150,0.4)';
          ctx.fillText(val.toFixed(1), x, y);
        }
      }
    }

    // Layer 5: contour lines — ye hai asli kamal
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // glow effect ke liye do passes — pehle thick dim, fir thin bright
    for (let pass = 0; pass < 2; pass++) {
      if (pass === 0) {
        // glow pass
        ctx.strokeStyle = 'rgba(100,240,255,0.3)';
        ctx.lineWidth = gridCols <= 40 ? 4 : 2.5;
      } else {
        // sharp pass
        ctx.strokeStyle = 'rgba(150,255,255,0.95)';
        ctx.lineWidth = gridCols <= 40 ? 2 : 1.2;
      }

      ctx.beginPath();
      for (let gy = 0; gy < gridRows; gy++) {
        for (let gx = 0; gx < gridCols; gx++) {
          const { segments } = getCellContours(gx, gy, cellW, cellH, threshold);
          for (const [p1, p2] of segments) {
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
          }
        }
      }
      ctx.stroke();
    }

    // info text — top-right corner
    ctx.font = '11px "JetBrains Mono",monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(167,139,250,0.5)';
    ctx.fillText(`${gridCols}×${gridRows}  θ=${threshold.toFixed(2)}`, canvasW - 10, 10);
    if (fieldType === 'paint') {
      ctx.fillStyle = 'rgba(167,139,250,0.35)';
      ctx.fillText('Left: raise | Right: lower', canvasW - 10, 26);
    }
    if (animateThreshold) {
      ctx.fillStyle = 'rgba(167,139,250,0.3)';
      ctx.fillText('animating...', canvasW - 10, fieldType === 'paint' ? 42 : 26);
    }
  }

  // --- Draw request — debounced ---
  let drawPending = false;
  function requestDraw() {
    if (!drawPending) {
      drawPending = true;
      requestAnimationFrame(() => {
        drawPending = false;
        draw();
      });
    }
  }

  // --- Animation loop ---
  function loop(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    if (!timestamp) timestamp = performance.now();

    if (animateThreshold) {
      // sinusoidal oscillation — smooth aur mesmerizing
      const elapsed = (timestamp - startTime) / 1000;
      const speed = 0.5; // cycles per second
      threshold = Math.sin(elapsed * Math.PI * speed) * 0.8;
      // slider bhi update kar
      threshSlider.value = threshold.toFixed(2);
      threshLabel.textContent = threshold.toFixed(2);
    }

    draw();

    animationId = requestAnimationFrame(loop);
  }

  // --- Mouse/Touch events ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvasW / rect.width),
      y: (e.clientY - rect.top) * (canvasH / rect.height),
    };
  }

  canvas.addEventListener('mousedown', (e) => {
    if (fieldType !== 'paint') return;
    e.preventDefault();
    isPainting = true;
    // left click = raise, right click = lower
    paintValue = e.button === 2 ? -1 : 1;
    const pos = getCanvasPos(e);
    paintAt(pos.x, pos.y, paintValue > 0);
    requestDraw();
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isPainting || fieldType !== 'paint') return;
    e.preventDefault();
    const pos = getCanvasPos(e);
    paintAt(pos.x, pos.y, paintValue > 0);
    requestDraw();
  });

  canvas.addEventListener('mouseup', () => {
    isPainting = false;
  });

  canvas.addEventListener('mouseleave', () => {
    isPainting = false;
  });

  // right-click prevent default — paint mode mein context menu nahi chahiye
  canvas.addEventListener('contextmenu', (e) => {
    if (fieldType === 'paint') e.preventDefault();
  });

  // touch support for paint mode
  canvas.addEventListener('touchstart', (e) => {
    if (fieldType !== 'paint') return;
    e.preventDefault();
    isPainting = true;
    paintValue = 1; // touch = always raise
    if (e.touches.length > 0) {
      const rect = canvas.getBoundingClientRect();
      const pos = {
        x: (e.touches[0].clientX - rect.left) * (canvasW / rect.width),
        y: (e.touches[0].clientY - rect.top) * (canvasH / rect.height),
      };
      paintAt(pos.x, pos.y, true);
      requestDraw();
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    if (!isPainting || fieldType !== 'paint') return;
    e.preventDefault();
    if (e.touches.length > 0) {
      const rect = canvas.getBoundingClientRect();
      const pos = {
        x: (e.touches[0].clientX - rect.left) * (canvasW / rect.width),
        y: (e.touches[0].clientY - rect.top) * (canvasH / rect.height),
      };
      paintAt(pos.x, pos.y, true);
      requestDraw();
    }
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    isPainting = false;
  }, { passive: false });

  // --- IntersectionObserver — sirf visible hone pe render karo ---
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) {
      resize();
      generateField();
      startTime = performance.now();
      loop();
    } else if (!isVisible && animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }, { threshold: 0.1 });
  obs.observe(container);
  // lab resume: restart loop when focus released
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });

  // --- Resize handler ---
  window.addEventListener('resize', () => {
    if (!isVisible) return;
    resize();
    generateField();
    requestDraw();
  });

  // --- Initial setup ---
  resize();
  generateField();
  requestDraw();
}
