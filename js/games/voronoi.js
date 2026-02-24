// ============================================================
// Interactive Voronoi Diagram — Seeds drag karo, cells morph hote dekhoge
// Click se naye seeds add karo, drag se reshape, double-click se hatao
// JFA nahi use kar rahe — reduced resolution brute force + upscale
// ============================================================

// yahi function export hoga — container dhundho, canvas banao, Voronoi chalao
export function initVoronoi() {
  const container = document.getElementById('voronoiContainer');
  if (!container) {
    console.warn('voronoiContainer nahi mila bhai, Voronoi demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';       // purple accent
  const ACCENT_RGB = '167,139,250';
  const BG_COLOR = '#0a0a0a';
  const BORDER_COLOR = '#1a1a2e';
  const MAX_SEEDS = 40;
  const SEED_RADIUS = 6;
  const DRAG_THRESHOLD = 8;       // pixels — itna move kare toh drag maano
  const COMPUTE_WIDTH = 200;      // low-res buffer width — performance ke liye
  const DRIFT_SPEED = 0.3;        // animate mode mein seeds kitna hile

  // seed colors — HSL evenly spaced, muted/pastel feel
  function seedHSL(index, total) {
    const hue = (index * 360 / total) % 360;
    return { h: hue, s: 65, l: 55 };
  }

  // HSL to RGB — pixel manipulation ke liye chahiye
  function hslToRGB(h, s, l) {
    const h2 = h / 360;
    const s2 = s / 100;
    const l2 = l / 100;
    let r, g, b;
    if (s2 === 0) {
      r = g = b = l2;
    } else {
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };
      const q = l2 < 0.5 ? l2 * (1 + s2) : l2 + s2 - l2 * s2;
      const p = 2 * l2 - q;
      r = hue2rgb(p, q, h2 + 1 / 3);
      g = hue2rgb(p, q, h2);
      b = hue2rgb(p, q, h2 - 1 / 3);
    }
    return {
      r: Math.round(r * 255),
      g: Math.round(g * 255),
      b: Math.round(b * 255),
    };
  }

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let seeds = [];           // [{x, y, hue, driftAngle, driftRadius}]
  let animationId = null;
  let isVisible = false;
  let voronoiDirty = true;  // recompute flag
  let isAnimating = false;  // seeds drift toggle
  let showEdges = true;     // cell borders toggle
  let showDelaunay = false; // triangulation overlay toggle
  let lastTime = 0;

  // drag state — seed ko pakad ke ghaseeto
  let dragSeedIdx = -1;
  let isDragging = false;
  let mouseDownPos = { x: 0, y: 0 };
  let mouseDownTime = 0;

  // low-res voronoi buffer — har pixel ka nearest seed ID
  let voronoiBuffer = null; // Uint16Array — seedIdx per pixel
  let bufferW = 0, bufferH = 0;

  // adjacency set — Delaunay ke liye
  let adjacency = new Set();

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

  // offscreen canvas — voronoi render ke liye
  const offCanvas = document.createElement('canvas');
  const offCtx = offCanvas.getContext('2d');

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

  // button factory — consistent dark theme styling
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:11px',
      'border-radius:6px',
      'border:1px solid rgba(167,139,250,0.25)',
      'background:rgba(167,139,250,0.08)',
      'color:#d0d0d0',
      'cursor:pointer',
      'font-family:"JetBrains Mono",monospace',
      'transition:background 0.15s,border-color 0.15s',
      'user-select:none',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(167,139,250,0.2)';
        btn.style.borderColor = 'rgba(167,139,250,0.5)';
      }
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(167,139,250,0.08)';
        btn.style.borderColor = 'rgba(167,139,250,0.25)';
      }
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // toggle button factory — on/off state track kare
  function createToggle(text, initial, onToggle) {
    const btn = createButton(text, () => {
      btn._active = !btn._active;
      updateToggleStyle(btn);
      onToggle(btn._active);
    });
    btn._active = initial;
    updateToggleStyle(btn);
    return btn;
  }

  function updateToggleStyle(btn) {
    if (btn._active) {
      btn.style.background = 'rgba(167,139,250,0.3)';
      btn.style.borderColor = ACCENT;
      btn.style.color = '#fff';
    } else {
      btn.style.background = 'rgba(167,139,250,0.08)';
      btn.style.borderColor = 'rgba(167,139,250,0.25)';
      btn.style.color = '#d0d0d0';
    }
  }

  // separator helper
  function addSep() {
    const sep = document.createElement('span');
    sep.style.cssText = 'width:1px;height:18px;background:rgba(255,255,255,0.1);margin:0 4px;';
    controlsDiv.appendChild(sep);
  }

  // buttons banao
  const addRandomBtn = createButton('Add Random', () => {
    addRandomSeeds(5);
    voronoiDirty = true;
    requestDraw();
  });

  const clearBtn = createButton('Clear', () => {
    seeds = [];
    voronoiDirty = true;
    adjacency.clear();
    requestDraw();
  });

  addSep();

  const animateBtn = createToggle('Animate', false, (active) => {
    isAnimating = active;
    if (active && isVisible && !animationId) {
      lastTime = performance.now();
      loop();
    }
  });

  const delaunayBtn = createToggle('Show Delaunay', false, (active) => {
    showDelaunay = active;
    voronoiDirty = true;
    requestDraw();
  });

  const edgesBtn = createToggle('Show Edges', true, (active) => {
    showEdges = active;
    voronoiDirty = true;
    requestDraw();
  });

  addSep();

  // preset dropdown
  const presetLabel = document.createElement('span');
  presetLabel.textContent = 'Preset:';
  presetLabel.style.cssText = 'color:#888;font-size:11px;font-family:"JetBrains Mono",monospace;';
  controlsDiv.appendChild(presetLabel);

  const presetSelect = document.createElement('select');
  presetSelect.style.cssText = [
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
  ['Random', 'Grid', 'Circle', 'Spiral'].forEach(name => {
    const opt = document.createElement('option');
    opt.value = name.toLowerCase();
    opt.textContent = name;
    opt.style.background = '#1a1a2e';
    opt.style.color = '#d0d0d0';
    presetSelect.appendChild(opt);
  });
  presetSelect.addEventListener('change', () => {
    loadPreset(presetSelect.value);
  });
  controlsDiv.appendChild(presetSelect);

  // stats row
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'gap:16px',
    'margin-top:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:#888',
    'flex-wrap:wrap',
    'align-items:center',
  ].join(';');
  container.appendChild(statsDiv);

  const statSeeds = document.createElement('span');
  const statCells = document.createElement('span');
  const statHint = document.createElement('span');
  statsDiv.appendChild(statSeeds);
  statsDiv.appendChild(statCells);
  statsDiv.appendChild(statHint);

  function updateStats() {
    statSeeds.textContent = 'Seeds: ' + seeds.length + '/' + MAX_SEEDS;
    statCells.textContent = seeds.length > 0 ? ('Cells: ' + seeds.length) : '';
    statHint.textContent = seeds.length === 0
      ? 'Click to add seeds'
      : 'Click: add | Drag: move | Dbl-click: remove';
    statHint.style.color = seeds.length === 0 ? 'rgba(167,139,250,0.5)' : '#555';
  }

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

    // offscreen canvas bhi resize
    offCanvas.width = canvas.width;
    offCanvas.height = canvas.height;
    offCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // low-res buffer dimensions recalculate
    bufferW = COMPUTE_WIDTH;
    bufferH = Math.round(CANVAS_HEIGHT * (COMPUTE_WIDTH / w));
    voronoiBuffer = new Uint16Array(bufferW * bufferH);

    voronoiDirty = true;
  }

  // --- Seed management ---

  function createSeed(x, y) {
    return {
      x: x,
      y: y,
      // animation ke liye — drift parameters
      driftAngle: Math.random() * Math.PI * 2,
      driftRadius: 20 + Math.random() * 40,
      driftSpeed: 0.2 + Math.random() * 0.4,
      // base position — animate mode mein iske around ghoomega
      baseX: x,
      baseY: y,
    };
  }

  function addRandomSeeds(count) {
    const pad = 30;
    for (let i = 0; i < count; i++) {
      if (seeds.length >= MAX_SEEDS) break;
      const x = pad + Math.random() * (canvasW - 2 * pad);
      const y = pad + Math.random() * (canvasH - 2 * pad);
      seeds.push(createSeed(x, y));
    }
    voronoiDirty = true;
    updateStats();
  }

  // --- Preset layouts ---
  function loadPreset(type) {
    seeds = [];
    adjacency.clear();
    const pad = 40;
    const w = canvasW - 2 * pad;
    const h = canvasH - 2 * pad;
    const cx = canvasW / 2;
    const cy = canvasH / 2;

    if (type === 'random') {
      // 18 random seeds
      for (let i = 0; i < 18; i++) {
        const x = pad + Math.random() * w;
        const y = pad + Math.random() * h;
        seeds.push(createSeed(x, y));
      }
    } else if (type === 'grid') {
      // 4x4 grid — 16 seeds evenly spaced
      const cols = 4, rows = 4;
      const spacingX = w / (cols + 1);
      const spacingY = h / (rows + 1);
      for (let r = 1; r <= rows; r++) {
        for (let c = 1; c <= cols; c++) {
          // thoda jitter add kar — perfect grid boring lagta hai
          const jx = (Math.random() - 0.5) * spacingX * 0.3;
          const jy = (Math.random() - 0.5) * spacingY * 0.3;
          seeds.push(createSeed(pad + c * spacingX + jx, pad + r * spacingY + jy));
        }
      }
    } else if (type === 'circle') {
      // 12 seeds circle mein + 1 center mein
      const radius = Math.min(w, h) * 0.35;
      seeds.push(createSeed(cx, cy)); // center seed
      for (let i = 0; i < 12; i++) {
        const angle = (i / 12) * Math.PI * 2;
        const x = cx + Math.cos(angle) * radius;
        const y = cy + Math.sin(angle) * radius;
        seeds.push(createSeed(x, y));
      }
    } else if (type === 'spiral') {
      // Archimedean spiral — 20 seeds
      const turns = 2.5;
      const maxR = Math.min(w, h) * 0.4;
      const count = 20;
      for (let i = 0; i < count; i++) {
        const t = i / (count - 1);
        const angle = t * turns * Math.PI * 2;
        const r = t * maxR;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        seeds.push(createSeed(x, y));
      }
    }

    voronoiDirty = true;
    updateStats();
    requestDraw();
  }

  // --- Voronoi computation — low-res brute force ---
  // har pixel pe nearest seed dhundho, buffer mein store karo
  function computeVoronoi() {
    if (seeds.length === 0) {
      voronoiDirty = false;
      return;
    }

    const scaleX = canvasW / bufferW;
    const scaleY = canvasH / bufferH;

    // brute force — har buffer pixel ke liye nearest seed
    for (let by = 0; by < bufferH; by++) {
      const realY = (by + 0.5) * scaleY;
      for (let bx = 0; bx < bufferW; bx++) {
        const realX = (bx + 0.5) * scaleX;
        let minDist = Infinity;
        let minIdx = 0;
        for (let s = 0; s < seeds.length; s++) {
          const dx = realX - seeds[s].x;
          const dy = realY - seeds[s].y;
          const d = dx * dx + dy * dy;
          if (d < minDist) {
            minDist = d;
            minIdx = s;
          }
        }
        voronoiBuffer[by * bufferW + bx] = minIdx;
      }
    }

    // adjacency build kar — Delaunay ke liye
    // adjacent pixels jinka seed different hai, woh seeds neighbors hain
    adjacency.clear();
    for (let by = 0; by < bufferH; by++) {
      for (let bx = 0; bx < bufferW; bx++) {
        const idx = voronoiBuffer[by * bufferW + bx];
        // right neighbor
        if (bx + 1 < bufferW) {
          const rIdx = voronoiBuffer[by * bufferW + bx + 1];
          if (rIdx !== idx) {
            // sorted pair store kar — duplicates avoid
            const key = idx < rIdx ? (idx * MAX_SEEDS + rIdx) : (rIdx * MAX_SEEDS + idx);
            adjacency.add(key);
          }
        }
        // bottom neighbor
        if (by + 1 < bufferH) {
          const bIdx = voronoiBuffer[(by + 1) * bufferW + bx];
          if (bIdx !== idx) {
            const key = idx < bIdx ? (idx * MAX_SEEDS + bIdx) : (bIdx * MAX_SEEDS + idx);
            adjacency.add(key);
          }
        }
      }
    }

    voronoiDirty = false;
  }

  // --- Voronoi render — low-res buffer se full canvas pe ---
  function renderVoronoi() {
    if (seeds.length === 0) return;

    // offscreen canvas pe draw — ImageData use karenge
    const imgData = offCtx.createImageData(bufferW, bufferH);
    const data = imgData.data;

    // precompute seed colors
    const rgbs = seeds.map((_, i) => {
      const c = seedHSL(i, seeds.length);
      // muted/pastel version — cell fill ke liye
      return hslToRGB(c.h, c.s * 0.5, c.l * 0.6);
    });

    // cell fills — har pixel apne seed ka color le
    for (let by = 0; by < bufferH; by++) {
      for (let bx = 0; bx < bufferW; bx++) {
        const seedIdx = voronoiBuffer[by * bufferW + bx];
        const rgb = rgbs[seedIdx];
        const pi = (by * bufferW + bx) * 4;
        data[pi] = rgb.r;
        data[pi + 1] = rgb.g;
        data[pi + 2] = rgb.b;
        data[pi + 3] = 80; // alpha — muted fill
      }
    }

    // edges detect kar — jahan neighbor ka seed different hai
    if (showEdges) {
      for (let by = 0; by < bufferH; by++) {
        for (let bx = 0; bx < bufferW; bx++) {
          const idx = voronoiBuffer[by * bufferW + bx];
          let isEdge = false;
          // right check
          if (bx + 1 < bufferW && voronoiBuffer[by * bufferW + bx + 1] !== idx) isEdge = true;
          // bottom check
          if (!isEdge && by + 1 < bufferH && voronoiBuffer[(by + 1) * bufferW + bx] !== idx) isEdge = true;
          // left check — smoother edges ke liye
          if (!isEdge && bx > 0 && voronoiBuffer[by * bufferW + bx - 1] !== idx) isEdge = true;
          // top check
          if (!isEdge && by > 0 && voronoiBuffer[(by - 1) * bufferW + bx] !== idx) isEdge = true;

          if (isEdge) {
            const pi = (by * bufferW + bx) * 4;
            // bright edge — white with decent alpha
            data[pi] = 255;
            data[pi + 1] = 255;
            data[pi + 2] = 255;
            data[pi + 3] = 100;
          }
        }
      }
    }

    // ImageData ko offscreen canvas pe put karo (low-res)
    // pehle offscreen canvas ko buffer size mein set karo
    offCanvas.width = bufferW;
    offCanvas.height = bufferH;
    offCtx.putImageData(imgData, 0, 0);
  }

  // --- Drawing ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // dark background
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, canvasW, canvasH);

    if (seeds.length > 0) {
      // voronoi recompute agar dirty hai
      if (voronoiDirty) {
        computeVoronoi();
        renderVoronoi();
      }

      // offscreen canvas se main canvas pe draw — nearest-neighbor upscale
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offCanvas, 0, 0, bufferW, bufferH, 0, 0, canvasW, canvasH);
      ctx.imageSmoothingEnabled = true;

      // Delaunay triangulation overlay — seeds ke beech lines
      if (showDelaunay) {
        ctx.strokeStyle = 'rgba(167,139,250,0.35)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        for (const key of adjacency) {
          const i = Math.floor(key / MAX_SEEDS);
          const j = key % MAX_SEEDS;
          if (i < seeds.length && j < seeds.length) {
            ctx.beginPath();
            ctx.moveTo(seeds[i].x, seeds[i].y);
            ctx.lineTo(seeds[j].x, seeds[j].y);
            ctx.stroke();
          }
        }
        ctx.setLineDash([]);
      }

      // seed points draw kar — bright dots with glow
      for (let i = 0; i < seeds.length; i++) {
        const s = seeds[i];
        const c = seedHSL(i, seeds.length);
        const rgb = hslToRGB(c.h, c.s, c.l);
        const colorStr = `rgb(${rgb.r},${rgb.g},${rgb.b})`;

        // glow effect
        ctx.shadowColor = colorStr;
        ctx.shadowBlur = 12;

        // filled circle
        ctx.beginPath();
        ctx.arc(s.x, s.y, SEED_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = colorStr;
        ctx.fill();

        // inner highlight
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.arc(s.x, s.y, SEED_RADIUS * 0.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,0.4)`;
        ctx.fill();

        // border ring
        ctx.beginPath();
        ctx.arc(s.x, s.y, SEED_RADIUS, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255,255,255,0.3)`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.shadowBlur = 0;
    }

    // hint text agar koi seeds nahi hain
    if (seeds.length === 0) {
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(167,139,250,0.35)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Click to add seed points, or choose a preset', canvasW / 2, canvasH / 2);
    }

    // seed count badge — top-right
    if (seeds.length > 0) {
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillStyle = 'rgba(167,139,250,0.5)';
      ctx.fillText(seeds.length + ' seeds', canvasW - 10, 10);
      if (isAnimating) {
        ctx.fillStyle = 'rgba(167,139,250,0.3)';
        ctx.fillText('animating...', canvasW - 10, 26);
      }
    }
  }

  // draw request — debounced
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

  // --- Animation loop — seeds drift karte hain ---
  function loop(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    if (!timestamp) timestamp = performance.now();
    const dt = Math.min((timestamp - lastTime) / 1000, 0.05); // cap at 50ms
    lastTime = timestamp;

    if (isAnimating && seeds.length > 0) {
      // seeds ko orbit kara — base position ke around
      for (const s of seeds) {
        s.driftAngle += s.driftSpeed * dt;
        s.x = s.baseX + Math.cos(s.driftAngle) * s.driftRadius * DRIFT_SPEED;
        s.y = s.baseY + Math.sin(s.driftAngle * 0.7) * s.driftRadius * DRIFT_SPEED;

        // canvas boundaries mein clamp kar
        s.x = Math.max(5, Math.min(canvasW - 5, s.x));
        s.y = Math.max(5, Math.min(canvasH - 5, s.y));
      }
      voronoiDirty = true;
    }

    draw();

    // animation tab — agar animate on hai ya visibility change hua toh loop chale
    if (isAnimating) {
      animationId = requestAnimationFrame(loop);
    } else {
      animationId = null;
    }
  }

  // --- Mouse/Touch interaction ---

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvasW / rect.width),
      y: (e.clientY - rect.top) * (canvasH / rect.height),
    };
  }

  function getTouchPos(touch) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (touch.clientX - rect.left) * (canvasW / rect.width),
      y: (touch.clientY - rect.top) * (canvasH / rect.height),
    };
  }

  // nearest seed dhundho given position se
  function findNearestSeed(px, py, maxDist) {
    let minDist = maxDist * maxDist;
    let minIdx = -1;
    for (let i = 0; i < seeds.length; i++) {
      const dx = px - seeds[i].x;
      const dy = py - seeds[i].y;
      const d = dx * dx + dy * dy;
      if (d < minDist) {
        minDist = d;
        minIdx = i;
      }
    }
    return minIdx;
  }

  // --- Mouse events ---
  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    mouseDownPos = pos;
    mouseDownTime = performance.now();

    // check karo koi seed paas mein hai kya
    const nearIdx = findNearestSeed(pos.x, pos.y, SEED_RADIUS * 3);
    if (nearIdx >= 0) {
      dragSeedIdx = nearIdx;
      isDragging = false; // abhi drag confirm nahi hua
      canvas.style.cursor = 'grabbing';
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (dragSeedIdx < 0) return;
    e.preventDefault();
    const pos = getCanvasPos(e);
    const dx = pos.x - mouseDownPos.x;
    const dy = pos.y - mouseDownPos.y;

    // drag threshold check — slight move ko click maano
    if (!isDragging && (dx * dx + dy * dy) > DRAG_THRESHOLD * DRAG_THRESHOLD) {
      isDragging = true;
    }

    if (isDragging) {
      // seed ko move karo
      seeds[dragSeedIdx].x = Math.max(5, Math.min(canvasW - 5, pos.x));
      seeds[dragSeedIdx].y = Math.max(5, Math.min(canvasH - 5, pos.y));
      // base position bhi update karo — animate mode ke liye
      seeds[dragSeedIdx].baseX = seeds[dragSeedIdx].x;
      seeds[dragSeedIdx].baseY = seeds[dragSeedIdx].y;
      voronoiDirty = true;
      requestDraw();
    }
  });

  canvas.addEventListener('mouseup', (e) => {
    const wasDragging = isDragging;
    const wasDragSeed = dragSeedIdx;
    isDragging = false;
    dragSeedIdx = -1;
    canvas.style.cursor = 'crosshair';

    // agar drag nahi hua tha toh click event handle karo
    if (!wasDragging && wasDragSeed < 0) {
      // koi seed select nahi tha — naya seed add kar
      if (seeds.length < MAX_SEEDS) {
        const pos = getCanvasPos(e);
        seeds.push(createSeed(pos.x, pos.y));
        voronoiDirty = true;
        updateStats();
        requestDraw();
      }
    }
  });

  // double-click — seed hatao
  canvas.addEventListener('dblclick', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const nearIdx = findNearestSeed(pos.x, pos.y, SEED_RADIUS * 4);
    if (nearIdx >= 0) {
      seeds.splice(nearIdx, 1);
      voronoiDirty = true;
      updateStats();
      requestDraw();
    }
  });

  // --- Touch events — mobile support ---
  let touchStartTime = 0;
  let lastTapTime = 0;
  let touchSeedIdx = -1;
  let isTouchDragging = false;

  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (e.touches.length !== 1) return;
    const pos = getTouchPos(e.touches[0]);
    mouseDownPos = pos;
    touchStartTime = performance.now();

    const nearIdx = findNearestSeed(pos.x, pos.y, SEED_RADIUS * 4);
    if (nearIdx >= 0) {
      touchSeedIdx = nearIdx;
      isTouchDragging = false;
    } else {
      touchSeedIdx = -1;
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (e.touches.length !== 1 || touchSeedIdx < 0) return;
    const pos = getTouchPos(e.touches[0]);
    const dx = pos.x - mouseDownPos.x;
    const dy = pos.y - mouseDownPos.y;

    if (!isTouchDragging && (dx * dx + dy * dy) > DRAG_THRESHOLD * DRAG_THRESHOLD) {
      isTouchDragging = true;
    }

    if (isTouchDragging) {
      seeds[touchSeedIdx].x = Math.max(5, Math.min(canvasW - 5, pos.x));
      seeds[touchSeedIdx].y = Math.max(5, Math.min(canvasH - 5, pos.y));
      seeds[touchSeedIdx].baseX = seeds[touchSeedIdx].x;
      seeds[touchSeedIdx].baseY = seeds[touchSeedIdx].y;
      voronoiDirty = true;
      requestDraw();
    }
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    const now = performance.now();

    if (!isTouchDragging) {
      // double-tap detection — seed remove
      if (now - lastTapTime < 300) {
        const pos = mouseDownPos;
        const nearIdx = findNearestSeed(pos.x, pos.y, SEED_RADIUS * 4);
        if (nearIdx >= 0) {
          seeds.splice(nearIdx, 1);
          voronoiDirty = true;
          updateStats();
          requestDraw();
          lastTapTime = 0;
          touchSeedIdx = -1;
          isTouchDragging = false;
          return;
        }
      }
      lastTapTime = now;

      // single tap — naya seed add (agar koi seed select nahi tha)
      if (touchSeedIdx < 0 && seeds.length < MAX_SEEDS) {
        const pos = mouseDownPos;
        seeds.push(createSeed(pos.x, pos.y));
        voronoiDirty = true;
        updateStats();
        requestDraw();
      }
    }

    touchSeedIdx = -1;
    isTouchDragging = false;
  }, { passive: false });

  // --- IntersectionObserver — sirf visible hone pe render karo ---
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) {
      resize();
      voronoiDirty = true;
      lastTime = performance.now();
      if (isAnimating) {
        loop();
      } else {
        requestDraw();
      }
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
    // seeds ko scale karna padega naye size ke hisaab se
    const oldW = canvasW;
    const oldH = canvasH;
    resize();

    if (oldW > 0 && oldH > 0) {
      const scaleX = canvasW / oldW;
      for (const s of seeds) {
        s.x *= scaleX;
        s.baseX *= scaleX;
        // y same rehta hai kyunki height fixed hai
      }
    }

    voronoiDirty = true;
    requestDraw();
  });

  // --- Initial setup ---
  resize();
  // default mein 15 random seeds se shuru kar
  addRandomSeeds(15);
  updateStats();
  requestDraw();
}
