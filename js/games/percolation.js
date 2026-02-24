// ============================================================
// Percolation — Phase transition ka visual demo
// NxN grid pe random cells occupy karo, Union-Find se clusters dhundho
// p_c ≈ 0.5927 pe spanning cluster ban jaata hai (top to bottom connect)
// Critical threshold — physics mein bahut important concept hai ye
// ============================================================

// yahi se shuru — grid bharo, clusters dhundho, gold mein spanning cluster dikhao
export function initPercolation() {
  const container = document.getElementById('percolationContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  const ACCENT_RGB = '245,158,11';
  const BG = '#111';
  const FONT = "'JetBrains Mono',monospace";
  const GRID_COLS = 80;
  const GRID_ROWS = 60;

  // --- State ---
  let animationId = null, isVisible = false, canvasW = 0;
  let probability = 0.59;        // occupation probability
  let showSpanning = true;       // spanning cluster highlight karo ya nahi
  let grid = [];                 // 0 = empty, 1 = occupied
  let parent = [];               // Union-Find ka parent array
  let rank = [];                 // Union-Find ka rank array
  let clusterColors = {};        // root -> color mapping
  let spanningRoot = -1;         // spanning cluster ka root (-1 = nahi hai)
  let clusterCount = 0;          // total clusters

  // --- DOM setup ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:${BG};border:1px solid rgba(${ACCENT_RGB},0.15);image-rendering:pixelated;image-rendering:crisp-edges;`;
  canvas.width = GRID_COLS;
  canvas.height = GRID_ROWS;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(GRID_COLS, GRID_ROWS);
  const pixels = imageData.data;

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  // --- Helpers ---
  function makeSlider(label, min, max, step, val, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;align-items:center;gap:6px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = `color:#6b6b6b;font-size:11px;font-family:${FONT};white-space:nowrap;`;
    lbl.textContent = label;
    wrap.appendChild(lbl);
    const slider = document.createElement('input');
    slider.type = 'range'; slider.min = String(min); slider.max = String(max);
    slider.step = String(step); slider.value = String(val);
    slider.style.cssText = `width:120px;height:4px;accent-color:${ACCENT};cursor:pointer;`;
    wrap.appendChild(slider);
    const vSpan = document.createElement('span');
    vSpan.style.cssText = `color:#f0f0f0;font-size:11px;font-family:${FONT};min-width:36px;`;
    vSpan.textContent = Number(val).toFixed(4);
    wrap.appendChild(vSpan);
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      vSpan.textContent = v.toFixed(4);
      onChange(v);
    });
    ctrl.appendChild(wrap);
    return { slider, vSpan };
  }

  function makeBtn(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = `padding:5px 12px;font-size:11px;border-radius:6px;cursor:pointer;background:rgba(${ACCENT_RGB},0.1);color:#b0b0b0;border:1px solid rgba(${ACCENT_RGB},0.25);font-family:${FONT};transition:all 0.2s;`;
    btn.addEventListener('mouseenter', () => { btn.style.background = `rgba(${ACCENT_RGB},0.25)`; btn.style.color = '#e0e0e0'; });
    btn.addEventListener('mouseleave', () => { btn.style.background = `rgba(${ACCENT_RGB},0.1)`; btn.style.color = '#b0b0b0'; });
    btn.addEventListener('click', onClick);
    ctrl.appendChild(btn);
    return btn;
  }

  function makeToggle(label, checked, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;align-items:center;gap:4px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = `color:#6b6b6b;font-size:11px;font-family:${FONT};`;
    lbl.textContent = label;
    wrap.appendChild(lbl);
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = checked;
    cb.style.cssText = `accent-color:${ACCENT};cursor:pointer;`;
    cb.addEventListener('change', () => onChange(cb.checked));
    wrap.appendChild(cb);
    ctrl.appendChild(wrap);
    return cb;
  }

  // --- Controls ---
  const probCtrl = makeSlider('p', 0.0, 1.0, 0.005, probability, (v) => {
    probability = v;
    generateGrid();
    render();
  });

  makeBtn('New Grid', () => { generateGrid(); render(); });

  makeToggle('Spanning', showSpanning, (v) => { showSpanning = v; render(); });

  // status label — spanning cluster mila ya nahi
  const statusSpan = document.createElement('span');
  statusSpan.style.cssText = `font-size:11px;font-family:${FONT};color:#6b6b6b;`;
  ctrl.appendChild(statusSpan);

  // --- Union-Find (Disjoint Set) ---
  // path compression + rank based union — almost O(1) amortized
  function ufFind(x) {
    // path compression — root tak jaake sab ko seedha root se jod do
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]; // path halving — thoda faster
      x = parent[x];
    }
    return x;
  }

  function ufUnion(a, b) {
    const ra = ufFind(a), rb = ufFind(b);
    if (ra === rb) return;
    // rank based merge — chhota tree bade ke neeche jaaye
    if (rank[ra] < rank[rb]) { parent[ra] = rb; }
    else if (rank[ra] > rank[rb]) { parent[rb] = ra; }
    else { parent[rb] = ra; rank[ra]++; }
  }

  // --- Grid generate karo ---
  function generateGrid() {
    const total = GRID_COLS * GRID_ROWS;
    grid = new Uint8Array(total);
    parent = new Int32Array(total);
    rank = new Int32Array(total);

    // har cell ko probability p se occupy karo
    for (let i = 0; i < total; i++) {
      grid[i] = Math.random() < probability ? 1 : 0;
      parent[i] = i;  // khud ka parent khud — initially sab alag
      rank[i] = 0;
    }

    // neighbors ke saath union karo — 4-connected
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const idx = r * GRID_COLS + c;
        if (!grid[idx]) continue;

        // right neighbor check karo
        if (c + 1 < GRID_COLS && grid[idx + 1]) {
          ufUnion(idx, idx + 1);
        }
        // bottom neighbor check karo
        if (r + 1 < GRID_ROWS && grid[idx + GRID_COLS]) {
          ufUnion(idx, idx + GRID_COLS);
        }
      }
    }

    // unique clusters count karo aur colors assign karo
    clusterColors = {};
    const rootSet = new Set();
    for (let i = 0; i < total; i++) {
      if (grid[i]) rootSet.add(ufFind(i));
    }
    clusterCount = rootSet.size;

    // har cluster ko random color do
    let colorIdx = 0;
    rootSet.forEach((root) => {
      // hue spread karo evenly — saturation aur lightness fixed
      const hue = (colorIdx * 137.5) % 360; // golden angle — acchi distribution
      clusterColors[root] = hue;
      colorIdx++;
    });

    // spanning cluster dhundho — top row se bottom row tak connected
    spanningRoot = -1;
    const topRoots = new Set();
    for (let c = 0; c < GRID_COLS; c++) {
      if (grid[c]) topRoots.add(ufFind(c));
    }
    // bottom row mein check karo koi top root match karta hai kya
    const bottomStart = (GRID_ROWS - 1) * GRID_COLS;
    for (let c = 0; c < GRID_COLS; c++) {
      if (grid[bottomStart + c]) {
        const root = ufFind(bottomStart + c);
        if (topRoots.has(root)) {
          spanningRoot = root;
          break; // pehla spanning cluster mil gaya — ek kaafi hai
        }
      }
    }

    // status update karo
    if (spanningRoot >= 0) {
      statusSpan.textContent = `Spanning: YES | Clusters: ${clusterCount}`;
      statusSpan.style.color = ACCENT;
    } else {
      statusSpan.textContent = `Spanning: NO | Clusters: ${clusterCount}`;
      statusSpan.style.color = '#6b6b6b';
    }
  }

  // --- Render ---
  function render() {
    const total = GRID_COLS * GRID_ROWS;

    for (let i = 0; i < total; i++) {
      const pi = i * 4;

      if (!grid[i]) {
        // empty cell — dark background
        pixels[pi] = 17; pixels[pi + 1] = 17; pixels[pi + 2] = 17; pixels[pi + 3] = 255;
        continue;
      }

      const root = ufFind(i);

      // spanning cluster gold mein highlight karo
      if (showSpanning && root === spanningRoot && spanningRoot >= 0) {
        pixels[pi] = 245; pixels[pi + 1] = 158; pixels[pi + 2] = 11; pixels[pi + 3] = 255;
        continue;
      }

      // baaki clusters — unique color by root
      const hue = clusterColors[root] || 0;
      // hue to RGB convert karo — simple HSL to RGB
      const h = hue / 60;
      const x = 1 - Math.abs(h % 2 - 1);
      let r1 = 0, g1 = 0, b1 = 0;
      if (h < 1) { r1 = 1; g1 = x; }
      else if (h < 2) { r1 = x; g1 = 1; }
      else if (h < 3) { g1 = 1; b1 = x; }
      else if (h < 4) { g1 = x; b1 = 1; }
      else if (h < 5) { r1 = x; b1 = 1; }
      else { r1 = 1; b1 = x; }
      // lightness adjust — 45% lightness, 70% saturation
      const l = 0.45, s = 0.7;
      const c = (1 - Math.abs(2 * l - 1)) * s;
      const m = l - c / 2;
      pixels[pi] = Math.floor((r1 * c + m) * 255);
      pixels[pi + 1] = Math.floor((g1 * c + m) * 255);
      pixels[pi + 2] = Math.floor((b1 * c + m) * 255);
      pixels[pi + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
  }

  // --- Resize handler --- canvas fixed GRID_SIZE hai but container change ho sakta hai
  function resize() {
    canvasW = container.clientWidth;
    // canvas.width aur height GRID_SIZE pe fixed rehni chahiye (pixel-mapped grid)
    // bas CSS size sync karo
  }
  resize();
  window.addEventListener('resize', resize);

  // --- Initial generate ---
  generateGrid();
  render();

  // --- Animation loop — mostly static, but re-render on visibility ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    // percolation static hai — re-render bas visibility change pe chahiye
    animationId = requestAnimationFrame(loop);
  }

  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) loop();
    else if (!isVisible && animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && isVisible && !animationId) loop(); });
}
