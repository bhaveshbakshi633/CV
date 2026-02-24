// ============================================================
// Abelian Sandpile — Self-organized criticality ka visual demo
// Center pe grains daal, cell ≥ 4 ho toh topple — 4 neighbors ko 1-1
// Cascade chalta hai jab tak stable na ho jaaye
// Fractal patterns emerge hote hain — nature ka apna art hai ye
// Bak-Tang-Wiesenfeld model — complexity science ka poster child
// ============================================================

// yahi se shuru — grains daal aur fractal banta dekh
export function initSandpile() {
  const container = document.getElementById('sandpileContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  const ACCENT_RGB = '245,158,11';
  const BG = '#111';
  const FONT = "'JetBrains Mono',monospace";
  const GRID_SIZE = 201;        // odd rakh taaki center exact ho
  const CENTER = Math.floor(GRID_SIZE / 2);

  // --- State ---
  let animationId = null, isVisible = false, canvasW = 0;
  let grid = new Uint8Array(GRID_SIZE * GRID_SIZE);
  let dropRate = 100;           // kitne grains per frame
  let isPaused = false;
  let totalGrains = 0;
  let totalTopples = 0;

  // toppling ke liye double buffer — ek read, ek write
  let tempGrid = new Uint8Array(GRID_SIZE * GRID_SIZE);

  // color palette — 0=dark, 1=green, 2=amber, 3=red
  const COLORS = [
    [17, 17, 17],         // 0 grains — dark bg
    [42, 68, 42],         // 1 grain — deep green
    [245, 158, 11],       // 2 grains — amber (accent color)
    [244, 68, 68],        // 3 grains — red-ish (almost unstable)
  ];

  // --- DOM setup ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.width = GRID_SIZE;
  canvas.height = GRID_SIZE;
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:${BG};border:1px solid rgba(${ACCENT_RGB},0.15);image-rendering:pixelated;image-rendering:crisp-edges;`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(GRID_SIZE, GRID_SIZE);
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
    slider.style.cssText = `width:90px;height:4px;accent-color:${ACCENT};cursor:pointer;`;
    wrap.appendChild(slider);
    const vSpan = document.createElement('span');
    vSpan.style.cssText = `color:#f0f0f0;font-size:11px;font-family:${FONT};min-width:32px;`;
    vSpan.textContent = String(val);
    wrap.appendChild(vSpan);
    slider.addEventListener('input', () => {
      const v = parseInt(slider.value);
      vSpan.textContent = String(v);
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

  // --- Controls ---
  makeSlider('Rate', 10, 1000, 10, dropRate, (v) => { dropRate = v; });

  const pauseBtn = makeBtn('Pause', () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Play' : 'Pause';
  });

  makeBtn('Reset', () => {
    grid.fill(0);
    totalGrains = 0;
    totalTopples = 0;
  });

  // stats display
  const statsSpan = document.createElement('span');
  statsSpan.style.cssText = `font-size:10px;font-family:${FONT};color:#6b6b6b;`;
  ctrl.appendChild(statsSpan);

  // --- Grid helpers ---
  function idx(x, y) { return y * GRID_SIZE + x; }

  // --- Ek grain center pe daal aur stable hone tak topple karo ---
  function dropGrain() {
    grid[idx(CENTER, CENTER)]++;
    totalGrains++;

    // topple karo jab tak koi bhi cell ≥ 4 hai
    // iterative approach — recursive se stack overflow nahi hoga
    let changed = true;
    while (changed) {
      changed = false;
      // tempGrid mein copy karo — simultaneous update ke liye
      tempGrid.set(grid);

      for (let y = 1; y < GRID_SIZE - 1; y++) {
        for (let x = 1; x < GRID_SIZE - 1; x++) {
          const i = idx(x, y);
          if (tempGrid[i] >= 4) {
            // topple — 4 grains nikaal, 1-1 neighbors ko de
            const toppleCount = Math.floor(tempGrid[i] / 4);
            grid[i] -= toppleCount * 4;
            grid[idx(x - 1, y)] += toppleCount;
            grid[idx(x + 1, y)] += toppleCount;
            grid[idx(x, y - 1)] += toppleCount;
            grid[idx(x, y + 1)] += toppleCount;
            totalTopples += toppleCount;
            changed = true;
          }
        }
      }

      // boundary pe grains gir jaate hain — edge cells se bahar nikal jaate hain
      for (let x = 0; x < GRID_SIZE; x++) {
        grid[idx(x, 0)] = Math.min(grid[idx(x, 0)], 3);
        grid[idx(x, GRID_SIZE - 1)] = Math.min(grid[idx(x, GRID_SIZE - 1)], 3);
      }
      for (let y = 0; y < GRID_SIZE; y++) {
        grid[idx(0, y)] = Math.min(grid[idx(0, y)], 3);
        grid[idx(GRID_SIZE - 1, y)] = Math.min(grid[idx(GRID_SIZE - 1, y)], 3);
      }
    }
  }

  // --- Optimized batch drop — ek frame mein multiple grains ---
  function dropBatch(count) {
    // grains daal do pehle sab — fir ek baar mein topple karo
    grid[idx(CENTER, CENTER)] += count;
    totalGrains += count;

    // topple cascade — jab tak stable na ho
    let changed = true;
    let safetyCounter = 0;
    const maxIterations = 5000; // infinite loop se bachne ke liye

    while (changed && safetyCounter < maxIterations) {
      changed = false;
      safetyCounter++;

      for (let y = 1; y < GRID_SIZE - 1; y++) {
        for (let x = 1; x < GRID_SIZE - 1; x++) {
          const i = idx(x, y);
          if (grid[i] >= 4) {
            const toppleCount = Math.floor(grid[i] / 4);
            grid[i] -= toppleCount * 4;
            grid[idx(x - 1, y)] += toppleCount;
            grid[idx(x + 1, y)] += toppleCount;
            grid[idx(x, y - 1)] += toppleCount;
            grid[idx(x, y + 1)] += toppleCount;
            totalTopples += toppleCount;
            changed = true;
          }
        }
      }

      // boundary grains clip karo
      for (let x = 0; x < GRID_SIZE; x++) {
        if (grid[idx(x, 0)] >= 4) grid[idx(x, 0)] = 3;
        if (grid[idx(x, GRID_SIZE - 1)] >= 4) grid[idx(x, GRID_SIZE - 1)] = 3;
      }
      for (let y = 0; y < GRID_SIZE; y++) {
        if (grid[idx(0, y)] >= 4) grid[idx(0, y)] = 3;
        if (grid[idx(GRID_SIZE - 1, y)] >= 4) grid[idx(GRID_SIZE - 1, y)] = 3;
      }
    }
  }

  // --- Render ---
  function render() {
    const total = GRID_SIZE * GRID_SIZE;
    for (let i = 0; i < total; i++) {
      const pi = i * 4;
      const val = Math.min(grid[i], 3); // 0-3 ke beech clamp karo display ke liye
      const c = COLORS[val];
      pixels[pi] = c[0];
      pixels[pi + 1] = c[1];
      pixels[pi + 2] = c[2];
      pixels[pi + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);

    // stats update
    const grainStr = totalGrains > 1000 ? (totalGrains / 1000).toFixed(1) + 'k' : String(totalGrains);
    const toppleStr = totalTopples > 1000 ? (totalTopples / 1000).toFixed(1) + 'k' : String(totalTopples);
    statsSpan.textContent = `grains: ${grainStr}  topples: ${toppleStr}`;
  }

  // --- Animation loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    if (!isPaused) {
      dropBatch(dropRate);
    }
    render();
    animationId = requestAnimationFrame(loop);
  }

  // --- Resize handler --- grid fixed GRID_SIZE hai, CSS size sync karo
  function resize() {
    canvasW = container.clientWidth;
    // canvas.width/height GRID_SIZE pe fixed hai (pixel-mapped grid)
    // sirf CSS size container ke saath sync hoti hai — wo style mein already hai
  }
  resize();
  window.addEventListener('resize', resize);

  // initial render — blank grid dikhao
  render();

  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) loop();
    else if (!isVisible && animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && isVisible && !animationId) loop(); });
}
