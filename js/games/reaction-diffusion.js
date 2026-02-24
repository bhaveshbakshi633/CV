// ============================================================
// Reaction-Diffusion Simulator — Gray-Scott Model
// Click/drag se chemical B seed karo, Turing patterns dekho
// Mesmerizing patterns: spots, stripes, waves, mitosis, coral
// Gray-Scott: A + 2B -> 3B, feed rate f, kill rate k
// ============================================================

// yahi se sab shuru hota hai — container dhundho, grid banao, patterns emerge hone do
export function initReactionDiffusion() {
  const container = document.getElementById('reactionDiffusionContainer');
  if (!container) {
    console.warn('reactionDiffusionContainer nahi mila bhai, reaction-diffusion skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 350;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';

  // diffusion coefficients — A fast diffuse hota hai, B slow
  const D_A = 1.0;
  const D_B = 0.5;

  // grid resolution — balance between detail aur performance
  const GRID_W = 200;
  const GRID_H = 150;

  // --- Presets — (f, k) pairs jo alag patterns banate hain ---
  const PRESETS = {
    'Spots':    { f: 0.035, k: 0.065 },
    'Stripes':  { f: 0.025, k: 0.060 },
    'Waves':    { f: 0.014, k: 0.054 },
    'Mitosis':  { f: 0.0367, k: 0.0649 },
    'Coral':    { f: 0.0545, k: 0.062 },
  };

  // --- State ---
  let feedRate = 0.035;   // A ka feed rate
  let killRate = 0.065;   // B ka kill rate
  let stepsPerFrame = 12; // simulation steps har animation frame pe
  let running = true;
  let isVisible = false;
  let animFrameId = null;

  // double-buffered grids — Float32Array for precision aur speed
  let gridA = new Float32Array(GRID_W * GRID_H);
  let gridB = new Float32Array(GRID_W * GRID_H);
  let nextA = new Float32Array(GRID_W * GRID_H);
  let nextB = new Float32Array(GRID_W * GRID_H);

  // mouse/touch state — painting ke liye
  let isDrawing = false;

  // --- Grid initialization ---
  // poora grid A=1, B=0 se bharo, fir kuch random circles mein B seed karo
  function initGrid() {
    // sab jagah A=1, B=0
    gridA.fill(1.0);
    gridB.fill(0.0);

    // random circles mein B seed karo — pattern shuru hone ke liye
    const numSeeds = 8 + Math.floor(Math.random() * 8);
    for (let s = 0; s < numSeeds; s++) {
      const cx = Math.floor(Math.random() * GRID_W);
      const cy = Math.floor(Math.random() * GRID_H);
      const radius = 3 + Math.floor(Math.random() * 5);
      seedCircle(cx, cy, radius);
    }
  }

  // ek circle mein B=1 seed karo — click pe bhi yahi use hoga
  function seedCircle(cx, cy, radius) {
    const r2 = radius * radius;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy > r2) continue;
        // toroidal wrap — edges se doosri side pe jaao
        const x = ((cx + dx) % GRID_W + GRID_W) % GRID_W;
        const y = ((cy + dy) % GRID_H + GRID_H) % GRID_H;
        const idx = y * GRID_W + x;
        gridA[idx] = 0.0;
        gridB[idx] = 1.0;
      }
    }
  }

  // --- Gray-Scott simulation step ---
  // ye hai core algorithm — ek timestep advance karo
  // A + 2B -> 3B reaction, diffusion laplacian se, feed/kill terms
  function simulate() {
    const w = GRID_W;
    const h = GRID_H;
    const f = feedRate;
    const k = killRate;
    const dA = D_A;
    const dB = D_B;
    const dt = 1.0; // timestep — normalized

    for (let y = 0; y < h; y++) {
      // wrap indices precompute — toroidal boundary
      const ym1 = y === 0 ? h - 1 : y - 1;
      const yp1 = y === h - 1 ? 0 : y + 1;
      const yOff = y * w;
      const ym1Off = ym1 * w;
      const yp1Off = yp1 * w;

      for (let x = 0; x < w; x++) {
        const xm1 = x === 0 ? w - 1 : x - 1;
        const xp1 = x === w - 1 ? 0 : x + 1;

        const idx = yOff + x;
        const a = gridA[idx];
        const b = gridB[idx];

        // laplacian — 3x3 convolution kernel
        // center weight = -1, edge weights = 0.2, corner weights = 0.05
        // ye standard 9-point laplacian hai — smoother than 5-point
        const lapA =
          gridA[ym1Off + xm1] * 0.05 +
          gridA[ym1Off + x]   * 0.2  +
          gridA[ym1Off + xp1] * 0.05 +
          gridA[yOff + xm1]   * 0.2  +
          a                   * -1.0 +
          gridA[yOff + xp1]   * 0.2  +
          gridA[yp1Off + xm1] * 0.05 +
          gridA[yp1Off + x]   * 0.2  +
          gridA[yp1Off + xp1] * 0.05;

        const lapB =
          gridB[ym1Off + xm1] * 0.05 +
          gridB[ym1Off + x]   * 0.2  +
          gridB[ym1Off + xp1] * 0.05 +
          gridB[yOff + xm1]   * 0.2  +
          b                   * -1.0 +
          gridB[yOff + xp1]   * 0.2  +
          gridB[yp1Off + xm1] * 0.05 +
          gridB[yp1Off + x]   * 0.2  +
          gridB[yp1Off + xp1] * 0.05;

        // Gray-Scott equations
        // dA/dt = Dₐ * ∇²A - A*B² + f*(1-A)
        // dB/dt = D_b * ∇²B + A*B² - (k+f)*B
        const abb = a * b * b;
        nextA[idx] = a + (dA * lapA - abb + f * (1.0 - a)) * dt;
        nextB[idx] = b + (dB * lapB + abb - (k + f) * b) * dt;

        // clamp — values 0-1 ke beech rakhna zaroori hai stability ke liye
        if (nextA[idx] < 0) nextA[idx] = 0;
        if (nextA[idx] > 1) nextA[idx] = 1;
        if (nextB[idx] < 0) nextB[idx] = 0;
        if (nextB[idx] > 1) nextB[idx] = 1;
      }
    }

    // swap buffers — double buffering, copy nahi karna padega
    const tmpA = gridA;
    const tmpB = gridB;
    gridA = nextA;
    gridB = nextB;
    nextA = tmpA;
    nextB = tmpB;
  }

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // canvas — simulation render hoga yahan
  // canvas element size = grid size, CSS stretch karega container width tak
  // image-rendering: pixelated — crisp pixels, blurry nahi
  const canvas = document.createElement('canvas');
  canvas.width = GRID_W;
  canvas.height = GRID_H;
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:#0a0a0a',
    'image-rendering:pixelated',
    'image-rendering:crisp-edges',
    'touch-action:none',
  ].join(';');
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  // ImageData ek baar banao, reuse karo — GC pressure kam hoga
  const imageData = ctx.createImageData(GRID_W, GRID_H);
  const pixels = imageData.data;

  // --- Controls row 1: Play/Pause, Clear, Presets ---
  const controlsDiv1 = document.createElement('div');
  controlsDiv1.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv1);

  // --- Controls row 2: Sliders ---
  const controlsDiv2 = document.createElement('div');
  controlsDiv2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:8px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv2);

  // --- Button helper ---
  function makeButton(text, parent, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'border-radius:6px',
      'padding:5px 12px',
      'font-size:11px',
      'font-family:"JetBrains Mono",monospace',
      'cursor:pointer',
      'transition:all 0.2s',
      'user-select:none',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#ffffff';
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
        btn.style.color = '#b0b0b0';
      }
    });
    btn.addEventListener('click', onClick);
    parent.appendChild(btn);
    return btn;
  }

  // button active state set karne ka helper
  function setActive(btn, active) {
    btn._active = active;
    if (active) {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.3)';
      btn.style.color = ACCENT;
      btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.5)';
    } else {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      btn.style.color = '#b0b0b0';
      btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
    }
  }

  // --- Slider helper ---
  function makeSlider(label, min, max, value, step, parent, onChange) {
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
    lbl.style.cssText = 'white-space:nowrap;min-width:18px;';
    wrap.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.value = value;
    slider.step = step;
    slider.style.cssText = 'width:80px;accent-color:' + ACCENT + ';cursor:pointer;';

    const valSpan = document.createElement('span');
    valSpan.textContent = Number(value).toFixed(step.toString().includes('.') ? step.toString().split('.')[1].length : 0);
    valSpan.style.cssText = 'min-width:38px;text-align:right;color:' + ACCENT + ';font-size:10px;';

    slider.addEventListener('input', () => {
      const v = Number(slider.value);
      const decimals = step.toString().includes('.') ? step.toString().split('.')[1].length : 0;
      valSpan.textContent = v.toFixed(decimals);
      onChange(v);
    });

    wrap.appendChild(slider);
    wrap.appendChild(valSpan);
    parent.appendChild(wrap);
    return { slider, valSpan };
  }

  // --- Row 1: Buttons ---

  // play/pause
  const playBtn = makeButton('Pause', controlsDiv1, () => {
    running = !running;
    playBtn.textContent = running ? 'Pause' : 'Play';
    setActive(playBtn, running);
  });
  setActive(playBtn, true);

  // clear/reset — grid reset kar do initial state mein
  makeButton('Clear', controlsDiv1, () => {
    initGrid();
  });

  // separator
  const sep = document.createElement('span');
  sep.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.3);font-size:11px;padding:0 2px;';
  sep.textContent = '|';
  controlsDiv1.appendChild(sep);

  // preset buttons — har preset active preset ko highlight karega
  let activePresetBtn = null;
  const presetBtns = {};

  Object.keys(PRESETS).forEach(name => {
    const preset = PRESETS[name];
    const btn = makeButton(name, controlsDiv1, () => {
      feedRate = preset.f;
      killRate = preset.k;
      // sliders update karo
      fSlider.slider.value = feedRate;
      fSlider.valSpan.textContent = feedRate.toFixed(4);
      kSlider.slider.value = killRate;
      kSlider.valSpan.textContent = killRate.toFixed(4);
      // grid reset karo — naye preset ke liye fresh start
      initGrid();
      // active preset highlight
      if (activePresetBtn) setActive(activePresetBtn, false);
      setActive(btn, true);
      activePresetBtn = btn;
    });
    presetBtns[name] = btn;
  });

  // default preset highlight
  setActive(presetBtns['Spots'], true);
  activePresetBtn = presetBtns['Spots'];

  // --- Row 2: Sliders ---

  // feed rate slider — 0.0001 step for precise preset values
  const fSlider = makeSlider('f', '0.01', '0.08', feedRate.toString(), '0.0001', controlsDiv2, (v) => {
    feedRate = v;
    // preset match check — agar koi preset match karta hai toh highlight karo
    updatePresetHighlight();
  });

  // kill rate slider — 0.0001 step for precise preset values
  const kSlider = makeSlider('k', '0.04', '0.07', killRate.toString(), '0.0001', controlsDiv2, (v) => {
    killRate = v;
    updatePresetHighlight();
  });

  // speed slider — steps per frame
  makeSlider('spd', '5', '30', stepsPerFrame.toString(), '1', controlsDiv2, (v) => {
    stepsPerFrame = v;
  });

  // preset highlight update — agar slider values kisi preset se match kare
  function updatePresetHighlight() {
    if (activePresetBtn) {
      setActive(activePresetBtn, false);
      activePresetBtn = null;
    }
    // check all presets
    for (const name of Object.keys(PRESETS)) {
      const p = PRESETS[name];
      if (Math.abs(feedRate - p.f) < 0.0005 && Math.abs(killRate - p.k) < 0.0005) {
        setActive(presetBtns[name], true);
        activePresetBtn = presetBtns[name];
        break;
      }
    }
  }

  // --- Mouse/touch interaction — click/drag se B seed karo ---
  function getGridPos(e) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }
    // CSS position se grid position nikal
    const cssX = clientX - rect.left;
    const cssY = clientY - rect.top;
    const gx = Math.floor((cssX / rect.width) * GRID_W);
    const gy = Math.floor((cssY / rect.height) * GRID_H);
    return { x: gx, y: gy };
  }

  function paintAtPos(e) {
    const pos = getGridPos(e);
    if (pos.x >= 0 && pos.x < GRID_W && pos.y >= 0 && pos.y < GRID_H) {
      // paint radius — thoda bada circle taaki impact dikhe
      seedCircle(pos.x, pos.y, 4);
    }
  }

  function handlePointerDown(e) {
    isDrawing = true;
    paintAtPos(e);
  }

  function handlePointerMove(e) {
    if (!isDrawing) return;
    paintAtPos(e);
  }

  function handlePointerUp() {
    isDrawing = false;
  }

  // mouse events
  canvas.addEventListener('mousedown', handlePointerDown);
  canvas.addEventListener('mousemove', handlePointerMove);
  canvas.addEventListener('mouseup', handlePointerUp);
  canvas.addEventListener('mouseleave', handlePointerUp);

  // touch events — mobile support
  canvas.addEventListener('touchstart', (e) => { e.preventDefault(); handlePointerDown(e); }, { passive: false });
  canvas.addEventListener('touchmove', (e) => { e.preventDefault(); handlePointerMove(e); }, { passive: false });
  canvas.addEventListener('touchend', handlePointerUp);

  // --- Rendering ---
  // chemical B concentration ko color mein map karo
  // dark (#0a0a0a) for B=0, purple/cyan gradient for high B
  function render() {
    const w = GRID_W;
    const h = GRID_H;

    for (let i = 0; i < w * h; i++) {
      const b = gridB[i];
      const pi = i * 4;

      // color mapping — B concentration se
      // low B = dark background, high B = bright purple-cyan
      // smooth gradient with multiple color stops for visual richness

      if (b < 0.01) {
        // bahut kam B — dark background
        pixels[pi]     = 10;  // R
        pixels[pi + 1] = 10;  // G
        pixels[pi + 2] = 10;  // B
      } else if (b < 0.15) {
        // low B — deep purple fade in
        const t = b / 0.15;
        pixels[pi]     = Math.floor(10 + 40 * t);  // R: 10 -> 50
        pixels[pi + 1] = Math.floor(10 + 10 * t);  // G: 10 -> 20
        pixels[pi + 2] = Math.floor(10 + 70 * t);  // B: 10 -> 80
      } else if (b < 0.35) {
        // mid-low — purple intensifying
        const t = (b - 0.15) / 0.2;
        pixels[pi]     = Math.floor(50 + 80 * t);   // R: 50 -> 130
        pixels[pi + 1] = Math.floor(20 + 40 * t);   // G: 20 -> 60
        pixels[pi + 2] = Math.floor(80 + 120 * t);  // B: 80 -> 200
      } else if (b < 0.55) {
        // mid — purple to cyan transition
        const t = (b - 0.35) / 0.2;
        pixels[pi]     = Math.floor(130 + 37 * t);  // R: 130 -> 167
        pixels[pi + 1] = Math.floor(60 + 79 * t);   // G: 60 -> 139
        pixels[pi + 2] = Math.floor(200 + 50 * t);  // B: 200 -> 250
      } else if (b < 0.75) {
        // high — bright purple-cyan
        const t = (b - 0.55) / 0.2;
        pixels[pi]     = Math.floor(167 - 67 * t);  // R: 167 -> 100
        pixels[pi + 1] = Math.floor(139 + 81 * t);  // G: 139 -> 220
        pixels[pi + 2] = Math.floor(250 + 5 * t);   // B: 250 -> 255
      } else {
        // very high B — bright cyan-white glow
        const t = Math.min(1, (b - 0.75) / 0.25);
        pixels[pi]     = Math.floor(100 + 155 * t);  // R: 100 -> 255
        pixels[pi + 1] = Math.floor(220 + 35 * t);   // G: 220 -> 255
        pixels[pi + 2] = 255;                         // B: 255
      }

      pixels[pi + 3] = 255; // alpha — always opaque
    }

    ctx.putImageData(imageData, 0, 0);
  }

  // --- Animation loop ---
  function gameLoop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animFrameId = null; return; }
    if (!isVisible) {
      animFrameId = null;
      return;
    }

    // simulation steps — multiple per frame for speed
    if (running) {
      for (let s = 0; s < stepsPerFrame; s++) {
        simulate();
      }
    }

    // render har frame pe
    render();

    animFrameId = requestAnimationFrame(gameLoop);
  }

  function startLoop() {
    if (!animFrameId && isVisible) {
      animFrameId = requestAnimationFrame(gameLoop);
    }
  }

  function stopLoop() {
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  }

  // --- IntersectionObserver — sirf visible hone pe CPU use karo ---
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          isVisible = true;
          startLoop();
        } else {
          isVisible = false;
          stopLoop();
        }
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animFrameId) gameLoop(); });

  // tab switch pe pause — background mein CPU waste mat kar
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      isVisible = false;
      stopLoop();
    } else {
      // check agar container still visible hai
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) {
        isVisible = true;
        startLoop();
      }
    }
  });

  // --- Resize handling ---
  // canvas element size fixed hai (GRID_W x GRID_H), sirf CSS resize hota hai
  // koi grid recalculation nahi chahiye — CSS stretch handles it
  const resizeObserver = new ResizeObserver(() => {
    // canvas style already width:100% hai, kuch special karne ki zaroorat nahi
    // but agar container resize hota hai toh re-render kar do current state
    if (isVisible && !animFrameId) {
      render();
    }
  });
  resizeObserver.observe(container);

  // --- Init ---
  initGrid();
  render(); // pehla frame draw kar — blank na dikhe
}
