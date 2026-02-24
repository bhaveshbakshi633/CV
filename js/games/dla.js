// ============================================================
// Diffusion-Limited Aggregation — Crystal Growth Simulator
// Random walkers stick on contact, building snowflake/coral-like
// fractal structures. Bohot satisfying — meditative crystal growth
// Lower stickiness = zyada branchy/fractal, higher = compact blob
// ============================================================

// yahi se sab shuru hota hai — container dhundho, grid banao, crystal ugao
export function initDLA() {
  const container = document.getElementById('dlaContainer');
  if (!container) {
    console.warn('dlaContainer nahi mila bhai, DLA skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';

  // grid resolution — computation ke liye, render bhi isi size pe hoga
  const GRID_W = 300;
  const GRID_H = 200;

  // --- State ---
  let grid = new Uint32Array(GRID_W * GRID_H); // 0 = empty, >0 = crystal (attachment order)
  let particleCount = 0;      // kitne particles crystal mein hain
  let maxOrder = 1;           // sabse bada attachment order — coloring ke liye
  let stickiness = 0.4;       // probability of sticking (low = branchy, high = compact)
  let walkersPerFrame = 800;  // speed control — har frame mein kitne walk steps
  let showWalkers = false;    // random walkers dikhane hain ya nahi
  let seedShape = 'Point';    // Point, Line, Circle
  let isVisible = false;
  let animFrameId = null;

  // bounding box track kar crystal ka — spawn radius optimize karne ke liye
  let crystalMinX = GRID_W;
  let crystalMaxX = 0;
  let crystalMinY = GRID_H;
  let crystalMaxY = 0;

  // active walkers — position track karo
  const MAX_WALKERS = 50;
  let walkers = []; // [{x, y}]

  // --- Grid helpers ---
  // index nikaal grid array mein
  function idx(x, y) {
    return y * GRID_W + x;
  }

  // check cell crystal hai ya nahi
  function isCrystal(x, y) {
    if (x < 0 || x >= GRID_W || y < 0 || y >= GRID_H) return false;
    return grid[idx(x, y)] > 0;
  }

  // crystal cell set kar — order track kar
  function setCrystal(x, y) {
    particleCount++;
    maxOrder = particleCount;
    grid[idx(x, y)] = particleCount;

    // bounding box update kar
    if (x < crystalMinX) crystalMinX = x;
    if (x > crystalMaxX) crystalMaxX = x;
    if (y < crystalMinY) crystalMinY = y;
    if (y > crystalMaxY) crystalMaxY = y;
  }

  // check agar ye cell crystal ke neighbor hai (4-connected)
  function hasNeighborCrystal(x, y) {
    return isCrystal(x - 1, y) || isCrystal(x + 1, y) ||
           isCrystal(x, y - 1) || isCrystal(x, y + 1);
  }

  // --- Seed placement ---
  // grid reset karo aur chosen shape ke seed place karo
  function resetSimulation() {
    grid.fill(0);
    particleCount = 0;
    maxOrder = 1;
    walkers = [];
    crystalMinX = GRID_W;
    crystalMaxX = 0;
    crystalMinY = GRID_H;
    crystalMaxY = 0;

    const cx = Math.floor(GRID_W / 2);
    const cy = Math.floor(GRID_H / 2);

    if (seedShape === 'Point') {
      // center mein ek pixel — classic snowflake growth
      setCrystal(cx, cy);
    } else if (seedShape === 'Line') {
      // neeche horizontal line — coral reef jaisa growth upar ki taraf
      const lineY = GRID_H - 5;
      for (let x = 20; x < GRID_W - 20; x++) {
        setCrystal(x, lineY);
      }
    } else if (seedShape === 'Circle') {
      // ring of seed pixels — inward+outward growth, bohot sundar
      const radius = Math.min(GRID_W, GRID_H) * 0.15;
      const steps = Math.floor(2 * Math.PI * radius * 1.5);
      for (let i = 0; i < steps; i++) {
        const angle = (i / steps) * 2 * Math.PI;
        const px = Math.round(cx + radius * Math.cos(angle));
        const py = Math.round(cy + radius * Math.sin(angle));
        if (px >= 0 && px < GRID_W && py >= 0 && py < GRID_H && !isCrystal(px, py)) {
          setCrystal(px, py);
        }
      }
    }

    updateCountDisplay();
  }

  // --- Spawn radius calculate kar ---
  // crystal ke bounding box ke bahar thoda margin rakh ke spawn karo
  function getSpawnRadius() {
    const dx = crystalMaxX - crystalMinX;
    const dy = crystalMaxY - crystalMinY;
    const crystalRadius = Math.max(dx, dy) / 2;
    // spawn radius crystal se 15-20 pixels bahar
    return Math.max(20, crystalRadius + 15);
  }

  // --- Random walker spawn kar ---
  // crystal ke aaas-paas circle pe spawn karo
  function spawnWalker() {
    const cx = (crystalMinX + crystalMaxX) / 2;
    const cy = (crystalMinY + crystalMaxY) / 2;
    const spawnR = getSpawnRadius();

    // random angle pe circle ke upar spawn
    const angle = Math.random() * 2 * Math.PI;
    let x = Math.round(cx + spawnR * Math.cos(angle));
    let y = Math.round(cy + spawnR * Math.sin(angle));

    // clamp grid ke andar
    x = Math.max(0, Math.min(GRID_W - 1, x));
    y = Math.max(0, Math.min(GRID_H - 1, y));

    // agar ye cell already crystal hai toh skip
    if (isCrystal(x, y)) return null;

    return { x, y };
  }

  // --- Random walk ek step ---
  // ±1 x ya y mein move kar
  function walkStep(w) {
    const dir = Math.floor(Math.random() * 4);
    if (dir === 0) w.x++;
    else if (dir === 1) w.x--;
    else if (dir === 2) w.y++;
    else w.y--;
  }

  // --- Main simulation step ---
  // har frame pe bohot saare walk steps process karo
  function simulateFrame() {
    const killRadius = getSpawnRadius() * 2.5; // bahut door gayi toh mardo
    const cx = (crystalMinX + crystalMaxX) / 2;
    const cy = (crystalMinY + crystalMaxY) / 2;
    let stepsLeft = walkersPerFrame;

    while (stepsLeft > 0) {
      // agar walkers kam hain toh naye spawn karo
      while (walkers.length < MAX_WALKERS) {
        const w = spawnWalker();
        if (w) walkers.push(w);
        else break;
      }

      if (walkers.length === 0) break;

      // har walker ko kuch steps do
      const stepsPerWalker = Math.max(1, Math.floor(stepsLeft / walkers.length));
      const toRemove = [];

      for (let i = 0; i < walkers.length; i++) {
        const w = walkers[i];
        let attached = false;
        let dead = false;

        for (let s = 0; s < stepsPerWalker; s++) {
          walkStep(w);

          // boundary check — grid ke bahar gayi toh mardo
          if (w.x < 0 || w.x >= GRID_W || w.y < 0 || w.y >= GRID_H) {
            dead = true;
            break;
          }

          // agar already crystal cell pe aa gayi toh hatao (shouldn't happen normally)
          if (isCrystal(w.x, w.y)) {
            dead = true;
            break;
          }

          // crystal neighbor check — chipakne ka mauka
          if (hasNeighborCrystal(w.x, w.y)) {
            // stickiness probability — low stickiness = walker bounces off, creates more branching
            if (Math.random() < stickiness) {
              setCrystal(w.x, w.y);
              attached = true;
              break;
            }
          }

          // kill radius check — bahut door chali gayi crystal se
          const distX = w.x - cx;
          const distY = w.y - cy;
          if (distX * distX + distY * distY > killRadius * killRadius) {
            dead = true;
            break;
          }
        }

        if (attached || dead) {
          toRemove.push(i);
        }
      }

      // dead/attached walkers hata do (reverse order mein taaki indices na bigdein)
      for (let i = toRemove.length - 1; i >= 0; i--) {
        walkers.splice(toRemove[i], 1);
      }

      stepsLeft -= stepsPerWalker * (walkers.length + toRemove.length);
      if (stepsLeft <= 0) break;
    }

    updateCountDisplay();
  }

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // canvas — yahan crystal render hoga
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
    'background:#000000',
    'image-rendering:pixelated',
    'image-rendering:crisp-edges',
  ].join(';');
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  // ImageData ek baar banao, reuse karo — GC pressure kam hoga
  const imageData = ctx.createImageData(GRID_W, GRID_H);
  const pixels = imageData.data;

  // --- Color gradient precompute karo ---
  // attachment order ke hisaab se color map — deep purple -> cyan -> gold
  // 256 entries ki lookup table — rendering fast hoga
  const COLOR_LUT_SIZE = 256;
  const colorLUT = new Uint8Array(COLOR_LUT_SIZE * 3);

  function buildColorLUT() {
    // gradient: deep purple (#2d1b69) -> cyan (#22d3ee) -> gold (#f59e0b)
    for (let i = 0; i < COLOR_LUT_SIZE; i++) {
      const t = i / (COLOR_LUT_SIZE - 1); // 0 to 1
      let r, g, b;

      if (t < 0.4) {
        // deep purple to cyan — early growth phase
        const s = t / 0.4;
        r = Math.floor(45 + (34 - 45) * s);    // 45 -> 34
        g = Math.floor(27 + (211 - 27) * s);    // 27 -> 211
        b = Math.floor(105 + (238 - 105) * s);  // 105 -> 238
      } else if (t < 0.75) {
        // cyan to bright white-ish — mid growth
        const s = (t - 0.4) / 0.35;
        r = Math.floor(34 + (220 - 34) * s);    // 34 -> 220
        g = Math.floor(211 + (240 - 211) * s);  // 211 -> 240
        b = Math.floor(238 + (255 - 238) * s);  // 238 -> 255
      } else {
        // white to gold — tips and late growth
        const s = (t - 0.75) / 0.25;
        r = Math.floor(220 + (245 - 220) * s);  // 220 -> 245
        g = Math.floor(240 + (158 - 240) * s);  // 240 -> 158
        b = Math.floor(255 + (11 - 255) * s);   // 255 -> 11
      }

      colorLUT[i * 3]     = r;
      colorLUT[i * 3 + 1] = g;
      colorLUT[i * 3 + 2] = b;
    }
  }
  buildColorLUT();

  // --- Rendering ---
  // crystal cells ko color karo attachment order ke hisaab se
  function render() {
    const totalPixels = GRID_W * GRID_H;
    const mo = maxOrder || 1;

    for (let i = 0; i < totalPixels; i++) {
      const pi = i * 4;
      const val = grid[i];

      if (val > 0) {
        // crystal cell — order ke hisaab se color LUT se utha
        const t = Math.min(1, (val - 1) / mo);
        const lutIdx = Math.floor(t * (COLOR_LUT_SIZE - 1));
        const li = lutIdx * 3;
        pixels[pi]     = colorLUT[li];
        pixels[pi + 1] = colorLUT[li + 1];
        pixels[pi + 2] = colorLUT[li + 2];
        pixels[pi + 3] = 255;
      } else {
        // empty cell — pure black
        pixels[pi]     = 0;
        pixels[pi + 1] = 0;
        pixels[pi + 2] = 0;
        pixels[pi + 3] = 255;
      }
    }

    // walkers render karo agar toggle on hai
    if (showWalkers) {
      for (let i = 0; i < walkers.length; i++) {
        const w = walkers[i];
        if (w.x >= 0 && w.x < GRID_W && w.y >= 0 && w.y < GRID_H) {
          const pi = (w.y * GRID_W + w.x) * 4;
          // dim purple dots — subtle but visible
          pixels[pi]     = 120;
          pixels[pi + 1] = 90;
          pixels[pi + 2] = 180;
          pixels[pi + 3] = 255;
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }

  // --- Controls banao ---
  // Row 1: Seed Shape, Reset, Show Walkers toggle
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

  // Row 2: Sliders
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

  // Stats row — particle count
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'justify-content:center',
    'gap:20px',
    'margin-top:8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:rgba(255,255,255,0.5)',
  ].join(';');
  container.appendChild(statsDiv);

  const countLabel = document.createElement('span');
  statsDiv.appendChild(countLabel);

  function updateCountDisplay() {
    countLabel.textContent = 'Particles: ' + particleCount;
    if (particleCount > 0) {
      countLabel.style.color = ACCENT;
    } else {
      countLabel.style.color = 'rgba(255,255,255,0.5)';
    }
  }

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
    const decimals = step.toString().includes('.') ? step.toString().split('.')[1].length : 0;
    valSpan.textContent = Number(value).toFixed(decimals);
    valSpan.style.cssText = 'min-width:38px;text-align:right;color:' + ACCENT + ';font-size:10px;';

    slider.addEventListener('input', () => {
      const v = Number(slider.value);
      valSpan.textContent = v.toFixed(decimals);
      onChange(v);
    });

    wrap.appendChild(slider);
    wrap.appendChild(valSpan);
    parent.appendChild(wrap);
    return { slider, valSpan };
  }

  // --- Dropdown helper ---
  function makeDropdown(label, options, defaultVal, parent, onChange) {
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
    lbl.style.cssText = 'white-space:nowrap;';
    wrap.appendChild(lbl);

    const select = document.createElement('select');
    select.style.cssText = [
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'border-radius:6px',
      'padding:4px 8px',
      'font-size:11px',
      'font-family:"JetBrains Mono",monospace',
      'cursor:pointer',
      'outline:none',
    ].join(';');

    options.forEach(opt => {
      const option = document.createElement('option');
      option.value = opt;
      option.textContent = opt;
      option.style.cssText = 'background:#1a1a2e;color:#b0b0b0;';
      if (opt === defaultVal) option.selected = true;
      select.appendChild(option);
    });

    select.addEventListener('change', () => {
      onChange(select.value);
    });

    wrap.appendChild(select);
    parent.appendChild(wrap);
    return select;
  }

  // --- Row 1: Seed Shape dropdown, Reset button, Show Walkers toggle ---

  // seed shape dropdown
  makeDropdown('Seed:', ['Point', 'Line', 'Circle'], seedShape, controlsDiv1, (val) => {
    seedShape = val;
    resetSimulation();
  });

  // reset button — clear karo aur naye se shuru
  makeButton('Reset', controlsDiv1, () => {
    resetSimulation();
  });

  // separator
  const sep = document.createElement('span');
  sep.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.3);font-size:11px;padding:0 2px;';
  sep.textContent = '|';
  controlsDiv1.appendChild(sep);

  // show walkers toggle
  const walkerBtn = makeButton('Walkers: Off', controlsDiv1, () => {
    showWalkers = !showWalkers;
    walkerBtn.textContent = showWalkers ? 'Walkers: On' : 'Walkers: Off';
    setActive(walkerBtn, showWalkers);
  });

  // --- Row 2: Sliders ---

  // stickiness slider — low = branchy fractal, high = compact blob
  makeSlider('Stick', '0.1', '1.0', stickiness.toString(), '0.05', controlsDiv2, (v) => {
    stickiness = v;
  });

  // speed slider — walkers per frame
  makeSlider('Speed', '100', '2000', walkersPerFrame.toString(), '100', controlsDiv2, (v) => {
    walkersPerFrame = v;
  });

  // --- Animation loop ---
  function gameLoop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animFrameId = null; return; }
    if (!isVisible) {
      animFrameId = null;
      return;
    }

    // simulation step — crystal grow karo
    simulateFrame();

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
  const resizeObserver = new ResizeObserver(() => {
    if (isVisible && !animFrameId) {
      render();
    }
  });
  resizeObserver.observe(container);

  // --- Init ---
  // Point seed se shuru — crystal grow hona shuru hoga automatically
  resetSimulation();
  render(); // pehla frame draw kar — blank na dikhe
}
