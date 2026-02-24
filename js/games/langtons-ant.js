// ============================================================
// Langton's Ant — Simplest 2D Turing Machine
// Chaos se order nikalta hai — ~10000 steps ke baad highway
// Multi-color turmites, multiple ants, stunning symmetric patterns
// Ye dekhke Turing khud impress ho jaata
// ============================================================

// main entry point — container dhundho, grid banao, ant chal padega
export function initLangtonsAnt() {
  const container = document.getElementById('langtonsAntContainer');
  if (!container) {
    console.warn('langtonsAntContainer nahi mila bhai, Langton skip');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';

  // grid resolution — enough detail for highway to emerge
  const GRID_W = 200;
  const GRID_H = 150;

  // --- Rainbow palette for multi-color turmites ---
  // har color state ka apna RGB hoga — visually distinct hona chahiye
  function buildPalette(numColors) {
    const pal = [];
    for (let i = 0; i < numColors; i++) {
      // HSL to RGB — evenly spaced hues, high saturation
      const hue = (i / numColors) * 360;
      const s = 0.85;
      const l = 0.6;
      // HSL to RGB conversion — standard formula
      const c = (1 - Math.abs(2 * l - 1)) * s;
      const x = c * (1 - Math.abs((hue / 60) % 2 - 1));
      const m = l - c / 2;
      let r, g, b;
      if (hue < 60)       { r = c; g = x; b = 0; }
      else if (hue < 120) { r = x; g = c; b = 0; }
      else if (hue < 180) { r = 0; g = c; b = x; }
      else if (hue < 240) { r = 0; g = x; b = c; }
      else if (hue < 300) { r = x; g = 0; b = c; }
      else                 { r = c; g = 0; b = x; }
      pal.push([
        Math.floor((r + m) * 255),
        Math.floor((g + m) * 255),
        Math.floor((b + m) * 255),
      ]);
    }
    return pal;
  }

  // --- Direction vectors ---
  // 0=up, 1=right, 2=down, 3=left (clockwise order)
  const DX = [0, 1, 0, -1];
  const DY = [-1, 0, 1, 0];

  // --- Presets — famous turmite rule strings ---
  const PRESETS = {
    'Classic':   'RL',
    'Symmetric': 'RLR',
    'Complex':   'RLLR',
    'Chaotic':   'LLRR',
  };

  // --- State ---
  let ruleString = 'RL';
  let numColors = ruleString.length;     // rule string length = number of colors
  let palette = buildPalette(numColors);
  let antCount = 1;
  let stepsPerFrame = 500;               // bohot steps per frame — highway jaldi dikhe
  let running = true;
  let isVisible = false;
  let animFrameId = null;
  let totalSteps = 0;

  // grid — Uint8Array mein cell color state (0 to numColors-1)
  let grid = new Uint8Array(GRID_W * GRID_H);

  // ants — array of {x, y, dir}
  // dir: 0=up, 1=right, 2=down, 3=left
  let ants = [];

  // --- Ant initialization ---
  // center se slightly offset karke place kar — multiple ants mein overlap na ho
  function initAnts() {
    ants = [];
    const cx = Math.floor(GRID_W / 2);
    const cy = Math.floor(GRID_H / 2);
    // offsets — ants ko thoda alag jagah rakho
    const offsets = [
      [0, 0],
      [20, 0],
      [0, 20],
      [-20, -20],
    ];
    for (let i = 0; i < antCount; i++) {
      const ox = offsets[i % offsets.length][0];
      const oy = offsets[i % offsets.length][1];
      ants.push({
        x: ((cx + ox) % GRID_W + GRID_W) % GRID_W,
        y: ((cy + oy) % GRID_H + GRID_H) % GRID_H,
        dir: i % 4, // har ant alag direction se shuru kare
      });
    }
  }

  // --- Full reset ---
  function resetSimulation() {
    numColors = ruleString.length;
    palette = buildPalette(numColors);
    grid = new Uint8Array(GRID_W * GRID_H);
    totalSteps = 0;
    initAnts();
  }

  // --- Core simulation: ek step for ek ant ---
  // Classic Langton's Ant rules + multi-color extension
  function stepAnt(ant) {
    const idx = ant.y * GRID_W + ant.x;
    const cellColor = grid[idx];

    // rule string se turn direction nikal
    // 'R' = right turn (clockwise), 'L' = left turn (counter-clockwise)
    const rule = ruleString[cellColor];
    if (rule === 'R') {
      ant.dir = (ant.dir + 1) % 4; // right turn — clockwise
    } else {
      ant.dir = (ant.dir + 3) % 4; // left turn — counter-clockwise (same as -1 mod 4)
    }

    // cell color advance karo — next color, wrap to 0
    grid[idx] = (cellColor + 1) % numColors;

    // ant aage badh — toroidal wrap
    ant.x = (ant.x + DX[ant.dir] + GRID_W) % GRID_W;
    ant.y = (ant.y + DY[ant.dir] + GRID_H) % GRID_H;
  }

  // multiple steps ek frame mein — speed ke liye
  function simulateSteps(count) {
    for (let s = 0; s < count; s++) {
      for (let a = 0; a < ants.length; a++) {
        stepAnt(ants[a]);
      }
    }
    totalSteps += count;
  }

  // --- DOM structure banao ---
  // pehle existing children preserve nahi karna — clean slate
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // canvas — grid render hoga yahan
  // canvas element size = grid size, CSS stretch karega
  // pixelated rendering — crisp pixels chahiye, blur nahi
  const canvas = document.createElement('canvas');
  canvas.width = GRID_W;
  canvas.height = GRID_H;
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'background:#0a0a0a',
    'image-rendering:pixelated',
    'image-rendering:crisp-edges',
  ].join(';');
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  // ImageData ek baar banao, reuse karo — har frame pe naya nahi banana
  const imageData = ctx.createImageData(GRID_W, GRID_H);
  const pixels = imageData.data;

  // --- Step counter display ---
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

  const stepLabel = document.createElement('span');
  const ruleLabel = document.createElement('span');
  statsDiv.appendChild(stepLabel);
  statsDiv.appendChild(ruleLabel);

  // --- Controls Row 1: Play/Pause, Reset, Presets ---
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

  // --- Controls Row 2: Speed slider, Ant count, Rule input ---
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

  // button active state toggle helper
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

  // --- Row 1: Play/Pause, Reset, Presets ---

  // play/pause button
  const playBtn = makeButton('Pause', controlsDiv1, () => {
    running = !running;
    playBtn.textContent = running ? 'Pause' : 'Play';
    setActive(playBtn, running);
  });
  setActive(playBtn, true);

  // reset button — sab saaf, naya shuru
  makeButton('Reset', controlsDiv1, () => {
    resetSimulation();
  });

  // separator
  const sep = document.createElement('span');
  sep.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.3);font-size:11px;padding:0 2px;';
  sep.textContent = '|';
  controlsDiv1.appendChild(sep);

  // preset buttons — har preset highlight hoga jab active
  let activePresetBtn = null;
  const presetBtns = {};

  Object.keys(PRESETS).forEach(name => {
    const rule = PRESETS[name];
    const btn = makeButton(name, controlsDiv1, () => {
      ruleString = rule;
      ruleInput.value = rule;
      resetSimulation();
      // highlight active preset
      if (activePresetBtn) setActive(activePresetBtn, false);
      setActive(btn, true);
      activePresetBtn = btn;
      // auto-play shuru kar
      running = true;
      playBtn.textContent = 'Pause';
      setActive(playBtn, true);
    });
    presetBtns[name] = btn;
  });

  // default preset highlight
  setActive(presetBtns['Classic'], true);
  activePresetBtn = presetBtns['Classic'];

  // --- Row 2: Speed, Ants, Rule input ---

  // speed slider — 1 to 5000 steps per frame
  const speedWrap = document.createElement('div');
  speedWrap.style.cssText = [
    'display:flex',
    'align-items:center',
    'gap:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:#888',
  ].join(';');

  const speedLbl = document.createElement('span');
  speedLbl.textContent = 'Speed';
  speedLbl.style.cssText = 'white-space:nowrap;';
  speedWrap.appendChild(speedLbl);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '5000';
  speedSlider.value = String(stepsPerFrame);
  speedSlider.style.cssText = 'width:80px;accent-color:' + ACCENT + ';cursor:pointer;';
  speedWrap.appendChild(speedSlider);

  const speedVal = document.createElement('span');
  speedVal.textContent = String(stepsPerFrame);
  speedVal.style.cssText = 'min-width:38px;text-align:right;color:' + ACCENT + ';font-size:10px;';
  speedWrap.appendChild(speedVal);

  speedSlider.addEventListener('input', () => {
    stepsPerFrame = parseInt(speedSlider.value);
    speedVal.textContent = String(stepsPerFrame);
  });
  controlsDiv2.appendChild(speedWrap);

  // ant count selector — 1 to 4
  const antWrap = document.createElement('div');
  antWrap.style.cssText = [
    'display:flex',
    'align-items:center',
    'gap:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:#888',
  ].join(';');

  const antLbl = document.createElement('span');
  antLbl.textContent = 'Ants';
  antLbl.style.cssText = 'white-space:nowrap;';
  antWrap.appendChild(antLbl);

  const antSlider = document.createElement('input');
  antSlider.type = 'range';
  antSlider.min = '1';
  antSlider.max = '4';
  antSlider.value = String(antCount);
  antSlider.style.cssText = 'width:50px;accent-color:' + ACCENT + ';cursor:pointer;';
  antWrap.appendChild(antSlider);

  const antVal = document.createElement('span');
  antVal.textContent = String(antCount);
  antVal.style.cssText = 'min-width:12px;text-align:right;color:' + ACCENT + ';font-size:10px;';
  antWrap.appendChild(antVal);

  antSlider.addEventListener('input', () => {
    antCount = parseInt(antSlider.value);
    antVal.textContent = String(antCount);
    resetSimulation();
  });
  controlsDiv2.appendChild(antWrap);

  // rule string input — custom turmite rules
  const ruleWrap = document.createElement('div');
  ruleWrap.style.cssText = [
    'display:flex',
    'align-items:center',
    'gap:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:#888',
  ].join(';');

  const ruleLbl = document.createElement('span');
  ruleLbl.textContent = 'Rule';
  ruleLbl.style.cssText = 'white-space:nowrap;';
  ruleWrap.appendChild(ruleLbl);

  const ruleInput = document.createElement('input');
  ruleInput.type = 'text';
  ruleInput.value = ruleString;
  ruleInput.maxLength = 12;
  ruleInput.style.cssText = [
    'width:70px',
    'background:rgba(' + ACCENT_RGB + ',0.08)',
    'color:' + ACCENT,
    'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
    'border-radius:4px',
    'padding:3px 6px',
    'font-size:11px',
    'font-family:"JetBrains Mono",monospace',
    'text-transform:uppercase',
    'text-align:center',
    'outline:none',
  ].join(';');

  ruleInput.addEventListener('focus', () => {
    ruleInput.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.5)';
  });
  ruleInput.addEventListener('blur', () => {
    ruleInput.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
  });

  // Enter ya blur pe rule apply kar
  function applyRule() {
    // sirf R aur L allowed hain — baaki sab ignore
    const cleaned = ruleInput.value.toUpperCase().replace(/[^RL]/g, '');
    if (cleaned.length < 2) {
      // minimum 2 characters chahiye — warna classic default
      ruleInput.value = ruleString;
      return;
    }
    ruleString = cleaned;
    ruleInput.value = cleaned;
    resetSimulation();
    // check agar koi preset match karta hai
    updatePresetHighlight();
    // auto-play
    running = true;
    playBtn.textContent = 'Pause';
    setActive(playBtn, true);
  }

  ruleInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      applyRule();
      ruleInput.blur();
    }
  });
  ruleInput.addEventListener('blur', () => {
    // sirf apply karo agar value badal gayi hai
    const cleaned = ruleInput.value.toUpperCase().replace(/[^RL]/g, '');
    if (cleaned.length >= 2 && cleaned !== ruleString) {
      applyRule();
    } else {
      ruleInput.value = ruleString;
    }
  });

  ruleWrap.appendChild(ruleInput);
  controlsDiv2.appendChild(ruleWrap);

  // preset highlight update — slider/input se change hone pe
  function updatePresetHighlight() {
    if (activePresetBtn) {
      setActive(activePresetBtn, false);
      activePresetBtn = null;
    }
    for (const name of Object.keys(PRESETS)) {
      if (PRESETS[name] === ruleString) {
        setActive(presetBtns[name], true);
        activePresetBtn = presetBtns[name];
        break;
      }
    }
  }

  // --- Rendering ---
  // grid ko ImageData mein convert kar — fast pixel-level rendering
  function render() {
    const w = GRID_W;
    const h = GRID_H;
    const nc = numColors;
    const isClassic = nc === 2;

    for (let i = 0; i < w * h; i++) {
      const pi = i * 4;
      const cellColor = grid[i];

      if (cellColor === 0) {
        // empty cell — dark background
        pixels[pi]     = 10;
        pixels[pi + 1] = 10;
        pixels[pi + 2] = 10;
      } else if (isClassic) {
        // classic 2-color mode — black cells white/purple-ish dikhao
        pixels[pi]     = 180;
        pixels[pi + 1] = 160;
        pixels[pi + 2] = 240;
      } else {
        // multi-color mode — rainbow palette se color utha
        const col = palette[cellColor];
        pixels[pi]     = col[0];
        pixels[pi + 1] = col[1];
        pixels[pi + 2] = col[2];
      }
      pixels[pi + 3] = 255; // alpha — always opaque
    }

    // ant positions mark kar — bright dot taaki dikhe kahan hai
    // ant colors — har ant ka alag color (laal, hara, neela, peela)
    const antColors = [
      [255, 60, 60],    // laal
      [60, 255, 100],   // hara
      [60, 150, 255],   // neela
      [255, 230, 50],   // peela
    ];

    for (let a = 0; a < ants.length; a++) {
      const ant = ants[a];
      // 3x3 area mein ant dikhao — ek pixel bahut chhota lagta hai
      const col = antColors[a % antColors.length];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ax = ((ant.x + dx) % w + w) % w;
          const ay = ((ant.y + dy) % h + h) % h;
          const api = (ay * w + ax) * 4;
          pixels[api]     = col[0];
          pixels[api + 1] = col[1];
          pixels[api + 2] = col[2];
          pixels[api + 3] = 255;
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }

  // --- Stats update ---
  function updateStats() {
    // step count — comma-separated for readability
    stepLabel.textContent = 'Steps: ' + totalSteps.toLocaleString();
    ruleLabel.textContent = 'Rule: ' + ruleString;

    // step label purple karo jab simulation chal rahi ho
    if (running) {
      stepLabel.style.color = ACCENT;
    } else {
      stepLabel.style.color = 'rgba(255,255,255,0.5)';
    }
    ruleLabel.style.color = 'rgba(255,255,255,0.4)';
  }

  // --- Animation loop ---
  function gameLoop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animFrameId = null; return; }
    if (!isVisible) {
      animFrameId = null;
      return;
    }

    // simulation steps — multiple per frame
    if (running) {
      simulateSteps(stepsPerFrame);
    }

    // render har frame pe
    render();
    updateStats();

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

  // --- IntersectionObserver — sirf visible hone pe run kar ---
  // background mein CPU waste nahi karna — performance ke liye zaroori
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

  // tab switch pe bhi pause kar — background tab mein CPU waste na ho
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      isVisible = false;
      stopLoop();
    } else {
      // check agar container abhi bhi visible hai
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) {
        isVisible = true;
        startLoop();
      }
    }
  });

  // --- Resize handling ---
  // canvas element size fixed hai (GRID_W x GRID_H), CSS stretch karta hai
  // resize pe sirf re-render karna hai agar loop nahi chal raha
  const resizeObserver = new ResizeObserver(() => {
    if (isVisible && !animFrameId) {
      render();
    }
  });
  resizeObserver.observe(container);

  // --- Init ---
  resetSimulation();
  render();
  updateStats();
}
