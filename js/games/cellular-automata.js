// ============================================================
// Elementary Cellular Automata — Wolfram Rules (0-255)
// 1D automaton: 3-cell neighborhood, 8-bit rule table
// Rule 30 chaotic, Rule 90 Sierpinski, Rule 110 Turing complete
// Classic Wolfram diagrams — har row ek generation hai
// ============================================================

// entry point — container dhundho aur automaton shuru karo
export function initCellularAutomata() {
  const container = document.getElementById('cellularAutomataContainer');
  if (!container) {
    console.warn('cellularAutomataContainer nahi mila DOM mein');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';
  const CELL_ON_COLOR = { r: 34, g: 211, b: 238 };  // #22d3ee — cyan
  const CELL_OFF_COLOR = { r: 10, g: 10, b: 10 };    // #0a0a0a — dark
  const CELL_PX = 2; // har cell kitne pixels wide/tall — information density max

  // --- State ---
  let ruleNumber = 30; // default Rule 30 — Wolfram ka favorite
  let ruleTable = new Uint8Array(8); // 8 patterns ka output
  let gridWidth = 0;
  let currentRow = null; // Uint8Array — current generation
  let displayBuffer = null; // sab rows ka buffer — ImageData mein render hoga
  let totalRows = 0; // canvas mein kitni rows fit hongi
  let currentGeneration = 0; // kitni generations compute ho chuki hain
  let rowsRendered = 0; // buffer mein kitni rows bhar chuki hain
  let initMode = 'single'; // 'single', 'random', 'custom'
  let customRow = null; // custom mode mein user ki row
  let speed = 5; // generations per frame
  let running = true;
  let isVisible = false;
  let animFrameId = null;
  let needsFullRedraw = false;

  // --- Rule table compute kar ---
  // rule number ka 8-bit binary representation = 8 patterns ka output
  // pattern index: left*4 + center*2 + right (MSB to LSB)
  function computeRuleTable(rule) {
    for (let i = 0; i < 8; i++) {
      ruleTable[i] = (rule >> i) & 1;
    }
  }

  // --- Grid initialization ---
  function initGrid() {
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const dpr = window.devicePixelRatio || 1;
    const canvasPixelW = Math.floor(w * dpr);
    const canvasPixelH = Math.floor(CANVAS_HEIGHT * dpr);

    gridWidth = Math.floor(canvasPixelW / CELL_PX);
    totalRows = Math.floor(canvasPixelH / CELL_PX);

    // canvas resize kar
    canvas.width = gridWidth * CELL_PX;
    canvas.height = totalRows * CELL_PX;
    canvas.style.width = w + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    // display buffer banao — sab rows store hongi
    displayBuffer = new Uint8Array(gridWidth * totalRows);

    resetSimulation();
  }

  // --- Simulation reset (rule aur init mode ke hisaab se) ---
  function resetSimulation() {
    computeRuleTable(ruleNumber);
    currentRow = new Uint8Array(gridWidth);

    if (initMode === 'single') {
      // sirf center cell ON
      currentRow[Math.floor(gridWidth / 2)] = 1;
    } else if (initMode === 'random') {
      // ~50% random fill
      for (let i = 0; i < gridWidth; i++) {
        currentRow[i] = Math.random() < 0.5 ? 1 : 0;
      }
    } else if (initMode === 'custom' && customRow) {
      // custom row copy kar — agar size match nahi karta toh center align
      const offset = Math.max(0, Math.floor((gridWidth - customRow.length) / 2));
      currentRow.fill(0);
      for (let i = 0; i < customRow.length && i + offset < gridWidth; i++) {
        currentRow[i + offset] = customRow[i];
      }
    } else {
      // fallback single center
      currentRow[Math.floor(gridWidth / 2)] = 1;
    }

    // display buffer saaf karo
    displayBuffer.fill(0);
    // pehli row copy karo buffer mein
    displayBuffer.set(currentRow, 0);

    currentGeneration = 0;
    rowsRendered = 1;
    needsFullRedraw = true;
  }

  // --- Ek generation advance karo ---
  // elementary CA rule: left, center, right neighbors dekho, rule table se output lo
  function stepGeneration() {
    const newRow = new Uint8Array(gridWidth);
    const w = gridWidth;

    for (let i = 0; i < w; i++) {
      // neighbors — wrap around (toroidal)
      const left = currentRow[(i - 1 + w) % w];
      const center = currentRow[i];
      const right = currentRow[(i + 1) % w];

      // 3-bit pattern: left*4 + center*2 + right
      const pattern = (left << 2) | (center << 1) | right;
      newRow[i] = ruleTable[pattern];
    }

    currentRow = newRow;
    currentGeneration++;

    // buffer mein row add karo
    if (rowsRendered < totalRows) {
      // abhi buffer full nahi hua — neeche add karo
      displayBuffer.set(currentRow, rowsRendered * gridWidth);
      rowsRendered++;
    } else {
      // buffer full — sab rows ek upar shift karo, neeche nayi row
      displayBuffer.copyWithin(0, gridWidth);
      displayBuffer.set(currentRow, (totalRows - 1) * gridWidth);
      needsFullRedraw = true;
    }
  }

  // --- DOM Structure ---
  // pehle existing children preserve karo nahi — clean slate
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // rule visualization — 8 patterns dikhayenge
  const ruleVizDiv = document.createElement('div');
  ruleVizDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:6px',
    'margin-bottom:10px',
    'align-items:center',
    'justify-content:center',
    'min-height:48px',
  ].join(';');
  container.appendChild(ruleVizDiv);

  // canvas — yahan automaton render hoga
  const canvas = document.createElement('canvas');
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

  // stats bar — generation count
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

  const genLabel = document.createElement('span');
  const ruleLabel = document.createElement('span');
  statsDiv.appendChild(genLabel);
  statsDiv.appendChild(ruleLabel);

  // controls row 1 — rule input, presets
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

  // controls row 2 — init mode, speed, reset
  const controlsDiv2 = document.createElement('div');
  controlsDiv2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:6px',
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
      if (!btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
        btn.style.color = '#ffffff';
      }
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

  // button active state helper
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

  // --- Rule visualization — 8 patterns show karo ---
  // har pattern: 3 input cells (top) → 1 output cell (bottom)
  // ye mini pixel art diagrams hain — Wolfram style
  function drawRuleVisualization() {
    while (ruleVizDiv.firstChild) ruleVizDiv.removeChild(ruleVizDiv.firstChild);

    const cellSz = 10; // har cell ka size pixels mein
    const gap = 1;

    // 8 patterns — 7 (111) se 0 (000) tak standard Wolfram order
    for (let p = 7; p >= 0; p--) {
      const patternCanvas = document.createElement('canvas');
      const pW = cellSz * 3 + gap * 2;
      const pH = cellSz * 2 + gap + 4; // 2 rows + gap + spacing
      patternCanvas.width = pW;
      patternCanvas.height = pH;
      patternCanvas.style.cssText = 'width:' + pW + 'px;height:' + pH + 'px;image-rendering:pixelated;';

      const pCtx = patternCanvas.getContext('2d');

      // 3 input cells — top row
      const bits = [(p >> 2) & 1, (p >> 1) & 1, p & 1];
      for (let i = 0; i < 3; i++) {
        const x = i * (cellSz + gap);
        if (bits[i]) {
          pCtx.fillStyle = '#22d3ee';
        } else {
          pCtx.fillStyle = '#2a2a2a';
        }
        pCtx.fillRect(x, 0, cellSz, cellSz);
      }

      // output cell — bottom row, center aligned
      const outX = 1 * (cellSz + gap);
      const outY = cellSz + gap + 2;
      const output = ruleTable[p];
      if (output) {
        pCtx.fillStyle = '#22d3ee';
      } else {
        pCtx.fillStyle = '#2a2a2a';
      }
      pCtx.fillRect(outX, outY, cellSz, cellSz);

      // subtle arrow indicator — center se neeche
      pCtx.fillStyle = 'rgba(255,255,255,0.15)';
      const arrowX = outX + Math.floor(cellSz / 2);
      pCtx.fillRect(arrowX, cellSz, 1, gap + 2);

      ruleVizDiv.appendChild(patternCanvas);
    }
  }

  // --- Row 1: Rule input + presets ---

  // rule number input wrapper — number field with up/down buttons
  const ruleInputWrap = document.createElement('div');
  ruleInputWrap.style.cssText = [
    'display:flex',
    'align-items:center',
    'gap:4px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:#888',
  ].join(';');
  controlsDiv1.appendChild(ruleInputWrap);

  const ruleLbl = document.createElement('span');
  ruleLbl.textContent = 'Rule:';
  ruleLbl.style.cssText = 'color:#b0b0b0;';
  ruleInputWrap.appendChild(ruleLbl);

  // down button
  const ruleDown = document.createElement('button');
  ruleDown.textContent = '\u25BC';
  ruleDown.style.cssText = [
    'background:rgba(' + ACCENT_RGB + ',0.1)',
    'color:#b0b0b0',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
    'border-radius:4px',
    'padding:2px 6px',
    'font-size:9px',
    'cursor:pointer',
    'line-height:1',
  ].join(';');
  ruleInputWrap.appendChild(ruleDown);

  // number input
  const ruleInput = document.createElement('input');
  ruleInput.type = 'number';
  ruleInput.min = '0';
  ruleInput.max = '255';
  ruleInput.value = '30';
  ruleInput.style.cssText = [
    'width:50px',
    'background:rgba(255,255,255,0.05)',
    'color:' + ACCENT,
    'border:1px solid rgba(' + ACCENT_RGB + ',0.3)',
    'border-radius:4px',
    'padding:4px 6px',
    'font-size:12px',
    'font-family:"JetBrains Mono",monospace',
    'text-align:center',
    'outline:none',
    '-moz-appearance:textfield',
  ].join(';');
  ruleInputWrap.appendChild(ruleInput);

  // up button
  const ruleUp = document.createElement('button');
  ruleUp.textContent = '\u25B2';
  ruleUp.style.cssText = ruleDown.style.cssText;
  ruleInputWrap.appendChild(ruleUp);

  // rule input handlers
  function applyRule(num) {
    num = Math.max(0, Math.min(255, Math.floor(num)));
    ruleNumber = num;
    ruleInput.value = num;
    computeRuleTable(num);
    drawRuleVisualization();
    resetSimulation();
    updatePresetHighlight();
  }

  ruleInput.addEventListener('change', () => {
    applyRule(parseInt(ruleInput.value) || 0);
  });
  ruleInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      applyRule(parseInt(ruleInput.value) || 0);
    }
  });
  ruleDown.addEventListener('click', () => {
    applyRule(ruleNumber - 1);
  });
  ruleUp.addEventListener('click', () => {
    applyRule(ruleNumber + 1);
  });

  // separator
  const sep1 = document.createElement('span');
  sep1.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.3);font-size:11px;padding:0 2px;';
  sep1.textContent = '|';
  controlsDiv1.appendChild(sep1);

  // preset buttons — notable rules
  const PRESETS = {
    'Rule 30': 30,
    'Rule 90': 90,
    'Rule 110': 110,
    'Rule 184': 184,
  };

  const presetBtns = {};
  Object.keys(PRESETS).forEach(name => {
    const btn = makeButton(name, controlsDiv1, () => {
      applyRule(PRESETS[name]);
    });
    presetBtns[name] = btn;
  });

  // separator
  const sep2 = document.createElement('span');
  sep2.style.cssText = sep1.style.cssText;
  sep2.textContent = '|';
  controlsDiv1.appendChild(sep2);

  // random rule button
  makeButton('Random Rule', controlsDiv1, () => {
    applyRule(Math.floor(Math.random() * 256));
  });

  // preset highlight update
  function updatePresetHighlight() {
    Object.keys(presetBtns).forEach(name => {
      setActive(presetBtns[name], PRESETS[name] === ruleNumber);
    });
  }

  // --- Row 2: Init mode, speed, reset ---

  // init mode buttons
  const initModes = { 'Single Cell': 'single', 'Random': 'random', 'Custom': 'custom' };
  const initBtns = {};

  Object.keys(initModes).forEach(label => {
    const mode = initModes[label];
    const btn = makeButton(label, controlsDiv2, () => {
      initMode = mode;
      if (mode === 'custom') {
        // custom mode mein — pehle ek empty row banao, user click karega
        customRow = new Uint8Array(gridWidth);
        resetSimulation();
        // running band karo — user paint karega pehle
        running = false;
        playBtn.textContent = 'Play';
        setActive(playBtn, false);
      } else {
        customRow = null;
        resetSimulation();
      }
      updateInitHighlight();
    });
    initBtns[label] = btn;
  });

  function updateInitHighlight() {
    Object.keys(initBtns).forEach(label => {
      setActive(initBtns[label], initModes[label] === initMode);
    });
  }

  // separator
  const sep3 = document.createElement('span');
  sep3.style.cssText = sep1.style.cssText;
  sep3.textContent = '|';
  controlsDiv2.appendChild(sep3);

  // speed slider
  const speedWrap = document.createElement('div');
  speedWrap.style.cssText = [
    'display:flex',
    'align-items:center',
    'gap:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:#888',
  ].join(';');
  controlsDiv2.appendChild(speedWrap);

  const speedLbl = document.createElement('span');
  speedLbl.textContent = 'Speed: 5';
  speedLbl.style.cssText = 'white-space:nowrap;color:#b0b0b0;';
  speedWrap.appendChild(speedLbl);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '50';
  speedSlider.value = '5';
  speedSlider.style.cssText = 'width:80px;accent-color:' + ACCENT + ';cursor:pointer;';
  speedSlider.addEventListener('input', () => {
    speed = parseInt(speedSlider.value);
    speedLbl.textContent = 'Speed: ' + speed;
  });
  speedWrap.appendChild(speedSlider);

  // separator
  const sep4 = document.createElement('span');
  sep4.style.cssText = sep1.style.cssText;
  sep4.textContent = '|';
  controlsDiv2.appendChild(sep4);

  // play/pause button
  const playBtn = makeButton('Pause', controlsDiv2, () => {
    running = !running;
    playBtn.textContent = running ? 'Pause' : 'Play';
    setActive(playBtn, running);
  });
  setActive(playBtn, true);

  // reset button — current rule + init mode ke saath fresh start
  makeButton('Reset', controlsDiv2, () => {
    resetSimulation();
    running = true;
    playBtn.textContent = 'Pause';
    setActive(playBtn, true);
  });

  // --- Canvas click handler — custom mode mein top row toggle ---
  canvas.addEventListener('click', (e) => {
    if (initMode !== 'custom') return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const cssY = e.clientY - rect.top;

    // sirf top row pe click allowed — pehli kuch pixel rows
    // canvas CSS height se map karo
    const rowCSS = CANVAS_HEIGHT / totalRows;
    const clickedRow = Math.floor(cssY / rowCSS);

    // sirf pehli row pe react karo agar simulation reset state mein hai
    if (clickedRow > 2 || rowsRendered > 3) return;

    // column nikal
    const colCSS = rect.width / gridWidth;
    const col = Math.floor(cssX / colCSS);

    if (col >= 0 && col < gridWidth) {
      // toggle cell — pehle naya value nikal, fir brush lagao
      if (!customRow) customRow = new Uint8Array(gridWidth);
      const newVal = customRow[col] ? 0 : 1;
      // thoda bada brush — 1px chhota dikhega, 5px brush use karo
      const brushSize = 2;
      for (let dx = -brushSize; dx <= brushSize; dx++) {
        const c = col + dx;
        if (c >= 0 && c < gridWidth) {
          customRow[c] = newVal;
        }
      }
      // row update karo display mein
      currentRow = new Uint8Array(customRow);
      displayBuffer.fill(0);
      displayBuffer.set(currentRow, 0);
      rowsRendered = 1;
      currentGeneration = 0;
      needsFullRedraw = true;
    }
  });

  // --- Rendering — ImageData se fast pixel rendering ---
  let imageData = null;
  let pixels = null;

  function ensureImageData() {
    if (!imageData || imageData.width !== canvas.width || imageData.height !== canvas.height) {
      imageData = ctx.createImageData(canvas.width, canvas.height);
      pixels = imageData.data;
      needsFullRedraw = true;
    }
  }

  function renderFull() {
    ensureImageData();
    const cw = canvas.width;
    const ch = canvas.height;
    const gw = gridWidth;
    const cpx = CELL_PX;

    // background — sab pixels dark bhar do
    for (let i = 0; i < pixels.length; i += 4) {
      pixels[i]     = CELL_OFF_COLOR.r;
      pixels[i + 1] = CELL_OFF_COLOR.g;
      pixels[i + 2] = CELL_OFF_COLOR.b;
      pixels[i + 3] = 255;
    }

    // har row render karo
    for (let row = 0; row < rowsRendered; row++) {
      const rowOffset = row * gw;
      const pyStart = row * cpx;

      for (let col = 0; col < gw; col++) {
        if (!displayBuffer[rowOffset + col]) continue; // OFF cell — already dark

        const pxStart = col * cpx;

        // CELL_PX x CELL_PX block bhar do
        for (let dy = 0; dy < cpx; dy++) {
          const py = pyStart + dy;
          if (py >= ch) break;
          for (let dx = 0; dx < cpx; dx++) {
            const px = pxStart + dx;
            if (px >= cw) break;
            const pi = (py * cw + px) * 4;
            pixels[pi]     = CELL_ON_COLOR.r;
            pixels[pi + 1] = CELL_ON_COLOR.g;
            pixels[pi + 2] = CELL_ON_COLOR.b;
            // alpha already 255
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
    needsFullRedraw = false;
  }

  // incremental render — sirf nayi row draw karo (jab scroll nahi ho raha)
  function renderIncrementalRow(rowIdx) {
    ensureImageData();
    const cw = canvas.width;
    const gw = gridWidth;
    const cpx = CELL_PX;
    const rowOffset = rowIdx * gw;
    const pyStart = rowIdx * cpx;

    for (let col = 0; col < gw; col++) {
      const on = displayBuffer[rowOffset + col];
      const color = on ? CELL_ON_COLOR : CELL_OFF_COLOR;
      const pxStart = col * cpx;

      for (let dy = 0; dy < cpx; dy++) {
        const py = pyStart + dy;
        if (py >= canvas.height) break;
        for (let dx = 0; dx < cpx; dx++) {
          const px = pxStart + dx;
          if (px >= cw) break;
          const pi = (py * cw + px) * 4;
          pixels[pi]     = color.r;
          pixels[pi + 1] = color.g;
          pixels[pi + 2] = color.b;
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }

  // --- Stats update ---
  function updateStats() {
    genLabel.textContent = 'Gen: ' + currentGeneration;
    ruleLabel.textContent = 'Rule: ' + ruleNumber;
    ruleLabel.style.color = ACCENT;
  }

  // --- Animation loop ---
  function gameLoop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animFrameId = null; return; }
    if (!isVisible) {
      animFrameId = null;
      return;
    }

    if (running && gridWidth > 0) {
      const prevRendered = rowsRendered;
      for (let s = 0; s < speed; s++) {
        stepGeneration();
      }

      // decide karo full redraw chahiye ya incremental
      if (needsFullRedraw) {
        renderFull();
      } else {
        // incremental — sirf nayi rows render karo
        for (let r = prevRendered; r < rowsRendered; r++) {
          renderIncrementalRow(r);
        }
      }
    } else if (needsFullRedraw) {
      renderFull();
    }

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

  // tab switch pe pause karo
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
  const resizeObserver = new ResizeObserver(() => {
    initGrid();
  });
  resizeObserver.observe(container);

  // --- Init ---
  computeRuleTable(ruleNumber);
  drawRuleVisualization();
  updatePresetHighlight();
  updateInitHighlight();
  initGrid();
  renderFull();
  updateStats();
}
