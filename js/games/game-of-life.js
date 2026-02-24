// ============================================================
// Conway's Game of Life — Dark Portfolio ke liye
// B3/S23 rules, toroidal grid, preset patterns, paint mode
// Classic cellular automaton — ye toh CS ka Shakespeare hai
// ============================================================

// main entry point — container dhundho aur life simulation shuru karo
export function initGameOfLife() {
  const container = document.getElementById('gameOfLifeContainer');
  if (!container) {
    console.warn('gameOfLifeContainer nahi mila DOM mein');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 380;
  const CELL_TARGET = 10; // target cell size in CSS pixels — auto-fit hoga
  const FADE_STEPS = 6; // marne ke baad kitne frames mein fade out ho
  const ACCENT = '#a78bfa'; // purple accent — portfolio theme

  // --- State ---
  let cols = 0;
  let rows = 0;
  let cellSize = CELL_TARGET;
  let grid = []; // 2D array — 0 = dead, 1+ = alive (age counter)
  let fadeGrid = []; // marne ke baad fade track karta hai — FADE_STEPS se 0 tak
  let generation = 0;
  let aliveCount = 0;
  let running = false;
  let speed = 10; // generations per second
  let isVisible = false;
  let animFrameId = null;
  let lastStepTime = 0;

  // drawing state — click/drag se cells toggle karna
  let isDrawing = false;
  let drawValue = 1; // 1 = alive bana, 0 = dead bana

  // --- Grid initialization ---
  // cols aur rows calculate kar canvas size se
  function calcGridSize(canvasW, canvasH) {
    // CSS pixels mein kaam kar — DPR canvas internal hai
    cols = Math.floor(canvasW / CELL_TARGET);
    rows = Math.floor(canvasH / CELL_TARGET);
    // clamp — bahut chhota ya bahut bada nahi chahiye
    cols = Math.max(20, Math.min(120, cols));
    rows = Math.max(15, Math.min(80, rows));
    // actual cell size calculate kar — leftover space distribute kar
    cellSize = Math.floor(canvasW / cols);
  }

  // empty grid bana — sab dead
  function createEmptyGrid() {
    grid = Array.from({ length: rows }, () => new Int8Array(cols));
    fadeGrid = Array.from({ length: rows }, () => new Int8Array(cols));
    generation = 0;
    aliveCount = 0;
  }

  // random fill — density % cells alive bana
  function randomFill(density) {
    createEmptyGrid();
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (Math.random() < density) {
          grid[r][c] = 1;
        }
      }
    }
    generation = 0;
    countAlive();
  }

  // alive cells gino
  function countAlive() {
    let count = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (grid[r][c] > 0) count++;
      }
    }
    aliveCount = count;
  }

  // --- Conway's Game of Life rules (B3/S23, toroidal) ---
  // ye hai core logic — ek generation advance kar
  function stepGeneration() {
    const newGrid = Array.from({ length: rows }, () => new Int8Array(cols));

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        // neighbors gino — 8-connected, wrap-around (toroidal)
        let neighbors = 0;
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            if (dr === 0 && dc === 0) continue;
            // toroidal wrap — edges se doosri side pe jaao
            const nr = (r + dr + rows) % rows;
            const nc = (c + dc + cols) % cols;
            if (grid[nr][nc] > 0) neighbors++;
          }
        }

        const alive = grid[r][c] > 0;

        if (alive) {
          if (neighbors === 2 || neighbors === 3) {
            // survive — age badhao (max 127 — Int8Array limit)
            newGrid[r][c] = Math.min(127, grid[r][c] + 1);
          } else {
            // mar gaya — fade start kar
            newGrid[r][c] = 0;
            fadeGrid[r][c] = FADE_STEPS;
          }
        } else {
          if (neighbors === 3) {
            // birth! — naya cell paida hua
            newGrid[r][c] = 1;
            fadeGrid[r][c] = 0;
          } else {
            newGrid[r][c] = 0;
          }
        }
      }
    }

    // fade grid update kar — har frame pe ek step kam
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (fadeGrid[r][c] > 0 && newGrid[r][c] === 0) {
          fadeGrid[r][c]--;
        } else if (newGrid[r][c] > 0) {
          fadeGrid[r][c] = 0;
        }
      }
    }

    grid = newGrid;
    generation++;
    countAlive();
  }

  // --- Preset Patterns ---
  // center of grid pe place karenge
  function placePattern(pattern) {
    createEmptyGrid();
    const startR = Math.floor(rows / 2 - pattern.length / 2);
    const startC = Math.floor(cols / 2 - (pattern[0] ? pattern[0].length / 2 : 0));

    for (let r = 0; r < pattern.length; r++) {
      for (let c = 0; c < pattern[r].length; c++) {
        const gr = startR + r;
        const gc = startC + c;
        if (gr >= 0 && gr < rows && gc >= 0 && gc < cols) {
          grid[gr][gc] = pattern[r][c] ? 1 : 0;
        }
      }
    }
    generation = 0;
    countAlive();
  }

  // pattern definitions — 1 = alive, 0 = dead
  const PRESETS = {
    'Glider': [
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 1],
    ],
    'Pulsar': (() => {
      // 13x13 pattern — period 3 oscillator, symmetric beast
      const p = Array.from({ length: 13 }, () => new Array(13).fill(0));
      const coords = [
        // top-left quadrant mirrored 4 times
        [0,2],[0,3],[0,4],[0,8],[0,9],[0,10],
        [2,0],[2,5],[2,7],[2,12],
        [3,0],[3,5],[3,7],[3,12],
        [4,0],[4,5],[4,7],[4,12],
        [5,2],[5,3],[5,4],[5,8],[5,9],[5,10],
        // mirror vertically
        [7,2],[7,3],[7,4],[7,8],[7,9],[7,10],
        [8,0],[8,5],[8,7],[8,12],
        [9,0],[9,5],[9,7],[9,12],
        [10,0],[10,5],[10,7],[10,12],
        [12,2],[12,3],[12,4],[12,8],[12,9],[12,10],
      ];
      coords.forEach(([r, c]) => { p[r][c] = 1; });
      return p;
    })(),
    'LWSS': [
      // lightweight spaceship — moves right
      [0, 1, 0, 0, 1],
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 1],
      [1, 1, 1, 1, 0],
    ],
    'Gosper Gun': (() => {
      // Gosper Glider Gun — period 30 gun, har 30 gen pe ek glider
      // canonical coordinates — row, col format (0-indexed)
      const g = Array.from({ length: 11 }, () => new Array(38).fill(0));
      // ye coordinates LifeWiki se verified hain
      const cells = [
        // left square
        [5,1],[5,2],[6,1],[6,2],
        // left part of gun
        [5,11],[6,11],[7,11],
        [4,12],[8,12],
        [3,13],[9,13],
        [3,14],[9,14],
        [6,15],
        [4,16],[8,16],
        [5,17],[6,17],[7,17],
        [6,18],
        // right part of gun
        [3,21],[4,21],[5,21],
        [3,22],[4,22],[5,22],
        [2,23],[6,23],
        [1,25],[2,25],[6,25],[7,25],
        // right square
        [3,35],[4,35],[3,36],[4,36],
      ];
      cells.forEach(([r, c]) => {
        if (r >= 0 && r < 11 && c >= 0 && c < 38) g[r][c] = 1;
      });
      return g;
    })(),
    'R-pentomino': [
      [0, 1, 1],
      [1, 1, 0],
      [0, 1, 0],
    ],
  };

  // --- DOM Structure ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // canvas — yahan grid render hoga
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(167,139,250,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
    'touch-action:none',
  ].join(';');
  container.appendChild(canvas);

  // stats bar — generation count + alive count
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
  const aliveLabel = document.createElement('span');
  statsDiv.appendChild(genLabel);
  statsDiv.appendChild(aliveLabel);

  // controls row 1 — play/pause, step, clear, random
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

  // controls row 2 — speed slider + presets
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
      'background:rgba(167,139,250,0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(167,139,250,0.25)',
      'border-radius:6px',
      'padding:5px 12px',
      'font-size:11px',
      'font-family:"JetBrains Mono",monospace',
      'cursor:pointer',
      'transition:all 0.2s',
      'user-select:none',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(167,139,250,0.25)';
      btn.style.color = '#ffffff';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(167,139,250,0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    parent.appendChild(btn);
    return btn;
  }

  // play/pause button — toggle simulation
  const playBtn = makeButton('Play', controlsDiv1, () => {
    running = !running;
    playBtn.textContent = running ? 'Pause' : 'Play';
    if (running) {
      playBtn.style.borderColor = 'rgba(167,139,250,0.5)';
      playBtn.style.color = ACCENT;
    } else {
      playBtn.style.borderColor = 'rgba(167,139,250,0.25)';
      playBtn.style.color = '#b0b0b0';
    }
  });

  // step button — ek generation aage badh
  makeButton('Step', controlsDiv1, () => {
    stepGeneration();
  });

  // clear button — sab saaf kar do
  makeButton('Clear', controlsDiv1, () => {
    running = false;
    playBtn.textContent = 'Play';
    playBtn.style.borderColor = 'rgba(167,139,250,0.25)';
    playBtn.style.color = '#b0b0b0';
    createEmptyGrid();
  });

  // random fill — 30% density
  makeButton('Random', controlsDiv1, () => {
    randomFill(0.3);
  });

  // --- Row 2: Speed slider ---
  const speedLabel = document.createElement('span');
  speedLabel.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
  speedLabel.textContent = 'Speed: 10';
  controlsDiv2.appendChild(speedLabel);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '30';
  speedSlider.value = '10';
  speedSlider.style.cssText = 'width:80px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
  speedSlider.addEventListener('input', () => {
    speed = parseInt(speedSlider.value);
    speedLabel.textContent = 'Speed: ' + speed;
  });
  controlsDiv2.appendChild(speedSlider);

  // --- Preset buttons ---
  // separator — thoda gap
  const sepSpan = document.createElement('span');
  sepSpan.style.cssText = 'color:rgba(167,139,250,0.3);font-size:11px;padding:0 4px;';
  sepSpan.textContent = '|';
  controlsDiv2.appendChild(sepSpan);

  Object.keys(PRESETS).forEach(name => {
    makeButton(name, controlsDiv2, () => {
      placePattern(PRESETS[name]);
      // auto-play shuru kar preset place karne pe
      running = true;
      playBtn.textContent = 'Pause';
      playBtn.style.borderColor = 'rgba(167,139,250,0.5)';
      playBtn.style.color = ACCENT;
    });
  });

  // --- Canvas sizing ---
  function resizeCanvas() {
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(CANVAS_HEIGHT * dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    // grid size recalculate kar — agar pehli baar hai toh grid bhi bana
    const oldCols = cols;
    const oldRows = rows;
    calcGridSize(w, CANVAS_HEIGHT);

    // agar grid size badal gaya toh naya grid bana
    if (cols !== oldCols || rows !== oldRows || grid.length === 0) {
      // try to preserve existing cells agar possible ho
      if (grid.length > 0 && oldCols > 0 && oldRows > 0) {
        const oldGrid = grid;
        const oldFade = fadeGrid;
        createEmptyGrid();
        // purane cells copy kar — jitne fit ho utne
        const minR = Math.min(oldRows, rows);
        const minC = Math.min(oldCols, cols);
        for (let r = 0; r < minR; r++) {
          for (let c = 0; c < minC; c++) {
            grid[r][c] = oldGrid[r] ? (oldGrid[r][c] || 0) : 0;
            fadeGrid[r][c] = oldFade[r] ? (oldFade[r][c] || 0) : 0;
          }
        }
        countAlive();
      } else {
        createEmptyGrid();
      }
    }
  }

  // --- Mouse/touch handlers — paint mode ---
  function getCellFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;

    if (e.touches) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    // CSS coords se grid coords nikal
    const cssX = clientX - rect.left;
    const cssY = clientY - rect.top;
    const c = Math.floor(cssX / cellSize);
    const r = Math.floor(cssY / cellSize);

    if (r >= 0 && r < rows && c >= 0 && c < cols) return { r, c };
    return null;
  }

  function handlePointerDown(e) {
    const cell = getCellFromEvent(e);
    if (!cell) return;
    isDrawing = true;
    // pehla click decide karega ki paint kar rahe hain ya erase
    drawValue = grid[cell.r][cell.c] > 0 ? 0 : 1;
    applyDraw(cell);
  }

  function handlePointerMove(e) {
    if (!isDrawing) return;
    const cell = getCellFromEvent(e);
    if (!cell) return;
    applyDraw(cell);
  }

  function handlePointerUp() {
    isDrawing = false;
  }

  function applyDraw(cell) {
    if (drawValue === 1) {
      grid[cell.r][cell.c] = 1;
      fadeGrid[cell.r][cell.c] = 0;
    } else {
      grid[cell.r][cell.c] = 0;
      fadeGrid[cell.r][cell.c] = 0;
    }
    countAlive();
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

  // --- Canvas rendering ---
  // rounded rect helper — duplicate code hatao
  function roundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function draw() {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const dpr = window.devicePixelRatio || 1;
    const cs = cellSize * dpr; // cell size in canvas pixels

    ctx.clearRect(0, 0, w, h);

    // grid lines — bahut subtle, ek single path mein batch kar
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let c = 0; c <= cols; c++) {
      const x = Math.floor(c * cs) + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, rows * cs);
    }
    for (let r = 0; r <= rows; r++) {
      const y = Math.floor(r * cs) + 0.5;
      ctx.moveTo(0, y);
      ctx.lineTo(cols * cs, y);
    }
    ctx.stroke();

    // cells draw kar
    const cornerR = Math.max(1, cs * 0.15); // rounded corners — subtle
    const pad = cs * 0.1; // cell ke andar padding

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const alive = grid[r][c] > 0;
        const fade = fadeGrid[r][c];

        if (!alive && fade <= 0) continue; // skip empty cells — performance

        const x = c * cs;
        const y = r * cs;
        const sx = x + pad;
        const sy = y + pad;
        const sw = cs - pad * 2;
        const sh = cs - pad * 2;

        if (alive) {
          // alive cell — bright with age-based color
          const age = grid[r][c];
          // naye cells purple, purane cells white-ish
          const t = Math.min(1, age / 20);

          const red = Math.floor(167 + (224 - 167) * t);
          const green = Math.floor(139 + (224 - 139) * t);
          const blue = Math.floor(250 + (255 - 250) * t);

          // glow effect — subtle shadow
          ctx.shadowColor = ACCENT;
          ctx.shadowBlur = age > 3 ? 4 : 8; // naye cells zyada glow
          ctx.fillStyle = `rgb(${red},${green},${blue})`;

          roundedRect(ctx, sx, sy, sw, sh, cornerR);
          ctx.fill();
          ctx.shadowBlur = 0;

        } else if (fade > 0) {
          // recently dead cell — fade out effect
          const alpha = (fade / FADE_STEPS) * 0.35;
          ctx.fillStyle = `rgba(167,139,250,${alpha})`;

          roundedRect(ctx, sx, sy, sw, sh, cornerR);
          ctx.fill();
        }
      }
    }
  }

  // --- Stats update ---
  function updateStats() {
    genLabel.textContent = 'Gen: ' + generation;
    aliveLabel.textContent = 'Alive: ' + aliveCount;

    // alive count ko accent color do agar cells hain
    if (aliveCount > 0) {
      aliveLabel.style.color = ACCENT;
    } else {
      aliveLabel.style.color = 'rgba(255,255,255,0.5)';
    }
  }

  // --- Animation loop ---
  function gameLoop(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animFrameId = null; return; }
    if (!isVisible) {
      animFrameId = null;
      return;
    }

    // simulation step — speed ke hisaab se
    if (running) {
      const stepInterval = 1000 / speed;
      if (timestamp - lastStepTime >= stepInterval) {
        stepGeneration();
        lastStepTime = timestamp;
      }
    }

    // render har frame pe — smooth 60fps
    draw();
    updateStats();

    animFrameId = requestAnimationFrame(gameLoop);
  }

  function startLoop() {
    if (!animFrameId && isVisible) {
      lastStepTime = performance.now();
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
  // performance ke liye zaroori — background mein CPU waste mat kar
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
    { threshold: 0.1 } // 10% dikhna chahiye start karne ke liye
  );
  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animFrameId) gameLoop(); });

  // --- Resize handling ---
  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas();
  });
  resizeObserver.observe(container);

  // --- Init ---
  resizeCanvas();
  // random pattern se shuru kar — blank canvas boring lagta hai
  randomFill(0.3);
  running = true;
  playBtn.textContent = 'Pause';
  playBtn.style.borderColor = 'rgba(167,139,250,0.5)';
  playBtn.style.color = ACCENT;

  // pehla frame draw kar — taaki blank na dikhe
  draw();
  updateStats();
}
