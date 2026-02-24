// ============================================================
// Ant Colony Optimization — Pheromone-based pathfinding simulation
// Cheentiyan pheromone chodti hain, shortest path emerge hota hai
// Marco Dorigo (1992) ka classic algorithm — nature se inspired
// ============================================================

// yahi entry point hai — container dhundho, canvas banao, colony shuru karo
export function initAntColony() {
  const container = document.getElementById('antColonyContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const ACCENT_RGB = '74,158,255';
  const FONT = "'JetBrains Mono', monospace";

  // grid dimensions — ~80x50 cells ka grid
  const GRID_COLS = 80;
  const GRID_ROWS = 50;

  // pheromone constants
  const MAX_PHEROMONE = 500; // overflow prevent karne ke liye cap
  const MIN_PHEROMONE = 0.01; // zero se thoda upar — numerical stability

  // ant movement — 8 directions (dx, dy pairs)
  const DIRS = [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],           [0, 1],
    [1, -1],  [1, 0],  [1, 1],
  ];

  // --- State ---
  let canvasW = 0, canvasH = 0, dpr = 1;
  let cellW = 0, cellH = 0;

  // grid arrays — obstacles aur pheromone
  let obstacles = []; // 2D boolean — true = wall
  let searchPheromone = []; // 2D float — ants search karte waqt chodti hain
  let returnPheromone = []; // 2D float — food le ke wapas aate waqt chodti hain

  // colony entities
  let nestPos = { r: 25, c: 10 }; // cheentiyon ka ghar — left side
  let foodSources = []; // [{r, c, food}] — max 3 food sources
  let ants = []; // [{r, c, hasFood, path, pathIdx, targetFood}]

  // simulation parameters — sliders se control hote hain
  let antCount = 200;
  let evaporationRate = 0.02;
  let pheromoneStrength = 5;
  let simSpeed = 1;
  let alpha = 2; // pheromone importance
  let beta = 1; // heuristic importance
  let explorationChance = 0.05; // random exploration probability

  // interaction state
  let interactionMode = 'wall'; // 'wall' ya 'food'
  let isDrawing = false;
  let isErasing = false; // right-click se erase

  // stats
  let foodCollected = 0;
  let antsCarrying = 0;
  let shortestPathLen = Infinity;
  let bestPath = []; // sabse chhota path — highlight ke liye

  // animation state
  let animationId = null;
  let isVisible = false;

  // --- DOM structure banao — pehle sab saaf karo ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — yahan sab draw hoga
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:#0a0a12',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Stats bar ---
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'margin-top:8px',
    'padding:8px 12px',
    'background:rgba(' + ACCENT_RGB + ',0.05)',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.12)',
    'border-radius:6px',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#b0b0b0',
    'display:flex',
    'flex-wrap:wrap',
    'gap:14px',
    'align-items:center',
  ].join(';');
  container.appendChild(statsDiv);

  // --- Controls row 1: sliders ---
  const controlsDiv1 = document.createElement('div');
  controlsDiv1.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:8px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv1);

  // --- Controls row 2: buttons ---
  const controlsDiv2 = document.createElement('div');
  controlsDiv2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:6px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv2);

  // ============================================================
  // UI FACTORIES — slider, button, select banane ke helpers
  // ============================================================

  // slider factory — label + range + value display
  function createSlider(parent, label, min, max, step, value, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:' + FONT + ';white-space:nowrap;';
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.style.cssText = 'width:70px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
    wrapper.appendChild(slider);

    const val = document.createElement('span');
    val.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:' + FONT + ';min-width:30px;';
    val.textContent = formatSliderVal(value, step);
    wrapper.appendChild(val);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      val.textContent = formatSliderVal(v, step);
      onChange(v);
    });

    parent.appendChild(wrapper);
    return { wrapper, slider, val };
  }

  // slider value ko sahi format mein dikhao
  function formatSliderVal(v, step) {
    if (step < 0.01) return v.toFixed(3);
    if (step < 1) return v.toFixed(2);
    return String(Math.round(v));
  }

  // button factory — consistent dark theme styling
  function createButton(parent, text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:11px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.08)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:' + FONT,
      'transition:all 0.15s ease',
      'outline:none',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.2)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.08)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    parent.appendChild(btn);
    return btn;
  }

  // ============================================================
  // CONTROLS — sliders aur buttons banao
  // ============================================================

  // row 1: sliders
  const antSlider = createSlider(controlsDiv1, 'Ants:', 50, 500, 10, antCount, v => {
    antCount = Math.round(v);
    adjustAntCount();
  });

  const evapSlider = createSlider(controlsDiv1, 'Evap:', 0.005, 0.05, 0.005, evaporationRate, v => {
    evaporationRate = v;
  });

  const strengthSlider = createSlider(controlsDiv1, 'Pher:', 1, 10, 1, pheromoneStrength, v => {
    pheromoneStrength = Math.round(v);
  });

  const speedSlider = createSlider(controlsDiv1, 'Speed:', 1, 5, 1, simSpeed, v => {
    simSpeed = Math.round(v);
  });

  // row 2: mode toggle + buttons

  // mode toggle — Paint Walls / Place Food
  const modeBtn = createButton(controlsDiv2, 'Mode: Paint Walls', () => {
    if (interactionMode === 'wall') {
      interactionMode = 'food';
      modeBtn.textContent = 'Mode: Place Food';
      modeBtn.style.borderColor = '#ef4444';
      modeBtn.style.color = '#ef6060';
    } else {
      interactionMode = 'wall';
      modeBtn.textContent = 'Mode: Paint Walls';
      modeBtn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
      modeBtn.style.color = '#b0b0b0';
    }
  });

  // clear pheromones — sirf trails saaf, obstacles rahe
  createButton(controlsDiv2, 'Clear Pheromones', () => {
    clearPheromones();
  });

  // clear all — full reset
  createButton(controlsDiv2, 'Clear All', () => {
    resetEverything();
  });

  // preset: Maze
  createButton(controlsDiv2, 'Preset: Maze', () => {
    applyMazePreset();
  });

  // preset: Obstacles
  createButton(controlsDiv2, 'Preset: Obstacles', () => {
    applyObstaclesPreset();
  });

  // hint text — user ko batao kya kya kar sakta hai
  const hintSpan = document.createElement('span');
  hintSpan.textContent = 'Click: walls \u00B7 Shift+Click: food \u00B7 Right-click: erase';
  hintSpan.style.cssText = 'color:rgba(148,163,184,0.4);font-size:10px;font-family:' + FONT + ';margin-left:auto;';
  controlsDiv2.appendChild(hintSpan);

  // ============================================================
  // CANVAS SIZING — DPR-aware responsive canvas
  // ============================================================

  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    canvasW = w;
    canvasH = CANVAS_HEIGHT;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(CANVAS_HEIGHT * dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // cell size calculate karo — grid ko canvas mein fit karo
    cellW = canvasW / GRID_COLS;
    cellH = canvasH / GRID_ROWS;
  }

  // ============================================================
  // GRID INITIALIZATION — obstacles, pheromone arrays banao
  // ============================================================

  function initGridArrays() {
    obstacles = [];
    searchPheromone = [];
    returnPheromone = [];
    for (let r = 0; r < GRID_ROWS; r++) {
      obstacles[r] = [];
      searchPheromone[r] = [];
      returnPheromone[r] = [];
      for (let c = 0; c < GRID_COLS; c++) {
        obstacles[r][c] = false;
        searchPheromone[r][c] = 0;
        returnPheromone[r][c] = 0;
      }
    }
  }

  function clearPheromones() {
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        searchPheromone[r][c] = 0;
        returnPheromone[r][c] = 0;
      }
    }
    bestPath = [];
    shortestPathLen = Infinity;
  }

  function resetEverything() {
    initGridArrays();
    foodSources = [];
    bestPath = [];
    shortestPathLen = Infinity;
    foodCollected = 0;
    antsCarrying = 0;

    // default scenario — nest left, food right, kuch obstacles beech mein
    nestPos = { r: 25, c: 10 };
    foodSources.push({ r: 25, c: 70, food: 500 });

    // kuch obstacles beech mein daalo — interesting starting scenario
    addDefaultObstacles();

    // ants ko re-initialize karo
    initAnts();
  }

  // default obstacles — simple barrier beech mein
  function addDefaultObstacles() {
    // vertical wall beech mein — gap ke saath
    const wallC = 40;
    for (let r = 5; r < 45; r++) {
      // gap chhodho nest aur food ke level pe
      if (r >= 20 && r <= 30) continue;
      obstacles[r][wallC] = true;
      obstacles[r][wallC + 1] = true;
    }

    // thode scattered blocks
    for (let i = 0; i < 15; i++) {
      const r = 5 + Math.floor(Math.random() * 40);
      const c = 20 + Math.floor(Math.random() * 40);
      // nest aur food ke aas paas mat daalo
      if (isNearNestOrFood(r, c)) continue;
      // 2x2 block banao
      for (let dr = 0; dr < 2; dr++) {
        for (let dc = 0; dc < 2; dc++) {
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < GRID_ROWS && nc >= 0 && nc < GRID_COLS) {
            obstacles[nr][nc] = true;
          }
        }
      }
    }
  }

  // check karo ki ye position nest ya food ke paas toh nahi
  function isNearNestOrFood(r, c) {
    const nd = Math.abs(r - nestPos.r) + Math.abs(c - nestPos.c);
    if (nd < 5) return true;
    for (const f of foodSources) {
      const fd = Math.abs(r - f.r) + Math.abs(c - f.c);
      if (fd < 5) return true;
    }
    return false;
  }

  // ============================================================
  // ANT INITIALIZATION & MANAGEMENT
  // ============================================================

  function initAnts() {
    ants = [];
    for (let i = 0; i < antCount; i++) {
      ants.push(createAnt());
    }
  }

  function createAnt() {
    return {
      r: nestPos.r,
      c: nestPos.c,
      hasFood: false,
      // path track karo — wapas aane ke liye aur pheromone deposit ke liye
      path: [{ r: nestPos.r, c: nestPos.c }],
      pathIdx: 0, // return path mein current position
      targetFood: null, // kis food source ki taraf ja raha hai
      stuckCounter: 0, // agar ant stuck ho jaaye
    };
  }

  function adjustAntCount() {
    // agar zyada chahiye toh naye banao
    while (ants.length < antCount) {
      ants.push(createAnt());
    }
    // agar kam chahiye toh hata do (peeche se)
    while (ants.length > antCount) {
      ants.pop();
    }
  }

  // ============================================================
  // ANT MOVEMENT LOGIC — ACO ka core
  // ============================================================

  // ant ka ek step — searching ya returning
  function moveAnt(ant) {
    if (ant.hasFood) {
      // wapas ja — nest ki taraf
      moveReturningAnt(ant);
    } else {
      // food dhundh — pheromone follow kar
      moveSearchingAnt(ant);
    }
  }

  // searching ant — pheromone + heuristic ke basis pe move karo
  function moveSearchingAnt(ant) {
    const { r, c } = ant;

    // sabse pehle check — kya food mil gaya?
    for (let fi = 0; fi < foodSources.length; fi++) {
      const f = foodSources[fi];
      if (f.food <= 0) continue;
      const dist = Math.abs(r - f.r) + Math.abs(c - f.c);
      if (dist <= 1) {
        // food mila! Utha le aur wapas ja
        ant.hasFood = true;
        ant.targetFood = fi;
        f.food--;
        // path length track karo — shortest path update
        if (ant.path.length < shortestPathLen) {
          shortestPathLen = ant.path.length;
          bestPath = ant.path.slice();
        }
        return;
      }
    }

    // available neighbors dhundho
    const neighbors = getValidNeighbors(r, c);
    if (neighbors.length === 0) {
      // stuck — respawn at nest
      respawnAnt(ant);
      return;
    }

    // random exploration chance — diversity maintain karo
    if (Math.random() < explorationChance) {
      const pick = neighbors[Math.floor(Math.random() * neighbors.length)];
      moveAntTo(ant, pick.r, pick.c, false);
      return;
    }

    // ACO probability calculation — P = tau^alpha * eta^beta / sum
    let totalWeight = 0;
    const weights = [];

    for (const n of neighbors) {
      // pheromone level — return pheromone ko prefer karo (strong signal)
      const tau = Math.max(MIN_PHEROMONE, returnPheromone[n.r][n.c] * 2 + searchPheromone[n.r][n.c] * 0.5);

      // heuristic — closest food source ki taraf distance
      let eta = 0.1; // default low heuristic
      for (const f of foodSources) {
        if (f.food <= 0) continue;
        const dist = Math.sqrt((n.r - f.r) ** 2 + (n.c - f.c) ** 2);
        const h = 1 / Math.max(1, dist);
        if (h > eta) eta = h;
      }

      // weight calculate — tau^alpha * eta^beta
      const w = Math.pow(tau, alpha) * Math.pow(eta, beta);
      weights.push(w);
      totalWeight += w;
    }

    // agar sab weights zero hain toh random jao
    if (totalWeight <= 0) {
      const pick = neighbors[Math.floor(Math.random() * neighbors.length)];
      moveAntTo(ant, pick.r, pick.c, false);
      return;
    }

    // roulette wheel selection — weighted probability
    let rand = Math.random() * totalWeight;
    let chosenIdx = 0;
    for (let i = 0; i < weights.length; i++) {
      rand -= weights[i];
      if (rand <= 0) {
        chosenIdx = i;
        break;
      }
    }

    const chosen = neighbors[chosenIdx];
    moveAntTo(ant, chosen.r, chosen.c, false);

    // search pheromone chhodo — faint trail
    searchPheromone[r][c] = Math.min(MAX_PHEROMONE, searchPheromone[r][c] + 0.1);

    // stuck detection — agar bahut der se kuch nahi mila
    ant.stuckCounter++;
    if (ant.stuckCounter > 500) {
      respawnAnt(ant);
    }
  }

  // returning ant — nest ki taraf jao, strong pheromone chhodo
  function moveReturningAnt(ant) {
    const { r, c } = ant;

    // kya nest pe pahunch gaya?
    const nestDist = Math.abs(r - nestPos.r) + Math.abs(c - nestPos.c);
    if (nestDist <= 1) {
      // nest pe aa gaya! Food deliver karo
      foodCollected++;

      // return path pe strong pheromone deposit karo
      // pheromone amount inversely proportional to path length
      const pathLen = ant.path.length;
      const depositAmount = pheromoneStrength * (100 / Math.max(1, pathLen));
      for (const p of ant.path) {
        if (p.r >= 0 && p.r < GRID_ROWS && p.c >= 0 && p.c < GRID_COLS) {
          returnPheromone[p.r][p.c] = Math.min(MAX_PHEROMONE, returnPheromone[p.r][p.c] + depositAmount);
        }
      }

      // reset ant — naya search shuru
      ant.hasFood = false;
      ant.r = nestPos.r;
      ant.c = nestPos.c;
      ant.path = [{ r: nestPos.r, c: nestPos.c }];
      ant.pathIdx = 0;
      ant.targetFood = null;
      ant.stuckCounter = 0;
      return;
    }

    // nest ki taraf move — greedy + slight randomness
    const neighbors = getValidNeighbors(r, c);
    if (neighbors.length === 0) {
      respawnAnt(ant);
      return;
    }

    // nest ki taraf sabse paas wala neighbor chuno
    let bestDist = Infinity;
    let bestNeighbor = neighbors[0];
    for (const n of neighbors) {
      const d = Math.sqrt((n.r - nestPos.r) ** 2 + (n.c - nestPos.c) ** 2);
      if (d < bestDist) {
        bestDist = d;
        bestNeighbor = n;
      }
    }

    // thoda randomness — 10% chance random neighbor
    if (Math.random() < 0.1) {
      bestNeighbor = neighbors[Math.floor(Math.random() * neighbors.length)];
    }

    moveAntTo(ant, bestNeighbor.r, bestNeighbor.c, true);

    // return pheromone chhodo current cell pe
    const depositAmount = pheromoneStrength * 0.3;
    returnPheromone[r][c] = Math.min(MAX_PHEROMONE, returnPheromone[r][c] + depositAmount);
  }

  // ant ko naye position pe le jao
  function moveAntTo(ant, nr, nc, isReturning) {
    ant.r = nr;
    ant.c = nc;
    if (!isReturning) {
      // path mein add karo — search path track
      ant.path.push({ r: nr, c: nc });
      // path bohut lamba na ho jaaye — memory bachao
      if (ant.path.length > 1000) {
        // trim karo — sirf last 500 positions rakho
        ant.path = ant.path.slice(-500);
      }
    }
  }

  // valid neighbors — obstacles aur bounds check
  function getValidNeighbors(r, c) {
    const result = [];
    for (const [dr, dc] of DIRS) {
      const nr = r + dr;
      const nc = c + dc;
      if (nr >= 0 && nr < GRID_ROWS && nc >= 0 && nc < GRID_COLS && !obstacles[nr][nc]) {
        result.push({ r: nr, c: nc });
      }
    }
    return result;
  }

  // ant stuck hai — nest pe wapas bhejo
  function respawnAnt(ant) {
    ant.r = nestPos.r;
    ant.c = nestPos.c;
    ant.hasFood = false;
    ant.path = [{ r: nestPos.r, c: nestPos.c }];
    ant.pathIdx = 0;
    ant.targetFood = null;
    ant.stuckCounter = 0;
  }

  // ============================================================
  // PHEROMONE DYNAMICS — evaporation
  // ============================================================

  function evaporatePheromones() {
    const decay = 1 - evaporationRate;
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        searchPheromone[r][c] *= decay;
        returnPheromone[r][c] *= decay;
        // bohut kam ho gaya toh zero kar do
        if (searchPheromone[r][c] < MIN_PHEROMONE) searchPheromone[r][c] = 0;
        if (returnPheromone[r][c] < MIN_PHEROMONE) returnPheromone[r][c] = 0;
      }
    }
  }

  // ============================================================
  // SIMULATION UPDATE — ek frame ka pura update
  // ============================================================

  function updateSimulation() {
    // speed ke hisaab se multiple steps per frame
    for (let s = 0; s < simSpeed; s++) {
      // har ant ko move karo
      antsCarrying = 0;
      for (const ant of ants) {
        moveAnt(ant);
        if (ant.hasFood) antsCarrying++;
      }

      // pheromone evaporate karo
      evaporatePheromones();
    }
  }

  // ============================================================
  // RENDERING — canvas pe sab draw karo
  // ============================================================

  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // background — dark
    ctx.fillStyle = '#0a0a12';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // --- Pheromone heatmap render karo ---
    // offscreen imageData use karenge — fast pixel manipulation
    const imgData = ctx.createImageData(Math.round(canvasW), Math.round(canvasH));
    const pixels = imgData.data;
    const imgW = imgData.width;
    const imgH = imgData.height;

    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        // pixel range calculate karo is cell ke liye
        const x0 = Math.floor(c * cellW);
        const y0 = Math.floor(r * cellH);
        const x1 = Math.floor((c + 1) * cellW);
        const y1 = Math.floor((r + 1) * cellH);

        // cell ka color decide karo
        let cr = 10, cg = 10, cb = 18, ca = 255; // default dark background

        if (obstacles[r][c]) {
          // obstacle — dark gray
          cr = 45; cg = 45; cb = 55;
        } else {
          // pheromone levels
          const sp = searchPheromone[r][c];
          const rp = returnPheromone[r][c];

          if (rp > 0.1 || sp > 0.1) {
            // pheromone heatmap — search = faint blue, return = bright green/cyan
            const searchIntensity = Math.min(1, sp / 10);
            const returnIntensity = Math.min(1, rp / 20);

            // combine — return pheromone dominant color
            if (returnIntensity > searchIntensity) {
              // return pheromone: dark purple -> cyan -> bright yellow-green
              const t = returnIntensity;
              if (t < 0.3) {
                // dark blue/purple
                cr = Math.round(20 + t * 60);
                cg = Math.round(15 + t * 80);
                cb = Math.round(60 + t * 150);
              } else if (t < 0.7) {
                // cyan/teal
                const t2 = (t - 0.3) / 0.4;
                cr = Math.round(38 + t2 * 20);
                cg = Math.round(39 + t2 * 180);
                cb = Math.round(105 + t2 * 50);
              } else {
                // bright green/yellow
                const t2 = (t - 0.7) / 0.3;
                cr = Math.round(58 + t2 * 170);
                cg = Math.round(219 - t2 * 20);
                cb = Math.round(155 - t2 * 100);
              }
            } else {
              // search pheromone: faint blue/purple
              const t = searchIntensity;
              cr = Math.round(15 + t * 40);
              cg = Math.round(12 + t * 25);
              cb = Math.round(40 + t * 120);
            }
          }
        }

        // pixels fill karo — cell ke area mein
        for (let py = y0; py < y1 && py < imgH; py++) {
          for (let px = x0; px < x1 && px < imgW; px++) {
            const idx = (py * imgW + px) * 4;
            pixels[idx] = cr;
            pixels[idx + 1] = cg;
            pixels[idx + 2] = cb;
            pixels[idx + 3] = ca;
          }
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);

    // --- Best path highlight — bright trail ---
    if (bestPath.length > 2) {
      ctx.strokeStyle = 'rgba(250,200,50,0.3)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(bestPath[0].c * cellW + cellW / 2, bestPath[0].r * cellH + cellH / 2);
      for (let i = 1; i < bestPath.length; i++) {
        ctx.lineTo(bestPath[i].c * cellW + cellW / 2, bestPath[i].r * cellH + cellH / 2);
      }
      ctx.stroke();
    }

    // --- Nest draw — green glow ---
    const nestX = nestPos.c * cellW + cellW / 2;
    const nestY = nestPos.r * cellH + cellH / 2;
    const nestRadius = Math.min(cellW, cellH) * 2.5;

    // glow effect
    const nestGlow = ctx.createRadialGradient(nestX, nestY, 0, nestX, nestY, nestRadius);
    nestGlow.addColorStop(0, 'rgba(16,185,129,0.5)');
    nestGlow.addColorStop(0.5, 'rgba(16,185,129,0.15)');
    nestGlow.addColorStop(1, 'rgba(16,185,129,0)');
    ctx.fillStyle = nestGlow;
    ctx.fillRect(nestX - nestRadius, nestY - nestRadius, nestRadius * 2, nestRadius * 2);

    // nest circle
    ctx.beginPath();
    ctx.arc(nestX, nestY, nestRadius * 0.4, 0, Math.PI * 2);
    ctx.fillStyle = '#10b981';
    ctx.fill();
    ctx.strokeStyle = 'rgba(16,185,129,0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // nest label
    ctx.font = '9px ' + FONT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#fff';
    ctx.fillText('NEST', nestX, nestY);

    // --- Food sources draw — red glow ---
    for (const f of foodSources) {
      const fx = f.c * cellW + cellW / 2;
      const fy = f.r * cellH + cellH / 2;
      const fRadius = Math.min(cellW, cellH) * 2;

      // glow
      const foodGlow = ctx.createRadialGradient(fx, fy, 0, fx, fy, fRadius);
      foodGlow.addColorStop(0, 'rgba(239,68,68,0.5)');
      foodGlow.addColorStop(0.5, 'rgba(239,68,68,0.15)');
      foodGlow.addColorStop(1, 'rgba(239,68,68,0)');
      ctx.fillStyle = foodGlow;
      ctx.fillRect(fx - fRadius, fy - fRadius, fRadius * 2, fRadius * 2);

      // food circle
      ctx.beginPath();
      ctx.arc(fx, fy, fRadius * 0.4, 0, Math.PI * 2);
      ctx.fillStyle = f.food > 0 ? '#ef4444' : '#6b2121';
      ctx.fill();
      ctx.strokeStyle = f.food > 0 ? 'rgba(239,68,68,0.6)' : 'rgba(107,33,33,0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // food count
      ctx.font = '8px ' + FONT;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#fff';
      ctx.fillText(String(f.food), fx, fy);
    }

    // --- Ants draw — tiny dots ---
    for (const ant of ants) {
      const ax = ant.c * cellW + cellW / 2;
      const ay = ant.r * cellH + cellH / 2;
      const antSize = Math.max(1, Math.min(cellW, cellH) * 0.35);

      ctx.beginPath();
      ctx.arc(ax, ay, antSize, 0, Math.PI * 2);

      if (ant.hasFood) {
        // food le ke ja raha — orange/golden dot
        ctx.fillStyle = '#f59e0b';
      } else {
        // searching — white/light gray dot
        ctx.fillStyle = 'rgba(220,220,230,0.8)';
      }
      ctx.fill();
    }

    // stats update
    updateStats();
  }

  // ============================================================
  // STATS UPDATE
  // ============================================================

  function updateStats() {
    // purane children hata do
    while (statsDiv.firstChild) statsDiv.removeChild(statsDiv.firstChild);

    const items = [
      { label: 'Ants', value: String(ants.length), color: ACCENT },
      { label: 'Carrying', value: String(antsCarrying), color: '#f59e0b' },
      { label: 'Collected', value: String(foodCollected), color: '#10b981' },
      { label: 'Best Path', value: shortestPathLen < Infinity ? String(shortestPathLen) : '-', color: '#fbbf24' },
    ];

    for (const item of items) {
      const span = document.createElement('span');
      span.style.cssText = 'white-space:nowrap;';

      const labelNode = document.createTextNode(item.label + ': ');
      span.appendChild(labelNode);

      const valSpan = document.createElement('span');
      valSpan.style.cssText = 'color:' + item.color + ';font-weight:600;';
      valSpan.textContent = item.value;
      span.appendChild(valSpan);

      statsDiv.appendChild(span);
    }
  }

  // ============================================================
  // PRESETS — maze aur obstacles
  // ============================================================

  function applyMazePreset() {
    // sab clear karo
    initGridArrays();
    foodSources = [];
    bestPath = [];
    shortestPathLen = Infinity;
    foodCollected = 0;

    // nest left, food right
    nestPos = { r: 25, c: 5 };
    foodSources.push({ r: 25, c: 74, food: 500 });

    // simple maze — vertical walls with gaps
    const walls = [
      { c: 15, gapStart: 35, gapEnd: 42 },
      { c: 30, gapStart: 8, gapEnd: 15 },
      { c: 45, gapStart: 30, gapEnd: 38 },
      { c: 60, gapStart: 12, gapEnd: 20 },
    ];

    for (const wall of walls) {
      for (let r = 2; r < GRID_ROWS - 2; r++) {
        if (r >= wall.gapStart && r <= wall.gapEnd) continue;
        obstacles[r][wall.c] = true;
        obstacles[r][wall.c + 1] = true;
      }
    }

    // horizontal connectors — maze ko interesting banao
    for (let c = 15; c < 30; c++) {
      obstacles[8][c] = true;
      obstacles[42][c] = true;
    }
    for (let c = 30; c < 45; c++) {
      obstacles[15][c] = true;
      obstacles[38][c] = true;
    }
    for (let c = 45; c < 60; c++) {
      obstacles[20][c] = true;
      obstacles[30][c] = true;
    }

    initAnts();
  }

  function applyObstaclesPreset() {
    // sab clear karo
    initGridArrays();
    foodSources = [];
    bestPath = [];
    shortestPathLen = Infinity;
    foodCollected = 0;

    // nest left, 2 food sources
    nestPos = { r: 25, c: 8 };
    foodSources.push({ r: 10, c: 65, food: 400 });
    foodSources.push({ r: 40, c: 60, food: 400 });

    // scattered rectangular blocks
    const blocks = [
      { r: 8, c: 25, w: 8, h: 12 },
      { r: 30, c: 20, w: 6, h: 10 },
      { r: 15, c: 40, w: 10, h: 5 },
      { r: 35, c: 42, w: 5, h: 8 },
      { r: 5, c: 50, w: 7, h: 6 },
      { r: 38, c: 30, w: 8, h: 4 },
      { r: 20, c: 55, w: 4, h: 12 },
      { r: 10, c: 32, w: 3, h: 8 },
    ];

    for (const block of blocks) {
      for (let r = block.r; r < block.r + block.h && r < GRID_ROWS; r++) {
        for (let c = block.c; c < block.c + block.w && c < GRID_COLS; c++) {
          if (r >= 0 && r < GRID_ROWS && c >= 0 && c < GRID_COLS) {
            obstacles[r][c] = true;
          }
        }
      }
    }

    initAnts();
  }

  // ============================================================
  // MOUSE/TOUCH INTERACTION
  // ============================================================

  // pixel position se grid cell nikalo
  function getCellFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const c = Math.floor(x / cellW);
    const r = Math.floor(y / cellH);
    if (r >= 0 && r < GRID_ROWS && c >= 0 && c < GRID_COLS) return { r, c };
    return null;
  }

  // obstacle paint karo — brush size 2x2
  function paintObstacle(r, c) {
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nr < GRID_ROWS && nc >= 0 && nc < GRID_COLS) {
          // nest aur food ke upar paint mat karo
          if (nr === nestPos.r && nc === nestPos.c) continue;
          let onFood = false;
          for (const f of foodSources) {
            if (nr === f.r && nc === f.c) { onFood = true; break; }
          }
          if (onFood) continue;
          obstacles[nr][nc] = true;
        }
      }
    }
  }

  // obstacle erase karo
  function eraseObstacle(r, c) {
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nr < GRID_ROWS && nc >= 0 && nc < GRID_COLS) {
          obstacles[nr][nc] = false;
        }
      }
    }
  }

  // food source place karo — max 3
  function placeFood(r, c) {
    if (foodSources.length >= 3) return; // max 3
    // nest ke upar mat rakho
    if (Math.abs(r - nestPos.r) + Math.abs(c - nestPos.c) < 3) return;
    // existing food ke upar mat rakho
    for (const f of foodSources) {
      if (Math.abs(r - f.r) + Math.abs(c - f.c) < 3) return;
    }
    // obstacle hata do wahan se
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nr < GRID_ROWS && nc >= 0 && nc < GRID_COLS) {
          obstacles[nr][nc] = false;
        }
      }
    }
    foodSources.push({ r, c, food: 500 });
  }

  // --- Mouse events ---
  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const cell = getCellFromEvent(e);
    if (!cell) return;

    // right click — erase mode
    if (e.button === 2) {
      isErasing = true;
      isDrawing = false;
      eraseObstacle(cell.r, cell.c);
      return;
    }

    // shift + click — food place karo (mode se independent)
    if (e.shiftKey) {
      placeFood(cell.r, cell.c);
      return;
    }

    // left click — mode ke hisaab se
    if (interactionMode === 'food') {
      placeFood(cell.r, cell.c);
    } else {
      isDrawing = true;
      isErasing = false;
      paintObstacle(cell.r, cell.c);
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    const cell = getCellFromEvent(e);
    if (!cell) return;

    if (isDrawing) {
      paintObstacle(cell.r, cell.c);
    } else if (isErasing) {
      eraseObstacle(cell.r, cell.c);
    }
  });

  canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    isErasing = false;
  });

  canvas.addEventListener('mouseleave', () => {
    isDrawing = false;
    isErasing = false;
  });

  // right-click menu band karo canvas pe
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  // --- Touch events ---
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const cell = getCellFromEvent(e);
    if (!cell) return;

    if (interactionMode === 'food') {
      placeFood(cell.r, cell.c);
    } else {
      isDrawing = true;
      paintObstacle(cell.r, cell.c);
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const cell = getCellFromEvent(e);
    if (!cell) return;
    paintObstacle(cell.r, cell.c);
  }, { passive: false });

  canvas.addEventListener('touchend', () => {
    isDrawing = false;
  });

  // ============================================================
  // ANIMATION LOOP — sirf jab visible ho tab animate karo
  // ============================================================

  function loop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) { animationId = null; return; }

    updateSimulation();
    draw();

    animationId = requestAnimationFrame(loop);
  }

  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    if (!animationId) animationId = requestAnimationFrame(loop);
  }

  function stopAnimation() {
    isVisible = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // --- IntersectionObserver — sirf visible hone pe animate karo ---
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          startAnimation();
        } else {
          stopAnimation();
        }
      });
    },
    { threshold: 0.1 }
  );

  observer.observe(container);

  // lab resume — jab pause hata toh wapas shuru karo
  document.addEventListener('lab:resume', () => {
    if (isVisible && !animationId) loop();
  });

  // tab visibility — battery bachao jab tab hidden ho
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // resize handler
  window.addEventListener('resize', () => {
    resizeCanvas();
  });

  // ============================================================
  // INITIALIZATION — sab set karo, colony shuru karo!
  // ============================================================

  resizeCanvas();
  initGridArrays();
  resetEverything();
}
