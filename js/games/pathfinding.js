// ============================================================
// Pathfinding Algorithm Visualizer — A*, Dijkstra, BFS, DFS, Greedy BFS
// Grid pe walls draw karo, start/end drag karo, algorithms ko explore hote dekho
// Generator pattern se har step yield hota hai — paint jaisa failta hai
// ============================================================

// yahi entry point hai — container pakdo, canvas banao, pathfinding shuru karo
export function initPathfinding() {
  const container = document.getElementById('pathfindingContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const FONT = "'JetBrains Mono', monospace";

  // cell types
  const EMPTY = 0;
  const WALL = 1;
  const START = 2;
  const END = 3;

  // colors — dark theme, satisfying exploration dikhne chahiye
  const COL = {
    empty:    '#0a0a0a',
    wall:     '#404050',
    start:    '#10b981',
    end:      '#ef4444',
    gridLine: '#1a1a2e',
    frontier: '#22d3ee',
    path:     '#f59e0b',
    pathGlow: '#fbbf24',
  };

  // visited color gradient — dark purple to blue based on visit order
  const VISITED_START = { r: 26, g: 26, b: 78 };   // #1a1a4e
  const VISITED_END   = { r: 45, g: 27, b: 105 };   // #2d1b69

  // algorithms metadata
  const ALGORITHMS = {
    astar:      { name: 'A*',               fn: astarGenerator },
    dijkstra:   { name: 'Dijkstra',         fn: dijkstraGenerator },
    bfs:        { name: 'BFS',              fn: bfsGenerator },
    dfs:        { name: 'DFS',              fn: dfsGenerator },
    greedy:     { name: 'Greedy Best-First', fn: greedyGenerator },
  };

  // --- State ---
  let cols = 40, rows = 0;
  let grid = [];
  let startPos = { r: 0, c: 0 };
  let endPos = { r: 0, c: 0 };
  let cellSize = 10;

  // drawing state
  let isDragging = false;
  let dragMode = null; // 'wall-draw', 'wall-erase', 'start', 'end'

  // algorithm state
  let currentAlgo = 'astar';
  let running = false;
  let generator = null;
  let stepsPerFrame = 5;
  let allowDiagonal = false;

  // visualization data
  let visitedCells = new Map();  // key -> visit order (0-based)
  let frontierCells = new Set(); // currently in open set
  let pathCells = new Set();     // final path cells
  let pathList = [];             // final path in order (for glow animation)
  let maxVisitOrder = 0;

  // stats
  let nodesExplored = 0;
  let pathLength = 0;
  let startTime = 0;
  let elapsedMs = 0;

  // golden glow animation — path milne ke baad sweep karta hai
  let glowPhase = -1; // -1 = inactive, 0+ = animating
  let glowSpeed = 0.03;

  // visibility + animation
  let isVisible = false;
  let animationId = null;

  // --- DOM structure banao ---
  const existingChildren = Array.from(container.children);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'border-radius:8px',
    'cursor:crosshair',
    'display:block',
    'touch-action:none',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Stats bar ---
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'justify-content:center',
    'flex-wrap:wrap',
    'gap:16px',
    'padding:6px 10px',
    'margin-top:6px',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#94a3b8',
    'background:rgba(167,139,250,0.06)',
    'border:1px solid rgba(167,139,250,0.12)',
    'border-radius:6px',
    'min-height:22px',
  ].join(';');
  container.appendChild(statsDiv);

  // --- Controls row 1: algorithm dropdown + action buttons ---
  const controlsDiv1 = document.createElement('div');
  controlsDiv1.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:6px',
    'margin-top:8px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv1);

  // algorithm dropdown
  const algoSelect = document.createElement('select');
  algoSelect.style.cssText = makeSelectCSS();
  for (const key of Object.keys(ALGORITHMS)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = ALGORITHMS[key].name;
    if (key === currentAlgo) opt.selected = true;
    algoSelect.appendChild(opt);
  }
  algoSelect.addEventListener('change', () => {
    currentAlgo = algoSelect.value;
  });
  controlsDiv1.appendChild(algoSelect);

  // start button
  const startBtn = document.createElement('button');
  startBtn.textContent = '\u25B6 Start';
  startBtn.style.cssText = makeActionBtnCSS();
  startBtn.addEventListener('click', () => {
    if (running) {
      stopAlgorithm();
    } else {
      runAlgorithm();
    }
  });
  controlsDiv1.appendChild(startBtn);

  // generate maze button
  const mazeBtn = document.createElement('button');
  mazeBtn.textContent = 'Generate Maze';
  mazeBtn.style.cssText = makeBtnCSS();
  mazeBtn.addEventListener('click', () => {
    if (running) stopAlgorithm();
    clearResults();
    runRecursiveDivision();
    requestDraw();
  });
  controlsDiv1.appendChild(mazeBtn);

  // clear button — walls + results dono saaf
  const clearBtn = document.createElement('button');
  clearBtn.textContent = 'Clear';
  clearBtn.style.cssText = makeBtnCSS();
  clearBtn.addEventListener('click', () => {
    if (running) stopAlgorithm();
    clearAll();
    requestDraw();
  });
  controlsDiv1.appendChild(clearBtn);

  // clear path button — sirf results saaf, walls rahe
  const clearPathBtn = document.createElement('button');
  clearPathBtn.textContent = 'Clear Path';
  clearPathBtn.style.cssText = makeBtnCSS();
  clearPathBtn.addEventListener('click', () => {
    if (running) stopAlgorithm();
    clearResults();
    requestDraw();
  });
  controlsDiv1.appendChild(clearPathBtn);

  // --- Controls row 2: speed slider + diagonal toggle ---
  const controlsDiv2 = document.createElement('div');
  controlsDiv2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:6px',
    'align-items:center',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#94a3b8',
  ].join(';');
  container.appendChild(controlsDiv2);

  // speed slider
  const speedGroup = makeSliderGroup('Speed', 1, 50, stepsPerFrame, (v) => {
    stepsPerFrame = v;
  });
  controlsDiv2.appendChild(speedGroup.wrapper);

  // diagonal toggle
  const diagBtn = document.createElement('button');
  diagBtn.textContent = 'Diagonal: OFF';
  diagBtn.style.cssText = makeBtnCSS();
  diagBtn.addEventListener('click', () => {
    allowDiagonal = !allowDiagonal;
    diagBtn.textContent = 'Diagonal: ' + (allowDiagonal ? 'ON' : 'OFF');
    diagBtn.style.borderColor = allowDiagonal ? ACCENT : 'rgba(167,139,250,0.25)';
    diagBtn.style.background = allowDiagonal ? 'rgba(167,139,250,0.2)' : 'rgba(167,139,250,0.06)';
  });
  controlsDiv2.appendChild(diagBtn);

  // instruction hint
  const hintSpan = document.createElement('span');
  hintSpan.textContent = 'Click: draw walls \u00B7 Drag green/red to move';
  hintSpan.style.cssText = 'color:rgba(148,163,184,0.5);font-size:10px;margin-left:auto;';
  controlsDiv2.appendChild(hintSpan);

  // ============================================================
  // CANVAS SIZING — responsive canvas with DPR handling
  // ============================================================

  let canvasW = 0, canvasH = 0, dpr = 1;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = container.clientWidth;
    canvasW = w;
    canvasH = CANVAS_HEIGHT;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(CANVAS_HEIGHT * dpr);
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // grid dimensions recalculate — ~40 columns, rows proportional
    cols = 40;
    cellSize = canvasW / cols;
    rows = Math.floor(canvasH / cellSize);
    if (rows < 5) rows = 5;

    rebuildGrid();
  }

  // grid rebuild — purana data preserve karne ki koshish karo
  function rebuildGrid() {
    const oldGrid = grid;
    const oldRows = oldGrid.length;
    const oldCols = oldRows > 0 ? oldGrid[0].length : 0;

    grid = [];
    for (let r = 0; r < rows; r++) {
      grid[r] = [];
      for (let c = 0; c < cols; c++) {
        if (r < oldRows && c < oldCols) {
          grid[r][c] = oldGrid[r][c];
        } else {
          grid[r][c] = EMPTY;
        }
      }
    }

    // start/end bounds check — agar grid se bahar chale gaye toh wapas lao
    startPos.r = Math.min(startPos.r, rows - 1);
    startPos.c = Math.min(startPos.c, cols - 1);
    endPos.r = Math.min(endPos.r, rows - 1);
    endPos.c = Math.min(endPos.c, cols - 1);

    // ensure start/end cells marked hain
    grid[startPos.r][startPos.c] = START;
    grid[endPos.r][endPos.c] = END;
  }

  // ============================================================
  // GRID UTILITIES
  // ============================================================

  function cellKey(r, c) {
    return r * cols + c;
  }

  function keyToRC(key) {
    return { r: Math.floor(key / cols), c: key % cols };
  }

  // neighbors — 4 or 8 connected depending on diagonal toggle
  function getNeighbors(r, c) {
    const result = [];
    // cardinal directions pehle
    const dirs4 = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    // diagonal directions
    const dirs8 = [[-1, -1], [-1, 1], [1, -1], [1, 1]];

    for (const [dr, dc] of dirs4) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] !== WALL) {
        result.push({ r: nr, c: nc, cost: 1 });
      }
    }

    if (allowDiagonal) {
      for (const [dr, dc] of dirs8) {
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] !== WALL) {
          // diagonal movement sirf tab allow karo jab dono adjacent cardinal cells wall nahi hain
          // nahi toh corner cutting ho jaayegi
          if (grid[r + dr][c] !== WALL && grid[r][c + dc] !== WALL) {
            result.push({ r: nr, c: nc, cost: Math.SQRT2 });
          }
        }
      }
    }

    return result;
  }

  // heuristic — octile distance for diagonal, manhattan for 4-connected
  function heuristic(r1, c1, r2, c2) {
    const dx = Math.abs(c1 - c2);
    const dy = Math.abs(r1 - r2);
    if (allowDiagonal) {
      // octile distance — sahi estimate for 8-connected grid
      return Math.max(dx, dy) + (Math.SQRT2 - 1) * Math.min(dx, dy);
    }
    return dx + dy; // manhattan
  }

  // path reconstruct from cameFrom map
  function reconstructPath(cameFrom, endKey) {
    const path = [];
    let current = endKey;
    while (current !== -1 && current !== undefined) {
      path.unshift(current);
      current = cameFrom.get(current);
    }
    return path;
  }

  // ============================================================
  // BINARY HEAP — priority queue for A*, Dijkstra, Greedy
  // efficient insert/extract min for big grids
  // ============================================================

  class MinHeap {
    constructor() {
      this.data = [];
      this.indices = new Map(); // key -> index in heap (for decrease-key)
    }

    get size() { return this.data.length; }

    // naya element daalo ya priority update karo
    push(key, priority) {
      if (this.indices.has(key)) {
        // already hai — priority update karo agar kam hai
        const idx = this.indices.get(key);
        if (priority < this.data[idx].p) {
          this.data[idx].p = priority;
          this._bubbleUp(idx);
        }
        return;
      }
      this.data.push({ k: key, p: priority });
      this.indices.set(key, this.data.length - 1);
      this._bubbleUp(this.data.length - 1);
    }

    // sabse chhoti priority wala nikal
    pop() {
      if (this.data.length === 0) return null;
      const top = this.data[0];
      this.indices.delete(top.k);

      const last = this.data.pop();
      if (this.data.length > 0) {
        this.data[0] = last;
        this.indices.set(last.k, 0);
        this._sinkDown(0);
      }
      return top;
    }

    has(key) { return this.indices.has(key); }

    _bubbleUp(idx) {
      while (idx > 0) {
        const parent = (idx - 1) >> 1;
        if (this.data[idx].p < this.data[parent].p) {
          this._swap(idx, parent);
          idx = parent;
        } else break;
      }
    }

    _sinkDown(idx) {
      const n = this.data.length;
      while (true) {
        let smallest = idx;
        const left = 2 * idx + 1;
        const right = 2 * idx + 2;
        if (left < n && this.data[left].p < this.data[smallest].p) smallest = left;
        if (right < n && this.data[right].p < this.data[smallest].p) smallest = right;
        if (smallest !== idx) {
          this._swap(idx, smallest);
          idx = smallest;
        } else break;
      }
    }

    _swap(i, j) {
      [this.data[i], this.data[j]] = [this.data[j], this.data[i]];
      this.indices.set(this.data[i].k, i);
      this.indices.set(this.data[j].k, j);
    }
  }

  // ============================================================
  // PATHFINDING ALGORITHMS — Generator functions
  // har yield ek step hai — frontier expand hota hai paint jaisa
  // ============================================================

  // --- A* --- f(n) = g(n) + h(n), optimal + informed
  function* astarGenerator() {
    const sk = cellKey(startPos.r, startPos.c);
    const ek = cellKey(endPos.r, endPos.c);

    const openSet = new MinHeap();
    const gScore = new Map();
    const cameFrom = new Map();
    const closedSet = new Set();

    gScore.set(sk, 0);
    cameFrom.set(sk, -1);
    openSet.push(sk, heuristic(startPos.r, startPos.c, endPos.r, endPos.c));

    // frontier track karo visualization ke liye
    frontierCells.clear();
    frontierCells.add(sk);

    while (openSet.size > 0) {
      const current = openSet.pop();
      const ck = current.k;
      frontierCells.delete(ck);

      if (closedSet.has(ck)) continue;
      closedSet.add(ck);

      // visited mark karo
      visitedCells.set(ck, maxVisitOrder++);
      nodesExplored++;

      yield; // ek step ruk — visualization dikhao

      // goal mila!
      if (ck === ek) {
        const path = reconstructPath(cameFrom, ek);
        for (const pk of path) pathCells.add(pk);
        pathList = path;
        pathLength = path.length;
        frontierCells.clear();
        return;
      }

      const { r, c } = keyToRC(ck);
      const neighbors = getNeighbors(r, c);

      for (const n of neighbors) {
        const nk = cellKey(n.r, n.c);
        if (closedSet.has(nk)) continue;

        const tentG = gScore.get(ck) + n.cost;
        if (!gScore.has(nk) || tentG < gScore.get(nk)) {
          gScore.set(nk, tentG);
          cameFrom.set(nk, ck);
          const f = tentG + heuristic(n.r, n.c, endPos.r, endPos.c);
          openSet.push(nk, f);
          frontierCells.add(nk);
        }
      }
    }

    // path nahi mila — goal unreachable
    frontierCells.clear();
  }

  // --- Dijkstra --- A* without heuristic, h=0, explores uniformly
  function* dijkstraGenerator() {
    const sk = cellKey(startPos.r, startPos.c);
    const ek = cellKey(endPos.r, endPos.c);

    const openSet = new MinHeap();
    const gScore = new Map();
    const cameFrom = new Map();
    const closedSet = new Set();

    gScore.set(sk, 0);
    cameFrom.set(sk, -1);
    openSet.push(sk, 0);

    frontierCells.clear();
    frontierCells.add(sk);

    while (openSet.size > 0) {
      const current = openSet.pop();
      const ck = current.k;
      frontierCells.delete(ck);

      if (closedSet.has(ck)) continue;
      closedSet.add(ck);

      visitedCells.set(ck, maxVisitOrder++);
      nodesExplored++;

      yield;

      if (ck === ek) {
        const path = reconstructPath(cameFrom, ek);
        for (const pk of path) pathCells.add(pk);
        pathList = path;
        pathLength = path.length;
        frontierCells.clear();
        return;
      }

      const { r, c } = keyToRC(ck);
      const neighbors = getNeighbors(r, c);

      for (const n of neighbors) {
        const nk = cellKey(n.r, n.c);
        if (closedSet.has(nk)) continue;

        const tentG = gScore.get(ck) + n.cost;
        if (!gScore.has(nk) || tentG < gScore.get(nk)) {
          gScore.set(nk, tentG);
          cameFrom.set(nk, ck);
          // Dijkstra — f = g, no heuristic
          openSet.push(nk, tentG);
          frontierCells.add(nk);
        }
      }
    }

    frontierCells.clear();
  }

  // --- BFS --- unweighted shortest path, queue-based
  function* bfsGenerator() {
    const sk = cellKey(startPos.r, startPos.c);
    const ek = cellKey(endPos.r, endPos.c);

    const queue = [sk];
    const visited = new Set([sk]);
    const cameFrom = new Map();
    cameFrom.set(sk, -1);

    frontierCells.clear();
    frontierCells.add(sk);

    while (queue.length > 0) {
      const ck = queue.shift();
      frontierCells.delete(ck);

      visitedCells.set(ck, maxVisitOrder++);
      nodesExplored++;

      yield;

      if (ck === ek) {
        const path = reconstructPath(cameFrom, ek);
        for (const pk of path) pathCells.add(pk);
        pathList = path;
        pathLength = path.length;
        frontierCells.clear();
        return;
      }

      const { r, c } = keyToRC(ck);
      const neighbors = getNeighbors(r, c);

      for (const n of neighbors) {
        const nk = cellKey(n.r, n.c);
        if (!visited.has(nk)) {
          visited.add(nk);
          cameFrom.set(nk, ck);
          queue.push(nk);
          frontierCells.add(nk);
        }
      }
    }

    frontierCells.clear();
  }

  // --- DFS --- depth-first, NOT optimal, dikhaata hai difference
  function* dfsGenerator() {
    const sk = cellKey(startPos.r, startPos.c);
    const ek = cellKey(endPos.r, endPos.c);

    const stack = [sk];
    const visited = new Set();
    const cameFrom = new Map();
    cameFrom.set(sk, -1);

    frontierCells.clear();
    frontierCells.add(sk);

    while (stack.length > 0) {
      const ck = stack.pop();
      frontierCells.delete(ck);

      if (visited.has(ck)) continue;
      visited.add(ck);

      visitedCells.set(ck, maxVisitOrder++);
      nodesExplored++;

      yield;

      if (ck === ek) {
        const path = reconstructPath(cameFrom, ek);
        for (const pk of path) pathCells.add(pk);
        pathList = path;
        pathLength = path.length;
        frontierCells.clear();
        return;
      }

      const { r, c } = keyToRC(ck);
      const neighbors = getNeighbors(r, c);

      // reverse order — consistent exploration direction
      for (let i = neighbors.length - 1; i >= 0; i--) {
        const n = neighbors[i];
        const nk = cellKey(n.r, n.c);
        if (!visited.has(nk)) {
          cameFrom.set(nk, ck);
          stack.push(nk);
          frontierCells.add(nk);
        }
      }
    }

    frontierCells.clear();
  }

  // --- Greedy Best-First --- sirf h(n) use karta hai, fast but not optimal
  function* greedyGenerator() {
    const sk = cellKey(startPos.r, startPos.c);
    const ek = cellKey(endPos.r, endPos.c);

    const openSet = new MinHeap();
    const closedSet = new Set();
    const cameFrom = new Map();

    cameFrom.set(sk, -1);
    openSet.push(sk, heuristic(startPos.r, startPos.c, endPos.r, endPos.c));

    frontierCells.clear();
    frontierCells.add(sk);

    while (openSet.size > 0) {
      const current = openSet.pop();
      const ck = current.k;
      frontierCells.delete(ck);

      if (closedSet.has(ck)) continue;
      closedSet.add(ck);

      visitedCells.set(ck, maxVisitOrder++);
      nodesExplored++;

      yield;

      if (ck === ek) {
        const path = reconstructPath(cameFrom, ek);
        for (const pk of path) pathCells.add(pk);
        pathList = path;
        pathLength = path.length;
        frontierCells.clear();
        return;
      }

      const { r, c } = keyToRC(ck);
      const neighbors = getNeighbors(r, c);

      for (const n of neighbors) {
        const nk = cellKey(n.r, n.c);
        if (closedSet.has(nk)) continue;

        if (!openSet.has(nk) && !closedSet.has(nk)) {
          cameFrom.set(nk, ck);
          // greedy — sirf heuristic use karo, g ignore karo
          openSet.push(nk, heuristic(n.r, n.c, endPos.r, endPos.c));
          frontierCells.add(nk);
        }
      }
    }

    frontierCells.clear();
  }

  // ============================================================
  // MAZE GENERATION — Recursive Division
  // grid ko recursively divide karo walls se, passages rakh ke
  // ============================================================

  function runRecursiveDivision() {
    // pehle saara grid clear kar — sab empty
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        grid[r][c] = EMPTY;
      }
    }

    // border walls banao
    for (let r = 0; r < rows; r++) {
      grid[r][0] = WALL;
      grid[r][cols - 1] = WALL;
    }
    for (let c = 0; c < cols; c++) {
      grid[0][c] = WALL;
      grid[rows - 1][c] = WALL;
    }

    // recursive division start karo inner area pe
    divide(1, 1, cols - 2, rows - 2);

    // start aur end place karo
    startPos = { r: 1, c: 1 };
    endPos = { r: rows - 2, c: cols - 2 };
    grid[startPos.r][startPos.c] = START;
    grid[endPos.r][endPos.c] = END;
  }

  function divide(x, y, w, h) {
    // agar area bohot chhota hai toh divide mat karo
    if (w < 2 || h < 2) return;

    // horizontal ya vertical divide — wider area ko lambi direction mein kaato
    const horizontal = h > w ? true : w > h ? false : Math.random() < 0.5;

    if (horizontal) {
      // horizontal wall banao — y direction mein random position
      const wallY = y + 1 + Math.floor(Math.random() * (h - 1));
      // passage — ek random jagah gap rakhna hai
      const passageX = x + Math.floor(Math.random() * w);

      for (let cx = x; cx < x + w; cx++) {
        if (cx !== passageX) {
          grid[wallY][cx] = WALL;
        }
      }

      // recursively divide dono halves
      divide(x, y, w, wallY - y);
      divide(x, wallY + 1, w, y + h - wallY - 1);
    } else {
      // vertical wall
      const wallX = x + 1 + Math.floor(Math.random() * (w - 1));
      const passageY = y + Math.floor(Math.random() * h);

      for (let ry = y; ry < y + h; ry++) {
        if (ry !== passageY) {
          grid[ry][wallX] = WALL;
        }
      }

      divide(x, y, wallX - x, h);
      divide(wallX + 1, y, x + w - wallX - 1, h);
    }
  }

  // ============================================================
  // ALGORITHM CONTROL
  // ============================================================

  function runAlgorithm() {
    if (running) return;

    clearResults();
    running = true;
    startTime = performance.now();
    startBtn.textContent = '\u25A0 Stop';
    algoSelect.disabled = true;

    // selected algorithm ka generator bana
    const algoFn = ALGORITHMS[currentAlgo].fn;
    generator = algoFn();

    // animation loop step karna shuru
    stepAlgorithm();
  }

  function stopAlgorithm() {
    running = false;
    generator = null;
    startBtn.textContent = '\u25B6 Start';
    algoSelect.disabled = false;
    frontierCells.clear();
    requestDraw();
  }

  function stepAlgorithm() {
    if (!running || !generator) return;

    // speed ke hisaab se multiple steps ek frame mein
    for (let s = 0; s < stepsPerFrame; s++) {
      const result = generator.next();
      if (result.done) {
        // algorithm complete ho gaya
        elapsedMs = performance.now() - startTime;
        running = false;
        generator = null;
        startBtn.textContent = '\u25B6 Start';
        algoSelect.disabled = false;
        frontierCells.clear();

        // golden glow animation shuru karo agar path mila
        if (pathList.length > 0) {
          glowPhase = 0;
        }

        updateStats();
        requestDraw();
        return;
      }
    }

    elapsedMs = performance.now() - startTime;
    updateStats();
    requestDraw();

    // next frame schedule karo
    requestAnimationFrame(stepAlgorithm);
  }

  // ============================================================
  // CLEAR FUNCTIONS
  // ============================================================

  function clearResults() {
    visitedCells.clear();
    frontierCells.clear();
    pathCells.clear();
    pathList = [];
    maxVisitOrder = 0;
    nodesExplored = 0;
    pathLength = 0;
    elapsedMs = 0;
    glowPhase = -1;
    updateStats();
  }

  function clearAll() {
    clearResults();
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        grid[r][c] = EMPTY;
      }
    }
    // start aur end defaults pe lao
    startPos = { r: Math.floor(rows / 2), c: 2 };
    endPos = { r: Math.floor(rows / 2), c: cols - 3 };
    grid[startPos.r][startPos.c] = START;
    grid[endPos.r][endPos.c] = END;
  }

  // ============================================================
  // MOUSE/TOUCH INTERACTION — wall draw, start/end drag
  // ============================================================

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
    const c = Math.floor(x / cellSize);
    const r = Math.floor(y / cellSize);
    if (r >= 0 && r < rows && c >= 0 && c < cols) return { r, c };
    return null;
  }

  function handlePointerDown(e) {
    const cell = getCellFromEvent(e);
    if (!cell) return;
    if (running) return; // algorithm chal raha hai toh drawing band

    isDragging = true;

    // start node drag?
    if (cell.r === startPos.r && cell.c === startPos.c) {
      dragMode = 'start';
      return;
    }
    // end node drag?
    if (cell.r === endPos.r && cell.c === endPos.c) {
      dragMode = 'end';
      return;
    }

    // wall draw ya erase — toggle based on what cell is
    if (grid[cell.r][cell.c] === WALL) {
      dragMode = 'wall-erase';
      grid[cell.r][cell.c] = EMPTY;
    } else if (grid[cell.r][cell.c] === EMPTY) {
      dragMode = 'wall-draw';
      grid[cell.r][cell.c] = WALL;
    }

    clearResults();
    requestDraw();
  }

  function handlePointerMove(e) {
    if (!isDragging) return;
    const cell = getCellFromEvent(e);
    if (!cell) return;

    if (dragMode === 'start') {
      // start node ko naye position pe le jao
      if (grid[cell.r][cell.c] !== END && grid[cell.r][cell.c] !== WALL) {
        grid[startPos.r][startPos.c] = EMPTY;
        startPos = { r: cell.r, c: cell.c };
        grid[startPos.r][startPos.c] = START;
        clearResults();
        requestDraw();
      }
      return;
    }

    if (dragMode === 'end') {
      // end node ko naye position pe le jao
      if (grid[cell.r][cell.c] !== START && grid[cell.r][cell.c] !== WALL) {
        grid[endPos.r][endPos.c] = EMPTY;
        endPos = { r: cell.r, c: cell.c };
        grid[endPos.r][endPos.c] = END;
        clearResults();
        requestDraw();
      }
      return;
    }

    // wall draw/erase — start/end pe draw mat karo
    if (grid[cell.r][cell.c] === START || grid[cell.r][cell.c] === END) return;

    if (dragMode === 'wall-draw') {
      if (grid[cell.r][cell.c] !== WALL) {
        grid[cell.r][cell.c] = WALL;
        clearResults();
        requestDraw();
      }
    } else if (dragMode === 'wall-erase') {
      if (grid[cell.r][cell.c] === WALL) {
        grid[cell.r][cell.c] = EMPTY;
        clearResults();
        requestDraw();
      }
    }
  }

  function handlePointerUp() {
    isDragging = false;
    dragMode = null;
  }

  // mouse events
  canvas.addEventListener('mousedown', handlePointerDown);
  canvas.addEventListener('mousemove', handlePointerMove);
  canvas.addEventListener('mouseup', handlePointerUp);
  canvas.addEventListener('mouseleave', handlePointerUp);

  // touch events — mobile pe bhi kaam kare
  canvas.addEventListener('touchstart', (e) => { e.preventDefault(); handlePointerDown(e); }, { passive: false });
  canvas.addEventListener('touchmove', (e) => { e.preventDefault(); handlePointerMove(e); }, { passive: false });
  canvas.addEventListener('touchend', handlePointerUp);
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  // ============================================================
  // CANVAS RENDERING — grid, cells, exploration, path, glow
  // ============================================================

  function draw() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    animationId = null;
    ctx.clearRect(0, 0, canvasW, canvasH);

    // background — dark
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // --- Grid cells draw karo ---
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = c * cellSize;
        const y = r * cellSize;
        const k = cellKey(r, c);
        const inset = 0.5; // grid line effect ke liye thoda gap

        // base cell — cell type ke hisaab se color
        if (grid[r][c] === WALL) {
          ctx.fillStyle = COL.wall;
          ctx.fillRect(x + inset, y + inset, cellSize - inset * 2, cellSize - inset * 2);
          continue;
        }

        if (grid[r][c] === START) {
          ctx.fillStyle = COL.start;
          ctx.fillRect(x + inset, y + inset, cellSize - inset * 2, cellSize - inset * 2);
          continue;
        }

        if (grid[r][c] === END) {
          ctx.fillStyle = COL.end;
          ctx.fillRect(x + inset, y + inset, cellSize - inset * 2, cellSize - inset * 2);
          continue;
        }

        // empty cell — check visited/frontier/path
        if (pathCells.has(k) && glowPhase < 0) {
          // final path — golden (agar glow animation nahi chal raha)
          ctx.fillStyle = COL.path;
          ctx.fillRect(x + inset, y + inset, cellSize - inset * 2, cellSize - inset * 2);
        } else if (frontierCells.has(k)) {
          // frontier — bright cyan
          ctx.fillStyle = COL.frontier;
          ctx.fillRect(x + inset, y + inset, cellSize - inset * 2, cellSize - inset * 2);
        } else if (visitedCells.has(k)) {
          // visited — purple/blue gradient based on visit order
          const order = visitedCells.get(k);
          const t = maxVisitOrder > 1 ? order / (maxVisitOrder - 1) : 0;
          const rr = Math.round(VISITED_START.r + (VISITED_END.r - VISITED_START.r) * t);
          const gg = Math.round(VISITED_START.g + (VISITED_END.g - VISITED_START.g) * t);
          const bb = Math.round(VISITED_START.b + (VISITED_END.b - VISITED_START.b) * t);
          ctx.fillStyle = 'rgb(' + rr + ',' + gg + ',' + bb + ')';
          ctx.fillRect(x + inset, y + inset, cellSize - inset * 2, cellSize - inset * 2);
        }
        // empty cells — background dikhaata hai, kuch draw nahi karte
      }
    }

    // --- Grid lines — very subtle ---
    ctx.strokeStyle = COL.gridLine;
    ctx.lineWidth = 0.5;
    for (let r = 0; r <= rows; r++) {
      ctx.beginPath();
      ctx.moveTo(0, r * cellSize);
      ctx.lineTo(cols * cellSize, r * cellSize);
      ctx.stroke();
    }
    for (let c = 0; c <= cols; c++) {
      ctx.beginPath();
      ctx.moveTo(c * cellSize, 0);
      ctx.lineTo(c * cellSize, rows * cellSize);
      ctx.stroke();
    }

    // --- Golden glow sweep animation — path milne ke baad ---
    if (glowPhase >= 0 && pathList.length > 0) {
      const totalCells = pathList.length;
      // glow sweep — ek bright line path pe sweep karti hai
      const headIdx = Math.floor(glowPhase * totalCells);

      for (let i = 0; i < totalCells; i++) {
        const k = pathList[i];
        const { r, c } = keyToRC(k);
        const x = c * cellSize;
        const y = r * cellSize;
        const inset = 0.5;

        if (i <= headIdx) {
          // glow head ke paas bright, peeche fade
          const dist = headIdx - i;
          const glowTail = 8; // kitne cells peeche tak glow failega
          let alpha = 1.0;
          if (dist > 0 && dist < glowTail) {
            alpha = 0.6 + 0.4 * (1 - dist / glowTail);
          } else if (dist >= glowTail) {
            alpha = 0.6;
          }

          // golden fill
          ctx.fillStyle = 'rgba(245,158,11,' + alpha + ')';
          ctx.fillRect(x + inset, y + inset, cellSize - inset * 2, cellSize - inset * 2);

          // glow effect — head ke paas bright glow
          if (dist < 3) {
            ctx.shadowColor = COL.pathGlow;
            ctx.shadowBlur = 8 * (1 - dist / 3);
            ctx.fillStyle = 'rgba(251,191,36,' + (0.5 * (1 - dist / 3)) + ')';
            ctx.fillRect(x + inset, y + inset, cellSize - inset * 2, cellSize - inset * 2);
            ctx.shadowBlur = 0;
          }
        }
      }

      // glow advance karo
      glowPhase += glowSpeed;
      if (glowPhase > 1.2) {
        // sweep complete — static golden path dikhaao
        glowPhase = -1;
      } else {
        // continue animation
        requestDraw();
      }
    }

    // --- Start/End labels ---
    const labelSize = Math.max(9, cellSize * 0.55);
    ctx.font = 'bold ' + labelSize + 'px ' + FONT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // start label — 'S' with dark outline
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillText('S', startPos.c * cellSize + cellSize / 2 + 0.5, startPos.r * cellSize + cellSize / 2 + 0.5);
    ctx.fillStyle = '#ffffff';
    ctx.fillText('S', startPos.c * cellSize + cellSize / 2, startPos.r * cellSize + cellSize / 2);

    // end label — 'E'
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillText('E', endPos.c * cellSize + cellSize / 2 + 0.5, endPos.r * cellSize + cellSize / 2 + 0.5);
    ctx.fillStyle = '#ffffff';
    ctx.fillText('E', endPos.c * cellSize + cellSize / 2, endPos.r * cellSize + cellSize / 2);
  }

  function requestDraw() {
    if (animationId) return;
    animationId = requestAnimationFrame(draw);
  }

  // ============================================================
  // STATS UPDATE — safe DOM manipulation, no innerHTML
  // ============================================================

  function updateStats() {
    const timeStr = elapsedMs > 0 ? (elapsedMs < 1000 ? elapsedMs.toFixed(0) + 'ms' : (elapsedMs / 1000).toFixed(2) + 's') : '-';

    // purane children hata do safely
    while (statsDiv.firstChild) statsDiv.removeChild(statsDiv.firstChild);

    const items = [
      { label: 'Algorithm', value: ALGORITHMS[currentAlgo].name, color: ACCENT },
      { label: 'Explored', value: '' + nodesExplored, color: '#c4b5fd' },
      { label: 'Path Length', value: pathLength > 0 ? '' + pathLength : '-', color: COL.path },
      { label: 'Time', value: timeStr, color: '#94a3b8' },
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
  // UI HELPERS — button/select/slider factory
  // ============================================================

  function makeSelectCSS() {
    return [
      'padding:5px 8px',
      'background:rgba(167,139,250,0.08)',
      'color:#e2e8f0',
      'border:1px solid rgba(167,139,250,0.25)',
      'border-radius:6px',
      'font-family:' + FONT,
      'font-size:11px',
      'cursor:pointer',
      'outline:none',
    ].join(';');
  }

  function makeBtnCSS() {
    return [
      'padding:5px 10px',
      'border:1px solid rgba(167,139,250,0.25)',
      'border-radius:6px',
      'background:rgba(167,139,250,0.06)',
      'color:#94a3b8',
      'font-family:' + FONT,
      'font-size:11px',
      'cursor:pointer',
      'transition:all 0.15s ease',
      'outline:none',
    ].join(';');
  }

  function makeActionBtnCSS() {
    return [
      'padding:5px 14px',
      'border:1px solid ' + ACCENT,
      'border-radius:6px',
      'background:rgba(167,139,250,0.15)',
      'color:#e2e8f0',
      'font-family:' + FONT,
      'font-size:11px',
      'font-weight:600',
      'cursor:pointer',
      'transition:all 0.15s ease',
      'outline:none',
    ].join(';');
  }

  function makeSliderGroup(label, min, max, initial, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const lbl = document.createElement('span');
    lbl.textContent = label + ': ' + initial;
    lbl.style.cssText = 'min-width:72px;font-family:' + FONT + ';font-size:11px;color:#94a3b8;';
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.value = initial;
    slider.style.cssText = [
      'width:100px',
      'height:4px',
      'accent-color:' + ACCENT,
      'cursor:pointer',
    ].join(';');
    slider.addEventListener('input', () => {
      const v = parseInt(slider.value);
      lbl.textContent = label + ': ' + v;
      onChange(v);
    });
    wrapper.appendChild(slider);

    return { wrapper, slider, label: lbl };
  }

  // ============================================================
  // INTERSECTION OBSERVER — sirf visible hone pe draw karo
  // ============================================================

  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) requestDraw();
    else if (!isVisible && animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  // lab resume: restart loop when focus released
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) draw(); });

  // resize handler
  window.addEventListener('resize', () => {
    resize();
    requestDraw();
  });

  // ============================================================
  // INITIALIZATION — sab set karo, grid banao, draw karo
  // ============================================================

  resize();

  // default start/end positions — grid ke left-center aur right-center
  startPos = { r: Math.floor(rows / 2), c: 2 };
  endPos = { r: Math.floor(rows / 2), c: cols - 3 };
  grid[startPos.r][startPos.c] = START;
  grid[endPos.r][endPos.c] = END;

  updateStats();
  requestDraw();
}
