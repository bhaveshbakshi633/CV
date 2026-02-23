// ============================================================
// Maze Solver — A* vs BFS vs DFS comparison
// Draw walls, place start/goal, dekho teeno algorithms race karte hue
// ============================================================

// main entry point — container dhundho aur maze demo shuru karo
export function initMaze() {
  const container = document.getElementById('mazeContainer');
  if (!container) {
    console.warn('mazeContainer nahi mila bhai, maze solver skip kar rahe hain');
    return;
  }

  // --- Grid constants ---
  const COLS = 20;
  const ROWS = 15;
  const CANVAS_HEIGHT = 380;

  // cell types — har cell ka ek state hoga
  const EMPTY = 0;
  const WALL = 1;
  const START = 2;
  const GOAL = 3;

  // --- Colors ---
  const COL_EMPTY = 'rgba(30,30,50,0.6)';
  const COL_WALL = 'rgba(200,200,220,0.25)';
  const COL_START = 'rgba(74,255,143,0.8)';
  const COL_GOAL = 'rgba(255,80,80,0.8)';
  const COL_GRID = 'rgba(74,158,255,0.08)';

  // algorithm colors — visited cells ke liye
  const COL_ASTAR = { visited: 'rgba(74,158,255,0.2)', path: 'rgba(74,158,255,0.7)', label: 'A*' };
  const COL_BFS = { visited: 'rgba(74,255,143,0.15)', path: 'rgba(74,255,143,0.6)', label: 'BFS' };
  const COL_DFS = { visited: 'rgba(249,158,11,0.15)', path: 'rgba(249,158,11,0.6)', label: 'DFS' };

  // --- State ---
  let grid = [];
  let startPos = { r: 1, c: 1 };
  let goalPos = { r: ROWS - 2, c: COLS - 2 };

  // drawing state
  let isDrawing = false;
  let drawMode = 'wall'; // 'wall' ya 'erase'
  let placingStart = false;
  let placingGoal = false;

  // solving state
  let solving = false;
  let solveSpeed = 20; // cells per second — slider se change hoga
  let solveAnimFrame = null;

  // algorithm visualization data
  let algoVis = {
    astar: { visited: new Set(), path: [], frontier: [], done: false, stats: { visited: 0, pathLen: 0 } },
    bfs: { visited: new Set(), path: [], frontier: [], done: false, stats: { visited: 0, pathLen: 0 } },
    dfs: { visited: new Set(), path: [], frontier: [], done: false, stats: { visited: 0, pathLen: 0 } },
  };

  // algorithm toggle — kaunsa dikhana hai
  let showAlgo = { astar: true, bfs: true, dfs: true };

  // visibility
  let isVisible = false;
  let animationId = null;

  // --- Grid initialization ---
  function initGrid() {
    grid = Array.from({ length: ROWS }, () => new Array(COLS).fill(EMPTY));
    grid[startPos.r][startPos.c] = START;
    grid[goalPos.r][goalPos.c] = GOAL;
    clearSolveData();
  }

  // solve data reset kar — paths aur visited sets saaf karo
  function clearSolveData() {
    solving = false;
    Object.keys(algoVis).forEach(k => {
      algoVis[k].visited = new Set();
      algoVis[k].path = [];
      algoVis[k].frontier = [];
      algoVis[k].done = false;
      algoVis[k].stats = { visited: 0, pathLen: 0 };
    });
    if (solveAnimFrame) {
      cancelAnimationFrame(solveAnimFrame);
      solveAnimFrame = null;
    }
  }

  // --- Cell key helper — "r,c" format ---
  function cellKey(r, c) {
    return r + ',' + c;
  }

  // --- Neighbors — 4-connected grid ---
  function getNeighbors(r, c) {
    const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    const result = [];
    for (const [dr, dc] of dirs) {
      const nr = r + dr;
      const nc = c + dc;
      if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && grid[nr][nc] !== WALL) {
        result.push({ r: nr, c: nc });
      }
    }
    return result;
  }

  // --- Manhattan distance — A* heuristic ---
  function manhattan(r1, c1, r2, c2) {
    return Math.abs(r1 - r2) + Math.abs(c1 - c2);
  }

  // --- Path reconstruct kar — cameFrom map se ---
  function reconstructPath(cameFrom, endKey) {
    const path = [];
    let current = endKey;
    while (current) {
      const [r, c] = current.split(',').map(Number);
      path.unshift({ r, c });
      current = cameFrom.get(current);
    }
    return path;
  }

  // --- A* algorithm — step by step generator ---
  // generator use kar rahe hain taaki har step pe yield kar sakein visualization ke liye
  function* astarGenerator() {
    const start = cellKey(startPos.r, startPos.c);
    const goal = cellKey(goalPos.r, goalPos.c);

    // priority queue — simple array sort karke (chhota grid hai, efficient nahi chahiye)
    const openSet = [{ key: start, f: 0 }];
    const gScore = new Map();
    gScore.set(start, 0);
    const cameFrom = new Map();
    cameFrom.set(start, null);

    while (openSet.length > 0) {
      // sabse kam f-score wala nikal
      openSet.sort((a, b) => a.f - b.f);
      const current = openSet.shift();

      algoVis.astar.visited.add(current.key);
      algoVis.astar.stats.visited = algoVis.astar.visited.size;
      yield; // har step pe ruk — visualization ke liye

      // goal mila? path bana aur return kar
      if (current.key === goal) {
        algoVis.astar.path = reconstructPath(cameFrom, goal);
        algoVis.astar.stats.pathLen = algoVis.astar.path.length;
        algoVis.astar.done = true;
        return;
      }

      const [cr, cc] = current.key.split(',').map(Number);
      const neighbors = getNeighbors(cr, cc);

      for (const n of neighbors) {
        const nKey = cellKey(n.r, n.c);
        const tentativeG = gScore.get(current.key) + 1;

        if (!gScore.has(nKey) || tentativeG < gScore.get(nKey)) {
          gScore.set(nKey, tentativeG);
          cameFrom.set(nKey, current.key);
          const f = tentativeG + manhattan(n.r, n.c, goalPos.r, goalPos.c);
          // agar already openSet mein nahi hai toh add kar
          if (!openSet.some(item => item.key === nKey)) {
            openSet.push({ key: nKey, f });
          }
        }
      }
    }

    // path nahi mila — goal unreachable hai
    algoVis.astar.done = true;
  }

  // --- BFS — breadth first search generator ---
  function* bfsGenerator() {
    const start = cellKey(startPos.r, startPos.c);
    const goal = cellKey(goalPos.r, goalPos.c);

    const queue = [start];
    const visited = new Set([start]);
    const cameFrom = new Map();
    cameFrom.set(start, null);

    while (queue.length > 0) {
      const current = queue.shift(); // FIFO — pehle aaya pehle jaayega

      algoVis.bfs.visited.add(current);
      algoVis.bfs.stats.visited = algoVis.bfs.visited.size;
      yield;

      if (current === goal) {
        algoVis.bfs.path = reconstructPath(cameFrom, goal);
        algoVis.bfs.stats.pathLen = algoVis.bfs.path.length;
        algoVis.bfs.done = true;
        return;
      }

      const [cr, cc] = current.split(',').map(Number);
      const neighbors = getNeighbors(cr, cc);

      for (const n of neighbors) {
        const nKey = cellKey(n.r, n.c);
        if (!visited.has(nKey)) {
          visited.add(nKey);
          cameFrom.set(nKey, current);
          queue.push(nKey);
        }
      }
    }

    algoVis.bfs.done = true;
  }

  // --- DFS — depth first search generator ---
  function* dfsGenerator() {
    const start = cellKey(startPos.r, startPos.c);
    const goal = cellKey(goalPos.r, goalPos.c);

    const stack = [start]; // LIFO — last in first out
    const visited = new Set();
    const cameFrom = new Map();
    cameFrom.set(start, null);

    while (stack.length > 0) {
      const current = stack.pop(); // stack se top element nikal

      if (visited.has(current)) continue;
      visited.add(current);

      algoVis.dfs.visited.add(current);
      algoVis.dfs.stats.visited = algoVis.dfs.visited.size;
      yield;

      if (current === goal) {
        algoVis.dfs.path = reconstructPath(cameFrom, goal);
        algoVis.dfs.stats.pathLen = algoVis.dfs.path.length;
        algoVis.dfs.done = true;
        return;
      }

      const [cr, cc] = current.split(',').map(Number);
      const neighbors = getNeighbors(cr, cc);

      // reverse kar taaki consistent order mein explore ho
      for (let i = neighbors.length - 1; i >= 0; i--) {
        const n = neighbors[i];
        const nKey = cellKey(n.r, n.c);
        if (!visited.has(nKey)) {
          cameFrom.set(nKey, current);
          stack.push(nKey);
        }
      }
    }

    algoVis.dfs.done = true;
  }

  // --- Maze generation — recursive backtracking ---
  function generateMaze() {
    // pehle sab wall bana de
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        grid[r][c] = WALL;
      }
    }

    // recursive backtracking — odd positions pe cells banayenge
    const visited = new Set();

    function carve(r, c) {
      visited.add(cellKey(r, c));
      grid[r][c] = EMPTY;

      // random directions — shuffle kar
      const dirs = [[0, 2], [0, -2], [2, 0], [-2, 0]];
      shuffleArray(dirs);

      for (const [dr, dc] of dirs) {
        const nr = r + dr;
        const nc = c + dc;
        if (nr > 0 && nr < ROWS - 1 && nc > 0 && nc < COLS - 1 && !visited.has(cellKey(nr, nc))) {
          // beech ki wall bhi hata — path bana do
          grid[r + dr / 2][c + dc / 2] = EMPTY;
          carve(nr, nc);
        }
      }
    }

    // odd position se start kar — grid alignment sahi rahe
    carve(1, 1);

    // start aur goal set kar
    startPos = { r: 1, c: 1 };
    goalPos = { r: ROWS - 2, c: COLS - 2 };
    // ensure goal cell is empty — agar wall hai toh hata
    grid[goalPos.r][goalPos.c] = EMPTY;
    grid[startPos.r][startPos.c] = EMPTY;

    // agar goal odd row/col pe nahi hai toh nearest odd pe le ja
    if (goalPos.r % 2 === 0) goalPos.r = Math.min(ROWS - 2, goalPos.r + 1);
    if (goalPos.c % 2 === 0) goalPos.c = Math.min(COLS - 2, goalPos.c + 1);
    grid[goalPos.r][goalPos.c] = EMPTY;

    grid[startPos.r][startPos.c] = START;
    grid[goalPos.r][goalPos.c] = GOAL;

    clearSolveData();
  }

  // Fisher-Yates shuffle
  function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  // --- DOM structure ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — grid yahan dikhega
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
    'touch-action:none',
  ].join(';');
  container.appendChild(canvas);

  // stats bar
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'justify-content:center',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:8px',
    'font-family:monospace',
    'font-size:11px',
    'color:#b0b0b0',
  ].join(';');
  container.appendChild(statsDiv);

  // controls bar — row 1
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

  // controls bar — row 2 (algorithm toggles + speed)
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
      'background:rgba(74,158,255,0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(74,158,255,0.25)',
      'border-radius:6px',
      'padding:5px 12px',
      'font-size:11px',
      'font-family:monospace',
      'cursor:pointer',
      'transition:all 0.2s',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(74,158,255,0.25)';
      btn.style.color = '#ffffff';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(74,158,255,0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    parent.appendChild(btn);
    return btn;
  }

  // row 1 buttons
  makeButton('Solve', controlsDiv1, startSolve);
  makeButton('Clear Paths', controlsDiv1, () => {
    clearSolveData();
  });
  makeButton('Clear All', controlsDiv1, () => {
    initGrid();
  });
  makeButton('Generate Maze', controlsDiv1, generateMaze);

  // draw mode toggle
  const drawBtn = makeButton('Draw: Wall', controlsDiv1, () => {
    drawMode = drawMode === 'wall' ? 'erase' : 'wall';
    drawBtn.textContent = 'Draw: ' + (drawMode === 'wall' ? 'Wall' : 'Erase');
  });

  // row 2 — speed slider
  const speedLabel = document.createElement('span');
  speedLabel.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:monospace;';
  speedLabel.textContent = 'Speed:';
  controlsDiv2.appendChild(speedLabel);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '100';
  speedSlider.value = '20';
  speedSlider.style.cssText = 'width:80px;height:4px;accent-color:rgba(74,158,255,0.8);cursor:pointer;';
  speedSlider.addEventListener('input', () => {
    solveSpeed = parseInt(speedSlider.value);
  });
  controlsDiv2.appendChild(speedSlider);

  // algorithm toggles
  function makeToggle(algoKey, label, color) {
    const btn = document.createElement('button');
    btn.textContent = label + ': ON';
    btn.style.cssText = [
      'background:rgba(74,158,255,0.1)',
      'color:' + color,
      'border:1px solid ' + color,
      'border-radius:6px',
      'padding:4px 10px',
      'font-size:10px',
      'font-family:monospace',
      'cursor:pointer',
      'opacity:0.8',
    ].join(';');
    btn.addEventListener('click', () => {
      showAlgo[algoKey] = !showAlgo[algoKey];
      btn.textContent = label + ': ' + (showAlgo[algoKey] ? 'ON' : 'OFF');
      btn.style.opacity = showAlgo[algoKey] ? '0.8' : '0.3';
    });
    controlsDiv2.appendChild(btn);
  }

  makeToggle('astar', 'A*', COL_ASTAR.path);
  makeToggle('bfs', 'BFS', COL_BFS.path);
  makeToggle('dfs', 'DFS', COL_DFS.path);

  // --- Canvas sizing ---
  let cellSize = 1;

  function resizeCanvas() {
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = w + 'px';

    // cell size calculate kar — container width se
    cellSize = Math.floor((w * dpr) / COLS);
  }

  // --- Mouse/touch mein grid coordinate nikal ---
  function getCellFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    let clientX, clientY;

    if (e.touches) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    const x = (clientX - rect.left) * dpr;
    const y = (clientY - rect.top) * dpr;
    const c = Math.floor(x / cellSize);
    const r = Math.floor(y / cellSize);

    if (r >= 0 && r < ROWS && c >= 0 && c < COLS) return { r, c };
    return null;
  }

  // --- Mouse/touch handlers — wall draw aur start/goal placement ---
  function handlePointerDown(e) {
    const cell = getCellFromEvent(e);
    if (!cell) return;

    // right click ya ctrl+click = goal placement
    if (e.button === 2 || e.ctrlKey) {
      e.preventDefault();
      // purana goal hata
      grid[goalPos.r][goalPos.c] = EMPTY;
      goalPos = { r: cell.r, c: cell.c };
      grid[goalPos.r][goalPos.c] = GOAL;
      clearSolveData();
      return;
    }

    // start/goal cell pe click hua? toh usse move mode mein mat jao
    if (grid[cell.r][cell.c] === START) {
      placingStart = true;
      isDrawing = false;
      return;
    }
    if (grid[cell.r][cell.c] === GOAL) {
      placingGoal = true;
      isDrawing = false;
      return;
    }

    // normal drawing — wall ya erase
    isDrawing = true;
    applyDraw(cell);
  }

  function handlePointerMove(e) {
    const cell = getCellFromEvent(e);
    if (!cell) return;

    if (placingStart) {
      grid[startPos.r][startPos.c] = EMPTY;
      startPos = { r: cell.r, c: cell.c };
      grid[startPos.r][startPos.c] = START;
      clearSolveData();
      return;
    }

    if (placingGoal) {
      grid[goalPos.r][goalPos.c] = EMPTY;
      goalPos = { r: cell.r, c: cell.c };
      grid[goalPos.r][goalPos.c] = GOAL;
      clearSolveData();
      return;
    }

    if (isDrawing) {
      applyDraw(cell);
    }
  }

  function handlePointerUp() {
    isDrawing = false;
    placingStart = false;
    placingGoal = false;
  }

  function applyDraw(cell) {
    // start/goal pe draw mat kar — wo protect karo
    if (grid[cell.r][cell.c] === START || grid[cell.r][cell.c] === GOAL) return;

    if (drawMode === 'wall') {
      grid[cell.r][cell.c] = WALL;
    } else {
      grid[cell.r][cell.c] = EMPTY;
    }
    clearSolveData();
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

  // right click prevent karo — goal placement ke liye use ho raha hai
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  // --- Solve function — teeno algorithms ek saath chala step by step ---
  let generators = {};

  function startSolve() {
    if (solving) return;
    clearSolveData();
    solving = true;

    // generators bana — A*, BFS, DFS
    generators = {
      astar: astarGenerator(),
      bfs: bfsGenerator(),
      dfs: dfsGenerator(),
    };

    animateSolve();
  }

  function animateSolve() {
    if (!solving) return;

    // speed ke hisaab se steps chala — ek frame mein
    const stepsPerFrame = Math.max(1, Math.ceil(solveSpeed / 10));

    let allDone = true;

    for (let s = 0; s < stepsPerFrame; s++) {
      Object.keys(generators).forEach(k => {
        if (!algoVis[k].done) {
          const result = generators[k].next();
          if (result.done) {
            algoVis[k].done = true;
          } else {
            allDone = false;
          }
        }
      });
    }

    // check agar koi bhi algorithm abhi bhi chal raha hai
    if (!Object.values(algoVis).every(a => a.done)) {
      allDone = false;
    }

    if (allDone) {
      solving = false;
    } else {
      solveAnimFrame = requestAnimationFrame(animateSolve);
    }
  }

  // --- Canvas rendering ---
  function draw() {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, w, h);

    // grid cells draw kar
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const x = c * cellSize;
        const y = r * cellSize;
        const key = cellKey(r, c);

        // base cell color
        let fillColor = COL_EMPTY;
        if (grid[r][c] === WALL) fillColor = COL_WALL;
        else if (grid[r][c] === START) fillColor = COL_START;
        else if (grid[r][c] === GOAL) fillColor = COL_GOAL;

        ctx.fillStyle = fillColor;
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

        // algorithm visited overlay — sirf agar toggle on hai
        if (grid[r][c] !== WALL && grid[r][c] !== START && grid[r][c] !== GOAL) {
          if (showAlgo.astar && algoVis.astar.visited.has(key)) {
            ctx.fillStyle = COL_ASTAR.visited;
            ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
          }
          if (showAlgo.bfs && algoVis.bfs.visited.has(key)) {
            ctx.fillStyle = COL_BFS.visited;
            ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
          }
          if (showAlgo.dfs && algoVis.dfs.visited.has(key)) {
            ctx.fillStyle = COL_DFS.visited;
            ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
          }
        }
      }
    }

    // final paths draw kar — bright colors mein
    function drawPath(path, color, offsetX, offsetY) {
      if (path.length < 2) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, cellSize * 0.25);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();

      for (let i = 0; i < path.length; i++) {
        const x = path[i].c * cellSize + cellSize / 2 + offsetX;
        const y = path[i].r * cellSize + cellSize / 2 + offsetY;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // paths ko thoda offset kar taaki overlap na ho — teeno visible rahein
    const pathOffset = Math.max(1, cellSize * 0.12);
    if (showAlgo.astar && algoVis.astar.path.length > 0) {
      drawPath(algoVis.astar.path, COL_ASTAR.path, -pathOffset, 0);
    }
    if (showAlgo.bfs && algoVis.bfs.path.length > 0) {
      drawPath(algoVis.bfs.path, COL_BFS.path, 0, -pathOffset);
    }
    if (showAlgo.dfs && algoVis.dfs.path.length > 0) {
      drawPath(algoVis.dfs.path, COL_DFS.path, pathOffset, pathOffset);
    }

    // start/goal labels — S aur G
    ctx.font = 'bold ' + Math.max(10, cellSize * 0.5) + 'px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillText('S', startPos.c * cellSize + cellSize / 2, startPos.r * cellSize + cellSize / 2);
    ctx.fillText('G', goalPos.c * cellSize + cellSize / 2, goalPos.r * cellSize + cellSize / 2);

    // instruction text — top left
    ctx.font = (10 * dpr) + 'px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.4)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Click: draw walls | Right-click: set goal | Drag S/G to move', 8 * dpr, 5 * dpr);
  }

  // --- Stats update ---
  function updateStats() {
    while (statsDiv.firstChild) statsDiv.removeChild(statsDiv.firstChild);

    function addStat(label, visited, pathLen, color) {
      const span = document.createElement('span');
      const labelNode = document.createTextNode(label + ': ');
      span.appendChild(labelNode);
      const valSpan = document.createElement('span');
      valSpan.style.color = color;
      valSpan.textContent = visited + ' visited, path: ' + pathLen;
      span.appendChild(valSpan);
      statsDiv.appendChild(span);
    }

    if (showAlgo.astar) {
      addStat('A*', algoVis.astar.stats.visited, algoVis.astar.stats.pathLen, COL_ASTAR.path);
    }
    if (showAlgo.bfs) {
      addStat('BFS', algoVis.bfs.stats.visited, algoVis.bfs.stats.pathLen, COL_BFS.path);
    }
    if (showAlgo.dfs) {
      addStat('DFS', algoVis.dfs.stats.visited, algoVis.dfs.stats.pathLen, COL_DFS.path);
    }
  }

  // --- Main render loop ---
  function animate() {
    if (!isVisible) {
      animationId = requestAnimationFrame(animate);
      return;
    }

    draw();
    updateStats();
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf visible hone pe render kar ---
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        isVisible = entry.isIntersecting;
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);

  // --- Init ---
  initGrid();
  resizeCanvas();

  window.addEventListener('resize', resizeCanvas);

  // render loop shuru kar
  animate();
}
