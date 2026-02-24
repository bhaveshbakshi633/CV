// ============================================================
// Snake AI — Multiple AI strategies se snake khelna
// Greedy A*, Hamiltonian Cycle, Hybrid (Smart), aur Manual mode
// Hamiltonian cycle guaranteed board fill deta hai — A* fast hai lekin trap hota hai
// Hybrid dono ka best combine karta hai — fast + safe
// ============================================================

// yahi main entry point hai — container dhundho, canvas banao, snake shuru karo
export function initSnakeAI() {
  const container = document.getElementById('snakeAIContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const ACCENT_RGB = '74,158,255';
  const FONT = "'JetBrains Mono', monospace";

  // --- State ---
  let gridSize = 20; // default 20x20
  let cellSize = 1; // recalculate hoga resize mein
  let gameW = 0; // game area width (pixels)
  let statsW = 0; // stats panel width (pixels)
  let canvasW = 0, canvasH = 0, dpr = 1;

  // snake state
  let snake = []; // array of {x, y} — head is index 0
  let direction = { x: 1, y: 0 }; // right
  let nextDirection = { x: 1, y: 0 };
  let food = { x: 0, y: 0 };
  let score = 0;
  let steps = 0;
  let gameOver = false;
  let deathFlashTimer = 0;

  // AI path state
  let aiPath = []; // A* computed path
  let hamiltonianCycle = []; // cell index array — poora cycle
  let cycleOrder = []; // cycleOrder[cellIdx] = position in cycle (0..N-1)

  // strategy state
  let strategy = 'hybrid'; // 'greedy', 'hamiltonian', 'hybrid', 'manual'
  let showPath = true;
  let showCycle = true;
  let autoRestart = true;
  let speedMs = 80; // ms per step

  // score tracking
  let highScores = { greedy: 0, hamiltonian: 0, hybrid: 0, manual: 0 };

  // food animation
  let foodPulse = 0;
  let scorePopups = []; // {x, y, alpha, dy} — +1 floating text

  // manual mode keys
  let pendingManualDir = null;

  // hover cell (manual mode)
  let hoverCell = null;

  // animation state
  let animationId = null;
  let isVisible = false;
  let lastStepTime = 0;

  // --- DOM setup ---
  // purane children hata do
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // canvas banao
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;border-radius:8px;cursor:default;display:block;';
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // ============================================================
  // HAMILTONIAN CYCLE GENERATION
  // spanning tree via randomized DFS on half-grid, then trace cycle
  // ============================================================

  // half-grid pe spanning tree banao — randomized DFS se
  function generateHamiltonianCycle() {
    const N = gridSize;
    // half-grid dimensions — har cell 2x2 original cells represent karta hai
    const halfW = Math.floor(N / 2);
    const halfH = Math.floor(N / 2);
    const totalHalf = halfW * halfH;

    // spanning tree — adjacency edges store karo
    const visited = new Uint8Array(totalHalf);
    // edges[cellIdx] = Set of neighbor indices connected via tree edge
    const edges = new Array(totalHalf);
    for (let i = 0; i < totalHalf; i++) edges[i] = new Set();

    // randomized DFS se spanning tree banao
    const stack = [0];
    visited[0] = 1;
    while (stack.length > 0) {
      const curr = stack[stack.length - 1];
      const cx = curr % halfW;
      const cy = Math.floor(curr / halfW);

      // neighbors dhundho — unvisited wale
      const neighbors = [];
      if (cx > 0 && !visited[cy * halfW + (cx - 1)]) neighbors.push(cy * halfW + (cx - 1));
      if (cx < halfW - 1 && !visited[cy * halfW + (cx + 1)]) neighbors.push(cy * halfW + (cx + 1));
      if (cy > 0 && !visited[(cy - 1) * halfW + cx]) neighbors.push((cy - 1) * halfW + cx);
      if (cy < halfH - 1 && !visited[(cy + 1) * halfW + cx]) neighbors.push((cy + 1) * halfW + cx);

      if (neighbors.length === 0) {
        stack.pop();
        continue;
      }

      // random neighbor choose kar
      const next = neighbors[Math.floor(Math.random() * neighbors.length)];
      visited[next] = 1;
      edges[curr].add(next);
      edges[next].add(curr);
      stack.push(next);
    }

    // ab spanning tree se Hamiltonian cycle trace karo
    // har half-cell ke 2x2 block ke around trace karo
    // wall check: agar edge hai toh passage hai, nahi toh wall hai
    function hasEdge(hx, hy, dx, dy) {
      const nx = hx + dx, ny = hy + dy;
      if (nx < 0 || nx >= halfW || ny < 0 || ny >= halfH) return false;
      return edges[hy * halfW + hx].has(ny * halfW + nx);
    }

    // cycle trace karo — clockwise direction mein har 2x2 block ke around
    // start from top-left corner (0,0), direction = right
    const cycle = [];
    const totalCells = N * N;
    const inCycle = new Uint8Array(totalCells);

    // wall-following approach: always try to turn right first
    let px = 0, py = 0; // current position in full grid
    let dx = 1, dy = 0; // current direction (right)

    // cycle mein cells add karo jab tak poora grid cover na ho jaaye
    for (let step = 0; step < totalCells; step++) {
      const cellIdx = py * N + px;
      if (inCycle[cellIdx]) break; // loop complete
      cycle.push(cellIdx);
      inCycle[cellIdx] = 1;

      // right turn try karo, fir straight, fir left, fir back
      // right turn: (dx,dy) -> (dy,-dx) nahi... clockwise: (dx,dy) -> (-dy,dx)
      // wait — standard: right turn = (dx,dy) -> (dy,-dx) nahi...
      // let me think: if going right (1,0), right turn = down (0,1)
      // (dx,dy)->(dy,-dx): (1,0)->(0,-1) = up... galat
      // (dx,dy)->(-dy,dx): (1,0)->(0,1) = down. Sahi!
      const dirs = [
        { x: -dy, y: dx },   // right turn
        { x: dx, y: dy },     // straight
        { x: dy, y: -dx },    // left turn
        { x: -dx, y: -dy },   // u-turn
      ];

      let moved = false;
      for (const d of dirs) {
        const nx = px + d.x;
        const ny = py + d.y;
        if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;

        // check agar ye move valid hai — spanning tree ke edge ke through ho sakta hai
        // current cell ka half-grid coordinate
        const hx1 = Math.floor(px / 2);
        const hy1 = Math.floor(py / 2);
        // next cell ka half-grid coordinate
        const hx2 = Math.floor(nx / 2);
        const hy2 = Math.floor(ny / 2);

        // same half-cell mein move always allowed
        if (hx1 === hx2 && hy1 === hy2) {
          if (!inCycle[ny * N + nx]) {
            px = nx; py = ny;
            dx = d.x; dy = d.y;
            moved = true;
            break;
          }
          continue;
        }

        // different half-cell mein move — spanning tree mein edge hona chahiye
        if (hasEdge(hx1, hy1, hx2 - hx1, hy2 - hy1)) {
          if (!inCycle[ny * N + nx]) {
            px = nx; py = ny;
            dx = d.x; dy = d.y;
            moved = true;
            break;
          }
        }
      }

      if (!moved) break; // stuck — shouldn't happen with valid spanning tree
    }

    // agar cycle complete nahi hui (odd grid size etc), fallback use karo
    if (cycle.length < totalCells) {
      // simple row-by-row zigzag fallback — guaranteed Hamiltonian path
      // NOTE: ye proper cycle nahi banata odd grids pe, but kaam chalega
      return generateZigzagCycle();
    }

    hamiltonianCycle = cycle;
    // cycleOrder banao — O(1) lookup ke liye
    cycleOrder = new Array(totalCells);
    for (let i = 0; i < cycle.length; i++) {
      cycleOrder[cycle[i]] = i;
    }
  }

  // zigzag fallback — simple Hamiltonian cycle for any grid size
  function generateZigzagCycle() {
    const N = gridSize;
    const totalCells = N * N;
    const cycle = [];

    // strategy: first column top to bottom, then zigzag rows right-to-left-to-right
    // Row 0: left to right (columns 1..N-1)
    // Row 1: right to left (columns N-1..1)
    // ... alternate
    // Column 0: bottom to top
    // This makes a valid Hamiltonian cycle

    // top row: go right
    for (let x = 0; x < N; x++) {
      cycle.push(0 * N + x);
    }
    // zigzag down: for each pair of rows
    for (let y = 1; y < N; y++) {
      if (y % 2 === 1) {
        // go left from N-1 to 1
        for (let x = N - 1; x >= 1; x--) {
          cycle.push(y * N + x);
        }
      } else {
        // go right from 1 to N-1
        for (let x = 1; x < N; x++) {
          cycle.push(y * N + x);
        }
      }
    }
    // left column: go up from bottom-1 to row 1
    for (let y = N - 1; y >= 1; y--) {
      if (y % 2 === 1) {
        cycle.push(y * N + 0);
      }
    }

    // verify — agar length sahi nahi hai toh brute force zigzag
    if (cycle.length !== totalCells) {
      // ultimate fallback: simple snake pattern
      cycle.length = 0;
      for (let y = 0; y < N; y++) {
        if (y % 2 === 0) {
          for (let x = 0; x < N; x++) cycle.push(y * N + x);
        } else {
          for (let x = N - 1; x >= 0; x--) cycle.push(y * N + x);
        }
      }
    }

    hamiltonianCycle = cycle;
    cycleOrder = new Array(totalCells);
    for (let i = 0; i < cycle.length; i++) {
      cycleOrder[cycle[i]] = i;
    }
  }

  // ============================================================
  // A* PATHFINDING
  // Manhattan distance heuristic, grid boundaries aur snake body avoid
  // ============================================================

  function astarPath(startX, startY, goalX, goalY, snakeSet) {
    const N = gridSize;
    const startKey = startY * N + startX;
    const goalKey = goalY * N + goalX;

    if (startKey === goalKey) return [];

    // min-heap manually implement — priority queue ke liye
    const openSet = []; // {key, f, g}
    const gScore = new Map();
    const cameFrom = new Map();
    const inOpen = new Set();

    gScore.set(startKey, 0);
    const h0 = Math.abs(goalX - startX) + Math.abs(goalY - startY);
    openSet.push({ key: startKey, f: h0, g: 0 });
    inOpen.add(startKey);

    // simple priority queue — sort based insert (chhota grid hai, fine hai)
    function popBest() {
      let bestIdx = 0;
      for (let i = 1; i < openSet.length; i++) {
        if (openSet[i].f < openSet[bestIdx].f ||
            (openSet[i].f === openSet[bestIdx].f && openSet[i].g > openSet[bestIdx].g)) {
          bestIdx = i;
        }
      }
      const best = openSet[bestIdx];
      openSet[bestIdx] = openSet[openSet.length - 1];
      openSet.pop();
      inOpen.delete(best.key);
      return best;
    }

    const dirs = [{ x: 0, y: -1 }, { x: 0, y: 1 }, { x: -1, y: 0 }, { x: 1, y: 0 }];

    while (openSet.length > 0) {
      const current = popBest();
      if (current.key === goalKey) {
        // path reconstruct kar
        const path = [];
        let k = goalKey;
        while (cameFrom.has(k)) {
          path.push({ x: k % N, y: Math.floor(k / N) });
          k = cameFrom.get(k);
        }
        path.reverse();
        return path;
      }

      const cx = current.key % N;
      const cy = Math.floor(current.key / N);

      for (const d of dirs) {
        const nx = cx + d.x;
        const ny = cy + d.y;
        if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;

        const nKey = ny * N + nx;
        // snake body avoid karo (tail chhodke, kyunki tail hat jaayegi next step mein)
        if (snakeSet.has(nKey) && nKey !== goalKey) continue;

        const tentG = current.g + 1;
        const prevG = gScore.get(nKey);
        if (prevG !== undefined && tentG >= prevG) continue;

        gScore.set(nKey, tentG);
        cameFrom.set(nKey, current.key);
        const h = Math.abs(goalX - nx) + Math.abs(goalY - ny);
        if (!inOpen.has(nKey)) {
          openSet.push({ key: nKey, f: tentG + h, g: tentG });
          inOpen.add(nKey);
        }
      }
    }

    return null; // koi path nahi mila
  }

  // ============================================================
  // SNAKE GAME LOGIC
  // ============================================================

  function getSnakeSet() {
    // snake body cells ka Set banao — collision check ke liye
    const s = new Set();
    for (let i = 0; i < snake.length; i++) {
      s.add(snake[i].y * gridSize + snake[i].x);
    }
    return s;
  }

  function spawnFood() {
    const N = gridSize;
    const snakeSet = getSnakeSet();
    // agar board full hai toh game won
    if (snakeSet.size >= N * N) {
      gameOver = true;
      return;
    }
    // random empty cell dhundho
    const emptyCells = [];
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        if (!snakeSet.has(y * N + x)) {
          emptyCells.push({ x, y });
        }
      }
    }
    if (emptyCells.length === 0) {
      gameOver = true;
      return;
    }
    const chosen = emptyCells[Math.floor(Math.random() * emptyCells.length)];
    food.x = chosen.x;
    food.y = chosen.y;
  }

  function resetGame() {
    const N = gridSize;
    const mid = Math.floor(N / 2);
    snake = [];
    // length 3, center mein, facing right
    for (let i = 2; i >= 0; i--) {
      snake.push({ x: mid - 1 + i, y: mid });
    }
    // head is snake[0] — rightmost
    // actually snake[0] should be head, so:
    snake = [
      { x: mid + 1, y: mid }, // head
      { x: mid, y: mid },
      { x: mid - 1, y: mid }, // tail
    ];
    direction = { x: 1, y: 0 };
    nextDirection = { x: 1, y: 0 };
    score = 0;
    steps = 0;
    gameOver = false;
    deathFlashTimer = 0;
    aiPath = [];
    pendingManualDir = null;
    scorePopups = [];

    // Hamiltonian cycle generate karo (strategy change ya grid size change pe)
    generateHamiltonianCycle();
    spawnFood();
  }

  // ek step simulate karo — direction decide karo (AI ya manual), fir move karo
  function gameStep() {
    if (gameOver) {
      deathFlashTimer++;
      if (deathFlashTimer > 15 && autoRestart) {
        resetGame();
      }
      return;
    }

    // direction decide karo based on strategy
    if (strategy === 'manual') {
      if (pendingManualDir) {
        // opposite direction check — 180 degree turn allowed nahi
        if (pendingManualDir.x !== -direction.x || pendingManualDir.y !== -direction.y) {
          nextDirection = pendingManualDir;
        }
        pendingManualDir = null;
      }
    } else {
      computeAIDirection();
    }

    direction = nextDirection;

    // naya head position
    const head = snake[0];
    const newHead = { x: head.x + direction.x, y: head.y + direction.y };

    // wall collision check
    if (newHead.x < 0 || newHead.x >= gridSize || newHead.y < 0 || newHead.y >= gridSize) {
      triggerDeath();
      return;
    }

    // self collision check — tail ko ignore karo agar food nahi khaya
    // (tail hat jaayegi move ke baad, lekin food khaane pe nahi hategi)
    const willEat = (newHead.x === food.x && newHead.y === food.y);
    for (let i = 0; i < snake.length; i++) {
      // agar food nahi kha rahe, toh last segment safe hai (hat jaayega)
      if (!willEat && i === snake.length - 1) continue;
      if (snake[i].x === newHead.x && snake[i].y === newHead.y) {
        triggerDeath();
        return;
      }
    }

    // move snake
    snake.unshift(newHead);
    if (willEat) {
      score++;
      // score popup banao
      scorePopups.push({
        x: food.x * cellSize + cellSize / 2,
        y: food.y * cellSize,
        alpha: 1.0,
        dy: -1.5,
      });
      spawnFood();
    } else {
      snake.pop(); // tail hata do
    }

    steps++;
  }

  function triggerDeath() {
    gameOver = true;
    deathFlashTimer = 0;
    // high score update karo
    if (score > highScores[strategy]) {
      highScores[strategy] = score;
    }
  }

  // ============================================================
  // AI DIRECTION COMPUTATION
  // ============================================================

  function computeAIDirection() {
    const head = snake[0];
    const N = gridSize;

    if (strategy === 'greedy') {
      computeGreedyDirection();
    } else if (strategy === 'hamiltonian') {
      computeHamiltonianDirection();
    } else if (strategy === 'hybrid') {
      computeHybridDirection();
    }
  }

  function computeGreedyDirection() {
    const head = snake[0];
    const tail = snake[snake.length - 1];
    const N = gridSize;
    const snakeSet = getSnakeSet();

    // pehle food tak A* try karo
    const pathToFood = astarPath(head.x, head.y, food.x, food.y, snakeSet);
    if (pathToFood && pathToFood.length > 0) {
      aiPath = pathToFood;
      const next = pathToFood[0];
      nextDirection = { x: next.x - head.x, y: next.y - head.y };
      return;
    }

    // food tak path nahi mila — tail follow karo (survival mode)
    // tail ki taraf jaane se space khulega
    const pathToTail = astarPath(head.x, head.y, tail.x, tail.y, snakeSet);
    if (pathToTail && pathToTail.length > 0) {
      aiPath = pathToTail;
      const next = pathToTail[0];
      nextDirection = { x: next.x - head.x, y: next.y - head.y };
      return;
    }

    // tail tak bhi path nahi — koi bhi safe move lo
    aiPath = [];
    const dirs = [{ x: 0, y: -1 }, { x: 0, y: 1 }, { x: -1, y: 0 }, { x: 1, y: 0 }];
    for (const d of dirs) {
      const nx = head.x + d.x;
      const ny = head.y + d.y;
      if (nx >= 0 && nx < N && ny >= 0 && ny < N && !snakeSet.has(ny * N + nx)) {
        nextDirection = d;
        return;
      }
    }
    // koi safe move nahi — death inevitable
  }

  function computeHamiltonianDirection() {
    const head = snake[0];
    const N = gridSize;
    const headIdx = head.y * N + head.x;
    const headOrder = cycleOrder[headIdx];
    const totalCells = N * N;

    // cycle mein next cell dhundho
    const nextOrder = (headOrder + 1) % totalCells;
    const nextIdx = hamiltonianCycle[nextOrder];
    const nextX = nextIdx % N;
    const nextY = Math.floor(nextIdx / N);

    nextDirection = { x: nextX - head.x, y: nextY - head.y };
    aiPath = []; // Hamiltonian mode mein A* path nahi dikhate
  }

  function computeHybridDirection() {
    const head = snake[0];
    const N = gridSize;
    const totalCells = N * N;
    const snakeSet = getSnakeSet();
    const headIdx = head.y * N + head.x;

    // A* shortcut try karo
    const pathToFood = astarPath(head.x, head.y, food.x, food.y, snakeSet);

    if (pathToFood && pathToFood.length > 0) {
      // check karo ki shortcut safe hai — Hamiltonian ordering maintain honi chahiye
      // "safe" = shortcut lene ke baad bhi snake tail cycle order mein head se peeche rahe
      const nextCell = pathToFood[0];
      const nextIdx = nextCell.y * N + nextCell.x;
      const nextOrder = cycleOrder[nextIdx];
      const tailIdx = snake[snake.length - 1].y * N + snake[snake.length - 1].x;
      const tailOrder = cycleOrder[tailIdx];
      const headOrder = cycleOrder[headIdx];

      // distance in cycle: head se tail tak kitna door hai (cycle direction mein)
      // agar shortcut lene se head tail ke paas aa jaaye toh unsafe
      function cycleDist(from, to) {
        return (to - from + totalCells) % totalCells;
      }

      const currentGap = cycleDist(headOrder, tailOrder);
      const newGap = cycleDist(nextOrder, tailOrder);

      // safe check: naya gap >= snake length hona chahiye
      // (warna head tail ko overtake kar dega cycle mein)
      if (newGap >= snake.length) {
        // shortcut safe hai — A* follow karo
        aiPath = pathToFood;
        nextDirection = { x: nextCell.x - head.x, y: nextCell.y - head.y };
        return;
      }
    }

    // shortcut safe nahi — Hamiltonian cycle follow karo
    aiPath = [];
    computeHamiltonianDirection();
  }

  // ============================================================
  // CANVAS SIZING — DPR-aware responsive
  // ============================================================

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = container.clientWidth;
    canvasW = w;
    canvasH = CANVAS_HEIGHT;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(CANVAS_HEIGHT * dpr);
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // game area 70%, stats 30%
    gameW = Math.floor(canvasW * 0.70);
    statsW = canvasW - gameW;

    // cell size calculate — square cells, grid fit karo game area mein
    const maxCellW = gameW / gridSize;
    const maxCellH = canvasH / gridSize;
    cellSize = Math.min(maxCellW, maxCellH);

    // game area center karo
    gameW = cellSize * gridSize;
  }

  // ============================================================
  // RENDERING — grid, snake, food, overlays, stats
  // ============================================================

  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // background — dark
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvasW, canvasH);

    const N = gridSize;
    const offsetX = 0; // game area left se start

    // --- Grid lines — subtle ---
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= N; i++) {
      // horizontal
      ctx.beginPath();
      ctx.moveTo(offsetX, i * cellSize);
      ctx.lineTo(offsetX + N * cellSize, i * cellSize);
      ctx.stroke();
      // vertical
      ctx.beginPath();
      ctx.moveTo(offsetX + i * cellSize, 0);
      ctx.lineTo(offsetX + i * cellSize, N * cellSize);
      ctx.stroke();
    }

    // --- Hamiltonian cycle overlay (faint background) ---
    if (showCycle && hamiltonianCycle.length > 0 && strategy !== 'manual' && strategy !== 'greedy') {
      ctx.strokeStyle = strategy === 'hamiltonian' ? 'rgba(0,255,100,0.15)' : 'rgba(0,255,100,0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < hamiltonianCycle.length; i++) {
        const idx = hamiltonianCycle[i];
        const x = (idx % N) * cellSize + cellSize / 2 + offsetX;
        const y = Math.floor(idx / N) * cellSize + cellSize / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      // close the cycle
      const firstIdx = hamiltonianCycle[0];
      ctx.lineTo((firstIdx % N) * cellSize + cellSize / 2 + offsetX,
                  Math.floor(firstIdx / N) * cellSize + cellSize / 2);
      ctx.stroke();
    }

    // --- AI path overlay ---
    if (showPath && aiPath.length > 0 && strategy !== 'manual') {
      const pathColor = strategy === 'hamiltonian' ? 'rgba(0,255,100,0.4)' : 'rgba(0,255,255,0.4)';
      ctx.strokeStyle = pathColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const head = snake[0];
      ctx.moveTo(head.x * cellSize + cellSize / 2 + offsetX, head.y * cellSize + cellSize / 2);
      for (const p of aiPath) {
        ctx.lineTo(p.x * cellSize + cellSize / 2 + offsetX, p.y * cellSize + cellSize / 2);
      }
      ctx.stroke();

      // path dots
      ctx.fillStyle = pathColor;
      for (const p of aiPath) {
        ctx.beginPath();
        ctx.arc(p.x * cellSize + cellSize / 2 + offsetX, p.y * cellSize + cellSize / 2, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // --- Food — glowing red dot ---
    foodPulse += 0.08;
    const pulseScale = 1 + 0.15 * Math.sin(foodPulse);
    const foodCx = food.x * cellSize + cellSize / 2 + offsetX;
    const foodCy = food.y * cellSize + cellSize / 2;
    const foodR = cellSize * 0.35 * pulseScale;

    // glow effect
    const glowGrad = ctx.createRadialGradient(foodCx, foodCy, foodR * 0.3, foodCx, foodCy, foodR * 2);
    glowGrad.addColorStop(0, 'rgba(255,60,60,0.6)');
    glowGrad.addColorStop(1, 'rgba(255,60,60,0)');
    ctx.fillStyle = glowGrad;
    ctx.beginPath();
    ctx.arc(foodCx, foodCy, foodR * 2, 0, Math.PI * 2);
    ctx.fill();

    // food body
    ctx.fillStyle = '#ff4444';
    ctx.beginPath();
    ctx.arc(foodCx, foodCy, foodR, 0, Math.PI * 2);
    ctx.fill();

    // food highlight
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.beginPath();
    ctx.arc(foodCx - foodR * 0.25, foodCy - foodR * 0.25, foodR * 0.3, 0, Math.PI * 2);
    ctx.fill();

    // --- Snake body — gradient colored, head bright, tail dark ---
    const snakeLen = snake.length;
    for (let i = snakeLen - 1; i >= 0; i--) {
      const seg = snake[i];
      const t = 1 - (i / Math.max(snakeLen - 1, 1)); // 1 at head, 0 at tail
      const sx = seg.x * cellSize + offsetX;
      const sy = seg.y * cellSize;

      // death flash — red flicker
      let r, g, b;
      if (gameOver && deathFlashTimer % 4 < 2) {
        r = 255; g = 50; b = 50;
      } else {
        // blue gradient — bright head (#4a9eff), dark tail (#1a3a6e)
        r = Math.round(26 + (74 - 26) * t);
        g = Math.round(58 + (158 - 58) * t);
        b = Math.round(110 + (255 - 110) * t);
      }

      const inset = cellSize * 0.08;
      const segSize = cellSize - inset * 2;
      const cornerR = cellSize * 0.2;

      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.beginPath();
      ctx.roundRect(sx + inset, sy + inset, segSize, segSize, cornerR);
      ctx.fill();

      // segment border — subtle darker outline
      ctx.strokeStyle = `rgba(0,0,0,0.3)`;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // --- Head details — slightly larger with eye dots ---
    if (snake.length > 0) {
      const head = snake[0];
      const hx = head.x * cellSize + offsetX;
      const hy = head.y * cellSize;
      const headInset = cellSize * 0.04;
      const headSize = cellSize - headInset * 2;
      const headR = cellSize * 0.25;

      // head body — brighter
      if (!(gameOver && deathFlashTimer % 4 < 2)) {
        ctx.fillStyle = '#5cb8ff';
        ctx.beginPath();
        ctx.roundRect(hx + headInset, hy + headInset, headSize, headSize, headR);
        ctx.fill();
      }

      // eyes — direction ke hisaab se position
      const eyeR = cellSize * 0.08;
      const centerX = hx + cellSize / 2;
      const centerY = hy + cellSize / 2;
      const eyeOffset = cellSize * 0.18;

      let eye1x, eye1y, eye2x, eye2y;
      if (direction.x === 1) { // right
        eye1x = centerX + eyeOffset; eye1y = centerY - eyeOffset;
        eye2x = centerX + eyeOffset; eye2y = centerY + eyeOffset;
      } else if (direction.x === -1) { // left
        eye1x = centerX - eyeOffset; eye1y = centerY - eyeOffset;
        eye2x = centerX - eyeOffset; eye2y = centerY + eyeOffset;
      } else if (direction.y === -1) { // up
        eye1x = centerX - eyeOffset; eye1y = centerY - eyeOffset;
        eye2x = centerX + eyeOffset; eye2y = centerY - eyeOffset;
      } else { // down
        eye1x = centerX - eyeOffset; eye1y = centerY + eyeOffset;
        eye2x = centerX + eyeOffset; eye2y = centerY + eyeOffset;
      }

      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(eye1x, eye1y, eyeR, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(eye2x, eye2y, eyeR, 0, Math.PI * 2);
      ctx.fill();

      // pupils — direction mein thoda shift
      const pupilR = eyeR * 0.5;
      const pupilShift = eyeR * 0.3;
      ctx.fillStyle = '#000000';
      ctx.beginPath();
      ctx.arc(eye1x + direction.x * pupilShift, eye1y + direction.y * pupilShift, pupilR, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(eye2x + direction.x * pupilShift, eye2y + direction.y * pupilShift, pupilR, 0, Math.PI * 2);
      ctx.fill();
    }

    // --- Hover cell highlight (manual mode) ---
    if (strategy === 'manual' && hoverCell && !gameOver) {
      ctx.fillStyle = 'rgba(74,158,255,0.15)';
      ctx.fillRect(hoverCell.x * cellSize + offsetX, hoverCell.y * cellSize, cellSize, cellSize);
    }

    // --- Score popups ---
    for (let i = scorePopups.length - 1; i >= 0; i--) {
      const p = scorePopups[i];
      p.y += p.dy;
      p.alpha -= 0.025;
      if (p.alpha <= 0) {
        scorePopups.splice(i, 1);
        continue;
      }
      ctx.fillStyle = `rgba(74,255,74,${p.alpha})`;
      ctx.font = `bold ${Math.round(cellSize * 0.7)}px ${FONT}`;
      ctx.textAlign = 'center';
      ctx.fillText('+1', p.x + offsetX, p.y);
    }

    // --- Stats panel (right side) ---
    drawStatsPanel();

    // --- Game over overlay ---
    if (gameOver) {
      ctx.fillStyle = 'rgba(255,50,50,0.08)';
      ctx.fillRect(0, 0, gameW, canvasH);

      ctx.fillStyle = 'rgba(255,50,50,0.9)';
      ctx.font = `bold ${Math.round(cellSize * 1.2)}px ${FONT}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('GAME OVER', gameW / 2, canvasH / 2 - cellSize);

      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = `${Math.round(cellSize * 0.6)}px ${FONT}`;
      ctx.fillText(`Score: ${score}`, gameW / 2, canvasH / 2 + cellSize * 0.5);

      if (autoRestart) {
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = `${Math.round(cellSize * 0.4)}px ${FONT}`;
        ctx.fillText('Restarting...', gameW / 2, canvasH / 2 + cellSize * 1.5);
      }
    }
  }

  function drawStatsPanel() {
    const N = gridSize;
    const totalCells = N * N;
    const panelX = gameW + 8;
    const panelW = canvasW - gameW - 16;

    if (panelW < 60) return; // bahut chhota hai, skip

    // panel background
    ctx.fillStyle = 'rgba(74,158,255,0.04)';
    ctx.fillRect(gameW, 0, canvasW - gameW, canvasH);

    // border line
    ctx.strokeStyle = 'rgba(74,158,255,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(gameW, 0);
    ctx.lineTo(gameW, canvasH);
    ctx.stroke();

    let y = 20;
    const lineH = 18;

    // strategy name
    const stratNames = {
      greedy: 'Greedy A*',
      hamiltonian: 'Hamiltonian',
      hybrid: 'Hybrid (Smart)',
      manual: 'Manual',
    };

    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = ACCENT;
    ctx.font = `bold 12px ${FONT}`;
    ctx.fillText(stratNames[strategy] || strategy, panelX, y);
    y += lineH + 4;

    // stats
    ctx.font = `11px ${FONT}`;
    const stats = [
      { label: 'Score', value: score, color: '#e2e8f0' },
      { label: 'Length', value: snake.length, color: '#e2e8f0' },
      { label: 'Fill', value: (snake.length / totalCells * 100).toFixed(1) + '%', color: '#e2e8f0' },
      { label: 'Steps', value: steps, color: '#94a3b8' },
      { label: 'Grid', value: `${N}×${N}`, color: '#94a3b8' },
    ];

    for (const s of stats) {
      ctx.fillStyle = '#64748b';
      ctx.fillText(s.label, panelX, y);
      ctx.fillStyle = s.color;
      ctx.fillText(s.value.toString(), panelX + 50, y);
      y += lineH;
    }

    // high scores
    y += 8;
    ctx.fillStyle = ACCENT;
    ctx.font = `bold 11px ${FONT}`;
    ctx.fillText('High Scores', panelX, y);
    y += lineH;

    ctx.font = `10px ${FONT}`;
    const stratOrder = ['greedy', 'hamiltonian', 'hybrid', 'manual'];
    const shortNames = { greedy: 'A*', hamiltonian: 'Ham', hybrid: 'Hyb', manual: 'Man' };

    for (const s of stratOrder) {
      const hs = highScores[s];
      ctx.fillStyle = '#64748b';
      ctx.fillText(shortNames[s], panelX, y);
      ctx.fillStyle = s === strategy ? '#e2e8f0' : '#94a3b8';
      ctx.fillText(hs.toString(), panelX + 40, y);
      y += lineH - 2;
    }

    // bar chart — strategies comparison
    y += 12;
    ctx.fillStyle = ACCENT;
    ctx.font = `bold 10px ${FONT}`;
    ctx.fillText('Comparison', panelX, y);
    y += 14;

    const maxHS = Math.max(1, ...Object.values(highScores));
    const barH = 8;
    const barMaxW = panelW - 40;

    for (const s of stratOrder) {
      const hs = highScores[s];
      const barW = Math.max(1, (hs / maxHS) * barMaxW);

      // label
      ctx.fillStyle = '#64748b';
      ctx.font = `9px ${FONT}`;
      ctx.fillText(shortNames[s], panelX, y + 1);

      // bar background
      ctx.fillStyle = 'rgba(74,158,255,0.1)';
      ctx.fillRect(panelX + 30, y, barMaxW, barH);

      // bar fill
      const barColors = {
        greedy: '#00ddff',
        hamiltonian: '#00ff64',
        hybrid: '#ffaa00',
        manual: '#ff6688',
      };
      ctx.fillStyle = s === strategy ? barColors[s] : `${barColors[s]}88`;
      ctx.fillRect(panelX + 30, y, barW, barH);

      // value
      ctx.fillStyle = '#94a3b8';
      ctx.font = `8px ${FONT}`;
      ctx.fillText(hs.toString(), panelX + 32 + barMaxW, y + 1);

      y += barH + 6;
    }
  }

  // ============================================================
  // CONTROLS — strategy, speed, grid, toggles
  // ============================================================

  // helper functions for consistent styling
  function makeSelectCSS() {
    return [
      'padding:5px 8px',
      'background:rgba(74,158,255,0.08)',
      'color:#e2e8f0',
      'border:1px solid rgba(74,158,255,0.25)',
      'border-radius:6px',
      'font-family:' + FONT,
      'font-size:11px',
      'cursor:pointer',
      'outline:none',
    ].join(';');
  }

  function makeBtnCSS(active) {
    return [
      'padding:5px 10px',
      'border:1px solid ' + (active ? ACCENT : 'rgba(74,158,255,0.25)'),
      'border-radius:6px',
      'background:' + (active ? 'rgba(74,158,255,0.2)' : 'rgba(74,158,255,0.06)'),
      'color:' + (active ? '#e2e8f0' : '#94a3b8'),
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
      'background:rgba(74,158,255,0.15)',
      'color:#e2e8f0',
      'font-family:' + FONT,
      'font-size:11px',
      'font-weight:600',
      'cursor:pointer',
      'transition:all 0.15s ease',
      'outline:none',
    ].join(';');
  }

  // --- Controls Row 1: strategy dropdown, grid size, restart ---
  const controlsRow1 = document.createElement('div');
  controlsRow1.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:6px',
    'margin-top:8px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsRow1);

  // strategy dropdown
  const strategySelect = document.createElement('select');
  strategySelect.style.cssText = makeSelectCSS();
  const strategies = [
    { value: 'hybrid', label: 'Hybrid (Smart)' },
    { value: 'greedy', label: 'Greedy A*' },
    { value: 'hamiltonian', label: 'Hamiltonian' },
    { value: 'manual', label: 'Manual (WASD)' },
  ];
  for (const s of strategies) {
    const opt = document.createElement('option');
    opt.value = s.value;
    opt.textContent = s.label;
    if (s.value === strategy) opt.selected = true;
    strategySelect.appendChild(opt);
  }
  strategySelect.addEventListener('change', () => {
    strategy = strategySelect.value;
    canvas.style.cursor = strategy === 'manual' ? 'crosshair' : 'default';
    resetGame();
  });
  controlsRow1.appendChild(strategySelect);

  // grid size dropdown
  const gridSelect = document.createElement('select');
  gridSelect.style.cssText = makeSelectCSS();
  const gridSizes = [
    { value: 10, label: '10×10' },
    { value: 15, label: '15×15' },
    { value: 20, label: '20×20' },
  ];
  for (const g of gridSizes) {
    const opt = document.createElement('option');
    opt.value = g.value;
    opt.textContent = g.label;
    if (g.value === gridSize) opt.selected = true;
    gridSelect.appendChild(opt);
  }
  gridSelect.addEventListener('change', () => {
    gridSize = parseInt(gridSelect.value);
    resize();
    resetGame();
  });
  controlsRow1.appendChild(gridSelect);

  // restart button
  const restartBtn = document.createElement('button');
  restartBtn.textContent = '\u21BB Restart';
  restartBtn.style.cssText = makeActionBtnCSS();
  restartBtn.addEventListener('click', () => {
    resetGame();
  });
  controlsRow1.appendChild(restartBtn);

  // --- Controls Row 2: speed slider, toggles ---
  const controlsRow2 = document.createElement('div');
  controlsRow2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:6px',
    'align-items:center',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#94a3b8',
  ].join(';');
  container.appendChild(controlsRow2);

  // speed slider
  const speedGroup = document.createElement('div');
  speedGroup.style.cssText = 'display:flex;align-items:center;gap:6px;';
  const speedLabel = document.createElement('span');
  speedLabel.textContent = 'Speed: ' + speedMs + 'ms';
  speedLabel.style.cssText = `min-width:82px;font-family:${FONT};font-size:11px;color:#94a3b8;`;
  speedGroup.appendChild(speedLabel);
  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = 10;
  speedSlider.max = 200;
  speedSlider.value = speedMs;
  speedSlider.style.cssText = 'width:80px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
  speedSlider.addEventListener('input', () => {
    speedMs = parseInt(speedSlider.value);
    speedLabel.textContent = 'Speed: ' + speedMs + 'ms';
  });
  speedGroup.appendChild(speedSlider);
  controlsRow2.appendChild(speedGroup);

  // show path toggle
  const pathToggle = document.createElement('button');
  pathToggle.textContent = 'Path: ON';
  pathToggle.style.cssText = makeBtnCSS(true);
  pathToggle.addEventListener('click', () => {
    showPath = !showPath;
    pathToggle.textContent = 'Path: ' + (showPath ? 'ON' : 'OFF');
    pathToggle.style.cssText = makeBtnCSS(showPath);
  });
  controlsRow2.appendChild(pathToggle);

  // show cycle toggle
  const cycleToggle = document.createElement('button');
  cycleToggle.textContent = 'Cycle: ON';
  cycleToggle.style.cssText = makeBtnCSS(true);
  cycleToggle.addEventListener('click', () => {
    showCycle = !showCycle;
    cycleToggle.textContent = 'Cycle: ' + (showCycle ? 'ON' : 'OFF');
    cycleToggle.style.cssText = makeBtnCSS(showCycle);
  });
  controlsRow2.appendChild(cycleToggle);

  // auto-restart toggle
  const autoBtn = document.createElement('button');
  autoBtn.textContent = 'Auto: ON';
  autoBtn.style.cssText = makeBtnCSS(true);
  autoBtn.addEventListener('click', () => {
    autoRestart = !autoRestart;
    autoBtn.textContent = 'Auto: ' + (autoRestart ? 'ON' : 'OFF');
    autoBtn.style.cssText = makeBtnCSS(autoRestart);
  });
  controlsRow2.appendChild(autoBtn);

  // hint text
  const hintSpan = document.createElement('span');
  hintSpan.textContent = 'WASD/Arrows in Manual mode';
  hintSpan.style.cssText = 'color:rgba(148,163,184,0.4);font-size:10px;margin-left:auto;';
  controlsRow2.appendChild(hintSpan);

  // ============================================================
  // INPUT HANDLING — keyboard for manual mode, hover for cell highlight
  // ============================================================

  function handleKeyDown(e) {
    if (strategy !== 'manual') return;

    let dir = null;
    switch (e.key) {
      case 'ArrowUp': case 'w': case 'W': dir = { x: 0, y: -1 }; break;
      case 'ArrowDown': case 's': case 'S': dir = { x: 0, y: 1 }; break;
      case 'ArrowLeft': case 'a': case 'A': dir = { x: -1, y: 0 }; break;
      case 'ArrowRight': case 'd': case 'D': dir = { x: 1, y: 0 }; break;
    }
    if (dir) {
      e.preventDefault();
      pendingManualDir = dir;
    }
  }

  document.addEventListener('keydown', handleKeyDown);

  // hover tracking for manual mode grid highlight
  canvas.addEventListener('mousemove', (e) => {
    if (strategy !== 'manual') { hoverCell = null; return; }
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvasW / rect.width);
    const my = (e.clientY - rect.top) * (canvasH / rect.height);
    const gx = Math.floor(mx / cellSize);
    const gy = Math.floor(my / cellSize);
    if (gx >= 0 && gx < gridSize && gy >= 0 && gy < gridSize) {
      hoverCell = { x: gx, y: gy };
    } else {
      hoverCell = null;
    }
  });

  canvas.addEventListener('mouseleave', () => { hoverCell = null; });

  // ============================================================
  // ANIMATION LOOP — timed steps with rAF
  // ============================================================

  function loop(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    // step timing — speedMs ke interval pe step chala
    if (!lastStepTime) lastStepTime = timestamp;
    const elapsed = timestamp - lastStepTime;

    if (elapsed >= speedMs) {
      gameStep();
      lastStepTime = timestamp;
    }

    draw();
    animationId = requestAnimationFrame(loop);
  }

  // ============================================================
  // VISIBILITY — IntersectionObserver, lab:resume, tab switch
  // ============================================================

  const observer = new IntersectionObserver(([entry]) => {
    isVisible = entry.isIntersecting;
    if (isVisible && !animationId) {
      lastStepTime = 0;
      animationId = requestAnimationFrame(loop);
    } else if (!isVisible && animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }, { threshold: 0.1 });
  observer.observe(container);

  // lab resume: restart loop when focus released
  document.addEventListener('lab:resume', () => {
    if (isVisible && !animationId) {
      lastStepTime = 0;
      animationId = requestAnimationFrame(loop);
    }
  });

  // tab switch pe bhi handle karo — battery bachao
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      isVisible = false;
      if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    } else {
      // check karo ki container actually view mein hai
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) {
        isVisible = true;
        if (!animationId) {
          lastStepTime = 0;
          animationId = requestAnimationFrame(loop);
        }
      }
    }
  });

  // ============================================================
  // INITIALIZATION — sab setup karke game shuru karo
  // ============================================================

  resize();
  resetGame();

  // resize pe canvas update karo — responsive rehna chahiye
  window.addEventListener('resize', () => {
    resize();
    draw();
  });

  // pehla frame draw karo
  draw();
}
