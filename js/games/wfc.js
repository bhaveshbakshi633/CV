// ============================================================
// Wave Function Collapse — Simple Tiled Model (Circuit/Pipe)
// Procedural generation visualizer — grid mein har cell collapse
// hoti hai, constraints propagate hote hain, aur pipes/circuits
// emerge hote hain chaos se. Quantum superposition jaisa feel.
// ============================================================

// yahi se sab shuru hota hai — container dhundho, WFC engine banao, magic dekho
export function initWFC() {
  const container = document.getElementById('wfcContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';       // purple accent — portfolio theme
  const PIPE_CYAN = '#22d3ee';    // cyan pipe color
  const PIPE_GREEN = '#10b981';   // green pipe color
  const TILE_BG = '#111122';      // dark tile background
  const DARK_BG = '#1a1a2e';      // main dark background

  // edge connection types — ye WFC ka core hai
  const OPEN = 1;   // connection hai is side pe
  const CLOSED = 0; // koi connection nahi

  // --- Tile Definitions ---
  // har tile ke 4 edges hain: [top, right, bottom, left]
  // OPEN = pipe is side se jaati hai, CLOSED = wall hai
  const TILE_DEFS = [
    { name: 'EMPTY',      edges: [CLOSED, CLOSED, CLOSED, CLOSED], weight: 2 },
    { name: 'STRAIGHT_H', edges: [CLOSED, OPEN,   CLOSED, OPEN  ], weight: 3 },
    { name: 'STRAIGHT_V', edges: [OPEN,   CLOSED, OPEN,   CLOSED], weight: 3 },
    { name: 'CORNER_TR',  edges: [OPEN,   OPEN,   CLOSED, CLOSED], weight: 2 },
    { name: 'CORNER_TL',  edges: [OPEN,   CLOSED, CLOSED, OPEN  ], weight: 2 },
    { name: 'CORNER_BR',  edges: [CLOSED, OPEN,   OPEN,   CLOSED], weight: 2 },
    { name: 'CORNER_BL',  edges: [CLOSED, CLOSED, OPEN,   OPEN  ], weight: 2 },
    { name: 'T_UP',       edges: [OPEN,   OPEN,   CLOSED, OPEN  ], weight: 1 },
    { name: 'T_DOWN',     edges: [CLOSED, OPEN,   OPEN,   OPEN  ], weight: 1 },
    { name: 'T_LEFT',     edges: [OPEN,   CLOSED, OPEN,   OPEN  ], weight: 1 },
    { name: 'T_RIGHT',    edges: [OPEN,   OPEN,   OPEN,   CLOSED], weight: 1 },
    { name: 'CROSS',      edges: [OPEN,   OPEN,   OPEN,   OPEN  ], weight: 0.5 },
    { name: 'END_T',      edges: [OPEN,   CLOSED, CLOSED, CLOSED], weight: 1 },
    { name: 'END_R',      edges: [CLOSED, OPEN,   CLOSED, CLOSED], weight: 1 },
    { name: 'END_B',      edges: [CLOSED, CLOSED, OPEN,   CLOSED], weight: 1 },
    { name: 'END_L',      edges: [CLOSED, CLOSED, CLOSED, OPEN  ], weight: 1 },
  ];

  const NUM_TILES = TILE_DEFS.length;

  // direction offsets — [dr, dc] aur opposite edge index
  // 0=top, 1=right, 2=bottom, 3=left
  const DIR_DR = [-1, 0, 1, 0];
  const DIR_DC = [0, 1, 0, -1];
  const OPPOSITE = [2, 3, 0, 1]; // top ka opposite bottom, etc.

  // --- Precompute compatibility ---
  // compatible[tileA][dir] = set of tiles jo tileA ke us dir mein rakh sakte hain
  const compatible = [];
  for (let a = 0; a < NUM_TILES; a++) {
    compatible[a] = [new Set(), new Set(), new Set(), new Set()];
    for (let b = 0; b < NUM_TILES; b++) {
      for (let d = 0; d < 4; d++) {
        // tileA ke dir d ka edge match hona chahiye tileB ke opposite edge se
        if (TILE_DEFS[a].edges[d] === TILE_DEFS[b].edges[OPPOSITE[d]]) {
          compatible[a][d].add(b);
        }
      }
    }
  }

  // --- Grid Size Presets ---
  const SIZE_PRESETS = {
    'Small':  { cols: 12, rows: 8  },
    'Medium': { cols: 20, rows: 15 },
    'Large':  { cols: 30, rows: 22 },
  };

  // --- State ---
  let gridCols = 20;
  let gridRows = 15;
  let cellSize = 0;       // har cell ka pixel size — dynamic calculate hoga
  let grid = [];           // 2D array of cell objects
  let collapsed = 0;       // kitne cells collapse ho chuke
  let totalCells = 0;
  let running = false;     // auto-run mode on hai ya nahi
  let speed = 5;           // steps per frame
  let isVisible = false;
  let animationId = null;
  let contradiction = false; // kya WFC fail ho gaya
  let flashTimers = {};    // flash animation timers
  let shimmerTime = 0;     // shimmer animation counter

  // --- DOM Setup ---
  const existingChildren = Array.from(container.children);
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;border-radius:8px;cursor:pointer;display:block;';
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls Bar ---
  const controls = document.createElement('div');
  controls.style.cssText = `
    display:flex; flex-wrap:wrap; gap:8px; align-items:center;
    padding:10px 0; font-family:'JetBrains Mono',monospace; font-size:12px;
    color:#ccc;
  `;
  container.appendChild(controls);

  // button factory — consistent dark theme buttons
  function makeBtn(label, onClick) {
    const b = document.createElement('button');
    b.textContent = label;
    b.style.cssText = `
      background:#2a2a4a; color:${ACCENT}; border:1px solid ${ACCENT}44;
      padding:5px 14px; border-radius:6px; cursor:pointer;
      font-family:'JetBrains Mono',monospace; font-size:12px;
      transition: all 0.2s;
    `;
    b.addEventListener('mouseenter', () => {
      b.style.background = '#3a3a5a';
      b.style.borderColor = ACCENT;
    });
    b.addEventListener('mouseleave', () => {
      b.style.background = '#2a2a4a';
      b.style.borderColor = ACCENT + '44';
    });
    b.addEventListener('click', onClick);
    return b;
  }

  // generate button — naye seed se restart
  const generateBtn = makeBtn('Generate', () => {
    initWFCGrid();
    if (!running) drawGrid();
  });
  controls.appendChild(generateBtn);

  // step button — ek step aage
  const stepBtn = makeBtn('Step', () => {
    if (contradiction || collapsed >= totalCells) return;
    performStep();
    drawGrid();
  });
  controls.appendChild(stepBtn);

  // auto/pause toggle
  const autoBtn = makeBtn('Auto', () => {
    running = !running;
    autoBtn.textContent = running ? 'Pause' : 'Auto';
    autoBtn.style.color = running ? PIPE_GREEN : ACCENT;
    autoBtn.style.borderColor = running ? PIPE_GREEN + '44' : ACCENT + '44';
    if (running && isVisible && !animationId) { lastTime = 0; loop(performance.now()); }
  });
  controls.appendChild(autoBtn);

  // speed slider
  const speedLabel = document.createElement('span');
  speedLabel.textContent = 'Speed:';
  speedLabel.style.marginLeft = '8px';
  controls.appendChild(speedLabel);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '20';
  speedSlider.value = String(speed);
  speedSlider.style.cssText = `
    width:80px; accent-color:${ACCENT}; cursor:pointer; vertical-align:middle;
  `;
  speedSlider.addEventListener('input', () => {
    speed = parseInt(speedSlider.value);
    speedVal.textContent = speed;
  });
  controls.appendChild(speedSlider);

  const speedVal = document.createElement('span');
  speedVal.textContent = String(speed);
  speedVal.style.cssText = `min-width:20px; text-align:center; color:${ACCENT};`;
  controls.appendChild(speedVal);

  // grid size selector
  const sizeLabel = document.createElement('span');
  sizeLabel.textContent = 'Grid:';
  sizeLabel.style.marginLeft = '8px';
  controls.appendChild(sizeLabel);

  const sizeSelect = document.createElement('select');
  sizeSelect.style.cssText = `
    background:#2a2a4a; color:${ACCENT}; border:1px solid ${ACCENT}44;
    padding:4px 8px; border-radius:6px; cursor:pointer;
    font-family:'JetBrains Mono',monospace; font-size:12px;
  `;
  Object.keys(SIZE_PRESETS).forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = `${name} ${SIZE_PRESETS[name].cols}x${SIZE_PRESETS[name].rows}`;
    if (name === 'Medium') opt.selected = true;
    sizeSelect.appendChild(opt);
  });
  sizeSelect.addEventListener('change', () => {
    const preset = SIZE_PRESETS[sizeSelect.value];
    gridCols = preset.cols;
    gridRows = preset.rows;
    resize();
    initWFCGrid();
    if (!running) drawGrid();
  });
  controls.appendChild(sizeSelect);

  // status display — progress aur state dikhata hai
  const statusSpan = document.createElement('span');
  statusSpan.style.cssText = `
    margin-left:auto; color:#888; font-size:11px;
  `;
  controls.appendChild(statusSpan);

  // --- Canvas Resize ---
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // cell size calculate kar — canvas mein fit hona chahiye
    const cellW = Math.floor(w / gridCols);
    const cellH = Math.floor(CANVAS_HEIGHT / gridRows);
    cellSize = Math.min(cellW, cellH);
  }

  resize();
  window.addEventListener('resize', () => {
    resize();
    drawGrid();
  });

  // --- WFC Cell ---
  // har cell ka apna state hai — possibilities ka set, collapsed tile, etc.
  function createCell() {
    // initially sab tiles possible hain
    const possible = new Set();
    for (let i = 0; i < NUM_TILES; i++) possible.add(i);
    return {
      possible,       // set of possible tile indices
      collapsed: -1,  // -1 = not collapsed, otherwise tile index
      entropy: NUM_TILES, // cached entropy
    };
  }

  // --- WFC Grid Init ---
  function initWFCGrid() {
    totalCells = gridRows * gridCols;
    collapsed = 0;
    contradiction = false;
    flashTimers = {};

    grid = [];
    for (let r = 0; r < gridRows; r++) {
      grid[r] = [];
      for (let c = 0; c < gridCols; c++) {
        grid[r][c] = createCell();
      }
    }

    updateStatus();
  }

  // --- Weighted random selection — tile choose kar weights ke basis pe
  function weightedRandom(possibleSet) {
    const arr = Array.from(possibleSet);
    let totalWeight = 0;
    for (const idx of arr) totalWeight += TILE_DEFS[idx].weight;

    let r = Math.random() * totalWeight;
    for (const idx of arr) {
      r -= TILE_DEFS[idx].weight;
      if (r <= 0) return idx;
    }
    return arr[arr.length - 1];
  }

  // --- Shannon entropy with noise — tie-breaking ke liye thoda noise add kar
  function calcEntropy(cell) {
    if (cell.collapsed >= 0) return Infinity; // already done
    const n = cell.possible.size;
    if (n <= 0) return -1; // contradiction!
    if (n === 1) return 0; // auto-collapse hoga

    // weighted entropy for better selection
    let sumW = 0, sumWLogW = 0;
    for (const idx of cell.possible) {
      const w = TILE_DEFS[idx].weight;
      sumW += w;
      sumWLogW += w * Math.log(w);
    }
    const entropy = Math.log(sumW) - sumWLogW / sumW;
    // thoda noise add kar taaaki ties randomly break hon
    return entropy + Math.random() * 0.0001;
  }

  // --- Find lowest entropy cell --- sabse kam possibilities wala dhundho
  function findLowestEntropy() {
    let minEntropy = Infinity;
    let bestR = -1, bestC = -1;

    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const cell = grid[r][c];
        if (cell.collapsed >= 0) continue; // already collapsed, skip

        const e = calcEntropy(cell);
        if (e < 0) {
          // contradiction detect ho gaya — koi valid tile nahi bacha
          contradiction = true;
          return null;
        }
        if (e < minEntropy) {
          minEntropy = e;
          bestR = r;
          bestC = c;
        }
      }
    }

    if (bestR < 0) return null; // sab collapse ho chuke
    return { r: bestR, c: bestC };
  }

  // --- Collapse a cell — ek tile choose kar aur fix kar do
  function collapseCell(r, c) {
    const cell = grid[r][c];
    if (cell.possible.size === 0) {
      contradiction = true;
      return false;
    }

    const chosen = weightedRandom(cell.possible);
    cell.possible = new Set([chosen]);
    cell.collapsed = chosen;
    cell.entropy = Infinity;
    collapsed++;

    // flash effect ke liye track kar
    const key = `${r},${c}`;
    flashTimers[key] = 1.0; // 1.0 se 0.0 tak fade hoga

    return true;
  }

  // --- Propagate constraints --- ye WFC ka asli kaam hai
  // queue-based BFS — collapsed cell ke neighbors ke options reduce kar
  function propagate(startR, startC) {
    const queue = [[startR, startC]];
    const visited = new Set();
    visited.add(`${startR},${startC}`);

    while (queue.length > 0) {
      const [r, c] = queue.shift();
      const cell = grid[r][c];

      // har direction mein neighbor check kar
      for (let d = 0; d < 4; d++) {
        const nr = r + DIR_DR[d];
        const nc = c + DIR_DC[d];

        // grid ke bahar hai toh skip
        if (nr < 0 || nr >= gridRows || nc < 0 || nc >= gridCols) continue;

        const neighbor = grid[nr][nc];
        if (neighbor.collapsed >= 0) continue; // already collapsed, skip

        // neighbor ke possible tiles mein se incompatible hata do
        const validForNeighbor = new Set();
        for (const myTile of cell.possible) {
          for (const nTile of compatible[myTile][d]) {
            if (neighbor.possible.has(nTile)) {
              validForNeighbor.add(nTile);
            }
          }
        }

        // kya kuch hata hai?
        if (validForNeighbor.size < neighbor.possible.size) {
          if (validForNeighbor.size === 0) {
            // contradiction! koi valid option nahi bacha neighbor ke paas
            contradiction = true;
            return;
          }

          neighbor.possible = validForNeighbor;
          neighbor.entropy = calcEntropy(neighbor);

          // agar sirf ek option bacha toh auto-collapse
          if (validForNeighbor.size === 1) {
            const only = validForNeighbor.values().next().value;
            neighbor.collapsed = only;
            collapsed++;
            const key = `${nr},${nc}`;
            flashTimers[key] = 0.7; // propagation collapse ka flash thoda kam
          }

          // is neighbor ke neighbors ko bhi check karna padega
          const nkey = `${nr},${nc}`;
          if (!visited.has(nkey)) {
            visited.add(nkey);
            queue.push([nr, nc]);
          }
        }
      }
    }
  }

  // --- One WFC step: find lowest entropy -> collapse -> propagate ---
  function performStep() {
    if (contradiction || collapsed >= totalCells) return false;

    const target = findLowestEntropy();
    if (!target) return false;

    if (!collapseCell(target.r, target.c)) {
      // contradiction pe restart — backtracking ki jagah fresh start
      initWFCGrid();
      return true;
    }

    propagate(target.r, target.c);

    if (contradiction) {
      // contradiction aayi toh restart
      initWFCGrid();
      return true;
    }

    updateStatus();
    return true;
  }

  // --- Status Update ---
  function updateStatus() {
    const pct = totalCells > 0 ? Math.round((collapsed / totalCells) * 100) : 0;
    let stateText = 'Generating...';
    if (contradiction) stateText = 'Contradiction! Restarting...';
    else if (collapsed >= totalCells) stateText = 'Complete!';
    statusSpan.textContent = `${collapsed}/${totalCells} (${pct}%) — ${stateText}`;
  }

  // --- Tile Drawing Functions ---
  // har tile type ke liye drawing function — pipes/circuits as graphics

  // grid offset calculate kar — center mein place karna hai
  function getGridOffset() {
    const w = container.clientWidth;
    const totalW = gridCols * cellSize;
    const totalH = gridRows * cellSize;
    const ox = Math.floor((w - totalW) / 2);
    const oy = Math.floor((CANVAS_HEIGHT - totalH) / 2);
    return { ox, oy };
  }

  // pipe drawing helper — glow effect ke saath line draw kar
  function drawPipe(x, y, cs, fromX, fromY, toX, toY, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, cs * 0.15);
    ctx.lineCap = 'round';
    // glow effect
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(x + fromX * cs, y + fromY * cs);
    ctx.lineTo(x + toX * cs, y + toY * cs);
    ctx.stroke();
    ctx.restore();
  }

  // arc drawing helper — corners ke liye curved pipe
  function drawArc(x, y, cs, centerX, centerY, radius, startAngle, endAngle, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, cs * 0.15);
    ctx.lineCap = 'round';
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(x + centerX * cs, y + centerY * cs, radius * cs, startAngle, endAngle);
    ctx.stroke();
    ctx.restore();
  }

  // individual tile draw karo — tile type ke basis pe
  function drawTile(x, y, cs, tileIdx, alpha) {
    const name = TILE_DEFS[tileIdx].name;
    // pipe color — random-ish based on position for variety
    const color = (x + y) % 3 === 0 ? PIPE_GREEN : PIPE_CYAN;

    ctx.globalAlpha = alpha;

    switch (name) {
      case 'EMPTY':
        // kuch nahi — sirf dark background (already drawn)
        break;

      case 'STRAIGHT_H':
        // horizontal line — left se right center mein
        drawPipe(x, y, cs, 0, 0.5, 1, 0.5, color);
        break;

      case 'STRAIGHT_V':
        // vertical line — top se bottom center mein
        drawPipe(x, y, cs, 0.5, 0, 0.5, 1, color);
        break;

      case 'CORNER_TR':
        // top se right — arc use kar
        drawArc(x, y, cs, 1, 0, 0.5, Math.PI * 0.5, Math.PI, color);
        break;

      case 'CORNER_TL':
        // top se left — arc
        drawArc(x, y, cs, 0, 0, 0.5, 0, Math.PI * 0.5, color);
        break;

      case 'CORNER_BR':
        // bottom se right — arc
        drawArc(x, y, cs, 1, 1, 0.5, Math.PI, Math.PI * 1.5, color);
        break;

      case 'CORNER_BL':
        // bottom se left — arc
        drawArc(x, y, cs, 0, 1, 0.5, Math.PI * 1.5, Math.PI * 2, color);
        break;

      case 'T_UP':
        // T-junction: top, left, right connected
        drawPipe(x, y, cs, 0, 0.5, 1, 0.5, color);   // horizontal
        drawPipe(x, y, cs, 0.5, 0, 0.5, 0.5, color);  // top half vertical
        break;

      case 'T_DOWN':
        // T-junction: bottom, left, right connected
        drawPipe(x, y, cs, 0, 0.5, 1, 0.5, color);   // horizontal
        drawPipe(x, y, cs, 0.5, 0.5, 0.5, 1, color);  // bottom half vertical
        break;

      case 'T_LEFT':
        // T-junction: top, bottom, left connected
        drawPipe(x, y, cs, 0.5, 0, 0.5, 1, color);   // vertical
        drawPipe(x, y, cs, 0, 0.5, 0.5, 0.5, color);  // left half horizontal
        break;

      case 'T_RIGHT':
        // T-junction: top, bottom, right connected
        drawPipe(x, y, cs, 0.5, 0, 0.5, 1, color);    // vertical
        drawPipe(x, y, cs, 0.5, 0.5, 1, 0.5, color);  // right half horizontal
        break;

      case 'CROSS':
        // sab directions connected — full cross
        drawPipe(x, y, cs, 0, 0.5, 1, 0.5, color);   // horizontal
        drawPipe(x, y, cs, 0.5, 0, 0.5, 1, color);   // vertical
        break;

      case 'END_T':
        // dead end — sirf top open, center mein dot
        drawPipe(x, y, cs, 0.5, 0, 0.5, 0.5, color);
        drawEndDot(x, y, cs, 0.5, 0.5, color);
        break;

      case 'END_R':
        // dead end — sirf right open
        drawPipe(x, y, cs, 0.5, 0.5, 1, 0.5, color);
        drawEndDot(x, y, cs, 0.5, 0.5, color);
        break;

      case 'END_B':
        // dead end — sirf bottom open
        drawPipe(x, y, cs, 0.5, 0.5, 0.5, 1, color);
        drawEndDot(x, y, cs, 0.5, 0.5, color);
        break;

      case 'END_L':
        // dead end — sirf left open
        drawPipe(x, y, cs, 0, 0.5, 0.5, 0.5, color);
        drawEndDot(x, y, cs, 0.5, 0.5, color);
        break;
    }

    ctx.globalAlpha = 1;
  }

  // dead end ka center dot — circuit node jaisa
  function drawEndDot(x, y, cs, cx, cy, color) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.arc(x + cx * cs, y + cy * cs, cs * 0.08, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // --- Superposition Shimmer Effect ---
  // uncollapsed cells ke liye — all possible tiles ka ghost overlay
  function drawSuperposition(x, y, cs, cell) {
    const n = cell.possible.size;
    if (n === 0) return;

    // shimmer background — dark mein subtle moving gradient
    const t = shimmerTime * 0.003;
    const shimmerAlpha = 0.03 + 0.02 * Math.sin(t + x * 0.1 + y * 0.1);

    ctx.save();

    // shimmering base — purple tint for uncertainty
    const grd = ctx.createRadialGradient(
      x + cs * 0.5 + Math.sin(t + y * 0.05) * cs * 0.2,
      y + cs * 0.5 + Math.cos(t + x * 0.05) * cs * 0.2,
      0,
      x + cs * 0.5, y + cs * 0.5, cs * 0.7
    );
    grd.addColorStop(0, `rgba(167, 139, 250, ${shimmerAlpha * 2})`);
    grd.addColorStop(1, `rgba(34, 211, 238, ${shimmerAlpha})`);
    ctx.fillStyle = grd;
    ctx.fillRect(x, y, cs, cs);

    // ghost tiles draw kar — semi-transparent overlay of possibilities
    // zyada tiles possible hain toh zyada faded
    const ghostAlpha = Math.max(0.04, 0.15 / Math.sqrt(n));
    for (const tIdx of cell.possible) {
      if (TILE_DEFS[tIdx].name === 'EMPTY') continue; // empty tiles ka ghost mat dikha
      drawTile(x, y, cs, tIdx, ghostAlpha);
    }

    // entropy indicator — kam options = brighter border
    const entropyRatio = n / NUM_TILES;
    const borderAlpha = 0.1 + (1 - entropyRatio) * 0.3;
    ctx.strokeStyle = `rgba(167, 139, 250, ${borderAlpha})`;
    ctx.lineWidth = 0.5;
    ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);

    // possibility count dikhao agar cell size kaafi bada ho
    if (cs > 20 && n < NUM_TILES) {
      ctx.fillStyle = `rgba(167, 139, 250, ${0.2 + (1 - entropyRatio) * 0.4})`;
      ctx.font = `${Math.max(8, cs * 0.3)}px 'JetBrains Mono', monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(n), x + cs * 0.5, y + cs * 0.5);
    }

    ctx.restore();
  }

  // --- Flash Effect --- abhi collapse hua cell bright flash kare
  function drawFlash(x, y, cs, intensity) {
    ctx.save();
    ctx.fillStyle = `rgba(167, 139, 250, ${intensity * 0.4})`;
    ctx.shadowColor = ACCENT;
    ctx.shadowBlur = 15 * intensity;
    ctx.fillRect(x - 1, y - 1, cs + 2, cs + 2);
    ctx.restore();
  }

  // --- Main Draw ---
  function drawGrid() {
    const w = container.clientWidth;
    ctx.clearRect(0, 0, w, CANVAS_HEIGHT);

    // dark background
    ctx.fillStyle = DARK_BG;
    ctx.fillRect(0, 0, w, CANVAS_HEIGHT);

    const { ox, oy } = getGridOffset();

    // har cell draw kar
    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const cell = grid[r][c];
        const x = ox + c * cellSize;
        const y = oy + r * cellSize;

        // tile background
        ctx.fillStyle = TILE_BG;
        ctx.fillRect(x, y, cellSize, cellSize);

        // subtle grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.03)';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, cellSize, cellSize);

        if (cell.collapsed >= 0) {
          // collapsed cell — final tile draw kar
          drawTile(x, y, cellSize, cell.collapsed, 1.0);

          // flash effect agar abhi collapse hua
          const key = `${r},${c}`;
          if (flashTimers[key] && flashTimers[key] > 0) {
            drawFlash(x, y, cellSize, flashTimers[key]);
          }
        } else {
          // uncollapsed — superposition shimmer effect
          drawSuperposition(x, y, cellSize, cell);
        }
      }
    }

    // contradiction indicator — red flash agar fail hua
    if (contradiction) {
      ctx.fillStyle = 'rgba(255, 50, 50, 0.1)';
      ctx.fillRect(0, 0, w, CANVAS_HEIGHT);
      ctx.fillStyle = 'rgba(255, 100, 100, 0.8)';
      ctx.font = "bold 16px 'JetBrains Mono', monospace";
      ctx.textAlign = 'center';
      ctx.fillText('Contradiction! Restarting...', w / 2, CANVAS_HEIGHT / 2);
    }

    // completion indicator
    if (!contradiction && collapsed >= totalCells) {
      ctx.fillStyle = `rgba(16, 185, 129, 0.06)`;
      ctx.fillRect(0, 0, w, CANVAS_HEIGHT);
    }
  }

  // --- Flash timers decay --- har frame pe flash intensity kam kar
  function updateFlashes(dt) {
    const decayRate = 3.0; // per second
    const keys = Object.keys(flashTimers);
    for (const key of keys) {
      flashTimers[key] -= decayRate * dt;
      if (flashTimers[key] <= 0) {
        delete flashTimers[key];
      }
    }
  }

  // --- Click handler — step mode pe click se ek step
  canvas.addEventListener('click', () => {
    if (!running) {
      if (contradiction || collapsed >= totalCells) {
        initWFCGrid();
      } else {
        performStep();
      }
      drawGrid();
    }
  });

  // --- Animation Loop ---
  let lastTime = 0;

  function loop(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    if (!lastTime) lastTime = timestamp;
    const dt = Math.min(0.05, (timestamp - lastTime) / 1000); // delta time in seconds
    lastTime = timestamp;

    shimmerTime = timestamp;

    // flash animations update kar
    updateFlashes(dt);

    // auto-run mode mein steps perform kar
    if (running && !contradiction && collapsed < totalCells) {
      for (let i = 0; i < speed; i++) {
        if (!performStep()) break;
      }
    }

    // auto-restart agar complete ho gaya aur running hai
    if (running && collapsed >= totalCells && !contradiction) {
      // thoda wait kar fir restart — completed grid ko 2 sec dikha
      if (!loop._completeTimer) {
        loop._completeTimer = timestamp;
      } else if (timestamp - loop._completeTimer > 2000) {
        loop._completeTimer = null;
        initWFCGrid();
      }
    }

    // contradiction pe bhi auto-restart
    if (running && contradiction) {
      initWFCGrid();
    }

    drawGrid();

    animationId = requestAnimationFrame(loop);
  }

  // --- Intersection Observer — sirf visible hone pe animate kar
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) {
      lastTime = 0;
      loop(performance.now());
    } else if (!isVisible && animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }, { threshold: 0.1 });
  obs.observe(container);
  // lab resume: restart loop when focus released
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });

  // --- Init ---
  initWFCGrid();
  drawGrid();
}
