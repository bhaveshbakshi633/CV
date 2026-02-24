// ============================================================
// Q-Learning — Gridworld Cliff-Walking
// Agent seekhta hai optimal path, Q-values update hoti hain,
// policy arrows dikhte hain — classic RL demo hai ye
// ============================================================

export function initQLearning() {
  const container = document.getElementById('qLearningContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  let animationId = null, isVisible = false, canvasW = 0;

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(74,158,255,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  function mkSlider(parent, label, id, min, max, val, step) {
    const lbl = document.createElement('label');
    lbl.style.cssText = "color:#ccc;font:12px 'JetBrains Mono',monospace";
    lbl.textContent = label + ' ';
    const inp = document.createElement('input');
    inp.type = 'range'; inp.min = min; inp.max = max; inp.value = val; inp.id = id;
    if (step) inp.step = step;
    inp.style.cssText = 'width:80px;vertical-align:middle';
    lbl.appendChild(inp);
    parent.appendChild(lbl);
    return inp;
  }
  function mkBtn(parent, text, id) {
    const b = document.createElement('button');
    b.textContent = text; b.id = id;
    b.style.cssText = "background:#333;color:#ccc;border:1px solid #555;padding:3px 8px;border-radius:4px;cursor:pointer;font:11px 'JetBrains Mono',monospace";
    parent.appendChild(b);
    return b;
  }

  // controls
  const lrSlider = mkSlider(ctrl, 'α:', 'ql-lr', 0.01, 1.0, 0.2, 0.01);
  const gammaSlider = mkSlider(ctrl, 'γ:', 'ql-gamma', 0.5, 0.99, 0.95, 0.01);
  const epsSlider = mkSlider(ctrl, 'ε:', 'ql-eps', 0.01, 0.5, 0.15, 0.01);
  const speedSlider = mkSlider(ctrl, 'Speed:', 'ql-speed', 1, 50, 10, 1);
  const btnTrain = mkBtn(ctrl, 'Train ▶', 'ql-train');
  const btnReset = mkBtn(ctrl, 'Reset', 'ql-reset');
  const btnShowPath = mkBtn(ctrl, 'Show Path', 'ql-path');
  const infoLbl = document.createElement('span');
  infoLbl.style.cssText = "color:#888;font:11px 'JetBrains Mono',monospace;margin-left:8px";
  ctrl.appendChild(infoLbl);

  // --- Grid setup: 10 columns x 8 rows ---
  const COLS = 10, ROWS = 8;
  // cell types: 0=empty, 1=wall, 2=cliff(death), 3=goal, 4=start
  const ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // up, down, left, right (dRow, dCol)
  const ACTION_NAMES = ['↑', '↓', '←', '→'];
  let grid = [];
  let Q = [];  // Q[row][col][action]
  let training = false;
  let episodes = 0;
  let agentR = 0, agentC = 0;  // agent position
  let startR = 7, startC = 0;
  let goalR = 7, goalC = 9;
  let showingPath = false;
  let pathCells = [];

  // cliff-walking layout banao
  function initGrid() {
    grid = [];
    for (let r = 0; r < ROWS; r++) {
      grid[r] = [];
      for (let c = 0; c < COLS; c++) grid[r][c] = 0;
    }
    // cliff bottom row mein (start aur goal ke beech)
    for (let c = 1; c < COLS - 1; c++) grid[7][c] = 2;
    // kuch walls bhi daal do beech mein
    grid[3][3] = 1; grid[3][4] = 1; grid[3][5] = 1;
    grid[5][6] = 1; grid[5][7] = 1;
    grid[1][1] = 1; grid[1][2] = 1;
    // start aur goal
    grid[startR][startC] = 4;
    grid[goalR][goalC] = 3;
  }

  // Q-table initialize — sab zero
  function initQ() {
    Q = [];
    for (let r = 0; r < ROWS; r++) {
      Q[r] = [];
      for (let c = 0; c < COLS; c++) {
        Q[r][c] = [0, 0, 0, 0]; // up, down, left, right
      }
    }
  }

  // valid move check — grid ke andar hai aur wall nahi hai
  function isValid(r, c) {
    return r >= 0 && r < ROWS && c >= 0 && c < COLS && grid[r][c] !== 1;
  }

  // reward function
  function getReward(r, c) {
    if (grid[r][c] === 3) return 100;    // goal reached — party!
    if (grid[r][c] === 2) return -100;   // cliff — seedha neeche gira
    return -1;                             // har step ka chhota penalty
  }

  // ε-greedy action select karo
  function chooseAction(r, c) {
    const eps = parseFloat(epsSlider.value);
    if (Math.random() < eps) {
      return Math.floor(Math.random() * 4);
    }
    // greedy — sabse bada Q value wala action
    let best = 0, bestVal = Q[r][c][0];
    for (let a = 1; a < 4; a++) {
      if (Q[r][c][a] > bestVal) { bestVal = Q[r][c][a]; best = a; }
    }
    return best;
  }

  // ek complete episode chala do
  function runEpisode() {
    const alpha = parseFloat(lrSlider.value);
    const gamma = parseFloat(gammaSlider.value);
    let r = startR, c = startC;
    let steps = 0;
    const maxSteps = 200;

    while (steps < maxSteps) {
      steps++;
      const a = chooseAction(r, c);
      let nr = r + ACTIONS[a][0];
      let nc = c + ACTIONS[a][1];

      // agar invalid move hai toh wahi reh
      if (!isValid(nr, nc)) { nr = r; nc = c; }

      const reward = getReward(nr, nc);

      // Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
      const maxNextQ = Math.max(...Q[nr][nc]);
      Q[r][c][a] += alpha * (reward + gamma * maxNextQ - Q[r][c][a]);

      // cliff ya goal pe episode khatam
      if (grid[nr][nc] === 2 || grid[nr][nc] === 3) break;

      r = nr;
      c = nc;
    }
    episodes++;
  }

  // greedy policy follow karke path nikalo
  function computePath() {
    pathCells = [];
    let r = startR, c = startC;
    const visited = new Set();
    let steps = 0;
    while (steps < 100) {
      pathCells.push({ r, c });
      const key = r * COLS + c;
      if (visited.has(key)) break; // loop detect karo
      visited.add(key);
      if (grid[r][c] === 3) break;

      // greedy action
      let best = 0, bestVal = Q[r][c][0];
      for (let a = 1; a < 4; a++) {
        if (Q[r][c][a] > bestVal) { bestVal = Q[r][c][a]; best = a; }
      }
      const nr = r + ACTIONS[best][0];
      const nc = c + ACTIONS[best][1];
      if (!isValid(nr, nc)) break;
      r = nr; c = nc;
      steps++;
    }
    return pathCells;
  }

  // --- Render ---
  function render() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    const cellW = canvasW / COLS;
    const cellH = CANVAS_HEIGHT / ROWS;

    // Q-value range nikalo for color mapping
    let minQ = 0, maxQ = 0;
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const maxA = Math.max(...Q[r][c]);
        if (maxA > maxQ) maxQ = maxA;
        if (maxA < minQ) minQ = maxA;
      }
    }
    const qRange = Math.max(maxQ - minQ, 1);

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const x = c * cellW, y = r * cellH;

        // cell background — value heatmap ya special type
        if (grid[r][c] === 1) {
          // wall — dark gray
          ctx.fillStyle = '#333';
        } else if (grid[r][c] === 2) {
          // cliff — red zone
          ctx.fillStyle = 'rgba(200,30,30,0.4)';
        } else if (grid[r][c] === 3) {
          // goal — green
          ctx.fillStyle = 'rgba(30,200,60,0.4)';
        } else if (grid[r][c] === 4) {
          // start
          ctx.fillStyle = 'rgba(74,158,255,0.2)';
        } else {
          // normal cell — Q-value heatmap
          const maxA = Math.max(...Q[r][c]);
          const norm = (maxA - minQ) / qRange; // 0 to 1
          if (maxA >= 0) {
            ctx.fillStyle = `rgba(30,200,60,${norm * 0.3})`;
          } else {
            ctx.fillStyle = `rgba(200,30,30,${(1 - norm) * 0.3})`;
          }
        }
        ctx.fillRect(x, y, cellW, cellH);

        // grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellW, cellH);

        // policy arrows — har valid cell mein
        if (grid[r][c] === 0 || grid[r][c] === 4) {
          const qVals = Q[r][c];
          const maxVal = Math.max(...qVals);
          const sumAbs = qVals.reduce((s, v) => s + Math.abs(v), 0);
          if (sumAbs > 0.01) {
            const cx = x + cellW / 2, cy = y + cellH / 2;
            // har action ka arrow — size proportional to Q value
            for (let a = 0; a < 4; a++) {
              const strength = Math.abs(qVals[a]) / sumAbs;
              if (strength < 0.05) continue;
              const len = strength * Math.min(cellW, cellH) * 0.4;
              const dx = ACTIONS[a][1], dy = ACTIONS[a][0]; // col=x, row=y
              const isMax = qVals[a] === maxVal;
              ctx.strokeStyle = isMax ? ACCENT : 'rgba(200,200,200,0.3)';
              ctx.lineWidth = isMax ? 2 : 1;
              ctx.beginPath();
              ctx.moveTo(cx, cy);
              ctx.lineTo(cx + dx * len, cy + dy * len);
              ctx.stroke();
              // arrow head for best action
              if (isMax) {
                const ax = cx + dx * len, ay = cy + dy * len;
                const angle = Math.atan2(dy, dx);
                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(ax - 6 * Math.cos(angle - 0.5), ay - 6 * Math.sin(angle - 0.5));
                ctx.lineTo(ax - 6 * Math.cos(angle + 0.5), ay - 6 * Math.sin(angle + 0.5));
                ctx.closePath();
                ctx.fillStyle = ACCENT;
                ctx.fill();
              }
            }
          }
        }

        // special cell labels
        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        ctx.textAlign = 'center';
        if (grid[r][c] === 3) {
          ctx.fillStyle = '#4ade80';
          ctx.fillText('GOAL', x + cellW / 2, y + cellH / 2 + 4);
        } else if (grid[r][c] === 4) {
          ctx.fillStyle = ACCENT;
          ctx.fillText('START', x + cellW / 2, y + cellH / 2 + 4);
        } else if (grid[r][c] === 2) {
          ctx.fillStyle = '#ef4444';
          ctx.fillText('☠', x + cellW / 2, y + cellH / 2 + 5);
        } else if (grid[r][c] === 1) {
          ctx.fillStyle = '#666';
          ctx.fillText('▓', x + cellW / 2, y + cellH / 2 + 4);
        }
      }
    }

    // greedy path dikha do agar show path on hai
    if (showingPath && pathCells.length > 1) {
      ctx.strokeStyle = '#ff8c42';
      ctx.lineWidth = 3;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pathCells[0].c * cellW + cellW / 2, pathCells[0].r * cellH + cellH / 2);
      for (let i = 1; i < pathCells.length; i++) {
        ctx.lineTo(pathCells[i].c * cellW + cellW / 2, pathCells[i].r * cellH + cellH / 2);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // agent dot at end of path
      const last = pathCells[pathCells.length - 1];
      ctx.fillStyle = '#ff8c42';
      ctx.beginPath();
      ctx.arc(last.c * cellW + cellW / 2, last.r * cellH + cellH / 2, 8, 0, Math.PI * 2);
      ctx.fill();
    }

    // info label update
    infoLbl.textContent = `Episodes: ${episodes}`;
  }

  function resetAll() {
    initGrid();
    initQ();
    episodes = 0;
    training = false;
    showingPath = false;
    pathCells = [];
    btnTrain.textContent = 'Train ▶';
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (training) {
      const speed = parseInt(speedSlider.value);
      for (let i = 0; i < speed; i++) runEpisode();
      if (showingPath) computePath();
    }
    render();
    animationId = requestAnimationFrame(loop);
  }

  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) loop();
    else if (!isVisible && animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && isVisible && !animationId) loop(); });

  // --- Events ---
  btnTrain.addEventListener('click', () => {
    training = !training;
    btnTrain.textContent = training ? 'Pause ⏸' : 'Train ▶';
    if (training && isVisible && !animationId) loop();
  });
  btnReset.addEventListener('click', resetAll);
  btnShowPath.addEventListener('click', () => {
    showingPath = !showingPath;
    if (showingPath) computePath();
    btnShowPath.textContent = showingPath ? 'Hide Path' : 'Show Path';
  });

  resetAll();
  render();
}
