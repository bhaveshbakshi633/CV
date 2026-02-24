// ============================================================
// MCTS Connect-4 — Monte Carlo Tree Search se AI khelta hai
// Human yellow, AI red — game tree growing dekhlo
// UCB1 se select, expand, rollout, backprop cycle dikhta hai
// ============================================================

export function initMCTS() {
  const container = document.getElementById('mctsContainer');
  if (!container) return;
  const CANVAS_HEIGHT = 400;
  let animationId = null, isVisible = false, canvasW = 0;

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:pointer;background:#111;border:1px solid rgba(74,158,255,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

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

  // --- Connect-4 constants ---
  const COLS = 7, ROWS = 6;
  const EMPTY = 0, HUMAN = 1, AI = 2; // human=yellow, AI=red
  const CELL_SIZE = 42;
  const BOARD_PAD = 10;

  // --- State ---
  let board = [];
  let currentPlayer = HUMAN;
  let gameOver = false;
  let winner = null;
  let thinkingIters = 500;
  let explorationC = 1.414;
  let aiThinking = false;
  let treeNodes = []; // MCTS tree nodes for visualization
  let statusText = 'Your turn — click a column';

  function initBoard() {
    board = [];
    for (let r = 0; r < ROWS; r++) {
      board.push(new Array(COLS).fill(EMPTY));
    }
    currentPlayer = HUMAN;
    gameOver = false;
    winner = null;
    aiThinking = false;
    treeNodes = [];
    statusText = 'Your turn — click a column';
  }

  // column mein piece drop karo — returns row ya -1 agar full
  function dropPiece(b, col, player) {
    for (let r = ROWS - 1; r >= 0; r--) {
      if (b[r][col] === EMPTY) {
        b[r][col] = player;
        return r;
      }
    }
    return -1;
  }

  // check win — kisi player ne 4 connect kiya?
  function checkWin(b, player) {
    // horizontal
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c <= COLS - 4; c++) {
        if (b[r][c] === player && b[r][c + 1] === player && b[r][c + 2] === player && b[r][c + 3] === player) return true;
      }
    }
    // vertical
    for (let r = 0; r <= ROWS - 4; r++) {
      for (let c = 0; c < COLS; c++) {
        if (b[r][c] === player && b[r + 1][c] === player && b[r + 2][c] === player && b[r + 3][c] === player) return true;
      }
    }
    // diagonal down-right
    for (let r = 0; r <= ROWS - 4; r++) {
      for (let c = 0; c <= COLS - 4; c++) {
        if (b[r][c] === player && b[r + 1][c + 1] === player && b[r + 2][c + 2] === player && b[r + 3][c + 3] === player) return true;
      }
    }
    // diagonal up-right
    for (let r = 3; r < ROWS; r++) {
      for (let c = 0; c <= COLS - 4; c++) {
        if (b[r][c] === player && b[r - 1][c + 1] === player && b[r - 2][c + 2] === player && b[r - 3][c + 3] === player) return true;
      }
    }
    return false;
  }

  function isBoardFull(b) {
    for (let c = 0; c < COLS; c++) {
      if (b[0][c] === EMPTY) return false;
    }
    return true;
  }

  function getValidMoves(b) {
    const moves = [];
    for (let c = 0; c < COLS; c++) {
      if (b[0][c] === EMPTY) moves.push(c);
    }
    return moves;
  }

  function copyBoard(b) {
    return b.map(r => [...r]);
  }

  // --- MCTS Node ---
  function createNode(board, player, move, parent) {
    return {
      board: copyBoard(board),
      player, move, parent,
      children: [],
      wins: 0, visits: 0,
      untriedMoves: getValidMoves(board)
    };
  }

  // UCB1 score — exploration vs exploitation balance
  function ucb1(node, parentVisits) {
    if (node.visits === 0) return Infinity;
    return (node.wins / node.visits) + explorationC * Math.sqrt(Math.log(parentVisits) / node.visits);
  }

  // MCTS: select best child using UCB1
  function selectChild(node) {
    let best = null, bestScore = -Infinity;
    for (const child of node.children) {
      const score = ucb1(child, node.visits);
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }
    return best;
  }

  // MCTS: random rollout — game khatam hone tak random moves khelo
  function rollout(b, player) {
    const bc = copyBoard(b);
    let p = player;
    for (let i = 0; i < 50; i++) {
      if (checkWin(bc, HUMAN)) return HUMAN;
      if (checkWin(bc, AI)) return AI;
      const moves = getValidMoves(bc);
      if (moves.length === 0) return 0; // draw
      const move = moves[Math.floor(Math.random() * moves.length)];
      dropPiece(bc, move, p);
      p = p === HUMAN ? AI : HUMAN;
    }
    return 0;
  }

  // MCTS main — ek iteration chala
  function mctsIteration(root) {
    // 1. Selection — leaf tak UCB1 se traverse karo
    let node = root;
    while (node.untriedMoves.length === 0 && node.children.length > 0) {
      node = selectChild(node);
    }
    // 2. Expansion — untried move se naya child banao
    if (node.untriedMoves.length > 0) {
      const moveIdx = Math.floor(Math.random() * node.untriedMoves.length);
      const move = node.untriedMoves.splice(moveIdx, 1)[0];
      const nb = copyBoard(node.board);
      const nextPlayer = node.player === HUMAN ? AI : HUMAN;
      dropPiece(nb, move, node.player);
      const child = createNode(nb, nextPlayer, move, node);
      node.children.push(child);
      node = child;
    }
    // 3. Rollout — random playout se result nikalo
    const result = rollout(node.board, node.player);
    // 4. Backpropagation — result ko root tak propagate karo
    while (node) {
      node.visits++;
      // AI ke perspective se score — AI jita toh +1, hara toh 0
      if (result === AI) node.wins += 1;
      else if (result === 0) node.wins += 0.5;
      node = node.parent;
    }
  }

  // AI ka turn — MCTS chalao aur best move return karo
  function aiMove() {
    aiThinking = true;
    statusText = 'AI soch raha hai...';

    const root = createNode(board, AI, -1, null);
    for (let i = 0; i < thinkingIters; i++) {
      mctsIteration(root);
    }

    // tree visualization ke liye nodes save karo
    treeNodes = [];
    function collectNodes(node, depth) {
      if (depth > 3) return; // sirf 3 levels dikhao
      treeNodes.push({ move: node.move, visits: node.visits, wins: node.wins, depth, children: node.children.length });
      for (const child of node.children) {
        collectNodes(child, depth + 1);
      }
    }
    collectNodes(root, 0);

    // best move — sabse zyada visits wala choose karo
    let bestChild = null, bestVisits = -1;
    for (const child of root.children) {
      if (child.visits > bestVisits) {
        bestVisits = child.visits;
        bestChild = child;
      }
    }

    if (bestChild) {
      dropPiece(board, bestChild.move, AI);
      if (checkWin(board, AI)) {
        gameOver = true;
        winner = AI;
        statusText = 'AI jeeta! New Game dabao';
      } else if (isBoardFull(board)) {
        gameOver = true;
        winner = null;
        statusText = 'Draw! New Game dabao';
      } else {
        currentPlayer = HUMAN;
        statusText = 'Your turn — click a column';
      }
    }
    aiThinking = false;
  }

  // human click handler
  canvas.addEventListener('click', (e) => {
    if (gameOver || aiThinking || currentPlayer !== HUMAN) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    // column calculate karo
    const boardW = COLS * CELL_SIZE;
    const boardX = (canvasW - boardW) / 2;
    const col = Math.floor((mx - boardX) / CELL_SIZE);
    if (col < 0 || col >= COLS) return;
    if (board[0][col] !== EMPTY) return;

    dropPiece(board, col, HUMAN);
    if (checkWin(board, HUMAN)) {
      gameOver = true;
      winner = HUMAN;
      statusText = 'Tum jeete! Sahi khela';
      return;
    }
    if (isBoardFull(board)) {
      gameOver = true;
      winner = null;
      statusText = 'Draw ho gaya!';
      return;
    }

    currentPlayer = AI;
    // AI ko next frame mein sochne do
    setTimeout(aiMove, 50);
  });

  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    const boardW = COLS * CELL_SIZE;
    const boardH = ROWS * CELL_SIZE;
    const boardX = (canvasW * 0.5) - (boardW / 2); // board left mein thoda shift
    const boardY = (CANVAS_HEIGHT - boardH) / 2;

    // board background
    ctx.fillStyle = 'rgba(20,40,80,0.8)';
    ctx.beginPath();
    ctx.roundRect(boardX - 5, boardY - 5, boardW + 10, boardH + 10, 8);
    ctx.fill();

    // cells draw karo
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const cx = boardX + c * CELL_SIZE + CELL_SIZE / 2;
        const cy = boardY + r * CELL_SIZE + CELL_SIZE / 2;
        // cell hole
        ctx.beginPath();
        ctx.arc(cx, cy, CELL_SIZE / 2 - 3, 0, Math.PI * 2);
        if (board[r][c] === EMPTY) {
          ctx.fillStyle = 'rgba(0,0,0,0.4)';
        } else if (board[r][c] === HUMAN) {
          ctx.fillStyle = '#ffd700'; // yellow — human
        } else {
          ctx.fillStyle = '#ff4444'; // red — AI
        }
        ctx.fill();
        // piece border
        if (board[r][c] !== EMPTY) {
          ctx.strokeStyle = 'rgba(255,255,255,0.3)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
    }

    // column hover indicator — top pe arrow
    // (skip if game over or AI thinking)

    // tree visualization — right side mein
    if (treeNodes.length > 0) {
      const treeX = boardX + boardW + 30;
      const treeW = canvasW - treeX - 10;
      if (treeW > 60) {
        ctx.fillStyle = 'rgba(74,158,255,0.7)';
        ctx.font = "10px 'JetBrains Mono',monospace";
        ctx.textAlign = 'left';
        ctx.fillText('MCTS Tree', treeX, 20);

        // depth 1 ke children — bar chart style
        const depth1 = treeNodes.filter(n => n.depth === 1);
        const maxVis = Math.max(1, ...depth1.map(n => n.visits));
        const barW = Math.min(30, treeW / (depth1.length + 1));
        for (let i = 0; i < depth1.length; i++) {
          const n = depth1[i];
          const barH = (n.visits / maxVis) * 150;
          const bx = treeX + i * (barW + 4);
          const by = 180 - barH;
          // bar
          const winRate = n.visits > 0 ? n.wins / n.visits : 0;
          ctx.fillStyle = `rgba(74,158,255,${0.3 + winRate * 0.7})`;
          ctx.fillRect(bx, by + 20, barW, barH);
          // visit count
          ctx.fillStyle = 'rgba(255,255,255,0.6)';
          ctx.font = "8px 'JetBrains Mono',monospace";
          ctx.textAlign = 'center';
          ctx.fillText(n.visits.toString(), bx + barW / 2, by + 16);
          // column label
          ctx.fillText(`C${n.move}`, bx + barW / 2, 195);
          // win rate
          ctx.fillStyle = winRate > 0.5 ? '#4a9eff' : '#ff6644';
          ctx.fillText((winRate * 100).toFixed(0) + '%', bx + barW / 2, 207);
        }

        // total stats
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = "10px 'JetBrains Mono',monospace";
        ctx.textAlign = 'left';
        const root = treeNodes[0];
        ctx.fillText(`Total sims: ${root.visits}`, treeX, 230);
        ctx.fillText(`Tree nodes: ${treeNodes.length}`, treeX, 245);
      }
    }

    // status text
    ctx.fillStyle = gameOver ? (winner === HUMAN ? '#ffd700' : winner === AI ? '#ff4444' : '#ccc') : '#4a9eff';
    ctx.font = "13px 'JetBrains Mono',monospace";
    ctx.textAlign = 'center';
    ctx.fillText(statusText, canvasW / 2, CANVAS_HEIGHT - 15);

    // winning line highlight — agar game khatam ho
    if (gameOver && winner) {
      ctx.strokeStyle = winner === HUMAN ? 'rgba(255,215,0,0.8)' : 'rgba(255,68,68,0.8)';
      ctx.lineWidth = 4;
      // find winning 4 — brute force check
      const p = winner;
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c <= COLS - 4; c++) {
          if (board[r][c] === p && board[r][c + 1] === p && board[r][c + 2] === p && board[r][c + 3] === p) {
            const x1 = boardX + c * CELL_SIZE + CELL_SIZE / 2;
            const x2 = boardX + (c + 3) * CELL_SIZE + CELL_SIZE / 2;
            const y1 = boardY + r * CELL_SIZE + CELL_SIZE / 2;
            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y1); ctx.stroke();
          }
        }
      }
      for (let r = 0; r <= ROWS - 4; r++) {
        for (let c = 0; c < COLS; c++) {
          if (board[r][c] === p && board[r + 1][c] === p && board[r + 2][c] === p && board[r + 3][c] === p) {
            const x1 = boardX + c * CELL_SIZE + CELL_SIZE / 2;
            const y1 = boardY + r * CELL_SIZE + CELL_SIZE / 2;
            const y2 = boardY + (r + 3) * CELL_SIZE + CELL_SIZE / 2;
            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1, y2); ctx.stroke();
          }
        }
      }
      for (let r = 0; r <= ROWS - 4; r++) {
        for (let c = 0; c <= COLS - 4; c++) {
          if (board[r][c] === p && board[r + 1][c + 1] === p && board[r + 2][c + 2] === p && board[r + 3][c + 3] === p) {
            const x1 = boardX + c * CELL_SIZE + CELL_SIZE / 2;
            const y1 = boardY + r * CELL_SIZE + CELL_SIZE / 2;
            const x2 = boardX + (c + 3) * CELL_SIZE + CELL_SIZE / 2;
            const y2 = boardY + (r + 3) * CELL_SIZE + CELL_SIZE / 2;
            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
          }
        }
      }
      for (let r = 3; r < ROWS; r++) {
        for (let c = 0; c <= COLS - 4; c++) {
          if (board[r][c] === p && board[r - 1][c + 1] === p && board[r - 2][c + 2] === p && board[r - 3][c + 3] === p) {
            const x1 = boardX + c * CELL_SIZE + CELL_SIZE / 2;
            const y1 = boardY + r * CELL_SIZE + CELL_SIZE / 2;
            const x2 = boardX + (c + 3) * CELL_SIZE + CELL_SIZE / 2;
            const y2 = boardY + (r - 3) * CELL_SIZE + CELL_SIZE / 2;
            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
          }
        }
      }
    }
  }

  // --- controls ---
  const itersSlider = mkSlider(ctrl, 'AI Thinking', 'mctsIters', 100, 2000, thinkingIters, 100);
  itersSlider.addEventListener('input', () => { thinkingIters = parseInt(itersSlider.value); });

  const cSlider = mkSlider(ctrl, 'Explore C', 'mctsC', 0.5, 3.0, explorationC, 0.1);
  cSlider.addEventListener('input', () => { explorationC = parseFloat(cSlider.value); });

  const newGameBtn = mkBtn(ctrl, 'New Game', 'mctsNewGame');
  newGameBtn.addEventListener('click', () => { initBoard(); });

  // --- render loop (no animation needed, just redraw on events) ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    draw();
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

  initBoard();
}
