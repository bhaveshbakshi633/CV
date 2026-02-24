// ============================================================
// Tic-Tac-Toe with Minimax + Alpha-Beta Pruning — Game Tree Visualization
// Human X khelta hai, AI O khelta hai. Poora search tree dikhta hai
// Alpha-beta pruning ka effect LIVE dekhoge — kitne nodes bach gaye
// ============================================================

// yahi entry point hai — container dhundho, canvas banao, game shuru karo
export function initMinimax() {
  const container = document.getElementById('minimaxContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';       // blue accent — AI wala color
  const RED = '#ef4444';           // O ka color / minimizer
  const GREEN = '#22c55e';         // positive score / X wins
  const GRAY = '#64748b';          // pruned branches
  const FONT = "'JetBrains Mono', monospace";

  // minimax scores
  const SCORE_X_WIN = 10;
  const SCORE_O_WIN = -10;
  const SCORE_DRAW = 0;

  // game tree display limits
  const MAX_TREE_DEPTH = 4;        // itne level tak tree dikhayenge — zyada ho toh collapse
  const MINI_BOARD_SIZE = 28;      // har node mein chhota board kitna bada
  const NODE_W = 48;               // node rectangle width
  const NODE_H = 52;               // node rectangle height
  const LEVEL_GAP = 72;            // vertical gap between tree levels
  const NODE_GAP_MIN = 6;          // minimum horizontal gap between sibling nodes

  // --- State ---
  let canvasW = 0, canvasH = 0, dpr = 1;
  let isVisible = false;
  let animationId = null;

  // game state
  // board: 0=empty, 1=X (human), 2=O (AI)
  let board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  let gameOver = false;
  let winner = 0;            // 0=none, 1=X, 2=O, 3=draw
  let winLine = null;        // [i, j, k] — winning cells ka index
  let humanTurn = true;
  let aiFirst = false;       // toggle — AI pehle khelega ya nahi
  let difficulty = 'perfect'; // 'perfect' ya 'easy'

  // alpha-beta toggle
  let useAlphaBeta = true;

  // animation state
  let animSpeed = 50;        // 0=instant, 1-100 = slow to fast
  let stepMode = false;      // step-through mode
  let waitingForStep = false;

  // tree data — har search ke baad populate hoga
  let treeRoot = null;        // root node of search tree
  let treeNodes = [];         // flat list for animation order
  let animIndex = 0;          // kitne nodes ab tak dikha chuke
  let animating = false;      // kya tree animation chal rahi hai
  let chosenPath = new Set(); // chosen move ka path — highlight ke liye
  let lastAnimTime = 0;

  // tree view scroll
  let treeScrollX = 0;
  let treeScrollY = 0;
  let treeDragging = false;
  let treeDragStartX = 0;
  let treeDragStartY = 0;
  let treeZoom = 1;

  // stats
  let nodesEvaluated = 0;
  let nodesPruned = 0;
  let nodesWithoutPruning = 0;

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // canvas — board + tree dono yahan
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.2)',
    'border-radius:8px',
    'background:rgba(2,2,8,0.5)',
    'cursor:pointer',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Stats bar ---
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'justify-content:space-between',
    'align-items:center',
    'padding:6px 10px',
    'margin-top:6px',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#94a3b8',
    'background:rgba(74,158,255,0.06)',
    'border:1px solid rgba(74,158,255,0.12)',
    'border-radius:6px',
    'gap:8px',
  ].join(';');
  container.appendChild(statsDiv);

  // stats spans
  const statNodes = document.createElement('span');
  const statPruned = document.createElement('span');
  const statEfficiency = document.createElement('span');
  const statScore = document.createElement('span');
  [statNodes, statPruned, statEfficiency, statScore].forEach(s => statsDiv.appendChild(s));

  // --- Controls row 1 ---
  const controlsDiv1 = document.createElement('div');
  controlsDiv1.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:6px',
    'align-items:center',
    'margin-top:8px',
  ].join(';');
  container.appendChild(controlsDiv1);

  // helper: button banao
  function makeBtn(text, active, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = btnCSS(active);
    btn.addEventListener('click', onClick);
    return btn;
  }

  function btnCSS(active) {
    return [
      'padding:5px 10px',
      'background:' + (active ? 'rgba(74,158,255,0.25)' : 'rgba(74,158,255,0.08)'),
      'color:' + (active ? '#e2e8f0' : '#94a3b8'),
      'border:1px solid ' + (active ? 'rgba(74,158,255,0.5)' : 'rgba(74,158,255,0.2)'),
      'border-radius:6px',
      'font-family:' + FONT,
      'font-size:11px',
      'cursor:pointer',
      'transition:all 0.15s',
      'outline:none',
      'white-space:nowrap',
    ].join(';');
  }

  // New Game button
  const newGameBtn = makeBtn('New Game', false, () => resetGame());
  controlsDiv1.appendChild(newGameBtn);

  // Alpha-Beta toggle
  const abToggle = makeBtn('Alpha-Beta: ON', true, () => {
    useAlphaBeta = !useAlphaBeta;
    abToggle.textContent = 'Alpha-Beta: ' + (useAlphaBeta ? 'ON' : 'OFF');
    abToggle.style.cssText = btnCSS(useAlphaBeta);
  });
  controlsDiv1.appendChild(abToggle);

  // AI First toggle
  const aiFirstBtn = makeBtn('AI First: OFF', false, () => {
    aiFirst = !aiFirst;
    aiFirstBtn.textContent = 'AI First: ' + (aiFirst ? 'ON' : 'OFF');
    aiFirstBtn.style.cssText = btnCSS(aiFirst);
    resetGame();
  });
  controlsDiv1.appendChild(aiFirstBtn);

  // Difficulty toggle
  const diffBtn = makeBtn('Difficulty: Perfect', true, () => {
    difficulty = difficulty === 'perfect' ? 'easy' : 'perfect';
    diffBtn.textContent = 'Difficulty: ' + (difficulty === 'perfect' ? 'Perfect' : 'Easy');
    diffBtn.style.cssText = btnCSS(difficulty === 'perfect');
  });
  controlsDiv1.appendChild(diffBtn);

  // Step mode toggle
  const stepBtn = makeBtn('Step Mode: OFF', false, () => {
    stepMode = !stepMode;
    stepBtn.textContent = 'Step Mode: ' + (stepMode ? 'ON' : 'OFF');
    stepBtn.style.cssText = btnCSS(stepMode);
  });
  controlsDiv1.appendChild(stepBtn);

  // Next Step button (step mode ke liye)
  const nextStepBtn = makeBtn('Next Step >', false, () => {
    if (stepMode && waitingForStep) {
      waitingForStep = false;
    }
  });
  controlsDiv1.appendChild(nextStepBtn);

  // --- Controls row 2: speed slider ---
  const controlsDiv2 = document.createElement('div');
  controlsDiv2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'align-items:center',
    'margin-top:6px',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#94a3b8',
  ].join(';');
  container.appendChild(controlsDiv2);

  // speed slider
  const speedLabel = document.createElement('span');
  speedLabel.textContent = 'Animation Speed:';
  controlsDiv2.appendChild(speedLabel);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '0';
  speedSlider.max = '100';
  speedSlider.value = String(animSpeed);
  speedSlider.style.cssText = 'width:120px;accent-color:' + ACCENT + ';cursor:pointer;';
  speedSlider.addEventListener('input', () => {
    animSpeed = parseInt(speedSlider.value);
    speedValSpan.textContent = animSpeed === 0 ? 'Instant' : animSpeed;
  });
  controlsDiv2.appendChild(speedSlider);

  const speedValSpan = document.createElement('span');
  speedValSpan.textContent = String(animSpeed);
  speedValSpan.style.color = '#e2e8f0';
  controlsDiv2.appendChild(speedValSpan);

  // zoom controls
  const zoomLabel = document.createElement('span');
  zoomLabel.textContent = '  Tree Zoom:';
  zoomLabel.style.marginLeft = '10px';
  controlsDiv2.appendChild(zoomLabel);

  const zoomOutBtn = makeBtn('−', false, () => {
    treeZoom = Math.max(0.3, treeZoom - 0.15);
    requestDraw();
  });
  controlsDiv2.appendChild(zoomOutBtn);

  const zoomResetBtn = makeBtn('Reset', false, () => {
    treeZoom = 1;
    treeScrollX = 0;
    treeScrollY = 0;
    requestDraw();
  });
  controlsDiv2.appendChild(zoomResetBtn);

  const zoomInBtn = makeBtn('+', false, () => {
    treeZoom = Math.min(2.5, treeZoom + 0.15);
    requestDraw();
  });
  controlsDiv2.appendChild(zoomInBtn);

  // ============================================================
  // CANVAS SIZING — DPR aware
  // ============================================================
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth * dpr;
    canvasH = CANVAS_HEIGHT * dpr;
    canvas.width = canvasW;
    canvas.height = canvasH;
  }

  // ============================================================
  // GAME LOGIC — board management
  // ============================================================

  // winning combos — 3 in a row ke saare possibilities
  const WIN_COMBOS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
    [0, 4, 8], [2, 4, 6],             // diagonals
  ];

  // check karo koi jeeta hai ya nahi
  function checkWinner(b) {
    for (const [a, c, d] of WIN_COMBOS) {
      if (b[a] && b[a] === b[c] && b[a] === b[d]) {
        return { winner: b[a], line: [a, c, d] };
      }
    }
    // draw check — koi empty cell nahi bacha
    if (b.every(c => c !== 0)) return { winner: 3, line: null };
    return { winner: 0, line: null };
  }

  // kya board pe koi move bachi hai
  function getEmptyCells(b) {
    const cells = [];
    for (let i = 0; i < 9; i++) {
      if (b[i] === 0) cells.push(i);
    }
    return cells;
  }

  // ============================================================
  // MINIMAX ALGORITHM — search tree ke saath
  // ============================================================

  // tree node structure
  // { board, move, score, alpha, beta, isMaximizer, children, pruned, depth, id }
  let nodeIdCounter = 0;

  // minimax with optional alpha-beta pruning
  // tree bhi banaata hai saath mein — visualization ke liye
  function minimaxSearch(b, depth, alpha, beta, isMaximizer, parentAlpha, parentBeta) {
    const nodeId = nodeIdCounter++;
    const node = {
      board: b.slice(),
      move: -1,
      score: null,
      alpha: parentAlpha,
      beta: parentBeta,
      isMaximizer: isMaximizer,
      children: [],
      pruned: false,
      depth: depth,
      id: nodeId,
    };

    // terminal state check — koi jeeta ya board full
    const result = checkWinner(b);
    if (result.winner !== 0) {
      if (result.winner === 1) node.score = SCORE_X_WIN - depth;       // X jeeta — depth penalty (jaldi jeetna better)
      else if (result.winner === 2) node.score = SCORE_O_WIN + depth;  // O jeeta — depth reward (deri se haarna better for X)
      else node.score = SCORE_DRAW;                                     // draw
      nodesEvaluated++;
      treeNodes.push(node);
      return node;
    }

    const emptyCells = getEmptyCells(b);

    if (isMaximizer) {
      // X ka turn — maximize karega
      let bestScore = -Infinity;
      for (let i = 0; i < emptyCells.length; i++) {
        const cell = emptyCells[i];
        const newBoard = b.slice();
        newBoard[cell] = 1; // X place karo

        const childNode = minimaxSearch(newBoard, depth + 1, alpha, beta, false, alpha, beta);
        childNode.move = cell;
        node.children.push(childNode);

        if (childNode.score > bestScore) bestScore = childNode.score;

        if (useAlphaBeta) {
          alpha = Math.max(alpha, childNode.score);
          // beta cutoff — minimizer isse better option already dhundh chuka
          if (beta <= alpha) {
            // baaki bachche prune kar do
            for (let j = i + 1; j < emptyCells.length; j++) {
              nodesPruned++;
              const prunedNode = {
                board: b.slice(),
                move: emptyCells[j],
                score: null,
                alpha: alpha,
                beta: beta,
                isMaximizer: false,
                children: [],
                pruned: true,
                depth: depth + 1,
                id: nodeIdCounter++,
              };
              // pruned node pe move dikhao board pe
              prunedNode.board[emptyCells[j]] = 1;
              node.children.push(prunedNode);
              treeNodes.push(prunedNode);
            }
            break;
          }
        }
      }
      node.score = bestScore;
      node.alpha = alpha;
    } else {
      // O ka turn — minimize karega
      let bestScore = Infinity;
      for (let i = 0; i < emptyCells.length; i++) {
        const cell = emptyCells[i];
        const newBoard = b.slice();
        newBoard[cell] = 2; // O place karo

        const childNode = minimaxSearch(newBoard, depth + 1, alpha, beta, true, alpha, beta);
        childNode.move = cell;
        node.children.push(childNode);

        if (childNode.score < bestScore) bestScore = childNode.score;

        if (useAlphaBeta) {
          beta = Math.min(beta, childNode.score);
          // alpha cutoff — maximizer isse better option already dhundh chuka
          if (beta <= alpha) {
            for (let j = i + 1; j < emptyCells.length; j++) {
              nodesPruned++;
              const prunedNode = {
                board: b.slice(),
                move: emptyCells[j],
                score: null,
                alpha: alpha,
                beta: beta,
                isMaximizer: true,
                children: [],
                pruned: true,
                depth: depth + 1,
                id: nodeIdCounter++,
              };
              prunedNode.board[emptyCells[j]] = 2;
              node.children.push(prunedNode);
              treeNodes.push(prunedNode);
            }
            break;
          }
        }
      }
      node.score = bestScore;
      node.beta = beta;
    }

    nodesEvaluated++;
    treeNodes.push(node);
    return node;
  }

  // without pruning count — estimate karo kitne nodes bina AB ke evaluate hote
  function countWithoutPruning(b, isMaximizer) {
    const result = checkWinner(b);
    if (result.winner !== 0) return 1;
    const emptyCells = getEmptyCells(b);
    let count = 1;
    for (const cell of emptyCells) {
      const newBoard = b.slice();
      newBoard[cell] = isMaximizer ? 1 : 2;
      count += countWithoutPruning(newBoard, !isMaximizer);
    }
    return count;
  }

  // AI ka best move dhundho
  function findAIMove() {
    // easy mode: 30% chance minimax use kare, 70% random
    if (difficulty === 'easy' && Math.random() > 0.3) {
      const empty = getEmptyCells(board);
      if (empty.length === 0) return -1;
      return empty[Math.floor(Math.random() * empty.length)];
    }

    // minimax search reset
    nodeIdCounter = 0;
    nodesEvaluated = 0;
    nodesPruned = 0;
    treeNodes = [];

    // AI = O = minimizer, toh isMaximizer false hoga jab AI ka turn hai
    // lekin agar aiFirst on hai toh AI = X = maximizer
    const aiIsMaximizer = aiFirst;
    const aiPiece = aiFirst ? 1 : 2;

    treeRoot = minimaxSearch(board.slice(), 0, -Infinity, Infinity, aiIsMaximizer, -Infinity, Infinity);

    // without pruning estimate
    nodesWithoutPruning = countWithoutPruning(board.slice(), aiIsMaximizer);

    // best child dhundho
    let bestChild = null;
    if (aiIsMaximizer) {
      let bestScore = -Infinity;
      for (const child of treeRoot.children) {
        if (!child.pruned && child.score > bestScore) {
          bestScore = child.score;
          bestChild = child;
        }
      }
    } else {
      let bestScore = Infinity;
      for (const child of treeRoot.children) {
        if (!child.pruned && child.score < bestScore) {
          bestScore = child.score;
          bestChild = child;
        }
      }
    }

    // chosen path mark karo — root se best child tak
    chosenPath.clear();
    if (bestChild) {
      chosenPath.add(treeRoot.id);
      markChosenPath(bestChild);
    }

    updateStats();
    return bestChild ? bestChild.move : -1;
  }

  // chosen path recursively mark karo
  function markChosenPath(node) {
    chosenPath.add(node.id);
    if (node.children.length > 0) {
      // best child of this node
      let best = null;
      if (node.isMaximizer) {
        let bestScore = -Infinity;
        for (const c of node.children) {
          if (!c.pruned && c.score !== null && c.score > bestScore) { bestScore = c.score; best = c; }
        }
      } else {
        let bestScore = Infinity;
        for (const c of node.children) {
          if (!c.pruned && c.score !== null && c.score < bestScore) { bestScore = c.score; best = c; }
        }
      }
      if (best) markChosenPath(best);
    }
  }

  // stats update karo
  function updateStats() {
    statNodes.textContent = 'Nodes: ' + nodesEvaluated;
    statPruned.textContent = 'Pruned: ' + nodesPruned;
    const eff = nodesWithoutPruning > 0 ? ((1 - nodesEvaluated / nodesWithoutPruning) * 100).toFixed(1) : '0.0';
    statEfficiency.textContent = 'Efficiency: ' + eff + '%';
    const aiScore = treeRoot ? treeRoot.score : 0;
    statScore.textContent = 'AI Score: ' + aiScore;
  }

  // ============================================================
  // GAME FLOW
  // ============================================================

  function resetGame() {
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    gameOver = false;
    winner = 0;
    winLine = null;
    treeRoot = null;
    treeNodes = [];
    animIndex = 0;
    animating = false;
    chosenPath.clear();
    nodesEvaluated = 0;
    nodesPruned = 0;
    nodesWithoutPruning = 0;
    treeScrollX = 0;
    treeScrollY = 0;
    treeZoom = 1;
    waitingForStep = false;

    // agar AI first hai toh AI ka turn pehle
    if (aiFirst) {
      humanTurn = false;
      // thoda delay se AI move kare taki reset dikhee
      setTimeout(() => aiPlay(), 150);
    } else {
      humanTurn = true;
    }

    updateStats();
    requestDraw();
  }

  // human move process karo
  function humanMove(cellIndex) {
    if (gameOver || !humanTurn || board[cellIndex] !== 0) return;
    if (animating) return; // animation ke dauran click mat lo

    const humanPiece = aiFirst ? 2 : 1;
    board[cellIndex] = humanPiece;

    // check karo game khatam hua ya nahi
    const result = checkWinner(board);
    if (result.winner !== 0) {
      gameOver = true;
      winner = result.winner;
      winLine = result.line;
      requestDraw();
      return;
    }

    humanTurn = false;
    requestDraw();

    // AI ka turn — thoda delay taki human move dikhe
    setTimeout(() => aiPlay(), 200);
  }

  // AI move kare
  function aiPlay() {
    if (gameOver) return;

    const move = findAIMove();
    if (move < 0) return;

    // tree animation start karo
    if (animSpeed > 0 && treeNodes.length > 0 && !stepMode) {
      animIndex = 0;
      animating = true;
      lastAnimTime = performance.now();
      // continuous rAF loop kick karo — draw() mein handle hoga
      if (!animationId && isVisible) {
        animationId = requestAnimationFrame(draw);
      }
    } else if (stepMode && treeNodes.length > 0) {
      // step mode — pehla node dikha, fir user click karega
      animIndex = 1;
      animating = true;
      waitingForStep = true;
      requestDraw();
    } else {
      // instant mode ya koi tree nahi — seedha move place karo
      placeAIMove(move);
      requestDraw();
    }
  }

  // AI ka move board pe rakho
  function placeAIMove(move) {
    const aiPiece = aiFirst ? 1 : 2;
    board[move] = aiPiece;
    animating = false;

    const result = checkWinner(board);
    if (result.winner !== 0) {
      gameOver = true;
      winner = result.winner;
      winLine = result.line;
    } else {
      humanTurn = true;
    }
    requestDraw();
  }

  // ============================================================
  // TREE LAYOUT CALCULATION — recursive positioning
  // ============================================================

  // har node ko x, y position do tree mein
  function layoutTree(node, depth, maxDepth) {
    if (!node) return { width: 0 };

    // agar pruned hai ya max depth cross ho gayi toh leaf maan lo
    if (node.pruned || depth >= maxDepth || node.children.length === 0) {
      node._layoutW = NODE_W;
      node._layoutX = 0;
      node._layoutY = depth * LEVEL_GAP;
      node._layoutDepth = depth;
      return { width: NODE_W };
    }

    // pehle bachchon ka layout karo
    let totalChildWidth = 0;
    const childLayouts = [];
    for (const child of node.children) {
      const cl = layoutTree(child, depth + 1, maxDepth);
      childLayouts.push(cl);
      totalChildWidth += cl.width;
    }
    // siblings ke beech gap
    totalChildWidth += Math.max(0, node.children.length - 1) * NODE_GAP_MIN;

    const myWidth = Math.max(NODE_W, totalChildWidth);
    node._layoutW = myWidth;
    node._layoutY = depth * LEVEL_GAP;
    node._layoutDepth = depth;

    // bachchon ko position do — centered under parent
    let childX = -totalChildWidth / 2;
    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i];
      child._layoutX = childX + child._layoutW / 2;
      childX += child._layoutW + NODE_GAP_MIN;
    }

    node._layoutX = 0; // root centered
    return { width: myWidth };
  }

  // ============================================================
  // DRAWING — board + tree + everything
  // ============================================================

  function draw() {
    // lab pause check
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    animationId = null;

    const W = canvasW;
    const H = canvasH;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W / dpr, H / dpr);

    const cssW = W / dpr;
    const cssH = H / dpr;

    // left section: board (35%)
    const boardAreaW = cssW * 0.33;
    const boardAreaH = cssH;
    drawBoard(0, 0, boardAreaW, boardAreaH);

    // divider line
    ctx.strokeStyle = 'rgba(74,158,255,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(boardAreaW, 0);
    ctx.lineTo(boardAreaW, cssH);
    ctx.stroke();

    // right section: tree (65%)
    const treeAreaX = boardAreaW;
    const treeAreaW = cssW - boardAreaW;
    drawTree(treeAreaX, 0, treeAreaW, cssH);

    // animation update — agar tree animation chal rahi hai
    if (animating && treeNodes.length > 0) {
      const now = performance.now();
      if (stepMode) {
        // step mode — har click pe ek node aage
        if (!waitingForStep) {
          animIndex++;
          waitingForStep = true;
          if (animIndex >= treeNodes.length) finishAnimation();
        }
        // step mode mein continuous loop nahi — sirf step pe draw
      } else if (animSpeed === 0) {
        // instant — saare nodes dikha do
        animIndex = treeNodes.length;
        finishAnimation();
      } else {
        // timed animation — speed ke hisaab se nodes dikha
        const delay = Math.max(5, 300 - animSpeed * 2.8);
        if (now - lastAnimTime > delay) {
          // ek saath kaafi nodes advance kar sakte hain agar delay chota hai
          const steps = Math.max(1, Math.floor((now - lastAnimTime) / delay));
          animIndex = Math.min(animIndex + steps, treeNodes.length);
          lastAnimTime = now;
          if (animIndex >= treeNodes.length) finishAnimation();
        }
        // continuous rAF loop jab tak animation chal rahi hai
        if (animating) {
          animationId = requestAnimationFrame(draw);
          return; // early return taki neeche wala animationId null na ho
        }
      }
    }
  }

  function finishAnimation() {
    animating = false;
    waitingForStep = false;
    // best move dhundho aur place karo
    if (treeRoot) {
      const aiIsMaximizer = aiFirst;
      let bestChild = null;
      if (aiIsMaximizer) {
        let bs = -Infinity;
        for (const c of treeRoot.children) { if (!c.pruned && c.score > bs) { bs = c.score; bestChild = c; } }
      } else {
        let bs = Infinity;
        for (const c of treeRoot.children) { if (!c.pruned && c.score < bs) { bs = c.score; bestChild = c; } }
      }
      if (bestChild && bestChild.move >= 0) {
        placeAIMove(bestChild.move);
      }
    }
  }

  // --- Board drawing ---
  function drawBoard(ox, oy, areaW, areaH) {
    const padding = 20;
    const boardSize = Math.min(areaW - padding * 2, areaH - padding * 2 - 20);
    const cellSize = boardSize / 3;
    const bx = ox + (areaW - boardSize) / 2;
    const by = oy + (areaH - boardSize) / 2;

    // title
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '13px ' + FONT;
    ctx.textAlign = 'center';
    const titleText = gameOver
      ? (winner === 3 ? 'Draw!' : (winner === 1 ? 'X Wins!' : 'O Wins!'))
      : (humanTurn ? 'Your Turn (' + (aiFirst ? 'O' : 'X') + ')' : 'AI Thinking...');
    ctx.fillText(titleText, ox + areaW / 2, by - 8);

    // grid lines
    ctx.strokeStyle = 'rgba(148,163,184,0.4)';
    ctx.lineWidth = 2;
    for (let i = 1; i < 3; i++) {
      // vertical
      ctx.beginPath();
      ctx.moveTo(bx + i * cellSize, by);
      ctx.lineTo(bx + i * cellSize, by + boardSize);
      ctx.stroke();
      // horizontal
      ctx.beginPath();
      ctx.moveTo(bx, by + i * cellSize);
      ctx.lineTo(bx + boardSize, by + i * cellSize);
      ctx.stroke();
    }

    // cells mein X aur O draw karo
    for (let i = 0; i < 9; i++) {
      const row = Math.floor(i / 3);
      const col = i % 3;
      const cx = bx + col * cellSize + cellSize / 2;
      const cy = by + row * cellSize + cellSize / 2;
      const r = cellSize * 0.3;

      if (board[i] === 1) {
        // X draw karo — blue
        drawX(cx, cy, r, ACCENT, 2.5);
      } else if (board[i] === 2) {
        // O draw karo — red
        drawO(cx, cy, r, RED, 2.5);
      }
    }

    // winning line highlight
    if (winLine) {
      const [a, b, c] = winLine;
      const ax = bx + (a % 3) * cellSize + cellSize / 2;
      const ay = by + Math.floor(a / 3) * cellSize + cellSize / 2;
      const cx2 = bx + (c % 3) * cellSize + cellSize / 2;
      const cy2 = by + Math.floor(c / 3) * cellSize + cellSize / 2;

      ctx.strokeStyle = winner === 1 ? ACCENT : RED;
      ctx.lineWidth = 4;
      ctx.shadowColor = winner === 1 ? ACCENT : RED;
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(cx2, cy2);
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // store board position for click handling
    canvas._boardBounds = { bx, by, cellSize, boardSize };
  }

  // X symbol draw karo
  function drawX(cx, cy, r, color, lw) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx - r, cy - r);
    ctx.lineTo(cx + r, cy + r);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx + r, cy - r);
    ctx.lineTo(cx - r, cy + r);
    ctx.stroke();
  }

  // O symbol draw karo
  function drawO(cx, cy, r, color, lw) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }

  // --- Tree drawing ---
  function drawTree(ox, oy, areaW, areaH) {
    if (!treeRoot || treeNodes.length === 0) {
      // empty state — message dikha
      ctx.fillStyle = '#475569';
      ctx.font = '12px ' + FONT;
      ctx.textAlign = 'center';
      ctx.fillText('Make a move to see the', ox + areaW / 2, oy + areaH / 2 - 10);
      ctx.fillText('AI search tree here', ox + areaW / 2, oy + areaH / 2 + 10);
      return;
    }

    // tree layout karo (agar already nahi kiya)
    const displayDepth = Math.min(MAX_TREE_DEPTH, getMaxDepth(treeRoot));
    layoutTree(treeRoot, 0, displayDepth);

    // clip to tree area
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, areaW, areaH);
    ctx.clip();

    // transform: scroll + zoom
    const centerX = ox + areaW / 2 + treeScrollX;
    const centerY = oy + 35 + treeScrollY;

    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.scale(treeZoom, treeZoom);

    // visible nodes ka set banao — animation ke liye
    const visibleNodeIds = new Set();
    for (let i = 0; i < Math.min(animating ? animIndex : treeNodes.length, treeNodes.length); i++) {
      visibleNodeIds.add(treeNodes[i].id);
    }

    // connections pehle draw karo (neeche rahein)
    drawTreeConnections(treeRoot, 0, 0, displayDepth, visibleNodeIds);

    // fir nodes draw karo
    drawTreeNodes(treeRoot, 0, 0, displayDepth, visibleNodeIds);

    ctx.restore();
    ctx.restore();

    // tree area label
    ctx.fillStyle = '#475569';
    ctx.font = '10px ' + FONT;
    ctx.textAlign = 'left';
    ctx.fillText('Search Tree (scroll to pan, buttons to zoom)', ox + 8, oy + 14);
  }

  // max depth of tree nikalo
  function getMaxDepth(node) {
    if (!node || node.children.length === 0) return node ? node.depth : 0;
    let max = node.depth;
    for (const c of node.children) {
      max = Math.max(max, getMaxDepth(c));
    }
    return max;
  }

  // tree connections recursively draw karo
  function drawTreeConnections(node, px, py, maxDepth, visibleIds) {
    if (!node || node._layoutDepth >= maxDepth) return;
    if (!visibleIds.has(node.id)) return;

    const nx = px + (node._layoutX || 0);
    const ny = node._layoutY + NODE_H / 2;

    for (const child of node.children) {
      if (!visibleIds.has(child.id)) continue;

      const cx = px + (child._layoutX || 0);
      const cy = child._layoutY - NODE_H / 2;

      // connection line style based on pruned/chosen status
      if (child.pruned) {
        ctx.strokeStyle = 'rgba(100,116,139,0.3)';
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
      } else if (chosenPath.has(node.id) && chosenPath.has(child.id)) {
        ctx.strokeStyle = ACCENT;
        ctx.setLineDash([]);
        ctx.lineWidth = 2;
        ctx.shadowColor = ACCENT;
        ctx.shadowBlur = 6;
      } else {
        // gradient: parent ka color se child ka color
        ctx.strokeStyle = node.isMaximizer ? 'rgba(74,158,255,0.3)' : 'rgba(239,68,68,0.3)';
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
      }

      ctx.beginPath();
      ctx.moveTo(nx, ny);
      // bezier curve — smoother connections
      const midY = (ny + cy) / 2;
      ctx.bezierCurveTo(nx, midY, cx, midY, cx, cy);
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.setLineDash([]);

      // child ke subtree ki connections
      drawTreeConnections(child, px, py, maxDepth, visibleIds);
    }
  }

  // tree nodes recursively draw karo
  function drawTreeNodes(node, px, py, maxDepth, visibleIds) {
    if (!node) return;
    if (!visibleIds.has(node.id)) return;
    if (node._layoutDepth > maxDepth) return;

    const nx = px + (node._layoutX || 0);
    const ny = node._layoutY;

    // node rectangle draw karo
    const halfW = NODE_W / 2;
    const halfH = NODE_H / 2;
    const rx = nx - halfW;
    const ry = ny - halfH;

    // background
    const isChosen = chosenPath.has(node.id);
    if (node.pruned) {
      ctx.fillStyle = 'rgba(100,116,139,0.1)';
    } else if (isChosen) {
      ctx.fillStyle = 'rgba(74,158,255,0.15)';
    } else {
      ctx.fillStyle = 'rgba(15,15,25,0.8)';
    }

    // rounded rect
    const cr = 5;
    ctx.beginPath();
    ctx.moveTo(rx + cr, ry);
    ctx.lineTo(rx + halfW * 2 - cr, ry);
    ctx.quadraticCurveTo(rx + halfW * 2, ry, rx + halfW * 2, ry + cr);
    ctx.lineTo(rx + halfW * 2, ry + halfH * 2 - cr);
    ctx.quadraticCurveTo(rx + halfW * 2, ry + halfH * 2, rx + halfW * 2 - cr, ry + halfH * 2);
    ctx.lineTo(rx + cr, ry + halfH * 2);
    ctx.quadraticCurveTo(rx, ry + halfH * 2, rx, ry + halfH * 2 - cr);
    ctx.lineTo(rx, ry + cr);
    ctx.quadraticCurveTo(rx, ry, rx + cr, ry);
    ctx.closePath();
    ctx.fill();

    // border — maximizer blue, minimizer red
    if (node.pruned) {
      ctx.strokeStyle = 'rgba(100,116,139,0.3)';
      ctx.setLineDash([3, 3]);
    } else if (isChosen) {
      ctx.strokeStyle = ACCENT;
      ctx.setLineDash([]);
      ctx.shadowColor = ACCENT;
      ctx.shadowBlur = 8;
    } else {
      ctx.strokeStyle = node.isMaximizer ? 'rgba(74,158,255,0.5)' : 'rgba(239,68,68,0.5)';
      ctx.setLineDash([]);
    }
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);

    // mini board draw karo node ke andar
    drawMiniBoard(nx, ny - 5, MINI_BOARD_SIZE, node.board);

    // score badge neeche
    if (node.score !== null && !node.pruned) {
      const scoreText = String(node.score);
      ctx.font = '8px ' + FONT;
      ctx.textAlign = 'center';
      // score ke hisaab se color — positive green, negative red, zero gray
      if (node.score > 0) ctx.fillStyle = GREEN;
      else if (node.score < 0) ctx.fillStyle = RED;
      else ctx.fillStyle = '#94a3b8';
      ctx.fillText(scoreText, nx, ny + halfH - 2);
    }

    // pruned label
    if (node.pruned) {
      ctx.font = '7px ' + FONT;
      ctx.fillStyle = '#ef4444';
      ctx.textAlign = 'center';
      ctx.fillText('CUT', nx, ny + halfH - 2);
    }

    // bachchon ko draw karo
    if (node._layoutDepth < maxDepth) {
      for (const child of node.children) {
        drawTreeNodes(child, px, py, maxDepth, visibleIds);
      }
    }
  }

  // chhota board draw karo (node ke andar wala)
  function drawMiniBoard(cx, cy, size, b) {
    const cs = size / 3; // cell size
    const startX = cx - size / 2;
    const startY = cy - size / 2;

    // grid lines
    ctx.strokeStyle = 'rgba(148,163,184,0.25)';
    ctx.lineWidth = 0.5;
    for (let i = 1; i < 3; i++) {
      ctx.beginPath();
      ctx.moveTo(startX + i * cs, startY);
      ctx.lineTo(startX + i * cs, startY + size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(startX, startY + i * cs);
      ctx.lineTo(startX + size, startY + i * cs);
      ctx.stroke();
    }

    // pieces
    for (let i = 0; i < 9; i++) {
      if (b[i] === 0) continue;
      const row = Math.floor(i / 3);
      const col = i % 3;
      const pcx = startX + col * cs + cs / 2;
      const pcy = startY + row * cs + cs / 2;
      const r = cs * 0.3;

      if (b[i] === 1) {
        drawX(pcx, pcy, r, ACCENT, 0.8);
      } else {
        drawO(pcx, pcy, r, RED, 0.8);
      }
    }
  }

  // ============================================================
  // INPUT HANDLING — click on board + tree panning
  // ============================================================

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // board click check
    const bb = canvas._boardBounds;
    if (bb && mx >= bb.bx && mx < bb.bx + bb.boardSize && my >= bb.by && my < bb.by + bb.boardSize) {
      const col = Math.floor((mx - bb.bx) / bb.cellSize);
      const row = Math.floor((my - bb.by) / bb.cellSize);
      const cellIndex = row * 3 + col;
      if (cellIndex >= 0 && cellIndex < 9) {
        humanMove(cellIndex);
      }
    }
  });

  // tree panning — mouse drag
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const cssW = canvasW / dpr;
    const boardAreaW = cssW * 0.33;

    // sirf tree area mein drag karo
    if (mx > boardAreaW) {
      treeDragging = true;
      treeDragStartX = e.clientX - treeScrollX;
      treeDragStartY = e.clientY - treeScrollY;
      canvas.style.cursor = 'grabbing';
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (treeDragging) {
      treeScrollX = e.clientX - treeDragStartX;
      treeScrollY = e.clientY - treeDragStartY;
      requestDraw();
    }
  });

  canvas.addEventListener('mouseup', () => {
    treeDragging = false;
    canvas.style.cursor = 'pointer';
  });

  canvas.addEventListener('mouseleave', () => {
    treeDragging = false;
    canvas.style.cursor = 'pointer';
  });

  // mouse wheel se zoom karo tree area mein
  canvas.addEventListener('wheel', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const cssW = canvasW / dpr;
    const boardAreaW = cssW * 0.33;

    if (mx > boardAreaW) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.08 : 0.08;
      treeZoom = Math.max(0.2, Math.min(3, treeZoom + delta));
      requestDraw();
    }
  }, { passive: false });

  // ============================================================
  // ANIMATION LOOP + VISIBILITY MANAGEMENT
  // ============================================================

  function requestDraw() {
    if (!animationId && isVisible) {
      animationId = requestAnimationFrame(draw);
    }
  }

  // intersection observer — sirf visible hone pe draw karo
  const observer = new IntersectionObserver(([entry]) => {
    isVisible = entry.isIntersecting;
    if (isVisible && !animationId) requestDraw();
    else if (!isVisible && animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }, { threshold: 0.1 });
  observer.observe(container);

  // lab resume event — jab focus wapas aaye
  document.addEventListener('lab:resume', () => {
    if (isVisible && !animationId) requestDraw();
  });

  // tab visibility change — battery bachao
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    } else {
      if (isVisible && !animationId) requestDraw();
    }
  });

  // resize handler
  window.addEventListener('resize', () => {
    resizeCanvas();
    requestDraw();
  });

  // ============================================================
  // INITIALIZATION — sab set karke game shuru karo
  // ============================================================

  resizeCanvas();
  resetGame();
  requestDraw();
}
