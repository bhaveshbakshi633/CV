// ============================================================
// RRT / RRT* Path Planner — 2D space mein obstacles ke beech raasta dhundho
// Tree organically grow hota hai — fractal jaisa lagta hai
// RRT* rewiring se shorter paths milte hain
// ============================================================

export function initRRT() {
  const container = document.getElementById('rrtContainer');
  if (!container) return;
  const CANVAS_HEIGHT = 400;
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

  // --- state ---
  let stepSize = 15;
  let goalBias = 0.1; // 10% chance goal ki taraf sample karo
  let useRRTStar = false;
  let tree = []; // [{x, y, parent, cost}]
  let path = []; // goal tak ka path (agar mila toh)
  let obstacles = [];
  let startPt = { x: 50, y: CANVAS_HEIGHT / 2 };
  let goalPt = { x: 0, y: CANVAS_HEIGHT / 2 }; // canvasW pe set hoga
  let goalRadius = 20;
  let running = true;
  let pathFound = false;
  let nodesPerFrame = 3;

  function dist(a, b) {
    const dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // random obstacles generate karo
  function generateObstacles() {
    obstacles = [];
    const count = 6 + Math.floor(Math.random() * 5);
    for (let i = 0; i < count; i++) {
      const w = 30 + Math.random() * 60;
      const h = 30 + Math.random() * 80;
      const x = 100 + Math.random() * (canvasW - 250);
      const y = 20 + Math.random() * (CANVAS_HEIGHT - 60 - h);
      obstacles.push({ x, y, w, h });
    }
  }

  // point obstacle ke andar hai?
  function inObstacle(px, py) {
    for (const o of obstacles) {
      if (px >= o.x && px <= o.x + o.w && py >= o.y && py <= o.y + o.h) return true;
    }
    return false;
  }

  // line segment obstacle se cross karta hai?
  function lineCollides(x1, y1, x2, y2) {
    // 20 points pe sample karke check karo — simple but effective
    const steps = 20;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const px = x1 + (x2 - x1) * t;
      const py = y1 + (y2 - y1) * t;
      if (inObstacle(px, py)) return true;
    }
    return false;
  }

  // RRT step — ek node add karo tree mein
  function rrtStep() {
    if (pathFound && !useRRTStar) return; // basic RRT mein path milne ke baad ruko

    // random point sample karo (goal bias ke saath)
    let sampleX, sampleY;
    if (Math.random() < goalBias) {
      sampleX = goalPt.x;
      sampleY = goalPt.y;
    } else {
      sampleX = Math.random() * canvasW;
      sampleY = Math.random() * CANVAS_HEIGHT;
    }

    // nearest node dhundho tree mein
    let nearestIdx = 0, nearestDist = Infinity;
    for (let i = 0; i < tree.length; i++) {
      const d = dist(tree[i], { x: sampleX, y: sampleY });
      if (d < nearestDist) {
        nearestDist = d;
        nearestIdx = i;
      }
    }

    const nearest = tree[nearestIdx];
    // step direction mein new point banao
    const angle = Math.atan2(sampleY - nearest.y, sampleX - nearest.x);
    const newX = nearest.x + Math.cos(angle) * stepSize;
    const newY = nearest.y + Math.sin(angle) * stepSize;

    // bounds check
    if (newX < 0 || newX > canvasW || newY < 0 || newY > CANVAS_HEIGHT) return;
    // collision check
    if (inObstacle(newX, newY)) return;
    if (lineCollides(nearest.x, nearest.y, newX, newY)) return;

    const newCost = nearest.cost + stepSize;

    if (useRRTStar) {
      // RRT* — nearby nodes mein se sabse sasta parent choose karo
      const rewireRadius = stepSize * 3;
      let bestParent = nearestIdx;
      let bestCost = newCost;

      for (let i = 0; i < tree.length; i++) {
        const d = dist(tree[i], { x: newX, y: newY });
        if (d < rewireRadius && !lineCollides(tree[i].x, tree[i].y, newX, newY)) {
          const cost = tree[i].cost + d;
          if (cost < bestCost) {
            bestCost = cost;
            bestParent = i;
          }
        }
      }

      const nodeIdx = tree.length;
      tree.push({ x: newX, y: newY, parent: bestParent, cost: bestCost });

      // rewire — nearby nodes ko naye node se connect karo agar cheaper hai
      for (let i = 0; i < tree.length - 1; i++) {
        const d = dist(tree[i], { x: newX, y: newY });
        if (d < rewireRadius) {
          const newCostVia = bestCost + d;
          if (newCostVia < tree[i].cost && !lineCollides(newX, newY, tree[i].x, tree[i].y)) {
            tree[i].parent = nodeIdx;
            tree[i].cost = newCostVia;
          }
        }
      }
    } else {
      // basic RRT — seedha add karo
      tree.push({ x: newX, y: newY, parent: nearestIdx, cost: newCost });
    }

    // goal check
    const lastNode = tree[tree.length - 1];
    if (dist(lastNode, goalPt) < goalRadius) {
      pathFound = true;
      // path trace karo goal se start tak
      rebuildPath();
    }
  }

  // path rebuild karo — goal node se parent follow karte hue start tak
  function rebuildPath() {
    path = [];
    // sabse paas wala goal node dhundho
    let bestIdx = -1, bestDist = Infinity;
    for (let i = 0; i < tree.length; i++) {
      const d = dist(tree[i], goalPt);
      if (d < goalRadius && (bestIdx === -1 || tree[i].cost < tree[bestIdx].cost)) {
        bestIdx = i;
      }
    }
    if (bestIdx === -1) return;
    let idx = bestIdx;
    while (idx !== -1) {
      path.unshift(tree[idx]);
      idx = tree[idx].parent;
    }
  }

  function resetTree() {
    goalPt.x = canvasW - 50;
    tree = [{ x: startPt.x, y: startPt.y, parent: -1, cost: 0 }];
    path = [];
    pathFound = false;
    running = true;
  }

  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // obstacles draw karo
    for (const o of obstacles) {
      ctx.fillStyle = 'rgba(100,50,50,0.6)';
      ctx.fillRect(o.x, o.y, o.w, o.h);
      ctx.strokeStyle = 'rgba(200,100,100,0.4)';
      ctx.lineWidth = 1;
      ctx.strokeRect(o.x, o.y, o.w, o.h);
    }

    // tree edges draw karo — fractal jaisa dikhta hai
    ctx.strokeStyle = 'rgba(74,158,255,0.15)';
    ctx.lineWidth = 1;
    for (let i = 1; i < tree.length; i++) {
      const node = tree[i];
      const parent = tree[node.parent];
      if (!parent) continue;
      ctx.beginPath();
      ctx.moveTo(parent.x, parent.y);
      ctx.lineTo(node.x, node.y);
      ctx.stroke();
    }

    // tree nodes — chhote dots
    ctx.fillStyle = 'rgba(74,158,255,0.25)';
    for (const node of tree) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // path highlight — agar mila toh
    if (path.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = '#4a9eff';
      ctx.lineWidth = 3;
      ctx.shadowColor = '#4a9eff';
      ctx.shadowBlur = 8;
      for (let i = 0; i < path.length; i++) {
        if (i === 0) ctx.moveTo(path[i].x, path[i].y);
        else ctx.lineTo(path[i].x, path[i].y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // start point — green
    ctx.beginPath();
    ctx.arc(startPt.x, startPt.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#4ade80';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // goal point — red circle
    ctx.beginPath();
    ctx.arc(goalPt.x, goalPt.y, goalRadius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,80,80,0.5)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(goalPt.x, goalPt.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#ff4444';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // info text
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = "11px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    const mode = useRRTStar ? 'RRT*' : 'RRT';
    ctx.fillText(`${mode}  |  Nodes: ${tree.length}  |  ${pathFound ? 'PATH FOUND' : 'Searching...'}`, 10, 18);
    if (pathFound && path.length > 0) {
      const pathCost = path[path.length - 1].cost;
      ctx.fillText(`Path cost: ${pathCost.toFixed(1)}`, 10, 32);
    }
  }

  // --- controls ---
  const stepSlider = mkSlider(ctrl, 'Step', 'rrtStep', 5, 40, stepSize, 1);
  stepSlider.addEventListener('input', () => { stepSize = parseInt(stepSlider.value); });

  const biasSlider = mkSlider(ctrl, 'Goal Bias %', 'rrtBias', 0, 50, goalBias * 100, 5);
  biasSlider.addEventListener('input', () => { goalBias = parseInt(biasSlider.value) / 100; });

  // RRT vs RRT* toggle
  const toggleBtn = mkBtn(ctrl, 'RRT', 'rrtToggle');
  toggleBtn.addEventListener('click', () => {
    useRRTStar = !useRRTStar;
    toggleBtn.textContent = useRRTStar ? 'RRT*' : 'RRT';
    toggleBtn.style.background = useRRTStar ? '#4a9eff' : '#333';
    toggleBtn.style.color = useRRTStar ? '#111' : '#ccc';
    resetTree();
  });

  const newObsBtn = mkBtn(ctrl, 'New Obstacles', 'rrtNewObs');
  newObsBtn.addEventListener('click', () => {
    generateObstacles();
    resetTree();
  });

  const resetBtn = mkBtn(ctrl, 'Reset', 'rrtReset');
  resetBtn.addEventListener('click', resetTree);

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    if (running) {
      for (let i = 0; i < nodesPerFrame; i++) {
        rrtStep();
      }
      // RRT* mein path continuously update karo
      if (useRRTStar && pathFound) rebuildPath();
      // basic RRT mein path milne ke baad ruko
      if (pathFound && !useRRTStar && tree.length > 2000) running = false;
      // RRT* mein 5000 nodes ke baad ruko
      if (useRRTStar && tree.length > 5000) running = false;
    }

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

  // init
  generateObstacles();
  resetTree();
}
