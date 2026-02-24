// ============================================================
// Decision Tree — 2D classification with tree visualization
// Click se points add karo, decision tree boundaries aur tree structure dekho
// ============================================================

// yahi entry point hai — left mein scatter plot, right mein tree diagram
export function initDecisionTree() {
  const container = document.getElementById('decisionTreeContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  // 3 classes ke colors — distinct rakhna zaroori hai
  const CLASS_COLORS = ['#4a9eff', '#ef4444', '#22c55e'];
  const CLASS_NAMES = ['Blue', 'Red', 'Green'];

  let animationId = null, isVisible = false, canvasW = 0;
  let points = [];        // [{x, y, cls: 0|1|2}]
  let currentClass = 0;   // user kaunsa class place kar raha hai
  let maxDepth = 4;       // tree kitna deep ja sakta hai
  let minSamples = 3;     // minimum samples split karne ke liye
  let tree = null;        // trained tree structure

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(74,158,255,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  // helpers
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
  const depthSlider = mkSlider(ctrl, 'Depth', 'dtDepth', 1, 8, maxDepth, 1);
  const depthVal = document.createElement('span');
  depthVal.style.cssText = "color:#4a9eff;font:11px 'JetBrains Mono',monospace;min-width:14px";
  depthVal.textContent = maxDepth;
  ctrl.appendChild(depthVal);
  depthSlider.addEventListener('input', () => { maxDepth = +depthSlider.value; depthVal.textContent = maxDepth; });

  const minSampSlider = mkSlider(ctrl, 'Min Samples', 'dtMinSamp', 1, 20, minSamples, 1);
  const minSampVal = document.createElement('span');
  minSampVal.style.cssText = "color:#4a9eff;font:11px 'JetBrains Mono',monospace;min-width:14px";
  minSampVal.textContent = minSamples;
  ctrl.appendChild(minSampVal);
  minSampSlider.addEventListener('input', () => { minSamples = +minSampSlider.value; minSampVal.textContent = minSamples; });

  // class select button — cycle karta hai 3 classes mein
  const classBtn = mkBtn(ctrl, 'Class: Blue', 'dtClass');
  classBtn.style.color = CLASS_COLORS[0];
  classBtn.style.borderColor = CLASS_COLORS[0];
  classBtn.addEventListener('click', () => {
    currentClass = (currentClass + 1) % 3;
    classBtn.textContent = 'Class: ' + CLASS_NAMES[currentClass];
    classBtn.style.color = CLASS_COLORS[currentClass];
    classBtn.style.borderColor = CLASS_COLORS[currentClass];
  });

  const growBtn = mkBtn(ctrl, 'Grow Tree', 'dtGrow');
  growBtn.style.background = 'rgba(74,158,255,0.2)';
  growBtn.style.borderColor = ACCENT;
  growBtn.addEventListener('click', growTree);

  mkBtn(ctrl, 'Clear', 'dtClear').addEventListener('click', () => { points = []; tree = null; draw(); });

  // stats
  const stats = document.createElement('div');
  stats.style.cssText = "font:11px 'JetBrains Mono',monospace;color:#888;margin-top:6px;";
  container.appendChild(stats);

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- Gini impurity calculate karo ---
  function gini(samples) {
    if (samples.length === 0) return 0;
    const counts = [0, 0, 0];
    samples.forEach(p => counts[p.cls]++);
    let imp = 1;
    counts.forEach(c => { const p = c / samples.length; imp -= p * p; });
    return imp;
  }

  // --- best split dhundho — axis-aligned, Gini minimise karo ---
  function findBestSplit(samples, depth) {
    if (samples.length < minSamples || depth >= maxDepth || gini(samples) === 0) return null;

    let bestGain = 0, bestFeature = null, bestThreshold = null;
    const parentGini = gini(samples);

    // x aur y dono features pe try karo
    for (const feature of ['x', 'y']) {
      // unique values sort karo, beech ke thresholds try karo
      const vals = samples.map(p => p[feature]).sort((a, b) => a - b);
      for (let i = 1; i < vals.length; i++) {
        if (vals[i] === vals[i - 1]) continue;
        const threshold = (vals[i] + vals[i - 1]) / 2;
        const left = samples.filter(p => p[feature] <= threshold);
        const right = samples.filter(p => p[feature] > threshold);
        if (left.length < 1 || right.length < 1) continue;

        // weighted gini calculate karo — information gain
        const wGini = (left.length * gini(left) + right.length * gini(right)) / samples.length;
        const gain = parentGini - wGini;
        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = feature;
          bestThreshold = threshold;
        }
      }
    }

    return bestGain > 0.001 ? { feature: bestFeature, threshold: bestThreshold } : null;
  }

  // --- tree recursively build karo ---
  function buildTree(samples, depth) {
    // majority class dhundho — ye leaf ka prediction hoga
    const counts = [0, 0, 0];
    samples.forEach(p => counts[p.cls]++);
    const majority = counts.indexOf(Math.max(...counts));

    const split = findBestSplit(samples, depth);
    if (!split) {
      // leaf node — predict majority class
      return { leaf: true, cls: majority, count: samples.length, counts, gini: gini(samples) };
    }

    const left = samples.filter(p => p[split.feature] <= split.threshold);
    const right = samples.filter(p => p[split.feature] > split.threshold);

    return {
      leaf: false,
      feature: split.feature,
      threshold: split.threshold,
      left: buildTree(left, depth + 1),
      right: buildTree(right, depth + 1),
      count: samples.length,
      counts,
      gini: gini(samples),
      depth
    };
  }

  // --- tree predict karo ek point ke liye ---
  function predict(node, x, y) {
    if (node.leaf) return node.cls;
    const val = node.feature === 'x' ? x : y;
    return val <= node.threshold ? predict(node.left, x, y) : predict(node.right, x, y);
  }

  // --- tree grow karo ---
  function growTree() {
    if (points.length < 3) return;
    tree = buildTree(points, 0);
    draw();
  }

  // --- tree ki depth count karo ---
  function treeDepth(node) {
    if (!node || node.leaf) return 0;
    return 1 + Math.max(treeDepth(node.left), treeDepth(node.right));
  }
  // --- tree ke total nodes count karo ---
  function treeNodes(node) {
    if (!node) return 0;
    if (node.leaf) return 1;
    return 1 + treeNodes(node.left) + treeNodes(node.right);
  }

  // --- click se points add karo (left half only) ---
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasW / rect.width);
    const y = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height);
    // sirf left half mein points add karo
    if (x > canvasW / 2) return;
    // scale x to 0-1 range (scatter plot coordinates)
    const sx = x / (canvasW / 2);
    const sy = y / CANVAS_HEIGHT;
    points.push({ x: sx, y: sy, cls: currentClass });
    draw();
  });

  // --- tree diagram draw karo (right half mein) ---
  function drawTreeDiagram(node, x, y, w, depth) {
    if (!node) return;
    const nodeR = 14;
    const levelH = 42;

    // node circle draw karo
    if (node.leaf) {
      // leaf — filled with class color
      ctx.fillStyle = CLASS_COLORS[node.cls];
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(x, y, nodeR, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = CLASS_COLORS[node.cls];
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // count label
      ctx.fillStyle = '#fff';
      ctx.font = "9px 'JetBrains Mono',monospace";
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.count, x, y);
    } else {
      // internal node — split info dikhao
      ctx.fillStyle = '#222';
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, nodeR, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      // split label
      ctx.fillStyle = '#ccc';
      ctx.font = "8px 'JetBrains Mono',monospace";
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const feat = node.feature === 'x' ? 'X' : 'Y';
      ctx.fillText(feat + '<' + node.threshold.toFixed(2), x, y);

      // children draw karo
      const childW = w / 2;
      const lx = x - childW / 2;
      const rx = x + childW / 2;
      const cy = y + levelH;

      // edges — connecting lines
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, y + nodeR); ctx.lineTo(lx, cy - nodeR); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x, y + nodeR); ctx.lineTo(rx, cy - nodeR); ctx.stroke();

      drawTreeDiagram(node.left, lx, cy, childW, depth + 1);
      drawTreeDiagram(node.right, rx, cy, childW, depth + 1);
    }
  }

  // --- main draw ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    const halfW = canvasW / 2;

    // --- LEFT HALF: scatter plot with decision boundaries ---
    ctx.save();

    // agar tree hai toh decision boundary draw karo — pixelated background
    if (tree) {
      const step = 4;
      for (let py = 0; py < CANVAS_HEIGHT; py += step) {
        for (let px = 0; px < halfW; px += step) {
          const sx = px / halfW;
          const sy = py / CANVAS_HEIGHT;
          const cls = predict(tree, sx, sy);
          ctx.fillStyle = CLASS_COLORS[cls];
          ctx.globalAlpha = 0.12;
          ctx.fillRect(px, py, step, step);
        }
      }
      ctx.globalAlpha = 1;

      // decision boundaries — jahan class change hoti hai wahan line draw karo
      drawBoundaries(tree, 0, 0, 1, 1);
    }

    // data points draw karo — left half mein
    points.forEach(p => {
      const px = p.x * halfW;
      const py = p.y * CANVAS_HEIGHT;
      ctx.fillStyle = CLASS_COLORS[p.cls];
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // hint text
    if (points.length === 0) {
      ctx.font = "13px 'JetBrains Mono',monospace";
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Click left half to add points', halfW / 2, CANVAS_HEIGHT / 2);
    }

    ctx.restore();

    // --- DIVIDER ---
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(halfW, 0); ctx.lineTo(halfW, CANVAS_HEIGHT); ctx.stroke();
    ctx.setLineDash([]);

    // labels
    ctx.font = "11px 'JetBrains Mono',monospace";
    ctx.fillStyle = ACCENT;
    ctx.textAlign = 'center';
    ctx.fillText('Decision Boundaries', halfW / 2, 16);
    ctx.fillText('Tree Structure', halfW + halfW / 2, 16);

    // --- RIGHT HALF: tree diagram ---
    if (tree) {
      const treeW = halfW - 40;
      drawTreeDiagram(tree, halfW + halfW / 2, 50, treeW, 0);
    } else {
      ctx.fillStyle = 'rgba(255,255,255,0.2)';
      ctx.textAlign = 'center';
      ctx.fillText('Grow tree to see structure', halfW + halfW / 2, CANVAS_HEIGHT / 2);
    }

    // stats update karo
    if (tree) {
      const d = treeDepth(tree);
      const n = treeNodes(tree);
      stats.textContent = `Points: ${points.length}  |  Tree Depth: ${d}  |  Nodes: ${n}  |  Gini(root): ${tree.gini.toFixed(3)}`;
    } else {
      stats.textContent = `Points: ${points.length}  |  Add points & click Grow Tree`;
    }
  }

  // --- decision boundaries draw karo — recursive splits as lines ---
  function drawBoundaries(node, xMin, yMin, xMax, yMax) {
    if (!node || node.leaf) return;
    const halfW = canvasW / 2;

    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1;

    if (node.feature === 'x') {
      // vertical split line
      const px = node.threshold * halfW;
      const pyMin = yMin * CANVAS_HEIGHT;
      const pyMax = yMax * CANVAS_HEIGHT;
      ctx.beginPath(); ctx.moveTo(px, pyMin); ctx.lineTo(px, pyMax); ctx.stroke();
      // recurse — left gets xMin..threshold, right gets threshold..xMax
      drawBoundaries(node.left, xMin, yMin, node.threshold, yMax);
      drawBoundaries(node.right, node.threshold, yMin, xMax, yMax);
    } else {
      // horizontal split line
      const py = node.threshold * CANVAS_HEIGHT;
      const pxMin = xMin * halfW;
      const pxMax = xMax * halfW;
      ctx.beginPath(); ctx.moveTo(pxMin, py); ctx.lineTo(pxMax, py); ctx.stroke();
      // recurse — left gets yMin..threshold, right gets threshold..yMax
      drawBoundaries(node.left, xMin, yMin, xMax, node.threshold);
      drawBoundaries(node.right, xMin, node.threshold, xMax, yMax);
    }
  }

  // --- animation loop (mostly static) ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    animationId = requestAnimationFrame(loop);
  }

  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible) { draw(); if (!animationId) loop(); }
    else if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && isVisible && !animationId) loop(); });

  draw();
}
