// ============================================================
// DBSCAN — Density-Based Spatial Clustering of Applications with Noise
// Click se points add karo ya preset datasets use karo, DBSCAN clusters bante dekho
// ============================================================

// yahi entry point hai — container dhundho, canvas banao, clustering chalao
export function initDBSCAN() {
  const container = document.getElementById('dbscanContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const POINT_R = 4;
  const MAX_POINTS = 600;
  // cluster colors — har cluster ko alag rang do
  const COLORS = [
    '#4a9eff', '#ef4444', '#22c55e', '#f59e0b', '#a855f7',
    '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#6366f1'
  ];

  let animationId = null, isVisible = false, canvasW = 0;
  // algorithm state
  let points = [];       // [{x, y, cluster: -1, type: 'unvisited'}]
  let eps = 40;          // epsilon radius
  let minPts = 4;        // minimum neighbors for core point
  let clusters = 0;      // kitne clusters bane
  let showEps = true;    // epsilon circles dikhao ya nahi
  let kmeansResult = []; // comparison ke liye K-Means result

  // --- DOM setup —  pehle saaf karo ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(74,158,255,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls row
  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  // helpers — slider aur button banane ke liye
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

  // epsilon aur minPts sliders
  const epsSlider = mkSlider(ctrl, '\u03B5', 'dbEps', 10, 100, eps, 1);
  const epsVal = document.createElement('span');
  epsVal.style.cssText = "color:#4a9eff;font:11px 'JetBrains Mono',monospace;min-width:24px";
  epsVal.textContent = eps;
  ctrl.appendChild(epsVal);

  const minPtsSlider = mkSlider(ctrl, 'minPts', 'dbMinPts', 2, 15, minPts, 1);
  const minPtsVal = document.createElement('span');
  minPtsVal.style.cssText = "color:#4a9eff;font:11px 'JetBrains Mono',monospace;min-width:18px";
  minPtsVal.textContent = minPts;
  ctrl.appendChild(minPtsVal);

  epsSlider.addEventListener('input', () => { eps = +epsSlider.value; epsVal.textContent = eps; });
  minPtsSlider.addEventListener('input', () => { minPts = +minPtsSlider.value; minPtsVal.textContent = minPts; });

  // preset buttons
  mkBtn(ctrl, 'Moons', 'dbMoons').addEventListener('click', () => loadPreset('moons'));
  mkBtn(ctrl, 'Circles', 'dbCircles').addEventListener('click', () => loadPreset('circles'));
  mkBtn(ctrl, 'Blobs', 'dbBlobs').addEventListener('click', () => loadPreset('blobs'));
  mkBtn(ctrl, 'Spiral', 'dbSpiral').addEventListener('click', () => loadPreset('spiral'));

  // action buttons
  const runBtn = mkBtn(ctrl, 'Run DBSCAN', 'dbRun');
  runBtn.style.background = 'rgba(74,158,255,0.2)';
  runBtn.style.borderColor = ACCENT;
  runBtn.addEventListener('click', runDBSCAN);

  mkBtn(ctrl, 'Compare K-Means', 'dbKmeans').addEventListener('click', runKMeansCompare);
  mkBtn(ctrl, 'Clear', 'dbClear').addEventListener('click', clearAll);

  // stats row
  const stats = document.createElement('div');
  stats.style.cssText = "font:11px 'JetBrains Mono',monospace;color:#888;margin-top:6px;";
  container.appendChild(stats);

  // --- resize ---
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- Gaussian random helper ---
  function gaussRand(m, s) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return m + s * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // --- Euclidean distance ---
  function dist(a, b) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2); }

  // --- preset datasets generate karo ---
  function loadPreset(type) {
    points = [];
    clusters = 0;
    kmeansResult = [];
    const cx = canvasW / 2, cy = CANVAS_HEIGHT / 2;
    const pad = 50;

    if (type === 'moons') {
      // do crescent moons — interleaving arcs, DBSCAN ke liye perfect
      const n = 100;
      const r = Math.min(canvasW - 2 * pad, CANVAS_HEIGHT - 2 * pad) * 0.3;
      for (let i = 0; i < n; i++) {
        const angle = Math.PI * i / n;
        // upper crescent
        points.push({ x: cx - r * 0.3 + r * Math.cos(angle) + gaussRand(0, 4), y: cy - r * 0.15 + r * Math.sin(angle) * 0.7 + gaussRand(0, 4), cluster: -1, type: 'unvisited' });
        // lower crescent — shifted aur flipped
        points.push({ x: cx + r * 0.3 - r * Math.cos(angle) + gaussRand(0, 4), y: cy + r * 0.15 - r * Math.sin(angle) * 0.7 + gaussRand(0, 4), cluster: -1, type: 'unvisited' });
      }
    } else if (type === 'circles') {
      // concentric rings — DBSCAN isko easily handle karta hai, K-Means nahi
      const innerR = 40, outerR = 120;
      for (let i = 0; i < 80; i++) {
        const a = Math.random() * 2 * Math.PI;
        const r = innerR + gaussRand(0, 8);
        points.push({ x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r, cluster: -1, type: 'unvisited' });
      }
      for (let i = 0; i < 120; i++) {
        const a = Math.random() * 2 * Math.PI;
        const r = outerR + gaussRand(0, 8);
        points.push({ x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r, cluster: -1, type: 'unvisited' });
      }
    } else if (type === 'blobs') {
      // 3 alag blobs — simple case
      const centers = [{ x: cx - 120, y: cy - 60 }, { x: cx + 100, y: cy - 40 }, { x: cx, y: cy + 80 }];
      centers.forEach(c => {
        for (let i = 0; i < 60; i++) {
          points.push({ x: gaussRand(c.x, 25), y: gaussRand(c.y, 25), cluster: -1, type: 'unvisited' });
        }
      });
      // noise points bhi daalo beech mein — DBSCAN inhe gray dikhayega
      for (let i = 0; i < 15; i++) {
        points.push({ x: pad + Math.random() * (canvasW - 2 * pad), y: pad + Math.random() * (CANVAS_HEIGHT - 2 * pad), cluster: -1, type: 'unvisited' });
      }
    } else if (type === 'spiral') {
      // do spirals — interleaving, DBSCAN ke liye challenge
      const n = 100;
      for (let i = 0; i < n; i++) {
        const t = (i / n) * 3 * Math.PI;
        const r1 = 20 + t * 12;
        // spiral 1
        points.push({ x: cx + r1 * Math.cos(t) + gaussRand(0, 5), y: cy + r1 * Math.sin(t) + gaussRand(0, 5), cluster: -1, type: 'unvisited' });
        // spiral 2 — opposite direction
        points.push({ x: cx - r1 * Math.cos(t) + gaussRand(0, 5), y: cy - r1 * Math.sin(t) + gaussRand(0, 5), cluster: -1, type: 'unvisited' });
      }
    }

    // clamp points canvas ke andar
    points.forEach(p => {
      p.x = Math.max(5, Math.min(canvasW - 5, p.x));
      p.y = Math.max(5, Math.min(CANVAS_HEIGHT - 5, p.y));
    });
    draw();
  }

  // --- DBSCAN algorithm ---
  function runDBSCAN() {
    // sab points reset karo
    points.forEach(p => { p.cluster = -1; p.type = 'unvisited'; });
    clusters = 0;
    kmeansResult = [];

    // range query — epsilon ke andar saare neighbors dhundho
    function rangeQuery(pIdx) {
      const neighbors = [];
      for (let i = 0; i < points.length; i++) {
        if (dist(points[pIdx], points[i]) <= eps) neighbors.push(i);
      }
      return neighbors;
    }

    // expandCluster — core point se cluster grow karo
    function expandCluster(pIdx, neighbors, clusterID) {
      points[pIdx].cluster = clusterID;
      points[pIdx].type = 'core';
      // queue based expansion — recursion nahi, iteration
      const queue = [...neighbors];
      const visited = new Set([pIdx]);
      while (queue.length > 0) {
        const qIdx = queue.shift();
        if (visited.has(qIdx)) continue;
        visited.add(qIdx);

        if (points[qIdx].type === 'unvisited' || points[qIdx].type === 'noise') {
          points[qIdx].cluster = clusterID;
          const qNeighbors = rangeQuery(qIdx);
          if (qNeighbors.length >= minPts) {
            points[qIdx].type = 'core';
            // naye neighbors queue mein daalo
            qNeighbors.forEach(n => { if (!visited.has(n)) queue.push(n); });
          } else {
            // border point — cluster ka hissa hai but core nahi
            if (points[qIdx].type !== 'core') points[qIdx].type = 'border';
          }
        }
      }
    }

    // main DBSCAN loop
    for (let i = 0; i < points.length; i++) {
      if (points[i].type !== 'unvisited') continue;
      const neighbors = rangeQuery(i);
      if (neighbors.length < minPts) {
        // noise point — kisi cluster mein nahi aayega
        points[i].type = 'noise';
        points[i].cluster = -1;
      } else {
        // naya cluster shuru karo
        expandCluster(i, neighbors, clusters);
        clusters++;
      }
    }

    updateStats();
    draw();
  }

  // --- K-Means for comparison ---
  function runKMeansCompare() {
    if (points.length < 3) return;
    // K = DBSCAN clusters count, minimum 2
    const K = Math.max(2, clusters || 3);
    // K-Means++ init
    const centroids = [{ ...points[Math.floor(Math.random() * points.length)] }];
    for (let c = 1; c < K; c++) {
      const dists = points.map(p => {
        let minD = Infinity;
        centroids.forEach(ct => { const d = dist(p, ct); if (d < minD) minD = d; });
        return minD * minD;
      });
      const total = dists.reduce((s, d) => s + d, 0);
      let r = Math.random() * total, cum = 0;
      for (let i = 0; i < dists.length; i++) {
        cum += dists[i];
        if (cum >= r) { centroids.push({ x: points[i].x, y: points[i].y }); break; }
      }
    }

    // 30 iterations chalao
    kmeansResult = new Array(points.length).fill(0);
    for (let iter = 0; iter < 30; iter++) {
      // assign
      points.forEach((p, i) => {
        let minD = Infinity, minC = 0;
        centroids.forEach((c, ci) => { const d = dist(p, c); if (d < minD) { minD = d; minC = ci; } });
        kmeansResult[i] = minC;
      });
      // update centroids
      for (let c = 0; c < K; c++) {
        let sx = 0, sy = 0, cnt = 0;
        points.forEach((p, i) => { if (kmeansResult[i] === c) { sx += p.x; sy += p.y; cnt++; } });
        if (cnt > 0) { centroids[c].x = sx / cnt; centroids[c].y = sy / cnt; }
      }
    }
    draw();
  }

  function clearAll() {
    points = []; clusters = 0; kmeansResult = [];
    updateStats();
    draw();
  }

  function updateStats() {
    const noise = points.filter(p => p.cluster === -1).length;
    const core = points.filter(p => p.type === 'core').length;
    stats.textContent = `Points: ${points.length}  |  Clusters: ${clusters}  |  Core: ${core}  |  Noise: ${noise}`;
  }

  // --- click se point add karo ---
  canvas.addEventListener('click', (e) => {
    if (points.length >= MAX_POINTS) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasW / rect.width);
    const y = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height);
    points.push({ x, y, cluster: -1, type: 'unvisited' });
    draw();
  });

  // --- drawing ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    const hasKmeans = kmeansResult.length === points.length;
    // agar kmeans bhi hai toh canvas split karo — left DBSCAN, right K-Means
    const splitX = hasKmeans ? canvasW / 2 : canvasW;

    // DBSCAN side — left (ya full agar K-Means nahi hai)
    ctx.save();
    if (hasKmeans) {
      // divider line
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(splitX, 0); ctx.lineTo(splitX, CANVAS_HEIGHT); ctx.stroke();
      ctx.setLineDash([]);
      // labels
      ctx.font = "12px 'JetBrains Mono',monospace";
      ctx.fillStyle = ACCENT;
      ctx.textAlign = 'center';
      ctx.fillText('DBSCAN', splitX / 2, 18);
      ctx.fillText('K-Means', splitX + splitX / 2, 18);
    }

    // epsilon circles — core points ke around halke se dikhao
    if (showEps && clusters > 0) {
      ctx.globalAlpha = 0.08;
      ctx.lineWidth = 1;
      points.forEach(p => {
        if (p.type === 'core') {
          ctx.beginPath();
          ctx.arc(hasKmeans ? p.x * (splitX / canvasW) : p.x, p.y, eps * (hasKmeans ? splitX / canvasW : 1), 0, Math.PI * 2);
          ctx.strokeStyle = p.cluster >= 0 ? COLORS[p.cluster % COLORS.length] : '#555';
          ctx.stroke();
        }
      });
      ctx.globalAlpha = 1;
    }

    // DBSCAN points draw karo
    points.forEach((p, i) => {
      const px = hasKmeans ? p.x * (splitX / canvasW) : p.x;
      const py = p.y;

      // point color — cluster ke hisaab se, noise gray
      if (p.cluster >= 0) {
        ctx.fillStyle = COLORS[p.cluster % COLORS.length];
      } else if (p.type === 'noise') {
        ctx.fillStyle = '#666';
      } else {
        ctx.fillStyle = '#aaa';
      }

      ctx.beginPath();
      ctx.arc(px, py, p.type === 'core' ? POINT_R + 1.5 : POINT_R, 0, Math.PI * 2);
      ctx.fill();

      // core points ko extra border do
      if (p.type === 'core') {
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      // noise points ko cross marker do
      if (p.type === 'noise') {
        ctx.strokeStyle = '#888';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px - 3, py - 3); ctx.lineTo(px + 3, py + 3);
        ctx.moveTo(px + 3, py - 3); ctx.lineTo(px - 3, py + 3);
        ctx.stroke();
      }
    });

    // K-Means side — right half agar compare mode on hai
    if (hasKmeans) {
      points.forEach((p, i) => {
        const px = splitX + p.x * (splitX / canvasW);
        const py = p.y;
        ctx.fillStyle = COLORS[kmeansResult[i] % COLORS.length];
        ctx.beginPath();
        ctx.arc(px, py, POINT_R, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // hint text agar koi points nahi hain
    if (points.length === 0) {
      ctx.font = "13px 'JetBrains Mono',monospace";
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Click to add points, or try a preset dataset', canvasW / 2, CANVAS_HEIGHT / 2);
    }

    ctx.restore();
    updateStats();
  }

  // --- animation loop (mostly static, redraw on demand) ---
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

  draw();
}
