// ============================================================
// K-Means Clustering — step-by-step visualization with Voronoi
// Click se data points add karo, K-Means dekho clusters bante hue
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, clustering chalao
export function initKMeans() {
  const container = document.getElementById('kmeansContainer');
  if (!container) {
    console.warn('kmeansContainer nahi mila bhai, K-Means demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 380;
  const POINT_RADIUS = 4;
  const CENTROID_RADIUS = 10;
  const VORONOI_STEP = 8;         // voronoi grid resolution — har 8px pe check karo
  const AUTO_STEP_INTERVAL = 600; // ms mein — auto-run ka interval
  const MAX_POINTS = 500;         // itne se zyada points mat rakhna
  const ACCENT = '#4a9eff';

  // cluster colors — HSL spacing se distinct colors
  // K max 8 hai, toh 8 colors rakh lo
  const CLUSTER_COLORS = [
    { h: 210, s: 90, l: 60 },  // nila — #4a9eff types
    { h: 350, s: 85, l: 60 },  // laal
    { h: 140, s: 75, l: 50 },  // hara
    { h: 40, s: 95, l: 55 },   // peela/orange
    { h: 280, s: 80, l: 65 },  // purple
    { h: 180, s: 80, l: 50 },  // teal/cyan
    { h: 20, s: 90, l: 55 },   // orange
    { h: 320, s: 75, l: 60 },  // pink
  ];

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let points = [];       // [{x, y, cluster: -1}]
  let centroids = [];    // [{x, y}]
  let K = 3;             // default clusters
  let iteration = 0;
  let converged = false;
  let isAutoRunning = false;
  let autoRunTimer = null;
  let isVisible = false;
  let animationId = null;
  let phase = 'assign';  // 'assign' ya 'update' — alternate karte hain

  // voronoi cache — har baar pixel scan karna expensive hai
  let voronoiDirty = true;
  let voronoiImageData = null;

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:rgba(2,2,8,0.5)',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // offscreen canvas — voronoi render karne ke liye
  const offCanvas = document.createElement('canvas');
  const offCtx = offCanvas.getContext('2d');

  // --- Controls section ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // button banane ka helper — consistent styling
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:11px',
      'border-radius:6px',
      'border:1px solid rgba(74,158,255,0.25)',
      'background:rgba(74,158,255,0.08)',
      'color:#d0d0d0',
      'cursor:pointer',
      'font-family:"JetBrains Mono",monospace',
      'transition:background 0.15s,border-color 0.15s',
      'user-select:none',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(74,158,255,0.2)';
      btn.style.borderColor = 'rgba(74,158,255,0.5)';
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(74,158,255,0.08)';
        btn.style.borderColor = 'rgba(74,158,255,0.25)';
      }
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // K selector — buttons 2-8
  const kLabel = document.createElement('span');
  kLabel.textContent = 'K:';
  kLabel.style.cssText = 'color:#888;font-size:11px;font-family:"JetBrains Mono",monospace;margin-left:2px;';
  controlsDiv.appendChild(kLabel);

  const kButtons = [];
  for (let i = 2; i <= 8; i++) {
    const kBtn = createButton(String(i), () => {
      K = i;
      updateKButtons();
      resetAlgorithm();
    });
    kButtons.push({ btn: kBtn, k: i });
  }

  // K buttons ka active state update kar
  function updateKButtons() {
    kButtons.forEach(({ btn, k }) => {
      if (k === K) {
        btn.style.background = 'rgba(74,158,255,0.35)';
        btn.style.borderColor = ACCENT;
        btn.style.color = '#fff';
        btn._active = true;
      } else {
        btn.style.background = 'rgba(74,158,255,0.08)';
        btn.style.borderColor = 'rgba(74,158,255,0.25)';
        btn.style.color = '#d0d0d0';
        btn._active = false;
      }
    });
  }
  updateKButtons();

  // separator
  const sep1 = document.createElement('span');
  sep1.style.cssText = 'width:1px;height:18px;background:rgba(255,255,255,0.1);margin:0 4px;';
  controlsDiv.appendChild(sep1);

  // step button — ek iteration aage badhao
  const stepBtn = createButton('Step', () => {
    if (points.length < K) return; // points K se kam hain toh kya cluster banayega
    if (converged) return;
    runOneStep();
    voronoiDirty = true;
    requestDraw();
  });

  // auto-run button
  const autoBtn = createButton('Auto', () => {
    toggleAutoRun();
  });

  // reset centroids — naye random centroids pick kar
  const resetCentBtn = createButton('New Centroids', () => {
    resetAlgorithm();
    requestDraw();
  });

  // clear all — sab saaf
  const clearBtn = createButton('Clear', () => {
    points = [];
    centroids = [];
    iteration = 0;
    converged = false;
    phase = 'assign';
    stopAutoRun();
    voronoiDirty = true;
    requestDraw();
  });

  // separator
  const sep2 = document.createElement('span');
  sep2.style.cssText = 'width:1px;height:18px;background:rgba(255,255,255,0.1);margin:0 4px;';
  controlsDiv.appendChild(sep2);

  // preset datasets
  const presetLabel = document.createElement('span');
  presetLabel.textContent = 'Presets:';
  presetLabel.style.cssText = 'color:#888;font-size:11px;font-family:"JetBrains Mono",monospace;';
  controlsDiv.appendChild(presetLabel);

  createButton('Blobs', () => loadPreset('blobs'));
  createButton('Moons', () => loadPreset('moons'));
  createButton('Ring', () => loadPreset('ring'));
  createButton('Uniform', () => loadPreset('uniform'));

  // stats row — iteration count aur variance dikhane ke liye
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'gap:16px',
    'margin-top:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:#888',
    'flex-wrap:wrap',
    'align-items:center',
  ].join(';');
  container.appendChild(statsDiv);

  const statIter = document.createElement('span');
  const statVariance = document.createElement('span');
  const statStatus = document.createElement('span');
  const statPoints = document.createElement('span');
  statsDiv.appendChild(statIter);
  statsDiv.appendChild(statVariance);
  statsDiv.appendChild(statStatus);
  statsDiv.appendChild(statPoints);

  // --- Canvas resize ---
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    canvasW = rect.width;
    canvasH = rect.height;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // offscreen canvas bhi resize kar — voronoi ke liye
    offCanvas.width = canvas.width;
    offCanvas.height = canvas.height;
    offCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    voronoiDirty = true;
  }

  // --- Utility functions ---

  // Euclidean distance squared — sqrt nahi chahiye comparison ke liye
  function distSq(x1, y1, x2, y2) {
    const dx = x1 - x2;
    const dy = y1 - y2;
    return dx * dx + dy * dy;
  }

  // actual Euclidean distance
  function dist(x1, y1, x2, y2) {
    return Math.sqrt(distSq(x1, y1, x2, y2));
  }

  // HSL se CSS string banao
  function hsl(h, s, l, a) {
    if (a !== undefined) return `hsla(${h},${s}%,${l}%,${a})`;
    return `hsl(${h},${s}%,${l}%)`;
  }

  // cluster ka color nikal
  function clusterColor(idx, alpha) {
    const c = CLUSTER_COLORS[idx % CLUSTER_COLORS.length];
    if (alpha !== undefined) return hsl(c.h, c.s, c.l, alpha);
    return hsl(c.h, c.s, c.l);
  }

  // cluster ka RGB nikal — voronoi pixel manipulation ke liye
  function clusterRGB(idx) {
    const c = CLUSTER_COLORS[idx % CLUSTER_COLORS.length];
    // HSL to RGB conversion — voronoi ke liye chahiye
    const h2 = c.h / 360;
    const s2 = c.s / 100;
    const l2 = c.l / 100;
    let r, g, b;
    if (s2 === 0) {
      r = g = b = l2;
    } else {
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };
      const q = l2 < 0.5 ? l2 * (1 + s2) : l2 + s2 - l2 * s2;
      const p = 2 * l2 - q;
      r = hue2rgb(p, q, h2 + 1 / 3);
      g = hue2rgb(p, q, h2);
      b = hue2rgb(p, q, h2 - 1 / 3);
    }
    return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
  }

  // Gaussian random — preset datasets ke liye
  function gaussRandom(mean, sigma) {
    let u1 = Math.random();
    let u2 = Math.random();
    while (u1 === 0) u1 = Math.random();
    return mean + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // --- Nearest centroid dhundho kisi point ke liye ---
  function nearestCentroid(px, py) {
    let minD = Infinity;
    let minIdx = 0;
    for (let i = 0; i < centroids.length; i++) {
      const d = distSq(px, py, centroids[i].x, centroids[i].y);
      if (d < minD) {
        minD = d;
        minIdx = i;
      }
    }
    return minIdx;
  }

  // --- K-Means Algorithm Steps ---

  // centroids randomly initialize kar — K-Means++ nahi, simple random pick from points
  function initCentroids() {
    if (points.length < K) return;

    // K-Means++ initialization — smarter starting positions
    centroids = [];
    // pehla centroid random point se pick kar
    const firstIdx = Math.floor(Math.random() * points.length);
    centroids.push({ x: points[firstIdx].x, y: points[firstIdx].y });

    // baaki centroids — probability proportional to distance squared
    for (let c = 1; c < K; c++) {
      // har point ka nearest centroid se distance calculate kar
      const distances = points.map(p => {
        let minD = Infinity;
        for (const cent of centroids) {
          const d = distSq(p.x, p.y, cent.x, cent.y);
          if (d < minD) minD = d;
        }
        return minD;
      });

      // cumulative distribution banao
      const total = distances.reduce((sum, d) => sum + d, 0);
      if (total === 0) {
        // sab points same jagah hain — random pick kar
        const idx = Math.floor(Math.random() * points.length);
        centroids.push({ x: points[idx].x, y: points[idx].y });
        continue;
      }

      const r = Math.random() * total;
      let cumSum = 0;
      for (let i = 0; i < distances.length; i++) {
        cumSum += distances[i];
        if (cumSum >= r) {
          centroids.push({ x: points[i].x, y: points[i].y });
          break;
        }
      }
    }

    // initial assignment bhi kar do
    points.forEach(p => { p.cluster = -1; });
    iteration = 0;
    converged = false;
    phase = 'assign';
    voronoiDirty = true;
  }

  // Assignment step — har point ko nearest centroid assign kar
  function assignStep() {
    let anyChanged = false;
    for (const p of points) {
      const newCluster = nearestCentroid(p.x, p.y);
      if (p.cluster !== newCluster) {
        anyChanged = true;
        p.cluster = newCluster;
      }
    }
    return anyChanged;
  }

  // Update step — centroids ko cluster mean pe move kar
  function updateStep() {
    let totalMovement = 0;
    for (let i = 0; i < K; i++) {
      const clusterPoints = points.filter(p => p.cluster === i);
      if (clusterPoints.length === 0) {
        // empty cluster — farthest point se re-initialize kar
        // ye K-Means ka known issue hai — empty clusters
        let maxD = -1;
        let farthestIdx = 0;
        for (let j = 0; j < points.length; j++) {
          const d = distSq(points[j].x, points[j].y,
            centroids[points[j].cluster].x, centroids[points[j].cluster].y);
          if (d > maxD) {
            maxD = d;
            farthestIdx = j;
          }
        }
        const oldX = centroids[i].x;
        const oldY = centroids[i].y;
        centroids[i].x = points[farthestIdx].x;
        centroids[i].y = points[farthestIdx].y;
        totalMovement += dist(oldX, oldY, centroids[i].x, centroids[i].y);
        continue;
      }

      // mean calculate kar
      const meanX = clusterPoints.reduce((s, p) => s + p.x, 0) / clusterPoints.length;
      const meanY = clusterPoints.reduce((s, p) => s + p.y, 0) / clusterPoints.length;

      totalMovement += dist(centroids[i].x, centroids[i].y, meanX, meanY);
      centroids[i].x = meanX;
      centroids[i].y = meanY;
    }
    return totalMovement;
  }

  // ek full step chala — assign then update, alternate karte hain
  function runOneStep() {
    if (points.length < K || centroids.length === 0) return;
    if (converged) return;

    if (phase === 'assign') {
      const changed = assignStep();
      phase = 'update';
      if (!changed && iteration > 0) {
        // assignments change nahi hue — converge ho gaya
        converged = true;
      }
    } else {
      const movement = updateStep();
      iteration++;
      phase = 'assign';
      voronoiDirty = true;

      // convergence check — centroids bahut kam hile
      if (movement < 0.5) {
        // ek aur assign check kar
        const changed = assignStep();
        if (!changed) {
          converged = true;
        }
        phase = 'update';
      }
    }

    updateStats();
  }

  // --- Within-cluster variance calculate kar ---
  function calcVariance() {
    if (centroids.length === 0 || points.length === 0) return 0;
    let totalVar = 0;
    for (const p of points) {
      if (p.cluster >= 0 && p.cluster < centroids.length) {
        totalVar += distSq(p.x, p.y, centroids[p.cluster].x, centroids[p.cluster].y);
      }
    }
    return totalVar / points.length;
  }

  // --- Stats update ---
  function updateStats() {
    statIter.textContent = 'Iter: ' + iteration;
    const variance = calcVariance();
    statVariance.textContent = 'WCSS: ' + (variance > 0 ? variance.toFixed(1) : '-');
    statStatus.textContent = converged ? 'Converged' : (centroids.length > 0 ? 'Running' : 'Ready');
    statStatus.style.color = converged ? '#4ade80' : (centroids.length > 0 ? ACCENT : '#888');
    statPoints.textContent = 'Points: ' + points.length;
  }

  // --- Auto-run toggle ---
  function toggleAutoRun() {
    if (isAutoRunning) {
      stopAutoRun();
    } else {
      startAutoRun();
    }
  }

  function startAutoRun() {
    if (points.length < K) return;
    if (centroids.length === 0) initCentroids();
    isAutoRunning = true;
    autoBtn.textContent = 'Pause';
    autoBtn.style.background = 'rgba(74,158,255,0.3)';
    autoBtn.style.borderColor = ACCENT;
    autoBtn._active = true;

    autoRunTimer = setInterval(() => {
      if (converged) {
        stopAutoRun();
        return;
      }
      runOneStep();
      voronoiDirty = true;
      requestDraw();
    }, AUTO_STEP_INTERVAL);
  }

  function stopAutoRun() {
    isAutoRunning = false;
    autoBtn.textContent = 'Auto';
    autoBtn.style.background = 'rgba(74,158,255,0.08)';
    autoBtn.style.borderColor = 'rgba(74,158,255,0.25)';
    autoBtn._active = false;
    if (autoRunTimer) {
      clearInterval(autoRunTimer);
      autoRunTimer = null;
    }
  }

  // --- Algorithm reset — centroids re-init karo, assignments clear karo ---
  function resetAlgorithm() {
    stopAutoRun();
    if (points.length >= K) {
      initCentroids();
    } else {
      centroids = [];
      iteration = 0;
      converged = false;
      phase = 'assign';
    }
    voronoiDirty = true;
    updateStats();
  }

  // --- Preset datasets ---
  function loadPreset(type) {
    stopAutoRun();
    points = [];
    centroids = [];
    iteration = 0;
    converged = false;
    phase = 'assign';

    // canvas ke andar points generate kar — padding rakh edges se
    const pad = 40;
    const w = canvasW - 2 * pad;
    const h = canvasH - 2 * pad;
    const cx = canvasW / 2;
    const cy = canvasH / 2;

    if (type === 'blobs') {
      // K clusters ke blobs — random centers around which Gaussian points
      const numPerBlob = Math.floor(150 / K);
      for (let c = 0; c < K; c++) {
        // cluster center random jagah rakh
        const bx = pad + Math.random() * w;
        const by = pad + Math.random() * h;
        const spread = Math.min(w, h) * 0.1;
        for (let i = 0; i < numPerBlob; i++) {
          points.push({
            x: gaussRandom(bx, spread),
            y: gaussRandom(by, spread),
            cluster: -1,
          });
        }
      }
    } else if (type === 'moons') {
      // 2 crescent moons — interleaving arcs
      const n = 80;
      const radius = Math.min(w, h) * 0.3;
      for (let i = 0; i < n; i++) {
        const angle = (Math.PI * i) / n;
        const noise = gaussRandom(0, radius * 0.08);
        // upper crescent
        points.push({
          x: cx + radius * Math.cos(angle) + gaussRandom(0, 5),
          y: cy - radius * 0.3 + radius * Math.sin(angle) * 0.6 + noise,
          cluster: -1,
        });
        // lower crescent — shifted aur flipped
        points.push({
          x: cx + radius * 0.5 - radius * Math.cos(angle) + gaussRandom(0, 5),
          y: cy + radius * 0.3 - radius * Math.sin(angle) * 0.6 + noise,
          cluster: -1,
        });
      }
    } else if (type === 'ring') {
      // concentric rings — andar wala aur baahar wala
      const innerR = Math.min(w, h) * 0.12;
      const outerR = Math.min(w, h) * 0.35;
      // inner cluster — small blob in center
      for (let i = 0; i < 60; i++) {
        const angle = Math.random() * 2 * Math.PI;
        const r = gaussRandom(0, innerR * 0.4);
        points.push({
          x: cx + Math.cos(angle) * Math.abs(r),
          y: cy + Math.sin(angle) * Math.abs(r),
          cluster: -1,
        });
      }
      // outer ring
      for (let i = 0; i < 120; i++) {
        const angle = Math.random() * 2 * Math.PI;
        const r = outerR + gaussRandom(0, outerR * 0.08);
        points.push({
          x: cx + Math.cos(angle) * r,
          y: cy + Math.sin(angle) * r,
          cluster: -1,
        });
      }
    } else if (type === 'uniform') {
      // uniform random spread — K-Means ko challenge karo
      for (let i = 0; i < 180; i++) {
        points.push({
          x: pad + Math.random() * w,
          y: pad + Math.random() * h,
          cluster: -1,
        });
      }
    }

    // points canvas ke andar clamp kar do — bahar nahi jaane chahiye
    for (const p of points) {
      p.x = Math.max(5, Math.min(canvasW - 5, p.x));
      p.y = Math.max(5, Math.min(canvasH - 5, p.y));
    }

    voronoiDirty = true;
    updateStats();
    requestDraw();
  }

  // --- Voronoi background render kar ---
  // low-res grid scan — har VORONOI_STEP pixel pe nearest centroid check kar
  function renderVoronoi() {
    if (centroids.length === 0) {
      voronoiImageData = null;
      return;
    }

    // offscreen canvas pe draw kar
    offCtx.clearRect(0, 0, canvasW, canvasH);

    // precompute cluster RGB values
    const rgbs = [];
    for (let i = 0; i < K; i++) {
      rgbs.push(clusterRGB(i));
    }

    // grid scan — har VORONOI_STEP px pe check kar nearest centroid
    // fir us block ko color kar — fast approximation of Voronoi
    for (let y = 0; y < canvasH; y += VORONOI_STEP) {
      for (let x = 0; x < canvasW; x += VORONOI_STEP) {
        // block ke center pe check kar
        const cx = x + VORONOI_STEP / 2;
        const cy = y + VORONOI_STEP / 2;
        const idx = nearestCentroid(cx, cy);
        const rgb = rgbs[idx];

        offCtx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},0.08)`;
        offCtx.fillRect(x, y, VORONOI_STEP, VORONOI_STEP);
      }
    }

    // voronoi boundaries bhi draw kar — jahan cluster change hota hai
    offCtx.strokeStyle = 'rgba(255,255,255,0.04)';
    offCtx.lineWidth = 1;
    for (let y = 0; y < canvasH; y += VORONOI_STEP) {
      for (let x = 0; x < canvasW; x += VORONOI_STEP) {
        const cx = x + VORONOI_STEP / 2;
        const cy = y + VORONOI_STEP / 2;
        const idx = nearestCentroid(cx, cy);

        // right neighbor check
        if (x + VORONOI_STEP < canvasW) {
          const rightIdx = nearestCentroid(cx + VORONOI_STEP, cy);
          if (rightIdx !== idx) {
            offCtx.beginPath();
            offCtx.moveTo(x + VORONOI_STEP, y);
            offCtx.lineTo(x + VORONOI_STEP, y + VORONOI_STEP);
            offCtx.stroke();
          }
        }
        // bottom neighbor check
        if (y + VORONOI_STEP < canvasH) {
          const bottomIdx = nearestCentroid(cx, cy + VORONOI_STEP);
          if (bottomIdx !== idx) {
            offCtx.beginPath();
            offCtx.moveTo(x, y + VORONOI_STEP);
            offCtx.lineTo(x + VORONOI_STEP, y + VORONOI_STEP);
            offCtx.stroke();
          }
        }
      }
    }

    voronoiDirty = false;
  }

  // --- Drawing functions ---

  // centroid diamond/star draw kar — bada aur glowing
  function drawCentroid(x, y, clusterIdx) {
    const color = CLUSTER_COLORS[clusterIdx % CLUSTER_COLORS.length];
    const colorStr = hsl(color.h, color.s, color.l);
    const r = CENTROID_RADIUS;

    // glow effect
    ctx.shadowColor = colorStr;
    ctx.shadowBlur = 12;

    // diamond shape — rotated square
    ctx.beginPath();
    ctx.moveTo(x, y - r);          // top
    ctx.lineTo(x + r * 0.7, y);    // right
    ctx.lineTo(x, y + r);          // bottom
    ctx.lineTo(x - r * 0.7, y);    // left
    ctx.closePath();

    ctx.fillStyle = colorStr;
    ctx.fill();

    // inner highlight — depth ke liye
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.moveTo(x, y - r * 0.5);
    ctx.lineTo(x + r * 0.35, y);
    ctx.lineTo(x, y + r * 0.5);
    ctx.lineTo(x - r * 0.35, y);
    ctx.closePath();
    ctx.fillStyle = hsl(color.h, color.s, Math.min(90, color.l + 25), 0.6);
    ctx.fill();

    // border
    ctx.beginPath();
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r * 0.7, y);
    ctx.lineTo(x, y + r);
    ctx.lineTo(x - r * 0.7, y);
    ctx.closePath();
    ctx.strokeStyle = hsl(color.h, color.s, Math.min(90, color.l + 15), 0.8);
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // data point draw kar — colored circle
  function drawPoint(p) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, POINT_RADIUS, 0, Math.PI * 2);

    if (p.cluster >= 0 && centroids.length > 0) {
      // assigned hai — cluster ka color de
      ctx.fillStyle = clusterColor(p.cluster, 0.8);
    } else {
      // unassigned — grey dikhao
      ctx.fillStyle = 'rgba(180,180,180,0.6)';
    }
    ctx.fill();
  }

  // poora scene draw kar
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // voronoi background — agar centroids hain toh
    if (centroids.length > 0) {
      if (voronoiDirty) renderVoronoi();
      // offscreen canvas se copy kar
      ctx.drawImage(offCanvas, 0, 0, canvasW * dpr, canvasH * dpr,
        0, 0, canvasW, canvasH);
    }

    // connection lines — point se centroid tak subtle lines
    if (centroids.length > 0) {
      ctx.lineWidth = 0.5;
      for (const p of points) {
        if (p.cluster >= 0 && p.cluster < centroids.length) {
          ctx.strokeStyle = clusterColor(p.cluster, 0.1);
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(centroids[p.cluster].x, centroids[p.cluster].y);
          ctx.stroke();
        }
      }
    }

    // data points draw kar
    for (const p of points) {
      drawPoint(p);
    }

    // centroids draw kar — points ke upar taaki visible rahen
    ctx.shadowBlur = 0;
    for (let i = 0; i < centroids.length; i++) {
      drawCentroid(centroids[i].x, centroids[i].y, i);
    }
    ctx.shadowBlur = 0;

    // hint text agar koi points nahi hain
    if (points.length === 0) {
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Click to add points, or try a preset', canvasW / 2, canvasH / 2);
    }

    // iteration badge — top-right corner mein
    if (centroids.length > 0) {
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillStyle = converged ? 'rgba(74,222,128,0.7)' : 'rgba(74,158,255,0.5)';
      const badge = converged ? 'Converged (iter ' + iteration + ')' : 'Iteration ' + iteration;
      ctx.fillText(badge, canvasW - 10, 10);

      // phase indicator — kya step hai next
      if (!converged) {
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.fillText('Next: ' + (phase === 'assign' ? 'Assign' : 'Update'), canvasW - 10, 26);
      }
    }

    // legend — bottom-left
    if (centroids.length > 0) {
      const lx = 10;
      let ly = canvasH - 10 - (Math.min(K, 4) - 1) * 16;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';

      const showK = Math.min(K, 4); // zyada K hai toh sirf 4 dikhao
      for (let i = 0; i < showK; i++) {
        const clusterPts = points.filter(p => p.cluster === i).length;
        ctx.fillStyle = clusterColor(i, 0.8);
        ctx.beginPath();
        ctx.arc(lx + 4, ly, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        ctx.fillText('C' + i + ': ' + clusterPts + ' pts', lx + 14, ly);
        ly += 16;
      }
      if (K > 4) {
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.fillText('+ ' + (K - 4) + ' more', lx + 14, ly);
      }
    }
  }

  // draw request — debounced taaki multiple calls mein ek baar draw ho
  let drawPending = false;
  function requestDraw() {
    if (!drawPending) {
      drawPending = true;
      requestAnimationFrame(() => {
        drawPending = false;
        draw();
      });
    }
  }

  // --- Click handler — points add karo canvas pe ---
  canvas.addEventListener('click', (e) => {
    if (points.length >= MAX_POINTS) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasW / rect.width);
    const y = (e.clientY - rect.top) * (canvasH / rect.height);

    points.push({ x, y, cluster: -1 });

    // agar algorithm chal raha tha toh naya point bhi assign kar do
    if (centroids.length > 0 && !converged) {
      const nearest = nearestCentroid(x, y);
      points[points.length - 1].cluster = nearest;
      // converged nahi raha — naya point aaya hai
      converged = false;
    }

    voronoiDirty = true;
    updateStats();
    requestDraw();
  });

  // touch support bhi — mobile pe bhi kaam kare
  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (points.length >= MAX_POINTS) return;
    if (e.changedTouches.length === 0) return;

    const touch = e.changedTouches[0];
    const rect = canvas.getBoundingClientRect();
    const x = (touch.clientX - rect.left) * (canvasW / rect.width);
    const y = (touch.clientY - rect.top) * (canvasH / rect.height);

    points.push({ x, y, cluster: -1 });

    if (centroids.length > 0 && !converged) {
      const nearest = nearestCentroid(x, y);
      points[points.length - 1].cluster = nearest;
      converged = false;
    }

    voronoiDirty = true;
    updateStats();
    requestDraw();
  });

  // --- IntersectionObserver — sirf visible hone pe resources use kar ---
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const wasVisible = isVisible;
      isVisible = entry.isIntersecting;
      if (isVisible && !wasVisible) {
        resizeCanvas();
        requestDraw();
      }
    });
  }, { threshold: 0.1 });
  observer.observe(container);

  // resize listener
  window.addEventListener('resize', () => {
    if (isVisible) {
      // points ko scale karna padega naye canvas size ke hisaab se
      const oldW = canvasW;
      const oldH = canvasH;
      resizeCanvas();

      if (oldW > 0 && oldH > 0) {
        const scaleX = canvasW / oldW;
        const scaleY = canvasH / oldH;
        for (const p of points) {
          p.x *= scaleX;
          p.y *= scaleY;
        }
        for (const c of centroids) {
          c.x *= scaleX;
          c.y *= scaleY;
        }
      }

      voronoiDirty = true;
      requestDraw();
    }
  });

  // --- Initial setup ---
  resizeCanvas();
  updateStats();
  requestDraw();
}
