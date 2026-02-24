// ============================================================
// Particle Swarm Optimization — 2D fitness landscape pe swarm
// Particles apna personal best aur global best follow karke optimize karte hain
// PSO ka social behavior dekhlo — swarm intelligence ka kamaal
// ============================================================

export function initPSO() {
  const container = document.getElementById('psoContainer');
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

  // pehle declare karo — resize mein use hoga, TDZ se bachne ke liye
  let heatmapDirty = true;
  let heatmapCanvas = null;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    heatmapDirty = true;
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

  // --- fitness landscapes ---
  const LANDSCAPES = {
    rastrigin: {
      name: 'Rastrigin',
      fn: (x, y) => 20 + x * x + y * y - 10 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y)),
      range: [-5.12, 5.12],
      optimum: { x: 0, y: 0 }
    },
    himmelblau: {
      name: 'Himmelblau',
      fn: (x, y) => {
        const a = x * x + y - 11;
        const b = x + y * y - 7;
        return a * a + b * b;
      },
      range: [-5, 5],
      optimum: { x: 3, y: 2 }
    },
    bowl: {
      name: 'Bowl',
      fn: (x, y) => x * x + y * y,
      range: [-5, 5],
      optimum: { x: 0, y: 0 }
    }
  };

  // --- state ---
  let currentLandscape = 'rastrigin';
  let inertia = 0.7;
  let swarmSize = 30;
  let particles = [];
  let gBest = null; // global best position
  let gBestVal = Infinity;
  let iteration = 0;
  const C1 = 1.5, C2 = 1.5; // cognitive aur social coefficients

  // heatmap color — blue to red through green/yellow
  function heatColor(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor(t < 0.5 ? 0 : (t - 0.5) * 2 * 255);
    const g = Math.floor(t < 0.5 ? t * 2 * 180 : (1 - t) * 2 * 180);
    const b = Math.floor(t < 0.5 ? (1 - t * 2) * 200 : 0);
    return `rgb(${r},${g},${b})`;
  }

  // world to canvas conversion
  function w2c(wx, wy) {
    const ls = LANDSCAPES[currentLandscape];
    const r = ls.range;
    const px = ((wx - r[0]) / (r[1] - r[0])) * canvasW;
    const py = ((wy - r[0]) / (r[1] - r[0])) * CANVAS_HEIGHT;
    return { x: px, y: py };
  }

  function c2w(cx, cy) {
    const ls = LANDSCAPES[currentLandscape];
    const r = ls.range;
    const wx = r[0] + (cx / canvasW) * (r[1] - r[0]);
    const wy = r[0] + (cy / CANVAS_HEIGHT) * (r[1] - r[0]);
    return { x: wx, y: wy };
  }

  // heatmap build karo — offscreen canvas pe
  function buildHeatmap() {
    if (!heatmapCanvas) heatmapCanvas = document.createElement('canvas');
    const hRes = Math.min(canvasW, 300); // resolution limit — performance ke liye
    const hH = Math.floor(hRes * (CANVAS_HEIGHT / canvasW));
    heatmapCanvas.width = hRes;
    heatmapCanvas.height = hH;
    const hctx = heatmapCanvas.getContext('2d');

    const ls = LANDSCAPES[currentLandscape];
    const fn = ls.fn, r = ls.range;

    // min/max dhundho normalization ke liye
    let minV = Infinity, maxV = -Infinity;
    for (let py = 0; py < hH; py++) {
      for (let px = 0; px < hRes; px++) {
        const wx = r[0] + (px / hRes) * (r[1] - r[0]);
        const wy = r[0] + (py / hH) * (r[1] - r[0]);
        const v = fn(wx, wy);
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
    const logMin = Math.log(minV + 1);
    const logMax = Math.log(maxV + 1);
    const logRange = logMax - logMin || 1;

    const imgData = hctx.createImageData(hRes, hH);
    for (let py = 0; py < hH; py++) {
      for (let px = 0; px < hRes; px++) {
        const wx = r[0] + (px / hRes) * (r[1] - r[0]);
        const wy = r[0] + (py / hH) * (r[1] - r[0]);
        const v = fn(wx, wy);
        const t = (Math.log(v + 1) - logMin) / logRange;
        // color gradient — dark blue to bright yellow/red
        const idx = (py * hRes + px) * 4;
        const tc = Math.max(0, Math.min(1, t));
        imgData.data[idx] = Math.floor(tc < 0.5 ? tc * 2 * 40 : 40 + (tc - 0.5) * 2 * 200);
        imgData.data[idx + 1] = Math.floor(tc < 0.5 ? 20 + tc * 2 * 80 : 100 - (tc - 0.5) * 2 * 80);
        imgData.data[idx + 2] = Math.floor(tc < 0.5 ? 80 + (1 - tc * 2) * 120 : 30 - tc * 20);
        imgData.data[idx + 3] = 180;
      }
    }
    hctx.putImageData(imgData, 0, 0);
    heatmapDirty = false;
  }

  // particles initialize karo
  function initSwarm() {
    const ls = LANDSCAPES[currentLandscape];
    const r = ls.range;
    particles = [];
    gBest = null;
    gBestVal = Infinity;
    iteration = 0;

    for (let i = 0; i < swarmSize; i++) {
      const x = r[0] + Math.random() * (r[1] - r[0]);
      const y = r[0] + Math.random() * (r[1] - r[0]);
      const val = ls.fn(x, y);
      const p = {
        x, y,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        pBestX: x, pBestY: y, pBestVal: val,
        trail: [{ x, y }]
      };
      particles.push(p);
      if (val < gBestVal) {
        gBestVal = val;
        gBest = { x, y };
      }
    }
  }

  // PSO step — velocity aur position update karo
  function psoStep() {
    const ls = LANDSCAPES[currentLandscape];
    const r = ls.range;
    const w = inertia;

    for (const p of particles) {
      const r1 = Math.random(), r2 = Math.random();
      // velocity update: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
      p.vx = w * p.vx + C1 * r1 * (p.pBestX - p.x) + C2 * r2 * (gBest.x - p.x);
      p.vy = w * p.vy + C1 * r1 * (p.pBestY - p.y) + C2 * r2 * (gBest.y - p.y);

      // velocity clamp
      const maxV = (r[1] - r[0]) * 0.1;
      p.vx = Math.max(-maxV, Math.min(maxV, p.vx));
      p.vy = Math.max(-maxV, Math.min(maxV, p.vy));

      // position update
      p.x += p.vx;
      p.y += p.vy;

      // boundary check — range ke andar rakho
      p.x = Math.max(r[0], Math.min(r[1], p.x));
      p.y = Math.max(r[0], Math.min(r[1], p.y));

      // trail mein add karo
      p.trail.push({ x: p.x, y: p.y });
      if (p.trail.length > 50) p.trail.shift();

      // fitness evaluate karo
      const val = ls.fn(p.x, p.y);

      // personal best update
      if (val < p.pBestVal) {
        p.pBestVal = val;
        p.pBestX = p.x;
        p.pBestY = p.y;
      }
      // global best update
      if (val < gBestVal) {
        gBestVal = val;
        gBest = { x: p.x, y: p.y };
      }
    }
    iteration++;
  }

  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // heatmap background draw karo
    if (heatmapDirty || !heatmapCanvas) buildHeatmap();
    if (heatmapCanvas) {
      ctx.drawImage(heatmapCanvas, 0, 0, canvasW, CANVAS_HEIGHT);
    }

    // particle trails — faded lines
    for (const p of particles) {
      if (p.trail.length < 2) continue;
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(74,158,255,0.15)';
      ctx.lineWidth = 1;
      for (let i = 0; i < p.trail.length; i++) {
        const c = w2c(p.trail[i].x, p.trail[i].y);
        if (i === 0) ctx.moveTo(c.x, c.y);
        else ctx.lineTo(c.x, c.y);
      }
      ctx.stroke();
    }

    // particles draw karo
    for (const p of particles) {
      const c = w2c(p.x, p.y);
      // velocity arrow
      const arrLen = Math.sqrt(p.vx * p.vx + p.vy * p.vy) * 20;
      if (arrLen > 2) {
        const angle = Math.atan2(p.vy, p.vx);
        const endC = w2c(p.x + p.vx * 3, p.y + p.vy * 3);
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1;
        ctx.moveTo(c.x, c.y);
        ctx.lineTo(endC.x, endC.y);
        ctx.stroke();
      }
      // particle dot
      ctx.beginPath();
      ctx.arc(c.x, c.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#4a9eff';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // personal best marker — chhota dot
      const pb = w2c(p.pBestX, p.pBestY);
      ctx.beginPath();
      ctx.arc(pb.x, pb.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(74,158,255,0.3)';
      ctx.fill();
    }

    // global best marker — bada diamond
    if (gBest) {
      const gb = w2c(gBest.x, gBest.y);
      ctx.save();
      ctx.shadowColor = '#4a9eff';
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.moveTo(gb.x, gb.y - 8);
      ctx.lineTo(gb.x + 6, gb.y);
      ctx.lineTo(gb.x, gb.y + 8);
      ctx.lineTo(gb.x - 6, gb.y);
      ctx.closePath();
      ctx.fillStyle = '#4a9eff';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();
    }

    // optimum marker
    const ls = LANDSCAPES[currentLandscape];
    const opt = w2c(ls.optimum.x, ls.optimum.y);
    ctx.beginPath();
    ctx.arc(opt.x, opt.y, 6, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // info text
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = "11px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillText(`Iter: ${iteration}  |  Best: ${gBestVal.toFixed(4)}  |  Particles: ${particles.length}`, 10, 18);
  }

  // --- controls ---
  // landscape selector
  const selWrap = document.createElement('label');
  selWrap.style.cssText = "color:#ccc;font:12px 'JetBrains Mono',monospace";
  selWrap.textContent = 'Landscape ';
  const sel = document.createElement('select');
  sel.id = 'psoLandscape';
  sel.style.cssText = "background:#222;color:#ccc;border:1px solid #555;padding:2px 6px;border-radius:4px;font:11px 'JetBrains Mono',monospace";
  Object.keys(LANDSCAPES).forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = LANDSCAPES[k].name;
    sel.appendChild(opt);
  });
  sel.value = currentLandscape;
  sel.addEventListener('change', () => {
    currentLandscape = sel.value;
    heatmapDirty = true;
    initSwarm();
  });
  selWrap.appendChild(sel);
  ctrl.appendChild(selWrap);

  const wSlider = mkSlider(ctrl, 'Inertia w', 'psoInertia', 0.1, 1.0, inertia, 0.05);
  wSlider.addEventListener('input', () => { inertia = parseFloat(wSlider.value); });

  const sizeLabel = document.createElement('span');
  sizeLabel.style.cssText = "color:#4a9eff;font:12px 'JetBrains Mono',monospace";
  sizeLabel.textContent = `N: ${swarmSize}`;
  ctrl.appendChild(sizeLabel);

  const resetBtn = mkBtn(ctrl, 'Reset', 'psoReset');
  resetBtn.addEventListener('click', () => { initSwarm(); });

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    psoStep();
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

  initSwarm();
  draw();
}
