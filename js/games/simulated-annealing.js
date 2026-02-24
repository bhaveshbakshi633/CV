// ============================================================
// Simulated Annealing vs Greedy Hill Climbing — side by side comparison
// SA escape karta hai local minima se temperature ke sahare
// Greedy wala wahin atak jaata hai — dekho fark
// ============================================================

export function initSimulatedAnnealing() {
  const container = document.getElementById('simulatedAnnealingContainer');
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

  // --- landscapes — same functions ---
  const LANDSCAPES = {
    rastrigin: {
      name: 'Rastrigin',
      fn: (x, y) => 20 + x * x + y * y - 10 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y)),
      range: [-5.12, 5.12]
    },
    himmelblau: {
      name: 'Himmelblau',
      fn: (x, y) => { const a = x * x + y - 11; const b = x + y * y - 7; return a * a + b * b; },
      range: [-5, 5]
    },
    bowl: {
      name: 'Bowl',
      fn: (x, y) => x * x + y * y + 3 * Math.sin(3 * x) * Math.sin(3 * y),
      range: [-5, 5]
    }
  };

  // --- state ---
  let currentLandscape = 'rastrigin';
  let coolingRate = 0.995;
  let initialTemp = 100;
  let heatmapDirty = true;
  let heatmapCanvas = null;
  let iteration = 0;

  // SA state
  let saX = 0, saY = 0, saEnergy = 0, saBestX = 0, saBestY = 0, saBestE = Infinity;
  let saTemp = initialTemp;
  let saTrail = [];
  // Greedy state
  let grX = 0, grY = 0, grEnergy = 0, grBestX = 0, grBestY = 0, grBestE = Infinity;
  let grTrail = [];

  // heatmap build — offscreen canvas
  function buildHeatmap() {
    if (!heatmapCanvas) heatmapCanvas = document.createElement('canvas');
    // sirf half width use karo — side by side dikhana hai
    const halfW = Math.floor(canvasW / 2);
    const hRes = Math.min(halfW, 200);
    const hH = Math.floor(hRes * (CANVAS_HEIGHT / halfW));
    heatmapCanvas.width = hRes;
    heatmapCanvas.height = hH;
    const hctx = heatmapCanvas.getContext('2d');

    const ls = LANDSCAPES[currentLandscape];
    const fn = ls.fn, r = ls.range;

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
        const idx = (py * hRes + px) * 4;
        const tc = Math.max(0, Math.min(1, t));
        imgData.data[idx] = Math.floor(20 + tc * 180);
        imgData.data[idx + 1] = Math.floor(tc < 0.5 ? 30 + tc * 2 * 70 : 100 - (tc - 0.5) * 2 * 70);
        imgData.data[idx + 2] = Math.floor(tc < 0.5 ? 80 + (1 - tc * 2) * 100 : 20);
        imgData.data[idx + 3] = 160;
      }
    }
    hctx.putImageData(imgData, 0, 0);
    heatmapDirty = false;
  }

  // world to canvas — for given half (0 = left SA, 1 = right Greedy)
  function w2c(wx, wy, half) {
    const ls = LANDSCAPES[currentLandscape];
    const r = ls.range;
    const halfW = canvasW / 2;
    const ox = half * halfW;
    const px = ox + ((wx - r[0]) / (r[1] - r[0])) * halfW;
    const py = ((wy - r[0]) / (r[1] - r[0])) * CANVAS_HEIGHT;
    return { x: px, y: py };
  }

  function resetSearch() {
    const ls = LANDSCAPES[currentLandscape];
    const r = ls.range;
    // random starting position — dono ko same jagah se shuru karo
    const sx = r[0] + Math.random() * (r[1] - r[0]);
    const sy = r[0] + Math.random() * (r[1] - r[0]);
    const e = ls.fn(sx, sy);

    saX = sx; saY = sy; saEnergy = e;
    saBestX = sx; saBestY = sy; saBestE = e;
    saTemp = initialTemp;
    saTrail = [{ x: sx, y: sy }];

    grX = sx; grY = sy; grEnergy = e;
    grBestX = sx; grBestY = sy; grBestE = e;
    grTrail = [{ x: sx, y: sy }];

    iteration = 0;
  }

  // SA step — random neighbor propose karo, accept/reject with Boltzmann probability
  function saStep() {
    const ls = LANDSCAPES[currentLandscape];
    const r = ls.range;
    const rng = (r[1] - r[0]);
    // step size proportional to temperature — garam hone pe bade jumps
    const stepSize = rng * 0.05 * Math.sqrt(saTemp / initialTemp);
    const nx = saX + (Math.random() - 0.5) * 2 * stepSize;
    const ny = saY + (Math.random() - 0.5) * 2 * stepSize;
    // boundary check
    const cx = Math.max(r[0], Math.min(r[1], nx));
    const cy = Math.max(r[0], Math.min(r[1], ny));
    const ne = ls.fn(cx, cy);
    const dE = ne - saEnergy;

    // accept better always, worse with probability exp(-dE/T)
    if (dE < 0 || Math.random() < Math.exp(-dE / saTemp)) {
      saX = cx; saY = cy; saEnergy = ne;
      saTrail.push({ x: cx, y: cy });
      if (saTrail.length > 500) saTrail.shift();
      if (ne < saBestE) {
        saBestE = ne;
        saBestX = cx; saBestY = cy;
      }
    }
    // cool down
    saTemp *= coolingRate;
    if (saTemp < 0.001) saTemp = 0.001;
  }

  // Greedy hill climbing step — sirf better accept karo
  function greedyStep() {
    const ls = LANDSCAPES[currentLandscape];
    const r = ls.range;
    const rng = (r[1] - r[0]);
    const stepSize = rng * 0.03;
    const nx = grX + (Math.random() - 0.5) * 2 * stepSize;
    const ny = grY + (Math.random() - 0.5) * 2 * stepSize;
    const cx = Math.max(r[0], Math.min(r[1], nx));
    const cy = Math.max(r[0], Math.min(r[1], ny));
    const ne = ls.fn(cx, cy);

    // sirf better accept karo — greedy hai ye
    if (ne < grEnergy) {
      grX = cx; grY = cy; grEnergy = ne;
      grTrail.push({ x: cx, y: cy });
      if (grTrail.length > 500) grTrail.shift();
      if (ne < grBestE) {
        grBestE = ne;
        grBestX = cx; grBestY = cy;
      }
    }
  }

  // temperature gauge draw karo — side pe vertical bar
  function drawTempGauge(x, y, w, h) {
    // background
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.strokeRect(x, y, w, h);

    // temperature bar — red=hot, blue=cold
    const tNorm = Math.min(1, saTemp / initialTemp);
    const barH = tNorm * (h - 4);
    const grad = ctx.createLinearGradient(x, y + h, x, y);
    grad.addColorStop(0, '#0066ff');
    grad.addColorStop(0.5, '#ffaa00');
    grad.addColorStop(1, '#ff2222');
    ctx.fillStyle = grad;
    ctx.fillRect(x + 2, y + h - 2 - barH, w - 4, barH);

    // temperature value
    ctx.fillStyle = tNorm > 0.5 ? '#ff6644' : '#4a9eff';
    ctx.font = "9px 'JetBrains Mono',monospace";
    ctx.textAlign = 'center';
    ctx.fillText(`T:${saTemp.toFixed(1)}`, x + w / 2, y - 4);
  }

  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    const halfW = canvasW / 2;

    // heatmap dono halves pe draw karo
    if (heatmapDirty || !heatmapCanvas) buildHeatmap();
    if (heatmapCanvas) {
      ctx.drawImage(heatmapCanvas, 0, 0, halfW, CANVAS_HEIGHT);
      ctx.drawImage(heatmapCanvas, halfW, 0, halfW, CANVAS_HEIGHT);
    }

    // divider line — beech mein
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(halfW, 0);
    ctx.lineTo(halfW, CANVAS_HEIGHT);
    ctx.stroke();

    // --- SA side (left) ---
    // trail
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(74,158,255,0.2)';
    ctx.lineWidth = 1;
    for (let i = 0; i < saTrail.length; i++) {
      const c = w2c(saTrail[i].x, saTrail[i].y, 0);
      if (i === 0) ctx.moveTo(c.x, c.y); else ctx.lineTo(c.x, c.y);
    }
    ctx.stroke();
    // current position
    const saC = w2c(saX, saY, 0);
    ctx.beginPath();
    ctx.arc(saC.x, saC.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#4a9eff';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // best marker
    const saBC = w2c(saBestX, saBestY, 0);
    ctx.beginPath();
    ctx.arc(saBC.x, saBC.y, 7, 0, Math.PI * 2);
    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 2]);
    ctx.stroke();
    ctx.setLineDash([]);

    // --- Greedy side (right) ---
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255,170,0,0.2)';
    ctx.lineWidth = 1;
    for (let i = 0; i < grTrail.length; i++) {
      const c = w2c(grTrail[i].x, grTrail[i].y, 1);
      if (i === 0) ctx.moveTo(c.x, c.y); else ctx.lineTo(c.x, c.y);
    }
    ctx.stroke();
    const grC = w2c(grX, grY, 1);
    ctx.beginPath();
    ctx.arc(grC.x, grC.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#ffaa00';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    const grBC = w2c(grBestX, grBestY, 1);
    ctx.beginPath();
    ctx.arc(grBC.x, grBC.y, 7, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffaa00';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 2]);
    ctx.stroke();
    ctx.setLineDash([]);

    // temperature gauge — left side pe
    drawTempGauge(10, 50, 20, 120);

    // labels
    ctx.fillStyle = '#4a9eff';
    ctx.font = "12px 'JetBrains Mono',monospace";
    ctx.textAlign = 'center';
    ctx.fillText('Simulated Annealing', halfW / 2, 18);
    ctx.fillStyle = '#ffaa00';
    ctx.fillText('Greedy Hill Climb', halfW + halfW / 2, 18);

    // stats
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillText(`Best: ${saBestE.toFixed(3)}`, 10, CANVAS_HEIGHT - 20);
    ctx.fillText(`Iter: ${iteration}`, 10, CANVAS_HEIGHT - 8);
    ctx.textAlign = 'right';
    ctx.fillText(`Best: ${grBestE.toFixed(3)}`, canvasW - 10, CANVAS_HEIGHT - 20);
    ctx.fillText(`Iter: ${iteration}`, canvasW - 10, CANVAS_HEIGHT - 8);

    // winner indicator — kaun better hai
    if (iteration > 50) {
      const winner = saBestE < grBestE ? 'SA' : 'Greedy';
      const wColor = saBestE < grBestE ? '#4a9eff' : '#ffaa00';
      ctx.fillStyle = wColor;
      ctx.font = "11px 'JetBrains Mono',monospace";
      ctx.textAlign = 'center';
      ctx.fillText(`Leading: ${winner}`, canvasW / 2, CANVAS_HEIGHT - 8);
    }
  }

  // --- controls ---
  const selWrap = document.createElement('label');
  selWrap.style.cssText = "color:#ccc;font:12px 'JetBrains Mono',monospace";
  selWrap.textContent = 'Landscape ';
  const sel = document.createElement('select');
  sel.id = 'saLandscape';
  sel.style.cssText = "background:#222;color:#ccc;border:1px solid #555;padding:2px 6px;border-radius:4px;font:11px 'JetBrains Mono',monospace";
  Object.keys(LANDSCAPES).forEach(k => {
    const opt = document.createElement('option');
    opt.value = k; opt.textContent = LANDSCAPES[k].name;
    sel.appendChild(opt);
  });
  sel.value = currentLandscape;
  sel.addEventListener('change', () => { currentLandscape = sel.value; heatmapDirty = true; resetSearch(); });
  selWrap.appendChild(sel);
  ctrl.appendChild(selWrap);

  const coolSlider = mkSlider(ctrl, 'Cooling', 'saCooling', 0.99, 0.999, coolingRate, 0.001);
  coolSlider.addEventListener('input', () => { coolingRate = parseFloat(coolSlider.value); });

  const tempSlider = mkSlider(ctrl, 'Init Temp', 'saInitTemp', 10, 500, initialTemp, 10);
  tempSlider.addEventListener('input', () => { initialTemp = parseFloat(tempSlider.value); });

  const restartBtn = mkBtn(ctrl, 'Restart', 'saRestart');
  restartBtn.addEventListener('click', resetSearch);

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    // 5 steps per frame — smooth animation ke liye
    for (let i = 0; i < 5; i++) {
      saStep();
      greedyStep();
      iteration++;
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

  resetSearch();
}
