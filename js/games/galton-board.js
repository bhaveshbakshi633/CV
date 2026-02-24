// ============================================================
// Galton Board — balls pegs se bounce karke normal distribution banate hain
// Central Limit Theorem ka physical demo — bias slider ke saath
// ============================================================

export function initGaltonBoard() {
  const container = document.getElementById('galtonBoardContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  let animationId = null, isVisible = false, canvasW = 0;

  // parameters
  let numRows = 10;
  let ballRate = 3; // frames ke beech kitne balls spawn honge
  let bias = 0.5; // 0.5 = fair, <0.5 = left bias, >0.5 = right bias

  // state
  let balls = []; // { x, y, vx, vy, row, bin, settled }
  let bins = []; // har bin mein kitni balls hain
  let frameCount = 0;

  // peg layout
  let pegSpacingX = 0, pegSpacingY = 0, pegStartY = 0, pegRadius = 3;
  let binWidth = 0, binStartY = 0;

  // --- DOM banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(245,158,11,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  function createSlider(label, min, max, step, val, onChange) {
    const w = document.createElement('div');
    w.style.cssText = 'display:flex;align-items:center;gap:5px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#6b6b6b;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    lbl.textContent = label;
    w.appendChild(lbl);
    const sl = document.createElement('input');
    sl.type = 'range'; sl.min = String(min); sl.max = String(max); sl.step = String(step); sl.value = String(val);
    sl.style.cssText = 'width:70px;height:4px;accent-color:#f59e0b;cursor:pointer;';
    w.appendChild(sl);
    const vl = document.createElement('span');
    vl.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:28px;";
    vl.textContent = step < 1 ? Number(val).toFixed(2) : String(val);
    w.appendChild(vl);
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      vl.textContent = step < 1 ? v.toFixed(2) : String(Math.round(v));
      onChange(v);
    });
    ctrl.appendChild(w);
  }

  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = "padding:5px 12px;font-size:11px;border-radius:6px;cursor:pointer;background:rgba(245,158,11,0.1);color:#b0b0b0;border:1px solid rgba(245,158,11,0.25);font-family:'JetBrains Mono',monospace;transition:all 0.2s ease;";
    btn.addEventListener('mouseenter', () => { btn.style.background = 'rgba(245,158,11,0.25)'; btn.style.color = '#e0e0e0'; });
    btn.addEventListener('mouseleave', () => { btn.style.background = 'rgba(245,158,11,0.1)'; btn.style.color = '#b0b0b0'; });
    btn.addEventListener('click', onClick);
    ctrl.appendChild(btn);
  }

  createSlider('rows', 5, 15, 1, numRows, (v) => { numRows = Math.round(v); resetBoard(); });
  createSlider('rate', 1, 8, 1, ballRate, (v) => { ballRate = Math.round(v); });
  createSlider('bias', 0.1, 0.9, 0.05, bias, (v) => { bias = v; });
  createButton('Reset', resetBoard);

  function resetBoard() {
    balls = [];
    bins = new Array(numRows + 1).fill(0);
    frameCount = 0;
  }
  resetBoard();

  // --- layout compute karo ---
  function computeLayout() {
    const numBins = numRows + 1;
    pegSpacingY = (CANVAS_HEIGHT * 0.55) / numRows;
    pegStartY = 50;
    pegSpacingX = canvasW / (numRows + 2);
    binWidth = canvasW / (numBins + 1);
    binStartY = pegStartY + numRows * pegSpacingY + 30;
    pegRadius = Math.max(2, Math.min(4, pegSpacingX * 0.12));
  }

  // --- peg position nikalo ---
  function pegPos(row, col) {
    // har row mein row+1 pegs hain, centered
    const numPegs = row + 1;
    const totalWidth = (numPegs - 1) * pegSpacingX;
    const startX = (canvasW - totalWidth) / 2;
    return {
      x: startX + col * pegSpacingX,
      y: pegStartY + row * pegSpacingY,
    };
  }

  // --- resize ---
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    computeLayout();
  }
  resize();
  window.addEventListener('resize', resize);

  // --- ball spawn karo ---
  function spawnBall() {
    // top center se start — thodi random offset
    balls.push({
      x: canvasW / 2 + (Math.random() - 0.5) * 4,
      y: 10,
      vx: 0,
      vy: 0,
      row: -1, // abhi kisi row pe nahi
      path: [], // konse pegs se guzri — left/right
      settled: false,
      bin: -1,
    });
  }

  // --- update balls ---
  function updateBalls() {
    const gravity = 0.3;
    const bounce = 0.4;

    for (let i = balls.length - 1; i >= 0; i--) {
      const b = balls[i];
      if (b.settled) continue;

      // gravity apply karo
      b.vy += gravity;
      b.x += b.vx;
      b.y += b.vy;

      // peg collision check karo
      for (let row = 0; row < numRows; row++) {
        const numPegs = row + 1;
        for (let col = 0; col < numPegs; col++) {
          const peg = pegPos(row, col);
          const dx = b.x - peg.x;
          const dy = b.y - peg.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const minDist = pegRadius + 3; // ball radius + peg radius

          if (dist < minDist && dist > 0) {
            // bounce direction — normalize karke push karo
            const nx = dx / dist;
            const ny = dy / dist;
            b.x = peg.x + nx * (minDist + 1);
            b.y = peg.y + ny * (minDist + 1);

            // velocity reflect karo normal ke saath
            const dot = b.vx * nx + b.vy * ny;
            b.vx = (b.vx - 2 * dot * nx) * bounce;
            b.vy = (b.vy - 2 * dot * ny) * bounce;

            // bias apply karo — slightly push left/right
            if (Math.random() < bias) {
              b.vx += 0.5;
            } else {
              b.vx -= 0.5;
            }
          }
        }
      }

      // bin mein settle ho gaya?
      if (b.y > binStartY) {
        b.settled = true;
        // konse bin mein giri — x position se decide karo
        const numBins = numRows + 1;
        const totalBinW = numBins * binWidth;
        const binStartX = (canvasW - totalBinW) / 2;
        let binIdx = Math.floor((b.x - binStartX) / binWidth);
        binIdx = Math.max(0, Math.min(numBins - 1, binIdx));
        b.bin = binIdx;
        bins[binIdx]++;
        // settled ball ko stack position pe rakh do
        b.x = binStartX + (binIdx + 0.5) * binWidth;
        b.y = CANVAS_HEIGHT - bins[binIdx] * 4 - 2;
      }

      // screen se bahar gaya toh hata do
      if (b.x < -50 || b.x > canvasW + 50 || b.y > CANVAS_HEIGHT + 50) {
        balls.splice(i, 1);
      }
    }

    // memory limit — zyada settled balls ho gayi toh purani hata do
    if (balls.length > 500) {
      let removed = 0;
      for (let i = 0; i < balls.length && removed < 50; i++) {
        if (balls[i].settled) {
          balls.splice(i, 1);
          removed++;
          i--;
        }
      }
    }
  }

  // --- draw ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    computeLayout();

    const numBins = numRows + 1;

    // pegs draw karo
    for (let row = 0; row < numRows; row++) {
      const numPegs = row + 1;
      for (let col = 0; col < numPegs; col++) {
        const peg = pegPos(row, col);
        ctx.beginPath();
        ctx.arc(peg.x, peg.y, pegRadius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(245,158,11,0.4)';
        ctx.fill();
      }
    }

    // bin dividers draw karo
    const totalBinW = numBins * binWidth;
    const binStartX = (canvasW - totalBinW) / 2;
    ctx.strokeStyle = 'rgba(245,158,11,0.15)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= numBins; i++) {
      const x = binStartX + i * binWidth;
      ctx.beginPath();
      ctx.moveTo(x, binStartY - 10);
      ctx.lineTo(x, CANVAS_HEIGHT);
      ctx.stroke();
    }

    // histogram bars draw karo
    const maxBin = Math.max(1, ...bins);
    const maxBarH = CANVAS_HEIGHT - binStartY - 10;
    for (let i = 0; i < numBins; i++) {
      if (bins[i] === 0) continue;
      const barH = (bins[i] / maxBin) * maxBarH;
      const x = binStartX + i * binWidth + 2;
      const w = binWidth - 4;
      ctx.fillStyle = 'rgba(245,158,11,0.25)';
      ctx.fillRect(x, CANVAS_HEIGHT - barH, w, barH);

      // count dikhao agar jagah ho
      if (bins[i] > 0 && binWidth > 18) {
        ctx.font = "9px 'JetBrains Mono',monospace";
        ctx.fillStyle = 'rgba(245,158,11,0.6)';
        ctx.textAlign = 'center';
        ctx.fillText(String(bins[i]), binStartX + (i + 0.5) * binWidth, CANVAS_HEIGHT - barH - 3);
      }
    }

    // theoretical normal curve overlay karo
    const totalBalls = bins.reduce((a, b) => a + b, 0);
    if (totalBalls > 10) {
      // mean aur std calculate karo actual data se
      let mean = 0, variance = 0;
      for (let i = 0; i < numBins; i++) {
        mean += i * bins[i];
      }
      mean /= totalBalls;
      for (let i = 0; i < numBins; i++) {
        variance += (i - mean) * (i - mean) * bins[i];
      }
      variance /= totalBalls;
      const std = Math.sqrt(variance);

      if (std > 0.1) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(34,211,238,0.5)';
        ctx.lineWidth = 1.5;
        for (let px = 0; px < numBins * 20; px++) {
          const i = px / 20;
          const z = (i - mean) / std;
          const gauss = Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
          const barH = gauss * totalBalls * (maxBarH / maxBin);
          const x = binStartX + (i + 0.5) / numBins * totalBinW;
          const y = CANVAS_HEIGHT - barH;
          if (px === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }

    // active (falling) balls draw karo
    for (let i = 0; i < balls.length; i++) {
      const b = balls[i];
      if (b.settled) continue;
      ctx.beginPath();
      ctx.arc(b.x, b.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#f59e0b';
      ctx.fill();
    }

    // settled balls bhi chhote dots ke roop mein
    for (let i = 0; i < balls.length; i++) {
      const b = balls[i];
      if (!b.settled) continue;
      ctx.beginPath();
      ctx.arc(b.x, Math.min(b.y, CANVAS_HEIGHT - 3), 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(245,158,11,0.5)';
      ctx.fill();
    }

    // labels
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.fillText('GALTON BOARD', 8, 14);
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(245,158,11,0.4)';
    ctx.fillText('n=' + totalBalls, canvasW - 8, 14);

    // funnel indicator — top center
    ctx.beginPath();
    ctx.moveTo(canvasW / 2 - 20, 5);
    ctx.lineTo(canvasW / 2, 25);
    ctx.lineTo(canvasW / 2 + 20, 5);
    ctx.strokeStyle = 'rgba(245,158,11,0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    frameCount++;
    // layout pehle compute karo — resize ke baad updateBalls sahi positions use kare
    computeLayout();

    // har kuch frames pe naya ball spawn karo
    if (frameCount % Math.max(1, 6 - ballRate) === 0) {
      spawnBall();
    }

    updateBalls();
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
}
