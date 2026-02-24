// ============================================================
// Magnetic Pendulum — 3 magnets ke upar pendulum, fractal basin map
// RK4 integration, drag karke starting position set karo
// ============================================================

export function initMagneticPendulum() {
  const container = document.getElementById('magneticPendulumContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  let animationId = null, isVisible = false, canvasW = 0;

  // --- pendulum state ---
  let px = 0.1, py = 0.1; // position
  let vx = 0, vy = 0; // velocity
  let friction = 0.02;
  let magnetStrength = 8.0;
  let gravity = 0.5; // restoring force (spring constant)

  // 3 magnets equilateral triangle mein
  const magnetRadius = 1.2;
  const magnets = [
    { x: magnetRadius * Math.cos(Math.PI / 2), y: magnetRadius * Math.sin(Math.PI / 2), color: '#ff4444' },
    { x: magnetRadius * Math.cos(Math.PI / 2 + 2 * Math.PI / 3), y: magnetRadius * Math.sin(Math.PI / 2 + 2 * Math.PI / 3), color: '#44ff44' },
    { x: magnetRadius * Math.cos(Math.PI / 2 + 4 * Math.PI / 3), y: magnetRadius * Math.sin(Math.PI / 2 + 4 * Math.PI / 3), color: '#4488ff' },
  ];

  // fractal basin map — progressively compute hoga
  let basinImg = null;
  let basinRow = 0;
  const BASIN_W = 200, BASIN_H = 200;
  let basinData = null;
  let basinDirty = true;

  // interaction
  let isDragging = false;
  let trail = [];
  const MAX_TRAIL = 600;

  // --- DOM banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(245,158,11,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls
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
    vl.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:32px;";
    vl.textContent = Number(val).toFixed(step < 0.1 ? 3 : step < 1 ? 2 : 1);
    w.appendChild(vl);
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      const dec = step < 0.1 ? 3 : step < 1 ? 2 : 1;
      vl.textContent = v.toFixed(dec);
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

  createSlider('friction', 0.001, 0.1, 0.001, friction, (v) => { friction = v; basinDirty = true; basinRow = 0; });
  createSlider('magnet', 1, 20, 0.5, magnetStrength, (v) => { magnetStrength = v; basinDirty = true; basinRow = 0; });
  createButton('Reset', () => {
    px = 0.1; py = 0.1; vx = 0; vy = 0;
    trail = [];
    basinDirty = true; basinRow = 0;
  });

  // --- coordinate transforms ---
  // sim range: -2.5 to 2.5
  const SIM_RANGE = 2.5;
  function simToCanvas(sx, sy) {
    const cx = (sx / SIM_RANGE + 1) * 0.5 * canvasW;
    const cy = (-sy / SIM_RANGE + 1) * 0.5 * CANVAS_HEIGHT;
    return [cx, cy];
  }
  function canvasToSim(cx, cy) {
    const sx = (cx / canvasW * 2 - 1) * SIM_RANGE;
    const sy = -(cy / CANVAS_HEIGHT * 2 - 1) * SIM_RANGE;
    return [sx, sy];
  }

  // --- physics: acceleration ---
  function accel(x, y, vvx, vvy) {
    let ax = -gravity * x - friction * vvx;
    let ay = -gravity * y - friction * vvy;
    // magnet forces: F = k * (ri - r) / |ri - r|^3
    for (let i = 0; i < magnets.length; i++) {
      const dx = magnets[i].x - x;
      const dy = magnets[i].y - y;
      const r2 = dx * dx + dy * dy + 0.01; // softening
      const r = Math.sqrt(r2);
      const r3 = r2 * r;
      ax += magnetStrength * dx / r3;
      ay += magnetStrength * dy / r3;
    }
    return [ax, ay];
  }

  // RK4 integration step
  function rk4(x, y, vvx, vvy, dt) {
    const [ax1, ay1] = accel(x, y, vvx, vvy);
    const [ax2, ay2] = accel(x + vvx * dt / 2, y + vvy * dt / 2, vvx + ax1 * dt / 2, vvy + ay1 * dt / 2);
    const [ax3, ay3] = accel(x + (vvx + ax1 * dt / 2) * dt / 2, y + (vvy + ay1 * dt / 2) * dt / 2, vvx + ax2 * dt / 2, vvy + ay2 * dt / 2);
    const [ax4, ay4] = accel(x + (vvx + ax2 * dt / 2) * dt, y + (vvy + ay2 * dt / 2) * dt, vvx + ax3 * dt, vvy + ay3 * dt);
    return [
      x + dt * (vvx + dt / 6 * (ax1 + ax2 + ax3)),
      y + dt * (vvy + dt / 6 * (ay1 + ay2 + ay3)),
      vvx + dt / 6 * (ax1 + 2 * ax2 + 2 * ax3 + ax4),
      vvy + dt / 6 * (ay1 + 2 * ay2 + 2 * ay3 + ay4),
    ];
  }

  // basin classify: simulate from (x,y) and return which magnet it lands on
  function classifyBasin(x0, y0) {
    let x = x0, y = y0, vvx = 0, vvy = 0;
    const dt = 0.02;
    for (let i = 0; i < 2000; i++) {
      const [nx, ny, nvx, nvy] = rk4(x, y, vvx, vvy, dt);
      x = nx; y = ny; vvx = nvx; vvy = nvy;
      // check karo kisi magnet ke paas pahuncha ya nahi
      const speed2 = vvx * vvx + vvy * vvy;
      if (speed2 < 0.001) {
        let minD = Infinity, minI = 0;
        for (let m = 0; m < magnets.length; m++) {
          const dd = (x - magnets[m].x) ** 2 + (y - magnets[m].y) ** 2;
          if (dd < minD) { minD = dd; minI = m; }
        }
        return minI;
      }
    }
    // default — closest magnet chun lo
    let minD = Infinity, minI = 0;
    for (let m = 0; m < magnets.length; m++) {
      const dd = (x - magnets[m].x) ** 2 + (y - magnets[m].y) ** 2;
      if (dd < minD) { minD = dd; minI = m; }
    }
    return minI;
  }

  // --- progressive basin rendering ---
  function computeBasinRows(rowCount) {
    if (!basinData) basinData = new Uint8Array(BASIN_W * BASIN_H);
    if (basinRow >= BASIN_H) return;
    const end = Math.min(basinRow + rowCount, BASIN_H);
    for (let gy = basinRow; gy < end; gy++) {
      for (let gx = 0; gx < BASIN_W; gx++) {
        const sx = (gx / (BASIN_W - 1) * 2 - 1) * SIM_RANGE;
        const sy = -(gy / (BASIN_H - 1) * 2 - 1) * SIM_RANGE;
        basinData[gy * BASIN_W + gx] = classifyBasin(sx, sy);
      }
    }
    basinRow = end;
    // image data banao
    const imgData = ctx.createImageData(BASIN_W, BASIN_H);
    const colors = [[255, 68, 68], [68, 255, 68], [68, 136, 255]];
    for (let i = 0; i < BASIN_W * BASIN_H; i++) {
      const c = colors[basinData[i]];
      const rendered = i < basinRow * BASIN_W;
      imgData.data[i * 4] = rendered ? c[0] : 17;
      imgData.data[i * 4 + 1] = rendered ? c[1] : 17;
      imgData.data[i * 4 + 2] = rendered ? c[2] : 17;
      imgData.data[i * 4 + 3] = rendered ? 60 : 255;
    }
    // offscreen canvas pe draw karo
    if (!basinImg) {
      basinImg = document.createElement('canvas');
      basinImg.width = BASIN_W;
      basinImg.height = BASIN_H;
    }
    basinImg.getContext('2d').putImageData(imgData, 0, 0);
  }

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

  // --- mouse interaction ---
  function getPos(e) {
    const r = canvas.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
    return canvasToSim(cx, cy);
  }

  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const [sx, sy] = getPos(e);
    px = sx; py = sy; vx = 0; vy = 0;
    trail = [];
    isDragging = true;
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const [sx, sy] = getPos(e);
    px = sx; py = sy; vx = 0; vy = 0;
  });
  canvas.addEventListener('mouseup', () => { isDragging = false; });
  canvas.addEventListener('mouseleave', () => { isDragging = false; });

  canvas.addEventListener('touchstart', (e) => { e.preventDefault(); const [sx, sy] = getPos(e); px = sx; py = sy; vx = 0; vy = 0; trail = []; isDragging = true; }, { passive: false });
  canvas.addEventListener('touchmove', (e) => { e.preventDefault(); if (!isDragging) return; const [sx, sy] = getPos(e); px = sx; py = sy; vx = 0; vy = 0; }, { passive: false });
  canvas.addEventListener('touchend', () => { isDragging = false; });

  // --- draw ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // basin map draw karo
    if (basinImg) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(basinImg, 0, 0, canvasW, CANVAS_HEIGHT);
    }

    // magnets draw karo
    for (let i = 0; i < magnets.length; i++) {
      const [cx, cy] = simToCanvas(magnets[i].x, magnets[i].y);
      ctx.beginPath();
      ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.fillStyle = magnets[i].color;
      ctx.shadowColor = magnets[i].color;
      ctx.shadowBlur = 15;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // trail draw karo
    if (trail.length > 1) {
      for (let i = 1; i < trail.length; i++) {
        const alpha = (i / trail.length) * 0.8;
        ctx.beginPath();
        ctx.moveTo(trail[i - 1][0], trail[i - 1][1]);
        ctx.lineTo(trail[i][0], trail[i][1]);
        ctx.strokeStyle = `rgba(245,158,11,${alpha})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    // pendulum bob draw karo
    const [bx, by] = simToCanvas(px, py);
    ctx.beginPath();
    ctx.arc(bx, by, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#f59e0b';
    ctx.shadowColor = '#f59e0b';
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;

    // center cross dikhao
    const [ccx, ccy] = simToCanvas(0, 0);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ccx - 8, ccy); ctx.lineTo(ccx + 8, ccy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ccx, ccy - 8); ctx.lineTo(ccx, ccy + 8); ctx.stroke();

    // label
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'left';
    ctx.fillText('MAGNETIC PENDULUM', 8, 14);
    if (basinRow < BASIN_H) {
      ctx.textAlign = 'right';
      ctx.fillText('computing basin... ' + Math.round(basinRow / BASIN_H * 100) + '%', canvasW - 8, 14);
    }
  }

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    // basin progressively compute karo — har frame 2 rows
    if (basinDirty && basinRow < BASIN_H) {
      computeBasinRows(2);
      if (basinRow >= BASIN_H) basinDirty = false;
    }

    // physics step — jab drag nahi ho raha
    if (!isDragging) {
      const dt = 0.01;
      for (let s = 0; s < 5; s++) {
        const [nx, ny, nvx, nvy] = rk4(px, py, vx, vy, dt);
        px = nx; py = ny; vx = nvx; vy = nvy;
      }
      const [cx, cy] = simToCanvas(px, py);
      trail.push([cx, cy]);
      if (trail.length > MAX_TRAIL) trail.shift();
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
}
