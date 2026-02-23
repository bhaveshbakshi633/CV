// ============================================================
// Projectile Optimizer — Gradient Descent se perfect shot dhundho
// Cannon se fire karo, target hit karo, optimization dekho live
// ============================================================

export function initProjectile() {
  const container = document.getElementById('projectileContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 320;
  const HEATMAP_SIZE = 60; // chhota heatmap — fast compute
  const GRAVITY = 9.81;
  const MAX_ITERATIONS = 150;
  const ITERATION_DELAY = 50; // ms between steps

  // parameter ranges
  const ANGLE_MIN = 5, ANGLE_MAX = 85; // degrees
  const VEL_MIN = 5, VEL_MAX = 40; // m/s — realistic range for canvas
  const DRAG_COEFF = 0.02;

  // --- State ---
  let canvasW = 0, canvasH = 0, dpr = 1;
  let targetPhysX = -1; // target position in meters (always on ground)
  let targetPlaced = false;

  // gradient descent state
  let currentAngle = 45, currentVel = 20;
  let bestAngle = 45, bestVel = 20, bestLoss = Infinity;
  let learningRate = 0.02;
  let airResistance = false;
  let iteration = 0;
  let isOptimizing = false;
  let optimizeTimer = null;

  // trajectory history
  let trajectoryHistory = []; // [{points, loss, angle, vel}]
  let gradientPath = []; // heatmap pe path

  // heatmap
  let heatmapCanvas = null;
  let heatmapValid = false;

  // fire animation
  let animationId = null;
  let isVisible = false;
  let fireAnimProgress = -1;
  let fireBestTrajectory = null;

  // --- DOM setup — header rakh, canvas neeche add kar ---
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;height:' + CANVAS_HEIGHT + 'px;display:block;border:1px solid rgba(74,158,255,0.15);border-radius:8px;cursor:crosshair;background:transparent;margin-top:8px;';
  container.appendChild(canvas);

  // stats bar
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = 'margin-top:8px;padding:8px 12px;background:rgba(74,158,255,0.05);border:1px solid rgba(74,158,255,0.12);border-radius:6px;font-family:monospace;font-size:12px;color:#b0b0b0;display:flex;flex-wrap:wrap;gap:16px;';
  container.appendChild(statsDiv);

  const statIter = document.createElement('span');
  const statLoss = document.createElement('span');
  const statAngle = document.createElement('span');
  const statVel = document.createElement('span');
  [statIter, statLoss, statAngle, statVel].forEach(el => statsDiv.appendChild(el));

  function updateStats() {
    statIter.textContent = 'Iteration: ' + iteration;
    statLoss.textContent = 'Loss: ' + (bestLoss === Infinity ? '\u2014' : bestLoss.toFixed(2) + 'm');
    statAngle.textContent = 'Angle: ' + bestAngle.toFixed(1) + '\u00B0';
    statVel.textContent = 'Velocity: ' + bestVel.toFixed(1) + ' m/s';
  }
  updateStats();

  // controls
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;align-items:center;';
  container.appendChild(controlsDiv);

  // button helper
  function mkBtn(text) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = 'padding:6px 16px;font-size:12px;border-radius:6px;cursor:pointer;background:rgba(74,158,255,0.1);color:#b0b0b0;border:1px solid rgba(74,158,255,0.25);font-family:monospace;transition:all 0.2s ease;';
    btn.onmouseenter = () => { btn.style.background = 'rgba(74,158,255,0.25)'; btn.style.color = '#e0e0e0'; };
    btn.onmouseleave = () => { btn.style.background = 'rgba(74,158,255,0.1)'; btn.style.color = '#b0b0b0'; };
    controlsDiv.appendChild(btn);
    return btn;
  }

  const optimizeBtn = mkBtn('Optimize');
  optimizeBtn.onclick = () => toggleOptimize();

  const fireBtn = mkBtn('Fire Best');
  fireBtn.onclick = () => fireBest();

  const resetBtn = mkBtn('Reset');
  resetBtn.onclick = () => resetAll();

  // LR slider
  const lrWrap = document.createElement('div');
  lrWrap.style.cssText = 'display:flex;align-items:center;gap:6px;';

  const lrLabel = document.createElement('span');
  lrLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace';
  lrLabel.textContent = 'LR:';
  lrWrap.appendChild(lrLabel);

  const lrSlider = document.createElement('input');
  lrSlider.type = 'range'; lrSlider.min = '0.005'; lrSlider.max = '0.08'; lrSlider.step = '0.001'; lrSlider.value = learningRate;
  lrSlider.style.cssText = 'width:80px;height:4px;accent-color:rgba(74,158,255,0.8);cursor:pointer;';
  lrWrap.appendChild(lrSlider);

  const lrVal = document.createElement('span');
  lrVal.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:monospace;min-width:36px;';
  lrVal.textContent = learningRate.toFixed(3);
  lrWrap.appendChild(lrVal);

  lrSlider.oninput = () => { learningRate = parseFloat(lrSlider.value); lrVal.textContent = learningRate.toFixed(3); };
  controlsDiv.appendChild(lrWrap);

  // Air drag toggle
  const arWrap = document.createElement('div');
  arWrap.style.cssText = 'display:flex;align-items:center;gap:6px;';

  const arLabel = document.createElement('span');
  arLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace';
  arLabel.textContent = 'Air Drag:';
  arWrap.appendChild(arLabel);

  const arCheck = document.createElement('input');
  arCheck.type = 'checkbox'; arCheck.checked = false;
  arCheck.style.cssText = 'accent-color:rgba(74,158,255,0.8);cursor:pointer;';
  arCheck.onchange = () => { airResistance = arCheck.checked; heatmapValid = false; };
  arWrap.appendChild(arCheck);
  controlsDiv.appendChild(arWrap);

  // --- Canvas sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvasH = CANVAS_HEIGHT;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    canvas.style.width = canvasW + 'px';
    canvas.style.height = canvasH + 'px';
    canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!heatmapCanvas) heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = HEATMAP_SIZE;
    heatmapCanvas.height = HEATMAP_SIZE;
    heatmapValid = false;
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Coordinate system ---
  const CANNON_X = 40;
  const GROUND_Y = CANVAS_HEIGHT - 25;

  // auto-scale: fit max range in canvas width
  function getScale() {
    const maxRange = (VEL_MAX * VEL_MAX) / GRAVITY;
    return (canvasW - 80) / maxRange;
  }

  function physToCanvas(px, py) {
    const s = getScale();
    return { x: CANNON_X + px * s, y: GROUND_Y - py * s };
  }

  function canvasToPhys(cx) {
    const s = getScale();
    return (cx - CANNON_X) / s;
  }

  // --- Physics simulation ---
  function simulateTrajectory(angleDeg, velocity) {
    const rad = angleDeg * Math.PI / 180;
    let vx = velocity * Math.cos(rad), vy = velocity * Math.sin(rad);
    let x = 0, y = 0;
    const dt = 0.02;
    const points = [physToCanvas(0, 0)];

    for (let t = 0; t < 15; t += dt) {
      if (airResistance) {
        const spd = Math.sqrt(vx * vx + vy * vy);
        vx -= DRAG_COEFF * spd * vx * dt;
        vy -= (GRAVITY + DRAG_COEFF * spd * vy) * dt;
      } else {
        vy -= GRAVITY * dt;
      }
      x += vx * dt;
      y += vy * dt;

      if (y < 0 && t > dt) {
        // interpolate landing
        const prevY = y - vy * dt;
        const frac = prevY / (prevY - y);
        const landX = (x - vx * dt) + vx * dt * frac;
        points.push(physToCanvas(landX, 0));
        return { points, landX };
      }
      points.push(physToCanvas(x, y));
    }
    return { points, landX: x };
  }

  // loss = horizontal distance in meters
  function computeLoss(angleDeg, velocity) {
    const { landX } = simulateTrajectory(angleDeg, velocity);
    return Math.abs(landX - targetPhysX);
  }

  // --- Heatmap ---
  function computeHeatmap() {
    if (heatmapValid || !targetPlaced) return;
    const ctx = heatmapCanvas.getContext('2d');
    const w = HEATMAP_SIZE, h = HEATMAP_SIZE;
    const img = ctx.createImageData(w, h);
    let maxLoss = 0;
    const grid = new Float32Array(w * h);

    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const angle = ANGLE_MIN + (ANGLE_MAX - ANGLE_MIN) * (i / (w - 1));
        const vel = VEL_MAX - (VEL_MAX - VEL_MIN) * (j / (h - 1));
        const loss = computeLoss(angle, vel);
        grid[j * w + i] = loss;
        if (loss > maxLoss) maxLoss = loss;
      }
    }

    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const idx = (j * w + i) * 4;
        const t = Math.min(grid[j * w + i] / Math.max(maxLoss, 1), 1);
        const inv = 1 - t;
        img.data[idx] = Math.round(inv * inv * 30);
        img.data[idx + 1] = Math.round(inv * 200);
        img.data[idx + 2] = Math.round(inv * 255);
        img.data[idx + 3] = 200;
      }
    }
    ctx.putImageData(img, 0, 0);
    heatmapValid = true;
  }

  // --- Gradient Descent ---
  function computeGradient(angle, vel) {
    const eps = 0.3;
    const loss = computeLoss(angle, vel);
    const dA = (computeLoss(angle + eps, vel) - computeLoss(angle - eps, vel)) / (2 * eps);
    const dV = (computeLoss(angle, vel + eps) - computeLoss(angle, vel - eps)) / (2 * eps);
    return { dA, dV, loss };
  }

  function gradientStep() {
    const g = computeGradient(currentAngle, currentVel);

    currentAngle -= learningRate * g.dA * (ANGLE_MAX - ANGLE_MIN);
    currentVel -= learningRate * g.dV * (VEL_MAX - VEL_MIN);

    // clamp
    currentAngle = Math.max(ANGLE_MIN, Math.min(ANGLE_MAX, currentAngle));
    currentVel = Math.max(VEL_MIN, Math.min(VEL_MAX, currentVel));

    const { points } = simulateTrajectory(currentAngle, currentVel);
    trajectoryHistory.push({ points, loss: g.loss, angle: currentAngle, vel: currentVel });

    if (g.loss < bestLoss) {
      bestLoss = g.loss;
      bestAngle = currentAngle;
      bestVel = currentVel;
    }

    gradientPath.push({ angle: currentAngle, vel: currentVel, loss: g.loss });
    iteration++;
    updateStats();
  }

  // --- Optimize toggle ---
  function toggleOptimize() {
    if (isOptimizing) {
      stopOptimize();
      return;
    }
    if (!targetPlaced) return;

    isOptimizing = true;
    iteration = 0;
    bestLoss = Infinity;
    trajectoryHistory = [];
    gradientPath = [];
    fireAnimProgress = -1;
    fireBestTrajectory = null;

    // random start
    currentAngle = ANGLE_MIN + Math.random() * (ANGLE_MAX - ANGLE_MIN);
    currentVel = VEL_MIN + Math.random() * (VEL_MAX - VEL_MIN);
    bestAngle = currentAngle;
    bestVel = currentVel;

    computeHeatmap();

    optimizeTimer = setInterval(() => {
      if (iteration >= MAX_ITERATIONS || bestLoss < 0.3) {
        stopOptimize();
        return;
      }
      gradientStep();
    }, ITERATION_DELAY);

    optimizeBtn.textContent = 'Stop';
  }

  function stopOptimize() {
    isOptimizing = false;
    if (optimizeTimer) { clearInterval(optimizeTimer); optimizeTimer = null; }
    optimizeBtn.textContent = 'Optimize';
  }

  function fireBest() {
    if (bestLoss === Infinity) return;
    fireBestTrajectory = simulateTrajectory(bestAngle, bestVel).points;
    fireAnimProgress = 0;
  }

  function resetAll() {
    stopOptimize();
    targetPlaced = false;
    targetPhysX = -1;
    iteration = 0;
    bestLoss = Infinity;
    bestAngle = 45; bestVel = 20;
    currentAngle = 45; currentVel = 20;
    trajectoryHistory = [];
    gradientPath = [];
    heatmapValid = false;
    fireAnimProgress = -1;
    fireBestTrajectory = null;
    updateStats();
  }

  // --- Click to place target — always on ground ---
  canvas.addEventListener('click', (e) => {
    if (isOptimizing) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;

    // target on ground in physics coords
    targetPhysX = Math.max(2, canvasToPhys(cx));
    targetPlaced = true;

    trajectoryHistory = [];
    gradientPath = [];
    iteration = 0;
    bestLoss = Infinity;
    fireAnimProgress = -1;
    fireBestTrajectory = null;
    heatmapValid = false;
    updateStats();
  });

  // --- Drawing ---
  function draw() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvasW, canvasH);

    // ground
    ctx.beginPath();
    ctx.moveTo(0, GROUND_Y);
    ctx.lineTo(canvasW, GROUND_Y);
    ctx.strokeStyle = 'rgba(74,158,255,0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = 'rgba(74,158,255,0.03)';
    ctx.fillRect(0, GROUND_Y, canvasW, canvasH - GROUND_Y);

    drawGroundMarkers(ctx);
    drawCannon(ctx);

    if (targetPlaced) {
      drawTarget(ctx);
    } else {
      ctx.font = '13px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.4)';
      ctx.textAlign = 'center';
      ctx.fillText('click anywhere to place target on ground', canvasW / 2, canvasH / 2);
    }

    drawTrajectories(ctx);
    drawBestTrajectory(ctx);
    if (fireAnimProgress >= 0 && fireBestTrajectory) drawFireAnim(ctx);
    if (heatmapValid && targetPlaced) drawHeatmap(ctx);
  }

  function drawGroundMarkers(ctx) {
    const s = getScale();
    ctx.font = '9px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.2)';
    ctx.textAlign = 'center';
    for (let m = 20; m < 200; m += 20) {
      const x = CANNON_X + m * s;
      if (x > canvasW - 10) break;
      ctx.fillText(m + 'm', x, GROUND_Y + 14);
      ctx.beginPath();
      ctx.moveTo(x, GROUND_Y - 2);
      ctx.lineTo(x, GROUND_Y + 2);
      ctx.strokeStyle = 'rgba(176,176,176,0.15)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  function drawCannon(ctx) {
    const baseX = CANNON_X, baseY = GROUND_Y;
    const angle = -(bestAngle * Math.PI / 180);
    const barrelLen = 28, barrelW = 7;

    ctx.save();
    ctx.translate(baseX, baseY);
    ctx.rotate(angle);

    const barGrad = ctx.createLinearGradient(0, -barrelW / 2, 0, barrelW / 2);
    barGrad.addColorStop(0, 'rgba(200,200,210,0.9)');
    barGrad.addColorStop(1, 'rgba(140,140,160,0.7)');
    ctx.fillStyle = barGrad;
    ctx.fillRect(0, -barrelW / 2, barrelLen, barrelW);
    ctx.strokeStyle = 'rgba(74,158,255,0.4)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, -barrelW / 2, barrelLen, barrelW);

    // muzzle
    ctx.fillStyle = 'rgba(40,40,50,0.8)';
    ctx.fillRect(barrelLen - 3, -barrelW / 2 + 1, 3, barrelW - 2);

    ctx.restore();

    // base
    ctx.beginPath();
    ctx.arc(baseX, baseY, 10, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(160,160,180,0.6)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(74,158,255,0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // angle readout
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(74,158,255,0.5)';
    ctx.textAlign = 'left';
    ctx.fillText(bestAngle.toFixed(0) + '\u00B0', baseX + 14, baseY - 8);
  }

  function drawTarget(ctx) {
    const pos = physToCanvas(targetPhysX, 0);
    const tx = pos.x, ty = GROUND_Y;

    // flag pole
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx, ty - 35);
    ctx.strokeStyle = 'rgba(239,68,68,0.6)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // flag
    ctx.beginPath();
    ctx.moveTo(tx, ty - 35);
    ctx.lineTo(tx + 15, ty - 28);
    ctx.lineTo(tx, ty - 21);
    ctx.closePath();
    ctx.fillStyle = 'rgba(239,68,68,0.7)';
    ctx.fill();

    // ground dot
    ctx.beginPath();
    ctx.arc(tx, ty, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(239,68,68,0.3)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(239,68,68,0.6)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // distance label
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(239,68,68,0.5)';
    ctx.textAlign = 'center';
    ctx.fillText(targetPhysX.toFixed(0) + 'm', tx, ty + 16);
  }

  function drawTrajectories(ctx) {
    const len = trajectoryHistory.length;
    for (let i = 0; i < len; i++) {
      const pts = trajectoryHistory[i].points;
      if (pts.length < 2) continue;
      const alpha = 0.04 + (i / len) * 0.15;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let j = 1; j < pts.length; j++) ctx.lineTo(pts[j].x, pts[j].y);
      ctx.strokeStyle = 'rgba(74,158,255,' + alpha + ')';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  function drawBestTrajectory(ctx) {
    if (bestLoss === Infinity || trajectoryHistory.length === 0) return;

    let best = null, minL = Infinity;
    for (const t of trajectoryHistory) {
      if (t.loss < minL) { minL = t.loss; best = t; }
    }
    if (!best || best.points.length < 2) return;

    const pts = best.points;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let j = 1; j < pts.length; j++) ctx.lineTo(pts[j].x, pts[j].y);

    ctx.shadowColor = 'rgba(0,255,255,0.5)';
    ctx.shadowBlur = 10;
    ctx.strokeStyle = 'rgba(0,255,255,0.85)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // landing dot
    const lp = pts[pts.length - 1];
    ctx.beginPath();
    ctx.arc(lp.x, lp.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,255,255,0.85)';
    ctx.fill();

    // distance error from target
    if (targetPlaced && minL > 0.5) {
      const tpos = physToCanvas(targetPhysX, 0);
      ctx.font = '10px monospace';
      ctx.fillStyle = minL < 2 ? 'rgba(16,185,129,0.8)' : 'rgba(239,68,68,0.6)';
      ctx.textAlign = 'center';
      ctx.fillText('\u0394' + minL.toFixed(1) + 'm', (lp.x + tpos.x) / 2, GROUND_Y + 16);
    }
  }

  function drawFireAnim(ctx) {
    fireAnimProgress += 0.018;
    if (fireAnimProgress > 1) { fireAnimProgress = -1; return; }

    const pts = fireBestTrajectory;
    const idx = Math.floor(fireAnimProgress * (pts.length - 1));
    const pt = pts[Math.min(idx, pts.length - 1)];

    // projectile glow
    ctx.shadowColor = 'rgba(255,200,50,0.9)';
    ctx.shadowBlur = 25;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,220,100,0.95)';
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // trail
    const trailLen = 20;
    for (let i = Math.max(0, idx - trailLen); i < idx; i++) {
      const a = (i - (idx - trailLen)) / trailLen * 0.5;
      ctx.beginPath();
      ctx.arc(pts[i].x, pts[i].y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,180,50,' + a + ')';
      ctx.fill();
    }

    // impact flash
    if (fireAnimProgress > 0.95) {
      const last = pts[pts.length - 1];
      const flash = (1 - (fireAnimProgress - 0.95) / 0.05);
      ctx.beginPath();
      ctx.arc(last.x, last.y, 15 * flash, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,220,100,' + (flash * 0.4) + ')';
      ctx.fill();
    }
  }

  function drawHeatmap(ctx) {
    if (!heatmapCanvas) return;

    const hmW = HEATMAP_SIZE, hmH = HEATMAP_SIZE;
    const hmX = canvasW - hmW - 14, hmY = 14;

    ctx.fillStyle = 'rgba(10,10,15,0.88)';
    ctx.fillRect(hmX - 3, hmY - 3, hmW + 6, hmH + 6);
    ctx.strokeStyle = 'rgba(74,158,255,0.2)';
    ctx.lineWidth = 1;
    ctx.strokeRect(hmX - 3, hmY - 3, hmW + 6, hmH + 6);

    ctx.drawImage(heatmapCanvas, hmX, hmY, hmW, hmH);

    // gradient path
    if (gradientPath.length > 1) {
      ctx.beginPath();
      for (let i = 0; i < gradientPath.length; i++) {
        const gp = gradientPath[i];
        const px = hmX + ((gp.angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)) * hmW;
        const py = hmY + ((VEL_MAX - gp.vel) / (VEL_MAX - VEL_MIN)) * hmH;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.strokeStyle = 'rgba(255,255,100,0.7)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // current dot
      const last = gradientPath[gradientPath.length - 1];
      const lx = hmX + ((last.angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)) * hmW;
      const ly = hmY + ((VEL_MAX - last.vel) / (VEL_MAX - VEL_MIN)) * hmH;
      ctx.beginPath();
      ctx.arc(lx, ly, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,100,1)';
      ctx.fill();
    }

    // labels
    ctx.font = '8px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.45)';
    ctx.textAlign = 'center';
    ctx.fillText('angle \u2192', hmX + hmW / 2, hmY + hmH + 10);
    ctx.textAlign = 'left';
    ctx.fillText('loss landscape', hmX, hmY - 5);

    ctx.save();
    ctx.translate(hmX - 6, hmY + hmH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('vel \u2192', 0, 0);
    ctx.restore();
  }

  // --- Animation loop ---
  function animate() {
    if (!isVisible) return;
    draw();
    animationId = requestAnimationFrame(animate);
  }

  function start() {
    if (isVisible) return;
    isVisible = true;
    animationId = requestAnimationFrame(animate);
  }

  function stop() {
    isVisible = false;
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => e.isIntersecting ? start() : stop());
  }, { threshold: 0.1 });
  observer.observe(container);

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop();
    else {
      const r = container.getBoundingClientRect();
      if (r.top < window.innerHeight && r.bottom > 0) start();
    }
  });
}
