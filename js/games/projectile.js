// ============================================================
// Projectile Optimizer — Gradient Descent se perfect shot dhundho
// Cannon se fire karo, target hit karo, optimization dekho live
// ============================================================

export function initProjectile() {
  const container = document.getElementById('projectileContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 320;
  const HEATMAP_SIZE = 70; // thoda bada heatmap — detail dikhane ke liye
  const GRAVITY = 9.81;
  const MAX_ITERATIONS = 200;
  const ITERATION_DELAY = 40; // ms between steps — thoda fast taaki user bore na ho
  const CONVERGE_THRESHOLD = 1.0; // 1m ke andar aaya toh converged maanenge

  // parameter ranges — ye canvas ke andar fit hone chahiye
  const ANGLE_MIN = 10, ANGLE_MAX = 80; // degrees
  const VEL_MIN = 5, VEL_MAX = 35; // m/s — max range ~125m, canvas mein fit hoga
  const DRAG_COEFF = 0.02;

  // --- Viridis-inspired colormap for heatmap ---
  // dark purple → blue → cyan → green → yellow (low loss = bright)
  const COLORMAP = [
    [68, 1, 84],      // 0.0 — dark purple (high loss)
    [72, 36, 117],     // 0.1
    [65, 68, 135],     // 0.2
    [53, 95, 141],     // 0.3
    [42, 120, 142],    // 0.4
    [33, 145, 140],    // 0.5
    [34, 168, 132],    // 0.6
    [68, 191, 112],    // 0.7
    [122, 209, 81],    // 0.8
    [189, 223, 38],    // 0.9
    [253, 231, 37]     // 1.0 — bright yellow (low loss = best)
  ];

  // colormap se color interpolate karna — t = 0 (bad) to 1 (good)
  function sampleColormap(t) {
    t = Math.max(0, Math.min(1, t));
    const idx = t * (COLORMAP.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, COLORMAP.length - 1);
    const f = idx - lo;
    return [
      Math.round(COLORMAP[lo][0] + (COLORMAP[hi][0] - COLORMAP[lo][0]) * f),
      Math.round(COLORMAP[lo][1] + (COLORMAP[hi][1] - COLORMAP[lo][1]) * f),
      Math.round(COLORMAP[lo][2] + (COLORMAP[hi][2] - COLORMAP[lo][2]) * f)
    ];
  }

  // --- State ---
  let canvasW = 0, canvasH = 0, dpr = 1;
  let targetPhysX = -1; // target position in meters (hamesha ground pe)
  let targetPlaced = false;

  // gradient descent state
  let currentAngle = 45, currentVel = 20;
  let bestAngle = 45, bestVel = 20, bestLoss = Infinity;
  let learningRate = 0.02;
  let airResistance = false;
  let iteration = 0;
  let isOptimizing = false;
  let optimizeTimer = null;
  let hasConverged = false; // loss < threshold hua ya nahi

  // trajectory history — har iteration ka record rakhenge
  let trajectoryHistory = []; // [{points, loss, angle, vel}]
  let gradientPath = []; // heatmap pe path dikhane ke liye

  // heatmap
  let heatmapCanvas = null;
  let heatmapValid = false;

  // fire animation state
  let animationId = null;
  let isVisible = false;
  let fireAnimProgress = -1;
  let fireBestTrajectory = null;
  let screenShakeAmount = 0; // impact pe screen hila denge
  let screenShakeDecay = 0;
  let impactParticles = []; // explosion particles

  // auto-target: pehli baar visible hone pe random target laga denge
  let autoTargetPlaced = false;

  // --- DOM setup — game-header/game-desc ke neeche append karna hai ---
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;height:' + CANVAS_HEIGHT + 'px;display:block;border:1px solid rgba(74,158,255,0.15);border-radius:8px;cursor:crosshair;background:transparent;margin-top:8px;';
  container.appendChild(canvas);

  // stats bar — iteration, loss, angle, velocity, trajectory count, convergence
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = 'margin-top:8px;padding:8px 12px;background:rgba(74,158,255,0.05);border:1px solid rgba(74,158,255,0.12);border-radius:6px;font-family:monospace;font-size:12px;color:#b0b0b0;display:flex;flex-wrap:wrap;gap:12px;align-items:center;';
  container.appendChild(statsDiv);

  // individual stat spans banao
  const statIter = document.createElement('span');
  const statLoss = document.createElement('span');
  const statAngle = document.createElement('span');
  const statVel = document.createElement('span');
  const statTrajs = document.createElement('span');
  const statConverge = document.createElement('span');
  statConverge.style.cssText = 'font-weight:600;transition:all 0.3s ease;';
  [statIter, statLoss, statAngle, statVel, statTrajs, statConverge].forEach(el => statsDiv.appendChild(el));

  function updateStats() {
    statIter.textContent = 'Iter: ' + iteration;
    statLoss.textContent = 'Loss: ' + (bestLoss === Infinity ? '\u2014' : bestLoss.toFixed(2) + 'm');
    statAngle.textContent = '\u03B8: ' + bestAngle.toFixed(1) + '\u00B0';
    statVel.textContent = 'v: ' + bestVel.toFixed(1) + ' m/s';
    statTrajs.textContent = 'Trajs: ' + trajectoryHistory.length;

    // convergence indicator — green glow jab hit ho jaaye
    if (hasConverged) {
      statConverge.textContent = '\u2713 CONVERGED';
      statConverge.style.color = '#10b981';
      statConverge.style.textShadow = '0 0 8px rgba(16,185,129,0.6)';
    } else if (bestLoss < 3 && bestLoss !== Infinity) {
      statConverge.textContent = '\u2248 CLOSE';
      statConverge.style.color = '#f59e0b';
      statConverge.style.textShadow = '0 0 6px rgba(245,158,11,0.4)';
    } else {
      statConverge.textContent = '';
      statConverge.style.textShadow = 'none';
    }
  }
  updateStats();

  // controls bar
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;align-items:center;';
  container.appendChild(controlsDiv);

  // button helper — consistent styling
  function mkBtn(text) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = 'padding:6px 16px;font-size:12px;border-radius:6px;cursor:pointer;background:rgba(74,158,255,0.1);color:#b0b0b0;border:1px solid rgba(74,158,255,0.25);font-family:monospace;transition:all 0.2s ease;';
    btn.addEventListener('mouseenter', () => { btn.style.background = 'rgba(74,158,255,0.25)'; btn.style.color = '#e0e0e0'; });
    btn.addEventListener('mouseleave', () => { btn.style.background = 'rgba(74,158,255,0.1)'; btn.style.color = '#b0b0b0'; });
    controlsDiv.appendChild(btn);
    return btn;
  }

  const optimizeBtn = mkBtn('Optimize');
  optimizeBtn.addEventListener('click', () => toggleOptimize());

  const fireBtn = mkBtn('\u{1F525} Fire Best');
  fireBtn.addEventListener('click', () => fireBest());

  const randomBtn = mkBtn('\u{1F3AF} Random Target');
  randomBtn.addEventListener('click', () => placeRandomTarget());

  const resetBtn = mkBtn('Reset');
  resetBtn.addEventListener('click', () => resetAll());

  // LR slider — learning rate control
  const lrWrap = document.createElement('div');
  lrWrap.style.cssText = 'display:flex;align-items:center;gap:6px;';

  const lrLabel = document.createElement('span');
  lrLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace';
  lrLabel.textContent = 'LR:';
  lrWrap.appendChild(lrLabel);

  const lrSlider = document.createElement('input');
  lrSlider.type = 'range';
  lrSlider.min = '0.005';
  lrSlider.max = '0.08';
  lrSlider.step = '0.001';
  lrSlider.value = String(learningRate);
  lrSlider.style.cssText = 'width:80px;height:4px;accent-color:rgba(74,158,255,0.8);cursor:pointer;';
  lrWrap.appendChild(lrSlider);

  const lrVal = document.createElement('span');
  lrVal.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:monospace;min-width:36px;';
  lrVal.textContent = learningRate.toFixed(3);
  lrWrap.appendChild(lrVal);

  lrSlider.addEventListener('input', () => {
    learningRate = parseFloat(lrSlider.value);
    lrVal.textContent = learningRate.toFixed(3);
  });
  controlsDiv.appendChild(lrWrap);

  // Air drag toggle
  const arWrap = document.createElement('div');
  arWrap.style.cssText = 'display:flex;align-items:center;gap:6px;';

  const arLabel = document.createElement('span');
  arLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace';
  arLabel.textContent = 'Drag:';
  arWrap.appendChild(arLabel);

  const arCheck = document.createElement('input');
  arCheck.type = 'checkbox';
  arCheck.checked = false;
  arCheck.style.cssText = 'accent-color:rgba(74,158,255,0.8);cursor:pointer;';
  arCheck.addEventListener('change', () => {
    airResistance = arCheck.checked;
    heatmapValid = false;
  });
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
  // cannon left mein hai, ground neeche — physics coords ko canvas coords mein convert karna
  const CANNON_X = 40;
  const GROUND_Y = CANVAS_HEIGHT - 25;

  // auto-scale: max possible range canvas width mein fit ho jaaye
  function getScale() {
    // max range = v^2 / g (45 degrees pe, no drag)
    const maxRange = (VEL_MAX * VEL_MAX) / GRAVITY;
    return (canvasW - 80) / maxRange;
  }

  // physics (meters) se canvas (pixels) mein convert
  function physToCanvas(px, py) {
    const s = getScale();
    return { x: CANNON_X + px * s, y: GROUND_Y - py * s };
  }

  // canvas x se physics x mein convert
  function canvasToPhys(cx) {
    const s = getScale();
    return (cx - CANNON_X) / s;
  }

  // --- Physics simulation ---
  // euler integration se trajectory simulate karo — drag optional hai
  function simulateTrajectory(angleDeg, velocity) {
    const rad = angleDeg * Math.PI / 180;
    let vx = velocity * Math.cos(rad), vy = velocity * Math.sin(rad);
    let x = 0, y = 0;
    const dt = 0.02;
    const points = [physToCanvas(0, 0)];

    for (let t = 0; t < 15; t += dt) {
      if (airResistance) {
        // quadratic drag — hawa ka resistance
        const spd = Math.sqrt(vx * vx + vy * vy);
        vx -= DRAG_COEFF * spd * vx * dt;
        vy -= (GRAVITY + DRAG_COEFF * spd * vy) * dt;
      } else {
        vy -= GRAVITY * dt;
      }
      x += vx * dt;
      y += vy * dt;

      // zameen pe aa gaya toh landing interpolate karo
      if (y < 0 && t > dt) {
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

  // loss = target se horizontal distance (meters mein)
  function computeLoss(angleDeg, velocity) {
    const { landX } = simulateTrajectory(angleDeg, velocity);
    return Math.abs(landX - targetPhysX);
  }

  // --- Heatmap computation ---
  // poora angle-velocity grid pe loss calculate karo — colormap se paint karo
  function computeHeatmap() {
    if (heatmapValid || !targetPlaced) return;
    const ctx = heatmapCanvas.getContext('2d');
    const w = HEATMAP_SIZE, h = HEATMAP_SIZE;
    const img = ctx.createImageData(w, h);
    let maxLoss = 0;
    const grid = new Float32Array(w * h);

    // pehle saari losses calculate karo
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const angle = ANGLE_MIN + (ANGLE_MAX - ANGLE_MIN) * (i / (w - 1));
        const vel = VEL_MAX - (VEL_MAX - VEL_MIN) * (j / (h - 1));
        const loss = computeLoss(angle, vel);
        grid[j * w + i] = loss;
        if (loss > maxLoss) maxLoss = loss;
      }
    }

    // ab colormap apply karo — low loss = bright (yellow-green), high loss = dark (purple)
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const idx = (j * w + i) * 4;
        // t = 0 means high loss (bad), t = 1 means low loss (good)
        const t = 1 - Math.min(grid[j * w + i] / Math.max(maxLoss, 1), 1);
        const [r, g, b] = sampleColormap(t);
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 220; // thoda transparent
      }
    }
    ctx.putImageData(img, 0, 0);
    heatmapValid = true;
  }

  // --- Gradient Descent ---
  // numerical gradient — central difference se
  function computeGradient(angle, vel) {
    const eps = 0.3;
    const loss = computeLoss(angle, vel);
    const dA = (computeLoss(angle + eps, vel) - computeLoss(angle - eps, vel)) / (2 * eps);
    const dV = (computeLoss(angle, vel + eps) - computeLoss(angle, vel - eps)) / (2 * eps);
    return { dA, dV, loss };
  }

  // ek gradient step lo — angle aur velocity update karo
  function gradientStep() {
    const g = computeGradient(currentAngle, currentVel);

    // scaled update — parameter range ke hisaab se
    currentAngle -= learningRate * g.dA * (ANGLE_MAX - ANGLE_MIN);
    currentVel -= learningRate * g.dV * (VEL_MAX - VEL_MIN);

    // clamp kar — bounds ke bahar nahi jaana
    currentAngle = Math.max(ANGLE_MIN, Math.min(ANGLE_MAX, currentAngle));
    currentVel = Math.max(VEL_MIN, Math.min(VEL_MAX, currentVel));

    const { points } = simulateTrajectory(currentAngle, currentVel);
    trajectoryHistory.push({ points, loss: g.loss, angle: currentAngle, vel: currentVel });

    // best track karo
    if (g.loss < bestLoss) {
      bestLoss = g.loss;
      bestAngle = currentAngle;
      bestVel = currentVel;
    }

    // convergence check
    if (bestLoss < CONVERGE_THRESHOLD) {
      hasConverged = true;
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
    hasConverged = false;
    iteration = 0;
    bestLoss = Infinity;
    trajectoryHistory = [];
    gradientPath = [];
    fireAnimProgress = -1;
    fireBestTrajectory = null;
    impactParticles = [];

    // random starting point — har baar naya position se shuru
    currentAngle = ANGLE_MIN + Math.random() * (ANGLE_MAX - ANGLE_MIN);
    currentVel = VEL_MIN + Math.random() * (VEL_MAX - VEL_MIN);
    bestAngle = currentAngle;
    bestVel = currentVel;

    computeHeatmap();

    optimizeTimer = setInterval(() => {
      if (iteration >= MAX_ITERATIONS || bestLoss < 0.2) {
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

  // --- Fire Best animation — dramatic version ---
  function fireBest() {
    if (bestLoss === Infinity) return;
    stopOptimize(); // optimize chal raha ho toh rok do
    fireBestTrajectory = simulateTrajectory(bestAngle, bestVel).points;
    fireAnimProgress = 0;
    screenShakeAmount = 0;
    screenShakeDecay = 0;
    impactParticles = [];
  }

  // --- Random Target placement ---
  // canvas ke beech mein kahi bhi target rakh do — engaging dikhega
  function placeRandomTarget() {
    if (isOptimizing) return;

    // scale ke hisaab se sensible range mein target lagao
    const maxRange = (VEL_MAX * VEL_MAX) / GRAVITY;
    // 20% se 80% range ke beech — na bahut paas na bahut door
    const minPhys = maxRange * 0.15;
    const maxPhys = maxRange * 0.75;
    targetPhysX = minPhys + Math.random() * (maxPhys - minPhys);
    targetPlaced = true;

    // purana data clear karo
    trajectoryHistory = [];
    gradientPath = [];
    iteration = 0;
    bestLoss = Infinity;
    hasConverged = false;
    fireAnimProgress = -1;
    fireBestTrajectory = null;
    impactParticles = [];
    heatmapValid = false;
    updateStats();
  }

  // --- Reset everything ---
  function resetAll() {
    stopOptimize();
    targetPlaced = false;
    targetPhysX = -1;
    iteration = 0;
    bestLoss = Infinity;
    bestAngle = 45;
    bestVel = 20;
    currentAngle = 45;
    currentVel = 20;
    trajectoryHistory = [];
    gradientPath = [];
    heatmapValid = false;
    hasConverged = false;
    fireAnimProgress = -1;
    fireBestTrajectory = null;
    impactParticles = [];
    screenShakeAmount = 0;
    updateStats();
  }

  // --- Click to place target — hamesha ground pe ---
  canvas.addEventListener('click', (e) => {
    if (isOptimizing) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;

    // minimum 2m door — cannon ke upar nahi rakh sakte
    targetPhysX = Math.max(2, canvasToPhys(cx));
    targetPlaced = true;
    hasConverged = false;

    trajectoryHistory = [];
    gradientPath = [];
    iteration = 0;
    bestLoss = Infinity;
    fireAnimProgress = -1;
    fireBestTrajectory = null;
    impactParticles = [];
    heatmapValid = false;
    updateStats();
  });

  // --- Drawing functions ---

  function draw() {
    const ctx = canvas.getContext('2d');

    // screen shake — impact pe canvas hila do
    ctx.save();
    if (screenShakeAmount > 0.5) {
      const sx = (Math.random() - 0.5) * screenShakeAmount;
      const sy = (Math.random() - 0.5) * screenShakeAmount;
      ctx.translate(sx, sy);
      screenShakeAmount *= 0.88; // decay
    } else {
      screenShakeAmount = 0;
    }

    ctx.clearRect(-10, -10, canvasW + 20, canvasH + 20);

    // ground line
    ctx.beginPath();
    ctx.moveTo(0, GROUND_Y);
    ctx.lineTo(canvasW, GROUND_Y);
    ctx.strokeStyle = 'rgba(74,158,255,0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // ground fill — subtle
    ctx.fillStyle = 'rgba(74,158,255,0.03)';
    ctx.fillRect(0, GROUND_Y, canvasW, canvasH - GROUND_Y);

    drawGroundMarkers(ctx);
    drawCannon(ctx);

    if (targetPlaced) {
      drawTarget(ctx);
    } else {
      // placeholder text — but ideally auto-target pehle aa jaayega
      ctx.font = '13px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.4)';
      ctx.textAlign = 'center';
      ctx.fillText('click anywhere to place target on ground', canvasW / 2, canvasH / 2);
    }

    // trajectories draw karo — bright colors, visible on dark bg
    drawTrajectories(ctx);
    drawBestTrajectory(ctx);

    // fire animation
    if (fireAnimProgress >= 0 && fireBestTrajectory) drawFireAnim(ctx);

    // impact particles
    if (impactParticles.length > 0) drawImpactParticles(ctx);

    // heatmap top-right corner mein
    if (heatmapValid && targetPlaced) drawHeatmap(ctx);

    ctx.restore(); // screen shake restore
  }

  // zameen pe distance markers
  function drawGroundMarkers(ctx) {
    const s = getScale();
    ctx.font = '9px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.25)';
    ctx.textAlign = 'center';
    // 10m interval se markers lagao
    for (let m = 10; m < 200; m += 10) {
      const x = CANNON_X + m * s;
      if (x > canvasW - 10) break;
      // har 20m pe label, baaki pe sirf tick
      if (m % 20 === 0) {
        ctx.fillText(m + 'm', x, GROUND_Y + 14);
      }
      ctx.beginPath();
      ctx.moveTo(x, GROUND_Y - 2);
      ctx.lineTo(x, GROUND_Y + 2);
      ctx.strokeStyle = 'rgba(176,176,176,0.15)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  // cannon draw karo — angle ke hisaab se rotate hoga
  function drawCannon(ctx) {
    const baseX = CANNON_X, baseY = GROUND_Y;
    const angle = -(bestAngle * Math.PI / 180);
    const barrelLen = 28, barrelW = 7;

    ctx.save();
    ctx.translate(baseX, baseY);
    ctx.rotate(angle);

    // barrel gradient
    const barGrad = ctx.createLinearGradient(0, -barrelW / 2, 0, barrelW / 2);
    barGrad.addColorStop(0, 'rgba(200,200,210,0.9)');
    barGrad.addColorStop(1, 'rgba(140,140,160,0.7)');
    ctx.fillStyle = barGrad;
    ctx.fillRect(0, -barrelW / 2, barrelLen, barrelW);
    ctx.strokeStyle = 'rgba(74,158,255,0.4)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, -barrelW / 2, barrelLen, barrelW);

    // muzzle — barrel ka end dark karo
    ctx.fillStyle = 'rgba(40,40,50,0.8)';
    ctx.fillRect(barrelLen - 3, -barrelW / 2 + 1, 3, barrelW - 2);

    ctx.restore();

    // base circle
    ctx.beginPath();
    ctx.arc(baseX, baseY, 10, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(160,160,180,0.6)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(74,158,255,0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // angle label
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(74,158,255,0.6)';
    ctx.textAlign = 'left';
    ctx.fillText(bestAngle.toFixed(0) + '\u00B0', baseX + 14, baseY - 8);
  }

  // target flag draw karo
  function drawTarget(ctx) {
    const pos = physToCanvas(targetPhysX, 0);
    const tx = pos.x, ty = GROUND_Y;

    // ground pe glow — target ke neeche
    const targetGlow = ctx.createRadialGradient(tx, ty, 0, tx, ty, 20);
    targetGlow.addColorStop(0, 'rgba(239,68,68,0.2)');
    targetGlow.addColorStop(1, 'rgba(239,68,68,0)');
    ctx.fillStyle = targetGlow;
    ctx.fillRect(tx - 20, ty - 10, 40, 20);

    // flag pole
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx, ty - 38);
    ctx.strokeStyle = 'rgba(239,68,68,0.7)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // flag triangle
    ctx.beginPath();
    ctx.moveTo(tx, ty - 38);
    ctx.lineTo(tx + 16, ty - 30);
    ctx.lineTo(tx, ty - 22);
    ctx.closePath();
    ctx.fillStyle = 'rgba(239,68,68,0.8)';
    ctx.fill();

    // ground circle
    ctx.beginPath();
    ctx.arc(tx, ty, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(239,68,68,0.3)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(239,68,68,0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // distance label
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(239,68,68,0.6)';
    ctx.textAlign = 'center';
    ctx.fillText(targetPhysX.toFixed(0) + 'm', tx, ty + 18);
  }

  // --- Trajectory drawing ---
  // pehle wali trajectories dim, naye wali bright — convergence dikhe
  function drawTrajectories(ctx) {
    const len = trajectoryHistory.length;
    if (len === 0) return;

    for (let i = 0; i < len; i++) {
      const pts = trajectoryHistory[i].points;
      if (pts.length < 2) continue;

      // progressive brightness — purani = dim, nayi = bright
      const progress = i / len;
      // base alpha 0.08 se start, latest 0.45 tak — dark bg pe bhi dikhe
      const alpha = 0.08 + progress * 0.37;

      // color bhi change karo — shuru mein red-ish, end mein cyan-ish
      // ye visually dikhayega ki convergence ho rahi hai
      const r = Math.round(255 * (1 - progress * 0.7));
      const g = Math.round(100 + 155 * progress);
      const b = Math.round(150 + 105 * progress);

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let j = 1; j < pts.length; j++) ctx.lineTo(pts[j].x, pts[j].y);
      ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  }

  // best trajectory — bright cyan glow ke saath
  function drawBestTrajectory(ctx) {
    if (bestLoss === Infinity || trajectoryHistory.length === 0) return;

    // sabse kam loss wali trajectory dhundho
    let best = null, minL = Infinity;
    for (const t of trajectoryHistory) {
      if (t.loss < minL) { minL = t.loss; best = t; }
    }
    if (!best || best.points.length < 2) return;

    const pts = best.points;

    // glow effect — shadow se
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let j = 1; j < pts.length; j++) ctx.lineTo(pts[j].x, pts[j].y);

    ctx.shadowColor = 'rgba(0,255,255,0.6)';
    ctx.shadowBlur = 12;
    ctx.strokeStyle = 'rgba(0,255,255,0.9)';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // landing dot — bada aur bright
    const lp = pts[pts.length - 1];
    ctx.beginPath();
    ctx.arc(lp.x, lp.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,255,255,0.9)';
    ctx.fill();

    // agar target se door hai toh error dikhao
    if (targetPlaced && minL > 0.5) {
      const tpos = physToCanvas(targetPhysX, 0);
      ctx.font = '10px monospace';
      ctx.fillStyle = minL < 2 ? 'rgba(16,185,129,0.9)' : 'rgba(239,68,68,0.7)';
      ctx.textAlign = 'center';
      ctx.fillText('\u0394' + minL.toFixed(1) + 'm', (lp.x + tpos.x) / 2, GROUND_Y + 18);

      // error line bhi draw karo — visual gap dikhane ke liye
      ctx.beginPath();
      ctx.setLineDash([3, 3]);
      ctx.moveTo(lp.x, GROUND_Y - 3);
      ctx.lineTo(tpos.x, GROUND_Y - 3);
      ctx.strokeStyle = minL < 2 ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // --- Fire animation — dramatic version ---
  // bada projectile, lamba trail, screen shake on impact, particles
  function drawFireAnim(ctx) {
    fireAnimProgress += 0.012; // slower = more dramatic
    if (fireAnimProgress > 1.15) {
      // animation khatam
      fireAnimProgress = -1;
      return;
    }

    const pts = fireBestTrajectory;
    const totalPts = pts.length;

    // actual projectile position
    const rawIdx = Math.min(fireAnimProgress, 1.0) * (totalPts - 1);
    const idx = Math.floor(rawIdx);
    const currentIdx = Math.min(idx, totalPts - 1);
    const pt = pts[currentIdx];

    // impact detect karo
    const hasImpacted = fireAnimProgress >= 0.98;

    if (!hasImpacted) {
      // --- Flying projectile ---

      // outer glow — bada aur dramatic
      ctx.shadowColor = 'rgba(255,180,50,0.95)';
      ctx.shadowBlur = 35;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 9, 0, Math.PI * 2);
      const projGrad = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, 9);
      projGrad.addColorStop(0, 'rgba(255,255,200,1)');
      projGrad.addColorStop(0.4, 'rgba(255,200,80,0.95)');
      projGrad.addColorStop(1, 'rgba(255,120,20,0.6)');
      ctx.fillStyle = projGrad;
      ctx.fill();
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;

      // lamba trail — 40 points tak
      const trailLen = 40;
      const trailStart = Math.max(0, currentIdx - trailLen);
      for (let i = trailStart; i < currentIdx; i++) {
        const progress = (i - trailStart) / trailLen;
        const a = progress * progress * 0.7; // quadratic fade-in
        const radius = 2 + progress * 4;
        ctx.beginPath();
        ctx.arc(pts[i].x, pts[i].y, radius, 0, Math.PI * 2);

        // trail color — orange se yellow gradient
        const tr = 255;
        const tg = Math.round(120 + progress * 100);
        const tb = Math.round(20 + progress * 30);
        ctx.fillStyle = 'rgba(' + tr + ',' + tg + ',' + tb + ',' + a + ')';
        ctx.fill();
      }
    }

    // --- Impact effects ---
    if (hasImpacted) {
      // screen shake trigger karo — ek baar
      if (screenShakeAmount < 1) {
        screenShakeAmount = 14; // intense shake
        // particles spawn karo
        spawnImpactParticles(pts[totalPts - 1]);
      }

      const impactT = (fireAnimProgress - 0.98) / 0.17; // 0 to 1 over remaining time
      const last = pts[totalPts - 1];

      // shockwave ring — expanding circle
      const ringRadius = 20 + impactT * 60;
      const ringAlpha = Math.max(0, 0.6 * (1 - impactT));
      ctx.beginPath();
      ctx.arc(last.x, last.y, ringRadius, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,200,80,' + ringAlpha + ')';
      ctx.lineWidth = 3 * (1 - impactT);
      ctx.stroke();

      // inner flash — bright white-yellow
      if (impactT < 0.5) {
        const flashSize = 25 * (1 - impactT * 2);
        const flashGrad = ctx.createRadialGradient(last.x, last.y, 0, last.x, last.y, flashSize);
        flashGrad.addColorStop(0, 'rgba(255,255,255,' + (0.8 * (1 - impactT * 2)) + ')');
        flashGrad.addColorStop(0.5, 'rgba(255,220,100,' + (0.5 * (1 - impactT * 2)) + ')');
        flashGrad.addColorStop(1, 'rgba(255,100,20,0)');
        ctx.beginPath();
        ctx.arc(last.x, last.y, flashSize, 0, Math.PI * 2);
        ctx.fillStyle = flashGrad;
        ctx.fill();
      }

      // ground scorch mark — permanent-ish
      const scorchAlpha = Math.min(impactT, 1) * 0.3;
      ctx.beginPath();
      ctx.ellipse(last.x, last.y + 2, 12, 4, 0, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(100,60,20,' + scorchAlpha + ')';
      ctx.fill();
    }
  }

  // --- Impact particles spawn karo ---
  function spawnImpactParticles(pos) {
    for (let i = 0; i < 20; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 1.5 + Math.random() * 4;
      impactParticles.push({
        x: pos.x,
        y: pos.y,
        vx: Math.cos(angle) * speed,
        vy: -Math.abs(Math.sin(angle) * speed) - Math.random() * 2, // upar jaayein mostly
        life: 1.0,
        decay: 0.015 + Math.random() * 0.025,
        size: 1.5 + Math.random() * 3,
        // random warm colors
        r: 200 + Math.floor(Math.random() * 55),
        g: 100 + Math.floor(Math.random() * 120),
        b: 20 + Math.floor(Math.random() * 40)
      });
    }
  }

  // --- Impact particles draw aur update ---
  function drawImpactParticles(ctx) {
    for (let i = impactParticles.length - 1; i >= 0; i--) {
      const p = impactParticles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.15; // gravity
      p.life -= p.decay;

      if (p.life <= 0) {
        impactParticles.splice(i, 1);
        continue;
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + p.r + ',' + p.g + ',' + p.b + ',' + (p.life * 0.8) + ')';
      ctx.fill();
    }
  }

  // --- Heatmap draw ---
  // top-right corner mein loss landscape dikhao
  function drawHeatmap(ctx) {
    if (!heatmapCanvas) return;

    const hmW = HEATMAP_SIZE, hmH = HEATMAP_SIZE;
    const hmX = canvasW - hmW - 14, hmY = 14;

    // background panel
    ctx.fillStyle = 'rgba(10,10,15,0.9)';
    ctx.fillRect(hmX - 4, hmY - 4, hmW + 8, hmH + 8);
    ctx.strokeStyle = 'rgba(74,158,255,0.2)';
    ctx.lineWidth = 1;
    ctx.strokeRect(hmX - 4, hmY - 4, hmW + 8, hmH + 8);

    // heatmap image draw karo
    ctx.drawImage(heatmapCanvas, hmX, hmY, hmW, hmH);

    // gradient descent ka path — bright yellow line
    if (gradientPath.length > 1) {
      ctx.beginPath();
      for (let i = 0; i < gradientPath.length; i++) {
        const gp = gradientPath[i];
        const px = hmX + ((gp.angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)) * hmW;
        const py = hmY + ((VEL_MAX - gp.vel) / (VEL_MAX - VEL_MIN)) * hmH;
        if (i === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // start dot — red
      const first = gradientPath[0];
      const fx = hmX + ((first.angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)) * hmW;
      const fy = hmY + ((VEL_MAX - first.vel) / (VEL_MAX - VEL_MIN)) * hmH;
      ctx.beginPath();
      ctx.arc(fx, fy, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239,68,68,1)';
      ctx.fill();

      // current position dot — bright white
      const last = gradientPath[gradientPath.length - 1];
      const lx = hmX + ((last.angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)) * hmW;
      const ly = hmY + ((VEL_MAX - last.vel) / (VEL_MAX - VEL_MIN)) * hmH;
      ctx.beginPath();
      ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,1)';
      ctx.fill();
      // glow
      ctx.shadowColor = 'rgba(255,255,255,0.8)';
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.arc(lx, ly, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,1)';
      ctx.fill();
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
    }

    // labels
    ctx.font = '8px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.5)';
    ctx.textAlign = 'center';
    ctx.fillText('angle \u2192', hmX + hmW / 2, hmY + hmH + 11);
    ctx.textAlign = 'left';
    ctx.fillText('loss landscape', hmX, hmY - 6);

    // velocity axis label — rotated
    ctx.save();
    ctx.translate(hmX - 7, hmY + hmH / 2);
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

    // pehli baar visible hua — auto target lagao taaki user ko turant kuch dikhe
    if (!autoTargetPlaced) {
      autoTargetPlaced = true;
      // thoda delay de — DOM settle hone de
      setTimeout(() => {
        if (!targetPlaced) {
          placeRandomTarget();
        }
      }, 300);
    }

    animationId = requestAnimationFrame(animate);
  }

  function stop() {
    isVisible = false;
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }

  // IntersectionObserver — sirf visible hone pe animate karo, performance ke liye
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        start();
      } else {
        stop();
      }
    });
  }, { threshold: 0.1 });
  observer.observe(container);

  // tab switch pe bhi handle karo
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stop();
    } else {
      const r = container.getBoundingClientRect();
      if (r.top < window.innerHeight && r.bottom > 0) start();
    }
  });
}
