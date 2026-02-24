// ============================================================
// Potential Field Navigation — Robot path planning with attractive/repulsive forces
// Goal attracts, obstacles repel — local minima ka problem bhi dekho
// ============================================================

// yahi entry point hai — vector field, heatmap, robot navigation
export function initPotentialField() {
  const container = document.getElementById('potentialFieldContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const ARROW_SPACING = 28;  // vector field arrows ka gap

  let animationId = null, isVisible = false, canvasW = 0;
  let goal = { x: 0, y: 0 };           // green star — attractive
  let obstacles = [];                    // [{x, y, r}] — red circles, repulsive
  let robot = { x: 0, y: 0, trail: [] }; // blue dot — follows gradient
  let attractStrength = 1.0;
  let repelStrength = 100;
  let robotSpeed = 2.5;
  let robotStuck = false;                // local minima mein fasa ya nahi
  let showHeatmap = true;

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(74,158,255,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

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
  const attSlider = mkSlider(ctrl, 'Attract', 'pfAtt', 0.1, 5, attractStrength, 0.1);
  const attVal = document.createElement('span');
  attVal.style.cssText = "color:#22c55e;font:11px 'JetBrains Mono',monospace;min-width:20px";
  attVal.textContent = attractStrength.toFixed(1);
  ctrl.appendChild(attVal);
  attSlider.addEventListener('input', () => { attractStrength = +attSlider.value; attVal.textContent = attractStrength.toFixed(1); });

  const repSlider = mkSlider(ctrl, 'Repel', 'pfRep', 10, 500, repelStrength, 10);
  const repVal = document.createElement('span');
  repVal.style.cssText = "color:#ef4444;font:11px 'JetBrains Mono',monospace;min-width:24px";
  repVal.textContent = repelStrength;
  ctrl.appendChild(repVal);
  repSlider.addEventListener('input', () => { repelStrength = +repSlider.value; repVal.textContent = repelStrength; });

  mkBtn(ctrl, 'Clear Obstacles', 'pfClearObs').addEventListener('click', () => {
    obstacles = []; robotStuck = false; robot.trail = []; draw();
  });
  mkBtn(ctrl, 'Reset Robot', 'pfResetBot').addEventListener('click', resetRobot);
  mkBtn(ctrl, 'Demo Setup', 'pfDemo').addEventListener('click', demoSetup);

  const stats = document.createElement('div');
  stats.style.cssText = "font:11px 'JetBrains Mono',monospace;color:#888;margin-top:6px;";
  container.appendChild(stats);

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // initial positions set karo
    if (goal.x === 0) {
      goal = { x: canvasW * 0.8, y: CANVAS_HEIGHT * 0.2 };
      resetRobot();
    }
  }
  resize();
  window.addEventListener('resize', resize);

  function resetRobot() {
    robot = { x: canvasW * 0.15, y: CANVAS_HEIGHT * 0.8, trail: [] };
    robotStuck = false;
  }

  // --- demo setup — local minima demonstrate karne ke liye ---
  function demoSetup() {
    goal = { x: canvasW * 0.8, y: CANVAS_HEIGHT * 0.2 };
    obstacles = [
      // barrier of obstacles — robot ko fasa dega
      { x: canvasW * 0.5, y: CANVAS_HEIGHT * 0.3, r: 30 },
      { x: canvasW * 0.5, y: CANVAS_HEIGHT * 0.5, r: 30 },
      { x: canvasW * 0.5 - 40, y: CANVAS_HEIGHT * 0.4, r: 25 },
      { x: canvasW * 0.5 + 40, y: CANVAS_HEIGHT * 0.4, r: 25 },
    ];
    resetRobot();
  }

  // --- potential field compute karo ---
  // attractive potential — goal ki taraf
  function attractivePotential(x, y) {
    const dx = x - goal.x, dy = y - goal.y;
    return 0.5 * attractStrength * (dx * dx + dy * dy);
  }

  // repulsive potential — obstacles se door
  function repulsivePotential(x, y) {
    let pot = 0;
    const rho0 = 80; // influence range
    obstacles.forEach(obs => {
      const dx = x - obs.x, dy = y - obs.y;
      const d = Math.sqrt(dx * dx + dy * dy) - obs.r;
      if (d < rho0 && d > 0) {
        pot += 0.5 * repelStrength * (1 / d - 1 / rho0) * (1 / d - 1 / rho0);
      } else if (d <= 0) {
        pot += 1e6; // obstacle ke andar bahut bada potential
      }
    });
    return pot;
  }

  // total potential
  function totalPotential(x, y) {
    return attractivePotential(x, y) + repulsivePotential(x, y);
  }

  // gradient compute karo — numerical differentiation
  function gradient(x, y) {
    const h = 1.0;
    const gx = (totalPotential(x + h, y) - totalPotential(x - h, y)) / (2 * h);
    const gy = (totalPotential(x, y + h) - totalPotential(x, y - h)) / (2 * h);
    return { gx, gy };
  }

  // --- click handlers ---
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasW / rect.width);
    const y = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height);

    if (e.shiftKey) {
      // shift+click — obstacle place karo
      obstacles.push({ x, y, r: 20 + Math.random() * 15 });
      robotStuck = false;
      robot.trail = [];
    } else {
      // normal click — goal move karo
      goal = { x, y };
      robotStuck = false;
      robot.trail = [];
    }
  });

  // --- heatmap draw karo — potential energy map ---
  function drawHeatmap() {
    if (!showHeatmap) return;
    const step = 6;
    // pehle potential range dhundho
    let minP = Infinity, maxP = -Infinity;
    for (let y = 0; y < CANVAS_HEIGHT; y += step * 3) {
      for (let x = 0; x < canvasW; x += step * 3) {
        const p = totalPotential(x, y);
        if (p < 1e5) { // obstacles ke andar ignore karo
          if (p < minP) minP = p;
          if (p > maxP) maxP = p;
        }
      }
    }
    const range = maxP - minP || 1;

    for (let y = 0; y < CANVAS_HEIGHT; y += step) {
      for (let x = 0; x < canvasW; x += step) {
        const p = totalPotential(x, y);
        const t = Math.min(1, (p - minP) / range);
        // dark (low potential) to bright (high potential)
        const r = Math.floor(20 + t * 40);
        const g = Math.floor(15 + (1 - t) * 25);
        const b = Math.floor(40 + (1 - t) * 60);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x, y, step, step);
      }
    }
  }

  // --- vector field arrows draw karo ---
  function drawVectorField() {
    const arrowLen = 12;
    for (let y = ARROW_SPACING / 2; y < CANVAS_HEIGHT; y += ARROW_SPACING) {
      for (let x = ARROW_SPACING / 2; x < canvasW; x += ARROW_SPACING) {
        const g = gradient(x, y);
        const mag = Math.sqrt(g.gx * g.gx + g.gy * g.gy);
        if (mag < 0.01) continue;

        // negative gradient direction — potential kam hone ki taraf
        const nx = -g.gx / mag;
        const ny = -g.gy / mag;
        const len = Math.min(arrowLen, arrowLen * Math.min(1, mag / 50));

        // arrow draw karo
        const ex = x + nx * len;
        const ey = y + ny * len;

        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        // arrowhead
        const headLen = 3;
        const angle = Math.atan2(ny, nx);
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - headLen * Math.cos(angle - 0.5), ey - headLen * Math.sin(angle - 0.5));
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - headLen * Math.cos(angle + 0.5), ey - headLen * Math.sin(angle + 0.5));
        ctx.stroke();
      }
    }
  }

  // --- goal draw karo — green star ---
  function drawGoal() {
    const x = goal.x, y = goal.y;
    ctx.save();
    ctx.translate(x, y);
    // star shape
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const r = i % 2 === 0 ? 12 : 5;
      const angle = (Math.PI / 2) + (i * Math.PI / 5);
      const sx = Math.cos(angle) * r;
      const sy = -Math.sin(angle) * r;
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }
    ctx.closePath();
    ctx.fillStyle = '#22c55e';
    ctx.shadowColor = '#22c55e';
    ctx.shadowBlur = 15;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
    // label
    ctx.fillStyle = '#22c55e';
    ctx.font = "9px 'JetBrains Mono',monospace";
    ctx.textAlign = 'center';
    ctx.fillText('GOAL', x, y - 16);
  }

  // --- obstacles draw karo — red circles ---
  function drawObstacles() {
    obstacles.forEach(obs => {
      // glow
      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.r + 5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239,68,68,0.1)';
      ctx.fill();
      // solid circle
      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239,68,68,0.4)';
      ctx.fill();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }

  // --- robot draw karo — blue dot with trail ---
  function drawRobot() {
    // trail
    if (robot.trail.length > 1) {
      ctx.strokeStyle = 'rgba(74,158,255,0.4)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      robot.trail.forEach((p, i) => {
        if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
    }

    // robot body
    ctx.beginPath();
    ctx.arc(robot.x, robot.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = ACCENT;
    ctx.shadowColor = ACCENT;
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // stuck indicator
    if (robotStuck) {
      ctx.fillStyle = '#f59e0b';
      ctx.font = "bold 11px 'JetBrains Mono',monospace";
      ctx.textAlign = 'center';
      ctx.fillText('STUCK! (Local Minimum)', robot.x, robot.y - 14);
    }
  }

  // --- robot ek step move karo ---
  function moveRobot() {
    if (robotStuck) return;

    const dToGoal = Math.sqrt((robot.x - goal.x) ** 2 + (robot.y - goal.y) ** 2);
    if (dToGoal < 10) return; // goal pe pahunch gaya

    const g = gradient(robot.x, robot.y);
    const mag = Math.sqrt(g.gx * g.gx + g.gy * g.gy);
    if (mag < 0.1) {
      // gradient bahut chhota — stuck ho gaya
      robotStuck = true;
      return;
    }

    // negative gradient direction mein move karo — potential kam karne ke liye
    const step = Math.min(robotSpeed, robotSpeed * Math.min(1, mag / 20));
    robot.x -= (g.gx / mag) * step;
    robot.y -= (g.gy / mag) * step;

    // canvas ke andar rakh
    robot.x = Math.max(5, Math.min(canvasW - 5, robot.x));
    robot.y = Math.max(5, Math.min(CANVAS_HEIGHT - 5, robot.y));

    // trail mein add karo
    robot.trail.push({ x: robot.x, y: robot.y });
    if (robot.trail.length > 500) robot.trail.shift();

    // stuck detection — agar bohot zyada trail points same jagah hain
    if (robot.trail.length > 20) {
      const last20 = robot.trail.slice(-20);
      const avgX = last20.reduce((s, p) => s + p.x, 0) / 20;
      const avgY = last20.reduce((s, p) => s + p.y, 0) / 20;
      const variance = last20.reduce((s, p) => s + (p.x - avgX) ** 2 + (p.y - avgY) ** 2, 0) / 20;
      if (variance < 0.5) robotStuck = true;
    }
  }

  // --- main draw ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    drawHeatmap();
    drawVectorField();
    drawObstacles();
    drawGoal();
    drawRobot();

    // instructions
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.textAlign = 'left';
    ctx.fillText('Click: move goal  |  Shift+Click: add obstacle', 10, CANVAS_HEIGHT - 10);

    // stats
    const dToGoal = Math.sqrt((robot.x - goal.x) ** 2 + (robot.y - goal.y) ** 2);
    stats.textContent = `Dist to goal: ${dToGoal.toFixed(1)}  |  Obstacles: ${obstacles.length}  |  ${robotStuck ? 'STUCK in local minimum!' : dToGoal < 10 ? 'Goal reached!' : 'Navigating...'}`;
  }

  // --- animation loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    moveRobot();
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
