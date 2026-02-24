// ============================================================
// Brachistochrone — 4 curves pe race: straight, parabola, arc, cycloid
// Cycloid hamesha jeetega — calculus of variations ka classic result
// ============================================================

export function initBrachistochrone() {
  const container = document.getElementById('brachistochroneContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  let animationId = null, isVisible = false, canvasW = 0;

  // race state
  let isRacing = false;
  let raceTime = 0;
  let showLabels = true;

  // ball positions — parametric t (0 to 1) along each curve
  let ballT = [0, 0, 0, 0];
  let ballFinished = [false, false, false, false];
  let finishTimes = [0, 0, 0, 0];
  const dt = 0.016; // ~60fps time step
  const g = 500; // gravity (pixels/s^2)

  // curve definitions — start (left,top) to end (right,bottom)
  let startX = 0, startY = 0, endX = 0, endY = 0;

  // curve colors
  const curveColors = ['#f59e0b', '#ff4444', '#44ff44', '#22d3ee'];
  const curveNames = ['Straight', 'Parabola', 'Circular Arc', 'Cycloid'];

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

  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = "padding:5px 12px;font-size:11px;border-radius:6px;cursor:pointer;background:rgba(245,158,11,0.1);color:#b0b0b0;border:1px solid rgba(245,158,11,0.25);font-family:'JetBrains Mono',monospace;transition:all 0.2s ease;";
    btn.addEventListener('mouseenter', () => { btn.style.background = 'rgba(245,158,11,0.25)'; btn.style.color = '#e0e0e0'; });
    btn.addEventListener('mouseleave', () => { btn.style.background = 'rgba(245,158,11,0.1)'; btn.style.color = '#b0b0b0'; });
    btn.addEventListener('click', onClick);
    ctrl.appendChild(btn);
    return btn;
  }

  const startBtn = createButton('Start Race', () => { resetRace(); isRacing = true; });
  createButton('Reset', resetRace);

  const labelBtn = createButton('Labels: ON', () => {
    showLabels = !showLabels;
    labelBtn.textContent = 'Labels: ' + (showLabels ? 'ON' : 'OFF');
  });

  function resetRace() {
    isRacing = false;
    raceTime = 0;
    ballT = [0, 0, 0, 0];
    ballFinished = [false, false, false, false];
    finishTimes = [0, 0, 0, 0];
  }

  // --- curve point generators ---
  // sab curves (startX, startY) se (endX, endY) tak jayengi
  function computeEndpoints() {
    const margin = 60;
    startX = margin;
    startY = 50;
    endX = canvasW - margin;
    endY = CANVAS_HEIGHT - 80;
  }

  // 1. Straight line
  function straightPoint(t) {
    return {
      x: startX + t * (endX - startX),
      y: startY + t * (endY - startY),
    };
  }

  // 2. Parabola — y = a*x^2, adjusted to go through both endpoints
  function parabolaPoint(t) {
    // parametric parabola: x changes linearly, y is quadratic
    const x = startX + t * (endX - startX);
    const y = startY + t * t * (endY - startY);
    return { x, y };
  }

  // 3. Circular arc — circle jo dono points se guzre
  function arcPoint(t) {
    // arc jo start se end tak jaaye, neeche ki taraf curve ho
    const dx = endX - startX;
    const dy = endY - startY;
    const midX = (startX + endX) / 2;
    const midY = (startY + endY) / 2;

    // radius bada rakh taaki gentle curve ho
    const chord = Math.sqrt(dx * dx + dy * dy);
    const R = chord * 0.7;

    // center find karo — chord ke perpendicular, neeche ki taraf
    const nx = -dy / chord;
    const ny = dx / chord;
    // sagitta from R
    const h = R - Math.sqrt(Math.max(0, R * R - chord * chord / 4));
    const cx = midX + nx * h;
    const cy = midY + ny * h;

    // angles
    const a1 = Math.atan2(startY - cy, startX - cx);
    const a2 = Math.atan2(endY - cy, endX - cx);

    // angle range fix karo — clockwise jaana chahiye
    let angleRange = a2 - a1;
    if (angleRange > Math.PI) angleRange -= 2 * Math.PI;
    if (angleRange < -Math.PI) angleRange += 2 * Math.PI;

    const angle = a1 + t * angleRange;
    return {
      x: cx + R * Math.cos(angle),
      y: cy + R * Math.sin(angle),
    };
  }

  // 4. Cycloid — x = r(theta - sin(theta)), y = r(1 - cos(theta))
  // scaled to fit start to end
  function cycloidPoint(t) {
    // cycloid parameter theta from 0 to theta_max
    // theta_max chosen so cycloid connects start to end
    const theta_max = Math.PI * 1.8; // approximately good for most aspect ratios
    const theta = t * theta_max;

    // raw cycloid (starts at origin, goes right and down)
    const rawX = theta - Math.sin(theta);
    const rawY = 1 - Math.cos(theta);

    // scale to fit endpoints
    const rawEndX = theta_max - Math.sin(theta_max);
    const rawEndY = 1 - Math.cos(theta_max);

    const scaleX = (endX - startX) / rawEndX;
    const scaleY = (endY - startY) / Math.max(rawEndY, 0.01);

    return {
      x: startX + rawX * scaleX,
      y: startY + rawY * scaleY,
    };
  }

  const curvePointFns = [straightPoint, parabolaPoint, arcPoint, cycloidPoint];

  // --- arc length derivative for speed calculation ---
  // ball speed from energy conservation: v = sqrt(2g * delta_y)
  // ds/dt = v, so we advance t based on ds
  function advanceBall(curveIdx, currentT) {
    if (currentT >= 1) return 1;

    const eps = 0.001;
    const fn = curvePointFns[curveIdx];
    const p = fn(currentT);

    // height drop from start — energy conservation
    const deltaY = Math.max(0, p.y - startY);
    const speed = Math.sqrt(2 * g * deltaY + 1); // +1 so it starts moving

    // arc length derivative: ds/dt at current t
    const p1 = fn(Math.min(currentT + eps, 1));
    const p0 = fn(Math.max(currentT - eps, 0));
    const dxdt = (p1.x - p0.x) / (2 * eps);
    const dydt = (p1.y - p0.y) / (2 * eps);
    const dsdt = Math.sqrt(dxdt * dxdt + dydt * dydt);

    if (dsdt < 0.1) return Math.min(1, currentT + 0.01);

    // dt_param = speed * dt_real / dsdt
    const dtParam = speed * dt / dsdt;
    return Math.min(1, currentT + dtParam);
  }

  // --- resize ---
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    computeEndpoints();
  }
  resize();
  window.addEventListener('resize', resize);

  // --- draw ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    computeEndpoints();

    // grid
    ctx.strokeStyle = 'rgba(245,158,11,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const y = (i / 10) * CANVAS_HEIGHT;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvasW, y); ctx.stroke();
    }

    // start/end markers
    ctx.beginPath();
    ctx.arc(startX, startY, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(245,158,11,0.3)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(245,158,11,0.5)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(endX, endY, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(245,158,11,0.3)';
    ctx.fill();
    ctx.stroke();

    // finish line
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(245,158,11,0.15)';
    ctx.beginPath();
    ctx.moveTo(endX, endY - 40);
    ctx.lineTo(endX, endY + 40);
    ctx.stroke();
    ctx.setLineDash([]);

    // curves draw karo
    for (let c = 0; c < 4; c++) {
      const fn = curvePointFns[c];
      ctx.beginPath();
      for (let t = 0; t <= 1; t += 0.005) {
        const p = fn(t);
        if (t === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.strokeStyle = curveColors[c];
      ctx.lineWidth = ballFinished[c] ? 2.5 : 1.5;
      ctx.globalAlpha = ballFinished[c] ? 0.9 : 0.4;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // curve label
      if (showLabels) {
        const lp = fn(0.35);
        ctx.font = "10px 'JetBrains Mono',monospace";
        ctx.fillStyle = curveColors[c];
        ctx.textAlign = 'left';
        ctx.globalAlpha = 0.6;
        ctx.fillText(curveNames[c], lp.x + 5, lp.y - 8);
        ctx.globalAlpha = 1;
      }
    }

    // balls draw karo
    for (let c = 0; c < 4; c++) {
      if (ballT[c] <= 0 && !isRacing) continue;
      const fn = curvePointFns[c];
      const p = fn(ballT[c]);

      // glow
      ctx.beginPath();
      ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = curveColors[c];
      ctx.shadowColor = curveColors[c];
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;

      // inner bright dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.globalAlpha = 0.6;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // title
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.fillText('BRACHISTOCHRONE', 8, 14);

    // race timer
    if (isRacing || raceTime > 0) {
      ctx.textAlign = 'right';
      ctx.fillStyle = 'rgba(245,158,11,0.5)';
      ctx.fillText('t=' + raceTime.toFixed(2) + 's', canvasW - 8, 14);
    }

    // finish times dikhao — right side
    let finished = ballFinished.some(f => f);
    if (finished) {
      // rank order nikalo
      const ranked = [0, 1, 2, 3].filter(i => ballFinished[i]).sort((a, b) => finishTimes[a] - finishTimes[b]);
      const still = [0, 1, 2, 3].filter(i => !ballFinished[i]);

      ctx.font = "11px 'JetBrains Mono',monospace";
      let yPos = 35;
      for (let r = 0; r < ranked.length; r++) {
        const c = ranked[r];
        const medal = r === 0 ? '\u{1F947}' : r === 1 ? '\u{1F948}' : r === 2 ? '\u{1F949}' : '';
        ctx.textAlign = 'right';
        ctx.fillStyle = curveColors[c];
        ctx.fillText(curveNames[c] + ' ' + finishTimes[c].toFixed(3) + 's', canvasW - 8, yPos);
        yPos += 16;
      }
    }

    // hint text
    if (!isRacing && raceTime === 0) {
      ctx.font = "12px 'JetBrains Mono',monospace";
      ctx.fillStyle = 'rgba(176,176,176,0.2)';
      ctx.textAlign = 'center';
      ctx.fillText('press Start Race — cycloid hamesha jeetega', canvasW / 2, CANVAS_HEIGHT - 15);
    }
  }

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    if (isRacing) {
      raceTime += dt;
      let allDone = true;
      for (let c = 0; c < 4; c++) {
        if (ballFinished[c]) continue;
        allDone = false;
        ballT[c] = advanceBall(c, ballT[c]);
        if (ballT[c] >= 1) {
          ballFinished[c] = true;
          finishTimes[c] = raceTime;
          ballT[c] = 1;
        }
      }
      if (allDone) isRacing = false;
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
