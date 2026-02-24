// ============================================================
// Orbital Slingshot — click+drag se spacecraft launch karo
// Central star + orbiting planets, gravitational slingshot effect
// RK4 integration, velocity-colored trail
// ============================================================
export function initOrbitalSlingshot() {
  const container = document.getElementById('orbitalSlingshotContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  let animationId = null, isVisible = false, canvasW = 0;

  // --- DOM setup ---
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
    vl.textContent = Number(val).toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0);
    w.appendChild(vl);
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      vl.textContent = v.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0);
      onChange(v);
    });
    ctrl.appendChild(w);
    return sl;
  }

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

  // --- physics constants ---
  const G = 500;  // gravitational constant — tune kiya hai visual ke liye
  const starMass = 200;
  let planetMass = 30;  // adjustable
  let showTrail = true;

  // sun center mein
  const star = { x: 0, y: 0, mass: starMass, radius: 18, color: '#ffcc00' };

  // do planets circular orbits mein
  let planets = [
    { angle: 0, orbitR: 120, speed: 0.012, mass: planetMass, radius: 8, color: '#4488ff', x: 0, y: 0 },
    { angle: Math.PI, orbitR: 200, speed: 0.007, mass: planetMass * 0.7, radius: 6, color: '#66ddaa', x: 0, y: 0 },
  ];

  // spacecraft state
  let spacecraft = null; // { x, y, vx, vy }
  let trail = [];        // [{ x, y, speed }]
  const MAX_TRAIL = 2000;
  let time = 0;

  // drag state — launch direction decide karne ke liye
  let isDragging = false;
  let dragStart = null;
  let dragEnd = null;

  // controls banao
  const slMass = createSlider('planet mass', 5, 80, 1, planetMass, v => {
    planetMass = v;
    planets[0].mass = v;
    planets[1].mass = v * 0.7;
  });
  const trailBtn = createButton('Trail: ON', () => {
    showTrail = !showTrail;
    trailBtn.textContent = showTrail ? 'Trail: ON' : 'Trail: OFF';
  });
  createButton('Reset', () => {
    spacecraft = null;
    trail = [];
    time = 0;
    planets[0].angle = 0;
    planets[1].angle = Math.PI;
  });

  // --- coordinate system ---
  // simulation space: center at (0,0), ~300 units radius visible
  const SIM_SCALE = 1.3; // kitna zoom hai
  function simToCanvas(sx, sy) {
    const scale = Math.min(canvasW, CANVAS_HEIGHT) / (600 / SIM_SCALE);
    return [canvasW / 2 + sx * scale, CANVAS_HEIGHT / 2 + sy * scale];
  }
  function canvasToSim(cx, cy) {
    const scale = Math.min(canvasW, CANVAS_HEIGHT) / (600 / SIM_SCALE);
    return [(cx - canvasW / 2) / scale, (cy - CANVAS_HEIGHT / 2) / scale];
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- click+drag se launch ---
  canvas.addEventListener('pointerdown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    isDragging = true;
    dragStart = { cx, cy };
    dragEnd = { cx, cy };
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!isDragging) return;
    const rect = canvas.getBoundingClientRect();
    dragEnd = { cx: e.clientX - rect.left, cy: e.clientY - rect.top };
  });
  canvas.addEventListener('pointerup', () => {
    if (!isDragging || !dragStart || !dragEnd) { isDragging = false; return; }
    // drag direction = launch direction (opposite of drag, like slingshot)
    const [sx, sy] = canvasToSim(dragStart.cx, dragStart.cy);
    const dx = dragStart.cx - dragEnd.cx;
    const dy = dragStart.cy - dragEnd.cy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 5) { isDragging = false; return; } // bahut chhota drag — ignore

    // velocity — drag length proportional to speed
    const speedScale = 0.015;
    const scale = Math.min(canvasW, CANVAS_HEIGHT) / (600 / SIM_SCALE);
    const vx = dx / scale * speedScale * 60;
    const vy = dy / scale * speedScale * 60;

    spacecraft = { x: sx, y: sy, vx, vy };
    trail = [];
    isDragging = false;
    dragStart = null;
    dragEnd = null;
  });
  canvas.addEventListener('pointerleave', () => { isDragging = false; });

  // --- gravitational acceleration ---
  // F = -GM/r^2, direction towards body
  function getAcceleration(x, y) {
    let ax = 0, ay = 0;
    // star se gravity
    const dxs = star.x - x, dys = star.y - y;
    const r2s = dxs * dxs + dys * dys + 100; // softening — collision prevent
    const rs = Math.sqrt(r2s);
    const fs = G * star.mass / r2s;
    ax += fs * dxs / rs;
    ay += fs * dys / rs;

    // planets se gravity
    for (const p of planets) {
      const dxp = p.x - x, dyp = p.y - y;
      const r2p = dxp * dxp + dyp * dyp + 25;
      const rp = Math.sqrt(r2p);
      const fp = G * p.mass / r2p;
      ax += fp * dxp / rp;
      ay += fp * dyp / rp;
    }
    return [ax, ay];
  }

  // --- RK4 integration — proper orbital mechanics ke liye zaroori ---
  function rk4Step(x, y, vx, vy, dt) {
    const [ax1, ay1] = getAcceleration(x, y);
    const x2 = x + vx * dt / 2, y2 = y + vy * dt / 2;
    const vx2 = vx + ax1 * dt / 2, vy2 = vy + ay1 * dt / 2;

    const [ax2, ay2] = getAcceleration(x2, y2);
    const x3 = x + vx2 * dt / 2, y3 = y + vy2 * dt / 2;
    const vx3 = vx + ax2 * dt / 2, vy3 = vy + ay2 * dt / 2;

    const [ax3, ay3] = getAcceleration(x3, y3);
    const x4 = x + vx3 * dt, y4 = y + vy3 * dt;
    const vx4 = vx + ax3 * dt, vy4 = vy + ay3 * dt;

    const [ax4, ay4] = getAcceleration(x4, y4);

    return [
      x + dt / 6 * (vx + 2 * vx2 + 2 * vx3 + vx4),
      y + dt / 6 * (vy + 2 * vy2 + 2 * vy3 + vy4),
      vx + dt / 6 * (ax1 + 2 * ax2 + 2 * ax3 + ax4),
      vy + dt / 6 * (ay1 + 2 * ay2 + 2 * ay3 + ay4),
    ];
  }

  // --- update planets orbital positions ---
  function updatePlanets() {
    for (const p of planets) {
      p.angle += p.speed;
      p.x = Math.cos(p.angle) * p.orbitR;
      p.y = Math.sin(p.angle) * p.orbitR;
    }
  }

  // --- velocity to color — blue=slow, amber=mid, red=fast ---
  function speedColor(speed) {
    const maxSpeed = 8;
    const s = Math.min(1, speed / maxSpeed);
    let r, g, b;
    if (s < 0.33) {
      // neela — slow
      const t = s / 0.33;
      r = 30 + 40 * t; g = 80 + 60 * t; b = 200 + 55 * t;
    } else if (s < 0.66) {
      // amber/peela — medium
      const t = (s - 0.33) / 0.33;
      r = 70 + 185 * t; g = 140 + 40 * t; b = 255 - 200 * t;
    } else {
      // laal — fast
      const t = (s - 0.66) / 0.34;
      r = 255; g = 180 - 150 * t; b = 55 - 55 * t;
    }
    return `rgb(${r|0},${g|0},${b|0})`;
  }

  // --- simulation step ---
  function step() {
    updatePlanets();
    time++;

    if (!spacecraft) return;

    // multiple RK4 substeps — accuracy ke liye
    const dt = 0.05;
    const substeps = 4;
    for (let i = 0; i < substeps; i++) {
      const [nx, ny, nvx, nvy] = rk4Step(
        spacecraft.x, spacecraft.y,
        spacecraft.vx, spacecraft.vy, dt
      );
      spacecraft.x = nx; spacecraft.y = ny;
      spacecraft.vx = nvx; spacecraft.vy = nvy;
    }

    // trail mein add karo
    const speed = Math.sqrt(spacecraft.vx * spacecraft.vx + spacecraft.vy * spacecraft.vy);
    trail.push({ x: spacecraft.x, y: spacecraft.y, speed });
    if (trail.length > MAX_TRAIL) trail.shift();

    // agar bahut door chala gaya toh hata do
    const r = Math.sqrt(spacecraft.x * spacecraft.x + spacecraft.y * spacecraft.y);
    if (r > 500) {
      spacecraft = null;
      return; // spacecraft gayab — aage check ki zarurat nahi
    }
    // star se collision check — pehle null check karo
    if (r < star.radius * 0.5) {
      spacecraft = null;
    }
  }

  // --- render ---
  function render() {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvasW, CANVAS_HEIGHT);

    // background stars — subtle dots
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    // deterministic random positions — seed based
    for (let i = 0; i < 80; i++) {
      const sx = ((i * 7919 + 1234) % 1000) / 1000 * canvasW;
      const sy = ((i * 6271 + 5678) % 1000) / 1000 * CANVAS_HEIGHT;
      const size = ((i * 3571) % 3) * 0.3 + 0.5;
      ctx.fillRect(sx, sy, size, size);
    }

    // planet orbit paths — dashed circles
    ctx.setLineDash([3, 6]);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (const p of planets) {
      const [cx, cy] = simToCanvas(0, 0);
      const scale = Math.min(canvasW, CANVAS_HEIGHT) / (600 / SIM_SCALE);
      ctx.beginPath();
      ctx.arc(cx, cy, p.orbitR * scale, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // trail draw karo — velocity colored
    if (showTrail && trail.length > 1) {
      for (let i = 1; i < trail.length; i++) {
        const [x1, y1] = simToCanvas(trail[i-1].x, trail[i-1].y);
        const [x2, y2] = simToCanvas(trail[i].x, trail[i].y);
        // purane trail segments fade karo
        const alpha = 0.2 + 0.6 * (i / trail.length);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = speedColor(trail[i].speed);
        ctx.globalAlpha = alpha;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // star (sun) draw karo — glowing effect
    const [scx, scy] = simToCanvas(star.x, star.y);
    // glow
    const grd = ctx.createRadialGradient(scx, scy, 0, scx, scy, star.radius * 2.5);
    grd.addColorStop(0, 'rgba(255,200,50,0.3)');
    grd.addColorStop(1, 'rgba(255,200,50,0)');
    ctx.fillStyle = grd;
    ctx.fillRect(scx - star.radius * 3, scy - star.radius * 3, star.radius * 6, star.radius * 6);
    // solid star
    ctx.beginPath();
    ctx.arc(scx, scy, star.radius, 0, Math.PI * 2);
    ctx.fillStyle = star.color;
    ctx.shadowColor = star.color;
    ctx.shadowBlur = 20;
    ctx.fill();
    ctx.shadowBlur = 0;

    // planets draw karo
    for (const p of planets) {
      const [px, py] = simToCanvas(p.x, p.y);
      ctx.beginPath();
      ctx.arc(px, py, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;
      // atmosphere effect — hex color ko alpha ke saath use karte hain
      ctx.beginPath();
      ctx.arc(px, py, p.radius + 3, 0, Math.PI * 2);
      ctx.globalAlpha = 0.2;
      ctx.strokeStyle = p.color;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // spacecraft draw karo
    if (spacecraft) {
      const [sx, sy] = simToCanvas(spacecraft.x, spacecraft.y);
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.shadowColor = '#ffffff';
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;

      // velocity vector dikhao — chhota arrow
      const speed = Math.sqrt(spacecraft.vx * spacecraft.vx + spacecraft.vy * spacecraft.vy);
      if (speed > 0.1) {
        const scale = Math.min(canvasW, CANVAS_HEIGHT) / (600 / SIM_SCALE);
        const arrowLen = Math.min(30, speed * 3);
        const nx = spacecraft.vx / speed, ny = spacecraft.vy / speed;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(sx + nx * arrowLen, sy + ny * arrowLen);
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // drag arrow — launch direction dikhao
    if (isDragging && dragStart && dragEnd) {
      const dx = dragStart.cx - dragEnd.cx;
      const dy = dragStart.cy - dragEnd.cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > 5) {
        // arrow dragStart se direction mein
        ctx.beginPath();
        ctx.moveTo(dragStart.cx, dragStart.cy);
        ctx.lineTo(dragStart.cx + dx, dragStart.cy + dy);
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        // arrowhead
        const angle = Math.atan2(dy, dx);
        const headLen = 10;
        ctx.beginPath();
        ctx.moveTo(dragStart.cx + dx, dragStart.cy + dy);
        ctx.lineTo(
          dragStart.cx + dx - headLen * Math.cos(angle - 0.3),
          dragStart.cy + dy - headLen * Math.sin(angle - 0.3)
        );
        ctx.moveTo(dragStart.cx + dx, dragStart.cy + dy);
        ctx.lineTo(
          dragStart.cx + dx - headLen * Math.cos(angle + 0.3),
          dragStart.cy + dy - headLen * Math.sin(angle + 0.3)
        );
        ctx.stroke();
        // launch position dot
        ctx.beginPath();
        ctx.arc(dragStart.cx, dragStart.cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = ACCENT;
        ctx.fill();
      }
    }

    // labels
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.4)';
    ctx.textAlign = 'left';
    ctx.fillText('ORBITAL SLINGSHOT', 8, 14);
    if (!spacecraft && trail.length === 0) {
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(176,176,176,0.3)';
      ctx.fillText('click + drag to launch spacecraft', canvasW / 2, CANVAS_HEIGHT - 12);
    }
    // slingshot detection — agar planet ke paas se guzra
    if (spacecraft) {
      for (const p of planets) {
        const dx = spacecraft.x - p.x;
        const dy = spacecraft.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < p.orbitR * 0.15) {
          ctx.textAlign = 'right';
          ctx.fillStyle = 'rgba(245,158,11,0.6)';
          ctx.fillText('SLINGSHOT!', canvasW - 8, 14);
        }
      }
    }

    // speed legend — bottom right
    if (spacecraft || trail.length > 0) {
      ctx.textAlign = 'right';
      ctx.fillStyle = 'rgba(176,176,176,0.3)';
      ctx.font = "9px 'JetBrains Mono',monospace";
      const speed = spacecraft
        ? Math.sqrt(spacecraft.vx * spacecraft.vx + spacecraft.vy * spacecraft.vy).toFixed(1)
        : '--';
      ctx.fillText('v=' + speed, canvasW - 8, CANVAS_HEIGHT - 8);
    }
  }

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    step();
    render();
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
