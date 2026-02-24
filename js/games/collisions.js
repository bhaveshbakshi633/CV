// ============================================================
// 2D Elastic Collisions — Momentum + Energy conservation ka live demo
// Click-drag se balls spawn karo, collisions dekho, physics samjho
// ============================================================

// yahi function bahar export hoga — container dhundho, canvas banao, physics chalao
export function initCollisions() {
  const container = document.getElementById('collisionsContainer');
  if (!container) {
    console.warn('collisionsContainer nahi mila bhai, collisions demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 350;
  const MIN_MASS = 1;
  const MAX_MASS = 20;
  const HOLD_MASS_RATE = 8; // mass per second jab hold karo
  const MAX_BALLS_DEFAULT = 30;
  const TRAIL_LENGTH = 18; // chhoti trail — subtle effect ke liye
  const MAX_INITIAL_VEL = 300;
  const VELOCITY_ARROW_SCALE = 0.15; // velocity vector ko arrow mein convert karne ka scale
  const FLASH_DURATION = 0.25; // collision flash kitni der dikhe (seconds)
  const ACCENT = '#f59e0b'; // amber accent — portfolio ka theme
  const ACCENT_RGB = '245,158,11';

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let balls = []; // [{x, y, vx, vy, mass, radius, trail, color, flash}]
  let restitution = 1.0; // 1 = perfectly elastic, 0 = perfectly inelastic
  let gravity = 0; // 0 = no gravity, positive = downward
  let maxBalls = MAX_BALLS_DEFAULT;
  let slowMo = false; // 0.25x speed toggle
  let isPaused = false;

  // collision flash effects — jab do balls takraayein toh sparkle dikhao
  let flashEffects = []; // [{x, y, radius, age, maxAge, color}]

  // spawn state — jab user click-drag se ball bana raha hai
  let isSpawning = false;
  let spawnStartTime = 0;
  let spawnX = 0, spawnY = 0;
  let spawnDragX = 0, spawnDragY = 0;
  let isDragging = false;

  // animation state
  let animationId = null;
  let isVisible = false;
  let lastTime = 0;

  // --- DOM structure banao — pehle sab saaf karo ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(245,158,11,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:rgba(2,2,8,0.5)',
  ].join(';');
  container.appendChild(canvas);

  // stats bar — momentum aur energy dikhane ke liye
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'margin-top:8px',
    'padding:8px 12px',
    'background:rgba(245,158,11,0.05)',
    'border:1px solid rgba(245,158,11,0.12)',
    'border-radius:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:#b0b0b0',
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'align-items:center',
  ].join(';');
  container.appendChild(statsDiv);

  // individual stat elements
  const statBalls = document.createElement('span');
  const statMomentumX = document.createElement('span');
  const statMomentumY = document.createElement('span');
  const statKE = document.createElement('span');
  [statBalls, statMomentumX, statMomentumY, statKE].forEach(el => statsDiv.appendChild(el));

  function updateStats() {
    // total momentum aur kinetic energy calculate karo
    let px = 0, py = 0, ke = 0;
    for (const b of balls) {
      px += b.mass * b.vx;
      py += b.mass * b.vy;
      ke += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy);
    }
    statBalls.textContent = 'Balls: ' + balls.length;
    statMomentumX.textContent = 'Px: ' + px.toFixed(1);
    statMomentumY.textContent = 'Py: ' + py.toFixed(1);
    statKE.textContent = 'KE: ' + ke.toFixed(1);
  }

  // controls section
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // button factory — consistent styling ke liye
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:6px 14px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(245,158,11,0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(245,158,11,0.25)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(245,158,11,0.25)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(245,158,11,0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // slider factory — label + slider + value display
  function createSlider(label, min, max, step, initial, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:"JetBrains Mono",monospace;';
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(initial);
    slider.style.cssText = 'width:70px;height:4px;accent-color:rgba(245,158,11,0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const val = document.createElement('span');
    val.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:"JetBrains Mono",monospace;min-width:32px;';
    val.textContent = String(initial);
    wrapper.appendChild(val);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      onChange(v, val);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, val };
  }

  // --- Controls banao ---

  // Pause/Play
  const pauseBtn = createButton('Pause', () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Play' : 'Pause';
  });

  // Clear all
  createButton('Clear', () => {
    balls = [];
    flashEffects = [];
  });

  // Slow-mo toggle
  const slowBtn = createButton('Slow-Mo', () => {
    slowMo = !slowMo;
    slowBtn.textContent = slowMo ? 'Normal' : 'Slow-Mo';
    slowBtn.style.color = slowMo ? ACCENT : '#b0b0b0';
    slowBtn.style.borderColor = slowMo ? 'rgba(245,158,11,0.5)' : 'rgba(245,158,11,0.25)';
  });

  // Presets dropdown
  const presetSelect = document.createElement('select');
  presetSelect.style.cssText = [
    'padding:6px 10px',
    'font-size:12px',
    'border-radius:6px',
    'cursor:pointer',
    'background:rgba(245,158,11,0.1)',
    'color:#b0b0b0',
    'border:1px solid rgba(245,158,11,0.25)',
    'font-family:"JetBrains Mono",monospace',
  ].join(';');

  const presets = ['Presets...', 'Pool Break', 'Random', "Newton's Cradle"];
  presets.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    opt.style.background = '#1a1a1a';
    opt.style.color = '#b0b0b0';
    presetSelect.appendChild(opt);
  });

  presetSelect.addEventListener('change', () => {
    loadPreset(presetSelect.value);
    presetSelect.value = 'Presets...';
  });
  controlsDiv.appendChild(presetSelect);

  // Restitution slider — 0 = perfectly inelastic, 1 = perfectly elastic
  createSlider('e:', 0, 1, 0.05, 1.0, (v, valEl) => {
    restitution = v;
    valEl.textContent = v.toFixed(2);
  });

  // Gravity slider — 0 to 500
  createSlider('g:', 0, 500, 10, 0, (v, valEl) => {
    gravity = v;
    valEl.textContent = v === 0 ? 'off' : v.toFixed(0);
  });

  // Max balls slider
  createSlider('max:', 5, 60, 1, MAX_BALLS_DEFAULT, (v, valEl) => {
    maxBalls = Math.round(v);
    valEl.textContent = String(maxBalls);
  });

  // --- Canvas sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = containerWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Mass se color decide karo ---
  // chhoti mass = halka amber, badi mass = dark amber/brown
  function massToColor(mass) {
    const t = Math.min(Math.max((mass - MIN_MASS) / (MAX_MASS - MIN_MASS), 0), 1);
    // light amber → deep orange → dark brown
    const r = Math.round(255 - t * 60);
    const g = Math.round(200 - t * 130);
    const b = Math.round(50 - t * 30);
    return { r, g, b };
  }

  // radius proportional to sqrt(mass) — area ~ mass
  function ballRadius(mass) {
    return Math.max(6, Math.sqrt(mass / MIN_MASS) * 8);
  }

  // --- Ball creation ---
  function createBall(x, y, vx, vy, mass) {
    const color = massToColor(mass);
    const radius = ballRadius(mass);
    return {
      x, y, vx, vy, mass, radius,
      trail: [], // [{x, y}]
      color,
      flash: 0, // collision flash timer — > 0 matlab abhi collision hua
    };
  }

  // --- Presets ---
  function loadPreset(name) {
    balls = [];
    flashEffects = [];
    const cx = canvasW / 2;
    const cy = canvasH / 2;

    switch (name) {
      case 'Pool Break': {
        // pool table waala triangle formation — ek cue ball tez chalegi
        const ballMass = 3;
        const r = ballRadius(ballMass);
        const spacing = r * 2.15; // thoda gap rakh taaki overlap na ho

        // triangle formation — 5 rows, right side mein
        const startX = cx + 80;
        const startY = cy;
        let count = 0;
        for (let row = 0; row < 5; row++) {
          for (let col = 0; col <= row; col++) {
            const bx = startX + row * spacing * 0.866; // cos(30) ≈ 0.866
            const by = startY + (col - row / 2) * spacing;
            balls.push(createBall(bx, by, 0, 0, ballMass));
            count++;
          }
        }

        // cue ball — left side se tez chalegi
        balls.push(createBall(cx - 120, cy, 250, (Math.random() - 0.5) * 20, ballMass));
        break;
      }

      case 'Random': {
        // 8-12 random balls — alag alag mass aur velocity
        const count = 8 + Math.floor(Math.random() * 5);
        for (let i = 0; i < count; i++) {
          const mass = MIN_MASS + Math.random() * (MAX_MASS - MIN_MASS);
          const r = ballRadius(mass);
          const x = r + Math.random() * (canvasW - 2 * r);
          const y = r + Math.random() * (canvasH - 2 * r);
          const speed = 30 + Math.random() * 120;
          const angle = Math.random() * Math.PI * 2;
          balls.push(createBall(x, y, Math.cos(angle) * speed, Math.sin(angle) * speed, mass));
        }
        break;
      }

      case "Newton's Cradle": {
        // line mein balls — ek end se push karo
        const ballMass = 5;
        const r = ballRadius(ballMass);
        const spacing = r * 2.05;
        const count = 7;
        const totalWidth = (count - 1) * spacing;
        const startX = cx - totalWidth / 2;

        for (let i = 0; i < count; i++) {
          balls.push(createBall(startX + i * spacing, cy, 0, 0, ballMass));
        }

        // pehli ball ko tez velocity do — left se aayegi
        // actually cradle mein pehli ball alag hoti hai, toh usse door rakh ke velocity dete hain
        balls[0].x -= spacing * 2;
        balls[0].vx = 180;

        // restitution 1 set karo — cradle tabhi kaam karega
        restitution = 1.0;
        // gravity off karo
        gravity = 0;
        break;
      }
    }
  }

  // --- Physics Step ---
  function physicsStep(dt) {
    if (balls.length === 0) return;

    // gravity apply karo — sabko neeche kheencho
    if (gravity > 0) {
      for (const b of balls) {
        b.vy += gravity * dt;
      }
    }

    // position update
    for (const b of balls) {
      b.x += b.vx * dt;
      b.y += b.vy * dt;
    }

    // wall collisions — canvas ke boundaries se bounce karo
    for (const b of balls) {
      // left wall
      if (b.x - b.radius < 0) {
        b.x = b.radius;
        b.vx = Math.abs(b.vx) * restitution;
      }
      // right wall
      if (b.x + b.radius > canvasW) {
        b.x = canvasW - b.radius;
        b.vx = -Math.abs(b.vx) * restitution;
      }
      // top wall
      if (b.y - b.radius < 0) {
        b.y = b.radius;
        b.vy = Math.abs(b.vy) * restitution;
      }
      // bottom wall
      if (b.y + b.radius > canvasH) {
        b.y = canvasH - b.radius;
        b.vy = -Math.abs(b.vy) * restitution;
      }
    }

    // ball-ball collision detection aur response
    handleBallCollisions();

    // trail update — position history rakh
    for (const b of balls) {
      b.trail.push({ x: b.x, y: b.y });
      if (b.trail.length > TRAIL_LENGTH) {
        b.trail.shift();
      }
    }

    // flash timers decay karo
    for (const b of balls) {
      if (b.flash > 0) b.flash -= dt;
    }

    // flash effects decay
    for (let i = flashEffects.length - 1; i >= 0; i--) {
      flashEffects[i].age += dt;
      if (flashEffects[i].age >= flashEffects[i].maxAge) {
        flashEffects.splice(i, 1);
      }
    }
  }

  // --- Ball-Ball Collision ---
  // 2D elastic/inelastic collision response with restitution coefficient
  function handleBallCollisions() {
    for (let i = 0; i < balls.length; i++) {
      for (let j = i + 1; j < balls.length; j++) {
        const a = balls[i];
        const b = balls[j];

        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const distSq = dx * dx + dy * dy;
        const minDist = a.radius + b.radius;

        // agar distance < sum of radii toh collision hai
        if (distSq < minDist * minDist && distSq > 0.0001) {
          const dist = Math.sqrt(distSq);

          // collision normal — a se b ki taraf
          const nx = dx / dist;
          const ny = dy / dist;

          // relative velocity along collision normal
          const dvx = a.vx - b.vx;
          const dvy = a.vy - b.vy;
          const dvDotN = dvx * nx + dvy * ny;

          // sirf tab resolve karo jab balls ek doosre ki taraf aa rahe hain
          // (relative velocity collision normal ke along positive hai)
          if (dvDotN > 0) {
            // impulse magnitude — momentum + energy conservation with restitution
            // j = -(1 + e) * (v_rel . n) / (1/m_a + 1/m_b)
            const impulse = (1 + restitution) * dvDotN / (1 / a.mass + 1 / b.mass);

            // velocity update — impulse apply karo
            a.vx -= (impulse / a.mass) * nx;
            a.vy -= (impulse / a.mass) * ny;
            b.vx += (impulse / b.mass) * nx;
            b.vy += (impulse / b.mass) * ny;

            // balls ko alag karo — overlap hata do
            const overlap = minDist - dist;
            const separationRatio = overlap / (1 / a.mass + 1 / b.mass);
            a.x -= (separationRatio / a.mass) * nx;
            a.y -= (separationRatio / a.mass) * ny;
            b.x += (separationRatio / b.mass) * nx;
            b.y += (separationRatio / b.mass) * ny;

            // collision flash effect — dono balls pe
            a.flash = FLASH_DURATION;
            b.flash = FLASH_DURATION;

            // sparkle effect collision point pe
            const collisionX = a.x + nx * a.radius;
            const collisionY = a.y + ny * a.radius;
            const impactSpeed = Math.abs(dvDotN);
            addFlashEffect(collisionX, collisionY, impactSpeed);
          }
        }
      }
    }
  }

  // collision flash — sparkle particles spawn karo
  function addFlashEffect(x, y, impactSpeed) {
    const intensity = Math.min(impactSpeed / 200, 1);
    const count = 3 + Math.floor(intensity * 5);
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 20 + Math.random() * 60 * intensity;
      flashEffects.push({
        x: x,
        y: y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        radius: 1 + Math.random() * 2 * intensity,
        age: 0,
        maxAge: 0.15 + Math.random() * 0.2,
      });
    }
  }

  // --- Mouse/Touch Events — ball spawn karo ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  canvas.addEventListener('mousedown', (e) => {
    const pos = getCanvasPos(e);
    isSpawning = true;
    isDragging = false;
    spawnStartTime = performance.now();
    spawnX = pos.x;
    spawnY = pos.y;
    spawnDragX = pos.x;
    spawnDragY = pos.y;
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isSpawning) return;
    const pos = getCanvasPos(e);
    spawnDragX = pos.x;
    spawnDragY = pos.y;
    const dragDist = Math.sqrt(Math.pow(spawnDragX - spawnX, 2) + Math.pow(spawnDragY - spawnY, 2));
    if (dragDist > 5) isDragging = true;
  });

  canvas.addEventListener('mouseup', () => {
    if (!isSpawning) return;
    finishSpawning();
  });

  // touch support
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    isSpawning = true;
    isDragging = false;
    spawnStartTime = performance.now();
    spawnX = pos.x;
    spawnY = pos.y;
    spawnDragX = pos.x;
    spawnDragY = pos.y;
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isSpawning) return;
    const pos = getCanvasPos(e);
    spawnDragX = pos.x;
    spawnDragY = pos.y;
    const dragDist = Math.sqrt(Math.pow(spawnDragX - spawnX, 2) + Math.pow(spawnDragY - spawnY, 2));
    if (dragDist > 5) isDragging = true;
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (!isSpawning) return;
    finishSpawning();
  }, { passive: false });

  function finishSpawning() {
    isSpawning = false;

    // max balls check — zyada hai toh spawn mat karo
    if (balls.length >= maxBalls) return;

    // mass — hold duration se proportional
    const holdTime = (performance.now() - spawnStartTime) / 1000;
    const mass = Math.min(MIN_MASS + holdTime * HOLD_MASS_RATE, MAX_MASS);

    // velocity — drag direction se (drag ke SAME direction mein)
    let vx = 0, vy = 0;
    if (isDragging) {
      vx = (spawnDragX - spawnX) * 2.5;
      vy = (spawnDragY - spawnY) * 2.5;
      // cap velocity
      const vel = Math.sqrt(vx * vx + vy * vy);
      if (vel > MAX_INITIAL_VEL) {
        vx = (vx / vel) * MAX_INITIAL_VEL;
        vy = (vy / vel) * MAX_INITIAL_VEL;
      }
    }

    balls.push(createBall(spawnX, spawnY, vx, vy, mass));
  }

  // --- Drawing ---
  function draw() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvasW, canvasH);

    // trails draw karo — subtle fading lines
    drawTrails(ctx);

    // flash effects — collision sparks
    drawFlashEffects(ctx);

    // balls draw karo
    drawBalls(ctx);

    // spawn preview — jab user drag kar raha hai
    if (isSpawning) {
      drawSpawnPreview(ctx);
    }

    // hint jab koi ball nahi hai
    if (balls.length === 0 && !isSpawning) {
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(245,158,11,0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('click to spawn \u2022 hold for mass \u2022 drag for velocity', canvasW / 2, canvasH / 2);
    }

    // stats update karo
    updateStats();
  }

  function drawTrails(ctx) {
    for (const ball of balls) {
      if (ball.trail.length < 2) continue;
      const { r, g, b } = ball.color;

      // fading trail segments — purana faded, naya thoda bright
      for (let i = 1; i < ball.trail.length; i++) {
        const alpha = (i / ball.trail.length) * 0.3;
        ctx.beginPath();
        ctx.moveTo(ball.trail[i - 1].x, ball.trail[i - 1].y);
        ctx.lineTo(ball.trail[i].x, ball.trail[i].y);
        ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
        ctx.lineWidth = Math.max(1, ball.radius * 0.3);
        ctx.stroke();
      }
    }
  }

  function drawFlashEffects(ctx) {
    for (const f of flashEffects) {
      const progress = f.age / f.maxAge;
      const alpha = 1 - progress;
      // particles apni position update karo
      f.x += f.vx * (1 / 60); // approximate frame dt
      f.y += f.vy * (1 / 60);

      ctx.beginPath();
      ctx.arc(f.x, f.y, f.radius * (1 - progress * 0.5), 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',' + (alpha * 0.8) + ')';
      ctx.fill();
    }
  }

  function drawBalls(ctx) {
    for (const ball of balls) {
      const { r, g, b } = ball.color;
      const rad = ball.radius;

      // glow effect — collision ke baad zyada glow
      const flashIntensity = ball.flash > 0 ? (ball.flash / FLASH_DURATION) : 0;
      const glowAlpha = 0.3 + flashIntensity * 0.5;
      ctx.shadowColor = flashIntensity > 0
        ? 'rgba(' + ACCENT_RGB + ',' + glowAlpha + ')'
        : 'rgba(' + r + ',' + g + ',' + b + ',' + glowAlpha + ')';
      ctx.shadowBlur = rad * (1.5 + flashIntensity * 3);

      // ball circle — radial gradient fill
      ctx.beginPath();
      ctx.arc(ball.x, ball.y, rad, 0, Math.PI * 2);

      const grad = ctx.createRadialGradient(ball.x - rad * 0.3, ball.y - rad * 0.3, 0, ball.x, ball.y, rad);
      if (flashIntensity > 0) {
        // collision flash — amber glow mix karo
        const fr = Math.min(Math.round(r + (245 - r) * flashIntensity), 255);
        const fg = Math.min(Math.round(g + (158 - g) * flashIntensity), 255);
        const fb = Math.min(Math.round(b + (11 - b) * flashIntensity), 255);
        grad.addColorStop(0, 'rgba(' + Math.min(fr + 60, 255) + ',' + Math.min(fg + 60, 255) + ',' + Math.min(fb + 60, 255) + ',1)');
        grad.addColorStop(0.7, 'rgba(' + fr + ',' + fg + ',' + fb + ',0.9)');
        grad.addColorStop(1, 'rgba(' + fr + ',' + fg + ',' + fb + ',0.5)');
      } else {
        grad.addColorStop(0, 'rgba(' + Math.min(r + 60, 255) + ',' + Math.min(g + 60, 255) + ',' + Math.min(b + 60, 255) + ',1)');
        grad.addColorStop(0.7, 'rgba(' + r + ',' + g + ',' + b + ',0.9)');
        grad.addColorStop(1, 'rgba(' + r + ',' + g + ',' + b + ',0.5)');
      }
      ctx.fillStyle = grad;
      ctx.fill();

      // shadow reset karo
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;

      // velocity vector arrow — har ball pe dikhao
      const speed = Math.sqrt(ball.vx * ball.vx + ball.vy * ball.vy);
      if (speed > 5) {
        const arrowLen = Math.min(speed * VELOCITY_ARROW_SCALE, rad * 4);
        const nx = ball.vx / speed;
        const ny = ball.vy / speed;

        const tipX = ball.x + nx * (rad + arrowLen);
        const tipY = ball.y + ny * (rad + arrowLen);
        const baseX = ball.x + nx * rad;
        const baseY = ball.y + ny * rad;

        // arrow line
        ctx.beginPath();
        ctx.moveTo(baseX, baseY);
        ctx.lineTo(tipX, tipY);
        ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // arrowhead
        const headLen = Math.min(6, arrowLen * 0.4);
        const headAngle = Math.atan2(ny, nx);
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(
          tipX - headLen * Math.cos(headAngle - 0.45),
          tipY - headLen * Math.sin(headAngle - 0.45)
        );
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(
          tipX - headLen * Math.cos(headAngle + 0.45),
          tipY - headLen * Math.sin(headAngle + 0.45)
        );
        ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }

  function drawSpawnPreview(ctx) {
    // preview ball — mass hold duration se
    const holdTime = (performance.now() - spawnStartTime) / 1000;
    const previewMass = Math.min(MIN_MASS + holdTime * HOLD_MASS_RATE, MAX_MASS);
    const rad = ballRadius(previewMass);
    const color = massToColor(previewMass);

    // ball preview — translucent
    ctx.beginPath();
    ctx.arc(spawnX, spawnY, rad, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(' + color.r + ',' + color.g + ',' + color.b + ',0.3)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // mass label
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.6)';
    ctx.textAlign = 'center';
    ctx.fillText('m=' + previewMass.toFixed(1), spawnX, spawnY - rad - 8);

    // velocity arrow — drag direction mein
    if (isDragging) {
      const vx = (spawnDragX - spawnX) * 2.5;
      const vy = (spawnDragY - spawnY) * 2.5;
      const vel = Math.sqrt(vx * vx + vy * vy);
      if (vel > 10) {
        const nx = vx / vel;
        const ny = vy / vel;
        const arrowLen = Math.min(vel * 0.3, 80);

        // dashed line — spawn point se drag point tak
        ctx.beginPath();
        ctx.moveTo(spawnX, spawnY);
        ctx.lineTo(spawnDragX, spawnDragY);
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // solid arrow — velocity direction mein
        ctx.beginPath();
        ctx.moveTo(spawnX, spawnY);
        ctx.lineTo(spawnX + nx * arrowLen, spawnY + ny * arrowLen);
        ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.7)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // arrowhead
        const headLen = 8;
        const headAngle = Math.atan2(ny, nx);
        ctx.beginPath();
        ctx.moveTo(spawnX + nx * arrowLen, spawnY + ny * arrowLen);
        ctx.lineTo(
          spawnX + nx * arrowLen - headLen * Math.cos(headAngle - 0.4),
          spawnY + ny * arrowLen - headLen * Math.sin(headAngle - 0.4)
        );
        ctx.moveTo(spawnX + nx * arrowLen, spawnY + ny * arrowLen);
        ctx.lineTo(
          spawnX + nx * arrowLen - headLen * Math.cos(headAngle + 0.4),
          spawnY + ny * arrowLen - headLen * Math.sin(headAngle + 0.4)
        );
        ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.7)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // velocity magnitude label
        const cappedVel = Math.min(vel, MAX_INITIAL_VEL);
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
        ctx.textAlign = 'center';
        ctx.fillText('v=' + cappedVel.toFixed(0), spawnX + nx * arrowLen * 0.5 + ny * 12, spawnY + ny * arrowLen * 0.5 - nx * 12);
      }
    }
  }

  // --- Animation Loop ---
  function animate(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    // delta time — variable timestep with clamp
    if (lastTime === 0) lastTime = timestamp;
    let dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;

    // clamp — tab switch se bahut bada dt aa sakta hai
    dt = Math.min(dt, 0.05);

    // slow-mo factor
    const timeScale = slowMo ? 0.25 : 1.0;

    if (!isPaused) {
      // substeps for stability — fast balls miss na karein collisions
      const effectiveDt = dt * timeScale;
      const subSteps = Math.max(2, Math.ceil(effectiveDt / 0.004));
      const subDt = effectiveDt / subSteps;

      for (let s = 0; s < subSteps; s++) {
        physicsStep(subDt);
      }
    }

    draw();
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf jab screen pe dikhe tab animate karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    lastTime = 0;
    animationId = requestAnimationFrame(animate);
  }

  function stopAnimation() {
    isVisible = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          startAnimation();
        } else {
          stopAnimation();
        }
      });
    },
    { threshold: 0.1 }
  );

  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) animate(); });

  // tab switch pe pause — wapas aaye toh check karo dikhra hai ya nahi
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // --- Initial state — kuch random balls se shuru karo taaki boring na lage ---
  function spawnInitialBalls() {
    const count = 5 + Math.floor(Math.random() * 3);
    for (let i = 0; i < count; i++) {
      const mass = MIN_MASS + Math.random() * 8;
      const r = ballRadius(mass);
      const x = r + 20 + Math.random() * (canvasW - 2 * r - 40);
      const y = r + 20 + Math.random() * (canvasH - 2 * r - 40);
      const speed = 40 + Math.random() * 100;
      const angle = Math.random() * Math.PI * 2;
      balls.push(createBall(x, y, Math.cos(angle) * speed, Math.sin(angle) * speed, mass));
    }
  }

  spawnInitialBalls();
}
