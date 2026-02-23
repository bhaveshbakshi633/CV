// ============================================================
// N-Body Gravity Simulator — Newton ka gravity formula, real-time
// Click se bodies place karo, drag se velocity do, physics dekho
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, gravity simulate karo
export function initNBody() {
  const container = document.getElementById('nbodyContainer');
  if (!container) {
    console.warn('nbodyContainer nahi mila bhai, nbody demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 350;
  const G = 500; // gravitational constant — scaled for visual effect
  const SOFTENING = 5; // softening factor — singularity se bachao jab bodies close hon
  const TRAIL_LENGTH = 200;
  const MIN_MASS = 5;
  const MAX_MASS = 100;
  const HOLD_MASS_RATE = 25; // mass per second hold karne pe
  const MAX_INITIAL_VEL = 150; // slingshot ki max velocity

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let bodies = []; // [{x, y, vx, vy, mass, trail: [{x,y}], color: {r,g,b}}]
  let timeScale = 1.0;
  let showTrails = true;
  let showFieldLines = false;
  let isPaused = false;

  // placing state — jab user body place kar raha ho
  let isPlacing = false;
  let placeStartTime = 0;
  let placeX = 0, placeY = 0;
  let placeDragX = 0, placeDragY = 0;
  let isDragging = false; // slingshot drag

  // animation state
  let animationId = null;
  let isVisible = false;
  let lastTime = 0;

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:rgba(2,2,8,0.5)',
  ].join(';');
  container.appendChild(canvas);

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

  // button banane ka helper
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:6px 14px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(74,158,255,0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(74,158,255,0.25)',
      'font-family:monospace',
      'transition:all 0.2s ease',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(74,158,255,0.25)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(74,158,255,0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // Pause/Play
  const pauseBtn = createButton('Pause', () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Play' : 'Pause';
  });

  // Clear
  createButton('Clear', () => {
    bodies = [];
  });

  // Presets dropdown
  const presetSelect = document.createElement('select');
  presetSelect.style.cssText = [
    'padding:6px 10px',
    'font-size:12px',
    'border-radius:6px',
    'cursor:pointer',
    'background:rgba(74,158,255,0.1)',
    'color:#b0b0b0',
    'border:1px solid rgba(74,158,255,0.25)',
    'font-family:monospace',
  ].join(';');

  const presets = ['Presets...', 'Binary Star', 'Solar System', 'Figure-8', 'Random Cluster'];
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

  // Time scale slider
  const tsWrapper = document.createElement('div');
  tsWrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

  const tsLabel = document.createElement('span');
  tsLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace;';
  tsLabel.textContent = 'Speed:';
  tsWrapper.appendChild(tsLabel);

  const tsSlider = document.createElement('input');
  tsSlider.type = 'range';
  tsSlider.min = '0.5';
  tsSlider.max = '5';
  tsSlider.step = '0.1';
  tsSlider.value = '1';
  tsSlider.style.cssText = 'width:70px;height:4px;accent-color:rgba(74,158,255,0.8);cursor:pointer;';
  tsWrapper.appendChild(tsSlider);

  const tsValue = document.createElement('span');
  tsValue.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:monospace;min-width:28px;';
  tsValue.textContent = '1.0x';
  tsWrapper.appendChild(tsValue);

  tsSlider.addEventListener('input', () => {
    timeScale = parseFloat(tsSlider.value);
    tsValue.textContent = timeScale.toFixed(1) + 'x';
  });
  controlsDiv.appendChild(tsWrapper);

  // Toggle: trails
  const trailToggle = document.createElement('div');
  trailToggle.style.cssText = 'display:flex;align-items:center;gap:4px;';
  const trailLabel = document.createElement('span');
  trailLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace;';
  trailLabel.textContent = 'Trails:';
  trailToggle.appendChild(trailLabel);
  const trailCheck = document.createElement('input');
  trailCheck.type = 'checkbox';
  trailCheck.checked = true;
  trailCheck.style.cssText = 'accent-color:rgba(74,158,255,0.8);cursor:pointer;';
  trailCheck.addEventListener('change', () => { showTrails = trailCheck.checked; });
  trailToggle.appendChild(trailCheck);
  controlsDiv.appendChild(trailToggle);

  // Toggle: field lines
  const fieldToggle = document.createElement('div');
  fieldToggle.style.cssText = 'display:flex;align-items:center;gap:4px;';
  const fieldLabel = document.createElement('span');
  fieldLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace;';
  fieldLabel.textContent = 'Field:';
  fieldToggle.appendChild(fieldLabel);
  const fieldCheck = document.createElement('input');
  fieldCheck.type = 'checkbox';
  fieldCheck.checked = false;
  fieldCheck.style.cssText = 'accent-color:rgba(74,158,255,0.8);cursor:pointer;';
  fieldCheck.addEventListener('change', () => { showFieldLines = fieldCheck.checked; });
  fieldToggle.appendChild(fieldCheck);
  controlsDiv.appendChild(fieldToggle);

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
  // chhoti mass = blue, medium = yellow, badi = red
  function massToColor(mass) {
    const t = Math.min((mass - MIN_MASS) / (MAX_MASS - MIN_MASS), 1);
    if (t < 0.33) {
      // blue to cyan
      const s = t / 0.33;
      return {
        r: Math.round(50 + 50 * s),
        g: Math.round(100 + 155 * s),
        b: 255,
      };
    } else if (t < 0.66) {
      // cyan to yellow
      const s = (t - 0.33) / 0.33;
      return {
        r: Math.round(100 + 155 * s),
        g: Math.round(255 - 55 * s),
        b: Math.round(255 - 200 * s),
      };
    } else {
      // yellow to red
      const s = (t - 0.66) / 0.34;
      return {
        r: 255,
        g: Math.round(200 - 150 * s),
        b: Math.round(55 - 55 * s),
      };
    }
  }

  // body ka visual radius — mass^(1/3) se proportional
  function bodyRadius(mass) {
    return Math.max(3, Math.pow(mass, 1 / 3) * 2.5);
  }

  // --- Body creation ---
  function createBody(x, y, vx, vy, mass) {
    const color = massToColor(mass);
    return {
      x, y, vx, vy, mass,
      trail: [],
      color,
      // acceleration store karo — Velocity Verlet ke liye chahiye
      ax: 0, ay: 0,
    };
  }

  // --- Presets ---
  function loadPreset(name) {
    bodies = [];
    const cx = canvasW / 2;
    const cy = canvasH / 2;

    switch (name) {
      case 'Binary Star': {
        // do massive stars ek doosre ke around orbit karte hain
        const sep = 80;
        const mass = 60;
        // orbital velocity calculate karo — circular orbit ke liye
        const orbVel = Math.sqrt(G * mass / (2 * sep));
        bodies.push(createBody(cx - sep / 2, cy, 0, -orbVel, mass));
        bodies.push(createBody(cx + sep / 2, cy, 0, orbVel, mass));
        break;
      }
      case 'Solar System': {
        // ek bada sun center mein, chhote planets orbit mein
        bodies.push(createBody(cx, cy, 0, 0, 90));
        // planets — alag distances pe
        const planetData = [
          { dist: 60, mass: 5, color: 'blue' },
          { dist: 100, mass: 8, color: 'green' },
          { dist: 145, mass: 6, color: 'red' },
          { dist: 195, mass: 15, color: 'yellow' },
        ];
        planetData.forEach(p => {
          // circular orbital velocity: v = sqrt(G * M_sun / r)
          const orbVel = Math.sqrt(G * 90 / p.dist);
          // random angle se shuru karo
          const angle = Math.random() * Math.PI * 2;
          const px = cx + Math.cos(angle) * p.dist;
          const py = cy + Math.sin(angle) * p.dist;
          // velocity tangential honi chahiye — perpendicular to radius
          const vx = -Math.sin(angle) * orbVel;
          const vy = Math.cos(angle) * orbVel;
          bodies.push(createBody(px, py, vx, vy, p.mass));
        });
        break;
      }
      case 'Figure-8': {
        // famous 3-body figure-8 solution — Chenciner-Montgomery
        // normalized conditions ko scale karo canvas ke liye
        const scale = 80;
        const vScale = 45;
        const m = 30;
        // initial conditions (approximate) — ye bilkul sahi figure-8 dega? nahi,
        // but close enough for a cool demo
        bodies.push(createBody(cx - scale, cy, 0.347 * vScale, 0.533 * vScale, m));
        bodies.push(createBody(cx + scale, cy, 0.347 * vScale, 0.533 * vScale, m));
        bodies.push(createBody(cx, cy, -0.694 * vScale, -1.066 * vScale, m));
        break;
      }
      case 'Random Cluster': {
        // 8-12 random bodies ek cluster mein
        const count = 8 + Math.floor(Math.random() * 5);
        for (let i = 0; i < count; i++) {
          const angle = Math.random() * Math.PI * 2;
          const dist = 30 + Math.random() * 100;
          const px = cx + Math.cos(angle) * dist;
          const py = cy + Math.sin(angle) * dist;
          // chhoti random velocity — cluster collapse slowly
          const vel = 10 + Math.random() * 20;
          const vAngle = angle + Math.PI / 2 + (Math.random() - 0.5) * 0.5;
          const vx = Math.cos(vAngle) * vel;
          const vy = Math.sin(vAngle) * vel;
          const mass = MIN_MASS + Math.random() * 30;
          bodies.push(createBody(px, py, vx, vy, mass));
        }
        break;
      }
    }

    // saari bodies ki initial acceleration calculate karo — Verlet ke liye zaroori hai
    computeAccelerations();
  }

  // --- Physics: Gravitational acceleration calculate karo ---
  function computeAccelerations() {
    // pehle sab zero kar do
    for (let i = 0; i < bodies.length; i++) {
      bodies[i].ax = 0;
      bodies[i].ay = 0;
    }

    // har pair ke beech gravitational force — Newton ka formula
    for (let i = 0; i < bodies.length; i++) {
      for (let j = i + 1; j < bodies.length; j++) {
        const dx = bodies[j].x - bodies[i].x;
        const dy = bodies[j].y - bodies[i].y;
        const distSq = dx * dx + dy * dy + SOFTENING * SOFTENING;
        const dist = Math.sqrt(distSq);

        // F = G * m1 * m2 / r^2, acceleration = F / m
        const force = G * bodies[i].mass * bodies[j].mass / distSq;
        const fx = force * dx / dist;
        const fy = force * dy / dist;

        // Newton's third law — equal and opposite
        bodies[i].ax += fx / bodies[i].mass;
        bodies[i].ay += fy / bodies[i].mass;
        bodies[j].ax -= fx / bodies[j].mass;
        bodies[j].ay -= fy / bodies[j].mass;
      }
    }
  }

  // --- Physics: Velocity Verlet integration ---
  // Euler se zyada stable hai — energy conservation better hai
  function physicsStep(dt) {
    if (bodies.length === 0) return;

    // Step 1: position update — x += v*dt + 0.5*a*dt^2
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      b.x += b.vx * dt + 0.5 * b.ax * dt * dt;
      b.y += b.vy * dt + 0.5 * b.ay * dt * dt;
    }

    // purani acceleration save karo — velocity update ke liye chahiye
    const oldAx = bodies.map(b => b.ax);
    const oldAy = bodies.map(b => b.ay);

    // Step 2: nayi acceleration calculate karo naye positions pe
    computeAccelerations();

    // Step 3: velocity update — v += 0.5*(a_old + a_new)*dt
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      b.vx += 0.5 * (oldAx[i] + b.ax) * dt;
      b.vy += 0.5 * (oldAy[i] + b.ay) * dt;
    }

    // trail update — current position add karo
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      b.trail.push({ x: b.x, y: b.y });
      if (b.trail.length > TRAIL_LENGTH) {
        b.trail.shift();
      }
    }

    // --- Collision detection — bodies merge karo agar touch karein ---
    handleCollisions();
  }

  function handleCollisions() {
    // saare pairs check karo — agar distance < sum of radii toh merge
    for (let i = bodies.length - 1; i >= 0; i--) {
      for (let j = i - 1; j >= 0; j--) {
        const a = bodies[i];
        const b = bodies[j];
        if (!a || !b) continue;

        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const minDist = bodyRadius(a.mass) + bodyRadius(b.mass);

        if (dist < minDist) {
          // merge — momentum conserve karo
          const totalMass = a.mass + b.mass;
          const newVx = (a.mass * a.vx + b.mass * b.vx) / totalMass;
          const newVy = (a.mass * a.vy + b.mass * b.vy) / totalMass;
          // center of mass position
          const newX = (a.mass * a.x + b.mass * b.x) / totalMass;
          const newY = (a.mass * a.y + b.mass * b.y) / totalMass;

          // badi body mein merge karo
          const bigger = a.mass >= b.mass ? j : i;
          const smaller = a.mass >= b.mass ? i : j;

          bodies[bigger].x = newX;
          bodies[bigger].y = newY;
          bodies[bigger].vx = newVx;
          bodies[bigger].vy = newVy;
          bodies[bigger].mass = Math.min(totalMass, MAX_MASS); // cap at max
          bodies[bigger].color = massToColor(bodies[bigger].mass);

          // chhoti body hata do
          bodies.splice(smaller, 1);
          break; // inner loop restart — indices badal gaye
        }
      }
    }
  }

  // --- Mouse/touch events — body place karo ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  canvas.addEventListener('mousedown', (e) => {
    const pos = getCanvasPos(e);
    isPlacing = true;
    isDragging = false;
    placeStartTime = performance.now();
    placeX = pos.x;
    placeY = pos.y;
    placeDragX = pos.x;
    placeDragY = pos.y;
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isPlacing) return;
    const pos = getCanvasPos(e);
    placeDragX = pos.x;
    placeDragY = pos.y;
    // agar kuch distance drag kiya toh slingshot mode mein aa jao
    const dragDist = Math.sqrt(Math.pow(placeDragX - placeX, 2) + Math.pow(placeDragY - placeY, 2));
    if (dragDist > 5) isDragging = true;
  });

  canvas.addEventListener('mouseup', () => {
    if (!isPlacing) return;
    finishPlacing();
  });

  // touch support
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    isPlacing = true;
    isDragging = false;
    placeStartTime = performance.now();
    placeX = pos.x;
    placeY = pos.y;
    placeDragX = pos.x;
    placeDragY = pos.y;
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isPlacing) return;
    const pos = getCanvasPos(e);
    placeDragX = pos.x;
    placeDragY = pos.y;
    const dragDist = Math.sqrt(Math.pow(placeDragX - placeX, 2) + Math.pow(placeDragY - placeY, 2));
    if (dragDist > 5) isDragging = true;
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (!isPlacing) return;
    finishPlacing();
  }, { passive: false });

  function finishPlacing() {
    isPlacing = false;

    // mass — hold duration se proportional
    const holdTime = (performance.now() - placeStartTime) / 1000; // seconds
    const mass = Math.min(MIN_MASS + holdTime * HOLD_MASS_RATE, MAX_MASS);

    // velocity — slingshot drag se (drag direction ke OPPOSITE mein)
    let vx = 0, vy = 0;
    if (isDragging) {
      vx = -(placeDragX - placeX) * 1.5; // opposite direction
      vy = -(placeDragY - placeY) * 1.5;
      // cap velocity
      const vel = Math.sqrt(vx * vx + vy * vy);
      if (vel > MAX_INITIAL_VEL) {
        vx = (vx / vel) * MAX_INITIAL_VEL;
        vy = (vy / vel) * MAX_INITIAL_VEL;
      }
    }

    bodies.push(createBody(placeX, placeY, vx, vy, mass));

    // nayi body ki acceleration calculate karo
    computeAccelerations();
  }

  // --- Drawing ---
  function draw() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvasW, canvasH);

    // subtle star background — chhote white dots
    drawStars(ctx);

    // gravitational field lines — agar toggle on hai
    if (showFieldLines && bodies.length > 0) {
      drawFieldLines(ctx);
    }

    // trails draw karo
    if (showTrails) {
      drawTrails(ctx);
    }

    // bodies draw karo
    drawBodies(ctx);

    // placing preview — jab user body place kar raha hai
    if (isPlacing) {
      drawPlacingPreview(ctx);
    }

    // body count dikhao
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'right';
    ctx.fillText('bodies: ' + bodies.length, canvasW - 10, 16);

    // hint jab koi body nahi hai
    if (bodies.length === 0 && !isPlacing) {
      ctx.font = '13px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('click to place bodies \u2022 hold for more mass \u2022 drag for velocity', canvasW / 2, canvasH / 2);
    }
  }

  // --- Background stars — static random dots ---
  // performance ke liye ek baar generate karo, baar baar mat banao
  let starPositions = null;
  function drawStars(ctx) {
    if (!starPositions) {
      starPositions = [];
      for (let i = 0; i < 60; i++) {
        starPositions.push({
          x: Math.random() * canvasW,
          y: Math.random() * canvasH,
          r: 0.3 + Math.random() * 0.8,
          a: 0.15 + Math.random() * 0.25,
        });
      }
    }
    for (const s of starPositions) {
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(200,210,255,' + s.a + ')';
      ctx.fill();
    }
  }

  function drawTrails(ctx) {
    for (const body of bodies) {
      if (body.trail.length < 2) continue;
      const { r, g, b } = body.color;

      ctx.beginPath();
      ctx.moveTo(body.trail[0].x, body.trail[0].y);
      for (let i = 1; i < body.trail.length; i++) {
        ctx.lineTo(body.trail[i].x, body.trail[i].y);
      }

      // gradient trail — purana faded, naya bright
      // simple approach — puri line ek alpha se
      const alpha = 0.3;
      ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
      ctx.lineWidth = 1;
      ctx.stroke();

      // trail ke individual segments bhi draw karo fading ke saath
      for (let i = 1; i < body.trail.length; i++) {
        const segAlpha = (i / body.trail.length) * 0.35;
        ctx.beginPath();
        ctx.moveTo(body.trail[i - 1].x, body.trail[i - 1].y);
        ctx.lineTo(body.trail[i].x, body.trail[i].y);
        ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + segAlpha + ')';
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }
  }

  function drawBodies(ctx) {
    for (const body of bodies) {
      const rad = bodyRadius(body.mass);
      const { r, g, b } = body.color;

      // glow effect
      ctx.shadowColor = 'rgba(' + r + ',' + g + ',' + b + ',0.5)';
      ctx.shadowBlur = rad * 2;

      // body circle
      ctx.beginPath();
      ctx.arc(body.x, body.y, rad, 0, Math.PI * 2);

      // radial gradient fill — center bright, edge fade
      const grad = ctx.createRadialGradient(body.x, body.y, 0, body.x, body.y, rad);
      grad.addColorStop(0, 'rgba(' + Math.min(r + 80, 255) + ',' + Math.min(g + 80, 255) + ',' + Math.min(b + 80, 255) + ',1)');
      grad.addColorStop(0.6, 'rgba(' + r + ',' + g + ',' + b + ',0.9)');
      grad.addColorStop(1, 'rgba(' + r + ',' + g + ',' + b + ',0.4)');
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
    }
  }

  function drawPlacingPreview(ctx) {
    // preview body — mass hold duration se
    const holdTime = (performance.now() - placeStartTime) / 1000;
    const previewMass = Math.min(MIN_MASS + holdTime * HOLD_MASS_RATE, MAX_MASS);
    const rad = bodyRadius(previewMass);
    const color = massToColor(previewMass);

    // body preview — translucent
    ctx.beginPath();
    ctx.arc(placeX, placeY, rad, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(' + color.r + ',' + color.g + ',' + color.b + ',0.4)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(' + color.r + ',' + color.g + ',' + color.b + ',0.6)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // mass text
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.6)';
    ctx.textAlign = 'center';
    ctx.fillText('m=' + previewMass.toFixed(0), placeX, placeY - rad - 6);

    // slingshot arrow — drag direction dikhao
    if (isDragging) {
      // arrow from body to drag point
      ctx.beginPath();
      ctx.moveTo(placeX, placeY);
      ctx.lineTo(placeDragX, placeDragY);
      ctx.strokeStyle = 'rgba(255,100,100,0.5)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // velocity direction — opposite of drag
      const vx = -(placeDragX - placeX);
      const vy = -(placeDragY - placeY);
      const vel = Math.sqrt(vx * vx + vy * vy);
      if (vel > 5) {
        const normX = vx / vel;
        const normY = vy / vel;
        const arrowLen = Math.min(vel * 0.5, 60);

        ctx.beginPath();
        ctx.moveTo(placeX, placeY);
        ctx.lineTo(placeX + normX * arrowLen, placeY + normY * arrowLen);
        ctx.strokeStyle = 'rgba(100,255,100,0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // arrowhead
        const headLen = 8;
        const headAngle = Math.atan2(normY, normX);
        ctx.beginPath();
        ctx.moveTo(placeX + normX * arrowLen, placeY + normY * arrowLen);
        ctx.lineTo(
          placeX + normX * arrowLen - headLen * Math.cos(headAngle - 0.4),
          placeY + normY * arrowLen - headLen * Math.sin(headAngle - 0.4)
        );
        ctx.moveTo(placeX + normX * arrowLen, placeY + normY * arrowLen);
        ctx.lineTo(
          placeX + normX * arrowLen - headLen * Math.cos(headAngle + 0.4),
          placeY + normY * arrowLen - headLen * Math.sin(headAngle + 0.4)
        );
        ctx.strokeStyle = 'rgba(100,255,100,0.5)';
        ctx.stroke();
      }
    }
  }

  function drawFieldLines(ctx) {
    // gravitational field — grid pe arrows draw karo showing force direction
    const gridSpacing = 40;
    const arrowLen = 12;

    for (let gx = gridSpacing / 2; gx < canvasW; gx += gridSpacing) {
      for (let gy = gridSpacing / 2; gy < canvasH; gy += gridSpacing) {
        // net gravitational acceleration at this point
        let ax = 0, ay = 0;
        for (const body of bodies) {
          const dx = body.x - gx;
          const dy = body.y - gy;
          const distSq = dx * dx + dy * dy + SOFTENING * SOFTENING;
          const dist = Math.sqrt(distSq);
          const accel = G * body.mass / distSq;
          ax += accel * dx / dist;
          ay += accel * dy / dist;
        }

        const mag = Math.sqrt(ax * ax + ay * ay);
        if (mag < 0.5) continue; // bahut weak field — skip karo

        // normalize and scale
        const nx = ax / mag;
        const ny = ay / mag;
        const len = Math.min(arrowLen, arrowLen * Math.log(1 + mag * 0.01));
        const alpha = Math.min(0.25, 0.05 + mag * 0.0005);

        ctx.beginPath();
        ctx.moveTo(gx, gy);
        ctx.lineTo(gx + nx * len, gy + ny * len);
        ctx.strokeStyle = 'rgba(74,158,255,' + alpha + ')';
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }
  }

  // --- Animation loop ---
  function animate(timestamp) {
    if (!isVisible) return;

    // delta time calculate karo — variable timestep
    if (lastTime === 0) lastTime = timestamp;
    let dt = (timestamp - lastTime) / 1000; // seconds
    lastTime = timestamp;

    // dt ko clamp karo — agar tab switch karke aaye toh bahut bada dt aa sakta hai
    dt = Math.min(dt, 0.05);

    // physics step — agar paused nahi hai
    if (!isPaused) {
      // substeps for stability — agar timeScale zyada hai toh zyada substeps
      const subSteps = Math.max(1, Math.ceil(timeScale * 2));
      const subDt = (dt * timeScale) / subSteps;

      for (let s = 0; s < subSteps; s++) {
        physicsStep(subDt);
      }

      // out of bounds bodies hata do — bahut door nikal gaye toh koi point nahi rakhne ka
      bodies = bodies.filter(b =>
        b.x > -500 && b.x < canvasW + 500 &&
        b.y > -500 && b.y < canvasH + 500
      );
    }

    draw();
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf jab dikhe tab animate karo ---
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

  // tab switch pe pause
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });
}
