// ============================================================
// Gravity Sandbox — N-body gravitational simulation
// Masses place karo, orbits dekho, stars banao, chaos enjoy karo
// Velocity Verlet integration with softening — energy conserving
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, gravity simulate karo
export function initGravity() {
  const container = document.getElementById('gravityContainer');
  if (!container) {
    console.warn('gravityContainer nahi mila bhai, gravity sandbox skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const G = 400;             // gravitational constant — visual ke liye scaled
  const SOFTENING = 4;       // singularity se bachne ke liye epsilon
  const TRAIL_MAX = 200;     // trail mein kitne points rakhne hain
  const STAR_MASS_THRESHOLD = 50; // isse zyada mass = star (golden glow)
  const SUBSTEPS = 4;        // physics substeps per frame — accuracy ke liye

  // --- Nice color palette ---
  // stars golden honge, planets ko ye colors milenge
  const PLANET_COLORS = [
    '#3b82f6', // blue
    '#10b981', // green/emerald
    '#ef4444', // red
    '#8b5cf6', // purple
    '#06b6d4', // cyan
    '#f97316', // orange
    '#ec4899', // pink
    '#14b8a6', // teal
  ];
  const STAR_COLOR = '#fbbf24'; // golden amber — taare ka rang

  // --- State ---
  let canvasW = 0, canvasH = 0, dpr = 1;
  let bodies = []; // {x, y, vx, vy, mass, radius, color, trail[], ax, ay}
  let timeScale = 1.0;
  let showTrails = true;
  let colorIndex = 0; // next planet ko konsa color dena hai

  // interaction state
  let isPlacing = false;       // naya body place ho raha hai (click + drag)
  let placeX = 0, placeY = 0;  // jahan click kiya — body ki position
  let dragX = 0, dragY = 0;    // jahan tak drag kiya — velocity decide karega
  let placeShift = false;       // shift hold = star mode (10x mass)

  // dragging existing body
  let isDraggingBody = false;
  let dragBodyIndex = -1;
  let physicsPaused = false;    // existing body drag ke time pause

  // animation state
  let animationId = null;
  let isVisible = false;
  let lastTime = 0;

  // background stars — ek baar generate, baar baar reuse
  let bgStars = null;

  // --- DOM structure banao — pehle purane children preserve karo ---
  // container ke andar existing children ko hata do (reinit ke liye)
  const existingChildren = [];
  while (container.firstChild) {
    existingChildren.push(container.removeChild(container.firstChild));
  }
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(245,158,11,0.15)',
    'border-radius:6px',
    'cursor:crosshair',
    'background:#000000',
  ].join(';');
  container.appendChild(canvas);

  // --- Controls section ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // button banane ka helper — dark theme, amber accent
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:11px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(245,158,11,0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(245,158,11,0.25)',
      "font-family:'JetBrains Mono',monospace",
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

  // toggle button helper — on/off state ke saath
  function createToggle(label, initial, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#6b6b6b;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = initial;
    cb.style.cssText = 'accent-color:#f59e0b;cursor:pointer;';
    cb.addEventListener('change', () => onChange(cb.checked));
    wrapper.appendChild(cb);

    controlsDiv.appendChild(wrapper);
    return cb;
  }

  // slider helper
  function createSlider(label, min, max, step, value, unit, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#6b6b6b;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.style.cssText = 'width:70px;height:4px;accent-color:#f59e0b;cursor:pointer;';
    wrapper.appendChild(slider);

    const val = document.createElement('span');
    val.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:32px;";
    val.textContent = Number(value).toFixed(1) + (unit || '');
    wrapper.appendChild(val);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      val.textContent = v.toFixed(1) + (unit || '');
      onChange(v);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, val };
  }

  // --- Controls banao ---

  // trails toggle
  createToggle('Trails', true, (v) => { showTrails = v; });

  // clear all
  createButton('Clear All', () => {
    bodies = [];
    colorIndex = 0;
  });

  // speed slider
  createSlider('Speed', 0.5, 3.0, 0.1, 1.0, 'x', (v) => { timeScale = v; });

  // --- Presets ---
  const presetNames = ['Binary Star', 'Solar System', 'Figure-8', 'Random'];
  presetNames.forEach(name => {
    createButton(name, () => loadPreset(name));
  });

  // --- Canvas sizing — DPR aware ---
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

    // background stars regenerate karo naye size ke liye
    bgStars = null;
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // ============================================================
  // BODY CREATION — mass se radius aur color decide hota hai
  // ============================================================

  function nextPlanetColor() {
    const c = PLANET_COLORS[colorIndex % PLANET_COLORS.length];
    colorIndex++;
    return c;
  }

  function createBody(x, y, vx, vy, mass, color) {
    // radius mass^(1/3) se proportional — physics mein density constant assume karo
    const radius = Math.max(3, Math.pow(mass, 1 / 3) * 2.2);
    // star ya planet — mass se decide hoga
    const isStar = mass >= STAR_MASS_THRESHOLD;
    const bodyColor = color || (isStar ? STAR_COLOR : nextPlanetColor());

    return {
      x, y, vx, vy, mass, radius,
      color: bodyColor,
      trail: [],
      ax: 0, ay: 0, // acceleration — Velocity Verlet ke liye
    };
  }

  // ============================================================
  // PRESETS — interesting initial configurations
  // ============================================================

  function loadPreset(name) {
    bodies = [];
    colorIndex = 0;
    const cx = canvasW / 2;
    const cy = canvasH / 2;

    switch (name) {
      case 'Binary Star': {
        // do massive stars ek doosre ke around orbit — center of mass pe stable
        const sep = 90;
        const mass = 65;
        // circular orbit ke liye velocity: v = sqrt(G * m_other / (2 * sep))
        // lekin dono equal mass hain toh: v = sqrt(G * m / (4 * sep/2))
        const orbVel = Math.sqrt(G * mass / (2 * sep));
        bodies.push(createBody(cx - sep / 2, cy, 0, -orbVel, mass, STAR_COLOR));
        bodies.push(createBody(cx + sep / 2, cy, 0, orbVel, mass, '#ff9500'));
        break;
      }
      case 'Solar System': {
        // ek bada sun center mein + 3 planets orbit mein
        const sunMass = 80;
        bodies.push(createBody(cx, cy, 0, 0, sunMass, STAR_COLOR));

        const planets = [
          { dist: 65, mass: 4 },
          { dist: 110, mass: 7 },
          { dist: 165, mass: 5 },
        ];
        planets.forEach(p => {
          // circular orbit: v = sqrt(G * M_sun / r)
          const orbVel = Math.sqrt(G * sunMass / p.dist);
          const angle = Math.random() * Math.PI * 2;
          const px = cx + Math.cos(angle) * p.dist;
          const py = cy + Math.sin(angle) * p.dist;
          // tangential velocity — perpendicular to radius vector
          const vx = -Math.sin(angle) * orbVel;
          const vy = Math.cos(angle) * orbVel;
          bodies.push(createBody(px, py, vx, vy, p.mass, null));
        });
        break;
      }
      case 'Figure-8': {
        // Chenciner-Montgomery 3-body periodic orbit
        // ye famous choreography solution hai — teeno bodies ek figure-8 trace karti hain
        const scale = 75;
        const vScale = 40;
        const m = 35;
        // approximate initial conditions — exact nahi hai but demo ke liye kaafi
        bodies.push(createBody(cx - scale, cy, 0.347 * vScale, 0.533 * vScale, m, '#3b82f6'));
        bodies.push(createBody(cx + scale, cy, 0.347 * vScale, 0.533 * vScale, m, '#ef4444'));
        bodies.push(createBody(cx, cy, -0.694 * vScale, -1.066 * vScale, m, '#10b981'));
        break;
      }
      case 'Random': {
        // 5-8 random bodies — kuch stars kuch planets
        const count = 5 + Math.floor(Math.random() * 4);
        for (let i = 0; i < count; i++) {
          const angle = Math.random() * Math.PI * 2;
          const dist = 40 + Math.random() * 120;
          const px = cx + Math.cos(angle) * dist;
          const py = cy + Math.sin(angle) * dist;
          // chhoti random velocity — slight orbital tendency
          const vel = 10 + Math.random() * 30;
          const vAngle = angle + Math.PI / 2 + (Math.random() - 0.5) * 0.6;
          const vx = Math.cos(vAngle) * vel;
          const vy = Math.sin(vAngle) * vel;
          const mass = 3 + Math.random() * 40;
          bodies.push(createBody(px, py, vx, vy, mass, null));
        }
        break;
      }
    }

    // initial acceleration compute karo — Velocity Verlet ke liye zaroori
    computeAccelerations();
  }

  // ============================================================
  // PHYSICS: N-body gravitational simulation
  // Velocity Verlet integration — symplectic, energy conserving
  // ============================================================

  // saari bodies ke beech gravitational acceleration calculate karo
  function computeAccelerations() {
    // pehle sab zero
    for (let i = 0; i < bodies.length; i++) {
      bodies[i].ax = 0;
      bodies[i].ay = 0;
    }

    // har pair (i, j) ke beech force — O(n^2) but n chhota hai toh theek
    for (let i = 0; i < bodies.length; i++) {
      for (let j = i + 1; j < bodies.length; j++) {
        const dx = bodies[j].x - bodies[i].x;
        const dy = bodies[j].y - bodies[i].y;
        // softened distance — singularity prevent karta hai
        const distSq = dx * dx + dy * dy + SOFTENING * SOFTENING;
        const dist = Math.sqrt(distSq);

        // F = G * m1 * m2 / (r^2 + eps^2)
        const force = G * bodies[i].mass * bodies[j].mass / distSq;
        const fx = force * dx / dist;
        const fy = force * dy / dist;

        // Newton's third law — equal and opposite reaction
        bodies[i].ax += fx / bodies[i].mass;
        bodies[i].ay += fy / bodies[i].mass;
        bodies[j].ax -= fx / bodies[j].mass;
        bodies[j].ay -= fy / bodies[j].mass;
      }
    }
  }

  // Velocity Verlet step — Euler se kahin better energy conservation
  function physicsStep(dt) {
    if (bodies.length < 1) return;

    // Step 1: position update — x += v*dt + 0.5*a*dt^2
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      b.x += b.vx * dt + 0.5 * b.ax * dt * dt;
      b.y += b.vy * dt + 0.5 * b.ay * dt * dt;
    }

    // purani acceleration save karo
    const oldAx = new Array(bodies.length);
    const oldAy = new Array(bodies.length);
    for (let i = 0; i < bodies.length; i++) {
      oldAx[i] = bodies[i].ax;
      oldAy[i] = bodies[i].ay;
    }

    // Step 2: nayi acceleration calculate karo naye positions se
    computeAccelerations();

    // Step 3: velocity update — v += 0.5*(a_old + a_new)*dt
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      b.vx += 0.5 * (oldAx[i] + b.ax) * dt;
      b.vy += 0.5 * (oldAy[i] + b.ay) * dt;
    }

    // trails update karo
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      b.trail.push({ x: b.x, y: b.y });
      if (b.trail.length > TRAIL_MAX) {
        b.trail.shift();
      }
    }

    // collision detection — merge jab distance < r1 + r2
    handleCollisions();
  }

  // collision handling — momentum conserve karo, mass add karo
  function handleCollisions() {
    for (let i = bodies.length - 1; i >= 0; i--) {
      for (let j = i - 1; j >= 0; j--) {
        const a = bodies[i];
        const b = bodies[j];
        if (!a || !b) continue;

        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const minDist = a.radius + b.radius;

        if (dist < minDist) {
          // merge — momentum conservation
          const totalMass = a.mass + b.mass;
          const newVx = (a.mass * a.vx + b.mass * b.vx) / totalMass;
          const newVy = (a.mass * a.vy + b.mass * b.vy) / totalMass;
          // center of mass position
          const newX = (a.mass * a.x + b.mass * b.x) / totalMass;
          const newY = (a.mass * a.y + b.mass * b.y) / totalMass;

          // badi body ko rakhte hain, chhoti ko hata do
          const keepIdx = a.mass >= b.mass ? i : j;
          const removeIdx = a.mass >= b.mass ? j : i;

          const kept = bodies[keepIdx];
          kept.x = newX;
          kept.y = newY;
          kept.vx = newVx;
          kept.vy = newVy;
          kept.mass = totalMass;
          kept.radius = Math.max(3, Math.pow(totalMass, 1 / 3) * 2.2);
          // star ban gaya? color update karo
          if (totalMass >= STAR_MASS_THRESHOLD && kept.color !== STAR_COLOR) {
            kept.color = STAR_COLOR;
          }

          // dragging body hata rahe hain toh drag cancel karo
          if (isDraggingBody && dragBodyIndex === removeIdx) {
            isDraggingBody = false;
            physicsPaused = false;
          }
          // drag index adjust karo agar zaroori ho
          if (isDraggingBody && dragBodyIndex > removeIdx) {
            dragBodyIndex--;
          }

          bodies.splice(removeIdx, 1);
          break; // inner loop se bahar — indices badal gaye
        }
      }
    }
  }

  // ============================================================
  // INTERACTION: Click-drag for placement, drag existing bodies
  // ============================================================

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  // existing body pe hit test — pakadne ke liye
  function hitTestBody(mx, my) {
    for (let i = bodies.length - 1; i >= 0; i--) {
      const b = bodies[i];
      const dx = mx - b.x;
      const dy = my - b.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      // thoda extra margin — easy grabbing ke liye
      if (dist < b.radius + 8) return i;
    }
    return -1;
  }

  function onPointerDown(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const shiftKey = e.shiftKey || false;

    // pehle check karo — kya existing body pe click kiya?
    const hitIdx = hitTestBody(pos.x, pos.y);
    if (hitIdx >= 0) {
      // existing body drag mode
      isDraggingBody = true;
      dragBodyIndex = hitIdx;
      physicsPaused = true;
      canvas.style.cursor = 'grabbing';
      return;
    }

    // naya body place karo — click position = body position
    isPlacing = true;
    placeX = pos.x;
    placeY = pos.y;
    dragX = pos.x;
    dragY = pos.y;
    placeShift = shiftKey;
  }

  function onPointerMove(e) {
    const pos = getCanvasPos(e);

    if (isDraggingBody) {
      // existing body ko move karo
      e.preventDefault();
      if (dragBodyIndex >= 0 && dragBodyIndex < bodies.length) {
        bodies[dragBodyIndex].x = pos.x;
        bodies[dragBodyIndex].y = pos.y;
        // velocity zero karo jab drag ho raha ho
        bodies[dragBodyIndex].vx = 0;
        bodies[dragBodyIndex].vy = 0;
        // trail clear karo
        bodies[dragBodyIndex].trail = [];
      }
      return;
    }

    if (isPlacing) {
      // drag position update — velocity arrow ke liye
      e.preventDefault();
      dragX = pos.x;
      dragY = pos.y;
      // shift key dynamically check karo
      if (e.shiftKey) placeShift = true;
      return;
    }

    // hover cursor — grab dikhao agar body pe ho
    const hitIdx = hitTestBody(pos.x, pos.y);
    canvas.style.cursor = hitIdx >= 0 ? 'grab' : 'crosshair';
  }

  function onPointerUp(e) {
    if (isDraggingBody) {
      // body chod do — physics resume
      isDraggingBody = false;
      physicsPaused = false;
      dragBodyIndex = -1;
      canvas.style.cursor = 'crosshair';
      // accelerations recalculate karo nayi position ke liye
      computeAccelerations();
      return;
    }

    if (isPlacing) {
      isPlacing = false;

      // mass decide karo — shift = 10x (star), normal = planet
      const baseMass = placeShift ? 70 : 7;

      // velocity = drag direction (click se drag end tak)
      // zyada drag = zyada velocity
      const velScale = 1.2;
      const vx = (dragX - placeX) * velScale;
      const vy = (dragY - placeY) * velScale;

      bodies.push(createBody(placeX, placeY, vx, vy, baseMass, null));

      // accelerations recalculate karo
      computeAccelerations();
    }
  }

  // mouse events
  canvas.addEventListener('mousedown', onPointerDown);
  canvas.addEventListener('mousemove', onPointerMove);
  canvas.addEventListener('mouseup', onPointerUp);
  canvas.addEventListener('mouseleave', () => {
    if (isDraggingBody) {
      isDraggingBody = false;
      physicsPaused = false;
      dragBodyIndex = -1;
      computeAccelerations();
    }
    if (isPlacing) {
      isPlacing = false;
    }
    canvas.style.cursor = 'crosshair';
  });

  // touch events
  canvas.addEventListener('touchstart', (e) => { onPointerDown(e); }, { passive: false });
  canvas.addEventListener('touchmove', (e) => { onPointerMove(e); }, { passive: false });
  canvas.addEventListener('touchend', (e) => { e.preventDefault(); onPointerUp(e); }, { passive: false });
  canvas.addEventListener('touchcancel', () => {
    isDraggingBody = false;
    physicsPaused = false;
    isPlacing = false;
    canvas.style.cursor = 'crosshair';
  });

  // ============================================================
  // DRAWING — bodies, trails, glow, velocity arrow, sab kuch yahan
  // ============================================================

  function draw() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvasW, canvasH);

    // background — pure black
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // subtle background stars — aesthetic ke liye
    drawBgStars(ctx);

    // trails draw karo — bodies ke peeche fading lines
    if (showTrails) {
      drawTrails(ctx);
    }

    // bodies draw karo — gradient circles with glow
    drawBodies(ctx);

    // velocity arrow — jab naya body place ho raha ho
    if (isPlacing) {
      drawVelocityArrow(ctx);
    }

    // body count — bottom right mein chhota text
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'right';
    ctx.fillText('bodies: ' + bodies.length, canvasW - 10, canvasH - 10);

    // hint jab koi body nahi hai
    if (bodies.length === 0 && !isPlacing) {
      ctx.font = "12px 'JetBrains Mono', monospace";
      ctx.fillStyle = 'rgba(176,176,176,0.25)';
      ctx.textAlign = 'center';
      ctx.fillText('click + drag to place \u2022 shift for star \u2022 drag existing to reposition', canvasW / 2, canvasH / 2);
    }
  }

  // --- Background stars — chhote static dots ---
  function drawBgStars(ctx) {
    if (!bgStars) {
      bgStars = [];
      const count = Math.floor(canvasW * canvasH / 3000);
      for (let i = 0; i < count; i++) {
        bgStars.push({
          x: Math.random() * canvasW,
          y: Math.random() * canvasH,
          r: 0.3 + Math.random() * 0.7,
          a: 0.1 + Math.random() * 0.2,
        });
      }
    }
    for (const s of bgStars) {
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(200,210,255,' + s.a + ')';
      ctx.fill();
    }
  }

  // --- Trails — fading lines behind each body ---
  function drawTrails(ctx) {
    for (const body of bodies) {
      if (body.trail.length < 2) continue;

      // color parse kar lo hex se rgb — trail ke liye alpha chahiye
      const rgb = hexToRgb(body.color);

      for (let i = 1; i < body.trail.length; i++) {
        // alpha fade — purane points transparent, naye bright
        const t = i / body.trail.length;
        const alpha = t * t * 0.45; // quadratic fade

        ctx.beginPath();
        ctx.moveTo(body.trail[i - 1].x, body.trail[i - 1].y);
        ctx.lineTo(body.trail[i].x, body.trail[i].y);
        ctx.strokeStyle = 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',' + alpha + ')';
        ctx.lineWidth = 1 + t * 0.8;
        ctx.stroke();
      }
    }
  }

  // --- Bodies draw — gradient circles with glow ---
  function drawBodies(ctx) {
    for (const body of bodies) {
      const { x, y, radius, mass, color } = body;
      const rgb = hexToRgb(color);
      const isStar = mass >= STAR_MASS_THRESHOLD;

      ctx.save();

      // glow effect — stars ko zyada glow, planets ko kam
      if (isStar) {
        ctx.shadowColor = color;
        ctx.shadowBlur = radius * 4;
      } else {
        ctx.shadowColor = color;
        ctx.shadowBlur = radius * 1.5;
      }

      // radial gradient — bright center, dark edge
      const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
      // center mein bright (white ke paas)
      const brightR = Math.min(rgb.r + 100, 255);
      const brightG = Math.min(rgb.g + 100, 255);
      const brightB = Math.min(rgb.b + 100, 255);
      grad.addColorStop(0, 'rgba(' + brightR + ',' + brightG + ',' + brightB + ',1)');
      grad.addColorStop(0.5, 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.9)');
      grad.addColorStop(1, 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.3)');

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // star ke liye extra outer glow ring
      if (isStar) {
        ctx.beginPath();
        ctx.arc(x, y, radius * 1.8, 0, Math.PI * 2);
        const outerGrad = ctx.createRadialGradient(x, y, radius * 0.8, x, y, radius * 1.8);
        outerGrad.addColorStop(0, 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.15)');
        outerGrad.addColorStop(1, 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0)');
        ctx.fillStyle = outerGrad;
        ctx.fill();
      }

      ctx.restore();
    }
  }

  // --- Velocity arrow — white arrow jab placing ---
  function drawVelocityArrow(ctx) {
    const dx = dragX - placeX;
    const dy = dragY - placeY;
    const len = Math.sqrt(dx * dx + dy * dy);

    // preview body dikhao — translucent circle
    const previewMass = placeShift ? 70 : 7;
    const previewRadius = Math.max(3, Math.pow(previewMass, 1 / 3) * 2.2);
    const previewColor = placeShift ? STAR_COLOR : PLANET_COLORS[colorIndex % PLANET_COLORS.length];
    const rgb = hexToRgb(previewColor);

    ctx.beginPath();
    ctx.arc(placeX, placeY, previewRadius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.4)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.6)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // shift indicator
    if (placeShift) {
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillStyle = 'rgba(251,191,36,0.7)';
      ctx.textAlign = 'center';
      ctx.fillText('STAR', placeX, placeY - previewRadius - 6);
    }

    // velocity arrow — sirf agar kuch drag kiya ho
    if (len > 5) {
      const nx = dx / len;
      const ny = dy / len;
      const arrowLen = Math.min(len, 100);

      // main line — white
      ctx.beginPath();
      ctx.moveTo(placeX, placeY);
      ctx.lineTo(placeX + nx * arrowLen, placeY + ny * arrowLen);
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // arrowhead
      const headLen = 8;
      const headAngle = Math.atan2(ny, nx);
      ctx.beginPath();
      ctx.moveTo(placeX + nx * arrowLen, placeY + ny * arrowLen);
      ctx.lineTo(
        placeX + nx * arrowLen - headLen * Math.cos(headAngle - 0.4),
        placeY + ny * arrowLen - headLen * Math.sin(headAngle - 0.4)
      );
      ctx.moveTo(placeX + nx * arrowLen, placeY + ny * arrowLen);
      ctx.lineTo(
        placeX + nx * arrowLen - headLen * Math.cos(headAngle + 0.4),
        placeY + ny * arrowLen - headLen * Math.sin(headAngle + 0.4)
      );
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // velocity magnitude text
      const velMag = (len * 1.2).toFixed(0);
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillStyle = 'rgba(255,255,255,0.45)';
      ctx.textAlign = 'center';
      ctx.fillText('v=' + velMag, placeX + nx * arrowLen * 0.5 + ny * 12, placeY + ny * arrowLen * 0.5 - nx * 12);
    }
  }

  // --- Hex color to RGB helper ---
  function hexToRgb(hex) {
    // handle named colors ya invalid — fallback to white
    if (!hex || hex.charAt(0) !== '#') return { r: 200, g: 200, b: 200 };
    const bigint = parseInt(hex.slice(1), 16);
    return {
      r: (bigint >> 16) & 255,
      g: (bigint >> 8) & 255,
      b: bigint & 255,
    };
  }

  // ============================================================
  // ANIMATION LOOP — requestAnimationFrame ke saath
  // ============================================================

  function animate(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    // delta time calculate karo
    if (lastTime === 0) lastTime = timestamp;
    let dt = (timestamp - lastTime) / 1000; // seconds mein
    lastTime = timestamp;

    // dt clamp — tab switch se aaye toh bahut bada dt aa sakta hai
    dt = Math.min(dt, 0.05);

    // physics step — sirf jab paused nahi hai
    if (!physicsPaused) {
      const scaledDt = dt * timeScale;
      const subDt = scaledDt / SUBSTEPS;

      for (let s = 0; s < SUBSTEPS; s++) {
        physicsStep(subDt);
      }

      // out of bounds bodies hata do — bahut door nikal gaye
      bodies = bodies.filter(b =>
        b.x > -500 && b.x < canvasW + 500 &&
        b.y > -500 && b.y < canvasH + 500
      );
    }

    draw();
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — off-screen hone pe pause karo, battery bachao ---
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

  // tab switch pe bhi pause — no wasted CPU
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // --- Binary Star preset se shuru karo by default ---
  loadPreset('Binary Star');
}
