// ============================================================
// Interactive Cloth Simulation — Verlet integration + distance constraints
// Grab karo, phenko, phaado, wind lagao — full satisfying physics
// Tearable cloth jo stress-based colors dikhata hai
// ============================================================

// yahi function bahar export hoga — container dhundho, canvas banao, kapda simulate karo
export function initCloth() {
  const container = document.getElementById('clothContainer');
  if (!container) {
    console.warn('clothContainer nahi mila bhai, cloth sim skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  const ACCENT_RGB = '245,158,11';

  // cloth grid dimensions
  const COLS = 30;
  const ROWS = 20;
  const SPACING = 12; // rest length between adjacent particles (pixels)
  const CONSTRAINT_ITERATIONS = 4; // zyada = stiffer cloth, kam = jelly jaisa
  const TEAR_THRESHOLD = 2.0; // rest length ka 2x se zyada stretch hoga toh toot jaayega

  // physics constants
  let gravity = 800; // pixels/s^2 — neeche kheenchta hai
  let windEnabled = false;
  let windStrength = 120; // horizontal wind force
  const DAMPING = 0.995; // velocity damping — thoda energy nikaalo har frame

  // --- State ---
  let canvasW = 0, canvasH = 0, dpr = 1;
  let particles = []; // [{x, y, oldX, oldY, pinned, mass}]
  let constraints = []; // [{p1, p2, restLength, active}]
  let constraintMap = {}; // "i-j" → constraint index, fast lookup ke liye
  let animationId = null;
  let isVisible = false;
  let lastTime = 0;

  // mouse interaction state
  let isMouseDown = false;
  let mouseX = 0, mouseY = 0;
  let prevMouseX = 0, prevMouseY = 0;
  let grabbedParticle = -1; // index of grabbed particle, -1 = none
  let isTearing = false; // shift+click ya right click = tear mode

  // cloth offset — center mein dikhane ke liye
  let clothOffsetX = 0;
  const clothOffsetY = 40; // top se thoda neeche

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:grab',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Info bar — particle count, links, wind status ---
  const infoDiv = document.createElement('div');
  infoDiv.style.cssText = [
    'margin-top:8px',
    'padding:6px 12px',
    'background:rgba(' + ACCENT_RGB + ',0.05)',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.12)',
    'border-radius:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:#b0b0b0',
    'display:flex',
    'flex-wrap:wrap',
    'gap:16px',
    'align-items:center',
    'justify-content:space-between',
  ].join(';');
  container.appendChild(infoDiv);

  const particleSpan = document.createElement('span');
  infoDiv.appendChild(particleSpan);

  const constraintSpan = document.createElement('span');
  infoDiv.appendChild(constraintSpan);

  const windSpan = document.createElement('span');
  infoDiv.appendChild(windSpan);

  function updateInfo() {
    let activeCount = 0;
    for (let i = 0; i < constraints.length; i++) {
      if (constraints[i].active) activeCount++;
    }
    let pinnedCount = 0;
    for (let i = 0; i < particles.length; i++) {
      if (particles[i].pinned) pinnedCount++;
    }
    particleSpan.textContent = 'particles: ' + particles.length + ' | pinned: ' + pinnedCount;
    constraintSpan.textContent = 'links: ' + activeCount + '/' + constraints.length;
    windSpan.textContent = windEnabled ? 'wind: ON (' + windStrength.toFixed(0) + ')' : 'wind: OFF';
    windSpan.style.color = windEnabled ? ACCENT : '#6b6b6b';
  }

  // --- Controls section ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:space-between',
  ].join(';');
  container.appendChild(controlsDiv);

  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:14px;flex:1;min-width:280px;';
  controlsDiv.appendChild(slidersDiv);

  const buttonsDiv = document.createElement('div');
  buttonsDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;';
  controlsDiv.appendChild(buttonsDiv);

  // --- Slider helper ---
  function createSlider(label, min, max, step, defaultVal, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = 'color:#b0b0b0;font-size:12px;font-weight:600;min-width:14px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText = 'width:85px;height:4px;accent-color:rgba(' + ACCENT_RGB + ',0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText = 'color:#b0b0b0;font-size:11px;min-width:32px;font-family:"JetBrains Mono",monospace;';
    valueEl.textContent = parseFloat(defaultVal).toFixed(step < 1 ? 1 : 0);
    wrapper.appendChild(valueEl);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = val.toFixed(step < 1 ? 1 : 0);
      onChange(val);
      updateInfo();
    });

    slidersDiv.appendChild(wrapper);
    return { slider, valueEl };
  }

  // --- Button helper ---
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:4px 10px',
      'font-size:11px',
      'border-radius:5px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.08)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.2)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.2)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.08)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    buttonsDiv.appendChild(btn);
    return btn;
  }

  // sliders — gravity aur wind strength
  createSlider('gravity', 0, 2000, 10, gravity, (v) => { gravity = v; });
  createSlider('wind', 0, 400, 5, windStrength, (v) => { windStrength = v; });

  // buttons — wind toggle, reset
  const windBtn = createButton('Wind: OFF', () => {
    windEnabled = !windEnabled;
    windBtn.textContent = windEnabled ? 'Wind: ON' : 'Wind: OFF';
    if (windEnabled) {
      windBtn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      windBtn.style.color = ACCENT;
      windBtn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.5)';
    } else {
      windBtn.style.background = 'rgba(' + ACCENT_RGB + ',0.08)';
      windBtn.style.color = '#b0b0b0';
      windBtn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.2)';
    }
    updateInfo();
  });

  createButton('Reset', () => {
    buildCloth();
    updateInfo();
  });

  // --- Canvas sizing — DPR ke saath crisp rendering ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = containerWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // cloth ko center mein rakhna hai
    clothOffsetX = (canvasW - (COLS - 1) * SPACING) / 2;
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // ============================================================
  // CLOTH BUILDING — particles aur constraints banao
  // ============================================================

  function buildCloth() {
    particles = [];
    constraints = [];
    constraintMap = {};
    grabbedParticle = -1;
    isMouseDown = false;
    isTearing = false;

    // cloth offset recalculate karo
    clothOffsetX = (canvasW - (COLS - 1) * SPACING) / 2;

    // particles banao — grid mein
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLS; col++) {
        const x = clothOffsetX + col * SPACING;
        const y = clothOffsetY + row * SPACING;

        particles.push({
          x: x,
          y: y,
          oldX: x, // verlet ke liye previous position = current (zero velocity)
          oldY: y,
          pinned: false,
          mass: 1,
        });
      }
    }

    // top-left aur top-right corners pin karo by default
    particles[0].pinned = true;
    particles[COLS - 1].pinned = true;

    // constraints banao — horizontal aur vertical connections
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLS; col++) {
        const idx = row * COLS + col;

        // horizontal constraint — right neighbor se connect karo
        if (col < COLS - 1) {
          const cIdx = constraints.length;
          constraints.push({
            p1: idx,
            p2: idx + 1,
            restLength: SPACING,
            active: true,
          });
          // map mein dono directions daal do — fast lookup ke liye
          constraintMap[idx + '-' + (idx + 1)] = cIdx;
          constraintMap[(idx + 1) + '-' + idx] = cIdx;
        }

        // vertical constraint — bottom neighbor se connect karo
        if (row < ROWS - 1) {
          const cIdx = constraints.length;
          constraints.push({
            p1: idx,
            p2: idx + COLS,
            restLength: SPACING,
            active: true,
          });
          constraintMap[idx + '-' + (idx + COLS)] = cIdx;
          constraintMap[(idx + COLS) + '-' + idx] = cIdx;
        }
      }
    }
  }

  // initial cloth banao
  buildCloth();

  // ============================================================
  // PHYSICS — Verlet integration + constraint solving
  // Verlet: newPos = 2*pos - oldPos + accel*dt^2
  // equivalent: newPos = pos + (pos - oldPos) + accel*dt^2
  // ============================================================

  function physicsStep(dt) {
    // dt clamp karo — bahut bada dt instability laata hai
    dt = Math.min(dt, 0.025);

    // --- 1. External forces apply karo (gravity + wind) ---
    const gravAccelY = gravity * dt * dt;

    // wind — sinusoidal variation for natural feel
    let windAccelX = 0;
    if (windEnabled) {
      const time = performance.now() / 1000;
      // turbulent wind — main direction + layered sine waves for gusts
      const turbulence = Math.sin(time * 2.3) * 0.3
                       + Math.sin(time * 5.7) * 0.15
                       + Math.sin(time * 0.7) * 0.55;
      windAccelX = windStrength * turbulence * dt * dt;
    }

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      if (p.pinned) continue; // pinned particles hilte nahi

      // verlet integration — position-based dynamics
      const velX = (p.x - p.oldX) * DAMPING;
      const velY = (p.y - p.oldY) * DAMPING;

      p.oldX = p.x;
      p.oldY = p.y;

      // naya position = purana + velocity + acceleration
      p.x += velX + windAccelX;
      p.y += velY + gravAccelY;
    }

    // --- 2. Grabbed particle ko mouse pe le aao ---
    if (grabbedParticle >= 0 && isMouseDown && !isTearing) {
      const p = particles[grabbedParticle];
      if (!p.pinned) {
        // direct snap to mouse — smooth feel ke liye
        p.x = mouseX;
        p.y = mouseY;
        // velocity inject karo via oldPos — throw ke time kaam aayegi
        p.oldX = mouseX - (mouseX - prevMouseX) * 0.5;
        p.oldY = mouseY - (mouseY - prevMouseY) * 0.5;
      }
    }

    // --- 3. Distance constraints solve karo ---
    // har iteration cloth ko stiffer banata hai
    for (let iter = 0; iter < CONSTRAINT_ITERATIONS; iter++) {
      for (let i = 0; i < constraints.length; i++) {
        const c = constraints[i];
        if (!c.active) continue;

        const p1 = particles[c.p1];
        const p2 = particles[c.p2];

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const distSq = dx * dx + dy * dy;
        const dist = Math.sqrt(distSq);

        if (dist < 0.0001) continue; // zero division se bachao

        // tear check — bahut zyada stretch hoga toh constraint tod do
        if (dist > c.restLength * TEAR_THRESHOLD) {
          c.active = false;
          continue;
        }

        // constraint correction — dono particles ko symmetrically adjust karo
        const diff = (c.restLength - dist) / dist;
        const offsetX = dx * diff * 0.5;
        const offsetY = dy * diff * 0.5;

        if (!p1.pinned) {
          p1.x -= offsetX;
          p1.y -= offsetY;
        }
        if (!p2.pinned) {
          p2.x += offsetX;
          p2.y += offsetY;
        }
      }
    }

    // --- 4. Boundary check — canvas ke andar rakho ---
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      if (p.pinned) continue;

      // bottom boundary — ground pe ruko, thoda bounce bhi do
      if (p.y > canvasH - 2) {
        p.y = canvasH - 2;
        // ground friction — horizontal velocity kam karo
        p.oldY = p.y + (p.y - p.oldY) * 0.1;
      }
      if (p.y < 1) p.y = 1;
      if (p.x < 1) p.x = 1;
      if (p.x > canvasW - 1) p.x = canvasW - 1;
    }
  }

  // ============================================================
  // MOUSE / TOUCH INTERACTION
  // ============================================================

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  // sabse nazdeeki particle dhundho mouse position se
  function findNearestParticle(mx, my, maxDist) {
    let nearest = -1;
    let minDistSq = maxDist * maxDist;

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const dx = p.x - mx;
      const dy = p.y - my;
      const distSq = dx * dx + dy * dy;
      if (distSq < minDistSq) {
        minDistSq = distSq;
        nearest = i;
      }
    }

    return nearest;
  }

  // tear — mouse ke aas paas ke constraints tod do
  function tearAtPoint(mx, my) {
    const tearRadiusSq = 15 * 15;
    for (let i = 0; i < constraints.length; i++) {
      const c = constraints[i];
      if (!c.active) continue;

      const p1 = particles[c.p1];
      const p2 = particles[c.p2];

      // constraint ke midpoint se distance check karo
      const midX = (p1.x + p2.x) * 0.5;
      const midY = (p1.y + p2.y) * 0.5;
      const dx = midX - mx;
      const dy = midY - my;

      if (dx * dx + dy * dy < tearRadiusSq) {
        c.active = false;
      }
    }
  }

  // check karo ye particle top row mein hai ya nahi — pin/unpin ke liye
  function isTopRow(particleIdx) {
    return particleIdx < COLS;
  }

  function onPointerDown(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    mouseX = pos.x;
    mouseY = pos.y;
    prevMouseX = mouseX;
    prevMouseY = mouseY;
    isMouseDown = true;

    // right click ya shift+click = tear mode
    isTearing = e.button === 2 || e.shiftKey;

    if (isTearing) {
      tearAtPoint(mouseX, mouseY);
      canvas.style.cursor = 'crosshair';
      return;
    }

    // normal click — nearest particle dhundho
    const nearest = findNearestParticle(mouseX, mouseY, 25);

    if (nearest >= 0) {
      // top row particle hai toh pin/unpin toggle karo
      if (isTopRow(nearest)) {
        particles[nearest].pinned = !particles[nearest].pinned;
        updateInfo();
        return; // grab nahi karna, sirf toggle
      }

      grabbedParticle = nearest;
      canvas.style.cursor = 'grabbing';
    }
  }

  function onPointerMove(e) {
    const pos = getCanvasPos(e);
    prevMouseX = mouseX;
    prevMouseY = mouseY;
    mouseX = pos.x;
    mouseY = pos.y;

    if (isMouseDown && isTearing) {
      tearAtPoint(mouseX, mouseY);
      return;
    }

    if (!isMouseDown) {
      // hover cursor — grab dikhao agar particle ke paas ho
      const nearest = findNearestParticle(mouseX, mouseY, 25);
      if (nearest >= 0) {
        canvas.style.cursor = isTopRow(nearest) ? 'pointer' : 'grab';
      } else {
        canvas.style.cursor = 'default';
      }
    }
  }

  function onPointerUp() {
    if (grabbedParticle >= 0) {
      // release ke time velocity inject karo — satisfying throw feel
      const p = particles[grabbedParticle];
      if (!p.pinned) {
        p.oldX = p.x - (mouseX - prevMouseX) * 1.5;
        p.oldY = p.y - (mouseY - prevMouseY) * 1.5;
      }
    }

    isMouseDown = false;
    grabbedParticle = -1;
    isTearing = false;
    canvas.style.cursor = 'grab';
  }

  // mouse events
  canvas.addEventListener('mousedown', onPointerDown);
  canvas.addEventListener('mousemove', onPointerMove);
  canvas.addEventListener('mouseup', onPointerUp);
  canvas.addEventListener('mouseleave', onPointerUp);

  // right click prevent karo — tear ke liye use ho raha hai
  canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); });

  // touch events
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const fakeEvent = {
      clientX: e.touches[0].clientX,
      clientY: e.touches[0].clientY,
      touches: e.touches,
      button: 0,
      shiftKey: false,
      preventDefault: () => {},
    };
    onPointerDown(fakeEvent);
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const fakeEvent = {
      clientX: e.touches[0].clientX,
      clientY: e.touches[0].clientY,
      touches: e.touches,
      preventDefault: () => {},
    };
    onPointerMove(fakeEvent);
  }, { passive: false });

  canvas.addEventListener('touchend', onPointerUp);
  canvas.addEventListener('touchcancel', onPointerUp);

  // ============================================================
  // RENDERING — filled mesh triangles, stress colors, pinned dots
  // ============================================================

  // constraint active hai ya nahi — fast lookup via map
  function isConstraintActive(i, j) {
    const idx = constraintMap[i + '-' + j];
    if (idx === undefined) return false;
    return constraints[idx].active;
  }

  // stress color — green (relaxed) → yellow (stretched) → red (near breaking)
  function getStressColor(avgDist, restLength, alpha) {
    const stretch = avgDist / restLength;
    // 1.0 = relaxed, TEAR_THRESHOLD = breaking point
    const stress = Math.max(0, Math.min(1, (stretch - 1.0) / (TEAR_THRESHOLD - 1.0)));

    let r, g, b;
    if (stress < 0.5) {
      // green → yellow transition
      const t = stress * 2;
      r = Math.round(80 + 175 * t);
      g = Math.round(200 - 30 * t);
      b = Math.round(60 - 40 * t);
    } else {
      // yellow → red transition
      const t = (stress - 0.5) * 2;
      r = 255;
      g = Math.round(170 - 150 * t);
      b = Math.round(20 - 10 * t);
    }

    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }

  // triangle ke edges ka average distance nikaalo stress ke liye
  function triangleStress(a, b, c) {
    const d1 = Math.sqrt((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y));
    const d2 = Math.sqrt((c.x - a.x) * (c.x - a.x) + (c.y - a.y) * (c.y - a.y));
    return (d1 + d2) * 0.5;
  }

  // cloth mostly intact hai ya nahi — hint text ke liye
  function isClothIntact() {
    let broken = 0;
    for (let i = 0; i < constraints.length; i++) {
      if (!constraints[i].active) broken++;
    }
    return broken < constraints.length * 0.05;
  }

  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // --- Filled triangle mesh draw karo ---
    // har quad (2x2 particles) ko 2 triangles mein split karo
    for (let row = 0; row < ROWS - 1; row++) {
      for (let col = 0; col < COLS - 1; col++) {
        const i0 = row * COLS + col;           // top-left
        const i1 = i0 + 1;                     // top-right
        const i2 = i0 + COLS;                  // bottom-left
        const i3 = i2 + 1;                     // bottom-right

        const p0 = particles[i0];
        const p1 = particles[i1];
        const p2 = particles[i2];
        const p3 = particles[i3];

        // check karo edges active hain ya nahi
        const topActive = isConstraintActive(i0, i1);
        const leftActive = isConstraintActive(i0, i2);
        const rightActive = isConstraintActive(i1, i3);
        const bottomActive = isConstraintActive(i2, i3);

        // upper-left triangle: p0, p1, p2
        if (topActive && leftActive) {
          const stress = triangleStress(p0, p1, p2);
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y);
          ctx.lineTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.closePath();
          ctx.fillStyle = getStressColor(stress, SPACING, 0.35);
          ctx.fill();
        }

        // lower-right triangle: p1, p3, p2
        if (rightActive && bottomActive) {
          const stress = triangleStress(p1, p3, p2);
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p3.x, p3.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.closePath();
          ctx.fillStyle = getStressColor(stress, SPACING, 0.35);
          ctx.fill();
        }
      }
    }

    // --- Constraint lines (subtle wireframe) ---
    ctx.lineWidth = 0.6;
    for (let i = 0; i < constraints.length; i++) {
      const c = constraints[i];
      if (!c.active) continue;

      const p1 = particles[c.p1];
      const p2 = particles[c.p2];

      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = getStressColor(dist, c.restLength, 0.25);
      ctx.stroke();
    }

    // --- Pinned particles — bright amber dots with glow ---
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      if (!p.pinned) continue;

      // outer glow pehle
      ctx.beginPath();
      ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.2)';
      ctx.fill();

      // inner bright dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = ACCENT;
      ctx.fill();
    }

    // --- Grabbed particle highlight — dashed ring ---
    if (grabbedParticle >= 0 && isMouseDown && !isTearing) {
      const p = particles[grabbedParticle];
      ctx.beginPath();
      ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // --- Tear mode cursor indicator — red dashed circle ---
    if (isTearing && isMouseDown) {
      ctx.beginPath();
      ctx.arc(mouseX, mouseY, 15, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(239,68,68,0.5)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // --- Hint text — jab cloth intact hai aur kuch nahi ho raha ---
    if (!isMouseDown && !windEnabled && isClothIntact()) {
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.25)';
      ctx.textAlign = 'center';
      ctx.fillText('grab cloth to drag \u2502 shift+click to tear \u2502 click top edge to pin', canvasW / 2, canvasH - 14);
    }
  }

  // ============================================================
  // ANIMATION LOOP
  // ============================================================

  function animate(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    if (lastTime === 0) lastTime = timestamp;
    let dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;

    // dt clamp — tab switch ke baad bahut bada dt aa sakta hai
    dt = Math.min(dt, 0.05);

    // substeps for stability — Verlet mein important hai
    const subSteps = 3;
    const subDt = dt / subSteps;

    for (let s = 0; s < subSteps; s++) {
      physicsStep(subDt);
    }

    draw();
    updateInfo();

    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — visible hone pe hi animate karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    lastTime = 0;
    resizeCanvas();
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

  // tab switch pe bhi pause karo — CPU/battery bachao
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // initial state — info update karo aur pehla frame draw karo
  updateInfo();
  resizeCanvas();
  draw();
}
