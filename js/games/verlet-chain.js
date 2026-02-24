// ============================================================
// Verlet Rope/Chain — Verlet integration se satisfying rope physics
// Particles array connected by distance constraints, drag karke khelo
// Modes: Rope (top pin), Bridge (both ends pinned), Free (no pins)
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, chain simulate karo
export function initVerletChain() {
  const container = document.getElementById('verletChainContainer');
  if (!container) {
    console.warn('verletChainContainer nahi mila bhai, verlet chain skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 380;
  const FLOOR_MARGIN = 30;          // floor canvas ke bottom se kitna upar

  // --- Default physics params — sliders se change honge ---
  let segmentCount = 25;            // kitne particles chain mein
  let gravity = 900;                // pixels/s^2 neeche ki taraf
  let constraintIters = 10;         // distance constraints kitni baar solve karna
  let dampingFactor = 0.99;         // velocity damping har frame pe
  let segmentLength = 12;           // har do particles ke beech ka rest distance

  // --- Mode: 'rope' | 'bridge' | 'free' ---
  let mode = 'rope';

  // --- Particles array — har particle mein x, y, oldX, oldY, pinned ---
  let particles = [];

  // --- Interaction state ---
  let isDragging = false;
  let dragIndex = -1;               // konsa particle pakda hai
  let mouseX = 0, mouseY = 0;       // current mouse position canvas coords mein
  let mouseInCanvas = false;         // mouse canvas ke andar hai ya nahi

  // --- Canvas dimensions ---
  let canvasW = 0, canvasH = 0, dpr = 1;

  // --- Animation state ---
  let animationId = null;
  let isVisible = false;
  let lastTime = 0;

  // --- Floor Y position ---
  let floorY = CANVAS_HEIGHT - FLOOR_MARGIN;

  // ============================================================
  // PARTICLE SYSTEM — initialize, reset, create
  // ============================================================

  function createParticles() {
    particles = [];
    const startX = canvasW / 2;
    const startY = 50;

    if (mode === 'rope') {
      // rope — top center se latkti hui, seedhi neeche
      for (let i = 0; i < segmentCount; i++) {
        const px = startX;
        const py = startY + i * segmentLength;
        particles.push({
          x: px, y: py,
          oldX: px, oldY: py,
          pinned: i === 0   // sirf pehla particle pinned
        });
      }
    } else if (mode === 'bridge') {
      // bridge — horizontal, dono ends pinned
      const totalLen = (segmentCount - 1) * segmentLength;
      const bridgeStartX = (canvasW - totalLen) / 2;
      const bridgeY = CANVAS_HEIGHT * 0.3;
      for (let i = 0; i < segmentCount; i++) {
        const px = bridgeStartX + i * segmentLength;
        const py = bridgeY;
        particles.push({
          x: px, y: py,
          oldX: px, oldY: py,
          pinned: i === 0 || i === segmentCount - 1
        });
      }
    } else {
      // free — koi pin nahi, center mein horizontal
      const totalLen = (segmentCount - 1) * segmentLength;
      const freeStartX = (canvasW - totalLen) / 2;
      const freeY = CANVAS_HEIGHT * 0.35;
      for (let i = 0; i < segmentCount; i++) {
        const px = freeStartX + i * segmentLength;
        const py = freeY;
        particles.push({
          x: px, y: py,
          oldX: px, oldY: py,
          pinned: false
        });
      }
    }
  }

  // ============================================================
  // DOM STRUCTURE — canvas + controls banao
  // ============================================================

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText += ';width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(245,158,11,0.15)',
    'border-radius:8px',
    'cursor:default',
    'background:transparent',
    'margin-top:8px',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

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

  // --- Slider helper — reusable slider factory ---
  function createSlider(label, min, max, step, defaultVal, unit, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = "color:#6b6b6b;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(defaultVal);
    slider.style.cssText = 'width:70px;height:4px;accent-color:#f59e0b;cursor:pointer;';
    wrapper.appendChild(slider);

    const val = document.createElement('span');
    val.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:36px;";
    const decimals = step < 1 ? (step < 0.01 ? 3 : 2) : 0;
    val.textContent = Number(defaultVal).toFixed(decimals) + (unit || '');
    wrapper.appendChild(val);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      val.textContent = v.toFixed(decimals) + (unit || '');
      onChange(v);
    });

    slidersDiv.appendChild(wrapper);
    return { slider, val };
  }

  // --- Button helper ---
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
      'white-space:nowrap',
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
    buttonsDiv.appendChild(btn);
    return btn;
  }

  // --- Mode toggle button helper — active state dikhane ke liye ---
  let modeButtons = {};
  function createModeButton(text, modeVal) {
    const btn = createButton(text, () => {
      mode = modeVal;
      updateModeButtons();
      createParticles();
    });
    modeButtons[modeVal] = btn;
    return btn;
  }

  function updateModeButtons() {
    // active mode button ko highlight karo
    Object.keys(modeButtons).forEach((key) => {
      const btn = modeButtons[key];
      if (key === mode) {
        btn.style.background = 'rgba(245,158,11,0.3)';
        btn.style.color = '#f59e0b';
        btn.style.borderColor = 'rgba(245,158,11,0.5)';
      } else {
        btn.style.background = 'rgba(245,158,11,0.1)';
        btn.style.color = '#b0b0b0';
        btn.style.borderColor = 'rgba(245,158,11,0.25)';
      }
    });
  }

  // --- Sliders banao ---
  const segSlider = createSlider('segs', 10, 50, 1, segmentCount, '', (v) => {
    segmentCount = Math.round(v);
    createParticles();
  });

  createSlider('gravity', 0, 2000, 10, gravity, '', (v) => { gravity = v; });

  createSlider('iters', 1, 20, 1, constraintIters, '', (v) => {
    constraintIters = Math.round(v);
  });

  createSlider('damp', 0.90, 1.00, 0.01, dampingFactor, '', (v) => { dampingFactor = v; });

  // --- Mode buttons ---
  createModeButton('Rope', 'rope');
  createModeButton('Bridge', 'bridge');
  createModeButton('Free', 'free');

  // reset button
  createButton('Reset', () => {
    createParticles();
    isDragging = false;
    dragIndex = -1;
  });

  // initial mode highlight
  updateModeButtons();

  // ============================================================
  // CANVAS SIZING — DPR aware, crisp rendering ke liye
  // ============================================================

  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;
    floorY = CANVAS_HEIGHT - FLOOR_MARGIN;

    canvas.width = containerWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvas();
  window.addEventListener('resize', () => {
    const oldW = canvasW;
    resizeCanvas();
    // agar canvas width change hua toh particles ko re-create karo
    // nahi toh purani positions se scene bigad jaayega
    if (Math.abs(oldW - canvasW) > 5) {
      createParticles();
    }
  });

  // ============================================================
  // PHYSICS — Verlet integration, constraint solving, floor collision
  // ============================================================

  function physicsStep(dt) {
    // dt ko clamp karo — tab switch se bahut bada dt aa sakta hai
    dt = Math.min(dt, 0.033);

    // --- Verlet integration: position update ---
    // new_pos = pos + (pos - old_pos) * damping + acceleration * dt^2
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      if (p.pinned) continue;

      // current velocity = pos - oldPos (Verlet mein velocity implicit hai)
      const vx = (p.x - p.oldX) * dampingFactor;
      const vy = (p.y - p.oldY) * dampingFactor;

      // purani position save karo
      p.oldX = p.x;
      p.oldY = p.y;

      // nayi position = current + velocity + gravity * dt^2
      p.x += vx;
      p.y += vy + gravity * dt * dt;
    }

    // --- Constraint solving — distance constraints ko enforce karo ---
    // multiple iterations se chain stiff hoti hai
    for (let iter = 0; iter < constraintIters; iter++) {
      // distance constraints — har consecutive pair ke beech
      for (let i = 0; i < particles.length - 1; i++) {
        const a = particles[i];
        const b = particles[i + 1];

        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // agar distance zero hai toh skip — division by zero se bachao
        if (dist < 0.0001) continue;

        // kitna correct karna hai — target distance vs actual
        const diff = (segmentLength - dist) / dist;
        const offsetX = dx * diff * 0.5;
        const offsetY = dy * diff * 0.5;

        // dono particles ko equally push/pull karo (agar pinned nahi hai)
        if (!a.pinned) {
          a.x -= offsetX;
          a.y -= offsetY;
        }
        if (!b.pinned) {
          b.x += offsetX;
          b.y += offsetY;
        }
      }

      // --- Floor collision — particles floor ke neeche nahi jaani chahiye ---
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        if (p.pinned) continue;

        if (p.y > floorY) {
          // bounce — thoda energy absorb karo
          p.y = floorY;
          // velocity reverse karo with restitution (bounce factor)
          const vyBefore = p.y - p.oldY;
          p.oldY = p.y + vyBefore * 0.3; // 0.3 restitution — thoda bouncy
        }

        // canvas ke bahar mat jaane do horizontally bhi
        if (p.x < 5) { p.x = 5; p.oldX = p.x; }
        if (p.x > canvasW - 5) { p.x = canvasW - 5; p.oldX = p.x; }

        // top boundary bhi
        if (p.y < 5) { p.y = 5; p.oldY = p.y; }
      }
    }

    // --- Rope mode: pehla particle mouse follow kare jab drag nahi ho raha ---
    if (mode === 'rope' && !isDragging && mouseInCanvas && particles.length > 0) {
      particles[0].x = mouseX;
      particles[0].y = mouseY;
    }

    // --- Dragged particle ko mouse pe le aao ---
    if (isDragging && dragIndex >= 0 && dragIndex < particles.length) {
      particles[dragIndex].x = mouseX;
      particles[dragIndex].y = mouseY;
    }
  }

  // ============================================================
  // INTERACTION — mouse/touch se particles ko pakdo
  // ============================================================

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  }

  // sabse closest particle dhundho mouse se
  function findClosestParticle(mx, my) {
    let closest = -1;
    let closestDist = Infinity;
    const grabRadius = 20; // kitne px ke andar grab ho jaaye

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const dx = p.x - mx;
      const dy = p.y - my;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < closestDist && dist < grabRadius) {
        closestDist = dist;
        closest = i;
      }
    }
    return closest;
  }

  function onPointerDown(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    mouseX = pos.x;
    mouseY = pos.y;
    mouseInCanvas = true;

    const closest = findClosestParticle(pos.x, pos.y);
    if (closest >= 0) {
      isDragging = true;
      dragIndex = closest;
      canvas.style.cursor = 'grabbing';
    }
  }

  function onPointerMove(e) {
    const pos = getCanvasPos(e);
    mouseX = pos.x;
    mouseY = pos.y;
    mouseInCanvas = true;

    if (isDragging) {
      e.preventDefault();
    } else {
      // hover pe cursor change karo
      const closest = findClosestParticle(pos.x, pos.y);
      canvas.style.cursor = closest >= 0 ? 'grab' : 'default';
    }
  }

  function onPointerUp() {
    if (isDragging) {
      isDragging = false;
      dragIndex = -1;
      canvas.style.cursor = 'default';
    }
  }

  function onPointerLeave() {
    mouseInCanvas = false;
    onPointerUp();
  }

  // mouse events
  canvas.addEventListener('mousedown', onPointerDown);
  canvas.addEventListener('mousemove', onPointerMove);
  canvas.addEventListener('mouseup', onPointerUp);
  canvas.addEventListener('mouseleave', onPointerLeave);

  // touch events — mobile pe bhi kaam kare
  canvas.addEventListener('touchstart', (e) => { onPointerDown(e); }, { passive: false });
  canvas.addEventListener('touchmove', (e) => { onPointerMove(e); }, { passive: false });
  canvas.addEventListener('touchend', onPointerUp, { passive: false });
  canvas.addEventListener('touchcancel', onPointerUp, { passive: false });

  // ============================================================
  // DRAWING — rope, beads, glow, floor, sab kuch yahan
  // ============================================================

  // color gradient calculate karo — chain ke along top se bottom
  function getParticleColor(index, alpha) {
    // amber (#f59e0b) se warm red (#ef4444) tak gradient
    const t = particles.length > 1 ? index / (particles.length - 1) : 0;

    // HSL mein interpolate — amber hue ~38, red hue ~0
    // amber: hsl(38, 92%, 50%)  red: hsl(10, 85%, 57%)
    const h = 38 - t * 28;             // 38 -> 10
    const s = 92 - t * 7;              // 92 -> 85
    const l = 50 + t * 7;              // 50 -> 57

    return 'hsla(' + h.toFixed(0) + ',' + s.toFixed(0) + '%,' + l.toFixed(0) + '%,' + alpha + ')';
  }

  function drawFloor() {
    // floor line — subtle dashed line
    ctx.strokeStyle = 'rgba(245,158,11,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(10, floorY);
    ctx.lineTo(canvasW - 10, floorY);
    ctx.stroke();
    ctx.setLineDash([]);

    // floor hatching — chhoti diagonal lines
    ctx.strokeStyle = 'rgba(245,158,11,0.07)';
    ctx.lineWidth = 1;
    for (let hx = 10; hx < canvasW - 10; hx += 12) {
      ctx.beginPath();
      ctx.moveTo(hx, floorY);
      ctx.lineTo(hx - 5, floorY + 8);
      ctx.stroke();
    }
  }

  function drawRopeSmooth() {
    if (particles.length < 2) return;

    // --- Thick rope using quadratic bezier curves through midpoints ---
    // ye technique smooth continuous curve deti hai particles ke through

    // rope shadow — depth ke liye
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // shadow pass — thoda offset, dark
    ctx.beginPath();
    ctx.moveTo(particles[0].x + 2, particles[0].y + 2);
    for (let i = 1; i < particles.length - 1; i++) {
      const midX = (particles[i].x + particles[i + 1].x) / 2 + 2;
      const midY = (particles[i].y + particles[i + 1].y) / 2 + 2;
      ctx.quadraticCurveTo(particles[i].x + 2, particles[i].y + 2, midX, midY);
    }
    const last = particles[particles.length - 1];
    ctx.lineTo(last.x + 2, last.y + 2);
    ctx.strokeStyle = 'rgba(0,0,0,0.2)';
    ctx.lineWidth = 6;
    ctx.stroke();

    // --- Main rope — gradient segments ---
    // har segment ko apna color dena hai, toh segments mein draw karna padega
    for (let i = 0; i < particles.length - 1; i++) {
      const a = particles[i];
      const b = particles[i + 1];

      // gradient along segment
      const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
      grad.addColorStop(0, getParticleColor(i, 0.7));
      grad.addColorStop(1, getParticleColor(i + 1, 0.7));

      ctx.beginPath();

      if (i === 0) {
        // pehla segment — seedhi line
        ctx.moveTo(a.x, a.y);
        if (particles.length > 2) {
          const midX = (b.x + particles[i + 2].x) / 2;
          const midY = (b.y + particles[i + 2].y) / 2;
          ctx.quadraticCurveTo(b.x, b.y, midX, midY);
        } else {
          ctx.lineTo(b.x, b.y);
        }
      } else if (i === particles.length - 2) {
        // last segment
        const prevMidX = (particles[i - 1].x + a.x) / 2;
        const prevMidY = (particles[i - 1].y + a.y) / 2;
        ctx.moveTo(prevMidX, prevMidY);
        ctx.quadraticCurveTo(a.x, a.y, b.x, b.y);
      } else {
        // middle segments — midpoint se midpoint
        const prevMidX = (particles[i - 1].x + a.x) / 2;
        const prevMidY = (particles[i - 1].y + a.y) / 2;
        const nextMidX = (b.x + particles[i + 2].x) / 2;
        const nextMidY = (b.y + particles[i + 2].y) / 2;
        ctx.moveTo(prevMidX, prevMidY);
        ctx.quadraticCurveTo(a.x, a.y, (a.x + b.x) / 2, (a.y + b.y) / 2);
      }

      ctx.strokeStyle = grad;
      ctx.lineWidth = 4;
      ctx.stroke();
    }

    ctx.restore();
  }

  function drawBeads() {
    // har particle pe ek bead (circle) draw karo
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const isGrabbed = isDragging && i === dragIndex;
      const isFirstInRope = mode === 'rope' && i === 0;
      const isPinnedPoint = p.pinned;

      // bead size — pinned points thode bade
      const radius = isPinnedPoint ? 5 : 3.5;
      const color = getParticleColor(i, 1.0);

      // glow agar grabbed hai ya first point (rope mode mein)
      if (isGrabbed || (isFirstInRope && mouseInCanvas)) {
        ctx.save();
        ctx.shadowColor = '#f59e0b';
        ctx.shadowBlur = isGrabbed ? 18 : 10;

        ctx.beginPath();
        ctx.arc(p.x, p.y, radius + 2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(245,158,11,0.3)';
        ctx.fill();

        ctx.restore();
      }

      // pinned point ka special marker — chhota diamond shape
      if (isPinnedPoint) {
        ctx.save();
        ctx.shadowColor = '#f59e0b';
        ctx.shadowBlur = 6;

        // outer ring
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius + 3, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(245,158,11,0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.restore();
      }

      // bead body
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // inner highlight — chhota bright spot, depth ke liye
      if (radius > 3) {
        ctx.beginPath();
        ctx.arc(p.x - radius * 0.25, p.y - radius * 0.25, radius * 0.35, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.25)';
        ctx.fill();
      }

      // grabbed ring — dashed circle
      if (isGrabbed) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius + 6, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(245,158,11,0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  function drawInfo() {
    // top-right mein chhota info text — current mode + stats
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(107,107,107,0.5)';

    const modeLabel = mode === 'rope' ? 'Rope' : mode === 'bridge' ? 'Bridge' : 'Free';
    ctx.fillText(modeLabel + ' | ' + segmentCount + ' segs | ' + constraintIters + ' iters', canvasW - 12, 18);
  }

  function drawHint() {
    // hint text — jab kuch nahi ho raha
    if (isDragging) return;

    ctx.font = "12px 'JetBrains Mono', monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.2)';
    ctx.textAlign = 'center';

    if (mode === 'rope') {
      ctx.fillText('move mouse to control top point, click any bead to drag', canvasW / 2, canvasH - 10);
    } else {
      ctx.fillText('click and drag any bead on the chain', canvasW / 2, canvasH - 10);
    }
  }

  // --- Main draw function ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    drawFloor();
    drawRopeSmooth();
    drawBeads();
    drawInfo();
    drawHint();
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

    // dt clamp karo — tab switch se aaye toh bahut bada dt aa sakta hai
    dt = Math.min(dt, 0.05);

    // substeps — stability ke liye multiple chhote steps
    const subSteps = 3;
    const subDt = dt / subSteps;
    for (let s = 0; s < subSteps; s++) {
      physicsStep(subDt);
    }

    draw();
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — off-screen hone pe pause karo ---
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

  // tab switch pe bhi pause karo — battery bachao
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // --- Initial setup ---
  resizeCanvas();
  createParticles();
  draw();
}
