// ============================================================
// Double Pendulum — Lagrangian mechanics se chaos theory ka live demo
// Drag karke angles set karo, rainbow trail dekho, chaos enjoy karo
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, pendulum simulate karo
export function initDoublePendulum() {
  const container = document.getElementById('doublePendulumContainer');
  if (!container) {
    console.warn('doublePendulumContainer nahi mila bhai, double pendulum skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const MAX_TRAIL = 500;

  // --- Default physics params — sliders se change honge ---
  let m1 = 2.0;   // pehle bob ka mass (kg)
  let m2 = 1.5;   // doosre bob ka mass (kg)
  let L1 = 120;   // pehli rod ki length (px)
  let L2 = 100;   // doosri rod ki length (px)
  let g = 9.81;   // gravity (m/s^2, scaled for pixels)
  let damping = 0.999; // energy loss — 1.0 = no damping
  let trailMax = 400;  // kitne trail points dikhane hain

  // --- Pendulum state ---
  // theta1, theta2 = angles from vertical (radians)
  // omega1, omega2 = angular velocities (rad/s)
  let theta1 = Math.PI / 2;   // 90 degrees se shuru — chaos ke liye accha starting point
  let theta2 = Math.PI / 1.2; // thoda alag angle — asymmetry chahiye chaos ke liye
  let omega1 = 0;
  let omega2 = 0;

  // trail — second bob ki position history
  let trail = []; // [{x, y, hue}]
  let hueCounter = 0; // rainbow ke liye hue cycle karta rahega

  // --- Interaction state ---
  let isDragging = false;
  let dragBob = 0;     // 1 ya 2 — konsa bob pakda hai
  let isPaused = false; // drag ke time pause, release pe start

  // --- Canvas dimensions ---
  let canvasW = 0, canvasH = 0, dpr = 1;

  // --- Pivot point — top center ---
  let pivotX = 0, pivotY = 0;

  // --- Animation state ---
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
    'border:1px solid rgba(245,158,11,0.15)',
    'border-radius:6px',
    'cursor:grab',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);

  // --- Controls section ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:14px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // slider banane ka helper — ye baar baar use hoga
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
    val.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:36px;";
    val.textContent = Number(value).toFixed(step < 1 ? (step < 0.01 ? 3 : 1) : 0) + (unit || '');
    wrapper.appendChild(val);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      // decimal places step size se decide karo
      const decimals = step < 1 ? (step < 0.01 ? 3 : 1) : 0;
      val.textContent = v.toFixed(decimals) + (unit || '');
      onChange(v);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, val };
  }

  // button banane ka helper
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

  // --- Sliders banao ---
  createSlider('m\u2081', 0.5, 5.0, 0.1, m1, 'kg', (v) => { m1 = v; });
  createSlider('m\u2082', 0.5, 5.0, 0.1, m2, 'kg', (v) => { m2 = v; });
  createSlider('g', 1.0, 20.0, 0.1, g, '', (v) => { g = v; });
  createSlider('trail', 50, 500, 10, trailMax, '', (v) => {
    trailMax = v;
    // trail ko naye max tak truncate karo
    if (trail.length > trailMax) trail = trail.slice(trail.length - trailMax);
  });
  createSlider('damp', 0.990, 1.000, 0.001, damping, '', (v) => { damping = v; });

  // reset button — sab default pe le aao
  createButton('Reset', () => {
    theta1 = Math.PI / 2;
    theta2 = Math.PI / 1.2;
    omega1 = 0;
    omega2 = 0;
    trail = [];
    hueCounter = 0;
    isPaused = false;
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

    // pivot top-center mein, thoda neeche taaki pendulum acche se dikhe
    pivotX = canvasW / 2;
    pivotY = canvasH * 0.2;
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Bob positions calculate karo angles se ---
  function getBobPositions() {
    const x1 = pivotX + L1 * Math.sin(theta1);
    const y1 = pivotY + L1 * Math.cos(theta1);
    const x2 = x1 + L2 * Math.sin(theta2);
    const y2 = y1 + L2 * Math.cos(theta2);
    return { x1, y1, x2, y2 };
  }

  // ============================================================
  // PHYSICS: Lagrangian mechanics — double pendulum equations of motion
  // ye equations Euler-Lagrange se derive hote hain
  // L = T - V (kinetic - potential energy)
  // RK4 integration use karenge stability ke liye
  // ============================================================

  // state = [theta1, omega1, theta2, omega2]
  // derivatives return karta hai = [omega1, alpha1, omega2, alpha2]
  function derivatives(state) {
    const [th1, w1, th2, w2] = state;
    const dth = th1 - th2; // angle difference — equations mein baar baar aata hai
    const sinDth = Math.sin(dth);
    const cosDth = Math.cos(dth);

    // denominator — dono alpha equations mein common hai
    // den = (2*m1 + m2 - m2*cos(2*th1 - 2*th2))
    // ye form numerically zyada stable hai
    const den = 2 * m1 + m2 - m2 * Math.cos(2 * dth);

    // alpha1 = d(omega1)/dt — Lagrangian se nikla hua
    // (-g*(2*m1+m2)*sin(th1) - m2*g*sin(th1-2*th2)
    //  - 2*sin(th1-th2)*m2*(w2^2*L2 + w1^2*L1*cos(th1-th2)))
    // / (L1 * den)
    const num1 = -g * (2 * m1 + m2) * Math.sin(th1)
               - m2 * g * Math.sin(th1 - 2 * th2)
               - 2 * sinDth * m2 * (w2 * w2 * L2 + w1 * w1 * L1 * cosDth);
    const alpha1 = num1 / (L1 * den);

    // alpha2 = d(omega2)/dt
    // (2*sin(th1-th2) * (w1^2*L1*(m1+m2) + g*(m1+m2)*cos(th1) + w2^2*L2*m2*cos(th1-th2)))
    // / (L2 * den)
    const num2 = 2 * sinDth * (w1 * w1 * L1 * (m1 + m2)
               + g * (m1 + m2) * Math.cos(th1)
               + w2 * w2 * L2 * m2 * cosDth);
    const alpha2 = num2 / (L2 * den);

    return [w1, alpha1, w2, alpha2];
  }

  // RK4 integration — 4th order Runge-Kutta, Euler se kahin zyada accurate
  // energy conservation achhi hoti hai isse, chaos mein bhi stable rehta hai
  function rk4Step(state, dt) {
    const k1 = derivatives(state);

    const s2 = [
      state[0] + k1[0] * dt / 2,
      state[1] + k1[1] * dt / 2,
      state[2] + k1[2] * dt / 2,
      state[3] + k1[3] * dt / 2,
    ];
    const k2 = derivatives(s2);

    const s3 = [
      state[0] + k2[0] * dt / 2,
      state[1] + k2[1] * dt / 2,
      state[2] + k2[2] * dt / 2,
      state[3] + k2[3] * dt / 2,
    ];
    const k3 = derivatives(s3);

    const s4 = [
      state[0] + k3[0] * dt,
      state[1] + k3[1] * dt,
      state[2] + k3[2] * dt,
      state[3] + k3[3] * dt,
    ];
    const k4 = derivatives(s4);

    // weighted average — RK4 ka formula
    return [
      state[0] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt / 6,
      state[1] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt / 6,
      state[2] + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) * dt / 6,
      state[3] + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) * dt / 6,
    ];
  }

  // --- Physics step — ek frame ka simulation ---
  function physicsStep(dt) {
    // RK4 se naya state nikaalo
    const state = [theta1, omega1, theta2, omega2];
    const newState = rk4Step(state, dt);

    theta1 = newState[0];
    omega1 = newState[1] * damping; // damping lagao — energy slowly kam hogi
    theta2 = newState[2];
    omega2 = newState[3] * damping;

    // trail update — second bob ki position store karo with rainbow hue
    const pos = getBobPositions();
    hueCounter = (hueCounter + 0.8) % 360; // slowly cycle through rainbow
    trail.push({ x: pos.x2, y: pos.y2, hue: hueCounter });

    // trail length limit karo
    if (trail.length > trailMax) {
      trail = trail.slice(trail.length - trailMax);
    }
  }

  // ============================================================
  // INTERACTION: Click/drag se bobs ko pakdo aur angle set karo
  // ============================================================

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  // check karo ki mouse/touch kisi bob ke paas hai ya nahi
  function hitTest(mx, my) {
    const { x1, y1, x2, y2 } = getBobPositions();
    const bobRadius = 16; // hit area thoda bada rakhte hain — easy grabbing ke liye

    // pehle second bob check karo — wo upar hota toh overlap mein priority mile
    const d2 = Math.sqrt((mx - x2) * (mx - x2) + (my - y2) * (my - y2));
    if (d2 < bobRadius + 10) return 2;

    const d1 = Math.sqrt((mx - x1) * (mx - x1) + (my - y1) * (my - y1));
    if (d1 < bobRadius + 10) return 1;

    return 0; // kisi bob pe nahi hai
  }

  function onPointerDown(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const hit = hitTest(pos.x, pos.y);

    if (hit > 0) {
      isDragging = true;
      dragBob = hit;
      isPaused = true; // simulation rok do jab drag kar rahe ho
      canvas.style.cursor = 'grabbing';
    }
  }

  function onPointerMove(e) {
    if (!isDragging) {
      // hover pe cursor change karo — grab dikhao agar bob pe ho
      const pos = getCanvasPos(e);
      const hit = hitTest(pos.x, pos.y);
      canvas.style.cursor = hit > 0 ? 'grab' : 'default';
      return;
    }

    e.preventDefault();
    const pos = getCanvasPos(e);

    if (dragBob === 1) {
      // pehla bob — angle pivot se calculate karo
      const dx = pos.x - pivotX;
      const dy = pos.y - pivotY;
      theta1 = Math.atan2(dx, dy); // atan2(sin component, cos component)
    } else if (dragBob === 2) {
      // doosra bob — angle pehle bob ke position se calculate karo
      const { x1, y1 } = getBobPositions();
      const dx = pos.x - x1;
      const dy = pos.y - y1;
      theta2 = Math.atan2(dx, dy);
    }

    // velocities zero karo jab drag ho raha ho — release pe fresh start
    omega1 = 0;
    omega2 = 0;
  }

  function onPointerUp() {
    if (isDragging) {
      isDragging = false;
      isPaused = false; // simulation wapas shuru
      canvas.style.cursor = 'grab';
      // trail clear karo — nayi trajectory ke liye fresh trail
      trail = [];
      hueCounter = 0;
    }
  }

  // mouse events
  canvas.addEventListener('mousedown', onPointerDown);
  canvas.addEventListener('mousemove', onPointerMove);
  canvas.addEventListener('mouseup', onPointerUp);
  canvas.addEventListener('mouseleave', onPointerUp);

  // touch events
  canvas.addEventListener('touchstart', (e) => { onPointerDown(e); }, { passive: false });
  canvas.addEventListener('touchmove', (e) => { onPointerMove(e); }, { passive: false });
  canvas.addEventListener('touchend', onPointerUp, { passive: false });
  canvas.addEventListener('touchcancel', onPointerUp, { passive: false });

  // ============================================================
  // DRAWING — pendulum, trail, glow, sab kuch yahan
  // ============================================================

  function draw() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvasW, canvasH);

    const { x1, y1, x2, y2 } = getBobPositions();

    // --- Rainbow trail draw karo — second bob ki path ---
    drawTrail(ctx);

    // --- Rods draw karo — pivot se bob1, bob1 se bob2 ---
    drawRods(ctx, x1, y1, x2, y2);

    // --- Pivot point ---
    drawPivot(ctx);

    // --- Bobs draw karo with glow ---
    drawBob(ctx, x1, y1, m1, '#f59e0b', isDragging && dragBob === 1);
    drawBob(ctx, x2, y2, m2, '#ef4444', isDragging && dragBob === 2);

    // --- Ghost position jab drag ho raha ho ---
    if (isDragging) {
      drawGhostIndicator(ctx);
    }

    // --- Energy display — top right mein chhota text ---
    drawEnergyInfo(ctx, x1, y1, x2, y2);

    // hint jab kuch nahi ho raha
    if (omega1 === 0 && omega2 === 0 && !isDragging && trail.length === 0) {
      ctx.font = "12px 'JetBrains Mono', monospace";
      ctx.fillStyle = 'rgba(176,176,176,0.25)';
      ctx.textAlign = 'center';
      ctx.fillText('drag either bob to set initial position', canvasW / 2, canvasH - 20);
    }
  }

  function drawTrail(ctx) {
    if (trail.length < 2) return;

    // trail segments — har segment ka apna hue hai (rainbow effect)
    for (let i = 1; i < trail.length; i++) {
      // alpha fade — purane points zyada transparent
      const t = i / trail.length; // 0 (oldest) to 1 (newest)
      const alpha = t * t * 0.7; // quadratic fade — smooth transition

      ctx.beginPath();
      ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
      ctx.lineTo(trail[i].x, trail[i].y);

      // rainbow hue — HSL use karo
      ctx.strokeStyle = 'hsla(' + trail[i].hue + ', 90%, 60%, ' + alpha + ')';
      ctx.lineWidth = 1.5 + t * 1.0; // naye points thode mote
      ctx.stroke();
    }
  }

  function drawRods(ctx, x1, y1, x2, y2) {
    // rod 1 — pivot se bob1
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(x1, y1);
    ctx.strokeStyle = 'rgba(200,200,200,0.4)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // rod 2 — bob1 se bob2
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'rgba(200,200,200,0.35)';
    ctx.lineWidth = 1.8;
    ctx.stroke();
  }

  function drawPivot(ctx) {
    // pivot point — chhota dot at top center
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(150,150,150,0.6)';
    ctx.fill();

    // pivot mounting bracket — aesthetic ke liye
    ctx.beginPath();
    ctx.moveTo(pivotX - 12, pivotY);
    ctx.lineTo(pivotX + 12, pivotY);
    ctx.strokeStyle = 'rgba(150,150,150,0.35)';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function drawBob(ctx, x, y, mass, color, isGrabbed) {
    // bob ka radius mass se proportional — taaki visually pata chale
    const radius = 8 + mass * 2.5;

    // glow effect — shadow blur se
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = isGrabbed ? radius * 3 : radius * 1.5;

    // radial gradient — center bright, edge fade
    const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
    grad.addColorStop(0, color);
    grad.addColorStop(0.6, color);
    grad.addColorStop(1, 'rgba(0,0,0,0)');

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // inner bright core — depth feel ke liye
    ctx.beginPath();
    ctx.arc(x - radius * 0.2, y - radius * 0.2, radius * 0.35, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.fill();

    ctx.restore();

    // grabbed indicator — ring dikhao
    if (isGrabbed) {
      ctx.beginPath();
      ctx.arc(x, y, radius + 5, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function drawGhostIndicator(ctx) {
    // chhota text dikhao — "release to start"
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = 'rgba(245,158,11,0.5)';
    ctx.textAlign = 'center';
    ctx.fillText('release to start', canvasW / 2, canvasH - 12);
  }

  function drawEnergyInfo(ctx, x1, y1, x2, y2) {
    // total energy calculate karo — debugging + cool info ke liye
    // KE = 0.5 * m * v^2 (using angular velocities)
    // PE = -m * g * h (height from pivot)

    // heights — pivot se neeche positive hai, toh cos(theta) * L = height below pivot
    const h1 = -L1 * Math.cos(theta1); // negative = below pivot
    const h2 = h1 - L2 * Math.cos(theta2);

    // potential energy (reference = pivot level)
    const PE = m1 * g * h1 + m2 * g * h2;

    // kinetic energy — double pendulum KE formula
    const v1sq = L1 * L1 * omega1 * omega1;
    const v2sq = L1 * L1 * omega1 * omega1
               + L2 * L2 * omega2 * omega2
               + 2 * L1 * L2 * omega1 * omega2 * Math.cos(theta1 - theta2);
    const KE = 0.5 * m1 * v1sq + 0.5 * m2 * v2sq;

    const totalE = (KE + PE) / 1000; // scale down for display

    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(107,107,107,0.5)';
    ctx.fillText('E: ' + totalE.toFixed(1), canvasW - 10, 18);
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
    let dt = (timestamp - lastTime) / 1000; // seconds mein convert
    lastTime = timestamp;

    // dt clamp karo — tab switch se aaye toh bahut bada dt aa sakta hai
    dt = Math.min(dt, 0.05);

    // physics step — sirf jab paused nahi hai
    if (!isPaused) {
      // substeps for stability — RK4 hai toh zyada zarurat nahi, but safe rehna
      // gravity scale karna padega — pixels mein kaam kar rahe hain
      const physDt = dt;
      const subSteps = 4; // 4 substeps per frame — smooth and stable
      const subDt = physDt / subSteps;

      for (let s = 0; s < subSteps; s++) {
        physicsStep(subDt);
      }
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
}
