// ============================================================
// Lorenz Attractor — Chaos theory ka poster child
// Do trajectories slightly alag initial conditions se — dekho kaise diverge hoti hain
// RK4 integration, 3D rotation, HSL color cycling
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, chaos simulate karo
export function initLorenz() {
  const container = document.getElementById('lorenzContainer');
  if (!container) {
    console.warn('lorenzContainer nahi mila bhai, Lorenz demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const MAX_TRAIL = 2000; // har trajectory mein max itne points rakhenge
  const INTEGRATION_DT = 0.005; // RK4 step size — chhota rakh stability ke liye
  const STEPS_PER_FRAME = 8; // har frame mein itne integration steps chalao
  const AUTO_ROTATE_SPEED = 0.0008; // radians per frame — dheere dheere ghoomega
  const ACCENT = '#a78bfa'; // purple accent — portfolio theme
  const ACCENT_RGB = '167,139,250';

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;

  // Lorenz parameters — sliders se change honge
  let sigma = 10;
  let rho = 28;
  let beta = 8 / 3;

  // 3D rotation angles — mouse drag se change honge
  let azimuth = -0.6; // horizontal rotation
  let elevation = -0.35; // vertical tilt
  let autoRotate = true;

  // mouse drag state
  let isDragging = false;
  let dragStartX = 0, dragStartY = 0;
  let dragStartAz = 0, dragStartEl = 0;

  // trajectory data — do trajectories for chaos comparison
  let trail1 = []; // primary trajectory — [{x, y, z}]
  let trail2 = []; // secondary trajectory — slightly different initial conditions
  let showDual = true; // dual trajectory toggle

  // current position of each trajectory
  let pos1 = { x: 1.0, y: 1.0, z: 1.0 };
  let pos2 = { x: 1.0, y: 1.0, z: 1.001 }; // 0.001 ka fark — chaos dekho

  // animation
  let animationId = null;
  let isVisible = false;

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(167,139,250,0.15)',
    'border-radius:8px',
    'cursor:grab',
    'background:rgba(2,2,8,0.5)',
    'touch-action:none',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls container
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Helper: slider banao ---
  function createSlider(label, min, max, step, value, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:"JetBrains Mono",monospace;min-width:18px;';
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.style.cssText = 'width:80px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
    wrapper.appendChild(slider);

    const valSpan = document.createElement('span');
    valSpan.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:"JetBrains Mono",monospace;min-width:32px;';
    valSpan.textContent = Number(value).toFixed(1);
    wrapper.appendChild(valSpan);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valSpan.textContent = v.toFixed(1);
      onChange(v);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, valSpan };
  }

  // --- Helper: button banao ---
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:6px 14px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(167,139,250,0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(167,139,250,0.25)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(167,139,250,0.25)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(167,139,250,0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // --- Helper: toggle checkbox banao ---
  function createToggle(label, checked, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:4px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:"JetBrains Mono",monospace;';
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = checked;
    cb.style.cssText = 'accent-color:' + ACCENT + ';cursor:pointer;';
    cb.addEventListener('change', () => { onChange(cb.checked); });
    wrapper.appendChild(cb);

    controlsDiv.appendChild(wrapper);
    return cb;
  }

  // --- Controls banao ---
  // sigma slider
  createSlider('\u03C3', 0, 50, 0.5, sigma, (v) => { sigma = v; });

  // rho slider
  createSlider('\u03C1', 0, 50, 0.5, rho, (v) => { rho = v; });

  // beta slider
  createSlider('\u03B2', 0, 10, 0.1, beta, (v) => { beta = v; });

  // Reset button
  createButton('Reset', resetSimulation);

  // Dual trajectory toggle
  createToggle('Dual', showDual, (v) => { showDual = v; });

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

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Lorenz system equations ---
  // dx/dt = sigma * (y - x)
  // dy/dt = x * (rho - z) - y
  // dz/dt = x * y - beta * z
  function lorenzDerivatives(x, y, z) {
    return {
      dx: sigma * (y - x),
      dy: x * (rho - z) - y,
      dz: x * y - beta * z,
    };
  }

  // --- RK4 integration — Euler se kaafi better stability ---
  // ek step lo aur naya position return karo
  function rk4Step(pos, dt) {
    const { x, y, z } = pos;

    // k1 — current point pe derivatives
    const k1 = lorenzDerivatives(x, y, z);

    // k2 — midpoint pe (k1 se estimate)
    const k2 = lorenzDerivatives(
      x + k1.dx * dt * 0.5,
      y + k1.dy * dt * 0.5,
      z + k1.dz * dt * 0.5
    );

    // k3 — midpoint pe (k2 se estimate)
    const k3 = lorenzDerivatives(
      x + k2.dx * dt * 0.5,
      y + k2.dy * dt * 0.5,
      z + k2.dz * dt * 0.5
    );

    // k4 — endpoint pe (k3 se estimate)
    const k4 = lorenzDerivatives(
      x + k3.dx * dt,
      y + k3.dy * dt,
      z + k3.dz * dt
    );

    // weighted average — RK4 formula
    return {
      x: x + (dt / 6) * (k1.dx + 2 * k2.dx + 2 * k3.dx + k4.dx),
      y: y + (dt / 6) * (k1.dy + 2 * k2.dy + 2 * k3.dy + k4.dy),
      z: z + (dt / 6) * (k1.dz + 2 * k2.dz + 2 * k3.dz + k4.dz),
    };
  }

  // --- 3D to 2D projection ---
  // simple rotation matrix: pehle azimuth (Y-axis), fir elevation (X-axis)
  // fir orthographic projection — z drop karo
  function project3D(x, y, z) {
    // center the attractor — Lorenz ka center roughly (0, 0, rho-1) ke around hota hai
    const cx = 0;
    const cy = 0;
    const cz = rho - 1; // vertical center offset

    // translate to center
    const tx = x - cx;
    const ty = y - cy;
    const tz = z - cz;

    // azimuth rotation (around Z-axis — top view rotation)
    const cosA = Math.cos(azimuth);
    const sinA = Math.sin(azimuth);
    const rx = tx * cosA - ty * sinA;
    const ry = tx * sinA + ty * cosA;
    const rz = tz;

    // elevation rotation (around X-axis — tilt up/down)
    const cosE = Math.cos(elevation);
    const sinE = Math.sin(elevation);
    const ey = ry * cosE - rz * sinE;
    const ez = ry * sinE + rz * cosE;

    // scale factor — Lorenz values roughly -20 to 20 range mein hain
    // canvas ke size ke hisaab se scale karo
    const scale = Math.min(canvasW, canvasH) / 55;

    // project to 2D — canvas ke center pe
    return {
      px: canvasW / 2 + rx * scale,
      py: canvasH / 2 - ey * scale, // Y flip — canvas Y neeche badhta hai
      depth: ez, // depth for optional effects
    };
  }

  // --- Simulation reset ---
  function resetSimulation() {
    // initial conditions — classic starting point
    pos1 = { x: 1.0, y: 1.0, z: 1.0 };
    // 0.001 ka fark — butterfly effect ka demo
    pos2 = { x: 1.0, y: 1.0, z: 1.001 };

    trail1 = [];
    trail2 = [];
  }

  // --- Physics update — har frame pe call hoga ---
  function updateSimulation() {
    for (let i = 0; i < STEPS_PER_FRAME; i++) {
      // RK4 step for trajectory 1
      pos1 = rk4Step(pos1, INTEGRATION_DT);
      trail1.push({ x: pos1.x, y: pos1.y, z: pos1.z });

      // trail length limit
      if (trail1.length > MAX_TRAIL) {
        trail1.shift();
      }

      // trajectory 2 — sirf agar dual mode on hai
      if (showDual) {
        pos2 = rk4Step(pos2, INTEGRATION_DT);
        trail2.push({ x: pos2.x, y: pos2.y, z: pos2.z });

        if (trail2.length > MAX_TRAIL) {
          trail2.shift();
        }
      }
    }

    // auto rotation — dheere dheere azimuth badhaao
    if (autoRotate && !isDragging) {
      azimuth += AUTO_ROTATE_SPEED;
    }
  }

  // --- NaN/Infinity check — diverge ho gaya toh reset karo ---
  function checkDivergence() {
    const limit = 1e6;
    if (!isFinite(pos1.x) || !isFinite(pos1.y) || !isFinite(pos1.z) ||
        Math.abs(pos1.x) > limit || Math.abs(pos1.y) > limit || Math.abs(pos1.z) > limit) {
      resetSimulation();
      return;
    }
    if (showDual) {
      if (!isFinite(pos2.x) || !isFinite(pos2.y) || !isFinite(pos2.z) ||
          Math.abs(pos2.x) > limit || Math.abs(pos2.y) > limit || Math.abs(pos2.z) > limit) {
        // sirf trail2 reset karo, trail1 theek hai
        pos2 = { x: pos1.x + 0.001, y: pos1.y, z: pos1.z };
        trail2 = [];
      }
    }
  }

  // --- Drawing ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // trail 1 draw karo — HSL color cycling position ke basis pe
    drawTrail(trail1, 0);

    // trail 2 draw karo — agar dual mode on hai
    if (showDual && trail2.length > 1) {
      drawTrail(trail2, 1);
    }

    // current position pe bright dot lagao
    drawCurrentPoint(pos1, 0);
    if (showDual) {
      drawCurrentPoint(pos2, 1);
    }

    // info text — top left corner
    drawInfo();
  }

  // --- Trail rendering with HSL color cycling and alpha fade ---
  function drawTrail(trail, trailIndex) {
    if (trail.length < 2) return;

    const len = trail.length;

    // har segment individually draw karo — color aur alpha change hota hai
    for (let i = 1; i < len; i++) {
      const p0 = project3D(trail[i - 1].x, trail[i - 1].y, trail[i - 1].z);
      const p1 = project3D(trail[i].x, trail[i].y, trail[i].z);

      // alpha — purana faded, naya bright
      // oldest point (i=1) pe minimum alpha, newest (i=len-1) pe max
      const t = i / len;
      const alpha = 0.05 + t * 0.75; // range: 0.05 to 0.8

      // hue — position-based cycling
      // trail 1: purple-blue-cyan range (220-300)
      // trail 2: orange-red-pink range (0-60)
      let hue;
      if (trailIndex === 0) {
        // z position se hue decide karo — neechle lobe alag color, upar alag
        const zNorm = (trail[i].z - 5) / 40; // roughly 0 to 1
        hue = 220 + zNorm * 80; // 220 (blue) to 300 (purple/magenta)
      } else {
        // second trajectory — warm colors
        const zNorm = (trail[i].z - 5) / 40;
        hue = 10 + zNorm * 50; // 10 (red-orange) to 60 (yellow-orange)
      }

      const saturation = 70 + t * 20; // newer = more saturated
      const lightness = 45 + t * 20; // newer = brighter

      ctx.beginPath();
      ctx.moveTo(p0.px, p0.py);
      ctx.lineTo(p1.px, p1.py);
      ctx.strokeStyle = 'hsla(' + hue + ',' + saturation + '%,' + lightness + '%,' + alpha + ')';
      ctx.lineWidth = 0.8 + t * 1.2; // newer = thicker — 0.8 to 2.0
      ctx.stroke();
    }
  }

  // --- Current position pe bright dot ---
  function drawCurrentPoint(pos, trailIndex) {
    const p = project3D(pos.x, pos.y, pos.z);

    // glow effect
    const glowColor = trailIndex === 0
      ? 'rgba(167,139,250,0.6)' // purple glow
      : 'rgba(250,167,100,0.6)'; // orange glow
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 12;

    ctx.beginPath();
    ctx.arc(p.px, p.py, 3, 0, Math.PI * 2);

    const fillColor = trailIndex === 0
      ? 'rgba(200,180,255,1)' // bright purple-white
      : 'rgba(255,200,150,1)'; // bright orange-white
    ctx.fillStyle = fillColor;
    ctx.fill();

    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }

  // --- Info text — parameters aur divergence indicator ---
  function drawInfo() {
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.fillText('\u03C3=' + sigma.toFixed(1) + '  \u03C1=' + rho.toFixed(1) + '  \u03B2=' + beta.toFixed(2), 10, 16);

    // agar dual mode on hai toh divergence distance dikhao
    if (showDual && trail1.length > 0 && trail2.length > 0) {
      const dx = pos1.x - pos2.x;
      const dy = pos1.y - pos2.y;
      const dz = pos1.z - pos2.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      ctx.fillText('\u0394d=' + dist.toFixed(2), 10, 30);
    }

    // points count
    ctx.textAlign = 'right';
    ctx.fillText('pts: ' + trail1.length, canvasW - 10, 16);

    // hint jab trail chhota ho
    if (trail1.length < 50) {
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.25)';
      ctx.textAlign = 'center';
      ctx.fillText('drag to rotate \u2022 sliders to tweak chaos', canvasW / 2, canvasH - 14);
    }
  }

  // --- Mouse/touch events — view rotation ke liye ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    const pos = getCanvasPos(e);
    dragStartX = pos.x;
    dragStartY = pos.y;
    dragStartAz = azimuth;
    dragStartEl = elevation;
    canvas.style.cursor = 'grabbing';
    autoRotate = false; // drag karte waqt auto-rotate band karo
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const pos = getCanvasPos(e);
    const dx = pos.x - dragStartX;
    const dy = pos.y - dragStartY;

    // sensitivity — pixels to radians
    azimuth = dragStartAz + dx * 0.008;
    // elevation clamp karo — ulta mat ho jaaye
    elevation = Math.max(-Math.PI / 2 + 0.1, Math.min(Math.PI / 2 - 0.1,
      dragStartEl + dy * 0.008));
  });

  canvas.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      canvas.style.cursor = 'grab';
      autoRotate = true; // drag band toh auto-rotate wapas chalu
    }
  });

  canvas.addEventListener('mouseleave', () => {
    if (isDragging) {
      isDragging = false;
      canvas.style.cursor = 'grab';
      autoRotate = true;
    }
  });

  // touch support — mobile pe bhi kaam kare
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isDragging = true;
    const pos = getCanvasPos(e);
    dragStartX = pos.x;
    dragStartY = pos.y;
    dragStartAz = azimuth;
    dragStartEl = elevation;
    autoRotate = false;
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDragging) return;
    const pos = getCanvasPos(e);
    const dx = pos.x - dragStartX;
    const dy = pos.y - dragStartY;

    azimuth = dragStartAz + dx * 0.008;
    elevation = Math.max(-Math.PI / 2 + 0.1, Math.min(Math.PI / 2 - 0.1,
      dragStartEl + dy * 0.008));
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    isDragging = false;
    autoRotate = true;
  }, { passive: false });

  // --- Animation loop ---
  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    updateSimulation();
    checkDivergence();
    draw();

    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf jab dikhe tab animate karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
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

  // tab switch pe pause — battery bachao
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // --- Initial state set karo ---
  resetSimulation();
}
