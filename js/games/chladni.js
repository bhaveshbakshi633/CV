// ============================================================
// Chladni Patterns — vibrating plate pe sand particles nodal lines pe settle hote hain
// z(x,y) = A*cos(n*pi*x/L)*cos(m*pi*y/L) mode shapes
// ============================================================

export function initChladni() {
  const container = document.getElementById('chladniContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  let animationId = null, isVisible = false, canvasW = 0;

  // mode numbers — frequency slider se change honge
  let modeN = 3, modeM = 2;
  let dampingFactor = 0.02; // kitni tez particles settle hongi
  const NUM_PARTICLES = 2000;

  // particles array — {x, y} normalized 0-1
  let particles = [];

  // preset modes — frequency slider ke steps
  const modes = [
    [1, 1], [2, 1], [1, 2], [2, 2], [3, 1], [1, 3],
    [3, 2], [2, 3], [3, 3], [4, 1], [1, 4], [4, 2],
    [2, 4], [4, 3], [3, 4], [4, 4], [5, 1], [1, 5],
    [5, 2], [2, 5], [5, 3], [3, 5], [5, 5], [6, 3],
  ];

  let freqIndex = 6; // default mode index

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

  // freq label alag se track karo
  let freqLabel = null;

  function createSlider(label, min, max, step, val, onChange) {
    const w = document.createElement('div');
    w.style.cssText = 'display:flex;align-items:center;gap:5px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#6b6b6b;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    lbl.textContent = label;
    w.appendChild(lbl);
    const sl = document.createElement('input');
    sl.type = 'range'; sl.min = String(min); sl.max = String(max); sl.step = String(step); sl.value = String(val);
    sl.style.cssText = 'width:90px;height:4px;accent-color:#f59e0b;cursor:pointer;';
    w.appendChild(sl);
    const vl = document.createElement('span');
    vl.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:40px;";
    vl.textContent = String(val);
    w.appendChild(vl);
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      vl.textContent = step < 1 ? v.toFixed(3) : String(Math.round(v));
      onChange(v);
    });
    ctrl.appendChild(w);
    return vl;
  }

  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = "padding:5px 12px;font-size:11px;border-radius:6px;cursor:pointer;background:rgba(245,158,11,0.1);color:#b0b0b0;border:1px solid rgba(245,158,11,0.25);font-family:'JetBrains Mono',monospace;transition:all 0.2s ease;";
    btn.addEventListener('mouseenter', () => { btn.style.background = 'rgba(245,158,11,0.25)'; btn.style.color = '#e0e0e0'; });
    btn.addEventListener('mouseleave', () => { btn.style.background = 'rgba(245,158,11,0.1)'; btn.style.color = '#b0b0b0'; });
    btn.addEventListener('click', onClick);
    ctrl.appendChild(btn);
  }

  freqLabel = createSlider('freq', 0, modes.length - 1, 1, freqIndex, (v) => {
    freqIndex = Math.round(v);
    modeN = modes[freqIndex][0];
    modeM = modes[freqIndex][1];
    freqLabel.textContent = '(' + modeN + ',' + modeM + ')';
  });
  freqLabel.textContent = '(' + modeN + ',' + modeM + ')';

  createSlider('damp', 0.005, 0.08, 0.005, dampingFactor, (v) => { dampingFactor = v; });
  createButton('Reset', resetParticles);

  // --- particles initialize karo randomly ---
  function resetParticles() {
    particles = [];
    for (let i = 0; i < NUM_PARTICLES; i++) {
      particles.push({
        x: Math.random(),
        y: Math.random(),
      });
    }
  }
  resetParticles();

  // --- Chladni function: z(x,y) at given position ---
  // z = cos(n*pi*x) * cos(m*pi*y) - cos(m*pi*x) * cos(n*pi*y)
  // ye dega nodal lines jahan z=0
  function chladni(x, y) {
    const n = modeN, m = modeM;
    return Math.cos(n * Math.PI * x) * Math.cos(m * Math.PI * y)
         - Math.cos(m * Math.PI * x) * Math.cos(n * Math.PI * y);
  }

  // gradient of chladni function — particles isko follow karke nodal lines pe jayenge
  function chladniGrad(x, y) {
    const n = modeN, m = modeM;
    const eps = 0.002;
    const zx = (chladni(x + eps, y) - chladni(x - eps, y)) / (2 * eps);
    const zy = (chladni(x, y + eps) - chladni(x, y - eps)) / (2 * eps);
    return [zx, zy];
  }

  // --- resize ---
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- update particles ---
  function updateParticles() {
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const z = chladni(p.x, p.y);
      const [gx, gy] = chladniGrad(p.x, p.y);

      // particles ko nodal lines ki taraf push karo
      // force = -z * gradient(z) — ye z=0 ke surfaces ki taraf le jaayega
      const fx = -z * gx * dampingFactor;
      const fy = -z * gy * dampingFactor;

      // thoda random jitter bhi daalo — realistic vibration ke liye
      const jitter = 0.001;
      p.x += fx + (Math.random() - 0.5) * jitter;
      p.y += fy + (Math.random() - 0.5) * jitter;

      // boundary clamp karo — plate ke andar rahe
      p.x = Math.max(0.01, Math.min(0.99, p.x));
      p.y = Math.max(0.01, Math.min(0.99, p.y));
    }
  }

  // --- draw ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // plate area — square, centered
    const plateSize = Math.min(canvasW - 40, CANVAS_HEIGHT - 40);
    const offsetX = (canvasW - plateSize) / 2;
    const offsetY = (CANVAS_HEIGHT - plateSize) / 2;

    // plate background
    ctx.fillStyle = 'rgba(20,20,30,0.8)';
    ctx.fillRect(offsetX, offsetY, plateSize, plateSize);
    ctx.strokeStyle = 'rgba(245,158,11,0.2)';
    ctx.lineWidth = 1;
    ctx.strokeRect(offsetX, offsetY, plateSize, plateSize);

    // nodal lines faintly dikhao — background reference ke liye
    ctx.strokeStyle = 'rgba(245,158,11,0.06)';
    ctx.lineWidth = 1;
    const gridRes = 80;
    for (let gy = 0; gy < gridRes; gy++) {
      for (let gx = 0; gx < gridRes; gx++) {
        const x0 = gx / gridRes, y0 = gy / gridRes;
        const x1 = (gx + 1) / gridRes, y1 = (gy + 1) / gridRes;
        const z00 = chladni(x0, y0), z10 = chladni(x1, y0);
        const z01 = chladni(x0, y1), z11 = chladni(x1, y1);
        // sign change detect karo — nodal line pass hoti hai
        if ((z00 > 0) !== (z10 > 0) || (z00 > 0) !== (z01 > 0)) {
          const cx = offsetX + (gx + 0.5) / gridRes * plateSize;
          const cy = offsetY + (gy + 0.5) / gridRes * plateSize;
          ctx.fillStyle = 'rgba(245,158,11,0.08)';
          ctx.fillRect(cx, cy, plateSize / gridRes, plateSize / gridRes);
        }
      }
    }

    // particles draw karo — amber dots
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const cx = offsetX + p.x * plateSize;
      const cy = offsetY + p.y * plateSize;
      // clipping check
      if (cx < offsetX || cx > offsetX + plateSize || cy < offsetY || cy > offsetY + plateSize) continue;

      ctx.beginPath();
      ctx.arc(cx, cy, 1.3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(245,158,11,0.7)';
      ctx.fill();
    }

    // mode label dikhao
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'left';
    ctx.fillText('CHLADNI PATTERNS', 8, 14);
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(245,158,11,0.4)';
    ctx.fillText('mode (' + modeN + ',' + modeM + ')', canvasW - 8, 14);
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.2)';
    ctx.fillText(NUM_PARTICLES + ' particles', 8, CANVAS_HEIGHT - 10);
  }

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    // multiple update steps per frame
    for (let s = 0; s < 3; s++) {
      updateParticles();
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
