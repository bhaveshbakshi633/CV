// ============================================================
// Chaos Game — Iterated Function Systems ka visual demo
// Regular polygon ke vertices rakh, random point se shuru karo
// Har step pe: random vertex chun, fraction r se us taraf move kar
// N=3, r=0.5 → Sierpinski triangle — fractal jaadu hai ye
// Restriction rules se aur bhi crazy patterns bante hain
// ============================================================

// yahi se shuru — vertices rakh, points plot kar, fractal dekh
export function initChaosGame() {
  const container = document.getElementById('chaosGameContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  const ACCENT_RGB = '245,158,11';
  const BG = '#111';
  const FONT = "'JetBrains Mono',monospace";
  const POINTS_PER_FRAME = 2000; // har frame mein itne points plot karo — speed ke liye

  // --- State ---
  let animationId = null, isVisible = false, canvasW = 0;
  let sides = 3;                // polygon ke sides (3-8)
  let jumpRatio = 0.5;          // fraction r — vertex ki taraf kitna move karo
  let restriction = 'none';     // 'none', 'no-repeat', 'no-adjacent'
  let totalPoints = 0;
  let isPaused = false;

  // current point position
  let currentX = 0, currentY = 0;
  let lastVertex = -1;          // pichla chosen vertex index

  // vertices of polygon
  let vertices = [];

  // --- DOM setup ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:${BG};border:1px solid rgba(${ACCENT_RGB},0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  // --- Helpers ---
  function makeSlider(label, min, max, step, val, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;align-items:center;gap:6px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = `color:#6b6b6b;font-size:11px;font-family:${FONT};white-space:nowrap;`;
    lbl.textContent = label;
    wrap.appendChild(lbl);
    const slider = document.createElement('input');
    slider.type = 'range'; slider.min = String(min); slider.max = String(max);
    slider.step = String(step); slider.value = String(val);
    slider.style.cssText = `width:80px;height:4px;accent-color:${ACCENT};cursor:pointer;`;
    wrap.appendChild(slider);
    const vSpan = document.createElement('span');
    const dec = step < 1 ? 2 : 0;
    vSpan.style.cssText = `color:#f0f0f0;font-size:11px;font-family:${FONT};min-width:28px;`;
    vSpan.textContent = Number(val).toFixed(dec);
    wrap.appendChild(vSpan);
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      vSpan.textContent = v.toFixed(dec);
      onChange(v);
    });
    ctrl.appendChild(wrap);
    return { slider, vSpan };
  }

  function makeBtn(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = `padding:5px 12px;font-size:11px;border-radius:6px;cursor:pointer;background:rgba(${ACCENT_RGB},0.1);color:#b0b0b0;border:1px solid rgba(${ACCENT_RGB},0.25);font-family:${FONT};transition:all 0.2s;`;
    btn.addEventListener('mouseenter', () => { btn.style.background = `rgba(${ACCENT_RGB},0.25)`; btn.style.color = '#e0e0e0'; });
    btn.addEventListener('mouseleave', () => { btn.style.background = `rgba(${ACCENT_RGB},0.1)`; btn.style.color = '#b0b0b0'; });
    btn.addEventListener('click', onClick);
    ctrl.appendChild(btn);
    return btn;
  }

  // --- Controls ---
  const sidesCtrl = makeSlider('Sides', 3, 8, 1, sides, (v) => {
    sides = Math.round(v);
    resetGame();
  });

  const ratioCtrl = makeSlider('Jump', 0.3, 0.7, 0.01, jumpRatio, (v) => {
    jumpRatio = v;
    resetGame();
  });

  // restriction dropdown banaate hain — select element
  const restrictWrap = document.createElement('div');
  restrictWrap.style.cssText = 'display:flex;align-items:center;gap:6px;';
  const restrictLbl = document.createElement('span');
  restrictLbl.style.cssText = `color:#6b6b6b;font-size:11px;font-family:${FONT};`;
  restrictLbl.textContent = 'Rule';
  restrictWrap.appendChild(restrictLbl);

  const restrictSelect = document.createElement('select');
  restrictSelect.style.cssText = `background:rgba(${ACCENT_RGB},0.1);color:#b0b0b0;border:1px solid rgba(${ACCENT_RGB},0.25);border-radius:6px;padding:4px 8px;font-size:11px;font-family:${FONT};cursor:pointer;outline:none;`;

  const optNone = document.createElement('option');
  optNone.value = 'none'; optNone.textContent = 'None';
  optNone.style.cssText = 'background:#1a1a2e;color:#b0b0b0;';
  restrictSelect.appendChild(optNone);

  const optNoRepeat = document.createElement('option');
  optNoRepeat.value = 'no-repeat'; optNoRepeat.textContent = 'No Repeat';
  optNoRepeat.style.cssText = 'background:#1a1a2e;color:#b0b0b0;';
  restrictSelect.appendChild(optNoRepeat);

  const optNoAdj = document.createElement('option');
  optNoAdj.value = 'no-adjacent'; optNoAdj.textContent = 'No Adjacent';
  optNoAdj.style.cssText = 'background:#1a1a2e;color:#b0b0b0;';
  restrictSelect.appendChild(optNoAdj);

  restrictSelect.addEventListener('change', () => {
    restriction = restrictSelect.value;
    resetGame();
  });
  restrictWrap.appendChild(restrictSelect);
  ctrl.appendChild(restrictWrap);

  makeBtn('Clear', () => { resetGame(); });

  const pauseBtn = makeBtn('Pause', () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Play' : 'Pause';
  });

  // stats
  const statsSpan = document.createElement('span');
  statsSpan.style.cssText = `font-size:10px;font-family:${FONT};color:#6b6b6b;`;
  ctrl.appendChild(statsSpan);

  // --- Resize ---
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // vertices recalculate karo — canvas size change hua toh
    calculateVertices();
    resetGame();
  }
  resize();
  window.addEventListener('resize', resize);

  // --- Polygon vertices calculate karo ---
  function calculateVertices() {
    vertices = [];
    const cx = canvasW / 2;
    const cy = CANVAS_HEIGHT / 2;
    const r = Math.min(canvasW, CANVAS_HEIGHT) * 0.42; // radius

    for (let i = 0; i < sides; i++) {
      // top se clockwise — -π/2 offset taaki pehla vertex top pe ho
      const angle = (2 * Math.PI * i / sides) - Math.PI / 2;
      vertices.push({
        x: cx + r * Math.cos(angle),
        y: cy + r * Math.sin(angle)
      });
    }
  }

  // --- Game reset ---
  function resetGame() {
    calculateVertices();
    // random starting point — polygon ke andar kahi bhi
    currentX = canvasW / 2 + (Math.random() - 0.5) * 50;
    currentY = CANVAS_HEIGHT / 2 + (Math.random() - 0.5) * 50;
    lastVertex = -1;
    totalPoints = 0;

    // canvas clear karo
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, canvasW, CANVAS_HEIGHT);

    // vertices draw karo — clear ke baad bhi dikhne chahiye
    drawVertices();
  }

  // --- Vertices draw karo ---
  function drawVertices() {
    for (let i = 0; i < vertices.length; i++) {
      const v = vertices[i];
      // vertex dot
      ctx.beginPath();
      ctx.arc(v.x, v.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = ACCENT;
      ctx.fill();

      // vertex label
      ctx.font = `9px ${FONT}`;
      ctx.fillStyle = `rgba(${ACCENT_RGB},0.7)`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      // label thoda bahar rakh do vertex se
      const cx = canvasW / 2, cy = CANVAS_HEIGHT / 2;
      const dx = v.x - cx, dy = v.y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const labelX = v.x + (dx / dist) * 14;
      const labelY = v.y + (dy / dist) * 14;
      ctx.fillText(String(i + 1), labelX, labelY);
    }
  }

  // --- Random vertex choose karo restriction ke saath ---
  function chooseVertex() {
    let chosen = Math.floor(Math.random() * sides);
    let attempts = 0;
    const maxAttempts = 50; // infinite loop se bacho

    while (attempts < maxAttempts) {
      if (restriction === 'none') break;

      if (restriction === 'no-repeat' && chosen === lastVertex) {
        chosen = Math.floor(Math.random() * sides);
        attempts++;
        continue;
      }

      if (restriction === 'no-adjacent') {
        // adjacent check — circular polygon mein
        if (lastVertex >= 0) {
          const diff = Math.abs(chosen - lastVertex);
          const isAdj = (diff === 1) || (diff === sides - 1);
          if (isAdj) {
            chosen = Math.floor(Math.random() * sides);
            attempts++;
            continue;
          }
        }
      }

      break;
    }

    lastVertex = chosen;
    return chosen;
  }

  // --- Points plot karo — har frame mein batch ---
  function plotPoints() {
    for (let i = 0; i < POINTS_PER_FRAME; i++) {
      const vi = chooseVertex();
      const v = vertices[vi];

      // current point se vertex ki taraf jumpRatio move karo
      currentX = currentX + (v.x - currentX) * jumpRatio;
      currentY = currentY + (v.y - currentY) * jumpRatio;
      totalPoints++;

      // pehle 20 points skip karo — transient phase (convergence ke liye)
      if (totalPoints < 20) continue;

      // point plot karo — tiny dot
      // color vertex ke hisaab se — har vertex ka apna hue
      const hue = (vi / sides) * 300;
      ctx.fillStyle = `hsla(${hue},80%,60%,0.6)`;
      ctx.fillRect(currentX, currentY, 1.2, 1.2);
    }
  }

  // initial draw
  calculateVertices();
  resetGame();

  // --- Animation loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    if (!isPaused) {
      plotPoints();
      // vertices redraw karo — points ke upar dikhne chahiye
      drawVertices();
    }

    // stats update
    const ptStr = totalPoints > 1000000 ? (totalPoints / 1000000).toFixed(1) + 'M' :
                  totalPoints > 1000 ? (totalPoints / 1000).toFixed(1) + 'k' : String(totalPoints);
    statsSpan.textContent = `points: ${ptStr}`;

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
