// ============================================================
// Pendulum Wave — Frequency ratio ka visual jaadu
// Alag alag lengths ke pendulums SHM karte hain, wave patterns
// emerge hote hain jab frequencies thodi thodi alag hoti hain
// T_k = T_cycle/(N_base + k), sab ek cycle baad sync ho jaate hain
// ============================================================

// yahi se sab shuru hota hai — pendulums latkao aur wave banao
export function initPendulumWave() {
  const container = document.getElementById('pendulumWaveContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  const ACCENT_RGB = '245,158,11';
  const BG = '#111';
  const FONT = "'JetBrains Mono',monospace";

  // --- State ---
  let animationId = null, isVisible = false, canvasW = 0;
  let pendulumCount = 15;
  let cyclePeriod = 60;       // seconds mein — ek full cycle jab sab sync ho jaayein
  let speed = 1.0;
  let isPaused = false;
  let simTime = 0;            // seconds mein simulation ka time
  const amplitude = 0.45;     // radians mein max angle
  const N_BASE = 51;          // base frequency multiple — zyada = tighter wave patterns
  const RAIL_Y = 40;          // rail ki Y position (top se)
  const BOB_RADIUS = 10;

  // --- DOM setup ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:${BG};border:1px solid rgba(${ACCENT_RGB},0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls container ---
  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  // --- Helper: slider banao ---
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
    const dec = step < 1 ? (step < 0.01 ? 2 : 1) : 0;
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

  // --- Helper: button banao ---
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
  makeSlider('Count', 10, 25, 1, pendulumCount, (v) => { pendulumCount = Math.round(v); });
  makeSlider('Cycle(s)', 20, 120, 5, cyclePeriod, (v) => { cyclePeriod = v; });
  makeSlider('Speed', 0.1, 3.0, 0.1, speed, (v) => { speed = v; });

  const pauseBtn = makeBtn('Pause', () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Play' : 'Pause';
  });

  makeBtn('Reset', () => { simTime = 0; });

  // --- Resize handler ---
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- Pendulum ka angle calculate karo ---
  // T_k = cyclePeriod / (N_BASE + k), angle = amplitude * cos(2*pi*t/T_k)
  function getAngle(k, t) {
    const Tk = cyclePeriod / (N_BASE + k);
    return amplitude * Math.cos(2 * Math.PI * t / Tk);
  }

  // --- Render function ---
  function draw() {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, canvasW, CANVAS_HEIGHT);

    // pendulums ke beech ka spacing calculate karo
    const margin = 40;
    const spacing = (canvasW - 2 * margin) / (pendulumCount - 1 || 1);

    // rail draw karo — top pe horizontal bar
    ctx.strokeStyle = `rgba(${ACCENT_RGB},0.4)`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(margin - 15, RAIL_Y);
    ctx.lineTo(canvasW - margin + 15, RAIL_Y);
    ctx.stroke();

    // rail ke supports — dono sides pe
    ctx.lineWidth = 2;
    ctx.strokeStyle = `rgba(${ACCENT_RGB},0.2)`;
    ctx.beginPath();
    ctx.moveTo(margin - 15, RAIL_Y);
    ctx.lineTo(margin - 15, RAIL_Y - 15);
    ctx.moveTo(canvasW - margin + 15, RAIL_Y);
    ctx.lineTo(canvasW - margin + 15, RAIL_Y - 15);
    ctx.stroke();

    // max string length — canvas ke bottom tak fit ho jaaye
    const maxLen = CANVAS_HEIGHT - RAIL_Y - BOB_RADIUS - 30;
    const minLen = maxLen * 0.35;

    // har pendulum draw karo
    for (let k = 0; k < pendulumCount; k++) {
      // length proportional to period squared (T ∝ √L, so L ∝ T²)
      // longer period = longer string
      const Tk = cyclePeriod / (N_BASE + k);
      const T0 = cyclePeriod / N_BASE;
      const Tmax = cyclePeriod / (N_BASE + pendulumCount - 1);
      // normalize: k=0 sabse lamba, k=N-1 sabse chhota
      const frac = (Tk - Tmax) / (T0 - Tmax || 1);
      const stringLen = minLen + frac * (maxLen - minLen);

      const angle = getAngle(k, simTime);
      const pivotX = margin + k * spacing;
      const pivotY = RAIL_Y;

      // bob position — pivot se angle pe latkta hai
      const bobX = pivotX + stringLen * Math.sin(angle);
      const bobY = pivotY + stringLen * Math.cos(angle);

      // string draw karo
      ctx.beginPath();
      ctx.moveTo(pivotX, pivotY);
      ctx.lineTo(bobX, bobY);
      ctx.strokeStyle = `rgba(255,255,255,0.15)`;
      ctx.lineWidth = 1;
      ctx.stroke();

      // pivot point — chhotey dot pe
      ctx.beginPath();
      ctx.arc(pivotX, pivotY, 2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${ACCENT_RGB},0.4)`;
      ctx.fill();

      // bob draw karo — gradient wala glow effect
      // color hue k ke hisaab se — rainbow across pendulums
      const hue = (k / pendulumCount) * 300;
      const grad = ctx.createRadialGradient(bobX, bobY, 0, bobX, bobY, BOB_RADIUS);
      grad.addColorStop(0, `hsla(${hue},85%,70%,0.95)`);
      grad.addColorStop(0.6, `hsla(${hue},80%,55%,0.8)`);
      grad.addColorStop(1, `hsla(${hue},70%,40%,0.3)`);

      ctx.beginPath();
      ctx.arc(bobX, bobY, BOB_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // subtle glow
      ctx.save();
      ctx.shadowColor = `hsla(${hue},80%,60%,0.4)`;
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(bobX, bobY, BOB_RADIUS * 0.6, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${hue},85%,65%,0.6)`;
      ctx.fill();
      ctx.restore();
    }

    // connecting line — saare bobs ko ek line se jod do (wave dikhega)
    ctx.beginPath();
    let first = true;
    for (let k = 0; k < pendulumCount; k++) {
      const Tk = cyclePeriod / (N_BASE + k);
      const T0 = cyclePeriod / N_BASE;
      const Tmax = cyclePeriod / (N_BASE + pendulumCount - 1);
      const frac = (Tk - Tmax) / (T0 - Tmax || 1);
      const stringLen = minLen + frac * (maxLen - minLen);
      const angle = getAngle(k, simTime);
      const pivotX = margin + k * spacing;
      const bobX = pivotX + stringLen * Math.sin(angle);
      const bobY = RAIL_Y + stringLen * Math.cos(angle);
      if (first) { ctx.moveTo(bobX, bobY); first = false; }
      else ctx.lineTo(bobX, bobY);
    }
    ctx.strokeStyle = `rgba(${ACCENT_RGB},0.25)`;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // info text — time aur count
    ctx.font = `10px ${FONT}`;
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.textAlign = 'left';
    ctx.fillText(`N=${pendulumCount}  t=${simTime.toFixed(1)}s  cycle=${cyclePeriod}s`, 8, 16);

    // cycle progress bar — kitna cycle complete hua
    const progress = (simTime % cyclePeriod) / cyclePeriod;
    const barW = 100, barH = 3, barX = canvasW - barW - 10, barY = 12;
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fillRect(barX, barY, barW, barH);
    ctx.fillStyle = `rgba(${ACCENT_RGB},0.5)`;
    ctx.fillRect(barX, barY, barW * progress, barH);
  }

  // --- Animation loop ---
  let lastTime = 0;
  function loop(timestamp) {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    if (lastTime === 0) lastTime = timestamp;
    const dt = (timestamp - lastTime) / 1000; // seconds mein
    lastTime = timestamp;

    if (!isPaused) {
      simTime += dt * speed;
    }

    draw();
    animationId = requestAnimationFrame(loop);
  }

  // --- Intersection Observer — off-screen pe band karo ---
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) { lastTime = 0; animationId = requestAnimationFrame(loop); }
    else if (!isVisible && animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) { lastTime = 0; animationId = requestAnimationFrame(loop); } });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && isVisible && !animationId) { lastTime = 0; animationId = requestAnimationFrame(loop); } });
}
