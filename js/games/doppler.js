// ============================================================
// Doppler Effect — Moving source se wavefront compression ka demo
// Source move karta hai, circular wavefronts emit hote hain
// Jab v > v_sound, Mach cone banta hai — supersonic boom wali feel
// Wavefronts age ke saath fade hote hain, naye bright hote hain
// ============================================================

// yahi se shuru — source chhodo aur waves dekhte jao
export function initDoppler() {
  const container = document.getElementById('dopplerContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  const ACCENT_RGB = '245,158,11';
  const BG = '#111';
  const FONT = "'JetBrains Mono',monospace";
  const SOUND_SPEED = 150;       // pixels/sec mein sound ki speed
  const EMIT_INTERVAL = 0.15;    // har 0.15 sec pe ek wavefront emit karo

  // --- State ---
  let animationId = null, isVisible = false, canvasW = 0;
  let sourceSpeed = 0.5;         // fraction of sound speed (0 to 2)
  let isPaused = false;
  let simTime = 0;
  let lastEmitTime = 0;

  // source position aur velocity
  let sourceX = 0, sourceY = CANVAS_HEIGHT / 2;
  let sourceVx = 0;

  // wavefronts array — har ek {cx, cy, r, birthTime}
  let wavefronts = [];
  const MAX_WAVEFRONTS = 120;    // memory limit — purane hata do

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
    slider.style.cssText = `width:100px;height:4px;accent-color:${ACCENT};cursor:pointer;`;
    wrap.appendChild(slider);
    const vSpan = document.createElement('span');
    const dec = step < 1 ? 2 : 0;
    vSpan.style.cssText = `color:#f0f0f0;font-size:11px;font-family:${FONT};min-width:36px;`;
    vSpan.textContent = Number(val).toFixed(dec);
    wrap.appendChild(vSpan);
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      vSpan.textContent = v.toFixed(dec);
      onChange(v, vSpan);
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
  const speedCtrl = makeSlider('v/c', 0.0, 2.0, 0.05, sourceSpeed, (v, span) => {
    sourceSpeed = v;
    // Mach number bhi dikhao
    span.textContent = v.toFixed(2) + (v > 1 ? ' M>' + v.toFixed(1) : '');
  });

  const pauseBtn = makeBtn('Pause', () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Play' : 'Pause';
  });

  makeBtn('Reset', () => {
    resetSim();
  });

  // --- Reset ---
  function resetSim() {
    wavefronts = [];
    simTime = 0;
    lastEmitTime = 0;
    sourceX = canvasW * 0.15;
    sourceY = CANVAS_HEIGHT / 2;
    sourceVx = sourceSpeed * SOUND_SPEED;
  }

  // --- Resize ---
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // pehli baar source ko set karo
  resetSim();

  // --- Simulation update ---
  function update(dt) {
    // source ki actual velocity calculate karo — slider se control
    sourceVx = sourceSpeed * SOUND_SPEED;

    // source move karo
    sourceX += sourceVx * dt;

    // agar source canvas ke bahar nikal jaaye toh wrap karo
    if (sourceX > canvasW + 50) {
      sourceX = -50;
      wavefronts = []; // purane wavefronts hata do — naye cycle ke liye
    }

    // wavefront emit karo har interval pe
    if (simTime - lastEmitTime >= EMIT_INTERVAL) {
      wavefronts.push({
        cx: sourceX,
        cy: sourceY,
        birthTime: simTime
      });
      lastEmitTime = simTime;

      // purane wavefronts hata do — MAX_WAVEFRONTS se zyada nahi chahiye
      if (wavefronts.length > MAX_WAVEFRONTS) {
        wavefronts.shift();
      }
    }
  }

  // --- Render ---
  function draw() {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, canvasW, CANVAS_HEIGHT);

    // --- Wavefronts draw karo — circles jo expand ho rahe hain ---
    for (let i = 0; i < wavefronts.length; i++) {
      const wf = wavefronts[i];
      const age = simTime - wf.birthTime;
      const radius = age * SOUND_SPEED; // r = v_sound * t

      // agar radius bahut bada ho gaya toh skip karo
      if (radius > canvasW * 1.5) continue;

      // age ke hisaab se opacity — naye bright, purane fade
      const maxAge = MAX_WAVEFRONTS * EMIT_INTERVAL;
      const ageFrac = Math.min(1, age / maxAge);
      const alpha = 0.7 * (1 - ageFrac * ageFrac); // quadratic fade

      if (alpha < 0.02) continue;

      // color — naye amber, purane blue shift (Doppler color analogy)
      // compression side = blue shift, expansion side = red shift
      // yahan hum age-based color kar rahe hain
      const hue = 30 + ageFrac * 180; // amber -> cyan/blue
      const lightness = 60 - ageFrac * 20;

      ctx.beginPath();
      ctx.arc(wf.cx, wf.cy, radius, 0, Math.PI * 2);
      ctx.strokeStyle = `hsla(${hue},80%,${lightness}%,${alpha})`;
      ctx.lineWidth = 1.5 * (1 - ageFrac * 0.5);
      ctx.stroke();
    }

    // --- Mach cone draw karo jab supersonic ho ---
    if (sourceSpeed > 1.0) {
      // Mach angle = arcsin(v_sound / v_source) = arcsin(1/M)
      const machAngle = Math.asin(1 / sourceSpeed);

      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = `rgba(255,80,80,0.5)`;
      ctx.lineWidth = 1.5;

      // cone lines — source se peeche ki taraf
      const coneLen = canvasW;
      ctx.beginPath();
      ctx.moveTo(sourceX, sourceY);
      ctx.lineTo(sourceX - coneLen * Math.cos(machAngle), sourceY - coneLen * Math.sin(machAngle));
      ctx.moveTo(sourceX, sourceY);
      ctx.lineTo(sourceX - coneLen * Math.cos(machAngle), sourceY + coneLen * Math.sin(machAngle));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      // Mach number label
      ctx.font = `10px ${FONT}`;
      ctx.fillStyle = 'rgba(255,80,80,0.6)';
      ctx.textAlign = 'left';
      ctx.fillText(`Mach ${sourceSpeed.toFixed(2)}  θ=${(machAngle * 180 / Math.PI).toFixed(1)}°`, sourceX + 15, sourceY - 20);
    }

    // --- Source draw karo — bright amber dot ---
    ctx.save();
    ctx.shadowColor = ACCENT;
    ctx.shadowBlur = 15;
    const grad = ctx.createRadialGradient(sourceX, sourceY, 0, sourceX, sourceY, 8);
    grad.addColorStop(0, '#fef3c7');
    grad.addColorStop(0.5, ACCENT);
    grad.addColorStop(1, 'rgba(245,158,11,0.2)');
    ctx.beginPath();
    ctx.arc(sourceX, sourceY, 8, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();

    // source ke direction arrow
    if (sourceVx > 0) {
      ctx.beginPath();
      ctx.moveTo(sourceX + 14, sourceY);
      ctx.lineTo(sourceX + 24, sourceY);
      ctx.lineTo(sourceX + 20, sourceY - 4);
      ctx.moveTo(sourceX + 24, sourceY);
      ctx.lineTo(sourceX + 20, sourceY + 4);
      ctx.strokeStyle = `rgba(${ACCENT_RGB},0.6)`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // --- Observer dots — left and right side pe ---
    const obsY = CANVAS_HEIGHT / 2;
    const obsLeftX = 30, obsRightX = canvasW - 30;

    // observer circles
    for (const ox of [obsLeftX, obsRightX]) {
      ctx.beginPath();
      ctx.arc(ox, obsY, 5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(100,200,255,0.5)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(100,200,255,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // --- Frequency indicator — wavefronts count near observers ---
    // left observer ke paas kitne wavefronts close hain
    let leftCount = 0, rightCount = 0;
    const detectR = 30; // detection radius
    for (let i = 0; i < wavefronts.length; i++) {
      const wf = wavefronts[i];
      const age = simTime - wf.birthTime;
      const r = age * SOUND_SPEED;
      // left observer se distance
      const dLeft = Math.abs(r - Math.sqrt((wf.cx - obsLeftX) ** 2 + (wf.cy - obsY) ** 2));
      if (dLeft < detectR) leftCount++;
      // right observer se distance
      const dRight = Math.abs(r - Math.sqrt((wf.cx - obsRightX) ** 2 + (wf.cy - obsY) ** 2));
      if (dRight < detectR) rightCount++;
    }

    ctx.font = `9px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(100,200,255,0.5)';
    ctx.fillText(`f↑ (${leftCount})`, obsLeftX, obsY + 18);
    ctx.fillText(`f↓ (${rightCount})`, obsRightX, obsY + 18);

    // --- Info text ---
    ctx.font = `10px ${FONT}`;
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.textAlign = 'left';
    const regime = sourceSpeed < 1 ? 'subsonic' : sourceSpeed === 1 ? 'sonic' : 'supersonic';
    ctx.fillText(`v/c=${sourceSpeed.toFixed(2)}  ${regime}  waves=${wavefronts.length}`, 8, 16);
  }

  // --- Animation loop ---
  let lastTime = 0;
  function loop(timestamp) {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    if (lastTime === 0) lastTime = timestamp;
    const dt = Math.min((timestamp - lastTime) / 1000, 0.05); // cap at 50ms
    lastTime = timestamp;

    if (!isPaused) {
      simTime += dt;
      update(dt);
    }
    draw();
    animationId = requestAnimationFrame(loop);
  }

  // --- Intersection Observer ---
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) { lastTime = 0; animationId = requestAnimationFrame(loop); }
    else if (!isVisible && animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) { lastTime = 0; animationId = requestAnimationFrame(loop); } });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && isVisible && !animationId) { lastTime = 0; animationId = requestAnimationFrame(loop); } });
}
