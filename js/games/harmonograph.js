// ============================================================
// Harmonograph — do damped pendulums se Lissajous-type spiraling patterns
// x(t) = A1*sin(f1*t+p1)*e^(-d1*t) + A2*sin(f2*t+p2)*e^(-d2*t)
// y(t) similar formula with f3,f4 — rainbow trail ke saath
// ============================================================
export function initHarmonograph() {
  const container = document.getElementById('harmonographContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  let animationId = null, isVisible = false, canvasW = 0;

  // --- DOM setup ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(245,158,11,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls
  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  function createSlider(label, min, max, step, val, onChange) {
    const w = document.createElement('div');
    w.style.cssText = 'display:flex;align-items:center;gap:5px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#6b6b6b;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    lbl.textContent = label;
    w.appendChild(lbl);
    const sl = document.createElement('input');
    sl.type = 'range'; sl.min = String(min); sl.max = String(max); sl.step = String(step); sl.value = String(val);
    sl.style.cssText = 'width:60px;height:4px;accent-color:#f59e0b;cursor:pointer;';
    w.appendChild(sl);
    const vl = document.createElement('span');
    vl.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:28px;";
    vl.textContent = Number(val).toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0);
    w.appendChild(vl);
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      vl.textContent = v.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0);
      onChange(v);
      needsRedraw = true;
    });
    ctrl.appendChild(w);
    return sl;
  }

  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = "padding:5px 12px;font-size:11px;border-radius:6px;cursor:pointer;background:rgba(245,158,11,0.1);color:#b0b0b0;border:1px solid rgba(245,158,11,0.25);font-family:'JetBrains Mono',monospace;transition:all 0.2s ease;";
    btn.addEventListener('mouseenter', () => { btn.style.background = 'rgba(245,158,11,0.25)'; btn.style.color = '#e0e0e0'; });
    btn.addEventListener('mouseleave', () => { btn.style.background = 'rgba(245,158,11,0.1)'; btn.style.color = '#b0b0b0'; });
    btn.addEventListener('click', onClick);
    ctrl.appendChild(btn);
    return btn;
  }

  // --- harmonograph parameters ---
  // do pendulums ka combination — x aur y dono mein alag alag
  let f1 = 2, f2 = 3, f3 = 3, f4 = 2; // frequencies
  let p1 = 0, p2 = Math.PI / 4, p3 = 0, p4 = Math.PI / 2; // phases
  let d1 = 0.004, d2 = 0.003, d3 = 0.004, d4 = 0.003; // damping
  let A1 = 1, A2 = 1, A3 = 1, A4 = 1; // amplitudes

  // drawing state
  let t = 0;                  // current time parameter
  let tMax = 0;               // kitna draw ho chuka hai
  const dt = 0.02;            // time increment per step
  const TOTAL_T = 300;        // max time — itna draw karenge
  const POINTS_PER_FRAME = 50; // har frame mein kitne naye points
  let needsRedraw = false;    // parameter change pe pura redraw
  let trailPoints = [];       // pre-computed points store karo
  const MAX_POINTS = 15000;   // max stored points

  // sliders — user frequency ratios adjust kar sake
  const slF1 = createSlider('f1', 1, 8, 0.1, f1, v => { f1 = v; });
  const slF2 = createSlider('f2', 1, 8, 0.1, f2, v => { f2 = v; });
  const slF3 = createSlider('f3', 1, 8, 0.1, f3, v => { f3 = v; });
  const slF4 = createSlider('f4', 1, 8, 0.1, f4, v => { f4 = v; });
  const slDamp = createSlider('damp', 0.001, 0.02, 0.001, d1, v => { d1 = d2 = d3 = d4 = v; });
  createButton('Randomize', randomize);
  createButton('Clear', clearDrawing);

  // --- coordinate function ---
  // ye actual harmonograph math hai
  function getPoint(t) {
    const x = A1 * Math.sin(f1 * t + p1) * Math.exp(-d1 * t)
            + A2 * Math.sin(f2 * t + p2) * Math.exp(-d2 * t);
    const y = A3 * Math.sin(f3 * t + p3) * Math.exp(-d3 * t)
            + A4 * Math.sin(f4 * t + p4) * Math.exp(-d4 * t);
    return [x, y];
  }

  // sim coordinates ko canvas pe map karo
  function simToCanvas(sx, sy) {
    const scale = Math.min(canvasW, CANVAS_HEIGHT) * 0.42;
    const cx = canvasW / 2 + sx * scale;
    const cy = CANVAS_HEIGHT / 2 + sy * scale;
    return [cx, cy];
  }

  // --- rainbow color — t ke hisaab se ---
  function hslToRgb(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;
    let r = 0, g = 0, b = 0;
    if (h < 60)       { r = c; g = x; }
    else if (h < 120) { r = x; g = c; }
    else if (h < 180) { g = c; b = x; }
    else if (h < 240) { g = x; b = c; }
    else if (h < 300) { r = x; b = c; }
    else              { r = c; b = x; }
    return `rgb(${(r+m)*255|0},${(g+m)*255|0},${(b+m)*255|0})`;
  }

  function getColor(t, tTotal) {
    // hue 0 se 360 tak cycle karega
    const hue = (t / tTotal * 720) % 360;
    return hslToRgb(hue, 0.8, 0.55);
  }

  // --- aesthetically pleasing random parameters ---
  function randomize() {
    // frequency ratios — integer ya simple fractions acche lagte hain
    const niceFreqs = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7];
    f1 = niceFreqs[Math.random() * niceFreqs.length | 0];
    f2 = niceFreqs[Math.random() * niceFreqs.length | 0];
    f3 = niceFreqs[Math.random() * niceFreqs.length | 0];
    f4 = niceFreqs[Math.random() * niceFreqs.length | 0];
    // phase random karo — 0 se 2π
    p1 = Math.random() * Math.PI * 2;
    p2 = Math.random() * Math.PI * 2;
    p3 = Math.random() * Math.PI * 2;
    p4 = Math.random() * Math.PI * 2;
    // damping — light variation
    const baseDamp = 0.002 + Math.random() * 0.008;
    d1 = baseDamp * (0.8 + Math.random() * 0.4);
    d2 = baseDamp * (0.8 + Math.random() * 0.4);
    d3 = baseDamp * (0.8 + Math.random() * 0.4);
    d4 = baseDamp * (0.8 + Math.random() * 0.4);
    // sliders update karo
    slF1.value = f1; slF1.dispatchEvent(new Event('input'));
    slF2.value = f2; slF2.dispatchEvent(new Event('input'));
    slF3.value = f3; slF3.dispatchEvent(new Event('input'));
    slF4.value = f4; slF4.dispatchEvent(new Event('input'));
    slDamp.value = d1; slDamp.dispatchEvent(new Event('input'));
    clearDrawing();
  }

  function clearDrawing() {
    t = 0;
    tMax = 0;
    trailPoints = [];
    needsRedraw = false;
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // resize pe pura redraw chahiye — points toh stored hain
    needsRedraw = true;
  }
  resize();
  window.addEventListener('resize', resize);

  // --- render ---
  function fullRedraw() {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvasW, CANVAS_HEIGHT);
    // sab stored points phir se draw karo
    if (trailPoints.length < 2) return;
    for (let i = 1; i < trailPoints.length; i++) {
      const pt = trailPoints[i];
      const prev = trailPoints[i - 1];
      const [cx1, cy1] = simToCanvas(prev.x, prev.y);
      const [cx2, cy2] = simToCanvas(pt.x, pt.y);
      ctx.beginPath();
      ctx.moveTo(cx1, cy1);
      ctx.lineTo(cx2, cy2);
      ctx.strokeStyle = getColor(pt.t, TOTAL_T);
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.7;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    needsRedraw = false;
  }

  function drawNewSegments(startIdx) {
    // sirf naye points draw karo — incremental drawing
    for (let i = startIdx; i < trailPoints.length; i++) {
      if (i === 0) continue;
      const pt = trailPoints[i];
      const prev = trailPoints[i - 1];
      const [cx1, cy1] = simToCanvas(prev.x, prev.y);
      const [cx2, cy2] = simToCanvas(pt.x, pt.y);
      ctx.beginPath();
      ctx.moveTo(cx1, cy1);
      ctx.lineTo(cx2, cy2);
      ctx.strokeStyle = getColor(pt.t, TOTAL_T);
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.7;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function render() {
    if (needsRedraw) {
      fullRedraw();
      return;
    }
    // labels draw karo — halka se, background ke upar
    // pehle ek chhota corner label
  }

  function drawLabels() {
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'left';
    ctx.fillText('HARMONOGRAPH', 8, 14);
    ctx.textAlign = 'right';
    const pct = Math.min(100, (tMax / TOTAL_T * 100)).toFixed(0);
    ctx.fillText(pct + '%', canvasW - 8, 14);
  }

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    if (needsRedraw) {
      fullRedraw();
      drawLabels();
    } else if (tMax < TOTAL_T) {
      // naye points generate karo
      const oldLen = trailPoints.length;
      for (let i = 0; i < POINTS_PER_FRAME; i++) {
        if (tMax >= TOTAL_T) break;
        const [x, y] = getPoint(tMax);
        trailPoints.push({ x, y, t: tMax });
        tMax += dt;
        // agar bahut zyada points ho gaye toh purane hatao
        if (trailPoints.length > MAX_POINTS) {
          trailPoints.shift();
          // shift ke baad full redraw chahiye
          needsRedraw = true;
          break;
        }
      }
      if (needsRedraw) {
        fullRedraw();
      } else {
        drawNewSegments(oldLen);
      }
      drawLabels();
    }

    // current pen position dikhao — amber dot
    if (tMax < TOTAL_T && trailPoints.length > 0) {
      const last = trailPoints[trailPoints.length - 1];
      const [cx, cy] = simToCanvas(last.x, last.y);
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = ACCENT;
      ctx.shadowColor = ACCENT;
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

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
