// ============================================================
// Lorentz Transform — Minkowski spacetime diagram ka interactive demo
// x-t axes, light cones 45° pe, boosted x'-t' axes velocity se
// Events place karo clicking se, velocity change karo slider se
// Length contraction aur time dilation apne aap dikh jaayegi
// x' = γ(x - vt), t' = γ(t - vx/c²) — special relativity ka dil
// ============================================================

// yahi se shuru — spacetime pe events rakh aur Lorentz boost dekh
export function initLorentzTransform() {
  const container = document.getElementById('lorentzTransformContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  const ACCENT_RGB = '245,158,11';
  const BG = '#111';
  const FONT = "'JetBrains Mono',monospace";

  // --- State ---
  let animationId = null, isVisible = false, canvasW = 0;
  let velocity = 0.0;           // v/c — -0.99 to 0.99
  let events = [];              // [{x, t, color}] — spacetime coordinates (rest frame)
  let showWorldlines = true;    // worldlines dikhao ya nahi
  let dragIdx = -1;             // kaunsa event drag ho raha hai

  // coordinate system — origin canvas ke center mein
  let originX = 0, originY = 0;
  let scale = 1;                // pixels per unit

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
    slider.style.cssText = `width:120px;height:4px;accent-color:${ACCENT};cursor:pointer;`;
    wrap.appendChild(slider);
    const vSpan = document.createElement('span');
    vSpan.style.cssText = `color:#f0f0f0;font-size:11px;font-family:${FONT};min-width:40px;`;
    vSpan.textContent = Number(val).toFixed(2) + 'c';
    wrap.appendChild(vSpan);
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      vSpan.textContent = v.toFixed(2) + 'c';
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

  function makeToggle(label, checked, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;align-items:center;gap:4px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = `color:#6b6b6b;font-size:11px;font-family:${FONT};`;
    lbl.textContent = label;
    wrap.appendChild(lbl);
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = checked;
    cb.style.cssText = `accent-color:${ACCENT};cursor:pointer;`;
    cb.addEventListener('change', () => onChange(cb.checked));
    wrap.appendChild(cb);
    ctrl.appendChild(wrap);
    return cb;
  }

  // --- Controls ---
  const velCtrl = makeSlider('v', -0.99, 0.99, 0.01, velocity, (v) => { velocity = v; });
  makeBtn('Clear Events', () => { events = []; });
  makeToggle('Worldlines', showWorldlines, (v) => { showWorldlines = v; });

  // gamma label
  const gammaSpan = document.createElement('span');
  gammaSpan.style.cssText = `font-size:11px;font-family:${FONT};color:rgba(${ACCENT_RGB},0.7);`;
  ctrl.appendChild(gammaSpan);

  // --- Resize ---
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    originX = canvasW / 2;
    originY = CANVAS_HEIGHT / 2;
    scale = Math.min(canvasW, CANVAS_HEIGHT) / 8; // 4 units each side
  }
  resize();
  window.addEventListener('resize', resize);

  // --- Coordinate transforms ---
  // spacetime (x,t) to pixel
  function toPixel(x, t) {
    return { px: originX + x * scale, py: originY - t * scale };
  }

  // pixel to spacetime
  function toSpacetime(px, py) {
    return { x: (px - originX) / scale, t: (originY - py) / scale };
  }

  // Lorentz boost — (x,t) se (x',t') nikaal do
  function lorentzBoost(x, t, v) {
    const g = 1 / Math.sqrt(1 - v * v); // gamma
    return {
      xp: g * (x - v * t),
      tp: g * (t - v * x)
    };
  }

  // --- Event colors — unique color per event ---
  const EVENT_COLORS = [
    '#ff6b6b', '#22d3ee', '#a78bfa', '#34d399', '#fb923c',
    '#f472b6', '#60a5fa', '#fbbf24', '#a3e635', '#e879f9'
  ];

  // --- Mouse handlers ---
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // check karo koi existing event ke paas hai kya — drag karne ke liye
    for (let i = 0; i < events.length; i++) {
      const p = toPixel(events[i].x, events[i].t);
      const dx = mx - p.px, dy = my - p.py;
      if (dx * dx + dy * dy < 144) { // 12px radius
        dragIdx = i;
        return;
      }
    }

    // naya event place karo
    const st = toSpacetime(mx, my);
    events.push({
      x: st.x,
      t: st.t,
      color: EVENT_COLORS[events.length % EVENT_COLORS.length]
    });
  });

  canvas.addEventListener('mousemove', (e) => {
    if (dragIdx < 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const st = toSpacetime(mx, my);
    events[dragIdx].x = st.x;
    events[dragIdx].t = st.t;
  });

  canvas.addEventListener('mouseup', () => { dragIdx = -1; });
  canvas.addEventListener('mouseleave', () => { dragIdx = -1; });

  // --- Draw ---
  function draw() {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, canvasW, CANVAS_HEIGHT);

    const v = velocity;
    const gamma = 1 / Math.sqrt(1 - v * v);
    gammaSpan.textContent = `γ=${gamma.toFixed(3)}`;

    // --- Light cones — 45° lines ---
    ctx.strokeStyle = 'rgba(255,255,100,0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    const extent = 5; // units
    // future light cone
    let p1 = toPixel(-extent, extent), p2 = toPixel(extent, extent);
    ctx.beginPath(); ctx.moveTo(toPixel(0, 0).px, toPixel(0, 0).py);
    ctx.lineTo(p1.px, p1.py);
    ctx.moveTo(toPixel(0, 0).px, toPixel(0, 0).py);
    ctx.lineTo(p2.px, p2.py);
    // past light cone
    p1 = toPixel(-extent, -extent); p2 = toPixel(extent, -extent);
    ctx.moveTo(toPixel(0, 0).px, toPixel(0, 0).py);
    ctx.lineTo(p1.px, p1.py);
    ctx.moveTo(toPixel(0, 0).px, toPixel(0, 0).py);
    ctx.lineTo(p2.px, p2.py);
    ctx.stroke();
    ctx.setLineDash([]);

    // light cone fill — halka yellow
    ctx.fillStyle = 'rgba(255,255,100,0.03)';
    // future cone
    ctx.beginPath();
    ctx.moveTo(toPixel(0, 0).px, toPixel(0, 0).py);
    ctx.lineTo(toPixel(-extent, extent).px, toPixel(-extent, extent).py);
    ctx.lineTo(toPixel(extent, extent).px, toPixel(extent, extent).py);
    ctx.closePath(); ctx.fill();
    // past cone
    ctx.beginPath();
    ctx.moveTo(toPixel(0, 0).px, toPixel(0, 0).py);
    ctx.lineTo(toPixel(-extent, -extent).px, toPixel(-extent, -extent).py);
    ctx.lineTo(toPixel(extent, -extent).px, toPixel(extent, -extent).py);
    ctx.closePath(); ctx.fill();

    // --- Rest frame axes (x, t) — white ---
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    // x-axis
    ctx.beginPath();
    ctx.moveTo(toPixel(-extent, 0).px, toPixel(-extent, 0).py);
    ctx.lineTo(toPixel(extent, 0).px, toPixel(extent, 0).py);
    ctx.stroke();
    // t-axis
    ctx.beginPath();
    ctx.moveTo(toPixel(0, -extent).px, toPixel(0, -extent).py);
    ctx.lineTo(toPixel(0, extent).px, toPixel(0, extent).py);
    ctx.stroke();

    // axis labels
    ctx.font = `11px ${FONT}`;
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.textAlign = 'left';
    ctx.fillText('x', toPixel(extent, 0).px + 5, toPixel(extent, 0).py + 4);
    ctx.textAlign = 'center';
    ctx.fillText('t', toPixel(0, extent).px, toPixel(0, extent).py - 8);

    // grid ticks — subtle dots
    ctx.fillStyle = 'rgba(255,255,255,0.06)';
    for (let i = -4; i <= 4; i++) {
      for (let j = -4; j <= 4; j++) {
        if (i === 0 && j === 0) continue;
        const p = toPixel(i, j);
        ctx.fillRect(p.px - 1, p.py - 1, 2, 2);
      }
    }

    // --- Boosted frame axes (x', t') — amber ---
    if (Math.abs(v) > 0.001) {
      ctx.strokeStyle = `rgba(${ACCENT_RGB},0.5)`;
      ctx.lineWidth = 1.5;

      // t' axis: x' = 0, so x = v*t — slope = 1/v in (x,t) space
      // draw line through origin with slope t/x = 1/v, i.e., x = v*t
      const tpEnd = extent;
      ctx.beginPath();
      ctx.moveTo(toPixel(-v * tpEnd, -tpEnd).px, toPixel(-v * tpEnd, -tpEnd).py);
      ctx.lineTo(toPixel(v * tpEnd, tpEnd).px, toPixel(v * tpEnd, tpEnd).py);
      ctx.stroke();

      // x' axis: t' = 0, so t = v*x — slope = v in (x,t) space
      const xpEnd = extent;
      ctx.beginPath();
      ctx.moveTo(toPixel(-xpEnd, -v * xpEnd).px, toPixel(-xpEnd, -v * xpEnd).py);
      ctx.lineTo(toPixel(xpEnd, v * xpEnd).px, toPixel(xpEnd, v * xpEnd).py);
      ctx.stroke();

      // boosted axis labels
      ctx.fillStyle = `rgba(${ACCENT_RGB},0.6)`;
      ctx.font = `10px ${FONT}`;
      ctx.textAlign = 'left';
      ctx.fillText("x'", toPixel(xpEnd, v * xpEnd).px + 5, toPixel(xpEnd, v * xpEnd).py - 5);
      ctx.fillText("t'", toPixel(v * tpEnd, tpEnd).px + 5, toPixel(v * tpEnd, tpEnd).py - 5);

      // boosted grid — hyperbolic calibration ticks
      ctx.fillStyle = `rgba(${ACCENT_RGB},0.15)`;
      for (let i = -4; i <= 4; i++) {
        if (i === 0) continue;
        // x' = i, t' = 0 ka rest frame position
        const xr = gamma * i;
        const tr = gamma * v * i;
        const p = toPixel(xr, tr);
        ctx.beginPath();
        ctx.arc(p.px, p.py, 2, 0, Math.PI * 2);
        ctx.fill();

        // x' = 0, t' = i ka rest frame position
        const xr2 = gamma * v * i;
        const tr2 = gamma * i;
        const p2 = toPixel(xr2, tr2);
        ctx.beginPath();
        ctx.arc(p2.px, p2.py, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // --- Events draw karo ---
    for (let i = 0; i < events.length; i++) {
      const ev = events[i];

      // worldlines — vertical lines through event (constant x)
      if (showWorldlines) {
        ctx.strokeStyle = ev.color + '25';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(toPixel(ev.x, -extent).px, toPixel(ev.x, -extent).py);
        ctx.lineTo(toPixel(ev.x, extent).px, toPixel(ev.x, extent).py);
        ctx.stroke();
      }

      // rest frame mein event ka position
      const pRest = toPixel(ev.x, ev.t);

      // event dot — rest frame
      ctx.save();
      ctx.shadowColor = ev.color;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(pRest.px, pRest.py, 6, 0, Math.PI * 2);
      ctx.fillStyle = ev.color;
      ctx.fill();
      ctx.restore();

      // boosted position bhi dikhao jab v ≠ 0
      if (Math.abs(v) > 0.001) {
        const boosted = lorentzBoost(ev.x, ev.t, v);
        // boosted event dikhao — boosted frame ke coordinates ko rest frame mein plot karo
        // actual pixel position remains same (event is same point in spacetime)
        // but we show the primed coordinates as text

        // coordinate label — rest frame
        ctx.font = `9px ${FONT}`;
        ctx.fillStyle = ev.color + 'aa';
        ctx.textAlign = 'left';
        ctx.fillText(`(${ev.x.toFixed(1)},${ev.t.toFixed(1)})`, pRest.px + 9, pRest.py - 3);
        // boosted coordinates
        ctx.fillStyle = `rgba(${ACCENT_RGB},0.6)`;
        ctx.fillText(`(${boosted.xp.toFixed(1)},${boosted.tp.toFixed(1)})'`, pRest.px + 9, pRest.py + 10);

        // dashed line from event to boosted axes showing projection
        ctx.setLineDash([2, 3]);
        ctx.strokeStyle = ev.color + '30';
        ctx.lineWidth = 0.8;

        // projection onto x' axis (t'=0 line: t = v*x)
        // project event onto x' axis: find point where t' = 0 on the line from event
        const xpProj = boosted.xp; // x' coordinate
        // that point in rest frame: x = γ*xp, t = γ*v*xp
        const projX = gamma * xpProj;
        const projT = gamma * v * xpProj;
        const pProj = toPixel(projX, projT);
        ctx.beginPath();
        ctx.moveTo(pRest.px, pRest.py);
        ctx.lineTo(pProj.px, pProj.py);
        ctx.stroke();

        // projection onto t' axis (x'=0 line: x = v*t)
        const tpProj = boosted.tp;
        const proj2X = gamma * v * tpProj;
        const proj2T = gamma * tpProj;
        const pProj2 = toPixel(proj2X, proj2T);
        ctx.beginPath();
        ctx.moveTo(pRest.px, pRest.py);
        ctx.lineTo(pProj2.px, pProj2.py);
        ctx.stroke();

        ctx.setLineDash([]);
      } else {
        // v = 0 — sirf rest coordinates dikhao
        ctx.font = `9px ${FONT}`;
        ctx.fillStyle = ev.color + 'aa';
        ctx.textAlign = 'left';
        ctx.fillText(`(${ev.x.toFixed(1)},${ev.t.toFixed(1)})`, pRest.px + 9, pRest.py + 4);
      }
    }

    // --- Origin dot ---
    ctx.beginPath();
    ctx.arc(originX, originY, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.fill();

    // --- Info text ---
    ctx.font = `10px ${FONT}`;
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.textAlign = 'left';
    ctx.fillText(`v=${v.toFixed(2)}c  γ=${gamma.toFixed(2)}  events=${events.length}`, 8, 16);
    ctx.fillText('click to place events, drag to move', 8, 28);

    // light cone labels
    ctx.fillStyle = 'rgba(255,255,100,0.15)';
    ctx.font = `9px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.fillText('future', originX, toPixel(0, extent).py + 15);
    ctx.fillText('past', originX, toPixel(0, -extent).py - 8);
  }

  // --- Animation loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
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
