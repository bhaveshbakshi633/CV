// ============================================================
// Joukowski Airfoil — Conformal mapping ka visual demo
// z-plane mein circle, w = z + 1/z se airfoil shape ban jaata hai
// Potential flow streamlines dono planes mein dikhte hain
// Circle center drag karo — airfoil shape change hogi live
// Aerodynamics ka foundation — lift generate karne ka raaz yahi hai
// ============================================================

// yahi se shuru — circle banao, map karo, airfoil dekho
export function initJoukowski() {
  const container = document.getElementById('joukowskiContainer');
  if (!container) return;

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  const ACCENT_RGB = '245,158,11';
  const BG = '#111';
  const FONT = "'JetBrains Mono',monospace";

  // --- State ---
  let animationId = null, isVisible = false, canvasW = 0;
  let offsetX = -0.15;        // circle center offset x (thoda left = airfoil shape)
  let offsetY = 0.10;         // circle center offset y (upar = camber/angle of attack)
  let radius = 1.15;          // circle ka radius (> 1 taaki circle critical point cover kare)
  let circulation = 0.5;      // Kutta condition se related — lift generate karta hai
  let isDragging = false;

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
    vSpan.style.cssText = `color:#f0f0f0;font-size:11px;font-family:${FONT};min-width:32px;`;
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

  // --- Controls ---
  const oxCtrl = makeSlider('cx', -0.5, 0.1, 0.01, offsetX, (v) => { offsetX = v; });
  const oyCtrl = makeSlider('cy', -0.3, 0.3, 0.01, offsetY, (v) => { offsetY = v; });
  makeSlider('R', 1.01, 1.5, 0.01, radius, (v) => { radius = v; });
  makeSlider('Γ', 0.0, 3.0, 0.1, circulation, (v) => { circulation = v; });

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

  // --- Joukowski transform ---
  // w = z + 1/z where z = x + iy
  function joukowski(zx, zy) {
    const r2 = zx * zx + zy * zy;
    if (r2 < 0.0001) return { wx: zx, wy: zy }; // singularity avoid karo
    return {
      wx: zx + zx / r2,     // Re(z + 1/z) = x + x/(x²+y²) = x(1 + 1/r²)
      wy: zy - zy / r2      // Im(z + 1/z) = y - y/(x²+y²) = y(1 - 1/r²)
    };
  }

  // --- Coordinate transform helpers ---
  // left half = z-plane, right half = w-plane
  function zToPixel(zx, zy) {
    const halfW = canvasW / 2;
    const zScale = Math.min(halfW, CANVAS_HEIGHT) / 5;
    return {
      px: halfW / 2 + zx * zScale,
      py: CANVAS_HEIGHT / 2 - zy * zScale
    };
  }

  function wToPixel(wx, wy) {
    const halfW = canvasW / 2;
    const wScale = Math.min(halfW, CANVAS_HEIGHT) / 7; // w-plane thoda zoom out
    return {
      px: canvasW / 2 + halfW / 2 + wx * wScale,
      py: CANVAS_HEIGHT / 2 - wy * wScale
    };
  }

  function pixelToZ(px, py) {
    const halfW = canvasW / 2;
    const zScale = Math.min(halfW, CANVAS_HEIGHT) / 5;
    return {
      zx: (px - halfW / 2) / zScale,
      zy: (CANVAS_HEIGHT / 2 - py) / zScale
    };
  }

  // --- Mouse drag — circle center move karo ---
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    // sirf left half mein drag — z-plane
    if (mx < canvasW / 2) {
      isDragging = true;
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const z = pixelToZ(mx, my);
    // offset update karo — slider sync bhi karo
    offsetX = Math.max(-0.5, Math.min(0.1, z.zx));
    offsetY = Math.max(-0.3, Math.min(0.3, z.zy));
    oxCtrl.slider.value = String(offsetX);
    oxCtrl.vSpan.textContent = offsetX.toFixed(2);
    oyCtrl.slider.value = String(offsetY);
    oyCtrl.vSpan.textContent = offsetY.toFixed(2);
  });

  canvas.addEventListener('mouseup', () => { isDragging = false; });
  canvas.addEventListener('mouseleave', () => { isDragging = false; });

  // --- Draw ---
  function draw() {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, canvasW, CANVAS_HEIGHT);

    const halfW = canvasW / 2;

    // --- Divider line ---
    ctx.strokeStyle = `rgba(${ACCENT_RGB},0.15)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(halfW, 0);
    ctx.lineTo(halfW, CANVAS_HEIGHT);
    ctx.stroke();

    // --- Panel labels ---
    ctx.font = `11px ${FONT}`;
    ctx.fillStyle = `rgba(${ACCENT_RGB},0.5)`;
    ctx.textAlign = 'center';
    ctx.fillText('z-plane (circle)', halfW / 2, 18);
    ctx.fillText('w-plane (airfoil)', halfW + halfW / 2, 18);

    // --- z-plane: axes ---
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 0.5;
    const zO = zToPixel(0, 0);
    ctx.beginPath(); ctx.moveTo(0, zO.py); ctx.lineTo(halfW, zO.py); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(zO.px, 0); ctx.lineTo(zO.px, CANVAS_HEIGHT); ctx.stroke();

    // --- w-plane: axes ---
    const wO = wToPixel(0, 0);
    ctx.beginPath(); ctx.moveTo(halfW, wO.py); ctx.lineTo(canvasW, wO.py); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(wO.px, 0); ctx.lineTo(wO.px, CANVAS_HEIGHT); ctx.stroke();

    // --- Critical points ±1 in z-plane ---
    const p1 = zToPixel(1, 0), pm1 = zToPixel(-1, 0);
    ctx.fillStyle = 'rgba(255,100,100,0.5)';
    ctx.beginPath(); ctx.arc(p1.px, p1.py, 3, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(pm1.px, pm1.py, 3, 0, Math.PI * 2); ctx.fill();

    // --- Streamlines in z-plane ---
    // uniform flow + doublet + vortex around circle
    // complex potential: W(z) = U(z' + R²/z') + iΓ/(2π) * ln(z')
    // where z' = z - center
    const cx = offsetX, cy = offsetY;
    const R = radius;
    const U = 1.0; // freestream velocity
    const G = circulation; // circulation strength

    // streamlines — constant psi contours ko trace karo
    // psi = U*(y' - R²*y'/r'²) + G/(2π)*ln(r')
    // where r' = |z - center|

    const numStreamlines = 18;
    const streamStep = 0.04;
    const maxSteps = 300;

    for (let s = 0; s < numStreamlines; s++) {
      // starting y position — evenly spaced from left edge
      const startY = -2.5 + (s + 0.5) * 5.0 / numStreamlines;
      const startX = -2.5;

      // streamline trace karo — Euler method with velocity field
      let zx = startX, zy = startY;
      const zPath = []; // z-plane path
      const wPath = []; // w-plane (mapped) path

      for (let step = 0; step < maxSteps; step++) {
        // z' = z - center
        const zxp = zx - cx, zyp = zy - cy;
        const r2 = zxp * zxp + zyp * zyp;

        // circle ke andar skip karo
        if (r2 < R * R * 0.95) break;
        if (r2 < 0.01) break;

        zPath.push(zToPixel(zx, zy));

        // Joukowski mapped point
        const mapped = joukowski(zx, zy);
        wPath.push(wToPixel(mapped.wx, mapped.wy));

        // velocity field: dW/dz = U(1 - R²/z'²) + iΓ/(2πz')
        // u - iv = dW/dz
        const r4 = r2 * r2;
        // R²/z'² = R²*(zxp - i*zyp)² / |z'|^4
        const r2zx2 = R * R * (zxp * zxp - zyp * zyp) / r4;
        const r2zxy = R * R * 2 * zxp * zyp / r4;

        const vortR = G / (2 * Math.PI * r2);

        // u = U*(1 - R²(x'²-y'²)/r⁴) + Γy'/(2πr²)
        const u = U * (1 - r2zx2) + vortR * zyp;
        // v = U*(2R²x'y'/r⁴) - Γx'/(2πr²)
        const vv = U * r2zxy - vortR * zxp;

        const speed = Math.sqrt(u * u + vv * vv);
        if (speed < 0.001) break;

        // normalize aur step
        zx += (u / speed) * streamStep;
        zy += (vv / speed) * streamStep;

        // canvas bounds check
        if (zx < -3 || zx > 3 || zy < -3 || zy > 3) break;
      }

      // z-plane streamline draw karo
      if (zPath.length > 1) {
        ctx.beginPath();
        ctx.moveTo(zPath[0].px, zPath[0].py);
        for (let i = 1; i < zPath.length; i++) {
          ctx.lineTo(zPath[i].px, zPath[i].py);
        }
        ctx.strokeStyle = `rgba(100,200,255,0.15)`;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }

      // w-plane streamline draw karo
      if (wPath.length > 1) {
        ctx.beginPath();
        ctx.moveTo(wPath[0].px, wPath[0].py);
        for (let i = 1; i < wPath.length; i++) {
          // jump detect karo — agar points bahut door hain toh line mat banao
          const dx = wPath[i].px - wPath[i - 1].px;
          const dy = wPath[i].py - wPath[i - 1].py;
          if (dx * dx + dy * dy > 2500) { ctx.moveTo(wPath[i].px, wPath[i].py); }
          else ctx.lineTo(wPath[i].px, wPath[i].py);
        }
        ctx.strokeStyle = `rgba(100,200,255,0.15)`;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }

    // --- Circle draw karo z-plane mein ---
    ctx.beginPath();
    const cPixel = zToPixel(cx, cy);
    const rPixel = zToPixel(cx + R, cy);
    const pixelR = Math.abs(rPixel.px - cPixel.px);
    ctx.arc(cPixel.px, cPixel.py, pixelR, 0, Math.PI * 2);
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.stroke();

    // circle center
    ctx.beginPath();
    ctx.arc(cPixel.px, cPixel.py, 3, 0, Math.PI * 2);
    ctx.fillStyle = ACCENT;
    ctx.fill();

    // --- Airfoil draw karo w-plane mein ---
    // circle ke boundary points ko map karo
    ctx.beginPath();
    const airfoilPoints = 200;
    let firstPoint = true;
    for (let i = 0; i <= airfoilPoints; i++) {
      const theta = (i / airfoilPoints) * 2 * Math.PI;
      const zx = cx + R * Math.cos(theta);
      const zy = cy + R * Math.sin(theta);
      const w = joukowski(zx, zy);
      const p = wToPixel(w.wx, w.wy);
      if (firstPoint) { ctx.moveTo(p.px, p.py); firstPoint = false; }
      else ctx.lineTo(p.px, p.py);
    }
    ctx.closePath();
    ctx.fillStyle = `rgba(${ACCENT_RGB},0.08)`;
    ctx.fill();
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.stroke();

    // trailing edge marker — w-plane mein z=1 ka map
    const te = joukowski(1, 0);
    const teP = wToPixel(te.wx, te.wy);
    ctx.beginPath();
    ctx.arc(teP.px, teP.py, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,100,100,0.7)';
    ctx.fill();

    // --- Info text ---
    ctx.font = `10px ${FONT}`;
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.textAlign = 'left';
    ctx.fillText(`center=(${cx.toFixed(2)},${cy.toFixed(2)})  R=${R.toFixed(2)}  Γ=${G.toFixed(1)}`, 8, CANVAS_HEIGHT - 8);
    ctx.fillText('drag circle center in z-plane', 8, CANVAS_HEIGHT - 22);
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
