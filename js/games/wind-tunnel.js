// ============================================================
// Wind Tunnel — Lattice Boltzmann Method (D2Q9) fluid simulation
// Obstacles draw karo, vortex streets dekho, view modes toggle karo
// ============================================================
export function initWindTunnel() {
  const container = document.getElementById('windTunnelContainer');
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

  // controls banao
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
    sl.style.cssText = 'width:70px;height:4px;accent-color:#f59e0b;cursor:pointer;';
    w.appendChild(sl);
    const vl = document.createElement('span');
    vl.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:32px;";
    vl.textContent = Number(val).toFixed(step < 0.01 ? 3 : step < 0.1 ? 2 : 1);
    w.appendChild(vl);
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      vl.textContent = v.toFixed(step < 0.01 ? 3 : step < 0.1 ? 2 : 1);
      onChange(v);
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

  // --- LBM parameters ---
  const NX = 200, NY = 100;
  let u0 = 0.1;   // inlet velocity — zyada kiya toh unstable hoga
  let tau = 0.6;   // relaxation time — viscosity control karta hai
  let viewMode = 0; // 0 = velocity, 1 = vorticity

  // D2Q9 lattice directions — standard 9-velocity model
  // ye fixed hai — ek cell mein 9 directions mein particles jaate hain
  const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
  const cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
  const w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];
  // opposite direction mapping — bounce-back ke liye chahiye
  const opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];

  // distribution functions — ye hi actual fluid state hai
  let f = new Float32Array(9 * NX * NY);
  let fTemp = new Float32Array(9 * NX * NY);
  let obstacle = new Uint8Array(NX * NY);
  // macroscopic quantities
  let rho = new Float32Array(NX * NY);
  let ux = new Float32Array(NX * NY);
  let uy = new Float32Array(NX * NY);
  let curl = new Float32Array(NX * NY);
  let imgData = null;
  // temporary canvas — render mein reuse karenge, har frame createElement nahi karenge
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = NX;
  tmpCanvas.height = NY;
  const tmpCtx = tmpCanvas.getContext('2d');

  // controls
  createSlider('speed', 0.02, 0.15, 0.01, u0, v => { u0 = v; });
  createSlider('tau', 0.51, 1.0, 0.01, tau, v => { tau = v; });
  const viewBtn = createButton('Vorticity', () => {
    viewMode = 1 - viewMode;
    viewBtn.textContent = viewMode === 0 ? 'Vorticity' : 'Velocity';
  });
  createButton('Clear', () => { clearObstacles(); initFlow(); });

  function idx(x, y) { return y * NX + x; }
  function fidx(q, x, y) { return (q * NY + y) * NX + x; }

  // --- circular obstacle banao ---
  function addCircle(cx0, cy0, r) {
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        const dx = x - cx0, dy = y - cy0;
        if (dx * dx + dy * dy <= r * r) {
          obstacle[idx(x, y)] = 1;
        }
      }
    }
  }

  function clearObstacles() {
    obstacle.fill(0);
  }

  // --- equilibrium distribution calculate karo ---
  // ye LBM ka core hai — f_eq(rho, u) formula
  function feq(q, rhoVal, uxVal, uyVal) {
    const cu = cx[q] * uxVal + cy[q] * uyVal;
    const u2 = uxVal * uxVal + uyVal * uyVal;
    return w[q] * rhoVal * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u2);
  }

  // --- initialize flow field ---
  function initFlow() {
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        const i = idx(x, y);
        rho[i] = 1.0;
        ux[i] = u0;
        uy[i] = 0;
        for (let q = 0; q < 9; q++) {
          f[fidx(q, x, y)] = feq(q, 1.0, u0, 0);
        }
      }
    }
  }

  // default circle obstacle rakh do — vortex street dikhega
  addCircle(NX * 0.25 | 0, NY * 0.5 | 0, NY * 0.1 | 0);
  initFlow();

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    imgData = ctx.createImageData(NX, NY);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- mouse se obstacle draw karo ---
  let isDrawing = false;
  function getGridPos(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width * NX;
    const my = (e.clientY - rect.top) / rect.height * NY;
    return [Math.floor(mx), Math.floor(my)];
  }

  function stampObstacle(gx, gy) {
    const r = 3;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy > r * r) continue;
        const nx = gx + dx, ny = gy + dy;
        if (nx > 0 && nx < NX - 1 && ny > 0 && ny < NY - 1) {
          obstacle[idx(nx, ny)] = 1;
        }
      }
    }
  }

  canvas.addEventListener('pointerdown', (e) => {
    isDrawing = true;
    const [gx, gy] = getGridPos(e);
    stampObstacle(gx, gy);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!isDrawing) return;
    const [gx, gy] = getGridPos(e);
    stampObstacle(gx, gy);
  });
  canvas.addEventListener('pointerup', () => { isDrawing = false; });
  canvas.addEventListener('pointerleave', () => { isDrawing = false; });

  // --- LBM simulation step ---
  function lbmStep() {
    const omega = 1 / tau;
    const oneMinusOmega = 1 - omega;

    // --- Collision + Streaming combined ---
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        const i = idx(x, y);
        if (obstacle[i]) continue;

        // macroscopic quantities — density aur velocity nikalo
        let rhoLocal = 0, uxLocal = 0, uyLocal = 0;
        for (let q = 0; q < 9; q++) {
          const fVal = f[fidx(q, x, y)];
          rhoLocal += fVal;
          uxLocal += cx[q] * fVal;
          uyLocal += cy[q] * fVal;
        }
        if (rhoLocal > 0) { uxLocal /= rhoLocal; uyLocal /= rhoLocal; }
        rho[i] = rhoLocal;
        ux[i] = uxLocal;
        uy[i] = uyLocal;

        // collision — BGK operator
        // f_new = f - (f - f_eq) / tau
        for (let q = 0; q < 9; q++) {
          const fEq = feq(q, rhoLocal, uxLocal, uyLocal);
          const fNew = oneMinusOmega * f[fidx(q, x, y)] + omega * fEq;
          // streaming — naye position pe bhejo
          const nx = x + cx[q];
          const ny = y + cy[q];
          if (nx >= 0 && nx < NX && ny >= 0 && ny < NY) {
            fTemp[fidx(q, nx, ny)] = fNew;
          }
        }
      }
    }

    // --- Bounce-back — obstacle wali cells pe ---
    // jo particle obstacle mein jaaye usko ulta bhej do
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        if (!obstacle[idx(x, y)]) continue;
        for (let q = 0; q < 9; q++) {
          const nx = x + cx[q];
          const ny = y + cy[q];
          if (nx >= 0 && nx < NX && ny >= 0 && ny < NY && !obstacle[idx(nx, ny)]) {
            fTemp[fidx(opp[q], x, y)] = f[fidx(q, x, y)];
          }
        }
      }
    }

    // swap buffers
    const tmp = f;
    f = fTemp;
    fTemp = tmp;

    // --- Boundary conditions ---
    // left inlet — Zou-He velocity inlet
    for (let y = 1; y < NY - 1; y++) {
      const i = idx(0, y);
      if (obstacle[i]) continue;
      rho[i] = 1.0;
      ux[i] = u0;
      uy[i] = 0;
      for (let q = 0; q < 9; q++) {
        f[fidx(q, 0, y)] = feq(q, 1.0, u0, 0);
      }
    }

    // right outlet — zero gradient
    for (let y = 1; y < NY - 1; y++) {
      for (let q = 0; q < 9; q++) {
        f[fidx(q, NX - 1, y)] = f[fidx(q, NX - 2, y)];
      }
    }

    // top/bottom walls — bounce back
    for (let x = 0; x < NX; x++) {
      for (let q = 0; q < 9; q++) {
        f[fidx(q, x, 0)] = feq(q, 1.0, u0, 0);
        f[fidx(q, x, NY - 1)] = feq(q, 1.0, u0, 0);
      }
    }

    // --- vorticity compute karo — curl of velocity ---
    for (let y = 1; y < NY - 1; y++) {
      for (let x = 1; x < NX - 1; x++) {
        curl[idx(x, y)] = (uy[idx(x + 1, y)] - uy[idx(x - 1, y)]) -
                           (ux[idx(x, y + 1)] - ux[idx(x, y - 1)]);
      }
    }
  }

  // --- colormap functions ---
  // cool to hot — neela se laal tak
  function velColor(speed) {
    const s = Math.min(1, speed / (u0 * 3));
    let r, g, b;
    if (s < 0.25) {
      const t = s / 0.25;
      r = 0; g = t * 120; b = 80 + t * 120;
    } else if (s < 0.5) {
      const t = (s - 0.25) / 0.25;
      r = 0; g = 120 + t * 135; b = 200 - t * 100;
    } else if (s < 0.75) {
      const t = (s - 0.5) / 0.25;
      r = t * 255; g = 255 - t * 80; b = 100 - t * 100;
    } else {
      const t = (s - 0.75) / 0.25;
      r = 255; g = 175 - t * 175; b = 0;
    }
    return [r | 0, g | 0, b | 0];
  }

  // vorticity — laal/neela diverging colormap
  function vortColor(v) {
    const s = Math.min(1, Math.max(-1, v * 50));
    if (s > 0) {
      const t = s;
      return [50 + 205 * t | 0, 30 * (1 - t) | 0, 30 * (1 - t) | 0];
    } else {
      const t = -s;
      return [30 * (1 - t) | 0, 30 * (1 - t) | 0, 50 + 205 * t | 0];
    }
  }

  function render() {
    if (!imgData) return;
    const d = imgData.data;
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        const i = idx(x, y);
        const p = i * 4;
        if (obstacle[i]) {
          // obstacle — grey dikhao
          d[p] = 80; d[p + 1] = 75; d[p + 2] = 70; d[p + 3] = 255;
        } else if (viewMode === 0) {
          // velocity magnitude
          const sp = Math.sqrt(ux[i] * ux[i] + uy[i] * uy[i]);
          const [r, g, b] = velColor(sp);
          d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = 255;
        } else {
          // vorticity — ghoomne ka pattern
          const [r, g, b] = vortColor(curl[i]);
          d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = 255;
        }
      }
    }

    // temporary canvas pe draw karo — reusable canvas, har frame naya nahi banate
    tmpCtx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmpCanvas, 0, 0, canvasW, CANVAS_HEIGHT);

    // label
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.4)';
    ctx.textAlign = 'left';
    ctx.fillText('WIND TUNNEL (LBM D2Q9)', 8, 14);
    ctx.textAlign = 'right';
    ctx.fillText(viewMode === 0 ? 'velocity' : 'vorticity', canvasW - 8, 14);
  }

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    // multiple steps per frame — fast convergence ke liye
    for (let i = 0; i < 10; i++) lbmStep();
    render();
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
