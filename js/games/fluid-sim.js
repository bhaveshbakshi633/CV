// ============================================================
// Navier-Stokes Fluid Simulator — Jos Stam's Stable Fluids
// Mouse move karo, vibrant colorful smoke swirl hoga real-time
// Advection, diffusion, pressure projection, vorticity confinement
// Crown jewel simulation — full CFD on a grid
// ============================================================

// yahi se sab shuru hota hai — container dhundho, grid banao, fluid simulate karo
export function initFluidSim() {
  const container = document.getElementById('fluidSimContainer');
  if (!container) {
    console.warn('fluidSimContainer nahi mila bhai, fluid sim skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const N = 128; // grid resolution — 128x128 cells, sweet spot for performance vs detail
  const SIZE = (N + 2) * (N + 2); // padded grid — boundary cells ke liye +2
  const ITER = 20; // Gauss-Seidel iterations — zyada = accurate, slower
  const ACCENT = '#4a9eff';

  // --- Simulation State ---
  // velocity fields — u (horizontal), v (vertical)
  let u = new Float32Array(SIZE);
  let v = new Float32Array(SIZE);
  let uPrev = new Float32Array(SIZE);
  let vPrev = new Float32Array(SIZE);

  // density fields — RGB channels alag alag track kar rahe hain vibrant colors ke liye
  let densR = new Float32Array(SIZE);
  let densG = new Float32Array(SIZE);
  let densB = new Float32Array(SIZE);
  let densRPrev = new Float32Array(SIZE);
  let densGPrev = new Float32Array(SIZE);
  let densBPrev = new Float32Array(SIZE);

  // vorticity field — curl store karne ke liye
  let curl = new Float32Array(SIZE);

  // --- Tunable Parameters ---
  let viscosity = 0.0003; // kinematic viscosity — fluid kitna thick hai
  let diffusion = 0.0001; // dye diffusion rate — color kitna spread hota hai
  let vorticityStrength = 15.0; // vorticity confinement — swirl kitna tight rahe
  let dt = 0.1; // timestep — stable hai Jos Stam ke method mein
  let rainbowMode = true; // rainbow injection vs single color

  // --- Interaction State ---
  let mouseX = 0, mouseY = 0;
  let prevMouseX = 0, prevMouseY = 0;
  let mouseDown = false;
  let mouseInCanvas = false;

  // --- Animation State ---
  let animationId = null;
  let isVisible = false;

  // ============================
  // CORE SOLVER — Jos Stam Style
  // ============================

  // helper: 2D index nikalo — (i, j) se flat array index
  function IX(i, j) {
    return i + (N + 2) * j;
  }

  // source values add karo field mein — external forces / dye injection
  function addSource(x, s) {
    for (let i = 0; i < SIZE; i++) {
      x[i] += dt * s[i];
    }
  }

  // boundary conditions set karo — walls pe velocity reflect hoti hai
  // b=1: horizontal velocity (left/right walls pe negate)
  // b=2: vertical velocity (top/bottom walls pe negate)
  // b=0: scalar field (density — simply copy)
  function setBnd(b, x) {
    for (let i = 1; i <= N; i++) {
      // left/right walls
      x[IX(0, i)]     = b === 1 ? -x[IX(1, i)] : x[IX(1, i)];
      x[IX(N + 1, i)] = b === 1 ? -x[IX(N, i)] : x[IX(N, i)];
      // top/bottom walls
      x[IX(i, 0)]     = b === 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
      x[IX(i, N + 1)] = b === 2 ? -x[IX(i, N)] : x[IX(i, N)];
    }
    // corners — average of neighbors
    x[IX(0, 0)]         = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N + 1)]     = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)]     = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
  }

  // Gauss-Seidel relaxation — diffusion solve karta hai iteratively
  // implicit method hai — unconditionally stable, isliye Jos Stam ne use kiya
  function diffuse(b, x, x0, diff) {
    const a = dt * diff * N * N;
    const denom = 1 + 4 * a;
    for (let k = 0; k < ITER; k++) {
      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          x[IX(i, j)] = (x0[IX(i, j)] + a * (
            x[IX(i - 1, j)] + x[IX(i + 1, j)] +
            x[IX(i, j - 1)] + x[IX(i, j + 1)]
          )) / denom;
        }
      }
      setBnd(b, x);
    }
  }

  // semi-Lagrangian advection — particle ko backwards trace karo, bilinear interpolate karo
  // ye method unconditionally stable hai — CFL condition ki zaroorat nahi
  function advect(b, d, d0, u_field, v_field) {
    const dt0 = dt * N;
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        // particle ki previous position trace karo — backwards in time
        let x = i - dt0 * u_field[IX(i, j)];
        let y = j - dt0 * v_field[IX(i, j)];

        // clamp to grid bounds — boundary ke andar rakho
        if (x < 0.5) x = 0.5;
        if (x > N + 0.5) x = N + 0.5;
        if (y < 0.5) y = 0.5;
        if (y > N + 0.5) y = N + 0.5;

        // bilinear interpolation — 4 neighboring cells se value nikalo
        const i0 = Math.floor(x);
        const i1 = i0 + 1;
        const j0 = Math.floor(y);
        const j1 = j0 + 1;

        const s1 = x - i0;
        const s0 = 1 - s1;
        const t1 = y - j0;
        const t0 = 1 - t1;

        d[IX(i, j)] =
          s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
          s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
      }
    }
    setBnd(b, d);
  }

  // pressure projection — Helmholtz-Hodge decomposition
  // divergence-free velocity field banata hai — incompressibility enforce karta hai
  // ye Navier-Stokes ka sabse important step hai
  function project(u_field, v_field, p, div) {
    const h = 1.0 / N;

    // divergence calculate karo — velocity field ka
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        div[IX(i, j)] = -0.5 * h * (
          u_field[IX(i + 1, j)] - u_field[IX(i - 1, j)] +
          v_field[IX(i, j + 1)] - v_field[IX(i, j - 1)]
        );
        p[IX(i, j)] = 0;
      }
    }
    setBnd(0, div);
    setBnd(0, p);

    // pressure Poisson equation solve karo — Gauss-Seidel se
    for (let k = 0; k < ITER; k++) {
      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          p[IX(i, j)] = (div[IX(i, j)] +
            p[IX(i - 1, j)] + p[IX(i + 1, j)] +
            p[IX(i, j - 1)] + p[IX(i, j + 1)]
          ) / 4;
        }
      }
      setBnd(0, p);
    }

    // pressure gradient subtract karo velocity se — divergence-free bana do
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        u_field[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N;
        v_field[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N;
      }
    }
    setBnd(1, u_field);
    setBnd(2, v_field);
  }

  // vorticity confinement — numerical damping counter karta hai
  // curl calculate karo, fir vortex ko strengthen karo
  // isse fluid ke swirls tight aur defined rehte hain
  function vorticityConfinement(u_field, v_field) {
    // pehle curl (vorticity) calculate karo har cell pe
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        // curl = dv/dx - du/dy
        curl[IX(i, j)] =
          0.5 * (v_field[IX(i + 1, j)] - v_field[IX(i - 1, j)]) -
          0.5 * (u_field[IX(i, j + 1)] - u_field[IX(i, j - 1)]);
      }
    }

    // ab vorticity confinement force apply karo
    for (let j = 2; j < N; j++) {
      for (let i = 2; i < N; i++) {
        // curl magnitude ka gradient nikalo — direction pata chalega
        const absCurl = Math.abs(curl[IX(i, j)]);
        // gradient of |curl| — normalized direction vector
        let dx = Math.abs(curl[IX(i + 1, j)]) - Math.abs(curl[IX(i - 1, j)]);
        let dy = Math.abs(curl[IX(i, j + 1)]) - Math.abs(curl[IX(i, j - 1)]);
        const len = Math.sqrt(dx * dx + dy * dy) + 1e-5;
        dx /= len;
        dy /= len;

        // force = epsilon * (N x omega) — cross product 2D mein
        u_field[IX(i, j)] += dt * vorticityStrength * (dy * curl[IX(i, j)]);
        v_field[IX(i, j)] -= dt * vorticityStrength * (dx * curl[IX(i, j)]);
      }
    }
  }

  // full velocity step — diffuse, advect, project with vorticity
  function velStep() {
    addSource(u, uPrev);
    addSource(v, vPrev);

    // vorticity confinement apply karo — projection se pehle
    vorticityConfinement(u, v);

    // diffuse velocity — viscosity ke hisaab se
    let tmp;
    tmp = uPrev; uPrev = u; u = tmp;
    tmp = vPrev; vPrev = v; v = tmp;
    diffuse(1, u, uPrev, viscosity);
    diffuse(2, v, vPrev, viscosity);

    // project — divergence-free banao diffusion ke baad
    project(u, v, uPrev, vPrev);

    // advect velocity — self-advection
    tmp = uPrev; uPrev = u; u = tmp;
    tmp = vPrev; vPrev = v; v = tmp;
    advect(1, u, uPrev, uPrev, vPrev);
    advect(2, v, vPrev, uPrev, vPrev);

    // project again — advection ke baad bhi divergence-free chahiye
    project(u, v, uPrev, vPrev);

    // source arrays clear karo next frame ke liye
    uPrev.fill(0);
    vPrev.fill(0);
  }

  // full density step — diffuse aur advect for each RGB channel
  function densStep() {
    addSource(densR, densRPrev);
    addSource(densG, densGPrev);
    addSource(densB, densBPrev);

    let tmp;
    // diffuse R
    tmp = densRPrev; densRPrev = densR; densR = tmp;
    diffuse(0, densR, densRPrev, diffusion);
    // diffuse G
    tmp = densGPrev; densGPrev = densG; densG = tmp;
    diffuse(0, densG, densGPrev, diffusion);
    // diffuse B
    tmp = densBPrev; densBPrev = densB; densB = tmp;
    diffuse(0, densB, densBPrev, diffusion);

    // advect RGB with velocity field
    tmp = densRPrev; densRPrev = densR; densR = tmp;
    advect(0, densR, densRPrev, u, v);
    tmp = densGPrev; densGPrev = densG; densG = tmp;
    advect(0, densG, densGPrev, u, v);
    tmp = densBPrev; densBPrev = densB; densB = tmp;
    advect(0, densB, densBPrev, u, v);

    // source arrays clear karo
    densRPrev.fill(0);
    densGPrev.fill(0);
    densBPrev.fill(0);
  }

  // sab kuch reset karo — tabula rasa
  function clearAll() {
    u.fill(0); v.fill(0);
    uPrev.fill(0); vPrev.fill(0);
    densR.fill(0); densG.fill(0); densB.fill(0);
    densRPrev.fill(0); densGPrev.fill(0); densBPrev.fill(0);
    curl.fill(0);
  }

  // ============================
  // DOM STRUCTURE
  // ============================

  // existing children preserve karo (game-header, game-desc, details wagairah)
  const existingChildren = Array.from(container.children);

  // canvas banao — fluid yahan render hoga
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;border-radius:8px;cursor:crosshair;display:block;';
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // offscreen canvas for pixel manipulation — main canvas pe scale karenge
  // grid size pe kaam karenge, fir stretch karenge canvas pe — performance boost
  const offCanvas = document.createElement('canvas');
  offCanvas.width = N;
  offCanvas.height = N;
  const offCtx = offCanvas.getContext('2d');
  const imageData = offCtx.createImageData(N, N);
  const pixels = imageData.data;

  // ============================
  // CONTROLS — dark theme, inline CSS
  // ============================
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
    'padding:10px',
    'background:#1a1a2e',
    'border-radius:6px',
    'border:1px solid rgba(74,158,255,0.15)',
  ].join(';');
  container.appendChild(controlsDiv);

  // slider banane ka helper — dark themed, JetBrains Mono
  function createSlider(label, min, max, step, value, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;flex-direction:column;gap:2px;min-width:120px;';

    const labelEl = document.createElement('label');
    labelEl.style.cssText = 'color:#8892a4;font-size:11px;font-family:"JetBrains Mono",monospace;';
    labelEl.textContent = label + ': ' + value;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = value;
    slider.style.cssText = 'width:100%;accent-color:' + ACCENT + ';cursor:pointer;height:4px;';

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      labelEl.textContent = label + ': ' + val;
      onChange(val);
    });

    wrap.appendChild(labelEl);
    wrap.appendChild(slider);
    controlsDiv.appendChild(wrap);
    return slider;
  }

  // button banane ka helper
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'background:rgba(74,158,255,0.1)',
      'color:' + ACCENT,
      'border:1px solid rgba(74,158,255,0.3)',
      'border-radius:4px',
      'padding:4px 12px',
      'cursor:pointer',
      'font-family:"JetBrains Mono",monospace',
      'font-size:11px',
      'transition:all 0.2s',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(74,158,255,0.25)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(74,158,255,0.1)';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // toggle button banane ka helper
  function createToggle(text, initialState, onToggle) {
    const btn = document.createElement('button');
    let state = initialState;

    function updateStyle() {
      btn.style.cssText = [
        'background:' + (state ? 'rgba(74,158,255,0.3)' : 'rgba(74,158,255,0.05)'),
        'color:' + (state ? '#fff' : '#8892a4'),
        'border:1px solid ' + (state ? ACCENT : 'rgba(74,158,255,0.2)'),
        'border-radius:4px',
        'padding:4px 12px',
        'cursor:pointer',
        'font-family:"JetBrains Mono",monospace',
        'font-size:11px',
        'transition:all 0.2s',
      ].join(';');
      btn.textContent = text + (state ? ' ON' : ' OFF');
    }
    updateStyle();

    btn.addEventListener('click', () => {
      state = !state;
      updateStyle();
      onToggle(state);
    });
    controlsDiv.appendChild(btn);
    return btn;
  }

  // controls banao
  createSlider('Viscosity', 0.0001, 0.001, 0.0001, viscosity, (val) => { viscosity = val; });
  createSlider('Diffusion', 0.00001, 0.001, 0.00001, diffusion, (val) => { diffusion = val; });
  createSlider('Vorticity', 0, 30, 0.5, vorticityStrength, (val) => { vorticityStrength = val; });
  createButton('Clear', clearAll);
  createToggle('Rainbow', rainbowMode, (val) => { rainbowMode = val; });

  // ============================
  // DPR-AWARE RESIZE
  // ============================
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // ============================
  // MOUSE / TOUCH INTERACTION
  // ============================

  // HSL to RGB conversion — rainbow colors ke liye
  function hslToRgb(h, s, l) {
    h = h % 360;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;
    let r, g, b;
    if (h < 60)       { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else               { r = c; g = 0; b = x; }
    return [(r + m), (g + m), (b + m)];
  }

  // mouse position se grid coordinates nikalo
  function getGridCoords(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = (clientX - rect.left) / rect.width;
    const scaleY = (clientY - rect.top) / rect.height;
    // grid coordinates — 1 to N range mein map karo
    const gi = Math.floor(scaleX * N) + 1;
    const gj = Math.floor(scaleY * N) + 1;
    return [
      Math.max(1, Math.min(N, gi)),
      Math.max(1, Math.min(N, gj)),
    ];
  }

  // mouse/touch move pe velocity aur dye inject karo
  function injectFluid(gridI, gridJ, velX, velY) {
    // velocity inject karo — mouse speed ke proportional
    const force = 5.0;
    const radius = 3; // injection radius — thoda spread karke inject karo

    // injection color decide karo — rainbow ya single
    let r, g, b;
    if (rainbowMode) {
      // time-based hue cycling — har moment naya color
      const hue = (performance.now() * 0.05) % 360;
      const rgb = hslToRgb(hue, 1.0, 0.55);
      r = rgb[0]; g = rgb[1]; b = rgb[2];
    } else {
      // single accent color — neon blue
      r = 0.29; g = 0.62; b = 1.0;
    }

    // radius ke andar sab cells pe inject karo — gaussian-ish falloff
    for (let dj = -radius; dj <= radius; dj++) {
      for (let di = -radius; di <= radius; di++) {
        const dist2 = di * di + dj * dj;
        if (dist2 > radius * radius) continue;

        const ci = gridI + di;
        const cj = gridJ + dj;
        if (ci < 1 || ci > N || cj < 1 || cj > N) continue;

        const idx = IX(ci, cj);
        // gaussian falloff — center pe zyada, edges pe kam
        const falloff = Math.exp(-dist2 / (radius * 0.8));

        // velocity inject karo
        uPrev[idx] += velX * force * falloff;
        vPrev[idx] += velY * force * falloff;

        // dye inject karo — RGB channels mein
        const dyeAmount = 150.0 * falloff;
        densRPrev[idx] += r * dyeAmount;
        densGPrev[idx] += g * dyeAmount;
        densBPrev[idx] += b * dyeAmount;
      }
    }
  }

  function handleMove(clientX, clientY) {
    const [gi, gj] = getGridCoords(clientX, clientY);
    const rect = canvas.getBoundingClientRect();

    // velocity = mouse movement direction, normalized to grid scale
    const velX = (clientX - prevMouseX) / rect.width * N;
    const velY = (clientY - prevMouseY) / rect.height * N;

    injectFluid(gi, gj, velX, velY);

    prevMouseX = clientX;
    prevMouseY = clientY;
    mouseX = clientX;
    mouseY = clientY;
  }

  // mouse events
  canvas.addEventListener('mousedown', (e) => {
    mouseDown = true;
    const rect = canvas.getBoundingClientRect();
    prevMouseX = e.clientX;
    prevMouseY = e.clientY;
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  canvas.addEventListener('mousemove', (e) => {
    mouseInCanvas = true;
    if (mouseDown) {
      handleMove(e.clientX, e.clientY);
    } else {
      // mouse hover pe bhi halka sa inject karo — responsive feel ke liye
      const [gi, gj] = getGridCoords(e.clientX, e.clientY);
      const rect = canvas.getBoundingClientRect();
      const velX = (e.clientX - prevMouseX) / rect.width * N * 0.3;
      const velY = (e.clientY - prevMouseY) / rect.height * N * 0.3;
      if (Math.abs(velX) > 0.1 || Math.abs(velY) > 0.1) {
        injectFluid(gi, gj, velX, velY);
      }
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    }
  });

  canvas.addEventListener('mouseup', () => { mouseDown = false; });
  canvas.addEventListener('mouseleave', () => { mouseDown = false; mouseInCanvas = false; });

  // touch events — mobile support
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    mouseDown = true;
    prevMouseX = touch.clientX;
    prevMouseY = touch.clientY;
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    handleMove(touch.clientX, touch.clientY);
  }, { passive: false });

  canvas.addEventListener('touchend', () => { mouseDown = false; });

  // ============================
  // RENDERING
  // ============================

  function render() {
    const cw = container.clientWidth;

    // density field se pixels banao — offscreen canvas pe
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const idx = IX(i + 1, j + 1); // grid indices 1-indexed hain
        const pIdx = (j * N + i) * 4; // pixel buffer index

        // RGB density values ko 0-255 mein map karo
        let r = densR[idx] * 3.0;
        let g = densG[idx] * 3.0;
        let b = densB[idx] * 3.0;

        // brightness boost — low density pe bhi thoda glow dikhe
        const totalDens = r + g + b;
        if (totalDens > 0.01) {
          // subtle glow effect — density ke saath brightness non-linearly badhao
          const boost = 1.0 + Math.min(totalDens * 0.02, 2.0);
          r *= boost;
          g *= boost;
          b *= boost;
        }

        // clamp to 0-255
        pixels[pIdx]     = Math.min(255, Math.max(0, r | 0)); // R
        pixels[pIdx + 1] = Math.min(255, Math.max(0, g | 0)); // G
        pixels[pIdx + 2] = Math.min(255, Math.max(0, b | 0)); // B
        pixels[pIdx + 3] = 255; // alpha — fully opaque
      }
    }

    // offscreen canvas pe ImageData put karo
    offCtx.putImageData(imageData, 0, 0);

    // main canvas pe draw karo — scaled to full size, smooth interpolation
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, cw, CANVAS_HEIGHT);

    // imageSmoothingEnabled ON rakhna — fluid smooth dikhna chahiye
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(offCanvas, 0, 0, N, N, 0, 0, cw, CANVAS_HEIGHT);
  }

  // ============================
  // ANIMATION LOOP
  // ============================

  // natural motion ke liye — jab user touch nahi kar raha tab bhi thoda movement rahe
  let ambientTime = 0;

  function loop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    // density slowly decay karo — nahi toh screen bhar jaayega color se
    for (let i = 0; i < SIZE; i++) {
      densR[i] *= 0.995;
      densG[i] *= 0.995;
      densB[i] *= 0.995;
    }

    // simulation step — pehle velocity, fir density
    velStep();
    densStep();

    // render karo
    render();

    animationId = requestAnimationFrame(loop);
  }

  // ============================
  // INTERSECTION OBSERVER — lazy animation, battery friendly
  // ============================
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) loop();
    else if (!isVisible && animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }, { threshold: 0.1 });
  obs.observe(container);
  // lab resume: restart loop when focus released
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });
}
