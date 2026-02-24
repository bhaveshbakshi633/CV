// ============================================================
// Ising Model — 2D lattice of spins, Metropolis algorithm
// Phase transition at T_c ~ 2.269 — ferromagnet to paramagnet
// ============================================================

export function initIsingModel() {
  const container = document.getElementById('isingModelContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  let animationId = null, isVisible = false, canvasW = 0;

  // lattice size
  const L = 100;
  let spins = new Int8Array(L * L); // +1 ya -1

  // physics parameters
  let temperature = 2.5; // T — critical point ~2.269
  let extField = 0; // external magnetic field
  let J = 1.0; // coupling constant

  // stats tracking
  let magnetization = 0;
  let energy = 0;

  // --- lattice random initialize karo ---
  function randomize() {
    for (let i = 0; i < L * L; i++) {
      spins[i] = Math.random() < 0.5 ? 1 : -1;
    }
    computeStats();
  }

  function computeStats() {
    magnetization = 0;
    energy = 0;
    for (let y = 0; y < L; y++) {
      for (let x = 0; x < L; x++) {
        const s = spins[y * L + x];
        magnetization += s;
        // right aur down neighbors se interaction energy
        const right = spins[y * L + ((x + 1) % L)];
        const down = spins[((y + 1) % L) * L + x];
        energy -= J * s * (right + down);
        energy -= extField * s;
      }
    }
  }

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

  function createSlider(label, min, max, step, val, onChange) {
    const w = document.createElement('div');
    w.style.cssText = 'display:flex;align-items:center;gap:5px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#6b6b6b;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    lbl.textContent = label;
    w.appendChild(lbl);
    const sl = document.createElement('input');
    sl.type = 'range'; sl.min = String(min); sl.max = String(max); sl.step = String(step); sl.value = String(val);
    sl.style.cssText = 'width:80px;height:4px;accent-color:#f59e0b;cursor:pointer;';
    w.appendChild(sl);
    const vl = document.createElement('span');
    vl.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:36px;";
    vl.textContent = Number(val).toFixed(step < 0.1 ? 2 : 1);
    w.appendChild(vl);
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      vl.textContent = v.toFixed(step < 0.1 ? 2 : 1);
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

  const tempLabel = createSlider('T', 0.1, 5.0, 0.05, temperature, (v) => { temperature = v; });
  createSlider('h', -2.0, 2.0, 0.1, extField, (v) => { extField = v; });
  createButton('Random', randomize);
  createButton('All +1', () => { spins.fill(1); computeStats(); });
  createButton('All -1', () => { spins.fill(-1); computeStats(); });

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

  // --- Metropolis algorithm: single spin flip ---
  function metropolisStep() {
    const x = Math.floor(Math.random() * L);
    const y = Math.floor(Math.random() * L);
    const idx = y * L + x;
    const s = spins[idx];

    // neighbors sum (periodic boundary conditions)
    const neighbors =
      spins[y * L + ((x + 1) % L)] +
      spins[y * L + ((x - 1 + L) % L)] +
      spins[((y + 1) % L) * L + x] +
      spins[((y - 1 + L) % L) * L + x];

    // energy change agar spin flip kare: deltaE = 2*J*s*sum_neighbors + 2*h*s
    const deltaE = 2 * J * s * neighbors + 2 * extField * s;

    // Metropolis acceptance: flip agar deltaE < 0 ya random < exp(-deltaE/T)
    if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperature)) {
      spins[idx] = -s;
      magnetization += -2 * s;
      energy += deltaE;
    }
  }

  // --- offscreen canvas ek baar banao, har frame pe reuse karo ---
  const imgCanvas = document.createElement('canvas');
  imgCanvas.width = L;
  imgCanvas.height = L;
  const imgCtx = imgCanvas.getContext('2d');

  // --- draw using ImageData for speed ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // lattice square — centered
    const latticeSize = Math.min(canvasW - 20, CANVAS_HEIGHT - 60);
    const offsetX = (canvasW - latticeSize) / 2;
    const offsetY = 10;
    const cellSize = latticeSize / L;

    // ImageData se fast rendering
    const imgData = imgCtx.createImageData(L, L);

    for (let i = 0; i < L * L; i++) {
      const pi = i * 4;
      if (spins[i] === 1) {
        // +1 = amber
        imgData.data[pi] = 245;
        imgData.data[pi + 1] = 158;
        imgData.data[pi + 2] = 11;
        imgData.data[pi + 3] = 255;
      } else {
        // -1 = dark
        imgData.data[pi] = 20;
        imgData.data[pi + 1] = 20;
        imgData.data[pi + 2] = 30;
        imgData.data[pi + 3] = 255;
      }
    }

    imgCtx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(imgCanvas, offsetX, offsetY, latticeSize, latticeSize);

    // border
    ctx.strokeStyle = 'rgba(245,158,11,0.2)';
    ctx.lineWidth = 1;
    ctx.strokeRect(offsetX, offsetY, latticeSize, latticeSize);

    // stats dikhao — bottom mein
    const statsY = offsetY + latticeSize + 18;
    const N2 = L * L;
    const m = magnetization / N2; // per site magnetization
    const e = energy / N2; // per site energy

    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.fillText('ISING MODEL', 8, 14);

    ctx.fillStyle = 'rgba(245,158,11,0.5)';
    ctx.fillText('m=' + m.toFixed(3), offsetX, statsY);
    ctx.fillText('E/N=' + e.toFixed(2), offsetX + 100, statsY);

    // T_c indicator
    ctx.fillStyle = 'rgba(34,211,238,0.4)';
    ctx.fillText('T\u2095\u2248' + (2.269).toFixed(3), offsetX + 220, statsY);

    // magnetization bar — visual indicator
    const barX = offsetX + latticeSize + 8;
    const barW = 12;
    if (barX + barW < canvasW) {
      const barH = latticeSize;
      const midY = offsetY + barH / 2;
      // background
      ctx.fillStyle = 'rgba(20,20,30,0.5)';
      ctx.fillRect(barX, offsetY, barW, barH);
      // magnetization level
      const magH = Math.abs(m) * barH / 2;
      if (m > 0) {
        ctx.fillStyle = 'rgba(245,158,11,0.5)';
        ctx.fillRect(barX, midY - magH, barW, magH);
      } else {
        ctx.fillStyle = 'rgba(100,100,200,0.5)';
        ctx.fillRect(barX, midY, barW, magH);
      }
      // center line
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.beginPath();
      ctx.moveTo(barX, midY);
      ctx.lineTo(barX + barW, midY);
      ctx.stroke();
    }
  }

  // --- init ---
  randomize();

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    // bahut saare flips per frame — L*L flips = 1 Monte Carlo sweep
    const flipsPerFrame = L * L * 2; // 2 sweeps per frame
    for (let i = 0; i < flipsPerFrame; i++) {
      metropolisStep();
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
