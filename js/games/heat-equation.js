// ============================================================
// 2D Heat Equation — explicit Euler diffusion, paint hot/cold with mouse
// Shift+drag se insulating walls banao — full thermal simulation
// ============================================================

export function initHeatEquation() {
  const container = document.getElementById('heatEquationContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  let animationId = null, isVisible = false, canvasW = 0;

  // grid — ~100x75 cells
  const GW = 100, GH = 75;
  let temp = new Float64Array(GW * GH); // temperature 0-1
  let walls = new Uint8Array(GW * GH); // 1 = insulating wall
  let diffusivity = 0.2;

  // interaction
  let isDrawing = false;
  let drawMode = 'hot'; // 'hot', 'cold', 'wall'
  let brushSize = 3;

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
    sl.style.cssText = 'width:70px;height:4px;accent-color:#f59e0b;cursor:pointer;';
    w.appendChild(sl);
    const vl = document.createElement('span');
    vl.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:32px;";
    vl.textContent = Number(val).toFixed(step < 0.1 ? 2 : 1);
    w.appendChild(vl);
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      vl.textContent = v.toFixed(step < 0.1 ? 2 : 1);
      onChange(v);
    });
    ctrl.appendChild(w);
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

  createSlider('\u03b1', 0.05, 0.5, 0.05, diffusivity, (v) => { diffusivity = v; });
  createButton('Reset', () => { temp.fill(0); walls.fill(0); });

  // info text
  const info = document.createElement('span');
  info.style.cssText = "color:#6b6b6b;font-size:10px;font-family:'JetBrains Mono',monospace;";
  info.textContent = 'click=hot | right=cold | shift+drag=wall';
  ctrl.appendChild(info);

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

  // --- mouse interaction ---
  function getGridPos(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    // rect.width/height use karo — border include hota hai, accurate mapping ke liye
    const gx = Math.floor((cx / rect.width) * GW);
    const gy = Math.floor((cy / rect.height) * GH);
    return [Math.max(0, Math.min(GW - 1, gx)), Math.max(0, Math.min(GH - 1, gy))];
  }

  function paintAt(gx, gy, e) {
    const shift = e.shiftKey;
    const rightBtn = e.buttons === 2 || e.button === 2;

    for (let dy = -brushSize; dy <= brushSize; dy++) {
      for (let dx = -brushSize; dx <= brushSize; dx++) {
        if (dx * dx + dy * dy > brushSize * brushSize) continue;
        const nx = gx + dx, ny = gy + dy;
        if (nx < 0 || nx >= GW || ny < 0 || ny >= GH) continue;
        const idx = ny * GW + nx;

        if (shift) {
          // wall mode — insulating wall banao
          walls[idx] = 1;
          temp[idx] = 0.5; // neutral temp
        } else if (rightBtn) {
          // cold paint karo
          walls[idx] = 0;
          temp[idx] = Math.max(0, temp[idx] - 0.3);
        } else {
          // hot paint karo
          walls[idx] = 0;
          temp[idx] = Math.min(1, temp[idx] + 0.3);
        }
      }
    }
  }

  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    isDrawing = true;
    const [gx, gy] = getGridPos(e);
    paintAt(gx, gy, e);
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const [gx, gy] = getGridPos(e);
    paintAt(gx, gy, e);
  });
  canvas.addEventListener('mouseup', () => { isDrawing = false; });
  canvas.addEventListener('mouseleave', () => { isDrawing = false; });
  canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); });

  // touch support
  canvas.addEventListener('touchstart', (e) => { e.preventDefault(); isDrawing = true; const [gx, gy] = getGridPos(e); paintAt(gx, gy, e); }, { passive: false });
  canvas.addEventListener('touchmove', (e) => { e.preventDefault(); if (!isDrawing) return; const [gx, gy] = getGridPos(e); paintAt(gx, gy, e); }, { passive: false });
  canvas.addEventListener('touchend', () => { isDrawing = false; });

  // --- scratch buffer — har step mein reuse hoga ---
  const _newTemp = new Float64Array(GW * GH);

  // --- physics: explicit Euler heat diffusion ---
  function diffuseStep() {
    const newTemp = _newTemp;
    // alpha clamped for stability: alpha * dt / dx^2 <= 0.25 (2D)
    const alpha = Math.min(diffusivity, 0.24);

    for (let y = 0; y < GH; y++) {
      for (let x = 0; x < GW; x++) {
        const idx = y * GW + x;
        // walls nahi diffuse hote — insulating boundary
        if (walls[idx]) {
          newTemp[idx] = temp[idx];
          continue;
        }

        // neighbors — Neumann boundary (zero flux)
        const left = x > 0 && !walls[idx - 1] ? temp[idx - 1] : temp[idx];
        const right = x < GW - 1 && !walls[idx + 1] ? temp[idx + 1] : temp[idx];
        const up = y > 0 && !walls[idx - GW] ? temp[idx - GW] : temp[idx];
        const down = y < GH - 1 && !walls[idx + GW] ? temp[idx + GW] : temp[idx];

        // laplacian: nabla^2 T = (left + right + up + down - 4*center)
        const laplacian = left + right + up + down - 4 * temp[idx];
        newTemp[idx] = temp[idx] + alpha * laplacian;

        // clamp 0-1
        newTemp[idx] = Math.max(0, Math.min(1, newTemp[idx]));
      }
    }
    temp.set(newTemp);
  }

  // --- temperature to color ---
  function tempToColor(t) {
    // blue -> cyan -> green -> yellow -> red
    // 5-stop gradient
    let r, g, b;
    if (t < 0.25) {
      const s = t / 0.25;
      r = 0; g = Math.round(s * 180); b = Math.round(180 + s * 75);
    } else if (t < 0.5) {
      const s = (t - 0.25) / 0.25;
      r = 0; g = Math.round(180 + s * 75); b = Math.round(255 - s * 255);
    } else if (t < 0.75) {
      const s = (t - 0.5) / 0.25;
      r = Math.round(s * 255); g = 255; b = 0;
    } else {
      const s = (t - 0.75) / 0.25;
      r = 255; g = Math.round(255 - s * 255); b = 0;
    }
    return [r, g, b];
  }

  // --- offscreen canvas ek baar banao, har frame pe reuse karo ---
  const imgCanvas = document.createElement('canvas');
  imgCanvas.width = GW;
  imgCanvas.height = GH;
  const imgCtx = imgCanvas.getContext('2d');

  // --- draw using ImageData for speed ---
  function draw() {
    const imgData = imgCtx.createImageData(GW, GH);

    for (let y = 0; y < GH; y++) {
      for (let x = 0; x < GW; x++) {
        const idx = y * GW + x;
        const pi = idx * 4;

        if (walls[idx]) {
          // walls grey dikhao
          imgData.data[pi] = 60;
          imgData.data[pi + 1] = 60;
          imgData.data[pi + 2] = 60;
          imgData.data[pi + 3] = 255;
        } else {
          const t = temp[idx];
          if (t < 0.01) {
            // background — dark
            imgData.data[pi] = 17;
            imgData.data[pi + 1] = 17;
            imgData.data[pi + 2] = 17;
            imgData.data[pi + 3] = 255;
          } else {
            const [r, g, b] = tempToColor(t);
            imgData.data[pi] = r;
            imgData.data[pi + 1] = g;
            imgData.data[pi + 2] = b;
            imgData.data[pi + 3] = Math.round(100 + t * 155);
          }
        }
      }
    }

    imgCtx.putImageData(imgData, 0, 0);

    // main canvas pe scale karke draw karo
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(imgCanvas, 0, 0, canvasW, CANVAS_HEIGHT);

    // labels
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.textAlign = 'left';
    ctx.fillText('2D HEAT EQUATION', 8, 14);

    // temperature scale bar
    const barX = canvasW - 25, barY = 20, barH = 60;
    for (let i = 0; i < barH; i++) {
      const t = 1 - i / barH;
      const [r, g, b] = tempToColor(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(barX, barY + i, 12, 1);
    }
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = "8px 'JetBrains Mono',monospace";
    ctx.textAlign = 'right';
    ctx.fillText('hot', barX - 3, barY + 6);
    ctx.fillText('cold', barX - 3, barY + barH);
  }

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    // multiple diffusion steps per frame — tez dikhne ke liye
    for (let s = 0; s < 5; s++) {
      diffuseStep();
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
