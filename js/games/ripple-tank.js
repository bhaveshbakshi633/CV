// ============================================================
// Ripple Tank — 2D wave interference simulation
// Click to place sources, shift+drag for walls, double-slit diffraction
// ============================================================
export function initRippleTank() {
  const container = document.getElementById('rippleTankContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  let animationId = null, isVisible = false, canvasW = 0;

  // --- DOM setup ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#000;border:1px solid rgba(245,158,11,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls - safe DOM construction
  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  const lblStyle = "color:#ccc;font:12px 'JetBrains Mono',monospace";
  const btnStyle = "background:#333;color:#ccc;border:1px solid #555;padding:3px 8px;border-radius:4px;cursor:pointer;font:11px 'JetBrains Mono',monospace";

  function mkSlider(label, id, min, max, val) {
    const lbl = document.createElement('label');
    lbl.style.cssText = lblStyle;
    lbl.textContent = label + ' ';
    const inp = document.createElement('input');
    inp.type = 'range'; inp.min = min; inp.max = max; inp.value = val; inp.id = id;
    inp.style.cssText = 'width:80px;vertical-align:middle';
    lbl.appendChild(inp);
    return lbl;
  }
  function mkBtn(text, id) {
    const b = document.createElement('button');
    b.textContent = text; b.id = id; b.style.cssText = btnStyle;
    return b;
  }
  ctrl.appendChild(mkSlider('Freq', 'rtFreq', 2, 20, 8));
  ctrl.appendChild(mkSlider('Speed', 'rtSpeed', 5, 30, 15));
  ctrl.appendChild(mkSlider('Damping', 'rtDamp', 90, 100, 98));
  ctrl.appendChild(mkBtn('Two Sources', 'rtPreset1'));
  ctrl.appendChild(mkBtn('Double Slit', 'rtPreset2'));
  ctrl.appendChild(mkBtn('Clear', 'rtClear'));
  container.appendChild(ctrl);

  // wave grid — 2D finite difference
  const GW = 200, GH = 150;
  let u = new Float32Array(GW * GH);
  let uPrev = new Float32Array(GW * GH);
  let walls = new Uint8Array(GW * GH);
  let sources = [];
  let t = 0;
  let imgData = null;

  function idx(x, y) { return y * GW + x; }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    imgData = ctx.createImageData(GW, GH);
  }
  resize();
  window.addEventListener('resize', resize);

  function clearSim() {
    u.fill(0); uPrev.fill(0); walls.fill(0);
    sources = []; t = 0;
  }

  function toGrid(ex, ey) {
    const rect = canvas.getBoundingClientRect();
    const gx = Math.floor((ex - rect.left) / rect.width * GW);
    const gy = Math.floor((ey - rect.top) / rect.height * GH);
    return [Math.max(0, Math.min(GW - 1, gx)), Math.max(0, Math.min(GH - 1, gy))];
  }

  // interaction
  let drawing = false;
  canvas.addEventListener('pointerdown', e => {
    const [gx, gy] = toGrid(e.clientX, e.clientY);
    if (e.shiftKey) {
      drawing = true;
      // pointer capture le lo — canvas ke bahar bhi events aayein
      canvas.setPointerCapture(e.pointerId);
      walls[idx(gx, gy)] = 1;
    } else {
      sources.push({ gx, gy });
    }
  });
  canvas.addEventListener('pointermove', e => {
    if (!drawing) return;
    const [gx, gy] = toGrid(e.clientX, e.clientY);
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const nx = gx + dx, ny = gy + dy;
        if (nx >= 0 && nx < GW && ny >= 0 && ny < GH) walls[idx(nx, ny)] = 1;
      }
  });
  canvas.addEventListener('pointerup', e => {
    drawing = false;
    if (canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }
  });

  // presets
  document.getElementById('rtClear').addEventListener('click', clearSim);
  document.getElementById('rtPreset1').addEventListener('click', () => {
    clearSim();
    sources.push({ gx: GW / 3 | 0, gy: GH / 2 | 0 });
    sources.push({ gx: 2 * GW / 3 | 0, gy: GH / 2 | 0 });
  });
  document.getElementById('rtPreset2').addEventListener('click', () => {
    clearSim();
    sources.push({ gx: 20, gy: GH / 2 | 0 });
    const wx = GW / 2 | 0;
    const slitW = 3, gap = 15;
    for (let y = 0; y < GH; y++) {
      const dy = Math.abs(y - GH / 2);
      if (dy < gap / 2 - slitW || (dy > gap / 2 + slitW && dy < gap + slitW) || dy > gap + 2 * slitW) {
        for (let dx = -1; dx <= 1; dx++) walls[idx(wx + dx, y)] = 1;
      }
    }
  });

  // scratch buffer ek baar allocate karo — har step mein reuse hoga
  let uNext = new Float32Array(GW * GH);

  function step() {
    const freq = +document.getElementById('rtFreq').value || 8;
    const c = (+document.getElementById('rtSpeed').value || 15) / 15;
    const damp = (+document.getElementById('rtDamp').value || 98) / 100;
    const c2 = c * c;

    for (const s of sources) {
      u[idx(s.gx, s.gy)] = Math.sin(t * freq * 0.05) * 2;
    }

    uNext.fill(0);
    for (let y = 1; y < GH - 1; y++) {
      for (let x = 1; x < GW - 1; x++) {
        const i = idx(x, y);
        if (walls[i]) { uNext[i] = 0; continue; }
        const lap = u[idx(x+1,y)] + u[idx(x-1,y)] + u[idx(x,y+1)] + u[idx(x,y-1)] - 4 * u[i];
        uNext[i] = (2 * u[i] - uPrev[i] + c2 * lap) * damp;
      }
    }
    uPrev.set(u);
    u.set(uNext);
    t++;
  }

  function render() {
    if (!imgData) return;
    const d = imgData.data;
    for (let y = 0; y < GH; y++) {
      for (let x = 0; x < GW; x++) {
        const i = idx(x, y);
        const p = (y * GW + x) * 4;
        if (walls[i]) {
          d[p] = 100; d[p+1] = 100; d[p+2] = 100; d[p+3] = 255;
        } else {
          const v = u[i];
          const intensity = Math.min(1, Math.abs(v));
          if (v > 0) {
            d[p] = 50 + 205 * intensity | 0; d[p+1] = 50 + 100 * (1 - intensity) | 0; d[p+2] = 50;
          } else {
            d[p] = 50; d[p+1] = 50 + 100 * (1 - intensity) | 0; d[p+2] = 50 + 205 * intensity | 0;
          }
          d[p+3] = 255;
        }
      }
    }
    // temporary canvas pe imgData put karo, fir scale karke main canvas pe draw karo
    if (!render._tmpCanvas) {
      render._tmpCanvas = document.createElement('canvas');
      render._tmpCanvas.width = GW;
      render._tmpCanvas.height = GH;
    }
    const tmpCtx = render._tmpCanvas.getContext('2d');
    tmpCtx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(render._tmpCanvas, 0, 0, canvasW, CANVAS_HEIGHT);

    const sx = canvasW / GW, sy = CANVAS_HEIGHT / GH;
    ctx.fillStyle = ACCENT;
    for (const s of sources) {
      ctx.beginPath();
      ctx.arc(s.gx * sx + sx/2, s.gy * sy + sy/2, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    for (let i = 0; i < 3; i++) step();
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
