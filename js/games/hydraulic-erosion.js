// ============================================================
// Hydraulic Erosion — terrain pe paani girata hai, mitti kaat-ta hai
// Procedural heightmap + droplet simulation, erosion ∝ speed × slope
// ============================================================
export function initHydraulicErosion() {
  const container = document.getElementById('hydraulicErosionContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  let animationId = null, isVisible = false, canvasW = 0;

  // --- DOM saaf karo aur naya setup ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(245,158,11,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls — sirf createElement use karo, safe DOM construction
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

  // --- terrain grid ---
  const GW = 200, GH = 150;
  let heightmap = new Float32Array(GW * GH);
  let waterMap = new Float32Array(GW * GH); // paani ka level track karo
  let imgData = null;
  // temporary canvas — render mein reuse karenge, har frame createElement nahi karenge
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = GW;
  tmpCanvas.height = GH;
  const tmpCtx = tmpCanvas.getContext('2d');

  // simulation parameters
  let rainRate = 80;       // boondein per frame
  let erosionRate = 0.3;   // kitna mitti kata jaye
  let depositionRate = 0.3;
  let evaporation = 0.02;
  let dropletLife = 60;    // ek boond kitne steps tak jeeti hai
  let totalDrops = 0;

  // sliders banao
  const slRain = createSlider('rain', 10, 200, 10, rainRate, v => { rainRate = v; });
  const slErosion = createSlider('erosion', 0.05, 1.0, 0.05, erosionRate, v => { erosionRate = v; });
  createButton('Reset', () => { generateTerrain(); waterMap.fill(0); totalDrops = 0; });

  // --- Perlin-jaisi noise — layered sine waves se banate hain ---
  // proper Perlin nahi likh rahe, sine layers kaafi acchi dikhti hain
  function noise2D(x, y) {
    // multiple octaves of sine waves — pseudo-random feel
    let val = 0;
    val += Math.sin(x * 0.05 + 1.3) * Math.cos(y * 0.07 + 0.8) * 0.5;
    val += Math.sin(x * 0.12 + 3.1) * Math.cos(y * 0.11 + 2.4) * 0.25;
    val += Math.sin(x * 0.23 + 5.7) * Math.cos(y * 0.19 + 4.1) * 0.125;
    val += Math.sin(x * 0.41 + 7.3) * Math.cos(y * 0.37 + 6.2) * 0.0625;
    // ek aur layer random direction mein
    val += Math.sin((x + y) * 0.08 + 2.7) * 0.15;
    val += Math.cos((x - y) * 0.13 + 4.5) * 0.1;
    return val;
  }

  // hash function taaki different terrains ban sakein
  let seed = Math.random() * 1000;
  function seededNoise(x, y) {
    return noise2D(x + seed, y + seed * 0.7);
  }

  function generateTerrain() {
    seed = Math.random() * 1000;
    for (let y = 0; y < GH; y++) {
      for (let x = 0; x < GW; x++) {
        let h = seededNoise(x, y);
        // edges pe neeche lao — island jaisa dikhega
        const ex = (x / GW - 0.5) * 2;
        const ey = (y / GH - 0.5) * 2;
        const edgeDist = 1 - Math.sqrt(ex * ex + ey * ey) * 0.8;
        h = h * Math.max(0, edgeDist);
        // normalize range mein lao
        heightmap[y * GW + x] = h * 0.5 + 0.5;
      }
    }
    // normalize 0-1 mein
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < GW * GH; i++) {
      if (heightmap[i] < mn) mn = heightmap[i];
      if (heightmap[i] > mx) mx = heightmap[i];
    }
    const range = mx - mn || 1;
    for (let i = 0; i < GW * GH; i++) {
      heightmap[i] = (heightmap[i] - mn) / range;
    }
    waterMap.fill(0);
  }
  generateTerrain();

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

  // --- gradient calculate karo — nabla h ---
  function getGradient(x, y) {
    // bilinear gradient — smooth flow ke liye zaroori hai
    const ix = Math.floor(x), iy = Math.floor(y);
    if (ix <= 0 || ix >= GW - 2 || iy <= 0 || iy >= GH - 2) return [0, 0];
    const fx = x - ix, fy = y - iy;
    // height values chaaon taraf se
    const h00 = heightmap[iy * GW + ix];
    const h10 = heightmap[iy * GW + ix + 1];
    const h01 = heightmap[(iy + 1) * GW + ix];
    const h11 = heightmap[(iy + 1) * GW + ix + 1];
    // bilinear interpolation se gradient nikalo
    const gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy;
    const gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx;
    return [gx, gy];
  }

  function getHeight(x, y) {
    const ix = Math.floor(x), iy = Math.floor(y);
    if (ix < 0 || ix >= GW - 1 || iy < 0 || iy >= GH - 1) return 0;
    const fx = x - ix, fy = y - iy;
    const h00 = heightmap[iy * GW + ix];
    const h10 = heightmap[iy * GW + ix + 1];
    const h01 = heightmap[(iy + 1) * GW + ix];
    const h11 = heightmap[(iy + 1) * GW + ix + 1];
    return h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy;
  }

  // heightmap modify karo — ek cell ke around deposit/erode karo
  function modifyTerrain(x, y, amount) {
    const ix = Math.floor(x), iy = Math.floor(y);
    if (ix < 1 || ix >= GW - 1 || iy < 1 || iy >= GH - 1) return;
    const fx = x - ix, fy = y - iy;
    // bilinear weights — jahan droplet hai uske paas zyada effect
    const w00 = (1 - fx) * (1 - fy);
    const w10 = fx * (1 - fy);
    const w01 = (1 - fx) * fy;
    const w11 = fx * fy;
    heightmap[iy * GW + ix] += amount * w00;
    heightmap[iy * GW + ix + 1] += amount * w10;
    heightmap[(iy + 1) * GW + ix] += amount * w01;
    heightmap[(iy + 1) * GW + ix + 1] += amount * w11;
  }

  // --- ek droplet simulate karo ---
  function simulateDroplet() {
    // random position se girata hai paani
    let x = Math.random() * (GW - 4) + 2;
    let y = Math.random() * (GH - 4) + 2;
    let dx = 0, dy = 0; // direction
    let speed = 0;
    let sediment = 0;   // kitni mitti utha rakhi hai
    let water = 1;       // paani ka volume
    const inertia = 0.3; // purani direction kitni yaad rakhe
    const capacity = 8;  // max sediment carry kar sakta hai
    const minSlope = 0.01;

    for (let step = 0; step < dropletLife; step++) {
      const ix = Math.floor(x), iy = Math.floor(y);
      if (ix < 1 || ix >= GW - 2 || iy < 1 || iy >= GH - 2) break;

      // gradient nikalo — paani neeche ki taraf behega
      const [gx, gy] = getGradient(x, y);
      // direction update — inertia + gradient
      dx = dx * inertia - gx * (1 - inertia);
      dy = dy * inertia - gy * (1 - inertia);
      // normalize direction
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len < 0.0001) {
        // agar flat area hai toh random direction mein bhej do
        const angle = Math.random() * Math.PI * 2;
        dx = Math.cos(angle); dy = Math.sin(angle);
      } else {
        dx /= len; dy /= len;
      }

      // naya position
      const nx = x + dx;
      const ny = y + dy;

      // height difference check karo
      const hOld = getHeight(x, y);
      const hNew = getHeight(nx, ny);
      const deltaH = hNew - hOld;

      // sediment capacity — speed aur slope pe depend karta hai
      const cappacity = Math.max(Math.abs(deltaH), minSlope) * speed * water * capacity;

      if (deltaH > 0) {
        // upar chadh raha hai — sediment deposit karo
        const toDrop = Math.min(sediment, deltaH);
        modifyTerrain(x, y, toDrop * depositionRate);
        sediment -= toDrop * depositionRate;
      } else if (sediment > cappacity) {
        // capacity se zyada hai — deposit karo
        const toDrop = (sediment - cappacity) * depositionRate;
        modifyTerrain(x, y, toDrop);
        sediment -= toDrop;
      } else {
        // erosion karo — mitti uthaao
        const toErode = Math.min((cappacity - sediment) * erosionRate, -deltaH);
        modifyTerrain(x, y, -toErode);
        sediment += toErode;
      }

      // speed update — gravity se accelerate, friction se slow
      speed = Math.sqrt(Math.max(0, speed * speed - deltaH * 4));

      // water evaporate hota hai
      water *= (1 - evaporation);

      // water map mein mark karo — visualization ke liye
      if (ix >= 0 && ix < GW && iy >= 0 && iy < GH) {
        waterMap[iy * GW + ix] = Math.min(1, waterMap[iy * GW + ix] + 0.15);
      }

      x = nx; y = ny;
    }
  }

  // --- step function — har frame mein kuch droplets simulate karo ---
  function step() {
    // water map slowly fade karo
    for (let i = 0; i < GW * GH; i++) {
      waterMap[i] *= 0.97;
    }
    // naye droplets girao
    for (let i = 0; i < rainRate; i++) {
      simulateDroplet();
      totalDrops++;
    }
  }

  // --- render ---
  function render() {
    if (!imgData) return;
    const d = imgData.data;
    for (let y = 0; y < GH; y++) {
      for (let x = 0; x < GW; x++) {
        const i = y * GW + x;
        const p = i * 4;
        const h = heightmap[i];
        const w = waterMap[i];
        let r, g, b;

        // color coding — height ke hisaab se
        if (h < 0.25) {
          // deep water — neela
          const t = h / 0.25;
          r = 20 + 20 * t;
          g = 40 + 60 * t;
          b = 120 + 60 * t;
        } else if (h < 0.4) {
          // beach/low — hara + peela
          const t = (h - 0.25) / 0.15;
          r = 40 + 80 * t;
          g = 100 + 80 * t;
          b = 180 - 130 * t;
        } else if (h < 0.65) {
          // mid — hara
          const t = (h - 0.4) / 0.25;
          r = 50 + 60 * t;
          g = 140 + 30 * t;
          b = 50 - 10 * t;
        } else if (h < 0.82) {
          // high — brown
          const t = (h - 0.65) / 0.17;
          r = 110 + 50 * t;
          g = 90 - 20 * t;
          b = 40 + 10 * t;
        } else {
          // peaks — safed
          const t = (h - 0.82) / 0.18;
          r = 160 + 95 * t;
          g = 150 + 105 * t;
          b = 140 + 115 * t;
        }

        // paani ka overlay — neela tint lagao
        if (w > 0.01) {
          const wAlpha = Math.min(0.8, w);
          r = r * (1 - wAlpha) + 30 * wAlpha;
          g = g * (1 - wAlpha) + 100 * wAlpha;
          b = b * (1 - wAlpha) + 220 * wAlpha;
        }

        d[p] = Math.max(0, Math.min(255, r | 0));
        d[p + 1] = Math.max(0, Math.min(255, g | 0));
        d[p + 2] = Math.max(0, Math.min(255, b | 0));
        d[p + 3] = 255;
      }
    }

    // temporary canvas pe draw karo fir stretch karo — reusable canvas, har frame naya nahi banate
    tmpCtx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmpCanvas, 0, 0, canvasW, CANVAS_HEIGHT);

    // stats dikhao
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.4)';
    ctx.textAlign = 'left';
    ctx.fillText('HYDRAULIC EROSION', 8, 14);
    ctx.textAlign = 'right';
    const kDrops = (totalDrops / 1000).toFixed(1);
    ctx.fillText(kDrops + 'k drops', canvasW - 8, 14);
  }

  // click se locally paani girane ka option
  canvas.addEventListener('pointerdown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width * GW;
    const my = (e.clientY - rect.top) / rect.height * GH;
    // click ke aas paas concentrated rain
    for (let i = 0; i < 200; i++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.random() * 15;
      const dx = Math.cos(angle) * radius;
      const dy = Math.sin(angle) * radius;
      const sx = mx + dx, sy = my + dy;
      if (sx < 2 || sx > GW - 3 || sy < 2 || sy > GH - 3) continue;
      // mini droplet simulate karo
      let x = sx, y = sy;
      let ddx = 0, ddy = 0, speed = 0, sediment = 0, water = 1;
      for (let step = 0; step < 40; step++) {
        const ix = Math.floor(x), iy = Math.floor(y);
        if (ix < 1 || ix >= GW - 2 || iy < 1 || iy >= GH - 2) break;
        const [gx, gy] = getGradient(x, y);
        ddx = ddx * 0.3 - gx * 0.7;
        ddy = ddy * 0.3 - gy * 0.7;
        const len = Math.sqrt(ddx * ddx + ddy * ddy);
        if (len > 0.0001) { ddx /= len; ddy /= len; }
        const nx = x + ddx, ny = y + ddy;
        const hOld = getHeight(x, y), hNew = getHeight(nx, ny);
        const deltaH = hNew - hOld;
        const cap = Math.max(Math.abs(deltaH), 0.01) * speed * water * 8;
        if (deltaH > 0) {
          modifyTerrain(x, y, Math.min(sediment, deltaH) * depositionRate);
          sediment -= Math.min(sediment, deltaH) * depositionRate;
        } else if (sediment > cap) {
          const drop = (sediment - cap) * depositionRate;
          modifyTerrain(x, y, drop); sediment -= drop;
        } else {
          const erode = Math.min((cap - sediment) * erosionRate, -deltaH);
          modifyTerrain(x, y, -erode); sediment += erode;
        }
        speed = Math.sqrt(Math.max(0, speed * speed - deltaH * 4));
        water *= 0.98;
        if (ix >= 0 && ix < GW && iy >= 0 && iy < GH) waterMap[iy * GW + ix] = Math.min(1, waterMap[iy * GW + ix] + 0.3);
        x = nx; y = ny;
      }
    }
  });

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    step();
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
