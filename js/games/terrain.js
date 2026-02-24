// ============================================================
// Procedural Terrain Generator — Perlin noise se duniya banao
// Heightmap aur Profile view toggle karo, sliders se tune karo
// Multiple octaves, persistence, lacunarity — full control
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, terrain generate karo
export function initTerrain() {
  const container = document.getElementById('terrainContainer');
  if (!container) {
    console.warn('terrainContainer nahi mila bhai, terrain demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 380;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';

  // --- Color bands for terrain height mapping ---
  // har band mein: [threshold, r, g, b] — threshold se neeche ye color lagega
  const TERRAIN_COLORS = [
    { max: 0.00, r: 20,  g: 40,  b: 80  }, // deep water — gehra neela
    { max: 0.15, r: 30,  g: 60,  b: 120 }, // deep water mid
    { max: 0.30, r: 45,  g: 90,  b: 160 }, // medium water
    { max: 0.40, r: 70,  g: 130, b: 190 }, // shallow water — halka neela
    { max: 0.45, r: 210, g: 190, b: 140 }, // sand / beach — ret
    { max: 0.50, r: 190, g: 170, b: 120 }, // dry sand
    { max: 0.55, r: 80,  g: 160, b: 60  }, // light grass — hari ghaas
    { max: 0.65, r: 50,  g: 130, b: 40  }, // grass
    { max: 0.75, r: 30,  g: 100, b: 30  }, // forest — gehra hara
    { max: 0.82, r: 25,  g: 80,  b: 25  }, // dense forest
    { max: 0.88, r: 100, g: 95,  b: 85  }, // rock — pathar
    { max: 0.94, r: 130, g: 125, b: 118 }, // high rock
    { max: 1.00, r: 230, g: 235, b: 240 }, // snow — barf ❄️
  ];

  // --- Perlin Noise Implementation ---
  // classic Perlin noise from scratch — no libraries needed

  // permutation table — random shuffled indices for hash function
  let perm = new Uint8Array(512);

  // gradient vectors for 2D — 4 directions + diagonals
  const GRAD2 = [
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1],
  ];

  // seed se permutation table banao — reproducible results
  function seedPermutation(seed) {
    // simple LCG random number generator — seed se deterministic sequence
    let s = seed & 0xffffffff;
    function rand() {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    }

    // Fisher-Yates shuffle — proper uniform distribution
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = p[i];
      p[i] = p[j];
      p[j] = tmp;
    }

    // double the table — wrap-around ke liye taaki modulo na karna pade
    for (let i = 0; i < 512; i++) {
      perm[i] = p[i & 255];
    }
  }

  // Perlin fade function — 6t^5 - 15t^4 + 10t^3
  // ye smooth interpolation deta hai — linear se kaafi better
  function fade(t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }

  // linear interpolation — a se b tak t fraction pe
  function lerp(a, b, t) {
    return a + t * (b - a);
  }

  // dot product of gradient vector aur distance vector
  function gradDot(hash, x, y) {
    const g = GRAD2[hash & 7]; // 8 gradient directions
    return g[0] * x + g[1] * y;
  }

  // 2D Perlin noise — ek point pe noise value nikalo
  // returns value roughly in range [-1, 1]
  function perlin2D(x, y) {
    // grid cell coordinates
    const xi = Math.floor(x) & 255;
    const yi = Math.floor(y) & 255;

    // fractional part within cell
    const xf = x - Math.floor(x);
    const yf = y - Math.floor(y);

    // fade curves — smooth interpolation weights
    const u = fade(xf);
    const v = fade(yf);

    // hash values for 4 corners of the cell
    const aa = perm[perm[xi] + yi];
    const ab = perm[perm[xi] + yi + 1];
    const ba = perm[perm[xi + 1] + yi];
    const bb = perm[perm[xi + 1] + yi + 1];

    // gradient dot products at each corner
    const g00 = gradDot(aa, xf, yf);
    const g10 = gradDot(ba, xf - 1, yf);
    const g01 = gradDot(ab, xf, yf - 1);
    const g11 = gradDot(bb, xf - 1, yf - 1);

    // bilinear interpolation with fade curves
    const x1 = lerp(g00, g10, u);
    const x2 = lerp(g01, g11, u);
    return lerp(x1, x2, v);
  }

  // fractal Brownian motion — multiple octaves of Perlin noise stack karo
  // ye realistic terrain banata hai — big features + fine detail
  function fbm(x, y, oct, pers, lac, sc) {
    let value = 0;
    let amplitude = 1;
    let frequency = sc;
    let maxAmp = 0; // normalization ke liye

    for (let i = 0; i < oct; i++) {
      value += amplitude * perlin2D(x * frequency, y * frequency);
      maxAmp += amplitude;
      amplitude *= pers; // har octave mein amplitude ghata do
      frequency *= lac; // har octave mein frequency badha do
    }

    // normalize to [0, 1] range — terrain height ke liye chahiye
    return (value / maxAmp + 1) * 0.5;
  }

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let isVisible = false;

  // terrain parameters — ye sliders se control honge
  let octaves = 4;
  let persistence = 0.50;
  let scale = 0.012;
  let lacunarity = 2.0;
  let waterLevel = 0.40;
  let seed = Math.floor(Math.random() * 99999) + 1;

  // view mode — 'heightmap' ya 'profile'
  let viewMode = 'heightmap';

  // cached heightmap data — re-render avoid karne ke liye
  let heightData = null; // Float32Array — har pixel ki height store karega
  let hmWidth = 0, hmHeight = 0;

  // click info — elevation display
  let clickInfo = null; // { x, y, elevation }

  // --- DOM Structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:rgba(2,2,8,0.5)',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // info bar — seed aur elevation dikhayega
  const infoBar = document.createElement('div');
  infoBar.style.cssText = [
    'display:flex',
    'justify-content:space-between',
    'align-items:center',
    'margin-top:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:rgba(' + ACCENT_RGB + ',0.6)',
    'min-height:18px',
    'flex-wrap:wrap',
    'gap:4px 12px',
  ].join(';');
  container.appendChild(infoBar);

  const seedSpan = document.createElement('span');
  seedSpan.textContent = 'Seed: ' + seed;
  infoBar.appendChild(seedSpan);

  const elevSpan = document.createElement('span');
  elevSpan.textContent = 'Click terrain for elevation';
  infoBar.appendChild(elevSpan);

  // controls container
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Helper: button banao ---
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
        btn.style.color = '#b0b0b0';
      }
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  function setButtonActive(btn, active) {
    btn._active = active;
    if (active) {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.35)';
      btn.style.color = ACCENT;
      btn.style.borderColor = ACCENT;
    } else {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      btn.style.color = '#b0b0b0';
      btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
    }
  }

  // --- Helper: slider banao ---
  function createSlider(label, min, max, value, step, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = [
      'display:flex',
      'align-items:center',
      'gap:6px',
      'font-family:"JetBrains Mono",monospace',
      'font-size:11px',
      'color:#888',
    ].join(';');

    const lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.whiteSpace = 'nowrap';
    wrap.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.value = value;
    slider.step = step;
    slider.style.cssText = [
      'width:80px',
      'accent-color:' + ACCENT,
      'cursor:pointer',
    ].join(';');

    const valSpan = document.createElement('span');
    valSpan.textContent = value;
    valSpan.style.cssText = 'min-width:36px;text-align:right;color:' + ACCENT + ';';

    slider.addEventListener('input', () => {
      valSpan.textContent = slider.value;
      onChange(Number(slider.value));
    });

    wrap.appendChild(slider);
    wrap.appendChild(valSpan);
    controlsDiv.appendChild(wrap);
    return { slider, valSpan };
  }

  // --- Controls banao ---

  // View toggle — heightmap / profile
  const hmBtn = createButton('Heightmap', () => {
    viewMode = 'heightmap';
    setButtonActive(hmBtn, true);
    setButtonActive(profBtn, false);
    clickInfo = null;
    renderTerrain();
  });
  const profBtn = createButton('Profile', () => {
    viewMode = 'profile';
    setButtonActive(profBtn, true);
    setButtonActive(hmBtn, false);
    clickInfo = null;
    renderTerrain();
  });
  setButtonActive(hmBtn, true); // default heightmap view

  // separator
  const sep1 = document.createElement('span');
  sep1.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.2);font-size:14px;';
  sep1.textContent = '|';
  controlsDiv.appendChild(sep1);

  // Regenerate button — naya seed, naya terrain
  createButton('Regenerate', () => {
    seed = Math.floor(Math.random() * 99999) + 1;
    seedSpan.textContent = 'Seed: ' + seed;
    clickInfo = null;
    generateAndRender();
  });

  // separator
  const sep2 = document.createElement('span');
  sep2.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.2);font-size:14px;';
  sep2.textContent = '|';
  controlsDiv.appendChild(sep2);

  // Sliders
  createSlider('Oct', 1, 8, octaves, 1, (v) => {
    octaves = v;
    generateAndRender();
  });

  createSlider('Pers', 0.1, 0.9, persistence, 0.05, (v) => {
    persistence = v;
    generateAndRender();
  });

  createSlider('Scale', 0.001, 0.05, scale, 0.001, (v) => {
    scale = v;
    generateAndRender();
  });

  createSlider('Water', 0.0, 0.6, waterLevel, 0.02, (v) => {
    waterLevel = v;
    // water level change pe sirf re-render — heightmap recalculate nahi karna
    renderTerrain();
  });

  // --- Canvas sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = containerWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    generateAndRender();
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Height color lookup ---
  // height value (0-1) se RGB color nikalo using terrain color bands
  function getTerrainColor(h, wl) {
    // water ke neeche hain toh water colors use karo
    // water ke upar hain toh land colors — but remap karo taaki water level ke just upar sand ho
    let mappedH;
    const safeWl = Math.max(wl, 0.001); // zero divide se bachao
    if (h <= safeWl) {
      // underwater — map h from [0, wl] to [0, 0.40] (water range)
      mappedH = (h / safeWl) * 0.40;
    } else {
      // land — map h from [wl, 1] to [0.40, 1.0] (land range)
      const landRange = Math.max(1 - safeWl, 0.001);
      mappedH = 0.40 + ((h - safeWl) / landRange) * 0.60;
    }

    // color bands mein dhundho
    let lower = TERRAIN_COLORS[0];
    let upper = TERRAIN_COLORS[0];

    for (let i = 0; i < TERRAIN_COLORS.length; i++) {
      if (mappedH <= TERRAIN_COLORS[i].max) {
        upper = TERRAIN_COLORS[i];
        lower = i > 0 ? TERRAIN_COLORS[i - 1] : TERRAIN_COLORS[i];
        break;
      }
    }

    // smooth interpolation between bands — banding artifacts avoid karo
    const range = upper.max - (lower === upper ? 0 : lower.max);
    const t = range > 0 ? (mappedH - lower.max) / range : 0;

    return {
      r: Math.round(lower.r + (upper.r - lower.r) * t),
      g: Math.round(lower.g + (upper.g - lower.g) * t),
      b: Math.round(lower.b + (upper.b - lower.b) * t),
    };
  }

  // --- Generate heightmap data ---
  function generateHeightData() {
    seedPermutation(seed);

    // heightmap resolution — CSS pixels use karo (not DPR scaled, for performance)
    hmWidth = Math.ceil(canvasW);
    hmHeight = Math.ceil(canvasH);

    if (hmWidth <= 0 || hmHeight <= 0) return;

    heightData = new Float32Array(hmWidth * hmHeight);

    for (let y = 0; y < hmHeight; y++) {
      for (let x = 0; x < hmWidth; x++) {
        // fbm se height value nikalo — [0, 1] range mein
        const h = fbm(x, y, octaves, persistence, lacunarity, scale);
        heightData[y * hmWidth + x] = h;
      }
    }
  }

  // --- Render Heightmap View ---
  // ImageData + putImageData — fast pixel-by-pixel rendering
  function renderHeightmap() {
    if (!heightData || hmWidth <= 0 || hmHeight <= 0) return;

    const imageData = ctx.createImageData(hmWidth, hmHeight);
    const data = imageData.data;

    for (let y = 0; y < hmHeight; y++) {
      for (let x = 0; x < hmWidth; x++) {
        const h = heightData[y * hmWidth + x];
        const color = getTerrainColor(h, waterLevel);

        // water level ke paas subtle highlight — coastline dikhao
        let brightBoost = 0;
        const distToWater = Math.abs(h - waterLevel);
        if (distToWater < 0.008) {
          brightBoost = (1 - distToWater / 0.008) * 40;
        }

        const idx = (y * hmWidth + x) * 4;
        data[idx] = Math.min(255, color.r + brightBoost);
        data[idx + 1] = Math.min(255, color.g + brightBoost);
        data[idx + 2] = Math.min(255, color.b + brightBoost);
        data[idx + 3] = 255;
      }
    }

    // temp canvas pe putImageData, fir main canvas pe draw (DPR handle)
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = hmWidth;
    tmpCanvas.height = hmHeight;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.putImageData(imageData, 0, 0);

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmpCanvas, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    // click elevation marker — agar koi point select hua hai
    if (clickInfo) {
      drawElevationMarker();
    }

    // labels
    drawHeightmapLabels();
  }

  // --- Render Profile View ---
  // side-view cross section — terrain ka cross-section dikhao center line pe
  function renderProfile() {
    if (!heightData || hmWidth <= 0 || hmHeight <= 0) return;

    const w = canvasW;
    const h = canvasH;
    ctx.clearRect(0, 0, w, h);

    // background grid
    drawProfileGrid(w, h);

    // cross section — heightmap ke center row se height values lo
    const midY = Math.floor(hmHeight / 2);
    const heights = [];
    for (let x = 0; x < hmWidth; x++) {
      heights.push(heightData[midY * hmWidth + x]);
    }

    // terrain profile as filled polygon — baseline se upar tak
    const baseline = h - 20; // neeche thoda gap rakh do
    const topMargin = 30;
    const drawH = baseline - topMargin; // usable drawing height

    // water level line position
    const waterY = baseline - waterLevel * drawH;

    // pehle water fill karo — background mein neela
    ctx.fillStyle = 'rgba(30, 70, 140, 0.25)';
    ctx.fillRect(0, waterY, w, baseline - waterY);

    // terrain ko segments mein draw karo — har segment ka apna color
    ctx.beginPath();
    const firstH = heights[0];
    const firstY = baseline - firstH * drawH;
    ctx.moveTo(0, baseline);
    ctx.lineTo(0, firstY);

    for (let x = 1; x < hmWidth; x++) {
      const terrainH = heights[x];
      const py = baseline - terrainH * drawH;
      ctx.lineTo((x / hmWidth) * w, py);
    }

    ctx.lineTo(w, baseline);
    ctx.closePath();

    // gradient fill — terrain colors
    // multiple colored bands draw karo
    ctx.save();
    ctx.clip();

    // har color band ko horizontal strip mein draw karo
    for (let i = TERRAIN_COLORS.length - 1; i >= 0; i--) {
      const band = TERRAIN_COLORS[i];
      const prevMax = i > 0 ? TERRAIN_COLORS[i - 1].max : 0;

      // band ka y range
      const bandTop = baseline - band.max * drawH;
      const bandBottom = baseline - prevMax * drawH;

      ctx.fillStyle = 'rgb(' + band.r + ',' + band.g + ',' + band.b + ')';
      ctx.fillRect(0, bandTop, w, bandBottom - bandTop + 1);
    }

    ctx.restore();

    // terrain outline — subtle border
    ctx.beginPath();
    ctx.moveTo(0, baseline - heights[0] * drawH);
    for (let x = 1; x < hmWidth; x++) {
      const py = baseline - heights[x] * drawH;
      ctx.lineTo((x / hmWidth) * w, py);
    }
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.3)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // --- Trees on green areas ---
    // simple triangle trees — jahan grass/forest hai wahan lagao
    drawTrees(heights, w, drawH, baseline);

    // water level line — dashed
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = 'rgba(70, 140, 210, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, waterY);
    ctx.lineTo(w, waterY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // water label
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(70, 140, 210, 0.7)';
    ctx.textAlign = 'left';
    ctx.fillText('water: ' + waterLevel.toFixed(2), 8, waterY - 5);

    // profile labels
    drawProfileLabels(w, h);
  }

  // --- Profile grid ---
  function drawProfileGrid(w, h) {
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.04)';
    ctx.lineWidth = 1;

    // horizontal lines
    for (let i = 0; i <= 8; i++) {
      const y = (i / 8) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // vertical lines
    for (let i = 0; i <= 16; i++) {
      const x = (i / 16) * w;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
  }

  // --- Trees rendering for profile view ---
  function drawTrees(heights, w, drawH, baseline) {
    // har kuch pixels pe check karo — agar green zone mein hai toh tree lagao
    const treeSpacing = Math.max(12, Math.floor(hmWidth / 60));

    // deterministic random for tree placement — seed based
    let treeSeed = seed * 7 + 31;
    function treeRand() {
      treeSeed = (treeSeed * 1103515245 + 12345) & 0x7fffffff;
      return treeSeed / 0x7fffffff;
    }

    for (let x = treeSpacing; x < hmWidth - treeSpacing; x += treeSpacing) {
      const h = heights[x];

      // sirf land pe — water ke upar aur snow ke neeche
      if (h <= waterLevel + 0.02 || h > 0.85) continue;

      // green zone check — grass/forest range
      const landFrac = (h - waterLevel) / (1 - waterLevel);
      if (landFrac < 0.08 || landFrac > 0.65) continue;

      // random variation — har jagah tree nahi chahiye
      if (treeRand() > 0.7) continue;

      const px = (x / hmWidth) * w;
      const py = baseline - h * drawH;

      // tree size — height pe depend karta hai
      const treeH = 8 + treeRand() * 10;
      const treeW = 4 + treeRand() * 4;

      // trunk — chhota brown rectangle
      ctx.fillStyle = 'rgba(100, 70, 40, 0.7)';
      ctx.fillRect(px - 1, py - 2, 2, 3);

      // canopy — triangle
      const greenShade = Math.floor(60 + treeRand() * 60);
      ctx.fillStyle = 'rgba(30,' + greenShade + ',25,0.75)';
      ctx.beginPath();
      ctx.moveTo(px, py - treeH);
      ctx.lineTo(px - treeW / 2, py - 2);
      ctx.lineTo(px + treeW / 2, py - 2);
      ctx.closePath();
      ctx.fill();
    }
  }

  // --- Elevation marker on heightmap ---
  function drawElevationMarker() {
    if (!clickInfo) return;

    const x = clickInfo.x;
    const y = clickInfo.y;
    const elev = clickInfo.elevation;

    // crosshair lines
    ctx.save();
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvasH);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvasW, y);
    ctx.stroke();

    ctx.setLineDash([]);

    // center dot
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // elevation label — tooltip style
    const label = 'h=' + elev.toFixed(3);
    ctx.font = '11px "JetBrains Mono", monospace';
    const metrics = ctx.measureText(label);
    const labelW = metrics.width + 12;
    const labelH = 20;

    // position — try right of click, flip if near edge
    let lx = x + 10;
    let ly = y - 10;
    if (lx + labelW > canvasW - 5) lx = x - labelW - 10;
    if (ly < 5) ly = y + 20;

    // background
    ctx.fillStyle = 'rgba(10, 10, 20, 0.85)';
    ctx.beginPath();
    ctx.roundRect(lx, ly - labelH + 4, labelW, labelH, 4);
    ctx.fill();
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // text
    ctx.fillStyle = ACCENT;
    ctx.textAlign = 'left';
    ctx.fillText(label, lx + 6, ly);

    ctx.restore();
  }

  // --- Heightmap labels ---
  function drawHeightmapLabels() {
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.fillText('HEIGHTMAP VIEW', 8, 14);

    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.3)';
    ctx.fillText('oct:' + octaves + ' pers:' + persistence.toFixed(2), canvasW - 8, 14);
  }

  // --- Profile labels ---
  function drawProfileLabels(w, h) {
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.fillText('PROFILE VIEW (center cross-section)', 8, 14);

    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.3)';
    ctx.fillText('oct:' + octaves + ' pers:' + persistence.toFixed(2), w - 8, 14);
  }

  // --- Main render function ---
  function renderTerrain() {
    if (!isVisible) return;
    if (!heightData) return;

    if (viewMode === 'heightmap') {
      renderHeightmap();
    } else {
      renderProfile();
    }

    updateInfoBar();
  }

  // --- Generate + Render combo ---
  function generateAndRender() {
    generateHeightData();
    renderTerrain();
  }

  // --- Info bar update ---
  function updateInfoBar() {
    seedSpan.textContent = 'Seed: ' + seed;

    if (clickInfo && viewMode === 'heightmap') {
      const h = clickInfo.elevation;
      let zone = 'unknown';
      if (h <= waterLevel) zone = 'water';
      else {
        const landFrac = (h - waterLevel) / (1 - waterLevel);
        if (landFrac < 0.10) zone = 'beach';
        else if (landFrac < 0.35) zone = 'grass';
        else if (landFrac < 0.60) zone = 'forest';
        else if (landFrac < 0.80) zone = 'rock';
        else zone = 'snow';
      }
      elevSpan.textContent = 'Elevation: ' + h.toFixed(3) + ' (' + zone + ')';
    } else {
      elevSpan.textContent = 'Click terrain for elevation';
    }
  }

  // --- Click handler — elevation at point ---
  canvas.addEventListener('click', (e) => {
    if (viewMode !== 'heightmap') return;
    if (!heightData) return;

    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // heightmap pixel coordinates
    const hx = Math.floor((mx / canvasW) * hmWidth);
    const hy = Math.floor((my / canvasH) * hmHeight);

    if (hx >= 0 && hx < hmWidth && hy >= 0 && hy < hmHeight) {
      const elevation = heightData[hy * hmWidth + hx];
      clickInfo = { x: mx, y: my, elevation: elevation };
      renderTerrain();
    }
  });

  // touch support
  canvas.addEventListener('touchend', (e) => {
    if (viewMode !== 'heightmap') return;
    if (!heightData) return;
    if (!e.changedTouches || !e.changedTouches.length) return;

    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const touch = e.changedTouches[0];
    const mx = touch.clientX - rect.left;
    const my = touch.clientY - rect.top;

    const hx = Math.floor((mx / canvasW) * hmWidth);
    const hy = Math.floor((my / canvasH) * hmHeight);

    if (hx >= 0 && hx < hmWidth && hy >= 0 && hy < hmHeight) {
      const elevation = heightData[hy * hmWidth + hx];
      clickInfo = { x: mx, y: my, elevation: elevation };
      renderTerrain();
    }
  });

  // --- IntersectionObserver — sirf jab dikhe tab render karo ---
  function startRendering() {
    if (isVisible) return;
    isVisible = true;
    resizeCanvas(); // ye generateAndRender call karega internally
  }

  function stopRendering() {
    isVisible = false;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          startRendering();
        } else {
          stopRendering();
        }
      });
    },
    { threshold: 0.1 }
  );

  observer.observe(container);

  // tab switch pe bhi handle karo
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopRendering();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startRendering();
    }
  });
}
