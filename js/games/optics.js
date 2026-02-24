// ============================================================
// 2D Ray Optics Bench — mirrors, lenses, prisms pe ray tracing
// Snell's law, reflection, refraction, dispersion — full optics demo
// click karke elements place karo, drag se move, scroll se rotate
// ============================================================

// yahi function export hoga — container dhundho, canvas banao, optics bench chalao
export function initOptics() {
  const container = document.getElementById('opticsContainer');
  if (!container) {
    console.warn('opticsContainer nahi mila bhai, optics demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b'; // amber accent
  const ACCENT_RGB = '245,158,11';
  const BG_DARK = '#1a1a2e';
  const MAX_BOUNCES = 10; // ek ray max kitni baar reflect/refract ho sakti hai
  const RAY_LENGTH = 2000; // ray kitni door tak jaaye agar kuch na mile
  const ELEMENT_HIT_RADIUS = 15; // element select karne ke liye proximity
  const GRID_SPACING = 40; // background grid ka spacing

  // refractive indices — different wavelengths ke liye (dispersion ke liye)
  // Cauchy's equation approximate: n(λ) = A + B/λ^2
  const GLASS_N_BASE = 1.52; // crown glass ka base refractive index
  const GLASS_DISPERSION = 0.008; // dispersion coefficient — wavelength pe depend karta hai

  // rainbow colors — VIBGYOR ka spectrum
  const SPECTRUM_COLORS = [
    { name: 'red',    wavelength: 700, color: '#ff0000', n: 1.510 },
    { name: 'orange', wavelength: 620, color: '#ff7700', n: 1.514 },
    { name: 'yellow', wavelength: 580, color: '#ffdd00', n: 1.517 },
    { name: 'green',  wavelength: 530, color: '#00ff44', n: 1.521 },
    { name: 'cyan',   wavelength: 490, color: '#00ccff', n: 1.525 },
    { name: 'blue',   wavelength: 460, color: '#0044ff', n: 1.530 },
    { name: 'violet', wavelength: 410, color: '#8800ff', n: 1.538 },
  ];

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;

  // light source — position, angle, ray count, type
  let lightSource = {
    x: 60, y: 200,
    angle: 0, // radians — direction of emission
    rayCount: 12,
    isPointSource: false, // false = parallel beam, true = point source fan
    fanAngle: Math.PI / 3, // point source fan ka total angle
  };

  // optical elements list — har element ka type, position, rotation, size
  let elements = [];

  // interaction state
  let selectedTool = null; // kaunsa element place karna hai (null = none)
  let dragTarget = null; // kya drag ho raha hai (null, 'source', ya element index)
  let dragOffsetX = 0, dragOffsetY = 0;
  let isDragging = false;
  let hoveredElement = -1; // hover highlight ke liye

  // settings
  let dispersionOn = false;
  let showNormals = false;

  // animation state
  let animationId = null;
  let isVisible = false;
  let needsRedraw = true;

  // precomputed rays — har frame draw ke liye cache
  let cachedRays = [];

  // --- DOM Structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(245,158,11,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:rgba(2,2,8,0.5)',
    'touch-action:none',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls — dark theme, inline CSS, JetBrains Mono ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:6px',
    'margin-top:10px',
    'align-items:center',
    'padding:10px',
    'background:' + BG_DARK,
    'border-radius:6px',
    'border:1px solid rgba(245,158,11,0.15)',
  ].join(';');
  container.appendChild(controlsDiv);

  // second row — sliders aur toggles
  const controlsDiv2 = document.createElement('div');
  controlsDiv2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:6px',
    'align-items:center',
    'padding:10px',
    'background:' + BG_DARK,
    'border-radius:6px',
    'border:1px solid rgba(245,158,11,0.15)',
  ].join(';');
  container.appendChild(controlsDiv2);

  // info bar
  const infoBar = document.createElement('div');
  infoBar.style.cssText = [
    'margin-top:6px',
    'padding:6px 10px',
    'font-size:11px',
    'font-family:"JetBrains Mono",monospace',
    'color:#8892a4',
    'background:rgba(26,26,46,0.5)',
    'border-radius:4px',
    'border:1px solid rgba(245,158,11,0.08)',
  ].join(';');
  infoBar.textContent = 'Select tool → Click canvas to place | Drag to move | Scroll to rotate | Double-click to delete';
  container.appendChild(infoBar);

  // --- Control Helpers ---

  // button banane ka helper — element placement tools ke liye
  function createToolButton(text, toolName, parentDiv) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.dataset.tool = toolName;
    btn.style.cssText = [
      'background:rgba(245,158,11,0.1)',
      'color:' + ACCENT,
      'border:1px solid rgba(245,158,11,0.3)',
      'border-radius:4px',
      'padding:4px 10px',
      'cursor:pointer',
      'font-family:"JetBrains Mono",monospace',
      'font-size:11px',
      'transition:all 0.2s ease',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      if (selectedTool !== toolName) {
        btn.style.background = 'rgba(245,158,11,0.25)';
      }
    });
    btn.addEventListener('mouseleave', () => {
      if (selectedTool !== toolName) {
        btn.style.background = 'rgba(245,158,11,0.1)';
        btn.style.color = ACCENT;
      }
    });
    btn.addEventListener('click', () => {
      // agar already selected hai toh deselect karo
      if (selectedTool === toolName) {
        selectedTool = null;
        updateToolButtons();
      } else {
        selectedTool = toolName;
        updateToolButtons();
      }
    });
    parentDiv.appendChild(btn);
    return btn;
  }

  // sab tool buttons ka style update karo — selected waale ko highlight
  const toolButtons = [];
  function updateToolButtons() {
    toolButtons.forEach(btn => {
      const isActive = selectedTool === btn.dataset.tool;
      btn.style.background = isActive ? 'rgba(245,158,11,0.35)' : 'rgba(245,158,11,0.1)';
      btn.style.color = isActive ? '#fbbf24' : ACCENT;
      btn.style.borderColor = isActive ? 'rgba(245,158,11,0.6)' : 'rgba(245,158,11,0.3)';
    });
    // cursor bhi change karo
    canvas.style.cursor = selectedTool ? 'copy' : 'crosshair';
  }

  // action button — clear, presets wagairah
  function createButton(text, onClick, parentDiv) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'background:rgba(245,158,11,0.1)',
      'color:' + ACCENT,
      'border:1px solid rgba(245,158,11,0.3)',
      'border-radius:4px',
      'padding:4px 10px',
      'cursor:pointer',
      'font-family:"JetBrains Mono",monospace',
      'font-size:11px',
      'transition:all 0.2s ease',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(245,158,11,0.25)';
      btn.style.color = '#fbbf24';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(245,158,11,0.1)';
      btn.style.color = ACCENT;
    });
    btn.addEventListener('click', onClick);
    parentDiv.appendChild(btn);
    return btn;
  }

  // toggle button — on/off state
  function createToggle(label, initial, onChange, parentDiv) {
    const btn = document.createElement('button');
    let active = initial;

    function updateStyle() {
      btn.style.cssText = [
        'padding:4px 10px',
        'font-size:11px',
        'border-radius:4px',
        'cursor:pointer',
        'font-family:"JetBrains Mono",monospace',
        'transition:all 0.2s ease',
        'white-space:nowrap',
        'background:' + (active ? 'rgba(245,158,11,0.25)' : 'rgba(245,158,11,0.05)'),
        'color:' + (active ? '#fbbf24' : '#6b7280'),
        'border:1px solid ' + (active ? 'rgba(245,158,11,0.5)' : 'rgba(245,158,11,0.15)'),
      ].join(';');
      btn.textContent = label + (active ? ' ON' : ' OFF');
    }

    updateStyle();
    btn.addEventListener('click', () => {
      active = !active;
      updateStyle();
      onChange(active);
    });
    parentDiv.appendChild(btn);
    return { btn, setActive: (v) => { active = v; updateStyle(); } };
  }

  // slider banane ka helper
  function createSlider(label, min, max, value, step, onChange, parentDiv) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = 'color:#8892a4;font-size:11px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
    lbl.textContent = label + ': ' + value;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.style.cssText = 'width:80px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      lbl.textContent = label + ': ' + v;
      onChange(v);
    });

    wrap.appendChild(lbl);
    wrap.appendChild(slider);
    parentDiv.appendChild(wrap);
    return { slider, label: lbl };
  }

  // --- Separator for visual grouping ---
  function addSep(parentDiv) {
    const sep = document.createElement('span');
    sep.style.cssText = 'width:1px;height:20px;background:rgba(245,158,11,0.2);display:inline-block;';
    parentDiv.appendChild(sep);
  }

  // --- Controls Row 1: Element tools ---
  // label pehle
  const toolLabel = document.createElement('span');
  toolLabel.style.cssText = 'color:#8892a4;font-size:11px;font-family:"JetBrains Mono",monospace;';
  toolLabel.textContent = 'Place:';
  controlsDiv.appendChild(toolLabel);

  toolButtons.push(createToolButton('Flat Mirror', 'flat-mirror', controlsDiv));
  toolButtons.push(createToolButton('Curved Mirror', 'curved-mirror', controlsDiv));
  toolButtons.push(createToolButton('Convex Lens', 'convex-lens', controlsDiv));
  toolButtons.push(createToolButton('Concave Lens', 'concave-lens', controlsDiv));
  toolButtons.push(createToolButton('Prism', 'prism', controlsDiv));

  addSep(controlsDiv);

  createButton('Clear All', () => {
    elements = [];
    selectedTool = null;
    updateToolButtons();
    traceAllRays();
    needsRedraw = true;
  }, controlsDiv);

  // --- Presets ---
  const presetSelect = document.createElement('select');
  presetSelect.style.cssText = [
    'padding:4px 8px',
    'font-size:11px',
    'border-radius:4px',
    'cursor:pointer',
    'background:rgba(245,158,11,0.1)',
    'color:#8892a4',
    'border:1px solid rgba(245,158,11,0.25)',
    'font-family:"JetBrains Mono",monospace',
  ].join(';');

  const presetNames = ['Presets...', 'Rainbow', 'Telescope', 'Periscope'];
  presetNames.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    opt.style.background = '#1a1a1a';
    opt.style.color = '#b0b0b0';
    presetSelect.appendChild(opt);
  });
  presetSelect.addEventListener('change', () => {
    loadPreset(presetSelect.value);
    presetSelect.value = 'Presets...';
  });
  controlsDiv.appendChild(presetSelect);

  // --- Controls Row 2: Sliders, toggles ---
  const raysSliderCtrl = createSlider('Rays', 5, 30, lightSource.rayCount, 1, (v) => {
    lightSource.rayCount = Math.round(v);
    traceAllRays();
    needsRedraw = true;
  }, controlsDiv2);

  addSep(controlsDiv2);

  // beam type toggle — parallel vs point source
  const beamToggle = createToggle('Point Source', lightSource.isPointSource, (v) => {
    lightSource.isPointSource = v;
    traceAllRays();
    needsRedraw = true;
  }, controlsDiv2);

  addSep(controlsDiv2);

  // dispersion toggle
  const dispersionToggle = createToggle('Dispersion', dispersionOn, (v) => {
    dispersionOn = v;
    traceAllRays();
    needsRedraw = true;
  }, controlsDiv2);

  // ============================
  // DPR-AWARE RESIZE
  // ============================
  function resize() {
    dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    canvasW = w;
    canvasH = CANVAS_HEIGHT;
    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    traceAllRays();
    needsRedraw = true;
  }
  resize();
  window.addEventListener('resize', resize);

  // ============================
  // OPTICAL ELEMENT DEFINITIONS
  // ============================

  // har element ek object hai: { type, x, y, angle, ...type-specific params }
  function createFlatMirror(x, y, angle) {
    return {
      type: 'flat-mirror',
      x, y,
      angle: angle || 0, // rotation in radians
      length: 80, // mirror ki length
    };
  }

  function createCurvedMirror(x, y, angle) {
    return {
      type: 'curved-mirror',
      x, y,
      angle: angle || 0,
      length: 100,
      radius: 120, // curvature radius — positive = concave, negative = convex
      isConcave: true,
    };
  }

  function createConvexLens(x, y, angle) {
    return {
      type: 'convex-lens',
      x, y,
      angle: angle || 0,
      height: 80, // lens ki height
      focalLength: 100, // focal length pixels mein
    };
  }

  function createConcaveLens(x, y, angle) {
    return {
      type: 'concave-lens',
      x, y,
      angle: angle || 0,
      height: 80,
      focalLength: 100,
    };
  }

  function createPrism(x, y, angle) {
    return {
      type: 'prism',
      x, y,
      angle: angle || 0,
      size: 60, // equilateral triangle ka side length
      refractiveIndex: GLASS_N_BASE,
    };
  }

  // factory — tool name se element banao
  function createElement(toolName, x, y) {
    switch (toolName) {
      case 'flat-mirror': return createFlatMirror(x, y);
      case 'curved-mirror': return createCurvedMirror(x, y);
      case 'convex-lens': return createConvexLens(x, y);
      case 'concave-lens': return createConcaveLens(x, y);
      case 'prism': return createPrism(x, y);
      default: return null;
    }
  }

  // ============================
  // GEOMETRY HELPERS — intersection, reflection, refraction
  // ============================

  // 2D vector dot product
  function dot(ax, ay, bx, by) { return ax * bx + ay * by; }

  // 2D vector cross product (scalar)
  function cross(ax, ay, bx, by) { return ax * by - ay * bx; }

  // vector normalize karo
  function normalize(x, y) {
    const len = Math.sqrt(x * x + y * y);
    if (len < 1e-12) return [0, 0];
    return [x / len, y / len];
  }

  // vector length
  function vecLen(x, y) { return Math.sqrt(x * x + y * y); }

  // ray-segment intersection — parametric method
  // ray: origin (ox,oy) + t*(dx,dy), t >= 0
  // segment: p1(x1,y1) to p2(x2,y2)
  // returns { t, point, u } ya null agar intersect nahi karta
  function raySegmentIntersect(ox, oy, dx, dy, x1, y1, x2, y2) {
    const sx = x2 - x1;
    const sy = y2 - y1;
    const denom = cross(dx, dy, sx, sy);
    if (Math.abs(denom) < 1e-10) return null; // parallel hai

    const tx = cross(x1 - ox, y1 - oy, sx, sy) / denom;
    const u = cross(x1 - ox, y1 - oy, dx, dy) / denom;

    if (tx < 0.01 || u < 0 || u > 1) return null; // intersection ray ke peeche hai ya segment ke bahar

    return {
      t: tx,
      point: [ox + tx * dx, oy + tx * dy],
      u: u, // segment pe kahan hit hua (0 to 1)
    };
  }

  // ray-arc intersection — curved mirror ke liye
  // arc defined by center, radius, startAngle, endAngle
  // returns closest valid intersection
  function rayArcIntersect(ox, oy, dx, dy, cx, cy, r, startAng, endAng) {
    // ray-circle intersection solve karo
    const fx = ox - cx;
    const fy = oy - cy;
    const a = dx * dx + dy * dy;
    const b = 2 * (fx * dx + fy * dy);
    const c = fx * fx + fy * fy - r * r;
    let disc = b * b - 4 * a * c;
    if (disc < 0) return null;
    disc = Math.sqrt(disc);

    let bestT = Infinity;
    let bestPoint = null;

    // dono solutions check karo
    for (const sign of [-1, 1]) {
      const t = (-b + sign * disc) / (2 * a);
      if (t < 0.01) continue;
      const px = ox + t * dx;
      const py = oy + t * dy;

      // check karo ki point arc ke range mein hai
      let ang = Math.atan2(py - cy, px - cx);
      // angle ko normalize karo startAng ke relative
      let diff = ang - startAng;
      while (diff < -Math.PI) diff += 2 * Math.PI;
      while (diff > Math.PI) diff -= 2 * Math.PI;
      let range = endAng - startAng;
      while (range < -Math.PI) range += 2 * Math.PI;
      while (range > Math.PI) range -= 2 * Math.PI;

      // simple range check — agar diff 0 aur range ke beech mein hai
      if ((range > 0 && diff >= 0 && diff <= range) ||
          (range < 0 && diff <= 0 && diff >= range) ||
          Math.abs(range) < 0.01) {
        // nahi, better approach — angular distance check karo
      }

      // simpler approach: check point is within angular span
      const angNorm = ((ang % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      const sNorm = ((startAng % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      const eNorm = ((endAng % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);

      let inArc = false;
      if (sNorm <= eNorm) {
        inArc = angNorm >= sNorm && angNorm <= eNorm;
      } else {
        inArc = angNorm >= sNorm || angNorm <= eNorm;
      }

      if (inArc && t < bestT) {
        bestT = t;
        bestPoint = [px, py];
      }
    }

    if (!bestPoint) return null;
    return { t: bestT, point: bestPoint };
  }

  // reflection — incident ray ka reflected direction nikalo
  // d = incident direction (normalized), n = surface normal (normalized)
  // r = d - 2(d.n)n
  function reflect(dx, dy, nx, ny) {
    const dn = dot(dx, dy, nx, ny);
    return [dx - 2 * dn * nx, dy - 2 * dn * ny];
  }

  // refraction — Snell's law apply karo
  // d = incident direction (normalized), n = surface normal (pointing towards incident side)
  // n1, n2 = refractive indices
  // returns refracted direction ya null agar total internal reflection ho
  function refract(dx, dy, nx, ny, n1, n2) {
    // cos(theta_i) = -dot(d, n) — normal incident side ki taraf point karta hai
    let cosI = -dot(dx, dy, nx, ny);

    // agar cos negative hai toh normal flip karo (ray doosri side se aa rahi hai)
    if (cosI < 0) {
      nx = -nx;
      ny = -ny;
      cosI = -cosI;
      // swap n1 and n2 bhi
      const temp = n1;
      n1 = n2;
      n2 = temp;
    }

    const ratio = n1 / n2;
    const sin2T = ratio * ratio * (1 - cosI * cosI);

    // total internal reflection check
    if (sin2T > 1.0) return null;

    const cosT = Math.sqrt(1 - sin2T);
    return [
      ratio * dx + (ratio * cosI - cosT) * nx,
      ratio * dy + (ratio * cosI - cosT) * ny,
    ];
  }

  // ============================
  // ELEMENT GEOMETRY — surfaces extract karo har element se
  // ============================

  // flat mirror ke endpoints nikalo
  function getFlatMirrorSegment(elem) {
    const halfLen = elem.length / 2;
    const cosA = Math.cos(elem.angle);
    const sinA = Math.sin(elem.angle);
    return {
      x1: elem.x - halfLen * cosA,
      y1: elem.y - halfLen * sinA,
      x2: elem.x + halfLen * cosA,
      y2: elem.y + halfLen * sinA,
      // normal — perpendicular to mirror surface
      nx: -sinA,
      ny: cosA,
    };
  }

  // curved mirror ka arc data nikalo
  function getCurvedMirrorArc(elem) {
    const r = elem.isConcave ? elem.radius : -elem.radius;
    const absR = Math.abs(r);

    // arc ka center — mirror ke peeche r distance pe
    const normalAngle = elem.angle + Math.PI / 2; // mirror surface ke perpendicular
    const cx = elem.x + (elem.isConcave ? 1 : -1) * absR * Math.cos(normalAngle);
    const cy = elem.y + (elem.isConcave ? 1 : -1) * absR * Math.sin(normalAngle);

    // arc ka angular span — mirror length se calculate karo
    const halfAngle = Math.asin(Math.min(elem.length / (2 * absR), 0.99));
    const centerAngle = Math.atan2(elem.y - cy, elem.x - cx);

    return {
      cx, cy,
      radius: absR,
      startAngle: centerAngle - halfAngle,
      endAngle: centerAngle + halfAngle,
      isConcave: elem.isConcave,
    };
  }

  // lens ke liye — thin lens approximation, ek vertical line segment jaisi treat karo
  function getLensSegment(elem) {
    const halfH = elem.height / 2;
    const cosA = Math.cos(elem.angle);
    const sinA = Math.sin(elem.angle);
    return {
      x1: elem.x - halfH * cosA,
      y1: elem.y - halfH * sinA,
      x2: elem.x + halfH * cosA,
      y2: elem.y + halfH * sinA,
      // optical axis direction — lens ke perpendicular
      ax: -sinA,
      ay: cosA,
    };
  }

  // prism ke vertices nikalo — equilateral triangle
  function getPrismVertices(elem) {
    const s = elem.size;
    const h = s * Math.sqrt(3) / 2; // triangle ki height
    // triangle centered at (x, y), tip upar
    const pts = [
      [0, -h * 2 / 3],     // top vertex
      [-s / 2, h / 3],     // bottom left
      [s / 2, h / 3],      // bottom right
    ];
    // rotate karo element angle se
    const cosA = Math.cos(elem.angle);
    const sinA = Math.sin(elem.angle);
    return pts.map(([px, py]) => [
      elem.x + px * cosA - py * sinA,
      elem.y + px * sinA + py * cosA,
    ]);
  }

  // prism ke 3 sides as segments
  function getPrismSegments(elem) {
    const verts = getPrismVertices(elem);
    const segs = [];
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3;
      const dx = verts[j][0] - verts[i][0];
      const dy = verts[j][1] - verts[i][1];
      // outward normal calculate karo
      const [nx, ny] = normalize(-dy, dx);
      // check karo ki normal bahar point karta hai (center se door)
      const mx = (verts[i][0] + verts[j][0]) / 2 - elem.x;
      const my = (verts[i][1] + verts[j][1]) / 2 - elem.y;
      const outward = dot(nx, ny, mx, my) > 0;
      segs.push({
        x1: verts[i][0], y1: verts[i][1],
        x2: verts[j][0], y2: verts[j][1],
        nx: outward ? nx : -nx,
        ny: outward ? ny : -ny,
      });
    }
    return segs;
  }

  // check karo point prism ke andar hai ya nahi — cross product method
  function isInsidePrism(px, py, elem) {
    const verts = getPrismVertices(elem);
    let sign = 0;
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3;
      const cp = cross(
        verts[j][0] - verts[i][0], verts[j][1] - verts[i][1],
        px - verts[i][0], py - verts[i][1]
      );
      if (cp > 0 && sign < 0) return false;
      if (cp < 0 && sign > 0) return false;
      if (cp !== 0) sign = cp > 0 ? 1 : -1;
    }
    return true;
  }

  // ============================
  // RAY TRACING ENGINE — ye hai main jugaad
  // ============================

  // ek single ray trace karo — origin, direction, max bounces
  // returns array of ray segments: [{x1,y1,x2,y2,color}]
  function traceRay(ox, oy, dx, dy, maxBounces, color, refractiveIndex) {
    const segments = [];
    let curOx = ox, curOy = oy;
    let curDx = dx, curDy = dy;
    let curN = refractiveIndex || 1.0; // current medium ka refractive index
    let bounces = 0;

    while (bounces < maxBounces) {
      // sabse nearest intersection dhundho
      let bestHit = null;
      let bestT = Infinity;
      let bestElemIdx = -1;
      let bestHitType = null; // 'reflect', 'refract-in', 'refract-out'
      let bestNormal = null;
      let bestElemN = 1.0; // element ka refractive index

      for (let i = 0; i < elements.length; i++) {
        const elem = elements[i];

        if (elem.type === 'flat-mirror') {
          const seg = getFlatMirrorSegment(elem);
          const hit = raySegmentIntersect(curOx, curOy, curDx, curDy, seg.x1, seg.y1, seg.x2, seg.y2);
          if (hit && hit.t < bestT) {
            bestT = hit.t;
            bestHit = hit.point;
            bestElemIdx = i;
            bestHitType = 'reflect';
            bestNormal = [seg.nx, seg.ny];
          }
        }

        else if (elem.type === 'curved-mirror') {
          const arc = getCurvedMirrorArc(elem);
          const hit = rayArcIntersect(curOx, curOy, curDx, curDy, arc.cx, arc.cy, arc.radius, arc.startAngle, arc.endAngle);
          if (hit && hit.t < bestT) {
            bestT = hit.t;
            bestHit = hit.point;
            bestElemIdx = i;
            bestHitType = 'reflect';
            // curved mirror ka normal — center se hit point ki direction
            const [hnx, hny] = normalize(hit.point[0] - arc.cx, hit.point[1] - arc.cy);
            // concave mirror ke liye normal inside ki taraf hona chahiye
            bestNormal = arc.isConcave ? [-hnx, -hny] : [hnx, hny];
          }
        }

        else if (elem.type === 'convex-lens' || elem.type === 'concave-lens') {
          // thin lens — segment se intersection, fir direction change karo
          const seg = getLensSegment(elem);
          const hit = raySegmentIntersect(curOx, curOy, curDx, curDy, seg.x1, seg.y1, seg.x2, seg.y2);
          if (hit && hit.t < bestT) {
            bestT = hit.t;
            bestHit = hit.point;
            bestElemIdx = i;
            bestHitType = elem.type === 'convex-lens' ? 'lens-convex' : 'lens-concave';
            bestNormal = [seg.ax, seg.ay]; // optical axis direction
          }
        }

        else if (elem.type === 'prism') {
          const prismSegs = getPrismSegments(elem);
          for (const seg of prismSegs) {
            const hit = raySegmentIntersect(curOx, curOy, curDx, curDy, seg.x1, seg.y1, seg.x2, seg.y2);
            if (hit && hit.t < bestT) {
              bestT = hit.t;
              bestHit = hit.point;
              bestElemIdx = i;
              // check karo ray prism ke andar ja rahi hai ya bahar
              const insideNow = Math.abs(curN - 1.0) > 0.01;
              bestHitType = insideNow ? 'refract-out' : 'refract-in';
              bestNormal = [seg.nx, seg.ny];
              bestElemN = elem.refractiveIndex;

              // agar dispersion on hai toh wavelength-specific n use karo
              if (dispersionOn && color) {
                const specEntry = SPECTRUM_COLORS.find(s => s.color === color);
                if (specEntry) bestElemN = specEntry.n;
              }
            }
          }
        }
      }

      // agar koi intersection nahi mila, ray canvas se bahar jaayegi
      if (!bestHit) {
        const endX = curOx + RAY_LENGTH * curDx;
        const endY = curOy + RAY_LENGTH * curDy;
        segments.push({ x1: curOx, y1: curOy, x2: endX, y2: endY, color });
        break;
      }

      // intersection mila — segment add karo
      segments.push({ x1: curOx, y1: curOy, x2: bestHit[0], y2: bestHit[1], color });

      // ab direction update karo based on hit type
      const [nx, ny] = bestNormal;

      if (bestHitType === 'reflect') {
        // simple reflection — angle of incidence = angle of reflection
        const [rx, ry] = reflect(curDx, curDy, nx, ny);
        curDx = rx;
        curDy = ry;
        curOx = bestHit[0] + rx * 0.1; // thoda aage shift karo taaki same surface pe dobara hit na ho
        curOy = bestHit[1] + ry * 0.1;
      }

      else if (bestHitType === 'lens-convex') {
        // thin convex lens — ray ko focal point ki taraf bend karo
        const elem = elements[bestElemIdx];
        const f = elem.focalLength;

        // optical axis direction
        const axAngle = elem.angle + Math.PI / 2;
        const [ax, ay] = [Math.cos(axAngle), Math.sin(axAngle)];

        // ray hit point ka lens center se distance (perpendicular to optical axis)
        const hx = bestHit[0] - elem.x;
        const hy = bestHit[1] - elem.y;
        const h_perp = dot(hx, hy, Math.cos(elem.angle), Math.sin(elem.angle));

        // lens bending — parallel ray focal point pe jaayegi
        // general case: deflection angle = -h/f (paraxial approximation)
        const deflection = -h_perp / f;

        // ray direction ko rotate karo deflection se — optical axis ke around
        const inDot = dot(curDx, curDy, ax, ay);
        const sign = inDot >= 0 ? 1 : -1;
        const perpDx = Math.cos(elem.angle);
        const perpDy = Math.sin(elem.angle);

        // new direction: original direction + deflection component
        let newDx = curDx + deflection * ax * sign;
        let newDy = curDy + deflection * ay * sign;
        const [ndx, ndy] = normalize(newDx, newDy);
        curDx = ndx;
        curDy = ndy;
        curOx = bestHit[0] + ndx * 0.1;
        curOy = bestHit[1] + ndy * 0.1;
      }

      else if (bestHitType === 'lens-concave') {
        // thin concave lens — ray ko focal point se door bend karo (diverge)
        const elem = elements[bestElemIdx];
        const f = elem.focalLength;

        const axAngle = elem.angle + Math.PI / 2;
        const [ax, ay] = [Math.cos(axAngle), Math.sin(axAngle)];

        const hx = bestHit[0] - elem.x;
        const hy = bestHit[1] - elem.y;
        const h_perp = dot(hx, hy, Math.cos(elem.angle), Math.sin(elem.angle));

        // concave lens diverge karti hai — positive deflection
        const deflection = h_perp / f;

        const inDot = dot(curDx, curDy, ax, ay);
        const sign = inDot >= 0 ? 1 : -1;

        let newDx = curDx + deflection * ax * sign;
        let newDy = curDy + deflection * ay * sign;
        const [ndx, ndy] = normalize(newDx, newDy);
        curDx = ndx;
        curDy = ndy;
        curOx = bestHit[0] + ndx * 0.1;
        curOy = bestHit[1] + ndy * 0.1;
      }

      else if (bestHitType === 'refract-in') {
        // Snell's law — air se glass mein jaana
        const result = refract(curDx, curDy, nx, ny, 1.0, bestElemN);
        if (result) {
          const [rx, ry] = normalize(result[0], result[1]);
          curDx = rx;
          curDy = ry;
          curN = bestElemN;
        } else {
          // total internal reflection — ye air to glass mein almost kabhi nahi hoga but safety ke liye
          const [rx, ry] = reflect(curDx, curDy, nx, ny);
          curDx = rx;
          curDy = ry;
        }
        curOx = bestHit[0] + curDx * 0.1;
        curOy = bestHit[1] + curDy * 0.1;
      }

      else if (bestHitType === 'refract-out') {
        // Snell's law — glass se air mein nikal raha hai
        const result = refract(curDx, curDy, nx, ny, bestElemN, 1.0);
        if (result) {
          const [rx, ry] = normalize(result[0], result[1]);
          curDx = rx;
          curDy = ry;
          curN = 1.0;
        } else {
          // total internal reflection! — glass ke andar hi reflect ho jaayega
          const [rx, ry] = reflect(curDx, curDy, nx, ny);
          curDx = rx;
          curDy = ry;
          // curN stays same — abhi bhi glass mein hai
        }
        curOx = bestHit[0] + curDx * 0.1;
        curOy = bestHit[1] + curDy * 0.1;
      }

      bounces++;
    }

    return segments;
  }

  // sab rays trace karo — light source se
  function traceAllRays() {
    cachedRays = [];
    const n = lightSource.rayCount;
    const srcAngle = lightSource.angle;

    if (lightSource.isPointSource) {
      // point source — fan of rays
      const totalFan = lightSource.fanAngle;
      const startAngle = srcAngle - totalFan / 2;

      // agar dispersion on hai toh har ray ke liye spectrum colors use karo
      if (dispersionOn) {
        for (let i = 0; i < n; i++) {
          const angle = n === 1 ? srcAngle : startAngle + (i / (n - 1)) * totalFan;
          const dx = Math.cos(angle);
          const dy = Math.sin(angle);
          // har ray ke liye sab colors trace karo
          for (const spec of SPECTRUM_COLORS) {
            const segs = traceRay(lightSource.x, lightSource.y, dx, dy, MAX_BOUNCES, spec.color, 1.0);
            cachedRays.push(...segs);
          }
        }
      } else {
        for (let i = 0; i < n; i++) {
          const angle = n === 1 ? srcAngle : startAngle + (i / (n - 1)) * totalFan;
          const dx = Math.cos(angle);
          const dy = Math.sin(angle);
          const segs = traceRay(lightSource.x, lightSource.y, dx, dy, MAX_BOUNCES, '#ffffff', 1.0);
          cachedRays.push(...segs);
        }
      }
    } else {
      // parallel beam — sab rays same direction mein, evenly spaced
      const beamWidth = 120; // beam ki total width
      const perpAngle = srcAngle + Math.PI / 2;
      const perpDx = Math.cos(perpAngle);
      const perpDy = Math.sin(perpAngle);
      const dirDx = Math.cos(srcAngle);
      const dirDy = Math.sin(srcAngle);

      if (dispersionOn) {
        for (let i = 0; i < n; i++) {
          const offset = n === 1 ? 0 : -beamWidth / 2 + (i / (n - 1)) * beamWidth;
          const ox = lightSource.x + offset * perpDx;
          const oy = lightSource.y + offset * perpDy;
          for (const spec of SPECTRUM_COLORS) {
            const segs = traceRay(ox, oy, dirDx, dirDy, MAX_BOUNCES, spec.color, 1.0);
            cachedRays.push(...segs);
          }
        }
      } else {
        for (let i = 0; i < n; i++) {
          const offset = n === 1 ? 0 : -beamWidth / 2 + (i / (n - 1)) * beamWidth;
          const ox = lightSource.x + offset * perpDx;
          const oy = lightSource.y + offset * perpDy;
          const segs = traceRay(ox, oy, dirDx, dirDy, MAX_BOUNCES, '#ffffff', 1.0);
          cachedRays.push(...segs);
        }
      }
    }
  }

  // ============================
  // DRAWING — canvas pe sab render karo
  // ============================

  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // background — dark with subtle grid
    ctx.fillStyle = 'rgba(2,2,8,0.95)';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // grid
    ctx.strokeStyle = 'rgba(245,158,11,0.04)';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < canvasW; x += GRID_SPACING) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvasH);
      ctx.stroke();
    }
    for (let y = 0; y < canvasH; y += GRID_SPACING) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasW, y);
      ctx.stroke();
    }

    // --- Rays draw karo ---
    drawRays();

    // --- Elements draw karo ---
    for (let i = 0; i < elements.length; i++) {
      drawElement(elements[i], i === hoveredElement);
    }

    // --- Light source draw karo ---
    drawLightSource();
  }

  // rays draw karo — glow effect ke saath
  function drawRays() {
    // pehle glow layer (blur ke saath)
    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    for (const seg of cachedRays) {
      ctx.strokeStyle = seg.color || '#ffffff';
      ctx.beginPath();
      ctx.moveTo(seg.x1, seg.y1);
      ctx.lineTo(seg.x2, seg.y2);
      ctx.stroke();
    }
    ctx.restore();

    // fir sharp line
    ctx.save();
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';
    ctx.globalAlpha = 0.85;
    for (const seg of cachedRays) {
      ctx.strokeStyle = seg.color || '#ffffff';
      ctx.beginPath();
      ctx.moveTo(seg.x1, seg.y1);
      ctx.lineTo(seg.x2, seg.y2);
      ctx.stroke();
    }
    ctx.restore();

    // focal point glow — jahan rays converge hoti hain
    detectAndDrawFocalGlow();
  }

  // focal point detection — rays kahan converge ho rahi hain
  function detectAndDrawFocalGlow() {
    if (cachedRays.length < 4) return;

    // ray endpoints collect karo jo canvas ke andar hain
    // cluster nearby endpoints
    const endPoints = [];
    for (const seg of cachedRays) {
      // last segment ka end point — agar canvas ke andar hai
      if (seg.x2 >= 0 && seg.x2 <= canvasW && seg.y2 >= 0 && seg.y2 <= canvasH) {
        // skip — ye toh last endpoint hai, hume intermediate convergence chahiye
      }
    }

    // better approach — ray segment pairs ke intersection points dhundho
    // but ye expensive hai, toh simple clustering karte hain hit points ka
    const hitPoints = [];
    for (const seg of cachedRays) {
      // agar segment short hai (meaning it hit something), add endpoint
      const len = vecLen(seg.x2 - seg.x1, seg.y2 - seg.y1);
      if (len < RAY_LENGTH * 0.5) {
        hitPoints.push([seg.x2, seg.y2]);
      }
    }

    if (hitPoints.length < 3) return;

    // simple clustering — nearby points group karo
    const clusters = [];
    const used = new Set();

    for (let i = 0; i < hitPoints.length; i++) {
      if (used.has(i)) continue;
      let cx = hitPoints[i][0], cy = hitPoints[i][1];
      let count = 1;
      used.add(i);

      for (let j = i + 1; j < hitPoints.length; j++) {
        if (used.has(j)) continue;
        const d = vecLen(hitPoints[j][0] - cx / count, hitPoints[j][1] - cy / count);
        if (d < 30) {
          cx += hitPoints[j][0];
          cy += hitPoints[j][1];
          count++;
          used.add(j);
        }
      }

      if (count >= 3) {
        clusters.push({ x: cx / count, y: cy / count, count });
      }
    }

    // glow draw karo har cluster pe
    for (const cl of clusters) {
      const intensity = Math.min(cl.count / 8, 1.0);
      const grad = ctx.createRadialGradient(cl.x, cl.y, 0, cl.x, cl.y, 20 + cl.count * 2);
      grad.addColorStop(0, `rgba(${ACCENT_RGB},${0.5 * intensity})`);
      grad.addColorStop(0.5, `rgba(${ACCENT_RGB},${0.15 * intensity})`);
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = grad;
      ctx.fillRect(cl.x - 30, cl.y - 30, 60, 60);
    }
  }

  // light source draw karo — glowing circle with direction indicator
  function drawLightSource() {
    const sx = lightSource.x;
    const sy = lightSource.y;

    // glow
    const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, 25);
    grad.addColorStop(0, 'rgba(255,255,200,0.6)');
    grad.addColorStop(0.5, 'rgba(255,200,50,0.2)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(sx - 30, sy - 30, 60, 60);

    // circle
    ctx.beginPath();
    ctx.arc(sx, sy, 10, 0, Math.PI * 2);
    ctx.fillStyle = '#ffdd44';
    ctx.fill();
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.stroke();

    // direction arrow — light source ki direction dikhao
    const arrowLen = 20;
    const ax = sx + arrowLen * Math.cos(lightSource.angle);
    const ay = sy + arrowLen * Math.sin(lightSource.angle);
    ctx.beginPath();
    ctx.moveTo(sx + 10 * Math.cos(lightSource.angle), sy + 10 * Math.sin(lightSource.angle));
    ctx.lineTo(ax, ay);
    ctx.strokeStyle = '#ffdd44';
    ctx.lineWidth = 2;
    ctx.stroke();

    // arrowhead
    const headLen = 6;
    const headAng = 0.5;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - headLen * Math.cos(lightSource.angle - headAng), ay - headLen * Math.sin(lightSource.angle - headAng));
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - headLen * Math.cos(lightSource.angle + headAng), ay - headLen * Math.sin(lightSource.angle + headAng));
    ctx.stroke();

    // point source indicator — fan ka outline dikhao
    if (lightSource.isPointSource) {
      const fanHalf = lightSource.fanAngle / 2;
      ctx.beginPath();
      ctx.arc(sx, sy, 18, lightSource.angle - fanHalf, lightSource.angle + fanHalf);
      ctx.strokeStyle = 'rgba(255,221,68,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  // individual element draw karo
  function drawElement(elem, isHovered) {
    ctx.save();

    const highlight = isHovered ? 1.0 : 0.7;

    if (elem.type === 'flat-mirror') {
      const seg = getFlatMirrorSegment(elem);
      // mirror surface — reflective silver line
      ctx.beginPath();
      ctx.moveTo(seg.x1, seg.y1);
      ctx.lineTo(seg.x2, seg.y2);
      ctx.strokeStyle = `rgba(180,200,220,${highlight})`;
      ctx.lineWidth = isHovered ? 4 : 3;
      ctx.stroke();

      // hatching — mirror ke peeche ki lines (mirror convention)
      const backNx = -seg.nx;
      const backNy = -seg.ny;
      const hatchLen = 8;
      const hatchCount = 8;
      ctx.strokeStyle = `rgba(120,140,160,${highlight * 0.5})`;
      ctx.lineWidth = 1;
      for (let i = 0; i <= hatchCount; i++) {
        const t = i / hatchCount;
        const px = seg.x1 + t * (seg.x2 - seg.x1);
        const py = seg.y1 + t * (seg.y2 - seg.y1);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + backNx * hatchLen, py + backNy * hatchLen);
        ctx.stroke();
      }
    }

    else if (elem.type === 'curved-mirror') {
      const arc = getCurvedMirrorArc(elem);
      // curved mirror — arc draw karo
      ctx.beginPath();
      ctx.arc(arc.cx, arc.cy, arc.radius, arc.startAngle, arc.endAngle);
      ctx.strokeStyle = `rgba(180,200,220,${highlight})`;
      ctx.lineWidth = isHovered ? 4 : 3;
      ctx.stroke();

      // concave/convex indicator
      const midAngle = (arc.startAngle + arc.endAngle) / 2;
      const indX = arc.cx + arc.radius * Math.cos(midAngle);
      const indY = arc.cy + arc.radius * Math.sin(midAngle);
      ctx.fillStyle = `rgba(180,200,220,${highlight * 0.5})`;
      ctx.font = '10px "JetBrains Mono",monospace';
      ctx.textAlign = 'center';
      ctx.fillText(elem.isConcave ? 'C' : 'V', indX, indY - 8);
    }

    else if (elem.type === 'convex-lens') {
      // double convex lens shape — )( jaisi shape
      const seg = getLensSegment(elem);
      const halfH = elem.height / 2;

      ctx.save();
      ctx.translate(elem.x, elem.y);
      ctx.rotate(elem.angle + Math.PI / 2);

      // left surface — convex curve
      const bulge = 12;
      ctx.beginPath();
      ctx.moveTo(0, -halfH);
      ctx.quadraticCurveTo(-bulge, 0, 0, halfH);
      ctx.strokeStyle = `rgba(100,180,255,${highlight})`;
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.stroke();

      // right surface — convex curve
      ctx.beginPath();
      ctx.moveTo(0, -halfH);
      ctx.quadraticCurveTo(bulge, 0, 0, halfH);
      ctx.stroke();

      // fill — translucent glass
      ctx.beginPath();
      ctx.moveTo(0, -halfH);
      ctx.quadraticCurveTo(-bulge, 0, 0, halfH);
      ctx.quadraticCurveTo(bulge, 0, 0, -halfH);
      ctx.closePath();
      ctx.fillStyle = `rgba(100,180,255,${0.1 * highlight})`;
      ctx.fill();

      // arrows on top and bottom — converging indication
      ctx.strokeStyle = `rgba(100,180,255,${highlight * 0.5})`;
      ctx.lineWidth = 1;
      // top arrow pointing inward
      ctx.beginPath();
      ctx.moveTo(-6, -halfH - 4);
      ctx.lineTo(0, -halfH);
      ctx.lineTo(6, -halfH - 4);
      ctx.stroke();
      // bottom arrow
      ctx.beginPath();
      ctx.moveTo(-6, halfH + 4);
      ctx.lineTo(0, halfH);
      ctx.lineTo(6, halfH + 4);
      ctx.stroke();

      ctx.restore();
    }

    else if (elem.type === 'concave-lens') {
      // double concave lens shape — )( flipped
      const halfH = elem.height / 2;

      ctx.save();
      ctx.translate(elem.x, elem.y);
      ctx.rotate(elem.angle + Math.PI / 2);

      const bulge = 12;
      // left surface — concave
      ctx.beginPath();
      ctx.moveTo(0, -halfH);
      ctx.quadraticCurveTo(bulge, 0, 0, halfH);
      ctx.strokeStyle = `rgba(100,180,255,${highlight})`;
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.stroke();

      // right surface — concave
      ctx.beginPath();
      ctx.moveTo(0, -halfH);
      ctx.quadraticCurveTo(-bulge, 0, 0, halfH);
      ctx.stroke();

      // fill
      ctx.beginPath();
      ctx.moveTo(0, -halfH);
      ctx.quadraticCurveTo(bulge, 0, 0, halfH);
      ctx.quadraticCurveTo(-bulge, 0, 0, -halfH);
      ctx.closePath();
      ctx.fillStyle = `rgba(100,180,255,${0.1 * highlight})`;
      ctx.fill();

      // diverging arrows
      ctx.strokeStyle = `rgba(100,180,255,${highlight * 0.5})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(-4, -halfH);
      ctx.lineTo(0, -halfH - 5);
      ctx.lineTo(4, -halfH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(-4, halfH);
      ctx.lineTo(0, halfH + 5);
      ctx.lineTo(4, halfH);
      ctx.stroke();

      ctx.restore();
    }

    else if (elem.type === 'prism') {
      const verts = getPrismVertices(elem);

      // prism triangle draw karo
      ctx.beginPath();
      ctx.moveTo(verts[0][0], verts[0][1]);
      ctx.lineTo(verts[1][0], verts[1][1]);
      ctx.lineTo(verts[2][0], verts[2][1]);
      ctx.closePath();

      // fill — translucent glass
      ctx.fillStyle = `rgba(140,120,255,${0.12 * highlight})`;
      ctx.fill();
      ctx.strokeStyle = `rgba(140,120,255,${highlight})`;
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.stroke();

      // dispersion indicator agar on hai
      if (dispersionOn) {
        ctx.fillStyle = `rgba(140,120,255,${highlight * 0.5})`;
        ctx.font = '9px "JetBrains Mono",monospace';
        ctx.textAlign = 'center';
        ctx.fillText('n=' + elem.refractiveIndex.toFixed(2), elem.x, elem.y + 3);
      }
    }

    ctx.restore();
  }

  // ============================
  // INTERACTION — mouse/touch handling
  // ============================

  // canvas coordinates nikalo event se
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  // nearest element dhundho position se
  function findNearestElement(mx, my) {
    let bestIdx = -1;
    let bestDist = ELEMENT_HIT_RADIUS;

    for (let i = 0; i < elements.length; i++) {
      const elem = elements[i];
      const d = vecLen(mx - elem.x, my - elem.y);
      // prism ke liye thoda bada radius — kyunki prism bada hota hai
      const hitR = elem.type === 'prism' ? elem.size * 0.6 : ELEMENT_HIT_RADIUS + 10;
      if (d < hitR && d < bestDist + 30) {
        bestDist = d;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  // check karo light source pe click hua hai
  function isOnLightSource(mx, my) {
    return vecLen(mx - lightSource.x, my - lightSource.y) < 15;
  }

  // --- Mouse Events ---
  let mouseDownTime = 0;
  let mouseDownPos = [0, 0];
  let rotatingElement = -1; // wheel se rotate ho raha element

  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const [mx, my] = getCanvasPos(e);
    mouseDownTime = Date.now();
    mouseDownPos = [mx, my];

    // pehle check karo light source pe click hua hai
    if (isOnLightSource(mx, my)) {
      dragTarget = 'source';
      dragOffsetX = mx - lightSource.x;
      dragOffsetY = my - lightSource.y;
      isDragging = true;
      return;
    }

    // fir check karo koi element pe click hua hai
    const elemIdx = findNearestElement(mx, my);
    if (elemIdx >= 0 && !selectedTool) {
      dragTarget = elemIdx;
      dragOffsetX = mx - elements[elemIdx].x;
      dragOffsetY = my - elements[elemIdx].y;
      isDragging = true;
      return;
    }

    // agar tool selected hai toh new element place karo
    if (selectedTool) {
      const newElem = createElement(selectedTool, mx, my);
      if (newElem) {
        elements.push(newElem);
        traceAllRays();
        needsRedraw = true;
      }
      return;
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    const [mx, my] = getCanvasPos(e);

    if (isDragging) {
      if (dragTarget === 'source') {
        lightSource.x = mx - dragOffsetX;
        lightSource.y = my - dragOffsetY;
        traceAllRays();
        needsRedraw = true;
      } else if (typeof dragTarget === 'number') {
        elements[dragTarget].x = mx - dragOffsetX;
        elements[dragTarget].y = my - dragOffsetY;
        traceAllRays();
        needsRedraw = true;
      }
      return;
    }

    // hover effect — kaunsa element hover ho raha hai
    const newHover = findNearestElement(mx, my);
    if (newHover !== hoveredElement) {
      hoveredElement = newHover;
      needsRedraw = true;
    }

    // cursor update
    if (selectedTool) {
      canvas.style.cursor = 'copy';
    } else if (isOnLightSource(mx, my) || newHover >= 0) {
      canvas.style.cursor = 'grab';
    } else {
      canvas.style.cursor = 'crosshair';
    }
  });

  canvas.addEventListener('mouseup', (e) => {
    isDragging = false;
    dragTarget = null;
  });

  canvas.addEventListener('mouseleave', () => {
    isDragging = false;
    dragTarget = null;
    hoveredElement = -1;
    needsRedraw = true;
  });

  // double click — element delete karo
  canvas.addEventListener('dblclick', (e) => {
    e.preventDefault();
    const [mx, my] = getCanvasPos(e);
    const elemIdx = findNearestElement(mx, my);
    if (elemIdx >= 0) {
      elements.splice(elemIdx, 1);
      hoveredElement = -1;
      traceAllRays();
      needsRedraw = true;
    }
  });

  // scroll/wheel — element rotate karo
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const [mx, my] = getCanvasPos(e);

    // pehle check karo light source pe hai
    if (isOnLightSource(mx, my)) {
      lightSource.angle += e.deltaY * 0.005;
      traceAllRays();
      needsRedraw = true;
      return;
    }

    // fir element dhundho
    const elemIdx = findNearestElement(mx, my);
    if (elemIdx >= 0) {
      elements[elemIdx].angle += e.deltaY * 0.005;
      traceAllRays();
      needsRedraw = true;
    }
  }, { passive: false });

  // --- Touch Events — mobile support ---
  let touchId = null;
  let touchStartTime = 0;
  let touchStartPos = [0, 0];
  let touchMoved = false;
  let lastTouchDist = 0; // pinch ke liye

  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (touchId !== null) return;
    const touch = e.changedTouches[0];
    touchId = touch.identifier;
    const [mx, my] = getCanvasPos(touch);
    touchStartTime = Date.now();
    touchStartPos = [mx, my];
    touchMoved = false;

    // light source check
    if (isOnLightSource(mx, my)) {
      dragTarget = 'source';
      dragOffsetX = mx - lightSource.x;
      dragOffsetY = my - lightSource.y;
      isDragging = true;
      return;
    }

    // element check
    const elemIdx = findNearestElement(mx, my);
    if (elemIdx >= 0 && !selectedTool) {
      dragTarget = elemIdx;
      dragOffsetX = mx - elements[elemIdx].x;
      dragOffsetY = my - elements[elemIdx].y;
      isDragging = true;
      return;
    }

    // tool selected — place element
    if (selectedTool) {
      const newElem = createElement(selectedTool, mx, my);
      if (newElem) {
        elements.push(newElem);
        traceAllRays();
        needsRedraw = true;
      }
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const touch = Array.from(e.changedTouches).find(t => t.identifier === touchId);
    if (!touch) return;
    const [mx, my] = getCanvasPos(touch);
    touchMoved = true;

    if (isDragging) {
      if (dragTarget === 'source') {
        lightSource.x = mx - dragOffsetX;
        lightSource.y = my - dragOffsetY;
      } else if (typeof dragTarget === 'number') {
        elements[dragTarget].x = mx - dragOffsetX;
        elements[dragTarget].y = my - dragOffsetY;
      }
      traceAllRays();
      needsRedraw = true;
    }

    // 2 finger rotation — pinch se element rotate karo
    if (e.touches.length === 2) {
      const t1 = e.touches[0];
      const t2 = e.touches[1];
      const dist = vecLen(t1.clientX - t2.clientX, t1.clientY - t2.clientY);
      if (lastTouchDist > 0) {
        // rotation by distance change isn't ideal, but works for basic rotation
        const midX = (getCanvasPos(t1)[0] + getCanvasPos(t2)[0]) / 2;
        const midY = (getCanvasPos(t1)[1] + getCanvasPos(t2)[1]) / 2;
        const elemIdx = findNearestElement(midX, midY);
        if (elemIdx >= 0) {
          const delta = (dist - lastTouchDist) * 0.01;
          elements[elemIdx].angle += delta;
          traceAllRays();
          needsRedraw = true;
        }
      }
      lastTouchDist = dist;
    }
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    const touch = Array.from(e.changedTouches).find(t => t.identifier === touchId);
    if (!touch) return;

    const [mx, my] = getCanvasPos(touch);

    // double tap delete — long press se delete karo
    if (!touchMoved && isDragging && typeof dragTarget === 'number') {
      const holdTime = Date.now() - touchStartTime;
      if (holdTime > 600) {
        elements.splice(dragTarget, 1);
        hoveredElement = -1;
        traceAllRays();
        needsRedraw = true;
      }
    }

    isDragging = false;
    dragTarget = null;
    touchId = null;
    lastTouchDist = 0;
  }, { passive: false });

  // ============================
  // PRESETS — ready-made configurations
  // ============================

  function loadPreset(name) {
    elements = [];
    // dispersion toggle bhi set karo based on preset

    switch (name) {
      case 'Rainbow': {
        // single prism with dispersion — spectrum dikhega
        dispersionOn = true;
        dispersionToggle.setActive(true);

        lightSource.x = 80;
        lightSource.y = canvasH / 2;
        lightSource.angle = 0;
        lightSource.isPointSource = false;
        lightSource.rayCount = 8;
        beamToggle.setActive(false);
        raysSliderCtrl.slider.value = '8';
        raysSliderCtrl.label.textContent = 'Rays: 8';

        elements.push(createPrism(canvasW * 0.4, canvasH / 2, 0));
        break;
      }

      case 'Telescope': {
        // 2 convex lenses — Keplerian telescope setup
        dispersionOn = false;
        dispersionToggle.setActive(false);

        lightSource.x = 60;
        lightSource.y = canvasH / 2;
        lightSource.angle = 0;
        lightSource.isPointSource = false;
        lightSource.rayCount = 10;
        beamToggle.setActive(false);
        raysSliderCtrl.slider.value = '10';
        raysSliderCtrl.label.textContent = 'Rays: 10';

        // objective lens — bada focal length
        const obj = createConvexLens(canvasW * 0.35, canvasH / 2, 0);
        obj.focalLength = 140;
        obj.height = 100;
        elements.push(obj);

        // eyepiece lens — chhota focal length
        const eye = createConvexLens(canvasW * 0.65, canvasH / 2, 0);
        eye.focalLength = 60;
        eye.height = 60;
        elements.push(eye);
        break;
      }

      case 'Periscope': {
        // 2 flat mirrors at 45 degrees — classic periscope
        dispersionOn = false;
        dispersionToggle.setActive(false);

        lightSource.x = 60;
        lightSource.y = canvasH * 0.25;
        lightSource.angle = 0;
        lightSource.isPointSource = false;
        lightSource.rayCount = 8;
        beamToggle.setActive(false);
        raysSliderCtrl.slider.value = '8';
        raysSliderCtrl.label.textContent = 'Rays: 8';

        // top mirror — 45 degree angle, ray ko neeche bhejega
        const mirror1 = createFlatMirror(canvasW * 0.45, canvasH * 0.25, Math.PI / 4);
        mirror1.length = 100;
        elements.push(mirror1);

        // bottom mirror — 45 degree angle, ray ko seedha bhejega
        const mirror2 = createFlatMirror(canvasW * 0.45, canvasH * 0.75, Math.PI / 4);
        mirror2.length = 100;
        elements.push(mirror2);
        break;
      }

      default:
        return;
    }

    traceAllRays();
    needsRedraw = true;
  }

  // ============================
  // ANIMATION LOOP — requestAnimationFrame
  // ============================

  function loop() {
    // lab pause check — sirf active sim animate ho
    if (window.__labPaused && window.__labPaused !== container.id) {
      animationId = null;
      return;
    }
    if (!isVisible) {
      animationId = null;
      return;
    }

    // sirf redraw jab zarurat ho — performance ke liye
    if (needsRedraw) {
      draw();
      needsRedraw = false;
    }

    animationId = requestAnimationFrame(loop);
  }

  // ============================
  // INTERSECTION OBSERVER — sirf visible hone pe animate karo
  // ============================
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) {
      resize();
      needsRedraw = true;
      loop();
    } else if (!isVisible && animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }, { threshold: 0.1 });
  obs.observe(container);

  // lab resume: jab focus wapas aaye toh loop restart karo
  document.addEventListener('lab:resume', () => {
    if (isVisible && !animationId) loop();
  });

  // tab visibility change — tab switch pe pause/resume
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    } else {
      if (isVisible && !animationId) loop();
    }
  });

  // --- Rainbow preset se start karo — default mein kuch dikhna chahiye ---
  loadPreset('Rainbow');
  traceAllRays();
  needsRedraw = true;
}
