// ============================================================
// 2D Rigid Body Physics Engine — SAT collision detection + impulse resolution
// Click se shapes spawn karo, drag se phenko, gravity mein girte dekho
// Box, Triangle, Pentagon, Hexagon, Circle — sab ke sab with proper rotation
// ============================================================

// yahi function bahar export hoga — container dhundho, canvas banao, physics chalao
export function initRigidBody() {
  const container = document.getElementById('rigidBodyContainer');
  if (!container) {
    console.warn('rigidBodyContainer nahi mila bhai, rigid body demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  const ACCENT_RGB = '245,158,11';
  const MAX_BODIES = 30;
  const FLOOR_HEIGHT = 30; // floor ki thickness
  const DENSITY = 0.002; // mass = area * density
  const SUBSTEPS = 6; // physics substeps per frame — stability ke liye
  const POSITIONAL_CORRECTION = 0.4; // sinking fix karne ka factor
  const SLOP = 0.5; // itna penetration allow hai bina correction ke

  // shape colors — har shape type ka alag color
  const SHAPE_COLORS = [
    { r: 245, g: 158, b: 11 },   // amber — box
    { r: 96, g: 165, b: 250 },   // blue — triangle
    { r: 74, g: 222, b: 128 },   // green — pentagon
    { r: 192, g: 132, b: 252 },  // purple — hexagon
    { r: 251, g: 146, b: 60 },   // orange — circle
  ];

  // --- State ---
  let canvasW = 0, canvasH = 0, dpr = 1;
  let bodies = []; // rigid body objects
  let gravity = 9.8;
  let restitution = 0.4;
  let friction = 0.3;
  let showVelocity = false;
  let selectedShape = 'box'; // dropdown se selected shape
  let animationId = null;
  let isVisible = false;
  let lastTime = 0;

  // mouse/touch state — spawn aur drag ke liye
  let isMouseDown = false;
  let mouseDownX = 0, mouseDownY = 0;
  let mouseCurrentX = 0, mouseCurrentY = 0;
  let isDragging = false;

  // --- DOM structure banao — pehle sab saaf karo ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // canvas
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

  // stats bar — body count dikhane ke liye
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'margin-top:8px',
    'padding:6px 12px',
    'background:rgba(' + ACCENT_RGB + ',0.05)',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.12)',
    'border-radius:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:#b0b0b0',
    'display:flex',
    'flex-wrap:wrap',
    'gap:16px',
    'align-items:center',
  ].join(';');
  container.appendChild(statsDiv);

  const bodyCountSpan = document.createElement('span');
  statsDiv.appendChild(bodyCountSpan);

  const energySpan = document.createElement('span');
  statsDiv.appendChild(energySpan);

  function updateStats() {
    bodyCountSpan.textContent = 'Bodies: ' + bodies.length + '/' + MAX_BODIES;
    // total kinetic energy — linear + rotational
    let ke = 0;
    for (const b of bodies) {
      if (b.isStatic) continue;
      ke += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy);
      ke += 0.5 * b.inertia * b.angularVel * b.angularVel;
    }
    energySpan.textContent = 'KE: ' + ke.toFixed(1);
  }

  // --- Controls section ---
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
      'padding:6px 14px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // --- Helper: slider banao ---
  function createSlider(label, min, max, step, initial, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(initial);
    slider.style.cssText = 'width:70px;height:4px;accent-color:rgba(' + ACCENT_RGB + ',0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const val = document.createElement('span');
    val.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:"JetBrains Mono",monospace;min-width:32px;';
    val.textContent = String(initial);
    wrapper.appendChild(val);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      onChange(v, val);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, val };
  }

  // --- Helper: dropdown banao ---
  function createDropdown(label, options, initial, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const select = document.createElement('select');
    select.style.cssText = [
      'padding:4px 8px',
      'font-size:12px',
      'border-radius:5px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:"JetBrains Mono",monospace',
    ].join(';');

    options.forEach(opt => {
      const option = document.createElement('option');
      option.value = opt.toLowerCase();
      option.textContent = opt;
      option.style.background = '#1a1a1a';
      option.style.color = '#b0b0b0';
      if (opt.toLowerCase() === initial.toLowerCase()) option.selected = true;
      select.appendChild(option);
    });

    select.addEventListener('change', () => onChange(select.value));
    wrapper.appendChild(select);
    controlsDiv.appendChild(wrapper);
    return select;
  }

  // --- Controls banao ---

  // shape selector dropdown
  createDropdown('Shape:', ['Box', 'Triangle', 'Pentagon', 'Hexagon', 'Circle'], 'Box', (v) => {
    selectedShape = v;
  });

  // gravity slider — 0 to 20
  createSlider('g:', 0, 20, 0.1, 9.8, (v, valEl) => {
    gravity = v;
    valEl.textContent = v.toFixed(1);
  });

  // restitution slider — 0 to 1
  createSlider('e:', 0, 1, 0.05, 0.4, (v, valEl) => {
    restitution = v;
    valEl.textContent = v.toFixed(2);
  });

  // friction slider — 0 to 1
  createSlider('\u00b5:', 0, 1, 0.05, 0.3, (v, valEl) => {
    friction = v;
    valEl.textContent = v.toFixed(2);
  });

  // velocity toggle button
  const velBtn = createButton('Vectors', () => {
    showVelocity = !showVelocity;
    velBtn.style.color = showVelocity ? ACCENT : '#b0b0b0';
    velBtn.style.borderColor = showVelocity ? 'rgba(' + ACCENT_RGB + ',0.5)' : 'rgba(' + ACCENT_RGB + ',0.25)';
  });

  // clear all button
  createButton('Clear', () => {
    bodies = bodies.filter(b => b.isStatic);
  });

  // tower preset button
  createButton('Tower', () => {
    loadTowerPreset();
  });

  // avalanche preset button
  createButton('Avalanche', () => {
    loadAvalanchePreset();
  });

  // ============================================================
  // CANVAS SIZING — DPR-aware crisp rendering
  // ============================================================

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
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // ============================================================
  // SHAPE GEOMETRY — convex polygon vertices banao
  // ============================================================

  // regular polygon ke vertices return karo — center (0,0) ke around
  function regularPolygonVertices(sides, radius) {
    const verts = [];
    for (let i = 0; i < sides; i++) {
      // top se shuru karo (- PI/2 offset) — natural dikhta hai
      const angle = (2 * Math.PI * i / sides) - Math.PI / 2;
      verts.push({ x: Math.cos(angle) * radius, y: Math.sin(angle) * radius });
    }
    return verts;
  }

  // box vertices — width x height ka rectangle, center (0,0)
  function boxVertices(w, h) {
    const hw = w / 2, hh = h / 2;
    return [
      { x: -hw, y: -hh },
      { x: hw, y: -hh },
      { x: hw, y: hh },
      { x: -hw, y: hh },
    ];
  }

  // polygon ka area nikaalo — shoelace formula
  function polygonArea(verts) {
    let area = 0;
    const n = verts.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += verts[i].x * verts[j].y;
      area -= verts[j].x * verts[i].y;
    }
    return Math.abs(area) / 2;
  }

  // polygon ka moment of inertia nikaalo — center ke around
  // formula: I = (1/12) * sum over edges of (cross^2 + cross*(dot of consecutive)) * |cross|
  function polygonInertia(verts, mass) {
    let numerator = 0;
    let denominator = 0;
    const n = verts.length;
    for (let i = 0; i < n; i++) {
      const v1 = verts[i];
      const v2 = verts[(i + 1) % n];
      const cross = Math.abs(v1.x * v2.y - v2.x * v1.y);
      numerator += cross * (
        v1.x * v1.x + v1.x * v2.x + v2.x * v2.x +
        v1.y * v1.y + v1.y * v2.y + v2.y * v2.y
      );
      denominator += cross;
    }
    if (denominator < 0.0001) return mass * 100; // fallback
    return (mass / 6) * (numerator / denominator);
  }

  // ============================================================
  // RIGID BODY CREATION
  // ============================================================

  let bodyIdCounter = 0;

  // convex polygon body banao
  function createPolygonBody(x, y, vertices, colorIdx, angleInit) {
    const area = polygonArea(vertices);
    const mass = area * DENSITY;
    const inertia = polygonInertia(vertices, mass);
    const angle = angleInit || 0;

    return {
      id: bodyIdCounter++,
      type: 'polygon',
      x: x, y: y,           // center of mass position
      vx: 0, vy: 0,         // linear velocity
      angle: angle,          // rotation angle (radians)
      angularVel: 0,         // angular velocity (rad/s)
      mass: mass,
      invMass: 1 / mass,
      inertia: inertia,
      invInertia: 1 / inertia,
      localVertices: vertices, // local space mein vertices (center ke around)
      restitution: restitution,
      friction: friction,
      isStatic: false,
      color: SHAPE_COLORS[colorIdx % SHAPE_COLORS.length],
    };
  }

  // circle body banao
  function createCircleBody(x, y, radius, colorIdx, angleInit) {
    const area = Math.PI * radius * radius;
    const mass = area * DENSITY;
    // circle ka inertia = (1/2) * m * r^2
    const inertia = 0.5 * mass * radius * radius;
    const angle = angleInit || 0;

    return {
      id: bodyIdCounter++,
      type: 'circle',
      x: x, y: y,
      vx: 0, vy: 0,
      angle: angle,
      angularVel: 0,
      mass: mass,
      invMass: 1 / mass,
      inertia: inertia,
      invInertia: 1 / inertia,
      radius: radius,
      restitution: restitution,
      friction: friction,
      isStatic: false,
      color: SHAPE_COLORS[colorIdx % SHAPE_COLORS.length],
    };
  }

  // static floor body banao — bahut badi mass, hilta nahi
  function createFloor() {
    const floorY = canvasH - FLOOR_HEIGHT / 2;
    const hw = canvasW / 2 + 50; // thoda extra taaki edges na dikhe
    const hh = FLOOR_HEIGHT / 2;
    const verts = [
      { x: -hw, y: -hh },
      { x: hw, y: -hh },
      { x: hw, y: hh },
      { x: -hw, y: hh },
    ];

    const body = {
      id: bodyIdCounter++,
      type: 'polygon',
      x: canvasW / 2, y: floorY,
      vx: 0, vy: 0,
      angle: 0,
      angularVel: 0,
      mass: 0, // infinite mass
      invMass: 0,
      inertia: 0,
      invInertia: 0,
      localVertices: verts,
      restitution: 0.3,
      friction: 0.5,
      isStatic: true,
      color: { r: 100, g: 100, b: 100 },
    };
    return body;
  }

  // body ke world-space vertices nikaalo — rotation apply karke
  function getWorldVertices(body) {
    if (body.type === 'circle') return []; // circle ke vertices nahi hote
    const cos = Math.cos(body.angle);
    const sin = Math.sin(body.angle);
    return body.localVertices.map(v => ({
      x: body.x + v.x * cos - v.y * sin,
      y: body.y + v.x * sin + v.y * cos,
    }));
  }

  // shape type se body banao — selected shape ke hisaab se
  function spawnShape(x, y, shapeType, vx, vy) {
    // max bodies check — purane hata do agar limit cross ho
    const dynamicBodies = bodies.filter(b => !b.isStatic);
    if (dynamicBodies.length >= MAX_BODIES) {
      // sabse purana dynamic body hata do
      const oldest = dynamicBodies[0];
      bodies = bodies.filter(b => b.id !== oldest.id);
    }

    let body;
    const angle = (Math.random() - 0.5) * 0.5; // slight random rotation

    switch (shapeType) {
      case 'box': {
        const w = 30 + Math.random() * 20;
        const h = 25 + Math.random() * 15;
        body = createPolygonBody(x, y, boxVertices(w, h), 0, angle);
        break;
      }
      case 'triangle': {
        const r = 20 + Math.random() * 12;
        body = createPolygonBody(x, y, regularPolygonVertices(3, r), 1, angle);
        break;
      }
      case 'pentagon': {
        const r = 18 + Math.random() * 10;
        body = createPolygonBody(x, y, regularPolygonVertices(5, r), 2, angle);
        break;
      }
      case 'hexagon': {
        const r = 18 + Math.random() * 10;
        body = createPolygonBody(x, y, regularPolygonVertices(6, r), 3, angle);
        break;
      }
      case 'circle': {
        const r = 15 + Math.random() * 10;
        body = createCircleBody(x, y, r, 4, angle);
        break;
      }
      default: {
        const w = 30 + Math.random() * 20;
        const h = 25 + Math.random() * 15;
        body = createPolygonBody(x, y, boxVertices(w, h), 0, angle);
        break;
      }
    }

    body.vx = vx || 0;
    body.vy = vy || 0;
    body.restitution = restitution;
    body.friction = friction;
    bodies.push(body);
    return body;
  }

  // ============================================================
  // SAT COLLISION DETECTION — Separating Axis Theorem
  // ============================================================

  // polygon ke edge normals nikaalo — ye potential separating axes hain
  function getAxes(worldVerts) {
    const axes = [];
    const n = worldVerts.length;
    for (let i = 0; i < n; i++) {
      const v1 = worldVerts[i];
      const v2 = worldVerts[(i + 1) % n];
      // edge vector
      const edgeX = v2.x - v1.x;
      const edgeY = v2.y - v1.y;
      // perpendicular normal (outward pointing — left normal)
      const len = Math.sqrt(edgeX * edgeX + edgeY * edgeY);
      if (len < 0.0001) continue;
      axes.push({ x: -edgeY / len, y: edgeX / len });
    }
    return axes;
  }

  // polygon ko axis pe project karo — min aur max dot product return karo
  function projectPolygon(worldVerts, axis) {
    let min = Infinity, max = -Infinity;
    for (const v of worldVerts) {
      const dot = v.x * axis.x + v.y * axis.y;
      if (dot < min) min = dot;
      if (dot > max) max = dot;
    }
    return { min, max };
  }

  // circle ko axis pe project karo
  function projectCircle(body, axis) {
    const dot = body.x * axis.x + body.y * axis.y;
    return { min: dot - body.radius, max: dot + body.radius };
  }

  // SAT collision check — polygon vs polygon
  // returns null if no collision, or { normal, depth, contacts } if collision
  function satPolygonPolygon(bodyA, bodyB) {
    const vertsA = getWorldVertices(bodyA);
    const vertsB = getWorldVertices(bodyB);
    const axesA = getAxes(vertsA);
    const axesB = getAxes(vertsB);

    let minOverlap = Infinity;
    let collisionNormal = null;

    // sabhi axes check karo — agar kisi pe bhi gap mile toh collision nahi hai
    const allAxes = axesA.concat(axesB);
    for (const axis of allAxes) {
      const projA = projectPolygon(vertsA, axis);
      const projB = projectPolygon(vertsB, axis);

      // overlap check
      const overlap = Math.min(projA.max - projB.min, projB.max - projA.min);
      if (overlap <= 0) return null; // separating axis mila — no collision

      if (overlap < minOverlap) {
        minOverlap = overlap;
        collisionNormal = { x: axis.x, y: axis.y };
      }
    }

    // normal ko A se B ki taraf point karwao
    const dx = bodyB.x - bodyA.x;
    const dy = bodyB.y - bodyA.y;
    if (dx * collisionNormal.x + dy * collisionNormal.y < 0) {
      collisionNormal.x = -collisionNormal.x;
      collisionNormal.y = -collisionNormal.y;
    }

    // contact points dhundho — closest vertex approach
    const contacts = findContactPoints(vertsA, vertsB, collisionNormal);

    return {
      normal: collisionNormal,
      depth: minOverlap,
      contacts: contacts,
    };
  }

  // contact points dhundho — sabse deep penetrating vertices
  function findContactPoints(vertsA, vertsB, normal) {
    const contacts = [];

    // A ke vertices jo B ke andar hain
    let minDist = Infinity;
    for (const v of vertsA) {
      const d = v.x * normal.x + v.y * normal.y;
      if (d < minDist + 0.5) {
        if (d < minDist - 0.5) {
          contacts.length = 0;
          minDist = d;
        }
        contacts.push({ x: v.x, y: v.y });
      }
    }

    // B ke vertices jo A ke andar hain (opposite normal mein)
    for (const v of vertsB) {
      const d = -(v.x * normal.x + v.y * normal.y);
      if (d < minDist + 0.5) {
        if (d < minDist - 0.5) {
          contacts.length = 0;
          minDist = d;
        }
        contacts.push({ x: v.x, y: v.y });
      }
    }

    // agar bahut zyada contacts mil gaye toh sirf 2 rakh (edge contact)
    if (contacts.length > 2) {
      contacts.length = 2;
    }

    // agar koi contact nahi mila toh fallback — midpoint use karo
    if (contacts.length === 0) {
      contacts.push({
        x: (vertsA[0].x + vertsB[0].x) * 0.5,
        y: (vertsA[0].y + vertsB[0].y) * 0.5,
      });
    }

    return contacts;
  }

  // circle vs polygon collision
  function collideCirclePolygon(circle, polygon) {
    const verts = getWorldVertices(polygon);
    const n = verts.length;

    let minDist = Infinity;
    let collisionNormal = null;
    let contactPoint = null;

    // har edge ke against check karo
    for (let i = 0; i < n; i++) {
      const v1 = verts[i];
      const v2 = verts[(i + 1) % n];

      // edge pe closest point dhundho circle center se
      const edgeX = v2.x - v1.x;
      const edgeY = v2.y - v1.y;
      const edgeLenSq = edgeX * edgeX + edgeY * edgeY;
      let t = ((circle.x - v1.x) * edgeX + (circle.y - v1.y) * edgeY) / edgeLenSq;
      t = Math.max(0, Math.min(1, t));

      const closestX = v1.x + t * edgeX;
      const closestY = v1.y + t * edgeY;

      const dx = circle.x - closestX;
      const dy = circle.y - closestY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < minDist) {
        minDist = dist;
        if (dist > 0.0001) {
          collisionNormal = { x: dx / dist, y: dy / dist };
        } else {
          // degenerate case — edge normal use karo
          const elen = Math.sqrt(edgeLenSq);
          collisionNormal = { x: -edgeY / elen, y: edgeX / elen };
        }
        contactPoint = { x: closestX, y: closestY };
      }
    }

    if (minDist > circle.radius) return null; // no collision

    // check karo circle polygon ke andar toh nahi hai — normal flip karna padega
    // point-in-polygon test simple winding
    let inside = true;
    for (let i = 0; i < n; i++) {
      const v1 = verts[i];
      const v2 = verts[(i + 1) % n];
      const cross = (v2.x - v1.x) * (circle.y - v1.y) - (v2.y - v1.y) * (circle.x - v1.x);
      if (cross > 0) { inside = false; break; }
    }

    let depth;
    if (inside) {
      // circle andar hai — normal reverse karo aur depth adjust karo
      collisionNormal.x = -collisionNormal.x;
      collisionNormal.y = -collisionNormal.y;
      depth = circle.radius + minDist;
    } else {
      depth = circle.radius - minDist;
    }

    return {
      normal: collisionNormal,
      depth: depth,
      contacts: [contactPoint],
    };
  }

  // circle vs circle collision — simple distance check
  function collideCircleCircle(a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const minDist = a.radius + b.radius;

    if (dist >= minDist) return null;
    if (dist < 0.0001) {
      // exactly same position — arbitrary normal
      return {
        normal: { x: 0, y: -1 },
        depth: minDist,
        contacts: [{ x: a.x, y: a.y - a.radius }],
      };
    }

    const normal = { x: dx / dist, y: dy / dist };
    const depth = minDist - dist;
    const contactX = a.x + normal.x * a.radius;
    const contactY = a.y + normal.y * a.radius;

    return {
      normal: normal,
      depth: depth,
      contacts: [{ x: contactX, y: contactY }],
    };
  }

  // do bodies ke beech collision detect karo — type ke hisaab se sahi function call karo
  function detectCollision(bodyA, bodyB) {
    if (bodyA.type === 'circle' && bodyB.type === 'circle') {
      return collideCircleCircle(bodyA, bodyB);
    }
    if (bodyA.type === 'circle' && bodyB.type === 'polygon') {
      return collideCirclePolygon(bodyA, bodyB);
    }
    if (bodyA.type === 'polygon' && bodyB.type === 'circle') {
      // swap karke call karo, fir normal reverse karo
      const result = collideCirclePolygon(bodyB, bodyA);
      if (result) {
        result.normal.x = -result.normal.x;
        result.normal.y = -result.normal.y;
      }
      return result;
    }
    // polygon vs polygon — SAT
    return satPolygonPolygon(bodyA, bodyB);
  }

  // ============================================================
  // IMPULSE-BASED COLLISION RESOLUTION
  // ============================================================

  function resolveCollision(bodyA, bodyB, collision) {
    const normal = collision.normal;

    for (const contact of collision.contacts) {
      // contact point se body centers tak ke vectors
      const r1x = contact.x - bodyA.x;
      const r1y = contact.y - bodyA.y;
      const r2x = contact.x - bodyB.x;
      const r2y = contact.y - bodyB.y;

      // contact point pe relative velocity nikaalo
      // v = linear_vel + angular_vel × r
      const v1x = bodyA.vx + (-bodyA.angularVel * r1y);
      const v1y = bodyA.vy + (bodyA.angularVel * r1x);
      const v2x = bodyB.vx + (-bodyB.angularVel * r2y);
      const v2y = bodyB.vy + (bodyB.angularVel * r2x);

      const relVx = v1x - v2x;
      const relVy = v1y - v2y;

      // relative velocity along collision normal
      const relVn = relVx * normal.x + relVy * normal.y;

      // agar bodies door ja rahe hain toh resolve mat karo
      if (relVn > 0) continue;

      // r cross n — angular contribution
      const r1CrossN = r1x * normal.y - r1y * normal.x;
      const r2CrossN = r2x * normal.y - r2y * normal.x;

      // effective mass inverse along normal
      const invMassSum = bodyA.invMass + bodyB.invMass +
        (r1CrossN * r1CrossN) * bodyA.invInertia +
        (r2CrossN * r2CrossN) * bodyB.invInertia;

      if (invMassSum < 0.0001) continue;

      // restitution — dono bodies ka minimum lo
      const e = Math.min(bodyA.restitution, bodyB.restitution);

      // normal impulse magnitude
      // contact count se divide karo taaki total impulse sahi rahe
      const j = -(1 + e) * relVn / (invMassSum * collision.contacts.length);

      // normal impulse apply karo
      const impulseX = j * normal.x;
      const impulseY = j * normal.y;

      bodyA.vx += impulseX * bodyA.invMass;
      bodyA.vy += impulseY * bodyA.invMass;
      bodyA.angularVel += (r1x * impulseY - r1y * impulseX) * bodyA.invInertia;

      bodyB.vx -= impulseX * bodyB.invMass;
      bodyB.vy -= impulseY * bodyB.invMass;
      bodyB.angularVel -= (r2x * impulseY - r2y * impulseX) * bodyB.invInertia;

      // --- Friction impulse ---
      // tangent direction nikaalo — normal ke perpendicular
      const tangentX = relVx - relVn * normal.x;
      const tangentY = relVy - relVn * normal.y;
      const tangentLen = Math.sqrt(tangentX * tangentX + tangentY * tangentY);

      if (tangentLen > 0.0001) {
        const tx = tangentX / tangentLen;
        const ty = tangentY / tangentLen;

        // tangent direction mein relative velocity
        const relVt = relVx * tx + relVy * ty;

        // r cross t — angular contribution for friction
        const r1CrossT = r1x * ty - r1y * tx;
        const r2CrossT = r2x * ty - r2y * tx;

        const invMassSumT = bodyA.invMass + bodyB.invMass +
          (r1CrossT * r1CrossT) * bodyA.invInertia +
          (r2CrossT * r2CrossT) * bodyB.invInertia;

        if (invMassSumT > 0.0001) {
          // friction impulse magnitude
          let jt = -relVt / (invMassSumT * collision.contacts.length);

          // Coulomb's law — friction impulse normal impulse se zyada nahi ho sakta
          const mu = Math.min(bodyA.friction, bodyB.friction);
          if (Math.abs(jt) > Math.abs(j) * mu) {
            jt = Math.sign(jt) * Math.abs(j) * mu;
          }

          // friction impulse apply karo
          const fImpulseX = jt * tx;
          const fImpulseY = jt * ty;

          bodyA.vx += fImpulseX * bodyA.invMass;
          bodyA.vy += fImpulseY * bodyA.invMass;
          bodyA.angularVel += (r1x * fImpulseY - r1y * fImpulseX) * bodyA.invInertia;

          bodyB.vx -= fImpulseX * bodyB.invMass;
          bodyB.vy -= fImpulseY * bodyB.invMass;
          bodyB.angularVel -= (r2x * fImpulseY - r2y * fImpulseX) * bodyB.invInertia;
        }
      }
    }

    // --- Positional correction — sinking prevent karo ---
    const correction = Math.max(collision.depth - SLOP, 0) * POSITIONAL_CORRECTION /
      (bodyA.invMass + bodyB.invMass + 0.0001);
    const corrX = correction * normal.x;
    const corrY = correction * normal.y;

    bodyA.x -= corrX * bodyA.invMass;
    bodyA.y -= corrY * bodyA.invMass;
    bodyB.x += corrX * bodyB.invMass;
    bodyB.y += corrY * bodyB.invMass;
  }

  // ============================================================
  // PHYSICS STEP — integration + collision detection + resolution
  // ============================================================

  function physicsStep(dt) {
    if (dt <= 0) return;

    // --- 1. Gravity apply karo ---
    for (const body of bodies) {
      if (body.isStatic) continue;
      // gravity sirf vertical mein — pixels/s^2 mein scale kiya hua
      body.vy += gravity * 100 * dt;
    }

    // --- 2. Velocity integrate karo — position update ---
    for (const body of bodies) {
      if (body.isStatic) continue;
      body.x += body.vx * dt;
      body.y += body.vy * dt;
      body.angle += body.angularVel * dt;
    }

    // --- 3. Collision detection aur resolution ---
    for (let i = 0; i < bodies.length; i++) {
      for (let j = i + 1; j < bodies.length; j++) {
        const a = bodies[i];
        const b = bodies[j];

        // dono static hain toh skip
        if (a.isStatic && b.isStatic) continue;

        // broad phase — AABB overlap check pehle (fast rejection)
        if (!aabbOverlap(a, b)) continue;

        // narrow phase — actual collision detect karo
        const collision = detectCollision(a, b);
        if (collision && collision.depth > 0) {
          resolveCollision(a, b, collision);
        }
      }
    }

    // --- 4. Boundary check — canvas se bahar ja rahe bodies hata do ---
    for (let i = bodies.length - 1; i >= 0; i--) {
      const b = bodies[i];
      if (b.isStatic) continue;
      // bahut neeche ya sides se bahar gaya toh hata do
      if (b.y > canvasH + 200 || b.x < -200 || b.x > canvasW + 200) {
        bodies.splice(i, 1);
      }
    }

    // --- 5. Angular velocity damping — thoda slow karo rotation ---
    for (const body of bodies) {
      if (body.isStatic) continue;
      body.angularVel *= 0.999;
      // bahut chhoti velocity zero kar do — resting bodies ke liye
      if (Math.abs(body.angularVel) < 0.001) body.angularVel = 0;
    }
  }

  // AABB (Axis-Aligned Bounding Box) overlap check — broad phase optimization
  function aabbOverlap(a, b) {
    const aabbA = getAABB(a);
    const aabbB = getAABB(b);
    return !(aabbA.maxX < aabbB.minX || aabbA.minX > aabbB.maxX ||
             aabbA.maxY < aabbB.minY || aabbA.minY > aabbB.maxY);
  }

  function getAABB(body) {
    if (body.type === 'circle') {
      return {
        minX: body.x - body.radius,
        maxX: body.x + body.radius,
        minY: body.y - body.radius,
        maxY: body.y + body.radius,
      };
    }
    const verts = getWorldVertices(body);
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const v of verts) {
      if (v.x < minX) minX = v.x;
      if (v.x > maxX) maxX = v.x;
      if (v.y < minY) minY = v.y;
      if (v.y > maxY) maxY = v.y;
    }
    return { minX, maxX, minY, maxY };
  }

  // ============================================================
  // PRESETS — Tower aur Avalanche
  // ============================================================

  function loadTowerPreset() {
    // purane dynamic bodies hata do
    bodies = bodies.filter(b => b.isStatic);

    // 6 layers ka tower — boxes stack karo
    const boxW = 40;
    const boxH = 25;
    const layers = 6;
    const startX = canvasW / 2;
    const floorTop = canvasH - FLOOR_HEIGHT;

    for (let row = 0; row < layers; row++) {
      const y = floorTop - boxH / 2 - row * (boxH + 1);
      // har layer mein thoda offset — realistic tower
      const offsetX = (Math.random() - 0.5) * 3;
      const body = createPolygonBody(startX + offsetX, y, boxVertices(boxW, boxH), 0, 0);
      body.restitution = restitution;
      body.friction = friction;
      bodies.push(body);
    }
  }

  function loadAvalanchePreset() {
    // purane dynamic bodies hata do
    bodies = bodies.filter(b => b.isStatic);

    // pyramid of mixed shapes — upar se spawn karo
    const floorTop = canvasH - FLOOR_HEIGHT;
    const baseCount = 5;
    const shapeTypes = ['box', 'triangle', 'pentagon', 'hexagon', 'circle'];

    for (let row = 0; row < baseCount; row++) {
      const shapesInRow = baseCount - row;
      const rowWidth = shapesInRow * 42;
      const startX = canvasW / 2 - rowWidth / 2 + 21;
      const y = floorTop - 20 - row * 35;

      for (let col = 0; col < shapesInRow; col++) {
        const x = startX + col * 42;
        const typeIdx = (row + col) % shapeTypes.length;
        const st = shapeTypes[typeIdx];
        let body;

        switch (st) {
          case 'box':
            body = createPolygonBody(x, y, boxVertices(32, 22), 0, (Math.random() - 0.5) * 0.1);
            break;
          case 'triangle':
            body = createPolygonBody(x, y, regularPolygonVertices(3, 18), 1, (Math.random() - 0.5) * 0.1);
            break;
          case 'pentagon':
            body = createPolygonBody(x, y, regularPolygonVertices(5, 16), 2, (Math.random() - 0.5) * 0.1);
            break;
          case 'hexagon':
            body = createPolygonBody(x, y, regularPolygonVertices(6, 16), 3, (Math.random() - 0.5) * 0.1);
            break;
          case 'circle':
            body = createCircleBody(x, y, 14, 4, 0);
            break;
          default:
            body = createPolygonBody(x, y, boxVertices(32, 22), 0, 0);
        }

        body.restitution = restitution;
        body.friction = friction;
        bodies.push(body);
      }
    }
  }

  // ============================================================
  // MOUSE / TOUCH EVENTS — spawn aur drag
  // ============================================================

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  canvas.addEventListener('mousedown', (e) => {
    const pos = getCanvasPos(e);
    isMouseDown = true;
    isDragging = false;
    mouseDownX = pos.x;
    mouseDownY = pos.y;
    mouseCurrentX = pos.x;
    mouseCurrentY = pos.y;
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isMouseDown) return;
    const pos = getCanvasPos(e);
    mouseCurrentX = pos.x;
    mouseCurrentY = pos.y;
    const dx = mouseCurrentX - mouseDownX;
    const dy = mouseCurrentY - mouseDownY;
    if (Math.sqrt(dx * dx + dy * dy) > 5) isDragging = true;
  });

  canvas.addEventListener('mouseup', () => {
    if (!isMouseDown) return;
    finishSpawn();
  });

  canvas.addEventListener('mouseleave', () => {
    if (!isMouseDown) return;
    finishSpawn();
  });

  // touch support
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    isMouseDown = true;
    isDragging = false;
    mouseDownX = pos.x;
    mouseDownY = pos.y;
    mouseCurrentX = pos.x;
    mouseCurrentY = pos.y;
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isMouseDown) return;
    const pos = getCanvasPos(e);
    mouseCurrentX = pos.x;
    mouseCurrentY = pos.y;
    const dx = mouseCurrentX - mouseDownX;
    const dy = mouseCurrentY - mouseDownY;
    if (Math.sqrt(dx * dx + dy * dy) > 5) isDragging = true;
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (!isMouseDown) return;
    finishSpawn();
  }, { passive: false });

  function finishSpawn() {
    isMouseDown = false;

    // floor pe click kiya toh ignore
    if (mouseDownY > canvasH - FLOOR_HEIGHT - 5) return;

    let vx = 0, vy = 0;
    if (isDragging) {
      // drag direction = velocity — throw ka feel
      vx = (mouseCurrentX - mouseDownX) * 3;
      vy = (mouseCurrentY - mouseDownY) * 3;
      // cap velocity
      const speed = Math.sqrt(vx * vx + vy * vy);
      if (speed > 600) {
        vx = (vx / speed) * 600;
        vy = (vy / speed) * 600;
      }
    }

    spawnShape(mouseDownX, mouseDownY, selectedShape, vx, vy);
    isDragging = false;
  }

  // ============================================================
  // DRAWING — shapes, floor, velocity vectors, spawn preview
  // ============================================================

  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // --- Floor draw karo — gradient strip with grid ---
    drawFloor();

    // --- Bodies draw karo ---
    for (const body of bodies) {
      if (body.isStatic) continue; // floor already drawn
      drawBody(body);
    }

    // --- Velocity vectors (toggleable) ---
    if (showVelocity) {
      for (const body of bodies) {
        if (body.isStatic) continue;
        drawVelocityVector(body);
      }
    }

    // --- Spawn preview — jab user drag kar raha hai ---
    if (isMouseDown) {
      drawSpawnPreview();
    }

    // --- Hint text jab koi body nahi hai ---
    if (bodies.filter(b => !b.isStatic).length === 0 && !isMouseDown) {
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('click to spawn \u2022 drag to throw', canvasW / 2, canvasH / 2 - 30);
    }

    updateStats();
  }

  function drawFloor() {
    const floorTop = canvasH - FLOOR_HEIGHT;

    // gradient fill — dark se thoda lighter
    const grad = ctx.createLinearGradient(0, floorTop, 0, canvasH);
    grad.addColorStop(0, 'rgba(' + ACCENT_RGB + ',0.12)');
    grad.addColorStop(1, 'rgba(' + ACCENT_RGB + ',0.04)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, floorTop, canvasW, FLOOR_HEIGHT);

    // top edge line — bright
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.3)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, floorTop);
    ctx.lineTo(canvasW, floorTop);
    ctx.stroke();

    // grid lines — vertical
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.06)';
    ctx.lineWidth = 1;
    const gridSpacing = 40;
    for (let x = gridSpacing; x < canvasW; x += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, floorTop);
      ctx.lineTo(x, canvasH);
      ctx.stroke();
    }

    // grid lines — horizontal
    for (let y = floorTop + gridSpacing / 2; y < canvasH; y += gridSpacing / 2) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasW, y);
      ctx.stroke();
    }
  }

  function drawBody(body) {
    const { r, g, b } = body.color;

    if (body.type === 'circle') {
      // --- Circle draw karo ---
      // semi-transparent fill
      ctx.beginPath();
      ctx.arc(body.x, body.y, body.radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.2)';
      ctx.fill();

      // outline
      ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.7)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // rotation indicator — center se edge tak line
      const cos = Math.cos(body.angle);
      const sin = Math.sin(body.angle);
      ctx.beginPath();
      ctx.moveTo(body.x, body.y);
      ctx.lineTo(body.x + cos * body.radius, body.y + sin * body.radius);
      ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.5)';
      ctx.lineWidth = 1;
      ctx.stroke();

    } else {
      // --- Polygon draw karo ---
      const verts = getWorldVertices(body);
      if (verts.length < 3) return;

      // semi-transparent fill
      ctx.beginPath();
      ctx.moveTo(verts[0].x, verts[0].y);
      for (let i = 1; i < verts.length; i++) {
        ctx.lineTo(verts[i].x, verts[i].y);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.2)';
      ctx.fill();

      // outline
      ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.7)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // rotation indicator — center se first vertex tak line
      ctx.beginPath();
      ctx.moveTo(body.x, body.y);
      ctx.lineTo(verts[0].x, verts[0].y);
      ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.4)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // center dot — subtle
    ctx.beginPath();
    ctx.arc(body.x, body.y, 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.6)';
    ctx.fill();
  }

  function drawVelocityVector(body) {
    const speed = Math.sqrt(body.vx * body.vx + body.vy * body.vy);
    if (speed < 5) return;

    const scale = 0.1;
    const arrowLen = Math.min(speed * scale, 60);
    const nx = body.vx / speed;
    const ny = body.vy / speed;

    const startX = body.x;
    const startY = body.y;
    const endX = startX + nx * arrowLen;
    const endY = startY + ny * arrowLen;

    // arrow line
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // arrowhead
    const headLen = Math.min(6, arrowLen * 0.3);
    const headAngle = Math.atan2(ny, nx);
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(endX - headLen * Math.cos(headAngle - 0.45), endY - headLen * Math.sin(headAngle - 0.45));
    ctx.moveTo(endX, endY);
    ctx.lineTo(endX - headLen * Math.cos(headAngle + 0.45), endY - headLen * Math.sin(headAngle + 0.45));
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawSpawnPreview() {
    // ghost shape dikhao spawn position pe
    ctx.save();

    ctx.beginPath();
    if (selectedShape === 'circle') {
      ctx.arc(mouseDownX, mouseDownY, 18, 0, Math.PI * 2);
    } else {
      let previewVerts;
      switch (selectedShape) {
        case 'box': previewVerts = boxVertices(35, 28); break;
        case 'triangle': previewVerts = regularPolygonVertices(3, 22); break;
        case 'pentagon': previewVerts = regularPolygonVertices(5, 20); break;
        case 'hexagon': previewVerts = regularPolygonVertices(6, 20); break;
        default: previewVerts = boxVertices(35, 28); break;
      }
      ctx.moveTo(mouseDownX + previewVerts[0].x, mouseDownY + previewVerts[0].y);
      for (let i = 1; i < previewVerts.length; i++) {
        ctx.lineTo(mouseDownX + previewVerts[i].x, mouseDownY + previewVerts[i].y);
      }
      ctx.closePath();
    }

    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.1)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // drag arrow — velocity preview
    if (isDragging) {
      const dx = mouseCurrentX - mouseDownX;
      const dy = mouseCurrentY - mouseDownY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist > 5) {
        // dashed line — spawn point se current position tak
        ctx.beginPath();
        ctx.moveTo(mouseDownX, mouseDownY);
        ctx.lineTo(mouseCurrentX, mouseCurrentY);
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // velocity arrow
        const nx = dx / dist;
        const ny = dy / dist;
        const arrowLen = Math.min(dist * 0.6, 60);

        ctx.beginPath();
        ctx.moveTo(mouseDownX, mouseDownY);
        ctx.lineTo(mouseDownX + nx * arrowLen, mouseDownY + ny * arrowLen);
        ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.7)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // arrowhead
        const headLen = 8;
        const headAngle = Math.atan2(ny, nx);
        ctx.beginPath();
        ctx.moveTo(mouseDownX + nx * arrowLen, mouseDownY + ny * arrowLen);
        ctx.lineTo(
          mouseDownX + nx * arrowLen - headLen * Math.cos(headAngle - 0.4),
          mouseDownY + ny * arrowLen - headLen * Math.sin(headAngle - 0.4)
        );
        ctx.moveTo(mouseDownX + nx * arrowLen, mouseDownY + ny * arrowLen);
        ctx.lineTo(
          mouseDownX + nx * arrowLen - headLen * Math.cos(headAngle + 0.4),
          mouseDownY + ny * arrowLen - headLen * Math.sin(headAngle + 0.4)
        );
        ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.7)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  // ============================================================
  // ANIMATION LOOP
  // ============================================================

  function loop(timestamp) {
    // lab pause check — sirf active sim animate ho
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    // delta time calculate karo
    if (lastTime === 0) lastTime = timestamp;
    let dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;

    // clamp — tab switch ke baad bahut bada dt aa sakta hai
    dt = Math.min(dt, 0.05);

    // substeps for stability — fast moving objects ke liye zaroori hai
    const subDt = dt / SUBSTEPS;
    for (let s = 0; s < SUBSTEPS; s++) {
      physicsStep(subDt);
    }

    draw();
    animationId = requestAnimationFrame(loop);
  }

  // --- IntersectionObserver — sirf jab screen pe dikhe tab animate karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    lastTime = 0;
    resizeCanvas();
    animationId = requestAnimationFrame(loop);
  }

  function stopAnimation() {
    isVisible = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          startAnimation();
        } else {
          stopAnimation();
        }
      });
    },
    { threshold: 0.1 }
  );

  observer.observe(container);

  // lab resume listener — jab pause hataye tab animation restart karo
  document.addEventListener('lab:resume', () => {
    if (isVisible && !animationId) loop(performance.now());
  });

  // tab visibility change — CPU bachao jab tab hidden ho
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // ============================================================
  // INITIAL STATE — floor + 3 default shapes
  // ============================================================

  function initialize() {
    bodies = [];
    bodyIdCounter = 0;

    // floor banao
    bodies.push(createFloor());

    // 3 default shapes — floor pe resting
    const floorTop = canvasH - FLOOR_HEIGHT;
    const spacing = canvasW / 4;

    // box — left side
    const box = createPolygonBody(spacing, floorTop - 22, boxVertices(40, 30), 0, 0.05);
    box.restitution = restitution;
    box.friction = friction;
    bodies.push(box);

    // pentagon — center mein
    const pent = createPolygonBody(spacing * 2, floorTop - 20, regularPolygonVertices(5, 20), 2, -0.1);
    pent.restitution = restitution;
    pent.friction = friction;
    bodies.push(pent);

    // circle — right side
    const circ = createCircleBody(spacing * 3, floorTop - 18, 18, 4, 0);
    circ.restitution = restitution;
    circ.friction = friction;
    bodies.push(circ);
  }

  initialize();

  // initial draw — taaki blank na dikhe
  resizeCanvas();
  draw();
}
