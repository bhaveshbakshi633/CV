// ============================================================
// Pressure-Based Soft Body Physics — deformable 2D bodies jo squish hoti hain
// Spring-particle ring model + internal pressure = jelly jaisi physics
// Click se spawn karo, drag karo, squish karo — mast physics sandbox
// ============================================================

// yahi function bahar export hoga — container dhundho, canvas banao, jelly banao
export function initSoftBody() {
  const container = document.getElementById('softBodyContainer');
  if (!container) {
    console.warn('softBodyContainer nahi mila bhai, soft body sim skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b';
  const ACCENT_RGB = '245,158,11';
  const N_PARTICLES = 20; // ek body mein kitne particles honge ring mein
  const MAX_BODIES = 8; // zyada bodies lag karegi CPU pe
  const FLOOR_HEIGHT = 6; // floor gradient ki height
  const SUB_STEPS = 8; // physics substeps — stability ke liye zaroori
  const GRAB_RADIUS = 30; // kitni door se particle grab ho sakta hai

  // --- Physics defaults (sliders se change honge) ---
  let stiffness = 200; // spring constant k
  let pressure = 1.5; // pressure multiplier
  let damping = 0.98; // velocity damping (1.0 = no damping)
  let gravityOn = true; // gravity toggle
  let showSprings = false; // spring lines dikhani hain ya nahi

  // --- Pretty colors palette — har body ko alag rang milega ---
  const BODY_COLORS = [
    { fill: 'rgba(245,158,11,0.18)', stroke: '#f59e0b', dot: '#fbbf24' },  // amber
    { fill: 'rgba(59,130,246,0.18)', stroke: '#3b82f6', dot: '#60a5fa' },  // blue
    { fill: 'rgba(168,85,247,0.18)', stroke: '#a855f7', dot: '#c084fc' },  // purple
    { fill: 'rgba(34,197,94,0.18)', stroke: '#22c55e', dot: '#4ade80' },   // green
    { fill: 'rgba(239,68,68,0.18)', stroke: '#ef4444', dot: '#f87171' },   // red
    { fill: 'rgba(236,72,153,0.18)', stroke: '#ec4899', dot: '#f472b6' },  // pink
    { fill: 'rgba(20,184,166,0.18)', stroke: '#14b8a6', dot: '#2dd4bf' },  // teal
    { fill: 'rgba(249,115,22,0.18)', stroke: '#f97316', dot: '#fb923c' },  // orange
  ];

  // --- State ---
  let canvasW = 0, canvasH = 0, dpr = 1;
  let bodies = []; // [{particles, springs, targetArea, colorIdx}]
  let animationId = null;
  let isVisible = false;
  let lastTime = 0;
  let colorCounter = 0; // next body ko kaunsa color milega

  // mouse/touch interaction state
  let mouseX = 0, mouseY = 0;
  let isMouseDown = false;
  let grabbedBodyIdx = -1; // konsa body pakda hua hai
  let grabbedParticleIdx = -1; // us body mein konsa particle
  let isDraggingBody = false; // existing body drag ho raha hai ya nahi

  // --- DOM structure banao — saaf karo pehle ---
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
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Info bar — body count ---
  const infoDiv = document.createElement('div');
  infoDiv.style.cssText = [
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
    'justify-content:space-between',
  ].join(';');
  container.appendChild(infoDiv);

  const bodyCountSpan = document.createElement('span');
  infoDiv.appendChild(bodyCountSpan);

  const hintSpan = document.createElement('span');
  hintSpan.style.cssText = 'color:#6b6b6b;font-size:11px;';
  hintSpan.textContent = 'click: spawn | drag: grab | dbl-click/right: delete';
  infoDiv.appendChild(hintSpan);

  function updateInfo() {
    bodyCountSpan.textContent = 'bodies: ' + bodies.length + '/' + MAX_BODIES;
    bodyCountSpan.style.color = bodies.length >= MAX_BODIES ? '#ef4444' : '#b0b0b0';
  }

  // --- Controls section ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:space-between',
  ].join(';');
  container.appendChild(controlsDiv);

  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:14px;flex:1;min-width:280px;';
  controlsDiv.appendChild(slidersDiv);

  const buttonsDiv = document.createElement('div');
  buttonsDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;';
  controlsDiv.appendChild(buttonsDiv);

  // --- Slider helper ---
  function createSlider(label, min, max, step, defaultVal, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = 'color:#b0b0b0;font-size:12px;font-weight:600;min-width:14px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText = 'width:85px;height:4px;accent-color:rgba(' + ACCENT_RGB + ',0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText = 'color:#b0b0b0;font-size:11px;min-width:32px;font-family:"JetBrains Mono",monospace;';
    valueEl.textContent = parseFloat(defaultVal).toFixed(step < 1 ? (step < 0.01 ? 3 : 1) : 0);
    wrapper.appendChild(valueEl);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = val.toFixed(step < 1 ? (step < 0.01 ? 3 : 1) : 0);
      onChange(val);
    });

    slidersDiv.appendChild(wrapper);
    return { slider, valueEl };
  }

  // --- Button helper ---
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:4px 10px',
      'font-size:11px',
      'border-radius:5px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.08)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.2)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.2)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.08)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    buttonsDiv.appendChild(btn);
    return btn;
  }

  // --- Toggle button helper — ON/OFF wale buttons ---
  function createToggle(textOn, textOff, defaultOn, onClick) {
    let isOn = defaultOn;
    const btn = createButton(isOn ? textOn : textOff, () => {
      isOn = !isOn;
      btn.textContent = isOn ? textOn : textOff;
      if (isOn) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
        btn.style.color = ACCENT;
        btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.5)';
      } else {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.08)';
        btn.style.color = '#b0b0b0';
        btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.2)';
      }
      onClick(isOn);
    });
    // initial state bhi set karo agar ON hai toh
    if (isOn) {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = ACCENT;
      btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.5)';
    }
    return btn;
  }

  // sliders banao
  createSlider('stiffness', 50, 500, 10, stiffness, (v) => { stiffness = v; });
  createSlider('pressure', 0.5, 3.0, 0.1, pressure, (v) => { pressure = v; });
  createSlider('damping', 0.9, 0.999, 0.001, damping, (v) => { damping = v; });

  // toggle buttons
  createToggle('Gravity: ON', 'Gravity: OFF', gravityOn, (v) => { gravityOn = v; });
  createToggle('Springs: ON', 'Springs: OFF', showSprings, (v) => { showSprings = v; });

  // clear all button
  createButton('Clear All', () => {
    bodies = [];
    colorCounter = 0;
    updateInfo();
  });

  // --- Canvas sizing — DPR aware crisp rendering ---
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
  // SOFT BODY CREATION — circle ring of particles + springs
  // ============================================================

  function createBody(cx, cy, radius) {
    // ring mein particles banao — circle ke perimeter pe equally spaced
    const particles = [];
    for (let i = 0; i < N_PARTICLES; i++) {
      const angle = (i / N_PARTICLES) * Math.PI * 2;
      const px = cx + Math.cos(angle) * radius;
      const py = cy + Math.sin(angle) * radius;
      particles.push({
        x: px,
        y: py,
        prevX: px, // verlet integration ke liye previous position
        prevY: py,
      });
    }

    // springs banao — consecutive particles ke beech circumference springs
    const springs = [];
    for (let i = 0; i < N_PARTICLES; i++) {
      const j = (i + 1) % N_PARTICLES;
      const dx = particles[j].x - particles[i].x;
      const dy = particles[j].y - particles[i].y;
      const restLen = Math.sqrt(dx * dx + dy * dy);
      springs.push({ i: i, j: j, restLength: restLen });
    }

    // cross-springs bhi daalo stability ke liye — opposite particles connect karo
    // har doosre particle ko uske diametrically opposite wale se jodo
    for (let i = 0; i < Math.floor(N_PARTICLES / 2); i++) {
      const j = (i + Math.floor(N_PARTICLES / 2)) % N_PARTICLES;
      const dx = particles[j].x - particles[i].x;
      const dy = particles[j].y - particles[i].y;
      const restLen = Math.sqrt(dx * dx + dy * dy);
      springs.push({ i: i, j: j, restLength: restLen });
    }

    // target area calculate karo — shoelace formula se initial area nikaalo
    const targetArea = computeArea(particles) * pressure;

    const colorIdx = colorCounter % BODY_COLORS.length;
    colorCounter++;

    return {
      particles: particles,
      springs: springs,
      targetArea: targetArea,
      colorIdx: colorIdx,
      radius: radius, // reference ke liye rakh lo
    };
  }

  // --- Shoelace formula — polygon ka area nikaalta hai ---
  // ye signed area deta hai, abs le lena
  function computeArea(particles) {
    let area = 0;
    const n = particles.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += particles[i].x * particles[j].y;
      area -= particles[j].x * particles[i].y;
    }
    return Math.abs(area) * 0.5;
  }

  // --- Edge ka outward normal nikaalo ---
  // consecutive particles ke beech ka edge, normal bahar ki taraf point karega
  function edgeNormal(p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.0001) return { nx: 0, ny: 0, len: 0 };
    // perpendicular normal — right-hand rule se bahar ki taraf
    return { nx: -dy / len, ny: dx / len, len: len };
  }

  // ============================================================
  // PHYSICS — Verlet integration + spring forces + pressure
  // ============================================================

  function physicsStep(dt) {
    const gravity = gravityOn ? 500 : 0; // pixels/s^2

    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b];
      const pts = body.particles;

      // --- 1. Verlet integration — position update with gravity ---
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];

        // current velocity nikaal lo from positions
        let vx = (p.x - p.prevX) * damping;
        let vy = (p.y - p.prevY) * damping;

        // previous position save karo
        p.prevX = p.x;
        p.prevY = p.y;

        // naya position = purana + velocity + gravity
        p.x += vx;
        p.y += vy + gravity * dt * dt;
      }

      // --- 2. Spring constraints solve karo ---
      // spring force: distance ko rest length ke paas laao
      for (let s = 0; s < body.springs.length; s++) {
        const spring = body.springs[s];
        const p1 = pts[spring.i];
        const p2 = pts[spring.j];

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 0.0001) continue;

        // spring force — Hooke's law ka position-based version
        // stiffness ko dt ke hisaab se scale karo
        const forceMag = stiffness * (dist - spring.restLength) * dt * dt;

        // direction normalize kar
        const nx = dx / dist;
        const ny = dy / dist;

        // dono particles ko symmetrically move karo
        const fx = nx * forceMag * 0.5;
        const fy = ny * forceMag * 0.5;

        p1.x += fx;
        p1.y += fy;
        p2.x -= fx;
        p2.y -= fy;
      }

      // --- 3. Pressure force — enclosed area vs target area ---
      const currentArea = computeArea(pts);
      if (currentArea < 0.1) continue; // degenerate body skip kar

      // pressure ratio — jab area chhota ho toh force bahar push kare
      const pressureForce = body.targetArea / currentArea;

      // har edge pe outward normal force lagao, proportional to pressure aur edge length
      for (let i = 0; i < pts.length; i++) {
        const j = (i + 1) % pts.length;
        const p1 = pts[i];
        const p2 = pts[j];

        const norm = edgeNormal(p1, p2);
        if (norm.len < 0.0001) continue;

        // force proportional to edge length * pressure difference
        const f = (pressureForce - 1.0) * norm.len * dt * dt * 50;

        // dono endpoints pe force equally distribute karo
        p1.x += norm.nx * f * 0.5;
        p1.y += norm.ny * f * 0.5;
        p2.x += norm.nx * f * 0.5;
        p2.y += norm.ny * f * 0.5;
      }

      // --- 4. Boundary collision — canvas ke andar rakho ---
      const floorY = canvasH - FLOOR_HEIGHT;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];

        // floor collision — bounce ke saath
        if (p.y > floorY) {
          p.y = floorY;
          // velocity reflect karo thoda damped
          const vy = p.y - p.prevY;
          p.prevY = p.y + vy * 0.3;
          // friction bhi lagao floor pe
          const vx = p.x - p.prevX;
          p.prevX = p.x - vx * 0.95;
        }

        // ceiling
        if (p.y < 2) {
          p.y = 2;
          p.prevY = p.y;
        }

        // left wall
        if (p.x < 2) {
          p.x = 2;
          const vx = p.x - p.prevX;
          p.prevX = p.x + vx * 0.3;
        }

        // right wall
        if (p.x > canvasW - 2) {
          p.x = canvasW - 2;
          const vx = p.x - p.prevX;
          p.prevX = p.x + vx * 0.3;
        }
      }
    }

    // --- 5. Body-body collision — different bodies ke particles ke beech repulsion ---
    for (let a = 0; a < bodies.length; a++) {
      for (let b = a + 1; b < bodies.length; b++) {
        resolveBodyCollision(bodies[a], bodies[b], dt);
      }
    }

    // --- 6. Mouse grab force --- grabbed particle ko mouse ke paas kheencho
    if (isDraggingBody && grabbedBodyIdx >= 0 && grabbedBodyIdx < bodies.length) {
      const body = bodies[grabbedBodyIdx];
      const p = body.particles[grabbedParticleIdx];
      // strong spring force towards mouse — satisfying drag feel
      const dx = mouseX - p.x;
      const dy = mouseY - p.y;
      p.x += dx * 0.3;
      p.y += dy * 0.3;
      // velocity bhi mouse direction mein set karo
      p.prevX = p.x - dx * 0.1;
      p.prevY = p.y - dy * 0.1;
    }
  }

  // --- Body-body collision — particle level repulsion ---
  // har particle doosre body ke particles ke paas aaye toh repel karo
  function resolveBodyCollision(bodyA, bodyB, dt) {
    const minDist = 8; // kitni distance pe repulsion shuru ho
    const minDistSq = minDist * minDist;
    const repulsionStrength = 0.4;

    for (let i = 0; i < bodyA.particles.length; i++) {
      const pa = bodyA.particles[i];
      for (let j = 0; j < bodyB.particles.length; j++) {
        const pb = bodyB.particles[j];

        const dx = pb.x - pa.x;
        const dy = pb.y - pa.y;
        const distSq = dx * dx + dy * dy;

        if (distSq < minDistSq && distSq > 0.0001) {
          const dist = Math.sqrt(distSq);
          const overlap = minDist - dist;

          // normalize direction
          const nx = dx / dist;
          const ny = dy / dist;

          // dono particles ko bahar dhakelo
          const push = overlap * repulsionStrength;
          pa.x -= nx * push;
          pa.y -= ny * push;
          pb.x += nx * push;
          pb.y += ny * push;
        }
      }
    }
  }

  // ============================================================
  // MOUSE / TOUCH INTERACTION
  // ============================================================

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  // sabse nazdeek body aur particle dhundho mouse position se
  function findNearestParticle(mx, my) {
    let bestDist = Infinity;
    let bestBody = -1;
    let bestParticle = -1;

    for (let b = 0; b < bodies.length; b++) {
      const pts = bodies[b].particles;
      for (let i = 0; i < pts.length; i++) {
        const dx = pts[i].x - mx;
        const dy = pts[i].y - my;
        const dist = dx * dx + dy * dy;
        if (dist < bestDist) {
          bestDist = dist;
          bestBody = b;
          bestParticle = i;
        }
      }
    }

    // GRAB_RADIUS ke andar hona chahiye
    if (bestDist < GRAB_RADIUS * GRAB_RADIUS) {
      return { bodyIdx: bestBody, particleIdx: bestParticle };
    }
    return null;
  }

  // check karo mouse kisi body ke andar hai ya nahi (point-in-polygon)
  function findBodyAtPoint(mx, my) {
    for (let b = 0; b < bodies.length; b++) {
      if (isPointInsideBody(bodies[b], mx, my)) return b;
    }
    return -1;
  }

  // ray casting point-in-polygon test
  function isPointInsideBody(body, px, py) {
    const pts = body.particles;
    let inside = false;
    const n = pts.length;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const xi = pts[i].x, yi = pts[i].y;
      const xj = pts[j].x, yj = pts[j].y;
      if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  }

  // body delete karo index se
  function deleteBody(idx) {
    if (idx >= 0 && idx < bodies.length) {
      bodies.splice(idx, 1);
      // grabbed state reset karo agar delete hua body grabbed tha
      if (grabbedBodyIdx === idx) {
        grabbedBodyIdx = -1;
        grabbedParticleIdx = -1;
        isDraggingBody = false;
      } else if (grabbedBodyIdx > idx) {
        grabbedBodyIdx--;
      }
      updateInfo();
    }
  }

  function onPointerDown(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    mouseX = pos.x;
    mouseY = pos.y;
    isMouseDown = true;

    // pehle check karo koi existing body ka particle grab ho raha hai ya nahi
    const nearest = findNearestParticle(mouseX, mouseY);
    if (nearest) {
      grabbedBodyIdx = nearest.bodyIdx;
      grabbedParticleIdx = nearest.particleIdx;
      isDraggingBody = true;
      canvas.style.cursor = 'grabbing';
      return;
    }

    // kisi body ke andar click hua toh bhi grab karo (centroid ke paas se)
    const bodyIdx = findBodyAtPoint(mouseX, mouseY);
    if (bodyIdx >= 0) {
      // sabse nazdeek wala particle pakad lo us body ka
      let bestDist = Infinity;
      let bestP = 0;
      const pts = bodies[bodyIdx].particles;
      for (let i = 0; i < pts.length; i++) {
        const dx = pts[i].x - mouseX;
        const dy = pts[i].y - mouseY;
        const d = dx * dx + dy * dy;
        if (d < bestDist) { bestDist = d; bestP = i; }
      }
      grabbedBodyIdx = bodyIdx;
      grabbedParticleIdx = bestP;
      isDraggingBody = true;
      canvas.style.cursor = 'grabbing';
      return;
    }

    // khaali jagah pe click — naya body spawn karo
    if (bodies.length < MAX_BODIES) {
      // random radius thoda vary karo — 30-50 px
      const radius = 30 + Math.random() * 20;
      const newBody = createBody(mouseX, mouseY, radius);
      bodies.push(newBody);
      updateInfo();
    }
  }

  function onPointerMove(e) {
    const pos = getCanvasPos(e);
    mouseX = pos.x;
    mouseY = pos.y;

    // agar drag nahi ho raha toh cursor update karo
    if (!isMouseDown) {
      const nearest = findNearestParticle(mouseX, mouseY);
      const bodyAtPoint = findBodyAtPoint(mouseX, mouseY);
      if (nearest || bodyAtPoint >= 0) {
        canvas.style.cursor = 'grab';
      } else {
        canvas.style.cursor = 'crosshair';
      }
    }
  }

  function onPointerUp() {
    isMouseDown = false;
    isDraggingBody = false;
    grabbedBodyIdx = -1;
    grabbedParticleIdx = -1;
    canvas.style.cursor = 'crosshair';
  }

  // right-click se delete
  function onContextMenu(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const bodyIdx = findBodyAtPoint(pos.x, pos.y);
    if (bodyIdx >= 0) {
      deleteBody(bodyIdx);
    } else {
      // nearest particle check karo — shayad body ke edge pe click hua
      const nearest = findNearestParticle(pos.x, pos.y);
      if (nearest) deleteBody(nearest.bodyIdx);
    }
  }

  // double-click se delete
  function onDblClick(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const bodyIdx = findBodyAtPoint(pos.x, pos.y);
    if (bodyIdx >= 0) {
      deleteBody(bodyIdx);
    } else {
      const nearest = findNearestParticle(pos.x, pos.y);
      if (nearest) deleteBody(nearest.bodyIdx);
    }
  }

  // mouse events
  canvas.addEventListener('mousedown', onPointerDown);
  canvas.addEventListener('mousemove', onPointerMove);
  canvas.addEventListener('mouseup', onPointerUp);
  canvas.addEventListener('mouseleave', onPointerUp);
  canvas.addEventListener('contextmenu', onContextMenu);
  canvas.addEventListener('dblclick', onDblClick);

  // touch events
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const fakeEvent = {
      clientX: e.touches[0].clientX,
      clientY: e.touches[0].clientY,
      touches: e.touches,
      button: 0,
      preventDefault: () => {},
    };
    onPointerDown(fakeEvent);
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const fakeEvent = {
      clientX: e.touches[0].clientX,
      clientY: e.touches[0].clientY,
      touches: e.touches,
      preventDefault: () => {},
    };
    onPointerMove(fakeEvent);
  }, { passive: false });

  canvas.addEventListener('touchend', onPointerUp);
  canvas.addEventListener('touchcancel', onPointerUp);

  // ============================================================
  // RENDERING — filled polygon bodies, spring lines, floor
  // ============================================================

  function drawFloor() {
    // gradient floor strip — canvas ke bottom pe
    const floorY = canvasH - FLOOR_HEIGHT;
    const grad = ctx.createLinearGradient(0, floorY, 0, canvasH);
    grad.addColorStop(0, 'rgba(' + ACCENT_RGB + ',0.15)');
    grad.addColorStop(1, 'rgba(' + ACCENT_RGB + ',0.03)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, floorY, canvasW, FLOOR_HEIGHT);

    // floor top line — subtle border
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, floorY);
    ctx.lineTo(canvasW, floorY);
    ctx.stroke();
  }

  function drawBody(body) {
    const pts = body.particles;
    const color = BODY_COLORS[body.colorIdx];

    // --- Spring lines (agar toggle ON hai) ---
    if (showSprings) {
      ctx.strokeStyle = color.stroke + '33'; // bahut halka
      ctx.lineWidth = 0.5;
      for (let s = 0; s < body.springs.length; s++) {
        const spring = body.springs[s];
        const p1 = pts[spring.i];
        const p2 = pts[spring.j];
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }

    // --- Filled polygon — body ka shape ---
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i].x, pts[i].y);
    }
    ctx.closePath();
    ctx.fillStyle = color.fill;
    ctx.fill();

    // --- Stroke outline ---
    ctx.strokeStyle = color.stroke + '88'; // semi-transparent
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // --- Particle dots — chhote gol binde ---
    for (let i = 0; i < pts.length; i++) {
      ctx.beginPath();
      ctx.arc(pts[i].x, pts[i].y, 2, 0, Math.PI * 2);
      ctx.fillStyle = color.dot;
      ctx.fill();
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // floor pehle draw karo — bodies uske upar aayengi
    drawFloor();

    // saari bodies draw karo
    for (let b = 0; b < bodies.length; b++) {
      drawBody(bodies[b]);
    }

    // grabbed particle highlight karo
    if (isDraggingBody && grabbedBodyIdx >= 0 && grabbedBodyIdx < bodies.length) {
      const p = bodies[grabbedBodyIdx].particles[grabbedParticleIdx];
      ctx.beginPath();
      ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // hint text — jab koi body nahi hai
    if (bodies.length === 0) {
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('click anywhere to spawn a soft body', canvasW / 2, canvasH / 2);
    }
  }

  // ============================================================
  // ANIMATION LOOP
  // ============================================================

  function loop(timestamp) {
    // lab pause check — sirf active sim animate ho
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) { animationId = null; return; }

    if (lastTime === 0) lastTime = timestamp;
    let dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;

    // dt clamp — tab switch ke baad bahut bada dt aa sakta hai, instability hogi
    dt = Math.min(dt, 0.033);

    // substeps — Verlet mein chhote steps zyada stable hote hain
    const subDt = dt / SUB_STEPS;
    for (let s = 0; s < SUB_STEPS; s++) {
      physicsStep(subDt);
    }

    draw();

    animationId = requestAnimationFrame(loop);
  }

  // --- Visibility management — IntersectionObserver ---
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

  // lab:resume event — jab koi doosri sim pause thi aur ye resume hui
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });

  // tab visibility — CPU/battery bachao
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
  // INITIAL STATE — 2 default bodies spawn karo alag heights pe
  // ============================================================

  function spawnDefaultBodies() {
    // pehli body — thoda left side, upar se girega
    const body1 = createBody(canvasW * 0.35, 80, 40);
    bodies.push(body1);

    // doosri body — right side, thodi neeche se shuru
    const body2 = createBody(canvasW * 0.65, 150, 35);
    bodies.push(body2);
  }

  // initial setup
  spawnDefaultBodies();
  updateInfo();
  resizeCanvas();
  draw();
}
