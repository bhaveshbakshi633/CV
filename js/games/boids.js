// ============================================================
// Boids Flocking Simulation — Craig Reynolds (1987) ka classic algorithm
// Separation, Alignment, Cohesion — teeno rules se emergent flocking behavior
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, boids udaao
export function initBoids() {
  const container = document.getElementById('boidsContainer');
  if (!container) {
    console.warn('boidsContainer nahi mila bhai, boids demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 380;
  const ACCENT = '#4a9eff';
  const ACCENT_RGB = '74,158,255';
  const TRAIL_LENGTH = 8; // frames ki trail — subtle effect
  const BOID_SIZE = 6; // triangle ka size
  const WALL_MARGIN = 50; // wall avoidance zone — pixels mein
  const WALL_TURN_FACTOR = 0.5; // kitni tezi se wall se mude
  const SCARE_DURATION = 2000; // ms — predator scare point kitni der active rahe
  const SCARE_RADIUS = 100; // predator scare ka radius
  const SCARE_STRENGTH = 3.0; // predator se kitna darr lage

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let boids = []; // [{x, y, vx, vy, ax, ay, trail: [{x,y}]}]
  let flockSize = 150;
  let visualRange = 75;
  let maxSpeed = 4;
  let maxForce = 0.15;
  let separationWeight = 1.5;
  let alignmentWeight = 1.0;
  let cohesionWeight = 1.0;
  let predatorMode = false;
  let scarePoints = []; // [{x, y, birth}] — temporary scare locations
  let mouseX = -1, mouseY = -1; // hover position
  let hoveredBoid = null; // closest boid to mouse — visual range dikhane ke liye

  // animation state
  let animationId = null;
  let isVisible = false;
  let lastTime = 0;

  // --- DOM structure banao ---
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

  // --- Stats bar ---
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'margin-top:8px',
    'padding:8px 12px',
    'background:rgba(' + ACCENT_RGB + ',0.05)',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.12)',
    'border-radius:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:#b0b0b0',
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'align-items:center',
  ].join(';');
  container.appendChild(statsDiv);

  const statCount = document.createElement('span');
  const statAvgSpeed = document.createElement('span');
  const statAvgHeading = document.createElement('span');
  const statScares = document.createElement('span');
  [statCount, statAvgSpeed, statAvgHeading, statScares].forEach(el => statsDiv.appendChild(el));

  function updateStats() {
    if (boids.length === 0) {
      statCount.textContent = 'Boids: 0';
      statAvgSpeed.textContent = 'Avg Speed: 0.0';
      statAvgHeading.textContent = 'Avg Heading: 0°';
      statScares.textContent = '';
      return;
    }
    let totalSpeed = 0;
    let sumCos = 0, sumSin = 0;
    for (const b of boids) {
      const spd = Math.sqrt(b.vx * b.vx + b.vy * b.vy);
      totalSpeed += spd;
      const angle = Math.atan2(b.vy, b.vx);
      sumCos += Math.cos(angle);
      sumSin += Math.sin(angle);
    }
    const avgSpeed = totalSpeed / boids.length;
    const avgHeading = Math.atan2(sumSin / boids.length, sumCos / boids.length);
    const avgHeadingDeg = ((avgHeading * 180 / Math.PI) + 360) % 360;

    statCount.textContent = 'Boids: ' + boids.length;
    statAvgSpeed.textContent = 'Avg Speed: ' + avgSpeed.toFixed(1);
    statAvgHeading.textContent = 'Avg Heading: ' + avgHeadingDeg.toFixed(0) + '\u00B0';
    if (predatorMode) {
      statScares.textContent = 'Scares: ' + scarePoints.length;
    } else {
      statScares.textContent = '';
    }
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

  // button factory — consistent styling
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

  // slider factory — label + range + value display
  function createSlider(label, min, max, step, value, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.style.cssText = 'width:65px;height:4px;accent-color:rgba(' + ACCENT_RGB + ',0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const val = document.createElement('span');
    val.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:"JetBrains Mono",monospace;min-width:28px;';
    val.textContent = Number(value).toFixed(step < 1 ? 1 : 0);
    wrapper.appendChild(val);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      val.textContent = v.toFixed(step < 1 ? 1 : 0);
      onChange(v);
    });

    controlsDiv.appendChild(wrapper);
    return { wrapper, slider, val };
  }

  // --- Slider Controls ---

  // separation weight
  const sepSlider = createSlider('Sep:', 0, 5, 0.1, separationWeight, v => {
    separationWeight = v;
  });

  // alignment weight
  const alignSlider = createSlider('Align:', 0, 5, 0.1, alignmentWeight, v => {
    alignmentWeight = v;
  });

  // cohesion weight
  const cohSlider = createSlider('Coh:', 0, 5, 0.1, cohesionWeight, v => {
    cohesionWeight = v;
  });

  // flock size
  const sizeSlider = createSlider('Flock:', 50, 300, 1, flockSize, v => {
    flockSize = Math.round(v);
    adjustFlockSize();
  });

  // visual range
  const rangeSlider = createSlider('Range:', 30, 150, 1, visualRange, v => {
    visualRange = Math.round(v);
  });

  // --- Buttons ---

  // reset button
  createButton('Reset', () => {
    initFlock();
  });

  // predator mode toggle
  const predatorBtn = createButton('Predator: OFF', () => {
    predatorMode = !predatorMode;
    predatorBtn.textContent = predatorMode ? 'Predator: ON' : 'Predator: OFF';
    if (predatorMode) {
      predatorBtn.style.background = 'rgba(255,80,80,0.25)';
      predatorBtn.style.borderColor = 'rgba(255,80,80,0.5)';
      predatorBtn.style.color = '#ff6060';
      canvas.style.cursor = 'crosshair';
    } else {
      predatorBtn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      predatorBtn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
      predatorBtn.style.color = '#b0b0b0';
      canvas.style.cursor = 'crosshair';
      scarePoints = [];
    }
  });

  // --- Presets dropdown ---
  const presetSelect = document.createElement('select');
  presetSelect.style.cssText = [
    'padding:6px 10px',
    'font-size:12px',
    'border-radius:6px',
    'cursor:pointer',
    'background:rgba(' + ACCENT_RGB + ',0.1)',
    'color:#b0b0b0',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
    'font-family:"JetBrains Mono",monospace',
  ].join(';');

  const presets = [
    { name: 'Presets...', sep: 1.5, align: 1.0, coh: 1.0, range: 75, size: 150 },
    { name: 'Default', sep: 1.5, align: 1.0, coh: 1.0, range: 75, size: 150 },
    { name: 'Tight Flock', sep: 0.8, align: 2.5, coh: 2.5, range: 60, size: 200 },
    { name: 'Loose Swarm', sep: 3.0, align: 0.5, coh: 0.3, range: 100, size: 120 },
    { name: 'Fast & Chaotic', sep: 4.0, align: 0.2, coh: 0.2, range: 40, size: 250 },
  ];

  presets.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.name;
    opt.textContent = p.name;
    opt.style.background = '#1a1a1a';
    opt.style.color = '#b0b0b0';
    presetSelect.appendChild(opt);
  });

  presetSelect.addEventListener('change', () => {
    const preset = presets.find(p => p.name === presetSelect.value);
    if (preset && preset.name !== 'Presets...') {
      applyPreset(preset);
    }
    presetSelect.value = 'Presets...';
  });
  controlsDiv.appendChild(presetSelect);

  // preset apply karo — sliders update + state update
  function applyPreset(p) {
    separationWeight = p.sep;
    alignmentWeight = p.align;
    cohesionWeight = p.coh;
    visualRange = p.range;
    flockSize = p.size;

    // sliders ko sync karo
    sepSlider.slider.value = String(p.sep);
    sepSlider.val.textContent = p.sep.toFixed(1);
    alignSlider.slider.value = String(p.align);
    alignSlider.val.textContent = p.align.toFixed(1);
    cohSlider.slider.value = String(p.coh);
    cohSlider.val.textContent = p.coh.toFixed(1);
    rangeSlider.slider.value = String(p.range);
    rangeSlider.val.textContent = String(p.range);
    sizeSlider.slider.value = String(p.size);
    sizeSlider.val.textContent = String(p.size);

    adjustFlockSize();
  }

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

    const c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Boid creation ---
  function createBoid(x, y) {
    // random direction se shuru — angle random, speed max ka 60-100%
    const angle = Math.random() * Math.PI * 2;
    const speed = maxSpeed * (0.6 + Math.random() * 0.4);
    return {
      x: x,
      y: y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      ax: 0,
      ay: 0,
      trail: [], // [{x, y}] — position history for trail rendering
    };
  }

  // flock initialize karo — random positions pe boids spawn karo
  function initFlock() {
    boids = [];
    scarePoints = [];
    for (let i = 0; i < flockSize; i++) {
      const x = Math.random() * (canvasW || 600);
      const y = Math.random() * (canvasH || CANVAS_HEIGHT);
      boids.push(createBoid(x, y));
    }
  }

  // flock size adjust — add ya remove karo boids
  function adjustFlockSize() {
    while (boids.length < flockSize) {
      const x = Math.random() * (canvasW || 600);
      const y = Math.random() * (canvasH || CANVAS_HEIGHT);
      boids.push(createBoid(x, y));
    }
    while (boids.length > flockSize) {
      boids.pop();
    }
  }

  // --- Boids Algorithm — Reynolds ka core logic ---

  // separation — paas ke boids se door bhaag
  function separation(boid) {
    let steerX = 0, steerY = 0;
    let count = 0;
    const minDist = visualRange * 0.4; // separation distance — visual range ka 40%

    for (const other of boids) {
      if (other === boid) continue;
      const dx = boid.x - other.x;
      const dy = boid.y - other.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < minDist && dist > 0) {
        // inversely proportional to distance — jitna paas utna zyada repel
        steerX += dx / dist / dist;
        steerY += dy / dist / dist;
        count++;
      }
    }

    if (count > 0) {
      steerX /= count;
      steerY /= count;
      // normalize and scale to max force
      const mag = Math.sqrt(steerX * steerX + steerY * steerY);
      if (mag > 0) {
        steerX = (steerX / mag) * maxSpeed - boid.vx;
        steerY = (steerY / mag) * maxSpeed - boid.vy;
        const steerMag = Math.sqrt(steerX * steerX + steerY * steerY);
        if (steerMag > maxForce) {
          steerX = (steerX / steerMag) * maxForce;
          steerY = (steerY / steerMag) * maxForce;
        }
      }
    }

    return { x: steerX, y: steerY };
  }

  // alignment — nearby boids ki average heading follow kar
  function alignment(boid) {
    let avgVx = 0, avgVy = 0;
    let count = 0;

    for (const other of boids) {
      if (other === boid) continue;
      const dx = other.x - boid.x;
      const dy = other.y - boid.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < visualRange) {
        avgVx += other.vx;
        avgVy += other.vy;
        count++;
      }
    }

    if (count > 0) {
      avgVx /= count;
      avgVy /= count;
      // steer towards average velocity
      const mag = Math.sqrt(avgVx * avgVx + avgVy * avgVy);
      if (mag > 0) {
        avgVx = (avgVx / mag) * maxSpeed;
        avgVy = (avgVy / mag) * maxSpeed;
      }
      let steerX = avgVx - boid.vx;
      let steerY = avgVy - boid.vy;
      const steerMag = Math.sqrt(steerX * steerX + steerY * steerY);
      if (steerMag > maxForce) {
        steerX = (steerX / steerMag) * maxForce;
        steerY = (steerY / steerMag) * maxForce;
      }
      return { x: steerX, y: steerY };
    }

    return { x: 0, y: 0 };
  }

  // cohesion — nearby boids ke center of mass ki taraf jao
  function cohesion(boid) {
    let avgX = 0, avgY = 0;
    let count = 0;

    for (const other of boids) {
      if (other === boid) continue;
      const dx = other.x - boid.x;
      const dy = other.y - boid.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < visualRange) {
        avgX += other.x;
        avgY += other.y;
        count++;
      }
    }

    if (count > 0) {
      avgX /= count;
      avgY /= count;
      // steer towards center of mass
      let desiredX = avgX - boid.x;
      let desiredY = avgY - boid.y;
      const mag = Math.sqrt(desiredX * desiredX + desiredY * desiredY);
      if (mag > 0) {
        desiredX = (desiredX / mag) * maxSpeed;
        desiredY = (desiredY / mag) * maxSpeed;
      }
      let steerX = desiredX - boid.vx;
      let steerY = desiredY - boid.vy;
      const steerMag = Math.sqrt(steerX * steerX + steerY * steerY);
      if (steerMag > maxForce) {
        steerX = (steerX / steerMag) * maxForce;
        steerY = (steerY / steerMag) * maxForce;
      }
      return { x: steerX, y: steerY };
    }

    return { x: 0, y: 0 };
  }

  // wall avoidance — canvas ke edges se door raho
  function wallAvoidance(boid) {
    let steerX = 0, steerY = 0;

    if (boid.x < WALL_MARGIN) {
      steerX += WALL_TURN_FACTOR * (1 - boid.x / WALL_MARGIN);
    }
    if (boid.x > canvasW - WALL_MARGIN) {
      steerX -= WALL_TURN_FACTOR * (1 - (canvasW - boid.x) / WALL_MARGIN);
    }
    if (boid.y < WALL_MARGIN) {
      steerY += WALL_TURN_FACTOR * (1 - boid.y / WALL_MARGIN);
    }
    if (boid.y > canvasH - WALL_MARGIN) {
      steerY -= WALL_TURN_FACTOR * (1 - (canvasH - boid.y) / WALL_MARGIN);
    }

    return { x: steerX, y: steerY };
  }

  // predator avoidance — scare points se bhaag
  function predatorAvoidance(boid) {
    let steerX = 0, steerY = 0;

    for (const scare of scarePoints) {
      const dx = boid.x - scare.x;
      const dy = boid.y - scare.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < SCARE_RADIUS && dist > 0) {
        // jitna paas utna zyada darr — inversely proportional
        const strength = SCARE_STRENGTH * (1 - dist / SCARE_RADIUS);
        steerX += (dx / dist) * strength;
        steerY += (dy / dist) * strength;
      }
    }

    return { x: steerX, y: steerY };
  }

  // --- Physics Update ---
  function updateBoids() {
    const now = performance.now();

    // expire old scare points
    scarePoints = scarePoints.filter(s => now - s.birth < SCARE_DURATION);

    for (const boid of boids) {
      // trail update — current position store karo
      boid.trail.push({ x: boid.x, y: boid.y });
      if (boid.trail.length > TRAIL_LENGTH) {
        boid.trail.shift();
      }

      // calculate forces — Reynolds ke teen rules
      const sep = separation(boid);
      const ali = alignment(boid);
      const coh = cohesion(boid);
      const wall = wallAvoidance(boid);

      // acceleration reset aur forces apply
      boid.ax = sep.x * separationWeight + ali.x * alignmentWeight + coh.x * cohesionWeight + wall.x;
      boid.ay = sep.y * separationWeight + ali.y * alignmentWeight + coh.y * cohesionWeight + wall.y;

      // predator avoidance agar active hai
      if (predatorMode && scarePoints.length > 0) {
        const pred = predatorAvoidance(boid);
        boid.ax += pred.x;
        boid.ay += pred.y;
      }

      // velocity update — acceleration add karo
      boid.vx += boid.ax;
      boid.vy += boid.ay;

      // speed limit lagao — maxSpeed se zyada nahi
      const speed = Math.sqrt(boid.vx * boid.vx + boid.vy * boid.vy);
      if (speed > maxSpeed) {
        boid.vx = (boid.vx / speed) * maxSpeed;
        boid.vy = (boid.vy / speed) * maxSpeed;
      }

      // minimum speed bhi rakho — boid ruka nahi rehna chahiye
      const minSpeed = maxSpeed * 0.3;
      if (speed < minSpeed && speed > 0) {
        boid.vx = (boid.vx / speed) * minSpeed;
        boid.vy = (boid.vy / speed) * minSpeed;
      }

      // position update
      boid.x += boid.vx;
      boid.y += boid.vy;

      // hard clamp — agar wall avoidance ke baad bhi bahar gaya toh wapas laao
      if (boid.x < 0) boid.x = 0;
      if (boid.x > canvasW) boid.x = canvasW;
      if (boid.y < 0) boid.y = 0;
      if (boid.y > canvasH) boid.y = canvasH;
    }
  }

  // --- Find hovered boid — mouse ke sabse paas wala ---
  function findHoveredBoid() {
    if (mouseX < 0 || mouseY < 0) {
      hoveredBoid = null;
      return;
    }
    let closest = null;
    let closestDist = 30; // 30px ke andar hona chahiye
    for (const boid of boids) {
      const dx = boid.x - mouseX;
      const dy = boid.y - mouseY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < closestDist) {
        closestDist = dist;
        closest = boid;
      }
    }
    hoveredBoid = closest;
  }

  // --- Render ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // scare points dikhao — red pulsating circles
    const now = performance.now();
    for (const scare of scarePoints) {
      const age = (now - scare.birth) / SCARE_DURATION;
      const alpha = Math.max(0, 0.4 * (1 - age));
      const radius = SCARE_RADIUS * (0.5 + 0.5 * age);
      ctx.beginPath();
      ctx.arc(scare.x, scare.y, radius, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,80,80,' + alpha + ')';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // inner pulse
      ctx.beginPath();
      ctx.arc(scare.x, scare.y, SCARE_RADIUS * 0.3 * (1 - age), 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,80,80,' + (alpha * 0.5) + ')';
      ctx.fill();
    }

    // hovered boid ka visual range circle dikhao
    if (hoveredBoid) {
      ctx.beginPath();
      ctx.arc(hoveredBoid.x, hoveredBoid.y, visualRange, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // separation zone bhi dikhao — inner ring
      const sepDist = visualRange * 0.4;
      ctx.beginPath();
      ctx.arc(hoveredBoid.x, hoveredBoid.y, sepDist, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,100,100,0.15)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // boids render karo
    for (const boid of boids) {
      const angle = Math.atan2(boid.vy, boid.vx);
      // heading se HSL hue calculate — 0-360 range
      const hue = ((angle * 180 / Math.PI) + 360) % 360;

      // trail draw — alpha fade ke saath
      if (boid.trail.length > 1) {
        for (let i = 1; i < boid.trail.length; i++) {
          const alpha = (i / boid.trail.length) * 0.3;
          ctx.beginPath();
          ctx.moveTo(boid.trail[i - 1].x, boid.trail[i - 1].y);
          ctx.lineTo(boid.trail[i].x, boid.trail[i].y);
          ctx.strokeStyle = 'hsla(' + hue + ',80%,60%,' + alpha + ')';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        // trail se current position tak bhi line
        if (boid.trail.length > 0) {
          const last = boid.trail[boid.trail.length - 1];
          const alpha = 0.3;
          ctx.beginPath();
          ctx.moveTo(last.x, last.y);
          ctx.lineTo(boid.x, boid.y);
          ctx.strokeStyle = 'hsla(' + hue + ',80%,60%,' + alpha + ')';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // triangle draw — velocity direction mein point karta hai
      const size = BOID_SIZE;
      const isHovered = boid === hoveredBoid;
      const brightness = isHovered ? '75%' : '60%';
      const bodyAlpha = isHovered ? 1.0 : 0.85;

      ctx.save();
      ctx.translate(boid.x, boid.y);
      ctx.rotate(angle);

      ctx.beginPath();
      // triangle — nose aage, do corners peeche
      ctx.moveTo(size, 0);
      ctx.lineTo(-size * 0.6, -size * 0.5);
      ctx.lineTo(-size * 0.6, size * 0.5);
      ctx.closePath();

      ctx.fillStyle = 'hsla(' + hue + ',80%,' + brightness + ',' + bodyAlpha + ')';
      ctx.fill();

      // hovered boid ke liye glow
      if (isHovered) {
        ctx.strokeStyle = 'hsla(' + hue + ',80%,70%,0.6)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      ctx.restore();
    }

    updateStats();
  }

  // --- Animation Loop ---
  function animate(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    if (lastTime === 0) lastTime = timestamp;
    const dt = Math.min(timestamp - lastTime, 50); // cap at 50ms — slowdowns handle karo
    lastTime = timestamp;

    // physics update — fixed step ke saath
    if (dt > 0) {
      findHoveredBoid();
      updateBoids();
    }

    draw();
    animationId = requestAnimationFrame(animate);
  }

  // --- Mouse events ---
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;
  });

  canvas.addEventListener('mouseleave', () => {
    mouseX = -1;
    mouseY = -1;
    hoveredBoid = null;
  });

  canvas.addEventListener('click', (e) => {
    if (!predatorMode) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    scarePoints.push({ x, y, birth: performance.now() });
  });

  // touch support — mobile ke liye
  canvas.addEventListener('touchstart', (e) => {
    if (!predatorMode) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    scarePoints.push({ x, y, birth: performance.now() });
  }, { passive: false });

  // --- IntersectionObserver — sirf jab dikhe tab animate karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    lastTime = 0;
    animationId = requestAnimationFrame(animate);
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
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) animate(); });

  // tab switch pe pause — battery bachao
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // --- Shuru karo! Initial flock banao ---
  initFlock();
}
