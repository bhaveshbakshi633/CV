// ============================================================
// Particle Life (Artificial Life) — Emergent behavior from simple rules
// Har particle group ka doosre groups ke saath attraction/repulsion hota hai
// Random rules se cells, chains, orbits, chaos — sab banta hai
// ============================================================

// yahi function export hoga — container dhundho, canvas banao, life simulate karo
export function initParticleLife() {
  const container = document.getElementById('particleLifeContainer');
  if (!container) {
    console.warn('particleLifeContainer nahi mila bhai, particle life skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b'; // amber accent — portfolio theme
  const ACCENT_RGB = '245,158,11';

  // particle group colors — 6 species, har ek ka apna rang
  const GROUP_COLORS = [
    { name: 'Red',     hex: '#ff4444', r: 255, g: 68,  b: 68  },
    { name: 'Green',   hex: '#44ff66', r: 68,  g: 255, b: 102 },
    { name: 'Blue',    hex: '#4488ff', r: 68,  g: 136, b: 255 },
    { name: 'Yellow',  hex: '#ffdd44', r: 255, g: 221, b: 68  },
    { name: 'Cyan',    hex: '#44ffee', r: 68,  g: 255, b: 238 },
    { name: 'Magenta', hex: '#ff44dd', r: 255, g: 68,  b: 221 },
  ];
  const NUM_GROUPS = GROUP_COLORS.length;

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let particles = []; // [{x, y, vx, vy, group}]
  let attractionMatrix = []; // NxN array — -1 to +1
  let particleCount = 400;
  let friction = 0.15; // damping per frame — velocity *= (1 - friction * dt)
  let interactionRadius = 80; // kitni door tak force lagegi
  let isPaused = false;
  let isVisible = false;
  let animationId = null;
  let lastTime = 0;

  // --- Attraction matrix initialize karo — random values ---
  function randomizeMatrix() {
    attractionMatrix = [];
    for (let i = 0; i < NUM_GROUPS; i++) {
      attractionMatrix[i] = [];
      for (let j = 0; j < NUM_GROUPS; j++) {
        // -1 se +1 ke beech random value
        attractionMatrix[i][j] = Math.random() * 2 - 1;
      }
    }
    updateMatrixDisplay();
  }

  // ek specific cell ko randomize karo — click pe
  function randomizeCell(i, j) {
    attractionMatrix[i][j] = Math.random() * 2 - 1;
    updateMatrixDisplay();
  }

  // --- Preset rule sets — tested combos jo acche patterns dete hain ---
  const PRESETS = {
    'Cells': () => {
      // cell-like clusters — same group attract, others repel
      for (let i = 0; i < NUM_GROUPS; i++) {
        for (let j = 0; j < NUM_GROUPS; j++) {
          if (i === j) {
            attractionMatrix[i][j] = 0.5 + Math.random() * 0.3; // self-attract strongly
          } else {
            attractionMatrix[i][j] = -0.3 - Math.random() * 0.5; // others repel
          }
        }
      }
      updateMatrixDisplay();
    },
    'Chains': () => {
      // chain formations — sequential groups attract each other
      for (let i = 0; i < NUM_GROUPS; i++) {
        for (let j = 0; j < NUM_GROUPS; j++) {
          if (i === j) {
            attractionMatrix[i][j] = 0.1 + Math.random() * 0.2;
          } else if (j === (i + 1) % NUM_GROUPS) {
            // next group ko attract karo — chain banegi
            attractionMatrix[i][j] = 0.6 + Math.random() * 0.3;
          } else if (j === (i - 1 + NUM_GROUPS) % NUM_GROUPS) {
            // previous group se slightly repel
            attractionMatrix[i][j] = -0.1 - Math.random() * 0.2;
          } else {
            attractionMatrix[i][j] = -0.2 + Math.random() * 0.3 - 0.15;
          }
        }
      }
      updateMatrixDisplay();
    },
    'Orbits': () => {
      // circular orbital patterns — asymmetric attractions
      for (let i = 0; i < NUM_GROUPS; i++) {
        for (let j = 0; j < NUM_GROUPS; j++) {
          if (i === j) {
            attractionMatrix[i][j] = -0.1 + Math.random() * 0.15; // weak self interaction
          } else if (j === (i + 1) % NUM_GROUPS) {
            attractionMatrix[i][j] = 0.8 + Math.random() * 0.2; // strongly chase next
          } else if (j === (i + 2) % NUM_GROUPS) {
            attractionMatrix[i][j] = 0.3 + Math.random() * 0.2;
          } else {
            attractionMatrix[i][j] = -0.5 - Math.random() * 0.3; // repel others
          }
        }
      }
      updateMatrixDisplay();
    },
    'Chaos': () => {
      // full random — kuch bhi ho sakta hai
      randomizeMatrix();
    },
  };

  // --- Particles create karo ---
  function createParticles(count) {
    particles = [];
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * canvasW,
        y: Math.random() * canvasH,
        vx: 0,
        vy: 0,
        group: Math.floor(Math.random() * NUM_GROUPS),
      });
    }
  }

  // burst of particles at specific position — canvas click pe
  function addBurst(cx, cy, count) {
    const burstRadius = 25;
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const dist = Math.random() * burstRadius;
      particles.push({
        x: cx + Math.cos(angle) * dist,
        y: cy + Math.sin(angle) * dist,
        vx: (Math.random() - 0.5) * 30,
        vy: (Math.random() - 0.5) * 30,
        group: Math.floor(Math.random() * NUM_GROUPS),
      });
    }
    // agar limit se zyada ho gaye toh purane hata do
    if (particles.length > 1200) {
      particles = particles.slice(particles.length - 1200);
    }
    updateParticleCountDisplay();
  }

  // --- Force function ---
  // ramp up from 0 at dist=0 to max at dist=rMax/3, then fall off to 0 at dist=rMax
  // ye smooth force curve deta hai — no sudden jumps
  function forceFunction(dist, rMax) {
    if (dist <= 0 || dist >= rMax) return 0;

    const beta = 0.3; // peak force distance as fraction of rMax
    const rPeak = rMax * beta;

    if (dist < rPeak) {
      // ramp up — linear from 0 to 1
      return dist / rPeak;
    } else {
      // fall off — linear from 1 to 0
      return 1 - (dist - rPeak) / (rMax - rPeak);
    }
  }

  // --- Physics step — N² brute force, <1000 particles pe 60fps chalega ---
  function physicsStep(dt) {
    if (particles.length === 0) return;

    const rMax = interactionRadius;
    const rMaxSq = rMax * rMax;
    // force strength — tune kiya hai taaki particles naturally move karein
    const forceStrength = 800;

    // har particle ke liye net force calculate karo
    for (let i = 0; i < particles.length; i++) {
      let fx = 0, fy = 0;
      const pi = particles[i];

      for (let j = 0; j < particles.length; j++) {
        if (i === j) continue;
        const pj = particles[j];

        // wrap-around distance — toroidal world
        let dx = pj.x - pi.x;
        let dy = pj.y - pi.y;

        // nearest image — wrap check
        if (dx > canvasW * 0.5) dx -= canvasW;
        else if (dx < -canvasW * 0.5) dx += canvasW;
        if (dy > canvasH * 0.5) dy -= canvasH;
        else if (dy < -canvasH * 0.5) dy += canvasH;

        const distSq = dx * dx + dy * dy;
        if (distSq >= rMaxSq || distSq < 0.01) continue;

        const dist = Math.sqrt(distSq);
        const attraction = attractionMatrix[pi.group][pj.group];

        // force magnitude — attraction * forceFunction curve
        const forceMag = attraction * forceFunction(dist, rMax) * forceStrength;

        // direction normalize karke force apply karo
        fx += (dx / dist) * forceMag;
        fy += (dy / dist) * forceMag;
      }

      // acceleration apply karo
      pi.vx += fx * dt;
      pi.vy += fy * dt;
    }

    // friction/damping — velocity ko slowly reduce karo
    // ye ensures particles infinite acceleration nahi karein
    const dampFactor = 1 - friction;
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.vx *= dampFactor;
      p.vy *= dampFactor;

      // position update
      p.x += p.vx * dt;
      p.y += p.vy * dt;

      // wrap-around boundaries — toroidal world
      if (p.x < 0) p.x += canvasW;
      else if (p.x >= canvasW) p.x -= canvasW;
      if (p.y < 0) p.y += canvasH;
      else if (p.y >= canvasH) p.y -= canvasH;
    }
  }

  // --- DOM Structure ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — particles yahan render honge
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:rgba(2,2,8,0.5)',
    'position:relative',
  ].join(';');
  container.appendChild(canvas);

  // --- Attraction Matrix Display ---
  // small NxN grid canvas ke corner mein — colored cells show karenge
  const matrixOverlay = document.createElement('div');
  matrixOverlay.style.cssText = [
    'position:absolute',
    'top:8px',
    'left:8px',
    'display:grid',
    'grid-template-columns:16px repeat(' + NUM_GROUPS + ', 18px)',
    'grid-template-rows:16px repeat(' + NUM_GROUPS + ', 18px)',
    'gap:1px',
    'background:rgba(0,0,0,0.7)',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.2)',
    'border-radius:6px',
    'padding:4px',
    'cursor:pointer',
    'z-index:2',
    'backdrop-filter:blur(4px)',
  ].join(';');

  // matrix wrapper — canvas ke andar position karne ke liye
  const canvasWrapper = document.createElement('div');
  canvasWrapper.style.cssText = 'position:relative;width:100%;';
  container.insertBefore(canvasWrapper, container.firstChild);
  canvasWrapper.appendChild(canvas);
  canvasWrapper.appendChild(matrixOverlay);

  // matrix cells banao — header row + header col + NxN cells
  const matrixCells = [];

  // top-left corner — empty
  const cornerCell = document.createElement('div');
  cornerCell.style.cssText = 'width:16px;height:16px;';
  matrixOverlay.appendChild(cornerCell);

  // top header — column group indicators
  for (let j = 0; j < NUM_GROUPS; j++) {
    const hdr = document.createElement('div');
    hdr.style.cssText = [
      'width:18px',
      'height:16px',
      'display:flex',
      'align-items:center',
      'justify-content:center',
      'font-family:"JetBrains Mono",monospace',
      'font-size:7px',
      'color:' + GROUP_COLORS[j].hex,
      'opacity:0.8',
    ].join(';');
    hdr.textContent = GROUP_COLORS[j].name[0]; // pehla letter
    matrixOverlay.appendChild(hdr);
  }

  // data rows — row header + cells
  for (let i = 0; i < NUM_GROUPS; i++) {
    // row header — group indicator
    const rowHdr = document.createElement('div');
    rowHdr.style.cssText = [
      'width:16px',
      'height:18px',
      'display:flex',
      'align-items:center',
      'justify-content:center',
      'font-family:"JetBrains Mono",monospace',
      'font-size:7px',
      'color:' + GROUP_COLORS[i].hex,
      'opacity:0.8',
    ].join(';');
    rowHdr.textContent = GROUP_COLORS[i].name[0];
    matrixOverlay.appendChild(rowHdr);

    matrixCells[i] = [];
    for (let j = 0; j < NUM_GROUPS; j++) {
      const cell = document.createElement('div');
      cell.style.cssText = [
        'width:18px',
        'height:18px',
        'border-radius:3px',
        'cursor:pointer',
        'transition:transform 0.15s',
        'display:flex',
        'align-items:center',
        'justify-content:center',
        'font-family:"JetBrains Mono",monospace',
        'font-size:7px',
        'color:rgba(255,255,255,0.6)',
      ].join(';');
      // click pe individual cell randomize karo
      const ci = i, cj = j;
      cell.addEventListener('click', (e) => {
        e.stopPropagation();
        randomizeCell(ci, cj);
        cell.style.transform = 'scale(1.3)';
        setTimeout(() => { cell.style.transform = 'scale(1)'; }, 150);
      });
      cell.addEventListener('mouseenter', () => {
        cell.style.transform = 'scale(1.15)';
      });
      cell.addEventListener('mouseleave', () => {
        cell.style.transform = 'scale(1)';
      });
      matrixOverlay.appendChild(cell);
      matrixCells[i][j] = cell;
    }
  }

  // matrix display update karo — colors set karo based on values
  function updateMatrixDisplay() {
    for (let i = 0; i < NUM_GROUPS; i++) {
      for (let j = 0; j < NUM_GROUPS; j++) {
        const val = attractionMatrix[i][j];
        const cell = matrixCells[i][j];

        // -1 = bright red (repel), 0 = dark/neutral, +1 = bright green (attract)
        let bgColor;
        if (val > 0) {
          const intensity = val; // 0 to 1
          bgColor = 'rgba(68,255,102,' + (intensity * 0.7 + 0.05).toFixed(2) + ')';
        } else {
          const intensity = -val; // 0 to 1
          bgColor = 'rgba(255,68,68,' + (intensity * 0.7 + 0.05).toFixed(2) + ')';
        }
        cell.style.background = bgColor;
        // value text — rounded to 1 decimal
        cell.textContent = val.toFixed(1);
      }
    }
  }

  // particle count display — canvas ke top-right mein
  const countDisplay = document.createElement('div');
  countDisplay.style.cssText = [
    'position:absolute',
    'top:8px',
    'right:8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:rgba(' + ACCENT_RGB + ',0.6)',
    'background:rgba(0,0,0,0.5)',
    'padding:4px 8px',
    'border-radius:4px',
    'backdrop-filter:blur(4px)',
    'z-index:2',
    'pointer-events:none',
  ].join(';');
  canvasWrapper.appendChild(countDisplay);

  function updateParticleCountDisplay() {
    countDisplay.textContent = 'n=' + particles.length;
  }

  // --- Controls ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // button helper — consistent styling
  function makeButton(text, parent, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'border-radius:6px',
      'padding:5px 12px',
      'font-size:11px',
      'font-family:"JetBrains Mono",monospace',
      'cursor:pointer',
      'transition:all 0.2s',
      'user-select:none',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#ffffff';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    parent.appendChild(btn);
    return btn;
  }

  // slider helper — label + range input + value display
  function makeSlider(label, min, max, step, value, parent, onChange) {
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
    slider.style.cssText = 'width:70px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
    wrapper.appendChild(slider);

    const valSpan = document.createElement('span');
    valSpan.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.7);font-size:10px;font-family:"JetBrains Mono",monospace;min-width:28px;';
    valSpan.textContent = String(value);
    wrapper.appendChild(valSpan);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valSpan.textContent = Number.isInteger(v) ? String(v) : v.toFixed(2);
      onChange(v);
    });

    parent.appendChild(wrapper);
    return { slider, valSpan };
  }

  // --- Row 1: Buttons ---
  // Pause/Play
  const pauseBtn = makeButton('Pause', controlsDiv, () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Play' : 'Pause';
    if (isPaused) {
      pauseBtn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.5)';
      pauseBtn.style.color = ACCENT;
    } else {
      pauseBtn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
      pauseBtn.style.color = '#b0b0b0';
    }
  });

  // Randomize All — nayi random rules generate karo
  makeButton('Randomize All', controlsDiv, () => {
    randomizeMatrix();
  });

  // Reset — sab particles nayi positions pe, rules same
  makeButton('Reset', controlsDiv, () => {
    createParticles(particleCount);
    updateParticleCountDisplay();
  });

  // --- Presets dropdown ---
  const presetSelect = document.createElement('select');
  presetSelect.style.cssText = [
    'padding:5px 10px',
    'font-size:11px',
    'border-radius:6px',
    'cursor:pointer',
    'background:rgba(' + ACCENT_RGB + ',0.1)',
    'color:#b0b0b0',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
    'font-family:"JetBrains Mono",monospace',
  ].join(';');

  const presetNames = ['Presets...', ...Object.keys(PRESETS)];
  presetNames.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    opt.style.background = '#1a1a1a';
    opt.style.color = '#b0b0b0';
    presetSelect.appendChild(opt);
  });

  presetSelect.addEventListener('change', () => {
    const name = presetSelect.value;
    if (PRESETS[name]) {
      PRESETS[name]();
      // particles reset karo preset ke saath
      createParticles(particleCount);
      updateParticleCountDisplay();
    }
    presetSelect.value = 'Presets...';
  });
  controlsDiv.appendChild(presetSelect);

  // --- Row 2: Sliders ---
  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:6px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(slidersDiv);

  // Particle count slider
  makeSlider('Count:', 100, 1000, 50, particleCount, slidersDiv, (v) => {
    particleCount = v;
    createParticles(particleCount);
    updateParticleCountDisplay();
  });

  // Friction slider
  makeSlider('Friction:', 0.01, 0.5, 0.01, friction, slidersDiv, (v) => {
    friction = v;
  });

  // Interaction radius slider
  makeSlider('Radius:', 20, 150, 5, interactionRadius, slidersDiv, (v) => {
    interactionRadius = v;
  });

  // --- Canvas click — burst of particles add karo ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  canvas.addEventListener('click', (e) => {
    // matrix overlay pe click hua toh ignore karo
    if (e.target !== canvas) return;
    const pos = getCanvasPos(e);
    addBurst(pos.x, pos.y, 20);
  });

  canvas.addEventListener('touchstart', (e) => {
    // touch events — matrix ke upar nahi toh burst karo
    const touch = e.touches[0];
    const matrixRect = matrixOverlay.getBoundingClientRect();
    if (touch.clientX >= matrixRect.left && touch.clientX <= matrixRect.right &&
        touch.clientY >= matrixRect.top && touch.clientY <= matrixRect.bottom) {
      return; // matrix pe touch — ignore karo
    }
    e.preventDefault();
    const pos = getCanvasPos(e);
    addBurst(pos.x, pos.y, 20);
  }, { passive: false });

  // --- Canvas sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = canvasWrapper.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = Math.floor(containerWidth * dpr);
    canvas.height = Math.floor(CANVAS_HEIGHT * dpr);
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Drawing ---
  function draw() {
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);

    // particles draw karo — chhote circles with group color
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const color = GROUP_COLORS[p.group];

      // subtle glow — low alpha bigger circle underneath
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + color.r + ',' + color.g + ',' + color.b + ',0.08)';
      ctx.fill();

      // main particle dot — solid circle
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + color.r + ',' + color.g + ',' + color.b + ',0.85)';
      ctx.fill();
    }

    // hint jab koi particle nahi hai
    if (particles.length === 0) {
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('click canvas to add particles \u2022 click matrix to change rules', canvasW / 2, canvasH / 2);
    }
  }

  // --- Animation loop ---
  function animate(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    // delta time — variable timestep with cap
    if (lastTime === 0) lastTime = timestamp;
    let dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;

    // dt clamp karo — tab switch se bada dt aa sakta hai
    dt = Math.min(dt, 0.05);

    if (!isPaused) {
      // substeps for stability — 2 substeps smooth enough
      const subSteps = 2;
      const subDt = dt / subSteps;
      for (let s = 0; s < subSteps; s++) {
        physicsStep(subDt);
      }
    }

    draw();
    updateParticleCountDisplay();
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf visible hone pe animate karo ---
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

  // tab switch pe pause — CPU waste mat karo background mein
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // --- Init — sab shuru karo ---
  randomizeMatrix();
  createParticles(particleCount);
  updateParticleCountDisplay();

  // pehla frame draw karo — blank na dikhe
  draw();
}
