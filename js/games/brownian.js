// ============================================================
// Brownian Motion — Random walk particles, diffusion ka live demo
// Tracked particle ka full trail dikhta hai, MSD plot saath mein
// Box-Muller transform se Gaussian random steps, ⟨r²⟩ = 4Dt verify karo
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, particles chhodo
export function initBrownian() {
  const container = document.getElementById('brownianContainer');
  if (!container) {
    console.warn('brownianContainer nahi mila bhai, Brownian demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b'; // amber accent
  const ACCENT_RGB = '245,158,11';
  const BG_COLOR = '#0a0a0a';
  const SIM_FRAC = 0.70; // left 70% simulation, right 30% MSD plot
  const MSD_SAMPLES = 300; // kitne MSD data points rakhne hain
  const MSD_WINDOW = 200; // har MSD sample mein kitne particles se average lein

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let simW = 0; // simulation area width
  let plotX = 0; // MSD plot ka left edge

  // tunable params — sliders se change honge
  let particleCount = 200;
  let stepSize = 2.0; // temperature proxy — bada step = zyada energy
  let showAllTrails = false;
  let isPaused = false;

  // particles array — har particle mein {x, y, color, startX, startY, trail[]}
  let particles = [];
  let trackedIdx = 0; // konsa particle tracked hai — trail pura dikhega

  // MSD tracking — time series data
  let msdData = []; // [{t, msd}] — measured MSD over time
  let frameCount = 0;
  let diffusionCoeff = 0; // estimated D from data

  // short trail length for non-tracked particles jab showAllTrails on ho
  const SHORT_TRAIL = 15;

  // animation
  let animationId = null;
  let isVisible = false;

  // warm color palette — particles ke liye
  const WARM_PALETTE = [
    '#ff6b6b', '#ee5a24', '#f0932b', '#f6e58d',
    '#ffbe76', '#ff7979', '#e056fd', '#be2edd',
    '#7ed6df', '#22a6b3', '#c7ecee', '#dff9fb',
    '#f9ca24', '#f0932b', '#eb4d4b', '#6ab04c',
    '#badc58', '#30336b', '#e056fd', '#686de0',
  ];

  // --- DOM structure banao ---
  // pehle ke children hata do lekin container ke existing non-canvas children preserve karo
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — simulation + MSD plot dono ek hi canvas pe
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:' + BG_COLOR,
    'touch-action:none',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls container ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Helper: slider banao ---
  function createSlider(label, min, max, step, value, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#6b6b6b;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;";
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.style.cssText = 'width:80px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
    wrapper.appendChild(slider);

    const valSpan = document.createElement('span');
    valSpan.style.cssText = "color:#f0f0f0;font-size:11px;font-family:'JetBrains Mono',monospace;min-width:30px;";
    const decimals = step < 1 ? (step < 0.01 ? 2 : 1) : 0;
    valSpan.textContent = Number(value).toFixed(decimals);
    wrapper.appendChild(valSpan);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valSpan.textContent = v.toFixed(decimals);
      onChange(v);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, valSpan };
  }

  // --- Helper: button banao ---
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:11px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      "font-family:'JetBrains Mono',monospace",
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

  // --- Helper: toggle checkbox ---
  function createToggle(label, checked, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:4px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = "color:#6b6b6b;font-size:11px;font-family:'JetBrains Mono',monospace;";
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = checked;
    cb.style.cssText = 'accent-color:' + ACCENT + ';cursor:pointer;';
    cb.addEventListener('change', () => { onChange(cb.checked); });
    wrapper.appendChild(cb);

    controlsDiv.appendChild(wrapper);
    return cb;
  }

  // --- Controls banao ---
  createSlider('Particles', 50, 500, 10, particleCount, (v) => {
    particleCount = Math.round(v);
    initParticles();
  });

  createSlider('Temp', 0.5, 6.0, 0.1, stepSize, (v) => {
    stepSize = v;
  });

  createToggle('Trails', showAllTrails, (v) => {
    showAllTrails = v;
    // trails clear karo jab toggle off ho — memory bachao
    if (!v) {
      particles.forEach((p, i) => {
        if (i !== trackedIdx) p.trail = [];
      });
    }
  });

  const pauseBtn = createButton('Pause', () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Play' : 'Pause';
  });

  createButton('Reset Track', () => {
    resetTrackedParticle();
  });

  // --- Canvas sizing — DPR aware ---
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

    // simulation aur plot areas recalculate karo
    simW = Math.floor(canvasW * SIM_FRAC);
    plotX = simW;
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // ============================================================
  // BOX-MULLER TRANSFORM — uniform random se Gaussian random banao
  // do uniform samples lo, do Gaussian samples nikalo
  // ye Brownian motion ke liye zaroori hai — random steps Gaussian hone chahiye
  // ============================================================
  function gaussianRandom() {
    let u1 = 0, u2 = 0;
    // 0 nahi chahiye — log(0) = -Infinity
    while (u1 === 0) u1 = Math.random();
    while (u2 === 0) u2 = Math.random();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  // ============================================================
  // PARTICLE INITIALIZATION — random positions aur colors set karo
  // ============================================================
  function initParticles() {
    particles = [];
    const margin = 20;

    for (let i = 0; i < particleCount; i++) {
      const x = margin + Math.random() * (simW - 2 * margin);
      const y = margin + Math.random() * (canvasH - 2 * margin);
      particles.push({
        x: x,
        y: y,
        startX: x, // initial position — MSD calculate karne ke liye
        startY: y,
        color: WARM_PALETTE[i % WARM_PALETTE.length],
        trail: [], // trail points — tracked particle ke liye full, baaki ke liye short
      });
    }

    // pehla particle tracked hai — center mein rakh do
    trackedIdx = 0;
    particles[0].x = simW / 2;
    particles[0].y = canvasH / 2;
    particles[0].startX = simW / 2;
    particles[0].startY = simW / 2;
    particles[0].trail = [];

    // MSD data reset
    msdData = [];
    frameCount = 0;
    diffusionCoeff = 0;
  }

  // tracked particle reset — click se ya button se
  function resetTrackedParticle() {
    const p = particles[trackedIdx];
    if (!p) return;
    p.x = simW / 2;
    p.y = canvasH / 2;
    p.startX = simW / 2;
    p.startY = canvasH / 2;
    p.trail = [];

    // sab particles ke start positions reset karo — MSD ke liye
    particles.forEach((pp) => {
      pp.startX = pp.x;
      pp.startY = pp.y;
    });

    msdData = [];
    frameCount = 0;
    diffusionCoeff = 0;
  }

  // ============================================================
  // PHYSICS UPDATE — random walk step for each particle
  // ============================================================
  function updateParticles() {
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];

      // Gaussian random step — Brownian motion ka core
      const dx = gaussianRandom() * stepSize;
      const dy = gaussianRandom() * stepSize;

      p.x += dx;
      p.y += dy;

      // boundary bounce — canvas ke andar rakh
      if (p.x < 0) { p.x = -p.x; }
      if (p.x > simW) { p.x = 2 * simW - p.x; }
      if (p.y < 0) { p.y = -p.y; }
      if (p.y > canvasH) { p.y = 2 * canvasH - p.y; }

      // clamp — edge case ke liye (double bounce se bahar na nikle)
      p.x = Math.max(0, Math.min(simW, p.x));
      p.y = Math.max(0, Math.min(canvasH, p.y));

      // trail update
      if (i === trackedIdx) {
        // tracked particle — full trail rakh, no limit
        p.trail.push({ x: p.x, y: p.y });
      } else if (showAllTrails) {
        // baaki particles — short trail
        p.trail.push({ x: p.x, y: p.y });
        if (p.trail.length > SHORT_TRAIL) {
          p.trail.shift();
        }
      }
    }

    // MSD calculate karo — sab particles ka average squared displacement
    frameCount++;
    if (frameCount % 3 === 0) { // har 3 frames mein ek sample — performance ke liye
      let sumR2 = 0;
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        const dx = p.x - p.startX;
        const dy = p.y - p.startY;
        sumR2 += dx * dx + dy * dy;
      }
      const msd = sumR2 / particles.length;
      msdData.push({ t: frameCount, msd: msd });

      // samples limit karo
      if (msdData.length > MSD_SAMPLES) {
        msdData.shift();
      }

      // diffusion coefficient estimate karo — ⟨r²⟩ = 4Dt se
      // simple: D = msd / (4 * t)
      // t ko frame count se proportional maano
      if (frameCount > 30) {
        diffusionCoeff = msd / (4 * frameCount);
      }
    }
  }

  // ============================================================
  // DRAWING — simulation area + MSD plot
  // ============================================================
  function draw() {
    // clear with background
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, canvasW, canvasH);

    // --- Simulation area draw karo ---
    drawSimulation();

    // --- Divider line — sim aur plot ke beech ---
    ctx.beginPath();
    ctx.moveTo(simW, 0);
    ctx.lineTo(simW, canvasH);
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.15)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // --- MSD plot draw karo ---
    drawMSDPlot();
  }

  function drawSimulation() {
    // clip to simulation area
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, simW, canvasH);
    ctx.clip();

    // --- Non-tracked particles ki short trails (agar enabled) ---
    if (showAllTrails) {
      for (let i = 0; i < particles.length; i++) {
        if (i === trackedIdx) continue;
        const p = particles[i];
        if (p.trail.length < 2) continue;

        ctx.beginPath();
        ctx.moveTo(p.trail[0].x, p.trail[0].y);
        for (let j = 1; j < p.trail.length; j++) {
          ctx.lineTo(p.trail[j].x, p.trail[j].y);
        }
        ctx.strokeStyle = p.color + '30'; // bahut halka — 0x30 alpha
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }

    // --- Tracked particle ka full trail — rainbow color by time ---
    const tracked = particles[trackedIdx];
    if (tracked && tracked.trail.length > 1) {
      const tLen = tracked.trail.length;
      for (let i = 1; i < tLen; i++) {
        const t = i / tLen; // 0 to 1 — old to new
        // rainbow hue — time ke saath cycle karo
        const hue = (i * 1.2) % 360;
        const alpha = 0.3 + t * 0.6; // purana faded, naya bright
        const width = 0.8 + t * 1.5;

        ctx.beginPath();
        ctx.moveTo(tracked.trail[i - 1].x, tracked.trail[i - 1].y);
        ctx.lineTo(tracked.trail[i].x, tracked.trail[i].y);
        ctx.strokeStyle = 'hsla(' + hue + ',85%,60%,' + alpha + ')';
        ctx.lineWidth = width;
        ctx.stroke();
      }
    }

    // --- Particles draw karo — chhotey dots ---
    for (let i = 0; i < particles.length; i++) {
      if (i === trackedIdx) continue; // tracked particle baad mein — upar dikhna chahiye
      const p = particles[i];

      ctx.beginPath();
      ctx.arc(p.x, p.y, 1.8, 0, Math.PI * 2);
      ctx.fillStyle = p.color + '90'; // semi-transparent
      ctx.fill();
    }

    // --- Tracked particle — bada aur bright ---
    if (tracked) {
      // glow effect
      ctx.save();
      ctx.shadowColor = '#fbbf24';
      ctx.shadowBlur = 15;

      ctx.beginPath();
      ctx.arc(tracked.x, tracked.y, 5, 0, Math.PI * 2);

      // gradient — center gold, edge fade
      const grad = ctx.createRadialGradient(tracked.x, tracked.y, 0, tracked.x, tracked.y, 5);
      grad.addColorStop(0, '#fef3c7');
      grad.addColorStop(0.5, '#fbbf24');
      grad.addColorStop(1, '#f59e0b');
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.restore();

      // outer ring — visual indicator
      ctx.beginPath();
      ctx.arc(tracked.x, tracked.y, 8, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(251,191,36,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // --- Info text ---
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.fillText('N=' + particles.length + '  T=' + stepSize.toFixed(1), 8, 14);

    if (tracked && tracked.trail.length > 10) {
      const dx = tracked.x - tracked.startX;
      const dy = tracked.y - tracked.startY;
      const r = Math.sqrt(dx * dx + dy * dy);
      ctx.fillText('r=' + r.toFixed(1) + '  steps=' + tracked.trail.length, 8, 26);
    }

    // hint jab fresh shuru ho
    if (frameCount < 60) {
      ctx.font = "12px 'JetBrains Mono', monospace";
      ctx.fillStyle = 'rgba(176,176,176,0.2)';
      ctx.textAlign = 'center';
      ctx.fillText('click to reposition tracked particle', simW / 2, canvasH - 12);
    }

    ctx.restore(); // clip restore
  }

  // ============================================================
  // MSD PLOT — ⟨r²⟩ vs time, theoretical line saath mein
  // ============================================================
  function drawMSDPlot() {
    const plotW = canvasW - plotX;
    const plotPad = { top: 30, right: 15, bottom: 35, left: 45 };
    const graphX = plotX + plotPad.left;
    const graphY = plotPad.top;
    const graphW = plotW - plotPad.left - plotPad.right;
    const graphH = canvasH - plotPad.top - plotPad.bottom;

    // --- Background ---
    ctx.fillStyle = 'rgba(10,10,10,0.8)';
    ctx.fillRect(plotX, 0, plotW, canvasH);

    // --- Title ---
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.7)';
    ctx.fillText('\u27E8r\u00B2\u27E9 vs time', plotX + plotW / 2, 16);

    // --- Subtle grid ---
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 0.5;
    const gridLines = 5;
    for (let i = 0; i <= gridLines; i++) {
      // horizontal
      const gy = graphY + (graphH * i / gridLines);
      ctx.beginPath();
      ctx.moveTo(graphX, gy);
      ctx.lineTo(graphX + graphW, gy);
      ctx.stroke();

      // vertical
      const gx = graphX + (graphW * i / gridLines);
      ctx.beginPath();
      ctx.moveTo(gx, graphY);
      ctx.lineTo(gx, graphY + graphH);
      ctx.stroke();
    }

    // --- Axes ---
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    // Y axis
    ctx.beginPath();
    ctx.moveTo(graphX, graphY);
    ctx.lineTo(graphX, graphY + graphH);
    ctx.stroke();
    // X axis
    ctx.beginPath();
    ctx.moveTo(graphX, graphY + graphH);
    ctx.lineTo(graphX + graphW, graphY + graphH);
    ctx.stroke();

    // --- Axis labels ---
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.textAlign = 'center';
    ctx.fillText('time', plotX + plotW / 2, canvasH - 5);

    ctx.save();
    ctx.translate(plotX + 12, canvasH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('\u27E8r\u00B2\u27E9', 0, 0);
    ctx.restore();

    if (msdData.length < 2) return;

    // data ranges nikaalo
    const tMin = msdData[0].t;
    const tMax = msdData[msdData.length - 1].t;
    let msdMax = 0;
    for (let i = 0; i < msdData.length; i++) {
      if (msdData[i].msd > msdMax) msdMax = msdData[i].msd;
    }
    // theoretical line ka max bhi check karo
    const theorMax = 4 * diffusionCoeff * tMax;
    if (theorMax > msdMax) msdMax = theorMax;

    if (msdMax < 1) msdMax = 1; // zero division se bacho
    msdMax *= 1.15; // thoda headroom

    const tRange = tMax - tMin || 1;

    // helper — data to pixel
    function toPixel(t, msd) {
      const px = graphX + ((t - tMin) / tRange) * graphW;
      const py = graphY + graphH - (msd / msdMax) * graphH;
      return { px, py };
    }

    // --- Theoretical line: ⟨r²⟩ = 4Dt ---
    if (diffusionCoeff > 0) {
      ctx.beginPath();
      const thStart = toPixel(tMin, 4 * diffusionCoeff * tMin);
      ctx.moveTo(thStart.px, thStart.py);

      const steps = 30;
      for (let i = 1; i <= steps; i++) {
        const t = tMin + (tRange * i / steps);
        const msd = 4 * diffusionCoeff * t;
        const pt = toPixel(t, msd);
        ctx.lineTo(pt.px, pt.py);
      }
      ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // label
      ctx.font = "8px 'JetBrains Mono', monospace";
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.6)';
      ctx.textAlign = 'right';
      ctx.fillText('4Dt', graphX + graphW - 4, graphY + 14);
    }

    // --- Measured MSD data line ---
    ctx.beginPath();
    const first = toPixel(msdData[0].t, msdData[0].msd);
    ctx.moveTo(first.px, first.py);

    for (let i = 1; i < msdData.length; i++) {
      const pt = toPixel(msdData[i].t, msdData[i].msd);
      ctx.lineTo(pt.px, pt.py);
    }
    ctx.strokeStyle = '#34d399'; // green — measured data
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // data label
    ctx.font = "8px 'JetBrains Mono', monospace";
    ctx.fillStyle = '#34d399';
    ctx.textAlign = 'left';
    ctx.fillText('measured', graphX + 4, graphY + 14);

    // --- D estimate text ---
    if (diffusionCoeff > 0) {
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillStyle = 'rgba(176,176,176,0.4)';
      ctx.textAlign = 'center';
      ctx.fillText('D\u2248' + diffusionCoeff.toFixed(2), plotX + plotW / 2, canvasH - 18);
    }

    // --- Y axis ticks ---
    ctx.font = "8px 'JetBrains Mono', monospace";
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = (msdMax * i / 4);
      const pt = toPixel(tMin, val);
      // short format — k for thousands
      let label;
      if (val >= 1000) {
        label = (val / 1000).toFixed(1) + 'k';
      } else {
        label = val.toFixed(0);
      }
      ctx.fillText(label, graphX - 4, pt.py + 3);
    }
  }

  // ============================================================
  // CLICK HANDLER — canvas pe click karo toh tracked particle wahan move ho
  // ============================================================
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // sirf simulation area mein click valid hai
    if (mx < simW && mx > 0 && my > 0 && my < canvasH) {
      const p = particles[trackedIdx];
      if (p) {
        p.x = mx;
        p.y = my;
        p.startX = mx;
        p.startY = my;
        p.trail = [];

        // sab particles ka start reset karo — MSD fresh ho jaaye
        particles.forEach((pp) => {
          pp.startX = pp.x;
          pp.startY = pp.y;
        });

        msdData = [];
        frameCount = 0;
        diffusionCoeff = 0;
      }
    }
  });

  // ============================================================
  // ANIMATION LOOP — rAF based
  // ============================================================
  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    if (!isPaused) {
      updateParticles();
    }
    draw();

    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — off-screen pe pause karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
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

  // tab switch pe bhi pause karo — battery bachao
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // --- Initialize aur shuru karo ---
  initParticles();
}
