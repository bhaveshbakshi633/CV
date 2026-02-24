// ============================================================
// Agent-Based SIR Epidemic Model — Classic epidemiology simulation
// Susceptible → Infected → Recovered (ya Dead) ka poora natak
// Click karo kisi pe aur dekho bimari kaise failti hai
// ============================================================

// yahi function export hoga — container dhundho, canvas banao, epidemic chalaao
export function initEpidemic() {
  const container = document.getElementById('epidemicContainer');
  if (!container) {
    console.warn('epidemicContainer nahi mila bhai, epidemic sim skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff'; // blue accent — AI category
  const ACCENT_RGB = '74,158,255';
  const BG_COLOR = '#1a1a2e';

  // agent state colors
  const STATE_COLORS = {
    S: { hex: '#4a9eff', r: 74,  g: 158, b: 255, label: 'Susceptible' },
    I: { hex: '#ef4444', r: 239, g: 68,  b: 68,  label: 'Infected'    },
    R: { hex: '#10b981', r: 16,  g: 185, b: 129, label: 'Recovered'   },
    V: { hex: '#a78bfa', r: 167, g: 139, b: 250, label: 'Vaccinated'  },
    D: { hex: '#404050', r: 64,  g: 64,  b: 80,  label: 'Dead'        },
  };

  // --- Simulation State ---
  let canvasW = 0, canvasH = 0;
  let simW = 0; // left panel width (agent sim area)
  let chartW = 0; // right panel width (epidemic curve)
  let dpr = 1;

  let agents = []; // [{x, y, vx, vy, state, infectionTimer, dirChangeTimer}]
  let populationCount = 200;
  let transmissionProb = 0.05;
  let recoveryTime = 200; // frames
  let infectionRadius = 20; // px
  let vaccinationPct = 0; // 0-80
  let mortalityPct = 5; // 0-20
  let socialDistancing = false;

  // epidemic curve data — har N frames pe ek sample
  const SAMPLE_INTERVAL = 3; // har 3 frames pe record karo
  let curveData = []; // [{s, i, r, d, t}]
  let frameCount = 0;
  let peakInfections = 0;
  let peakTime = 0;
  let totalInfected = 0; // R0 estimate ke liye
  let initialInfectors = 0; // R0 ke liye — pehle generation ke infected count

  // R0 tracking — generation based
  let generationInfections = []; // har infected ne kitne infect kiye
  let r0Estimate = 0;

  // animation state
  let animationId = null;
  let isVisible = false;
  let epidemicStarted = false;

  // --- DOM Structure ---
  const existingChildren = Array.from(container.children);
  container.style.cssText = 'width:100%;position:relative;';

  // canvas wrapper — canvas ke andar overlay position karne ke liye
  const canvasWrapper = document.createElement('div');
  canvasWrapper.style.cssText = 'position:relative;width:100%;';
  container.appendChild(canvasWrapper);

  // main canvas — agents + epidemic curve dono yahan render honge
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:pointer',
    'background:' + BG_COLOR,
  ].join(';');
  canvasWrapper.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // hint overlay — "Click to infect" message
  const hintOverlay = document.createElement('div');
  hintOverlay.style.cssText = [
    'position:absolute',
    'top:50%',
    'left:32.5%',
    'transform:translate(-50%,-50%)',
    'font-family:"JetBrains Mono",monospace',
    'font-size:13px',
    'color:rgba(' + ACCENT_RGB + ',0.5)',
    'pointer-events:none',
    'z-index:2',
    'text-align:center',
    'transition:opacity 0.5s',
  ].join(';');
  hintOverlay.textContent = 'Click to infect a person';
  canvasWrapper.appendChild(hintOverlay);

  // --- Stats Panel ---
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

  // individual stat spans banao
  function makeStatSpan(color) {
    const span = document.createElement('span');
    span.style.cssText = 'color:' + color + ';white-space:nowrap;';
    statsDiv.appendChild(span);
    return span;
  }

  const statS = makeStatSpan(STATE_COLORS.S.hex);
  const statI = makeStatSpan(STATE_COLORS.I.hex);
  const statR = makeStatSpan(STATE_COLORS.R.hex);
  const statD = makeStatSpan(STATE_COLORS.D.hex);
  const statR0 = makeStatSpan('#b0b0b0');
  const statPeak = makeStatSpan('rgba(' + ACCENT_RGB + ',0.6)');

  function updateStats() {
    let sCount = 0, iCount = 0, rCount = 0, dCount = 0, vCount = 0;
    for (const a of agents) {
      if (a.state === 'S') sCount++;
      else if (a.state === 'I') iCount++;
      else if (a.state === 'R') rCount++;
      else if (a.state === 'D') dCount++;
      else if (a.state === 'V') vCount++;
    }
    // vaccinated ko susceptible mein mat gino — alag dikhao
    statS.textContent = 'S:' + sCount + (vCount > 0 ? ' V:' + vCount : '');
    statI.textContent = 'I:' + iCount;
    statR.textContent = 'R:' + rCount;
    statD.textContent = dCount > 0 ? 'D:' + dCount : '';
    statR0.textContent = 'R\u2080\u2248' + r0Estimate.toFixed(1);
    statPeak.textContent = peakInfections > 0 ? 'Peak:' + peakInfections + ' @t=' + peakTime : '';
  }

  // --- Controls ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // button factory — consistent dark theme styling
  function makeButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:6px 14px',
      'font-size:11px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
      'user-select:none',
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
  function makeSlider(label, min, max, step, value, onChange) {
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
    slider.style.cssText = 'width:65px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
    wrapper.appendChild(slider);

    const valSpan = document.createElement('span');
    valSpan.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.7);font-size:10px;font-family:"JetBrains Mono",monospace;min-width:28px;';
    valSpan.textContent = Number.isInteger(step) ? String(value) : Number(value).toFixed(2);
    wrapper.appendChild(valSpan);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valSpan.textContent = Number.isInteger(step) || step >= 1 ? String(Math.round(v)) : v.toFixed(2);
      onChange(v);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, valSpan, wrapper };
  }

  // --- Sliders row 1: buttons + main controls ---
  makeButton('Reset', () => {
    resetSimulation();
  });

  // social distancing toggle
  const sdBtn = makeButton('Distancing: OFF', () => {
    socialDistancing = !socialDistancing;
    sdBtn.textContent = socialDistancing ? 'Distancing: ON' : 'Distancing: OFF';
    if (socialDistancing) {
      sdBtn.style.background = 'rgba(16,185,129,0.25)';
      sdBtn.style.borderColor = 'rgba(16,185,129,0.5)';
      sdBtn.style.color = '#10b981';
    } else {
      sdBtn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      sdBtn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
      sdBtn.style.color = '#b0b0b0';
    }
    // existing agents ki speed update karo
    updateAgentSpeeds();
  });

  // sliders row 2
  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:6px',
    'align-items:center',
  ].join(';');
  container.appendChild(slidersDiv);

  // slider factory for second row — same logic, different parent
  function makeSlider2(label, min, max, step, value, onChange) {
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
    slider.style.cssText = 'width:65px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
    wrapper.appendChild(slider);

    const valSpan = document.createElement('span');
    valSpan.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.7);font-size:10px;font-family:"JetBrains Mono",monospace;min-width:28px;';
    valSpan.textContent = Number.isInteger(step) || step >= 1 ? String(Math.round(value)) : Number(value).toFixed(2);
    wrapper.appendChild(valSpan);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valSpan.textContent = Number.isInteger(step) || step >= 1 ? String(Math.round(v)) : v.toFixed(2);
      onChange(v);
    });

    slidersDiv.appendChild(wrapper);
    return { slider, valSpan, wrapper };
  }

  // population slider — reset karna padega
  makeSlider2('Pop:', 50, 400, 10, populationCount, (v) => {
    populationCount = Math.round(v);
  });

  // transmission rate
  makeSlider2('Trans:', 0.01, 0.20, 0.01, transmissionProb, (v) => {
    transmissionProb = v;
  });

  // recovery time
  makeSlider2('Recovery:', 100, 500, 10, recoveryTime, (v) => {
    recoveryTime = Math.round(v);
  });

  // infection radius
  makeSlider2('Radius:', 10, 40, 1, infectionRadius, (v) => {
    infectionRadius = Math.round(v);
  });

  // vaccination %
  makeSlider2('Vacc%:', 0, 80, 5, vaccinationPct, (v) => {
    vaccinationPct = Math.round(v);
  });

  // mortality %
  makeSlider2('Mort%:', 0, 20, 1, mortalityPct, (v) => {
    mortalityPct = Math.round(v);
  });

  // --- Agent creation ---
  const BASE_SPEED = 1.2;
  const DISTANCING_FACTOR = 0.3; // 70% reduction

  function createAgent(x, y, state) {
    const angle = Math.random() * Math.PI * 2;
    const speed = (socialDistancing ? BASE_SPEED * DISTANCING_FACTOR : BASE_SPEED) * (0.5 + Math.random() * 0.5);
    return {
      x: x,
      y: y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      state: state, // S, I, R, V, D
      infectionTimer: 0,
      dirChangeTimer: Math.floor(30 + Math.random() * 30), // random walk direction change
      secondaryInfections: 0, // kitne logon ko isne infect kiya — R0 tracking
    };
  }

  function updateAgentSpeeds() {
    const speedMultiplier = socialDistancing ? DISTANCING_FACTOR : 1.0;
    for (const a of agents) {
      if (a.state === 'D') continue; // dead agents move nahi karte
      const currentSpeed = Math.sqrt(a.vx * a.vx + a.vy * a.vy);
      if (currentSpeed > 0) {
        const targetSpeed = BASE_SPEED * speedMultiplier * (0.5 + Math.random() * 0.5);
        a.vx = (a.vx / currentSpeed) * targetSpeed;
        a.vy = (a.vy / currentSpeed) * targetSpeed;
      }
    }
  }

  function resetSimulation() {
    agents = [];
    curveData = [];
    frameCount = 0;
    peakInfections = 0;
    peakTime = 0;
    totalInfected = 0;
    initialInfectors = 0;
    generationInfections = [];
    r0Estimate = 0;
    epidemicStarted = false;

    // hint wapas dikhao
    hintOverlay.style.opacity = '1';

    // agents banao
    const vaccCount = Math.floor(populationCount * vaccinationPct / 100);
    for (let i = 0; i < populationCount; i++) {
      const x = Math.random() * simW;
      const y = Math.random() * canvasH;
      const state = i < vaccCount ? 'V' : 'S';
      agents.push(createAgent(x, y, state));
    }
  }

  // --- Physics Step ---
  function physicsStep() {
    const speedMultiplier = socialDistancing ? DISTANCING_FACTOR : 1.0;
    const targetSpeed = BASE_SPEED * speedMultiplier;

    for (const a of agents) {
      // dead agents move nahi karte
      if (a.state === 'D') continue;

      // direction change timer — random walk behavior
      a.dirChangeTimer--;
      if (a.dirChangeTimer <= 0) {
        const angle = Math.random() * Math.PI * 2;
        const speed = targetSpeed * (0.5 + Math.random() * 0.5);
        a.vx = Math.cos(angle) * speed;
        a.vy = Math.sin(angle) * speed;
        a.dirChangeTimer = Math.floor(30 + Math.random() * 30);
      }

      // position update
      a.x += a.vx;
      a.y += a.vy;

      // bounce off walls — sim area ke andar raho
      if (a.x < 3) { a.x = 3; a.vx = Math.abs(a.vx); }
      if (a.x > simW - 3) { a.x = simW - 3; a.vx = -Math.abs(a.vx); }
      if (a.y < 3) { a.y = 3; a.vy = Math.abs(a.vy); }
      if (a.y > canvasH - 3) { a.y = canvasH - 3; a.vy = -Math.abs(a.vy); }
    }
  }

  // --- Disease Mechanics ---
  function diseaseStep() {
    if (!epidemicStarted) return;

    const radiusSq = infectionRadius * infectionRadius;

    for (const a of agents) {
      if (a.state !== 'I') continue;

      // infection timer badhao
      a.infectionTimer++;

      // recovery check — recovery time ke baad recover ya die
      if (a.infectionTimer >= recoveryTime) {
        // mortality check
        if (Math.random() * 100 < mortalityPct) {
          a.state = 'D';
          a.vx = 0;
          a.vy = 0;
        } else {
          a.state = 'R';
        }
        // R0 tracking — is agent ne kitne infect kiye, record karo
        generationInfections.push(a.secondaryInfections);
        // R0 estimate update — average secondary infections
        if (generationInfections.length > 0) {
          let sum = 0;
          for (const gi of generationInfections) sum += gi;
          r0Estimate = sum / generationInfections.length;
        }
        continue;
      }

      // try infecting nearby susceptible agents
      for (const b of agents) {
        if (b.state !== 'S') continue;

        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const distSq = dx * dx + dy * dy;

        if (distSq < radiusSq && distSq > 0) {
          // transmission probability check — per frame
          if (Math.random() < transmissionProb) {
            b.state = 'I';
            b.infectionTimer = 0;
            b.secondaryInfections = 0;
            a.secondaryInfections++;
            totalInfected++;
          }
        }
      }
    }

    // peak tracking
    let currentInfected = 0;
    for (const a of agents) {
      if (a.state === 'I') currentInfected++;
    }
    if (currentInfected > peakInfections) {
      peakInfections = currentInfected;
      peakTime = frameCount;
    }
  }

  // --- Epidemic Curve Data Collection ---
  function collectCurveData() {
    if (!epidemicStarted) return;
    if (frameCount % SAMPLE_INTERVAL !== 0) return;

    let s = 0, i = 0, r = 0, d = 0;
    for (const a of agents) {
      if (a.state === 'S' || a.state === 'V') s++; // vacc ko S mein count karo curve ke liye
      else if (a.state === 'I') i++;
      else if (a.state === 'R') r++;
      else if (a.state === 'D') d++;
    }

    curveData.push({ s, i, r, d, t: frameCount });

    // zyada data ho gaya toh purana hata do — max 600 points
    if (curveData.length > 600) {
      curveData = curveData.slice(curveData.length - 600);
    }
  }

  // --- Canvas Sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = canvasWrapper.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;
    simW = Math.floor(containerWidth * 0.65); // left 65% simulation
    chartW = containerWidth - simW; // right 35% chart

    canvas.width = Math.floor(containerWidth * dpr);
    canvas.height = Math.floor(CANVAS_HEIGHT * dpr);
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    // hint position update — left panel ke center mein
    hintOverlay.style.left = (simW / 2) + 'px';
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Drawing: Agent Simulation Area ---
  function drawAgents() {
    // background — sim area
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, simW, canvasH);

    // subtle grid lines — depth feel
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 0.5;
    const gridSize = 40;
    for (let gx = 0; gx < simW; gx += gridSize) {
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, canvasH);
      ctx.stroke();
    }
    for (let gy = 0; gy < canvasH; gy += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(simW, gy);
      ctx.stroke();
    }

    // pulsing animation ke liye time
    const pulsePhase = (frameCount % 40) / 40; // 0-1 cycle
    const pulseAlpha = 0.08 + 0.06 * Math.sin(pulsePhase * Math.PI * 2);

    // agents draw karo
    for (const a of agents) {
      const col = STATE_COLORS[a.state];

      // infected agents ke liye infection radius dikhao — faint red circle
      if (a.state === 'I') {
        ctx.beginPath();
        ctx.arc(a.x, a.y, infectionRadius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(' + col.r + ',' + col.g + ',' + col.b + ',' + (pulseAlpha * 0.5) + ')';
        ctx.fill();
        ctx.strokeStyle = 'rgba(' + col.r + ',' + col.g + ',' + col.b + ',' + (pulseAlpha * 1.2) + ')';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }

      // glow — infected agents ke liye pulsing glow
      if (a.state === 'I') {
        ctx.beginPath();
        ctx.arc(a.x, a.y, 6 + pulseAlpha * 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(' + col.r + ',' + col.g + ',' + col.b + ',' + (pulseAlpha * 0.8) + ')';
        ctx.fill();
      }

      // main agent dot
      const radius = a.state === 'D' ? 2.5 : 3;
      ctx.beginPath();
      ctx.arc(a.x, a.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + col.r + ',' + col.g + ',' + col.b + ',' + (a.state === 'D' ? 0.5 : 0.9) + ')';
      ctx.fill();
    }

    // divider line — sim area aur chart ke beech
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(simW, 0);
    ctx.lineTo(simW, canvasH);
    ctx.stroke();
  }

  // --- Drawing: Epidemic Curve Chart ---
  function drawChart() {
    // chart area background — slightly different shade
    ctx.fillStyle = 'rgba(12,12,24,0.8)';
    ctx.fillRect(simW, 0, chartW, canvasH);

    const chartLeft = simW + 35; // axis label space
    const chartRight = canvasW - 10;
    const chartTop = 30;
    const chartBottom = canvasH - 30;
    const chartPlotW = chartRight - chartLeft;
    const chartPlotH = chartBottom - chartTop;

    if (chartPlotW <= 0 || chartPlotH <= 0) return;

    // chart title
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.6)';
    ctx.textAlign = 'center';
    ctx.fillText('Epidemic Curve', simW + chartW / 2, 18);

    // axes draw karo
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 0.5;

    // Y axis
    ctx.beginPath();
    ctx.moveTo(chartLeft, chartTop);
    ctx.lineTo(chartLeft, chartBottom);
    ctx.stroke();

    // X axis
    ctx.beginPath();
    ctx.moveTo(chartLeft, chartBottom);
    ctx.lineTo(chartRight, chartBottom);
    ctx.stroke();

    // Y axis label
    ctx.save();
    ctx.translate(simW + 12, chartTop + chartPlotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.4)';
    ctx.textAlign = 'center';
    ctx.fillText('Count', 0, 0);
    ctx.restore();

    // X axis label
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.4)';
    ctx.textAlign = 'center';
    ctx.fillText('Time', simW + chartW / 2, canvasH - 5);

    if (curveData.length < 2) {
      // empty chart hint
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.25)';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for', simW + chartW / 2, canvasH / 2 - 8);
      ctx.fillText('epidemic...', simW + chartW / 2, canvasH / 2 + 8);
      return;
    }

    // Y axis max — total population
    const yMax = populationCount;

    // Y axis tick marks
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'right';
    const yTicks = 4;
    for (let t = 0; t <= yTicks; t++) {
      const val = Math.round(yMax * t / yTicks);
      const yPos = chartBottom - (t / yTicks) * chartPlotH;
      ctx.fillText(String(val), chartLeft - 4, yPos + 3);
      // grid line
      if (t > 0 && t < yTicks) {
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.beginPath();
        ctx.moveTo(chartLeft, yPos);
        ctx.lineTo(chartRight, yPos);
        ctx.stroke();
      }
    }

    // data plot karo — 3 lines: S (blue), I (red), R (green) + D (gray)
    const dataLen = curveData.length;
    const xStep = chartPlotW / Math.max(dataLen - 1, 1);

    // line drawing function
    function drawLine(key, color, lineWidth) {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      for (let i = 0; i < dataLen; i++) {
        const x = chartLeft + i * xStep;
        const val = curveData[i][key];
        const y = chartBottom - (val / yMax) * chartPlotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // filled area for infected curve — dramatic effect
    ctx.beginPath();
    ctx.moveTo(chartLeft, chartBottom);
    for (let i = 0; i < dataLen; i++) {
      const x = chartLeft + i * xStep;
      const val = curveData[i].i;
      const y = chartBottom - (val / yMax) * chartPlotH;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(chartLeft + (dataLen - 1) * xStep, chartBottom);
    ctx.closePath();
    ctx.fillStyle = 'rgba(239,68,68,0.08)';
    ctx.fill();

    // lines draw — order: S, D, R, I (I sabse upar)
    drawLine('s', 'rgba(74,158,255,0.6)', 1.2);
    drawLine('d', 'rgba(64,64,80,0.6)', 1);
    drawLine('r', 'rgba(16,185,129,0.6)', 1.2);
    drawLine('i', 'rgba(239,68,68,0.9)', 1.5);

    // legend — bottom right corner of chart
    const legendX = chartRight - 55;
    const legendY = chartTop + 8;
    const legendItems = [
      { label: 'S', color: STATE_COLORS.S.hex },
      { label: 'I', color: STATE_COLORS.I.hex },
      { label: 'R', color: STATE_COLORS.R.hex },
      { label: 'D', color: STATE_COLORS.D.hex },
    ];
    ctx.font = '9px "JetBrains Mono", monospace';
    legendItems.forEach((item, idx) => {
      const ly = legendY + idx * 14;
      // colored square
      ctx.fillStyle = item.color;
      ctx.fillRect(legendX, ly - 4, 8, 8);
      // label
      ctx.fillStyle = 'rgba(176,176,176,0.5)';
      ctx.textAlign = 'left';
      ctx.fillText(item.label, legendX + 12, ly + 3);
    });
  }

  // --- Combined Draw ---
  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);
    drawAgents();
    drawChart();
    updateStats();
  }

  // --- Click to Infect ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  function infectNearest(cx, cy) {
    // sirf sim area ke andar click count karo
    if (cx > simW) return;

    let closest = null;
    let closestDist = Infinity;

    for (const a of agents) {
      if (a.state !== 'S') continue;
      const dx = a.x - cx;
      const dy = a.y - cy;
      const dist = dx * dx + dy * dy;
      if (dist < closestDist) {
        closestDist = dist;
        closest = a;
      }
    }

    if (closest) {
      closest.state = 'I';
      closest.infectionTimer = 0;
      closest.secondaryInfections = 0;
      initialInfectors++;
      totalInfected++;

      if (!epidemicStarted) {
        epidemicStarted = true;
        // hint chhupa do
        hintOverlay.style.opacity = '0';
      }
    }
  }

  canvas.addEventListener('click', (e) => {
    const pos = getCanvasPos(e);
    infectNearest(pos.x, pos.y);
  });

  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    infectNearest(pos.x, pos.y);
  }, { passive: false });

  // --- Animation Loop ---
  function loop(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    frameCount++;
    physicsStep();
    diseaseStep();
    collectCurveData();
    draw();

    animationId = requestAnimationFrame(loop);
  }

  // --- IntersectionObserver — sirf visible hone pe animate karo ---
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) animationId = requestAnimationFrame(loop);
    else if (!isVisible && animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  // lab resume: restart loop when focus released
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });

  // tab visibility — background mein CPU waste mat karo
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      isVisible = false;
      if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) {
        isVisible = true;
        if (!animationId) animationId = requestAnimationFrame(loop);
      }
    }
  });

  // --- Init — sab shuru karo ---
  resetSimulation();
  draw(); // pehla frame render karo — blank na dikhe
}
