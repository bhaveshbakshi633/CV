// ============================================================
// Spring-Mass-Damper System — Second-order ODE ka interactive visualization
// m*x'' + c*x' + k*x = 0 — drag karke initial displacement do, chhod ke dekho physics
// Under/over/critically damped teenon regimes dikhate hain
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, spring chalao
export function initSpringMass() {
  const container = document.getElementById('springMassContainer');
  if (!container) {
    console.warn('springMassContainer nahi mila bhai, spring-mass skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const TOTAL_HEIGHT = 400;
  const ANIM_HEIGHT_RATIO = 0.6;  // upar 60% animation, neeche 40% plot
  const ANIM_HEIGHT = Math.round(TOTAL_HEIGHT * ANIM_HEIGHT_RATIO);
  const PLOT_HEIGHT = TOTAL_HEIGHT - ANIM_HEIGHT;
  const DT = 1 / 60;
  const PLOT_HISTORY = 400;        // kitne data points rakhne hain plot mein
  const PLOT_TIME_WINDOW = 6;      // seconds dikhane hain x-axis pe

  // spring drawing constants
  const WALL_WIDTH = 14;
  const WALL_X = 40;               // wall left edge
  const MASS_W = 50;
  const MASS_H = 40;
  const SPRING_COILS = 12;         // zig-zag mein kitne coils
  const DAMPER_WIDTH = 16;
  const DAMPER_PISTON_LEN = 20;
  const GROUND_Y_OFFSET = 10;      // ground line mass ke neeche

  // --- Physics state ---
  let mass = 1.5;                   // kg
  let springK = 15;                 // N/m
  let dampingC = 2;                 // Ns/m
  let x = 0;                        // displacement from equilibrium (pixels → meters scaled)
  let v = 0;                        // velocity
  let equilibriumX = 0;             // canvas mein equilibrium position
  let isSimulating = false;
  let simTime = 0;

  // plot data — time aur displacement store karenge
  let plotData = [];                // [{t, x}]

  // drag state — mass block ko pakad ke kheenchna
  let isDragging = false;
  let dragOffsetX = 0;

  // animation state
  let animationId = null;
  let isVisible = false;

  // --- Damping regime calculate karo ---
  function getDampingRatio() {
    // zeta = c / (2 * sqrt(m * k))
    return dampingC / (2 * Math.sqrt(mass * springK));
  }

  function getRegimeLabel() {
    const zeta = getDampingRatio();
    if (zeta < 0.001) return 'Undamped';
    if (Math.abs(zeta - 1) < 0.02) return 'Critically Damped';
    if (zeta < 1) return 'Underdamped';
    return 'Overdamped';
  }

  function getRegimeColor() {
    const zeta = getDampingRatio();
    if (zeta < 0.001) return '#60a5fa';       // blue — no damping
    if (Math.abs(zeta - 1) < 0.02) return '#4ade80'; // green — critical
    if (zeta < 1) return '#3b82f6';            // blue — underdamped
    return '#f97316';                           // orange — overdamped
  }

  // --- DOM structure banate hain ---
  // pehle game-header aur game-desc bachao, baaki hata do
  const existingHeader = container.querySelector('.game-header');
  const existingDesc = container.querySelector('.game-desc');
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  if (existingHeader) container.appendChild(existingHeader);
  if (existingDesc) container.appendChild(existingDesc);
  container.style.cssText += ';width:100%;position:relative;';

  // main canvas — spring animation + time plot dono ek hi canvas mein
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + TOTAL_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(245,158,11,0.15)',
    'border-radius:8px',
    'cursor:default',
    'background:transparent',
    'margin-top:8px',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Info bar — regime + zeta display ---
  const infoDiv = document.createElement('div');
  infoDiv.style.cssText = [
    'margin-top:8px',
    'padding:6px 12px',
    'background:rgba(245,158,11,0.05)',
    'border:1px solid rgba(245,158,11,0.12)',
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

  const regimeSpan = document.createElement('span');
  regimeSpan.style.cssText = 'font-weight:600;';
  infoDiv.appendChild(regimeSpan);

  const zetaSpan = document.createElement('span');
  infoDiv.appendChild(zetaSpan);

  const freqSpan = document.createElement('span');
  infoDiv.appendChild(freqSpan);

  function updateInfo() {
    const zeta = getDampingRatio();
    const wn = Math.sqrt(springK / mass);
    const regime = getRegimeLabel();
    const color = getRegimeColor();
    regimeSpan.textContent = regime;
    regimeSpan.style.color = color;
    zetaSpan.textContent = '\u03b6 = ' + zeta.toFixed(3);
    zetaSpan.style.color = color;
    freqSpan.textContent = '\u03c9n = ' + wn.toFixed(2) + ' rad/s';
  }

  // --- Controls: sliders + buttons ---
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
    labelEl.style.cssText = 'color:#b0b0b0;font-size:12px;font-weight:600;min-width:14px;font-family:"JetBrains Mono",monospace;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText = 'width:85px;height:4px;accent-color:rgba(245,158,11,0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText = 'color:#b0b0b0;font-size:11px;min-width:32px;font-family:"JetBrains Mono",monospace;';
    valueEl.textContent = parseFloat(defaultVal).toFixed(1);
    wrapper.appendChild(valueEl);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = val.toFixed(step < 0.1 ? 2 : 1);
      onChange(val);
      updateInfo();
    });

    slidersDiv.appendChild(wrapper);
    return { slider, valueEl };
  }

  // mass, spring constant, damping sliders
  const massSlider = createSlider('m', 0.5, 5, 0.1, mass, (v) => { mass = v; });
  const kSlider = createSlider('k', 1, 50, 0.5, springK, (v) => { springK = v; });
  const cSlider = createSlider('c', 0, 20, 0.1, dampingC, (v) => { dampingC = v; });

  // --- Button helper ---
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:4px 10px',
      'font-size:11px',
      'border-radius:5px',
      'cursor:pointer',
      'background:rgba(245,158,11,0.08)',
      'color:#b0b0b0',
      'border:1px solid rgba(245,158,11,0.2)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(245,158,11,0.2)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(245,158,11,0.08)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    buttonsDiv.appendChild(btn);
    return btn;
  }

  // --- Preset + Reset buttons ---
  function applyPreset(m, k, c) {
    mass = m; springK = k; dampingC = c;
    massSlider.slider.value = m;
    massSlider.valueEl.textContent = m.toFixed(1);
    kSlider.slider.value = k;
    kSlider.valueEl.textContent = k.toFixed(1);
    cSlider.slider.value = c;
    cSlider.valueEl.textContent = c.toFixed(1);
    resetSim();
    updateInfo();
  }

  // presets — underdamped, critical, overdamped, no damping
  createButton('Underdamped', () => applyPreset(1.5, 20, 2));
  createButton('Critical', () => {
    // zeta = 1 → c = 2*sqrt(m*k)
    const m = 1.5, k = 20;
    const c = 2 * Math.sqrt(m * k);
    applyPreset(m, k, parseFloat(c.toFixed(1)));
  });
  createButton('Overdamped', () => applyPreset(1.0, 10, 15));
  createButton('No Damping', () => applyPreset(1.5, 20, 0));
  createButton('Reset', () => resetSim());

  // --- Canvas sizing — DPR handle karna zaroori hai crisp rendering ke liye ---
  let canvasW = 0, canvasH = 0, dpr = 1;

  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = TOTAL_HEIGHT;

    canvas.width = containerWidth * dpr;
    canvas.height = TOTAL_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = TOTAL_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // equilibrium position — canvas ke beech mein thoda right
    equilibriumX = WALL_X + WALL_WIDTH + (canvasW - WALL_X - WALL_WIDTH) * 0.5;
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Sim reset ---
  function resetSim() {
    x = 0;
    v = 0;
    simTime = 0;
    plotData = [];
    isSimulating = false;
    isDragging = false;
    canvas.style.cursor = 'default';
  }

  // --- Physics: RK4 integration ---
  // ODE: x'' = -(k/m)*x - (c/m)*x'
  // state vector: [x, v]
  function derivatives(state) {
    const [pos, vel] = state;
    const accel = -(springK / mass) * pos - (dampingC / mass) * vel;
    return [vel, accel];
  }

  function rk4Step(state, dt) {
    const k1 = derivatives(state);
    const s2 = [state[0] + k1[0] * dt * 0.5, state[1] + k1[1] * dt * 0.5];
    const k2 = derivatives(s2);
    const s3 = [state[0] + k2[0] * dt * 0.5, state[1] + k2[1] * dt * 0.5];
    const k3 = derivatives(s3);
    const s4 = [state[0] + k3[0] * dt, state[1] + k3[1] * dt];
    const k4 = derivatives(s4);

    return [
      state[0] + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
      state[1] + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    ];
  }

  function physicsStep() {
    if (!isSimulating) return;
    const result = rk4Step([x, v], DT);
    x = result[0];
    v = result[1];
    simTime += DT;

    // plot data mein add kar
    plotData.push({ t: simTime, x: x });
    if (plotData.length > PLOT_HISTORY) {
      plotData.shift();
    }

    // agar motion negligible ho gaya toh ruk ja — energy waste nahi karni
    if (Math.abs(x) < 0.0005 && Math.abs(v) < 0.0005 && simTime > 0.5) {
      x = 0;
      v = 0;
      isSimulating = false;
    }
  }

  // --- Displacement ko pixels mein convert karo ---
  // x = 1.0 means ~100px displacement (adjustable feel ke liye)
  const PX_PER_UNIT = 100;

  function getMassScreenX() {
    return equilibriumX + x * PX_PER_UNIT;
  }

  // --- Mouse/touch interaction — mass block drag karo ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  function isInsideMass(px, py) {
    const massX = getMassScreenX();
    const centerY = ANIM_HEIGHT * 0.5;
    return (
      px >= massX - MASS_W / 2 - 10 &&
      px <= massX + MASS_W / 2 + 10 &&
      py >= centerY - MASS_H / 2 - 10 &&
      py <= centerY + MASS_H / 2 + 10
    );
  }

  function handlePointerDown(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    // sirf animation area mein click hua toh drag karo
    if (pos.y > ANIM_HEIGHT) return;
    if (isInsideMass(pos.x, pos.y)) {
      isDragging = true;
      isSimulating = false;
      dragOffsetX = pos.x - getMassScreenX();
      canvas.style.cursor = 'grabbing';
      // plot reset karo jab naya drag shuru ho
      plotData = [];
      simTime = 0;
    }
  }

  function handlePointerMove(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    if (isDragging) {
      // displacement calculate kar from equilibrium
      const newScreenX = pos.x - dragOffsetX;
      x = (newScreenX - equilibriumX) / PX_PER_UNIT;
      // clamp karo — zyada door mat jaane do
      const maxDisp = (canvasW - WALL_X - WALL_WIDTH - MASS_W) / (2 * PX_PER_UNIT);
      x = Math.max(-maxDisp * 0.9, Math.min(maxDisp * 0.9, x));
      v = 0;
    } else if (pos.y <= ANIM_HEIGHT) {
      // hover pe cursor change karo
      canvas.style.cursor = isInsideMass(pos.x, pos.y) ? 'grab' : 'default';
    }
  }

  function handlePointerUp() {
    if (isDragging) {
      isDragging = false;
      canvas.style.cursor = 'default';
      // chhoda toh simulate shuru karo — initial displacement set hai, velocity 0
      if (Math.abs(x) > 0.01) {
        isSimulating = true;
        simTime = 0;
        plotData = [{ t: 0, x: x }];
      }
    }
  }

  // mouse events
  canvas.addEventListener('mousedown', handlePointerDown);
  canvas.addEventListener('mousemove', handlePointerMove);
  canvas.addEventListener('mouseup', handlePointerUp);
  canvas.addEventListener('mouseleave', handlePointerUp);

  // touch events
  canvas.addEventListener('touchstart', handlePointerDown, { passive: false });
  canvas.addEventListener('touchmove', handlePointerMove, { passive: false });
  canvas.addEventListener('touchend', handlePointerUp);
  canvas.addEventListener('touchcancel', handlePointerUp);

  // ============================================================
  // DRAWING FUNCTIONS — yahan asli maza hai
  // ============================================================

  // --- Wall draw karo — left side pe hatched wall ---
  function drawWall(centerY) {
    const wallTop = centerY - ANIM_HEIGHT * 0.35;
    const wallBot = centerY + ANIM_HEIGHT * 0.35;

    // wall background
    ctx.fillStyle = 'rgba(245,158,11,0.12)';
    ctx.fillRect(WALL_X, wallTop, WALL_WIDTH, wallBot - wallTop);

    // wall border — right edge
    ctx.strokeStyle = 'rgba(245,158,11,0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(WALL_X + WALL_WIDTH, wallTop);
    ctx.lineTo(WALL_X + WALL_WIDTH, wallBot);
    ctx.stroke();

    // hatch lines — diagonal lines inside wall
    ctx.strokeStyle = 'rgba(245,158,11,0.2)';
    ctx.lineWidth = 1;
    const spacing = 8;
    for (let i = 0; i < (wallBot - wallTop + WALL_WIDTH) / spacing; i++) {
      const startY = wallTop + i * spacing;
      ctx.beginPath();
      ctx.moveTo(WALL_X, startY);
      ctx.lineTo(WALL_X + WALL_WIDTH, startY - WALL_WIDTH);
      ctx.stroke();
    }
  }

  // --- Zig-zag spring draw karo ---
  function drawSpring(startX, endX, centerY, amplitude) {
    const springStartX = startX;
    const springEndX = endX;
    const len = springEndX - springStartX;

    if (len < 10) {
      // spring bahut compressed hai — seedhi line draw kar do
      ctx.beginPath();
      ctx.moveTo(springStartX, centerY);
      ctx.lineTo(springEndX, centerY);
      ctx.strokeStyle = 'rgba(245,158,11,0.6)';
      ctx.lineWidth = 2;
      ctx.stroke();
      return;
    }

    // lead-in aur lead-out straight segments
    const leadLen = Math.min(12, len * 0.1);
    const coilStartX = springStartX + leadLen;
    const coilEndX = springEndX - leadLen;
    const coilLen = coilEndX - coilStartX;

    ctx.beginPath();
    ctx.moveTo(springStartX, centerY);
    ctx.lineTo(coilStartX, centerY);

    // zig-zag coils — amplitude spring compression/extension se adjust hota hai
    // jab compressed ho tab amplitude zyada, stretched ho tab kam
    const restLen = (equilibriumX - MASS_W / 2) - (WALL_X + WALL_WIDTH);
    const compressionRatio = Math.max(0.3, Math.min(2.0, len / Math.max(restLen, 1)));
    const coilAmp = amplitude / compressionRatio;

    const segments = SPRING_COILS * 2;
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const px = coilStartX + t * coilLen;
      // alternate up/down — sine wave se smooth corners
      const py = centerY + Math.sin(t * Math.PI * SPRING_COILS) * coilAmp;
      ctx.lineTo(px, py);
    }

    ctx.lineTo(springEndX, centerY);

    ctx.strokeStyle = 'rgba(245,158,11,0.6)';
    ctx.lineWidth = 1.8;
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  // --- Damper symbol draw karo — piston-like, spring ke parallel ---
  function drawDamper(startX, endX, centerY, offsetY) {
    const damperY = centerY + offsetY;
    const len = endX - startX;

    if (len < 20) return; // bahut chhota hai toh skip

    const leadLen = Math.min(15, len * 0.12);
    const bodyStartX = startX + leadLen;
    const bodyEndX = endX - leadLen;
    const bodyLen = bodyEndX - bodyStartX;
    const halfW = DAMPER_WIDTH / 2;

    ctx.strokeStyle = 'rgba(245,158,11,0.4)';
    ctx.lineWidth = 1.5;

    // left rod — wall se cylinder tak
    ctx.beginPath();
    ctx.moveTo(startX, damperY);
    ctx.lineTo(bodyStartX, damperY);
    ctx.stroke();

    // cylinder body — rectangle
    const cylStartX = bodyStartX;
    const cylLen = bodyLen * 0.55;
    ctx.strokeStyle = 'rgba(245,158,11,0.35)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(cylStartX, damperY - halfW, cylLen, DAMPER_WIDTH);

    // piston rod — cylinder ke andar se bahar mass tak
    const pistonStartX = cylStartX + cylLen;
    ctx.strokeStyle = 'rgba(245,158,11,0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pistonStartX, damperY);
    ctx.lineTo(endX, damperY);
    ctx.stroke();

    // piston head — cylinder ke andar vertical line
    const pistonHeadX = Math.min(pistonStartX, cylStartX + cylLen - 2);
    ctx.strokeStyle = 'rgba(245,158,11,0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pistonHeadX, damperY - halfW + 2);
    ctx.lineTo(pistonHeadX, damperY + halfW - 2);
    ctx.stroke();

    // "c" label
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(245,158,11,0.35)';
    ctx.textAlign = 'center';
    ctx.fillText('c', cylStartX + cylLen * 0.5, damperY + halfW + 12);
  }

  // --- Mass block draw karo ---
  function drawMass(massScreenX, centerY) {
    const left = massScreenX - MASS_W / 2;
    const top = centerY - MASS_H / 2;

    // mass block shadow — subtle depth effect
    ctx.fillStyle = 'rgba(245,158,11,0.06)';
    ctx.fillRect(left + 3, top + 3, MASS_W, MASS_H);

    // mass block body
    ctx.fillStyle = 'rgba(245,158,11,0.15)';
    ctx.fillRect(left, top, MASS_W, MASS_H);

    // mass block border
    ctx.strokeStyle = 'rgba(245,158,11,0.55)';
    ctx.lineWidth = 2;
    ctx.strokeRect(left, top, MASS_W, MASS_H);

    // dragging glow effect
    if (isDragging) {
      ctx.shadowColor = 'rgba(245,158,11,0.4)';
      ctx.shadowBlur = 12;
      ctx.strokeStyle = 'rgba(245,158,11,0.8)';
      ctx.strokeRect(left, top, MASS_W, MASS_H);
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
    }

    // "m" label
    ctx.font = '14px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(245,158,11,0.7)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('m', massScreenX, centerY);
    ctx.textBaseline = 'alphabetic';
  }

  // --- Ground line draw karo ---
  function drawGround(centerY) {
    const groundY = centerY + MASS_H / 2 + GROUND_Y_OFFSET;

    // ground line
    ctx.strokeStyle = 'rgba(245,158,11,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(WALL_X, groundY);
    ctx.lineTo(canvasW - 20, groundY);
    ctx.stroke();

    // ground hatching — chhoti diagonal lines
    const hatchSpacing = 10;
    const hatchLen = 6;
    ctx.strokeStyle = 'rgba(245,158,11,0.1)';
    for (let hx = WALL_X; hx < canvasW - 20; hx += hatchSpacing) {
      ctx.beginPath();
      ctx.moveTo(hx, groundY);
      ctx.lineTo(hx - hatchLen, groundY + hatchLen);
      ctx.stroke();
    }
  }

  // --- Equilibrium dashed line ---
  function drawEquilibrium(centerY) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(245,158,11,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(equilibriumX, centerY - MASS_H / 2 - 15);
    ctx.lineTo(equilibriumX, centerY + MASS_H / 2 + GROUND_Y_OFFSET - 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // label
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(245,158,11,0.25)';
    ctx.textAlign = 'center';
    ctx.fillText('x=0', equilibriumX, centerY - MASS_H / 2 - 20);
  }

  // --- Displacement arrow ---
  function drawDisplacementArrow(massScreenX, centerY) {
    if (Math.abs(x) < 0.02) return;

    const arrowY = centerY - MASS_H / 2 - 12;
    const startAX = equilibriumX;
    const endAX = massScreenX;

    // arrow line
    ctx.strokeStyle = getRegimeColor() + 'aa';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(startAX, arrowY);
    ctx.lineTo(endAX, arrowY);
    ctx.stroke();

    // arrowhead
    const dir = endAX > startAX ? 1 : -1;
    const headLen = 6;
    ctx.beginPath();
    ctx.moveTo(endAX, arrowY);
    ctx.lineTo(endAX - dir * headLen, arrowY - 4);
    ctx.lineTo(endAX - dir * headLen, arrowY + 4);
    ctx.closePath();
    ctx.fillStyle = getRegimeColor() + 'aa';
    ctx.fill();

    // displacement value
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = getRegimeColor();
    ctx.textAlign = 'center';
    ctx.fillText('x=' + x.toFixed(2), (startAX + endAX) / 2, arrowY - 6);
  }

  // --- Velocity indicator ---
  function drawVelocityIndicator(massScreenX, centerY) {
    if (Math.abs(v) < 0.05 || !isSimulating) return;

    const velArrowY = centerY + MASS_H / 2 + 8;
    const velScale = 15; // pixels per unit velocity
    const velPx = v * velScale;
    const clampedVel = Math.max(-80, Math.min(80, velPx));

    ctx.strokeStyle = 'rgba(96,165,250,0.5)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(massScreenX, velArrowY);
    ctx.lineTo(massScreenX + clampedVel, velArrowY);
    ctx.stroke();

    // arrowhead
    if (Math.abs(clampedVel) > 5) {
      const dir = clampedVel > 0 ? 1 : -1;
      ctx.beginPath();
      ctx.moveTo(massScreenX + clampedVel, velArrowY);
      ctx.lineTo(massScreenX + clampedVel - dir * 5, velArrowY - 3);
      ctx.lineTo(massScreenX + clampedVel - dir * 5, velArrowY + 3);
      ctx.closePath();
      ctx.fillStyle = 'rgba(96,165,250,0.5)';
      ctx.fill();
    }
  }

  // --- Top animation area draw karo ---
  function drawAnimation() {
    const centerY = ANIM_HEIGHT * 0.5;
    const massScreenX = getMassScreenX();
    const wallRightX = WALL_X + WALL_WIDTH;
    const massLeftX = massScreenX - MASS_W / 2;

    // separator line between animation and plot
    ctx.strokeStyle = 'rgba(245,158,11,0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, ANIM_HEIGHT);
    ctx.lineTo(canvasW, ANIM_HEIGHT);
    ctx.stroke();

    // sab elements draw karo
    drawGround(centerY);
    drawEquilibrium(centerY);
    drawWall(centerY);

    // spring — upar wala connection
    const springY = centerY - 10;
    drawSpring(wallRightX, massLeftX, springY, 14);

    // damper — neeche wala connection (spring ke parallel)
    const damperY = 14;
    drawDamper(wallRightX, massLeftX, centerY, damperY);

    // connecting verticals — wall side
    ctx.strokeStyle = 'rgba(245,158,11,0.3)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(wallRightX, springY);
    ctx.lineTo(wallRightX, centerY + damperY);
    ctx.stroke();

    // connecting verticals — mass side
    ctx.beginPath();
    ctx.moveTo(massLeftX, springY);
    ctx.lineTo(massLeftX, centerY + damperY);
    ctx.stroke();

    // mass block
    drawMass(massScreenX, centerY);

    // displacement arrow
    drawDisplacementArrow(massScreenX, centerY);

    // velocity indicator
    drawVelocityIndicator(massScreenX, centerY);

    // hint text — jab kuch nahi ho raha tab dikhao
    if (!isSimulating && Math.abs(x) < 0.01 && plotData.length === 0) {
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('drag the mass block to set initial displacement', canvasW / 2, ANIM_HEIGHT - 16);
    }
  }

  // --- Bottom plot area — x(t) vs time ---
  function drawPlot() {
    const plotTop = ANIM_HEIGHT + 1;
    const plotBottom = canvasH;
    const pH = plotBottom - plotTop;
    const padding = { top: 14, bottom: 16, left: 50, right: 20 };
    const plotAreaW = canvasW - padding.left - padding.right;
    const plotAreaH = pH - padding.top - padding.bottom;
    const plotAreaTop = plotTop + padding.top;
    const plotAreaLeft = padding.left;

    const regimeColor = getRegimeColor();

    // grid lines — horizontal
    ctx.strokeStyle = 'rgba(245,158,11,0.06)';
    ctx.lineWidth = 1;
    const hGridLines = 4;
    for (let i = 0; i <= hGridLines; i++) {
      const gy = plotAreaTop + (plotAreaH / hGridLines) * i;
      ctx.beginPath();
      ctx.moveTo(plotAreaLeft, gy);
      ctx.lineTo(plotAreaLeft + plotAreaW, gy);
      ctx.stroke();
    }

    // zero line — center pe thodi bright
    const zeroY = plotAreaTop + plotAreaH / 2;
    ctx.strokeStyle = 'rgba(245,158,11,0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotAreaLeft, zeroY);
    ctx.lineTo(plotAreaLeft + plotAreaW, zeroY);
    ctx.stroke();

    // vertical grid lines
    const vGridLines = 6;
    for (let i = 0; i <= vGridLines; i++) {
      const gx = plotAreaLeft + (plotAreaW / vGridLines) * i;
      ctx.strokeStyle = 'rgba(245,158,11,0.04)';
      ctx.beginPath();
      ctx.moveTo(gx, plotAreaTop);
      ctx.lineTo(gx, plotAreaTop + plotAreaH);
      ctx.stroke();
    }

    // Y-axis labels
    if (plotData.length > 0) {
      // auto-scale Y axis
      let maxX = 0;
      for (let i = 0; i < plotData.length; i++) {
        const absX = Math.abs(plotData[i].x);
        if (absX > maxX) maxX = absX;
      }
      maxX = Math.max(maxX, 0.2); // minimum range
      const yScale = plotAreaH / (2 * maxX * 1.1);

      // Y labels
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.35)';
      ctx.textAlign = 'right';
      const yLabelVal = maxX * 1.1;
      ctx.fillText('+' + yLabelVal.toFixed(1), plotAreaLeft - 4, plotAreaTop + 4);
      ctx.fillText('0', plotAreaLeft - 4, zeroY + 3);
      ctx.fillText('-' + yLabelVal.toFixed(1), plotAreaLeft - 4, plotAreaTop + plotAreaH + 2);

      // time axis — auto-scrolling
      let tStart, tEnd;
      if (plotData.length > 0) {
        tEnd = plotData[plotData.length - 1].t;
        tStart = Math.max(0, tEnd - PLOT_TIME_WINDOW);
      } else {
        tStart = 0;
        tEnd = PLOT_TIME_WINDOW;
      }

      // time labels
      ctx.textAlign = 'center';
      for (let i = 0; i <= vGridLines; i++) {
        const tVal = tStart + (tEnd - tStart) * (i / vGridLines);
        const gx = plotAreaLeft + (plotAreaW / vGridLines) * i;
        ctx.fillText(tVal.toFixed(1) + 's', gx, plotAreaTop + plotAreaH + 12);
      }

      // data line draw karo
      if (plotData.length > 1) {
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < plotData.length; i++) {
          const d = plotData[i];
          if (d.t < tStart) continue;
          const px = plotAreaLeft + ((d.t - tStart) / (tEnd - tStart)) * plotAreaW;
          const py = zeroY - d.x * yScale;
          const clampedPy = Math.max(plotAreaTop, Math.min(plotAreaTop + plotAreaH, py));
          if (!started) {
            ctx.moveTo(px, clampedPy);
            started = true;
          } else {
            ctx.lineTo(px, clampedPy);
          }
        }
        ctx.strokeStyle = regimeColor;
        ctx.lineWidth = 2;
        ctx.stroke();

        // area fill — halka shade line ke neeche
        if (started) {
          // line ke end se zero line tak, fir wapas start
          const lastD = plotData[plotData.length - 1];
          const lastPx = plotAreaLeft + ((lastD.t - tStart) / (tEnd - tStart)) * plotAreaW;
          ctx.lineTo(lastPx, zeroY);

          // pehla visible point dhundho
          let firstVisIdx = 0;
          for (let i = 0; i < plotData.length; i++) {
            if (plotData[i].t >= tStart) { firstVisIdx = i; break; }
          }
          const firstPx = plotAreaLeft + ((plotData[firstVisIdx].t - tStart) / (tEnd - tStart)) * plotAreaW;
          ctx.lineTo(firstPx, zeroY);
          ctx.closePath();
          ctx.fillStyle = regimeColor.replace(')', ',0.06)').replace('rgb(', 'rgba(');
          // simple alpha add — hex colors ke liye
          ctx.globalAlpha = 0.08;
          ctx.fill();
          ctx.globalAlpha = 1.0;
        }
      }

      // current position dot — plot pe
      if (plotData.length > 0) {
        const lastD = plotData[plotData.length - 1];
        if (lastD.t >= tStart) {
          const dotPx = plotAreaLeft + ((lastD.t - tStart) / (tEnd - tStart)) * plotAreaW;
          const dotPy = zeroY - lastD.x * yScale;
          const clampedDotPy = Math.max(plotAreaTop, Math.min(plotAreaTop + plotAreaH, dotPy));

          ctx.beginPath();
          ctx.arc(dotPx, clampedDotPy, 3, 0, Math.PI * 2);
          ctx.fillStyle = regimeColor;
          ctx.fill();
        }
      }
    } else {
      // empty state — labels dikhao
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.25)';
      ctx.textAlign = 'center';
      ctx.fillText('x(t) vs time', plotAreaLeft + plotAreaW / 2, zeroY + 3);
    }

    // axis label
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'left';
    ctx.fillText('x(t)', plotAreaLeft + 2, plotAreaTop - 3);
  }

  // --- Envelope lines for underdamped case ---
  function drawEnvelope() {
    if (plotData.length < 2) return;
    const zeta = getDampingRatio();
    if (zeta >= 1 || zeta < 0.001) return; // sirf underdamped ke liye

    const plotTop = ANIM_HEIGHT + 1;
    const pH = canvasH - plotTop;
    const padding = { top: 14, bottom: 16, left: 50, right: 20 };
    const plotAreaW = canvasW - padding.left - padding.right;
    const plotAreaH = pH - padding.top - padding.bottom;
    const plotAreaTop = plotTop + padding.top;
    const plotAreaLeft = padding.left;
    const zeroY = plotAreaTop + plotAreaH / 2;

    let maxX = 0;
    for (let i = 0; i < plotData.length; i++) {
      const absX = Math.abs(plotData[i].x);
      if (absX > maxX) maxX = absX;
    }
    maxX = Math.max(maxX, 0.2);
    const yScale = plotAreaH / (2 * maxX * 1.1);

    // envelope: A0 * e^(-zeta * wn * t)
    const wn = Math.sqrt(springK / mass);
    const A0 = plotData[0].x;
    if (Math.abs(A0) < 0.05) return;

    const tEnd = plotData[plotData.length - 1].t;
    const tStart = Math.max(0, tEnd - PLOT_TIME_WINDOW);
    const tPlotStart = plotData[0].t;

    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;

    // upper envelope
    ctx.strokeStyle = 'rgba(245,158,11,0.2)';
    ctx.beginPath();
    let started = false;
    const steps = 60;
    for (let i = 0; i <= steps; i++) {
      const t = tStart + (tEnd - tStart) * (i / steps);
      if (t < tPlotStart) continue;
      const dt = t - tPlotStart;
      const env = Math.abs(A0) * Math.exp(-zeta * wn * dt);
      const px = plotAreaLeft + ((t - tStart) / (tEnd - tStart)) * plotAreaW;
      const py = zeroY - env * yScale;
      if (!started) { ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // lower envelope
    ctx.beginPath();
    started = false;
    for (let i = 0; i <= steps; i++) {
      const t = tStart + (tEnd - tStart) * (i / steps);
      if (t < tPlotStart) continue;
      const dt = t - tPlotStart;
      const env = Math.abs(A0) * Math.exp(-zeta * wn * dt);
      const px = plotAreaLeft + ((t - tStart) / (tEnd - tStart)) * plotAreaW;
      const py = zeroY + env * yScale;
      if (!started) { ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.setLineDash([]);
  }

  // --- Main render function ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);
    drawAnimation();
    drawPlot();
    drawEnvelope();
  }

  // --- Animation loop ---
  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    physicsStep();
    draw();
    updateInfo();

    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — visible hone pe hi animate karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    resizeCanvas();
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

  // tab switch pe bhi pause/resume karo
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // initial info update
  updateInfo();
  // initial draw — taaki blank na dikhe
  resizeCanvas();
  draw();
}
