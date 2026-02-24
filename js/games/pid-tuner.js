// ============================================================
// PID Tuner Demo — Interactive ball-follows-cursor with PID control
// Robotics portfolio ke liye banaya hai ye
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, PID chalao
export function initPIDTuner() {
  const container = document.getElementById('pidTunerContainer');
  if (!container) {
    console.warn('pidTunerContainer nahi mila bhai, PID tuner skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const BALL_RADIUS = 12;
  const TRAIL_LENGTH = 30;
  const ERROR_HISTORY_LENGTH = 200;
  const MAIN_CANVAS_HEIGHT = 280;
  const PLOT_CANVAS_HEIGHT = 80;
  const VELOCITY_DAMPING = 0.95;
  const ANTI_WINDUP_LIMIT = 500; // integral ko control mein rakhne ke liye
  const DT = 1 / 60; // ~60fps assume kar rahe hain

  // --- State variables ---
  let ballX = 0, ballY = 0;
  let velX = 0, velY = 0;
  let integralX = 0, integralY = 0;
  let prevErrorX = 0, prevErrorY = 0;
  let targetX = 0, targetY = 0;
  let mouseInCanvas = false;

  // trail store karne ke liye — position + error magnitude
  let trail = [];
  // error history for plot — last 200 frames
  let errorHistory = [];

  // PID gains — default values, sliders se change honge
  let Kp = 8.0;
  let Ki = 0.3;
  let Kd = 2.5;

  // animation state
  let animationId = null;
  let isVisible = false;

  // --- DOM structure banate hain ---
  // pehle container saaf karo
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — ball yahan move karega
  const mainCanvas = document.createElement('canvas');
  mainCanvas.style.cssText = [
    'width:100%',
    'height:' + MAIN_CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(249,158,11,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
  ].join(';');
  container.appendChild(mainCanvas);

  // error plot canvas — neeche chhota graph
  const plotCanvas = document.createElement('canvas');
  plotCanvas.style.cssText = [
    'width:100%',
    'height:' + PLOT_CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(249,158,11,0.15)',
    'border-radius:8px',
    'margin-top:8px',
    'background:transparent',
  ].join(';');
  container.appendChild(plotCanvas);

  // controls section — sliders + presets
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:12px',
    'align-items:center',
    'justify-content:space-between',
  ].join(';');
  container.appendChild(controlsDiv);

  // sliders ka container
  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:16px;flex:1;min-width:280px;';
  controlsDiv.appendChild(slidersDiv);

  // presets ka container
  const presetsDiv = document.createElement('div');
  presetsDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;';
  controlsDiv.appendChild(presetsDiv);

  // --- Slider banane ka helper function ---
  // har slider ke saath label aur current value dikhega
  function createSlider(label, min, max, step, defaultVal, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = 'color:#b0b0b0;font-size:13px;font-weight:600;min-width:22px;font-family:monospace;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText = 'width:90px;height:4px;accent-color:rgba(249,158,11,0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText = 'color:#b0b0b0;font-size:12px;min-width:30px;font-family:monospace;';
    valueEl.textContent = parseFloat(defaultVal).toFixed(2);
    wrapper.appendChild(valueEl);

    // slider change hone pe value update kar aur callback fire kar
    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = val.toFixed(2);
      onChange(val);
    });

    slidersDiv.appendChild(wrapper);
    // slider ref return kar rahe hain taaki presets se set kar sakein
    return { slider, valueEl };
  }

  // teen sliders — Kp, Ki, Kd (ranges badha diye hain)
  const kpSlider = createSlider('Kp', 0, 15, 0.1, Kp, (v) => { Kp = v; });
  const kiSlider = createSlider('Ki', 0, 5, 0.01, Ki, (v) => { Ki = v; });
  const kdSlider = createSlider('Kd', 0, 8, 0.1, Kd, (v) => { Kd = v; });

  // --- Preset buttons ---
  // har preset mein Kp, Ki, Kd ki values set hain
  const presets = [
    { name: 'Well-tuned', kp: 8.0, ki: 0.3, kd: 2.5 },
    { name: 'Underdamped', kp: 14.0, ki: 0.5, kd: 0.3 },
    { name: 'Overdamped', kp: 2.0, ki: 0.05, kd: 7.0 },
    { name: 'Unstable', kp: 15.0, ki: 5.0, kd: 0.0 },
  ];

  // preset button click pe sliders update karo aur state reset karo
  function applyPreset(preset) {
    Kp = preset.kp;
    Ki = preset.ki;
    Kd = preset.kd;

    kpSlider.slider.value = preset.kp;
    kpSlider.valueEl.textContent = preset.kp.toFixed(2);
    kiSlider.slider.value = preset.ki;
    kiSlider.valueEl.textContent = preset.ki.toFixed(2);
    kdSlider.slider.value = preset.kd;
    kdSlider.valueEl.textContent = preset.kd.toFixed(2);

    // integral reset karo — nahi toh purana integral naye preset mein gadbad karega
    integralX = 0;
    integralY = 0;
    prevErrorX = 0;
    prevErrorY = 0;
  }

  presets.forEach((preset) => {
    const btn = document.createElement('button');
    btn.textContent = preset.name;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(249,158,11,0.08)',
      'color:#b0b0b0',
      'border:1px solid rgba(249,158,11,0.2)',
      'font-family:monospace',
      'transition:all 0.2s ease',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(249,158,11,0.2)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(249,158,11,0.08)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', () => applyPreset(preset));
    presetsDiv.appendChild(btn);
  });

  // --- Canvas sizing — retina display ke liye DPR handle karna zaroori hai ---
  function resizeCanvases() {
    const dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;

    // main canvas
    mainCanvas.width = containerWidth * dpr;
    mainCanvas.height = MAIN_CANVAS_HEIGHT * dpr;
    mainCanvas.style.width = containerWidth + 'px';
    mainCanvas.style.height = MAIN_CANVAS_HEIGHT + 'px';

    // plot canvas
    plotCanvas.width = containerWidth * dpr;
    plotCanvas.height = PLOT_CANVAS_HEIGHT * dpr;
    plotCanvas.style.width = containerWidth + 'px';
    plotCanvas.style.height = PLOT_CANVAS_HEIGHT + 'px';

    // contexts ko scale karo DPR se
    const mainCtx = mainCanvas.getContext('2d');
    mainCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const plotCtx = plotCanvas.getContext('2d');
    plotCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);

  // --- Ball ko canvas ke center mein initialize kar ---
  function initBallPosition() {
    const w = mainCanvas.clientWidth;
    const h = mainCanvas.clientHeight;
    ballX = w / 2;
    ballY = h / 2;
    targetX = w / 2;
    targetY = h / 2;
  }
  initBallPosition();

  // --- Mouse/touch events ---
  // hover se sirf preview crosshair dikhao, CLICK se target set karo
  let previewX = -100, previewY = -100;

  function getCanvasPos(e) {
    const rect = mainCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  // mousemove — sirf preview position update, target NAHI
  mainCanvas.addEventListener('mousemove', (e) => {
    const pos = getCanvasPos(e);
    previewX = pos.x;
    previewY = pos.y;
    mouseInCanvas = true;
  });

  mainCanvas.addEventListener('mouseleave', () => {
    mouseInCanvas = false;
    previewX = -100;
    previewY = -100;
  });

  // CLICK — yahan pe target set hota hai
  mainCanvas.addEventListener('click', (e) => {
    const pos = getCanvasPos(e);
    targetX = pos.x;
    targetY = pos.y;
    // integral reset karo — naye target pe purana integral gadbad karega
    integralX = 0;
    integralY = 0;
    prevErrorX = 0;
    prevErrorY = 0;
    mouseInCanvas = true;
  });

  // touch support — touchstart se target set, touchmove se drag
  mainCanvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    targetX = pos.x;
    targetY = pos.y;
    integralX = 0;
    integralY = 0;
    prevErrorX = 0;
    prevErrorY = 0;
    mouseInCanvas = true;
  }, { passive: false });

  mainCanvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    targetX = pos.x;
    targetY = pos.y;
    mouseInCanvas = true;
  }, { passive: false });

  mainCanvas.addEventListener('touchend', () => {
    mouseInCanvas = false;
  });

  // --- PID controller — asli dimaag yahan hai ---
  // x aur y independently calculate karte hain
  function pidStep() {
    // error calculate kar — target minus current
    const errorX = targetX - ballX;
    const errorY = targetY - ballY;

    // integral accumulate kar — anti-windup clamp lagao nahi toh integral wind-up se
    // ball pagal ho jaayegi
    integralX += errorX * DT;
    integralY += errorY * DT;
    integralX = Math.max(-ANTI_WINDUP_LIMIT, Math.min(ANTI_WINDUP_LIMIT, integralX));
    integralY = Math.max(-ANTI_WINDUP_LIMIT, Math.min(ANTI_WINDUP_LIMIT, integralY));

    // derivative nikal — current error minus previous error
    const derivX = (errorX - prevErrorX) / DT;
    const derivY = (errorY - prevErrorY) / DT;

    // PID output = P + I + D — ye acceleration ban jaayega
    const accelX = Kp * errorX + Ki * integralX + Kd * derivX;
    const accelY = Kp * errorY + Ki * integralY + Kd * derivY;

    // velocity update kar — acceleration add kar aur damping lagao
    // damping ke bina ball kabhi rukegi nahi
    velX = (velX + accelX * DT) * VELOCITY_DAMPING;
    velY = (velY + accelY * DT) * VELOCITY_DAMPING;

    // position update kar — simple euler integration
    ballX += velX * DT;
    ballY += velY * DT;

    // ball ko canvas ke andar rakh — boundary se bahar nahi jaani chahiye
    const w = mainCanvas.clientWidth;
    const h = mainCanvas.clientHeight;
    if (ballX < BALL_RADIUS) { ballX = BALL_RADIUS; velX *= -0.3; }
    if (ballX > w - BALL_RADIUS) { ballX = w - BALL_RADIUS; velX *= -0.3; }
    if (ballY < BALL_RADIUS) { ballY = BALL_RADIUS; velY *= -0.3; }
    if (ballY > h - BALL_RADIUS) { ballY = h - BALL_RADIUS; velY *= -0.3; }

    // previous error save kar — next frame mein derivative ke liye chahiye
    prevErrorX = errorX;
    prevErrorY = errorY;

    // error magnitude return kar — trail color aur display ke liye
    return Math.sqrt(errorX * errorX + errorY * errorY);
  }

  // --- Trail color decide karo error ke basis pe ---
  // green = close, yellow = medium, red = door
  function getTrailColor(errorMag, alpha) {
    if (errorMag < 30) {
      // green — sahi track kar raha hai
      return 'rgba(74,222,128,' + alpha + ')';
    } else if (errorMag < 150) {
      // yellow zone — thoda peeche hai, linearly interpolate karo
      const t = (errorMag - 30) / 120;
      const r = Math.round(74 + (250 - 74) * t);
      const g = Math.round(222 + (204 - 222) * t);
      const b = Math.round(128 + (0 - 128) * t);
      return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    } else {
      // red — bahut door hai, PID struggle kar raha hai
      return 'rgba(239,68,68,' + alpha + ')';
    }
  }

  // --- Main canvas draw karo ---
  function drawMainCanvas(errorMag) {
    const ctx = mainCanvas.getContext('2d');
    const w = mainCanvas.clientWidth;
    const h = mainCanvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    // trail draw kar — purani positions fading circles ke roop mein
    for (let i = 0; i < trail.length; i++) {
      const t = trail[i];
      // puraane points zyada transparent honge
      const alpha = ((i + 1) / trail.length) * 0.5;
      const radius = BALL_RADIUS * ((i + 1) / trail.length) * 0.7;
      ctx.beginPath();
      ctx.arc(t.x, t.y, Math.max(2, radius), 0, Math.PI * 2);
      ctx.fillStyle = getTrailColor(t.error, alpha);
      ctx.fill();
    }

    // error vector line — ball se target tak patli dashed line
    ctx.beginPath();
    ctx.moveTo(ballX, ballY);
    ctx.lineTo(targetX, targetY);
    ctx.strokeStyle = 'rgba(249,158,11,0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // target marker — solid crosshair jahan click kiya tha
    const crossSize = 8;
    ctx.beginPath();
    ctx.moveTo(targetX - crossSize, targetY);
    ctx.lineTo(targetX + crossSize, targetY);
    ctx.moveTo(targetX, targetY - crossSize);
    ctx.lineTo(targetX, targetY + crossSize);
    ctx.strokeStyle = 'rgba(249,158,11,0.7)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // target ke around chhota circle
    ctx.beginPath();
    ctx.arc(targetX, targetY, 4, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(249,158,11,0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // hover preview — faint crosshair cursor ke neeche (sirf jab hover ho)
    if (mouseInCanvas && previewX > 0) {
      ctx.beginPath();
      ctx.moveTo(previewX - 5, previewY);
      ctx.lineTo(previewX + 5, previewY);
      ctx.moveTo(previewX, previewY - 5);
      ctx.lineTo(previewX, previewY + 5);
      ctx.strokeStyle = 'rgba(249,158,11,0.2)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // ball draw kar — glow effect ke saath
    // pehle glow (shadow) set kar
    ctx.shadowColor = 'rgba(96,165,250,0.6)';
    ctx.shadowBlur = 16;
    ctx.beginPath();
    ctx.arc(ballX, ballY, BALL_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(96,165,250,0.9)'; // bright blue
    ctx.fill();
    // glow band kar — baaki drawings pe nahi chahiye
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // ball ke andar chhota bright center — glass effect jaisa
    ctx.beginPath();
    ctx.arc(ballX - 3, ballY - 3, BALL_RADIUS * 0.35, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(200,225,255,0.5)';
    ctx.fill();

    // error magnitude text — top-right corner mein
    ctx.font = '12px monospace';
    ctx.fillStyle = '#b0b0b0';
    ctx.textAlign = 'right';
    ctx.fillText('error: ' + errorMag.toFixed(1) + 'px', w - 10, 20);

    // hint dikhao — click to set target
    if (!mouseInCanvas) {
      ctx.font = '13px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.4)';
      ctx.textAlign = 'center';
      ctx.fillText('click to set target', w / 2, h / 2 + 40);
    }
  }

  // --- Error-over-time plot draw karo ---
  // ye chhota graph hai jo oscillation patterns clearly dikhata hai
  function drawErrorPlot() {
    const ctx = plotCanvas.getContext('2d');
    const w = plotCanvas.clientWidth;
    const h = plotCanvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    if (errorHistory.length < 2) return;

    // Y-axis auto-scale — maximum error dhundho aur thoda padding do
    let maxError = 0;
    for (let i = 0; i < errorHistory.length; i++) {
      if (errorHistory[i] > maxError) maxError = errorHistory[i];
    }
    // minimum Y range rakh — nahi toh flat line pe graph pagal lagta hai
    maxError = Math.max(maxError, 20);
    const padding = { top: 8, bottom: 14, left: 40, right: 10 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    // grid lines — halki lines taaki graph readable lage
    ctx.strokeStyle = 'rgba(249,158,11,0.07)';
    ctx.lineWidth = 1;
    const gridLines = 3;
    for (let i = 0; i <= gridLines; i++) {
      const y = padding.top + (plotH / gridLines) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();

      // Y-axis labels — error values dikhao
      const val = maxError * (1 - i / gridLines);
      ctx.font = '9px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.4)';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(0), padding.left - 4, y + 3);
    }

    // error line chart draw karo
    ctx.beginPath();
    for (let i = 0; i < errorHistory.length; i++) {
      const x = padding.left + (i / (ERROR_HISTORY_LENGTH - 1)) * plotW;
      const y = padding.top + plotH * (1 - errorHistory[i] / maxError);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(249,158,11,0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // area fill — line ke neeche halka fill taaki graph aur readable lage
    if (errorHistory.length > 1) {
      const lastX = padding.left + ((errorHistory.length - 1) / (ERROR_HISTORY_LENGTH - 1)) * plotW;
      ctx.lineTo(lastX, padding.top + plotH);
      ctx.lineTo(padding.left, padding.top + plotH);
      ctx.closePath();
      ctx.fillStyle = 'rgba(249,158,11,0.04)';
      ctx.fill();
    }

    // axis label
    ctx.font = '9px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.textAlign = 'center';
    ctx.fillText('error over time', w / 2, h - 2);
  }

  // --- Main animation loop ---
  // har frame mein PID step chala, trail update kar, draw kar
  function animate() {
    if (!isVisible) return;

    // PID step — physics update
    const errorMag = pidStep();

    // trail mein current position add kar
    trail.push({ x: ballX, y: ballY, error: errorMag });
    // purane trail points hata — sirf last TRAIL_LENGTH rakh
    if (trail.length > TRAIL_LENGTH) {
      trail.shift();
    }

    // error history mein add kar — plot ke liye
    errorHistory.push(errorMag);
    if (errorHistory.length > ERROR_HISTORY_LENGTH) {
      errorHistory.shift();
    }

    // draw sab kuch
    drawMainCanvas(errorMag);
    drawErrorPlot();

    // next frame maang
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf jab visible ho tab animate karo ---
  // CPU waste nahi karni jab user dekh hi nahi raha
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
    { threshold: 0.1 } // 10% dikhe toh start karo
  );

  observer.observe(container);

  // page hidden ho jaye (tab switch) toh bhi pause karo — battery bachao
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      // check karo ki container abhi bhi visible hai ya nahi
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });
}
