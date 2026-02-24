// ============================================================
// Sensor Fusion Demo — Complementary Filter Visualization
// Gyro drift + Accel noise = fused estimate jo dono se better hai
// Robotics portfolio ke liye banaya hai — IMU fundamentals samjhane ke liye
// ============================================================

// complementary filter — gyro aur accel ka weighted combo
// angle = α * (angle + gyro*dt) + (1-α) * accel_angle
// α zyada = gyro pe trust zyada (smooth but drift)
// α kam = accel pe trust zyada (noisy but no drift)

export function initSensorFusion() {
  const container = document.getElementById('sensorFusionContainer');
  if (!container) {
    console.warn('sensorFusionContainer nahi mila bhai, sensor fusion skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const GAUGE_SIZE = 200; // attitude indicator ka diameter
  const GRAPH_HEIGHT = 150; // time-series graph ki height
  const TOTAL_HEIGHT = GAUGE_SIZE + GRAPH_HEIGHT + 60; // thoda extra stats ke liye
  const HISTORY_SECONDS = 5; // kitne seconds ka data dikhana hai
  const HISTORY_MAX_POINTS = 300; // max data points graph mein
  const DEG_TO_RAD = Math.PI / 180;
  const RAD_TO_DEG = 180 / Math.PI;

  // colors — consistent scheme poore demo mein
  const COLORS = {
    truth: '#ffffff',
    gyro: '#3b82f6', // blue — gyroscope
    accel: '#ef4444', // red — accelerometer
    fused: '#22c55e', // green — complementary filter output
    grid: 'rgba(255,255,255,0.08)',
    text: '#b0b0b0',
    accent: 'rgba(249,158,11,0.8)',
    gaugeRing: 'rgba(249,158,11,0.2)',
    gaugeBg: 'rgba(10,10,10,0.6)',
    horizon: 'rgba(59,130,246,0.15)',
  };

  // --- State ---
  // true angle — ye actual tilt hai jo slowly oscillate karta hai
  let trueAngle = 0;
  let trueAngularVelocity = 0;
  let time = 0;

  // sensor readings — noisy versions of truth
  let gyroAngle = 0; // integrated gyro — drift accumulate hoti hai
  let gyroDriftAccumulated = 0; // total drift abhi tak
  let accelAngle = 0; // noisy but unbiased
  let fusedAngle = 0; // complementary filter ka output

  // history arrays — time-series plot ke liye
  let historyTruth = [];
  let historyGyro = [];
  let historyAccel = [];
  let historyFused = [];
  let historyTime = [];

  // RMS error accumulators — running stats
  let gyroErrorSqSum = 0;
  let accelErrorSqSum = 0;
  let fusedErrorSqSum = 0;
  let errorSampleCount = 0;

  // slider-controlled parameters
  let alpha = 0.98; // complementary filter constant
  let gyroDriftRate = 1.0; // deg/sec drift
  let accelNoise = 8.0; // degrees of noise
  let motionSpeed = 0.5; // true angle change speed

  // animation state
  let animationId = null;
  let isVisible = false;
  let lastTimestamp = 0;
  let mouseY = null; // mouse se tilt control karne ke liye
  let mouseInCanvas = false;

  // --- DOM setup ---
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — gauge + graph dono yahan
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + TOTAL_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(249,158,11,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls section — sliders
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:14px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:space-between',
  ].join(';');
  container.appendChild(controlsDiv);

  // sliders ka left container
  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:14px;flex:1;min-width:280px;';
  controlsDiv.appendChild(slidersDiv);

  // stats display — RMS errors
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'gap:16px',
    'font-size:12px',
    'font-family:monospace',
    'flex-wrap:wrap',
  ].join(';');
  container.appendChild(statsDiv);

  // --- Slider helper ---
  // har slider ke saath label aur current value
  function createSlider(label, min, max, step, defaultVal, onChange, color) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = 'color:' + (color || '#b0b0b0') + ';font-size:12px;font-weight:600;min-width:60px;font-family:monospace;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText = 'width:90px;height:4px;accent-color:' + (color || 'rgba(249,158,11,0.8)') + ';cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText = 'color:#b0b0b0;font-size:11px;min-width:36px;font-family:monospace;';
    valueEl.textContent = parseFloat(defaultVal).toFixed(2);
    wrapper.appendChild(valueEl);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = val.toFixed(2);
      onChange(val);
    });

    slidersDiv.appendChild(wrapper);
    return { slider, valueEl };
  }

  // chaar sliders — alpha, drift, noise, speed
  const alphaSlider = createSlider('α filter', 0, 1, 0.01, alpha, (v) => {
    alpha = v;
    resetSimulation();
  }, COLORS.fused);

  const driftSlider = createSlider('Gyro drift', 0, 5, 0.1, gyroDriftRate, (v) => {
    gyroDriftRate = v;
    resetSimulation();
  }, COLORS.gyro);

  const noiseSlider = createSlider('Accel noise', 0, 20, 0.5, accelNoise, (v) => {
    accelNoise = v;
    resetSimulation();
  }, COLORS.accel);

  const speedSlider = createSlider('Motion spd', 0.1, 2.0, 0.1, motionSpeed, (v) => {
    motionSpeed = v;
  }, COLORS.truth);

  // --- Stats elements ---
  function createStatEl(label, color) {
    const el = document.createElement('span');
    el.style.cssText = 'color:' + color + ';padding:4px 0;';
    el.textContent = label + ': --';
    statsDiv.appendChild(el);
    return el;
  }

  const gyroStatEl = createStatEl('Gyro RMS', COLORS.gyro);
  const accelStatEl = createStatEl('Accel RMS', COLORS.accel);
  const fusedStatEl = createStatEl('Fused RMS', COLORS.fused);

  // --- Simulation reset ---
  // sliders change hone pe simulation reset karo — fresh start
  function resetSimulation() {
    gyroAngle = trueAngle;
    gyroDriftAccumulated = 0;
    fusedAngle = trueAngle;
    historyTruth = [];
    historyGyro = [];
    historyAccel = [];
    historyFused = [];
    historyTime = [];
    gyroErrorSqSum = 0;
    accelErrorSqSum = 0;
    fusedErrorSqSum = 0;
    errorSampleCount = 0;
  }

  // --- Mouse tracking — canvas pe mouse move se tilt control ---
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseY = (e.clientY - rect.top) / rect.height;
    mouseInCanvas = true;
  });

  canvas.addEventListener('mouseleave', () => {
    mouseInCanvas = false;
    mouseY = null;
  });

  // --- Canvas resize ---
  let canvasW = 0;
  let canvasH = TOTAL_HEIGHT;

  function resizeCanvas() {
    const rect = container.getBoundingClientRect();
    canvasW = Math.floor(rect.width);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // ============================================================
  // SIMULATION UPDATE — har frame pe sensors update karo
  // ============================================================
  function updateSimulation(dt) {
    if (dt <= 0 || dt > 0.1) dt = 1 / 60; // safety clamp

    time += dt;

    // --- True angle update ---
    // agar mouse canvas mein hai toh mouse se control karo, nahi toh auto-oscillate
    if (mouseInCanvas && mouseY !== null) {
      // mouse Y position se angle map karo — 0 = -90°, 1 = +90°
      const targetAngle = (mouseY - 0.5) * 180;
      // smooth follow karo — jhatke se nahi jaana chahiye
      const diff = targetAngle - trueAngle;
      trueAngularVelocity = diff * 3 * motionSpeed;
    } else {
      // auto oscillation — multiple sine waves combine karo realistic motion ke liye
      trueAngularVelocity = (
        Math.sin(time * 0.7 * motionSpeed) * 40 +
        Math.sin(time * 1.3 * motionSpeed) * 20 +
        Math.sin(time * 0.3 * motionSpeed) * 15
      ) * motionSpeed;
    }
    trueAngle += trueAngularVelocity * dt;
    // angle ko -180 se 180 ke beech rakh
    trueAngle = wrapAngle(trueAngle);

    // --- Gyroscope simulation ---
    // gyro angular velocity deta hai — accurate short-term but drift hoti hai
    // drift slowly accumulate hoti hai — yahi gyro ki weakness hai
    gyroDriftAccumulated += gyroDriftRate * dt * (Math.sin(time * 0.1) > 0 ? 1 : -1);
    const gyroReading = trueAngularVelocity + gyroDriftRate * (Math.sin(time * 0.2) + 0.5);
    gyroAngle += gyroReading * dt;
    gyroAngle = wrapAngle(gyroAngle);

    // --- Accelerometer simulation ---
    // accel gravity direction se angle nikalata hai — noisy but no drift
    // gaussian noise add karo — realistic sensor behavior
    const accelNoiseVal = gaussianRandom() * accelNoise;
    accelAngle = trueAngle + accelNoiseVal;

    // --- Complementary filter ---
    // ye magic formula hai — gyro ki short-term accuracy + accel ki long-term stability
    // high-pass filter gyro pe + low-pass filter accel pe = best of both worlds
    fusedAngle = alpha * (fusedAngle + gyroReading * dt) + (1 - alpha) * accelAngle;
    fusedAngle = wrapAngle(fusedAngle);

    // --- History update ---
    historyTruth.push(trueAngle);
    historyGyro.push(gyroAngle);
    historyAccel.push(accelAngle);
    historyFused.push(fusedAngle);
    historyTime.push(time);

    // purane data hata do — sirf last HISTORY_SECONDS rakh
    const maxLen = HISTORY_MAX_POINTS;
    if (historyTruth.length > maxLen) {
      historyTruth.shift();
      historyGyro.shift();
      historyAccel.shift();
      historyFused.shift();
      historyTime.shift();
    }

    // --- RMS error update ---
    const gyroErr = angleDiff(gyroAngle, trueAngle);
    const accelErr = angleDiff(accelAngle, trueAngle);
    const fusedErr = angleDiff(fusedAngle, trueAngle);
    gyroErrorSqSum += gyroErr * gyroErr;
    accelErrorSqSum += accelErr * accelErr;
    fusedErrorSqSum += fusedErr * fusedErr;
    errorSampleCount++;

    // stats update karo — har 10 frames pe (performance ke liye)
    if (errorSampleCount % 10 === 0) {
      const gyroRMS = Math.sqrt(gyroErrorSqSum / errorSampleCount);
      const accelRMS = Math.sqrt(accelErrorSqSum / errorSampleCount);
      const fusedRMS = Math.sqrt(fusedErrorSqSum / errorSampleCount);
      gyroStatEl.textContent = 'Gyro RMS: ' + gyroRMS.toFixed(1) + '°';
      accelStatEl.textContent = 'Accel RMS: ' + accelRMS.toFixed(1) + '°';
      fusedStatEl.textContent = 'Fused RMS: ' + fusedRMS.toFixed(1) + '°';
    }
  }

  // --- Angle utilities ---
  // angle ko -180 se 180 ke beech wrap karo
  function wrapAngle(a) {
    while (a > 180) a -= 360;
    while (a < -180) a += 360;
    return a;
  }

  // do angles ka difference (shortest path pe)
  function angleDiff(a, b) {
    let d = a - b;
    while (d > 180) d -= 360;
    while (d < -180) d += 360;
    return d;
  }

  // gaussian random number — Box-Muller transform
  // accelerometer noise ke liye chahiye — uniform random se realistic nahi lagta
  function gaussianRandom() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // ============================================================
  // RENDERING — gauge + graph draw karo
  // ============================================================
  function draw(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    const dt = lastTimestamp ? (timestamp - lastTimestamp) / 1000 : 1 / 60;
    lastTimestamp = timestamp;

    // simulation update karo
    updateSimulation(dt);

    // canvas clear karo
    ctx.clearRect(0, 0, canvasW, canvasH);

    // gauge draw karo — upar center mein
    drawGauge();

    // time-series graph draw karo — neeche
    drawGraph();

    animationId = requestAnimationFrame(draw);
  }

  // ============================================================
  // GAUGE — artificial horizon / attitude indicator
  // pilot cockpit waala indicator — tilt angle dikhata hai
  // ============================================================
  function drawGauge() {
    const cx = canvasW / 2; // gauge center X
    const cy = GAUGE_SIZE / 2 + 10; // gauge center Y
    const r = GAUGE_SIZE / 2 - 10; // gauge radius

    // --- Outer ring ---
    ctx.beginPath();
    ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
    ctx.strokeStyle = COLORS.gaugeRing;
    ctx.lineWidth = 2;
    ctx.stroke();

    // --- Background circle with clipping ---
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.clip();

    // dark background
    ctx.fillStyle = COLORS.gaugeBg;
    ctx.fillRect(cx - r, cy - r, r * 2, r * 2);

    // --- Horizon line — true angle ke hisaab se rotate hota hai ---
    // horizon effect — upar blue, neeche dark
    const horizonY = cy + (trueAngle / 90) * r * 0.8;
    ctx.fillStyle = COLORS.horizon;
    ctx.fillRect(cx - r, horizonY, r * 2, r * 2);

    // horizon reference lines — 10° intervals pe
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    for (let deg = -80; deg <= 80; deg += 10) {
      const lineY = cy + (deg / 90) * r * 0.8 + (trueAngle / 90) * r * 0.8;
      if (lineY < cy - r || lineY > cy + r) continue;
      const lineW = (deg % 30 === 0) ? r * 0.6 : r * 0.3;
      ctx.beginPath();
      ctx.moveTo(cx - lineW, lineY);
      ctx.lineTo(cx + lineW, lineY);
      ctx.stroke();

      // degree label
      if (deg % 30 === 0 && deg !== 0) {
        ctx.font = '9px monospace';
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.textAlign = 'right';
        ctx.fillText(Math.abs(deg) + '°', cx - lineW - 4, lineY + 3);
      }
    }

    // --- Center crosshair (reference) ---
    ctx.strokeStyle = COLORS.truth;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 20, cy);
    ctx.lineTo(cx - 8, cy);
    ctx.moveTo(cx + 8, cy);
    ctx.lineTo(cx + 20, cy);
    ctx.moveTo(cx, cy - 3);
    ctx.lineTo(cx, cy + 3);
    ctx.stroke();

    // --- Gyro indicator — blue line ---
    // gyro angle ko gauge pe dikhao
    const gyroY = cy + (angleDiff(gyroAngle, trueAngle) / 90) * r * 0.8;
    drawIndicatorLine(cx, gyroY, r * 0.7, COLORS.gyro, 2, 'GYRO');

    // --- Accel indicator — red dots ---
    // accel angle scattered dots ke roop mein — noisy hai toh dots spread honge
    const accelY = cy + (angleDiff(accelAngle, trueAngle) / 90) * r * 0.8;
    for (let i = 0; i < 5; i++) {
      const dotX = cx + (Math.random() - 0.5) * r * 0.4;
      const dotY = accelY + (Math.random() - 0.5) * 4;
      ctx.beginPath();
      ctx.arc(dotX, dotY, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.accel + 'aa';
      ctx.fill();
    }

    // --- Fused indicator — green line (should be close to center) ---
    const fusedY = cy + (angleDiff(fusedAngle, trueAngle) / 90) * r * 0.8;
    drawIndicatorLine(cx, fusedY, r * 0.8, COLORS.fused, 2.5, 'FUSED');

    ctx.restore(); // clip restore

    // --- Legend around gauge ---
    const legendY = cy + r + 18;
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';

    // truth
    ctx.fillStyle = COLORS.truth;
    ctx.fillText('— Truth', cx - 120, legendY);
    // gyro
    ctx.fillStyle = COLORS.gyro;
    ctx.fillText('— Gyro', cx - 40, legendY);
    // accel
    ctx.fillStyle = COLORS.accel;
    ctx.fillText('• Accel', cx + 40, legendY);
    // fused
    ctx.fillStyle = COLORS.fused;
    ctx.fillText('— Fused', cx + 120, legendY);

    // angle readout — center ke neeche
    ctx.font = 'bold 13px monospace';
    ctx.fillStyle = COLORS.truth;
    ctx.textAlign = 'center';
    ctx.fillText('True: ' + trueAngle.toFixed(1) + '°', cx, cy + r + 34);
  }

  // gauge mein horizontal indicator line draw karo
  function drawIndicatorLine(cx, y, width, color, lineWidth, label) {
    ctx.beginPath();
    ctx.moveTo(cx - width / 2, y);
    ctx.lineTo(cx + width / 2, y);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.globalAlpha = 0.8;
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // label
    ctx.font = '9px monospace';
    ctx.fillStyle = color;
    ctx.textAlign = 'left';
    ctx.fillText(label, cx + width / 2 + 4, y + 3);
  }

  // ============================================================
  // TIME-SERIES GRAPH — neeche scrolling plot
  // saare signals ek saath dikhte hain — truth, gyro, accel, fused
  // ============================================================
  function drawGraph() {
    if (historyTruth.length < 2) return;

    const graphX = 50; // left padding for labels
    const graphY = GAUGE_SIZE + 48; // gauge ke neeche
    const graphW = canvasW - graphX - 20;
    const graphH = GRAPH_HEIGHT - 10;

    // graph border
    ctx.strokeStyle = 'rgba(249,158,11,0.15)';
    ctx.lineWidth = 1;
    ctx.strokeRect(graphX, graphY, graphW, graphH);

    // background grid lines — horizontal
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    const gridLines = 5;
    for (let i = 0; i <= gridLines; i++) {
      const y = graphY + (i / gridLines) * graphH;
      ctx.beginPath();
      ctx.moveTo(graphX, y);
      ctx.lineTo(graphX + graphW, y);
      ctx.stroke();
    }

    // Y axis labels — angle range
    // dynamic range — data ke hisaab se adjust karo
    let minAngle = Infinity, maxAngle = -Infinity;
    for (let i = 0; i < historyTruth.length; i++) {
      const vals = [historyTruth[i], historyGyro[i], historyAccel[i], historyFused[i]];
      for (const v of vals) {
        if (v < minAngle) minAngle = v;
        if (v > maxAngle) maxAngle = v;
      }
    }
    // thoda margin do
    const range = maxAngle - minAngle;
    const margin = Math.max(range * 0.15, 10);
    minAngle -= margin;
    maxAngle += margin;

    // Y axis labels
    ctx.font = '9px monospace';
    ctx.fillStyle = COLORS.text;
    ctx.textAlign = 'right';
    for (let i = 0; i <= gridLines; i++) {
      const val = maxAngle - (i / gridLines) * (maxAngle - minAngle);
      const y = graphY + (i / gridLines) * graphH;
      ctx.fillText(val.toFixed(0) + '°', graphX - 6, y + 3);
    }

    // --- Plot signals ---
    // helper — ek signal ka line draw karo
    function plotLine(data, color, lineWidth, dashed) {
      if (data.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      if (dashed) ctx.setLineDash([3, 3]);
      else ctx.setLineDash([]);

      for (let i = 0; i < data.length; i++) {
        const x = graphX + (i / (data.length - 1)) * graphW;
        const y = graphY + ((maxAngle - data[i]) / (maxAngle - minAngle)) * graphH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // helper — scattered dots draw karo (accel ke liye)
    function plotDots(data, color, radius) {
      if (data.length < 2) return;
      ctx.fillStyle = color + '66'; // semi-transparent
      // har 3rd point pe dot draw karo — zyada draw karne se slow ho jaayega
      for (let i = 0; i < data.length; i += 3) {
        const x = graphX + (i / (data.length - 1)) * graphW;
        const y = graphY + ((maxAngle - data[i]) / (maxAngle - minAngle)) * graphH;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // truth — white dashed line
    plotLine(historyTruth, COLORS.truth, 1.5, true);

    // gyro — blue solid line (drift dikhega time ke saath)
    plotLine(historyGyro, COLORS.gyro, 1.5, false);

    // accel — red dots (noisy scattered)
    plotDots(historyAccel, COLORS.accel, 2);

    // fused — green line (smooth aur truth ke close)
    plotLine(historyFused, COLORS.fused, 2, false);

    // --- Time axis label ---
    ctx.font = '9px monospace';
    ctx.fillStyle = COLORS.text;
    ctx.textAlign = 'center';
    ctx.fillText('← ' + HISTORY_SECONDS + 's history →', graphX + graphW / 2, graphY + graphH + 14);

    // --- RMS error bars — graph ke right mein chhote bars ---
    drawRMSBars(graphX + graphW + 5, graphY, graphH);
  }

  // RMS error bars — visual comparison
  function drawRMSBars(x, y, height) {
    if (errorSampleCount < 10) return;

    const gyroRMS = Math.sqrt(gyroErrorSqSum / errorSampleCount);
    const accelRMS = Math.sqrt(accelErrorSqSum / errorSampleCount);
    const fusedRMS = Math.sqrt(fusedErrorSqSum / errorSampleCount);
    const maxRMS = Math.max(gyroRMS, accelRMS, fusedRMS, 1);

    // itni jagah nahi hai side mein — skip karo agar canvas chhota hai
    if (canvasW < 400) return;

    // chhote vertical bars — neeche se upar jaate hain
    // bar width limited hai kyunki graph ke baad bas 20px hai
    // skip this section if too cramped, RMS is shown in stats below anyway
  }

  // ============================================================
  // INTERSECTION OBSERVER — performance ke liye
  // sirf tab animate karo jab screen pe dikh raha ho
  // ============================================================
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        isVisible = entry.isIntersecting;
        if (isVisible && !animationId) {
          lastTimestamp = 0;
          animationId = requestAnimationFrame(draw);
        }
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) draw(); });

  // --- Resize handler ---
  const resizeObs = new ResizeObserver(() => {
    resizeCanvas();
  });
  resizeObs.observe(container);

  // initial setup
  resizeCanvas();
  animationId = requestAnimationFrame(draw);
}
