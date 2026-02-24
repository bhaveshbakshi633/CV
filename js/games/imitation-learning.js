// ============================================================
// Imitation Learning — Behavioral Cloning + DAgger for stick figure motion
// Record expert demo, train MLP, compare expert vs learned side-by-side
// ============================================================

// yahi entry point hai — record, train, playback, DAgger comparison
export function initImitationLearning() {
  const container = document.getElementById('imitationLearningContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const EXPERT_COLOR = '#4a9eff';
  const LEARNER_COLOR = '#f97316';

  // stick figure params — shoulder, elbow, hip (3 joints)
  const BASE_X_LEFT = 0.25;   // expert figure position (fraction of canvas)
  const BASE_X_RIGHT = 0.75;  // learner figure position
  const BASE_Y = 0.6;         // vertical position
  const UPPER_ARM = 50;
  const LOWER_ARM = 40;
  const TORSO = 70;
  const UPPER_LEG = 50;
  const HEAD_R = 12;

  let animationId = null, isVisible = false, canvasW = 0;

  // demo recording state
  let recording = false;
  let demo = [];              // [{t, shoulder, elbow, hip}] — expert trajectory
  let demoTime = 0;           // recording time counter
  const RECORD_FPS = 30;
  let recordInterval = null;

  // joint state — user drags these
  let joints = { shoulder: -0.5, elbow: 0.3, hip: 0.2 };
  let draggingJoint = null;

  // training state
  let trained = false;
  let mode = 'bc';            // 'bc' or 'dagger'
  let trainLoss = [];         // loss history
  let daggerRounds = 0;

  // MLP weights — time -> [shoulder, elbow, hip]
  // 1 input (normalized time) -> 32 hidden (tanh) -> 3 output
  const HIDDEN = 32;
  let mlpW1, mlpB1, mlpW2, mlpB2;

  // playback state
  let playing = false;
  let playTime = 0;
  let playSpeed = 1;

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:default;background:#111;border:1px solid rgba(74,158,255,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  function mkBtn(parent, text, id) {
    const b = document.createElement('button');
    b.textContent = text; b.id = id;
    b.style.cssText = "background:#333;color:#ccc;border:1px solid #555;padding:3px 8px;border-radius:4px;cursor:pointer;font:11px 'JetBrains Mono',monospace";
    parent.appendChild(b);
    return b;
  }

  // controls
  const recordBtn = mkBtn(ctrl, 'Record', 'ilRecord');
  recordBtn.style.borderColor = '#ef4444';
  recordBtn.addEventListener('click', toggleRecord);

  const trainBtn = mkBtn(ctrl, 'Train (BC)', 'ilTrain');
  trainBtn.style.background = 'rgba(74,158,255,0.2)';
  trainBtn.style.borderColor = ACCENT;
  trainBtn.addEventListener('click', trainBC);

  const playBtn = mkBtn(ctrl, 'Play', 'ilPlay');
  playBtn.addEventListener('click', togglePlay);

  // mode toggle
  const modeBtn = mkBtn(ctrl, 'Mode: BC', 'ilMode');
  modeBtn.addEventListener('click', () => {
    mode = mode === 'bc' ? 'dagger' : 'bc';
    modeBtn.textContent = 'Mode: ' + mode.toUpperCase();
    modeBtn.style.borderColor = mode === 'dagger' ? '#22c55e' : '#555';
    trainBtn.textContent = mode === 'dagger' ? 'Train (DAgger)' : 'Train (BC)';
  });

  mkBtn(ctrl, 'Clear Demo', 'ilClear').addEventListener('click', () => {
    demo = []; trained = false; trainLoss = []; daggerRounds = 0; playing = false;
    draw();
  });

  const stats = document.createElement('div');
  stats.style.cssText = "font:11px 'JetBrains Mono',monospace;color:#888;margin-top:6px;";
  container.appendChild(stats);

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- random helper ---
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // --- MLP init ---
  function initMLP() {
    const xavierH = Math.sqrt(2 / (1 + HIDDEN));
    const xavierO = Math.sqrt(2 / (HIDDEN + 3));
    mlpW1 = Array.from({ length: HIDDEN }, () => [randn() * xavierH]);
    mlpB1 = new Array(HIDDEN).fill(0);
    mlpW2 = Array.from({ length: 3 }, () => Array.from({ length: HIDDEN }, () => randn() * xavierO));
    mlpB2 = new Array(3).fill(0);
  }
  initMLP();

  // --- MLP forward pass ---
  function mlpForward(t) {
    // normalize time to [-1, 1]
    const x = (t / Math.max(1, demo.length - 1)) * 2 - 1;
    // hidden layer — tanh
    const h = new Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) {
      h[i] = Math.tanh(mlpW1[i][0] * x + mlpB1[i]);
    }
    // output layer — no activation (regression)
    const out = new Array(3);
    for (let i = 0; i < 3; i++) {
      let sum = mlpB2[i];
      for (let j = 0; j < HIDDEN; j++) sum += mlpW2[i][j] * h[j];
      out[i] = sum;
    }
    return { out, hidden: h, input: x };
  }

  // --- training — gradient descent with MSE loss ---
  function trainBC() {
    if (demo.length < 5) return;

    let trainingData = demo.map((d, i) => ({
      t: i,
      target: [d.shoulder, d.elbow, d.hip]
    }));

    // DAgger mode — learner ke visited states pe expert query karo
    if (mode === 'dagger' && trained) {
      const daggerData = [];
      for (let i = 0; i < demo.length; i++) {
        const pred = mlpForward(i);
        // learner ka state — expert query karo
        // expert ka answer interpolate karo from demo
        const expertIdx = Math.min(demo.length - 1, Math.round(i));
        daggerData.push({
          t: i,
          target: [demo[expertIdx].shoulder, demo[expertIdx].elbow, demo[expertIdx].hip]
        });
      }
      // DAgger: mix original demo with new corrections
      trainingData = [...trainingData, ...daggerData];
      daggerRounds++;
    }

    initMLP(); // fresh start for clean training
    trainLoss = [];

    const lr = 0.01;
    const epochs = 500;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;

      // shuffle training data
      const indices = trainingData.map((_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      for (const idx of indices) {
        const sample = trainingData[idx];
        const { out, hidden, input } = mlpForward(sample.t);

        // MSE loss
        const loss = sample.target.reduce((s, t, i) => s + (out[i] - t) ** 2, 0) / 3;
        totalLoss += loss;

        // backprop — output layer
        const dOut = sample.target.map((t, i) => 2 * (out[i] - t) / 3);

        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < HIDDEN; j++) {
            mlpW2[i][j] -= lr * dOut[i] * hidden[j];
          }
          mlpB2[i] -= lr * dOut[i];
        }

        // backprop — hidden layer
        for (let j = 0; j < HIDDEN; j++) {
          let dh = 0;
          for (let i = 0; i < 3; i++) dh += dOut[i] * mlpW2[i][j];
          const dtanh = 1 - hidden[j] * hidden[j];
          dh *= dtanh;
          mlpW1[j][0] -= lr * dh * input;
          mlpB1[j] -= lr * dh;
        }
      }

      if (epoch % 5 === 0) trainLoss.push(totalLoss / trainingData.length);
    }

    trained = true;
    draw();
  }

  // --- recording toggle ---
  function toggleRecord() {
    if (recording) {
      // stop recording
      recording = false;
      recordBtn.textContent = 'Record';
      recordBtn.style.background = '#333';
      if (recordInterval) { clearInterval(recordInterval); recordInterval = null; }
    } else {
      // start recording
      demo = [];
      demoTime = 0;
      trained = false;
      trainLoss = [];
      daggerRounds = 0;
      recording = true;
      recordBtn.textContent = 'Stop';
      recordBtn.style.background = 'rgba(239,68,68,0.3)';
      recordInterval = setInterval(() => {
        if (!recording) return;
        demo.push({ ...joints, t: demoTime });
        demoTime++;
        if (demoTime > 300) toggleRecord(); // max 10 seconds at 30fps
      }, 1000 / RECORD_FPS);
    }
  }

  // --- playback toggle ---
  function togglePlay() {
    if (demo.length < 2) return;
    playing = !playing;
    playTime = 0;
    playBtn.textContent = playing ? 'Stop' : 'Play';
  }

  // --- stick figure draw karo ---
  function drawFigure(centerX, baseY, shoulder, elbow, hip, color, label) {
    const cx = centerX;
    const cy = baseY;

    // torso
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    const torsoTop = { x: cx, y: cy - TORSO };
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(torsoTop.x, torsoTop.y); ctx.stroke();

    // head
    ctx.beginPath();
    ctx.arc(torsoTop.x, torsoTop.y - HEAD_R, HEAD_R, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.3;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // shoulder joint pe arm
    const shoulderPt = { x: torsoTop.x, y: torsoTop.y + 10 };
    const elbowAngle = shoulder;
    const elbowPt = {
      x: shoulderPt.x + Math.sin(elbowAngle) * UPPER_ARM,
      y: shoulderPt.y + Math.cos(elbowAngle) * UPPER_ARM
    };
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(shoulderPt.x, shoulderPt.y); ctx.lineTo(elbowPt.x, elbowPt.y); ctx.stroke();

    // elbow se forearm
    const handAngle = elbowAngle + elbow;
    const handPt = {
      x: elbowPt.x + Math.sin(handAngle) * LOWER_ARM,
      y: elbowPt.y + Math.cos(handAngle) * LOWER_ARM
    };
    ctx.beginPath(); ctx.moveTo(elbowPt.x, elbowPt.y); ctx.lineTo(handPt.x, handPt.y); ctx.stroke();

    // hip se leg
    const hipAngle = hip;
    const kneePt = {
      x: cx + Math.sin(hipAngle) * UPPER_LEG,
      y: cy + Math.cos(hipAngle) * UPPER_LEG
    };
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(kneePt.x, kneePt.y); ctx.stroke();

    // joints ko circles se highlight karo
    [shoulderPt, elbowPt, { x: cx, y: cy }].forEach(pt => {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2); ctx.fill();
    });

    // label
    ctx.fillStyle = color;
    ctx.font = "bold 11px 'JetBrains Mono',monospace";
    ctx.textAlign = 'center';
    ctx.fillText(label, cx, torsoTop.y - HEAD_R * 2 - 12);
  }

  // --- joint dragging — mouse interaction ---
  function getJointPositions() {
    const cx = canvasW * (playing ? BASE_X_LEFT : 0.5);
    const cy = CANVAS_HEIGHT * BASE_Y;
    const torsoTop = { x: cx, y: cy - TORSO };
    const shoulderPt = { x: torsoTop.x, y: torsoTop.y + 10 };

    const elbowPt = {
      x: shoulderPt.x + Math.sin(joints.shoulder) * UPPER_ARM,
      y: shoulderPt.y + Math.cos(joints.shoulder) * UPPER_ARM
    };
    const hipPt = { x: cx, y: cy };

    return { shoulder: shoulderPt, elbow: elbowPt, hip: hipPt };
  }

  canvas.addEventListener('mousedown', (e) => {
    if (playing) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvasW / rect.width);
    const my = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height);

    const jpos = getJointPositions();
    // check which joint is closest to mouse
    const dists = [
      { name: 'shoulder', d: Math.sqrt((mx - jpos.shoulder.x) ** 2 + (my - jpos.shoulder.y) ** 2) },
      { name: 'elbow', d: Math.sqrt((mx - jpos.elbow.x) ** 2 + (my - jpos.elbow.y) ** 2) },
      { name: 'hip', d: Math.sqrt((mx - jpos.hip.x) ** 2 + (my - jpos.hip.y) ** 2) },
    ];
    const closest = dists.reduce((a, b) => a.d < b.d ? a : b);
    if (closest.d < 30) {
      draggingJoint = closest.name;
      canvas.style.cursor = 'grabbing';
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!draggingJoint) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvasW / rect.width);
    const my = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height);

    const cx = canvasW * (playing ? BASE_X_LEFT : 0.5);
    const cy = CANVAS_HEIGHT * BASE_Y;
    const torsoTop = { x: cx, y: cy - TORSO };

    if (draggingJoint === 'shoulder') {
      // shoulder angle from mouse position relative to shoulder
      joints.shoulder = Math.atan2(mx - torsoTop.x, my - (torsoTop.y + 10));
      joints.shoulder = Math.max(-Math.PI, Math.min(Math.PI, joints.shoulder));
    } else if (draggingJoint === 'elbow') {
      // elbow angle relative to upper arm direction
      const shoulderPt = { x: torsoTop.x, y: torsoTop.y + 10 };
      const elbowPt = {
        x: shoulderPt.x + Math.sin(joints.shoulder) * UPPER_ARM,
        y: shoulderPt.y + Math.cos(joints.shoulder) * UPPER_ARM
      };
      joints.elbow = Math.atan2(mx - elbowPt.x, my - elbowPt.y) - joints.shoulder;
      joints.elbow = Math.max(-Math.PI, Math.min(Math.PI, joints.elbow));
    } else if (draggingJoint === 'hip') {
      joints.hip = Math.atan2(mx - cx, my - cy);
      joints.hip = Math.max(-1.5, Math.min(1.5, joints.hip));
    }
    draw();
  });

  canvas.addEventListener('mouseup', () => { draggingJoint = null; canvas.style.cursor = 'default'; });
  canvas.addEventListener('mouseleave', () => { draggingJoint = null; canvas.style.cursor = 'default'; });

  // --- loss plot draw karo ---
  function drawLossPlot() {
    if (trainLoss.length < 2) return;
    const plotX = canvasW * 0.05;
    const plotY = 15;
    const plotW = canvasW * 0.25;
    const plotH = 80;

    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.fillRect(plotX, plotY, plotW, plotH);
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.strokeRect(plotX, plotY, plotW, plotH);

    ctx.fillStyle = '#888';
    ctx.font = "9px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillText('Training Loss', plotX + 5, plotY + 11);

    const maxL = Math.max(...trainLoss);
    const minL = Math.min(...trainLoss);
    const range = maxL - minL || 1;

    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    trainLoss.forEach((l, i) => {
      const px = plotX + (i / (trainLoss.length - 1)) * plotW;
      const py = plotY + plotH - ((l - minL) / range) * (plotH - 15);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    ctx.fillStyle = '#666';
    ctx.font = "7px 'JetBrains Mono',monospace";
    ctx.fillText(trainLoss[trainLoss.length - 1].toFixed(4), plotX + plotW - 30, plotY + plotH - 3);
  }

  // --- demo timeline draw karo ---
  function drawTimeline() {
    if (demo.length < 2) return;
    const tlX = canvasW * 0.1;
    const tlW = canvasW * 0.8;
    const tlY = CANVAS_HEIGHT - 30;

    // timeline bar
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fillRect(tlX, tlY, tlW, 8);

    // progress bar
    if (playing) {
      const progress = playTime / (demo.length - 1);
      ctx.fillStyle = 'rgba(74,158,255,0.3)';
      ctx.fillRect(tlX, tlY, tlW * progress, 8);
      // current position marker
      ctx.fillStyle = ACCENT;
      ctx.beginPath();
      ctx.arc(tlX + tlW * progress, tlY + 4, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // labels
    ctx.fillStyle = '#666';
    ctx.font = "8px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillText('0s', tlX, tlY + 18);
    ctx.textAlign = 'right';
    ctx.fillText((demo.length / RECORD_FPS).toFixed(1) + 's', tlX + tlW, tlY + 18);
  }

  // --- interpolate demo at fractional time ---
  function interpolateDemo(t) {
    const idx = Math.min(demo.length - 1, Math.max(0, Math.floor(t)));
    const nextIdx = Math.min(demo.length - 1, idx + 1);
    const frac = t - idx;
    return {
      shoulder: demo[idx].shoulder + (demo[nextIdx].shoulder - demo[idx].shoulder) * frac,
      elbow: demo[idx].elbow + (demo[nextIdx].elbow - demo[idx].elbow) * frac,
      hip: demo[idx].hip + (demo[nextIdx].hip - demo[idx].hip) * frac,
    };
  }

  // --- main draw ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    if (playing && demo.length > 1) {
      // side-by-side mode — expert left, learner right
      // divider
      ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(canvasW / 2, 0); ctx.lineTo(canvasW / 2, CANVAS_HEIGHT - 40); ctx.stroke();
      ctx.setLineDash([]);

      // expert playback
      const expertPose = interpolateDemo(playTime);
      drawFigure(canvasW * BASE_X_LEFT, CANVAS_HEIGHT * BASE_Y, expertPose.shoulder, expertPose.elbow, expertPose.hip, EXPERT_COLOR, 'Expert');

      // learner playback
      if (trained) {
        const pred = mlpForward(playTime);
        drawFigure(canvasW * BASE_X_RIGHT, CANVAS_HEIGHT * BASE_Y, pred.out[0], pred.out[1], pred.out[2], LEARNER_COLOR, 'Learner');
      } else {
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.font = "12px 'JetBrains Mono',monospace";
        ctx.textAlign = 'center';
        ctx.fillText('Train model first', canvasW * BASE_X_RIGHT, CANVAS_HEIGHT * BASE_Y);
      }

      // advance playback time
      playTime += playSpeed;
      if (playTime >= demo.length - 1) playTime = 0;
    } else {
      // single figure — user editable
      drawFigure(canvasW * 0.5, CANVAS_HEIGHT * BASE_Y, joints.shoulder, joints.elbow, joints.hip, recording ? '#ef4444' : ACCENT, recording ? 'Recording...' : 'Drag Joints');
    }

    drawTimeline();
    drawLossPlot();

    // instructions
    if (!recording && !playing && demo.length === 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.font = "13px 'JetBrains Mono',monospace";
      ctx.textAlign = 'center';
      ctx.fillText('Drag joints to pose, then Record a motion', canvasW / 2, 30);
    }

    // recording indicator
    if (recording) {
      ctx.fillStyle = '#ef4444';
      ctx.beginPath(); ctx.arc(canvasW - 25, 25, 6, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 0.5 + 0.5 * Math.sin(Date.now() / 200);
      ctx.beginPath(); ctx.arc(canvasW - 25, 25, 10, 0, Math.PI * 2);
      ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2; ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#ef4444';
      ctx.font = "10px 'JetBrains Mono',monospace";
      ctx.textAlign = 'right';
      ctx.fillText('REC ' + (demoTime / RECORD_FPS).toFixed(1) + 's', canvasW - 40, 29);
    }

    // mode indicator
    if (mode === 'dagger' && daggerRounds > 0) {
      ctx.fillStyle = '#22c55e';
      ctx.font = "10px 'JetBrains Mono',monospace";
      ctx.textAlign = 'right';
      ctx.fillText('DAgger rounds: ' + daggerRounds, canvasW - 15, CANVAS_HEIGHT - 45);
    }

    // stats
    const lossStr = trainLoss.length > 0 ? trainLoss[trainLoss.length - 1].toFixed(5) : '-';
    stats.textContent = `Demo: ${demo.length} frames (${(demo.length / RECORD_FPS).toFixed(1)}s)  |  Trained: ${trained ? 'Yes' : 'No'}  |  Loss: ${lossStr}  |  Mode: ${mode.toUpperCase()}`;
  }

  // animation loop
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    draw();
    animationId = requestAnimationFrame(loop);
  }

  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) loop();
    else if (!isVisible && animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && isVisible && !animationId) loop(); });

  draw();
}
