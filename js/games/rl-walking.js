// ============================================================
// RL Walking — Stick figure learning to walk with REINFORCE policy gradient
// Isometric 3D rendering, torque-controlled joints, reward plot
// ============================================================

// yahi entry point hai — physics sim + RL training loop + isometric rendering
export function initRLWalking() {
  const container = document.getElementById('rlWalkingContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const DT = 0.02;           // physics timestep — 50Hz
  const GRAVITY = -9.8;
  const GROUND_Y = 0;        // ground level
  const TORSO_LEN = 0.6;     // torso length
  const THIGH_LEN = 0.4;     // upper leg
  const SHIN_LEN = 0.35;     // lower leg

  let animationId = null, isVisible = false, canvasW = 0;
  let trainingSpeed = 2;     // episodes per frame
  let learningRate = 0.005;
  let paused = false;
  let episode = 0;
  let rewards = [];           // reward history per episode
  let bestReward = -Infinity;

  // neural network weights — simple 2-layer MLP
  // state: [body_x, body_y, body_angle, 4 joint_angles, 4 joint_velocities, body_vx, body_vy] = 13D
  // action: 4 torques (continuous)
  const STATE_DIM = 13;
  const HIDDEN = 16;
  const ACTION_DIM = 4;

  // weight matrices — Xavier initialization
  let W1, b1, W2, b2;
  // log_std — learnable standard deviation for policy
  let logStd;

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:default;background:#111;border:1px solid rgba(74,158,255,0.15);`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const ctrl = document.createElement('div');
  ctrl.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;align-items:center;';
  container.appendChild(ctrl);

  function mkSlider(parent, label, id, min, max, val, step) {
    const lbl = document.createElement('label');
    lbl.style.cssText = "color:#ccc;font:12px 'JetBrains Mono',monospace";
    lbl.textContent = label + ' ';
    const inp = document.createElement('input');
    inp.type = 'range'; inp.min = min; inp.max = max; inp.value = val; inp.id = id;
    if (step) inp.step = step;
    inp.style.cssText = 'width:80px;vertical-align:middle';
    lbl.appendChild(inp);
    parent.appendChild(lbl);
    return inp;
  }
  function mkBtn(parent, text, id) {
    const b = document.createElement('button');
    b.textContent = text; b.id = id;
    b.style.cssText = "background:#333;color:#ccc;border:1px solid #555;padding:3px 8px;border-radius:4px;cursor:pointer;font:11px 'JetBrains Mono',monospace";
    parent.appendChild(b);
    return b;
  }

  const speedSlider = mkSlider(ctrl, 'Speed', 'rlwSpeed', 1, 10, trainingSpeed, 1);
  const speedVal = document.createElement('span');
  speedVal.style.cssText = "color:#4a9eff;font:11px 'JetBrains Mono',monospace;min-width:18px";
  speedVal.textContent = trainingSpeed;
  ctrl.appendChild(speedVal);
  speedSlider.addEventListener('input', () => { trainingSpeed = +speedSlider.value; speedVal.textContent = trainingSpeed; });

  const lrSlider = mkSlider(ctrl, 'LR', 'rlwLR', 0.001, 0.05, learningRate, 0.001);
  const lrVal = document.createElement('span');
  lrVal.style.cssText = "color:#4a9eff;font:11px 'JetBrains Mono',monospace;min-width:36px";
  lrVal.textContent = learningRate.toFixed(3);
  ctrl.appendChild(lrVal);
  lrSlider.addEventListener('input', () => { learningRate = +lrSlider.value; lrVal.textContent = learningRate.toFixed(3); });

  const pauseBtn = mkBtn(ctrl, 'Pause', 'rlwPause');
  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  });

  mkBtn(ctrl, 'Reset', 'rlwReset').addEventListener('click', () => {
    initWeights();
    episode = 0; rewards = []; bestReward = -Infinity;
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

  // --- random helpers ---
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // --- weight initialization ---
  function initWeights() {
    const xavierH = Math.sqrt(2 / (STATE_DIM + HIDDEN));
    const xavierO = Math.sqrt(2 / (HIDDEN + ACTION_DIM));
    W1 = Array.from({ length: HIDDEN }, () => Array.from({ length: STATE_DIM }, () => randn() * xavierH));
    b1 = new Array(HIDDEN).fill(0);
    W2 = Array.from({ length: ACTION_DIM }, () => Array.from({ length: HIDDEN }, () => randn() * xavierO));
    b2 = new Array(ACTION_DIM).fill(0);
    logStd = new Array(ACTION_DIM).fill(-0.5); // initial std ~ 0.6
  }
  initWeights();

  // --- forward pass — tanh activation ---
  function forward(state) {
    // hidden layer
    const h = new Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) {
      let sum = b1[i];
      for (let j = 0; j < STATE_DIM; j++) sum += W1[i][j] * state[j];
      h[i] = Math.tanh(sum);
    }
    // output — mean of Gaussian policy
    const mean = new Array(ACTION_DIM);
    for (let i = 0; i < ACTION_DIM; i++) {
      let sum = b2[i];
      for (let j = 0; j < HIDDEN; j++) sum += W2[i][j] * h[j];
      mean[i] = Math.tanh(sum) * 5; // clamp torque to [-5, 5]
    }
    return { mean, hidden: h };
  }

  // --- sample action from policy ---
  function sampleAction(state) {
    const { mean, hidden } = forward(state);
    const action = mean.map((m, i) => m + Math.exp(logStd[i]) * randn());
    return { action, mean, hidden };
  }

  // --- walker physics state ---
  function createWalker() {
    return {
      x: 0, y: TORSO_LEN + THIGH_LEN + SHIN_LEN + 0.05,  // body center
      vx: 0, vy: 0,
      angle: 0, angVel: 0,     // torso angle
      // joints: [left_hip, left_knee, right_hip, right_knee]
      joints: [0.2, -0.3, -0.2, -0.3],
      jointVels: [0, 0, 0, 0],
    };
  }

  // --- get state vector ---
  function getState(w) {
    return [
      w.x * 0.1, w.y, w.angle,
      ...w.joints, ...w.jointVels,
      w.vx, w.vy
    ];
  }

  // --- forward kinematics — joint positions compute karo ---
  function getPositions(w) {
    // torso center
    const cx = w.x, cy = w.y;
    // torso endpoints
    const torsoTop = { x: cx + Math.sin(w.angle) * TORSO_LEN * 0.5, y: cy + Math.cos(w.angle) * TORSO_LEN * 0.5 };
    const hip = { x: cx - Math.sin(w.angle) * TORSO_LEN * 0.5, y: cy - Math.cos(w.angle) * TORSO_LEN * 0.5 };

    // left leg
    const lHipAngle = w.angle + w.joints[0];
    const lKnee = { x: hip.x - Math.sin(lHipAngle) * THIGH_LEN, y: hip.y - Math.cos(lHipAngle) * THIGH_LEN };
    const lKneeAngle = lHipAngle + w.joints[1];
    const lFoot = { x: lKnee.x - Math.sin(lKneeAngle) * SHIN_LEN, y: lKnee.y - Math.cos(lKneeAngle) * SHIN_LEN };

    // right leg
    const rHipAngle = w.angle + w.joints[2];
    const rKnee = { x: hip.x - Math.sin(rHipAngle) * THIGH_LEN, y: hip.y - Math.cos(rHipAngle) * THIGH_LEN };
    const rKneeAngle = rHipAngle + w.joints[3];
    const rFoot = { x: rKnee.x - Math.sin(rKneeAngle) * SHIN_LEN, y: rKnee.y - Math.cos(rKneeAngle) * SHIN_LEN };

    return { torsoTop, hip, lKnee, lFoot, rKnee, rFoot };
  }

  // --- physics step — simplified ragdoll ---
  function physicsStep(w, torques) {
    // torques ko joints pe apply karo
    for (let i = 0; i < 4; i++) {
      const t = Math.max(-5, Math.min(5, torques[i]));
      w.jointVels[i] += t * DT * 3;
      w.jointVels[i] *= 0.95; // damping — nahi toh oscillate karega
      w.joints[i] += w.jointVels[i] * DT;
      // joint limits — realistic range
      if (i % 2 === 0) { // hip
        w.joints[i] = Math.max(-1.2, Math.min(1.2, w.joints[i]));
      } else { // knee
        w.joints[i] = Math.max(-1.5, Math.min(0.1, w.joints[i]));
      }
    }

    // torso dynamics — simplified
    const torqueOnBody = (torques[0] + torques[2]) * 0.1;
    w.angVel += torqueOnBody * DT;
    w.angVel += GRAVITY * Math.sin(w.angle) * DT * 0.3;
    w.angVel *= 0.98;
    w.angle += w.angVel * DT;

    // gravity
    w.vy += GRAVITY * DT;
    w.y += w.vy * DT;
    w.x += w.vx * DT;

    // ground contact — feet se check karo
    const pos = getPositions(w);
    let groundContact = false;
    [pos.lFoot, pos.rFoot].forEach(foot => {
      if (foot.y <= GROUND_Y) {
        groundContact = true;
        // ground reaction force
        w.vy = Math.max(0, w.vy);
        w.y += (GROUND_Y - foot.y) * 0.5;
        // friction — forward velocity se walking effect
        w.vx += (foot.x < w.x ? 0.3 : -0.3) * DT;
      }
    });

    // body minimum height maintain karo
    if (w.y < TORSO_LEN * 0.3) {
      w.y = TORSO_LEN * 0.3;
      w.vy = 0;
    }

    w.vx *= 0.99; // air resistance

    return groundContact;
  }

  // --- run one episode ---
  function runEpisode() {
    const walker = createWalker();
    const maxSteps = 150;
    let totalReward = 0;
    const trajectory = []; // {state, action, reward, mean, hidden}

    for (let t = 0; t < maxSteps; t++) {
      const state = getState(walker);
      const { action, mean, hidden } = sampleAction(state);

      // physics step
      physicsStep(walker, action);

      // reward: forward velocity - energy cost - fall penalty
      const forwardV = walker.vx;
      const energy = action.reduce((s, a) => s + a * a, 0) * 0.01;
      const alive = walker.y > TORSO_LEN * 0.2 ? 0.1 : -1;
      const upright = -Math.abs(walker.angle) * 0.5;
      const reward = forwardV * 2 + alive + upright - energy;

      totalReward += reward;
      trajectory.push({ state, action, mean, hidden, reward });

      // fallen check
      if (walker.y < TORSO_LEN * 0.15 || Math.abs(walker.angle) > 1.5) break;
    }

    return { totalReward, trajectory, finalWalker: walker };
  }

  // --- REINFORCE policy gradient update ---
  function updatePolicy(trajectory, totalReward) {
    const T = trajectory.length;
    // baseline = average reward (simple)
    const baseline = rewards.length > 0 ? rewards.slice(-20).reduce((s, r) => s + r, 0) / Math.min(20, rewards.length) : 0;

    // discounted returns compute karo
    const returns = new Array(T);
    let G = 0;
    for (let t = T - 1; t >= 0; t--) {
      G = trajectory[t].reward + 0.99 * G;
      returns[t] = G;
    }

    // gradient accumulate karo
    const dW1 = Array.from({ length: HIDDEN }, () => new Array(STATE_DIM).fill(0));
    const db1 = new Array(HIDDEN).fill(0);
    const dW2 = Array.from({ length: ACTION_DIM }, () => new Array(HIDDEN).fill(0));
    const db2 = new Array(ACTION_DIM).fill(0);
    const dLogStd = new Array(ACTION_DIM).fill(0);

    for (let t = 0; t < T; t++) {
      const { state, action, mean, hidden } = trajectory[t];
      const advantage = returns[t] - baseline;
      const std = logStd.map(ls => Math.exp(ls));

      // d log pi / d theta — Gaussian policy gradient
      for (let a = 0; a < ACTION_DIM; a++) {
        const diff = action[a] - mean[a];
        const dMean = diff / (std[a] * std[a]); // d log pi / d mean
        const scaledGrad = dMean * advantage;

        // backprop through output layer
        const dTanh = (1 - Math.tanh(mean[a] / 5) ** 2) * 5; // derivative of tanh scaling
        for (let h = 0; h < HIDDEN; h++) {
          dW2[a][h] += scaledGrad * dTanh * hidden[h];
        }
        db2[a] += scaledGrad * dTanh;

        // log_std gradient
        dLogStd[a] += (diff * diff / (std[a] * std[a]) - 1) * advantage;
      }

      // backprop through hidden layer (simplified — just update W1 based on W2 gradient)
      for (let h = 0; h < HIDDEN; h++) {
        let dh = 0;
        for (let a = 0; a < ACTION_DIM; a++) {
          const diff = action[a] - mean[a];
          const dTanh = (1 - Math.tanh(mean[a] / 5) ** 2) * 5;
          dh += W2[a][h] * (diff / (Math.exp(logStd[a]) ** 2)) * advantage * dTanh;
        }
        const dtanh = 1 - hidden[h] * hidden[h]; // tanh derivative
        dh *= dtanh;
        for (let s = 0; s < STATE_DIM; s++) {
          dW1[h][s] += dh * state[s];
        }
        db1[h] += dh;
      }
    }

    // gradient apply karo — REINFORCE update with clipping
    const scale = learningRate / T;
    const clipVal = 0.1;
    for (let i = 0; i < HIDDEN; i++) {
      for (let j = 0; j < STATE_DIM; j++) {
        W1[i][j] += Math.max(-clipVal, Math.min(clipVal, scale * dW1[i][j]));
      }
      b1[i] += Math.max(-clipVal, Math.min(clipVal, scale * db1[i]));
    }
    for (let i = 0; i < ACTION_DIM; i++) {
      for (let j = 0; j < HIDDEN; j++) {
        W2[i][j] += Math.max(-clipVal, Math.min(clipVal, scale * dW2[i][j]));
      }
      b2[i] += Math.max(-clipVal, Math.min(clipVal, scale * db2[i]));
      logStd[i] += Math.max(-clipVal, Math.min(clipVal, scale * 0.1 * dLogStd[i]));
      logStd[i] = Math.max(-2, Math.min(1, logStd[i])); // clamp log_std
    }
  }

  // --- current display walker for animation ---
  let displayWalker = createWalker();
  let displayStep = 0;
  let displayTrajectory = [];

  // --- isometric projection — 3D to 2D ---
  function iso(x3d, y3d, z3d) {
    // isometric: x goes right, y goes up, z goes into screen at 30 degrees
    const isoAngle = Math.PI / 6; // 30 degrees
    const scale = 120;
    const px = (x3d - z3d * Math.cos(isoAngle)) * scale + canvasW * 0.4;
    const py = CANVAS_HEIGHT * 0.75 - y3d * scale + z3d * Math.sin(isoAngle) * scale * 0.3;
    return { x: px, y: py };
  }

  // --- draw stick figure in isometric ---
  function drawWalker(w) {
    const pos = getPositions(w);
    const z = 0; // z = 0, flat projection but with isometric effect

    // torso — thick line
    const p1 = iso(pos.torsoTop.x, pos.torsoTop.y, z);
    const p2 = iso(pos.hip.x, pos.hip.y, z);
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();

    // head
    ctx.fillStyle = ACCENT;
    ctx.beginPath(); ctx.arc(p1.x, p1.y - 8, 6, 0, Math.PI * 2); ctx.fill();

    // left leg — blue-ish
    const lk = iso(pos.lKnee.x, pos.lKnee.y, z - 0.05);
    const lf = iso(pos.lFoot.x, pos.lFoot.y, z - 0.05);
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(p2.x, p2.y); ctx.lineTo(lk.x, lk.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(lk.x, lk.y); ctx.lineTo(lf.x, lf.y); ctx.stroke();
    // knee joint
    ctx.fillStyle = '#60a5fa';
    ctx.beginPath(); ctx.arc(lk.x, lk.y, 3, 0, Math.PI * 2); ctx.fill();
    // foot
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath(); ctx.arc(lf.x, lf.y, 3, 0, Math.PI * 2); ctx.fill();

    // right leg — orange
    const rk = iso(pos.rKnee.x, pos.rKnee.y, z + 0.05);
    const rf = iso(pos.rFoot.x, pos.rFoot.y, z + 0.05);
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(p2.x, p2.y); ctx.lineTo(rk.x, rk.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(rk.x, rk.y); ctx.lineTo(rf.x, rf.y); ctx.stroke();
    ctx.fillStyle = '#fb923c';
    ctx.beginPath(); ctx.arc(rk.x, rk.y, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#f97316';
    ctx.beginPath(); ctx.arc(rf.x, rf.y, 3, 0, Math.PI * 2); ctx.fill();
  }

  // --- reward plot draw karo ---
  function drawRewardPlot() {
    const plotX = canvasW * 0.55;
    const plotY = 20;
    const plotW = canvasW * 0.42;
    const plotH = 150;

    // background
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.fillRect(plotX, plotY, plotW, plotH);
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.strokeRect(plotX, plotY, plotW, plotH);

    // title
    ctx.fillStyle = '#888';
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillText('Episode Reward', plotX + 5, plotY + 12);

    if (rewards.length < 2) return;

    const n = rewards.length;
    const visible = Math.min(n, 200);
    const startIdx = n - visible;
    const slice = rewards.slice(startIdx);

    let minR = Math.min(...slice);
    let maxR = Math.max(...slice);
    if (maxR === minR) { maxR += 1; minR -= 1; }

    // moving average
    const avgWindow = Math.min(20, slice.length);

    // reward curve
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1;
    ctx.beginPath();
    slice.forEach((r, i) => {
      const px = plotX + (i / (visible - 1)) * plotW;
      const py = plotY + plotH - ((r - minR) / (maxR - minR)) * (plotH - 20);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // moving average line
    if (slice.length > avgWindow) {
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = avgWindow; i < slice.length; i++) {
        const avg = slice.slice(i - avgWindow, i).reduce((s, r) => s + r, 0) / avgWindow;
        const px = plotX + (i / (visible - 1)) * plotW;
        const py = plotY + plotH - ((avg - minR) / (maxR - minR)) * (plotH - 20);
        if (i === avgWindow) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // zero line
    if (minR < 0 && maxR > 0) {
      const zeroY = plotY + plotH - ((0 - minR) / (maxR - minR)) * (plotH - 20);
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(plotX, zeroY); ctx.lineTo(plotX + plotW, zeroY); ctx.stroke();
      ctx.setLineDash([]);
    }

    // axis labels
    ctx.fillStyle = '#666';
    ctx.font = "8px 'JetBrains Mono',monospace";
    ctx.textAlign = 'right';
    ctx.fillText(maxR.toFixed(1), plotX - 3, plotY + 12);
    ctx.fillText(minR.toFixed(1), plotX - 3, plotY + plotH);
  }

  // --- ground draw karo — isometric grid ---
  function drawGround() {
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = -5; i <= 10; i++) {
      const p1 = iso(i * 0.3, 0, -1);
      const p2 = iso(i * 0.3, 0, 1);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    }
    for (let j = -3; j <= 3; j++) {
      const p1 = iso(-2, 0, j * 0.3);
      const p2 = iso(3, 0, j * 0.3);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    }
    // ground line — more visible
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 2;
    const gl = iso(-3, 0, 0);
    const gr = iso(5, 0, 0);
    ctx.beginPath(); ctx.moveTo(gl.x, gl.y); ctx.lineTo(gr.x, gr.y); ctx.stroke();
  }

  // --- main draw ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    drawGround();

    // animate current display walker
    if (displayTrajectory.length > 0 && displayStep < displayTrajectory.length) {
      const { state, action } = displayTrajectory[displayStep];
      physicsStep(displayWalker, action);
      displayStep++;
      if (displayStep >= displayTrajectory.length) {
        displayStep = 0;
        displayWalker = createWalker();
      }
    }
    drawWalker(displayWalker);

    // reward plot
    drawRewardPlot();

    // info text
    ctx.fillStyle = '#888';
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillText('REINFORCE Policy Gradient', 10, 20);
    ctx.fillStyle = '#666';
    ctx.fillText('4-DOF Stick Figure Walker', 10, 34);

    // legend
    ctx.fillStyle = '#3b82f6';
    ctx.fillText('\u25CF Left leg', 10, CANVAS_HEIGHT - 30);
    ctx.fillStyle = '#f97316';
    ctx.fillText('\u25CF Right leg', 10, CANVAS_HEIGHT - 16);

    // stats
    const lastReward = rewards.length > 0 ? rewards[rewards.length - 1].toFixed(2) : '-';
    const avg20 = rewards.length > 0 ? (rewards.slice(-20).reduce((s, r) => s + r, 0) / Math.min(20, rewards.length)).toFixed(2) : '-';
    stats.textContent = `Episode: ${episode}  |  Last: ${lastReward}  |  Avg(20): ${avg20}  |  Best: ${bestReward.toFixed(2)}  |  ${paused ? 'PAUSED' : 'Training...'}`;
  }

  // --- animation loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    // training steps
    if (!paused) {
      for (let i = 0; i < trainingSpeed; i++) {
        const result = runEpisode();
        updatePolicy(result.trajectory, result.totalReward);
        rewards.push(result.totalReward);
        if (result.totalReward > bestReward) {
          bestReward = result.totalReward;
          // best episode ko display ke liye save karo
          displayTrajectory = result.trajectory;
          displayWalker = createWalker();
          displayStep = 0;
        }
        episode++;
      }
    }

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
