// ============================================================
// Inverse Kinematics Playground — CCD, Jacobian Transpose, FABRIK
// 2D planar arm with multiple segments, drag target, watch solvers converge
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, IK chalao
export function initIKPlayground() {
  const container = document.getElementById('ikPlaygroundContainer');
  if (!container) {
    console.warn('ikPlaygroundContainer nahi mila bhai, IK playground skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 350;
  const JOINT_RADIUS = 6;
  const BASE_Y_OFFSET = 0.75;      // base canvas ke neeche 75% pe hoga
  const MAX_SOLVER_ITERS = 100;     // solver kitne iterations tak try kare
  const CONVERGENCE_THRESHOLD = 2;  // px mein — itna paas aa gaya toh bas
  const GHOST_MAX = 8;              // kitne ghost arms dikhane hain (intermediate steps)

  // --- State variables ---
  let canvasW = 0, canvasH = 0;
  let animationId = null;
  let isVisible = false;

  // arm configuration
  let numSegments = 3;              // kitne segments hain
  let segmentLength = 60;           // har segment ki length
  let solverMode = 'ccd';           // 'ccd' | 'jacobian' | 'fabrik'
  let showSteps = true;             // intermediate steps dikhane hain kya
  let stepSpeed = 3;                // kitne steps per frame (animation speed)

  // arm state — angles radians mein, base se shuru
  let jointAngles = [];             // [theta0, theta1, theta2, ...]
  let baseX = 0, baseY = 0;

  // target position
  let targetX = 0, targetY = 0;
  let isDragging = false;

  // solver state — step-by-step solve karne ke liye
  let solverActive = false;
  let solverIter = 0;
  let solverGhosts = [];            // [{angles: [...]}, ...] — intermediate states
  let solverResult = '';            // "CCD: converged in 12 iterations"
  let solverConverged = false;

  // --- DOM structure banate hain ---
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(59,130,246,0.2)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls section
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:12px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Button/Slider helper functions ---
  function createButton(label, active, onClick) {
    const btn = document.createElement('button');
    btn.textContent = label;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'font-family:monospace',
      'transition:all 0.2s',
      active
        ? 'background:rgba(59,130,246,0.2);color:#3b82f6;border:1px solid rgba(59,130,246,0.4)'
        : 'background:rgba(255,255,255,0.05);color:#888;border:1px solid rgba(255,255,255,0.1)',
    ].join(';');
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  function createSlider(label, min, max, step, defaultVal, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = 'color:#b0b0b0;font-size:11px;font-weight:600;font-family:monospace;white-space:nowrap;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText = 'width:70px;height:4px;accent-color:rgba(59,130,246,0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText = 'color:#b0b0b0;font-size:11px;min-width:20px;font-family:monospace;';
    valueEl.textContent = defaultVal;
    wrapper.appendChild(valueEl);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = Number.isInteger(val) ? val : val.toFixed(1);
      onChange(val);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, valueEl };
  }

  // solver mode buttons
  const modeButtons = {};
  modeButtons.ccd = createButton('CCD', true, () => setSolverMode('ccd'));
  modeButtons.jacobian = createButton('Jacobian', false, () => setSolverMode('jacobian'));
  modeButtons.fabrik = createButton('FABRIK', false, () => setSolverMode('fabrik'));

  // separator
  const sep1 = document.createElement('div');
  sep1.style.cssText = 'width:1px;height:20px;background:rgba(255,255,255,0.1);';
  controlsDiv.appendChild(sep1);

  // sliders
  const segCountSlider = createSlider('Joints', 2, 5, 1, numSegments, (v) => {
    numSegments = Math.round(v);
    initArm();
    triggerSolve();
  });

  const segLenSlider = createSlider('Length', 30, 100, 5, segmentLength, (v) => {
    segmentLength = v;
    triggerSolve();
  });

  const speedSlider = createSlider('Speed', 1, 10, 1, stepSpeed, (v) => {
    stepSpeed = Math.round(v);
  });

  // separator
  const sep2 = document.createElement('div');
  sep2.style.cssText = 'width:1px;height:20px;background:rgba(255,255,255,0.1);';
  controlsDiv.appendChild(sep2);

  // show steps toggle
  const stepsBtn = createButton('Steps: ON', true, () => {
    showSteps = !showSteps;
    stepsBtn.textContent = showSteps ? 'Steps: ON' : 'Steps: OFF';
    stepsBtn.style.background = showSteps ? 'rgba(59,130,246,0.2)' : 'rgba(255,255,255,0.05)';
    stepsBtn.style.color = showSteps ? '#3b82f6' : '#888';
    stepsBtn.style.borderColor = showSteps ? 'rgba(59,130,246,0.4)' : 'rgba(255,255,255,0.1)';
  });

  // result display — solver convergence info
  const resultDiv = document.createElement('div');
  resultDiv.style.cssText = 'font-family:monospace;font-size:11px;color:#888;margin-top:4px;width:100%;';
  container.appendChild(resultDiv);

  // --- Solver mode switch ---
  function setSolverMode(newMode) {
    solverMode = newMode;
    Object.keys(modeButtons).forEach(key => {
      const isActive = key === newMode;
      modeButtons[key].style.background = isActive ? 'rgba(59,130,246,0.2)' : 'rgba(255,255,255,0.05)';
      modeButtons[key].style.color = isActive ? '#3b82f6' : '#888';
      modeButtons[key].style.borderColor = isActive ? 'rgba(59,130,246,0.4)' : 'rgba(255,255,255,0.1)';
    });
    triggerSolve();
  }

  // --- Canvas resize ---
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvasW = rect.width;
    canvasH = rect.height;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // base position update kar
    baseX = canvasW / 2;
    baseY = canvasH * BASE_Y_OFFSET;
  }

  // --- Arm initialization ---
  function initArm() {
    jointAngles = [];
    for (let i = 0; i < numSegments; i++) {
      // sab upar ki taraf point kare initially — -PI/2 per segment
      jointAngles.push(-Math.PI / 2 / numSegments);
    }
  }

  // --- Forward kinematics — angles se end-effector position nikaal ---
  function forwardKinematics(angles, base_x, base_y, segLen) {
    const joints = [{ x: base_x, y: base_y }];
    let cumAngle = 0;
    for (let i = 0; i < angles.length; i++) {
      cumAngle += angles[i];
      const prev = joints[joints.length - 1];
      joints.push({
        x: prev.x + segLen * Math.cos(cumAngle),
        y: prev.y + segLen * Math.sin(cumAngle),
      });
    }
    return joints;
  }

  // end-effector position shortcut
  function getEndEffector(angles) {
    const joints = forwardKinematics(angles, baseX, baseY, segmentLength);
    return joints[joints.length - 1];
  }

  // total workspace radius — saare segments straight kare toh kitna door pahunche
  function getWorkspaceRadius() {
    return numSegments * segmentLength;
  }

  // --- CCD Solver — Cyclic Coordinate Descent ---
  // ek ek joint rotate kar, end-effector ko target ke paas laa
  function solveCCD_step(angles, tx, ty) {
    const newAngles = angles.slice();

    // last joint se start kar, peeche jaa — CCD ka standard order
    for (let i = newAngles.length - 1; i >= 0; i--) {
      // is joint ki position nikaal
      const joints = forwardKinematics(newAngles, baseX, baseY, segmentLength);
      const joint = joints[i];
      const end = joints[joints.length - 1];

      // joint se end-effector ka angle
      const toEnd = Math.atan2(end.y - joint.y, end.x - joint.x);
      // joint se target ka angle
      const toTarget = Math.atan2(ty - joint.y, tx - joint.x);

      // fark nikaal — itna rotate karna hai
      let delta = toTarget - toEnd;
      // wrap to [-PI, PI]
      while (delta > Math.PI) delta -= 2 * Math.PI;
      while (delta < -Math.PI) delta += 2 * Math.PI;

      // angle update kar
      newAngles[i] += delta;
    }

    return newAngles;
  }

  // --- Jacobian Transpose Solver ---
  // Jacobian matrix bana, transpose multiply kar, gradient direction mein angles update kar
  function solveJacobian_step(angles, tx, ty) {
    const newAngles = angles.slice();
    const joints = forwardKinematics(newAngles, baseX, baseY, segmentLength);
    const end = joints[joints.length - 1];

    // error — target minus current end-effector
    const ex = tx - end.x;
    const ey = ty - end.y;

    // Jacobian transpose method — J^T * e gives angle update direction
    // J[i] = [-sin(cumAngle_i_to_end) * remaining_length, cos(cumAngle_i_to_end) * remaining_length]
    // simplified: partial derivatives of end-effector w.r.t. each joint angle
    const dTheta = [];
    for (let i = 0; i < newAngles.length; i++) {
      // is joint se end-effector tak ka vector
      const jx = end.x - joints[i].x;
      const jy = end.y - joints[i].y;

      // cross product component — 2D mein z-axis rotation hai
      // J_i = [-jy, jx] (2D revolute joint ka Jacobian column)
      // J^T * e = -jy * ex + jx * ey
      const jte = -jy * ex + jx * ey;

      dTheta.push(jte);
    }

    // step size — adaptive, error magnitude pe depend kare
    const errMag = Math.sqrt(ex * ex + ey * ey);
    const alpha = 0.00005 * Math.min(errMag, 100);

    for (let i = 0; i < newAngles.length; i++) {
      newAngles[i] += alpha * dTheta[i];
    }

    return newAngles;
  }

  // --- FABRIK Solver — Forward And Backward Reaching IK ---
  // positions pe kaam karta hai, angles pe nahi — bahut intuitive method
  function solveFABRIK_step(angles, tx, ty) {
    // pehle current joint positions nikaal
    let joints = forwardKinematics(angles, baseX, baseY, segmentLength);
    const n = joints.length; // n = numSegments + 1

    // positions extract kar
    let positions = joints.map(j => ({ x: j.x, y: j.y }));

    // FORWARD pass — end-effector ko target pe le jaa, baaki pull karo
    positions[n - 1] = { x: tx, y: ty };
    for (let i = n - 2; i >= 0; i--) {
      const dx = positions[i].x - positions[i + 1].x;
      const dy = positions[i].y - positions[i + 1].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 0.001) continue;
      const ratio = segmentLength / dist;
      positions[i] = {
        x: positions[i + 1].x + dx * ratio,
        y: positions[i + 1].y + dy * ratio,
      };
    }

    // BACKWARD pass — base ko wapas fix position pe le jaa, baaki push karo
    positions[0] = { x: baseX, y: baseY };
    for (let i = 0; i < n - 1; i++) {
      const dx = positions[i + 1].x - positions[i].x;
      const dy = positions[i + 1].y - positions[i].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 0.001) continue;
      const ratio = segmentLength / dist;
      positions[i + 1] = {
        x: positions[i].x + dx * ratio,
        y: positions[i].y + dy * ratio,
      };
    }

    // positions se angles reconstruct kar
    const newAngles = [];
    let cumAngle = 0;
    for (let i = 0; i < numSegments; i++) {
      const dx = positions[i + 1].x - positions[i].x;
      const dy = positions[i + 1].y - positions[i].y;
      const absAngle = Math.atan2(dy, dx);
      newAngles.push(absAngle - cumAngle);
      cumAngle = absAngle;
    }

    return newAngles;
  }

  // --- Solver trigger — target change hone pe call kar ---
  function triggerSolve() {
    solverActive = true;
    solverIter = 0;
    solverGhosts = [];
    solverConverged = false;
    solverResult = '';
    // starting angles store kar first ghost ke liye
    solverGhosts.push({ angles: jointAngles.slice() });
  }

  // solver ka ek step execute kar — animate karne ke liye
  function solverStep() {
    if (!solverActive) return;

    // check workspace limit — target reachable hai ya nahi
    const dx = targetX - baseX;
    const dy = targetY - baseY;
    const targetDist = Math.sqrt(dx * dx + dy * dy);
    const maxReach = getWorkspaceRadius();

    // effective target — agar unreachable hai toh closest reachable point use kar
    let effTx = targetX;
    let effTy = targetY;
    if (targetDist > maxReach) {
      effTx = baseX + (dx / targetDist) * (maxReach - 1);
      effTy = baseY + (dy / targetDist) * (maxReach - 1);
    }

    for (let s = 0; s < stepSpeed; s++) {
      if (solverConverged || solverIter >= MAX_SOLVER_ITERS) {
        solverActive = false;
        return;
      }

      // solver step based on mode
      let newAngles;
      if (solverMode === 'ccd') {
        newAngles = solveCCD_step(jointAngles, effTx, effTy);
      } else if (solverMode === 'jacobian') {
        newAngles = solveJacobian_step(jointAngles, effTx, effTy);
      } else {
        newAngles = solveFABRIK_step(jointAngles, effTx, effTy);
      }

      jointAngles = newAngles;
      solverIter++;

      // ghost store kar — limited count
      if (showSteps && solverGhosts.length < GHOST_MAX) {
        // har kuch iterations pe ek ghost save kar
        const interval = Math.max(1, Math.floor(MAX_SOLVER_ITERS / GHOST_MAX));
        if (solverIter % interval === 0) {
          solverGhosts.push({ angles: jointAngles.slice() });
        }
      }

      // convergence check — end-effector target ke kitna paas hai
      const end = getEndEffector(jointAngles);
      const err = Math.sqrt((end.x - effTx) ** 2 + (end.y - effTy) ** 2);
      if (err < CONVERGENCE_THRESHOLD) {
        solverConverged = true;
        const modeNames = { ccd: 'CCD', jacobian: 'Jacobian', fabrik: 'FABRIK' };
        solverResult = modeNames[solverMode] + ': converged in ' + solverIter + ' iterations (err: ' + err.toFixed(1) + 'px)';
        solverActive = false;
        return;
      }
    }

    // iteration limit pe pahunch gaye bina converge kiye
    if (solverIter >= MAX_SOLVER_ITERS) {
      const end = getEndEffector(jointAngles);
      const err = Math.sqrt((end.x - targetX) ** 2 + (end.y - targetY) ** 2);
      const modeNames = { ccd: 'CCD', jacobian: 'Jacobian', fabrik: 'FABRIK' };
      solverResult = modeNames[solverMode] + ': ' + solverIter + ' iterations, err: ' + err.toFixed(1) + 'px';
      solverActive = false;
    }
  }

  // --- Mouse/Touch events ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  canvas.addEventListener('mousedown', (e) => {
    const pos = getCanvasPos(e);
    targetX = pos.x;
    targetY = pos.y;
    isDragging = true;
    triggerSolve();
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const pos = getCanvasPos(e);
    targetX = pos.x;
    targetY = pos.y;
    triggerSolve();
  });

  canvas.addEventListener('mouseup', () => { isDragging = false; });
  canvas.addEventListener('mouseleave', () => { isDragging = false; });

  // touch support
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    targetX = pos.x;
    targetY = pos.y;
    isDragging = true;
    triggerSolve();
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDragging) return;
    const pos = getCanvasPos(e);
    targetX = pos.x;
    targetY = pos.y;
    triggerSolve();
  }, { passive: false });

  canvas.addEventListener('touchend', () => { isDragging = false; });

  // --- Drawing functions ---

  // arm draw kar — joints + segments
  function drawArm(angles, alpha, color, lineWidth) {
    const joints = forwardKinematics(angles, baseX, baseY, segmentLength);

    // segments — gradient wali line
    for (let i = 0; i < joints.length - 1; i++) {
      const from = joints[i];
      const to = joints[i + 1];

      // segment line
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = color.replace('ALPHA', alpha.toFixed(2));
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    // joints — circles
    for (let i = 0; i < joints.length; i++) {
      ctx.beginPath();
      ctx.arc(joints[i].x, joints[i].y, i === 0 ? JOINT_RADIUS + 2 : JOINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = color.replace('ALPHA', (alpha * 0.7).toFixed(2));
      ctx.fill();
      ctx.strokeStyle = color.replace('ALPHA', alpha.toFixed(2));
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // joint angles ka arc indicator draw kar
  function drawJointArcs(angles) {
    const joints = forwardKinematics(angles, baseX, baseY, segmentLength);
    let cumAngle = 0;

    for (let i = 0; i < angles.length; i++) {
      const joint = joints[i];
      const prevAngle = cumAngle;
      cumAngle += angles[i];

      // incoming direction — pehle joint ke liye vertical reference
      const refAngle = i === 0 ? -Math.PI / 2 : prevAngle;

      // arc draw kar — joint ke around
      const arcRadius = 15;
      const startA = Math.min(refAngle, refAngle + angles[i]);
      const endA = Math.max(refAngle, refAngle + angles[i]);

      // sirf significant angles dikhao
      if (Math.abs(angles[i]) < 0.05) continue;

      ctx.beginPath();
      ctx.arc(joint.x, joint.y, arcRadius, startA, endA);
      ctx.strokeStyle = 'rgba(251,191,36,0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // angle value text — agar jagah ho toh
      const midAngle = (startA + endA) / 2;
      const textR = arcRadius + 10;
      const textX = joint.x + textR * Math.cos(midAngle);
      const textY = joint.y + textR * Math.sin(midAngle);
      const degVal = Math.round(angles[i] * 180 / Math.PI);
      ctx.fillStyle = 'rgba(251,191,36,0.5)';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(degVal + '°', textX, textY);
    }
  }

  // workspace boundary draw kar — maximum reach circle
  function drawWorkspaceBoundary() {
    const radius = getWorkspaceRadius();
    const dx = targetX - baseX;
    const dy = targetY - baseY;
    const targetDist = Math.sqrt(dx * dx + dy * dy);

    // sirf dikhao jab target unreachable ho
    const isUnreachable = targetDist > radius;

    ctx.beginPath();
    ctx.arc(baseX, baseY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = isUnreachable ? 'rgba(239,68,68,0.3)' : 'rgba(59,130,246,0.1)';
    ctx.lineWidth = isUnreachable ? 1.5 : 1;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    if (isUnreachable) {
      // "Unreachable" label
      ctx.fillStyle = 'rgba(239,68,68,0.5)';
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Unreachable', baseX, baseY - radius - 10);
    }
  }

  // target dot draw kar
  function drawTarget() {
    // target glow
    ctx.beginPath();
    ctx.arc(targetX, targetY, 10, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16,185,129,0.15)';
    ctx.fill();

    // target dot
    ctx.beginPath();
    ctx.arc(targetX, targetY, 5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16,185,129,0.9)';
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // crosshair lines
    ctx.strokeStyle = 'rgba(16,185,129,0.3)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(targetX - 12, targetY);
    ctx.lineTo(targetX + 12, targetY);
    ctx.moveTo(targetX, targetY - 12);
    ctx.lineTo(targetX, targetY + 12);
    ctx.stroke();
  }

  // base mount draw kar — jahan arm fixed hai
  function drawBase() {
    // base platform — chhoti rectangle
    ctx.fillStyle = 'rgba(255,255,255,0.1)';
    ctx.fillRect(baseX - 20, baseY, 40, 8);
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    ctx.strokeRect(baseX - 20, baseY, 40, 8);

    // ground line
    ctx.beginPath();
    ctx.moveTo(baseX - 35, baseY + 8);
    ctx.lineTo(baseX + 35, baseY + 8);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // hatching — ground dikhane ke liye
    for (let i = -30; i <= 30; i += 8) {
      ctx.beginPath();
      ctx.moveTo(baseX + i, baseY + 8);
      ctx.lineTo(baseX + i - 6, baseY + 14);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.stroke();
    }
  }

  // --- Main render ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // workspace boundary
    drawWorkspaceBoundary();

    // base mount
    drawBase();

    // ghost arms — intermediate solver steps
    if (showSteps && solverGhosts.length > 1) {
      for (let i = 0; i < solverGhosts.length - 1; i++) {
        const alpha = 0.08 + 0.12 * (i / solverGhosts.length);
        drawArm(solverGhosts[i].angles, alpha, 'rgba(59,130,246,ALPHA)', 2);
      }
    }

    // main arm — current state
    drawArm(jointAngles, 0.85, 'rgba(59,130,246,ALPHA)', 4);

    // joint angle arcs
    drawJointArcs(jointAngles);

    // target
    drawTarget();

    // result text update kar
    resultDiv.textContent = solverResult;
    resultDiv.style.color = solverConverged ? '#10b981' : '#f59e0b';

    // instruction text jab kuch nahi hua hai
    if (!solverResult && !solverActive) {
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.font = '13px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Click anywhere to set target', canvasW / 2, 30);
    }
  }

  // --- Main animation loop ---
  function animate() {
    animationId = requestAnimationFrame(animate);
    if (!isVisible) return;

    // solver step execute kar agar active hai
    if (solverActive) {
      solverStep();
    }

    draw();
  }

  // --- IntersectionObserver — sirf visible hone pe animate kar ---
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      isVisible = entry.isIntersecting;
      if (isVisible && !animationId) {
        resizeCanvas();
        animate();
      }
    });
  }, { threshold: 0.1 });
  observer.observe(container);

  // resize listener
  window.addEventListener('resize', () => {
    if (isVisible) {
      resizeCanvas();
    }
  });

  // --- Initial setup ---
  resizeCanvas();
  initArm();
  // default target — base ke upar thoda left
  targetX = baseX - segmentLength * 1.5;
  targetY = baseY - segmentLength * 2;
  triggerSolve();
  animate();
}
