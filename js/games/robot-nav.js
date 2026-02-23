// ============================================================
// 2D Robot Navigator — RRT path planning + Potential Field mode
// Obstacle draw karo, start/goal set karo, path plan karo, robot follow kare
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, navigation chalao
export function initRobotNav() {
  const container = document.getElementById('robotNavContainer');
  if (!container) {
    console.warn('robotNavContainer nahi mila bhai, robot nav skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 350;
  const ROBOT_RADIUS = 12;
  const RRT_MAX_ITER = 2000;        // zyada iterations = better path milne ka chance
  const RRT_STEP_SIZE = 20;         // har step mein kitna aage badhe
  const RRT_GOAL_BIAS = 0.15;       // 15% chance seedha goal ki taraf jaaye
  const ROBOT_MAX_SPEED = 2.5;      // pixels per frame — zyada tez kiya toh overshoot karega
  const ROBOT_TURN_RATE = 0.08;     // radians per frame — smooth turning ke liye
  const POTENTIAL_RESOLUTION = 8;   // heatmap grid ka resolution — chhota = zyada detail but slow
  const REPULSIVE_RANGE = 60;       // obstacle se kitni door tak repulsive force lage
  const REPULSIVE_STRENGTH = 5000;  // kitni tezi se bhaage obstacle se
  const ATTRACTIVE_STRENGTH = 0.5;  // goal ki taraf kitna attract ho

  // --- State variables ---
  let canvasW = 0, canvasH = 0;
  let obstacles = [];               // [{x, y, w, h}, ...]
  let startPos = null;              // {x, y}
  let goalPos = null;               // {x, y}
  let rrtTree = [];                 // [{x, y, parent}, ...] — parent index hai
  let rrtPath = [];                 // [{x, y}, ...] — final path
  let isPlanning = false;           // RRT chal raha hai kya
  let planningStep = 0;             // current RRT iteration for animation
  let mode = 'obstacles';           // 'obstacles' | 'start' | 'goal'
  let planMode = 'rrt';             // 'rrt' | 'potential'
  let robot = null;                 // {x, y, theta, pathIndex}
  let isFollowing = false;          // robot path follow kar raha hai kya
  let potentialField = null;        // heatmap cache
  let animationId = null;
  let isVisible = false;

  // obstacle drawing ke liye temp state
  let drawStart = null;             // {x, y} — jahan se draw shuru kiya

  // --- DOM structure banate hain ---
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — sab yahan draw hoga
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(16,185,129,0.2)',
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
    'gap:8px',
    'margin-top:12px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Button banane ka helper ---
  function createButton(label, active, onClick) {
    const btn = document.createElement('button');
    btn.textContent = label;
    btn.style.cssText = [
      'padding:5px 14px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'font-family:monospace',
      'transition:all 0.2s',
      active
        ? 'background:rgba(16,185,129,0.2);color:#10b981;border:1px solid rgba(16,185,129,0.4)'
        : 'background:rgba(255,255,255,0.05);color:#888;border:1px solid rgba(255,255,255,0.1)',
    ].join(';');
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // mode buttons — Draw Obstacles, Set Start, Set Goal
  const modeButtons = {};
  modeButtons.obstacles = createButton('Draw Obstacles', true, () => setMode('obstacles'));
  modeButtons.start = createButton('Set Start', false, () => setMode('start'));
  modeButtons.goal = createButton('Set Goal', false, () => setMode('goal'));

  // separator
  const sep1 = document.createElement('div');
  sep1.style.cssText = 'width:1px;height:20px;background:rgba(255,255,255,0.1);';
  controlsDiv.appendChild(sep1);

  // action buttons
  const planBtn = createButton('Plan Path', false, () => startPlanning());
  const followBtn = createButton('Follow', false, () => startFollowing());
  const clearBtn = createButton('Clear', false, () => clearAll());

  // separator
  const sep2 = document.createElement('div');
  sep2.style.cssText = 'width:1px;height:20px;background:rgba(255,255,255,0.1);';
  controlsDiv.appendChild(sep2);

  // RRT vs Potential Field toggle
  const toggleBtn = createButton('Mode: RRT', false, () => togglePlanMode());

  // --- Mode switching ---
  function setMode(newMode) {
    mode = newMode;
    // buttons ka style update kar
    Object.keys(modeButtons).forEach(key => {
      const isActive = key === newMode;
      modeButtons[key].style.background = isActive ? 'rgba(16,185,129,0.2)' : 'rgba(255,255,255,0.05)';
      modeButtons[key].style.color = isActive ? '#10b981' : '#888';
      modeButtons[key].style.borderColor = isActive ? 'rgba(16,185,129,0.4)' : 'rgba(255,255,255,0.1)';
    });
  }

  // RRT / Potential Field toggle
  function togglePlanMode() {
    planMode = planMode === 'rrt' ? 'potential' : 'rrt';
    toggleBtn.textContent = planMode === 'rrt' ? 'Mode: RRT' : 'Mode: Potential';
    // potential field cache invalidate kar
    potentialField = null;
    rrtTree = [];
    rrtPath = [];
    isPlanning = false;
    isFollowing = false;
    robot = null;
  }

  // --- Canvas resize handler ---
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvasW = rect.width;
    canvasH = rect.height;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // potential field recalculate karna padega resize pe
    potentialField = null;
  }

  // --- Mouse/touch event helpers ---
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  }

  // --- Canvas events ---
  canvas.addEventListener('mousedown', (e) => {
    const pos = getCanvasPos(e);
    if (mode === 'obstacles') {
      // obstacle draw shuru — click se drag karke rectangle banayega
      drawStart = pos;
    } else if (mode === 'start') {
      startPos = pos;
      // start set kiya toh purana plan clear kar
      rrtTree = [];
      rrtPath = [];
      isFollowing = false;
      robot = null;
      potentialField = null;
    } else if (mode === 'goal') {
      goalPos = pos;
      // goal set kiya toh purana plan clear kar
      rrtTree = [];
      rrtPath = [];
      isFollowing = false;
      robot = null;
      potentialField = null;
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    // obstacle draw ho raha hai toh preview dikhao — drawing handled in render
    if (mode === 'obstacles' && drawStart) {
      // state update nahi chahiye — render mein current mouse position se preview banega
      canvas._currentMouse = getCanvasPos(e);
    }
  });

  canvas.addEventListener('mouseup', (e) => {
    if (mode === 'obstacles' && drawStart) {
      const pos = getCanvasPos(e);
      const x = Math.min(drawStart.x, pos.x);
      const y = Math.min(drawStart.y, pos.y);
      const w = Math.abs(pos.x - drawStart.x);
      const h = Math.abs(pos.y - drawStart.y);
      // chhote rectangles ignore kar — accidental click hoga
      if (w > 5 && h > 5) {
        obstacles.push({ x, y, w, h });
        potentialField = null; // field recalculate karana padega
      }
      drawStart = null;
      canvas._currentMouse = null;
    }
  });

  // touch support — mobile pe bhi kaam kare
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    if (mode === 'obstacles') {
      drawStart = pos;
    } else if (mode === 'start') {
      startPos = pos;
      rrtTree = []; rrtPath = []; isFollowing = false; robot = null; potentialField = null;
    } else if (mode === 'goal') {
      goalPos = pos;
      rrtTree = []; rrtPath = []; isFollowing = false; robot = null; potentialField = null;
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (mode === 'obstacles' && drawStart) {
      canvas._currentMouse = getCanvasPos(e);
    }
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    if (mode === 'obstacles' && drawStart && canvas._currentMouse) {
      const pos = canvas._currentMouse;
      const x = Math.min(drawStart.x, pos.x);
      const y = Math.min(drawStart.y, pos.y);
      const w = Math.abs(pos.x - drawStart.x);
      const h = Math.abs(pos.y - drawStart.y);
      if (w > 5 && h > 5) {
        obstacles.push({ x, y, w, h });
        potentialField = null;
      }
      drawStart = null;
      canvas._currentMouse = null;
    }
  });

  // --- Collision detection ---
  // point obstacle ke andar hai ya nahi — robot radius bhi check kar
  function isColliding(px, py, radius) {
    for (const obs of obstacles) {
      // rectangle se circle collision — closest point dhundho
      const cx = Math.max(obs.x, Math.min(px, obs.x + obs.w));
      const cy = Math.max(obs.y, Math.min(py, obs.y + obs.h));
      const dx = px - cx;
      const dy = py - cy;
      if (dx * dx + dy * dy < radius * radius) return true;
    }
    // canvas boundary bhi check kar
    if (px - radius < 0 || px + radius > canvasW || py - radius < 0 || py + radius > canvasH) {
      return true;
    }
    return false;
  }

  // line segment obstacle se cross toh nahi kar raha — RRT edge validation ke liye
  function isEdgeFree(x1, y1, x2, y2) {
    const steps = Math.ceil(Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 3);
    for (let i = 0; i <= steps; i++) {
      const t = i / Math.max(steps, 1);
      const px = x1 + (x2 - x1) * t;
      const py = y1 + (y2 - y1) * t;
      if (isColliding(px, py, ROBOT_RADIUS)) return false;
    }
    return true;
  }

  // --- RRT Path Planning ---
  // RRT tree grow kar raha hai — random sample le aur nearest node se connect kar
  function rrtStep() {
    if (!startPos || !goalPos) return false;

    // goal bias — kabhi kabhi seedha goal ki taraf try kar
    let sample;
    if (Math.random() < RRT_GOAL_BIAS) {
      sample = { x: goalPos.x, y: goalPos.y };
    } else {
      sample = { x: Math.random() * canvasW, y: Math.random() * canvasH };
    }

    // nearest node dhundho tree mein
    let nearestIdx = 0;
    let nearestDist = Infinity;
    for (let i = 0; i < rrtTree.length; i++) {
      const dx = rrtTree[i].x - sample.x;
      const dy = rrtTree[i].y - sample.y;
      const dist = dx * dx + dy * dy;
      if (dist < nearestDist) {
        nearestDist = dist;
        nearestIdx = i;
      }
    }

    const nearest = rrtTree[nearestIdx];
    const dx = sample.x - nearest.x;
    const dy = sample.y - nearest.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    // step size limit kar — zyada door ka sample ho toh step size tak hi jaa
    const stepDist = Math.min(dist, RRT_STEP_SIZE);
    const newX = nearest.x + (dx / dist) * stepDist;
    const newY = nearest.y + (dy / dist) * stepDist;

    // collision check — edge free hai toh node add kar
    if (!isColliding(newX, newY, ROBOT_RADIUS) && isEdgeFree(nearest.x, nearest.y, newX, newY)) {
      rrtTree.push({ x: newX, y: newY, parent: nearestIdx });

      // goal ke paas pahunch gaye? path mil gaya!
      const gx = goalPos.x - newX;
      const gy = goalPos.y - newY;
      if (Math.sqrt(gx * gx + gy * gy) < RRT_STEP_SIZE * 1.5) {
        // goal ko bhi tree mein add kar
        if (isEdgeFree(newX, newY, goalPos.x, goalPos.y)) {
          rrtTree.push({ x: goalPos.x, y: goalPos.y, parent: rrtTree.length - 1 });
          // path extract kar — goal se start tak backtrack
          extractPath();
          return true; // path mil gaya
        }
      }
    }
    return false; // abhi nahi mila, aur try karo
  }

  // RRT tree se final path nikaal — backtrack from goal to start
  function extractPath() {
    rrtPath = [];
    let idx = rrtTree.length - 1;
    while (idx >= 0) {
      rrtPath.unshift({ x: rrtTree[idx].x, y: rrtTree[idx].y });
      idx = rrtTree[idx].parent !== undefined ? rrtTree[idx].parent : -1;
    }
  }

  // planning shuru kar — tree initialize karke step-by-step grow karega
  function startPlanning() {
    if (!startPos || !goalPos) return;
    rrtTree = [{ x: startPos.x, y: startPos.y, parent: undefined }];
    rrtPath = [];
    isPlanning = true;
    planningStep = 0;
    isFollowing = false;
    robot = null;

    if (planMode === 'potential') {
      // potential field mode mein planning nahi — seedha field calculate kar
      computePotentialField();
      isPlanning = false;
      // potential field mein "path" nahi hota — robot gradient follow karega
      rrtPath = [goalPos]; // dummy taaki follow button kaam kare
    }
  }

  // --- Potential Field computation ---
  // attractive + repulsive field ka heatmap banao
  function computePotentialField() {
    if (!goalPos) return;
    const cols = Math.ceil(canvasW / POTENTIAL_RESOLUTION);
    const rows = Math.ceil(canvasH / POTENTIAL_RESOLUTION);
    potentialField = {
      data: new Float32Array(cols * rows),
      cols, rows,
      maxVal: 0
    };

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const px = c * POTENTIAL_RESOLUTION + POTENTIAL_RESOLUTION / 2;
        const py = r * POTENTIAL_RESOLUTION + POTENTIAL_RESOLUTION / 2;

        // attractive potential — goal ki taraf quadratic
        const dgx = px - goalPos.x;
        const dgy = py - goalPos.y;
        const attractive = ATTRACTIVE_STRENGTH * (dgx * dgx + dgy * dgy);

        // repulsive potential — har obstacle se
        let repulsive = 0;
        for (const obs of obstacles) {
          // obstacle ke closest point se distance
          const cx = Math.max(obs.x, Math.min(px, obs.x + obs.w));
          const cy = Math.max(obs.y, Math.min(py, obs.y + obs.h));
          const ddx = px - cx;
          const ddy = py - cy;
          const d = Math.sqrt(ddx * ddx + ddy * ddy);
          if (d < REPULSIVE_RANGE && d > 0.1) {
            // inverse square — paas aao toh bahut zyada force
            repulsive += REPULSIVE_STRENGTH * (1 / d - 1 / REPULSIVE_RANGE) ** 2;
          } else if (d <= 0.1) {
            repulsive += REPULSIVE_STRENGTH * 100; // andar hai toh max repulsion
          }
        }

        const total = attractive + repulsive;
        potentialField.data[r * cols + c] = total;
        if (total > potentialField.maxVal) potentialField.maxVal = total;
      }
    }
  }

  // potential field ka gradient nikaal — robot isko follow karega
  function getPotentialGradient(px, py) {
    if (!potentialField) return { gx: 0, gy: 0 };
    const { data, cols, rows } = potentialField;
    const c = Math.floor(px / POTENTIAL_RESOLUTION);
    const r = Math.floor(py / POTENTIAL_RESOLUTION);

    // boundary check
    if (c <= 0 || c >= cols - 1 || r <= 0 || r >= rows - 1) return { gx: 0, gy: 0 };

    // central difference se gradient approximate kar
    const dUdx = (data[r * cols + (c + 1)] - data[r * cols + (c - 1)]) / (2 * POTENTIAL_RESOLUTION);
    const dUdy = (data[(r + 1) * cols + c] - data[(r - 1) * cols + c]) / (2 * POTENTIAL_RESOLUTION);

    return { gx: -dUdx, gy: -dUdy }; // negative gradient — potential kam hone ki taraf jaa
  }

  // --- Robot following ---
  function startFollowing() {
    if (!startPos) return;
    if (planMode === 'rrt' && rrtPath.length === 0) return;
    if (planMode === 'potential' && !potentialField) return;

    robot = {
      x: startPos.x,
      y: startPos.y,
      theta: 0, // heading angle
      pathIndex: 0
    };
    isFollowing = true;
  }

  // robot ko ek step aage badhao — differential drive kinematics use kar
  function updateRobot() {
    if (!robot || !isFollowing) return;

    if (planMode === 'rrt') {
      // RRT path follow kar — next waypoint ki taraf jaa
      if (robot.pathIndex >= rrtPath.length) {
        isFollowing = false;
        return;
      }

      const target = rrtPath[robot.pathIndex];
      const dx = target.x - robot.x;
      const dy = target.y - robot.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      // waypoint ke paas pahunch gaye toh next pe jaa
      if (dist < ROBOT_RADIUS) {
        robot.pathIndex++;
        return;
      }

      // desired heading calculate kar
      const desiredTheta = Math.atan2(dy, dx);

      // smooth turning — ek dum se nahi mude, dheere dheere
      let angleDiff = desiredTheta - robot.theta;
      // angle wrap karo -PI to PI mein
      while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
      while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

      robot.theta += Math.sign(angleDiff) * Math.min(Math.abs(angleDiff), ROBOT_TURN_RATE);

      // differential drive — forward direction mein move kar
      const speed = Math.min(ROBOT_MAX_SPEED, dist * 0.1);
      robot.x += Math.cos(robot.theta) * speed;
      robot.y += Math.sin(robot.theta) * speed;

    } else {
      // Potential field mode — gradient follow kar
      const grad = getPotentialGradient(robot.x, robot.y);
      const gmag = Math.sqrt(grad.gx * grad.gx + grad.gy * grad.gy);

      if (gmag < 0.001) {
        // local minimum mein fas gaya ya goal pe pahunch gaya
        const dgx = goalPos.x - robot.x;
        const dgy = goalPos.y - robot.y;
        if (Math.sqrt(dgx * dgx + dgy * dgy) < ROBOT_RADIUS * 2) {
          isFollowing = false; // goal reached!
        }
        return;
      }

      // gradient direction mein jaa
      const desiredTheta = Math.atan2(grad.gy, grad.gx);
      let angleDiff = desiredTheta - robot.theta;
      while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
      while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

      robot.theta += Math.sign(angleDiff) * Math.min(Math.abs(angleDiff), ROBOT_TURN_RATE);

      const speed = Math.min(ROBOT_MAX_SPEED, gmag * 2);
      const newX = robot.x + Math.cos(robot.theta) * speed;
      const newY = robot.y + Math.sin(robot.theta) * speed;

      // collision check — obstacle mein ghusne mat do
      if (!isColliding(newX, newY, ROBOT_RADIUS)) {
        robot.x = newX;
        robot.y = newY;
      }

      // goal pe pahunch gaye?
      const dgx = goalPos.x - robot.x;
      const dgy = goalPos.y - robot.y;
      if (Math.sqrt(dgx * dgx + dgy * dgy) < ROBOT_RADIUS * 2) {
        isFollowing = false;
      }
    }
  }

  // --- Clear everything ---
  function clearAll() {
    obstacles = [];
    startPos = null;
    goalPos = null;
    rrtTree = [];
    rrtPath = [];
    isPlanning = false;
    isFollowing = false;
    robot = null;
    potentialField = null;
    drawStart = null;
    canvas._currentMouse = null;
  }

  // --- Rendering ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // potential field heatmap draw kar — agar potential mode hai aur field calculated hai
    if (planMode === 'potential' && potentialField) {
      drawPotentialHeatmap();
    }

    // obstacles draw kar
    ctx.fillStyle = 'rgba(255,255,255,0.12)';
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    for (const obs of obstacles) {
      ctx.fillRect(obs.x, obs.y, obs.w, obs.h);
      ctx.strokeRect(obs.x, obs.y, obs.w, obs.h);
    }

    // drawing preview — abhi draw ho raha hai wala rectangle
    if (drawStart && canvas._currentMouse) {
      const m = canvas._currentMouse;
      const x = Math.min(drawStart.x, m.x);
      const y = Math.min(drawStart.y, m.y);
      const w = Math.abs(m.x - drawStart.x);
      const h = Math.abs(m.y - drawStart.y);
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.setLineDash([4, 4]);
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }

    // start position — green dot
    if (startPos) {
      ctx.beginPath();
      ctx.arc(startPos.x, startPos.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(16,185,129,0.8)';
      ctx.fill();
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.stroke();
      // "S" label
      ctx.fillStyle = '#fff';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('S', startPos.x, startPos.y);
    }

    // goal position — red dot
    if (goalPos) {
      ctx.beginPath();
      ctx.arc(goalPos.x, goalPos.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239,68,68,0.8)';
      ctx.fill();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.stroke();
      // "G" label
      ctx.fillStyle = '#fff';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('G', goalPos.x, goalPos.y);
    }

    // RRT tree draw kar — thin cyan lines
    if (planMode === 'rrt' && rrtTree.length > 1) {
      ctx.strokeStyle = 'rgba(34,211,238,0.25)';
      ctx.lineWidth = 0.8;
      for (let i = 1; i < rrtTree.length; i++) {
        const node = rrtTree[i];
        const parent = rrtTree[node.parent];
        if (!parent) continue;
        ctx.beginPath();
        ctx.moveTo(parent.x, parent.y);
        ctx.lineTo(node.x, node.y);
        ctx.stroke();
      }
    }

    // RRT final path — bright green
    if (planMode === 'rrt' && rrtPath.length > 1) {
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.moveTo(rrtPath[0].x, rrtPath[0].y);
      for (let i = 1; i < rrtPath.length; i++) {
        ctx.lineTo(rrtPath[i].x, rrtPath[i].y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // robot draw kar — circular body with heading arrow
    if (robot) {
      drawRobot(robot.x, robot.y, robot.theta);
    }

    // instruction text — jab kuch set nahi hai
    if (!startPos || !goalPos) {
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.font = '13px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const text = !startPos ? 'Draw obstacles, then set Start & Goal' : 'Now set the Goal position';
      ctx.fillText(text, canvasW / 2, canvasH / 2);
    }
  }

  // robot ka visual — green circle + heading arrow
  function drawRobot(x, y, theta) {
    // body — emerald green circle
    ctx.beginPath();
    ctx.arc(x, y, ROBOT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16,185,129,0.6)';
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.stroke();

    // heading arrow — chhota arrow jo direction dikhaye
    const arrowLen = ROBOT_RADIUS + 6;
    const ax = x + Math.cos(theta) * arrowLen;
    const ay = y + Math.sin(theta) * arrowLen;
    ctx.beginPath();
    ctx.moveTo(x + Math.cos(theta) * ROBOT_RADIUS * 0.5, y + Math.sin(theta) * ROBOT_RADIUS * 0.5);
    ctx.lineTo(ax, ay);
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // arrowhead — chhoti triangle
    const headLen = 6;
    const headAngle = 0.5;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - headLen * Math.cos(theta - headAngle), ay - headLen * Math.sin(theta - headAngle));
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - headLen * Math.cos(theta + headAngle), ay - headLen * Math.sin(theta + headAngle));
    ctx.stroke();
  }

  // potential field heatmap — red near obstacles, blue near goal
  function drawPotentialHeatmap() {
    if (!potentialField) return;
    const { data, cols, rows, maxVal } = potentialField;
    if (maxVal === 0) return;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = data[r * cols + c];
        // normalize 0-1 mein — log scale use kar taaki subtle dikhein
        const norm = Math.min(1, Math.log(1 + val) / Math.log(1 + maxVal));

        // low potential = blue (goal ke paas), high potential = red (obstacle ke paas)
        const red = Math.floor(norm * 200);
        const blue = Math.floor((1 - norm) * 150);
        const alpha = 0.08 + norm * 0.15;

        ctx.fillStyle = `rgba(${red},${Math.floor(norm * 30)},${blue},${alpha})`;
        ctx.fillRect(
          c * POTENTIAL_RESOLUTION,
          r * POTENTIAL_RESOLUTION,
          POTENTIAL_RESOLUTION,
          POTENTIAL_RESOLUTION
        );
      }
    }
  }

  // --- Main animation loop ---
  function animate() {
    animationId = requestAnimationFrame(animate);
    if (!isVisible) return;

    // RRT step-by-step grow kar — har frame mein kuch nodes add kar
    if (isPlanning && planMode === 'rrt') {
      // har frame mein 5 RRT steps kar — smooth animation ke liye
      for (let i = 0; i < 5; i++) {
        planningStep++;
        if (rrtStep() || planningStep > RRT_MAX_ITER) {
          isPlanning = false;
          break;
        }
      }
    }

    // robot update kar agar follow mode mein hai
    updateRobot();

    // sab draw kar
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

  // --- Resize listener ---
  window.addEventListener('resize', () => {
    if (isVisible) resizeCanvas();
  });

  // initial setup
  resizeCanvas();
  animate();
}
