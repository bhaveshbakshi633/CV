// ============================================================
// 2.5D Raycaster — Wolfenstein 3D style first-person maze explorer
// DDA raycasting, minimap with ray visualization, recursive backtracker maze
// WASD/Arrows + mouse look, fog, wall type colors, split view
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, dungeon explore karo
export function initRaycaster() {
  const container = document.getElementById('raycasterContainer');
  if (!container) {
    console.warn('raycasterContainer nahi mila bhai, raycaster demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';
  const MAP_SIZE = 16; // 16x16 grid

  // wall type colors — [r, g, b] format
  // 0 = empty, 1+ = wall types
  const WALL_COLORS = [
    null,                    // 0 = empty, koi color nahi
    [180, 60, 60],           // 1 = brick red — classic Wolfenstein vibes
    [140, 140, 150],         // 2 = stone gray — dungeon walls
    [70, 130, 70],           // 3 = moss green — damp corridors
    [160, 110, 60],          // 4 = wood brown — door frames
  ];

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let isVisible = false;
  let animationId = null;

  // player state — floating point grid coordinates
  let playerX = 1.5;
  let playerY = 1.5;
  let playerDir = 0; // radians — 0 = right, PI/2 = down
  const MOVE_SPEED = 0.04;
  const ROT_SPEED = 0.035;
  const MOUSE_SENSITIVITY = 0.003;

  // config — sliders se control honge
  let fov = 66 * (Math.PI / 180); // default 66 degrees
  let fogDistance = 12;
  let showMinimap = true;
  let colorMode = 'type'; // 'solid' | 'distance' | 'type'

  // input state — kaunsi keys dabai hain
  const keys = {};
  let pointerLocked = false;

  // ray hit data — minimap mein dikhane ke liye cache karenge
  let rayHits = []; // [{hitX, hitY}] — har ray ka wall hit point

  // --- Map ---
  // 0 = empty, 1-4 = wall types
  let map = [];

  // recursive backtracker se maze generate kar
  function generateMaze() {
    // pehle sab wall bana do — type 1 se fill
    map = [];
    for (let y = 0; y < MAP_SIZE; y++) {
      const row = [];
      for (let x = 0; x < MAP_SIZE; x++) {
        row.push(1);
      }
      map.push(row);
    }

    // visited tracker — carving ke liye
    const visited = new Set();

    function carve(cx, cy) {
      visited.add(cy + ',' + cx);
      map[cy][cx] = 0;

      // random directions — shuffle kar ke explore kar
      const dirs = [[0, 2], [0, -2], [2, 0], [-2, 0]];
      shuffleArray(dirs);

      for (const [dy, dx] of dirs) {
        const nx = cx + dx;
        const ny = cy + dy;
        // boundary check — odd cells pe hi carve karo
        if (nx > 0 && nx < MAP_SIZE - 1 && ny > 0 && ny < MAP_SIZE - 1 && !visited.has(ny + ',' + nx)) {
          // beech ki wall bhi hata do — path connect kar
          map[cy + dy / 2][cx + dx / 2] = 0;
          carve(nx, ny);
        }
      }
    }

    // top-left odd cell se shuru
    carve(1, 1);

    // ab walls ko random types assign kar — variety ke liye
    for (let y = 0; y < MAP_SIZE; y++) {
      for (let x = 0; x < MAP_SIZE; x++) {
        if (map[y][x] !== 0) {
          // random wall type — 1 se 4 tak
          map[y][x] = 1 + Math.floor(Math.random() * WALL_COLORS.length - 1);
          if (map[y][x] < 1) map[y][x] = 1;
          if (map[y][x] >= WALL_COLORS.length) map[y][x] = WALL_COLORS.length - 1;
        }
      }
    }

    // player ko top-left mein rakh, facing right
    playerX = 1.5;
    playerY = 1.5;
    playerDir = 0;
  }

  // Fisher-Yates shuffle — proper random permutation
  function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
  }

  // --- DOM Structure banao ---
  // pehle existing children preserve karo (agar koi hai)
  const existingChildren = [];
  while (container.firstChild) {
    existingChildren.push(container.removeChild(container.firstChild));
  }
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — yahan 3D view aur minimap dono render honge
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:#0a0a14',
    'touch-action:none',
  ].join(';');
  canvas.tabIndex = 0; // focus ke liye chahiye keyboard events catch karne
  canvas.style.outline = 'none';
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // info bar — instructions dikhayega
  const infoBar = document.createElement('div');
  infoBar.style.cssText = [
    'display:flex',
    'justify-content:space-between',
    'align-items:center',
    'margin-top:6px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:rgba(' + ACCENT_RGB + ',0.5)',
    'min-height:18px',
    'flex-wrap:wrap',
    'gap:4px 12px',
  ].join(';');
  container.appendChild(infoBar);

  const infoLeft = document.createElement('span');
  infoLeft.textContent = 'WASD/Arrows: move | Mouse: look (click to capture)';
  infoBar.appendChild(infoLeft);

  const infoRight = document.createElement('span');
  infoRight.textContent = 'FOV: ' + Math.round(fov * 180 / Math.PI) + '°';
  infoBar.appendChild(infoRight);

  // controls container
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Helper: button banao (dark theme, purple accent) ---
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
        btn.style.color = '#b0b0b0';
      }
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  function setButtonActive(btn, active) {
    btn._active = active;
    if (active) {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.35)';
      btn.style.color = ACCENT;
      btn.style.borderColor = ACCENT;
    } else {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      btn.style.color = '#b0b0b0';
      btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
    }
  }

  // --- Helper: slider banao ---
  function createSlider(label, min, max, value, step, onChange) {
    const wrap = document.createElement('div');
    wrap.style.cssText = [
      'display:flex',
      'align-items:center',
      'gap:6px',
      'font-family:"JetBrains Mono",monospace',
      'font-size:11px',
      'color:#888',
    ].join(';');

    const lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.whiteSpace = 'nowrap';
    wrap.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.value = value;
    slider.step = step;
    slider.style.cssText = [
      'width:80px',
      'accent-color:' + ACCENT,
      'cursor:pointer',
    ].join(';');

    const valSpan = document.createElement('span');
    valSpan.textContent = value;
    valSpan.style.cssText = 'min-width:36px;text-align:right;color:' + ACCENT + ';';

    slider.addEventListener('input', () => {
      valSpan.textContent = slider.value;
      onChange(Number(slider.value));
    });

    wrap.appendChild(slider);
    wrap.appendChild(valSpan);
    controlsDiv.appendChild(wrap);
    return { slider, valSpan };
  }

  // --- Controls banao ---

  // Generate Maze button
  createButton('Generate Maze', () => {
    generateMaze();
  });

  // separator
  const sep1 = document.createElement('span');
  sep1.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.2);font-size:14px;';
  sep1.textContent = '|';
  controlsDiv.appendChild(sep1);

  // FOV slider — 40 se 120 degrees
  const fovSlider = createSlider('FOV', 40, 120, Math.round(fov * 180 / Math.PI), 1, (v) => {
    fov = v * (Math.PI / 180);
    infoRight.textContent = 'FOV: ' + v + '°';
  });

  // Fog distance slider
  createSlider('Fog', 4, 20, fogDistance, 1, (v) => {
    fogDistance = v;
  });

  // separator
  const sep2 = document.createElement('span');
  sep2.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.2);font-size:14px;';
  sep2.textContent = '|';
  controlsDiv.appendChild(sep2);

  // Minimap toggle
  const minimapBtn = createButton('Minimap: ON', () => {
    showMinimap = !showMinimap;
    minimapBtn.textContent = 'Minimap: ' + (showMinimap ? 'ON' : 'OFF');
    setButtonActive(minimapBtn, showMinimap);
  });
  setButtonActive(minimapBtn, true);

  // separator
  const sep3 = document.createElement('span');
  sep3.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.2);font-size:14px;';
  sep3.textContent = '|';
  controlsDiv.appendChild(sep3);

  // Wall color mode — cycle through modes
  let colorModes = ['type', 'distance', 'solid'];
  let colorModeLabels = { type: 'By Type', distance: 'Distance', solid: 'Solid' };
  const colorBtn = createButton('Color: By Type', () => {
    const idx = colorModes.indexOf(colorMode);
    colorMode = colorModes[(idx + 1) % colorModes.length];
    colorBtn.textContent = 'Color: ' + colorModeLabels[colorMode];
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
  }

  // --- Keyboard input ---
  function onKeyDown(e) {
    keys[e.key.toLowerCase()] = true;
    keys[e.code] = true;
    // arrow keys aur WASD ke liye default scroll prevent karo
    if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 'a', 's', 'd'].includes(e.key.toLowerCase())) {
      e.preventDefault();
    }
  }

  function onKeyUp(e) {
    keys[e.key.toLowerCase()] = false;
    keys[e.code] = false;
  }

  canvas.addEventListener('keydown', onKeyDown);
  canvas.addEventListener('keyup', onKeyUp);

  // global listener bhi lagao — agar canvas focused nahi hai toh bhi kaam kare
  // lekin sirf tab jab canvas visible ho
  document.addEventListener('keydown', (e) => {
    if (isVisible && pointerLocked) {
      onKeyDown(e);
    }
  });
  document.addEventListener('keyup', (e) => {
    if (isVisible) {
      onKeyUp(e);
    }
  });

  // --- Pointer lock — mouse se rotation control ---
  canvas.addEventListener('click', () => {
    if (!pointerLocked) {
      canvas.requestPointerLock();
    }
  });

  document.addEventListener('pointerlockchange', () => {
    pointerLocked = (document.pointerLockElement === canvas);
    canvas.style.cursor = pointerLocked ? 'none' : 'crosshair';
  });

  document.addEventListener('mousemove', (e) => {
    if (pointerLocked) {
      playerDir += e.movementX * MOUSE_SENSITIVITY;
    }
  });

  // --- Touch controls — virtual joystick style ---
  let touchStartX = 0;
  let touchStartY = 0;
  let touchMoveX = 0;
  let touchMoveY = 0;
  let touchActive = false;
  let touchRotateStartX = 0;
  let touchRotateActive = false;

  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (e.touches.length === 1) {
      // single touch — left side = move, right side = rotate
      const rect = canvas.getBoundingClientRect();
      const tx = e.touches[0].clientX - rect.left;
      if (tx < canvasW * 0.5) {
        // left side — movement joystick
        touchActive = true;
        touchStartX = e.touches[0].clientX;
        touchStartY = e.touches[0].clientY;
        touchMoveX = 0;
        touchMoveY = 0;
      } else {
        // right side — rotation
        touchRotateActive = true;
        touchRotateStartX = e.touches[0].clientX;
      }
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (touchActive && e.touches.length >= 1) {
      touchMoveX = e.touches[0].clientX - touchStartX;
      touchMoveY = e.touches[0].clientY - touchStartY;
    }
    if (touchRotateActive && e.touches.length >= 1) {
      const lastTouch = e.touches[e.touches.length - 1];
      const dx = lastTouch.clientX - touchRotateStartX;
      playerDir += dx * 0.005;
      touchRotateStartX = lastTouch.clientX;
    }
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    touchActive = false;
    touchRotateActive = false;
    touchMoveX = 0;
    touchMoveY = 0;
  }, { passive: false });

  // --- Player movement — collision detection ke saath ---
  function updatePlayer() {
    const cosDir = Math.cos(playerDir);
    const sinDir = Math.sin(playerDir);

    let moveX = 0;
    let moveY = 0;

    // forward/backward — W/Up
    if (keys['w'] || keys['arrowup'] || keys['ArrowUp']) {
      moveX += cosDir * MOVE_SPEED;
      moveY += sinDir * MOVE_SPEED;
    }
    // backward — S/Down
    if (keys['s'] || keys['arrowdown'] || keys['ArrowDown']) {
      moveX -= cosDir * MOVE_SPEED;
      moveY -= sinDir * MOVE_SPEED;
    }

    // strafe left — Q (A bhi rotate karega)
    if (keys['q']) {
      moveX += sinDir * MOVE_SPEED;
      moveY -= cosDir * MOVE_SPEED;
    }
    // strafe right — E
    if (keys['e']) {
      moveX -= sinDir * MOVE_SPEED;
      moveY += cosDir * MOVE_SPEED;
    }

    // rotate — A/Left, D/Right (keyboard rotation when mouse not locked)
    if (keys['a'] || keys['arrowleft'] || keys['ArrowLeft']) {
      playerDir -= ROT_SPEED;
    }
    if (keys['d'] || keys['arrowright'] || keys['ArrowRight']) {
      playerDir += ROT_SPEED;
    }

    // touch input se bhi move/rotate
    if (touchActive) {
      const deadzone = 15;
      if (Math.abs(touchMoveY) > deadzone) {
        const touchForward = -touchMoveY * 0.0004;
        moveX += cosDir * touchForward;
        moveY += sinDir * touchForward;
      }
      if (Math.abs(touchMoveX) > deadzone) {
        const touchStrafe = touchMoveX * 0.0004;
        moveX -= sinDir * touchStrafe;
        moveY += cosDir * touchStrafe;
      }
    }

    // collision detection — wall mein ghusne mat dena
    // X aur Y independently check karo — sliding along walls ke liye
    const margin = 0.2; // player ka collision radius

    const newX = playerX + moveX;
    const newY = playerY + moveY;

    // X axis check
    if (moveX !== 0) {
      const checkX = newX + (moveX > 0 ? margin : -margin);
      if (map[Math.floor(playerY)]) {
        const cellX = Math.floor(checkX);
        const cellYa = Math.floor(playerY - margin);
        const cellYb = Math.floor(playerY + margin);
        let blocked = false;
        if (cellX >= 0 && cellX < MAP_SIZE) {
          if (cellYa >= 0 && cellYa < MAP_SIZE && map[cellYa][cellX] > 0) blocked = true;
          if (cellYb >= 0 && cellYb < MAP_SIZE && map[cellYb][cellX] > 0) blocked = true;
        } else {
          blocked = true;
        }
        if (!blocked) playerX = newX;
      }
    }

    // Y axis check
    if (moveY !== 0) {
      const checkY = newY + (moveY > 0 ? margin : -margin);
      if (checkY >= 0 && checkY < MAP_SIZE) {
        const cellY = Math.floor(checkY);
        const cellXa = Math.floor(playerX - margin);
        const cellXb = Math.floor(playerX + margin);
        let blocked = false;
        if (cellY >= 0 && cellY < MAP_SIZE) {
          if (cellXa >= 0 && cellXa < MAP_SIZE && map[cellY][cellXa] > 0) blocked = true;
          if (cellXb >= 0 && cellXb < MAP_SIZE && map[cellY][cellXb] > 0) blocked = true;
        } else {
          blocked = true;
        }
        if (!blocked) playerY = newY;
      }
    }
  }

  // --- DDA Raycasting — ek ray cast kar aur wall hit return kar ---
  // returns { dist, wallType, side, hitX, hitY, mapX, mapY, texU }
  function castRay(originX, originY, angle) {
    const dirX = Math.cos(angle);
    const dirY = Math.sin(angle);

    // current grid cell
    let mapX = Math.floor(originX);
    let mapY = Math.floor(originY);

    // DDA step sizes — ek grid line cross karne mein kitna distance lagega
    // division by zero se bachao — agar direction component 0 hai toh bahut bada number do
    const deltaDistX = Math.abs(dirX) < 1e-10 ? 1e10 : Math.abs(1.0 / dirX);
    const deltaDistY = Math.abs(dirY) < 1e-10 ? 1e10 : Math.abs(1.0 / dirY);

    // step direction aur initial side distance
    let stepX, stepY;
    let sideDistX, sideDistY;

    if (dirX < 0) {
      stepX = -1;
      sideDistX = (originX - mapX) * deltaDistX;
    } else {
      stepX = 1;
      sideDistX = (mapX + 1.0 - originX) * deltaDistX;
    }

    if (dirY < 0) {
      stepY = -1;
      sideDistY = (originY - mapY) * deltaDistY;
    } else {
      stepY = 1;
      sideDistY = (mapY + 1.0 - originY) * deltaDistY;
    }

    // DDA loop — grid cells step through karo jab tak wall na mile
    let side = 0; // 0 = X face hit, 1 = Y face hit
    let hit = false;
    let maxSteps = MAP_SIZE * 3; // infinite loop se bachao

    while (!hit && maxSteps > 0) {
      maxSteps--;

      // next grid line — X ya Y, jo paas hai
      if (sideDistX < sideDistY) {
        sideDistX += deltaDistX;
        mapX += stepX;
        side = 0;
      } else {
        sideDistY += deltaDistY;
        mapY += stepY;
        side = 1;
      }

      // boundary check
      if (mapX < 0 || mapX >= MAP_SIZE || mapY < 0 || mapY >= MAP_SIZE) {
        hit = true;
        break;
      }

      // wall check
      if (map[mapY][mapX] > 0) {
        hit = true;
      }
    }

    // perpendicular distance calculate kar — fisheye correction ke liye
    let perpDist;
    if (side === 0) {
      perpDist = sideDistX - deltaDistX;
    } else {
      perpDist = sideDistY - deltaDistY;
    }
    if (perpDist < 0.001) perpDist = 0.001; // zero division se bachao

    // wall hit point — texture coordinate ke liye
    let hitX, hitY;
    if (side === 0) {
      hitX = originX + perpDist * dirX;
      hitY = originY + perpDist * dirY;
    } else {
      hitX = originX + perpDist * dirX;
      hitY = originY + perpDist * dirY;
    }

    // texture U coordinate — 0 se 1 ke beech
    let texU;
    if (side === 0) {
      texU = hitY - Math.floor(hitY);
    } else {
      texU = hitX - Math.floor(hitX);
    }

    // wall type nikal
    let wallType = 1;
    if (mapX >= 0 && mapX < MAP_SIZE && mapY >= 0 && mapY < MAP_SIZE) {
      wallType = map[mapY][mapX];
    }

    return {
      dist: perpDist,
      wallType: wallType,
      side: side, // 0 = EW face, 1 = NS face
      hitX: hitX,
      hitY: hitY,
      mapX: mapX,
      mapY: mapY,
      texU: texU,
    };
  }

  // --- Wall color calculate kar — mode, type, distance, side ke hisaab se ---
  function getWallColor(wallType, dist, side) {
    let r, g, b;

    if (colorMode === 'solid') {
      // solid purple-ish color
      r = 120; g = 80; b = 160;
    } else if (colorMode === 'distance') {
      // distance se color change — neela se laal
      const t = Math.min(dist / fogDistance, 1.0);
      r = Math.round(100 + 120 * (1 - t));
      g = Math.round(60 + 40 * (1 - t));
      b = Math.round(160 * (1 - t) + 80 * t);
    } else {
      // by type — WALL_COLORS se color utha
      const wc = WALL_COLORS[wallType] || WALL_COLORS[1];
      r = wc[0]; g = wc[1]; b = wc[2];
    }

    // side shading — NS faces thoda dark karo (simple ambient occlusion feel)
    if (side === 1) {
      r = Math.round(r * 0.7);
      g = Math.round(g * 0.7);
      b = Math.round(b * 0.7);
    }

    // distance fog — door ki walls dark karo
    const fogFactor = Math.max(0, 1.0 - dist / fogDistance);
    r = Math.round(r * fogFactor);
    g = Math.round(g * fogFactor);
    b = Math.round(b * fogFactor);

    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }

  // --- Main render function ---
  function render() {
    const w = canvasW;
    const h = canvasH;

    // clear canvas
    ctx.clearRect(0, 0, w, h);

    // layout calculate kar — minimap left 35%, 3D view right 65%
    let minimapW = 0;
    let viewX = 0;
    let viewW = w;

    if (showMinimap) {
      minimapW = Math.floor(w * 0.35);
      viewX = minimapW;
      viewW = w - minimapW;
    }

    // --- Ceiling aur Floor gradients ---
    // ceiling — dark gray gradient, horizon ke paas lighter
    const ceilGrad = ctx.createLinearGradient(viewX, 0, viewX, h / 2);
    ceilGrad.addColorStop(0, '#0a0a14');
    ceilGrad.addColorStop(0.6, '#12121e');
    ceilGrad.addColorStop(1, '#1a1a2a');
    ctx.fillStyle = ceilGrad;
    ctx.fillRect(viewX, 0, viewW, h / 2);

    // floor — darker gradient, horizon ke paas lighter
    const floorGrad = ctx.createLinearGradient(viewX, h / 2, viewX, h);
    floorGrad.addColorStop(0, '#18181f');
    floorGrad.addColorStop(0.4, '#111118');
    floorGrad.addColorStop(1, '#08080c');
    ctx.fillStyle = floorGrad;
    ctx.fillRect(viewX, h / 2, viewW, h / 2);

    // --- Raycasting — har 2 pixels pe ek ray cast kar (performance ke liye) ---
    const stripWidth = 2; // 2 pixel wide strips — ray count halve
    const numRays = Math.ceil(viewW / stripWidth);
    rayHits = []; // minimap ke liye cache clear

    for (let i = 0; i < numRays; i++) {
      // ray angle calculate kar — FOV spread across screen
      const screenX = (i / numRays) - 0.5; // -0.5 se 0.5 tak
      const rayAngle = playerDir + screenX * fov;

      // ray cast kar
      const hit = castRay(playerX, playerY, rayAngle);

      // minimap ke liye hit point save kar
      rayHits.push({ hitX: hit.hitX, hitY: hit.hitY });

      // fisheye correction — perpendicular distance already use ho rahi hai
      // but additional correction for angled rays
      const correctedDist = hit.dist * Math.cos(rayAngle - playerDir);

      // wall strip height calculate kar
      const wallHeight = h / correctedDist;

      // strip position on screen
      const drawStart = Math.max(0, (h - wallHeight) / 2);
      const drawEnd = Math.min(h, (h + wallHeight) / 2);
      const drawX = viewX + i * stripWidth;

      // wall color
      const wallColor = getWallColor(hit.wallType, correctedDist, hit.side);

      // wall strip draw kar
      ctx.fillStyle = wallColor;
      ctx.fillRect(drawX, drawStart, stripWidth, drawEnd - drawStart);

      // simple vertical stripe texture — wall pe subtle lines
      if (correctedDist < fogDistance * 0.8) {
        // texU se stripe pattern banao — vertical lines on wall surface
        const stripePhase = hit.texU * 8;
        const stripeAlpha = (Math.sin(stripePhase * Math.PI * 2) * 0.5 + 0.5) * 0.08;
        const fogFactor = Math.max(0, 1.0 - correctedDist / fogDistance);
        ctx.fillStyle = 'rgba(255,255,255,' + (stripeAlpha * fogFactor).toFixed(3) + ')';
        ctx.fillRect(drawX, drawStart, stripWidth, drawEnd - drawStart);
      }
    }

    // --- 3D view border ---
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.1)';
    ctx.lineWidth = 1;
    if (showMinimap) {
      ctx.beginPath();
      ctx.moveTo(viewX, 0);
      ctx.lineTo(viewX, h);
      ctx.stroke();
    }

    // --- Minimap render ---
    if (showMinimap) {
      renderMinimap(minimapW, h);
    }

    // --- HUD overlay — compass direction ---
    const compassDirs = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE'];
    let normDir = ((playerDir % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
    const compassIdx = Math.round(normDir / (Math.PI / 4)) % 8;
    const compassStr = compassDirs[compassIdx];

    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
    ctx.fillText(compassStr, viewX + viewW / 2, 16);

    // crosshair — center mein chhota dot
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
    ctx.beginPath();
    ctx.arc(viewX + viewW / 2, h / 2, 2, 0, Math.PI * 2);
    ctx.fill();

    // position info — bottom left of 3D view
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.25)';
    ctx.fillText(
      'pos: (' + playerX.toFixed(1) + ', ' + playerY.toFixed(1) + ')  dir: ' + (normDir * 180 / Math.PI).toFixed(0) + '°',
      viewX + 8, h - 8
    );
  }

  // --- Minimap rendering — top-down view with rays ---
  function renderMinimap(mmW, mmH) {
    // minimap background — semi-transparent dark
    ctx.fillStyle = 'rgba(5, 5, 12, 0.85)';
    ctx.fillRect(0, 0, mmW, mmH);

    // map grid calculate kar — cell size
    const padding = 8;
    const mapDrawW = mmW - padding * 2;
    const mapDrawH = mmH - padding * 2;
    const cellW = mapDrawW / MAP_SIZE;
    const cellH = mapDrawH / MAP_SIZE;

    // grid cells draw kar — walls as filled squares
    for (let y = 0; y < MAP_SIZE; y++) {
      for (let x = 0; x < MAP_SIZE; x++) {
        const drawX = padding + x * cellW;
        const drawY = padding + y * cellH;

        if (map[y][x] > 0) {
          // wall cell — color by type
          const wt = map[y][x];
          const wc = WALL_COLORS[wt] || WALL_COLORS[1];
          ctx.fillStyle = 'rgba(' + wc[0] + ',' + wc[1] + ',' + wc[2] + ',0.5)';
          ctx.fillRect(drawX, drawY, cellW, cellH);
        } else {
          // empty cell — subtle grid
          ctx.fillStyle = 'rgba(255,255,255,0.02)';
          ctx.fillRect(drawX, drawY, cellW, cellH);
        }

        // grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(drawX, drawY, cellW, cellH);
      }
    }

    // --- Rays draw kar — fan of thin lines from player to wall hits ---
    const pxX = padding + playerX * cellW;
    const pxY = padding + playerY * cellH;

    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.12)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    // har 4th ray hi draw karo minimap mein — bahut zyada lines messy lagti hain
    for (let i = 0; i < rayHits.length; i += 3) {
      const rh = rayHits[i];
      const rhX = padding + rh.hitX * cellW;
      const rhY = padding + rh.hitY * cellH;
      ctx.moveTo(pxX, pxY);
      ctx.lineTo(rhX, rhY);
    }
    ctx.stroke();

    // --- Player dot + direction line ---
    // direction line — player ke direction mein thodi lambi line
    const dirLen = cellW * 1.5;
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pxX, pxY);
    ctx.lineTo(pxX + Math.cos(playerDir) * dirLen, pxY + Math.sin(playerDir) * dirLen);
    ctx.stroke();

    // player dot
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(pxX, pxY, 3, 0, Math.PI * 2);
    ctx.fill();

    // FOV cone outline — subtle
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const fovLen = cellW * 5;
    const leftAngle = playerDir - fov / 2;
    const rightAngle = playerDir + fov / 2;
    ctx.moveTo(pxX, pxY);
    ctx.lineTo(pxX + Math.cos(leftAngle) * fovLen, pxY + Math.sin(leftAngle) * fovLen);
    ctx.moveTo(pxX, pxY);
    ctx.lineTo(pxX + Math.cos(rightAngle) * fovLen, pxY + Math.sin(rightAngle) * fovLen);
    ctx.stroke();

    // minimap label
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.fillText('MINIMAP', padding + 2, padding + 10);

    // minimap border — right edge
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mmW, 0);
    ctx.lineTo(mmW, mmH);
    ctx.stroke();
  }

  // --- Main animation loop ---
  function gameLoop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = requestAnimationFrame(gameLoop);
      return;
    }

    // player movement update
    updatePlayer();

    // render everything
    render();

    animationId = requestAnimationFrame(gameLoop);
  }

  // --- IntersectionObserver — sirf jab dikhe tab render karo ---
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          isVisible = true;
        } else {
          isVisible = false;
          // pointer lock release kar agar chhupa toh
          if (pointerLocked && document.pointerLockElement === canvas) {
            document.exitPointerLock();
          }
          // sab keys release kar — nahi toh ghost movement hogi
          Object.keys(keys).forEach(k => { keys[k] = false; });
        }
      });
    },
    { threshold: 0.1 }
  );

  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) gameLoop(); });

  // tab switch pe bhi handle karo
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      isVisible = false;
      Object.keys(keys).forEach(k => { keys[k] = false; });
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) isVisible = true;
    }
  });

  // --- Init ---
  generateMaze();
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // game loop shuru kar
  gameLoop();
}
