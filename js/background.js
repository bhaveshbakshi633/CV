// Matrix Rain background - robotics/math symbols girte hain atmospheric style
// Har column mein characters fall karte hain different speeds pe

export function initBackground() {
  // reduced motion wale ke liye kuch mat kar
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const canvas = document.createElement('canvas');
  canvas.id = 'bgCanvas';
  canvas.style.cssText = 'position:fixed;inset:0;z-index:1;pointer-events:none;';
  document.body.insertBefore(canvas, document.body.firstChild);

  const ctx = canvas.getContext('2d');
  let W, H;
  let paused = false;
  let lastTime = 0;

  // mouse position track kar - proximity effect ke liye
  const mouse = { x: -9999, y: -9999 };

  // robotics/math symbols ka pool - ye girenge matrix style
  const SYMBOLS = [
    'θ₁', 'θ₂', 'q̇', 'τ', 'π(a|s)', '∇J',
    'FK', 'IK', 'PID', 'Kp', 'Kd', 'RL',
    'PPO', 'SAC', '∂', '∫', 'Σ', 'λ',
    'γ', 'ε', 'α', 'η', 'ω', 'φ',
    'ψ', 'δ', '∞', '≈', '←', '→',
    '↑', '↓',
  ];

  // columns ka spacing calculate kar
  const COL_MIN = 25;
  const COL_MAX = 35;
  const CHAR_SIZE_MIN = 12;
  const CHAR_SIZE_MAX = 14;
  const FONT_FAMILY = '"JetBrains Mono", monospace';

  // mouse proximity config
  const MOUSE_RADIUS = 200;
  const MOUSE_RADIUS_SQ = MOUSE_RADIUS * MOUSE_RADIUS; // sqrt avoid karne ke liye

  // opacity ranges - base (bina mouse ke)
  const BASE_OPACITY_MIN = 0.03;
  const BASE_OPACITY_MAX = 0.15;
  // mouse ke paas wali opacity
  const HOVER_OPACITY_MIN = 0.25;
  const HOVER_OPACITY_MAX = 0.4;

  // speed range for columns
  const SPEED_MIN = 0.3;
  const SPEED_MAX = 1.2;
  const MOUSE_SPEED_MULTIPLIER = 2.0;

  // lerp factor - kitni smoothly brightness transition kare
  const LERP_FACTOR = 0.06;

  // color range - cyan-blue-green palette
  // r: 0-60, g: 180-200, b: 160-255
  // har column ka apna fixed color hoga
  function randomColor() {
    const r = Math.floor(Math.random() * 61);          // 0-60
    const g = Math.floor(180 + Math.random() * 21);     // 180-200
    const b = Math.floor(160 + Math.random() * 96);     // 160-255
    return { r, g, b };
  }

  // columns array - ye pre-allocated rahega, resize pe recalculate hoga
  let columns = [];

  // har column mein multiple characters honge - ye struct hai ek column ka
  // column = { x, speed, chars: [{ symbol, y, opacity, targetOpacity, fontSize }], color }

  function createColumn(x) {
    const speed = SPEED_MIN + Math.random() * (SPEED_MAX - SPEED_MIN);
    const color = randomColor();
    const fontSize = CHAR_SIZE_MIN + Math.random() * (CHAR_SIZE_MAX - CHAR_SIZE_MIN);
    // kitne chars honge is column mein - viewport height ke hisaab se
    const charSpacing = fontSize * 1.6; // vertical gap between chars
    const numChars = Math.ceil(H / charSpacing) + 3; // thode extra rakho overflow ke liye

    // chars array pre-allocate kar
    const chars = new Array(numChars);
    // random starting offset de - sab ek saath nahi girne chahiye
    const startY = -Math.random() * H;

    for (let i = 0; i < numChars; i++) {
      chars[i] = {
        symbol: SYMBOLS[(Math.random() * SYMBOLS.length) | 0],
        y: startY + i * charSpacing,
        // current opacity - lerp se smooth hoga
        opacity: BASE_OPACITY_MIN + Math.random() * (BASE_OPACITY_MAX - BASE_OPACITY_MIN),
        targetOpacity: 0,
        // depth factor - top chars brighter, bottom chars dimmer within column
        depthFactor: 0,
      };
    }

    return {
      x,
      speed,
      color,
      fontSize,
      charSpacing,
      chars,
      numChars,
      // mouse proximity ka current multiplier - lerp se smooth hoga
      speedMultiplier: 1.0,
      targetSpeedMultiplier: 1.0,
    };
  }

  function initColumns() {
    columns = [];
    // columns ka spacing random rakho COL_MIN aur COL_MAX ke beech
    let x = Math.random() * COL_MIN; // thoda offset de start mein
    while (x < W) {
      columns.push(createColumn(x));
      x += COL_MIN + Math.random() * (COL_MAX - COL_MIN);
    }
  }

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
    // columns recalculate kar
    initColumns();
  }

  function update(dt) {
    // har column update kar
    for (let c = 0; c < columns.length; c++) {
      const col = columns[c];

      // mouse proximity check - column level pe kar, efficient hai
      const dx = mouse.x - col.x;
      // rough check: agar column x mouse se bahut door hai toh skip
      const isNearMouseX = Math.abs(dx) < MOUSE_RADIUS;

      // target speed multiplier set kar
      col.targetSpeedMultiplier = 1.0;

      // har character update kar
      for (let i = 0; i < col.numChars; i++) {
        const ch = col.chars[i];

        // character ko neeche le ja
        ch.y += col.speed * col.speedMultiplier * dt * 0.06;

        // agar screen ke neeche chala gaya toh upar bhej do - recycle karo
        if (ch.y > H + col.fontSize) {
          ch.y -= col.numChars * col.charSpacing;
          // naya symbol de do recycle pe
          ch.symbol = SYMBOLS[(Math.random() * SYMBOLS.length) | 0];
        }

        // depth factor calculate kar - column ke andar upar bright, neeche dim
        // 0 = top of screen, 1 = bottom of screen
        const normalizedY = ch.y / H;
        // clamp 0-1 ke beech
        const clamped = normalizedY < 0 ? 0 : normalizedY > 1 ? 1 : normalizedY;
        // top bright (1.0), bottom dim (0.3)
        ch.depthFactor = 1.0 - clamped * 0.7;

        // mouse proximity check - character level pe accurate check
        let mouseInfluence = 0;
        if (isNearMouseX && ch.y > 0 && ch.y < H) {
          const dy = mouse.y - ch.y;
          const distSq = dx * dx + dy * dy;
          if (distSq < MOUSE_RADIUS_SQ) {
            // 0 to 1 - 1 jab mouse ke bilkul paas ho
            mouseInfluence = 1 - Math.sqrt(distSq) / MOUSE_RADIUS;
            // column speed bhi badha do
            if (mouseInfluence > 0.3) {
              col.targetSpeedMultiplier = MOUSE_SPEED_MULTIPLIER;
            }
          }
        }

        // target opacity calculate kar - base + mouse influence
        const baseOp = BASE_OPACITY_MIN + (BASE_OPACITY_MAX - BASE_OPACITY_MIN) * ch.depthFactor;
        const hoverOp = HOVER_OPACITY_MIN + (HOVER_OPACITY_MAX - HOVER_OPACITY_MIN) * ch.depthFactor;
        ch.targetOpacity = baseOp + (hoverOp - baseOp) * mouseInfluence;

        // smooth lerp karo opacity ko - transition smooth dikhega
        ch.opacity += (ch.targetOpacity - ch.opacity) * LERP_FACTOR;
      }

      // speed multiplier bhi smooth lerp karo
      col.speedMultiplier += (col.targetSpeedMultiplier - col.speedMultiplier) * LERP_FACTOR;
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // har column draw kar
    for (let c = 0; c < columns.length; c++) {
      const col = columns[c];
      const { r, g, b } = col.color;

      // font ek baar set kar per column - saves ctx calls
      ctx.font = `${col.fontSize}px ${FONT_FAMILY}`;

      for (let i = 0; i < col.numChars; i++) {
        const ch = col.chars[i];

        // sirf visible chars draw kar - jo screen ke andar hain
        if (ch.y < -col.fontSize || ch.y > H + col.fontSize) continue;

        // opacity bahut kam hai toh skip kar - performance ke liye
        if (ch.opacity < 0.005) continue;

        ctx.fillStyle = `rgba(${r},${g},${b},${ch.opacity})`;
        ctx.fillText(ch.symbol, col.x, ch.y);
      }
    }
  }

  function loop(now) {
    if (!paused) {
      // delta time calculate kar - frame rate independent movement ke liye
      const dt = lastTime ? Math.min(now - lastTime, 50) : 16; // 50ms cap to avoid jumps
      lastTime = now;
      update(dt);
      draw();
    } else {
      lastTime = 0; // pause ke baad dt reset kar
    }
    requestAnimationFrame(loop);
  }

  // event listeners lagao
  window.addEventListener('resize', resize);

  // mouse track kar
  document.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });
  document.addEventListener('mouseleave', () => {
    mouse.x = -9999;
    mouse.y = -9999;
  });

  // tab visibility - pause jab tab hidden ho, battery bachao
  document.addEventListener('visibilitychange', () => {
    paused = document.hidden;
  });

  // sab setup ho gaya, shuru karo
  resize();
  requestAnimationFrame(loop);
}
