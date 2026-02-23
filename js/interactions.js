// interactions.js — the fun stuff
// 3D tilt cards, terminal typing, RL training curve, robot arm IK

// ---- 3D TILT ON BENTO CARDS ----
export function initTiltCards() {
  const cards = document.querySelectorAll('.bento-card');
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReducedMotion || !cards.length) return;

  cards.forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      // rotate: max 6deg
      const rotateX = ((y - centerY) / centerY) * -6;
      const rotateY = ((x - centerX) / centerX) * 6;

      card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px) scale(1.01)`;

      // move shine effect
      const shine = card.querySelector('.bento-shine');
      if (shine) {
        shine.style.background = `radial-gradient(circle at ${x}px ${y}px, rgba(255,255,255,0.06) 0%, transparent 60%)`;
      }
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
      const shine = card.querySelector('.bento-shine');
      if (shine) shine.style.background = '';
    });

    // inject shine overlay layer
    if (!card.querySelector('.bento-shine')) {
      const shine = document.createElement('div');
      shine.className = 'bento-shine';
      card.appendChild(shine);
    }
  });
}

// ---- TERMINAL TYPING EFFECT ----
export function initTypingEffect() {
  const el = document.getElementById('heroSubtitle');
  if (!el) return;

  const rawText = el.dataset.text || '';
  if (!rawText) return;

  // short delay before clearing pre-filled content and starting typing
  // this ensures the subtitle is visible on first paint
  setTimeout(() => {
    while (el.firstChild) el.removeChild(el.firstChild);
    const cursor = document.createElement('span');
    cursor.className = 'typing-cursor';
    cursor.textContent = '|';
    el.appendChild(cursor);
    startTyping(el, rawText, cursor);
  }, 400);
}

function startTyping(el, rawText, cursor) {

  let i = 0;
  const speed = 45; // ms per char
  let pastDash = false; // track if we're past the mdash
  let accentSpan = null;

  function type() {
    if (i >= rawText.length) {
      // fade cursor after done
      setTimeout(() => {
        cursor.classList.add('blink-out');
      }, 1500);
      return;
    }

    const char = rawText[i];

    // handle &mdash; entity
    if (char === '&') {
      const entityEnd = rawText.indexOf(';', i);
      if (entityEnd !== -1) {
        const entity = rawText.substring(i, entityEnd + 1);
        if (entity === '&mdash;') {
          el.insertBefore(document.createTextNode(' \u2014 '), cursor);
          pastDash = true;
        } else {
          // decode other entities via temp element
          const temp = document.createElement('span');
          temp.textContent = entity; // safe: textContent not innerHTML
          el.insertBefore(document.createTextNode(temp.textContent), cursor);
        }
        i = entityEnd + 1;
        setTimeout(type, speed * 3);
        return;
      }
    }

    // after the dash, wrap in accent color span
    if (pastDash) {
      if (!accentSpan) {
        accentSpan = document.createElement('em');
        accentSpan.className = 'typed-accent';
        el.insertBefore(accentSpan, cursor);
      }
      accentSpan.textContent += char;
    } else {
      el.insertBefore(document.createTextNode(char), cursor);
    }

    i++;

    // variable speed for natural feel
    const delay = (char === ' ') ? speed * 1.3 :
                  (char === ',') ? speed * 2.5 :
                  speed + Math.random() * 20;

    setTimeout(type, delay);
  }

  // start after a short delay
  setTimeout(type, 600);
}

// ---- RL TRAINING CURVE VISUALIZER ----
export function initRLVisualizer() {
  const canvas = document.getElementById('rlCanvas');
  if (!canvas) return;

  const container = document.getElementById('rlVisualizer');
  const ctx = canvas.getContext('2d');
  let animStarted = false;
  let animFrame = 0;
  const totalFrames = 200;

  function resize() {
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = 180;
  }

  resize();
  window.addEventListener('resize', resize);

  // generate reward curves (3 runs with different seeds)
  function generateCurve(seed) {
    const points = [];
    let reward = -0.5 + seed * 0.1;
    for (let i = 0; i < totalFrames; i++) {
      const progress = i / totalFrames;
      const target = 0.85 + seed * 0.05;
      const noise = (Math.sin(i * 0.3 + seed * 7) * 0.08 +
                     Math.sin(i * 0.7 + seed * 3) * 0.04) *
                     (1 - progress * 0.7);
      reward += (target - reward) * 0.025 + noise * 0.15;
      reward = Math.max(-0.6, Math.min(1.0, reward));
      points.push(reward);
    }
    return points;
  }

  const curves = [
    { points: generateCurve(0), color: 'rgba(74, 158, 255, 1.0)', width: 2.5 },
    { points: generateCurve(1), color: 'rgba(74, 158, 255, 0.5)', width: 1.5 },
    { points: generateCurve(2), color: 'rgba(74, 158, 255, 0.3)', width: 1 }
  ];

  function drawGrid() {
    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 10, bottom: 20, left: 40, right: 10 };

    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1;

    // horizontal grid lines
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (h - pad.top - pad.bottom) * (i / 4);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
    }

    // y-axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    const labels = ['1.0', '0.5', '0.0', '-0.5'];
    labels.forEach((label, i) => {
      const y = pad.top + (h - pad.top - pad.bottom) * (i / (labels.length - 1));
      ctx.fillText(label, pad.left - 6, y + 3);
    });

    // x-axis label
    ctx.textAlign = 'center';
    ctx.fillText('steps (k)', w / 2, h - 2);

    return pad;
  }

  function draw(frame) {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const pad = drawGrid();
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const visiblePoints = Math.min(frame, totalFrames);

    curves.forEach(curve => {
      ctx.beginPath();
      ctx.strokeStyle = curve.color;
      ctx.lineWidth = curve.width;
      ctx.lineJoin = 'round';

      for (let i = 0; i < visiblePoints; i++) {
        const x = pad.left + (i / totalFrames) * plotW;
        const normalizedY = (curve.points[i] + 0.6) / 1.6;
        const y = pad.top + plotH * (1 - normalizedY);

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    // leading dot on main curve
    if (visiblePoints > 0 && visiblePoints < totalFrames) {
      const lastIdx = visiblePoints - 1;
      const x = pad.left + (lastIdx / totalFrames) * plotW;
      const normalizedY = (curves[0].points[lastIdx] + 0.6) / 1.6;
      const y = pad.top + plotH * (1 - normalizedY);

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = curves[0].color;
      ctx.fill();

      // glow ring
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(74, 158, 255, 0.15)';
      ctx.fill();
    }
  }

  function animate() {
    if (animFrame <= totalFrames + 20) {
      draw(animFrame);
      animFrame++;
      requestAnimationFrame(animate);
    } else {
      // pause then loop
      setTimeout(() => {
        animFrame = 0;
        animate();
      }, 3000);
    }
  }

  // trigger on scroll into view
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !animStarted) {
        animStarted = true;
        resize();
        animate();
      }
    });
  }, { threshold: 0.3 });

  observer.observe(container);
}

// ---- INTERACTIVE ROBOT ARM (FABRIK MULTI-SEGMENT IK) ----
// 6-segment articulated arm with FABRIK solver, trajectory trail, joint angle readouts
export function initRobotArm() {
  const canvas = document.getElementById('heroArmCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // FABRIK chain - 6 segments, MASSIVE industrial arm
  const segLengths = [130, 110, 90, 72, 55, 40];
  const numJoints = segLengths.length + 1;
  let joints = []; // {x, y} for each joint
  let trail = []; // end effector trail
  const MAX_TRAIL = 60;

  let mouseX = 0, mouseY = 0;
  let targetX = 0, targetY = 0;
  let idleTime = 0;
  let lastMove = Date.now();

  function resize() {
    const section = canvas.closest('.hero-section');
    if (!section) return;
    canvas.width = section.offsetWidth;
    canvas.height = section.offsetHeight;
    initChain();
  }

  // base position - RIGHT side of hero, below photo area
  function getBase() {
    return { x: canvas.width * 0.72, y: canvas.height * 0.88 };
  }

  function initChain() {
    const base = getBase();
    joints = [{ x: base.x, y: base.y }];
    let angle = -Math.PI / 4; // start pointing up-right
    for (let i = 0; i < segLengths.length; i++) {
      const prev = joints[i];
      joints.push({
        x: prev.x + segLengths[i] * Math.cos(angle),
        y: prev.y + segLengths[i] * Math.sin(angle)
      });
      angle += 0.15;
    }
    trail = [];
  }

  // FABRIK solver - forward and backward reaching
  function solveFABRIK(tx, ty, iterations) {
    const base = getBase();
    const totalLen = segLengths.reduce((a, b) => a + b, 0);
    const dx = tx - base.x;
    const dy = ty - base.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    // clamp target to reachable workspace
    if (dist > totalLen * 0.95) {
      const scale = (totalLen * 0.95) / dist;
      tx = base.x + dx * scale;
      ty = base.y + dy * scale;
    }

    for (let iter = 0; iter < iterations; iter++) {
      // forward reaching (end effector to base)
      joints[numJoints - 1].x = tx;
      joints[numJoints - 1].y = ty;
      for (let i = numJoints - 2; i >= 0; i--) {
        const ddx = joints[i].x - joints[i + 1].x;
        const ddy = joints[i].y - joints[i + 1].y;
        const d = Math.sqrt(ddx * ddx + ddy * ddy) || 0.001;
        const ratio = segLengths[i] / d;
        joints[i].x = joints[i + 1].x + ddx * ratio;
        joints[i].y = joints[i + 1].y + ddy * ratio;
      }

      // backward reaching (base to end effector)
      joints[0].x = base.x;
      joints[0].y = base.y;
      for (let i = 1; i < numJoints; i++) {
        const ddx = joints[i].x - joints[i - 1].x;
        const ddy = joints[i].y - joints[i - 1].y;
        const d = Math.sqrt(ddx * ddx + ddy * ddy) || 0.001;
        const ratio = segLengths[i - 1] / d;
        joints[i].x = joints[i - 1].x + ddx * ratio;
        joints[i].y = joints[i - 1].y + ddy * ratio;
      }
    }
  }

  function lerp(a, b, t) { return a + (b - a) * t; }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // smooth follow
    targetX = lerp(targetX, mouseX, 0.05);
    targetY = lerp(targetY, mouseY, 0.05);

    // idle demo trajectory - figure-8 pattern when no mouse movement
    const now = Date.now();
    if (now - lastMove > 3000) {
      const t = now * 0.001;
      const base = getBase();
      const cx = base.x + 120;
      const cy = base.y - 60;
      targetX = cx + Math.sin(t * 0.8) * 80;
      targetY = cy + Math.sin(t * 1.6) * 40;
    }

    solveFABRIK(targetX, targetY, 5);

    // end effector trail
    const end = joints[numJoints - 1];
    trail.push({ x: end.x, y: end.y });
    if (trail.length > MAX_TRAIL) trail.shift();

    // --- THICC INDUSTRIAL ARM DRAWING ---

    // end effector trail — glowing green path
    if (trail.length > 2) {
      for (let i = 1; i < trail.length; i++) {
        const alpha = (i / trail.length) * 0.25;
        ctx.beginPath();
        ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
        ctx.lineTo(trail[i].x, trail[i].y);
        ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`;
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.stroke();
      }
    }

    // segments — thicc industrial links with metallic gradient
    for (let i = 0; i < segLengths.length; i++) {
      const a = joints[i];
      const b = joints[i + 1];
      const progress = i / segLengths.length;
      const thickness = 16 - progress * 9; // 16px base → 7px tip (MASSIVE)
      const alpha = 0.7 - progress * 0.25;

      // outer glow — atmospheric
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = `rgba(74, 158, 255, ${alpha * 0.12})`;
      ctx.lineWidth = thickness + 16;
      ctx.lineCap = 'round';
      ctx.stroke();

      // segment body — metallic gradient effect
      const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
      grad.addColorStop(0, `rgba(74, 158, 255, ${alpha})`);
      grad.addColorStop(0.5, `rgba(120, 190, 255, ${alpha * 0.9})`);
      grad.addColorStop(1, `rgba(74, 158, 255, ${alpha * 0.8})`);

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = grad;
      ctx.lineWidth = thickness;
      ctx.stroke();

      // highlight line — metallic specular
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = `rgba(180, 220, 255, ${alpha * 0.25})`;
      ctx.lineWidth = thickness * 0.3;
      ctx.stroke();
    }

    // joints — servo housings (rounded rectangles with detail)
    for (let i = 0; i < numJoints; i++) {
      const j = joints[i];
      const progress = i / (numJoints - 1);
      const size = 14 - progress * 7; // 14px base → 7px tip

      // servo housing — rounded rect
      ctx.save();
      if (i > 0) {
        const prev = joints[i - 1];
        const angle = Math.atan2(j.y - prev.y, j.x - prev.x);
        ctx.translate(j.x, j.y);
        ctx.rotate(angle);
      } else {
        ctx.translate(j.x, j.y);
      }

      // housing body
      const w = size * 2.2;
      const h = size * 1.6;
      ctx.beginPath();
      ctx.roundRect(-w / 2, -h / 2, w, h, 3);
      ctx.fillStyle = `rgba(50, 120, 200, ${0.6 - progress * 0.25})`;
      ctx.fill();
      ctx.strokeStyle = `rgba(100, 180, 255, ${0.4 - progress * 0.15})`;
      ctx.lineWidth = 1;
      ctx.stroke();

      // center bolt
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.35, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(150, 200, 255, ${0.7 - progress * 0.3})`;
      ctx.fill();

      ctx.restore();
    }

    // end effector — bright green gripper indicator
    ctx.beginPath();
    ctx.arc(end.x, end.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.95)';
    ctx.fill();
    // glow rings
    ctx.beginPath();
    ctx.arc(end.x, end.y, 14, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.15)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(end.x, end.y, 24, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.05)';
    ctx.fill();

    // --- HUD PANEL — joint angles near base, controller GUI style ---
    const base = getBase();
    const hudX = base.x - 160;
    const hudY = base.y - 20;

    // HUD background
    ctx.fillStyle = 'rgba(10, 15, 25, 0.7)';
    ctx.beginPath();
    ctx.roundRect(hudX, hudY, 140, segLengths.length * 16 + 22, 6);
    ctx.fill();
    ctx.strokeStyle = 'rgba(74, 158, 255, 0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // HUD title
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = 'rgba(74, 158, 255, 0.6)';
    ctx.textAlign = 'left';
    ctx.fillText('JOINT STATE', hudX + 8, hudY + 14);

    // joint angle readouts
    for (let i = 0; i < segLengths.length; i++) {
      const prev = joints[i];
      const curr = joints[i + 1];
      const angle = Math.atan2(curr.y - prev.y, curr.x - prev.x);
      const deg = ((angle * 180 / Math.PI + 360) % 360).toFixed(1);

      const y = hudY + 28 + i * 16;
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillStyle = 'rgba(74, 158, 255, 0.45)';
      ctx.fillText(`J${i + 1}:`, hudX + 8, y);
      ctx.fillStyle = 'rgba(74, 158, 255, 0.75)';
      ctx.fillText(`${deg}\u00B0`, hudX + 35, y);

      // mini angle bar
      const barW = 50;
      const barFill = (parseFloat(deg) % 360) / 360;
      ctx.fillStyle = 'rgba(74, 158, 255, 0.1)';
      ctx.fillRect(hudX + 78, y - 8, barW, 6);
      ctx.fillStyle = 'rgba(74, 158, 255, 0.35)';
      ctx.fillRect(hudX + 78, y - 8, barW * barFill, 6);
    }

    // workspace hint — subtle dashed circle
    const totalLen = segLengths.reduce((a, b) => a + b, 0);
    ctx.beginPath();
    ctx.arc(base.x, base.y, totalLen, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(74, 158, 255, 0.04)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 10]);
    ctx.stroke();
    ctx.setLineDash([]);

    if (!prefersReducedMotion) {
      requestAnimationFrame(draw);
    }
  }

  // mouse tracking on hero — ONLY when hero is visible
  const heroSection = canvas.closest('.hero-section');
  let heroVisible = true;

  if (heroSection) {
    heroSection.addEventListener('mousemove', e => {
      if (!heroVisible) return; // hero se scroll kar gaya toh ignore
      const rect = canvas.getBoundingClientRect();
      mouseX = e.clientX - rect.left;
      mouseY = e.clientY - rect.top;
      lastMove = Date.now();
    });

    // IntersectionObserver — jab hero viewport se baahar jaaye toh arm idle mode mein
    const heroObs = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        heroVisible = entry.isIntersecting;
        if (!heroVisible) {
          // force idle mode — figure-8 demo chalegi
          lastMove = 0;
        }
      });
    }, { threshold: 0.1 });
    heroObs.observe(heroSection);

    // idle position — upper right area
    const base = getBase();
    mouseX = base.x + 100;
    mouseY = base.y - 80;
    targetX = mouseX;
    targetY = mouseY;
  }

  resize();
  window.addEventListener('resize', resize);

  if (!prefersReducedMotion) {
    draw();
  }
}

// ---- PARTICLE SWARM RL TRAINING SIMULATION ----
export function initParticleSwarm() {
  const container = document.getElementById('particleSwarm');
  if (!container) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;display:block';
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const NUM = 45, TRAIL_LEN = 5;
  let running = false, step = 0, mouseIn = false, mouseX = 0, mouseY = 0;

  function resize() {
    canvas.width = container.getBoundingClientRect().width;
    canvas.height = 200;
  }
  resize();
  window.addEventListener('resize', resize);

  const particles = Array.from({ length: NUM }, () => ({
    x: Math.random() * canvas.width, y: Math.random() * canvas.height,
    vx: (Math.random() - 0.5) * 2, vy: (Math.random() - 0.5) * 2,
    policy: Math.random() * 0.05, trail: []
  }));

  // red (untrained) to green (trained) interpolation
  function pColor(q, alpha) {
    return `rgba(${Math.round(239 + (16 - 239) * q)},${Math.round(68 + (185 - 68) * q)},${Math.round(68 + (129 - 68) * q)},${alpha})`;
  }

  canvas.addEventListener('mouseenter', () => { mouseIn = true; });
  canvas.addEventListener('mouseleave', () => { mouseIn = false; });
  canvas.addEventListener('mousemove', e => {
    const r = canvas.getBoundingClientRect();
    mouseX = e.clientX - r.left;
    mouseY = e.clientY - r.top;
  });

  function update() {
    const rate = mouseIn ? 0.006 : 0.002; // mouse hover pe 3x training speed
    const w = canvas.width, h = canvas.height;
    particles.forEach(p => {
      p.policy = Math.min(1, p.policy + rate * (0.5 + Math.random() * 0.5));
      if (mouseIn && p.policy > 0.15) {
        // trained agents track the reward signal (cursor)
        const dx = mouseX - p.x, dy = mouseY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = p.policy * 0.08;
        p.vx += (dx / dist) * force;
        p.vy += (dy / dist) * force;
      } else {
        p.vx += (Math.random() - 0.5) * 0.3;
        p.vy += (Math.random() - 0.5) * 0.3;
      }
      p.vx *= 0.96; p.vy *= 0.96;
      const spd = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
      if (spd > 3) { p.vx *= 3 / spd; p.vy *= 3 / spd; }
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x += w; if (p.x > w) p.x -= w;
      if (p.y < 0) p.y += h; if (p.y > h) p.y -= h;
      p.trail.push({ x: p.x, y: p.y });
      if (p.trail.length > TRAIL_LEN) p.trail.shift();
    });
    step++;
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      if (p.trail.length > 1) {
        ctx.beginPath();
        ctx.moveTo(p.trail[0].x, p.trail[0].y);
        for (let i = 1; i < p.trail.length; i++) ctx.lineTo(p.trail[i].x, p.trail[i].y);
        ctx.strokeStyle = pColor(p.policy, 0.08);
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2 + p.policy, 0, Math.PI * 2);
      ctx.fillStyle = pColor(p.policy, 0.5 + p.policy * 0.3);
      ctx.fill();
    });
    // --- dashboard stats — real-time training info ---
    const avgPolicy = particles.reduce((s, p) => s + p.policy, 0) / NUM;
    const bestPolicy = Math.max(...particles.map(p => p.policy));
    const avgColor = avgPolicy > 0.5 ? 'rgba(16,185,129,0.6)' : 'rgba(239,68,68,0.5)';

    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.fillText(`agents: ${NUM}`, 8, canvas.height - 28);
    ctx.fillStyle = avgColor;
    ctx.fillText(`avg_reward: ${avgPolicy.toFixed(3)}`, 8, canvas.height - 14);
    ctx.fillStyle = 'rgba(16,185,129,0.6)';

    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.fillText(`best: ${bestPolicy.toFixed(3)}`, canvas.width - 8, canvas.height - 28);
    ctx.fillStyle = 'rgba(74,158,255,0.4)';
    ctx.fillText(`step: ${step}`, canvas.width - 8, canvas.height - 14);
  }

  function loop() {
    if (!running) return;
    update();
    draw();
    requestAnimationFrame(loop);
  }

  // sirf visible hone pe chale — performance ke liye
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !running) { running = true; loop(); }
      else if (!entry.isIntersecting) { running = false; }
    });
  }, { threshold: 0.2 });
  observer.observe(container);
}
