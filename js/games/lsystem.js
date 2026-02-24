// ============================================================
// L-System Fractal Trees — Lindenmayer System se organic plants banao
// Turtle graphics interpretation, preset rules, auto-scaling
// Nature ki language code mein — string rewriting se jungle ugao
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, pedh ugao
export function initLSystem() {
  const container = document.getElementById('lsystemContainer');
  if (!container) {
    console.warn('lsystemContainer nahi mila bhai, L-System demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';

  // --- Presets — har ek mein axiom, rules, default angle, iterations ---
  const PRESETS = {
    'Binary Tree': {
      axiom: 'F',
      rules: { F: 'F[-F]F[+F]F' },
      angle: 25,
      iterations: 4,
      segLen: 4,
      startAngle: -90, // upar ki taraf ugao
    },
    'Fern': {
      axiom: 'X',
      rules: { X: 'F+[[X]-X]-F[-FX]+X', F: 'FF' },
      angle: 25,
      iterations: 5,
      segLen: 4,
      startAngle: -90,
    },
    'Koch Snowflake': {
      axiom: 'F--F--F',
      rules: { F: 'F+F--F+F' },
      angle: 60,
      iterations: 4,
      segLen: 3,
      startAngle: 0,
    },
    'Dragon Curve': {
      axiom: 'FX',
      rules: { X: 'X+YF+', Y: '-FX-Y' },
      angle: 90,
      iterations: 10,
      segLen: 5,
      startAngle: 0,
    },
    'Sierpinski': {
      axiom: 'F-G-G',
      rules: { F: 'F-G+F+G-F', G: 'GG' },
      angle: 120,
      iterations: 6,
      segLen: 4,
      startAngle: 0,
    },
    'Bush': {
      axiom: 'F',
      rules: { F: 'FF+[+F-F-F]-[-F+F+F]' },
      angle: 22.5,
      iterations: 4,
      segLen: 5,
      startAngle: -90,
    },
  };

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let isVisible = false;
  let animFrameId = null;
  let needsRender = true;

  // current L-System configuration
  let currentPreset = 'Fern';
  let iterations = PRESETS['Fern'].iterations;
  let angle = PRESETS['Fern'].angle;
  let segLen = PRESETS['Fern'].segLen;
  let randomJitter = 0; // 0 = no jitter, zyada = zyada random
  let lString = ''; // computed L-System string
  let totalSegments = 0;

  // cached segments — [{x1,y1,x2,y2,depth}] turtle walk ka result
  let segments = [];

  // --- DOM Structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — yahan tree render hoga
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:default',
    'background:rgba(2,2,8,0.5)',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // stats bar — iteration count + total segments
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'justify-content:center',
    'gap:20px',
    'margin-top:8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:rgba(255,255,255,0.5)',
  ].join(';');
  container.appendChild(statsDiv);

  const iterLabel = document.createElement('span');
  const segLabel = document.createElement('span');
  const strLenLabel = document.createElement('span');
  statsDiv.appendChild(iterLabel);
  statsDiv.appendChild(segLabel);
  statsDiv.appendChild(strLenLabel);

  // controls row 1 — preset buttons
  const controlsDiv1 = document.createElement('div');
  controlsDiv1.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv1);

  // controls row 2 — sliders aur buttons
  const controlsDiv2 = document.createElement('div');
  controlsDiv2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:6px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv2);

  // --- Helper: button banao ---
  function makeButton(text, parent, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'border-radius:6px',
      'padding:5px 12px',
      'font-size:11px',
      'font-family:"JetBrains Mono",monospace',
      'cursor:pointer',
      'transition:all 0.2s',
      'user-select:none',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
        btn.style.color = '#e0e0e0';
      }
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
        btn.style.color = '#b0b0b0';
      }
    });
    btn.addEventListener('click', onClick);
    parent.appendChild(btn);
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
  function createSlider(label, min, max, value, step, parent, onChange) {
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
    valSpan.style.cssText = 'min-width:32px;text-align:right;color:' + ACCENT + ';';

    slider.addEventListener('input', () => {
      const val = Number(slider.value);
      valSpan.textContent = Number.isInteger(val) ? val : val.toFixed(1);
      onChange(val);
    });

    wrap.appendChild(slider);
    wrap.appendChild(valSpan);
    parent.appendChild(wrap);
    return { slider, valSpan, wrap };
  }

  // --- Preset buttons banao ---
  const presetButtons = {};
  Object.keys(PRESETS).forEach(name => {
    const btn = makeButton(name, controlsDiv1, () => {
      selectPreset(name);
    });
    presetButtons[name] = btn;
  });

  function selectPreset(name) {
    const preset = PRESETS[name];
    if (!preset) return;

    currentPreset = name;
    iterations = preset.iterations;
    angle = preset.angle;
    segLen = preset.segLen;

    // sliders update kar
    iterSlider.slider.value = iterations;
    iterSlider.valSpan.textContent = iterations;
    angleSlider.slider.value = angle;
    angleSlider.valSpan.textContent = Number.isInteger(angle) ? angle : angle.toFixed(1);
    segLenSlider.slider.value = segLen;
    segLenSlider.valSpan.textContent = segLen;

    // button highlight update kar
    Object.keys(presetButtons).forEach(k => {
      setButtonActive(presetButtons[k], k === name);
    });

    generateAndRender();
  }

  // --- Sliders banao ---
  const iterSlider = createSlider('Iter', 1, 7, iterations, 1, controlsDiv2, (val) => {
    iterations = val;
    generateAndRender();
  });

  const angleSlider = createSlider('Angle', 10, 120, angle, 0.5, controlsDiv2, (val) => {
    angle = val;
    generateAndRender();
  });

  const segLenSlider = createSlider('Len', 1, 15, segLen, 0.5, controlsDiv2, (val) => {
    segLen = val;
    generateAndRender();
  });

  // randomize button — angle mein jitter daal
  const randomBtn = makeButton('Randomize', controlsDiv2, () => {
    // jitter toggle kar — 0 se 15 degree random
    randomJitter = randomJitter > 0 ? 0 : 15;
    setButtonActive(randomBtn, randomJitter > 0);
    generateAndRender();
  });

  // grow / regenerate button
  makeButton('Grow', controlsDiv2, () => {
    generateAndRender();
  });

  // --- L-System string generation ---
  // axiom se shuru kar, rules apply kar N baar
  function generateLString() {
    const preset = PRESETS[currentPreset];
    if (!preset) return '';

    let str = preset.axiom;
    const rules = preset.rules;

    // safety check — exponential growth, max string length limit
    const MAX_STRING_LEN = 2000000;

    for (let i = 0; i < iterations; i++) {
      let next = '';
      for (let j = 0; j < str.length; j++) {
        const ch = str[j];
        if (rules[ch] !== undefined) {
          next += rules[ch];
        } else {
          next += ch;
        }
        // agar bahut bada ho gaya toh rok do — browser hang nahi hona chahiye
        if (next.length > MAX_STRING_LEN) {
          next = next.substring(0, MAX_STRING_LEN);
          break;
        }
      }
      str = next;
      if (str.length > MAX_STRING_LEN) break;
    }

    return str;
  }

  // --- Turtle graphics interpretation ---
  // L-System string se segments banao (x1,y1) -> (x2,y2) with depth info
  function interpretTurtle(str) {
    const segs = [];
    const stack = []; // push/pop ke liye — [ aur ] ke liye
    const rad = Math.PI / 180;
    const preset = PRESETS[currentPreset];
    const startAngleDeg = preset ? preset.startAngle : -90;

    // turtle state — position aur direction
    let x = 0;
    let y = 0;
    let dir = startAngleDeg * rad; // radians mein angle
    let depth = 0; // branch depth — thickness ke liye
    let maxDepth = 0;

    for (let i = 0; i < str.length; i++) {
      const ch = str[i];

      if (ch === 'F' || ch === 'G') {
        // aage badho aur line draw karo
        const jitter = randomJitter > 0 ? (Math.random() - 0.5) * randomJitter * rad : 0;
        const currentDir = dir + jitter;
        const nx = x + Math.cos(currentDir) * segLen;
        const ny = y + Math.sin(currentDir) * segLen;

        segs.push({
          x1: x, y1: y,
          x2: nx, y2: ny,
          depth: depth,
        });

        x = nx;
        y = ny;

      } else if (ch === 'f') {
        // aage badho but draw mat karo — move without drawing
        const nx = x + Math.cos(dir) * segLen;
        const ny = y + Math.sin(dir) * segLen;
        x = nx;
        y = ny;

      } else if (ch === '+') {
        // right turn — clockwise
        dir += angle * rad;

      } else if (ch === '-') {
        // left turn — counter-clockwise
        dir -= angle * rad;

      } else if (ch === '[') {
        // state save karo — branch shuru
        stack.push({ x, y, dir, depth });
        depth++;
        if (depth > maxDepth) maxDepth = depth;

      } else if (ch === ']') {
        // state restore karo — branch khatam, waapas trunk pe
        if (stack.length > 0) {
          const state = stack.pop();
          x = state.x;
          y = state.y;
          dir = state.dir;
          depth = state.depth;
        }
      }
      // X, Y jaise symbols ignore karo — drawing mein koi role nahi
    }

    // maxDepth store karo har segment ke liye normalize karne ke liye
    for (let i = 0; i < segs.length; i++) {
      segs[i].maxDepth = maxDepth;
    }

    return segs;
  }

  // --- Auto-fit: bounding box nikal ke canvas mein fit karo ---
  function computeBounds(segs) {
    if (segs.length === 0) return { minX: -1, minY: -1, maxX: 1, maxY: 1 };

    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (let i = 0; i < segs.length; i++) {
      const s = segs[i];
      if (s.x1 < minX) minX = s.x1;
      if (s.x2 < minX) minX = s.x2;
      if (s.y1 < minY) minY = s.y1;
      if (s.y2 < minY) minY = s.y2;
      if (s.x1 > maxX) maxX = s.x1;
      if (s.x2 > maxX) maxX = s.x2;
      if (s.y1 > maxY) maxY = s.y1;
      if (s.y2 > maxY) maxY = s.y2;
    }

    return { minX, minY, maxX, maxY };
  }

  // --- Color computation — depth se color decide karo ---
  // tree presets ke liye: brown trunk -> green leaves
  // geometric presets ke liye: purple gradient
  function isTreePreset() {
    return currentPreset === 'Binary Tree' ||
           currentPreset === 'Fern' ||
           currentPreset === 'Bush';
  }

  function getSegmentColor(depth, maxDepth, t) {
    // t = segment position in the overall list (0 to 1)
    if (isTreePreset()) {
      // trunk = brown, branches = dark green, tips = bright green
      const depthT = maxDepth > 0 ? Math.min(1, depth / maxDepth) : 0;

      // trunk se leaf tak interpolate karo
      // trunk: rgb(101, 67, 33) — dark brown
      // mid:   rgb(34, 100, 34) — forest green
      // tips:  rgb(50, 205, 50) — lime green (spring leaves)
      let r, g, b;

      if (depthT < 0.3) {
        // trunk zone — brown
        const f = depthT / 0.3;
        r = Math.floor(101 + (60 - 101) * f);
        g = Math.floor(67 + (90 - 67) * f);
        b = Math.floor(33 + (30 - 33) * f);
      } else if (depthT < 0.7) {
        // branch zone — brown se green
        const f = (depthT - 0.3) / 0.4;
        r = Math.floor(60 + (34 - 60) * f);
        g = Math.floor(90 + (130 - 90) * f);
        b = Math.floor(30 + (34 - 30) * f);
      } else {
        // leaf zone — green se bright green
        const f = (depthT - 0.7) / 0.3;
        r = Math.floor(34 + (80 - 34) * f);
        g = Math.floor(130 + (210 - 130) * f);
        b = Math.floor(34 + (80 - 34) * f);
      }

      return { r, g, b };
    } else {
      // geometric patterns — purple/blue gradient
      const hue = 260 + t * 60; // purple se blue
      const sat = 70 + t * 20;
      const light = 45 + t * 25;
      return hslToRgb(hue, sat, light);
    }
  }

  // HSL to RGB conversion helper
  function hslToRgb(h, s, l) {
    h = h % 360;
    s = s / 100;
    l = l / 100;

    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;

    let r = 0, g = 0, b = 0;
    if (h < 60)       { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else              { r = c; g = 0; b = x; }

    return {
      r: Math.round((r + m) * 255),
      g: Math.round((g + m) * 255),
      b: Math.round((b + m) * 255),
    };
  }

  // --- Main render function ---
  function renderTree() {
    if (canvasW <= 0 || canvasH <= 0) return;

    // canvas saaf karo
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (segments.length === 0) {
      // koi segment nahi — instructions dikhao
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
      ctx.font = '14px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Select a preset to grow a fractal', canvasW / 2, canvasH / 2);
      ctx.restore();
      return;
    }

    // bounding box se transform compute karo — auto-fit
    const bounds = computeBounds(segments);
    const bw = bounds.maxX - bounds.minX;
    const bh = bounds.maxY - bounds.minY;

    // padding rakh do edges pe
    const padX = canvasW * 0.08;
    const padY = canvasH * 0.08;
    const availW = canvasW - padX * 2;
    const availH = canvasH - padY * 2;

    // scale compute karo — aspect ratio maintain karo
    let scale;
    if (bw <= 0 && bh <= 0) {
      scale = 1;
    } else if (bw <= 0) {
      scale = availH / bh;
    } else if (bh <= 0) {
      scale = availW / bw;
    } else {
      scale = Math.min(availW / bw, availH / bh);
    }

    // center compute karo
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    const offsetX = canvasW / 2 - centerX * scale;
    const offsetY = canvasH / 2 - centerY * scale;

    // maxDepth nikal lo for color computation
    let maxDepth = 0;
    for (let i = 0; i < segments.length; i++) {
      if (segments[i].maxDepth > maxDepth) maxDepth = segments[i].maxDepth;
    }

    // --- Segments render karo — thick trunk, thin branches ---
    // pehle thicker (low depth) segments draw karo, fir thin (high depth)
    // isse branches trunk ke upar dikhein
    const sortedSegs = segments.slice().sort((a, b) => a.depth - b.depth);

    // base thickness — scale ke hisaab se adjust karo
    const baseThickness = Math.max(1, Math.min(6, scale * segLen * 0.35));

    for (let i = 0; i < sortedSegs.length; i++) {
      const seg = sortedSegs[i];
      const depthT = maxDepth > 0 ? seg.depth / maxDepth : 0;
      const t = i / sortedSegs.length; // position in list

      // transformed coordinates
      const x1 = seg.x1 * scale + offsetX;
      const y1 = seg.y1 * scale + offsetY;
      const x2 = seg.x2 * scale + offsetX;
      const y2 = seg.y2 * scale + offsetY;

      // thickness — trunk thick, branches thin
      // exponential decay for more natural look
      const thickness = baseThickness * Math.pow(0.55, seg.depth);
      const lineW = Math.max(0.5, thickness);

      // color nikal
      const col = getSegmentColor(seg.depth, maxDepth, t);

      // alpha — deeper branches thodi transparent
      const alpha = Math.max(0.5, 1.0 - depthT * 0.3);

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = 'rgba(' + col.r + ',' + col.g + ',' + col.b + ',' + alpha + ')';
      ctx.lineWidth = lineW;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    // --- Leaf tip glow effect — sirf tree presets ke liye ---
    if (isTreePreset() && maxDepth > 0) {
      // leaf tips = maximum depth wale segments ke end points
      // unhe identify karo — wo segments jinke end point pe koi aur segment start nahi hota
      // simplified: just use max depth segments

      ctx.save();

      for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        if (seg.depth < maxDepth - 1) continue; // sirf tips pe glow

        const x2 = seg.x2 * scale + offsetX;
        const y2 = seg.y2 * scale + offsetY;

        // soft radial glow
        const glowRadius = Math.max(3, baseThickness * 1.5);
        const gradient = ctx.createRadialGradient(x2, y2, 0, x2, y2, glowRadius);
        gradient.addColorStop(0, 'rgba(80,220,80,0.5)');
        gradient.addColorStop(0.4, 'rgba(50,205,50,0.2)');
        gradient.addColorStop(1, 'rgba(50,205,50,0)');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x2, y2, glowRadius, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();
    }

    // --- Geometric patterns ke liye glow ---
    if (!isTreePreset()) {
      ctx.save();
      ctx.globalCompositeOperation = 'screen';

      // poore tree pe subtle purple glow
      for (let i = 0; i < segments.length; i += Math.max(1, Math.floor(segments.length / 500))) {
        const seg = segments[i];
        const x2 = seg.x2 * scale + offsetX;
        const y2 = seg.y2 * scale + offsetY;
        const t = i / segments.length;

        const glowR = Math.max(2, baseThickness * 0.8);
        const gradient = ctx.createRadialGradient(x2, y2, 0, x2, y2, glowR);
        const col = getSegmentColor(seg.depth, maxDepth, t);
        gradient.addColorStop(0, 'rgba(' + col.r + ',' + col.g + ',' + col.b + ',0.3)');
        gradient.addColorStop(1, 'rgba(' + col.r + ',' + col.g + ',' + col.b + ',0)');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x2, y2, glowR, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();
    }

    ctx.restore();
  }

  // --- Generate L-System string + segments, then render ---
  function generateAndRender() {
    // L-System string generate karo
    lString = generateLString();

    // turtle walk se segments banao
    segments = interpretTurtle(lString);
    totalSegments = segments.length;

    // stats update karo
    updateStats();

    // render schedule karo
    needsRender = true;
    scheduleFrame();
  }

  // --- Stats update ---
  function updateStats() {
    iterLabel.textContent = 'Iter: ' + iterations;
    segLabel.textContent = 'Segments: ' + totalSegments.toLocaleString();
    strLenLabel.textContent = 'String: ' + lString.length.toLocaleString();

    // segments count ko accent color do agar kuch hai
    segLabel.style.color = totalSegments > 0 ? ACCENT : 'rgba(255,255,255,0.5)';
  }

  // --- Canvas sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = Math.floor(containerWidth * dpr);
    canvas.height = Math.floor(CANVAS_HEIGHT * dpr);
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    // rendering zaruri hai resize ke baad
    needsRender = true;
    scheduleFrame();
  }

  // --- Animation / render scheduling ---
  function scheduleFrame() {
    if (animFrameId || !isVisible) return;
    animFrameId = requestAnimationFrame(() => {
      animFrameId = null;
      if (needsRender) {
        needsRender = false;
        renderTree();
      }
    });
  }

  // --- IntersectionObserver — sirf visible hone pe render karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    needsRender = true;
    scheduleFrame();
  }

  function stopAnimation() {
    isVisible = false;
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
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

  // tab switch pe pause/resume
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // resize handle karo
  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas();
  });
  resizeObserver.observe(container);

  // --- Init ---
  resizeCanvas();
  // default preset select karo — Fern se shuru karenge, sab se accha dikhta hai
  selectPreset('Fern');
}
