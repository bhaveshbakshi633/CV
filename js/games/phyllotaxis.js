// ============================================================
// Phyllotaxis / Golden Angle Spiral — Nature ka sabse efficient packing
// Sunflower seeds ka pattern: why 137.507° is special
// Golden ratio, Fibonacci spirals, aur optimal packing ka visual proof
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, sunflower ugao
export function initPhyllotaxis() {
  const container = document.getElementById('phyllotaxisContainer');
  if (!container) {
    console.warn('phyllotaxisContainer nahi mila bhai, phyllotaxis demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';
  // golden angle — 360 / phi^2 = 137.50776... degrees
  const GOLDEN_ANGLE = 137.50776405003785;
  const DEG_TO_RAD = Math.PI / 180;

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let isVisible = false;
  let animFrameId = null;

  // spiral parameters — user control karega inhe
  let divergenceAngle = GOLDEN_ANGLE; // degrees mein
  let seedCount = 500;
  let seedSize = 3.5;
  let colorMode = 'rainbow'; // rainbow | radius | spiral | solid
  let animateAngle = false; // angle sweep animation toggle
  let animAngleValue = 100; // sweep shuru yahaan se karega
  let animDirection = 1; // 1 = badhao, -1 = ghataao
  let animSpeed = 0.15; // degrees per frame — smooth sweep

  // growth animation state
  let visibleSeeds = 0; // kitne seeds abhi dikhenge
  let growthRate = 8; // per frame kitne naye seeds aayenge
  let isGrowing = true; // kya abhi growth ho rahi hai

  // precomputed seeds array — [{x, y, n, r, angle}]
  let seeds = [];

  // --- DOM Structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — seeds yahan dikhenge
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:default',
    'background:#0a0a0a',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // stats bar — angle value + Fibonacci spiral count dikhayega
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'justify-content:center',
    'gap:18px',
    'margin-top:8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:rgba(255,255,255,0.5)',
    'flex-wrap:wrap',
  ].join(';');
  container.appendChild(statsDiv);

  const angleLabel = document.createElement('span');
  const seedsLabel = document.createElement('span');
  const spiralLabel = document.createElement('span');
  statsDiv.appendChild(angleLabel);
  statsDiv.appendChild(seedsLabel);
  statsDiv.appendChild(spiralLabel);

  // controls row 1 — sliders
  const controlsDiv1 = document.createElement('div');
  controlsDiv1.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv1);

  // controls row 2 — dropdown, buttons
  const controlsDiv2 = document.createElement('div');
  controlsDiv2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
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
      'white-space:nowrap',
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
      'width:100px',
      'accent-color:' + ACCENT,
      'cursor:pointer',
    ].join(';');

    const valSpan = document.createElement('span');
    valSpan.style.cssText = 'min-width:42px;text-align:right;color:' + ACCENT + ';';
    // initial value format
    const numVal = Number(value);
    if (step < 0.01) {
      valSpan.textContent = numVal.toFixed(3);
    } else if (step < 1) {
      valSpan.textContent = numVal.toFixed(1);
    } else {
      valSpan.textContent = numVal;
    }

    slider.addEventListener('input', () => {
      const val = Number(slider.value);
      if (step < 0.01) {
        valSpan.textContent = val.toFixed(3);
      } else if (step < 1) {
        valSpan.textContent = val.toFixed(1);
      } else {
        valSpan.textContent = val;
      }
      onChange(val);
    });

    wrap.appendChild(slider);
    wrap.appendChild(valSpan);
    parent.appendChild(wrap);
    return { slider, valSpan, wrap };
  }

  // --- Helper: select dropdown banao ---
  function createSelect(options, selected, parent, onChange) {
    const sel = document.createElement('select');
    sel.style.cssText = [
      'padding:5px 8px',
      'font-size:11px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'font-family:"JetBrains Mono",monospace',
    ].join(';');
    options.forEach(function(opt) {
      const o = document.createElement('option');
      o.value = opt.value;
      o.textContent = opt.label;
      o.style.background = '#1a1a2e';
      if (opt.value === selected) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('change', function() { onChange(sel.value); });
    parent.appendChild(sel);
    return sel;
  }

  // --- Controls banao ---

  // angle slider — 0 to 360 degrees, default golden angle
  const angleSld = createSlider('Angle°', 0, 360, divergenceAngle, 0.001, controlsDiv1, function(val) {
    divergenceAngle = val;
    if (!animateAngle) {
      // manual angle change — seeds recalculate karo, growth reset karo
      computeSeeds();
      visibleSeeds = seedCount; // manual change pe saare seeds dikhao
      isGrowing = false;
    }
  });

  // seed count slider
  const countSld = createSlider('Seeds', 50, 2000, seedCount, 1, controlsDiv1, function(val) {
    seedCount = Math.round(val);
    computeSeeds();
    // agar growth chal rahi thi toh continue karo, nahi toh sabhi dikhao
    if (!isGrowing) {
      visibleSeeds = seedCount;
    }
  });

  // seed size slider
  const sizeSld = createSlider('Size', 1, 8, seedSize, 0.5, controlsDiv1, function(val) {
    seedSize = val;
  });

  // color mode dropdown
  createSelect(
    [
      { value: 'rainbow', label: 'Rainbow' },
      { value: 'radius', label: 'Radius' },
      { value: 'spiral', label: 'Spiral Arms' },
      { value: 'solid', label: 'Solid' },
    ],
    colorMode,
    controlsDiv2,
    function(val) {
      colorMode = val;
    }
  );

  // animate angle toggle button
  const animBtn = makeButton('Animate Angle', controlsDiv2, function() {
    animateAngle = !animateAngle;
    setButtonActive(animBtn, animateAngle);
    if (animateAngle) {
      // sweep 100 se 180 ke beech karega — golden angle ko cross karega
      animAngleValue = 100;
      animDirection = 1;
    }
  });

  // regrow button — growth animation dubara shuru karo
  makeButton('Regrow', controlsDiv2, function() {
    visibleSeeds = 0;
    isGrowing = true;
    computeSeeds();
  });

  // golden angle button — seedha golden angle pe snap karo
  makeButton('Golden φ', controlsDiv2, function() {
    divergenceAngle = GOLDEN_ANGLE;
    angleSld.slider.value = GOLDEN_ANGLE;
    angleSld.valSpan.textContent = GOLDEN_ANGLE.toFixed(3);
    animateAngle = false;
    setButtonActive(animBtn, false);
    computeSeeds();
    visibleSeeds = 0;
    isGrowing = true;
  });

  // --- Phyllotaxis core algorithm ---
  // har seed ke liye polar coordinates compute karo
  function computeSeeds() {
    seeds = [];
    // scaling constant — canvas mein fit karne ke liye
    // radius = c * sqrt(n), last seed ka radius canvas ke half se thoda chhota hona chahiye
    const maxRadius = Math.min(canvasW, canvasH) * 0.45;
    // c = maxRadius / sqrt(seedCount)
    const c = seedCount > 0 ? maxRadius / Math.sqrt(seedCount) : 1;
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const angleRad = divergenceAngle * DEG_TO_RAD;

    for (let n = 0; n < seedCount; n++) {
      const theta = n * angleRad;
      const r = c * Math.sqrt(n);
      const x = cx + r * Math.cos(theta);
      const y = cy + r * Math.sin(theta);
      seeds.push({ x: x, y: y, n: n, r: r, angle: theta });
    }
  }

  // --- HSL to RGB helper — color modes ke liye ---
  function hslToRgb(h, s, l) {
    h = ((h % 360) + 360) % 360;
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

  // --- Fibonacci spiral arm detection ---
  // golden angle pe seeds Fibonacci spiral arms mein arrange hote hain
  // arm number = n mod fibNumber, aur different fib numbers different spirals dikhate hain
  // hum 13 spiral arms use karenge — sunflower mein commonly visible
  function getSpiralArm(n) {
    // 13 clockwise spirals — ye Fibonacci number hai
    // agar angle golden hai toh ye clearly dikhenge
    return n % 13;
  }

  // --- Seed color compute karo based on color mode ---
  function getSeedColor(seed, maxR) {
    const n = seed.n;
    const r = seed.r;

    if (colorMode === 'rainbow') {
      // hue = n * golden_angle mod 360 — spiral structure dikhata hai
      // ye isiliye kaam karta hai kyunki golden angle irrational hai
      const hue = (n * 137.5) % 360;
      const rgb = hslToRgb(hue, 75, 55);
      return 'rgb(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ')';
    }

    if (colorMode === 'radius') {
      // center se bahar gradient — purple se cyan
      const t = maxR > 0 ? r / maxR : 0;
      const hue = 270 - t * 120; // purple (270) se cyan/green (150)
      const sat = 70 + t * 20;
      const light = 35 + t * 25;
      const rgb = hslToRgb(hue, sat, light);
      return 'rgb(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ')';
    }

    if (colorMode === 'spiral') {
      // Fibonacci spiral arms — har arm ko alag color do
      const arm = getSpiralArm(n);
      const hue = (arm / 13) * 360;
      const rgb = hslToRgb(hue, 80, 55);
      return 'rgb(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ')';
    }

    // solid — accent color
    return ACCENT;
  }

  // --- Count visible spiral arms — educational info ke liye ---
  // approximate method: angle ke basis pe spoke pattern detect karo
  function countSpiralArms() {
    // golden angle pe ye Fibonacci numbers return karega
    // doosre angles pe ye clearly spokes count karega
    if (seeds.length < 50) return '—';

    // divergence angle ko rational approximation se check karo
    // agar angle = 360 * p/q hai (rational) toh exactly q spokes dikhenge
    // golden angle irrational hai toh Fibonacci spirals dikhte hain

    // simple heuristic: angular gaps count karo outer seeds mein
    // last 100 seeds le lo, unke angles sort karo, largest gaps count karo
    const outerSeeds = seeds.slice(Math.max(0, seeds.length - 100));
    if (outerSeeds.length < 10) return '—';

    // angles mod 2*PI nikal ke sort karo
    const TWO_PI = Math.PI * 2;
    const angles = outerSeeds.map(function(s) {
      return ((s.angle % TWO_PI) + TWO_PI) % TWO_PI;
    }).sort(function(a, b) { return a - b; });

    // consecutive angular gaps nikal
    const gaps = [];
    for (let i = 1; i < angles.length; i++) {
      gaps.push(angles[i] - angles[i - 1]);
    }
    // wrap-around gap bhi add karo
    gaps.push(TWO_PI - angles[angles.length - 1] + angles[0]);

    // median gap nikal — agar sab gaps similar hain toh uniform distribution hai
    gaps.sort(function(a, b) { return a - b; });
    const medianGap = gaps[Math.floor(gaps.length / 2)];

    // "spoke" tab count hota hai jab gap median se bahut bada ho
    // golden angle pe ye nahi hoga, doosre angles pe hoga
    let spokeCount = 0;
    for (let i = 0; i < gaps.length; i++) {
      if (gaps[i] > medianGap * 2.5) spokeCount++;
    }

    if (spokeCount > 0) {
      return spokeCount + ' spokes';
    }

    // golden angle ke paas — Fibonacci spirals report karo
    // difference from golden angle check karo
    const diff = Math.abs(divergenceAngle - GOLDEN_ANGLE);
    if (diff < 0.5) {
      return '8, 13, 21 spirals (Fib!)';
    }
    return 'no clear spokes';
  }

  // --- Stats update ---
  function updateStats() {
    const diff = Math.abs(divergenceAngle - GOLDEN_ANGLE);
    let angleText = 'Angle: ' + divergenceAngle.toFixed(3) + '°';
    if (diff < 0.01) {
      angleText += ' ≈ Golden!';
      angleLabel.style.color = ACCENT;
    } else {
      angleLabel.style.color = 'rgba(255,255,255,0.5)';
    }
    angleLabel.textContent = angleText;

    const shown = Math.min(visibleSeeds, seedCount);
    seedsLabel.textContent = 'Seeds: ' + shown + ' / ' + seedCount;

    spiralLabel.textContent = countSpiralArms();
  }

  // --- Main render function ---
  function render() {
    if (canvasW <= 0 || canvasH <= 0) return;

    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // background clear — #0a0a0a
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvasW, canvasH);

    if (seeds.length === 0) {
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
      ctx.font = '14px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Computing seeds...', canvasW / 2, canvasH / 2);
      ctx.restore();
      return;
    }

    // maxR nikal — color gradient ke liye
    const maxR = seeds.length > 0 ? seeds[seeds.length - 1].r : 1;
    const shown = Math.min(visibleSeeds, seeds.length);

    // --- Seeds draw karo ---
    for (let i = 0; i < shown; i++) {
      const seed = seeds[i];
      const color = getSeedColor(seed, maxR);

      // seed size — thoda radius dependent bana sakte hain
      // center mein chhote, bahar bade — ya constant
      const t = maxR > 0 ? seed.r / maxR : 0;
      const radius = seedSize * (0.6 + 0.4 * t); // center mein 60%, bahar 100%

      // glow / shadow effect — subtle depth ke liye
      // sirf larger seeds pe, performance ke liye
      if (radius > 2) {
        ctx.beginPath();
        ctx.arc(seed.x, seed.y, radius + 1.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.4)';
        ctx.fill();
      }

      // main seed circle
      ctx.beginPath();
      ctx.arc(seed.x, seed.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // subtle highlight — top-left pe ek chhota sa bright spot
      if (radius > 2.5) {
        ctx.beginPath();
        ctx.arc(seed.x - radius * 0.25, seed.y - radius * 0.25, radius * 0.35, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.15)';
        ctx.fill();
      }
    }

    // --- Center marker — chhota sa dot ---
    ctx.beginPath();
    ctx.arc(canvasW / 2, canvasH / 2, 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
    ctx.fill();

    // --- Golden angle indicator — agar angle golden ke paas hai toh subtle glow ---
    const diff = Math.abs(divergenceAngle - GOLDEN_ANGLE);
    if (diff < 1.0 && shown > 100) {
      // subtle radial glow center se — golden packing ka visual cue
      const glowIntensity = Math.max(0, 1 - diff / 1.0) * 0.08;
      const gradient = ctx.createRadialGradient(
        canvasW / 2, canvasH / 2, 0,
        canvasW / 2, canvasH / 2, Math.min(canvasW, canvasH) * 0.45
      );
      gradient.addColorStop(0, 'rgba(' + ACCENT_RGB + ',' + glowIntensity + ')');
      gradient.addColorStop(1, 'rgba(' + ACCENT_RGB + ',0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvasW, canvasH);
    }

    ctx.restore();
    updateStats();
  }

  // --- Animation loop ---
  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animFrameId = null; return; }
    if (!isVisible) return;

    // growth animation — seeds ek ek karke dikhao
    if (isGrowing && visibleSeeds < seedCount) {
      visibleSeeds = Math.min(visibleSeeds + growthRate, seedCount);
      if (visibleSeeds >= seedCount) {
        isGrowing = false;
      }
    }

    // angle sweep animation — slowly angle change karo
    if (animateAngle) {
      animAngleValue += animSpeed * animDirection;

      // 100 se 180 ke beech sweep karo — golden angle 137.5 is in this range
      if (animAngleValue >= 180) {
        animAngleValue = 180;
        animDirection = -1;
      } else if (animAngleValue <= 100) {
        animAngleValue = 100;
        animDirection = 1;
      }

      divergenceAngle = animAngleValue;

      // slider bhi sync karo
      angleSld.slider.value = divergenceAngle;
      angleSld.valSpan.textContent = divergenceAngle.toFixed(3);

      // seeds recompute — angle change hua hai
      computeSeeds();
      // sweep mein sabhi seeds dikhao, growth nahi
      visibleSeeds = seedCount;
      isGrowing = false;
    }

    render();
    animFrameId = requestAnimationFrame(animate);
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

    // seeds recompute — canvas size change hua toh positions update karo
    computeSeeds();
  }

  // --- IntersectionObserver — sirf visible hone pe animate karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    animFrameId = requestAnimationFrame(animate);
  }

  function stopAnimation() {
    isVisible = false;
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  }

  const observer = new IntersectionObserver(
    function(entries) {
      entries.forEach(function(entry) {
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
  document.addEventListener('lab:resume', () => { if (isVisible && !animFrameId) animate(); });

  // tab switch pe pause/resume — battery bachao
  document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // resize handle karo
  const resizeObserver = new ResizeObserver(function() {
    resizeCanvas();
  });
  resizeObserver.observe(container);

  // --- Init ---
  resizeCanvas();
  // golden angle se shuru — sunflower pattern, rainbow coloring, growth animation
  visibleSeeds = 0;
  isGrowing = true;
  computeSeeds();
}
