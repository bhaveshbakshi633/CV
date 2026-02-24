// ============================================================
// String Art / Cardioid Patterns — Mathematical beauty pure form mein
// Numbered pins around shapes, multiplication rule se connect karo
// Envelope of lines creates stunning curves — cardioid, nephroid, etc.
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, strings draw karo
export function initStringArt() {
  const container = document.getElementById('stringArtContainer');
  if (!container) {
    console.warn('stringArtContainer nahi mila bhai, String Art demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let isVisible = false;
  let animFrameId = null;

  // string art parameters — yahi control karenge pattern
  let pinCount = 200;
  let multiplier = 2.0;
  let colorMode = 'rainbow'; // solid, rainbow, gradient
  let shape = 'circle';      // circle, square, triangle
  let lineOpacity = 0.35;
  let showPins = true;
  let animateMultiplier = true;
  let animSpeed = 0.003; // multiplier increment per frame

  // animation state — multiplier ko slowly increase karo
  let animMultiplier = 2.0;
  let animDirection = 1; // +1 badhao, -1 ghatao

  // --- DOM structure banao ---
  // pehle se jo bhi children hain unhe preserve karo nahi... hata do aur fresh banao
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — yahan string art dikhega, black background for max contrast
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:default',
    'background:#000',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // info bar — current multiplier aur pattern info
  const infoBar = document.createElement('div');
  infoBar.style.cssText = [
    'display:flex',
    'justify-content:center',
    'gap:20px',
    'margin-top:8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:rgba(' + ACCENT_RGB + ',0.6)',
    'min-height:18px',
    'flex-wrap:wrap',
  ].join(';');
  container.appendChild(infoBar);

  const multLabel = document.createElement('span');
  const patternLabel = document.createElement('span');
  const lineCountLabel = document.createElement('span');
  infoBar.appendChild(multLabel);
  infoBar.appendChild(patternLabel);
  infoBar.appendChild(lineCountLabel);

  // controls container
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv);

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
    valSpan.textContent = Number.isInteger(Number(value)) ? value : Number(value).toFixed(1);
    valSpan.style.cssText = 'min-width:32px;text-align:right;color:' + ACCENT + ';';

    slider.addEventListener('input', () => {
      const val = Number(slider.value);
      valSpan.textContent = Number.isInteger(val) ? val : val.toFixed(1);
      onChange(val);
    });

    wrap.appendChild(slider);
    wrap.appendChild(valSpan);
    controlsDiv.appendChild(wrap);
    return { slider, valSpan };
  }

  // --- Helper: select dropdown banao ---
  function createSelect(label, options, selected, onChange) {
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

    const sel = document.createElement('select');
    sel.style.cssText = [
      'padding:4px 8px',
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

    wrap.appendChild(sel);
    controlsDiv.appendChild(wrap);
    return sel;
  }

  // --- Helper: toggle button banao ---
  function createToggle(text, initial, onToggle) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn._active = initial;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:11px',
      'border-radius:6px',
      'cursor:pointer',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s',
      'user-select:none',
      'white-space:nowrap',
    ].join(';');

    function updateStyle() {
      if (btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.35)';
        btn.style.color = ACCENT;
        btn.style.border = '1px solid ' + ACCENT;
      } else {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
        btn.style.color = '#b0b0b0';
        btn.style.border = '1px solid rgba(' + ACCENT_RGB + ',0.25)';
      }
    }
    updateStyle();

    btn.addEventListener('mouseenter', function() {
      if (!btn._active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
        btn.style.color = '#e0e0e0';
      }
    });
    btn.addEventListener('mouseleave', function() {
      updateStyle();
    });
    btn.addEventListener('click', function() {
      btn._active = !btn._active;
      updateStyle();
      onToggle(btn._active);
    });

    controlsDiv.appendChild(btn);
    return btn;
  }

  // --- Controls banao ---
  // pin count slider — 20 se 360
  const pinSlider = createSlider('Pins', 20, 360, pinCount, 1, function(val) {
    pinCount = val;
  });

  // multiplier slider — 2 se 100, float step 0.1
  const multSlider = createSlider('Mult', 2, 100, multiplier, 0.1, function(val) {
    multiplier = val;
    // agar animate off hai toh manual value use karo
    if (!animateMultiplier) {
      animMultiplier = val;
    }
  });

  // line opacity slider — 0.1 se 0.8
  createSlider('Opacity', 0.1, 0.8, lineOpacity, 0.05, function(val) {
    lineOpacity = val;
  });

  // color mode dropdown
  createSelect('Color', [
    { value: 'rainbow', label: 'Rainbow' },
    { value: 'solid', label: 'Solid' },
    { value: 'gradient', label: 'Gradient' },
  ], colorMode, function(val) {
    colorMode = val;
  });

  // shape dropdown
  createSelect('Shape', [
    { value: 'circle', label: 'Circle' },
    { value: 'square', label: 'Square' },
    { value: 'triangle', label: 'Triangle' },
  ], shape, function(val) {
    shape = val;
  });

  // animate multiplier toggle — yahi asli magic hai
  const animToggle = createToggle('Animate', animateMultiplier, function(active) {
    animateMultiplier = active;
    if (active) {
      // animation shuru — current multiplier se
      animMultiplier = multiplier;
    }
  });

  // show pins toggle
  createToggle('Pins', showPins, function(active) {
    showPins = active;
  });

  // --- Pin position computation ---
  // shape ke hisaab se N pins ki positions return karo
  function computePinPositions(n, cx, cy, radius) {
    var positions = [];

    if (shape === 'circle') {
      // classic circle — pins evenly spaced on circumference
      for (var i = 0; i < n; i++) {
        // top se shuru karo (-PI/2), clockwise
        var angle = (i / n) * Math.PI * 2 - Math.PI / 2;
        positions.push({
          x: cx + Math.cos(angle) * radius,
          y: cy + Math.sin(angle) * radius,
        });
      }
    } else if (shape === 'square') {
      // pins along square perimeter — evenly distributed on all 4 sides
      var half = radius; // half side length
      var perimeter = half * 8; // total perimeter = 4 * (2*half)
      for (var i = 0; i < n; i++) {
        var t = (i / n) * perimeter;
        var side = half * 2; // ek side ki length
        var px, py;

        if (t < side) {
          // top side — left to right
          px = cx - half + t;
          py = cy - half;
        } else if (t < side * 2) {
          // right side — top to bottom
          px = cx + half;
          py = cy - half + (t - side);
        } else if (t < side * 3) {
          // bottom side — right to left
          px = cx + half - (t - side * 2);
          py = cy + half;
        } else {
          // left side — bottom to top
          px = cx - half;
          py = cy + half - (t - side * 3);
        }

        positions.push({ x: px, y: py });
      }
    } else if (shape === 'triangle') {
      // equilateral triangle — pins along 3 edges
      var h = radius * Math.sqrt(3) / 2; // triangle height adjustment
      // 3 vertices — top, bottom-left, bottom-right
      var v0x = cx, v0y = cy - radius;                          // top
      var v1x = cx - radius * Math.sin(Math.PI * 2 / 3), v1y = cy - radius * Math.cos(Math.PI * 2 / 3); // bottom-left
      var v2x = cx - radius * Math.sin(Math.PI * 4 / 3), v2y = cy - radius * Math.cos(Math.PI * 4 / 3); // bottom-right
      var vertices = [
        { x: v0x, y: v0y },
        { x: v1x, y: v1y },
        { x: v2x, y: v2y },
      ];

      // har side pe roughly equal pins
      for (var i = 0; i < n; i++) {
        var t = (i / n) * 3; // 0-3, har integer ek side
        var sideIdx = Math.floor(t) % 3;
        var frac = t - Math.floor(t);
        var a = vertices[sideIdx];
        var b = vertices[(sideIdx + 1) % 3];

        positions.push({
          x: a.x + (b.x - a.x) * frac,
          y: a.y + (b.y - a.y) * frac,
        });
      }
    }

    return positions;
  }

  // --- HSL to RGB string helper ---
  function hslStr(h, s, l, a) {
    return 'hsla(' + (h % 360) + ',' + s + '%,' + l + '%,' + a + ')';
  }

  // --- Get pattern name from multiplier ---
  function getPatternName(mult) {
    var rounded = Math.round(mult * 10) / 10;
    if (rounded === 2) return 'Cardioid';
    if (rounded === 3) return 'Nephroid';
    if (rounded === 4) return '3-Cusp Epicycloid';
    if (rounded === 5) return '4-Cusp Epicycloid';
    if (rounded >= 6 && rounded <= 10) return (rounded - 1).toFixed(0) + '-Cusp Epicycloid';
    if (rounded > 10) return 'Complex Pattern';
    return 'Pattern';
  }

  // --- Main render function ---
  function render() {
    if (canvasW <= 0 || canvasH <= 0) return;

    // canvas saaf karo — pure black background for max contrast
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // center aur radius compute karo — padding ke saath
    var cx = canvasW / 2;
    var cy = canvasH / 2;
    var radius = Math.min(canvasW, canvasH) * 0.42; // thoda margin chhodo

    // current multiplier decide karo — animated ya manual
    var currentMult = animateMultiplier ? animMultiplier : multiplier;

    // pin positions compute karo
    var pins = computePinPositions(pinCount, cx, cy, radius);

    // --- Strings draw karo — yahi asli beauty hai ---
    // additive blending for glowing effect — overlapping lines brighter dikhein
    ctx.globalCompositeOperation = 'lighter';

    var lineCount = 0;

    for (var i = 0; i < pinCount; i++) {
      // destination pin = (i * multiplier) mod N
      var destFloat = (i * currentMult) % pinCount;
      var dest = Math.floor(destFloat) % pinCount;

      // agar same pin hai toh skip — self-loop ka koi matlab nahi
      if (dest === i) continue;

      var from = pins[i];
      var to = pins[dest];

      // color decide karo mode ke hisaab se
      var strokeColor;

      if (colorMode === 'rainbow') {
        // hue based on pin index — full rainbow cycle
        var hue = (i / pinCount) * 360;
        strokeColor = hslStr(hue, 85, 55, lineOpacity);
      } else if (colorMode === 'solid') {
        // accent purple — single color
        strokeColor = 'rgba(' + ACCENT_RGB + ',' + lineOpacity + ')';
      } else if (colorMode === 'gradient') {
        // color based on string length/angle — warm to cool
        var dx = to.x - from.x;
        var dy = to.y - from.y;
        var angle = Math.atan2(dy, dx);
        // angle se hue map karo — -PI to PI -> 200 to 340 (cool blue to warm pink)
        var hue = 200 + ((angle + Math.PI) / (Math.PI * 2)) * 140;
        strokeColor = hslStr(hue, 75, 50, lineOpacity);
      }

      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 0.8;
      ctx.stroke();

      lineCount++;
    }

    // blending normal pe waapas lao — pins aur text ke liye
    ctx.globalCompositeOperation = 'source-over';

    // --- Pin markers draw karo ---
    if (showPins) {
      for (var i = 0; i < pins.length; i++) {
        var pin = pins[i];
        ctx.beginPath();
        ctx.arc(pin.x, pin.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fill();
      }
    }

    // --- Info bar update karo ---
    multLabel.textContent = 'Multiplier: ' + currentMult.toFixed(1);
    multLabel.style.color = ACCENT;
    patternLabel.textContent = getPatternName(currentMult);
    lineCountLabel.textContent = 'Lines: ' + lineCount;

    // multiplier slider ko sync karo agar animate on hai
    if (animateMultiplier) {
      multSlider.slider.value = currentMult.toFixed(1);
      multSlider.valSpan.textContent = currentMult.toFixed(1);
    }
  }

  // --- Animation loop ---
  function loop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animFrameId = null; return; }
    if (!isVisible) {
      animFrameId = null;
      return;
    }

    // multiplier animate karo — slowly increase, bounce back
    if (animateMultiplier) {
      animMultiplier += animSpeed * animDirection;

      // bounce between 2 and 50 — mesmerizing range
      if (animMultiplier >= 50) {
        animMultiplier = 50;
        animDirection = -1;
      } else if (animMultiplier <= 2) {
        animMultiplier = 2;
        animDirection = 1;
      }
    }

    render();
    animFrameId = requestAnimationFrame(loop);
  }

  // --- Canvas sizing — DPR aware ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    var containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = Math.floor(containerWidth * dpr);
    canvas.height = Math.floor(CANVAS_HEIGHT * dpr);
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // --- IntersectionObserver — sirf visible hone pe render karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    if (!animFrameId) {
      loop();
    }
  }

  function stopAnimation() {
    isVisible = false;
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  }

  var observer = new IntersectionObserver(
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
  document.addEventListener('lab:resume', () => { if (isVisible && !animFrameId) loop(); });

  // tab switch pe pause/resume
  document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
      stopAnimation();
    } else {
      var rect = container.getBoundingClientRect();
      var inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // resize handle karo
  window.addEventListener('resize', function() {
    resizeCanvas();
  });

  // --- Init ---
  resizeCanvas();
  // initial render — animation shuru hone se pehle ek frame dikha do
  render();
}
