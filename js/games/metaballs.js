// ============================================================
// Metaballs — Organic blobs that merge smoothly
// Field function f(x,y) = Σ(rᵢ² / ((x-xᵢ)² + (y-yᵢ)²))
// Jab blobs paas aate hain toh organic merge hota hai
// Two modes: Fill (glow) aur Contour (marching squares)
// ============================================================

// yahi function export hoga — container dhundho, canvas banao, blobs bounce kara do
export function initMetaballs() {
  const container = document.getElementById('metaballsContainer');
  if (!container) {
    console.warn('metaballsContainer nahi mila bhai, metaballs skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';
  const MAX_BALLS = 15;
  // field compute karne ki resolution — performance ke liye reduced rakhte hain
  const FIELD_W = 200;
  const THRESHOLD_DEFAULT = 1.0;

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let fieldW = FIELD_W;
  let fieldH = 1; // resize mein calculate hoga
  let threshold = THRESHOLD_DEFAULT;
  let speed = 1.0;
  let renderMode = 'fill'; // 'fill' ya 'contour'
  let isVisible = false;
  let animationId = null;
  let lastTime = 0;
  let isPaused = false;

  // metaballs array — {x, y, vx, vy, radius, hue}
  let balls = [];

  // drag state — existing ball ko drag karne ke liye
  let dragIndex = -1;
  let dragOffsetX = 0;
  let dragOffsetY = 0;

  // field buffer — reduced resolution pe compute hoga
  let fieldData = null;

  // --- Metaball creation ---
  function createBall(x, y) {
    // random radius 20-60, random velocity, unique hue
    const radius = 20 + Math.random() * 40;
    const angle = Math.random() * Math.PI * 2;
    const spd = 20 + Math.random() * 40;
    return {
      x: x !== undefined ? x : Math.random() * canvasW,
      y: y !== undefined ? y : Math.random() * canvasH,
      vx: Math.cos(angle) * spd,
      vy: Math.sin(angle) * spd,
      radius: radius,
      // hue evenly distribute karo existing balls ke hisaab se
      hue: (balls.length * 360 / Math.max(7, balls.length + 1)) % 360,
    };
  }

  // initial 7 metaballs banao
  function initBalls() {
    balls = [];
    for (let i = 0; i < 7; i++) {
      const b = createBall();
      // hue evenly space karo
      b.hue = (i * 360 / 7) % 360;
      balls.push(b);
    }
  }

  // hues redistribute karo — jab add/remove hota hai toh colors even rakhne ke liye
  function redistributeHues() {
    for (let i = 0; i < balls.length; i++) {
      balls[i].hue = (i * 360 / balls.length) % 360;
    }
  }

  // --- Physics update — bounce off walls ---
  function updatePhysics(dt) {
    for (let i = 0; i < balls.length; i++) {
      // dragged ball ko skip karo — user control mein hai
      if (i === dragIndex) continue;

      const b = balls[i];
      b.x += b.vx * dt * speed;
      b.y += b.vy * dt * speed;

      // wall bounce — elastic reflection
      if (b.x - b.radius * 0.3 < 0) {
        b.x = b.radius * 0.3;
        b.vx = Math.abs(b.vx);
      } else if (b.x + b.radius * 0.3 > canvasW) {
        b.x = canvasW - b.radius * 0.3;
        b.vx = -Math.abs(b.vx);
      }

      if (b.y - b.radius * 0.3 < 0) {
        b.y = b.radius * 0.3;
        b.vy = Math.abs(b.vy);
      } else if (b.y + b.radius * 0.3 > canvasH) {
        b.y = canvasH - b.radius * 0.3;
        b.vy = -Math.abs(b.vy);
      }
    }
  }

  // --- Field computation ---
  // f(x,y) = Σ (rᵢ² / ((x-xᵢ)² + (y-yᵢ)²))
  // ye reduced resolution pe compute hota hai, fir scale up karte hain
  function computeField() {
    const fw = fieldW;
    const fh = fieldH;
    const scaleX = canvasW / fw;
    const scaleY = canvasH / fh;

    // field data — har pixel pe total field value + dominant ball index
    if (!fieldData || fieldData.length !== fw * fh * 2) {
      fieldData = new Float32Array(fw * fh * 2); // [fieldVal, dominantBallIdx] pairs
    }

    const n = balls.length;
    // precompute ball positions in field space aur radius squared
    const bx = new Float32Array(n);
    const by = new Float32Array(n);
    const br2 = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      bx[i] = balls[i].x / scaleX;
      by[i] = balls[i].y / scaleY;
      br2[i] = (balls[i].radius / scaleX) * (balls[i].radius / scaleX);
    }

    for (let fy = 0; fy < fh; fy++) {
      for (let fx = 0; fx < fw; fx++) {
        let totalField = 0;
        let maxContrib = 0;
        let dominantIdx = 0;

        for (let i = 0; i < n; i++) {
          const dx = fx - bx[i];
          const dy = fy - by[i];
          const distSq = dx * dx + dy * dy;
          // avoid division by zero — minimum distance
          const contrib = br2[i] / (distSq + 1);
          totalField += contrib;
          if (contrib > maxContrib) {
            maxContrib = contrib;
            dominantIdx = i;
          }
        }

        const idx = (fy * fw + fx) * 2;
        fieldData[idx] = totalField;
        fieldData[idx + 1] = dominantIdx;
      }
    }
  }

  // --- HSL to RGB conversion ---
  // h: 0-360, s: 0-1, l: 0-1 => [r, g, b] 0-255
  function hslToRgb(h, s, l) {
    h = h / 360;
    let r, g, b;
    if (s === 0) {
      r = g = b = l;
    } else {
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1 / 3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1 / 3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }

  // --- Fill mode rendering — ImageData pixel manipulation ---
  function renderFill(ctx) {
    const fw = fieldW;
    const fh = fieldH;
    const n = balls.length;
    if (n === 0) return;

    // offscreen canvas bana ke field render karo reduced resolution pe
    const offCanvas = document.createElement('canvas');
    offCanvas.width = fw;
    offCanvas.height = fh;
    const offCtx = offCanvas.getContext('2d');
    const imageData = offCtx.createImageData(fw, fh);
    const pixels = imageData.data;

    // precompute ball colors — RGB cached rakhte hain
    const ballColors = [];
    for (let i = 0; i < n; i++) {
      ballColors.push(hslToRgb(balls[i].hue, 0.8, 0.55));
    }

    const scaleX = canvasW / fw;
    const scaleY = canvasH / fh;

    // precompute ball field-space positions aur radius squared
    const bx = new Float32Array(n);
    const by = new Float32Array(n);
    const br2 = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      bx[i] = balls[i].x / scaleX;
      by[i] = balls[i].y / scaleY;
      br2[i] = (balls[i].radius / scaleX) * (balls[i].radius / scaleX);
    }

    for (let fy = 0; fy < fh; fy++) {
      for (let fx = 0; fx < fw; fx++) {
        const fIdx = (fy * fw + fx) * 2;
        const fieldVal = fieldData[fIdx];
        const pIdx = (fy * fw + fx) * 4;

        if (fieldVal < threshold * 0.3) {
          // bilkul bahar — dark background
          pixels[pIdx] = 0;
          pixels[pIdx + 1] = 0;
          pixels[pIdx + 2] = 0;
          pixels[pIdx + 3] = 255;
        } else if (fieldVal < threshold) {
          // boundary glow zone — smooth fade in
          // weighted color blend karo based on each ball's contribution
          let rTotal = 0, gTotal = 0, bTotal = 0, wTotal = 0;
          for (let i = 0; i < n; i++) {
            const dx = fx - bx[i];
            const dy = fy - by[i];
            const distSq = dx * dx + dy * dy;
            const contrib = br2[i] / (distSq + 1);
            const col = ballColors[i];
            rTotal += col[0] * contrib;
            gTotal += col[1] * contrib;
            bTotal += col[2] * contrib;
            wTotal += contrib;
          }

          // glow intensity — boundary ke paas bright, door pe dim
          const glowT = (fieldVal - threshold * 0.3) / (threshold * 0.7);
          const intensity = glowT * glowT * 0.4; // quadratic falloff

          if (wTotal > 0) {
            pixels[pIdx] = Math.min(255, (rTotal / wTotal) * intensity) | 0;
            pixels[pIdx + 1] = Math.min(255, (gTotal / wTotal) * intensity) | 0;
            pixels[pIdx + 2] = Math.min(255, (bTotal / wTotal) * intensity) | 0;
          } else {
            pixels[pIdx] = 0;
            pixels[pIdx + 1] = 0;
            pixels[pIdx + 2] = 0;
          }
          pixels[pIdx + 3] = 255;
        } else {
          // threshold ke andar — blob ke andar
          // weighted color blend — jis ball ka zyada contribution uska zyada color
          let rTotal = 0, gTotal = 0, bTotal = 0, wTotal = 0;
          for (let i = 0; i < n; i++) {
            const dx = fx - bx[i];
            const dy = fy - by[i];
            const distSq = dx * dx + dy * dy;
            const contrib = br2[i] / (distSq + 1);
            const col = ballColors[i];
            rTotal += col[0] * contrib;
            gTotal += col[1] * contrib;
            bTotal += col[2] * contrib;
            wTotal += contrib;
          }

          // brightness — field strength se proportional, clamped
          const brightness = Math.min(1.0, 0.5 + (fieldVal - threshold) * 0.15);

          if (wTotal > 0) {
            pixels[pIdx] = Math.min(255, (rTotal / wTotal) * brightness) | 0;
            pixels[pIdx + 1] = Math.min(255, (gTotal / wTotal) * brightness) | 0;
            pixels[pIdx + 2] = Math.min(255, (bTotal / wTotal) * brightness) | 0;
          } else {
            pixels[pIdx] = 0;
            pixels[pIdx + 1] = 0;
            pixels[pIdx + 2] = 0;
          }
          pixels[pIdx + 3] = 255;
        }
      }
    }

    // offscreen canvas pe ImageData daal do
    offCtx.putImageData(imageData, 0, 0);

    // main canvas pe scale up karke draw karo
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'medium';
    ctx.drawImage(offCanvas, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  }

  // --- Contour mode rendering — marching squares ---
  function renderContour(ctx) {
    const fw = fieldW;
    const fh = fieldH;
    const n = balls.length;
    if (n === 0) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // black background
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasW, canvasH);

    const scaleX = canvasW / fw;
    const scaleY = canvasH / fh;

    // multiple contour levels — concentric outlines
    const levels = [0.5, 1.0, 1.5, 2.0];
    const levelAlphas = [0.25, 0.5, 0.7, 0.9];

    for (let lvlIdx = 0; lvlIdx < levels.length; lvlIdx++) {
      const level = levels[lvlIdx] * threshold;
      const alpha = levelAlphas[lvlIdx];

      // marching squares — har cell ke 4 corners check karo
      // agar corner field >= level toh 1, nahi toh 0
      // 4 bits se 16 cases bante hain — edges draw karo
      ctx.lineWidth = lvlIdx === levels.length - 1 ? 2 : 1;

      for (let fy = 0; fy < fh - 1; fy++) {
        for (let fx = 0; fx < fw - 1; fx++) {
          // 4 corners ki field values
          const f00 = fieldData[(fy * fw + fx) * 2];
          const f10 = fieldData[(fy * fw + fx + 1) * 2];
          const f01 = fieldData[((fy + 1) * fw + fx) * 2];
          const f11 = fieldData[((fy + 1) * fw + fx + 1) * 2];

          // 4-bit case index — marching squares lookup
          let caseIdx = 0;
          if (f00 >= level) caseIdx |= 1;
          if (f10 >= level) caseIdx |= 2;
          if (f11 >= level) caseIdx |= 4;
          if (f01 >= level) caseIdx |= 8;

          // 0 aur 15 mein koi edge nahi — skip
          if (caseIdx === 0 || caseIdx === 15) continue;

          // dominant ball ka color use karo is cell ke liye
          const centerFIdx = (fy * fw + fx) * 2;
          const domBall = fieldData[centerFIdx + 1] | 0;
          const hue = balls[domBall] ? balls[domBall].hue : 0;
          const [cr, cg, cb] = hslToRgb(hue, 0.85, 0.6);
          ctx.strokeStyle = 'rgba(' + cr + ',' + cg + ',' + cb + ',' + alpha + ')';

          // linear interpolation se exact edge position nikalo
          // lerp(a, b, level) = fraction along edge where field = level
          const lerp = (a, b) => {
            if (Math.abs(b - a) < 0.0001) return 0.5;
            return (level - a) / (b - a);
          };

          // 4 edges ke midpoints — interpolated
          // top edge (f00 -> f10)
          const tx = fx + lerp(f00, f10);
          const ty = fy;
          // right edge (f10 -> f11)
          const rx = fx + 1;
          const ry = fy + lerp(f10, f11);
          // bottom edge (f01 -> f11)
          const bx = fx + lerp(f01, f11);
          const by = fy + 1;
          // left edge (f00 -> f01)
          const lx = fx;
          const ly = fy + lerp(f00, f01);

          // marching squares — 16 cases, har case mein kaunsi edges connect hongi
          ctx.beginPath();
          switch (caseIdx) {
            case 1:  // bottom-left triangle
              ctx.moveTo(tx * scaleX, ty * scaleY);
              ctx.lineTo(lx * scaleX, ly * scaleY);
              break;
            case 2:  // bottom-right triangle
              ctx.moveTo(tx * scaleX, ty * scaleY);
              ctx.lineTo(rx * scaleX, ry * scaleY);
              break;
            case 3:  // left-right horizontal
              ctx.moveTo(lx * scaleX, ly * scaleY);
              ctx.lineTo(rx * scaleX, ry * scaleY);
              break;
            case 4:  // top-right triangle
              ctx.moveTo(rx * scaleX, ry * scaleY);
              ctx.lineTo(bx * scaleX, by * scaleY);
              break;
            case 5:  // saddle — ambiguous case, 2 edges
              ctx.moveTo(tx * scaleX, ty * scaleY);
              ctx.lineTo(rx * scaleX, ry * scaleY);
              ctx.moveTo(lx * scaleX, ly * scaleY);
              ctx.lineTo(bx * scaleX, by * scaleY);
              break;
            case 6:  // top-bottom vertical
              ctx.moveTo(tx * scaleX, ty * scaleY);
              ctx.lineTo(bx * scaleX, by * scaleY);
              break;
            case 7:  // only top-left corner outside
              ctx.moveTo(lx * scaleX, ly * scaleY);
              ctx.lineTo(bx * scaleX, by * scaleY);
              break;
            case 8:  // top-left triangle
              ctx.moveTo(lx * scaleX, ly * scaleY);
              ctx.lineTo(bx * scaleX, by * scaleY);
              break;
            case 9:  // top-bottom vertical (other direction)
              ctx.moveTo(tx * scaleX, ty * scaleY);
              ctx.lineTo(bx * scaleX, by * scaleY);
              break;
            case 10: // saddle — ambiguous case, 2 edges
              ctx.moveTo(tx * scaleX, ty * scaleY);
              ctx.lineTo(lx * scaleX, ly * scaleY);
              ctx.moveTo(rx * scaleX, ry * scaleY);
              ctx.lineTo(bx * scaleX, by * scaleY);
              break;
            case 11: // only top-right corner outside
              ctx.moveTo(rx * scaleX, ry * scaleY);
              ctx.lineTo(bx * scaleX, by * scaleY);
              break;
            case 12: // left-right horizontal (other direction)
              ctx.moveTo(lx * scaleX, ly * scaleY);
              ctx.lineTo(rx * scaleX, ry * scaleY);
              break;
            case 13: // only bottom-right corner outside
              ctx.moveTo(tx * scaleX, ty * scaleY);
              ctx.lineTo(rx * scaleX, ry * scaleY);
              break;
            case 14: // only bottom-left corner outside
              ctx.moveTo(tx * scaleX, ty * scaleY);
              ctx.lineTo(lx * scaleX, ly * scaleY);
              break;
          }
          ctx.stroke();
        }
      }
    }
  }

  // --- DOM structure banao ---
  // pehle existing children preserve karo (agar koi hai toh hata do)
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — metaballs yahan render honge
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:#000',
    'touch-action:none',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // ball count display — canvas ke corner mein
  const countDisplay = document.createElement('div');
  countDisplay.style.cssText = [
    'position:absolute',
    'top:8px',
    'right:8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:rgba(' + ACCENT_RGB + ',0.6)',
    'background:rgba(0,0,0,0.6)',
    'padding:4px 8px',
    'border-radius:4px',
    'backdrop-filter:blur(4px)',
    'z-index:2',
    'pointer-events:none',
  ].join(';');

  // canvas wrapper — absolute positioning ke liye
  const canvasWrapper = document.createElement('div');
  canvasWrapper.style.cssText = 'position:relative;width:100%;';
  // canvas already container mein hai, wrapper mein daalo
  container.insertBefore(canvasWrapper, container.firstChild);
  canvasWrapper.appendChild(canvas);
  canvasWrapper.appendChild(countDisplay);

  function updateCountDisplay() {
    countDisplay.textContent = 'balls: ' + balls.length;
  }

  // --- Controls Row 1: Buttons ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // button helper — consistent dark theme styling
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
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#ffffff';
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

  function setActive(btn, active) {
    btn._active = active;
    if (active) {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.3)';
      btn.style.color = ACCENT;
      btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.5)';
    } else {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.1)';
      btn.style.color = '#b0b0b0';
      btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
    }
  }

  // slider helper
  function makeSlider(label, min, max, step, value, parent, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const lbl = document.createElement('span');
    lbl.style.cssText = [
      'color:#888',
      'font-size:11px',
      'font-family:"JetBrains Mono",monospace',
      'white-space:nowrap',
    ].join(';');
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.style.cssText = 'width:75px;accent-color:' + ACCENT + ';cursor:pointer;';
    wrapper.appendChild(slider);

    const valSpan = document.createElement('span');
    valSpan.style.cssText = [
      'color:' + ACCENT,
      'font-size:10px',
      'font-family:"JetBrains Mono",monospace',
      'min-width:28px',
      'text-align:right',
    ].join(';');
    const decimals = step.toString().includes('.') ? step.toString().split('.')[1].length : 0;
    valSpan.textContent = Number(value).toFixed(decimals);
    wrapper.appendChild(valSpan);

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valSpan.textContent = v.toFixed(decimals);
      onChange(v);
    });

    parent.appendChild(wrapper);
    return { slider, valSpan };
  }

  // --- Row 1 buttons ---

  // Fill / Contour toggle
  const modeBtn = makeButton('Fill', controlsDiv, () => {
    renderMode = renderMode === 'fill' ? 'contour' : 'fill';
    modeBtn.textContent = renderMode === 'fill' ? 'Fill' : 'Contour';
  });
  setActive(modeBtn, true);

  // Pause/Play
  const pauseBtn = makeButton('Pause', controlsDiv, () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Play' : 'Pause';
    setActive(pauseBtn, isPaused);
  });

  // Add Random — nayi ball add karo random position pe
  makeButton('Add Random', controlsDiv, () => {
    if (balls.length >= MAX_BALLS) return;
    balls.push(createBall());
    redistributeHues();
    updateCountDisplay();
  });

  // Clear — sab hata do
  makeButton('Clear', controlsDiv, () => {
    balls = [];
    updateCountDisplay();
  });

  // Reset — initial state pe waapas jao
  makeButton('Reset', controlsDiv, () => {
    initBalls();
    threshold = THRESHOLD_DEFAULT;
    speed = 1.0;
    renderMode = 'fill';
    isPaused = false;
    modeBtn.textContent = 'Fill';
    setActive(modeBtn, true);
    pauseBtn.textContent = 'Pause';
    setActive(pauseBtn, false);
    thresholdSlider.slider.value = String(THRESHOLD_DEFAULT);
    thresholdSlider.valSpan.textContent = THRESHOLD_DEFAULT.toFixed(1);
    speedSlider.slider.value = '1.0';
    speedSlider.valSpan.textContent = '1.0';
    updateCountDisplay();
  });

  // --- Row 2: Sliders ---
  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:6px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(slidersDiv);

  // threshold slider — metaball boundary control
  const thresholdSlider = makeSlider('Threshold:', 0.5, 3.0, 0.1, threshold, slidersDiv, (v) => {
    threshold = v;
  });

  // speed slider — bounce speed control
  const speedSlider = makeSlider('Speed:', 0.1, 3.0, 0.1, speed, slidersDiv, (v) => {
    speed = v;
  });

  // --- Canvas interaction ---

  // canvas position helper — mouse/touch event se canvas coordinates nikalo
  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }
    return {
      x: (clientX - rect.left) * (canvasW / rect.width),
      y: (clientY - rect.top) * (canvasH / rect.height),
    };
  }

  // find closest ball to position — drag ke liye
  function findClosestBall(px, py) {
    let closest = -1;
    let minDist = Infinity;
    for (let i = 0; i < balls.length; i++) {
      const dx = px - balls[i].x;
      const dy = py - balls[i].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      // ball ke radius ke 1.5x ke andar ho toh hit maano
      if (dist < balls[i].radius * 1.5 && dist < minDist) {
        minDist = dist;
        closest = i;
      }
    }
    return closest;
  }

  // --- Mouse events ---
  canvas.addEventListener('mousedown', (e) => {
    const pos = getCanvasPos(e);
    const idx = findClosestBall(pos.x, pos.y);
    if (idx >= 0) {
      // existing ball pe click — drag shuru karo
      dragIndex = idx;
      dragOffsetX = balls[idx].x - pos.x;
      dragOffsetY = balls[idx].y - pos.y;
      // drag ke time velocity zero kar do — smooth feel hoga
      balls[idx].vx = 0;
      balls[idx].vy = 0;
      canvas.style.cursor = 'grabbing';
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (dragIndex >= 0) {
      const pos = getCanvasPos(e);
      balls[dragIndex].x = pos.x + dragOffsetX;
      balls[dragIndex].y = pos.y + dragOffsetY;
    }
  });

  canvas.addEventListener('mouseup', (e) => {
    if (dragIndex >= 0) {
      // drag release — thoda velocity de do based on mouse direction
      const b = balls[dragIndex];
      b.vx = (Math.random() - 0.5) * 40;
      b.vy = (Math.random() - 0.5) * 40;
      dragIndex = -1;
      canvas.style.cursor = 'crosshair';
    }
  });

  canvas.addEventListener('mouseleave', () => {
    if (dragIndex >= 0) {
      const b = balls[dragIndex];
      b.vx = (Math.random() - 0.5) * 40;
      b.vy = (Math.random() - 0.5) * 40;
      dragIndex = -1;
      canvas.style.cursor = 'crosshair';
    }
  });

  // click — agar drag nahi hua toh new ball add karo
  let mouseDownPos = null;
  canvas.addEventListener('mousedown', (e) => {
    mouseDownPos = getCanvasPos(e);
  });

  canvas.addEventListener('click', (e) => {
    // agar drag hua tha toh click mat count karo
    if (!mouseDownPos) return;
    const pos = getCanvasPos(e);
    const dx = pos.x - mouseDownPos.x;
    const dy = pos.y - mouseDownPos.y;
    const movedDist = Math.sqrt(dx * dx + dy * dy);
    mouseDownPos = null;

    // agar ball pe click tha (drag intent) toh skip
    if (findClosestBall(pos.x, pos.y) >= 0) return;
    // agar move hua toh skip — ye drag tha
    if (movedDist > 5) return;

    // nahi toh new ball add karo
    if (balls.length >= MAX_BALLS) return;
    const newBall = createBall(pos.x, pos.y);
    balls.push(newBall);
    redistributeHues();
    updateCountDisplay();
  });

  // double-click — remove closest ball
  canvas.addEventListener('dblclick', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const idx = findClosestBall(pos.x, pos.y);
    if (idx >= 0) {
      balls.splice(idx, 1);
      redistributeHues();
      updateCountDisplay();
    }
  });

  // right-click — remove closest ball
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const idx = findClosestBall(pos.x, pos.y);
    if (idx >= 0) {
      balls.splice(idx, 1);
      redistributeHues();
      updateCountDisplay();
    }
  });

  // --- Touch events ---
  let touchStartTime = 0;
  let touchStartPos = null;

  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    touchStartTime = Date.now();
    touchStartPos = pos;

    const idx = findClosestBall(pos.x, pos.y);
    if (idx >= 0) {
      dragIndex = idx;
      dragOffsetX = balls[idx].x - pos.x;
      dragOffsetY = balls[idx].y - pos.y;
      balls[idx].vx = 0;
      balls[idx].vy = 0;
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (dragIndex >= 0 && e.touches.length > 0) {
      const pos = getCanvasPos(e);
      balls[dragIndex].x = pos.x + dragOffsetX;
      balls[dragIndex].y = pos.y + dragOffsetY;
    }
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    const elapsed = Date.now() - touchStartTime;

    if (dragIndex >= 0) {
      const b = balls[dragIndex];
      b.vx = (Math.random() - 0.5) * 40;
      b.vy = (Math.random() - 0.5) * 40;
      dragIndex = -1;
    } else if (touchStartPos && elapsed < 300) {
      // quick tap — add new ball
      if (balls.length < MAX_BALLS) {
        const newBall = createBall(touchStartPos.x, touchStartPos.y);
        balls.push(newBall);
        redistributeHues();
        updateCountDisplay();
      }
    }

    touchStartPos = null;
  }, { passive: false });

  // --- Canvas sizing ---
  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = canvasWrapper.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;

    canvas.width = Math.floor(containerWidth * dpr);
    canvas.height = Math.floor(CANVAS_HEIGHT * dpr);
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';

    // field resolution bhi update karo — aspect ratio maintain karo
    fieldW = FIELD_W;
    fieldH = Math.floor(FIELD_W * (CANVAS_HEIGHT / containerWidth));
    if (fieldH < 1) fieldH = 1;

    // field buffer re-allocate
    fieldData = new Float32Array(fieldW * fieldH * 2);
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // --- Drawing ---
  function draw() {
    // field compute karo — dono modes ke liye chahiye
    computeField();

    if (renderMode === 'fill') {
      renderFill(ctx);
    } else {
      renderContour(ctx);
    }

    // hint jab koi ball nahi hai
    if (balls.length === 0) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('click to add metaballs \u2022 drag to move \u2022 right-click to remove', canvasW / 2, canvasH / 2);
    }

    updateCountDisplay();
  }

  // --- Animation loop ---
  function animate(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    // delta time — variable timestep with cap
    if (lastTime === 0) lastTime = timestamp;
    let dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;

    // dt clamp — tab switch se bada dt aa sakta hai
    dt = Math.min(dt, 0.05);

    if (!isPaused) {
      updatePhysics(dt);
    }

    draw();
    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf visible hone pe animate karo ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    lastTime = 0;
    animationId = requestAnimationFrame(animate);
  }

  function stopAnimation() {
    isVisible = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
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
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) animate(); });

  // tab switch pe pause — background mein CPU waste mat karo
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // --- Init — sab shuru karo ---
  initBalls();
  updateCountDisplay();

  // pehla frame draw karo — blank na dikhe
  // field compute ke liye canvas dimensions chahiye, jo resize mein set ho chuke hain
  computeField();
  draw();
}
