// ============================================================
// Hopfield Network — Associative Memory ka demo
// Patterns memorize karo, noise daalo, network apne aap recall karega
// Binary neurons ka grid jo energy minimize karta hai
// ============================================================

export function initHopfield() {
  const container = document.getElementById('hopfieldContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  let animationId = null, isVisible = false, canvasW = 0;

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:crosshair;background:#111;border:1px solid rgba(74,158,255,0.15);`;
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

  // controls
  const btnSmiley = mkBtn(ctrl, 'Smiley', 'hf-smiley');
  const btnCross = mkBtn(ctrl, 'Cross', 'hf-cross');
  const btnLetter = mkBtn(ctrl, 'Letter H', 'hf-letter');
  const btnMemorize = mkBtn(ctrl, 'Memorize', 'hf-mem');
  const noiseSlider = mkSlider(ctrl, 'Noise%:', 'hf-noise', 5, 60, 25, 5);
  const btnCorrupt = mkBtn(ctrl, 'Corrupt', 'hf-corrupt');
  const btnRecall = mkBtn(ctrl, 'Recall ▶', 'hf-recall');
  const btnClear = mkBtn(ctrl, 'Clear', 'hf-clear');
  const infoLbl = document.createElement('span');
  infoLbl.style.cssText = "color:#888;font:11px 'JetBrains Mono',monospace;margin-left:8px";
  ctrl.appendChild(infoLbl);

  // --- State ---
  const N = 20;                    // 20x20 grid
  const TOTAL = N * N;             // 400 neurons
  let state = new Array(TOTAL);    // current neuron states (+1 or -1)
  let W = [];                      // weight matrix (TOTAL x TOTAL)
  let patterns = [];               // memorized patterns list
  let isRecalling = false;
  let drawMode = true;             // click to draw
  let energy = 0;
  let recallSteps = 0;

  // --- Weight matrix initialize (sab zero) ---
  function initWeights() {
    W = [];
    for (let i = 0; i < TOTAL; i++) {
      W[i] = new Float32Array(TOTAL); // zero-initialized
    }
  }

  // --- Pattern presets ---
  // 20x20 grid mein patterns — 1=on, -1=off
  function makeSmiley() {
    const p = new Array(TOTAL).fill(-1);
    const set = (r, c) => { if (r >= 0 && r < N && c >= 0 && c < N) p[r * N + c] = 1; };
    // aankhein — left eye
    for (let r = 5; r <= 7; r++) for (let c = 5; c <= 7; c++) set(r, c);
    // right eye
    for (let r = 5; r <= 7; r++) for (let c = 12; c <= 14; c++) set(r, c);
    // muh — smile curve
    set(13, 5); set(13, 6); set(14, 7); set(14, 8); set(14, 9);
    set(14, 10); set(14, 11); set(14, 12); set(13, 13); set(13, 14);
    // naak
    set(9, 9); set(10, 9); set(9, 10); set(10, 10);
    return p;
  }
  function makeCross() {
    const p = new Array(TOTAL).fill(-1);
    const set = (r, c) => { if (r >= 0 && r < N && c >= 0 && c < N) p[r * N + c] = 1; };
    // vertical bar
    for (let r = 2; r < 18; r++) { set(r, 9); set(r, 10); }
    // horizontal bar
    for (let c = 2; c < 18; c++) { set(9, c); set(10, c); }
    return p;
  }
  function makeLetterH() {
    const p = new Array(TOTAL).fill(-1);
    const set = (r, c) => { if (r >= 0 && r < N && c >= 0 && c < N) p[r * N + c] = 1; };
    // left vertical
    for (let r = 3; r < 17; r++) { set(r, 5); set(r, 6); }
    // right vertical
    for (let r = 3; r < 17; r++) { set(r, 13); set(r, 14); }
    // horizontal bridge
    for (let c = 5; c <= 14; c++) { set(9, c); set(10, c); }
    return p;
  }

  // --- Hebbian learning: W_ij = (1/P) * Σ p_i·p_j ---
  function memorizePatterns() {
    initWeights();
    if (patterns.length === 0) return;
    for (const pat of patterns) {
      for (let i = 0; i < TOTAL; i++) {
        for (let j = i + 1; j < TOTAL; j++) {
          const dw = pat[i] * pat[j] / patterns.length;
          W[i][j] += dw;
          W[j][i] += dw;
        }
        // diagonal zero rakhna hai — self-connection nahi chahiye
        W[i][i] = 0;
      }
    }
    infoLbl.textContent = `Patterns memorized: ${patterns.length} | Energy: ${computeEnergy().toFixed(0)}`;
  }

  // --- Energy function: E = -½ΣΣ W_ij·s_i·s_j ---
  function computeEnergy() {
    let E = 0;
    for (let i = 0; i < TOTAL; i++) {
      for (let j = i + 1; j < TOTAL; j++) {
        E -= W[i][j] * state[i] * state[j];
      }
    }
    return E;
  }

  // --- Async update: random neuron uthao, sign(Σ W_ij·s_j) lagao ---
  function recallStep(nSteps) {
    for (let s = 0; s < nSteps; s++) {
      const i = Math.floor(Math.random() * TOTAL);
      let sum = 0;
      for (let j = 0; j < TOTAL; j++) {
        sum += W[i][j] * state[j];
      }
      state[i] = sum >= 0 ? 1 : -1;
      recallSteps++;
    }
    energy = computeEnergy();
    infoLbl.textContent = `Steps: ${recallSteps} | Energy: ${energy.toFixed(0)} | Patterns: ${patterns.length}`;
  }

  // --- Noise add karo (corrupt) ---
  function corruptState() {
    const noiseP = parseInt(noiseSlider.value) / 100;
    for (let i = 0; i < TOTAL; i++) {
      if (Math.random() < noiseP) state[i] *= -1; // flip karo
    }
    recallSteps = 0;
    energy = computeEnergy();
    infoLbl.textContent = `Corrupted ${(noiseP * 100).toFixed(0)}% | Energy: ${energy.toFixed(0)}`;
  }

  // --- Render ---
  function render() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // grid draw karo — har cell ek neuron
    const gridSize = Math.min(canvasW - 40, CANVAS_HEIGHT - 20);
    const cellSize = gridSize / N;
    const offsetX = (canvasW - gridSize) / 2;
    const offsetY = (CANVAS_HEIGHT - gridSize) / 2;

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const x = offsetX + c * cellSize;
        const y = offsetY + r * cellSize;
        const idx = r * N + c;
        // +1 = blue (on), -1 = dark (off)
        if (state[idx] === 1) {
          ctx.fillStyle = ACCENT;
        } else {
          ctx.fillStyle = '#1a1a2e';
        }
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
      }
    }

    // grid border
    ctx.strokeStyle = 'rgba(74,158,255,0.2)';
    ctx.lineWidth = 1;
    ctx.strokeRect(offsetX, offsetY, gridSize, gridSize);

    // title
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = '#888';
    ctx.textAlign = 'left';
    ctx.fillText('Click cells to draw, then Memorize', offsetX, offsetY - 5);
  }

  // --- Mouse events — drawing on grid ---
  let isMouseDown = false;
  canvas.addEventListener('mousedown', (e) => {
    isMouseDown = true;
    toggleCell(e);
  });
  canvas.addEventListener('mousemove', (e) => {
    if (isMouseDown) toggleCell(e);
  });
  canvas.addEventListener('mouseup', () => { isMouseDown = false; });
  canvas.addEventListener('mouseleave', () => { isMouseDown = false; });

  function toggleCell(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const gridSize = Math.min(canvasW - 40, CANVAS_HEIGHT - 20);
    const cellSize = gridSize / N;
    const offsetX = (canvasW - gridSize) / 2;
    const offsetY = (CANVAS_HEIGHT - gridSize) / 2;
    const c = Math.floor((mx - offsetX) / cellSize);
    const r = Math.floor((my - offsetY) / cellSize);
    if (r >= 0 && r < N && c >= 0 && c < N) {
      state[r * N + c] = 1; // draw mode — set to 1
    }
  }

  // --- State management ---
  function clearState() {
    state.fill(-1);
    recallSteps = 0;
    isRecalling = false;
    btnRecall.textContent = 'Recall ▶';
  }

  function loadPreset(presetFn) {
    const p = presetFn();
    for (let i = 0; i < TOTAL; i++) state[i] = p[i];
    recallSteps = 0;
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (isRecalling) {
      // har frame 50 neurons update karo — smooth convergence dikhao
      recallStep(50);
    }
    render();
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

  // --- Event listeners ---
  btnSmiley.addEventListener('click', () => loadPreset(makeSmiley));
  btnCross.addEventListener('click', () => loadPreset(makeCross));
  btnLetter.addEventListener('click', () => loadPreset(makeLetterH));
  btnMemorize.addEventListener('click', () => {
    // current state ko pattern mein store karo
    patterns.push(state.slice());
    memorizePatterns();
  });
  btnCorrupt.addEventListener('click', corruptState);
  btnRecall.addEventListener('click', () => {
    isRecalling = !isRecalling;
    btnRecall.textContent = isRecalling ? 'Stop ⏸' : 'Recall ▶';
    if (isRecalling && isVisible && !animationId) loop();
  });
  btnClear.addEventListener('click', () => {
    clearState();
    patterns = [];
    initWeights();
    infoLbl.textContent = 'Cleared all';
  });

  // --- Init ---
  initWeights();
  clearState();
  // default: smiley memorize karke dikhao
  loadPreset(makeSmiley);
  patterns.push(state.slice());
  loadPreset(makeCross);
  patterns.push(state.slice());
  memorizePatterns();
  // start with smiley visible
  loadPreset(makeSmiley);
  render();
}
