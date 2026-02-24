// ============================================================
// Attention Mechanism Visualization
// Transformer ka attention dikhao — Q×K → softmax → weighted arcs
// Hardcoded sentences ke saath multi-head attention ka demo
// ============================================================

export function initAttentionViz() {
  const container = document.getElementById('attentionVizContainer');
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

  function mkBtn(parent, text, id) {
    const b = document.createElement('button');
    b.textContent = text; b.id = id;
    b.style.cssText = "background:#333;color:#ccc;border:1px solid #555;padding:3px 8px;border-radius:4px;cursor:pointer;font:11px 'JetBrains Mono',monospace";
    parent.appendChild(b);
    return b;
  }

  // controls
  const btnS1 = mkBtn(ctrl, 'Sentence 1', 'av-s1');
  const btnS2 = mkBtn(ctrl, 'Sentence 2', 'av-s2');
  const btnS3 = mkBtn(ctrl, 'Sentence 3', 'av-s3');
  const btnS4 = mkBtn(ctrl, 'Sentence 4', 'av-s4');
  const btnH1 = mkBtn(ctrl, 'Head 1', 'av-h1');
  const btnH2 = mkBtn(ctrl, 'Head 2', 'av-h2');
  const btnH3 = mkBtn(ctrl, 'Head 3', 'av-h3');
  const btnH4 = mkBtn(ctrl, 'Head 4', 'av-h4');
  const btnToggle = mkBtn(ctrl, 'View: Arcs', 'av-toggle');
  const infoLbl = document.createElement('span');
  infoLbl.style.cssText = "color:#888;font:11px 'JetBrains Mono',monospace;margin-left:8px";
  ctrl.appendChild(infoLbl);

  // --- Pre-computed attention data ---
  // Har sentence ke 4 heads ke attention weights
  // Ye manually designed hain taaki meaningful patterns dikhein
  const sentences = [
    {
      words: ['The', 'cat', 'sat', 'on', 'the', 'warm', 'mat'],
      // 4 heads — har ek NxN attention matrix
      heads: [
        // Head 1: positional — paas wale words pe attend
        softmaxRows(bandedMatrix(7, 2)),
        // Head 2: subject-verb — cat→sat, mat→warm
        softmaxRows(sparseMatrix(7, [[1,2,0.9],[2,1,0.8],[5,6,0.9],[6,5,0.7],[0,1,0.6]])),
        // Head 3: noun attention — articles attend nouns
        softmaxRows(sparseMatrix(7, [[0,1,0.95],[4,6,0.95],[3,6,0.5],[3,1,0.3]])),
        // Head 4: global/uniform
        softmaxRows(uniformMatrix(7, 0.3))
      ]
    },
    {
      words: ['I', 'love', 'coding', 'in', 'Python', 'and', 'Rust'],
      heads: [
        softmaxRows(bandedMatrix(7, 2)),
        softmaxRows(sparseMatrix(7, [[0,1,0.9],[1,2,0.85],[4,2,0.7],[6,2,0.7],[5,4,0.6],[5,6,0.6]])),
        softmaxRows(sparseMatrix(7, [[2,4,0.9],[2,6,0.8],[4,6,0.7],[1,0,0.9]])),
        softmaxRows(sparseMatrix(7, [[0,0,0.3],[1,1,0.3],[2,2,0.3],[3,3,0.3],[4,4,0.3],[5,5,0.3],[6,6,0.3]]))
      ]
    },
    {
      words: ['The', 'robot', 'moved', 'its', 'arm', 'carefully'],
      heads: [
        softmaxRows(bandedMatrix(6, 2)),
        softmaxRows(sparseMatrix(6, [[1,2,0.9],[2,1,0.8],[3,1,0.85],[4,3,0.7],[5,2,0.9]])),
        softmaxRows(sparseMatrix(6, [[0,1,0.95],[3,4,0.9],[2,4,0.6],[5,4,0.5]])),
        softmaxRows(sparseMatrix(6, [[4,1,0.8],[2,5,0.7],[1,4,0.6],[5,2,0.8]]))
      ]
    },
    {
      words: ['Attention', 'is', 'all', 'you', 'need'],
      heads: [
        softmaxRows(bandedMatrix(5, 2)),
        softmaxRows(sparseMatrix(5, [[0,2,0.9],[3,4,0.85],[1,0,0.8],[4,3,0.7]])),
        softmaxRows(sparseMatrix(5, [[2,0,0.9],[4,0,0.85],[4,2,0.7],[3,4,0.6]])),
        softmaxRows(sparseMatrix(5, [[0,0,0.6],[1,1,0.5],[2,2,0.5],[3,3,0.5],[4,4,0.6]]))
      ]
    }
  ];

  // --- Helper functions for creating attention matrices ---
  function bandedMatrix(n, bandwidth) {
    const m = [];
    for (let i = 0; i < n; i++) {
      m[i] = [];
      for (let j = 0; j < n; j++) {
        const dist = Math.abs(i - j);
        m[i][j] = dist <= bandwidth ? Math.exp(-dist * 0.5) : 0.01;
      }
    }
    return m;
  }

  function sparseMatrix(n, entries) {
    // entries: [[from, to, weight], ...]
    const m = [];
    for (let i = 0; i < n; i++) {
      m[i] = [];
      for (let j = 0; j < n; j++) m[i][j] = 0.05;
    }
    for (const [i, j, w] of entries) {
      if (i < n && j < n) m[i][j] = w;
    }
    return m;
  }

  function uniformMatrix(n, val) {
    const m = [];
    for (let i = 0; i < n; i++) {
      m[i] = [];
      for (let j = 0; j < n; j++) m[i][j] = val + Math.random() * 0.1;
    }
    return m;
  }

  function softmaxRows(m) {
    const n = m.length;
    const result = [];
    for (let i = 0; i < n; i++) {
      result[i] = [];
      let maxVal = -Infinity;
      for (let j = 0; j < m[i].length; j++) {
        if (m[i][j] > maxVal) maxVal = m[i][j];
      }
      let sum = 0;
      for (let j = 0; j < m[i].length; j++) {
        result[i][j] = Math.exp((m[i][j] - maxVal) * 3); // temperature=3 for sharper distributions
        sum += result[i][j];
      }
      for (let j = 0; j < m[i].length; j++) result[i][j] /= sum;
    }
    return result;
  }

  // --- State ---
  let currentSentence = 0;
  let currentHead = 0;
  let viewMode = 'arcs';  // 'arcs' ya 'heatmap'
  let hoveredWord = -1;    // mouse hover pe highlight
  let animProgress = 0;    // animation progress 0→1
  let animating = false;

  // --- Render ---
  function render() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    const sent = sentences[currentSentence];
    const words = sent.words;
    const n = words.length;
    const attn = sent.heads[currentHead];

    if (viewMode === 'arcs') {
      renderArcs(words, attn, n);
    } else {
      renderHeatmap(words, attn, n);
    }

    // info
    infoLbl.textContent = `Sentence ${currentSentence + 1} | Head ${currentHead + 1} | ${viewMode}`;
  }

  function renderArcs(words, attn, n) {
    // word boxes top area mein
    const boxH = 30;
    const topY = 60;
    const wordSpacing = Math.min(100, (canvasW - 40) / n);
    const startX = (canvasW - wordSpacing * n) / 2 + wordSpacing / 2;

    // word boxes draw karo
    ctx.font = "bold 12px 'JetBrains Mono', monospace";
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i++) {
      const x = startX + i * wordSpacing;
      const isHov = i === hoveredWord;
      // box
      ctx.fillStyle = isHov ? 'rgba(74,158,255,0.3)' : 'rgba(74,158,255,0.1)';
      ctx.strokeStyle = isHov ? ACCENT : 'rgba(74,158,255,0.3)';
      ctx.lineWidth = isHov ? 2 : 1;
      const tw = ctx.measureText(words[i]).width + 16;
      ctx.fillRect(x - tw / 2, topY, tw, boxH);
      ctx.strokeRect(x - tw / 2, topY, tw, boxH);
      // text
      ctx.fillStyle = isHov ? '#fff' : '#ccc';
      ctx.fillText(words[i], x, topY + 20);
    }

    // attention arcs — curves between words
    const arcBaseY = topY + boxH + 10;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const weight = attn[i][j];
        if (weight < 0.05) continue;

        // agar koi word hovered hai toh sirf uske connections dikhao
        if (hoveredWord >= 0 && hoveredWord !== i) continue;

        const x1 = startX + i * wordSpacing;
        const x2 = startX + j * wordSpacing;
        const dist = Math.abs(x2 - x1);
        const arcHeight = dist * 0.4 + 20;

        ctx.strokeStyle = `rgba(74,158,255,${weight * 0.8 * (animating ? Math.min(1, animProgress * 2) : 1)})`;
        ctx.lineWidth = weight * 5;
        ctx.beginPath();
        ctx.moveTo(x1, arcBaseY);
        // quadratic bezier arc neeche ki taraf
        ctx.quadraticCurveTo(
          (x1 + x2) / 2,
          arcBaseY + arcHeight,
          x2, arcBaseY
        );
        ctx.stroke();

        // weight value likho arc ke beech mein (sirf bade weights ke liye)
        if (weight > 0.15) {
          const midX = (x1 + x2) / 2;
          const midY = arcBaseY + arcHeight * 0.55;
          ctx.font = "9px 'JetBrains Mono', monospace";
          ctx.fillStyle = 'rgba(255,255,255,0.5)';
          ctx.fillText(weight.toFixed(2), midX, midY);
        }
      }
    }

    // Q×K → softmax explanation
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = '#666';
    ctx.textAlign = 'left';
    ctx.fillText('Hover words to see attention flow', 10, 25);
    ctx.fillText('Arc thickness = attention weight', 10, 40);
  }

  function renderHeatmap(words, attn, n) {
    // NxN heatmap grid
    const gridSize = Math.min(canvasW - 140, CANVAS_HEIGHT - 80);
    const cellSize = gridSize / n;
    const offsetX = 80;
    const offsetY = 50;

    // column labels (top) — key words
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.textAlign = 'center';
    ctx.fillStyle = '#aaa';
    for (let j = 0; j < n; j++) {
      ctx.save();
      ctx.translate(offsetX + j * cellSize + cellSize / 2, offsetY - 8);
      ctx.rotate(-0.5);
      ctx.fillText(words[j], 0, 0);
      ctx.restore();
    }

    // row labels (left) — query words
    ctx.textAlign = 'right';
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = hoveredWord === i ? '#fff' : '#aaa';
      ctx.fillText(words[i], offsetX - 8, offsetY + i * cellSize + cellSize / 2 + 4);
    }

    // heatmap cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = offsetX + j * cellSize;
        const y = offsetY + i * cellSize;
        const w = attn[i][j];

        // color intensity — blue gradient
        const intensity = Math.floor(w * 255);
        ctx.fillStyle = `rgb(${30 + intensity * 0.2}, ${50 + intensity * 0.4}, ${80 + intensity * 0.7})`;
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

        // highlight on hover
        if (i === hoveredWord || j === hoveredWord) {
          ctx.strokeStyle = 'rgba(255,255,255,0.3)';
          ctx.lineWidth = 1;
          ctx.strokeRect(x, y, cellSize - 1, cellSize - 1);
        }

        // weight text — sirf bade cells mein
        if (cellSize > 35) {
          ctx.font = "9px 'JetBrains Mono', monospace";
          ctx.textAlign = 'center';
          ctx.fillStyle = w > 0.3 ? '#fff' : 'rgba(255,255,255,0.4)';
          ctx.fillText(w.toFixed(2), x + cellSize / 2, y + cellSize / 2 + 3);
        }
      }
    }

    // axis labels
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = '#888';
    ctx.textAlign = 'center';
    ctx.fillText('Keys (K)', offsetX + gridSize / 2, offsetY - 25);
    ctx.save();
    ctx.translate(15, offsetY + gridSize / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Queries (Q)', 0, 0);
    ctx.restore();

    // color scale legend
    const legendX = offsetX + gridSize + 20;
    const legendH = gridSize;
    for (let y = 0; y < legendH; y++) {
      const v = 1 - y / legendH;
      const intensity = Math.floor(v * 255);
      ctx.fillStyle = `rgb(${30 + intensity * 0.2}, ${50 + intensity * 0.4}, ${80 + intensity * 0.7})`;
      ctx.fillRect(legendX, offsetY + y, 15, 1);
    }
    ctx.fillStyle = '#aaa';
    ctx.textAlign = 'left';
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillText('1.0', legendX + 18, offsetY + 8);
    ctx.fillText('0.0', legendX + 18, offsetY + legendH);
  }

  // --- Mouse tracking for word hover ---
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const sent = sentences[currentSentence];
    const n = sent.words.length;
    hoveredWord = -1;

    if (viewMode === 'arcs') {
      const wordSpacing = Math.min(100, (canvasW - 40) / n);
      const startX = (canvasW - wordSpacing * n) / 2 + wordSpacing / 2;
      const topY = 60;
      for (let i = 0; i < n; i++) {
        const x = startX + i * wordSpacing;
        if (Math.abs(mx - x) < wordSpacing / 2 && my > topY - 5 && my < topY + 40) {
          hoveredWord = i;
          break;
        }
      }
    } else {
      // heatmap — row hover
      const gridSize = Math.min(canvasW - 140, CANVAS_HEIGHT - 80);
      const cellSize = gridSize / n;
      const offsetY = 50;
      const row = Math.floor((my - offsetY) / cellSize);
      if (row >= 0 && row < n) hoveredWord = row;
    }
  });
  canvas.addEventListener('mouseleave', () => { hoveredWord = -1; });

  function switchSentence(idx) {
    currentSentence = idx;
    hoveredWord = -1;
    // animation trigger
    animating = true;
    animProgress = 0;
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
    if (animating) {
      animProgress += 0.03;
      if (animProgress >= 1) { animating = false; animProgress = 1; }
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

  // --- Events ---
  btnS1.addEventListener('click', () => switchSentence(0));
  btnS2.addEventListener('click', () => switchSentence(1));
  btnS3.addEventListener('click', () => switchSentence(2));
  btnS4.addEventListener('click', () => switchSentence(3));
  btnH1.addEventListener('click', () => { currentHead = 0; });
  btnH2.addEventListener('click', () => { currentHead = 1; });
  btnH3.addEventListener('click', () => { currentHead = 2; });
  btnH4.addEventListener('click', () => { currentHead = 3; });
  btnToggle.addEventListener('click', () => {
    viewMode = viewMode === 'arcs' ? 'heatmap' : 'arcs';
    btnToggle.textContent = viewMode === 'arcs' ? 'View: Arcs' : 'View: Heatmap';
  });

  // head buttons highlight
  const headBtns = [btnH1, btnH2, btnH3, btnH4];
  headBtns.forEach((b, i) => {
    b.addEventListener('click', () => {
      headBtns.forEach(hb => { hb.style.borderColor = '#555'; });
      b.style.borderColor = ACCENT;
    });
  });
  btnH1.style.borderColor = ACCENT; // default highlight

  // sentence buttons highlight
  const sentBtns = [btnS1, btnS2, btnS3, btnS4];
  sentBtns.forEach((b, i) => {
    b.addEventListener('click', () => {
      sentBtns.forEach(sb => { sb.style.borderColor = '#555'; });
      b.style.borderColor = ACCENT;
    });
  });
  btnS1.style.borderColor = ACCENT;

  // --- Init ---
  render();
}
