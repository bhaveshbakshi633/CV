// ============================================================
// Word2Vec — Pre-defined word embeddings with semantic clusters
// Words ka 2D map — click karo, nearest neighbors dekho, vector arithmetic karo
// ============================================================

// yahi entry point hai — words ka 2D scatter plot with interactions
export function initWord2Vec() {
  const container = document.getElementById('word2vecContainer');
  if (!container) return;

  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';

  // hardcoded 2D embeddings — 5 semantic clusters: countries, colors, animals, verbs, adjectives
  // positions manually tuned taaki clusters visible hon
  const WORDS = [
    // countries — top-left cluster
    { w: 'india', x: 80, y: 60, cat: 0 }, { w: 'china', x: 110, y: 45, cat: 0 },
    { w: 'japan', x: 130, y: 70, cat: 0 }, { w: 'brazil', x: 65, y: 90, cat: 0 },
    { w: 'france', x: 95, y: 110, cat: 0 }, { w: 'germany', x: 120, y: 95, cat: 0 },
    { w: 'russia', x: 55, y: 50, cat: 0 }, { w: 'canada', x: 75, y: 130, cat: 0 },
    { w: 'italy', x: 140, y: 110, cat: 0 }, { w: 'spain', x: 105, y: 130, cat: 0 },
    { w: 'usa', x: 90, y: 80, cat: 0 }, { w: 'uk', x: 130, y: 55, cat: 0 },
    { w: 'australia', x: 50, y: 75, cat: 0 }, { w: 'mexico', x: 60, y: 115, cat: 0 },
    { w: 'egypt', x: 145, y: 85, cat: 0 }, { w: 'korea', x: 115, y: 60, cat: 0 },
    { w: 'thailand', x: 100, y: 50, cat: 0 }, { w: 'turkey', x: 135, y: 100, cat: 0 },
    { w: 'argentina', x: 70, y: 100, cat: 0 }, { w: 'nigeria', x: 155, y: 70, cat: 0 },
    // colors — top-right cluster
    { w: 'red', x: 520, y: 55, cat: 1 }, { w: 'blue', x: 545, y: 70, cat: 1 },
    { w: 'green', x: 530, y: 90, cat: 1 }, { w: 'yellow', x: 510, y: 80, cat: 1 },
    { w: 'purple', x: 555, y: 50, cat: 1 }, { w: 'orange', x: 500, y: 65, cat: 1 },
    { w: 'pink', x: 560, y: 85, cat: 1 }, { w: 'black', x: 535, y: 110, cat: 1 },
    { w: 'white', x: 515, y: 105, cat: 1 }, { w: 'brown', x: 490, y: 95, cat: 1 },
    { w: 'gray', x: 540, y: 40, cat: 1 }, { w: 'violet', x: 570, y: 65, cat: 1 },
    { w: 'cyan', x: 550, y: 100, cat: 1 }, { w: 'gold', x: 505, y: 45, cat: 1 },
    { w: 'silver', x: 525, y: 115, cat: 1 }, { w: 'crimson', x: 480, y: 55, cat: 1 },
    { w: 'scarlet', x: 495, y: 75, cat: 1 }, { w: 'teal', x: 565, y: 95, cat: 1 },
    { w: 'maroon', x: 485, y: 110, cat: 1 }, { w: 'navy', x: 575, y: 55, cat: 1 },
    // animals — bottom-left cluster
    { w: 'dog', x: 90, y: 290, cat: 2 }, { w: 'cat', x: 115, y: 275, cat: 2 },
    { w: 'horse', x: 75, y: 310, cat: 2 }, { w: 'bird', x: 130, y: 300, cat: 2 },
    { w: 'fish', x: 105, y: 325, cat: 2 }, { w: 'lion', x: 60, y: 285, cat: 2 },
    { w: 'tiger', x: 55, y: 305, cat: 2 }, { w: 'bear', x: 80, y: 330, cat: 2 },
    { w: 'wolf', x: 65, y: 270, cat: 2 }, { w: 'eagle', x: 140, y: 280, cat: 2 },
    { w: 'snake', x: 120, y: 340, cat: 2 }, { w: 'rabbit', x: 100, y: 260, cat: 2 },
    { w: 'monkey', x: 50, y: 295, cat: 2 }, { w: 'dolphin', x: 135, y: 320, cat: 2 },
    { w: 'elephant', x: 85, y: 350, cat: 2 }, { w: 'deer', x: 70, y: 340, cat: 2 },
    { w: 'penguin', x: 145, y: 310, cat: 2 }, { w: 'shark', x: 110, y: 350, cat: 2 },
    { w: 'whale', x: 95, y: 365, cat: 2 }, { w: 'owl', x: 125, y: 260, cat: 2 },
    // verbs — bottom-right cluster
    { w: 'run', x: 500, y: 280, cat: 3 }, { w: 'walk', x: 520, y: 295, cat: 3 },
    { w: 'jump', x: 490, y: 310, cat: 3 }, { w: 'swim', x: 540, y: 275, cat: 3 },
    { w: 'fly', x: 530, y: 305, cat: 3 }, { w: 'eat', x: 510, y: 320, cat: 3 },
    { w: 'sleep', x: 555, y: 290, cat: 3 }, { w: 'think', x: 545, y: 310, cat: 3 },
    { w: 'write', x: 485, y: 295, cat: 3 }, { w: 'read', x: 505, y: 260, cat: 3 },
    { w: 'speak', x: 560, y: 270, cat: 3 }, { w: 'listen', x: 570, y: 285, cat: 3 },
    { w: 'dance', x: 495, y: 340, cat: 3 }, { w: 'sing', x: 515, y: 345, cat: 3 },
    { w: 'climb', x: 480, y: 265, cat: 3 }, { w: 'throw', x: 475, y: 285, cat: 3 },
    { w: 'catch', x: 535, y: 335, cat: 3 }, { w: 'push', x: 550, y: 325, cat: 3 },
    { w: 'pull', x: 565, y: 305, cat: 3 }, { w: 'kick', x: 485, y: 330, cat: 3 },
    // adjectives — center cluster
    { w: 'fast', x: 290, y: 170, cat: 4 }, { w: 'slow', x: 310, y: 185, cat: 4 },
    { w: 'big', x: 280, y: 195, cat: 4 }, { w: 'small', x: 325, y: 170, cat: 4 },
    { w: 'hot', x: 270, y: 155, cat: 4 }, { w: 'cold', x: 340, y: 190, cat: 4 },
    { w: 'bright', x: 300, y: 150, cat: 4 }, { w: 'dark', x: 320, y: 205, cat: 4 },
    { w: 'happy', x: 260, y: 180, cat: 4 }, { w: 'sad', x: 335, y: 160, cat: 4 },
    { w: 'loud', x: 285, y: 210, cat: 4 }, { w: 'quiet', x: 350, y: 175, cat: 4 },
    { w: 'strong', x: 275, y: 140, cat: 4 }, { w: 'weak', x: 345, y: 200, cat: 4 },
    { w: 'tall', x: 295, y: 130, cat: 4 }, { w: 'short', x: 330, y: 215, cat: 4 },
    { w: 'young', x: 255, y: 165, cat: 4 }, { w: 'old', x: 355, y: 185, cat: 4 },
    { w: 'rich', x: 305, y: 140, cat: 4 }, { w: 'poor', x: 315, y: 220, cat: 4 },
  ];

  const CAT_NAMES = ['Countries', 'Colors', 'Animals', 'Verbs', 'Adjectives'];
  const CAT_COLORS = ['#4a9eff', '#f59e0b', '#22c55e', '#ef4444', '#a855f7'];

  let animationId = null, isVisible = false, canvasW = 0;
  let selectedWord = null;    // click se selected word index
  let analogyMode = false;    // vector arithmetic mode
  let analogyWords = [];      // [A, B, C] indices for A - B + C
  let showClusters = true;    // cluster labels dikhao
  // pan/zoom state
  let panX = 0, panY = 0, zoom = 1;
  let isDragging = false, dragStartX = 0, dragStartY = 0, panStartX = 0, panStartY = 0;

  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `width:100%;height:${CANVAS_HEIGHT}px;display:block;border-radius:6px;cursor:grab;background:#111;border:1px solid rgba(74,158,255,0.15);`;
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
  const analogyBtn = mkBtn(ctrl, 'Analogy: OFF', 'w2vAnalogy');
  analogyBtn.addEventListener('click', () => {
    analogyMode = !analogyMode;
    analogyWords = [];
    analogyBtn.textContent = 'Analogy: ' + (analogyMode ? 'ON' : 'OFF');
    analogyBtn.style.borderColor = analogyMode ? ACCENT : '#555';
    analogyBtn.style.background = analogyMode ? 'rgba(74,158,255,0.2)' : '#333';
    selectedWord = null;
    draw();
  });

  const clusterBtn = mkBtn(ctrl, 'Labels: ON', 'w2vLabels');
  clusterBtn.addEventListener('click', () => {
    showClusters = !showClusters;
    clusterBtn.textContent = 'Labels: ' + (showClusters ? 'ON' : 'OFF');
    draw();
  });

  mkBtn(ctrl, 'Reset View', 'w2vReset').addEventListener('click', () => {
    panX = 0; panY = 0; zoom = 1;
    draw();
  });

  // analogy status line
  const analogyInfo = document.createElement('div');
  analogyInfo.style.cssText = "font:11px 'JetBrains Mono',monospace;color:#888;margin-top:4px;min-height:18px;";
  container.appendChild(analogyInfo);

  // stats
  const stats = document.createElement('div');
  stats.style.cssText = "font:11px 'JetBrains Mono',monospace;color:#888;margin-top:2px;";
  container.appendChild(stats);

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- coordinate transforms — world se screen ---
  function toScreen(wx, wy) {
    // scale words to fill canvas, phir zoom aur pan lagao
    const scaleX = canvasW / 650;
    const scaleY = CANVAS_HEIGHT / 420;
    const s = Math.min(scaleX, scaleY) * zoom;
    return {
      x: wx * s + panX + (canvasW - 650 * s) / 2,
      y: wy * s + panY + (CANVAS_HEIGHT - 420 * s) / 2
    };
  }
  function toWorld(sx, sy) {
    const scaleX = canvasW / 650;
    const scaleY = CANVAS_HEIGHT / 420;
    const s = Math.min(scaleX, scaleY) * zoom;
    return {
      x: (sx - panX - (canvasW - 650 * s) / 2) / s,
      y: (sy - panY - (CANVAS_HEIGHT - 420 * s) / 2) / s
    };
  }

  // --- Euclidean distance in embedding space ---
  function embDist(i, j) {
    return Math.sqrt((WORDS[i].x - WORDS[j].x) ** 2 + (WORDS[i].y - WORDS[j].y) ** 2);
  }

  // --- nearest neighbors dhundho ---
  function findNearest(idx, k) {
    const dists = WORDS.map((_, i) => ({ i, d: embDist(idx, i) }))
      .filter(d => d.i !== idx)
      .sort((a, b) => a.d - b.d);
    return dists.slice(0, k);
  }

  // --- vector arithmetic: A - B + C ka nearest word dhundho ---
  function vectorArithmetic(aIdx, bIdx, cIdx) {
    const rx = WORDS[aIdx].x - WORDS[bIdx].x + WORDS[cIdx].x;
    const ry = WORDS[aIdx].y - WORDS[bIdx].y + WORDS[cIdx].y;
    // closest word dhundho (A, B, C ko skip karo)
    let minD = Infinity, minIdx = -1;
    WORDS.forEach((w, i) => {
      if (i === aIdx || i === bIdx || i === cIdx) return;
      const d = Math.sqrt((w.x - rx) ** 2 + (w.y - ry) ** 2);
      if (d < minD) { minD = d; minIdx = i; }
    });
    return { resultIdx: minIdx, rx, ry };
  }

  // --- click handler ---
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    // check karo koi word hit hua kya
    let hitIdx = -1;
    for (let i = 0; i < WORDS.length; i++) {
      const sp = toScreen(WORDS[i].x, WORDS[i].y);
      if (Math.abs(sp.x - sx) < 25 && Math.abs(sp.y - sy) < 12) { hitIdx = i; break; }
    }

    if (hitIdx >= 0) {
      if (analogyMode) {
        // analogy mode — 3 words select karo A, B, C
        if (analogyWords.length < 3) {
          analogyWords.push(hitIdx);
          if (analogyWords.length === 3) {
            // compute analogy result
            const result = vectorArithmetic(analogyWords[0], analogyWords[1], analogyWords[2]);
            const a = WORDS[analogyWords[0]].w;
            const b = WORDS[analogyWords[1]].w;
            const c = WORDS[analogyWords[2]].w;
            const r = result.resultIdx >= 0 ? WORDS[result.resultIdx].w : '?';
            analogyInfo.textContent = `${a} - ${b} + ${c} = ${r}`;
          } else {
            const labels = ['A', 'B', 'C'];
            const picked = analogyWords.map((idx, i) => labels[i] + '=' + WORDS[idx].w).join(', ');
            analogyInfo.textContent = `Pick ${3 - analogyWords.length} more: ${picked}`;
          }
        } else {
          // reset analogy
          analogyWords = [hitIdx];
          analogyInfo.textContent = 'A=' + WORDS[hitIdx].w + ', pick B and C';
        }
      } else {
        selectedWord = hitIdx === selectedWord ? null : hitIdx;
      }
      draw();
    } else {
      // pan start karo
      isDragging = true;
      dragStartX = sx; dragStartY = sy;
      panStartX = panX; panStartY = panY;
      canvas.style.cursor = 'grabbing';
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    panX = panStartX + (sx - dragStartX);
    panY = panStartY + (sy - dragStartY);
    draw();
  });

  canvas.addEventListener('mouseup', () => { isDragging = false; canvas.style.cursor = 'grab'; });
  canvas.addEventListener('mouseleave', () => { isDragging = false; canvas.style.cursor = 'grab'; });

  // zoom with scroll
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    zoom = Math.max(0.3, Math.min(3, zoom * zoomFactor));
    draw();
  }, { passive: false });

  // --- drawing ---
  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // cluster labels — background mein bade text
    if (showClusters) {
      ctx.font = "bold 16px 'JetBrains Mono',monospace";
      ctx.textAlign = 'center';
      // har cluster ka center dhundho
      for (let cat = 0; cat < 5; cat++) {
        const catWords = WORDS.filter(w => w.cat === cat);
        const cx = catWords.reduce((s, w) => s + w.x, 0) / catWords.length;
        const cy = catWords.reduce((s, w) => s + w.y, 0) / catWords.length;
        const sp = toScreen(cx, cy);
        ctx.fillStyle = CAT_COLORS[cat];
        ctx.globalAlpha = 0.12;
        ctx.fillText(CAT_NAMES[cat], sp.x, sp.y - 20 * zoom);
        ctx.globalAlpha = 1;
      }
    }

    // nearest neighbor lines — selected word ke liye
    if (selectedWord !== null && !analogyMode) {
      const neighbors = findNearest(selectedWord, 5);
      const sp = toScreen(WORDS[selectedWord].x, WORDS[selectedWord].y);
      neighbors.forEach(n => {
        const np = toScreen(WORDS[n.i].x, WORDS[n.i].y);
        ctx.strokeStyle = ACCENT;
        ctx.globalAlpha = 0.3;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(sp.x, sp.y); ctx.lineTo(np.x, np.y); ctx.stroke();
        ctx.globalAlpha = 1;
      });
    }

    // analogy visualization
    if (analogyMode && analogyWords.length === 3) {
      const result = vectorArithmetic(analogyWords[0], analogyWords[1], analogyWords[2]);
      // A -> B line (dashed red)
      const spA = toScreen(WORDS[analogyWords[0]].x, WORDS[analogyWords[0]].y);
      const spB = toScreen(WORDS[analogyWords[1]].x, WORDS[analogyWords[1]].y);
      const spC = toScreen(WORDS[analogyWords[2]].x, WORDS[analogyWords[2]].y);
      const spR = toScreen(result.rx, result.ry);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(spB.x, spB.y); ctx.lineTo(spA.x, spA.y); ctx.stroke();
      // C -> result (same vector, green)
      ctx.strokeStyle = '#22c55e';
      ctx.beginPath(); ctx.moveTo(spC.x, spC.y); ctx.lineTo(spR.x, spR.y); ctx.stroke();
      ctx.setLineDash([]);
      // result point
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath(); ctx.arc(spR.x, spR.y, 6, 0, Math.PI * 2); ctx.fill();
    }

    // words draw karo — dots + labels
    WORDS.forEach((w, i) => {
      const sp = toScreen(w.x, w.y);
      const isSelected = i === selectedWord;
      const isAnalogy = analogyWords.includes(i);
      const isAnalogyResult = analogyMode && analogyWords.length === 3 &&
        vectorArithmetic(analogyWords[0], analogyWords[1], analogyWords[2]).resultIdx === i;

      // dot
      const r = isSelected || isAnalogy || isAnalogyResult ? 6 : 3.5;
      ctx.fillStyle = isAnalogyResult ? '#f59e0b' : isAnalogy ? '#ef4444' : CAT_COLORS[w.cat];
      ctx.globalAlpha = isSelected || isAnalogy || isAnalogyResult ? 1 : 0.7;
      ctx.beginPath(); ctx.arc(sp.x, sp.y, r, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;

      // glow on selected/analogy
      if (isSelected || isAnalogy || isAnalogyResult) {
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = 10;
        ctx.beginPath(); ctx.arc(sp.x, sp.y, r, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
      }

      // label — sirf zoom enough ho ya selected ho tab dikhao
      if (zoom > 0.6 || isSelected || isAnalogy || isAnalogyResult) {
        ctx.font = (isSelected || isAnalogy || isAnalogyResult ? "bold 11px" : "9px") + " 'JetBrains Mono',monospace";
        ctx.fillStyle = isSelected || isAnalogy || isAnalogyResult ? '#fff' : '#aaa';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(w.w, sp.x, sp.y - r - 2);
      }
    });

    // selected word info
    if (selectedWord !== null && !analogyMode) {
      const w = WORDS[selectedWord];
      const neighbors = findNearest(selectedWord, 5);
      stats.textContent = `"${w.w}" [${CAT_NAMES[w.cat]}] → Nearest: ${neighbors.map(n => WORDS[n.i].w).join(', ')}`;
    } else if (!analogyMode) {
      stats.textContent = `${WORDS.length} words  |  5 clusters  |  Click word for neighbors  |  Drag to pan, scroll to zoom`;
    }

    // hint
    if (analogyMode && analogyWords.length < 3) {
      ctx.font = "12px 'JetBrains Mono',monospace";
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('Click 3 words for A - B + C analogy', canvasW / 2, CANVAS_HEIGHT - 15);
    }
  }

  // animation loop — har frame pe redraw taaki resize/pan smooth rahe
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    draw();
    animationId = requestAnimationFrame(loop);
  }

  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible) { draw(); if (!animationId) loop(); }
    else if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }, { threshold: 0.1 });
  obs.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && isVisible && !animationId) loop(); });

  draw();
}
