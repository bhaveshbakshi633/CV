// ============================================================
// Bayesian Inference — Beta-Binomial coin flip model
// Prior Beta(a,b) + data → Posterior Beta(a+heads, b+tails)
// Dekho kaise posterior narrow hota hai observations ke saath
// ============================================================

export function initBayesianInference() {
  const container = document.getElementById('bayesianInferenceContainer');
  if (!container) return;
  const CANVAS_HEIGHT = 400;
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

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvasW = container.clientWidth;
    canvas.width = canvasW * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

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

  // --- state ---
  let priorAlpha = 2, priorBeta = 2;
  let heads = 0, tails = 0;
  let trueProb = 0.65; // hidden true probability
  let flipHistory = []; // 'H' ya 'T' ka array

  // Beta function helpers — log gamma se Beta PDF nikaalo
  // Stirling approximation for log(gamma(x))
  function logGamma(x) {
    if (x <= 0) return 0;
    // Lanczos approximation — sufficient accuracy ke liye
    const g = 7;
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) {
      return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    }
    x -= 1;
    let a = c[0];
    const t = x + g + 0.5;
    for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  // Beta PDF: f(x; a, b) = x^(a-1) * (1-x)^(b-1) / B(a,b)
  function betaPDF(x, a, b) {
    if (x <= 0 || x >= 1) return 0;
    const logB = logGamma(a) + logGamma(b) - logGamma(a + b);
    return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB);
  }

  // likelihood function — binomial likelihood (proportional)
  function likelihoodFn(theta) {
    if (heads === 0 && tails === 0) return 1;
    return Math.pow(theta, heads) * Math.pow(1 - theta, tails);
  }

  // coin flip karo
  function flipCoin() {
    const result = Math.random() < trueProb;
    if (result) {
      heads++;
      flipHistory.push('H');
    } else {
      tails++;
      flipHistory.push('T');
    }
  }

  function reset() {
    heads = 0;
    tails = 0;
    flipHistory = [];
  }

  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);

    // plot area dimensions
    const padL = 60, padR = 30, padT = 40, padB = 60;
    const plotW = canvasW - padL - padR;
    const plotH = CANVAS_HEIGHT - padT - padB;

    // x-axis: theta 0 to 1
    // y-axis: density

    // pehle sab curves ki max density dhundho — Y axis scaling ke liye
    const N = 200;
    let maxDensity = 0;
    const priorVals = [], postVals = [], likeVals = [];
    const postAlpha = priorAlpha + heads;
    const postBeta = priorBeta + tails;

    // likelihood normalize karo
    let maxLike = 0;
    for (let i = 0; i <= N; i++) {
      const theta = i / N;
      const lv = likelihoodFn(theta);
      if (lv > maxLike) maxLike = lv;
    }

    for (let i = 0; i <= N; i++) {
      const theta = i / N;
      const pv = betaPDF(theta, priorAlpha, priorBeta);
      const postV = betaPDF(theta, postAlpha, postBeta);
      const lv = maxLike > 0 ? likelihoodFn(theta) / maxLike : 0;

      priorVals.push(pv);
      postVals.push(postV);
      likeVals.push(lv);

      if (pv > maxDensity) maxDensity = pv;
      if (postV > maxDensity) maxDensity = postV;
    }
    // likelihood ko scale karo taaki dikhne mein comparable rahe
    const likeScale = maxDensity * 0.6; // likelihood ko chhota dikhao
    if (maxDensity === 0) maxDensity = 1;

    // grid lines
    ctx.strokeStyle = 'rgba(74,158,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = padL + (i / 10) * plotW;
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    }
    for (let i = 0; i <= 5; i++) {
      const y = padT + (i / 5) * plotH;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    }

    // helper — curve draw karo
    function drawCurve(vals, color, fill, scale) {
      const s = scale || maxDensity;
      ctx.beginPath();
      for (let i = 0; i <= N; i++) {
        const x = padL + (i / N) * plotW;
        const y = padT + plotH - (vals[i] / s) * plotH * 0.9;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      if (fill) {
        ctx.lineTo(padL + plotW, padT + plotH);
        ctx.lineTo(padL, padT + plotH);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // prior curve — halki blue
    drawCurve(priorVals, 'rgba(74,158,255,0.5)', 'rgba(74,158,255,0.05)');

    // likelihood curve — orange, scaled
    const scaledLike = likeVals.map(v => v * likeScale);
    drawCurve(scaledLike, 'rgba(255,170,0,0.6)', 'rgba(255,170,0,0.05)', maxDensity);

    // posterior curve — bright green
    drawCurve(postVals, '#4ade80', 'rgba(74,222,128,0.08)');

    // true probability vertical line — dashed
    const trueX = padL + trueProb * plotW;
    ctx.setLineDash([5, 3]);
    ctx.strokeStyle = 'rgba(255,100,100,0.6)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(trueX, padT);
    ctx.lineTo(trueX, padT + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // MAP estimate — posterior ka peak
    const mapTheta = (postAlpha - 1) / (postAlpha + postBeta - 2);
    if (postAlpha > 1 && postBeta > 1) {
      const mapX = padL + Math.max(0, Math.min(1, mapTheta)) * plotW;
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(74,222,128,0.7)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(mapX, padT);
      ctx.lineTo(mapX, padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // posterior mean
    const postMean = postAlpha / (postAlpha + postBeta);
    const meanX = padL + postMean * plotW;
    ctx.beginPath();
    ctx.arc(meanX, padT + plotH + 15, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#4ade80';
    ctx.fill();

    // x-axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = "10px 'JetBrains Mono',monospace";
    ctx.textAlign = 'center';
    for (let i = 0; i <= 10; i++) {
      const x = padL + (i / 10) * plotW;
      ctx.fillText((i / 10).toFixed(1), x, padT + plotH + 30);
    }
    ctx.fillText('\u03B8 (probability of heads)', padL + plotW / 2, padT + plotH + 48);

    // y-axis label
    ctx.save();
    ctx.translate(15, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.textAlign = 'center';
    ctx.fillText('Density', 0, 0);
    ctx.restore();

    // legend — top right
    const lgX = canvasW - 200, lgY = 15;
    ctx.font = "11px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';

    ctx.fillStyle = 'rgba(74,158,255,0.7)';
    ctx.fillRect(lgX, lgY, 12, 3);
    ctx.fillText(`Prior Beta(${priorAlpha.toFixed(1)},${priorBeta.toFixed(1)})`, lgX + 18, lgY + 5);

    ctx.fillStyle = 'rgba(255,170,0,0.7)';
    ctx.fillRect(lgX, lgY + 16, 12, 3);
    ctx.fillText('Likelihood', lgX + 18, lgY + 21);

    ctx.fillStyle = '#4ade80';
    ctx.fillRect(lgX, lgY + 32, 12, 3);
    ctx.fillText(`Post Beta(${postAlpha.toFixed(1)},${postBeta.toFixed(1)})`, lgX + 18, lgY + 37);

    ctx.fillStyle = 'rgba(255,100,100,0.6)';
    ctx.fillRect(lgX, lgY + 48, 12, 3);
    ctx.fillText(`True \u03B8 = ${trueProb.toFixed(2)}`, lgX + 18, lgY + 53);

    // stats — top left
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = "11px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillText(`Flips: ${heads + tails}  |  H: ${heads}  T: ${tails}`, padL, 20);
    ctx.fillText(`Post Mean: ${postMean.toFixed(3)}  |  MAP: ${(postAlpha > 1 && postBeta > 1) ? mapTheta.toFixed(3) : 'N/A'}`, padL, 34);

    // recent flips — bottom mein chhoti coin icons
    const coinY = padT + plotH + 38;
    const maxShow = Math.min(flipHistory.length, Math.floor((plotW) / 14));
    const startIdx = Math.max(0, flipHistory.length - maxShow);
    for (let i = startIdx; i < flipHistory.length; i++) {
      const fx = padL + (i - startIdx) * 14;
      ctx.beginPath();
      ctx.arc(fx + 5, coinY, 5, 0, Math.PI * 2);
      ctx.fillStyle = flipHistory[i] === 'H' ? 'rgba(74,222,128,0.6)' : 'rgba(255,100,100,0.6)';
      ctx.fill();
    }
  }

  // --- controls ---
  const alphaSlider = mkSlider(ctrl, '\u03B1', 'biAlpha', 0.5, 10, priorAlpha, 0.5);
  alphaSlider.addEventListener('input', () => { priorAlpha = parseFloat(alphaSlider.value); });

  const betaSlider = mkSlider(ctrl, '\u03B2', 'biBeta', 0.5, 10, priorBeta, 0.5);
  betaSlider.addEventListener('input', () => { priorBeta = parseFloat(betaSlider.value); });

  const truePSlider = mkSlider(ctrl, 'True P', 'biTrueP', 0.05, 0.95, trueProb, 0.05);
  truePSlider.addEventListener('input', () => { trueProb = parseFloat(truePSlider.value); });

  const flipBtn = mkBtn(ctrl, 'Flip 1x', 'biFlip1');
  flipBtn.addEventListener('click', () => { flipCoin(); });

  const flip10Btn = mkBtn(ctrl, 'Flip 10x', 'biFlip10');
  flip10Btn.addEventListener('click', () => { for (let i = 0; i < 10; i++) flipCoin(); });

  const flip100Btn = mkBtn(ctrl, 'Flip 100x', 'biFlip100');
  flip100Btn.addEventListener('click', () => { for (let i = 0; i < 100; i++) flipCoin(); });

  const resetBtn = mkBtn(ctrl, 'Reset', 'biReset');
  resetBtn.addEventListener('click', reset);

  // --- render loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    draw();
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

  draw();
}
