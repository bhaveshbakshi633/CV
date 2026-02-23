// ============================================================
// Multi-Armed Bandit — Exploration vs Exploitation ka classic problem
// 3 algorithms race karenge: epsilon-greedy, UCB1, Thompson Sampling
// ============================================================

// main entry point — container dhundho aur demo shuru karo
export function initBandit() {
  const container = document.getElementById('banditContainer');
  if (!container) {
    console.warn('banditContainer nahi mila bhai, bandit demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const NUM_ARMS = 5; // 5 slot machines
  const CANVAS_HEIGHT = 200; // slot machines ka visualization
  const CHART_HEIGHT = 130; // reward + regret charts
  const MAX_CHART_POINTS = 200; // chart mein kitne points dikhane hain
  const CHART_SAMPLE_INTERVAL = 5; // har 5 pulls pe ek data point

  // --- Colors — har algorithm ka apna rang ---
  const COLORS = {
    epsGreedy: { main: 'rgba(249,158,11,0.9)', dim: 'rgba(249,158,11,0.3)', label: 'e-greedy' },
    ucb: { main: 'rgba(74,158,255,0.9)', dim: 'rgba(74,158,255,0.3)', label: 'UCB1' },
    thompson: { main: 'rgba(74,255,143,0.9)', dim: 'rgba(74,255,143,0.3)', label: 'Thompson' },
  };

  // --- State ---
  let trueProbs = []; // har arm ki hidden true reward probability
  let showTruth = false; // true values dikhao ya nahi
  let pullsPerTick = 1; // speed control
  let totalPulls = 0;
  let isVisible = false;
  let animationId = null;
  let running = true; // auto-run by default

  // algorithm states — har ek ka alag state maintain karenge
  let algos = {};

  // chart data — reward aur regret track karenge
  let chartData = {
    epsGreedy: { rewards: [], regrets: [] },
    ucb: { rewards: [], regrets: [] },
    thompson: { rewards: [], regrets: [] },
  };

  // --- Algorithm initialization ---
  function initAlgorithms() {
    trueProbs = Array.from({ length: NUM_ARMS }, () => 0.1 + Math.random() * 0.8);
    totalPulls = 0;

    // epsilon-greedy state
    algos.epsGreedy = {
      counts: new Array(NUM_ARMS).fill(0),
      values: new Array(NUM_ARMS).fill(0),
      totalReward: 0,
      epsilon: 0.1,
    };

    // UCB1 state
    algos.ucb = {
      counts: new Array(NUM_ARMS).fill(0),
      values: new Array(NUM_ARMS).fill(0),
      totalReward: 0,
      totalPulls: 0,
    };

    // Thompson Sampling state — Beta distribution ke parameters
    algos.thompson = {
      alpha: new Array(NUM_ARMS).fill(1), // successes + 1
      beta: new Array(NUM_ARMS).fill(1),  // failures + 1
      counts: new Array(NUM_ARMS).fill(0),
      values: new Array(NUM_ARMS).fill(0),
      totalReward: 0,
    };

    // chart data reset
    Object.keys(chartData).forEach(k => {
      chartData[k].rewards = [];
      chartData[k].regrets = [];
    });
  }

  // --- Arm pull karo — reward milega ya nahi (Bernoulli) ---
  function pullArm(armIdx) {
    return Math.random() < trueProbs[armIdx] ? 1 : 0;
  }

  // --- Optimal arm dhundho — regret calculate karne ke liye ---
  function optimalProb() {
    return Math.max(...trueProbs);
  }

  // --- Epsilon-Greedy: epsilon chance se random, warna best estimated ---
  function epsGreedySelect(state) {
    if (Math.random() < state.epsilon) {
      return Math.floor(Math.random() * NUM_ARMS);
    }
    // best estimated value wala arm choose kar
    let bestArm = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < NUM_ARMS; i++) {
      if (state.values[i] > bestVal) {
        bestVal = state.values[i];
        bestArm = i;
      }
    }
    return bestArm;
  }

  // --- UCB1: upper confidence bound — exploration bonus add kar ---
  function ucbSelect(state) {
    // pehle sab arms ek baar try kar — cold start
    for (let i = 0; i < NUM_ARMS; i++) {
      if (state.counts[i] === 0) return i;
    }
    // UCB1 formula: Q(a) + sqrt(2 * ln(t) / N(a))
    let bestArm = 0;
    let bestUCB = -Infinity;
    const logT = Math.log(state.totalPulls);
    for (let i = 0; i < NUM_ARMS; i++) {
      const ucbVal = state.values[i] + Math.sqrt(2 * logT / state.counts[i]);
      if (ucbVal > bestUCB) {
        bestUCB = ucbVal;
        bestArm = i;
      }
    }
    return bestArm;
  }

  // --- Thompson Sampling: Beta distribution se sample kar ---
  function thompsonSelect(state) {
    // har arm ke liye Beta(alpha, beta) se sample le, sabse bada choose kar
    let bestArm = 0;
    let bestSample = -Infinity;
    for (let i = 0; i < NUM_ARMS; i++) {
      // Beta distribution sample — jdtsmith trick se approximate
      const sample = betaSample(state.alpha[i], state.beta[i]);
      if (sample > bestSample) {
        bestSample = sample;
        bestArm = i;
      }
    }
    return bestArm;
  }

  // --- Beta distribution sampling — Gamma distribution trick use karenge ---
  function gammaSample(shape) {
    // Marsaglia and Tsang's method for Gamma(shape, 1)
    if (shape < 1) {
      return gammaSample(shape + 1) * Math.pow(Math.random(), 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x, v;
      do {
        // Box-Muller normal sample
        x = normalSample();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  // standard normal sample — Box-Muller transform
  function normalSample() {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }

  // Beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b))
  function betaSample(a, b) {
    const x = gammaSample(a);
    const y = gammaSample(b);
    return x / (x + y + 1e-10);
  }

  // --- Ek step chala — teeno algorithms ko ek pull do ---
  function step() {
    totalPulls++;
    const opt = optimalProb();

    // Epsilon-Greedy ka turn
    const epsArm = epsGreedySelect(algos.epsGreedy);
    const epsReward = pullArm(epsArm);
    algos.epsGreedy.counts[epsArm]++;
    algos.epsGreedy.totalReward += epsReward;
    // incremental mean update — efficient hai
    const epsN = algos.epsGreedy.counts[epsArm];
    algos.epsGreedy.values[epsArm] += (epsReward - algos.epsGreedy.values[epsArm]) / epsN;

    // UCB1 ka turn
    algos.ucb.totalPulls++;
    const ucbArm = ucbSelect(algos.ucb);
    const ucbReward = pullArm(ucbArm);
    algos.ucb.counts[ucbArm]++;
    algos.ucb.totalReward += ucbReward;
    const ucbN = algos.ucb.counts[ucbArm];
    algos.ucb.values[ucbArm] += (ucbReward - algos.ucb.values[ucbArm]) / ucbN;

    // Thompson Sampling ka turn
    const thArm = thompsonSelect(algos.thompson);
    const thReward = pullArm(thArm);
    algos.thompson.counts[thArm]++;
    algos.thompson.totalReward += thReward;
    const thN = algos.thompson.counts[thArm];
    algos.thompson.values[thArm] += (thReward - algos.thompson.values[thArm]) / thN;
    // Beta distribution update — success/failure count
    if (thReward === 1) {
      algos.thompson.alpha[thArm]++;
    } else {
      algos.thompson.beta[thArm]++;
    }

    // chart data update — har CHART_SAMPLE_INTERVAL pe ek point
    if (totalPulls % CHART_SAMPLE_INTERVAL === 0) {
      const avgEps = algos.epsGreedy.totalReward / totalPulls;
      const avgUcb = algos.ucb.totalReward / totalPulls;
      const avgTh = algos.thompson.totalReward / totalPulls;

      chartData.epsGreedy.rewards.push(avgEps);
      chartData.ucb.rewards.push(avgUcb);
      chartData.thompson.rewards.push(avgTh);

      // cumulative regret: (optimal - actual) summed over time
      const epsRegret = totalPulls * opt - algos.epsGreedy.totalReward;
      const ucbRegret = totalPulls * opt - algos.ucb.totalReward;
      const thRegret = totalPulls * opt - algos.thompson.totalReward;

      chartData.epsGreedy.regrets.push(epsRegret);
      chartData.ucb.regrets.push(ucbRegret);
      chartData.thompson.regrets.push(thRegret);

      // chart truncate kar agar bahut bada ho gaya
      if (chartData.epsGreedy.rewards.length > MAX_CHART_POINTS) {
        Object.keys(chartData).forEach(k => {
          chartData[k].rewards.shift();
          chartData[k].regrets.shift();
        });
      }
    }
  }

  // --- DOM structure ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — slot machines ka visualization
  const mainCanvas = document.createElement('canvas');
  mainCanvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px',
    'background:transparent',
  ].join(';');
  container.appendChild(mainCanvas);

  // chart canvas — reward + regret comparison
  const chartCanvas = document.createElement('canvas');
  chartCanvas.style.cssText = [
    'width:100%',
    'height:' + CHART_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px',
    'margin-top:8px',
    'background:transparent',
  ].join(';');
  container.appendChild(chartCanvas);

  // stats bar
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'justify-content:center',
    'flex-wrap:wrap',
    'gap:16px',
    'margin-top:8px',
    'font-family:monospace',
    'font-size:12px',
    'color:#b0b0b0',
  ].join(';');
  container.appendChild(statsDiv);

  // controls bar
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

  // --- Button helper ---
  function makeButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'background:rgba(74,158,255,0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(74,158,255,0.25)',
      'border-radius:6px',
      'padding:6px 14px',
      'font-size:12px',
      'font-family:monospace',
      'cursor:pointer',
      'transition:all 0.2s',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(74,158,255,0.25)';
      btn.style.color = '#ffffff';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(74,158,255,0.1)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // reset button — naye random true values
  makeButton('Reset', () => {
    initAlgorithms();
  });

  // speed control
  const speedLabel = document.createElement('span');
  speedLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace;';
  speedLabel.textContent = 'Speed:';
  controlsDiv.appendChild(speedLabel);

  const speedSelect = document.createElement('select');
  speedSelect.style.cssText = [
    'background:rgba(74,158,255,0.1)',
    'color:#b0b0b0',
    'border:1px solid rgba(74,158,255,0.25)',
    'border-radius:6px',
    'padding:4px 8px',
    'font-size:12px',
    'font-family:monospace',
    'cursor:pointer',
  ].join(';');
  [
    { val: 1, label: '1/sec' },
    { val: 10, label: '10/sec' },
    { val: 100, label: '100/sec' },
  ].forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.val;
    opt.textContent = s.label;
    opt.style.cssText = 'background:#1a1a2e;color:#b0b0b0;';
    speedSelect.appendChild(opt);
  });
  speedSelect.addEventListener('change', () => {
    pullsPerTick = parseInt(speedSelect.value);
  });
  controlsDiv.appendChild(speedSelect);

  // reveal truth toggle
  const truthBtn = makeButton('Reveal Truth: OFF', () => {
    showTruth = !showTruth;
    truthBtn.textContent = 'Reveal Truth: ' + (showTruth ? 'ON' : 'OFF');
    if (showTruth) {
      truthBtn.style.borderColor = 'rgba(74,158,255,0.6)';
      truthBtn.style.color = '#4a9eff';
    } else {
      truthBtn.style.borderColor = 'rgba(74,158,255,0.25)';
      truthBtn.style.color = '#b0b0b0';
    }
  });

  // --- Canvas resize ---
  function resizeCanvases() {
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const dpr = window.devicePixelRatio || 1;

    mainCanvas.width = w * dpr;
    mainCanvas.height = CANVAS_HEIGHT * dpr;
    chartCanvas.width = w * dpr;
    chartCanvas.height = CHART_HEIGHT * dpr;

    mainCanvas.style.width = w + 'px';
    chartCanvas.style.width = w + 'px';
  }

  // --- Main canvas draw — slot machines ---
  function drawSlotMachines() {
    const ctx = mainCanvas.getContext('2d');
    const w = mainCanvas.width;
    const h = mainCanvas.height;
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, w, h);

    const padX = 30 * dpr;
    const padTop = 30 * dpr;
    const padBot = 40 * dpr;
    const barGap = 20 * dpr;
    const usableW = w - padX * 2;
    const barW = (usableW - barGap * (NUM_ARMS - 1)) / NUM_ARMS;
    const barMaxH = h - padTop - padBot;

    // har arm ke liye bar draw kar
    for (let i = 0; i < NUM_ARMS; i++) {
      const barX = padX + i * (barW + barGap);

      // background bar — full height, dim
      ctx.fillStyle = 'rgba(74,158,255,0.05)';
      ctx.beginPath();
      ctx.roundRect(barX, padTop, barW, barMaxH, 4 * dpr);
      ctx.fill();

      // true probability bar — agar reveal on hai toh dikhao
      if (showTruth) {
        const trueH = trueProbs[i] * barMaxH;
        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        ctx.beginPath();
        ctx.roundRect(barX, padTop + barMaxH - trueH, barW, trueH, 4 * dpr);
        ctx.fill();

        // true value label
        ctx.font = (10 * dpr) + 'px monospace';
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        ctx.textAlign = 'center';
        ctx.fillText('p=' + trueProbs[i].toFixed(2), barX + barW / 2, padTop + barMaxH - trueH - 5 * dpr);
      }

      // pull count intensity — zyada pull hua toh zyada bright
      const maxCount = Math.max(
        algos.epsGreedy.counts[i],
        algos.ucb.counts[i],
        algos.thompson.counts[i],
        1
      );
      const intensity = Math.min(1, maxCount / (totalPulls / NUM_ARMS + 1));
      ctx.fillStyle = 'rgba(74,158,255,' + (0.05 + intensity * 0.15) + ')';
      ctx.beginPath();
      ctx.roundRect(barX, padTop, barW, barMaxH, 4 * dpr);
      ctx.fill();

      // teeno algorithms ke estimated values — colored markers
      const algoList = [
        { key: 'epsGreedy', col: COLORS.epsGreedy },
        { key: 'ucb', col: COLORS.ucb },
        { key: 'thompson', col: COLORS.thompson },
      ];

      const markerR = 5 * dpr;
      const markerSpacing = barW / (algoList.length + 1);

      algoList.forEach((algo, aIdx) => {
        const est = algo.key === 'thompson'
          ? algos.thompson.values[i]
          : algos[algo.key].values[i];
        const markerY = padTop + barMaxH * (1 - Math.max(0, Math.min(1, est)));
        const markerX = barX + markerSpacing * (aIdx + 1);

        // marker dot
        ctx.fillStyle = algo.col.main;
        ctx.beginPath();
        ctx.arc(markerX, markerY, markerR, 0, Math.PI * 2);
        ctx.fill();

        // glow effect
        ctx.fillStyle = algo.col.dim;
        ctx.beginPath();
        ctx.arc(markerX, markerY, markerR * 2, 0, Math.PI * 2);
        ctx.fill();
      });

      // arm number label — neeche
      ctx.font = (11 * dpr) + 'px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.5)';
      ctx.textAlign = 'center';
      ctx.fillText('Arm ' + (i + 1), barX + barW / 2, h - 10 * dpr);

      // estimated value label — bar ke upar
      const avgEst = (algos.epsGreedy.values[i] + algos.ucb.values[i] + algos.thompson.values[i]) / 3;
      ctx.font = (9 * dpr) + 'px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.4)';
      ctx.fillText(avgEst.toFixed(2), barX + barW / 2, padTop - 8 * dpr);
    }

    // legend — top right mein
    const legendX = w - 140 * dpr;
    const legendY = 15 * dpr;
    ctx.font = (10 * dpr) + 'px monospace';
    Object.values(COLORS).forEach((col, idx) => {
      ctx.fillStyle = col.main;
      ctx.beginPath();
      ctx.arc(legendX, legendY + idx * 16 * dpr, 4 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgba(176,176,176,0.6)';
      ctx.textAlign = 'left';
      ctx.fillText(col.label, legendX + 10 * dpr, legendY + idx * 16 * dpr + 4 * dpr);
    });
  }

  // --- Chart canvas — reward comparison + regret ---
  function drawCharts() {
    const ctx = chartCanvas.getContext('2d');
    const w = chartCanvas.width;
    const h = chartCanvas.height;
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, w, h);

    // do charts side by side — left: avg reward, right: cumulative regret
    const halfW = w / 2;
    const padL = 35 * dpr;
    const padR = 10 * dpr;
    const padT = 20 * dpr;
    const padB = 15 * dpr;

    // --- Left chart: Average Reward ---
    function drawLineChart(offsetX, chartW, title, dataKey, autoRange) {
      const plotW = chartW - padL - padR;
      const plotH = h - padT - padB;

      // title
      ctx.font = (9 * dpr) + 'px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.5)';
      ctx.textAlign = 'center';
      ctx.fillText(title, offsetX + padL + plotW / 2, 12 * dpr);

      // find y range
      let yMin = Infinity, yMax = -Infinity;
      Object.keys(chartData).forEach(k => {
        const arr = chartData[k][dataKey];
        arr.forEach(v => {
          if (v < yMin) yMin = v;
          if (v > yMax) yMax = v;
        });
      });
      if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
      if (yMax - yMin < 0.01) { yMin -= 0.1; yMax += 0.1; }

      // grid
      ctx.strokeStyle = 'rgba(74,158,255,0.06)';
      ctx.lineWidth = 1;
      for (let g = 0; g <= 3; g++) {
        const gy = padT + plotH * (1 - g / 3);
        ctx.beginPath();
        ctx.moveTo(offsetX + padL, gy);
        ctx.lineTo(offsetX + padL + plotW, gy);
        ctx.stroke();
      }

      // y labels
      ctx.font = (8 * dpr) + 'px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.35)';
      ctx.textAlign = 'right';
      for (let g = 0; g <= 3; g++) {
        const val = yMin + (yMax - yMin) * (g / 3);
        const gy = padT + plotH * (1 - g / 3);
        ctx.fillText(val.toFixed(autoRange ? 0 : 2), offsetX + padL - 4 * dpr, gy + 3 * dpr);
      }

      // lines — har algorithm ke liye
      const algoKeys = ['epsGreedy', 'ucb', 'thompson'];
      const algoColors = [COLORS.epsGreedy.main, COLORS.ucb.main, COLORS.thompson.main];

      algoKeys.forEach((k, aIdx) => {
        const data = chartData[k][dataKey];
        if (data.length < 2) return;

        ctx.strokeStyle = algoColors[aIdx];
        ctx.lineWidth = 1.5 * dpr;
        ctx.lineJoin = 'round';
        ctx.beginPath();

        for (let i = 0; i < data.length; i++) {
          const x = offsetX + padL + (i / Math.max(data.length - 1, 1)) * plotW;
          const y = padT + plotH * (1 - (data[i] - yMin) / (yMax - yMin + 1e-8));
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
    }

    drawLineChart(0, halfW, 'Avg Reward', 'rewards', false);
    drawLineChart(halfW, halfW, 'Cumulative Regret', 'regrets', true);
  }

  // --- Stats update ---
  function updateStats() {
    // pehle saaf kar
    while (statsDiv.firstChild) statsDiv.removeChild(statsDiv.firstChild);

    function addStat(label, value, color) {
      const span = document.createElement('span');
      const labelNode = document.createTextNode(label);
      span.appendChild(labelNode);
      const valSpan = document.createElement('span');
      valSpan.style.color = color;
      valSpan.textContent = value;
      span.appendChild(valSpan);
      statsDiv.appendChild(span);
    }

    addStat('Pulls: ', totalPulls.toString(), '#4a9eff');
    addStat('e-greedy: ', algos.epsGreedy.totalReward.toFixed(1), COLORS.epsGreedy.main);
    addStat('UCB1: ', algos.ucb.totalReward.toFixed(1), COLORS.ucb.main);
    addStat('Thompson: ', algos.thompson.totalReward.toFixed(1), COLORS.thompson.main);
  }

  // --- Frame counter for speed control ---
  let frameCount = 0;

  // --- Animation loop ---
  function animate() {
    if (!isVisible) {
      animationId = requestAnimationFrame(animate);
      return;
    }

    frameCount++;

    // speed control — slow speed pe har 60th frame, fast pe har frame
    if (running) {
      let stepsThisFrame;
      if (pullsPerTick === 1) {
        // 1 pull per second — har 60th frame pe ek pull (assuming 60fps)
        stepsThisFrame = frameCount % 60 === 0 ? 1 : 0;
      } else if (pullsPerTick === 10) {
        // 10 pulls per second — har 6th frame pe ek pull
        stepsThisFrame = frameCount % 6 === 0 ? 1 : 0;
      } else {
        // 100 pulls per second — har frame pe 2 pulls (60fps * 2 = 120, close enough)
        stepsThisFrame = 2;
      }

      for (let i = 0; i < stepsThisFrame; i++) {
        step();
      }
    }

    drawSlotMachines();
    drawCharts();

    // stats har 10th frame pe update kar — DOM thrashing mat kar
    if (frameCount % 10 === 0) {
      updateStats();
    }

    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf visible hone pe animate kar ---
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        isVisible = entry.isIntersecting;
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);

  // --- Init sab kuch ---
  initAlgorithms();
  resizeCanvases();
  updateStats();

  window.addEventListener('resize', resizeCanvases);

  // animation shuru kar
  animate();
}
