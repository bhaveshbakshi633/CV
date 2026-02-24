// ============================================================
// Genetic Algorithm — TSP Solver Visualization
// Population evolve karke shortest route dhundhna hai cities ke beech
// Selection, crossover, mutation — evolution ka magic dekhlo
// ============================================================

// yahi main entry point hai — container dhundho, canvas banao, evolution shuru karo
export function initGenetic() {
  const container = document.getElementById('geneticContainer');
  if (!container) {
    console.warn('geneticContainer nahi mila bhai, genetic algorithm skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const MAIN_CANVAS_HEIGHT = 280;
  const CHART_CANVAS_HEIGHT = 120;
  const CITY_RADIUS = 5;
  const CITY_COLOR = '#4a9eff';
  const ROUTE_COLOR = 'rgba(74, 158, 255, 0.35)';
  const BEST_ROUTE_COLOR = '#4a9eff';
  const BEST_ROUTE_WIDTH = 2.5;
  const ELITISM_COUNT = 2; // top N seedha next generation mein jaayenge bina crossover ke
  const MAX_HISTORY = 300; // chart mein kitni generations dikhani hain
  const DEFAULT_CITY_COUNT = 18; // shuru mein kitni cities randomly place karni hain

  // --- State variables ---
  let cities = []; // [{x, y}, ...] — normalized coordinates (0-1 range)
  let population = []; // [[cityIndex, cityIndex, ...], ...] — har individual ek route hai
  let bestRoute = []; // sabse acchi route ab tak
  let bestDistance = Infinity;
  let generation = 0;
  let initialBestDistance = Infinity; // improvement percentage ke liye
  let isRunning = false; // evolution chal rahi hai ya pause hai
  let distanceHistory = []; // har generation ki best distance — chart ke liye
  let animationId = null;
  let isVisible = false;

  // tunable parameters — sliders se change honge
  let populationSize = 100;
  let mutationRate = 0.05;
  let crossoverRate = 0.85;
  let speedMultiplier = 1; // kitni generations per frame

  // --- DOM structure banate hain ---
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — cities aur routes yahan dikhenge
  const mainCanvas = document.createElement('canvas');
  mainCanvas.style.cssText = [
    'width:100%',
    'height:' + MAIN_CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.2)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
  ].join(';');
  container.appendChild(mainCanvas);
  const mainCtx = mainCanvas.getContext('2d');

  // chart canvas — fitness/distance over generations
  const chartCanvas = document.createElement('canvas');
  chartCanvas.style.cssText = [
    'width:100%',
    'height:' + CHART_CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.2)',
    'border-radius:8px',
    'margin-top:8px',
    'background:transparent',
  ].join(';');
  container.appendChild(chartCanvas);
  const chartCtx = chartCanvas.getContext('2d');

  // stats bar — generation, distance, improvement
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'justify-content:space-between',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:8px',
    'padding:8px 12px',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:12px',
    'color:#8a8a8a',
  ].join(';');
  container.appendChild(statsDiv);

  const genStat = document.createElement('span');
  genStat.textContent = 'Gen: 0';
  statsDiv.appendChild(genStat);

  const distStat = document.createElement('span');
  distStat.textContent = 'Best: --';
  statsDiv.appendChild(distStat);

  const improveStat = document.createElement('span');
  improveStat.textContent = 'Improv: --';
  statsDiv.appendChild(improveStat);

  const cityStat = document.createElement('span');
  cityStat.textContent = 'Cities: 0';
  statsDiv.appendChild(cityStat);

  // controls section — buttons + sliders
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:space-between',
  ].join(';');
  container.appendChild(controlsDiv);

  // buttons ka container
  const buttonsDiv = document.createElement('div');
  buttonsDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;';
  controlsDiv.appendChild(buttonsDiv);

  // sliders ka container
  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:14px;flex:1;min-width:280px;justify-content:flex-end;';
  controlsDiv.appendChild(slidersDiv);

  // preset layouts ka container — buttons ke neeche
  const presetsDiv = document.createElement('div');
  presetsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:6px',
    'align-items:center',
  ].join(';');
  container.appendChild(presetsDiv);

  const presetLabel = document.createElement('span');
  presetLabel.textContent = 'Layout:';
  presetLabel.style.cssText = 'color:#606060;font-size:12px;font-family:"JetBrains Mono",monospace;';
  presetsDiv.appendChild(presetLabel);

  // --- Button banane ka helper ---
  function createButton(text, onClick, isActive) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 14px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(74,158,255,0.08)',
      'color:#b0b0b0',
      'border:1px solid rgba(74,158,255,0.2)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(74,158,255,0.2)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      if (!isActive || !isActive()) {
        btn.style.background = 'rgba(74,158,255,0.08)';
        btn.style.color = '#b0b0b0';
      }
    });
    btn.addEventListener('click', onClick);
    return btn;
  }

  // --- Slider banane ka helper ---
  function createSlider(label, min, max, step, defaultVal, onChange, formatFn) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = 'color:#707070;font-size:11px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText = 'width:80px;height:4px;accent-color:rgba(74,158,255,0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText = 'color:#808080;font-size:11px;min-width:28px;font-family:"JetBrains Mono",monospace;';
    const fmt = formatFn || ((v) => v.toString());
    valueEl.textContent = fmt(parseFloat(defaultVal));
    wrapper.appendChild(valueEl);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = fmt(val);
      onChange(val);
    });

    slidersDiv.appendChild(wrapper);
    return { slider, valueEl };
  }

  // --- Main buttons ---
  const startBtn = createButton('Start', () => {
    if (cities.length < 3) return; // kam se kam 3 cities chahiye
    isRunning = !isRunning;
    startBtn.textContent = isRunning ? 'Pause' : 'Start';
    if (isRunning) {
      startBtn.style.background = 'rgba(74,158,255,0.25)';
      startBtn.style.color = '#e0e0e0';
    } else {
      startBtn.style.background = 'rgba(74,158,255,0.08)';
      startBtn.style.color = '#b0b0b0';
    }
  }, () => isRunning);
  buttonsDiv.appendChild(startBtn);

  const resetBtn = createButton('Reset', () => {
    isRunning = false;
    startBtn.textContent = 'Start';
    startBtn.style.background = 'rgba(74,158,255,0.08)';
    startBtn.style.color = '#b0b0b0';
    generation = 0;
    bestDistance = Infinity;
    initialBestDistance = Infinity;
    bestRoute = [];
    distanceHistory = [];
    population = [];
    if (cities.length >= 3) {
      initPopulation();
    }
    updateStats();
    drawMain();
    drawChart();
  });
  buttonsDiv.appendChild(resetBtn);

  const clearBtn = createButton('Clear', () => {
    isRunning = false;
    startBtn.textContent = 'Start';
    startBtn.style.background = 'rgba(74,158,255,0.08)';
    startBtn.style.color = '#b0b0b0';
    cities = [];
    population = [];
    bestRoute = [];
    bestDistance = Infinity;
    initialBestDistance = Infinity;
    generation = 0;
    distanceHistory = [];
    updateStats();
    drawMain();
    drawChart();
  });
  buttonsDiv.appendChild(clearBtn);

  // --- Sliders ---
  createSlider('Mut', 0.01, 0.30, 0.01, mutationRate, (v) => { mutationRate = v; }, (v) => v.toFixed(2));
  createSlider('Pop', 20, 200, 10, populationSize, (v) => {
    populationSize = v;
    // population size change hone pe reset karna padega
    if (cities.length >= 3) {
      isRunning = false;
      startBtn.textContent = 'Start';
      startBtn.style.background = 'rgba(74,158,255,0.08)';
      startBtn.style.color = '#b0b0b0';
      generation = 0;
      bestDistance = Infinity;
      initialBestDistance = Infinity;
      bestRoute = [];
      distanceHistory = [];
      initPopulation();
      updateStats();
      drawMain();
      drawChart();
    }
  }, (v) => v.toString());
  createSlider('Cross', 0.5, 1.0, 0.05, crossoverRate, (v) => { crossoverRate = v; }, (v) => v.toFixed(2));
  createSlider('Speed', 1, 50, 1, speedMultiplier, (v) => { speedMultiplier = v; }, (v) => v + 'x');

  // --- Preset layout buttons ---
  const presets = [
    { name: 'Random 15', fn: () => generateRandom(15) },
    { name: 'Random 25', fn: () => generateRandom(25) },
    { name: 'Circle', fn: generateCircle },
    { name: 'Grid', fn: generateGrid },
  ];

  presets.forEach((preset) => {
    const btn = createButton(preset.name, () => {
      isRunning = false;
      startBtn.textContent = 'Start';
      startBtn.style.background = 'rgba(74,158,255,0.08)';
      startBtn.style.color = '#b0b0b0';
      preset.fn();
      generation = 0;
      bestDistance = Infinity;
      initialBestDistance = Infinity;
      bestRoute = [];
      distanceHistory = [];
      initPopulation();
      updateStats();
      drawMain();
      drawChart();
    });
    presetsDiv.appendChild(btn);
  });

  // --- Canvas sizing — retina display ke liye DPR handle karna zaroori hai ---
  let canvasW = 0, canvasH = 0;
  let chartW = 0, chartH = 0;

  function resizeCanvases() {
    const dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;

    // main canvas
    canvasW = containerWidth;
    canvasH = MAIN_CANVAS_HEIGHT;
    mainCanvas.width = containerWidth * dpr;
    mainCanvas.height = MAIN_CANVAS_HEIGHT * dpr;
    mainCanvas.style.width = containerWidth + 'px';
    mainCanvas.style.height = MAIN_CANVAS_HEIGHT + 'px';
    mainCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // chart canvas
    chartW = containerWidth;
    chartH = CHART_CANVAS_HEIGHT;
    chartCanvas.width = containerWidth * dpr;
    chartCanvas.height = CHART_CANVAS_HEIGHT * dpr;
    chartCanvas.style.width = containerWidth + 'px';
    chartCanvas.style.height = CHART_CANVAS_HEIGHT + 'px';
    chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    drawMain();
    drawChart();
  }

  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);

  // --- City generation functions ---
  // normalized coordinates use kar rahe hain (0 to 1) taaki resize pe sab sahi rahe
  // drawing ke time canvas dimensions se multiply karenge

  function generateRandom(count) {
    cities = [];
    const padding = 0.08; // edges se thoda andar rakh, nahi toh dikhega nahi sahi se
    for (let i = 0; i < count; i++) {
      cities.push({
        x: padding + Math.random() * (1 - 2 * padding),
        y: padding + Math.random() * (1 - 2 * padding),
      });
    }
  }

  function generateCircle() {
    cities = [];
    const count = 20;
    const cx = 0.5, cy = 0.5;
    const radius = 0.38;
    for (let i = 0; i < count; i++) {
      const angle = (2 * Math.PI * i) / count;
      cities.push({
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
      });
    }
  }

  function generateGrid() {
    cities = [];
    const rows = 4, cols = 5;
    const padding = 0.12;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        cities.push({
          x: padding + (c / (cols - 1)) * (1 - 2 * padding),
          y: padding + (r / (rows - 1)) * (1 - 2 * padding),
        });
      }
    }
  }

  // --- Click to add cities ---
  function getCanvasPos(e) {
    const rect = mainCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) / rect.width, // normalized 0-1
      y: (clientY - rect.top) / rect.height,
    };
  }

  mainCanvas.addEventListener('click', (e) => {
    const pos = getCanvasPos(e);
    // boundary check — canvas ke andar hi add karo
    if (pos.x < 0.02 || pos.x > 0.98 || pos.y < 0.02 || pos.y > 0.98) return;

    cities.push({ x: pos.x, y: pos.y });

    // population re-initialize karo nayi city ke saath
    if (cities.length >= 3) {
      generation = 0;
      bestDistance = Infinity;
      initialBestDistance = Infinity;
      bestRoute = [];
      distanceHistory = [];
      initPopulation();
    }
    updateStats();
    drawMain();
    drawChart();
  });

  mainCanvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    if (pos.x < 0.02 || pos.x > 0.98 || pos.y < 0.02 || pos.y > 0.98) return;

    cities.push({ x: pos.x, y: pos.y });

    if (cities.length >= 3) {
      generation = 0;
      bestDistance = Infinity;
      initialBestDistance = Infinity;
      bestRoute = [];
      distanceHistory = [];
      initPopulation();
    }
    updateStats();
    drawMain();
    drawChart();
  }, { passive: false });

  // --- Distance calculation ---
  // do cities ke beech Euclidean distance — canvas pixel space mein
  function cityDist(a, b) {
    const dx = (cities[a].x - cities[b].x) * canvasW;
    const dy = (cities[a].y - cities[b].y) * canvasH;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // puri route ki total distance — circular tour hai, last se first pe bhi waapas
  function routeDistance(route) {
    let total = 0;
    for (let i = 0; i < route.length; i++) {
      const next = (i + 1) % route.length;
      total += cityDist(route[i], route[next]);
    }
    return total;
  }

  // --- Population initialization ---
  // random permutations se shuru karo — har individual ek valid route hai
  function initPopulation() {
    population = [];
    const n = cities.length;
    if (n < 3) return;

    for (let i = 0; i < populationSize; i++) {
      // Fisher-Yates shuffle se random permutation banao
      const route = Array.from({ length: n }, (_, idx) => idx);
      for (let j = n - 1; j > 0; j--) {
        const k = Math.floor(Math.random() * (j + 1));
        [route[j], route[k]] = [route[k], route[j]];
      }
      population.push(route);
    }

    // pehli generation ki best dhundho
    evaluatePopulation();
  }

  // --- Evaluate population — fitness nikalo sabki ---
  function evaluatePopulation() {
    let bestDist = Infinity;
    let bestIdx = 0;

    for (let i = 0; i < population.length; i++) {
      const dist = routeDistance(population[i]);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }

    if (bestDist < bestDistance) {
      bestDistance = bestDist;
      bestRoute = [...population[bestIdx]];
      // pehli baar improve hua toh initial distance set kar — improvement % ke liye
      if (initialBestDistance === Infinity) {
        initialBestDistance = bestDist;
      }
    }
  }

  // --- Tournament Selection ---
  // random k individuals mein se sabse accha choose karo
  // tournament size 3-5 accha hai — bahut bada karo toh diversity mar jaayegi
  function tournamentSelect(distances) {
    const tournamentSize = 3;
    let bestIdx = Math.floor(Math.random() * population.length);
    let bestDist = distances[bestIdx];

    for (let i = 1; i < tournamentSize; i++) {
      const idx = Math.floor(Math.random() * population.length);
      if (distances[idx] < bestDist) {
        bestDist = distances[idx];
        bestIdx = idx;
      }
    }
    return bestIdx;
  }

  // --- Ordered Crossover (OX) ---
  // TSP ke liye special crossover — normal crossover se duplicate cities aa jaayengi
  // ek segment parent1 se lo, baaki parent2 ke order mein bharo
  function orderedCrossover(parent1, parent2) {
    const n = parent1.length;
    // do random cut points choose karo
    let start = Math.floor(Math.random() * n);
    let end = Math.floor(Math.random() * n);
    if (start > end) [start, end] = [end, start];
    // kam se kam 1 element ka segment hona chahiye
    if (start === end) {
      end = Math.min(end + 1, n - 1);
    }

    // child banao — pehle segment parent1 se copy karo
    const child = new Array(n).fill(-1);
    const inChild = new Set();

    for (let i = start; i <= end; i++) {
      child[i] = parent1[i];
      inChild.add(parent1[i]);
    }

    // baaki positions parent2 ke order mein bharo — jo already nahi hai
    let pos = (end + 1) % n;
    for (let i = 0; i < n; i++) {
      const idx = (end + 1 + i) % n;
      const city = parent2[idx];
      if (!inChild.has(city)) {
        child[pos] = city;
        pos = (pos + 1) % n;
      }
    }

    return child;
  }

  // --- Swap Mutation ---
  // do random cities ki position swap kar do route mein
  function swapMutation(route) {
    const n = route.length;
    const mutated = [...route];
    if (Math.random() < mutationRate) {
      const i = Math.floor(Math.random() * n);
      let j = Math.floor(Math.random() * n);
      // same index pe swap ka matlab kuch nahi — doosra choose karo
      while (j === i) j = Math.floor(Math.random() * n);
      [mutated[i], mutated[j]] = [mutated[j], mutated[i]];
    }
    // high mutation rate pe occasionally double swap bhi kar — diversity badhao
    if (mutationRate > 0.1 && Math.random() < mutationRate * 0.5) {
      const i = Math.floor(Math.random() * n);
      let j = Math.floor(Math.random() * n);
      while (j === i) j = Math.floor(Math.random() * n);
      [mutated[i], mutated[j]] = [mutated[j], mutated[i]];
    }
    return mutated;
  }

  // --- Ek generation evolve karo ---
  function evolveGeneration() {
    if (population.length === 0 || cities.length < 3) return;

    const n = population.length;
    // sabki distance pehle nikal lo — bar bar calculate nahi karni
    const distances = population.map((route) => routeDistance(route));

    // sort by distance — elitism ke liye
    const indices = Array.from({ length: n }, (_, i) => i);
    indices.sort((a, b) => distances[a] - distances[b]);

    const newPopulation = [];

    // elitism — top individuals seedha next generation mein
    for (let i = 0; i < Math.min(ELITISM_COUNT, n); i++) {
      newPopulation.push([...population[indices[i]]]);
    }

    // baaki population crossover + mutation se banao
    while (newPopulation.length < populationSize) {
      // parents select karo tournament se
      const parent1Idx = tournamentSelect(distances);
      const parent2Idx = tournamentSelect(distances);

      let child;
      if (Math.random() < crossoverRate) {
        // crossover karo
        child = orderedCrossover(population[parent1Idx], population[parent2Idx]);
      } else {
        // crossover nahi — parent ka copy le lo
        child = [...population[parent1Idx]];
      }

      // mutation lagao
      child = swapMutation(child);
      newPopulation.push(child);
    }

    population = newPopulation;
    generation++;

    // evaluate new population
    evaluatePopulation();

    // history mein add karo — chart ke liye
    distanceHistory.push(bestDistance);
    if (distanceHistory.length > MAX_HISTORY) {
      distanceHistory.shift();
    }
  }

  // --- Stats update ---
  function updateStats() {
    genStat.textContent = 'Gen: ' + generation;
    if (bestDistance < Infinity && cities.length >= 3) {
      distStat.textContent = 'Best: ' + bestDistance.toFixed(1) + 'px';
      if (initialBestDistance < Infinity && initialBestDistance > 0) {
        const improvement = ((initialBestDistance - bestDistance) / initialBestDistance) * 100;
        improveStat.textContent = 'Improv: ' + improvement.toFixed(1) + '%';
        // improvement zyada ho toh green, kam ho toh default
        improveStat.style.color = improvement > 10 ? '#4ade80' : '#8a8a8a';
      } else {
        improveStat.textContent = 'Improv: --';
        improveStat.style.color = '#8a8a8a';
      }
    } else {
      distStat.textContent = 'Best: --';
      improveStat.textContent = 'Improv: --';
      improveStat.style.color = '#8a8a8a';
    }
    cityStat.textContent = 'Cities: ' + cities.length;
  }

  // --- Main canvas draw ---
  function drawMain() {
    const w = canvasW;
    const h = canvasH;
    mainCtx.clearRect(0, 0, w, h);

    // grid lines — subtle background pattern
    mainCtx.strokeStyle = 'rgba(74,158,255,0.04)';
    mainCtx.lineWidth = 1;
    const gridSpacing = 40;
    for (let x = gridSpacing; x < w; x += gridSpacing) {
      mainCtx.beginPath();
      mainCtx.moveTo(x, 0);
      mainCtx.lineTo(x, h);
      mainCtx.stroke();
    }
    for (let y = gridSpacing; y < h; y += gridSpacing) {
      mainCtx.beginPath();
      mainCtx.moveTo(0, y);
      mainCtx.lineTo(w, y);
      mainCtx.stroke();
    }

    if (cities.length === 0) {
      // empty state — hint dikhao
      mainCtx.fillStyle = 'rgba(74,158,255,0.25)';
      mainCtx.font = '14px "JetBrains Mono", monospace';
      mainCtx.textAlign = 'center';
      mainCtx.fillText('Click to add cities or choose a preset layout', w / 2, h / 2);
      mainCtx.textAlign = 'left';
      return;
    }

    // best route draw karo — agar hai toh
    if (bestRoute.length > 0 && cities.length >= 3) {
      // pehle population ki kuch random routes faint mein dikhao — diversity visualize ho
      const showCount = Math.min(8, population.length);
      for (let p = 0; p < showCount; p++) {
        // har 5th-10th route dikhao — sabko dikhane ki zarurat nahi
        const idx = Math.floor((p / showCount) * population.length);
        if (idx === 0) continue; // best route alag se draw karenge
        const route = population[idx];
        if (!route) continue;

        mainCtx.beginPath();
        mainCtx.strokeStyle = 'rgba(74,158,255,0.06)';
        mainCtx.lineWidth = 1;
        for (let i = 0; i <= route.length; i++) {
          const cityIdx = route[i % route.length];
          const cx = cities[cityIdx].x * w;
          const cy = cities[cityIdx].y * h;
          if (i === 0) mainCtx.moveTo(cx, cy);
          else mainCtx.lineTo(cx, cy);
        }
        mainCtx.stroke();
      }

      // best route — bright highlight
      mainCtx.beginPath();
      mainCtx.strokeStyle = BEST_ROUTE_COLOR;
      mainCtx.lineWidth = BEST_ROUTE_WIDTH;
      mainCtx.shadowColor = CITY_COLOR;
      mainCtx.shadowBlur = 6;
      for (let i = 0; i <= bestRoute.length; i++) {
        const cityIdx = bestRoute[i % bestRoute.length];
        const cx = cities[cityIdx].x * w;
        const cy = cities[cityIdx].y * h;
        if (i === 0) mainCtx.moveTo(cx, cy);
        else mainCtx.lineTo(cx, cy);
      }
      mainCtx.stroke();
      mainCtx.shadowBlur = 0;

      // route direction arrows — har segment ke beech mein chhoti arrow
      mainCtx.fillStyle = 'rgba(74,158,255,0.5)';
      for (let i = 0; i < bestRoute.length; i++) {
        const from = bestRoute[i];
        const to = bestRoute[(i + 1) % bestRoute.length];
        const fx = cities[from].x * w;
        const fy = cities[from].y * h;
        const tx = cities[to].x * w;
        const ty = cities[to].y * h;
        // midpoint pe arrow
        const mx = (fx + tx) / 2;
        const my = (fy + ty) / 2;
        const angle = Math.atan2(ty - fy, tx - fx);
        const arrowSize = 4;

        mainCtx.save();
        mainCtx.translate(mx, my);
        mainCtx.rotate(angle);
        mainCtx.beginPath();
        mainCtx.moveTo(arrowSize, 0);
        mainCtx.lineTo(-arrowSize, -arrowSize * 0.7);
        mainCtx.lineTo(-arrowSize, arrowSize * 0.7);
        mainCtx.closePath();
        mainCtx.fill();
        mainCtx.restore();
      }
    }

    // cities draw karo — sabse upar taaki route lines ke peeche na chhupen
    for (let i = 0; i < cities.length; i++) {
      const cx = cities[i].x * w;
      const cy = cities[i].y * h;

      // outer glow
      mainCtx.beginPath();
      mainCtx.arc(cx, cy, CITY_RADIUS + 3, 0, Math.PI * 2);
      mainCtx.fillStyle = 'rgba(74,158,255,0.15)';
      mainCtx.fill();

      // main dot
      mainCtx.beginPath();
      mainCtx.arc(cx, cy, CITY_RADIUS, 0, Math.PI * 2);
      mainCtx.fillStyle = CITY_COLOR;
      mainCtx.fill();

      // city number — chhota label
      mainCtx.fillStyle = 'rgba(255,255,255,0.5)';
      mainCtx.font = '9px "JetBrains Mono", monospace';
      mainCtx.textAlign = 'center';
      mainCtx.fillText(i.toString(), cx, cy - CITY_RADIUS - 5);
    }
    mainCtx.textAlign = 'left';
  }

  // --- Chart canvas draw — distance over generations ---
  function drawChart() {
    const w = chartW;
    const h = chartH;
    chartCtx.clearRect(0, 0, w, h);

    // background grid
    chartCtx.strokeStyle = 'rgba(74,158,255,0.04)';
    chartCtx.lineWidth = 1;
    for (let y = 20; y < h; y += 25) {
      chartCtx.beginPath();
      chartCtx.moveTo(0, y);
      chartCtx.lineTo(w, y);
      chartCtx.stroke();
    }

    if (distanceHistory.length < 2) {
      chartCtx.fillStyle = 'rgba(74,158,255,0.2)';
      chartCtx.font = '11px "JetBrains Mono", monospace';
      chartCtx.textAlign = 'center';
      chartCtx.fillText('Distance over generations will appear here', w / 2, h / 2);
      chartCtx.textAlign = 'left';
      return;
    }

    // padding for labels
    const padL = 50;
    const padR = 10;
    const padT = 18;
    const padB = 20;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // y-axis range — min aur max distance dhundho
    const minDist = Math.min(...distanceHistory);
    const maxDist = Math.max(...distanceHistory);
    // range thoda extend karo taaki line edge pe na chipke
    const range = maxDist - minDist || 1;
    const yMin = minDist - range * 0.05;
    const yMax = maxDist + range * 0.1;

    // y-axis labels
    chartCtx.fillStyle = '#505050';
    chartCtx.font = '9px "JetBrains Mono", monospace';
    chartCtx.textAlign = 'right';
    const ySteps = 3;
    for (let i = 0; i <= ySteps; i++) {
      const val = yMax - (i / ySteps) * (yMax - yMin);
      const y = padT + (i / ySteps) * plotH;
      chartCtx.fillText(val.toFixed(0), padL - 6, y + 3);
      // horizontal gridline
      chartCtx.strokeStyle = 'rgba(74,158,255,0.06)';
      chartCtx.beginPath();
      chartCtx.moveTo(padL, y);
      chartCtx.lineTo(padL + plotW, y);
      chartCtx.stroke();
    }
    chartCtx.textAlign = 'left';

    // x-axis label
    chartCtx.fillStyle = '#404040';
    chartCtx.font = '9px "JetBrains Mono", monospace';
    chartCtx.textAlign = 'center';
    chartCtx.fillText('Generation', padL + plotW / 2, h - 3);
    chartCtx.textAlign = 'left';

    // y-axis label
    chartCtx.save();
    chartCtx.translate(10, padT + plotH / 2);
    chartCtx.rotate(-Math.PI / 2);
    chartCtx.fillStyle = '#404040';
    chartCtx.font = '9px "JetBrains Mono", monospace';
    chartCtx.textAlign = 'center';
    chartCtx.fillText('Distance', 0, 0);
    chartCtx.restore();

    // gradient fill under the line
    const gradient = chartCtx.createLinearGradient(0, padT, 0, padT + plotH);
    gradient.addColorStop(0, 'rgba(74,158,255,0.15)');
    gradient.addColorStop(1, 'rgba(74,158,255,0.01)');

    // fill area
    chartCtx.beginPath();
    for (let i = 0; i < distanceHistory.length; i++) {
      const x = padL + (i / (distanceHistory.length - 1)) * plotW;
      const y = padT + ((yMax - distanceHistory[i]) / (yMax - yMin)) * plotH;
      if (i === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    }
    // close the area
    chartCtx.lineTo(padL + plotW, padT + plotH);
    chartCtx.lineTo(padL, padT + plotH);
    chartCtx.closePath();
    chartCtx.fillStyle = gradient;
    chartCtx.fill();

    // line draw karo
    chartCtx.beginPath();
    chartCtx.strokeStyle = CITY_COLOR;
    chartCtx.lineWidth = 1.5;
    for (let i = 0; i < distanceHistory.length; i++) {
      const x = padL + (i / (distanceHistory.length - 1)) * plotW;
      const y = padT + ((yMax - distanceHistory[i]) / (yMax - yMin)) * plotH;
      if (i === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    }
    chartCtx.stroke();

    // current best point — chart ke end pe bright dot
    if (distanceHistory.length > 0) {
      const lastX = padL + plotW;
      const lastY = padT + ((yMax - distanceHistory[distanceHistory.length - 1]) / (yMax - yMin)) * plotH;
      chartCtx.beginPath();
      chartCtx.arc(lastX, lastY, 3, 0, Math.PI * 2);
      chartCtx.fillStyle = CITY_COLOR;
      chartCtx.fill();
    }
  }

  // --- Animation loop ---
  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    if (isRunning && cities.length >= 3 && population.length > 0) {
      // speed multiplier ke hisaab se multiple generations per frame chalao
      const gens = Math.min(speedMultiplier, 50);
      for (let i = 0; i < gens; i++) {
        evolveGeneration();
      }
      updateStats();
      drawMain();
      drawChart();
    }

    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver — sirf jab visible ho tab animate karo ---
  // CPU waste nahi karni jab user dekh hi nahi raha
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
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
    { threshold: 0.1 } // 10% dikhe toh start karo
  );

  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) animate(); });

  // page hidden ho jaye (tab switch) toh bhi pause karo — battery bachao
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      // check karo ki container abhi bhi visible hai ya nahi
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // --- Initial setup — random cities se shuru karo ---
  generateRandom(DEFAULT_CITY_COUNT);
  initPopulation();
  updateStats();
  drawMain();
  drawChart();
}
