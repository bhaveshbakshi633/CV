// ============================================================
// Evolution Creatures — 2D stick creatures evolve to walk
// Verlet physics + sinusoidal muscles + genetic evolution
// Tournament selection se best walkers dhundho
// ============================================================

export function initEvolutionCreatures() {
  const container = document.getElementById('evolutionCreaturesContainer');
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

  // --- helpers ---
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

  // --- constants ---
  const GRAVITY = 0.15;
  const GROUND_Y = CANVAS_HEIGHT - 40;
  const FRICTION = 0.8;
  const SIM_STEPS = 300; // kitne frames per evaluation
  const BONE_COUNT_MIN = 3, BONE_COUNT_MAX = 5;
  const MUSCLE_COUNT_MIN = 2, MUSCLE_COUNT_MAX = 3;
  const POP_SIZE = 40;

  // --- state ---
  let mutRate = 0.15;
  let generation = 0;
  let speedUp = 1;
  let population = [];
  let bestCreature = null;
  let bestFitness = -Infinity;
  let allTimeBestFitness = -Infinity;
  let fitnessHistory = [];
  let simFrame = 0;
  let evaluatingIdx = 0;
  let phase = 'eval'; // 'eval' ya 'show'
  let showFrame = 0;

  // --- genome structure ---
  // genome: { boneCount, boneLengths[], jointAngles[], muscles[{from,to,amp,freq,phase}] }
  function randomGenome() {
    const boneCount = BONE_COUNT_MIN + Math.floor(Math.random() * (BONE_COUNT_MAX - BONE_COUNT_MIN + 1));
    const boneLengths = [];
    const jointAngles = [];
    for (let i = 0; i < boneCount; i++) {
      boneLengths.push(20 + Math.random() * 30);
      jointAngles.push(-Math.PI / 3 + Math.random() * (2 * Math.PI / 3));
    }
    const muscleCount = MUSCLE_COUNT_MIN + Math.floor(Math.random() * (MUSCLE_COUNT_MAX - MUSCLE_COUNT_MIN + 1));
    const muscles = [];
    for (let i = 0; i < muscleCount; i++) {
      const from = Math.floor(Math.random() * (boneCount + 1));
      let to = Math.floor(Math.random() * (boneCount + 1));
      while (to === from) to = Math.floor(Math.random() * (boneCount + 1));
      muscles.push({
        from, to,
        amp: 0.5 + Math.random() * 3,
        freq: 0.5 + Math.random() * 4,
        phase: Math.random() * Math.PI * 2
      });
    }
    return { boneCount, boneLengths, jointAngles, muscles };
  }

  // genome se creature banao — verlet particles
  function createCreature(genome) {
    // joints as verlet particles
    const joints = [];
    let x = 150, y = GROUND_Y - 30;
    joints.push({ x, y, px: x, py: y });
    let angle = -Math.PI / 4;
    for (let i = 0; i < genome.boneCount; i++) {
      angle += genome.jointAngles[i];
      const len = genome.boneLengths[i];
      x += Math.cos(angle) * len;
      y += Math.sin(angle) * len;
      joints.push({ x, y, px: x, py: y });
    }
    // bones = constraints between consecutive joints
    const bones = [];
    for (let i = 0; i < genome.boneCount; i++) {
      const dx = joints[i + 1].x - joints[i].x;
      const dy = joints[i + 1].y - joints[i].y;
      bones.push({ a: i, b: i + 1, len: Math.sqrt(dx * dx + dy * dy) });
    }
    return { joints, bones, genome, startX: joints[0].x };
  }

  // verlet physics step
  function simStep(creature, t) {
    const { joints, bones, genome } = creature;
    // muscle forces apply karo
    for (const m of genome.muscles) {
      if (m.from >= joints.length || m.to >= joints.length) continue;
      const force = m.amp * Math.sin(m.freq * t * 0.1 + m.phase);
      const ja = joints[m.from], jb = joints[m.to];
      const dx = jb.x - ja.x, dy = jb.y - ja.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      // velocity mein add karo (verlet: position difference = velocity)
      jb.x += fx * 0.3;
      jb.y += fy * 0.3;
      ja.x -= fx * 0.3;
      ja.y -= fy * 0.3;
    }

    // verlet integration
    for (const j of joints) {
      const vx = (j.x - j.px) * 0.98;
      const vy = (j.y - j.py) * 0.98;
      j.px = j.x;
      j.py = j.y;
      j.x += vx;
      j.y += vy + GRAVITY;
    }

    // bone constraints solve karo — 5 iterations
    for (let iter = 0; iter < 5; iter++) {
      for (const b of bones) {
        const ja = joints[b.a], jb = joints[b.b];
        const dx = jb.x - ja.x, dy = jb.y - ja.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const diff = (dist - b.len) / dist * 0.5;
        const ox = dx * diff, oy = dy * diff;
        ja.x += ox; ja.y += oy;
        jb.x -= ox; jb.y -= oy;
      }
      // ground collision
      for (const j of joints) {
        if (j.y > GROUND_Y) {
          j.y = GROUND_Y;
          // friction
          const vx = j.x - j.px;
          j.px = j.x - vx * FRICTION;
        }
      }
    }
  }

  // fitness = average x displacement of all joints
  function evaluateFitness(genome) {
    const creature = createCreature(genome);
    for (let t = 0; t < SIM_STEPS; t++) {
      simStep(creature, t);
    }
    let totalDx = 0;
    for (const j of creature.joints) {
      totalDx += j.x;
    }
    const avgX = totalDx / creature.joints.length;
    return avgX - creature.startX;
  }

  // mutation
  function mutateGenome(genome) {
    const g = JSON.parse(JSON.stringify(genome));
    for (let i = 0; i < g.boneLengths.length; i++) {
      if (Math.random() < mutRate) {
        g.boneLengths[i] += (Math.random() - 0.5) * 15;
        g.boneLengths[i] = Math.max(10, Math.min(60, g.boneLengths[i]));
      }
      if (Math.random() < mutRate) {
        g.jointAngles[i] += (Math.random() - 0.5) * 0.5;
      }
    }
    for (const m of g.muscles) {
      if (Math.random() < mutRate) m.amp += (Math.random() - 0.5) * 1.0;
      if (Math.random() < mutRate) m.freq += (Math.random() - 0.5) * 1.0;
      if (Math.random() < mutRate) m.phase += (Math.random() - 0.5) * 1.0;
      m.amp = Math.max(0.1, Math.min(5, m.amp));
      m.freq = Math.max(0.1, Math.min(6, m.freq));
    }
    return g;
  }

  // tournament selection
  function tournamentSelect(pop, fitnesses) {
    const i = Math.floor(Math.random() * pop.length);
    const j = Math.floor(Math.random() * pop.length);
    return fitnesses[i] > fitnesses[j] ? pop[i] : pop[j];
  }

  // population initialize
  function initPopulation() {
    population = [];
    for (let i = 0; i < POP_SIZE; i++) {
      population.push(randomGenome());
    }
    generation = 0;
    bestFitness = -Infinity;
    allTimeBestFitness = -Infinity;
    fitnessHistory = [];
    bestCreature = null;
  }

  // ek generation evolve karo — synchronous evaluation
  function evolveGeneration() {
    const fitnesses = population.map(g => evaluateFitness(g));
    let bestIdx = 0;
    for (let i = 1; i < fitnesses.length; i++) {
      if (fitnesses[i] > fitnesses[bestIdx]) bestIdx = i;
    }
    bestFitness = fitnesses[bestIdx];
    if (bestFitness > allTimeBestFitness) {
      allTimeBestFitness = bestFitness;
      bestCreature = createCreature(population[bestIdx]);
    }
    fitnessHistory.push(bestFitness);
    if (fitnessHistory.length > 200) fitnessHistory.shift();

    // next generation banao
    const newPop = [];
    // elitism — top 2 seedha rakho
    const sorted = fitnesses.map((f, i) => ({ f, i })).sort((a, b) => b.f - a.f);
    newPop.push(JSON.parse(JSON.stringify(population[sorted[0].i])));
    newPop.push(JSON.parse(JSON.stringify(population[sorted[1].i])));

    while (newPop.length < POP_SIZE) {
      const parent = tournamentSelect(population, fitnesses);
      newPop.push(mutateGenome(parent));
    }
    population = newPop;
    generation++;
  }

  // --- Controls ---
  const mutSlider = mkSlider(ctrl, 'Mutation', 'ecMutRate', 0.01, 0.5, mutRate, 0.01);
  mutSlider.addEventListener('input', () => { mutRate = parseFloat(mutSlider.value); });

  const speedSlider = mkSlider(ctrl, 'Speed', 'ecSpeed', 1, 10, speedUp, 1);
  speedSlider.addEventListener('input', () => { speedUp = parseInt(speedSlider.value); });

  const genLabel = document.createElement('span');
  genLabel.style.cssText = "color:#4a9eff;font:12px 'JetBrains Mono',monospace";
  genLabel.textContent = 'Gen: 0';
  ctrl.appendChild(genLabel);

  const fitLabel = document.createElement('span');
  fitLabel.style.cssText = "color:#4a9eff;font:12px 'JetBrains Mono',monospace";
  fitLabel.textContent = 'Best: 0';
  ctrl.appendChild(fitLabel);

  const restartBtn = mkBtn(ctrl, 'Restart', 'ecRestart');
  restartBtn.addEventListener('click', () => {
    initPopulation();
    showFrame = 0;
  });

  // --- show best creature walking ---
  let showCreature = null;

  function drawCreature(creature, offsetX, alpha) {
    if (!creature) return;
    const { joints, bones } = creature;
    // bones draw karo
    ctx.strokeStyle = `rgba(74,158,255,${alpha})`;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    for (const b of bones) {
      const ja = joints[b.a], jb = joints[b.b];
      ctx.beginPath();
      ctx.moveTo(ja.x + offsetX, ja.y);
      ctx.lineTo(jb.x + offsetX, jb.y);
      ctx.stroke();
    }
    // joints draw karo
    for (const j of joints) {
      ctx.beginPath();
      ctx.arc(j.x + offsetX, j.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(74,158,255,${alpha})`;
      ctx.fill();
      ctx.strokeStyle = `rgba(255,255,255,${alpha * 0.5})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  function drawFitnessGraph() {
    if (fitnessHistory.length < 2) return;
    const gx = canvasW - 180, gy = 10, gw = 170, gh = 80;
    // background
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(gx, gy, gw, gh);
    ctx.strokeStyle = 'rgba(74,158,255,0.3)';
    ctx.strokeRect(gx, gy, gw, gh);
    // graph
    const maxF = Math.max(...fitnessHistory);
    const minF = Math.min(...fitnessHistory);
    const range = maxF - minF || 1;
    ctx.beginPath();
    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < fitnessHistory.length; i++) {
      const px = gx + (i / (fitnessHistory.length - 1)) * gw;
      const py = gy + gh - ((fitnessHistory[i] - minF) / range) * (gh - 10);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    // label
    ctx.fillStyle = 'rgba(74,158,255,0.7)';
    ctx.font = "9px 'JetBrains Mono',monospace";
    ctx.textAlign = 'left';
    ctx.fillText('Fitness / Gen', gx + 4, gy + 10);
  }

  function draw() {
    ctx.clearRect(0, 0, canvasW, CANVAS_HEIGHT);
    // ground
    ctx.fillStyle = 'rgba(74,158,255,0.08)';
    ctx.fillRect(0, GROUND_Y, canvasW, CANVAS_HEIGHT - GROUND_Y);
    ctx.strokeStyle = 'rgba(74,158,255,0.3)';
    ctx.beginPath();
    ctx.moveTo(0, GROUND_Y);
    ctx.lineTo(canvasW, GROUND_Y);
    ctx.stroke();
    // grid lines on ground for reference
    ctx.strokeStyle = 'rgba(74,158,255,0.08)';
    for (let x = 0; x < canvasW; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, GROUND_Y);
      ctx.lineTo(x, CANVAS_HEIGHT);
      ctx.stroke();
    }

    // best creature dikhao with camera follow
    if (showCreature) {
      // camera offset calculate karo — creature ko center mein rakho
      let avgX = 0;
      for (const j of showCreature.joints) avgX += j.x;
      avgX /= showCreature.joints.length;
      const cameraOff = canvasW / 3 - avgX;
      drawCreature(showCreature, cameraOff, 1.0);
      // distance marker
      const dist = avgX - showCreature.startX;
      ctx.fillStyle = '#4a9eff';
      ctx.font = "12px 'JetBrains Mono',monospace";
      ctx.textAlign = 'left';
      ctx.fillText(`Distance: ${dist.toFixed(1)}px`, 10, 25);
    }

    // fitness graph
    drawFitnessGraph();

    // generation info
    genLabel.textContent = `Gen: ${generation}`;
    fitLabel.textContent = `Best: ${allTimeBestFitness.toFixed(1)}px`;
  }

  // --- main loop ---
  function loop() {
    if (!isVisible) { animationId = null; return; }
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }

    for (let s = 0; s < speedUp; s++) {
      // har frame mein evolve karte raho background mein
      evolveGeneration();
    }

    // best creature ko replay karo
    if (allTimeBestFitness > -Infinity) {
      // har generation ke baad best creature ka fresh replay
      if (!showCreature || showFrame >= SIM_STEPS) {
        // best genome se nayi creature banao replay ke liye
        const sorted = population.map((g, i) => ({ g, f: evaluateFitness(g), i })).sort((a, b) => b.f - a.f);
        showCreature = createCreature(sorted[0].g);
        showFrame = 0;
      }
      simStep(showCreature, showFrame);
      showFrame++;
    }

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

  // --- init ---
  initPopulation();
}
