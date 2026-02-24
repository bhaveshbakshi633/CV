// ============================================================
// NEAT — NeuroEvolution of Augmenting Topologies
// 2D creatures obstacle avoid karenge, neural networks evolve honge
// Species color-coded, champion ka network topology dikhega real-time
// Innovation numbers se competing conventions problem solve hota hai
// ============================================================

// yahi main entry point hai — container dhundho, canvas banao, evolution shuru karo
export function initNeat() {
  const container = document.getElementById('neatContainer');
  if (!container) {
    console.warn('neatContainer nahi mila bhai, NEAT skip kar rahe hain');
    return;
  }

  // --- purane children hata do, fresh start ---
  while (container.firstChild) container.removeChild(container.firstChild);

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';
  const ACCENT_RGB = '74,158,255';

  // --- NEAT hyperparameters ---
  let populationSize = 50;
  let speedMultiplier = 1;
  let mutationRateMultiplier = 0.3; // slider se control hoga
  let gapSizeFactor = 1.0; // 1.0 = easy, 0.4 = hard

  // --- Game world constants ---
  const GRAVITY = 0.35;
  const JUMP_FORCE = -5.5;
  const DOWN_FORCE = 2.5;
  const CREATURE_RADIUS = 5;
  const SCROLL_SPEED = 2.5;
  const OBSTACLE_WIDTH = 25;
  const MIN_GAP_SIZE = 60;
  const MAX_GAP_SIZE = 130;
  const OBSTACLE_SPACING = 200; // pixels ke beech obstacles
  const WORLD_HEIGHT = CANVAS_HEIGHT; // game arena ki height
  const TIME_LIMIT = 2000; // max frames per generation

  // --- NEAT config ---
  const C1 = 1.0; // excess gene coefficient
  const C2 = 1.0; // disjoint gene coefficient
  const C3 = 0.4; // weight difference coefficient
  const COMPAT_THRESHOLD = 3.0; // compatibility distance threshold speciation ke liye
  const ELITISM_COUNT = 2;

  // --- Species ke liye predefined colors ---
  const SPECIES_COLORS = [
    '#4a9eff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8',
    '#ff922b', '#20c997', '#f06595', '#748ffc', '#69db7c',
    '#ffa94d', '#da77f2', '#63e6be', '#a9e34b', '#e599f7',
    '#74c0fc', '#ff8787', '#8ce99a', '#ffe066', '#b197fc',
  ];

  // --- Global innovation counter ---
  let globalInnovation = 0;
  // innovation history — {fromNode_toNode: innovationNumber}
  // taaki same structural mutation ko same innovation number mile
  let innovationHistory = {};

  // --- State variables ---
  let generation = 1;
  let allTimeBestFitness = 0;
  let genBestFitness = 0;
  let genAvgFitness = 0;
  let fitnessHistory = []; // {best, avg} per generation — chart ke liye
  const MAX_HISTORY = 50;
  let animationId = null;
  let isVisible = false;

  // population aur species
  let population = []; // array of {genome, creature, fitness, speciesId}
  let species = []; // array of {id, members[], representative, color}
  let speciesCounter = 0;

  // game state
  let obstacles = [];
  let frameCount = 0;
  let aliveCount = 0;
  let generationRunning = false;

  // champion ka genome — network visualization ke liye
  let championGenome = null;

  // ===================== NEAT Genome =====================

  // node types
  const NODE_INPUT = 'input';
  const NODE_OUTPUT = 'output';
  const NODE_HIDDEN = 'hidden';

  // input node IDs: 0-4, output node IDs: 5-6
  const INPUT_COUNT = 5;
  const OUTPUT_COUNT = 2;
  const INPUT_IDS = [0, 1, 2, 3, 4];
  const OUTPUT_IDS = [5, 6];
  let nextNodeId = 7; // hidden nodes yahan se shuru honge

  // naya innovation number lo — same structural mutation ka same number milega
  function getInnovation(fromNode, toNode) {
    const key = fromNode + '_' + toNode;
    if (innovationHistory[key] !== undefined) {
      return innovationHistory[key];
    }
    globalInnovation++;
    innovationHistory[key] = globalInnovation;
    return globalInnovation;
  }

  // naya genome banao — minimal network (inputs directly to outputs)
  function createGenome() {
    const nodes = [];
    // input nodes
    for (let i = 0; i < INPUT_COUNT; i++) {
      nodes.push({ id: INPUT_IDS[i], type: NODE_INPUT });
    }
    // output nodes
    for (let i = 0; i < OUTPUT_COUNT; i++) {
      nodes.push({ id: OUTPUT_IDS[i], type: NODE_OUTPUT });
    }

    const connections = [];
    // har input ko har output se connect kar — initial minimal topology
    for (let i = 0; i < INPUT_COUNT; i++) {
      for (let j = 0; j < OUTPUT_COUNT; j++) {
        connections.push({
          from: INPUT_IDS[i],
          to: OUTPUT_IDS[j],
          weight: (Math.random() * 2 - 1) * 1.0,
          enabled: true,
          innovation: getInnovation(INPUT_IDS[i], OUTPUT_IDS[j]),
        });
      }
    }

    return { nodes: nodes.slice(), connections: connections.slice() };
  }

  // genome ka deep copy banao
  function cloneGenome(g) {
    return {
      nodes: g.nodes.map(n => ({ id: n.id, type: n.type })),
      connections: g.connections.map(c => ({
        from: c.from, to: c.to, weight: c.weight,
        enabled: c.enabled, innovation: c.innovation,
      })),
    };
  }

  // ===================== Neural Network Forward Pass =====================

  // sigmoid activation — output 0-1 ke beech
  function sigmoid(x) {
    if (x > 10) return 1;
    if (x < -10) return 0;
    return 1.0 / (1.0 + Math.exp(-x));
  }

  // genome se network evaluate karo — topological sort karke
  function evaluateNetwork(genome, inputs) {
    // node values initialize karo
    const values = {};
    const computed = {};

    // input nodes ki value set karo
    for (let i = 0; i < INPUT_COUNT; i++) {
      values[INPUT_IDS[i]] = inputs[i];
      computed[INPUT_IDS[i]] = true;
    }

    // enabled connections ka adjacency list banao
    const incomingConnections = {};
    for (const conn of genome.connections) {
      if (!conn.enabled) continue;
      if (!incomingConnections[conn.to]) incomingConnections[conn.to] = [];
      incomingConnections[conn.to].push(conn);
    }

    // iterative evaluation — max 10 passes (cycles handle karne ke liye)
    // NEAT mein recurrent connections bhi ho sakte hain, lekin hum feedforward hi rakhenge
    for (let pass = 0; pass < 10; pass++) {
      let allComputed = true;
      for (const node of genome.nodes) {
        if (node.type === NODE_INPUT) continue;
        if (computed[node.id]) continue;

        const incoming = incomingConnections[node.id];
        if (!incoming || incoming.length === 0) {
          values[node.id] = 0;
          computed[node.id] = true;
          continue;
        }

        // check karo ki saare incoming nodes compute ho chuke hain
        let ready = true;
        for (const conn of incoming) {
          if (!computed[conn.from]) {
            ready = false;
            break;
          }
        }

        if (!ready) {
          allComputed = false;
          continue;
        }

        // weighted sum + sigmoid activation
        let sum = 0;
        for (const conn of incoming) {
          sum += (values[conn.from] || 0) * conn.weight;
        }
        values[node.id] = sigmoid(sum);
        computed[node.id] = true;
      }

      if (allComputed) break;
    }

    // agar koi node compute nahi hua toh 0.5 default rakh
    for (const node of genome.nodes) {
      if (!computed[node.id]) values[node.id] = 0.5;
    }

    // output values return karo
    return OUTPUT_IDS.map(id => values[id] || 0.5);
  }

  // ===================== NEAT Mutations =====================

  // weight mutation — gaussian noise add karo existing weights mein
  function mutateWeights(genome) {
    for (const conn of genome.connections) {
      if (Math.random() < 0.9) {
        // perturb — chhota change
        conn.weight += (Math.random() * 2 - 1) * 0.5 * mutationRateMultiplier;
        // clamp karo extreme values ke liye
        conn.weight = Math.max(-8, Math.min(8, conn.weight));
      } else {
        // completely naya random weight — exploration ke liye
        conn.weight = (Math.random() * 2 - 1) * 2;
      }
    }
  }

  // naya connection add karo — do random nodes ke beech
  function mutateAddConnection(genome) {
    // valid pairs dhundho — input→hidden, input→output, hidden→hidden, hidden→output
    // self-connections aur duplicate connections nahi chahiye
    const existingConns = new Set();
    for (const conn of genome.connections) {
      existingConns.add(conn.from + '_' + conn.to);
    }

    // possible source nodes (input + hidden)
    const sourceNodes = genome.nodes.filter(n => n.type !== NODE_OUTPUT);
    // possible target nodes (hidden + output)
    const targetNodes = genome.nodes.filter(n => n.type !== NODE_INPUT);

    // 20 attempts — random pair try karo
    for (let attempt = 0; attempt < 20; attempt++) {
      const from = sourceNodes[Math.floor(Math.random() * sourceNodes.length)];
      const to = targetNodes[Math.floor(Math.random() * targetNodes.length)];

      if (from.id === to.id) continue;
      if (existingConns.has(from.id + '_' + to.id)) continue;

      // naya connection add karo
      genome.connections.push({
        from: from.id,
        to: to.id,
        weight: (Math.random() * 2 - 1) * 1.0,
        enabled: true,
        innovation: getInnovation(from.id, to.id),
      });
      break;
    }
  }

  // naya node add karo — existing connection split karo
  function mutateAddNode(genome) {
    // sirf enabled connections mein se choose karo
    const enabledConns = genome.connections.filter(c => c.enabled);
    if (enabledConns.length === 0) return;

    const conn = enabledConns[Math.floor(Math.random() * enabledConns.length)];

    // purani connection disable karo
    conn.enabled = false;

    // naya hidden node banao
    const newNodeId = nextNodeId++;
    genome.nodes.push({ id: newNodeId, type: NODE_HIDDEN });

    // do nayi connections — from→newNode (weight 1) aur newNode→to (old weight)
    // isse network behavior initially same rahega
    genome.connections.push({
      from: conn.from,
      to: newNodeId,
      weight: 1.0,
      enabled: true,
      innovation: getInnovation(conn.from, newNodeId),
    });
    genome.connections.push({
      from: newNodeId,
      to: conn.to,
      weight: conn.weight,
      enabled: true,
      innovation: getInnovation(newNodeId, conn.to),
    });
  }

  // connection toggle — enable/disable random connection
  function mutateToggleConnection(genome) {
    if (genome.connections.length === 0) return;
    const conn = genome.connections[Math.floor(Math.random() * genome.connections.length)];
    conn.enabled = !conn.enabled;
  }

  // saari mutations apply karo probability ke hisaab se
  function mutateGenome(genome) {
    // weight mutation — 80% chance
    if (Math.random() < 0.8) {
      mutateWeights(genome);
    }
    // add connection — 25% chance (scaled by mutation rate)
    if (Math.random() < 0.25 * mutationRateMultiplier) {
      mutateAddConnection(genome);
    }
    // add node — 5% chance (scaled by mutation rate)
    if (Math.random() < 0.05 * mutationRateMultiplier) {
      mutateAddNode(genome);
    }
    // toggle connection — 5% chance
    if (Math.random() < 0.05 * mutationRateMultiplier) {
      mutateToggleConnection(genome);
    }
  }

  // ===================== NEAT Crossover =====================

  // do genomes ka crossover — innovation numbers se align karke
  // more fit parent ke excess/disjoint genes lete hain
  function crossover(parent1, parent2, fitness1, fitness2) {
    // ensure parent1 is the fitter one
    let p1 = parent1, p2 = parent2;
    if (fitness2 > fitness1) {
      p1 = parent2;
      p2 = parent1;
    }

    // innovation number se index karo
    const p1Conns = {};
    for (const c of p1.connections) {
      p1Conns[c.innovation] = c;
    }
    const p2Conns = {};
    for (const c of p2.connections) {
      p2Conns[c.innovation] = c;
    }

    // saare innovation numbers collect karo
    const allInnovations = new Set();
    for (const c of p1.connections) allInnovations.add(c.innovation);
    for (const c of p2.connections) allInnovations.add(c.innovation);

    const childConnections = [];
    const nodeIds = new Set();

    // input aur output nodes toh hamesha honge
    for (const id of INPUT_IDS) nodeIds.add(id);
    for (const id of OUTPUT_IDS) nodeIds.add(id);

    for (const innov of allInnovations) {
      const c1 = p1Conns[innov];
      const c2 = p2Conns[innov];

      if (c1 && c2) {
        // matching gene — random parent se lo
        const chosen = Math.random() < 0.5 ? c1 : c2;
        childConnections.push({
          from: chosen.from, to: chosen.to,
          weight: chosen.weight,
          // agar dono mein se kisi mein disabled hai toh 75% chance disabled rahega
          enabled: (!c1.enabled || !c2.enabled) ? (Math.random() < 0.75 ? false : true) : chosen.enabled,
          innovation: innov,
        });
        nodeIds.add(chosen.from);
        nodeIds.add(chosen.to);
      } else if (c1) {
        // excess/disjoint gene from fitter parent — include karo
        childConnections.push({
          from: c1.from, to: c1.to, weight: c1.weight,
          enabled: c1.enabled, innovation: innov,
        });
        nodeIds.add(c1.from);
        nodeIds.add(c1.to);
      }
      // c2 only (from less fit parent) — ignore karo
    }

    // child nodes banao
    const childNodes = [];
    for (const id of nodeIds) {
      // node type dhundho
      if (INPUT_IDS.includes(id)) {
        childNodes.push({ id, type: NODE_INPUT });
      } else if (OUTPUT_IDS.includes(id)) {
        childNodes.push({ id, type: NODE_OUTPUT });
      } else {
        childNodes.push({ id, type: NODE_HIDDEN });
      }
    }

    return { nodes: childNodes, connections: childConnections };
  }

  // ===================== Speciation =====================

  // do genomes ke beech compatibility distance nikalo
  function compatibilityDistance(genome1, genome2) {
    const g1 = {};
    let maxInnov1 = 0;
    for (const c of genome1.connections) {
      g1[c.innovation] = c;
      if (c.innovation > maxInnov1) maxInnov1 = c.innovation;
    }

    const g2 = {};
    let maxInnov2 = 0;
    for (const c of genome2.connections) {
      g2[c.innovation] = c;
      if (c.innovation > maxInnov2) maxInnov2 = c.innovation;
    }

    let excess = 0; // genes beyond the other's max innovation
    let disjoint = 0; // genes within range but not matching
    let matching = 0;
    let weightDiff = 0;

    const threshold = Math.min(maxInnov1, maxInnov2);

    const allInnovations = new Set();
    for (const c of genome1.connections) allInnovations.add(c.innovation);
    for (const c of genome2.connections) allInnovations.add(c.innovation);

    for (const innov of allInnovations) {
      const in1 = g1[innov] !== undefined;
      const in2 = g2[innov] !== undefined;

      if (in1 && in2) {
        matching++;
        weightDiff += Math.abs(g1[innov].weight - g2[innov].weight);
      } else if (innov > threshold) {
        excess++;
      } else {
        disjoint++;
      }
    }

    const N = Math.max(genome1.connections.length, genome2.connections.length, 1);
    const avgWeightDiff = matching > 0 ? weightDiff / matching : 0;

    return (C1 * excess / N) + (C2 * disjoint / N) + (C3 * avgWeightDiff);
  }

  // population ko species mein divide karo
  function speciate() {
    // pehle existing species ke members clear karo
    for (const sp of species) {
      sp.members = [];
    }

    // har individual ko kisi species mein daal
    for (const individual of population) {
      let placed = false;

      for (const sp of species) {
        // representative se compare karo
        const dist = compatibilityDistance(individual.genome, sp.representative);
        if (dist < COMPAT_THRESHOLD) {
          sp.members.push(individual);
          individual.speciesId = sp.id;
          placed = true;
          break;
        }
      }

      if (!placed) {
        // nayi species banao
        speciesCounter++;
        const newSpecies = {
          id: speciesCounter,
          members: [individual],
          representative: cloneGenome(individual.genome),
          color: SPECIES_COLORS[(speciesCounter - 1) % SPECIES_COLORS.length],
        };
        species.push(newSpecies);
        individual.speciesId = newSpecies.id;
      }
    }

    // khaali species hata do
    species = species.filter(sp => sp.members.length > 0);

    // har species ka representative update karo — random member
    for (const sp of species) {
      const randomIdx = Math.floor(Math.random() * sp.members.length);
      sp.representative = cloneGenome(sp.members[randomIdx].genome);
    }
  }

  // ===================== Selection & Reproduction =====================

  // tournament selection — size 3
  function tournamentSelect(pool) {
    let best = pool[Math.floor(Math.random() * pool.length)];
    for (let i = 1; i < 3; i++) {
      const candidate = pool[Math.floor(Math.random() * pool.length)];
      if (candidate.fitness > best.fitness) {
        best = candidate;
      }
    }
    return best;
  }

  // nayi generation banao
  function createNextGeneration() {
    // fitness sharing — species size se divide karo
    for (const sp of species) {
      for (const individual of sp.members) {
        individual.adjustedFitness = individual.fitness / sp.members.length;
      }
    }

    // saari population ko fitness se sort karo
    const sorted = [...population].sort((a, b) => b.fitness - a.fitness);

    // generation stats update karo
    genBestFitness = sorted[0].fitness;
    genAvgFitness = population.reduce((sum, ind) => sum + ind.fitness, 0) / population.length;
    if (genBestFitness > allTimeBestFitness) {
      allTimeBestFitness = genBestFitness;
    }

    // champion genome save karo visualization ke liye
    championGenome = cloneGenome(sorted[0].genome);

    // fitness history mein add karo
    fitnessHistory.push({ best: genBestFitness, avg: genAvgFitness });
    if (fitnessHistory.length > MAX_HISTORY) fitnessHistory.shift();

    const newPopulation = [];

    // elitism — top individuals seedha next gen mein
    for (let i = 0; i < Math.min(ELITISM_COUNT, sorted.length); i++) {
      newPopulation.push({
        genome: cloneGenome(sorted[i].genome),
        creature: null,
        fitness: 0,
        speciesId: sorted[i].speciesId,
      });
    }

    // baaki population crossover + mutation se
    // har species ko proportional offspring milenge based on total adjusted fitness
    const totalAdjFitness = population.reduce((sum, ind) => sum + (ind.adjustedFitness || 0), 0);

    while (newPopulation.length < populationSize) {
      // species select karo — weighted by adjusted fitness
      let parent1, parent2;

      if (totalAdjFitness > 0 && species.length > 0) {
        // species level selection — interspecies breeding 5% chance
        if (Math.random() < 0.05 && species.length > 1) {
          const sp1 = species[Math.floor(Math.random() * species.length)];
          const sp2 = species[Math.floor(Math.random() * species.length)];
          parent1 = tournamentSelect(sp1.members);
          parent2 = tournamentSelect(sp2.members);
        } else {
          // same species se parents select karo
          // fitness proportional species selection
          let r = Math.random() * totalAdjFitness;
          let chosenSpecies = species[0];
          for (const sp of species) {
            const spFitness = sp.members.reduce((sum, ind) => sum + (ind.adjustedFitness || 0), 0);
            r -= spFitness;
            if (r <= 0) {
              chosenSpecies = sp;
              break;
            }
          }
          parent1 = tournamentSelect(chosenSpecies.members);
          parent2 = tournamentSelect(chosenSpecies.members);
        }
      } else {
        // fallback — random parents
        parent1 = sorted[Math.floor(Math.random() * sorted.length)];
        parent2 = sorted[Math.floor(Math.random() * sorted.length)];
      }

      // crossover karo
      let childGenome = crossover(parent1.genome, parent2.genome, parent1.fitness, parent2.fitness);

      // mutation lagao
      mutateGenome(childGenome);

      newPopulation.push({
        genome: childGenome,
        creature: null,
        fitness: 0,
        speciesId: 0,
      });
    }

    population = newPopulation;

    // speciation — nayi population ko species mein baant
    speciate();

    generation++;
  }

  // ===================== Game World =====================

  // creature banao — position, velocity, alive status
  function createCreature() {
    return {
      x: 60,
      y: WORLD_HEIGHT / 2 + (Math.random() - 0.5) * 80,
      vy: 0,
      alive: true,
      distance: 0,
    };
  }

  // obstacles generate karo — vertical bars with gaps
  function generateObstacles() {
    obstacles = [];
    // pehla obstacle thoda aage se shuru — creatures ko settle hone do
    let x = 350;
    // kaafi saare obstacles pre-generate karo
    for (let i = 0; i < 100; i++) {
      const gapSize = MIN_GAP_SIZE + (MAX_GAP_SIZE - MIN_GAP_SIZE) * gapSizeFactor;
      const gapTop = 40 + Math.random() * (WORLD_HEIGHT - gapSize - 80);
      obstacles.push({
        x: x,
        gapTop: gapTop,
        gapBottom: gapTop + gapSize,
        passed: false,
      });
      x += OBSTACLE_SPACING;
    }
  }

  // next obstacle dhundho jo creature ke aage ho
  function getNextObstacle(creatureX) {
    for (const obs of obstacles) {
      if (obs.x + OBSTACLE_WIDTH > creatureX) {
        return obs;
      }
    }
    return obstacles[obstacles.length - 1];
  }

  // creature ka ek physics step
  function stepCreature(individual) {
    const creature = individual.creature;
    if (!creature.alive) return;

    // neural network ke inputs banao — normalized 0-1
    const nextObs = getNextObstacle(creature.x);
    const distToObs = nextObs ? (nextObs.x - creature.x) / 400 : 1.0;
    const gapTop = nextObs ? nextObs.gapTop / WORLD_HEIGHT : 0.5;
    const gapBottom = nextObs ? nextObs.gapBottom / WORLD_HEIGHT : 0.5;
    const creatureY = creature.y / WORLD_HEIGHT;
    const creatureVy = (creature.vy + 10) / 20; // normalize velocity

    const inputs = [creatureY, distToObs, gapTop, gapBottom, creatureVy];

    // network evaluate karo
    const outputs = evaluateNetwork(individual.genome, inputs);

    // outputs se action decide karo
    if (outputs[0] > 0.5) {
      // jump / move up
      creature.vy = JUMP_FORCE;
    }
    if (outputs[1] > 0.5) {
      // move down — actively neeche jao
      creature.vy += DOWN_FORCE;
    }

    // gravity apply karo
    creature.vy += GRAVITY;
    // terminal velocity limit
    creature.vy = Math.max(-8, Math.min(8, creature.vy));
    creature.y += creature.vy;

    // distance track karo — yahi fitness hai
    creature.distance += SCROLL_SPEED;

    // collision checks
    // ceiling/floor
    if (creature.y < CREATURE_RADIUS || creature.y > WORLD_HEIGHT - CREATURE_RADIUS) {
      creature.alive = false;
      individual.fitness = creature.distance;
      return;
    }

    // obstacle collision
    for (const obs of obstacles) {
      // kya creature obstacle ke x-range mein hai?
      if (creature.x + CREATURE_RADIUS > obs.x &&
          creature.x - CREATURE_RADIUS < obs.x + OBSTACLE_WIDTH) {
        // kya gap ke bahar hai?
        if (creature.y - CREATURE_RADIUS < obs.gapTop ||
            creature.y + CREATURE_RADIUS > obs.gapBottom) {
          creature.alive = false;
          individual.fitness = creature.distance;
          return;
        }
      }
    }
  }

  // ek frame ka game step — sabhi creatures update karo
  function gameStep() {
    if (!generationRunning) return;

    // obstacles scroll karo (creatures fixed x pe hain, obstacles aate hain)
    for (const obs of obstacles) {
      obs.x -= SCROLL_SPEED;
    }

    // purane obstacles hata do
    while (obstacles.length > 0 && obstacles[0].x + OBSTACLE_WIDTH < 0) {
      obstacles.shift();
    }

    // naye obstacles add karo agar zarurat ho
    const lastObs = obstacles[obstacles.length - 1];
    if (lastObs && lastObs.x < 800) {
      const gapSize = MIN_GAP_SIZE + (MAX_GAP_SIZE - MIN_GAP_SIZE) * gapSizeFactor;
      const gapTop = 40 + Math.random() * (WORLD_HEIGHT - gapSize - 80);
      obstacles.push({
        x: lastObs.x + OBSTACLE_SPACING,
        gapTop: gapTop,
        gapBottom: gapTop + gapSize,
        passed: false,
      });
    }

    // har creature ko step karo
    aliveCount = 0;
    for (const individual of population) {
      if (individual.creature && individual.creature.alive) {
        stepCreature(individual);
        if (individual.creature.alive) aliveCount++;
      }
    }

    frameCount++;

    // generation khatam check karo — sab mare ya time limit
    if (aliveCount === 0 || frameCount >= TIME_LIMIT) {
      // jo abhi bhi alive hain unki fitness set karo
      for (const individual of population) {
        if (individual.creature && individual.creature.alive) {
          individual.fitness = individual.creature.distance;
        }
      }
      generationRunning = false;
    }
  }

  // nayi generation shuru karo
  function startGeneration() {
    generateObstacles();
    frameCount = 0;
    aliveCount = populationSize;

    // har individual ko creature assign karo
    for (const individual of population) {
      individual.creature = createCreature();
      individual.fitness = 0;
    }

    generationRunning = true;
  }

  // population initialize karo
  function initPopulation() {
    population = [];
    species = [];
    speciesCounter = 0;
    generation = 1;
    allTimeBestFitness = 0;
    genBestFitness = 0;
    genAvgFitness = 0;
    fitnessHistory = [];
    championGenome = null;
    // innovation history reset — fresh start
    globalInnovation = 0;
    innovationHistory = {};
    nextNodeId = 7;

    for (let i = 0; i < populationSize; i++) {
      population.push({
        genome: createGenome(),
        creature: null,
        fitness: 0,
        speciesId: 0,
      });
    }

    // initial speciation
    speciate();

    // pehli generation shuru karo
    startGeneration();
  }

  // ===================== DOM Structure =====================

  // canvas banao — dynamically
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.2)',
    'border-radius:8px',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // stats bar
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = [
    'display:flex',
    'justify-content:space-between',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:8px',
    'padding:8px 12px',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:11px',
    'color:#8a8a8a',
  ].join(';');
  container.appendChild(statsDiv);

  // stat elements banao — safe DOM methods (no innerHTML)
  function createStatEl(labelText, valueText, valueColor) {
    const span = document.createElement('span');
    const label = document.createTextNode(labelText);
    const valueSpan = document.createElement('span');
    valueSpan.style.color = valueColor || ACCENT;
    valueSpan.textContent = valueText;
    span.appendChild(label);
    span.appendChild(valueSpan);
    return span;
  }

  const genStat = createStatEl('Gen: ', '1', ACCENT);
  const aliveStat = createStatEl('Alive: ', '0/0', ACCENT);
  const speciesStat = createStatEl('Species: ', '0', ACCENT);
  const bestStat = createStatEl('Best: ', '0', ACCENT);
  const allTimeStat = createStatEl('ATB: ', '0', '#4aff8f');
  [genStat, aliveStat, speciesStat, bestStat, allTimeStat].forEach(el => statsDiv.appendChild(el));

  // controls section
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:12px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Slider helper ---
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
    slider.style.cssText = 'width:70px;height:4px;accent-color:rgba(' + ACCENT_RGB + ',0.8);cursor:pointer;';
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

    controlsDiv.appendChild(wrapper);
    return { slider, valueEl };
  }

  // --- Button helper ---
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 14px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.08)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.2)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.2)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.08)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // --- Controls banao ---
  createSlider('Pop', 20, 100, 5, populationSize, (v) => { populationSize = v; }, (v) => v.toString());
  createSlider('Speed', 1, 10, 1, speedMultiplier, (v) => { speedMultiplier = v; }, (v) => v + 'x');
  createSlider('Mut', 0.1, 0.5, 0.05, mutationRateMultiplier, (v) => { mutationRateMultiplier = v; }, (v) => v.toFixed(2));
  createSlider('Diff', 0.4, 1.0, 0.1, gapSizeFactor, (v) => { gapSizeFactor = v; }, (v) => {
    if (v >= 0.8) return 'Easy';
    if (v >= 0.6) return 'Med';
    return 'Hard';
  });

  // generation counter display — prominent
  const genDisplay = document.createElement('span');
  genDisplay.style.cssText = [
    'color:' + ACCENT,
    'font-size:13px',
    'font-family:"JetBrains Mono",monospace',
    'font-weight:bold',
    'padding:4px 10px',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.3)',
    'border-radius:6px',
  ].join(';');
  genDisplay.textContent = 'Gen 1';
  controlsDiv.appendChild(genDisplay);

  // restart button
  createButton('Restart', () => {
    initPopulation();
    updateStats();
  });

  // ===================== Canvas Resize =====================

  let canvasW = 0, canvasH = 0;

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;

    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;
    canvas.width = containerWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // ===================== Drawing =====================

  // species ka color dhundho
  function getSpeciesColor(speciesId) {
    for (const sp of species) {
      if (sp.id === speciesId) return sp.color;
    }
    return '#666666';
  }

  // game arena draw karo — left 60% of canvas
  function drawArena() {
    const arenaW = canvasW * 0.6;
    const arenaH = canvasH;

    // subtle border for arena
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(arenaW, 0);
    ctx.lineTo(arenaW, arenaH);
    ctx.stroke();

    // background grid — barely visible
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.03)';
    ctx.lineWidth = 1;
    for (let x = 0; x < arenaW; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, arenaH);
      ctx.stroke();
    }
    for (let y = 0; y < arenaH; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(arenaW, y);
      ctx.stroke();
    }

    // ceiling aur floor lines
    ctx.strokeStyle = 'rgba(255,100,100,0.2)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(arenaW, 0);
    ctx.moveTo(0, arenaH);
    ctx.lineTo(arenaW, arenaH);
    ctx.stroke();

    // obstacles draw karo
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, arenaW, arenaH);
    ctx.clip();

    for (const obs of obstacles) {
      if (obs.x > arenaW + 20 || obs.x + OBSTACLE_WIDTH < -20) continue;

      // top bar
      ctx.fillStyle = 'rgba(255,100,100,0.15)';
      ctx.fillRect(obs.x, 0, OBSTACLE_WIDTH, obs.gapTop);
      ctx.strokeStyle = 'rgba(255,100,100,0.3)';
      ctx.lineWidth = 1;
      ctx.strokeRect(obs.x, 0, OBSTACLE_WIDTH, obs.gapTop);

      // bottom bar
      ctx.fillStyle = 'rgba(255,100,100,0.15)';
      ctx.fillRect(obs.x, obs.gapBottom, OBSTACLE_WIDTH, arenaH - obs.gapBottom);
      ctx.strokeStyle = 'rgba(255,100,100,0.3)';
      ctx.strokeRect(obs.x, obs.gapBottom, OBSTACLE_WIDTH, arenaH - obs.gapBottom);

      // gap highlight — subtle
      ctx.fillStyle = 'rgba(100,255,100,0.03)';
      ctx.fillRect(obs.x, obs.gapTop, OBSTACLE_WIDTH, obs.gapBottom - obs.gapTop);
    }

    // creatures draw karo — triangles, color by species
    for (const individual of population) {
      if (!individual.creature) continue;
      const c = individual.creature;
      if (c.x > arenaW + 10) continue;

      const color = getSpeciesColor(individual.speciesId);
      const alpha = c.alive ? 0.7 : 0.1;

      ctx.save();
      ctx.translate(c.x, c.y);

      // triangle pointing right — direction of movement
      const size = CREATURE_RADIUS;
      ctx.beginPath();
      ctx.moveTo(size * 1.2, 0);
      ctx.lineTo(-size * 0.8, -size * 0.7);
      ctx.lineTo(-size * 0.8, size * 0.7);
      ctx.closePath();

      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      ctx.fill();

      // alive creatures ko subtle glow do
      if (c.alive) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 4;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      ctx.globalAlpha = 1;
      ctx.restore();
    }

    ctx.restore(); // clip restore
  }

  // champion ka neural network topology draw karo — right 40%
  function drawNetwork() {
    const netX = canvasW * 0.6 + 10;
    const netW = canvasW * 0.4 - 20;
    const netH = canvasH * 0.55;
    const netY = 10;

    // "Champion Network" label
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.5)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Champion Network', netX + netW / 2, netY + 10);

    if (!championGenome && population.length > 0) {
      // pehli generation mein best current individual dhundho
      let best = population[0];
      for (const ind of population) {
        if (ind.creature && ind.creature.alive && ind.creature.distance > (best.creature ? best.creature.distance : 0)) {
          best = ind;
        }
      }
      championGenome = cloneGenome(best.genome);
    }

    if (!championGenome) return;

    const genome = championGenome;

    // nodes ko layers mein organize karo — topological positioning
    const nodePositions = {};
    const layerAssignment = {};

    // input nodes — left side
    const inputNodes = genome.nodes.filter(n => n.type === NODE_INPUT);
    const outputNodes = genome.nodes.filter(n => n.type === NODE_OUTPUT);
    const hiddenNodes = genome.nodes.filter(n => n.type === NODE_HIDDEN);

    // simple layering: input=0, hidden=1 (ya zyada layers), output=last
    for (const n of inputNodes) layerAssignment[n.id] = 0;
    for (const n of outputNodes) layerAssignment[n.id] = 2;

    if (hiddenNodes.length > 0) {
      // hidden nodes ke liye — connections ke basis pe layer decide karo
      for (const n of hiddenNodes) {
        layerAssignment[n.id] = 1; // default
      }

      // multiple hidden layers detect karo — agar hidden→hidden connection hai
      let changed = true;
      let iterations = 0;
      while (changed && iterations < 10) {
        changed = false;
        iterations++;
        for (const conn of genome.connections) {
          if (!conn.enabled) continue;
          const fromLayer = layerAssignment[conn.from];
          const toLayer = layerAssignment[conn.to];
          if (fromLayer !== undefined && toLayer !== undefined) {
            if (fromLayer >= toLayer && conn.to !== conn.from) {
              const toNode = genome.nodes.find(n => n.id === conn.to);
              const fromNode = genome.nodes.find(n => n.id === conn.from);
              if (fromNode && toNode && fromNode.type !== NODE_INPUT && toNode.type === NODE_HIDDEN) {
                layerAssignment[conn.to] = fromLayer + 1;
                changed = true;
              }
            }
          }
        }
      }

      // output nodes ko sabse last layer pe rakh
      let maxHiddenLayer = 1;
      for (const n of hiddenNodes) {
        if (layerAssignment[n.id] > maxHiddenLayer) {
          maxHiddenLayer = layerAssignment[n.id];
        }
      }
      for (const n of outputNodes) {
        layerAssignment[n.id] = maxHiddenLayer + 1;
      }
    }

    // layers collect karo
    const layerMap = {};
    for (const n of genome.nodes) {
      const layer = layerAssignment[n.id] || 0;
      if (!layerMap[layer]) layerMap[layer] = [];
      layerMap[layer].push(n);
    }

    const layerKeys = Object.keys(layerMap).map(Number).sort((a, b) => a - b);
    const numLayers = layerKeys.length;

    // positions calculate karo
    const drawArea = {
      x: netX + 25,
      y: netY + 22,
      w: netW - 50,
      h: netH - 35,
    };

    for (let li = 0; li < numLayers; li++) {
      const layerIdx = layerKeys[li];
      const nodesInLayer = layerMap[layerIdx];
      const x = drawArea.x + (li / Math.max(numLayers - 1, 1)) * drawArea.w;

      for (let ni = 0; ni < nodesInLayer.length; ni++) {
        const y = drawArea.y + ((ni + 0.5) / nodesInLayer.length) * drawArea.h;
        nodePositions[nodesInLayer[ni].id] = { x, y };
      }
    }

    // connections draw karo — pehle lines, fir nodes upar
    for (const conn of genome.connections) {
      const fromPos = nodePositions[conn.from];
      const toPos = nodePositions[conn.to];
      if (!fromPos || !toPos) continue;

      if (!conn.enabled) {
        // disabled connection — dashed dim line
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = 'rgba(100,100,100,0.15)';
        ctx.lineWidth = 0.5;
      } else {
        // enabled connection — color by weight sign, thickness by magnitude
        ctx.setLineDash([]);
        const absWeight = Math.abs(conn.weight);
        const thickness = Math.max(0.5, Math.min(3, absWeight * 0.8));
        const alpha = Math.max(0.15, Math.min(0.8, absWeight * 0.3));

        if (conn.weight > 0) {
          ctx.strokeStyle = 'rgba(100,255,100,' + alpha + ')';
        } else {
          ctx.strokeStyle = 'rgba(255,100,100,' + alpha + ')';
        }
        ctx.lineWidth = thickness;
      }

      ctx.beginPath();
      ctx.moveTo(fromPos.x, fromPos.y);
      ctx.lineTo(toPos.x, toPos.y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // nodes draw karo
    const nodeRadius = 6;
    const inputLabels = ['y', 'dist', 'gT', 'gB', 'vy'];
    const outputLabels = ['up', 'dn'];

    for (const node of genome.nodes) {
      const pos = nodePositions[node.id];
      if (!pos) continue;

      // node circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);

      if (node.type === NODE_INPUT) {
        ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.6)';
        ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.8)';
      } else if (node.type === NODE_OUTPUT) {
        ctx.fillStyle = 'rgba(255,200,50,0.6)';
        ctx.strokeStyle = 'rgba(255,200,50,0.8)';
      } else {
        // hidden node — network evolve ho raha hai!
        ctx.fillStyle = 'rgba(200,100,255,0.5)';
        ctx.strokeStyle = 'rgba(200,100,255,0.7)';
      }

      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();

      // labels — input aur output nodes ke liye
      ctx.fillStyle = 'rgba(200,200,200,0.6)';
      ctx.font = '8px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';

      if (node.type === NODE_INPUT) {
        const idx = INPUT_IDS.indexOf(node.id);
        if (idx >= 0) ctx.fillText(inputLabels[idx], pos.x, pos.y + nodeRadius + 10);
      } else if (node.type === NODE_OUTPUT) {
        const idx = OUTPUT_IDS.indexOf(node.id);
        if (idx >= 0) ctx.fillText(outputLabels[idx], pos.x, pos.y + nodeRadius + 10);
      } else {
        // hidden node ka id dikhao
        ctx.fillText('h' + node.id, pos.x, pos.y + nodeRadius + 10);
      }
    }

    // network stats — connections, nodes count
    const enabledConns = genome.connections.filter(c => c.enabled).length;
    const totalConns = genome.connections.length;
    ctx.fillStyle = 'rgba(150,150,150,0.4)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(
      genome.nodes.length + ' nodes, ' + enabledConns + '/' + totalConns + ' conns',
      netX + netW / 2, netY + netH - 2
    );
  }

  // fitness graph draw karo — right side, bottom portion
  function drawFitnessGraph() {
    const graphX = canvasW * 0.6 + 10;
    const graphW = canvasW * 0.4 - 20;
    const graphY = canvasH * 0.58;
    const graphH = canvasH * 0.38;

    // "Fitness" label
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Fitness (gen vs best/avg)', graphX + graphW / 2, graphY + 8);

    if (fitnessHistory.length < 2) {
      ctx.fillStyle = 'rgba(150,150,150,0.3)';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText('data aayega...', graphX + graphW / 2, graphY + graphH / 2);
      return;
    }

    const padL = 35;
    const padR = 8;
    const padT = 16;
    const padB = 8;
    const plotX = graphX + padL;
    const plotY = graphY + padT;
    const plotW = graphW - padL - padR;
    const plotH = graphH - padT - padB;

    // y-axis range dhundho
    let maxFit = 0;
    for (const h of fitnessHistory) {
      if (h.best > maxFit) maxFit = h.best;
    }
    maxFit = maxFit || 100;
    maxFit *= 1.1;

    // grid lines
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.06)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 3; i++) {
      const y = plotY + (i / 3) * plotH;
      ctx.beginPath();
      ctx.moveTo(plotX, y);
      ctx.lineTo(plotX + plotW, y);
      ctx.stroke();
    }

    // y-axis labels
    ctx.fillStyle = 'rgba(150,150,150,0.4)';
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 3; i++) {
      const val = maxFit * (1 - i / 3);
      const y = plotY + (i / 3) * plotH;
      ctx.fillText(val.toFixed(0), plotX - 4, y + 3);
    }

    // avg fitness line — dimmer
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(150,150,150,0.3)';
    ctx.lineWidth = 1;
    const n = fitnessHistory.length;
    for (let i = 0; i < n; i++) {
      const x = plotX + (i / Math.max(n - 1, 1)) * plotW;
      const y = plotY + plotH * (1 - fitnessHistory[i].avg / maxFit);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // best fitness line — bright
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(' + ACCENT_RGB + ',0.8)';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < n; i++) {
      const x = plotX + (i / Math.max(n - 1, 1)) * plotW;
      const y = plotY + plotH * (1 - fitnessHistory[i].best / maxFit);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // fill under best line
    const grad = ctx.createLinearGradient(0, plotY, 0, plotY + plotH);
    grad.addColorStop(0, 'rgba(' + ACCENT_RGB + ',0.1)');
    grad.addColorStop(1, 'rgba(' + ACCENT_RGB + ',0.0)');
    ctx.lineTo(plotX + plotW, plotY + plotH);
    ctx.lineTo(plotX, plotY + plotH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // legend
    ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.6)';
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('-- best', plotX + 2, plotY + plotH + 7);
    ctx.fillStyle = 'rgba(150,150,150,0.5)';
    ctx.fillText('-- avg', plotX + 48, plotY + plotH + 7);
  }

  // stats update karo — safe DOM methods, no innerHTML
  function updateStats() {
    // genStat update
    genStat.lastChild.textContent = String(generation);
    // aliveStat update
    aliveStat.lastChild.textContent = aliveCount + '/' + populationSize;
    // speciesStat update
    speciesStat.lastChild.textContent = String(species.length);
    // bestStat update
    bestStat.lastChild.textContent = String(Math.round(genBestFitness));
    // allTimeStat update
    allTimeStat.lastChild.textContent = String(Math.round(allTimeBestFitness));
    // genDisplay update
    genDisplay.textContent = 'Gen ' + generation;
  }

  // full frame draw karo
  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    drawArena();
    drawNetwork();
    drawFitnessGraph();
  }

  // ===================== Animation Loop =====================

  function loop() {
    // lab pause check
    if (window.__labPaused && window.__labPaused !== container.id) {
      animationId = null;
      return;
    }

    if (!isVisible) {
      animationId = null;
      return;
    }

    // speed multiplier ke hisaab se multiple steps per frame
    for (let i = 0; i < speedMultiplier; i++) {
      if (generationRunning) {
        gameStep();
      }

      // agar generation khatam ho gayi toh nayi shuru karo
      if (!generationRunning) {
        createNextGeneration();
        startGeneration();
        updateStats();
      }
    }

    draw();

    // stats har kuch frames mein update karo — performance ke liye
    if (frameCount % 5 === 0) {
      // alive count refresh karo stats ke liye
      aliveCount = 0;
      for (const individual of population) {
        if (individual.creature && individual.creature.alive) aliveCount++;
      }
      updateStats();
    }

    animationId = requestAnimationFrame(loop);
  }

  // ===================== Visibility Management =====================

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const wasVisible = isVisible;
        isVisible = entry.isIntersecting;
        if (isVisible && !wasVisible) {
          if (!animationId) animationId = requestAnimationFrame(loop);
        } else if (!isVisible && wasVisible) {
          if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
          }
        }
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);

  // lab resume listener — jab lab pause se resume ho
  document.addEventListener('lab:resume', () => {
    if (isVisible && !animationId) loop();
  });

  // tab visibility — tab switch pe pause karo, wapas aane pe resume
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) {
        isVisible = true;
        if (!animationId) animationId = requestAnimationFrame(loop);
      }
    }
  });

  // ===================== Initialization =====================

  // sab setup karke evolution shuru karo — generation 1 automatic
  initPopulation();
  updateStats();
  draw();

  // pehla frame ke liye — agar visible hai toh loop shuru karo
  if (!animationId) animationId = requestAnimationFrame(loop);
}
