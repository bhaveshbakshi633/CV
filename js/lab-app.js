// lab-app.js — entry point for Interactive Learning Lab
// imports all 50 simulations, initializes each when visible

// dormant games (6) — already built, just need containers
import { initBandit } from './games/bandit.js';
import { initCircuit } from './games/circuit.js';
import { initFFT } from './games/fft.js';
import { initIKPlayground } from './games/ik-playground.js';
import { initKalman } from './games/kalman.js';
import { initSensorFusion } from './games/sensor-fusion.js';

// batch 1 simulations (14)
import { initDoublePendulum } from './games/double-pendulum.js';
import { initNeuralPlayground } from './games/neural-playground.js';
import { initLorenz } from './games/lorenz.js';
import { initGenetic } from './games/genetic.js';
import { initFourierDraw } from './games/fourier-draw.js';
import { initKMeans } from './games/kmeans.js';
import { initCollisions } from './games/collisions.js';
import { initMandelbrot } from './games/mandelbrot.js';
import { initGameOfLife } from './games/game-of-life.js';
import { initSpringMass } from './games/spring-mass.js';
import { initWaveSim } from './games/wave-sim.js';
import { initPerceptron } from './games/perceptron.js';
import { initBackprop } from './games/backprop.js';
import { initBezier } from './games/bezier.js';

// batch 2 simulations (8)
import { initParticleLife } from './games/particle-life.js';
import { initBoids } from './games/boids.js';
import { initCloth } from './games/cloth.js';
import { initLSystem } from './games/lsystem.js';
import { initReactionDiffusion } from './games/reaction-diffusion.js';
import { initVerletChain } from './games/verlet-chain.js';
import { initSorting } from './games/sorting.js';
import { initTerrain } from './games/terrain.js';

// batch 3 simulations (10) — researched picks
import { initStrangeAttractors } from './games/strange-attractors.js';
import { initVoronoi } from './games/voronoi.js';
import { initForceGraph } from './games/force-graph.js';
import { initEpidemic } from './games/epidemic.js';
import { initDLA } from './games/dla.js';
import { initFluidSim } from './games/fluid-sim.js';
import { initFallingSand } from './games/falling-sand.js';
import { initPathfinding } from './games/pathfinding.js';
import { initGradientDescent } from './games/gradient-descent.js';
import { initWFC } from './games/wfc.js';

// batch 4 simulations (12) — more variety
import { initElectricField } from './games/electric-field.js';
import { initGravity } from './games/gravity.js';
import { initBrownian } from './games/brownian.js';
import { initTuringMachine } from './games/turing-machine.js';
import { initLangtonsAnt } from './games/langtons-ant.js';
import { initCellularAutomata } from './games/cellular-automata.js';
import { initMetaballs } from './games/metaballs.js';
import { initMarchingSquares } from './games/marching-squares.js';
import { initPhyllotaxis } from './games/phyllotaxis.js';
import { initStringArt } from './games/string-art.js';
import { initPixelSort } from './games/pixel-sort.js';
import { initRaycaster } from './games/raycaster.js';

// batch 5 simulations (10) — physics + AI/RL focused
import { initSoftBody } from './games/soft-body.js';
import { initRigidBody } from './games/rigid-body.js';
import { initOptics } from './games/optics.js';
import { initMagneticField } from './games/magnetic-field.js';
import { initFlappyRL } from './games/flappy-rl.js';
import { initNeat } from './games/neat.js';
import { initAntColony } from './games/ant-colony.js';
import { initMinimax } from './games/minimax.js';
import { initSnakeAI } from './games/snake-ai.js';
import { initConvolution } from './games/convolution.js';


// batch 6 — physics (20 new sims)
import { initRippleTank } from './games/ripple-tank.js';
import { initMagneticPendulum } from './games/magnetic-pendulum.js';
import { initQuantumTunneling } from './games/quantum-tunneling.js';
import { initChladni } from './games/chladni.js';
import { initGaltonBoard } from './games/galton-board.js';
import { initHeatEquation } from './games/heat-equation.js';
import { initIsingModel } from './games/ising-model.js';
import { initBrachistochrone } from './games/brachistochrone.js';
import { initPendulumWave } from './games/pendulum-wave.js';
import { initDoppler } from './games/doppler.js';
import { initPercolation } from './games/percolation.js';
import { initLorentzTransform } from './games/lorentz-transform.js';
import { initJoukowski } from './games/joukowski.js';
import { initSandpile } from './games/sandpile.js';
import { initChaosGame } from './games/chaos-game.js';
import { initHydraulicErosion } from './games/hydraulic-erosion.js';
import { initWindTunnel } from './games/wind-tunnel.js';
import { initHarmonograph } from './games/harmonograph.js';
import { initOrbitalSlingshot } from './games/orbital-slingshot.js';

// batch 6 — AI/ML/RL (23 new sims)
import { initGanLab } from './games/gan-lab.js';
import { initMCMC } from './games/mcmc.js';
import { initQLearning } from './games/q-learning.js';
import { initSOM } from './games/som.js';
import { initHopfield } from './games/hopfield.js';
import { initTSNE } from './games/tsne.js';
import { initDiffusionModel } from './games/diffusion-model.js';
import { initAttentionViz } from './games/attention-viz.js';
import { initEvolutionCreatures } from './games/evolution-creatures.js';
import { initNeuralCA } from './games/neural-ca.js';
import { initPSO } from './games/pso.js';
import { initSimulatedAnnealing } from './games/simulated-annealing.js';
import { initMCTS } from './games/mcts.js';
import { initBayesianInference } from './games/bayesian-inference.js';
import { initRRT } from './games/rrt.js';
import { initGMM } from './games/gmm.js';
import { initDBSCAN } from './games/dbscan.js';
import { initDecisionTree } from './games/decision-tree.js';
import { initWord2Vec } from './games/word2vec.js';
import { initBayesianOpt } from './games/bayesian-opt.js';
import { initPotentialField } from './games/potential-field.js';
import { initRLWalking } from './games/rl-walking.js';
import { initImitationLearning } from './games/imitation-learning.js';

function init() {
  // saare sims ko init karo — har ek apna container dhundh lega
  // IntersectionObserver har sim ke andar hai already

  // dormant games activate
  initBandit();
  initCircuit();
  initFFT();
  initIKPlayground();
  initKalman();
  initSensorFusion();

  // batch 1
  initDoublePendulum();
  initNeuralPlayground();
  initLorenz();
  initGenetic();
  initFourierDraw();
  initKMeans();
  initCollisions();
  initMandelbrot();
  initGameOfLife();
  initSpringMass();
  initWaveSim();
  initPerceptron();
  initBackprop();
  initBezier();

  // batch 2
  initParticleLife();
  initBoids();
  initCloth();
  initLSystem();
  initReactionDiffusion();
  initVerletChain();
  initSorting();
  initTerrain();

  // batch 3
  initStrangeAttractors();
  initVoronoi();
  initForceGraph();
  initEpidemic();
  initDLA();
  initFluidSim();
  initFallingSand();
  initPathfinding();
  initGradientDescent();
  initWFC();

  // batch 4
  initElectricField();
  initGravity();
  initBrownian();
  initTuringMachine();
  initLangtonsAnt();
  initCellularAutomata();
  initMetaballs();
  initMarchingSquares();
  initPhyllotaxis();
  initStringArt();
  initPixelSort();
  initRaycaster();

  // batch 5
  initSoftBody();
  initRigidBody();
  initOptics();
  initMagneticField();
  initFlappyRL();
  initNeat();
  initAntColony();
  initMinimax();
  initSnakeAI();
  initConvolution();

  // batch 6 — physics
  initRippleTank();
  initMagneticPendulum();
  initQuantumTunneling();
  initChladni();
  initGaltonBoard();
  initHeatEquation();
  initIsingModel();
  initBrachistochrone();
  initPendulumWave();
  initDoppler();
  initPercolation();
  initLorentzTransform();
  initJoukowski();
  initSandpile();
  initChaosGame();
  initHydraulicErosion();
  initWindTunnel();
  initHarmonograph();
  initOrbitalSlingshot();

  // batch 6 — AI/ML/RL
  initGanLab();
  initMCMC();
  initQLearning();
  initSOM();
  initHopfield();
  initTSNE();
  initDiffusionModel();
  initAttentionViz();
  initEvolutionCreatures();
  initNeuralCA();
  initPSO();
  initSimulatedAnnealing();
  initMCTS();
  initBayesianInference();
  initRRT();
  initGMM();
  initDBSCAN();
  initDecisionTree();
  initWord2Vec();
  initBayesianOpt();
  initPotentialField();
  initRLWalking();
  initImitationLearning();
}

// Performance manager — ek sim interact karo toh baaki pause
// jab scroll karo toh sab resume ho jaaye
function setupSimFocusManager() {
  const content = document.querySelector('.lab-content');
  if (!content) return;

  // jab koi sim pe click/touch ho, usko active mark karo
  content.addEventListener('pointerdown', (e) => {
    const sim = e.target.closest('.lab-sim');
    if (!sim) return;
    window.__labPaused = sim.id;
  });

  // scroll karne pe sab resume — user aage badh gaya
  let scrollTimer;
  window.addEventListener('scroll', () => {
    if (!window.__labPaused) return;
    clearTimeout(scrollTimer);
    scrollTimer = setTimeout(() => {
      window.__labPaused = null;
      document.dispatchEvent(new Event('lab:resume'));
    }, 250);
  });

  // tab switch pe sab pause
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      window.__labPaused = '__all__';
    } else {
      window.__labPaused = null;
      document.dispatchEvent(new Event('lab:resume'));
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { init(); setupSimFocusManager(); });
} else {
  init();
  setupSimFocusManager();
}
