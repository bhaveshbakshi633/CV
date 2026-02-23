// app.js — entry point
// loads project data, initializes all modules

import { projectData } from './data/projects.js';
import { initProtection } from './protection.js';
import { renderBentoGrid, renderCapabilitySections, initRenderer } from './renderer.js';
import { initNavigation } from './navigation.js';
import { initScrollAnimations, initTabVisibility } from './animations.js';
import { initLightbox } from './lightbox.js';
import { initTiltCards, initTypingEffect, initRobotArm } from './interactions.js';
import { initBackground } from './background.js';
import { initRewards } from './rewards.js';
import { initEasterEggs } from './easter-eggs.js';
import { initGridworld } from './games/gridworld.js';
import { initPIDTuner } from './games/pid-tuner.js';
import { initCartPole } from './games/cartpole.js';
import { initBandit } from './games/bandit.js';
import { initMaze } from './games/maze.js';
import { initRobotNav } from './games/robot-nav.js';
import { initKalman } from './games/kalman.js';
import { initIKPlayground } from './games/ik-playground.js';
import { initProjectile } from './games/projectile.js';
import { initNBody } from './games/nbody.js';
import { initFFT } from './games/fft.js';
import { initCircuit } from './games/circuit.js';
import { initSensorFusion } from './games/sensor-fusion.js';
import { initCursorTrail, initSectionBeams, initBootSequence, initAchievements } from './extras.js';

function getAllProjects(data) {
  return [...data.flagship, ...data.supporting, ...data.archived];
}

function init() {
  const allProjects = getAllProjects(projectData);

  // terminal boot sequence — plays once per session, before everything else
  initBootSequence();

  // atmospheric background — matrix rain (before everything else)
  initBackground();

  // re-render callback — protection module calls this after unlock/settings change
  const reRender = () => {
    renderBentoGrid(projectData);
    renderCapabilitySections(projectData);
  };

  // init modules (order matters: protection before renderer)
  initProtection(allProjects, reRender);
  initRenderer(projectData);
  initLightbox();
  initNavigation();

  // render dynamic content
  renderBentoGrid(projectData);
  renderCapabilitySections(projectData);

  // animations — after content is in the DOM
  initScrollAnimations();
  initTabVisibility();

  // interactive gimmicks
  initTiltCards();
  initTypingEffect();
  initRobotArm();

  // playable games — each finds its own container
  // real-time control
  initPIDTuner();
  initKalman();
  initSensorFusion();
  // learning systems
  initGridworld();
  initCartPole();
  initBandit();
  initMaze();
  // human-robot interaction
  initRobotNav();
  initIKPlayground();
  // hardware & signal processing
  initCircuit();
  initFFT();
  // standalone physics demos
  initProjectile();
  initNBody();

  // easter eggs — konami, sudo, photo click
  initEasterEggs();

  // extras — cursor trail, section beams, achievements
  initCursorTrail();
  initSectionBeams();
  initAchievements();

  // RL reward signals — gamified browsing
  initRewards();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
