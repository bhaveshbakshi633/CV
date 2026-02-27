// extras.js — portfolio ke extra masaledaar features
// cursor trail, section beams, boot sequence, achievements, scroll progress
// sab independent hain, koi bhi init function alag se call kar sakte ho


// ============================================================
// 0. SCROLL PROGRESS BAR — page ke top pe scroll indicator
// ============================================================

export function initScrollProgress() {
  // progress bar element banao
  const bar = document.createElement('div');
  bar.className = 'scroll-progress';
  document.body.appendChild(bar);

  // scroll pe width update karo — 0% se 100% tak
  function updateProgress() {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    // agar page scroll nahi hota (chota content) toh 100% dikha do
    const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 100;
    bar.style.width = progress + '%';
  }

  // passive listener — performance ke liye
  window.addEventListener('scroll', updateProgress, { passive: true });
  // initial state set karo
  updateProgress();
}

// ============================================================
// 1. MAGNETIC CURSOR TRAIL — glowing particles jo cursor follow kare
// ============================================================

export function initCursorTrail() {
  // reduced motion waale ko baksh do — accessibility respect karo
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // canvas banao — poore page pe overlay, click-through
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:9990;pointer-events:none;';
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // canvas size sync karo window ke saath
  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // particle pool — max 50, usse zyada nahi warna performance jhand ho jaayegi
  const MAX_PARTICLES = 50;
  const particles = [];

  // section ke hisaab se color decide karo
  // data-capability attribute check karke upar walk karo DOM mein
  const CAPABILITY_COLORS = {
    'real-time-control': { r: 245, g: 158, b: 11 },   // amber
    'learning-systems':  { r: 74,  g: 158, b: 255 },  // blue
    'human-robot-interaction': { r: 16, g: 185, b: 129 }, // emerald
    'hardware-build':    { r: 249, g: 115, b: 22 }    // orange
  };
  const DEFAULT_COLOR = { r: 74, g: 158, b: 255 }; // blue default

  // cursor position track karo
  let mouseX = -100, mouseY = -100;

  // ye function cursor ke neeche ka section dhundhta hai
  function getColorAtPoint(x, y) {
    const el = document.elementFromPoint(x, y);
    if (!el) return DEFAULT_COLOR;

    // element se upar walk karo jab tak data-capability na mile
    let node = el;
    while (node && node !== document.body) {
      const cap = node.getAttribute?.('data-capability');
      if (cap && CAPABILITY_COLORS[cap]) {
        return CAPABILITY_COLORS[cap];
      }
      node = node.parentElement;
    }
    return DEFAULT_COLOR;
  }

  // mousemove pe 1-2 particles spawn karo
  document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;

    // current section ka color utha lo
    const color = getColorAtPoint(mouseX, mouseY);

    // 1-2 particles — random decide karo
    const count = 1 + Math.round(Math.random() * 0.6);
    for (let i = 0; i < count; i++) {
      if (particles.length >= MAX_PARTICLES) break;

      particles.push({
        x: mouseX,
        y: mouseY,
        // random drift direction — thoda sa ghoomega fade hote hue
        dx: (Math.random() - 0.5) * 1.5,
        dy: (Math.random() - 0.5) * 1.5,
        // random size 2-4px
        radius: 2 + Math.random() * 2,
        // opacity 0.6 se start, 800ms mein 0 tak jaayega
        opacity: 0.6,
        // color RGB values copy karo
        r: color.r,
        g: color.g,
        b: color.b,
        // lifetime tracking — 800ms total
        born: performance.now(),
        lifetime: 800
      });
    }
  });

  // animation loop — har frame pe particles update + draw karo
  let lastFrame = performance.now();

  function animate(now) {
    requestAnimationFrame(animate);

    // delta time nikalo — smooth animation ke liye
    const dt = now - lastFrame;
    lastFrame = now;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ulta loop chala — remove karte waqt index shift na ho
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      const age = now - p.born;
      const progress = age / p.lifetime; // 0 se 1 tak

      // time khatam? hata do
      if (progress >= 1) {
        particles.splice(i, 1);
        continue;
      }

      // opacity fade karo — linear 0.6 se 0
      p.opacity = 0.6 * (1 - progress);

      // position update — drift karo
      p.x += p.dx * (dt / 16);
      p.y += p.dy * (dt / 16);

      // draw circle
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.r},${p.g},${p.b},${p.opacity})`;
      ctx.fill();
    }
  }

  requestAnimationFrame(animate);
}


// ============================================================
// 2. SECTION TRANSITION BEAMS — laser scanner effect jab section aaye
// ============================================================

export function initSectionBeams() {
  // reduced motion? skip
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // section accent colors — beam ka color section ke hisaab se
  const BEAM_COLORS = {
    'real-time-control': '#f59e0b',
    'learning-systems': '#4a9eff',
    'human-robot-interaction': '#10b981',
    'hardware-build': '#f97316'
  };
  // flagships section ka default color
  const DEFAULT_BEAM_COLOR = '#4a9eff';

  // track karo konsa section already fire ho chuka — sirf ek baar per page load
  const firedSections = new Set();

  // beam fire karne ka function
  function fireBeam(sectionEl) {
    const id = sectionEl.id || sectionEl.className;
    if (firedSections.has(id)) return; // already fire ho chuka, skip
    firedSections.add(id);

    // section ka color dhundho
    const capability = sectionEl.getAttribute('data-capability');
    const color = BEAM_COLORS[capability] || DEFAULT_BEAM_COLOR;

    // section ko position: relative banao taaki beam uske andar rahe
    const pos = getComputedStyle(sectionEl).position;
    if (pos === 'static') sectionEl.style.position = 'relative';

    // beam element banao — section ke andar absolute position
    // ab scroll karte waqt section ke saath move karega, overlay nahi hoga
    const beam = document.createElement('div');
    beam.style.cssText = [
      'position: absolute',
      'top: 0',
      'left: 0',
      'width: 100%',
      'height: 2px',
      'background: linear-gradient(90deg, ' + color + ', ' + color + '88, transparent)',
      'z-index: 2',
      'pointer-events: none',
      'transform: scaleX(0)',
      'transform-origin: left center',
      'opacity: 1'
    ].join(';');
    sectionEl.appendChild(beam);

    // sweep animation — left se right 0.3s mein
    requestAnimationFrame(() => {
      beam.style.transition = 'transform 0.3s cubic-bezier(0.16, 1, 0.3, 1)';
      beam.style.transform = 'scaleX(1)';
    });

    // sweep complete hone ke baad fade out karo
    setTimeout(() => {
      beam.style.transition = 'opacity 0.25s ease-out';
      beam.style.opacity = '0';
    }, 300);

    // cleanup — DOM se hata do
    setTimeout(() => {
      beam.remove();
    }, 600);
  }

  // IntersectionObserver lagao capability + flagships sections pe
  const sections = document.querySelectorAll('.capability-section, .flagships-section');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        fireBeam(entry.target);
      }
    });
  }, {
    threshold: 0.1
  });

  sections.forEach(section => observer.observe(section));
}


// ============================================================
// 3. TERMINAL BOOT SEQUENCE — fake system startup on first visit
// ============================================================

export function initBootSequence() {
  // FOUC prevention — boot-pending class hatao taaki content dikhe
  // agar boot already dikha chuka hai toh seedha reveal karo
  if (sessionStorage.getItem('bootShown')) {
    document.body.classList.remove('boot-pending');
    return;
  }
  sessionStorage.setItem('bootShown', 'true');

  // boot lines — terminal output jaisi
  const BOOT_LINES = [
    { time: '0.000', text: 'Initializing system...',          dots: false },
    { time: '0.120', text: 'Loading neural networks',          dots: true, suffix: 'OK' },
    { time: '0.340', text: 'Calibrating sensors',              dots: true, suffix: 'OK' },
    { time: '0.510', text: 'Starting control loops',           dots: true, suffix: 'OK' },
    { time: '0.680', text: 'Connecting to hardware',           dots: true, suffix: 'OK' },
    { time: '0.850', text: 'Running safety checks',            dots: true, suffix: 'OK' },
    { time: '1.020', text: 'All systems nominal.',             dots: false },
    { time: '1.200', text: 'Welcome, visitor.',                dots: false }
  ];

  // overlay banao — fullscreen terminal
  const overlay = document.createElement('div');
  overlay.style.cssText = [
    'position: fixed', 'top: 0', 'left: 0', 'width: 100%', 'height: 100%',
    'background: #0a0a0a', 'z-index: 10000',
    'display: flex', 'flex-direction: column', 'justify-content: center',
    'padding: 20% 10%', 'box-sizing: border-box',
    "font-family: 'JetBrains Mono', monospace", 'font-size: 14px',
    'color: #10b981', 'overflow: hidden'
  ].join(';');
  document.body.appendChild(overlay);

  // boot overlay aa gaya — ab CSS pseudo-element wala cover hatao
  // body content abhi bhi boot overlay ke peeche hidden rahega
  document.body.classList.remove('boot-pending');

  // body scroll band karo jab tak boot chal raha hai
  const originalOverflow = document.body.style.overflow;
  document.body.style.overflow = 'hidden';

  // ek ek line render karo with delay
  let lineIndex = 0;
  const LINE_DELAY = 180; // ms between lines (150-200ms range mein)

  function renderNextLine() {
    if (lineIndex >= BOOT_LINES.length) {
      // sab lines ho gayi — 400ms ruko, fir fade out karo
      setTimeout(() => {
        overlay.style.transition = 'opacity 0.5s ease-out';
        overlay.style.opacity = '0';
        setTimeout(() => {
          overlay.remove();
          document.body.style.overflow = originalOverflow;
        }, 500);
      }, 400);
      return;
    }

    const line = BOOT_LINES[lineIndex];
    const lineEl = document.createElement('div');
    lineEl.style.cssText = 'margin-bottom: 4px; white-space: pre; opacity: 0;';
    overlay.appendChild(lineEl);

    // fade in line
    requestAnimationFrame(() => {
      lineEl.style.transition = 'opacity 0.1s';
      lineEl.style.opacity = '1';
    });

    if (line.dots) {
      // dots waali line — pehle text + dots animate karo, fir OK lagao
      const prefix = '[' + line.time.padStart(8) + '] ' + line.text + ' ';
      lineEl.textContent = prefix;

      // dots ek ek karke aayenge — aligned dikhna chahiye
      const totalDots = 11;
      const targetWidth = 40;
      const paddingDots = Math.max(totalDots, targetWidth - prefix.length);

      let dotCount = 0;
      const dotInterval = setInterval(() => {
        dotCount++;
        lineEl.textContent = prefix + '.'.repeat(dotCount);
        if (dotCount >= paddingDots) {
          clearInterval(dotInterval);
          // OK lagao thodi der baad
          setTimeout(() => {
            lineEl.textContent = prefix + '.'.repeat(paddingDots) + ' ' + line.suffix;
          }, 60);
        }
      }, 15); // har 15ms pe ek dot
    } else {
      // normal line — seedha text
      lineEl.textContent = '[' + line.time.padStart(8) + '] ' + line.text;
    }

    lineIndex++;
    setTimeout(renderNextLine, LINE_DELAY);
  }

  // shuru karo — thoda delay do taaki page settle ho jaaye
  setTimeout(renderNextLine, 200);
}


// ============================================================
// 4. ACHIEVEMENT TOASTS — gamified browsing rewards
// ============================================================

export function initAchievements() {
  // achievement definitions — kya karna hai unlock karne ke liye
  const ACHIEVEMENTS = {
    explorer: {
      name: 'Explorer',
      icon: '\u{1F9ED}', // compass
      desc: 'Scrolled past all 4 capability sections'
    },
    deepDiver: {
      name: 'Deep Diver',
      icon: '\u{1F4A1}', // lightbulb
      desc: 'Opened a project detail view'
    },
    speedRunner: {
      name: 'Speed Runner',
      icon: '\u26A1',    // lightning
      desc: 'Reached contact in under 30 seconds'
    },
    treasureHunter: {
      name: 'Treasure Hunter',
      icon: '\u{1F3C6}', // trophy
      desc: 'Found an easter egg'
    },
    theCurious: {
      name: 'The Curious',
      icon: '\u{1F50D}', // magnifying glass
      desc: 'Hovered over 3+ bento cards'
    },
    fullStack: {
      name: 'Full Stack',
      icon: '\u{1F30D}', // globe
      desc: 'Visited all sections'
    }
  };

  // styles inject karo — achievement toast ke liye
  const style = document.createElement('style');
  style.textContent = [
    '.achievement-toast {',
    '  position: fixed; bottom: 24px; right: 24px; z-index: 9999;',
    '  display: flex; align-items: center; gap: 12px;',
    '  background: rgba(10, 10, 10, 0.95);',
    '  border: 1px solid rgba(16, 185, 129, 0.4);',
    '  border-radius: 10px; padding: 14px 20px;',
    "  font-family: 'JetBrains Mono', monospace;",
    '  box-shadow: 0 0 20px rgba(16, 185, 129, 0.15), 0 4px 12px rgba(0,0,0,0.5);',
    '  transform: translateX(120%);',
    '  transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);',
    '  max-width: 320px;',
    '  pointer-events: none;',
    '}',
    '.achievement-toast.show { transform: translateX(0); }',
    '.achievement-toast.hide {',
    '  transform: translateX(120%);',
    '  transition: transform 0.3s ease-in;',
    '}',
    '.achievement-toast-icon { font-size: 24px; flex-shrink: 0; }',
    '.achievement-toast-text { display: flex; flex-direction: column; gap: 2px; }',
    '.achievement-toast-label {',
    '  font-size: 10px; color: rgba(16, 185, 129, 0.7);',
    '  text-transform: uppercase; letter-spacing: 1px;',
    '}',
    '.achievement-toast-name { font-size: 13px; color: #10b981; font-weight: 600; }'
  ].join('\n');
  document.head.appendChild(style);

  // sessionStorage se already unlocked achievements load karo
  function getUnlocked() {
    try {
      return JSON.parse(sessionStorage.getItem('achievements') || '[]');
    } catch { return []; }
  }
  function saveUnlocked(list) {
    sessionStorage.setItem('achievements', JSON.stringify(list));
  }

  // toast queue — ek time pe ek hi toast dikhao
  const toastQueue = [];
  let toastActive = false;

  function showAchievementToast(achievementKey) {
    const unlocked = getUnlocked();
    if (unlocked.includes(achievementKey)) return; // already unlocked, skip

    // mark as unlocked
    unlocked.push(achievementKey);
    saveUnlocked(unlocked);

    // queue mein daal do
    toastQueue.push(achievementKey);
    processQueue();
  }

  function processQueue() {
    if (toastActive || toastQueue.length === 0) return;
    toastActive = true;

    const key = toastQueue.shift();
    const achievement = ACHIEVEMENTS[key];
    if (!achievement) { toastActive = false; processQueue(); return; }

    // toast element banao — safe DOM construction, no innerHTML
    const toast = document.createElement('div');
    toast.className = 'achievement-toast';

    const iconSpan = document.createElement('span');
    iconSpan.className = 'achievement-toast-icon';
    iconSpan.textContent = achievement.icon;

    const textDiv = document.createElement('div');
    textDiv.className = 'achievement-toast-text';

    const labelSpan = document.createElement('span');
    labelSpan.className = 'achievement-toast-label';
    labelSpan.textContent = 'Achievement Unlocked';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'achievement-toast-name';
    nameSpan.textContent = achievement.name;

    textDiv.appendChild(labelSpan);
    textDiv.appendChild(nameSpan);
    toast.appendChild(iconSpan);
    toast.appendChild(textDiv);
    document.body.appendChild(toast);

    // slide in
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        toast.classList.add('show');
      });
    });

    // 3 seconds baad slide out karo
    setTimeout(() => {
      toast.classList.remove('show');
      toast.classList.add('hide');
      setTimeout(() => {
        toast.remove();
        toastActive = false;
        processQueue(); // next toast process karo
      }, 350);
    }, 3000);
  }

  // --- EXPLORER: sab 4 capability sections scroll kar ke dekhe ---
  const capabilitySeen = new Set();
  const capabilitySections = document.querySelectorAll('.capability-section');

  if (capabilitySections.length > 0) {
    const explorerObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const cap = entry.target.getAttribute('data-capability');
          if (cap) capabilitySeen.add(cap);
          // chaaron dekh liye? unlock!
          if (capabilitySeen.size >= 4) {
            showAchievementToast('explorer');
          }
        }
      });
    }, { threshold: 0.3 });

    capabilitySections.forEach(s => explorerObserver.observe(s));
  }

  // --- DEEP DIVER: project detail open kiya ---
  // project detail view ka observer — jab active class aaye
  const detailView = document.getElementById('projectDetailView');
  if (detailView) {
    const detailObserver = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.attributeName === 'class' && detailView.classList.contains('active')) {
          showAchievementToast('deepDiver');
          break;
        }
      }
    });
    detailObserver.observe(detailView, { attributes: true, attributeFilter: ['class'] });
  }

  // --- SPEED RUNNER: contact section 30 sec ke andar reach kiya ---
  const pageLoadTime = Date.now();
  const contactSection = document.getElementById('contact');
  if (contactSection) {
    const speedObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const elapsed = Date.now() - pageLoadTime;
          if (elapsed <= 30000) { // 30 seconds ke andar!
            showAchievementToast('speedRunner');
          }
          // ek baar check karna kaafi hai — chahe mile ya na mile
          speedObserver.unobserve(contactSection);
        }
      });
    }, { threshold: 0.1 });
    speedObserver.observe(contactSection);
  }

  // --- TREASURE HUNTER: easter egg trigger hua ---
  // easter-eggs.js se custom event sun rahe hain
  document.addEventListener('easter-egg-triggered', () => {
    showAchievementToast('treasureHunter');
  });

  // --- THE CURIOUS: 3+ different bento cards pe hover kiya ---
  const hoveredCards = new Set();
  // event delegation use karo — performance ke liye better
  const bentoGrid = document.getElementById('bentoGrid');
  if (bentoGrid) {
    bentoGrid.addEventListener('mouseover', (e) => {
      const card = e.target.closest('.bento-card');
      if (!card) return;
      const projectId = card.getAttribute('data-project-id');
      if (projectId) hoveredCards.add(projectId);
      if (hoveredCards.size >= 3) {
        showAchievementToast('theCurious');
      }
    });
  }

  // --- FULL STACK: about, experience, contact — teeno visit kiye ---
  const fullStackSections = new Set();
  const targetIds = ['about', 'experience', 'contact'];
  const fullStackTargets = targetIds
    .map(id => document.getElementById(id))
    .filter(Boolean);

  if (fullStackTargets.length > 0) {
    const fullStackObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          fullStackSections.add(entry.target.id);
          if (targetIds.every(id => fullStackSections.has(id))) {
            showAchievementToast('fullStack');
          }
        }
      });
    }, { threshold: 0.2 });

    fullStackTargets.forEach(s => fullStackObserver.observe(s));
  }
}
