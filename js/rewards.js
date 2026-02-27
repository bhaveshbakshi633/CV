// RL reward signals - portfolio pe har action pe floating reward dikhata hai
// kyunki life mein bhi toh reward signals milte hain, bas dikhte nahi

const POS_COLOR = '#10b981';
const NEG_COLOR = '#ef4444';
const MAX_FLOATS = 5;

let totalReward = 0;
let counterEl = null;
let activeFloats = 0;
let hoverTimestamps = new WeakMap();
let idleTimer = null;

function injectStyles() {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes rl-float { from { opacity:1; transform:translateY(0); } to { opacity:0; transform:translateY(-40px); } }
    .rl-float { position:fixed; font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600;
      pointer-events:none; z-index:9999; animation:rl-float 1.2s ease-out forwards; }
    .rl-episode-counter { position:fixed; bottom:20px; left:20px; font-family:'JetBrains Mono',monospace;
      font-size:11px; color:rgba(16,185,129,0.3); z-index:50; display:flex; gap:8px;
      transition:color 0.3s; pointer-events:auto; cursor:default; }
    .rl-episode-counter:hover { color:rgba(16,185,129,0.6); }
    .rl-counter-value { font-weight:600; }
  `;
  document.head.appendChild(style);
}

function createCounter() {
  counterEl = document.createElement('div');
  counterEl.className = 'rl-episode-counter';
  counterEl.innerHTML = `<span class="rl-counter-label">episode_reward</span><span class="rl-counter-value">0.0</span>`;
  document.body.appendChild(counterEl);
}

function spawnReward(x, y, value) {
  if (activeFloats >= MAX_FLOATS) return;
  totalReward += value;
  counterEl.querySelector('.rl-counter-value').textContent = totalReward.toFixed(1);

  const span = document.createElement('span');
  span.className = 'rl-float';
  span.textContent = (value >= 0 ? '+' : '') + value.toFixed(1);
  span.style.color = value >= 0 ? POS_COLOR : NEG_COLOR;
  span.style.left = x + 'px';
  span.style.top = y + 'px';
  document.body.appendChild(span);
  activeFloats++;

  span.addEventListener('animationend', () => { span.remove(); activeFloats--; });
}

// idle timer - 5 sec kuch nahi kiya toh penalty milegi
function resetIdle() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    spawnReward(window.innerWidth / 2, window.innerHeight / 2, -0.1);
    resetIdle();
  }, 5000);
}

export function initRewards() {
  // mobile pe reward signals band — regular visitors confuse hote hain
  // "+0.3" floating dekh ke lagta hai page mein bug hai
  if (window.innerWidth < 768) return;

  injectStyles();
  createCounter();
  resetIdle();

  // project card click: +1.0
  document.addEventListener('click', (e) => {
    resetIdle();
    const card = e.target.closest('.bento-card, .capability-project-card');
    if (card) { spawnReward(e.clientX, e.clientY, 1.0); return; }

    // contact link click: +2.0
    const link = e.target.closest('a[href*="mailto"], a[href*="github"], a[href*="linkedin"], .contact a, footer a');
    if (link) spawnReward(e.clientX, e.clientY, 2.0);
  });

  // card hover: +0.1 (debounced - max 1 per second per element)
  document.addEventListener('mouseover', (e) => {
    resetIdle();
    const card = e.target.closest('.bento-card, .capability-project-card');
    if (!card) return;
    const now = Date.now();
    const last = hoverTimestamps.get(card) || 0;
    if (now - last < 1000) return;
    hoverTimestamps.set(card, now);
    const rect = card.getBoundingClientRect();
    spawnReward(rect.left + rect.width / 2, rect.top, 0.1);
  });

  // section scroll into view: +0.3 (sirf ek baar per section)
  const seen = new WeakSet();
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting && !seen.has(entry.target)) {
        seen.add(entry.target);
        const rect = entry.target.getBoundingClientRect();
        spawnReward(rect.left + rect.width / 2, rect.top + 20, 0.3);
      }
    }
  }, { threshold: 0.3 });

  document.querySelectorAll('section').forEach((s) => observer.observe(s));
}
