// easter eggs — portfolio mein thoda mazaa aaye toh kya bura hai
const EMOJIS = ['🤖', '🦾', '⚡', '🔧'];
const KONAMI = ['ArrowUp','ArrowUp','ArrowDown','ArrowDown','ArrowLeft','ArrowRight','ArrowLeft','ArrowRight','b','a'];

export function initEasterEggs() {
  // inject styles ek baar — saari animations ke liye
  const style = document.createElement('style');
  style.textContent = `
    .easter-particle { position: fixed; font-size: 20px; pointer-events: none; z-index: 9999; }
    .easter-toast {
      position: fixed; bottom: 60px; left: 50%; transform: translateX(-50%);
      font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #10b981;
      background: rgba(15, 15, 15, 0.95); border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 10px 24px; border-radius: 8px; z-index: 9999;
      animation: easter-toast-in 0.3s ease-out, easter-toast-out 0.5s ease-in 2.5s forwards;
    }
    .easter-combo {
      position: absolute; font-family: 'JetBrains Mono', monospace; font-size: 14px;
      font-weight: 700; color: #10b981; pointer-events: none; z-index: 9999; white-space: nowrap;
    }
    @keyframes easter-toast-in { from { opacity: 0; transform: translateX(-50%) translateY(10px); } to { opacity: 1; transform: translateX(-50%) translateY(0); } }
    @keyframes easter-toast-out { from { opacity: 1; } to { opacity: 0; } }
  `;
  document.head.appendChild(style);

  setupKonami();
  setupPhotoClick();
  setupSudo();
}

// --- toast helper: bana, dikha, hata ---
function showToast(text) {
  const el = document.createElement('div');
  el.className = 'easter-toast';
  el.textContent = text;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

// --- konami code: ↑↑↓↓←→←→BA ---
function setupKonami() {
  let seq = [], triggered = false;
  document.addEventListener('keydown', e => {
    if (triggered || ['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
    seq.push(e.key);
    if (seq.length > KONAMI.length) seq.shift();
    if (seq.length === KONAMI.length && seq.every((k, i) => k === KONAMI[i])) {
      triggered = true;
      spawnParticleBurst();
      showToast('Achievement Unlocked: Master Roboticist');
      document.dispatchEvent(new CustomEvent('easter-egg-triggered'));
    }
  });
}

// particle explosion — center se nikle, gravity se gire, fade ho jaayein
function spawnParticleBurst() {
  const count = 30 + Math.floor(Math.random() * 11);
  const cx = window.innerWidth / 2, cy = window.innerHeight / 2;
  for (let i = 0; i < count; i++) {
    const el = document.createElement('span');
    el.className = 'easter-particle';
    el.textContent = EMOJIS[Math.floor(Math.random() * EMOJIS.length)];
    el.style.left = `${cx}px`;
    el.style.top = `${cy}px`;
    document.body.appendChild(el);

    const angle = Math.random() * Math.PI * 2;
    const speed = 150 + Math.random() * 250;
    const dx = Math.cos(angle) * speed;
    const dy = Math.sin(angle) * speed;

    el.animate([
      { transform: 'translate(0, 0) scale(1)', opacity: 1 },
      { transform: `translate(${dx}px, ${dy + 300}px) scale(0.3)`, opacity: 0 }
    ], { duration: 1200 + Math.random() * 600, easing: 'cubic-bezier(.25,.46,.45,.94)', fill: 'forwards' })
      .onfinish = () => el.remove();
  }
}

// --- profile photo: 5 clicks in 2 seconds ---
function setupPhotoClick() {
  const photo = document.querySelector('.profile-photo') || document.querySelector('.hero img');
  if (!photo) return;

  let clicks = [], active = false;
  photo.addEventListener('click', () => {
    if (active) return;
    const now = Date.now();
    clicks.push(now);
    clicks = clicks.filter(t => now - t < 2000);
    if (clicks.length >= 5) {
      active = true;
      clicks = [];
      glitchPhoto(photo, () => { active = false; });
      document.dispatchEvent(new CustomEvent('easter-egg-triggered'));
    }
  });
}

function glitchPhoto(photo, done) {
  const orig = { filter: photo.style.filter, transform: photo.style.transform };
  photo.style.transition = 'filter 0.15s, transform 0.15s';
  photo.style.filter = 'hue-rotate(180deg) saturate(2)';
  photo.style.transform = (orig.transform || '') + ' skewX(5deg)';

  // floating "+5.0 COMBO" text — RL reward style
  const combo = document.createElement('span');
  combo.className = 'easter-combo';
  combo.textContent = '+5.0 COMBO';
  photo.parentElement.style.position = photo.parentElement.style.position || 'relative';
  photo.parentElement.appendChild(combo);
  combo.style.left = '50%';
  combo.style.top = '0';
  combo.animate([
    { transform: 'translateX(-50%) translateY(0)', opacity: 1 },
    { transform: 'translateX(-50%) translateY(-40px)', opacity: 0 }
  ], { duration: 1000, easing: 'ease-out', fill: 'forwards' })
    .onfinish = () => combo.remove();

  setTimeout(() => {
    photo.style.filter = orig.filter;
    photo.style.transform = orig.transform;
    setTimeout(() => { photo.style.transition = ''; done(); }, 200);
  }, 1000);
}

// --- sudo command: type "sudo" anywhere ---
function setupSudo() {
  let buf = '';
  document.addEventListener('keydown', e => {
    if (['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
    buf += e.key.length === 1 ? e.key : '';
    if (buf.length > 10) buf = buf.slice(-10);
    if (buf.endsWith('sudo')) {
      buf = '';
      showToast('$ sudo access granted. Welcome, admin.');
      document.dispatchEvent(new CustomEvent('easter-egg-triggered'));
      const sub = document.querySelector('.hero-subtitle') || document.querySelector('.subtitle');
      if (sub) {
        const original = sub.textContent;
        sub.textContent = 'Root Access Engineer \u2014 Full Control';
        sub.style.color = '#10b981';
        setTimeout(() => { sub.textContent = original; sub.style.color = ''; }, 3000);
      }
    }
  });
}
