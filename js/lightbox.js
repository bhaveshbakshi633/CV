// lightbox.js — fullscreen image gallery
// arrow nav, keyboard support, click-outside close

let currentImages = [];
let currentIndex = 0;
let lightboxEl, lightboxImg, lightboxCounter;

function update() {
  if (!currentImages.length) return;
  lightboxImg.src = currentImages[currentIndex];
  lightboxCounter.textContent = `${currentIndex + 1} / ${currentImages.length}`;
}

function next() {
  currentIndex = (currentIndex + 1) % currentImages.length;
  update();
}

function prev() {
  currentIndex = (currentIndex - 1 + currentImages.length) % currentImages.length;
  update();
}

function close() {
  lightboxEl.classList.remove('active');
  document.body.style.overflow = '';
}

export function openLightbox(images, index = 0) {
  currentImages = images;
  currentIndex = index;
  update();
  lightboxEl.classList.add('active');
  document.body.style.overflow = 'hidden';
}

export function isLightboxActive() {
  return lightboxEl?.classList.contains('active');
}

export function initLightbox() {
  lightboxEl = document.getElementById('lightbox');
  lightboxImg = document.getElementById('lightboxImage');
  lightboxCounter = document.getElementById('lightboxCounter');

  if (!lightboxEl) return;

  document.getElementById('lightboxClose').addEventListener('click', close);
  document.getElementById('lightboxPrev').addEventListener('click', prev);
  document.getElementById('lightboxNext').addEventListener('click', next);

  lightboxEl.addEventListener('click', e => {
    if (e.target === lightboxEl) close();
  });

  // keyboard nav — highest priority (lightbox on top of everything)
  document.addEventListener('keydown', e => {
    if (!lightboxEl.classList.contains('active')) return;
    if (e.key === 'Escape') { e.stopImmediatePropagation(); close(); }
    if (e.key === 'ArrowRight') next();
    if (e.key === 'ArrowLeft') prev();
  });
}
