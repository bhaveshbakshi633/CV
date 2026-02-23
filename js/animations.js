// animations.js — scroll reveal + tab visibility (pause videos when hidden)

export function initScrollAnimations() {
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (prefersReducedMotion) {
    document.querySelectorAll('.scroll-reveal').forEach(el => el.classList.add('revealed'));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.05,
    rootMargin: '0px 0px -20px 0px'
  });

  // bento cards get a CSS keyframe entrance, NOT scroll-reveal
  // (they're the first content after hero — must be visible)
  document.querySelectorAll('.bento-card').forEach((card, i) => {
    card.style.animationDelay = `${i * 0.12}s`;
    card.classList.add('bento-enter');
  });

  // only add scroll-reveal to sections BELOW the bento grid
  const belowFoldSelectors = [
    '.capability-section',
    '.capability-project-card',
    '.about-section',
    '.experience-section',
    '.contact-section'
  ];

  belowFoldSelectors.forEach(selector => {
    document.querySelectorAll(selector).forEach(el => {
      if (!el.classList.contains('scroll-reveal')) {
        el.classList.add('scroll-reveal');
        observer.observe(el);
      }
    });
  });

  // immediately reveal anything already in the viewport
  // (prevents content from being stuck at opacity 0)
  requestAnimationFrame(() => {
    document.querySelectorAll('.scroll-reveal').forEach(el => {
      const rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight + 50 && rect.bottom > 0) {
        el.classList.add('revealed');
        observer.unobserve(el);
      }
    });
  });
}

export function initTabVisibility() {
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      document.body.classList.add('tab-hidden');
      document.querySelectorAll('video').forEach(v => v.pause());
    } else {
      document.body.classList.remove('tab-hidden');
      // resume autoplay videos in visible containers
      document.querySelectorAll('.bento-card video, .hero-video-container video').forEach(v => {
        v.play().catch(() => {});
      });
    }
  });
}
