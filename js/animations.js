// animations.js — cinematic scroll reveal + tab visibility (pause videos when hidden)
// premium slide-up + stagger — Linear/Apple level smooth entrance

export function initScrollAnimations() {
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (prefersReducedMotion) {
    // reduced motion waalon ko seedha dikha do, animation nahi chahiye unhe
    document.querySelectorAll('.scroll-reveal, .reveal').forEach(el => el.classList.add('revealed'));
    return;
  }

  // cinematic reveal observer — 30px slide-up ke saath fade in
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        revealObserver.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.08,
    rootMargin: '0px 0px -40px 0px'
  });

  // bento cards ko keyframe entrance dena — hero ke turant baad hain
  document.querySelectorAll('.bento-card').forEach((card, i) => {
    card.style.animationDelay = `${i * 0.12}s`;
    card.classList.add('bento-enter');
  });

  // capability sections ke children ko stagger karo — waterfall effect
  document.querySelectorAll('.capability-section').forEach(section => {
    // section itself gets cinematic reveal
    section.classList.add('reveal');
    revealObserver.observe(section);

    // children ko stagger class lagao — header, desc, projects grid
    const staggerParent = section;
    const staggerChildren = staggerParent.querySelectorAll(
      ':scope > .capability-section-header, :scope > .capability-desc, :scope > .capability-projects'
    );
    if (staggerChildren.length > 0) {
      staggerParent.classList.add('reveal-stagger');
      staggerChildren.forEach(child => {
        child.classList.add('reveal');
        revealObserver.observe(child);
      });
    }
  });

  // capability project cards ko bhi reveal lagao — stagger ke saath
  document.querySelectorAll('.capability-projects').forEach(grid => {
    grid.classList.add('reveal-stagger');
    grid.querySelectorAll('.capability-project-card').forEach(card => {
      card.classList.add('reveal');
      revealObserver.observe(card);
    });
  });

  // baaki sections — about, experience, contact ko cinematic reveal
  const otherSections = [
    '.about-section',
    '.experience-section',
    '.contact-section'
  ];

  otherSections.forEach(selector => {
    document.querySelectorAll(selector).forEach(el => {
      if (!el.classList.contains('reveal')) {
        el.classList.add('reveal');
        revealObserver.observe(el);
      }
    });
  });

  // game containers bhi reveal hone chahiye — scroll karte waqt
  document.querySelectorAll('.game-container, .game-standalone').forEach(el => {
    if (!el.classList.contains('reveal')) {
      el.classList.add('reveal');
      revealObserver.observe(el);
    }
  });

  // viewport mein jo already hai, seedha reveal karo — opacity 0 pe nahi rehna chahiye
  requestAnimationFrame(() => {
    document.querySelectorAll('.reveal:not(.revealed)').forEach(el => {
      const rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight + 50 && rect.bottom > 0) {
        el.classList.add('revealed');
        revealObserver.unobserve(el);
      }
    });
    // puraane scroll-reveal waalon ko bhi handle karo
    document.querySelectorAll('.scroll-reveal:not(.revealed)').forEach(el => {
      const rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight + 50 && rect.bottom > 0) {
        el.classList.add('revealed');
      }
    });
  });
}

// lazy images ko smoothly fade-in karo jab load ho jaayein
export function initLazyImageFade() {
  document.querySelectorAll('img[loading="lazy"]').forEach(img => {
    if (img.complete) {
      // already load ho chuki hai — seedha dikha do
      img.classList.add('loaded');
    } else {
      img.addEventListener('load', () => img.classList.add('loaded'), { once: true });
    }
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
