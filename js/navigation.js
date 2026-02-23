// navigation.js — topbar scroll, sidebar toggle, smooth scroll, active sections

let sidebar, sidebarOverlay, hamburger, topbar;
let projectDetailView, mainContent;

function toggleSidebar() {
  sidebar.classList.toggle('open');
  sidebarOverlay.classList.toggle('active');
  hamburger.classList.toggle('active');
  document.body.style.overflow = sidebar.classList.contains('open') ? 'hidden' : '';
}

export function closeSidebar() {
  if (!sidebar) return;
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('active');
  hamburger.classList.remove('active');
  document.body.style.overflow = '';
}

function handleNavClick(e) {
  const link = e.target.closest('.nav-link, .topbar-link');
  if (!link) return;

  e.preventDefault();

  // close project detail if open
  if (projectDetailView && projectDetailView.classList.contains('active')) {
    mainContent.classList.remove('hidden');
    projectDetailView.classList.remove('active');
  }

  const targetId = link.getAttribute('href')?.slice(1);
  const target = document.getElementById(targetId);
  if (target) target.scrollIntoView({ behavior: 'smooth' });

  closeSidebar();
}

function initTopbarScroll() {
  window.addEventListener('scroll', () => {
    if (window.scrollY > 80) {
      topbar.classList.add('scrolled');
    } else {
      topbar.classList.remove('scrolled');
    }
  }, { passive: true });
}

function initActiveSectionTracking() {
  const sections = document.querySelectorAll('.capability-section');
  const topbarLinks = document.querySelectorAll('.topbar-link');

  if (!sections.length || !topbarLinks.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const capability = entry.target.dataset.capability;
        topbarLinks.forEach(link => {
          link.classList.toggle('active', link.dataset.capability === capability);
        });
      }
    });
  }, { threshold: 0.2, rootMargin: '-80px 0px -50% 0px' });

  sections.forEach(section => observer.observe(section));
}

function initCopyButtons() {
  document.querySelectorAll('.contact-copy').forEach(btn => {
    btn.addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();
      const text = btn.dataset.copy;
      navigator.clipboard.writeText(text).then(() => {
        const toast = document.getElementById('toast');
        toast.textContent = 'Copied to clipboard';
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 2000);
      });
    });
  });
}

export function initNavigation() {
  sidebar = document.getElementById('sidebar');
  sidebarOverlay = document.getElementById('sidebarOverlay');
  hamburger = document.getElementById('hamburger');
  topbar = document.getElementById('topbar');
  projectDetailView = document.getElementById('projectDetailView');
  mainContent = document.getElementById('main');

  // sidebar toggle
  hamburger?.addEventListener('click', toggleSidebar);
  document.getElementById('sidebarClose')?.addEventListener('click', closeSidebar);
  sidebarOverlay?.addEventListener('click', closeSidebar);

  // nav links — sidebar + topbar
  document.querySelectorAll('.nav-link, .topbar-link').forEach(link => {
    link.addEventListener('click', handleNavClick);
  });

  // topbar bg on scroll
  initTopbarScroll();

  // highlight active capability section in topbar
  initActiveSectionTracking();

  // copy-to-clipboard buttons
  initCopyButtons();

  // keyboard: Escape closes sidebar (lower priority than lightbox)
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && sidebar?.classList.contains('open')) {
      closeSidebar();
    }
  });

  // responsive: auto-close sidebar on desktop
  window.addEventListener('resize', () => {
    if (window.innerWidth >= 1024) closeSidebar();
  });
}
