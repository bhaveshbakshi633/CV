// =============================================
// SUPER CV PORTFOLIO - Creative JavaScript
// Particles, Typing, Custom Cursor, Animations
// =============================================

// Project Data
const projects = [
  {
    id: 'atv',
    title: 'All-Terrain Vehicle (ATV) Design',
    shortTitle: 'ATV Design',
    description: `
      Engineered a lightweight, high-performance ATV chassis using advanced finite element analysis.
      The project involved complete mechanical design from concept to manufacturing, with a focus on
      durability and performance across diverse terrain conditions.

      Key achievements include optimized weight distribution for improved handling, structural analysis
      ensuring safety under extreme loads, and design modifications that reduced manufacturing costs
      by 25% while maintaining quality standards.
    `,
    tags: ['SolidWorks', 'FEA', 'Manufacturing', 'Ansys'],
    images: ['assets/projects/atv/img1.jpg', 'assets/projects/atv/img2.jpg', 'assets/projects/atv/img3.jpg'],
    video: 'assets/projects/atv/video.mp4',
    thumbnail: 'assets/projects/atv/img1.jpg'
  },
  {
    id: 'surgical-robot',
    title: 'Surgical Robotics Assistant',
    shortTitle: 'Surgical Robot',
    description: `
      Developed modular robotic systems for precision surgical applications. The project focused on
      creating reliable, accurate robotic assistance for minimally invasive procedures.

      Features include advanced motion control with sub-millimeter precision, real-time feedback
      systems for surgeon guidance, and intuitive operator interfaces. The system was designed with
      safety as the primary concern, incorporating multiple redundant safety mechanisms.
    `,
    tags: ['Robotics', 'Medical', 'Control Systems', 'Python'],
    images: ['assets/projects/surgical-robot/img1.jpg', 'assets/projects/surgical-robot/img2.jpg'],
    video: 'assets/projects/surgical-robot/video.mp4',
    thumbnail: 'assets/projects/surgical-robot/img1.jpg'
  },
  {
    id: 'chess-board',
    title: 'Automated Chess Board System',
    shortTitle: 'Chess Board',
    description: `
      Created an interactive chessboard featuring electromagnetic piece movement, computer vision
      for move detection, and network connectivity for remote gameplay.

      The system combines mechanical precision with intelligent software algorithms. Uses XY gantry
      system with electromagnets to physically move chess pieces, Reed switches for piece detection,
      and a custom chess engine for AI gameplay.
    `,
    tags: ['Automation', 'Computer Vision', 'IoT', 'Arduino'],
    images: ['assets/projects/chess-board/img1.jpg', 'assets/projects/chess-board/img2.jpg', 'assets/projects/chess-board/img3.jpg'],
    video: 'assets/projects/chess-board/video.mp4',
    thumbnail: 'assets/projects/chess-board/img1.jpg'
  },
  {
    id: 'ir-remote',
    title: 'Smart IR Control Network',
    shortTitle: 'IR Remote',
    description: `
      Designed a microcontroller-based infrared remote system capable of device identification,
      learning, and selective control.

      Features intelligent protocol detection that can learn and replicate IR signals from any
      device. Includes a centralized home automation management system with smartphone app control.
    `,
    tags: ['Arduino', 'IoT', 'Home Automation', 'ESP32'],
    images: ['assets/projects/ir-remote/img1.jpg', 'assets/projects/ir-remote/img2.jpg'],
    video: 'assets/projects/ir-remote/video.mp4',
    thumbnail: 'assets/projects/ir-remote/img1.jpg'
  }
];

// DOM Elements
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const sidebarClose = document.getElementById('sidebarClose');
const hamburger = document.getElementById('hamburger');
const projectList = document.getElementById('projectList');
const contentWrapper = document.getElementById('contentWrapper');
const projectDetailView = document.getElementById('projectDetailView');
const projectDetail = document.getElementById('projectDetail');
const backBtn = document.getElementById('backBtn');
const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightboxImage');
const lightboxClose = document.getElementById('lightboxClose');
const lightboxPrev = document.getElementById('lightboxPrev');
const lightboxNext = document.getElementById('lightboxNext');
const lightboxCounter = document.getElementById('lightboxCounter');
const cursor = document.getElementById('cursor');
const cursorFollower = document.getElementById('cursorFollower');
const openSidebarBtn = document.getElementById('openSidebarBtn');

// State
let currentProject = null;
let currentImageIndex = 0;
let currentImages = [];
let mouseX = 0, mouseY = 0;
let cursorX = 0, cursorY = 0;

// =============================================
// PARTICLE BACKGROUND
// =============================================
function initParticles() {
  const canvas = document.getElementById('particleCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let particles = [];

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  class Particle {
    constructor() {
      this.reset();
    }

    reset() {
      this.x = Math.random() * canvas.width;
      this.y = Math.random() * canvas.height;
      this.size = Math.random() * 2 + 0.5;
      this.speedX = (Math.random() - 0.5) * 0.5;
      this.speedY = (Math.random() - 0.5) * 0.5;
      this.opacity = Math.random() * 0.5 + 0.2;
    }

    update() {
      this.x += this.speedX;
      this.y += this.speedY;

      if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
      if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
    }

    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 240, 255, ${this.opacity})`;
      ctx.fill();
    }
  }

  function init() {
    resize();
    particles = [];
    const particleCount = Math.min(100, Math.floor((canvas.width * canvas.height) / 15000));
    for (let i = 0; i < particleCount; i++) {
      particles.push(new Particle());
    }
  }

  function drawLines() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 150) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0, 240, 255, ${0.1 * (1 - distance / 150)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  }

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    particles.forEach(particle => {
      particle.update();
      particle.draw();
    });

    drawLines();
    requestAnimationFrame(animate);
  }

  window.addEventListener('resize', init);
  init();
  animate();
}

// =============================================
// CUSTOM CURSOR
// =============================================
function initCursor() {
  if (!cursor || !cursorFollower || window.innerWidth < 1024) return;

  // Cursor position - dono ko same position pe rakhna hai
  let targetX = 0, targetY = 0;
  let followerX = 0, followerY = 0;

  document.addEventListener('mousemove', (e) => {
    targetX = e.clientX;
    targetY = e.clientY;
  });

  function animateCursor() {
    // Follower smoothly follows with delay
    followerX += (targetX - followerX) * 0.15;
    followerY += (targetY - followerY) * 0.15;

    // Dot follows mouse directly (instant)
    cursor.style.left = targetX + 'px';
    cursor.style.top = targetY + 'px';

    // Circle follows with smooth delay
    cursorFollower.style.left = followerX + 'px';
    cursorFollower.style.top = followerY + 'px';

    requestAnimationFrame(animateCursor);
  }

  animateCursor();

  // Hover effect on interactive elements
  const interactiveElements = document.querySelectorAll('a, button, .skill-card, .contact-card, .stat-card, .gallery-item');
  interactiveElements.forEach(el => {
    el.addEventListener('mouseenter', () => cursorFollower.classList.add('hover'));
    el.addEventListener('mouseleave', () => cursorFollower.classList.remove('hover'));
  });
}

// =============================================
// TYPING EFFECT
// =============================================
function initTyping() {
  const typingElement = document.getElementById('typingText');
  if (!typingElement) return;

  const phrases = [
    'Robotics Innovation Specialist',
    'Mechanical Design Expert',
    'Automation Engineer',
    'Problem Solver',
    'Future Builder'
  ];

  let phraseIndex = 0;
  let charIndex = 0;
  let isDeleting = false;
  let typingSpeed = 100;

  function type() {
    const currentPhrase = phrases[phraseIndex];

    if (isDeleting) {
      typingElement.textContent = currentPhrase.substring(0, charIndex - 1);
      charIndex--;
      typingSpeed = 50;
    } else {
      typingElement.textContent = currentPhrase.substring(0, charIndex + 1);
      charIndex++;
      typingSpeed = 100;
    }

    if (!isDeleting && charIndex === currentPhrase.length) {
      typingSpeed = 2000; // Pause at end
      isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
      isDeleting = false;
      phraseIndex = (phraseIndex + 1) % phrases.length;
      typingSpeed = 500; // Pause before new phrase
    }

    setTimeout(type, typingSpeed);
  }

  setTimeout(type, 1000);
}

// =============================================
// NUMBER COUNTER ANIMATION
// =============================================
function initCounters() {
  const counters = document.querySelectorAll('.stat-number[data-count]');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const target = entry.target;
        const count = parseInt(target.dataset.count);
        animateCounter(target, count);
        observer.unobserve(target);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(counter => observer.observe(counter));
}

function animateCounter(element, target) {
  let current = 0;
  const duration = 2000;
  const increment = target / (duration / 16);

  function update() {
    current += increment;
    if (current < target) {
      element.textContent = Math.floor(current);
      requestAnimationFrame(update);
    } else {
      element.textContent = target;
    }
  }

  update();
}

// =============================================
// CHARACTER REVEAL ANIMATION
// =============================================
function initCharReveal() {
  const chars = document.querySelectorAll('.char');
  chars.forEach((char, index) => {
    char.style.setProperty('--char-index', index);
  });
}

// =============================================
// SIDEBAR FUNCTIONS
// =============================================
function toggleSidebar() {
  sidebar.classList.toggle('open');
  sidebarOverlay.classList.toggle('active');
  hamburger.classList.toggle('active');
  document.body.style.overflow = sidebar.classList.contains('open') ? 'hidden' : '';
}

function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('active');
  hamburger.classList.remove('active');
  document.body.style.overflow = '';
}

// =============================================
// PROJECT LIST RENDER
// =============================================
function renderProjectList() {
  projectList.innerHTML = projects.map(project => `
    <li class="project-item">
      <button class="project-toggle" data-project-id="${project.id}">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="18" height="18" rx="2"></rect>
        </svg>
        ${project.shortTitle}
        <svg class="arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="9,18 15,12 9,6"></polyline>
        </svg>
      </button>
      <div class="project-dropdown" id="dropdown-${project.id}">
        <div class="project-dropdown-content">
          <img src="${project.thumbnail}" alt="${project.title}" class="project-thumbnail" onerror="this.style.display='none'">
          <div class="project-tags-mini">
            ${project.tags.slice(0, 3).map(tag => `<span class="tag-mini">${tag}</span>`).join('')}
          </div>
          <button class="view-project-btn" data-project-id="${project.id}">View Details</button>
        </div>
      </div>
    </li>
  `).join('');

  document.querySelectorAll('.project-toggle').forEach(btn => {
    btn.addEventListener('click', () => toggleProjectDropdown(btn.dataset.projectId));
  });

  document.querySelectorAll('.view-project-btn').forEach(btn => {
    btn.addEventListener('click', () => showProjectDetail(btn.dataset.projectId));
  });
}

function toggleProjectDropdown(projectId) {
  const toggle = document.querySelector(`.project-toggle[data-project-id="${projectId}"]`);
  const dropdown = document.getElementById(`dropdown-${projectId}`);

  if (toggle.classList.contains('expanded')) {
    toggle.classList.remove('expanded');
    dropdown.classList.remove('open');
  } else {
    document.querySelectorAll('.project-toggle').forEach(t => t.classList.remove('expanded'));
    document.querySelectorAll('.project-dropdown').forEach(d => d.classList.remove('open'));
    toggle.classList.add('expanded');
    dropdown.classList.add('open');
  }
}

// =============================================
// PROJECT DETAIL VIEW
// =============================================
function showProjectDetail(projectId) {
  const project = projects.find(p => p.id === projectId);
  if (!project) return;

  currentProject = project;

  document.querySelectorAll('.project-toggle').forEach(t => {
    t.classList.remove('active');
    if (t.dataset.projectId === projectId) t.classList.add('active');
  });

  projectDetail.innerHTML = `
    <div class="project-detail-header">
      <h1 class="project-detail-title">${project.title}</h1>
      <div class="project-detail-tags">
        ${project.tags.map(tag => `<span class="project-tag">${tag}</span>`).join('')}
      </div>
    </div>

    <div class="project-detail-description">
      ${project.description.trim().split('\n\n').map(p => `<p>${p.trim()}</p>`).join('')}
    </div>

    ${project.images?.length ? `
      <div class="project-gallery">
        <h3 class="gallery-title">Project Gallery</h3>
        <div class="gallery-grid">
          ${project.images.map((img, i) => `
            <div class="gallery-item" data-index="${i}">
              <img src="${img}" alt="${project.title} - Image ${i + 1}" onerror="this.parentElement.style.display='none'">
            </div>
          `).join('')}
        </div>
      </div>
    ` : ''}

    ${project.video ? `
      <div class="project-video">
        <h3 class="video-title">Project Video</h3>
        <div class="video-container">
          <video controls preload="metadata">
            <source src="${project.video}" type="video/mp4">
            Your browser does not support video.
          </video>
        </div>
      </div>
    ` : ''}
  `;

  document.querySelectorAll('.gallery-item').forEach(item => {
    item.addEventListener('click', () => {
      currentImages = project.images;
      openLightbox(parseInt(item.dataset.index));
    });
  });

  contentWrapper.classList.add('hidden');
  projectDetailView.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });

  if (window.innerWidth < 1024) closeSidebar();
}

function hideProjectDetail() {
  contentWrapper.classList.remove('hidden');
  projectDetailView.classList.remove('active');
  currentProject = null;
  document.querySelectorAll('.project-toggle').forEach(t => t.classList.remove('active'));
}

// =============================================
// LIGHTBOX
// =============================================
function openLightbox(index) {
  currentImageIndex = index;
  updateLightboxImage();
  lightbox.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeLightboxModal() {
  lightbox.classList.remove('active');
  document.body.style.overflow = '';
}

function updateLightboxImage() {
  if (!currentImages.length) return;
  lightboxImage.src = currentImages[currentImageIndex];
  lightboxCounter.textContent = `${currentImageIndex + 1} / ${currentImages.length}`;
}

function nextImage() {
  currentImageIndex = (currentImageIndex + 1) % currentImages.length;
  updateLightboxImage();
}

function prevImage() {
  currentImageIndex = (currentImageIndex - 1 + currentImages.length) % currentImages.length;
  updateLightboxImage();
}

// =============================================
// NAVIGATION
// =============================================
function handleNavigation(e) {
  const link = e.target.closest('.nav-link');
  if (!link) return;

  e.preventDefault();
  if (projectDetailView.classList.contains('active')) hideProjectDetail();

  const targetId = link.getAttribute('href').slice(1);
  const target = document.getElementById(targetId);
  if (target) target.scrollIntoView({ behavior: 'smooth' });

  if (window.innerWidth < 1024) closeSidebar();
}

// =============================================
// SCROLL ANIMATIONS
// =============================================
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.section').forEach(section => {
    section.style.opacity = '0';
    section.style.transform = 'translateY(30px)';
    section.style.transition = 'all 0.8s ease';
    observer.observe(section);
  });
}

// Add visible state styles
const style = document.createElement('style');
style.textContent = `.section.visible { opacity: 1 !important; transform: translateY(0) !important; }`;
document.head.appendChild(style);

// =============================================
// INIT
// =============================================
document.addEventListener('DOMContentLoaded', () => {
  // Initialize features
  initParticles();
  initCursor();
  initTyping();
  initCounters();
  initCharReveal();
  initScrollAnimations();
  renderProjectList();

  // Event listeners
  hamburger?.addEventListener('click', toggleSidebar);
  sidebarClose?.addEventListener('click', closeSidebar);
  sidebarOverlay?.addEventListener('click', closeSidebar);

  // Browse Projects button - mobile pe sidebar kholo, desktop pe first project kholo
  openSidebarBtn?.addEventListener('click', () => {
    if (window.innerWidth < 1024) {
      // Mobile pe sidebar toggle karo
      toggleSidebar();
    } else {
      // Desktop pe first project expand karo aur uski detail dikhao
      const firstProject = projects[0];
      if (firstProject) {
        toggleProjectDropdown(firstProject.id);
        // Thodi der baad project detail view kholo
        setTimeout(() => {
          showProjectDetail(firstProject.id);
        }, 300);
      }
    }
  });

  backBtn?.addEventListener('click', hideProjectDetail);
  lightboxClose?.addEventListener('click', closeLightboxModal);
  lightboxPrev?.addEventListener('click', prevImage);
  lightboxNext?.addEventListener('click', nextImage);

  lightbox?.addEventListener('click', (e) => {
    if (e.target === lightbox) closeLightboxModal();
  });

  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', handleNavigation);
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (lightbox?.classList.contains('active')) {
      if (e.key === 'Escape') closeLightboxModal();
      if (e.key === 'ArrowRight') nextImage();
      if (e.key === 'ArrowLeft') prevImage();
    } else if (e.key === 'Escape' && sidebar?.classList.contains('open')) {
      closeSidebar();
    }
  });

  // Resize handler
  window.addEventListener('resize', () => {
    if (window.innerWidth >= 1024) {
      sidebarOverlay?.classList.remove('active');
      hamburger?.classList.remove('active');
      document.body.style.overflow = '';
    }
    initCursor();
  });
});

// Export for external use
window.portfolioProjects = projects;
window.addProject = (data) => { projects.push(data); renderProjectList(); };
