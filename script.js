// =============================================
// SUPER CV PORTFOLIO - JavaScript
// =============================================

// Project Data - yahan pe saare projects ki info hai
// Images aur videos ke paths ko update karna hai jab media add karo
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
    images: [
      'assets/projects/atv/img1.jpg',
      'assets/projects/atv/img2.jpg',
      'assets/projects/atv/img3.jpg'
    ],
    video: 'assets/projects/atv/video.mp4',
    thumbnail: 'assets/projects/atv/img1.jpg',
    links: []
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
    images: [
      'assets/projects/surgical-robot/img1.jpg',
      'assets/projects/surgical-robot/img2.jpg'
    ],
    video: 'assets/projects/surgical-robot/video.mp4',
    thumbnail: 'assets/projects/surgical-robot/img1.jpg',
    links: []
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
      and a custom chess engine for AI gameplay. Supports both local and online multiplayer modes.
    `,
    tags: ['Automation', 'Computer Vision', 'IoT', 'Arduino'],
    images: [
      'assets/projects/chess-board/img1.jpg',
      'assets/projects/chess-board/img2.jpg',
      'assets/projects/chess-board/img3.jpg'
    ],
    video: 'assets/projects/chess-board/video.mp4',
    thumbnail: 'assets/projects/chess-board/img1.jpg',
    links: []
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
      The system can control TVs, ACs, fans, and other IR-controlled devices through a single interface.
    `,
    tags: ['Arduino', 'IoT', 'Home Automation', 'ESP32'],
    images: [
      'assets/projects/ir-remote/img1.jpg',
      'assets/projects/ir-remote/img2.jpg'
    ],
    video: 'assets/projects/ir-remote/video.mp4',
    thumbnail: 'assets/projects/ir-remote/img1.jpg',
    links: []
  }
];

// DOM Elements - sabhi elements ko yahan grab kar rahe hain
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

// State - current state track karne ke liye
let currentProject = null;
let currentImageIndex = 0;
let currentImages = [];

// =============================================
// SIDEBAR FUNCTIONS
// =============================================

// Sidebar toggle karne ka function
function toggleSidebar() {
  sidebar.classList.toggle('open');
  sidebarOverlay.classList.toggle('active');
  hamburger.classList.toggle('active');
  document.body.style.overflow = sidebar.classList.contains('open') ? 'hidden' : '';
}

// Sidebar close karne ka function
function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('active');
  hamburger.classList.remove('active');
  document.body.style.overflow = '';
}

// =============================================
// PROJECT LIST RENDER
// =============================================

// Sidebar me projects render karne ka function
function renderProjectList() {
  projectList.innerHTML = projects.map(project => `
    <li class="project-item">
      <button class="project-toggle" data-project-id="${project.id}">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        </svg>
        ${project.shortTitle}
        <svg class="arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
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

  // Event listeners add karo toggle buttons pe
  document.querySelectorAll('.project-toggle').forEach(btn => {
    btn.addEventListener('click', () => toggleProjectDropdown(btn.dataset.projectId));
  });

  // View details buttons pe listeners
  document.querySelectorAll('.view-project-btn').forEach(btn => {
    btn.addEventListener('click', () => showProjectDetail(btn.dataset.projectId));
  });
}

// Project dropdown toggle karne ka function
function toggleProjectDropdown(projectId) {
  const toggle = document.querySelector(`.project-toggle[data-project-id="${projectId}"]`);
  const dropdown = document.getElementById(`dropdown-${projectId}`);

  // Agar already expanded hai toh close karo
  if (toggle.classList.contains('expanded')) {
    toggle.classList.remove('expanded');
    dropdown.classList.remove('open');
  } else {
    // Pehle sabhi dropdowns close karo
    document.querySelectorAll('.project-toggle').forEach(t => t.classList.remove('expanded'));
    document.querySelectorAll('.project-dropdown').forEach(d => d.classList.remove('open'));

    // Current dropdown open karo
    toggle.classList.add('expanded');
    dropdown.classList.add('open');
  }
}

// =============================================
// PROJECT DETAIL VIEW
// =============================================

// Project detail show karne ka function
function showProjectDetail(projectId) {
  const project = projects.find(p => p.id === projectId);
  if (!project) return;

  currentProject = project;

  // Mark project as active in sidebar
  document.querySelectorAll('.project-toggle').forEach(t => {
    t.classList.remove('active');
    if (t.dataset.projectId === projectId) {
      t.classList.add('active');
    }
  });

  // Render project detail
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

    ${project.images && project.images.length > 0 ? `
      <div class="project-gallery">
        <h3 class="gallery-title">Project Gallery</h3>
        <div class="gallery-grid">
          ${project.images.map((img, index) => `
            <div class="gallery-item" data-index="${index}">
              <img src="${img}" alt="${project.title} - Image ${index + 1}" onerror="this.parentElement.style.display='none'">
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
            Your browser does not support video playback.
          </video>
        </div>
      </div>
    ` : ''}

    ${project.links && project.links.length > 0 ? `
      <div class="project-links">
        ${project.links.map(link => `
          <a href="${link.url}" target="_blank" class="project-link">
            ${link.label}
          </a>
        `).join('')}
      </div>
    ` : ''}
  `;

  // Gallery items pe click listener add karo
  document.querySelectorAll('.gallery-item').forEach(item => {
    item.addEventListener('click', () => {
      currentImages = project.images;
      openLightbox(parseInt(item.dataset.index));
    });
  });

  // View switch karo
  contentWrapper.classList.add('hidden');
  projectDetailView.classList.add('active');

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Mobile pe sidebar close karo
  if (window.innerWidth < 1024) {
    closeSidebar();
  }
}

// Back to overview function
function hideProjectDetail() {
  contentWrapper.classList.remove('hidden');
  projectDetailView.classList.remove('active');
  currentProject = null;

  // Remove active state from sidebar
  document.querySelectorAll('.project-toggle').forEach(t => t.classList.remove('active'));
}

// =============================================
// LIGHTBOX FUNCTIONS
// =============================================

// Lightbox open karne ka function
function openLightbox(index) {
  currentImageIndex = index;
  updateLightboxImage();
  lightbox.classList.add('active');
  document.body.style.overflow = 'hidden';
}

// Lightbox close karne ka function
function closeLightbox() {
  lightbox.classList.remove('active');
  document.body.style.overflow = '';
}

// Lightbox image update karne ka function
function updateLightboxImage() {
  if (currentImages.length === 0) return;
  lightboxImage.src = currentImages[currentImageIndex];
  lightboxCounter.textContent = `${currentImageIndex + 1} / ${currentImages.length}`;
}

// Next image
function nextImage() {
  currentImageIndex = (currentImageIndex + 1) % currentImages.length;
  updateLightboxImage();
}

// Previous image
function prevImage() {
  currentImageIndex = (currentImageIndex - 1 + currentImages.length) % currentImages.length;
  updateLightboxImage();
}

// =============================================
// NAVIGATION FUNCTIONS
// =============================================

// Smooth scroll navigation
function handleNavigation(e) {
  const link = e.target.closest('.nav-link');
  if (!link) return;

  e.preventDefault();

  // Pehle project detail view hide karo agar open hai
  if (projectDetailView.classList.contains('active')) {
    hideProjectDetail();
  }

  // Target section pe scroll karo
  const targetId = link.getAttribute('href').slice(1);
  const target = document.getElementById(targetId);

  if (target) {
    target.scrollIntoView({ behavior: 'smooth' });
  }

  // Mobile pe sidebar close karo
  if (window.innerWidth < 1024) {
    closeSidebar();
  }
}

// =============================================
// EVENT LISTENERS
// =============================================

// Page load hone pe
document.addEventListener('DOMContentLoaded', () => {
  // Projects render karo
  renderProjectList();

  // Sidebar toggle
  hamburger.addEventListener('click', toggleSidebar);
  sidebarClose.addEventListener('click', closeSidebar);
  sidebarOverlay.addEventListener('click', closeSidebar);

  // Back button
  backBtn.addEventListener('click', hideProjectDetail);

  // Lightbox controls
  lightboxClose.addEventListener('click', closeLightbox);
  lightboxPrev.addEventListener('click', prevImage);
  lightboxNext.addEventListener('click', nextImage);

  // Lightbox background click se close
  lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox || e.target.classList.contains('lightbox-content')) {
      closeLightbox();
    }
  });

  // Navigation links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', handleNavigation);
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (lightbox.classList.contains('active')) {
      if (e.key === 'Escape') closeLightbox();
      if (e.key === 'ArrowRight') nextImage();
      if (e.key === 'ArrowLeft') prevImage();
    } else {
      if (e.key === 'Escape' && sidebar.classList.contains('open')) {
        closeSidebar();
      }
    }
  });

  // Window resize pe sidebar handle karo
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (window.innerWidth >= 1024) {
        // Desktop pe sidebar always visible
        sidebarOverlay.classList.remove('active');
        hamburger.classList.remove('active');
        document.body.style.overflow = '';
      }
    }, 100);
  });
});

// =============================================
// UTILITY FUNCTIONS
// =============================================

// Naya project add karne ka function - isko use kar sakte ho naye projects ke liye
function addProject(projectData) {
  projects.push(projectData);
  renderProjectList();
}

// Projects ko export kar rahe hain agar kahi aur use karna ho
window.portfolioProjects = projects;
window.addProject = addProject;
