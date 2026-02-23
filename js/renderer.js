// renderer.js — bento grid, capability sections, project detail overlay

import { isProjectProtected, showPasswordModal } from './protection.js';
import { openLightbox, isLightboxActive } from './lightbox.js';

let savedScrollPosition = 0;
let mainContent, projectDetailView, projectDetail;
let allProjectsFlat = [];

const CAPABILITY_COLORS = {
  'real-time-control': 'var(--accent-control)',
  'learning-systems': 'var(--accent-rl)',
  'human-robot-interaction': 'var(--accent-hri)',
  'hardware-build': 'var(--accent-hw)'
};

function getAllProjects(data) {
  return [...data.flagship, ...data.supporting, ...data.archived];
}

function findProject(id) {
  return allProjectsFlat.find(p => p.id === id);
}

// ---- Bento Grid (4 flagships, 2x2) ----

function renderBentoCard(project) {
  const hasVideo = project.videos && project.videos.length > 0;
  const mediaSrc = hasVideo ? project.videos[0].src : project.thumbnail;
  const accentColor = CAPABILITY_COLORS[project.capability] || 'var(--accent-control)';
  const isProtected = isProjectProtected(project.id);

  return `
    <article class="bento-card ${isProtected ? 'protected' : ''}"
             data-project-id="${project.id}"
             data-capability="${project.capability}"
             style="--card-accent: ${accentColor}"
             tabindex="0" role="button"
             aria-label="View ${project.shortTitle} project">
      ${isProtected ? `
        <div class="bento-protected-overlay">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
          </svg>
          <span>Confidential</span>
        </div>
      ` : ''}
      <div class="bento-media">
        ${hasVideo ? `
          <video src="${mediaSrc}" muted loop playsinline preload="metadata"
                 poster="${project.thumbnail || ''}"></video>
        ` : project.thumbnail ? `
          <img src="${project.thumbnail}" alt="${project.shortTitle}" loading="lazy" />
        ` : `
          <div class="bento-placeholder">
            <span class="bento-placeholder-icon">&#9881;</span>
            <span class="bento-placeholder-text">${project.shortTitle}</span>
          </div>
        `}
      </div>
      <div class="bento-info">
        <h3 class="bento-title">${project.shortTitle}</h3>
        <p class="bento-hero-line">${project.heroLine}</p>
        <div class="bento-tags">
          ${project.tags.slice(0, 4).map(tag => `<span class="bento-tag">${tag}</span>`).join('')}
        </div>
      </div>
    </article>`;
}

export function renderBentoGrid(data) {
  const grid = document.getElementById('bentoGrid');
  if (!grid) return;

  grid.innerHTML = data.flagship.map(renderBentoCard).join('');

  // click + keyboard handlers
  grid.querySelectorAll('.bento-card').forEach(card => {
    const handler = () => showProjectDetail(card.dataset.projectId);
    card.addEventListener('click', handler);
    card.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); }
    });
  });

  initVideoObserver(grid);
}

function initVideoObserver(container) {
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReducedMotion) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.play().catch(() => {});
      } else {
        entry.target.pause();
      }
    });
  }, { threshold: 0.2 });

  container.querySelectorAll('video').forEach(v => observer.observe(v));
}

// ---- Capability Sections ----

function renderCapabilityCard(project) {
  const isProtected = isProjectProtected(project.id);
  const isFlagship = project.tier === 'flagship';
  const hasVideo = project.videos && project.videos.length > 0;

  return `
    <div class="capability-project-card ${isFlagship ? 'flagship' : ''} ${isProtected ? 'protected' : ''}"
         data-project-id="${project.id}" tabindex="0" role="button"
         aria-label="View ${project.shortTitle} project">
      ${isProtected ? '<span class="cap-card-lock">Locked</span>' : ''}
      <div class="cap-card-media">
        ${hasVideo ? `
          <video src="${project.videos[0].src}" muted loop playsinline preload="metadata"
                 poster="${project.thumbnail || ''}"></video>
        ` : project.thumbnail ? `
          <img src="${project.thumbnail}" alt="${project.shortTitle}" loading="lazy" />
        ` : `
          <div class="cap-card-placeholder">${project.shortTitle}</div>
        `}
      </div>
      <div class="cap-card-body">
        ${isFlagship ? '<span class="cap-card-badge">Flagship</span>' : ''}
        ${project.ongoing ? '<span class="cap-card-badge ongoing">In Progress</span>' : ''}
        <h4 class="cap-card-title">${project.shortTitle}</h4>
        <p class="cap-card-desc">${project.outcome || project.oneLiner || ''}</p>
        <div class="cap-card-tags">
          ${project.tags.slice(0, 3).map(tag => `<span class="cap-tag">${tag}</span>`).join('')}
        </div>
      </div>
    </div>`;
}

export function renderCapabilitySections(data) {
  const allProjects = getAllProjects(data);

  const capabilityMap = {
    'real-time-control': document.getElementById('capabilityRealTimeControl'),
    'learning-systems': document.getElementById('capabilityLearningSystems'),
    'human-robot-interaction': document.getElementById('capabilityHRI'),
    'hardware-build': document.getElementById('capabilityHardware')
  };

  const tierOrder = { flagship: 0, supporting: 1, archived: 2 };

  Object.entries(capabilityMap).forEach(([capability, container]) => {
    if (!container) return;

    const capProjects = allProjects
      .filter(p => p.capability === capability)
      .sort((a, b) => (tierOrder[a.tier] || 2) - (tierOrder[b.tier] || 2));

    container.innerHTML = capProjects.map(renderCapabilityCard).join('');

    // click + keyboard
    container.querySelectorAll('.capability-project-card').forEach(card => {
      const handler = () => showProjectDetail(card.dataset.projectId);
      card.addEventListener('click', handler);
      card.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); }
      });
    });

    initVideoObserver(container);
  });
}

// ---- Project Detail ----

export function showProjectDetail(projectId) {
  const project = findProject(projectId);
  if (!project) return;

  if (isProjectProtected(projectId)) {
    showPasswordModal(() => showProjectDetail(projectId));
    return;
  }

  savedScrollPosition = window.scrollY;

  const detail = document.getElementById('projectDetail');
  detail.innerHTML = buildDetailHTML(project);

  mainContent.classList.add('hidden');
  projectDetailView.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'instant' });

  // gallery click → lightbox
  detail.querySelectorAll('.gallery-item').forEach(item => {
    item.addEventListener('click', () => {
      openLightbox(project.images, parseInt(item.dataset.index));
    });
  });

  // video: hover unmute, click fullscreen
  detail.querySelectorAll('.project-video-player').forEach(video => {
    video.addEventListener('mouseenter', () => { video.muted = false; });
    video.addEventListener('mouseleave', () => { video.muted = true; });
    video.addEventListener('click', () => {
      video.controls = true;
      if (video.requestFullscreen) video.requestFullscreen();
      else if (video.webkitRequestFullscreen) video.webkitRequestFullscreen();
    });
  });

  // fullscreen exit — hide controls
  const fsHandler = () => {
    if (!document.fullscreenElement) {
      detail.querySelectorAll('.project-video-player').forEach(v => { v.controls = false; });
    }
  };
  document.addEventListener('fullscreenchange', fsHandler, { once: true });

  // tech deep dive toggle
  const deepDiveBtn = detail.querySelector('.tech-deepdive-btn');
  if (deepDiveBtn) {
    deepDiveBtn.addEventListener('click', () => {
      const section = detail.querySelector('.tech-deepdive-section');
      const expanding = !section.classList.contains('expanded');
      section.classList.toggle('expanded');
      deepDiveBtn.classList.toggle('active');
      if (expanding) {
        setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
      }
    });
  }
}

export function hideProjectDetail() {
  mainContent.classList.remove('hidden');
  projectDetailView.classList.remove('active');
  window.scrollTo({ top: savedScrollPosition, behavior: 'instant' });
}

function buildDetailHTML(project) {
  return `
    <div class="project-detail-header">
      <div class="project-title-row">
        <h1 class="project-detail-title">${project.title}</h1>
        ${project.technicalDeepDive ? `
          <button class="tech-deepdive-btn">
            <span class="deepdive-icon">&#9889;</span>
            <span>Technical Deep Dive</span>
            <svg class="deepdive-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2">
              <polyline points="6,9 12,15 18,9"></polyline>
            </svg>
          </button>
        ` : ''}
      </div>
      ${project.ongoing ? `
        <div class="ongoing-badge-large">
          <span class="ongoing-dot"></span>
          ${project.status || 'In Progress'}
        </div>
      ` : ''}
      ${project.oneLiner ? `<p class="project-detail-oneliner">${project.oneLiner}</p>` : ''}
      <div class="project-detail-tags">
        ${project.tags.map(tag => `<span class="project-tag">${tag}</span>`).join('')}
      </div>
    </div>

    <div class="project-detail-description">
      ${formatDescription(project.description)}
    </div>

    ${renderMetrics(project)}
    ${renderGallery(project)}
    ${renderVideos(project)}
    ${renderTechDeepDive(project)}
  `;
}

function formatDescription(desc) {
  if (!desc) return '';
  return desc.trim().split('\n\n').map(p => {
    const trimmed = p.trim();
    if (!trimmed) return '';
    if (trimmed.includes('\u2022')) {
      const lines = trimmed.split('\n').map(l => l.trim());
      const items = lines.filter(l => l.startsWith('\u2022')).map(l => `<li>${l.substring(1).trim()}</li>`).join('');
      const intro = lines.filter(l => !l.startsWith('\u2022')).join(' ');
      return intro ? `<p>${intro}</p><ul class="tech-list">${items}</ul>` : `<ul class="tech-list">${items}</ul>`;
    }
    return `<p>${trimmed}</p>`;
  }).join('');
}

function renderMetrics(project) {
  const metrics = project.technicalDeepDive?.metrics;
  if (!metrics?.length) return '';
  return `
    <div class="project-metrics-grid">
      ${metrics.map(m => `
        <div class="project-metric-card">
          <span class="metric-value">${m.value}</span>
          <span class="metric-label">${m.label}</span>
        </div>
      `).join('')}
    </div>`;
}

function renderGallery(project) {
  if (!project.images?.length || project.hideGallery) return '';
  return `
    <div class="project-gallery">
      <h3 class="gallery-title">Gallery</h3>
      <div class="gallery-grid">
        ${project.images.map((img, i) => `
          <div class="gallery-item" data-index="${i}">
            <img src="${img}" alt="${project.title} - ${i + 1}" loading="lazy"
                 onerror="this.parentElement.style.display='none'" />
          </div>
        `).join('')}
      </div>
    </div>`;
}

function renderVideos(project) {
  if (!project.videos?.length) return '';
  return `
    <div class="project-videos">
      <h3 class="video-title">Videos</h3>
      <div class="videos-grid">
        ${project.videos.map(vid => `
          <div class="video-card">
            <div class="video-container">
              <video class="project-video-player" autoplay muted loop playsinline preload="auto">
                <source src="${vid.src}" type="video/mp4">
              </video>
            </div>
            ${vid.title ? `<p class="video-label">${vid.title}</p>` : ''}
          </div>
        `).join('')}
      </div>
    </div>`;
}

function renderTechDeepDive(project) {
  if (!project.technicalDeepDive) return '';
  const td = project.technicalDeepDive;
  return `
    <div class="tech-deepdive-section" id="techDeepDive">
      <div class="tech-deepdive-header">
        <div class="tech-deepdive-badge">
          <span class="deepdive-pulse"></span>
          <span>&#9889;</span>
        </div>
        <div class="tech-deepdive-title">
          <h3>Technical Deep Dive</h3>
          <p>Architecture, Implementation & Details</p>
        </div>
      </div>
      <div class="tech-deepdive-content">
        ${td.sections.map((section, i) => `
          <div class="tech-section">
            <div class="tech-section-header">
              <span class="tech-section-number">${i + 1}</span>
              <span class="tech-section-title">${section.title}</span>
            </div>
            <div class="tech-section-content">
              ${section.content}
            </div>
          </div>
        `).join('')}
        ${td.references ? `
          <div class="tech-references">
            <h5>References</h5>
            <div class="reference-list">
              ${td.references.map(ref => `
                <a href="${ref.url}" target="_blank" rel="noopener" class="reference-link">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                       width="14" height="14">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15 3 21 3 21 9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                  </svg>
                  ${ref.title}
                </a>
              `).join('')}
            </div>
          </div>
        ` : ''}
      </div>
    </div>`;
}

export function initRenderer(data) {
  mainContent = document.getElementById('main');
  projectDetailView = document.getElementById('projectDetailView');
  projectDetail = document.getElementById('projectDetail');
  allProjectsFlat = getAllProjects(data);

  // back button
  document.getElementById('backBtn')?.addEventListener('click', hideProjectDetail);

  // keyboard: Escape closes detail (but not if lightbox is open)
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && projectDetailView?.classList.contains('active')) {
      if (isLightboxActive()) return;
      hideProjectDetail();
    }
  });
}
