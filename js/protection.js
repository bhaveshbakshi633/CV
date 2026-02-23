// protection.js — password-based content protection
// recruiter access system for confidential projects

const DEFAULT_PROTECTED_IDS = ['naamika', 'xr-teleop', 'sim2real-deploy'];
const PROTECTED_HASH = '9f2feb701519cf3605ae8075e865857b97b17927bbd5045e33e3fb498a43b040';

let PROTECTED_PROJECT_IDS = JSON.parse(localStorage.getItem('protectedProjectIds')) || [...DEFAULT_PROTECTED_IDS];
let protectedUnlocked = false;
let reRenderCallback = null;

async function checkPassword(input) {
  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('') === PROTECTED_HASH;
}

export function isProjectProtected(projectId) {
  return PROTECTED_PROJECT_IDS.includes(projectId) && !protectedUnlocked;
}

export function isUnlocked() {
  return protectedUnlocked;
}

function saveProtectedIds() {
  localStorage.setItem('protectedProjectIds', JSON.stringify(PROTECTED_PROJECT_IDS));
}

export function showPasswordModal(onSuccess) {
  const existing = document.getElementById('passwordModal');
  if (existing) existing.remove();

  const modal = document.createElement('div');
  modal.id = 'passwordModal';
  modal.className = 'pw-modal-overlay';
  modal.innerHTML = `
    <div class="pw-modal-card">
      <div class="pw-modal-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
          <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
        </svg>
      </div>
      <h3>Protected Content</h3>
      <p class="pw-subtitle">This project contains confidential material.<br>Enter the access password to view.</p>
      <input type="password" id="protectedPwInput" placeholder="Enter password" autocomplete="off">
      <div id="pwError" class="pw-error">Incorrect password</div>
      <div class="pw-modal-actions">
        <button id="pwCancel" class="pw-btn-cancel">Cancel</button>
        <button id="pwSubmit" class="pw-btn-unlock">Unlock</button>
      </div>
    </div>`;
  document.body.appendChild(modal);

  const input = document.getElementById('protectedPwInput');
  const error = document.getElementById('pwError');

  input.focus();

  async function tryUnlock() {
    if (await checkPassword(input.value)) {
      protectedUnlocked = true;
      modal.remove();
      if (reRenderCallback) reRenderCallback();
      if (onSuccess) onSuccess();
    } else {
      error.style.display = 'block';
      input.value = '';
      input.focus();
    }
  }

  document.getElementById('pwSubmit').addEventListener('click', tryUnlock);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') tryUnlock(); });
  document.getElementById('pwCancel').addEventListener('click', () => modal.remove());
  modal.addEventListener('click', e => { if (e.target === modal) modal.remove(); });
}

function showProtectionSettings(allProjects) {
  if (!protectedUnlocked) {
    showPasswordModal(() => showProtectionSettings(allProjects));
    return;
  }

  const existing = document.getElementById('protectionSettingsModal');
  if (existing) existing.remove();

  const modal = document.createElement('div');
  modal.id = 'protectionSettingsModal';
  modal.className = 'pw-modal-overlay';
  modal.innerHTML = `
    <div class="pw-modal-card" style="max-width:460px;text-align:left">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
        <div class="pw-modal-icon" style="margin:0;width:36px;height:36px;flex-shrink:0">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:18px;height:18px">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
          </svg>
        </div>
        <h3 style="margin:0;font-size:1.05rem">Protection Settings</h3>
      </div>
      <p style="color:var(--text-secondary);font-size:0.8rem;margin:0 0 16px">Select which projects require password to view:</p>
      <div id="protectionCheckboxes" style="max-height:350px;overflow-y:auto;margin-bottom:16px">
        ${allProjects.map(p => `
          <label class="prot-setting-label">
            <input type="checkbox" value="${p.id}" ${PROTECTED_PROJECT_IDS.includes(p.id) ? 'checked' : ''}>
            <span class="prot-setting-name">${p.shortTitle || p.title}</span>
            <span class="prot-setting-cap">${p.capability}</span>
          </label>
        `).join('')}
      </div>
      <div class="pw-modal-actions">
        <button id="protSettingsCancel" class="pw-btn-cancel">Cancel</button>
        <button id="protSettingsSave" class="pw-btn-unlock">Save</button>
      </div>
    </div>`;
  document.body.appendChild(modal);

  document.getElementById('protSettingsSave').addEventListener('click', () => {
    const checked = modal.querySelectorAll('input[type="checkbox"]:checked');
    PROTECTED_PROJECT_IDS = Array.from(checked).map(cb => cb.value);
    saveProtectedIds();
    modal.remove();
    if (reRenderCallback) reRenderCallback();
  });

  document.getElementById('protSettingsCancel').addEventListener('click', () => modal.remove());
  modal.addEventListener('click', e => { if (e.target === modal) modal.remove(); });
}

export function initProtection(allProjects, onReRender) {
  reRenderCallback = onReRender;

  const settingsBtn = document.getElementById('protectionSettingsBtn');
  if (settingsBtn) {
    settingsBtn.addEventListener('click', () => showProtectionSettings(allProjects));
  }
}
