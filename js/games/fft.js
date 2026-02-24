// ============================================================
// FFT Signal Processing Visualizer
// Sine waves add karo, freehand draw karo, FFT dekho real-time
// Filters lagao, audio suno — full signal processing demo
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, FFT chalao
export function initFFT() {
  const container = document.getElementById('fftContainer');
  if (!container) {
    console.warn('fftContainer nahi mila bhai, FFT demo skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const TIME_CANVAS_HEIGHT = 180;
  const FREQ_CANVAS_HEIGHT = 180;
  const SAMPLE_RATE = 512; // FFT ke liye samples
  const MAX_FREQ = 20; // Hz max display range
  const TWO_PI = Math.PI * 2;

  // --- State ---
  let canvasW = 0;
  let dpr = 1;

  // wave components — har ek {freq: Hz, amp: 0-1} hai
  let waves = [];

  // freehand drawing state
  let isDrawMode = false;
  let drawnSignal = null; // Float32Array — user ne draw kiya hua signal
  let isDrawing = false;

  // noise amount — 0 to 1
  let noiseAmount = 0;

  // filter state
  let activeFilter = null; // null, 'lowpass', 'highpass', 'bandpass'
  let filterCutoffLow = 5; // Hz
  let filterCutoffHigh = 15; // Hz

  // audio state
  let audioCtx = null;
  let isAudioPlaying = false;
  let audioOscillators = [];
  let audioGain = null;

  // animation
  let animationId = null;
  let isVisible = false;
  let phase = 0; // time domain animation phase

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // time domain canvas — waveform yahan dikhega
  const timeCanvas = document.createElement('canvas');
  timeCanvas.style.cssText = [
    'width:100%',
    'height:' + TIME_CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:8px 8px 0 0',
    'cursor:crosshair',
    'background:transparent',
  ].join(';');
  container.appendChild(timeCanvas);

  // separator line
  const sep = document.createElement('div');
  sep.style.cssText = 'height:1px;background:rgba(74,158,255,0.1);';
  container.appendChild(sep);

  // frequency domain canvas — FFT bars yahan dikhenge
  const freqCanvas = document.createElement('canvas');
  freqCanvas.style.cssText = [
    'width:100%',
    'height:' + FREQ_CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:0 0 8px 8px',
    'background:transparent',
  ].join(';');
  container.appendChild(freqCanvas);

  // wave tags container — active waves ki list
  const waveTagsDiv = document.createElement('div');
  waveTagsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:6px',
    'margin-top:8px',
    'min-height:24px',
  ].join(';');
  container.appendChild(waveTagsDiv);

  // controls container
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Helper: button banao ---
  function createButton(text, onClick, parent) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'background:rgba(74,158,255,0.1)',
      'color:#b0b0b0',
      'border:1px solid rgba(74,158,255,0.25)',
      'font-family:monospace',
      'transition:all 0.2s ease',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(74,158,255,0.25)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn.dataset.active) {
        btn.style.background = 'rgba(74,158,255,0.1)';
        btn.style.color = '#b0b0b0';
      }
    });
    btn.addEventListener('click', onClick);
    (parent || controlsDiv).appendChild(btn);
    return btn;
  }

  function setButtonActive(btn, active) {
    btn.dataset.active = active ? '1' : '';
    if (active) {
      btn.style.background = 'rgba(74,158,255,0.35)';
      btn.style.color = '#e0e0e0';
      btn.style.borderColor = 'rgba(74,158,255,0.5)';
    } else {
      btn.style.background = 'rgba(74,158,255,0.1)';
      btn.style.color = '#b0b0b0';
      btn.style.borderColor = 'rgba(74,158,255,0.25)';
    }
  }

  // --- Add Wave button + mini form ---
  const addWaveBtn = createButton('+ Add Wave', showAddWaveForm);

  // mini form — initially hidden
  const waveForm = document.createElement('div');
  waveForm.style.cssText = [
    'display:none',
    'align-items:center',
    'gap:8px',
    'padding:6px 10px',
    'background:rgba(74,158,255,0.05)',
    'border:1px solid rgba(74,158,255,0.15)',
    'border-radius:6px',
  ].join(';');
  controlsDiv.appendChild(waveForm);

  // frequency slider
  const freqSliderLabel = document.createElement('span');
  freqSliderLabel.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:monospace;';
  freqSliderLabel.textContent = 'Hz:';
  waveForm.appendChild(freqSliderLabel);

  const freqSlider = document.createElement('input');
  freqSlider.type = 'range';
  freqSlider.min = '1';
  freqSlider.max = '20';
  freqSlider.step = '0.5';
  freqSlider.value = '5';
  freqSlider.style.cssText = 'width:60px;height:4px;accent-color:rgba(74,158,255,0.8);cursor:pointer;';
  waveForm.appendChild(freqSlider);

  const freqValLabel = document.createElement('span');
  freqValLabel.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:monospace;min-width:28px;';
  freqValLabel.textContent = '5.0';
  waveForm.appendChild(freqValLabel);

  freqSlider.addEventListener('input', () => {
    freqValLabel.textContent = parseFloat(freqSlider.value).toFixed(1);
  });

  // amplitude slider
  const ampSliderLabel = document.createElement('span');
  ampSliderLabel.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:monospace;';
  ampSliderLabel.textContent = 'Amp:';
  waveForm.appendChild(ampSliderLabel);

  const ampSlider = document.createElement('input');
  ampSlider.type = 'range';
  ampSlider.min = '0.05';
  ampSlider.max = '1';
  ampSlider.step = '0.05';
  ampSlider.value = '0.5';
  ampSlider.style.cssText = 'width:50px;height:4px;accent-color:rgba(74,158,255,0.8);cursor:pointer;';
  waveForm.appendChild(ampSlider);

  const ampValLabel = document.createElement('span');
  ampValLabel.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:monospace;min-width:28px;';
  ampValLabel.textContent = '0.50';
  waveForm.appendChild(ampValLabel);

  ampSlider.addEventListener('input', () => {
    ampValLabel.textContent = parseFloat(ampSlider.value).toFixed(2);
  });

  // confirm button
  createButton('OK', () => {
    const freq = parseFloat(freqSlider.value);
    const amp = parseFloat(ampSlider.value);
    addWave(freq, amp);
    waveForm.style.display = 'none';
  }, waveForm);

  function showAddWaveForm() {
    waveForm.style.display = waveForm.style.display === 'none' ? 'flex' : 'none';
  }

  // --- Draw Mode toggle ---
  const drawBtn = createButton('Draw Mode', () => {
    isDrawMode = !isDrawMode;
    setButtonActive(drawBtn, isDrawMode);
    if (isDrawMode) {
      // draw mode on — clear drawn signal
      drawnSignal = new Float32Array(SAMPLE_RATE);
      timeCanvas.style.cursor = 'crosshair';
    } else {
      drawnSignal = null;
    }
  });

  // --- Noise slider ---
  const noiseWrapper = document.createElement('div');
  noiseWrapper.style.cssText = 'display:flex;align-items:center;gap:4px;';
  const noiseLabel = document.createElement('span');
  noiseLabel.style.cssText = 'color:#b0b0b0;font-size:12px;font-family:monospace;';
  noiseLabel.textContent = 'Noise:';
  noiseWrapper.appendChild(noiseLabel);

  const noiseSlider = document.createElement('input');
  noiseSlider.type = 'range';
  noiseSlider.min = '0';
  noiseSlider.max = '1';
  noiseSlider.step = '0.02';
  noiseSlider.value = '0';
  noiseSlider.style.cssText = 'width:60px;height:4px;accent-color:rgba(239,68,68,0.8);cursor:pointer;';
  noiseWrapper.appendChild(noiseSlider);

  const noiseValLabel = document.createElement('span');
  noiseValLabel.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:monospace;min-width:20px;';
  noiseValLabel.textContent = '0';
  noiseWrapper.appendChild(noiseValLabel);

  noiseSlider.addEventListener('input', () => {
    noiseAmount = parseFloat(noiseSlider.value);
    noiseValLabel.textContent = noiseAmount.toFixed(2);
  });
  controlsDiv.appendChild(noiseWrapper);

  // --- Filter buttons ---
  const filterSep = document.createElement('span');
  filterSep.style.cssText = 'color:rgba(74,158,255,0.2);font-size:14px;';
  filterSep.textContent = '|';
  controlsDiv.appendChild(filterSep);

  const lpBtn = createButton('Low Pass', () => toggleFilter('lowpass', lpBtn));
  const hpBtn = createButton('High Pass', () => toggleFilter('highpass', hpBtn));
  const bpBtn = createButton('Band Pass', () => toggleFilter('bandpass', bpBtn));
  const filterBtns = [lpBtn, hpBtn, bpBtn];

  function toggleFilter(type, btn) {
    if (activeFilter === type) {
      activeFilter = null;
      filterBtns.forEach(b => setButtonActive(b, false));
    } else {
      activeFilter = type;
      filterBtns.forEach(b => setButtonActive(b, false));
      setButtonActive(btn, true);
    }
  }

  // --- Listen toggle (Web Audio) ---
  const listenBtn = createButton('Listen', toggleAudio);

  function toggleAudio() {
    if (isAudioPlaying) {
      stopAudio();
    } else {
      startAudio();
    }
    setButtonActive(listenBtn, isAudioPlaying);
  }

  // --- Canvas sizing ---
  function resizeCanvases() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;

    // time canvas
    timeCanvas.width = containerWidth * dpr;
    timeCanvas.height = TIME_CANVAS_HEIGHT * dpr;
    timeCanvas.style.width = containerWidth + 'px';
    timeCanvas.style.height = TIME_CANVAS_HEIGHT + 'px';
    const tCtx = timeCanvas.getContext('2d');
    tCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // freq canvas
    freqCanvas.width = containerWidth * dpr;
    freqCanvas.height = FREQ_CANVAS_HEIGHT * dpr;
    freqCanvas.style.width = containerWidth + 'px';
    freqCanvas.style.height = FREQ_CANVAS_HEIGHT + 'px';
    const fCtx = freqCanvas.getContext('2d');
    fCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);

  // --- Wave management ---
  function addWave(freq, amp) {
    waves.push({ freq, amp });
    updateWaveTags();
    updateAudio();
  }

  function removeWave(index) {
    waves.splice(index, 1);
    updateWaveTags();
    updateAudio();
  }

  function updateWaveTags() {
    // wave tags re-render karo
    while (waveTagsDiv.firstChild) waveTagsDiv.removeChild(waveTagsDiv.firstChild);

    waves.forEach((w, i) => {
      const tag = document.createElement('div');
      tag.style.cssText = [
        'display:flex',
        'align-items:center',
        'gap:4px',
        'padding:3px 8px',
        'background:rgba(74,158,255,0.08)',
        'border:1px solid rgba(74,158,255,0.2)',
        'border-radius:12px',
        'font-size:11px',
        'font-family:monospace',
        'color:#b0b0b0',
      ].join(';');

      const text = document.createElement('span');
      text.textContent = w.freq.toFixed(1) + 'Hz \u00D7 ' + w.amp.toFixed(2);
      tag.appendChild(text);

      // remove button — chhota cross
      const removeBtn = document.createElement('span');
      removeBtn.textContent = '\u00D7';
      removeBtn.style.cssText = 'cursor:pointer;color:rgba(239,68,68,0.6);font-size:14px;margin-left:2px;';
      removeBtn.addEventListener('click', () => removeWave(i));
      removeBtn.addEventListener('mouseenter', () => { removeBtn.style.color = 'rgba(239,68,68,1)'; });
      removeBtn.addEventListener('mouseleave', () => { removeBtn.style.color = 'rgba(239,68,68,0.6)'; });
      tag.appendChild(removeBtn);

      waveTagsDiv.appendChild(tag);
    });

    if (waves.length === 0 && !isDrawMode) {
      const hint = document.createElement('span');
      hint.style.cssText = 'color:rgba(176,176,176,0.3);font-size:11px;font-family:monospace;';
      hint.textContent = 'add waves or enable draw mode';
      waveTagsDiv.appendChild(hint);
    }
  }
  updateWaveTags();

  // --- Signal generation ---
  // time domain signal generate karo — saari waves ka sum + noise
  function generateSignal(phaseOffset) {
    const signal = new Float32Array(SAMPLE_RATE);

    if (isDrawMode && drawnSignal) {
      // draw mode — user ka drawn signal use karo
      for (let i = 0; i < SAMPLE_RATE; i++) {
        signal[i] = drawnSignal[i];
      }
    } else {
      // sine waves ka sum
      for (let i = 0; i < SAMPLE_RATE; i++) {
        const t = i / SAMPLE_RATE; // 0 to 1 — ek second ka window
        let val = 0;
        for (const w of waves) {
          val += w.amp * Math.sin(TWO_PI * w.freq * t + phaseOffset * w.freq);
        }
        signal[i] = val;
      }
    }

    // noise add karo — Gaussian approximation (Box-Muller light version)
    if (noiseAmount > 0) {
      for (let i = 0; i < SAMPLE_RATE; i++) {
        // simple Gaussian noise — 6 random numbers ka average (central limit theorem)
        let noise = 0;
        for (let j = 0; j < 6; j++) noise += Math.random();
        noise = (noise - 3) / 3; // roughly N(0,1)
        signal[i] += noise * noiseAmount;
      }
    }

    return signal;
  }

  // --- FFT implementation ---
  // Cooley-Tukey radix-2 DIT FFT — zero dependency, pure JS
  function fft(signal) {
    const N = signal.length;
    // pad to nearest power of 2 agar zarurat ho
    let n = 1;
    while (n < N) n *= 2;

    // real aur imaginary arrays
    const real = new Float32Array(n);
    const imag = new Float32Array(n);
    for (let i = 0; i < N; i++) real[i] = signal[i];

    // bit-reversal permutation
    for (let i = 0; i < n; i++) {
      const j = bitReverse(i, Math.log2(n));
      if (i < j) {
        // swap real
        let tmp = real[i];
        real[i] = real[j];
        real[j] = tmp;
        // swap imag
        tmp = imag[i];
        imag[i] = imag[j];
        imag[j] = tmp;
      }
    }

    // butterfly computation — bottom-up
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;
      const angleStep = -TWO_PI / size;

      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < halfSize; j++) {
          const angle = angleStep * j;
          const cosA = Math.cos(angle);
          const sinA = Math.sin(angle);

          const evenIdx = i + j;
          const oddIdx = i + j + halfSize;

          const tReal = cosA * real[oddIdx] - sinA * imag[oddIdx];
          const tImag = sinA * real[oddIdx] + cosA * imag[oddIdx];

          real[oddIdx] = real[evenIdx] - tReal;
          imag[oddIdx] = imag[evenIdx] - tImag;
          real[evenIdx] += tReal;
          imag[evenIdx] += tImag;
        }
      }
    }

    // magnitude spectrum calculate karo
    const magnitudes = new Float32Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      magnitudes[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]) / (n / 2);
    }

    return { magnitudes, real, imag, n };
  }

  // bit reverse — FFT ke liye zaroori hai
  function bitReverse(x, bits) {
    let result = 0;
    for (let i = 0; i < bits; i++) {
      result = (result << 1) | (x & 1);
      x >>= 1;
    }
    return result;
  }

  // --- Filtering ---
  // frequency domain mein filter apply karo
  function applyFilter(magnitudes) {
    if (!activeFilter) return magnitudes;

    const filtered = new Float32Array(magnitudes.length);
    const freqResolution = SAMPLE_RATE / (magnitudes.length * 2); // Hz per bin

    for (let i = 0; i < magnitudes.length; i++) {
      const freq = i * freqResolution;

      switch (activeFilter) {
        case 'lowpass':
          // low pass — filterCutoffLow ke neeche pass, upar cut
          filtered[i] = freq <= filterCutoffLow ? magnitudes[i] : magnitudes[i] * 0.05;
          break;
        case 'highpass':
          // high pass — filterCutoffHigh ke upar pass, neeche cut
          filtered[i] = freq >= filterCutoffHigh * 0.5 ? magnitudes[i] : magnitudes[i] * 0.05;
          break;
        case 'bandpass':
          // band pass — cutoffLow to cutoffHigh ke beech pass
          filtered[i] = (freq >= filterCutoffLow * 0.5 && freq <= filterCutoffHigh) ? magnitudes[i] : magnitudes[i] * 0.05;
          break;
      }
    }

    return filtered;
  }

  // filtered signal reconstruct karo frequency domain se
  function reconstructFilteredSignal(fftResult) {
    if (!activeFilter) return null;

    const { real, imag, n } = fftResult;
    const freqResolution = SAMPLE_RATE / n;

    // filter apply karo frequency domain mein
    const filteredReal = new Float32Array(n);
    const filteredImag = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const freq = (i <= n / 2) ? i * freqResolution : (n - i) * freqResolution;
      let pass = 1;

      switch (activeFilter) {
        case 'lowpass':
          pass = freq <= filterCutoffLow ? 1 : 0.05;
          break;
        case 'highpass':
          pass = freq >= filterCutoffHigh * 0.5 ? 1 : 0.05;
          break;
        case 'bandpass':
          pass = (freq >= filterCutoffLow * 0.5 && freq <= filterCutoffHigh) ? 1 : 0.05;
          break;
      }

      filteredReal[i] = real[i] * pass;
      filteredImag[i] = imag[i] * pass;
    }

    // inverse FFT — simple version: conjugate, FFT, conjugate, scale
    // conjugate
    for (let i = 0; i < n; i++) filteredImag[i] = -filteredImag[i];

    // bit-reversal
    for (let i = 0; i < n; i++) {
      const j = bitReverse(i, Math.log2(n));
      if (i < j) {
        let tmp = filteredReal[i];
        filteredReal[i] = filteredReal[j];
        filteredReal[j] = tmp;
        tmp = filteredImag[i];
        filteredImag[i] = filteredImag[j];
        filteredImag[j] = tmp;
      }
    }

    // butterfly
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;
      const angleStep = -TWO_PI / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < halfSize; j++) {
          const angle = angleStep * j;
          const cosA = Math.cos(angle);
          const sinA = Math.sin(angle);
          const evenIdx = i + j;
          const oddIdx = i + j + halfSize;
          const tReal = cosA * filteredReal[oddIdx] - sinA * filteredImag[oddIdx];
          const tImag = sinA * filteredReal[oddIdx] + cosA * filteredImag[oddIdx];
          filteredReal[oddIdx] = filteredReal[evenIdx] - tReal;
          filteredImag[oddIdx] = filteredImag[evenIdx] - tImag;
          filteredReal[evenIdx] += tReal;
          filteredImag[evenIdx] += tImag;
        }
      }
    }

    // conjugate again and scale
    const result = new Float32Array(SAMPLE_RATE);
    for (let i = 0; i < SAMPLE_RATE; i++) {
      result[i] = filteredReal[i] / n;
    }

    return result;
  }

  // --- Freehand drawing on time canvas ---
  function getTimeCanvasPos(e) {
    const rect = timeCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  function drawAtPos(pos) {
    if (!drawnSignal) return;
    // x position ko sample index mein convert karo
    const sampleIdx = Math.floor((pos.x / canvasW) * SAMPLE_RATE);
    // y position ko -1 to 1 amplitude mein convert karo
    const amplitude = 1 - (pos.y / TIME_CANVAS_HEIGHT) * 2;

    // surrounding samples bhi fill karo — smooth drawing ke liye
    const brushWidth = Math.max(1, Math.floor(SAMPLE_RATE / canvasW * 3));
    for (let i = sampleIdx - brushWidth; i <= sampleIdx + brushWidth; i++) {
      if (i >= 0 && i < SAMPLE_RATE) {
        const dist = Math.abs(i - sampleIdx) / brushWidth;
        const weight = 1 - dist;
        drawnSignal[i] = drawnSignal[i] * (1 - weight) + amplitude * weight;
      }
    }
  }

  timeCanvas.addEventListener('mousedown', (e) => {
    if (!isDrawMode) return;
    isDrawing = true;
    drawAtPos(getTimeCanvasPos(e));
  });

  timeCanvas.addEventListener('mousemove', (e) => {
    if (!isDrawing || !isDrawMode) return;
    drawAtPos(getTimeCanvasPos(e));
  });

  timeCanvas.addEventListener('mouseup', () => { isDrawing = false; });
  timeCanvas.addEventListener('mouseleave', () => { isDrawing = false; });

  // touch support for drawing
  timeCanvas.addEventListener('touchstart', (e) => {
    if (!isDrawMode) return;
    e.preventDefault();
    isDrawing = true;
    drawAtPos(getTimeCanvasPos(e));
  }, { passive: false });

  timeCanvas.addEventListener('touchmove', (e) => {
    if (!isDrawing || !isDrawMode) return;
    e.preventDefault();
    drawAtPos(getTimeCanvasPos(e));
  }, { passive: false });

  timeCanvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    isDrawing = false;
  }, { passive: false });

  // --- Drawing functions ---
  function drawTimeCanvas(signal, filteredSignal) {
    const ctx = timeCanvas.getContext('2d');
    const w = canvasW;
    const h = TIME_CANVAS_HEIGHT;
    ctx.clearRect(0, 0, w, h);

    // center line — zero axis
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.strokeStyle = 'rgba(74,158,255,0.1)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // grid lines — ±0.5, ±1.0
    [0.25, 0.75].forEach(yFrac => {
      ctx.beginPath();
      ctx.moveTo(0, h * yFrac);
      ctx.lineTo(w, h * yFrac);
      ctx.strokeStyle = 'rgba(74,158,255,0.05)';
      ctx.stroke();
    });

    if (signal.length === 0) {
      ctx.font = '13px monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('add waves or enable draw mode', w / 2, h / 2 + 5);
      return;
    }

    // amplitude range dhundho — auto-scale
    let maxAmp = 0;
    for (let i = 0; i < signal.length; i++) {
      if (Math.abs(signal[i]) > maxAmp) maxAmp = Math.abs(signal[i]);
    }
    maxAmp = Math.max(maxAmp, 0.1); // minimum range rakh
    const scale = (h / 2 - 10) / maxAmp;

    // agar filter active hai toh original signal faint dikhao
    if (filteredSignal && activeFilter) {
      ctx.beginPath();
      for (let i = 0; i < SAMPLE_RATE; i++) {
        const x = (i / SAMPLE_RATE) * w;
        const y = h / 2 - signal[i] * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(74,158,255,0.15)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // main signal (ya filtered signal) draw karo
    const drawSignal = (filteredSignal && activeFilter) ? filteredSignal : signal;
    ctx.beginPath();
    for (let i = 0; i < SAMPLE_RATE; i++) {
      const x = (i / SAMPLE_RATE) * w;
      const y = h / 2 - drawSignal[i] * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    // glow effect
    ctx.shadowColor = 'rgba(0,200,255,0.3)';
    ctx.shadowBlur = 6;
    ctx.strokeStyle = activeFilter ? 'rgba(16,185,129,0.8)' : 'rgba(0,200,255,0.7)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // noise visible karo — red tint agar noise hai
    if (noiseAmount > 0) {
      ctx.font = '9px monospace';
      ctx.fillStyle = 'rgba(239,68,68,' + (0.3 + noiseAmount * 0.4) + ')';
      ctx.textAlign = 'right';
      ctx.fillText('noise: ' + (noiseAmount * 100).toFixed(0) + '%', w - 10, 14);
    }

    // label
    ctx.font = '9px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'left';
    ctx.fillText('TIME DOMAIN', 8, 14);

    // draw mode indicator
    if (isDrawMode) {
      ctx.font = '10px monospace';
      ctx.fillStyle = 'rgba(249,158,11,0.5)';
      ctx.textAlign = 'right';
      ctx.fillText('DRAW MODE', w - 10, h - 8);
    }
  }

  function drawFreqCanvas(magnitudes, filteredMagnitudes) {
    const ctx = freqCanvas.getContext('2d');
    const w = canvasW;
    const h = FREQ_CANVAS_HEIGHT;
    ctx.clearRect(0, 0, w, h);

    // label
    ctx.font = '9px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'left';
    ctx.fillText('FREQUENCY DOMAIN (FFT)', 8, 14);

    if (!magnitudes || magnitudes.length === 0) return;

    const padding = { top: 25, bottom: 20, left: 10, right: 10 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    // sirf MAX_FREQ Hz tak dikhao
    const freqResolution = SAMPLE_RATE / (magnitudes.length * 2);
    const maxBin = Math.ceil(MAX_FREQ / freqResolution);
    const displayBins = Math.min(maxBin, magnitudes.length);

    // max magnitude for scaling
    let maxMag = 0;
    for (let i = 1; i < displayBins; i++) {
      if (magnitudes[i] > maxMag) maxMag = magnitudes[i];
    }
    maxMag = Math.max(maxMag, 0.01);

    // frequency axis labels
    ctx.font = '9px monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.3)';
    ctx.textAlign = 'center';
    for (let f = 0; f <= MAX_FREQ; f += 5) {
      const x = padding.left + (f / MAX_FREQ) * plotW;
      ctx.fillText(f + 'Hz', x, h - 4);

      // vertical grid line
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + plotH);
      ctx.strokeStyle = 'rgba(74,158,255,0.05)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // bars draw karo — gradient blue to green
    const barWidth = Math.max(2, plotW / displayBins - 1);
    const drawMags = (filteredMagnitudes && activeFilter) ? filteredMagnitudes : magnitudes;

    // agar filter hai toh original bhi faint dikhao
    if (filteredMagnitudes && activeFilter) {
      for (let i = 1; i < displayBins; i++) {
        const x = padding.left + (i / displayBins) * plotW;
        const barH = (magnitudes[i] / maxMag) * plotH;
        ctx.fillStyle = 'rgba(74,158,255,0.08)';
        ctx.fillRect(x, padding.top + plotH - barH, barWidth, barH);
      }
    }

    // main bars
    for (let i = 1; i < displayBins; i++) {
      const x = padding.left + (i / displayBins) * plotW;
      const barH = (drawMags[i] / maxMag) * plotH;

      // gradient color — low freq = blue, high freq = green
      const t = i / displayBins;
      const r = Math.round(30 + 20 * t);
      const g = Math.round(100 + 155 * t);
      const b = Math.round(255 - 120 * t);

      // bar draw karo
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.7)';
      ctx.fillRect(x, padding.top + plotH - barH, barWidth, barH);

      // glow top pe
      if (barH > 5) {
        ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.3)';
        ctx.fillRect(x - 1, padding.top + plotH - barH - 2, barWidth + 2, 4);
      }
    }

    // peaks detect karo aur label lagao
    const peaks = findPeaks(drawMags, displayBins, freqResolution);
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    for (const peak of peaks) {
      const x = padding.left + (peak.bin / displayBins) * plotW;
      const barH = (peak.magnitude / maxMag) * plotH;
      const y = padding.top + plotH - barH - 8;

      ctx.fillStyle = 'rgba(255,255,100,0.7)';
      ctx.fillText(peak.freq.toFixed(1) + 'Hz', x + barWidth / 2, y);
    }

    // filter region dikhao — highlight band
    if (activeFilter) {
      drawFilterRegion(ctx, padding, plotW, plotH);
    }
  }

  function findPeaks(magnitudes, displayBins, freqResolution) {
    // simple peak detection — local maxima jo threshold se upar hain
    const peaks = [];
    const threshold = 0.05; // minimum magnitude for peak

    for (let i = 2; i < displayBins - 1; i++) {
      if (magnitudes[i] > magnitudes[i - 1] &&
        magnitudes[i] > magnitudes[i + 1] &&
        magnitudes[i] > threshold) {
        peaks.push({
          bin: i,
          freq: i * freqResolution,
          magnitude: magnitudes[i],
        });
      }
    }

    // top 5 peaks tak limit karo — zyada labels cluttered lagenge
    peaks.sort((a, b) => b.magnitude - a.magnitude);
    return peaks.slice(0, 5);
  }

  function drawFilterRegion(ctx, padding, plotW, plotH) {
    // filter ki pass band region highlight karo
    let lowFreq = 0, highFreq = MAX_FREQ;

    switch (activeFilter) {
      case 'lowpass':
        lowFreq = 0;
        highFreq = filterCutoffLow;
        break;
      case 'highpass':
        lowFreq = filterCutoffHigh * 0.5;
        highFreq = MAX_FREQ;
        break;
      case 'bandpass':
        lowFreq = filterCutoffLow * 0.5;
        highFreq = filterCutoffHigh;
        break;
    }

    const x1 = padding.left + (lowFreq / MAX_FREQ) * plotW;
    const x2 = padding.left + (highFreq / MAX_FREQ) * plotW;

    ctx.fillStyle = 'rgba(16,185,129,0.06)';
    ctx.fillRect(x1, padding.top, x2 - x1, plotH);

    ctx.strokeStyle = 'rgba(16,185,129,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x1, padding.top);
    ctx.lineTo(x1, padding.top + plotH);
    ctx.moveTo(x2, padding.top);
    ctx.lineTo(x2, padding.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // label
    ctx.font = '9px monospace';
    ctx.fillStyle = 'rgba(16,185,129,0.5)';
    ctx.textAlign = 'center';
    ctx.fillText(activeFilter.toUpperCase(), (x1 + x2) / 2, padding.top + plotH + 12);
  }

  // --- Web Audio API ---
  function startAudio() {
    if (waves.length === 0) return;

    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioGain = audioCtx.createGain();
      audioGain.gain.value = 0.15; // chhota volume — surprise nahi chahiye
      audioGain.connect(audioCtx.destination);

      audioOscillators = [];
      for (const w of waves) {
        const osc = audioCtx.createOscillator();
        const oscGain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.value = w.freq * 50; // scale up — 1Hz sunai nahi dega
        oscGain.gain.value = w.amp * 0.3;
        osc.connect(oscGain);
        oscGain.connect(audioGain);
        osc.start();
        audioOscillators.push({ osc, gain: oscGain });
      }

      isAudioPlaying = true;
    } catch (e) {
      console.warn('Web Audio nahi chala:', e);
    }
  }

  function stopAudio() {
    if (audioOscillators.length > 0) {
      audioOscillators.forEach(o => {
        try { o.osc.stop(); } catch (_) { /* already stopped */ }
      });
      audioOscillators = [];
    }
    if (audioCtx) {
      audioCtx.close().catch(() => {});
      audioCtx = null;
    }
    isAudioPlaying = false;
  }

  function updateAudio() {
    // waves change hue toh audio restart karo
    if (isAudioPlaying) {
      stopAudio();
      startAudio();
      setButtonActive(listenBtn, isAudioPlaying);
    }
  }

  // --- Animation loop ---
  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) return;

    // phase advance karo — waveform animate hoga
    phase += 0.03;

    // signal generate karo
    const signal = generateSignal(phase);

    // FFT compute karo
    const fftResult = fft(signal);
    const magnitudes = fftResult.magnitudes;

    // filter apply karo (agar active hai)
    const filteredMagnitudes = activeFilter ? applyFilter(magnitudes) : null;
    const filteredSignal = activeFilter ? reconstructFilteredSignal(fftResult) : null;

    // draw both canvases
    drawTimeCanvas(signal, filteredSignal);
    drawFreqCanvas(magnitudes, filteredMagnitudes);

    animationId = requestAnimationFrame(animate);
  }

  // --- IntersectionObserver ---
  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    animationId = requestAnimationFrame(animate);
  }

  function stopAnimation() {
    isVisible = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
    // audio bhi band karo jab visible nahi hai
    if (isAudioPlaying) {
      stopAudio();
      setButtonActive(listenBtn, false);
    }
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          startAnimation();
        } else {
          stopAnimation();
        }
      });
    },
    { threshold: 0.1 }
  );

  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) animate(); });

  // tab switch pe pause
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });
}
