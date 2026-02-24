// ============================================================
// Turing Machine Simulator — Visual tape-based Turing machine
// Animated head movement, state transitions, preset programs
// Tape scroll hota hai, head center mein rehta hai — clean look
// ============================================================

// yahi entry point hai — container pakdo, canvas banao, Turing machine chalu karo
export function initTuringMachine() {
  const container = document.getElementById('turingMachineContainer');
  if (!container) {
    console.warn('turingMachineContainer nahi mila bhai, Turing machine skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#4a9eff';              // blue accent — AI category
  const ACCENT_DIM = 'rgba(74,158,255,'; // prefix for alpha variants
  const HEAD_COLOR = '#00e5ff';           // cyan — head highlight
  const HALT_COLOR = '#22c55e';           // green — halted indicator
  const BG = '#0a0a0a';
  const FONT = "'JetBrains Mono', monospace";

  // tape cell dimensions (CSS pixels)
  const CELL_SIZE = 40;
  const CELL_GAP = 3;
  const TAPE_Y = 160;    // tape ki vertical position canvas mein
  const HEAD_ARROW_H = 18;

  // --- Turing Machine Programs ---
  // har rule: [currentState, readSymbol] => [writeSymbol, direction, nextState]
  // direction: 'L' ya 'R'
  // state 'HALT' pe machine ruk jaati hai
  const PROGRAMS = {
    'Binary Counter': {
      description: 'Binary number ko increment karta hai',
      initialTape: ['1', '0', '1', '1'],
      startPos: 3,    // rightmost bit se shuru
      startState: 'q0',
      rules: {
        // q0: rightmost bit dhundho, right jaao
        'q0,1': ['0', 'L', 'q0'],   // 1 mila toh 0 karo, carry left
        'q0,0': ['1', 'R', 'HALT'], // 0 mila toh 1 karo, done
        'q0,_': ['1', 'R', 'HALT'], // blank mila toh 1 likh do — overflow
      },
    },
    'Busy Beaver 3': {
      description: '3-state busy beaver — 6 ones likhta hai before halting',
      initialTape: [],
      startPos: 0,
      startState: 'A',
      rules: {
        // proven optimal 3-state busy beaver: 6 ones, 14 steps
        'A,_': ['1', 'R', 'B'],
        'A,1': ['1', 'L', 'C'],
        'B,_': ['1', 'L', 'A'],
        'B,1': ['1', 'R', 'B'],
        'C,_': ['1', 'L', 'B'],
        'C,1': ['1', 'R', 'HALT'],
      },
    },
    'Busy Beaver 4': {
      description: '4-state busy beaver — 13 ones likhta hai before halting',
      initialTape: [],
      startPos: 0,
      startState: 'A',
      rules: {
        // proven optimal 4-state busy beaver: 13 ones, 107 steps
        'A,_': ['1', 'R', 'B'],
        'A,1': ['1', 'L', 'B'],
        'B,_': ['1', 'L', 'A'],
        'B,1': ['0', 'L', 'C'],
        'C,_': ['1', 'R', 'HALT'],
        'C,1': ['1', 'L', 'D'],
        'D,_': ['1', 'R', 'D'],
        'D,1': ['0', 'R', 'A'],
      },
    },
    'Palindrome': {
      description: 'Binary string palindrome hai ya nahi check karta hai',
      // example: '1011' is NOT a palindrome
      initialTape: ['1', '0', '1', '0', '1'],
      startPos: 0,
      startState: 'q0',
      rules: {
        // q0: leftmost symbol padho
        'q0,0': ['_', 'R', 'q1'],     // 0 tha — yaad rakh, right end pe match karo
        'q0,1': ['_', 'R', 'q3'],     // 1 tha — yaad rakh, right end pe match karo
        'q0,_': ['Y', 'R', 'HALT'],   // blank/empty — palindrome hai (sab match ho gaye)

        // q1: right end pe jaao (0 match karne ke liye)
        'q1,0': ['0', 'R', 'q1'],
        'q1,1': ['1', 'R', 'q1'],
        'q1,_': ['_', 'L', 'q2'],     // blank mila — ab left jaao ek step

        // q2: rightmost symbol check karo — 0 hona chahiye
        'q2,0': ['_', 'L', 'q5'],     // match! wapas left jaao
        'q2,1': ['N', 'R', 'HALT'],   // mismatch — not palindrome
        'q2,_': ['Y', 'R', 'HALT'],   // sab khatam — palindrome hai

        // q3: right end pe jaao (1 match karne ke liye)
        'q3,0': ['0', 'R', 'q3'],
        'q3,1': ['1', 'R', 'q3'],
        'q3,_': ['_', 'L', 'q4'],     // blank mila — left jaao

        // q4: rightmost symbol check karo — 1 hona chahiye
        'q4,1': ['_', 'L', 'q5'],     // match! wapas left jaao
        'q4,0': ['N', 'R', 'HALT'],   // mismatch
        'q4,_': ['Y', 'R', 'HALT'],   // sab khatam — palindrome

        // q5: wapas left end pe jaao
        'q5,0': ['0', 'L', 'q5'],
        'q5,1': ['1', 'L', 'q5'],
        'q5,_': ['_', 'R', 'q0'],     // left end pe pahunche — fir se start
      },
    },
  };

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let isVisible = false;
  let animationId = null;

  // turing machine state
  let tape = {};             // sparse tape — index => symbol
  let headPos = 0;           // head ki current position
  let headDisplayX = 0;      // animated X position (lerp ke liye)
  let currentState = 'q0';
  let currentProgram = 'Binary Counter';
  let stepCount = 0;
  let halted = false;
  let running = false;
  let stepSpeed = 200;       // ms per step

  // animation state
  let lastStepTime = 0;
  let animPhase = 'idle';    // 'idle', 'read', 'write', 'move', 'stateChange'
  let animProgress = 0;      // 0 to 1
  let pendingWrite = null;   // kya likhna hai animation ke baad
  let pendingMove = 0;       // kitna move karna hai
  let pendingState = '';      // next state
  let highlightedRule = '';   // currently active rule key
  let tapeOffset = 0;        // smooth scroll offset
  let targetTapeOffset = 0;
  let headGlow = 0;          // glow pulse animation

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — tape aur head yahan dikhega
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid ' + ACCENT_DIM + '0.2)',
    'border-radius:8px',
    'background:' + BG,
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls row 1: program selector + main buttons ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:8px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // program dropdown
  const programSelect = document.createElement('select');
  programSelect.style.cssText = [
    'padding:5px 8px',
    'background:' + ACCENT_DIM + '0.08)',
    'color:#e2e8f0',
    'border:1px solid ' + ACCENT_DIM + '0.25)',
    'border-radius:6px',
    'font-family:' + FONT,
    'font-size:11px',
    'cursor:pointer',
    'outline:none',
  ].join(';');
  for (const name of Object.keys(PROGRAMS)) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    if (name === currentProgram) opt.selected = true;
    programSelect.appendChild(opt);
  }
  programSelect.addEventListener('change', () => {
    currentProgram = programSelect.value;
    resetMachine();
  });
  controlsDiv.appendChild(programSelect);

  // step button
  const stepBtn = makeButton('Step', controlsDiv, () => {
    if (halted) return;
    if (running) {
      running = false;
      runBtn.textContent = '\u25B6 Run';
    }
    executeStep();
  });

  // run/pause button
  const runBtn = makeButton('\u25B6 Run', controlsDiv, () => {
    if (halted) return;
    running = !running;
    runBtn.textContent = running ? '\u23F8 Pause' : '\u25B6 Run';
    if (running) {
      lastStepTime = performance.now();
    }
  });

  // reset button
  makeButton('\u21BB Reset', controlsDiv, () => {
    running = false;
    runBtn.textContent = '\u25B6 Run';
    resetMachine();
  });

  // step counter
  const stepLabel = document.createElement('span');
  stepLabel.style.cssText = [
    'font-family:' + FONT,
    'font-size:11px',
    'color:#94a3b8',
    'margin-left:auto',
  ].join(';');
  controlsDiv.appendChild(stepLabel);

  // --- Controls row 2: speed slider ---
  const speedDiv = document.createElement('div');
  speedDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:6px',
    'align-items:center',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#94a3b8',
  ].join(';');
  container.appendChild(speedDiv);

  const speedLabel = document.createElement('span');
  speedLabel.textContent = 'Speed: 200ms';
  speedLabel.style.cssText = 'min-width:100px;';
  speedDiv.appendChild(speedLabel);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '50';
  speedSlider.max = '500';
  speedSlider.value = '200';
  speedSlider.style.cssText = [
    'width:140px',
    'height:4px',
    'accent-color:' + ACCENT,
    'cursor:pointer',
  ].join(';');
  speedSlider.addEventListener('input', () => {
    stepSpeed = parseInt(speedSlider.value);
    speedLabel.textContent = 'Speed: ' + stepSpeed + 'ms';
  });
  speedDiv.appendChild(speedSlider);

  // description label
  const descLabel = document.createElement('span');
  descLabel.style.cssText = 'color:#64748b;margin-left:auto;';
  speedDiv.appendChild(descLabel);

  // --- Button helper ---
  function makeButton(text, parent, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:5px 14px',
      'border:1px solid ' + ACCENT_DIM + '0.25)',
      'border-radius:6px',
      'background:' + ACCENT_DIM + '0.08)',
      'color:#e2e8f0',
      'font-family:' + FONT,
      'font-size:11px',
      'cursor:pointer',
      'transition:all 0.15s ease',
      'outline:none',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = ACCENT_DIM + '0.2)';
      btn.style.borderColor = ACCENT;
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = ACCENT_DIM + '0.08)';
      btn.style.borderColor = ACCENT_DIM + '0.25)';
    });
    btn.addEventListener('click', onClick);
    parent.appendChild(btn);
    return btn;
  }

  // ============================================================
  // TURING MACHINE LOGIC
  // ============================================================

  // tape se symbol padho — blank '_' agar kuch nahi hai
  function readTape(pos) {
    return tape[pos] || '_';
  }

  // tape pe symbol likho
  function writeTape(pos, symbol) {
    if (symbol === '_') {
      delete tape[pos];
    } else {
      tape[pos] = symbol;
    }
  }

  // ek step execute karo — read, write, move, state change
  function executeStep() {
    if (halted) return;

    const readSym = readTape(headPos);
    const ruleKey = currentState + ',' + readSym;
    const rule = PROGRAMS[currentProgram].rules[ruleKey];

    if (!rule) {
      // koi rule nahi mila — machine halt ho gayi (undefined transition)
      halted = true;
      highlightedRule = '';
      updateStepLabel();
      return;
    }

    const [writeSym, dir, nextState] = rule;
    highlightedRule = ruleKey;

    // tape pe likho
    writeTape(headPos, writeSym);

    // head move karo
    if (dir === 'L') {
      headPos--;
    } else if (dir === 'R') {
      headPos++;
    }

    // state change
    currentState = nextState;
    stepCount++;

    // halt check
    if (currentState === 'HALT') {
      halted = true;
    }

    updateStepLabel();
  }

  // machine reset karo — selected program ke initial state mein
  function resetMachine() {
    const prog = PROGRAMS[currentProgram];
    tape = {};
    headPos = prog.startPos;
    headDisplayX = headPos;
    currentState = prog.startState;
    stepCount = 0;
    halted = false;
    highlightedRule = '';
    tapeOffset = headPos;
    targetTapeOffset = headPos;

    // initial tape load karo
    for (let i = 0; i < prog.initialTape.length; i++) {
      if (prog.initialTape[i] !== '_') {
        tape[i] = prog.initialTape[i];
      }
    }

    descLabel.textContent = prog.description;
    updateStepLabel();
  }

  function updateStepLabel() {
    if (halted) {
      // check karo palindrome result
      const result = readTape(headPos);
      let haltMsg = 'HALTED';
      if (currentProgram === 'Palindrome') {
        // 'Y' means palindrome, 'N' means not
        // scan tape for Y or N
        let found = '';
        for (const k of Object.keys(tape)) {
          if (tape[k] === 'Y') found = 'YES';
          if (tape[k] === 'N') found = 'NO';
        }
        if (found) haltMsg += ' \u2014 ' + found;
      }
      stepLabel.textContent = 'Steps: ' + stepCount + '  |  ' + haltMsg;
      stepLabel.style.color = HALT_COLOR;
    } else {
      stepLabel.textContent = 'Steps: ' + stepCount + '  |  State: ' + currentState;
      stepLabel.style.color = '#94a3b8';
    }
  }

  // tape ki min/max positions nikal — rendering ke liye
  function getTapeBounds() {
    const keys = Object.keys(tape).map(Number);
    keys.push(headPos);
    if (keys.length === 0) return { min: -5, max: 5 };
    const mn = Math.min(...keys);
    const mx = Math.max(...keys);
    // thoda extra space dikhao edges pe
    return { min: mn - 3, max: mx + 3 };
  }

  // ============================================================
  // CANVAS RENDERING
  // ============================================================

  function resizeCanvas() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    canvasW = rect.width;
    canvasH = rect.height;
    canvas.width = Math.round(canvasW * dpr);
    canvas.height = Math.round(canvasH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function draw(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    animationId = null;

    // auto run — step execute karo agar time aa gaya
    if (running && !halted) {
      if (timestamp - lastStepTime >= stepSpeed) {
        executeStep();
        lastStepTime = timestamp;
        if (halted) {
          running = false;
          runBtn.textContent = '\u25B6 Run';
        }
      }
    }

    // head position smoothly lerp karo
    const lerpSpeed = 0.15;
    headDisplayX += (headPos - headDisplayX) * lerpSpeed;
    if (Math.abs(headPos - headDisplayX) < 0.01) headDisplayX = headPos;

    // tape offset smoothly track karo head ke saath
    targetTapeOffset = headPos;
    tapeOffset += (targetTapeOffset - tapeOffset) * lerpSpeed;

    // glow pulse
    headGlow = (Math.sin(timestamp * 0.004) + 1) * 0.5;

    ctx.clearRect(0, 0, canvasW, canvasH);

    drawStateDisplay();
    drawTape();
    drawHead();
    drawRuleTable();

    // loop continue karo agar visible hai
    if (isVisible) {
      animationId = requestAnimationFrame(draw);
    }
  }

  function drawStateDisplay() {
    // state badge — canvas ke top center mein
    const cx = canvasW / 2;
    const y = 30;

    // "State:" label
    ctx.font = '12px ' + FONT;
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Current State', cx, y - 2);

    // state value — bada aur bold
    ctx.font = 'bold 24px ' + FONT;
    if (halted) {
      ctx.fillStyle = HALT_COLOR;
      ctx.fillText('HALT', cx, y + 26);
    } else {
      ctx.fillStyle = ACCENT;
      ctx.fillText(currentState, cx, y + 26);
    }

    // state badge background — subtle glow
    const badgeW = 80;
    const badgeH = 50;
    ctx.strokeStyle = halted ? HALT_COLOR : ACCENT_DIM + '0.3)';
    ctx.lineWidth = 1;
    roundedRect(ctx, cx - badgeW / 2, y - 14, badgeW, badgeH, 8);
    ctx.stroke();

    // reading symbol indicator
    if (!halted) {
      const readSym = readTape(headPos);
      ctx.font = '11px ' + FONT;
      ctx.fillStyle = '#64748b';
      ctx.fillText('Reading: ' + readSym, cx, y + 55);
    }

    // step count — top right
    ctx.font = '11px ' + FONT;
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.fillText('Steps: ' + stepCount, canvasW - 15, 20);
    ctx.textAlign = 'center';

    // program name — top left
    ctx.textAlign = 'left';
    ctx.fillStyle = '#4a6480';
    ctx.fillText(currentProgram, 15, 20);
    ctx.textAlign = 'center';
  }

  function drawTape() {
    const tapeY = TAPE_Y;
    const cellW = CELL_SIZE;
    const cellH = CELL_SIZE;

    // kitne cells dikhenge screen pe
    const visibleCells = Math.ceil(canvasW / (cellW + CELL_GAP)) + 4;
    const centerCell = tapeOffset;

    // tape background strip
    ctx.fillStyle = 'rgba(255,255,255,0.02)';
    roundedRect(ctx, 0, tapeY - 2, canvasW, cellH + 4, 4);
    ctx.fill();

    // har visible cell draw karo
    for (let i = -Math.floor(visibleCells / 2); i <= Math.floor(visibleCells / 2); i++) {
      const cellIdx = Math.round(centerCell) + i;

      // cell ki screen X position
      const offsetFromCenter = cellIdx - tapeOffset;
      const cellX = canvasW / 2 + offsetFromCenter * (cellW + CELL_GAP) - cellW / 2;

      // screen ke bahar hai toh skip
      if (cellX + cellW < -cellW || cellX > canvasW + cellW) continue;

      const symbol = readTape(cellIdx);
      const isHead = cellIdx === headPos;

      // cell background
      if (isHead) {
        // head cell — bright highlight with glow
        const glowAlpha = 0.15 + headGlow * 0.1;
        ctx.shadowColor = HEAD_COLOR;
        ctx.shadowBlur = 12 + headGlow * 6;
        ctx.fillStyle = 'rgba(0,229,255,' + glowAlpha + ')';
        roundedRect(ctx, cellX, tapeY, cellW, cellH, 5);
        ctx.fill();
        ctx.shadowBlur = 0;

        // head cell border — bright
        ctx.strokeStyle = HEAD_COLOR;
        ctx.lineWidth = 2;
        roundedRect(ctx, cellX, tapeY, cellW, cellH, 5);
        ctx.stroke();
      } else {
        // normal cell
        ctx.fillStyle = 'rgba(255,255,255,0.03)';
        roundedRect(ctx, cellX, tapeY, cellW, cellH, 4);
        ctx.fill();

        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        roundedRect(ctx, cellX, tapeY, cellW, cellH, 4);
        ctx.stroke();
      }

      // symbol draw karo
      ctx.font = 'bold 16px ' + FONT;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      if (symbol === '_') {
        ctx.fillStyle = '#333';
        ctx.fillText('_', cellX + cellW / 2, tapeY + cellH / 2);
      } else if (symbol === 'Y') {
        ctx.fillStyle = HALT_COLOR;
        ctx.fillText('Y', cellX + cellW / 2, tapeY + cellH / 2);
      } else if (symbol === 'N') {
        ctx.fillStyle = '#ef4444';
        ctx.fillText('N', cellX + cellW / 2, tapeY + cellH / 2);
      } else {
        ctx.fillStyle = isHead ? '#ffffff' : '#c0c0c0';
        ctx.fillText(symbol, cellX + cellW / 2, tapeY + cellH / 2);
      }

      // cell index — neeche chhota number
      ctx.font = '8px ' + FONT;
      ctx.fillStyle = '#333';
      ctx.fillText(String(cellIdx), cellX + cellW / 2, tapeY + cellH + 10);
    }
  }

  function drawHead() {
    const tapeY = TAPE_Y;
    const cellW = CELL_SIZE;

    // head arrow — tape ke upar triangle pointing down
    const offsetFromCenter = headDisplayX - tapeOffset;
    const headX = canvasW / 2 + offsetFromCenter * (cellW + CELL_GAP);
    const arrowY = tapeY - 8;
    const arrowW = 12;

    // triangle
    ctx.fillStyle = HEAD_COLOR;
    ctx.shadowColor = HEAD_COLOR;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(headX, arrowY);
    ctx.lineTo(headX - arrowW / 2, arrowY - HEAD_ARROW_H);
    ctx.lineTo(headX + arrowW / 2, arrowY - HEAD_ARROW_H);
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;

    // "HEAD" label
    ctx.font = '9px ' + FONT;
    ctx.fillStyle = HEAD_COLOR;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('HEAD', headX, arrowY - HEAD_ARROW_H - 3);
  }

  function drawRuleTable() {
    const prog = PROGRAMS[currentProgram];
    const rules = prog.rules;
    const ruleKeys = Object.keys(rules);

    // rule table position — tape ke neeche
    const tableY = TAPE_Y + CELL_SIZE + 30;
    const rowH = 18;
    const colW = canvasW < 500 ? 80 : 100;

    // header
    ctx.font = 'bold 10px ' + FONT;
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'left';
    ctx.fillText('Transition Rules', 15, tableY);

    // table header row
    const headerY = tableY + 14;
    const headers = ['State', 'Read', 'Write', 'Move', 'Next'];
    const colWidths = [55, 42, 42, 42, 55];
    let hx = 15;

    ctx.font = 'bold 9px ' + FONT;
    ctx.fillStyle = '#4a6480';
    for (let h = 0; h < headers.length; h++) {
      ctx.fillText(headers[h], hx, headerY);
      hx += colWidths[h];
    }

    // separator line
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(15, headerY + 5);
    ctx.lineTo(15 + colWidths.reduce((a, b) => a + b, 0), headerY + 5);
    ctx.stroke();

    // rule rows — 2 column layout agar jagah hai
    ctx.font = '9px ' + FONT;
    const maxRowsPerCol = Math.floor((canvasH - headerY - 20) / rowH);
    const useDoubleCol = ruleKeys.length > maxRowsPerCol && canvasW > 450;

    for (let i = 0; i < ruleKeys.length; i++) {
      const key = ruleKeys[i];
      const [writeSym, dir, nextState] = rules[key];
      const [state, readSym] = key.split(',');

      // position — single ya double column
      let rx, ry;
      if (useDoubleCol) {
        const col = i < Math.ceil(ruleKeys.length / 2) ? 0 : 1;
        const row = col === 0 ? i : i - Math.ceil(ruleKeys.length / 2);
        rx = 15 + col * (colWidths.reduce((a, b) => a + b, 0) + 20);
        ry = headerY + 12 + row * rowH;
      } else {
        rx = 15;
        ry = headerY + 12 + i * rowH;
      }

      // check karo ye row screen ke bahar toh nahi
      if (ry > canvasH - 5) continue;

      const isActive = key === highlightedRule && !halted;

      // highlight background agar active rule hai
      if (isActive) {
        ctx.fillStyle = ACCENT_DIM + '0.12)';
        const totalW = colWidths.reduce((a, b) => a + b, 0) + 4;
        roundedRect(ctx, rx - 2, ry - 10, totalW, rowH - 2, 3);
        ctx.fill();
      }

      // rule data
      const vals = [state, readSym, writeSym, dir, nextState];
      let vx = rx;
      for (let v = 0; v < vals.length; v++) {
        if (isActive) {
          ctx.fillStyle = v === 4 ? ACCENT : '#e2e8f0'; // next state accent mein
        } else {
          ctx.fillStyle = '#64748b';
        }
        ctx.textAlign = 'left';
        ctx.fillText(vals[v], vx, ry);
        vx += colWidths[v];
      }
    }
  }

  // rounded rect helper
  function roundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ============================================================
  // INTERSECTION OBSERVER — sirf visible hone pe animate karo
  // ============================================================

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      isVisible = entry.isIntersecting;
      if (isVisible && !animationId) {
        animationId = requestAnimationFrame(draw);
      }
    }
  }, { threshold: 0.1 });
  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) draw(); });

  // resize handler
  const resizeObserver = new ResizeObserver(() => {
    if (!isVisible) return;
    resizeCanvas();
  });
  resizeObserver.observe(canvas);

  // ============================================================
  // INITIALIZATION — sab set karo
  // ============================================================

  resizeCanvas();
  resetMachine();

  // pehla frame draw kar
  if (isVisible) {
    animationId = requestAnimationFrame(draw);
  }
}
