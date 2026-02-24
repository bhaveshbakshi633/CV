// ============================================================
// Falling Sand / Powder Toy — Cellular Automaton Physics Sandbox
// Har pixel ek element hai — sand girta hai, water behta hai,
// fire jalta hai, oil tairta hai — sab simple rules se
// Ye wahi powder toy hai jo bachpan mein ghanton khelte the
// ============================================================

// main entry point — container dhundho, grid banao, chaos shuru karo
export function initFallingSand() {
  const container = document.getElementById('fallingSandContainer');
  if (!container) {
    console.warn('fallingSandContainer nahi mila bhai, sand simulation skip');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#f59e0b'; // amber accent — portfolio theme
  const ACCENT_RGB = '245,158,11';

  // element types — integer enum, fast comparison ke liye
  const EMPTY = 0;
  const SAND = 1;
  const WATER = 2;
  const STONE = 3;
  const FIRE = 4;
  const WOOD = 5;
  const OIL = 6;
  const STEAM = 7;

  // element info — display names, colors, descriptions
  const ELEMENTS = {
    [SAND]:  { name: 'Sand',  colors: ['#c2b280','#d4c28a','#b8a670','#bfad6e','#c8bc8c'] },
    [WATER]: { name: 'Water', colors: ['#2196F3','#1E88E5','#42A5F5','#1976D2','#2962FF'] },
    [STONE]: { name: 'Stone', colors: ['#808080','#909090','#707070','#757575','#858585'] },
    [FIRE]:  { name: 'Fire',  colors: ['#FF6600','#FF4400','#FFAA00','#FF3300','#FF8800'] },
    [WOOD]:  { name: 'Wood',  colors: ['#8B4513','#A0522D','#6B3410','#7B3B12','#935C30'] },
    [OIL]:   { name: 'Oil',   colors: ['#4a4a00','#5a5a10','#3a3a00','#505010','#454500'] },
    [STEAM]: { name: 'Steam', colors: ['#aabbdd','#99aacc','#bbccee','#8899bb','#aabbcc'] },
  };

  // --- Grid setup ---
  // grid resolution — canvas se chhota, har cell multiple pixels cover karta hai
  const GRID_W = 200;
  let GRID_H = Math.floor(GRID_W * (CANVAS_HEIGHT / 600)); // proportional height
  GRID_H = Math.max(120, GRID_H); // minimum height toh rakh

  // main grid arrays — Uint8Array fast hai aur memory efficient
  let cells = new Uint8Array(GRID_W * GRID_H);       // element type
  let meta = new Uint8Array(GRID_W * GRID_H);         // lifetime/heat metadata
  let moved = new Uint8Array(GRID_W * GRID_H);        // "already processed this frame" flag
  let colorIdx = new Uint8Array(GRID_W * GRID_H);     // random color index for texture

  // --- State ---
  let animationId = null;
  let isVisible = false;
  let selectedElement = SAND;
  let brushSize = 3;
  let simSpeed = 2; // simulation steps per frame
  let isMouseDown = false;
  let mouseGridX = -1, mouseGridY = -1;

  // color index har cell ke liye random assign kar — texture ke liye
  for (let i = 0; i < colorIdx.length; i++) {
    colorIdx[i] = Math.floor(Math.random() * 5);
  }

  // --- Existing children preserve karo (agar hai toh) ---
  const existingChildren = Array.from(container.children);

  // --- Canvas setup ---
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;border-radius:8px;cursor:crosshair;display:block;';
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // offscreen canvas for grid rendering — grid resolution pe draw, fir scale up
  const offCanvas = document.createElement('canvas');
  offCanvas.width = GRID_W;
  offCanvas.height = GRID_H;
  const offCtx = offCanvas.getContext('2d');

  // --- Resize handler ---
  let canvasW = 0, canvasH = 0;
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    canvasW = w;
    canvasH = CANVAS_HEIGHT;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(CANVAS_HEIGHT * dpr);
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- Helper functions ---
  // grid index calculate kar — bounds check ke saath
  function idx(x, y) {
    return y * GRID_W + x;
  }

  function inBounds(x, y) {
    return x >= 0 && x < GRID_W && y >= 0 && y < GRID_H;
  }

  function getCell(x, y) {
    if (!inBounds(x, y)) return STONE; // boundary = solid wall
    return cells[idx(x, y)];
  }

  function setCell(x, y, type) {
    if (!inBounds(x, y)) return;
    const i = idx(x, y);
    cells[i] = type;
    // naya color index assign kar jab cell type change ho
    colorIdx[i] = Math.floor(Math.random() * 5);
  }

  function getMeta(x, y) {
    if (!inBounds(x, y)) return 0;
    return meta[idx(x, y)];
  }

  function setMeta(x, y, val) {
    if (!inBounds(x, y)) return;
    meta[idx(x, y)] = val;
  }

  // swap two cells — position aur metadata dono swap karo
  function swapCells(x1, y1, x2, y2) {
    if (!inBounds(x1, y1) || !inBounds(x2, y2)) return;
    const i1 = idx(x1, y1);
    const i2 = idx(x2, y2);
    // cell type swap
    const tmpC = cells[i1]; cells[i1] = cells[i2]; cells[i2] = tmpC;
    // metadata swap
    const tmpM = meta[i1]; meta[i1] = meta[i2]; meta[i2] = tmpM;
    // color index swap
    const tmpCol = colorIdx[i1]; colorIdx[i1] = colorIdx[i2]; colorIdx[i2] = tmpCol;
    // moved flags set kar — double processing rokne ke liye
    moved[i1] = 1;
    moved[i2] = 1;
  }

  // --- Physics simulation ---
  // bottom se top process karo — falling particles sahi move karein
  // horizontal scan direction random karo — left-right bias nahi chahiye
  function simulate() {
    // moved array clear karo
    moved.fill(0);

    // scan direction randomize — har frame pe random left-to-right ya right-to-left
    const leftToRight = Math.random() < 0.5;

    // bottom se top — row by row
    for (let y = GRID_H - 1; y >= 0; y--) {
      const startX = leftToRight ? 0 : GRID_W - 1;
      const endX = leftToRight ? GRID_W : -1;
      const stepX = leftToRight ? 1 : -1;

      for (let x = startX; x !== endX; x += stepX) {
        const i = idx(x, y);
        if (moved[i]) continue; // already processed — skip

        const type = cells[i];
        if (type === EMPTY || type === STONE) continue; // empty aur stone kuch nahi karte

        switch (type) {
          case SAND:  updateSand(x, y); break;
          case WATER: updateWater(x, y); break;
          case FIRE:  updateFire(x, y); break;
          case WOOD:  updateWood(x, y); break;
          case OIL:   updateOil(x, y); break;
          case STEAM: updateSteam(x, y); break;
        }
      }
    }
  }

  // --- SAND physics ---
  // gravity se girta hai, angle of repose pe pile up hota hai
  function updateSand(x, y) {
    const below = getCell(x, y + 1);

    // seedha neeche gir — agar empty ya liquid hai
    if (below === EMPTY) {
      swapCells(x, y, x, y + 1);
      return;
    }
    // sand pani mein doob jaata hai — swap karo
    if (below === WATER || below === OIL) {
      swapCells(x, y, x, y + 1);
      return;
    }

    // neeche nahi ja sakta — diagonal try kar (random order, angle of repose)
    const dir = Math.random() < 0.5 ? -1 : 1;
    const dl = getCell(x + dir, y + 1);
    const dr = getCell(x - dir, y + 1);

    if (dl === EMPTY || dl === WATER || dl === OIL) {
      swapCells(x, y, x + dir, y + 1);
    } else if (dr === EMPTY || dr === WATER || dr === OIL) {
      swapCells(x, y, x - dir, y + 1);
    }
    // nahi gir sakta — ruk ja bhai
  }

  // --- WATER physics ---
  // girta hai, fir sideways behta hai — containers fill karta hai
  function updateWater(x, y) {
    const below = getCell(x, y + 1);

    // seedha neeche gir
    if (below === EMPTY) {
      swapCells(x, y, x, y + 1);
      return;
    }
    // pani oil ke neeche jaata hai (oil halka hai)
    if (below === OIL) {
      swapCells(x, y, x, y + 1);
      return;
    }

    // diagonal try kar
    const dir = Math.random() < 0.5 ? -1 : 1;
    const dl = getCell(x + dir, y + 1);
    const dr = getCell(x - dir, y + 1);

    if (dl === EMPTY) {
      swapCells(x, y, x + dir, y + 1);
      return;
    }
    if (dr === EMPTY) {
      swapCells(x, y, x - dir, y + 1);
      return;
    }

    // neeche nahi ja sakta — sideways flow kar (random direction)
    const sideDir = Math.random() < 0.5 ? -1 : 1;
    const sl = getCell(x + sideDir, y);
    const sr = getCell(x - sideDir, y);

    if (sl === EMPTY) {
      swapCells(x, y, x + sideDir, y);
    } else if (sr === EMPTY) {
      swapCells(x, y, x - sideDir, y);
    }
  }

  // --- FIRE physics ---
  // upar uthta hai, flicker karta hai, wood/oil jalata hai, limited lifetime
  function updateFire(x, y) {
    const i = idx(x, y);
    let life = meta[i];

    // lifetime khatam — bujh ja bhai
    if (life === 0) {
      // fire bujhne pe empty ho jaata hai
      cells[i] = EMPTY;
      meta[i] = 0;
      return;
    }

    // lifetime ghata
    meta[i] = life - 1;

    // aas paas check kar — kya jalane layak hai?
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx, ny = y + dy;
        const neighbor = getCell(nx, ny);

        // wood ko aag laga — probabilistic spread (thoda time lagta hai)
        if (neighbor === WOOD && Math.random() < 0.03) {
          setCell(nx, ny, FIRE);
          setMeta(nx, ny, 60 + Math.floor(Math.random() * 40)); // wood jyada der jalta hai
        }
        // oil instantly jal jaata hai — highly flammable
        if (neighbor === OIL && Math.random() < 0.6) {
          setCell(nx, ny, FIRE);
          setMeta(nx, ny, 30 + Math.floor(Math.random() * 30));
        }
        // pani se steam ban jaata hai
        if (neighbor === WATER && Math.random() < 0.15) {
          setCell(nx, ny, STEAM);
          setMeta(nx, ny, 80 + Math.floor(Math.random() * 60));
          // ye fire cell bhi bujh jaaye pani ke touch se
          if (Math.random() < 0.4) {
            cells[i] = STEAM;
            meta[i] = 60 + Math.floor(Math.random() * 40);
            return;
          }
        }
      }
    }

    // fire upar uthta hai — random horizontal drift ke saath
    const drift = Math.random() < 0.33 ? -1 : (Math.random() < 0.5 ? 1 : 0);
    const above = getCell(x + drift, y - 1);

    if (above === EMPTY && inBounds(x + drift, y - 1)) {
      swapCells(x, y, x + drift, y - 1);
    } else if (getCell(x, y - 1) === EMPTY) {
      swapCells(x, y, x, y - 1);
    }
    // agar kahi nahi ja sakta toh flicker in place — ye normal hai fire ke liye
  }

  // --- WOOD physics ---
  // static hai — hil nahi sakta. Bas jalta hai jab fire adjacent ho
  function updateWood(x, y) {
    // wood khud se kuch nahi karta — fire spread fire update mein handle hota hai
    // but extra check: agar chaaron taraf fire hai toh jaldi jal
    let fireNeighbors = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        if (getCell(x + dx, y + dy) === FIRE) fireNeighbors++;
      }
    }
    // jyada fire neighbors = jyada chance of catching fire
    if (fireNeighbors >= 2 && Math.random() < 0.05 * fireNeighbors) {
      setCell(x, y, FIRE);
      setMeta(x, y, 50 + Math.floor(Math.random() * 50));
    }
  }

  // --- OIL physics ---
  // water jaisa behta hai but halka hai — pani pe tairta hai
  // highly flammable — fire touch kare toh turant jal jaaye
  function updateOil(x, y) {
    const below = getCell(x, y + 1);

    // seedha neeche gir — agar empty hai
    if (below === EMPTY) {
      swapCells(x, y, x, y + 1);
      return;
    }
    // oil pani pe tairta hai — agar neeche water hai, SWAP MAT KAR
    // (water update mein water neeche jaata hai oil ke, so oil naturally upar aata hai)

    // diagonal try kar
    const dir = Math.random() < 0.5 ? -1 : 1;
    const dl = getCell(x + dir, y + 1);
    const dr = getCell(x - dir, y + 1);

    if (dl === EMPTY) {
      swapCells(x, y, x + dir, y + 1);
      return;
    }
    if (dr === EMPTY) {
      swapCells(x, y, x - dir, y + 1);
      return;
    }

    // sideways flow
    const sideDir = Math.random() < 0.5 ? -1 : 1;
    const sl = getCell(x + sideDir, y);
    const sr = getCell(x - sideDir, y);

    if (sl === EMPTY) {
      swapCells(x, y, x + sideDir, y);
    } else if (sr === EMPTY) {
      swapCells(x, y, x - sideDir, y);
    }
  }

  // --- STEAM physics ---
  // upar uthta hai, disperse hota hai, fir condense hokar water ban jaata hai
  function updateSteam(x, y) {
    const i = idx(x, y);
    let life = meta[i];

    // lifetime khatam — condense back to water
    if (life === 0) {
      cells[i] = WATER;
      meta[i] = 0;
      colorIdx[i] = Math.floor(Math.random() * 5);
      return;
    }

    // lifetime ghata
    meta[i] = life - 1;

    // upar uth — random horizontal drift ke saath (steam diffuse hota hai)
    const drift = Math.random() < 0.4 ? -1 : (Math.random() < 0.5 ? 1 : 0);
    const targetX = x + drift;
    const targetY = y - 1;

    if (inBounds(targetX, targetY) && getCell(targetX, targetY) === EMPTY) {
      swapCells(x, y, targetX, targetY);
    } else if (getCell(x, y - 1) === EMPTY) {
      swapCells(x, y, x, y - 1);
    } else {
      // upar nahi ja sakta — sideways try kar (dispersion)
      const sd = Math.random() < 0.5 ? -1 : 1;
      if (getCell(x + sd, y) === EMPTY) {
        swapCells(x, y, x + sd, y);
      } else if (getCell(x - sd, y) === EMPTY) {
        swapCells(x, y, x - sd, y);
      }
    }
  }

  // --- Rendering ---
  // grid resolution pe ImageData banao, fir canvas pe scale karke draw karo
  function render() {
    const imgData = offCtx.createImageData(GRID_W, GRID_H);
    const data = imgData.data;

    for (let y = 0; y < GRID_H; y++) {
      for (let x = 0; x < GRID_W; x++) {
        const i = idx(x, y);
        const pi = (y * GRID_W + x) * 4; // pixel index in ImageData
        const type = cells[i];

        if (type === EMPTY) {
          // dark background — almost black
          data[pi]     = 12;
          data[pi + 1] = 12;
          data[pi + 2] = 20;
          data[pi + 3] = 255;
          continue;
        }

        const ci = colorIdx[i] % 5; // color variation index
        const el = ELEMENTS[type];
        if (!el) {
          data[pi] = 0; data[pi+1] = 0; data[pi+2] = 0; data[pi+3] = 255;
          continue;
        }

        // fire ka color lifetime pe depend karta hai — jyada life = jyada bright
        if (type === FIRE) {
          const life = meta[i];
          const t = Math.min(1, life / 80); // 0 to 1, higher = more alive
          // hot = yellow/white, dying = red/dark
          const r = Math.min(255, 180 + Math.floor(75 * t));
          const g = Math.min(255, Math.floor(40 + 180 * t * Math.random()));
          const b = Math.floor(20 * t * Math.random());
          data[pi]     = r;
          data[pi + 1] = g;
          data[pi + 2] = b;
          data[pi + 3] = 255;
          continue;
        }

        // steam ka alpha lifetime pe depend karta hai — fade out hota hai
        if (type === STEAM) {
          const life = meta[i];
          const t = Math.min(1, life / 100);
          data[pi]     = 160 + Math.floor(60 * t);
          data[pi + 1] = 170 + Math.floor(60 * t);
          data[pi + 2] = 220 + Math.floor(35 * t);
          data[pi + 3] = Math.floor(100 + 155 * t);
          continue;
        }

        // baaki elements — hex color parse karke use karo
        const hex = el.colors[ci];
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        data[pi]     = r;
        data[pi + 1] = g;
        data[pi + 2] = b;
        data[pi + 3] = 255;
      }
    }

    // offscreen canvas pe ImageData put karo
    offCtx.putImageData(imgData, 0, 0);

    // main canvas pe scale karke draw — pixelated look chahiye
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.drawImage(offCanvas, 0, 0, GRID_W, GRID_H, 0, 0, canvasW, canvasH);

    // brush preview draw karo — jahan mouse hai wahan circle dikhao
    if (mouseGridX >= 0 && mouseGridY >= 0) {
      const cellW = canvasW / GRID_W;
      const cellH = canvasH / GRID_H;
      const cx = (mouseGridX + 0.5) * cellW;
      const cy = (mouseGridY + 0.5) * cellH;
      const radius = brushSize * cellW;
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // --- Mouse/Touch interaction ---
  // canvas coordinates se grid coordinates mein convert karo
  function canvasToGrid(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const relX = clientX - rect.left;
    const relY = clientY - rect.top;
    const gx = Math.floor((relX / rect.width) * GRID_W);
    const gy = Math.floor((relY / rect.height) * GRID_H);
    return { x: gx, y: gy };
  }

  // brush se paint karo — circular area mein selected element daal
  function paintBrush(gx, gy) {
    for (let dy = -brushSize; dy <= brushSize; dy++) {
      for (let dx = -brushSize; dx <= brushSize; dx++) {
        // circular brush — square nahi chahiye
        if (dx * dx + dy * dy > brushSize * brushSize) continue;
        const nx = gx + dx;
        const ny = gy + dy;
        if (!inBounds(nx, ny)) continue;

        const existing = getCell(nx, ny);

        if (selectedElement === EMPTY) {
          // eraser mode — sab kuch mita do
          setCell(nx, ny, EMPTY);
          setMeta(nx, ny, 0);
        } else if (selectedElement === FIRE) {
          // fire toh kahi bhi laga do except stone pe
          if (existing === EMPTY || existing === WOOD || existing === OIL) {
            setCell(nx, ny, FIRE);
            setMeta(nx, ny, 40 + Math.floor(Math.random() * 60));
          }
        } else if (selectedElement === STEAM) {
          if (existing === EMPTY) {
            setCell(nx, ny, STEAM);
            setMeta(nx, ny, 80 + Math.floor(Math.random() * 80));
          }
        } else {
          // baaki elements sirf empty pe paint ho
          if (existing === EMPTY) {
            setCell(nx, ny, selectedElement);
            setMeta(nx, ny, 0);
          }
        }
      }
    }
  }

  // mouse events — painting shuru/band karo
  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    isMouseDown = true;
    const { x, y } = canvasToGrid(e.clientX, e.clientY);
    mouseGridX = x; mouseGridY = y;
    paintBrush(x, y);
  });

  canvas.addEventListener('mousemove', (e) => {
    const { x, y } = canvasToGrid(e.clientX, e.clientY);
    mouseGridX = x; mouseGridY = y;
    if (isMouseDown) {
      paintBrush(x, y);
    }
  });

  canvas.addEventListener('mouseup', () => { isMouseDown = false; });
  canvas.addEventListener('mouseleave', () => {
    isMouseDown = false;
    mouseGridX = -1; mouseGridY = -1;
  });

  // touch events — mobile pe bhi kaam kare
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isMouseDown = true;
    const touch = e.touches[0];
    const { x, y } = canvasToGrid(touch.clientX, touch.clientY);
    mouseGridX = x; mouseGridY = y;
    paintBrush(x, y);
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const { x, y } = canvasToGrid(touch.clientX, touch.clientY);
    mouseGridX = x; mouseGridY = y;
    if (isMouseDown) {
      paintBrush(x, y);
    }
  }, { passive: false });

  canvas.addEventListener('touchend', () => {
    isMouseDown = false;
    mouseGridX = -1; mouseGridY = -1;
  });

  // --- Controls UI ---
  // helper: colored swatch + text wala button banao (safe DOM methods, no innerHTML)
  function makeElementButton(swatchColor, labelText, parent, onClick) {
    const btn = document.createElement('button');
    btn.style.cssText = [
      'background:rgba(' + ACCENT_RGB + ',0.08)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'border-radius:6px',
      'padding:5px 10px',
      'font-size:11px',
      'font-family:"JetBrains Mono",monospace',
      'cursor:pointer',
      'transition:all 0.2s',
      'user-select:none',
      'display:inline-flex',
      'align-items:center',
      'gap:4px',
    ].join(';');

    // color swatch span
    const swatch = document.createElement('span');
    swatch.style.cssText = 'display:inline-block;width:10px;height:10px;border-radius:2px;' +
      'background:' + swatchColor + ';border:1px solid rgba(255,255,255,0.15);';
    btn.appendChild(swatch);

    // label text
    const label = document.createTextNode(labelText);
    btn.appendChild(label);

    btn.addEventListener('click', onClick);
    parent.appendChild(btn);
    return btn;
  }

  // plain button helper — bina swatch ke
  function makeButton(text, parent, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'background:rgba(' + ACCENT_RGB + ',0.08)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.25)',
      'border-radius:6px',
      'padding:5px 12px',
      'font-size:11px',
      'font-family:"JetBrains Mono",monospace',
      'cursor:pointer',
      'transition:all 0.2s',
      'user-select:none',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.25)';
      btn.style.color = '#ffffff';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.08)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    parent.appendChild(btn);
    return btn;
  }

  // element selector row — colored buttons for each element
  const controlsRow1 = document.createElement('div');
  controlsRow1.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:6px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsRow1);

  // controls row 2 — brush size, speed, clear
  const controlsRow2 = document.createElement('div');
  controlsRow2.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:6px',
    'align-items:center',
    'justify-content:center',
  ].join(';');
  container.appendChild(controlsRow2);

  // --- Element selector buttons ---
  const elementBtns = [];

  // eraser button — EMPTY select karo
  const eraserBtn = makeElementButton('#1a1a2e', 'Erase', controlsRow1, () => {
    selectedElement = EMPTY;
    updateElementButtons();
  });
  elementBtns.push({ btn: eraserBtn, type: EMPTY });

  // har element ke liye ek colored button banao
  const elementTypes = [SAND, WATER, STONE, FIRE, WOOD, OIL, STEAM];
  elementTypes.forEach((type) => {
    const el = ELEMENTS[type];
    const btn = makeElementButton(el.colors[0], el.name, controlsRow1, () => {
      selectedElement = type;
      updateElementButtons();
    });
    elementBtns.push({ btn, type });
  });

  // active element button highlight karo
  function updateElementButtons() {
    elementBtns.forEach(({ btn, type }) => {
      if (type === selectedElement) {
        btn.dataset.active = '1';
        btn.style.borderColor = ACCENT;
        btn.style.color = '#ffffff';
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.2)';
        btn.style.boxShadow = '0 0 8px rgba(' + ACCENT_RGB + ',0.3)';
      } else {
        btn.dataset.active = '';
        btn.style.borderColor = 'rgba(' + ACCENT_RGB + ',0.25)';
        btn.style.color = '#b0b0b0';
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.08)';
        btn.style.boxShadow = 'none';
      }
    });
  }
  // initial state — sand selected hai by default
  updateElementButtons();

  // hover effects for element buttons
  elementBtns.forEach(({ btn }) => {
    btn.addEventListener('mouseenter', () => {
      if (!btn.dataset.active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.15)';
        btn.style.color = '#dddddd';
      }
    });
    btn.addEventListener('mouseleave', () => {
      if (!btn.dataset.active) {
        btn.style.background = 'rgba(' + ACCENT_RGB + ',0.08)';
        btn.style.color = '#b0b0b0';
      }
    });
  });

  // --- Row 2: Brush size slider ---
  const brushLabel = document.createElement('span');
  brushLabel.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
  brushLabel.textContent = 'Brush: ' + brushSize;
  controlsRow2.appendChild(brushLabel);

  const brushSlider = document.createElement('input');
  brushSlider.type = 'range';
  brushSlider.min = '1';
  brushSlider.max = '10';
  brushSlider.value = String(brushSize);
  brushSlider.style.cssText = 'width:70px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
  brushSlider.addEventListener('input', () => {
    brushSize = parseInt(brushSlider.value);
    brushLabel.textContent = 'Brush: ' + brushSize;
  });
  controlsRow2.appendChild(brushSlider);

  // separator
  const sep1 = document.createElement('span');
  sep1.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.3);font-size:11px;padding:0 2px;';
  sep1.textContent = '|';
  controlsRow2.appendChild(sep1);

  // --- Speed slider ---
  const speedLabel = document.createElement('span');
  speedLabel.style.cssText = 'color:#b0b0b0;font-size:11px;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
  speedLabel.textContent = 'Speed: ' + simSpeed;
  controlsRow2.appendChild(speedLabel);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '5';
  speedSlider.value = String(simSpeed);
  speedSlider.style.cssText = 'width:70px;height:4px;accent-color:' + ACCENT + ';cursor:pointer;';
  speedSlider.addEventListener('input', () => {
    simSpeed = parseInt(speedSlider.value);
    speedLabel.textContent = 'Speed: ' + simSpeed;
  });
  controlsRow2.appendChild(speedSlider);

  // separator
  const sep2 = document.createElement('span');
  sep2.style.cssText = 'color:rgba(' + ACCENT_RGB + ',0.3);font-size:11px;padding:0 2px;';
  sep2.textContent = '|';
  controlsRow2.appendChild(sep2);

  // --- Clear button ---
  makeButton('Clear', controlsRow2, () => {
    // poora grid saaf kar do — tabula rasa
    cells.fill(EMPTY);
    meta.fill(0);
    moved.fill(0);
    // color indices bhi fresh random kar
    for (let i = 0; i < colorIdx.length; i++) {
      colorIdx[i] = Math.floor(Math.random() * 5);
    }
  });

  // --- Animation loop ---
  function loop() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    // simSpeed steps per frame — jitna speed utne baar simulate
    for (let s = 0; s < simSpeed; s++) {
      simulate();
    }

    render();
    animationId = requestAnimationFrame(loop);
  }

  // --- Visibility observer — tab se bahar jaane pe band kar, aane pe chalu ---
  const obs = new IntersectionObserver(([e]) => {
    isVisible = e.isIntersecting;
    if (isVisible && !animationId) loop();
    else if (!isVisible && animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }, { threshold: 0.1 });
  obs.observe(container);
  // lab resume: restart loop when focus released
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) loop(); });
}
