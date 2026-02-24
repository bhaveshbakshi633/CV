// ============================================================
// Force-Directed Graph Layout — Coulomb repulsion + Hooke springs
// nodes ek dusre ko dhakelte hain, edges spring ki tarah kheenchte hain
// jab system settle hota hai toh readable network layout ban jaata hai
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, graph chalao
export function initForceGraph() {
  const container = document.getElementById('forceGraphContainer');
  if (!container) {
    console.warn('forceGraphContainer nahi mila bhai, force-graph skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 400;
  const ACCENT = '#a78bfa';
  const ACCENT_RGB = '167,139,250';
  const BG_COLOR = '#0a0a0a';
  const EDGE_COLOR = 'rgba(255,255,255,0.25)';
  const EDGE_WIDTH = 1.5;

  // node color palette — blues, greens, purples, cyans
  const NODE_COLORS = [
    '#a78bfa', '#7c3aed', '#818cf8', '#6366f1',
    '#34d399', '#10b981', '#2dd4bf', '#06b6d4',
    '#60a5fa', '#3b82f6', '#8b5cf6', '#c084fc',
  ];

  // --- Physics defaults ---
  let kRepel = 500;       // coulomb repulsion strength
  let kSpring = 0.01;     // hooke spring constant
  let restLength = 80;    // spring rest length pixels mein
  let kCenter = 0.001;    // center gravity pull
  let damping = 0.85;     // velocity damping per frame

  // --- Graph state ---
  let nodes = [];         // [{id, x, y, vx, vy, radius, color, pinned}]
  let edges = [];         // [{source, target}] — indices into nodes
  let nextNodeId = 0;

  // --- Interaction state ---
  let draggedNode = null;       // currently dragged node
  let selectedNode = null;      // pehle click pe select, doosre pe edge bana
  let isDragging = false;
  let dragStartX = 0, dragStartY = 0;
  let dragMoved = false;        // track karo ki actually drag hua ya sirf click tha
  const DRAG_THRESHOLD = 5;     // pixels — isse kam move hua toh click maano

  // animation state
  let animationId = null;
  let isVisible = false;

  // --- DOM structure banao ---
  // pehle game-header aur game-desc bachao
  const existingHeader = container.querySelector('.game-header');
  const existingDesc = container.querySelector('.game-desc');
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  if (existingHeader) container.appendChild(existingHeader);
  if (existingDesc) container.appendChild(existingDesc);
  container.style.cssText += ';width:100%;position:relative;';

  // main canvas
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(' + ACCENT_RGB + ',0.15)',
    'border-radius:8px',
    'cursor:grab',
    'background:' + BG_COLOR,
    'margin-top:8px',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Controls bar ---
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:10px',
    'margin-top:10px',
    'align-items:center',
    'justify-content:flex-start',
  ].join(';');
  container.appendChild(controlsDiv);

  // --- Slider helper ---
  function createSlider(label, min, max, step, defaultVal, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:4px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = 'color:#b0b0b0;font-size:11px;font-weight:600;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = defaultVal;
    slider.style.cssText = 'width:70px;height:4px;accent-color:rgba(' + ACCENT_RGB + ',0.8);cursor:pointer;';
    wrapper.appendChild(slider);

    const valueEl = document.createElement('span');
    valueEl.style.cssText = 'color:#b0b0b0;font-size:10px;min-width:28px;font-family:"JetBrains Mono",monospace;';
    valueEl.textContent = parseFloat(defaultVal).toFixed(step < 1 ? 2 : 0);
    wrapper.appendChild(valueEl);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valueEl.textContent = step < 1 ? val.toFixed(2) : val.toFixed(0);
      onChange(val);
    });

    controlsDiv.appendChild(wrapper);
    return { slider, valueEl };
  }

  // --- Dropdown helper ---
  function createDropdown(label, options, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:4px;';

    const labelEl = document.createElement('span');
    labelEl.style.cssText = 'color:#b0b0b0;font-size:11px;font-weight:600;font-family:"JetBrains Mono",monospace;white-space:nowrap;';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    const select = document.createElement('select');
    select.style.cssText = [
      'padding:3px 6px',
      'font-size:11px',
      'border-radius:4px',
      'cursor:pointer',
      'background:#1a1a2e',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.2)',
      'font-family:"JetBrains Mono",monospace',
      'outline:none',
    ].join(';');

    options.forEach((opt) => {
      const optEl = document.createElement('option');
      optEl.value = opt;
      optEl.textContent = opt;
      select.appendChild(optEl);
    });

    select.addEventListener('change', () => onChange(select.value));
    wrapper.appendChild(select);
    controlsDiv.appendChild(wrapper);
    return select;
  }

  // --- Button helper ---
  function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = [
      'padding:4px 10px',
      'font-size:11px',
      'border-radius:5px',
      'cursor:pointer',
      'background:rgba(' + ACCENT_RGB + ',0.08)',
      'color:#b0b0b0',
      'border:1px solid rgba(' + ACCENT_RGB + ',0.2)',
      'font-family:"JetBrains Mono",monospace',
      'transition:all 0.2s ease',
      'white-space:nowrap',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.2)';
      btn.style.color = '#e0e0e0';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(' + ACCENT_RGB + ',0.08)';
      btn.style.color = '#b0b0b0';
    });
    btn.addEventListener('click', onClick);
    controlsDiv.appendChild(btn);
    return btn;
  }

  // --- Controls banao ---
  const presetSelect = createDropdown('Preset', ['Random', 'Tree', 'Ring', 'Mesh', 'Star'], (val) => {
    loadPreset(val);
  });

  createSlider('Repulsion', 100, 2000, 10, kRepel, (v) => { kRepel = v; });
  createSlider('Spring', 30, 200, 1, restLength, (v) => { restLength = v; });
  createSlider('Damping', 0.70, 0.99, 0.01, damping, (v) => { damping = v; });
  createButton('Clear', () => {
    nodes = [];
    edges = [];
    nextNodeId = 0;
    selectedNode = null;
    draggedNode = null;
  });

  // --- Hint text ---
  const hintDiv = document.createElement('div');
  hintDiv.style.cssText = [
    'margin-top:6px',
    'padding:4px 8px',
    'font-family:"JetBrains Mono",monospace',
    'font-size:10px',
    'color:rgba(176,176,176,0.4)',
    'text-align:center',
  ].join(';');
  hintDiv.textContent = 'click to add node \u2022 drag to move \u2022 click two nodes to connect \u2022 shift+click to delete';
  container.appendChild(hintDiv);

  // --- Canvas sizing ---
  let canvasW = 0, canvasH = 0, dpr = 1;

  function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const containerWidth = container.clientWidth;
    canvasW = containerWidth;
    canvasH = CANVAS_HEIGHT;
    canvas.width = containerWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = CANVAS_HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // ============================================================
  // NODE & EDGE MANAGEMENT
  // ============================================================

  // naya node banao — position, velocity, color sab set karo
  function addNode(x, y) {
    const id = nextNodeId++;
    const colorIdx = id % NODE_COLORS.length;
    const node = {
      id: id,
      x: x,
      y: y,
      vx: 0,
      vy: 0,
      radius: 12, // base radius, baad mein degree se adjust karenge
      color: NODE_COLORS[colorIdx],
      pinned: false,
    };
    nodes.push(node);
    return node;
  }

  // do nodes ke beech edge banao — duplicate check bhi kar
  function addEdge(sourceIdx, targetIdx) {
    if (sourceIdx === targetIdx) return;
    // duplicate check — same edge already toh nahi hai
    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      if ((e.source === sourceIdx && e.target === targetIdx) ||
          (e.source === targetIdx && e.target === sourceIdx)) {
        return; // already hai, skip
      }
    }
    edges.push({ source: sourceIdx, target: targetIdx });
  }

  // node delete karo aur usse connected saare edges bhi hata do
  function removeNode(nodeIdx) {
    if (nodeIdx < 0 || nodeIdx >= nodes.length) return;
    // pehle edges hata do jo is node se connected hain
    edges = edges.filter((e) => e.source !== nodeIdx && e.target !== nodeIdx);
    // ab edges ke indices fix karo — jo node hata hai usse bade indices kam karo
    edges = edges.map((e) => ({
      source: e.source > nodeIdx ? e.source - 1 : e.source,
      target: e.target > nodeIdx ? e.target - 1 : e.target,
    }));
    nodes.splice(nodeIdx, 1);
    // selected node reset karo agar wahi tha
    if (selectedNode !== null) {
      if (selectedNode === nodeIdx) selectedNode = null;
      else if (selectedNode > nodeIdx) selectedNode--;
    }
  }

  // node ka degree (connections count) nikaalo — radius adjust karne ke liye
  function getNodeDegree(nodeIdx) {
    let deg = 0;
    for (let i = 0; i < edges.length; i++) {
      if (edges[i].source === nodeIdx || edges[i].target === nodeIdx) deg++;
    }
    return deg;
  }

  // node ka display radius — zyada connections = bada node
  function getDisplayRadius(nodeIdx) {
    const deg = getNodeDegree(nodeIdx);
    return 10 + Math.min(deg * 2, 12); // 10 se 22 tak
  }

  // ============================================================
  // PRESETS — readymade graph structures
  // ============================================================

  function loadPreset(name) {
    nodes = [];
    edges = [];
    nextNodeId = 0;
    selectedNode = null;
    draggedNode = null;

    const cx = canvasW / 2;
    const cy = canvasH / 2;

    if (name === 'Tree') {
      // binary tree depth 4 — 15 nodes
      const positions = [];
      const depth = 4;
      const totalNodes = Math.pow(2, depth) - 1;
      const levelGap = (canvasH - 80) / depth;

      for (let i = 0; i < totalNodes; i++) {
        // level aur position calculate karo
        const level = Math.floor(Math.log2(i + 1));
        const posInLevel = i - (Math.pow(2, level) - 1);
        const nodesInLevel = Math.pow(2, level);
        const xSpacing = (canvasW - 100) / (nodesInLevel + 1);
        const nx = 50 + xSpacing * (posInLevel + 1) + (Math.random() - 0.5) * 20;
        const ny = 40 + level * levelGap + (Math.random() - 0.5) * 15;
        addNode(nx, ny);
      }

      // parent-child edges
      for (let i = 0; i < totalNodes; i++) {
        const leftChild = 2 * i + 1;
        const rightChild = 2 * i + 2;
        if (leftChild < totalNodes) addEdge(i, leftChild);
        if (rightChild < totalNodes) addEdge(i, rightChild);
      }

    } else if (name === 'Ring') {
      // 12 nodes circle mein
      const n = 12;
      const radius = Math.min(canvasW, canvasH) * 0.3;
      for (let i = 0; i < n; i++) {
        const angle = (2 * Math.PI * i) / n - Math.PI / 2;
        addNode(cx + radius * Math.cos(angle), cy + radius * Math.sin(angle));
      }
      for (let i = 0; i < n; i++) {
        addEdge(i, (i + 1) % n);
      }

    } else if (name === 'Mesh') {
      // 4x4 grid — adjacent cells connected
      const rows = 4, cols = 4;
      const spacingX = (canvasW - 160) / (cols - 1);
      const spacingY = (canvasH - 120) / (rows - 1);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          addNode(
            80 + c * spacingX + (Math.random() - 0.5) * 20,
            60 + r * spacingY + (Math.random() - 0.5) * 20
          );
        }
      }
      // horizontal aur vertical edges
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const idx = r * cols + c;
          if (c < cols - 1) addEdge(idx, idx + 1);        // right neighbor
          if (r < rows - 1) addEdge(idx, idx + cols);     // bottom neighbor
        }
      }

    } else if (name === 'Star') {
      // 1 center + 10 outer
      addNode(cx, cy); // center node
      const outerCount = 10;
      const radius = Math.min(canvasW, canvasH) * 0.3;
      for (let i = 0; i < outerCount; i++) {
        const angle = (2 * Math.PI * i) / outerCount - Math.PI / 2;
        addNode(cx + radius * Math.cos(angle), cy + radius * Math.sin(angle));
        addEdge(0, i + 1); // center se connect karo
      }

    } else {
      // Random — default preset, 15 nodes, 30% edge probability
      const n = 15;
      for (let i = 0; i < n; i++) {
        addNode(
          80 + Math.random() * (canvasW - 160),
          60 + Math.random() * (canvasH - 120)
        );
      }
      // random edges — 30% probability
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (Math.random() < 0.3) {
            addEdge(i, j);
          }
        }
      }
    }
  }

  // ============================================================
  // PHYSICS ENGINE — Coulomb repulsion + Hooke attraction + center gravity
  // ============================================================

  function physicsStep() {
    const n = nodes.length;
    if (n === 0) return;

    // forces array — har node ke liye fx, fy
    const fx = new Float64Array(n);
    const fy = new Float64Array(n);

    // --- Repulsion: saare node pairs ke beech Coulomb force ---
    // F = kRepel / d^2, direction: away from each other
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = nodes[j].x - nodes[i].x;
        let dy = nodes[j].y - nodes[i].y;
        let distSq = dx * dx + dy * dy;
        // bahut close hain toh minimum distance set karo — division by zero se bachne ke liye
        if (distSq < 1) distSq = 1;
        const dist = Math.sqrt(distSq);
        // repulsion force magnitude
        const force = kRepel / distSq;
        const forceX = (dx / dist) * force;
        const forceY = (dy / dist) * force;
        // Newton's third law — equal and opposite
        fx[i] -= forceX;
        fy[i] -= forceY;
        fx[j] += forceX;
        fy[j] += forceY;
      }
    }

    // --- Attraction: connected nodes ke beech Hooke spring ---
    // F = kSpring * (distance - restLength), direction: toward each other
    for (let e = 0; e < edges.length; e++) {
      const edge = edges[e];
      const si = edge.source;
      const ti = edge.target;
      const dx = nodes[ti].x - nodes[si].x;
      const dy = nodes[ti].y - nodes[si].y;
      let dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 0.1) dist = 0.1;
      // spring force — positive means attract, negative means repel
      const displacement = dist - restLength;
      const force = kSpring * displacement;
      const forceX = (dx / dist) * force;
      const forceY = (dy / dist) * force;
      fx[si] += forceX;
      fy[si] += forceY;
      fx[ti] -= forceX;
      fy[ti] -= forceY;
    }

    // --- Center gravity: halka sa pull toward canvas center ---
    const centerX = canvasW / 2;
    const centerY = canvasH / 2;
    for (let i = 0; i < n; i++) {
      fx[i] += (centerX - nodes[i].x) * kCenter;
      fy[i] += (centerY - nodes[i].y) * kCenter;
    }

    // --- Euler integration + damping ---
    for (let i = 0; i < n; i++) {
      const node = nodes[i];
      // pinned ya dragged node ko physics se chhod do
      if (node.pinned || node === draggedNode) continue;

      // velocity update — force add karo
      node.vx = (node.vx + fx[i]) * damping;
      node.vy = (node.vy + fy[i]) * damping;

      // max velocity clamp — zyada fast mat jaane do
      const speed = Math.sqrt(node.vx * node.vx + node.vy * node.vy);
      const maxSpeed = 8;
      if (speed > maxSpeed) {
        node.vx = (node.vx / speed) * maxSpeed;
        node.vy = (node.vy / speed) * maxSpeed;
      }

      // position update
      node.x += node.vx;
      node.y += node.vy;

      // boundary clamping — canvas ke andar rakh
      const r = getDisplayRadius(i);
      node.x = Math.max(r, Math.min(canvasW - r, node.x));
      node.y = Math.max(r, Math.min(canvasH - r, node.y));
    }
  }

  // ============================================================
  // RENDERING — nodes, edges, glow, labels sab draw karo
  // ============================================================

  function draw() {
    ctx.clearRect(0, 0, canvasW, canvasH);

    // background fill
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, canvasW, canvasH);

    // --- Edges draw karo ---
    for (let e = 0; e < edges.length; e++) {
      const edge = edges[e];
      const s = nodes[edge.source];
      const t = nodes[edge.target];
      if (!s || !t) continue;

      ctx.beginPath();
      // slight curve — quadratic bezier for visual appeal
      const midX = (s.x + t.x) / 2;
      const midY = (s.y + t.y) / 2;
      // perpendicular offset for curve — distance based
      const dx = t.x - s.x;
      const dy = t.y - s.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      // curve amount — zyada door toh zyada curve
      const curveOffset = Math.min(dist * 0.08, 15);
      // perpendicular direction
      const px = -dy / (dist || 1) * curveOffset;
      const py = dx / (dist || 1) * curveOffset;

      ctx.moveTo(s.x, s.y);
      ctx.quadraticCurveTo(midX + px, midY + py, t.x, t.y);
      ctx.strokeStyle = EDGE_COLOR;
      ctx.lineWidth = EDGE_WIDTH;
      ctx.stroke();
    }

    // --- Nodes draw karo ---
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const r = getDisplayRadius(i);
      const isSelected = selectedNode === i;

      // glow effect — shadowBlur se subtle glow
      ctx.save();
      ctx.shadowColor = node.color;
      ctx.shadowBlur = isSelected ? 20 : 10;

      // node circle fill
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fillStyle = node.color;
      ctx.globalAlpha = 0.85;
      ctx.fill();
      ctx.globalAlpha = 1.0;

      // border — selected pe bright, nahi toh subtle
      ctx.strokeStyle = isSelected ? '#ffffff' : 'rgba(255,255,255,0.3)';
      ctx.lineWidth = isSelected ? 2.5 : 1;
      ctx.stroke();

      ctx.restore(); // shadow reset

      // label — node ID dikhao
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(node.id), node.x, node.y);
      ctx.textBaseline = 'alphabetic';
    }

    // --- Node count aur edge count dikhao top-left mein ---
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(176,176,176,0.35)';
    ctx.textAlign = 'left';
    ctx.fillText('nodes: ' + nodes.length + '  edges: ' + edges.length, 10, 18);

    // --- Selected node indicator ---
    if (selectedNode !== null && selectedNode < nodes.length) {
      ctx.fillStyle = 'rgba(' + ACCENT_RGB + ',0.4)';
      ctx.textAlign = 'right';
      ctx.fillText('selected: ' + nodes[selectedNode].id + ' (click another to connect)', canvasW - 10, 18);
    }

    // --- Empty state hint ---
    if (nodes.length === 0) {
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(176,176,176,0.25)';
      ctx.textAlign = 'center';
      ctx.fillText('click anywhere to add nodes', canvasW / 2, canvasH / 2);
    }
  }

  // ============================================================
  // MOUSE / TOUCH INTERACTION
  // ============================================================

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvasW / rect.width),
      y: (clientY - rect.top) * (canvasH / rect.height),
    };
  }

  // kaunsa node hai mouse ke paas — radius check karo
  function findNodeAt(px, py) {
    for (let i = nodes.length - 1; i >= 0; i--) {
      const node = nodes[i];
      const r = getDisplayRadius(i);
      const dx = px - node.x;
      const dy = py - node.y;
      if (dx * dx + dy * dy <= (r + 4) * (r + 4)) {
        return i;
      }
    }
    return -1;
  }

  function handlePointerDown(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const nodeIdx = findNodeAt(pos.x, pos.y);

    // shift+click ya right click — node delete karo
    if (e.shiftKey || e.button === 2) {
      if (nodeIdx >= 0) {
        removeNode(nodeIdx);
      }
      return;
    }

    if (nodeIdx >= 0) {
      // node pe click — drag shuru karo ya edge connect karo
      draggedNode = nodes[nodeIdx];
      isDragging = true;
      dragStartX = pos.x;
      dragStartY = pos.y;
      dragMoved = false;
      canvas.style.cursor = 'grabbing';
    }
  }

  function handlePointerMove(e) {
    e.preventDefault();
    const pos = getCanvasPos(e);

    if (isDragging && draggedNode) {
      // check karo ki actually move hua hai threshold se zyada
      const movedDist = Math.sqrt(
        Math.pow(pos.x - dragStartX, 2) + Math.pow(pos.y - dragStartY, 2)
      );
      if (movedDist > DRAG_THRESHOLD) {
        dragMoved = true;
      }
      // node ko mouse position pe le jao
      draggedNode.x = pos.x;
      draggedNode.y = pos.y;
      draggedNode.vx = 0;
      draggedNode.vy = 0;
    } else {
      // hover pe cursor change karo
      const nodeIdx = findNodeAt(pos.x, pos.y);
      canvas.style.cursor = nodeIdx >= 0 ? 'grab' : 'crosshair';
    }
  }

  function handlePointerUp(e) {
    if (isDragging && draggedNode) {
      const draggedIdx = nodes.indexOf(draggedNode);

      if (!dragMoved) {
        // click tha, drag nahi — edge creation ya selection
        if (selectedNode !== null && selectedNode !== draggedIdx) {
          // doosra node click hua — edge banao
          addEdge(selectedNode, draggedIdx);
          selectedNode = null;
        } else if (selectedNode === draggedIdx) {
          // same node pe dubara click — deselect
          selectedNode = null;
        } else {
          // pehla node select karo
          selectedNode = draggedIdx;
        }
      } else {
        // drag tha — release karo, physics physics sambhal lega
        // selection reset mat karo drag pe
      }

      isDragging = false;
      draggedNode = null;
      canvas.style.cursor = 'grab';
    }
  }

  function handleClick(e) {
    // agar drag hua tha toh click ignore karo — pointerUp ne handle kar liya
    if (dragMoved) return;

    const pos = getCanvasPos(e);
    const nodeIdx = findNodeAt(pos.x, pos.y);

    // shift+click handled in pointerDown
    if (e.shiftKey) return;

    // empty space pe click — naya node banao
    if (nodeIdx < 0 && !isDragging) {
      addNode(pos.x, pos.y);
      selectedNode = null;
    }
  }

  // right-click context menu band karo
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const nodeIdx = findNodeAt(pos.x, pos.y);
    if (nodeIdx >= 0) {
      removeNode(nodeIdx);
    }
  });

  // mouse events
  canvas.addEventListener('mousedown', handlePointerDown);
  canvas.addEventListener('mousemove', handlePointerMove);
  canvas.addEventListener('mouseup', handlePointerUp);
  canvas.addEventListener('mouseleave', () => {
    if (isDragging) {
      isDragging = false;
      draggedNode = null;
      canvas.style.cursor = 'grab';
    }
  });
  canvas.addEventListener('click', handleClick);

  // double-click — node pin/unpin toggle
  canvas.addEventListener('dblclick', (e) => {
    e.preventDefault();
    const pos = getCanvasPos(e);
    const nodeIdx = findNodeAt(pos.x, pos.y);
    if (nodeIdx >= 0) {
      nodes[nodeIdx].pinned = !nodes[nodeIdx].pinned;
    }
  });

  // touch events
  canvas.addEventListener('touchstart', handlePointerDown, { passive: false });
  canvas.addEventListener('touchmove', handlePointerMove, { passive: false });
  canvas.addEventListener('touchend', (e) => {
    if (isDragging && draggedNode) {
      const draggedIdx = nodes.indexOf(draggedNode);
      if (!dragMoved) {
        if (selectedNode !== null && selectedNode !== draggedIdx) {
          addEdge(selectedNode, draggedIdx);
          selectedNode = null;
        } else if (selectedNode === draggedIdx) {
          selectedNode = null;
        } else {
          selectedNode = draggedIdx;
        }
      }
      isDragging = false;
      draggedNode = null;
    }
  });
  canvas.addEventListener('touchcancel', () => {
    isDragging = false;
    draggedNode = null;
  });

  // ============================================================
  // ANIMATION LOOP
  // ============================================================

  function animate() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }
    physicsStep();
    draw();
    animationId = requestAnimationFrame(animate);
  }

  function startAnimation() {
    if (isVisible) return;
    isVisible = true;
    resizeCanvas();
    animationId = requestAnimationFrame(animate);
  }

  function stopAnimation() {
    isVisible = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // --- IntersectionObserver — visible hone pe hi animate karo ---
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

  // tab switch pe bhi pause/resume karo
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopAnimation();
    } else {
      const rect = container.getBoundingClientRect();
      const inView = rect.top < window.innerHeight && rect.bottom > 0;
      if (inView) startAnimation();
    }
  });

  // --- Initial preset load karo taaki blank na dikhe ---
  loadPreset('Random');
  resizeCanvas();
  draw();
}
