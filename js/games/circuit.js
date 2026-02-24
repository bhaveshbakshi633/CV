// ============================================================
// Interactive Circuit Builder — grid pe components place karo, simulate karo
// Ohm's law + Kirchhoff's se current calculate karta hai
// Robotics portfolio ke liye banaya hai ye
// ============================================================

// yahi se sab shuru hota hai — container dhundho, canvas banao, circuit chalao
export function initCircuit() {
  const container = document.getElementById('circuitContainer');
  if (!container) {
    console.warn('circuitContainer nahi mila bhai, circuit builder skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const GRID_COLS = 20;
  const GRID_ROWS = 12;
  const CANVAS_HEIGHT = 380;
  const TOOLBAR_HEIGHT = 48;
  const DOT_SPEED_SCALE = 120; // current dots kitni tez chalein
  const OVERCURRENT_THRESHOLD = 0.05; // 50mA se zyada toh LED pop
  const LED_FORWARD_VOLTAGE = 2.0; // LED ka voltage drop
  const COMPONENT_COLORS = {
    wire: '#e0e0e0',
    resistor: '#f59e0b',
    led_red: '#ef4444',
    led_green: '#22c55e',
    led_blue: '#3b82f6',
    battery: '#60a5fa',
    switch_on: '#22c55e',
    switch_off: '#ef4444',
    current_dot: '#fde047',
    grid_point: 'rgba(255,255,255,0.12)',
    grid_hover: 'rgba(249,158,11,0.5)',
    highlight: 'rgba(249,158,11,0.3)',
  };

  // --- State ---
  // components ka list — har component mein type, position, value, connections
  let components = [];
  let selectedTool = null; // kaunsa tool select hai toolbar se
  let wireStart = null; // wire placement ke liye — pehla point
  let placementStart = null; // 2-terminal component ka first point
  let simulating = false; // simulation chal rahi hai ya nahi
  let animationId = null;
  let isVisible = false;
  let hoveredNode = null; // mouse kis grid node pe hai
  let hoveredComponent = null; // mouse kis component pe hai
  let lastTime = 0;
  let currentFlowPhase = 0; // animated dots ka phase

  // grid node positions cache — baar baar calculate nahi karna padega
  let gridPositions = [];
  let gridSpacingX = 0;
  let gridSpacingY = 0;
  let canvasOffsetX = 0;
  let canvasOffsetY = 0;

  // circuit solver ka result — har component pe voltage, current
  let solverResults = new Map();

  // component id counter — unique id dene ke liye
  let nextId = 1;

  // --- DOM setup ---
  // container saaf karo
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.style.cssText = 'width:100%;position:relative;';

  // toolbar banao — component buttons ke liye
  const toolbar = document.createElement('div');
  toolbar.style.cssText = [
    'display:flex',
    'gap:6px',
    'margin-bottom:8px',
    'align-items:center',
    'flex-wrap:wrap',
    'padding:6px 0',
  ].join(';');
  container.appendChild(toolbar);

  // canvas — yahan circuit dikhega
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(249,158,11,0.15)',
    'border-radius:8px',
    'cursor:crosshair',
    'background:transparent',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // controls — simulate toggle, clear, presets
  const controlsDiv = document.createElement('div');
  controlsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:8px',
    'margin-top:10px',
    'align-items:center',
  ].join(';');
  container.appendChild(controlsDiv);

  // tooltip/hover info
  const tooltip = document.createElement('div');
  tooltip.style.cssText = [
    'position:absolute',
    'background:rgba(20,20,20,0.95)',
    'color:#e0e0e0',
    'padding:4px 10px',
    'border-radius:6px',
    'font-size:12px',
    'font-family:monospace',
    'pointer-events:none',
    'opacity:0',
    'transition:opacity 0.15s',
    'border:1px solid rgba(249,158,11,0.3)',
    'z-index:10',
    'white-space:nowrap',
  ].join(';');
  container.appendChild(tooltip);

  // --- Toolbar buttons ---
  // har tool ka ek button — select karo toh highlight ho jaaye
  const tools = [
    { id: 'wire', label: '⏤ Wire', icon: '—' },
    { id: 'battery', label: '⚡ Battery', icon: '🔋' },
    { id: 'resistor', label: '⫘ Resistor', icon: 'R' },
    { id: 'led', label: '💡 LED', icon: 'D' },
    { id: 'switch', label: '⏻ Switch', icon: 'S' },
  ];

  const toolButtons = {};
  tools.forEach((tool) => {
    const btn = document.createElement('button');
    btn.textContent = tool.label;
    btn.dataset.tool = tool.id;
    btn.style.cssText = [
      'padding:5px 12px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'border:1px solid rgba(249,158,11,0.3)',
      'background:rgba(249,158,11,0.08)',
      'color:#b0b0b0',
      'transition:all 0.2s',
      'font-family:monospace',
    ].join(';');
    btn.addEventListener('click', () => selectTool(tool.id));
    toolbar.appendChild(btn);
    toolButtons[tool.id] = btn;
  });

  // toolbar mein ek separator daalo
  const sep = document.createElement('div');
  sep.style.cssText = 'width:1px;height:24px;background:rgba(249,158,11,0.2);margin:0 4px;';
  toolbar.appendChild(sep);

  // status text — kya ho raha hai batata hai
  const statusText = document.createElement('span');
  statusText.style.cssText = 'color:#888;font-size:12px;font-family:monospace;margin-left:auto;';
  statusText.textContent = 'Tool select karo ↑';
  toolbar.appendChild(statusText);

  // --- tool selection ---
  function selectTool(toolId) {
    selectedTool = toolId;
    wireStart = null;
    placementStart = null;
    // sab buttons reset karo, selected ko highlight karo
    Object.keys(toolButtons).forEach((id) => {
      if (id === toolId) {
        toolButtons[id].style.background = 'rgba(249,158,11,0.25)';
        toolButtons[id].style.color = '#f59e0b';
        toolButtons[id].style.borderColor = 'rgba(249,158,11,0.6)';
      } else {
        toolButtons[id].style.background = 'rgba(249,158,11,0.08)';
        toolButtons[id].style.color = '#b0b0b0';
        toolButtons[id].style.borderColor = 'rgba(249,158,11,0.3)';
      }
    });
    updateStatus();
  }

  function updateStatus() {
    if (!selectedTool) {
      statusText.textContent = 'Tool select karo ↑';
    } else if (selectedTool === 'wire' && wireStart) {
      statusText.textContent = 'Wire: end point click karo';
    } else if (placementStart) {
      statusText.textContent = selectedTool + ': end point click karo';
    } else if (selectedTool === 'wire') {
      statusText.textContent = 'Wire: start point click karo';
    } else {
      statusText.textContent = selectedTool + ': start point click karo';
    }
  }

  // --- Control buttons ---
  function createButton(label, onClick) {
    const btn = document.createElement('button');
    btn.textContent = label;
    btn.style.cssText = [
      'padding:5px 14px',
      'font-size:12px',
      'border-radius:6px',
      'cursor:pointer',
      'border:1px solid rgba(249,158,11,0.3)',
      'background:rgba(249,158,11,0.08)',
      'color:#b0b0b0',
      'transition:all 0.2s',
      'font-family:monospace',
    ].join(';');
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(249,158,11,0.2)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(249,158,11,0.08)';
    });
    btn.addEventListener('click', onClick);
    return btn;
  }

  // simulate toggle button
  const simBtn = createButton('▶ Simulate', () => {
    simulating = !simulating;
    simBtn.textContent = simulating ? '⏸ Stop' : '▶ Simulate';
    simBtn.style.color = simulating ? '#22c55e' : '#b0b0b0';
    if (simulating) {
      solveCircuit();
    }
  });
  controlsDiv.appendChild(simBtn);

  // clear all button
  const clearBtn = createButton('✕ Clear All', () => {
    components = [];
    solverResults.clear();
    simulating = false;
    simBtn.textContent = '▶ Simulate';
    simBtn.style.color = '#b0b0b0';
    wireStart = null;
    placementStart = null;
  });
  controlsDiv.appendChild(clearBtn);

  // separator
  const sep2 = document.createElement('div');
  sep2.style.cssText = 'width:1px;height:24px;background:rgba(249,158,11,0.2);margin:0 4px;';
  controlsDiv.appendChild(sep2);

  // preset label
  const presetLabel = document.createElement('span');
  presetLabel.style.cssText = 'color:#888;font-size:12px;font-family:monospace;';
  presetLabel.textContent = 'Presets:';
  controlsDiv.appendChild(presetLabel);

  // preset buttons — ready-made circuits load karte hain
  const presets = [
    { name: 'Series', build: buildSeriesPreset },
    { name: 'Parallel', build: buildParallelPreset },
    { name: 'Divider', build: buildVoltageDividerPreset },
  ];

  presets.forEach((preset) => {
    const btn = createButton(preset.name, () => {
      components = [];
      solverResults.clear();
      simulating = false;
      simBtn.textContent = '▶ Simulate';
      simBtn.style.color = '#b0b0b0';
      wireStart = null;
      placementStart = null;
      preset.build();
    });
    controlsDiv.appendChild(btn);
  });

  // --- Canvas resize handler ---
  // canvas ko container ki width match karna hai
  function resizeCanvas() {
    const rect = container.getBoundingClientRect();
    const w = Math.floor(rect.width);
    // high DPI ke liye pixel ratio use karo
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // grid spacing calculate karo
    const padX = 30;
    const padY = 20;
    gridSpacingX = (w - 2 * padX) / (GRID_COLS - 1);
    gridSpacingY = (CANVAS_HEIGHT - 2 * padY) / (GRID_ROWS - 1);
    canvasOffsetX = padX;
    canvasOffsetY = padY;

    // grid positions cache update karo
    gridPositions = [];
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        gridPositions.push({
          col: c,
          row: r,
          x: canvasOffsetX + c * gridSpacingX,
          y: canvasOffsetY + r * gridSpacingY,
        });
      }
    }
  }

  // --- Grid utility functions ---
  // mouse position se nearest grid node dhundho
  function nearestGridNode(mx, my) {
    let bestDist = Infinity;
    let best = null;
    for (const gp of gridPositions) {
      const dx = mx - gp.x;
      const dy = my - gp.y;
      const dist = dx * dx + dy * dy;
      if (dist < bestDist) {
        bestDist = dist;
        best = gp;
      }
    }
    // snap threshold — zyada door ho toh null
    const threshold = Math.max(gridSpacingX, gridSpacingY) * 0.6;
    if (bestDist > threshold * threshold) return null;
    return best;
  }

  // grid node key — unique identifier for a grid point
  function nodeKey(col, row) {
    return col + ',' + row;
  }

  // canvas mouse position nikalo
  function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }

  // --- Component creation ---
  // component banao aur list mein daalo
  function addComponent(type, startNode, endNode, value) {
    const comp = {
      id: nextId++,
      type: type,
      start: { col: startNode.col, row: startNode.row },
      end: { col: endNode.col, row: endNode.row },
      value: value,
      switchOn: true, // switch ke liye — default on
      ledColor: 'red', // LED ka color
      popped: false, // LED pop ho gaya ya nahi
    };
    components.push(comp);
    // agar simulate chal raha hai toh circuit solve karo phir se
    if (simulating) {
      solveCircuit();
    }
    return comp;
  }

  // --- Mouse event handlers ---
  canvas.addEventListener('mousemove', (e) => {
    const pos = getMousePos(e);
    hoveredNode = nearestGridNode(pos.x, pos.y);

    // hover pe component check karo — tooltip ke liye
    hoveredComponent = null;
    for (const comp of components) {
      if (isPointNearComponent(pos.x, pos.y, comp)) {
        hoveredComponent = comp;
        break;
      }
    }

    // tooltip update karo
    if (hoveredComponent && simulating && solverResults.has(hoveredComponent.id)) {
      const result = solverResults.get(hoveredComponent.id);
      const rect = canvas.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      tooltip.style.left = (e.clientX - containerRect.left + 12) + 'px';
      tooltip.style.top = (e.clientY - containerRect.top - 30) + 'px';
      tooltip.style.opacity = '1';
      tooltip.textContent = formatComponentInfo(hoveredComponent, result);
    } else {
      tooltip.style.opacity = '0';
    }
  });

  canvas.addEventListener('mouseleave', () => {
    hoveredNode = null;
    hoveredComponent = null;
    tooltip.style.opacity = '0';
  });

  canvas.addEventListener('click', (e) => {
    if (!selectedTool) return;
    const pos = getMousePos(e);
    const node = nearestGridNode(pos.x, pos.y);
    if (!node) return;

    // switch toggle — agar switch pe click kiya aur koi tool select nahi hai...
    // actually pehle check karo ki koi existing switch pe click toh nahi kiya
    for (const comp of components) {
      if (comp.type === 'switch' && isPointNearComponent(pos.x, pos.y, comp)) {
        comp.switchOn = !comp.switchOn;
        if (simulating) solveCircuit();
        return;
      }
    }

    // wire ya 2-terminal component placement
    if (selectedTool === 'wire' || selectedTool === 'resistor' || selectedTool === 'led' || selectedTool === 'battery' || selectedTool === 'switch') {
      if (!placementStart) {
        // pehla point — yaad rakh
        placementStart = { col: node.col, row: node.row };
        if (selectedTool === 'wire') wireStart = placementStart;
        updateStatus();
      } else {
        // doosra point — component place karo
        // same point pe dono end nahi ho sakte
        if (placementStart.col === node.col && placementStart.row === node.row) return;

        let value = getDefaultValue(selectedTool);
        const startN = { col: placementStart.col, row: placementStart.row };
        const endN = { col: node.col, row: node.row };
        addComponent(selectedTool, startN, endN, value);

        placementStart = null;
        wireStart = null;
        updateStatus();
      }
    }
  });

  // right click se cancel placement
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    placementStart = null;
    wireStart = null;
    updateStatus();
  });

  // switch toggle ke liye — canvas pe double click
  canvas.addEventListener('dblclick', (e) => {
    const pos = getMousePos(e);
    for (const comp of components) {
      if (comp.type === 'switch' && isPointNearComponent(pos.x, pos.y, comp)) {
        comp.switchOn = !comp.switchOn;
        if (simulating) solveCircuit();
        return;
      }
    }
  });

  // --- Default values ---
  function getDefaultValue(type) {
    switch (type) {
      case 'battery': return 9; // 9V battery
      case 'resistor': return 1000; // 1kΩ
      case 'led': return LED_FORWARD_VOLTAGE;
      case 'switch': return 0; // no resistance when on
      case 'wire': return 0; // ideal wire
      default: return 0;
    }
  }

  // --- Component hover detection ---
  // check karo mouse component ke paas hai ya nahi
  function isPointNearComponent(mx, my, comp) {
    const sx = canvasOffsetX + comp.start.col * gridSpacingX;
    const sy = canvasOffsetY + comp.start.row * gridSpacingY;
    const ex = canvasOffsetX + comp.end.col * gridSpacingX;
    const ey = canvasOffsetY + comp.end.row * gridSpacingY;

    // line segment se distance nikalo
    const dx = ex - sx;
    const dy = ey - sy;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return false;

    let t = ((mx - sx) * dx + (my - sy) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));

    const closestX = sx + t * dx;
    const closestY = sy + t * dy;
    const distSq = (mx - closestX) * (mx - closestX) + (my - closestY) * (my - closestY);

    return distSq < 144; // 12px radius
  }

  // --- Component info format karo tooltip ke liye ---
  function formatComponentInfo(comp, result) {
    if (!result) return comp.type;
    const v = result.voltage != null ? result.voltage.toFixed(2) + 'V' : '?V';
    const i = result.current != null ? (result.current * 1000).toFixed(1) + 'mA' : '?mA';
    switch (comp.type) {
      case 'battery': return 'Battery: ' + comp.value + 'V | ' + i;
      case 'resistor': return 'R=' + formatResistance(comp.value) + ' | ' + v + ' | ' + i;
      case 'led': return 'LED: Vf=' + comp.value.toFixed(1) + 'V | ' + i + (comp.popped ? ' [POPPED]' : '');
      case 'switch': return 'Switch: ' + (comp.switchOn ? 'ON' : 'OFF') + ' | ' + i;
      case 'wire': return 'Wire | ' + i;
      default: return comp.type;
    }
  }

  // resistance ko readable format mein banao
  function formatResistance(r) {
    if (r >= 1000) return (r / 1000).toFixed(1) + 'kΩ';
    return r + 'Ω';
  }

  // ============================================================
  // CIRCUIT SOLVER — Ohm's law aur Kirchhoff's use karke
  // simple series/parallel circuits solve karta hai
  // ============================================================
  function solveCircuit() {
    solverResults.clear();

    // graph banao — har grid node ek vertex, har component ek edge
    const graph = buildGraph();
    if (!graph) return;

    // sabse pehle battery dhundho
    const batteries = components.filter((c) => c.type === 'battery');
    if (batteries.length === 0) return;

    // har battery ke liye circuit trace karo
    for (const battery of batteries) {
      // battery ke positive terminal se negative terminal tak ka path dhundho
      const batteryStartKey = nodeKey(battery.start.col, battery.start.row);
      const batteryEndKey = nodeKey(battery.end.col, battery.end.row);

      // BFS se path dhundho — battery end se battery start tak
      // (conventional current battery + se - tak jaata hai, external circuit mein)
      const path = findPath(graph, batteryStartKey, batteryEndKey, battery.id);
      if (!path || path.length === 0) {
        // circuit complete nahi hai — koi current nahi
        continue;
      }

      // path ke components se total resistance nikalo
      let totalResistance = 0;
      let totalVoltageDrop = 0; // LEDs ka forward voltage
      let hasOpenSwitch = false;
      const pathComponents = [];

      for (const edge of path) {
        const comp = components.find((c) => c.id === edge.compId);
        if (!comp) continue;
        pathComponents.push(comp);

        if (comp.type === 'resistor') {
          totalResistance += comp.value;
        } else if (comp.type === 'led') {
          totalVoltageDrop += comp.value;
          totalResistance += 10; // LED ki thodi internal resistance
        } else if (comp.type === 'switch') {
          if (!comp.switchOn) {
            hasOpenSwitch = true;
          }
        }
        // wire ka 0 resistance hai
      }

      // agar koi switch off hai toh current zero
      if (hasOpenSwitch) {
        for (const comp of pathComponents) {
          solverResults.set(comp.id, { voltage: 0, current: 0 });
        }
        solverResults.set(battery.id, { voltage: battery.value, current: 0 });
        continue;
      }

      // minimum resistance — short circuit se bachao
      if (totalResistance < 1) totalResistance = 1;

      // effective voltage = battery voltage - LED forward drops
      let effectiveVoltage = battery.value - totalVoltageDrop;
      if (effectiveVoltage < 0) effectiveVoltage = 0;

      // Ohm's law: I = V / R
      const current = effectiveVoltage / totalResistance;

      // har component ke liye voltage aur current set karo
      for (const comp of pathComponents) {
        let compVoltage = 0;
        if (comp.type === 'resistor') {
          compVoltage = current * comp.value;
        } else if (comp.type === 'led') {
          compVoltage = comp.value;
          // overcurrent check — LED pop ho jaayega
          if (current > OVERCURRENT_THRESHOLD) {
            comp.popped = true;
          }
        }
        solverResults.set(comp.id, { voltage: compVoltage, current: current });
      }
      solverResults.set(battery.id, { voltage: battery.value, current: current });
    }

    // parallel circuit detection — simplified approach
    // agar same two nodes ke beech multiple paths hain toh parallel hai
    detectAndSolveParallel(graph, batteries);
  }

  // graph banao — adjacency list with component edges
  function buildGraph() {
    const adj = new Map();

    for (const comp of components) {
      const sk = nodeKey(comp.start.col, comp.start.row);
      const ek = nodeKey(comp.end.col, comp.end.row);

      if (!adj.has(sk)) adj.set(sk, []);
      if (!adj.has(ek)) adj.set(ek, []);

      // dono taraf edge daalo — undirected graph hai
      adj.get(sk).push({ node: ek, compId: comp.id });
      adj.get(ek).push({ node: sk, compId: comp.id });
    }

    return adj;
  }

  // BFS se path dhundho — start se end tak
  function findPath(graph, startKey, endKey, skipCompId) {
    if (startKey === endKey) return [];

    const visited = new Set();
    // queue mein {node, path} store karo
    const queue = [{ node: startKey, path: [] }];
    visited.add(startKey);

    while (queue.length > 0) {
      const { node, path } = queue.shift();
      const neighbors = graph.get(node) || [];

      for (const edge of neighbors) {
        if (edge.compId === skipCompId) {
          // battery khud ko skip karo — circuit ke through jaana hai
          if (path.length === 0) continue;
          // but end pe battery pe aa sakte hain
          if (edge.node === endKey) {
            return path;
          }
          continue;
        }
        if (visited.has(edge.node)) continue;

        const newPath = [...path, edge];
        if (edge.node === endKey) {
          return newPath;
        }

        visited.add(edge.node);
        queue.push({ node: edge.node, path: newPath });
      }
    }

    return null; // path nahi mila — circuit incomplete
  }

  // parallel circuit detection — simplified
  // same battery ke terminals ke beech multiple independent paths dhundho
  function detectAndSolveParallel(graph, batteries) {
    for (const battery of batteries) {
      const sk = nodeKey(battery.start.col, battery.start.row);
      const ek = nodeKey(battery.end.col, battery.end.row);

      // saare paths dhundho (max 5 — infinite loop se bachao)
      const allPaths = findAllPaths(graph, sk, ek, battery.id, 5);
      if (allPaths.length <= 1) continue;

      // parallel resistance calculate karo
      let invRTotal = 0;
      const pathResistances = [];

      for (const path of allPaths) {
        let r = 0;
        let vDrop = 0;
        let open = false;
        for (const edge of path) {
          const comp = components.find((c) => c.id === edge.compId);
          if (!comp) continue;
          if (comp.type === 'resistor') r += comp.value;
          else if (comp.type === 'led') { r += 10; vDrop += comp.value; }
          else if (comp.type === 'switch' && !comp.switchOn) open = true;
        }
        if (open) { pathResistances.push(Infinity); continue; }
        if (r < 1) r = 1;
        pathResistances.push(r);
        invRTotal += 1 / r;
      }

      if (invRTotal === 0) continue;

      const rTotal = 1 / invRTotal;
      const totalCurrent = battery.value / rTotal;

      // har path mein current distribute karo
      for (let i = 0; i < allPaths.length; i++) {
        if (pathResistances[i] === Infinity) continue;
        const pathCurrent = battery.value / pathResistances[i];
        for (const edge of allPaths[i]) {
          const comp = components.find((c) => c.id === edge.compId);
          if (!comp) continue;
          let v = 0;
          if (comp.type === 'resistor') v = pathCurrent * comp.value;
          else if (comp.type === 'led') {
            v = comp.value;
            if (pathCurrent > OVERCURRENT_THRESHOLD) comp.popped = true;
          }
          solverResults.set(comp.id, { voltage: v, current: pathCurrent });
        }
      }
      solverResults.set(battery.id, { voltage: battery.value, current: totalCurrent });
    }
  }

  // DFS se saare paths dhundho — maxPaths tak
  function findAllPaths(graph, startKey, endKey, skipCompId, maxPaths) {
    const results = [];
    const visited = new Set();
    visited.add(startKey);

    function dfs(node, path) {
      if (results.length >= maxPaths) return;
      const neighbors = graph.get(node) || [];

      for (const edge of neighbors) {
        if (edge.compId === skipCompId) {
          if (edge.node === endKey && path.length > 0) {
            results.push([...path]);
          }
          continue;
        }
        if (visited.has(edge.node)) continue;

        visited.add(edge.node);
        path.push(edge);

        if (edge.node === endKey) {
          results.push([...path]);
        } else {
          dfs(edge.node, path);
        }

        path.pop();
        visited.delete(edge.node);
      }
    }

    dfs(startKey, []);
    return results;
  }

  // ============================================================
  // PRESET CIRCUITS — ready-made circuits
  // ============================================================

  // Series circuit: Battery → Resistor → LED → Battery
  function buildSeriesPreset() {
    // battery left side pe, resistor beech mein, LED right mein
    // ek horizontal line banate hain row 5-6 pe
    const r = 5;
    addComponent('battery', { col: 2, row: r }, { col: 2, row: r + 4 }, 9);
    addComponent('wire', { col: 2, row: r }, { col: 6, row: r }, 0);
    addComponent('resistor', { col: 6, row: r }, { col: 10, row: r }, 470);
    addComponent('wire', { col: 10, row: r }, { col: 14, row: r }, 0);
    addComponent('led', { col: 14, row: r }, { col: 17, row: r }, LED_FORWARD_VOLTAGE);
    addComponent('wire', { col: 17, row: r }, { col: 17, row: r + 4 }, 0);
    addComponent('wire', { col: 17, row: r + 4 }, { col: 2, row: r + 4 }, 0);
  }

  // Parallel circuit: Battery ke dono ends se do alag paths
  function buildParallelPreset() {
    const r = 3;
    // battery left vertical
    addComponent('battery', { col: 2, row: r }, { col: 2, row: r + 6 }, 9);
    // top rail
    addComponent('wire', { col: 2, row: r }, { col: 8, row: r }, 0);
    // branch 1 — resistor 1kΩ
    addComponent('resistor', { col: 8, row: r }, { col: 8, row: r + 6 }, 1000);
    // branch 2 — resistor 2kΩ
    addComponent('wire', { col: 8, row: r }, { col: 14, row: r }, 0);
    addComponent('resistor', { col: 14, row: r }, { col: 14, row: r + 6 }, 2000);
    // bottom rail
    addComponent('wire', { col: 8, row: r + 6 }, { col: 2, row: r + 6 }, 0);
    addComponent('wire', { col: 14, row: r + 6 }, { col: 8, row: r + 6 }, 0);
  }

  // Voltage divider: Battery → R1 → middle tap → R2 → Ground
  function buildVoltageDividerPreset() {
    const r = 3;
    addComponent('battery', { col: 3, row: r }, { col: 3, row: r + 6 }, 9);
    addComponent('wire', { col: 3, row: r }, { col: 8, row: r }, 0);
    addComponent('resistor', { col: 8, row: r }, { col: 8, row: r + 3 }, 1000);
    // middle tap — yahan voltage measure hoga
    addComponent('resistor', { col: 8, row: r + 3 }, { col: 8, row: r + 6 }, 2000);
    addComponent('wire', { col: 8, row: r + 6 }, { col: 3, row: r + 6 }, 0);
    // indicator wire — voltage divider output
    addComponent('wire', { col: 8, row: r + 3 }, { col: 14, row: r + 3 }, 0);
    addComponent('led', { col: 14, row: r + 3 }, { col: 14, row: r + 6 }, LED_FORWARD_VOLTAGE);
    addComponent('wire', { col: 14, row: r + 6 }, { col: 8, row: r + 6 }, 0);
  }

  // ============================================================
  // RENDERING — sab kuch canvas pe draw karo
  // ============================================================
  function draw(timestamp) {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    if (!isVisible) {
      animationId = null;
      return;
    }

    const dt = lastTime ? (timestamp - lastTime) / 1000 : 1 / 60;
    lastTime = timestamp;

    // current flow animation phase update karo
    if (simulating) {
      currentFlowPhase += dt * 2;
      if (currentFlowPhase > 100) currentFlowPhase -= 100;
    }

    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = CANVAS_HEIGHT;

    ctx.clearRect(0, 0, w, h);

    // grid points draw karo
    drawGrid(w, h);

    // components draw karo
    for (const comp of components) {
      drawComponent(comp);
    }

    // current flow dots draw karo (sirf simulate mode mein)
    if (simulating) {
      for (const comp of components) {
        drawCurrentDots(comp);
      }
    }

    // placement preview — agar user component place kar raha hai
    if (placementStart && hoveredNode) {
      drawPlacementPreview();
    }

    // hovered grid node highlight karo
    if (hoveredNode && selectedTool) {
      ctx.beginPath();
      ctx.arc(hoveredNode.x, hoveredNode.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = COMPONENT_COLORS.grid_hover;
      ctx.fill();
    }

    animationId = requestAnimationFrame(draw);
  }

  // grid dots draw karo
  function drawGrid() {
    for (const gp of gridPositions) {
      ctx.beginPath();
      ctx.arc(gp.x, gp.y, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = COMPONENT_COLORS.grid_point;
      ctx.fill();
    }
  }

  // ek component draw karo — type ke hisaab se different shape
  function drawComponent(comp) {
    const sx = canvasOffsetX + comp.start.col * gridSpacingX;
    const sy = canvasOffsetY + comp.start.row * gridSpacingY;
    const ex = canvasOffsetX + comp.end.col * gridSpacingX;
    const ey = canvasOffsetY + comp.end.row * gridSpacingY;

    // component direction vector
    const dx = ex - sx;
    const dy = ey - sy;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len === 0) return;
    const nx = dx / len;
    const ny = dy / len;

    // hover highlight
    const isHovered = hoveredComponent && hoveredComponent.id === comp.id;

    switch (comp.type) {
      case 'wire':
        drawWire(sx, sy, ex, ey, isHovered);
        break;
      case 'battery':
        drawBattery(sx, sy, ex, ey, nx, ny, len, comp, isHovered);
        break;
      case 'resistor':
        drawResistor(sx, sy, ex, ey, nx, ny, len, comp, isHovered);
        break;
      case 'led':
        drawLED(sx, sy, ex, ey, nx, ny, len, comp, isHovered);
        break;
      case 'switch':
        drawSwitch(sx, sy, ex, ey, nx, ny, len, comp, isHovered);
        break;
    }
  }

  // --- Wire draw ---
  function drawWire(sx, sy, ex, ey, isHovered) {
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = isHovered ? '#ffffff' : COMPONENT_COLORS.wire;
    ctx.lineWidth = isHovered ? 2.5 : 1.5;
    ctx.stroke();

    // endpoints pe dots
    drawEndpoint(sx, sy);
    drawEndpoint(ex, ey);
  }

  // --- Battery draw — do parallel lines (badi aur chhoti) ---
  function drawBattery(sx, sy, ex, ey, nx, ny, len, comp, isHovered) {
    // lead wires
    const midX = (sx + ex) / 2;
    const midY = (sy + ey) / 2;
    const plateGap = 8;
    const bigPlateLen = 14;
    const smallPlateLen = 8;

    // perpendicular direction
    const px = -ny;
    const py = nx;

    // wire start se plate tak
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(midX - nx * plateGap, midY - ny * plateGap);
    ctx.strokeStyle = isHovered ? '#93c5fd' : COMPONENT_COLORS.battery;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // badi plate (positive)
    ctx.beginPath();
    ctx.moveTo(midX - nx * plateGap + px * bigPlateLen, midY - ny * plateGap + py * bigPlateLen);
    ctx.lineTo(midX - nx * plateGap - px * bigPlateLen, midY - ny * plateGap - py * bigPlateLen);
    ctx.strokeStyle = isHovered ? '#93c5fd' : COMPONENT_COLORS.battery;
    ctx.lineWidth = 3;
    ctx.stroke();

    // + sign
    ctx.font = 'bold 10px monospace';
    ctx.fillStyle = '#93c5fd';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('+', midX - nx * plateGap + px * (bigPlateLen + 8), midY - ny * plateGap + py * (bigPlateLen + 8));

    // chhoti plate (negative)
    ctx.beginPath();
    ctx.moveTo(midX + nx * plateGap + px * smallPlateLen, midY + ny * plateGap + py * smallPlateLen);
    ctx.lineTo(midX + nx * plateGap - px * smallPlateLen, midY + ny * plateGap - py * smallPlateLen);
    ctx.strokeStyle = isHovered ? '#93c5fd' : COMPONENT_COLORS.battery;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // wire plate se end tak
    ctx.beginPath();
    ctx.moveTo(midX + nx * plateGap, midY + ny * plateGap);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = isHovered ? '#93c5fd' : COMPONENT_COLORS.battery;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // voltage label
    ctx.font = '10px monospace';
    ctx.fillStyle = '#93c5fd';
    ctx.textAlign = 'center';
    ctx.fillText(comp.value + 'V', midX + px * 20, midY + py * 20);

    drawEndpoint(sx, sy);
    drawEndpoint(ex, ey);
  }

  // --- Resistor draw — zigzag pattern ---
  function drawResistor(sx, sy, ex, ey, nx, ny, len, comp, isHovered) {
    const midX = (sx + ex) / 2;
    const midY = (sy + ey) / 2;
    const zigLen = Math.min(len * 0.5, 40); // zigzag section ki length
    const zigCount = 5;
    const zigAmp = 6; // zigzag ka amplitude

    // perpendicular direction
    const px = -ny;
    const py = nx;

    // lead wire — start se zigzag tak
    const zigStart = midX - nx * zigLen;
    const zigStartY = midY - ny * zigLen;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(zigStart, zigStartY);
    ctx.strokeStyle = isHovered ? '#fbbf24' : COMPONENT_COLORS.resistor;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // zigzag draw karo
    ctx.beginPath();
    ctx.moveTo(zigStart, zigStartY);
    for (let i = 0; i <= zigCount * 2; i++) {
      const t = i / (zigCount * 2);
      const cx = zigStart + nx * (zigLen * 2) * t;
      const cy = zigStartY + ny * (zigLen * 2) * t;
      const amp = (i % 2 === 0 ? 1 : -1) * zigAmp;
      ctx.lineTo(cx + px * amp, cy + py * amp);
    }
    const zigEnd = midX + nx * zigLen;
    const zigEndY = midY + ny * zigLen;
    ctx.lineTo(zigEnd, zigEndY);
    ctx.strokeStyle = isHovered ? '#fbbf24' : COMPONENT_COLORS.resistor;
    ctx.lineWidth = 2;
    ctx.stroke();

    // lead wire — zigzag se end tak
    ctx.beginPath();
    ctx.moveTo(zigEnd, zigEndY);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = isHovered ? '#fbbf24' : COMPONENT_COLORS.resistor;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // resistance label
    ctx.font = '10px monospace';
    ctx.fillStyle = isHovered ? '#fbbf24' : '#d97706';
    ctx.textAlign = 'center';
    ctx.fillText(formatResistance(comp.value), midX + px * 16, midY + py * 16);

    drawEndpoint(sx, sy);
    drawEndpoint(ex, ey);
  }

  // --- LED draw — triangle + lines ---
  function drawLED(sx, sy, ex, ey, nx, ny, len, comp, isHovered) {
    const midX = (sx + ex) / 2;
    const midY = (sy + ey) / 2;
    const px = -ny;
    const py = nx;
    const size = 10;

    // lead wires
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(midX - nx * size, midY - ny * size);
    ctx.strokeStyle = COMPONENT_COLORS.wire;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(midX + nx * size, midY + ny * size);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = COMPONENT_COLORS.wire;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // LED color decide karo
    const ledColors = ['#ef4444', '#22c55e', '#3b82f6'];
    const colorIdx = comp.id % 3;
    let color = ledColors[colorIdx];
    comp.ledColor = ['red', 'green', 'blue'][colorIdx];

    // agar simulate ho raha hai aur current flow ho rahi hai toh glow karo
    const result = solverResults.get(comp.id);
    let glowing = false;
    let brightness = 0;
    if (simulating && result && result.current > 0.0001 && !comp.popped) {
      glowing = true;
      brightness = Math.min(result.current / 0.03, 1); // 30mA pe full brightness
    }

    if (comp.popped) {
      color = '#555'; // popped LED dim ho jaayega
    }

    // LED circle draw karo
    ctx.beginPath();
    ctx.arc(midX, midY, size, 0, Math.PI * 2);
    if (glowing) {
      // glow effect — radial gradient
      const grad = ctx.createRadialGradient(midX, midY, 0, midX, midY, size * 2.5);
      grad.addColorStop(0, color);
      grad.addColorStop(0.5, color + '80');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.fill();
      // inner circle
      ctx.beginPath();
      ctx.arc(midX, midY, size, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.5 + brightness * 0.5;
      ctx.fill();
      ctx.globalAlpha = 1.0;
    } else {
      ctx.fillStyle = color + '30';
      ctx.fill();
    }
    ctx.strokeStyle = isHovered ? '#ffffff' : color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // D label inside
    ctx.font = 'bold 10px monospace';
    ctx.fillStyle = comp.popped ? '#888' : color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(comp.popped ? '✕' : 'D', midX, midY);

    drawEndpoint(sx, sy);
    drawEndpoint(ex, ey);
  }

  // --- Switch draw ---
  function drawSwitch(sx, sy, ex, ey, nx, ny, len, comp, isHovered) {
    const midX = (sx + ex) / 2;
    const midY = (sy + ey) / 2;
    const gapLen = 12;

    // lead wire — start
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(midX - nx * gapLen, midY - ny * gapLen);
    ctx.strokeStyle = COMPONENT_COLORS.wire;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    if (comp.switchOn) {
      // closed switch — seedhi line
      ctx.beginPath();
      ctx.moveTo(midX - nx * gapLen, midY - ny * gapLen);
      ctx.lineTo(midX + nx * gapLen, midY + ny * gapLen);
      ctx.strokeStyle = isHovered ? '#4ade80' : COMPONENT_COLORS.switch_on;
      ctx.lineWidth = 2.5;
      ctx.stroke();
    } else {
      // open switch — tirchi line (gap dikhao)
      const px = -ny;
      const py = nx;
      ctx.beginPath();
      ctx.moveTo(midX - nx * gapLen, midY - ny * gapLen);
      ctx.lineTo(midX + nx * gapLen * 0.3 + px * 10, midY + ny * gapLen * 0.3 + py * 10);
      ctx.strokeStyle = isHovered ? '#f87171' : COMPONENT_COLORS.switch_off;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // lead wire — end
    ctx.beginPath();
    ctx.moveTo(midX + nx * gapLen, midY + ny * gapLen);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = COMPONENT_COLORS.wire;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // contact dots
    ctx.beginPath();
    ctx.arc(midX - nx * gapLen, midY - ny * gapLen, 3, 0, Math.PI * 2);
    ctx.fillStyle = comp.switchOn ? COMPONENT_COLORS.switch_on : COMPONENT_COLORS.switch_off;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(midX + nx * gapLen, midY + ny * gapLen, 3, 0, Math.PI * 2);
    ctx.fillStyle = comp.switchOn ? COMPONENT_COLORS.switch_on : COMPONENT_COLORS.switch_off;
    ctx.fill();

    // label
    ctx.font = '10px monospace';
    const px = -ny;
    const py = nx;
    ctx.fillStyle = comp.switchOn ? '#4ade80' : '#f87171';
    ctx.textAlign = 'center';
    ctx.fillText(comp.switchOn ? 'ON' : 'OFF', midX + px * 16, midY + py * 16);

    drawEndpoint(sx, sy);
    drawEndpoint(ex, ey);
  }

  // --- Endpoint dot ---
  function drawEndpoint(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.fill();
  }

  // --- Current flow dots — animated yellow dots wire ke upar ---
  function drawCurrentDots(comp) {
    const result = solverResults.get(comp.id);
    if (!result || result.current < 0.0001) return;
    if (comp.type === 'switch' && !comp.switchOn) return;

    const sx = canvasOffsetX + comp.start.col * gridSpacingX;
    const sy = canvasOffsetY + comp.start.row * gridSpacingY;
    const ex = canvasOffsetX + comp.end.col * gridSpacingX;
    const ey = canvasOffsetY + comp.end.row * gridSpacingY;

    const dx = ex - sx;
    const dy = ey - sy;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1) return;

    // current magnitude se speed aur dot count decide karo
    const speed = Math.min(result.current * DOT_SPEED_SCALE, 3);
    const dotCount = Math.max(2, Math.min(8, Math.floor(len / 20)));

    for (let i = 0; i < dotCount; i++) {
      // phase offset har dot ke liye alag
      let t = ((currentFlowPhase * speed + i / dotCount) % 1);
      const px = sx + dx * t;
      const py = sy + dy * t;

      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = COMPONENT_COLORS.current_dot;
      ctx.globalAlpha = 0.7 + Math.sin(t * Math.PI) * 0.3;
      ctx.fill();
      ctx.globalAlpha = 1.0;
    }
  }

  // --- Placement preview — ghost component dikhao jab user place kar raha hai ---
  function drawPlacementPreview() {
    if (!placementStart || !hoveredNode) return;
    const sx = canvasOffsetX + placementStart.col * gridSpacingX;
    const sy = canvasOffsetY + placementStart.row * gridSpacingY;
    const ex = hoveredNode.x;
    const ey = hoveredNode.y;

    ctx.save();
    ctx.globalAlpha = 0.4;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // start point highlight
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(249,158,11,0.5)';
    ctx.fill();
  }

  // ============================================================
  // INTERSECTION OBSERVER — sirf visible hone pe animate karo
  // ============================================================
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        isVisible = entry.isIntersecting;
        if (isVisible && !animationId) {
          lastTime = 0;
          animationId = requestAnimationFrame(draw);
        }
      });
    },
    { threshold: 0.1 }
  );
  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) draw(); });

  // --- Resize listener ---
  const resizeObs = new ResizeObserver(() => {
    resizeCanvas();
  });
  resizeObs.observe(container);

  // initial setup
  resizeCanvas();
  animationId = requestAnimationFrame(draw);
}
