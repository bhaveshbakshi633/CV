// ============================================================
// Sorting Algorithm Visualizer — 6 classic sorts as generators
// Bars upar neeche hilte hain, comparisons/swaps count hote hain
// Generator pattern se har step yield hota hai — smooth animation
// ============================================================

// yahi entry point hai — container pakdo, canvas banao, sorting shuru karo
export function initSorting() {
  const container = document.getElementById('sortingContainer');
  if (!container) {
    console.warn('sortingContainer nahi mila bhai, sorting visualizer skip kar rahe hain');
    return;
  }

  // --- Constants ---
  const CANVAS_HEIGHT = 350;
  const ACCENT = '#a78bfa';           // purple accent for controls
  const BAR_DEFAULT = '#4a9eff';      // neela — default bar color
  const BAR_COMPARE = '#facc15';      // peela — comparing
  const BAR_SWAP = '#ef4444';         // laal — swapping
  const BAR_SORTED = '#22c55e';       // hara — sorted position
  const BAR_PIVOT = '#f97316';        // orange — pivot (quicksort)
  const FONT = "'JetBrains Mono', monospace";

  // algorithm metadata — naam aur complexity
  const ALGORITHMS = {
    bubble:    { name: 'Bubble Sort',    time: 'O(n\u00B2)',      space: 'O(1)',     fn: bubbleSort },
    selection: { name: 'Selection Sort', time: 'O(n\u00B2)',      space: 'O(1)',     fn: selectionSort },
    insertion: { name: 'Insertion Sort', time: 'O(n\u00B2)',      space: 'O(1)',     fn: insertionSort },
    merge:     { name: 'Merge Sort',     time: 'O(n log n)', space: 'O(n)',     fn: mergeSort },
    quick:     { name: 'Quick Sort',     time: 'O(n log n)', space: 'O(log n)', fn: quickSort },
    heap:      { name: 'Heap Sort',      time: 'O(n log n)', space: 'O(1)',     fn: heapSort },
    shell:     { name: 'Shell Sort',     time: 'O(n\u00B9\u00B7\u00B3)', space: 'O(1)',     fn: shellSort },
    cocktail:  { name: 'Cocktail Shaker',time: 'O(n\u00B2)',      space: 'O(1)',     fn: cocktailSort },
    comb:      { name: 'Comb Sort',      time: 'O(n\u00B2)',      space: 'O(1)',     fn: combSort },
    counting:  { name: 'Counting Sort',  time: 'O(n+k)',     space: 'O(k)',     fn: countingSort },
    radix:     { name: 'Radix Sort',     time: 'O(nk)',      space: 'O(n+k)',   fn: radixSort },
    tim:       { name: 'Tim Sort',       time: 'O(n log n)', space: 'O(n)',     fn: timSort },
  };

  // initial order presets
  const ORDER_TYPES = ['Random', 'Nearly Sorted', 'Reversed', 'Few Unique'];

  // --- State ---
  let canvasW = 0, canvasH = 0;
  let dpr = 1;
  let arr = [];                    // actual array of values
  let barColors = [];              // har bar ka current color
  let arraySize = 50;              // default 50 bars
  let currentAlgo = 'merge';       // default merge sort
  let currentOrder = 'Random';
  let speed = 50;                  // ms delay between steps (1-200)
  let comparisons = 0;
  let swaps = 0;
  let isSorting = false;           // sorting chal rahi hai ya nahi
  let generator = null;            // current sort generator
  let sortTimer = null;            // setTimeout id for stepping
  let isVisible = false;
  let animationId = null;
  let sortedIndices = new Set();   // sorted positions track karne ke liye

  // --- DOM structure banao ---
  while (container.firstChild) container.removeChild(container.firstChild);
  container.style.cssText = 'width:100%;position:relative;';

  // main canvas — bars yahan dikhenge
  const canvas = document.createElement('canvas');
  canvas.style.cssText = [
    'width:100%',
    'height:' + CANVAS_HEIGHT + 'px',
    'display:block',
    'border:1px solid rgba(167,139,250,0.2)',
    'border-radius:8px',
    'background:rgba(2,2,8,0.5)',
  ].join(';');
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // --- Info bar — algorithm name + complexity ---
  const infoDiv = document.createElement('div');
  infoDiv.style.cssText = [
    'display:flex',
    'justify-content:space-between',
    'align-items:center',
    'padding:6px 10px',
    'margin-top:6px',
    'font-family:' + FONT,
    'font-size:12px',
    'color:#94a3b8',
    'background:rgba(167,139,250,0.06)',
    'border:1px solid rgba(167,139,250,0.12)',
    'border-radius:6px',
  ].join(';');
  container.appendChild(infoDiv);

  const algoNameSpan = document.createElement('span');
  algoNameSpan.style.cssText = 'color:#e2e8f0;font-weight:600;';
  infoDiv.appendChild(algoNameSpan);

  const complexitySpan = document.createElement('span');
  complexitySpan.style.cssText = 'color:#94a3b8;';
  infoDiv.appendChild(complexitySpan);

  const countersSpan = document.createElement('span');
  countersSpan.style.cssText = 'color:#94a3b8;';
  infoDiv.appendChild(countersSpan);

  // --- Algorithm selector buttons ---
  const algoBtnDiv = document.createElement('div');
  algoBtnDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:6px',
    'margin-top:8px',
  ].join(';');
  container.appendChild(algoBtnDiv);

  const algoButtons = {};
  for (const key of Object.keys(ALGORITHMS)) {
    const btn = document.createElement('button');
    btn.textContent = ALGORITHMS[key].name;
    btn.dataset.algo = key;
    btn.style.cssText = makeButtonCSS(false);
    btn.addEventListener('click', () => {
      if (isSorting) return; // sorting ke dauran switch nahi hoga
      selectAlgorithm(key);
    });
    algoBtnDiv.appendChild(btn);
    algoButtons[key] = btn;
  }

  // --- Controls row 1: sliders ---
  const slidersDiv = document.createElement('div');
  slidersDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:16px',
    'align-items:center',
    'margin-top:8px',
    'font-family:' + FONT,
    'font-size:11px',
    'color:#94a3b8',
  ].join(';');
  container.appendChild(slidersDiv);

  // array size slider
  const sizeGroup = makeSliderGroup('Size', 20, 200, arraySize, (v) => {
    if (isSorting) return;
    arraySize = v;
    generateArray();
    requestDraw();
  });
  slidersDiv.appendChild(sizeGroup.wrapper);

  // speed slider — 1ms (fast) to 200ms (slow)
  // slider value 1-200, but UX wise left=slow, right=fast
  // so invert: slider 1=slow(200ms), slider 200=fast(1ms)
  const speedGroup = makeSliderGroup('Speed', 1, 200, 201 - speed, (v) => {
    speed = 201 - v; // invert — slider right means faster
    speedGroup.label.textContent = 'Speed: ' + v;
  });
  slidersDiv.appendChild(speedGroup.wrapper);

  // --- Controls row 2: order selector + action buttons ---
  const actionsDiv = document.createElement('div');
  actionsDiv.style.cssText = [
    'display:flex',
    'flex-wrap:wrap',
    'gap:6px',
    'align-items:center',
    'margin-top:8px',
  ].join(';');
  container.appendChild(actionsDiv);

  // order selector dropdown
  const orderSelect = document.createElement('select');
  orderSelect.style.cssText = [
    'padding:5px 8px',
    'background:rgba(167,139,250,0.08)',
    'color:#e2e8f0',
    'border:1px solid rgba(167,139,250,0.25)',
    'border-radius:6px',
    'font-family:' + FONT,
    'font-size:11px',
    'cursor:pointer',
    'outline:none',
  ].join(';');
  for (const o of ORDER_TYPES) {
    const opt = document.createElement('option');
    opt.value = o;
    opt.textContent = o;
    if (o === currentOrder) opt.selected = true;
    orderSelect.appendChild(opt);
  }
  orderSelect.addEventListener('change', () => {
    if (isSorting) return;
    currentOrder = orderSelect.value;
    generateArray();
    requestDraw();
  });
  actionsDiv.appendChild(orderSelect);

  // sort button
  const sortBtn = document.createElement('button');
  sortBtn.textContent = '\u25B6 Sort';
  sortBtn.style.cssText = makeActionButtonCSS();
  sortBtn.addEventListener('click', () => {
    if (isSorting) {
      stopSorting();
    } else {
      startSorting();
    }
  });
  actionsDiv.appendChild(sortBtn);

  // shuffle button
  const shuffleBtn = document.createElement('button');
  shuffleBtn.textContent = '\u21BB Shuffle';
  shuffleBtn.style.cssText = makeActionButtonCSS();
  shuffleBtn.addEventListener('click', () => {
    if (isSorting) stopSorting();
    generateArray();
    requestDraw();
  });
  actionsDiv.appendChild(shuffleBtn);

  // ============================================================
  // SORTING ALGORITHMS — Generator functions
  // har step pe yield karte hain taaki animation ho sake
  // ============================================================

  // --- Bubble Sort ---
  // sabse simple — adjacent elements compare karo, swap karo agar galat order mein
  function* bubbleSort(a) {
    const n = a.length;
    for (let i = 0; i < n - 1; i++) {
      let swapped = false;
      for (let j = 0; j < n - i - 1; j++) {
        yield { type: 'compare', indices: [j, j + 1] };
        if (a[j] > a[j + 1]) {
          [a[j], a[j + 1]] = [a[j + 1], a[j]];
          yield { type: 'swap', indices: [j, j + 1] };
          swapped = true;
        }
      }
      // last element is now sorted
      yield { type: 'sorted', indices: [n - i - 1] };
      if (!swapped) {
        // already sorted — baaki sab bhi mark karo
        for (let k = 0; k < n - i - 1; k++) {
          yield { type: 'sorted', indices: [k] };
        }
        break;
      }
    }
    // pehla element bhi sorted hai
    yield { type: 'sorted', indices: [0] };
  }

  // --- Selection Sort ---
  // minimum dhundho unsorted part mein, swap karo correct position pe
  function* selectionSort(a) {
    const n = a.length;
    for (let i = 0; i < n - 1; i++) {
      let minIdx = i;
      for (let j = i + 1; j < n; j++) {
        yield { type: 'compare', indices: [minIdx, j] };
        if (a[j] < a[minIdx]) {
          minIdx = j;
        }
      }
      if (minIdx !== i) {
        [a[i], a[minIdx]] = [a[minIdx], a[i]];
        yield { type: 'swap', indices: [i, minIdx] };
      }
      yield { type: 'sorted', indices: [i] };
    }
    yield { type: 'sorted', indices: [n - 1] };
  }

  // --- Insertion Sort ---
  // ek ek element uthao, sahi jagah daalo sorted part mein
  function* insertionSort(a) {
    const n = a.length;
    yield { type: 'sorted', indices: [0] };
    for (let i = 1; i < n; i++) {
      const key = a[i];
      let j = i - 1;
      yield { type: 'compare', indices: [i, j] };
      while (j >= 0 && a[j] > key) {
        a[j + 1] = a[j];
        yield { type: 'swap', indices: [j, j + 1] };
        j--;
        if (j >= 0) {
          yield { type: 'compare', indices: [j, j + 1] };
        }
      }
      a[j + 1] = key;
      // i tak sab sorted hai
      for (let k = 0; k <= i; k++) {
        yield { type: 'sorted', indices: [k] };
      }
    }
  }

  // --- Merge Sort ---
  // divide and conquer — recursively split karo, merge karo sorted halves
  function* mergeSort(a, left, right) {
    if (left === undefined) { left = 0; right = a.length - 1; }
    if (left >= right) {
      if (left === right) yield { type: 'sorted', indices: [left] };
      return;
    }

    const mid = Math.floor((left + right) / 2);
    yield* mergeSort(a, left, mid);
    yield* mergeSort(a, mid + 1, right);
    yield* merge(a, left, mid, right);
  }

  function* merge(a, left, mid, right) {
    const leftArr = a.slice(left, mid + 1);
    const rightArr = a.slice(mid + 1, right + 1);
    let i = 0, j = 0, k = left;

    while (i < leftArr.length && j < rightArr.length) {
      yield { type: 'compare', indices: [left + i, mid + 1 + j] };
      if (leftArr[i] <= rightArr[j]) {
        a[k] = leftArr[i];
        i++;
      } else {
        a[k] = rightArr[j];
        j++;
      }
      yield { type: 'overwrite', indices: [k] };
      k++;
    }

    while (i < leftArr.length) {
      a[k] = leftArr[i];
      yield { type: 'overwrite', indices: [k] };
      i++;
      k++;
    }

    while (j < rightArr.length) {
      a[k] = rightArr[j];
      yield { type: 'overwrite', indices: [k] };
      j++;
      k++;
    }

    // merge ho gaya — ye range sorted hai
    for (let m = left; m <= right; m++) {
      if (left === 0 && right === a.length - 1) {
        yield { type: 'sorted', indices: [m] };
      }
    }
  }

  // --- Quick Sort (Lomuto partition) ---
  // pivot last element, partition karo, recursively sort karo
  function* quickSort(a, low, high) {
    if (low === undefined) { low = 0; high = a.length - 1; }
    if (low >= high) {
      if (low === high) yield { type: 'sorted', indices: [low] };
      return;
    }

    // Lomuto partition
    const pivot = a[high];
    yield { type: 'pivot', indices: [high] };
    let i = low - 1;

    for (let j = low; j < high; j++) {
      yield { type: 'compare', indices: [j, high] };
      if (a[j] <= pivot) {
        i++;
        if (i !== j) {
          [a[i], a[j]] = [a[j], a[i]];
          yield { type: 'swap', indices: [i, j] };
        }
      }
    }

    // pivot ko sahi jagah daalo
    i++;
    if (i !== high) {
      [a[i], a[high]] = [a[high], a[i]];
      yield { type: 'swap', indices: [i, high] };
    }
    yield { type: 'sorted', indices: [i] };

    yield* quickSort(a, low, i - 1);
    yield* quickSort(a, i + 1, high);
  }

  // --- Heap Sort ---
  // max-heap banao, ek ek element extract karo top se
  function* heapSort(a) {
    const n = a.length;

    // max-heap build karo — bottom-up heapify
    for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
      yield* heapify(a, n, i);
    }

    // ek ek element extract — top (max) ko end pe daalo
    for (let i = n - 1; i > 0; i--) {
      [a[0], a[i]] = [a[i], a[0]];
      yield { type: 'swap', indices: [0, i] };
      yield { type: 'sorted', indices: [i] };
      yield* heapify(a, i, 0);
    }
    yield { type: 'sorted', indices: [0] };
  }

  function* heapify(a, heapSize, rootIdx) {
    let largest = rootIdx;
    const left = 2 * rootIdx + 1;
    const right = 2 * rootIdx + 2;

    if (left < heapSize) {
      yield { type: 'compare', indices: [left, largest] };
      if (a[left] > a[largest]) {
        largest = left;
      }
    }

    if (right < heapSize) {
      yield { type: 'compare', indices: [right, largest] };
      if (a[right] > a[largest]) {
        largest = right;
      }
    }

    if (largest !== rootIdx) {
      [a[rootIdx], a[largest]] = [a[largest], a[rootIdx]];
      yield { type: 'swap', indices: [rootIdx, largest] };
      yield* heapify(a, heapSize, largest);
    }
  }

  // --- Shell Sort ---
  // insertion sort but gap sequence se — pehle bade gaps, fir chhote
  // Knuth sequence: gap = 3*gap + 1
  function* shellSort(a) {
    const n = a.length;
    // Knuth sequence se gaps generate karo
    let gap = 1;
    while (gap < Math.floor(n / 3)) gap = gap * 3 + 1;

    while (gap >= 1) {
      for (let i = gap; i < n; i++) {
        const temp = a[i];
        let j = i;
        yield { type: 'compare', indices: [j, j - gap] };
        while (j >= gap && a[j - gap] > temp) {
          a[j] = a[j - gap];
          yield { type: 'swap', indices: [j, j - gap] };
          j -= gap;
          if (j >= gap) yield { type: 'compare', indices: [j, j - gap] };
        }
        a[j] = temp;
      }
      gap = Math.floor(gap / 3);
    }
    // sab sorted mark karo
    for (let i = 0; i < n; i++) yield { type: 'sorted', indices: [i] };
  }

  // --- Cocktail Shaker Sort ---
  // bidirectional bubble sort — left to right, fir right to left
  function* cocktailSort(a) {
    const n = a.length;
    let start = 0, end = n - 1, swapped = true;

    while (swapped) {
      swapped = false;

      // left to right pass
      for (let i = start; i < end; i++) {
        yield { type: 'compare', indices: [i, i + 1] };
        if (a[i] > a[i + 1]) {
          [a[i], a[i + 1]] = [a[i + 1], a[i]];
          yield { type: 'swap', indices: [i, i + 1] };
          swapped = true;
        }
      }
      yield { type: 'sorted', indices: [end] };
      end--;

      if (!swapped) break;
      swapped = false;

      // right to left pass — yahi cocktail ka twist hai
      for (let i = end; i > start; i--) {
        yield { type: 'compare', indices: [i, i - 1] };
        if (a[i] < a[i - 1]) {
          [a[i], a[i - 1]] = [a[i - 1], a[i]];
          yield { type: 'swap', indices: [i, i - 1] };
          swapped = true;
        }
      }
      yield { type: 'sorted', indices: [start] };
      start++;
    }
    // baaki sab bhi sorted
    for (let i = start; i <= end; i++) yield { type: 'sorted', indices: [i] };
  }

  // --- Comb Sort ---
  // bubble sort mein gap shrink karte jao — turtle problem solve hota hai
  function* combSort(a) {
    const n = a.length;
    let gap = n;
    const SHRINK = 1.3;
    let sorted = false;

    while (!sorted) {
      gap = Math.floor(gap / SHRINK);
      if (gap <= 1) {
        gap = 1;
        sorted = true; // agar koi swap nahi hua toh done
      }

      for (let i = 0; i + gap < n; i++) {
        yield { type: 'compare', indices: [i, i + gap] };
        if (a[i] > a[i + gap]) {
          [a[i], a[i + gap]] = [a[i + gap], a[i]];
          yield { type: 'swap', indices: [i, i + gap] };
          sorted = false;
        }
      }
    }
    for (let i = 0; i < n; i++) yield { type: 'sorted', indices: [i] };
  }

  // --- Counting Sort ---
  // non-comparison sort — values ko buckets mein gin lo, fir output karo
  // yahan values 0-1 range mein hain toh 100 buckets use karte hain
  function* countingSort(a) {
    const n = a.length;
    const BUCKETS = 100;

    // har value ko bucket mein convert karo (0 to BUCKETS-1)
    const counts = new Array(BUCKETS).fill(0);
    for (let i = 0; i < n; i++) {
      const bucket = Math.min(BUCKETS - 1, Math.floor(a[i] * BUCKETS));
      counts[bucket]++;
      yield { type: 'compare', indices: [i] }; // counting step dikhao
    }

    // cumulative sum — prefix sum banao
    for (let i = 1; i < BUCKETS; i++) {
      counts[i] += counts[i - 1];
    }

    // output array banao — stable sort, right to left
    const output = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      const bucket = Math.min(BUCKETS - 1, Math.floor(a[i] * BUCKETS));
      counts[bucket]--;
      output[counts[bucket]] = a[i];
    }

    // copy back to original array with animation
    for (let i = 0; i < n; i++) {
      a[i] = output[i];
      yield { type: 'overwrite', indices: [i] };
      // batch mein sorted mark karo
      if (i % 3 === 0 || i === n - 1) {
        for (let k = Math.max(0, i - 2); k <= i; k++) {
          yield { type: 'sorted', indices: [k] };
        }
      }
    }
  }

  // --- Radix Sort (LSD) ---
  // digit by digit sort — least significant digit pehle
  // values 0-1 ko 0-999 mein scale karke 3 passes karte hain
  function* radixSort(a) {
    const n = a.length;
    const SCALE = 1000; // 3 digit precision
    // scaled integer version banao
    const scaled = a.map(v => Math.floor(v * SCALE));

    for (let exp = 1; exp <= SCALE; exp *= 10) {
      // counting sort by current digit
      const buckets = Array.from({ length: 10 }, () => []);

      for (let i = 0; i < n; i++) {
        const digit = Math.floor(scaled[i] / exp) % 10;
        buckets[digit].push({ val: a[i], sval: scaled[i] });
        yield { type: 'compare', indices: [i] };
      }

      // buckets se wapas array mein daalo
      let idx = 0;
      for (let d = 0; d < 10; d++) {
        for (const item of buckets[d]) {
          a[idx] = item.val;
          scaled[idx] = item.sval;
          yield { type: 'overwrite', indices: [idx] };
          idx++;
        }
      }
    }

    for (let i = 0; i < n; i++) yield { type: 'sorted', indices: [i] };
  }

  // --- Tim Sort (simplified) ---
  // real-world hybrid: insertion sort chhote runs pe, fir merge
  // Python aur Java ka default sort yahi hai
  function* timSort(a) {
    const n = a.length;
    const RUN = 32; // minimum run length

    // Step 1: insertion sort small runs pe
    for (let start = 0; start < n; start += RUN) {
      const end = Math.min(start + RUN - 1, n - 1);
      // insertion sort [start, end]
      for (let i = start + 1; i <= end; i++) {
        const key = a[i];
        let j = i - 1;
        if (j >= start) yield { type: 'compare', indices: [i, j] };
        while (j >= start && a[j] > key) {
          a[j + 1] = a[j];
          yield { type: 'swap', indices: [j, j + 1] };
          j--;
          if (j >= start) yield { type: 'compare', indices: [j, j + 1] };
        }
        a[j + 1] = key;
      }
    }

    // Step 2: merge runs — double the size each pass
    for (let size = RUN; size < n; size *= 2) {
      for (let left = 0; left < n; left += 2 * size) {
        const mid = Math.min(left + size - 1, n - 1);
        const right = Math.min(left + 2 * size - 1, n - 1);
        if (mid < right) {
          yield* merge(a, left, mid, right);
        }
      }
    }

    // sab sorted
    for (let i = 0; i < n; i++) yield { type: 'sorted', indices: [i] };
  }

  // ============================================================
  // ARRAY GENERATION
  // ============================================================

  function generateArray() {
    arr = [];
    sortedIndices.clear();
    comparisons = 0;
    swaps = 0;

    switch (currentOrder) {
      case 'Random':
        for (let i = 0; i < arraySize; i++) {
          arr.push(Math.random());
        }
        break;

      case 'Nearly Sorted':
        // pehle sorted banao, fir thode random swaps
        for (let i = 0; i < arraySize; i++) {
          arr.push((i + 1) / arraySize);
        }
        // ~10% elements ko randomly swap kar do
        const numSwaps = Math.floor(arraySize * 0.1);
        for (let s = 0; s < numSwaps; s++) {
          const a = Math.floor(Math.random() * arraySize);
          const b = Math.floor(Math.random() * arraySize);
          [arr[a], arr[b]] = [arr[b], arr[a]];
        }
        break;

      case 'Reversed':
        for (let i = 0; i < arraySize; i++) {
          arr.push(1 - i / arraySize);
        }
        break;

      case 'Few Unique':
        // sirf 5-6 unique values rakhte hain
        const uniqueCount = Math.min(6, Math.max(3, Math.floor(arraySize / 10)));
        const vals = [];
        for (let i = 0; i < uniqueCount; i++) {
          vals.push((i + 1) / (uniqueCount + 1));
        }
        for (let i = 0; i < arraySize; i++) {
          arr.push(vals[Math.floor(Math.random() * uniqueCount)]);
        }
        break;
    }

    barColors = new Array(arraySize).fill(BAR_DEFAULT);
    updateInfo();
  }

  // ============================================================
  // SORTING CONTROL
  // ============================================================

  function startSorting() {
    if (isSorting) return;

    // pehle check karo — kya already sorted hai?
    // fresh array pe sort karo
    isSorting = true;
    comparisons = 0;
    swaps = 0;
    sortedIndices.clear();
    barColors = new Array(arr.length).fill(BAR_DEFAULT);

    // controls disable karo
    updateControlStates();
    sortBtn.textContent = '\u25A0 Stop';

    // generator shuru karo selected algorithm se
    const algoFn = ALGORITHMS[currentAlgo].fn;
    generator = algoFn(arr);

    // stepping shuru karo
    stepSort();
  }

  function stopSorting() {
    isSorting = false;
    generator = null;
    if (sortTimer) {
      clearTimeout(sortTimer);
      sortTimer = null;
    }
    sortBtn.textContent = '\u25B6 Sort';
    updateControlStates();
    requestDraw();
  }

  function stepSort() {
    if (!isSorting || !generator) return;

    // speed ke hisaab se multiple steps per call — fast hone pe zyada steps
    // speed 1ms pe 50 steps per frame, speed 200ms pe 1 step per frame
    const stepsPerFrame = speed <= 5 ? 50 : speed <= 20 ? 10 : speed <= 50 ? 3 : 1;

    for (let s = 0; s < stepsPerFrame; s++) {
      const result = generator.next();
      if (result.done) {
        // sorting complete — sab green karo
        finishSorting();
        return;
      }

      const step = result.value;
      processStep(step);
    }

    updateInfo();
    requestDraw();

    // next step schedule karo
    sortTimer = setTimeout(stepSort, Math.max(1, speed));
  }

  function processStep(step) {
    // pehle sab non-sorted bars ko default color pe reset karo
    for (let i = 0; i < barColors.length; i++) {
      if (!sortedIndices.has(i)) {
        barColors[i] = BAR_DEFAULT;
      }
    }

    switch (step.type) {
      case 'compare':
        comparisons++;
        for (const idx of step.indices) {
          if (idx >= 0 && idx < barColors.length && !sortedIndices.has(idx)) {
            barColors[idx] = BAR_COMPARE;
          }
        }
        break;

      case 'swap':
        swaps++;
        for (const idx of step.indices) {
          if (idx >= 0 && idx < barColors.length) {
            barColors[idx] = BAR_SWAP;
          }
        }
        break;

      case 'overwrite':
        // merge sort mein direct overwrite hota hai — swap jaisa dikhao
        swaps++;
        for (const idx of step.indices) {
          if (idx >= 0 && idx < barColors.length) {
            barColors[idx] = BAR_SWAP;
          }
        }
        break;

      case 'sorted':
        for (const idx of step.indices) {
          if (idx >= 0 && idx < barColors.length) {
            sortedIndices.add(idx);
            barColors[idx] = BAR_SORTED;
          }
        }
        break;

      case 'pivot':
        for (const idx of step.indices) {
          if (idx >= 0 && idx < barColors.length) {
            barColors[idx] = BAR_PIVOT;
          }
        }
        break;
    }
  }

  function finishSorting() {
    // sab bars green karo — sorted hai sab
    for (let i = 0; i < barColors.length; i++) {
      barColors[i] = BAR_SORTED;
      sortedIndices.add(i);
    }
    isSorting = false;
    generator = null;
    sortBtn.textContent = '\u25B6 Sort';
    updateControlStates();
    updateInfo();
    requestDraw();
  }

  // ============================================================
  // UI HELPERS
  // ============================================================

  function selectAlgorithm(key) {
    currentAlgo = key;
    for (const k of Object.keys(algoButtons)) {
      algoButtons[k].style.cssText = makeButtonCSS(k === key);
    }
    updateInfo();
  }

  function updateInfo() {
    const algo = ALGORITHMS[currentAlgo];
    algoNameSpan.textContent = algo.name;
    complexitySpan.textContent = 'Time: ' + algo.time + '  Space: ' + algo.space;
    countersSpan.textContent = 'Comparisons: ' + comparisons + '  Swaps: ' + swaps;
  }

  function updateControlStates() {
    // sorting ke dauran size slider aur algo buttons disable karo
    const disabled = isSorting;
    sizeGroup.slider.disabled = disabled;
    orderSelect.disabled = disabled;
    for (const btn of Object.values(algoButtons)) {
      btn.style.opacity = disabled ? '0.4' : '1';
      btn.style.pointerEvents = disabled ? 'none' : 'auto';
    }
  }

  function makeButtonCSS(active) {
    return [
      'padding:5px 10px',
      'border:1px solid ' + (active ? ACCENT : 'rgba(167,139,250,0.25)'),
      'border-radius:6px',
      'background:' + (active ? 'rgba(167,139,250,0.2)' : 'rgba(167,139,250,0.06)'),
      'color:' + (active ? '#e2e8f0' : '#94a3b8'),
      'font-family:' + FONT,
      'font-size:11px',
      'cursor:pointer',
      'transition:all 0.15s ease',
      'outline:none',
    ].join(';');
  }

  function makeActionButtonCSS() {
    return [
      'padding:5px 14px',
      'border:1px solid ' + ACCENT,
      'border-radius:6px',
      'background:rgba(167,139,250,0.15)',
      'color:#e2e8f0',
      'font-family:' + FONT,
      'font-size:11px',
      'font-weight:600',
      'cursor:pointer',
      'transition:all 0.15s ease',
      'outline:none',
    ].join(';');
  }

  function makeSliderGroup(label, min, max, initial, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px;';

    const lbl = document.createElement('span');
    lbl.textContent = label + ': ' + initial;
    lbl.style.cssText = 'min-width:80px;font-family:' + FONT + ';font-size:11px;color:#94a3b8;';
    wrapper.appendChild(lbl);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.value = initial;
    slider.style.cssText = [
      'width:120px',
      'height:4px',
      'accent-color:' + ACCENT,
      'cursor:pointer',
    ].join(';');
    slider.addEventListener('input', () => {
      const v = parseInt(slider.value);
      lbl.textContent = label + ': ' + v;
      onChange(v);
    });
    wrapper.appendChild(slider);

    return { wrapper, slider, label: lbl };
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

  function draw() {
    // lab pause: only active sim animates
    if (window.__labPaused && window.__labPaused !== container.id) { animationId = null; return; }
    animationId = null;
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (arr.length === 0) return;

    const n = arr.length;
    const padding = 4;                                 // canvas ke sides pe thoda gap
    const totalBarSpace = canvasW - 2 * padding;
    const gap = n > 100 ? 0.5 : n > 60 ? 1 : 1.5;    // bars ke beech gap
    const barW = Math.max(1, (totalBarSpace - (n - 1) * gap) / n);
    const maxBarH = canvasH - 10;                      // top pe thoda margin

    for (let i = 0; i < n; i++) {
      const x = padding + i * (barW + gap);
      const h = Math.max(2, arr[i] * maxBarH);        // minimum 2px height taaki dikhe
      const y = canvasH - h;

      ctx.fillStyle = barColors[i] || BAR_DEFAULT;
      // rounded top — sirf jab bar enough wide ho
      if (barW >= 4) {
        const radius = Math.min(2, barW / 3);
        ctx.beginPath();
        ctx.moveTo(x, canvasH);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.lineTo(x + barW - radius, y);
        ctx.quadraticCurveTo(x + barW, y, x + barW, y + radius);
        ctx.lineTo(x + barW, canvasH);
        ctx.closePath();
        ctx.fill();
      } else {
        ctx.fillRect(x, y, barW, h);
      }
    }
  }

  function requestDraw() {
    if (animationId) return;
    animationId = requestAnimationFrame(draw);
  }

  // ============================================================
  // INTERSECTION OBSERVER — sirf visible hone pe animate karo
  // ============================================================

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      isVisible = entry.isIntersecting;
      if (isVisible) {
        resizeCanvas();
        requestDraw();
      }
    }
  }, { threshold: 0.1 });
  observer.observe(container);
  document.addEventListener('lab:resume', () => { if (isVisible && !animationId) draw(); });

  // resize handler
  const resizeObserver = new ResizeObserver(() => {
    if (!isVisible) return;
    resizeCanvas();
    requestDraw();
  });
  resizeObserver.observe(canvas);

  // ============================================================
  // INITIALIZATION — sab set karo
  // ============================================================

  resizeCanvas();
  selectAlgorithm(currentAlgo);
  generateArray();
  updateInfo();
  requestDraw();
}
