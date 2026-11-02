// === Tab Navigation ===
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    link.classList.add('active');
    document.getElementById('tab-' + link.dataset.tab).classList.add('active');
    if (link.dataset.tab === 'metrics' && !metricsLoaded) loadMetrics();
    if (link.dataset.tab === 'speedup' && !speedupLoaded) loadSpeedup();
    if (link.dataset.tab === 'plots' && !plotsLoaded) loadPlots();
  });
});

let currentPage = 1, metricsLoaded = false, speedupLoaded = false, plotsLoaded = false;

// === Vehicle Grid ===
async function loadVehicles(page = 1) {
  currentPage = page;
  try {
    const res = await fetch(`/api/vehicles?page=${page}&per_page=18`);
    const data = await res.json();
    const grid = document.getElementById('vehicle-grid');
    grid.innerHTML = data.vehicles.map(v => `
      <div class="vehicle-thumb" onclick="selectVehicle(${v.vehicle_id}, this)" title="Vehicle ${v.vehicle_id}">
        <img src="${v.thumbnail}" alt="Vehicle ${v.vehicle_id}" loading="lazy">
        <span class="badge">${v.num_images} cam</span>
      </div>
    `).join('');
    renderPagination(data.page, data.total_pages);
  } catch (e) { console.error('Failed to load vehicles:', e); }
}

function renderPagination(current, total) {
  const el = document.getElementById('pagination');
  let html = '';
  const start = Math.max(1, current - 2), end = Math.min(total, current + 2);
  if (start > 1) html += `<button onclick="loadVehicles(1)">1</button>`;
  if (start > 2) html += `<button disabled>…</button>`;
  for (let i = start; i <= end; i++)
    html += `<button class="${i === current ? 'active' : ''}" onclick="loadVehicles(${i})">${i}</button>`;
  if (end < total - 1) html += `<button disabled>…</button>`;
  if (end < total) html += `<button onclick="loadVehicles(${total})">${total}</button>`;
  el.innerHTML = html;
}

async function selectVehicle(vid, el) {
  document.querySelectorAll('.vehicle-thumb').forEach(t => t.classList.remove('selected'));
  if (el) el.classList.add('selected');
  const res = await fetch(`/api/vehicle/${vid}`);
  const data = await res.json();
  document.getElementById('empty-state').classList.add('hidden');
  const results = document.getElementById('match-results');
  results.classList.remove('hidden');
  const queryImg = data.query_images[0];
  let html = `
    <div class="match-header">
      <img class="query-preview" src="${queryImg.path}" alt="Query">
      <div class="match-header-info">
        <h2>Vehicle #${data.vehicle_id}</h2>
        <p>${data.total_matches} images across ${data.num_cameras} cameras</p>
      </div>
    </div>`;
  for (const [camId, imgs] of Object.entries(data.by_camera)) {
    html += `<div class="camera-group"><h4>Camera ${camId} (${imgs.length} image${imgs.length > 1 ? 's' : ''})</h4><div class="camera-images">`;
    imgs.forEach(img => { html += `<img src="${img.path}" alt="Cam ${camId}" loading="lazy">`; });
    html += `</div></div>`;
  }
  results.innerHTML = html;
}

// === Metrics Tab ===
async function loadMetrics() {
  const res = await fetch('/api/metrics');
  const m = await res.json();
  metricsLoaded = true;
  const bl = m.baseline, tcm = m.tcm_enhanced, filt = m.filtering;

  document.getElementById('metrics-grid').innerHTML = `
    <div class="metric-card"><div class="metric-label">mAP Improvement</div>
      <div class="metric-value green">+${(tcm.mAP - bl.mAP).toFixed(1)}%</div>
      <div class="metric-sub">${bl.mAP}% → ${tcm.mAP}%</div></div>
    <div class="metric-card"><div class="metric-label">Search Reduction</div>
      <div class="metric-value cyan">${filt.sequential.reduction_pct}%</div>
      <div class="metric-sub">Sequential tracking</div></div>
    <div class="metric-card"><div class="metric-label">Speedup Factor</div>
      <div class="metric-value orange">${filt.sequential.speedup}</div>
      <div class="metric-sub">${filt.sequential.total_pairs.toLocaleString()} → ${filt.sequential.retained} pairs</div></div>
    <div class="metric-card"><div class="metric-label">Recall</div>
      <div class="metric-value accent">${tcm.rank5}%</div>
      <div class="metric-sub">Rank-5 accuracy</div></div>`;

  const metrics = [
    { label: 'mAP', b: bl.mAP, t: tcm.mAP },
    { label: 'Rank-1', b: bl.rank1, t: tcm.rank1 },
    { label: 'Rank-5', b: bl.rank5, t: tcm.rank5 },
    { label: 'Rank-10', b: bl.rank10, t: tcm.rank10 }
  ];
  let barHtml = '<div class="chart-card"><h3>Baseline vs TCM-Enhanced</h3>';
  metrics.forEach(m => {
    barHtml += `<div class="bar-group">
      <div class="bar-label"><span>${m.label}</span></div>
      <div class="bar-track"><div class="bar-fill baseline" style="width:${m.b}%">${m.b}% Baseline</div></div>
      <div class="bar-track" style="margin-top:4px"><div class="bar-fill tcm" style="width:${m.t}%">${m.t}% TCM</div></div>
    </div>`;
  });
  barHtml += '</div>';

  barHtml += `<div class="chart-card"><h3>Filtering Efficiency</h3>
    <div class="bar-group"><div class="bar-label"><span>Sequential Tracking</span><span class="highlight">${filt.sequential.reduction_pct}%</span></div>
      <div class="bar-track"><div class="bar-fill tcm" style="width:${filt.sequential.reduction_pct}%">${filt.sequential.filtered.toLocaleString()} filtered</div></div></div>
    <div class="bar-group"><div class="bar-label"><span>Full Gallery Eval</span><span class="highlight">${filt.full_gallery.reduction_pct}%</span></div>
      <div class="bar-track"><div class="bar-fill baseline" style="width:${filt.full_gallery.reduction_pct}%">${filt.full_gallery.filtered.toLocaleString()} filtered</div></div></div></div>`;
  document.getElementById('chart-row').innerHTML = barHtml;

  // Ablation table
  let tbl = `<div class="table-section"><h3>Ablation Study — Sensitivity Analysis</h3><table><tr>
    <th>Configuration</th><th>mAP</th><th>Rank-1</th><th>Recall</th><th>Filter Rate</th></tr>
    <tr><td><em>Baseline (No TCM)</em></td><td>${bl.mAP}%</td><td>${bl.rank1}%</td><td>100%</td><td>0%</td></tr>`;
  m.ablation.tortuosity.forEach(r => {
    const ours = r.tau === 1.2 ? ' class="highlight"' : '';
    tbl += `<tr><td>τ ≈ ${r.tau}${r.tau === 1.2 ? ' <strong>(Ours)</strong>' : ''}</td>
      <td${ours}>${r.mAP}%</td><td>${r.rank1}%</td><td${ours}>${r.recall}%</td><td>${r.filter_rate}%</td></tr>`;
  });
  m.ablation.traffic.forEach(r => {
    const ours = r.mode.includes('Dynamic') ? ' class="highlight"' : '';
    tbl += `<tr><td>${r.mode}${r.mode.includes('Dynamic') ? ' <strong>(Ours)</strong>' : ''}</td>
      <td${ours}>${r.mAP}%</td><td>${r.rank1}%</td><td${ours}>${r.recall}%</td><td>${r.filter_rate}%</td></tr>`;
  });
  tbl += '</table></div>';
  document.getElementById('ablation-section').innerHTML = tbl;
}

// === Speedup Tab ===
async function loadSpeedup() {
  const res = await fetch('/api/metrics');
  const m = await res.json();
  speedupLoaded = true;
  const seq = m.filtering.sequential;
  const fg = m.filtering.full_gallery;
  document.getElementById('speedup-content').innerHTML = `
    <div class="speedup-grid">
      <div class="speedup-card"><h3>Sequential Tracking Speedup</h3>
        <div class="big-number">${seq.speedup}</div>
        <p>Search space reduced from ${seq.total_pairs.toLocaleString()} to just ${seq.retained.toLocaleString()} candidates</p></div>
      <div class="speedup-card"><h3>Search Space Reduction</h3>
        <div class="big-number">${seq.reduction_pct}%</div>
        <p>Impossible matches eliminated via physics-based filtering</p></div>
      <div class="speedup-card"><h3>Full Gallery Reduction</h3>
        <div class="big-number">${fg.reduction_pct}%</div>
        <p>${fg.filtered.toLocaleString()} of ${fg.total_pairs.toLocaleString()} pairs filtered across all queries</p></div>
      <div class="speedup-card"><h3>TCM Overhead</h3>
        <div class="big-number">O(1)</div>
        <p>Per-candidate: 1 lookup + 2 divisions + 1 comparison</p></div>
    </div>
    <div class="chart-card">
      <h3>TCM Filtering Pipeline — Sequential Tracking</h3>
      <div class="funnel">
        <div class="funnel-step"><div class="step-num">1</div><div class="funnel-info">
          <h4>Full Gallery</h4><p>${seq.total_pairs.toLocaleString()} candidate pairs from camera pair observations</p></div></div>
        <div class="funnel-arrow">↓ T<sub>min</sub> filter (too fast = physically impossible)</div>
        <div class="funnel-step"><div class="step-num">2</div><div class="funnel-info">
          <h4>Speed Limit Check</h4><p>Eliminates pairs where Δt < T<sub>min</sub> — vehicle would exceed max speed</p></div></div>
        <div class="funnel-arrow">↓ T<sub>max</sub> filter (too slow = implausible match)</div>
        <div class="funnel-step"><div class="step-num">3</div><div class="funnel-info">
          <h4>Congestion Window Check</h4><p>Eliminates pairs where Δt > T<sub>max</sub> · α(t) — beyond rush-hour adjusted bounds</p></div></div>
        <div class="funnel-arrow">↓ ${seq.reduction_pct}% eliminated</div>
        <div class="funnel-step"><div class="step-num">4</div><div class="funnel-info">
          <h4>Valid Candidates: <span class="highlight">${seq.retained.toLocaleString()}</span></h4><p>Only physically plausible matches proceed to visual ranking (${seq.speedup} speedup)</p></div></div>
      </div>
    </div>
    <div class="chart-card" style="margin-top:1.5rem">
      <h3>Camera Network Overview</h3>
      <p style="color:var(--text2);font-size:.9rem;margin-bottom:1rem">
        ${m.camera_network.num_cameras} cameras, pairwise driving distances: ${m.camera_network.min_distance_m}m — ${m.camera_network.max_distance_m}m (mean: ${m.camera_network.mean_distance_m}m). Tortuosity factor τ = ${m.camera_network.tortuosity}.
        Road types assigned by distance: Highway (>5km), Arterial (1-5km), Residential (200m-1km), Parking (<200m).
      </p>
      <table><tr><th>Road Type</th><th>Distance</th><th>V<sub>max</sub></th><th>V<sub>avg</sub></th><th>V<sub>min</sub></th></tr>
        <tr><td>Highway</td><td>>5 km</td><td>120 km/h</td><td>80 km/h</td><td>40 km/h</td></tr>
        <tr><td>Arterial</td><td>1–5 km</td><td>60 km/h</td><td>35 km/h</td><td>10 km/h</td></tr>
        <tr><td>Residential</td><td>200m–1 km</td><td>40 km/h</td><td>20 km/h</td><td>5 km/h</td></tr>
        <tr><td>Parking</td><td><200m</td><td>20 km/h</td><td>10 km/h</td><td>1 km/h</td></tr></table>
    </div>`;
}

// === Plots Tab ===
async function loadPlots() {
  const res = await fetch('/api/metrics');
  const m = await res.json();
  plotsLoaded = true;
  const bl = m.baseline, tcm = m.tcm_enhanced, seq = m.filtering.sequential, fg = m.filtering.full_gallery;
  const plots = [
    {
      file: 'performance_comparison.png', title: 'Performance Comparison',
      desc: `Baseline (TransReID) vs TCM-Enhanced: mAP ${bl.mAP}%→${tcm.mAP}%, with zoomed axes showing the +${(tcm.mAP - bl.mAP).toFixed(1)}% improvement clearly.`
    },
    {
      file: 'search_space_reduction.png', title: 'Search Space Reduction',
      desc: `Sequential tracking: ${seq.total_pairs.toLocaleString()}→${seq.retained.toLocaleString()} pairs (${seq.speedup} speedup, ${seq.reduction_pct}% eliminated). Full gallery: ${fg.reduction_pct}% reduction across ${(fg.total_pairs/1e6).toFixed(1)}M pairs.`
    },
    {
      file: 'ablation_study.png', title: 'Ablation Study',
      desc: 'Sensitivity analysis of tortuosity (τ≈1.0/1.2/1.4/1.6) and traffic model (static vs dynamic α). Our τ≈1.2 + dynamic α(t) is optimal.'
    },
    {
      file: 'filtering_pipeline.png', title: 'TCM Filtering Pipeline',
      desc: `Step-by-step funnel diagram: Full Gallery → Speed Limit Filter (T_min) → Congestion Window (T_max·α) → ${seq.retained.toLocaleString()} Valid Candidates.`
    },
    {
      file: 'temporal_constraints.png', title: 'Temporal Constraint Model',
      desc: 'Road type speed profiles (Highway/Arterial/Residential/Parking) and valid time windows vs camera distance with rush-hour extension.'
    },
    {
      file: 'improvement_summary.png', title: 'Executive Summary',
      desc: `Key results at a glance: ${seq.speedup} speedup, ${seq.reduction_pct}% elimination, +${(tcm.mAP - bl.mAP).toFixed(1)}% mAP, O(1) per-pair cost.`
    },
  ];
  document.getElementById('plots-grid').innerHTML = plots.map(p => `
    <div class="plot-card"><h3>${p.title}</h3><p>${p.desc}</p>
      <img src="/images/plots/${p.file}" alt="${p.title}" loading="lazy">
    </div>`).join('');
}

// Initial load
loadVehicles(1);
