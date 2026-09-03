import { Viewer } from './viewer.js';
import { DeckView } from './deck-view.js';

const $ = (sel) => document.querySelector(sel);
const screens = {
  upload: $('#screen-upload'),
  loading: $('#screen-loading'),
  view: $('#screen-view'),
};
const state = {
  jobId: null,
  status: null,   // 'queued' | 'running' | 'done' | 'error'
  viewer: null,
  deckView: null,
  activeView: '3d',   // '3d' | 'map'
  poll: null,
  gcp: [],        // {col,row,ox,oy,h,height}
  samples: {},
};

// ------------------------------------------------------------------ asset base
function assetUrl(jobId, name) {
  return `/api/jobs/${jobId}/asset/${name}`;
}

// ------------------------------------------------------------------ screen nav
function showScreen(name) {
  for (const [k, el] of Object.entries(screens)) el.classList.toggle('active', k === name);
  $('#btn-new').classList.toggle('hidden', name === 'upload');
  if (name === 'view') {
    // let the renderer know container is visible
    setTimeout(() => state.viewer?._resize(), 0);
    setTimeout(() => state.deckView?.resize(), 0);
  }
}

// ------------------------------------------------------------------ upload UI
const drop = $('#dropzone');
const fileInput = $('#file-input');
let chosenFile = null;

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) pickFile(f);
});
fileInput.addEventListener('change', () => pickFile(fileInput.files[0]));

function pickFile(f) {
  if (!f) return;
  const ok = /\.(png|jpe?g|tiff?)$/i.test(f.name);
  $('#file-input').value = '';
  if (!ok) {
    $('#upload-error').textContent = 'Unsupported file — use PNG / JPG / GeoTIFF.';
    return;
  }
  $('#upload-error').textContent = '';
  chosenFile = f;
  $('#dropzone-name').textContent = f.name;
  $('#fp-meta').textContent = fileKind(f.name) + ' · ' + fileSize(f.size);
  $('.dz-empty').classList.add('hidden');
  $('#file-preview').classList.remove('hidden');
  $('#btn-process').disabled = false;
}

function fileKind(name) {
  if (/\.(tiff?)$/i.test(name)) return 'GeoTIFF';
  if (/\.png$/i.test(name)) return 'PNG';
  return 'JPG';
}
function fileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(2) + ' MB';
}

$('#fp-remove').addEventListener('click', (e) => {
  e.stopPropagation();
  resetUpload();
});
$('#ref-input').addEventListener('change', () => {
  const r = $('#ref-input').files[0];
  $('#ref-picker-label').textContent = r ? 'Reference selected' : 'Choose reference';
  $('#ref-picker-file').textContent = r ? r.name : '.tif / .tiff';
});

$('#btn-process').addEventListener('click', async () => {
  if (!chosenFile) return;
  const body = new FormData();
  body.append('image', chosenFile);
  const bbox = $('#bbox').value.trim();
  if (bbox) body.append('bbox', bbox);
  const ref = $('#ref-input').files[0];
  if (ref) body.append('reference', ref);
  await startJob('/api/process', { method: 'POST', body });
});

// ------------------------------------------------------------------ samples
async function loadSamples() {
  try {
    const r = await fetch('/api/samples');
    state.samples = await r.json();
    for (const btn of document.querySelectorAll('.sample')) {
      const label = state.samples[btn.dataset.sample];
      if (label) btn.textContent = label;
    }
  } catch { /* samples stay empty if unavailable */ }
}

document.querySelectorAll('.sample').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const name = btn.dataset.sample;
    await startJob(`/api/samples/${name}/process`, { method: 'POST' });
  });
});

$('#btn-new').addEventListener('click', () => {
  state.poll && clearInterval(state.poll);
  state.viewer?.dispose(); state.viewer = null;
  state.deckView?.dispose(); state.deckView = null;
  resetUpload();
  showScreen('upload');
});

function resetUpload() {
  chosenFile = null;
  $('#dropzone-name').textContent = '';
  $('#fp-meta').textContent = '';
  $('.dz-empty').classList.remove('hidden');
  $('#file-preview').classList.add('hidden');
  $('#btn-process').disabled = true;
  $('#ref-input').value = '';
  $('#ref-picker-label').textContent = 'Choose reference';
  $('#ref-picker-file').textContent = '.tif / .tiff';
  $('#bbox').value = '';
  $('#upload-error').textContent = '';
}

// ------------------------------------------------------------------ job lifecycle
async function startJob(url, opts) {
  showScreen('loading');
  $('#progress-label').textContent = 'submitting…';
  $('#warnings').innerHTML = '';
  $('#progress-bar').style.width = '0%';
  try {
    const r = await fetch(url, opts);
    if (!r.ok) {
      const detail = (await r.json().catch(() => null))?.detail || `HTTP ${r.status}`;
      throw new Error(detail);
    }
    const j = await r.json();
    state.jobId = j.job_id;
    $('#job-state').textContent = `job ${j.job_id.slice(0, 6)}`;
    $('#btn-new').classList.remove('hidden');
    state.poll = setInterval(pollJob, 1800);
    await pollJob();
  } catch (e) {
    failLoading(e.message);
  }
}

const PROGRESS_LABELS = {
  georef: 'Reading georeference…',
  depth: 'Estimating depth / structure…',
  calibrate: 'Scaling heights…',
  terrain: 'Fetching terrain baseline (SRTM)…',
  dsm: 'Building DSM…',
  validate: 'Validating against reference…',
  export: 'Preparing 3D assets…',
  done: 'Done.',
};

async function pollJob() {
  if (!state.jobId) return;
  const r = await fetch(`/api/jobs/${state.jobId}`);
  if (!r.ok) return;
  const j = await r.json();
  if (j.status === 'error') {
    clearInterval(state.poll);
    failLoading(j.error || 'processing failed');
    return;
  }
  $('#progress-label').textContent = PROGRESS_LABELS[j.progress] || j.progress || '…';
  const pct = j.progress === 'done' ? 100 : stagePct(j.progress);
  $('#progress-bar').style.width = `${pct}%`;
  if (j.warnings && j.warnings.length) {
    $('#warnings').innerHTML = j.warnings.map((w) => `<div>⚠ ${w.replace(/[<>&]/g, '')}</div>`).join('');
  }
  if (j.status === 'done') {
    clearInterval(state.poll);
    await enterView(j);
    // after first view, keep polling-free (status only needed on demand)
  }
}

function stagePct(stage) {
  const idx = Object.keys(PROGRESS_LABELS).indexOf(stage);
  return idx < 0 ? 20 : Math.round((idx / (Object.keys(PROGRESS_LABELS).length - 1)) * 100);
}

function failLoading(msg) {
  $('#progress-label').textContent = 'Failed: ' + msg;
  $('#upload-error').textContent = msg;
  clearInterval(state.poll);
  showScreen('upload');
}

// ------------------------------------------------------------------ viewer setup
async function enterView(j) {
  const header = j.header;
  showScreen('view');

  // fetch binary + textures
  const heightsBuf = await fetch(assetUrl(state.jobId, header.assets.heights)).then((r) => r.arrayBuffer());
  const heights = new Float32Array(heightsBuf);
  let struct = null;
  if (header.assets.struct) {
    const buf = await fetch(assetUrl(state.jobId, header.assets.struct)).then((r) => r.arrayBuffer());
    struct = new Float32Array(buf);
  }

  // region grid (full-res heights for robust region stats)
  let region = null;
  if (header.region) {
    try {
      const rh = new Float32Array(
        await fetch(assetUrl(state.jobId, header.region.heights)).then((r) => r.arrayBuffer()));
      let rs = null;
      if (header.region.struct) {
        rs = new Float32Array(
          await fetch(assetUrl(state.jobId, header.region.struct)).then((r) => r.arrayBuffer()));
      }
      region = {
        heights: rh, struct: rs,
        gridW: header.region.grid[0], gridH: header.region.grid[1],
        origW: header.region.orig[0], origH: header.region.orig[1],
      };
    } catch (e) { console.warn('region grid unavailable', e); }
  }

  state.viewer = new Viewer($('#viewport'), {
    base: '',
    jobId: state.jobId,
    onPick: (info) => showPick(info),
    onGcpAdd: (info) => gcpAddClick(info),
    onRegion: (stats) => showRegion(stats),
  });
  state.viewer.init(
    header,
    heights,
    struct,
    assetUrl(state.jobId, header.assets.texture),
    header.assets.error ? assetUrl(state.jobId, header.assets.error) : null,
    region,
  );

  $('#btn-region').classList.toggle('hidden', !region);

  // metrics panel
  if (j.metrics && j.metrics.n > 0) showMetrics(j.metrics, header);
  else hideMetrics();

  $('#btn-download').classList.remove('hidden');
  $('#btn-err').classList.toggle('hidden', !header.assets.error);
  $('#btn-err').classList.toggle('active', false);

  // status chip with mode info
  $('#status').classList.remove('hidden');
  const mode = header.mode === 'absolute'
    ? (header.agl ? 'georeferenced · structural (AGL) heights' : 'georeferenced · SRTM terrain + buildings')
    : 'relative heights';
  const gsd = header.gsd_m ? ` · ~${(header.gsd_m).toFixed(2)} m/px` : '';
  $('#status').textContent = `${mode}${gsd} · double-click to read height`;

  // Deck.gl map view (View B)
  state.deckView?.dispose();
  state.deckView = null;
  const view3d = $('#viewport'), viewMap = $('#viewport-deck');
  const deckBtn = $('#btn-view-map'), threeBtn = $('#btn-view-3d');
  if (header.deck) {
    try {
      state.deckView = new DeckView(viewMap, {
        header,
        jobId: state.jobId,
        onPickHeight: (info) => showDeckPick(info),
      });
      state.deckView.init(header, {
        heightsUrl: assetUrl(state.jobId, header.deck.heightsUrl),
        textureUrl: assetUrl(state.jobId, header.deck.textureUrl),
      }, heightsBuf);
      state.activeView = '3d';
      view3d.classList.remove('hidden');
      viewMap.classList.add('hidden');
      deckBtn.classList.remove('hidden');
      threeBtn.classList.add('hidden');
      deckBtn.classList.remove('active');
      threeBtn.classList.add('active');
    } catch (e) {
      console.warn('Deck view unavailable:', e);
      state.deckView = null;
      deckBtn.classList.add('hidden');
      threeBtn.classList.add('hidden');
    }
  } else {
    deckBtn.classList.add('hidden');
    threeBtn.classList.add('hidden');
  }
}

// deck.gl map-style pick -> reuse status chip
function showDeckPick({ height, lon, lat }) {
  const el = $('#status');
  el.classList.remove('hidden');
  let s = Number.isFinite(height) ? `height ${fmtH(height)} m` : 'terrain selected';
  if (Number.isFinite(lon) && Number.isFinite(lat)) s += `  ·  lon ${lon.toFixed(5)}, lat ${lat.toFixed(5)}`;
  el.textContent = s + '  (map view)';
  clearTimeout(state._statusT);
  state._statusT = setTimeout(() => el.classList.add('hidden'), 6000);
}

// toggle between Three.js flythrough (View A) and Deck.gl map (View B)
function switchView(which) {
  if (which === state.activeView) return;
  const view3d = $('#viewport'), viewMap = $('#viewport-deck');
  const deckBtn = $('#btn-view-map'), threeBtn = $('#btn-view-3d');
  state.activeView = which;
  if (which === 'map') {
    view3d.classList.add('hidden');
    viewMap.classList.remove('hidden');
    deckBtn.classList.add('active');
    threeBtn.classList.remove('active');
    // show the "back to 3D" toggle, hide the active-map toggle
    threeBtn.classList.remove('hidden');
    deckBtn.classList.add('hidden');
    // pause the Three render loop to save resources
    state.viewer?.pause && state.viewer.pause(true);
    state.deckView?.resize();
  } else {
    viewMap.classList.add('hidden');
    view3d.classList.remove('hidden');
    deckBtn.classList.remove('active');
    threeBtn.classList.add('active');
    // show the "map" toggle, hide the active-3D toggle
    deckBtn.classList.remove('hidden');
    threeBtn.classList.add('hidden');
    state.viewer?.pause && state.viewer.pause(false);
    state.viewer?._resize();
  }
}

$('#btn-view-map').addEventListener('click', () => switchView('map'));
$('#btn-view-3d').addEventListener('click', () => switchView('3d'))

function showPick({ height, agl, ox, oy }) {
  const el = $('#status');
  el.classList.remove('hidden');
  let s = `height ${fmtH(height)} m`;
  if (Number.isFinite(agl) && agl !== height) s += `  ·  structure ${fmtH(agl)} m`;
  el.textContent = s + `  (px ${Math.round(ox)}, ${Math.round(oy)})`;
  clearTimeout(state._statusT);
  state._statusT = setTimeout(() => el.classList.add('hidden'), 6000);
}

function fmtH(v) {
  return Number.isFinite(v) ? v.toFixed(1) : '—';
}

function showMetrics(m, header) {
  $('#metrics').classList.remove('hidden');
  const grid = $('#metric-grid');
  const rows = [
    ['RMSE', `${m.rmse?.toFixed(2)} m`],
    ['MAE', `${m.mae?.toFixed(2)} m`],
    ['Pearson r', m.pearson != null ? m.pearson.toFixed(3) : '—'],
    ['Bias', m.bias != null ? `${m.bias.toFixed(2)} m` : '—'],
    ['Pixels', m.n ? Number(m.n).toLocaleString() : '—'],
    ['Max err', m.max_err != null ? `${m.max_err.toFixed(1)} m` : '—'],
  ];
  grid.innerHTML = rows
    .filter(([, v]) => v !== '—')
    .map(([k, v]) => `<span>${k}</span><b>${v}</b>`)
    .join('');
}

function hideMetrics() {
  $('#metrics').classList.add('hidden');
}

// ------------------------------------------------------------------ toolbar
$('#btn-reset').addEventListener('click', () => {
  const v = state.viewer;
  if (!v) return;
  const { grid_w: gw, grid_h: gh } = v.header;
  v.flyTo(gw / 2, gh / 2, Math.max(gw, gh));
});

$('#btn-fly').addEventListener('click', () => {
  const flying = state.viewer?.toggleFly();
  $('#btn-fly').classList.toggle('active', !!flying);
  $('#status').textContent = flying ? '✈ fly mode: WASD + mouse (R reset, Esc exits)' : 'orbit mode';
  $('#status').classList.remove('hidden');
});

$('#btn-rgb').addEventListener('click', () => setMode('rgb'));
$('#btn-elev').addEventListener('click', () => setMode('elevation'));
$('#btn-slope').addEventListener('click', () => setMode('slope'));

function setMode(mode) {
  const v = state.viewer;
  if (!v) return;
  const map = { rgb: '#btn-rgb', elevation: '#btn-elev', slope: '#btn-slope', error: '#btn-err' };
  if (mode === 'slope' && !v.header.gsd_m) {
    $('#status').textContent = 'Slope overlay needs a georeferenced image (missing bounding box / GSD).';
    $('#status').classList.remove('hidden');
    clearTimeout(state._statusT);
    state._statusT = setTimeout(() => $('#status').classList.add('hidden'), 5000);
    return;
  }
  v.setMode(mode);
  for (const [, sel] of Object.entries(map)) $(sel).classList.remove('active');
  const btn = map[mode];
  if (btn) $(btn).classList.add('active');
}

$('#btn-err').addEventListener('click', () => {
  const active = $('#btn-err').classList.toggle('active');
  state.viewer.setMode(active ? 'error' : 'rgb');
  if (!active) $('#btn-rgb').classList.remove('active');
});

$('#xrange').addEventListener('input', () => {
  const v = state.viewer;
  if (!v) return;
  v.setExag(parseInt($('#xrange').value, 10));
});

$('#quality').addEventListener('change', () => {
  state.viewer?.setQuality(+$('#quality').value);
});

// ------------------------------------------------------------------ GCP calibration
const gcpPanel = $('#gcp-panel');
$('#btn-gcp').addEventListener('click', () => {
  gcpPanel.classList.toggle('hidden');
  if (gcpPanel.classList.contains('hidden')) endGcpMode();
  else beginGcpMode();
});

function beginGcpMode() {
  state.gcp = [];
  state.viewer?.clearGcpMarks();
  state.viewer?.setGcpMode(true, gcpAddClick);
  renderGcpList();
  $('#gcp-status').textContent = '';
  $('#btn-gcp-apply').disabled = true;
}

function endGcpMode() {
  state.viewer?.setGcpMode(false);
  state.viewer?.clearGcpMarks();
}

function gcpAddClick(info) {
  state.gcp.push({ col: info.col, row: info.row, ox: info.ox, oy: info.oy, height: info.height, h: '' });
  state.viewer?.addGcpMark({ col: info.col, row: info.row });
  renderGcpList();
  $('#btn-gcp-apply').disabled = false;
}

function renderGcpList() {
  const list = $('#gcp-list');
  list.innerHTML = '';
  state.gcp.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'gcp-row';
    row.innerHTML = `<span class="idx">${i + 1}</span>`;
    const hint = document.createElement('input');
    hint.type = 'text';
    hint.placeholder = Number.isFinite(p.height) ? `height ≈ ${p.height.toFixed(1)} m` : `known height (m)`;
    hint.value = p.h;
    hint.addEventListener('input', () => { p.h = hint.value; });
    const rm = document.createElement('button');
    rm.textContent = '✕';
    rm.addEventListener('click', () => {
      state.gcp.splice(i, 1);
      renderGcpList();
      if (!state.gcp.length) $('#btn-gcp-apply').disabled = true;
    });
    row.appendChild(hint);
    row.appendChild(rm);
    list.appendChild(row);
  });
  $('#btn-gcp-apply').disabled = state.gcp.length === 0;
}

$('#btn-gcp-apply').addEventListener('click', async () => {
  const pts = state.gcp
    .map((p) => ({ x: p.ox, y: p.oy, h: parseFloat(p.h) }))
    .filter((p) => Number.isFinite(p.h));
  if (!pts.length) {
    $('#gcp-status').textContent = 'Enter a known height for at least one point.';
    return;
  }
  $('#gcp-status').textContent = 'Recalibrating…';
  try {
    const r = await fetch(`/api/jobs/${state.jobId}/refit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points: pts }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail || 'refit failed');
    // wait for completion
    await waitDone();
    const j = await (await fetch(`/api/jobs/${state.jobId}`)).json();
    state.viewer?.dispose();
    state.deckView?.dispose(); state.deckView = null;
    await enterView(j);
    $('#gcp-status').textContent = j.warnings?.[0] || 'Done.';
    if (j.metrics && j.metrics.n > 0) showMetrics(j.metrics, j.header);
  } catch (e) {
    $('#gcp-status').textContent = `Error: ${e.message}`;
  }
});

async function waitDone() {
  for (;;) {
    const r = await fetch(`/api/jobs/${state.jobId}`);
    const j = await r.json();
    if (j.status === 'done') return;
    if (j.status === 'error') throw new Error(j.error || 'job failed');
    await new Promise((res) => setTimeout(res, 1500));
  }
}

$('#btn-gcp-cancel').addEventListener('click', () => {
  gcpPanel.classList.add('hidden');
  endGcpMode();
});

// ------------------------------------------------------------------ region stats + region-GCP
const regionPanel = $('#region-panel');
const regionStats = $('#region-stats');
let regionState = null;   // last computed region stats

$('#btn-region').addEventListener('click', () => {
  const turningOn = regionPanel.classList.contains('hidden') || !state.viewer?.regionMode;
  regionPanel.classList.toggle('hidden', !turningOn);
  if (turningOn) {
    state.viewer?.setRegionMode(true);
    $('#region-status').textContent = 'Click to add points; click the first point to close the polygon.';
    $('#btn-region-gcp').disabled = true;
  } else {
    state.viewer?.setRegionMode(false);
  }
});

function showRegion(stats) {
  if (!stats) return;
  regionState = stats;
  if (regionPanel.classList.contains('hidden')) regionPanel.classList.remove('hidden');
  state.viewer?.setRegionMode(true);
  const rows = [
    ['median', `${fmtH(stats.median)} m`],
    ['mean', `${fmtH(stats.mean)} m`],
    ['σ (spread)', `${fmtH(stats.sigma)} m`],
    ['min / max', `${fmtH(stats.min)} / ${fmtH(stats.max)} m`],
    ['pixels', Number(stats.n).toLocaleString()],
  ];
  if (Number.isFinite(stats.structMedian)) {
    rows.push(['structure', `${fmtH(stats.structMedian)} m`]);
  }
  regionStats.innerHTML = rows
    .map(([k, v]) => `<span>${k}</span><b>${v}</b>`)
    .join('');
  $('#btn-region-gcp').disabled = false;
  $('#region-status').textContent =
    `Region tool active. Click “Use region as GCP” to recalibrate, or draw a new polygon.`;
}

$('#btn-region-done').addEventListener('click', () => {
  regionPanel.classList.add('hidden');
  state.viewer?.setRegionMode(false);
  regionState = null;
});

$('#btn-region-gcp').addEventListener('click', async () => {
  if (!regionState) return;
  const known = parseFloat($('#region-known').value);
  const h = Number.isFinite(known) ? known : regionState.height;
  const pts = [{ x: regionState.ox, y: regionState.oy, h }];
  $('#region-status').textContent =
    `Using region median (${fmtH(regionState.height)} m)${Number.isFinite(known) ? ` → ${h.toFixed(2)} m` : ''} as GCP; recalibrating…`;
  try {
    const r = await fetch(`/api/jobs/${state.jobId}/refit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points: pts }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail || 'refit failed');
    await waitDone();
    const j = await (await fetch(`/api/jobs/${state.jobId}`)).json();
    state.viewer?.dispose();
    state.deckView?.dispose(); state.deckView = null;
    regionState = null;
    regionPanel.classList.add('hidden');
    $('#region-known').value = '';
    await enterView(j);
    $('#region-status').textContent = j.warnings?.[0] || 'Done. Region recalibrated.';
  } catch (e) {
    $('#region-status').textContent = `Error: ${e.message}`;
  }
});


$('#btn-download').addEventListener('click', () => {
  window.open(`/api/jobs/${state.jobId}/download`, '_blank');
});

// ------------------------------------------------------------------ init
loadSamples();