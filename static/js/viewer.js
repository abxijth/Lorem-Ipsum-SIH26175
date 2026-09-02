import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FlyControls } from 'three/addons/controls/FlyControls.js';

const VIEW_SCALE = 1; // heights are already in metres; keep world metres for GSD-aware slope/jumps

export class Viewer {
  /**
   * @param {HTMLElement} container
   * @param {object} hooks  { base, jobId, onPick({col,row,ox,oy,height,agl}), onGcpAdd(point) }
   */
  constructor(container, hooks) {
    this.container = container;
    this.hooks = hooks;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.orbit = null;
    this.fly = null;
    this.orbitOn = true;
    this.mesh = null;
    this.heights = null;   // Float32Array, render grid
    this.struct = null;    // Float32Array, render grid
    this.header = null;
    this.mode = 'rgb';
    this.quality = 512;
    this.exag = 6;
    this.gcpMode = false;
    this.gcpMarks = [];
    this.regionMode = false;
    this._regionHeights = null;
    this._regionStruct = null;
    this.regionGridW = 0;
    this.regionGridH = 0;
    this._rgbTex = null;
    this._errTex = null;
    this._overlayTex = null;
    this._ray = new THREE.Raycaster();
    this._anim = null;
    this._last = 0;
  }

  // ---------------------------------------------------------------- lifecycle
  init(header, heights, struct, texUrl, errUrl, region) {
    this.header = header;
    this.heights = heights;
    this.struct = struct;
    if (region) {
      this._regionHeights = region.heights;
      this._regionStruct = region.struct || null;
      this.regionGridW = region.gridW;
      this.regionGridH = region.gridH;
      this.regionOrigW = region.origW;
      this.regionOrigH = region.origH;
    }

    const { grid_w: gw, grid_h: gh } = header;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05070a);

    this.camera = new THREE.PerspectiveCamera(62, 1, 0.1, 1e6);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.08;
    this.orbit.maxPolarAngle = Math.PI * 0.86;

    this.fly = new FlyControls(this.camera, this.renderer.domElement);
    this.fly.movementSpeed = 90;
    this.fly.rollSpeed = 0.18;
    this.fly.dragToLook = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 0.95));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(gw * 0.5, -gh * 0.5, gw * 0.6);
    this.scene.add(sun);

    this._buildTerrain();

    // frame camera around the tile
    const cx = gw / 2, cy = gh / 2, span = Math.max(gw, gh);
    this.camera.position.set(cx, cy - span * 0.55, span * 0.75);
    this.camera.lookAt(cx, cy, 0);
    this.orbit.target.set(cx, cy, 0);
    this.orbit.update();

    this._loadRGB(texUrl);
    if (errUrl) this._loadErr(errUrl);

    this._bindEvents();
    this._resize();
    this._anim = requestAnimationFrame(this._tick);
  }

  dispose() {
    cancelAnimationFrame(this._anim);
    window.removeEventListener('resize', this._onResize);
    if (this._regionCleanup) { try { this._regionCleanup(); } catch (e) { /* noop */ } }
    if (this.mesh) this.mesh.geometry.dispose();
    if (this.renderer) this.renderer.dispose();
    this.container.innerHTML = '';
  }

  // pause/resume the render loop (used when switching to the Deck.gl view)
  pause(on) {
    if (on) {
      cancelAnimationFrame(this._anim);
      this._anim = null;
    } else if (!this._anim) {
      this._last = performance.now();
      this._anim = requestAnimationFrame(this._tick);
    }
  }

  // ---------------------------------------------------------------- controls
  setMode(mode) { this.mode = mode; this._applyMap(); }

  setQuality(q) { this.quality = q; this._buildTerrain(); this._applyMap(); }

  setExag(v) {
    this.exag = v;
    if (!this.mesh) return;
    const { px, py } = this.mesh.geometry.userData;
    const pos = this.mesh.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const h = this.heights[py[i] * this.header.grid_w + px[i]];
      pos.setZ(i, Number.isFinite(h) ? h * this.exag : 0);
    }
    pos.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
  }

  toggleFly() {
    this.orbitOn = !this.orbitOn;
    this.orbit.enabled = this.orbitOn;
    this.fly.enabled = !this.orbitOn;
    return !this.orbitOn;
  }

  isFly() { return !this.orbitOn; }

  flyTo(cx, cy, dist) {
    this.orbit.enabled = true;
    this.orbitOn = true;
    this.fly.enabled = false;
    this.camera.position.set(cx, cy - dist * 0.55, dist * 0.75);
    this.camera.lookAt(cx, cy, 0);
    this.orbit.target.set(cx, cy, 0);
    this.orbit.update();
  }

  setGcpMode(on, onGcpAdd) {
    this.gcpMode = on;
    this._onGcpAdd = onGcpAdd;
    this.renderer.domElement.style.cursor = on ? 'crosshair' : '';
  }

  addGcpMark(point) {
    const mark = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(2, this.header.grid_w * 0.004), 16),
      new THREE.MeshBasicMaterial({ color: 0xffcc33 }),
    );
    const z = this._heightAt(point.col, point.row);
    mark.position.set(point.col, point.row, (Number.isFinite(z) ? z : 0) * this.exag + 50);
    this.scene.add(mark);
    this.gcpMarks.push(mark);
  }

  clearGcpMarks() {
    for (const m of this.gcpMarks) this.scene.remove(m);
    this.gcpMarks = [];
  }

  // ---------------------------------------------------------------- region select
  /** Toggle region-select mode. Region grid must be present to enable. */
  setRegionMode(on) {
    this.regionMode = on;
    this.renderer.domElement.style.cursor = on ? 'crosshair' : '';
    if (on) this._beginRegionDrag();
    else this._clearRegionBox();
  }

  _beginRegionDrag() {
    const el = this.renderer.domElement;
    let box = document.getElementById('region-select-box');
    if (!box) {
      box = document.createElement('div');
      box.id = 'region-select-box';
      this.container.appendChild(box);
    }
    let start = null;

    const onDown = (e) => {
      if (e.button !== 0) return;
      if (this.orbit) this.orbit.enabled = false;
      start = { x: e.clientX, y: e.clientY };
      box.style.display = 'block';
      box.style.left = `${e.clientX}px`;
      box.style.top = `${e.clientY}px`;
      box.style.width = '0px';
      box.style.height = '0px';
      box.dataset.active = '1';
    };
    const onMove = (e) => {
      if (!start || box.dataset.active !== '1') return;
      const x = Math.min(e.clientX, start.x), y = Math.min(e.clientY, start.y);
      box.style.left = `${x}px`;
      box.style.top = `${y}px`;
      box.style.width = `${Math.abs(e.clientX - start.x)}px`;
      box.style.height = `${Math.abs(e.clientY - start.y)}px`;
    };
    const onUp = (e) => {
      if (!start || box.dataset.active !== '1') return;
      box.dataset.active = '0';
      const wdt = Math.abs(e.clientX - start.x), hgt = Math.abs(e.clientY - start.y);
      if (wdt < 5 || hgt < 5) { start = null; box.style.display = 'none'; return; }
      const x0 = Math.min(start.x, e.clientX), y0 = Math.min(start.y, e.clientY);
      start = null;
      this._regionFromScreenRect(x0, y0, x0 + wdt, y0 + hgt);
      box.style.display = 'none';
    };
    const cleanup = () => {
      el.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      if (this.orbit) this.orbit.enabled = true;
      this._clearRegionBox();
    };
    this._regionCleanup = cleanup;
    el.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }

  _clearRegionBox() {
    const box = document.getElementById('region-select-box');
    if (box) box.style.display = 'none';
  }

  /** Screen rect -> raycast 4 corners -> region grid rect -> stats. */
  _regionFromScreenRect(x0, y0, x1, y1) {
    const corners = [
      this._rayToGrid(x0, y0), this._rayToGrid(x1, y0),
      this._rayToGrid(x1, y1), this._rayToGrid(x0, y1),
    ];
    const pts = corners.filter(Boolean);
    if (pts.length < 2) return;

    let minC = Infinity, maxC = -Infinity, minR = Infinity, maxR = -Infinity;
    for (const p of pts) {
      if (p.col < minC) minC = p.col;
      if (p.col > maxC) maxC = p.col;
      if (p.row < minR) minR = p.row;
      if (p.row > maxR) maxR = p.row;
    }
    const stats = this._regionStatsFor(minC, minR, maxC, maxR);
    if (stats && this.hooks.onRegion) this.hooks.onRegion(stats);
  }

  _rayToGrid(clientX, clientY) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - r.left) / r.width) * 2 - 1,
      -((clientY - r.top) / r.height) * 2 + 1,
    );
    this._ray.setFromCamera(ndc, this.camera);
    const hits = this._ray.intersectObject(this.mesh);
    if (!hits.length) return null;
    const pt = hits[0].point;
    const { grid_w: gw, grid_h: gh } = this.header;
    return {
      col: Math.min(gw - 1, Math.max(0, Math.round(pt.x))),
      row: Math.min(gh - 1, Math.max(0, Math.round(pt.y))),
    };
  }

  /** Summarize heights in grid rect [c0..c1, r0..r1] using the region grid. */
  _regionStatsFor(c0, r0, c1, r1) {
    if (!this._regionHeights) return null;
    const gw = this.header.grid_w, gh = this.header.grid_h;
    const rgw = this.regionGridW || gw, rgh = this.regionGridH || gh;
    const scX = (rgw - 1) / (gw - 1 || 1);
    const scY = (rgh - 1) / (gh - 1 || 1);

    const vals = [];
    const structs = [];
    for (let row = Math.max(0, r0); row <= Math.min(gh - 1, r1); row++) {
      for (let col = Math.max(0, c0); col <= Math.min(gw - 1, c1); col++) {
        const gi = Math.round(row * scY) * rgw + Math.round(col * scX);
        const v = this._regionHeights[gi];
        if (Number.isFinite(v)) vals.push(v);
        if (this._regionStruct && Number.isFinite(this._regionStruct[gi])) structs.push(this._regionStruct[gi]);
      }
    }
    if (!vals.length) return null;

    vals.sort((a, b) => a - b);
    const median = vals.length % 2 ? vals[(vals.length - 1) >> 1]
      : 0.5 * (vals[vals.length / 2 - 1] + vals[vals.length / 2]);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance = vals.reduce((a, b) => a + (b - mean) * (b - mean), 0) / vals.length;

    // region centre in original-image pixel coords (for a GCP point)
    const cc = 0.5 * (Math.max(c0, 0) + Math.min(c1, gw - 1));
    const rc = 0.5 * (Math.max(r0, 0) + Math.min(r1, gh - 1));
    const ox = (cc / (gw - 1 || 1)) * (this.regionOrigW - 1 || 0);
    const oy = (rc / (gh - 1 || 1)) * (this.regionOrigH - 1 || 0);

    let structMedian = NaN;
    if (structs.length) {
      structs.sort((a, b) => a - b);
      structMedian = structs.length % 2 ? structs[(structs.length - 1) >> 1]
        : 0.5 * (structs[structs.length / 2 - 1] + structs[structs.length / 2]);
    }

    return {
      n: vals.length,
      min: vals[0], max: vals[vals.length - 1],
      mean, median, sigma: Math.sqrt(variance),
      structMedian,
      ox, oy, height: median,
    };
  }

  // ---------------------------------------------------------------- terrain
  _buildTerrain() {
    const { grid_w: gw, grid_h: gh } = this.header;
    const cols = Math.min(this.quality, gw);
    const rows = Math.min(this.quality, gh);
    const scX = gw > 1 ? (gw - 1) / (cols - 1 || 1) : 1;
    const scY = gh > 1 ? (gh - 1) / (rows - 1 || 1) : 1;

    const px = Float32Array.from({ length: cols * rows }, (_, i) => (i % cols) * scX);
    const py = Float32Array.from({ length: cols * rows }, (_, i) => Math.floor(i / cols) * scY);
    const positions = new Float32Array(cols * rows * 3);
    const uvs = new Float32Array(cols * rows * 2);
    for (let i = 0; i < px.length; i++) {
      const h = this.heights[py[i] * gw + px[i]];
      positions[i * 3] = px[i];
      positions[i * 3 + 1] = py[i];
      positions[i * 3 + 2] = Number.isFinite(h) ? h * this.exag : 0;
      uvs[i * 2] = px[i] / (gw - 1 || 1);
      uvs[i * 2 + 1] = py[i] / (gh - 1 || 1);
    }
    const indices = [];
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        const a = r * cols + c, b = a + 1, d = a + cols, e = d + 1;
        indices.push(a, b, d, b, e, d);
      }
    }

    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    // stash grid coords for height re-read in setExag / picking
    geo.userData = { px, py };

    this.mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
      color: 0xffffff, shininess: 4, side: THREE.DoubleSide,
    }));
    this.mesh.name = 'terrain';
    this.scene.add(this.mesh);
    this._cols = cols; this._rows = rows;

    // rebuild overlays referencing old texture grid dims
    this._applyMap();
  }

  // ---------------------------------------------------------------- maps / overlays
  _loadRGB(url) {
    const loader = new THREE.TextureLoader();
    loader.load(url, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 4;
      this._rgbTex = tex;
      if (this.mode === 'rgb') this._applyMap();
    });
  }

  _loadErr(url) {
    const loader = new THREE.TextureLoader();
    loader.load(url, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      this._errTex = tex;
      if (this.mode === 'error') this._applyMap();
    });
  }

  _applyMap() {
    if (!this.mesh) return;
    let tex = null;
    if (this.mode === 'rgb') tex = this._rgbTex;
    else if (this.mode === 'error') tex = this._errTex;
    else tex = this._buildOverlay(this.mode);
    this.mesh.material.map = tex;
    this.mesh.material.needsUpdate = true;
  }

  _buildOverlay(mode) {
    const { grid_w: gw, grid_h: gh } = this.header;
    const W = Math.min(1024, gw), H = Math.min(1024, gh);
    const sx = gw / W, sy = gh / H;
    const data = new Uint8ClampedArray(W * H * 4);

    // sample height at overlay resolution
    const vals = new Float32Array(W * H);
    for (let j = 0; j < H; j++) {
      const gy = Math.round(j * sy);
      for (let i = 0; i < W; i++) {
        const gx = Math.round(i * sx);
        const v = this.heights[gy * gw + gx];
        vals[j * W + i] = Number.isFinite(v) ? v : NaN;
      }
    }

    if (mode === 'elevation') {
      const [lo, hi] = this.header.h_range;
      this._colorize(vals, data, lo, hi, VIRIDIS);
    } else if (mode === 'slope') {
      const gsd = this.header.gsd_m || 1;
      const slopes = new Float32Array(W * H);
      for (let j = 0; j < H; j++) {
        for (let i = 0; i < W; i++) {
          const c = vals[j * W + i];
          const r = i + 1 < W ? vals[j * W + i + 1] : NaN;
          const dn = j + 1 < H ? vals[(j + 1) * W + i] : NaN;
          const dx = (Number.isFinite(c) && Number.isFinite(r)) ? (r - c) / gsd : NaN;
          const dy = (Number.isFinite(c) && Number.isFinite(dn)) ? (dn - c) / gsd : NaN;
          if (!Number.isFinite(dx) && !Number.isFinite(dy)) { slopes[j * W + i] = NaN; continue; }
          const m = Math.hypot(Number.isFinite(dx) ? dx : 0, Number.isFinite(dy) ? dy : 0);
          slopes[j * W + i] = Math.atan(Math.abs(m)) * 180 / Math.PI;
        }
      }
      this._colorize(slopes, data, 0, 45, MAGMA);
    }
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.putImageData(new ImageData(data, W, H), 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  _colorize(vals, out, lo, hi, stops) {
    const span = (hi - lo) || 1;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (!Number.isFinite(v)) { out[i * 4 + 3] = 0; continue; }
      const t = Math.min(1, Math.max(0, (v - lo) / span));
      const [r, g, b] = ramp(stops, t);
      out[i * 4] = r; out[i * 4 + 1] = g; out[i * 4 + 2] = b; out[i * 4 + 3] = 255;
    }
  }

  // ---------------------------------------------------------------- picking
  _bindEvents() {
    this._onResize = () => this._resize();
    window.addEventListener('resize', this._onResize);
    this.renderer.domElement.addEventListener('click', (e) => this._handleClick(e));
    this.renderer.domElement.addEventListener('dblclick', (e) => {
      if (!this.gcpMode) this._handlePick(e);
    });
  }

  _ndc(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1,
    );
  }

  _intersect(e) {
    this._ray.setFromCamera(this._ndc(e), this.camera);
    const hits = this._ray.intersectObject(this.mesh);
    if (!hits.length) return null;
    const pt = hits[0].point;
    const { grid_w: gw, grid_h: gh, orig_w, orig_h } = this.header;
    const col = Math.min(gw - 1, Math.max(0, Math.round(pt.x)));
    const row = Math.min(gh - 1, Math.max(0, Math.round(pt.y)));
    const height = this._heightAt(col, row);
    const agl = this._structAt(col, row);
    const ox = ((col) / (gw - 1 || 1)) * (orig_w - 1);
    const oy = ((row) / (gh - 1 || 1)) * (orig_h - 1);
    return { col, row, ox, oy, height, agl };
  }

  _handleClick(e) {
    if (this.gcpMode && this.hooks.onGcpAdd) {
      const info = this._intersect(e);
      if (info) this.hooks.onGcpAdd(info);
    }
  }

  _handlePick(e) {
    const info = this._intersect(e);
    if (info && this.hooks.onPick) this.hooks.onPick(info);
  }

  _heightAt(col, row) {
    const { grid_w: gw, grid_h: gh } = this.header;
    const pi = row * gw + col;
    const v = this.heights[pi];
    return Number.isFinite(v) ? v : NaN;
  }

  _structAt(col, row) {
    if (!this.struct) return NaN;
    const { grid_w: gw } = this.header;
    const v = this.struct[row * gw + col];
    return Number.isFinite(v) ? v : NaN;
  }

  // ---------------------------------------------------------------- frame
  _tick = () => {
    const now = performance.now();
    const dt = Math.min(0.1, (now - this._last) / 1000);
    this._last = now;
    if (this.orbitOn && this.orbit) this.orbit.update();
    if (!this.orbitOn && this.fly) this.fly.update(dt);
    if (this.renderer && this.scene && this.camera) this.renderer.render(this.scene, this.camera);
    this._anim = requestAnimationFrame(this._tick);
  };

  _resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
}

// --- tiny built-in perceptual ramps (viridis-ish / magma-ish) --------------
const VIRIDIS = [
  [68, 1, 84], [71, 44, 122], [59, 82, 139], [44, 113, 142], [33, 144, 140],
  [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37],
];
const MAGMA = [
  [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
  [229, 80, 100], [255, 113, 77], [255, 160, 81], [252, 230, 110],
];
function ramp(stops, t) {
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}