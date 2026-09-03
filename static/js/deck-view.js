// Deck.gl map-style terrain view (View B). Uses the vendored deck.gl UMD build
// (window.deck) with a ScatterplotLayer point cloud showing only the terrain
// structure (no image drape), colorized by height over a dark map background.
// over a dark background. No build step, no runtime CDN, no worker dependency.

// Viridis-ish colormap for elevation coloring
const VIRIDIS = [
  [68, 1, 84], [71, 44, 122], [59, 82, 139], [44, 113, 142], [33, 144, 140],
  [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37],
];

function viridisColor(t) {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
    200,
  ];
}

export class DeckView {
  /**
   * @param {HTMLElement} container  the #viewport-deck element
   * @param {object} cfg  { header, jobId, onPickHeight({height, lon, lat}) }
   */
  constructor(container, cfg) {
    if (!window.deck || typeof window.deck.Deck !== 'function') {
      throw new Error('deck.gl failed to load');
    }
    this.container = container;
    this.cfg = cfg;
    this.deck = null;
  }

  /** Build the view from a header with a `deck` block + resolved asset URLs + Float32 heights. */
  init(header, urls, heightsBuf) {
    const d = header && header.deck;
    if (!d) throw new Error('no deck assets in header');
    this.header = header;

    const [w, s, e, n] = d.bounds || [0, 0, 0, 0];
    const lon = (w + e) / 2;
    const lat = (s + n) / 2;
    const lonSpan = Math.abs(e - w) || 0.001;
    const latSpan = Math.abs(n - s) || 0.001;

    // compute zoom so terrain fills ~70% of a 1400px-wide viewport
    const cosLat = Math.cos(lat * Math.PI / 180);
    const terrainMeters = lonSpan * cosLat * 111320;
    const targetMpp = terrainMeters / (0.7 * 1400);
    const initZoom = Math.min(19, Math.max(4,
      Math.log2(40075000 * cosLat / (256 * targetMpp))));

    // build point positions + colors from the Float32 heights
    const gw = header.grid_w, gh = header.grid_h;
    const heights = new Float32Array(heightsBuf);
    const hMin = header.h_range ? header.h_range[0] : 0;
    const hMax = header.h_range ? header.h_range[1] : 1;
    const hSpan = (hMax - hMin) || 1;

    // subsample to at most 400×400 for performance
    const maxPts = 400;
    const step = Math.max(1, Math.floor(Math.max(gw, gh) / maxPts));

    const positions = [];
    const colors = [];
    for (let row = 0; row < gh; row += step) {
      for (let col = 0; col < gw; col += step) {
        const h = heights[row * gw + col];
        if (!Number.isFinite(h)) continue;
        const pLon = w + (col / (gw - 1)) * lonSpan;
        const pLat = s + (row / (gh - 1)) * latSpan;
        positions.push([pLon, pLat, h]);
        colors.push(viridisColor((h - hMin) / hSpan));
      }
    }

    const pointSize = Math.max(2, Math.min(12, Math.floor(12000 / positions.length)));

    const layers = [
      new window.deck.ScatterplotLayer({
        id: 'terrain-points',
        data: positions,
        getPosition: (d) => d,
        getFillColor: (d, { index }) => colors[index] || [128, 128, 128, 200],
        getRadius: pointSize,
        radiusUnits: 'pixels',
        pickable: true,
        onClick: (info) => this._onPick(info),
      }),
    ];

    this.deck = new window.deck.Deck({
      parent: this.container,
      controller: { maxPitch: 89.5 },
      initialViewState: {
        longitude: lon,
        latitude: lat,
        zoom: initZoom,
        pitch: 55,
        bearing: -25,
        minZoom: 4,
        maxZoom: 19,
      },
      layers,
      view: new window.deck.MapView({ repeat: false }),
      clearColor: [10, 10, 10, 255],
      getTooltip: () => null,
    });

    try { window.__deckInst = this.deck; } catch (e) { /* ignore */ }
    this._resize();
  }

  _onPick(info) {
    if (!info || !this.cfg.onPickHeight) return;
    const pos = info.coordinate || (info.object);
    let height = NaN, lon = NaN, lat = NaN;
    if (Array.isArray(pos)) {
      lon = pos[0]; lat = pos[1]; height = pos[2];
    }
    this.cfg.onPickHeight({ height, lon, lat });
  }

  resize() { this._resize(); }

  _resize() {
    if (!this.deck) return;
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.deck.setProps({ width: w, height: h });
  }

  dispose() {
    if (this.deck) {
      try { this.deck.finalize(); } catch (e) { /* noop */ }
      this.deck = null;
    }
    this.container.innerHTML = '';
  }
}
