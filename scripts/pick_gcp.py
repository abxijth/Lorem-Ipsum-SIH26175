"""Interactive helper: click ground-control points on an image, enter known
heights, and print a ready-to-paste --gcp string for depthwizard.cli.

Usage:
    python scripts/pick_gcp.py IMAGE [--n 2] [--out gcp.txt]

Click N points on the image, then press 'q' (or close the window). For each
point you'll be asked for its known height (metres). The --gcp string is printed
(and optionally written to --out).

Tip: pick two points with *different* known heights for a slope+intercept refit;
a single point only shifts the calibration level.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from depthwizard.georef import _read_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="image or GeoTIFF to examine")
    ap.add_argument("--n", type=int, default=2, help="number of points to click")
    ap.add_argument("--out", default="", help="file to write the --gcp string to")
    args = ap.parse_args()

    rgb = _read_image(Path(args.image))
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except Exception:
        sys.exit("interactive display unavailable; pick coordinates manually "
                 "and pass them via --gcp 'x,y,height;...'")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb)
    ax.set_title(f"click {args.n} points (then press 'q'): min-height and max-height points work best")
    points: list[tuple[float, float]] = []

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        points.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, "r+", ms=18, mew=2)
        ax.set_title(f"{len(points)}/{args.n} points (press 'q' when done)")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not points:
        sys.exit("no points clicked; aborting")
    points = points[: args.n]

    x, y = zip(*points)
    lo, hi = min(x), max(x)
    std = np.ptp(np.asarray(x))
    print("\npixels (round to image coords, 0-based top-left):")
    print(f"  window of X click range: {lo:.0f} .. {hi:.0f}  (range {hi-lo:.0f} px, "
          f"~{std:.0f}px, approx GSD unknown)")

    heights = []
    for i, (px, py) in enumerate(points):
        try:
            h = float(input(f"  point {i + 1} at ({px:.0f}, {py:.0f}) — known height (m) > "))
        except (EOFError, KeyboardInterrupt):
            sys.exit("\naborted")
        heights.append(h)

    gcp = ";".join(f"{px:.0f},{py:.0f},{h:.2f}" for (px, py), h in zip(points, heights))
    print("\n--gcp string:\n  " + gcp)
    if args.out:
        Path(args.out).write_text(gcp)
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()