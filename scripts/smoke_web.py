"""End-to-end smoke test for the web API.

Usage (from repo root, venv active):
    python scripts/smoke_web.py [--case hilly|forest] [--no-refit]

Boots the FastAPI app via TestClient (no server needed), POSTs a georeferenced
sample + bbox + reference, polls the job to completion, asserts metrics close to
the validated numbers, downloads the DSM asset, and exercises a GCP refit
round-trip. Prints PASS/FAIL per check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cfg")
os.makedirs("/tmp/mpl-cfg", exist_ok=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

EXPECTED_RMSE = {"hilly": 3.1, "forest": 8.2}  # validated numbers (fine-tuned)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="hilly", choices=["hilly", "forest"])
    ap.add_argument("--no-refit", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    case_dir = root / "data" / "external" / f"{args.case}_asheville" if args.case == "hilly" \
        else root / "data" / "external" / f"forest_gsmnp"
    if args.case == "hilly":
        case_dir = root / "data" / "external" / "hilly_asheville"
    meta = json.loads((case_dir / "meta.json").read_text())
    bbox = " ".join(str(v) for v in meta["bbox"])

    client = TestClient(app)
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # 1. process
    print("POST /api/process")
    with (case_dir / "input.png").open("rb") as img, (case_dir / "reference.tif").open("rb") as ref:
        resp = client.post(
            "/api/process",
            files={"image": ("input.png", img, "image/png"),
                   "reference": ("ref.tif", ref, "image/tiff")},
            data={"bbox": bbox},
        )
        check("process returns job", resp.status_code == 200, f"status {resp.status_code}")
        jid = resp.json()["job_id"]

    # 2. poll
    deadline = time.time() + 900
    status = None
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{jid}")
        status = r.json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(3)
    check("job completes", status["status"] == "done", f"status={status['status']} err={status.get('error')}")
    check("rmse in range", status["metrics"].get("rmse", 999) < EXPECTED_RMSE[args.case] + 0.5,
          f"rmse={status['metrics'].get('rmse')}")
    check("metrics complete", all(k in status["metrics"] for k in ("rmse", "mae", "pearson", "bias", "n")))

    for asset_name, url in status["header"].get("assets", {}).items():
        resp = client.get(f"/api/jobs/{jid}/asset/{url}")
        ok = resp.status_code == 200 and len(resp.content) > 0
        check(f"asset {asset_name} serves", ok, f"status {resp.status_code}")

    # Deck.gl TerrainLayer assets (View B)
    deck = status["header"].get("deck")
    check("deck header present", deck is not None)
    if deck:
        for name in ("heightsUrl", "textureUrl"):
            resp = client.get(f"/api/jobs/{jid}/asset/{deck[name]}")
            ok = resp.status_code == 200 and len(resp.content) > 0
            check(f"deck {name} serves", ok, f"status {resp.status_code}")
        check("deck decoder present",
              all(k in deck.get("elevationDecoder", {}) for k in ("rScaler", "gScaler", "bScaler")),
              f"decoder={deck.get('elevationDecoder')}")
        check("deck bounds present", len(deck.get("bounds") or []) == 4, f"bounds={deck.get('bounds')}")

    resp = client.get(f"/api/jobs/{jid}/download")
    check("dsm download serves", resp.status_code == 200 and resp.content[:2] == b"II")

    # 3. refit round-trip (click-ish GCP: two points + plausible heights)
    if not args.no_refit:
        # pick two pixels near the middle; heights = DSM values there (+small delta)
        hdr = status["header"]
        ox, oy = hdr["orig_w"] // 2, hdr["orig_h"] // 2
        body = {
            "points": [
                {"x": ox, "y": oy, "h": 700.0},
                {"x": ox + 40, "y": oy + 40, "h": 680.0},
            ]
        }
        resp = client.post(f"/api/jobs/{jid}/refit", json=body)
        check("refit accepted", resp.status_code == 200, f"status {resp.status_code}")
        deadline = time.time() + 300
        while time.time() < deadline:
            r = client.get(f"/api/jobs/{jid}")
            st = r.json()
            if st["status"] in ("done", "error"):
                break
            time.sleep(2)
        check("refit completes", st["status"] == "done", f"status={st['status']} err={st.get('error')}")

    client.delete(f"/api/jobs/{jid}")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURES: {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())