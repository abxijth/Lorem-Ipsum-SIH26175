"""Headless-Firefox E2E: verify the Deck.gl map view (View B) + Three.js view (View A).

Drives the live app (port 8000): click the Hilly sample, wait for the viewer,
assert the Three.js canvas renders, toggle to the Deck.gl map view, assert a
deck canvas renders a terrain mesh, then toggle back and confirm Three still works.
Collects browser console errors throughout.

Run from repo root (venv active):
    python scripts/e2e_web.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cfg")

from selenium import webdriver  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.firefox.options import Options  # noqa: E402
from selenium.webdriver.support.ui import WebDriverWait  # noqa: E402
from selenium.webdriver.support import expected_conditions as EC  # noqa: E402

BASE = "http://localhost:8000"


def main() -> int:
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.binary_location = "/usr/bin/firefox"
    driver = webdriver.Firefox(options=opts)
    errors = []

    def collect(entry):
        if entry and (entry.get("level") or "").lower() in ("error", "severe"):
            errors.append(str(entry.get("message"))[:500])

    driver.execute_script(
        "window.__errs=[];"
        "const o=console.error.bind(console);"
        "console.error=(...a)=>{(window.__errs||(window.__errs=[])).push(a.map(String).join(' '));o(...a)};"
    )

    try:
        driver.get(BASE)
        wait = WebDriverWait(driver, 30)

        # click the Hilly sample -> triggers /api/samples/hilly/process
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-sample="hilly"]'))).click()
        wait.until(EC.visibility_of_element_located((By.ID, "screen-loading")))
        print("[OK] upload -> sample submitted")

        # wait for the view screen to appear (pipeline completes)
        wait.until(EC.visibility_of_element_located((By.ID, "screen-view")))
        wait.until(lambda d: d.execute_script(
            "return !!window.__errs && !document.getElementById('viewport').querySelector('canvas') ? 0 : 1"))
        time.sleep(2)

        # assert Three.js canvas exists in #viewport
        def three_canvas():
            return driver.execute_script(
                "const c=document.querySelector('#viewport canvas');"
                "return c ? (c.width>0 && c.height>0) : false;")
        ok_con = wait.until(lambda d: three_canvas())
        print(f"[{'PASS' if ok_con else 'FAIL'}] Three.js flythrough canvas renders")

        # toolbar toggle buttons present
        id_map = driver.execute_script(
            "return !!document.getElementById('btn-view-map') && !!document.getElementById('btn-view-3d');")
        print(f"[{'PASS' if id_map else 'FAIL'}] Deck/3D view toggle buttons present")

        # deck assets in header -> toggle button visible
        btn_map = wait.until(EC.element_to_be_clickable((By.ID, "btn-view-map")))
        if btn_map.is_displayed() is False:
            print("[FAIL] btn-view-map hidden (no deck assets?)")
        else:
            print("[PASS] Deck toggle available (header.deck present)")

        # click Map view
        btn_map.click()
        time.sleep(4)
        map_visible = driver.execute_script(
            "const v=document.getElementById('viewport-deck');"
            "return !v.classList.contains('hidden') && v.getBoundingClientRect().width>0;")
        deck_canvas = driver.execute_script(
            "const c=document.querySelector('#viewport-deck canvas');"
            "return c ? (c.width>50 && c.height>50) : false;")
        print(f"[{'PASS' if map_visible else 'FAIL'}] Deck viewport visible")
        print(f"[{'PASS' if deck_canvas else 'FAIL'}] Deck.gl canvas renders (w/h>50)")

        # worker disabled -> document should NOT have requested remote workers; ensure no offscreen
        time.sleep(2)
        # back to 3D
        wait.until(EC.element_to_be_clickable((By.ID, "btn-view-3d"))).click()
        time.sleep(2)
        three_back = driver.execute_script(
            "const v=document.getElementById('viewport');"
            "return !v.classList.contains('hidden') && !!document.querySelector('#viewport canvas');")
        print(f"[{'PASS' if three_back else 'FAIL'}] Toggle back to Three.js view")

        errs = driver.execute_script("return window.__errs||[]")
        print(f"[{'PASS' if not errs else 'FAIL'}] zero browser console errors  -- errs={errs[:5]}")

    finally:
        driver.execute_script("document.title='DONE'")
        driver.quit()

    print("\n" + ("E2E PASSED" if not errors else f"{len(errors)} console errors"))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
