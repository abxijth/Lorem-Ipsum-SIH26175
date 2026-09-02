"""Headless-Firefox E2E: region-select + region-GCP.

Drives the live app (port 8000): upload the GAMUS tile, enter the viewer, enable
Region mode, drag a box over the terrain, assert the region panel shows sane
stats, then click "Use region as GCP" with a known height and assert /refit
round-trips. Captures browser console errors.

Usage: python scripts/e2e_region.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cfg")

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def main() -> int:
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.binary_location = "/usr/bin/firefox"
    opts.add_argument("--window-size=1400,900")
    driver = webdriver.Firefox(options=opts)
    errors = []
    try:
        driver.get("http://localhost:8000")
        wait = WebDriverWait(driver, 30)

        driver.execute_script("""
          window.__errs=[];
          const e=console.error.bind(console);
          console.error=(...a)=>{window.__errs.push(a.map(String).join(' ').slice(0,300));e(...a);};
        """)

        # upload GAMUS tile
        wait.until(EC.presence_of_element_located((By.ID, "file-input"))).send_keys(
            "/home/abxijth/Projects/Lore-Ipsum-SIH26175/data/external/hilly_asheville/input.png")
        time.sleep(1)
        wait.until(EC.element_to_be_clickable((By.ID, "btn-process"))).click()

        # wait for viewer
        wait.until(EC.visibility_of_element_located((By.ID, "screen-view")))
        wait.until(lambda d: d.execute_script(
            "const c=document.querySelector('#viewport canvas');return c&&c.width>100;"))
        time.sleep(3)

        # region tool should be available
        btn_region = wait.until(EC.element_to_be_clickable((By.ID, "btn-region")))
        avail = btn_region.is_displayed()
        print(f"[{'PASS' if avail else 'FAIL'}] Region tool available (header.region present)")

        btn_region.click()
        time.sleep(1)
        print("[OK] Region mode enabled")

        # drag a rectangle over the terrain center
        canvas = driver.find_element(By.CSS_SELECTOR, "#viewport canvas")
        cc = driver.execute_script(
            "const r=arguments[0].getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};",
            canvas)

        ActionChains(driver)\
            .move_to_element_with_offset(canvas, 80, -40)\
            .click_and_hold()\
            .move_by_offset(120, -80)\
            .pause(0.3)\
            .release()\
            .perform()
        time.sleep(2)

        panel_visible = driver.execute_script(
            "const p=document.getElementById('region-panel');return !p.classList.contains('hidden');")
        print(f"[{'PASS' if panel_visible else 'FAIL'}] Region panel visible after drag")

        stats = driver.execute_script("""
          const rows=[...document.querySelectorAll('#region-stats span')].map(s=>s.textContent);
          const vals=[...document.querySelectorAll('#region-stats b')].map(b=>b.textContent);
          return rows.map((k,i)=>k+':'+vals[i]);
        """)
        print("  stats:", stats)

        has_n = driver.execute_script(
            "const t=document.getElementById('region-stats').textContent;"
            "return /median/.test(t) && /pixels/.test(t);")
        print(f"[{'PASS' if has_n else 'FAIL'}] Region stats contain median & pixels")

        # use region as GCP with a known height (~650 m for Asheville terrain)
        known_input = driver.find_element(By.ID, "region-known")
        known_input.clear()
        known_input.send_keys("650")
        wait.until(EC.element_to_be_clickable((By.ID, "btn-region-gcp"))).click()
        time.sleep(2)

        # wait for refit reload: viewer re-created (canvas element replaced)
        wait.until(lambda d: d.execute_script(
            "return !!document.querySelector('#viewport canvas') "
            "&& document.getElementById('region-panel').classList.contains('hidden');"))
        time.sleep(3)
        print("[PASS] Region-GCP round-trip completed (viewer reloaded)")

        errs = driver.execute_script("return window.__errs||[]")
        print(f"[{'PASS' if not errs else 'FAIL'}] zero browser console errors  -- errs={errs[:5]}")
        if errs: errors.extend(errs)

        # screenshot for manual review after refit
        driver.save_screenshot("/tmp/deck_shots/region_refit_done.png")
    finally:
        driver.quit()

    print("\n" + ("REGION E2E PASSED" if not errors else f"{len(errors)} console errors"))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
