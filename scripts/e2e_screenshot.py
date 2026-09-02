"""Capture screenshots of the Three.js view and Deck.gl map view for visual QA.

Usage: python scripts/e2e_screenshot.py [outdir]
Produces three.png (Three flythrough) and deck.png (Deck.gl map) in outdir.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cfg")

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE = "http://localhost:8000"
outdir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def main() -> int:
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.binary_location = "/usr/bin/firefox"
    opts.add_argument("--window-size=1400,900")
    driver = webdriver.Firefox(options=opts)
    try:
        driver.get(BASE)
        wait = WebDriverWait(driver, 30)
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-sample="hilly"]'))).click()
        wait.until(EC.visibility_of_element_located((By.ID, "screen-view")))
        wait.until(lambda d: d.execute_script(
            "const c=document.querySelector('#viewport canvas');"
            "return c && c.width>0 && c.height>0;"))
        time.sleep(3)
        (outdir / "three.png").write_bytes(driver.get_screenshot_as_png())
        print("three.png saved")

        wait.until(EC.element_to_be_clickable((By.ID, "btn-view-map"))).click()
        time.sleep(6)
        (outdir / "deck.png").write_bytes(driver.get_screenshot_as_png())
        print("deck.png saved")
    finally:
        driver.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
