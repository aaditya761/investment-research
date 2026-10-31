import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parents[1] / "themescan" / "ui.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def render(lines):
    src = HTML.read_text()
    a = src.index("/* ---------- turn plain command output into a readable report ---------- */")
    b = src.index("/* ---------- jobs:")
    js = ("const esc = s => String(s??'').replace(/[&<>\"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));\n" + src[a:b] +
          "\nconsole.log(renderReport(JSON.parse(require('fs').readFileSync(0, 'utf8'))));")
    import json

    out = subprocess.run(["node", "-e", js], input=json.dumps(lines), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_tight_negative_numbers_stay_in_their_columns():
    html = render(["      ic_21d  ic_63d  ic_126d", "accel  -0.009  -0.005  0.007", "rs_chg -0.011 -0.020 -0.027", "d200   0.043  0.064  0.108"])
    assert "<table" in html and html.count("<tr") == 4
    assert re.search(r"rs_chg</td><td class=\"r neg\">-0.011</td><td class=\"r neg\">-0.020</td><td class=\"r neg\">-0.027", html)


def test_headings_bullets_dated_items_and_italic_lines():
    html = render(["== BEING BOUGHT ==", "- first point", "  19 Oct 2026  Gold rallies on safe-haven demand", "_No earlier snapshot._"])
    assert "<h3>BEING BOUGHT</h3>" in html and "<li>first point</li>" in html and 'class="dt">19 Oct 2026' in html and "<i>No earlier snapshot.</i>" in html


def test_html_in_output_is_escaped():
    assert "<script>" not in render(["<script>alert(1)</script>"])
