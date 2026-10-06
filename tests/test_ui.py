import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from themescan import ui


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_config(server):
    html = urllib.request.urlopen(server + "/").read().decode()
    assert "Themescan" in html
    code, cfg = call(server, "/api/config")
    assert code == 200 and "themes.yaml" in cfg["universes"]


def test_scan_and_detail_demo(server):
    code, d = call(server, "/api/scan", {"universe": "themes.yaml", "demo": True})
    assert code == 200 and len(d["rows"]) > 20 and {"score", "stage", "leading"} <= set(d["rows"][0])
    code, t = call(server, "/api/detail", {"universe": "themes.yaml", "demo": True, "theme": d["rows"][0]["index"]})
    assert code == 200 and t["rows"]


def test_inputs_are_validated(server):
    assert call(server, "/api/scan", {"universe": "../../etc/passwd"})[0] == 400
    assert call(server, "/api/run/research", {"hist": "../../etc/passwd"})[0] == 400
    assert call(server, "/api/run/rm", {})[0] == 400


def test_job_runs_and_streams(server):
    code, j = call(server, "/api/run/headlines", {"universe": "themes.yaml", "demo": True, "themes": ["gold"], "n": 1})
    assert code == 200 and "themescan" in j["command"]
    for _ in range(60):
        _, s = call(server, f"/api/job/{j['id']}?since=0")
        if s["done"]:
            break
        time.sleep(0.5)
    assert s["done"]
