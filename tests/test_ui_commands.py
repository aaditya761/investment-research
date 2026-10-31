import pytest

from themescan import ui
from themescan.cli import build_parser


@pytest.fixture(autouse=True)
def _hist(monkeypatch):
    monkeypatch.setattr(ui, "hist_files", lambda: ["hist_global.csv", "hist_india.csv"])


CASES = {
    "prices": {}, "earnings": {}, "earnings+refresh": {"refresh": True}, "newstone": {},
    "altfetch": {}, "altfetch+all": {"news": True, "refresh": True},
    "fundamentals": {"themes": ["gold"]}, "headlines": {"n": 3, "offline": True}, "calendar": {"days": 14},
    "chains": {"offline": True}, "weekly": {"offline": True}, "newsarchive": {}, "narratives": {"month": "2026-09"},
    "clusters": {"seed": "g:ai_semis", "offline": True}, "backtest": {"start": "2020-01-01", "step": 5},
    "research": {"hist": "hist_global.csv", "validate": "hist_india.csv"},
    "portfolio": {"hist": "hist_global.csv", "n": 5, "every": 4, "cost": 10}, "log": {"offline": True}, "scorelog": {},
}


@pytest.mark.parametrize("name,params", CASES.items())
def test_every_ui_job_is_a_valid_cli_invocation(name, params):
    cmd = name.split("+")[0]
    argv = ui.build_argv(cmd, dict(params, universe="themes.yaml"))
    build_parser().parse_args(argv)  # raises SystemExit if the real CLI would reject these arguments
