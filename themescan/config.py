from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Theme:
    key: str
    label: str
    etfs: list[str] = field(default_factory=list)
    stocks: list[str] = field(default_factory=list)

    @property
    def tickers(self) -> list[str]:
        return list(dict.fromkeys(self.etfs + self.stocks))


@dataclass
class Config:
    benchmark: str
    themes: dict[str, Theme]
    events: list[dict]

    @property
    def all_tickers(self) -> list[str]:
        out = {self.benchmark}
        for t in self.themes.values():
            out.update(t.tickers)
        return sorted(out)


def load_config(path: str | Path = "themes.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    themes = {
        k: Theme(k, v.get("label", k), [str(x) for x in v.get("etfs", [])], [str(x) for x in v.get("stocks", [])])
        for k, v in raw["themes"].items()
    }
    return Config(str(raw.get("benchmark", "ACWI")), themes, raw.get("events", []))
