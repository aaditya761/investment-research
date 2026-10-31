import pandas as pd

from themescan import news as nw


def test_tone_direction_and_negation():
    assert nw.tone("Micron stock surges as AI demand lifts forecast") > 0.3
    assert nw.tone("Chip stocks plunge as fears of a slowdown grow") < -0.3
    assert nw.tone("Stocks not rallying as investors wait") < 0  # negation flips the sign
    assert nw.tone("Company announces date for annual meeting") == 0


def test_catalyst_tags():
    c = nw.catalysts("Nvidia raises guidance as data center demand booms")
    assert "earnings/guidance" in c and "capex/demand" in c
    assert "policy/regulation" in nw.catalysts("US tightens export controls on chips")


def _archive():
    rows = []
    for m in ["2022-01", "2022-02", "2022-03", "2022-04", "2022-05"]:
        for th in ["a", "b", "c", "d"]:
            for i in range(10):
                rows.append(dict(month=m, theme=th, title=f"Routine update number {i} for {th}", date=pd.Timestamp(m + "-10"), source="x"))
    for th in ["a", "b", "c", "d"]:  # a new phrase appears in every theme in 2022-06
        for i in range(10):
            rows.append(dict(month="2022-06", theme=th, title=f"Generative boom reshapes {th} sector {i}", date=pd.Timestamp("2022-06-10"), source="x"))
    return pd.DataFrame(rows)


def test_term_bursts_find_new_cross_theme_phrase():
    b = nw.term_bursts(_archive(), "2022-06", min_df=8, min_themes=3)
    assert len(b) and any("generative" in t for t in b.term.head(5))
    assert b.themes.min() >= 3


def test_narrative_spread_counts_themes():
    sp = nw.narrative_spread(_archive(), r"\bgenerative\b")
    assert sp.loc["2022-06", "themes_over_15pct"] == 4 and sp.loc["2022-03", "themes_over_15pct"] == 0


def test_llm_json_parser_and_fallback(monkeypatch):
    assert nw.parse_llm_json('noise [{"direction":"BULLISH","catalyst":"capex","importance":9}]', 1) == [dict(direction="bullish", catalyst="capex", importance=5)]
    assert nw.parse_llm_json("not json", 1) is None
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert nw.classify_llm(["x"], "Gold") is None  # no key -> graceful None


def test_divergence_flags():
    assert nw.divergence(80, -0.3, -0.1) == "price strong, news souring"
    assert nw.divergence(20, 0.3, 0.2) == "price weak, news improving"
    assert nw.divergence(55, 0.0, 0.0) == ""


def test_date_words_are_not_terms():
    t = nw.terms("Stocks slide on Friday October 30 as chip fears grow")
    assert not any(w in t for w in ("friday", "october", "friday october"))


def test_importing_yfinance_through_wrapper_keeps_warnings_silenced():
    import warnings

    from themescan import _yf  # noqa: F401

    assert warnings.filters[0][0] == "ignore"  # yfinance's own 'default' filter must not stay on top
