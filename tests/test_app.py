"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Regel-Wechsel mit passendem Regler, Karten, Paar, Würfel-Knopf, Permalink-Grenzen, Extremwerte, Experiment auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gn_constants as C
import gn_evaluation as E
import gn_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_shows_the_sum_ahead_and_all_charts():
    at = _run()
    _ok(at)
    assert len(at.metric) >= 6 and len(at.get("plotly_chart")) == 3 and any("Die Summe zählt" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_share_preset_shows_the_mean_winning_verdict():
    at = _run()
    next(b for b in at.button if b.key == "preset_Anteil: Mittelwert genügt").click().run()
    _ok(at)
    assert any("Hier gewinnt der Mittelwert" in w.value for w in at.warning)


@pytest.mark.parametrize("rule,key", [("Anteil", "frac_slider"), ("Anzahl", "count_slider"), ("Dichte", "deg_slider")])
def test_rule_shows_only_its_own_threshold_slider(rule, key):
    at = _run(rule_select=rule)
    _ok(at)
    keys = {s.key for s in at.slider}
    assert key in keys and {"frac_slider", "count_slider", "deg_slider"} - {key} <= {"frac_slider", "count_slider", "deg_slider"} and not ({"frac_slider", "count_slider", "deg_slider"} - {key}) & keys


def test_constant_only_features_with_type_rule_show_the_note_and_no_pair():
    at = _run(features_select=C.FEATURE_MODES[1], rule_select="Anzahl")
    _ok(at)
    assert any("Modus 'nur Konstante'" in i.value for i in at.info) and not any("Ob das für die Regel" in c.value for c in at.caption)


def test_map_views_and_epoch_slider_run():
    at = _run()
    for view in ("Vorhersage des GIN", "Wahres Etikett", "Grad"):
        at.radio(key="map_mode").set_value(view).run()
        _ok(at)
    at.slider(key="epoch_slider").set_value(1).run()
    _ok(at)


def test_pair_table_is_shown_with_equal_share():
    at = _run()
    _ok(at)
    assert any("Beide Kunden haben denselben Anteil" in c.value for c in at.caption)


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Gebiet generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["n"] = "9999"
    at.query_params["radius"] = "5.7"
    at.query_params["frac"] = "0.33"
    at.query_params["count"] = "99"
    at.query_params["features"] = "Unsinn"
    at.query_params["rule"] = "Anteil"
    at.query_params["layers"] = "abc"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["radius_slider"] == 5.5 and at.session_state["frac_slider"] == 0.35 and at.session_state["count_slider"] == C.COUNT_MAX
    assert at.session_state["features_select"] == C.FEATURE_MODES[0] and at.session_state["rule_select"] == "Anteil" and at.session_state["layers_slider"] == C.DEFAULT_LAYERS


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN, radius_slider=C.RADIUS_MIN), dict(n_slider=C.N_MAX, radius_slider=C.RADIUS_MAX), dict(known_slider=C.KNOWN_MIN, layers_slider=3),
                                dict(share_slider=C.SHARE_A_MIN, rule_select="Anteil", frac_slider=C.FRAC_MAX), dict(rule_select="Dichte", deg_slider=C.DEG_MAX), dict(rule_select="Anzahl", count_slider=C.COUNT_MAX, share_slider=C.SHARE_A_MIN)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_aggregation_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0,))
    at = _run()
    next(b for b in at.button if b.key == "agg_start").click().run()
    _ok(at)
    assert at.session_state["agg_on"] and sum(1 for w in at.warning if w.value.startswith("**Befund:**")) == 3 and len(at.get("plotly_chart")) == 6


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo auch das GIN nicht mehr unterscheidet" in s.value for s in at.subheader)
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "{pts(" not in el.value
