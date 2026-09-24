"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import gn_constants as C
import gn_evaluation as E
import gn_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("n", "n_slider"), ("radius", "radius_slider"), ("share", "share_slider"), ("frac", "frac_slider"), ("known", "known_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-9
        assert p["features"] in C.FEATURE_MODES and p["rule"] in C.RULES


def test_standard_preset_equals_the_default_settings_except_for_the_seed():
    p = P.PRESETS["Standardfall (Anzahl ≥ 3)"]
    assert E.Settings(p["n"], p["radius"], p["share"], p["features"], p["rule"], float(p["count"]), p["known"], p["layers"], p["seed"]) == E.Settings(seed=p["seed"], threshold=float(C.DEFAULT_COUNT))


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and set(P.STEPS) == {"n_slider", "radius_slider", "share_slider", "frac_slider", "known_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert P.SETTING_SPECS["rule_select"].caster("Dichte") == "Dichte"
