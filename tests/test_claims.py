"""Jede Zahl aus README und PRESET_HELP als Test. Einzelläufe nur mit Strukturgrenzen (plattformrobust), Mehr-Seed-Zahlen mit großzügigen Bändern, Ausdrucksstärke exakt."""

import pytest

import gn_constants as C
import gn_evaluation as E
import gn_presets as P


def _preset(name):
    p = P.PRESETS[name]
    thr = {"Anteil": p["frac"], "Anzahl": float(p["count"]), "Dichte": float(p["deg"])}[p["rule"]]
    return E.analyse(E.Settings(p["n"], p["radius"], p["share"], p["features"], p["rule"], thr, p["known"], p["layers"], p["seed"]))


def test_standard_preset_the_sum_beats_the_mean_networks_and_the_mlp_is_at_chance():
    a = _preset("Standardfall (Anzahl ≥ 3)")
    assert a.acc("gin") - a.acc("sage") > 0.02 and a.acc("gin") - a.acc("gcn") > 0.05 and a.acc("gin") > 0.85 and abs(a.acc("mlp") - a.majority_rate()) < 0.08


def test_share_preset_the_mean_wins():
    a = _preset("Anteil: Mittelwert genügt")
    assert a.acc("sage") - a.acc("gin") > 0.02 and a.acc("sage") >= 0.95 and a.acc("gin") - a.acc("mlp") > 0.1


def test_density_preset_the_sum_leads_by_a_lot():
    a = _preset("Dichte (Grad ≥ 7)")
    assert a.acc("gin") - max(a.acc("sage"), a.acc("gcn"), a.acc("gin_mean")) > 0.05 and a.acc("mlp") < a.majority_rate() + 0.05


def test_density_without_features_only_the_sum_sees_the_degree():
    a = _preset("Dichte ohne Merkmale")
    assert a.acc("gin") >= 0.9 and a.acc("sage") < 0.7 and a.acc("gin_mean") < 0.7 and a.acc("gcn") > a.acc("sage") + 0.05 and abs(a.acc("mlp") - a.majority_rate()) < 0.1


def test_few_known_customers_and_one_layer_presets():
    a = _preset("Wenige bekannte Kunden (10 %)")
    assert int(a.train_mask.sum()) == 30 and a.acc("gin") - a.acc("sage") > 0.03
    b = _preset("Eine Schicht")
    assert b.acc("gin") - b.acc("sage") > 0.1 and b.acc("gin") >= 0.9


@pytest.fixture(scope="module")
def rows():
    return {(r["rule"], r["features"]): r for r in E.aggregation_experiment()}


def test_share_rule_the_mean_wins_and_the_unscaled_sum_falls_far_behind(rows):
    r = rows[("Anteil", C.FEATURE_MODES[0])]
    assert r["gin_vs_sage"] < -0.04 and r["wins_sage"] == 0 and r["sage"] >= 0.97 and r["gin_vs_mean"] > 0.0 and abs(r["chance"] - 0.688) < 0.03 and 0.28 < r["positive"] < 0.35
    assert r["raw_vs_gin"] < -0.09 and r["gin_raw_train"] < r["gin_train"] - 0.08 and r["gin"] > r["gcn"] + 0.1


def test_count_rule_the_sum_wins_in_every_area(rows):
    r = rows[("Anzahl", C.FEATURE_MODES[0])]
    assert r["gin_vs_sage"] > 0.04 and r["wins_sage"] == 12 and r["gin_vs_mean"] > 0.05 and r["raw_vs_gin"] < -0.01 and r["gin"] > 0.92 and r["gcn"] < r["sage"]


def test_density_rule_the_sum_wins_by_ten_points(rows):
    r = rows[("Dichte", C.FEATURE_MODES[0])]
    assert r["gin_vs_sage"] > 0.07 and r["wins_sage"] >= 11 and r["gin_vs_mean"] > 0.07 and r["gin"] > 0.95 and abs(r["gin_mean"] - r["sage"]) < 0.05


def test_density_rule_without_features_the_sum_is_the_only_one_that_counts(rows):
    r = rows[("Dichte", C.FEATURE_MODES[1])]
    assert r["gin"] > 0.95 and r["sage"] < 0.65 and r["gin_mean"] < 0.65 and 0.6 < r["gcn"] < 0.8 and r["gin_vs_sage"] > 0.35 and abs(r["mlp"] - r["chance"]) < 0.06


def test_the_perceptron_is_enough_in_all_four_cases(rows):
    assert all(abs(r["perc_vs_gin"]) < 0.03 for r in rows.values())


def test_readme_levels_of_the_other_models_and_label_shares(rows):
    a, z, d, k = rows[("Anteil", C.FEATURE_MODES[0])], rows[("Anzahl", C.FEATURE_MODES[0])], rows[("Dichte", C.FEATURE_MODES[0])], rows[("Dichte", C.FEATURE_MODES[1])]
    assert 0.85 < a["gin_mean"] < 0.93 and 0.82 < z["gin_mean"] < 0.9 and 0.82 < d["gin_mean"] < 0.9
    assert 0.73 < a["gcn"] < 0.82 and 0.74 < z["gcn"] < 0.83 and 0.77 < d["gcn"] < 0.85
    assert 0.42 < z["positive"] < 0.5 and 0.44 < d["positive"] < 0.52 and abs(z["mlp"] - z["chance"]) < 0.06 and abs(d["mlp"] - d["chance"]) < 0.06 and abs(a["mlp"] - a["chance"]) < 0.03
    assert z["gin_vs_mean"] > 0.05


def test_default_area_degrees_match_the_readme():
    import numpy as np
    import gn_scenario as S
    means, maxima = zip(*[(S.generate(300, 5.5, 0.4, seed=s).degree.mean(), S.generate(300, 5.5, 0.4, seed=s).degree.max()) for s in range(12)])
    assert 6.5 < float(np.mean(means)) < 8.5 and max(maxima) <= 40
