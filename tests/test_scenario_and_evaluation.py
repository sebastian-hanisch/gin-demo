"""Vehikel (Reproduzierbarkeit, Ungleichmäßigkeit, Regeln von Hand, Aufteilung, Paar mit gleichem Anteil) und Auswertung (Analyse, Experiment, Ausdrucksstärke)."""

import numpy as np
import pytest

import gn_constants as C
import gn_evaluation as E
import gn_scenario as S


def test_generate_is_reproducible_shaped_and_features_follow_the_mode():
    a, b = S.generate(120, 5.5, 0.4, seed=4), S.generate(120, 5.5, 0.4, seed=4)
    assert np.array_equal(a.A, b.A) and np.array_equal(a.X, b.X) and a.X.shape == (120, 3) and not np.array_equal(a.A, S.generate(120, 5.5, 0.4, seed=5).A)
    assert np.all(a.X[:, 2] == 1) and np.allclose(a.X[:, 0] + a.X[:, 1], 1) and np.array_equal(a.X[:, 0], a.is_a)
    assert S.generate(120, 5.5, 0.4, C.FEATURE_MODES[1], seed=4).X.shape == (120, 1)


def test_adjacency_is_symmetric_without_loops_and_follows_the_radius():
    g = S.generate(100, 6.0, 0.4, seed=2)
    assert np.array_equal(g.A, g.A.T) and np.all(np.diag(g.A) == 0)
    d = np.linalg.norm(g.xy[:, None] - g.xy[None], axis=2)
    assert np.array_equal(g.A > 0, (d < 6.0) & (d > 0)) and g.n_edges() == int(g.A.sum() / 2)


def test_density_is_uneven_and_grows_with_the_radius():
    g = S.generate(300, 5.5, 0.4, seed=1)
    assert g.degree.max() > 2.5 * g.degree.mean() and g.degree.min() <= 1
    assert S.generate(300, 9.0, 0.4, seed=1).degree.mean() > 1.8 * g.degree.mean()


def test_rules_by_hand_on_a_star_with_two_type_a_leaves():
    """Mitte 0 mit Blättern 1..4; Typ A: 1 und 2. Mitte: Grad 4, zwei Nachbarn vom Typ A -> Anteil 1/2, Anzahl 2. Blatt 1: Grad 1, ein Nachbar (die Mitte, Typ B)."""
    A_ = S.star_graph(4)
    is_a = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    g = S.Graph(np.zeros((5, 2)), A_, is_a, np.ones((5, 1)), 0)
    assert list(S.labels(g, "Anzahl", 2)) == [1, 0, 0, 0, 0] and list(S.labels(g, "Anzahl", 3)) == [0, 0, 0, 0, 0]
    assert list(S.labels(g, "Anteil", 0.5)) == [1, 0, 0, 0, 0] and list(S.labels(g, "Anteil", 0.6)) == [0, 0, 0, 0, 0]
    assert list(S.labels(g, "Dichte", 4)) == [1, 0, 0, 0, 0] and list(S.labels(g, "Dichte", 1)) == [1, 1, 1, 1, 1]
    with pytest.raises(ValueError):
        S.labels(g, "Unsinn", 1)


def test_share_rule_is_zero_for_isolated_customers():
    g = S.Graph(np.zeros((3, 2)), np.zeros((3, 3)), np.array([1.0, 1.0, 0.0]), np.ones((3, 1)), 0)
    assert list(S.labels(g, "Anteil", 0.2)) == [0, 0, 0]


def test_split_sizes_and_both_classes_on_both_sides():
    g = S.generate(200, 5.5, 0.4, seed=3)
    y = S.labels(g, "Anzahl", 3)
    tr = S.split(y, 0.3, 3)
    assert abs(tr.sum() - 60) <= 1 and set(y[tr]) == {0, 1} and set(y[~tr]) == {0, 1} and np.array_equal(tr, S.split(y, 0.3, 3)) and not np.array_equal(tr, S.split(y, 0.3, 4))


def test_find_same_mean_pair_has_equal_share_and_a_clear_degree_gap():
    g = S.generate(300, 5.5, 0.4, seed=4)
    i, j = S.find_same_mean_pair(g)
    assert g.count_a[i] / g.degree[i] == pytest.approx(g.count_a[j] / g.degree[j]) and g.degree[j] - g.degree[i] >= 3
    lonely = S.Graph(np.zeros((3, 2)), np.zeros((3, 3)), np.zeros(3), np.ones((3, 1)), 0)
    assert S.find_same_mean_pair(lonely) is None


def test_small_graphs():
    assert S.cycle_graph(6).sum() == 12 and np.all(S.cycle_graph(6).sum(axis=1) == 2)
    T = S.two_cycles(3)
    assert T.shape == (6, 6) and T[:3, 3:].sum() == 0 and np.all(T.sum(axis=1) == 2)
    st = S.star_graph(3)
    assert st[0].sum() == 3 and st[1:].sum() == 3


def test_analyse_is_cached_and_consistent():
    s = E.Settings(n=100, seed=2)
    a = E.analyse(s)
    assert a is E.analyse(s) and set(a.models) == set(E.MAIN_KINDS) and a.pred_history.shape == (C.EPOCHS, 100)
    assert a.acc("gin") == pytest.approx(a.models["gin"].history["test_acc"][-1]) and 0.5 <= a.majority_rate() <= 1


def test_aggregation_experiment_rows_are_consistent():
    rows = E.aggregation_experiment(cases=(("Anzahl", 3, C.FEATURE_MODES[0]), ("Dichte", 6, C.FEATURE_MODES[1])), seeds=(0, 1), base=E.Settings(n=120))
    assert [r["rule"] for r in rows] == ["Anzahl", "Dichte"] and rows[1]["features"] == C.FEATURE_MODES[1]
    for r in rows:
        assert r["gin_vs_sage"] == pytest.approx(r["gin"] - r["sage"]) and r["gin_vs_mean"] == pytest.approx(r["gin"] - r["gin_mean"]) and r["perc_vs_gin"] == pytest.approx(r["gin_perc"] - r["gin"])
        assert r["raw_vs_gin"] == pytest.approx(r["gin_raw"] - r["gin"]) and 0 <= r["wins_sage"] <= 2 and 0.5 <= r["chance"] <= 1 and all(0 <= r[k + "_train"] <= 1 for k in E.MAIN_KINDS + ("gin_perc", "gin_raw"))


def test_expressiveness_table_values():
    rows = {r["kind"]: r for r in E.expressiveness_table()}
    assert all(r["cycle_vs_triangles"] for r in rows.values())
    assert rows["mlp"]["star_2_vs_3"] and rows["sage"]["star_2_vs_3"] and rows["gin_mean"]["star_2_vs_3"] and not rows["gcn"]["star_2_vs_3"] and not rows["gin_raw"]["star_2_vs_3"]
