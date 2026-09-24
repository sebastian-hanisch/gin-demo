"""Kern: Matrizen von Hand, GIN-Schicht von Hand, Gradienten aller Modelle gegen zentrale Differenzen, Grenzfälle, Ausdrucksstärke (Kreis gegen zwei Dreiecke, Stern) gegen einen unabhängigen 1-WL-Test, Training."""

import numpy as np
import pytest

import gn_algorithm as A
import gn_evaluation as E
import gn_scenario as S


def path_graph():
    """Pfad 0 - 1 - 2."""
    Am = np.zeros((3, 3))
    Am[0, 1] = Am[1, 0] = Am[1, 2] = Am[2, 1] = 1.0
    return Am


# --- Matrizen und GIN-Schicht von Hand --------------------------------------------------------------------------------------------------


def test_matrices_by_hand_on_a_path():
    Am = path_graph()                                                  # Grade 1, 2, 1; mittlerer Grad 4/3
    assert A.mean_matrix(Am)[1] == pytest.approx([0.5, 0, 0.5]) and A.mean_matrix(Am)[0] == pytest.approx([0, 1, 0])
    assert A.gin_matrix(Am, "gin_raw")[1] == pytest.approx([1, 1, 1]) and A.gin_matrix(Am, "gin_raw")[0] == pytest.approx([1, 1, 0])
    assert A.gin_matrix(Am, "gin")[1] == pytest.approx([0.75, 1, 0.75])                    # (1 + eps) I + A / (4/3)
    assert A.gin_matrix(Am, "gin_mean")[1] == pytest.approx([0.5, 1, 0.5])
    B = A.normalized_adjacency(Am)
    assert B[1, 1] == pytest.approx(1 / 3) and B[0, 1] == pytest.approx(1 / np.sqrt(2 * 3)) and B[0, 0] == pytest.approx(1 / 2)


def test_gin_layer_with_identity_weights_computes_one_plus_degree_on_constant_features():
    """Merkmale 1, W1 = W2 = I, Verschiebungen 0: nach einer GIN-Schicht mit ungeteilter Summe steht 1 + Grad in jeder Einheit (Pfad: 2, 3, 2)."""
    X = np.ones((3, 2))
    p = {"W1": np.eye(2), "b1": np.zeros(2), "W2": np.eye(2), "b2": np.zeros(2)}
    H, _ = A.layer_forward("gin_raw", p, {"M": A.gin_matrix(path_graph(), "gin_raw")}, X)
    assert H[:, 0] == pytest.approx([2, 3, 2]) and H[:, 1] == pytest.approx([2, 3, 2])
    Hm, _ = A.layer_forward("gin_mean", p, {"M": A.gin_matrix(path_graph(), "gin_mean")}, X)
    assert Hm == pytest.approx(2 * np.ones((3, 2)))                                              # das Mittel kennt den Grad nicht: überall 1 + 1


def test_sage_layer_by_hand_and_mlp_limit():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    p = {"Ws": np.eye(2), "Wn": 2 * np.eye(2), "b": np.zeros(2)}
    H, _ = A.layer_forward("sage", p, {"P": A.mean_matrix(path_graph())}, X)
    assert H[1] == pytest.approx([3, 3]) and H[0] == pytest.approx([1, 2]) and H[2] == pytest.approx([2, 4])       # x_1 + 2 * mean(x_0, x_2)


# --- Gradienten ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", A.KINDS)
@pytest.mark.parametrize("layers", [1, 2])
def test_gradients_agree_with_central_differences(kind, layers):
    g = S.generate(40, 7.0, 0.4, seed=1)
    y = S.labels(g, "Anzahl", 3)
    tr = S.split(y, 0.5, 1)
    lay, out = A.init_params(kind, g.X.shape[1], layers, 6, 2, 1)
    mats = A.matrices(kind, g.A)
    _, gl, go = A.loss_and_grads(kind, lay, out, mats, g.X, y, tr, 5e-4)
    loss = lambda: A.loss_and_grads(kind, lay, out, mats, g.X, y, tr, 5e-4)[0]
    rng = np.random.default_rng(0)
    items = [(p, k, gl[i][k]) for i, p in enumerate(lay) for k in p] + [(out, k, go[k]) for k in out]
    for p, k, grad in items:
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in p[k].shape)
            old = p[k][idx]
            p[k][idx] = old + 1e-6
            up = loss()
            p[k][idx] = old - 1e-6
            dn = loss()
            p[k][idx] = old
            assert grad[idx] == pytest.approx((up - dn) / 2e-6, rel=1e-4, abs=1e-8)


def test_sage_without_neighbor_weights_is_exactly_the_mlp():
    """Wn = 0: Verlust und Gradient für Ws sind die des MLP (unabhängige Rechnung ohne Nachbarn)."""
    g = S.generate(40, 7.0, 0.4, seed=1)
    y = S.labels(g, "Anzahl", 3)
    tr = S.split(y, 0.5, 1)
    lay, out = A.init_params("sage", g.X.shape[1], 2, 6, 2, 1)
    mlp = [{"W": p["Ws"].copy(), "b": p["b"].copy()} for p in lay]
    for p in lay:
        p["Wn"][:] = 0.0
    ls, gs, gso = A.loss_and_grads("sage", lay, out, A.matrices("sage", g.A), g.X, y, tr, 5e-4)
    lm, gm, gmo = A.loss_and_grads("mlp", mlp, out, {}, g.X, y, tr, 5e-4)
    assert ls == pytest.approx(lm) and all(np.allclose(gs[i]["Ws"], gm[i]["W"]) and np.allclose(gs[i]["b"], gm[i]["b"]) for i in range(2)) and np.allclose(gso["W"], gmo["W"])


# --- Ausdrucksstärke ------------------------------------------------------------------------------------------------------------------------


def wl_colors(Am, X, rounds=4):
    """Unabhängiger 1-WL-Test (Farbverfeinerung): Farben als Tupel, Rückgabe die sortierte Liste der Farben nach den Runden."""
    n = len(Am)
    col = [tuple(X[i]) for i in range(n)]
    for _ in range(rounds):
        col = [(col[i], tuple(sorted(col[j] for j in np.flatnonzero(Am[i])))) for i in range(n)]
        col = [hash(c) for c in col]
    return sorted(col)


def wl_same(A1, X1, A2, X2):
    return len(A1) == len(A2) and wl_colors(A1, X1) == wl_colors(A2, X2)


def test_cycle_versus_two_triangles_is_indistinguishable_for_every_model_and_for_wl():
    one = np.ones((6, 1))
    assert wl_same(S.cycle_graph(6), one, S.two_cycles(3), one)
    for kind in A.KINDS:
        assert E.indistinguishable(kind, S.cycle_graph(6), one, S.two_cycles(3), one), kind


def test_star_center_is_separated_only_by_models_that_can_count():
    one3, one4 = np.ones((3, 1)), np.ones((4, 1))
    same = {k: E.center_embedding_equal(k, S.star_graph(2), one3, S.star_graph(3), one4) for k in A.KINDS}
    assert same["mlp"] and same["sage"] and same["gin_mean"]                          # ohne Zählen: dieselbe Einbettung
    assert not same["gin_raw"] and not same["gin"] and not same["gcn"] and not same["gin_perc"]


def test_gin_separates_graphs_that_wl_separates_and_not_more():
    """Pfad mit vier Knoten gegen Stern mit drei Blättern: WL trennt sie, GIN mit ungeteilter Summe auch; im Kreis sind alle Lagen einer Markierung gleich; mit einer Markierung sind Kreis und Dreiecke trennbar - für WL und GIN gleich."""
    path4 = np.zeros((4, 4))
    for i in range(3):
        path4[i, i + 1] = path4[i + 1, i] = 1.0
    star3 = S.star_graph(3)
    one = np.ones((4, 1))
    assert not wl_same(path4, one, star3, one) and not E.indistinguishable("gin_raw", path4, one, star3, one)
    X_a = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    X_b = np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]])
    C6, T2 = S.cycle_graph(6), S.two_cycles(3)
    assert wl_same(C6, X_a, C6, X_b) and E.indistinguishable("gin_raw", C6, X_a, C6, X_b)                    # ein markierter Knoten im Kreis: alle Lagen gleich
    assert not wl_same(C6, X_a, T2, X_a) and not E.indistinguishable("gin_raw", C6, X_a, T2, X_a)            # ein markierter Knoten trennt Kreis und Dreiecke: WL und GIN entscheiden gleich


# --- Training ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", A.KINDS)
def test_training_is_reproducible_and_reduces_the_loss(kind):
    g = S.generate(100, 6.0, 0.4, seed=3)
    y = S.labels(g, "Anzahl", 3)
    tr = S.split(y, 0.4, 3)
    a = A.train(kind, g.A, g.X, y, tr, epochs=60, seed=2)
    b = A.train(kind, g.A, g.X, y, tr, epochs=60, seed=2)
    assert a.history["loss"] == b.history["loss"] and a.history["loss"][-1] < 0.85 * a.history["loss"][0]
    assert float((A.predict(a, g.A, g.X)[~tr] == y[~tr]).mean()) == pytest.approx(a.history["test_acc"][-1])


def test_prediction_on_another_graph_uses_only_shared_weights_and_history_shapes():
    g1, g2 = S.generate(80, 6.0, 0.4, seed=1), S.generate(120, 6.0, 0.4, seed=2)
    y = S.labels(g1, "Dichte", 6)
    m = A.train("gin", g1.A, g1.X, y, S.split(y, 0.4, 1), epochs=20, seed=1, record_pred=True)
    assert A.predict(m, g2.A, g2.X).shape == (120,) and m.history["pred"].shape == (20, 80) and A.embeddings(m, g2.A, g2.X).shape == (120, 16)
