"""Unabhängige Orakel für die GIN-Demo: Schleifen-Referenz aller sieben Modellarten (mlp, gcn, sage, gin, gin_raw, gin_mean, gin_perc) samt Äquivarianz und zentralen Differenzen; Etiketten mit exakten Brüchen;
Suchradius-Nachbarschaft gegen scipy; Gleichanteilpaar per Brute Force über Brüche. Ergänzt test_algorithm.py (Handrechnung, WL-Vergleich) um Zufallsinstanzen."""

from fractions import Fraction

import numpy as np
import pytest

import gn_algorithm as A
import gn_scenario as S


def _rand_graph(rng, n, p):
    M = np.triu((rng.random((n, n)) < p).astype(float), 1)
    return M + M.T


def _loop_forward(kind, layers, out, Ad, X):
    n = len(Ad)
    deg = [int(Ad[i].sum()) for i in range(n)]
    mean_deg = max(sum(deg) / n, 1.0)
    H = X.copy()
    for p in layers:
        width = (p["b"] if "b" in p else p["b1"] if kind == "gin_perc" else p["b2"]).shape[0]
        new = np.zeros((n, width))
        for i in range(n):
            nb = np.flatnonzero(Ad[i])
            if kind == "mlp":
                new[i] = np.maximum(H[i] @ p["W"] + p["b"], 0)
            elif kind == "gcn":
                agg = H[i] / (deg[i] + 1)
                for j in nb:
                    agg = agg + H[j] / np.sqrt((deg[i] + 1) * (deg[j] + 1))
                new[i] = np.maximum(agg @ p["W"] + p["b"], 0)
            elif kind == "sage":
                m = np.mean([H[j] for j in nb], axis=0) if len(nb) else np.zeros(H.shape[1])
                new[i] = np.maximum(H[i] @ p["Ws"] + m @ p["Wn"] + p["b"], 0)
            else:
                s = sum((H[j] for j in nb), np.zeros(H.shape[1]))
                s = (s / len(nb) if len(nb) else s) if kind == "gin_mean" else s if kind == "gin_raw" else s / mean_deg
                z = H[i] + s
                if kind == "gin_perc":
                    new[i] = np.maximum(z @ p["W1"] + p["b1"], 0)
                else:
                    new[i] = np.maximum(np.maximum(z @ p["W1"] + p["b1"], 0) @ p["W2"] + p["b2"], 0)
        H = new
    return H @ out["W"] + out["b"]


@pytest.mark.parametrize("kind", A.KINDS)
def test_forward_loss_and_gradients_match_independent_references(kind):
    rng = np.random.default_rng(4)
    for t in range(12):
        n = int(rng.integers(1, 9))
        Ad = _rand_graph(rng, n, rng.random())
        d, L, K = int(rng.integers(1, 4)), int(rng.integers(1, 4)), int(rng.integers(2, 4))
        layers, out = A.init_params(kind, d, L, int(rng.integers(2, 6)), K, t)
        for p in layers:
            for k in p:
                if k.startswith("b"):
                    p[k] = rng.normal(size=p[k].shape) * 0.3
        out["b"] = rng.normal(size=K) * 0.3
        X = rng.normal(size=(n, d))
        mats = A.matrices(kind, Ad)
        lg = A.forward(kind, layers, out, mats, X)[0]
        assert np.allclose(lg, _loop_forward(kind, layers, out, Ad, X))
        perm = rng.permutation(n)
        assert np.allclose(A.forward(kind, layers, out, A.matrices(kind, Ad[np.ix_(perm, perm)]), X[perm])[0], lg[perm])
        y = rng.integers(0, K, size=n)
        tr = rng.random(n) < 0.6
        tr[0] = True
        wd = float(rng.choice([0.0, 5e-4, 0.05]))
        loss, gl, go = A.loss_and_grads(kind, layers, out, mats, X, y, tr, wd)
        ce = np.mean([np.log(np.exp(lg[i] - lg[i].max()).sum()) + lg[i].max() - lg[i][y[i]] for i in np.flatnonzero(tr)])
        reg = 0.5 * wd * (sum((p[k] ** 2).sum() for p in layers for k in p if k.startswith("W")) + (out["W"] ** 2).sum())
        assert loss == pytest.approx(ce + reg, abs=1e-7)
        slots = [(p, k) for p in layers for k in p] + [(out, k) for k in out]
        an = [gl[i][k] for i, p in enumerate(layers) for k in p] + [go[k] for k in out]
        for si in rng.permutation(len(slots))[:6]:
            p, k = slots[si]
            idx = tuple(rng.integers(0, s) for s in p[k].shape)
            old = p[k][idx]
            p[k][idx] = old + 1e-6
            a = A.loss_and_grads(kind, layers, out, mats, X, y, tr, wd)[0]
            p[k][idx] = old - 1e-6
            b = A.loss_and_grads(kind, layers, out, mats, X, y, tr, wd)[0]
            p[k][idx] = old
            assert an[si][idx] == pytest.approx((a - b) / 2e-6, rel=1e-4, abs=1e-5), k


def test_radius_graph_and_labels_match_scipy_and_exact_fractions():
    spatial = pytest.importorskip("scipy.spatial")
    rng = np.random.default_rng(5)
    for t in range(25):
        n = int(rng.integers(5, 100))
        radius = float(rng.choice([4.0, 5.5, 9.0]))
        g = S.generate(n, radius, float(rng.choice([0.2, 0.4, 0.6])), "Typ und Konstante", t)
        R = np.zeros((n, n))
        for i, j in spatial.cKDTree(g.xy).query_pairs(radius * (1 - 1e-12)):
            R[i, j] = R[j, i] = 1.0
        assert np.array_equal(g.A, R)
        deg = R.sum(axis=1).astype(int)
        cnt = (R @ g.is_a).astype(int)
        for q in (Fraction(1, 5), Fraction(1, 2), Fraction(3, 5), Fraction(1, 3)):
            ref = [1 if deg[i] > 0 and Fraction(int(cnt[i]), int(deg[i])) >= q else 0 for i in range(n)]
            assert np.array_equal(S.labels(g, "Anteil", float(q)), ref)
        assert np.array_equal(S.labels(g, "Anzahl", 3), (cnt >= 3).astype(int))
        assert np.array_equal(S.labels(g, "Dichte", 7), (deg >= 7).astype(int))
        # Gleichanteilpaar: größter Gradunterschied bei exakt gleichem Anteil, Grad des ersten mindestens 2
        best = None
        for i in range(n):
            if deg[i] < 2:
                continue
            for j in range(n):
                if deg[j] > 0 and Fraction(int(cnt[j]), int(deg[j])) == Fraction(int(cnt[i]), int(deg[i])) and deg[j] - deg[i] >= 3:
                    best = max(best or 0, int(deg[j] - deg[i]))
        pair = S.find_same_mean_pair(g)
        if best is None:
            assert pair is None
        else:
            assert pair is not None and deg[pair[1]] - deg[pair[0]] == best
            assert Fraction(int(cnt[pair[0]]), int(deg[pair[0]])) == Fraction(int(cnt[pair[1]]), int(deg[pair[1]]))
