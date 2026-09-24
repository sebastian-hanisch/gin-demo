"""Auswertung: GIN gegen GCN, GraphSAGE und MLP auf einem Kundendichte-Graphen; ein Experiment mit drei Auswertungen (welche Aggregation für welche Frage, Skalierung der Summe, Perzeptron statt MLP) und ein Ausdrucksstärke-Vergleich ohne Training."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import gn_algorithm as A
import gn_constants as C
import gn_scenario as S

MAIN_KINDS = ("mlp", "gcn", "sage", "gin", "gin_mean")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    radius: float = C.DEFAULT_RADIUS
    share_a: float = C.DEFAULT_SHARE_A
    features: str = C.FEATURE_MODES[0]
    rule: str = C.DEFAULT_RULE
    threshold: float = C.DEFAULT_COUNT
    known: float = C.DEFAULT_KNOWN
    layers: int = C.DEFAULT_LAYERS
    seed: int = 3


@dataclass
class Analysis:
    settings: Settings
    graph: S.Graph
    y: np.ndarray
    train_mask: np.ndarray
    models: dict
    pred: dict
    pred_history: np.ndarray        # Vorhersage des GIN je Epoche

    def acc(self, kind):
        m = ~self.train_mask
        return float((self.pred[kind][m] == self.y[m]).mean())

    def majority_rate(self):
        m = ~self.train_mask
        return float(max(self.y[m].mean(), 1 - self.y[m].mean()))


def _generate(s):
    g = S.generate(s.n, s.radius, s.share_a, s.features, s.seed)
    y = S.labels(g, s.rule, s.threshold)
    return g, y, S.split(y, s.known, s.seed)


@lru_cache(maxsize=64)
def analyse(settings):
    g, y, tr = _generate(settings)
    models = {k: A.train(k, g.A, g.X, y, tr, layers=settings.layers, seed=settings.seed, record_pred=(k == "gin")) for k in MAIN_KINDS}
    pred = {k: A.predict(m, g.A, g.X) for k, m in models.items()}
    return Analysis(settings, g, y, tr, models, pred, models["gin"].history["pred"])


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


# --- Experiment 1: welche Aggregation für welche Frage ------------------------------------------------------------------------------------------

CASES = (("Anteil", 0.5, C.FEATURE_MODES[0]), ("Anzahl", 3, C.FEATURE_MODES[0]), ("Dichte", 7, C.FEATURE_MODES[0]), ("Dichte", 7, C.FEATURE_MODES[1]))


def aggregation_experiment(cases=None, seeds=None, base=None):
    """Je Fall (Regel, Schwelle, Merkmale) alle Modelle über feste Seeds; dazu GIN mit Ein-Schicht-Perzeptron statt MLP und GIN mit ungeteilter Summe (mit Trainingsgenauigkeit)."""
    cases = CASES if cases is None else cases
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    kinds = MAIN_KINDS + ("gin_perc", "gin_raw")
    rows = []
    for rule, thr, feat in cases:
        res = {k: [] for k in kinds}
        train = {k: [] for k in kinds}
        pos, rate = [], []
        for s in seeds:
            st = Settings(base.n, base.radius, base.share_a, feat, rule, thr, base.known, base.layers, s)
            g, y, tr = _generate(st)
            pos.append(float(y.mean()))
            te = ~tr
            rate.append(float(max(y[te].mean(), 1 - y[te].mean())))
            for k in kinds:
                m = A.train(k, g.A, g.X, y, tr, layers=st.layers, seed=s)
                res[k].append(m.history["test_acc"][-1])
                train[k].append(m.history["train_acc"][-1])
        row = {"rule": rule, "threshold": thr, "features": feat, "n_seeds": len(seeds), "positive": float(np.mean(pos)), "chance": float(np.mean(rate))}
        for k, v in res.items():
            row[k], row[k + "_se"] = _mean_se(v)
            row[k + "_train"] = float(np.mean(train[k]))
        row["gin_vs_sage"], row["gin_vs_sage_se"] = _mean_se(np.array(res["gin"]) - np.array(res["sage"]))
        row["gin_vs_mean"], row["gin_vs_mean_se"] = _mean_se(np.array(res["gin"]) - np.array(res["gin_mean"]))
        row["perc_vs_gin"], row["perc_vs_gin_se"] = _mean_se(np.array(res["gin_perc"]) - np.array(res["gin"]))
        row["raw_vs_gin"], row["raw_vs_gin_se"] = _mean_se(np.array(res["gin_raw"]) - np.array(res["gin"]))
        row["wins_sage"] = int(np.sum(np.array(res["gin"]) > np.array(res["sage"])))
        rows.append(row)
    return rows


# --- Ausdrucksstärke ohne Training ---------------------------------------------------------------------------------------------------------------


def graph_embedding(kind, A_, X_, seed=0, layers=2, hidden=16):
    """Knoteneinbettungen eines Modells mit Zufallsgewichten, zeilenweise sortiert (Fingerabdruck der Menge der Knoten)."""
    m = A.random_model(kind, X_.shape[1], layers, hidden, 2, seed)
    H = A.embeddings(m, A_, X_)
    return np.sort(np.round(H, 9), axis=0)


def indistinguishable(kind, A1, X1, A2, X2, seeds=range(3)):
    """True, wenn das Modell mit Zufallsgewichten für beide Graphen dieselbe Menge von Knoteneinbettungen liefert (für mehrere Zufallsgewichte)."""
    for s in seeds:
        h1, h2 = graph_embedding(kind, A1, X1, s), graph_embedding(kind, A2, X2, s)
        if h1.shape != h2.shape or not np.allclose(h1, h2, atol=1e-8):
            return False
    return True


def center_embedding_equal(kind, A1, X1, A2, X2, seeds=range(3)):
    """True, wenn der Knoten 0 (die Mitte) in beiden Graphen dieselbe Einbettung erhält (für mehrere Zufallsgewichte)."""
    for s in seeds:
        m = A.random_model(kind, X1.shape[1], 2, 16, 2, s)
        if not np.allclose(A.embeddings(m, A1, X1)[0], A.embeddings(m, A2, X2)[0], atol=1e-8):
            return False
    return True


def expressiveness_table(kinds=("mlp", "gcn", "sage", "gin_mean", "gin_raw")):
    """Zwei Fälle mit lauter gleichen Merkmalen: (a) sechs Stopps als ein Kreis gegen zwei Dreiecke (jeder Knoten hat zwei Nachbarn): dieselbe Menge von Einbettungen? (b) die Mitte eines Sterns mit zwei gegen die mit drei
    gleichen Blättern: dieselbe Einbettung?"""
    one = lambda n: np.ones((n, 1))
    rows = []
    for k in kinds:
        rows.append({"kind": k, "cycle_vs_triangles": indistinguishable(k, S.cycle_graph(6), one(6), S.two_cycles(3), one(6)),
                     "star_2_vs_3": center_embedding_equal(k, S.star_graph(2), one(3), S.star_graph(3), one(4))})
    return rows
