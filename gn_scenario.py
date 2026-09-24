"""Vehikel H "Kundendichte": n Kunden in einem 100 x 100 km großen Gebiet, ein Teil in dichten Ballungen, der Rest gleichmäßig verteilt - der Grad (Zahl der Kunden im Suchradius) schwankt also stark. Jeder Kunde hat
einen Typ (A oder B, unabhängig von der Lage) und eine Konstante 1 als Merkmale (Modus "nur Konstante": nur die 1). Gesucht wird ein Etikett, das allein aus der Nachbarschaft folgt:

  Anteil  - mindestens der Anteil q der Nachbarn ist vom Typ A          (ein Mittelwert genügt)
  Anzahl  - mindestens t Nachbarn sind vom Typ A                        (gezählt werden muss)
  Dichte  - mindestens d Nachbarn überhaupt                              (Grad; der Typ ist egal)"""

from dataclasses import dataclass

import numpy as np

import gn_constants as C


@dataclass(frozen=True)
class Graph:
    xy: np.ndarray            # (n, 2)
    A: np.ndarray             # (n, n) symmetrische 0/1-Nachbarschaft ohne Schleifen
    is_a: np.ndarray          # (n,) 1.0, wenn Typ A
    X: np.ndarray             # (n, F) Merkmale
    seed: int

    @property
    def n(self):
        return len(self.xy)

    @property
    def degree(self):
        return self.A.sum(axis=1)

    @property
    def count_a(self):
        """Zahl der Nachbarn vom Typ A."""
        return self.A @ self.is_a

    def n_edges(self):
        return int(np.triu(self.A, 1).sum())


def generate(n=C.DEFAULT_N, radius=C.DEFAULT_RADIUS, share_a=C.DEFAULT_SHARE_A, features=C.FEATURE_MODES[0], seed=0):
    rng = np.random.default_rng(seed)
    n_cl = int(round(C.CLUSTER_SHARE * n))
    centers = rng.random((C.CLUSTERS, 2)) * C.AREA
    xy = np.vstack([centers[rng.integers(0, C.CLUSTERS, n_cl)] + rng.normal(size=(n_cl, 2)) * C.CLUSTER_SIGMA, rng.random((n - n_cl, 2)) * C.AREA])
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    A = ((d < radius) & (d > 0)).astype(float)
    is_a = (rng.random(n) < share_a).astype(float)
    if features == C.FEATURE_MODES[0]:
        X = np.stack([is_a, 1.0 - is_a, np.ones(n)], axis=1)
    else:
        X = np.ones((n, 1))
    return Graph(xy, A, is_a, X, int(seed))


def labels(g, rule, threshold):
    """Etikett 1, wenn die Regel erfüllt ist. Schwelle: Anteil q (0..1) bei "Anteil", Zahl t bei "Anzahl", Grad d bei "Dichte"."""
    deg, cnt = g.degree, g.count_a
    if rule == "Anteil":
        return (cnt >= threshold * np.maximum(deg, 1) - 1e-12).astype(int) * (deg > 0)
    if rule == "Anzahl":
        return (cnt >= threshold).astype(int)
    if rule == "Dichte":
        return (deg >= threshold).astype(int)
    raise ValueError(rule)


def split(y, known, seed):
    """Zufällig `known` (Anteil) der Kunden bekannt (mindestens ein Kunde je Klasse und mindestens einer je Klasse zur Prüfung, wenn möglich); alle übrigen dienen der Prüfung."""
    rng = np.random.default_rng([seed, 7])
    n = len(y)
    m = max(2, int(round(known * n)))
    train = np.zeros(n, dtype=bool)
    train[rng.permutation(n)[:m]] = True
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if not train[idx].any() and len(idx) > 1:
            train[rng.choice(idx)] = True
        if train[idx].all() and len(idx) > 1:
            train[rng.choice(idx)] = False
    return train


def find_same_mean_pair(g, min_gap=3):
    """Zwei Kunden mit demselben Anteil von Nachbarn vom Typ A, aber deutlich verschiedener Zahl von Nachbarn (Unterschied im Grad mindestens `min_gap`): für einen Mittelwert sehen ihre Nachbarschaften gleich aus,
    für eine Summe nicht. Rückgabe (i, j) mit dem größten Gradunterschied (Grad_i < Grad_j) oder None."""
    deg, cnt = g.degree, g.count_a
    frac = np.where(deg > 0, cnt / np.maximum(deg, 1), -1.0)
    best = None
    for i in range(g.n):
        if deg[i] < 2:
            continue
        for j in np.flatnonzero((np.abs(frac - frac[i]) < 1e-12) & (deg - deg[i] >= min_gap)):
            gap = deg[j] - deg[i]
            if best is None or gap > best[0]:
                best = (gap, i, int(j))
    return None if best is None else (best[1], best[2])


# --- kleine Vergleichsgraphen (ohne Zufall) ------------------------------------------------------------------------------------------------


def cycle_graph(k):
    A = np.zeros((k, k))
    for i in range(k):
        A[i, (i + 1) % k] = A[(i + 1) % k, i] = 1.0
    return A


def two_cycles(k):
    """Zwei getrennte Kreise mit je k Knoten."""
    A = np.zeros((2 * k, 2 * k))
    A[:k, :k] = cycle_graph(k)
    A[k:, k:] = cycle_graph(k)
    return A


def star_graph(leaves):
    """Knoten 0 in der Mitte, `leaves` Blätter."""
    A = np.zeros((leaves + 1, leaves + 1))
    A[0, 1:] = A[1:, 0] = 1.0
    return A
