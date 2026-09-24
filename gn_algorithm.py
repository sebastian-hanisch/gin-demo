"""Graph-Netze auf Knoten-Ebene, von Grund auf in numpy: MLP, GCN, GraphSAGE und GIN (Xu/Hu/Leskovec/Jegelka 2019) in einem gemeinsamen Rahmen.

Alle Modelle: L Nachrichtenschichten (Breite 16, ReLU) und eine lineare Ausgabeschicht auf den Knoteneinbettungen; jede Matrix hat einen Verschiebungsterm (Bias), Gewichtszerfall nur auf den Matrizen.
  mlp       H' = ReLU(H W + b)                                            keine Nachbarn
  gcn       H' = ReLU(A_hat H W + b)                                      A_hat = D^-1/2 (A + I) D^-1/2 (Mittel, mit Schleife)
  sage      H' = ReLU(H Ws + (P H) Wn + b)                                P = Zeilenmittel über die Nachbarn
  gin       H' = ReLU( MLP( (1 + eps) H + (A / c) H ) )                   Summe über die Nachbarn, durch die Konstante c (mittlerer Grad) geteilt; MLP = zwei Matrizen mit ReLU dazwischen, eps = 0
  gin_raw   wie gin, aber c = 1 (ungeteilte Summe)
  gin_mean  wie gin, aber Mittel statt Summe: (1 + eps) H + P H            (trennt die Aggregation von den übrigen Bausteinen)
  gin_perc  wie gin, aber statt des MLP nur EINE Matrix (Ein-Schicht-Perzeptron)"""

from dataclasses import dataclass, field

import numpy as np

import gn_constants as C

KINDS = ("mlp", "gcn", "sage", "gin", "gin_raw", "gin_mean", "gin_perc")
GIN_KINDS = ("gin", "gin_raw", "gin_mean", "gin_perc")


def normalized_adjacency(A):
    B = A + np.eye(len(A))
    inv = 1.0 / np.sqrt(B.sum(axis=1))
    return B * inv[:, None] * inv[None, :]


def mean_matrix(A):
    d = A.sum(axis=1, keepdims=True)
    return np.divide(A, d, out=np.zeros_like(A, dtype=float), where=d > 0)


def gin_matrix(A, kind, eps=0.0):
    """(1 + eps) I + Aggregation: Summe geteilt durch den mittleren Grad (gin, gin_perc), ungeteilte Summe (gin_raw) oder Mittel (gin_mean)."""
    n = len(A)
    if kind == "gin_mean":
        agg = mean_matrix(A)
    elif kind == "gin_raw":
        agg = A
    else:
        agg = A / max(float(A.sum(axis=1).mean()), 1.0)
    return (1.0 + eps) * np.eye(n) + agg


def matrices(kind, A):
    if kind == "gcn":
        return {"Ahat": normalized_adjacency(A)}
    if kind == "sage":
        return {"P": mean_matrix(A)}
    if kind in GIN_KINDS:
        return {"M": gin_matrix(A, kind)}
    return {}


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def _glorot(rng, a, b):
    lim = np.sqrt(6.0 / (a + b))
    return rng.uniform(-lim, lim, size=(a, b))


def init_params(kind, d, layers, hidden, n_classes, seed):
    """Liste der Schicht-Parameter (Wörterbuch je Schicht) und die Ausgabeschicht; Matrizen Glorot, Verschiebungen 0."""
    rng = np.random.default_rng([seed, 31])
    out = []
    din = d
    for _ in range(layers):
        if kind in ("mlp", "gcn"):
            p = {"W": _glorot(rng, din, hidden), "b": np.zeros(hidden)}
        elif kind == "sage":
            p = {"Ws": _glorot(rng, din, hidden), "Wn": _glorot(rng, din, hidden), "b": np.zeros(hidden)}
        elif kind == "gin_perc":
            p = {"W1": _glorot(rng, din, hidden), "b1": np.zeros(hidden)}
        else:
            p = {"W1": _glorot(rng, din, hidden), "b1": np.zeros(hidden), "W2": _glorot(rng, hidden, hidden), "b2": np.zeros(hidden)}
        out.append(p)
        din = hidden
    return out, {"W": _glorot(rng, hidden, n_classes), "b": np.zeros(n_classes)}


def layer_forward(kind, p, mats, H):
    if kind == "mlp":
        Z = H @ p["W"] + p["b"]
        return np.maximum(Z, 0), (H, Z)
    if kind == "gcn":
        AH = mats["Ahat"] @ H
        Z = AH @ p["W"] + p["b"]
        return np.maximum(Z, 0), (AH, Z)
    if kind == "sage":
        PH = mats["P"] @ H
        Z = H @ p["Ws"] + PH @ p["Wn"] + p["b"]
        return np.maximum(Z, 0), (H, PH, Z)
    MH = mats["M"] @ H
    U = MH @ p["W1"] + p["b1"]
    if kind == "gin_perc":
        return np.maximum(U, 0), (MH, U)
    R = np.maximum(U, 0)
    V = R @ p["W2"] + p["b2"]
    return np.maximum(V, 0), (MH, U, R, V)


def layer_backward(kind, p, mats, dout, cache):
    """Rückgabe (Gradient an der Eingabe, Gradienten der Parameter). Gewichtszerfall wird außerhalb addiert."""
    if kind == "mlp":
        H, Z = cache
        dZ = dout * (Z > 0)
        return dZ @ p["W"].T, {"W": H.T @ dZ, "b": dZ.sum(axis=0)}
    if kind == "gcn":
        AH, Z = cache
        dZ = dout * (Z > 0)
        return mats["Ahat"].T @ (dZ @ p["W"].T), {"W": AH.T @ dZ, "b": dZ.sum(axis=0)}
    if kind == "sage":
        H, PH, Z = cache
        dZ = dout * (Z > 0)
        return dZ @ p["Ws"].T + mats["P"].T @ (dZ @ p["Wn"].T), {"Ws": H.T @ dZ, "Wn": PH.T @ dZ, "b": dZ.sum(axis=0)}
    if kind == "gin_perc":
        MH, U = cache
        dU = dout * (U > 0)
        return mats["M"].T @ (dU @ p["W1"].T), {"W1": MH.T @ dU, "b1": dU.sum(axis=0)}
    MH, U, R, V = cache
    dV = dout * (V > 0)
    dR = dV @ p["W2"].T
    dU = dR * (U > 0)
    return mats["M"].T @ (dU @ p["W1"].T), {"W1": MH.T @ dU, "b1": dU.sum(axis=0), "W2": R.T @ dV, "b2": dV.sum(axis=0)}


def forward(kind, layers, out, mats, X):
    H = X
    caches = []
    for p in layers:
        H, c = layer_forward(kind, p, mats, H)
        caches.append(c)
    return H @ out["W"] + out["b"], H, caches


def loss_and_grads(kind, layers, out, mats, X, y, train, wd):
    logits, H, caches = forward(kind, layers, out, mats, X)
    P = softmax(logits)
    idx = np.flatnonzero(train)
    ce = -np.log(P[idx, y[idx]] + C.EPS).mean()
    dZ = np.zeros_like(logits)
    dZ[idx] = P[idx]
    dZ[idx, y[idx]] -= 1.0
    dZ /= len(idx)
    g_out = {"W": H.T @ dZ + wd * out["W"], "b": dZ.sum(axis=0)}
    reg = 0.5 * wd * float((out["W"] ** 2).sum())
    dH = dZ @ out["W"].T
    g_layers = [None] * len(layers)
    for l in range(len(layers) - 1, -1, -1):
        dH, g = layer_backward(kind, layers[l], mats, dH, caches[l])
        for k in g:
            if k.startswith("W"):
                g[k] = g[k] + wd * layers[l][k]
                reg += 0.5 * wd * float((layers[l][k] ** 2).sum())
        g_layers[l] = g
    return ce + reg, g_layers, g_out


@dataclass
class Model:
    kind: str
    layers: list
    out: dict
    history: dict = field(default_factory=dict)


def predict(model, A, X):
    return forward(model.kind, model.layers, model.out, matrices(model.kind, A), X)[0].argmax(axis=1)


def embeddings(model, A, X):
    """Knoteneinbettungen nach der letzten Nachrichtenschicht."""
    return forward(model.kind, model.layers, model.out, matrices(model.kind, A), X)[1]


def random_model(kind, d, layers, hidden, n_classes, seed):
    """Modell mit Zufallsgewichten (ohne Training) - für Ausdrucksstärke-Tests."""
    lay, out = init_params(kind, d, layers, hidden, n_classes, seed)
    return Model(kind, lay, out)


def train(kind, A, X, y, train_mask, layers=C.DEFAULT_LAYERS, hidden=C.HIDDEN, epochs=C.EPOCHS, lr=C.LEARNING_RATE, wd=C.WEIGHT_DECAY, seed=0, test_mask=None, record_pred=False):
    """Adam-Training. Verlauf: Verlust, Trainings- und Testgenauigkeit (Test = alle nicht bekannten Knoten, nur zur Anzeige)."""
    assert kind in KINDS
    n_classes = int(y.max()) + 1
    lay, out = init_params(kind, X.shape[1], layers, hidden, n_classes, seed)
    mats = matrices(kind, A)
    slots = [(p, k) for p in lay for k in p] + [(out, k) for k in out]
    mo = [np.zeros_like(p[k]) for p, k in slots]
    ve = [np.zeros_like(p[k]) for p, k in slots]
    test_mask = ~train_mask if test_mask is None else test_mask
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    preds = []
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        loss, g_layers, g_out = loss_and_grads(kind, lay, out, mats, X, y, train_mask, wd)
        flat = [g_layers[i][k] for i, p in enumerate(lay) for k in p] + [g_out[k] for k in out]
        for i, (p, k) in enumerate(slots):
            mo[i] = b1 * mo[i] + (1 - b1) * flat[i]
            ve[i] = b2 * ve[i] + (1 - b2) * flat[i] ** 2
            p[k] -= lr * (mo[i] / (1 - b1 ** t)) / (np.sqrt(ve[i] / (1 - b2 ** t)) + eps)
        pred = forward(kind, lay, out, mats, X)[0].argmax(axis=1)
        hist["loss"].append(float(loss))
        hist["train_acc"].append(float((pred[train_mask] == y[train_mask]).mean()))
        hist["test_acc"].append(float((pred[test_mask] == y[test_mask]).mean()))
        if record_pred:
            preds.append(pred)
    if record_pred:
        hist["pred"] = np.array(preds)
    return Model(kind, lay, out, hist)
