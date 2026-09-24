"""Plotly-Abbildungen der GIN-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gn_constants as C

REF_COLOR = "#7f7f7f"
GOOD = "#54a24b"
WARN = "#f58518"
BAD = "#e45756"
COLORS = {"mlp": "#7f7f7f", "gcn": "#b279a2", "sage": "#4c78a8", "gin": "#17becf", "gin_mean": "#9ecae1", "gin_perc": "#54a24b", "gin_raw": "#f58518"}
NAMES = {"mlp": "MLP (ohne Nachbarn)", "gcn": "GCN", "sage": "GraphSAGE", "gin": "GIN (Summe)", "gin_mean": "GIN mit Mittel", "gin_perc": "GIN mit Ein-Schicht-Perzeptron", "gin_raw": "GIN mit ungeteilter Summe"}
CLASS_COLORS = ("#4c78a8", "#e45756")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_map(a, epoch, mode="pred", pair=None):
    """Kunden im Gebiet. mode: 'pred' (Vorhersage des GIN, Kreuz = falsch, Farbe = Klasse), 'truth' (wahres Etikett) oder 'degree' (Grad als Farbe). Schwarze Ringe: bekanntes Etikett; optional zwei hervorgehobene Kunden."""
    g = a.graph
    fig = go.Figure()
    iu = np.array(np.nonzero(np.triu(g.A, 1)))
    xs, ys = [], []
    for i, j in zip(*iu):
        xs += [g.xy[i, 0], g.xy[j, 0], None]
        ys += [g.xy[i, 1], g.xy[j, 1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(150,150,150,0.25)", width=0.6), hoverinfo="skip", showlegend=False))
    if mode == "degree":
        fig.add_trace(go.Scatter(x=g.xy[:, 0], y=g.xy[:, 1], mode="markers", marker=dict(size=8, color=g.degree, colorscale="Viridis", showscale=True, colorbar=dict(title="Grad", thickness=12), line=dict(color="white", width=0.5)),
                                 name="Kunde", showlegend=False, hovertemplate="Grad %{marker.color:.0f}<extra></extra>"))
    else:
        pred = a.pred_history[epoch - 1] if mode == "pred" else a.y
        labels = ("Regel nicht erfüllt", "Regel erfüllt")
        for c in (0, 1):
            ok = (pred == c) & (pred == a.y)
            bad = (pred == c) & (pred != a.y)
            fig.add_trace(go.Scatter(x=g.xy[ok, 0], y=g.xy[ok, 1], mode="markers", marker=dict(size=8, color=CLASS_COLORS[c], line=dict(color="white", width=0.6)), name=labels[c], hovertemplate=f"{labels[c]}<extra></extra>"))
            if mode == "pred" and bad.any():
                fig.add_trace(go.Scatter(x=g.xy[bad, 0], y=g.xy[bad, 1], mode="markers", marker=dict(size=10, color=CLASS_COLORS[c], symbol="x", line=dict(color=CLASS_COLORS[c], width=2)), showlegend=False,
                                         hovertemplate=f"vorhergesagt: {labels[c]} (falsch)<extra></extra>"))
    tr = a.train_mask
    fig.add_trace(go.Scatter(x=g.xy[tr, 0], y=g.xy[tr, 1], mode="markers", marker=dict(size=14, symbol="circle-open", color="black", line=dict(width=1.5)), name="bekanntes Etikett"))
    if pair is not None:
        for idx, (node, label) in enumerate(zip(pair, ("Kunde 1", "Kunde 2"))):
            fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", marker=dict(size=18, symbol="star", color=WARN if idx == 0 else GOOD, line=dict(color="black", width=1)), name=label))
    fig.update_xaxes(range=[-2, C.AREA + 2], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y")
    fig.update_yaxes(range=[-2, C.AREA + 2], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, 470).update_layout(legend=dict(orientation="h", y=-0.05), margin=dict(l=10, r=10, t=10, b=10))


def build_curves(a, epoch):
    """Links: Genauigkeit auf den unbekannten Kunden aller Modelle über die Epochen; rechts: Trainingsverlust von GCN, GraphSAGE und GIN."""
    E = len(a.models["gin"].history["loss"])
    xs = list(range(1, E + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Genauigkeit auf unbekannten Kunden", "Trainingsverlust"), horizontal_spacing=0.12)
    for kind in ("gin", "sage", "gcn", "gin_mean", "mlp"):
        fig.add_trace(go.Scatter(x=xs, y=[100 * v for v in a.models[kind].history["test_acc"]], mode="lines", name=NAMES[kind], line=dict(color=COLORS[kind], width=2.5 if kind == "gin" else 1.8)), row=1, col=1)
    for kind in ("gin", "sage", "gcn"):
        fig.add_trace(go.Scatter(x=xs, y=a.models[kind].history["loss"], mode="lines", name=NAMES[kind], line=dict(color=COLORS[kind], width=2.5 if kind == "gin" else 1.8), showlegend=False), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=epoch, line=dict(color=WARN, dash="dash"), row=1, col=col)
    fig.update_xaxes(title_text="Epoche")
    fig.update_yaxes(title_text="Prozent", range=[0, 102], row=1, col=1)
    fig.update_yaxes(title_text="Kreuzentropie + Zerfall", row=1, col=2)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def case_label(r):
    extra = " (nur Konstante)" if r["features"] == C.FEATURE_MODES[1] else ""
    if r["rule"] == "Anteil":
        return f"Anteil ≥ {int(100 * r['threshold'])} %{extra}"
    if r["rule"] == "Anzahl":
        return f"Anzahl ≥ {int(r['threshold'])}{extra}"
    return f"Dichte ≥ {int(r['threshold'])}{extra}"


def _bars(rows, kinds, ymax=102, height=360):
    fig = go.Figure()
    labels = [case_label(r) for r in rows]
    for k in kinds:
        fig.add_trace(go.Bar(x=labels, y=[100 * r[k] for r in rows], error_y=dict(type="data", array=[100 * r[k + "_se"] for r in rows]), name=NAMES[k], marker=dict(color=COLORS[k])))
    fig.add_trace(go.Scatter(x=labels, y=[100 * r["chance"] for r in rows], mode="markers", name="Raten", marker=dict(symbol="line-ew", size=26, color="black", line=dict(width=2))))
    fig.update_yaxes(title_text="Genauigkeit auf unbekannten Kunden (%)", range=[30, ymax])
    return _base(fig, height).update_layout(barmode="group", legend=dict(orientation="h", y=-0.3))


def build_aggregation(rows):
    return _bars(rows, ("mlp", "gcn", "sage", "gin_mean", "gin"))


def build_scale(rows):
    """Testgenauigkeit (Balken) und Trainingsgenauigkeit (Punkte) von GIN mit geteilter und ungeteilter Summe, nur die Fälle mit Typ-Merkmal."""
    rows = [r for r in rows if r["features"] == C.FEATURE_MODES[0]]
    fig = go.Figure()
    labels = [case_label(r) for r in rows]
    for k in ("gin", "gin_raw"):
        fig.add_trace(go.Bar(x=labels, y=[100 * r[k] for r in rows], error_y=dict(type="data", array=[100 * r[k + "_se"] for r in rows]), name=NAMES[k] + " (Test)", marker=dict(color=COLORS[k])))
        fig.add_trace(go.Scatter(x=labels, y=[100 * r[k + "_train"] for r in rows], mode="markers", name=NAMES[k] + " (Training)", marker=dict(symbol="diamond", size=11, color="white", line=dict(color=COLORS[k], width=2))))
    fig.update_yaxes(title_text="Genauigkeit (%)", range=[50, 102])
    return _base(fig, 340).update_layout(barmode="group", legend=dict(orientation="h", y=-0.35))


def build_perceptron(rows):
    return _bars(rows, ("gin", "gin_perc"), height=320)
