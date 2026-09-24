"""GIN - Summe statt Mittel: Nachbarn zählen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Graph-Neural-Network-Linie der "Konzepte"-Reihe (Nachfolger von gcn-demo, Geschwister von graphsage-demo): GCN und GraphSAGE mitteln die Nachbarn und verlieren damit, wie viele es sind. Ein Graph Isomorphism Network
(Xu et al. 2019) summiert - und kann zählen. Die Demo misst, wann das hilft (Anzahl, Dichte), wann das Mittel besser ist (Anteil), was die Summe zum Trainieren braucht und wo auch GIN nicht mehr unterscheidet.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import gn_constants as C
from gn_evaluation import CASES, Settings, aggregation_experiment, analyse, expressiveness_table
from gn_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from gn_scenario import find_same_mean_pair
from gn_visualization import NAMES, build_aggregation, build_curves, build_map, build_perceptron, build_scale, case_label

st.set_page_config(page_title="GIN – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


def pts(x, se=None, digits=1):
    s = f"{'+' if x >= 0 else '−'}{de(abs(100 * x), digits)}"
    return s + (f" ± {de(100 * se, digits)}" if se is not None else "")


@st.cache_data(show_spinner=False)
def _aggregation(cases, seeds):
    return aggregation_experiment(cases=cases, seeds=seeds)


@st.cache_data(show_spinner=False)
def _expressiveness():
    return expressiveness_table()


st.title("🔢 GIN – Summe statt Mittel: Nachbarn zählen")
st.markdown(
    """
Ein **GCN** und **GraphSAGE** mitteln die Nachbarn eines Kunden - und verlieren damit, **wie viele** es sind: zwei Nachbarn vom Typ A und zehn Nachbarn vom Typ A haben denselben Mittelwert. Ein **Graph Isomorphism Network (GIN)**
(Xu/Hu/Leskovec/Jegelka 2019) **summiert** stattdessen und rechnet die Summe mit einem kleinen MLP um; die Summe behält die Zahl. Die Demo stellt drei Fragen an dieselbe Nachbarschaft - *Anteil* (mindestens die Hälfte vom Typ A),
*Anzahl* (mindestens drei vom Typ A) und *Dichte* (mindestens sieben Nachbarn) - und misst, welche Aggregation zu welcher Frage passt: bei der Anzahl und der Dichte gewinnt die Summe, beim **Anteil** gewinnt der Mittelwert.
"""
)
st.caption(
    "Fünftes Stück der **Graph-Neural-Network-Linie** der \"Konzepte\"-Reihe, Geschwister von **graphsage-demo** und Nachfolger von **gcn-demo**; das Netz ist von Grund auf in numpy geschrieben, alle Daten sind erzeugt. "
    "**Bezug zu OR:** Kundendichte und Zahl der Aufträge in der Nähe bestimmen Auslastung und Tourenplanung - Größen, die man zählt, nicht mittelt."
)

with st.expander("So funktioniert ein GIN", expanded=True):
    st.markdown(
        """
1. **Summe statt Mittel.** Eine Schicht bildet $Z = (1+\\epsilon)\\,H + \\tilde A\\,H$: den Kunden selbst plus die **Summe** der Nachbarn (hier $\\epsilon = 0$; $\\tilde A = A/c$ mit dem mittleren Grad $c$ - siehe Skalierung unten).
   Danach folgt ein **MLP** (zwei Matrizen mit ReLU dazwischen); die Ausgabe ist die neue Einbettung des Kunden.
2. **Warum die Summe zählt.** Bei lauter gleichen Nachbarn ist der Mittelwert immer derselbe (zwei Nachbarn oder zehn), die Summe wächst mit ihrer Zahl. Xu et al. zeigen: Summe plus MLP kann verschiedene Mengen von Nachbarn unterscheiden, die ein
   Mittel oder ein Maximum gleichsetzen; das GIN ist damit so ausdrucksstark wie der Weisfeiler-Lehman-Test (1-WL) auf Graphen.
3. **Vergleich.** Dasselbe Gerüst (zwei Nachrichtenschichten mit 16 Einheiten, eine lineare Ausgabeschicht, Verschiebungsterme) mit GCN-, GraphSAGE- oder GIN-Schicht; dazu **GIN mit Mittel** (dieselbe Schicht, nur der Mittelwert statt der Summe),
   damit man die Aggregation von den übrigen Bausteinen trennen kann, und das **MLP** ohne Nachbarn.
4. **Lernen.** Nur die bekannten Etiketten zählen, Adam, Gewichtszerfall, feste 200 Epochen, kein Dropout, nichts am Test abgestimmt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_nodes = st.slider("Kunden", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Zahl der Kunden im Gebiet; 60 % liegen in drei dichten Ballungen, der Rest gleichmäßig verteilt - der Grad schwankt also stark.")
    radius = st.slider("Suchradius", *bounds("radius_slider"), key="radius_slider", step=C.RADIUS_STEP, help="Zwei Kunden sind Nachbarn, wenn sie näher als der Suchradius beieinander liegen. Größerer Radius = mehr Nachbarn.")
    share = st.slider("Anteil Typ A", *bounds("share_slider"), key="share_slider", step=C.SHARE_A_STEP, help="Anteil der Kunden vom Typ A (der Rest ist Typ B); der Typ hängt nicht von der Lage ab.")
    features = st.selectbox("Merkmale", C.FEATURE_MODES, key="features_select", help="'Typ und Konstante': das Netz sieht den Typ jedes Kunden und eine 1. 'nur Konstante': nur die 1 - dann bleibt allein der Graph.")
    rule = st.selectbox("Frage an die Nachbarschaft", C.RULES, key="rule_select", help="Welche Regel das Etikett bestimmt (1 = Regel erfüllt): " + " | ".join(f"{k}: {v}" for k, v in C.RULE_TEXT.items()))
    if rule == "Anteil":
        thr = st.slider("Mindestanteil Typ A unter den Nachbarn", *bounds("frac_slider"), key="frac_slider", step=C.FRAC_STEP, help="Etikett 1, wenn mindestens dieser Anteil der Nachbarn vom Typ A ist.")
    elif rule == "Anzahl":
        thr = st.slider("Mindestzahl Nachbarn vom Typ A", *bounds("count_slider"), key="count_slider", help="Etikett 1, wenn mindestens so viele Nachbarn vom Typ A sind.")
    else:
        thr = st.slider("Mindestgrad (Zahl der Nachbarn)", *bounds("deg_slider"), key="deg_slider", help="Etikett 1, wenn der Kunde mindestens so viele Nachbarn hat (der Typ spielt keine Rolle).")
    known = st.slider("Anteil bekannter Kunden", *bounds("known_slider"), key="known_slider", step=C.KNOWN_STEP, help="Anteil der Kunden, deren Etikett dem Netz beim Training verraten wird; die übrigen dienen der Prüfung.")
    layers = st.slider("Nachrichtenschichten", *bounds("layers_slider"), key="layers_slider", help="Zahl der Nachrichtenschichten (für alle Modelle gleich).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Lage, Typen, bekannte Kunden und Anfangsgewichte fest.")
    st.button("🎲 Neues Gebiet generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n_nodes), "radius_slider": round(float(radius), 2), "share_slider": round(float(share), 2), "features_select": features, "rule_select": rule,
                   "frac_slider": round(float(st.session_state["frac_slider"]), 2), "count_slider": int(st.session_state["count_slider"]), "deg_slider": int(st.session_state["deg_slider"]),
                   "known_slider": round(float(known), 2), "layers_slider": int(layers), "seed_input": int(seed)})

settings = Settings(int(n_nodes), round(float(radius), 2), round(float(share), 2), features, rule, float(thr), round(float(known), 2), int(layers), int(seed))
with st.spinner("Trainiere die Modelle..."):
    a = analyse(settings)
g = a.graph
n = g.n
deg = g.degree

# --- Das Kundengebiet ---------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Das Kundengebiet und was die Netze lernen")
if features == C.FEATURE_MODES[1] and rule != "Dichte":
    st.info("Im Modus 'nur Konstante' sehen die Netze den Typ der Kunden nicht: die Regel kann höchstens über den Grad erraten werden, weil Kunden mit vielen Nachbarn auch mehr Nachbarn vom Typ A haben.")
c1, c2 = st.columns([1, 1])
with c1:
    epoch = st.slider("Trainingsepoche", 1, C.EPOCHS, C.EPOCHS, key="epoch_slider", help="Wie weit das GIN trainiert ist; die Karte zeigt seine Vorhersage nach dieser Epoche.")
with c2:
    mode_label = st.radio("Karte zeigt", ["Vorhersage des GIN", "Wahres Etikett", "Grad"], horizontal=True, key="map_mode")
mode = {"Vorhersage des GIN": "pred", "Wahres Etikett": "truth", "Grad": "degree"}[mode_label]
st.plotly_chart(build_map(a, epoch, mode), width="stretch", key="map_chart")
unknown = ~a.train_mask
pred_now = a.pred_history[epoch - 1]
st.caption(
    f"{n} Kunden, {g.n_edges()} Kanten, Grad im Mittel {de(deg.mean())} (höchstens {int(deg.max())}); Regel: {C.RULE_TEXT[rule]} - bei {pct(float(a.y.mean()))} der Kunden erfüllt. {int(a.train_mask.sum())} bekannte Etiketten (schwarze Ringe), "
    f"{int(unknown.sum())} unbekannte; nach Epoche {epoch} liegt das GIN bei {pct(float((pred_now[unknown] == a.y[unknown]).mean()))} richtig auf den unbekannten Kunden (Kreuz = falsch)."
)

st.markdown("---")

# --- Vergleich ----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 GIN gegen GCN, GraphSAGE und MLP")
cols = st.columns(6)
acc = {k: a.acc(k) for k in a.models}
cols[0].metric("GIN (Summe)", pct(acc["gin"], 1), help="Genauigkeit auf den unbekannten Kunden nach 200 Epochen.")
cols[1].metric("GraphSAGE", pct(acc["sage"], 1), delta=f"{pts(acc['gin'] - acc['sage'])} Punkte GIN gegen GraphSAGE", delta_color="off", help="Mittel über die Nachbarn, eigenes Gewicht für den Kunden.")
cols[2].metric("GCN", pct(acc["gcn"], 1), delta=f"{pts(acc['gin'] - acc['gcn'])} Punkte GIN gegen GCN", delta_color="off", help="Mittel mit symmetrischer Normierung.")
cols[3].metric("GIN mit Mittel", pct(acc["gin_mean"], 1), delta=f"{pts(acc['gin'] - acc['gin_mean'])} Punkte Summe gegen Mittel", delta_color="off", help="Dieselbe Schicht wie das GIN, nur mit Mittelwert statt Summe.")
cols[4].metric("MLP (ohne Nachbarn)", pct(acc["mlp"], 1), help="Dasselbe Gerüst ohne Kanten.")
cols[5].metric("Häufigste Klasse (Raten)", pct(a.majority_rate(), 1), help="Anteil der häufigeren Klasse unter den unbekannten Kunden.")
st.plotly_chart(build_curves(a, epoch), width="stretch", key="curves_chart")
best_mean = max(acc["sage"], acc["gcn"], acc["gin_mean"])
if acc["gin"] - best_mean > 0.03:
    st.success(f"✅ Die Summe zählt: GIN {pct(acc['gin'], 1)} gegen {pct(acc['sage'], 1)} (GraphSAGE), {pct(acc['gcn'], 1)} (GCN) und {pct(acc['gin_mean'], 1)} (GIN mit Mittel). Die Regel '{rule}' hängt an der Zahl der Nachbarn, die ein Mittel nicht kennt.")
elif acc["sage"] - acc["gin"] > 0.03:
    st.warning(f"⚠️ Hier gewinnt der Mittelwert: GraphSAGE {pct(acc['sage'], 1)} gegen GIN {pct(acc['gin'], 1)}. Die Regel '{rule}' fragt nach einem Anteil - ein Mittelwert ist genau diese Größe, die Summe muss sie erst aus Summe und Grad zusammensetzen.")
else:
    st.info(f"Kein klarer Unterschied: GIN {pct(acc['gin'], 1)}, GraphSAGE {pct(acc['sage'], 1)}, GCN {pct(acc['gcn'], 1)}, GIN mit Mittel {pct(acc['gin_mean'], 1)}.")

st.markdown("---")

# --- Zwei Kunden mit gleichem Anteil -----------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Zwei Kunden, für den Mittelwert gleich")
pair = find_same_mean_pair(g) if features == C.FEATURE_MODES[0] else None
if pair is None:
    st.caption("In diesem Gebiet gibt es kein Kundenpaar mit demselben Anteil von Typ A unter den Nachbarn und deutlich verschiedener Nachbarzahl (oder die Merkmale zeigen den Typ nicht).")
else:
    i, j = pair
    st.plotly_chart(build_map(a, C.EPOCHS, "truth", pair=pair), width="stretch", key="pair_map")
    rows = []
    c_scale = max(float(deg.mean()), 1.0)
    for label, k in (("Kunde 1", i), ("Kunde 2", j)):
        rows.append({"Kunde": f"{label} (Nr. {k})", "Nachbarn": int(deg[k]), "davon Typ A": int(g.count_a[k]), "Anteil Typ A (was ein Mittel sieht)": pct(g.count_a[k] / deg[k]),
                     "Summe ÷ mittlerer Grad (was GIN sieht)": de(g.count_a[k] / c_scale, 2), "Etikett": int(a.y[k]), "GIN sagt": int(a.pred["gin"][k]), "GraphSAGE sagt": int(a.pred["sage"][k])})
    st.dataframe(rows, hide_index=True)
    st.caption(
        f"Beide Kunden haben denselben Anteil von Nachbarn vom Typ A ({pct(g.count_a[i] / deg[i])}), aber {int(deg[i])} beziehungsweise {int(deg[j])} Nachbarn. Ein Mittelwert-Netz (GraphSAGE, GCN, GIN mit Mittel) sieht in der ersten Schicht für beide dieselbe Nachbarschaft; "
        f"die Summe unterscheidet sie ({int(g.count_a[i])} gegen {int(g.count_a[j])} Nachbarn vom Typ A). Ob das für die Regel '{rule}' nötig ist, zeigt die Spalte Etikett."
    )

st.markdown("---")

# --- Experiment -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Welche Aggregation für welche Frage? Skalierung, MLP")
st.caption(f"Standardgebiet ({C.DEFAULT_N} Kunden, Suchradius {de(C.DEFAULT_RADIUS)}, {pct(C.DEFAULT_SHARE_A)} Typ A, {pct(C.DEFAULT_KNOWN)} bekannte Kunden, zwei Schichten); vier Fälle, Mittel über {len(C.EXP_SEEDS)} feste Seeds "
           "(Fehlerbalken: Standardfehler; die schwarze Linie ist das Raten). Drei Auswertungen aus einem Lauf. Dauer etwa eine Minute.")
if st.button("Aggregation, Skalierung und MLP durchrechnen", key="agg_start"):
    st.session_state["agg_on"] = True
if st.session_state.get("agg_on"):
    rows_a = _aggregation(CASES, C.EXP_SEEDS)
    st.markdown("##### Summe gegen Mittel")
    st.plotly_chart(build_aggregation(rows_a), width="stretch", key="agg_chart")
    parts = "; ".join(f"{case_label(r)}: GIN {pct(r['gin'], 1)}, GraphSAGE {pct(r['sage'], 1)} ({pts(r['gin_vs_sage'], r['gin_vs_sage_se'])} Punkte)" for r in rows_a)
    r_share, r_count, r_dens, r_const = rows_a
    st.warning(
        f"**Befund:** {parts}. Beim Anteil gewinnt das Mittel (GraphSAGE in {r_share['n_seeds'] - r_share['wins_sage']} von {r_share['n_seeds']} Gebieten vorn); bei der Anzahl und der Dichte gewinnt die Summe (GIN in {r_count['wins_sage']} beziehungsweise {r_dens['wins_sage']} von {r_count['n_seeds']} Gebieten vorn). "
        f"Ohne Typ-Merkmal ist der Unterschied am größten: GIN {pct(r_const['gin'], 1)}, GCN {pct(r_const['gcn'], 1)}, GraphSAGE {pct(r_const['sage'], 1)}, GIN mit Mittel {pct(r_const['gin_mean'], 1)} - "
        f"ein Mittelwert über lauter gleiche Nachbarn ist immer 1, nur die Summe kennt den Grad (das GCN ahnt ihn über seine Normierung). Gegen dieselbe Schicht mit Mittelwert (GIN mit Mittel) liegt die Summe bei der Dichte {pts(r_dens['gin_vs_mean'], r_dens['gin_vs_mean_se'])} Punkte vorn."
    )
    st.markdown("##### Braucht die Summe eine Skalierung?")
    st.plotly_chart(build_scale(rows_a), width="stretch", key="scale_chart")
    parts = "; ".join(f"{case_label(r)}: {pts(r['raw_vs_gin'], r['raw_vs_gin_se'])} (Training {pct(r['gin_raw_train'])} gegen {pct(r['gin_train'])})" for r in rows_a if r["features"] == C.FEATURE_MODES[0])
    st.warning(
        f"**Befund:** GIN mit ungeteilter Summe minus GIN mit durch den mittleren Grad geteilter Summe in Punkten - {parts}. Die ungeteilte Summe wächst mit dem Grad; beim Anteil bleibt schon die Trainingsgenauigkeit zurück, und dort kostet sie am meisten. "
        "Das geteilte GIN ist dasselbe Modell bis auf eine feste Konstante $c$ vor der Summe (im Aufsatz ersetzt Batch-Normalisierung diese Konstante)."
    )
    st.markdown("##### Braucht das GIN sein MLP?")
    st.plotly_chart(build_perceptron(rows_a), width="stretch", key="perc_chart")
    parts = "; ".join(f"{case_label(r)}: {pts(r['perc_vs_gin'], r['perc_vs_gin_se'])}" for r in rows_a)
    st.warning(
        f"**Befund:** GIN mit nur einer Matrix statt des MLP minus GIN mit MLP in Punkten - {parts}. In diesen Aufgaben genügt ein Ein-Schicht-Perzeptron; die Theorie verlangt das MLP, um beliebige Mengen von Nachbarn zu trennen, "
        "die Zähl-Aufgaben hier brauchen es nicht."
    )

st.markdown("---")

# --- Wo GIN scheitert -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo auch das GIN nicht mehr unterscheidet")
st.caption("Ohne Training: Modelle mit Zufallsgewichten, alle Kunden mit demselben Merkmal (1). Ist die Einbettung gleich, kann kein Training den Unterschied lernen.")
table = _expressiveness()
mark = lambda same: "**gleich**" if same else "verschieden"
st.markdown(
    "| Modell | 6 Stopps als ein Kreis gegen zwei Dreiecke (Menge der Einbettungen) | Mitte eines Sterns mit 2 gegen 3 Blättern |\n|---|---|---|\n"
    + "\n".join(f"| {NAMES.get(r['kind'], r['kind'])} | {mark(r['cycle_vs_triangles'])} | {mark(r['star_2_vs_3'])} |" for r in table)
)
st.caption(
    "Jeder Stopp hat im Kreis wie in den zwei Dreiecken genau zwei Nachbarn - für ein Nachrichten-Netz sieht jeder Knoten überall gleich aus (1-WL). Ob eine Tour über sechs Stopps oder zwei Touren über je drei gemeint sind, entscheidet kein GCN, GraphSAGE oder GIN; "
    "dafür braucht es zusätzliche Information über die Lage im Graphen (nächstes Stück der Linie, Graph Transformer). Beim Stern trennt die Summe (GIN) die zwei von den drei Blättern, das Mittel nicht; das GCN trennt sie über seine Gradnormierung."
)

st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Frage hängt an einer Zahl (Anzahl, Grad)** | Hängt sie an einem Anteil, ist der Mittelwert die passendere Größe: GraphSAGE liegt beim Anteil deutlich vor dem GIN (Experiment oben). | – |
| **Die Summe lässt sich trainieren** | Ungeteilt wächst sie mit dem Grad; hier teilt eine feste Konstante (mittlerer Grad) sie, im Aufsatz die Batch-Normalisierung. | – |
| **Nachbarschaftsmengen unterscheiden genügt** | Zwei verschiedene Graphen, in denen jeder Knoten gleich aussieht (Kreis gegen zwei Dreiecke), bleiben auch für das GIN gleich (1-WL-Grenze). | Graph Transformer (nächstes Stück) |
| **Der ganze Graph passt in den Speicher** | Dichte Matrizen in dieser Demo; große Graphen brauchen Stichproben der Nachbarschaft. | GraphSAGE (Vorgänger) |
| **Erzeugte Daten, zwölf Gebiete je Messpunkt** | Ein Vehikel mit Regeln, die genau eine Aggregation belohnen; die Zahlen gelten für diese Größen, Standardfehler sind groß; kein Test auf echten Kundendaten. | – |
"""
)
st.caption("Die Linie: GCN → GraphSAGE, GAT → GATv2, GIN → Graph Transformer (Graph Transformer noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Graph.** Knoten $i = 1..n$ mit Merkmalen $x_i \in \mathbb R^F$ (Typ A/B als one-hot plus eine 1), Nachbarschaft $A_{ij} = 1$, wenn $\lVert p_i - p_j \rVert < r$. Etikett $y_i = 1$, wenn die Regel erfüllt ist:
Anteil $\frac{1}{d_i}\sum_{j \in \mathcal N(i)} a_j \ge q$, Anzahl $\sum_{j \in \mathcal N(i)} a_j \ge t$, Dichte $d_i \ge d$ mit $a_j = 1$ für Typ A.

**Schichten.** $H^{(0)} = X$, alle Modelle $H^{(l)} = \mathrm{ReLU}(\cdot)$, danach $Z = H^{(L)} W_o + b_o$.
- GCN: $\hat A H W + b$ mit $\hat A = D^{-1/2}(A + I)D^{-1/2}$.
- GraphSAGE: $H W_s + P H W_n + b$ mit $P = D^{-1}A$.
- GIN: $\mathrm{MLP}\big((1+\epsilon) H + \tfrac1c A H\big)$, $\mathrm{MLP}(U) = \mathrm{ReLU}(U W_1 + b_1) W_2 + b_2$, $\epsilon = 0$, $c$ = mittlerer Grad (Skalierung $c = 1$: "ungeteilte Summe"; Mittel statt Summe: "GIN mit Mittel"; nur $W_1$: "Ein-Schicht-Perzeptron").

**Verlust und Gradienten.** Kreuzentropie auf den bekannten Kunden plus Gewichtszerfall $5 \cdot 10^{-4}$ auf den Matrizen (nicht auf den Verschiebungen). Für die GIN-Schicht mit Ausgabegradient $\delta$: $\delta_V = \delta \odot \mathbb 1[V > 0]$,
$\delta_U = (\delta_V W_2^\top) \odot \mathbb 1[U > 0]$, $\partial W_2 = R^\top \delta_V$, $\partial W_1 = Z^\top \delta_U$ mit $Z = M H$, Eingabegradient $M^\top (\delta_U W_1^\top)$, $M = (1+\epsilon)I + \tfrac1c A$.

**Ausdrucksstärke.** Sind alle Merkmale gleich und hat jeder Knoten $k$ Nachbarn, ist jede Einbettung nach jeder Schicht dieselbe; ein Kreis mit sechs Knoten und zwei Dreiecke sind so nicht zu trennen (1-WL). Ein Mittelwert über $k$ gleiche Nachbarn hängt nicht von $k$ ab.

Implementiert in `gn_algorithm.py` (Schichten, Gradienten, Adam), `gn_scenario.py` (Vehikel, kleine Vergleichsgraphen), `gn_evaluation.py` (Analyse, Experiment, Ausdrucksstärke).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
