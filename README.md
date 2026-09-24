# 🔢 GIN – Summe statt Mittel: Nachbarn zählen

Fünftes Stück der **Graph-Neural-Network-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning – Geschwister von
[graphsage-demo](https://sebastianhanisch-graphsage-demo.streamlit.app/) und Nachfolger von [gcn-demo](https://sebastianhanisch-gcn-demo.streamlit.app/) (GCN → GraphSAGE, GAT → GATv2, GIN → Graph Transformer; der Graph Transformer ist noch nicht gebaut).

Ein **GCN** und **GraphSAGE** mitteln die Nachbarn eines Kunden – und verlieren damit, **wie viele** es sind: zwei Nachbarn vom Typ A und zehn Nachbarn vom Typ A haben denselben Mittelwert. Ein **Graph Isomorphism Network (GIN)**
(Xu/Hu/Leskovec/Jegelka 2019) **summiert** stattdessen, $H' = \mathrm{MLP}\big((1+\epsilon)H + AH\big)$, und rechnet die Summe mit einem kleinen MLP um. Die Demo stellt drei Fragen an dieselbe Nachbarschaft – **Anteil** (mindestens die Hälfte der Nachbarn vom Typ A),
**Anzahl** (mindestens drei) und **Dichte** (mindestens sieben Nachbarn, der Typ ist egal) – und misst, welche Aggregation zu welcher Frage passt. Alle Daten sind erzeugt, das Netz ist von Grund auf in numpy geschrieben (Vorwärts- und Rückwärtsrechnung von Hand).

**Bezug zu OR:** Kundendichte und die Zahl der Aufträge in der Nähe bestimmen Auslastung und Tourenplanung – Größen, die man zählt, nicht mittelt.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie sah für GIN vor, dass es die Ausdrucksschwäche des Mittelwerts behebt. Das **bestätigt sich – aber nur, wenn die Frage zählt**, und die Vorab-Messung hat drei Korrekturen geliefert:

1. **Der Mittelwert gewinnt, wenn die Frage ein Anteil ist.** Beim Anteil liegt GraphSAGE 6,8 Punkte *vor* dem GIN (GIN in keinem von 12 Gebieten vorn): ein Anteil *ist* ein Mittelwert. Die Summe gewinnt bei der Anzahl (+6,9) und der Dichte (+10,3), und ohne Typ-Merkmal ist der Abstand
   riesig (+43,1 Punkte): ein Mittel über lauter gleiche Nachbarn ist immer 1.
2. **Die ungeteilte Summe lässt sich schlecht trainieren.** Die erste Vorab-Messung (anderes Gerüst ohne Verschiebungsterme, ungeteilte Summe, nicht als Test abgesichert) lag *unter* GraphSAGE (GIN 74–79 % gegen 87–90 %); die Trainingsgenauigkeit blieb bei 79 %. Im endgültigen Gerüst zeigt die Ablation denselben Effekt (siehe Befunde). Die Summe wächst mit dem Grad (bis 25 Nachbarn). Im Aufsatz sorgt Batch-Normalisierung dafür; hier teilt eine feste
   Konstante (der mittlere Grad) die Summe – dasselbe Modell bis auf diese Konstante. Ergebnis: beim Anteil kostet die ungeteilte Summe 14,6 Punkte, bei Anzahl und Dichte 2,9 und 1,6.
3. **Das MLP im GIN ist in diesen Aufgaben nicht nötig.** Die Theorie verlangt es, um beliebige Nachbarschaftsmengen zu trennen; ein Ein-Schicht-Perzeptron liegt hier gleichauf (Unterschied höchstens 0,7 Punkte, innerhalb des Standardfehlers).

Außerdem wurde die Rahmenarchitektur geändert: ohne **Verschiebungsterm** kann ein Netz, dessen einziges Merkmal die 1 ist, keine Schwelle setzen (die Funktion bleibt homogen). Alle Modelle haben deshalb hier Verschiebungsterme und eine lineare Ausgabeschicht; GraphSAGE ist
in diesem Rahmen neu gerechnet (ohne Stichprobe, nicht identisch mit graphsage-demo).

## Modell

- **Vehikel H "Kundendichte"** (`gn_scenario.py`): 300 Kunden im 100 × 100 km großen Gebiet, 60 % in drei dichten Ballungen (σ = 7 km), der Rest gleichmäßig; Nachbarn = Kunden im **Suchradius** (5,5 km; mittlerer Grad etwa 7, höchstens etwa 25); Typ A oder B (40 % A, unabhängig von der Lage);
  Merkmale: Typ (one-hot) und eine 1, oder nur die 1. Etikett je Kunde: Anteil ≥ 50 %, Anzahl ≥ 3 oder Grad ≥ 7 (Etikett 1 bei 31 %, 46 %, 48 % der Kunden); 30 % der Kunden sind bekannt.
- **Modelle** (`gn_algorithm.py`): zwei Nachrichtenschichten (16 Einheiten, ReLU) und eine lineare Ausgabeschicht, Kreuzentropie auf den bekannten Kunden, Gewichtszerfall $5\cdot10^{-4}$ auf den Matrizen, Adam (0,01), 200 Epochen, **kein Dropout, nichts am Test abgestimmt**.
  **GCN** ($\hat A H W$), **GraphSAGE** ($H W_s + P H W_n$), **GIN** ($\mathrm{MLP}((1+\epsilon)H + \tfrac1c AH)$, $\epsilon = 0$, $c$ = mittlerer Grad), **MLP** ohne Nachbarn.
- **Ablationen:** **GIN mit Mittel** (dieselbe Schicht, nur der Mittelwert statt der Summe – trennt die Aggregation von den übrigen Bausteinen), **GIN mit ungeteilter Summe** ($c = 1$), **GIN mit Ein-Schicht-Perzeptron** (nur eine Matrix statt des MLP).

## Methodik

- **Handrechnungen:** Matrizen auf einem Pfad (Grade 1, 2, 1; mittlerer Grad 4/3); eine GIN-Schicht mit Einheitsmatrizen und Merkmalen 1 liefert $1 + $ Grad (Pfad: 2, 3, 2), mit Mittel überall 2; SAGE-Schicht; Regeln auf einem Stern mit zwei Blättern vom Typ A.
- **Gegenproben:** **Gradienten aller Parameter aller sieben Modellvarianten gegen zentrale Differenzen** (1 und 2 Schichten), Grenzfall "GraphSAGE mit $W_n = 0$ ist exakt das MLP" (unabhängige Rechnung); **Ausdrucksstärke gegen einen unabhängigen 1-WL-Test** (Farbverfeinerung):
  Kreis mit sechs Knoten gegen zwei Dreiecke sind für **alle** Modelle und für 1-WL gleich, Pfad gegen Stern für 1-WL und GIN verschieden, ein markierter Knoten trennt Kreis und Dreiecke für beide gleich.
- **Statistik:** Experiment über 12 feste Seeds, Fehlerbalken = Standardfehler, Differenzen **gepaart je Seed**.
- **Literatur** (nicht nachgebaut): Xu, Hu, Leskovec, Jegelka 2019 ("How powerful are graph neural networks?", ICLR); Weisfeiler/Lehman 1968 (Farbverfeinerung).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Anteil** (12 Seeds) | GraphSAGE 99,2 %, GIN 92,3 % (**−6,8 ± 0,8**, GraphSAGE in 12 von 12 Gebieten vorn), GIN mit Mittel 89,6 %, GCN 77,5 %, MLP 68,8 % (= Raten). Das Mittel ist die richtige Größe. | `test_share_rule_the_mean_wins_and_the_unscaled_sum_falls_far_behind` |
| **Anzahl** | GIN 94,7 %, GraphSAGE 87,7 % (**+6,9 ± 1,2**, GIN in 12 von 12 vorn), GIN mit Mittel 86,7 % (+8,0 ± 1,2 für die Summe), GCN 78,8 %, MLP 51,0 % (Raten 54,1 %). | `test_count_rule_the_sum_wins_in_every_area` |
| **Dichte** | GIN 96,9 %, GraphSAGE 86,6 % (**+10,3 ± 0,8**, GIN in 12 von 12 vorn), GIN mit Mittel 86,3 % (+10,6 ± 0,9), GCN 80,8 %, MLP 49,9 %. | `test_density_rule_the_sum_wins_by_ten_points` |
| **Dichte ohne Merkmale** (nur die 1) | GIN **98,8 %**, GCN 68,0 %, GraphSAGE **55,8 %**, GIN mit Mittel 55,2 %, MLP 50,4 % (Raten 52,7 %): **+43,1 ± 1,2** Punkte. Ein Mittel über lauter gleiche Nachbarn ist immer 1; das GCN ahnt den Grad über seine Normierung. | `test_density_rule_without_features_the_sum_is_the_only_one_that_counts` |
| **Skalierung** (GIN mit ungeteilter Summe minus GIN) | Anteil **−14,6 ± 1,5** (Training 86,0 % gegen 98,3 %), Anzahl −2,9 ± 0,7, Dichte −1,6 ± 0,6, Dichte ohne Merkmale −0,9 ± 0,5. | `test_share_rule_the_mean_wins_and_the_unscaled_sum_falls_far_behind`, `test_count_rule_the_sum_wins_in_every_area` |
| **Braucht das GIN sein MLP?** (Ein-Schicht-Perzeptron minus MLP) | Anteil −0,7 ± 1,9, Anzahl −0,5 ± 0,7, Dichte +0,3 ± 0,2, Dichte ohne Merkmale +0,0 ± 0,1: **nein**, in diesen Aufgaben genügt eine Matrix. | `test_the_perceptron_is_enough_in_all_four_cases` |
| **Wo GIN scheitert** (Zufallsgewichte, lauter gleiche Merkmale) | Sechs Stopps als ein Kreis gegen zwei Dreiecke: **gleiche** Einbettungen bei MLP, GCN, GraphSAGE, GIN mit Mittel und GIN mit ungeteilter Summe (jeder Knoten hat zwei Nachbarn). Mitte eines Sterns mit 2 gegen 3 Blättern: gleich bei MLP, GraphSAGE, GIN mit Mittel; **verschieden** bei GCN und GIN. | `test_cycle_versus_two_triangles_is_indistinguishable_for_every_model_and_for_wl`, `test_star_center_is_separated_only_by_models_that_can_count`, `test_expressiveness_table_values` |
| Standardfall (Preset, Seed 4) | Anzahl ≥ 3: GIN 92,9 %, GraphSAGE 86,7 %, GCN 82,9 %, GIN mit Mittel 85,2 %, MLP 50,5 % (Raten 50,5 %). | `test_standard_preset_the_sum_beats_the_mean_networks_and_the_mlp_is_at_chance` |
| Anteil-Preset (Seed 2) | GraphSAGE 100,0 %, GIN 93,3 %, GIN mit Mittel 85,2 %, GCN 75,2 %, MLP 65,7 %. | `test_share_preset_the_mean_wins` |
| Dichte-Preset (Seed 0) | GIN 95,2 %, GCN 84,8 %, GIN mit Mittel 84,8 %, GraphSAGE 84,3 %, MLP 45,2 % (Raten 52,4 %). | `test_density_preset_the_sum_leads_by_a_lot` |
| Dichte ohne Merkmale (Preset, Seed 2) | GIN 99,5 %, GCN 73,3 %, GraphSAGE 57,1 %, GIN mit Mittel 57,1 %, MLP 53,3 %. | `test_density_without_features_only_the_sum_sees_the_degree` |
| Wenige bekannte Kunden (Preset, Seed 5, 30 von 300) | GIN 96,3 %, GIN mit Mittel 89,3 %, GraphSAGE 86,3 %, GCN 76,3 %, MLP 51,9 %. | `test_few_known_customers_and_one_layer_presets` |
| Eine Schicht (Preset, Seed 4) | GIN 99,0 %, GraphSAGE 82,4 %, GIN mit Mittel 82,4 %, GCN 80,5 %, MLP 50,5 %. | `test_few_known_customers_and_one_layer_presets` |

Die Preset-Zeilen sind **Einzelgebiete** (Seeds nahe dem Median über zwölf Gebiete, Klassen nicht stark ungleich groß); die Tests prüfen dort nur Strukturgrenzen. Belastbar sind die Mehr-Seed-Zeilen.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Frage hängt an einer Zahl (Anzahl, Grad)** | Hängt sie an einem Anteil, ist der Mittelwert die passendere Größe: GraphSAGE liegt beim Anteil vor dem GIN. | – |
| **Die Summe lässt sich trainieren** | Ungeteilt wächst sie mit dem Grad und kostet beim Anteil 14,6 Punkte; hier teilt eine feste Konstante (mittlerer Grad) sie, im Aufsatz die Batch-Normalisierung. | – |
| **Nachbarschaftsmengen unterscheiden genügt** | Zwei verschiedene Graphen, in denen jeder Knoten gleich aussieht (Kreis gegen zwei Dreiecke), bleiben auch für das GIN gleich (1-WL-Grenze): eine Tour über sechs Stopps oder zwei über je drei erkennt kein Nachrichten-Netz. | Graph Transformer (nächstes Stück) |
| **Der ganze Graph passt in den Speicher** | Dichte Matrizen in dieser Demo. | GraphSAGE (Vorgänger) |
| **Erzeugte Daten, zwölf Gebiete je Messpunkt** | Ein Vehikel mit Regeln, die genau eine Aggregation belohnen; die Zahlen gelten für diese Größen, Standardfehler sind groß; kein Test auf echten Kundendaten; nur zwei Schichten und die Standard-Hyperparameter gemessen. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, rund 2 Minuten): Kern per Handrechnung und Gegenprobe (Matrizen, GIN- und SAGE-Schicht, Gradienten aller Varianten, MLP-Grenzfall, Ausdrucksstärke gegen den unabhängigen 1-WL-Test, Training), Vehikel und Auswertung (Regeln von Hand, Aufteilung,
Paar mit gleichem Anteil, Experimentzeilen, Ausdrucksstärke-Tabelle), Preset- und Permalink-Klemmen, AppTest-Rauchtests (jedes Preset, Regel-Wechsel mit passendem Regler, Extremwerte, Experiment auf Abruf) und `test_claims.py` (jede Zahl aus diesem README; Einzelgebiete nur mit
Strukturgrenzen, Mehr-Seed-Zahlen mit großzügigen Bändern).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `gn_constants.py` | Regler-Grenzen, Netz-Konstanten, Regeln, Experiment-Seeds |
| `gn_presets.py` | Permalink/Presets-Mechanik |
| `gn_scenario.py` | Vehikel H, Regeln, Aufteilung, Paar mit gleichem Anteil, kleine Vergleichsgraphen |
| `gn_algorithm.py` | GCN/GraphSAGE/GIN-Varianten (Schichten, Gradienten, Adam) |
| `gn_evaluation.py` | Analyse, Experiment, Ausdrucksstärke |
| `gn_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Lernbares $\epsilon$, Batch-Normalisierung, Graph-Klassifikation mit Summen-Auslese über alle Knoten (hier Knoten-Klassifikation), Dropout, Feinabstimmung der Hyperparameter.
- Die Ursachenforschung, warum GraphSAGE beim Anteil vor dem GIN mit Mittel liegt (beide mitteln; GraphSAGE hat ein eigenes Gewicht für den Kunden).
- Das letzte Stück der Linie (Graph Transformer); ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
