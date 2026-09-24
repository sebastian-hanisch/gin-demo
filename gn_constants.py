"""Konstanten der GIN-Demo: Vehikel H "Kundendichte" (Kunden mit ungleichmäßiger Dichte, Nachbarschaft im Suchradius, Typ A/B je Kunde; gesucht: Kunden, die eine Nachbarschaftsregel erfüllen), Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999
AREA = 100.0

N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 400, 20, 300
RADIUS_MIN, RADIUS_MAX, RADIUS_STEP, DEFAULT_RADIUS = 4.0, 9.0, 0.5, 5.5
SHARE_A_MIN, SHARE_A_MAX, SHARE_A_STEP, DEFAULT_SHARE_A = 0.2, 0.6, 0.05, 0.4
KNOWN_MIN, KNOWN_MAX, KNOWN_STEP, DEFAULT_KNOWN = 0.1, 0.6, 0.05, 0.3
LAYERS_MIN, LAYERS_MAX, DEFAULT_LAYERS = 1, 3, 2
HIDDEN = 16
EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

CLUSTERS = 3
CLUSTER_SHARE = 0.6                                  # Anteil der Kunden in dichten Ballungen (Rest gleichmäßig im Gebiet)
CLUSTER_SIGMA = 7.0

RULES = ("Anteil", "Anzahl", "Dichte")
RULE_TEXT = {
    "Anteil": "mindestens ein bestimmter Anteil der Nachbarn ist vom Typ A",
    "Anzahl": "mindestens eine bestimmte Zahl der Nachbarn ist vom Typ A",
    "Dichte": "der Kunde hat mindestens eine bestimmte Zahl von Nachbarn (Typ egal)",
}
FEATURE_MODES = ("Typ und Konstante", "nur Konstante")
DEFAULT_RULE = "Anzahl"
FRAC_MIN, FRAC_MAX, FRAC_STEP, DEFAULT_FRAC = 0.2, 0.8, 0.05, 0.5
COUNT_MIN, COUNT_MAX, DEFAULT_COUNT = 1, 8, 3
DEG_MIN, DEG_MAX, DEFAULT_DEG = 3, 15, 7

# --- Experimente (feste Seeds) --------------------------------------------------------------------------------------------------------------

EXP_SEEDS = tuple(range(12))
