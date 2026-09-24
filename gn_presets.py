"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. sg_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import gn_constants as C


def _choice(options):
    def cast(value):
        if str(value) not in options:
            raise ValueError(value)
        return str(value)
    return cast


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "radius_slider": SettingSpec("radius", float, C.DEFAULT_RADIUS, C.RADIUS_MIN, C.RADIUS_MAX),
    "share_slider": SettingSpec("share", float, C.DEFAULT_SHARE_A, C.SHARE_A_MIN, C.SHARE_A_MAX),
    "features_select": SettingSpec("features", _choice(C.FEATURE_MODES), C.FEATURE_MODES[0]),
    "rule_select": SettingSpec("rule", _choice(C.RULES), C.DEFAULT_RULE),
    "frac_slider": SettingSpec("frac", float, C.DEFAULT_FRAC, C.FRAC_MIN, C.FRAC_MAX),
    "count_slider": SettingSpec("count", int, C.DEFAULT_COUNT, C.COUNT_MIN, C.COUNT_MAX),
    "deg_slider": SettingSpec("deg", int, C.DEFAULT_DEG, C.DEG_MIN, C.DEG_MAX),
    "known_slider": SettingSpec("known", float, C.DEFAULT_KNOWN, C.KNOWN_MIN, C.KNOWN_MAX),
    "layers_slider": SettingSpec("layers", int, C.DEFAULT_LAYERS, C.LAYERS_MIN, C.LAYERS_MAX),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "radius": "radius_slider", "share": "share_slider", "features": "features_select", "rule": "rule_select", "frac": "frac_slider", "count": "count_slider", "deg": "deg_slider",
               "known": "known_slider", "layers": "layers_slider", "seed": "seed_input"}
STEPS = {"n_slider": C.N_STEP, "radius_slider": C.RADIUS_STEP, "share_slider": C.SHARE_A_STEP, "frac_slider": C.FRAC_STEP, "known_slider": C.KNOWN_STEP}


def _p(**kw):
    base = {"n": C.DEFAULT_N, "radius": C.DEFAULT_RADIUS, "share": C.DEFAULT_SHARE_A, "features": C.FEATURE_MODES[0], "rule": C.DEFAULT_RULE, "frac": C.DEFAULT_FRAC, "count": C.DEFAULT_COUNT, "deg": C.DEFAULT_DEG,
            "known": C.DEFAULT_KNOWN, "layers": C.DEFAULT_LAYERS, "seed": 4}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall (Anzahl ≥ 3)": _p(),
    "Anteil: Mittelwert genügt": _p(rule="Anteil", seed=2),
    "Dichte (Grad ≥ 7)": _p(rule="Dichte", seed=0),
    "Dichte ohne Merkmale": _p(rule="Dichte", features=C.FEATURE_MODES[1], seed=2),
    "Wenige bekannte Kunden (10 %)": _p(known=0.1, seed=5),
    "Eine Schicht": _p(layers=1),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall (Anzahl ≥ 3)": "Seed 4, 300 Kunden, Regel 'mindestens 3 Nachbarn vom Typ A', 30 % bekannt: GIN 92,9 %, GraphSAGE 86,7 %, GCN 82,9 %, GIN mit Mittel 85,2 %, MLP 50,5 % (Raten 50,5 %). Im Mittel über 12 Gebiete liegt die Summe etwa 7 Punkte vor GraphSAGE.",
    "Anteil: Mittelwert genügt": "Seed 2, Regel 'mindestens die Hälfte der Nachbarn vom Typ A': GraphSAGE 100,0 %, GIN 93,3 %, GIN mit Mittel 85,2 %, GCN 75,2 %, MLP 65,7 % (Raten 65,7 %). Ein Anteil ist genau ein Mittelwert - hier gewinnt GraphSAGE; im Mittel über 12 Gebiete liegt GraphSAGE etwa 7 Punkte vor dem GIN.",
    "Dichte (Grad ≥ 7)": "Seed 0, Regel 'mindestens 7 Nachbarn' (der Typ spielt keine Rolle): GIN 95,2 %, GCN 84,8 %, GIN mit Mittel 84,8 %, GraphSAGE 84,3 %, MLP 45,2 % (Raten 52,4 %). Die Summe kennt den Grad; im Mittel über 12 Gebiete liegt sie etwa 10 Punkte vor GraphSAGE.",
    "Dichte ohne Merkmale": "Seed 2, dieselbe Regel, aber die Netze sehen nur eine Konstante 1: GIN 99,5 %, GCN 73,3 %, GraphSAGE 57,1 %, GIN mit Mittel 57,1 %, MLP 53,3 %. Ein Mittelwert über lauter gleiche Nachbarn ist immer 1 - nur die Summe kennt den Grad (das GCN ahnt ihn über seine Normierung).",
    "Wenige bekannte Kunden (10 %)": "Seed 5, nur 30 bekannte Kunden: GIN 96,3 %, GIN mit Mittel 89,3 %, GraphSAGE 86,3 %, GCN 76,3 %, MLP 51,9 %. Der Vorsprung der Summe bleibt auch mit wenigen Etiketten.",
    "Eine Schicht": "Seed 4, nur eine Nachrichtenschicht: GIN 99,0 %, GraphSAGE 82,4 %, GIN mit Mittel 82,4 %, GCN 80,5 %, MLP 50,5 %. Die Regel hängt nur an den direkten Nachbarn; die Summe braucht dafür keine zweite Schicht.",
}
