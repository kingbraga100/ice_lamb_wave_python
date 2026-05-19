# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd

from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d


# =========================================================
# PATHS
# =========================================================
def find_project_dir():
    here = Path(__file__).resolve()

    for parent in [here.parent] + list(here.parents):
        if (parent / "data_temperature_ice").exists():
            return parent

    raise RuntimeError(
        "Could not find project folder. "
        "Make sure this script is inside the repository and that "
        "'data_temperature_ice' exists in the main project folder."
    )


PROJECT_DIR = find_project_dir()
DATA_DIR = PROJECT_DIR / "data_temperature_ice"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

folders = {
    "B1_initial_baseline": DATA_DIR / "data_temperature_2026_05_17_1",
    "B2_initial_baseline": DATA_DIR / "data_temperature_2026_05_17_2",

    "I1cm_1": DATA_DIR / "data_temperature_2026_05_17_3",
    "I1cm_2": DATA_DIR / "data_temperature_2026_05_17_4",

    "I2cm_D1_1": DATA_DIR / "data_temperature_2026_05_17_5",
    "I2cm_D1_2": DATA_DIR / "data_temperature_2026_05_17_6",

    "I2cm_D2_1": DATA_DIR / "data_temperature_2026_05_17_7",
    "I2cm_D2_2": DATA_DIR / "data_temperature_2026_05_17_8",

    "I4cm_1": DATA_DIR / "data_temperature_2026_05_17_9",
    "I4cm_2": DATA_DIR / "data_temperature_2026_05_17_10",

    "AFTER_baseline": DATA_DIR / "data_temperature_2026_05_17_11",
}

OUTPUT_CSV = RESULTS_DIR / "ml_fixed_reference_metrics_may17.csv"


# =========================================================
# SETTINGS
# =========================================================
REF_TEMP_C = -20.0

EMITTER_THRESHOLD = 0.10

A0_START_US = 85
A0_END_US = 165

SMOOTH_SIGMA = 3
PERCENTILE = 95


# =========================================================
# LABELS
# =========================================================
dataset_condition = {
    "B1_initial_baseline": "Baseline",
    "B2_initial_baseline": "Baseline",

    "I1cm_1": "Ø1 cm ice",
    "I1cm_2": "Ø1 cm ice",

    "I2cm_D1_1": "Ø2 cm ice",
    "I2cm_D1_2": "Ø2 cm ice",

    "I2cm_D2_1": "Ø2 cm ice repeat",
    "I2cm_D2_2": "Ø2 cm ice repeat",

    "I4cm_1": "Ø4 cm ice",
    "I4cm_2": "Ø4 cm ice",

    "AFTER_baseline": "After removal",
}

dataset_sweep = {
    "B1_initial_baseline": "Cooling",
    "B2_initial_baseline": "Heating",

    "I1cm_1": "Cooling",
    "I1cm_2": "Heating",

    "I2cm_D1_1": "Cooling",
    "I2cm_D1_2": "Heating",

    "I2cm_D2_1": "Cooling",
    "I2cm_D2_2": "Heating",

    "I4cm_1": "Cooling",
    "I4cm_2": "Heating",

    "AFTER_baseline": "After",
}


# =========================================================
# BASIC FUNCTIONS
# =========================================================
def parse_temperature_from_filename(name):
    m = re.search(r"wave_T([-+]?\d+\.?\d*)C", name)
    if m is None:
        return None
    return float(m.group(1))


def detect_emitter_start(t, emitter):
    e = emitter - np.median(emitter)
    idx = np.where(np.abs(e) > EMITTER_THRESHOLD)[0]

    if len(idx) == 0:
        return None

    return idx[0]


def compute_envelope(receiver):
    env = np.abs(hilbert(receiver))
    env = gaussian_filter1d(env, sigma=SMOOTH_SIGMA)
    return env


def safe_corrcoef(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    a = a[good]
    b = b[good]

    if len(a) < 5:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


def compute_sdc(current, reference):
    good = np.isfinite(current) & np.isfinite(reference)
    current = current[good]
    reference = reference[good]

    if len(current) < 5:
        return np.nan

    denom = np.sum(reference ** 2)

    if denom <= 0:
        return np.nan

    return np.sum((current - reference) ** 2) / denom


# =========================================================
# LOADING
# =========================================================
def load_waveform(path, dataset):
    name = path.name
    temp = parse_temperature_from_filename(name)

    if temp is None:
        return None

    try:
        df = pd.read_csv(path, sep=";", skiprows=3, decimal=",")
    except Exception:
        return None

    required = ["Time_us", "Emitter_V", "Receiver_V"]

    if not all(c in df.columns for c in required):
        return None

    t = pd.to_numeric(df["Time_us"], errors="coerce").to_numpy()
    emitter = pd.to_numeric(df["Emitter_V"], errors="coerce").to_numpy()
    receiver = pd.to_numeric(df["Receiver_V"], errors="coerce").to_numpy()

    good = np.isfinite(t) & np.isfinite(emitter) & np.isfinite(receiver)

    t = t[good]
    emitter = emitter[good]
    receiver = receiver[good]

    if len(t) < 300:
        return None

    order = np.argsort(t)

    t = t[order]
    emitter = emitter[order]
    receiver = receiver[order]

    idx0 = detect_emitter_start(t, emitter)

    if idx0 is None:
        return None

    t0 = t[idx0]
    t = t - t0

    pre = (t >= -30) & (t <= -5)

    if np.sum(pre) >= 10:
        receiver = receiver - np.median(receiver[pre])
    else:
        receiver = receiver - np.median(receiver)

    envelope = compute_envelope(receiver)

    a0_mask = (t >= A0_START_US) & (t <= A0_END_US)

    if np.sum(a0_mask) < 40:
        return None

    return {
        "dataset": dataset,
        "condition": dataset_condition[dataset],
        "sweep": dataset_sweep[dataset],
        "file": name,
        "Temperature_C": temp,
        "t": t,
        "receiver": receiver,
        "envelope": envelope,
    }


def load_all_waveforms():
    items = []

    print("\nProject folder:")
    print(PROJECT_DIR)

    print("\nLoading waveform files...")

    for dataset, folder in folders.items():
        paths = sorted(folder.glob("wave_T*.csv"))

        count = 0

        for path in paths:
            item = load_waveform(path, dataset)

            if item is not None:
                items.append(item)
                count += 1

        print(f"{dataset:22s}: {count:4d} valid files")

    if len(items) == 0:
        raise RuntimeError(
            "No valid waveform files loaded. "
            "Check that the May 17 folders are inside data_temperature_ice."
        )

    return items


# =========================================================
# FIXED REFERENCE METRICS
# =========================================================
def choose_fixed_reference(items):
    baselines = [it for it in items if it["condition"] == "Baseline"]

    if len(baselines) == 0:
        raise RuntimeError("No baseline files found.")

    ref = min(baselines, key=lambda it: abs(it["Temperature_C"] - REF_TEMP_C))

    print("\nFixed reference selected:")
    print(f"  file        : {ref['file']}")
    print(f"  dataset     : {ref['dataset']}")
    print(f"  temperature : {ref['Temperature_C']:.2f} °C")

    return ref


def get_a0(item):
    t = item["t"]
    s = item["receiver"]
    e = item["envelope"]

    mask = (t >= A0_START_US) & (t <= A0_END_US)

    return t[mask], s[mask], e[mask]


def compute_metrics(items, ref):
    t_ref, s_ref, e_ref = get_a0(ref)

    rows = []

    for item in items:
        t, s, e = get_a0(item)

        if len(t) < 40:
            continue

        s_ref_i = np.interp(t, t_ref, s_ref)
        e_ref_i = np.interp(t, t_ref, e_ref)

        s0 = s - np.mean(s)
        sr0 = s_ref_i - np.mean(s_ref_i)

        e0 = e - np.mean(e)
        er0 = e_ref_i - np.mean(e_ref_i)

        rows.append({
            "dataset": item["dataset"],
            "condition": item["condition"],
            "sweep": item["sweep"],
            "file": item["file"],
            "Temperature_C": item["Temperature_C"],

            "fixed_reference_file": ref["file"],
            "fixed_reference_temperature_C": ref["Temperature_C"],

            "A0_p95_mV": np.percentile(e * 1000.0, PERCENTILE),
            "A0_max_mV": np.max(e * 1000.0),

            "Energy_wave_mV2": np.sum((s * 1000.0) ** 2),
            "Energy_env_mV2": np.sum((e * 1000.0) ** 2),

            "Corr_wave_fixed_ref": safe_corrcoef(s0, sr0),
            "Corr_env_fixed_ref": safe_corrcoef(e0, er0),

            "SDC_wave_fixed_ref": compute_sdc(s0, sr0),
            "SDC_env_fixed_ref": compute_sdc(e0, er0),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["condition", "sweep", "Temperature_C"])
    df = df.reset_index(drop=True)

    return df


# =========================================================
# MAIN
# =========================================================
def main():
    items = load_all_waveforms()
    ref = choose_fixed_reference(items)

    df = compute_metrics(items, ref)

    df.to_csv(OUTPUT_CSV, sep=";", decimal=",", index=False)

    print("\nSaved CSV:")
    print(OUTPUT_CSV)

    print("\nCounts:")
    print(df.groupby(["condition", "sweep"])["file"].count())


if __name__ == "__main__":
    main()