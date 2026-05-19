# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:25:12 2026

@author: ricar
"""

# -*- coding: utf-8 -*-
"""


This version:
- no outlier detection
- no file moving
- minimal printing
- temperature-matched baseline reference
- simple y-axis labels
- clearer titles
- legend inside, two columns
- uses Ø symbol for ice diameter in legend
"""

import os
import re
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import hilbert, correlate, correlation_lags
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D


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
FIG_DIR = PROJECT_DIR / "figures" / "compute_a0_metrics_temperature_ice"

RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Keep BASE_PATH as the data folder so the dataset definitions below stay simple.
BASE_PATH = str(DATA_DIR)

folders = {
    "B1_initial_baseline": os.path.join(BASE_PATH, "data_temperature_2026_05_17_1"),
    "B2_initial_baseline": os.path.join(BASE_PATH, "data_temperature_2026_05_17_2"),

    "I1cm_1": os.path.join(BASE_PATH, "data_temperature_2026_05_17_3"),
    "I1cm_2": os.path.join(BASE_PATH, "data_temperature_2026_05_17_4"),

    "I2cm_D1_1": os.path.join(BASE_PATH, "data_temperature_2026_05_17_5"),
    "I2cm_D1_2": os.path.join(BASE_PATH, "data_temperature_2026_05_17_6"),

    "I2cm_D2_1": os.path.join(BASE_PATH, "data_temperature_2026_05_17_7"),
    "I2cm_D2_2": os.path.join(BASE_PATH, "data_temperature_2026_05_17_8"),

    "I4cm_1": os.path.join(BASE_PATH, "data_temperature_2026_05_17_9"),
    "I4cm_2": os.path.join(BASE_PATH, "data_temperature_2026_05_17_10"),

    "AFTER_baseline": os.path.join(BASE_PATH, "data_temperature_2026_05_17_11"),
}

OUTPUT_CSV = RESULTS_DIR / "compute_a0_metrics_temperature_ice.csv"



# =========================================================
# SETTINGS
# =========================================================
EMITTER_THRESHOLD = 0.10

A0_START = 85
A0_END = 165

SMOOTH_SIGMA = 3
PERCENTILE = 95

SAVE_RESULTS = True
SAVE_FIGURES = True
SHOW_FIGURES = False

REFERENCE_BASELINE_DATASETS = [
    "B1_initial_baseline",
    "B2_initial_baseline",
]

MAX_REFERENCE_TEMP_DIFF = 1.0
MAX_SHIFT_US = 5.0


# =========================================================
# VISUAL STYLE
# =========================================================
condition_colors = {
    "Baseline": "blue",
    "Ø1 cm ice": "darkorange",
    "Ø2 cm ice": "green",
    "Ø2 cm ice repeat": "purple",
    "Ø4 cm ice": "red",
    "After removal": "black",
}

condition_markers = {
    "Baseline": "o",
    "Ø1 cm ice": "s",
    "Ø2 cm ice": "^",
    "Ø2 cm ice repeat": "D",
    "Ø4 cm ice": "P",
    "After removal": "x",
}

sweep_linestyles = {
    "Cooling": "-",
    "Heating": "--",
    "After": "-.",
}

sweep_alpha = {
    "Cooling": 1.00,
    "Heating": 0.55,
    "After": 1.00,
}

sweep_width = {
    "Cooling": 3.0,
    "Heating": 2.4,
    "After": 2.8,
}

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

plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
})

# Font sizes for report subfigures
AXIS_LABEL_FONT = 26
TICK_FONT = 22
PZT_LABEL_FONT = 24
LEGEND_FONT = 20

# =========================================================
# BASIC FUNCTIONS
# =========================================================
def parse_temperature_from_filename(name):
    m = re.search(r"wave_T([-+]?\d+\.?\d*)C", name)
    return float(m.group(1)) if m else None


def detect_emitter_start(t, emitter):
    e = emitter - np.median(emitter)
    idx = np.where(np.abs(e) > EMITTER_THRESHOLD)[0]
    return idx[0] if len(idx) else None


def compute_envelope(receiver):
    return gaussian_filter1d(np.abs(hilbert(receiver)), sigma=SMOOTH_SIGMA)


def safe_corrcoef(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    a = a[good]
    b = b[good]

    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


def compute_sdc(current, reference):
    good = np.isfinite(current) & np.isfinite(reference)
    current = current[good]
    reference = reference[good]

    denom = np.sum(reference ** 2)

    if len(current) < 5 or denom <= 0:
        return np.nan

    return np.sum((current - reference) ** 2) / denom


def xcorr_tof_shift_us(current, reference, dt_us):
    current = current - np.mean(current)
    reference = reference - np.mean(reference)

    if np.std(current) == 0 or np.std(reference) == 0:
        return np.nan, np.nan

    c = correlate(current, reference, mode="full")
    lags = correlation_lags(len(current), len(reference), mode="full")

    max_shift_points = int(round(MAX_SHIFT_US / dt_us))
    valid = np.abs(lags) <= max_shift_points

    if not np.any(valid):
        return np.nan, np.nan

    valid_lags = lags[valid]
    valid_corr = c[valid]

    best_lag_points = valid_lags[np.argmax(valid_corr)]
    best_lag_us = best_lag_points * dt_us

    return best_lag_points, best_lag_us


# =========================================================
# LOADING
# =========================================================
def load_file(path, dataset):
    name = os.path.basename(path)
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

    a0_mask = (t >= A0_START) & (t <= A0_END)
    if np.sum(a0_mask) < 40:
        return None

    return {
        "dataset": dataset,
        "condition": dataset_condition[dataset],
        "sweep": dataset_sweep[dataset],
        "file": name,
        "path": path,
        "temp": temp,
        "t": t,
        "receiver": receiver,
        "envelope": envelope,
        "t0": t0,
    }


def load_all_data():
    all_data = {}

    print("\nLoading files...")

    for dataset, folder in folders.items():
        files = glob.glob(os.path.join(folder, "wave_T*.csv"))
        data = []

        for path in files:
            item = load_file(path, dataset)
            if item is not None:
                data.append(item)

        data.sort(key=lambda d: d["temp"])
        all_data[dataset] = data

        print(f"{dataset:22s}: {len(data):4d} files")

    return all_data


# =========================================================
# METRICS
# =========================================================
def build_reference_library(all_data):
    refs = []

    for dataset in REFERENCE_BASELINE_DATASETS:
        refs.extend(all_data.get(dataset, []))

    refs.sort(key=lambda d: d["temp"])
    return refs


def choose_reference(temp, refs):
    ref = min(refs, key=lambda d: abs(d["temp"] - temp))
    diff = abs(ref["temp"] - temp)

    if diff > MAX_REFERENCE_TEMP_DIFF:
        return None, diff

    return ref, diff


def get_a0_vectors(current, reference):
    t = current["t"]
    s = current["receiver"]
    e = current["envelope"]

    tr = reference["t"]
    sr = reference["receiver"]
    er = reference["envelope"]

    mask = (t >= A0_START) & (t <= A0_END)
    mask_r = (tr >= A0_START) & (tr <= A0_END)

    if np.sum(mask) < 40 or np.sum(mask_r) < 40:
        return None

    t_a0 = t[mask]
    s_a0 = s[mask]
    e_a0 = e[mask]

    tr_a0 = tr[mask_r]
    sr_a0 = sr[mask_r]
    er_a0 = er[mask_r]

    sr_i = np.interp(t_a0, tr_a0, sr_a0)
    er_i = np.interp(t_a0, tr_a0, er_a0)

    return t_a0, s_a0, e_a0, sr_i, er_i


def compute_metrics(all_data):
    refs = build_reference_library(all_data)

    if len(refs) == 0:
        raise RuntimeError("No baseline reference files found.")

    rows = []

    print("\nComputing metrics...")

    for dataset, data in all_data.items():
        for d in data:
            ref, ref_diff = choose_reference(d["temp"], refs)

            if ref is None:
                continue

            vectors = get_a0_vectors(d, ref)

            if vectors is None:
                continue

            t_a0, s_a0, e_a0, sr_a0, er_a0 = vectors

            dt_us = np.median(np.diff(t_a0))

            s0 = s_a0 - np.mean(s_a0)
            sr0 = sr_a0 - np.mean(sr_a0)

            e0 = e_a0 - np.mean(e_a0)
            er0 = er_a0 - np.mean(er_a0)

            lag_w_points, lag_w_us = xcorr_tof_shift_us(s0, sr0, dt_us)
            lag_e_points, lag_e_us = xcorr_tof_shift_us(e0, er0, dt_us)

            rows.append({
                "dataset": dataset,
                "condition": d["condition"],
                "sweep": d["sweep"],
                "file": d["file"],
                "Temperature_C": d["temp"],

                "Reference_dataset": ref["dataset"],
                "Reference_file": ref["file"],
                "Reference_temperature_C": ref["temp"],
                "Reference_temperature_difference_C": ref_diff,

                "A0_p95_mV": np.percentile(e_a0 * 1000.0, PERCENTILE),
                "A0_max_mV": np.max(e_a0 * 1000.0),

                "Energy_wave_mV2": np.sum((s_a0 * 1000.0) ** 2),
                "Energy_env_mV2": np.sum((e_a0 * 1000.0) ** 2),

                "Corr_wave": safe_corrcoef(s0, sr0),
                "Corr_env": safe_corrcoef(e0, er0),

                "SDC_wave": compute_sdc(s0, sr0),
                "SDC_env": compute_sdc(e0, er0),

                "Lag_wave_points": lag_w_points,
                "Lag_wave_us": lag_w_us,

                "Lag_env_points": lag_e_points,
                "Lag_env_us": lag_e_us,
            })

    return pd.DataFrame(rows)


# =========================================================
# PLOTTING
# =========================================================
def add_legend(ax, loc):
    labels = [
        "Baseline",
        "Ø1 cm ice",
        "Ø2 cm ice",
        "Ø2 cm ice repeat",
        "Ø4 cm ice",
        "After removal",
    ]

    handles = []

    for label in labels:
        handles.append(
            Line2D(
                [0], [0],
                marker=condition_markers[label],
                color=condition_colors[label],
                linestyle="None",
                markersize=9,
                markerfacecolor=condition_colors[label] if condition_markers[label] != "x" else "none",
                markeredgecolor=condition_colors[label],
                label=label,
            )
        )

    for sweep in ["Cooling", "Heating", "After"]:
        handles.append(
            Line2D(
                [0], [0],
                color="gray",
                linestyle=sweep_linestyles[sweep],
                linewidth=sweep_width[sweep],
                alpha=sweep_alpha[sweep],
                label=sweep,
            )
        )

    ax.legend(
        handles=handles,
        loc=loc,
        ncol=2,
        framealpha=0.92,
        borderpad=0.55,
        labelspacing=0.38,
        columnspacing=1.1,
        handlelength=2.3,
    )


def finish_figure(fig, name):
    fig.tight_layout()

    if SAVE_FIGURES:
        safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)
        path = FIG_DIR / (safe + ".png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def plot_metric(df, ycol, ylabel, title, ylim=None, zero_line=False, legend_loc="best"):
    fig, ax = plt.subplots(figsize=(14.2, 8.5))

    for dataset, sub in df.groupby("dataset"):
        condition = dataset_condition[dataset]
        sweep = dataset_sweep[dataset]

        ax.scatter(
            sub["Temperature_C"],
            sub[ycol],
            s=90,
            marker=condition_markers[condition],
            color=condition_colors[condition],
            alpha=max(0.40, sweep_alpha[sweep]),
            edgecolors="black" if condition_markers[condition] != "x" else None,
            linewidths=0.35,
            zorder=3,
        )

        good = np.isfinite(sub["Temperature_C"]) & np.isfinite(sub[ycol])

        if np.sum(good) >= 3:
            x = sub.loc[good, "Temperature_C"].to_numpy()
            y = sub.loc[good, ycol].to_numpy()

            p = np.polyfit(x, y, 1)
            xfit = np.linspace(np.min(x), np.max(x), 100)
            yfit = np.polyval(p, xfit)

            ax.plot(
                xfit,
                yfit,
                color=condition_colors[condition],
                linestyle=sweep_linestyles[sweep],
                linewidth=sweep_width[sweep],
                alpha=sweep_alpha[sweep] * 0.70,
                zorder=2,
            )

    if zero_line:
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.8)

    if ylim is not None:
        ax.set_ylim(*ylim)

    xmin = np.nanmin(df["Temperature_C"])
    xmax = np.nanmax(df["Temperature_C"])
    ax.set_xlim(xmin - 0.3, xmax + 1.0)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(True, alpha=0.28)

    add_legend(ax, legend_loc)
    finish_figure(fig, title)


# =========================================================
# MAIN
# =========================================================
all_data = load_all_data()
metrics_df = compute_metrics(all_data)

if len(metrics_df) == 0:
    raise RuntimeError("No metrics computed.")

if SAVE_RESULTS:
    metrics_df.to_csv(OUTPUT_CSV, sep=";", decimal=",", index=False)
    print(f"\nSaved metrics CSV:\n{OUTPUT_CSV}")

print("\nCounts:")
print(metrics_df.groupby(["condition", "sweep"])["file"].count())


# =========================================================
# PLOTS
# =========================================================

plot_metric(
    metrics_df,
    "Corr_wave",
    "Waveform correlation",
    "A0 waveform correlation, ice height h = 2 mm",
    ylim=(-0.50, 1.02),
    legend_loc="lower right",
)

plot_metric(
    metrics_df,
    "Corr_env",
    "Envelope correlation",
    "A0 envelope correlation, ice height h = 2 mm",
    ylim=(-0.50, 1.02),
    legend_loc="lower right",
)

plot_metric(
    metrics_df,
    "SDC_wave",
    "Waveform SDC",
    "A0 waveform SDC, ice height h = 2 mm",
    ylim=(-0.05, None),
    zero_line=True,
    legend_loc="upper right",
)

plot_metric(
    metrics_df,
    "SDC_env",
    "Envelope SDC",
    "A0 envelope SDC, ice height h = 2 mm",
    ylim=(-0.05, None),
    zero_line=True,
    legend_loc="upper right",
)

plot_metric(
    metrics_df,
    "Lag_wave_us",
    "Waveform ToF shift (µs)",
    "A0 waveform ToF shift, ice height h = 2 mm",
    ylim=(-5.2, 5.2),
    zero_line=True,
    legend_loc="upper right",
)

plot_metric(
    metrics_df,
    "Lag_env_us",
    "Envelope ToF shift (µs)",
    "A0 envelope ToF shift, ice height h = 2 mm",
    ylim=(-6.5, 6.5),
    zero_line=True,
    legend_loc="upper right",
)

plot_metric(
    metrics_df,
    "A0_p95_mV",
    "Envelope amplitude, 95th percentile (mV)",
    "A0 envelope amplitude, ice height h = 2 mm",
    legend_loc="lower right",
)

plot_metric(
    metrics_df,
    "A0_max_mV",
    "Maximum envelope amplitude (mV)",
    "A0 maximum envelope amplitude, ice height h = 2 mm",
    legend_loc="lower right",
)

plot_metric(
    metrics_df,
    "Energy_wave_mV2",
    "Waveform energy (mV²)",
    "A0 waveform energy, ice height h = 2 mm",
    legend_loc="lower right",
)

plot_metric(
    metrics_df,
    "Energy_env_mV2",
    "Envelope energy (mV²)",
    "A0 envelope energy, ice height h = 2 mm",
    legend_loc="lower right",
)

print("\nDone.")