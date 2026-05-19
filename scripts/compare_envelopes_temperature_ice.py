# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
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
FIG_DIR = PROJECT_DIR / "figures" / "temperature_ice_envelopes"

RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================
TEST_NAME = "Ice-patch diameter sensitivity and baseline-recovery test"

TARGET_TEMPS = [-20.0, -15.0, -10.0, -5.0]

EMITTER_THRESHOLD = 0.10
SMOOTH_SIGMA = 3

T_PLOT_MIN = -20
T_PLOT_MAX = 250

A0_START = 85
A0_END = 165

SAVE_SUMMARY = True
OUT_FILE = RESULTS_DIR / "may17_envelope_waveform_diameter_summary.csv"

SAVE_FIGURES = True
SHOW_FIGURES = False
SAVE_PDF = True

FIG_COUNTER = 0


# =========================================================
# PLOT STYLE
# =========================================================
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
    "lines.linewidth": 3.0,
})


# =========================================================
# DATASETS
# =========================================================
datasets = {
    "B_C": {
        "case": "Initial baseline cooling",
        "condition": "Baseline",
        "group": "Initial baseline",
        "sweep": "Cooling",
        "folder": DATA_DIR / "data_temperature_2026_05_17_1",
    },
    "B_H": {
        "case": "Initial baseline heating",
        "condition": "Baseline",
        "group": "Initial baseline",
        "sweep": "Heating",
        "folder": DATA_DIR / "data_temperature_2026_05_17_2",
    },

    "D1_C": {
        "case": "Ø1 cm ice cooling, h = 2 mm",
        "condition": "Ø1 cm ice",
        "group": "Ø1 cm ice",
        "sweep": "Cooling",
        "folder": DATA_DIR / "data_temperature_2026_05_17_3",
    },
    "D1_H": {
        "case": "Ø1 cm ice heating, h = 2 mm",
        "condition": "Ø1 cm ice",
        "group": "Ø1 cm ice",
        "sweep": "Heating",
        "folder": DATA_DIR / "data_temperature_2026_05_17_4",
    },

    "D2_C": {
        "case": "Ø2 cm ice cooling, h = 2 mm",
        "condition": "Ø2 cm ice",
        "group": "Ø2 cm ice",
        "sweep": "Cooling",
        "folder": DATA_DIR / "data_temperature_2026_05_17_5",
    },
    "D2_H": {
        "case": "Ø2 cm ice heating, h = 2 mm",
        "condition": "Ø2 cm ice",
        "group": "Ø2 cm ice",
        "sweep": "Heating",
        "folder": DATA_DIR / "data_temperature_2026_05_17_6",
    },

    "D2R_C": {
        "case": "Ø2 cm ice repeat cooling, h = 2 mm",
        "condition": "Ø2 cm ice repeat",
        "group": "Ø2 cm ice repeat",
        "sweep": "Cooling",
        "folder": DATA_DIR / "data_temperature_2026_05_17_7",
    },
    "D2R_H": {
        "case": "Ø2 cm ice repeat heating, h = 2 mm",
        "condition": "Ø2 cm ice repeat",
        "group": "Ø2 cm ice repeat",
        "sweep": "Heating",
        "folder": DATA_DIR / "data_temperature_2026_05_17_8",
    },

    "D4_C": {
        "case": "Ø4 cm ice cooling, h = 2 mm",
        "condition": "Ø4 cm ice",
        "group": "Ø4 cm ice",
        "sweep": "Cooling",
        "folder": DATA_DIR / "data_temperature_2026_05_17_9",
    },
    "D4_H": {
        "case": "Ø4 cm ice heating, h = 2 mm",
        "condition": "Ø4 cm ice",
        "group": "Ø4 cm ice",
        "sweep": "Heating",
        "folder": DATA_DIR / "data_temperature_2026_05_17_10",
    },

    "AFTER": {
        "case": "Baseline after ice removal",
        "condition": "After removal",
        "group": "Baseline after",
        "sweep": "After",
        "folder": DATA_DIR / "data_temperature_2026_05_17_11",
    },
}


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
    "Cooling": 3.2,
    "Heating": 2.5,
    "After": 3.0,
}


# =========================================================
# HELPERS
# =========================================================
def safe_name(text):
    text = str(text).replace("Ø", "diam")
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)
    return text.strip("_")


def finish_figure(name):
    global FIG_COUNTER

    plt.tight_layout()

    if SAVE_FIGURES:
        FIG_COUNTER += 1

        base = f"{FIG_COUNTER:03d}_{safe_name(name)}"

        png_path = FIG_DIR / f"{base}.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved PNG: {png_path}")

        if SAVE_PDF:
            pdf_path = FIG_DIR / f"{base}.pdf"
            plt.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved PDF: {pdf_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def parse_temperature_from_filename(name):
    match = re.search(r"wave_T([-+]?\d+\.?\d*)C", name)

    if match is None:
        return None

    return float(match.group(1))


def detect_emitter_start(emitter, threshold):
    emitter_c = emitter - np.median(emitter)
    idx = np.where(np.abs(emitter_c) > threshold)[0]

    if len(idx) == 0:
        return None

    return idx[0]


def find_closest_file(folder, target_temp):
    files = sorted(folder.glob("wave_T*.csv"))
    candidates = []

    for f in files:
        temp = parse_temperature_from_filename(f.name)

        if temp is None:
            continue

        candidates.append((abs(temp - target_temp), temp, f))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][2]


def load_and_align_file(filepath):
    name = filepath.name
    temp = parse_temperature_from_filename(name)

    df = pd.read_csv(filepath, sep=";", skiprows=3, decimal=",")

    required = ["Time_us", "Emitter_V", "Receiver_V"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"{name}: missing column {col}")

    t = pd.to_numeric(df["Time_us"], errors="coerce").to_numpy()
    emitter = pd.to_numeric(df["Emitter_V"], errors="coerce").to_numpy()
    receiver = pd.to_numeric(df["Receiver_V"], errors="coerce").to_numpy()

    valid = np.isfinite(t) & np.isfinite(emitter) & np.isfinite(receiver)

    t = t[valid]
    emitter = emitter[valid]
    receiver = receiver[valid]

    if len(t) < 100:
        raise ValueError(f"{name}: too few valid points")

    order = np.argsort(t)

    t = t[order]
    emitter = emitter[order]
    receiver = receiver[order]

    idx0 = detect_emitter_start(emitter, EMITTER_THRESHOLD)

    if idx0 is None:
        peak = np.max(np.abs(emitter - np.median(emitter)))
        raise ValueError(
            f"{name}: emitter not detected. "
            f"Emitter peak = {peak:.3f} V, threshold = {EMITTER_THRESHOLD:.3f} V"
        )

    t0 = t[idx0]
    t_aligned = t - t0

    pre_mask = (t_aligned >= -30) & (t_aligned <= -5)

    if np.sum(pre_mask) > 10:
        emitter = emitter - np.median(emitter[pre_mask])
        receiver = receiver - np.median(receiver[pre_mask])
    else:
        emitter = emitter - np.median(emitter)
        receiver = receiver - np.median(receiver)

    env = np.abs(hilbert(receiver))
    env = gaussian_filter1d(env, sigma=SMOOTH_SIGMA)

    return {
        "file": name,
        "temp": temp,
        "t": t_aligned,
        "emitter": emitter,
        "receiver": receiver,
        "envelope": env,
    }


def interp_to_common_time(t_source, y_source, t_common):
    return np.interp(t_common, t_source, y_source)


def get_style(key):
    info = datasets[key]

    color = condition_colors[info["condition"]]
    linestyle = sweep_linestyles[info["sweep"]]
    alpha = sweep_alpha[info["sweep"]]
    linewidth = sweep_width[info["sweep"]]

    return color, linestyle, alpha, linewidth


def add_clean_legend(ax, loc="upper left", ncol=2):
    condition_handles = [
        Line2D([0], [0], color="blue", lw=4, label="Baseline"),
        Line2D([0], [0], color="darkorange", lw=4, label="Ø1 cm ice"),
        Line2D([0], [0], color="green", lw=4, label="Ø2 cm ice"),
        Line2D([0], [0], color="purple", lw=4, label="Ø2 cm ice repeat"),
        Line2D([0], [0], color="red", lw=4, label="Ø4 cm ice"),
        Line2D([0], [0], color="black", lw=4, label="After removal"),
    ]

    sweep_handles = [
        Line2D([0], [0], color="gray", lw=4, linestyle="-", alpha=1.00, label="Cooling"),
        Line2D([0], [0], color="gray", lw=4, linestyle="--", alpha=0.55, label="Heating"),
        Line2D([0], [0], color="gray", lw=4, linestyle="-.", alpha=1.00, label="After"),
    ]

    handles = condition_handles + sweep_handles

    ax.legend(
        handles=handles,
        loc=loc,
        fontsize=20,
        ncol=ncol,
        framealpha=0.92,
        borderpad=0.50,
        labelspacing=0.35,
        columnspacing=1.10,
        handlelength=2.40,
    )


def make_average_initial_baseline_envelope(selected, t_common):
    if "B_C" not in selected or "B_H" not in selected:
        return None

    env_c = interp_to_common_time(
        selected["B_C"]["t"],
        selected["B_C"]["envelope"] * 1000,
        t_common,
    )

    env_h = interp_to_common_time(
        selected["B_H"]["t"],
        selected["B_H"]["envelope"] * 1000,
        t_common,
    )

    return 0.5 * (env_c + env_h)


def make_average_initial_baseline_receiver(selected, t_common):
    if "B_C" not in selected or "B_H" not in selected:
        return None

    s_c = interp_to_common_time(
        selected["B_C"]["t"],
        selected["B_C"]["receiver"] * 1000,
        t_common,
    )

    s_h = interp_to_common_time(
        selected["B_H"]["t"],
        selected["B_H"]["receiver"] * 1000,
        t_common,
    )

    return 0.5 * (s_c + s_h)


def format_temp_for_title(temp):
    return f"{temp:.0f} °C"


# =========================================================
# MAIN
# =========================================================
def main():
    all_summary_rows = []

    print("\nProject folder:")
    print(PROJECT_DIR)

    for target_temp in TARGET_TEMPS:
        print("\n" + "=" * 70)
        print(f"PROCESSING TARGET TEMPERATURE: {target_temp:.0f} °C")
        print("=" * 70)

        selected = {}

        for key, info in datasets.items():
            f = find_closest_file(info["folder"], target_temp)

            if f is None:
                print(f"{key}: no valid files found")
                continue

            try:
                d = load_and_align_file(f)
                selected[key] = d

                mask_a0 = (d["t"] >= A0_START) & (d["t"] <= A0_END)

                if np.sum(mask_a0) > 5:
                    a0_max = np.max(d["envelope"][mask_a0]) * 1000
                    a0_p95 = np.percentile(d["envelope"][mask_a0], 95) * 1000
                else:
                    a0_max = np.nan
                    a0_p95 = np.nan

                print(
                    f"{key:8s} | "
                    f"{info['condition']:18s} | "
                    f"{info['sweep']:7s} | "
                    f"Tsel = {d['temp']:6.2f} °C | "
                    f"A0 max = {a0_max:7.2f} mV | "
                    f"A0 p95 = {a0_p95:7.2f} mV"
                )

            except Exception as e:
                print(f"{key}: ERROR -> {e}")

        if len(selected) == 0:
            print("No valid files loaded for this temperature.")
            continue

        global_ref_key = "B_C"

        if global_ref_key not in selected:
            print(f"Global reference {global_ref_key} not loaded. Skipping plots.")
            continue

        global_ref = selected[global_ref_key]
        t_common = np.linspace(T_PLOT_MIN, T_PLOT_MAX, 3000)

        env_ref_global = interp_to_common_time(
            global_ref["t"],
            global_ref["envelope"] * 1000,
            t_common,
        )

        env_ref_avg = make_average_initial_baseline_envelope(selected, t_common)

        if env_ref_avg is None:
            print("Average initial baseline reference unavailable. Using cooling baseline only.")
            env_ref_avg = env_ref_global.copy()

        temp_title = format_temp_for_title(target_temp)

        # =====================================================
        # PLOT 1: ENVELOPE COMPARISON
        # =====================================================
        fig, ax = plt.subplots(figsize=(18, 8.5))

        for key, d in selected.items():
            color, linestyle, alpha, linewidth = get_style(key)

            env_common = interp_to_common_time(
                d["t"],
                d["envelope"] * 1000,
                t_common,
            )

            ax.plot(
                t_common,
                env_common,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=linewidth,
            )

        ax.axvline(0, color="black", linestyle=":", linewidth=2.5)
        ax.axvspan(A0_START, A0_END, color="gray", alpha=0.20)

        ax.set_xlabel("Time after emitter start (µs)")
        ax.set_ylabel("Envelope (mV)")
        ax.set_title(f"A0 envelope comparison near {temp_title}, ice height h = 2 mm")
        ax.grid(True, alpha=0.30)

        add_clean_legend(ax, loc="upper left", ncol=2)

        finish_figure(f"A0 envelope comparison near {target_temp:.0f} C")

        # =====================================================
        # PLOT 2: ENVELOPE DIFFERENCE TO COOLING BASELINE
        # =====================================================
        fig, ax = plt.subplots(figsize=(18, 8.5))

        for key, d in selected.items():
            if key == global_ref_key:
                continue

            color, linestyle, alpha, linewidth = get_style(key)

            env_common = interp_to_common_time(
                d["t"],
                d["envelope"] * 1000,
                t_common,
            )

            diff = env_common - env_ref_global

            ax.plot(
                t_common,
                diff,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=linewidth,
            )

        ax.axhline(0, color="black", linestyle="-", linewidth=1.5)
        ax.axvline(0, color="black", linestyle=":", linewidth=2.5)
        ax.axvspan(A0_START, A0_END, color="gray", alpha=0.20)

        ax.set_xlabel("Time after emitter start (µs)")
        ax.set_ylabel("Envelope difference (mV)")
        ax.set_title(f"A0 envelope difference near {temp_title}, reference = cooling baseline")
        ax.grid(True, alpha=0.30)

        add_clean_legend(ax, loc="upper left", ncol=2)

        finish_figure(f"A0 envelope difference cooling baseline near {target_temp:.0f} C")

        # =====================================================
        # PLOT 3: ABSOLUTE ENVELOPE DIFFERENCE TO COOLING BASELINE
        # =====================================================
        fig, ax = plt.subplots(figsize=(18, 8.5))

        for key, d in selected.items():
            if key == global_ref_key:
                continue

            color, linestyle, alpha, linewidth = get_style(key)

            env_common = interp_to_common_time(
                d["t"],
                d["envelope"] * 1000,
                t_common,
            )

            absdiff = np.abs(env_common - env_ref_global)

            ax.plot(
                t_common,
                absdiff,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=linewidth,
            )

        ax.axvline(0, color="black", linestyle=":", linewidth=2.5)
        ax.axvspan(A0_START, A0_END, color="gray", alpha=0.20)

        ax.set_xlabel("Time after emitter start (µs)")
        ax.set_ylabel("|Envelope difference| (mV)")
        ax.set_title(f"A0 absolute envelope difference near {temp_title}, reference = cooling baseline")
        ax.grid(True, alpha=0.30)

        add_clean_legend(ax, loc="upper left", ncol=2)

        finish_figure(f"A0 absolute envelope difference cooling baseline near {target_temp:.0f} C")

        # =====================================================
        # PLOT 4: ENVELOPE DIFFERENCE TO AVERAGE INITIAL BASELINE
        # =====================================================
        fig, ax = plt.subplots(figsize=(18, 8.5))

        for key, d in selected.items():
            if key in ["B_C", "B_H"]:
                continue

            color, linestyle, alpha, linewidth = get_style(key)

            env_common = interp_to_common_time(
                d["t"],
                d["envelope"] * 1000,
                t_common,
            )

            diff = env_common - env_ref_avg

            ax.plot(
                t_common,
                diff,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=linewidth,
            )

        ax.axhline(0, color="black", linestyle="-", linewidth=1.5)
        ax.axvline(0, color="black", linestyle=":", linewidth=2.5)
        ax.axvspan(A0_START, A0_END, color="gray", alpha=0.20)

        ax.set_xlabel("Time after emitter start (µs)")
        ax.set_ylabel("Envelope difference (mV)")
        ax.set_title(f"A0 envelope difference near {temp_title}, reference = average baseline")
        ax.grid(True, alpha=0.30)

        add_clean_legend(ax, loc="upper left", ncol=2)

        finish_figure(f"A0 envelope difference average baseline near {target_temp:.0f} C")

        # =====================================================
        # PLOT 5: RECEIVER WAVEFORM COMPARISON
        # =====================================================
        fig, ax = plt.subplots(figsize=(18, 8.5))

        for key, d in selected.items():
            color, linestyle, alpha, linewidth = get_style(key)

            s_common = interp_to_common_time(
                d["t"],
                d["receiver"] * 1000,
                t_common,
            )

            ax.plot(
                t_common,
                s_common,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=linewidth,
            )

        ax.axvline(0, color="black", linestyle=":", linewidth=2.5)
        ax.axvspan(A0_START, A0_END, color="gray", alpha=0.20)

        ax.set_xlabel("Time after emitter start (µs)")
        ax.set_ylabel("Receiver (mV)")
        ax.set_title(f"A0 waveform comparison near {temp_title}, ice height h = 2 mm")
        ax.grid(True, alpha=0.30)

        add_clean_legend(ax, loc="upper left", ncol=2)

        finish_figure(f"A0 waveform comparison near {target_temp:.0f} C")

        # =====================================================
        # SUMMARY METRICS
        # =====================================================
        rows = []

        for key, d in selected.items():
            env_common = interp_to_common_time(
                d["t"],
                d["envelope"] * 1000,
                t_common,
            )

            a0_mask = (t_common >= A0_START) & (t_common <= A0_END)

            a0_max = np.max(env_common[a0_mask])
            a0_p95 = np.percentile(env_common[a0_mask], 95)
            a0_energy = np.sum(env_common[a0_mask] ** 2)
            a0_rms = np.sqrt(np.mean(env_common[a0_mask] ** 2))

            diff_global = env_common - env_ref_global
            diff_global_a0 = diff_global[a0_mask]

            diff_global_rms = np.sqrt(np.mean(diff_global_a0 ** 2))
            diff_global_area = np.sum(np.abs(diff_global_a0))

            diff_avg = env_common - env_ref_avg
            diff_avg_a0 = diff_avg[a0_mask]

            diff_avg_rms = np.sqrt(np.mean(diff_avg_a0 ** 2))
            diff_avg_area = np.sum(np.abs(diff_avg_a0))

            row = {
                "test_name": TEST_NAME,
                "target_temperature_C": target_temp,
                "dataset": key,
                "case": datasets[key]["case"],
                "condition": datasets[key]["condition"],
                "group": datasets[key]["group"],
                "sweep": datasets[key]["sweep"],
                "selected_temperature_C": d["temp"],
                "file": d["file"],

                "A0_start_us": A0_START,
                "A0_end_us": A0_END,

                "A0_max_mV": a0_max,
                "A0_p95_mV": a0_p95,
                "A0_energy_mV2": a0_energy,
                "A0_rms_mV": a0_rms,

                "diff_cooling_baseline_RMS_mV": diff_global_rms,
                "diff_cooling_baseline_area_mV": diff_global_area,

                "diff_avg_baseline_RMS_mV": diff_avg_rms,
                "diff_avg_baseline_area_mV": diff_avg_area,
            }

            rows.append(row)
            all_summary_rows.append(row)

        summary = pd.DataFrame(rows)

        print("\n===== SUMMARY METRICS FOR TARGET", target_temp, "°C =====")
        print(summary[[
            "dataset",
            "condition",
            "sweep",
            "selected_temperature_C",
            "A0_max_mV",
            "A0_rms_mV",
            "diff_cooling_baseline_RMS_mV",
            "diff_avg_baseline_RMS_mV",
        ]])

        # =====================================================
        # PLOT 6: BAR PLOT DIFFERENCE RMS
        # =====================================================
        fig, ax = plt.subplots(figsize=(18, 8.5))

        x = np.arange(len(summary))
        bar_colors = [condition_colors[c] for c in summary["condition"]]

        ax.bar(
            x,
            summary["diff_avg_baseline_RMS_mV"],
            color=bar_colors,
            alpha=0.85,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(summary["dataset"], rotation=35, ha="right")
        ax.set_ylabel("RMS envelope difference (mV)")
        ax.set_title(f"A0 envelope difference summary near {temp_title}, reference = average baseline")
        ax.grid(axis="y", alpha=0.30)

        bar_handles = [
            Line2D([0], [0], color="blue", lw=8, label="Baseline"),
            Line2D([0], [0], color="darkorange", lw=8, label="Ø1 cm ice"),
            Line2D([0], [0], color="green", lw=8, label="Ø2 cm ice"),
            Line2D([0], [0], color="purple", lw=8, label="Ø2 cm ice repeat"),
            Line2D([0], [0], color="red", lw=8, label="Ø4 cm ice"),
            Line2D([0], [0], color="black", lw=8, label="After removal"),
        ]

        ax.legend(
            handles=bar_handles,
            loc="upper left",
            fontsize=20,
            ncol=2,
            framealpha=0.92,
        )

        finish_figure(f"A0 envelope difference summary near {target_temp:.0f} C")

    all_summary = pd.DataFrame(all_summary_rows)

    if SAVE_SUMMARY:
        all_summary.to_csv(OUT_FILE, sep=";", decimal=",", index=False)

    print("\nSaved summary CSV:")
    print(OUT_FILE)

    print("\nTest name:")
    print(TEST_NAME)


if __name__ == "__main__":
    main()