# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert


# =========================
# PATHS
# =========================
def find_project_dir():
    here = Path(__file__).resolve()

    for parent in [here.parent] + list(here.parents):
        if (parent / "data_rest").exists():
            return parent

    raise RuntimeError(
        "Could not find the project folder. Make sure this script is inside the "
        "repository and that 'data_rest' exists in the main project folder."
    )


PROJECT_DIR = find_project_dir()
DATA_DIR = PROJECT_DIR / "data_rest"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures" / "validation"

RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def data_file(name):
    candidates = [
        DATA_DIR / name,
        DATA_DIR / "data" / name,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {name}. Looked in:\n"
        f"  {DATA_DIR}\n"
        f"  {DATA_DIR / 'data'}"
    )


# =========================
# FILES
# =========================
files = {
    "Baseline 1": data_file("doubletape_baseline18_24_03_110kHz.csv"),
    "Washer": data_file("doubletape_washer18_24_03_110kHz.csv"),
    "Small washer": data_file("doubletape_smallwasher18_24_03_110kHz.csv"),
    "Glaze": data_file("doubletape_glaze18_24_03_110kHz.csv"),
    "Small glaze": data_file("doubletape_smallglaze18_24_03_110kHz.csv"),
    "Baseline 2": data_file("doubletape_baselineafter18_24_03_110kHz.csv"),
    "Side washer": data_file("doubletape_sidewasher18_24_03_110kHz.csv"),
    "Side small washer": data_file("doubletape_sidesmallwasher18_24_03_110kHz.csv"),
    "Side glaze": data_file("doubletape_sideglaze18_24_03_110kHz.csv"),
    "Side small glaze": data_file("doubletape_sidesmallglaze18_24_03_110kHz.csv"),
    "Baseline 3": data_file("doubletape_baselinefinal18_24_03_110kHz.csv"),
}

OUTPUT_CSV = RESULTS_DIR / "compare_washer_glaze_tests_metrics.csv"


# =========================
# SETTINGS
# =========================
A0_START = 90e-6
A0_END = 250e-6

PLOT_START = 0e-6
PLOT_END = 250e-6

SAVE_FIGURES = True
SHOW_FIGURES = False
SAVE_RESULTS = True


# =========================
# STYLE
# =========================
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 28,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})


# =========================
# FUNCTIONS
# =========================
def safe_name(text):
    text = str(text).replace("Ø", "diam")
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)
    return text.strip("_")


def to_float_eu(x):
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return float(str(x).replace(",", "."))


def load_csv(filename):
    data = np.genfromtxt(
        filename,
        delimiter=";",
        skip_header=1,
        converters={
            0: to_float_eu,
            1: to_float_eu,
            2: to_float_eu,
            3: to_float_eu,
        }
    )

    t = data[:, 0] * 1e-6
    emitter = data[:, 1]
    receiver = data[:, 2]

    return t, emitter, receiver


def compute_metrics(reference, current):
    corr = np.corrcoef(reference, current)[0, 1]
    sdc = np.sum((reference - current) ** 2) / np.sum(reference ** 2)

    e_ref = np.sum(reference ** 2)
    e_cur = np.sum(current ** 2)
    energy_change = (e_ref - e_cur) / e_ref

    rms = np.sqrt(np.mean((reference - current) ** 2))

    return corr, sdc, energy_change, rms


def save_or_show(fig, name):
    fig.tight_layout()

    if SAVE_FIGURES:
        path = FIG_DIR / f"{safe_name(name)}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# =========================
# MAIN
# =========================
def main():
    print("\nProject folder:")
    print(PROJECT_DIR)

    data = {}
    env = {}

    for label, path in files.items():
        t, _, receiver = load_csv(path)
        data[label] = {"t": t, "r": receiver}
        env[label] = np.abs(hilbert(receiver))

    t_ref = data["Baseline 1"]["t"]
    mask_a0 = (t_ref >= A0_START) & (t_ref <= A0_END)
    mask_plot = (t_ref >= PLOT_START) & (t_ref <= PLOT_END)

    comparisons = [
        ("Baseline 1", "Washer"),
        ("Baseline 1", "Small washer"),
        ("Baseline 1", "Glaze"),
        ("Baseline 1", "Small glaze"),
        ("Baseline 2", "Side washer"),
        ("Baseline 2", "Side small washer"),
        ("Baseline 2", "Side glaze"),
        ("Baseline 2", "Side small glaze"),
    ]

    baseline_repeatability = [
        ("Baseline 1", "Baseline 2"),
        ("Baseline 2", "Baseline 3"),
    ]

    rows = []

    print("\nA0 region metrics")

    for ref_label, cur_label in comparisons + baseline_repeatability:
        ref = data[ref_label]["r"]
        cur = data[cur_label]["r"]

        corr, sdc, energy, rms = compute_metrics(ref[mask_a0], cur[mask_a0])

        rows.append({
            "reference": ref_label,
            "comparison": cur_label,
            "correlation": corr,
            "SDC": sdc,
            "energy_change": energy,
            "RMS_difference_mV": rms,
        })

        print(f"{ref_label:12s} vs {cur_label:20s} | Corr={corr: .4f} | SDC={sdc: .4f} | RMS={rms: .3f} mV")

    results = pd.DataFrame(rows)

    if SAVE_RESULTS:
        results.to_csv(OUTPUT_CSV, sep=";", decimal=",", index=False)
        print("\nSaved metrics:")
        print(OUTPUT_CSV)

    all_signals = [d["r"] for d in data.values()]
    all_envs = list(env.values())

    all_diffs = []
    for ref_label, cur_label in comparisons + baseline_repeatability:
        all_diffs.append(data[cur_label]["r"] - data[ref_label]["r"])

    recv_lim = max(np.max(np.abs(sig[mask_plot])) for sig in all_signals) * 1.2
    env_lim = max(np.max(e[mask_plot]) for e in all_envs) * 1.2
    diff_lim = max(np.max(np.abs(d[mask_plot])) for d in all_diffs) * 1.2

    # Figure 1: first position
    fig = plt.figure(figsize=(16, 18))

    ax1 = plt.subplot(3, 1, 1)
    for label in ["Baseline 1", "Washer", "Small washer"]:
        ax1.plot(data[label]["t"][mask_plot] * 1e6, data[label]["r"][mask_plot], linewidth=2.2, label=label)
    ax1.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax1.set_ylabel("Receiver (mV)")
    ax1.set_title("Baseline 1 vs washer materials")
    ax1.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax1.set_ylim(-recv_lim, recv_lim)
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    for label in ["Baseline 1", "Glaze", "Small glaze"]:
        ax2.plot(data[label]["t"][mask_plot] * 1e6, data[label]["r"][mask_plot], linewidth=2.2, label=label)
    ax2.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax2.set_ylabel("Receiver (mV)")
    ax2.set_title("Baseline 1 vs glaze materials")
    ax2.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax2.set_ylim(-recv_lim, recv_lim)
    ax2.grid(True)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3)
    for label in ["Baseline 1", "Baseline 2", "Baseline 3"]:
        ax3.plot(data[label]["t"][mask_plot] * 1e6, data[label]["r"][mask_plot], linewidth=2.2, label=label)
    ax3.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax3.set_ylabel("Receiver (mV)")
    ax3.set_xlabel("Time relative to emitter (µs)")
    ax3.set_title("Baseline repeatability")
    ax3.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax3.set_ylim(-recv_lim, recv_lim)
    ax3.grid(True)
    ax3.legend()

    save_or_show(fig, "washer_glaze_first_position")

    # Figure 2: side position
    fig = plt.figure(figsize=(16, 18))

    ax1 = plt.subplot(3, 1, 1)
    for label in ["Baseline 2", "Side washer", "Side small washer"]:
        ax1.plot(data[label]["t"][mask_plot] * 1e6, data[label]["r"][mask_plot], linewidth=2.2, label=label)
    ax1.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax1.set_ylabel("Receiver (mV)")
    ax1.set_title("Baseline 2 vs side washer materials")
    ax1.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax1.set_ylim(-recv_lim, recv_lim)
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    for label in ["Baseline 2", "Side glaze", "Side small glaze"]:
        ax2.plot(data[label]["t"][mask_plot] * 1e6, data[label]["r"][mask_plot], linewidth=2.2, label=label)
    ax2.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax2.set_ylabel("Receiver (mV)")
    ax2.set_title("Baseline 2 vs side glaze materials")
    ax2.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax2.set_ylim(-recv_lim, recv_lim)
    ax2.grid(True)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3)
    for label in ["Baseline 1", "Baseline 2", "Baseline 3"]:
        ax3.plot(data[label]["t"][mask_plot] * 1e6, data[label]["r"][mask_plot], linewidth=2.2, label=label)
    ax3.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax3.set_ylabel("Receiver (mV)")
    ax3.set_xlabel("Time relative to emitter (µs)")
    ax3.set_title("All baselines")
    ax3.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax3.set_ylim(-recv_lim, recv_lim)
    ax3.grid(True)
    ax3.legend()

    save_or_show(fig, "washer_glaze_side_position")

    # Figure 3: difference signals
    fig = plt.figure(figsize=(16, 24))

    diff_titles = [
        ("Washer − Baseline 1", data["Washer"]["r"] - data["Baseline 1"]["r"]),
        ("Small washer − Baseline 1", data["Small washer"]["r"] - data["Baseline 1"]["r"]),
        ("Glaze − Baseline 1", data["Glaze"]["r"] - data["Baseline 1"]["r"]),
        ("Small glaze − Baseline 1", data["Small glaze"]["r"] - data["Baseline 1"]["r"]),
        ("Side washer − Baseline 2", data["Side washer"]["r"] - data["Baseline 2"]["r"]),
        ("Side small washer − Baseline 2", data["Side small washer"]["r"] - data["Baseline 2"]["r"]),
        ("Side glaze − Baseline 2", data["Side glaze"]["r"] - data["Baseline 2"]["r"]),
        ("Side small glaze − Baseline 2", data["Side small glaze"]["r"] - data["Baseline 2"]["r"]),
    ]

    for i, (title, diff_sig) in enumerate(diff_titles, start=1):
        ax = plt.subplot(8, 1, i)
        ax.plot(t_ref[mask_plot] * 1e6, diff_sig[mask_plot], color="purple", linewidth=2.2)
        ax.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10)
        ax.set_ylabel("Diff. (mV)")
        ax.set_title(title)
        ax.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
        ax.set_ylim(-diff_lim, diff_lim)
        ax.grid(True)
        if i == 8:
            ax.set_xlabel("Time relative to emitter (µs)")

    save_or_show(fig, "washer_glaze_difference_signals")

    # Figure 4: envelopes
    fig = plt.figure(figsize=(16, 18))

    ax1 = plt.subplot(3, 1, 1)
    for label in ["Baseline 1", "Washer", "Small washer", "Glaze", "Small glaze"]:
        ax1.plot(data[label]["t"][mask_plot] * 1e6, env[label][mask_plot], linewidth=2.2, label=label)
    ax1.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax1.set_ylabel("Envelope (mV)")
    ax1.set_title("Envelope: first position")
    ax1.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax1.set_ylim(0, env_lim)
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    for label in ["Baseline 2", "Side washer", "Side small washer", "Side glaze", "Side small glaze"]:
        ax2.plot(data[label]["t"][mask_plot] * 1e6, env[label][mask_plot], linewidth=2.2, label=label)
    ax2.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax2.set_ylabel("Envelope (mV)")
    ax2.set_title("Envelope: side position")
    ax2.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax2.set_ylim(0, env_lim)
    ax2.grid(True)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3)
    for label in ["Baseline 1", "Baseline 2", "Baseline 3"]:
        ax3.plot(data[label]["t"][mask_plot] * 1e6, env[label][mask_plot], linewidth=2.2, label=label)
    ax3.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax3.set_ylabel("Envelope (mV)")
    ax3.set_xlabel("Time relative to emitter (µs)")
    ax3.set_title("Baseline envelopes")
    ax3.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax3.set_ylim(0, env_lim)
    ax3.grid(True)
    ax3.legend()

    save_or_show(fig, "washer_glaze_envelopes")

    print("\nDone.")


if __name__ == "__main__":
    main()
