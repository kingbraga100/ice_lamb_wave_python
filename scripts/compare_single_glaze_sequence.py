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
    "Baseline 1": data_file("doubletape_baseline20_17_04_110kHz.csv"),
    "Small glaze": data_file("doubletape_smallglaze20_17_04_110kHz.csv"),
    "Baseline 2": data_file("doubletape_baselineafter20_17_04_110kHz.csv"),
    "Side glaze": data_file("doubletape_sideglaze20_17_04_110kHz.csv"),
    "Baseline 3": data_file("doubletape_baselinefinal20_17_04_110kHz.csv"),
}

OUTPUT_CSV = RESULTS_DIR / "compare_single_glaze_sequence_metrics.csv"


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
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
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

    pairs_main = [
        ("Baseline 1", "Small glaze"),
        ("Small glaze", "Baseline 2"),
        ("Baseline 2", "Side glaze"),
        ("Side glaze", "Baseline 3"),
    ]

    pairs_repeatability = [
        ("Baseline 1", "Baseline 2"),
        ("Baseline 2", "Baseline 3"),
    ]

    rows = []

    print("\nA0 region metrics")

    for ref_label, cur_label in pairs_main + pairs_repeatability:
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

        print(f"{ref_label:12s} vs {cur_label:12s} | Corr={corr: .4f} | SDC={sdc: .4f} | RMS={rms: .3f} mV")

    results = pd.DataFrame(rows)

    if SAVE_RESULTS:
        results.to_csv(OUTPUT_CSV, sep=";", decimal=",", index=False)
        print("\nSaved metrics:")
        print(OUTPUT_CSV)

    recv_lim = max(np.max(np.abs(d["r"][mask_plot])) for d in data.values()) * 1.2

    diff_signals = [
        data["Small glaze"]["r"] - data["Baseline 1"]["r"],
        data["Baseline 2"]["r"] - data["Small glaze"]["r"],
        data["Side glaze"]["r"] - data["Baseline 2"]["r"],
        data["Baseline 3"]["r"] - data["Side glaze"]["r"],
        data["Baseline 2"]["r"] - data["Baseline 1"]["r"],
        data["Baseline 3"]["r"] - data["Baseline 2"]["r"],
    ]

    diff_lim = max(np.max(np.abs(d[mask_plot])) for d in diff_signals) * 1.2
    env_lim = max(np.max(e[mask_plot]) for e in env.values()) * 1.2

    # Figure 1: raw signals
    fig = plt.figure(figsize=(15, 18))

    ax1 = plt.subplot(3, 1, 1)
    for label, color, ls in [("Baseline 1", "black", "-"), ("Small glaze", "red", "--")]:
        ax1.plot(data[label]["t"][mask_plot] * 1e6, data[label]["r"][mask_plot], color=color, linestyle=ls, linewidth=2.2, label=label)
    ax1.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax1.set_ylabel("Receiver (mV)")
    ax1.set_title("Baseline 1 vs small glaze")
    ax1.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax1.set_ylim(-recv_lim, recv_lim)
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    for label, color, ls in [("Baseline 2", "black", "-"), ("Side glaze", "blue", "--")]:
        ax2.plot(data[label]["t"][mask_plot] * 1e6, data[label]["r"][mask_plot], color=color, linestyle=ls, linewidth=2.2, label=label)
    ax2.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax2.set_ylabel("Receiver (mV)")
    ax2.set_title("Baseline 2 vs side glaze")
    ax2.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax2.set_ylim(-recv_lim, recv_lim)
    ax2.grid(True)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(data["Baseline 3"]["t"][mask_plot] * 1e6, data["Baseline 3"]["r"][mask_plot], color="green", linewidth=2.2, label="Baseline 3")
    ax3.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax3.set_ylabel("Receiver (mV)")
    ax3.set_xlabel("Time relative to emitter (µs)")
    ax3.set_title("Final baseline")
    ax3.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax3.set_ylim(-recv_lim, recv_lim)
    ax3.grid(True)
    ax3.legend()

    save_or_show(fig, "single_glaze_sequence_raw_signals")

    # Figure 2: differences
    fig = plt.figure(figsize=(15, 22))

    diff_titles = [
        ("Small glaze − Baseline 1", data["Small glaze"]["r"] - data["Baseline 1"]["r"], "purple"),
        ("Baseline 2 − Small glaze", data["Baseline 2"]["r"] - data["Small glaze"]["r"], "purple"),
        ("Side glaze − Baseline 2", data["Side glaze"]["r"] - data["Baseline 2"]["r"], "purple"),
        ("Baseline 3 − Side glaze", data["Baseline 3"]["r"] - data["Side glaze"]["r"], "purple"),
        ("Baseline 2 − Baseline 1", data["Baseline 2"]["r"] - data["Baseline 1"]["r"], "darkgreen"),
        ("Baseline 3 − Baseline 2", data["Baseline 3"]["r"] - data["Baseline 2"]["r"], "darkgreen"),
    ]

    for i, (title, diff, color) in enumerate(diff_titles, start=1):
        ax = plt.subplot(6, 1, i)
        ax.plot(t_ref[mask_plot] * 1e6, diff[mask_plot], color=color, linewidth=2.2)
        ax.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10)
        ax.set_ylabel("Diff. (mV)")
        ax.set_title(title)
        ax.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
        ax.set_ylim(-diff_lim, diff_lim)
        ax.grid(True)
        if i == 6:
            ax.set_xlabel("Time relative to emitter (µs)")

    save_or_show(fig, "single_glaze_sequence_difference_signals")

    # Figure 3: envelopes
    fig = plt.figure(figsize=(15, 18))

    ax1 = plt.subplot(3, 1, 1)
    for label, color, ls in [("Baseline 1", "black", "-"), ("Small glaze", "red", "--")]:
        ax1.plot(data[label]["t"][mask_plot] * 1e6, env[label][mask_plot], color=color, linestyle=ls, linewidth=2.2, label=f"{label} envelope")
    ax1.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax1.set_ylabel("Envelope (mV)")
    ax1.set_title("Envelope: Baseline 1 vs small glaze")
    ax1.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax1.set_ylim(0, env_lim)
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    for label, color, ls in [("Baseline 2", "black", "-"), ("Side glaze", "blue", "--")]:
        ax2.plot(data[label]["t"][mask_plot] * 1e6, env[label][mask_plot], color=color, linestyle=ls, linewidth=2.2, label=f"{label} envelope")
    ax2.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax2.set_ylabel("Envelope (mV)")
    ax2.set_title("Envelope: Baseline 2 vs side glaze")
    ax2.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax2.set_ylim(0, env_lim)
    ax2.grid(True)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(data["Baseline 3"]["t"][mask_plot] * 1e6, env["Baseline 3"][mask_plot], color="green", linewidth=2.2, label="Baseline 3 envelope")
    ax3.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax3.set_ylabel("Envelope (mV)")
    ax3.set_xlabel("Time relative to emitter (µs)")
    ax3.set_title("Envelope: final baseline")
    ax3.set_xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    ax3.set_ylim(0, env_lim)
    ax3.grid(True)
    ax3.legend()

    save_or_show(fig, "single_glaze_sequence_envelopes")

    # Figure 4: focused comparison
    fig = plt.figure(figsize=(13, 6))
    plt.plot(t_ref[mask_plot] * 1e6, data["Baseline 1"]["r"][mask_plot], color="black", linewidth=2.5, label="Baseline 1")
    plt.plot(t_ref[mask_plot] * 1e6, data["Small glaze"]["r"][mask_plot], color="red", linestyle="--", linewidth=2.5, label="Small glaze")
    plt.plot(t_ref[mask_plot] * 1e6, data["Baseline 2"]["r"][mask_plot], color="blue", linestyle="--", linewidth=2.5, label="Baseline 2")
    plt.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    plt.title("Baseline 1 vs small glaze vs baseline 2")
    plt.ylabel("Receiver (mV)")
    plt.xlabel("Time relative to emitter (µs)")
    plt.xlim(PLOT_START * 1e6, PLOT_END * 1e6)
    plt.ylim(-recv_lim, recv_lim)
    plt.grid(True)
    plt.legend()
    save_or_show(fig, "single_glaze_sequence_focused_raw")

    print("\nDone.")


if __name__ == "__main__":
    main()
