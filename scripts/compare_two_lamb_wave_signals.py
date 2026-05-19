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
    """Find a data file either in data_rest/ or data_rest/data/."""
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
FILE_BASE = data_file("doubletape_baselineafter17_17_04_110kHz.csv")
FILE_WASH = data_file("doubletape_baseline17_17_04_110kHz.csv")

OUTPUT_CSV = RESULTS_DIR / "compare_two_lamb_wave_signals_metrics.csv"
OUTPUT_FIG = FIG_DIR / "compare_two_lamb_wave_signals.png"


# =========================
# SETTINGS
# =========================
A0_START = 90e-6
A0_END = 250e-6

SAVE_FIGURE = True
SHOW_FIGURE = False
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


def main():
    print("\nProject folder:")
    print(PROJECT_DIR)

    t_base, _, r_base = load_csv(FILE_BASE)
    t_wash, _, r_wash = load_csv(FILE_WASH)

    r_base = r_base * 1000.0
    r_wash = r_wash * 1000.0

    env_base = np.abs(hilbert(r_base))
    env_wash = np.abs(hilbert(r_wash))

    diff = r_wash - r_base

    mask_a0 = (t_base > A0_START) & (t_base < A0_END)

    corr_full, sdc_full, energy_full, rms_full = compute_metrics(r_base, r_wash)
    corr_a0, sdc_a0, energy_a0, rms_a0 = compute_metrics(
        r_base[mask_a0],
        r_wash[mask_a0],
    )

    rows = [
        {
            "region": "full_signal",
            "correlation": corr_full,
            "SDC": sdc_full,
            "energy_change": energy_full,
            "RMS_difference_mV": rms_full,
            "reference_file": FILE_BASE.name,
            "comparison_file": FILE_WASH.name,
        },
        {
            "region": "A0_window",
            "correlation": corr_a0,
            "SDC": sdc_a0,
            "energy_change": energy_a0,
            "RMS_difference_mV": rms_a0,
            "reference_file": FILE_BASE.name,
            "comparison_file": FILE_WASH.name,
        },
    ]

    results = pd.DataFrame(rows)

    print("\n--- METRICS ---")
    print(results)

    if SAVE_RESULTS:
        results.to_csv(OUTPUT_CSV, sep=";", decimal=",", index=False)
        print("\nSaved metrics:")
        print(OUTPUT_CSV)

    recv_lim = max(np.max(np.abs(r_base)), np.max(np.abs(r_wash))) * 1.2

    fig = plt.figure(figsize=(15, 16))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_base * 1e6, r_base, color="black", linewidth=2.5, label="Reference")
    ax1.plot(t_wash * 1e6, r_wash, color="red", linestyle="--", linewidth=2.5, label="Comparison")
    ax1.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10, label="A0 window")
    ax1.set_ylabel("Receiver (mV)")
    ax1.set_title("Lamb-wave response: reference vs comparison")
    ax1.set_xlim(-80, 450)
    ax1.set_ylim(-recv_lim, recv_lim)
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t_base * 1e6, diff, color="purple", linewidth=2.5)
    ax2.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10)
    ax2.set_ylabel("Difference (mV)")
    ax2.set_title("Difference signal")
    ax2.set_xlim(-80, 450)
    ax2.grid(True)

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t_base * 1e6, env_base, color="black", linewidth=2.5, label="Reference envelope")
    ax3.plot(t_wash * 1e6, env_wash, color="red", linestyle="--", linewidth=2.5, label="Comparison envelope")
    ax3.axvspan(A0_START * 1e6, A0_END * 1e6, color="purple", alpha=0.10)
    ax3.set_ylabel("Envelope (mV)")
    ax3.set_xlabel("Time relative to emitter (µs)")
    ax3.set_title("Envelope comparison")
    ax3.set_xlim(-80, 450)
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()

    if SAVE_FIGURE:
        fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
        print("Saved figure:")
        print(OUTPUT_FIG)

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
