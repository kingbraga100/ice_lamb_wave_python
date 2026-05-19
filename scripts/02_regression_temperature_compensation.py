# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures" / "ml_temperature_compensation"

RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = RESULTS_DIR / "ml_fixed_reference_metrics_may17.csv"

OUTPUT_CSV = RESULTS_DIR / "ml_temperature_compensated_metrics_may17.csv"
SUMMARY_CSV = RESULTS_DIR / "ml_temperature_compensation_summary_may17.csv"


# =========================================================
# SETTINGS
# =========================================================
POLY_DEGREE = 2
RIDGE_ALPHA = 1e-8

N_ENSEMBLE = 300
TRAIN_FRACTION = 0.70
RANDOM_SEED = 42

SAVE_FIGURES = True
SHOW_FIGURES = False
SAVE_RESULTS = True

TRAIN_CONDITION = "Baseline"

METRICS = [
    "A0_p95_mV",
    "Energy_wave_mV2",
    "SDC_wave_fixed_ref",
    "Corr_wave_fixed_ref",
]

# Optional envelope metrics if you want more figures:
# METRICS = [
#     "A0_p95_mV",
#     "Energy_wave_mV2",
#     "SDC_wave_fixed_ref",
#     "Corr_wave_fixed_ref",
#     "SDC_env_fixed_ref",
#     "Corr_env_fixed_ref",
# ]


# =========================================================
# STYLE
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

condition_order = [
    "Baseline",
    "Ø1 cm ice",
    "Ø2 cm ice",
    "Ø2 cm ice repeat",
    "Ø4 cm ice",
    "After removal",
]

metric_labels = {
    "A0_p95_mV": "A0 envelope p95 (mV)",
    "A0_max_mV": "A0 maximum envelope (mV)",
    "Energy_wave_mV2": "A0 waveform energy (mV²)",
    "Energy_env_mV2": "A0 envelope energy (mV²)",
    "SDC_wave_fixed_ref": "A0 waveform SDC",
    "SDC_env_fixed_ref": "A0 envelope SDC",
    "Corr_wave_fixed_ref": "A0 waveform correlation",
    "Corr_env_fixed_ref": "A0 envelope correlation",
}

rmse_labels = {
    "A0_p95_mV": "RMSE (mV)",
    "A0_max_mV": "RMSE (mV)",
    "Energy_wave_mV2": "RMSE (mV²)",
    "Energy_env_mV2": "RMSE (mV²)",
    "SDC_wave_fixed_ref": "RMSE",
    "SDC_env_fixed_ref": "RMSE",
    "Corr_wave_fixed_ref": "RMSE",
    "Corr_env_fixed_ref": "RMSE",
}

plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 20,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})


# =========================================================
# REGRESSION FUNCTIONS
# =========================================================
def scale_temperature(T, Tmin, Tmax):
    T = np.asarray(T, dtype=float)

    if np.isclose(Tmax, Tmin):
        return np.zeros_like(T)

    return 2.0 * (T - Tmin) / (Tmax - Tmin) - 1.0


def build_poly_matrix(T_scaled, degree):
    X = np.zeros((len(T_scaled), degree + 1))

    for j in range(degree + 1):
        X[:, j] = T_scaled ** j

    return X


def fit_polynomial(T, y, degree, ridge_alpha=0.0, Tmin=None, Tmax=None):
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(T) & np.isfinite(y)
    T = T[good]
    y = y[good]

    if len(T) < degree + 2:
        raise ValueError("Not enough points for regression.")

    if Tmin is None:
        Tmin = np.min(T)

    if Tmax is None:
        Tmax = np.max(T)

    T_scaled = scale_temperature(T, Tmin, Tmax)
    X = build_poly_matrix(T_scaled, degree)

    H = X.T @ X
    b = X.T @ y

    R = np.eye(H.shape[0])
    R[0, 0] = 0.0

    H_reg = H + ridge_alpha * R
    w = np.linalg.solve(H_reg, b)

    return {
        "weights": w,
        "degree": degree,
        "Tmin": Tmin,
        "Tmax": Tmax,
        "cond_H": np.linalg.cond(H),
        "cond_Hreg": np.linalg.cond(H_reg),
    }


def predict_polynomial(T, model):
    T = np.asarray(T, dtype=float)

    T_scaled = scale_temperature(T, model["Tmin"], model["Tmax"])
    X = build_poly_matrix(T_scaled, model["degree"])

    return X @ model["weights"]


def mse(y, y_pred):
    good = np.isfinite(y) & np.isfinite(y_pred)

    if np.sum(good) == 0:
        return np.nan

    return np.mean((y[good] - y_pred[good]) ** 2)


def monte_carlo_regression(T, y, T_grid):
    rng = np.random.default_rng(RANDOM_SEED)

    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(T) & np.isfinite(y)
    T = T[good]
    y = y[good]

    n = len(T)

    if n < POLY_DEGREE + 4:
        raise ValueError("Too few points for Monte Carlo regression.")

    Tmin = np.min(T)
    Tmax = np.max(T)

    n_train = int(round(TRAIN_FRACTION * n))
    n_train = max(POLY_DEGREE + 2, n_train)
    n_train = min(n_train, n - 1)

    Ji = np.zeros(N_ENSEMBLE)
    Jo = np.zeros(N_ENSEMBLE)
    pred_pop = np.zeros((len(T_grid), N_ENSEMBLE))

    for k in range(N_ENSEMBLE):
        idx = rng.permutation(n)

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        T_train = T[train_idx]
        y_train = y[train_idx]

        T_test = T[test_idx]
        y_test = y[test_idx]

        model = fit_polynomial(
            T_train,
            y_train,
            degree=POLY_DEGREE,
            ridge_alpha=RIDGE_ALPHA,
            Tmin=Tmin,
            Tmax=Tmax,
        )

        y_train_pred = predict_polynomial(T_train, model)
        y_test_pred = predict_polynomial(T_test, model)

        Ji[k] = mse(y_train, y_train_pred)
        Jo[k] = mse(y_test, y_test_pred)

        pred_pop[:, k] = predict_polynomial(T_grid, model)

    full_model = fit_polynomial(
        T,
        y,
        degree=POLY_DEGREE,
        ridge_alpha=RIDGE_ALPHA,
        Tmin=Tmin,
        Tmax=Tmax,
    )

    return {
        "Ji": Ji,
        "Jo": Jo,
        "RMSE_i": np.sqrt(Ji),
        "RMSE_o": np.sqrt(Jo),
        "pred_mean": np.mean(pred_pop, axis=1),
        "pred_std": np.std(pred_pop, axis=1),
        "full_model": full_model,
    }


# =========================================================
# PLOTTING FUNCTIONS
# =========================================================
def safe_name(text):
    text = str(text).replace("Ø", "diam")
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)
    return text.strip("_")


def scatter_conditions(ax, df, ycol):
    for cond in condition_order:
        sub = df[df["condition"] == cond]

        if len(sub) == 0:
            continue

        marker = condition_markers[cond]
        color = condition_colors[cond]

        ax.scatter(
            sub["Temperature_C"],
            sub[ycol],
            s=70,
            marker=marker,
            color=color,
            alpha=0.85,
            edgecolors="black" if marker != "x" else None,
            linewidths=0.35,
            zorder=3,
        )


def add_legend(ax):
    handles = []

    for cond in condition_order:
        handles.append(
            Line2D(
                [0], [0],
                marker=condition_markers[cond],
                linestyle="None",
                color=condition_colors[cond],
                markerfacecolor=condition_colors[cond] if condition_markers[cond] != "x" else "none",
                markeredgecolor=condition_colors[cond],
                markersize=8,
                label=cond,
            )
        )

    handles.append(
        Line2D(
            [0], [0],
            color="black",
            linewidth=2.5,
            label="Baseline regression",
        )
    )

    handles.append(
        Line2D(
            [0], [0],
            color="gray",
            linewidth=7,
            alpha=0.25,
            label="95% band",
        )
    )

    ax.legend(
        handles=handles,
        ncol=2,
        framealpha=0.92,
        loc="best",
    )


def plot_metric_and_residual(df, metric, T_grid, mc, residual_col):
    ylabel = metric_labels.get(metric, metric)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    ax0, ax1 = axes

    scatter_conditions(ax0, df, metric)

    ax0.plot(
        T_grid,
        mc["pred_mean"],
        color="black",
        linewidth=2.8,
    )

    ax0.fill_between(
        T_grid,
        mc["pred_mean"] - 1.96 * mc["pred_std"],
        mc["pred_mean"] + 1.96 * mc["pred_std"],
        color="gray",
        alpha=0.25,
    )

    ax0.set_xlabel("Temperature (°C)")
    ax0.set_ylabel(ylabel)
    ax0.set_title("Baseline temperature model")
    ax0.grid(True, alpha=0.28)
    add_legend(ax0)

    scatter_conditions(ax1, df, residual_col)

    ax1.axhline(
        0,
        color="black",
        linewidth=2.0,
        alpha=0.8,
    )

    ax1.fill_between(
        T_grid,
        -1.96 * mc["pred_std"],
        +1.96 * mc["pred_std"],
        color="gray",
        alpha=0.25,
    )

    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel(f"Residual of {ylabel}")
    ax1.set_title("Temperature-corrected residual")
    ax1.grid(True, alpha=0.28)

    fig.suptitle(
        f"{ylabel} — baseline-only polynomial regression, degree {POLY_DEGREE}",
        fontsize=21,
        y=1.02,
    )

    fig.tight_layout()

    if SAVE_FIGURES:
        path = FIG_DIR / f"{safe_name('compensation_' + metric)}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def plot_rmse_histogram(metric, mc):
    ylabel = metric_labels.get(metric, metric)

    fig, ax = plt.subplots(figsize=(8.5, 6))

    ax.hist(
        mc["RMSE_i"],
        bins=35,
        alpha=0.65,
        label=r"$J_i$ train RMSE",
    )

    ax.hist(
        mc["RMSE_o"],
        bins=35,
        alpha=0.65,
        label=r"$J_o$ test RMSE",
    )

    ax.set_xlabel(rmse_labels.get(metric, "RMSE"))
    ax.set_ylabel("Count")
    ax.set_title(f"Monte Carlo train/test error: {ylabel}")
    ax.grid(True, alpha=0.28)
    ax.legend()

    fig.tight_layout()

    if SAVE_FIGURES:
        path = FIG_DIR / f"{safe_name('rmse_histogram_' + metric)}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# MAIN
# =========================================================
def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find input CSV:\n{INPUT_CSV}\n\n"
            "Run 01_export_fixed_reference_metrics_for_ml.py first."
        )

    df = pd.read_csv(INPUT_CSV, sep=";", decimal=",")

    df["Temperature_C"] = pd.to_numeric(df["Temperature_C"], errors="coerce")

    for metric in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df.dropna(subset=["Temperature_C", "condition"]).copy()

    print("\nProject folder:")
    print(PROJECT_DIR)

    print("\nLoaded:")
    print(INPUT_CSV)

    print("\nCounts:")
    print(df.groupby("condition")["file"].count())

    baseline = df[df["condition"] == TRAIN_CONDITION].copy()

    if len(baseline) < POLY_DEGREE + 4:
        raise RuntimeError("Too few baseline points.")

    T_all = df["Temperature_C"].to_numpy()
    T_grid = np.linspace(np.nanmin(T_all), np.nanmax(T_all), 250)

    corrected_df = df.copy()
    summary_rows = []

    for metric in METRICS:
        print("\n" + "=" * 70)
        print(f"Metric: {metric}")

        if metric not in df.columns:
            print("Skipping: column not found.")
            continue

        train = baseline[["Temperature_C", metric]].dropna()

        if len(train) < POLY_DEGREE + 4:
            print("Skipping: too few baseline points.")
            continue

        mc = monte_carlo_regression(
            train["Temperature_C"].to_numpy(),
            train[metric].to_numpy(),
            T_grid,
        )

        pred_col = f"{metric}_baseline_prediction"
        residual_col = f"{metric}_temperature_residual"

        corrected_df[pred_col] = predict_polynomial(
            corrected_df["Temperature_C"].to_numpy(),
            mc["full_model"],
        )

        corrected_df[residual_col] = corrected_df[metric] - corrected_df[pred_col]

        print(f"Condition number H:    {mc['full_model']['cond_H']:.3e}")
        print(f"Condition number Hreg: {mc['full_model']['cond_Hreg']:.3e}")
        print(f"Mean Ji RMSE:          {np.nanmean(mc['RMSE_i']):.4g}")
        print(f"Mean Jo RMSE:          {np.nanmean(mc['RMSE_o']):.4g}")

        summary_rows.append({
            "metric": metric,
            "degree": POLY_DEGREE,
            "mean_Ji_RMSE": np.nanmean(mc["RMSE_i"]),
            "std_Ji_RMSE": np.nanstd(mc["RMSE_i"]),
            "mean_Jo_RMSE": np.nanmean(mc["RMSE_o"]),
            "std_Jo_RMSE": np.nanstd(mc["RMSE_o"]),
            "cond_H": mc["full_model"]["cond_H"],
            "cond_Hreg": mc["full_model"]["cond_Hreg"],
            "weights": mc["full_model"]["weights"].tolist(),
        })

        plot_metric_and_residual(corrected_df, metric, T_grid, mc, residual_col)
        plot_rmse_histogram(metric, mc)

    if SAVE_RESULTS:
        corrected_df.to_csv(OUTPUT_CSV, sep=";", decimal=",", index=False)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_CSV, sep=";", decimal=",", index=False)

        print("\nSaved:")
        print(OUTPUT_CSV)
        print(SUMMARY_CSV)

    print("\nDone.")


if __name__ == "__main__":
    main()