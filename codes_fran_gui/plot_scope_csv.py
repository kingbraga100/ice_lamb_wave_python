import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
csv_file = "scope_waveform_voltage_current_phase.csv"
try:
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
    print("First 5 rows of data:")
    print(df.head())
    print("\nData summary:")
    print(df.describe())
except FileNotFoundError:
    print(f"Error: CSV file '{csv_file}' not found.")
    exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Check if required columns exist
required_columns = [
    "Time (s)", "Frequency (Hz)", "CH1 Voltage (V)", "CH2 Current Filtered (mA)",
    "Phase Live Time (s)", "Phase(CH2-CH1) (deg)",
    "|Y| (S)", "|Y| (mS)", "|Z| (ohm)", "G (S)", "B (S)"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns in CSV: {missing_columns}")
    exit(1)

# Plot each column in a separate figure
for col in required_columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    if col in ["Time (s)", "Phase Live Time (s)"]:
        # For time columns, plot as line against index
        ax.plot(df.index, df[col], label=col, color='blue')
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        ax.set_title(f"{col}")
    elif col == "Frequency (Hz)":
        # Frequency vs index or time
        ax.plot(df.index, df[col], label=col, color='blue')
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        ax.set_title(f"{col}")
    else:
        # For metrics, plot against frequency if available, else time
        x_col = "Frequency (Hz)" if "Frequency (Hz)" in df.columns and df["Frequency (Hz)"].notna().any() else "Time (s)"
        ax.plot(df[x_col], df[col], label=col, color='blue')
        ax.set_xlabel(x_col)
        ax.set_ylabel(col)
        ax.set_title(f"{col} vs {x_col}")
    ax.grid(True)
    ax.legend()
    plt.show()