# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:27:07 2026

@author: ricar
"""

import numpy as np
import pyvisa
import matplotlib.pyplot as plt

# =========================
# STYLE
# =========================
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 32,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20
})

# =========================
# SETTINGS
# =========================
RESOURCE = 'USB::0x0699::0x03C4::C010201::INSTR'

# =========================
# CONNECT
# =========================
rm = pyvisa.ResourceManager()
scope = rm.open_resource(RESOURCE)
scope.timeout = 15000

print(scope.query("*IDN?"))

# ❗ IMPORTANT:
# Do NOT touch acquisition (you already pressed SINGLE manually)

# =========================
# READ FUNCTION (ROBUST)
# =========================
def read_channel(ch):

    scope.write(f"DATA:SOURCE {ch}")

    # 🔴 FORCE CONSISTENT FORMAT
    scope.write("DATA:ENC RIB")      # signed binary
    scope.write("DATA:WIDTH 2")      # 16-bit

    scope.write("DATA:START 1")

    # 🔴 GET REAL RECORD LENGTH (CRITICAL)
    record_length = int(scope.query("HOR:RECO?"))
    scope.write(f"DATA:STOP {record_length}")

    # Scaling
    xincr = float(scope.query("WFMPRE:XINCR?"))
    xzero = float(scope.query("WFMPRE:XZERO?"))
    ymult = float(scope.query("WFMPRE:YMULT?"))
    yzero = float(scope.query("WFMPRE:YZERO?"))
    yoff  = float(scope.query("WFMPRE:YOFF?"))

    raw = scope.query_binary_values(
        "CURVE?",
        datatype='h',
        container=np.array
    )

    y = (raw - yoff) * ymult + yzero
    t = np.arange(len(raw)) * xincr + xzero

    return t, y

# =========================
# READ CHANNELS
# =========================
t, ch1 = read_channel("CH1")
_, ch2 = read_channel("CH2")

scope.close()

# =========================
# DEBUG
# =========================
print("Time range:", t[0]*1e6, "to", t[-1]*1e6, "µs")
print("CH1 max:", np.max(ch1), "min:", np.min(ch1))
print("CH2 max:", np.max(ch2), "min:", np.min(ch2))

# =========================
# ALIGN TIME (EMITTER START)
# =========================
ch1_centered = ch1 - np.mean(ch1)

# simple threshold detection
threshold = 0.2 * np.max(np.abs(ch1_centered))

indices = np.where(np.abs(ch1_centered) > threshold)[0]

if len(indices) == 0:
    raise RuntimeError("Emitter start not detected")

t0 = t[indices[0]]
t_shift = t - t0

# =========================
# LIMIT TO 0 → 500 µs
# =========================
mask = (t_shift >= 0) & (t_shift <= 500e-6)

t_plot = t_shift[mask]
ch1_plot = ch1_centered[mask]
ch2_plot = (ch2 - np.mean(ch2))[mask]

# =========================
# PLOT
# =========================
plt.figure(figsize=(14,7))

plt.plot(t_plot*1e6, ch1_plot, label="Emitter (CH1)", linewidth=2)
plt.plot(t_plot*1e6, ch2_plot*1000, label="Receiver (CH2, mV)", linewidth=2)

plt.xlabel("Time (µs)")
plt.ylabel("Voltage")
plt.title("Lamb Wave Capture (0 to 500 µs)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()