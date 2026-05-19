# ======================================
# PIEZO IMPEDANCE SWEEP — FINAL CLEAN VERSION
# ======================================

import numpy as np
import pyvisa
import time
import matplotlib.pyplot as plt

# ======================================
# USER SETTINGS
# ======================================

F_START = 2000
F_END   = 6000
F_STEP  = 20

STEP_TIME = 0.5

RESOURCE = 'USB::0x0699::0x03C4::C010201::INSTR'

# Scaling
VOLTAGE_SCALE = 20
CURRENT_GAIN  = 1.5

# ======================================
# CONNECT
# ======================================

rm = pyvisa.ResourceManager()
scope = rm.open_resource(RESOURCE)
scope.timeout = 5000

print("Connected to:", scope.query("*IDN?"))

# ======================================
# CONFIGURE
# ======================================

scope.write("MEASUrement:MEAS1:TYPE RMS")
scope.write("MEASUrement:MEAS1:SOURCE CH1")

scope.write("MEASUrement:MEAS2:TYPE RMS")
scope.write("MEASUrement:MEAS2:SOURCE CH2")

scope.write("MEASUrement:MEAS3:TYPE FREQuency")
scope.write("MEASUrement:MEAS3:SOURCE CH1")

try:
    scope.write("MEASUrement:MEAS4:TYPE PHASe")
    scope.write("MEASUrement:MEAS4:SOURCE CH1,CH2")
    phase_available = True
except:
    phase_available = False

scope.write("MEASUrement:STATistics OFF")

# ======================================
# STORAGE
# ======================================

frequencies = []
V_values = []
I_values = []
Z_values = []
Y_values = []
phase_values = []

# ======================================
# SWEEP
# ======================================

n_steps = int((F_END - F_START)/F_STEP) + 1

print("Start sweep...")
time.sleep(2)

start_time = time.time()

for i in range(n_steps):

    step_start = start_time + i * STEP_TIME
    measure_time = step_start + 0.4

    while time.time() < measure_time:
        time.sleep(0.001)

    try:
        V_rms = float(scope.query("MEASUrement:MEAS1:VALue?"))
        I_rms = float(scope.query("MEASUrement:MEAS2:VALue?"))
        freq  = float(scope.query("MEASUrement:MEAS3:VALue?"))

        V_real = VOLTAGE_SCALE * V_rms
        I_real = I_rms / CURRENT_GAIN

        if I_real < 1e-12:
            Z = np.nan
            Y = np.nan
        else:
            Z = V_real / I_real
            Y = I_real / V_real

        if phase_available:
            phase = float(scope.query("MEASUrement:MEAS4:VALue?"))
        else:
            phase = np.nan

        frequencies.append(freq)
        V_values.append(V_real)
        I_values.append(I_real)
        Z_values.append(Z)
        Y_values.append(Y)
        phase_values.append(phase)

        print(f"{freq:8.1f} Hz | V={V_real:.3f} V | I={I_real:.6f} A | Z={Z:.2f} Ω | Phase={phase:.2f}")

    except Exception as e:
        print("Measurement error:", e)

scope.close()

# ======================================
# PROCESS DATA
# ======================================

frequencies = np.array(frequencies)
V_values = np.array(V_values)
I_values = np.array(I_values)
Z_values = np.array(Z_values)
Y_values = np.array(Y_values)
phase_values = np.array(phase_values)

# Sort
idx = np.argsort(frequencies)

frequencies = frequencies[idx]
V_values = V_values[idx]
I_values = I_values[idx]
Z_values = Z_values[idx]
Y_values = Y_values[idx]
phase_values = phase_values[idx]

frequencies_kHz = frequencies / 1000

# ======================================
# PHASE CLEANING
# ======================================

def phase_diff(a, b):
    return (a - b + 180) % 360 - 180

phase_clean = []
prev = None

for p in phase_values:

    if np.isnan(p) or abs(p) > 170:
        phase_clean.append(np.nan)
        continue

    if prev is None:
        phase_clean.append(p)
        prev = p
        continue

    if abs(phase_diff(p, prev)) > 60:
        phase_clean.append(np.nan)
    else:
        phase_clean.append(p)
        prev = p

phase_clean = np.array(phase_clean)

# ======================================
# INTERPOLATE
# ======================================

valid = ~np.isnan(phase_clean)

phase_interp = np.copy(phase_clean)
phase_interp[~valid] = np.interp(
    frequencies_kHz[~valid],
    frequencies_kHz[valid],
    phase_clean[valid]
)

# ======================================
# UNWRAP
# ======================================

phase_unwrapped = np.unwrap(np.deg2rad(phase_interp))
phase_unwrapped = np.rad2deg(phase_unwrapped)

# ======================================
# PLOTS (ONLY CLEAN PHASE)
# ======================================

plt.figure(figsize=(12,6))
plt.plot(frequencies_kHz, np.abs(Z_values), '-o', markersize=3)
plt.xlabel("Frequency (kHz)")
plt.ylabel("|Z| (Ohm)")
plt.title("Piezo Impedance")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(frequencies_kHz, np.abs(Y_values), '-o', markersize=3)
plt.xlabel("Frequency (kHz)")
plt.ylabel("|Y| (S)")
plt.title("Piezo Admittance")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(frequencies_kHz, phase_unwrapped, '-', linewidth=2)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Phase (deg)")
plt.title("Clean Phase Response")
plt.grid(True)
plt.show()

# ======================================
# SAVE EVERYTHING
# ======================================

np.savetxt(
    "piezo_impedance_full_clean.csv",
    np.column_stack((
        frequencies_kHz,
        V_values,
        I_values,
        Z_values,
        Y_values,
        phase_unwrapped
    )),
    delimiter=",",
    header="Frequency_kHz, Voltage_V, Current_A, Z_Ohm, Y_S, Phase_deg",
    comments=""
)

print("Saved full clean dataset")