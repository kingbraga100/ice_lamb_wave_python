import numpy as np
import pyvisa
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

# =========================
# PRESENTATION STYLE
# =========================
plt.rcParams.update({
    "font.size": 26,
    "axes.titlesize": 38,
    "axes.labelsize": 34,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
})

# =========================
# USER SETTINGS
# =========================
RESOURCE = 'USB::0x0699::0x03C4::C010201::INSTR'
DISTANCE = 0.10
FREQUENCY_KHZ = 110

S0_START = 15e-6
S0_END   = 45e-6

A0_START = 50e-6
A0_END   = 70e-6

EMITTER_START_THRESHOLD = 0.1
S0_THRESHOLD = 0.01

BASELINE_START = -10e-6
BASELINE_END   = -5e-6

# =========================
# CONNECT TO OSCILLOSCOPE
# =========================
rm = pyvisa.ResourceManager()
scope = rm.open_resource(RESOURCE)
scope.timeout = 10000

print(scope.query("*IDN?"))

scope.write("DATA:WIDTH 1")
scope.write("DATA:ENC RPB")
scope.write("ACQ:STATE STOP")

def read_channel(ch):

    scope.write(f"DATA:SOURCE {ch}")

    xincr = float(scope.query("WFMPRE:XINCR?"))
    xzero = float(scope.query("WFMPRE:XZERO?"))
    ymult = float(scope.query("WFMPRE:YMULT?"))
    yzero = float(scope.query("WFMPRE:YZERO?"))
    yoff  = float(scope.query("WFMPRE:YOFF?"))

    raw = scope.query_binary_values("CURVE?", datatype='B', container=np.array)

    y = (raw - yoff) * ymult + yzero
    t = np.arange(len(raw)) * xincr + xzero

    return t, y

# =========================
# READ SIGNALS
# =========================
t, ch1 = read_channel("CH1")
_, ch2 = read_channel("CH2")

scope.close()

# =========================
# DETECT EMITTER START
# =========================
ch1_trigger = ch1 - np.median(ch1)

indices_t0 = np.where(np.abs(ch1_trigger) > EMITTER_START_THRESHOLD)[0]

if len(indices_t0) == 0:
    raise RuntimeError("Emitter start not detected")

i0 = indices_t0[0]
t0 = t[i0]

# Align time
t_shift = t - t0

# =========================
# BASELINE CORRECTION
# =========================

baseline_mask = (t_shift >= BASELINE_START) & (t_shift <= BASELINE_END)

if np.sum(baseline_mask) < 10:
    raise RuntimeError("Not enough samples in baseline region")

# Compute baseline using SAME window
baseline_ch1 = np.median(ch1[baseline_mask])
baseline_ch2 = np.median(ch2[baseline_mask])

# Remove baseline
ch1_centered = ch1 - baseline_ch1
ch2_centered = ch2 - baseline_ch2

# Convert receiver to mV
ch2_mV = ch2_centered * 1000

# =========================
# ENVELOPE
# =========================
env2 = np.abs(hilbert(ch2_centered))
env2 = gaussian_filter1d(env2, sigma=5)
env2_mV = env2 * 1000

# =========================
# DETECT S0
# =========================
mask_s0 = (t_shift > S0_START) & (t_shift < S0_END)

indices_s0 = np.where(mask_s0 &
                      (np.abs(ch2_centered) > S0_THRESHOLD))[0]

if len(indices_s0) == 0:
    raise RuntimeError("S0 not detected")

i_s0 = indices_s0[0]
tof_s0 = t_shift[i_s0]

# =========================
# DETECT A0
# =========================
mask_a0 = (t_shift > A0_START) & (t_shift < A0_END)

env_a0 = env2[mask_a0]

mins = []
for i in range(1, len(env_a0)-1):
    if env_a0[i] < env_a0[i-1] and env_a0[i] < env_a0[i+1]:
        mins.append(i)

if len(mins) == 0:
    raise RuntimeError("A0 minimum not found")

i_min = mins[-1]

for i in range(i_min+1, len(env_a0)-1):
    if env_a0[i] > env_a0[i-1]:
        i_rise = i
        break

i_a0 = np.where(mask_a0)[0][i_rise]
tof_a0 = t_shift[i_a0]

# =========================
# COMPUTE SYMMETRIC LIMITS
# =========================

emit_lim = np.max(np.abs(ch1_centered)) * 1.1
recv_lim = np.max(np.abs(ch2_mV)) * 1.1

# =========================
# PLOT
# =========================

fig, ax1 = plt.subplots(figsize=(15,8))

# Emitter
line1, = ax1.plot(
    t_shift*1e6,
    ch1_centered,
    color='blue',
    linewidth=3,
    label="Emitter"
)

ax1.set_ylabel("Emitter (V)", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-emit_lim, emit_lim)
ax1.grid(True)

# Receiver axis
ax2 = ax1.twinx()

line2, = ax2.plot(
    t_shift*1e6,
    ch2_mV,
    color='darkorange',
    linewidth=2.5,
    label="Receiver"
)

line3, = ax2.plot(
    t_shift*1e6,
    env2_mV,
    '--',
    color='green',
    linewidth=2,
    label="Envelope"
)

ax2.set_ylabel("Receiver (mV)", color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.set_ylim(-recv_lim, recv_lim)

# zero lines
ax1.axhline(0, color='blue', alpha=0.3)
ax2.axhline(0, color='darkorange', alpha=0.3)

# Regions
s0_patch = ax1.axvspan(
    tof_s0*1e6,
    tof_a0*1e6,
    color='#f5c27a',
    alpha=0.45,
    label="S0 region"
)

a0_patch = ax1.axvspan(
    tof_a0*1e6,
    t_shift[-1]*1e6,
    color='#d6c6f5',
    alpha=0.45,
    label="A0 region"
)

# Labels
ax1.set_xlabel("Time relative to emitter (µs)")
plt.title(f"Lamb Wave Detection - {FREQUENCY_KHZ} kHz")

handles = [line1, line2, line3, s0_patch, a0_patch]
labels = [h.get_label() for h in handles]

ax1.legend(handles, labels, loc="upper left", fontsize=24)

ax1.set_xlim(-80,500)

plt.tight_layout()
plt.show()