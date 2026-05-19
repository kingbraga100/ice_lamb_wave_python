import time
import numpy as np
import threading
import nidaqmx
from nidaqmx.constants import ThermocoupleType, CJCSource, TemperatureUnits
import pyvisa
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
import csv
import os
from datetime import datetime

# =========================
# SETTINGS
# =========================
CHANNEL = "cDAQ3Mod1/ai3"
TC_TYPE = ThermocoupleType.K
RESOURCE = 'USB::0x0699::0x03C4::C010201::INSTR'

DISTANCE = 0.10
FREQUENCY_KHZ = 110

S0_START = 15e-6
S0_END   = 45e-6
A0_START = 80e-6

EMITTER_START_THRESHOLD = 0.2

SUMMARY_FILE = "summary_results_05_05_2026.csv"

# =========================
# INIT SCOPE
# =========================
rm = pyvisa.ResourceManager()
scope = rm.open_resource(RESOURCE)
scope.timeout = 10000

scope.write("DATA:WIDTH 2")   # faster + better resolution
scope.write("DATA:ENC RIB")

print(scope.query("*IDN?"))

# =========================
# SHARED VARIABLES
# =========================
latest_temp = np.nan
temp_buffer = []
capture_flag = False
stop_flag = False

# =========================
# TEMP THREAD (SMOOTH + FAST)
# =========================
def temp_display_loop(task):
    global latest_temp, temp_buffer, stop_flag

    while not stop_flag:
        try:
            val = task.read()

            temp_buffer.append(val)
            if len(temp_buffer) > 5:
                temp_buffer.pop(0)

            latest_temp = np.mean(temp_buffer)

            print(f"\rTemperature: {latest_temp:.3f} °C", end="")

            time.sleep(0.3)

        except Exception:
            break

# =========================
# INPUT THREAD
# =========================
def input_loop():
    global capture_flag, stop_flag

    while True:
        cmd = input("\nPress ENTER to capture (q to quit): ")

        if cmd.lower() == 'q':
            stop_flag = True
            break

        capture_flag = True

# =========================
# READ SCOPE
# =========================
def read_channel(ch):

    scope.write(f"DATA:SOURCE {ch}")

    xincr = float(scope.query("WFMPRE:XINCR?"))
    xzero = float(scope.query("WFMPRE:XZERO?"))
    ymult = float(scope.query("WFMPRE:YMULT?"))
    yzero = float(scope.query("WFMPRE:YZERO?"))
    yoff  = float(scope.query("WFMPRE:YOFF?"))

    raw = scope.query_binary_values("CURVE?", datatype='h', container=np.array)

    y = (raw - yoff) * ymult + yzero
    t = np.arange(len(raw)) * xincr + xzero

    return t, y

# =========================
# MAIN
# =========================
with nidaqmx.Task() as task:

    task.ai_channels.add_ai_thrmcpl_chan(
        CHANNEL,
        units=TemperatureUnits.DEG_C,
        thermocouple_type=TC_TYPE,
        cjc_source=CJCSource.BUILT_IN
    )

    threading.Thread(target=temp_display_loop, args=(task,), daemon=True).start()
    threading.Thread(target=input_loop, daemon=True).start()

    print("\n🔥 Live monitoring started")

    while not stop_flag:

        if capture_flag:

            capture_flag = False

            temp = latest_temp  # ⚡ instant, no delay

            if np.isnan(temp):
                print("\nNo valid temperature yet")
                continue

            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

            print("\n🎯 Capture triggered")

            # ===== READ SIGNAL =====
            t, ch1 = read_channel("CH1")
            _, ch2 = read_channel("CH2")

            # ===== PROCESS =====
            ch1_trigger = ch1 - np.median(ch1)
            indices_t0 = np.where(np.abs(ch1_trigger) > EMITTER_START_THRESHOLD)[0]

            if len(indices_t0) == 0:
                print("Emitter not detected")
                continue

            t_shift = t - t[indices_t0[0]]

            # ===== ROBUST BASELINE =====
            baseline_mask = t_shift < 0

            if np.sum(baseline_mask) < 5:
                baseline_mask = t_shift < (np.min(t_shift) + 20e-6)

            ch1_centered = ch1 - np.median(ch1[baseline_mask])
            ch2_centered = ch2 - np.median(ch2[baseline_mask])

            # ===== ENVELOPE =====
            env2 = np.abs(hilbert(ch2_centered))
            env2 = gaussian_filter1d(env2, sigma=5)

            # ===== RESULTS =====
            S0_mask = (t_shift > S0_START) & (t_shift < S0_END)
            A0_mask = (t_shift > A0_START)

            if np.sum(S0_mask) == 0 or np.sum(A0_mask) == 0:
                print("Wave window issue (check oscilloscope time scale)")
                continue

            S0_amp = np.max(env2[S0_mask])
            A0_amp = np.max(env2[A0_mask])

            # =========================
            # PRINT SUMMARY
            # =========================
            print("\n===== SUMMARY =====")
            print(f"T = {temp:.3f} °C")
            print(f"S0 amplitude = {S0_amp*1000:.2f} mV")
            print(f"A0 amplitude = {A0_amp*1000:.2f} mV")

            # =========================
            # SAVE FULL WAVE
            # =========================
            filename = f"wave_T{temp:.2f}C_{timestamp_str}.csv"

            data = np.column_stack([
                t_shift * 1e6,
                ch1_centered,
                ch2_centered,
                env2
            ])

            data_str = np.char.replace(data.astype(str), '.', ',')

            with open(filename, "w", encoding="utf-8") as f:

                f.write(f"Temperature_C;{str(temp).replace('.',',')}\n")
                f.write(f"Timestamp;{timestamp_str}\n\n")
                f.write("Time_us;Emitter_V;Receiver_V;Envelope_V\n")

                for row in data_str:
                    f.write(";".join(row) + "\n")

            print(f"Saved: {filename}")

            # =========================
            # SAVE SUMMARY FILE
            # =========================
            file_exists = os.path.isfile(SUMMARY_FILE)

            with open(SUMMARY_FILE, "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=';')

                if not file_exists:
                    writer.writerow(["Timestamp","Temperature","S0_amp_mV","A0_amp_mV"])

                writer.writerow([
                    timestamp_str,
                    str(temp).replace('.',','),
                    str(S0_amp*1000).replace('.',','),
                    str(A0_amp*1000).replace('.',',')
                ])

        time.sleep(0.05)

scope.close()
print("\nFinished.")