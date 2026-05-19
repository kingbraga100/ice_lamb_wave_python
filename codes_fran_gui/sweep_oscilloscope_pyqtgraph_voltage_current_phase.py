import os
import sys
import time
import csv
from pathlib import Path
import traceback

import numpy as np
import pyvisa

QT_LIB = None
try:
    from PyQt5 import QtCore, QtWidgets
    QT_LIB = "PyQt5"
except ImportError:
    from PySide6 import QtCore, QtWidgets
    QT_LIB = "PySide6"

Signal = QtCore.pyqtSignal if hasattr(QtCore, "pyqtSignal") else QtCore.Signal
os.environ["PYQTGRAPH_QT_LIB"] = QT_LIB
import pyqtgraph as pg

# ======================================
# USER SETTINGS
# ======================================
DEFAULT_ACQ_DURATION = 60.0
DEFAULT_USE_ELAPSED_TIME = True
POLL_TIME_MS = 50
RESOURCE = "USB::0x0699::0x03C4::C010201::INSTR"
SAVE_FILE = "scope_waveform_voltage_current_phase.csv"
POINTS = 1000
UI_RENDER_INTERVAL_MS = 33
AUTO_RANGE_INTERVAL_S = 0.35
PLOT_ANTIALIAS = False
PREAMBLE_REFRESH_S = 2.0

# CH2 current scaling. If your scope CH2 is already in mA, keep this at 1.0.
CURRENT_SCALE_MA_PER_V = 1.5
CH2_LOW_PASS_CUTOFF_HZ = 2000.0
CURRENT_FILTER_TYPE_DEFAULT = "lpf_1pole"
CURRENT_FILTER_BUTTER_ORDER_DEFAULT = 4
PHASE_HISTORY_SECONDS = 120.0
ADMITTANCE_FREQ_MAX_POINTS = 5000
SCOPE_FREQ_QUERY_REFRESH_S = 0.25
DEFAULT_DYNAMIC_CUTOFF_ENABLED = False
DEFAULT_DYNAMIC_CUTOFF_MULTIPLIER = 2.0
DEFAULT_DYNAMIC_CUTOFF_MIN_HZ = 200.0
DEFAULT_DYNAMIC_CUTOFF_MAX_HZ = 200000.0


class LiveStartWorker(QtCore.QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref

    def run(self):
        try:
            self.app_ref._connect_scope_blocking()
            try:
                self.app_ref.scope.write("ACQuire:STATE RUN")
            except Exception:
                pass
            self.finished.emit("Live running on scope. Ready to acquire waveform.")
        except Exception as exc:
            print(traceback.format_exc())
            self.error.emit(str(exc))


class WaveformLiveWorker(QtCore.QObject):
    waveform_ready = Signal(object)  # {"waveforms": {channel: (t, v)}, "scope_freq_hz": float|nan}
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, app_ref, channels, poll_time_ms, use_scope_frequency=False):
        super().__init__()
        self.app_ref = app_ref
        self.channels = channels
        self.poll_time_ms = poll_time_ms
        self.use_scope_frequency = bool(use_scope_frequency)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            while not self._stop:
                payload = {}
                for ch in self.channels:
                    if self._stop:
                        break
                    t_arr, v_arr = self.app_ref.fetch_waveform(ch)
                    payload[ch] = (t_arr, v_arr)
                if payload:
                    scope_freq_hz = np.nan
                    if self.use_scope_frequency:
                        scope_freq_hz = self.app_ref.fetch_scope_ch1_frequency_hz()
                    self.waveform_ready.emit({"waveforms": payload, "scope_freq_hz": scope_freq_hz})
                QtCore.QThread.msleep(int(self.poll_time_ms))
            self.finished.emit("Live polling stopped.")
        except Exception as exc:
            print(traceback.format_exc())
            self.error.emit(str(exc))


class ProcessingWorker(QtCore.QObject):
    processed = Signal(object)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._pending_job = None
        self._busy = False
        self._stop = False

    def stop(self):
        self._stop = True
        self._pending_job = None

    def enqueue_job(self, job):
        if self._stop:
            return
        self._pending_job = job
        if self._busy:
            return
        self._busy = True
        QtCore.QTimer.singleShot(0, self._process_pending)

    def _process_pending(self):
        try:
            while (not self._stop) and (self._pending_job is not None):
                job = self._pending_job
                self._pending_job = None
                result = self._compute_job(job)
                if result is not None:
                    self.processed.emit(result)
        except Exception as exc:
            print(traceback.format_exc())
            self.error.emit(str(exc))
        finally:
            self._busy = False

    @staticmethod
    def _estimate_dt(t):
        t = np.asarray(t, dtype=float)
        if t.size < 2:
            return np.nan
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            return np.nan
        return dt

    @staticmethod
    def _low_pass_filter(signal, dt, cutoff_hz):
        x = np.asarray(signal, dtype=float)
        if x.size == 0:
            return x
        if not np.isfinite(dt) or dt <= 0 or cutoff_hz <= 0:
            return x.copy()
        rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
        alpha = dt / (rc + dt)
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, x.size):
            y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
        return y

    @staticmethod
    def _moving_average_filter(signal, dt, cutoff_hz):
        x = np.asarray(signal, dtype=float)
        if x.size == 0:
            return x
        if not np.isfinite(dt) or dt <= 0 or cutoff_hz <= 0:
            return x.copy()

        # Approximate MA length from desired cutoff (fc ~= 0.443 / (N*dt))
        n = int(max(1, round(0.443 / (float(cutoff_hz) * float(dt)))))
        n = min(n, int(x.size))
        if n <= 1:
            return x.copy()

        kernel = np.ones(n, dtype=float) / float(n)
        return np.convolve(x, kernel, mode="same")

    @staticmethod
    def _moving_average_window_len(dt, cutoff_hz, max_len):
        if not np.isfinite(dt) or dt <= 0 or cutoff_hz <= 0:
            return 1
        n = int(max(1, round(0.443 / (float(cutoff_hz) * float(dt)))))
        return int(max(1, min(int(max_len), n)))

    @staticmethod
    def _biquad_low_pass(signal, dt, cutoff_hz, q):
        x = np.asarray(signal, dtype=float)
        if x.size == 0:
            return x
        if not np.isfinite(dt) or dt <= 0 or cutoff_hz <= 0 or q <= 0:
            return x.copy()

        fs = 1.0 / float(dt)
        if (not np.isfinite(fs)) or fs <= 0:
            return x.copy()
        f0 = min(float(cutoff_hz), 0.49 * fs)
        if f0 <= 0:
            return x.copy()

        w0 = 2.0 * np.pi * f0 / fs
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * float(q))

        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0

        y = np.empty_like(x)
        z1 = 0.0
        z2 = 0.0
        for i in range(x.size):
            xn = float(x[i])
            yn = b0 * xn + z1
            z1 = b1 * xn - a1 * yn + z2
            z2 = b2 * xn - a2 * yn
            y[i] = yn
        return y

    @staticmethod
    def _butterworth_filter(signal, dt, cutoff_hz, order):
        x = np.asarray(signal, dtype=float)
        if x.size == 0:
            return x

        ord_clean = 2 if int(order) <= 2 else 4
        if ord_clean == 2:
            return ProcessingWorker._biquad_low_pass(x, dt, cutoff_hz, q=1.0 / np.sqrt(2.0))

        # 4th-order Butterworth as cascade of two biquads with Butterworth Qs.
        y = ProcessingWorker._biquad_low_pass(x, dt, cutoff_hz, q=0.541196100146197)
        y = ProcessingWorker._biquad_low_pass(y, dt, cutoff_hz, q=1.306562964876377)
        return y

    @staticmethod
    def _biquad_coeffs_low_pass(dt, cutoff_hz, q):
        if not np.isfinite(dt) or dt <= 0 or cutoff_hz <= 0 or q <= 0:
            return None
        fs = 1.0 / float(dt)
        if (not np.isfinite(fs)) or fs <= 0:
            return None
        f0 = min(float(cutoff_hz), 0.49 * fs)
        if f0 <= 0:
            return None
        w0 = 2.0 * np.pi * f0 / fs
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * float(q))
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0
        return b0, b1, b2, a1, a2

    @staticmethod
    def _biquad_phase_deg(dt, cutoff_hz, q, freq_hz):
        coeffs = ProcessingWorker._biquad_coeffs_low_pass(dt, cutoff_hz, q)
        if coeffs is None or (not np.isfinite(freq_hz)) or freq_hz <= 0.0:
            return np.nan
        b0, b1, b2, a1, a2 = coeffs
        fs = 1.0 / float(dt)
        omega = 2.0 * np.pi * min(float(freq_hz), 0.49 * fs) / fs
        z1 = np.exp(-1j * omega)
        z2 = np.exp(-2j * omega)
        num = b0 + b1 * z1 + b2 * z2
        den = 1.0 + a1 * z1 + a2 * z2
        if den == 0:
            return np.nan
        h = num / den
        return float(np.degrees(np.angle(h)))

    @staticmethod
    def _filter_phase_deg(filter_type, dt, cutoff_hz, filter_order, freq_hz):
        ft = str(filter_type).strip().lower() if filter_type is not None else CURRENT_FILTER_TYPE_DEFAULT
        if (not np.isfinite(freq_hz)) or freq_hz <= 0.0 or (not np.isfinite(dt)) or dt <= 0:
            return np.nan
        if ft in ("none", "raw", "off"):
            return 0.0
        if ft in ("lpf_1pole",):
            fs = 1.0 / float(dt)
            alpha = float(dt) / ((1.0 / (2.0 * np.pi * float(cutoff_hz))) + float(dt)) if cutoff_hz > 0 else 1.0
            beta = 1.0 - alpha
            omega = 2.0 * np.pi * min(float(freq_hz), 0.49 * fs) / fs
            z1 = np.exp(-1j * omega)
            h = alpha / (1.0 - beta * z1)
            return float(np.degrees(np.angle(h)))
        if ft in ("moving_average", "ma"):
            n = ProcessingWorker._moving_average_window_len(dt, cutoff_hz, max_len=1000000)
            if n <= 1:
                return 0.0
            fs = 1.0 / float(dt)
            omega = 2.0 * np.pi * min(float(freq_hz), 0.49 * fs) / fs
            k = np.arange(n, dtype=float)
            h = np.mean(np.exp(-1j * omega * k))
            return float(np.degrees(np.angle(h)))
        if ft in ("butterworth", "butter", "bw"):
            ord_clean = 2 if int(filter_order) <= 2 else 4
            if ord_clean == 2:
                return ProcessingWorker._biquad_phase_deg(dt, cutoff_hz, q=1.0 / np.sqrt(2.0), freq_hz=freq_hz)
            p1 = ProcessingWorker._biquad_phase_deg(dt, cutoff_hz, q=0.541196100146197, freq_hz=freq_hz)
            p2 = ProcessingWorker._biquad_phase_deg(dt, cutoff_hz, q=1.306562964876377, freq_hz=freq_hz)
            if not np.isfinite(p1) or not np.isfinite(p2):
                return np.nan
            return float(p1 + p2)
        return np.nan

    @staticmethod
    def _apply_current_filter(signal, dt, cutoff_hz, filter_type, filter_order):
        ft = str(filter_type).strip().lower() if filter_type is not None else CURRENT_FILTER_TYPE_DEFAULT
        if ft in ("none", "raw", "off"):
            return np.asarray(signal, dtype=float).copy()
        if ft in ("moving_average", "ma"):
            return ProcessingWorker._moving_average_filter(signal, dt, cutoff_hz)
        if ft in ("butterworth", "butter", "bw"):
            return ProcessingWorker._butterworth_filter(signal, dt, cutoff_hz, order=filter_order)
        return ProcessingWorker._low_pass_filter(signal, dt, cutoff_hz)

    @staticmethod
    def _estimate_phase_and_frequency(v1, i2, t1, t2):
        x = np.asarray(v1, dtype=float)
        y = np.asarray(i2, dtype=float)
        n = min(x.size, y.size)
        if n < 64:
            return np.nan, np.nan

        dt1 = ProcessingWorker._estimate_dt(t1)
        dt2 = ProcessingWorker._estimate_dt(t2)
        if not np.isfinite(dt1) or not np.isfinite(dt2):
            return np.nan, np.nan

        x = x[:n]
        y = y[:n]
        x = x - np.mean(x)
        y = y - np.mean(y)
        rms_x = np.sqrt(np.mean(x * x))
        rms_y = np.sqrt(np.mean(y * y))
        if np.allclose(x, 0.0) or np.allclose(y, 0.0) or rms_x <= 0.0 or rms_y <= 0.0:
            return np.nan, np.nan

        window = np.hanning(n)
        xw = x * window
        yw = y * window
        xf = np.fft.rfft(xw)
        yf = np.fft.rfft(yw)
        if xf.size < 3 or yf.size < 3:
            return np.nan, np.nan

        mag = np.abs(xf)
        mag[0] = 0.0
        k = int(np.argmax(mag))
        if k <= 0 or k >= yf.size:
            return np.nan, np.nan

        phase_rad = np.angle(yf[k]) - np.angle(xf[k])
        phase_rad = (phase_rad + np.pi) % (2.0 * np.pi) - np.pi
        freq_hz = float(k) / (float(n) * float(dt1))
        if not np.isfinite(freq_hz) or freq_hz <= 0.0:
            freq_hz = np.nan
        return float(np.degrees(phase_rad)), freq_hz

    @staticmethod
    def _estimate_dominant_frequency_hz(signal, t):
        x = np.asarray(signal, dtype=float)
        n = x.size
        if n < 32:
            return np.nan
        dt = ProcessingWorker._estimate_dt(t)
        if not np.isfinite(dt) or dt <= 0:
            return np.nan
        x = x - np.mean(x)
        if np.allclose(x, 0.0):
            return np.nan
        xf = np.fft.rfft(x * np.hanning(n))
        if xf.size < 3:
            return np.nan
        mag = np.abs(xf)
        mag[0] = 0.0
        k = int(np.argmax(mag))
        if k <= 0:
            return np.nan
        freq_hz = float(k) / (float(n) * float(dt))
        if not np.isfinite(freq_hz) or freq_hz <= 0.0:
            return np.nan
        return freq_hz

    @staticmethod
    def _estimate_phase_deg_at_frequency(v1, i2, t1, t2, target_freq_hz):
        x = np.asarray(v1, dtype=float)
        y = np.asarray(i2, dtype=float)
        n = min(x.size, y.size)
        if n < 64:
            return np.nan

        dt1 = ProcessingWorker._estimate_dt(t1)
        dt2 = ProcessingWorker._estimate_dt(t2)
        if not np.isfinite(dt1) or not np.isfinite(dt2):
            return np.nan
        if not np.isfinite(target_freq_hz) or float(target_freq_hz) <= 0.0:
            return np.nan

        x = x[:n] - np.mean(x[:n])
        y = y[:n] - np.mean(y[:n])
        rms_x = np.sqrt(np.mean(x * x))
        rms_y = np.sqrt(np.mean(y * y))
        if np.allclose(x, 0.0) or np.allclose(y, 0.0) or rms_x <= 0.0 or rms_y <= 0.0:
            return np.nan

        xf = np.fft.rfft(x * np.hanning(n))
        yf = np.fft.rfft(y * np.hanning(n))
        if xf.size < 3 or yf.size < 3:
            return np.nan

        freqs = np.fft.rfftfreq(n, d=float(dt1))
        if freqs.size != xf.size:
            return np.nan
        nyquist = 0.5 / float(dt1)
        f = min(float(target_freq_hz), 0.99 * nyquist)
        k = int(np.argmin(np.abs(freqs - f)))
        if k <= 0 or k >= yf.size:
            return np.nan

        phase_rad = np.angle(yf[k]) - np.angle(xf[k])
        phase_rad = (phase_rad + np.pi) % (2.0 * np.pi) - np.pi
        return float(np.degrees(phase_rad))

    @staticmethod
    def _compute_electrical_metrics(v1_volt, i2_ma, phase_deg):
        v = np.asarray(v1_volt, dtype=float)
        i_ma = np.asarray(i2_ma, dtype=float)
        n = min(v.size, i_ma.size)
        if n < 8:
            return np.nan, np.nan, np.nan, np.nan

        v = v[:n]
        i_a = i_ma[:n] / 1000.0
        mask = np.isfinite(v) & np.isfinite(i_a)
        if np.count_nonzero(mask) < 8:
            return np.nan, np.nan, np.nan, np.nan

        v = v[mask]
        i_a = i_a[mask]
        vrms = float(np.sqrt(np.mean(v * v)))
        irms = float(np.sqrt(np.mean(i_a * i_a)))
        if not np.isfinite(vrms) or not np.isfinite(irms) or vrms <= 0.0 or irms <= 0.0:
            return np.nan, np.nan, np.nan, np.nan

        y_mag_s = irms / vrms
        z_mag_ohm = vrms / irms
        if not np.isfinite(phase_deg):
            return y_mag_s, z_mag_ohm, np.nan, np.nan

        phi_rad = float(np.deg2rad(phase_deg))
        g_s = y_mag_s * np.cos(phi_rad)
        b_s = y_mag_s * np.sin(phi_rad)
        return float(y_mag_s), float(z_mag_ohm), float(g_s), float(b_s)

    def _compute_job(self, job):
        payload = job.get("waveforms", {})
        cutoff_hz = float(job.get("cutoff_hz", CH2_LOW_PASS_CUTOFF_HZ))
        filter_type = str(job.get("filter_type", CURRENT_FILTER_TYPE_DEFAULT))
        filter_order = int(job.get("filter_order", CURRENT_FILTER_BUTTER_ORDER_DEFAULT))
        use_scope_frequency = bool(job.get("use_scope_frequency", False))
        scope_freq_hz = float(job.get("scope_freq_hz", np.nan))
        phase_compensation_enabled = bool(job.get("phase_compensation_enabled", False))
        dynamic_cutoff_enabled = bool(job.get("dynamic_cutoff_enabled", False))
        dynamic_cutoff_multiplier = float(job.get("dynamic_cutoff_multiplier", 2.0))
        dynamic_cutoff_min_hz = float(job.get("dynamic_cutoff_min_hz", CH2_LOW_PASS_CUTOFF_HZ))
        dynamic_cutoff_max_hz = float(job.get("dynamic_cutoff_max_hz", CH2_LOW_PASS_CUTOFF_HZ))
        current_scale = float(job.get("current_scale_ma_per_v", CURRENT_SCALE_MA_PER_V))

        if "CH1" not in payload or "CH2" not in payload:
            return {"ok": False, "status": "Waiting for both CH1 and CH2 waveforms..."}

        t1, ch1_v = payload["CH1"]
        t2, ch2_raw = payload["CH2"]
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        ch1_v = np.asarray(ch1_v, dtype=float)
        ch2_raw = np.asarray(ch2_raw, dtype=float)

        dt2 = self._estimate_dt(t2)
        ch2_current_ma = ch2_raw * current_scale
        freq_source_hz = scope_freq_hz if (use_scope_frequency and np.isfinite(scope_freq_hz) and scope_freq_hz > 0.0) else self._estimate_dominant_frequency_hz(ch1_v, t1)
        cutoff_hz_used = cutoff_hz
        if dynamic_cutoff_enabled and np.isfinite(freq_source_hz):
            k = max(0.1, float(dynamic_cutoff_multiplier))
            fc_min = max(0.1, min(dynamic_cutoff_min_hz, dynamic_cutoff_max_hz))
            fc_max = max(fc_min, max(dynamic_cutoff_min_hz, dynamic_cutoff_max_hz))
            cutoff_hz_used = min(fc_max, max(fc_min, k * float(freq_source_hz)))
        ch2_current_ma_filtered = self._apply_current_filter(ch2_current_ma, dt2, cutoff_hz_used, filter_type, filter_order)

        phase_fft_deg, freq_fft_hz = self._estimate_phase_and_frequency(ch1_v, ch2_current_ma_filtered, t1, t2)
        freq_hz = scope_freq_hz if (use_scope_frequency and np.isfinite(scope_freq_hz) and scope_freq_hz > 0.0) else freq_fft_hz
        if use_scope_frequency and np.isfinite(freq_hz):
            phase_meas_deg = self._estimate_phase_deg_at_frequency(ch1_v, ch2_current_ma_filtered, t1, t2, freq_hz)
            if not np.isfinite(phase_meas_deg):
                phase_meas_deg = phase_fft_deg
        else:
            phase_meas_deg = phase_fft_deg

        phase_filter_deg = self._filter_phase_deg(filter_type, dt2, cutoff_hz_used, filter_order, freq_hz)
        phase_deg = phase_meas_deg
        if phase_compensation_enabled and np.isfinite(phase_meas_deg) and np.isfinite(phase_filter_deg):
            phase_deg = phase_meas_deg - phase_filter_deg
            phase_deg = (phase_deg + 180.0) % 360.0 - 180.0

        previous_phase_values = getattr(self, "phase_values", [])
        if np.isfinite(phase_deg) and len(previous_phase_values) > 0 and np.isfinite(previous_phase_values[-1]):
            prev_phase_deg = float(previous_phase_values[-1])
            delta = (phase_deg - prev_phase_deg + 180.0) % 360.0 - 180.0
            if abs(delta) > 30.0:
                phase_deg = prev_phase_deg

        y_mag_s, z_mag_ohm, g_s, b_s = self._compute_electrical_metrics(ch1_v, ch2_current_ma_filtered, phase_deg)

        return {
            "ok": True,
            "t1": t1,
            "ch1_v": ch1_v,
            "t2": t2,
            "ch2_current_ma_filtered": ch2_current_ma_filtered,
            "phase_deg": phase_deg,
            "phase_meas_deg": phase_meas_deg,
            "phase_filter_deg": phase_filter_deg,
            "freq_hz": freq_hz,
            "cutoff_hz_used": float(cutoff_hz_used),
            "y_mag_s": y_mag_s,
            "z_mag_ohm": z_mag_ohm,
            "g_s": g_s,
            "b_s": b_s,
        }


class SweepApp(QtWidgets.QMainWindow):
    process_payload = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oscilloscope Viewer: Voltage, Current, Phase, Admittance (PyQt + pyqtgraph)")
        self.resize(1100, 860)

        self.rm = None
        self.scope = None
        self.connected = False
        self.busy_live = False
        self.live_running = False
        self.recording = False
        self.record_start_time = None

        self.live_thread = None
        self.live_worker = None
        self.processing_thread = None
        self.processing_worker = None
        self.pending_processed_payload = None
        self.waveform_preamble_cache = {}
        self.last_data_source = None
        self.last_data_stop_points = None
        self.last_scope_freq_query_ts = 0.0
        self.cached_scope_freq_hz = np.nan
        self.last_autorange_ts = 0.0
        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setInterval(UI_RENDER_INTERVAL_MS)
        self.render_timer.timeout.connect(self._consume_pending_processed_payload)

        # CH1 -> voltage (V), CH2 -> filtered current (mA)
        self.last_waveforms = {"CH1": (np.array([]), np.array([])), "CH2": (np.array([]), np.array([]))}
        self.phase_times = []
        self.phase_values = []
        self.admittance_values_s = []
        self.impedance_values_ohm = []
        self.conductance_values_s = []
        self.susceptance_values_s = []
        self.admittance_freq_hz = []
        self.admittance_freq_values_s = []
        self.live_plot_zero_time = None

        self.save_path = str((Path.cwd() / SAVE_FILE).resolve())

        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.splitterMoved.connect(lambda *_: self._relayout_kpi_cards())
        layout.addWidget(self.main_splitter, stretch=1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(560)
        self.left_scroll = QtWidgets.QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.left_scroll.setMinimumWidth(300)
        self.left_scroll.setMaximumWidth(620)
        self.left_scroll.setWidget(left_panel)
        self.main_splitter.addWidget(self.left_scroll)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setSizes([420, 1000])

        acq_group = QtWidgets.QGroupBox("Acquisition")
        acq_layout = QtWidgets.QVBoxLayout(acq_group)

        acq_btn_row = QtWidgets.QHBoxLayout()
        self.live_btn = QtWidgets.QPushButton("Live Start")
        self.live_btn.clicked.connect(self.live_start)
        acq_btn_row.addWidget(self.live_btn)
        self.stop_live_btn = QtWidgets.QPushButton("Stop Live")
        self.stop_live_btn.clicked.connect(self.stop_live)
        self.stop_live_btn.setEnabled(False)
        acq_btn_row.addWidget(self.stop_live_btn)
        self.acquire_btn = QtWidgets.QPushButton("Acquire")
        self.acquire_btn.clicked.connect(self.toggle_acquire)
        self.acquire_btn.setEnabled(False)
        acq_btn_row.addWidget(self.acquire_btn)
        acq_layout.addLayout(acq_btn_row)

        self.elapsed_checkbox = QtWidgets.QCheckBox("Elapsed time enabled")
        self.elapsed_checkbox.setChecked(DEFAULT_USE_ELAPSED_TIME)
        acq_layout.addWidget(self.elapsed_checkbox)

        acq_form = QtWidgets.QFormLayout()
        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 86400.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setValue(DEFAULT_ACQ_DURATION)
        acq_form.addRow("Duration [s]", self.duration_spin)

        self.cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.cutoff_spin.setRange(0.1, 1_000_000.0)
        self.cutoff_spin.setDecimals(1)
        self.cutoff_spin.setSingleStep(100.0)
        self.cutoff_spin.setValue(CH2_LOW_PASS_CUTOFF_HZ)
        self.cutoff_spin.valueChanged.connect(self._on_cutoff_changed)
        acq_form.addRow("LPF Cutoff [Hz]", self.cutoff_spin)

        self.dynamic_cutoff_cb = QtWidgets.QCheckBox("Dynamic Cutoff")
        self.dynamic_cutoff_cb.setChecked(DEFAULT_DYNAMIC_CUTOFF_ENABLED)
        self.dynamic_cutoff_cb.toggled.connect(self._on_dynamic_cutoff_toggled)
        acq_form.addRow("Cutoff Mode", self.dynamic_cutoff_cb)

        self.dynamic_cutoff_k_spin = QtWidgets.QDoubleSpinBox()
        self.dynamic_cutoff_k_spin.setRange(0.1, 20.0)
        self.dynamic_cutoff_k_spin.setDecimals(2)
        self.dynamic_cutoff_k_spin.setSingleStep(0.1)
        self.dynamic_cutoff_k_spin.setValue(DEFAULT_DYNAMIC_CUTOFF_MULTIPLIER)
        self.dynamic_cutoff_k_spin.valueChanged.connect(self._on_dynamic_cutoff_params_changed)
        acq_form.addRow("Dynamic k (fc=k*f)", self.dynamic_cutoff_k_spin)

        self.dynamic_cutoff_min_spin = QtWidgets.QDoubleSpinBox()
        self.dynamic_cutoff_min_spin.setRange(0.1, 1_000_000.0)
        self.dynamic_cutoff_min_spin.setDecimals(1)
        self.dynamic_cutoff_min_spin.setSingleStep(100.0)
        self.dynamic_cutoff_min_spin.setValue(DEFAULT_DYNAMIC_CUTOFF_MIN_HZ)
        self.dynamic_cutoff_min_spin.valueChanged.connect(self._on_dynamic_cutoff_params_changed)
        acq_form.addRow("Dynamic fc min [Hz]", self.dynamic_cutoff_min_spin)

        self.dynamic_cutoff_max_spin = QtWidgets.QDoubleSpinBox()
        self.dynamic_cutoff_max_spin.setRange(0.1, 1_000_000.0)
        self.dynamic_cutoff_max_spin.setDecimals(1)
        self.dynamic_cutoff_max_spin.setSingleStep(100.0)
        self.dynamic_cutoff_max_spin.setValue(DEFAULT_DYNAMIC_CUTOFF_MAX_HZ)
        self.dynamic_cutoff_max_spin.valueChanged.connect(self._on_dynamic_cutoff_params_changed)
        acq_form.addRow("Dynamic fc max [Hz]", self.dynamic_cutoff_max_spin)

        self.filter_type_combo = QtWidgets.QComboBox()
        self.filter_type_combo.addItem("1-pole LPF (IIR)", "lpf_1pole")
        self.filter_type_combo.addItem("Butterworth (IIR)", "butterworth")
        self.filter_type_combo.addItem("Moving Average (FIR)", "moving_average")
        self.filter_type_combo.addItem("None (Raw CH2)", "none")
        self.filter_type_combo.setCurrentIndex(
            max(self.filter_type_combo.findData(CURRENT_FILTER_TYPE_DEFAULT), 0)
        )
        self.filter_type_combo.currentIndexChanged.connect(self._on_filter_type_changed)
        acq_form.addRow("Current Filter", self.filter_type_combo)

        self.filter_order_spin = QtWidgets.QSpinBox()
        self.filter_order_spin.setRange(2, 4)
        self.filter_order_spin.setSingleStep(2)
        self.filter_order_spin.setValue(CURRENT_FILTER_BUTTER_ORDER_DEFAULT)
        self.filter_order_spin.valueChanged.connect(self._on_filter_order_changed)
        acq_form.addRow("Butterworth Order", self.filter_order_spin)

        self.phase_compensation_cb = QtWidgets.QCheckBox("Phase Compensation")
        self.phase_compensation_cb.setChecked(False)
        self.phase_compensation_cb.toggled.connect(self._on_phase_compensation_toggled)
        acq_form.addRow("Phase", self.phase_compensation_cb)

        self.poll_time_spin = QtWidgets.QSpinBox()
        self.poll_time_spin.setRange(10, 10_000)
        self.poll_time_spin.setSingleStep(10)
        self.poll_time_spin.setValue(POLL_TIME_MS)
        self.poll_time_spin.valueChanged.connect(self._on_poll_time_changed)
        acq_form.addRow("Poll Time [ms]", self.poll_time_spin)

        self.use_scope_freq_cb = QtWidgets.QCheckBox("Use Scope CH1 Frequency")
        self.use_scope_freq_cb.setChecked(False)
        self.use_scope_freq_cb.toggled.connect(self._on_scope_freq_toggled)
        acq_form.addRow("Frequency Source", self.use_scope_freq_cb)
        acq_layout.addLayout(acq_form)
        left_layout.addWidget(acq_group)

        view_group = QtWidgets.QGroupBox("Dashboard View")
        view_layout = QtWidgets.QVBoxLayout(view_group)
        self.show_waveform_overlay_cb = QtWidgets.QCheckBox("V + I")
        self.show_waveform_overlay_cb.setChecked(True)
        self.show_piezo_overlay_cb = QtWidgets.QCheckBox("Phase + |Y|")
        self.show_piezo_overlay_cb.setChecked(True)
        self.show_trend_overlay_cb = QtWidgets.QCheckBox("Norm |Z| + |Y|")
        self.show_trend_overlay_cb.setChecked(False)
        self.show_nyquist_cb = QtWidgets.QCheckBox("Nyquist (G-B)")
        self.show_nyquist_cb.setChecked(False)
        self.show_y_vs_f_cb = QtWidgets.QCheckBox("|Y| vs f")
        self.show_y_vs_f_cb.setChecked(False)

        for cb in (
            self.show_waveform_overlay_cb,
            self.show_piezo_overlay_cb,
            self.show_trend_overlay_cb,
            self.show_nyquist_cb,
            self.show_y_vs_f_cb,
        ):
            cb.toggled.connect(self._apply_plot_visibility)
            view_layout.addWidget(cb)

        preset_row = QtWidgets.QHBoxLayout()
        self.view_basic_btn = QtWidgets.QPushButton("Basic")
        self.view_basic_btn.clicked.connect(self._set_view_preset_basic)
        preset_row.addWidget(self.view_basic_btn)

        self.view_piezo_btn = QtWidgets.QPushButton("Piezo")
        self.view_piezo_btn.clicked.connect(self._set_view_preset_piezo)
        preset_row.addWidget(self.view_piezo_btn)

        self.view_advanced_btn = QtWidgets.QPushButton("Advanced")
        self.view_advanced_btn.clicked.connect(self._set_view_preset_advanced)
        preset_row.addWidget(self.view_advanced_btn)
        view_layout.addLayout(preset_row)
        left_layout.addWidget(view_group)

        export_group = QtWidgets.QGroupBox("Data & Export")
        export_layout = QtWidgets.QVBoxLayout(export_group)
        self.clear_btn = QtWidgets.QPushButton("Clear Plot")
        self.clear_btn.clicked.connect(self.clear_data)
        export_layout.addWidget(self.clear_btn)
        self.save_as_btn = QtWidgets.QPushButton("Save As...")
        self.save_as_btn.clicked.connect(self.choose_save_path)
        export_layout.addWidget(self.save_as_btn)
        self.export_mpl_btn = QtWidgets.QPushButton("Export Matplotlib...")
        self.export_mpl_btn.clicked.connect(self.export_matplotlib_plot)
        export_layout.addWidget(self.export_mpl_btn)
        self.export_plotly_btn = QtWidgets.QPushButton("Export Plotly...")
        self.export_plotly_btn.clicked.connect(self.export_plotly_plot)
        export_layout.addWidget(self.export_plotly_btn)
        left_layout.addWidget(export_group)

        status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        self.status_label = QtWidgets.QLabel("Press Live Start, then Acquire.")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        self.kpi_grid = QtWidgets.QGridLayout()
        self.kpi_grid.setHorizontalSpacing(8)
        self.kpi_grid.setVerticalSpacing(8)
        self.kpi_value_labels = {}
        self.kpi_cards = {}

        self.kpi_specs = [
            ("phase", "Phase", "deg", "#1f8a65"),
            ("y", "|Y|", "mS", "#b4781e"),
            ("z", "|Z|", "ohm", "#5f3db8"),
            ("g", "G", "mS", "#0f6ab4"),
            ("b", "B", "mS", "#8a2be2"),
        ]
        for key, title, unit, color in self.kpi_specs:
            card, value_label = self._create_kpi_card(title, unit, color)
            self.kpi_cards[key] = card
            self.kpi_value_labels[key] = value_label
        status_layout.addLayout(self.kpi_grid)
        self._relayout_kpi_cards()

        self.metrics_label = QtWidgets.QLabel("KPI summary: --")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        status_layout.addWidget(self.metrics_label)

        self.config_label = QtWidgets.QLabel("")
        self.config_label.setWordWrap(True)
        status_layout.addWidget(self.config_label)
        self._update_config_label()
        self._on_filter_type_changed(self.filter_type_combo.currentIndex())
        self._on_dynamic_cutoff_toggled(self.dynamic_cutoff_cb.isChecked())

        self.save_path_label = QtWidgets.QLabel(f"Save: {self.save_path}")
        self.save_path_label.setWordWrap(True)
        self.save_path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        status_layout.addWidget(self.save_path_label)
        left_layout.addWidget(status_group)
        left_layout.addStretch(1)

        self.tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.tabs, stretch=1)

        live_tab = QtWidgets.QWidget()
        live_layout = QtWidgets.QVBoxLayout(live_tab)
        self.tabs.addTab(live_tab, "Live")

        (
            self.waveform_overlay_plot,
            self.waveform_voltage_curve,
            self.waveform_current_curve,
            self.waveform_overlay_right_vb,
        ) = self._create_dual_axis_plot(
            title="Waveform Overlay: CH1 Voltage + CH2 Current",
            x_label="Time",
            x_units="s",
            left_label="Voltage",
            left_units="V",
            right_label="Current",
            right_units="mA",
            left_pen=pg.mkPen((30, 144, 255), width=2),
            right_pen=pg.mkPen((220, 70, 70), width=2),
        )
        self._configure_plot_performance(self.waveform_overlay_plot)
        self.waveform_overlay_plot.setMinimumHeight(240)

        (
            self.piezo_overlay_plot,
            self.piezo_phase_curve,
            self.piezo_admittance_curve,
            self.piezo_overlay_right_vb,
        ) = self._create_dual_axis_plot(
            title="Piezo Overlay: Phase + |Y|",
            x_label="Live Time",
            x_units="s",
            left_label="Phase",
            left_units="deg",
            right_label="|Y|",
            right_units="mS",
            left_pen=pg.mkPen((30, 150, 80), width=2),
            right_pen=pg.mkPen((180, 120, 30), width=2),
        )
        self._configure_plot_performance(self.piezo_overlay_plot)
        self.phase_zero_line = pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen((100, 100, 100), style=QtCore.Qt.DashLine))
        self.piezo_overlay_plot.addItem(self.phase_zero_line)
        self.piezo_overlay_plot.setMinimumHeight(240)

        self.trend_overlay_plot = pg.PlotWidget()
        self.trend_overlay_plot.setBackground("w")
        self.trend_overlay_plot.showGrid(x=True, y=True, alpha=0.2)
        self.trend_overlay_plot.setLabel("bottom", "Live Time", units="s")
        self.trend_overlay_plot.setLabel("left", "Normalized", units="a.u.")
        self.trend_overlay_plot.setTitle("Trend Overlay: Normalized |Z| and |Y|")
        self.trend_y_norm_curve = self.trend_overlay_plot.plot([], [], pen=pg.mkPen((180, 120, 30), width=2))
        self.trend_z_norm_curve = self.trend_overlay_plot.plot([], [], pen=pg.mkPen((120, 80, 180), width=2))
        self.trend_overlay_plot.setXLink(self.piezo_overlay_plot)
        self._configure_plot_performance(self.trend_overlay_plot)
        self.trend_overlay_plot.setMinimumHeight(200)

        self.nyquist_plot = pg.PlotWidget()
        self.nyquist_plot.setBackground("w")
        self.nyquist_plot.showGrid(x=True, y=True, alpha=0.2)
        self.nyquist_plot.setLabel("bottom", "G", units="mS")
        self.nyquist_plot.setLabel("left", "B", units="mS")
        self.nyquist_plot.setTitle("Nyquist: B vs G")
        self.nyquist_curve = self.nyquist_plot.plot([], [], pen=pg.mkPen((40, 40, 40), width=1.5), symbol="o", symbolSize=4)
        self._configure_plot_performance(self.nyquist_plot)
        self.nyquist_plot.setMinimumHeight(200)

        self.admittance_freq_plot = pg.PlotWidget()
        self.admittance_freq_plot.setBackground("w")
        self.admittance_freq_plot.showGrid(x=True, y=True, alpha=0.2)
        self.admittance_freq_plot.setLabel("bottom", "Frequency", units="Hz")
        self.admittance_freq_plot.setLabel("left", "|Y|", units="mS")
        self.admittance_freq_plot.setTitle("Admittance Magnitude vs Frequency")
        self.admittance_freq_curve = self.admittance_freq_plot.plot(
            [],
            [],
            pen=pg.mkPen((180, 120, 30), width=1.5),
            symbol="o",
            symbolSize=4,
        )
        self.admittance_freq_plot.getPlotItem().setDownsampling(auto=False)
        self.admittance_freq_plot.getPlotItem().setClipToView(True)
        self.admittance_freq_plot.setMinimumHeight(200)

        self.advanced_tabs = QtWidgets.QTabWidget()
        self.advanced_tabs.setTabPosition(QtWidgets.QTabWidget.North)

        trend_tab = QtWidgets.QWidget()
        trend_layout = QtWidgets.QVBoxLayout(trend_tab)
        trend_layout.setContentsMargins(0, 0, 0, 0)
        trend_layout.addWidget(self.trend_overlay_plot)
        self.advanced_tab_idx_trend = self.advanced_tabs.addTab(trend_tab, "Trend")

        nyquist_tab = QtWidgets.QWidget()
        nyquist_layout = QtWidgets.QVBoxLayout(nyquist_tab)
        nyquist_layout.setContentsMargins(0, 0, 0, 0)
        nyquist_layout.addWidget(self.nyquist_plot)
        self.advanced_tab_idx_nyquist = self.advanced_tabs.addTab(nyquist_tab, "Nyquist")

        admittance_freq_tab = QtWidgets.QWidget()
        admittance_freq_layout = QtWidgets.QVBoxLayout(admittance_freq_tab)
        admittance_freq_layout.setContentsMargins(0, 0, 0, 0)
        admittance_freq_layout.addWidget(self.admittance_freq_plot)
        self.advanced_tab_idx_admittance_freq = self.advanced_tabs.addTab(admittance_freq_tab, "|Y| vs f")
        self.advanced_tabs.setMinimumHeight(220)

        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plot_splitter.setChildrenCollapsible(False)
        self.plot_splitter.setHandleWidth(10)
        self.plot_splitter.addWidget(self.waveform_overlay_plot)
        self.plot_splitter.addWidget(self.piezo_overlay_plot)
        self.plot_splitter.addWidget(self.advanced_tabs)
        self.plot_splitter.setStretchFactor(0, 4)
        self.plot_splitter.setStretchFactor(1, 4)
        self.plot_splitter.setStretchFactor(2, 2)
        self.plot_splitter.setSizes([360, 360, 260])
        live_layout.addWidget(self.plot_splitter, stretch=1)

        self._apply_plot_visibility()

        help_tab = QtWidgets.QWidget()
        help_layout = QtWidgets.QVBoxLayout(help_tab)
        self.help_text = QtWidgets.QTextBrowser()
        self.help_text.setReadOnly(True)
        self.help_text.setHtml(self._help_html())
        help_layout.addWidget(self.help_text)
        self.tabs.addTab(help_tab, "Help")

    def _create_kpi_card(self, title, unit, accent_color):
        card = QtWidgets.QFrame()
        card.setFrameShape(QtWidgets.QFrame.StyledPanel)
        card.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        card.setMinimumHeight(74)
        card.setStyleSheet(
            "QFrame {"
            "background-color: #f7f9fc;"
            "border: 1px solid #d7deea;"
            "border-radius: 8px;"
            "}"
        )
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(8, 6, 8, 6)
        card_layout.setSpacing(4)

        title_label = QtWidgets.QLabel(f"{title} [{unit}]")
        title_label.setStyleSheet("color: #4b5563; font-size: 11px; font-weight: 600;")
        value_label = QtWidgets.QLabel("--")
        value_label.setStyleSheet(f"color: {accent_color}; font-size: 17px; font-weight: 700;")
        value_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        value_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        value_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        value_label.setMinimumHeight(24)

        card_layout.addWidget(title_label)
        card_layout.addWidget(value_label)
        return card, value_label

    def _relayout_kpi_cards(self):
        if not hasattr(self, "kpi_grid") or not hasattr(self, "kpi_specs"):
            return

        width = int(self.left_scroll.viewport().width()) if hasattr(self, "left_scroll") else int(self.width())
        columns = 2 if width >= 430 else 1

        while self.kpi_grid.count():
            item = self.kpi_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        for c in range(max(columns, 1)):
            self.kpi_grid.setColumnStretch(c, 1)

        for idx, (key, _title, _unit, _color) in enumerate(self.kpi_specs):
            row = idx // columns
            col = idx % columns
            self.kpi_grid.addWidget(self.kpi_cards[key], row, col)

    def _configure_plot_performance(self, plot_widget):
        plot_item = plot_widget.getPlotItem()
        plot_item.setClipToView(True)
        plot_item.setDownsampling(auto=True, mode="peak")

    def _create_dual_axis_plot(
        self,
        title,
        x_label,
        x_units,
        left_label,
        left_units,
        right_label,
        right_units,
        left_pen,
        right_pen,
    ):
        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setTitle(title)
        plot.setLabel("bottom", x_label, units=x_units)
        plot.setLabel("left", left_label, units=left_units)

        plot_item = plot.getPlotItem()
        plot_item.showAxis("right")
        plot_item.getAxis("right").setLabel(right_label, units=right_units)

        right_vb = pg.ViewBox()
        plot_item.scene().addItem(right_vb)
        plot_item.getAxis("right").linkToView(right_vb)
        right_vb.setXLink(plot_item.vb)

        left_curve = plot.plot([], [], pen=left_pen)
        right_curve = pg.PlotDataItem([], [], pen=right_pen)
        right_vb.addItem(right_curve)

        def update_views():
            right_vb.setGeometry(plot_item.vb.sceneBoundingRect())
            right_vb.linkedViewChanged(plot_item.vb, right_vb.XAxis)

        plot_item.vb.sigResized.connect(update_views)
        update_views()
        return plot, left_curve, right_curve, right_vb

    def _maybe_autorange(self):
        now = time.monotonic()
        if (now - self.last_autorange_ts) < float(AUTO_RANGE_INTERVAL_S):
            return
        self.last_autorange_ts = now

        if self.waveform_overlay_plot.isVisible():
            self.waveform_overlay_plot.getPlotItem().autoRange()
            self.waveform_overlay_right_vb.autoRange()
        if self.piezo_overlay_plot.isVisible():
            self.piezo_overlay_plot.getPlotItem().autoRange()
            self.piezo_overlay_right_vb.autoRange()
        if self.trend_overlay_plot.isVisible():
            self.trend_overlay_plot.getPlotItem().autoRange()
        if self.nyquist_plot.isVisible():
            self.nyquist_plot.getPlotItem().autoRange()
        if self.admittance_freq_plot.isVisible():
            self.admittance_freq_plot.getPlotItem().autoRange()

    def _connect_scope_blocking(self):
        if self.connected:
            return

        self.rm = pyvisa.ResourceManager()
        self.scope = self.rm.open_resource(RESOURCE)
        self.scope.timeout = 5000
        self.scope.chunk_size = 1024 * 1024

        idn = self.scope.query("*IDN?").strip()
        print("Connected to:", idn)

        try:
            self.scope.write("HEADer 0")
        except Exception:
            pass

        self._setup_waveform_transfer()
        self.waveform_preamble_cache = {}
        self.last_data_source = None
        self.last_data_stop_points = None
        self.last_scope_freq_query_ts = 0.0
        self.cached_scope_freq_hz = np.nan

        self.connected = True
        print("Scope connected/configured.")

    def _setup_waveform_transfer(self):
        try:
            self.scope.write("DATa:ENCdg RIBinary")
        except Exception:
            self.scope.write("DATa:ENCdg SRIBinary")
        self.scope.write("DATa:WIDth 1")
        self.scope.write("DATa:STARt 1")

    def _set_data_source(self, channel):
        if self.last_data_source == channel:
            return
        self.scope.write(f"DATa:SOUrce {channel}")
        self.last_data_source = channel

    def _get_preamble_cached(self, channel, force_refresh=False):
        now = time.monotonic()
        cached = self.waveform_preamble_cache.get(channel)
        if (not force_refresh) and cached is not None:
            if (now - float(cached["ts"])) < float(PREAMBLE_REFRESH_S):
                return cached["vals"]

        self._set_data_source(channel)
        try:
            xincr, xzero, ymult, yzero, yoff, nr_pt = self._query_waveform_preamble("WFMOutpre")
        except Exception:
            xincr, xzero, ymult, yzero, yoff, nr_pt = self._query_waveform_preamble("WFMPRE")

        stop_points = POINTS
        if nr_pt is not None and nr_pt > 0:
            stop_points = min(POINTS, nr_pt)

        vals = (xincr, xzero, ymult, yzero, yoff, stop_points)
        self.waveform_preamble_cache[channel] = {"ts": now, "vals": vals}
        return vals

    def _query_waveform_preamble(self, prefix):
        xincr = float(self.scope.query(f"{prefix}:XINcr?").strip())
        xzero = float(self.scope.query(f"{prefix}:XZEro?").strip())
        ymult = float(self.scope.query(f"{prefix}:YMUlt?").strip())
        yzero = float(self.scope.query(f"{prefix}:YZEro?").strip())
        yoff = float(self.scope.query(f"{prefix}:YOFf?").strip())
        nr_pt = None
        for key in ("NR_Pt", "NRPT"):
            try:
                nr_pt = int(float(self.scope.query(f"{prefix}:{key}?").strip()))
                break
            except Exception:
                continue
        return xincr, xzero, ymult, yzero, yoff, nr_pt

    def fetch_waveform(self, channel):
        self._set_data_source(channel)
        xincr, xzero, ymult, yzero, yoff, stop_points = self._get_preamble_cached(channel)
        if self.last_data_stop_points != stop_points:
            self.scope.write(f"DATa:STOP {stop_points}")
            self.last_data_stop_points = stop_points

        raw = self.scope.query_binary_values(
            "CURVE?",
            datatype="b",
            container=np.array,
            is_big_endian=True,
        )

        # Retry once with refreshed preamble in case scope settings changed.
        if raw.size <= 0:
            xincr, xzero, ymult, yzero, yoff, stop_points = self._get_preamble_cached(channel, force_refresh=True)
            if self.last_data_stop_points != stop_points:
                self.scope.write(f"DATa:STOP {stop_points}")
                self.last_data_stop_points = stop_points
            raw = self.scope.query_binary_values(
                "CURVE?",
                datatype="b",
                container=np.array,
                is_big_endian=True,
            )

        values = (raw.astype(float) - yoff) * ymult + yzero
        # Use relative time for plotting so the trace is always visible and easy to read.
        times = np.arange(len(raw), dtype=float) * xincr
        return times, values

    def _query_scope_frequency_once(self):
        # Try common Tektronix SCPI immediate measurement paths.
        attempts = [
            ("MEASUrement:IMMed:TYPe FREQuency", "MEASUrement:IMMed:SOUrce1 CH1", "MEASUrement:IMMed:VALue?"),
            ("MEASUrement:IMMed:TYPE FREQUENCY", "MEASUrement:IMMed:SOURCE1 CH1", "MEASUrement:IMMed:VALUE?"),
            ("MEASUrement:MEAS1:TYPe FREQuency", "MEASUrement:MEAS1:SOUrce1 CH1", "MEASUrement:MEAS1:VALue?"),
            ("MEASUrement:MEAS1:TYPE FREQUENCY", "MEASUrement:MEAS1:SOURCE1 CH1", "MEASUrement:MEAS1:VALUE?"),
        ]
        for cmd_type, cmd_source, cmd_query in attempts:
            try:
                self.scope.write(cmd_type)
                self.scope.write(cmd_source)
                raw = self.scope.query(cmd_query).strip()
                val = float(raw)
                if np.isfinite(val) and val > 0.0:
                    return val
            except Exception:
                continue
        return np.nan

    def fetch_scope_ch1_frequency_hz(self):
        now = time.monotonic()
        if (now - float(self.last_scope_freq_query_ts)) < float(SCOPE_FREQ_QUERY_REFRESH_S):
            return float(self.cached_scope_freq_hz)
        self.last_scope_freq_query_ts = now
        try:
            freq_hz = self._query_scope_frequency_once()
        except Exception:
            freq_hz = np.nan
        if np.isfinite(freq_hz) and freq_hz > 0.0:
            self.cached_scope_freq_hz = float(freq_hz)
        return float(self.cached_scope_freq_hz) if np.isfinite(self.cached_scope_freq_hz) else np.nan

    def live_start(self):
        if self.busy_live:
            return
        if self.live_running:
            self.set_status("Live already running.")
            return

        self.busy_live = True
        self.live_btn.setEnabled(False)
        self.stop_live_btn.setEnabled(False)
        self.acquire_btn.setEnabled(False)
        self.set_status("Connecting to scope... (GUI stays responsive)")

        self.live_thread = QtCore.QThread(self)
        self.live_worker = LiveStartWorker(self)
        self.live_worker.moveToThread(self.live_thread)
        self.live_thread.started.connect(self.live_worker.run)
        self.live_worker.finished.connect(self._on_live_start_finished)
        self.live_worker.error.connect(self._on_live_start_error)
        self.live_worker.finished.connect(self.live_thread.quit)
        self.live_worker.error.connect(self.live_thread.quit)
        self.live_thread.finished.connect(self.live_worker.deleteLater)
        self.live_thread.finished.connect(self.live_thread.deleteLater)
        self.live_thread.start()

    def _on_live_start_finished(self, msg):
        self.busy_live = False
        self.live_btn.setEnabled(True)
        print("Live start sent to scope.")
        self._start_live_polling()
        self.set_status(msg + " Live plots running.")

    def _on_live_start_error(self, err):
        self.busy_live = False
        self.live_btn.setEnabled(True)
        self.stop_live_btn.setEnabled(False)
        self.acquire_btn.setEnabled(True)
        self.set_status(f"Error: {err}")
        QtWidgets.QMessageBox.critical(self, "Scope error", err)

    def toggle_acquire(self):
        if self.busy_live:
            self.set_status("Wait for Live Start connection to finish.")
            return
        if not self.live_running:
            self.set_status("Press Live Start first.")
            return
        if self.recording:
            self.stop_acquisition()
            return

        self.recording = True
        self.record_start_time = time.time()
        self.acquire_btn.setText("Stop Acquisition")
        self.live_btn.setEnabled(False)
        self.stop_live_btn.setEnabled(True)
        self.elapsed_checkbox.setEnabled(False)
        self.duration_spin.setEnabled(False)
        self.set_status("Recording current live waveform stream...")

    def stop_acquisition(self):
        if not self.recording:
            return
        self.recording = False
        self.record_start_time = None
        self.acquire_btn.setText("Acquire")
        self.live_btn.setEnabled(False if self.live_running else True)
        self.stop_live_btn.setEnabled(True if self.live_running else False)
        self.elapsed_checkbox.setEnabled(True)
        self.duration_spin.setEnabled(True)
        self.set_status("Acquisition finished.")
        print("Acquisition finished.")
        self.save_last_waveforms()

    def _estimate_dt(self, t):
        t = np.asarray(t, dtype=float)
        if t.size < 2:
            return np.nan
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            return np.nan
        return dt

    def _current_cutoff_hz(self):
        try:
            return float(self.cutoff_spin.value())
        except Exception:
            return float(CH2_LOW_PASS_CUTOFF_HZ)

    def _current_filter_type(self):
        try:
            value = self.filter_type_combo.currentData()
            if value is None:
                return CURRENT_FILTER_TYPE_DEFAULT
            return str(value)
        except Exception:
            return CURRENT_FILTER_TYPE_DEFAULT

    def _current_filter_label(self):
        mapping = {
            "lpf_1pole": "1-pole LPF",
            "butterworth": "Butterworth",
            "moving_average": "Moving Average",
            "none": "None (Raw)",
        }
        return mapping.get(self._current_filter_type(), "1-pole LPF")

    def _current_filter_order(self):
        try:
            val = int(self.filter_order_spin.value())
            return 2 if val <= 2 else 4
        except Exception:
            return int(CURRENT_FILTER_BUTTER_ORDER_DEFAULT)

    def _current_poll_time_ms(self):
        try:
            return int(self.poll_time_spin.value())
        except Exception:
            return int(POLL_TIME_MS)

    def _dynamic_cutoff_enabled(self):
        try:
            return bool(self.dynamic_cutoff_cb.isChecked())
        except Exception:
            return bool(DEFAULT_DYNAMIC_CUTOFF_ENABLED)

    def _dynamic_cutoff_multiplier(self):
        try:
            return float(self.dynamic_cutoff_k_spin.value())
        except Exception:
            return float(DEFAULT_DYNAMIC_CUTOFF_MULTIPLIER)

    def _dynamic_cutoff_min_hz(self):
        try:
            return float(self.dynamic_cutoff_min_spin.value())
        except Exception:
            return float(DEFAULT_DYNAMIC_CUTOFF_MIN_HZ)

    def _dynamic_cutoff_max_hz(self):
        try:
            return float(self.dynamic_cutoff_max_spin.value())
        except Exception:
            return float(DEFAULT_DYNAMIC_CUTOFF_MAX_HZ)

    def _use_scope_frequency(self):
        try:
            return bool(self.use_scope_freq_cb.isChecked())
        except Exception:
            return False

    def _phase_compensation_enabled(self):
        try:
            return bool(self.phase_compensation_cb.isChecked())
        except Exception:
            return False

    def _update_config_label(self):
        filter_type = self._current_filter_type()
        filter_label = self._current_filter_label()
        if filter_type == "none":
            filter_text = filter_label
        elif filter_type == "butterworth":
            filter_text = f"{filter_label}, order {self._current_filter_order()}, cutoff {self._current_cutoff_hz():.1f} Hz"
        else:
            filter_text = f"{filter_label}, cutoff {self._current_cutoff_hz():.1f} Hz"
        freq_src = "Scope CH1" if self._use_scope_frequency() else "FFT dominant bin"
        phase_mode = "compensated" if self._phase_compensation_enabled() else "raw"
        cutoff_mode = (
            f"dynamic (k={self._dynamic_cutoff_multiplier():.2f}, {self._dynamic_cutoff_min_hz():.1f}-{self._dynamic_cutoff_max_hz():.1f} Hz)"
            if self._dynamic_cutoff_enabled()
            else "fixed"
        )
        self.config_label.setText(
            f"CH1 = Voltage (V), CH2 = Current (mA), Filter: {filter_text}, Cutoff mode: {cutoff_mode}, Poll: {self._current_poll_time_ms()} ms, Freq src: {freq_src}, Phase: {phase_mode}"
        )

    def _on_cutoff_changed(self, _value):
        self._update_config_label()

    def _on_filter_type_changed(self, _index):
        filter_type = self._current_filter_type()
        use_cutoff = filter_type != "none"
        self.cutoff_spin.setEnabled(use_cutoff)
        self.filter_order_spin.setEnabled(filter_type == "butterworth")
        self._update_config_label()

    def _on_filter_order_changed(self, _value):
        self._update_config_label()

    def _on_poll_time_changed(self, value):
        try:
            if self.live_worker is not None:
                self.live_worker.poll_time_ms = int(value)
        except Exception:
            pass
        self._update_config_label()

    def _on_scope_freq_toggled(self, checked):
        try:
            if self.live_worker is not None:
                self.live_worker.use_scope_frequency = bool(checked)
        except Exception:
            pass
        self._update_config_label()

    def _on_dynamic_cutoff_toggled(self, checked):
        enabled = bool(checked)
        self.dynamic_cutoff_k_spin.setEnabled(enabled)
        self.dynamic_cutoff_min_spin.setEnabled(enabled)
        self.dynamic_cutoff_max_spin.setEnabled(enabled)
        self._update_config_label()

    def _on_dynamic_cutoff_params_changed(self, _value):
        self._update_config_label()

    def _on_phase_compensation_toggled(self, _checked):
        self._update_config_label()

    def _set_view_preset_basic(self):
        self.show_waveform_overlay_cb.setChecked(True)
        self.show_piezo_overlay_cb.setChecked(False)
        self.show_trend_overlay_cb.setChecked(False)
        self.show_nyquist_cb.setChecked(False)
        self.show_y_vs_f_cb.setChecked(False)

    def _set_view_preset_piezo(self):
        self.show_waveform_overlay_cb.setChecked(True)
        self.show_piezo_overlay_cb.setChecked(True)
        self.show_trend_overlay_cb.setChecked(False)
        self.show_nyquist_cb.setChecked(False)
        self.show_y_vs_f_cb.setChecked(False)

    def _set_view_preset_advanced(self):
        self.show_waveform_overlay_cb.setChecked(True)
        self.show_piezo_overlay_cb.setChecked(True)
        self.show_trend_overlay_cb.setChecked(True)
        self.show_nyquist_cb.setChecked(True)
        self.show_y_vs_f_cb.setChecked(True)

    def _apply_plot_visibility(self):
        self.waveform_overlay_plot.setVisible(self.show_waveform_overlay_cb.isChecked())
        self.piezo_overlay_plot.setVisible(self.show_piezo_overlay_cb.isChecked())
        show_trend = self.show_trend_overlay_cb.isChecked()
        show_nyquist = self.show_nyquist_cb.isChecked()
        show_y_vs_f = self.show_y_vs_f_cb.isChecked()

        self.advanced_tabs.setTabEnabled(self.advanced_tab_idx_trend, show_trend)
        self.advanced_tabs.setTabEnabled(self.advanced_tab_idx_nyquist, show_nyquist)
        self.advanced_tabs.setTabEnabled(self.advanced_tab_idx_admittance_freq, show_y_vs_f)
        self.advanced_tabs.setVisible(show_trend or show_nyquist or show_y_vs_f)

        if show_trend:
            self.advanced_tabs.setCurrentIndex(self.advanced_tab_idx_trend)
        elif show_nyquist:
            self.advanced_tabs.setCurrentIndex(self.advanced_tab_idx_nyquist)
        elif show_y_vs_f:
            self.advanced_tabs.setCurrentIndex(self.advanced_tab_idx_admittance_freq)
        self.last_autorange_ts = 0.0
        self._maybe_autorange()

    def _help_html(self):
        return """
        <h3>Electrical Metrics (Piezo)</h3>
        <p><b>Signals used</b></p>
        <ul>
          <li>CH1: Voltage, V(t) [V]</li>
          <li>CH2: Current, I(t) [mA] after scaling + selected filter (1-pole LPF / Butterworth 2nd-4th / Moving Average / None)</li>
        </ul>
        <p><b>Metrics computed at each live update</b></p>
        <ul>
          <li>If 1-pole LPF is selected: y[n] = y[n-1] + alpha * (x[n] - y[n-1]), alpha = dt / (RC + dt), RC = 1/(2*pi*fc)</li>
          <li>If Butterworth is selected: digital IIR Butterworth low-pass (order 2 or 4), cutoff set by LPF Cutoff [Hz]</li>
          <li>If Moving Average is selected: CH2 is averaged with a window sized from the cutoff setting</li>
          <li>Dynamic cutoff option: fc = clamp(k * f, fc_min, fc_max), where f comes from selected frequency source</li>
          <li>Vrms = sqrt(mean(V(t)^2))</li>
          <li>Irms = sqrt(mean(I(t)^2)), with I in Ampere</li>
          <li>|Y| = Irms / Vrms [S]</li>
          <li>|Z| = Vrms / Irms [ohm]</li>
          <li>f source selectable: FFT dominant bin or direct scope CH1 frequency measurement</li>
          <li>phi = phase(I) - phase(V) [deg], estimated at selected frequency source</li>
          <li>Phase Compensation option subtracts estimated filter phase at f from measured phase</li>
          <li>G = |Y| * cos(phi) [S]</li>
          <li>B = |Y| * sin(phi) [S]</li>
        </ul>
        <p><b>Notes</b></p>
        <ul>
          <li>The phase method assumes a dominant tone in the waveform.</li>
          <li>Displayed |Y| history is shown in mS for readability.</li>
          <li>Use the View toggles to avoid overcrowding.</li>
        </ul>
        <p><b>Overlay set</b></p>
        <ul>
          <li>V + I: waveform overlay with dual Y axis.</li>
          <li>Phase + |Y|: live piezo behavior overlay with dual Y axis.</li>
          <li>Norm |Z| + |Y|: trend comparison on normalized scale [0..1].</li>
          <li>Nyquist: B vs G path in the admittance plane.</li>
          <li>|Y| vs f: admittance magnitude versus dominant frequency from FFT.</li>
          <li>Trend, Nyquist and |Y| vs f share the same panel via tabs to reduce clutter.</li>
          <li>Plot panels are in a vertical splitter: drag separators to resize visible plots.</li>
          <li>Controls and KPI cards are grouped in the left dashboard column.</li>
          <li>KPI cards show latest Phase, |Y|, |Z|, G, B values.</li>
          <li>Live rendering uses latest-frame strategy to keep the GUI responsive under load.</li>
        </ul>
        """

    def _convert_ch2_to_current_ma(self, ch2_values):
        return np.asarray(ch2_values, dtype=float) * float(CURRENT_SCALE_MA_PER_V)

    def _low_pass_filter(self, signal, dt, cutoff_hz):
        x = np.asarray(signal, dtype=float)
        if x.size == 0:
            return x
        if not np.isfinite(dt) or dt <= 0 or cutoff_hz <= 0:
            return x.copy()
        rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
        alpha = dt / (rc + dt)
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, x.size):
            y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
        return y

    def _compute_electrical_metrics(self, v1_volt, i2_ma, phase_deg):
        v = np.asarray(v1_volt, dtype=float)
        i_ma = np.asarray(i2_ma, dtype=float)
        n = min(v.size, i_ma.size)
        if n < 8:
            return np.nan, np.nan, np.nan, np.nan

        v = v[:n]
        i_a = i_ma[:n] / 1000.0
        mask = np.isfinite(v) & np.isfinite(i_a)
        if np.count_nonzero(mask) < 8:
            return np.nan, np.nan, np.nan, np.nan

        v = v[mask]
        i_a = i_a[mask]
        vrms = float(np.sqrt(np.mean(v * v)))
        irms = float(np.sqrt(np.mean(i_a * i_a)))
        if not np.isfinite(vrms) or not np.isfinite(irms) or vrms <= 0.0 or irms <= 0.0:
            return np.nan, np.nan, np.nan, np.nan

        y_mag_s = irms / vrms
        z_mag_ohm = vrms / irms

        if not np.isfinite(phase_deg):
            return y_mag_s, z_mag_ohm, np.nan, np.nan

        phi_rad = float(np.deg2rad(phase_deg))
        g_s = y_mag_s * np.cos(phi_rad)
        b_s = y_mag_s * np.sin(phi_rad)
        return float(y_mag_s), float(z_mag_ohm), float(g_s), float(b_s)

    def _normalize_series(self, arr):
        x = np.asarray(arr, dtype=float).copy()
        if x.size == 0:
            return x
        mask = np.isfinite(x)
        if not np.any(mask):
            x[:] = np.nan
            return x
        x_valid = x[mask]
        x_min = float(np.min(x_valid))
        x_max = float(np.max(x_valid))
        span = x_max - x_min
        if span <= 0.0 or not np.isfinite(span):
            x[:] = np.nan
            x[mask] = 0.0
            return x
        x[mask] = (x_valid - x_min) / span
        x[~mask] = np.nan
        return x

    def _fmt_metric(self, value, scale=1.0, fmt=".3f", signed=False):
        if not np.isfinite(value):
            return "n/a"
        v = float(value) * float(scale)
        if signed:
            return f"{v:+{fmt}}"
        return f"{v:{fmt}}"

    def _update_metrics_bar(self, phase_deg, y_mag_s, z_mag_ohm, g_s, b_s):
        phase_val = self._fmt_metric(phase_deg, fmt=".2f", signed=True)
        y_val = self._fmt_metric(y_mag_s, scale=1000.0, fmt=".3f")
        z_val = self._fmt_metric(z_mag_ohm, fmt=".2f")
        g_val = self._fmt_metric(g_s, scale=1000.0, fmt=".3f")
        b_val = self._fmt_metric(b_s, scale=1000.0, fmt=".3f")

        self.kpi_value_labels["phase"].setText(phase_val)
        self.kpi_value_labels["y"].setText(y_val)
        self.kpi_value_labels["z"].setText(z_val)
        self.kpi_value_labels["g"].setText(g_val)
        self.kpi_value_labels["b"].setText(b_val)

        self.metrics_label.setText(
            f"KPI summary: Phase={phase_val} deg | |Y|={y_val} mS | |Z|={z_val} ohm | G={g_val} mS | B={b_val} mS"
        )

    def _estimate_phase_deg(self, v1, i2, t1, t2):
        x = np.asarray(v1, dtype=float)
        y = np.asarray(i2, dtype=float)
        n = min(x.size, y.size)
        if n < 32:
            return np.nan

        dt1 = self._estimate_dt(t1)
        dt2 = self._estimate_dt(t2)
        if not np.isfinite(dt1) or not np.isfinite(dt2):
            return np.nan

        x = x[:n]
        y = y[:n]

        x = x - np.mean(x)
        y = y - np.mean(y)

        if np.allclose(x, 0.0) or np.allclose(y, 0.0):
            return np.nan

        window = np.hanning(n)
        xw = x * window
        yw = y * window

        xf = np.fft.rfft(xw)
        yf = np.fft.rfft(yw)
        if xf.size < 3 or yf.size < 3:
            return np.nan

        mag = np.abs(xf)
        mag[0] = 0.0
        k = int(np.argmax(mag))
        if k <= 0 or k >= yf.size:
            return np.nan

        phase_rad = np.angle(yf[k]) - np.angle(xf[k])
        phase_rad = (phase_rad + np.pi) % (2.0 * np.pi) - np.pi
        return float(np.degrees(phase_rad))

    def _append_live_history(self, phase_deg, freq_hz, y_mag_s, z_mag_ohm, g_s, b_s):
        now = time.time()
        if self.live_plot_zero_time is None:
            self.live_plot_zero_time = now
        t_live = now - self.live_plot_zero_time

        self.phase_times.append(float(t_live))
        self.phase_values.append(float(phase_deg) if np.isfinite(phase_deg) else np.nan)
        self.admittance_values_s.append(float(y_mag_s) if np.isfinite(y_mag_s) else np.nan)
        self.impedance_values_ohm.append(float(z_mag_ohm) if np.isfinite(z_mag_ohm) else np.nan)
        self.conductance_values_s.append(float(g_s) if np.isfinite(g_s) else np.nan)
        self.susceptance_values_s.append(float(b_s) if np.isfinite(b_s) else np.nan)
        self.admittance_freq_hz.append(float(freq_hz) if np.isfinite(freq_hz) else np.nan)
        self.admittance_freq_values_s.append(float(y_mag_s) if np.isfinite(y_mag_s) else np.nan)

        t_min = t_live - float(PHASE_HISTORY_SECONDS)
        while self.phase_times and self.phase_times[0] < t_min:
            self.phase_times.pop(0)
            self.phase_values.pop(0)
            self.admittance_values_s.pop(0)
            self.impedance_values_ohm.pop(0)
            self.conductance_values_s.pop(0)
            self.susceptance_values_s.pop(0)

        if len(self.admittance_freq_hz) > int(ADMITTANCE_FREQ_MAX_POINTS):
            extra = len(self.admittance_freq_hz) - int(ADMITTANCE_FREQ_MAX_POINTS)
            del self.admittance_freq_hz[:extra]
            del self.admittance_freq_values_s[:extra]

        t_hist = np.asarray(self.phase_times, dtype=float)
        phase_arr = np.asarray(self.phase_values, dtype=float)
        y_ms_arr = np.asarray(self.admittance_values_s, dtype=float) * 1000.0
        z_ohm_arr = np.asarray(self.impedance_values_ohm, dtype=float)
        g_ms_arr = np.asarray(self.conductance_values_s, dtype=float) * 1000.0
        b_ms_arr = np.asarray(self.susceptance_values_s, dtype=float) * 1000.0

        self.piezo_phase_curve.setData(t_hist, phase_arr)
        self.piezo_admittance_curve.setData(t_hist, y_ms_arr)

        y_norm = self._normalize_series(y_ms_arr)
        z_norm = self._normalize_series(z_ohm_arr)
        self.trend_y_norm_curve.setData(t_hist, y_norm)
        self.trend_z_norm_curve.setData(t_hist, z_norm)

        self.nyquist_curve.setData(g_ms_arr, b_ms_arr)
        f_arr = np.asarray(self.admittance_freq_hz, dtype=float)
        y_f_ms_arr = np.asarray(self.admittance_freq_values_s, dtype=float) * 1000.0
        mask = np.isfinite(f_arr) & np.isfinite(y_f_ms_arr)
        fx = f_arr[mask]
        fy = y_f_ms_arr[mask]
        if fx.size > 1:
            span = float(np.nanmax(fx) - np.nanmin(fx))
            # Guard against pyqtgraph auto-downsample edge case when x-span is ~0.
            if not np.isfinite(span) or span <= 0.0:
                fx = fx[-1:]
                fy = fy[-1:]
        self.admittance_freq_curve.setData(fx, fy)
        self._maybe_autorange()

    def _on_waveform_ready(self, payload):
        if not self.live_running or self.processing_worker is None:
            return
        if isinstance(payload, dict) and "waveforms" in payload:
            waveforms = payload.get("waveforms", {})
            scope_freq_hz = float(payload.get("scope_freq_hz", np.nan))
        else:
            waveforms = payload if isinstance(payload, dict) else {}
            scope_freq_hz = np.nan
        self.process_payload.emit(
            {
                "waveforms": waveforms,
                "cutoff_hz": self._current_cutoff_hz(),
                "filter_type": self._current_filter_type(),
                "filter_order": self._current_filter_order(),
                "use_scope_frequency": self._use_scope_frequency(),
                "scope_freq_hz": scope_freq_hz,
                "phase_compensation_enabled": self._phase_compensation_enabled(),
                "dynamic_cutoff_enabled": self._dynamic_cutoff_enabled(),
                "dynamic_cutoff_multiplier": self._dynamic_cutoff_multiplier(),
                "dynamic_cutoff_min_hz": self._dynamic_cutoff_min_hz(),
                "dynamic_cutoff_max_hz": self._dynamic_cutoff_max_hz(),
                "current_scale_ma_per_v": float(CURRENT_SCALE_MA_PER_V),
            }
        )

    def _on_processed_payload(self, processed):
        self.pending_processed_payload = processed

    def _consume_pending_processed_payload(self):
        processed = self.pending_processed_payload
        if processed is None:
            return
        self.pending_processed_payload = None
        self._render_processed_payload(processed)

    def _render_processed_payload(self, processed):
        if not processed.get("ok", False):
            self.set_status(processed.get("status", "Processing returned no data."))
            return

        t1 = processed["t1"]
        ch1_v = processed["ch1_v"]
        t2 = processed["t2"]
        ch2_current_ma_filtered = processed["ch2_current_ma_filtered"]
        phase_deg = processed["phase_deg"]
        phase_meas_deg = processed.get("phase_meas_deg", phase_deg)
        phase_filter_deg = processed.get("phase_filter_deg", np.nan)
        freq_hz = processed.get("freq_hz", np.nan)
        cutoff_hz_used = float(processed.get("cutoff_hz_used", self._current_cutoff_hz()))
        y_mag_s = processed["y_mag_s"]
        z_mag_ohm = processed["z_mag_ohm"]
        g_s = processed["g_s"]
        b_s = processed["b_s"]

        self.last_waveforms["CH1"] = (t1, ch1_v)
        self.last_waveforms["CH2"] = (t2, ch2_current_ma_filtered)

        self.waveform_voltage_curve.setData(t1, ch1_v)
        self.waveform_current_curve.setData(t2, ch2_current_ma_filtered)

        self._append_live_history(phase_deg, freq_hz, y_mag_s, z_mag_ohm, g_s, b_s)
        self._update_metrics_bar(phase_deg, y_mag_s, z_mag_ohm, g_s, b_s)

        f_text = "n/a" if not np.isfinite(freq_hz) else f"{freq_hz:.1f} Hz"
        phase_text = "n/a" if not np.isfinite(phase_deg) else f"{phase_deg:+.2f} deg"
        phase_filter_text = "n/a" if not np.isfinite(phase_filter_deg) else f"{phase_filter_deg:+.2f} deg"
        y_text = "n/a" if not np.isfinite(y_mag_s) else f"{(1000.0 * y_mag_s):.3f} mS"
        z_text = "n/a" if not np.isfinite(z_mag_ohm) else f"{z_mag_ohm:.2f} ohm"
        cutoff_text = "n/a" if not np.isfinite(cutoff_hz_used) else f"{cutoff_hz_used:.1f} Hz"
        if self._phase_compensation_enabled():
            phase_meas_text = "n/a" if not np.isfinite(phase_meas_deg) else f"{phase_meas_deg:+.2f} deg"
            phase_block = f"Phase: {phase_text} (raw {phase_meas_text}, filt {phase_filter_text})"
        else:
            phase_block = f"Phase: {phase_text}"
        status = (
            f"Live: CH1:{t1.size} pts, CH2:{t2.size} pts | f: {f_text} | "
            f"{phase_block} | |Y|: {y_text} | |Z|: {z_text} | fc: {cutoff_text}"
        )

        if self.recording:
            elapsed = time.time() - self.record_start_time
            status = f"Recording... {elapsed:.1f}s | " + status
            if self.elapsed_checkbox.isChecked() and elapsed >= float(self.duration_spin.value()):
                self.stop_acquisition()
                return

        self.set_status(status)

    def _start_live_polling(self):
        self.phase_times = []
        self.phase_values = []
        self.admittance_values_s = []
        self.impedance_values_ohm = []
        self.conductance_values_s = []
        self.susceptance_values_s = []
        self.admittance_freq_hz = []
        self.admittance_freq_values_s = []
        self.live_plot_zero_time = None
        self.waveform_preamble_cache = {}
        self.last_data_source = None
        self.last_data_stop_points = None
        self.last_scope_freq_query_ts = 0.0
        self.cached_scope_freq_hz = np.nan
        self.pending_processed_payload = None
        self.last_autorange_ts = 0.0
        self.waveform_voltage_curve.setData([], [])
        self.waveform_current_curve.setData([], [])
        self.piezo_phase_curve.setData([], [])
        self.piezo_admittance_curve.setData([], [])
        self.trend_y_norm_curve.setData([], [])
        self.trend_z_norm_curve.setData([], [])
        self.nyquist_curve.setData([], [])
        self.admittance_freq_curve.setData([], [])
        self._update_metrics_bar(np.nan, np.nan, np.nan, np.nan, np.nan)

        self.processing_thread = QtCore.QThread(self)
        self.processing_worker = ProcessingWorker()
        self.processing_worker.moveToThread(self.processing_thread)
        self.process_payload.connect(self.processing_worker.enqueue_job)
        self.processing_worker.processed.connect(self._on_processed_payload)
        self.processing_worker.error.connect(self._on_processing_error)
        self.processing_thread.finished.connect(self.processing_worker.deleteLater)
        self.processing_thread.finished.connect(self.processing_thread.deleteLater)
        self.processing_thread.start()

        self.live_thread = QtCore.QThread(self)
        self.live_worker = WaveformLiveWorker(
            self,
            self.selected_channels(),
            self._current_poll_time_ms(),
            use_scope_frequency=self._use_scope_frequency(),
        )
        self.live_worker.moveToThread(self.live_thread)
        self.live_thread.started.connect(self.live_worker.run)
        self.live_worker.waveform_ready.connect(self._on_waveform_ready)
        self.live_worker.finished.connect(self._on_live_polling_finished)
        self.live_worker.error.connect(self._on_live_polling_error)
        self.live_worker.finished.connect(self.live_thread.quit)
        self.live_worker.error.connect(self.live_thread.quit)
        self.live_thread.finished.connect(self.live_worker.deleteLater)
        self.live_thread.finished.connect(self.live_thread.deleteLater)
        self.live_thread.start()
        self.render_timer.start()
        self.live_running = True
        self.acquire_btn.setEnabled(True)
        self.stop_live_btn.setEnabled(True)
        self.live_btn.setEnabled(False)

    def stop_live(self):
        if not self.live_running:
            self.set_status("Live is not running.")
            return
        self.set_status("Stopping live...")
        if self.live_worker is not None:
            self.live_worker.stop()

    def _on_live_polling_finished(self, msg):
        self.live_running = False
        self.render_timer.stop()
        self.pending_processed_payload = None
        self._stop_processing_pipeline()
        self.live_worker = None
        self.live_thread = None
        if self.recording:
            self.stop_acquisition()
        self.acquire_btn.setEnabled(False)
        self.stop_live_btn.setEnabled(False)
        self.live_btn.setEnabled(True)
        self.set_status(msg)

    def _on_live_polling_error(self, err):
        self.live_running = False
        self.render_timer.stop()
        self.pending_processed_payload = None
        self._stop_processing_pipeline()
        self.live_worker = None
        self.live_thread = None
        if self.recording:
            self.recording = False
        self.acquire_btn.setText("Acquire")
        self.acquire_btn.setEnabled(False)
        self.stop_live_btn.setEnabled(False)
        self.live_btn.setEnabled(True)
        self.elapsed_checkbox.setEnabled(True)
        self.duration_spin.setEnabled(True)
        self.set_status(f"Acquisition error: {err}")
        QtWidgets.QMessageBox.critical(self, "Acquisition error", err)

    def _on_processing_error(self, err):
        self.set_status(f"Processing error: {err}")

    def _stop_processing_pipeline(self):
        try:
            if self.processing_worker is not None:
                self.process_payload.disconnect(self.processing_worker.enqueue_job)
        except Exception:
            pass
        try:
            if self.processing_worker is not None:
                self.processing_worker.stop()
                self.processing_worker.deleteLater()
        except Exception:
            pass
        try:
            if self.processing_thread is not None and self.processing_thread.isRunning():
                self.processing_thread.quit()
                self.processing_thread.wait(1000)
        except Exception:
            pass
        self.processing_worker = None
        self.processing_thread = None

    def save_last_waveforms(self):
        has_waveforms = any(self.last_waveforms[ch][0].size > 0 for ch in ("CH1", "CH2"))
        has_phase = len(self.phase_times) > 0
        has_metrics = len(self.admittance_values_s) > 0 or len(self.impedance_values_ohm) > 0
        if not has_waveforms and not has_phase and not has_metrics:
            self.set_status("No waveform data to save.")
            return

        base_path = Path(self.save_path)
        out_path = base_path if base_path.suffix else base_path.with_suffix(".csv")

        t_ch1, v_ch1 = self.last_waveforms["CH1"]
        t_ch2, i_ch2 = self.last_waveforms["CH2"]

        time_ref = None
        if t_ch1.size > 0:
            time_ref = t_ch1
        if t_ch2.size > 0 and (time_ref is None or t_ch2.size > time_ref.size):
            time_ref = t_ch2

        phase_t = np.asarray(self.phase_times, dtype=float)
        phase_v = np.asarray(self.phase_values, dtype=float)
        freq_hz = np.asarray(self.admittance_freq_hz, dtype=float)
        y_s = np.asarray(self.admittance_values_s, dtype=float)
        z_ohm = np.asarray(self.impedance_values_ohm, dtype=float)
        g_s = np.asarray(self.conductance_values_s, dtype=float)
        b_s = np.asarray(self.susceptance_values_s, dtype=float)

        n_rows = 0
        if time_ref is not None:
            n_rows = max(n_rows, int(time_ref.size))
        n_rows = max(n_rows, int(phase_t.size))
        n_rows = max(n_rows, int(freq_hz.size), int(y_s.size), int(z_ohm.size), int(g_s.size), int(b_s.size))

        if n_rows <= 0:
            self.set_status("No waveform data to save.")
            return

        time_arr = np.full(n_rows, np.nan, dtype=float)
        freq_arr = np.full(n_rows, np.nan, dtype=float)
        ch1_arr = np.full(n_rows, np.nan, dtype=float)
        ch2_arr = np.full(n_rows, np.nan, dtype=float)
        phase_t_arr = np.full(n_rows, np.nan, dtype=float)
        phase_v_arr = np.full(n_rows, np.nan, dtype=float)
        y_arr = np.full(n_rows, np.nan, dtype=float)
        z_arr = np.full(n_rows, np.nan, dtype=float)
        g_arr = np.full(n_rows, np.nan, dtype=float)
        b_arr = np.full(n_rows, np.nan, dtype=float)

        if time_ref is not None and time_ref.size > 0:
            nt = min(n_rows, time_ref.size)
            time_arr[:nt] = time_ref[:nt]

            if v_ch1.size > 0:
                n1 = min(nt, v_ch1.size)
                ch1_arr[:n1] = v_ch1[:n1]
            if i_ch2.size > 0:
                n2 = min(nt, i_ch2.size)
                ch2_arr[:n2] = i_ch2[:n2]

        if phase_t.size > 0:
            np_phase = min(n_rows, phase_t.size)
            phase_t_arr[:np_phase] = phase_t[:np_phase]
            phase_v_arr[:np_phase] = phase_v[:np_phase]
        if freq_hz.size > 0:
            n_freq = min(n_rows, freq_hz.size)
            freq_arr[:n_freq] = freq_hz[:n_freq]
        if y_s.size > 0:
            n_y = min(n_rows, y_s.size)
            y_arr[:n_y] = y_s[:n_y]
        if z_ohm.size > 0:
            n_z = min(n_rows, z_ohm.size)
            z_arr[:n_z] = z_ohm[:n_z]
        if g_s.size > 0:
            n_g = min(n_rows, g_s.size)
            g_arr[:n_g] = g_s[:n_g]
        if b_s.size > 0:
            n_b = min(n_rows, b_s.size)
            b_arr[:n_b] = b_s[:n_b]

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Time (s)",
                    "Frequency (Hz)",
                    "CH1 Voltage (V)",
                    "CH2 Current Filtered (mA)",
                    "Phase Live Time (s)",
                    "Phase(CH2-CH1) (deg)",
                    "|Y| (S)",
                    "|Y| (mS)",
                    "|Z| (ohm)",
                    "G (S)",
                    "B (S)",
                ]
            )
            for i in range(n_rows):
                row = [
                    "" if np.isnan(time_arr[i]) else float(time_arr[i]),
                    "" if np.isnan(freq_arr[i]) else float(freq_arr[i]),
                    "" if np.isnan(ch1_arr[i]) else float(ch1_arr[i]),
                    "" if np.isnan(ch2_arr[i]) else float(ch2_arr[i]),
                    "" if np.isnan(phase_t_arr[i]) else float(phase_t_arr[i]),
                    "" if np.isnan(phase_v_arr[i]) else float(phase_v_arr[i]),
                    "" if np.isnan(y_arr[i]) else float(y_arr[i]),
                    "" if np.isnan(y_arr[i]) else float(1000.0 * y_arr[i]),
                    "" if np.isnan(z_arr[i]) else float(z_arr[i]),
                    "" if np.isnan(g_arr[i]) else float(g_arr[i]),
                    "" if np.isnan(b_arr[i]) else float(b_arr[i]),
                ]
                writer.writerow(row)

        print(f"Saved {out_path}")
        self.set_status(f"Saved: {out_path}")

    def _has_plot_data(self):
        has_wave = any(self.last_waveforms[ch][0].size > 0 for ch in ("CH1", "CH2"))
        has_phase = len(self.phase_times) > 0
        has_metrics = len(self.admittance_values_s) > 0 or len(self.impedance_values_ohm) > 0
        return has_wave or has_phase or has_metrics

    def export_matplotlib_plot(self):
        if not self._has_plot_data():
            self.set_status("No plot data to export.")
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.set_status(f"Matplotlib unavailable: {exc}")
            QtWidgets.QMessageBox.warning(self, "Matplotlib missing", f"Cannot import matplotlib:\n{exc}")
            return

        default_base = Path(self.save_path).with_suffix("")
        default_png = str(default_base.with_name(default_base.name + "_plots.png"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export plots (Matplotlib)",
            default_png,
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*)",
        )
        if not path:
            return

        t1, v1 = self.last_waveforms["CH1"]
        t2, i2 = self.last_waveforms["CH2"]
        pt = np.asarray(self.phase_times, dtype=float)
        pv = np.asarray(self.phase_values, dtype=float)
        y_ms = np.asarray(self.admittance_values_s, dtype=float) * 1000.0
        z_ohm = np.asarray(self.impedance_values_ohm, dtype=float)

        fig, axes = plt.subplots(5, 1, figsize=(12, 14), constrained_layout=True)

        axes[0].plot(t1, v1, color="dodgerblue", linewidth=1.7)
        axes[0].set_title("CH1 Voltage")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Voltage [V]")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t2, i2, color="firebrick", linewidth=1.7)
        axes[1].set_title("CH2 Current (Filtered)")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Current [mA]")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(pt, pv, color="seagreen", linewidth=1.7)
        axes[2].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        axes[2].set_title("Phase(CH2 vs CH1)")
        axes[2].set_xlabel("Live Time [s]")
        axes[2].set_ylabel("Phase [deg]")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(pt, y_ms, color="#b4781e", linewidth=1.7)
        axes[3].set_title("Admittance Magnitude")
        axes[3].set_xlabel("Live Time [s]")
        axes[3].set_ylabel("|Y| [mS]")
        axes[3].grid(True, alpha=0.3)

        axes[4].plot(pt, z_ohm, color="#7850b4", linewidth=1.7)
        axes[4].set_title("Impedance Magnitude")
        axes[4].set_xlabel("Live Time [s]")
        axes[4].set_ylabel("|Z| [ohm]")
        axes[4].grid(True, alpha=0.3)

        try:
            fig.savefig(path, dpi=180)
            self.set_status(f"Matplotlib export saved: {path}")
        except Exception as exc:
            self.set_status(f"Matplotlib export error: {exc}")
            QtWidgets.QMessageBox.critical(self, "Matplotlib export error", str(exc))
        finally:
            plt.close(fig)

    def export_plotly_plot(self):
        if not self._has_plot_data():
            self.set_status("No plot data to export.")
            return

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except Exception as exc:
            self.set_status(f"Plotly unavailable: {exc}")
            QtWidgets.QMessageBox.warning(self, "Plotly missing", f"Cannot import plotly:\n{exc}")
            return

        default_base = Path(self.save_path).with_suffix("")
        default_html = str(default_base.with_name(default_base.name + "_plots.html"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export plots (Plotly)",
            default_html,
            "HTML files (*.html);;All files (*)",
        )
        if not path:
            return

        t1, v1 = self.last_waveforms["CH1"]
        t2, i2 = self.last_waveforms["CH2"]
        pt = np.asarray(self.phase_times, dtype=float)
        pv = np.asarray(self.phase_values, dtype=float)
        y_ms = np.asarray(self.admittance_values_s, dtype=float) * 1000.0
        z_ohm = np.asarray(self.impedance_values_ohm, dtype=float)

        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=(
                "CH1 Voltage",
                "CH2 Current (Filtered)",
                "Phase(CH2 vs CH1)",
                "Admittance Magnitude",
                "Impedance Magnitude",
            ),
        )
        fig.add_trace(
            go.Scatter(x=t1, y=v1, mode="lines", name="CH1 Voltage", line=dict(color="dodgerblue", width=2)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=t2, y=i2, mode="lines", name="CH2 Current Filtered", line=dict(color="firebrick", width=2)),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=pt, y=pv, mode="lines", name="Phase", line=dict(color="seagreen", width=2)),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=pt, y=y_ms, mode="lines", name="|Y|", line=dict(color="#b4781e", width=2)),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=pt, y=z_ohm, mode="lines", name="|Z|", line=dict(color="#7850b4", width=2)),
            row=5,
            col=1,
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="gray", row=3, col=1)
        fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
        fig.update_yaxes(title_text="Current [mA]", row=2, col=1)
        fig.update_yaxes(title_text="Phase [deg]", row=3, col=1)
        fig.update_yaxes(title_text="|Y| [mS]", row=4, col=1)
        fig.update_yaxes(title_text="|Z| [ohm]", row=5, col=1)
        fig.update_xaxes(title_text="Time [s]", row=1, col=1)
        fig.update_xaxes(title_text="Time [s]", row=2, col=1)
        fig.update_xaxes(title_text="Live Time [s]", row=3, col=1)
        fig.update_xaxes(title_text="Live Time [s]", row=4, col=1)
        fig.update_xaxes(title_text="Live Time [s]", row=5, col=1)
        fig.update_layout(height=1400, width=1200, title="Oscilloscope Results", template="plotly_white")

        try:
            fig.write_html(path, include_plotlyjs="cdn", full_html=True)
            self.set_status(f"Plotly export saved: {path}")
        except Exception as exc:
            self.set_status(f"Plotly export error: {exc}")
            QtWidgets.QMessageBox.critical(self, "Plotly export error", str(exc))

    def clear_data(self):
        if self.recording:
            return
        self.pending_processed_payload = None
        self.last_waveforms = {"CH1": (np.array([]), np.array([])), "CH2": (np.array([]), np.array([]))}
        self.phase_times = []
        self.phase_values = []
        self.admittance_values_s = []
        self.impedance_values_ohm = []
        self.conductance_values_s = []
        self.susceptance_values_s = []
        self.admittance_freq_hz = []
        self.admittance_freq_values_s = []
        self.live_plot_zero_time = None
        self.waveform_voltage_curve.setData([], [])
        self.waveform_current_curve.setData([], [])
        self.piezo_phase_curve.setData([], [])
        self.piezo_admittance_curve.setData([], [])
        self.trend_y_norm_curve.setData([], [])
        self.trend_z_norm_curve.setData([], [])
        self.nyquist_curve.setData([], [])
        self.admittance_freq_curve.setData([], [])
        self._update_metrics_bar(np.nan, np.nan, np.nan, np.nan, np.nan)
        self.set_status("Plots/data cleared.")

    def choose_save_path(self):
        if self.recording:
            self.set_status("Stop acquisition before changing save path.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save waveform CSV",
            self.save_path,
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            self.save_path = path
            self.save_path_label.setText(f"Save: {self.save_path}")
            self.set_status("Save path updated.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_kpi_cards()

    def set_status(self, text):
        self.status_label.setText(text)

    def selected_channels(self):
        return ["CH1", "CH2"]

    def closeEvent(self, event):
        try:
            self.render_timer.stop()
        except Exception:
            pass
        try:
            self._stop_processing_pipeline()
        except Exception:
            pass
        try:
            if self.live_worker is not None:
                self.live_worker.stop()
        except Exception:
            pass

        try:
            if self.live_thread is not None and self.live_thread.isRunning():
                self.live_thread.quit()
                self.live_thread.wait(1000)
        except Exception:
            pass

        try:
            if self.scope is not None:
                self.scope.close()
        except Exception:
            pass
        try:
            if self.rm is not None:
                self.rm.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    pg.setConfigOptions(antialias=PLOT_ANTIALIAS)
    app = QtWidgets.QApplication(sys.argv)
    win = SweepApp()
    win.show()
    print(f"Qt backend: {QT_LIB}")
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    sys.exit(exec_fn())


if __name__ == "__main__":
    main()
