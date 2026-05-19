from __future__ import annotations

import os
import wave
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
try:
    from scipy.io import wavfile as scipy_wavfile
    HAS_SCIPY_WAV = True
except Exception:
    HAS_SCIPY_WAV = False

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, RegenerationMode
    from nidaqmx.stream_writers import AnalogMultiChannelWriter
    HAS_NI = True
except Exception:
    HAS_NI = False


_WAV_RAW_CACHE: Dict[Tuple[str, float], Tuple[float, np.ndarray]] = {}
_WAV_RESAMPLED_CACHE: Dict[Tuple[str, float, float], np.ndarray] = {}


def _cache_put_limited(cache: Dict, key, value, max_entries: int):
    cache[key] = value
    while len(cache) > max_entries:
        cache.pop(next(iter(cache)))


def expand_ao_channels(spec: str) -> List[str]:
    parts = [p.strip() for p in str(spec).split(',') if p.strip()]
    out: List[str] = []
    for p in parts:
        if '/ao' in p and ':' in p:
            head, tail = p.split('/ao', 1)
            a, b = tail.split(':', 1)
            a, b = int(a), int(b)
            lo, hi = (a, b) if a <= b else (b, a)
            out.extend([f"{head}/ao{i}" for i in range(lo, hi + 1)])
        else:
            out.append(p)
    return out


def ao_key(name: str) -> str:
    s = str(name).strip().lower()
    if '/ao' in s:
        return f"ao{s.rsplit('/ao', 1)[1]}"
    if s.startswith('ao'):
        return s
    return ''


def default_channel_cfg(index: int, ao_name: str) -> Dict[str, Any]:
    return {
        'name': f'Channel {index + 1}',
        'enabled': index == 0,
        'ao_channel': ao_name,
        'waveform': 'Sine',
        'frequency_hz': 1000.0,
        'amplitude_vpk': 1.0,
        'offset_v': 0.0,
        'phase_deg': 0.0,
        'duty_cycle': 50.0,
        'symmetry': 50.0,
        'slider_limits': {
            'freq_min_hz': 0.1,
            'freq_max_hz': 100000.0,
            'amp_min_vpk': 0.0,
            'amp_max_vpk': 20.0,
            'off_min_v': -20.0,
            'off_max_v': 20.0,
            'phase_min_deg': -360.0,
            'phase_max_deg': 360.0,
            'duty_min_pct': 0.1,
            'duty_max_pct': 99.9,
            'sym_min_pct': 0.1,
            'sym_max_pct': 99.9,
        },
        'slider_resolution': {
            'freq_hz': 0.1,
            'amp_vpk': 0.01,
            'off_v': 0.01,
            'phase_deg': 0.1,
            'duty_pct': 0.1,
            'sym_pct': 0.1,
        },
    }


def _square(cycles: np.ndarray, duty: float) -> np.ndarray:
    frac = np.mod(cycles, 1.0)
    return np.where(frac < duty, 1.0, -1.0)


def _saw_or_triangle(cycles: np.ndarray, width: float, triangle: bool) -> np.ndarray:
    frac = np.mod(cycles, 1.0)
    w = float(np.clip(width, 1e-4, 1 - 1e-4))
    if triangle:
        up = -1.0 + 2.0 * (frac / w)
        down = 1.0 - 2.0 * ((frac - w) / (1.0 - w))
    else:
        up = -1.0 + 2.0 * (frac / w)
        down = -1.0 + 2.0 * ((frac - w) / (1.0 - w))
    return np.where(frac < w, up, down)


def generate_waveform_row(cfg: Dict[str, Any], t: np.ndarray) -> np.ndarray:
    wave = str(cfg.get('waveform', 'Sine')).strip().lower()
    f = float(cfg.get('frequency_hz', 1000.0))
    amp = float(cfg.get('amplitude_vpk', 1.0))
    off = float(cfg.get('offset_v', 0.0))
    ph = np.deg2rad(float(cfg.get('phase_deg', 0.0)))

    if wave == 'dc':
        y = np.ones_like(t)
    else:
        cycles = f * t + ph / (2.0 * np.pi)
        if wave == 'square':
            duty = float(np.clip(float(cfg.get('duty_cycle', 50.0)) / 100.0, 1e-3, 1 - 1e-3))
            y = _square(cycles, duty)
        elif wave == 'triangle':
            width = float(np.clip(float(cfg.get('symmetry', 50.0)) / 100.0, 1e-3, 1 - 1e-3))
            y = _saw_or_triangle(cycles, width, triangle=True)
        elif wave == 'sawtooth':
            width = float(np.clip(float(cfg.get('symmetry', 50.0)) / 100.0, 1e-3, 1 - 1e-3))
            y = _saw_or_triangle(cycles, width, triangle=False)
        elif wave == 'noise':
            y = np.random.randn(t.size)
        else:
            y = np.sin(2.0 * np.pi * f * t + ph)
    return off + amp * y


def _choose_periodic_sample_count(fs: float, n_target: int, channel_cfgs: List[Dict[str, Any]]) -> int:
    """Pick a nearby sample count that minimizes phase seam error at loop wrap."""
    if n_target < 64:
        return max(64, int(n_target))

    tonal = {'sine', 'square', 'triangle', 'sawtooth'}
    freqs: List[float] = []
    for cfg in channel_cfgs:
        if not bool(cfg.get('enabled', True)):
            continue
        w = str(cfg.get('waveform', 'Sine')).strip().lower()
        if w not in tonal:
            continue
        f = abs(float(cfg.get('frequency_hz', 0.0)))
        if f > 1e-9:
            freqs.append(f)
    if not freqs:
        return int(n_target)

    lo = max(64, int(round(0.75 * n_target)))
    hi = int(round(1.25 * n_target))
    if hi - lo > 25000:
        pad = 12500
        lo = max(64, int(n_target - pad))
        hi = int(n_target + pad)

    n_vals = np.arange(lo, hi + 1, dtype=np.float64)
    ff = np.asarray(freqs, dtype=np.float64) / float(fs)
    cyc = ff[:, None] * n_vals[None, :]
    frac = np.abs(cyc - np.round(cyc))
    frac = np.minimum(frac, 1.0 - frac)
    seam_err = np.max(frac, axis=0)
    penalty = 0.02 * np.abs((n_vals - float(n_target)) / max(1.0, float(n_target)))
    score = seam_err + penalty
    i_best = int(np.argmin(score))
    return int(n_vals[i_best])


def build_output_buffer_for_task(
    fs: float,
    duration: float,
    channel_cfgs: List[Dict[str, Any]],
    task_channels: List[str],
) -> np.ndarray:
    n_target = max(32, int(round(float(fs) * float(duration))))
    n = _choose_periodic_sample_count(float(fs), int(n_target), channel_cfgs)
    t = np.arange(n, dtype=np.float64) / float(fs)

    active = [c for c in channel_cfgs if bool(c.get('enabled', True))]
    if not task_channels:
        if not active:
            return np.zeros((1, n), dtype=np.float64)
        return np.asarray([generate_waveform_row(c, t) for c in active], dtype=np.float64)

    by_ao = {}
    for cfg in active:
        key = ao_key(cfg.get('ao_channel', ''))
        if key and key not in by_ao:
            by_ao[key] = cfg

    rows = []
    for ch in task_channels:
        cfg = by_ao.get(ao_key(ch))
        if cfg is None:
            rows.append(np.zeros(n, dtype=np.float64))
        else:
            rows.append(generate_waveform_row(cfg, t))
    return np.asarray(rows, dtype=np.float64)


def _sweep_phase(t: np.ndarray, f0: float, f1: float, duration: float, kind: str) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    T = max(1e-9, float(duration))
    if str(kind).lower().startswith('log') and f0 > 0 and f1 > 0 and abs(f1 - f0) > 1e-12:
        r = f1 / f0
        if abs(r - 1.0) < 1e-12:
            return 2.0 * np.pi * f0 * t
        return 2.0 * np.pi * (f0 * T / np.log(r)) * (np.power(r, t / T) - 1.0)
    k = (f1 - f0) / T
    return 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)


def _chirp_freq_trace(n: int, f0: float, f1: float, kind: str) -> np.ndarray:
    n = max(2, int(n))
    if str(kind).lower().startswith('log') and f0 > 0 and f1 > 0:
        return np.geomspace(float(f0), float(f1), n, dtype=np.float64)
    return np.linspace(float(f0), float(f1), n, dtype=np.float64)


def _step_freq_values(f0: float, f1: float, step_hz: float) -> np.ndarray:
    step = abs(float(step_hz))
    if step <= 1e-12:
        step = 1.0
    if f1 >= f0:
        vals = np.arange(float(f0), float(f1) + 0.5 * step, step, dtype=np.float64)
        if vals.size == 0 or vals[-1] < f1:
            vals = np.append(vals, float(f1))
        return vals
    vals = np.arange(float(f0), float(f1) - 0.5 * step, -step, dtype=np.float64)
    if vals.size == 0 or vals[-1] > f1:
        vals = np.append(vals, float(f1))
    return vals


def _sweep_freq_trace(fs: float, duration: float, sweep_cfg: Dict[str, Any]) -> np.ndarray:
    mode = str(sweep_cfg.get('mode', 'Chirp')).strip().lower()
    direction = str(sweep_cfg.get('direction', 'Up')).strip().lower()
    kind = str(sweep_cfg.get('sweep_type', 'Linear'))
    f_start = float(sweep_cfg.get('f_start_hz', 100.0))
    f_stop = float(sweep_cfg.get('f_stop_hz', 1000.0))
    if mode.startswith('step'):
        step_hz = float(sweep_cfg.get('step_hz', 10.0))
        step_dwell_s = max(1e-4, float(sweep_cfg.get('step_dwell_s', 0.05)))
        hold = max(1, int(round(step_dwell_s * float(fs))))
        up_vals = _step_freq_values(f_start, f_stop, step_hz)
        down_vals = _step_freq_values(f_stop, f_start, step_hz)
        if direction.startswith('down'):
            vals = down_vals
        elif direction.startswith('up-down'):
            vals = np.concatenate([up_vals, down_vals[1:]]) if down_vals.size > 1 else up_vals
        else:
            vals = up_vals
        return np.repeat(vals, hold).astype(np.float64, copy=False)

    n = max(32, int(round(float(fs) * float(duration))))
    if direction.startswith('down'):
        return _chirp_freq_trace(n, f_stop, f_start, kind)
    if direction.startswith('up-down'):
        n1 = max(2, n // 2)
        n2 = max(2, n - n1)
        up = _chirp_freq_trace(n1, f_start, f_stop, kind)
        down = _chirp_freq_trace(n2, f_stop, f_start, kind)
        return np.concatenate([up, down[1:]])
    return _chirp_freq_trace(n, f_start, f_stop, kind)


def _fade_envelope(n: int, fs: float, fade_in_s: float, fade_out_s: float) -> np.ndarray:
    env = np.ones(int(n), dtype=np.float64)
    n_in = max(0, int(round(max(0.0, float(fade_in_s)) * float(fs))))
    n_out = max(0, int(round(max(0.0, float(fade_out_s)) * float(fs))))
    if n_in > 0:
        n_in = min(n_in, env.size)
        env[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float64)
    if n_out > 0:
        n_out = min(n_out, env.size)
        env[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float64)
    return env


def build_sweep_buffer_for_task(
    fs: float,
    duration: float,
    channel_cfgs: List[Dict[str, Any]],
    task_channels: List[str],
    sweep_cfg: Dict[str, Any],
) -> np.ndarray:
    freq = _sweep_freq_trace(fs=fs, duration=duration, sweep_cfg=sweep_cfg)
    n = int(freq.size)
    max_samples = int(sweep_cfg.get('max_samples', 0))
    if max_samples > 0 and n > max_samples:
        raise ValueError(
            f"Sweep buffer too large ({n} samples). Reduce duration/steps or sample rate."
        )

    ph = 2.0 * np.pi * np.cumsum(freq, dtype=np.float64) / float(fs)
    if ph.size:
        ph -= ph[0]
    env = _fade_envelope(
        n=n,
        fs=fs,
        fade_in_s=float(sweep_cfg.get('fade_in_s', 0.0)),
        fade_out_s=float(sweep_cfg.get('fade_out_s', 0.0)),
    )

    active = [c for c in channel_cfgs if bool(c.get('enabled', True))]
    if not task_channels:
        if not active:
            return np.zeros((1, n), dtype=np.float64)
        rows = []
        for cfg in active:
            amp = float(cfg.get('amplitude_vpk', 1.0))
            off = float(cfg.get('offset_v', 0.0))
            ph0 = np.deg2rad(float(cfg.get('phase_deg', 0.0)))
            rows.append(off + amp * (np.sin(ph + ph0) * env))
        return np.asarray(rows, dtype=np.float64)

    by_ao = {}
    for cfg in active:
        key = ao_key(cfg.get('ao_channel', ''))
        if key and key not in by_ao:
            by_ao[key] = cfg

    rows = []
    for ch in task_channels:
        cfg = by_ao.get(ao_key(ch))
        if cfg is None:
            rows.append(np.zeros(n, dtype=np.float64))
            continue
        amp = float(cfg.get('amplitude_vpk', 1.0))
        off = float(cfg.get('offset_v', 0.0))
        ph0 = np.deg2rad(float(cfg.get('phase_deg', 0.0)))
        rows.append(off + amp * (np.sin(ph + ph0) * env))
    return np.asarray(rows, dtype=np.float64)


def _decode_pcm(raw: bytes, sample_width: int, channels: int) -> np.ndarray:
    if sample_width == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        x = (x - 128.0) / 128.0
    elif sample_width == 2:
        x = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.0
    elif sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i = (b[:, 0].astype(np.int32) |
             (b[:, 1].astype(np.int32) << 8) |
             (b[:, 2].astype(np.int32) << 16))
        i = np.where(i & 0x800000, i - 0x1000000, i)
        x = i.astype(np.float64) / 8388608.0
    elif sample_width == 4:
        x = np.frombuffer(raw, dtype='<i4').astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f'Unsupported WAV sample width: {sample_width} bytes')
    if channels <= 1:
        return x.reshape(1, -1)
    return x.reshape(-1, channels).T


def load_wav_file(path: str) -> Tuple[float, np.ndarray]:
    p = str(path).strip()
    if not p:
        raise ValueError('WAV path is empty.')
    if HAS_SCIPY_WAV:
        try:
            sr, arr = scipy_wavfile.read(p)
            x = np.asarray(arr)
            if x.ndim == 1:
                x = x[None, :]
            else:
                x = x.T
            if np.issubdtype(x.dtype, np.integer):
                info = np.iinfo(x.dtype)
                denom = float(max(abs(info.min), abs(info.max)))
                x = x.astype(np.float64) / denom
            elif np.issubdtype(x.dtype, np.floating):
                x = x.astype(np.float64)
            else:
                x = x.astype(np.float64)
            return float(sr), x
        except Exception:
            pass
    with wave.open(p, 'rb') as w:
        n_ch = int(w.getnchannels())
        sr = float(w.getframerate())
        sw = int(w.getsampwidth())
        raw = w.readframes(int(w.getnframes()))
    data = _decode_pcm(raw, sw, n_ch)
    return sr, np.asarray(data, dtype=np.float64)


def _resample_rows_linear(data: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if abs(float(src_fs) - float(dst_fs)) <= 1e-12:
        return data.copy()
    n_src = data.shape[1]
    if n_src < 2:
        return data.copy()
    dur = (n_src - 1) / float(src_fs)
    n_dst = max(2, int(round(dur * float(dst_fs))) + 1)
    x_src = np.linspace(0.0, dur, n_src, dtype=np.float64)
    x_dst = np.linspace(0.0, dur, n_dst, dtype=np.float64)
    out = np.zeros((data.shape[0], n_dst), dtype=np.float64)
    for i in range(data.shape[0]):
        out[i] = np.interp(x_dst, x_src, data[i])
    return out


def _load_wav_resampled_cached(path: str, dst_fs: float) -> Tuple[float, np.ndarray]:
    p = str(path).strip()
    if not p:
        raise ValueError('WAV path is empty.')
    p = os.path.abspath(os.path.expanduser(p))
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    mtime = float(os.path.getmtime(p))

    raw_key = (p, mtime)
    raw = _WAV_RAW_CACHE.get(raw_key)
    if raw is None:
        src_fs, wav = load_wav_file(p)
        raw = (float(src_fs), np.asarray(wav, dtype=np.float64))
        _cache_put_limited(_WAV_RAW_CACHE, raw_key, raw, max_entries=4)

    src_fs, wav = raw
    res_key = (p, mtime, float(dst_fs))
    out = _WAV_RESAMPLED_CACHE.get(res_key)
    if out is None:
        out = _resample_rows_linear(wav, src_fs=src_fs, dst_fs=float(dst_fs))
        _cache_put_limited(_WAV_RESAMPLED_CACHE, res_key, out, max_entries=8)
    return float(src_fs), np.asarray(out, dtype=np.float64)


def build_wav_buffer_for_task(
    fs: float,
    task_channels: List[str],
    wav_cfg: Dict[str, Any],
    preview_duration_s: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    _, wav_cached = _load_wav_resampled_cached(str(wav_cfg.get('wav_path', '')), dst_fs=float(fs))
    wav = wav_cached.copy()

    if bool(wav_cfg.get('normalize', True)):
        m = float(np.max(np.abs(wav))) if wav.size else 0.0
        if m > 1e-12:
            wav = wav / m

    gain = float(wav_cfg.get('gain', 1.0))
    offset = float(wav_cfg.get('offset_v', 0.0))
    wav = wav * gain + offset

    if wav.shape[1] < 2:
        wav = np.pad(wav, ((0, 0), (0, 1)), mode='edge')

    if not task_channels:
        if preview_duration_s is not None:
            n_target = max(64, int(round(max(1e-6, float(preview_duration_s)) * float(fs))))
            if wav.shape[1] < n_target:
                rep = max(1, int(np.ceil(n_target / max(1, wav.shape[1]))))
                wav = np.tile(wav, (1, rep))
            wav = wav[:, :n_target]
        total_seconds = wav.shape[1] / float(fs)
        return np.asarray(wav, dtype=np.float64), float(total_seconds)

    n_ch = len(task_channels)
    rows = np.zeros((n_ch, wav.shape[1]), dtype=np.float64)
    map_raw = wav_cfg.get('ao_to_wav_map', [])
    map_list: List[int] = []
    if isinstance(map_raw, (list, tuple)):
        for x in map_raw:
            try:
                map_list.append(int(x))
            except Exception:
                map_list.append(0)

    if not map_list:
        use = min(n_ch, wav.shape[0])
        rows[:use, :] = wav[:use, :]
        if wav.shape[0] == 1 and n_ch > 1:
            rows[:, :] = wav[0][None, :]
    else:
        for i in range(n_ch):
            src = map_list[i] if i < len(map_list) else (i + 1)
            if src <= 0:
                continue  # mute this AO
            j = int(src) - 1  # 1-based input
            if 0 <= j < wav.shape[0]:
                rows[i, :] = wav[j, :]
            elif wav.shape[0] == 1:
                rows[i, :] = wav[0, :]

    if preview_duration_s is not None:
        n_target = max(64, int(round(max(1e-6, float(preview_duration_s)) * float(fs))))
        if rows.shape[1] < n_target:
            n_rep = max(1, int(np.ceil(n_target / max(1, rows.shape[1]))))
            rows = np.tile(rows, (1, n_rep))
        rows = rows[:, :n_target]
        total_seconds = rows.shape[1] / float(fs)
        return np.asarray(rows, dtype=np.float64), float(total_seconds)

    play_mode = str(wav_cfg.get('play_mode', 'Free-run')).strip().lower()
    rep_count = max(1, int(wav_cfg.get('repeat_count', 1)))
    max_duration = float(wav_cfg.get('max_duration_s', 0.0))
    total_seconds = rows.shape[1] / float(fs)

    if play_mode.startswith('repeat'):
        rows = np.tile(rows, (1, rep_count))
        total_seconds = rows.shape[1] / float(fs)
    elif play_mode.startswith('max'):
        if max_duration > 0:
            n_target = max(2, int(round(max_duration * float(fs))))
            n_rep = max(1, int(np.ceil(n_target / rows.shape[1])))
            rows = np.tile(rows, (1, n_rep))[:, :n_target]
            total_seconds = rows.shape[1] / float(fs)

    max_samples = int(wav_cfg.get('max_samples', 20_000_000))
    if rows.shape[1] > max_samples:
        raise ValueError(
            f"WAV playback buffer too large ({rows.shape[1]} samples per channel). "
            f"Reduce repeat/duration or sample rate."
        )

    return np.asarray(rows, dtype=np.float64), float(total_seconds)


class AoEngine:
    def __init__(self):
        self._task = None
        self._writer = None
        self._running = False
        self._task_sig: Optional[Tuple] = None
        self._sim_last = None

    def _close_task(self):
        if self._task is not None:
            try:
                self._task.stop()
            except Exception:
                pass
            try:
                self._task.close()
            except Exception:
                pass
        self._task = None
        self._writer = None
        self._running = False
        self._task_sig = None

    def stop(self):
        self._close_task()

    def _ensure_task(self, io_cfg: Dict[str, Any], n_samps: int, run_mode: str):
        fs = float(io_cfg['sample_rate'])
        vmin = float(io_cfg['ao_vmin'])
        vmax = float(io_cfg['ao_vmax'])
        channels = tuple(expand_ao_channels(io_cfg['ao_channels']))
        if not channels:
            raise RuntimeError('No AO channels configured.')
        run_mode = str(run_mode).strip().lower()
        sig = (channels, fs, vmin, vmax, int(n_samps), run_mode)
        if self._task is not None and self._task_sig == sig:
            return channels

        self._close_task()
        self._task = nidaqmx.Task()
        for ch in channels:
            self._task.ao_channels.add_ao_voltage_chan(ch, min_val=vmin, max_val=vmax)
        finite = run_mode.startswith('finite')
        self._task.timing.cfg_samp_clk_timing(
            rate=fs,
            sample_mode=AcquisitionType.FINITE if finite else AcquisitionType.CONTINUOUS,
            samps_per_chan=int(n_samps),
        )
        try:
            self._task.out_stream.regen_mode = (
                RegenerationMode.DONT_ALLOW_REGENERATION if finite else RegenerationMode.ALLOW_REGENERATION
            )
        except Exception:
            pass
        try:
            # Keep HW output buffer aligned to the waveform block length for clean regeneration.
            self._task.out_stream.output_buf_size = int(n_samps)
        except Exception:
            pass
        self._writer = AnalogMultiChannelWriter(self._task.out_stream, auto_start=False)
        self._task_sig = sig
        return channels

    def apply(self, io_cfg: Dict[str, Any], data: np.ndarray, run_mode: str = 'continuous'):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError('Output data must be 2D: channels x samples.')
        if not HAS_NI:
            self._sim_last = data.copy()
            return

        channels = self._ensure_task(io_cfg, data.shape[1], run_mode=run_mode)
        if data.shape[0] != len(channels):
            raise ValueError(f'Output channel mismatch: data has {data.shape[0]}, task has {len(channels)}')

        if self._running:
            self._task.stop()
            self._running = False
        self._writer.write_many_sample(data)
        self._task.start()
        self._running = True

    def is_done(self) -> bool:
        if self._task is None:
            return True
        try:
            return bool(self._task.is_task_done())
        except Exception:
            return False
