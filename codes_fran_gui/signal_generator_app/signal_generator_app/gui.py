from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from generator_backend import (
    HAS_NI,
    AoEngine,
    expand_ao_channels,
    ao_key,
    default_channel_cfg,
    build_output_buffer_for_task,
    build_sweep_buffer_for_task,
    build_wav_buffer_for_task,
)


SETTINGS = str(Path.home() / '.signal_generator_settings.json')


def parse_int_csv(text: str) -> List[int]:
    raw = str(text or '').replace(';', ',').replace(' ', ',')
    out: List[int] = []
    for p in raw.split(','):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out


def prepare_output_payload(
    settings: Dict[str, Any],
    preview_window_s: Optional[float] = None,
) -> Dict[str, Any]:
    io = settings['io']
    channels = expand_ao_channels(io['ao_channels'])
    if not channels:
        raise RuntimeError('No AO channels configured.')

    source_mode = str(settings.get('source_mode', 'Channel Waveforms')).strip().lower()
    run_mode = 'continuous'
    auto_stop_s = None
    preview_cap_s = None
    if preview_window_s is not None:
        preview_cap_s = max(0.01, float(preview_window_s) * 1.5)

    if source_mode.startswith('sweep'):
        sweep = settings.get('sweep', {})
        duration = float(sweep.get('duration_s', io.get('buffer_duration_s', 0.2)))
        if preview_cap_s is not None:
            duration = min(duration, preview_cap_s)
        data = build_sweep_buffer_for_task(
            fs=float(io['sample_rate']),
            duration=duration,
            channel_cfgs=settings.get('channels', []),
            task_channels=channels,
            sweep_cfg=sweep,
        )
        source_label = 'sweep'
    elif source_mode.startswith('wav'):
        wav = settings.get('wav', {})
        data, total_s = build_wav_buffer_for_task(
            fs=float(io['sample_rate']),
            task_channels=channels,
            wav_cfg=wav,
            preview_duration_s=preview_cap_s,
        )
        play_mode = str(wav.get('play_mode', 'Free-run loop')).strip().lower()
        if not play_mode.startswith('free'):
            run_mode = 'finite'
            auto_stop_s = float(total_s)
        source_label = 'wav'
    else:
        duration = float(io['buffer_duration_s'])
        if preview_cap_s is not None:
            duration = min(duration, preview_cap_s)
        data = build_output_buffer_for_task(
            fs=float(io['sample_rate']),
            duration=duration,
            channel_cfgs=settings.get('channels', []),
            task_channels=channels,
        )
        source_label = 'channel'

    vmin = float(io['ao_vmin'])
    vmax = float(io['ao_vmax'])
    clipped = int(np.count_nonzero((data < vmin) | (data > vmax)))
    if bool(io.get('clip_output', True)):
        data = np.clip(data, vmin, vmax)

    return {
        'channels': channels,
        'fs': float(io['sample_rate']),
        'data': np.asarray(data, dtype=np.float64),
        'clipped': clipped,
        'run_mode': run_mode,
        'auto_stop_s': auto_stop_s,
        'source_label': source_label,
    }


class ChannelPanel(QtWidgets.QGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, index: int, cfg: Dict[str, Any], parent=None):
        super().__init__(f"Generator Channel {index + 1}", parent)
        self.index = index
        limits = cfg.get('slider_limits', {})
        if not isinstance(limits, dict):
            limits = {}
        resolution = cfg.get('slider_resolution', {})
        if not isinstance(resolution, dict):
            resolution = {}

        self._freq_min = 0.1
        self._freq_max = 100000.0
        self._freq_ceiling = 1e6
        self._freq_user_min = float(limits.get('freq_min_hz', 0.1))
        self._freq_user_max = float(limits.get('freq_max_hz', 100000.0))
        self._amp_min = float(limits.get('amp_min_vpk', 0.0))
        self._amp_max = float(limits.get('amp_max_vpk', 20.0))
        self._off_min = float(limits.get('off_min_v', -20.0))
        self._off_max = float(limits.get('off_max_v', 20.0))
        self._phase_min = float(limits.get('phase_min_deg', -360.0))
        self._phase_max = float(limits.get('phase_max_deg', 360.0))
        self._duty_min = float(limits.get('duty_min_pct', 0.1))
        self._duty_max = float(limits.get('duty_max_pct', 99.9))
        self._sym_min = float(limits.get('sym_min_pct', 0.1))
        self._sym_max = float(limits.get('sym_max_pct', 99.9))
        self._freq_res_hz = float(resolution.get('freq_hz', 0.1))
        self._amp_res_vpk = float(resolution.get('amp_vpk', 0.01))
        self._off_res_v = float(resolution.get('off_v', 0.01))
        self._phase_res_deg = float(resolution.get('phase_deg', 0.1))
        self._duty_res_pct = float(resolution.get('duty_pct', 0.1))
        self._sym_res_pct = float(resolution.get('sym_pct', 0.1))

        self.cb_enabled = QtWidgets.QCheckBox('Enable')
        self.cb_enabled.setChecked(bool(cfg.get('enabled', index == 0)))

        self.le_name = QtWidgets.QLineEdit(str(cfg.get('name', f'Channel {index + 1}')))
        self.le_ao = QtWidgets.QLineEdit(str(cfg.get('ao_channel', f'ao{index}')))
        self.cmb_wave = QtWidgets.QComboBox()
        self.cmb_wave.addItems(['Sine', 'Square', 'Triangle', 'Sawtooth', 'DC', 'Noise'])
        self.cmb_wave.setCurrentText(str(cfg.get('waveform', 'Sine')))

        self.sb_freq = QtWidgets.QDoubleSpinBox()
        self.sb_freq.setRange(0.001, 1e6)
        self.sb_freq.setDecimals(4)
        self.sb_freq.setSingleStep(0.1)
        self.sb_freq.setValue(float(cfg.get('frequency_hz', 1000.0)))
        self.sb_freq.setMaximumWidth(90)
        self.sl_freq = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.sl_freq.setRange(0, 1000)
        self.sl_freq.setMinimumHeight(250)
        self.sl_freq.setFixedWidth(12)
        self.sl_freq.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        self.sb_amp = QtWidgets.QDoubleSpinBox()
        self.sb_amp.setRange(0.0, 20.0)
        self.sb_amp.setDecimals(4)
        self.sb_amp.setSingleStep(0.01)
        self.sb_amp.setValue(float(cfg.get('amplitude_vpk', 1.0)))
        self.sb_amp.setMaximumWidth(90)
        self.sl_amp = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.sl_amp.setRange(0, 10000)
        self.sl_amp.setMinimumHeight(250)
        self.sl_amp.setFixedWidth(12)
        self.sl_amp.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        self.sb_off = QtWidgets.QDoubleSpinBox()
        self.sb_off.setRange(-20.0, 20.0)
        self.sb_off.setDecimals(4)
        self.sb_off.setSingleStep(0.01)
        self.sb_off.setValue(float(cfg.get('offset_v', 0.0)))
        self.sb_off.setMaximumWidth(90)
        self.sl_off = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.sl_off.setRange(0, 10000)
        self.sl_off.setMinimumHeight(250)
        self.sl_off.setFixedWidth(12)
        self.sl_off.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        self.sb_phase = QtWidgets.QDoubleSpinBox()
        self.sb_phase.setRange(-360.0, 360.0)
        self.sb_phase.setDecimals(3)
        self.sb_phase.setSingleStep(0.1)
        self.sb_phase.setValue(float(cfg.get('phase_deg', 0.0)))
        self.sb_phase.setMaximumWidth(90)
        self.sl_phase = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.sl_phase.setRange(0, 10000)
        self.sl_phase.setMinimumHeight(250)
        self.sl_phase.setFixedWidth(12)
        self.sl_phase.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        self.sb_duty = QtWidgets.QDoubleSpinBox()
        self.sb_duty.setRange(0.1, 99.9)
        self.sb_duty.setDecimals(2)
        self.sb_duty.setSingleStep(0.1)
        self.sb_duty.setValue(float(cfg.get('duty_cycle', 50.0)))
        self.sb_duty.setMaximumWidth(90)
        self.sl_duty = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.sl_duty.setRange(0, 10000)
        self.sl_duty.setMinimumHeight(250)
        self.sl_duty.setFixedWidth(12)
        self.sl_duty.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        self.sb_sym = QtWidgets.QDoubleSpinBox()
        self.sb_sym.setRange(0.1, 99.9)
        self.sb_sym.setDecimals(2)
        self.sb_sym.setSingleStep(0.1)
        self.sb_sym.setValue(float(cfg.get('symmetry', 50.0)))
        self.sb_sym.setMaximumWidth(90)
        self.sl_sym = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.sl_sym.setRange(0, 10000)
        self.sl_sym.setMinimumHeight(250)
        self.sl_sym.setFixedWidth(12)
        self.sl_sym.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        self.ctrl_freq = self._ctrl_pair(
            lambda: self._nudge_freq(-1),
            lambda: self._nudge_freq(+1),
        )
        self.ctrl_amp = self._ctrl_pair(
            lambda: self._nudge_step(self.sb_amp, -1),
            lambda: self._nudge_step(self.sb_amp, +1),
        )
        self.ctrl_off = self._ctrl_pair(
            lambda: self._nudge_step(self.sb_off, -1),
            lambda: self._nudge_step(self.sb_off, +1),
        )
        self.ctrl_phase = self._ctrl_pair(
            lambda: self._nudge_step(self.sb_phase, -1),
            lambda: self._nudge_step(self.sb_phase, +1),
        )
        self.ctrl_duty = self._ctrl_pair(
            lambda: self._nudge_step(self.sb_duty, -1),
            lambda: self._nudge_step(self.sb_duty, +1),
        )
        self.ctrl_sym = self._ctrl_pair(
            lambda: self._nudge_step(self.sb_sym, -1),
            lambda: self._nudge_step(self.sb_sym, +1),
        )

        meta = QtWidgets.QFormLayout()
        meta.addRow(self.cb_enabled)
        meta.addRow('Name', self.le_name)
        meta.addRow('AO map', self.le_ao)
        meta.addRow('Waveform', self.cmb_wave)
        self.btn_slider_limits = QtWidgets.QPushButton('Slider limits...')
        meta.addRow(self.btn_slider_limits)

        cols = [
            self._param_column('Frequency [Hz]', self.sb_freq, self.sl_freq, self.ctrl_freq),
            self._param_column('Amplitude [Vpk]', self.sb_amp, self.sl_amp, self.ctrl_amp),
            self._param_column('Offset [V]', self.sb_off, self.sl_off, self.ctrl_off),
            self._param_column('Phase [deg]', self.sb_phase, self.sl_phase, self.ctrl_phase),
            self._param_column('Duty [%]', self.sb_duty, self.sl_duty, self.ctrl_duty),
            self._param_column('Symmetry [%]', self.sb_sym, self.sl_sym, self.ctrl_sym),
        ]
        strips = QtWidgets.QGridLayout()
        strips.setHorizontalSpacing(6)
        strips.setVerticalSpacing(10)
        for i, col in enumerate(cols):
            strips.addWidget(col, i // 3, i % 3)
        strips.setColumnStretch(0, 1)
        strips.setColumnStretch(1, 1)
        strips.setColumnStretch(2, 1)
        strips.setRowStretch(0, 1)
        strips.setRowStretch(1, 1)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(meta)
        root.addLayout(strips)
        root.addStretch(1)

        self._apply_all_slider_limits(emit_changed=False)

        self.sb_freq.valueChanged.connect(self._sync_freq_slider_from_spin)
        self.sl_freq.valueChanged.connect(self._sync_freq_spin_from_slider)
        self.sb_amp.valueChanged.connect(self._sync_amp_slider_from_spin)
        self.sl_amp.valueChanged.connect(self._sync_amp_spin_from_slider)
        self.sb_off.valueChanged.connect(self._sync_off_slider_from_spin)
        self.sl_off.valueChanged.connect(self._sync_off_spin_from_slider)
        self.sb_phase.valueChanged.connect(self._sync_phase_slider_from_spin)
        self.sl_phase.valueChanged.connect(self._sync_phase_spin_from_slider)
        self.sb_duty.valueChanged.connect(self._sync_duty_slider_from_spin)
        self.sl_duty.valueChanged.connect(self._sync_duty_spin_from_slider)
        self.sb_sym.valueChanged.connect(self._sync_sym_slider_from_spin)
        self.sl_sym.valueChanged.connect(self._sync_sym_spin_from_slider)
        self._sync_freq_slider_from_spin(self.sb_freq.value())
        self._sync_amp_slider_from_spin(self.sb_amp.value())
        self._sync_off_slider_from_spin(self.sb_off.value())
        self._sync_phase_slider_from_spin(self.sb_phase.value())
        self._sync_duty_slider_from_spin(self.sb_duty.value())
        self._sync_sym_slider_from_spin(self.sb_sym.value())

        self.cmb_wave.currentTextChanged.connect(self._update_ui)
        self._update_ui()

        self.cb_enabled.stateChanged.connect(self.changed.emit)
        self.le_name.editingFinished.connect(self.changed.emit)
        self.le_ao.editingFinished.connect(self.changed.emit)
        self.cmb_wave.currentTextChanged.connect(self.changed.emit)
        self.sb_freq.valueChanged.connect(self.changed.emit)
        self.sb_amp.valueChanged.connect(self.changed.emit)
        self.sb_off.valueChanged.connect(self.changed.emit)
        self.sb_phase.valueChanged.connect(self.changed.emit)
        self.sb_duty.valueChanged.connect(self.changed.emit)
        self.sb_sym.valueChanged.connect(self.changed.emit)
        self.btn_slider_limits.clicked.connect(self._edit_slider_limits)

    def _h(self, widgets):
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(4)
        for it in widgets:
            l.addWidget(it)
        l.addStretch(0)
        return w

    def _param_column(self, title: str, spin: QtWidgets.QDoubleSpinBox, slider: QtWidgets.QSlider, ctrl: QtWidgets.QWidget):
        w = QtWidgets.QWidget()
        w.setMinimumWidth(92)
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        l = QtWidgets.QVBoxLayout(w)
        l.setContentsMargins(1, 1, 1, 1)
        l.setSpacing(3)
        lab = QtWidgets.QLabel(title)
        lab.setAlignment(QtCore.Qt.AlignCenter)
        lab.setWordWrap(True)
        l.addWidget(lab)
        srow = QtWidgets.QHBoxLayout()
        srow.setContentsMargins(0, 0, 0, 0)
        srow.setSpacing(0)
        srow.addStretch(1)
        srow.addWidget(slider, 0, QtCore.Qt.AlignHCenter)
        srow.addStretch(1)
        l.addLayout(srow)
        l.addWidget(spin, 0, QtCore.Qt.AlignHCenter)
        l.addWidget(ctrl, 0, QtCore.Qt.AlignHCenter)
        l.addStretch(1)
        return w

    def _ctrl_pair(self, on_minus, on_plus):
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)
        b_minus = QtWidgets.QToolButton()
        b_minus.setText('-')
        b_minus.setAutoRepeat(True)
        b_minus.setAutoRepeatDelay(250)
        b_minus.setAutoRepeatInterval(60)
        b_minus.setFixedSize(24, 24)
        b_minus.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        b_plus = QtWidgets.QToolButton()
        b_plus.setText('+')
        b_plus.setAutoRepeat(True)
        b_plus.setAutoRepeatDelay(250)
        b_plus.setAutoRepeatInterval(60)
        b_plus.setFixedSize(24, 24)
        b_plus.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        b_minus.clicked.connect(on_minus)
        b_plus.clicked.connect(on_plus)
        l.addWidget(b_minus)
        l.addWidget(b_plus)
        w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        return w

    def _nudge_spin(self, spin: QtWidgets.QDoubleSpinBox, delta: float):
        v = float(spin.value()) + float(delta)
        v = float(np.clip(v, float(spin.minimum()), float(spin.maximum())))
        spin.setValue(v)

    def _nudge_step(self, spin: QtWidgets.QDoubleSpinBox, sign: int):
        step = float(spin.singleStep())
        mods = QtWidgets.QApplication.keyboardModifiers()
        if mods & QtCore.Qt.ShiftModifier:
            step *= 10.0
        elif mods & QtCore.Qt.ControlModifier:
            step *= 0.1
        self._nudge_spin(spin, float(sign) * step)

    def _nudge_freq(self, sign: int):
        step = max(1e-6, float(self._freq_res_hz))
        mods = QtWidgets.QApplication.keyboardModifiers()
        if mods & QtCore.Qt.ShiftModifier:
            step *= 10.0
        elif mods & QtCore.Qt.ControlModifier:
            step *= 0.1
        self._nudge_spin(self.sb_freq, float(sign) * step)

    def _sanitize_pair(self, lo: float, hi: float, min_allowed: float, max_allowed: float, min_span: float) -> tuple[float, float]:
        lo = float(np.clip(lo, min_allowed, max_allowed))
        hi = float(np.clip(hi, min_allowed, max_allowed))
        if hi < lo + min_span:
            hi = min(max_allowed, lo + min_span)
            if hi < lo + min_span:
                lo = max(min_allowed, hi - min_span)
        return lo, hi

    def _sanitize_resolution(self, step: float, min_step: float, max_step: float) -> float:
        if not np.isfinite(step):
            return min_step
        return float(np.clip(abs(float(step)), min_step, max_step))

    def _quantize(self, value: float, step: float, lo: float, hi: float) -> float:
        v = float(np.clip(value, lo, hi))
        if step <= 0.0:
            return v
        q = lo + round((v - lo) / step) * step
        return float(np.clip(q, lo, hi))

    def _apply_spin_steps(self):
        self._freq_res_hz = self._sanitize_resolution(self._freq_res_hz, 1e-6, 1e6)
        self._amp_res_vpk = self._sanitize_resolution(self._amp_res_vpk, 1e-6, 1000.0)
        self._off_res_v = self._sanitize_resolution(self._off_res_v, 1e-6, 1000.0)
        self._phase_res_deg = self._sanitize_resolution(self._phase_res_deg, 1e-4, 3600.0)
        self._duty_res_pct = self._sanitize_resolution(self._duty_res_pct, 1e-4, 99.9)
        self._sym_res_pct = self._sanitize_resolution(self._sym_res_pct, 1e-4, 99.9)
        self.sb_freq.setSingleStep(self._freq_res_hz)
        self.sb_amp.setSingleStep(self._amp_res_vpk)
        self.sb_off.setSingleStep(self._off_res_v)
        self.sb_phase.setSingleStep(self._phase_res_deg)
        self.sb_duty.setSingleStep(self._duty_res_pct)
        self.sb_sym.setSingleStep(self._sym_res_pct)

    def _apply_linear_limit(
        self,
        spin: QtWidgets.QDoubleSpinBox,
        slider: QtWidgets.QSlider,
        lo: float,
        hi: float,
        min_allowed: float,
        max_allowed: float,
        min_span: float,
    ) -> tuple[float, float]:
        lo, hi = self._sanitize_pair(lo, hi, min_allowed, max_allowed, min_span)
        spin.setRange(lo, hi)
        if spin.value() < lo:
            spin.setValue(lo)
        elif spin.value() > hi:
            spin.setValue(hi)
        self._sync_slider_from_spin_linear(spin.value(), slider, lo, hi)
        return lo, hi

    def _apply_freq_limit(self):
        max_allowed = max(0.1, float(self._freq_ceiling))
        req_lo, req_hi = self._sanitize_pair(self._freq_user_min, self._freq_user_max, 1e-3, 1e6, 1e-3)
        self._freq_user_min = req_lo
        self._freq_user_max = req_hi
        lo = min(req_lo, max_allowed)
        hi = min(req_hi, max_allowed)
        if hi < lo + 1e-3:
            lo = max(1e-3, hi - 1e-3)
        self._freq_min = lo
        self._freq_max = hi
        self.sb_freq.setRange(self._freq_min, self._freq_max)
        if self.sb_freq.value() < self._freq_min:
            self.sb_freq.setValue(self._freq_min)
        elif self.sb_freq.value() > self._freq_max:
            self.sb_freq.setValue(self._freq_max)
        self._sync_freq_slider_from_spin(self.sb_freq.value())

    def _apply_all_slider_limits(self, emit_changed: bool):
        self._apply_spin_steps()
        self._apply_freq_limit()
        self._amp_min, self._amp_max = self._apply_linear_limit(
            self.sb_amp, self.sl_amp, self._amp_min, self._amp_max, 0.0, 1000.0, 1e-6
        )
        self._off_min, self._off_max = self._apply_linear_limit(
            self.sb_off, self.sl_off, self._off_min, self._off_max, -1000.0, 1000.0, 1e-6
        )
        self._phase_min, self._phase_max = self._apply_linear_limit(
            self.sb_phase, self.sl_phase, self._phase_min, self._phase_max, -3600.0, 3600.0, 1e-3
        )
        self._duty_min, self._duty_max = self._apply_linear_limit(
            self.sb_duty, self.sl_duty, self._duty_min, self._duty_max, 0.1, 99.9, 1e-3
        )
        self._sym_min, self._sym_max = self._apply_linear_limit(
            self.sb_sym, self.sl_sym, self._sym_min, self._sym_max, 0.1, 99.9, 1e-3
        )
        if emit_changed:
            self.changed.emit()

    def _to_slider_linear(self, value: float, lo: float, hi: float, slider: QtWidgets.QSlider) -> int:
        if hi <= lo:
            return slider.minimum()
        p = (float(np.clip(value, lo, hi)) - lo) / (hi - lo)
        smin, smax = slider.minimum(), slider.maximum()
        return int(round(smin + p * (smax - smin)))

    def _from_slider_linear(self, sv: int, lo: float, hi: float, slider: QtWidgets.QSlider) -> float:
        if hi <= lo:
            return lo
        smin, smax = slider.minimum(), slider.maximum()
        p = (float(sv) - smin) / max(1e-12, float(smax - smin))
        return float(lo + p * (hi - lo))

    def _sync_slider_from_spin_linear(self, v: float, slider: QtWidgets.QSlider, lo: float, hi: float):
        slider.blockSignals(True)
        slider.setValue(self._to_slider_linear(v, lo, hi, slider))
        slider.blockSignals(False)

    def _freq_to_slider(self, f_hz: float) -> int:
        f = float(np.clip(f_hz, self._freq_min, self._freq_max))
        lo = math.log10(self._freq_min)
        hi = math.log10(self._freq_max)
        p = (math.log10(f) - lo) / max(1e-12, (hi - lo))
        return int(round(p * self.sl_freq.maximum()))

    def _slider_to_freq(self, s: int) -> float:
        p = float(s) / max(1, self.sl_freq.maximum())
        lo = math.log10(self._freq_min)
        hi = math.log10(self._freq_max)
        return float(10 ** (lo + p * (hi - lo)))

    def _sync_freq_slider_from_spin(self, v: float):
        self.sl_freq.blockSignals(True)
        self.sl_freq.setValue(self._freq_to_slider(v))
        self.sl_freq.blockSignals(False)

    def _sync_freq_spin_from_slider(self, v: int):
        f = self._quantize(self._slider_to_freq(v), self._freq_res_hz, self._freq_min, self._freq_max)
        self.sb_freq.blockSignals(True)
        self.sb_freq.setValue(f)
        self.sb_freq.blockSignals(False)
        self.changed.emit()

    def _sync_amp_slider_from_spin(self, v: float):
        self._sync_slider_from_spin_linear(v, self.sl_amp, self._amp_min, self._amp_max)

    def _sync_amp_spin_from_slider(self, v: int):
        amp = self._quantize(
            self._from_slider_linear(v, self._amp_min, self._amp_max, self.sl_amp),
            self._amp_res_vpk,
            self._amp_min,
            self._amp_max,
        )
        self.sb_amp.blockSignals(True)
        self.sb_amp.setValue(amp)
        self.sb_amp.blockSignals(False)
        self.changed.emit()

    def _sync_off_slider_from_spin(self, v: float):
        self._sync_slider_from_spin_linear(v, self.sl_off, self._off_min, self._off_max)

    def _sync_off_spin_from_slider(self, v: int):
        off = self._quantize(
            self._from_slider_linear(v, self._off_min, self._off_max, self.sl_off),
            self._off_res_v,
            self._off_min,
            self._off_max,
        )
        self.sb_off.blockSignals(True)
        self.sb_off.setValue(off)
        self.sb_off.blockSignals(False)
        self.changed.emit()

    def _sync_phase_slider_from_spin(self, v: float):
        self._sync_slider_from_spin_linear(v, self.sl_phase, self._phase_min, self._phase_max)

    def _sync_phase_spin_from_slider(self, v: int):
        phase = self._quantize(
            self._from_slider_linear(v, self._phase_min, self._phase_max, self.sl_phase),
            self._phase_res_deg,
            self._phase_min,
            self._phase_max,
        )
        self.sb_phase.blockSignals(True)
        self.sb_phase.setValue(phase)
        self.sb_phase.blockSignals(False)
        self.changed.emit()

    def _sync_duty_slider_from_spin(self, v: float):
        self._sync_slider_from_spin_linear(v, self.sl_duty, self._duty_min, self._duty_max)

    def _sync_duty_spin_from_slider(self, v: int):
        duty = self._quantize(
            self._from_slider_linear(v, self._duty_min, self._duty_max, self.sl_duty),
            self._duty_res_pct,
            self._duty_min,
            self._duty_max,
        )
        self.sb_duty.blockSignals(True)
        self.sb_duty.setValue(duty)
        self.sb_duty.blockSignals(False)
        self.changed.emit()

    def _sync_sym_slider_from_spin(self, v: float):
        self._sync_slider_from_spin_linear(v, self.sl_sym, self._sym_min, self._sym_max)

    def _sync_sym_spin_from_slider(self, v: int):
        sym = self._quantize(
            self._from_slider_linear(v, self._sym_min, self._sym_max, self.sl_sym),
            self._sym_res_pct,
            self._sym_min,
            self._sym_max,
        )
        self.sb_sym.blockSignals(True)
        self.sb_sym.setValue(sym)
        self.sb_sym.blockSignals(False)
        self.changed.emit()

    def set_frequency_max(self, max_hz: float):
        self._freq_ceiling = float(np.clip(max_hz, 0.1, 1e6))
        self._apply_all_slider_limits(emit_changed=False)

    def _edit_slider_limits(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Slider limits - CH {self.index + 1}")
        form = QtWidgets.QGridLayout(dlg)
        form.addWidget(QtWidgets.QLabel('Parameter'), 0, 0)
        form.addWidget(QtWidgets.QLabel('Min'), 0, 1)
        form.addWidget(QtWidgets.QLabel('Max'), 0, 2)
        form.addWidget(QtWidgets.QLabel('Resolution'), 0, 3)

        def mk_spin(lo: float, hi: float, val: float, dec: int, step: float) -> QtWidgets.QDoubleSpinBox:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(dec)
            sb.setSingleStep(step)
            sb.setValue(float(val))
            sb.setMaximumWidth(120)
            return sb

        rows = [
            (
                'Frequency [Hz]',
                mk_spin(0.001, 1e6, self._freq_user_min, 4, 0.1),
                mk_spin(0.001, 1e6, self._freq_user_max, 4, 0.1),
                mk_spin(1e-6, 1e6, self._freq_res_hz, 6, 0.1),
            ),
            (
                'Amplitude [Vpk]',
                mk_spin(0.0, 1000.0, self._amp_min, 4, 0.01),
                mk_spin(0.0, 1000.0, self._amp_max, 4, 0.01),
                mk_spin(1e-6, 1000.0, self._amp_res_vpk, 6, 0.01),
            ),
            (
                'Offset [V]',
                mk_spin(-1000.0, 1000.0, self._off_min, 4, 0.01),
                mk_spin(-1000.0, 1000.0, self._off_max, 4, 0.01),
                mk_spin(1e-6, 1000.0, self._off_res_v, 6, 0.01),
            ),
            (
                'Phase [deg]',
                mk_spin(-3600.0, 3600.0, self._phase_min, 3, 0.1),
                mk_spin(-3600.0, 3600.0, self._phase_max, 3, 0.1),
                mk_spin(1e-4, 3600.0, self._phase_res_deg, 4, 0.1),
            ),
            (
                'Duty [%]',
                mk_spin(0.1, 99.9, self._duty_min, 2, 0.1),
                mk_spin(0.1, 99.9, self._duty_max, 2, 0.1),
                mk_spin(1e-4, 99.9, self._duty_res_pct, 4, 0.1),
            ),
            (
                'Symmetry [%]',
                mk_spin(0.1, 99.9, self._sym_min, 2, 0.1),
                mk_spin(0.1, 99.9, self._sym_max, 2, 0.1),
                mk_spin(1e-4, 99.9, self._sym_res_pct, 4, 0.1),
            ),
        ]
        for r, (name, sb_lo, sb_hi, sb_res) in enumerate(rows, start=1):
            form.addWidget(QtWidgets.QLabel(name), r, 0)
            form.addWidget(sb_lo, r, 1)
            form.addWidget(sb_hi, r, 2)
            form.addWidget(sb_res, r, 3)

        hint = QtWidgets.QLabel(
            f"Frequency max is capped by sample rate (Nyquist). Current cap: {self._freq_ceiling:.6g} Hz"
        )
        hint.setWordWrap(True)
        form.addWidget(hint, len(rows) + 1, 0, 1, 4)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addWidget(btns, len(rows) + 2, 0, 1, 4)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        self._freq_user_min = float(rows[0][1].value())
        self._freq_user_max = float(rows[0][2].value())
        self._amp_min = float(rows[1][1].value())
        self._amp_max = float(rows[1][2].value())
        self._off_min = float(rows[2][1].value())
        self._off_max = float(rows[2][2].value())
        self._phase_min = float(rows[3][1].value())
        self._phase_max = float(rows[3][2].value())
        self._duty_min = float(rows[4][1].value())
        self._duty_max = float(rows[4][2].value())
        self._sym_min = float(rows[5][1].value())
        self._sym_max = float(rows[5][2].value())
        self._freq_res_hz = float(rows[0][3].value())
        self._amp_res_vpk = float(rows[1][3].value())
        self._off_res_v = float(rows[2][3].value())
        self._phase_res_deg = float(rows[3][3].value())
        self._duty_res_pct = float(rows[4][3].value())
        self._sym_res_pct = float(rows[5][3].value())
        self._apply_all_slider_limits(emit_changed=True)

    def _update_ui(self):
        w = self.cmb_wave.currentText().strip().lower()
        en_freq = w not in ('dc', 'noise')
        en_phase = w not in ('dc', 'noise')
        self.sb_freq.setEnabled(en_freq)
        self.sl_freq.setEnabled(en_freq)
        self.ctrl_freq.setEnabled(en_freq)
        self.sb_phase.setEnabled(en_phase)
        self.sl_phase.setEnabled(en_phase)
        self.ctrl_phase.setEnabled(en_phase)
        self.sb_duty.setEnabled(w == 'square')
        self.sl_duty.setEnabled(w == 'square')
        self.ctrl_duty.setEnabled(w == 'square')
        self.sb_sym.setEnabled(w in ('triangle', 'sawtooth'))
        self.sl_sym.setEnabled(w in ('triangle', 'sawtooth'))
        self.ctrl_sym.setEnabled(w in ('triangle', 'sawtooth'))

    def value(self) -> Dict[str, Any]:
        return {
            'name': self.le_name.text().strip() or f'Channel {self.index + 1}',
            'enabled': self.cb_enabled.isChecked(),
            'ao_channel': self.le_ao.text().strip(),
            'waveform': self.cmb_wave.currentText(),
            'frequency_hz': float(self.sb_freq.value()),
            'amplitude_vpk': float(self.sb_amp.value()),
            'offset_v': float(self.sb_off.value()),
            'phase_deg': float(self.sb_phase.value()),
            'duty_cycle': float(self.sb_duty.value()),
            'symmetry': float(self.sb_sym.value()),
            'slider_limits': {
                'freq_min_hz': float(self._freq_user_min),
                'freq_max_hz': float(self._freq_user_max),
                'amp_min_vpk': float(self._amp_min),
                'amp_max_vpk': float(self._amp_max),
                'off_min_v': float(self._off_min),
                'off_max_v': float(self._off_max),
                'phase_min_deg': float(self._phase_min),
                'phase_max_deg': float(self._phase_max),
                'duty_min_pct': float(self._duty_min),
                'duty_max_pct': float(self._duty_max),
                'sym_min_pct': float(self._sym_min),
                'sym_max_pct': float(self._sym_max),
            },
            'slider_resolution': {
                'freq_hz': float(self._freq_res_hz),
                'amp_vpk': float(self._amp_res_vpk),
                'off_v': float(self._off_res_v),
                'phase_deg': float(self._phase_res_deg),
                'duty_pct': float(self._duty_res_pct),
                'sym_pct': float(self._sym_res_pct),
            },
        }


class SignalWorker(QtCore.QThread):
    preview = QtCore.pyqtSignal(dict)
    status = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, settings: Dict[str, Any]):
        super().__init__()
        self._mutex = QtCore.QMutex()
        self._settings = settings
        self._revision = 1
        self._stop = False
        self._min_apply_interval_s = 0.09

    def update_settings(self, settings: Dict[str, Any]):
        self._mutex.lock()
        try:
            self._settings = settings
            self._revision += 1
        finally:
            self._mutex.unlock()

    def request_stop(self):
        self._stop = True

    def _snapshot(self):
        self._mutex.lock()
        try:
            return self._settings, self._revision
        finally:
            self._mutex.unlock()

    def run(self):
        engine = AoEngine()
        last_rev = -1
        last_apply_t = 0.0
        deadline = None
        try:
            while not self._stop:
                settings, rev = self._snapshot()
                if rev != last_rev:
                    dt = time.monotonic() - last_apply_t
                    if last_rev >= 0 and dt < self._min_apply_interval_s:
                        self.msleep(int(max(1, round((self._min_apply_interval_s - dt) * 1000.0))))
                        continue
                    # Pull the newest settings right before applying to coalesce bursts.
                    settings, rev = self._snapshot()
                    if rev == last_rev:
                        continue
                    io = settings['io']
                    payload = prepare_output_payload(settings)
                    engine.apply(io, payload['data'], run_mode=payload['run_mode'])
                    self.preview.emit({
                        'channels': payload['channels'],
                        'fs': payload['fs'],
                        'data': payload['data'],
                    })
                    mode = 'SIM' if not HAS_NI else 'NI'
                    msg = f"Running free-running ({mode}) | source={payload['source_label']} | {len(payload['channels'])} ch"
                    if payload['run_mode'] == 'finite':
                        msg = f"Running finite ({mode}) | source={payload['source_label']} | {len(payload['channels'])} ch"
                    if payload['clipped'] > 0 and bool(io.get('clip_output', True)):
                        msg += f" | clipped samples: {payload['clipped']}"
                    self.status.emit(msg)
                    deadline = None
                    if payload['auto_stop_s'] is not None:
                        deadline = time.monotonic() + float(payload['auto_stop_s'])
                    last_rev = rev
                    last_apply_t = time.monotonic()

                if deadline is not None:
                    done = (HAS_NI and engine.is_done()) or (time.monotonic() >= deadline)
                    if done:
                        self.status.emit('Playback completed')
                        break
                self.msleep(20)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            engine.stop()
            self.status.emit('Stopped')


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NI Signal Generator (Standalone)')
        self.resize(1520, 930)
        self.worker: Optional[SignalWorker] = None
        self.channel_panels: List[ChannelPanel] = []
        self.curves = []
        self.curve_names: List[str] = []
        self._live_update_timer = QtCore.QTimer(self)
        self._live_update_timer.setSingleShot(True)
        self._live_update_timer.setInterval(33)  # ~30 fps UI refresh budget
        self._live_update_timer.timeout.connect(self._flush_live_update)
        self._preview_cap_factor = 1.5
        self._build_ui()
        self._load_settings()
        self._sync_channel_panels()
        self._update_frequency_limits()
        self._update_wav_map_hint()
        self._on_source_mode_changed()
        self._on_sweep_mode_changed()
        self._on_wav_play_mode_changed()
        self._update_sweep_speed_label()
        self._refresh_preview_only()

    def _build_ui(self):
        self.le_ao = QtWidgets.QLineEdit('cDAQ5Mod1/ao0')
        self.sb_fs = QtWidgets.QDoubleSpinBox()
        self.sb_fs.setRange(100, 500000)
        self.sb_fs.setDecimals(2)
        self.sb_fs.setValue(409600.0)

        self.sb_vmin = QtWidgets.QDoubleSpinBox()
        self.sb_vmin.setRange(-20, 0)
        self.sb_vmin.setValue(-10.0)
        self.sb_vmax = QtWidgets.QDoubleSpinBox()
        self.sb_vmax.setRange(0, 20)
        self.sb_vmax.setValue(10.0)

        self.sb_buf = QtWidgets.QDoubleSpinBox()
        self.sb_buf.setRange(0.005, 10.0)
        self.sb_buf.setDecimals(4)
        self.sb_buf.setValue(0.2)

        self.sb_view = QtWidgets.QDoubleSpinBox()
        self.sb_view.setRange(0.001, 2.0)
        self.sb_view.setDecimals(4)
        self.sb_view.setValue(0.02)

        self.cb_clip = QtWidgets.QCheckBox('Clip to AO range')
        self.cb_clip.setChecked(True)
        self.btn_sync = QtWidgets.QPushButton('Sync Channels from AO list')
        self.btn_start = QtWidgets.QPushButton('▶ Start')
        self.btn_stop = QtWidgets.QPushButton('⏹ Stop')
        self.btn_stop.setEnabled(False)
        self.lbl_status = QtWidgets.QLabel('Ready')

        form_out = QtWidgets.QFormLayout()
        form_out.addRow('AO channels', self.le_ao)
        form_out.addRow('Sample rate [Hz]', self.sb_fs)
        form_out.addRow('AO range [V]', self._h([self.sb_vmin, self.sb_vmax]))
        form_out.addRow('Generator buffer [s]', self.sb_buf)
        form_out.addRow('Preview window [s]', self.sb_view)
        form_out.addRow(self.cb_clip)
        form_out.addRow(self.btn_sync)
        form_out.addRow(self._h([self.btn_start, self.btn_stop]))
        form_out.addRow('Status', self.lbl_status)
        grp_main = QtWidgets.QGroupBox('Output Settings')
        grp_main.setLayout(form_out)

        self.cmb_source = QtWidgets.QComboBox()
        self.cmb_source.addItems(['Channel Waveforms', 'Sweep (Chirp)', 'WAV File'])

        self.page_source = QtWidgets.QStackedWidget()

        page_ch = QtWidgets.QWidget()
        l0 = QtWidgets.QVBoxLayout(page_ch)
        l0.addWidget(QtWidgets.QLabel('Uses per-channel waveform/frequency settings below.'))
        l0.addStretch(1)

        page_sw = QtWidgets.QWidget()
        form_sw = QtWidgets.QFormLayout(page_sw)
        self.sb_sw_f0 = QtWidgets.QDoubleSpinBox()
        self.sb_sw_f0.setRange(0.001, 1e6)
        self.sb_sw_f0.setValue(100.0)
        self.sb_sw_f1 = QtWidgets.QDoubleSpinBox()
        self.sb_sw_f1.setRange(0.001, 1e6)
        self.sb_sw_f1.setValue(5000.0)
        self.sb_sw_dur = QtWidgets.QDoubleSpinBox()
        self.sb_sw_dur.setRange(0.01, 86400.0)
        self.sb_sw_dur.setDecimals(4)
        self.sb_sw_dur.setValue(1.0)
        self.cmb_sw_mode = QtWidgets.QComboBox()
        self.cmb_sw_mode.addItems(['Chirp', 'Step'])
        self.cmb_sw_type = QtWidgets.QComboBox()
        self.cmb_sw_type.addItems(['Linear', 'Log'])
        self.cmb_sw_dir = QtWidgets.QComboBox()
        self.cmb_sw_dir.addItems(['Up', 'Down', 'Up-Down'])
        self.sb_sw_step = QtWidgets.QDoubleSpinBox()
        self.sb_sw_step.setRange(0.001, 1e6)
        self.sb_sw_step.setDecimals(4)
        self.sb_sw_step.setValue(10.0)
        self.sb_sw_step_dwell = QtWidgets.QDoubleSpinBox()
        self.sb_sw_step_dwell.setRange(0.0001, 10.0)
        self.sb_sw_step_dwell.setDecimals(4)
        self.sb_sw_step_dwell.setValue(0.05)
        self.sb_sw_fade_in = QtWidgets.QDoubleSpinBox()
        self.sb_sw_fade_in.setRange(0.0, 10.0)
        self.sb_sw_fade_in.setDecimals(4)
        self.sb_sw_fade_in.setValue(0.0)
        self.sb_sw_fade_out = QtWidgets.QDoubleSpinBox()
        self.sb_sw_fade_out.setRange(0.0, 10.0)
        self.sb_sw_fade_out.setDecimals(4)
        self.sb_sw_fade_out.setValue(0.0)
        self.lbl_sw_speed = QtWidgets.QLabel('-')
        self.lbl_sw_speed.setWordWrap(True)
        self.lbl_sw_speed.setStyleSheet('color: #444;')
        self._sw_step_rows = []
        form_sw.addRow('Start freq [Hz]', self.sb_sw_f0)
        form_sw.addRow('Stop freq [Hz]', self.sb_sw_f1)
        form_sw.addRow('Sweep mode', self.cmb_sw_mode)
        form_sw.addRow('Direction', self.cmb_sw_dir)
        form_sw.addRow('Sweep duration [s]', self.sb_sw_dur)
        form_sw.addRow('Sweep type', self.cmb_sw_type)
        form_sw.addRow('Step size [Hz]', self.sb_sw_step)
        form_sw.addRow('Step dwell [s]', self.sb_sw_step_dwell)
        form_sw.addRow('Fade in [s]', self.sb_sw_fade_in)
        form_sw.addRow('Fade out [s]', self.sb_sw_fade_out)
        form_sw.addRow('Sweep speed', self.lbl_sw_speed)

        page_wav = QtWidgets.QWidget()
        form_wav = QtWidgets.QFormLayout(page_wav)
        self.le_wav = QtWidgets.QLineEdit('')
        self.btn_wav_browse = QtWidgets.QPushButton('Browse...')
        self.cb_wav_norm = QtWidgets.QCheckBox('Normalize WAV to +/-1 before gain')
        self.cb_wav_norm.setChecked(True)
        self.sb_wav_gain = QtWidgets.QDoubleSpinBox()
        self.sb_wav_gain.setRange(0.0, 20.0)
        self.sb_wav_gain.setDecimals(4)
        self.sb_wav_gain.setValue(1.0)
        self.sb_wav_off = QtWidgets.QDoubleSpinBox()
        self.sb_wav_off.setRange(-20.0, 20.0)
        self.sb_wav_off.setDecimals(4)
        self.sb_wav_off.setValue(0.0)
        self.cmb_wav_play = QtWidgets.QComboBox()
        self.cmb_wav_play.addItems(['Free-run loop', 'Repeat N times', 'Max duration'])
        self.sb_wav_repeat = QtWidgets.QSpinBox()
        self.sb_wav_repeat.setRange(1, 100000)
        self.sb_wav_repeat.setValue(1)
        self.sb_wav_maxdur = QtWidgets.QDoubleSpinBox()
        self.sb_wav_maxdur.setRange(0.01, 36000.0)
        self.sb_wav_maxdur.setDecimals(3)
        self.sb_wav_maxdur.setValue(10.0)
        self.le_wav_map = QtWidgets.QLineEdit('')
        self.le_wav_map.setPlaceholderText('1,2,0,... (AO order; 0 = mute)')
        self.btn_wav_map_auto = QtWidgets.QPushButton('Auto 1:1')
        self.lbl_wav_map_hint = QtWidgets.QLabel('')
        form_wav.addRow('WAV file', self._h([self.le_wav, self.btn_wav_browse]))
        form_wav.addRow(self.cb_wav_norm)
        form_wav.addRow('Gain', self.sb_wav_gain)
        form_wav.addRow('Offset [V]', self.sb_wav_off)
        form_wav.addRow('WAV->AO map', self._h([self.le_wav_map, self.btn_wav_map_auto]))
        form_wav.addRow('AO order', self.lbl_wav_map_hint)
        form_wav.addRow('Playback mode', self.cmb_wav_play)
        form_wav.addRow('Repeat count', self.sb_wav_repeat)
        form_wav.addRow('Max duration [s]', self.sb_wav_maxdur)

        self.page_source.addWidget(page_ch)
        self.page_source.addWidget(page_sw)
        self.page_source.addWidget(page_wav)

        src_wrap = QtWidgets.QWidget()
        src_l = QtWidgets.QVBoxLayout(src_wrap)
        src_l.setContentsMargins(0, 0, 0, 0)
        src_l.addWidget(self.cmb_source)
        src_l.addWidget(self.page_source)
        grp_src = QtWidgets.QGroupBox('Source')
        gs = QtWidgets.QVBoxLayout(grp_src)
        gs.addWidget(src_wrap)

        self.tabs_channels = QtWidgets.QTabWidget()
        self.tabs_channels.setDocumentMode(True)
        self.tabs_channels.setTabPosition(QtWidgets.QTabWidget.North)
        grp_channels = QtWidgets.QGroupBox('Per-Channel Parameters')
        lch = QtWidgets.QVBoxLayout(grp_channels)
        lch.addWidget(self.tabs_channels)

        col_channels = QtWidgets.QVBoxLayout()
        col_channels.addWidget(grp_channels)
        channels_wrap = QtWidgets.QWidget()
        channels_wrap.setLayout(col_channels)
        channels_scroll = QtWidgets.QScrollArea()
        channels_scroll.setWidgetResizable(True)
        channels_scroll.setMinimumWidth(640)
        channels_scroll.setWidget(channels_wrap)

        col_settings = QtWidgets.QVBoxLayout()
        col_settings.addWidget(grp_main)
        col_settings.addWidget(grp_src)
        col_settings.addStretch(1)
        settings_wrap = QtWidgets.QWidget()
        settings_wrap.setLayout(col_settings)
        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumWidth(430)
        settings_scroll.setWidget(settings_wrap)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel('bottom', 'Time [s]')
        self.plot.setLabel('left', 'Voltage [V]')
        self.plot.getPlotItem().setDownsampling(auto=True, mode='peak')
        self.plot.getPlotItem().setClipToView(True)

        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(channels_scroll, 0)
        lay.addWidget(settings_scroll, 0)
        lay.addWidget(self.plot, 1)
        lay.setStretch(0, 3)
        lay.setStretch(1, 2)
        lay.setStretch(2, 4)

        self.btn_sync.clicked.connect(self._sync_channel_panels)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_wav_browse.clicked.connect(self._browse_wav)
        self.btn_wav_map_auto.clicked.connect(self._auto_wav_map_1to1)
        self.le_ao.editingFinished.connect(self._sync_channel_panels)
        self.le_ao.editingFinished.connect(self._on_live_params_changed)
        self.sb_fs.valueChanged.connect(self._update_frequency_limits)
        self.sb_fs.valueChanged.connect(self._on_live_params_changed)
        self.sb_vmin.valueChanged.connect(self._on_live_params_changed)
        self.sb_vmax.valueChanged.connect(self._on_live_params_changed)
        self.sb_buf.valueChanged.connect(self._on_live_params_changed)
        self.sb_view.valueChanged.connect(self._on_live_params_changed)
        self.cb_clip.stateChanged.connect(self._on_live_params_changed)

        self.cmb_source.currentTextChanged.connect(self._on_source_mode_changed)
        self.cmb_source.currentTextChanged.connect(self._on_live_params_changed)

        self.sb_sw_f0.valueChanged.connect(self._on_live_params_changed)
        self.sb_sw_f1.valueChanged.connect(self._on_live_params_changed)
        self.sb_sw_dur.valueChanged.connect(self._on_live_params_changed)
        self.cmb_sw_type.currentTextChanged.connect(self._on_live_params_changed)
        self.cmb_sw_mode.currentTextChanged.connect(self._on_sweep_mode_changed)
        self.cmb_sw_mode.currentTextChanged.connect(self._on_live_params_changed)
        self.cmb_sw_dir.currentTextChanged.connect(self._on_live_params_changed)
        self.sb_sw_step.valueChanged.connect(self._on_live_params_changed)
        self.sb_sw_step_dwell.valueChanged.connect(self._on_live_params_changed)
        self.sb_sw_fade_in.valueChanged.connect(self._on_live_params_changed)
        self.sb_sw_fade_out.valueChanged.connect(self._on_live_params_changed)

        self.sb_sw_f0.valueChanged.connect(self._update_sweep_speed_label)
        self.sb_sw_f1.valueChanged.connect(self._update_sweep_speed_label)
        self.sb_sw_dur.valueChanged.connect(self._update_sweep_speed_label)
        self.cmb_sw_type.currentTextChanged.connect(self._update_sweep_speed_label)
        self.cmb_sw_mode.currentTextChanged.connect(self._update_sweep_speed_label)
        self.cmb_sw_dir.currentTextChanged.connect(self._update_sweep_speed_label)
        self.sb_sw_step.valueChanged.connect(self._update_sweep_speed_label)
        self.sb_sw_step_dwell.valueChanged.connect(self._update_sweep_speed_label)

        self.le_wav.editingFinished.connect(self._on_live_params_changed)
        self.le_wav_map.editingFinished.connect(self._on_live_params_changed)
        self.cb_wav_norm.stateChanged.connect(self._on_live_params_changed)
        self.sb_wav_gain.valueChanged.connect(self._on_live_params_changed)
        self.sb_wav_off.valueChanged.connect(self._on_live_params_changed)
        self.cmb_wav_play.currentTextChanged.connect(self._on_wav_play_mode_changed)
        self.cmb_wav_play.currentTextChanged.connect(self._on_live_params_changed)
        self.sb_wav_repeat.valueChanged.connect(self._on_live_params_changed)
        self.sb_wav_maxdur.valueChanged.connect(self._on_live_params_changed)

    def _h(self, widgets):
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        for it in widgets:
            l.addWidget(it)
        return w

    def _browse_wav(self):
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open WAV', '', 'WAV files (*.wav)')
        if fp:
            self.le_wav.setText(fp)
            self._on_live_params_changed()

    def _on_source_mode_changed(self):
        mode = self.cmb_source.currentText().lower()
        if mode.startswith('sweep'):
            self.page_source.setCurrentIndex(1)
        elif mode.startswith('wav'):
            self.page_source.setCurrentIndex(2)
        else:
            self.page_source.setCurrentIndex(0)

    def _on_sweep_mode_changed(self):
        is_step = self.cmb_sw_mode.currentText().strip().lower().startswith('step')
        self.sb_sw_dur.setEnabled(not is_step)
        self.cmb_sw_type.setEnabled(not is_step)
        self.sb_sw_step.setEnabled(is_step)
        self.sb_sw_step_dwell.setEnabled(is_step)
        self._update_sweep_speed_label()

    def _update_sweep_speed_label(self):
        f0 = float(self.sb_sw_f0.value())
        f1 = float(self.sb_sw_f1.value())
        mode = self.cmb_sw_mode.currentText().strip().lower()
        direction = self.cmb_sw_dir.currentText().strip().lower()
        if mode.startswith('step'):
            step = max(1e-12, float(self.sb_sw_step.value()))
            dwell = max(1e-12, float(self.sb_sw_step_dwell.value()))
            v = step / dwell
            sign = '±' if direction.startswith('up-down') else ('-' if direction.startswith('down') else '+')
            self.lbl_sw_speed.setText(f"{sign}{v:.6g} Hz/s (equivalent step rate)")
            return

        T = max(1e-12, float(self.sb_sw_dur.value()))
        span = float(f1 - f0)
        if direction.startswith('up-down'):
            T = max(1e-12, T * 0.5)
        sweep_type = self.cmb_sw_type.currentText().strip().lower()
        if sweep_type.startswith('log') and f0 > 0 and f1 > 0 and abs(f1 - f0) > 1e-12:
            k = np.log(f1 / f0) / T
            s0 = f0 * k
            s1 = f1 * k
            if direction.startswith('down'):
                s0, s1 = -s1, -s0
            elif direction.startswith('up-down'):
                s0, s1 = abs(s0), abs(s1)
                self.lbl_sw_speed.setText(f"±[{s0:.6g}..{s1:.6g}] Hz/s (log, variable)")
                return
            self.lbl_sw_speed.setText(f"[{s0:.6g}..{s1:.6g}] Hz/s (log, variable)")
            return

        v = span / T
        if direction.startswith('down'):
            v = -abs(v)
        elif direction.startswith('up-down'):
            self.lbl_sw_speed.setText(f"±{abs(v):.6g} Hz/s")
            return
        else:
            v = abs(v)
        sign = '-' if v < 0 else '+'
        self.lbl_sw_speed.setText(f"{sign}{abs(v):.6g} Hz/s")

    def _on_wav_play_mode_changed(self):
        m = self.cmb_wav_play.currentText().lower()
        self.sb_wav_repeat.setEnabled(m.startswith('repeat'))
        self.sb_wav_maxdur.setEnabled(m.startswith('max'))

    def _update_wav_map_hint(self):
        ao_list = expand_ao_channels(self.le_ao.text())
        if not ao_list:
            self.lbl_wav_map_hint.setText('No AO channels configured')
            return
        self.lbl_wav_map_hint.setText(', '.join(ao_list))

    def _auto_wav_map_1to1(self):
        ao_list = expand_ao_channels(self.le_ao.text())
        if not ao_list:
            return
        self.le_wav_map.setText(','.join(str(i + 1) for i in range(len(ao_list))))
        self._on_live_params_changed()

    def _clear_channel_layout(self):
        while self.tabs_channels.count():
            w = self.tabs_channels.widget(0)
            self.tabs_channels.removeTab(0)
            if w is not None:
                w.deleteLater()

    def _sync_channel_panels(self):
        current = {ao_key(p.le_ao.text()): p.value() for p in self.channel_panels}
        ao_list = expand_ao_channels(self.le_ao.text())
        if not ao_list:
            ao_list = ['ao0']

        self._clear_channel_layout()
        self.channel_panels = []
        for i, ch in enumerate(ao_list):
            cfg = current.get(ao_key(ch), default_channel_cfg(i, ch))
            cfg['ao_channel'] = ch
            panel = ChannelPanel(i, cfg)
            panel.changed.connect(self._on_live_params_changed)
            self.channel_panels.append(panel)
            tab = QtWidgets.QWidget()
            lt = QtWidgets.QVBoxLayout(tab)
            lt.setContentsMargins(6, 6, 6, 6)
            lt.addWidget(panel)
            lt.addStretch(1)
            self.tabs_channels.addTab(tab, f"CH {i + 1}")
        self._update_frequency_limits()
        self._update_wav_map_hint()
        self._on_live_params_changed()

    def _update_frequency_limits(self):
        fs = float(self.sb_fs.value())
        fmax = max(0.1, 0.49 * fs)
        for panel in self.channel_panels:
            panel.set_frequency_max(fmax)

    def _collect(self) -> Dict[str, Any]:
        return {
            'io': {
                'ao_channels': self.le_ao.text().strip(),
                'sample_rate': float(self.sb_fs.value()),
                'ao_vmin': float(self.sb_vmin.value()),
                'ao_vmax': float(self.sb_vmax.value()),
                'buffer_duration_s': float(self.sb_buf.value()),
                'clip_output': bool(self.cb_clip.isChecked()),
            },
            'source_mode': self.cmb_source.currentText(),
            'sweep': {
                'f_start_hz': float(self.sb_sw_f0.value()),
                'f_stop_hz': float(self.sb_sw_f1.value()),
                'duration_s': float(self.sb_sw_dur.value()),
                'mode': self.cmb_sw_mode.currentText(),
                'direction': self.cmb_sw_dir.currentText(),
                'sweep_type': self.cmb_sw_type.currentText(),
                'step_hz': float(self.sb_sw_step.value()),
                'step_dwell_s': float(self.sb_sw_step_dwell.value()),
                'fade_in_s': float(self.sb_sw_fade_in.value()),
                'fade_out_s': float(self.sb_sw_fade_out.value()),
            },
            'wav': {
                'wav_path': self.le_wav.text().strip(),
                'normalize': bool(self.cb_wav_norm.isChecked()),
                'gain': float(self.sb_wav_gain.value()),
                'offset_v': float(self.sb_wav_off.value()),
                'ao_to_wav_map': parse_int_csv(self.le_wav_map.text()),
                'play_mode': self.cmb_wav_play.currentText(),
                'repeat_count': int(self.sb_wav_repeat.value()),
                'max_duration_s': float(self.sb_wav_maxdur.value()),
            },
            'channels': [p.value() for p in self.channel_panels],
            'ui': {
                'preview_window_s': float(self.sb_view.value()),
            },
        }

    def _save_settings(self, settings: Dict[str, Any]):
        try:
            with open(SETTINGS, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        if not os.path.exists(SETTINGS):
            self._on_wav_play_mode_changed()
            return
        try:
            with open(SETTINGS, 'r', encoding='utf-8') as f:
                s = json.load(f)
        except Exception:
            self._on_wav_play_mode_changed()
            return

        io = s.get('io', {})
        self.le_ao.setText(io.get('ao_channels', self.le_ao.text()))
        self.sb_fs.setValue(float(io.get('sample_rate', self.sb_fs.value())))
        self.sb_vmin.setValue(float(io.get('ao_vmin', self.sb_vmin.value())))
        self.sb_vmax.setValue(float(io.get('ao_vmax', self.sb_vmax.value())))
        self.sb_buf.setValue(float(io.get('buffer_duration_s', self.sb_buf.value())))
        self.sb_view.setValue(float(s.get('ui', {}).get('preview_window_s', self.sb_view.value())))
        self.cb_clip.setChecked(bool(io.get('clip_output', self.cb_clip.isChecked())))

        self.cmb_source.setCurrentText(str(s.get('source_mode', self.cmb_source.currentText())))

        sw = s.get('sweep', {})
        self.sb_sw_f0.setValue(float(sw.get('f_start_hz', self.sb_sw_f0.value())))
        self.sb_sw_f1.setValue(float(sw.get('f_stop_hz', self.sb_sw_f1.value())))
        self.sb_sw_dur.setValue(float(sw.get('duration_s', self.sb_sw_dur.value())))
        self.cmb_sw_mode.setCurrentText(str(sw.get('mode', self.cmb_sw_mode.currentText())))
        self.cmb_sw_dir.setCurrentText(str(sw.get('direction', self.cmb_sw_dir.currentText())))
        self.cmb_sw_type.setCurrentText(str(sw.get('sweep_type', self.cmb_sw_type.currentText())))
        self.sb_sw_step.setValue(float(sw.get('step_hz', self.sb_sw_step.value())))
        self.sb_sw_step_dwell.setValue(float(sw.get('step_dwell_s', self.sb_sw_step_dwell.value())))
        self.sb_sw_fade_in.setValue(float(sw.get('fade_in_s', self.sb_sw_fade_in.value())))
        self.sb_sw_fade_out.setValue(float(sw.get('fade_out_s', self.sb_sw_fade_out.value())))
        self._on_sweep_mode_changed()
        self._update_sweep_speed_label()

        wav = s.get('wav', {})
        self.le_wav.setText(str(wav.get('wav_path', self.le_wav.text())))
        self.cb_wav_norm.setChecked(bool(wav.get('normalize', self.cb_wav_norm.isChecked())))
        self.sb_wav_gain.setValue(float(wav.get('gain', self.sb_wav_gain.value())))
        self.sb_wav_off.setValue(float(wav.get('offset_v', self.sb_wav_off.value())))
        map_saved = wav.get('ao_to_wav_map', [])
        if isinstance(map_saved, list):
            self.le_wav_map.setText(','.join(str(int(x)) for x in map_saved))
        self.cmb_wav_play.setCurrentText(str(wav.get('play_mode', self.cmb_wav_play.currentText())))
        self.sb_wav_repeat.setValue(int(wav.get('repeat_count', self.sb_wav_repeat.value())))
        self.sb_wav_maxdur.setValue(float(wav.get('max_duration_s', self.sb_wav_maxdur.value())))
        self._on_wav_play_mode_changed()

        saved_channels = s.get('channels', [])
        if not saved_channels:
            return
        saved = {ao_key(c.get('ao_channel', '')): c for c in saved_channels}
        ao_list = expand_ao_channels(self.le_ao.text())
        if not ao_list:
            return
        self._clear_channel_layout()
        self.channel_panels = []
        for i, ch in enumerate(ao_list):
            cfg = saved.get(ao_key(ch), default_channel_cfg(i, ch))
            cfg['ao_channel'] = ch
            panel = ChannelPanel(i, cfg)
            panel.changed.connect(self._on_live_params_changed)
            self.channel_panels.append(panel)
            tab = QtWidgets.QWidget()
            lt = QtWidgets.QVBoxLayout(tab)
            lt.setContentsMargins(6, 6, 6, 6)
            lt.addWidget(panel)
            lt.addStretch(1)
            self.tabs_channels.addTab(tab, f"CH {i + 1}")

    def _set_plot(self, fs: float, data: np.ndarray, names: List[str]):
        if data.ndim != 2:
            return

        n = data.shape[1]
        n_win = int(round(max(1e-6, float(self.sb_view.value())) * float(fs)))
        n_win = max(64, min(n, n_win))
        view = data[:, -n_win:]
        t = np.arange(n_win, dtype=float) / float(fs)

        max_pts = 3000
        if n_win > max_pts:
            step = int(np.ceil(n_win / max_pts))
            view = view[:, ::step]
            t = t[::step]

        if self.curve_names != names or len(self.curves) != view.shape[0]:
            self.plot.clear()
            self.plot.addLegend()
            self.curves = []
            self.curve_names = list(names)
            for i, name in enumerate(names):
                self.curves.append(self.plot.plot(name=name, pen=pg.mkPen(pg.intColor(i), width=1.8)))
        for i, c in enumerate(self.curves):
            c.setData(t, view[i])

        y_min = float(np.min(view))
        y_max = float(np.max(view))
        span = y_max - y_min
        if span < 1e-9:
            span = 1.0
            y_min -= 0.5
            y_max += 0.5
        margin = 0.08 * span
        self.plot.setXRange(float(t[0]), float(t[-1]) if t.size > 1 else float(t[0] + 1e-6), padding=0)
        self.plot.setYRange(y_min - margin, y_max + margin, padding=0)

    def _refresh_preview_only(self, settings: Optional[Dict[str, Any]] = None):
        if settings is None:
            settings = self._collect()
        try:
            payload = prepare_output_payload(
                settings,
                preview_window_s=float(self.sb_view.value()) * self._preview_cap_factor,
            )
        except Exception as e:
            self.plot.clear()
            self.plot.addLegend()
            self.curves = []
            self.curve_names = []
            self.lbl_status.setText(str(e))
            return

        self._set_plot(payload['fs'], payload['data'], payload['channels'])
        if self.worker is None or not self.worker.isRunning():
            mode = 'SIM preview only' if not HAS_NI else 'Preview only (not running)'
            if payload['clipped'] > 0 and bool(settings['io'].get('clip_output', True)):
                mode += f" | clipped samples: {payload['clipped']}"
            self.lbl_status.setText(mode)

    def _flush_live_update(self):
        settings = self._collect()
        if self.worker and self.worker.isRunning():
            self.worker.update_settings(settings)
            return
        self._refresh_preview_only(settings)

    def _on_live_params_changed(self):
        # Coalesce fast UI events (slider drags/auto-repeat buttons) into one live update.
        self._live_update_timer.start()

    def _start(self):
        if self.worker and self.worker.isRunning():
            return
        settings = self._collect()
        self._save_settings(settings)
        self.worker = SignalWorker(settings)
        self.worker.preview.connect(self._on_worker_preview)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()

    def _on_worker_preview(self, payload: Dict[str, Any]):
        fs = float(payload['fs'])
        channels = list(payload['channels'])
        data = np.asarray(payload['data'], dtype=np.float64)
        self._set_plot(fs, data, channels)

    def _on_worker_failed(self, text: str):
        QtWidgets.QMessageBox.critical(self, 'Generator failed', text)

    def _on_worker_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.worker = None
        self._refresh_preview_only()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(1500)
        super().closeEvent(event)


def run():
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=False)
    w = MainWindow()
    w.show()
    app.exec_()
