# Lamb-Wave Ice Detection Python Codes

This repository contains the Python codes developed for a research project on Lamb-wave based ice detection using piezoelectric transducers.

The scripts were used to acquire, process, visualize and analyze guided-wave signals measured on a plate under baseline, temperature-varying and ice-loaded conditions.

## Project overview

The objective of the work is to investigate whether piezoelectric transducers can be used to detect ice formation through changes in Lamb-wave signals.

The analysis focuses mainly on the A0 wave packet. Several signal features are extracted, including:

- envelope amplitude
- signal energy
- waveform correlation
- signal difference coefficient (SDC)
- time-of-flight shift
- envelope and waveform comparisons

A simple regression-based temperature-compensation method is also included. This is used to separate normal temperature-dependent signal variations from ice-induced changes.

## Repository structure

```text
ice_lamb_wave_python/
│
├── data_temperature_ice/
│   └── May 17 temperature and ice datasets
│
├── data_rest/
│   └── preliminary tests, washer/glaze cases, and other saved data
│
├── scripts/
│   ├── acquisition/
│   ├── characterization/
│   ├── validation/
│   ├── temperature_ice_analysis/
│   └── machine_learning/
│
├── figures/
│   └── generated figures
│
├── README.md
├── requirements.txt
└── .gitignore
