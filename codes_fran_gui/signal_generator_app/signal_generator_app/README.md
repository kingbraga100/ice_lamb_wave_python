# Standalone NI Signal Generator

App dedicata solo alla generazione di segnale (senza sweep/analisi).

## Run

```bash
python main.py
```

## Funzioni principali

- Output NI AO continuo con rigenerazione hardware.
- Modalita sorgente:
  - `Channel Waveforms` (multi-canale live)
  - `Sweep (Chirp/Step)` con:
    - `Chirp` o `Step`
    - direzione `Up`, `Down`, `Up-Down`
    - fade `in/out`
    - indicatore velocita sweep in `Hz/s`
  - `WAV File` (free-run loop, repeat N, max duration)
- WAV:
  - supporto PCM int e float (`float32/float64` dove disponibile via SciPy)
  - mapping esplicito `WAV->AO` con lista (es. `2,0,1` => `ao0<-ch2`, `ao1 mute`, `ao2<-ch1`)
- Parametri live per ogni canale:
  - waveform (`Sine`, `Square`, `Triangle`, `Sawtooth`, `DC`, `Noise`)
  - frequency, amplitude, offset, phase
  - duty cycle (square)
  - symmetry (triangle/sawtooth)
  - enable/disable e mapping AO
- Visualizzazione live del segnale generato (preview software).
- Modalità simulazione automatica se NI-DAQmx non è disponibile.
