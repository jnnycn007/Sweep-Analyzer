# SW3 + Power Analyzer

Tkinter GUI to align and analyze **SW3/eegbin** optical measurements with **electrical power** logs (CSV).  
Includes grouping, many‑to‑many associations, global **trimming (HH:MM:SS)**, and plots: **Intensity vs Time**, **Optics vs Power**, **Decay Overlay**, and correlation/scatter.

![Quick Start Guide - How To](https://github.com/OSLUV/Sweep-Analyzer/blob/main/HowTo.png)

## Features
- Load SW3/eegbin files and power CSV logs
- Group files and define SW3 ↔ Power associations
- Trim at group level and per‑plot (HH:MM:SS or seconds)
- Align by epoch timestamp with nearest‑merge tolerance
- Time-aware EMA option for irregular SW3 sampling cadence
- Automatic cadence-cliff compensation for SW3 filtering (sample-count aware)
- Faster power preprocessing for long runs (column-selective CSV load, session cache, fast EMA on near-uniform cadence)
- Non-blocking OVP alignment prep in a background worker (warns while data is still building)
- Power log gaps render as blank intervals (no misleading flat connectors)
- IVT, OVP, decay overlay, and correlation/scatter plots
- Session save/load with full UI state

## Docs
- [User Guide](docs/USER_GUIDE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [File Formats](docs/FILE_FORMATS.md)
- [Development](docs/DEVELOPMENT.md)
- [Testing](docs/TESTING.md)

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python gui.py
```

## Run tests

```bash
./venv/bin/python -m unittest discover -s tests -v
```

## Tune on Open Excimer Dataset

```bash
./venv/bin/python scripts/tune_open_excimer.py \
  --dataset-dir "OSLUV Data/OSLUV Experiments/Open Excimer" \
  --write-json docs/open_excimer_recommendations.json
```

> Dependencies are pinned in `requirements.txt`. Update cautiously; SciPy/Matplotlib wheels must be available for your Python/OS/arch.

## Continuous Integration (GitHub Actions)

This repo includes a cross‑platform workflow to build on **Ubuntu, Windows, and macOS (Intel & Apple Silicon)** and attach artifacts to releases when you push a tag starting with `v` (e.g. `v1.0.0`).  
See [`.github/workflows/build.yml`](.github/workflows/build.yml) for details.

### Usage
1. Commit and push to `main` to build and upload CI artifacts.
2. Create a version tag to publish a release:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

### macOS Gatekeeper

The macOS `.app` bundle is **unsigned**. Open it via right‑click → Open (or code sign & notarize with your Apple Developer ID). You will then need to open System Settings → Privacy and Security → Security and click "Open Anyway"

![MacOS Security Check](https://github.com/OSLUV/Sweep-Analyzer/blob/main/mac_security.png)

## Notes

- The GUI uses `TkAgg` and bundles Tcl/Tk via PyInstaller. On Linux CI we also install the `tk` package so import works at build‑time.
- The app dynamically imports `eegbin`, so we include it as a PyInstaller **hidden import** (same for `util`).
- On Linux you must run from inside the extracted folder so bundled Tcl/Tk & Pillow binaries are found.
```bash
tar -xzf SW3PowerAnalyzer-linux-<arch>.tar.gz
cd SW3PowerAnalyzer
./SW3PowerAnalyzer
```
