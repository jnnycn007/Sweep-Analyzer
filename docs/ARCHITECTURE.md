**Overview**
SW3 + Power Analyzer is a single‑window desktop app that aligns optical irradiance measurements (SW3/eegbin files) with electrical power logs (CSV). The app builds a project in memory (files, groups, associations, trims), computes aligned time series, and produces plots for inspection and export.

**Data Flow**
1. Load SW3/eegbin files and power CSVs into the project.
2. Create groups and assign files to groups.
3. Map SW3 files to power files (many‑to‑many) within a group.
4. Apply group‑level trims (start/end) to define the analysis window.
5. Resample power data, compute EMAs, and align by nearest timestamps.
   - Power CSV loading reads only required columns and caches parsed dataframes by file signature.
   - Near-regular resampled power cadence uses a fast EMA path for large runs.
   - Very long windows auto-upscale power resample interval to keep preprocessing tractable.
6. Break plotted lines across large timestamp gaps so outages are visually blank.
7. Plot IVT, OVP, decay overlays, and correlation/scatter; optionally export aligned CSV.
8. OVP alignment prep runs in a background worker; UI actions warn while data is still building.
9. OVP plotting uses prepared SW3 and power series directly, not only aligned overlap rows, so partial streams are preserved.

**Core Modules**
- `gui.py`: Tkinter UI, session management, and all analysis/plot logic.
- `eegbin.py`: File format reader/writer and data models for SW3 scans.
- `util.py`: Small helpers (inclusive range, optional imgui UI helpers).
- `hooks/rth_pil_tk.py`: PyInstaller runtime hook for Pillow/Tk.
- `.github/workflows/build.yml`: CI workflow to build cross‑platform binaries.

**Key Data Models**
- `FileRecord`: Project‑level record of a single SW3 or power CSV file.
- `GroupRecord`: A group of files plus trim settings and SW3↔Power associations.
- `LampScan` and `GoniometerRow`: Parsed SW3/eegbin data structures.
- Aligned DataFrame: A merged time series with columns for intensity and power EMAs plus provenance IDs/labels.

**Plot Categories**
- IVT (Intensity vs Time): Per file or group; optional EMA and normalization.
- OVP (Optics vs Power): Group‑level dual‑axis time series.
- Correlation/Scatter: Cross‑correlation and regression view of aligned data.
- GDO (Group Decay Overlay): Normalized decay curves per group.

**Design Notes**
- The GUI intentionally keeps all state in memory and stores session JSON for repeatability.
- Eegbin parsing is isolated in `eegbin.py`; the GUI treats it as a pure loader.
- Normalization to 1 m is applied by inverse‑square scaling of irradiance values.
- EMA can run in time‑aware mode so smoothing remains stable when sample cadence changes.
