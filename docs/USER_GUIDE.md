**Quick Start**
1. Launch the app: `python gui.py`.
2. Add SW3/eegbin files via File → Add SW3…
3. Add power CSV files via File → Add Power CSV…
4. Create a group, assign files to it, and map SW3 ↔ Power associations.
5. Adjust trims and analysis controls, then plot.

**Core Workflow**
- Files are loaded into the project with friendly labels.
- Groups are used to organize datasets and apply trims.
- Associations define which SW3 files align against which power logs.
- Alignment is done by nearest timestamps with a tolerance.

**Intensity vs Time (IVT)**
- Choose a single SW3 file or a group.
- Optional normalization to 1 m applies inverse‑square scaling.
- Trims can be applied per plot and at the group level.
- `Time-aware EMA` makes smoothing use elapsed time, which is more reliable when SW3 sampling cadence drifts over long runs.
- SW3 filtering now includes automatic cadence-cliff compensation based on sample-count onset within each file.
- The plot supports an interactive spectrum inspector (click a point/line).

**Optics vs Power (OVP)**
- Select a group and run Analyze Group.
- Analyze now prepares aligned data in a background worker.
  - If data is not ready yet, the app warns instead of freezing.
  - When status says aligned data is ready, run Analyze Group again to plot.
  - Session load now auto-starts preprocessing for all plot paths:
    - SW3 plot cache for IVT/GDO,
    - aligned OVP/correlation cache for the active/best-associated group.
- Power is resampled and EMA‑smoothed before alignment.
- Power preprocessing is optimized for large logs:
  - only required CSV columns are loaded,
  - parsed power logs are cached for the session,
  - near-uniform resampled cadence uses a fast EMA path,
  - very long windows auto-increase power resample seconds to bound point count.
- Optical EMA and time-aware mode are inherited from the IVT tab's global settings.
- `Auto from optics+cadence` can derive Power EMA automatically from optical EMA and observed dataset cadence.
- Missing intervals in power logs are shown as gaps (blank), not connected flat lines.
- OVP now plots optical and power streams independently, so optical-only and power-only time regions remain visible.
- Long visual legends are automatically moved outside the plot area.
- Plots show optical irradiance and power on twin axes.

**Correlation & Scatter**
- Computes cross‑correlation over a configurable lag window.
- Scatter plot shows intensity vs power with a regression line.

**Group Decay Overlay (GDO)**
- Plots normalized EMA decay per group.
- Uses the same global optical EMA and time-aware settings from the IVT tab.
- Useful for comparing decay profiles across experiments.

**Exporting**
- Export aligned CSV after Analyze Group.
- Saved columns include timestamps, EMAs, IDs, and friendly labels.

**Troubleshooting**
- If a CSV is not recognized, update candidate column names in Settings → Power Columns…
- If plots show no data, verify trims and timestamp overlap.
- If spectral inspection opens nothing, the selected rows may not contain spectral data.
- If analysis is still slow on very large groups, increase `Resample (s)` to reduce the number of power points before smoothing/alignment.
- Reloading files invalidates old aligned cache and automatically queues a background reprocess of the last analyzed/active group.
