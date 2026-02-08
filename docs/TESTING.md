**Automated Tests**
Run the regression suite from repo root:

```bash
./venv/bin/python -m unittest discover -s tests -v
```

Current tests cover:
- time-aware EMA behavior across cadence shifts
- explicit NaN gap insertion for plotting
- power preprocessing behavior across data outages
- dataset smoke check for Open Excimer folder presence

**Dataset Tuning Workflow**
Use the Open Excimer long-run data to derive suggested GUI defaults:

```bash
./venv/bin/python scripts/tune_open_excimer.py \
  --dataset-dir "OSLUV Data/OSLUV Experiments/Open Excimer" \
  --write-json docs/open_excimer_recommendations.json
```

This prints cadence summaries for SW3/CSV files and suggested values for:
- `time_weighted_ema`
- `intensity_ema_span`
- `power_ema_span`
- `overlay_ema_span`
- `resample_seconds`
- `align_tolerance_s`

To generate slowdown/cliff visuals:

```bash
./venv/bin/python scripts/plot_open_excimer_tuning.py \
  --dataset-dir "OSLUV Data/OSLUV Experiments/Open Excimer" \
  --output-dir "OSLUV Data/OSLUV Experiments/Open Excimer/plots/tuning"
```
