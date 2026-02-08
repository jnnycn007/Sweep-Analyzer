#!/usr/bin/env python3
"""Analyze long-run SW3 + power logs and suggest plotting/tuning settings.

This tool is designed for the Open Excimer dataset pattern (multiple SW3 chunks and
multiple power CSV chunks). It extracts cadence/gap characteristics and proposes
GUI defaults for time-aware smoothing and resampling.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import WM2_TO_UW_CM2, detect_cadence_cliff_by_samples, ema_timeaware

EEGBIN3_MAGIC = b"EEGBIN3\n"
EEGBIN3_SEP = b"\n\nGZIP FOLLOWS\n\n"
ROW_PAIR_RE = re.compile(
    rb'"timestamp"\s*:\s*([-+0-9.eE]+)\s*,\s*"integral_result"\s*:\s*([-+0-9.eE]+)'
)

TS_CANDIDATES = ["Timestamp", "timestamp", "epoch", "Epoch", "time", "Time"]
W_CANDIDATES = ["W_Active", "watts", "Watts", "Power_W", "W"]
SPAN_CANDIDATES = [15, 31, 61, 121, 241]


@dataclass
class CadenceMetrics:
    count: int
    first_ts: float
    last_ts: float
    duration_h: float
    median_dt_s: float
    p90_dt_s: float
    p99_dt_s: float
    early_dt_s: float
    late_dt_s: float
    slowdown_ratio: float
    gaps_gt_5x_median: int


@dataclass
class Sw3Series:
    path: Path
    timestamps: np.ndarray
    intensity_uW_cm2: np.ndarray
    metrics: CadenceMetrics


def _find_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    colmap = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in colmap:
            return colmap[cand.lower()]
    return None


def _read_sw3_metadata_bytes(path: Path, chunk_size: int = 1024 * 1024) -> bytes:
    with path.open("rb") as fd:
        magic = fd.read(len(EEGBIN3_MAGIC))
        if magic != EEGBIN3_MAGIC:
            raise ValueError(f"Unsupported/non-v3 SW3 file: {path}")

        buf = bytearray()
        while True:
            chunk = fd.read(chunk_size)
            if not chunk:
                raise ValueError(f"Failed to find v3 metadata separator in {path}")
            buf.extend(chunk)
            idx = buf.find(EEGBIN3_SEP)
            if idx != -1:
                return bytes(buf[:idx])


def _series_cadence_metrics(timestamps: np.ndarray) -> CadenceMetrics:
    ts = np.asarray(timestamps, dtype=float)
    if ts.size < 2:
        return CadenceMetrics(
            count=int(ts.size),
            first_ts=float(ts[0]) if ts.size else float("nan"),
            last_ts=float(ts[-1]) if ts.size else float("nan"),
            duration_h=0.0,
            median_dt_s=float("nan"),
            p90_dt_s=float("nan"),
            p99_dt_s=float("nan"),
            early_dt_s=float("nan"),
            late_dt_s=float("nan"),
            slowdown_ratio=float("nan"),
            gaps_gt_5x_median=0,
        )

    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return CadenceMetrics(
            count=int(ts.size),
            first_ts=float(ts[0]),
            last_ts=float(ts[-1]),
            duration_h=max(0.0, (float(ts[-1]) - float(ts[0])) / 3600.0),
            median_dt_s=float("nan"),
            p90_dt_s=float("nan"),
            p99_dt_s=float("nan"),
            early_dt_s=float("nan"),
            late_dt_s=float("nan"),
            slowdown_ratio=float("nan"),
            gaps_gt_5x_median=0,
        )

    k = max(1, dt.size // 10)
    median_dt = float(np.median(dt))
    early = float(np.median(dt[:k]))
    late = float(np.median(dt[-k:]))
    slowdown = float(late / early) if early > 0 else float("nan")
    gaps = int(np.sum(dt > (5.0 * median_dt)))
    return CadenceMetrics(
        count=int(ts.size),
        first_ts=float(ts[0]),
        last_ts=float(ts[-1]),
        duration_h=max(0.0, (float(ts[-1]) - float(ts[0])) / 3600.0),
        median_dt_s=median_dt,
        p90_dt_s=float(np.percentile(dt, 90)),
        p99_dt_s=float(np.percentile(dt, 99)),
        early_dt_s=early,
        late_dt_s=late,
        slowdown_ratio=slowdown,
        gaps_gt_5x_median=gaps,
    )


def parse_sw3_integral_series(path: Path) -> Sw3Series:
    metadata = _read_sw3_metadata_bytes(path)

    t: List[float] = []
    y: List[float] = []
    for m in ROW_PAIR_RE.finditer(metadata):
        ts = float(m.group(1))
        integral_wm2 = float(m.group(2))
        if not math.isfinite(ts) or not math.isfinite(integral_wm2):
            continue
        t.append(ts)
        y.append(integral_wm2 * WM2_TO_UW_CM2)

    if not t:
        raise ValueError(f"No timestamp/integral rows found in {path}")

    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    y_arr = y_arr[order]
    metrics = _series_cadence_metrics(t_arr)
    return Sw3Series(path=path, timestamps=t_arr, intensity_uW_cm2=y_arr, metrics=metrics)


def load_power_series(path: Path) -> Tuple[np.ndarray, np.ndarray, CadenceMetrics]:
    df = pd.read_csv(path, low_memory=False)
    t_col = _find_col(df.columns, TS_CANDIDATES)
    w_col = _find_col(df.columns, W_CANDIDATES)
    if not t_col or not w_col:
        raise ValueError(f"Missing timestamp/watts columns in {path}")

    out = df[[t_col, w_col]].rename(columns={t_col: "Timestamp", w_col: "W_Active"}).copy()
    out["Timestamp"] = pd.to_numeric(out["Timestamp"], errors="coerce")
    out["W_Active"] = pd.to_numeric(out["W_Active"], errors="coerce")
    out = out.dropna(subset=["Timestamp", "W_Active"]).sort_values("Timestamp")
    t = out["Timestamp"].to_numpy(dtype=float)
    w = out["W_Active"].to_numpy(dtype=float)
    return t, w, _series_cadence_metrics(t)


def roughness(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    dv = np.diff(v)
    mad_dv = float(np.median(np.abs(dv - np.median(dv))))
    scale = float(np.percentile(v, 75) - np.percentile(v, 25))
    if scale <= 0:
        scale = float(np.std(v))
    if scale <= 0:
        return float("nan")
    return mad_dv / scale


def choose_intensity_span(sw3_series: Sequence[Sw3Series]) -> Tuple[int, Dict[int, float]]:
    by_span: Dict[int, float] = {}
    raw_scores = []
    for s in sw3_series:
        raw_scores.append(roughness(s.intensity_uW_cm2))
    raw_target = float(np.nanmedian(raw_scores))

    for span in SPAN_CANDIDATES:
        scores = []
        for s in sw3_series:
            y_ema = ema_timeaware(s.timestamps, s.intensity_uW_cm2, span=span)
            scores.append(roughness(y_ema))
        by_span[span] = float(np.nanmedian(scores))

    # Pick the first span that removes at least ~35% of roughness.
    for span in SPAN_CANDIDATES:
        r = by_span[span]
        if math.isfinite(raw_target) and math.isfinite(r) and r <= (0.65 * raw_target):
            return span, by_span

    # Fallback: span with minimum median roughness.
    best = min(by_span.items(), key=lambda kv: kv[1] if math.isfinite(kv[1]) else float("inf"))[0]
    return int(best), by_span


def recommend_parameters(sw3_series: Sequence[Sw3Series], power_metrics: Sequence[CadenceMetrics]) -> Dict[str, object]:
    sw3_median_dt = float(np.nanmedian([s.metrics.median_dt_s for s in sw3_series]))
    sw3_pph = float(np.nanmedian([s.metrics.count / max(s.metrics.duration_h, 1e-6) for s in sw3_series]))
    slowdown = float(np.nanmedian([s.metrics.slowdown_ratio for s in sw3_series]))
    cliff_samples = []
    for s in sw3_series:
        _, cliff_idx = detect_cadence_cliff_by_samples(s.timestamps, threshold_ratio=3.0)
        if cliff_idx is not None:
            cliff_samples.append(float(cliff_idx))
    cliff_sample_median = float(np.nanmedian(cliff_samples)) if cliff_samples else float("nan")

    power_median_dt = float(np.nanmedian([m.median_dt_s for m in power_metrics]))

    intensity_span, roughness_by_span = choose_intensity_span(sw3_series)

    # Keep power denser than SW3 but not overly noisy.
    # Target ~4-6 power points per SW3 sample period.
    resample_seconds = int(np.clip(round(sw3_median_dt / 5.0), 1, 60)) if math.isfinite(sw3_median_dt) else 5
    effective_power_step = max(power_median_dt, float(resample_seconds)) if math.isfinite(power_median_dt) else float(resample_seconds)

    # Match power smoothing timescale to optical smoothing timescale.
    target_smoothing_s = float(intensity_span) * max(sw3_median_dt, 1.0)
    power_span = int(np.clip(round(target_smoothing_s / max(effective_power_step, 1.0)), 31, 2000))

    align_tolerance_s = int(np.clip(round(max(sw3_median_dt * 0.35, effective_power_step * 2.0)), 2, 120))

    notes = [
        "Use time-aware EMA to remove sample-count bias from variable SW3 cadence.",
        "Keep legend outside plot area for large groups (already implemented).",
        "Render missing log intervals as gaps (already implemented via NaN line breaks).",
    ]

    return {
        "time_weighted_ema": True,
        "intensity_ema_span": int(intensity_span),
        "power_ema_span": int(power_span),
        "overlay_ema_span": int(intensity_span),
        "resample_seconds": int(resample_seconds),
        "align_tolerance_s": int(align_tolerance_s),
        "derived": {
            "sw3_median_dt_s": sw3_median_dt,
            "sw3_points_per_hour": sw3_pph,
            "sw3_slowdown_ratio": slowdown,
            "sw3_cliff_sample_idx_median": cliff_sample_median,
            "power_median_dt_s": power_median_dt,
            "optical_roughness_by_span": roughness_by_span,
        },
        "notes": notes,
    }


def print_table(title: str, rows: List[Dict[str, object]], keys: Sequence[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("(none)")
        return
    widths = {k: max(len(k), *(len(f"{r.get(k, '')}") for r in rows)) for k in keys}
    header = "  ".join(f"{k:{widths[k]}}" for k in keys)
    print(header)
    print("  ".join("-" * widths[k] for k in keys))
    for r in rows:
        print("  ".join(f"{r.get(k, ''):{widths[k]}}" for k in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Open Excimer dataset and recommend GUI settings")
    parser.add_argument(
        "--dataset-dir",
        default="OSLUV Data/OSLUV Experiments/Open Excimer",
        help="Path to folder containing .sw3 and .csv files",
    )
    parser.add_argument(
        "--write-json",
        default="",
        help="Optional output path for recommendation JSON",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    sw3_paths = sorted(dataset_dir.glob("*.sw3"))
    csv_paths = sorted(dataset_dir.glob("*.csv"))
    if not sw3_paths:
        raise SystemExit(f"No .sw3 files found in {dataset_dir}")
    if not csv_paths:
        raise SystemExit(f"No .csv files found in {dataset_dir}")

    print(f"Dataset: {dataset_dir}")
    print(f"SW3 files: {len(sw3_paths)}")
    print(f"CSV files: {len(csv_paths)}")

    sw3_series: List[Sw3Series] = []
    for path in sw3_paths:
        print(f"Analyzing SW3: {path.name}")
        sw3_series.append(parse_sw3_integral_series(path))

    power_info = []
    for path in csv_paths:
        print(f"Analyzing CSV: {path.name}")
        _, _, metrics = load_power_series(path)
        power_info.append((path, metrics))

    sw3_rows = []
    for s in sw3_series:
        m = s.metrics
        _, cliff_idx = detect_cadence_cliff_by_samples(s.timestamps, threshold_ratio=3.0)
        sw3_rows.append(
            {
                "file": s.path.name,
                "count": m.count,
                "hours": f"{m.duration_h:.1f}",
                "med_dt_s": f"{m.median_dt_s:.2f}",
                "p99_dt_s": f"{m.p99_dt_s:.2f}",
                "slowdown": f"{m.slowdown_ratio:.2f}",
                "cliff_idx": cliff_idx if cliff_idx is not None else "",
                "gaps": m.gaps_gt_5x_median,
            }
        )

    power_rows = []
    for path, m in power_info:
        power_rows.append(
            {
                "file": path.name,
                "count": m.count,
                "hours": f"{m.duration_h:.1f}",
                "med_dt_s": f"{m.median_dt_s:.2f}",
                "p99_dt_s": f"{m.p99_dt_s:.2f}",
                "gaps": m.gaps_gt_5x_median,
            }
        )

    print_table("SW3 cadence summary", sw3_rows, ["file", "count", "hours", "med_dt_s", "p99_dt_s", "slowdown", "cliff_idx", "gaps"])
    print_table("Power cadence summary", power_rows, ["file", "count", "hours", "med_dt_s", "p99_dt_s", "gaps"])

    recommendations = recommend_parameters(sw3_series, [m for _, m in power_info])

    print("\nRecommended GUI settings")
    print("------------------------")
    for k in [
        "time_weighted_ema",
        "intensity_ema_span",
        "power_ema_span",
        "overlay_ema_span",
        "resample_seconds",
        "align_tolerance_s",
    ]:
        print(f"{k}: {recommendations[k]}")

    print("\nDerived metrics")
    print("---------------")
    for k, v in recommendations["derived"].items():
        print(f"{k}: {v}")

    if args.write_json:
        out_path = Path(args.write_json)
        out_path.write_text(json.dumps(recommendations, indent=2), encoding="utf-8")
        print(f"\nWrote recommendation JSON: {out_path}")


if __name__ == "__main__":
    main()
