#!/usr/bin/env python3
"""Generate slowdown/cadence plots for Open Excimer SW3 files."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import detect_cadence_cliff_by_samples
from scripts.tune_open_excimer import Sw3Series, parse_sw3_integral_series


def short_name(name: str) -> str:
    n = name.replace("B1_OpenEx_20uH_", "").replace(".sw3", "")
    return n


def binned_median_curve(timestamps: np.ndarray, bins: int = 24):
    t = np.asarray(timestamps, dtype=float)
    if t.size < 3:
        x = np.linspace(0.0, 100.0, bins)
        y = np.full_like(x, np.nan, dtype=float)
        return x, y

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        x = np.linspace(0.0, 100.0, bins)
        y = np.full_like(x, np.nan, dtype=float)
        return x, y

    edges = np.linspace(0, dt.size, bins + 1, dtype=int)
    y: List[float] = []
    x: List[float] = []
    for i in range(bins):
        a = edges[i]
        b = edges[i + 1]
        seg = dt[a:b]
        if seg.size == 0:
            y.append(float("nan"))
        else:
            y.append(float(np.median(seg)))
        x.append(((i + 0.5) / bins) * 100.0)
    return np.asarray(x), np.asarray(y)


def binned_quantile_curves(
    timestamps: np.ndarray, bins: int = 24
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(timestamps, dtype=float)
    if t.size < 3:
        x = np.linspace(0.0, 100.0, bins)
        y50 = np.full_like(x, np.nan, dtype=float)
        y95 = np.full_like(x, np.nan, dtype=float)
        return x, y50, y95

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        x = np.linspace(0.0, 100.0, bins)
        y50 = np.full_like(x, np.nan, dtype=float)
        y95 = np.full_like(x, np.nan, dtype=float)
        return x, y50, y95

    edges = np.linspace(0, dt.size, bins + 1, dtype=int)
    x: List[float] = []
    y50: List[float] = []
    y95: List[float] = []
    for i in range(bins):
        a = edges[i]
        b = edges[i + 1]
        seg = dt[a:b]
        if seg.size == 0:
            y50.append(float("nan"))
            y95.append(float("nan"))
        else:
            y50.append(float(np.percentile(seg, 50)))
            y95.append(float(np.percentile(seg, 95)))
        x.append(((i + 0.5) / bins) * 100.0)
    return np.asarray(x), np.asarray(y50), np.asarray(y95)


def baseline_dt_s(timestamps: np.ndarray) -> float:
    t = np.asarray(timestamps, dtype=float)
    if t.size < 3:
        return float("nan")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan")
    n0 = max(10, int(0.2 * dt.size))
    n0 = min(n0, dt.size)
    return float(np.median(dt[:n0]))


def detect_cliff_onset_pct(
    x_pct: np.ndarray,
    ratio95: np.ndarray,
    threshold: float = 3.0,
    sustain_window: int = 5,
    sustain_hits: int = 3,
    start_after_pct: float = 20.0,
) -> Optional[float]:
    valid = np.isfinite(ratio95)
    if not valid.any():
        return None

    x = x_pct[valid]
    r = ratio95[valid]
    n = r.size
    if n < sustain_window:
        return float(x[np.argmax(r)]) if np.nanmax(r) >= threshold else None

    for i in range(n - sustain_window + 1):
        if x[i] < start_after_pct:
            continue
        w = r[i : i + sustain_window]
        if np.sum(w >= threshold) >= sustain_hits and np.median(w) >= threshold:
            return float(x[i])
    return None


def save_slowdown_bar(sw3: List[Sw3Series], out_path: Path):
    rows = []
    for s in sw3:
        m = s.metrics
        rows.append((short_name(s.path.name), m.slowdown_ratio, m.early_dt_s, m.late_dt_s))
    rows.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else -1), reverse=True)

    labels = [r[0] for r in rows]
    ratios = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    bars = ax.bar(labels, ratios, color="#3465a4", alpha=0.88)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_title("SW3 Sampling Slowdown Ratio by File (late median dt / early median dt)")
    ax.set_ylabel("Slowdown ratio")
    ax.set_xlabel("SW3 file chunk")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=35)

    for b, v in zip(bars, ratios):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_progress_curves(sw3: List[Sw3Series], out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    cmap = plt.get_cmap("tab10")

    for i, s in enumerate(sw3):
        x, y = binned_median_curve(s.timestamps, bins=24)
        label = f"{short_name(s.path.name)} (x{s.metrics.slowdown_ratio:.2f})"
        ax.plot(x, y, linewidth=2.0, alpha=0.9, color=cmap(i % 10), label=label)

    ax.set_title("SW3 Sampling Interval Drift Within Each File")
    ax.set_xlabel("File progress (%)")
    ax.set_ylabel("Median sampling interval per progress bin (s)")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Keep main plot readable by moving legend outside.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(out_path)
    plt.close(fig)


def save_progress_cliff_curves(sw3: List[Sw3Series], out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    cmap = plt.get_cmap("tab10")

    for i, s in enumerate(sw3):
        x, y50, y95 = binned_quantile_curves(s.timestamps, bins=24)
        base = baseline_dt_s(s.timestamps)
        if not np.isfinite(base) or base <= 0:
            continue

        r50 = y50 / base
        r95 = y95 / base
        cliff = detect_cliff_onset_pct(x, r95, threshold=3.0, sustain_window=5, sustain_hits=3, start_after_pct=20.0)

        name = short_name(s.path.name)
        if cliff is None:
            label = f"{name} (no cliff)"
        else:
            label = f"{name} (cliff~{cliff:.1f}%)"

        c = cmap(i % 10)
        ax.plot(x, r50, linewidth=2.0, alpha=0.95, color=c, label=label)
        ax.plot(x, r95, linewidth=1.5, alpha=0.7, color=c, linestyle="--")

    ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.2)
    ax.axhline(3.0, color="#aa0000", linestyle="--", linewidth=1.2)
    ax.set_title("SW3 Cliff View: Interval / Early Baseline (solid=P50, dashed=P95)")
    ax.set_xlabel("File progress (%)")
    ax.set_ylabel("Interval ratio vs early baseline")
    ax.set_ylim(0.6, None)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(out_path)
    plt.close(fig)


def save_cliff_onset_bar(sw3: List[Sw3Series], out_path: Path):
    rows = []
    for s in sw3:
        x, _, y95 = binned_quantile_curves(s.timestamps, bins=24)
        base = baseline_dt_s(s.timestamps)
        if not np.isfinite(base) or base <= 0:
            continue
        ratio95 = y95 / base
        cliff = detect_cliff_onset_pct(x, ratio95, threshold=3.0, sustain_window=5, sustain_hits=3, start_after_pct=20.0)
        max95 = float(np.nanmax(ratio95)) if np.isfinite(ratio95).any() else float("nan")
        rows.append((short_name(s.path.name), cliff, max95))

    rows.sort(key=lambda r: (r[1] if r[1] is not None else 999.0))
    labels = [r[0] for r in rows]
    values = [r[1] if r[1] is not None else 100.0 for r in rows]
    colors = ["#cc4444" if r[1] is not None else "#999999" for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_title("SW3 Estimated Cliff Onset (% progress where P95 >= 3x baseline, sustained)")
    ax.set_ylabel("Cliff onset progress (%)")
    ax.set_xlabel("SW3 file chunk")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=35)

    for b, (_, cliff, max95) in zip(bars, rows):
        txt = "none" if cliff is None else f"{cliff:.1f}%"
        if np.isfinite(max95):
            txt += f"\\nmax95={max95:.1f}x"
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 1.2, txt, ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_cliff_onset_samples_bar(sw3: List[Sw3Series], out_path: Path):
    rows = []
    for s in sw3:
        base_dt, cliff_idx = detect_cadence_cliff_by_samples(s.timestamps, threshold_ratio=3.0)
        pct = (100.0 * cliff_idx / max(len(s.timestamps) - 1, 1)) if cliff_idx is not None else float("nan")
        rows.append((short_name(s.path.name), cliff_idx, pct, base_dt))

    rows.sort(key=lambda r: (r[1] if r[1] is not None else 10**9))
    labels = [r[0] for r in rows]
    values = [r[1] if r[1] is not None else 0 for r in rows]
    colors = ["#b22222" if r[1] is not None else "#999999" for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_title("SW3 Cliff Onset by Sample Count (sustained cadence slowdown)")
    ax.set_ylabel("Sample index at cliff onset")
    ax.set_xlabel("SW3 file chunk")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=35)

    for b, (_, idx, pct, _) in zip(bars, rows):
        if idx is None:
            txt = "none"
        else:
            txt = f"{idx}\\n({pct:.1f}%)"
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + max(max(values) * 0.01, 2), txt, ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary_csv(sw3: List[Sw3Series], out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as fd:
        w = csv.writer(fd)
        w.writerow([
            "file",
            "count",
            "hours",
            "median_dt_s",
            "p90_dt_s",
            "p99_dt_s",
            "early_dt_s",
            "late_dt_s",
            "slowdown_ratio",
            "gaps_gt_5x_median",
            "baseline_dt_s",
            "cliff_sample_idx_ratio_ge_3x_sustained",
            "cliff_onset_pct_p95_ge_3x_sustained",
            "max_p95_ratio",
        ])
        for s in sw3:
            m = s.metrics
            _, cliff_sample_idx = detect_cadence_cliff_by_samples(s.timestamps, threshold_ratio=3.0)
            x, _, y95 = binned_quantile_curves(s.timestamps, bins=24)
            base = baseline_dt_s(s.timestamps)
            if np.isfinite(base) and base > 0:
                r95 = y95 / base
            else:
                r95 = np.full_like(y95, np.nan)
            cliff = detect_cliff_onset_pct(x, r95, threshold=3.0, sustain_window=5, sustain_hits=3, start_after_pct=20.0)
            max95 = float(np.nanmax(r95)) if np.isfinite(r95).any() else float("nan")
            w.writerow([
                s.path.name,
                m.count,
                f"{m.duration_h:.4f}",
                f"{m.median_dt_s:.6f}",
                f"{m.p90_dt_s:.6f}",
                f"{m.p99_dt_s:.6f}",
                f"{m.early_dt_s:.6f}",
                f"{m.late_dt_s:.6f}",
                f"{m.slowdown_ratio:.6f}",
                m.gaps_gt_5x_median,
                f"{base:.6f}" if np.isfinite(base) else "",
                cliff_sample_idx if cliff_sample_idx is not None else "",
                f"{cliff:.3f}" if cliff is not None else "",
                f"{max95:.6f}" if np.isfinite(max95) else "",
            ])


def main():
    parser = argparse.ArgumentParser(description="Plot slowdown/cadence outputs for Open Excimer SW3 files")
    parser.add_argument(
        "--dataset-dir",
        default="OSLUV Data/OSLUV Experiments/Open Excimer",
        help="Directory containing SW3 files",
    )
    parser.add_argument(
        "--output-dir",
        default="OSLUV Data/OSLUV Experiments/Open Excimer/plots/tuning",
        help="Directory for generated plots",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sw3_paths = sorted(dataset_dir.glob("*.sw3"))
    if not sw3_paths:
        raise SystemExit(f"No SW3 files found in {dataset_dir}")

    sw3: List[Sw3Series] = []
    for path in sw3_paths:
        print(f"Parsing {path.name}")
        sw3.append(parse_sw3_integral_series(path))

    bar_path = out_dir / "sw3_slowdown_ratio_bar.png"
    curve_path = out_dir / "sw3_sampling_interval_progress.png"
    cliff_curve_path = out_dir / "sw3_sampling_interval_cliff_view.png"
    cliff_bar_path = out_dir / "sw3_cliff_onset_bar.png"
    cliff_samples_path = out_dir / "sw3_cliff_onset_samples_bar.png"
    csv_path = out_dir / "sw3_slowdown_summary.csv"

    save_slowdown_bar(sw3, bar_path)
    save_progress_curves(sw3, curve_path)
    save_progress_cliff_curves(sw3, cliff_curve_path)
    save_cliff_onset_bar(sw3, cliff_bar_path)
    save_cliff_onset_samples_bar(sw3, cliff_samples_path)
    write_summary_csv(sw3, csv_path)

    print(f"Wrote: {bar_path}")
    print(f"Wrote: {curve_path}")
    print(f"Wrote: {cliff_curve_path}")
    print(f"Wrote: {cliff_bar_path}")
    print(f"Wrote: {cliff_samples_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
