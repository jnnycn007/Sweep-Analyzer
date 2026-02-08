#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SW3 + Power Analyzer GUI

Layout
------
• Single vertical stack: Files (top) → Groups (middle) → Controls (bottom)

Capabilities
------------
• Load SW3/eegbin optical files and power CSV logs
• Label files; create groups; associate SW3↔Power (many-to-many)
• **Group-level trimming** (HH:MM:SS) at start and/or end; applied globally:
  - Intensity vs Time (file & group modes)
  - Optics vs Power (time series + correlation/scatter/export)
  - Group Decay Overlay
• Align streams by epoch timestamps (power resampled to N seconds; nearest-merge with tolerance)
• Per-analysis controls: show/hide points & EMA lines; independent alphas
• Friendly names in legends (no F###/G###), paired colors for OVP
• Sessions save/load everything (files, groups, associations, trims, settings)

Notes
-----
• No dependency on imgui; a tiny util shim is injected if util.py is missing.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from queue import Empty, Queue
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import colormaps as mpl_colormaps

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

# ------------------------------------------------------------------
# Optional util shim (so importing eegbin doesn't require imgui)
# ------------------------------------------------------------------

def _ensure_util_shim():
    """Install a minimal util shim so eegbin can import without imgui."""
    import types
    if 'util' in sys.modules:
        return
    util = types.ModuleType('util')
    def inclusive_range(start, stop, step):
        """Fallback inclusive_range used when util.py is unavailable."""
        if step == 0: return []
        out = []
        if step > 0:
            while start <= stop:
                out.append(start); start += step
        else:
            while start >= stop:
                out.append(start); start += step
        return out
    def do_editable_raw(preamble, value, units="", width=100):
        """No-op placeholder for imgui editable input."""
        return (False, value)
    def do_editable(preamble, value, units="", width=100, enable=True):
        """No-op placeholder for imgui editable input."""
        return value
    util.inclusive_range = inclusive_range
    util.do_editable_raw = do_editable_raw
    util.do_editable = do_editable
    sys.modules['util'] = util

_ensure_util_shim()

# lazily imported module
eegbin = None
_EXTRA_MODULE_PATHS: List[str] = []

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _get_cmap(name: str):
    """Return a Matplotlib colormap by name with fallback for older versions."""
    try:
        return mpl_colormaps.get_cmap(name)
    except Exception:
        return plt.get_cmap(name)

def human_datetime(epoch_s: float) -> str:
    """Format epoch seconds into a local-time human string."""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_s))
    except Exception:
        return ""

def parse_hhmmss(s: str) -> int:
    """Parse seconds or HH:MM:SS into total seconds (supports leading '-')."""
    if s is None:
        return 0
    s = str(s).strip()
    if not s:
        return 0
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    parts = s.split(":")
    parts = [p for p in parts if p != ""]
    try:
        parts = [int(p) for p in parts]
    except Exception as e:
        raise ValueError(f"Invalid time format: {s}") from e
    if len(parts) == 2:
        return parts[0]*60 + parts[1]
    if len(parts) == 3:
        return parts[0]*3600 + parts[1]*60 + parts[2]
    raise ValueError(f"Invalid time format: {s}")

def fmt_hhmmss(seconds: int) -> str:
    """Format seconds as HH:MM:SS (supports negative values)."""
    neg = seconds < 0
    s = abs(int(seconds))
    hh, rem = divmod(s, 3600)
    mm, ss = divmod(rem, 60)
    out = f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"-{out}" if neg else out

def _add_module_path(path: str):
    """Add a folder to sys.path and remember it for the session."""
    if path and path not in sys.path:
        sys.path.insert(0, path)
    if path and path not in _EXTRA_MODULE_PATHS:
        _EXTRA_MODULE_PATHS.append(path)

def ensure_eegbin_imported():
    """Add saved module paths; import eegbin (prompts for folder if needed)."""
    global eegbin
    if eegbin is not None:
        return
    import importlib
    for p in list(_EXTRA_MODULE_PATHS):
        _add_module_path(p)
    try:
        eegbin = importlib.import_module("eegbin")
        return
    except ModuleNotFoundError as e1:
        missing = getattr(e1, "name", "eegbin")
        if messagebox.askyesno("Locate module",
                               "Could not import '{}'.\nSelect the folder that contains 'eegbin.py' and 'util.py'."
                               .format(missing)):
            folder = filedialog.askdirectory(title="Select folder containing eegbin.py and util.py")
            if folder:
                _add_module_path(folder)
                try:
                    eegbin = importlib.import_module("eegbin")
                    return
                except Exception:
                    pass
        messagebox.showerror("Import error",
                             "Could not import 'eegbin'. Missing: {}\n\n"
                             "Tip: Put gui.py next to eegbin.py/util.py, or use Settings → Add Module Path…".format(missing))
        raise

def ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Return an exponential moving average (EMA) with the given span."""
    if span <= 1:
        return np.asarray(arr, dtype=float)
    s = pd.Series(arr, dtype="float64")
    return s.ewm(span=int(span), adjust=False).mean().to_numpy()

def _median_positive_step_s(timestamps_s: np.ndarray) -> float:
    """Return the median positive delta between timestamps in seconds."""
    t = np.asarray(timestamps_s, dtype=float)
    if t.size < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    return float(np.median(dt))

def _is_nearly_regular_cadence(
    timestamps_s: np.ndarray,
    p_lo: float = 5.0,
    p_hi: float = 95.0,
    max_ratio: float = 1.25,
) -> bool:
    """Return True when timestamp deltas are close to a fixed cadence."""
    t = np.asarray(timestamps_s, dtype=float)
    if t.size < 4:
        return True
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 3:
        return True
    lo = float(np.percentile(dt, p_lo))
    hi = float(np.percentile(dt, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0:
        return False
    return (hi / lo) <= float(max_ratio)

def detect_cadence_cliff_by_samples(
    timestamps_s: np.ndarray,
    threshold_ratio: float = 3.0,
) -> Tuple[float, Optional[int]]:
    """Detect cadence cliff in sample-count domain.

    Returns:
    - baseline_dt_s: robust early-file baseline sampling interval (seconds)
    - cliff_sample_idx: sample index where sustained slowdown starts, or None
    """
    t = np.asarray(timestamps_s, dtype=float)
    if t.size < 4:
        return 1.0, None

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 8:
        return _median_positive_step_s(t), None

    baseline_n = int(np.clip(dt.size // 5, 300, 6000))
    baseline_n = min(baseline_n, dt.size)
    baseline_dt = float(np.median(dt[:baseline_n]))
    if not np.isfinite(baseline_dt) or baseline_dt <= 0:
        baseline_dt = _median_positive_step_s(t)
        return baseline_dt, None

    window = int(np.clip(baseline_n // 5, 31, 401))
    rolling = pd.Series(dt, dtype="float64").rolling(window=window, min_periods=window).median().to_numpy()
    ratio = rolling / baseline_dt

    sustain_run = int(np.clip(window // 6, 6, 40))
    start_idx = min(max(baseline_n, window), dt.size - 1)
    hits = 0
    for j in range(start_idx, dt.size):
        r = ratio[j]
        if np.isfinite(r) and r >= float(threshold_ratio):
            hits += 1
        else:
            hits = 0
        if hits >= sustain_run:
            # dt[j] is between samples j and j+1.
            return baseline_dt, int(j + 1)
    return baseline_dt, None

def ema_timeaware(timestamps_s: np.ndarray, values: np.ndarray, span: int) -> np.ndarray:
    """EMA on irregular samples using elapsed time instead of sample count.

    `span` is interpreted as "roughly this many median samples of memory", then
    converted to a time half-life from the observed median sample interval.

    This version is cadence-cliff aware:
    - detects sustained slowdown by sample index (not file percent),
    - increases post-cliff smoothing memory,
    - caps per-step decay for alpha calculation so sparse late samples do not
      become excessively jumpy.
    """
    y = np.asarray(values, dtype=float)
    t = np.asarray(timestamps_s, dtype=float)
    if y.size == 0:
        return y
    if y.size == 1 or span <= 1:
        return y.copy()

    dt_ref = _median_positive_step_s(t)
    baseline_dt, cliff_idx = detect_cadence_cliff_by_samples(t)
    halflife_s = max(1.0, float(max(1, int(span))) * dt_ref)
    gap_reset_s = 10.0 * halflife_s
    dt_alpha_cap_s = max(baseline_dt * 2.5, dt_ref)
    post_cliff_halflife_mult = 1.75
    ln2 = np.log(2.0)

    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, y.size):
        raw_dt = t[i] - t[i - 1]
        if not np.isfinite(raw_dt) or raw_dt <= 0:
            raw_dt = dt_ref
        if raw_dt > gap_reset_s:
            # Hard reset across very large gaps to avoid stale-memory carryover.
            out[i] = y[i]
            continue

        dt_for_alpha = min(raw_dt, dt_alpha_cap_s)
        halflife_i = halflife_s
        if cliff_idx is not None and i >= cliff_idx:
            halflife_i = halflife_s * post_cliff_halflife_mult

        alpha = 1.0 - np.exp(-(ln2 * dt_for_alpha) / halflife_i)
        out[i] = out[i - 1] + alpha * (y[i] - out[i - 1])
    return out

def ema_adaptive(timestamps_s: np.ndarray, values: np.ndarray, span: int, timeaware: bool) -> np.ndarray:
    """Compute EMA in sample-count mode or time-aware mode."""
    if not timeaware:
        return ema(values, span)
    return ema_timeaware(timestamps_s, values, span)

def _insert_nan_gaps(timestamps_s: np.ndarray, values: np.ndarray, min_gap_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Insert NaNs so lines break across large sampling/data gaps."""
    t = np.asarray(timestamps_s, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size <= 1:
        return t, y
    out_t = [t[0]]
    out_y = [y[0]]
    for i in range(1, t.size):
        if (t[i] - t[i - 1]) > float(min_gap_s):
            out_t.append(np.nan)
            out_y.append(np.nan)
        out_t.append(t[i])
        out_y.append(y[i])
    return np.asarray(out_t, dtype=float), np.asarray(out_y, dtype=float)

def _gap_threshold_s(timestamps_s: np.ndarray, floor_s: float) -> float:
    """Estimate a practical line-break threshold from observed cadence."""
    cadence = _median_positive_step_s(np.asarray(timestamps_s, dtype=float))
    return max(float(floor_s), 8.0 * cadence)

def _ellipsize(label: str, max_len: int = 56) -> str:
    """Shorten long labels for legends."""
    s = str(label)
    if len(s) <= max_len:
        return s
    return f"{s[:max_len-3]}..."

def _apply_smart_legend(ax, handles, labels):
    """Place legend inside for small sets, outside for dense legends."""
    if not labels:
        return False
    labels = [_ellipsize(lab) for lab in labels]
    outside = len(labels) > 10 or max(len(x) for x in labels) > 42
    if outside:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=8,
            frameon=True,
        )
    else:
        ax.legend(handles, labels, loc="best", fontsize=9)
    return outside

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (slope, intercept, r) for y ~ x using scipy if available."""
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan"), float("nan")
    if scipy_stats is not None:
        r = scipy_stats.pearsonr(x, y)[0]
        slope, intercept, *_ = scipy_stats.linregress(x, y)
        return float(slope), float(intercept), float(r)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), 0.0
    r = float(np.corrcoef(x, y)[0,1])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), r

def merge_asof_seconds(df_left: pd.DataFrame, df_right: pd.DataFrame, tolerance_s: int) -> pd.DataFrame:
    """Merge two time series by nearest timestamp within a tolerance (seconds)."""
    left = df_left.copy()
    right = df_right.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], unit="s")
    right["timestamp"] = pd.to_datetime(right["timestamp"], unit="s")
    left = left.sort_values("timestamp")
    right = right.sort_values("timestamp")
    merged = pd.merge_asof(left, right, on="timestamp", direction="nearest",
                           tolerance=pd.Timedelta(seconds=int(tolerance_s)))
    merged["timestamp_s"] = merged["timestamp"].astype("int64") // 10**9
    return merged

def _prepare_power_series(power_df: pd.DataFrame, tmin: float, tmax: float,
                          ema_span: int, resample_seconds: int, timeaware_ema: bool) -> Optional[pd.DataFrame]:
    """Select window, resample, compute EMA; return columns ['timestamp','power_ema'] or None if empty."""
    ema_span = max(1, int(ema_span))
    resample_seconds = max(1, int(resample_seconds))
    # Vectorized numeric path for speed on very large logs.
    ts_all = power_df["Timestamp"].to_numpy(dtype=float)
    w_all = power_df["W_Active"].to_numpy(dtype=float)
    valid = np.isfinite(ts_all) & np.isfinite(w_all)
    if not valid.any():
        return None
    ts = ts_all[valid]
    w = w_all[valid]

    # Time window clip.
    in_win = (ts >= float(tmin)) & (ts <= float(tmax))
    if not in_win.any():
        return None
    ts = ts[in_win]
    w = w[in_win]
    if ts.size == 0:
        return None

    # Ensure sorted timestamps.
    if np.any(np.diff(ts) < 0):
        order = np.argsort(ts)
        ts = ts[order]
        w = w[order]

    # Aggregate onto integer resample buckets without creating full time grids.
    bucket = np.floor(ts / float(resample_seconds)).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(bucket) != 0) + 1]
    sums = np.add.reduceat(w, starts)
    counts = np.diff(np.r_[starts, w.size])
    power = sums / counts
    ts_resampled = bucket[starts].astype(np.float64) * float(resample_seconds)

    # Split segments by large outages and EMA each segment independently.
    gap_split_s = max(float(10 * resample_seconds), 60.0)
    split_idx = np.flatnonzero(np.diff(ts_resampled) > gap_split_s) + 1
    bounds = np.r_[0, split_idx, ts_resampled.size]

    out_ts_parts = []
    out_ema_parts = []
    for i in range(bounds.size - 1):
        a = int(bounds[i])
        b = int(bounds[i + 1])
        if b - a <= 0:
            continue
        seg_ts = ts_resampled[a:b]
        seg_power = power[a:b]
        if timeaware_ema and _is_nearly_regular_cadence(seg_ts):
            # Resampled power is typically near-uniform; fast EMA avoids very
            # long Python loops for million-point runs while preserving shape.
            seg_ema = ema(seg_power, ema_span)
        else:
            seg_ema = ema_adaptive(seg_ts, seg_power, ema_span, timeaware_ema)
        out_ts_parts.append(seg_ts)
        out_ema_parts.append(seg_ema)

    if not out_ts_parts:
        return None
    out_ts = np.concatenate(out_ts_parts)
    out_ema = np.concatenate(out_ema_parts)
    return pd.DataFrame({"timestamp": out_ts.astype(np.int64), "power_ema": out_ema})

def _slice_time_window(df: pd.DataFrame, tmin: float, tmax: float, ts_col: str = "timestamp") -> pd.DataFrame:
    """Fast slice for a dataframe sorted by integer timestamp column."""
    if df is None or df.empty:
        return df
    ts = df[ts_col].to_numpy(dtype=float)
    lo = int(np.searchsorted(ts, tmin, side="left"))
    hi = int(np.searchsorted(ts, tmax, side="right"))
    return df.iloc[lo:hi]

def _estimate_step_s_from_meta(meta: Optional[Dict[str, object]]) -> Optional[float]:
    """Estimate mean sampling step from file metadata if available."""
    if not meta:
        return None
    first = meta.get("first_ts")
    last = meta.get("last_ts")
    count = meta.get("count")
    if not isinstance(first, (int, float)) or not isinstance(last, (int, float)):
        return None
    if not isinstance(count, (int, float)) or count is None:
        return None
    n = int(count)
    if n <= 1:
        return None
    duration = float(last) - float(first)
    if not np.isfinite(duration) or duration <= 0:
        return None
    return duration / float(n - 1)

def recommend_power_ema_span(
    intensity_ema_span: int,
    sw3_median_step_s: float,
    resample_seconds: int,
    power_step_s: Optional[float] = None,
) -> int:
    """Recommend power EMA span from optics EMA and observed cadence.

    The goal is to keep smoothing in comparable *time* units between optical
    and power channels.
    """
    intensity_ema_span = max(1, int(intensity_ema_span))
    sw3_median_step_s = float(sw3_median_step_s) if np.isfinite(sw3_median_step_s) else 1.0
    sw3_median_step_s = max(1.0, sw3_median_step_s)
    resample_seconds = max(1, int(resample_seconds))
    if power_step_s is None or not np.isfinite(power_step_s) or power_step_s <= 0:
        effective_power_step_s = float(resample_seconds)
    else:
        effective_power_step_s = max(float(power_step_s), float(resample_seconds))

    target_smoothing_s = float(intensity_ema_span) * sw3_median_step_s
    span = int(round(target_smoothing_s / effective_power_step_s))
    return int(np.clip(span, 5, 4000))

def choose_effective_resample_seconds(
    duration_s: float,
    requested_seconds: int,
    max_points: int = 600_000,
) -> int:
    """Upscale resample interval for very long windows to bound point count."""
    requested = max(1, int(requested_seconds))
    if not np.isfinite(duration_s) or duration_s <= 0:
        return requested
    max_points = max(50_000, int(max_points))
    auto_seconds = int(np.ceil(float(duration_s) / float(max_points)))
    return max(requested, auto_seconds, 1)

def get_cmap_colors(n: int, cmap_name: str) -> List[Tuple[float, float, float, float]]:
    """Return N colors sampled from a colormap."""
    cmap = _get_cmap(cmap_name)
    xs = np.linspace(0.1, 0.9, max(1, n))
    return [cmap(x) for x in xs]

def get_solarized_colors(n: int) -> List[Tuple[float, float, float, float]]:
    """Return N colors from a Solarized-like palette (cycled)."""
    palette = [
        "#6c71c4",  # violet
        "#268bd2",  # blue
        "#2aa198",  # cyan
        "#859900",  # green
        "#b58900",  # yellow
        "#cb4b16",  # orange
        "#d33682",  # magenta
        "#586e75",  # base01
        "#073642",  # base02
    ]
    return [mpl_colors.to_rgba(palette[i % len(palette)]) for i in range(max(1, n))]

def _to_rgb(c):
    """Convert a color to an RGB numpy array."""
    return np.array(mpl_colors.to_rgb(c), dtype=float)

def lighten(color, amount=0.6):
    """Lighten color by mixing with white; amount in [0..1]."""
    c = _to_rgb(color)
    return tuple(np.clip(1 - (1 - c) * (1 - amount), 0, 1))

def darken(color, amount=0.35):
    """Darken color by scaling toward black; amount in [0..1]."""
    c = _to_rgb(color)
    return tuple(np.clip(c * (1 - (1 - amount)), 0, 1))


# Unit conversion: 1 W/m^2 = 100 µW/cm^2
WM2_TO_UW_CM2 = 100.0

# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

@dataclass
class FileRecord:
    """Project-level metadata for a loaded file (SW3 or Power CSV)."""
    file_id: str
    kind: str   # 'sw3' or 'power'
    path: str
    label: str
    meta: Dict[str, object] = field(default_factory=dict)

@dataclass
class GroupRecord:
    """Grouping of files plus per-group trimming and associations."""
    group_id: str
    name: str
    trim_start_s: int = 0
    trim_end_s: int = 0
    file_ids: Set[str] = field(default_factory=set)
    associations: Dict[str, Set[str]] = field(default_factory=dict)  # sw3_id -> set(power_id)

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

class App(tk.Tk):
    """Main Tkinter application for SW3 + Power Analyzer."""
    def __init__(self):
        """Initialize UI, state, and plotting caches."""
        super().__init__()
        self.title("SW3 + Power Analyzer")
        self._set_initial_geometry()
        self.minsize(920, 620)
        self.option_add("*tearOff", False)
        self._configure_styles()

        self.files: Dict[str, FileRecord] = {}
        self.groups: Dict[str, GroupRecord] = {}

        # display maps for friendly names
        self._sw3_display_to_id: Dict[str, str] = {}
        self._group_display_to_id: Dict[str, str] = {}

        self.power_column_map = {
            "timestamp": ["Timestamp", "timestamp", "epoch", "Epoch", "time", "Time"],
            "watts": ["W_Active", "watts", "Watts", "Power_W", "W"]
        }

        self._init_vars()
        self._build_ui()

        # figure registry and aligned cache
        self._figs: Dict[str, List[plt.Figure]] = {"ivt": [], "ovp": [], "gdo": [], "corr": []}
        self._aligned_cache: Optional[pd.DataFrame] = None
        self._aligned_cache_gid: Optional[str] = None
        self._aligned_cache_signature: Optional[Tuple] = None
        self._last_analyzed_gid: Optional[str] = None
        self._power_csv_cache: Dict[Tuple, Tuple[pd.DataFrame, Dict[str, object]]] = {}
        self._analysis_job_id: int = 0
        self._analysis_running_gid: Optional[str] = None
        self._analysis_running_signature: Optional[Tuple] = None
        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_queue: Queue = Queue()
        self._pending_reprocess_gid: Optional[str] = None
        self._ovp_series_cache: Optional[Dict[str, object]] = None
        self._sw3_plot_cache: Dict[Tuple[str, bool, bool], List[Tuple[float, float]]] = {}
        self._sw3_preprocess_job_id: int = 0
        self._sw3_preprocess_thread: Optional[threading.Thread] = None
        self._sw3_preprocess_running_signature: Optional[Tuple] = None
        self._sw3_preprocess_queue: Queue = Queue()
        self._reload_job_id: int = 0
        self._reload_thread: Optional[threading.Thread] = None
        self._reload_queue: Queue = Queue()

    def _set_initial_geometry(self):
        """Fit startup window to screen so bottom controls/status remain visible."""
        self.update_idletasks()
        screen_w = int(self.winfo_screenwidth() or 1440)
        screen_h = int(self.winfo_screenheight() or 900)
        width = max(1024, min(1320, screen_w - 120))
        height = max(680, min(880, screen_h - 140))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2 - 10)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _configure_styles(self):
        """Apply a cohesive ttk style pass for a cleaner, denser UI."""
        style = ttk.Style(self)
        # Keep the OS-native ttk theme so controls match platform styling.
        style.configure(".", font=("TkDefaultFont", 10))
        style.configure("Treeview", rowheight=22)
        style.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"))
        style.configure("TNotebook.Tab", padding=(12, 5))
        style.configure("Section.TLabelframe", padding=(7, 5))
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        style.configure("Toolbar.TButton", padding=(8, 4))
        style.configure("Primary.TButton", padding=(10, 5))
        style.configure("Status.TLabel", padding=(8, 4))
        style.configure("Hint.TLabel", foreground="#586e75")

    # ---- control vars ----
    def _init_vars(self):
        """Initialize Tkinter variables for controls and state."""
        # analysis params
        # Tuned on Open Excimer long-run dataset (see docs/open_excimer_recommendations.json)
        self.intensity_ema_span = tk.IntVar(self, value=15)
        self.power_ema_span     = tk.IntVar(self, value=86)
        self.overlay_ema_span   = tk.IntVar(self, value=15)
        self.trim_start_s       = tk.IntVar(self, value=0)   # per-IVT extra trim
        self.trim_end_s         = tk.IntVar(self, value=0)
        self.align_tolerance_s  = tk.IntVar(self, value=6)
        self.ccf_max_lag_s      = tk.IntVar(self, value=180)
        self.resample_seconds   = tk.IntVar(self, value=3)
        self.time_weighted_ema  = tk.BooleanVar(self, value=True)
        self.auto_power_ema     = tk.BooleanVar(self, value=True)

        # filters
        self.normalize_to_1m    = tk.BooleanVar(self, value=True)
        self.only_yaw_roll_zero = tk.BooleanVar(self, value=True)

        # per-analysis style
        self.ivt_show_points    = tk.BooleanVar(self, value=True)
        self.ivt_show_ema       = tk.BooleanVar(self, value=True)
        self.ivt_point_alpha    = tk.DoubleVar(self, value=0.25)
        self.ivt_line_alpha     = tk.DoubleVar(self, value=1.0)

        self.ovp_show_points    = tk.BooleanVar(self, value=False)
        self.ovp_show_int_ema   = tk.BooleanVar(self, value=True)
        self.ovp_show_pow_ema   = tk.BooleanVar(self, value=True)
        self.ovp_point_alpha    = tk.DoubleVar(self, value=0.15)
        self.ovp_line_alpha     = tk.DoubleVar(self, value=1.0)

        self.gdo_show_points    = tk.BooleanVar(self, value=False)
        self.gdo_show_ema       = tk.BooleanVar(self, value=True)
        self.gdo_point_alpha    = tk.DoubleVar(self, value=0.15)
        self.gdo_line_alpha     = tk.DoubleVar(self, value=1.0)

        # combos (friendly)
        self.ivt_sw3_display    = tk.StringVar(self, value="")
        self.ivt_group_display  = tk.StringVar(self, value="")
        self.ovp_group_display  = tk.StringVar(self, value="")

        # source mode
        self.source_mode        = tk.StringVar(self, value="file")  # 'file' or 'group'
        self.combine_group_sw3  = tk.BooleanVar(self, value=True)

        # UI
        self.controls_visible   = tk.BooleanVar(self, value=True)

    # ---- UI ----
    def _build_ui(self):
        """Construct the main window layout."""
        self._build_menu()

        # Single vertical panedwindow with three panes
        self._paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        self._paned.pack(fill=tk.BOTH, expand=True)

        files_container = ttk.Frame(self._paned)
        groups_container = ttk.Frame(self._paned)
        self._controls_frame = ttk.Frame(self._paned)

        self._paned.add(files_container, weight=3)
        self._paned.add(groups_container, weight=2)
        self._paned.add(self._controls_frame, weight=1)

        self._build_files_frame(files_container)
        self._build_groups_frame(groups_container)
        self._build_controls(self._controls_frame)

        # status / busy indicator (explicit bottom strip so it stays visible)
        status_shell = tk.Frame(self, bg="#fdf6e3", bd=1, relief=tk.SUNKEN, height=30)
        status_shell.pack(fill=tk.X, side=tk.BOTTOM)
        status_shell.pack_propagate(False)
        status_bar = tk.Frame(status_shell, bg="#fdf6e3")
        status_bar.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

        self.status_var = tk.StringVar(self, value="Ready.")
        self._status_title = tk.Label(
            status_bar,
            text="Status:",
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
            bg="#fdf6e3",
            fg="#586e75",
        )
        self._status_title.pack(side=tk.LEFT, padx=(0, 6))
        self._status_label = tk.Label(
            status_bar,
            textvariable=self.status_var,
            anchor="w",
            bg="#fdf6e3",
            fg="#586e75",
        )
        self._status_label.pack(fill=tk.X, side=tk.LEFT, expand=True)

        busy_shell = tk.Frame(status_bar, bg="#fdf6e3")
        busy_shell.pack(side=tk.RIGHT)
        self._busy_var = tk.StringVar(self, value="Background: idle")
        self._busy_label = tk.Label(
            busy_shell,
            textvariable=self._busy_var,
            anchor="e",
            bg="#fdf6e3",
            fg="#657b83",
        )
        self._busy_label.pack(side=tk.RIGHT, padx=(0, 8))
        self._busy_pb = ttk.Progressbar(busy_shell, orient=tk.HORIZONTAL, mode="indeterminate", length=130)
        self._busy_pb.stop()
        self._busy_pb.pack_forget()
        self._busy_active = False

        # initial vertical sash positions (favor controls) with minimum Files height
        self.after(120, self._set_initial_sashes)

    def _build_menu(self):
        """Create the menu bar and bind menu actions."""
        menubar = tk.Menu(self)

        # File
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Add SW3…", command=self.on_add_sw3_files)
        file_menu.add_command(label="Add Power CSV…", command=self.on_add_power_files)
        file_menu.add_separator()
        file_menu.add_command(label="Reload Files", command=self.on_reload_all_files)
        file_menu.add_separator()
        file_menu.add_command(label="Save Session…", command=self.on_save_session)
        file_menu.add_command(label="Load Session…", command=self.on_load_session)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        # Settings
        settings_menu = tk.Menu(menubar, tearoff=False)
        settings_menu.add_command(label="Power Columns…", command=self.on_set_power_columns)
        settings_menu.add_separator()
        settings_menu.add_command(label="Add Module Path…", command=self.on_add_module_path)
        settings_menu.add_command(label="List Module Paths…", command=self.on_list_module_paths)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # View
        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_checkbutton(label="Show Controls Panel", variable=self.controls_visible, command=self.on_toggle_controls)
        view_menu.add_command(label="Compact Controls Height", command=self.on_compact_controls_height)
        view_menu.add_command(label="Favor Controls Section", command=self.on_favor_controls)
        view_menu.add_separator()
        view_menu.add_command(label="Maximize Files Section", command=self.on_maximize_files)
        view_menu.add_command(label="Maximize Groups Section", command=self.on_maximize_groups)
        view_menu.add_command(label="Maximize Controls Section", command=self.on_maximize_controls)
        menubar.add_cascade(label="View", menu=view_menu)

        # Help
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", "SW3 + Power Analyzer\nAligns SW3/eegbin optical data with power logs by epoch time."
        ))
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_files_frame(self, parent):
        """Build the Files pane (file list and actions)."""
        frm = ttk.LabelFrame(parent, text="Files")
        frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        toolbar = ttk.Frame(frm)
        toolbar.pack(fill=tk.X, padx=4, pady=(6, 4))
        ttk.Label(toolbar, text="Actions:", style="Hint.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Add SW3", style="Toolbar.TButton", command=self.on_add_sw3_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add Power CSV", style="Toolbar.TButton", command=self.on_add_power_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Assign Group", style="Toolbar.TButton", command=self.on_assign_files_to_group).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(toolbar, text="Rename", style="Toolbar.TButton", command=self.on_rename_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Remove", style="Toolbar.TButton", command=self.on_remove_files).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(
            toolbar,
            text="Reload Selected",
            style="Toolbar.TButton",
            command=lambda: self.on_reload_all_files(only_selected=True),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Reload All", style="Toolbar.TButton", command=self.on_reload_all_files).pack(side=tk.LEFT, padx=2)

        # --- Treeview container BELOW the button bar ---
        container = ttk.Frame(frm)
        container.pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 6))

        cols = ("label", "kind", "first", "last", "count", "path")
        self.tv_files = ttk.Treeview(container, columns=cols, show="headings", selectmode="extended")
        heading = {
            "label": "Label",
            "kind": "Kind",
            "first": "First Seen",
            "last": "Last Seen",
            "count": "Rows",
            "path": "Path",
        }
        for c in cols:
            self.tv_files.heading(c, text=heading[c])
            if c == "label":
                w = 220
            elif c == "path":
                w = 360
            elif c == "kind":
                w = 80
            elif c == "count":
                w = 90
            else:
                w = 170
            self.tv_files.column(c, width=w, anchor="w", stretch=True)
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tv_files.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.tv_files.xview)
        self.tv_files.configure(yscroll=vsb.set, xscroll=hsb.set)
        self.tv_files.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1); container.columnconfigure(0, weight=1)

        self.tv_files.bind("<Double-1>", lambda e: self.on_rename_file())

    def _build_groups_frame(self, parent):
        """Build the Groups pane (groups list and associations)."""
        frm = ttk.LabelFrame(parent, text="Groups & Associations")
        frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        top = ttk.Frame(frm)
        top.pack(fill=tk.X, padx=4, pady=(6, 4))
        ttk.Label(top, text="Actions:", style="Hint.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top, text="New Group", style="Toolbar.TButton", command=self.on_new_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Rename", style="Toolbar.TButton", command=self.on_rename_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Delete", style="Toolbar.TButton", command=self.on_delete_group).pack(side=tk.LEFT, padx=2)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(top, text="Associations", style="Toolbar.TButton", command=self.on_edit_associations).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Set Trim", style="Toolbar.TButton", command=self.on_set_group_trim).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Clear Trim", style="Toolbar.TButton", command=self.on_clear_group_trim).pack(side=tk.LEFT, padx=2)

        container = ttk.Frame(frm)
        container.pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 6))
        cols = ("name", "trim", "sw3_count", "power_count")
        self.tv_groups = ttk.Treeview(container, columns=cols, show="headings", selectmode="browse")
        heading = {"name": "Group", "trim": "Trim", "sw3_count": "SW3", "power_count": "Power"}
        for c in cols:
            self.tv_groups.heading(c, text=heading[c])
            if c == "name":
                w = 260
            elif c == "trim":
                w = 220
            else:
                w = 90
            self.tv_groups.column(c, width=w, anchor="w", stretch=True)
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tv_groups.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.tv_groups.xview)
        self.tv_groups.configure(yscroll=vsb.set, xscroll=hsb.set)
        self.tv_groups.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1); container.columnconfigure(0, weight=1)
        self.tv_groups.bind("<<TreeviewSelect>>", lambda e: self._on_group_selection_changed())

    def _build_controls(self, parent):
        """Build the Controls pane (analysis controls and plots)."""
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        def make_section(tab, title: str):
            sec = ttk.LabelFrame(tab, text=title, style="Section.TLabelframe")
            sec.pack(fill=tk.X, pady=(0, 5))
            return sec

        def make_row(sec, pady=(0, 2)):
            row = ttk.Frame(sec)
            row.pack(fill=tk.X, padx=4, pady=pady)
            return row

        # ---- Intensity vs Time ----
        tab_ivt = ttk.Frame(nb, padding=(8, 7))
        nb.add(tab_ivt, text="Intensity vs Time")

        ivt_src = make_section(tab_ivt, "Source")
        r0 = make_row(ivt_src)
        ttk.Label(r0, text="Mode").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Radiobutton(r0, text="Single file", value="file", variable=self.source_mode).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(r0, text="Group", value="group", variable=self.source_mode).pack(side=tk.LEFT)
        r0b = make_row(ivt_src)
        ttk.Label(r0b, text="SW3").pack(side=tk.LEFT, padx=(0, 4))
        self.cb_ivt_sw3 = ttk.Combobox(r0b, state="readonly", width=34, textvariable=self.ivt_sw3_display)
        self.cb_ivt_sw3.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(r0b, text="Group").pack(side=tk.LEFT, padx=(0, 4))
        self.cb_ivt_group = ttk.Combobox(r0b, state="readonly", width=30, textvariable=self.ivt_group_display)
        self.cb_ivt_group.pack(side=tk.LEFT)
        r0c = make_row(ivt_src, pady=(0, 0))
        ttk.Checkbutton(r0c, text="Combine all SW3 files in selected group", variable=self.combine_group_sw3).pack(
            side=tk.LEFT
        )

        ivt_filter = make_section(tab_ivt, "Filtering & Window")
        r1 = make_row(ivt_filter)
        ttk.Label(r1, text="Optics EMA span").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(r1, textvariable=self.intensity_ema_span, width=7).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(r1, text="Time-aware EMA", variable=self.time_weighted_ema).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(r1, text="Normalize to 1 m", variable=self.normalize_to_1m).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(r1, text="Yaw=0, Roll=0 only", variable=self.only_yaw_roll_zero).pack(side=tk.LEFT)
        r1b = make_row(ivt_filter, pady=(0, 0))
        ttk.Label(r1b, text="Trim start").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(r1b, textvariable=self.trim_start_s, width=9).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(r1b, text="s").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(r1b, text="Trim end").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(r1b, textvariable=self.trim_end_s, width=9).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(r1b, text="s").pack(side=tk.LEFT)

        ivt_display = make_section(tab_ivt, "Display")
        r2 = make_row(ivt_display)
        ttk.Checkbutton(r2, text="Show points", variable=self.ivt_show_points).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(r2, text="Show EMA", variable=self.ivt_show_ema).pack(side=tk.LEFT)
        r2b = make_row(ivt_display, pady=(0, 0))
        ttk.Label(r2b, text="Points alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(r2b, variable=self.ivt_point_alpha, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Label(r2b, text="EMA alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(r2b, variable=self.ivt_line_alpha, from_=0.1, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT
        )

        ivt_actions = ttk.Frame(tab_ivt)
        ivt_actions.pack(fill=tk.X)
        ttk.Button(ivt_actions, text="Plot", style="Primary.TButton", command=self.on_plot_intensity_vs_time).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(ivt_actions, text="Save Figure...", style="Toolbar.TButton", command=self.on_save_last_figure).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(
            ivt_actions,
            text="Close IVT/Spectrum Plots",
            style="Toolbar.TButton",
            command=lambda: self._close_plots('ivt', also=('spec',))
        ).pack(side=tk.LEFT, padx=10)
        ttk.Label(
            tab_ivt,
            text="Trim accepts seconds (90) or HH:MM:SS (00:01:30).",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        # ---- Optics vs Power ----
        tab_ovp = ttk.Frame(nb, padding=(8, 7))
        nb.add(tab_ovp, text="Optics vs Power")
        ovp_cfg = make_section(tab_ovp, "Group & Alignment")
        o1 = make_row(ovp_cfg)
        ttk.Label(o1, text="Group").pack(side=tk.LEFT, padx=(0, 4))
        self.cb_group = ttk.Combobox(o1, state="readonly", width=42, textvariable=self.ovp_group_display)
        self.cb_group.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(o1, text="Optics EMA comes from Intensity tab", style="Hint.TLabel").pack(side=tk.LEFT)
        o1b = make_row(ovp_cfg, pady=(0, 0))
        ttk.Label(o1b, text="Power EMA").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(o1b, textvariable=self.power_ema_span, width=7).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(o1b, text="Auto from optics + cadence", variable=self.auto_power_ema).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(o1b, text="Resample").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(o1b, textvariable=self.resample_seconds, width=7).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(o1b, text="s").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(o1b, text="Align tol").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(o1b, textvariable=self.align_tolerance_s, width=7).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(o1b, text="s").pack(side=tk.LEFT)

        ovp_display = make_section(tab_ovp, "Display")
        o2 = make_row(ovp_display)
        ttk.Checkbutton(o2, text="Show points", variable=self.ovp_show_points).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(o2, text="Show optics EMA", variable=self.ovp_show_int_ema).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(o2, text="Show power EMA", variable=self.ovp_show_pow_ema).pack(side=tk.LEFT)
        o2b = make_row(ovp_display, pady=(0, 0))
        ttk.Label(o2b, text="Points alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(o2b, variable=self.ovp_point_alpha, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Label(o2b, text="Lines alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(o2b, variable=self.ovp_line_alpha, from_=0.1, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT
        )

        ovp_actions = ttk.Frame(tab_ovp)
        ovp_actions.pack(fill=tk.X)
        ttk.Button(ovp_actions, text="Analyze Group", style="Primary.TButton", command=self.on_analyze_group).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(
            ovp_actions,
            text="Correlation & Scatter",
            style="Toolbar.TButton",
            command=self.on_corr_and_scatter,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(ovp_actions, text="Export Aligned CSV...", style="Toolbar.TButton", command=self.on_export_aligned_csv).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(ovp_actions, text="Save Figure...", style="Toolbar.TButton", command=self.on_save_last_figure).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(
            ovp_actions,
            text="Close OVP Plots",
            style="Toolbar.TButton",
            command=lambda: self._close_plots('ovp', also=('corr',))
        ).pack(side=tk.LEFT, padx=10)

        # ---- Group Decay Overlay ----
        tab_gdo = ttk.Frame(nb, padding=(8, 7))
        nb.add(tab_gdo, text="Group Decay Overlay")
        gdo_display = make_section(tab_gdo, "Display")
        g1 = make_row(gdo_display)
        ttk.Label(g1, text="Optics EMA comes from Intensity tab", style="Hint.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(g1, text="Show points", variable=self.gdo_show_points).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(g1, text="Show EMA", variable=self.gdo_show_ema).pack(side=tk.LEFT)
        g1b = make_row(gdo_display, pady=(0, 0))
        ttk.Label(g1b, text="Points alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(g1b, variable=self.gdo_point_alpha, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Label(g1b, text="Lines alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(g1b, variable=self.gdo_line_alpha, from_=0.1, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT
        )

        gdo_actions = ttk.Frame(tab_gdo)
        gdo_actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            gdo_actions,
            text="Plot Selected Groups",
            style="Primary.TButton",
            command=lambda: self.on_plot_group_decay(selected_only=True),
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            gdo_actions,
            text="Plot All Groups",
            style="Toolbar.TButton",
            command=lambda: self.on_plot_group_decay(selected_only=False),
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(gdo_actions, text="Close GDO Plots", style="Toolbar.TButton", command=lambda: self._close_plots('gdo')).pack(side=tk.LEFT, padx=10)

        ttk.Label(tab_gdo, text="Pick groups below (default: all groups).", style="Hint.TLabel").pack(anchor="w", pady=(0, 4))
        frame_list = ttk.LabelFrame(tab_gdo, text="Group Selection", style="Section.TLabelframe")
        frame_list.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.lb_groups_select = tk.Listbox(frame_list, selectmode=tk.EXTENDED, exportselection=False)
        vs = ttk.Scrollbar(frame_list, orient="vertical", command=self.lb_groups_select.yview)
        self.lb_groups_select.configure(yscroll=vs.set)
        self.lb_groups_select.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        frame_list.rowconfigure(0, weight=1)
        frame_list.columnconfigure(0, weight=1)
        bsel = ttk.Frame(tab_gdo)
        bsel.pack(fill=tk.X, pady=2)
        ttk.Button(bsel, text="Select All", style="Toolbar.TButton", command=self._select_all_groups).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            bsel,
            text="Select None",
            style="Toolbar.TButton",
            command=lambda: self.lb_groups_select.selection_clear(0, tk.END),
        ).pack(side=tk.LEFT, padx=3)

    # ---- View helpers ----
    def _set_initial_sashes(self):
        """Favor Controls, but guarantee Files has enough height to show its button bar."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height() or self.winfo_height() or 800
            MIN_FILES = 180
            MIN_GROUPS = 140
            MIN_CONTROLS = 260
            # initial proportional split
            pos0 = int(total_h * 0.18)
            pos1 = int(total_h * 0.42)
            # enforce minimum heights
            pos0 = max(MIN_FILES, pos0)
            pos1 = max(pos0 + MIN_GROUPS, pos1)
            if (total_h - pos1) < MIN_CONTROLS:
                pos1 = max(pos0 + MIN_GROUPS, total_h - MIN_CONTROLS)
            pos1 = min(pos1, total_h - 40)
            self._paned.sashpos(0, pos0)
            self._paned.sashpos(1, pos1)
        except Exception:
            pass

    def on_toggle_controls(self):
        """Show or hide the Controls pane based on the checkbox."""
        vis = self.controls_visible.get()
        try:
            panes = self._paned.panes()
            if vis and str(self._controls_frame) not in panes:
                self._paned.add(self._controls_frame, weight=1)
            elif not vis and str(self._controls_frame) in panes:
                self._paned.forget(self._controls_frame)
        except Exception:
            pass

    def on_compact_controls_height(self):
        """Shrink the controls section to a compact but usable height (~280 px)."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            # Keep controls usable while still compact.
            self._paned.sashpos(0, 150)
            self._paned.sashpos(1, max(300, total_h - 280))
        except Exception:
            pass

    def on_favor_controls(self):
        """Give more height to the Controls section (Files ~25%, Groups ~30%, Controls ~45%)."""
        try:
            self._set_initial_sashes()
        except Exception:
            pass

    def on_maximize_files(self):
        """Resize the panes to maximize the Files section."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            self._paned.sashpos(0, total_h - 40)
            self._paned.sashpos(1, total_h - 20)
        except Exception:
            pass

    def on_maximize_groups(self):
        """Resize the panes to maximize the Groups section."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            self._paned.sashpos(0, 120)
            self._paned.sashpos(1, total_h - 40)
        except Exception:
            pass

    def on_maximize_controls(self):
        """Resize the panes to maximize the Controls section."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            self._paned.sashpos(0, 120)
            self._paned.sashpos(1, total_h - 10)
        except Exception:
            pass

    # ---- File ops ----
    def on_add_sw3_files(self):
        """Prompt for SW3/eegbin files and add them to the project."""
        paths = filedialog.askopenfilenames(title="Add SW3/eegbin Files",
                                            filetypes=[("SW3/eegbin", "*.sw3 *.eegbin *.bin *.*")])
        if not paths: return
        ensure_eegbin_imported()
        n=0
        for path in paths:
            sweep = self._load_sweep_from_path(path, warn=True, trace=True)
            if sweep is None:
                continue
            meta = self._meta_from_sweep(sweep)
            fid = self._next_file_id()
            self.files[fid] = FileRecord(fid, "sw3", path, os.path.basename(path), meta)
            n += 1
        if n:
            self._refresh_files_tv()
            self._refresh_display_mappings()
            self._invalidate_aligned_cache()
            self._set_status(f"Added {n} SW3 file(s).")

    def on_add_power_files(self):
        """Prompt for power CSV files and add them to the project."""
        paths = filedialog.askopenfilenames(title="Add Power CSV Files",
                                            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not paths: return
        n=0
        for path in paths:
            try:
                _, meta = self._load_power_csv_with_meta(path)
                fid = self._next_file_id()
                self.files[fid] = FileRecord(fid, "power", path, os.path.basename(path), meta)
                n+=1
            except Exception as e:
                traceback.print_exc(); messagebox.showwarning("Load error", f"Failed to load {path}\n{e}")
        if n:
            self._refresh_files_tv()
            self._refresh_display_mappings()
            self._invalidate_aligned_cache()
            self._set_status(f"Added {n} power file(s).")

    def _meta_from_sweep(self, sweep) -> Dict[str, object]:
        """Extract summary metadata (timestamps/count) from a sweep."""
        rows = [r for r in sweep.rows if getattr(r, "timestamp", None) is not None]
        ts = [r.timestamp for r in rows]
        first = float(min(ts)) if ts else None
        last = float(max(ts)) if ts else None
        return {"first_ts": first, "last_ts": last, "count": len(rows), "lamp_name": getattr(sweep, "lamp_name", "")}

    def _power_map_signature(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Return a stable signature for current power column candidate settings."""
        ts = tuple(str(x).strip().lower() for x in self.power_column_map.get("timestamp", []))
        watts = tuple(str(x).strip().lower() for x in self.power_column_map.get("watts", []))
        return ts, watts

    def _power_file_signature(self, path: str) -> Tuple[int, int]:
        """Return (mtime_ns, size_bytes) used to invalidate power CSV cache entries."""
        st = os.stat(path)
        return int(st.st_mtime_ns), int(st.st_size)

    def _load_power_csv_with_meta(self, path: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """Load a power CSV file and return (dataframe, metadata)."""
        abs_path = os.path.abspath(path)
        map_sig = self._power_map_signature()
        file_sig = self._power_file_signature(abs_path)
        cache_key = (abs_path, file_sig, map_sig)
        cached = self._power_csv_cache.get(cache_key)
        if cached is not None:
            return cached[0], dict(cached[1])

        header = pd.read_csv(abs_path, nrows=0)
        ts_col = self._detect_col(header.columns, self.power_column_map["timestamp"])
        w_col  = self._detect_col(header.columns, self.power_column_map["watts"])
        if not ts_col or not w_col:
            raise ValueError(
                f"CSV must include timestamp + watts. Candidates tried: {self.power_column_map}. "
                f"Found: {list(header.columns)}"
            )

        df = pd.read_csv(
            abs_path,
            usecols=[ts_col, w_col],
            low_memory=False,
        ).rename(columns={ts_col: "Timestamp", w_col: "W_Active"})
        df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
        df["W_Active"]  = pd.to_numeric(df["W_Active"], errors="coerce")
        df = df.dropna(subset=["Timestamp","W_Active"])
        if not df.empty and np.any(np.diff(df["Timestamp"].to_numpy(dtype=float)) < 0):
            df = df.sort_values("Timestamp", kind="mergesort")
        df = df.reset_index(drop=True)
        meta = {"first_ts": float(df["Timestamp"].min()) if not df.empty else None,
                "last_ts":  float(df["Timestamp"].max()) if not df.empty else None,
                "count": int(len(df))}
        result = (df, meta)
        # Simple cap keeps memory bounded on repeated analysis runs.
        if len(self._power_csv_cache) >= 24:
            self._power_csv_cache.clear()
        self._power_csv_cache[cache_key] = result
        return result[0], dict(result[1])

    def _detect_col(self, cols, candidates: List[str]) -> Optional[str]:
        """Pick the first column matching a list of candidate names."""
        m = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in m:
                return m[cand.lower()]
        return None

    def _refresh_files_tv(self):
        """Refresh the Files treeview with current metadata."""
        self.tv_files.delete(*self.tv_files.get_children())
        for fid, rec in self.files.items():
            first = human_datetime(rec.meta.get("first_ts")) if rec.meta.get("first_ts") else ""
            last  = human_datetime(rec.meta.get("last_ts")) if rec.meta.get("last_ts") else ""
            count = rec.meta.get("count","")
            self.tv_files.insert("", "end", iid=fid, values=(rec.label, rec.kind, first, last, count, rec.path))

    def _refresh_groups_tv(self):
        """Refresh the Groups treeview and selection list."""
        self.tv_groups.delete(*self.tv_groups.get_children())
        for gid, g in self.groups.items():
            sw3c = sum(1 for fid in g.file_ids if self.files.get(fid, FileRecord("", "", "", "")).kind == "sw3")
            powc = sum(1 for fid in g.file_ids if self.files.get(fid, FileRecord("", "", "", "")).kind == "power")
            self.tv_groups.insert("", "end", iid=gid, values=(g.name, self._format_group_trim(g), sw3c, powc))
        self._refresh_group_select_list()
        self._refresh_display_mappings()

    def _refresh_group_select_list(self):
        """Refresh the group listbox used by decay overlay selection."""
        self.lb_groups_select.delete(0, tk.END)
        for gid, g in self.groups.items():
            self.lb_groups_select.insert(tk.END, f"{gid}  {g.name}")

    def _refresh_display_mappings(self):
        """Refresh friendly display labels for files and groups."""
        # files (sw3)
        sw3_ids = [fid for fid, fr in self.files.items() if fr.kind=="sw3"]
        labels = [self.files[fid].label for fid in sw3_ids]
        counts = {}
        for lab in labels: counts[lab] = counts.get(lab,0)+1
        self._sw3_display_to_id.clear()
        sw3_display = []
        for fid in sw3_ids:
            lab = self.files[fid].label
            disp = lab if counts[lab]==1 else f"{lab} [{fid}]"
            self._sw3_display_to_id[disp] = fid
            sw3_display.append(disp)
        self.cb_ivt_sw3["values"] = sw3_display
        # groups
        self._group_display_to_id.clear()
        group_display = []
        name_counts = {}
        for g in self.groups.values():
            name_counts[g.name] = name_counts.get(g.name,0)+1
        for gid, g in self.groups.items():
            disp = g.name if name_counts[g.name]==1 else f"{g.name} [{gid}]"
            self._group_display_to_id[disp] = gid
            group_display.append(disp)
        self.cb_ivt_group["values"] = group_display
        self.cb_group["values"] = group_display

    def _display_to_sw3_id(self, disp: str) -> Optional[str]:
        """Resolve a displayed SW3 label back to its file id."""
        return self._sw3_display_to_id.get(disp)

    def _display_to_group_id(self, disp: str) -> Optional[str]:
        """Resolve a displayed Group label back to its group id."""
        return self._group_display_to_id.get(disp)

    def _load_sweep_from_path(self, path: str, warn: bool = True, trace: bool = False):
        """Load an SW3/eegbin file and return the sweep object or None on failure."""
        try:
            with open(path, "rb") as fd:
                buf = fd.read()
            ensure_eegbin_imported()
            try:
                return eegbin.load_eegbin3(buf, from_path=path)
            except Exception:
                return eegbin.load_eegbin2(buf, from_path=path)
        except Exception as e:
            if trace:
                traceback.print_exc()
            if warn:
                messagebox.showwarning("Load error", f"Failed to load {path}\n{e}")
            return None

    def _row_intensity_uW_cm2(self, row, normalize_to_1m: bool) -> float:
        """Compute intensity in µW/cm² from a row, optionally normalized to 1 m."""
        I_wm2 = float(getattr(row.capture, "integral_result", 0.0) or 0.0)
        if normalize_to_1m:
            dist_m = float(getattr(row.coords, "lin_mm", 0.0) or 0.0) / 1000.0
            if dist_m > 0:
                I_wm2 = I_wm2 * (dist_m ** 2)
        return I_wm2 * WM2_TO_UW_CM2

    def _extract_intensity_rows(
        self,
        sweep,
        only_zero: bool,
        normalize_to_1m: bool,
        include_row: bool = False,
    ):
        """Return sorted (timestamp, intensity[, row, sweep]) rows from a sweep."""
        out = []
        for r in getattr(sweep, "rows", []):
            if hasattr(r, "valid") and not r.valid:
                continue
            if only_zero and not (
                getattr(r.coords, "yaw_deg", None) == 0 and getattr(r.coords, "roll_deg", None) == 0
            ):
                continue
            ts = getattr(r, "timestamp", None)
            if ts is None:
                continue
            intensity = self._row_intensity_uW_cm2(r, normalize_to_1m)
            if include_row:
                out.append((float(ts), intensity, r, sweep))
            else:
                out.append((float(ts), intensity))
        out.sort(key=lambda x: x[0])
        return out

    # ---- File actions ----
    def on_rename_file(self):
        """Rename the selected file's display label."""
        sel = self.tv_files.selection()
        if not sel: return
        if len(sel) > 1:
            messagebox.showinfo("Rename", "Select a single file to rename."); return
        fid = sel[0]
        rec = self.files[fid]
        new = simpledialog.askstring("Rename", "New label:", initialvalue=rec.label, parent=self)
        if new:
            rec.label = new.strip()
            self._refresh_files_tv()
            self._refresh_display_mappings()
            self._invalidate_aligned_cache()

    def on_remove_files(self):
        """Remove selected files from the project and all group mappings."""
        sel = list(self.tv_files.selection())
        if not sel: return
        if not messagebox.askyesno("Remove", f"Remove {len(sel)} file(s) from the project?"):
            return
        for fid in sel:
            for g in self.groups.values():
                g.file_ids.discard(fid)
                g.associations.pop(fid, None)
                for k in list(g.associations.keys()):
                    g.associations[k].discard(fid)
                    if not g.associations[k]:
                        g.associations.pop(k, None)
            self.files.pop(fid, None)
        self._refresh_files_tv()
        self._refresh_groups_tv()
        self._invalidate_aligned_cache()

    def on_assign_files_to_group(self):
        """Assign selected files to a chosen group."""
        sel = list(self.tv_files.selection())
        if not sel:
            messagebox.showinfo("Assign", "Select files (SW3 and/or Power) to assign to a group."); return
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        for fid in sel: g.file_ids.add(fid)
        self._refresh_groups_tv()
        self._invalidate_aligned_cache()
        self._set_status(f"Assigned {len(sel)} file(s) to group '{g.name}'.")

    # ---- Groups ----
    def on_new_group(self):
        """Create a new group."""
        name = simpledialog.askstring("New Group", "Group name:", parent=self)
        if not name: return
        gid = self._next_group_id()
        self.groups[gid] = GroupRecord(gid, name.strip())
        self._refresh_groups_tv()
        self._set_status(f"Created group '{name}'.")

    def on_rename_group(self):
        """Rename the selected group."""
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        new = simpledialog.askstring("Rename Group", "New name:", initialvalue=g.name, parent=self)
        if new:
            g.name = new.strip()
            self._refresh_groups_tv()
            self._invalidate_aligned_cache()

    def on_delete_group(self):
        """Delete the selected group (does not delete files)."""
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        if not messagebox.askyesno("Delete Group", "Delete this group (files remain in project)?"):
            return
        del self.groups[gid]
        self._refresh_groups_tv()
        self._invalidate_aligned_cache()

    def on_edit_associations(self):
        """Open the associations dialog for the selected group."""
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        sw3s = [(fid, self.files[fid]) for fid in g.file_ids if self.files[fid].kind=="sw3"]
        pows = [(fid, self.files[fid]) for fid in g.file_ids if self.files[fid].kind=="power"]
        if not sw3s or not pows:
            messagebox.showinfo("Associations", "Assign at least one SW3 and one Power file to the group first."); return
        dlg = AssociationDialog(self, g, sw3s, pows); self.wait_window(dlg)
        if dlg.updated:
            self._invalidate_aligned_cache()
            self._set_status("Updated associations.")

    def _format_group_trim(self, g) -> str:
        """Return a user-friendly trim string for a group."""
        s = fmt_hhmmss(max(0, int(getattr(g, "trim_start_s", 0))))
        e = fmt_hhmmss(max(0, int(getattr(g, "trim_end_s", 0))))
        return "—" if (s=="00:00:00" and e=="00:00:00") else f"{s} | {e}"

    def _group_time_window(
        self,
        g: GroupRecord,
        files_map: Optional[Dict[str, FileRecord]] = None,
    ) -> Optional[Tuple[float, float]]:
        """Compute trimmed time window [tmin,tmax] from group's SW3 files and trims.
           Returns None if window degenerates or no SW3 meta is available.
        """
        file_lookup = self.files if files_map is None else files_map
        sw3_times = []
        for fid in g.file_ids:
            fr = file_lookup.get(fid)
            if fr and fr.kind == "sw3":
                ft = fr.meta.get("first_ts"); lt = fr.meta.get("last_ts")
                if isinstance(ft, (int, float)) and isinstance(lt, (int, float)):
                    sw3_times.append((float(ft), float(lt)))
        if not sw3_times:
            return None
        base_start = min(a for a, _ in sw3_times)
        base_end   = max(b for _, b in sw3_times)
        tmin = base_start + max(0, int(getattr(g, "trim_start_s", 0)))
        tmax = base_end   - max(0, int(getattr(g, "trim_end_s", 0)))
        if tmax <= tmin:
            return None
        return (tmin, tmax)

    def _filter_rows_by_window(self, rows: List[Tuple[float, float]], tmin: Optional[float], tmax: Optional[float]):
        """Rows is a list of (timestamp, value). Return rows within [tmin, tmax]."""
        if tmin is None or tmax is None:
            return rows
        return [rv for rv in rows if (rv[0] >= tmin and rv[0] <= tmax)]

    def on_set_group_trim(self):
        """Prompt for and apply group-level trim values."""
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        # Prompt for start and end trim (HH:MM:SS)
        top = tk.Toplevel(self); top.title(f"Set Trim — {g.name}")
        ttk.Label(top, text="Start trim (HH:MM:SS):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        e_start = ttk.Entry(top, width=16)
        e_start.insert(0, fmt_hhmmss(max(0, int(getattr(g, "trim_start_s", 0)))))
        e_start.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(top, text="End trim (HH:MM:SS):").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        e_end = ttk.Entry(top, width=16)
        e_end.insert(0, fmt_hhmmss(max(0, int(getattr(g, "trim_end_s", 0)))))
        e_end.grid(row=1, column=1, sticky="w", padx=6, pady=6)
        btns = ttk.Frame(top); btns.grid(row=2, column=0, columnspan=2, sticky="e", padx=6, pady=6)
        def save():
            try:
                g.trim_start_s = max(0, parse_hhmmss(e_start.get()))
                g.trim_end_s   = max(0, parse_hhmmss(e_end.get()))
            except Exception as ex:
                messagebox.showwarning("Invalid", str(ex)); return
            top.destroy()
            self._refresh_groups_tv()
            self._invalidate_aligned_cache()
        ttk.Button(btns, text="OK", command=save).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side=tk.RIGHT, padx=4)

    def on_clear_group_trim(self):
        """Clear trim values for the selected group."""
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        g.trim_start_s = 0; g.trim_end_s = 0
        self._refresh_groups_tv()
        self._invalidate_aligned_cache()

    def _ensure_group_selected_or_prompt(self) -> Optional[str]:
        """Return a selected group id or prompt the user to choose one."""
        sel = self.tv_groups.selection()
        if sel: return sel[0]
        if not self.groups:
            messagebox.showinfo("Groups", "Create a group first."); return None
        choices = list(self.groups.keys()); labels = [self.groups[c].name for c in choices]
        idx = simpledialog.askinteger("Select Group", "Enter group number:\n"+ "\n".join(f"{i+1}. {labels[i]}" for i in range(len(labels))),
                                      minvalue=1, maxvalue=len(labels), parent=self)
        return choices[idx-1] if idx else None

    def _on_group_selection_changed(self):
        """Handle group selection changes (currently no-op)."""
        pass  # selection is used via friendly comboboxes

    # ---- Settings ----
    def on_set_power_columns(self):
        """Edit the candidate column names used for power CSV detection."""
        top = tk.Toplevel(self); top.title("Power Columns Map"); top.resizable(True, False)
        ttk.Label(top, text="Candidate names for timestamp column (comma‑separated):").pack(anchor="w", padx=6, pady=(6,0))
        e1 = ttk.Entry(top, width=60); e1.insert(0, ", ".join(self.power_column_map["timestamp"])); e1.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(top, text="Candidate names for watts column (comma‑separated):").pack(anchor="w", padx=6, pady=(6,0))
        e2 = ttk.Entry(top, width=60); e2.insert(0, ", ".join(self.power_column_map["watts"])); e2.pack(fill=tk.X, padx=6, pady=3)
        def save():
            self.power_column_map["timestamp"] = [s.strip() for s in e1.get().split(",") if s.strip()]
            self.power_column_map["watts"] = [s.strip() for s in e2.get().split(",") if s.strip()]
            self._power_csv_cache.clear()
            self._invalidate_aligned_cache()
            top.destroy()
        b = ttk.Frame(top); b.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(b, text="OK", command=save).pack(side=tk.RIGHT, padx=3)
        ttk.Button(b, text="Cancel", command=top.destroy).pack(side=tk.RIGHT, padx=3)

    def on_add_module_path(self):
        """Add a folder to the module search path for eegbin/util."""
        folder = filedialog.askdirectory(title="Add folder to Python module search path (eegbin/util)")
        if not folder: return
        _add_module_path(folder); self._set_status(f"Added module path: {folder}")

    def on_list_module_paths(self):
        """Show the currently registered extra module paths."""
        if not _EXTRA_MODULE_PATHS:
            messagebox.showinfo("Module Paths", "No extra module paths added."); return
        msg = "Extra module search paths:\n\n" + "\n".join("• "+p for p in _EXTRA_MODULE_PATHS)
        messagebox.showinfo("Module Paths", msg)

    # ---- Reload ----
    def _start_background_reload(self, targets: List[str]) -> bool:
        """Start asynchronous metadata reload for selected/all files."""
        if self._reload_thread is not None and self._reload_thread.is_alive():
            return False

        reload_items: List[Tuple[str, str, str, str]] = []
        has_sw3 = False
        for fid in targets:
            rec = self.files.get(fid)
            if not rec:
                continue
            reload_items.append((fid, rec.kind, rec.path, rec.label))
            has_sw3 = has_sw3 or (rec.kind == "sw3")
        if not reload_items:
            return False

        if has_sw3:
            # Ensure eegbin imports happen on the UI thread to avoid
            # background-thread dialogs if the module path is missing.
            try:
                ensure_eegbin_imported()
            except Exception:
                return False

        self._reload_job_id += 1
        job_id = self._reload_job_id
        total = len(reload_items)
        self._set_status(f"Reload started ({total} file(s))...")
        self._set_busy(True, "Reloading file metadata...")

        def _worker():
            updates: Dict[str, Dict[str, object]] = {}
            errors: List[Tuple[str, str]] = []
            ok = 0
            try:
                for idx, (fid, kind, path, label) in enumerate(reload_items, start=1):
                    shown = os.path.basename(path) or label or fid
                    self._reload_queue.put(("progress", job_id, idx, total, shown))
                    try:
                        if kind == "sw3":
                            sweep = self._load_sweep_from_path(path, warn=False, trace=False)
                            if sweep is None:
                                raise RuntimeError("Failed to load SW3 file.")
                            meta = self._meta_from_sweep(sweep)
                        else:
                            _, meta = self._load_power_csv_with_meta(path)
                        updates[fid] = meta
                        ok += 1
                    except Exception as e:
                        errors.append((label or fid, str(e)))
            except Exception:
                errors.append(("reload worker", traceback.format_exc()))
            finally:
                err = max(0, total - ok)
                self._reload_queue.put(("done", job_id, updates, ok, err, errors))

        self._reload_thread = threading.Thread(
            target=_worker,
            name=f"reload-meta-{job_id}",
            daemon=True,
        )
        self._reload_thread.start()
        self.after(120, self._poll_background_reload)
        return True

    def _poll_background_reload(self):
        """Consume background reload progress and apply final metadata updates."""
        saw_done = False
        while True:
            try:
                event = self._reload_queue.get_nowait()
            except Empty:
                break
            etype = event[0]
            if etype == "progress":
                _, job_id, idx, total, shown = event
                if job_id != self._reload_job_id:
                    continue
                msg = f"Reloading {idx}/{total}: {shown}"
                self._set_status(msg)
                self._set_busy(True, msg)
                continue
            if etype == "done":
                _, job_id, updates, ok, err, errors = event
                if job_id != self._reload_job_id:
                    continue
                saw_done = True
                for fid, meta in updates.items():
                    rec = self.files.get(fid)
                    if rec is not None:
                        rec.meta = meta
                self._refresh_files_tv()
                self._refresh_display_mappings()
                self._set_status(f"Reloaded {ok}/{ok + err} file(s){'; errors: ' + str(err) if err else ''}.")
                if err:
                    details = []
                    for lab, text in errors[:5]:
                        details.append(f"• {lab}: {text}")
                    more = "" if len(errors) <= 5 else f"\n• ... and {len(errors) - 5} more"
                    messagebox.showwarning(
                        "Reload",
                        "Reload completed with errors:\n\n" + "\n".join(details) + more,
                    )
                if ok:
                    self._invalidate_aligned_cache()
                    self._trigger_reprocess_after_reload()

        if self._reload_thread is not None and self._reload_thread.is_alive():
            self.after(120, self._poll_background_reload)
            return
        if self._reload_thread is not None and not saw_done:
            # Catch any final queued completion event after thread exit.
            self.after(60, self._poll_background_reload)
            return
        self._reload_thread = None
        self._set_busy(self._is_any_background_busy())

    def on_reload_all_files(self, only_selected: bool=False):
        """Reload file metadata from disk (all or selected) in background."""
        if self._reload_thread is not None and self._reload_thread.is_alive():
            messagebox.showinfo("Reload", "Reload is already running in the background.")
            return
        targets = list(self.tv_files.selection()) if only_selected else list(self.files.keys())
        if only_selected and not targets:
            messagebox.showinfo("Reload", "No files selected.")
            return
        if not targets:
            self._set_status("No files available to reload.")
            return
        started = self._start_background_reload(targets)
        if not started:
            messagebox.showinfo("Reload", "Could not start reload.")

    # ---- Session I/O ----
    def on_save_session(self):
        """Save the full session (files, groups, controls) to JSON."""
        path = filedialog.asksaveasfilename(title="Save Session", defaultextension=".json",
                                            filetypes=[("Session JSON","*.json")])
        if not path: return
        data = {
            "power_column_map": self.power_column_map,
            "module_paths": list(_EXTRA_MODULE_PATHS),
            "files": [asdict(fr) for fr in self.files.values()],
            "groups": [{
                "group_id": g.group_id, "name": g.name,
                "trim_start_s": getattr(g, "trim_start_s", 0), "trim_end_s": getattr(g, "trim_end_s", 0),
                "file_ids": list(g.file_ids),
                "associations": {k: list(v) for k, v in g.associations.items()},
            } for g in self.groups.values()],
            "controls": {
                "intensity_ema_span": self.intensity_ema_span.get(),
                "power_ema_span": self.power_ema_span.get(),
                "overlay_ema_span": self.overlay_ema_span.get(),
                "trim_start_s": self.trim_start_s.get(),
                "trim_end_s": self.trim_end_s.get(),
                "align_tolerance_s": self.align_tolerance_s.get(),
                "ccf_max_lag_s": self.ccf_max_lag_s.get(),
                "resample_seconds": self.resample_seconds.get(),
                "time_weighted_ema": self.time_weighted_ema.get(),
                "auto_power_ema": self.auto_power_ema.get(),
                "normalize_to_1m": self.normalize_to_1m.get(),
                "only_yaw_roll_zero": self.only_yaw_roll_zero.get(),
                "ivt_show_points": self.ivt_show_points.get(),
                "ivt_show_ema": self.ivt_show_ema.get(),
                "ivt_point_alpha": float(self.ivt_point_alpha.get()),
                "ivt_line_alpha": float(self.ivt_line_alpha.get()),
                "ovp_show_points": self.ovp_show_points.get(),
                "ovp_show_int_ema": self.ovp_show_int_ema.get(),
                "ovp_show_pow_ema": self.ovp_show_pow_ema.get(),
                "ovp_point_alpha": float(self.ovp_point_alpha.get()),
                "ovp_line_alpha": float(self.ovp_line_alpha.get()),
                "gdo_show_points": self.gdo_show_points.get(),
                "gdo_show_ema": self.gdo_show_ema.get(),
                "gdo_point_alpha": float(self.gdo_point_alpha.get()),
                "gdo_line_alpha": float(self.gdo_line_alpha.get()),
                "source_mode": self.source_mode.get(),
                "combine_group_sw3": self.combine_group_sw3.get(),
                "ivt_sw3_display": self.ivt_sw3_display.get(),
                "ivt_group_display": self.ivt_group_display.get(),
                "ovp_group_display": self.ovp_group_display.get(),
                "controls_visible": self.controls_visible.get(),
            }
        }
        with open(path, "w") as fd: json.dump(data, fd, indent=2)
        self._set_status(f"Saved session to {path}")

    def on_load_session(self):
        """Load a session JSON file and restore UI state."""
        path = filedialog.askopenfilename(title="Load Session",
                                          filetypes=[("Session JSON","*.json"),("All files","*.*")])
        if not path: return
        with open(path, "r") as fd: data = json.load(fd)
        _EXTRA_MODULE_PATHS.clear()
        for p in data.get("module_paths", []): _add_module_path(p)
        self.power_column_map = data.get("power_column_map", self.power_column_map)
        self._power_csv_cache.clear()
        self.files = {fr["file_id"]: FileRecord(**fr) for fr in data.get("files", [])}
        self.groups = {}
        for g in data.get("groups", []):
            gr = GroupRecord(group_id=g["group_id"], name=g["name"],
                             trim_start_s=g.get("trim_start_s", 0), trim_end_s=g.get("trim_end_s", 0))
            gr.file_ids = set(g.get("file_ids", []))
            gr.associations = {k: set(v) for k,v in g.get("associations", {}).items()}
            self.groups[gr.group_id] = gr
        c = data.get("controls", {})
        def set_if(k,var):
            if k in c: var.set(c[k])
        for k,var in [
            ("intensity_ema_span", self.intensity_ema_span),
            ("power_ema_span", self.power_ema_span),
            ("overlay_ema_span", self.overlay_ema_span),
            ("trim_start_s", self.trim_start_s),
            ("trim_end_s", self.trim_end_s),
            ("align_tolerance_s", self.align_tolerance_s),
            ("ccf_max_lag_s", self.ccf_max_lag_s),
            ("resample_seconds", self.resample_seconds),
            ("time_weighted_ema", self.time_weighted_ema),
            ("auto_power_ema", self.auto_power_ema),
            ("normalize_to_1m", self.normalize_to_1m),
            ("only_yaw_roll_zero", self.only_yaw_roll_zero),
            ("ivt_show_points", self.ivt_show_points),
            ("ivt_show_ema", self.ivt_show_ema),
            ("ivt_point_alpha", self.ivt_point_alpha),
            ("ivt_line_alpha", self.ivt_line_alpha),
            ("ovp_show_points", self.ovp_show_points),
            ("ovp_show_int_ema", self.ovp_show_int_ema),
            ("ovp_show_pow_ema", self.ovp_show_pow_ema),
            ("ovp_point_alpha", self.ovp_point_alpha),
            ("ovp_line_alpha", self.ovp_line_alpha),
            ("gdo_show_points", self.gdo_show_points),
            ("gdo_show_ema", self.gdo_show_ema),
            ("gdo_point_alpha", self.gdo_point_alpha),
            ("gdo_line_alpha", self.gdo_line_alpha),
            ("source_mode", self.source_mode),
            ("combine_group_sw3", self.combine_group_sw3),
            ("ivt_sw3_display", self.ivt_sw3_display),
            ("ivt_group_display", self.ivt_group_display),
            ("ovp_group_display", self.ovp_group_display),
            ("controls_visible", self.controls_visible),
        ]: set_if(k,var)

        self._refresh_files_tv()
        self._refresh_groups_tv()
        self._refresh_display_mappings()
        self.on_toggle_controls()
        self._invalidate_aligned_cache()
        self._last_analyzed_gid = self._current_ovp_group_id()
        self._autostart_processing_after_session_load()

    def _invalidate_aligned_cache(self):
        """Drop cached aligned data after source files/settings changes."""
        self._aligned_cache = None
        self._aligned_cache_gid = None
        self._aligned_cache_signature = None
        self._ovp_series_cache = None
        self._sw3_plot_cache.clear()

    def _snapshot_analysis_params(self) -> Dict[str, object]:
        """Capture analysis settings so background workers avoid Tk variable access."""
        return {
            "intensity_ema_span": int(self.intensity_ema_span.get()),
            "power_ema_span": int(self.power_ema_span.get()),
            "align_tolerance_s": int(self.align_tolerance_s.get()),
            "time_weighted_ema": bool(self.time_weighted_ema.get()),
            "auto_power_ema": bool(self.auto_power_ema.get()),
            "only_yaw_roll_zero": bool(self.only_yaw_roll_zero.get()),
            "normalize_to_1m": bool(self.normalize_to_1m.get()),
            "resample_seconds": int(self.resample_seconds.get()),
        }

    def _analysis_params_signature(self, params: Dict[str, object]) -> Tuple:
        """Create a comparable immutable signature for analysis settings."""
        return (
            int(params["intensity_ema_span"]),
            int(params["power_ema_span"]),
            int(params["align_tolerance_s"]),
            bool(params["time_weighted_ema"]),
            bool(params["auto_power_ema"]),
            bool(params["only_yaw_roll_zero"]),
            bool(params["normalize_to_1m"]),
            int(params["resample_seconds"]),
        )

    def _current_ovp_group_id(self) -> Optional[str]:
        """Resolve the active OVP group id from combobox or current tree selection."""
        gdisp = self.ovp_group_display.get()
        gid = self._display_to_group_id(gdisp)
        if gid and gid in self.groups:
            return gid
        sel = self.tv_groups.selection()
        if sel and sel[0] in self.groups:
            return sel[0]
        return None

    def _start_background_alignment(self, gid: str) -> bool:
        """Start asynchronous aligned-data build for a group. Returns True if started."""
        if not gid or gid not in self.groups:
            return False
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            self._pending_reprocess_gid = gid
            if self._analysis_running_gid == gid:
                self._set_status(f"Background processing already running for '{self.groups[gid].name}'.")
            else:
                running_name = "unknown"
                if self._analysis_running_gid and self._analysis_running_gid in self.groups:
                    running_name = self.groups[self._analysis_running_gid].name
                self._set_status(
                    f"Background processing busy on '{running_name}'. Queued '{self.groups[gid].name}' next."
                )
            return False

        params = self._snapshot_analysis_params()
        params_sig = self._analysis_params_signature(params)
        self._analysis_job_id += 1
        job_id = self._analysis_job_id
        group_name = self.groups[gid].name
        self._analysis_running_gid = gid
        self._analysis_running_signature = params_sig
        self._set_status(f"Background processing started for '{group_name}'...")

        def _worker():
            try:
                aligned, plot_cache = self._build_aligned_for_group(
                    gid,
                    analysis_params=params,
                    ui_progress=False,
                    return_plot_cache=True,
                )
                self._analysis_queue.put((job_id, gid, aligned, None, params_sig, plot_cache))
            except Exception:
                self._analysis_queue.put((job_id, gid, None, traceback.format_exc(), params_sig, None))

        self._analysis_thread = threading.Thread(
            target=_worker,
            name=f"aligned-build-{gid}-{job_id}",
            daemon=True,
        )
        self._analysis_thread.start()
        self._set_busy(True, f"Preparing OVP cache ({group_name})...")
        self.after(120, self._poll_background_alignment)
        return True

    def _poll_background_alignment(self):
        """Harvest completed background alignment jobs and update cache/status."""
        saw_result = False
        while True:
            try:
                job_id, gid, aligned, err, params_sig, plot_cache = self._analysis_queue.get_nowait()
            except Empty:
                break
            saw_result = True
            if job_id != self._analysis_job_id:
                continue
            self._analysis_running_gid = None
            self._analysis_running_signature = None
            if err is not None:
                self._invalidate_aligned_cache()
                print(err, file=sys.stderr)
                self._set_status("Background processing failed. Check terminal for traceback.")
                messagebox.showwarning("Analyze", "Background processing failed. See terminal for details.")
                continue
            group_name = self.groups[gid].name if gid in self.groups else gid
            if plot_cache is not None:
                self._ovp_series_cache = plot_cache
            if aligned is None or aligned.empty:
                self._aligned_cache = None
                self._aligned_cache_gid = None
                self._aligned_cache_signature = None
                if plot_cache and (
                    plot_cache.get("sw3_series_by_id") or plot_cache.get("power_series_by_id")
                ):
                    self._set_status(
                        f"Background processing finished for '{group_name}' (partial streams found, but no aligned overlap)."
                    )
                else:
                    self._set_status(f"Background processing finished for '{group_name}' with no aligned data.")
                continue
            self._aligned_cache = aligned
            self._aligned_cache_gid = gid
            self._aligned_cache_signature = params_sig
            self._last_analyzed_gid = gid
            self._set_status(f"Aligned data ready for '{group_name}'. Click Analyze Group to plot.")

        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            running_name = ""
            if self._analysis_running_gid and self._analysis_running_gid in self.groups:
                running_name = self.groups[self._analysis_running_gid].name
            if running_name:
                self._set_busy(True, f"Preparing OVP cache ({running_name})...")
            else:
                self._set_busy(True, "Preparing OVP cache...")
            self.after(120, self._poll_background_alignment)
            return

        if self._pending_reprocess_gid:
            gid = self._pending_reprocess_gid
            self._pending_reprocess_gid = None
            self._start_background_alignment(gid)
            return

        if saw_result and self._analysis_running_gid is None and not self._pending_reprocess_gid:
            self._analysis_thread = None
        if self._analysis_thread is not None and not self._analysis_thread.is_alive() and not self._pending_reprocess_gid:
            self._analysis_thread = None
            self._analysis_running_gid = None
            self._analysis_running_signature = None
        self._set_busy(self._is_any_background_busy())

    def _ensure_group_aligned_ready(self, gid: str, action_label: str) -> Optional[pd.DataFrame]:
        """Return aligned cache for gid, or start background processing and warn."""
        if not gid or gid not in self.groups:
            messagebox.showinfo(action_label, "Select a Group.")
            return None
        current_sig = self._analysis_params_signature(self._snapshot_analysis_params())
        if (
            self._aligned_cache_gid == gid
            and self._aligned_cache is not None
            and not self._aligned_cache.empty
            and self._aligned_cache_signature == current_sig
        ):
            return self._aligned_cache
        if self._aligned_cache_gid == gid and self._aligned_cache_signature != current_sig:
            self._invalidate_aligned_cache()
        ovp_cache = self._ovp_series_cache or {}
        if (
            ovp_cache.get("gid") == gid
            and ovp_cache.get("signature") == current_sig
            and (
                bool(ovp_cache.get("sw3_series_by_id"))
                or bool(ovp_cache.get("power_series_by_id"))
            )
            and (self._analysis_thread is None or not self._analysis_thread.is_alive())
        ):
            messagebox.showinfo(
                action_label,
                "No aligned overlap is available with current settings. Adjust trims or alignment tolerance.",
            )
            return None
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            if self._analysis_running_gid == gid and self._analysis_running_signature == current_sig:
                messagebox.showinfo(
                    action_label,
                    f"Data for '{self.groups[gid].name}' is still processing in background. Try again when ready.",
                )
            else:
                self._pending_reprocess_gid = gid
                messagebox.showinfo(
                    action_label,
                    f"Data for '{self.groups[gid].name}' is queued after current background processing.",
                )
            return None
        started = self._start_background_alignment(gid)
        if started:
            messagebox.showinfo(
                action_label,
                f"Preparing '{self.groups[gid].name}' in background. Try again when status says data is ready.",
            )
        else:
            messagebox.showinfo(action_label, "Could not start background processing for this group.")
        return None

    def _ensure_group_ovp_ready(self, gid: str, action_label: str) -> Optional[Dict[str, object]]:
        """Return OVP series cache for gid, or start background processing and warn."""
        if not gid or gid not in self.groups:
            messagebox.showinfo(action_label, "Select a Group.")
            return None
        current_sig = self._analysis_params_signature(self._snapshot_analysis_params())
        cache = self._ovp_series_cache or {}
        if (
            cache.get("gid") == gid
            and cache.get("signature") == current_sig
            and (
                bool(cache.get("sw3_series_by_id"))
                or bool(cache.get("power_series_by_id"))
            )
        ):
            return cache
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            if self._analysis_running_gid == gid and self._analysis_running_signature == current_sig:
                messagebox.showinfo(
                    action_label,
                    f"Data for '{self.groups[gid].name}' is still processing in background. Try again when ready.",
                )
            else:
                self._pending_reprocess_gid = gid
                messagebox.showinfo(
                    action_label,
                    f"Data for '{self.groups[gid].name}' is queued after current background processing.",
                )
            return None
        started = self._start_background_alignment(gid)
        if started:
            messagebox.showinfo(
                action_label,
                f"Preparing '{self.groups[gid].name}' in background. Try again when status says data is ready.",
            )
        else:
            messagebox.showinfo(action_label, "Could not start background processing for this group.")
        return None

    def _trigger_reprocess_after_reload(self):
        """After reload, reprocess the most relevant group in background."""
        gid = self._pick_group_for_background_prep()
        if gid is not None:
            self._start_background_alignment(gid)
        self._start_background_sw3_preprocess_all()

    def _pick_group_for_background_prep(self) -> Optional[str]:
        """Choose the best group candidate for automatic background processing."""
        if self._last_analyzed_gid and self._last_analyzed_gid in self.groups:
            return self._last_analyzed_gid

        gid = self._current_ovp_group_id()
        if gid and gid in self.groups:
            return gid

        scored = []
        for gid_i, g in self.groups.items():
            if not g.associations:
                continue
            pair_count = sum(len(v) for v in g.associations.values())
            sw3_count = sum(
                1 for fid in g.file_ids
                if fid in self.files and self.files[fid].kind == "sw3"
            )
            pow_count = sum(
                1 for fid in g.file_ids
                if fid in self.files and self.files[fid].kind == "power"
            )
            scored.append((pair_count, min(sw3_count, pow_count), sw3_count + pow_count, gid_i))

        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][3]

    def _autostart_processing_after_session_load(self):
        """Kick off background processing immediately after session load."""
        self._start_background_sw3_preprocess_all()
        gid = self._pick_group_for_background_prep()
        if gid is None:
            self._set_status("Loaded session (no associated group available for auto-processing).")
            return
        started = self._start_background_alignment(gid)
        if not started:
            self._set_status(
                f"Loaded session. Auto-processing queued for '{self.groups[gid].name}'."
            )

    def _sw3_preprocess_signature(self, only_zero: bool, normalize_to_1m: bool) -> Tuple:
        """Return a stable signature for SW3 plot-cache preprocessing."""
        sw3_ids = tuple(sorted(
            fid for fid, rec in self.files.items()
            if rec.kind == "sw3"
        ))
        return (bool(only_zero), bool(normalize_to_1m), sw3_ids)

    def _start_background_sw3_preprocess_all(self) -> bool:
        """Build SW3 rows cache for IVT/GDO plots in background."""
        only_zero = bool(self.only_yaw_roll_zero.get())
        normalize_to_1m = bool(self.normalize_to_1m.get())
        sig = self._sw3_preprocess_signature(only_zero, normalize_to_1m)
        sw3_ids = list(sig[2])
        if not sw3_ids:
            return False
        if self._sw3_preprocess_thread is not None and self._sw3_preprocess_thread.is_alive():
            if self._sw3_preprocess_running_signature == sig:
                return False
            return False

        files_map = dict(self.files)
        self._sw3_preprocess_job_id += 1
        job_id = self._sw3_preprocess_job_id
        self._sw3_preprocess_running_signature = sig
        self._set_status(f"Background SW3 preprocessing started ({len(sw3_ids)} files)...")

        def _worker():
            cache_updates: Dict[Tuple[str, bool, bool], List[Tuple[float, float]]] = {}
            loaded = 0
            for fid in sw3_ids:
                rec = files_map.get(fid)
                if not rec or rec.kind != "sw3":
                    continue
                sweep = self._load_sweep_from_path(rec.path, warn=False, trace=False)
                if sweep is None:
                    continue
                rows = self._extract_intensity_rows(
                    sweep,
                    only_zero=only_zero,
                    normalize_to_1m=normalize_to_1m,
                    include_row=False,
                )
                cache_updates[(fid, only_zero, normalize_to_1m)] = rows
                loaded += 1
            self._sw3_preprocess_queue.put((job_id, sig, cache_updates, loaded, len(sw3_ids)))

        self._sw3_preprocess_thread = threading.Thread(
            target=_worker,
            name=f"sw3-prep-{job_id}",
            daemon=True,
        )
        self._sw3_preprocess_thread.start()
        self._set_busy(True, "Preparing SW3 cache...")
        self.after(120, self._poll_background_sw3_preprocess)
        return True

    def _poll_background_sw3_preprocess(self):
        """Harvest SW3 background preprocessing results."""
        saw_result = False
        while True:
            try:
                job_id, sig, cache_updates, loaded, total = self._sw3_preprocess_queue.get_nowait()
            except Empty:
                break
            saw_result = True
            if job_id != self._sw3_preprocess_job_id:
                continue
            self._sw3_plot_cache.update(cache_updates)
            self._sw3_preprocess_running_signature = None
            self._set_status(f"SW3 preprocessing ready ({loaded}/{total} files).")

        if self._sw3_preprocess_thread is not None and self._sw3_preprocess_thread.is_alive():
            self._set_busy(True, "Preparing SW3 cache...")
            self.after(120, self._poll_background_sw3_preprocess)
            return
        if saw_result:
            self._sw3_preprocess_thread = None
        self._set_busy(self._is_any_background_busy())

    def _get_sw3_rows_cached(
        self,
        fid: str,
        only_zero: bool,
        normalize_to_1m: bool,
        warn: bool = True,
    ) -> List[Tuple[float, float]]:
        """Get SW3 rows from cache or load on demand."""
        key = (fid, bool(only_zero), bool(normalize_to_1m))
        rows = self._sw3_plot_cache.get(key)
        if rows is not None:
            return rows
        rec = self.files.get(fid)
        if not rec or rec.kind != "sw3":
            return []
        sweep = self._load_sweep_from_path(rec.path, warn=warn, trace=False)
        if sweep is None:
            return []
        rows = self._extract_intensity_rows(
            sweep,
            only_zero=bool(only_zero),
            normalize_to_1m=bool(normalize_to_1m),
            include_row=False,
        )
        self._sw3_plot_cache[key] = rows
        return rows

    # ------------------------------------------------------------------
    # Analysis: Intensity vs Time
    # ------------------------------------------------------------------
    def on_plot_intensity_vs_time(self):
        """Plot intensity vs time for a file or group selection."""
        # Replace any existing IVT plot(s)
        self._close_plots('ivt', also=('spec',))

        mode = self.source_mode.get()
        ema_span = int(self.intensity_ema_span.get())
        trim_start = int(self.trim_start_s.get())
        trim_end   = int(self.trim_end_s.get())
        timeaware_ema = self.time_weighted_ema.get()
        only_zero  = self.only_yaw_roll_zero.get()
        norm_1m    = self.normalize_to_1m.get()
        show_pts   = self.ivt_show_points.get()
        show_line  = self.ivt_show_ema.get()
        p_alpha    = float(self.ivt_point_alpha.get())
        l_alpha    = float(self.ivt_line_alpha.get())

        def rows_from_sw3(fid: str):
            return self._get_sw3_rows_cached(
                fid=fid,
                only_zero=only_zero,
                normalize_to_1m=norm_1m,
                warn=True,
            )

        fig = plt.figure(); ax = fig.add_subplot(111)

        if mode == "file":
            disp = self.ivt_sw3_display.get()
            fid = self._display_to_sw3_id(disp)
            if not fid:
                messagebox.showinfo("Plot", "Select an SW3 file."); return
            rows = rows_from_sw3(fid)

            # Apply per-action trim (seconds) relative to series start
            if rows:
                t = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
                t0 = t[0]
                if trim_start>0: mask = (t - t0) >= trim_start; t,y = t[mask], y[mask]
                if trim_end>0:   tend = t[-1]; mask = (tend - t) >= trim_end; t,y = t[mask], y[mask]
                # If the file belongs to a trimmed group, clip to group window as well
                for g in self.groups.values():
                    if fid in g.file_ids:
                        win = self._group_time_window(g)
                        if win is not None:
                            tmin, tmax = win
                            mask = (t >= tmin) & (t <= tmax)
                            t, y = t[mask], y[mask]
                        break
            else:
                messagebox.showinfo("Plot", "No rows after filtering."); return

            if t.size==0:
                messagebox.showinfo("Plot", "No data to plot after trims."); return
            y_ema = ema_adaptive(t, y, ema_span, timeaware_ema)
            th = (t - t[0]) / 3600.0
            if show_pts: ax.scatter(th, y, s=8, alpha=p_alpha, label=f"{self.files[fid].label} (raw)")
            if show_line:
                t_line, y_line = _insert_nan_gaps(t, y_ema, _gap_threshold_s(t, floor_s=60.0))
                ax.plot((t_line - t[0]) / 3600.0, y_line, linewidth=2, alpha=l_alpha, label=f"{self.files[fid].label} EMA")
        else:
            gdisp = self.ivt_group_display.get()
            gid = self._display_to_group_id(gdisp)
            if not gid or gid not in self.groups: messagebox.showinfo("Plot", "Select a Group."); return
            g = self.groups[gid]
            win = self._group_time_window(g)
            sw3_ids = [fid for fid in g.file_ids if self.files.get(fid) and self.files[fid].kind=="sw3"]
            if not sw3_ids: messagebox.showinfo("Plot", "No SW3 files in this group."); return
            colors = get_cmap_colors(len(sw3_ids), "viridis")
            if self.combine_group_sw3.get():
                all_rows = []
                for fid in sw3_ids:
                    all_rows.extend(rows_from_sw3(fid))
                if win is not None and all_rows:
                    tmin, tmax = win
                    all_rows = self._filter_rows_by_window(all_rows, tmin, tmax)
                if not all_rows: messagebox.showinfo("Plot", "No rows after filtering."); return
                all_rows.sort(key=lambda x: x[0])
                t = np.array([r[0] for r in all_rows]); y = np.array([r[1] for r in all_rows])
                y_ema = ema_adaptive(t, y, ema_span, timeaware_ema); th = (t - t[0]) / 3600.0
                if show_pts: ax.scatter(th, y, s=8, alpha=p_alpha, label=f"{g.name} (raw)")
                if show_line:
                    t_line, y_line = _insert_nan_gaps(t, y_ema, _gap_threshold_s(t, floor_s=60.0))
                    ax.plot((t_line - t[0]) / 3600.0, y_line, linewidth=2, alpha=l_alpha, label=f"{g.name} EMA")
            else:
                for i, fid in enumerate(sw3_ids):
                    rows = rows_from_sw3(fid)
                    if win is not None and rows:
                        tmin, tmax = win
                        rows = self._filter_rows_by_window(rows, tmin, tmax)
                    if not rows: continue
                    t = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
                    y_ema = ema_adaptive(t, y, ema_span, timeaware_ema); th = (t - t[0]) / 3600.0
                    c = colors[i]
                    if show_pts: ax.scatter(th, y, s=8, alpha=p_alpha, label=f"{self.files[fid].label} (raw)", color=c)
                    if show_line:
                        t_line, y_line = _insert_nan_gaps(t, y_ema, _gap_threshold_s(t, floor_s=60.0))
                        ax.plot((t_line - t[0]) / 3600.0, y_line, linewidth=2, alpha=l_alpha, label=f"{self.files[fid].label} EMA", color=c)

        ax.set_title("Measured Light Intensity vs Time")
        ax.set_xlabel("Time since start (hours)")
        ax.set_ylabel("Normalized Intensity at 1 m (µW/cm²)")
        ax.grid(True, linestyle="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        outside = _apply_smart_legend(ax, handles, labels)
        if outside:
            fig.tight_layout(rect=(0, 0, 0.78, 1))
        else:
            fig.tight_layout()
        self._register_fig('ivt', fig)
        self._attach_wavelength_inspector_to_ivt(fig)
        plt.show(block=False)

    # ------------------------------------------------------------------
    # Analysis: Build aligned frame for a group (applies group trims)
    # ------------------------------------------------------------------
    def _build_aligned_for_group(
        self,
        gid: str,
        analysis_params: Optional[Dict[str, object]] = None,
        ui_progress: bool = True,
        return_plot_cache: bool = False,
    ):
        """Return a merged intensity/power dataframe for a group or None."""
        g = self.groups.get(gid)
        if not g or not g.associations:
            return (None, None) if return_plot_cache else None
        files_map = dict(self.files)

        p = dict(analysis_params or self._snapshot_analysis_params())
        ema_int = int(p["intensity_ema_span"])
        ema_pow_manual = int(p["power_ema_span"])
        tol_s = int(p["align_tolerance_s"])
        timeaware_ema = bool(p["time_weighted_ema"])
        auto_power_ema = bool(p["auto_power_ema"])
        only_zero = bool(p["only_yaw_roll_zero"])
        norm_1m = bool(p["normalize_to_1m"])
        base_resample_s = max(1, int(p["resample_seconds"]))

        def report(status: str):
            if not ui_progress:
                return
            self._set_status(status)
            self.update_idletasks()

        frames = []
        win = self._group_time_window(g, files_map=files_map)

        # ------------------------------------------------------------------
        # Stage 1: preprocess SW3 once per file
        # ------------------------------------------------------------------
        sw3_ids = list(g.associations.keys())
        sw3_df_by_id: Dict[str, pd.DataFrame] = {}
        sw3_range_by_id: Dict[str, Tuple[float, float]] = {}
        pair_list: List[Tuple[str, str]] = []
        power_to_sw3: Dict[str, List[str]] = {}

        for i, sw3_id in enumerate(sw3_ids, start=1):
            report(f"Preparing SW3 {i}/{len(sw3_ids)}...")

            sw3_rec = files_map.get(sw3_id)
            if not sw3_rec or sw3_rec.kind != "sw3":
                continue

            sw_rows = self._get_sw3_rows_cached(
                fid=sw3_id,
                only_zero=only_zero,
                normalize_to_1m=norm_1m,
                warn=ui_progress,
            )
            if win is not None and sw_rows:
                tmin_win, tmax_win = win
                sw_rows = self._filter_rows_by_window(sw_rows, tmin_win, tmax_win)
            if not sw_rows:
                continue

            t_sw = np.array([r[0] for r in sw_rows], dtype=float)
            y_sw = np.array([r[1] for r in sw_rows], dtype=float)
            sw3_df_by_id[sw3_id] = pd.DataFrame(
                {"timestamp": t_sw, "intensity_ema": ema_adaptive(t_sw, y_sw, ema_int, timeaware_ema)}
            )
            sw3_range_by_id[sw3_id] = (float(t_sw.min()), float(t_sw.max()))

            for pid in g.associations.get(sw3_id, set()):
                pow_rec = files_map.get(pid)
                if not pow_rec or pow_rec.kind != "power":
                    continue
                pair_list.append((sw3_id, pid))
                power_to_sw3.setdefault(pid, []).append(sw3_id)

        if not pair_list:
            return (None, None) if return_plot_cache else None

        # Auto-tune power EMA from optics EMA and observed group cadence.
        sw3_step_candidates = []
        for sid, df_sw in sw3_df_by_id.items():
            ts = df_sw["timestamp"].to_numpy(dtype=float)
            sw3_step_candidates.append(_median_positive_step_s(ts))
        sw3_step_candidates = [x for x in sw3_step_candidates if np.isfinite(x) and x > 0]
        sw3_median_step_s = float(np.median(sw3_step_candidates)) if sw3_step_candidates else 1.0

        power_meta_step_candidates = []
        for pid in power_to_sw3.keys():
            rec = files_map.get(pid)
            if not rec:
                continue
            step_s = _estimate_step_s_from_meta(rec.meta)
            if step_s is not None and np.isfinite(step_s) and step_s > 0:
                power_meta_step_candidates.append(float(step_s))
        power_step_s = float(np.median(power_meta_step_candidates)) if power_meta_step_candidates else None

        if auto_power_ema:
            ema_pow = recommend_power_ema_span(
                intensity_ema_span=ema_int,
                sw3_median_step_s=sw3_median_step_s,
                resample_seconds=base_resample_s,
                power_step_s=power_step_s,
            )
            if ui_progress and ema_pow != self.power_ema_span.get():
                self.power_ema_span.set(ema_pow)
            report(f"Auto power EMA: {ema_pow} (optics span {ema_int}, sw3 step {sw3_median_step_s:.2f}s)")
        else:
            ema_pow = max(1, ema_pow_manual)

        # ------------------------------------------------------------------
        # Stage 2: preprocess power once per file across union window
        # ------------------------------------------------------------------
        power_series_by_id: Dict[str, pd.DataFrame] = {}
        power_ids = list(power_to_sw3.keys())
        for i, pid in enumerate(power_ids, start=1):
            report(f"Preparing power {i}/{len(power_ids)}...")

            pow_rec = files_map.get(pid)
            if not pow_rec or pow_rec.kind != "power":
                continue

            try:
                power_df, _ = self._load_power_csv_with_meta(pow_rec.path)
            except Exception as e:
                if ui_progress:
                    messagebox.showwarning("Load error", f"Failed to load CSV {pow_rec.path}\n{e}")
                continue

            linked_sw3 = [sid for sid in power_to_sw3.get(pid, []) if sid in sw3_range_by_id]
            if not linked_sw3:
                continue
            power_tmin = min(sw3_range_by_id[sid][0] for sid in linked_sw3) - tol_s
            power_tmax = max(sw3_range_by_id[sid][1] for sid in linked_sw3) + tol_s
            power_duration_s = max(0.0, float(power_tmax - power_tmin))
            effective_resample_s = choose_effective_resample_seconds(
                duration_s=power_duration_s,
                requested_seconds=base_resample_s,
            )
            # Keep smoothing timescale stable when auto-upscaling resample.
            ema_pow_effective = max(1, int(round(float(ema_pow) * base_resample_s / effective_resample_s)))
            if effective_resample_s > base_resample_s:
                report(
                    f"Preparing power {i}/{len(power_ids)} (auto resample {effective_resample_s}s for {power_duration_s/3600.0:.1f}h)..."
                )

            power_series = _prepare_power_series(
                power_df,
                power_tmin,
                power_tmax,
                ema_pow_effective,
                effective_resample_s,
                timeaware_ema,
            )
            if power_series is None or power_series.empty:
                continue
            power_series_by_id[pid] = power_series

        plot_cache = {
            "gid": gid,
            "signature": self._analysis_params_signature(p),
            "sw3_series_by_id": sw3_df_by_id,
            "power_series_by_id": power_series_by_id,
            "sw3_label_by_id": {sid: files_map[sid].label for sid in sw3_df_by_id.keys() if sid in files_map},
            "power_label_by_id": {pid: files_map[pid].label for pid in power_series_by_id.keys() if pid in files_map},
        }

        # ------------------------------------------------------------------
        # Stage 3: pairwise alignment from cached/prepared data
        # ------------------------------------------------------------------
        for i, (sw3_id, pid) in enumerate(pair_list, start=1):
            report(f"Aligning pair {i}/{len(pair_list)}...")

            df_sw = sw3_df_by_id.get(sw3_id)
            pdf_full = power_series_by_id.get(pid)
            sw3_rec = files_map.get(sw3_id)
            pow_rec = files_map.get(pid)
            if df_sw is None or pdf_full is None or sw3_rec is None or pow_rec is None:
                continue

            sw_tmin, sw_tmax = sw3_range_by_id[sw3_id]
            pdf = _slice_time_window(pdf_full, sw_tmin - tol_s, sw_tmax + tol_s, ts_col="timestamp")
            if pdf is None or pdf.empty:
                continue

            df_join = merge_asof_seconds(df_sw, pdf, tol_s).dropna(subset=["power_ema"])
            if df_join.empty:
                continue
            df_join["group_id"] = gid
            df_join["sw3_id"] = sw3_id
            df_join["power_id"] = pid
            df_join["sw3_label"] = sw3_rec.label
            df_join["power_label"] = pow_rec.label
            frames.append(df_join)

        if not frames:
            return (None, plot_cache) if return_plot_cache else None
        aligned = pd.concat(frames, ignore_index=True).sort_values("timestamp_s")
        if return_plot_cache:
            return aligned, plot_cache
        return aligned

    # ------------------------------------------------------------------
    # Analysis: OVP (time series) and correlation/scatter
    # ------------------------------------------------------------------
    def on_analyze_group(self):
        """Plot optics vs power using prepared series (or queue background prep)."""
        gid = self._current_ovp_group_id()
        ovp_cache = self._ensure_group_ovp_ready(gid, "Analyze")
        if ovp_cache is None:
            return
        aligned = None
        current_sig = self._analysis_params_signature(self._snapshot_analysis_params())
        if (
            self._aligned_cache is not None
            and not self._aligned_cache.empty
            and self._aligned_cache_gid == gid
            and self._aligned_cache_signature == current_sig
        ):
            aligned = self._aligned_cache
        self._plot_ovp_from_series(gid, ovp_cache, aligned)

    def _plot_ovp_from_series(
        self,
        gid: str,
        ovp_cache: Dict[str, object],
        aligned: Optional[pd.DataFrame] = None,
    ):
        """Render OVP plot from prepared SW3/power series cache."""
        # Replace any existing OVP plot(s)
        self._close_plots('ovp')

        show_pts = self.ovp_show_points.get()
        show_int = self.ovp_show_int_ema.get()
        show_pow = self.ovp_show_pow_ema.get()
        p_alpha  = float(self.ovp_point_alpha.get())
        l_alpha  = float(self.ovp_line_alpha.get())

        sw3_series_by_id = dict(ovp_cache.get("sw3_series_by_id", {}))
        power_series_by_id = dict(ovp_cache.get("power_series_by_id", {}))
        sw3_label_by_id = dict(ovp_cache.get("sw3_label_by_id", {}))
        power_label_by_id = dict(ovp_cache.get("power_label_by_id", {}))

        if not sw3_series_by_id and not power_series_by_id:
            messagebox.showinfo("Analyze", "No optical/power series available to plot.")
            return

        sw3_ids = sorted(sw3_series_by_id.keys())
        power_ids = sorted(power_series_by_id.keys())
        sw3_colors = get_solarized_colors(len(sw3_ids))
        power_colors = get_solarized_colors(len(power_ids))
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        t0_candidates = []
        for sid in sw3_ids:
            df_sw = sw3_series_by_id[sid]
            if df_sw is not None and not df_sw.empty:
                t0_candidates.append(float(df_sw["timestamp"].min()))
        for pid in power_ids:
            df_pow = power_series_by_id[pid]
            if df_pow is not None and not df_pow.empty:
                t0_candidates.append(float(df_pow["timestamp"].min()))
        if aligned is not None and not aligned.empty:
            t0_candidates.append(float(aligned["timestamp_s"].min()))
        if not t0_candidates:
            messagebox.showinfo("Analyze", "No data points available to plot.")
            return
        t0_global = min(t0_candidates)

        for i, sid in enumerate(sw3_ids):
            df_sw = sw3_series_by_id[sid]
            if df_sw is None or df_sw.empty:
                continue
            t_sw = df_sw["timestamp"].to_numpy(dtype=float)
            y_sw = df_sw["intensity_ema"].to_numpy(dtype=float)
            c_int = lighten(sw3_colors[i], 0.42)
            sw_label = sw3_label_by_id.get(sid, sid)
            gap_floor = max(float(self.align_tolerance_s.get()) * 3.0, 30.0)
            gap_s = _gap_threshold_s(t_sw, floor_s=gap_floor)
            if show_int:
                t_line, y_line = _insert_nan_gaps(t_sw, y_sw, gap_s)
                ax1.plot(
                    (t_line - t0_global) / 3600.0,
                    y_line,
                    label=f"{sw_label} (optical irradiance)",
                    alpha=l_alpha,
                    color=c_int,
                    linewidth=2,
                )
            if show_pts:
                ax1.scatter((t_sw - t0_global) / 3600.0, y_sw, s=8, alpha=p_alpha, color=c_int)

        for i, pid in enumerate(power_ids):
            df_pow = power_series_by_id[pid]
            if df_pow is None or df_pow.empty:
                continue
            t_pow = df_pow["timestamp"].to_numpy(dtype=float)
            y_pow = df_pow["power_ema"].to_numpy(dtype=float)
            c_pow = darken(power_colors[i], 0.35)
            pw_label = power_label_by_id.get(pid, pid)
            gap_floor = max(float(self.resample_seconds.get()) * 3.0, 30.0)
            gap_s = _gap_threshold_s(t_pow, floor_s=gap_floor)
            if show_pow:
                t_line, y_line = _insert_nan_gaps(t_pow, y_pow, gap_s)
                ax2.plot(
                    (t_line - t0_global) / 3600.0,
                    y_line,
                    label=f"{pw_label} (power)",
                    alpha=l_alpha,
                    color=c_pow,
                    linewidth=2,
                )

        ax1.set_xlabel("Time since earliest series start (hours)")
        ax1.set_ylabel("Intensity (µW/cm²)")
        ax2.set_ylabel("Power (W)", labelpad=10)
        ax2.tick_params(axis="y", pad=4)
        ax1.grid(True, linestyle="--", alpha=0.5)

        opt_handles, opt_labels = ax1.get_legend_handles_labels()
        pow_handles, pow_labels = ax2.get_legend_handles_labels()
        total_labels = len(opt_labels) + len(pow_labels)
        outside = total_labels > 6 or max([len(x) for x in (opt_labels + pow_labels)] + [0]) > 36
        if outside:
            if opt_labels:
                leg_opt = fig.legend(
                    opt_handles,
                    opt_labels,
                    title="Optical",
                    loc="upper left",
                    bbox_to_anchor=(0.695, 0.93),
                    bbox_transform=fig.transFigure,
                    borderaxespad=0.0,
                    fontsize=8,
                    frameon=True,
                )
            if pow_labels:
                fig.legend(
                    pow_handles,
                    pow_labels,
                    title="Power",
                    loc="lower left",
                    bbox_to_anchor=(0.695, 0.10),
                    bbox_transform=fig.transFigure,
                    borderaxespad=0.0,
                    fontsize=8,
                    frameon=True,
                )
        else:
            seen = {}
            for h, lab in zip(opt_handles + pow_handles, opt_labels + pow_labels):
                if lab not in seen:
                    seen[lab] = h
            _apply_smart_legend(ax1, list(seen.values()), list(seen.keys()))

        ax1.set_title(f"Optical irradiance and power vs time — {self.groups[gid].name}")
        if outside:
            # Reserve a dedicated right-side column for legends so they do not
            # overlap the right y-axis ticks/label.
            fig.subplots_adjust(left=0.08, right=0.60, top=0.92, bottom=0.12)
        else:
            fig.tight_layout()
        self._register_fig('ovp', fig)
        plt.show(block=False)
        self._set_status(f"Analyzed group '{self.groups[gid].name}' — time series plotted.")

    def on_corr_and_scatter(self):
        """Plot cross-correlation and scatter/regression for aligned data."""
        # Replace any existing correlation/scatter plots
        self._close_plots('corr')

        gid = self._current_ovp_group_id()
        aligned = self._ensure_group_aligned_ready(gid, "Correlation")
        if aligned is None:
            return

        x = aligned["intensity_ema"].to_numpy(dtype=float)
        y = aligned["power_ema"].to_numpy(dtype=float)

        # Cross-correlation scan
        lags = np.arange(-int(self.ccf_max_lag_s.get()), int(self.ccf_max_lag_s.get())+1, 1, dtype=int)
        ccf_vals = []
        for lag in lags:
            if lag < 0:
                x_lag = x[-lag:]; y_lag = y[:len(x_lag)]
            elif lag > 0:
                y_lag = y[lag:];  x_lag = x[:len(y_lag)]
            else:
                x_lag = x; y_lag = y
            if x_lag.size==0 or y_lag.size==0 or np.std(x_lag)==0 or np.std(y_lag)==0:
                ccf_vals.append(np.nan)
            else:
                ccf_vals.append(float(np.corrcoef(x_lag, y_lag)[0,1]))
        ccf_vals = np.array(ccf_vals, dtype=float)
        best_idx = int(np.nanargmax(np.abs(ccf_vals))) if np.isfinite(ccf_vals).any() else None
        best_lag = int(lags[best_idx]) if best_idx is not None else None
        best_ccf = float(ccf_vals[best_idx]) if best_idx is not None else float("nan")

        # 1) CCF plot
        fig_ccf = plt.figure(); axc = fig_ccf.add_subplot(111)
        axc.plot(lags, ccf_vals); axc.axvline(0, linestyle="--", alpha=0.5)
        if best_idx is not None:
            axc.axvline(best_lag, linestyle=":")
            y_text = np.nanmax(ccf_vals) if np.isfinite(ccf_vals).any() else 0.0
            axc.text(best_lag, y_text, f"best lag={best_lag}s\nr={best_ccf:.3f}", ha="center", va="bottom")
        axc.set_title("Cross‑correlation")
        axc.set_xlabel("Lag (s) [positive = intensity lags power]")
        axc.set_ylabel("Correlation")
        axc.grid(True, linestyle="--", alpha=0.5)
        fig_ccf.tight_layout()
        self._register_fig('corr', fig_ccf)
        plt.show(block=False)

        # 2) Scatter + regression
        slope, intercept, r_lin = linear_regression(aligned["power_ema"].to_numpy(), aligned["intensity_ema"].to_numpy())
        fig_sc = plt.figure(); axs = fig_sc.add_subplot(111)
        # color per pair (use friendly labels)
        pair_keys = sorted(set(zip(aligned["sw3_id"], aligned["power_id"])))
        base_colors = get_cmap_colors(len(pair_keys), "viridis")
        for i, (sw_id, pow_id) in enumerate(pair_keys):
            dfp = aligned[(aligned["sw3_id"]==sw_id) & (aligned["power_id"]==pow_id)]
            axs.scatter(dfp["power_ema"], dfp["intensity_ema"], s=10, alpha=0.25,
                        label=f"{dfp['sw3_label'].iloc[0]} vs {dfp['power_label'].iloc[0]}", color=base_colors[i])
        if np.isfinite(slope) and np.isfinite(intercept):
            xs = np.linspace(aligned["power_ema"].min(), aligned["power_ema"].max(), 200)
            ys = slope*xs + intercept
            axs.plot(xs, ys, linewidth=2, alpha=0.9, label=f"Fit: y={slope:.3f}x+{intercept:.3f}  r={r_lin:.3f}")
        axs.set_xlabel("Power (W) [EMA]"); axs.set_ylabel("Intensity (µW/cm²) [EMA]"); axs.grid(True, linestyle="--", alpha=0.5)
        handles, labels = axs.get_legend_handles_labels()
        outside = _apply_smart_legend(axs, handles, labels)
        if outside:
            fig_sc.tight_layout(rect=(0, 0, 0.78, 1))
        else:
            fig_sc.tight_layout()
        self._register_fig('corr', fig_sc); plt.show(block=False)

    def on_export_aligned_csv(self):
        """Export the last aligned dataframe to CSV."""
        gid = self._current_ovp_group_id()
        aligned = self._ensure_group_aligned_ready(gid, "Export")
        if aligned is None:
            return
        path = filedialog.asksaveasfilename(title="Export Aligned CSV", defaultextension=".csv",
                                            filetypes=[("CSV","*.csv")])
        if not path: return
        df = aligned.rename(columns={
            "timestamp_s":"Timestamp", "intensity_ema":"Intensity_EMA_uW_cm2", "power_ema":"Power_EMA_W"
        }).copy()
        keep = ["Timestamp","Intensity_EMA_uW_cm2","Power_EMA_W","group_id","sw3_id","power_id","sw3_label","power_label"]
        df[keep].to_csv(path, index=False)
        self._set_status(f"Exported aligned CSV to {path}")

    # ------------------------------------------------------------------
    # Analysis: Group Decay Overlay (applies trim)
    # ------------------------------------------------------------------
    def on_plot_group_decay(self, selected_only: bool):
        """Plot decay overlay for selected or all groups."""
        # Replace any existing GDO plot(s)
        self._close_plots('gdo')

        ema_span = int(self.intensity_ema_span.get())
        timeaware_ema = self.time_weighted_ema.get()
        only_zero = self.only_yaw_roll_zero.get()
        norm_1m   = self.normalize_to_1m.get()
        show_pts  = self.gdo_show_points.get()
        show_line = self.gdo_show_ema.get()
        p_alpha   = float(self.gdo_point_alpha.get())
        l_alpha   = float(self.gdo_line_alpha.get())

        if selected_only:
            idxs = self.lb_groups_select.curselection()
            gids = [self.lb_groups_select.get(i).split("  ")[0] for i in idxs]
        else:
            gids = list(self.groups.keys())
        if not gids:
            messagebox.showinfo("Plot", "No groups selected."); return

        base_colors = get_cmap_colors(len(gids), "viridis")
        fig = plt.figure(); ax = fig.add_subplot(111)

        for i, gid in enumerate(gids):
            g = self.groups.get(gid); 
            if not g: continue
            win = self._group_time_window(g)
            sw3_ids = [fid for fid in g.file_ids if self.files.get(fid) and self.files[fid].kind=="sw3"]
            rows_all = []
            for sid in sw3_ids:
                rows_all.extend(
                    self._get_sw3_rows_cached(
                        fid=sid,
                        only_zero=only_zero,
                        normalize_to_1m=norm_1m,
                        warn=True,
                    )
                )
            if win is not None and rows_all:
                tmin, tmax = win
                rows_all = self._filter_rows_by_window(rows_all, tmin, tmax)
            if not rows_all: continue
            rows_all.sort(key=lambda x:x[0])
            t = np.array([r[0] for r in rows_all]); y = np.array([r[1] for r in rows_all])
            y_ema = ema_adaptive(t, y, ema_span, timeaware_ema)
            y_pct = (y_ema/y_ema.max())*100.0 if y_ema.max()>0 else np.zeros_like(y_ema)
            th = (t - t[0]) / 3600.0
            c = base_colors[i]
            if show_pts: ax.scatter(th, y_pct, s=8, alpha=p_alpha, color=c, label=f"{g.name} (pts)")
            if show_line:
                t_line, y_line = _insert_nan_gaps(t, y_pct, _gap_threshold_s(t, floor_s=60.0))
                ax.plot((t_line - t[0]) / 3600.0, y_line, linewidth=2, alpha=l_alpha, color=c, label=g.name)

        ax.set_title("Decay Overlay by Group (EMA, normalized to each group’s peak)")
        ax.set_xlabel("Hours since group start"); ax.set_ylabel("Intensity (% of peak)")
        ax.grid(True, linestyle="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        outside = _apply_smart_legend(ax, handles, labels)
        if outside:
            fig.tight_layout(rect=(0, 0, 0.78, 1))
        else:
            fig.tight_layout()
        self._register_fig('gdo', fig)
        plt.show(block=False)

    # ---- Plot utils ----
    # ------------------------------------------------------------------
    # Interactive: Wavelength spectrum inspector for IVT plot
    # ------------------------------------------------------------------
    def _build_ivt_series_data_for_current_selection(self):
        """
        Build IVT series (timestamps in hours as plotted) **plus** row/sweep refs
        from the currently selected File or Group, honoring trims, zero-only, and
        1 m normalization toggle.
        Returns: list of dicts: {'label','t_hours','rows','sweeps'}
        """
        mode = self.source_mode.get()
        trim_start = int(self.trim_start_s.get())
        trim_end   = int(self.trim_end_s.get())
        only_zero  = self.only_yaw_roll_zero.get()
        norm_1m    = self.normalize_to_1m.get()

        ensure_eegbin_imported()

        def _extract_rows(fid: str):
            rec = self.files.get(fid)
            if not rec or rec.kind != "sw3":
                return []
            sweep = self._load_sweep_from_path(rec.path, warn=False, trace=False)
            if sweep is None:
                return []
            return self._extract_intensity_rows(sweep, only_zero, norm_1m, include_row=True)

        series = []

        if mode == "file":
            disp = self.ivt_sw3_display.get()
            fid = self._display_to_sw3_id(disp)
            if not fid:
                return []
            rows = _extract_rows(fid)
            if not rows:
                return []

            import numpy as np
            t = np.array([r[0] for r in rows], dtype=float)
            if t.size == 0:
                return []
            t0 = t[0]
            # Per-action trims
            mask = np.ones_like(t, dtype=bool)
            if trim_start > 0:
                mask &= (t - t0) >= trim_start
            if trim_end > 0:
                tend = t[-1]
                mask &= (tend - t) >= trim_end
            # Clip to group window if this file is in a trimmed group
            for g in self.groups.values():
                if fid in g.file_ids:
                    win = self._group_time_window(g)
                    if win is not None:
                        tmin, tmax = win
                        mask &= (t >= tmin) & (t <= tmax)
                    break
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                return []
            th = (t[idxs] - t[idxs][0]) / 3600.0
            series.append({
                "label": self.files[fid].label,
                "t_hours": th,
                "rows": [rows[i][2] for i in idxs],
                "sweeps": [rows[i][3] for i in idxs],
            })
        else:
            # Group mode
            gdisp = self.ivt_group_display.get()
            gid = self._display_to_group_id(gdisp)
            if not gid or gid not in self.groups:
                return []
            g = self.groups[gid]
            win = self._group_time_window(g)
            sw3_ids = [fid for fid in g.file_ids if self.files.get(fid) and self.files[fid].kind == "sw3"]
            if not sw3_ids:
                return []

            if self.combine_group_sw3.get():
                all_rows = []
                for fid in sw3_ids:
                    rows = _extract_rows(fid)
                    all_rows.extend(rows)
                if win is not None and all_rows:
                    tmin, tmax = win
                    all_rows = [rv for rv in all_rows if (rv[0] >= tmin and rv[0] <= tmax)]
                if not all_rows:
                    return []
                all_rows.sort(key=lambda x: x[0])
                import numpy as np
                t = np.array([rv[0] for rv in all_rows], dtype=float)
                th = (t - t[0]) / 3600.0
                series.append({
                    "label": g.name,
                    "t_hours": th,
                    "rows": [rv[2] for rv in all_rows],
                    "sweeps": [rv[3] for rv in all_rows],
                })
            else:
                for fid in sw3_ids:
                    rows = _extract_rows(fid)
                    if win is not None and rows:
                        tmin, tmax = win
                        rows = [rv for rv in rows if (rv[0] >= tmin and rv[0] <= tmax)]
                    if not rows:
                        continue
                    import numpy as np
                    t = np.array([rv[0] for rv in rows], dtype=float)
                    th = (t - t[0]) / 3600.0
                    series.append({
                        "label": self.files[fid].label,
                        "t_hours": th,
                        "rows": [rv[2] for rv in rows],
                        "sweeps": [rv[3] for rv in rows],
                    })

        return series

    def _attach_wavelength_inspector_to_ivt(self, fig: plt.Figure):
        """
        Attach pick and click handlers to IVT plot to open a Spectrum window.
        Respects toolbar zoom/pan, so it won't trigger while using those tools.
        """
        if not fig.axes:
            return
        ax = fig.axes[0]
        series = self._build_ivt_series_data_for_current_selection()
        if not series:
            return
        ax._ivt_series = series  # stash on axes

        # --- Toolbar guard: ignore while zoom/pan is active ---
        def _toolbar_mode_active():
            try:
                tb = getattr(fig.canvas.manager, "toolbar", None)
            except Exception:
                tb = None
            if tb is None:
                tb = getattr(fig.canvas, "toolbar", None)
            mode = ""
            try:
                mode = getattr(tb, "mode", "") or ""
            except Exception:
                mode = ""
            m = str(mode).lower()
            if m and (("zoom" in m) or ("pan" in m)):
                return True
            try:
                active = getattr(tb, "_active", None)
                if active:
                    return True
            except Exception:
                pass
            return False

        # --- Open-once guard: dedupe pick + click ---
        import time
        if not hasattr(fig, "_spec_click_guard"):
            fig._spec_click_guard = {"last_open_key": None, "t_last_open": 0.0}

        def _evt_key(mev):
            return (int(getattr(mev, "x", -1)),
                    int(getattr(mev, "y", -1)),
                    getattr(mev, "button", None))

        def _open_once(mouse_event, opener_callable, dt=0.25):
            if _toolbar_mode_active():
                return
            if getattr(mouse_event, "button", None) != 1:
                return  # left click only
            g = fig._spec_click_guard
            key = _evt_key(mouse_event)
            now = time.monotonic()
            if g["last_open_key"] == key and (now - g["t_last_open"] < dt):
                return
            g["last_open_key"] = key
            g["t_last_open"] = now
            opener_callable()

        # Tag artists with their base series by label (strip EMA/raw suffixes)
        def _base_label(lbl: str) -> str:
            if not isinstance(lbl, str): return ""
            s = lbl.replace(" EMA", "").replace(" (raw)", "")
            return s

        label_to_series = {s["label"]: s for s in series}
        for artist in list(ax.lines) + list(ax.collections):
            try:
                artist.set_picker(True)
            except Exception:
                pass
            sdict = label_to_series.get(_base_label(getattr(artist, "get_label", lambda: "")()))
            setattr(artist, "_ivt_series_meta", sdict)

        import numpy as np
        def nearest_index(sdict, xh):
            th = sdict["t_hours"]
            if getattr(th, "size", len(th)) == 0:
                return None
            return int(np.argmin(np.abs(th - float(xh))))

        def _nearest_with_spectrum(sdict, idx):
            rows = sdict["rows"]
            th = sdict["t_hours"]
            def has_spec(r):
                sr = getattr(r.capture, "spectral_result", None)
                return sr is not None and len(sr) > 0
            if idx is None:
                return None
            if has_spec(rows[idx]):
                return idx
            have = [i for i, r in enumerate(rows) if has_spec(r)]
            if not have:
                return None
            return min(have, key=lambda i: abs(th[i] - th[idx]))

        def show_for_index(sdict, idx, reason="click"):
            if sdict is None or idx is None:
                return
            idx2 = _nearest_with_spectrum(sdict, idx)
            if idx2 is None:
                return
            row = sdict["rows"][idx2]; sweep = sdict["sweeps"][idx2]
            self._show_wavelength_plot(row, sweep, sdict["label"])

        def on_pick(event):
            # Skip while toolbar is active or not a left click
            if _toolbar_mode_active():
                return
            mev = getattr(event, "mouseevent", None)
            if getattr(mev, "button", None) != 1:
                return

            def _open_from_pick():
                sdict = getattr(event.artist, "_ivt_series_meta", None)
                if sdict is None:
                    return
                if hasattr(event.artist, "get_offsets") and getattr(event, "ind", None):
                    show_for_index(sdict, int(event.ind[0]), reason="pick-point")
                else:
                    if mev and mev.xdata is not None:
                        show_for_index(sdict, nearest_index(sdict, mev.xdata), reason="pick-line")

            _open_once(mev, _open_from_pick)

        def on_click(event):
            if _toolbar_mode_active():
                return
            if event.button != 1:
                return
            if event.inaxes is not ax or event.xdata is None:
                return

            def _open_from_click():
                best = None
                best_dx = float("inf")
                for sdict in ax._ivt_series:
                    idx = nearest_index(sdict, event.xdata)
                    if idx is None:
                        continue
                    dx = abs(sdict["t_hours"][idx] - event.xdata)
                    if dx < best_dx:
                        best_dx = dx
                        best = (sdict, idx)
                if best is not None:
                    show_for_index(best[0], best[1], reason="click")

            _open_once(event, _open_from_click)

        if not hasattr(fig, "_spectrum_handlers_connected"):
            fig.canvas.mpl_connect("pick_event", on_pick)
            fig.canvas.mpl_connect("button_press_event", on_click)
            fig._spectrum_handlers_connected = True

    def _show_wavelength_plot(self, row, sweep, src_label: str):
        """
        New window with Wavelength (nm) vs Spectral Intensity,
        applying 1 m scaling if enabled.
        """
        import numpy as np
        wvls = np.asarray(getattr(sweep, "spectral_wavelengths", []), dtype=float)
        vals = np.asarray(list(getattr(row.capture, "spectral_result", []) or []), dtype=float)
        if wvls.size == 0 or vals.size == 0:
            return
        if self.normalize_to_1m.get():
            dist_m = float(getattr(row.coords, "lin_mm", 0.0) or 0.0) / 1000.0
            if dist_m > 0:
                vals = vals * (dist_m ** 2)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(wvls, vals, linewidth=1.6)
        t_s = getattr(row, "timestamp", None)
        yaw = getattr(row.coords, "yaw_deg", None)
        roll = getattr(row.coords, "roll_deg", None)
        title_bits = []
        if t_s is not None:
            title_bits.append(f"t={t_s:.0f}s")
        if yaw is not None and roll is not None:
            title_bits.append(f"yaw {yaw}, roll {roll}")
        if src_label:
            title_bits.append(src_label)
        ax.set_title("Spectrum — " + " | ".join(title_bits))
        units = getattr(sweep, "spectral_units", "")
        ylab = f"Spectral Intensity ({units})"
        if self.normalize_to_1m.get():
            ylab += ", scaled to 1 m"
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        self._register_fig('spec', fig)
        try:
            plt.show(block=False)
        except Exception:
            pass

    def _register_fig(self, cat: str, fig: plt.Figure):
        """Track a figure handle under a category key."""
        self._figs.setdefault(cat, []).append(fig)

    def _close_plots(self, cat: str, also: Tuple[str, ...]=()):
        """Close plot windows for one or more categories."""
        cats = (cat,) + tuple(also)
        for c in cats:
            figs = self._figs.get(c, [])
            for f in figs:
                try: plt.close(f)
                except Exception: pass
            self._figs[c] = []
        self._set_status(f"Closed {', '.join(cats)} plot windows.")

    def _select_all_groups(self):
        """Select all groups in the decay overlay listbox."""
        self.lb_groups_select.selection_set(0, tk.END)

    def on_save_last_figure(self):
        """Save the most recently created figure (any category)."""
        for cat in ("ivt","ovp","gdo","corr"):
            if self._figs.get(cat):
                last_fig = self._figs[cat][-1]
                path = filedialog.asksaveasfilename(title="Save Figure", defaultextension=".png",
                                                    filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")])
                if not path: return
                last_fig.savefig(path, dpi=150, bbox_inches="tight")
                self._set_status(f"Saved figure to {path}")
                return
        messagebox.showinfo("Save Figure", "No figure to save yet.")

    def _set_status(self, text: str):
        """Set the status bar text."""
        self.status_var.set(text)
        try:
            self.update_idletasks()
        except Exception:
            pass

    def _is_any_background_busy(self) -> bool:
        """Return True when any background preprocessing worker is active."""
        a_busy = self._analysis_thread is not None and self._analysis_thread.is_alive()
        s_busy = self._sw3_preprocess_thread is not None and self._sw3_preprocess_thread.is_alive()
        r_busy = self._reload_thread is not None and self._reload_thread.is_alive()
        return bool(a_busy or s_busy or r_busy)

    def _set_busy(self, busy: bool, text: str = "Processing..."):
        """Toggle a subtle busy indicator in the status bar."""
        busy = bool(busy)
        if busy:
            busy_text = f"Background: {text}"
            self._busy_var.set(busy_text)
            self._set_status(busy_text)
            if not self._busy_active:
                self._busy_pb.pack(side=tk.RIGHT, padx=(0, 6), pady=4)
                self._busy_pb.start(12)
                self._busy_active = True
            return
        self._busy_var.set("Background: idle")
        if self._busy_active:
            self._busy_pb.stop()
            self._busy_pb.pack_forget()
            self._busy_active = False

    def _next_file_id(self) -> str:
        """Return the next available file id (F###)."""
        i=1
        while f"F{i:03d}" in self.files: i+=1
        return f"F{i:03d}"

    def _next_group_id(self) -> str:
        """Return the next available group id (G###)."""
        i=1
        while f"G{i:03d}" in self.groups: i+=1
        return f"G{i:03d}"

# ------------------------------------------------------------------
# Association dialog (friendly labels; shows full mapping)
# ------------------------------------------------------------------

class AssociationDialog(tk.Toplevel):
    """Dialog for mapping SW3 files to Power files within a group."""
    def __init__(self, master: App, group: GroupRecord, sw3_list, power_list):
        """Build the associations editor UI."""
        super().__init__(master)
        self.title(f"Associations for '{group.name}'"); self.resizable(True, True)
        self.group = group; self.updated = False
        self.master = master

        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL); pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        left = ttk.Frame(pane); pane.add(left, weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=1)

        ttk.Label(left, text="SW3 files").pack(anchor="w")
        self.lb_sw3 = tk.Listbox(left, selectmode=tk.SINGLE, exportselection=False)
        for fid, fr in sw3_list: self.lb_sw3.insert(tk.END, f"{fr.label} ({fid})")
        self.sw3_ids = [fid for fid,_ in sw3_list]; self.lb_sw3.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        ttk.Label(right, text="Power files").pack(anchor="w")
        self.lb_pow = tk.Listbox(right, selectmode=tk.MULTIPLE, exportselection=False)
        for fid, fr in power_list: self.lb_pow.insert(tk.END, f"{fr.label} ({fid})")
        self.pow_ids = [fid for fid,_ in power_list]; self.lb_pow.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        btns = ttk.Frame(self); btns.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(btns, text="Map selected SW3 → selected Power", command=self._map_selected).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Remove mapping", command=self._remove_mapping).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=3)

        self.tv_map = ttk.Treeview(self, columns=("sw3", "powers"), show="headings", height=8)
        self.tv_map.heading("sw3", text="SW3"); self.tv_map.heading("powers", text="Power")
        self.tv_map.column("sw3", stretch=True, width=260); self.tv_map.column("powers", stretch=True, width=480)
        self.tv_map.pack(fill=tk.BOTH, expand=True, padx=6, pady=6); self._refresh_mapping()
        self.lb_sw3.bind("<<ListboxSelect>>", lambda e: self._refresh_mapping())

    def _map_selected(self):
        """Associate the selected SW3 file with selected power files."""
        idx_sw = self.lb_sw3.curselection(); idx_pw = self.lb_pow.curselection()
        if not idx_sw or not idx_pw: return
        sw_id = self.sw3_ids[idx_sw[0]]; pow_ids = [self.pow_ids[i] for i in idx_pw]
        s = self.group.associations.get(sw_id, set()); s |= set(pow_ids)
        self.group.associations[sw_id] = s; self.updated = True; self._refresh_mapping()

    def _remove_mapping(self):
        """Remove associations for the selected SW3 file."""
        idx_sw = self.lb_sw3.curselection(); 
        if not idx_sw: return
        sw_id = self.sw3_ids[idx_sw[0]]; idx_pw = self.lb_pow.curselection()
        if not idx_pw: self.group.associations.pop(sw_id, None)
        else:
            s = self.group.associations.get(sw_id, set())
            for i in idx_pw: s.discard(self.pow_ids[i])
            if s: self.group.associations[sw_id] = s
            else: self.group.associations.pop(sw_id, None)
        self.updated = True; self._refresh_mapping()

    def _refresh_mapping(self):
        """Refresh the visible mapping table."""
        for item in self.tv_map.get_children(): self.tv_map.delete(item)
        # Always list full mapping with friendly labels
        for sw_id, pset in sorted(self.group.associations.items()):
            try: sw_label = self.master.files[sw_id].label
            except Exception: sw_label = sw_id
            power_labels = []
            for pid in sorted(list(pset)):
                try: power_labels.append(self.master.files[pid].label)
                except Exception: power_labels.append(pid)
            self.tv_map.insert("", "end", values=(sw_label, ", ".join(power_labels)))

# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

def main():
    """Launch the GUI application."""
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
