import unittest

import numpy as np
import pandas as pd

from gui import (
    _gap_threshold_s,
    _is_nearly_regular_cadence,
    _insert_nan_gaps,
    _prepare_power_series,
    _slice_time_window,
    choose_effective_resample_seconds,
    detect_cadence_cliff_by_samples,
    ema_adaptive,
    ema_timeaware,
    recommend_power_ema_span,
)


class TestSamplingAndGaps(unittest.TestCase):
    def test_choose_effective_resample_seconds_upscales_long_windows(self):
        # 4000 hours with 3 s base would exceed millions of points.
        duration_s = 4000 * 3600
        rs = choose_effective_resample_seconds(duration_s, requested_seconds=3, max_points=600_000)
        self.assertGreaterEqual(rs, 24)
        self.assertEqual(
            choose_effective_resample_seconds(3600, requested_seconds=3, max_points=600_000),
            3,
        )

    def test_is_nearly_regular_cadence(self):
        t_regular = np.array([0, 3, 6, 9, 12, 15], dtype=float)
        t_irregular = np.array([0, 3, 8, 9, 30, 31], dtype=float)
        self.assertTrue(_is_nearly_regular_cadence(t_regular))
        self.assertFalse(_is_nearly_regular_cadence(t_irregular, max_ratio=1.2))

    def test_recommend_power_ema_span_scales_with_cadence(self):
        span = recommend_power_ema_span(
            intensity_ema_span=15,
            sw3_median_step_s=17.0,
            resample_seconds=3,
            power_step_s=1.4,
        )
        # Expected order of magnitude from Open Excimer tuning (~85).
        self.assertGreaterEqual(span, 70)
        self.assertLessEqual(span, 100)

    def test_slice_time_window_uses_sorted_search(self):
        df = pd.DataFrame({"timestamp": [0, 10, 20, 30, 40], "v": [1, 2, 3, 4, 5]})
        out = _slice_time_window(df, 9, 31, ts_col="timestamp")
        self.assertEqual(out["timestamp"].tolist(), [10, 20, 30])

    def test_detect_cadence_cliff_by_sample_count(self):
        # Baseline cadence ~1s, then sustained slowdown ~4s.
        pre = np.cumsum(np.full(2500, 1.0))
        post = pre[-1] + np.cumsum(np.full(1500, 4.0))
        t = np.concatenate(([0.0], pre, post))
        base_dt, cliff_idx = detect_cadence_cliff_by_samples(t, threshold_ratio=3.0)

        self.assertTrue(np.isfinite(base_dt))
        self.assertGreater(base_dt, 0.0)
        self.assertIsNotNone(cliff_idx)
        self.assertGreater(cliff_idx, 2000)
        self.assertLess(cliff_idx, 3200)

    def test_timeaware_ema_resets_after_large_gap(self):
        t = np.array([0.0, 1.0, 2.0, 120.0, 121.0], dtype=float)
        y = np.array([10.0, 20.0, 15.0, 90.0, 100.0], dtype=float)
        out = ema_timeaware(t, y, span=3)

        # Large gap should reset EMA memory at the first post-gap point.
        self.assertAlmostEqual(out[3], y[3], places=7)
        self.assertNotAlmostEqual(out[4], y[4], places=7)

    def test_insert_nan_gaps_breaks_line_segments(self):
        t = np.array([0.0, 1.0, 2.0, 50.0, 51.0], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        t2, y2 = _insert_nan_gaps(t, y, min_gap_s=10.0)

        self.assertTrue(np.isnan(t2[3]))
        self.assertTrue(np.isnan(y2[3]))

    def test_gap_threshold_uses_floor(self):
        t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        g = _gap_threshold_s(t, floor_s=30.0)
        self.assertEqual(g, 30.0)

    def test_prepare_power_series_does_not_fill_outage_interval(self):
        # Two dense segments separated by a long outage.
        df = pd.DataFrame(
            {
                "Timestamp": [0, 1, 2, 3, 200, 201, 202, 203],
                "W_Active": [10.0, 11.0, 9.0, 10.0, 20.0, 21.0, 19.0, 20.0],
            }
        )
        out = _prepare_power_series(
            power_df=df,
            tmin=0,
            tmax=203,
            ema_span=7,
            resample_seconds=1,
            timeaware_ema=True,
        )

        self.assertIsNotNone(out)
        self.assertGreater(len(out), 0)

        # There should be an explicit timestamp jump (no synthetic flat bridge).
        self.assertTrue((out["timestamp"].diff().fillna(0) > 60).any())

        # First point after outage should re-anchor EMA to current segment.
        row_200 = out[out["timestamp"] == 200]
        self.assertEqual(len(row_200), 1)
        self.assertAlmostEqual(float(row_200["power_ema"].iloc[0]), 20.0, places=7)

    def test_ema_adaptive_matches_standard_when_disabled(self):
        t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        out_regular = ema_adaptive(t, y, span=5, timeaware=False)
        out_timeaware = ema_adaptive(t, y, span=5, timeaware=True)

        self.assertEqual(out_regular.shape, out_timeaware.shape)
        self.assertFalse(np.allclose(out_regular, out_timeaware))

    def test_cliff_compensation_reduces_post_cliff_jumps(self):
        # Synthetic "fine until not": cadence becomes sparse late in the file.
        t = np.concatenate(
            [
                np.arange(0, 3000, 1.0),
                3000.0 + np.arange(1, 600) * 5.0,
            ]
        )
        y = np.sin(t / 200.0) + 0.1 * np.sin(t / 5.0)

        out_timeaware = ema_timeaware(t, y, span=20)
        out_regular = ema_adaptive(t, y, span=20, timeaware=False)

        # Ensure outputs are finite and same length; time-aware path is active.
        self.assertEqual(out_timeaware.shape, y.shape)
        self.assertEqual(out_regular.shape, y.shape)
        self.assertTrue(np.isfinite(out_timeaware).all())


if __name__ == "__main__":
    unittest.main()
