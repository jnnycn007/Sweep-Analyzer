"""eegbin file format reader/writer and scan utilities.

This module provides the in-memory models used by SW3 scans along with helpers
for loading legacy eegbin v1/v2 and current eegbin v3 (version 6) files.

Units
-----
- `integral_result`: W/m^2
- `spectral_result`: W/m^2/nm (per wavelength bin)
- `spectral_wavelengths`: nm
- `timestamp`: epoch seconds
- `lin_mm`: measurement distance in millimeters
"""

from __future__ import annotations

import array
import base64
import gzip
import json
import math
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from util import inclusive_range


class Flags(Enum):
    """Bit flags attached to each goniometer row."""

    IGNORE = 1 << 0
    NO_SPECTRAL = 1 << 1
    DIODE_CORR_MODE_0_AZ = 1 << 2
    DIODE_CORR_MODE_1_NONE = 1 << 3
    DIODE_CORR_MODE_2_SPECIAL = 1 << 4
    SWEEP_START_WARMUP = 1 << 6
    SWEEP_START_SCAN = 1 << 7
    SWEEP_START_CUSTOM = 1 << 8
    DISCARDED_DURING_SWEEP = 1 << 9
    HIGHRES_MODE = 1 << 10


@dataclass
class SpectrumCapture:
    """Spectrum + integral capture for a single goniometer position."""

    integral_result: Optional[float]  # W/m^2
    spectral_result: Optional[Sequence[float]]  # W/m^2/nm
    az_correction: float
    spectral_time_us: float
    integral_time_us: float
    spectral_saturation: float
    integral_saturation: float
    settings: Optional[object]

    def get_spectral_point(self, scan: "LampScan", nm: float) -> float:
        """Interpolate the spectrum at a single wavelength (nm)."""
        low = [(i, x) for i, x in enumerate(scan.spectral_wavelengths) if x <= nm][-1]
        hi = [(i, x) for i, x in enumerate(scan.spectral_wavelengths) if x >= nm][0]
        dist = hi[1] - low[1]
        if dist == 0:
            return float(self.spectral_result[low[0]])
        pos = (nm - low[1]) / dist

        low_v = float(self.spectral_result[low[0]])
        hi_v = float(self.spectral_result[hi[0]])

        return (low_v * (1 - pos)) + (hi_v * pos)


@dataclass
class Coordinates:
    """Goniometer coordinates for a measurement row."""

    yaw_deg: float
    roll_deg: float
    lin_mm: float

    def scale_to_reference(self, value: float, reference: float = 1) -> float:
        """Scale an irradiance value to a reference distance in meters."""
        return value * ((self.lin_mm / (reference * 1000)) ** 2)

    @classmethod
    def normalize_coordinates(cls, yaw: float, roll: float) -> Tuple[float, float]:
        """Normalize to IES type‑C coordinates: 0<=yaw<=360, 0<=roll<=360.

        Negative yaw is reflected, and roll is rotated by 180 degrees.
        """
        if yaw < 0:
            yaw = -yaw
            roll = roll + 180

        roll = roll % 360
        return yaw, roll

    def get_normal_yr(self) -> Tuple[float, float]:
        """Return normalized (yaw, roll) in the IES type‑C convention."""
        return self.normalize_coordinates(self.yaw_deg, self.roll_deg)


@dataclass
class GoniometerRow:
    """Single sample: coordinates, capture data, and metadata."""

    coords: Coordinates
    capture: SpectrumCapture
    timestamp: float
    flags: int
    notes: Optional[str]
    diag_notes: Optional[object]

    @property
    def valid(self) -> bool:
        """True when the row is not ignored/discarded by sweep flags."""
        return (not (self.flags & Flags.IGNORE.value)) and (
            not (self.flags & Flags.DISCARDED_DURING_SWEEP.value)
        )

    def compute_reference_integral(
        self,
        scan: "LampScan",
        wvl_range: Tuple[float, float],
        reference_dist: float,
        force_integral: bool = False,
    ) -> float:
        """Return the integrated irradiance scaled to a reference distance.

        Uses the integral channel when `force_integral` is True or spectral data
        is missing; otherwise integrates spectral data across `wvl_range`.
        """
        if force_integral or (self.flags & Flags.NO_SPECTRAL.value) or (not self.capture.spectral_result):
            integral = self.capture.integral_result
            if not isinstance(integral, (int, float)) or not math.isfinite(integral):
                return 0.0
            return self.coords.scale_to_reference(float(integral), reference_dist)
        return self.coords.scale_to_reference(
            scan.integrate_spectral(self.capture.spectral_result, *wvl_range),
            reference_dist,
        )


class MajorAxis(Enum):
    """Major axis for a sweep phase."""

    NA = 0
    LINEAR = 1
    ROLL = 2  # 'normal'
    YAW = 3


class PhaseType(Enum):
    """Phase categories found in a scan."""

    LINEAR_PULLBACK = 1
    SPECTRAL_WEB = 2
    INTEGRAL_WEB = 3
    WARMUP_WAIT = 4
    SPECTRUM_POINT = 5

    @property
    def is_web(self) -> bool:
        """Return True if the phase is a spectral or integral web."""
        return self == self.SPECTRAL_WEB or self == self.INTEGRAL_WEB


@dataclass
class GoniometerPhase:
    """Grouping of rows that belong to a scan phase."""

    major_axis: MajorAxis
    phase_type: PhaseType
    name: str
    members: List[GoniometerRow]


@dataclass
class LampScan:
    """Full scan with metadata, wavelengths, and all rows."""

    lamp_name: str
    notes: str
    lamp_desc: object
    lamp_desc_filename: str

    spectral_wavelengths: List[float]

    spectral_units: str
    integral_units: str
    rows: List[GoniometerRow]
    phases: List[GoniometerPhase]

    _from_path: Optional[str] = None
    _spectral_integral_cache: Optional[Dict[Tuple[int, Optional[float], Optional[float], object], float]] = None
    _wvl_index_cache: Optional[Dict[float, int]] = None
    _bin_width_for_wvl_cache: Optional[Dict[float, float]] = None

    def __post_init__(self):
        """Initialize internal caches after dataclass construction."""
        self._spectral_integral_cache = {}
        self._wvl_index_cache = {}
        self._bin_width_for_wvl_cache = {}

    @classmethod
    def make_spectral_wavelengths_from_legacy(
        cls, wvl_start: float, wvl_stop: float, wvl_step: float
    ) -> List[float]:
        """Rebuild wavelength bins used by legacy files."""
        out: List[float] = []
        n = wvl_start
        while n <= wvl_stop:
            out.append(n)
            n += wvl_step
        return out

    def index_wvl(self, w: float) -> int:
        """Return index of the first wavelength >= w (or last bin)."""
        if w in self._wvl_index_cache:
            return self._wvl_index_cache[w]
        idx = len(self.spectral_wavelengths) - 1
        for i, wvl in enumerate(self.spectral_wavelengths):
            if wvl >= w:
                idx = i
                break
        self._wvl_index_cache[w] = idx
        return idx

    def get_bin_width(self, bin_num: int) -> float:
        """Return width of a spectral bin (nm).

        The last bin has zero width in this legacy convention.
        """
        if bin_num >= len(self.spectral_wavelengths) - 1:
            return 0.0
        return self.spectral_wavelengths[bin_num + 1] - self.spectral_wavelengths[bin_num]

    def get_bin_width_for_wvl(self, wvl: float) -> float:
        """Return cached bin width for a specific wavelength value."""
        if wvl in self._bin_width_for_wvl_cache:
            return self._bin_width_for_wvl_cache[wvl]
        v = self.get_bin_width(self.spectral_wavelengths.index(wvl))
        self._bin_width_for_wvl_cache[wvl] = v
        return v

    def integrate_spectral(
        self,
        spectral: Optional[Sequence[float]],
        start: Optional[float] = None,
        end: Optional[float] = None,
        weighting: Optional[Callable[[float], float]] = None,
    ) -> float:
        """Integrate a spectrum across a wavelength band.

        Parameters
        ----------
        spectral: sequence of spectral values (W/m^2/nm)
        start, end: wavelength limits (nm). Defaults to min/max.
        weighting: optional weighting function applied per wavelength.
        """
        if spectral is None:
            return 0.0

        if isinstance(spectral, array.array):
            spectral_key = id(spectral)
        else:
            spectral_key = hash(tuple(spectral))
        cachekey = (spectral_key, start, end, weighting)
        cached = self._spectral_integral_cache.get(cachekey)
        if cached is not None:
            return cached

        if start is None:
            start = min(self.spectral_wavelengths)
        if end is None:
            end = max(self.spectral_wavelengths)

        istart = self.index_wvl(start)
        iend = self.index_wvl(end) or 1

        total = 0.0
        for wvl, value in zip(self.spectral_wavelengths[istart:iend], spectral[istart:iend]):
            v = float(value) * self.get_bin_width_for_wvl(wvl)
            if weighting:
                v *= weighting(wvl)
            total += v

        self._spectral_integral_cache[cachekey] = total
        return total

    def get_best_value_in_band(
        self, row: GoniometerRow, start: Optional[float] = None, end: Optional[float] = None
    ) -> float:
        """Return spectral integral if present; otherwise the integral channel."""
        if row.capture.spectral_result and len(row.capture.spectral_result) > 3:
            return self.integrate_spectral(row.capture.spectral_result, start, end)
        if row.capture.integral_result is not None:
            return float(row.capture.integral_result)
        return 0.0

    def get_point_yr(self, yaw: float, roll: float, need: bool = True) -> Optional[GoniometerRow]:
        """Return a row matching yaw/roll (exact), optionally raising if missing."""
        for row in self.rows:
            if not row.valid:
                continue
            if row.coords.yaw_deg == yaw and row.coords.roll_deg == roll:
                return row

        if need:
            raise IndexError("No Row")
        return None

    def get_point_yr_norm_all(
        self, yaw: float, roll: float, filter_valid: bool = True
    ) -> List[GoniometerRow]:
        """Return all rows matching normalized yaw/roll coordinates."""
        yaw_n, roll_n = Coordinates.normalize_coordinates(yaw, roll)
        res = []
        for row in self.rows:
            if filter_valid and not row.valid:
                continue

            ry, rr = row.coords.get_normal_yr()

            if ry == yaw_n and rr == roll_n:
                res.append(row)

            if row.coords.yaw_deg == yaw_n and row.coords.roll_deg == roll_n:
                res.append(row)

            if ry == 0 and yaw_n == 0:
                res.append(row)  # assume rotational symmetry for 0

        return res

    def get_point_yr_norm(self, yaw: float, roll: float, if_missing: str = "next_roll") -> GoniometerRow:
        """Return the first normalized match; optionally fall back to next roll."""
        res = self.get_point_yr_norm_all(yaw, roll)

        if res:
            return res[0]

        if if_missing == "next_roll":
            print(
                f"WARN: {self.lamp_name}: Missing point: ({yaw}, {roll}) -- taking next assuming rotational symmetry"
            )
            return self.get_point_yr_norm(yaw, roll + 22.5, "die")
        raise IndexError(f"{self.lamp_name}: Failed to get_point_yr_norm({yaw}, {roll})")

    def get_point_ae(self, az: float, el: float, need: bool = True) -> Optional[GoniometerRow]:
        """Return a row by azimuth/elevation coordinate (legacy mapping)."""
        for row in self.rows:
            if not row.valid:
                continue

            if row.coords.roll_deg == 0 and row.coords.yaw_deg == az and el == 0:
                return row
            if row.coords.roll_deg == 90 and row.coords.yaw_deg == el and az == 0:
                return row
            if row.coords.yaw_deg == 0 and az == 0 and el == 0:
                return row

        if need:
            raise IndexError("No Row")
        return None

    def get_rolls(self) -> set[float]:
        """Return the set of roll angles in the scan."""
        return set(x.coords.roll_deg for x in self.rows)

    def get_contiguous_arcs(
        self, roll: float, normalize: bool = False, ignore_ignores: bool = True
    ) -> List[List[GoniometerRow]]:
        """Return contiguous arcs of rows at a given roll angle."""
        if normalize:
            roll = Coordinates.normalize_coordinates(0, roll)[1]
        arcs: List[List[GoniometerRow]] = []
        in_arc = False
        still_in_dT: Optional[bool] = None
        for row in self.rows:
            if row.flags & Flags.SWEEP_START_WARMUP.value:
                still_in_dT = True
            if row.flags & Flags.SWEEP_START_SCAN.value:
                still_in_dT = False

            if still_in_dT is not None and still_in_dT:
                continue

            if ignore_ignores and (row.flags & Flags.IGNORE.value):
                continue

            r = row.coords.roll_deg
            if normalize:
                r = Coordinates.normalize_coordinates(row.coords.yaw_deg, r)[1]

            if r == roll:
                if not in_arc:
                    arcs.append([])
                    in_arc = True
                arcs[-1].append(row)
            else:
                in_arc = False
        return arcs

    def get_webspec(self) -> Tuple[List[float], List[float]]:
        """Return yaw/roll grids for a spectral web (0..90 / 0..360)."""
        yaws = list(set(row.coords.yaw_deg for row in self.rows))
        yaws.sort()
        rolls = list(set(row.coords.roll_deg for row in self.rows))
        rolls.sort()

        yawstep = yaws[1] - yaws[0]
        rollstep = rolls[1] - rolls[0]
        yaws = inclusive_range(0, 90, yawstep)
        rolls = inclusive_range(0, 360, rollstep)

        return (yaws, rolls)


# ---------------------------------------------------------------------------
# Legacy EEG bin v1 (binary row-based)
# ---------------------------------------------------------------------------

def load_legacy_eegbin1(buffer: bytes, name: str, distance_mm: float, new_axes: bool) -> LampScan:
    """Load an older eegbin1 format buffer into a LampScan."""
    header = "=dddidddd"
    header_sz = struct.calcsize(header)

    def read(n: int) -> bytes:
        nonlocal buffer
        if len(buffer) < n:
            return b""
        r = buffer[:n]
        buffer = buffer[n:]
        return r

    expected_ws: Optional[Tuple[float, float, float]] = None

    rows: List[GoniometerRow] = []
    while True:
        buf = read(header_sz)
        if len(buf) != header_sz:
            break
        az, el, li, length, wstart, wstop, wstep, integral_result = struct.unpack(header, buf)
        if expected_ws is None:
            expected_ws = (wstart, wstop, wstep)
        else:
            assert (wstart, wstop, wstep) == expected_ws, "Nonconstant spectral wavelength parameters"
        l = length * 8
        buf = read(length * 8)
        if len(buf) != l:
            break
        a = array.array("d", buf)

        if new_axes:
            yaw = az
            roll = el
        else:
            if az == 0:
                roll = 90
                yaw = el
            elif el == 0:
                roll = 0
                yaw = az
            else:
                assert False, "Non axial point"

        rows.append(
            GoniometerRow(
                Coordinates(yaw, roll, distance_mm),
                SpectrumCapture(integral_result, a, -1, -1, -1, -1, -1, None),
                0,
                0,
                None,
                None,
            )
        )

    return LampScan(
        lamp_name=name,
        notes="Imported from legacy eegbin1 file",
        lamp_desc={},
        lamp_desc_filename="?",
        spectral_wavelengths=LampScan.make_spectral_wavelengths_from_legacy(wstart, wstop, wstep),
        spectral_units="W/m2/nm",
        integral_units="W/m2",
        rows=rows,
        phases=[],
    )


row_fmt_3 = "<dddddii"
row_fmt_4 = "<dddddidddi"
sep = b"\n\nB64 FOLLOWS\n\n"

# yaw, roll, lin, timestamp, integral, spec_settings key (unused), az correction,
# spec time, integ time, flags

def save_eegbin2(scan: LampScan) -> bytes:
    """Serialize a scan to eegbin v2 bytes (metadata JSON + base64 rows)."""
    buffer = b""

    metadata = {
        "version": 5,
        "scan": {k: v for k, v in scan.__dict__.items() if k not in ["rows", "phases"] and not k.startswith("_")},
        "row_notes": {str(i): k.notes for i, k in enumerate(scan.rows) if k.notes},
    }
    buffer += json.dumps(metadata, indent=4).encode("utf-8")
    buffer += sep

    rows_buf = b""
    for row in scan.rows:
        rows_buf += struct.pack(
            row_fmt_4,
            row.coords.yaw_deg,
            row.coords.roll_deg,
            row.coords.lin_mm,
            row.timestamp,
            row.capture.integral_result,
            0,
            row.capture.az_correction,
            row.capture.spectral_time_us,
            row.capture.integral_time_us,
            row.flags,
        )

        if not (row.flags & Flags.NO_SPECTRAL.value):
            rows_buf += bytes(array.array("d", row.capture.spectral_result))

    buffer += base64.standard_b64encode(rows_buf)
    return buffer


def load_eegbin2(buffer: bytes, from_path: Optional[str] = None) -> LampScan:
    """Deserialize eegbin v2 bytes into a LampScan."""
    metadata, rows_buf = buffer.split(sep, 1)
    metadata = json.loads(metadata)
    assert metadata["version"] in [3, 4, 5], "Unknown version"
    if metadata["version"] < 5 and "spectral_wavelengths" not in metadata["scan"]:
        metadata["scan"]["spectral_wavelengths"] = LampScan.make_spectral_wavelengths_from_legacy(
            metadata["scan"]["wvl_start"], metadata["scan"]["wvl_stop"], metadata["scan"]["wvl_step"]
        )
        del metadata["scan"]["wvl_start"]
        del metadata["scan"]["wvl_stop"]
        del metadata["scan"]["wvl_step"]

    rows_buf = base64.standard_b64decode(rows_buf)

    scan = LampScan(**metadata["scan"], rows=[], phases=[], _from_path=from_path)

    row_points_len = 8 * len(scan.spectral_wavelengths)

    rows_buf_ptr = 0

    def take(n: int) -> bytes:
        nonlocal rows_buf, rows_buf_ptr
        r = rows_buf[rows_buf_ptr : rows_buf_ptr + n]
        rows_buf_ptr += n
        return r

    i = 0
    while True:
        if rows_buf_ptr == len(rows_buf):
            break

        try:
            if metadata["version"] == 3:
                yaw_deg, roll_deg, lin_mm, timestamp, integral_result, _, flags = struct.unpack(
                    row_fmt_3, take(struct.calcsize(row_fmt_3))
                )
                az_correction, spectral_time_us, integral_time_us = -1, -1, -1
            else:
                yaw_deg, roll_deg, lin_mm, timestamp, integral_result, _, az_correction, spectral_time_us, integral_time_us, flags = struct.unpack(
                    row_fmt_4, take(struct.calcsize(row_fmt_4))
                )

            if not (flags & Flags.NO_SPECTRAL.value):
                points = array.array("d", take(row_points_len))
            else:
                points = None

            notes = metadata["row_notes"].get(str(i))
            scan.rows.append(
                GoniometerRow(
                    Coordinates(yaw_deg, roll_deg, lin_mm),
                    SpectrumCapture(
                        integral_result, points, az_correction, spectral_time_us, integral_time_us, -1, -1, None
                    ),
                    timestamp,
                    flags,
                    notes,
                    None,
                )
            )
            i += 1
            if (i % 1000) == 0:
                sys.stderr.write(f"Loaded {i} rows...\n")
        except struct.error:
            sys.stderr.write("E: Failed to unpack as expected\n")
            break

    return scan


eegbin3_magic = "EEGBIN3\n".encode("ascii")
eegbin3_sep = "\n\nGZIP FOLLOWS\n\n".encode("ascii")


def save_eegbin3(scan: LampScan, fd) -> bytes:
    """Serialize a scan to eegbin v3 (version 6) to a file-like object."""

    def row_to_json(row: GoniometerRow) -> Dict[str, object]:
        return {
            "yaw_deg": row.coords.yaw_deg,
            "roll_deg": row.coords.roll_deg,
            "lin_mm": row.coords.lin_mm,
            "timestamp": row.timestamp,
            "integral_result": row.capture.integral_result,
            "az_correction": row.capture.az_correction,
            "spectral_time_us": row.capture.spectral_time_us,
            "integral_time_us": row.capture.integral_time_us,
            "spectral_saturation": row.capture.spectral_saturation,
            "integral_saturation": row.capture.integral_saturation,
            "notes": row.notes,
            "diag_notes": row.diag_notes,
            "flags": [x.name for x in Flags if (row.flags & x.value)],
        }

    def phase_to_json(phase: GoniometerPhase) -> Dict[str, object]:
        return {
            "major_axis": phase.major_axis.name,
            "phase_type": phase.phase_type.name,
            "name": phase.name,
            "members": [scan.rows.index(r) for r in phase.members],
        }

    print("Building metadata...")

    buffer = eegbin3_magic
    fd.write(buffer)

    metadata = {
        "version": 6,
        "scan": {k: v for k, v in scan.__dict__.items() if k not in ["rows", "phases"] and not k.startswith("_")},
        "rows": [row_to_json(r) for r in scan.rows],
        "phases": [phase_to_json(a) for a in scan.phases],
        "is_gzipped": False,
    }

    print("json.dumps...")

    buffer = json.dumps(metadata, indent=None).encode("utf-8")

    fd.write(buffer)
    fd.write(eegbin3_sep)

    print("Building array data...")

    for row in scan.rows:
        if not (row.flags & Flags.NO_SPECTRAL.value):
            fd.write(bytes(array.array("d", row.capture.spectral_result)))

    print("Done.")
    return buffer


def load_eegbin3(buffer: bytes, from_path: Optional[str] = None) -> LampScan:
    """Deserialize eegbin v3 (version 6) bytes into a LampScan."""
    assert buffer.startswith(eegbin3_magic), "Bad magic"
    buffer = buffer[len(eegbin3_magic) :]
    metadata, rows_buf = buffer.split(eegbin3_sep, 1)
    metadata = json.loads(metadata)
    assert metadata["version"] == 6, "Bad version"

    if metadata.get("is_gzipped", True):
        rows_buf = gzip.decompress(rows_buf)

    scan = LampScan(**metadata["scan"], rows=[], phases=[], _from_path=from_path)

    row_points_len = 8 * len(scan.spectral_wavelengths)

    rows_buf_ptr = 0

    def take(n: int) -> bytes:
        nonlocal rows_buf, rows_buf_ptr
        r = rows_buf[rows_buf_ptr : rows_buf_ptr + n]
        rows_buf_ptr += n
        return r

    for row in metadata["rows"]:
        flags = sum(Flags[x].value for x in row["flags"])

        points = None
        if not (flags & Flags.NO_SPECTRAL.value):
            points = array.array("d", take(row_points_len))

        scan.rows.append(
            GoniometerRow(
                Coordinates(row["yaw_deg"], row["roll_deg"], row["lin_mm"]),
                SpectrumCapture(
                    row["integral_result"],
                    points,
                    row["az_correction"],
                    row["spectral_time_us"],
                    row["integral_time_us"],
                    row.get("spectral_saturation", -1),
                    row.get("integral_saturation", -1),
                    None,
                ),
                row["timestamp"],
                flags,
                row["notes"],
                row.get("diag_notes", None),
            )
        )

    for phase in metadata["phases"]:
        scan.phases.append(
            GoniometerPhase(
                MajorAxis[phase["major_axis"]],
                PhaseType[phase["phase_type"]],
                phase["name"],
                [scan.rows[i] for i in phase["members"]],
            )
        )

    return scan
