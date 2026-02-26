import argparse
import csv
import json
import math
import re
import struct
import sys
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog

C = 299792458.0
MU_SUN = 1.3271244e20  # m^3/s^2
EPOCH_1950 = datetime(1950, 1, 1, tzinfo=timezone.utc)
ODF_REC_BYTES = 36
TDF_REC_BYTES = 288

TDF5_DOPPLER_PR_SIGN_BITS_START_BIT = 893  # within radiometric block (Item 73)
TDF5_DOPPLER_PR_SIGN_BITS_BITS = 4
TDF5_DOPPLER_PR_VALUE_START_BIT = 897  # within radiometric block (Item 74)
TDF5_DOPPLER_PR_VALUE_BITS = 32

# Nominal carrier frequencies for converting Doppler residual [Hz] -> fractional frequency y [-].
# (Exact carrier depends on the tracking setup; we keep these as documented, configurable constants.)
CARRIER_HZ_BY_DOWNLINK_BAND_ID = {
    1: 2.295e9,  # S-band (approx)
    2: 8.4e9,  # X-band (approx)
    3: 32.0e9,  # Ka-band (approx)
}


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


@dataclass(frozen=True)
class ModelRow:
    t: datetime
    r1_m: float
    r2_m: float
    b_m: float
    bdot_mps: float
    r1dot_mps: float
    r2dot_mps: float


@dataclass(frozen=True)
class Point:
    t_days: float
    y_obs: float
    y_model: float

    @property
    def residual(self) -> float:
        return self.y_obs - self.y_model


@dataclass(frozen=True)
class OdfObs:
    time_utc: datetime
    doppler_hz: float
    y: float
    station_rx: int
    station_tx: int
    data_type_id: int
    downlink_band_id: int
    uplink_band_id: int
    source_file: str


def _repo_root() -> Path:
    return _ROOT


def _try_load_frozen_beta(root: Path) -> Tuple[Optional[float], str]:
    path = root / "output" / "theory" / "frozen_parameters.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None, "output/theory/frozen_parameters.json (missing)"

    try:
        j = json.loads(path.read_text(encoding="utf-8"))
        beta = float(j["beta"])
        return beta, "output/theory/frozen_parameters.json:beta"
    except Exception:
        return None, "output/theory/frozen_parameters.json:beta (read failed)"


def _extract_bits(v: int, total_bits: int, start_bit: int, bits: int) -> int:
    # PDS3 MSB_BIT_STRING: START_BIT is 1-based from MSB.
    shift = total_bits - (start_bit + bits - 1)
    # 条件分岐: `shift < 0` を満たす経路を評価する。
    if shift < 0:
        raise ValueError("Invalid bit range")

    return (v >> shift) & ((1 << bits) - 1)


def _signed36_from_sign4_low32(sign4: int, low32: int) -> int:
    # Assemble a 36-bit two's complement signed integer from a 4-bit sign extension and 32-bit payload.
    # This matches the "SIGN BITS ... (artifact of 36-bit binary definitions)" pattern used in DSN formats.
    v = ((int(sign4) & 0xF) << 32) | (int(low32) & 0xFFFFFFFF)
    # 条件分岐: `v & (1 << 35)` を満たす経路を評価する。
    if v & (1 << 35):
        v -= 1 << 36

    return int(v)


def _bin_time_series_mean(
    pairs: Sequence[Tuple[float, float]],
    *,
    bin_seconds: int,
) -> List[Dict[str, float]]:
    # 条件分岐: `bin_seconds <= 0` を満たす経路を評価する。
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be > 0")

    # Integer binning on seconds from t_ref (t_days is already relative to b_min time).

    acc: Dict[int, Tuple[float, int]] = {}
    for t_days, y in pairs:
        # 条件分岐: `not math.isfinite(t_days) or not math.isfinite(y)` を満たす経路を評価する。
        if not math.isfinite(t_days) or not math.isfinite(y):
            continue

        sec = int(round(t_days * 86400.0))
        key = sec // int(bin_seconds)
        s, n = acc.get(key, (0.0, 0))
        acc[key] = (s + float(y), n + 1)

    out: List[Dict[str, float]] = []
    for key in sorted(acc):
        s, n = acc[key]
        # 条件分岐: `n <= 0` を満たす経路を評価する。
        if n <= 0:
            continue

        t_days_bin = (key * int(bin_seconds)) / 86400.0
        out.append({"t_days": float(t_days_bin), "y_mean": float(s / n), "n": float(n)})

    return out


def _bin_time_series(
    pairs: Sequence[Tuple[float, float]],
    *,
    bin_seconds: int,
    stat: str,
) -> List[Dict[str, float]]:
    stat_norm = str(stat).strip().lower()
    # 条件分岐: `stat_norm in ("mean", "avg", "average")` を満たす経路を評価する。
    if stat_norm in ("mean", "avg", "average"):
        return _bin_time_series_mean(pairs, bin_seconds=bin_seconds)

    # 条件分岐: `stat_norm not in ("median",)` を満たす経路を評価する。

    if stat_norm not in ("median",):
        raise ValueError(f"Unknown bin stat: {stat} (expected: mean|median)")

    # Median is more robust to outliers; bins are small (order 10^2-10^3),
    # so collecting per-bin values is acceptable here.

    import statistics

    # 条件分岐: `bin_seconds <= 0` を満たす経路を評価する。
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be > 0")

    acc: Dict[int, List[float]] = {}
    for t_days, y in pairs:
        # 条件分岐: `not math.isfinite(t_days) or not math.isfinite(y)` を満たす経路を評価する。
        if not math.isfinite(t_days) or not math.isfinite(y):
            continue

        sec = int(round(t_days * 86400.0))
        key = sec // int(bin_seconds)
        acc.setdefault(key, []).append(float(y))

    out: List[Dict[str, float]] = []
    for key in sorted(acc):
        ys = acc[key]
        # 条件分岐: `not ys` を満たす経路を評価する。
        if not ys:
            continue

        t_days_bin = (key * int(bin_seconds)) / 86400.0
        out.append({"t_days": float(t_days_bin), "y_mean": float(statistics.median(ys)), "n": float(len(ys))})

    return out


def _detrend_polynomial(
    series: Sequence[Dict[str, float]],
    *,
    poly_order: int,
    exclude_inner_days: float,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    """
    Remove a low-order polynomial baseline estimated from the *outer* parts of the window.

    Motivation:
    - Raw Doppler-like observables include large kinematic/operational components.
    - Cassini γ tests focus on the small Shapiro-induced signature near conjunction.
    - We fit the baseline on |t| >= exclude_inner_days and subtract it, so the center shape remains.

    Returns (updated_series, info_dict).
    """
    # 条件分岐: `poly_order < 0` を満たす経路を評価する。
    if poly_order < 0:
        raise ValueError("poly_order must be >= 0")

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        print(f"[warn] numpy not available; skipping detrend: {e}")
        return list(series), {"detrend": "skipped", "reason": "numpy not available"}

    xs = np.array([float(r["t_days"]) for r in series], dtype=float)
    ys = np.array([float(r["y_mean"]) for r in series], dtype=float)
    mask = np.abs(xs) >= float(exclude_inner_days)
    n_fit = int(mask.sum())
    # 条件分岐: `n_fit < (poly_order + 1)` を満たす経路を評価する。
    if n_fit < (poly_order + 1):
        return list(series), {
            "detrend": "skipped",
            "reason": f"insufficient points for poly fit (need {poly_order+1}, got {n_fit})",
        }

    coef = np.polyfit(xs[mask], ys[mask], int(poly_order))
    baseline = np.polyval(coef, xs)
    resid = ys - baseline

    out: List[Dict[str, float]] = []
    for r, b, rr in zip(series, baseline.tolist(), resid.tolist()):
        out.append(
            {
                "t_days": float(r["t_days"]),
                "y_mean": float(r["y_mean"]),
                "n": float(r.get("n", 0.0)),
                "baseline": float(b),
                "y_detrended": float(rr),
            }
        )

    info = {
        "detrend": "poly",
        "poly_order": int(poly_order),
        "exclude_inner_days": float(exclude_inner_days),
        "coef": [float(c) for c in coef.tolist()],
        "n_fit": int(n_fit),
        "n_total": int(len(series)),
    }
    return out, info


def _filter_bins_by_min_count(series: Sequence[Dict[str, float]], *, min_count: int) -> List[Dict[str, float]]:
    """
    Filter binned time series rows by minimum sample count per bin.

    Motivation:
    - Very small-n bins can produce extreme outliers after detrending (e.g. brief gaps or mode changes).
    - Cassini plots are intended to show the smooth conjunction signature; we prefer stable bins.
    """
    k = int(min_count)
    # 条件分岐: `k <= 0` を満たす経路を評価する。
    if k <= 0:
        return list(series)

    out: List[Dict[str, float]] = []
    for r in series:
        n = r.get("n")
        try:
            n_i = int(round(float(n))) if n is not None else 0
        except Exception:
            n_i = 0

        # 条件分岐: `n_i < k` を満たす経路を評価する。

        if n_i < k:
            continue

        out.append(dict(r))

    return out


def _parse_odf3c_span(lbl_text: str) -> Tuple[int, int]:
    # Example:
    #   ^ODF3C_TABLE = ("C32...ODF",6)
    #   OBJECT = ODF3C_TABLE ... ROWS = 91863
    m_ptr = re.search(r"^\^ODF3C_TABLE\s*=\s*\(\"[^\"]+\"\s*,\s*(\d+)\s*\)", lbl_text, flags=re.MULTILINE)
    # 条件分岐: `not m_ptr` を満たす経路を評価する。
    if not m_ptr:
        raise RuntimeError("Failed to locate ^ODF3C_TABLE pointer in .LBL")

    start_record_1 = int(m_ptr.group(1))

    m_rows = re.search(r"OBJECT\s*=\s*ODF3C_TABLE.*?^\s*ROWS\s*=\s*(\d+)\s*$", lbl_text, flags=re.DOTALL | re.MULTILINE)
    # 条件分岐: `not m_rows` を満たす経路を評価する。
    if not m_rows:
        raise RuntimeError("Failed to locate ODF3C_TABLE ROWS in .LBL")

    rows = int(m_rows.group(1))

    # 条件分岐: `start_record_1 <= 0 or rows <= 0` を満たす経路を評価する。
    if start_record_1 <= 0 or rows <= 0:
        raise RuntimeError("Invalid ODF3C span in .LBL")

    return start_record_1, rows


def _iter_cached_odf_files(pds_root: Path, *, doy_start: int, doy_stop: int) -> Iterable[Tuple[int, int, Path]]:
    # Yield (cors, doy, odf_path)
    for odf in sorted(pds_root.glob("cors_*/sce1_*/odf/*.odf")):
        try:
            cors = int(odf.parents[2].name.split("_", 1)[1])
            doy = int(odf.parents[1].name.split("_", 1)[1])
        except Exception:
            continue

        # 条件分岐: `doy_start <= doy <= doy_stop` を満たす経路を評価する。

        if doy_start <= doy <= doy_stop:
            yield cors, doy, odf


def load_odf_doppler_observations(
    pds_root: Path,
    *,
    doy_start: int,
    doy_stop: int,
    want_data_type_id: int = 12,
    want_downlink_band_id: Optional[int] = 3,
    want_stations: Optional[Sequence[int]] = None,
) -> List[OdfObs]:
    """
    Extract Doppler observables from cached ODF files.

    - Uses ODF3C_TABLE layout (36-byte logical records) defined in each file's .LBL.
    - Converts Doppler residual [Hz] to fractional frequency y [-] using a nominal carrier frequency.
    """

    # 条件分岐: `want_downlink_band_id is not None and want_downlink_band_id not in CARRIER_HZ...` を満たす経路を評価する。
    if want_downlink_band_id is not None and want_downlink_band_id not in CARRIER_HZ_BY_DOWNLINK_BAND_ID:
        raise ValueError(f"Unsupported downlink band id: {want_downlink_band_id}")

    obs: List[OdfObs] = []
    stations_set = set(int(s) for s in want_stations) if want_stations else None

    for cors, doy, odf_path in _iter_cached_odf_files(pds_root, doy_start=doy_start, doy_stop=doy_stop):
        lbl_path = odf_path.with_suffix(".lbl")
        # 条件分岐: `not lbl_path.exists()` を満たす経路を評価する。
        if not lbl_path.exists():
            # Mirror stores lowercase; try uppercase as a fallback.
            alt = odf_path.with_suffix(".LBL")
            lbl_path = alt if alt.exists() else lbl_path

        # 条件分岐: `not lbl_path.exists()` を満たす経路を評価する。

        if not lbl_path.exists():
            print(f"[warn] missing label for ODF: {odf_path}")
            continue

        lbl_txt = lbl_path.read_text(encoding="utf-8", errors="replace")
        start_rec_1, n_rows = _parse_odf3c_span(lbl_txt)
        start_idx = start_rec_1 - 1  # 0-based

        b = odf_path.read_bytes()
        n_records = len(b) // ODF_REC_BYTES
        end_idx = min(n_records, start_idx + n_rows)
        # 条件分岐: `start_idx < 0 or start_idx >= n_records` を満たす経路を評価する。
        if start_idx < 0 or start_idx >= n_records:
            print(f"[warn] invalid ODF3C span for {odf_path} (records={n_records}, start={start_idx})")
            continue

        for i in range(start_idx, end_idx):
            rec = b[i * ODF_REC_BYTES : (i + 1) * ODF_REC_BYTES]
            # 条件分岐: `len(rec) != ODF_REC_BYTES` を満たす経路を評価する。
            if len(rec) != ODF_REC_BYTES:
                continue

            t_sec = struct.unpack(">I", rec[0:4])[0]
            items23 = struct.unpack(">I", rec[4:8])[0]
            frac_ms = _extract_bits(items23, 32, 1, 10)

            obs_int = struct.unpack(">i", rec[8:12])[0]
            obs_frac = struct.unpack(">i", rec[12:16])[0]
            doppler_hz = float(obs_int) + float(obs_frac) * 1e-9

            items619 = int.from_bytes(rec[16:28], "big", signed=False)
            data_type_id = _extract_bits(items619, 96, 20, 6)
            downlink_band_id = _extract_bits(items619, 96, 26, 2)
            uplink_band_id = _extract_bits(items619, 96, 28, 2)
            station_rx = _extract_bits(items619, 96, 4, 7)
            station_tx = _extract_bits(items619, 96, 11, 7)
            is_bad = _extract_bits(items619, 96, 32, 1) != 0
            # 条件分岐: `is_bad` を満たす経路を評価する。
            if is_bad:
                continue

            # 条件分岐: `want_data_type_id is not None and data_type_id != int(want_data_type_id)` を満たす経路を評価する。

            if want_data_type_id is not None and data_type_id != int(want_data_type_id):
                continue

            # 条件分岐: `want_downlink_band_id is not None and downlink_band_id != int(want_downlink_b...` を満たす経路を評価する。

            if want_downlink_band_id is not None and downlink_band_id != int(want_downlink_band_id):
                continue

            # 条件分岐: `stations_set is not None and station_rx not in stations_set` を満たす経路を評価する。

            if stations_set is not None and station_rx not in stations_set:
                continue

            carrier_hz = CARRIER_HZ_BY_DOWNLINK_BAND_ID.get(downlink_band_id)
            # 条件分岐: `not carrier_hz` を満たす経路を評価する。
            if not carrier_hz:
                continue

            t_utc = EPOCH_1950 + timedelta(seconds=int(t_sec), milliseconds=int(frac_ms))
            obs.append(
                OdfObs(
                    time_utc=t_utc,
                    doppler_hz=doppler_hz,
                    y=doppler_hz / carrier_hz,
                    station_rx=station_rx,
                    station_tx=station_tx,
                    data_type_id=data_type_id,
                    downlink_band_id=downlink_band_id,
                    uplink_band_id=uplink_band_id,
                    source_file=str(odf_path.relative_to(pds_root).as_posix()),
                )
            )

    obs.sort(key=lambda r: r.time_utc)
    # 条件分岐: `not obs` を満たす経路を評価する。
    if not obs:
        raise RuntimeError("No ODF Doppler observations found in the requested cache/range.")

    return obs


@dataclass(frozen=True)
class TdfObs:
    time_utc: datetime
    doppler_pseudoresidual_hz: float
    y: float
    frequency_level_indicator: int
    doppler_ref_receiver_frequency_hz: float
    station_id: int
    downlink_band_id: int
    sample_data_type_id: int
    sample_interval_cs: int
    doppler_channel_count: int
    ground_mode: int
    doppler_pr_tolerance: int
    doppler_noise_tolerance: int
    total_slipped_cycles: int
    doppler_noise_mhz: float
    received_signal_strength_centidbm: int
    pr_sign4: int
    pr_low32_u: int
    pr_signed36: int
    pr_mhz: float
    source_file: str


def _parse_tdf5_span(lbl_text: str) -> Tuple[int, int]:
    # ^TDF5_TABLE = ("...TDF",3)
    m_ptr = re.search(r"^\^TDF5_TABLE\s*=\s*\(\"[^\"]+\"\s*,\s*(\d+)\s*\)", lbl_text, flags=re.MULTILINE)
    # 条件分岐: `not m_ptr` を満たす経路を評価する。
    if not m_ptr:
        raise RuntimeError("Failed to locate ^TDF5_TABLE pointer in .LBL")

    start_record_1 = int(m_ptr.group(1))

    m_rows = re.search(r"OBJECT\s*=\s*TDF5_TABLE.*?^\s*ROWS\s*=\s*(\d+)\s*$", lbl_text, flags=re.DOTALL | re.MULTILINE)
    # 条件分岐: `not m_rows` を満たす経路を評価する。
    if not m_rows:
        raise RuntimeError("Failed to locate TDF5_TABLE ROWS in .LBL")

    rows = int(m_rows.group(1))

    # 条件分岐: `start_record_1 <= 0 or rows <= 0` を満たす経路を評価する。
    if start_record_1 <= 0 or rows <= 0:
        raise RuntimeError("Invalid TDF5 span in .LBL")

    return start_record_1, rows


def _iter_cached_tdf_files(pds_root: Path, *, doy_start: int, doy_stop: int) -> Iterable[Tuple[int, int, Path]]:
    for tdf in sorted(pds_root.glob("cors_*/sce1_*/tdf/*.tdf")):
        try:
            cors = int(tdf.parents[2].name.split("_", 1)[1])
            doy = int(tdf.parents[1].name.split("_", 1)[1])
        except Exception:
            continue

        # 条件分岐: `doy_start <= doy <= doy_stop` を満たす経路を評価する。

        if doy_start <= doy <= doy_stop:
            yield cors, doy, tdf


def load_tdf_doppler_pseudoresiduals(
    pds_root: Path,
    *,
    doy_start: int,
    doy_stop: int,
    want_downlink_band_id: Optional[int] = 3,
    want_sample_data_type_id: Optional[int] = 2,
    want_stations: Optional[Sequence[int]] = None,
    want_doppler_channel_count: Optional[int] = None,
    want_ground_mode: Optional[int] = None,
    fixed_point_frac_bits: Optional[int] = 0,
    max_abs_pr_hz: Optional[float] = None,
    require_pr_tolerance: bool = True,
    exclude_slipped_cycles: bool = True,
    exclude_pr_sentinel: bool = True,
    exclude_rss_sentinel: bool = True,
    y_sign: float = -1.0,
) -> List[TdfObs]:
    """
    Extract Doppler pseudo-residuals from cached TDF files (TDF5_TABLE).

    Uses the PDS label-defined layout (TDF5_TABLE, COLUMN 16 "RADIOMETRIC BLOCK"):
      - SIGN BITS DOPPLER PSEUDORESIDUAL (Item 73): START_BIT=893, BITS=4
      - DOPPLER PSEUDORESIDUAL (Item 74): MILLIHERTZ, START_BIT=897, BITS=32 (within the radiometric block)

    Notes:
    - DSNの旧形式では、36-bit量を (上位4bitの符号拡張 + 下位32bit) に分割して保持することがある。
    - `fixed_point_frac_bits` を指定すると、pr_signed36 / 2^fixed_point_frac_bits を mHz とみなして復元する。
      （値のスケールが合わない場合は 0/None にして生値比較も可能。）
    - 取得した一次データには、疑似残差に「無効値（センチネル）」が混入することがある。
      例：pr_signed36 == -1048576000 (mHz) は -1048576 Hz (= -2^20 Hz) で、現実的な残差ではないため除外する。
      また、受信電力（Item 89）が -25.60 dBm (=-2560 centidBm) のレコードは、同時に巨大な疑似残差/ノイズが現れるため除外する。
    """

    stations_set = set(int(s) for s in want_stations) if want_stations else None
    out: List[TdfObs] = []

    for cors, doy, tdf_path in _iter_cached_tdf_files(pds_root, doy_start=doy_start, doy_stop=doy_stop):
        lbl_path = tdf_path.with_suffix(".lbl")
        # 条件分岐: `not lbl_path.exists()` を満たす経路を評価する。
        if not lbl_path.exists():
            alt = tdf_path.with_suffix(".LBL")
            lbl_path = alt if alt.exists() else lbl_path

        # 条件分岐: `not lbl_path.exists()` を満たす経路を評価する。

        if not lbl_path.exists():
            print(f"[warn] missing label for TDF: {tdf_path}")
            continue

        lbl_txt = lbl_path.read_text(encoding="utf-8", errors="replace")
        start_rec_1, n_rows = _parse_tdf5_span(lbl_txt)
        start_idx = start_rec_1 - 1

        b = tdf_path.read_bytes()
        rec_bytes = TDF_REC_BYTES
        n_records = len(b) // rec_bytes
        end_idx = min(n_records, start_idx + n_rows)
        # 条件分岐: `start_idx < 0 or start_idx >= n_records` を満たす経路を評価する。
        if start_idx < 0 or start_idx >= n_records:
            print(f"[warn] invalid TDF5 span for {tdf_path} (records={n_records}, start={start_idx})")
            continue

        # Column layout (SFOC-NAV-2-25 format): fixed offsets.

        dt_block_off = 9  # START_BYTE=10, BYTES=9
        dt_block_len = 9
        dtype1_off = 18  # START_BYTE=19, BYTES=4
        status_block1_off = 27  # START_BYTE=28, BYTES=3
        status_block1_len = 3
        sample_interval_off = 32  # START_BYTE=33, BYTES=4
        radiometric_off = 55  # START_BYTE=56, BYTES=125
        radiometric_len = 125
        doppler_block_off = 182  # START_BYTE=183, BYTES=7
        doppler_block_len = 7
        doppler_pr_byte_off = radiometric_off + ((TDF5_DOPPLER_PR_VALUE_START_BIT - 1) // 8)  # 0-based
        sign_byte_off = radiometric_off + ((TDF5_DOPPLER_PR_SIGN_BITS_START_BIT - 1) // 8)  # 0-based
        sign_bit_in_byte = (TDF5_DOPPLER_PR_SIGN_BITS_START_BIT - 1) % 8  # MSB-based bit index in byte
        # 条件分岐: `sign_bit_in_byte != 4` を満たす経路を評価する。
        if sign_bit_in_byte != 4:
            # With current TDF5 labels this should be the low nibble; keep a safe fallback.
            sign_byte_off = -1

        for i in range(start_idx, end_idx):
            rec = b[i * rec_bytes : (i + 1) * rec_bytes]
            # 条件分岐: `len(rec) != rec_bytes` を満たす経路を評価する。
            if len(rec) != rec_bytes:
                continue

            # Parse time tag (year/doy/h/m/s) from DATE-TIME BLOCK.

            dt_bits = int.from_bytes(rec[dt_block_off : dt_block_off + dt_block_len], "big", signed=False)
            year_mod = _extract_bits(dt_bits, 72, 1, 12)
            doy_v = _extract_bits(dt_bits, 72, 13, 16)
            hour = _extract_bits(dt_bits, 72, 29, 8)
            minute = _extract_bits(dt_bits, 72, 37, 8)
            second = _extract_bits(dt_bits, 72, 45, 8)
            year = 1900 + int(year_mod)
            # 条件分岐: `doy_v <= 0 or doy_v > 366` を満たす経路を評価する。
            if doy_v <= 0 or doy_v > 366:
                continue

            t_utc = (
                datetime(year, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=int(doy_v) - 1, hours=int(hour), minutes=int(minute), seconds=int(second))
            )

            dtype1 = struct.unpack(">I", rec[dtype1_off : dtype1_off + 4])[0]
            station_id = _extract_bits(dtype1, 32, 1, 10)
            downlink_band_id = _extract_bits(dtype1, 32, 11, 8)
            sample_data_type_id = _extract_bits(dtype1, 32, 19, 6)
            doppler_channel_count = _extract_bits(dtype1, 32, 25, 4)
            ground_mode = _extract_bits(dtype1, 32, 29, 4)
            sample_interval_cs = struct.unpack(">I", rec[sample_interval_off : sample_interval_off + 4])[0]

            # 条件分岐: `want_downlink_band_id is not None and downlink_band_id != int(want_downlink_b...` を満たす経路を評価する。
            if want_downlink_band_id is not None and downlink_band_id != int(want_downlink_band_id):
                continue

            # 条件分岐: `want_sample_data_type_id is not None and sample_data_type_id != int(want_samp...` を満たす経路を評価する。

            if want_sample_data_type_id is not None and sample_data_type_id != int(want_sample_data_type_id):
                continue

            # 条件分岐: `want_doppler_channel_count is not None and doppler_channel_count != int(want_...` を満たす経路を評価する。

            if want_doppler_channel_count is not None and doppler_channel_count != int(want_doppler_channel_count):
                continue

            # 条件分岐: `want_ground_mode is not None and ground_mode != int(want_ground_mode)` を満たす経路を評価する。

            if want_ground_mode is not None and ground_mode != int(want_ground_mode):
                continue

            # 条件分岐: `stations_set is not None and station_id not in stations_set` を満たす経路を評価する。

            if stations_set is not None and station_id not in stations_set:
                continue

            # 条件分岐: `downlink_band_id not in CARRIER_HZ_BY_DOWNLINK_BAND_ID` を満たす経路を評価する。

            if downlink_band_id not in CARRIER_HZ_BY_DOWNLINK_BAND_ID:
                continue

            carrier_hz = CARRIER_HZ_BY_DOWNLINK_BAND_ID[int(downlink_band_id)]

            # DOPPLER BLOCK quality flags (Column 19, Items 84-89)
            doppler_block_bits = int.from_bytes(
                rec[doppler_block_off : doppler_block_off + doppler_block_len], "big", signed=False
            )
            doppler_pr_tol = _extract_bits(doppler_block_bits, 56, 1, 1)
            doppler_noise_tol = _extract_bits(doppler_block_bits, 56, 2, 1)
            total_slipped_cycles = _extract_bits(doppler_block_bits, 56, 11, 10)

            doppler_noise_u = _extract_bits(doppler_block_bits, 56, 21, 18)
            doppler_noise_s = int(doppler_noise_u)
            # 条件分岐: `doppler_noise_s & (1 << 17)` を満たす経路を評価する。
            if doppler_noise_s & (1 << 17):
                doppler_noise_s -= 1 << 18

            rss_u = _extract_bits(doppler_block_bits, 56, 39, 18)
            rss_s = int(rss_u)
            # 条件分岐: `rss_s & (1 << 17)` を満たす経路を評価する。
            if rss_s & (1 << 17):
                rss_s -= 1 << 18

            # 条件分岐: `exclude_rss_sentinel and int(rss_s) == -2560` を満たす経路を評価する。

            if exclude_rss_sentinel and int(rss_s) == -2560:
                continue

            # 条件分岐: `require_pr_tolerance and int(doppler_pr_tol) != 0` を満たす経路を評価する。

            if require_pr_tolerance and int(doppler_pr_tol) != 0:
                continue

            # 条件分岐: `exclude_slipped_cycles and int(total_slipped_cycles) > 0` を満たす経路を評価する。

            if exclude_slipped_cycles and int(total_slipped_cycles) > 0:
                continue

            # Doppler pseudo-residual:
            # - Often stored as a legacy 36-bit quantity split into (Item 73 sign bits, Item 74 low bits).
            # - fixed_point_frac_bits allows interpreting pr_signed36 as fixed-point mHz.

            if doppler_pr_byte_off + 4 > len(rec):
                continue

            pr_low32_u = struct.unpack(">I", rec[doppler_pr_byte_off : doppler_pr_byte_off + 4])[0]
            pr_sign4 = int(rec[sign_byte_off] & 0x0F) if sign_byte_off >= 0 else 0
            pr_signed36 = _signed36_from_sign4_low32(pr_sign4, pr_low32_u)

            # Known invalid/sentinel pseudo-residual used in DSN-derived products.
            # (-1000 * 2^20) mHz == -1048576 Hz.
            if exclude_pr_sentinel and int(pr_signed36) == -1048576000:
                continue

            frac_bits = int(fixed_point_frac_bits) if fixed_point_frac_bits is not None else 0
            # 条件分岐: `frac_bits < 0 or frac_bits > 30` を満たす経路を評価する。
            if frac_bits < 0 or frac_bits > 30:
                raise ValueError(f"fixed_point_frac_bits out of range: {fixed_point_frac_bits}")

            pr_mhz = float(pr_signed36) / float(2**frac_bits)
            pr_hz = pr_mhz * 1e-3
            # 条件分岐: `max_abs_pr_hz is not None and math.isfinite(float(max_abs_pr_hz)) and float(m...` を満たす経路を評価する。
            if max_abs_pr_hz is not None and math.isfinite(float(max_abs_pr_hz)) and float(max_abs_pr_hz) >= 0.0:
                # 条件分岐: `abs(float(pr_hz)) > float(max_abs_pr_hz)` を満たす経路を評価する。
                if abs(float(pr_hz)) > float(max_abs_pr_hz):
                    continue

            y = float(y_sign) * (pr_hz / carrier_hz)

            # STATUS BLOCK 1: Item 22 (frequency level indicator, DCO vs Sky)
            status1_bits = int.from_bytes(
                rec[status_block1_off : status_block1_off + status_block1_len], "big", signed=False
            )
            frequency_level_indicator = _extract_bits(status1_bits, 8 * int(status_block1_len), 21, 1)

            # RADIOMETRIC BLOCK: Item 43/44 (Doppler reference / receiver sky frequency).
            # Reconstruct: (H/P)*1e3 + (L/P)*1e-6 [Hz] (per casrssis.txt).
            radiometric_bits = int.from_bytes(
                rec[radiometric_off : radiometric_off + radiometric_len], "big", signed=False
            )
            freq_hp = _extract_bits(radiometric_bits, 8 * int(radiometric_len), 149, 32)
            freq_lp = _extract_bits(radiometric_bits, 8 * int(radiometric_len), 181, 32)
            doppler_ref_receiver_frequency_hz = (float(freq_hp) * 1e3) + (float(freq_lp) * 1e-6)

            out.append(
                TdfObs(
                    time_utc=t_utc,
                    doppler_pseudoresidual_hz=pr_hz,
                    y=y,
                    frequency_level_indicator=int(frequency_level_indicator),
                    doppler_ref_receiver_frequency_hz=float(doppler_ref_receiver_frequency_hz),
                    station_id=int(station_id),
                    downlink_band_id=int(downlink_band_id),
                    sample_data_type_id=int(sample_data_type_id),
                    sample_interval_cs=int(sample_interval_cs),
                    doppler_channel_count=int(doppler_channel_count),
                    ground_mode=int(ground_mode),
                    doppler_pr_tolerance=int(doppler_pr_tol),
                    doppler_noise_tolerance=int(doppler_noise_tol),
                    total_slipped_cycles=int(total_slipped_cycles),
                    doppler_noise_mhz=float(doppler_noise_s),
                    received_signal_strength_centidbm=int(rss_s),
                    pr_sign4=int(pr_sign4),
                    pr_low32_u=int(pr_low32_u),
                    pr_signed36=int(pr_signed36),
                    pr_mhz=float(pr_mhz),
                    source_file=str(tdf_path.relative_to(pds_root).as_posix()),
                )
            )

    out.sort(key=lambda r: r.time_utc)
    # 条件分岐: `not out` を満たす経路を評価する。
    if not out:
        raise RuntimeError("No TDF Doppler pseudo-residuals found in the requested cache/range.")

    return out


def _choose_best_common_ground_mode(
    obs_a: Sequence[TdfObs],
    obs_b: Sequence[TdfObs],
    *,
    fixed_ground_mode: Optional[int] = None,
) -> Optional[int]:
    """
    Choose a ground_mode that maximizes the shared usable volume between two datasets.

    Selection metric: maximize min(n_a, n_b) over matching (gm). Ties break by
    (n_a + n_b).
    """

    counts_a: Dict[int, int] = {}
    counts_b: Dict[int, int] = {}
    for o in obs_a:
        gm = int(o.ground_mode)
        counts_a[gm] = counts_a.get(gm, 0) + 1

    for o in obs_b:
        gm = int(o.ground_mode)
        counts_b[gm] = counts_b.get(gm, 0) + 1

    best: Optional[int] = None
    best_score = (-1, -1)
    for gm in set(counts_a).intersection(counts_b):
        # 条件分岐: `fixed_ground_mode is not None and gm != int(fixed_ground_mode)` を満たす経路を評価する。
        if fixed_ground_mode is not None and gm != int(fixed_ground_mode):
            continue

        na = counts_a[gm]
        nb = counts_b[gm]
        score = (min(na, nb), na + nb)
        # 条件分岐: `score > best_score` を満たす経路を評価する。
        if score > best_score:
            best_score = score
            best = gm

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        return None

    return int(best)


def _auto_filter_tdf_obs_by_mode(
    obs: Sequence[TdfObs],
    *,
    want_ground_mode: Optional[int],
    want_doppler_channel_count: Optional[int],
) -> Tuple[List[TdfObs], Optional[int], Optional[int]]:
    """
    If the caller did not specify ground_mode/channel_count, choose the most common
    pair within this dataset and filter to it. Returns (filtered, effective_gm, effective_cc).
    """

    # 条件分岐: `not obs` を満たす経路を評価する。
    if not obs:
        return [], None, None

    # 条件分岐: `want_ground_mode is not None and want_doppler_channel_count is not None` を満たす経路を評価する。

    if want_ground_mode is not None and want_doppler_channel_count is not None:
        gm = int(want_ground_mode)
        cc = int(want_doppler_channel_count)
        filtered = [o for o in obs if int(o.ground_mode) == gm and int(o.doppler_channel_count) == cc]
        return filtered, gm, cc

    # Count (gm,cc)

    counts: Dict[Tuple[int, int], int] = {}
    for o in obs:
        k = (int(o.ground_mode), int(o.doppler_channel_count))
        counts[k] = counts.get(k, 0) + 1

    # Apply partial constraints if provided.

    best: Optional[Tuple[int, int]] = None
    best_n = -1
    for (gm, cc), n in counts.items():
        # 条件分岐: `want_ground_mode is not None and gm != int(want_ground_mode)` を満たす経路を評価する。
        if want_ground_mode is not None and gm != int(want_ground_mode):
            continue

        # 条件分岐: `want_doppler_channel_count is not None and cc != int(want_doppler_channel_count)` を満たす経路を評価する。

        if want_doppler_channel_count is not None and cc != int(want_doppler_channel_count):
            continue

        # 条件分岐: `n > best_n` を満たす経路を評価する。

        if n > best_n:
            best_n = n
            best = (gm, cc)

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        return list(obs), None, None

    gm, cc = int(best[0]), int(best[1])
    filtered = [o for o in obs if int(o.ground_mode) == gm and int(o.doppler_channel_count) == cc]
    return filtered, gm, cc


def parse_dt_utc(s: str) -> datetime:
    # The input should be ISO-8601 with UTC offset.
    dt = datetime.fromisoformat(s)
    # 条件分岐: `dt.tzinfo is None` を満たす経路を評価する。
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def load_model_rows(path: Path) -> List[ModelRow]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[ModelRow] = []
        for r in reader:
            rows.append(
                ModelRow(
                    t=parse_dt_utc(r["time_utc"]),
                    r1_m=float(r["r1_m"]),
                    r2_m=float(r["r2_m"]),
                    b_m=float(r["b_m"]),
                    bdot_mps=float(r["bdot_mps"]),
                    r1dot_mps=float(r["r1dot_mps"]),
                    r2dot_mps=float(r["r2dot_mps"]),
                )
            )

    # 条件分岐: `not rows` を満たす経路を評価する。

    if not rows:
        raise RuntimeError(f"Model CSV is empty: {path}")

    return rows


def find_bmin_time(rows: Sequence[ModelRow]) -> datetime:
    best = min(rows, key=lambda r: r.b_m)
    return best.t


def t_days_from_ref(t: datetime, t_ref: datetime) -> float:
    return (t - t_ref).total_seconds() / 86400.0


def y_full_from_geometry(row: ModelRow, beta: float) -> float:
    # Full (exact derivative of the b-approx Eq(1)) in observable sign convention:
    #   y = - d(Delta_t)/dt
    gamma = 2.0 * beta - 1.0
    coef = 2.0 * (1.0 + gamma) * MU_SUN / (C**3)  # = 4β * MU/c^3
    return -coef * (
        (row.r1dot_mps / row.r1_m)
        + (row.r2dot_mps / row.r2_m)
        - 2.0 * (row.bdot_mps / row.b_m)
    )

def y_eq2_from_geometry(row: ModelRow, beta: float) -> float:
    # Eq(2) approximation used in Cassini (observable sign convention):
    #   y ≈ 4(1+gamma) GM/c^3 * (1/b) db/dt
    gamma = 2.0 * beta - 1.0
    coef = 4.0 * (1.0 + gamma) * MU_SUN / (C**3)  # = 8β * MU/c^3
    return coef * (row.bdot_mps / row.b_m)


def build_model_series(
    rows: Sequence[ModelRow], beta: float, mode: str
) -> Tuple[List[float], List[float]]:
    t_ref = find_bmin_time(rows)
    xs: List[float] = []
    ys: List[float] = []
    # 条件分岐: `mode not in {"eq2", "full"}` を満たす経路を評価する。
    if mode not in {"eq2", "full"}:
        raise ValueError(f"Unknown mode: {mode}")

    for r in rows:
        xs.append(t_days_from_ref(r.t, t_ref))
        # 条件分岐: `mode == "eq2"` を満たす経路を評価する。
        if mode == "eq2":
            ys.append(y_eq2_from_geometry(r, beta=beta))
        else:
            ys.append(y_full_from_geometry(r, beta=beta))

    return xs, ys


def interp_linear(xs: Sequence[float], ys: Sequence[float], x: float) -> Optional[float]:
    # 条件分岐: `not xs` を満たす経路を評価する。
    if not xs:
        return None

    # 条件分岐: `x < xs[0] or x > xs[-1]` を満たす経路を評価する。

    if x < xs[0] or x > xs[-1]:
        return None

    i = bisect_left(xs, x)
    # 条件分岐: `i <= 0` を満たす経路を評価する。
    if i <= 0:
        return ys[0]

    # 条件分岐: `i >= len(xs)` を満たす経路を評価する。

    if i >= len(xs):
        return ys[-1]

    x0 = xs[i - 1]
    x1 = xs[i]
    y0 = ys[i - 1]
    y1 = ys[i]
    # 条件分岐: `x1 == x0` を満たす経路を評価する。
    if x1 == x0:
        return y0

    u = (x - x0) / (x1 - x0)
    return y0 + (y1 - y0) * u


def load_digitized_points(path: Path) -> List[Tuple[float, float]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: List[Tuple[float, float]] = []
        for r in reader:
            out.append((float(r["t_days"]), float(r["y_digit"])))

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise RuntimeError(f"Digitized CSV is empty: {path}")

    out.sort(key=lambda p: p[0])
    return out


def match_points(
    observed: Sequence[Tuple[float, float]],
    model_t_days: Sequence[float],
    model_y: Sequence[float],
    *,
    align_offset: bool = True,
) -> List[Point]:
    points: List[Point] = []
    for t_days, y_obs in observed:
        y_model = interp_linear(model_t_days, model_y, t_days)
        # 条件分岐: `y_model is None` を満たす経路を評価する。
        if y_model is None:
            continue

        points.append(Point(t_days=t_days, y_obs=y_obs, y_model=y_model))

    # 条件分岐: `not points` を満たす経路を評価する。

    if not points:
        raise RuntimeError("No overlapping points between observed data and model.")

    # 条件分岐: `align_offset` を満たす経路を評価する。

    if align_offset:
        # Align by a constant offset (same convention as LLR): y_model <- y_model + k
        k = sum((p.y_obs - p.y_model) for p in points) / len(points)
        points = [Point(t_days=p.t_days, y_obs=p.y_obs, y_model=p.y_model + k) for p in points]

    return points


def rmse(values: Sequence[float]) -> float:
    # 条件分岐: `not values` を満たす経路を評価する。
    if not values:
        return math.nan

    return math.sqrt(sum(v * v for v in values) / len(values))


def mae(values: Sequence[float]) -> float:
    # 条件分岐: `not values` を満たす経路を評価する。
    if not values:
        return math.nan

    return sum(abs(v) for v in values) / len(values)


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    # 条件分岐: `len(xs) != len(ys) or len(xs) < 2` を満たす経路を評価する。
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    # 条件分岐: `vx == 0.0 or vy == 0.0` を満たす経路を評価する。
    if vx == 0.0 or vy == 0.0:
        return math.nan

    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def write_points_csv(path: Path, points: Sequence[Point]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_days", "y_obs", "y_model", "residual"])
        for p in points:
            w.writerow([f"{p.t_days:.15g}", f"{p.y_obs:.15e}", f"{p.y_model:.15e}", f"{p.residual:.15e}"])


def compute_window(points: Sequence[Point], window_days: Optional[float]) -> List[Point]:
    # 条件分岐: `window_days is None` を満たす経路を評価する。
    if window_days is None:
        return list(points)

    return [p for p in points if abs(p.t_days) <= window_days]


def write_metrics_csv(path: Path, points: Sequence[Point]) -> None:
    windows = [
        ("all (available points)", None),
        ("-10 to +10 days", 10.0),
        ("-3 to +3 days", 3.0),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "window",
                "n",
                "rmse",
                "mae",
                "corr",
                "max_obs",
                "min_obs",
                "max_model",
                "min_model",
            ]
        )
        for name, wdays in windows:
            subset = compute_window(points, wdays)
            resids = [p.residual for p in subset]
            yd = [p.y_obs for p in subset]
            ym = [p.y_model for p in subset]
            w.writerow(
                [
                    name,
                    len(subset),
                    f"{rmse(resids):.15e}",
                    f"{mae(resids):.15e}",
                    f"{pearson_corr(yd, ym):.15g}",
                    f"{(max(yd) if yd else math.nan):.15e}",
                    f"{(min(yd) if yd else math.nan):.15e}",
                    f"{(max(ym) if ym else math.nan):.15e}",
                    f"{(min(ym) if ym else math.nan):.15e}",
                ]
            )


def try_plot(
    out_dir: Path,
    model_t_days: Sequence[float],
    model_y: Sequence[float],
    points: Sequence[Point],
    beta: float,
    obs_label: str,
    best_beta_zoom: Optional[Tuple[float, Sequence[float]]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print("matplotlib not available; skipping plots:", e)
        return

    _set_japanese_font()

    xs_pts = [p.t_days for p in points]
    yd_pts = [p.y_obs for p in points]
    ym_pts = [p.y_model for p in points]
    res_pts = [p.residual for p in points]

    # Full overlay
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model_t_days, model_y, label=f"P-model（β={beta:g}）", linewidth=2, color="tab:blue")
    ax.scatter(xs_pts, yd_pts, s=12, label=obs_label, alpha=0.85, color="tab:orange")
    ax.set_title("Cassini：ドップラー y（観測 vs P-model）", fontsize=13)
    ax.set_xlabel("t（日, b_min からの相対）")
    ax.set_ylabel("y（周波数比）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cassini_fig2_overlay_full.png", dpi=180)
    plt.close(fig)

    # Zoom +/- 10 days
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model_t_days, model_y, label=f"P-model（β={beta:g}）", linewidth=2, color="tab:blue")
    ax.scatter(xs_pts, yd_pts, s=12, label=obs_label, alpha=0.85, color="tab:orange")
    ax.set_xlim(-10, 10)
    ax.set_title("Cassini：ドップラー y（±10日, 観測 vs P-model）", fontsize=13)
    ax.set_xlabel("t（日, b_min からの相対）")
    ax.set_ylabel("y（周波数比）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cassini_fig2_overlay_zoom10d.png", dpi=180)
    plt.close(fig)

    # Residuals
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.5)
    ax.scatter(xs_pts, res_pts, s=12, alpha=0.85, color="tab:red")
    ax.set_title("Cassini：残差（観測 - P-model）", fontsize=13)
    ax.set_xlabel("t（日, b_min からの相対）")
    ax.set_ylabel("残差（観測 - モデル）")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cassini_fig2_residuals.png", dpi=180)
    plt.close(fig)

    # Best beta overlay (zoom) if provided
    if best_beta_zoom is not None:
        best_beta, best_model_y = best_beta_zoom
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(model_t_days, best_model_y, label=f"P-model（最適β={best_beta:g}）", linewidth=2, color="tab:blue")
        ax.scatter(xs_pts, yd_pts, s=12, label=obs_label, alpha=0.85, color="tab:orange")
        ax.set_xlim(-10, 10)
        ax.set_title("Cassini：ドップラー y（±10日, 最適β）", fontsize=13)
        ax.set_xlabel("t（日, b_min からの相対）")
        ax.set_ylabel("y（周波数比）")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "cassini_fig2_overlay_bestbeta_zoom10d.png", dpi=180)
        plt.close(fig)


def run_beta_sweep(
    observed: Sequence[Tuple[float, float]],
    rows: Sequence[ModelRow],
    out_csv: Path,
    out_png: Optional[Path],
    mode: str,
    beta_start: float,
    beta_stop: float,
    beta_step: float,
    align_offset: bool,
) -> Tuple[float, List[float]]:
    betas: List[float] = []
    rmse_all: List[float] = []
    rmse_10: List[float] = []
    rmse_3: List[float] = []

    beta = beta_start
    while beta <= beta_stop + (beta_step / 10.0):
        model_t_days, model_y = build_model_series(rows, beta=beta, mode=mode)
        points = match_points(observed, model_t_days, model_y, align_offset=align_offset)

        def _rmse_for_window(days: Optional[float]) -> float:
            subset = compute_window(points, days)
            return rmse([p.residual for p in subset])

        betas.append(beta)
        rmse_all.append(_rmse_for_window(None))
        rmse_10.append(_rmse_for_window(10.0))
        rmse_3.append(_rmse_for_window(3.0))
        beta += beta_step

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["beta", "rmse_all", "rmse_10", "rmse_3"])
        for b, a, z10, z3 in zip(betas, rmse_all, rmse_10, rmse_3):
            w.writerow([f"{b:.15g}", f"{a:.15e}", f"{z10:.15e}", f"{z3:.15e}"])

    best_idx = min(range(len(betas)), key=lambda i: rmse_10[i])
    best_beta = betas[best_idx]

    # 条件分岐: `out_png is None` を満たす経路を評価する。
    if out_png is None:
        return best_beta, rmse_10

    # Plot sweep if matplotlib exists

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print("matplotlib not available; skipping beta sweep plot:", e)
        return best_beta, rmse_10

    _set_japanese_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(betas, rmse_all, label="RMSE（全区間）")
    ax.plot(betas, rmse_10, label="RMSE（±10日）")
    ax.plot(betas, rmse_3, label="RMSE（±3日）")
    ax.axvline(best_beta, color="k", linewidth=1, alpha=0.5, label=f"最適β={best_beta:g}")
    ax.set_title("Cassini：βスイープ（RMSE）", fontsize=13)
    ax.set_xlabel("β")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return best_beta, rmse_10


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Cassini SCE1 observation y(t) against P-model (offline-first; uses cached data when possible)."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="P-model beta. If omitted, read output/theory/frozen_parameters.json (fallback: 1.0).",
    )
    parser.add_argument(
        "--model",
        choices=["eq2", "full"],
        default="eq2",
        help="Model for y(t): eq2 (Cassini Eq(2) approx) or full (exact derivative of Eq(1))",
    )
    parser.add_argument(
        "--model-csv",
        type=str,
        default="output/private/cassini/cassini_shapiro_y_full.csv",
        help="Model geometry CSV (default: output/private/cassini/cassini_shapiro_y_full.csv)",
    )
    parser.add_argument(
        "--digitized-csv",
        type=str,
        default="data/cassini/cassini_fig2_digitized_raw.csv",
        help="Digitized points (t_days,y_digit) (default: data/cassini/cassini_fig2_digitized_raw.csv)",
    )
    parser.add_argument(
        "--source",
        choices=["pds_tdf", "pds_tdf_raw", "pds_odf_raw", "digitized"],
        default="pds_tdf",
        help=(
            "Observation source: pds_tdf (PDS一次データTDF → 平滑化＋デトレンド) / digitized (論文図デジタイズ) "
            "/ pds_tdf_raw (PDS TDFの生値, debug) / pds_odf_raw (PDS ODFの生値, debug). Default: pds_tdf"
        ),
    )
    parser.add_argument("--doy-start", type=int, default=162, help="PDS ODF: start DOY (inclusive). Default: 162")
    parser.add_argument("--doy-stop", type=int, default=182, help="PDS ODF: stop DOY (inclusive). Default: 182")
    parser.add_argument(
        "--odf-downlink-band",
        choices=["ka", "x", "any"],
        default="ka",
        help="PDS ODF: downlink band filter (ka/x/any). Default: ka",
    )
    parser.add_argument(
        "--tdf-downlink-band",
        choices=["ka", "x", "any"],
        default="ka",
        help="PDS TDF: downlink band filter (ka/x/any). Default: ka",
    )
    parser.add_argument(
        "--tdf-plasma-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PDS TDF: Ka+X の2周波でプラズマ（コロナ）成分を除去する（両方のTDFがある場合）。Default: enabled",
    )
    parser.add_argument(
        "--tdf-sample-type",
        choices=["high", "low", "any"],
        default="high",
        help="PDS TDF: sample data type (high-rate/low-rate Doppler). Default: high",
    )
    parser.add_argument(
        "--tdf-ground-mode",
        type=int,
        default=-1,
        help=(
            "PDS TDF: ground mode filter（Item 14）。-1 で無指定（全て）。"
            "代表: 1=1-way Doppler, 2=2-way Doppler, 3=3-way Doppler, 4=3-way coherent. Default: -1"
        ),
    )
    parser.add_argument(
        "--tdf-doppler-channel-count",
        type=int,
        default=-1,
        help="PDS TDF: Doppler channel count filter（Item 13）。-1 で無指定（全て）。Default: -1",
    )
    parser.add_argument(
        "--tdf-fixed-frac-bits",
        type=int,
        default=0,
        help=(
            "PDS TDF: 36-bit量(pr_signed36)を mHz に復元する際の小数ビット数。"
            "PDSラベル上は Item 74 が MILLIHERTZ（整数）なので通常は 0。"
            "値が固定小数点に見える場合のみ 18 等を試す。Default: 0"
        ),
    )
    parser.add_argument(
        "--tdf-y-sign",
        type=int,
        choices=[-1, 1],
        default=1,
        help=(
            "PDS TDF: 疑似残差→y 変換時の符号。"
            "TDF側の定義（O-C / C-O 等）により反転が必要な場合がある。Default: 1"
        ),
    )
    parser.add_argument(
        "--tdf-max-abs-pr-hz",
        type=float,
        default=10.0,
        help="PDS TDF: |Doppler pseudo-residual| の上限 [Hz]。負の値で無効化。Default: 10",
    )
    parser.add_argument(
        "--tdf-plasma-fill-mode",
        choices=["dual", "merged"],
        default="merged",
        help=(
            "PDS TDF: Ka+Xプラズマ補正が可能な場合に、どの系列を“観測”として採用するか。"
            "dual=共通ビン（Ka+Xが同時にある区間）のみ、merged=欠測を単独バンドで補完。Default: merged"
        ),
    )
    parser.add_argument(
        "--tdf-require-pr-tolerance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PDS TDF: Item 84 (DOPPLER PSEUDORESIDUAL TOLERANCE) が 0 のレコードのみ使う。Default: enabled",
    )
    parser.add_argument(
        "--tdf-exclude-slipped-cycles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PDS TDF: Item 87 (TOTAL SLIPPED CYCLES) > 0 を除外する。Default: enabled",
    )
    parser.add_argument(
        "--tdf-exclude-pr-sentinel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PDS TDF: 疑似残差の無効値（例: -1048576 Hz）を除外する。Default: enabled",
    )
    parser.add_argument(
        "--tdf-exclude-rss-sentinel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PDS TDF: 受信電力の無効値（-25.60 dBm）を除外する。Default: enabled",
    )
    parser.add_argument(
        "--tdf-bin-seconds",
        type=int,
        default=3600,
        help="PDS TDF: 平滑化の時間ビン幅 [秒]（平均）。Default: 3600 (60分)",
    )
    parser.add_argument(
        "--tdf-bin-stat",
        choices=["mean", "median"],
        default="median",
        help="PDS TDF: 時間ビン内の代表値。Default: median",
    )
    parser.add_argument(
        "--tdf-min-bin-count",
        type=int,
        default=1,
        help="PDS TDF: 平滑化後、1ビンに含まれる最小サンプル数 n。小さいビンは除外。Default: 1",
    )
    parser.add_argument(
        "--tdf-detrend-poly-order",
        type=int,
        default=0,
        help="PDS TDF: デトレンドの多項式次数。Default: 0",
    )
    parser.add_argument(
        "--tdf-detrend-exclude-inner-days",
        type=float,
        default=5.0,
        help="PDS TDF: |t| < この日数はフィットから除外（中心の形を守る）。Default: 5.0",
    )
    parser.add_argument(
        "--tdf-reconstruct-shapiro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "PDS TDF: Doppler pseudo-residual（観測-予測）に参照Shapiro y_ref(t) を足し戻し、"
            "論文図スケールの y(t) を復元して比較する。Default: enabled"
        ),
    )
    parser.add_argument(
        "--tdf-reconstruct-beta",
        type=float,
        default=None,
        help="PDS TDF: 参照Shapiro復元で使う β。省略時は --beta と同じ（凍結β→fallback=1.0）。",
    )
    parser.add_argument(
        "--odf-stations",
        type=str,
        default="",
        help="PDS ODF: comma-separated station IDs to include (e.g., 25,45,54). Default: all.",
    )
    parser.add_argument(
        "--tdf-stations",
        type=str,
        default="",
        help="PDS TDF: comma-separated station IDs to include (e.g., 25,45,65). Default: all.",
    )
    parser.add_argument(
        "--no-align-offset",
        action="store_true",
        help="Do not align by a constant offset between observed and model.",
    )
    parser.add_argument("--no-sweep", action="store_true", help="Skip beta sweep outputs")
    parser.add_argument("--no-plots", action="store_true", help="Write CSVs but skip all PNG outputs")
    parser.add_argument("--beta-start", type=float, default=0.999985)
    parser.add_argument("--beta-stop", type=float, default=1.000015)
    parser.add_argument("--beta-step", type=float, default=2.5e-7)
    args = parser.parse_args()

    root = _repo_root()
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `args.beta is not None` を満たす経路を評価する。
    if args.beta is not None:
        beta = float(args.beta)
        beta_source = "cli"
    else:
        beta, beta_source = _try_load_frozen_beta(root)
        # 条件分岐: `beta is None` を満たす経路を評価する。
        if beta is None:
            beta = 1.0
            beta_source = "default_beta_1"

    tdf_plasma_requested = bool(getattr(args, "tdf_plasma_correct", False))
    tdf_plasma_used = False
    tdf_plasma_fallback_reason: Optional[str] = None
    tdf_plasma_common_bins: Optional[int] = None
    tdf_plasma_common_station_ids: List[int] = []
    tdf_reconstruct_shapiro = bool(getattr(args, "tdf_reconstruct_shapiro", True))
    tdf_reconstruct_beta = float(args.tdf_reconstruct_beta) if args.tdf_reconstruct_beta is not None else float(beta)
    tdf_reconstruct_beta_source = "cli" if args.tdf_reconstruct_beta is not None else beta_source

    want_ground_mode: Optional[int] = None if int(args.tdf_ground_mode) < 0 else int(args.tdf_ground_mode)
    want_doppler_channel_count: Optional[int] = (
        None if int(args.tdf_doppler_channel_count) < 0 else int(args.tdf_doppler_channel_count)
    )
    tdf_effective_ground_mode: Optional[int] = None
    tdf_effective_doppler_channel_count: Optional[int] = None
    tdf_effective_doppler_channel_count_by_band: Dict[int, int] = {}
    tdf_effective_downlink_bands: List[int] = []
    dual_binned_out: Optional[Path] = None
    merged_binned_out: Optional[Path] = None

    model_csv = (root / args.model_csv).resolve()

    rows = load_model_rows(model_csv)

    model_t_days, model_y = build_model_series(rows, beta=beta, mode=args.model)

    align_offset = not bool(args.no_align_offset)

    requested_source = str(args.source)
    effective_source = requested_source
    fallback_reason: Optional[str] = None

    obs_label = "観測（一次データ: PDS ODF）"
    observed_pairs: List[Tuple[float, float]] = []
    generated_pds_series = False
    recon_tag = "（疑似残差+参照Shapiro復元）" if tdf_reconstruct_shapiro else ""

    # 条件分岐: `args.source == "digitized"` を満たす経路を評価する。
    if args.source == "digitized":
        digitized_csv = (root / args.digitized_csv).resolve()
        digitized = load_digitized_points(digitized_csv)
        observed_pairs = list(digitized)
        obs_label = "観測（論文図をデジタイズ）"
        effective_source = "digitized"
    # 条件分岐: 前段条件が不成立で、`args.source == "pds_odf_raw"` を追加評価する。
    elif args.source == "pds_odf_raw":
        pds_root = root / "data" / "cassini" / "pds_sce1"
        # 条件分岐: `not pds_root.exists()` を満たす経路を評価する。
        if not pds_root.exists():
            print("[warn] PDS cache not found. Falling back to digitized points.")
            digitized_csv = (root / args.digitized_csv).resolve()
            digitized = load_digitized_points(digitized_csv)
            observed_pairs = list(digitized)
            obs_label = "観測（論文図をデジタイズ）"
            effective_source = "digitized_fallback"
            fallback_reason = "pds_cache_missing"
        else:
            want_band = str(args.odf_downlink_band).lower()
            want_downlink_band_id = 3 if want_band == "ka" else 2 if want_band == "x" else None
            stations = [int(s.strip()) for s in str(args.odf_stations).split(",") if s.strip()] or None
            odf_obs = load_odf_doppler_observations(
                pds_root,
                doy_start=int(args.doy_start),
                doy_stop=int(args.doy_stop),
                want_data_type_id=12,
                want_downlink_band_id=want_downlink_band_id,
                want_stations=stations,
            )
            t_ref = find_bmin_time(rows)
            for r in odf_obs:
                observed_pairs.append((t_days_from_ref(r.time_utc, t_ref), float(r.y)))

            # Persist extracted observation for traceability (still output/, because it's generated).

            obs_out = out_dir / "cassini_sce1_odf_observed_raw.csv"
            with obs_out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "time_utc",
                        "t_days",
                        "y_obs",
                        "doppler_hz",
                        "station_rx",
                        "station_tx",
                        "data_type_id",
                        "downlink_band_id",
                        "uplink_band_id",
                        "source_file",
                    ]
                )
                for r in odf_obs:
                    w.writerow(
                        [
                            r.time_utc.isoformat(),
                            f"{t_days_from_ref(r.time_utc, t_ref):.15g}",
                            f"{r.y:.15e}",
                            f"{r.doppler_hz:.15e}",
                            r.station_rx,
                            r.station_tx,
                            r.data_type_id,
                            r.downlink_band_id,
                            r.uplink_band_id,
                            r.source_file,
                        ]
                    )

            print("Wrote:", obs_out)
            obs_label = "観測（一次データ: PDS ODF, raw Doppler）"
            effective_source = "pds_odf_raw"
    # 条件分岐: 前段条件が不成立で、`args.source == "pds_tdf_raw"` を追加評価する。
    elif args.source == "pds_tdf_raw":
        # Debug: TDF pseudo-residuals without smoothing/detrend.
        pds_root = root / "data" / "cassini" / "pds_sce1"
        # 条件分岐: `not pds_root.exists()` を満たす経路を評価する。
        if not pds_root.exists():
            print("[warn] PDS cache not found. Falling back to digitized points.")
            digitized_csv = (root / args.digitized_csv).resolve()
            digitized = load_digitized_points(digitized_csv)
            observed_pairs = list(digitized)
            obs_label = "観測（論文図をデジタイズ）"
            effective_source = "digitized_fallback"
            fallback_reason = "pds_cache_missing"
        else:
            want_band = str(args.tdf_downlink_band).lower()
            want_downlink_band_id = 3 if want_band == "ka" else 2 if want_band == "x" else None
            sample = str(args.tdf_sample_type).lower()
            want_sample_id = 1 if sample == "high" else 2 if sample == "low" else None
            stations = [int(s.strip()) for s in str(args.tdf_stations).split(",") if s.strip()] or None

            tdf_obs = load_tdf_doppler_pseudoresiduals(
                pds_root,
                doy_start=int(args.doy_start),
                doy_stop=int(args.doy_stop),
                want_downlink_band_id=want_downlink_band_id,
                want_sample_data_type_id=want_sample_id,
                want_stations=stations,
                want_doppler_channel_count=want_doppler_channel_count,
                want_ground_mode=want_ground_mode,
                fixed_point_frac_bits=int(args.tdf_fixed_frac_bits),
                max_abs_pr_hz=None if float(args.tdf_max_abs_pr_hz) < 0 else float(args.tdf_max_abs_pr_hz),
                require_pr_tolerance=bool(args.tdf_require_pr_tolerance),
                exclude_slipped_cycles=bool(args.tdf_exclude_slipped_cycles),
                exclude_pr_sentinel=bool(args.tdf_exclude_pr_sentinel),
                exclude_rss_sentinel=bool(args.tdf_exclude_rss_sentinel),
                y_sign=float(int(args.tdf_y_sign)),
            )
            tdf_effective_ground_mode = want_ground_mode
            tdf_effective_doppler_channel_count = want_doppler_channel_count
            t_ref = find_bmin_time(rows)
            for r in tdf_obs:
                observed_pairs.append((t_days_from_ref(r.time_utc, t_ref), float(r.y)))

            obs_out = out_dir / "cassini_sce1_tdf_extracted.csv"
            with obs_out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "time_utc",
                        "t_days",
                        "y_obs",
                        "doppler_pseudoresidual_hz",
                        "doppler_pseudoresidual_mhz",
                        "frequency_level_indicator",
                        "doppler_ref_receiver_frequency_hz",
                        "doppler_pr_tolerance",
                        "doppler_noise_tolerance",
                        "total_slipped_cycles",
                        "doppler_noise_mhz",
                        "received_signal_strength_centidbm",
                        "pr_sign4",
                        "pr_low32_u",
                        "pr_signed36",
                        "sample_interval_cs",
                        "doppler_channel_count",
                        "ground_mode",
                        "station_id",
                        "downlink_band_id",
                        "sample_data_type_id",
                        "source_file",
                    ]
                )
                for r in tdf_obs:
                    w.writerow(
                        [
                            r.time_utc.isoformat(),
                            f"{t_days_from_ref(r.time_utc, t_ref):.15g}",
                            f"{r.y:.15e}",
                            f"{r.doppler_pseudoresidual_hz:.15e}",
                            f"{r.pr_mhz:.15e}",
                            r.frequency_level_indicator,
                            f"{r.doppler_ref_receiver_frequency_hz:.15e}",
                            r.doppler_pr_tolerance,
                            r.doppler_noise_tolerance,
                            r.total_slipped_cycles,
                            f"{r.doppler_noise_mhz:.15g}",
                            r.received_signal_strength_centidbm,
                            r.pr_sign4,
                            r.pr_low32_u,
                            r.pr_signed36,
                            r.sample_interval_cs,
                            r.doppler_channel_count,
                            r.ground_mode,
                            r.station_id,
                            r.downlink_band_id,
                            r.sample_data_type_id,
                            r.source_file,
                        ]
                    )

            print("Wrote:", obs_out)
            obs_label = "観測（一次データ: PDS TDF, Doppler疑似残差・生）"
            effective_source = "pds_tdf_raw"
    else:
        # Primary: TDF pseudo-residuals -> binning + detrend to make a paper-like small y(t).
        pds_root = root / "data" / "cassini" / "pds_sce1"
        # 条件分岐: `not pds_root.exists()` を満たす経路を評価する。
        if not pds_root.exists():
            print("[warn] PDS cache not found. Falling back to digitized points.")
            digitized_csv = (root / args.digitized_csv).resolve()
            digitized = load_digitized_points(digitized_csv)
            observed_pairs = list(digitized)
            obs_label = "観測（論文図をデジタイズ）"
            effective_source = "digitized_fallback"
            fallback_reason = "pds_cache_missing"
        else:
            want_band = str(args.tdf_downlink_band).lower()
            want_downlink_band_id = 3 if want_band == "ka" else 2 if want_band == "x" else None
            sample = str(args.tdf_sample_type).lower()
            want_sample_id = 1 if sample == "high" else 2 if sample == "low" else None
            stations = [int(s.strip()) for s in str(args.tdf_stations).split(",") if s.strip()] or None

            t_ref = find_bmin_time(rows)

            # Prefer dual-frequency plasma correction (Ka+X) when available.
            has_ka = any(pds_root.glob("cors_*/sce1_*/tdf/*k252v0.tdf"))
            has_x = any(pds_root.glob("cors_*/sce1_*/tdf/*xmmmv0.tdf"))
            use_plasma = bool(tdf_plasma_requested) and has_ka and has_x

            tdf_obs_for_extract: List[TdfObs] = []
            detrended: List[Dict[str, float]] = []
            detrend_info: Dict[str, object] = {}

            # 条件分岐: `use_plasma` を満たす経路を評価する。
            if use_plasma:
                f_ka = float(CARRIER_HZ_BY_DOWNLINK_BAND_ID[3])
                f_x = float(CARRIER_HZ_BY_DOWNLINK_BAND_ID[2])
                denom = (f_ka**2) - (f_x**2)
                # 条件分岐: `denom == 0.0` を満たす経路を評価する。
                if denom == 0.0:
                    use_plasma = False
                    tdf_plasma_fallback_reason = "invalid_carrier_denominator"
                else:
                    # Decode both bands, then combine:
                    #   y(f) = y_gr + A/f^2  (dispersive plasma term)
                    #   y_gr = (y_ka*f_ka^2 - y_x*f_x^2) / (f_ka^2 - f_x^2)
                    tdf_obs_all = load_tdf_doppler_pseudoresiduals(
                        pds_root,
                        doy_start=int(args.doy_start),
                        doy_stop=int(args.doy_stop),
                        want_downlink_band_id=None,
                        want_sample_data_type_id=want_sample_id,
                        want_stations=stations,
                        want_doppler_channel_count=want_doppler_channel_count,
                        want_ground_mode=want_ground_mode,
                        fixed_point_frac_bits=int(args.tdf_fixed_frac_bits),
                        max_abs_pr_hz=None if float(args.tdf_max_abs_pr_hz) < 0 else float(args.tdf_max_abs_pr_hz),
                        require_pr_tolerance=bool(args.tdf_require_pr_tolerance),
                        exclude_slipped_cycles=bool(args.tdf_exclude_slipped_cycles),
                        exclude_pr_sentinel=bool(args.tdf_exclude_pr_sentinel),
                        exclude_rss_sentinel=bool(args.tdf_exclude_rss_sentinel),
                        y_sign=float(int(args.tdf_y_sign)),
                    )
                    tdf_obs_ka = [r for r in tdf_obs_all if int(r.downlink_band_id) == 3]
                    tdf_obs_x = [r for r in tdf_obs_all if int(r.downlink_band_id) == 2]
                    # 条件分岐: `(not tdf_obs_ka) or (not tdf_obs_x)` を満たす経路を評価する。
                    if (not tdf_obs_ka) or (not tdf_obs_x):
                        use_plasma = False
                        tdf_plasma_fallback_reason = "missing_ka_or_x_records"
                    else:
                        # Restrict to stations that exist in *both* bands.
                        # (In practice, Ka-band may only exist for a subset of stations; mixing stations causes cancellation.)
                        stations_ka = {int(r.station_id) for r in tdf_obs_ka}
                        stations_x = {int(r.station_id) for r in tdf_obs_x}
                        common_stations = stations_ka.intersection(stations_x)
                        # 条件分岐: `not common_stations` を満たす経路を評価する。
                        if not common_stations:
                            use_plasma = False
                            tdf_plasma_fallback_reason = "no_common_stations_between_ka_and_x"
                        else:
                            tdf_plasma_common_station_ids = sorted(int(s) for s in common_stations)
                            tdf_obs_ka = [r for r in tdf_obs_ka if int(r.station_id) in common_stations]
                            tdf_obs_x = [r for r in tdf_obs_x if int(r.station_id) in common_stations]

                        # 条件分岐: `use_plasma` を満たす経路を評価する。

                        if use_plasma:
                            # Restrict to a consistent ground tracking mode first.
                            # (Note: Doppler channel count may differ between Ka/X even for the same session.)
                            gm_eff = _choose_best_common_ground_mode(
                                tdf_obs_ka,
                                tdf_obs_x,
                                fixed_ground_mode=want_ground_mode,
                            )
                            # 条件分岐: `gm_eff is None` を満たす経路を評価する。
                            if gm_eff is None:
                                use_plasma = False
                                tdf_plasma_fallback_reason = "no_common_ground_mode"
                            else:
                                tdf_effective_ground_mode = int(gm_eff)
                                tdf_obs_ka = [r for r in tdf_obs_ka if int(r.ground_mode) == int(gm_eff)]
                                tdf_obs_x = [r for r in tdf_obs_x if int(r.ground_mode) == int(gm_eff)]

                                # Within each band, further restrict to the most common Doppler channel count
                                # (or the user-specified one) to avoid mode mixing discontinuities.
                                tdf_obs_ka, _gm_ka, cc_ka_eff = _auto_filter_tdf_obs_by_mode(
                                    tdf_obs_ka,
                                    want_ground_mode=int(gm_eff),
                                    want_doppler_channel_count=want_doppler_channel_count,
                                )
                                tdf_obs_x, _gm_x, cc_x_eff = _auto_filter_tdf_obs_by_mode(
                                    tdf_obs_x,
                                    want_ground_mode=int(gm_eff),
                                    want_doppler_channel_count=want_doppler_channel_count,
                                )
                                # 条件分岐: `cc_ka_eff is not None` を満たす経路を評価する。
                                if cc_ka_eff is not None:
                                    tdf_effective_doppler_channel_count_by_band[3] = int(cc_ka_eff)

                                # 条件分岐: `cc_x_eff is not None` を満たす経路を評価する。

                                if cc_x_eff is not None:
                                    tdf_effective_doppler_channel_count_by_band[2] = int(cc_x_eff)

                                # 条件分岐: `(not tdf_obs_ka) or (not tdf_obs_x)` を満たす経路を評価する。

                                if (not tdf_obs_ka) or (not tdf_obs_x):
                                    use_plasma = False
                                    tdf_plasma_fallback_reason = "no_records_after_ground_mode_filter"

                        # 条件分岐: `not use_plasma` を満たす経路を評価する。

                        if not use_plasma:
                            # Skip the rest of plasma processing; fallback is handled below.
                            pass
                        else:
                            raw_pairs_ka: List[Tuple[float, float]] = []
                            raw_pairs_x: List[Tuple[float, float]] = []
                            for r in tdf_obs_ka:
                                raw_pairs_ka.append((t_days_from_ref(r.time_utc, t_ref), float(r.y)))

                            for r in tdf_obs_x:
                                raw_pairs_x.append((t_days_from_ref(r.time_utc, t_ref), float(r.y)))

                            raw_pairs_ka.sort(key=lambda p: p[0])
                            raw_pairs_x.sort(key=lambda p: p[0])

                            binned_ka = _bin_time_series(raw_pairs_ka, bin_seconds=int(args.tdf_bin_seconds), stat=str(args.tdf_bin_stat))
                            binned_x = _bin_time_series(raw_pairs_x, bin_seconds=int(args.tdf_bin_seconds), stat=str(args.tdf_bin_stat))
                            binned_ka = _filter_bins_by_min_count(binned_ka, min_count=int(args.tdf_min_bin_count))
                            binned_x = _filter_bins_by_min_count(binned_x, min_count=int(args.tdf_min_bin_count))
                            by_t_ka = {float(r["t_days"]): r for r in binned_ka}
                            by_t_x = {float(r["t_days"]): r for r in binned_x}
                            common_t = sorted(set(by_t_ka).intersection(by_t_x))
                            merged_t = sorted(set(by_t_ka).union(by_t_x))
                            # 条件分岐: `not common_t` を満たす経路を評価する。
                            if not common_t:
                                use_plasma = False
                                tdf_plasma_fallback_reason = "no_common_bins"
                            else:
                                tdf_plasma_common_bins = int(len(common_t))
                                dual_binned_out = out_dir / "cassini_sce1_tdf_dual_binned.csv"
                                with dual_binned_out.open("w", newline="", encoding="utf-8") as f:
                                    w = csv.writer(f)
                                    w.writerow(
                                        [
                                            "t_days",
                                            "y_ka_mean",
                                            "n_ka",
                                            "y_x_mean",
                                            "n_x",
                                            "y_plasma_corrected_mean",
                                            "n",
                                        ]
                                    )

                                    plasma_series: List[Dict[str, float]] = []
                                    for t_days in common_t:
                                        rka = by_t_ka[float(t_days)]
                                        rx = by_t_x[float(t_days)]
                                        y_ka = float(rka["y_mean"])
                                        y_x = float(rx["y_mean"])
                                        n_ka = int(round(float(rka.get("n", 0.0))))
                                        n_x = int(round(float(rx.get("n", 0.0))))
                                        y_gr = (y_ka * (f_ka**2) - y_x * (f_x**2)) / denom
                                        n_common = min(n_ka, n_x) if (n_ka > 0 and n_x > 0) else max(n_ka, n_x)
                                        w.writerow(
                                            [
                                                f"{float(t_days):.15g}",
                                                f"{y_ka:.15e}",
                                                n_ka,
                                                f"{y_x:.15e}",
                                                n_x,
                                                f"{y_gr:.15e}",
                                                n_common,
                                            ]
                                        )
                                        plasma_series.append(
                                            {"t_days": float(t_days), "y_mean": float(y_gr), "n": float(n_common)}
                                        )

                                print("Wrote:", dual_binned_out)

                                # Build a merged binned series:
                                # - Use plasma-corrected y_gr when both Ka and X exist for the same bin.
                                # - Fill missing bins with single-band values (prefer Ka if present, else X).
                                #   This prevents the "t<0 side missing" artifact when Ka is absent before b_min.
                                merged_binned_out = out_dir / "cassini_sce1_tdf_binned_merged.csv"
                                merged_series: List[Dict[str, float]] = []
                                mode_counts: Dict[str, int] = {"dual": 0, "ka_only": 0, "x_only": 0}

                                with merged_binned_out.open("w", newline="", encoding="utf-8") as f:
                                    w = csv.writer(f)
                                    w.writerow(
                                        [
                                            "t_days",
                                            "mode",
                                            "y_used_mean",
                                            "n_used",
                                            "y_plasma_corrected_mean",
                                            "y_ka_mean",
                                            "n_ka",
                                            "y_x_mean",
                                            "n_x",
                                        ]
                                    )

                                    for t_days in merged_t:
                                        rka = by_t_ka.get(float(t_days))
                                        rx = by_t_x.get(float(t_days))

                                        y_ka = float(rka["y_mean"]) if rka is not None else math.nan
                                        y_x = float(rx["y_mean"]) if rx is not None else math.nan
                                        n_ka = int(round(float(rka.get("n", 0.0)))) if rka is not None else 0
                                        n_x = int(round(float(rx.get("n", 0.0)))) if rx is not None else 0

                                        y_gr: Optional[float] = None
                                        # 条件分岐: `(rka is not None) and (rx is not None)` を満たす経路を評価する。
                                        if (rka is not None) and (rx is not None):
                                            y_gr = float((y_ka * (f_ka**2) - y_x * (f_x**2)) / denom)
                                            y_used = float(y_gr)
                                            n_used = min(n_ka, n_x) if (n_ka > 0 and n_x > 0) else max(n_ka, n_x)
                                            mode = "dual"
                                        # 条件分岐: 前段条件が不成立で、`rka is not None` を追加評価する。
                                        elif rka is not None:
                                            y_used = float(y_ka)
                                            n_used = int(n_ka)
                                            mode = "ka_only"
                                        else:
                                            y_used = float(y_x)
                                            n_used = int(n_x)
                                            mode = "x_only"

                                        mode_counts[mode] = int(mode_counts.get(mode, 0)) + 1
                                        merged_series.append(
                                            {"t_days": float(t_days), "y_mean": float(y_used), "n": float(n_used)}
                                        )
                                        w.writerow(
                                            [
                                                f"{float(t_days):.15g}",
                                                mode,
                                                f"{float(y_used):.15e}",
                                                int(n_used),
                                                (f"{float(y_gr):.15e}" if y_gr is not None else ""),
                                                (f"{float(y_ka):.15e}" if math.isfinite(float(y_ka)) else ""),
                                                int(n_ka),
                                                (f"{float(y_x):.15e}" if math.isfinite(float(y_x)) else ""),
                                                int(n_x),
                                            ]
                                        )

                                print("Wrote:", merged_binned_out)

                                chosen_mode = str(getattr(args, "tdf_plasma_fill_mode", "dual")).strip().lower()
                                # 条件分岐: `chosen_mode not in ("dual", "merged")` を満たす経路を評価する。
                                if chosen_mode not in ("dual", "merged"):
                                    chosen_mode = "dual"

                                detrend_input = plasma_series if chosen_mode == "dual" else merged_series
                                detrended, detrend_info = _detrend_polynomial(
                                    detrend_input,
                                    poly_order=int(args.tdf_detrend_poly_order),
                                    exclude_inner_days=float(args.tdf_detrend_exclude_inner_days),
                                )
                                detrend_info = dict(detrend_info)
                                detrend_info["merged_mode_counts"] = dict(mode_counts)
                                detrend_info["plasma_fill_mode"] = str(chosen_mode)
                                tdf_plasma_used = True
                                tdf_effective_downlink_bands = [2, 3]
                                tdf_obs_for_extract = sorted([*tdf_obs_ka, *tdf_obs_x], key=lambda r: r.time_utc)
                                # 条件分岐: `chosen_mode == "dual"` を満たす経路を評価する。
                                if chosen_mode == "dual":
                                    obs_label = f"観測（一次データ: PDS TDF（Ka+Xプラズマ補正、共通ビンのみ{recon_tag}）→ 平滑化＋デトレンド）"
                                else:
                                    obs_label = f"観測（一次データ: PDS TDF（Ka+Xプラズマ補正、欠測は単独バンド{recon_tag}）→ 平滑化＋デトレンド）"

                                effective_source = "pds_tdf_processed_plasma"

            # 条件分岐: `not tdf_plasma_used` を満たす経路を評価する。

            if not tdf_plasma_used:
                # 条件分岐: `bool(tdf_plasma_requested) and not use_plasma` を満たす経路を評価する。
                if bool(tdf_plasma_requested) and not use_plasma:
                    # Plasma correction requested but not possible in current cache/run.
                    if tdf_plasma_fallback_reason is None:
                        # 条件分岐: `not has_ka or not has_x` を満たす経路を評価する。
                        if not has_ka or not has_x:
                            tdf_plasma_fallback_reason = "missing_ka_or_x_files"
                        else:
                            tdf_plasma_fallback_reason = "plasma_correction_disabled_or_failed"

                # 条件分岐: `want_downlink_band_id is None` を満たす経路を評価する。

                if want_downlink_band_id is None:
                    # "any" is ambiguous for processed pipelines; keep deterministic behavior.
                    want_downlink_band_id = 3

                tdf_obs = load_tdf_doppler_pseudoresiduals(
                    pds_root,
                    doy_start=int(args.doy_start),
                    doy_stop=int(args.doy_stop),
                    want_downlink_band_id=want_downlink_band_id,
                    want_sample_data_type_id=want_sample_id,
                    want_stations=stations,
                    want_doppler_channel_count=want_doppler_channel_count,
                    want_ground_mode=want_ground_mode,
                    fixed_point_frac_bits=int(args.tdf_fixed_frac_bits),
                    max_abs_pr_hz=None if float(args.tdf_max_abs_pr_hz) < 0 else float(args.tdf_max_abs_pr_hz),
                    require_pr_tolerance=bool(args.tdf_require_pr_tolerance),
                    exclude_slipped_cycles=bool(args.tdf_exclude_slipped_cycles),
                    exclude_pr_sentinel=bool(args.tdf_exclude_pr_sentinel),
                    exclude_rss_sentinel=bool(args.tdf_exclude_rss_sentinel),
                    y_sign=float(int(args.tdf_y_sign)),
                )
                tdf_obs, gm_eff, cc_eff = _auto_filter_tdf_obs_by_mode(
                    tdf_obs,
                    want_ground_mode=want_ground_mode,
                    want_doppler_channel_count=want_doppler_channel_count,
                )
                # 条件分岐: `gm_eff is not None` を満たす経路を評価する。
                if gm_eff is not None:
                    tdf_effective_ground_mode = int(gm_eff)

                # 条件分岐: `cc_eff is not None` を満たす経路を評価する。

                if cc_eff is not None:
                    tdf_effective_doppler_channel_count = int(cc_eff)
                    tdf_effective_doppler_channel_count_by_band[int(want_downlink_band_id)] = int(cc_eff)

                tdf_obs_for_extract = list(tdf_obs)
                tdf_effective_downlink_bands = [int(want_downlink_band_id)]

                raw_pairs: List[Tuple[float, float]] = []
                for r in tdf_obs:
                    raw_pairs.append((t_days_from_ref(r.time_utc, t_ref), float(r.y)))

                raw_pairs.sort(key=lambda p: p[0])

                binned = _bin_time_series(raw_pairs, bin_seconds=int(args.tdf_bin_seconds), stat=str(args.tdf_bin_stat))
                binned = _filter_bins_by_min_count(binned, min_count=int(args.tdf_min_bin_count))
                detrended, detrend_info = _detrend_polynomial(
                    binned,
                    poly_order=int(args.tdf_detrend_poly_order),
                    exclude_inner_days=float(args.tdf_detrend_exclude_inner_days),
                )
                obs_label = f"観測（一次データ: PDS TDF{recon_tag} → 平滑化＋デトレンド）"
                effective_source = "pds_tdf_processed"

            obs_out = out_dir / "cassini_sce1_tdf_extracted.csv"
            with obs_out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "time_utc",
                        "t_days",
                        "y_obs",
                        "doppler_pseudoresidual_hz",
                        "doppler_pseudoresidual_mhz",
                        "frequency_level_indicator",
                        "doppler_ref_receiver_frequency_hz",
                        "doppler_pr_tolerance",
                        "doppler_noise_tolerance",
                        "total_slipped_cycles",
                        "doppler_noise_mhz",
                        "received_signal_strength_centidbm",
                        "pr_sign4",
                        "pr_low32_u",
                        "pr_signed36",
                        "sample_interval_cs",
                        "doppler_channel_count",
                        "ground_mode",
                        "station_id",
                        "downlink_band_id",
                        "sample_data_type_id",
                        "source_file",
                    ]
                )
                for r in tdf_obs_for_extract:
                    w.writerow(
                        [
                            r.time_utc.isoformat(),
                            f"{t_days_from_ref(r.time_utc, t_ref):.15g}",
                            f"{r.y:.15e}",
                            f"{r.doppler_pseudoresidual_hz:.15e}",
                            f"{r.pr_mhz:.15e}",
                            r.frequency_level_indicator,
                            f"{r.doppler_ref_receiver_frequency_hz:.15e}",
                            r.doppler_pr_tolerance,
                            r.doppler_noise_tolerance,
                            r.total_slipped_cycles,
                            f"{r.doppler_noise_mhz:.15g}",
                            r.received_signal_strength_centidbm,
                            r.pr_sign4,
                            r.pr_low32_u,
                            r.pr_signed36,
                            r.sample_interval_cs,
                            r.doppler_channel_count,
                            r.ground_mode,
                            r.station_id,
                            r.downlink_band_id,
                            r.sample_data_type_id,
                            r.source_file,
                        ]
                    )

            print("Wrote:", obs_out)

            proc_out = out_dir / "cassini_sce1_tdf_paperlike.csv"
            ref_model_t_days: Optional[List[float]] = None
            ref_model_y: Optional[List[float]] = None
            # 条件分岐: `tdf_reconstruct_shapiro` を満たす経路を評価する。
            if tdf_reconstruct_shapiro:
                ref_model_t_days, ref_model_y = build_model_series(rows, beta=tdf_reconstruct_beta, mode=args.model)

            for r in detrended:
                t_days = float(r["t_days"])
                y_resid = float(r.get("y_detrended", r.get("y_mean", 0.0)))
                y_ref = (
                    interp_linear(ref_model_t_days, ref_model_y, t_days)
                    if (ref_model_t_days is not None and ref_model_y is not None)
                    else None
                )
                y_ref_f = float(y_ref) if (y_ref is not None and math.isfinite(float(y_ref))) else math.nan
                r["y_model_ref"] = y_ref_f
                # 条件分岐: `tdf_reconstruct_shapiro and math.isfinite(y_ref_f)` を満たす経路を評価する。
                if tdf_reconstruct_shapiro and math.isfinite(y_ref_f):
                    r["y_reconstructed"] = y_resid + y_ref_f
                else:
                    r["y_reconstructed"] = y_resid

            with proc_out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t_days", "y_mean", "n", "baseline", "y_detrended", "y_model_ref", "y_reconstructed"])
                for r in detrended:
                    w.writerow(
                        [
                            f"{float(r['t_days']):.15g}",
                            f"{float(r['y_mean']):.15e}",
                            int(round(float(r.get("n", 0.0)))),
                            f"{float(r.get('baseline', 0.0)):.15e}",
                            f"{float(r.get('y_detrended', r.get('y_mean', 0.0))):.15e}",
                            f"{float(r.get('y_model_ref', math.nan)):.15e}",
                            f"{float(r.get('y_reconstructed', r.get('y_detrended', r.get('y_mean', 0.0)))):.15e}",
                        ]
                    )

            print("Wrote:", proc_out)
            print("[info] detrend:", detrend_info)

            observed_pairs = [
                (float(r["t_days"]), float(r.get("y_reconstructed", r.get("y_detrended", r["y_mean"]))))
                for r in detrended
            ]
            generated_pds_series = True
            # 条件分岐: `not tdf_effective_downlink_bands` を満たす経路を評価する。
            if not tdf_effective_downlink_bands:
                # Keep a safe fallback for metadata even if something went wrong above.
                tdf_effective_downlink_bands = [3]

    observed_pairs.sort(key=lambda p: p[0])

    # Sanity check: compare our PDS-derived y(t) against the published digitized curve.
    # This helps validate that the PDS decoding + smoothing/detrend produces a paper-like y(t).
    # Note: PDS TDF provides Doppler pseudo-residual (observed - predicted); we add back a reference Shapiro model unless disabled.
    # comparable to what is typically shown in Cassini γ (Shapiro) analyses.
    if generated_pds_series:
        try:
            digitized_csv = (root / args.digitized_csv).resolve()
            digitized = load_digitized_points(digitized_csv)

            obs_t = [t for t, _y in observed_pairs]
            obs_y = [_y for _t, _y in observed_pairs]

            def _metrics_for_shift(shift_days: float) -> Dict[str, float]:
                diffs: List[float] = []
                ys_dig: List[float] = []
                ys_pds: List[float] = []
                for t_days, y_d in digitized:
                    # Align digitized curve (paper) to our b_min time definition by shifting the time axis.
                    y_p = interp_linear(obs_t, obs_y, float(t_days) + float(shift_days))
                    # 条件分岐: `y_p is None` を満たす経路を評価する。
                    if y_p is None:
                        continue

                    diffs.append(float(y_p - y_d))
                    ys_dig.append(float(y_d))
                    ys_pds.append(float(y_p))

                return {
                    "n": float(len(diffs)),
                    "rmse": float(rmse(diffs)) if diffs else math.nan,
                    "corr": float(pearson_corr(ys_dig, ys_pds)) if diffs else math.nan,
                }

            # (1) raw, no-shift comparison

            m0 = _metrics_for_shift(0.0)
            n0 = int(round(float(m0.get("n", 0.0))))
            # Avoid picking a shift that only matches a much smaller subset of points (edge artifacts).
            n_min = max(30, int(round(n0 * 0.9)))

            # (2) best time-shift comparison (min RMSE)
            best_shift_days = 0.0
            best = dict(m0)
            best_rmse = float(best.get("rmse", math.nan))
            for i in range(-200, 201):  # -10..+10 days in 0.05-day steps
                shift = i * 0.05
                m = _metrics_for_shift(float(shift))
                # 条件分岐: `int(round(float(m.get("n", 0.0)))) < n_min` を満たす経路を評価する。
                if int(round(float(m.get("n", 0.0)))) < n_min:
                    continue

                r = float(m.get("rmse", math.nan))
                # 条件分岐: `not math.isfinite(r)` を満たす経路を評価する。
                if not math.isfinite(r):
                    continue

                # 条件分岐: `(not math.isfinite(best_rmse)) or r < best_rmse` を満たす経路を評価する。

                if (not math.isfinite(best_rmse)) or r < best_rmse:
                    best_rmse = float(r)
                    best = dict(m)
                    best_shift_days = float(shift)
                # 条件分岐: 前段条件が不成立で、`r == best_rmse` を追加評価する。
                elif r == best_rmse:
                    # Tie-breaker: larger correlation (prefer same-shape alignment)
                    c = float(m.get("corr", math.nan))
                    cb = float(best.get("corr", math.nan))
                    # 条件分岐: `math.isfinite(c) and (not math.isfinite(cb) or c > cb)` を満たす経路を評価する。
                    if math.isfinite(c) and (not math.isfinite(cb) or c > cb):
                        best = dict(m)
                        best_shift_days = float(shift)

            rmse_d = float(best.get("rmse", math.nan))
            corr_d = float(best.get("corr", math.nan))
            n_d = int(round(float(best.get("n", 0.0))))

            m_out = out_dir / "cassini_pds_vs_digitized_metrics.csv"
            with m_out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "n",
                        "rmse",
                        "corr",
                        "shift_days",
                        "rmse_zero_shift",
                        "corr_zero_shift",
                        "n_zero_shift",
                    ]
                )
                w.writerow(
                    [
                        n_d,
                        f"{rmse_d:.15e}",
                        f"{corr_d:.15g}",
                        f"{best_shift_days:.3f}",
                        f"{float(m0.get('rmse', math.nan)):.15e}",
                        f"{float(m0.get('corr', math.nan)):.15g}",
                        int(round(float(m0.get('n', 0.0)))),
                    ]
                )

            print("Wrote:", m_out)

            # 条件分岐: `not args.no_plots` を満たす経路を評価する。
            if not args.no_plots:
                try:
                    import matplotlib.pyplot as plt  # type: ignore

                    _set_japanese_font()
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.scatter(obs_t, obs_y, s=12, alpha=0.75, color="tab:blue", label="PDS（処理後）")
                    ax.scatter(
                        [float(t) + float(best_shift_days) for t, _y in digitized],
                        [y for _t, y in digitized],
                        s=10,
                        alpha=0.8,
                        color="tab:orange",
                        label=f"論文図デジタイズ（時間シフト {best_shift_days:+.2f} 日）",
                    )
                    ax.set_title("Cassini：PDS一次データ（処理後） vs 論文図デジタイズ", fontsize=13)
                    ax.set_xlabel("t（日, b_min からの相対）")
                    ax.set_ylabel("y（周波数比）")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(out_dir / "cassini_pds_vs_digitized.png", dpi=180)
                    plt.close(fig)
                except Exception as e:
                    print(f"[warn] failed to plot PDS vs digitized: {e}")
        except Exception as e:
            print(f"[warn] failed to compute PDS vs digitized metrics: {e}")

    points = match_points(observed_pairs, model_t_days, model_y, align_offset=align_offset)

    points_out = out_dir / "cassini_fig2_matched_points.csv"
    metrics_out = out_dir / "cassini_fig2_metrics.csv"
    write_points_csv(points_out, points)
    write_metrics_csv(metrics_out, points)
    print("Wrote:", points_out)
    print("Wrote:", metrics_out)

    best_beta_by_rmse10: Optional[float] = None
    best_beta_zoom = None
    # 条件分岐: `not args.no_sweep` を満たす経路を評価する。
    if not args.no_sweep:
        sweep_csv = out_dir / "cassini_beta_sweep_rmse.csv"
        sweep_png = None if args.no_plots else (out_dir / "cassini_beta_sweep_rmse.png")
        best_beta, _rmse10 = run_beta_sweep(
            observed=observed_pairs,
            rows=rows,
            out_csv=sweep_csv,
            out_png=sweep_png,
            mode=args.model,
            beta_start=args.beta_start,
            beta_stop=args.beta_stop,
            beta_step=args.beta_step,
            align_offset=align_offset,
        )
        print("Wrote:", sweep_csv)
        # 条件分岐: `sweep_png is not None` を満たす経路を評価する。
        if sweep_png is not None:
            print("Wrote:", sweep_png)

        # 条件分岐: `best_beta is not None` を満たす経路を評価する。

        if best_beta is not None:
            best_beta_by_rmse10 = float(best_beta)
            _, best_model_y = build_model_series(rows, beta=best_beta, mode=args.model)
            best_beta_zoom = (best_beta, best_model_y)

    # 条件分岐: `not args.no_plots` を満たす経路を評価する。

    if not args.no_plots:
        try_plot(
            out_dir=out_dir,
            model_t_days=model_t_days,
            model_y=model_y,
            points=points,
            beta=beta,
            obs_label=obs_label,
            best_beta_zoom=best_beta_zoom,
        )

    # Persist run metadata so the summary report can accurately label the observation source.

    try:
        cassini_pds_root = root / "data" / "cassini" / "pds_sce1"
        meta = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "argv": list(sys.argv),
            "requested_source": requested_source,
            "effective_source": effective_source,
            "fallback_reason": fallback_reason,
            "params": {
                "beta": float(beta),
                "beta_source": str(beta_source),
                "model": str(args.model),
                "align_offset": bool(align_offset),
                "doy_start": int(args.doy_start),
                "doy_stop": int(args.doy_stop),
                "odf_downlink_band": str(args.odf_downlink_band),
                "tdf_downlink_band": str(args.tdf_downlink_band),
                "tdf_plasma_correct": bool(tdf_plasma_requested),
                "tdf_plasma_used": bool(tdf_plasma_used),
                "tdf_plasma_fallback_reason": tdf_plasma_fallback_reason,
                "tdf_effective_downlink_bands": list(tdf_effective_downlink_bands),
                "tdf_plasma_common_bins": tdf_plasma_common_bins,
                "tdf_plasma_common_station_ids": list(tdf_plasma_common_station_ids),
                "tdf_sample_type": str(args.tdf_sample_type),
                "tdf_ground_mode": int(args.tdf_ground_mode),
                "tdf_doppler_channel_count": int(args.tdf_doppler_channel_count),
                "tdf_effective_ground_mode": tdf_effective_ground_mode,
                "tdf_effective_doppler_channel_count": tdf_effective_doppler_channel_count,
                "tdf_effective_doppler_channel_count_by_band": dict(tdf_effective_doppler_channel_count_by_band),
                "tdf_fixed_frac_bits": int(args.tdf_fixed_frac_bits),
                "tdf_y_sign": int(args.tdf_y_sign),
                "tdf_max_abs_pr_hz": float(args.tdf_max_abs_pr_hz),
                "tdf_require_pr_tolerance": bool(args.tdf_require_pr_tolerance),
                "tdf_exclude_slipped_cycles": bool(args.tdf_exclude_slipped_cycles),
                "tdf_exclude_pr_sentinel": bool(args.tdf_exclude_pr_sentinel),
                "tdf_exclude_rss_sentinel": bool(args.tdf_exclude_rss_sentinel),
                "tdf_plasma_fill_mode": str(args.tdf_plasma_fill_mode),
                "tdf_bin_seconds": int(args.tdf_bin_seconds),
                "tdf_bin_stat": str(args.tdf_bin_stat),
                "tdf_min_bin_count": int(args.tdf_min_bin_count),
                "tdf_detrend_poly_order": int(args.tdf_detrend_poly_order),
                "tdf_detrend_exclude_inner_days": float(args.tdf_detrend_exclude_inner_days),
                "tdf_reconstruct_shapiro": bool(tdf_reconstruct_shapiro),
                "tdf_reconstruct_beta": float(tdf_reconstruct_beta),
                "tdf_reconstruct_beta_source": str(tdf_reconstruct_beta_source),
            },
            "inputs": {
                "model_csv": str(model_csv),
                "digitized_csv": str((root / args.digitized_csv).resolve()),
                "pds_root": str(cassini_pds_root),
                "pds_manifest_odf": str((cassini_pds_root / "manifest_odf.json")),
                "pds_manifest_tdf": str((cassini_pds_root / "manifest_tdf.json")),
            },
            "outputs": {
                "points_csv": str(points_out),
                "metrics_csv": str(metrics_out),
                "best_beta_by_rmse10": best_beta_by_rmse10,
                "pds_extracted_csv": str(out_dir / "cassini_sce1_tdf_extracted.csv"),
                "pds_processed_csv": str(out_dir / "cassini_sce1_tdf_paperlike.csv"),
                "pds_dual_binned_csv": str(dual_binned_out) if dual_binned_out is not None else None,
                "pds_binned_merged_csv": str(merged_binned_out) if merged_binned_out is not None else None,
                "pds_vs_digitized_metrics_csv": str(out_dir / "cassini_pds_vs_digitized_metrics.csv"),
            },
            "counts": {
                "n_observed_points": int(len(observed_pairs)),
                "n_matched_points": int(len(points)),
            },
            "labels": {
                "obs_label": str(obs_label),
            },
        }
        meta_path = out_dir / "cassini_fig2_run_metadata.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Wrote:", meta_path)
    except Exception as e:
        print(f"[warn] failed to write Cassini run metadata: {e}")

    try:
        def _safe_rel(p: Path) -> str:
            try:
                return str(p.relative_to(root)).replace("\\", "/")
            except Exception:
                return str(p).replace("\\", "/")

        def _window_metrics(window_days: Optional[float]) -> Dict[str, float]:
            subset = compute_window(points, window_days)
            resids = [p.residual for p in subset]
            yd = [p.y_obs for p in subset]
            ym = [p.y_model for p in subset]
            return {
                "n": float(len(subset)),
                "rmse": float(rmse(resids)) if resids else math.nan,
                "corr": float(pearson_corr(yd, ym)) if resids else math.nan,
            }

        m_all = _window_metrics(None)
        m_10 = _window_metrics(10.0)
        m_3 = _window_metrics(3.0)

        worklog.append_event(
            {
                "event_type": "cassini_fig2_overlay",
                "argv": sys.argv,
                "inputs": {
                    "model_csv": _safe_rel(model_csv),
                    "pds_root": _safe_rel(root / "data" / "cassini" / "pds_sce1"),
                },
                "params": {
                    "beta": float(beta),
                    "beta_source": str(beta_source),
                    "model": str(args.model),
                    "source_requested": str(requested_source),
                    "source_effective": str(effective_source),
                    "tdf_plasma_correct": bool(tdf_plasma_requested),
                    "tdf_plasma_used": bool(tdf_plasma_used),
                    "tdf_plasma_fallback_reason": tdf_plasma_fallback_reason,
                    "tdf_effective_downlink_bands": list(tdf_effective_downlink_bands),
                    "tdf_plasma_common_bins": tdf_plasma_common_bins,
                    "tdf_plasma_common_station_ids": list(tdf_plasma_common_station_ids),
                    "tdf_plasma_fill_mode": str(args.tdf_plasma_fill_mode),
                    "tdf_sample_type": str(args.tdf_sample_type),
                    "tdf_ground_mode": int(args.tdf_ground_mode),
                    "tdf_doppler_channel_count": int(args.tdf_doppler_channel_count),
                    "tdf_effective_ground_mode": tdf_effective_ground_mode,
                    "tdf_effective_doppler_channel_count": tdf_effective_doppler_channel_count,
                    "tdf_effective_doppler_channel_count_by_band": dict(tdf_effective_doppler_channel_count_by_band),
                    "tdf_fixed_frac_bits": int(args.tdf_fixed_frac_bits),
                    "tdf_y_sign": int(args.tdf_y_sign),
                    "tdf_max_abs_pr_hz": float(args.tdf_max_abs_pr_hz),
                    "tdf_require_pr_tolerance": bool(args.tdf_require_pr_tolerance),
                    "tdf_exclude_slipped_cycles": bool(args.tdf_exclude_slipped_cycles),
                    "tdf_exclude_pr_sentinel": bool(args.tdf_exclude_pr_sentinel),
                    "tdf_exclude_rss_sentinel": bool(args.tdf_exclude_rss_sentinel),
                    "tdf_bin_seconds": int(args.tdf_bin_seconds),
                    "tdf_bin_stat": str(args.tdf_bin_stat),
                    "tdf_min_bin_count": int(args.tdf_min_bin_count),
                    "tdf_detrend_poly_order": int(args.tdf_detrend_poly_order),
                    "tdf_detrend_exclude_inner_days": float(args.tdf_detrend_exclude_inner_days),
                    "tdf_reconstruct_shapiro": bool(tdf_reconstruct_shapiro),
                    "tdf_reconstruct_beta": float(tdf_reconstruct_beta),
                    "tdf_reconstruct_beta_source": str(tdf_reconstruct_beta_source),
                },
                "metrics": {
                    "all": m_all,
                    "pm10d": m_10,
                    "pm3d": m_3,
                },
                "outputs": {
                    "out_dir": out_dir,
                    "points_csv": points_out,
                    "metrics_csv": metrics_out,
                    "overlay_zoom10d_png": out_dir / "cassini_fig2_overlay_zoom10d.png",
                    "residuals_png": out_dir / "cassini_fig2_residuals.png",
                    "pds_extracted_csv": out_dir / "cassini_sce1_tdf_extracted.csv",
                    "pds_processed_csv": out_dir / "cassini_sce1_tdf_paperlike.csv",
                    "pds_dual_binned_csv": dual_binned_out,
                    "pds_binned_merged_csv": merged_binned_out,
                    "run_metadata_json": out_dir / "cassini_fig2_run_metadata.json",
                },
            }
        )
    except Exception:
        pass


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
