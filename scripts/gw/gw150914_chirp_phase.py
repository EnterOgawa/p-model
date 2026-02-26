from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import sys
import urllib.request
import urllib.error
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, filtfilt, hilbert, stft, welch  # noqa: E402

try:  # numpy>=2.0 moved RankWarning to numpy.exceptions
    from numpy.exceptions import RankWarning as _NP_RANK_WARNING  # type: ignore
except Exception:  # pragma: no cover
    _NP_RANK_WARNING = getattr(np, "RankWarning", Warning)

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _download(url: str, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not force` を満たす経路を評価する。
    if dst.exists() and not force:
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-gw/1.0"})
    print(f"[dl] {url}")
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")


def _normalize_gwosc_version(version: str) -> str:
    v = (version or "").strip()
    # 条件分岐: `not v` を満たす経路を評価する。
    if not v:
        return "v3"

    # 条件分岐: `not v.lower().startswith("v")` を満たす経路を評価する。

    if not v.lower().startswith("v"):
        v = "v" + v

    return v


def _candidate_gwosc_versions(version: str) -> List[str]:
    v = (version or "").strip().lower()
    # 条件分岐: `not v or v == "auto"` を満たす経路を評価する。
    if not v or v == "auto":
        return ["v3", "v2", "v1"]

    return [_normalize_gwosc_version(version)]


def _gwosc_event_json_url(*, catalog: str, event: str, version: str) -> str:
    cat = (catalog or "").strip() or "GWTC-1-confident"
    ev = (event or "").strip()
    v = _normalize_gwosc_version(version)
    return f"https://gwosc.org/eventapi/json/{cat}/{ev}/{v}"


def _pick_event_from_eventapi_json(obj: Dict[str, Any], *, event: str) -> Tuple[str, Dict[str, Any]]:
    events = obj.get("events") or {}
    # 条件分岐: `not isinstance(events, dict) or not events` を満たす経路を評価する。
    if not isinstance(events, dict) or not events:
        raise ValueError("invalid event JSON: missing 'events'")

    target = (event or "").strip()
    target_low = target.lower()

    # Prefer exact key match (some APIs may use GW150914 as the key).
    if target in events and isinstance(events[target], dict):
        return target, events[target]

    # Prefer commonName match (GW150914-v3 is a typical key).

    for k, v in events.items():
        # 条件分岐: `not isinstance(v, dict)` を満たす経路を評価する。
        if not isinstance(v, dict):
            continue

        common = str(v.get("commonName") or "").strip()
        # 条件分岐: `common and common.lower() == target_low` を満たす経路を評価する。
        if common and common.lower() == target_low:
            return str(k), v

    # Fallback: the first event in the payload.

    k0 = next(iter(events.keys()))
    v0 = events[k0]
    # 条件分岐: `not isinstance(v0, dict)` を満たす経路を評価する。
    if not isinstance(v0, dict):
        raise ValueError("invalid event JSON: first event is not an object")

    return str(k0), v0


def _pick_strain_entry(
    strain_list: List[Dict[str, Any]],
    *,
    detector: str,
    prefer_duration_s: int,
    prefer_sampling_rate_hz: int,
) -> Optional[Dict[str, Any]]:
    det = (detector or "").strip()
    # 条件分岐: `not det` を満たす経路を評価する。
    if not det:
        return None

    cand = [e for e in (strain_list or []) if isinstance(e, dict) and str(e.get("detector") or "") == det]
    cand = [e for e in cand if str(e.get("format") or "").strip().lower() == "txt" and str(e.get("url") or "").strip()]
    # 条件分岐: `not cand` を満たす経路を評価する。
    if not cand:
        return None

    # Prefer the usual 32s snippet for quick checks.

    def _int(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    cand_d = [e for e in cand if _int(e.get("duration")) == int(prefer_duration_s)]
    # 条件分岐: `not cand_d` を満たす経路を評価する。
    if not cand_d:
        cand_d = sorted(cand, key=lambda e: (_int(e.get("duration")) or 10**9, _int(e.get("sampling_rate")) or 10**9))[:1]

    cand_sr = [e for e in cand_d if _int(e.get("sampling_rate")) == int(prefer_sampling_rate_hz)]
    # 条件分岐: `cand_sr` を満たす経路を評価する。
    if cand_sr:
        return cand_sr[0]

    # Fallback: smallest sampling rate among the chosen duration bucket.

    return sorted(cand_d, key=lambda e: _int(e.get("sampling_rate")) or 10**9)[0]


def _fetch_inputs(
    data_dir: Path,
    *,
    event: str,
    catalog: str,
    version: str,
    detectors: List[str],
    prefer_duration_s: int,
    prefer_sampling_rate_hz: int,
    offline: bool,
    force: bool,
) -> Dict[str, Any]:
    """
    Cache event JSON + strain TXT.GZ data from GWOSC.
    """
    event_name = (event or "").strip()
    # 条件分岐: `not event_name` を満たす経路を評価する。
    if not event_name:
        raise ValueError("event name is required")

    event_json_path = data_dir / f"{event_name}_event.json"

    data_dir.mkdir(parents=True, exist_ok=True)
    event_json_url = ""
    v_used = ""
    # 条件分岐: `offline` を満たす経路を評価する。
    if offline:
        # 条件分岐: `not event_json_path.exists()` を満たす経路を評価する。
        if not event_json_path.exists():
            raise FileNotFoundError("offline and missing: event_json")
    else:
        last_404: Optional[Exception] = None
        for v_try in _candidate_gwosc_versions(version):
            url_try = _gwosc_event_json_url(catalog=catalog, event=event_name, version=v_try)
            try:
                _download(url_try, event_json_path, force=force)
                event_json_url = url_try
                v_used = v_try
                break
            except urllib.error.HTTPError as e:
                # 条件分岐: `int(getattr(e, "code", 0) or 0) == 404 and len(_candidate_gwosc_versions(vers...` を満たす経路を評価する。
                if int(getattr(e, "code", 0) or 0) == 404 and len(_candidate_gwosc_versions(version)) > 1:
                    last_404 = e
                    continue

                raise
        else:
            raise last_404 or RuntimeError("failed to fetch event JSON (all candidate versions failed)")

    ev_obj = json.loads(event_json_path.read_text(encoding="utf-8"))
    event_key, event_info = _pick_event_from_eventapi_json(ev_obj, event=event_name)
    v_from_json = event_info.get("version")
    # 条件分岐: `v_from_json is not None` を満たす経路を評価する。
    if v_from_json is not None:
        v_used = _normalize_gwosc_version(str(v_from_json))

    # 条件分岐: `not v_used` を満たす経路を評価する。

    if not v_used:
        v_used = _candidate_gwosc_versions(version)[0]

    # 条件分岐: `not event_json_url` を満たす経路を評価する。

    if not event_json_url:
        event_json_url = _gwosc_event_json_url(catalog=catalog, event=event_name, version=v_used)

    strain = event_info.get("strain") or []
    # 条件分岐: `not isinstance(strain, list)` を満たす経路を評価する。
    if not isinstance(strain, list):
        strain = []

    strain_list = [e for e in strain if isinstance(e, dict)]

    selected: Dict[str, Dict[str, Any]] = {}
    strain_paths: Dict[str, Path] = {}
    strain_urls: Dict[str, str] = {}
    missing: List[str] = []

    for det in detectors:
        entry = _pick_strain_entry(
            strain_list,
            detector=det,
            prefer_duration_s=int(prefer_duration_s),
            prefer_sampling_rate_hz=int(prefer_sampling_rate_hz),
        )
        # 条件分岐: `not entry` を満たす経路を評価する。
        if not entry:
            continue

        url = str(entry.get("url") or "").strip()
        # 条件分岐: `not url` を満たす経路を評価する。
        if not url:
            continue

        fname = Path(url.split("?", 1)[0].split("#", 1)[0]).name
        p = data_dir / fname
        # 条件分岐: `offline` を満たす経路を評価する。
        if offline:
            # 条件分岐: `not p.exists()` を満たす経路を評価する。
            if not p.exists():
                missing.append(f"{det}:{fname}")
                continue
        else:
            _download(url, p, force=force)

        strain_paths[det] = p
        strain_urls[det] = url
        selected[det] = {
            "detector": det,
            "GPSstart": entry.get("GPSstart"),
            "sampling_rate": entry.get("sampling_rate"),
            "duration": entry.get("duration"),
            "format": entry.get("format"),
            "url": url,
            "path": str(p).replace("\\", "/"),
        }

    # 条件分岐: `offline and missing` を満たす経路を評価する。

    if offline and missing:
        raise FileNotFoundError("offline and missing strain files: " + ", ".join(missing))

    meta = {
        "generated_utc": _iso_utc_now(),
        "source": f"GWOSC ({(catalog or '').strip() or 'GWTC-1-confident'} / {event_name} / {v_used})",
        "urls": {"event_json": event_json_url, **{f"{k}_strain": u for k, u in strain_urls.items()}},
        "files": {
            "event_json": {"path": str(event_json_path).replace("\\", "/"), "sha256": _sha256(event_json_path)},
            **{k: {"path": str(p).replace("\\", "/"), "sha256": _sha256(p)} for k, p in strain_paths.items()},
        },
        "selection": {
            "event_key": event_key,
            "event_common_name": str(event_info.get("commonName") or ""),
            "api_version": str(v_used),
            "detectors_requested": detectors,
            "prefer_duration_s": int(prefer_duration_s),
            "prefer_sampling_rate_hz": int(prefer_sampling_rate_hz),
            "selected": selected,
        },
        "note": "TXT.GZ はサンプル列（時間列は header の GPSstart とサンプルレートから再構成）。",
    }
    (data_dir / f"{event_name.lower()}_sources.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"paths": {"event_json": event_json_path, "strain": strain_paths}, "meta": meta, "event_info": event_info}


def _parse_gwosc_txt_gz(path: Path) -> Tuple[float, float, np.ndarray]:
    """
    Return (gps_start, fs_hz, strain) for GWOSC TXT.GZ.
    """
    gps_start = float("nan")
    fs = float("nan")
    header_lines: List[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for _ in range(12):
            line = f.readline()
            # 条件分岐: `not line` を満たす経路を評価する。
            if not line:
                break

            # 条件分岐: `not line.startswith("#")` を満たす経路を評価する。

            if not line.startswith("#"):
                break

            header_lines.append(line.strip())

    for ln in header_lines:
        # 条件分岐: `"samples per second" in ln` を満たす経路を評価する。
        if "samples per second" in ln:
            parts = ln.split()
            for i, tok in enumerate(parts):
                # 条件分岐: `tok.isdigit()` を満たす経路を評価する。
                if tok.isdigit():
                    fs = float(tok)
                    break

        # 条件分岐: `"starting GPS" in ln` を満たす経路を評価する。

        if "starting GPS" in ln:
            # "# starting GPS 1126259447 duration 32"
            parts = ln.replace("#", "").split()
            for i, tok in enumerate(parts):
                # 条件分岐: `tok == "GPS" and i + 1 < len(parts)` を満たす経路を評価する。
                if tok == "GPS" and i + 1 < len(parts):
                    try:
                        gps_start = float(parts[i + 1])
                    except Exception:
                        pass

                    break

    # 条件分岐: `not math.isfinite(fs) or fs <= 0` を満たす経路を評価する。

    if not math.isfinite(fs) or fs <= 0:
        raise ValueError(f"failed to parse fs from header: {path}")

    # 条件分岐: `not math.isfinite(gps_start)` を満たす経路を評価する。

    if not math.isfinite(gps_start):
        # Fallback: parse from filename "...-GPSSTART-DURATION.txt.gz"
        toks = path.name.split("-")
        # 条件分岐: `len(toks) >= 3` を満たす経路を評価する。
        if len(toks) >= 3:
            try:
                gps_start = float(toks[-3])
            except Exception:
                pass

    # 条件分岐: `not math.isfinite(gps_start)` を満たす経路を評価する。

    if not math.isfinite(gps_start):
        raise ValueError(f"failed to parse GPSstart from header/filename: {path}")

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        strain = np.loadtxt(f, comments="#", dtype=np.float64)

    return gps_start, fs, strain


def _bandpass(x: np.ndarray, fs: float, *, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    lo = float(f_lo) / nyq
    hi = float(f_hi) / nyq
    # 条件分岐: `not (0 < lo < hi < 1)` を満たす経路を評価する。
    if not (0 < lo < hi < 1):
        raise ValueError(f"invalid bandpass: {f_lo}-{f_hi} Hz (fs={fs})")

    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, x)


def _whiten_fft(
    x: np.ndarray,
    fs: float,
    *,
    f_lo: float,
    f_hi: float,
    welch_nperseg: int = 4096,
) -> np.ndarray:
    """
    Rough frequency-domain whitening + band-limiting.

    - PSD via Welch.
    - Divide FFT by sqrt(PSD).
    - Zero out frequencies outside [f_lo, f_hi].
    """
    # 条件分岐: `x.size < 4` を満たす経路を評価する。
    if x.size < 4:
        return x.copy()

    # 条件分岐: `not (math.isfinite(fs) and fs > 0)` を満たす経路を評価する。

    if not (math.isfinite(fs) and fs > 0):
        raise ValueError("invalid fs")

    nperseg = int(welch_nperseg)
    # 条件分岐: `nperseg <= 0` を満たす経路を評価する。
    if nperseg <= 0:
        nperseg = 4096

    nperseg = min(nperseg, int(x.size))

    f_psd, pxx = welch(x, fs=float(fs), nperseg=nperseg)
    pxx = np.asarray(pxx, dtype=np.float64)
    pxx = np.maximum(pxx, 1e-24)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(int(x.size), d=1.0 / float(fs))
    psd_i = np.interp(freqs, np.asarray(f_psd, dtype=np.float64), pxx)
    psd_i = np.maximum(psd_i, 1e-24)
    W = X / np.sqrt(psd_i)

    lo = float(f_lo)
    hi = float(f_hi)
    # 条件分岐: `not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi)` を満たす経路を評価する。
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError(f"invalid band: {f_lo}-{f_hi}")

    mask = (freqs >= lo) & (freqs <= hi)
    W2 = np.zeros_like(W)
    W2[mask] = W[mask]
    return np.fft.irfft(W2, n=int(x.size))


def _extract_frequency_track_stft_guided(
    t: np.ndarray,
    x: np.ndarray,
    fs: float,
    *,
    t_window: Tuple[float, float],
    f_range: Tuple[float, float],
    mc_min_msun: float,
    mc_max_msun: float,
    mc_steps: int,
    delta_hz: float,
    amp_percentile: float,
    nperseg: int,
    noverlap: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Guided STFT track extraction (BNS向け):
      - STFTを作り、周波数ごとの定常成分（線）を中央値正規化で抑える
      - 四重極チャープ則の f_pred(t; Mc) に沿う信号が最大になる Mc を粗探索
      - 得られた Mc に沿って、各時刻ビンで ±delta の範囲の最大点を f(t) として抽出
    """
    # 条件分岐: `nperseg <= 0` を満たす経路を評価する。
    if nperseg <= 0:
        raise ValueError("nperseg must be positive")

    # 条件分岐: `noverlap < 0 or noverlap >= nperseg` を満たす経路を評価する。

    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must be in [0, nperseg)")

    # 条件分岐: `mc_steps < 5` を満たす経路を評価する。

    if mc_steps < 5:
        mc_steps = 5

    # 条件分岐: `not (mc_max_msun > mc_min_msun > 0)` を満たす経路を評価する。

    if not (mc_max_msun > mc_min_msun > 0):
        raise ValueError("invalid chirp-mass range")

    f, tt, Z = stft(
        x,
        fs=float(fs),
        window="hann",
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        boundary=None,
        padded=False,
    )

    # STFT times are seconds from series start; convert to event-relative seconds.
    t0 = float(t[0])
    t_stft = t0 + tt

    time_mask = (t_stft >= float(t_window[0])) & (t_stft <= float(t_window[1]))
    freq_mask = (f >= float(f_range[0])) & (f <= float(f_range[1]))
    # 条件分岐: `(not np.any(time_mask)) or (not np.any(freq_mask))` を満たす経路を評価する。
    if (not np.any(time_mask)) or (not np.any(freq_mask)):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), {"ok": 0.0, "reason": "empty_window"}

    mag = np.abs(Z)
    mag_tf = mag[np.ix_(freq_mask, time_mask)]  # [freq, time]
    f_sub = f[freq_mask]
    t_sub = t_stft[time_mask]

    # Suppress stationary lines: normalize magnitude per frequency by median over time.
    baseline = np.median(mag_tf, axis=1)
    mag_n = mag_tf / (baseline[:, None] + 1e-12)

    # chirp prediction: f(t) = (K/(tc-t))^{3/8} / pi, with tc ~ 0 (event GPS time).
    G = 6.67430e-11
    c = 299792458.0
    M_sun = 1.988409870698051e30

    dt = -t_sub  # since tc~0
    dt = np.asarray(dt, dtype=np.float64)
    valid_t = np.isfinite(dt) & (dt > 0)
    # 条件分岐: `not np.any(valid_t)` を満たす経路を評価する。
    if not np.any(valid_t):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), {"ok": 0.0, "reason": "nonpositive_dt"}

    mc_grid = np.linspace(float(mc_min_msun), float(mc_max_msun), int(mc_steps), dtype=np.float64)
    delta = float(delta_hz)
    # 条件分岐: `not (math.isfinite(delta) and delta > 0)` を満たす経路を評価する。
    if not (math.isfinite(delta) and delta > 0):
        delta = 25.0

    scores = np.zeros_like(mc_grid, dtype=np.float64)
    for i, mc_msun in enumerate(mc_grid):
        mc = float(mc_msun) * M_sun
        K = (5.0 / 256.0) * ((G * mc / c**3) ** (-5.0 / 3.0))
        f_pred = np.full_like(dt, np.nan, dtype=np.float64)
        f_pred[valid_t] = np.power(K / dt[valid_t], 3.0 / 8.0) / math.pi

        score = 0.0
        for j, fp in enumerate(f_pred):
            # 条件分岐: `not math.isfinite(fp)` を満たす経路を評価する。
            if not math.isfinite(fp):
                continue

            lo = fp - delta
            hi = fp + delta
            idx = np.where((f_sub >= lo) & (f_sub <= hi))[0]
            # 条件分岐: `idx.size == 0` を満たす経路を評価する。
            if idx.size == 0:
                continue

            score += float(np.max(mag_n[idx, int(j)]))

        scores[int(i)] = float(score)

    best_i = int(np.nanargmax(scores)) if np.any(np.isfinite(scores)) else 0
    mc_best = float(mc_grid[best_i])

    # Extract ridge near predicted f(t) for best Mc.
    mc = mc_best * M_sun
    K = (5.0 / 256.0) * ((G * mc / c**3) ** (-5.0 / 3.0))
    f_pred = np.full_like(dt, np.nan, dtype=np.float64)
    f_pred[valid_t] = np.power(K / dt[valid_t], 3.0 / 8.0) / math.pi

    f_sel = np.full_like(t_sub, np.nan, dtype=np.float64)
    a_sel = np.full_like(t_sub, np.nan, dtype=np.float64)
    for j, fp in enumerate(f_pred):
        # 条件分岐: `not math.isfinite(fp)` を満たす経路を評価する。
        if not math.isfinite(fp):
            continue

        lo = fp - delta
        hi = fp + delta
        idx = np.where((f_sub >= lo) & (f_sub <= hi))[0]
        # 条件分岐: `idx.size == 0` を満たす経路を評価する。
        if idx.size == 0:
            continue

        local = mag_n[idx, int(j)]
        k_local = int(np.nanargmax(local))
        k = int(idx[k_local])
        f_sel[int(j)] = float(f_sub[k])
        a_sel[int(j)] = float(local[k_local])

    keep = np.isfinite(f_sel) & np.isfinite(t_sub)
    # 条件分岐: `np.any(keep)` を満たす経路を評価する。
    if np.any(keep):
        # Optional amplitude percentile cut on the guided ridge (keep enough points for fit).
        try:
            cut = float(np.percentile(a_sel[keep], float(amp_percentile)))
            keep2 = keep & (a_sel >= cut)
            # 条件分岐: `int(np.sum(keep2)) >= 10` を満たす経路を評価する。
            if int(np.sum(keep2)) >= 10:
                keep = keep2
        except Exception:
            pass

    meta = {
        "ok": 1.0 if int(np.sum(keep)) >= 10 else 0.0,
        "mc_best_msun": mc_best,
        "mc_grid_msun": [float(mc_min_msun), float(mc_max_msun), int(mc_steps)],
        "delta_hz": float(delta),
        "n_time_bins": int(len(t_sub)),
        "n_track": int(np.sum(keep)),
    }

    return t_sub[keep], f_sel[keep], meta


def _extract_instantaneous_frequency(
    t: np.ndarray,
    x: np.ndarray,
    fs: float,
    *,
    t_window: Tuple[float, float],
    f_range: Tuple[float, float],
    amp_percentile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / float(fs)

    analytic = hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    f_inst = np.diff(phase) / (2.0 * math.pi * dt)
    t_mid = 0.5 * (t[1:] + t[:-1])
    amp = np.abs(analytic[:-1])

    mask = (t_mid >= float(t_window[0])) & (t_mid <= float(t_window[1]))
    mask &= np.isfinite(f_inst)
    mask &= (f_inst >= float(f_range[0])) & (f_inst <= float(f_range[1]))

    # 条件分岐: `not np.any(mask)` を満たす経路を評価する。
    if not np.any(mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    amp_cut = np.percentile(amp[mask], float(amp_percentile))
    mask &= amp >= amp_cut

    t_sel = t_mid[mask]
    f_sel = f_inst[mask]
    return t_sel, f_sel


def _extract_frequency_track_stft(
    t: np.ndarray,
    x: np.ndarray,
    fs: float,
    *,
    t_window: Tuple[float, float],
    f_range: Tuple[float, float],
    amp_percentile: float,
    nperseg: int,
    noverlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a rough f(t) track via STFT ridge (peak frequency per time bin).

    This is usually more stable than Hilbert instantaneous frequency on noisy data.
    """
    # 条件分岐: `nperseg <= 0` を満たす経路を評価する。
    if nperseg <= 0:
        raise ValueError("nperseg must be positive")

    # 条件分岐: `noverlap < 0 or noverlap >= nperseg` を満たす経路を評価する。

    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must be in [0, nperseg)")

    f, tt, Z = stft(
        x,
        fs=float(fs),
        window="hann",
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        boundary=None,
        padded=False,
    )

    # STFT times are seconds from series start; convert to event-relative seconds.
    t0 = float(t[0])
    t_stft = t0 + tt

    time_mask = (t_stft >= float(t_window[0])) & (t_stft <= float(t_window[1]))
    freq_mask = (f >= float(f_range[0])) & (f <= float(f_range[1]))
    # 条件分岐: `(not np.any(time_mask)) or (not np.any(freq_mask))` を満たす経路を評価する。
    if (not np.any(time_mask)) or (not np.any(freq_mask)):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    mag = np.abs(Z)
    mag_tf = mag[np.ix_(freq_mask, time_mask)]  # [freq, time]
    max_mag = np.max(mag_tf, axis=0)
    finite = np.isfinite(max_mag)
    # 条件分岐: `not np.any(finite)` を満たす経路を評価する。
    if not np.any(finite):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    amp_cut = np.percentile(max_mag[finite], float(amp_percentile))
    keep_local = np.where(max_mag >= amp_cut)[0]
    # 条件分岐: `keep_local.size == 0` を満たす経路を評価する。
    if keep_local.size == 0:
        keep_local = np.argsort(max_mag)[-min(10, max_mag.size) :]

    f_sel: List[float] = []
    t_sel: List[float] = []
    f_sub = f[freq_mask]
    t_sub = t_stft[time_mask]
    for j in keep_local:
        col = mag_tf[:, int(j)]
        # 条件分岐: `col.size == 0` を満たす経路を評価する。
        if col.size == 0:
            continue

        k = int(np.nanargmax(col))
        f_peak = float(f_sub[k])
        t_peak = float(t_sub[int(j)])
        # 条件分岐: `not (math.isfinite(f_peak) and math.isfinite(t_peak))` を満たす経路を評価する。
        if not (math.isfinite(f_peak) and math.isfinite(t_peak)):
            continue

        f_sel.append(f_peak)
        t_sel.append(t_peak)

    # 条件分岐: `not t_sel` を満たす経路を評価する。

    if not t_sel:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    order = np.argsort(np.array(t_sel, dtype=np.float64))
    t_out = np.array([t_sel[i] for i in order], dtype=np.float64)
    f_out = np.array([f_sel[i] for i in order], dtype=np.float64)
    return t_out, f_out


def _fit_chirp_mass_from_track(t: np.ndarray, f: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Fit t = tc - A * f^{-8/3} and derive chirp mass from A.
    """
    min_points = 10
    # 条件分岐: `len(t) < min_points` を満たす経路を評価する。
    if len(t) < min_points:
        return {"ok": 0.0, "reason": "n_track_too_short", "n_total": float(len(t))}, np.zeros(len(t), dtype=bool)

    x_all = np.power(f, -8.0 / 3.0)
    y_all = t
    base = np.isfinite(x_all) & np.isfinite(y_all) & (f > 0)
    # 条件分岐: `int(np.sum(base)) < min_points` を満たす経路を評価する。
    if int(np.sum(base)) < min_points:
        return {
            "ok": 0.0,
            "reason": "n_valid_points_too_short",
            "n_total": float(len(t)),
            "n_base": float(int(np.sum(base))),
        }, np.zeros(len(t), dtype=bool)

    x = x_all[base]
    y = y_all[base]

    def _polyfit1(x1: np.ndarray, y1: np.ndarray) -> Tuple[float, float, bool]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", _NP_RANK_WARNING)
            m1, b1 = np.polyfit(x1, y1, 1)

        had_warn = any(getattr(ww, "category", None) is not None and issubclass(ww.category, _NP_RANK_WARNING) for ww in w)
        return float(m1), float(b1), bool(had_warn)

    # Robust inlier selection: iteratively reject outliers in (t, f^{-8/3}) linear space.

    inliers = np.ones(len(x), dtype=bool)
    sigma_s = float("nan")
    rank_warn = False
    for _ in range(3):
        # 条件分岐: `int(np.sum(inliers)) < min_points` を満たす経路を評価する。
        if int(np.sum(inliers)) < min_points:
            break

        m, b, w1 = _polyfit1(x[inliers], y[inliers])
        rank_warn = rank_warn or w1
        y_hat = m * x + b
        resid = y - y_hat

        med = float(np.median(resid[inliers]))
        mad = float(np.median(np.abs(resid[inliers] - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(resid[inliers]))
        sigma_s = float(sigma)
        # 条件分岐: `not (math.isfinite(sigma_s) and sigma_s > 0)` を満たす経路を評価する。
        if not (math.isfinite(sigma_s) and sigma_s > 0):
            break

        new_inliers = np.abs(resid - med) <= (3.5 * sigma_s)
        # 条件分岐: `int(np.sum(new_inliers)) < min_points` を満たす経路を評価する。
        if int(np.sum(new_inliers)) < min_points:
            break

        # 条件分岐: `bool(np.all(new_inliers == inliers))` を満たす経路を評価する。

        if bool(np.all(new_inliers == inliers)):
            break

        inliers = new_inliers

    # 条件分岐: `int(np.sum(inliers)) < min_points` を満たす経路を評価する。

    if int(np.sum(inliers)) < min_points:
        return {
            "ok": 0.0,
            "reason": "n_inliers_too_short",
            "n_total": float(len(t)),
            "n_base": float(int(np.sum(base))),
            "n_inliers": float(int(np.sum(inliers))),
            "sigma_t_s": sigma_s,
        }, np.zeros(len(t), dtype=bool)

    # Linear fit (final): y = m x + b  (m ~ -A, b ~ tc)

    m, b, w2 = _polyfit1(x[inliers], y[inliers])
    rank_warn = rank_warn or w2
    y_hat = m * x + b
    resid = y - y_hat

    # R^2 on inliers (t-space)
    y_in = y[inliers]
    y_hat_in = y_hat[inliers]
    ss_res = float(np.sum((y_in - y_hat_in) ** 2))
    ss_tot = float(np.sum((y_in - float(np.mean(y_in))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    A = -float(m)
    tc = float(b)

    # Convert A -> chirp mass (GR quadrupole inspiral, Newtonian order)
    # A = (5/256) * π^{-8/3} * (G M_c / c^3)^{-5/3}
    G = 6.67430e-11
    c = 299792458.0
    M_sun = 1.988409870698051e30

    mc_kg = float("nan")
    # 条件分岐: `A > 0 and math.isfinite(A)` を満たす経路を評価する。
    if A > 0 and math.isfinite(A):
        denom = A * (256.0 / 5.0) * (math.pi ** (8.0 / 3.0))
        # 条件分岐: `denom > 0` を満たす経路を評価する。
        if denom > 0:
            gm_over_c3 = denom ** (-3.0 / 5.0)
            mc_kg = (c**3 / G) * gm_over_c3

    mc_msun = mc_kg / M_sun if math.isfinite(mc_kg) else float("nan")

    # Secondary goodness-of-fit in f-space (for visualization quality).
    # Only meaningful when A>0 (so f(t) is real for t<tc).
    r2_f = float("nan")
    # 条件分岐: `A > 0 and math.isfinite(A)` を満たす経路を評価する。
    if A > 0 and math.isfinite(A):
        dt = tc - y_in
        f_hat = np.full_like(y_in, np.nan, dtype=np.float64)
        mask = dt > 0
        f_hat[mask] = np.power(A / dt[mask], 3.0 / 8.0)
        finite_hat = np.isfinite(f_hat)
        # 条件分岐: `np.any(finite_hat)` を満たす経路を評価する。
        if np.any(finite_hat):
            f_in = f[base][inliers]
            ss_res_f = float(np.sum((f_in[finite_hat] - f_hat[finite_hat]) ** 2))
            ss_tot_f = float(np.sum((f_in[finite_hat] - float(np.mean(f_in[finite_hat]))) ** 2))
            r2_f = 1.0 - (ss_res_f / ss_tot_f) if ss_tot_f > 0 else float("nan")

    out_mask = np.zeros(len(t), dtype=bool)
    base_idx = np.flatnonzero(base)
    out_mask[base_idx[inliers]] = True

    # Require physically meaningful A (A>0) and a finite chirp mass.
    if not math.isfinite(A):
        reason = "A_nonfinite"
    # 条件分岐: 前段条件が不成立で、`not (A > 0)` を追加評価する。
    elif not (A > 0):
        reason = "A_nonpositive"
    # 条件分岐: 前段条件が不成立で、`not math.isfinite(mc_msun)` を追加評価する。
    elif not math.isfinite(mc_msun):
        reason = "mc_nonfinite"
    else:
        reason = "unknown"

    # 条件分岐: `reason != "unknown"` を満たす経路を評価する。

    if reason != "unknown":
        return {
            "ok": 0.0,
            "reason": reason,
            "tc_s": tc,
            "A_s": A,
            "chirp_mass_msun": mc_msun,
            "r2": r2,
            "r2_f": r2_f,
            "n_total": float(len(t)),
            "n_base": float(int(np.sum(base))),
            "n_inliers": float(int(np.sum(inliers))),
            "sigma_t_s": sigma_s,
            "rank_warning": 1.0 if rank_warn else 0.0,
        }, out_mask

    return {
        "ok": 1.0,
        "tc_s": tc,
        "A_s": A,
        "chirp_mass_msun": mc_msun,
        "r2": r2,
        "r2_f": r2_f,
        "n_total": float(len(t)),
        "n_base": float(int(np.sum(base))),
        "n_inliers": float(int(np.sum(inliers))),
        "sigma_t_s": sigma_s,
        "rank_warning": 1.0 if rank_warn else 0.0,
    }, out_mask


def _repair_track_monotonic(t: np.ndarray, f: np.ndarray, *, max_points: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    """
    Best-effort cleanup for noisy instantaneous-frequency tracks:
    - Remove non-finite / non-positive f
    - Sort by time
    - Downsample by chunk-median to suppress jitter
    - Enforce non-decreasing f(t) (inspiral chirp prior)
    """
    t0 = np.asarray(t, dtype=np.float64)
    f0 = np.asarray(f, dtype=np.float64)
    mask = np.isfinite(t0) & np.isfinite(f0) & (f0 > 0)
    t0 = t0[mask]
    f0 = f0[mask]
    # 条件分岐: `t0.size == 0` を満たす経路を評価する。
    if t0.size == 0:
        return t0, f0

    order = np.argsort(t0)
    t1 = t0[order]
    f1 = f0[order]

    # 条件分岐: `int(max_points) > 0 and t1.size > int(max_points)` を満たす経路を評価する。
    if int(max_points) > 0 and t1.size > int(max_points):
        step = int(math.ceil(float(t1.size) / float(max_points)))
        t_ds: List[float] = []
        f_ds: List[float] = []
        for i in range(0, int(t1.size), int(step)):
            j = min(int(t1.size), i + int(step))
            t_ds.append(float(np.median(t1[i:j])))
            f_ds.append(float(np.median(f1[i:j])))

        t1 = np.asarray(t_ds, dtype=np.float64)
        f1 = np.asarray(f_ds, dtype=np.float64)

    # Enforce inspiral chirp direction: f should be non-decreasing with time.

    f1 = np.maximum.accumulate(f1)
    mask2 = np.isfinite(t1) & np.isfinite(f1) & (f1 > 0)
    return t1[mask2], f1[mask2]


def _fit_summary(fit: Dict[str, Any]) -> Dict[str, Any]:
    # 条件分岐: `not isinstance(fit, dict)` を満たす経路を評価する。
    if not isinstance(fit, dict):
        return {}

    keys = [
        "ok",
        "reason",
        "tc_s",
        "A_s",
        "chirp_mass_msun",
        "r2",
        "r2_f",
        "rank_warning",
        "n_total",
        "n_base",
        "n_inliers",
        "sigma_t_s",
        "repair",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        # 条件分岐: `k in fit` を満たす経路を評価する。
        if k in fit:
            out[k] = fit.get(k)

    return out


def _predict_f_from_fit(t: np.ndarray, *, tc: float, A: float) -> np.ndarray:
    # From t = tc - A f^{-8/3} => f = (A/(tc-t))^{3/8}
    if not (math.isfinite(A) and A > 0):
        return np.full_like(t, np.nan, dtype=np.float64)

    dt = tc - t
    out = np.full_like(t, np.nan, dtype=np.float64)
    mask = dt > 0
    out[mask] = np.power(A / dt[mask], 3.0 / 8.0)
    return out


def _render(
    results: List[Dict[str, Any]],
    *,
    out_png: Path,
    title: str,
    note_lines: List[str],
    event_name: str,
) -> None:
    _set_japanese_font()

    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12.4, 3.9 * nrows), dpi=140)
    axes = np.asarray(axes)
    # 条件分岐: `axes.ndim == 1` を満たす経路を評価する。
    if axes.ndim == 1:
        axes = axes.reshape((1, 2))

    for (ax_f, ax_lin), r in zip(axes, results):
        det = str(r.get("detector") or "")
        t_sel = np.asarray(r.get("t_sel"), dtype=np.float64)
        f_sel = np.asarray(r.get("f_sel"), dtype=np.float64)

        inlier_mask = np.asarray(r.get("inlier_mask"), dtype=bool)
        # 条件分岐: `inlier_mask.size != t_sel.size` を満たす経路を評価する。
        if inlier_mask.size != t_sel.size:
            inlier_mask = np.ones(t_sel.size, dtype=bool)

        outlier_mask = ~inlier_mask

        fit = r.get("fit") or {}
        tc = float(fit.get("tc_s", 0.0))
        A = float(fit.get("A_s", float("nan")))

        # Left: frequency track f(t)
        if np.any(outlier_mask):
            ax_f.scatter(
                t_sel[outlier_mask],
                f_sel[outlier_mask],
                s=12,
                alpha=0.25,
                color="#888",
                label=f"{det}: 除外点",
            )

        ax_f.scatter(t_sel[inlier_mask], f_sel[inlier_mask], s=14, alpha=0.8, label=f"{det}: 抽出 f(t)")

        t_line = np.linspace(float(np.min(t_sel)), float(np.max(t_sel)), 600)
        f_line = _predict_f_from_fit(t_line, tc=tc, A=A)
        ax_f.plot(t_line, f_line, color="#111", lw=1.8, label="四重極チャープ則（fit）")

        ax_f.axvline(0.0, color="#666", lw=1.1, ls="--", alpha=0.8)
        ax_f.set_title(f"{det}: 周波数トラック f(t)", fontsize=11)
        ax_f.set_ylabel("周波数 f [Hz]")
        ax_f.set_xlabel(f"時刻 t - t_event [s]（t_event={event_name}のGPS時刻）")
        ax_f.grid(True, alpha=0.25)
        ax_f.legend(loc="lower right")

        cm = float(fit.get("chirp_mass_msun", float("nan")))
        r2 = float(fit.get("r2", float("nan")))
        r2_f = float(fit.get("r2_f", float("nan")))
        sigma_t_s = float(fit.get("sigma_t_s", float("nan")))
        n_in = int(fit.get("n_inliers", float("nan"))) if "n_inliers" in fit else int(np.sum(inlier_mask))
        n_base = int(fit.get("n_base", float("nan"))) if "n_base" in fit else int(t_sel.size)
        txt = (
            f"M_c≈{cm:.2g} M☉,  R²(t)={r2:.4f},  R²(f)={r2_f:.4f}\n"
            f"採用/候補={n_in}/{n_base},  σ_t≈{sigma_t_s:.3g} s"
        )
        ax_f.text(0.02, 0.98, txt, transform=ax_f.transAxes, va="top", ha="left", fontsize=9.6, color="#111")

        # Right: linearized relation t = tc - A f^{-8/3}
        x_all = np.power(f_sel, -8.0 / 3.0)

        x_in = x_all[inlier_mask]
        y_in = t_sel[inlier_mask]
        x_out = x_all[outlier_mask]
        y_out = t_sel[outlier_mask]

        # 条件分岐: `np.any(outlier_mask)` を満たす経路を評価する。
        if np.any(outlier_mask):
            ax_lin.scatter(x_out, y_out, s=12, alpha=0.25, color="#888", label="除外点")

        ax_lin.scatter(x_in, y_in, s=14, alpha=0.8, label="採用点")

        # 条件分岐: `math.isfinite(A)` を満たす経路を評価する。
        if math.isfinite(A):
            x0 = float(np.min(x_in))
            x1 = float(np.max(x_in))
            x_line = np.linspace(x0, x1, 200)
            y_line = tc - A * x_line
            ax_lin.plot(x_line, y_line, color="#111", lw=1.8, label="t = t_c − A f^{-8/3}（fit）")

        ax_lin.set_title(f"{det}: 直線化（t vs f^(-8/3)）", fontsize=11)
        ax_lin.set_xlabel("f^(-8/3) [Hz^(-8/3)]")
        ax_lin.set_ylabel("時刻 t - t_event [s]")
        ax_lin.grid(True, alpha=0.25)
        ax_lin.legend(loc="upper right")

    fig.suptitle(title, y=0.99)

    # 条件分岐: `note_lines` を満たす経路を評価する。
    if note_lines:
        fig.text(0.01, 0.01, "\n".join(note_lines), ha="left", va="bottom", fontsize=9.2, color="#333")

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _render_public(
    results: List[Dict[str, Any]],
    *,
    out_png: Path,
    title: str,
    note_lines: List[str],
    event_name: str,
) -> None:
    _set_japanese_font()

    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=150)

    colors = {
        "H1": "#1f77b4",
        "L1": "#d62728",
    }

    for r in results:
        det = str(r.get("detector") or "")
        color = colors.get(det, "#2ca02c")

        t_sel = np.asarray(r.get("t_sel"), dtype=np.float64)
        f_sel = np.asarray(r.get("f_sel"), dtype=np.float64)
        inlier_mask = np.asarray(r.get("inlier_mask"), dtype=bool)
        # 条件分岐: `inlier_mask.size != t_sel.size` を満たす経路を評価する。
        if inlier_mask.size != t_sel.size:
            inlier_mask = np.ones(t_sel.size, dtype=bool)

        fit = r.get("fit") or {}
        tc = float(fit.get("tc_s", 0.0))
        A = float(fit.get("A_s", float("nan")))

        # 条件分岐: `t_sel.size` を満たす経路を評価する。
        if t_sel.size:
            ax.scatter(
                t_sel[inlier_mask],
                f_sel[inlier_mask],
                s=14,
                alpha=0.75,
                color=color,
                label=f"{det}: 観測から抽出した周波数（採用点）",
            )

        # 条件分岐: `t_sel.size and math.isfinite(A) and A > 0` を満たす経路を評価する。

        if t_sel.size and math.isfinite(A) and A > 0:
            t_in = t_sel[inlier_mask]
            # 条件分岐: `t_in.size` を満たす経路を評価する。
            if t_in.size:
                t0 = float(np.min(t_in))
                t1 = float(np.max(t_in))
                t_line = np.linspace(t0, t1, 300)
                f_line = _predict_f_from_fit(t_line, tc=tc, A=A)
                ax.plot(t_line, f_line, color=color, lw=2.0, alpha=0.9, label=f"{det}: 単純モデル曲線（fit）")

    ax.axvline(0.0, color="#666", lw=1.1, ls="--", alpha=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("周波数 f [Hz]")
    ax.set_xlabel(f"時刻 t - t_event [s]（t_event={event_name}のGPS時刻）")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9.5)

    # 条件分岐: `note_lines` を満たす経路を評価する。
    if note_lines:
        fig.text(0.01, 0.01, "\n".join(note_lines), ha="left", va="bottom", fontsize=9.0, color="#333")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _fit_waveform_template_from_chirp(
    *,
    t: np.ndarray,
    xf: np.ndarray,
    tc: float,
    A: float,
    t_window: Tuple[float, float],
) -> Dict[str, Any]:
    min_match_samples = 64
    min_match_duration_s = 0.01

    # Newtonian quadrupole chirp:
    #   t = tc - A f^{-8/3}
    # => f(t) = (A/(tc-t))^{3/8}
    # phase (up to a constant):
    #   phi(t) = phi0 - (16π/5) A^{3/8} (tc-t)^{5/8}
    # amplitude envelope (up to a constant):
    #   amp(t) ∝ f^{2/3} ∝ (tc-t)^{-1/4}
    t0_req, t1_req = float(t_window[0]), float(t_window[1])
    window_req = [t0_req, t1_req]

    # 条件分岐: `not (math.isfinite(tc) and math.isfinite(A) and A > 0)` を満たす経路を評価する。
    if not (math.isfinite(tc) and math.isfinite(A) and A > 0):
        return {
            "ok": 0.0,
            "reason": "invalid_fit",
            "fit": {"tc_s": tc, "A_s": A, "window_s": window_req, "window_s_requested": window_req},
        }

    t0, t1 = float(t0_req), float(t1_req)
    auto_shifted = False

    def _select_window(_t0: float, _t1: float) -> Tuple[np.ndarray, np.ndarray]:
        m = (t >= _t0) & (t <= _t1)
        # 条件分岐: `not np.any(m)` を満たす経路を評価する。
        if not np.any(m):
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

        return np.asarray(t[m], dtype=np.float64), np.asarray(xf[m], dtype=np.float64)

    t_win, x_win = _select_window(t0, t1)
    # 条件分岐: `t_win.size == 0` を満たす経路を評価する。
    if t_win.size == 0:
        return {
            "ok": 0.0,
            "reason": "empty_window",
            "fit": {"tc_s": tc, "A_s": A, "window_s": [t0, t1], "window_s_requested": window_req},
        }

    dt = float(tc) - t_win
    mask_pos = dt > 0
    # 条件分岐: `not np.any(mask_pos)` を満たす経路を評価する。
    if not np.any(mask_pos):
        # If the requested window is entirely after t_c, shift it earlier (same duration) so it ends at t_c-ε.
        dur = float(t1_req) - float(t0_req)
        eps = 1e-3
        t1 = min(float(t1_req), float(tc) - float(eps))
        t0 = float(t1) - float(dur)
        t_win, x_win = _select_window(t0, t1)
        # 条件分岐: `t_win.size == 0` を満たす経路を評価する。
        if t_win.size == 0:
            return {
                "ok": 0.0,
                "reason": "tc_before_window",
                "fit": {"tc_s": tc, "A_s": A, "window_s": [t0, t1], "window_s_requested": window_req},
            }

        dt = float(tc) - t_win
        mask_pos = dt > 0
        # 条件分岐: `not np.any(mask_pos)` を満たす経路を評価する。
        if not np.any(mask_pos):
            return {
                "ok": 0.0,
                "reason": "tc_before_window",
                "fit": {"tc_s": tc, "A_s": A, "window_s": [t0, t1], "window_s_requested": window_req},
            }

        auto_shifted = True

    n_window = int(t_win.size)
    n_dropped_after_tc = int(np.count_nonzero(~mask_pos))

    t_win = t_win[mask_pos]
    x_win = x_win[mask_pos]
    dt = dt[mask_pos]

    n = int(t_win.size)
    dur_s = float(np.max(t_win) - np.min(t_win)) if n >= 2 else 0.0
    # 条件分岐: `n < min_match_samples or dur_s < min_match_duration_s` を満たす経路を評価する。
    if n < min_match_samples or dur_s < min_match_duration_s:
        return {
            "ok": 0.0,
            "reason": "match_window_too_short",
            "fit": {
                "tc_s": float(tc),
                "A_s": float(A),
                "n": int(n),
                "n_window": int(n_window),
                "n_dropped_after_tc": int(n_dropped_after_tc),
                "window_duration_s": float(dur_s),
                "min_match_samples": int(min_match_samples),
                "min_match_duration_s": float(min_match_duration_s),
                "window_s": [t0, t1],
                "window_s_requested": window_req,
                "window_auto_shifted": bool(auto_shifted),
            },
        }

    phi = -(16.0 * math.pi / 5.0) * (float(A) ** (3.0 / 8.0)) * np.power(dt, 5.0 / 8.0)
    amp = np.power(dt, -1.0 / 4.0)

    g0 = amp * np.cos(phi)
    g1 = amp * np.sin(phi)
    G = np.vstack([g0, g1]).T

    # Linear least squares for amplitude+phase (avoid nonlinear fit).
    coeff, _, _, _ = np.linalg.lstsq(G, x_win, rcond=None)
    tmpl = G @ coeff
    resid = x_win - tmpl

    # Cosine similarity (normalized dot); robust for (nearly) zero-mean bandpassed waveforms.
    denom = float(np.linalg.norm(x_win) * np.linalg.norm(tmpl))
    overlap = float(np.dot(x_win, tmpl) / denom) if denom > 0 else float("nan")

    rmse = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")
    rms_x = float(np.sqrt(np.mean(x_win**2))) if x_win.size else float("nan")
    rmse_rel = (rmse / rms_x) if (math.isfinite(rmse) and math.isfinite(rms_x) and rms_x > 0) else float("nan")

    c0 = float(coeff[0]) if coeff.size >= 1 else float("nan")
    c1 = float(coeff[1]) if coeff.size >= 2 else float("nan")
    amp0 = float(math.hypot(c0, c1)) if math.isfinite(c0) and math.isfinite(c1) else float("nan")
    phi0 = float(math.atan2(c1, c0)) if math.isfinite(c0) and math.isfinite(c1) else float("nan")

    return {
        "ok": 1.0,
        "auto_shifted": bool(auto_shifted),
        "t_win": t_win,
        "x_win": x_win,
        "template": tmpl,
        "residual": resid,
        "fit": {
            "tc_s": float(tc),
            "A_s": float(A),
            "coeff_c0": c0,
            "coeff_c1": c1,
            "amp0": amp0,
            "phi0_rad": phi0,
            "overlap": overlap,
            "rmse": rmse,
            "rmse_rel": rmse_rel,
            "n": int(n),
            "n_window": int(n_window),
            "n_dropped_after_tc": int(n_dropped_after_tc),
            "window_duration_s": float(dur_s),
            "min_match_samples": int(min_match_samples),
            "min_match_duration_s": float(min_match_duration_s),
            "window_s": [t0, t1],
            "window_s_requested": window_req,
            "window_auto_shifted": bool(auto_shifted),
        },
    }


def _render_waveform_compare(
    results: List[Dict[str, Any]],
    *,
    out_png: Path,
    title: str,
    note_lines: List[str],
) -> None:
    _set_japanese_font()

    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12.4, 3.6 * nrows), dpi=140)
    axes = np.asarray(axes)
    # 条件分岐: `axes.ndim == 1` を満たす経路を評価する。
    if axes.ndim == 1:
        axes = axes.reshape((1, 2))

    for (ax_w, ax_r), r in zip(axes, results):
        det = str(r.get("detector") or "")
        wf = r.get("waveform_fit") or {}
        # 条件分岐: `not isinstance(wf, dict) or not wf.get("ok")` を満たす経路を評価する。
        if not isinstance(wf, dict) or not wf.get("ok"):
            ax_w.set_title(f"{det}: 波形比較（失敗）")
            reason = str(wf.get("reason") or "") if isinstance(wf, dict) else ""
            msg = "波形match計算: 失敗" + (f"（{reason}）" if reason else "")
            ax_w.text(0.02, 0.95, msg, transform=ax_w.transAxes, ha="left", va="top")
            ax_w.axis("off")
            ax_r.axis("off")
            continue

        t_win = np.asarray(wf.get("t_win"), dtype=np.float64)
        x_win = np.asarray(wf.get("x_win"), dtype=np.float64)
        tmpl = np.asarray(wf.get("template"), dtype=np.float64)
        resid = np.asarray(wf.get("residual"), dtype=np.float64)
        fit = wf.get("fit") or {}

        ax_w.plot(t_win, x_win, color="#888", lw=1.2, label="観測（前処理後）")
        ax_w.plot(t_win, tmpl, color="#111", lw=1.7, label="単純モデル（四重極）best-fit")
        ax_w.axvline(0.0, color="#666", lw=1.1, ls="--", alpha=0.8)
        ax_w.set_title(f"{det}: 波形（窓内）", fontsize=11)
        ax_w.set_ylabel("strain（相対）")
        ax_w.set_xlabel("時刻 t - t_event [s]")
        ax_w.grid(True, alpha=0.25)
        ax_w.legend(loc="lower right", fontsize=9.2)

        overlap = float(fit.get("overlap", float("nan")))
        rmse_rel = float(fit.get("rmse_rel", float("nan")))
        cm = float((r.get("fit") or {}).get("chirp_mass_msun", float("nan")))
        ax_w.text(
            0.02,
            0.98,
            f"M_c≈{cm:.2g} M☉\nmatch={overlap:.3f}, RMSE/RMS={rmse_rel:.3f}",
            transform=ax_w.transAxes,
            ha="left",
            va="top",
            fontsize=9.4,
            color="#111",
        )

        ax_r.plot(t_win, resid, color="#1f77b4", lw=1.2)
        ax_r.axhline(0.0, color="#666", lw=1.1, ls="--", alpha=0.8)
        ax_r.axvline(0.0, color="#666", lw=1.1, ls="--", alpha=0.8)
        ax_r.set_title(f"{det}: 残差（観測−モデル）", fontsize=11)
        ax_r.set_ylabel("residual")
        ax_r.set_xlabel("時刻 t - t_event [s]")
        ax_r.grid(True, alpha=0.25)

    fig.suptitle(title, y=0.99)

    # 条件分岐: `note_lines` を満たす経路を評価する。
    if note_lines:
        fig.text(0.01, 0.01, "\n".join(note_lines), ha="left", va="bottom", fontsize=9.2, color="#333")

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _render_waveform_compare_public(
    results: List[Dict[str, Any]],
    *,
    out_png: Path,
    title: str,
    note_lines: List[str],
) -> None:
    _set_japanese_font()

    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10.8, 3.6 * nrows), dpi=150)
    axes = np.asarray(axes)
    # 条件分岐: `axes.ndim == 0` を満たす経路を評価する。
    if axes.ndim == 0:
        axes = axes.reshape((1,))

    colors = {"H1": "#1f77b4", "L1": "#d62728"}

    for ax, r in zip(axes, results):
        det = str(r.get("detector") or "")
        wf = r.get("waveform_fit") or {}
        # 条件分岐: `not isinstance(wf, dict) or not wf.get("ok")` を満たす経路を評価する。
        if not isinstance(wf, dict) or not wf.get("ok"):
            ax.set_title(f"{det}: 波形比較（失敗）")
            reason = str(wf.get("reason") or "") if isinstance(wf, dict) else ""
            # 条件分岐: `reason` を満たす経路を評価する。
            if reason:
                ax.text(0.02, 0.95, f"失敗: {reason}", transform=ax.transAxes, ha="left", va="top", fontsize=10)

            ax.axis("off")
            continue

        t_win = np.asarray(wf.get("t_win"), dtype=np.float64)
        x_win = np.asarray(wf.get("x_win"), dtype=np.float64)
        tmpl = np.asarray(wf.get("template"), dtype=np.float64)
        fit = wf.get("fit") or {}

        color = colors.get(det, "#2ca02c")
        ax.plot(t_win, x_win, color="#888", lw=1.1, label="観測（前処理後）")
        ax.plot(t_win, tmpl, color=color, lw=2.0, label="単純モデル（四重極）")
        ax.axvline(0.0, color="#666", lw=1.1, ls="--", alpha=0.8)
        ax.set_title(f"{det}: 波形（窓内）", fontsize=12)
        ax.set_ylabel("strain（相対）")
        ax.set_xlabel("時刻 t - t_event [s]")
        ax.grid(True, alpha=0.25)

        overlap = float(fit.get("overlap", float("nan")))
        ax.text(0.02, 0.96, f"match={overlap:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#111")

        ax.legend(loc="lower right", fontsize=9.5)

    fig.suptitle(title, y=0.99)
    # 条件分岐: `note_lines` を満たす経路を評価する。
    if note_lines:
        fig.text(0.01, 0.01, "\n".join(note_lines), ha="left", va="bottom", fontsize=9.0, color="#333")

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="GW chirp (phase) check using GWOSC open data.")
    ap.add_argument("--event", type=str, default="GW150914", help="GWOSC event name (default: GW150914).")
    ap.add_argument(
        "--catalog",
        type=str,
        default="GWTC-1-confident",
        help="GWOSC event catalog shortName (default: GWTC-1-confident).",
    )
    ap.add_argument("--version", type=str, default="auto", help="GWOSC event API version (default: auto = try v3→v2→v1).")
    ap.add_argument("--detectors", type=str, default="H1,L1", help="Comma-separated detectors (default: H1,L1).")
    ap.add_argument("--prefer-duration", type=int, default=32, help="Prefer this duration (sec) snippet (default: 32).")
    ap.add_argument("--prefer-fs", type=int, default=4096, help="Prefer this sampling rate (Hz) (default: 4096).")
    ap.add_argument("--offline", action="store_true", help="Do not use network; require cached data in data/gw/<event>/")
    ap.add_argument("--force", action="store_true", help="Re-download cached inputs.")
    ap.add_argument("--out-png", type=str, default=None)
    ap.add_argument("--out-png-public", type=str, default=None)
    ap.add_argument("--out-wave-png", type=str, default=None)
    ap.add_argument("--out-wave-png-public", type=str, default=None)
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument(
        "--method",
        type=str,
        default="stft",
        choices=["stft", "hilbert"],
        help="Frequency track extraction method (default: stft ridge).",
    )
    ap.add_argument("--f-lo", type=float, default=30.0)
    ap.add_argument("--f-hi", type=float, default=350.0)
    ap.add_argument(
        "--whiten",
        action="store_true",
        help="Apply rough PSD whitening (Welch+FFT) before tracking (useful for long/weak chirps like BNS).",
    )
    ap.add_argument(
        "--whiten-nperseg",
        type=int,
        default=4096,
        help="Welch nperseg for whitening PSD estimate (default: 4096).",
    )
    ap.add_argument("--track-window", type=str, default="-0.2,-0.01", help="t range (sec) used for chirp fit, relative to event.")
    ap.add_argument("--track-frange", type=str, default="30,300", help="frequency range (Hz) used for chirp fit.")
    ap.add_argument("--wave-window", type=str, default="-0.08,-0.01", help="t range (sec) used for waveform compare, relative to event.")
    ap.add_argument(
        "--wave-frange",
        type=str,
        default=None,
        help="frequency range (Hz) used to derive waveform compare window (overrides --wave-window). Example: 70,300",
    )
    ap.add_argument("--amp-percentile", type=float, default=95.0, help="Keep points above this amplitude percentile in window.")
    ap.add_argument("--stft-nperseg", type=int, default=1024, help="STFT nperseg (samples).")
    ap.add_argument("--stft-noverlap", type=int, default=896, help="STFT noverlap (samples).")
    ap.add_argument(
        "--stft-guided",
        action="store_true",
        help="Use guided STFT tracking with a chirp-mass grid search (recommended for GW170817/BNS).",
    )
    ap.add_argument("--guided-mc-min", type=float, default=0.8, help="Guided STFT: chirp mass min [M_sun] (default: 0.8).")
    ap.add_argument("--guided-mc-max", type=float, default=2.5, help="Guided STFT: chirp mass max [M_sun] (default: 2.5).")
    ap.add_argument("--guided-mc-steps", type=int, default=86, help="Guided STFT: chirp mass grid steps (default: 86).")
    ap.add_argument("--guided-delta-hz", type=float, default=25.0, help="Guided STFT: +/- frequency window around f_pred(t) (Hz).")
    args = ap.parse_args()

    event_name = (str(args.event) or "").strip() or "GW150914"
    event_slug = event_name.lower()
    data_dir = root / "data" / "gw" / event_slug

    default_png = root / "output" / "private" / "gw" / f"{event_slug}_chirp_phase.png"
    default_png_public = root / "output" / "private" / "gw" / f"{event_slug}_chirp_phase_public.png"
    default_wave_png = root / "output" / "private" / "gw" / f"{event_slug}_waveform_compare.png"
    default_wave_png_public = root / "output" / "private" / "gw" / f"{event_slug}_waveform_compare_public.png"
    default_json = root / "output" / "private" / "gw" / f"{event_slug}_chirp_phase_metrics.json"

    out_png = Path(args.out_png) if args.out_png else default_png
    out_png_public = Path(args.out_png_public) if args.out_png_public else default_png_public
    out_wave_png = Path(args.out_wave_png) if args.out_wave_png else default_wave_png
    out_wave_png_public = Path(args.out_wave_png_public) if args.out_wave_png_public else default_wave_png_public
    out_json = Path(args.out_json) if args.out_json else default_json

    detectors = [d.strip() for d in str(args.detectors).split(",") if d.strip()]
    # 条件分岐: `not detectors` を満たす経路を評価する。
    if not detectors:
        detectors = ["H1", "L1"]

    try:
        fetch = _fetch_inputs(
            data_dir,
            event=event_name,
            catalog=str(args.catalog),
            version=str(args.version),
            detectors=detectors,
            prefer_duration_s=int(args.prefer_duration),
            prefer_sampling_rate_hz=int(args.prefer_fs),
            offline=bool(args.offline),
            force=bool(args.force),
        )
    except Exception as e:
        print(f"[err] fetch inputs failed: {e}")
        return 2

    # Event time (GPS)

    event_info = fetch.get("event_info") or {}
    gps_event = float(event_info.get("GPS") or float("nan"))
    # 条件分岐: `not math.isfinite(gps_event)` を満たす経路を評価する。
    if not math.isfinite(gps_event):
        print("[err] event GPS not found in event JSON; abort")
        return 2

    t0 = tuple(float(x.strip()) for x in str(args.track_window).split(","))
    fr = tuple(float(x.strip()) for x in str(args.track_frange).split(","))
    w0 = tuple(float(x.strip()) for x in str(args.wave_window).split(","))
    wave_frange: Optional[Tuple[float, float]] = None
    # 条件分岐: `args.wave_frange is not None and str(args.wave_frange).strip()` を満たす経路を評価する。
    if args.wave_frange is not None and str(args.wave_frange).strip():
        try:
            frw = [float(x.strip()) for x in str(args.wave_frange).split(",")]
        except Exception as e:
            print(f"[err] invalid --wave-frange: {e}")
            return 2

        # 条件分岐: `len(frw) != 2` を満たす経路を評価する。

        if len(frw) != 2:
            print("[err] invalid --wave-frange: expected 2 comma-separated floats (e.g. 70,300)")
            return 2

        flo, fhi = float(frw[0]), float(frw[1])
        # 条件分岐: `not (math.isfinite(flo) and math.isfinite(fhi) and flo > 0 and fhi > 0)` を満たす経路を評価する。
        if not (math.isfinite(flo) and math.isfinite(fhi) and flo > 0 and fhi > 0):
            print("[err] invalid --wave-frange: f must be finite and >0")
            return 2

        # 条件分岐: `flo > fhi` を満たす経路を評価する。

        if flo > fhi:
            flo, fhi = fhi, flo

        wave_frange = (flo, fhi)

    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    def _extract_and_fit(
        method: str,
        *,
        t: np.ndarray,
        xf: np.ndarray,
        fs: float,
        t_window: Tuple[float, float],
        f_range: Tuple[float, float],
        amp_percentile: float,
        stft_guided: bool,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
        guided_meta: Dict[str, Any] = {}
        # 条件分岐: `method == "hilbert"` を満たす経路を評価する。
        if method == "hilbert":
            t_sel, f_sel = _extract_instantaneous_frequency(
                t,
                xf,
                fs,
                t_window=t_window,
                f_range=f_range,
                amp_percentile=float(amp_percentile),
            )
        else:
            # 条件分岐: `bool(stft_guided)` を満たす経路を評価する。
            if bool(stft_guided):
                t_sel, f_sel, guided_meta = _extract_frequency_track_stft_guided(
                    t,
                    xf,
                    fs,
                    t_window=t_window,
                    f_range=f_range,
                    mc_min_msun=float(args.guided_mc_min),
                    mc_max_msun=float(args.guided_mc_max),
                    mc_steps=int(args.guided_mc_steps),
                    delta_hz=float(args.guided_delta_hz),
                    amp_percentile=float(amp_percentile),
                    nperseg=int(args.stft_nperseg),
                    noverlap=int(args.stft_noverlap),
                )
            else:
                t_sel, f_sel = _extract_frequency_track_stft(
                    t,
                    xf,
                    fs,
                    t_window=t_window,
                    f_range=f_range,
                    amp_percentile=float(amp_percentile),
                    nperseg=int(args.stft_nperseg),
                    noverlap=int(args.stft_noverlap),
                )

        fit, inlier_mask = _fit_chirp_mass_from_track(t_sel, f_sel)

        # 条件分岐: `guided_meta and isinstance(fit, dict)` を満たす経路を評価する。
        if guided_meta and isinstance(fit, dict):
            fit = dict(fit)
            fit["guided"] = guided_meta

        # If the raw track yields an unphysical fit (A<=0 etc.), try a monotonic repair (Hilbert tracks are noisy).

        reason = str((fit or {}).get("reason") or "").strip()
        # 条件分岐: `(not (fit or {}).get("ok")) and reason in {"A_nonpositive", "mc_nonfinite"} a...` を満たす経路を評価する。
        if (not (fit or {}).get("ok")) and reason in {"A_nonpositive", "mc_nonfinite"} and int(len(t_sel)) >= 10:
            t_rep, f_rep = _repair_track_monotonic(t_sel, f_sel)
            # 条件分岐: `int(len(t_rep)) >= 10` を満たす経路を評価する。
            if int(len(t_rep)) >= 10:
                fit2, inlier2 = _fit_chirp_mass_from_track(t_rep, f_rep)
                # 条件分岐: `(fit2 or {}).get("ok")` を満たす経路を評価する。
                if (fit2 or {}).get("ok"):
                    fit2 = dict(fit2)
                    fit2["repair"] = {"kind": "monotonic_median", "n_before": int(len(t_sel)), "n_after": int(len(t_rep))}
                    # 条件分岐: `guided_meta` を満たす経路を評価する。
                    if guided_meta:
                        fit2["guided"] = guided_meta

                    return t_rep, f_rep, fit2, inlier2

        return t_sel, f_sel, fit, inlier_mask

    method_requested = str(args.method).lower().strip()
    # 条件分岐: `method_requested not in {"stft", "hilbert"}` を満たす経路を評価する。
    if method_requested not in {"stft", "hilbert"}:
        method_requested = "stft"

    strain_paths = (fetch.get("paths") or {}).get("strain") or {}
    # 条件分岐: `not isinstance(strain_paths, dict)` を満たす経路を評価する。
    if not isinstance(strain_paths, dict):
        strain_paths = {}

    for det_label in detectors:
        p0 = strain_paths.get(det_label)
        # 条件分岐: `not p0` を満たす経路を評価する。
        if not p0:
            skipped.append({"detector": det_label, "reason": "strain_not_available"})
            continue

        p = Path(p0)
        gps_start, fs, strain = _parse_gwosc_txt_gz(p)

        t = (gps_start + np.arange(len(strain), dtype=np.float64) / fs) - gps_event
        x = strain.astype(np.float64) - float(np.mean(strain))

        preprocess_cache: Dict[str, np.ndarray] = {}

        def _get_xf(kind: str) -> Tuple[str, np.ndarray]:
            k = (kind or "").strip().lower() or "bandpass"
            # 条件分岐: `k in preprocess_cache` を満たす経路を評価する。
            if k in preprocess_cache:
                label = "whiten+bandpass" if k == "whiten" else "bandpass"
                return label, preprocess_cache[k]

            # 条件分岐: `k == "whiten"` を満たす経路を評価する。

            if k == "whiten":
                xf0 = _whiten_fft(
                    x,
                    fs,
                    f_lo=float(args.f_lo),
                    f_hi=float(args.f_hi),
                    welch_nperseg=int(args.whiten_nperseg),
                )
                preprocess_cache[k] = xf0
                return "whiten+bandpass", xf0

            xf0 = _bandpass(x, fs, f_lo=float(args.f_lo), f_hi=float(args.f_hi), order=4)
            preprocess_cache[k] = xf0
            return "bandpass", xf0

        def _expand_window(window: Tuple[float, float]) -> Tuple[float, float]:
            start, end = float(window[0]), float(window[1])
            # 条件分岐: `not (math.isfinite(start) and math.isfinite(end) and start < end)` を満たす経路を評価する。
            if not (math.isfinite(start) and math.isfinite(end) and start < end):
                return window

            # 条件分岐: `start <= -2.0` を満たす経路を評価する。

            if start <= -2.0:
                return window

            new_start = min(start * 3.0, -0.6)
            new_start = max(new_start, -2.0)
            return (float(new_start), float(end))

        base_pre = "whiten" if bool(args.whiten) else "bandpass"
        base_amp = float(args.amp_percentile)
        base_stft_guided = bool(args.stft_guided)

        attempts: List[Dict[str, Any]] = []
        picked: Optional[Dict[str, Any]] = None

        def _try_one(
            *,
            method: str,
            preprocess_kind: str,
            t_window: Tuple[float, float],
            f_range: Tuple[float, float],
            amp_percentile: float,
            stft_guided: bool,
            label: str,
        ) -> Optional[Dict[str, Any]]:
            nonlocal picked
            preprocess_label, xf = _get_xf(preprocess_kind)
            t_sel, f_sel, fit, inlier_mask = _extract_and_fit(
                method,
                t=t,
                xf=xf,
                fs=fs,
                t_window=t_window,
                f_range=f_range,
                amp_percentile=amp_percentile,
                stft_guided=stft_guided,
            )
            method_used_label = method
            # 条件分岐: `method_used_label == "stft" and bool(stft_guided)` を満たす経路を評価する。
            if method_used_label == "stft" and bool(stft_guided):
                method_used_label = "stft_guided"

            attempts.append(
                {
                    "label": label,
                    "method": method,
                    "method_used": method_used_label,
                    "preprocess": preprocess_label,
                    "track_window_s": [float(t_window[0]), float(t_window[1])],
                    "track_frange_hz": [float(f_range[0]), float(f_range[1])],
                    "amp_percentile": float(amp_percentile),
                    "stft_guided": bool(stft_guided),
                    "n_track": int(len(t_sel)),
                    "fit": _fit_summary(fit),
                }
            )
            # 条件分岐: `(fit or {}).get("ok")` を満たす経路を評価する。
            if (fit or {}).get("ok"):
                picked = {
                    "detector": det_label,
                    "method_used": method_used_label,
                    "preprocess": preprocess_label,
                    "fs_hz": fs,
                    "gps_start": gps_start,
                    "n": int(len(strain)),
                    "t": t,
                    "xf": xf,
                    "t_sel": t_sel,
                    "f_sel": f_sel,
                    "inlier_mask": inlier_mask,
                    "fit": fit,
                    "params_used": {
                        "preprocess": preprocess_kind,
                        "track_window_s": [float(t_window[0]), float(t_window[1])],
                        "track_frange_hz": [float(f_range[0]), float(f_range[1])],
                        "amp_percentile": float(amp_percentile),
                        "stft_guided": bool(stft_guided),
                    },
                }
                return picked

            return None

        # Attempt 1: requested method (profile default)

        _try_one(
            method=method_requested,
            preprocess_kind=base_pre,
            t_window=t0,
            f_range=fr,
            amp_percentile=base_amp,
            stft_guided=base_stft_guided,
            label="primary",
        )

        # Attempt 2: alternate method (same preprocessing/window)
        method_alt = "hilbert" if method_requested == "stft" else "stft"
        # 条件分岐: `picked is None` を満たす経路を評価する。
        if picked is None:
            _try_one(
                method=method_alt,
                preprocess_kind=base_pre,
                t_window=t0,
                f_range=fr,
                amp_percentile=base_amp,
                stft_guided=base_stft_guided,
                label="alt_method",
            )

        # Attempt 3: relaxed fallback (detector weaknesses, esp. V1): whiten + longer window + lower percentile.

        if picked is None:
            relaxed_window = _expand_window(t0)
            relaxed_fr = (float(fr[0]), float(max(fr[1], 350.0)))
            relaxed_amp = float(min(base_amp, 75.0))
            _try_one(
                method="hilbert",
                preprocess_kind="whiten",
                t_window=relaxed_window,
                f_range=relaxed_fr,
                amp_percentile=relaxed_amp,
                stft_guided=False,
                label="relaxed_whiten_hilbert",
            )

        # 条件分岐: `picked is None` を満たす経路を評価する。

        if picked is None:
            primary_fit = attempts[0].get("fit") if attempts else {}
            alt_fit = attempts[1].get("fit") if len(attempts) >= 2 else {}
            reasons: List[str] = []
            for a in attempts:
                r = str(((a.get("fit") or {}).get("reason") or "")).strip()
                # 条件分岐: `r and r not in reasons` を満たす経路を評価する。
                if r and r not in reasons:
                    reasons.append(r)

            subreason = " / ".join(reasons[:3]) if reasons else ""
            skipped.append(
                {
                    "detector": det_label,
                    "reason": "fit_failed",
                    **({"subreason": subreason} if subreason else {}),
                    "method_primary": method_requested,
                    "method_alt": method_alt,
                    "n_track_primary": int((attempts[0].get("n_track") if attempts else 0) or 0),
                    "n_track_alt": int((attempts[1].get("n_track") if len(attempts) >= 2 else 0) or 0),
                    "fit_primary": primary_fit,
                    "fit_alt": alt_fit,
                    "attempts": attempts,
                }
            )
            msg = f"[warn] fit failed: {det_label} (primary={method_requested}; alt={method_alt})"
            # 条件分岐: `subreason` を満たす経路を評価する。
            if subreason:
                msg += f" subreason={subreason}"

            print(msg)
            continue

        assert picked is not None
        picked["fit_attempts"] = attempts
        results.append(picked)

    # 条件分岐: `not results` を満たす経路を評価する。

    if not results:
        print("[err] no valid detector tracks; abort")
        return 2

    title = f"重力波（{event_name}）：位相（周波数上昇＝chirp）の一致"
    methods = ", ".join([f"{r['detector']}={r.get('method_used','?')}" for r in results])
    preprocesses = ", ".join(sorted({str(r.get("preprocess") or "") for r in results if r.get("preprocess")}))
    note_lines = [
        "データ: GWOSC公開 strain（32秒, 4 kHz, H1/L1）",
        f"前処理: {preprocesses or 'bandpass'} {args.f_lo:g}–{args.f_hi:g} Hz + 周波数トラック抽出（Hilbert / STFT）",
        f"fit: t = t_c − A f^(−8/3)（四重極チャープ則, Newton近似）を {t0[0]:g}〜{t0[1]:g} s に当てはめ",
    ]
    # 条件分岐: `methods` を満たす経路を評価する。
    if methods:
        note_lines.append(f"抽出法: {methods}")

    note_lines.append("注: ここでの M_c はデータからの簡易推定（厳密なパラメータ推定ではない）。")
    _render(results, out_png=out_png, title=title, note_lines=note_lines, event_name=event_name)

    public_title = f"{event_name}：周波数が上がる（chirp）様子と単純モデルの一致"
    public_notes = [
        "観測: GWOSC公開 strain（H1/L1）から周波数トラック f(t) を抽出",
        "線: 単純モデル（四重極チャープ則, Newton近似）への当てはめ",
    ]
    _render_public(results, out_png=out_png_public, title=public_title, note_lines=public_notes, event_name=event_name)

    # Waveform comparison (Newtonian quadrupole chirp template; amplitude+phase fitted in-window).
    def _t_from_f(f_hz: float, *, tc: float, A: float) -> float:
        return float(tc) - float(A) * float(f_hz) ** (-8.0 / 3.0)

    for r in results:
        fit = r.get("fit") or {}
        tc = float(fit.get("tc_s", float("nan")))
        A = float(fit.get("A_s", float("nan")))
        t_window = w0
        # 条件分岐: `wave_frange is not None and math.isfinite(tc) and math.isfinite(A) and A > 0` を満たす経路を評価する。
        if wave_frange is not None and math.isfinite(tc) and math.isfinite(A) and A > 0:
            flo, fhi = wave_frange
            try:
                tw0 = _t_from_f(flo, tc=tc, A=A)
                tw1 = _t_from_f(fhi, tc=tc, A=A)
                # 条件分岐: `math.isfinite(tw0) and math.isfinite(tw1)` を満たす経路を評価する。
                if math.isfinite(tw0) and math.isfinite(tw1):
                    t_window = (min(float(tw0), float(tw1)), max(float(tw0), float(tw1)))
            except Exception:
                pass

        wf = _fit_waveform_template_from_chirp(t=r["t"], xf=r["xf"], tc=tc, A=A, t_window=t_window)
        # 条件分岐: `wave_frange is not None and isinstance(wf, dict)` を満たす経路を評価する。
        if wave_frange is not None and isinstance(wf, dict):
            try:
                ffit = wf.get("fit") if isinstance(wf.get("fit"), dict) else None
                # 条件分岐: `ffit is not None` を満たす経路を評価する。
                if ffit is not None:
                    ffit["window_frange_hz"] = [float(wave_frange[0]), float(wave_frange[1])]
                    ffit["window_policy"] = "frange"
            except Exception:
                pass

        r["waveform_fit"] = wf

    wave_title = f"{event_name}：波形（前処理後）と単純モデル（四重極）テンプレートの比較"
    wave_notes = [
        "観測: GWOSC公開 strain（H1/L1）を前処理（bandpass/whiten+bandpass）した波形",
        "モデル: 四重極チャープ則から作る簡易テンプレート（振幅/位相は窓内で最小二乗）",
    ]
    # 条件分岐: `wave_frange is not None` を満たす経路を評価する。
    if wave_frange is not None:
        wave_notes.append(f"match窓: 周波数帯 {wave_frange[0]:g}..{wave_frange[1]:g} Hz を t=t_c−A f^(−8/3) で時刻窓に変換")

    _render_waveform_compare(results, out_png=out_wave_png, title=wave_title, note_lines=wave_notes)

    wave_public_title = f"{event_name}：観測波形と単純モデル（四重極）の重ね合わせ（窓内）"
    wave_public_notes = [
        "観測: 前処理後の波形（H1/L1）",
        "線: 単純モデル（四重極）を窓内で最も合うように調整",
    ]
    # 条件分岐: `wave_frange is not None` を満たす経路を評価する。
    if wave_frange is not None:
        wave_public_notes.append(f"match窓: {wave_frange[0]:g}..{wave_frange[1]:g} Hz に対応する時刻範囲")

    _render_waveform_compare_public(results, out_png=out_wave_png_public, title=wave_public_title, note_lines=wave_public_notes)

    event_json_path = Path((fetch.get("paths") or {}).get("event_json") or "")
    out: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "inputs": {
            "data_dir": str(data_dir).replace("\\", "/"),
            "event_json": str(event_json_path).replace("\\", "/"),
            "catalog": str(args.catalog),
            "version": str(((fetch.get("meta") or {}).get("selection") or {}).get("api_version") or ""),
            "detectors_requested": detectors,
        },
        "outputs": {
            "chirp_phase_png": str(out_png).replace("\\", "/"),
            "chirp_phase_public_png": str(out_png_public).replace("\\", "/"),
            "waveform_compare_png": str(out_wave_png).replace("\\", "/"),
            "waveform_compare_public_png": str(out_wave_png_public).replace("\\", "/"),
            "metrics_json": str(out_json).replace("\\", "/"),
        },
        "event": {"name": event_name, "gps_event": gps_event},
        "params": {
            "bandpass_hz": [float(args.f_lo), float(args.f_hi)],
            "whiten": bool(args.whiten),
            "whiten_nperseg": int(args.whiten_nperseg),
            "track_window_s": [t0[0], t0[1]],
            "track_frange_hz": [fr[0], fr[1]],
            "wave_window_s": [w0[0], w0[1]],
            "wave_window_policy": "frange" if wave_frange is not None else "time",
            **({"wave_frange_hz": [float(wave_frange[0]), float(wave_frange[1])]} if wave_frange is not None else {}),
            "amp_percentile": float(args.amp_percentile),
            "method_requested": method_requested,
            "stft_nperseg": int(args.stft_nperseg),
            "stft_noverlap": int(args.stft_noverlap),
            "stft_guided": bool(args.stft_guided),
            "guided_mc_range_msun": [float(args.guided_mc_min), float(args.guided_mc_max), int(args.guided_mc_steps)],
            "guided_delta_hz": float(args.guided_delta_hz),
        },
        "detectors": [
            {
                "detector": str(r["detector"]),
                "method_used": str(r.get("method_used") or ""),
                "preprocess": str(r.get("preprocess") or ""),
                "fit": r["fit"],
                "waveform_fit": (
                    {
                        **(
                            ((r.get("waveform_fit") or {}).get("fit") or {})
                            if isinstance((r.get("waveform_fit") or {}).get("fit"), dict)
                            else {}
                        ),
                        **(
                            {
                                "ok": float((r.get("waveform_fit") or {}).get("ok", 0.0)),
                                **(
                                    {"reason": str((r.get("waveform_fit") or {}).get("reason"))}
                                    if (r.get("waveform_fit") or {}).get("reason")
                                    else {}
                                ),
                            }
                            if isinstance(r.get("waveform_fit") or {}, dict)
                            else {}
                        ),
                    }
                    if isinstance(r.get("waveform_fit") or {}, dict)
                    else None
                ),
                "fs_hz": r["fs_hz"],
                "gps_start": r["gps_start"],
                "n": r["n"],
                "n_track": int(len(r["t_sel"])),
            }
            for r in results
        ],
        "skipped_detectors": skipped,
        "sources": fetch["meta"],
        "notes": note_lines,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        event_type = "gw150914_chirp_phase" if event_slug == "gw150914" else f"gw_{event_slug}_chirp_phase"
        worklog.append_event(
            {
                "event_type": event_type,
                "argv": list(sys.argv),
                "inputs": {"data_dir": data_dir},
                "outputs": {
                    "png": out_png,
                    "public_png": out_png_public,
                    "wave_png": out_wave_png,
                    "wave_public_png": out_wave_png_public,
                    "metrics_json": out_json,
                },
                "summary": {
                    "event": event_name,
                    "chirp_mass_msun": {r["detector"]: float(r["fit"]["chirp_mass_msun"]) for r in results},
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] pub : {out_png_public}")
    print(f"[ok] wave: {out_wave_png}")
    print(f"[ok] wpub: {out_wave_png_public}")
    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
