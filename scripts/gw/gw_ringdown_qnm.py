from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import h5py  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("h5py is required for GW ringdown inputs") from e

try:
    from scipy.signal import butter, sosfiltfilt  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("scipy is required for ringdown bandpass filtering") from e

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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _infer_zenodo_record_id_from_url(url: str) -> Optional[int]:
    try:
        parsed = urllib.parse.urlparse(str(url))
        parts = [p for p in parsed.path.split("/") if p]
        for i, p in enumerate(parts):
            # 条件分岐: `p == "records" and i + 1 < len(parts)` を満たす経路を評価する。
            if p == "records" and i + 1 < len(parts):
                return int(parts[i + 1])
    except Exception:
        return None

    return None


def _zenodo_file_url(*, record_id: int, filename: str) -> str:
    return f"https://zenodo.org/api/records/{int(record_id)}/files/{filename}/content"


def _ensure_extracted(tar_path: Path, *, extract_dir: Path, members: Sequence[str]) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        for name in members:
            out_path = extract_dir / name
            # 条件分岐: `out_path.exists()` を満たす経路を評価する。
            if out_path.exists():
                continue

            tf.extract(tf.getmember(name), path=extract_dir)


def _read_pandas_hdf_block0(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        g = f["samples"]
        cols = [c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c) for c in g["block0_items"][...]]
        vals = np.asarray(g["block0_values"][...], dtype=np.float64)

    # 条件分岐: `vals.ndim != 2 or vals.shape[1] != len(cols)` を満たす経路を評価する。

    if vals.ndim != 2 or vals.shape[1] != len(cols):
        raise ValueError("unexpected pandas HDF5 shape")

    return {name: vals[:, i] for i, name in enumerate(cols)}


def _select_preferred_posterior(event_info: Dict[str, Any]) -> Optional[str]:
    params = event_info.get("parameters") or {}
    # 条件分岐: `not isinstance(params, dict) or not params` を満たす経路を評価する。
    if not isinstance(params, dict) or not params:
        return None

    best: Optional[str] = None
    for _, rec in params.items():
        # 条件分岐: `not isinstance(rec, dict)` を満たす経路を評価する。
        if not isinstance(rec, dict):
            continue

        data_url = str(rec.get("data_url") or "").strip()
        # 条件分岐: `not data_url` を満たす経路を評価する。
        if not data_url:
            continue

        # 条件分岐: `bool(rec.get("is_preferred"))` を満たす経路を評価する。

        if bool(rec.get("is_preferred")):
            return data_url

        # 条件分岐: `best is None` を満たす経路を評価する。

        if best is None:
            best = data_url

    return best


def _summarize_1d(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "p16_p84": [float(v) for v in np.quantile(x, [0.16, 0.84])],
        "p05_p95": [float(v) for v in np.quantile(x, [0.05, 0.95])],
    }


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


def _gwosc_catalog_url(catalog: str) -> str:
    cat = (catalog or "").strip()
    # 条件分岐: `not cat` を満たす経路を評価する。
    if not cat:
        raise ValueError("catalog is required")

    return f"https://gwosc.org/eventapi/json/{cat}/"


def _gwosc_event_json_url(*, catalog: str, event: str, version: str) -> str:
    cat = (catalog or "").strip()
    # 条件分岐: `not cat` を満たす経路を評価する。
    if not cat:
        raise ValueError("catalog is required")

    ev = (event or "").strip()
    # 条件分岐: `not ev` を満たす経路を評価する。
    if not ev:
        raise ValueError("event is required")

    v = _normalize_gwosc_version(version)
    return f"https://gwosc.org/eventapi/json/{cat}/{ev}/{v}"


def _resolve_event_common_name(*, catalog: str, event: str) -> str:
    target = (event or "").strip()
    # 条件分岐: `not target` を満たす経路を評価する。
    if not target:
        raise ValueError("event name is required")

    # 条件分岐: `"_" in target` を満たす経路を評価する。

    if "_" in target:
        return target

    url = _gwosc_catalog_url(catalog)
    obj = json.loads(urllib.request.urlopen(url, timeout=60).read().decode("utf-8"))
    events = obj.get("events") or {}
    # 条件分岐: `not isinstance(events, dict) or not events` を満たす経路を評価する。
    if not isinstance(events, dict) or not events:
        raise ValueError(f"invalid catalog payload: {catalog}")

    cand: List[Tuple[float, str]] = []
    for k, v in events.items():
        # 条件分岐: `not isinstance(v, dict)` を満たす経路を評価する。
        if not isinstance(v, dict):
            continue

        common = str(v.get("commonName") or "").strip()
        # 条件分岐: `not common` を満たす経路を評価する。
        if not common:
            continue

        # 条件分岐: `common.startswith(target)` を満たす経路を評価する。

        if common.startswith(target):
            snr = v.get("network_matched_filter_snr")
            try:
                snr_f = float(snr) if snr is not None else float("nan")
            except Exception:
                snr_f = float("nan")

            cand.append((snr_f, common))
        # 条件分岐: 前段条件が不成立で、`str(k).startswith(target)` を追加評価する。
        elif str(k).startswith(target):
            snr = v.get("network_matched_filter_snr")
            try:
                snr_f = float(snr) if snr is not None else float("nan")
            except Exception:
                snr_f = float("nan")

            common2 = common or str(k).split("-", 1)[0]
            cand.append((snr_f, common2))

    # 条件分岐: `not cand` を満たす経路を評価する。

    if not cand:
        raise ValueError(f"event '{target}' not found in catalog '{catalog}'")

    # Prefer the highest network SNR when multiple matches exist.

    cand_sorted = sorted(cand, key=lambda t: (-(t[0] if math.isfinite(t[0]) else -1.0), t[1]))
    chosen = cand_sorted[0][1]
    # 条件分岐: `len({c for _, c in cand}) > 1` を満たす経路を評価する。
    if len({c for _, c in cand}) > 1:
        print(f"[warn] multiple matches for '{target}', picked: {chosen}")

    return chosen


def _pick_hdf5_strain_entry(
    strain_list: Sequence[Dict[str, Any]],
    *,
    detector: str,
    prefer_sampling_rate_hz: int,
) -> Optional[Dict[str, Any]]:
    det = (detector or "").strip()
    # 条件分岐: `not det` を満たす経路を評価する。
    if not det:
        return None

    cand = [e for e in (strain_list or []) if isinstance(e, dict) and str(e.get("detector") or "") == det]
    cand = [e for e in cand if str(e.get("format") or "").strip().lower() == "hdf5" and str(e.get("url") or "").strip()]
    # 条件分岐: `not cand` を満たす経路を評価する。
    if not cand:
        return None

    def _int(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    preferred = [e for e in cand if _int(e.get("sampling_rate")) == int(prefer_sampling_rate_hz)]
    # 条件分岐: `preferred` を満たす経路を評価する。
    if preferred:
        # Prefer the shortest duration within the preferred sampling-rate bucket.
        preferred = sorted(preferred, key=lambda e: _int(e.get("duration")) or 10**9)
        return preferred[0]

    # Fallback: smallest sampling rate.

    return sorted(cand, key=lambda e: _int(e.get("sampling_rate")) or 10**9)[0]


def _download_event_and_strain(
    *,
    data_dir: Path,
    event: str,
    catalog: str,
    version: str,
    detectors: Sequence[str],
    prefer_sampling_rate_hz: int,
    offline: bool,
    force: bool,
) -> Dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)

    resolved_event = _resolve_event_common_name(catalog=catalog, event=event)

    event_json_url = ""
    event_json_path = data_dir / f"{resolved_event}_event.json"
    # 条件分岐: `offline` を満たす経路を評価する。
    if offline:
        # 条件分岐: `not event_json_path.exists()` を満たす経路を評価する。
        if not event_json_path.exists():
            raise FileNotFoundError("offline and missing event JSON: " + str(event_json_path))
    else:
        last_404: Optional[Exception] = None
        for v_try in _candidate_gwosc_versions(version):
            url_try = _gwosc_event_json_url(catalog=catalog, event=resolved_event, version=v_try)
            try:
                _download(url_try, event_json_path, force=force)
                event_json_url = url_try
                break
            except urllib.error.HTTPError as e:
                # 条件分岐: `int(getattr(e, "code", 0) or 0) == 404 and len(_candidate_gwosc_versions(vers...` を満たす経路を評価する。
                if int(getattr(e, "code", 0) or 0) == 404 and len(_candidate_gwosc_versions(version)) > 1:
                    last_404 = e
                    continue

                raise

        # 条件分岐: `event_json_url == "" and last_404 is not None` を満たす経路を評価する。

        if event_json_url == "" and last_404 is not None:
            raise last_404

    ev_obj = _read_json(event_json_path)
    events = ev_obj.get("events") or {}
    # 条件分岐: `not isinstance(events, dict) or not events` を満たす経路を評価する。
    if not isinstance(events, dict) or not events:
        raise ValueError("invalid event JSON: missing 'events'")
    # The payload should contain exactly one event for this endpoint.

    event_key = next(iter(events.keys()))
    event_info = events[event_key]
    # 条件分岐: `not isinstance(event_info, dict)` を満たす経路を評価する。
    if not isinstance(event_info, dict):
        raise ValueError("invalid event JSON: event is not an object")

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
        entry = _pick_hdf5_strain_entry(
            strain_list,
            detector=det,
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

    sources = {
        "generated_utc": _iso_utc_now(),
        "source": f"GWOSC ({catalog} / {resolved_event})",
        "urls": {"event_json": event_json_url or _gwosc_event_json_url(catalog=catalog, event=resolved_event, version="v1")},
        "files": {
            "event_json": {"path": str(event_json_path).replace("\\", "/"), "sha256": _sha256(event_json_path)},
            **{k: {"path": str(p).replace("\\", "/"), "sha256": _sha256(p)} for k, p in strain_paths.items()},
        },
        "selection": {
            "event_input": event,
            "event_key": str(event_key),
            "event_common_name": str(event_info.get("commonName") or resolved_event),
            "catalog": catalog,
            "api_versions_tried": _candidate_gwosc_versions(version),
            "prefer_sampling_rate_hz": int(prefer_sampling_rate_hz),
            "selected": selected,
        },
    }
    sources_path = data_dir / f"{(data_dir.name or 'gw')}_sources.json"
    _write_json(sources_path, sources)

    return {
        "paths": {"event_json": event_json_path, "strain": strain_paths, "sources": sources_path},
        "meta": sources,
        "event_info": event_info,
        "resolved_event": resolved_event,
    }


@dataclass(frozen=True)
class FitVariant:
    bandpass_hz: Tuple[float, float]
    start_s: float
    duration_s: float
    fmin_hz: float
    fmax_hz: float
    tau_min_s: float
    tau_max_s: float
    f_step_hz: float
    tau_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bandpass_hz": [float(self.bandpass_hz[0]), float(self.bandpass_hz[1])],
            "start_s": float(self.start_s),
            "duration_s": float(self.duration_s),
            "f_grid_hz": {
                "min": float(self.fmin_hz),
                "max": float(self.fmax_hz),
                "step": float(self.f_step_hz),
            },
            "tau_grid_s": {
                "min": float(self.tau_min_s),
                "max": float(self.tau_max_s),
                "steps": int(self.tau_steps),
            },
        }


def _bandpass(x: np.ndarray, fs_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = 0.5 * fs_hz
    low = max(1e-6, float(low_hz) / nyq)
    high = min(0.999999, float(high_hz) / nyq)
    # 条件分岐: `not (0.0 < low < high < 1.0)` を満たす経路を評価する。
    if not (0.0 < low < high < 1.0):
        raise ValueError(f"invalid bandpass: {low_hz}-{high_hz} Hz for fs={fs_hz} Hz")

    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def _slice_strain_hdf5(
    path: Path,
    *,
    gps_center: float,
    pre_s: float,
    post_s: float,
) -> Tuple[np.ndarray, float]:
    with h5py.File(path, "r") as f:
        ds = f["strain"]["Strain"]
        xstart = float(ds.attrs.get("Xstart"))
        dt = float(ds.attrs.get("Xspacing"))
        fs = 1.0 / dt
        n = int(ds.shape[0])

        t0 = gps_center - float(pre_s)
        t1 = gps_center + float(post_s)
        i0 = int(round((t0 - xstart) / dt))
        i1 = int(round((t1 - xstart) / dt))
        i0 = max(0, min(n, i0))
        i1 = max(0, min(n, i1))
        # 条件分岐: `i1 <= i0` を満たす経路を評価する。
        if i1 <= i0:
            raise ValueError("empty slice after bounds adjustment")

        x = np.asarray(ds[i0:i1], dtype=np.float64)
        return x, fs


def _fit_damped_sinusoid(
    t: np.ndarray,
    y: np.ndarray,
    *,
    f_grid_hz: np.ndarray,
    tau_grid_s: np.ndarray,
) -> Dict[str, Any]:
    # 条件分岐: `len(t) != len(y)` を満たす経路を評価する。
    if len(t) != len(y):
        raise ValueError("t and y length mismatch")

    # 条件分岐: `len(t) < 20` を満たす経路を評価する。

    if len(t) < 20:
        raise ValueError("too few samples for ringdown fit")

    y0 = y - float(np.mean(y))
    sst = float(np.sum((y0 - float(np.mean(y0))) ** 2))
    best: Optional[Dict[str, Any]] = None

    for tau in tau_grid_s:
        # 条件分岐: `not (tau > 0)` を満たす経路を評価する。
        if not (tau > 0):
            continue

        env = np.exp(-t / float(tau))
        for f in f_grid_hz:
            w = 2.0 * math.pi * float(f)
            c = env * np.cos(w * t)
            s = env * np.sin(w * t)
            a = np.column_stack([c, s])
            coeff, _, _, _ = np.linalg.lstsq(a, y0, rcond=None)
            yhat = a @ coeff
            resid = y0 - yhat
            sse = float(np.sum(resid**2))
            # 条件分岐: `best is None or sse < float(best["sse"])` を満たす経路を評価する。
            if best is None or sse < float(best["sse"]):
                a0 = float(coeff[0])
                b0 = float(coeff[1])
                amp = float(math.hypot(a0, b0))
                phase = float(math.atan2(-b0, a0))
                rmse = float(math.sqrt(sse / float(len(t))))
                r2 = float("nan") if sst <= 0 else float(1.0 - sse / sst)
                best = {
                    "f_hz": float(f),
                    "tau_s": float(tau),
                    "coeff": {"a_cos": a0, "b_sin": b0},
                    "amp": amp,
                    "phase_rad": phase,
                    "rmse": rmse,
                    "r2": r2,
                    "sse": sse,
                    "n": int(len(t)),
                }

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        raise RuntimeError("fit failed (no candidates)")

    return best


def _summarize_variants(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    # 条件分岐: `not ok` を満たす経路を評価する。
    if not ok:
        return {"status": "no_ok_variants", "n_variants": len(rows)}

    f_vals = np.array([float(r["fit"]["f_hz"]) for r in ok], dtype=np.float64)
    tau_vals = np.array([float(r["fit"]["tau_s"]) for r in ok], dtype=np.float64)
    med_f = float(np.median(f_vals))
    med_tau = float(np.median(tau_vals))
    p16_f, p84_f = [float(x) for x in np.quantile(f_vals, [0.16, 0.84])]
    p16_tau, p84_tau = [float(x) for x in np.quantile(tau_vals, [0.16, 0.84])]
    best = min(ok, key=lambda r: float((r.get("fit") or {}).get("sse") or float("inf")))
    return {
        "status": "ok",
        "n_variants": len(rows),
        "n_ok": int(len(ok)),
        "best": best,
        "median": {"f_hz": med_f, "tau_s": med_tau},
        "p16_p84": {"f_hz": [p16_f, p84_f], "tau_s": [p16_tau, p84_tau]},
    }


def _parse_bandpass_list(spec: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    s = (spec or "").strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return out

    for part in s.split(","):
        p = part.strip()
        # 条件分岐: `not p` を満たす経路を評価する。
        if not p:
            continue

        # 条件分岐: `"-" not in p` を満たす経路を評価する。

        if "-" not in p:
            raise ValueError(f"invalid bandpass entry: {p}")

        lo_s, hi_s = p.split("-", 1)
        out.append((float(lo_s), float(hi_s)))

    return out


def _parse_float_list(spec: str) -> List[float]:
    out: List[float] = []
    s = (spec or "").strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return out

    for part in s.split(","):
        p = part.strip()
        # 条件分岐: `not p` を満たす経路を評価する。
        if not p:
            continue

        out.append(float(p))

    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    _set_japanese_font()

    ap = argparse.ArgumentParser(description="Ringdown QNM (damped sinusoid) fit using GWOSC public HDF5 strain.")
    ap.add_argument("--event", type=str, default="GW250114", help="GWOSC event name or prefix (default: GW250114).")
    ap.add_argument("--catalog", type=str, default="O4_Discovery_Papers", help="GWOSC catalog shortName.")
    ap.add_argument("--version", type=str, default="auto", help="GWOSC event API version (auto=v3→v2→v1).")
    ap.add_argument("--slug", type=str, default="gw250114", help="Output/data slug (default: gw250114).")
    ap.add_argument("--offline", action="store_true", help="Offline mode (use cached files only).")
    ap.add_argument("--force", action="store_true", help="Force re-download (online) and overwrite cached files.")
    ap.add_argument(
        "--method",
        type=str,
        choices=["data_release", "strain_damped_sine"],
        default="data_release",
        help="QNM source: GW250114 data release (recommended) or direct damped-sine fit on strain.",
    )
    ap.add_argument("--prefer-fs-hz", type=int, default=4096, help="Preferred sampling rate for HDF5 strain.")
    ap.add_argument("--slice-pre-s", type=float, default=1.0, help="Seconds before GPS to slice (default: 1.0).")
    ap.add_argument("--slice-post-s", type=float, default=1.0, help="Seconds after GPS to slice (default: 1.0).")
    ap.add_argument("--t0-list-s", type=str, default="0,0.002,0.004,0.006,0.008", help="Ringdown start offsets (s).")
    ap.add_argument("--fit-duration-s", type=float, default=0.05, help="Fit duration in seconds.")
    ap.add_argument(
        "--bandpass-list-hz",
        type=str,
        default="80-400,100-450,80-512,50-400",
        help="Bandpass list as 'lo-hi,...' (Hz).",
    )
    ap.add_argument("--f-step-hz", type=float, default=1.0, help="Frequency grid step (Hz).")
    ap.add_argument("--tau-min-s", type=float, default=0.0015, help="Tau grid min (s).")
    ap.add_argument("--tau-max-s", type=float, default=0.02, help="Tau grid max (s).")
    ap.add_argument("--tau-steps", type=int, default=48, help="Tau grid steps (logspace).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = _repo_root()
    data_dir = root / "data" / "gw" / str(args.slug)
    out_dir = root / "output" / "private" / "gw"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{args.slug}_ringdown_qnm_fit.json"
    out_png = out_dir / f"{args.slug}_ringdown_qnm_fit.png"

    detectors = [] if str(args.method) == "data_release" else ["H1", "L1"]
    fetch = _download_event_and_strain(
        data_dir=data_dir,
        event=str(args.event),
        catalog=str(args.catalog),
        version=str(args.version),
        detectors=detectors,
        prefer_sampling_rate_hz=int(args.prefer_fs_hz),
        offline=bool(args.offline),
        force=bool(args.force),
    )
    info = fetch["event_info"]
    gps_event = float(info.get("GPS"))

    # 条件分岐: `str(args.method) == "data_release"` を満たす経路を評価する。
    if str(args.method) == "data_release":
        preferred_url = _select_preferred_posterior(info) or ""
        record_id = _infer_zenodo_record_id_from_url(preferred_url)
        resolved = str(fetch.get("resolved_event") or str(args.event))
        event_prefix = resolved.split("_", 1)[0] if resolved else "GW250114"
        tar_name = f"{event_prefix}_data_release.tar.gz"
        tar_url = (
            _zenodo_file_url(record_id=record_id, filename=tar_name)
            if record_id is not None
            else "https://zenodo.org/api/records/16877102/files/GW250114_data_release.tar.gz/content"
        )
        tar_path = data_dir / tar_name
        # 条件分岐: `bool(args.offline)` を満たす経路を評価する。
        if bool(args.offline):
            # 条件分岐: `not tar_path.exists()` を満たす経路を評価する。
            if not tar_path.exists():
                raise FileNotFoundError("offline and missing data release tar: " + str(tar_path))
        else:
            _download(tar_url, tar_path, force=bool(args.force))

        extract_dir = data_dir / "data_release"
        members = [
            "data/220_amps_fs_gammas_merged_timescans.hdf5",
            "data/220+221_amps_fs_gammas_merged_timescans.hdf5",
        ]
        _ensure_extracted(tar_path, extract_dir=extract_dir, members=members)

        p220 = extract_dir / members[0]
        p221 = extract_dir / members[1]
        s220 = _read_pandas_hdf_block0(p220)
        s221 = _read_pandas_hdf_block0(p221)

        def by_start(samples: Dict[str, np.ndarray], *, f_key: str, g_key: str) -> List[Dict[str, Any]]:
            start = np.asarray(samples["start time [M]"], dtype=np.float64)
            out_rows: List[Dict[str, Any]] = []
            for t in sorted({float(x) for x in np.unique(start)}):
                m = np.isclose(start, float(t), rtol=0.0, atol=1e-12)
                f = np.asarray(samples[f_key][m], dtype=np.float64)
                g = np.asarray(samples[g_key][m], dtype=np.float64)
                tau = 1.0 / g
                out_rows.append(
                    {
                        "start_time_m": float(t),
                        "f_hz": _summarize_1d(f),
                        "gamma_1_s": _summarize_1d(g),
                        "tau_s": _summarize_1d(tau),
                    }
                )

            return out_rows

        rows_220 = by_start(s220, f_key="f_220", g_key="g_220")
        rows_221 = by_start(s221, f_key="f_221", g_key="g_221")

        t_ref_220 = 10.5
        t_ref_221 = 6.0

        def pick_ref(rows: List[Dict[str, Any]], t_ref: float) -> Dict[str, Any]:
            # 条件分岐: `not rows` を満たす経路を評価する。
            if not rows:
                return {"status": "no_rows"}

            chosen = min(rows, key=lambda r: abs(float(r["start_time_m"]) - float(t_ref)))
            return {"status": "ok", "t_ref_m": float(t_ref), "chosen": chosen}

        ref_220 = pick_ref(rows_220, t_ref_220)
        ref_221 = pick_ref(rows_221, t_ref_221)

        # 条件分岐: `ref_220.get("status") != "ok"` を満たす経路を評価する。
        if ref_220.get("status") != "ok":
            raise RuntimeError("ringdown 220 data missing")

        f0 = float(ref_220["chosen"]["f_hz"]["median"])
        tau0 = float(ref_220["chosen"]["tau_s"]["median"])

        out = {
            "generated_utc": _iso_utc_now(),
            "event": {
                "input": str(args.event),
                "resolved_common_name": resolved,
                "catalog": str(args.catalog),
                "gps": gps_event,
            },
            "method": "data_release",
            "sources": {
                "data_release": {"url": tar_url, "path": str(tar_path).replace("\\", "/"), "sha256": _sha256(tar_path)},
                "preferred_posterior_url": preferred_url,
            },
            "results": {
                "qnm_220": {"reference": ref_220, "by_start_time": rows_220},
                "qnm_221": {"reference": ref_221, "by_start_time": rows_221},
                "combined": {
                    "status": "ok",
                    "detectors": ["data_release"],
                    "median": {"f_hz": f0, "tau_s": tau0},
                    "note": "combined is defined as the reference 220-mode median (f, tau=1/g) from the data release.",
                },
            },
            "outputs": {"json": str(out_json).replace("\\", "/"), "png": str(out_png).replace("\\", "/")},
            "notes": [
                "GW250114 data release provides ringdown scans across start times in units of t_M (M).",
                "tau is reported as 1/g where g is the damping rate (s^-1) in the data release.",
            ],
        }
        _write_json(out_json, out)

        # Plot: f,tau vs start time (220).
        def plot_series(ax, rows: List[Dict[str, Any]], *, key: str, scale: float, ylabel: str) -> None:
            xs = np.array([r["start_time_m"] for r in rows], dtype=np.float64)
            med = np.array([float(r[key]["median"]) for r in rows], dtype=np.float64) * scale
            p16 = np.array([float(r[key]["p16_p84"][0]) for r in rows], dtype=np.float64) * scale
            p84 = np.array([float(r[key]["p16_p84"][1]) for r in rows], dtype=np.float64) * scale
            ax.fill_between(xs, p16, p84, color="#4C78A8", alpha=0.25, linewidth=0.0)
            ax.plot(xs, med, color="#4C78A8", lw=2.0, marker="o", ms=3)
            ax.axvline(t_ref_220, color="black", lw=1.0, ls="--", alpha=0.8)
            ax.set_xlabel("start time (M)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        plot_series(axes[0], rows_220, key="f_hz", scale=1.0, ylabel="f_220 (Hz)")
        plot_series(axes[1], rows_220, key="tau_s", scale=1000.0, ylabel="tau_220 (ms)")
        fig.suptitle(f"GW250114 ringdown QNM (data release): {resolved}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

        try:
            worklog.append_event(
                {
                    "event_type": f"gw_{args.slug}_ringdown_qnm_fit",
                    "argv": list(sys.argv),
                    "inputs": {"data_release_tar": str(tar_path).replace("\\", "/")},
                    "outputs": {"json": str(out_json).replace("\\", "/"), "png": str(out_png).replace("\\", "/")},
                    "summary": {"f_220_hz": f0, "tau_220_s": tau0},
                }
            )
        except Exception:
            pass

        print(f"[ok] json: {out_json}")
        print(f"[ok] png : {out_png}")
        return 0

    bandpasses = _parse_bandpass_list(str(args.bandpass_list_hz))
    t0_list = _parse_float_list(str(args.t0_list_s))
    # 条件分岐: `not bandpasses or not t0_list` を満たす経路を評価する。
    if not bandpasses or not t0_list:
        raise ValueError("bandpass/t0 list is empty")

    fit_duration_s = float(args.fit_duration_s)
    tau_grid = np.logspace(math.log10(float(args.tau_min_s)), math.log10(float(args.tau_max_s)), int(args.tau_steps))

    results: Dict[str, Any] = {}
    per_detector_best: Dict[str, Dict[str, Any]] = {}

    for det, path in (fetch.get("paths") or {}).get("strain", {}).items():
        x_raw, fs = _slice_strain_hdf5(
            Path(path),
            gps_center=gps_event,
            pre_s=float(args.slice_pre_s),
            post_s=float(args.slice_post_s),
        )
        t_rel = np.arange(len(x_raw), dtype=np.float64) / float(fs) - float(args.slice_pre_s)
        x_raw = x_raw - float(np.mean(x_raw))

        rows: List[Dict[str, Any]] = []
        for bp in bandpasses:
            try:
                x_f = _bandpass(x_raw, fs, bp[0], bp[1])
            except Exception as e:
                rows.append({"ok": False, "bandpass_hz": list(bp), "error": str(e)})
                continue

            for t0 in t0_list:
                t_start = float(t0)
                t_end = t_start + fit_duration_s
                m = (t_rel >= t_start) & (t_rel < t_end)
                # 条件分岐: `not bool(np.any(m))` を満たす経路を評価する。
                if not bool(np.any(m)):
                    rows.append({"ok": False, "bandpass_hz": list(bp), "start_s": t_start, "error": "empty_window"})
                    continue

                y = x_f[m]
                t_fit = t_rel[m] - t_start

                fmin = max(20.0, float(bp[0]))
                fmax = min(float(bp[1]), 0.45 * float(fs))
                # 条件分岐: `fmax <= fmin + float(args.f_step_hz)` を満たす経路を評価する。
                if fmax <= fmin + float(args.f_step_hz):
                    rows.append({"ok": False, "bandpass_hz": list(bp), "start_s": t_start, "error": "invalid_f_grid"})
                    continue

                f_grid = np.arange(fmin, fmax + 0.5 * float(args.f_step_hz), float(args.f_step_hz), dtype=np.float64)
                try:
                    fit = _fit_damped_sinusoid(t_fit, y, f_grid_hz=f_grid, tau_grid_s=tau_grid)
                    rows.append(
                        {
                            "ok": True,
                            "variant": FitVariant(
                                bandpass_hz=(float(bp[0]), float(bp[1])),
                                start_s=t_start,
                                duration_s=fit_duration_s,
                                fmin_hz=float(fmin),
                                fmax_hz=float(fmax),
                                tau_min_s=float(args.tau_min_s),
                                tau_max_s=float(args.tau_max_s),
                                f_step_hz=float(args.f_step_hz),
                                tau_steps=int(args.tau_steps),
                            ).to_dict(),
                            "fit": fit,
                        }
                    )
                except Exception as e:
                    rows.append({"ok": False, "variant": {"bandpass_hz": list(bp), "start_s": t_start}, "error": str(e)})

        summary = _summarize_variants(rows)
        results[det] = {
            "fs_hz": float(fs),
            "slice": {"pre_s": float(args.slice_pre_s), "post_s": float(args.slice_post_s)},
            "variants": rows,
            "summary": summary,
        }
        # 条件分岐: `summary.get("status") == "ok"` を満たす経路を評価する。
        if summary.get("status") == "ok":
            per_detector_best[det] = summary["best"]

    # Combined: use median over detectors for now (systematic floor = detector-to-detector scatter).

    comb: Dict[str, Any] = {"status": "no_detectors"}
    # 条件分岐: `per_detector_best` を満たす経路を評価する。
    if per_detector_best:
        f_list = [float(v["fit"]["f_hz"]) for v in per_detector_best.values()]
        tau_list = [float(v["fit"]["tau_s"]) for v in per_detector_best.values()]
        comb = {
            "status": "ok",
            "detectors": list(per_detector_best.keys()),
            "median": {"f_hz": float(np.median(np.array(f_list))), "tau_s": float(np.median(np.array(tau_list)))},
            "detector_scatter": {
                "f_hz": float(np.std(np.array(f_list), ddof=1)) if len(f_list) > 1 else 0.0,
                "tau_s": float(np.std(np.array(tau_list), ddof=1)) if len(tau_list) > 1 else 0.0,
            },
        }

    out = {
        "generated_utc": _iso_utc_now(),
        "event": {
            "input": str(args.event),
            "resolved_common_name": fetch.get("resolved_event"),
            "catalog": str(args.catalog),
            "gps": gps_event,
        },
        "outputs": {"json": str(out_json).replace("\\", "/"), "png": str(out_png).replace("\\", "/")},
        "sources": fetch.get("meta"),
        "results": {"per_detector": results, "combined": comb},
        "notes": [
            "QNM fit is a minimal damped-sinusoid model on bandpassed strain (per-detector).",
            "Systematics are represented by explicit sweeps over (bandpass, ringdown_start).",
        ],
    }
    _write_json(out_json, out)

    # Plot: best fit per detector (time-domain overlay).
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for ax, det in zip(axes, ["H1", "L1"]):
        rec = results.get(det) or {}
        summ = (rec.get("summary") or {}) if isinstance(rec, dict) else {}
        # 条件分岐: `summ.get("status") != "ok"` を満たす経路を評価する。
        if summ.get("status") != "ok":
            ax.set_title(f"{det}: no fit")
            ax.axis("off")
            continue

        best = summ.get("best") or {}
        var = (best.get("variant") or {})
        fit = best.get("fit") or {}
        bp = (var.get("bandpass_hz") or [float("nan"), float("nan")])
        t0 = float(var.get("start_s") or 0.0)
        dur = float(var.get("duration_s") or fit_duration_s)

        # Re-load slice for plotting (small; OK).
        path = (fetch.get("paths") or {}).get("strain", {}).get(det)
        x_raw, fs = _slice_strain_hdf5(Path(path), gps_center=gps_event, pre_s=float(args.slice_pre_s), post_s=float(args.slice_post_s))
        t_rel = np.arange(len(x_raw), dtype=np.float64) / float(fs) - float(args.slice_pre_s)
        x_raw = x_raw - float(np.mean(x_raw))
        x_f = _bandpass(x_raw, fs, float(bp[0]), float(bp[1]))
        m = (t_rel >= t0) & (t_rel < t0 + dur)
        y = x_f[m]
        t_fit = t_rel[m] - t0

        f_hz = float(fit.get("f_hz"))
        tau_s = float(fit.get("tau_s"))
        a0 = float((fit.get("coeff") or {}).get("a_cos"))
        b0 = float((fit.get("coeff") or {}).get("b_sin"))
        env = np.exp(-t_fit / tau_s)
        yhat = env * (a0 * np.cos(2 * math.pi * f_hz * t_fit) + b0 * np.sin(2 * math.pi * f_hz * t_fit))

        ax.plot(t_fit * 1000.0, y, lw=1.0, label="data (bandpass)")
        ax.plot(t_fit * 1000.0, yhat, lw=2.0, label="fit")
        ax.set_title(f"{det}: f={f_hz:.1f} Hz, tau={tau_s*1000:.2f} ms, band={bp[0]}-{bp[1]} Hz, t0={t0*1000:.1f} ms")
        ax.set_xlabel("t - t0 (ms)")
        ax.set_ylabel("strain (arb.)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"GWOSC ringdown QNM fit: {fetch.get('resolved_event')} ({args.catalog})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    try:
        worklog.append_event(
            {
                "event_type": f"gw_{args.slug}_ringdown_qnm_fit",
                "argv": list(sys.argv),
                "inputs": {"data_dir": str(data_dir).replace("\\", "/")},
                "outputs": {"json": str(out_json).replace("\\", "/"), "png": str(out_png).replace("\\", "/")},
                "summary": {"event": fetch.get("resolved_event"), "combined": comb},
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] png : {out_png}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
