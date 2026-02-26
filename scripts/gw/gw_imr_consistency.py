from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
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
    raise RuntimeError("h5py is required to read GWOSC posterior HDF5") from e

try:
    from scipy.stats import chi2, norm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("scipy is required for gaussian z-scores") from e

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

        # 条件分岐: `common.startswith(target) or str(k).startswith(target)` を満たす経路を評価する。

        if common.startswith(target) or str(k).startswith(target):
            snr = v.get("network_matched_filter_snr")
            try:
                snr_f = float(snr) if snr is not None else float("nan")
            except Exception:
                snr_f = float("nan")

            cand.append((snr_f, common))

    # 条件分岐: `not cand` を満たす経路を評価する。

    if not cand:
        raise ValueError(f"event '{target}' not found in catalog '{catalog}'")

    cand_sorted = sorted(cand, key=lambda t: (-(t[0] if math.isfinite(t[0]) else -1.0), t[1]))
    chosen = cand_sorted[0][1]
    # 条件分岐: `len({c for _, c in cand}) > 1` を満たす経路を評価する。
    if len({c for _, c in cand}) > 1:
        print(f"[warn] multiple matches for '{target}', picked: {chosen}")

    return chosen


def _fetch_event_json(
    *,
    data_dir: Path,
    event: str,
    catalog: str,
    version: str,
    offline: bool,
    force: bool,
) -> Dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)

    resolved_event = _resolve_event_common_name(catalog=catalog, event=event)
    event_json_path = data_dir / f"{resolved_event}_event.json"

    event_json_url = ""
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

    obj = _read_json(event_json_path)
    events = obj.get("events") or {}
    # 条件分岐: `not isinstance(events, dict) or not events` を満たす経路を評価する。
    if not isinstance(events, dict) or not events:
        raise ValueError("invalid event JSON: missing 'events'")

    event_key = next(iter(events.keys()))
    event_info = events[event_key]
    # 条件分岐: `not isinstance(event_info, dict)` を満たす経路を評価する。
    if not isinstance(event_info, dict):
        raise ValueError("invalid event JSON: event is not an object")

    return {
        "resolved_event": resolved_event,
        "event_key": str(event_key),
        "event_info": event_info,
        "event_json_path": event_json_path,
        "event_json_url": event_json_url or _gwosc_event_json_url(catalog=catalog, event=resolved_event, version="v1"),
    }


def _infer_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    for i, p in enumerate(parts):
        # 条件分岐: `p == "files" and i + 1 < len(parts)` を満たす経路を評価する。
        if p == "files" and i + 1 < len(parts):
            return parts[i + 1]

    name = Path(parsed.path).name
    # 条件分岐: `name and name != "content"` を満たす経路を評価する。
    if name and name != "content":
        return name

    return "posterior_samples.h5"


def _select_preferred_posterior(event_info: Dict[str, Any], *, prefer_waveform: str) -> Tuple[str, Dict[str, Any]]:
    params = event_info.get("parameters") or {}
    # 条件分岐: `not isinstance(params, dict) or not params` を満たす経路を評価する。
    if not isinstance(params, dict) or not params:
        raise ValueError("event JSON missing parameters dict")

    prefer = (prefer_waveform or "").strip()
    entries: List[Tuple[int, str, str, Dict[str, Any]]] = []
    for name, rec in params.items():
        # 条件分岐: `not isinstance(rec, dict)` を満たす経路を評価する。
        if not isinstance(rec, dict):
            continue

        data_url = str(rec.get("data_url") or "").strip()
        # 条件分岐: `not data_url` を満たす経路を評価する。
        if not data_url:
            continue

        wf = str(rec.get("waveform_family") or "").strip()
        # 条件分岐: `prefer and wf and wf != prefer` を満たす経路を評価する。
        if prefer and wf and wf != prefer:
            continue

        is_pref = 1 if bool(rec.get("is_preferred")) else 0
        date = str(rec.get("date_added") or "").strip()
        entries.append((is_pref, date, str(name), rec))

    # 条件分岐: `not entries` を満たす経路を評価する。

    if not entries:
        raise ValueError("no posterior samples found in event parameters")

    entries_sorted = sorted(entries, key=lambda t: (-t[0], t[1], t[2]))
    chosen = entries_sorted[0]
    return chosen[2], chosen[3]


def _find_posterior_samples_dataset(h5: h5py.File) -> h5py.Dataset:
    found: List[h5py.Dataset] = []

    def visitor(name: str, obj: Any) -> None:
        # 条件分岐: `not isinstance(obj, h5py.Dataset)` を満たす経路を評価する。
        if not isinstance(obj, h5py.Dataset):
            return

        # 条件分岐: `name.endswith("posterior_samples")` を満たす経路を評価する。

        if name.endswith("posterior_samples"):
            found.append(obj)

    h5.visititems(visitor)
    # 条件分岐: `not found` を満たす経路を評価する。
    if not found:
        raise ValueError("posterior_samples dataset not found in HDF5")

    required = {"final_mass", "final_spin"}
    for ds in found:
        names = set(getattr(ds.dtype, "names", ()) or ())
        # 条件分岐: `required.issubset(names)` を満たす経路を評価する。
        if required.issubset(names):
            return ds

    return found[0]


# Berti+ (2006) fit for l=m=2, n=0 (fundamental) mode.

def _qnm_220_omega_m(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = np.clip(a, 0.0, 0.999999)
    return 1.5251 - 1.1568 * (1.0 - a) ** 0.1292


def _qnm_220_quality_factor(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = np.clip(a, 0.0, 0.999999)
    return 0.7000 + 1.4187 * (1.0 - a) ** (-0.4990)


def _qnm_220_f_tau_hz_s(*, m_det_msun: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = 6.67430e-11
    c = 299792458.0
    m_sun = 1.98847e30
    m_kg = np.asarray(m_det_msun, dtype=np.float64) * m_sun
    t_m = g * m_kg / (c**3)  # seconds
    omega_m = _qnm_220_omega_m(a)
    f_hz = omega_m / (2.0 * math.pi * t_m)
    q = _qnm_220_quality_factor(a)
    tau_s = q / (math.pi * f_hz)
    return f_hz, tau_s


def _invert_qnm_220_to_spin(Q: float) -> Optional[float]:
    # Q = 0.7000 + 1.4187*(1-a)^(-0.4990)
    if not (Q > 0.7000):
        return None

    x = 1.4187 / (Q - 0.7000)
    one_minus_a = x ** (1.0 / 0.4990)
    a = 1.0 - one_minus_a
    # 条件分岐: `not math.isfinite(a)` を満たす経路を評価する。
    if not math.isfinite(a):
        return None

    return float(min(0.999999, max(0.0, a)))


def _invert_qnm_220_to_mass_msun(*, f_hz: float, a: float) -> Optional[float]:
    # 条件分岐: `not (f_hz > 0.0)` を満たす経路を評価する。
    if not (f_hz > 0.0):
        return None

    g = 6.67430e-11
    c = 299792458.0
    m_sun = 1.98847e30
    omega_m = float(_qnm_220_omega_m(np.array([a]))[0])
    m_kg = (omega_m * (c**3)) / (2.0 * math.pi * g * float(f_hz))
    return float(m_kg / m_sun)


def _median_and_sigma_from_p16_p84(p16_p84: Sequence[float]) -> Tuple[float, float]:
    # 条件分岐: `not isinstance(p16_p84, (list, tuple)) or len(p16_p84) != 2` を満たす経路を評価する。
    if not isinstance(p16_p84, (list, tuple)) or len(p16_p84) != 2:
        return float("nan"), float("nan")

    p16, p84 = float(p16_p84[0]), float(p16_p84[1])
    med = 0.5 * (p16 + p84)
    sig = 0.5 * (p84 - p16)
    return med, sig


def main(argv: Optional[Sequence[str]] = None) -> int:
    _set_japanese_font()

    ap = argparse.ArgumentParser(description="IMR consistency proxy: ringdown QNM vs GWOSC posterior final mass/spin.")
    ap.add_argument("--event", type=str, default="GW250114", help="GWOSC event name or prefix (default: GW250114).")
    ap.add_argument("--catalog", type=str, default="O4_Discovery_Papers", help="GWOSC catalog shortName.")
    ap.add_argument("--version", type=str, default="auto", help="GWOSC event API version (auto=v3→v2→v1).")
    ap.add_argument("--slug", type=str, default="gw250114", help="Output/data slug (default: gw250114).")
    ap.add_argument("--offline", action="store_true", help="Offline mode (use cached files only).")
    ap.add_argument("--force", action="store_true", help="Force re-download (online) and overwrite cached files.")
    ap.add_argument("--prefer-waveform", type=str, default="", help="Prefer waveform_family (default: use is_preferred).")
    ap.add_argument(
        "--ringdown-json",
        type=str,
        default="",
        help="Optional path to ringdown fit json (default: output/private/gw/<slug>_ringdown_qnm_fit.json).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = _repo_root()
    data_dir = root / "data" / "gw" / str(args.slug)
    out_dir = root / "output" / "private" / "gw"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{args.slug}_imr_consistency.json"
    out_png = out_dir / f"{args.slug}_imr_consistency.png"

    ring_json_path = Path(args.ringdown_json) if str(args.ringdown_json).strip() else (out_dir / f"{args.slug}_ringdown_qnm_fit.json")
    # 条件分岐: `not ring_json_path.exists()` を満たす経路を評価する。
    if not ring_json_path.exists():
        out = {
            "generated_utc": _iso_utc_now(),
            "status": "blocked_missing_input",
            "reason": f"missing ringdown json: {str(ring_json_path).replace('\\\\','/')}",
        }
        _write_json(out_json, out)
        print(f"[blocked] {out['reason']}")
        return 2

    ring = _read_json(ring_json_path)
    comb = ((ring.get("results") or {}).get("combined") or {}) if isinstance(ring.get("results") or {}, dict) else {}
    # 条件分岐: `str(comb.get("status") or "") != "ok"` を満たす経路を評価する。
    if str(comb.get("status") or "") != "ok":
        out = {
            "generated_utc": _iso_utc_now(),
            "status": "blocked_missing_input",
            "reason": "ringdown combined status is not ok",
        }
        _write_json(out_json, out)
        print("[blocked] ringdown combined status not ok")
        return 2

    f_meas = float(((comb.get("median") or {}).get("f_hz")) or float("nan"))
    tau_meas = float(((comb.get("median") or {}).get("tau_s")) or float("nan"))
    # 条件分岐: `not (math.isfinite(f_meas) and math.isfinite(tau_meas))` を満たす経路を評価する。
    if not (math.isfinite(f_meas) and math.isfinite(tau_meas)):
        raise ValueError("invalid ringdown measurement in json")

    # Measurement uncertainty proxy:
    # - statistical: half-width of (p16,p84) at the reference ringdown start time (from data release)
    # - systematic: scatter of medians across start-time scan

    res = (ring.get("results") or {}) if isinstance(ring.get("results") or {}, dict) else {}
    q220 = (res.get("qnm_220") or {}) if isinstance(res.get("qnm_220") or {}, dict) else {}
    ref = (q220.get("reference") or {}) if isinstance(q220.get("reference") or {}, dict) else {}
    chosen = (ref.get("chosen") or {}) if isinstance(ref.get("chosen") or {}, dict) else {}
    f_p16_p84 = ((chosen.get("f_hz") or {}).get("p16_p84") or []) if isinstance(chosen.get("f_hz") or {}, dict) else []
    tau_p16_p84 = ((chosen.get("tau_s") or {}).get("p16_p84") or []) if isinstance(chosen.get("tau_s") or {}, dict) else []
    _, sig_f_stat = _median_and_sigma_from_p16_p84(f_p16_p84)
    _, sig_tau_stat = _median_and_sigma_from_p16_p84(tau_p16_p84)

    rows = q220.get("by_start_time") if isinstance(q220.get("by_start_time"), list) else []
    f_meds = np.array([float((r.get("f_hz") or {}).get("median")) for r in rows if isinstance(r, dict)], dtype=np.float64)
    tau_meds = np.array([float((r.get("tau_s") or {}).get("median")) for r in rows if isinstance(r, dict)], dtype=np.float64)
    sig_f_sys = float(np.std(f_meds, ddof=1)) if len(f_meds) > 2 else 0.0
    sig_tau_sys = float(np.std(tau_meds, ddof=1)) if len(tau_meds) > 2 else 0.0

    sig_f_meas = float(math.sqrt(max(0.0, sig_f_stat) ** 2 + sig_f_sys**2))
    sig_tau_meas = float(math.sqrt(max(0.0, sig_tau_stat) ** 2 + sig_tau_sys**2))

    ev = _fetch_event_json(
        data_dir=data_dir,
        event=str(args.event),
        catalog=str(args.catalog),
        version=str(args.version),
        offline=bool(args.offline),
        force=bool(args.force),
    )
    info = ev["event_info"]
    gps = float(info.get("GPS"))

    pe_name, pe = _select_preferred_posterior(info, prefer_waveform=str(args.prefer_waveform))
    data_url = str(pe.get("data_url") or "").strip()
    # 条件分岐: `not data_url` を満たす経路を評価する。
    if not data_url:
        raise ValueError("selected posterior missing data_url")

    fname = _infer_filename_from_url(data_url)
    # 条件分岐: `not fname.lower().endswith(".h5") and not fname.lower().endswith(".hdf5")` を満たす経路を評価する。
    if not fname.lower().endswith(".h5") and not fname.lower().endswith(".hdf5"):
        fname = f"{fname}.h5"

    post_path = data_dir / fname
    # 条件分岐: `bool(args.offline)` を満たす経路を評価する。
    if bool(args.offline):
        # 条件分岐: `not post_path.exists()` を満たす経路を評価する。
        if not post_path.exists():
            raise FileNotFoundError("offline and missing posterior: " + str(post_path))
    else:
        _download(data_url, post_path, force=bool(args.force))

    with h5py.File(post_path, "r") as f:
        ds = _find_posterior_samples_dataset(f)
        arr = ds[...]

    m_det = np.asarray(arr["final_mass"], dtype=np.float64)
    a_f = np.asarray(arr["final_spin"], dtype=np.float64)
    f_pred, tau_pred = _qnm_220_f_tau_hz_s(m_det_msun=m_det, a=a_f)

    def q(x: np.ndarray, probs: Sequence[float]) -> List[float]:
        return [float(v) for v in np.quantile(x, probs)]

    pred_summary = {
        "f_hz": {"median": float(np.median(f_pred)), "p16_p84": q(f_pred, [0.16, 0.84])},
        "tau_s": {"median": float(np.median(tau_pred)), "p16_p84": q(tau_pred, [0.16, 0.84])},
        "final_mass_det_msun": {"median": float(np.median(m_det)), "p16_p84": q(m_det, [0.16, 0.84])},
        "final_spin": {"median": float(np.median(a_f)), "p16_p84": q(a_f, [0.16, 0.84])},
    }

    _, sig_f_pred = _median_and_sigma_from_p16_p84(pred_summary["f_hz"]["p16_p84"])
    _, sig_tau_pred = _median_and_sigma_from_p16_p84(pred_summary["tau_s"]["p16_p84"])

    def z_score(x: float, mu: float, sig1: float, sig2: float) -> float:
        sig = math.sqrt(max(0.0, float(sig1)) ** 2 + max(0.0, float(sig2)) ** 2)
        # 条件分岐: `not (sig > 0.0)` を満たす経路を評価する。
        if not (sig > 0.0):
            return float("nan")

        return (float(x) - float(mu)) / sig

    z_f = z_score(f_meas, pred_summary["f_hz"]["median"], sig_f_meas, sig_f_pred)
    z_tau = z_score(tau_meas, pred_summary["tau_s"]["median"], sig_tau_meas, sig_tau_pred)

    # Ringdown-inferred (M,a) from measured (f,tau) for mode 220.
    q_meas = math.pi * f_meas * tau_meas
    a_rd = _invert_qnm_220_to_spin(q_meas)
    m_rd = _invert_qnm_220_to_mass_msun(f_hz=f_meas, a=a_rd) if a_rd is not None else None

    rd_inferred = {"mode": "220", "Q": q_meas, "final_spin": a_rd, "final_mass_det_msun": m_rd}

    # Propagate (f,tau) uncertainty → (M,a) by Monte Carlo for a more meaningful consistency check.
    rd_mc: Dict[str, Any] = {"status": "not_computed"}
    # 条件分岐: `sig_f_meas > 0 and sig_tau_meas > 0` を満たす経路を評価する。
    if sig_f_meas > 0 and sig_tau_meas > 0:
        g = 6.67430e-11
        c = 299792458.0
        m_sun = 1.98847e30
        rng = np.random.default_rng(0)
        n_mc = 8000
        f_s = rng.normal(f_meas, sig_f_meas, size=n_mc)
        tau_s = rng.normal(tau_meas, sig_tau_meas, size=n_mc)
        ok = (f_s > 0.0) & (tau_s > 0.0)
        q_s = math.pi * f_s * tau_s
        ok &= q_s > 0.7000
        f_s = f_s[ok]
        tau_s = tau_s[ok]
        q_s = q_s[ok]
        # 条件分岐: `len(f_s) > 50` を満たす経路を評価する。
        if len(f_s) > 50:
            one_minus_a = (1.4187 / (q_s - 0.7000)) ** (1.0 / 0.4990)
            a_s = np.clip(1.0 - one_minus_a, 0.0, 0.999999)
            omega_m = _qnm_220_omega_m(a_s)
            m_kg = (omega_m * (c**3)) / (2.0 * math.pi * g * f_s)
            m_s = m_kg / m_sun
            ok2 = np.isfinite(m_s) & np.isfinite(a_s) & (m_s > 0)
            m_s = np.asarray(m_s[ok2], dtype=np.float64)
            a_s = np.asarray(a_s[ok2], dtype=np.float64)
            rd_mc = {
                "status": "ok",
                "n_mc": int(n_mc),
                "n_ok": int(len(m_s)),
                "final_mass_det_msun": {"median": float(np.median(m_s)), "p16_p84": q(m_s, [0.16, 0.84])},
                "final_spin": {"median": float(np.median(a_s)), "p16_p84": q(a_s, [0.16, 0.84])},
            }

    # Compare ringdown-inferred to PE posterior (Gaussian; include ringdown MC variance when available).

    _, sig_m_det = _median_and_sigma_from_p16_p84(pred_summary["final_mass_det_msun"]["p16_p84"])
    _, sig_a_f = _median_and_sigma_from_p16_p84(pred_summary["final_spin"]["p16_p84"])
    _, sig_m_rd = _median_and_sigma_from_p16_p84(((rd_mc.get("final_mass_det_msun") or {}).get("p16_p84") or []))
    _, sig_a_rd = _median_and_sigma_from_p16_p84(((rd_mc.get("final_spin") or {}).get("p16_p84") or []))
    m_cmp = float((rd_mc.get("final_mass_det_msun") or {}).get("median")) if rd_mc.get("status") == "ok" else m_rd
    a_cmp = float((rd_mc.get("final_spin") or {}).get("median")) if rd_mc.get("status") == "ok" else a_rd
    z_m_1d = z_score(m_cmp, pred_summary["final_mass_det_msun"]["median"], sig_m_rd, sig_m_det) if m_cmp is not None else float("nan")
    z_a_1d = z_score(a_cmp, pred_summary["final_spin"]["median"], sig_a_rd, sig_a_f) if a_cmp is not None else float("nan")

    # 2D Mahalanobis on (M,a) between PE mean and ringdown MC mean (covariances added).
    md2 = float("nan")
    p_md2 = float("nan")
    # 条件分岐: `rd_mc.get("status") == "ok" and len(m_det) > 50` を満たす経路を評価する。
    if rd_mc.get("status") == "ok" and len(m_det) > 50:
        mu_pe = np.array([float(np.mean(m_det)), float(np.mean(a_f))], dtype=np.float64)
        cov_pe = np.cov(np.vstack([m_det, a_f]), ddof=1)
        mu_rd = np.array([float(m_cmp), float(a_cmp)], dtype=np.float64)
        # Approximate ringdown covariance from MC p16/p84 (diagonal), to avoid storing full samples.
        cov_rd = np.diag([float(sig_m_rd) ** 2, float(sig_a_rd) ** 2])
        cov_tot = cov_pe + cov_rd
        try:
            inv = np.linalg.inv(cov_tot)
            diff = mu_rd - mu_pe
            md2 = float(diff.T @ inv @ diff)
            p_md2 = float(chi2.sf(md2, df=2))
        except Exception:
            md2 = float("nan")
            p_md2 = float("nan")

    out = {
        "generated_utc": _iso_utc_now(),
        "status": "ok",
        "event": {"resolved_common_name": ev["resolved_event"], "catalog": str(args.catalog), "gps": gps},
        "inputs": {
            "ringdown_json": str(ring_json_path).replace("\\", "/"),
            "event_json": str(ev["event_json_path"]).replace("\\", "/"),
            "posterior": str(post_path).replace("\\", "/"),
        },
        "sources": {
            "event_json_url": ev["event_json_url"],
            "posterior_url": data_url,
            "posterior_ref": (pe.get("links") or {}).get("Data products in Zenodo"),
            "sha256": {
                "ringdown_json": _sha256(ring_json_path),
                "event_json": _sha256(ev["event_json_path"]),
                "posterior": _sha256(post_path),
            },
            "selection": {
                "posterior_key": pe_name,
                "posterior_meta": {k: pe.get(k) for k in ["waveform_family", "date_added", "is_preferred"] if k in pe},
            },
        },
        "ringdown_measurement": {
            "f_hz": f_meas,
            "tau_s": tau_meas,
            "sigma_proxy": {"f_hz": sig_f_meas, "tau_s": sig_tau_meas},
        },
        "gr_prediction_from_pe": pred_summary,
        "consistency": {
            "mode": "220",
            "z_f": z_f,
            "z_tau": z_tau,
            "ringdown_inferred_final": rd_inferred,
            "ringdown_inferred_final_mc": rd_mc,
            "z_final_mass_det_1d": z_m_1d,
            "z_final_spin_1d": z_a_1d,
            "mahalanobis2_m_a": md2,
            "p_value_mahalanobis2": p_md2,
        },
        "outputs": {"json": str(out_json).replace("\\", "/"), "png": str(out_png).replace("\\", "/")},
        "notes": [
            "This is an IMR-consistency *proxy*: ringdown 220 fit vs full IMR PE posterior (final mass/spin) on GWOSC.",
            "A full IMR split (inspiral vs post-inspiral posteriors) is not available in this payload; add when public.",
        ],
    }
    _write_json(out_json, out)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    ax_f, ax_tau, ax_m, ax_a = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax_f.hist(f_pred, bins=80, color="#4C78A8", alpha=0.8, density=True)
    ax_f.axvline(f_meas, color="black", lw=1.5, label="measured")
    ax_f.set_title(f"QNM f220: pred vs meas (z={z_f:.2f})")
    ax_f.set_xlabel("f (Hz)")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(fontsize=9)

    ax_tau.hist(tau_pred * 1000.0, bins=80, color="#F58518", alpha=0.8, density=True)
    ax_tau.axvline(tau_meas * 1000.0, color="black", lw=1.5, label="measured")
    ax_tau.set_title(f"QNM tau220: pred vs meas (z={z_tau:.2f})")
    ax_tau.set_xlabel("tau (ms)")
    ax_tau.grid(True, alpha=0.3)
    ax_tau.legend(fontsize=9)

    ax_m.hist(m_det, bins=80, color="#54A24B", alpha=0.8, density=True)
    # 条件分岐: `m_cmp is not None and math.isfinite(float(m_cmp))` を満たす経路を評価する。
    if m_cmp is not None and math.isfinite(float(m_cmp)):
        ax_m.axvline(float(m_cmp), color="black", lw=1.5, label="ringdown inferred")

    ax_m.set_title(f"Final mass (det)  z1d={z_m_1d:.2f}")
    ax_m.set_xlabel("M_f (M_sun)")
    ax_m.grid(True, alpha=0.3)
    ax_m.legend(fontsize=9)

    ax_a.hist(a_f, bins=80, color="#B279A2", alpha=0.8, density=True)
    # 条件分岐: `a_cmp is not None and math.isfinite(float(a_cmp))` を満たす経路を評価する。
    if a_cmp is not None and math.isfinite(float(a_cmp)):
        ax_a.axvline(float(a_cmp), color="black", lw=1.5, label="ringdown inferred")

    ax_a.set_title(f"Final spin  z1d={z_a_1d:.2f}")
    ax_a.set_xlabel("a_f")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=9)

    fig.suptitle(f"GW250114 IMR consistency proxy: {ev['resolved_event']} ({args.catalog})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    try:
        worklog.append_event(
            {
                "event_type": f"gw_{args.slug}_imr_consistency",
                "argv": list(sys.argv),
                "inputs": {"ringdown_json": str(ring_json_path).replace("\\", "/")},
                "outputs": {"json": str(out_json).replace("\\", "/"), "png": str(out_png).replace("\\", "/")},
                "summary": out.get("consistency"),
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
