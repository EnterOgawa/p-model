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


def _select_preferred_posterior_url(event_info: Dict[str, Any]) -> Optional[str]:
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
    # Zenodo API: .../files/<filename>/content
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

    # Prefer a dataset that contains the required fields.

    required = {"mass_1_source", "mass_2_source", "a_1", "a_2", "final_mass_source", "final_spin"}
    for ds in found:
        names = set(getattr(ds.dtype, "names", ()) or ())
        # 条件分岐: `required.issubset(names)` を満たす経路を評価する。
        if required.issubset(names):
            return ds

    return found[0]


def _kerr_horizon_area(m: np.ndarray, a: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    a = np.clip(a, 0.0, 0.999999)
    return 8.0 * math.pi * (m**2) * (1.0 + np.sqrt(1.0 - a**2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    _set_japanese_font()

    ap = argparse.ArgumentParser(description="Area theorem check using the GW250114 public data release (Zenodo via GWOSC).")
    ap.add_argument("--event", type=str, default="GW250114", help="GWOSC event name or prefix (default: GW250114).")
    ap.add_argument("--catalog", type=str, default="O4_Discovery_Papers", help="GWOSC catalog shortName.")
    ap.add_argument("--version", type=str, default="auto", help="GWOSC event API version (auto=v3→v2→v1).")
    ap.add_argument("--slug", type=str, default="gw250114", help="Output/data slug (default: gw250114).")
    ap.add_argument("--offline", action="store_true", help="Offline mode (use cached files only).")
    ap.add_argument("--force", action="store_true", help="Force re-download (online) and overwrite cached files.")
    ap.add_argument("--reference-inspiral-time", type=float, default=-40.0, help="Inspiral truncation time (M) to report.")
    ap.add_argument("--seed", type=int, default=250114, help="RNG seed for length-matching subsampling.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = _repo_root()
    data_dir = root / "data" / "gw" / str(args.slug)
    out_dir = root / "output" / "private" / "gw"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{args.slug}_area_theorem_test.json"
    out_png = out_dir / f"{args.slug}_area_theorem_test.png"

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

    preferred_url = _select_preferred_posterior_url(info) or ""
    record_id = _infer_zenodo_record_id_from_url(preferred_url)
    resolved = str(ev["resolved_event"])
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
        "data/area_law_inspiral_data.hdf5",
        "data/remnant_area_pyring_reweighted.npy",
        "data/ringdown_areas/220_10.5M_final_mass_spin_area.hdf5",
        "data/ringdown_areas/220+221_6M_final_mass_spin_area.hdf5",
    ]
    _ensure_extracted(tar_path, extract_dir=extract_dir, members=members)

    insp_path = extract_dir / members[0]
    pyr_path = extract_dir / members[1]
    rd220_path = extract_dir / members[2]
    rd221_path = extract_dir / members[3]

    with h5py.File(insp_path, "r") as f:
        times = np.asarray(f["times"][...], dtype=np.float64)
        insp_sets = [np.asarray(f[f"area_insp_{i}"][...], dtype=np.float64) for i in range(len(times))]

    ref_time = float(args.reference_inspiral_time)
    idxs = np.where(np.isclose(times, ref_time, rtol=0.0, atol=0.5))[0]
    # 条件分岐: `len(idxs) == 0` を満たす経路を評価する。
    if len(idxs) == 0:
        raise ValueError("reference-inspiral-time not found in times grid")

    hist_idx = int(idxs[0])
    insp_ref = insp_sets[hist_idx]

    pyr = np.asarray(np.load(pyr_path), dtype=np.float64)
    with h5py.File(rd220_path, "r") as f:
        rem_rd = np.asarray(f["Area_f"][...], dtype=np.float64)

    with h5py.File(rd221_path, "r") as f:
        rem_rd_221 = np.asarray(f["Area_f"][...], dtype=np.float64)

    rng = np.random.default_rng(int(args.seed))
    n_match = int(min(len(rem_rd), len(pyr)))
    rem_rd_sub = rng.choice(rem_rd, size=n_match, replace=False)
    pyr_sub = rng.choice(pyr, size=n_match, replace=False)
    rem_combined = np.concatenate([rem_rd_sub, pyr_sub])

    def sigma_gaussian(rem: np.ndarray, insp: np.ndarray) -> float:
        mu_r = float(np.mean(rem))
        sig_r = float(np.std(rem, ddof=0))
        mu_i = float(np.mean(insp))
        sig_i = float(np.std(insp, ddof=0))
        denom = math.sqrt(sig_r**2 + sig_i**2)
        return float("nan") if not (denom > 0) else float((mu_r - mu_i) / denom)

    sigma_ref = {
        "reference_inspiral_time": ref_time,
        "sigma_gaussian_combined": sigma_gaussian(rem_combined, insp_ref),
        "sigma_gaussian_pyring_only": sigma_gaussian(pyr, insp_ref),
        "sigma_gaussian_ringdown_only": sigma_gaussian(rem_rd, insp_ref),
        "sigma_gaussian_ringdown_220+221": sigma_gaussian(rem_rd_221, insp_ref),
    }

    sigma_by_time: List[Dict[str, Any]] = []
    sigma_pyr_by_time: List[Dict[str, Any]] = []
    sigma_rd_by_time: List[Dict[str, Any]] = []
    for t, insp in zip(times.tolist(), insp_sets):
        sigma_by_time.append({"time": float(t), "sigma": sigma_gaussian(rem_combined, insp)})
        sigma_pyr_by_time.append({"time": float(t), "sigma": sigma_gaussian(pyr, insp)})
        sigma_rd_by_time.append({"time": float(t), "sigma": sigma_gaussian(rem_rd, insp)})

    sig_vals = np.array([float(r["sigma"]) for r in sigma_by_time], dtype=np.float64)
    min_sigma = float(np.nanmin(sig_vals))
    t_first_5 = None
    for r in sigma_by_time:
        # 条件分岐: `float(r["sigma"]) >= 5.0` を満たす経路を評価する。
        if float(r["sigma"]) >= 5.0:
            t_first_5 = float(r["time"])
            break

    summary = {
        "n_inspiral_ref": int(len(insp_ref)),
        "n_ringdown_220": int(len(rem_rd)),
        "n_pyring_220": int(len(pyr)),
        "n_combined": int(len(rem_combined)),
        "sigma_ref": sigma_ref,
        "sigma_min_over_times_combined": min_sigma,
        "first_time_sigma_ge_5_combined": t_first_5,
    }

    out = {
        "generated_utc": _iso_utc_now(),
        "status": "ok",
        "event": {"resolved_common_name": resolved, "catalog": str(args.catalog), "gps": gps},
        "sources": {
            "data_release": {"url": tar_url, "path": str(tar_path).replace("\\", "/"), "sha256": _sha256(tar_path)},
            "files": {
                "inspiral_areas": {"path": str(insp_path).replace("\\", "/"), "sha256": _sha256(insp_path)},
                "pyring_areas": {"path": str(pyr_path).replace("\\", "/"), "sha256": _sha256(pyr_path)},
                "ringdown_220_areas": {"path": str(rd220_path).replace("\\", "/"), "sha256": _sha256(rd220_path)},
                "ringdown_220_221_areas": {"path": str(rd221_path).replace("\\", "/"), "sha256": _sha256(rd221_path)},
            },
            "preferred_posterior_url": preferred_url,
        },
        "summary": summary,
        "sigma_vs_inspiral_time": {
            "combined": sigma_by_time,
            "pyring_only": sigma_pyr_by_time,
            "ringdown_only": sigma_rd_by_time,
        },
        "outputs": {"json": str(out_json).replace("\\", "/"), "png": str(out_png).replace("\\", "/")},
        "notes": [
            "This reproduces the GW250114 area-law / area-theorem check from the public data release.",
            "Significance uses the Gaussian z = (mean(remnant)-mean(inspiral))/sqrt(var sum), matching the release macro script.",
            "Combined remnant distribution is a length-matched concat of ringdown(220@10.5M) and pyRing(220) samples.",
        ],
    }
    _write_json(out_json, out)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2))
    t = np.array([r["time"] for r in sigma_by_time], dtype=np.float64)
    s = np.array([r["sigma"] for r in sigma_by_time], dtype=np.float64)
    ax.plot(t, s, marker="o", ms=4, lw=2.0, label="combined (ringdown+pyRing)")
    ax.axvline(ref_time, color="black", lw=1.2, ls="--", alpha=0.8, label="reference time")
    ax.axhline(5.0, color="grey", lw=1.0, ls=":", alpha=0.8)
    ax.set_xlabel("inspiral truncation time (M)")
    ax.set_ylabel("Gaussian significance (σ)")
    ax.set_title(f"GW250114 area theorem (data release): σ_ref={sigma_ref['sigma_gaussian_combined']:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    try:
        worklog.append_event(
            {
                "event_type": f"gw_{args.slug}_area_theorem_test",
                "argv": list(sys.argv),
                "inputs": {"data_release_tar": str(tar_path).replace("\\", "/")},
                "outputs": {"json": str(out_json).replace("\\", "/"), "png": str(out_png).replace("\\", "/")},
                "summary": summary,
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
