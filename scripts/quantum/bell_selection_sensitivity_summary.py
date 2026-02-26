from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _manifest_brief(path: Path) -> Optional[Dict[str, Any]]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    m = _read_json(path)
    files = m.get("files") if isinstance(m.get("files"), list) else []
    return {
        "manifest": str(path),
        "dataset": m.get("dataset"),
        "n_files": int(len(files)),
    }


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。

    if not math.isfinite(v):
        return None

    return v


def _min_max(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    v = [x for x in values if math.isfinite(x)]
    # 条件分岐: `not v` を満たす経路を評価する。
    if not v:
        return None, None

    return float(min(v)), float(max(v))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 7 / Step 7.4: summarize selection sensitivity across representative Bell datasets "
            "(NIST time-tag + trial-based, Weihs 1998 time-tag, Delft event-ready, optional Kwiat2013). "
            "This script reads the fixed products under output/public/quantum/bell/*."
        )
    )
    ap.add_argument(
        "--out-tag",
        default="bell_selection_sensitivity_summary",
        help="Output tag for png/json (default: bell_selection_sensitivity_summary).",
    )
    # Backward-compatible args (older versions used 'nist-trial-tag' and 'weihs-out-tag').
    ap.add_argument(
        "--nist-trial-tag",
        default="03_43_afterfixingModeLocking_s3600",
        help="(deprecated) Trial tag used by older outputs; ignored if --nist-dataset-id is set.",
    )
    ap.add_argument(
        "--nist-dataset-id",
        default=None,
        help="Dataset id under output/public/quantum/bell/ (default: derived from --nist-trial-tag).",
    )
    ap.add_argument(
        "--weihs-out-tag",
        default="weihs1998_longdist_longdist1",
        help="(deprecated) Older sweep out_tag; ignored if --weihs-dataset-id is set.",
    )
    ap.add_argument(
        "--weihs-dataset-id",
        default=None,
        help="Dataset id under output/public/quantum/bell/ (default: --weihs-out-tag).",
    )
    ap.add_argument(
        "--delft2015-dataset-id",
        default="delft_hensen2015",
        help="Dataset id under output/public/quantum/bell/ (default: delft_hensen2015).",
    )
    ap.add_argument(
        "--delft2016-dataset-id",
        default="delft_hensen2016_srep30289",
        help="Dataset id under output/public/quantum/bell/ (default: delft_hensen2016_srep30289).",
    )
    ap.add_argument(
        "--kwiat-dataset-id",
        default="kwiat2013_prl111_130406_05082013_15",
        help="(optional) Dataset id under output/public/quantum/bell/ (default: kwiat2013_prl111_130406_05082013_15).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)
    bell_dir = out_dir / "bell"

    # Source manifests (best-effort; used for reproducibility context).
    nist_src_manifest = _manifest_brief(root / "data" / "quantum" / "sources" / "nist_belltestdata" / "manifest.json")
    weihs_src_manifest = _manifest_brief(root / "data" / "quantum" / "sources" / "zenodo_7185335" / "manifest.json")
    delft15_src_manifest = _manifest_brief(root / "data" / "quantum" / "sources" / "delft_hensen2015" / "manifest.json")
    delft16_src_manifest = _manifest_brief(root / "data" / "quantum" / "sources" / "delft_hensen2016_srep30289" / "manifest.json")
    kwiat_src_manifest = _manifest_brief(root / "data" / "quantum" / "sources" / "kwiat2013_prl111_130406" / "manifest.json")

    giustina_src_dir = root / "data" / "quantum" / "sources" / "giustina2015_prl115_250401"
    giustina_manifest_path = giustina_src_dir / "manifest.json"
    giustina_src_manifest = _manifest_brief(giustina_manifest_path)
    giustina_block_info: Dict[str, Any] = {
        "status": "blocked",
        "reason": (
            "Raw photon time-tag click logs are not present in the public paper/supplement bundle. "
            "Data availability indicates data are available upon request; APS supplemental is also "
            "often blocked by Cloudflare in automated environments."
        ),
        "source_manifest": giustina_src_manifest or ({"manifest": str(giustina_manifest_path)} if giustina_src_dir.exists() else None),
    }
    # 条件分岐: `giustina_src_manifest is not None` を満たす経路を評価する。
    if giustina_src_manifest is not None:
        mf = _read_json(giustina_manifest_path)
        attempt = mf.get("aps_supplemental_attempt") if isinstance(mf.get("aps_supplemental_attempt"), dict) else {}
        giustina_block_info["aps_supplemental_http_status"] = int(attempt.get("http_status", 0) or 0) if attempt else None
        giustina_block_info["aps_supplemental_blocked_by_cloudflare"] = (
            bool(attempt.get("blocked_by_cloudflare")) if isinstance(attempt.get("blocked_by_cloudflare"), bool) else None
        )

    nist_dataset_id = str(args.nist_dataset_id) if args.nist_dataset_id else str(args.nist_trial_tag)
    # 条件分岐: `not nist_dataset_id.startswith("nist_")` を満たす経路を評価する。
    if not nist_dataset_id.startswith("nist_"):
        nist_dataset_id = "nist_" + nist_dataset_id

    weihs_dataset_id = str(args.weihs_dataset_id) if args.weihs_dataset_id else str(args.weihs_out_tag)
    delft15_dataset_id = str(args.delft2015_dataset_id)
    delft16_dataset_id = str(args.delft2016_dataset_id)
    kwiat_dataset_id = str(args.kwiat_dataset_id)

    # --- Load NIST (fixed bell products)
    nist_ws_path = bell_dir / nist_dataset_id / "window_sweep_metrics.json"
    nist_cov_path = bell_dir / nist_dataset_id / "covariance.json"
    # 条件分岐: `not nist_ws_path.exists()` を満たす経路を評価する。
    if not nist_ws_path.exists():
        raise SystemExit(f"[fail] missing NIST window sweep: {nist_ws_path}")

    nist_ws = _read_json(nist_ws_path)
    nist_rows = nist_ws.get("rows") if isinstance(nist_ws.get("rows"), list) else []
    nist_windows_ns: List[float] = []
    nist_j_sweep: List[float] = []
    nist_j_sweep_sub: List[float] = []
    for r in nist_rows:
        w = _safe_float(r.get("window_ns"))
        j = _safe_float(r.get("J_prob"))
        # 条件分岐: `w is None or j is None` を満たす経路を評価する。
        if w is None or j is None:
            continue

        nist_windows_ns.append(float(w))
        nist_j_sweep.append(float(j))
        jsub = _safe_float(r.get("J_prob_subtracted_clipped"))
        nist_j_sweep_sub.append(float(jsub) if jsub is not None else float("nan"))

    # 条件分岐: `(not nist_windows_ns) or (len(nist_windows_ns) != len(nist_j_sweep))` を満たす経路を評価する。

    if (not nist_windows_ns) or (len(nist_windows_ns) != len(nist_j_sweep)):
        raise SystemExit("[fail] NIST window_sweep_metrics.json missing rows.window_ns / rows.J_prob")

    nist_trial_best = _safe_float(
        ((nist_ws.get("trial_based") or {}) if isinstance(nist_ws.get("trial_based"), dict) else {}).get("J_prob")
    )
    nist_j_min, nist_j_max = _min_max(nist_j_sweep)
    nist_j_sub_min, nist_j_sub_max = _min_max(nist_j_sweep_sub)
    ksa = _safe_float((nist_ws.get("time_tag_delay_ks") or {}).get("alice") if isinstance(nist_ws.get("time_tag_delay_ks"), dict) else None)
    ksb = _safe_float((nist_ws.get("time_tag_delay_ks") or {}).get("bob") if isinstance(nist_ws.get("time_tag_delay_ks"), dict) else None)

    nist_rec_w = None
    nist_rec_method = None
    # 条件分岐: `nist_cov_path.exists()` を満たす経路を評価する。
    if nist_cov_path.exists():
        nist_cov = _read_json(nist_cov_path)
        nw = nist_cov.get("natural_window") if isinstance(nist_cov.get("natural_window"), dict) else {}
        nist_rec_w = _safe_float(nw.get("recommended_window_ns"))
        nist_rec_method = nw.get("method")

    # --- Load Kwiat/Christensen 2013 (optional; fixed bell products)

    kwiat_ws_path = bell_dir / kwiat_dataset_id / "window_sweep_metrics.json"
    kwiat_cov_path = bell_dir / kwiat_dataset_id / "covariance.json"
    kwiat_windows_ns: List[float] = []
    kwiat_j_sweep: List[float] = []
    kwiat_j_min = None
    kwiat_j_max = None
    kwiat_trial_best = None
    kwiat_rec_w = None
    kwiat_rec_method = None
    # 条件分岐: `kwiat_ws_path.exists()` を満たす経路を評価する。
    if kwiat_ws_path.exists():
        kwiat_ws = _read_json(kwiat_ws_path)
        kwiat_rows = kwiat_ws.get("rows") if isinstance(kwiat_ws.get("rows"), list) else []
        kwiat_windows_ns = [float(r.get("window_ns")) for r in kwiat_rows if _safe_float(r.get("window_ns")) is not None]
        kwiat_j_sweep = [float(r.get("J_prob")) for r in kwiat_rows if _safe_float(r.get("J_prob")) is not None]
        # 条件分岐: `kwiat_windows_ns and (len(kwiat_windows_ns) == len(kwiat_j_sweep))` を満たす経路を評価する。
        if kwiat_windows_ns and (len(kwiat_windows_ns) == len(kwiat_j_sweep)):
            kwiat_j_min, kwiat_j_max = _min_max(kwiat_j_sweep)

        kwiat_trial_best = _safe_float(
            ((kwiat_ws.get("trial_based") or {}) if isinstance(kwiat_ws.get("trial_based"), dict) else {}).get("J_prob")
        )
        # 条件分岐: `kwiat_cov_path.exists()` を満たす経路を評価する。
        if kwiat_cov_path.exists():
            kwiat_cov = _read_json(kwiat_cov_path)
            nw = kwiat_cov.get("natural_window") if isinstance(kwiat_cov.get("natural_window"), dict) else {}
            kwiat_rec_w = _safe_float(nw.get("recommended_window_ns"))
            kwiat_rec_method = nw.get("method")

    # --- Load Weihs 1998 (fixed bell products)

    weihs_ws_path = bell_dir / weihs_dataset_id / "window_sweep_metrics.json"
    weihs_cov_path = bell_dir / weihs_dataset_id / "covariance.json"
    # 条件分岐: `not weihs_ws_path.exists()` を満たす経路を評価する。
    if not weihs_ws_path.exists():
        raise SystemExit(f"[fail] missing Weihs window sweep: {weihs_ws_path}")

    weihs_ws = _read_json(weihs_ws_path)
    weihs_rows = weihs_ws.get("rows") if isinstance(weihs_ws.get("rows"), list) else []
    weihs_x: List[float] = []
    weihs_y: List[float] = []
    weihs_y_sub: List[float] = []
    for r in weihs_rows:
        w = _safe_float(r.get("window_ns"))
        y = _safe_float(r.get("S_fixed_abs"))
        # 条件分岐: `w is None or y is None` を満たす経路を評価する。
        if w is None or y is None:
            continue

        weihs_x.append(float(w))
        weihs_y.append(float(y))
        ysub = _safe_float(r.get("S_fixed_accidental_subtracted_abs"))
        weihs_y_sub.append(float(ysub) if ysub is not None else float("nan"))

    # 条件分岐: `(not weihs_x) or (len(weihs_x) != len(weihs_y))` を満たす経路を評価する。

    if (not weihs_x) or (len(weihs_x) != len(weihs_y)):
        raise SystemExit("[fail] Weihs window_sweep_metrics.json missing rows.window_ns / rows.S_fixed_abs")

    weihs_s_min, weihs_s_max = _min_max(weihs_y)
    weihs_s_sub_min, weihs_s_sub_max = _min_max(weihs_y_sub)

    weihs_rec_w = None
    weihs_rec_method = None
    # 条件分岐: `weihs_cov_path.exists()` を満たす経路を評価する。
    if weihs_cov_path.exists():
        weihs_cov = _read_json(weihs_cov_path)
        nw = weihs_cov.get("natural_window") if isinstance(weihs_cov.get("natural_window"), dict) else {}
        weihs_rec_w = _safe_float(nw.get("recommended_window_ns"))
        weihs_rec_method = nw.get("method")

    # --- Load Delft (fixed bell products)

    delft15_off_path = bell_dir / delft15_dataset_id / "offset_sweep_metrics.json"
    delft15_cov_path = bell_dir / delft15_dataset_id / "covariance.json"
    # 条件分岐: `not delft15_off_path.exists()` を満たす経路を評価する。
    if not delft15_off_path.exists():
        raise SystemExit(f"[fail] missing Delft 2015 offset sweep: {delft15_off_path}")

    delft15 = _read_json(delft15_off_path)
    delft15_base = delft15.get("baseline", {}) if isinstance(delft15.get("baseline"), dict) else {}
    delft15_s_base = _safe_float(delft15_base.get("S"))
    delft15_se_base = _safe_float(delft15_base.get("S_err"))
    delft15_rows = delft15.get("rows") if isinstance(delft15.get("rows"), list) else []
    delft15_offsets_ps = [float(r.get("start_offset_ps")) for r in delft15_rows if _safe_float(r.get("start_offset_ps")) is not None]
    delft15_s_sweep = [float(r.get("S")) for r in delft15_rows if _safe_float(r.get("S")) is not None]
    delft15_s_min, delft15_s_max = _min_max(delft15_s_sweep)
    # 条件分岐: `(not delft15_offsets_ps) or (len(delft15_offsets_ps) != len(delft15_s_sweep))` を満たす経路を評価する。
    if (not delft15_offsets_ps) or (len(delft15_offsets_ps) != len(delft15_s_sweep)):
        raise SystemExit("[fail] Delft 2015 offset_sweep_metrics.json missing rows.start_offset_ps / rows.S")

    delft15_rec_off_ps = 0.0
    # 条件分岐: `delft15_cov_path.exists()` を満たす経路を評価する。
    if delft15_cov_path.exists():
        cov = _read_json(delft15_cov_path)
        nw = cov.get("natural_window") if isinstance(cov.get("natural_window"), dict) else {}
        v = _safe_float(nw.get("recommended_start_offset_ps"))
        # 条件分岐: `v is not None` を満たす経路を評価する。
        if v is not None:
            delft15_rec_off_ps = v

    delft16_off_path = bell_dir / delft16_dataset_id / "offset_sweep_metrics.json"
    delft16_cov_path = bell_dir / delft16_dataset_id / "covariance.json"
    delft16_offsets_ps: List[float] = []
    delft16_s_sweep: List[float] = []
    delft16_s_base: Optional[float] = None
    delft16_se_base: Optional[float] = None
    delft16_s_min: Optional[float] = None
    delft16_s_max: Optional[float] = None
    delft16_rec_off_ps = 0.0
    # 条件分岐: `delft16_off_path.exists()` を満たす経路を評価する。
    if delft16_off_path.exists():
        delft16 = _read_json(delft16_off_path)
        d16_base = delft16.get("baseline", {}) if isinstance(delft16.get("baseline"), dict) else {}
        delft16_s_base = _safe_float(d16_base.get("S_combined"))
        delft16_se_base = _safe_float(d16_base.get("S_combined_err"))
        d16_rows = delft16.get("rows") if isinstance(delft16.get("rows"), list) else []
        delft16_offsets_ps = [float(r.get("start_offset_ps")) for r in d16_rows if _safe_float(r.get("start_offset_ps")) is not None]
        delft16_s_sweep = [float(r.get("S_combined")) for r in d16_rows if _safe_float(r.get("S_combined")) is not None]
        delft16_s_min, delft16_s_max = _min_max(delft16_s_sweep)
        # 条件分岐: `(not delft16_offsets_ps) or (len(delft16_offsets_ps) != len(delft16_s_sweep))` を満たす経路を評価する。
        if (not delft16_offsets_ps) or (len(delft16_offsets_ps) != len(delft16_s_sweep)):
            raise SystemExit("[fail] Delft 2016 offset_sweep_metrics.json missing rows.start_offset_ps / rows.S_combined")

        # 条件分岐: `delft16_cov_path.exists()` を満たす経路を評価する。

        if delft16_cov_path.exists():
            cov = _read_json(delft16_cov_path)
            nw = cov.get("natural_window") if isinstance(cov.get("natural_window"), dict) else {}
            v = _safe_float(nw.get("recommended_start_offset_ps"))
            # 条件分岐: `v is not None` を満たす経路を評価する。
            if v is not None:
                delft16_rec_off_ps = v

    # --- Plot

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=150)

    # Weihs
    ax = axes[0]
    ax.plot(weihs_x, weihs_y, marker="o", lw=1.8, label="raw")
    # 条件分岐: `any(math.isfinite(v) for v in weihs_y_sub)` を満たす経路を評価する。
    if any(math.isfinite(v) for v in weihs_y_sub):
        ax.plot(weihs_x, weihs_y_sub, marker="s", lw=1.6, color="tab:orange", label="accidental-subtracted")

    ax.axhline(2.0, color="0.25", ls="--", lw=1.0, label="local bound |S|=2")
    # 条件分岐: `weihs_rec_w is not None and weihs_rec_w > 0` を満たす経路を評価する。
    if weihs_rec_w is not None and weihs_rec_w > 0:
        ax.axvline(weihs_rec_w, color="tab:orange", ls=":", lw=1.6, label=f"natural window ≈ {weihs_rec_w:.3g} ns")

    ax.set_xscale("log")
    ax.set_title(f"Weihs 1998: |S| vs window ({weihs_dataset_id})")
    ax.set_xlabel("window half-width (ns)")
    ax.set_ylabel("|S| (fixed variant)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True)

    # NIST
    ax = axes[1]
    ax.plot(nist_windows_ns, nist_j_sweep, marker="o", lw=1.8, label="coincidence-based J_prob")
    # 条件分岐: `any(math.isfinite(v) for v in nist_j_sweep_sub)` を満たす経路を評価する。
    if any(math.isfinite(v) for v in nist_j_sweep_sub):
        ax.plot(nist_windows_ns, nist_j_sweep_sub, marker="s", lw=1.6, color="tab:green", label="accidental-subtracted (clipped)")

    # 条件分岐: `nist_trial_best is not None` を満たす経路を評価する。

    if nist_trial_best is not None:
        ax.axhline(nist_trial_best, color="tab:orange", ls=":", lw=1.6, label="trial-based J_prob (best)")

    # 条件分岐: `nist_rec_w is not None and nist_rec_w > 0` を満たす経路を評価する。

    if nist_rec_w is not None and nist_rec_w > 0:
        ax.axvline(nist_rec_w, color="tab:orange", ls=":", lw=1.2, alpha=0.7)

    # 条件分岐: `kwiat_windows_ns and kwiat_j_sweep and (len(kwiat_windows_ns) == len(kwiat_j_...` を満たす経路を評価する。

    if kwiat_windows_ns and kwiat_j_sweep and (len(kwiat_windows_ns) == len(kwiat_j_sweep)):
        ax.plot(kwiat_windows_ns, kwiat_j_sweep, marker="s", lw=1.8, label=f"Kwiat2013 sweep ({kwiat_dataset_id})")
        # 条件分岐: `kwiat_trial_best is not None` を満たす経路を評価する。
        if kwiat_trial_best is not None:
            ax.axhline(kwiat_trial_best, color="tab:green", ls=":", lw=1.4, alpha=0.9, label="Kwiat2013 baseline")

        # 条件分岐: `kwiat_rec_w is not None and kwiat_rec_w > 0` を満たす経路を評価する。

        if kwiat_rec_w is not None and kwiat_rec_w > 0:
            ax.axvline(kwiat_rec_w, color="tab:green", ls=":", lw=1.0, alpha=0.5)

    ax.axhline(0.0, color="0.25", ls="--", lw=1.0)
    title_bits = ["NIST: CH J_prob vs window"]
    # 条件分岐: `(ksa is not None) or (ksb is not None)` を満たす経路を評価する。
    if (ksa is not None) or (ksb is not None):
        title_bits.append(f"KS(A)={ksa:.3f}" if ksa is not None else "KS(A)=n/a")
        title_bits.append(f"KS(B)={ksb:.3f}" if ksb is not None else "KS(B)=n/a")

    ax.set_title(" | ".join(title_bits))
    ax.set_xscale("log")
    ax.set_xlabel("window half-width (ns)")
    ax.set_ylabel("CH J_prob (A1=0,B1=0)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True)

    # Delft
    ax = axes[2]
    off_ns_15 = [x / 1000.0 for x in delft15_offsets_ps]
    ax.plot(off_ns_15, delft15_s_sweep, marker="o", ms=3, lw=1.6, label="2015 sweep")
    # 条件分岐: `delft15_s_base is not None` を満たす経路を評価する。
    if delft15_s_base is not None:
        # 条件分岐: `delft15_se_base is not None` を満たす経路を評価する。
        if delft15_se_base is not None:
            ax.axhspan(
                delft15_s_base - delft15_se_base, delft15_s_base + delft15_se_base, color="tab:orange", alpha=0.10
            )

        ax.axhline(delft15_s_base, color="tab:orange", lw=1.2, label="2015 baseline")

    # 条件分岐: `delft16_offsets_ps` を満たす経路を評価する。

    if delft16_offsets_ps:
        off_ns_16 = [x / 1000.0 for x in delft16_offsets_ps]
        ax.plot(off_ns_16, delft16_s_sweep, marker="s", ms=3, lw=1.6, label="2016 sweep")
        # 条件分岐: `delft16_s_base is not None` を満たす経路を評価する。
        if delft16_s_base is not None:
            # 条件分岐: `delft16_se_base is not None` を満たす経路を評価する。
            if delft16_se_base is not None:
                ax.axhspan(
                    delft16_s_base - delft16_se_base, delft16_s_base + delft16_se_base, color="tab:green", alpha=0.08
                )

            ax.axhline(delft16_s_base, color="tab:green", lw=1.2, label="2016 baseline (combined)")

    ax.axvline(float(delft15_rec_off_ps) / 1000.0, color="0.25", ls=":", lw=1.0)
    ax.axhline(2.0, color="0.25", ls="--", lw=1.0, label="local bound S=2")
    ax.set_title("Delft (event-ready): CHSH S vs start offset")
    ax.set_xlabel("start offset (ns)")
    ax.set_ylabel("CHSH S")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True)

    fig.tight_layout()

    out_tag = str(args.out_tag)
    out_png = out_dir / f"{out_tag}.png"
    out_json = out_dir / f"{out_tag}.json"
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": _utc_now(),
        "step": "Phase 7 / Step 7.4 (selection sensitivity summary)",
        "nist": {
            "trial_tag": str(args.nist_trial_tag),
            "dataset_id": nist_dataset_id,
            "source_manifest": nist_src_manifest,
            "trial_metrics": str(nist_ws_path),
            "time_tag_bias_metrics": None,
            "ks_setting0_vs_1": {"alice": ksa, "bob": ksb},
            "j_prob_trial_best": nist_trial_best,
            "j_prob_sweep_min": nist_j_min,
            "j_prob_sweep_max": nist_j_max,
            "j_prob_sweep_subtracted_clipped_min": nist_j_sub_min,
            "j_prob_sweep_subtracted_clipped_max": nist_j_sub_max,
            "recommended_window_ns": nist_rec_w,
            "recommended_window_method": nist_rec_method,
        },
        "kwiat2013": {
            "dataset_id": kwiat_dataset_id if kwiat_ws_path.exists() else None,
            "source_manifest": kwiat_src_manifest,
            "window_metrics": str(kwiat_ws_path) if kwiat_ws_path.exists() else None,
            "j_prob_trial_best": kwiat_trial_best,
            "j_prob_sweep_min": kwiat_j_min,
            "j_prob_sweep_max": kwiat_j_max,
            "recommended_window_ns": kwiat_rec_w,
            "recommended_window_method": kwiat_rec_method,
        },
        "weihs": {
            "out_tag": str(args.weihs_out_tag),
            "dataset_id": weihs_dataset_id,
            "source_manifest": weihs_src_manifest,
            "csv": str(weihs_ws_path),
            "abs_S_min": weihs_s_min,
            "abs_S_max": weihs_s_max,
            "abs_S_accidental_subtracted_min": weihs_s_sub_min,
            "abs_S_accidental_subtracted_max": weihs_s_sub_max,
            "recommended_window_ns": weihs_rec_w,
            "recommended_window_method": weihs_rec_method,
        },
        "delft2015": {
            "source_manifest": delft15_src_manifest,
            "metrics": str(delft15_off_path),
            "baseline_S": delft15_s_base,
            "baseline_S_err": delft15_se_base,
            "sweep_S_min": delft15_s_min,
            "sweep_S_max": delft15_s_max,
            "recommended_start_offset_ps": delft15_rec_off_ps,
        },
        "delft2016": {
            "source_manifest": delft16_src_manifest,
            "metrics": str(delft16_off_path) if delft16_off_path.exists() else None,
            "baseline_combined_S": delft16_s_base,
            "baseline_combined_S_err": delft16_se_base,
            "sweep_combined_S_min": delft16_s_min,
            "sweep_combined_S_max": delft16_s_max,
            "recommended_start_offset_ps": delft16_rec_off_ps,
        },
        "giustina2015": giustina_block_info,
        "outputs": {"png": str(out_png), "json": str(out_json)},
        "repro": {
            "script": "python -B scripts/quantum/bell_selection_sensitivity_summary.py",
            "inputs": [
                str(nist_ws_path),
                str(nist_cov_path) if nist_cov_path.exists() else None,
                str(kwiat_ws_path) if kwiat_ws_path.exists() else None,
                str(kwiat_cov_path) if kwiat_cov_path.exists() else None,
                str(weihs_ws_path),
                str(weihs_cov_path) if weihs_cov_path.exists() else None,
                str(delft15_off_path),
                str(delft15_cov_path) if delft15_cov_path.exists() else None,
                str(delft16_off_path) if delft16_off_path.exists() else None,
                str(delft16_cov_path) if delft16_cov_path.exists() else None,
            ],
        },
        "notes": [
            "This summary compares how analysis/selection knobs move reported test statistics across datasets.",
            "NIST uses CH J_prob; Weihs/Delft use CHSH S. They are plotted separately for sensitivity inspection.",
        ],
    }

    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
