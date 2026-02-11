from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write("")
            return
        headers = list(rows[0].keys())
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(h) for h in headers])


def _fmt_range(r: Sequence[float]) -> str:
    if len(r) != 2:
        return ""
    return f"{float(r[0]):g}-{float(r[1]):g}K"


def _safe_float(v: object) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _gate_ok(
    max_abs_z: Optional[float],
    reduced_chi2: Optional[float],
    *,
    max_abs_z_le: float,
    reduced_chi2_le: Optional[float],
) -> Optional[bool]:
    if max_abs_z is None:
        return None
    if max_abs_z > max_abs_z_le:
        return False
    if reduced_chi2_le is None or reduced_chi2 is None:
        return True
    return bool(reduced_chi2 <= reduced_chi2_le)


def _all_true(values: Sequence[Optional[bool]]) -> Optional[bool]:
    if any(v is False for v in values):
        return False
    if values and all(v is True for v in values):
        return True
    return None


def _minimax_gate_summary(
    split_summaries: Sequence[Dict[str, Any]],
    *,
    max_abs_z_le: float,
    reduced_chi2_le: Optional[float],
) -> Dict[str, Any]:
    splits_ordered = [str(sp.get("split") or "") for sp in split_summaries if sp.get("split") is not None]
    splits_ordered = [s for s in splits_ordered if s]
    n_splits = len(splits_ordered)

    model_to_split_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sp in split_summaries:
        sp_name = str(sp.get("split") or "")
        if not sp_name:
            continue
        for row in sp.get("models", []) if isinstance(sp.get("models"), list) else []:
            if not isinstance(row, dict):
                continue
            if row.get("supported") is not True:
                continue
            mid = str(row.get("model_id") or "")
            if not mid:
                continue
            model_to_split_rows.setdefault(mid, {})[sp_name] = row

    candidates: List[Tuple[float, float, float, str]] = []
    for mid, by_split in model_to_split_rows.items():
        if len(by_split) != n_splits:
            continue
        worst_test_max = max(
            float(v) if v is not None else float("inf") for v in (_safe_float(by_split[s].get("test_max_abs_z")) for s in splits_ordered)
        )
        worst_test_chi2 = max(
            float(v) if v is not None else float("inf")
            for v in (_safe_float(by_split[s].get("test_reduced_chi2")) for s in splits_ordered)
        )
        worst_train_max = max(
            float(v) if v is not None else float("inf") for v in (_safe_float(by_split[s].get("train_max_abs_z")) for s in splits_ordered)
        )
        candidates.append((worst_test_max, worst_test_chi2, worst_train_max, mid))

    recommended_model_id = min(candidates)[-1] if candidates else None
    by_split_rows = model_to_split_rows.get(str(recommended_model_id), {}) if recommended_model_id else {}

    train_ok_by_split: List[Optional[bool]] = []
    test_ok_by_split: List[Optional[bool]] = []
    per_split: List[Dict[str, Any]] = []
    for sp_name in splits_ordered:
        row = by_split_rows.get(sp_name, {})
        tr_max = _safe_float(row.get("train_max_abs_z"))
        te_max = _safe_float(row.get("test_max_abs_z"))
        tr_chi2 = _safe_float(row.get("train_reduced_chi2"))
        te_chi2 = _safe_float(row.get("test_reduced_chi2"))
        tr_ok = _gate_ok(tr_max, tr_chi2, max_abs_z_le=max_abs_z_le, reduced_chi2_le=reduced_chi2_le)
        te_ok = _gate_ok(te_max, te_chi2, max_abs_z_le=max_abs_z_le, reduced_chi2_le=reduced_chi2_le)
        train_ok_by_split.append(tr_ok)
        test_ok_by_split.append(te_ok)
        per_split.append(
            {
                "split": sp_name,
                "train_gate_ok": tr_ok,
                "test_gate_ok": te_ok,
                "train_max_abs_z": tr_max,
                "train_reduced_chi2": tr_chi2,
                "test_max_abs_z": te_max,
                "test_reduced_chi2": te_chi2,
            }
        )

    summary: Dict[str, Any] = {
        "criteria": {"max_abs_z_le": float(max_abs_z_le), "reduced_chi2_le": float(reduced_chi2_le) if reduced_chi2_le is not None else None},
        "recommended_model_by_minimax_test_max_abs_z": {"model_id": recommended_model_id},
        "strict_ok": _all_true(train_ok_by_split),
        "holdout_ok": _all_true(test_ok_by_split),
        "by_split": per_split,
    }

    if recommended_model_id:
        worst_test_max = max((float(v) if v is not None else float("inf")) for v in (r.get("test_max_abs_z") for r in per_split))
        summary["recommended_model_by_minimax_test_max_abs_z"]["worst_test_max_abs_z"] = (
            float(worst_test_max) if math.isfinite(float(worst_test_max)) else None
        )
    return summary


def main() -> int:
    out_dir = _ROOT / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [
        {
            "dataset": "Si α(T)",
            "metrics_json": out_dir / "condensed_silicon_thermal_expansion_gruneisen_holdout_splits_metrics.json",
            "repro": "python -B scripts/quantum/condensed_silicon_thermal_expansion_gruneisen_holdout_splits.py",
        },
        {
            "dataset": "Si α(T) (DOS+γ(ω) basis)",
            "metrics_json": out_dir
            / "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_gamma_omega_pwlinear_split_leaky_mode_softening_kim2015_fig2_bulkmodulus_ridge1e-06_leak2p40e-01_warp1p32_model_metrics.json",
            "repro": (
                "python -B scripts/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_model.py "
                "--groups 4 --enforce-signs --use-bulk-modulus --mode-softening kim2015_fig2_features "
                "--gamma-omega-model pwlinear_split_leaky --gamma-omega-pwlinear-leak 0.24 --gamma-omega-pwlinear-warp-power 1.32 "
                "--ridge-factor 1e-06"
            ),
        },
        {
            "dataset": "Si Cp(T)",
            "metrics_json": out_dir / "condensed_silicon_heat_capacity_holdout_splits_metrics.json",
            "repro": "python -B scripts/quantum/condensed_silicon_heat_capacity_holdout_splits.py",
        },
        {
            "dataset": "Si B(T)",
            "metrics_json": out_dir / "condensed_silicon_bulk_modulus_holdout_splits_metrics.json",
            "repro": "python -B scripts/quantum/condensed_silicon_bulk_modulus_holdout_splits.py",
        },
        {
            "dataset": "Cu κ(T)",
            "metrics_json": out_dir / "condensed_ofhc_copper_thermal_conductivity_holdout_splits_metrics.json",
            "repro": "python -B scripts/quantum/condensed_ofhc_copper_thermal_conductivity_holdout_splits.py",
        },
    ]

    rows_csv: List[Dict[str, Any]] = []
    dataset_summaries: List[Dict[str, Any]] = []
    missing_inputs: List[Dict[str, Any]] = []

    # Operational threshold (align with other z-based audits).
    z_thr = 3.0
    reduced_chi2_thr = 9.0

    for inp in inputs:
        ds_name = str(inp.get("dataset") or "")
        in_metrics = Path(str(inp.get("metrics_json")))
        if not in_metrics.exists():
            missing_inputs.append({"dataset": ds_name, "expected": _rel(in_metrics), "repro": str(inp.get("repro") or "")})
            continue

        payload = json.loads(in_metrics.read_text(encoding="utf-8"))
        splits = payload.get("splits")
        holdout_splits = payload.get("holdout_splits")

        if (not isinstance(splits, list) or not splits) and (not isinstance(holdout_splits, list) or not holdout_splits):
            missing_inputs.append(
                {
                    "dataset": ds_name,
                    "expected": _rel(in_metrics),
                    "reason": "splits/holdout_splits missing/empty",
                }
            )
            continue

        split_summaries: List[Dict[str, Any]] = []
        dataset_summary_extra: Dict[str, Any] = {}
        if isinstance(splits, list) and splits:
            for sp in splits:
                if not isinstance(sp, dict):
                    continue
                sp_name = str(sp.get("name", ""))
                train_r = sp.get("train_T_K") if isinstance(sp.get("train_T_K"), list) else []
                test_r = sp.get("test_T_K") if isinstance(sp.get("test_T_K"), list) else []
                models = sp.get("models") if isinstance(sp.get("models"), dict) else {}

                best_model = None
                best_test_max_abs_z = None

                model_rows: List[Dict[str, Any]] = []
                for mid, m in models.items():
                    if not isinstance(m, dict):
                        continue
                    supported = (m.get("supported") is not False)
                    train = m.get("train") if isinstance(m.get("train"), dict) else {}
                    test = m.get("test") if isinstance(m.get("test"), dict) else {}

                    train_max = _safe_float(train.get("max_abs_z"))
                    test_max = _safe_float(test.get("max_abs_z"))
                    train_chi2 = _safe_float(train.get("reduced_chi2"))
                    test_chi2 = _safe_float(test.get("reduced_chi2"))
                    train_n = int(train.get("n", 0) or 0)
                    test_n = int(test.get("n", 0) or 0)
                    train_ex = int(train.get("exceed_3sigma_n", 0) or 0)
                    test_ex = int(test.get("exceed_3sigma_n", 0) or 0)

                    row = {
                        "dataset": ds_name,
                        "split": sp_name,
                        "train_range": _fmt_range(train_r),
                        "test_range": _fmt_range(test_r),
                        "model_id": str(mid),
                        "supported": bool(supported),
                        "reason": str(m.get("reason") or "") if not supported else "",
                        "train_n": train_n,
                        "train_max_abs_z": train_max,
                        "train_rms_z": _safe_float(train.get("rms_z")),
                        "train_reduced_chi2": train_chi2,
                        "train_exceed_abs_z_gt3_n": train_ex,
                        "train_exceed_abs_z_gt3_frac": (float(train_ex) / float(train_n)) if train_n > 0 else None,
                        "train_pass_all_abs_z_le_3": (
                            bool(train_max is not None and train_max <= z_thr) if train_max is not None else None
                        ),
                        "train_pass_reduced_chi2_le_thr": (
                            bool(train_chi2 is not None and train_chi2 <= reduced_chi2_thr) if train_chi2 is not None else None
                        ),
                        "train_gate_ok": _gate_ok(
                            train_max, train_chi2, max_abs_z_le=z_thr, reduced_chi2_le=reduced_chi2_thr
                        ),
                        "test_n": test_n,
                        "test_max_abs_z": test_max,
                        "test_rms_z": _safe_float(test.get("rms_z")),
                        "test_reduced_chi2": test_chi2,
                        "test_exceed_abs_z_gt3_n": test_ex,
                        "test_exceed_abs_z_gt3_frac": (float(test_ex) / float(test_n)) if test_n > 0 else None,
                        "test_pass_all_abs_z_le_3": (
                            bool(test_max is not None and test_max <= z_thr) if test_max is not None else None
                        ),
                        "test_pass_reduced_chi2_le_thr": (
                            bool(test_chi2 is not None and test_chi2 <= reduced_chi2_thr) if test_chi2 is not None else None
                        ),
                        "test_gate_ok": _gate_ok(test_max, test_chi2, max_abs_z_le=z_thr, reduced_chi2_le=reduced_chi2_thr),
                    }
                    rows_csv.append(row)
                    model_rows.append(row)

                    if supported and test_max is not None:
                        if best_test_max_abs_z is None or float(test_max) < float(best_test_max_abs_z):
                            best_test_max_abs_z = float(test_max)
                            best_model = str(mid)

                split_summaries.append(
                    {
                        "split": sp_name,
                        "train_range": _fmt_range(train_r),
                        "test_range": _fmt_range(test_r),
                        "best_model_by_test_max_abs_z": best_model,
                        "best_test_max_abs_z": best_test_max_abs_z,
                        "models": model_rows,
                    }
                )

            dataset_summary_extra["audit_gates"] = _minimax_gate_summary(
                split_summaries, max_abs_z_le=z_thr, reduced_chi2_le=reduced_chi2_thr
            )
        else:
            # Alternative format: a single-model metrics JSON containing holdout_splits + falsification gates.
            if not isinstance(holdout_splits, list) or not holdout_splits:
                missing_inputs.append({"dataset": ds_name, "expected": _rel(in_metrics), "reason": "holdout_splits missing/empty"})
                continue

            model_id = "model"
            try:
                params0 = holdout_splits[0].get("params") if isinstance(holdout_splits[0], dict) else {}
                if isinstance(params0, dict) and isinstance(params0.get("gamma_omega_model"), str) and params0.get("gamma_omega_model"):
                    model_id = f"gamma_omega:{params0.get('gamma_omega_model')}"
            except Exception:
                model_id = "model"

            fals = payload.get("falsification") if isinstance(payload.get("falsification"), dict) else {}
            strict_ok = fals.get("strict_ok")
            holdout_ok = fals.get("holdout_ok")
            strict_criteria = fals.get("strict_criteria") if isinstance(fals.get("strict_criteria"), dict) else {}
            z_thr0 = _safe_float(strict_criteria.get("max_abs_z_le")) or z_thr
            reduced_chi2_thr0 = _safe_float(strict_criteria.get("reduced_chi2_le")) or reduced_chi2_thr
            dataset_summary_extra["falsification"] = {
                "strict_ok": (bool(strict_ok) if isinstance(strict_ok, bool) else None),
                "holdout_ok": (bool(holdout_ok) if isinstance(holdout_ok, bool) else None),
                "strict_criteria": strict_criteria or None,
            }
            if isinstance(payload.get("model"), dict) and payload.get("model", {}).get("name"):
                dataset_summary_extra["model"] = {"name": str(payload.get("model", {}).get("name"))}

            for sp in holdout_splits:
                if not isinstance(sp, dict):
                    continue
                sp_name = str(sp.get("name", ""))
                train_r = sp.get("train_T_K") if isinstance(sp.get("train_T_K"), list) else []
                test_r = sp.get("test_T_K") if isinstance(sp.get("test_T_K"), list) else []
                train = sp.get("train") if isinstance(sp.get("train"), dict) else {}
                test = sp.get("test") if isinstance(sp.get("test"), dict) else {}

                train_max = _safe_float(train.get("max_abs_z"))
                test_max = _safe_float(test.get("max_abs_z"))
                train_chi2 = _safe_float(train.get("reduced_chi2"))
                test_chi2 = _safe_float(test.get("reduced_chi2"))
                train_n = int(train.get("n", 0) or 0)
                test_n = int(test.get("n", 0) or 0)
                train_ex = int(train.get("exceed_3sigma_n", 0) or 0)
                test_ex = int(test.get("exceed_3sigma_n", 0) or 0)

                row = {
                    "dataset": ds_name,
                    "split": sp_name,
                    "train_range": _fmt_range(train_r),
                    "test_range": _fmt_range(test_r),
                    "model_id": str(model_id),
                    "supported": True,
                    "reason": "",
                    "train_n": train_n,
                    "train_max_abs_z": train_max,
                    "train_rms_z": _safe_float(train.get("rms_z")),
                    "train_reduced_chi2": train_chi2,
                    "train_exceed_abs_z_gt3_n": train_ex,
                    "train_exceed_abs_z_gt3_frac": (float(train_ex) / float(train_n)) if train_n > 0 else None,
                    "train_pass_all_abs_z_le_3": (
                        bool(train_max is not None and train_max <= z_thr0) if train_max is not None else None
                    ),
                    "train_pass_reduced_chi2_le_thr": (
                        bool(train_chi2 is not None and train_chi2 <= reduced_chi2_thr0) if train_chi2 is not None else None
                    ),
                    "train_gate_ok": _gate_ok(
                        train_max, train_chi2, max_abs_z_le=z_thr0, reduced_chi2_le=reduced_chi2_thr0
                    ),
                    "test_n": test_n,
                    "test_max_abs_z": test_max,
                    "test_rms_z": _safe_float(test.get("rms_z")),
                    "test_reduced_chi2": test_chi2,
                    "test_exceed_abs_z_gt3_n": test_ex,
                    "test_exceed_abs_z_gt3_frac": (float(test_ex) / float(test_n)) if test_n > 0 else None,
                    "test_pass_all_abs_z_le_3": (bool(test_max is not None and test_max <= z_thr0) if test_max is not None else None),
                    "test_pass_reduced_chi2_le_thr": (
                        bool(test_chi2 is not None and test_chi2 <= reduced_chi2_thr0) if test_chi2 is not None else None
                    ),
                    "test_gate_ok": _gate_ok(test_max, test_chi2, max_abs_z_le=z_thr0, reduced_chi2_le=reduced_chi2_thr0),
                }
                rows_csv.append(row)

                split_summaries.append(
                    {
                        "split": sp_name,
                        "train_range": _fmt_range(train_r),
                        "test_range": _fmt_range(test_r),
                        "best_model_by_test_max_abs_z": str(model_id),
                        "best_test_max_abs_z": test_max,
                        "models": [row],
                    }
                )

            audit_gates = _minimax_gate_summary(
                split_summaries, max_abs_z_le=z_thr0, reduced_chi2_le=reduced_chi2_thr0
            )
            if isinstance(dataset_summary_extra.get("falsification"), dict):
                fals0 = dataset_summary_extra["falsification"]
                if isinstance(fals0.get("strict_ok"), bool):
                    audit_gates["strict_ok"] = bool(fals0["strict_ok"])
                    audit_gates["strict_ok_source"] = "metrics_falsification"
                if isinstance(fals0.get("holdout_ok"), bool):
                    audit_gates["holdout_ok"] = bool(fals0["holdout_ok"])
                    audit_gates["holdout_ok_source"] = "metrics_falsification"
            dataset_summary_extra["audit_gates"] = audit_gates

        dataset_summary: Dict[str, Any] = {
            "dataset": ds_name,
            "inputs": {"holdout_metrics_json": {"path": _rel(in_metrics), "sha256": _sha256(in_metrics)}},
            "n_splits": int(len(split_summaries)),
            "splits": split_summaries,
        }
        dataset_summary.update(dataset_summary_extra)
        dataset_summaries.append(dataset_summary)

    out_csv = out_dir / "condensed_holdout_audit.csv"
    out_json = out_dir / "condensed_holdout_audit_summary.json"
    out_png = out_dir / "condensed_holdout_audit.png"

    _write_csv(out_csv, rows_csv)

    # Plot (optional): one panel per dataset.
    try:
        import matplotlib.pyplot as plt

        if dataset_summaries:
            fig_h = 3.2 * max(1, len(dataset_summaries))
            fig, axes = plt.subplots(len(dataset_summaries), 1, figsize=(12.8, fig_h), dpi=160, squeeze=False)
            for ax, ds in zip(axes[:, 0], dataset_summaries):
                ds_name = str(ds.get("dataset") or "")
                ds_rows = [r for r in rows_csv if r.get("dataset") == ds_name]
                splits_unique = sorted({str(r.get("split", "")) for r in ds_rows if r.get("split")})
                models_unique = sorted({str(r.get("model_id", "")) for r in ds_rows if r.get("model_id")})
                if not splits_unique or not models_unique or len(models_unique) > 8 or len(splits_unique) > 12:
                    ax.axis("off")
                    continue

                width = 0.8 / max(1, len(models_unique))
                x0 = list(range(len(splits_unique)))
                for j, mid in enumerate(models_unique):
                    ys = []
                    for sp in splits_unique:
                        vals = [r for r in ds_rows if r.get("split") == sp and r.get("model_id") == mid]
                        v = vals[0].get("test_max_abs_z") if vals else None
                        ys.append(float(v) if isinstance(v, (int, float)) else float("nan"))
                    xs = [x + (j - (len(models_unique) - 1) / 2) * width for x in x0]
                    ax.bar(xs, ys, width=width, label=mid, alpha=0.85)

                ax.axhline(z_thr, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
                ax.set_xticks(x0)
                ax.set_xticklabels(splits_unique)
                ax.set_ylabel("test max abs(z)")
                ax.set_title(f"Holdout severity: {ds_name}")
                ax.grid(axis="y", linestyle=":", alpha=0.35)
                ax.legend(fontsize=8, loc="upper left", ncol=2)

            fig.tight_layout()
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
    except Exception:
        pass

    out_payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": 7,
        "step": "7.20.6",
        "inputs": {
            "holdout_metrics": [
                {
                    "dataset": str(inp.get("dataset") or ""),
                    "path": _rel(Path(str(inp.get("metrics_json")))),
                    "exists": Path(str(inp.get("metrics_json"))).exists(),
                    "sha256": (_sha256(Path(str(inp.get("metrics_json")))) if Path(str(inp.get("metrics_json"))).exists() else None),
                    "repro": str(inp.get("repro") or ""),
                }
                for inp in inputs
            ]
        },
        "thresholds": {
            # Backward-compatible key (older scripts referenced this name).
            "z_outlier_abs_gt": z_thr,
            "max_abs_z_le": z_thr,
            "reduced_chi2_le": reduced_chi2_thr,
            "note": "Operational gates used to label holdout severity; not universal physics thresholds.",
        },
        "summary": {
            "datasets_n": int(len(dataset_summaries)),
            "datasets": dataset_summaries,
            "missing_inputs": missing_inputs,
            "note": "This audit catalogs temperature-band sensitivity (procedure/systematic) rather than claiming a universally valid condensed-matter prediction model.",
        },
        "outputs": {"csv": _rel(out_csv), "json": _rel(out_json), "png": _rel(out_png) if out_png.exists() else None},
    }
    out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "domain": "quantum",
            "action": "condensed_holdout_audit",
            "outputs": [out_json, out_csv, out_png if out_png.exists() else None],
            "params": {"max_abs_z_le": z_thr, "reduced_chi2_le": reduced_chi2_thr},
            "result": {"datasets_n": int(len(dataset_summaries)), "rows_n": int(len(rows_csv)), "missing_inputs_n": int(len(missing_inputs))},
        }
    )

    print("[ok] wrote:")
    print(f"- {out_json}")
    print(f"- {out_csv}")
    if out_png.exists():
        print(f"- {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
