#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_final_branch_decision.py

Phase 4 / Step 4.14.6:
4.14.1〜4.14.5（p_t / 条件A / 独立ガードレール / BAO ε入口 / DDR再接続）を合流し、
「次フェーズへ進む条件／棄却または写像改訂の条件」を機械可読な形で固定する。

出力（固定）:
- output/private/cosmology/cosmology_final_branch_decision_metrics.json
- output/private/cosmology/cosmology_final_branch_decision.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


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
        return


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _z_score(x: float, mu: float, sigma: float) -> Optional[float]:
    # 条件分岐: `not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in (x,...` を満たす経路を評価する。
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in (x, mu, sigma)):
        return None

    # 条件分岐: `float(sigma) <= 0` を満たす経路を評価する。

    if float(sigma) <= 0:
        return None

    return (float(x) - float(mu)) / float(sigma)


def _classify_abs_sigma(abs_sigma: Optional[float]) -> Optional[str]:
    # 条件分岐: `abs_sigma is None` を満たす経路を評価する。
    if abs_sigma is None:
        return None

    a = float(abs_sigma)
    # 条件分岐: `not math.isfinite(a)` を満たす経路を評価する。
    if not math.isfinite(a):
        return None

    # 条件分岐: `a < 1.0` を満たす経路を評価する。

    if a < 1.0:
        return "ok"

    # 条件分岐: `a < 3.0` を満たす経路を評価する。

    if a < 3.0:
        return "mixed"

    return "ng"


def _classify_abs_z(abs_z: Optional[float]) -> Optional[str]:
    # Use the project's standard |z| thresholds for "ok/mixed/ng".
    if abs_z is None:
        return None

    a = float(abs_z)
    # 条件分岐: `not math.isfinite(a)` を満たす経路を評価する。
    if not math.isfinite(a):
        return None

    # 条件分岐: `a < 3.0` を満たす経路を評価する。

    if a < 3.0:
        return "ok"

    # 条件分岐: `a < 5.0` を満たす経路を評価する。

    if a < 5.0:
        return "mixed"

    return "ng"


@dataclass(frozen=True)
class StudyMax:
    name: str
    abs_sigma: Optional[float]


def _extract_ledger_maxima(ledger: Dict[str, Any]) -> Tuple[List[StudyMax], Optional[float]]:
    studies = ledger.get("studies", [])
    maxima: List[StudyMax] = []
    global_max: Optional[float] = None

    # 条件分岐: `not isinstance(studies, list)` を満たす経路を評価する。
    if not isinstance(studies, list):
        return maxima, None

    for s in studies:
        # 条件分岐: `not isinstance(s, dict)` を満たす経路を評価する。
        if not isinstance(s, dict):
            continue

        # 条件分岐: `s.get("status") != "ok"` を満たす経路を評価する。

        if s.get("status") != "ok":
            continue

        md = s.get("max_delta")
        # 条件分岐: `not isinstance(md, dict)` を満たす経路を評価する。
        if not isinstance(md, dict):
            continue

        v = md.get("abs_sigma")
        # 条件分岐: `not isinstance(v, (int, float)) or not math.isfinite(float(v))` を満たす経路を評価する。
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            continue

        vv = float(v)
        maxima.append(StudyMax(name=str(s.get("name")), abs_sigma=vv))
        global_max = vv if global_max is None else max(global_max, vv)

    return maxima, global_max


def _find_ddr_fit(candidate_search: Dict[str, Any], *, ddr_id: str, fit_key: str) -> Optional[Dict[str, Any]]:
    per_ddr = candidate_search.get("results", {}).get("per_ddr", [])
    # 条件分岐: `not isinstance(per_ddr, list)` を満たす経路を評価する。
    if not isinstance(per_ddr, list):
        return None

    for r in per_ddr:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        # 条件分岐: `r.get("ddr", {}).get("id") != ddr_id` を満たす経路を評価する。

        if r.get("ddr", {}).get("id") != ddr_id:
            continue

        block = r.get(fit_key)
        # 条件分岐: `not isinstance(block, dict)` を満たす経路を評価する。
        if not isinstance(block, dict):
            return None

        fit = block.get("fit")
        # 条件分岐: `not isinstance(fit, dict)` を満たす経路を評価する。
        if not isinstance(fit, dict):
            return None

        return fit

    return None


def _plot_sigma_like(summary_rows: List[Tuple[str, float]], out_png: Path) -> Optional[str]:
    # 条件分岐: `not summary_rows` を満たす経路を評価する。
    if not summary_rows:
        return None

    _set_japanese_font()

    labels = [r[0] for r in summary_rows]
    vals = [r[1] for r in summary_rows]

    fig_h = max(3.5, 0.55 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = list(range(len(labels)))
    ax.barh(y, vals, color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("sigma-like indicator (smaller is better)")
    ax.set_title("Cosmology final branch decision (Phase 4 / Step 4.14.6)")
    for x in (1.0, 3.0, 5.0):
        ax.axvline(x, color="#888", ls="--", lw=1.0, alpha=0.8)

    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return str(out_png)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pt-fit",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_sn_time_dilation_pt_fit.json"),
        help="Path to cosmology_sn_time_dilation_pt_fit.json",
    )
    p.add_argument(
        "--condition-a-memo",
        default=str(_ROOT / "doc" / "cosmology" / "TIME_DILATION_CONDITION_A.md"),
        help="Path to TIME_DILATION_CONDITION_A.md (spec memo)",
    )
    p.add_argument(
        "--bao-ledger",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_epsilon_entry_sensitivity_ledger.json"),
        help="Path to cosmology_bao_epsilon_entry_sensitivity_ledger.json",
    )
    p.add_argument(
        "--ddr-required",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_ddr_reconnection_conditions_metrics.json"),
        help="Path to cosmology_ddr_reconnection_conditions_metrics.json",
    )
    p.add_argument(
        "--ddr-candidate-search",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_distance_indicator_rederivation_candidate_search_metrics.json"),
        help="Path to cosmology_distance_indicator_rederivation_candidate_search_metrics.json",
    )
    p.add_argument(
        "--out-json",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_final_branch_decision_metrics.json"),
        help="Output json path",
    )
    p.add_argument(
        "--out-png",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_final_branch_decision.png"),
        help="Output png path",
    )
    args = p.parse_args()

    pt_path = Path(args.pt_fit).resolve()
    ledger_path = Path(args.bao_ledger).resolve()
    ddr_required_path = Path(args.ddr_required).resolve()
    ddr_candidate_path = Path(args.ddr_candidate_search).resolve()
    cond_a_path = Path(args.condition_a_memo).resolve()

    out_json = Path(args.out_json).resolve()
    out_png = Path(args.out_png).resolve()

    pt = _read_json(pt_path) if pt_path.exists() else {}
    ledger = _read_json(ledger_path) if ledger_path.exists() else {}
    ddr_required = _read_json(ddr_required_path) if ddr_required_path.exists() else {}
    ddr_candidate = _read_json(ddr_candidate_path) if ddr_candidate_path.exists() else {}

    # ---- p_t (time dilation) ----
    fit_all = pt.get("fits", {}).get("all", {})
    p_t_fit = float(fit_all.get("p_t_fit")) if isinstance(fit_all.get("p_t_fit"), (int, float)) else None
    p_t_sigma = float(fit_all.get("p_t_sigma_sym")) if isinstance(fit_all.get("p_t_sigma_sym"), (int, float)) else None
    z_to_1 = _z_score(p_t_fit, 1.0, p_t_sigma) if (p_t_fit is not None and p_t_sigma is not None) else None
    z_to_0 = _z_score(p_t_fit, 0.0, p_t_sigma) if (p_t_fit is not None and p_t_sigma is not None) else None

    # ---- BAO ε entry ledger ----
    study_maxima, ledger_global_max = _extract_ledger_maxima(ledger)
    ledger_status = _classify_abs_sigma(ledger_global_max)

    # ---- DDR representative ----
    rep_ddr = ddr_required.get("representative_ddr", {})
    rep_ddr_id = rep_ddr.get("id") if isinstance(rep_ddr, dict) else None
    rep_required_combo = None
    rep_single_mech = None
    # 条件分岐: `isinstance(rep_ddr, dict)` を満たす経路を評価する。
    if isinstance(rep_ddr, dict):
        rep_required_combo = {
            "epsilon0_obs": rep_ddr.get("epsilon0_obs"),
            "epsilon0_sigma": rep_ddr.get("epsilon0_sigma"),
            "delta_epsilon_needed": rep_ddr.get("delta_epsilon_needed"),
        }

    rep_single_mech = ddr_required.get("representative_decision_summary")

    best_independent_fit = (
        _find_ddr_fit(ddr_candidate, ddr_id=str(rep_ddr_id), fit_key="best_independent") if rep_ddr_id else None
    )
    best_any_fit = _find_ddr_fit(ddr_candidate, ddr_id=str(rep_ddr_id), fit_key="best_any") if rep_ddr_id else None

    ddr_best_independent_max_abs_z = None
    # 条件分岐: `isinstance(best_independent_fit, dict) and isinstance(best_independent_fit.ge...` を満たす経路を評価する。
    if isinstance(best_independent_fit, dict) and isinstance(best_independent_fit.get("max_abs_z"), (int, float)):
        ddr_best_independent_max_abs_z = float(best_independent_fit["max_abs_z"])

    ddr_best_independent_status = _classify_abs_z(ddr_best_independent_max_abs_z)

    # ---- branch decision (provisional) ----
    reasons: List[str] = []
    # 条件分岐: `z_to_1 is not None` を満たす経路を評価する。
    if z_to_1 is not None:
        reasons.append(f"SN time dilation p_t is consistent with 1: |z|={abs(z_to_1):.3g}.")

    # 条件分岐: `z_to_0 is not None` を満たす経路を評価する。

    if z_to_0 is not None:
        reasons.append(f"SN time dilation rejects p_t=0 at |z|={abs(z_to_0):.3g}.")

    # 条件分岐: `ledger_global_max is not None` を満たす経路を評価する。

    if ledger_global_max is not None:
        reasons.append(f"BAO ε entry procedure sensitivity: max |Δε|/σ ≈ {ledger_global_max:.3g} ({ledger_status}).")

    # 条件分岐: `ddr_best_independent_max_abs_z is not None` を満たす経路を評価する。

    if ddr_best_independent_max_abs_z is not None:
        reasons.append(
            f"DDR representative (independent) reconnection exists in this parameterization: max|z|≈{ddr_best_independent_max_abs_z:.3g} ({ddr_best_independent_status})."
        )

    branch_status: str
    # Rule: proceed if DDR best_independent is ok (<3) and p_t is not in tension with 1.
    if ddr_best_independent_status == "ok" and (z_to_1 is None or abs(z_to_1) < 3.0):
        branch_status = "proceed_with_reconnection_path"
    else:
        branch_status = "not_adopted_yet"

    next_actions = [
        "If BAO ε is used as a decisive falsifier, freeze peakfit settings and treat procedure-induced shifts as a systematic budget (use the ε entry ledger).",
        "Revisit the cosmology branch after GN-z11 is public (JWST/MAST Step 4.6).",
    ]

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "Phase 4 / Step 4.14.6 (final branch decision)",
        "inputs": {
            "pt_fit": str(pt_path) if pt_path.exists() else None,
            "condition_a_memo": str(cond_a_path) if cond_a_path.exists() else None,
            "bao_epsilon_entry_ledger": str(ledger_path) if ledger_path.exists() else None,
            "ddr_reconnection_required": str(ddr_required_path) if ddr_required_path.exists() else None,
            "ddr_candidate_search": str(ddr_candidate_path) if ddr_candidate_path.exists() else None,
        },
        "summary": {
            "time_dilation_pt": {
                "p_t_fit": p_t_fit,
                "p_t_sigma_1sigma": p_t_sigma,
                "z_to_1": z_to_1,
                "z_to_0": z_to_0,
                "status": _classify_abs_z(abs(z_to_1)) if z_to_1 is not None else None,
            },
            "condition_A": {
                "adopted": bool(cond_a_path.exists()),
                "policy": "A (axiom/frozen spec)",
                "memo": str(cond_a_path) if cond_a_path.exists() else None,
            },
            "bao_epsilon_entry": {
                "global_max_abs_sigma": ledger_global_max,
                "status": ledger_status,
                "study_maxima": [{"name": s.name, "max_abs_sigma": s.abs_sigma} for s in study_maxima],
            },
            "ddr_reconnection": {
                "representative_ddr_id": rep_ddr_id,
                "representative_required_combo": rep_required_combo,
                "representative_single_mechanism_summary": rep_single_mech,
                "representative_best_any_fit": best_any_fit,
                "representative_best_independent_fit": best_independent_fit,
                "representative_best_independent_status": ddr_best_independent_status,
            },
        },
        "decision": {
            "branch_status": branch_status,
            "reasons": reasons,
            "note": "This is a provisional branch decision based on the current fixed I/F and error budgets; it is not a claim that any particular mechanism is correct.",
            "next_actions": next_actions,
        },
        "outputs": {"json": str(out_json), "png": str(out_png)},
    }

    _write_json(out_json, payload)

    # plot sigma-like summary indicators
    sigma_like: List[Tuple[str, float]] = []
    # 条件分岐: `z_to_1 is not None` を満たす経路を評価する。
    if z_to_1 is not None:
        sigma_like.append(("SN time dilation: |p_t-1|/σ", abs(float(z_to_1))))

    # 条件分岐: `ledger_global_max is not None` を満たす経路を評価する。

    if ledger_global_max is not None:
        sigma_like.append(("BAO ε entry: max |Δε|/σ", float(ledger_global_max)))

    # 条件分岐: `ddr_best_independent_max_abs_z is not None` を満たす経路を評価する。

    if ddr_best_independent_max_abs_z is not None:
        sigma_like.append(("DDR reconnection (rep): max|z|", float(ddr_best_independent_max_abs_z)))

    png_written = _plot_sigma_like(sigma_like, out_png)
    # 条件分岐: `png_written is None` を満たす経路を評価する。
    if png_written is None:
        payload["outputs"]["png"] = None
        _write_json(out_json, payload)

    worklog.append_event(
        {
            "event_type": "cosmology_final_branch_decision",
            "generated_utc": payload["generated_utc"],
            "inputs": payload["inputs"],
            "outputs": payload["outputs"],
            "decision": payload["decision"],
        }
    )


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
