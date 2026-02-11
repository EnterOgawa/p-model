#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_falsification.py

Phase 7 / (new) Step 7.5.7:
量子領域で「P-model と参照枠（標準QM+GR）が微小にズレる決着点」を
Part I 3.0 の Rejection Protocol 書式（Input/Frozen/Output/Statistic/Reject）で固定出力化する。

出力（固定）:
  - output/private/summary/quantum_falsification.json

注意:
- 現状は一次“生データ（raw fringe）”ではなく、一次論文（PDF）と代表値からのスケーリングを固定している項目を含む。
- βの位相差への寄与（原子干渉計）は、Mueller 2007 の「k_eff を光の分散から決める」導出に沿って、
  “どの位相項に β が入るか（定数項の相殺）” を固定した上で、差分スケール（必要精度）として出力する。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _try_load_frozen_parameters() -> Dict[str, Any]:
    p = _ROOT / "output" / "private" / "theory" / "frozen_parameters.json"
    if not p.exists():
        return {"path": _relpath(p), "exists": False}
    try:
        data = _read_json(p)
    except Exception:
        return {"path": _relpath(p), "exists": True, "parse_error": True}
    out: Dict[str, Any] = {"path": _relpath(p), "exists": True}
    for k in ("beta", "beta_sigma", "gamma_pmodel", "gamma_pmodel_sigma", "delta"):
        if k in data:
            out[k] = data.get(k)
    policy = data.get("policy")
    if isinstance(policy, dict):
        out["policy"] = {kk: policy.get(kk) for kk in ("fit_predict_separation", "beta_source", "delta_source", "note")}
    return out


def _n_eff(n0: int, n1: int) -> Optional[float]:
    if n0 <= 0 or n1 <= 0:
        return None
    return float(n0) * float(n1) / float(n0 + n1)


def _ks_z(ks: Optional[float], n0: int, n1: int) -> Optional[float]:
    if ks is None or not math.isfinite(float(ks)):
        return None
    ne = _n_eff(int(n0), int(n1))
    if ne is None:
        return None
    return float(ks) * math.sqrt(float(ne))


def _extract_ks_delay(entry: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (ks_alice, ks_bob) if present.
    Accepts keys: {"Alice","Bob"} or {"A","B"}.
    """
    ks = entry.get("ks_delay")
    if not isinstance(ks, dict):
        return (None, None)
    ks_a = _as_float(ks.get("Alice"))
    ks_b = _as_float(ks.get("Bob"))
    if ks_a is None:
        ks_a = _as_float(ks.get("A"))
    if ks_b is None:
        ks_b = _as_float(ks.get("B"))
    return (ks_a, ks_b)


def _extract_delay_signature(entry: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns (method, alice_sig, bob_sig) if present.
    """
    sig = entry.get("delay_signature")
    if not isinstance(sig, dict):
        return (None, None, None)
    method = sig.get("method") if isinstance(sig.get("method"), dict) else None
    a = sig.get("Alice") if isinstance(sig.get("Alice"), dict) else None
    b = sig.get("Bob") if isinstance(sig.get("Bob"), dict) else None
    return (method, a, b)


def _criterion(
    *,
    cid: str,
    test_id: str,
    title: str,
    value: Any,
    op: str,
    threshold: Any,
    passed: Optional[bool],
    gate: bool,
    unit: str = "",
    rationale: str = "",
    source: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "id": str(cid),
        "test_id": str(test_id),
        "title": str(title),
        "value": value,
        "op": str(op),
        "threshold": threshold,
        "pass": passed,
        "gate": bool(gate),
        "unit": str(unit),
        "rationale": str(rationale),
        "source": source or None,
        "notes": str(notes),
    }


def _estimate_atom_interferometer_beta_delta(
    atom_metrics: Dict[str, Any],
    *,
    beta_frozen: Optional[float],
) -> Dict[str, Any]:
    beta_dep = (
        (atom_metrics.get("results") or {}).get("beta_phase_dependence")
        if isinstance(atom_metrics.get("results"), dict)
        else None
    )
    if isinstance(beta_dep, dict) and beta_dep.get("status") == "ok":
        return beta_dep

    # Fallback (kept for robustness if the upstream metrics have not been regenerated):
    # conservative “absolute potential” scaling (upper-bound style; not a strict derivation).
    phi_ref = _as_float(
        ((atom_metrics.get("results") or {}) if isinstance(atom_metrics.get("results"), dict) else {}).get("phi_ref_rad")
    )
    if phi_ref is None:
        return {"status": "missing_phi_ref_rad"}
    if beta_frozen is None or not math.isfinite(float(beta_frozen)):
        return {"status": "missing_beta_frozen"}

    c = 299_792_458.0
    gm_earth = 3.986_004_418e14  # m^3/s^2 (WGS84)
    r_earth = 6_378_137.0  # m (WGS84 equatorial radius)
    phi_earth = -gm_earth / r_earth  # m^2/s^2 (negative)

    d_beta = float(beta_frozen) - 1.0
    frac_lin = -2.0 * d_beta * (phi_earth / (c**2))
    dphi_lin = float(phi_ref) * float(frac_lin)
    sigma_phi_required_1sigma = abs(dphi_lin) / 3.0

    return {
        "status": "ok",
        "model": {
            "name": "fallback_absolute_potential_upper_bound",
            "note": "Fallback only. Regenerate output/public/quantum/atom_interferometer_gravimeter_phase_metrics.json to use the fixed Step 7.5.8 model.",
        },
        "beta_frozen": float(beta_frozen),
        "delta_beta_vs_gr": float(d_beta),
        "relative_delta_phase": float(frac_lin),
        "delta_phase_rad": float(dphi_lin),
        "sigma_phase_required_1sigma_for_3sigma_detection_rad": float(sigma_phi_required_1sigma),
    }


def _load_nist_setting_counts(dataset_id: str) -> Optional[Dict[str, List[int]]]:
    npz = _ROOT / "output" / "public" / "quantum" / "bell" / dataset_id / "normalized_events.npz"
    if not npz.exists():
        return None
    try:
        with np.load(npz) as data:
            a_set = data["alice_click_setting"].astype(np.int64, copy=False)
            b_set = data["bob_click_setting"].astype(np.int64, copy=False)
    except Exception:
        return None
    ca = np.bincount(a_set, minlength=2)[:2].astype(int).tolist()
    cb = np.bincount(b_set, minlength=2)[:2].astype(int).tolist()
    return {"alice_clicks_by_setting": ca, "bob_clicks_by_setting": cb}


def _load_nist_delay_stats(dataset_id: str) -> Optional[Dict[str, Any]]:
    """
    Load "physical" (ns) delay summary for NIST time-tag datasets.

    Source is the fixed output from scripts/quantum/nist_belltest_time_tag_reanalysis.py:
      output/public/quantum/nist_belltest_time_tag_bias_metrics__<out_tag>.json

    dataset_id example:
      nist_03_43_afterfixingModeLocking_s3600  -> out_tag=03_43_afterfixingModeLocking_s3600
    """
    if not dataset_id.startswith("nist_"):
        return None
    out_tag = dataset_id[len("nist_") :]
    p = _ROOT / "output" / "public" / "quantum" / f"nist_belltest_time_tag_bias_metrics__{out_tag}.json"
    if not p.exists():
        return None
    try:
        j = _read_json(p)
    except Exception:
        return None
    ds = j.get("delay_stats_ns")
    if not isinstance(ds, dict):
        return None
    a = ds.get("alice") if isinstance(ds.get("alice"), dict) else {}
    b = ds.get("bob") if isinstance(ds.get("bob"), dict) else {}
    a0 = _as_float(a.get("setting0_median"))
    a1 = _as_float(a.get("setting1_median"))
    b0 = _as_float(b.get("setting0_median"))
    b1 = _as_float(b.get("setting1_median"))
    if (a0 is None or a1 is None) and (b0 is None or b1 is None):
        return None
    return {
        "source": _relpath(p),
        "by_side": {
            "alice": (
                None
                if (a0 is None or a1 is None)
                else {"setting0_median": float(a0), "setting1_median": float(a1), "delta_median_0_minus_1": float(a0 - a1)}
            ),
            "bob": (
                None
                if (b0 is None or b1 is None)
                else {"setting0_median": float(b0), "setting1_median": float(b1), "delta_median_0_minus_1": float(b0 - b1)}
            ),
        },
    }


def _load_trial_setting_counts(dataset_id: str) -> Optional[Dict[str, List[int]]]:
    p = _ROOT / "output" / "public" / "quantum" / "bell" / dataset_id / "trial_based_counts.json"
    if not p.exists():
        return None
    try:
        j = _read_json(p)
    except Exception:
        return None
    counts = j.get("counts") if isinstance(j.get("counts"), dict) else {}
    a = counts.get("alice_clicks_by_setting")
    b = counts.get("bob_clicks_by_setting")
    if isinstance(a, list) and isinstance(b, list) and len(a) >= 2 and len(b) >= 2:
        try:
            return {"alice_clicks_by_setting": [int(a[0]), int(a[1])], "bob_clicks_by_setting": [int(b[0]), int(b[1])]}
        except Exception:
            return None
    return None


def _load_weihs_pair_counts_at_ref_window(dataset_id: str) -> Optional[Dict[str, Any]]:
    p = _ROOT / "output" / "public" / "quantum" / "bell" / dataset_id / "window_sweep_metrics.json"
    if not p.exists():
        return None
    try:
        wj = _read_json(p)
    except Exception:
        return None
    cfg = wj.get("config") if isinstance(wj.get("config"), dict) else {}
    ref_window = _as_float(cfg.get("ref_window_ns"))
    if ref_window is None:
        return None
    rows = wj.get("rows")
    if not isinstance(rows, list):
        return None
    best = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        w = _as_float(r.get("window_ns"))
        if w is None:
            continue
        if abs(w - float(ref_window)) < 1e-12:
            best = r
            break
    if not isinstance(best, dict):
        return None
    n_by = best.get("n_by_setting")
    if not (isinstance(n_by, list) and len(n_by) == 2 and all(isinstance(x, list) and len(x) == 2 for x in n_by)):
        return None
    try:
        n00 = int(n_by[0][0])
        n01 = int(n_by[0][1])
        n10 = int(n_by[1][0])
        n11 = int(n_by[1][1])
    except Exception:
        return None
    return {"ref_window_ns": float(ref_window), "pairs_by_setting_pair": [[n00, n01], [n10, n11]]}


def build_quantum_falsification(*, frozen: Dict[str, Any]) -> Dict[str, Any]:
    beta_frozen = _as_float(frozen.get("beta"))

    cow_path = _ROOT / "output" / "public" / "quantum" / "cow_phase_shift_metrics.json"
    atom_path = _ROOT / "output" / "public" / "quantum" / "atom_interferometer_gravimeter_phase_metrics.json"
    bell_pack_path = _ROOT / "output" / "public" / "quantum" / "bell" / "falsification_pack.json"
    bell_sel_summary_path = _ROOT / "output" / "public" / "quantum" / "bell_selection_sensitivity_summary.json"

    cow = _read_json(cow_path) if cow_path.exists() else {}
    atom = _read_json(atom_path) if atom_path.exists() else {}
    bell_pack = _read_json(bell_pack_path) if bell_pack_path.exists() else {}
    bell_sel = _read_json(bell_sel_summary_path) if bell_sel_summary_path.exists() else {}

    criteria: List[Dict[str, Any]] = []

    # ----------------
    # Interferometry: COW (β does not enter at leading order; consistency check)
    # ----------------
    criteria.append(
        _criterion(
            cid="cow_beta_dependence",
            test_id="cow_neutron_interferometer",
            title="COW（中性子干渉）の重力位相（最低次）が β に依存しない（現モデル）",
            value=0.0,
            op="==",
            threshold=0.0,
            passed=True,
            gate=False,
            unit="rad (Δφ_beta)",
            rationale="現実装のCOW位相は Δφ = -m g H^2/(ħ v0)（物質波の重力ポテンシャル差）であり、光伝播自由度 β（n(P)）は入らない。",
            source={"cow_phase_shift_metrics_json": _relpath(cow_path)},
            notes="差分予測（決着点）としては弱いが、P-model の dτ/dt と物質波干渉が矛盾しないことの整合チェック。",
        )
    )

    # ----------------
    # Interferometry: Atom interferometer (β micro-difference estimate via optical phase)
    # ----------------
    atom_beta = _estimate_atom_interferometer_beta_delta(atom, beta_frozen=beta_frozen)
    criteria.append(
        _criterion(
            cid="atom_beta_phase_delta_est",
            test_id="atom_interferometer_gravimeter",
            title="原子干渉計（gravimeter）の β 依存位相差（Mueller導出に沿う差分スケール；GRとの差分）",
            value=atom_beta,
            op="struct",
            threshold={"note": "Use sigma_phase_required_1sigma_for_3sigma_detection_rad as the 3σ decisiveness target."},
            passed=None,
            gate=False,
            unit="rad",
            rationale="β は光伝播（n(P)）に入り、Mueller 2007 の扱い（k_eff を光の分散から決める）ではレーザー位相項を通じて位相へ入る。定数項は相殺/吸収されるため、差分は干渉計内のポテンシャル差で抑圧される（本JSONではスケールと必要精度を固定する）。",
            source={
                "atom_interferometer_gravimeter_phase_metrics_json": _relpath(atom_path),
                "frozen_parameters_json": frozen.get("path"),
            },
            notes="value.models.A は“絶対ポテンシャル”の上限見積もり、value.models.B が局所差分（本稿の採用；chosen_model_for_falsification）。",
        )
    )

    # ----------------
    # Bell: setting-dependent delay signature (Δmedian; z) — freeze as a physical (ns) statistic
    # ----------------
    z_threshold = 3.0
    bell_rows: List[Dict[str, Any]] = []
    datasets = bell_pack.get("datasets") if isinstance(bell_pack.get("datasets"), list) else []
    for d in datasets:
        if not isinstance(d, dict):
            continue
        dataset_id = str(d.get("dataset_id") or "")
        method, sig_a, sig_b = _extract_delay_signature(d)
        z_a = _as_float(sig_a.get("z_delta_median")) if isinstance(sig_a, dict) else None
        z_b = _as_float(sig_b.get("z_delta_median")) if isinstance(sig_b, dict) else None
        if z_a is None and z_b is None:
            continue

        d_a = _as_float(sig_a.get("delta_median_0_minus_1_ns")) if isinstance(sig_a, dict) else None
        d_b = _as_float(sig_b.get("delta_median_0_minus_1_ns")) if isinstance(sig_b, dict) else None
        s_a = _as_float(sig_a.get("sigma_delta_median_ns")) if isinstance(sig_a, dict) else None
        s_b = _as_float(sig_b.get("sigma_delta_median_ns")) if isinstance(sig_b, dict) else None

        ks_a, ks_b = _extract_ks_delay(d)

        bell_rows.append(
            {
                "dataset_id": dataset_id,
                "delay_method": method,
                "delay_delta_median_ns": {"alice": d_a, "bob": d_b},
                "delay_sigma_delta_median_ns": {"alice": s_a, "bob": s_b},
                "delay_z_alice": z_a,
                "delay_z_bob": z_b,
                "ks_delay_alice_legacy": ks_a,
                "ks_delay_bob_legacy": ks_b,
            }
        )

    # Create a single criterion summarizing the “fast switching” decisiveness gate.
    min_z = None
    zs_all: List[float] = []
    for r in bell_rows:
        for k in ("delay_z_alice", "delay_z_bob"):
            v = r.get(k)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                zs_all.append(float(v))
    if zs_all:
        min_z = min(zs_all)

    # Focus the “fast switching” decisiveness gate on datasets where rapid basis switching is a core feature.
    # (Operational classification by dataset id; can be refined when per-dataset metadata is added.)
    fast_ids = [r["dataset_id"] for r in bell_rows if str(r.get("dataset_id") or "").startswith(("weihs1998_", "nist_"))]
    zs_fast: List[float] = []
    for r in bell_rows:
        if str(r.get("dataset_id") or "") not in set(fast_ids):
            continue
        z_candidates: List[float] = []
        for k in ("delay_z_alice", "delay_z_bob"):
            v = r.get(k)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                z_candidates.append(float(v))
        if z_candidates:
            zs_fast.append(max(z_candidates))
    min_z_fast = min(zs_fast) if zs_fast else None

    criteria.append(
        _criterion(
            cid="bell_delay_signature_ks_3sigma",
            test_id="bell_fast_switching_delay_signature",
            title="Bell（高速スイッチング等）: setting 依存の遅延シグネチャが 3σ 相当で見える（Δmedian(ns) / σ）",
            value={
                "z_threshold": z_threshold,
                "min_delay_z_observed_all": min_z,
                "fast_switching_dataset_ids": fast_ids,
                "min_delay_z_observed_fast_switching": min_z_fast,
                "per_dataset": bell_rows,
            },
            op="min_fast max(z_side) >= 3",
            threshold=z_threshold,
            passed=(None if min_z_fast is None else bool(float(min_z_fast) >= z_threshold)),
            gate=False,
            unit="z = |Δmedian(ns)| / σ(Δmedian)",
            rationale="高速スイッチング環境では、time-tag の遅延分布が setting に依存して変化し得る。これを「遅延量（Δmedian(ns)）」と「残差（z）」として固定し、3σ を閾値として [Reject] に接続する。",
            source={
                "bell_falsification_pack_json": _relpath(bell_pack_path),
                "bell_selection_sensitivity_summary_json": _relpath(bell_sel_summary_path) if bell_sel_summary_path.exists() else None,
            },
            notes="σ(Δmedian) は中央値近傍の mid-quantile 幅から近似している（bell_primary_products.py 側で定義）。KSは proxy として併記する（legacy）。Giustina2015 click log 等の一次データが入手でき次第、同一指標へ拡張する。",
        )
    )

    gate_fail = [c for c in criteria if c.get("gate") and c.get("pass") is False]
    gate_unknown = [c for c in criteria if c.get("gate") and c.get("pass") is None]
    overall_gate_pass = (len(gate_fail) == 0) and (len(gate_unknown) == 0)

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 7, "step": "7.5.7", "name": "Quantum: decisive micro-differences (interference + Bell delay)"},
        "inputs": {
            "frozen_parameters": frozen,
            "cow_phase_shift_metrics_json": _relpath(cow_path),
            "atom_interferometer_gravimeter_phase_metrics_json": _relpath(atom_path),
            "bell_falsification_pack_json": _relpath(bell_pack_path),
            "bell_selection_sensitivity_summary_json": _relpath(bell_sel_summary_path) if bell_sel_summary_path.exists() else None,
        },
        "policy": {
            "beta_frozen": beta_frozen,
            "note": "Part I の凍結βを固定した上で、“微小差分”が観測統計（3σ）を超える決着点を出力として固定する（初版）。",
        },
        "rejection_protocol": {
            "format": ["Input", "Frozen", "Output", "Statistic", "Reject"],
            "reference": "doc/paper/10_part1_core_theory.md:3.0",
        },
        "criteria": criteria,
        "summary": {
            "gate_pass": bool(overall_gate_pass),
            "gate_fail_n": len(gate_fail),
            "gate_unknown_n": len(gate_unknown),
            "gate_fail_ids": [str(c.get("id")) for c in gate_fail],
            "gate_unknown_ids": [str(c.get("id")) for c in gate_unknown],
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate Phase 7 quantum falsification/decisiveness pack JSON.")
    ap.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "quantum_falsification.json"),
        help="Output JSON path (default: output/private/summary/quantum_falsification.json).",
    )
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frozen = _try_load_frozen_parameters()
    payload = build_quantum_falsification(frozen=frozen)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] wrote: {_relpath(out_path)}")
    try:
        worklog.append_event(
            {
                "event_type": "summary_quantum_falsification",
                "phase": "7.5.7",
                "inputs": {
                    "frozen_parameters_json": frozen.get("path"),
                    "cow_phase_shift_metrics_json": payload.get("inputs", {}).get("cow_phase_shift_metrics_json"),
                    "atom_interferometer_gravimeter_phase_metrics_json": payload.get("inputs", {}).get(
                        "atom_interferometer_gravimeter_phase_metrics_json"
                    ),
                    "bell_falsification_pack_json": payload.get("inputs", {}).get("bell_falsification_pack_json"),
                },
                "outputs": {"quantum_falsification_json": _relpath(out_path)},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
