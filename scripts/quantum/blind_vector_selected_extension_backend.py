#!/usr/bin/env python3
"""
Build the selected-extension backend pack for the blind-vector rerun lane.

This helper is the concrete implementation counterpart of the theorem-side
backend adapter contract frozen in `.5199-.5202`. It reuses the legacy
vector-Q-ball numerical/profile backend and the full-coupled ladder backend
to rebuild the retained exact ladder under the adopted selected extension
`Sigma_*^(pilot-HS)`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
SCALAR_SPECTRUM = PUBLIC_OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
VECTOR_SPIN = PUBLIC_OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
EXACT_HANDOFF = PUBLIC_OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
PHASE3_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_"
    "blind_vector_observable_gate_numeric_evaluation_metrics.json"
)
SCALAR_TARGET = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_"
    "reconciliation_review_numeric_evaluation_metrics.json"
)
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"


# 関数: 必須 input artifact / module の存在を確認する。
def require(path: Path) -> None:
    """Abort immediately when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON artifact into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: ローカル Python module を動的に読み込む。

def load_module(path: Path, module_name: str):
    """Load one Python module from a local file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: evidence の長い list から readable sample を切り出す。

def sample_rows(rows: list[dict], count: int = 8) -> list[dict]:
    """Return a short sample from a long row list."""
    if len(rows) <= count:
        return rows

    step = max(1, len(rows) // count)
    sampled = [rows[index] for index in range(0, len(rows), step)]
    return sampled[:count]


# 関数: selected-extension backend pack を actual solver call chain から再構成する。

def build_selected_extension_backend_pack(
    ell_values: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Materialize the blind-vector backend pack on the adopted selected extension."""
    for path in (
        SCALAR_SPECTRUM,
        VECTOR_SPIN,
        EXACT_HANDOFF,
        PHASE3_EVAL,
        SCALAR_TARGET,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
    ):
        require(path)

    profile_backend = load_module(NUMERICAL_BRANCH, "wavep_blind_vector_profile_backend")
    ladder_backend = load_module(FULL_COUPLED_BRANCH, "wavep_blind_vector_ladder_backend")

    scalar_payload = read_json(SCALAR_SPECTRUM)
    spin_payload = read_json(VECTOR_SPIN)
    exact_payload = read_json(EXACT_HANDOFF)
    phase3_payload = read_json(PHASE3_EVAL)
    scalar_target_payload = read_json(SCALAR_TARGET)

    scalar_modes = list(scalar_payload["evidence"]["discrete_mass_mode_rows"])
    lambda_rot = float(spin_payload["summary"]["lambda_rot_value"])
    retained_anchor = dict(exact_payload["summary"]["best_exact_match_or_none"])
    retained_q_window = {
        "zero": 0.0,
        "q_theory_over_m0": float(scalar_target_payload["summary"]["q_theory_over_m0"]),
        "m0": 1.0,
    }
    alpha_exact_at_q_theory = float(scalar_target_payload["summary"]["alpha_exact_at_q_theory"])
    blind_alpha_at_q_theory = float(phase3_payload["summary"]["blind_alpha_at_q_theory"])
    blind_target_keys = {
        "blind_F_at_zero": float(phase3_payload["summary"]["blind_F_at_zero"]),
        "blind_F_at_q_theory": float(phase3_payload["summary"]["blind_F_at_q_theory"]),
        "blind_F_at_m0": float(phase3_payload["summary"]["blind_F_at_m0"]),
        "blind_alpha_at_q_theory": blind_alpha_at_q_theory,
        "delta_alpha_sel_exact": float(blind_alpha_at_q_theory - alpha_exact_at_q_theory),
    }

    ell_scan_rows = {
        int(ell): profile_backend.scan_ell_sector(int(ell)) for ell in ell_values
    }
    base_modes_by_ell = {
        int(ell): profile_backend.interpolate_integer_modes(ell_scan_rows[int(ell)], int(ell))
        for ell in ell_values
    }
    exact_ladder = ladder_backend.build_exact_ladder(scalar_modes, base_modes_by_ell, lambda_rot)
    exact_comparisons, best_exact_match = ladder_backend.compare_known_targets(exact_ladder)

    return {
        "selected_extension_label": "Sigma_*^(pilot-HS)",
        "ell_values": [int(ell) for ell in ell_values],
        "scalar_modes": scalar_modes,
        "lambda_rot": lambda_rot,
        "retained_q_window": retained_q_window,
        "blind_target_keys": blind_target_keys,
        "retained_anchor_row": retained_anchor,
        "ell_scan_rows": ell_scan_rows,
        "ell_scan_counts": {int(ell): len(rows) for ell, rows in ell_scan_rows.items()},
        "base_modes_by_ell": base_modes_by_ell,
        "base_mode_counts": {
            int(ell): len(rows) for ell, rows in base_modes_by_ell.items()
        },
        "exact_ladder": exact_ladder,
        "exact_ladder_row_count": len(exact_ladder),
        "exact_comparisons": exact_comparisons,
        "comparison_row_count": len(exact_comparisons),
        "best_exact_match": best_exact_match,
        "available_k_values": sorted({int(row["k"]) for row in exact_ladder}),
        "max_detected_k": max(int(row["k"]) for row in exact_ladder),
        "evidence_samples": {
            "ell_scan_rows": {
                int(ell): sample_rows(rows) for ell, rows in ell_scan_rows.items()
            },
            "base_modes_by_ell": {
                int(ell): sample_rows(rows) for ell, rows in base_modes_by_ell.items()
            },
            "exact_ladder_rows": sample_rows(exact_ladder),
            "exact_comparison_rows": sample_rows(exact_comparisons),
        },
    }


# 関数: helper 実行時に short summary を返す。

def main() -> None:
    """Run the selected-extension backend helper and print a compact summary."""
    pack = build_selected_extension_backend_pack()
    summary = {
        "selected_extension_label": pack["selected_extension_label"],
        "ell_scan_counts": pack["ell_scan_counts"],
        "base_mode_counts": pack["base_mode_counts"],
        "exact_ladder_row_count": pack["exact_ladder_row_count"],
        "comparison_row_count": pack["comparison_row_count"],
        "best_exact_match": pack["best_exact_match"],
        "retained_anchor_row": pack["retained_anchor_row"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 関数: CLI entrypoint から helper summary を出力する。

if __name__ == "__main__":
    main()
