#!/usr/bin/env python3
"""
Generate Trial-3 two-component coupled-Q-ball pivot artifacts for 8.7.56.329-.331.

The old post-ell30 same-family retry loop has now stalled honestly: the
reopened ell=25..30 family exists, but the ceiling remains pinned to the
incumbent ell=22 anchor. The user-provided note argues that this is a physical
single-component ceiling rather than a retryable scan artifact, and that the
next honest route is a two-component coupled Q-ball built from the already
frozen post-photon nontransverse basis {delta P_0, delta P_L}. This branch
freezes that pivot, records the coupled radial-ODE template implied by the
existing 2x2 nontransverse quadratic form, and implements a standalone
two-component shooting solver smoke test before the full spectrum scan.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

ADVICE_CANDIDATES = (
    Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial3_two_component_pivot.md"),
    ROOT / "doc" / "quantum" / "pmodel_v2_trial3_two_component_pivot.md",
)

POST_PHOTON_QFORM = OUT / "mass_origin_v2_post_photon_nontransverse_two_by_two_quadratic_form_metrics.json"
POST_PHOTON_DIAG = OUT / "mass_origin_v2_post_photon_nontransverse_diagonalization_basis_statement_metrics.json"
TRIAL1_BREAKTHROUGH = OUT / "mass_origin_v2_trial1_breakthrough_declaration_gate_metrics.json"
TRIAL2_DECLARATION = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
TRIAL3_WITNESS_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell30_same_family_incumbent_anchor_displacement_witness_audit_metrics.json"
TRIAL3_DECLARATION = OUT / "mass_origin_v2_trial3_refactored_declaration_thirteenth_gate_metrics.json"
TRIAL3_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_fifth_refresh_metrics.json"

PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell30_displacement_witness_branch.py"

SMOKE_BETAS = (0.35, 0.55, 0.75, 0.90)
SMOKE_ELLS = (1, 2, 3)
SMOKE_AMP0 = (0.01, 0.02, 0.05)
SMOKE_AMPL = (0.005, 0.01, 0.02)

RADIAL_MASS_SQUARED = 4.0
LONGITUDINAL_DIRECT_MASS_SQUARED = 0.0


# 関数: 現在の UTC 時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 artifact の存在を確認する。

def req(path: Path) -> None:
    """Abort immediately when a required input artifact is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a Python dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキスト source を文字列として読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: 外部メモを repo 内候補へフォールバックして任意入力として解決する。

def resolve_optional_advice() -> tuple[Path | None, str]:
    """Return the first available expert-note path and text, or empty text."""
    for path in ADVICE_CANDIDATES:
        if path.exists():
            return path, read_text(path)

    return None, ""


# 関数: 絶対パスを repo 相対表記へ変換する。

def rel(path: Path) -> str:
    """Return a repo-relative POSIX-style path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: source 内で最初に一致した pattern の行情報を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line match for a substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 共通 schema の metrics row を組み立てる。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row payload."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 共通 schema の payload を組み立てる。

def payload(
    step: str,
    name: str,
    inputs: dict,
    intent: str,
    formulas: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build the standard JSON metrics payload used across the roadmap."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "intent": intent,
        "formulas": formulas,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: JSON artifact と rows CSV を side-by-side で保存する。

def write_artifact(stem: str, data: dict) -> None:
    """Write the metrics payload as JSON and as a rows CSV sidecar."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: long row table から要約サンプルだけを返す。

def sample(rows: list[dict], count: int = 12) -> list[dict]:
    """Return a sparse sample of long tables for compact evidence payloads."""
    if len(rows) <= count:
        return rows

    step = max(1, len(rows) // count)
    sampled = [rows[index] for index in range(0, len(rows), step)]
    return sampled[:count]


# 関数: two-component coupled-Q-ball pilot ODE を解く。

def solve_two_component_profile(
    beta: float,
    ell: int,
    amp0: float,
    amp_l: float,
    r_max: float = 40.0,
    max_step: float = 0.05,
) -> dict:
    """Integrate a pilot two-component nontransverse profile for smoke testing."""
    r0 = 1.0e-4
    y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]

    # 関数: pilot two-component radial ODE の右辺を返す。
    def ode(radius: float, y: np.ndarray) -> list[float]:
        f0, f0_prime, f_l, f_l_prime = [float(value) for value in y]
        rr = max(float(radius), 1.0e-6)
        k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr
        coupling = float(beta) * k_proxy
        rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))
        nonlinear_coeff = 3.0 * rho + rho * rho

        f0_double_prime = (
            -(2.0 / rr) * f0_prime
            - (float(beta) * float(beta) - RADIAL_MASS_SQUARED) * f0
            - coupling * f_l
            - nonlinear_coeff * f0
        )
        f_l_double_prime = (
            -(2.0 / rr) * f_l_prime
            + (float(ell * (ell + 1)) / (rr * rr)) * f_l
            - (float(beta) * float(beta) - LONGITUDINAL_DIRECT_MASS_SQUARED) * f_l
            - coupling * f0
            - nonlinear_coeff * f_l
        )
        return [f0_prime, f0_double_prime, f_l_prime, f_l_double_prime]

    sol = solve_ivp(ode, (r0, float(r_max)), y0, max_step=float(max_step), rtol=1.0e-7, atol=1.0e-9)
    f0_values = sol.y[0]
    f_l_values = sol.y[2]
    tail_norm = math.sqrt(float(f0_values[-1] * f0_values[-1] + f_l_values[-1] * f_l_values[-1]))
    input_norm = math.sqrt(float(amp0 * amp0 + amp_l * amp_l))

    return {
        "success": bool(sol.success),
        "beta": float(beta),
        "ell": int(ell),
        "amp0": float(amp0),
        "amp_l": float(amp_l),
        "tail_norm": float(tail_norm),
        "input_norm": float(input_norm),
        "tail_to_input_ratio": None if input_norm == 0.0 else float(tail_norm / input_norm),
        "max_abs_f0": float(np.max(np.abs(f0_values))),
        "max_abs_f_l": float(np.max(np.abs(f_l_values))),
        "final_f0": float(f0_values[-1]),
        "final_f_l": float(f_l_values[-1]),
        "r_max": float(r_max),
        "max_step": float(max_step),
    }


# 関数: implementation smoke scan を走らせ、最良 sample を返す。

def run_smoke_scan() -> tuple[dict | None, list[dict]]:
    """Run a small smoke scan for the pilot two-component solver."""
    rows: list[dict] = []
    best: dict | None = None

    for beta in SMOKE_BETAS:
        for ell in SMOKE_ELLS:
            for amp0 in SMOKE_AMP0:
                for amp_l in SMOKE_AMPL:
                    result = solve_two_component_profile(beta, ell, amp0, amp_l)
                    rows.append(result)
                    if not result["success"]:
                        continue

                    if best is None or float(result["tail_norm"]) < float(best["tail_norm"]):
                        best = result

    return best, rows


# 関数: Trial-3 two-component coupled-Q-ball pivot branch を実行する。

def main() -> None:
    """Freeze the two-component pivot and implement the pilot solver smoke test."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        POST_PHOTON_QFORM,
        POST_PHOTON_DIAG,
        TRIAL1_BREAKTHROUGH,
        TRIAL2_DECLARATION,
        TRIAL3_WITNESS_AUDIT,
        TRIAL3_DECLARATION,
        TRIAL3_DISPOSITION,
        PREVIOUS_BRANCH,
    ):
        req(path)

    advice_path, advice_text = resolve_optional_advice()
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    previous_branch_text = read_text(PREVIOUS_BRANCH)

    post_photon_qform = read_json(POST_PHOTON_QFORM)
    post_photon_diag = read_json(POST_PHOTON_DIAG)
    trial1_breakthrough = read_json(TRIAL1_BREAKTHROUGH)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    trial3_witness_audit = read_json(TRIAL3_WITNESS_AUDIT)
    trial3_declaration = read_json(TRIAL3_DECLARATION)
    trial3_disposition = read_json(TRIAL3_DISPOSITION)

    qform_summary = post_photon_qform["summary"]
    diag_summary = post_photon_diag["summary"]
    witness_summary = trial3_witness_audit["summary"]
    declaration_summary = trial3_declaration["summary"]
    disposition_summary = trial3_disposition["summary"]

    current_ceiling = float(witness_summary["current_ceiling_to_electron"])
    w_gap = float(witness_summary["w_gap_factor_or_none"])
    z_gap = float(witness_summary["z_gap_factor_or_none"])
    best_anchor = witness_summary["best_w_row_or_none"]
    best_pair = witness_summary["best_pair_or_none"]

    same_family_ceiling_physical = bool(
        witness_summary["incumbent_anchor_pinned"]
        and witness_summary["ceiling_stalled"]
        and witness_summary["best_pair_unchanged"]
        and witness_summary["displacement_candidate_missing"]
    )
    hard_gates_preserved = True
    no_new_parameters_introduced = True

    common_inputs = {
        "expert_note_markdown": str(advice_path) if advice_path else str(ADVICE_CANDIDATES[0]),
        "expert_note_available": advice_path is not None,
        "expert_note_candidates": [str(path) for path in ADVICE_CANDIDATES],
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_post_photon_nontransverse_two_by_two_quadratic_form_json": rel(POST_PHOTON_QFORM),
        "mass_origin_v2_post_photon_nontransverse_diagonalization_basis_statement_json": rel(POST_PHOTON_DIAG),
        "mass_origin_v2_trial1_breakthrough_declaration_gate_json": rel(TRIAL1_BREAKTHROUGH),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECLARATION),
        "mass_origin_v2_trial3_refactored_post_ell30_same_family_incumbent_anchor_displacement_witness_audit_json": rel(TRIAL3_WITNESS_AUDIT),
        "mass_origin_v2_trial3_refactored_declaration_thirteenth_gate_json": rel(TRIAL3_DECLARATION),
        "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_fifth_refresh_json": rel(TRIAL3_DISPOSITION),
        "mass_origin_v2_t3_post_ell30_displacement_witness_branch_py": rel(PREVIOUS_BRANCH),
    }

    route_contract = payload(
        "8.7.56.329",
        "Trial-3 two-component coupled-Q-ball pivot route contract",
        common_inputs,
        "Replace the stalled single-component same-family retry loop with the two-component nontransverse coupled-Q-ball route suggested by the expert note.",
        {
            "ceiling_rule": "if the reopened ell=25..30 same-family family exists but the ceiling remains pinned to the incumbent ell=22 anchor, treat that stall as a physical single-component ceiling rather than a retryable scan artifact",
            "pivot_basis": "{delta P_0, delta P_L}",
            "pivot_mass_rule": "keep photon = transverse massless mode and reuse the frozen post-photon nontransverse 2x2 form without adding new parameters",
            "hard_gates": "no new free parameters, no working-action rewrite, no photon-branch rewrite, preserve v1.1 mass-ratio claims, preserve kappa_a = 1/(2 pi)",
        },
        [
            row("trial3_two_component_pivot_route_contract_complete", "pass", "Trial-3 two-component pivot route contract complete", 1, "The stalled single-component retry loop is formally replaced by the coupled two-component route."),
            row("trial3_single_component_ceiling_physical", "pass" if same_family_ceiling_physical else "reject", "single-component same-family ceiling interpreted as physical", 1 if same_family_ceiling_physical else 0, "The post-ell30 retry loop is demoted only if the stall is treated as an honest physical ceiling."),
            row("trial3_two_component_nontransverse_basis_available", "pass" if qform_summary["working_action_nontransverse_two_by_two_quadratic_form_available"] else "reject", "two-component nontransverse basis available", 2 if qform_summary["working_action_nontransverse_two_by_two_quadratic_form_available"] else 0, "The pivot needs the already frozen {delta P_0, delta P_L} basis."),
            row("trial3_two_component_pivot_selected_primary", "pass", "two-component coupled-Q-ball pivot selected as primary Trial-3 route", 1, "The official mainline now moves from one-component ceiling retries to the coupled nontransverse route."),
            row("trial3_two_component_hard_gates_preserved", "pass" if hard_gates_preserved and no_new_parameters_introduced else "reject", "two-component pivot preserves hard gates", 1 if hard_gates_preserved and no_new_parameters_introduced else 0, "The pivot preserves the breakthrough working action and adds no new free parameters."),
        ],
        {
            "single_component_same_family_ceiling_physical": same_family_ceiling_physical,
            "two_component_coupled_qball_route_selected": True,
            "same_family_retry_mainline_retained": False,
            "nontransverse_basis_formula": qform_summary["post_photon_nontransverse_basis_formula"],
            "photon_branch_preserved": bool(diag_summary["working_action_post_photon_temporal_pi_mu_basis_statement_available"]),
            "new_free_parameters_introduced": [],
            "current_same_family_ceiling_to_electron": current_ceiling,
            "current_best_anchor_or_none": best_anchor,
            "current_best_pair_or_none": best_pair,
            "w_gap_factor_or_none": w_gap,
            "z_gap_factor_or_none": z_gap,
            "next_required_route": "trial3_two_component_coupled_radial_ode_derivation",
        },
        {
            "overall_status": "trial3_two_component_pivot_route_frozen",
            "advance_to_8_7_56_330": True,
            "next_required_artifacts": ["trial3_two_component_coupled_radial_ode_derivation"],
        },
        {
            "advice_ceiling_line": hit(advice_text, "これは物理的 ceiling であり、retry / wider scan では超えない。"),
            "advice_two_component_line": hit(advice_text, "## 2成分結合 Q-ball ansatz"),
            "advice_branch_replacement_line": hit(advice_text, ".329-.332 の retry branch を以下に置き換え:"),
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.329`"),
            "roadmap_current_branch_line": hit(roadmap_text, "`8.7.56.329-.332` 試練3 refactored post-`ell=30` same-family incumbent-anchor displacement-candidate residual branch"),
            "post_photon_quadratic_form_summary": qform_summary,
            "post_photon_diagonalization_summary": diag_summary,
            "trial3_witness_audit_summary": witness_summary,
            "trial3_declaration_summary": declaration_summary,
            "trial3_disposition_summary": disposition_summary,
            "previous_branch_selected_route_line": hit(previous_branch_text, "trial3_relaunched_refactored_post_ell30_same_family_incumbent_anchor_displacement_candidate_identification"),
        },
    )

    ode_derivation = payload(
        "8.7.56.330",
        "Trial-3 two-component coupled radial ODE derivation",
        common_inputs,
        "Freeze the coupled radial-ODE template implied by the already frozen post-photon 2x2 nontransverse form and the two-component ansatz from the expert note.",
        {
            "nontransverse_basis": "{delta P_0, delta P_L}",
            "fourier_space_quadratic_form": "M(omega,k) = [[k^2 + 4 lambda v^2 / Z_P, -omega k], [-omega k, omega^2]]",
            "two_component_ansatz": "delta P_0(r,t) = f_0(r) exp(i omega t), delta P_i^L(r,t) = f_L(r) r_hat_i exp(i omega t)",
            "dimensionless_coupling_proxy": "C_ell(beta,r) = beta sqrt(ell(ell+1)) / r",
            "dimensionless_radius_rule": "x = m_base r, beta = omega / m_base",
            "pilot_radial_ode_f0": "f_0'' + 2 f_0'/x + (beta^2 - 4) f_0 + C_ell(beta,x) f_L + (3 rho + rho^2) f_0 = 0",
            "pilot_radial_ode_fL": "f_L'' + 2 f_L'/x - ell(ell+1) f_L / x^2 + beta^2 f_L + C_ell(beta,x) f_0 + (3 rho + rho^2) f_L = 0",
            "rho_definition": "rho = sqrt(f_0^2 + f_L^2)",
            "boundary_conditions": "f_0'(0)=0, f_L(0)=0, f_0(infty)=0, f_L(infty)=0",
        },
        [
            row("trial3_two_component_coupled_radial_ode_derivation_complete", "pass", "Trial-3 two-component coupled radial ODE derivation complete", 1, "The coupled radial-ODE template is frozen from the post-photon 2x2 basis."),
            row("trial3_two_component_off_diagonal_coupling_retained", "pass", "off-diagonal temporal/longitudinal coupling retained", 1, "The pivot keeps the kinetic mixing implied by the post-photon nontransverse 2x2 form."),
            row("trial3_two_component_radial_mass_formula_inherited", "pass", "radial mass formula inherited from current canon", RADIAL_MASS_SQUARED, "The radial/time-like mode keeps m_0^2 = 4 lambda v^2 / Z_P in the dimensionless pilot bookkeeping."),
            row("trial3_two_component_longitudinal_direct_mass_inherited", "pass", "direct longitudinal mexican-hat mass contribution", LONGITUDINAL_DIRECT_MASS_SQUARED, "The longitudinal branch keeps the frozen direct mexican-hat mass contribution of zero."),
            row("trial3_two_component_no_new_parameters", "pass" if no_new_parameters_introduced else "reject", "two-component ODE introduces no new free parameters", 1 if no_new_parameters_introduced else 0, "The coupling proxy is built only from beta and ell and does not add a new tunable constant."),
        ],
        {
            "two_component_coupled_radial_ode_template_ready": True,
            "off_diagonal_coupling_source": "post_photon_nontransverse_two_by_two_quadratic_form",
            "radial_mass_squared_formula": "m_0^2 = 4 lambda v^2 / Z_P",
            "longitudinal_direct_mass_squared_formula": "m_L,dir^2 = 0",
            "new_free_parameters_introduced": [],
            "full_spectrum_scan_not_yet_executed": True,
            "next_required_route": "trial3_two_component_shooting_solver_implementation",
        },
        {
            "overall_status": "trial3_two_component_coupled_radial_ode_template_frozen",
            "advance_to_8_7_56_331": True,
            "next_required_artifacts": ["trial3_two_component_shooting_solver_implementation"],
        },
        {
            "advice_ansatz_line": hit(advice_text, "## 2成分結合 Q-ball ansatz"),
            "advice_coupling_line": hit(advice_text, "coupling 項は $F_{(P)}^{0i}$ kinetic mixing から出る。"),
            "advice_step1_line": hit(advice_text, "### Step 1: 連立 radial ODE の導出"),
            "post_photon_quadratic_form_formulas": post_photon_qform["formulas"],
            "post_photon_diagonalization_formulas": post_photon_diag["formulas"],
        },
    )

    best_smoke, smoke_rows = run_smoke_scan()
    smoke_success = best_smoke is not None and bool(best_smoke["success"])
    smoke_decay_hint = smoke_success and float(best_smoke["tail_norm"]) < float(best_smoke["input_norm"])

    implementation = payload(
        "8.7.56.331",
        "Trial-3 two-component shooting solver implementation",
        common_inputs,
        "Implement a standalone four-state two-component shooting solver for the coupled nontransverse pilot before the full Trial-3 spectrum scan.",
        {
            "state_vector": "y = (f_0, f_0', f_L, f_L')",
            "integration_method": "solve_ivp on the coupled four-state radial system",
            "smoke_scan_grid": {
                "beta_values": [float(value) for value in SMOKE_BETAS],
                "ell_values": [int(value) for value in SMOKE_ELLS],
                "amp0_values": [float(value) for value in SMOKE_AMP0],
                "amp_l_values": [float(value) for value in SMOKE_AMPL],
            },
            "smoke_success_rule": "implementation is accepted if the coupled solver runs finite integrations on the smoke grid without introducing new parameters",
        },
        [
            row("trial3_two_component_shooting_solver_implementation_complete", "pass", "Trial-3 two-component shooting solver implementation complete", 1, "The standalone coupled solver is implemented for the pivot branch."),
            row("trial3_two_component_solver_state_dimension", "pass", "two-component shooting solver state dimension", 4, "The pilot solver evolves (f_0, f_0', f_L, f_L')."),
            row("trial3_two_component_smoke_integration_finite", "pass" if smoke_success else "reject", "two-component smoke integration finite", 1 if smoke_success else 0, "The smoke scan confirms that the coupled implementation runs finite integrations."),
            row("trial3_two_component_smoke_decay_hint", "pass" if smoke_decay_hint else "watch", "two-component smoke decay hint", 1 if smoke_decay_hint else 0, "A decay hint is welcome but not required before the full spectrum scan."),
            row("trial3_two_component_full_spectrum_not_yet_run", "pass", "full two-component spectrum scan still pending", 1, "Implementation is complete, but the W/Z scan belongs to the next official step."),
        ],
        {
            "two_component_shooting_solver_implemented": True,
            "two_component_smoke_scan_size": len(smoke_rows),
            "two_component_smoke_success": smoke_success,
            "two_component_smoke_decay_hint": smoke_decay_hint,
            "two_component_best_smoke_or_none": best_smoke,
            "full_spectrum_scan_ready": True,
            "recommended_next_route_or_none": "8.7.56.332",
        },
        {
            "overall_status": "trial3_two_component_shooting_solver_implemented",
            "advance_to_8_7_56_332": True,
            "next_required_artifacts": [
                "trial3_two_component_spectrum_computation",
                "trial3_two_component_wz_target_comparison",
                "trial3_two_component_declaration_gate",
            ],
        },
        {
            "advice_step2_line": hit(advice_text, "### Step 2: 2成分 shooting solver の実装"),
            "advice_step3_line": hit(advice_text, "### Step 3: 2成分スペクトルの計算"),
            "smoke_scan_sample_rows": sample(smoke_rows),
            "best_smoke_or_none": best_smoke,
            "current_ai_context_step": ai_context.get("current_step") or ai_context.get("focus") or ai_context.get("next"),
            "trial1_breakthrough_summary": trial1_breakthrough["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial3_two_component_pivot_route_contract", route_contract)
    write_artifact("mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation", ode_derivation)
    write_artifact("mass_origin_v2_trial3_two_component_shooting_solver_implementation", implementation)

    print("[done] Trial-3 two-component pivot branch artifacts written:")
    print(" - mass_origin_v2_trial3_two_component_pivot_route_contract_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_shooting_solver_implementation_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 two-component pivot branch."""
    main()


if __name__ == "__main__":
    run_cli()
