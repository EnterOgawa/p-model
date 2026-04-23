#!/usr/bin/env python3
"""Generate 8.7.56.1543-.1546 microscopic functional derivation artifacts.

This branch tests whether the current public pack can push the already explicit
generic matter current and microscopic rotational couplings all the way down to
an explicit source functional on the restored exact vector / Q-ball branch.

The key candidate bridge is Part III-A's `psi <-> P` map. The computation here
checks whether that bridge is strong enough to identify the microscopic fermion
bilinears

- psi_bar gamma^mu (1-gamma^5) psi / 2
- psi_bar sigma^{mu nu} psi

with explicit functionals of the restored exact vector branch profiles.

The result is intentionally strict. A scalar envelope / Noether-energy bridge
does not count as a spinor-bilinear constitutive map unless the pack writes that
map explicitly.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
COMPUTATION_EXPERT_SHARE = (
    ROOT / "doc" / "quantum" / "43_trial2_numeric_alpha_vector_qball_computation_reactivation_expert_share.md"
)
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EMBEDDING_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1539_1542_matter_rot_embedding_audit_declaration_gate_metrics.json"
)
CURRENT_DERIVATION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1535_1538_charge_current_closure_derivation_declaration_gate_metrics.json"
)
JEFF_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_jeff_derivation_20260328.md")

STEP_TAG = "8.7.56.1543-1546"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor microscopic matter-current / rotational-source functional derivation"
STEM = build_compact_artifact_stem(STEP_TAG, "micro_source_fn_deriv", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_matter_rot_embedding_missing_microscopic_functional_derivation_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_microscopic_functional_derivation_failed_scalar_to_spinor_constitutive_gap_reopen_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_scalar_to_spinor_constitutive_map_reopen_audit"
)
NEXT_ROUTE = "8.7.56.1547"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: 表示用の相対パスを返す。

def display_path(path: Path) -> str:
    """Convert one path into repo-relative display text when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: metrics row を構成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: payload を構成する。

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: JSON/CSV 成果物を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one CSV rows table."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: branch で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return the explicit bridge and microscopic target surfaces."""
    return {
        "psi_to_p_bridge": (
            "psi = sqrt(2 omega_* P_*^2) * delta P_+ / P_*"
        ),
        "scalar_noether_bridge": (
            "delta T_(2)^(00) = hbar omega_* |psi|^2 + (hbar^2 / 2m_*) |nabla psi|^2 + m_* phi |psi|^2 + ..."
        ),
        "microscopic_chiral_surface": (
            "-lambda_rot g_P psi_bar gamma^mu (1-gamma^5)/2 psi P_mu"
        ),
        "microscopic_pauli_surface": (
            "-(lambda_rot g_P / 4m) psi_bar sigma^{mu nu} psi F^(P)_{mu nu}"
        ),
        "missing_constitutive_map": (
            "{delta P_+/P_*, |psi|^2, phase} -> {psi_bar gamma^mu (1-gamma^5) psi,"
            " psi_bar sigma^{mu nu} psi}"
        ),
    }


# 関数: `.1543-.1546` を実行する。

def main() -> None:
    """Execute the microscopic matter-current / rotational-source derivation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        COMPUTATION_EXPERT_SHARE,
        PART1,
        PART3A,
        PART5,
        EMBEDDING_GATE,
        CURRENT_DERIVATION_GATE,
        JEFF_NOTE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    expert_share_text = read_text(COMPUTATION_EXPERT_SHARE)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    jeff_note_text = read_text(JEFF_NOTE)

    embedding_summary = read_json(EMBEDDING_GATE)["summary"]
    current_derivation_summary = read_json(CURRENT_DERIVATION_GATE)["summary"]
    formulas = build_formulae()

    part1_micro_chiral_hit = hit(part1_text, "\\bar{\\psi}\\gamma^\\mu\\frac{1-\\gamma^5}{2}\\psi")
    part1_micro_pauli_hit = hit(part1_text, "\\bar{\\psi}\\sigma^{\\mu\\nu}\\psi")
    part3a_psi_bridge_hit = hit(part3a_text, "\\psi(x,t)")
    part3a_delta_p_hit = hit(part3a_text, "\\delta P_{+}(x,t)")
    part3a_noether_hit = hit(part3a_text, "\\partial_{\\mu}\\delta T_{(2)}^{\\mu 0}=0")
    part3a_scalar_limit_hit = hit(
        part3a_text,
        "未導出3：スピン/電荷/ゲージ場/相互作用項"
    )
    part3a_wavefunction_min_hit = hit(part3a_text, "P-model における波動関数の最小定義")
    jeff_step3_hit = hit(jeff_note_text, "### Step 3: J_eff^μ の構造を読む")
    jeff_case1_hit = hit(jeff_note_text, "Case I: J_eff⁰ ≈ |f₀|²")

    prior_embedding_ready = bool(
        embedding_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and embedding_summary.get("microscopic_functional_derivation_admissible_next", False)
        and not embedding_summary.get("nonzero_source_embedding_opened", True)
    )
    same_field_on_shell_zero_retained = bool(
        current_derivation_summary.get("same_field_on_shell_linear_source_zero", False)
    )

    psi_to_p_bridge_available = bool(
        part3a_psi_bridge_hit and part3a_delta_p_hit and part3a_noether_hit and part3a_wavefunction_min_hit
    )
    scalar_noether_envelope_bridge_available = psi_to_p_bridge_available
    microscopic_bilinear_surface_available = bool(part1_micro_chiral_hit and part1_micro_pauli_hit)
    spin_charge_gauge_interaction_derived = False
    scalar_to_spinor_identification_available = False
    microscopic_chiral_current_constitutive_map_available = False
    microscopic_pauli_tensor_constitutive_map_available = False
    microscopic_matter_current_functional_derived = False
    microscopic_rotational_source_functional_derived = False
    nonzero_source_embedding_opened = False

    scalar_bridge_real_channel_count = 2.0
    chiral_target_component_count = 4.0
    pauli_target_component_count = 6.0
    target_to_bridge_component_ratio = (
        (chiral_target_component_count + pauli_target_component_count)
        / scalar_bridge_real_channel_count
    )

    constitutive_map_reopen_required = bool(
        prior_embedding_ready
        and same_field_on_shell_zero_retained
        and psi_to_p_bridge_available
        and microscopic_bilinear_surface_available
        and not spin_charge_gauge_interaction_derived
        and not scalar_to_spinor_identification_available
        and not microscopic_chiral_current_constitutive_map_available
        and not microscopic_pauli_tensor_constitutive_map_available
        and not microscopic_matter_current_functional_derived
        and not microscopic_rotational_source_functional_derived
        and not nonzero_source_embedding_opened
    )
    effective_source_theorem_attempt_admissible_now = False
    observable_dictionary_gate_admissible_now = False

    rows = [
        row(
            "prior_embedding_ready",
            "pass" if prior_embedding_ready else "reject",
            "prior matter/rot embedding audit ready",
            truth(prior_embedding_ready),
            "This branch only starts after the pack has already been narrowed to microscopic functional derivation.",
        ),
        row(
            "same_field_on_shell_zero_retained",
            "pass" if same_field_on_shell_zero_retained else "reject",
            "same-field on-shell zero retained",
            truth(same_field_on_shell_zero_retained),
            "The same-field linear source is already zero, so any nonzero source must come from a new microscopic embedding.",
        ),
        row(
            "psi_to_p_bridge_available",
            "pass" if psi_to_p_bridge_available else "reject",
            "psi-to-P bridge available",
            truth(psi_to_p_bridge_available),
            "Part III-A does provide a minimal psi <-> P bridge based on the positive-frequency envelope and Noether energy flow.",
        ),
        row(
            "scalar_noether_envelope_bridge_available",
            "pass" if scalar_noether_envelope_bridge_available else "reject",
            "scalar Noether-envelope bridge available",
            truth(scalar_noether_envelope_bridge_available),
            "The bridge closes only as a scalar envelope / energy-density proxy, not yet as a microscopic spinor current.",
        ),
        row(
            "microscopic_bilinear_surface_available",
            "pass" if microscopic_bilinear_surface_available else "reject",
            "microscopic bilinear surface available",
            truth(microscopic_bilinear_surface_available),
            "Part I retains explicit chiral-current and Pauli-type bilinear templates in the microscopic coupling sector.",
        ),
        row(
            "spin_charge_gauge_interaction_derived",
            "pass" if spin_charge_gauge_interaction_derived else "reject",
            "spin/charge/gauge interaction derived in psi-to-P bridge",
            truth(spin_charge_gauge_interaction_derived),
            "Part III-A explicitly marks spin/charge/gauge/interactions as underived, so the bridge does not yet close the microscopic sector.",
        ),
        row(
            "scalar_to_spinor_identification_available",
            "pass" if scalar_to_spinor_identification_available else "reject",
            "scalar-envelope to Dirac-spinor identification available",
            truth(scalar_to_spinor_identification_available),
            "The current pack never identifies the Schr-envelope psi with the fermionic spinor psi appearing in the microscopic bilinears.",
        ),
        row(
            "microscopic_chiral_current_constitutive_map_available",
            "pass" if microscopic_chiral_current_constitutive_map_available else "reject",
            "microscopic chiral-current constitutive map available",
            truth(microscopic_chiral_current_constitutive_map_available),
            "No explicit map turns delta P_+/P_* or the restored exact vector branch into psi_bar gamma^mu (1-gamma^5) psi / 2.",
        ),
        row(
            "microscopic_pauli_tensor_constitutive_map_available",
            "pass" if microscopic_pauli_tensor_constitutive_map_available else "reject",
            "microscopic Pauli-tensor constitutive map available",
            truth(microscopic_pauli_tensor_constitutive_map_available),
            "No explicit map turns the restored exact vector branch into psi_bar sigma^{mu nu} psi either.",
        ),
        row(
            "scalar_bridge_real_channel_count",
            "pass",
            "scalar bridge real channel count proxy",
            scalar_bridge_real_channel_count,
            "The explicit psi <-> P bridge fixes amplitude and phase of a complex scalar envelope.",
        ),
        row(
            "microscopic_target_component_count",
            "pass",
            "microscopic target component count proxy",
            chiral_target_component_count + pauli_target_component_count,
            "The retained microscopic targets are a 4-current plus a 2-form tensor before any constitutive reduction.",
        ),
        row(
            "target_to_bridge_component_ratio",
            "pass",
            "target-to-bridge component ratio proxy",
            target_to_bridge_component_ratio,
            "This count proxy is not a theorem, but it shows why the scalar envelope bridge alone does not close the microscopic bilinear sector.",
        ),
        row(
            "microscopic_matter_current_functional_derived",
            "pass" if microscopic_matter_current_functional_derived else "reject",
            "microscopic matter-current functional derived",
            truth(microscopic_matter_current_functional_derived),
            "The branch fails to derive J_matter^mu[P^Qball] from the currently explicit scalar envelope bridge.",
        ),
        row(
            "microscopic_rotational_source_functional_derived",
            "pass" if microscopic_rotational_source_functional_derived else "reject",
            "microscopic rotational-source functional derived",
            truth(microscopic_rotational_source_functional_derived),
            "The branch also fails to derive the lambda_rot source functional on the restored exact vector / Q-ball branch.",
        ),
        row(
            "nonzero_source_embedding_opened",
            "pass" if nonzero_source_embedding_opened else "reject",
            "nonzero source embedding opened",
            truth(nonzero_source_embedding_opened),
            "Because the constitutive map is still absent, the current pack does not yet open a nonzero microscopic source embedding.",
        ),
        row(
            "constitutive_map_reopen_required",
            "pass" if constitutive_map_reopen_required else "reject",
            "constitutive-map reopen required",
            truth(constitutive_map_reopen_required),
            "The next honest lane is to reopen the scalar-to-spinor / bilinear constitutive map itself before retrying the source theorem.",
        ),
        row(
            "effective_source_theorem_attempt_admissible_now",
            "pass" if effective_source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            truth(effective_source_theorem_attempt_admissible_now),
            "Retrying the source theorem now would only repeat the already-localized constitutive-map gap.",
        ),
        row(
            "observable_dictionary_gate_admissible_now",
            "pass" if observable_dictionary_gate_admissible_now else "reject",
            "observable dictionary gate admissible now",
            truth(observable_dictionary_gate_admissible_now),
            "Observable-dictionary work remains downstream of an exact current closure and a successful source theorem.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "unified_roadmap_note": display_path(UNIFIED_ROADMAP),
            "computation_expert_share_note": display_path(COMPUTATION_EXPERT_SHARE),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "jeff_derivation_note": display_path(JEFF_NOTE),
        },
        "prior_metrics": {
            "embedding_gate": display_path(EMBEDDING_GATE),
            "current_derivation_gate": display_path(CURRENT_DERIVATION_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "psi_to_p_bridge_available": psi_to_p_bridge_available,
        "scalar_noether_envelope_bridge_available": scalar_noether_envelope_bridge_available,
        "spin_charge_gauge_interaction_derived": spin_charge_gauge_interaction_derived,
        "scalar_to_spinor_identification_available": scalar_to_spinor_identification_available,
        "microscopic_chiral_current_constitutive_map_available": microscopic_chiral_current_constitutive_map_available,
        "microscopic_pauli_tensor_constitutive_map_available": microscopic_pauli_tensor_constitutive_map_available,
        "microscopic_matter_current_functional_derived": microscopic_matter_current_functional_derived,
        "microscopic_rotational_source_functional_derived": microscopic_rotational_source_functional_derived,
        "nonzero_source_embedding_opened": nonzero_source_embedding_opened,
        "constitutive_map_reopen_required": constitutive_map_reopen_required,
        "effective_source_theorem_attempt_admissible_now": effective_source_theorem_attempt_admissible_now,
        "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
        "scalar_bridge_real_channel_count": scalar_bridge_real_channel_count,
        "microscopic_target_component_count": chiral_target_component_count + pauli_target_component_count,
        "target_to_bridge_component_ratio": target_to_bridge_component_ratio,
        "scalar_strong_candidate_retained": True,
        "blind_vector_no_go_retained": True,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "part_hits": {
            "part1_micro_chiral": part1_micro_chiral_hit,
            "part1_micro_pauli": part1_micro_pauli_hit,
            "part3a_psi_bridge": part3a_psi_bridge_hit,
            "part3a_delta_p": part3a_delta_p_hit,
            "part3a_noether": part3a_noether_hit,
            "part3a_wavefunction_min": part3a_wavefunction_min_hit,
            "part3a_underived_spin_charge_gauge": part3a_scalar_limit_hit,
            "jeff_step3": jeff_step3_hit,
            "jeff_case1": jeff_case1_hit,
        },
        "carry_over": {
            "embedding_summary": embedding_summary,
            "current_derivation_summary": current_derivation_summary,
        },
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": 0.2998913524347805,
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_F_at_q_theory": -0.083735013520183,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    inventory_paths = write_artifact(
        "inventory",
        payload(
            "8.7.56.1543",
            f"{STEP_NAME} inventory",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    audit_paths = write_artifact(
        "audit",
        payload(
            "8.7.56.1544",
            f"{STEP_NAME} audit",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1545",
            f"{STEP_NAME} declaration gate",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    route_paths = write_artifact(
        "route_sync",
        payload(
            "8.7.56.1546",
            f"{STEP_NAME} route sync",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )

    print("[ok] microscopic matter-current / rotational-source functional derivation artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
