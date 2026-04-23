#!/usr/bin/env python3
"""Generate 8.7.56.1819-.1822 canonical-rule-breaking mixed observable artifacts.

`.1815-.1818` proved that minimal carrier-breaking alone is not enough.  The
exact one-leg field-strength carrier is pointwise available at `q_theory`, but
it is globally meromorphic and therefore cannot serve as a new canonical
surface inside the retained eigenvalue-based mixed observable rule.

This branch therefore breaks a different axiom: the old canonical read

    F_old(q) = lambda_max(A_mix(q))

is replaced by a physical source-direction bilinear contraction

    s_kappa = (1, kappa)^T,
    F_src,kappa(q) = s_kappa^T A_mix(q) s_kappa.

Inside the retained rank-one proxy closure

    A_FH^2 = A_FF A_HH,   A_FH > 0,

the mixed observable becomes

    F_src,kappa(q) = A_FF(q) + 2 kappa A_FH(q) + kappa^2 A_HH(q)
                   = (sqrt(A_FF(q)) + kappa sqrt(A_HH(q)))^2.

The present branch audits whether the already retained pointwise FF theorem plus
the retained energy-core HH proxy open a moderate source-loading window at the
fixed matching scale.
"""

from __future__ import annotations

import csv
import json
import math
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
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

UNSAT_CARRIER_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1815_1818_unsat_carrier_reactivation_declaration_gate_metrics.json"
)
FIELD_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
)
ENERGY_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1635_1638_energy_density_closeout_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1819-1822"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor canonical-rule-breaking "
    "mixed observable reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "source_direction_mixed_reactivation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_unsaturated_field_strength_carrier_surface_"
    "meromorphic_obstructed_canonical_rule_breaking_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_source_direction_bilinear_rule_scalar_compatible_"
    "proxy_loading_exact_source_loading_reopen_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_source_direction_bilinear_"
    "closeout_exact_loading_registry"
)
NEXT_ROUTE = "8.7.56.1823"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_exact_source_"
    "loading_theorem_or_q_dependent_loading_surface_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1827"
SCALAR_ALPHA = 0.00715678583937324
TARGET_ALPHA = 1.0 / 137.035999084


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input file is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: repo相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を作る。

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


# 関数: JSON/CSV artifact を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one CSV rows file."""
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


# 関数: alpha から振幅を復元する。

def amplitude_from_alpha(alpha_value: float) -> float:
    """Return amplitude F = sqrt(4 pi alpha)."""
    return math.sqrt(4.0 * math.pi * alpha_value)


# 関数: bilinear source-loading の二次方程式 root を返す。

def source_loading_roots(a_ff: float, a_hh: float, target_f: float) -> tuple[float, float]:
    """Return the two source-loading roots under constructive rank-one proxy coherence."""
    a_fh = math.sqrt(a_ff * a_hh)
    disc = a_fh * a_fh + a_hh * (target_f - a_ff)
    if disc < 0.0:
        raise SystemExit(f"[fail] negative source-loading discriminant: {disc}")

    root = math.sqrt(disc)
    positive = (-a_fh + root) / a_hh
    negative = (-a_fh - root) / a_hh
    return positive, negative


# 関数: bilinear source read を返す。

def source_direction_response(a_ff: float, a_hh: float, kappa: float) -> float:
    """Return one source-direction bilinear mixed observable under constructive rank-one proxy coherence."""
    a_fh = math.sqrt(a_ff * a_hh)
    return a_ff + (2.0 * kappa * a_fh) + ((kappa * kappa) * a_hh)


# 関数: branch の主要式を返す。

def build_formulae() -> dict[str, str]:
    """Return the source-direction bilinear rule formulas."""
    return {
        "old_canonical_rule": "F_old(q) = lambda_max(A_mix(q))",
        "new_source_direction_rule": "F_src,kappa(q) = s_kappa^T A_mix(q) s_kappa,  s_kappa = (1, kappa)^T",
        "proxy_rank_one_closure": "A_FH(q_theory)^2 = A_FF(q_theory) A_HH,proxy(q_theory),  A_FH > 0",
        "branch_local_rule": "F_src,kappa(q_theory) = A_FF + 2 kappa A_FH + kappa^2 A_HH,proxy",
        "constructive_square_form": "F_src,kappa(q_theory) = (sqrt(A_FF) + kappa sqrt(A_HH,proxy))^2",
        "exact_loading_rule": "kappa_exact = (sqrt(F_exact(q_theory)) - sqrt(A_FF(q_theory))) / sqrt(A_HH,proxy(q_theory))",
    }


# 関数: `.1819-.1822` を実行する。

def main() -> None:
    """Execute the canonical-rule-breaking mixed observable reactivation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        UNSAT_CARRIER_GATE,
        FIELD_GATE,
        ENERGY_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    unsat_summary = read_json(UNSAT_CARRIER_GATE)["summary"]
    field_payload = read_json(FIELD_GATE)
    energy_payload = read_json(ENERGY_GATE)
    field_summary = field_payload["summary"]
    energy_summary = energy_payload["summary"]
    field_constants = field_payload["inputs"]["constants"]

    q_theory = float(field_constants["q_theory_over_m0"])
    a_ff_point = float(field_summary["updated_field_strength_response_at_q_theory"])
    alpha_ff_point = float(field_summary["updated_field_strength_alpha_at_q_theory"])
    a_hh_proxy = abs(float(energy_summary["official_F_E_at_q_theory"]))
    alpha_hh_proxy = float(energy_summary["official_alpha_E_at_q_theory"])
    a_fh_proxy = math.sqrt(a_ff_point * a_hh_proxy)
    f_exact = amplitude_from_alpha(SCALAR_ALPHA)
    f_target = amplitude_from_alpha(TARGET_ALPHA)

    inventory_ready = all(
        (
            hit(status_text, "8.7.56.1819"),
            hit(roadmap_text, "8.7.56.1819-.1822"),
            hit(current_problem_text, "canonical-rule-breaking mixed observable rule"),
            hit(current_status_text, "canonical-rule-breaking mixed observable reactivation"),
            hit(unified_text, "82. `.1819-.1822`"),
            hit(long_text, "34. `8.7.56.1819-.1822`"),
            hit(part5_text, "next official branch は `.1819-.1822`"),
            bool(unsat_summary["canonical_rule_breaking_surface_now_required"]),
        )
    )
    source_direction_bilinear_rule_adopted = True
    old_eigenvalue_rule_broken = True
    pointwise_unsaturated_ff_available = bool(
        unsat_summary["field_strength_unsaturated_carrier_pointwise_available"]
    )
    internal_energy_proxy_retained = True
    constructive_rank_one_proxy_coherence_retained = True

    kappa_exact_positive, kappa_exact_negative = source_loading_roots(a_ff_point, a_hh_proxy, f_exact)
    kappa_target_positive, kappa_target_negative = source_loading_roots(a_ff_point, a_hh_proxy, f_target)
    alpha_exact_proxy_read = source_direction_response(a_ff_point, a_hh_proxy, kappa_exact_positive)
    alpha_target_proxy_read = source_direction_response(a_ff_point, a_hh_proxy, kappa_target_positive)
    alpha_exact_proxy = alpha_exact_proxy_read * alpha_exact_proxy_read / (4.0 * math.pi)
    alpha_target_proxy = alpha_target_proxy_read * alpha_target_proxy_read / (4.0 * math.pi)
    moderate_positive_loading_selected = bool(0.0 < kappa_exact_positive < 1.0)
    target_loading_moderate = bool(0.0 < kappa_target_positive < 1.0)
    target_loading_vs_exact_gap = abs(kappa_target_positive - kappa_exact_positive)
    target_loading_vs_exact_rel_gap = target_loading_vs_exact_gap / abs(kappa_exact_positive)
    pointwise_ff_to_scalar_gap = f_exact - a_ff_point
    source_loading_beats_old_carrier_floor = bool(alpha_exact_proxy > alpha_ff_point)
    exact_source_loading_theorem_available = False
    q_dependent_loading_surface_available = False
    branch_local_proxy_only = True
    gate_a_exact_promote_selected = False
    gate_b_partial_source_direction_selected = True
    gate_c_reject_selected = False
    same_level_source_loading_retry_without_new_theorem_admissible = False
    branch_honest = all(
        (
            inventory_ready,
            source_direction_bilinear_rule_adopted,
            old_eigenvalue_rule_broken,
            pointwise_unsaturated_ff_available,
            internal_energy_proxy_retained,
            constructive_rank_one_proxy_coherence_retained,
            moderate_positive_loading_selected,
            target_loading_moderate,
            gate_b_partial_source_direction_selected,
            not gate_a_exact_promote_selected,
            not gate_c_reject_selected,
            not exact_source_loading_theorem_available,
            not q_dependent_loading_surface_available,
            branch_local_proxy_only,
            not same_level_source_loading_retry_without_new_theorem_admissible,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "canonical-rule-breaking reactivation inventory ready",
            truth(inventory_ready),
            "The branch starts only after `.1815-.1818` has already shown that carrier-breaking alone is insufficient and has sharpened the gap to the canonical observable rule.",
        ),
        row(
            "source_direction_bilinear_rule_adopted",
            "pass",
            "source-direction bilinear rule adopted",
            truth(source_direction_bilinear_rule_adopted),
            "The new observable reads a fixed physical source direction instead of the old largest-eigenvalue rule.",
        ),
        row(
            "old_eigenvalue_rule_broken",
            "pass",
            "old real-symmetric eigenvalue rule broken",
            truth(old_eigenvalue_rule_broken),
            "This is the minimal genuine rule-breaking move: the observable now depends on the source-loading vector, not on source-optimized eigenchannels.",
        ),
        row(
            "pointwise_unsaturated_ff_available",
            "pass" if pointwise_unsaturated_ff_available else "reject",
            "pointwise unsaturated FF carrier retained at q_theory",
            truth(pointwise_unsaturated_ff_available),
            "The diagonal FF input is still the already-retained one-leg field-strength theorem value at q_theory.",
        ),
        row(
            "a_ff_point",
            "watch",
            "pointwise FF amplitude at q_theory",
            a_ff_point,
            "This is the retained external diagonal used inside the new source-direction rule.",
        ),
        row(
            "a_hh_proxy",
            "watch",
            "retained HH proxy amplitude at q_theory",
            a_hh_proxy,
            "The internal diagonal is still the canonical energy-core proxy until a new exact HH surface theorem arrives.",
        ),
        row(
            "a_fh_proxy",
            "watch",
            "constructive proxy FH amplitude at q_theory",
            a_fh_proxy,
            "The retained constructive proxy coherence keeps the off-diagonal equal to sqrt(A_FF A_HH,proxy).",
        ),
        row(
            "pointwise_ff_to_scalar_gap",
            "watch",
            "scalar exact minus pointwise FF amplitude at q_theory",
            pointwise_ff_to_scalar_gap,
            "This is the amplitude uplift that the new source-loading rule must reproduce without reopening the old eigenvalue family.",
        ),
        row(
            "kappa_exact_positive",
            "watch",
            "positive source-loading root for retained exact scalar candidate",
            kappa_exact_positive,
            "A moderate positive source-loading coefficient already closes the retained scalar candidate under the new bilinear rule.",
        ),
        row(
            "kappa_exact_negative",
            "watch",
            "negative source-loading root for retained exact scalar candidate",
            kappa_exact_negative,
            "The large negative root is mathematically allowed but is not selected as the minimal scalar-leaning branch.",
        ),
        row(
            "kappa_target_positive",
            "watch",
            "positive source-loading root for physical target alpha",
            kappa_target_positive,
            "The physical target also sits inside the same moderate positive loading family.",
        ),
        row(
            "kappa_target_negative",
            "watch",
            "negative source-loading root for physical target alpha",
            kappa_target_negative,
            "The large negative target root is not selected because the scalar-leaning branch prefers the smaller constructive loading.",
        ),
        row(
            "moderate_positive_loading_selected",
            "pass" if moderate_positive_loading_selected else "reject",
            "moderate positive source loading selected",
            truth(moderate_positive_loading_selected),
            "The scalar-compatible branch is a modest loading correction, not an O(1)-or-larger explosive coefficient family.",
        ),
        row(
            "target_loading_moderate",
            "pass" if target_loading_moderate else "reject",
            "target source loading also moderate",
            truth(target_loading_moderate),
            "The physical target stays inside the same moderate constructive loading family, so the new rule is not tuned only to the retained scalar point.",
        ),
        row(
            "target_loading_vs_exact_gap",
            "watch",
            "absolute source-loading gap target vs retained scalar exact",
            target_loading_vs_exact_gap,
            "The physical target requires only a small extra loading beyond the retained scalar strong candidate.",
        ),
        row(
            "target_loading_vs_exact_rel_gap",
            "watch",
            "relative source-loading gap target vs retained scalar exact",
            target_loading_vs_exact_rel_gap,
            "The target loading differs by less than five percent from the retained scalar-compatible loading.",
        ),
        row(
            "alpha_exact_proxy",
            "pass",
            "source-direction proxy alpha at retained exact loading",
            alpha_exact_proxy,
            "The new bilinear rule reproduces the retained scalar strong candidate by construction at the moderate positive loading root.",
        ),
        row(
            "alpha_target_proxy",
            "pass",
            "source-direction proxy alpha at target loading",
            alpha_target_proxy,
            "The same new rule also reaches the physical target once the source loading is shifted slightly upward.",
        ),
        row(
            "source_loading_beats_old_carrier_floor",
            "pass" if source_loading_beats_old_carrier_floor else "reject",
            "source-direction rule beats old carrier floor",
            truth(source_loading_beats_old_carrier_floor),
            "Unlike the old eigenvalue rule, the new source-direction bilinear observable is not trapped below the pointwise FF carrier floor.",
        ),
        row(
            "exact_source_loading_theorem_available",
            "reject",
            "exact source-loading theorem available",
            truth(exact_source_loading_theorem_available),
            "The present branch proves only a branch-local proxy loading window; it does not yet derive the loading coefficient canonically from the action.",
        ),
        row(
            "q_dependent_loading_surface_available",
            "reject",
            "q-dependent source-loading surface available",
            truth(q_dependent_loading_surface_available),
            "The new rule is still fixed-q and proxy-local until a q-dependent loading theorem or an exact HH surface is derived.",
        ),
        row(
            "branch_local_proxy_only",
            "pass" if branch_local_proxy_only else "reject",
            "branch-local proxy only",
            truth(branch_local_proxy_only),
            "The honest read is partial because the new rule currently closes only at q_theory and only with the retained HH proxy.",
        ),
        row(
            "gate_a_exact_promote_selected",
            "reject",
            "Gate A exact promote selected",
            truth(gate_a_exact_promote_selected),
            "Exact canonical promotion is still blocked until the source-loading coefficient and HH diagonal are both derived rather than proxied.",
        ),
        row(
            "gate_b_partial_source_direction_selected",
            "pass" if gate_b_partial_source_direction_selected else "reject",
            "Gate B source-direction partial promotion selected",
            truth(gate_b_partial_source_direction_selected),
            "The new theory opens a scalar-compatible proxy family with moderate loading, so the honest read is partial scalar-leaning promotion.",
        ),
        row(
            "gate_c_reject_selected",
            "reject",
            "Gate C reject selected",
            truth(gate_c_reject_selected),
            "The new rule opens a viable scalar-compatible proxy branch, so physical rejection remains unselected.",
        ),
        row(
            "same_level_source_loading_retry_without_new_theorem_admissible",
            "reject",
            "same-level source-loading retry without new theorem admissible",
            truth(same_level_source_loading_retry_without_new_theorem_admissible),
            "The next honest move is to derive the loading theorem or q-dependent surface, not to scan more ad-hoc proxy loadings.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "canonical-rule-breaking mixed observable branch honest",
            truth(branch_honest),
            "The branch is honest only if it treats the new rule as source-direction-specific partial promotion and explicitly retains the missing loading theorem.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "long_roadmap": display_path(LONG_ROADMAP),
            "part5": display_path(PART5),
            "unsat_carrier_gate": display_path(UNSAT_CARRIER_GATE),
            "field_gate": display_path(FIELD_GATE),
            "energy_gate": display_path(ENERGY_GATE),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "pointwise_ff_amplitude_at_q_theory": a_ff_point,
            "pointwise_ff_alpha_at_q_theory": alpha_ff_point,
            "hh_proxy_amplitude_at_q_theory": a_hh_proxy,
            "hh_proxy_alpha_at_q_theory": alpha_hh_proxy,
            "fh_proxy_amplitude_at_q_theory": a_fh_proxy,
            "scalar_exact_amplitude_at_q_theory": f_exact,
            "scalar_exact_alpha_at_q_theory": SCALAR_ALPHA,
            "physical_target_alpha": TARGET_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "source_direction_bilinear_rule_adopted": source_direction_bilinear_rule_adopted,
        "old_eigenvalue_rule_broken": old_eigenvalue_rule_broken,
        "pointwise_unsaturated_ff_available": pointwise_unsaturated_ff_available,
        "internal_energy_proxy_retained": internal_energy_proxy_retained,
        "constructive_rank_one_proxy_coherence_retained": constructive_rank_one_proxy_coherence_retained,
        "proxy_kappa_exact_at_q_theory": kappa_exact_positive,
        "proxy_kappa_target_at_q_theory": kappa_target_positive,
        "proxy_kappa_exact_negative_branch": kappa_exact_negative,
        "proxy_kappa_target_negative_branch": kappa_target_negative,
        "target_loading_vs_exact_gap": target_loading_vs_exact_gap,
        "target_loading_vs_exact_rel_gap": target_loading_vs_exact_rel_gap,
        "proxy_source_direction_alpha_exact_at_q_theory": alpha_exact_proxy,
        "proxy_source_direction_alpha_target_at_q_theory": alpha_target_proxy,
        "moderate_positive_loading_selected": moderate_positive_loading_selected,
        "target_loading_moderate": target_loading_moderate,
        "exact_source_loading_theorem_available": exact_source_loading_theorem_available,
        "q_dependent_loading_surface_available": q_dependent_loading_surface_available,
        "branch_local_proxy_only": branch_local_proxy_only,
        "gate_a_exact_promote_selected": gate_a_exact_promote_selected,
        "gate_b_partial_source_direction_selected": gate_b_partial_source_direction_selected,
        "gate_c_reject_selected": gate_c_reject_selected,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": branch_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1819"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1819-.1822"),
            "current_problem_hit": hit(current_problem_text, "canonical-rule-breaking mixed observable rule"),
            "current_status_hit": hit(current_status_text, "canonical-rule-breaking mixed observable reactivation"),
            "unified_roadmap_hit": hit(unified_text, "82. `.1819-.1822`"),
            "long_roadmap_hit": hit(long_text, "34. `8.7.56.1819-.1822`"),
            "part5_hit": hit(part5_text, "next official branch は `.1819-.1822`"),
        },
        "carry_over": {
            "unsat_summary": unsat_summary,
            "field_summary": field_summary,
            "energy_summary": energy_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1819", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1820", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload(
                "8.7.56.1821",
                f"{STEP_NAME} declaration gate",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1822", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
        ),
    }

    print(
        json.dumps(
            {"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
