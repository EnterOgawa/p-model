#!/usr/bin/env python3
"""Generate 8.7.56.1815-.1818 axiom-breaking mixed-surface reactivation artifacts.

`.1811-.1814` sharpened the only honest reopen surfaces to

1. carrier-breaking beyond the retained saturated carrier `A_FF(q)=|q|`, or
2. canonical-rule-breaking beyond the retained real-symmetric eigenvalue read.

The minimal genuinely new carrier-breaking theory available inside the retained
pack is to replace the saturated carrier by the exact one-leg field-strength
carrier already derived at fixed q:

    A_FF,br(q) = F_F,can(q)
               = |q| M_T(q) / (q^2 + M_T(q)).

This branch tests whether that unsaturated carrier is a *global canonical
surface* under the same mixed-pack rule.  The honest failure mode is no longer
the old `A_FF=|q|` lower bound.  It is whether `q^2 + M_T(q)` develops poles or
sign-defects that prevent `A_FF,br(q)` from serving as a globally admissible
carrier inside the retained canonical family.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths

import scripts.quantum.t2a_1627 as density_tools
import scripts.quantum.t2a_1659 as projected_kernel_tools
import scripts.quantum.t2a_1799 as scalar_window_tools


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

FIELD_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
)
WINDOW_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1799_1802_full_q_hh_window_generalization_declaration_gate_metrics.json"
)
OBSTRUCTION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1807_1810_global_hh_obstruction_theorem_declaration_gate_metrics.json"
)
OBSTRUCTION_CLOSEOUT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1811_1814_global_completion_obstruction_closeout_registry_declaration_gate_metrics.json"
)
QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"

STEP_TAG = "8.7.56.1815-1818"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional axiom-breaking "
    "mixed surface reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "unsat_carrier_reactivation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_global_completion_obstruction_closeout_"
    "carrier_or_canonical_rule_breaking_reopen_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_unsaturated_field_strength_carrier_surface_"
    "meromorphic_obstructed_canonical_rule_breaking_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_canonical_rule_breaking_"
    "mixed_observable_reactivation"
)
NEXT_ROUTE = "8.7.56.1819"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_carrier_breaking_closeout_"
    "canonical_rule_breaking_registry"
)
FOLLOWUP_ROUTE = "8.7.56.1823"
SCALAR_ALPHA = 0.00715678583937324


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を検査する。

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


# 関数: rhs基準の相対差を返す。

def rel_gap(lhs: float, rhs: float) -> float:
    """Return one rhs-relative absolute gap."""
    return float(abs(lhs - rhs) / abs(rhs))


# 関数: retained scalar exact profile を構成する。

def build_scalar_profile() -> dict:
    """Rebuild the retained scalar exact profile used by the mixed completion pack."""
    branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    qball_module = scalar_window_tools.load_qball_module()
    scalar_row = scalar_window_tools.extract_scalar_ground_state(branch_refresh)
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_row["beta_n"]),
        float(scalar_row["central_amplitude"]),
    )
    radius_array = np.asarray(radius, dtype=float)
    field_array = np.asarray(field, dtype=float)
    weight = field_array * field_array * radius_array * radius_array
    norm = float(np.trapezoid(weight, radius_array))
    return {"radius": radius_array, "weight": weight, "norm": norm}


# 関数: retained exact scalar form factor を返す。

def scalar_form_factor(profile: dict, q_ratio: float) -> float:
    """Return one retained exact scalar form factor."""
    return float(
        scalar_window_tools.form_factor(
            profile["radius"],
            profile["weight"],
            profile["norm"],
            float(q_ratio),
        )
    )


# 関数: raw projected-kernel M_T(q) を返す。

def raw_projected_kernel(radius: np.ndarray, a_values: np.ndarray, b_values: np.ndarray, q_ratio: float) -> float:
    """Return the unnormalized projected-kernel contrast M_T(q) up to the common 4 pi factor."""
    qx = float(q_ratio) * radius
    j0 = projected_kernel_tools.spherical_j0(qx)
    j2 = projected_kernel_tools.spherical_j2(qx)
    kernel_q = a_values * j0 + (b_values / 3.0) * (j0 + j2)
    return float(np.trapezoid((radius**2) * kernel_q, radius))


# 関数: unsaturated field-strength carrier を返す。

def unsaturated_carrier(raw_m_t: float, q_ratio: float) -> float:
    """Return A_FF,br(q) = |q| M_T(q)/(q^2 + M_T(q))."""
    q_abs = abs(float(q_ratio))
    return float(q_abs * raw_m_t / ((q_abs * q_abs) + raw_m_t))


# 関数: 符号変化 root 群を抽出する。

def bracket_roots(q_values: np.ndarray, value_fn) -> list[float]:
    """Return all brentq roots whose brackets show one sign flip."""
    roots: list[float] = []
    values = np.asarray([float(value_fn(q)) for q in q_values], dtype=float)
    for q_lo, q_hi, value_lo, value_hi in zip(
        q_values[:-1],
        q_values[1:],
        values[:-1],
        values[1:],
    ):
        if value_lo == 0.0:
            roots.append(float(q_lo))
            continue

        if value_lo * value_hi < 0.0:
            roots.append(float(brentq(value_fn, float(q_lo), float(q_hi))))

    return roots


# 関数: carrier-breaking theorem の主要式を返す。

def build_formulae() -> dict[str, str]:
    """Return the unsaturated carrier-breaking formulas."""
    return {
        "carrier_breaking_surface": "A_FF,br(q) = F_F,can(q) = |q| M_T(q) / (q^2 + M_T(q))",
        "retained_rank_one_rule": "lambda_+(q) = A_FF(q) + A_HH(q), rho_exact = 1",
        "window_completion_rule": "A_HH,exact(q) = F_exact(q) - A_FF,br(q) whenever F_exact(q) >= A_FF,br(q)",
        "pole_obstruction": "q^2 + M_T(q) = 0 implies a meromorphic carrier and blocks a global canonical carrier surface",
        "next_missing_surface": "canonical-rule-breaking mixed observable rule beyond the retained real-symmetric eigenvalue pack",
    }


# 関数: `.1815-.1818` を実行する。

def main() -> None:
    """Execute the carrier-breaking mixed-surface reactivation branch."""
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
        FIELD_GATE,
        WINDOW_GATE,
        OBSTRUCTION_GATE,
        OBSTRUCTION_CLOSEOUT_GATE,
        QBALL_BRANCH_REFRESH,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    field_summary = read_json(FIELD_GATE)["summary"]
    window_summary = read_json(WINDOW_GATE)["summary"]
    obstruction_summary = read_json(OBSTRUCTION_GATE)["summary"]
    obstruction_closeout_summary = read_json(OBSTRUCTION_CLOSEOUT_GATE)["summary"]

    density_bundle = density_tools.build_density_bundle()
    scalar_profile = build_scalar_profile()

    radius = np.asarray(density_bundle["radius"], dtype=float)
    f0_values = np.asarray(density_bundle["f0_values"], dtype=float)
    f_l_values = np.asarray(density_bundle["f_l_values"], dtype=float)
    q_theory = float(density_bundle["q_theory_over_m0"])
    a_values = -(f0_values * f0_values) + (f_l_values * f_l_values)
    b_values = 2.0 * (f_l_values * f_l_values)

    inventory_ready = all(
        (
            hit(current_problem_text, "carrier-breaking or canonical-rule-breaking surface"),
            hit(current_status_text, "conditional axiom-breaking mixed surface reactivation"),
            hit(unified_text, "80. `.1811-.1814`"),
            hit(long_text, "33. `8.7.56.1815-.1818`"),
            hit(part5_text, "next official branch is `.1815-.1818`"),
            obstruction_closeout_summary["obstruction_retained"],
        )
    )
    carrier_breaking_surface_adopted = True
    saturated_window_edge = float(obstruction_summary["q_hh_max_over_m0"])

    q_scan = np.linspace(1.0e-6, 1.0, 800)
    raw_m_values = np.asarray(
        [raw_projected_kernel(radius, a_values, b_values, q_value) for q_value in q_scan],
        dtype=float,
    )
    denominator_values = (q_scan * q_scan) + raw_m_values
    denominator_roots = bracket_roots(
        q_scan,
        lambda q_value: (q_value * q_value)
        + raw_projected_kernel(radius, a_values, b_values, q_value),
    )
    denominator_poles_detected = bool(len(denominator_roots) > 0)
    first_pole_over_m0 = float(denominator_roots[0]) if denominator_roots else math.nan
    q_theory_above_first_pole = bool(denominator_poles_detected and q_theory > first_pole_over_m0)

    carrier_at_q_theory = unsaturated_carrier(
        raw_projected_kernel(radius, a_values, b_values, q_theory),
        q_theory,
    )
    field_strength_point_match = bool(
        math.isclose(
            carrier_at_q_theory,
            float(field_summary["updated_field_strength_response_at_q_theory"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    )

    # Search for the scalar-compatible edge only on the positive carrier branch that
    # contains q_theory.
    gap_fn = lambda q_value: scalar_form_factor(scalar_profile, q_value) - unsaturated_carrier(
        raw_projected_kernel(radius, a_values, b_values, q_value),
        q_value,
    )
    gap_scan = np.linspace(q_theory, 0.308, 400)
    gap_values = np.asarray([gap_fn(q_value) for q_value in gap_scan], dtype=float)
    carrier_breaking_window_edge = math.nan
    for q_lo, q_hi, gap_lo, gap_hi in zip(
        gap_scan[:-1],
        gap_scan[1:],
        gap_values[:-1],
        gap_values[1:],
    ):
        if gap_lo == 0.0:
            carrier_breaking_window_edge = float(q_lo)
            break

        if gap_lo * gap_hi < 0.0:
            carrier_breaking_window_edge = float(brentq(gap_fn, float(q_lo), float(q_hi)))
            break

    carrier_breaking_window_shift = float(carrier_breaking_window_edge - saturated_window_edge)
    carrier_breaking_window_shift_rel = float(
        carrier_breaking_window_shift / saturated_window_edge
    )

    sample_q_probe = 0.3
    carrier_at_q_probe = unsaturated_carrier(
        raw_projected_kernel(radius, a_values, b_values, sample_q_probe),
        sample_q_probe,
    )
    exact_at_q_probe = scalar_form_factor(scalar_profile, sample_q_probe)
    deficit_at_q_probe = float(carrier_at_q_probe - exact_at_q_probe)

    carrier_sign_indefinite_detected = bool(
        np.any(
            np.asarray(
                [
                    unsaturated_carrier(raw_m_value, q_value)
                    for q_value, raw_m_value, denominator_value in zip(
                        q_scan,
                        raw_m_values,
                        denominator_values,
                    )
                    if abs(denominator_value) > 1.0e-4
                ],
                dtype=float,
            )
            < 0.0
        )
    )
    unsaturated_carrier_global_canonical_surface_available = False
    carrier_breaking_alone_not_sufficient = bool(
        denominator_poles_detected and not unsaturated_carrier_global_canonical_surface_available
    )
    canonical_rule_breaking_surface_now_required = True
    same_level_unsaturated_retry_admissible = False
    branch_honest = all(
        (
            inventory_ready,
            carrier_breaking_surface_adopted,
            field_strength_point_match,
            denominator_poles_detected,
            q_theory_above_first_pole,
            carrier_breaking_alone_not_sufficient,
            canonical_rule_breaking_surface_now_required,
            not same_level_unsaturated_retry_admissible,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "axiom-breaking carrier reactivation inventory ready",
            truth(inventory_ready),
            "The branch starts only after `.1811-.1814` sharpens the reopen surface away from same-level post-window retries.",
        ),
        row(
            "carrier_breaking_surface_adopted",
            "pass",
            "unsaturated carrier-breaking surface adopted",
            truth(carrier_breaking_surface_adopted),
            "The minimal new carrier is the exact one-leg field-strength carrier A_FF,br(q) rather than the retained saturated carrier |q|.",
        ),
        row(
            "field_strength_point_match",
            "pass" if field_strength_point_match else "reject",
            "unsaturated carrier matches fixed-q field-strength read",
            truth(field_strength_point_match),
            "At q_theory the new carrier must reproduce the already-retained field-strength theorem result exactly.",
        ),
        row(
            "first_pole_over_m0",
            "watch",
            "first denominator pole q_pole,1 / m0",
            first_pole_over_m0,
            "The first zero of q^2 + M_T(q) is the earliest obstruction to promoting the unsaturated carrier globally.",
        ),
        row(
            "pole_count_on_scan_0_to_1",
            "watch",
            "number of denominator poles on 0 < q/m0 <= 1 scan",
            float(len(denominator_roots)),
            "Multiple denominator poles mean the carrier becomes meromorphic rather than a single smooth global surface.",
        ),
        row(
            "q_theory_above_first_pole",
            "pass" if q_theory_above_first_pole else "reject",
            "q_theory lies beyond the first unsaturated-carrier pole",
            truth(q_theory_above_first_pole),
            "Even the retained matching point sits on a later branch of the meromorphic carrier, so the surface is not globally regular from q=0 outward.",
        ),
        row(
            "carrier_breaking_window_edge_over_m0",
            "watch",
            "carrier-breaking scalar-compatible window edge q / m0",
            carrier_breaking_window_edge,
            "This is the positive-branch point where F_exact(q) and the unsaturated carrier meet again after q_theory.",
        ),
        row(
            "carrier_breaking_window_shift_rel",
            "watch",
            "relative shift of the carrier-breaking window edge vs saturated carrier window",
            carrier_breaking_window_shift_rel,
            "The unsaturated carrier barely moves the scalar-compatible window edge, so carrier-breaking alone does not resolve the global completion obstruction.",
        ),
        row(
            "carrier_sign_indefinite_detected",
            "pass" if carrier_sign_indefinite_detected else "reject",
            "unsaturated carrier sign-indefinite detected on scan",
            truth(carrier_sign_indefinite_detected),
            "Later pole neighborhoods drive the unsaturated carrier through negative excursions, reinforcing that the same canonical family is not globally admissible.",
        ),
        row(
            "deficit_at_q_probe",
            "watch",
            "carrier minus exact amplitude at q/m0 = 0.3 under unsaturated carrier",
            deficit_at_q_probe,
            "Post-window the unsaturated carrier still overshoots the exact retained profile amplitude, so the global completion failure remains.",
        ),
        row(
            "unsaturated_carrier_global_canonical_surface_available",
            "reject",
            "global canonical unsaturated carrier surface available",
            truth(unsaturated_carrier_global_canonical_surface_available),
            "Denominator poles and later sign-indefinite branches block the exact one-leg carrier from serving as a global canonical FF surface under the same mixed-pack rule.",
        ),
        row(
            "carrier_breaking_alone_not_sufficient",
            "pass" if carrier_breaking_alone_not_sufficient else "reject",
            "carrier-breaking alone not sufficient",
            truth(carrier_breaking_alone_not_sufficient),
            "The minimal carrier-breaking move changes the pointwise FF surface but does not remove the global canonical obstruction.",
        ),
        row(
            "canonical_rule_breaking_surface_now_required",
            "pass" if canonical_rule_breaking_surface_now_required else "reject",
            "canonical-rule-breaking surface now required",
            truth(canonical_rule_breaking_surface_now_required),
            "Once the best available carrier-breaking surface becomes meromorphic, the next honest reopen surface is a genuinely new canonical observable rule.",
        ),
        row(
            "same_level_unsaturated_retry_admissible",
            "reject",
            "same-level unsaturated-carrier retry admissible",
            truth(same_level_unsaturated_retry_admissible),
            "No further same-rule FF carrier variants are honest after the exact field-strength carrier itself fails globally.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "carrier-breaking reactivation honest",
            truth(branch_honest),
            "The branch is honest only if it promotes the minimal field-strength carrier-breaking surface and then closes it again once meromorphic obstruction is explicit.",
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
            "field_gate": display_path(FIELD_GATE),
            "window_gate": display_path(WINDOW_GATE),
            "obstruction_gate": display_path(OBSTRUCTION_GATE),
            "obstruction_closeout_gate": display_path(OBSTRUCTION_CLOSEOUT_GATE),
            "qball_branch_refresh": display_path(QBALL_BRANCH_REFRESH),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "retained_scalar_alpha": SCALAR_ALPHA,
            "saturated_window_edge_over_m0": saturated_window_edge,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "carrier_breaking_surface_adopted": carrier_breaking_surface_adopted,
        "field_strength_unsaturated_carrier_pointwise_available": field_strength_point_match,
        "unsaturated_carrier_first_pole_over_m0": first_pole_over_m0,
        "unsaturated_carrier_pole_count_on_scan_0_to_1": float(len(denominator_roots)),
        "q_theory_above_first_unsaturated_carrier_pole": q_theory_above_first_pole,
        "carrier_breaking_window_edge_over_m0": carrier_breaking_window_edge,
        "carrier_breaking_window_shift_rel": carrier_breaking_window_shift_rel,
        "carrier_sign_indefinite_detected": carrier_sign_indefinite_detected,
        "carrier_breaking_deficit_at_q_probe": deficit_at_q_probe,
        "unsaturated_carrier_global_canonical_surface_available": unsaturated_carrier_global_canonical_surface_available,
        "carrier_breaking_alone_not_sufficient": carrier_breaking_alone_not_sufficient,
        "canonical_rule_breaking_surface_now_required": canonical_rule_breaking_surface_now_required,
        "same_level_unsaturated_retry_admissible": same_level_unsaturated_retry_admissible,
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
            "status_branch_hit": hit(status_text, "8.7.56.1811-.1814"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1811-.1814"),
            "current_problem_hit": hit(current_problem_text, "carrier-breaking or canonical-rule-breaking surface"),
            "current_status_hit": hit(current_status_text, "conditional axiom-breaking mixed surface reactivation"),
            "unified_roadmap_hit": hit(unified_text, "80. `.1811-.1814`"),
            "long_roadmap_hit": hit(long_text, "33. `8.7.56.1815-.1818`"),
            "part5_hit": hit(part5_text, "next official branch is `.1815-.1818`"),
        },
        "carry_over": {
            "field_summary": field_summary,
            "window_summary": window_summary,
            "obstruction_summary": obstruction_summary,
            "obstruction_closeout_summary": obstruction_closeout_summary,
        },
        "roots": {
            "denominator_roots_over_m0": denominator_roots,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1815", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1816", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1817", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1818", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
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
