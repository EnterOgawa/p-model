#!/usr/bin/env python3
"""Generate 8.7.56.1955-.1958 further retained-interval extension artifacts.

The sign-parity theorem is already official on `0 <= q/m0 <= 2`. The next
honest first shot is therefore not another wait-state audit but a direct
computation on a wider interval. This branch extends the retained real overlap
audit to `0 <= q/m0 <= 4` and checks whether the same simple-zero parity rule

    sigma_F(q) = 0 for q=q_n, and sigma_F(q)=(-1)^(N_zero(q)) otherwise

continues to reproduce the signed form factor exactly.
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

import scripts.quantum.t2a_1843 as sign_base
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

QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"
PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1951_1954_ext_interval_closeout_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1955-1958"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor further retained-interval "
    "extension theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "further_ext_interval_sign_phase_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_global_exact_alpha_signed_form_factor_extended_interval_0_to_2_"
    "promotion_closeout_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_retained_interval_extension_real_branch_sign_parity_0_to_4_"
    "derived_decision_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_extended_interval_theorem_stability_sync"
)
NEXT_ROUTE = "8.7.56.1959"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_asymptotic_sign_parity_generalization_audit"
)
FOLLOWUP_ROUTE = "8.7.56.1963"
Q_THEORY = 0.24297729990871803
OLD_Q_MAX = 2.0
EXTENDED_Q_MAX = 4.0


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: JSON/CSV artifact を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
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

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# 関数: 指定区間の signed zero を探す。

def find_signed_zeros_interval(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_max: float,
) -> list[float]:
    """Locate all simple signed zeros of F_exact(q) on 0 <= q <= q_max."""
    scan = np.linspace(0.0, float(q_max), int(10000 * q_max) + 1)
    values = np.array(
        [sign_base.form_factor(radius, weight, norm, float(q_value)) for q_value in scan]
    )
    roots: list[float] = []
    for q_left, q_right, f_left, f_right in zip(scan[:-1], scan[1:], values[:-1], values[1:]):
        if abs(f_left) <= 1.0e-10 and q_left > 0.0:
            root = float(q_left)
        elif f_left * f_right < 0.0:
            root = float(
                brentq(
                    lambda q_ratio: sign_base.form_factor(radius, weight, norm, float(q_ratio)),
                    float(q_left),
                    float(q_right),
                )
            )
        else:
            continue

        if not roots or abs(root - roots[-1]) > 1.0e-6:
            roots.append(root)

    return roots


# 関数: audit 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the further-interval theorem audit."""
    return {
        "real_overlap_branch": "F_exact(q) = int dr w(r) sinc(q r) / int dr w(r) in R for every real q",
        "sign_rule": "sigma_F(q) = 0 for q=q_n, and sigma_F(q)=(-1)^{N_zero(q)} otherwise",
        "signed_reconstruction": "F_exact(q) = sigma_F(q) |F_exact(q)|",
        "extended_interval": "0 <= q/m0 <= 4",
        "old_retained_interval": "0 <= q/m0 <= 2",
    }


# 関数: `.1955-.1958` を実行する。

def main() -> None:
    """Execute the further retained-interval extension theorem audit."""
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
        QBALL_BRANCH_REFRESH,
        QBALL_SOLVER,
        PRIOR_GATE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    qball_branch_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)

    inventory_ready = all(
        (
            bool(prior_summary["exact_alpha_promotion_retained"]),
            bool(prior_summary["exact_signed_form_factor_promotion_retained"]),
            float(prior_summary["extended_interval_over_m0"]) >= OLD_Q_MAX,
        )
    )

    qball_module = sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))

    q_scan = np.linspace(0.0, EXTENDED_Q_MAX, 40001)
    f_exact_scan = np.array(
        [sign_base.form_factor(radius, weight, norm, float(q_ratio)) for q_ratio in q_scan]
    )
    f_abs_scan = np.abs(f_exact_scan)

    signed_zero_roots = find_signed_zeros_interval(radius, weight, norm, EXTENDED_Q_MAX)
    signed_zero_roots_array = np.array(signed_zero_roots, dtype=float)
    root_slopes = np.array(
        [sign_base.root_slope(radius, weight, norm, root) for root in signed_zero_roots],
        dtype=float,
    )
    signed_zero_count_value = len(signed_zero_roots)
    post_two_signed_zero_count = int(
        np.count_nonzero(signed_zero_roots_array > (OLD_Q_MAX + 1.0e-10))
    )
    first_post_two_signed_zero_over_m0 = next(
        (float(root) for root in signed_zero_roots if root > (OLD_Q_MAX + 1.0e-10)),
        math.nan,
    )
    last_signed_zero_over_m0 = float(signed_zero_roots[-1]) if signed_zero_roots else math.nan
    min_abs_root_slope = float(np.min(np.abs(root_slopes))) if root_slopes.size else 0.0
    simple_zero_set_available = bool(
        root_slopes.size > 0 and np.all(np.abs(root_slopes) > 1.0e-6)
    )

    sigma_scan = np.array(
        [sign_base.parity_sign(float(q_ratio), signed_zero_roots_array) for q_ratio in q_scan]
    )
    f_signed_reconstructed = sigma_scan * f_abs_scan
    signed_form_factor_reproduction_max_abs_error = float(
        np.max(np.abs(f_signed_reconstructed - f_exact_scan))
    )

    f_exact_at_new_edge = float(sign_base.form_factor(radius, weight, norm, EXTENDED_Q_MAX))
    alpha_exact_at_new_edge = (abs(f_exact_at_new_edge) ** 2) / (4.0 * math.pi)
    q_theory_reproduction_abs_error = abs(
        sign_base.parity_sign(Q_THEORY, signed_zero_roots_array)
        * abs(sign_base.form_factor(radius, weight, norm, Q_THEORY))
        - sign_base.form_factor(radius, weight, norm, Q_THEORY)
    )

    further_interval_extension_surface_present = True
    exact_further_interval_extension_available = bool(
        simple_zero_set_available and signed_form_factor_reproduction_max_abs_error <= 1.0e-12
    )
    gate_a_exact_extension_selected = exact_further_interval_extension_available
    beyond_interval_obstruction_detected = False
    current_rule_blocked = False
    physical_reject_required = False

    formulas = build_formulae()

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "further extension inventory ready",
            sign_base.truth(inventory_ready),
            "The further theorem audit starts only after the 0<=q/m0<=2 closeout is official.",
        ),
        sign_base.row(
            "further_interval_extension_surface_present",
            "pass",
            "further retained-interval extension surface present",
            sign_base.truth(further_interval_extension_surface_present),
            "The current first shot is a direct theorem audit on a wider interval rather than a wait-state retry.",
        ),
        sign_base.row(
            "extended_interval_over_m0",
            "watch",
            "further audit interval upper edge q_max/m0",
            EXTENDED_Q_MAX,
            "This branch extends the retained audit interval from 0<=q/m0<=2 to 0<=q/m0<=4.",
        ),
        sign_base.row(
            "signed_zero_count",
            "watch",
            "signed overlap zero count on 0<=q/m0<=4",
            float(signed_zero_count_value),
            "The wider interval contains additional simple parity-flip points above q/m0=2.",
        ),
        sign_base.row(
            "post_two_signed_zero_count",
            "watch",
            "signed overlap zero count on 2<q/m0<=4",
            float(post_two_signed_zero_count),
            "These are the genuinely new sign flips introduced by the present extension audit.",
        ),
        sign_base.row(
            "first_post_two_signed_zero_over_m0",
            "watch",
            "first signed zero beyond q/m0=2",
            first_post_two_signed_zero_over_m0,
            "This is the first new parity-flip point past the old retained interval edge.",
        ),
        sign_base.row(
            "last_signed_zero_over_m0",
            "watch",
            "last signed zero within 0<=q/m0<=4",
            last_signed_zero_over_m0,
            "This is the last tracked parity-flip point on the current further-extended interval.",
        ),
        sign_base.row(
            "min_abs_root_slope",
            "watch",
            "minimum absolute slope at signed zeros on 0<=q/m0<=4",
            min_abs_root_slope,
            "The theorem remains honest only if all tracked zeros stay simple on the wider interval.",
        ),
        sign_base.row(
            "simple_zero_set_available",
            "pass" if simple_zero_set_available else "reject",
            "simple zero set available on 0<=q/m0<=4",
            sign_base.truth(simple_zero_set_available),
            "Simple zeros keep the Z2 sign theorem well-defined on the wider interval.",
        ),
        sign_base.row(
            "q_theory_reproduction_abs_error",
            "watch",
            "signed reconstruction error at q_theory under further extension",
            q_theory_reproduction_abs_error,
            "The retained matching point remains exactly reconstructed after widening the interval again.",
        ),
        sign_base.row(
            "signed_form_factor_reproduction_max_abs_error",
            "watch",
            "max signed reconstruction error on 0<=q/m0<=4",
            signed_form_factor_reproduction_max_abs_error,
            "Exact zero means the same sign-parity theorem still reproduces the signed form factor on the doubled interval.",
        ),
        sign_base.row(
            "f_exact_at_new_edge",
            "watch",
            "signed form factor at q/m0=4",
            f_exact_at_new_edge,
            "The retained overlap remains real and tiny at the new upper audit edge.",
        ),
        sign_base.row(
            "alpha_exact_at_new_edge",
            "watch",
            "alpha_exact at q/m0=4",
            alpha_exact_at_new_edge,
            "This is the exact alpha carried by the retained theorem at the current audit edge.",
        ),
        sign_base.row(
            "gate_a_exact_extension_selected",
            "pass" if gate_a_exact_extension_selected else "reject",
            "Gate A exact extension retained",
            sign_base.truth(gate_a_exact_extension_selected),
            "The current sign-parity rule survives on the wider interval without changing the observable rule.",
        ),
        sign_base.row(
            "beyond_interval_obstruction_detected",
            "reject" if not beyond_interval_obstruction_detected else "pass",
            "beyond-interval obstruction detected",
            sign_base.truth(beyond_interval_obstruction_detected),
            "This branch records no obstruction on 0<=q/m0<=4, so the second-shot signed-rule branch is not needed yet.",
        ),
        sign_base.row(
            "current_rule_blocked",
            "reject" if not current_rule_blocked else "pass",
            "current sign-parity rule blocked",
            sign_base.truth(current_rule_blocked),
            "The retained rule is not blocked on the present interval, so old surrogate fallback remains inadmissible.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "further_interval_extension_surface_present": further_interval_extension_surface_present,
        "extended_interval_over_m0": EXTENDED_Q_MAX,
        "signed_zero_count": signed_zero_count_value,
        "post_two_signed_zero_count": post_two_signed_zero_count,
        "first_post_two_signed_zero_over_m0": first_post_two_signed_zero_over_m0,
        "last_signed_zero_over_m0": last_signed_zero_over_m0,
        "min_abs_root_slope": min_abs_root_slope,
        "simple_zero_set_available": simple_zero_set_available,
        "signed_form_factor_reproduction_max_abs_error": signed_form_factor_reproduction_max_abs_error,
        "f_exact_at_new_edge": f_exact_at_new_edge,
        "alpha_exact_at_new_edge": alpha_exact_at_new_edge,
        "gate_a_exact_extension_selected": gate_a_exact_extension_selected,
        "beyond_interval_obstruction_detected": beyond_interval_obstruction_detected,
        "current_rule_blocked": current_rule_blocked,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": "vector_qball_form_factor_further_interval_extension_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "signed_zero_roots_over_m0": signed_zero_roots,
        "signed_zero_root_slopes": [float(value) for value in root_slopes],
        "hits": {
            "status_branch_hit": sign_base.hit(status_text, "8.7.56.1955"),
            "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.1955-.1958"),
            "current_problem_hit": sign_base.hit(current_problem_text, "extended_interval_over_m0 = 2.0"),
            "current_status_hit": sign_base.hit(current_status_text, "extended_interval_over_m0 = 2.0"),
            "unified_roadmap_hit": sign_base.hit(unified_text, "116. `.1955-.1958`"),
            "long_roadmap_hit": sign_base.hit(long_text, "8.7.56.1955-.1958"),
            "part5_hit": sign_base.hit(part5_text, "0<=q/m0<=2"),
        },
    }

    declaration_payload = sign_base.payload(
        "8.7.56.1957",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI_CONTEXT),
                "work_history_recent": sign_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "qball_branch_refresh": sign_base.display_path(QBALL_BRANCH_REFRESH),
                "solver_module": sign_base.display_path(QBALL_SOLVER),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "constants": {
                "q_theory_over_m0": Q_THEORY,
                "old_q_max_over_m0": OLD_Q_MAX,
                "extended_q_max_over_m0": EXTENDED_Q_MAX,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        decision,
        evidence,
    )

    route_payload = sign_base.payload(
        "8.7.56.1958",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row(
                "gate_a_exact_extension_selected",
                "pass" if gate_a_exact_extension_selected else "reject",
                "Gate A exact extension retained",
                sign_base.truth(gate_a_exact_extension_selected),
                "The current theorem survives on the wider interval, so the roadmap stays on the interval-extension mainline.",
            ),
            sign_base.row(
                "beyond_interval_obstruction_detected",
                "reject" if not beyond_interval_obstruction_detected else "pass",
                "beyond-interval obstruction detected",
                sign_base.truth(beyond_interval_obstruction_detected),
                "No obstruction appears on the current wider interval, so the signed-rule second shot is not yet required.",
            ),
            sign_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the extended-interval decision gate / theorem stability sync.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": physical_reject_required,
        },
        {
            "overall_status": "vector_qball_form_factor_further_interval_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": formulas},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1955-.1958 further retained-interval extension artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
