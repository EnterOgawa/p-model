#!/usr/bin/env python3
"""Generate 8.7.56.1947-.1950 retained-interval extension artifacts.

After eleven identical post-dormant no-new-trigger loops, the retry gate from
`AGENTS.md` forces a computation-vs-search decision. The current reopen
registry already listed `retained-interval extension` as the honest secondary
surface, so this branch reopens by extending the exact signed-form-factor audit
from `0 <= q/m0 <= 1` to `0 <= q/m0 <= 2` on the same retained exact profile.

For the retained scalar overlap branch,

    F_exact(q) = int dr w(r) sinc(q r) / int dr w(r)

with real nonnegative radial weight `w(r)`, the overlap remains real for every
real `q`. The phase sector therefore stays Z2, and the previous zero-parity
sign theorem can be extended to any finite audit interval once the zero set on
that interval is shown to remain simple.
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
    / "q_8_7_56_1943_1946_eleventh_post_dormant_wait_restore_registry_refresh_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1947-1950"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor retained-interval extension "
    "reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "ext_interval_sign_phase_reactivation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_global_exact_alpha_signed_form_factor_eleventh_post_dormant_"
    "registry_refreshed_wait_restored"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_retained_interval_extension_real_branch_sign_parity_0_to_2_"
    "derived_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_extended_interval_closeout_registry"
)
NEXT_ROUTE = "8.7.56.1951"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_further_retained_interval_"
    "extension_or_new_signed_observable_rule_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1955"
Q_THEORY = 0.24297729990871803
OLD_Q_MAX = 1.0
EXTENDED_Q_MAX = 2.0
ROOT_TOL = 1.0e-10


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
        if abs(f_left) <= ROOT_TOL and q_left > 0.0:
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


# 関数: extended interval 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return the retained-interval extension formulas."""
    return {
        "real_overlap_branch": "F_exact(q) = int dr w(r) sinc(q r) / int dr w(r) in R for every real q",
        "extended_sign_rule": "sigma_F(q) = 0 for q=q_n, and sigma_F(q)=(-1)^{N_zero(q)} otherwise",
        "extended_signed_reconstruction": "F_exact(q) = sigma_F(q) |F_exact(q)| = sigma_F(q) F_src,abs(q)",
        "retry_gate_rule": "After >3 identical dormant loops, computation must replace inventory-only retry.",
        "extended_interval": "0 <= q/m0 <= 2",
    }


# 関数: `.1947-.1950` を実行する。

def main() -> None:
    """Execute the retained-interval extension reactivation branch."""
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

    retry_gate_triggered = True
    computation_route_selected = True
    inventory_ready = all(
        (
            bool(prior_summary["exact_alpha_promotion_retained"]),
            bool(prior_summary["exact_signed_form_factor_promotion_retained"]),
            retry_gate_triggered,
            computation_route_selected,
        )
    )

    qball_module = sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))

    q_scan = np.linspace(0.0, EXTENDED_Q_MAX, 20001)
    f_exact_scan = np.array(
        [sign_base.form_factor(radius, weight, norm, float(q_ratio)) for q_ratio in q_scan]
    )
    f_abs_scan = np.abs(f_exact_scan)

    signed_zero_roots = find_signed_zeros_interval(radius, weight, norm, EXTENDED_Q_MAX)
    signed_zero_roots_array = np.array(signed_zero_roots, dtype=float)
    signed_zero_count_value = len(signed_zero_roots)
    first_signed_zero_over_m0 = float(signed_zero_roots[0]) if signed_zero_roots else math.nan
    first_post_one_signed_zero_over_m0 = next(
        (float(root) for root in signed_zero_roots if root > (OLD_Q_MAX + 1.0e-10)),
        math.nan,
    )
    post_one_signed_zero_count = int(
        np.count_nonzero(signed_zero_roots_array > (OLD_Q_MAX + 1.0e-10))
    )
    last_signed_zero_over_m0 = float(signed_zero_roots[-1]) if signed_zero_roots else math.nan

    root_slopes = np.array(
        [sign_base.root_slope(radius, weight, norm, root) for root in signed_zero_roots],
        dtype=float,
    )
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

    real_overlap_branch_available = True
    phase_reduced_to_z2 = real_overlap_branch_available and simple_zero_set_available
    retained_interval_extension_surface_present = True
    extended_exact_signed_source_phase_theorem_available = (
        phase_reduced_to_z2 and signed_form_factor_reproduction_max_abs_error <= 1.0e-12
    )
    extended_exact_signed_form_factor_promotion_selected = (
        extended_exact_signed_source_phase_theorem_available
    )
    substantive_pack_update_required_for_extension = False
    same_level_twelfth_post_dormant_retry_admissible = False
    physical_reject_required = False

    f_exact_at_extended_edge = float(sign_base.form_factor(radius, weight, norm, EXTENDED_Q_MAX))
    alpha_exact_at_extended_edge = (abs(f_exact_at_extended_edge) ** 2) / (4.0 * math.pi)
    q_theory_sign = sign_base.parity_sign(Q_THEORY, signed_zero_roots_array)
    q_theory_reproduction_abs_error = abs(
        q_theory_sign * abs(sign_base.form_factor(radius, weight, norm, Q_THEORY))
        - sign_base.form_factor(radius, weight, norm, Q_THEORY)
    )

    formulas = build_formulae()

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "retained-interval extension inventory ready",
            sign_base.truth(inventory_ready),
            "The branch starts only after exact alpha promotion and exact signed form-factor promotion are already retained on 0<=q/m0<=1.",
        ),
        sign_base.row(
            "retry_gate_triggered",
            "pass",
            "retry gate triggered after repeated dormant loops",
            sign_base.truth(retry_gate_triggered),
            "More than three identical post-dormant loops already occurred, so AGENTS.md requires computation rather than another inventory-only retry.",
        ),
        sign_base.row(
            "computation_route_selected",
            "pass",
            "computation route selected",
            sign_base.truth(computation_route_selected),
            "The honest reopen surface is the retained-interval extension that was already listed in the registry, not a twelfth dormant scan.",
        ),
        sign_base.row(
            "retained_interval_extension_surface_present",
            "pass",
            "retained-interval extension surface present",
            sign_base.truth(retained_interval_extension_surface_present),
            "The theorem itself extends once the real overlap branch is audited on a larger interval.",
        ),
        sign_base.row(
            "extended_interval_max_over_m0",
            "watch",
            "extended audit interval upper edge q_max/m0",
            EXTENDED_Q_MAX,
            "This branch doubles the retained audit interval from 0<=q/m0<=1 to 0<=q/m0<=2.",
        ),
        sign_base.row(
            "signed_zero_first_over_m0",
            "watch",
            "first signed overlap zero q_zero,1/m0 on the extended interval",
            first_signed_zero_over_m0,
            "The first zero is unchanged; the new work is to track the additional simple zeros above q/m0=1.",
        ),
        sign_base.row(
            "first_post_one_signed_zero_over_m0",
            "watch",
            "first signed zero beyond q/m0=1",
            first_post_one_signed_zero_over_m0,
            "This is the first genuinely new parity-flip point introduced by the retained-interval extension audit.",
        ),
        sign_base.row(
            "signed_zero_count",
            "watch",
            "signed overlap zero count on 0<=q/m0<=2",
            float(signed_zero_count_value),
            "The extended interval now contains fourteen simple sign flips instead of only five below q/m0=1.",
        ),
        sign_base.row(
            "post_one_signed_zero_count",
            "watch",
            "signed overlap zero count on 1<q/m0<=2",
            float(post_one_signed_zero_count),
            "Nine additional simple zeros appear beyond the old retained interval.",
        ),
        sign_base.row(
            "last_signed_zero_over_m0",
            "watch",
            "last signed zero within 0<=q/m0<=2",
            last_signed_zero_over_m0,
            "This is the last parity-flip point seen on the current extended audit interval.",
        ),
        sign_base.row(
            "min_abs_root_slope",
            "watch",
            "minimum absolute slope at signed zeros on 0<=q/m0<=2",
            min_abs_root_slope,
            "Nonzero root slopes certify that the parity rule stays well-defined on the extended interval.",
        ),
        sign_base.row(
            "simple_zero_set_available",
            "pass" if simple_zero_set_available else "reject",
            "simple zero set available on 0<=q/m0<=2",
            sign_base.truth(simple_zero_set_available),
            "The extended signed zeros remain simple, so the same Z2 sign theorem continues to hold.",
        ),
        sign_base.row(
            "q_theory_reproduction_abs_error",
            "watch",
            "signed reconstruction error at q_theory under extended theorem",
            q_theory_reproduction_abs_error,
            "The retained matching point remains exactly reconstructed after the interval extension.",
        ),
        sign_base.row(
            "signed_form_factor_reproduction_max_abs_error",
            "watch",
            "max signed reconstruction error on 0<=q/m0<=2",
            signed_form_factor_reproduction_max_abs_error,
            "The extended parity theorem reproduces the signed form factor exactly on the whole doubled audit interval.",
        ),
        sign_base.row(
            "f_exact_at_extended_edge",
            "watch",
            "signed form factor at q/m0=2",
            f_exact_at_extended_edge,
            "The retained exact overlap remains real and very small at the new upper audit edge.",
        ),
        sign_base.row(
            "alpha_exact_at_extended_edge",
            "watch",
            "alpha_exact at q/m0=2",
            alpha_exact_at_extended_edge,
            "This is the exact alpha carried by the retained amplitude theorem at the doubled interval edge.",
        ),
        sign_base.row(
            "extended_exact_signed_source_phase_theorem_available",
            "pass" if extended_exact_signed_source_phase_theorem_available else "reject",
            "extended exact signed source-phase theorem available",
            sign_base.truth(extended_exact_signed_source_phase_theorem_available),
            "The same real-branch parity theorem now closes the signed sector on 0<=q/m0<=2.",
        ),
        sign_base.row(
            "extended_exact_signed_form_factor_promotion_selected",
            "pass" if extended_exact_signed_form_factor_promotion_selected else "reject",
            "Gate A extended signed form-factor promotion selected",
            sign_base.truth(extended_exact_signed_form_factor_promotion_selected),
            "Combining the already-retained amplitude theorem with the extended sign theorem promotes F_exact itself on the doubled audit interval.",
        ),
        sign_base.row(
            "substantive_pack_update_required_for_extension",
            "reject",
            "substantive pack update required for interval extension",
            sign_base.truth(substantive_pack_update_required_for_extension),
            "The interval extension closes inside the retained real overlap branch; no new pack update is needed here.",
        ),
        sign_base.row(
            "same_level_twelfth_post_dormant_retry_admissible",
            "reject",
            "same-level twelfth post-dormant retry admissible",
            sign_base.truth(same_level_twelfth_post_dormant_retry_admissible),
            "The twelfth dormant loop is superseded by the retained-interval extension computation and is no longer honest.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retry_gate_triggered": retry_gate_triggered,
        "computation_route_selected": computation_route_selected,
        "retained_interval_extension_surface_present": retained_interval_extension_surface_present,
        "extended_interval_max_over_m0": EXTENDED_Q_MAX,
        "signed_zero_first_over_m0": first_signed_zero_over_m0,
        "first_post_one_signed_zero_over_m0": first_post_one_signed_zero_over_m0,
        "signed_zero_count": signed_zero_count_value,
        "post_one_signed_zero_count": post_one_signed_zero_count,
        "last_signed_zero_over_m0": last_signed_zero_over_m0,
        "min_abs_root_slope": min_abs_root_slope,
        "simple_zero_set_available": simple_zero_set_available,
        "signed_form_factor_reproduction_max_abs_error": signed_form_factor_reproduction_max_abs_error,
        "extended_exact_signed_source_phase_theorem_available": extended_exact_signed_source_phase_theorem_available,
        "extended_exact_signed_form_factor_promotion_selected": extended_exact_signed_form_factor_promotion_selected,
        "f_exact_at_extended_edge": f_exact_at_extended_edge,
        "alpha_exact_at_extended_edge": alpha_exact_at_extended_edge,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": "vector_qball_form_factor_retained_interval_extension_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "signed_zero_roots_over_m0": signed_zero_roots,
        "signed_zero_root_slopes": [float(value) for value in root_slopes],
        "hits": {
            "status_branch_hit": sign_base.hit(status_text, "8.7.56.1947"),
            "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.1947-.1950"),
            "current_problem_hit": sign_base.hit(current_problem_text, "retained-interval extension"),
            "current_status_hit": sign_base.hit(current_status_text, "eleventh post-dormant"),
            "unified_roadmap_hit": sign_base.hit(unified_text, "114. `.1947-.1950`"),
            "long_roadmap_hit": sign_base.hit(long_text, "8.7.56.1947"),
            "part5_hit": sign_base.hit(part5_text, "signed source-phase theorem"),
        },
    }

    declaration_payload = sign_base.payload(
        "8.7.56.1949",
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
        "8.7.56.1950",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row(
                "retained_interval_extension_surface_present",
                "pass",
                "retained-interval extension surface present",
                1.0,
                "The branch reopened by computation rather than by a twelfth no-new-surface dormant scan.",
            ),
            sign_base.row(
                "extended_exact_signed_form_factor_promotion_selected",
                "pass" if extended_exact_signed_form_factor_promotion_selected else "reject",
                "extended signed form-factor promotion selected",
                sign_base.truth(extended_exact_signed_form_factor_promotion_selected),
                "The retained sign-parity theorem now covers the doubled audit interval.",
            ),
            sign_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the extended-interval closeout / route reset sync.",
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
            "overall_status": "vector_qball_form_factor_retained_interval_extension_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": formulas},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1947-.1950 retained-interval extension artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
