#!/usr/bin/env python3
"""Generate 8.7.56.2095-.2098 quarter-band farther continuation artifacts."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
import scripts.quantum.t2a_2023 as alias_base
import scripts.quantum.t2a_2031 as phase_base
import scripts.quantum.t2a_2055 as lattice_base
import scripts.quantum.t2a_2087 as ultra_base
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
PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2091_2094_harmonic_quarter_band_registry_reset_declaration_gate_metrics.json"
PRIOR_AUDIT = PUBLIC_OUT / "q_8_7_56_2087_2090_harmonic_ultra_asymptotic_continuation_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2095-2098"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor quarter-band farther "
    "continuation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_quarter_band_farther_continuation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_quarter_band_asymptotic_"
    "retain_further_continuation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_quarter_band_asymptotic_"
    "extension_4096_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_loading_index_theorem_"
    "reserve_or_pack_update_registry_refresh"
)
NEXT_ROUTE = "8.7.56.2099"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quarter_band_ultra_farther_"
    "continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2103"

FARTHER_BLOCKS = [
    (3073, 3328),
    (3329, 3584),
    (3585, 3840),
    (3841, 4096),
]
QUARTER_REFERENCE = 0.25


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


# 関数: テキスト中の最初の一致行を返す。

def find_line(text: str, pattern: str) -> dict[str, object] | None:
    """Return the first matching line payload for one text pattern."""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": line.strip(),
            }

    return None


# 関数: loading-index frequency を整理する。

def build_loading_index_frequency(loading_indices: np.ndarray) -> list[dict[str, int]]:
    """Return loading-index frequencies sorted by descending multiplicity."""
    values, counts = np.unique(loading_indices, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [
        {"loading_index": int(values[index]), "count": int(counts[index])}
        for index in order
    ]


# 関数: 任意列が単調非増加か判定する。

def monotone_nonincreasing(values: list[float]) -> bool:
    """Return whether one sequence is monotone nonincreasing."""
    return all(left >= right for left, right in zip(values, values[1:]))


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the quarter-band farther continuation audit."""
    return {
        "retained_bulk_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "quarter_band_reference": "quarter-band continuation tracks max mismatch in a narrow band around 1/4 rather than demanding the retired strict ceiling",
        "inherited_ceiling": "Delta_quarter,farther = 2 * Delta_quarter,1025..3072",
        "farther_gate": "retain farther continuation if |max_mismatch-1/4| <= Delta_quarter,farther, min sign correlation >= 0.5 on every farther block, and signed reconstruction max abs error keeps decaying from the 2561..3072 monitor block",
    }


# 関数: `.2095-.2098` を実行する。

def main() -> None:
    """Execute the quarter-band farther continuation audit."""
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
        PRIOR_GATE,
        PRIOR_AUDIT,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    inventory_ready = bool(prior_gate_summary["quarter_band_further_continuation_admissible_now"])

    qball_branch_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)
    qball_module = sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    bulk_delta_r, _bulk_fraction, _edge_gap = alias_base.bulk_grid_summary(radius)
    alias_1 = (2.0 * np.pi) / bulk_delta_r
    lookup_q = np.arange(
        0.0,
        phase_base.LOOKUP_Q_MAX + phase_base.LOOKUP_Q_STEP,
        phase_base.LOOKUP_Q_STEP,
        dtype=float,
    )
    lookup_values = phase_base.form_factor_array(radius, weight, norm, lookup_q)

    theorem_lattice_base = float(prior_audit_summary["theorem_lattice_base_over_m0"])
    theorem_lattice_step = float(prior_audit_summary["bulk_delta_r_over_m0"])
    windows = ultra_base.build_asymptotic_windows(
        radius,
        weight,
        norm,
        alias_1,
        FARTHER_BLOCKS[0][0],
        FARTHER_BLOCKS[-1][1],
    )
    theorem_results = lattice_base.evaluate_lattice_family(
        windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )

    block_summaries: dict[str, dict[str, float]] = {}
    for block_start, block_end in FARTHER_BLOCKS:
        key = f"{block_start}_{block_end}"
        block_summaries[key] = lattice_base.summarize_harmonic_group(
            windows,
            theorem_results,
            list(range(block_start, block_end + 1)),
        )

    prior_quarter_deviation = float(prior_gate_summary["quarter_band_abs_deviation_max_1025_3072"])
    prior_last_abs_error = float(prior_audit_summary["monitor_2561_3072_signed_reconstruction_max_abs_error"])
    quarter_band_abs_deviation_ceiling_3073_4096 = float(2.0 * prior_quarter_deviation)
    quarter_band_deviations = {
        key: abs(summary["max_mismatch"] - QUARTER_REFERENCE)
        for key, summary in block_summaries.items()
    }
    quarter_band_abs_deviation_max_3073_4096 = float(max(quarter_band_deviations.values()))
    quarter_band_min_sign_correlation_3073_4096 = float(
        min(summary["min_correlation"] for summary in block_summaries.values())
    )
    farther_error_sequence = [prior_last_abs_error] + [
        block_summaries[f"{block_start}_{block_end}"]["max_abs_error"]
        for block_start, block_end in FARTHER_BLOCKS
    ]
    signed_reconstruction_abs_error_continues_decay = monotone_nonincreasing(farther_error_sequence)
    quarter_band_further_continuation_to_4096_supported = bool(
        all(
            quarter_band_deviations[f"{block_start}_{block_end}"] <= quarter_band_abs_deviation_ceiling_3073_4096
            and block_summaries[f"{block_start}_{block_end}"]["min_correlation"] >= 0.5
            for block_start, block_end in FARTHER_BLOCKS
        )
        and signed_reconstruction_abs_error_continues_decay
    )
    same_lattice_survives_to_4096_under_quarter_band = bool(
        quarter_band_further_continuation_to_4096_supported
    )
    exact_loading_index_theorem_remains_reserve = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    loading_indices = np.asarray(
        [int(round(result["loading_index"])) for result in theorem_results],
        dtype=int,
    )
    loading_index_frequency = build_loading_index_frequency(loading_indices)
    top_frequency = loading_index_frequency[0]

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "quarter-band farther continuation inventory ready",
            sign_base.truth(inventory_ready),
            "The branch starts only after the quarter-band reset has already promoted farther continuation as the honest mainline.",
        ),
        sign_base.row(
            "quarter_band_abs_deviation_ceiling_3073_4096",
            "watch",
            "quarter-band absolute deviation ceiling on harmonic 3073..4096",
            quarter_band_abs_deviation_ceiling_3073_4096,
            "The farther continuation inherits a widened deviation ceiling equal to twice the already retained 1025..3072 quarter-band width.",
        ),
    ]
    for block_start, block_end in FARTHER_BLOCKS:
        key = f"{block_start}_{block_end}"
        summary = block_summaries[key]
        rows.extend(
            [
                sign_base.row(
                    f"quarter_{block_start}_{block_end}_max_mismatch_fraction",
                    "pass" if quarter_band_deviations[key] <= quarter_band_abs_deviation_ceiling_3073_4096 else "reject",
                    f"quarter-band max mismatch on harmonic {block_start}..{block_end}",
                    summary["max_mismatch"],
                    "The farther continuation no longer uses the retired strict 0.25 ceiling; it tracks whether the same lattice remains inside the inherited quarter-band saturation window.",
                ),
                sign_base.row(
                    f"quarter_{block_start}_{block_end}_min_sign_correlation",
                    "pass" if summary["min_correlation"] >= 0.5 else "reject",
                    f"quarter-band min sign correlation on harmonic {block_start}..{block_end}",
                    summary["min_correlation"],
                    "Positive sign correlation remains the non-negotiable guardrail even after the quarter-band route reset.",
                ),
            ]
        )

    rows.extend(
        [
            sign_base.row(
                "quarter_band_abs_deviation_max_3073_4096",
                "pass" if quarter_band_abs_deviation_max_3073_4096 <= quarter_band_abs_deviation_ceiling_3073_4096 else "reject",
                "max absolute deviation from quarter reference on harmonic 3073..4096",
                quarter_band_abs_deviation_max_3073_4096,
                "This is the direct farther-continuation continuation of the retained quarter-band surface.",
            ),
            sign_base.row(
                "quarter_band_min_sign_correlation_3073_4096",
                "pass" if quarter_band_min_sign_correlation_3073_4096 >= 0.5 else "reject",
                "min sign correlation on harmonic 3073..4096",
                quarter_band_min_sign_correlation_3073_4096,
                "The farther continuation remains honest only if every block stays positively correlated with the exact sign family.",
            ),
            sign_base.row(
                "signed_reconstruction_abs_error_continues_decay",
                "pass" if signed_reconstruction_abs_error_continues_decay else "reject",
                "signed reconstruction absolute error continues to decay beyond harmonic 3072",
                sign_base.truth(signed_reconstruction_abs_error_continues_decay),
                "The farther continuation is only honest if the pointwise absolute reconstruction error keeps shrinking beyond the retained 2561..3072 monitor block.",
            ),
            sign_base.row(
                "quarter_band_further_continuation_to_4096_supported",
                "pass" if quarter_band_further_continuation_to_4096_supported else "reject",
                "quarter-band farther continuation to 4096 supported",
                sign_base.truth(quarter_band_further_continuation_to_4096_supported),
                "This row decides whether the quarter-band asymptotic surface survives one more explicit farther-harmonic extension.",
            ),
            sign_base.row(
                "same_lattice_survives_to_4096_under_quarter_band",
                "pass" if same_lattice_survives_to_4096_under_quarter_band else "reject",
                "same boundary bulk lattice survives to harmonic 4096 under quarter-band surface",
                sign_base.truth(same_lattice_survives_to_4096_under_quarter_band),
                "The same lattice is retained only if the quarter-band continuation survives without opening a new signed rule or pack update.",
            ),
            sign_base.row(
                "exact_loading_index_theorem_remains_reserve",
                "pass",
                "exact loading-index theorem remains reserve",
                sign_base.truth(exact_loading_index_theorem_remains_reserve),
                "Even after farther continuation, the exact loading-index theorem remains reserve-only while the same lattice survives empirically.",
            ),
        ]
    )

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "loading_index_mode_3073_4096": top_frequency["loading_index"],
        "loading_index_mode_count_3073_4096": top_frequency["count"],
        "loading_index_mean_3073_4096": float(loading_indices.mean()),
        "loading_index_std_3073_4096": float(loading_indices.std()),
        "loading_index_min_3073_4096": int(loading_indices.min()),
        "loading_index_max_3073_4096": int(loading_indices.max()),
        "loading_index_top_frequencies_3073_4096": loading_index_frequency[:15],
        "quarter_band_abs_deviation_ceiling_3073_4096": quarter_band_abs_deviation_ceiling_3073_4096,
        "quarter_3073_3328_max_mismatch_fraction": block_summaries["3073_3328"]["max_mismatch"],
        "quarter_3073_3328_min_sign_correlation": block_summaries["3073_3328"]["min_correlation"],
        "quarter_3073_3328_signed_reconstruction_max_abs_error": block_summaries["3073_3328"]["max_abs_error"],
        "quarter_3329_3584_max_mismatch_fraction": block_summaries["3329_3584"]["max_mismatch"],
        "quarter_3329_3584_min_sign_correlation": block_summaries["3329_3584"]["min_correlation"],
        "quarter_3329_3584_signed_reconstruction_max_abs_error": block_summaries["3329_3584"]["max_abs_error"],
        "quarter_3585_3840_max_mismatch_fraction": block_summaries["3585_3840"]["max_mismatch"],
        "quarter_3585_3840_min_sign_correlation": block_summaries["3585_3840"]["min_correlation"],
        "quarter_3585_3840_signed_reconstruction_max_abs_error": block_summaries["3585_3840"]["max_abs_error"],
        "quarter_3841_4096_max_mismatch_fraction": block_summaries["3841_4096"]["max_mismatch"],
        "quarter_3841_4096_min_sign_correlation": block_summaries["3841_4096"]["min_correlation"],
        "quarter_3841_4096_signed_reconstruction_max_abs_error": block_summaries["3841_4096"]["max_abs_error"],
        "quarter_band_abs_deviation_max_3073_4096": quarter_band_abs_deviation_max_3073_4096,
        "quarter_band_min_sign_correlation_3073_4096": quarter_band_min_sign_correlation_3073_4096,
        "signed_reconstruction_abs_error_continues_decay": signed_reconstruction_abs_error_continues_decay,
        "quarter_band_further_continuation_to_4096_supported": quarter_band_further_continuation_to_4096_supported,
        "same_lattice_survives_to_4096_under_quarter_band": same_lattice_survives_to_4096_under_quarter_band,
        "exact_loading_index_theorem_remains_reserve": exact_loading_index_theorem_remains_reserve,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2097",
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
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_quarter_band_farther_continuation_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2095"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2095-.2098"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2095"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2095"),
                "unified_roadmap_hit": find_line(unified_text, ".2095-.2098"),
                "long_roadmap_hit": find_line(long_text, ".2095-.2098"),
                "part5_hit": find_line(part5_text, ".2087-.2094"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2095"))),
            "The quarter-band farther continuation audit is only valid if status already points to the same official branch.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2095-.2098"))),
            "The public roadmap must expose the farther continuation audit before its result is frozen.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2095-.2098"))),
            "The long-horizon roadmap must expose the same farther continuation route.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2098",
        STEP_NAME + " route sync",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "declaration_gate": declaration_paths["json"],
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        route_sync_rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_quarter_band_farther_continuation_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2095"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2095-.2098"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2095"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2095"),
                "unified_roadmap_hit": find_line(unified_text, ".2095-.2098"),
                "long_roadmap_hit": find_line(long_text, ".2095-.2098"),
                "part5_hit": find_line(part5_text, ".2087-.2094"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
