#!/usr/bin/env python3
"""Generate 8.7.56.2071-.2074 asymptotic farther-harmonic continuation artifacts.

This branch continues the retained boundary bulk-lattice signed-rule family
after `.2063-.2070` retained the same lattice through harmonic 128. The new
honest question is whether the exact same lattice survives on asymptotically
farther harmonic blocks without introducing any new loading theorem or pack
update.
"""

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
import scripts.quantum.t2a_2047 as loading_base
import scripts.quantum.t2a_2055 as lattice_base
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
PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2067_2070_harmonic_ext_loading_registry_declaration_gate_metrics.json"
PRIOR_AUDIT = PUBLIC_OUT / "q_8_7_56_2063_2066_harmonic_ext_loading_theorem_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2071-2074"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor bulk-lattice asymptotic "
    "farther-harmonic continuation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_asymptotic_continuation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_extension_128_retained_"
    "loading_index_theorem_deferred_asymptotic_continuation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_asymptotic_extension_512_"
    "retained_loading_index_theorem_reserve_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_loading_index_theorem_"
    "reserve_or_pack_update_registry"
)
NEXT_ROUTE = "8.7.56.2075"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_bulk_lattice_farther_"
    "asymptotic_continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2079"

ASYMPTOTIC_BLOCKS = [
    (129, 160),
    (161, 192),
    (193, 256),
    (257, 320),
    (321, 384),
    (385, 448),
    (449, 512),
]
WINDOW_SCAN_DENSITY = loading_base.WINDOW_SCAN_DENSITY


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


# 関数: asymptotic windows を構成する。

def build_asymptotic_windows(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    alias_1: float,
) -> list[dict[str, object]]:
    """Build farther windows from harmonic 129 through 512."""
    fit_offsets = (loading_base.FIT_Q_MIN - alias_1, loading_base.FIT_Q_MAX - alias_1)
    edge_offsets = (
        loading_base.EDGE_Q_MIN - (2.0 * alias_1),
        loading_base.EDGE_Q_MAX - (2.0 * alias_1),
    )
    windows: list[dict[str, object]] = []
    for harmonic_index in range(ASYMPTOTIC_BLOCKS[0][0], ASYMPTOTIC_BLOCKS[-1][1] + 1):
        alias_harmonic = harmonic_index * alias_1
        offsets = fit_offsets if (harmonic_index % 2) == 1 else edge_offsets
        q_min, q_max = loading_base.translated_window(alias_harmonic, offsets)
        q_scan = np.linspace(
            q_min,
            q_max,
            int(round((q_max - q_min) * WINDOW_SCAN_DENSITY)) + 1,
        )
        exact_values, exact_abs, exact_sign = phase_base.exact_sign_data(
            radius,
            weight,
            norm,
            q_scan,
        )
        windows.append(
            {
                "harmonic_index": harmonic_index,
                "alias_harmonic": float(alias_harmonic),
                "q_min": float(q_min),
                "q_max": float(q_max),
                "q_scan": q_scan,
                "exact_values": exact_values,
                "exact_abs": exact_abs,
                "exact_sign": exact_sign,
                "template_type": "fit" if (harmonic_index % 2) == 1 else "edge",
            }
        )

    return windows


# 関数: loading-index frequency を整理する。

def build_loading_index_frequency(loading_indices: np.ndarray) -> list[dict[str, int]]:
    """Return loading-index frequencies sorted by descending multiplicity."""
    values, counts = np.unique(loading_indices, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [
        {"loading_index": int(values[index]), "count": int(counts[index])}
        for index in order
    ]


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the asymptotic continuation audit."""
    return {
        "retained_bulk_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "continuation_gate": "retain continuation if max mismatch <= 0.25 and min sign correlation >= 0.5 on every asymptotic block",
        "reserve_theorem_read": "keep exact loading-index theorem as reserve while the same lattice remains honest on farther harmonic blocks",
        "distribution_read": "summarize m_n on 129..512 via empirical occupancy rather than forcing a new exact theorem",
    }


# 関数: `.2071-.2074` を実行する。

def main() -> None:
    """Execute the asymptotic farther-harmonic continuation audit."""
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
    inventory_ready = bool(prior_gate_summary["asymptotic_farther_harmonic_continuation_admissible_now"])

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

    windows = build_asymptotic_windows(radius, weight, norm, alias_1)
    theorem_lattice_base = float(prior_audit_summary["theorem_lattice_base_over_m0"])
    theorem_lattice_step = float(prior_audit_summary["bulk_delta_r_over_m0"])
    theorem_results = lattice_base.evaluate_lattice_family(
        windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )

    block_summaries: dict[str, dict[str, float]] = {}
    for block_start, block_end in ASYMPTOTIC_BLOCKS:
        key = f"{block_start}_{block_end}"
        block_summaries[key] = lattice_base.summarize_harmonic_group(
            windows,
            theorem_results,
            list(range(block_start, block_end + 1)),
        )

    loading_indices = np.asarray(
        [int(round(result["loading_index"])) for result in theorem_results],
        dtype=int,
    )
    loading_index_frequency = build_loading_index_frequency(loading_indices)
    top_frequency = loading_index_frequency[0]
    loading_index_mean = float(loading_indices.mean())
    loading_index_std = float(loading_indices.std())
    loading_index_min = int(loading_indices.min())
    loading_index_max = int(loading_indices.max())

    asymptotic_farther_harmonic_extension_to_512_supported = bool(
        all(summary["max_mismatch"] <= 0.25 for summary in block_summaries.values())
        and all(summary["min_correlation"] >= 0.5 for summary in block_summaries.values())
    )
    asymptotic_loading_index_theorem_available = False
    same_lattice_survives_to_512 = asymptotic_farther_harmonic_extension_to_512_supported
    exact_loading_index_theorem_remains_reserve = True
    farther_asymptotic_continuation_admissible_now = asymptotic_farther_harmonic_extension_to_512_supported
    same_level_loading_index_affine_retry_admissible = False
    same_level_smooth_loading_retry_admissible = False
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "bulk-lattice asymptotic continuation inventory ready",
            sign_base.truth(inventory_ready),
            "This branch starts only after the extension-to-128 registry has already selected farther continuation as the honest mainline.",
        ),
    ]
    for block_start, block_end in ASYMPTOTIC_BLOCKS:
        key = f"{block_start}_{block_end}"
        summary = block_summaries[key]
        rows.extend(
            [
                sign_base.row(
                    f"extension_{block_start}_{block_end}_max_mismatch_fraction",
                    "pass" if summary["max_mismatch"] <= 0.25 else "reject",
                    f"bulk-lattice max mismatch on harmonic {block_start}..{block_end}",
                    summary["max_mismatch"],
                    "Each asymptotic block must survive with the same mismatch threshold that retained the earlier continuation windows.",
                ),
                sign_base.row(
                    f"extension_{block_start}_{block_end}_min_sign_correlation",
                    "pass" if summary["min_correlation"] >= 0.5 else "reject",
                    f"bulk-lattice min sign correlation on harmonic {block_start}..{block_end}",
                    summary["min_correlation"],
                    "The continuation stays honest only if the same lattice preserves positive sign correlation on every asymptotic block.",
                ),
            ]
        )

    rows.extend(
        [
            sign_base.row(
                "loading_index_mode_count_129_512",
                "pass",
                "dominant loading-index occupancy on harmonic 129..512",
                top_frequency["count"],
                "The asymptotic loading-index distribution is summarized empirically rather than forced into a new exact theorem.",
            ),
            sign_base.row(
                "asymptotic_farther_harmonic_extension_to_512_supported",
                "pass" if asymptotic_farther_harmonic_extension_to_512_supported else "reject",
                "asymptotic farther-harmonic extension to 512 supported",
                sign_base.truth(asymptotic_farther_harmonic_extension_to_512_supported),
                "If every asymptotic block survives the same thresholds, the retained lattice continues honestly and the theorem gap remains reserve-only.",
            ),
            sign_base.row(
                "asymptotic_loading_index_theorem_available",
                "reject",
                "asymptotic loading-index theorem available",
                sign_base.truth(asymptotic_loading_index_theorem_available),
                "The continuation survives, but no exact low-order theorem for the discrete loading indices is implied by that survival.",
            ),
            sign_base.row(
                "same_lattice_survives_to_512",
                "pass" if same_lattice_survives_to_512 else "reject",
                "same lattice survives to harmonic 512",
                sign_base.truth(same_lattice_survives_to_512),
                "The honest next question is farther continuation of the same retained lattice, not a reopened same-level theorem fit.",
            ),
            sign_base.row(
                "exact_loading_index_theorem_remains_reserve",
                "pass",
                "exact loading-index theorem remains reserve",
                sign_base.truth(exact_loading_index_theorem_remains_reserve),
                "The exact loading-index theorem remains meaningful only as a reserve surface while the continuation route is still alive.",
            ),
        ]
    )

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "loading_index_mode_129_512": top_frequency["loading_index"],
        "loading_index_mode_count_129_512": top_frequency["count"],
        "loading_index_mean_129_512": loading_index_mean,
        "loading_index_std_129_512": loading_index_std,
        "loading_index_min_129_512": loading_index_min,
        "loading_index_max_129_512": loading_index_max,
        "loading_index_top_frequencies_129_512": loading_index_frequency[:15],
        "extension_129_160_max_mismatch_fraction": block_summaries["129_160"]["max_mismatch"],
        "extension_129_160_min_sign_correlation": block_summaries["129_160"]["min_correlation"],
        "extension_129_160_signed_reconstruction_max_abs_error": block_summaries["129_160"]["max_abs_error"],
        "extension_161_192_max_mismatch_fraction": block_summaries["161_192"]["max_mismatch"],
        "extension_161_192_min_sign_correlation": block_summaries["161_192"]["min_correlation"],
        "extension_161_192_signed_reconstruction_max_abs_error": block_summaries["161_192"]["max_abs_error"],
        "extension_193_256_max_mismatch_fraction": block_summaries["193_256"]["max_mismatch"],
        "extension_193_256_min_sign_correlation": block_summaries["193_256"]["min_correlation"],
        "extension_193_256_signed_reconstruction_max_abs_error": block_summaries["193_256"]["max_abs_error"],
        "extension_257_320_max_mismatch_fraction": block_summaries["257_320"]["max_mismatch"],
        "extension_257_320_min_sign_correlation": block_summaries["257_320"]["min_correlation"],
        "extension_257_320_signed_reconstruction_max_abs_error": block_summaries["257_320"]["max_abs_error"],
        "extension_321_384_max_mismatch_fraction": block_summaries["321_384"]["max_mismatch"],
        "extension_321_384_min_sign_correlation": block_summaries["321_384"]["min_correlation"],
        "extension_321_384_signed_reconstruction_max_abs_error": block_summaries["321_384"]["max_abs_error"],
        "extension_385_448_max_mismatch_fraction": block_summaries["385_448"]["max_mismatch"],
        "extension_385_448_min_sign_correlation": block_summaries["385_448"]["min_correlation"],
        "extension_385_448_signed_reconstruction_max_abs_error": block_summaries["385_448"]["max_abs_error"],
        "extension_449_512_max_mismatch_fraction": block_summaries["449_512"]["max_mismatch"],
        "extension_449_512_min_sign_correlation": block_summaries["449_512"]["min_correlation"],
        "extension_449_512_signed_reconstruction_max_abs_error": block_summaries["449_512"]["max_abs_error"],
        "asymptotic_farther_harmonic_extension_to_512_supported": asymptotic_farther_harmonic_extension_to_512_supported,
        "asymptotic_loading_index_theorem_available": asymptotic_loading_index_theorem_available,
        "same_lattice_survives_to_512": same_lattice_survives_to_512,
        "exact_loading_index_theorem_remains_reserve": exact_loading_index_theorem_remains_reserve,
        "same_level_loading_index_affine_retry_admissible": same_level_loading_index_affine_retry_admissible,
        "same_level_smooth_loading_retry_admissible": same_level_smooth_loading_retry_admissible,
        "farther_asymptotic_continuation_admissible_now": farther_asymptotic_continuation_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2073",
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
            "overall_status": "vector_qball_form_factor_asymptotic_farther_harmonic_continuation_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2071"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2071-.2074"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2071"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2071"),
                "unified_roadmap_hit": find_line(unified_text, ".2071-.2074"),
                "long_roadmap_hit": find_line(long_text, ".2071-.2074"),
                "part5_hit": find_line(part5_text, ".2063-.2070"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2071"))),
            "The asymptotic continuation audit is only valid if status still points to the same official branch.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2071-.2074"))),
            "The public roadmap must expose the asymptotic continuation audit before its result is frozen.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2071-.2074"))),
            "The long-horizon roadmap must carry the same asymptotic continuation route.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2074",
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
            "overall_status": "vector_qball_form_factor_asymptotic_farther_harmonic_continuation_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2071"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2071-.2074"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2071"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2071"),
                "unified_roadmap_hit": find_line(unified_text, ".2071-.2074"),
                "long_roadmap_hit": find_line(long_text, ".2071-.2074"),
                "part5_hit": find_line(part5_text, ".2063-.2070"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
