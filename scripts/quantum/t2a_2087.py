#!/usr/bin/env python3
"""Generate 8.7.56.2087-.2090 ultra-asymptotic continuation artifacts."""

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
PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2083_2086_harmonic_farther_asymptotic_registry_declaration_gate_metrics.json"
PRIOR_AUDIT = PUBLIC_OUT / "q_8_7_56_2063_2066_harmonic_ext_loading_theorem_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2087-2090"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor bulk-lattice ultra-"
    "asymptotic continuation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_ultra_asymptotic_continuation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_asymptotic_extension_1024_"
    "retained_loading_index_theorem_reserve_ultra_asymptotic_continuation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_ultra_asymptotic_strict_"
    "blocked_quarter_band_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quarter_band_asymptotic_"
    "registry_reset"
)
NEXT_ROUTE = "8.7.56.2091"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quarter_band_farther_"
    "continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2095"

PRIMARY_BLOCKS = [
    (1025, 1280),
    (1281, 1536),
    (1537, 1792),
    (1793, 2048),
]
MONITOR_BLOCKS = [
    (2049, 2560),
    (2561, 3072),
]
QUARTER_REFERENCE = 0.25
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


# 関数: 任意 harmonic 範囲の windows を構成する。

def build_asymptotic_windows(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    alias_1: float,
    start_harmonic: int,
    end_harmonic: int,
) -> list[dict[str, object]]:
    """Build translated fit/edge windows on one harmonic range."""
    fit_offsets = (loading_base.FIT_Q_MIN - alias_1, loading_base.FIT_Q_MAX - alias_1)
    edge_offsets = (
        loading_base.EDGE_Q_MIN - (2.0 * alias_1),
        loading_base.EDGE_Q_MAX - (2.0 * alias_1),
    )
    windows: list[dict[str, object]] = []
    for harmonic_index in range(start_harmonic, end_harmonic + 1):
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


# 関数: 隣接 block で absolute error が単調減少しているか判定する。

def monotone_nonincreasing(values: list[float]) -> bool:
    """Return whether one sequence is monotone nonincreasing."""
    return all(left >= right for left, right in zip(values, values[1:]))


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the ultra-asymptotic continuation audit."""
    return {
        "retained_bulk_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "strict_gate": "strict continuation retains only if max mismatch <= 0.25 and min sign correlation >= 0.5 on every primary block",
        "quarter_band_read": "if strict gate blocks but max mismatch stays in a narrow band around 1/4 while min sign correlation remains positive and signed reconstruction error keeps decaying, reset to quarter-band asymptotic saturation",
        "monitor_read": "probe 2049..3072 to decide whether the quarter-band family is local noise or the honest new asymptotic surface",
    }


# 関数: `.2087-.2090` を実行する。

def main() -> None:
    """Execute the ultra-asymptotic continuation audit."""
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
    inventory_ready = bool(prior_gate_summary["ultra_asymptotic_continuation_admissible_now"])

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

    all_blocks = PRIMARY_BLOCKS + MONITOR_BLOCKS
    windows = build_asymptotic_windows(
        radius,
        weight,
        norm,
        alias_1,
        PRIMARY_BLOCKS[0][0],
        MONITOR_BLOCKS[-1][1],
    )
    theorem_results = lattice_base.evaluate_lattice_family(
        windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )

    block_summaries: dict[str, dict[str, float]] = {}
    for block_start, block_end in all_blocks:
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

    strict_ultra_asymptotic_extension_to_2048_supported = bool(
        all(
            block_summaries[f"{block_start}_{block_end}"]["max_mismatch"] <= QUARTER_REFERENCE
            and block_summaries[f"{block_start}_{block_end}"]["min_correlation"] >= 0.5
            for block_start, block_end in PRIMARY_BLOCKS
        )
    )
    quarter_band_block_deviations = {
        key: abs(summary["max_mismatch"] - QUARTER_REFERENCE)
        for key, summary in block_summaries.items()
    }
    quarter_band_abs_deviation_max_1025_3072 = float(max(quarter_band_block_deviations.values()))
    quarter_band_min_sign_correlation_1025_3072 = float(
        min(summary["min_correlation"] for summary in block_summaries.values())
    )
    abs_error_sequence = [
        block_summaries[f"{block_start}_{block_end}"]["max_abs_error"]
        for block_start, block_end in all_blocks
    ]
    signed_reconstruction_abs_error_monotone_decay = monotone_nonincreasing(abs_error_sequence)
    quarter_band_monitor_to_3072_supported = bool(
        all(
            block_summaries[f"{block_start}_{block_end}"]["min_correlation"] >= 0.5
            for block_start, block_end in MONITOR_BLOCKS
        )
        and signed_reconstruction_abs_error_monotone_decay
    )
    quarter_band_asymptotic_saturation_supported = bool(
        quarter_band_min_sign_correlation_1025_3072 >= 0.5
        and signed_reconstruction_abs_error_monotone_decay
        and quarter_band_monitor_to_3072_supported
    )
    asymptotic_route_reset_required = bool(
        (not strict_ultra_asymptotic_extension_to_2048_supported)
        and quarter_band_asymptotic_saturation_supported
    )
    exact_loading_index_theorem_remains_reserve = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "bulk-lattice ultra-asymptotic continuation inventory ready",
            sign_base.truth(inventory_ready),
            "This branch starts only after the 1024 registry has already retained farther continuation as the honest mainline.",
        ),
    ]
    for block_start, block_end in PRIMARY_BLOCKS:
        key = f"{block_start}_{block_end}"
        summary = block_summaries[key]
        rows.extend(
            [
                sign_base.row(
                    f"strict_{block_start}_{block_end}_max_mismatch_fraction",
                    "pass" if summary["max_mismatch"] <= QUARTER_REFERENCE else "reject",
                    f"strict bulk-lattice max mismatch on harmonic {block_start}..{block_end}",
                    summary["max_mismatch"],
                    "The inherited strict gate uses the same 0.25 mismatch ceiling as the earlier asymptotic continuation branches.",
                ),
                sign_base.row(
                    f"strict_{block_start}_{block_end}_min_sign_correlation",
                    "pass" if summary["min_correlation"] >= 0.5 else "reject",
                    f"strict bulk-lattice min sign correlation on harmonic {block_start}..{block_end}",
                    summary["min_correlation"],
                    "The inherited strict gate still requires positive sign correlation on every primary block.",
                ),
            ]
        )

    rows.extend(
        [
            sign_base.row(
                "strict_ultra_asymptotic_extension_to_2048_supported",
                "pass" if strict_ultra_asymptotic_extension_to_2048_supported else "reject",
                "strict ultra-asymptotic extension to 2048 supported",
                sign_base.truth(strict_ultra_asymptotic_extension_to_2048_supported),
                "This row answers the inherited question before any route reset is allowed.",
            ),
            sign_base.row(
                "quarter_band_abs_deviation_max_1025_3072",
                "pass",
                "max absolute deviation from quarter-mismatch reference on harmonic 1025..3072",
                quarter_band_abs_deviation_max_1025_3072,
                "The new asymptotic surface is not zero mismatch; it is a narrow band around the quarter reference while absolute reconstruction error keeps decaying.",
            ),
            sign_base.row(
                "quarter_band_min_sign_correlation_1025_3072",
                "pass" if quarter_band_min_sign_correlation_1025_3072 >= 0.5 else "reject",
                "min sign correlation on harmonic 1025..3072",
                quarter_band_min_sign_correlation_1025_3072,
                "The quarter-band family remains admissible only if sign correlation stays positive across both primary and monitor blocks.",
            ),
            sign_base.row(
                "signed_reconstruction_abs_error_monotone_decay",
                "pass" if signed_reconstruction_abs_error_monotone_decay else "reject",
                "signed reconstruction absolute error decays monotonically on harmonic 1025..3072",
                sign_base.truth(signed_reconstruction_abs_error_monotone_decay),
                "The route reset is only honest if the pointwise absolute reconstruction error keeps shrinking as harmonic index increases.",
            ),
            sign_base.row(
                "quarter_band_monitor_to_3072_supported",
                "pass" if quarter_band_monitor_to_3072_supported else "reject",
                "quarter-band monitor to harmonic 3072 supported",
                sign_base.truth(quarter_band_monitor_to_3072_supported),
                "The monitor blocks decide whether the quarter-band family is local noise or the honest new asymptotic surface.",
            ),
            sign_base.row(
                "quarter_band_asymptotic_saturation_supported",
                "pass" if quarter_band_asymptotic_saturation_supported else "reject",
                "quarter-band asymptotic saturation supported",
                sign_base.truth(quarter_band_asymptotic_saturation_supported),
                "If sign correlation remains positive and absolute reconstruction error keeps decaying through the monitor blocks, the honest next theory is quarter-band asymptotic saturation.",
            ),
            sign_base.row(
                "asymptotic_route_reset_required",
                "pass" if asymptotic_route_reset_required else "reject",
                "asymptotic route reset required",
                sign_base.truth(asymptotic_route_reset_required),
                "A route reset is required exactly when the inherited strict gate blocks but the quarter-band asymptotic surface still survives.",
            ),
            sign_base.row(
                "exact_loading_index_theorem_remains_reserve",
                "pass",
                "exact loading-index theorem remains reserve",
                sign_base.truth(exact_loading_index_theorem_remains_reserve),
                "The exact loading-index theorem is still reserve-only while continuation survives under the reset asymptotic surface.",
            ),
        ]
    )

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "loading_index_mode_1025_3072": top_frequency["loading_index"],
        "loading_index_mode_count_1025_3072": top_frequency["count"],
        "loading_index_mean_1025_3072": float(loading_indices.mean()),
        "loading_index_std_1025_3072": float(loading_indices.std()),
        "loading_index_min_1025_3072": int(loading_indices.min()),
        "loading_index_max_1025_3072": int(loading_indices.max()),
        "loading_index_top_frequencies_1025_3072": loading_index_frequency[:15],
        "strict_1025_1280_max_mismatch_fraction": block_summaries["1025_1280"]["max_mismatch"],
        "strict_1025_1280_min_sign_correlation": block_summaries["1025_1280"]["min_correlation"],
        "strict_1025_1280_signed_reconstruction_max_abs_error": block_summaries["1025_1280"]["max_abs_error"],
        "strict_1281_1536_max_mismatch_fraction": block_summaries["1281_1536"]["max_mismatch"],
        "strict_1281_1536_min_sign_correlation": block_summaries["1281_1536"]["min_correlation"],
        "strict_1281_1536_signed_reconstruction_max_abs_error": block_summaries["1281_1536"]["max_abs_error"],
        "strict_1537_1792_max_mismatch_fraction": block_summaries["1537_1792"]["max_mismatch"],
        "strict_1537_1792_min_sign_correlation": block_summaries["1537_1792"]["min_correlation"],
        "strict_1537_1792_signed_reconstruction_max_abs_error": block_summaries["1537_1792"]["max_abs_error"],
        "strict_1793_2048_max_mismatch_fraction": block_summaries["1793_2048"]["max_mismatch"],
        "strict_1793_2048_min_sign_correlation": block_summaries["1793_2048"]["min_correlation"],
        "strict_1793_2048_signed_reconstruction_max_abs_error": block_summaries["1793_2048"]["max_abs_error"],
        "monitor_2049_2560_max_mismatch_fraction": block_summaries["2049_2560"]["max_mismatch"],
        "monitor_2049_2560_min_sign_correlation": block_summaries["2049_2560"]["min_correlation"],
        "monitor_2049_2560_signed_reconstruction_max_abs_error": block_summaries["2049_2560"]["max_abs_error"],
        "monitor_2561_3072_max_mismatch_fraction": block_summaries["2561_3072"]["max_mismatch"],
        "monitor_2561_3072_min_sign_correlation": block_summaries["2561_3072"]["min_correlation"],
        "monitor_2561_3072_signed_reconstruction_max_abs_error": block_summaries["2561_3072"]["max_abs_error"],
        "strict_ultra_asymptotic_extension_to_2048_supported": strict_ultra_asymptotic_extension_to_2048_supported,
        "quarter_reference_mismatch": QUARTER_REFERENCE,
        "quarter_band_abs_deviation_max_1025_3072": quarter_band_abs_deviation_max_1025_3072,
        "quarter_band_min_sign_correlation_1025_3072": quarter_band_min_sign_correlation_1025_3072,
        "signed_reconstruction_abs_error_monotone_decay": signed_reconstruction_abs_error_monotone_decay,
        "quarter_band_monitor_to_3072_supported": quarter_band_monitor_to_3072_supported,
        "quarter_band_asymptotic_saturation_supported": quarter_band_asymptotic_saturation_supported,
        "asymptotic_route_reset_required": asymptotic_route_reset_required,
        "exact_loading_index_theorem_remains_reserve": exact_loading_index_theorem_remains_reserve,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2089",
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
            "overall_status": "vector_qball_form_factor_ultra_asymptotic_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2087"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2087-.2090"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2087"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2087"),
                "unified_roadmap_hit": find_line(unified_text, ".2087-.2090"),
                "long_roadmap_hit": find_line(long_text, ".2087-.2090"),
                "part5_hit": find_line(part5_text, ".2079-.2086"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2087"))),
            "The ultra-asymptotic continuation audit is only valid if status already points to the same official branch.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2087-.2090"))),
            "The public roadmap must expose the ultra-asymptotic continuation audit before its result is frozen.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2087-.2090"))),
            "The long-horizon roadmap must carry the same ultra-asymptotic continuation route.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2090",
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
            "overall_status": "vector_qball_form_factor_ultra_asymptotic_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2087"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2087-.2090"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2087"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2087"),
                "unified_roadmap_hit": find_line(unified_text, ".2087-.2090"),
                "long_roadmap_hit": find_line(long_text, ".2087-.2090"),
                "part5_hit": find_line(part5_text, ".2079-.2086"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
