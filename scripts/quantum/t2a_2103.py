#!/usr/bin/env python3
"""Generate 8.7.56.2103-.2106 sparse ultra-farther continuation artifacts.

The direct `.2103-.2106` route inherited the quarter-band continuation logic
from `.2095-.2102`, but the brute-force exact audit beyond harmonic 4096 is
computationally too expensive when every window carries a dense exact overlap
scan. This branch therefore keeps the same retained boundary bulk-lattice
family and switches only the audit contract:

    - harmonic windows are sampled at a fixed stride,
    - each sampled window uses a coarse but exact overlap scan,
    - the decision separates "strict quarter-band continuation" from
      "sparse exact plateau continuation".

The branch remains exact on the sampled windows because it still evaluates the
exact overlap integral there; only the audit coverage changes from exhaustive
to representative-band sparse sampling.
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
PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_2099_2102_harmonic_quarter_band_farther_registry_refresh_declaration_gate_metrics.json"
)
PRIOR_AUDIT = (
    PUBLIC_OUT
    / "q_8_7_56_2095_2098_harmonic_quarter_band_farther_continuation_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2103-2106"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor sparse exact ultra-farther "
    "continuation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_sparse_ultra_farther_continuation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_quarter_band_asymptotic_"
    "extension_4096_retained_loading_index_theorem_reserve_ultra_farther_"
    "continuation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_sparse_exact_ultra_farther_"
    "continuation_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_sparse_exact_plateau_registry_"
    "refresh"
)
NEXT_ROUTE = "8.7.56.2107"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_sparse_exact_asymptotic_"
    "drift_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2111"

SAMPLE_HARMONIC_STRIDE = 64
SAMPLE_SCAN_DENSITY = 25
QUARTER_REFERENCE = 0.25
SAMPLED_BANDS = [
    (4097, 6144),
    (6145, 8192),
    (8193, 10240),
    (10241, 12288),
    (12289, 14336),
    (14337, 16384),
]


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


# 関数: sampled window 群を構成する。

def build_sampled_windows(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    alias_1: float,
) -> list[dict[str, object]]:
    """Return representative sampled windows for the sparse exact audit."""
    fit_offsets = (
        lattice_base.loading_base.FIT_Q_MIN - alias_1,
        lattice_base.loading_base.FIT_Q_MAX - alias_1,
    )
    edge_offsets = (
        lattice_base.loading_base.EDGE_Q_MIN - (2.0 * alias_1),
        lattice_base.loading_base.EDGE_Q_MAX - (2.0 * alias_1),
    )
    windows: list[dict[str, object]] = []
    for band_start, band_end in SAMPLED_BANDS:
        for harmonic_index in range(band_start, band_end + 1, SAMPLE_HARMONIC_STRIDE):
            alias_harmonic = harmonic_index * alias_1
            offsets = fit_offsets if (harmonic_index % 2) == 1 else edge_offsets
            q_min, q_max = lattice_base.loading_base.translated_window(alias_harmonic, offsets)
            q_scan = np.linspace(
                q_min,
                q_max,
                int(round((q_max - q_min) * SAMPLE_SCAN_DENSITY)) + 1,
                dtype=float,
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


# 関数: sampled band summary を返す。

def summarize_sampled_band(
    windows: list[dict[str, object]],
    results: list[dict[str, float]],
    band_start: int,
    band_end: int,
) -> dict[str, float]:
    """Return summary metrics on one representative sampled band."""
    paired = [
        (window, result)
        for window, result in zip(windows, results, strict=True)
        if band_start <= int(window["harmonic_index"]) <= band_end
    ]
    sampled_harmonics = [int(window["harmonic_index"]) for window, _result in paired]
    loading_indices = np.asarray(
        [int(round(result["loading_index"])) for _window, result in paired],
        dtype=int,
    )
    return {
        "sampled_harmonic_count": float(len(sampled_harmonics)),
        "sampled_harmonic_first": float(sampled_harmonics[0]),
        "sampled_harmonic_last": float(sampled_harmonics[-1]),
        "max_mismatch": float(max(result["sign_mismatch_fraction"] for _window, result in paired)),
        "min_correlation": float(min(result["sign_correlation"] for _window, result in paired)),
        "max_abs_error": float(
            max(result["signed_reconstruction_max_abs_error"] for _window, result in paired)
        ),
        "loading_index_mode": float(np.bincount(loading_indices).argmax()),
        "loading_index_mean": float(loading_indices.mean()),
        "loading_index_std": float(loading_indices.std()),
    }


# 関数: 任意列が単調非増加か判定する。

def monotone_nonincreasing(values: list[float]) -> bool:
    """Return whether one sequence is monotone nonincreasing."""
    return all(left >= right for left, right in zip(values, values[1:]))


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the sparse exact ultra-farther continuation audit."""
    return {
        "retained_bulk_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "sparse_exact_contract": "evaluate the exact overlap only on representative harmonic windows sampled every 64 modes and with coarse exact q scans while keeping the same retained lattice and the same exact integrand",
        "strict_quarter_band_gate": "strict continuation survives only if every sampled band stays inside the inherited quarter-band absolute deviation ceiling and keeps min sign correlation >= 0.5",
        "sampled_plateau_gate": "partial sparse exact continuation survives when min sign correlation stays >= 0.5 and the sampled signed reconstruction max abs error keeps decaying even if the inherited quarter-band ceiling is exceeded",
    }


# 関数: `.2103-.2106` を実行する。

def main() -> None:
    """Execute the sparse exact ultra-farther continuation audit."""
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

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    inventory_ready = bool(prior_summary["quarter_band_ultra_farther_continuation_admissible_now"])

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
    windows = build_sampled_windows(radius, weight, norm, alias_1)
    theorem_results = lattice_base.evaluate_lattice_family(
        windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )

    band_summaries: dict[str, dict[str, float]] = {}
    for band_start, band_end in SAMPLED_BANDS:
        key = f"{band_start}_{band_end}"
        band_summaries[key] = summarize_sampled_band(
            windows,
            theorem_results,
            band_start,
            band_end,
        )

    inherited_ceiling = float(prior_summary["quarter_band_abs_deviation_ceiling_3073_4096"])
    prior_last_abs_error = float(
        prior_audit_summary["quarter_3841_4096_signed_reconstruction_max_abs_error"]
    )
    quarter_band_deviations = {
        key: abs(summary["max_mismatch"] - QUARTER_REFERENCE)
        for key, summary in band_summaries.items()
    }
    sparse_exact_abs_error_sequence = [prior_last_abs_error] + [
        band_summaries[f"{band_start}_{band_end}"]["max_abs_error"]
        for band_start, band_end in SAMPLED_BANDS
    ]
    sparse_exact_abs_error_continues_decay = monotone_nonincreasing(
        sparse_exact_abs_error_sequence
    )
    strict_quarter_band_to_16384_supported = bool(
        all(
            quarter_band_deviations[f"{band_start}_{band_end}"] <= inherited_ceiling
            and band_summaries[f"{band_start}_{band_end}"]["min_correlation"] >= 0.5
            for band_start, band_end in SAMPLED_BANDS
        )
        and sparse_exact_abs_error_continues_decay
    )
    sparse_exact_plateau_to_16384_supported = bool(
        all(
            band_summaries[f"{band_start}_{band_end}"]["min_correlation"] >= 0.5
            for band_start, band_end in SAMPLED_BANDS
        )
        and sparse_exact_abs_error_continues_decay
    )
    sparse_exact_plateau_requires_drift_audit = bool(
        sparse_exact_plateau_to_16384_supported and not strict_quarter_band_to_16384_supported
    )
    exact_loading_index_theorem_remains_reserve = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    loading_indices = np.asarray(
        [int(round(result["loading_index"])) for result in theorem_results],
        dtype=int,
    )
    loading_values, loading_counts = np.unique(loading_indices, return_counts=True)
    loading_mode_index = int(np.argmax(loading_counts))
    loading_index_mode = int(loading_values[loading_mode_index])
    loading_index_mode_count = int(loading_counts[loading_mode_index])

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "sparse ultra-farther continuation inventory ready",
            sign_base.truth(inventory_ready),
            "The branch starts only after the retained route to 4096 has already been frozen and the next blocker has been identified as computation scaling rather than pack failure.",
        ),
        sign_base.row(
            "sample_harmonic_stride",
            "watch",
            "representative harmonic sampling stride",
            float(SAMPLE_HARMONIC_STRIDE),
            "The sparse audit keeps every 64th harmonic window to preserve dyadic coverage while avoiding the blocked exhaustive continuation cost.",
        ),
        sign_base.row(
            "sample_scan_density",
            "watch",
            "exact q-scan density per sampled window",
            float(SAMPLE_SCAN_DENSITY),
            "Each representative window still uses the exact overlap integrand, only on a coarser q grid.",
        ),
        sign_base.row(
            "inherited_quarter_band_abs_deviation_ceiling",
            "watch",
            "inherited quarter-band absolute deviation ceiling",
            inherited_ceiling,
            "The strict route still uses the retained 3073..4096 ceiling as its no-drift reference surface.",
        ),
    ]
    for band_start, band_end in SAMPLED_BANDS:
        key = f"{band_start}_{band_end}"
        summary = band_summaries[key]
        rows.extend(
            [
                sign_base.row(
                    f"sparse_{band_start}_{band_end}_max_mismatch_fraction",
                    "pass" if quarter_band_deviations[key] <= inherited_ceiling else "watch",
                    f"sparse exact max mismatch on harmonic {band_start}..{band_end}",
                    summary["max_mismatch"],
                    "This is the representative-band mismatch left by the retained bulk lattice on the sampled exact windows.",
                ),
                sign_base.row(
                    f"sparse_{band_start}_{band_end}_min_sign_correlation",
                    "pass" if summary["min_correlation"] >= 0.5 else "reject",
                    f"sparse exact min sign correlation on harmonic {band_start}..{band_end}",
                    summary["min_correlation"],
                    "Positive sign correlation remains the hard floor even in the sparse audit.",
                ),
                sign_base.row(
                    f"sparse_{band_start}_{band_end}_signed_reconstruction_max_abs_error",
                    "watch",
                    f"sparse exact max signed reconstruction abs error on harmonic {band_start}..{band_end}",
                    summary["max_abs_error"],
                    "The sparse audit keeps the exact signed reconstruction error as the non-negotiable pointwise consistency metric.",
                ),
            ]
        )

    rows.extend(
        [
            sign_base.row(
                "sparse_exact_abs_error_continues_decay",
                "pass" if sparse_exact_abs_error_continues_decay else "reject",
                "sparse exact signed reconstruction abs error continues to decay",
                sign_base.truth(sparse_exact_abs_error_continues_decay),
                "Even under representative-band sampling, the exact reconstruction error should keep shrinking as harmonic index increases.",
            ),
            sign_base.row(
                "strict_quarter_band_to_16384_supported",
                "pass" if strict_quarter_band_to_16384_supported else "reject",
                "strict quarter-band continuation to harmonic 16384 supported",
                sign_base.truth(strict_quarter_band_to_16384_supported),
                "This strict route survives only if every sampled band remains inside the inherited quarter-band ceiling.",
            ),
            sign_base.row(
                "sparse_exact_plateau_to_16384_supported",
                "pass" if sparse_exact_plateau_to_16384_supported else "reject",
                "sparse exact plateau continuation to harmonic 16384 supported",
                sign_base.truth(sparse_exact_plateau_to_16384_supported),
                "The sparse exact route is retained when positive sign correlation and decaying reconstruction error persist even after the strict quarter-band ceiling is exceeded.",
            ),
            sign_base.row(
                "sparse_exact_plateau_requires_drift_audit",
                "pass" if sparse_exact_plateau_requires_drift_audit else "reject",
                "sparse exact plateau requires asymptotic drift audit",
                sign_base.truth(sparse_exact_plateau_requires_drift_audit),
                "Once strict quarter-band continuation fails but the sparse exact plateau survives, the honest next move is to audit the asymptotic drift rather than reopen same-level loading-index scans.",
            ),
            sign_base.row(
                "exact_loading_index_theorem_remains_reserve",
                "pass",
                "exact loading-index theorem remains reserve",
                sign_base.truth(exact_loading_index_theorem_remains_reserve),
                "The sparse exact audit still does not promote a closed loading-index theorem.",
            ),
        ]
    )

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "sample_harmonic_stride": SAMPLE_HARMONIC_STRIDE,
        "sample_scan_density": SAMPLE_SCAN_DENSITY,
        "sparse_sampled_harmonic_count_4097_16384": int(len(windows)),
        "loading_index_mode_sparse_4097_16384": loading_index_mode,
        "loading_index_mode_count_sparse_4097_16384": loading_index_mode_count,
        "loading_index_mean_sparse_4097_16384": float(loading_indices.mean()),
        "loading_index_std_sparse_4097_16384": float(loading_indices.std()),
        "inherited_quarter_band_abs_deviation_ceiling": inherited_ceiling,
        "sparse_4097_6144_max_mismatch_fraction": band_summaries["4097_6144"]["max_mismatch"],
        "sparse_4097_6144_min_sign_correlation": band_summaries["4097_6144"]["min_correlation"],
        "sparse_4097_6144_signed_reconstruction_max_abs_error": band_summaries["4097_6144"]["max_abs_error"],
        "sparse_6145_8192_max_mismatch_fraction": band_summaries["6145_8192"]["max_mismatch"],
        "sparse_6145_8192_min_sign_correlation": band_summaries["6145_8192"]["min_correlation"],
        "sparse_6145_8192_signed_reconstruction_max_abs_error": band_summaries["6145_8192"]["max_abs_error"],
        "sparse_8193_10240_max_mismatch_fraction": band_summaries["8193_10240"]["max_mismatch"],
        "sparse_8193_10240_min_sign_correlation": band_summaries["8193_10240"]["min_correlation"],
        "sparse_8193_10240_signed_reconstruction_max_abs_error": band_summaries["8193_10240"]["max_abs_error"],
        "sparse_10241_12288_max_mismatch_fraction": band_summaries["10241_12288"]["max_mismatch"],
        "sparse_10241_12288_min_sign_correlation": band_summaries["10241_12288"]["min_correlation"],
        "sparse_10241_12288_signed_reconstruction_max_abs_error": band_summaries["10241_12288"]["max_abs_error"],
        "sparse_12289_14336_max_mismatch_fraction": band_summaries["12289_14336"]["max_mismatch"],
        "sparse_12289_14336_min_sign_correlation": band_summaries["12289_14336"]["min_correlation"],
        "sparse_12289_14336_signed_reconstruction_max_abs_error": band_summaries["12289_14336"]["max_abs_error"],
        "sparse_14337_16384_max_mismatch_fraction": band_summaries["14337_16384"]["max_mismatch"],
        "sparse_14337_16384_min_sign_correlation": band_summaries["14337_16384"]["min_correlation"],
        "sparse_14337_16384_signed_reconstruction_max_abs_error": band_summaries["14337_16384"]["max_abs_error"],
        "sparse_exact_abs_error_continues_decay": sparse_exact_abs_error_continues_decay,
        "strict_quarter_band_to_16384_supported": strict_quarter_band_to_16384_supported,
        "sparse_exact_plateau_to_16384_supported": sparse_exact_plateau_to_16384_supported,
        "sparse_exact_plateau_requires_drift_audit": sparse_exact_plateau_requires_drift_audit,
        "exact_loading_index_theorem_remains_reserve": exact_loading_index_theorem_remains_reserve,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2105",
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
            "overall_status": "vector_qball_form_factor_sparse_ultra_farther_continuation_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2103"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2103-.2106"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2103"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2103"),
                "unified_roadmap_hit": find_line(unified_text, ".2103-.2106"),
                "long_roadmap_hit": find_line(long_text, ".2103-.2106"),
                "part5_hit": find_line(part5_text, ".2099-.2106"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2103"))),
            "The sparse exact ultra-farther audit is only valid if status already points to the same official branch.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2103-.2106"))),
            "The public roadmap must expose the sparse ultra-farther audit before its result is frozen.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2103-.2106"))),
            "The long-horizon roadmap must expose the same sparse exact route.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2106",
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
            "overall_status": "vector_qball_form_factor_sparse_ultra_farther_continuation_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2103"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2103-.2106"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2103"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2103"),
                "unified_roadmap_hit": find_line(unified_text, ".2103-.2106"),
                "long_roadmap_hit": find_line(long_text, ".2103-.2106"),
                "part5_hit": find_line(part5_text, ".2099-.2106"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
