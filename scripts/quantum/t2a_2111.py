#!/usr/bin/env python3
"""Generate 8.7.56.2111-.2114 sparse exact asymptotic drift artifacts.

The retained boundary bulk-lattice family survives as a sparse exact plateau
through harmonic 16384, but the previous branch explicitly deferred the honest
question of where that plateau begins to drift. This branch keeps the same
retained lattice, the same exact overlap integrand, and the same representative
window logic, then pushes the sparse audit farther.
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
    / "q_8_7_56_2107_2110_harmonic_sparse_plateau_registry_refresh_declaration_gate_metrics.json"
)
PRIOR_AUDIT = (
    PUBLIC_OUT
    / "q_8_7_56_2103_2106_harmonic_sparse_ultra_farther_continuation_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2111-2114"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor sparse exact asymptotic "
    "drift audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_sparse_asymptotic_drift_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_sparse_exact_plateau_to_16384_"
    "partial_retain_asymptotic_drift_audit_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_sparse_exact_asymptotic_"
    "drift_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_sparse_exact_plateau_drift_"
    "registry_refresh"
)
NEXT_ROUTE = "8.7.56.2115"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_sparse_exact_drift_law_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2119"

PRIMARY_SAMPLE_HARMONIC_STRIDE = 128
STRESS_SAMPLE_HARMONIC_STRIDE = 256
SAMPLE_SCAN_DENSITY = 25
PLATEAU_SIGN_CORRELATION_FLOOR = 0.5
PRIMARY_BANDS = [
    (16385, 20480),
    (20481, 24576),
    (24577, 28672),
    (28673, 32768),
]
STRESS_BANDS = [
    (32769, 40960),
    (40961, 49152),
    (49153, 57344),
    (57345, 65536),
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


# 関数: farther sampled window 群を構成する。

def build_sampled_windows(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    alias_1: float,
    bands: list[tuple[int, int]],
    harmonic_stride: int,
) -> list[dict[str, object]]:
    """Return representative farther harmonic windows under one sparse contract."""
    fit_offsets = (
        lattice_base.loading_base.FIT_Q_MIN - alias_1,
        lattice_base.loading_base.FIT_Q_MAX - alias_1,
    )
    edge_offsets = (
        lattice_base.loading_base.EDGE_Q_MIN - (2.0 * alias_1),
        lattice_base.loading_base.EDGE_Q_MAX - (2.0 * alias_1),
    )
    windows: list[dict[str, object]] = []
    for band_start, band_end in bands:
        for harmonic_index in range(band_start, band_end + 1, harmonic_stride):
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


# 関数: one sampled band の summary を返す。

def summarize_sampled_band(
    windows: list[dict[str, object]],
    results: list[dict[str, float]],
    band_start: int,
    band_end: int,
) -> dict[str, float]:
    """Return summary metrics on one representative farther band."""
    paired = [
        (window, result)
        for window, result in zip(windows, results, strict=True)
        if band_start <= int(window["harmonic_index"]) <= band_end
    ]
    loading_indices = np.asarray(
        [int(round(result["loading_index"])) for _window, result in paired],
        dtype=int,
    )
    return {
        "sampled_harmonic_count": float(len(paired)),
        "sampled_harmonic_first": float(int(paired[0][0]["harmonic_index"])),
        "sampled_harmonic_last": float(int(paired[-1][0]["harmonic_index"])),
        "max_mismatch": float(max(result["sign_mismatch_fraction"] for _window, result in paired)),
        "min_correlation": float(min(result["sign_correlation"] for _window, result in paired)),
        "max_abs_error": float(
            max(result["signed_reconstruction_max_abs_error"] for _window, result in paired)
        ),
        "loading_index_mode": float(np.bincount(loading_indices).argmax()),
        "loading_index_mean": float(loading_indices.mean()),
        "loading_index_std": float(loading_indices.std()),
    }


# 関数: 数列が単調非増加か判定する。

def monotone_nonincreasing(values: list[float]) -> bool:
    """Return whether one sequence is monotone nonincreasing."""
    return all(left >= right for left, right in zip(values, values[1:]))


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the sparse exact asymptotic drift audit."""
    return {
        "retained_bulk_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "sparse_primary_contract": "evaluate the same retained lattice on representative exact windows for harmonic 16385..32768 with stride 128 and q-density 25",
        "sparse_stress_contract": "evaluate the same retained lattice on representative exact windows for harmonic 32769..65536 with stride 256 and q-density 25",
        "plateau_retain_rule": "retain farther sparse plateau while min sign correlation >= 0.5 and the sampled signed reconstruction max abs error keeps decaying",
        "drift_rule": "declare asymptotic drift once farther max mismatch rises above the inherited sparse plateau ceiling from harmonic 4097..16384",
    }


# 関数: `.2111-.2114` を実行する。

def main() -> None:
    """Execute the sparse exact asymptotic drift audit."""
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
    inventory_ready = bool(prior_summary["sparse_exact_asymptotic_drift_audit_admissible_now"])

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
    inherited_sparse_plateau_ceiling = max(
        float(prior_audit_summary[f"sparse_{band_start}_{band_end}_max_mismatch_fraction"])
        for band_start, band_end in [
            (4097, 6144),
            (6145, 8192),
            (8193, 10240),
            (10241, 12288),
            (12289, 14336),
            (14337, 16384),
        ]
    )
    prior_last_abs_error = float(
        prior_audit_summary["sparse_14337_16384_signed_reconstruction_max_abs_error"]
    )

    primary_windows = build_sampled_windows(
        radius,
        weight,
        norm,
        alias_1,
        PRIMARY_BANDS,
        PRIMARY_SAMPLE_HARMONIC_STRIDE,
    )
    primary_results = lattice_base.evaluate_lattice_family(
        primary_windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )
    stress_windows = build_sampled_windows(
        radius,
        weight,
        norm,
        alias_1,
        STRESS_BANDS,
        STRESS_SAMPLE_HARMONIC_STRIDE,
    )
    stress_results = lattice_base.evaluate_lattice_family(
        stress_windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )

    primary_summaries = {
        f"{band_start}_{band_end}": summarize_sampled_band(
            primary_windows,
            primary_results,
            band_start,
            band_end,
        )
        for band_start, band_end in PRIMARY_BANDS
    }
    stress_summaries = {
        f"{band_start}_{band_end}": summarize_sampled_band(
            stress_windows,
            stress_results,
            band_start,
            band_end,
        )
        for band_start, band_end in STRESS_BANDS
    }

    combined_abs_error_sequence = [prior_last_abs_error]
    combined_abs_error_sequence.extend(
        primary_summaries[f"{band_start}_{band_end}"]["max_abs_error"]
        for band_start, band_end in PRIMARY_BANDS
    )
    combined_abs_error_sequence.extend(
        stress_summaries[f"{band_start}_{band_end}"]["max_abs_error"]
        for band_start, band_end in STRESS_BANDS
    )
    sparse_exact_abs_error_continues_decay = monotone_nonincreasing(
        combined_abs_error_sequence
    )
    sparse_exact_asymptotic_drift_detected = bool(
        any(
            primary_summaries[f"{band_start}_{band_end}"]["max_mismatch"]
            > inherited_sparse_plateau_ceiling
            for band_start, band_end in PRIMARY_BANDS
        )
    )
    farther_sparse_plateau_to_32768_supported = bool(
        all(
            primary_summaries[f"{band_start}_{band_end}"]["min_correlation"]
            >= PLATEAU_SIGN_CORRELATION_FLOOR
            for band_start, band_end in PRIMARY_BANDS
        )
        and sparse_exact_abs_error_continues_decay
    )
    stress_sparse_plateau_to_57344_supported = bool(
        all(
            stress_summaries[f"{band_start}_{band_end}"]["min_correlation"]
            >= PLATEAU_SIGN_CORRELATION_FLOOR
            for band_start, band_end in STRESS_BANDS[:-1]
        )
        and sparse_exact_abs_error_continues_decay
    )
    stress_sign_floor_break_57345_65536 = bool(
        stress_summaries["57345_65536"]["min_correlation"] < PLATEAU_SIGN_CORRELATION_FLOOR
    )
    exact_loading_index_theorem_remains_reserve = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    all_loading_indices = np.asarray(
        [int(round(result["loading_index"])) for result in primary_results + stress_results],
        dtype=int,
    )
    loading_values, loading_counts = np.unique(all_loading_indices, return_counts=True)
    loading_mode_index = int(np.argmax(loading_counts))
    loading_index_mode = int(loading_values[loading_mode_index])
    loading_index_mode_count = int(loading_counts[loading_mode_index])

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "sparse exact asymptotic drift inventory ready",
            sign_base.truth(inventory_ready),
            "The drift audit starts only after the sparse exact plateau has already been frozen through harmonic 16384.",
        ),
        sign_base.row(
            "primary_sample_harmonic_stride",
            "watch",
            "primary farther representative harmonic stride",
            float(PRIMARY_SAMPLE_HARMONIC_STRIDE),
            "The first farther continuation audit keeps every 128th harmonic window while preserving exact overlap evaluation on each sampled window.",
        ),
        sign_base.row(
            "stress_sample_harmonic_stride",
            "watch",
            "stress farther representative harmonic stride",
            float(STRESS_SAMPLE_HARMONIC_STRIDE),
            "The stress continuation audit keeps every 256th harmonic window to push farther while respecting the computation-scaling reset.",
        ),
        sign_base.row(
            "sample_scan_density",
            "watch",
            "exact q-scan density per sampled farther window",
            float(SAMPLE_SCAN_DENSITY),
            "Each sampled farther window still uses the exact overlap integrand on a coarse but exact q grid.",
        ),
        sign_base.row(
            "inherited_sparse_plateau_ceiling_4097_16384",
            "watch",
            "inherited sparse plateau max mismatch ceiling from harmonic 4097..16384",
            inherited_sparse_plateau_ceiling,
            "The drift audit uses the previous sparse exact plateau ceiling as its retained no-drift reference surface.",
        ),
    ]

    for band_start, band_end in PRIMARY_BANDS:
        key = f"{band_start}_{band_end}"
        summary = primary_summaries[key]
        rows.extend(
            [
                sign_base.row(
                    f"primary_{band_start}_{band_end}_max_mismatch_fraction",
                    "pass"
                    if summary["max_mismatch"] <= inherited_sparse_plateau_ceiling
                    else "watch",
                    f"primary farther sparse max mismatch on harmonic {band_start}..{band_end}",
                    summary["max_mismatch"],
                    "This is the farther sparse mismatch left by the retained bulk lattice on the primary continuation bands.",
                ),
                sign_base.row(
                    f"primary_{band_start}_{band_end}_min_sign_correlation",
                    "pass"
                    if summary["min_correlation"] >= PLATEAU_SIGN_CORRELATION_FLOOR
                    else "reject",
                    f"primary farther sparse min sign correlation on harmonic {band_start}..{band_end}",
                    summary["min_correlation"],
                    "Positive sign correlation remains the hard retention floor on every primary farther band.",
                ),
                sign_base.row(
                    f"primary_{band_start}_{band_end}_signed_reconstruction_max_abs_error",
                    "watch",
                    f"primary farther sparse max signed reconstruction abs error on harmonic {band_start}..{band_end}",
                    summary["max_abs_error"],
                    "The exact signed reconstruction error must keep decaying even after the mismatch ceiling starts to drift upward.",
                ),
            ]
        )

    for band_start, band_end in STRESS_BANDS:
        key = f"{band_start}_{band_end}"
        summary = stress_summaries[key]
        rows.extend(
            [
                sign_base.row(
                    f"stress_{band_start}_{band_end}_max_mismatch_fraction",
                    "watch",
                    f"stress farther sparse max mismatch on harmonic {band_start}..{band_end}",
                    summary["max_mismatch"],
                    "The stress bands track whether the retained sparse plateau eventually bends away from the inherited ceiling.",
                ),
                sign_base.row(
                    f"stress_{band_start}_{band_end}_min_sign_correlation",
                    "pass"
                    if summary["min_correlation"] >= PLATEAU_SIGN_CORRELATION_FLOOR
                    else "reject",
                    f"stress farther sparse min sign correlation on harmonic {band_start}..{band_end}",
                    summary["min_correlation"],
                    "The first correlation-floor break on the stress audit is the honest point where the farther sparse plateau stops being retainable.",
                ),
                sign_base.row(
                    f"stress_{band_start}_{band_end}_signed_reconstruction_max_abs_error",
                    "watch",
                    f"stress farther sparse max signed reconstruction abs error on harmonic {band_start}..{band_end}",
                    summary["max_abs_error"],
                    "Even on the stress bands, the exact signed reconstruction error remains tiny and keeps decaying.",
                ),
            ]
        )

    rows.extend(
        [
            sign_base.row(
                "sparse_exact_abs_error_continues_decay",
                "pass" if sparse_exact_abs_error_continues_decay else "reject",
                "sparse exact signed reconstruction abs error continues to decay through harmonic 65536",
                sign_base.truth(sparse_exact_abs_error_continues_decay),
                "The retained sparse exact family continues to improve pointwise even while the mismatch ceiling drifts.",
            ),
            sign_base.row(
                "sparse_exact_asymptotic_drift_detected",
                "pass" if sparse_exact_asymptotic_drift_detected else "reject",
                "sparse exact asymptotic drift detected beyond harmonic 16384",
                sign_base.truth(sparse_exact_asymptotic_drift_detected),
                "Drift is declared once the farther primary bands exceed the inherited sparse plateau ceiling.",
            ),
            sign_base.row(
                "farther_sparse_plateau_to_32768_supported",
                "pass" if farther_sparse_plateau_to_32768_supported else "reject",
                "farther sparse plateau retained through harmonic 32768",
                sign_base.truth(farther_sparse_plateau_to_32768_supported),
                "The retained sparse plateau is still honest through the primary farther bands because sign correlation stays above the floor.",
            ),
            sign_base.row(
                "stress_sparse_plateau_to_57344_supported",
                "pass" if stress_sparse_plateau_to_57344_supported else "reject",
                "stress sparse plateau retained through harmonic 57344",
                sign_base.truth(stress_sparse_plateau_to_57344_supported),
                "The first three stress bands still satisfy the positive-correlation floor, so the farther plateau remains honest through harmonic 57344.",
            ),
            sign_base.row(
                "stress_sign_floor_break_57345_65536",
                "pass" if stress_sign_floor_break_57345_65536 else "reject",
                "first sparse sign-correlation floor break appears on harmonic 57345..65536",
                sign_base.truth(stress_sign_floor_break_57345_65536),
                "The last stress band is the first sampled farther band where the retained sparse plateau drops below the 0.5 sign-correlation floor.",
            ),
            sign_base.row(
                "exact_loading_index_theorem_remains_reserve",
                "pass",
                "exact loading-index theorem remains reserve",
                sign_base.truth(exact_loading_index_theorem_remains_reserve),
                "The farther drift audit still does not promote a closed loading-index theorem.",
            ),
        ]
    )

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "primary_sample_harmonic_stride": PRIMARY_SAMPLE_HARMONIC_STRIDE,
        "stress_sample_harmonic_stride": STRESS_SAMPLE_HARMONIC_STRIDE,
        "sample_scan_density": SAMPLE_SCAN_DENSITY,
        "inherited_sparse_plateau_ceiling_4097_16384": inherited_sparse_plateau_ceiling,
        "farther_sampled_harmonic_count_16385_65536": int(
            len(primary_windows) + len(stress_windows)
        ),
        "loading_index_mode_sparse_16385_65536": loading_index_mode,
        "loading_index_mode_count_sparse_16385_65536": loading_index_mode_count,
        "loading_index_mean_sparse_16385_65536": float(all_loading_indices.mean()),
        "loading_index_std_sparse_16385_65536": float(all_loading_indices.std()),
        "primary_16385_20480_max_mismatch_fraction": primary_summaries["16385_20480"]["max_mismatch"],
        "primary_16385_20480_min_sign_correlation": primary_summaries["16385_20480"]["min_correlation"],
        "primary_16385_20480_signed_reconstruction_max_abs_error": primary_summaries["16385_20480"]["max_abs_error"],
        "primary_20481_24576_max_mismatch_fraction": primary_summaries["20481_24576"]["max_mismatch"],
        "primary_20481_24576_min_sign_correlation": primary_summaries["20481_24576"]["min_correlation"],
        "primary_20481_24576_signed_reconstruction_max_abs_error": primary_summaries["20481_24576"]["max_abs_error"],
        "primary_24577_28672_max_mismatch_fraction": primary_summaries["24577_28672"]["max_mismatch"],
        "primary_24577_28672_min_sign_correlation": primary_summaries["24577_28672"]["min_correlation"],
        "primary_24577_28672_signed_reconstruction_max_abs_error": primary_summaries["24577_28672"]["max_abs_error"],
        "primary_28673_32768_max_mismatch_fraction": primary_summaries["28673_32768"]["max_mismatch"],
        "primary_28673_32768_min_sign_correlation": primary_summaries["28673_32768"]["min_correlation"],
        "primary_28673_32768_signed_reconstruction_max_abs_error": primary_summaries["28673_32768"]["max_abs_error"],
        "stress_32769_40960_max_mismatch_fraction": stress_summaries["32769_40960"]["max_mismatch"],
        "stress_32769_40960_min_sign_correlation": stress_summaries["32769_40960"]["min_correlation"],
        "stress_32769_40960_signed_reconstruction_max_abs_error": stress_summaries["32769_40960"]["max_abs_error"],
        "stress_40961_49152_max_mismatch_fraction": stress_summaries["40961_49152"]["max_mismatch"],
        "stress_40961_49152_min_sign_correlation": stress_summaries["40961_49152"]["min_correlation"],
        "stress_40961_49152_signed_reconstruction_max_abs_error": stress_summaries["40961_49152"]["max_abs_error"],
        "stress_49153_57344_max_mismatch_fraction": stress_summaries["49153_57344"]["max_mismatch"],
        "stress_49153_57344_min_sign_correlation": stress_summaries["49153_57344"]["min_correlation"],
        "stress_49153_57344_signed_reconstruction_max_abs_error": stress_summaries["49153_57344"]["max_abs_error"],
        "stress_57345_65536_max_mismatch_fraction": stress_summaries["57345_65536"]["max_mismatch"],
        "stress_57345_65536_min_sign_correlation": stress_summaries["57345_65536"]["min_correlation"],
        "stress_57345_65536_signed_reconstruction_max_abs_error": stress_summaries["57345_65536"]["max_abs_error"],
        "sparse_exact_abs_error_continues_decay": sparse_exact_abs_error_continues_decay,
        "sparse_exact_asymptotic_drift_detected": sparse_exact_asymptotic_drift_detected,
        "farther_sparse_plateau_to_32768_supported": farther_sparse_plateau_to_32768_supported,
        "stress_sparse_plateau_to_57344_supported": stress_sparse_plateau_to_57344_supported,
        "stress_sign_floor_break_57345_65536": stress_sign_floor_break_57345_65536,
        "exact_loading_index_theorem_remains_reserve": exact_loading_index_theorem_remains_reserve,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2113",
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
            "overall_status": "vector_qball_form_factor_sparse_asymptotic_drift_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2111"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2111-.2114"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2111"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2111"),
                "unified_roadmap_hit": find_line(unified_text, ".2111-.2114"),
                "long_roadmap_hit": find_line(long_text, ".2111-.2114"),
                "part5_hit": find_line(part5_text, ".2107-.2114"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2111"))),
            "The sparse exact asymptotic drift audit is only valid if status already points to the same official branch.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2111-.2114"))),
            "The public roadmap must expose the sparse exact drift audit before its result is frozen.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2111-.2114"))),
            "The long-horizon roadmap must expose the same sparse exact drift route.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2114",
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
            "overall_status": "vector_qball_form_factor_sparse_asymptotic_drift_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2111"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2111-.2114"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2111"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2111"),
                "unified_roadmap_hit": find_line(unified_text, ".2111-.2114"),
                "long_roadmap_hit": find_line(long_text, ".2111-.2114"),
                "part5_hit": find_line(part5_text, ".2107-.2114"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
