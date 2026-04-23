#!/usr/bin/env python3
"""Materialize one source-backed hyperfine correction beyond the Fermi baseline.

Purpose:
    Trial-2 currently localizes its two-surface split to the H I 21 cm Fermi
    baseline. The next honest reopen test is whether the current public pack
    already contains one deterministic correction source beyond the tree-level
    Fermi-contact formula.

    The retained QED-vacuum source cache already includes the Gabrielse et al.
    electron g-2 PDF. This backend extracts the directly measured `g/2` token
    from that PDF and promotes the corrected hyperfine map

        nu_hfs,g2(alpha) = (g_e / 2) * nu_hfs,Fermi(alpha)

    as one source-backed deterministic correction surface.

Inputs:
    - data/quantum/sources/arxiv_0801.1134v2.pdf
    - data/quantum/sources/nist_atspec_handbook/extracted_values.json
    - scripts/quantum/trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend.py

Outputs:
    - One in-memory pack consumed by `.5959-.5962` wrappers
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend import (
    ALPHA_CODATA,
    ALPHA_COMMON,
    ALPHA_P_4D_CAN,
    ALPHA_P_4D_VERTEX,
    ALPHA_P_FROZEN,
    ATSPEC_EXTRACTED,
    hydrogen_hyperfine_fermi_frequency_hz,
)


G2_PDF = ROOT / "data" / "quantum" / "sources" / "arxiv_0801.1134v2.pdf"
G_OVER_2_PATTERN = re.compile(r"g/2\s*=\s*([0-9][0-9.\s]*)\(\s*(\d+)\s*\)", re.I)


# 関数: JSON payload を 1 本読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: g/2 measured token を PDF から抽出する。

def extract_g_over_2_from_pdf(path: Path) -> dict:
    """Return the directly measured `g/2` token from the retained Gabrielse PDF."""
    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    text = text.replace("\u00a0", " ")
    match = G_OVER_2_PATTERN.search(text)
    if not match:
        raise RuntimeError(f"[fail] could not extract g/2 token from PDF text: {path}")

    value_text = re.sub(r"\s+", "", match.group(1)).strip()
    value = float(value_text)
    unc_digits = int(match.group(2))
    decimals = len(value_text.split(".", 1)[1]) if "." in value_text else 0
    sigma = float(unc_digits) * (10.0 ** (-decimals))
    return {
        "token": match.group(0),
        "value": value,
        "sigma": sigma,
    }


# 関数: g/2 corrected hyperfine frequency を返す。

def hydrogen_hyperfine_g2_corrected_frequency_hz(alpha_value: float, *, g_over_2: float) -> float:
    """Return the H I 21 cm baseline corrected by the measured electron `g/2`."""
    return float(g_over_2 * hydrogen_hyperfine_fermi_frequency_hz(alpha_value))


# 関数: 1 候補 alpha の corrected prediction row を返す。

def build_prediction_row(
    *,
    alpha_label: str,
    alpha_value: float,
    observed_hz: float,
    sigma_hz: float,
    g_over_2: float,
) -> dict:
    """Return one prediction row on the corrected hyperfine surface."""
    predicted_hz = hydrogen_hyperfine_g2_corrected_frequency_hz(alpha_value, g_over_2=g_over_2)
    delta_hz = float(predicted_hz - observed_hz)
    rel_error = float(delta_hz / observed_hz)
    z_sigma = float(delta_hz / sigma_hz) if sigma_hz > 0.0 else math.nan
    return {
        "alpha_label": alpha_label,
        "alpha_value": float(alpha_value),
        "predicted_hz": predicted_hz,
        "observed_hz": float(observed_hz),
        "delta_hz": delta_hz,
        "relative_error_vs_observed": rel_error,
        "sigma_units_vs_observed": z_sigma,
    }


# 関数: `.5959-.5962` 用の corrected hyperfine pack を返す。

def build_trial2_hyperfine_g2_correction_materialization_pack() -> dict:
    """Return the retained source-backed `g/2` hyperfine correction pack."""
    extracted = read_json(ATSPEC_EXTRACTED)
    hyperfine = extracted["hydrogen_hyperfine_21cm"]
    observed_hz = float(hyperfine["f_hz"])
    sigma_hz = float(hyperfine["sigma_hz"])

    g2_payload = extract_g_over_2_from_pdf(G2_PDF)
    g_over_2 = float(g2_payload["value"])
    a_e = float(g_over_2 - 1.0)

    predictions = [
        build_prediction_row(
            alpha_label="alpha_P_frozen",
            alpha_value=ALPHA_P_FROZEN,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
            g_over_2=g_over_2,
        ),
        build_prediction_row(
            alpha_label="alpha_common",
            alpha_value=ALPHA_COMMON,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
            g_over_2=g_over_2,
        ),
        build_prediction_row(
            alpha_label="alpha_P_4D_can",
            alpha_value=ALPHA_P_4D_CAN,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
            g_over_2=g_over_2,
        ),
        build_prediction_row(
            alpha_label="alpha_P_4D_vertex",
            alpha_value=ALPHA_P_4D_VERTEX,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
            g_over_2=g_over_2,
        ),
        build_prediction_row(
            alpha_label="alpha_CODATA",
            alpha_value=ALPHA_CODATA,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
            g_over_2=g_over_2,
        ),
    ]
    predictions_sorted = sorted(
        predictions,
        key=lambda row: abs(float(row["relative_error_vs_observed"])),
    )
    pmodel_sorted = [
        row
        for row in predictions_sorted
        if str(row["alpha_label"]).startswith("alpha_P_") or str(row["alpha_label"]) == "alpha_common"
    ]

    surface = {
        "surface_id": "hydrogen_hyperfine_21cm_g2_corrected_baseline",
        "label": "Hydrogen hyperfine 21 cm g/2-corrected baseline",
        "formula": "nu_hfs_g2(alpha) = (g_e / 2) * nu_hfs_Fermi(alpha)",
        "formula_expanded": (
            "nu_hfs_g2(alpha) = (g_e / 2) * (8/3) * alpha^4 * (mu_p / mu_B) * "
            "(mu_red / m_e)^3 * m_e c^2 / h"
        ),
        "correction_source_kind": "direct_measured_g_over_2_token",
        "current_alpha_rerun_ready_now": True,
        "independent_observable_now": True,
        "primary_score_admissible_now": True,
        "selected_primary_target_now": True,
        "observed_hz": observed_hz,
        "sigma_hz": sigma_hz,
        "source_token_hz": str(hyperfine["token"]),
        "g_over_2_token": str(g2_payload["token"]),
        "predictions": predictions_sorted,
        "notes": (
            "This corrected surface applies the directly measured electron g/2 "
            "factor to the retained H I 21 cm Fermi-contact baseline. It does not "
            "reuse the alpha value extracted in the same paper."
        ),
    }

    return {
        "alpha_constants": {
            "alpha_P_frozen": ALPHA_P_FROZEN,
            "alpha_common": ALPHA_COMMON,
            "alpha_P_4D_can": ALPHA_P_4D_CAN,
            "alpha_P_4D_vertex": ALPHA_P_4D_VERTEX,
            "alpha_CODATA": ALPHA_CODATA,
        },
        "surface": surface,
        "source_constants": {
            "g_over_2": g_over_2,
            "g_over_2_sigma": float(g2_payload["sigma"]),
            "a_e": a_e,
            "a_e_ppm": float(a_e * 1.0e6),
        },
        "summary": {
            "hyperfine_corrected_surface_id": str(surface["surface_id"]),
            "hyperfine_corrected_surface_ready_now": True,
            "g_over_2_token_extracted_now": True,
            "best_overall_alpha_label": str(predictions_sorted[0]["alpha_label"]),
            "best_overall_is_codata_now": str(predictions_sorted[0]["alpha_label"]) == "alpha_CODATA",
            "best_overall_relative_error_vs_observed": float(
                predictions_sorted[0]["relative_error_vs_observed"]
            ),
            "best_pmodel_alpha_label": str(pmodel_sorted[0]["alpha_label"]),
            "best_pmodel_relative_error_vs_observed": float(
                pmodel_sorted[0]["relative_error_vs_observed"]
            ),
            "observed_hz": observed_hz,
            "sigma_hz": sigma_hz,
            "g_over_2": g_over_2,
            "a_e": a_e,
            "primary_score_admissible_now": True,
        },
        "trial2_hyperfine_g2_correction_materialized_now": True,
        "trial2_hyperfine_corrected_surface_ready_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the corrected-hyperfine materialization backend directly."""
    pack = build_trial2_hyperfine_g2_correction_materialization_pack()
    summary = pack["summary"]
    print("[trial2_hyperfine_g2_correction_materialization_backend]")
    print(f"  hyperfine_corrected_surface_id = {summary['hyperfine_corrected_surface_id']}")
    print(f"  g_over_2 = {summary['g_over_2']}")
    print(f"  best_overall_alpha_label = {summary['best_overall_alpha_label']}")
    print(
        "  best_overall_relative_error_vs_observed = "
        f"{summary['best_overall_relative_error_vs_observed']}"
    )


if __name__ == "__main__":
    main()
