#!/usr/bin/env python3
"""Generate 8.7.56.843-.846 Trial-2 numeric alpha H0^(P)-Z_P bridge statement artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from mass_origin_v2_trial2_numeric_alpha_h0p_bridge_branch import (
    OUT,
    ROOT,
    display_path,
    first_hit,
    hit,
    payload,
    read_json,
    read_text,
    require,
    row,
    target_record,
    write_artifact,
)


ADVICE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_h0p_bridge.md")
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

PRIOR_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_h0p_bridge_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventh_refresh_metrics.json"

CURRENT_ROUTE = "trial2_numeric_alpha_newton_limit_h0p_zp_bridge_statement_identification"
CURRENT_ARTIFACT = "trial2_numeric_alpha_h0p_zp_bridge_statement"
NEXT_ROUTE = "8.7.56.847"
NEXT_BRANCH = "8.7.56.847-.850"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_newton_limit_h0p_zp_bridge_literal_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_h0p_zp_bridge_literal"


# Function: return the ordered bridge-literal patterns used across the branch.
def bridge_literal_patterns() -> list[str]:
    """Return the ordered equation-literal patterns for the H0^(P)-Z_P bridge."""
    return [
        r"Z_P = \frac{m_0^2}{(H_0^{(P)})^2}",
        r"Z_P = \frac{m_0^2 \cdot C_{\rm bg}}{(H_0^{(P)})^2}",
        r"Z_P(H_0^{(P)})^2 = m_0^2",
        r"Z_P (H_0^{(P)})^2 = m_0^2",
        r"Z_P(H_0^{(P)})^2 = m_0^2 C_{\rm bg}",
        r"Z_P (H_0^{(P)})^2 = m_0^2 C_{\rm bg}",
    ]


# Function: return the ordered bridge-statement patterns that would close the public statement route.

def bridge_statement_patterns() -> list[str]:
    """Return the ordered prose-level patterns for the H0^(P)-Z_P bridge statement."""
    return [
        "late-time dispersion relation",
        "background-wave dispersion relation",
        "H_0^{(P)}-Z_P bridge statement",
        "H0^(P)-Z_P bridge statement",
    ]


# Function: execute the H0^(P)-Z_P bridge statement residual branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha H0^(P)-Z_P bridge statement residual branch."""
    for path in (
        ADVICE,
        PART1,
        PART2,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIOR_GATE,
        PRIOR_ROUTE,
    ):
        require(path)

    advice_text = read_text(ADVICE)
    part1_text = read_text(PART1)
    part2_text = read_text(PART2)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)

    prior_gate_summary = prior_gate["summary"]
    prior_route_summary = prior_route["summary"]
    public_pack_text = "\n".join([part1_text, part2_text, part3a_text, part5_text])

    advice_bridge_literal_hit = first_hit(advice_text, bridge_literal_patterns())
    advice_background_factor_hit = first_hit(advice_text, [r"C_{\rm bg}", "C_bg"])
    part1_m0_formula_hit = hit(part1_text, r"m_0^2 = \frac{4\lambda v^2}{Z_P}")
    part1_electron_identification_hit = hit(part1_text, r"M_{(1,0,0,0)} = m_e")
    part2_background_wave_hit = hit(part2_text, r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]")
    part2_h0p_scale_hit = first_hit(
        part2_text,
        [
            r"a_0=\frac{cH_{0}^{(P)}}{2\pi}",
            r"\omega_{\mathrm{bg}}=H_{0}^{(P)}",
        ],
    )
    public_bridge_statement_hit = first_hit(public_pack_text, bridge_statement_patterns())
    public_bridge_literal_hit = first_hit(public_pack_text, bridge_literal_patterns())

    computation_formula_ready = bool(prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"])
    absolute_normalization_dictionary_ready = bool(
        prior_gate_summary["trial2_numeric_alpha_absolute_normalization_dictionary_ready"]
    )
    h0p_bridge_pivot_retained = bool(prior_gate_summary["h0p_bridge_pivot_adopted"])
    prior_statement_route_active = (
        prior_gate_summary["selected_residual_route"] == CURRENT_ROUTE
        and prior_gate_summary["missing_v2_artifact"] == CURRENT_ARTIFACT
        and prior_route_summary["selected_next_generation_route"] == CURRENT_ROUTE
    )
    h0p_background_law_ready = part2_background_wave_hit is not None and part2_h0p_scale_hit is not None
    bridge_statement_lineage_ready = (
        advice_bridge_literal_hit is not None
        and advice_background_factor_hit is not None
        and part1_m0_formula_hit is not None
        and part1_electron_identification_hit is not None
        and h0p_background_law_ready
        and computation_formula_ready
        and absolute_normalization_dictionary_ready
        and h0p_bridge_pivot_retained
        and prior_statement_route_active
    )
    explicit_h0p_zp_bridge_statement_available = public_bridge_statement_hit is not None
    explicit_h0p_zp_bridge_literal_available = public_bridge_literal_hit is not None
    dominant_blocker_is_h0p_zp_bridge_literal_absence = (
        bridge_statement_lineage_ready and not explicit_h0p_zp_bridge_literal_available
    )

    common_inputs = {
        "expert_note_markdown": display_path(ADVICE),
        "part1_markdown": display_path(PART1),
        "part2_markdown": display_path(PART2),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "mass_origin_v2_trial2_numeric_alpha_h0p_bridge_declaration_gate_json": display_path(PRIOR_GATE),
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventh_refresh_json": display_path(PRIOR_ROUTE),
    }

    inventory_targets = [
        target_record(
            "advice_h0p_bridge_literal_candidate",
            ADVICE,
            advice_text,
            r"Z_P = \frac{m_0^2 \cdot C_{\rm bg}}{(H_0^{(P)})^2}",
            "The expert advice already provides the minimal bridge equation candidate.",
        ),
        target_record(
            "advice_background_factor_candidate",
            ADVICE,
            advice_text,
            r"C_{\rm bg}",
            "The expert advice also names the background factor that would complete the bridge statement.",
        ),
        target_record(
            "part1_m0_formula",
            PART1,
            part1_text,
            r"m_0^2 = \frac{4\lambda v^2}{Z_P}",
            "Part I still carries the m0-Z_P relation required before the H0^(P)-Z_P bridge can be stated explicitly.",
        ),
        target_record(
            "part1_electron_identification_statement",
            PART1,
            part1_text,
            r"M_{(1,0,0,0)} = m_e",
            "Part I still carries the electron-identification absolute-normalization dictionary.",
        ),
        target_record(
            "part2_background_wave_law",
            PART2,
            part2_text,
            r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]",
            "Part II still carries the late-time background-wave law from which H0^(P) is read.",
        ),
        target_record(
            "part2_h0p_scale_mapping",
            PART2,
            part2_text,
            r"\omega_{\mathrm{bg}}=H_{0}^{(P)}",
            "Part II still exposes H0^(P) as the late-time background-wave scale.",
        ),
        target_record(
            "part3a_current_bridge_statement_blocker",
            PART3A,
            part3a_text,
            "trial2_numeric_alpha_h0p_zp_bridge_statement",
            "Part III-A already names the H0^(P)-Z_P bridge statement as the current blocker.",
        ),
        target_record(
            "part5_current_bridge_statement_blocker",
            PART5,
            part5_text,
            "trial2_numeric_alpha_h0p_zp_bridge_statement",
            "Part V already names the H0^(P)-Z_P bridge statement as the current blocker.",
        ),
        target_record(
            "status_current_bridge_statement_branch",
            STATUS,
            status_text,
            "8.7.56.843-.846",
            "STATUS already exposes the bridge-statement branch as the current official branch.",
        ),
        target_record(
            "roadmap_current_bridge_statement_branch",
            ROADMAP,
            roadmap_text,
            "8.7.56.843-.846",
            "ROADMAP already exposes the bridge-statement branch as the current official branch.",
        ),
    ]
    inventory_ready = all(target["present"] for target in inventory_targets) and bridge_statement_lineage_ready

    inventory_payload = payload(
        "8.7.56.843",
        "Trial-2 numeric alpha H0^(P)-Z_P bridge statement source inventory",
        common_inputs,
        "Freeze the explicit H0^(P)-Z_P bridge statement lineage and show that the remaining blocker is now the missing minimal bridge literal inside that statement route.",
        {
            "statement_rule": "the current residual route is justified only if the public paper pack already carries the computation formula, the electron-identification dictionary, the late-time H0^(P) background law, and the expert bridge candidate lineage",
            "literal_rule": "the minimal literal candidate is the explicit equation Z_P = m_0^2/(H_0^(P))^2 or its C_bg-weighted equivalent",
            "continuity_rule": "the bridge-statement residual is admissible only if the prior gate already retired the same-sector wording retry from the mainline",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_zp_bridge_statement_inventory_complete",
                "pass" if inventory_ready else "reject",
                "H0^(P)-Z_P bridge statement inventory complete",
                1 if inventory_ready else 0,
                "The advice literal, Part I normalization, Part II background law, and current bridge-statement blocker are frozen as one pack.",
            ),
            row(
                "trial2_numeric_alpha_h0p_zp_bridge_statement_lineage_ready",
                "pass" if bridge_statement_lineage_ready else "reject",
                "H0^(P)-Z_P bridge statement lineage ready",
                1 if bridge_statement_lineage_ready else 0,
                "The bridge-statement lineage is meaningful only after the H0^(P)-Z_P pivot is already active.",
            ),
            row(
                "trial2_numeric_alpha_explicit_h0p_zp_bridge_statement_available_in_inventory",
                "pass" if explicit_h0p_zp_bridge_statement_available else "reject",
                "explicit H0^(P)-Z_P bridge statement available in inventory",
                1 if explicit_h0p_zp_bridge_statement_available else 0,
                "The public pack still does not expose a prose-level bridge statement.",
            ),
            row(
                "trial2_numeric_alpha_explicit_h0p_zp_bridge_literal_available_in_inventory",
                "pass" if explicit_h0p_zp_bridge_literal_available else "reject",
                "explicit H0^(P)-Z_P bridge literal available in inventory",
                1 if explicit_h0p_zp_bridge_literal_available else 0,
                "The public pack still does not expose the minimal bridge equation literal.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "computation_formula_ready": computation_formula_ready,
            "absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "h0p_background_law_ready": h0p_background_law_ready,
            "h0p_bridge_pivot_retained": h0p_bridge_pivot_retained,
            "bridge_statement_lineage_ready": bridge_statement_lineage_ready,
            "explicit_h0p_zp_bridge_statement_available": explicit_h0p_zp_bridge_statement_available,
            "explicit_h0p_zp_bridge_literal_available": explicit_h0p_zp_bridge_literal_available,
        },
        {
            "overall_status": "trial2_numeric_alpha_h0p_zp_bridge_statement_inventory_frozen",
            "advance_to_8_7_56_844": inventory_ready,
            "next_required_artifacts": [],
        },
        {
            "inventory_targets": inventory_targets,
            "advice_bridge_literal_hit": advice_bridge_literal_hit,
            "public_bridge_statement_hit": public_bridge_statement_hit,
            "public_bridge_literal_hit": public_bridge_literal_hit,
            "current_ai_context_step": ai_context["current_step"],
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
        },
    )

    audit_payload = payload(
        "8.7.56.844",
        "Trial-2 numeric alpha H0^(P)-Z_P bridge statement audit",
        common_inputs,
        "Audit whether the current public canon already contains the explicit H0^(P)-Z_P bridge statement and, if not, identify the minimal missing literal inside that statement route.",
        {
            "availability_rule": "numeric alpha can use the H0^(P)-Z_P bridge only once the explicit statement and its literal equation are public-canonical in the paper pack",
            "statement_rule": "the prose-level statement must tie the late-time background law to the Part I mass normalization without introducing a new sector hop",
            "literal_rule": "the first minimal literal to close is the equation Z_P = m_0^2/(H_0^(P))^2 or its C_bg-weighted equivalent",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_zp_bridge_statement_audit_complete",
                "pass",
                "H0^(P)-Z_P bridge statement audit complete",
                1,
                "This step decides whether the explicit bridge statement already exists or whether the first missing surface is its minimal literal.",
            ),
            row(
                "trial2_numeric_alpha_explicit_h0p_zp_bridge_statement_available",
                "pass" if explicit_h0p_zp_bridge_statement_available else "reject",
                "explicit H0^(P)-Z_P bridge statement available",
                1 if explicit_h0p_zp_bridge_statement_available else 0,
                "The current public pack still does not expose the bridge statement as prose-level canon.",
            ),
            row(
                "trial2_numeric_alpha_explicit_h0p_zp_bridge_literal_available",
                "pass" if explicit_h0p_zp_bridge_literal_available else "reject",
                "explicit H0^(P)-Z_P bridge literal available",
                1 if explicit_h0p_zp_bridge_literal_available else 0,
                "The current public pack still does not expose the minimal bridge equation literal.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_h0p_zp_bridge_literal_absence",
                "pass" if dominant_blocker_is_h0p_zp_bridge_literal_absence else "reject",
                "dominant blocker is H0^(P)-Z_P bridge literal absence",
                1 if dominant_blocker_is_h0p_zp_bridge_literal_absence else 0,
                "The statement route is now coherent enough that its first missing public surface is the bridge literal itself.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "explicit_h0p_zp_bridge_statement_available": explicit_h0p_zp_bridge_statement_available,
            "explicit_h0p_zp_bridge_literal_available": explicit_h0p_zp_bridge_literal_available,
            "dominant_blocker_is_h0p_zp_bridge_literal_absence": dominant_blocker_is_h0p_zp_bridge_literal_absence,
            "first_route_to_close_after_audit_or_none": "trial2_numeric_alpha_h0p_zp_bridge_statement_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_h0p_zp_bridge_statement_audited",
            "advance_to_8_7_56_845": True,
            "next_required_artifacts": [],
        },
        {
            "inventory_summary": inventory_payload["summary"],
            "public_bridge_statement_hit": public_bridge_statement_hit,
            "public_bridge_literal_hit": public_bridge_literal_hit,
        },
    )

    gate_payload = payload(
        "8.7.56.845",
        "Trial-2 numeric alpha H0^(P)-Z_P bridge statement declaration gate",
        common_inputs,
        "Close the bridge-statement residual honestly: retain the H0^(P)-Z_P pivot, keep numeric alpha open, and reclassify the next blocker as the missing minimal bridge literal.",
        {
            "gate_rule": "if the bridge-statement lineage is coherent but the explicit equation literal is still absent, the next official blocker is that bridge literal",
            "numeric_rule": "numeric alpha remains open until the bridge literal becomes public-canonical",
            "continuity_rule": "the H0^(P)-Z_P pivot and the electron-identification dictionary remain active while the bridge literal is missing",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_zp_bridge_statement_gate_complete",
                "pass",
                "H0^(P)-Z_P bridge statement declaration gate complete",
                1,
                "The bridge-statement residual is now closed as far as current public canon allows.",
            ),
            row(
                "trial2_numeric_alpha_h0p_bridge_pivot_retained_after_statement_gate",
                "pass" if h0p_bridge_pivot_retained else "reject",
                "H0^(P)-Z_P bridge pivot retained after statement gate",
                1 if h0p_bridge_pivot_retained else 0,
                "The current route remains the H0^(P)-Z_P pivot rather than the retired same-sector wording family.",
            ),
            row(
                "trial2_numeric_alpha_bridge_statement_residual_closed_to_literal",
                "pass" if dominant_blocker_is_h0p_zp_bridge_literal_absence else "reject",
                "bridge statement residual closed to bridge literal residual",
                1 if dominant_blocker_is_h0p_zp_bridge_literal_absence else 0,
                "The first missing public-canonical surface inside the statement route is now the bridge literal itself.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_statement_gate",
                "pass" if explicit_h0p_zp_bridge_literal_available else "reject",
                "numeric alpha from current pack ready after statement gate",
                1 if explicit_h0p_zp_bridge_literal_available else 0,
                "Numeric alpha stays open because the explicit bridge literal is still absent.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": computation_formula_ready,
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "trial2_numeric_alpha_closeout_ready": False,
            "h0p_bridge_pivot_retained": h0p_bridge_pivot_retained,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_h0p_zp_bridge_statement_gate_closed_bridge_literal_open",
            "advance_to_8_7_56_846": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "audit_summary": audit_payload["summary"],
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
        },
    )

    route_payload = payload(
        "8.7.56.846",
        "Trial-2 numeric alpha next-generation route contract one-hundred-eighth refresh",
        common_inputs,
        "Refresh the next-generation contract after the H0^(P)-Z_P bridge statement residual: keep precision-alpha on the mainline, keep the strong side on reserve, and promote the missing bridge literal as the next official blocker.",
        {
            "selected_route_rule": "the next official route is the missing minimal H0^(P)-Z_P bridge literal",
            "mainline_rule": "precision-alpha remains the next-generation mainline after the statement residual closes to the literal residual",
            "reserve_rule": "strong-side non-Abelian, running, and confinement gaps remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_zp_bridge_statement_gate_closed_before_refresh",
                "pass",
                "H0^(P)-Z_P bridge statement declaration gate closed before route refresh",
                1,
                "The next-generation contract is refreshed only after the statement gate closes.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_h0p_zp_bridge_literal",
                "pass",
                "H0^(P)-Z_P bridge literal route selected",
                1,
                "The next route now targets the explicit bridge literal inside the statement route.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_statement_gate",
                "pass" if prior_route_summary["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after statement gate",
                1 if prior_route_summary["precision_alpha_mainline_retained"] else 0,
                "The mainline remains Trial-2 numeric alpha, not the strong-side reserve.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_statement_gate",
                "pass" if prior_route_summary["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained after statement gate",
                1 if prior_route_summary["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains exploratory and is not promoted by the bridge-statement residual.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": prior_route_summary["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(prior_route_summary["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": absolute_normalization_dictionary_ready,
            "h0p_bridge_pivot_retained": h0p_bridge_pivot_retained,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_eighth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_h0p_zp_bridge_literal_source_inventory",
                "trial2_numeric_alpha_h0p_zp_bridge_literal_audit",
            ],
        },
        {
            "gate_summary": gate_payload["summary"],
            "prior_route_summary": prior_route_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_h0p_zp_bridge_statement_source_inventory",
        inventory_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_h0p_zp_bridge_statement_audit",
        audit_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_h0p_zp_bridge_statement_declaration_gate",
        gate_payload,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_eighth_refresh", route_payload)

    print("[done] 8.7.56.843-.846 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_h0p_zp_bridge_statement_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_h0p_zp_bridge_statement_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_h0p_zp_bridge_statement_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_eighth_refresh_metrics.json")
    print(f" - next official branch should move to {NEXT_BRANCH}")


# Function: run the H0^(P)-Z_P bridge statement branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha H0^(P)-Z_P bridge statement branch."""
    main()


if __name__ == "__main__":
    run_cli()
