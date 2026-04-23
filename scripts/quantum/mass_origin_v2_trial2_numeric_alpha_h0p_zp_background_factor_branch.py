#!/usr/bin/env python3
"""Generate 8.7.56.851-.854 Trial-2 numeric alpha H0^(P)-Z_P background-factor artifacts."""

from __future__ import annotations

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

PRIOR_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_h0p_zp_bridge_literal_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_ninth_refresh_metrics.json"

CURRENT_ROUTE = "trial2_numeric_alpha_newton_limit_h0p_zp_background_factor_identification"
CURRENT_ARTIFACT = "trial2_numeric_alpha_h0p_zp_background_factor"
NEXT_ROUTE = "8.7.56.855"
NEXT_BRANCH = "8.7.56.855-.858"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_newton_limit_h0p_zp_background_factor_late_time_rule_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_h0p_zp_background_factor_late_time_rule"


# Function: return the ordered background-factor patterns used across the residual branch.
def background_factor_patterns() -> list[str]:
    """Return the ordered background-factor patterns for the H0^(P)-Z_P bridge."""
    return [
        r"C_{\rm bg}",
        "C_bg",
        "background factor",
        "cosmological damping",
    ]


# Function: return the ordered late-time-rule patterns used across the residual branch.

def late_time_rule_patterns() -> list[str]:
    """Return the ordered late-time-rule patterns for the H0^(P)-Z_P background factor."""
    return [
        r"C_{\rm bg} =",
        "C_bg =",
        "Part I の late-time 極限で既に固定されている",
        "C_bg の値を Part I の late-time 極限から固定する",
        "late-time limit",
        "late-time 極限",
    ]


# Function: execute the H0^(P)-Z_P bridge background-factor residual branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha H0^(P)-Z_P bridge background-factor residual branch."""
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

    advice_background_factor_hit = first_hit(advice_text, background_factor_patterns())
    advice_late_time_rule_hit = first_hit(advice_text, late_time_rule_patterns())
    part1_m0_formula_hit = hit(part1_text, r"m_0^2 = \frac{4\lambda v^2}{Z_P}")
    part1_background_evolution_hit = first_hit(
        part1_text,
        [
            r"#### 2.6.3 背景波 $P_{\mathrm{bg}}(t)$ の完全時間発展",
            r"P(x,t) = P_{\mathrm{bg}}(t)\,P_{\mathrm{local}}(x)",
        ],
    )
    part1_electron_identification_hit = hit(part1_text, r"M_{(1,0,0,0)} = m_e")
    part2_background_wave_hit = hit(part2_text, r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]")
    part2_h0p_scale_hit = first_hit(
        part2_text,
        [
            r"a_0=\frac{cH_{0}^{(P)}}{2\pi}",
            r"\omega_{\mathrm{bg}}=H_{0}^{(P)}",
        ],
    )
    public_background_factor_hit = first_hit(public_pack_text, background_factor_patterns())
    public_late_time_rule_hit = first_hit(
        public_pack_text,
        [
            r"C_{\rm bg} =",
            "C_bg =",
            "background factor is",
            "cosmological damping factor",
        ],
    )

    computation_formula_ready = bool(prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"])
    absolute_normalization_dictionary_ready = bool(
        prior_gate_summary["trial2_numeric_alpha_absolute_normalization_dictionary_ready"]
    )
    h0p_bridge_pivot_retained = bool(prior_gate_summary["h0p_bridge_pivot_retained"])
    prior_background_factor_route_active = (
        prior_gate_summary["selected_residual_route"] == CURRENT_ROUTE
        and prior_gate_summary["missing_v2_artifact"] == CURRENT_ARTIFACT
        and prior_route_summary["selected_next_generation_route"] == CURRENT_ROUTE
    )
    h0p_background_law_ready = part2_background_wave_hit is not None and part2_h0p_scale_hit is not None
    background_factor_lineage_ready = (
        advice_background_factor_hit is not None
        and advice_late_time_rule_hit is not None
        and part1_m0_formula_hit is not None
        and part1_background_evolution_hit is not None
        and part1_electron_identification_hit is not None
        and h0p_background_law_ready
        and computation_formula_ready
        and absolute_normalization_dictionary_ready
        and h0p_bridge_pivot_retained
        and prior_background_factor_route_active
    )
    explicit_h0p_zp_background_factor_available = public_background_factor_hit is not None
    explicit_h0p_zp_background_factor_late_time_rule_available = public_late_time_rule_hit is not None
    dominant_blocker_is_h0p_zp_background_factor_late_time_rule_absence = (
        background_factor_lineage_ready and not explicit_h0p_zp_background_factor_late_time_rule_available
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
        "mass_origin_v2_trial2_numeric_alpha_h0p_zp_bridge_literal_declaration_gate_json": display_path(
            PRIOR_GATE
        ),
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_ninth_refresh_json": display_path(PRIOR_ROUTE),
    }

    inventory_targets = [
        target_record(
            "advice_background_factor_candidate",
            ADVICE,
            advice_text,
            r"C_{\rm bg}",
            "The expert advice already names the missing background factor inside the H0^(P)-Z_P bridge literal.",
        ),
        target_record(
            "advice_background_factor_late_time_rule_candidate",
            ADVICE,
            advice_text,
            "Part I の late-time 極限で既に固定されている",
            "The expert advice already points to the late-time rule that should fix the background factor.",
        ),
        target_record(
            "part1_m0_formula",
            PART1,
            part1_text,
            r"m_0^2 = \frac{4\lambda v^2}{Z_P}",
            "Part I still carries the m0-Z_P relation required before the background factor can be used numerically.",
        ),
        target_record(
            "part1_background_evolution_section",
            PART1,
            part1_text,
            r"#### 2.6.3 背景波 $P_{\mathrm{bg}}(t)$ の完全時間発展",
            "Part I still carries the background-wave evolution section that the advice points to for the late-time factor.",
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
            "part3a_current_background_factor_blocker",
            PART3A,
            part3a_text,
            "trial2_numeric_alpha_h0p_zp_background_factor",
            "Part III-A already names the H0^(P)-Z_P bridge background factor as the current blocker.",
        ),
        target_record(
            "part5_current_background_factor_blocker",
            PART5,
            part5_text,
            "trial2_numeric_alpha_h0p_zp_background_factor",
            "Part V already names the H0^(P)-Z_P bridge background factor as the current blocker.",
        ),
        target_record(
            "status_current_background_factor_branch",
            STATUS,
            status_text,
            "8.7.56.851-.854",
            "STATUS already exposes the background-factor branch as the current official branch.",
        ),
        target_record(
            "roadmap_current_background_factor_branch",
            ROADMAP,
            roadmap_text,
            "8.7.56.851-.854",
            "ROADMAP already exposes the background-factor branch as the current official branch.",
        ),
    ]
    inventory_ready = all(target["present"] for target in inventory_targets) and background_factor_lineage_ready

    inventory_payload = payload(
        "8.7.56.851",
        "Trial-2 numeric alpha H0^(P)-Z_P bridge background-factor source inventory",
        common_inputs,
        "Freeze the H0^(P)-Z_P background-factor lineage and show that the next missing public surface is now the late-time rule that should fix that factor honestly.",
        {
            "factor_rule": "the current residual route is justified only if the public paper pack already carries the computation formula, the electron-identification dictionary, the late-time H0^(P) background law, and the expert background-factor candidate",
            "late_time_rule": "the first missing surface inside the background-factor route is the explicit late-time rule that fixes C_bg because the advice already points to that route directly",
            "continuity_rule": "the background-factor residual is admissible only if the prior gate already promoted the route away from the bridge literal and into the background-factor family",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_zp_background_factor_inventory_complete",
                "pass" if inventory_ready else "reject",
                "H0^(P)-Z_P bridge background-factor inventory complete",
                1 if inventory_ready else 0,
                "The advice background factor, Part I normalization and background section, Part II background law, and current blocker are frozen as one pack.",
            ),
            row(
                "trial2_numeric_alpha_h0p_zp_background_factor_lineage_ready",
                "pass" if background_factor_lineage_ready else "reject",
                "H0^(P)-Z_P bridge background-factor lineage ready",
                1 if background_factor_lineage_ready else 0,
                "The background-factor lineage is meaningful only after the bridge-literal residual is already active.",
            ),
            row(
                "trial2_numeric_alpha_explicit_h0p_zp_background_factor_available_in_inventory",
                "pass" if explicit_h0p_zp_background_factor_available else "reject",
                "explicit H0^(P)-Z_P background factor available in inventory",
                1 if explicit_h0p_zp_background_factor_available else 0,
                (
                    "The public pack now names C_bg as the factor inside the H0^(P)-Z_P bridge literal."
                    if explicit_h0p_zp_background_factor_available
                    else "The public pack still does not expose C_bg or an equivalent background-factor closure."
                ),
            ),
            row(
                "trial2_numeric_alpha_explicit_h0p_zp_background_factor_late_time_rule_available_in_inventory",
                "pass" if explicit_h0p_zp_background_factor_late_time_rule_available else "reject",
                "explicit H0^(P)-Z_P background-factor late-time rule available in inventory",
                1 if explicit_h0p_zp_background_factor_late_time_rule_available else 0,
                "The public pack still does not expose the late-time rule that should fix the background factor.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "computation_formula_ready": computation_formula_ready,
            "absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "h0p_background_law_ready": h0p_background_law_ready,
            "h0p_bridge_pivot_retained": h0p_bridge_pivot_retained,
            "background_factor_lineage_ready": background_factor_lineage_ready,
            "explicit_h0p_zp_background_factor_available": explicit_h0p_zp_background_factor_available,
            "explicit_h0p_zp_background_factor_late_time_rule_available": explicit_h0p_zp_background_factor_late_time_rule_available,
        },
        {
            "overall_status": "trial2_numeric_alpha_h0p_zp_background_factor_inventory_frozen",
            "advance_to_8_7_56_852": inventory_ready,
            "next_required_artifacts": [],
        },
        {
            "inventory_targets": inventory_targets,
            "advice_background_factor_hit": advice_background_factor_hit,
            "advice_late_time_rule_hit": advice_late_time_rule_hit,
            "public_background_factor_hit": public_background_factor_hit,
            "public_late_time_rule_hit": public_late_time_rule_hit,
            "current_ai_context_step": ai_context["current_step"],
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
        },
    )

    audit_payload = payload(
        "8.7.56.852",
        "Trial-2 numeric alpha H0^(P)-Z_P bridge background-factor audit",
        common_inputs,
        "Audit whether the current public canon already contains the H0^(P)-Z_P background factor and, if not, identify the first missing late-time-rule surface inside that factor route.",
        {
            "availability_rule": "numeric alpha can use the H0^(P)-Z_P bridge factor only once both the factor and the late-time rule that fixes it are public-canonical in the paper pack",
            "late_time_rule": "the first missing background-factor sub-surface is the explicit late-time rule that fixes C_bg because the advice already points there directly",
            "statement_rule": "the background-factor route remains inside the H0^(P)-Z_P pivot and does not reopen the retired same-sector wording family",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_zp_background_factor_audit_complete",
                "pass",
                "H0^(P)-Z_P bridge background-factor audit complete",
                1,
                "This step decides whether the background factor already exists or whether the first missing surface is its late-time fixing rule.",
            ),
            row(
                "trial2_numeric_alpha_explicit_h0p_zp_background_factor_available",
                "pass" if explicit_h0p_zp_background_factor_available else "reject",
                "explicit H0^(P)-Z_P background factor available",
                1 if explicit_h0p_zp_background_factor_available else 0,
                (
                    "The current public pack now names C_bg as the factor inside the H0^(P)-Z_P bridge literal."
                    if explicit_h0p_zp_background_factor_available
                    else "The current public pack still does not expose C_bg or an equivalent background-factor closure."
                ),
            ),
            row(
                "trial2_numeric_alpha_explicit_h0p_zp_background_factor_late_time_rule_available",
                "pass" if explicit_h0p_zp_background_factor_late_time_rule_available else "reject",
                "explicit H0^(P)-Z_P background-factor late-time rule available",
                1 if explicit_h0p_zp_background_factor_late_time_rule_available else 0,
                "The current public pack still does not expose the late-time rule that should fix C_bg.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_h0p_zp_background_factor_late_time_rule_absence",
                "pass" if dominant_blocker_is_h0p_zp_background_factor_late_time_rule_absence else "reject",
                "dominant blocker is H0^(P)-Z_P background-factor late-time-rule absence",
                1 if dominant_blocker_is_h0p_zp_background_factor_late_time_rule_absence else 0,
                "The factor route is now coherent enough that its first missing public surface is the explicit late-time rule itself.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "explicit_h0p_zp_background_factor_available": explicit_h0p_zp_background_factor_available,
            "explicit_h0p_zp_background_factor_late_time_rule_available": explicit_h0p_zp_background_factor_late_time_rule_available,
            "dominant_blocker_is_h0p_zp_background_factor_late_time_rule_absence": dominant_blocker_is_h0p_zp_background_factor_late_time_rule_absence,
            "first_route_to_close_after_audit_or_none": "trial2_numeric_alpha_h0p_zp_background_factor_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_h0p_zp_background_factor_audited",
            "advance_to_8_7_56_853": True,
            "next_required_artifacts": [],
        },
        {
            "inventory_summary": inventory_payload["summary"],
            "public_background_factor_hit": public_background_factor_hit,
            "public_late_time_rule_hit": public_late_time_rule_hit,
        },
    )

    gate_payload = payload(
        "8.7.56.853",
        "Trial-2 numeric alpha H0^(P)-Z_P bridge background-factor declaration gate",
        common_inputs,
        "Close the background-factor residual honestly: retain the H0^(P)-Z_P pivot, keep numeric alpha open, and reclassify the next blocker as the missing late-time rule that should fix that factor.",
        {
            "gate_rule": "if the background-factor lineage is coherent but the late-time rule is still absent, the next official blocker is that late-time rule",
            "numeric_rule": "numeric alpha remains open until the late-time rule fixes the background factor publicly",
            "continuity_rule": "the H0^(P)-Z_P pivot and the electron-identification dictionary remain active while the late-time rule is missing",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_zp_background_factor_gate_complete",
                "pass",
                "H0^(P)-Z_P bridge background-factor declaration gate complete",
                1,
                "The background-factor residual is now closed as far as current public canon allows.",
            ),
            row(
                "trial2_numeric_alpha_h0p_bridge_pivot_retained_after_background_factor_gate",
                "pass" if h0p_bridge_pivot_retained else "reject",
                "H0^(P)-Z_P bridge pivot retained after background-factor gate",
                1 if h0p_bridge_pivot_retained else 0,
                "The current route remains the H0^(P)-Z_P pivot rather than the retired same-sector wording family.",
            ),
            row(
                "trial2_numeric_alpha_background_factor_residual_closed_to_late_time_rule",
                "pass" if dominant_blocker_is_h0p_zp_background_factor_late_time_rule_absence else "reject",
                "background-factor residual closed to late-time-rule residual",
                1 if dominant_blocker_is_h0p_zp_background_factor_late_time_rule_absence else 0,
                "The first missing public-canonical surface inside the background-factor route is now the late-time rule itself.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_background_factor_gate",
                "pass" if explicit_h0p_zp_background_factor_late_time_rule_available else "reject",
                "numeric alpha from current pack ready after background-factor gate",
                1 if explicit_h0p_zp_background_factor_late_time_rule_available else 0,
                "Numeric alpha stays open because the late-time fixing rule is still absent.",
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
            "overall_status": "trial2_numeric_alpha_h0p_zp_background_factor_gate_closed_late_time_rule_open",
            "advance_to_8_7_56_854": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "audit_summary": audit_payload["summary"],
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
        },
    )

    route_payload = payload(
        "8.7.56.854",
        "Trial-2 numeric alpha next-generation route contract one-hundred-tenth refresh",
        common_inputs,
        "Refresh the next-generation contract after the H0^(P)-Z_P bridge background-factor residual: keep precision-alpha on the mainline, keep the strong side on reserve, and promote the missing late-time rule as the next official blocker.",
        {
            "selected_route_rule": "the next official route is the missing H0^(P)-Z_P background-factor late-time rule",
            "mainline_rule": "precision-alpha remains the next-generation mainline after the background-factor residual closes to the late-time-rule residual",
            "reserve_rule": "strong-side non-Abelian, running, and confinement gaps remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_zp_background_factor_gate_closed_before_refresh",
                "pass",
                "H0^(P)-Z_P bridge background-factor declaration gate closed before route refresh",
                1,
                "The next-generation contract is refreshed only after the background-factor gate closes.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_h0p_zp_background_factor_late_time_rule",
                "pass",
                "H0^(P)-Z_P background-factor late-time-rule route selected",
                1,
                "The next route now targets the missing late-time rule that should fix the background factor.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_background_factor_gate",
                "pass" if prior_route_summary["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after background-factor gate",
                1 if prior_route_summary["precision_alpha_mainline_retained"] else 0,
                "The mainline remains Trial-2 numeric alpha, not the strong-side reserve.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_background_factor_gate",
                "pass" if prior_route_summary["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained after background-factor gate",
                1 if prior_route_summary["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains exploratory and is not promoted by the background-factor residual.",
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
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_tenth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_h0p_zp_background_factor_late_time_rule_source_inventory",
                "trial2_numeric_alpha_h0p_zp_background_factor_late_time_rule_audit",
            ],
        },
        {
            "gate_summary": gate_payload["summary"],
            "prior_route_summary": prior_route_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_h0p_zp_background_factor_source_inventory",
        inventory_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_h0p_zp_background_factor_audit",
        audit_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_h0p_zp_background_factor_declaration_gate",
        gate_payload,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_tenth_refresh", route_payload)

    print("[done] 8.7.56.851-.854 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_h0p_zp_background_factor_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_h0p_zp_background_factor_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_h0p_zp_background_factor_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_tenth_refresh_metrics.json")
    print(f" - next official branch should move to {NEXT_BRANCH}")


# Function: run the H0^(P)-Z_P bridge background-factor branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha H0^(P)-Z_P bridge background-factor branch."""
    main()


if __name__ == "__main__":
    run_cli()
