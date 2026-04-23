#!/usr/bin/env python3
"""Generate Trial-4 exploratory artifacts for 8.7.56.13-.16.

Trial-4 is explicitly exploratory: it does not attempt a first-principles QCD
replacement inside the current v2.0 canon. Instead it:

1. inventories the currently frozen `P_i` internal-degree / hadron-scale pack,
2. audits whether an honest SU(3)-like non-Abelian structure is already present,
3. evaluates whether running / confinement have at least a qualitative foothold,
   and
4. freezes the exploratory declaration and the resulting v3.0-hold gate.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"

TRIAL2_PAPER_SYNC_GATE = OUT / "mass_origin_v2_trial2_paper_side_sync_reopened_declaration_gate_metrics.json"
TRIAL2_TRIAL4_DISPOSITION = OUT / "mass_origin_v2_trial2_paper_sync_trial4_disposition_44th_refresh_metrics.json"
TRIAL3_CLOSEOUT_GATE = OUT / "mass_origin_v2_t3_t2_coupled_localization_closeout_declaration_gate_metrics.json"
QCD_BASELINE = OUT / "qcd_hadron_masses_baseline_metrics.json"
SIGNED_V2 = OUT / "nuclear_effective_potential_pion_constrained_signed_v2_metrics.json"
KQ_SCAN = OUT / "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_metrics.json"
CHANNEL_SPLIT = OUT / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_metrics.json"

NEXT_ROUTE = "8.7.56.407"

PART1_VECTOR_LINE = "P_\\mu=(P_t,P_1,P_2,P_3)"
PART1_STATIC_LIMIT = "J^i=0 \\Rightarrow P_i=0"
PART3A_FREEZE = "cross-scale freeze 仮説"
PART3A_STRONG_IF = "強い相互作用側の核I/F"
PRIMARY_PDG_LINE = "Particle Data Group (PDG) RPP 2024"
PRIMARY_QCD_SECTION = "強い相互作用（ハドロン/QCD；基準値の固定）"


# 関数: UTC 現在時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 path の存在を確認する。

def req(path: Path) -> None:
    """Abort immediately when a required input path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 text source を読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: repo 相対 POSIX path を返す。

def rel(path: Path) -> str:
    """Return a repository-relative POSIX path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 指定した部分文字列の最初の hit 行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 指定した複数 pattern の hit 数を数える。

def hit_count(text: str, patterns: list[str]) -> int:
    """Count how many patterns are present in the given text."""
    return sum(1 for pattern in patterns if pattern in text)


# 関数: 共通 schema の row を組み立てる。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row payload."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 共通 schema の payload を組み立てる。

def payload(
    step: str,
    name: str,
    inputs: dict,
    intent: str,
    formulas: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build the standard JSON metrics payload used across the roadmap."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "intent": intent,
        "formulas": formulas,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: JSON artifact と rows CSV を side-by-side で保存する。

def write_artifact(stem: str, data: dict) -> None:
    """Write the metrics payload as JSON and as a rows CSV sidecar."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: wording target の present/absent を監査する。

def audit_target(
    file_key: str,
    path: Path,
    text: str,
    pattern: str,
    note: str,
    expected_present: bool = True,
) -> dict:
    """Audit whether a wording target is present or intentionally absent."""
    target_hit = hit(text, pattern)
    present = target_hit is not None
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "expected_present": expected_present,
        "present": present,
        "matched_expectation": present is expected_present,
        "note": note,
        "evidence": target_hit,
    }


# 関数: hadron baseline row を label で取り出す。

def baseline_row(data: dict, label: str) -> dict:
    """Return a hadron-baseline row by label."""
    for item in data.get("rows", []):
        if item.get("label") == label:
            return item

    raise SystemExit(f"[fail] missing hadron baseline label: {label}")


# 関数: signed-V2 metrics から singlet 予言行を抜き出す。

def extract_signed_v2_predictions(data: dict) -> list[dict]:
    """Extract the signed-V2 singlet predictions per dataset."""
    predictions: list[dict] = []
    for item in data.get("results_by_dataset", []):
        fit_singlet = item.get("fit_singlet", {})
        ere = fit_singlet.get("ere", {}) if isinstance(fit_singlet, dict) else {}
        predictions.append(
            {
                "label": item.get("label"),
                "eq_label": item.get("eq_label"),
                "target_v2s_fm3": item.get("inputs", {}).get("singlet", {}).get("v2s_fm3"),
                "pred_v2s_fm3": ere.get("v2_fm3"),
            }
        )

    return predictions


# 関数: k-q scan から within_all 候補を抽出する。

def extract_within_all_rows(data: dict, key: str) -> list[dict]:
    """Extract rows that satisfy the within-all criterion from a scan payload."""
    container = data.get(key, {})
    rows = container.get("rows", []) if isinstance(container, dict) else []
    return [item for item in rows if item.get("within_all") is True]


# 関数: Trial-4 source inventory を構築する。

def build_inventory(
    common_inputs: dict,
    part1_text: str,
    part3a_text: str,
    primary_sources_text: str,
    qcd_baseline: dict,
    signed_v2: dict,
    kq_scan: dict,
    channel_split: dict,
    trial2_gate: dict,
    trial3_gate: dict,
) -> dict:
    """Freeze the Trial-4 exploratory source pack."""
    inventory_targets = [
        audit_target(
            "part1_vector_definition",
            PART1,
            part1_text,
            PART1_VECTOR_LINE,
            "Part I must already expose the four-vector time-wave definition with three spatial components.",
        ),
        audit_target(
            "part1_static_limit",
            PART1,
            part1_text,
            PART1_STATIC_LIMIT,
            "Part I must already freeze the static-limit reduction that turns off P_i when J^i=0.",
        ),
        audit_target(
            "part3a_cross_scale_freeze",
            PART3A,
            part3a_text,
            PART3A_FREEZE,
            "Part III-A must explicitly state that running is frozen under the current canon.",
        ),
        audit_target(
            "part3a_strong_if_reference",
            PART3A,
            part3a_text,
            PART3A_STRONG_IF,
            "Part III-A must already classify the strong-interaction side as a nuclear-interface reference rather than a closed first-principles derivation.",
        ),
        audit_target(
            "primary_sources_qcd_section",
            PRIMARY_SOURCES,
            primary_sources_text,
            PRIMARY_QCD_SECTION,
            "PRIMARY_SOURCES must carry the hadron/QCD baseline section.",
        ),
        audit_target(
            "primary_sources_pdg_line",
            PRIMARY_SOURCES,
            primary_sources_text,
            PRIMARY_PDG_LINE,
            "PRIMARY_SOURCES must point to the PDG RPP 2024 mass table used by the hadron baseline.",
        ),
    ]
    inventory_ready = all(item["matched_expectation"] for item in inventory_targets)
    upstream_ready = bool(
        trial2_gate["summary"]["trial2_current_paper_state_synced"]
        and trial3_gate["summary"]["trial3_two_component_closeout_pass_under_coupled_localization"]
    )
    pi_pm = baseline_row(qcd_baseline, "π±")
    proton = baseline_row(qcd_baseline, "p")
    scan_support_ready = bool(
        qcd_baseline.get("rows")
        and signed_v2.get("results_by_dataset")
        and kq_scan.get("barrier_tail_kq_scan", {}).get("rows")
        and channel_split.get("barrier_tail_channel_split_kq_scan", {}).get("rows")
    )
    required_target_count = len(inventory_targets) + 4
    present_target_count = sum(1 for item in inventory_targets if item["matched_expectation"]) + int(scan_support_ready) * 4

    return payload(
        "8.7.56.13",
        "trial4_nonabelian_color_like_internal_degree_inventory",
        common_inputs,
        "Freeze the Trial-4 exploratory source pack spanning the current P_i definition, the strong-side nuclear interface wording, the PDG hadron baseline, and the pion-constrained phenomenological nuclear artifacts.",
        {
            "inventory_rule": "inventory passes only if the current canon already exposes P_i as a three-component channel, explicitly freezes running, and retains the hadron-scale / nuclear-interface artifact pack",
            "trial4_scope_rule": "Trial-4 remains exploratory: it may reuse hadron-scale and nuclear-interface artifacts, but it may not claim a first-principles QCD derivation at this stage",
        },
        [
            row(
                "trial4_inventory_complete",
                "pass" if inventory_ready and upstream_ready and scan_support_ready else "reject",
                "Trial-4 source inventory complete",
                1 if inventory_ready and upstream_ready and scan_support_ready else 0,
                "The exploratory branch can start only after Trial-2 sync, Trial-3 closeout, and the QCD/nuclear support pack are all present.",
            ),
            row(
                "trial4_upstream_ready",
                "pass" if upstream_ready else "reject",
                "upstream Trial-2 / Trial-3 state ready",
                1 if upstream_ready else 0,
                "Trial-4 is released only after Trial-2 paper sync and Trial-3 honest closeout.",
            ),
            row(
                "trial4_required_target_count",
                "pass",
                "required source targets",
                required_target_count,
                "Inventory counts wording surfaces plus four machine-readable support artifacts.",
            ),
            row(
                "trial4_present_target_count",
                "pass" if present_target_count == required_target_count else "reject",
                "present source targets",
                present_target_count,
                "All required wording surfaces and support artifacts should be present before the exploratory audit starts.",
            ),
        ],
        {
            "inventory_ready": inventory_ready and upstream_ready and scan_support_ready,
            "upstream_ready": upstream_ready,
            "required_target_count": required_target_count,
            "present_target_count": present_target_count,
            "pi_pm_compton_lambda_fm": float(pi_pm["compton_lambda_fm"]),
            "proton_mass_mev": float(proton["mass_mev"]),
            "first_route_to_close_or_none": "trial4_su3_analogy_structural_audit",
        },
        {
            "overall_status": "trial4_exploratory_inventory_frozen",
            "advance_to_8_7_56_14": inventory_ready and upstream_ready and scan_support_ready,
            "next_required_artifacts": [] if inventory_ready and upstream_ready and scan_support_ready else ["trial4_nonabelian_color_like_internal_degree_inventory"],
        },
        {
            "inventory_targets": inventory_targets,
            "trial2_gate_summary": trial2_gate["summary"],
            "trial3_gate_summary": trial3_gate["summary"],
            "hadron_baseline_excerpt": {"pi_pm": pi_pm, "proton": proton},
            "artifact_paths": {
                "qcd_hadron_masses_baseline_metrics_json": rel(QCD_BASELINE),
                "nuclear_effective_potential_pion_constrained_signed_v2_metrics_json": rel(SIGNED_V2),
                "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_metrics_json": rel(KQ_SCAN),
                "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_metrics_json": rel(CHANNEL_SPLIT),
            },
        },
    )


# 関数: SU(3)-analogy structural audit を構築する。

def build_structural_audit(
    common_inputs: dict,
    part1_text: str,
    part3a_text: str,
    inventory: dict,
) -> dict:
    """Audit how far the current canon reaches toward an honest SU(3)-like structure."""
    nonabelian_terms = ["SU(3)", "non-Abelian", "color", "f^{abc}", "T^a"]
    combined_text = "\n".join([part1_text, part3a_text])
    nonabelian_surface_hits = hit_count(combined_text, nonabelian_terms)
    p_i_three_component_candidate = bool(hit(part1_text, PART1_VECTOR_LINE))
    internal_rotation_candidate = p_i_three_component_candidate
    explicit_su3_generator_basis = nonabelian_surface_hits > 0 and "SU(3)" in combined_text
    explicit_nonabelian_structure_constant_pack = "f^{abc}" in combined_text
    explicit_color_charge_representation = "color" in combined_text
    su3_analogy_structural_pass = bool(
        internal_rotation_candidate
        and explicit_su3_generator_basis
        and explicit_nonabelian_structure_constant_pack
        and explicit_color_charge_representation
    )

    return payload(
        "8.7.56.14",
        "trial4_su3_analogy_structural_audit",
        common_inputs,
        "Audit whether the current canon already upgrades the three spatial P_i components into an honest non-Abelian SU(3)-like internal structure, rather than only leaving a component-count analogy candidate.",
        {
            "candidate_rule": "a color-like foothold starts only when the current canon exposes P_i as a three-component channel that can be re-read as an internal-degree candidate",
            "structural_pass_rule": "an honest SU(3)-analogy pass requires an explicit generator basis, a non-Abelian closure surface, and a color-charge representation in the current canon",
        },
        [
            row(
                "trial4_pi_three_component_candidate_available",
                "pass" if p_i_three_component_candidate else "reject",
                "P_i three-component candidate available",
                1 if p_i_three_component_candidate else 0,
                "Part I already freezes P_mu=(P_t,P_1,P_2,P_3), so Trial-4 can at least start from a three-component candidate.",
            ),
            row(
                "trial4_color_like_internal_rotation_candidate_available",
                "pass" if internal_rotation_candidate else "reject",
                "color-like internal-rotation candidate available",
                1 if internal_rotation_candidate else 0,
                "The exploratory branch can reinterpret the three spatial components as a color-like candidate only at the analogy level.",
            ),
            row(
                "trial4_explicit_su3_generator_basis_available",
                "pass" if explicit_su3_generator_basis else "reject",
                "explicit SU(3)-generator basis available",
                1 if explicit_su3_generator_basis else 0,
                "Current canon does not yet expose a generator basis or equivalent color index surface.",
            ),
            row(
                "trial4_explicit_nonabelian_closure_available",
                "pass" if explicit_nonabelian_structure_constant_pack else "reject",
                "explicit non-Abelian closure available",
                1 if explicit_nonabelian_structure_constant_pack else 0,
                "No current-canon commutator / structure-constant closure is written for Trial-4.",
            ),
            row(
                "trial4_su3_analogy_structural_pass",
                "pass" if su3_analogy_structural_pass else "reject",
                "SU(3)-analogy structural pass",
                1 if su3_analogy_structural_pass else 0,
                "Component-count analogy is not enough; an honest SU(3)-like closure would need explicit non-Abelian structure that is currently absent.",
            ),
        ],
        {
            "exploratory_color_like_foothold_available": internal_rotation_candidate,
            "explicit_su3_generator_basis_available": explicit_su3_generator_basis,
            "explicit_nonabelian_structure_constant_pack_available": explicit_nonabelian_structure_constant_pack,
            "explicit_color_charge_representation_available": explicit_color_charge_representation,
            "nonabelian_surface_hit_count": nonabelian_surface_hits,
            "su3_analogy_structural_pass": su3_analogy_structural_pass,
            "first_route_to_close_or_none": "trial4_running_confinement_qualitative_pilot",
        },
        {
            "overall_status": "trial4_su3_analogy_structural_audit_complete",
            "advance_to_8_7_56_15": inventory["summary"]["inventory_ready"],
            "next_required_artifacts": [] if inventory["summary"]["inventory_ready"] else ["trial4_nonabelian_color_like_internal_degree_inventory"],
        },
        {
            "inventory_summary": inventory["summary"],
            "part1_vector_hit": hit(part1_text, PART1_VECTOR_LINE),
            "part1_static_limit_hit": hit(part1_text, PART1_STATIC_LIMIT),
            "part3a_cross_scale_freeze_hit": hit(part3a_text, PART3A_FREEZE),
            "nonabelian_surface_patterns": nonabelian_terms,
        },
    )


# 関数: running / confinement qualitative pilot を構築する。

def build_running_confinement_pilot(
    common_inputs: dict,
    part3a_text: str,
    qcd_baseline: dict,
    signed_v2: dict,
    kq_scan: dict,
    channel_split: dict,
    structural_audit: dict,
) -> dict:
    """Evaluate whether the current canon has any honest qualitative strong-sector foothold."""
    pi_pm = baseline_row(qcd_baseline, "π±")
    signed_v2_predictions = extract_signed_v2_predictions(signed_v2)
    signed_v2_rescues_target = all(
        float(item["pred_v2s_fm3"]) < 0 for item in signed_v2_predictions if item["pred_v2s_fm3"] is not None
    )
    kq_within_all_rows = extract_within_all_rows(kq_scan, "barrier_tail_kq_scan")
    best_kq_within_all = sorted(
        kq_within_all_rows,
        key=lambda item: (
            item.get("max_dist_to_env_fm3", 1e9),
            abs(float(item.get("barrier_height_factor", 0.0)) - 1.0),
            abs(float(item.get("tail_depth_factor", 0.0)) - 1.0),
        ),
    )[0] if kq_within_all_rows else None
    channel_split_within_all_rows = extract_within_all_rows(channel_split, "barrier_tail_channel_split_kq_scan")
    selected_channel_split = channel_split.get("barrier_tail_channel_split_kq_scan", {}).get("selected", {})
    cross_scale_freeze_running_ignored = bool(hit(part3a_text, PART3A_FREEZE))
    qualitative_color_like_foothold = bool(structural_audit["summary"]["exploratory_color_like_foothold_available"])
    pion_scale_to_nuclear_foothold = best_kq_within_all is not None
    running_qualitative_foothold_available = False
    confinement_qualitative_foothold_available = False
    overall_qualitative_foothold = bool(qualitative_color_like_foothold or pion_scale_to_nuclear_foothold)

    return payload(
        "8.7.56.15",
        "trial4_running_confinement_qualitative_pilot",
        common_inputs,
        "Evaluate whether the strong-interaction exploratory branch has at least one honest qualitative foothold, while separating hadron-scale / nuclear-interface evidence from absent running and confinement statements.",
        {
            "running_rule": "an honest running foothold requires explicit scale dependence beyond the cross-scale freeze rule",
            "confinement_rule": "an honest confinement foothold requires a first-principles confining or bound-tail statement, not merely a phenomenological pion-constrained nuclear fit",
            "exploratory_rule": "Trial-4 may remain scientifically live in exploratory mode when at least one qualitative foothold survives, even if SU(3), running, and confinement are not yet honest current-canon claims",
        },
        [
            row(
                "trial4_pion_scale_baseline_available",
                "pass",
                "pion-scale baseline available",
                1,
                "PDG-fixed pion Compton length is already available as the hadron-scale baseline for the exploratory branch.",
            ),
            row(
                "trial4_signed_v2_rescue_with_current_two_range_available",
                "pass" if signed_v2_rescues_target else "reject",
                "signed-V2 rescue with current two-range ansatz available",
                1 if signed_v2_rescues_target else 0,
                "The minimal signed-V2 ansatz still predicts positive singlet v2s and therefore does not rescue the strong-side target by itself.",
            ),
            row(
                "trial4_pion_constrained_nuclear_foothold_available",
                "pass" if pion_scale_to_nuclear_foothold else "reject",
                "pion-constrained nuclear foothold available",
                1 if pion_scale_to_nuclear_foothold else 0,
                "A global barrier+tail scan admits within-envelope singlet v2s rows, so a phenomenological hadron-scale to nuclear-interface foothold exists.",
            ),
            row(
                "trial4_running_qualitative_foothold_available",
                "pass" if running_qualitative_foothold_available else "reject",
                "running qualitative foothold available",
                1 if running_qualitative_foothold_available else 0,
                "Current canon explicitly freezes running, so Trial-4 cannot honestly claim a running derivation yet.",
            ),
            row(
                "trial4_confinement_qualitative_foothold_available",
                "pass" if confinement_qualitative_foothold_available else "reject",
                "confinement qualitative foothold available",
                1 if confinement_qualitative_foothold_available else 0,
                "The current branch has phenomenological nuclear fits but no explicit confinement statement or first-principles confining tail.",
            ),
            row(
                "trial4_overall_qualitative_foothold_exists",
                "pass" if overall_qualitative_foothold else "reject",
                "overall Trial-4 qualitative foothold exists",
                1 if overall_qualitative_foothold else 0,
                "Exploratory viability survives because the color-like candidate and the pion-scale nuclear bridge provide a foothold, even though running/confinement remain absent.",
            ),
        ],
        {
            "pi_pm_compton_lambda_fm": float(pi_pm["compton_lambda_fm"]),
            "cross_scale_freeze_running_ignored": cross_scale_freeze_running_ignored,
            "signed_v2_rescues_target": signed_v2_rescues_target,
            "best_signed_v2_predictions": signed_v2_predictions,
            "pion_scale_to_nuclear_foothold_available": pion_scale_to_nuclear_foothold,
            "best_barrier_tail_within_all_or_none": best_kq_within_all,
            "channel_split_within_all_count": len(channel_split_within_all_rows),
            "selected_channel_split_or_none": selected_channel_split,
            "running_qualitative_foothold_available": running_qualitative_foothold_available,
            "confinement_qualitative_foothold_available": confinement_qualitative_foothold_available,
            "overall_trial4_qualitative_foothold_exists": overall_qualitative_foothold,
            "first_route_to_close_or_none": "trial4_exploratory_declaration_v3_hold_gate",
        },
        {
            "overall_status": "trial4_running_confinement_qualitative_pilot_complete",
            "advance_to_8_7_56_16": True,
            "next_required_artifacts": [],
        },
        {
            "structural_audit_summary": structural_audit["summary"],
            "part3a_cross_scale_freeze_hit": hit(part3a_text, PART3A_FREEZE),
            "kq_within_all_count": len(kq_within_all_rows),
            "channel_split_within_all_count": len(channel_split_within_all_rows),
        },
    )


# 関数: Trial-4 exploratory declaration gate を構築する。

def build_gate(
    common_inputs: dict,
    structural_audit: dict,
    pilot: dict,
    prior_disposition: dict,
) -> dict:
    """Freeze the Trial-4 exploratory declaration and the v3.0-hold recommendation."""
    exploratory_condition = bool(
        structural_audit["summary"]["exploratory_color_like_foothold_available"]
        and pilot["summary"]["overall_trial4_qualitative_foothold_exists"]
    )
    v3_mainline_promotion_ready = bool(
        structural_audit["summary"]["su3_analogy_structural_pass"]
        and (
            pilot["summary"]["running_qualitative_foothold_available"]
            or pilot["summary"]["confinement_qualitative_foothold_available"]
        )
    )
    v3_hold_recommended = not v3_mainline_promotion_ready

    return payload(
        "8.7.56.16",
        "trial4_exploratory_declaration_v3_hold_gate",
        common_inputs,
        "Freeze whether Trial-4 closes as an exploratory foothold with a v3.0 hold recommendation, or whether it already upgrades into an explicit next-mainline strong-interaction route.",
        {
            "gate_rule": "close Trial-4 exploratory branch once the source inventory is complete, the color-like candidate is explicit, and the remaining gaps are honestly classified as missing non-Abelian / running / confinement structure",
            "promotion_rule": "promotion beyond exploratory mode requires explicit SU(3)-like closure plus an honest running or confinement foothold under the current canon",
            "next_route_rule": "once Trial-4 exploratory gate closes, move to the integrated v2.0 closeout / v3.0 hold contract branch",
        },
        [
            row(
                "trial4_exploratory_branch_closeable",
                "pass" if exploratory_condition else "reject",
                "Trial-4 exploratory branch closeable",
                1 if exploratory_condition else 0,
                "The exploratory branch closes once at least one honest foothold survives and the missing items are transparently classified as v3.0-grade work.",
            ),
            row(
                "trial4_exploratory_foothold_confirmed",
                "pass" if exploratory_condition else "reject",
                "Trial-4 exploratory foothold confirmed",
                1 if exploratory_condition else 0,
                "Current canon retains a color-like candidate and a phenomenological pion-scale-to-nuclear foothold.",
            ),
            row(
                "trial4_v3_mainline_promotion_ready",
                "pass" if v3_mainline_promotion_ready else "reject",
                "Trial-4 v3.0 mainline promotion ready",
                1 if v3_mainline_promotion_ready else 0,
                "Promotion would require explicit non-Abelian closure and a running/confinement route, which are still absent.",
            ),
            row(
                "trial4_v3_hold_recommended",
                "pass" if v3_hold_recommended else "reject",
                "Trial-4 v3.0 hold recommended",
                1 if v3_hold_recommended else 0,
                "The honest declaration is exploratory foothold retained, v3.0 hold recommended.",
            ),
            row(
                "v2_program_integrated_closeout_ready_after_trial4",
                "pass" if exploratory_condition else "reject",
                "v2.0 program integrated closeout ready after Trial-4",
                1 if exploratory_condition else 0,
                "After Trial-4 closes as exploratory, the next official work is the integrated v2.0 closeout / v3.0 hold contract branch.",
            ),
        ],
        {
            "trial4_pass_level": "exploratory_color_like_and_hadron_scale_foothold_v3_hold_recommended" if exploratory_condition else "exploratory_gate_open",
            "trial4_exploratory_branch_closeable": exploratory_condition,
            "trial4_v3_mainline_promotion_ready": v3_mainline_promotion_ready,
            "trial4_v3_hold_recommended": v3_hold_recommended,
            "v2_program_integrated_closeout_ready_after_trial4": exploratory_condition,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial4_exploratory_gate_closed_v3_hold_recommended" if exploratory_condition else "trial4_exploratory_gate_open",
            "advance_to_next_route": exploratory_condition,
            "next_required_artifacts": [] if exploratory_condition else ["trial4_exploratory_declaration_v3_hold_gate"],
        },
        {
            "structural_audit_summary": structural_audit["summary"],
            "qualitative_pilot_summary": pilot["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )


# 関数: main routine を実行する。

def main() -> None:
    """Run the Trial-4 exploratory branch and emit all metrics artifacts."""
    for path in [
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART3A,
        TRIAL2_PAPER_SYNC_GATE,
        TRIAL2_TRIAL4_DISPOSITION,
        TRIAL3_CLOSEOUT_GATE,
        QCD_BASELINE,
        SIGNED_V2,
        KQ_SCAN,
        CHANNEL_SPLIT,
    ]:
        req(path)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    primary_sources_text = read_text(PRIMARY_SOURCES)
    trial2_gate = read_json(TRIAL2_PAPER_SYNC_GATE)
    prior_disposition = read_json(TRIAL2_TRIAL4_DISPOSITION)
    trial3_gate = read_json(TRIAL3_CLOSEOUT_GATE)
    qcd_baseline = read_json(QCD_BASELINE)
    signed_v2 = read_json(SIGNED_V2)
    kq_scan = read_json(KQ_SCAN)
    channel_split = read_json(CHANNEL_SPLIT)

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "primary_sources_markdown": rel(PRIMARY_SOURCES),
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_trial2_paper_side_sync_reopened_declaration_gate_json": rel(TRIAL2_PAPER_SYNC_GATE),
        "mass_origin_v2_trial2_paper_sync_trial4_disposition_44th_refresh_json": rel(TRIAL2_TRIAL4_DISPOSITION),
        "mass_origin_v2_t3_t2_coupled_localization_closeout_declaration_gate_json": rel(TRIAL3_CLOSEOUT_GATE),
        "qcd_hadron_masses_baseline_metrics_json": rel(QCD_BASELINE),
        "nuclear_effective_potential_pion_constrained_signed_v2_metrics_json": rel(SIGNED_V2),
        "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_metrics_json": rel(KQ_SCAN),
        "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_metrics_json": rel(CHANNEL_SPLIT),
    }

    inventory = build_inventory(
        common_inputs,
        part1_text,
        part3a_text,
        primary_sources_text,
        qcd_baseline,
        signed_v2,
        kq_scan,
        channel_split,
        trial2_gate,
        trial3_gate,
    )
    structural_audit = build_structural_audit(common_inputs, part1_text, part3a_text, inventory)
    pilot = build_running_confinement_pilot(
        common_inputs,
        part3a_text,
        qcd_baseline,
        signed_v2,
        kq_scan,
        channel_split,
        structural_audit,
    )
    gate = build_gate(common_inputs, structural_audit, pilot, prior_disposition)

    write_artifact("mass_origin_v2_trial4_nonabelian_color_like_internal_degree_inventory", inventory)
    write_artifact("mass_origin_v2_trial4_su3_analogy_structural_audit", structural_audit)
    write_artifact("mass_origin_v2_trial4_running_confinement_qualitative_pilot", pilot)
    write_artifact("mass_origin_v2_trial4_exploratory_declaration_v3_hold_gate", gate)

    print("[ok] generated Trial-4 exploratory artifacts:")
    print(" - mass_origin_v2_trial4_nonabelian_color_like_internal_degree_inventory_metrics.json")
    print(" - mass_origin_v2_trial4_su3_analogy_structural_audit_metrics.json")
    print(" - mass_origin_v2_trial4_running_confinement_qualitative_pilot_metrics.json")
    print(" - mass_origin_v2_trial4_exploratory_declaration_v3_hold_gate_metrics.json")


if __name__ == "__main__":
    main()
