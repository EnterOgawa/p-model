#!/usr/bin/env python3
"""Refresh the current Trial-2 numeric-alpha expert bundle while external response is pending."""

from __future__ import annotations

import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRIVATE_OUT = ROOT / "output" / "private" / "quantum"
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
SUMMARY_OUT = ROOT / "output" / "private" / "summary"

EXPERT_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_two_sector_hierarchy.md")

FILES_TO_COPY = (
    ROOT / "doc" / "STATUS.md",
    ROOT / "doc" / "ROADMAP.md",
    ROOT / "doc" / "AI_CONTEXT_MIN.json",
    ROOT / "doc" / "WORK_HISTORY_RECENT.md",
    ROOT / "doc" / "PRIMARY_SOURCES.md",
    ROOT / "doc" / "paper" / "10_part1_core_theory.md",
    ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md",
    ROOT / "doc" / "paper" / "14_part5_future_predictions.md",
    ROOT / "doc" / "quantum" / "16_electromagnetism_charge_maxwell_photon.md",
    EXPERT_NOTE,
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_statement_source_inventory_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_statement_audit_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_statement_declaration_gate_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_second_refresh_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_literal_source_inventory_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_literal_audit_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_literal_declaration_gate_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_third_refresh_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_source_inventory_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_audit_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_declaration_gate_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_fourth_refresh_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh_source_inventory_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh_audit_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh_declaration_gate_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_fifth_refresh_metrics.json",
    ROOT / "scripts" / "quantum" / "t2a_1019.py",
    ROOT / "scripts" / "quantum" / "t2a_1023.py",
    ROOT / "scripts" / "quantum" / "t2a_1027.py",
    ROOT / "scripts" / "quantum" / "t2a_1031.py",
    SUMMARY_OUT / "pmodel_paper.pdf",
    SUMMARY_OUT / "pmodel_paper_part3a_quantum_foundations.pdf",
    SUMMARY_OUT / "pmodel_paper_part5_future_predictions.pdf",
)


# Function: return the current UTC stamp for bundle naming.
def utc_stamp() -> str:
    """Return the current UTC stamp for the bundle name."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# Function: fail early when a required input is missing.

def require(path: Path) -> None:
    """Require one path to exist before continuing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: write one UTF-8 text file into the bundle.

def write_text(path: Path, text: str) -> None:
    """Write one UTF-8 text file into the bundle."""
    path.write_text(text, encoding="utf-8")


# Function: copy one source file into the bundle directory.

def copy_file(bundle_dir: Path, source: Path) -> str:
    """Copy one source file into the bundle directory and return the copied filename."""
    destination = bundle_dir / source.name
    shutil.copy2(source, destination)
    return destination.name


# Function: build the latest expert bundle from the current waiting state.

def build_bundle() -> dict[str, object]:
    """Build a new expert-review bundle for the current external-wait state."""
    stamp = utc_stamp()
    bundle_dir = PRIVATE_OUT / f"expert_review_bundle_{stamp}"
    bundle_zip = PRIVATE_OUT / f"expert_review_bundle_{stamp}.zip"

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    if bundle_zip.exists():
        bundle_zip.unlink()

    bundle_dir.mkdir(parents=True, exist_ok=True)

    write_text(
        bundle_dir / "README.txt",
        """Expert review bundle

Purpose
- Current route: Trial-2 numeric alpha two-sector hierarchy pivot.
- Current official blocker: no positive public EM-sector normalization surface under current canon.
- Current operational state: mechanical wording descent is stopped; expert-response intake is the only next official route.
""",
    )
    write_text(
        bundle_dir / "EXPERT_NOTE.txt",
        """Expert note

Current fixed state
- The computation route, electron-identification dictionary, H0^(P)-Z_P pivot, raw final computation, and retry triage judgment are fixed.
- The two-sector hierarchy memo is the current alternate-computation pivot.
- Current canon still exposes a single-Z_P photon normalization surface plus local Maxwell/QED adoption.
- Statement/literal search for positive EM-sector normalization produced no new public-canonical surface.
- The roadmap is therefore waiting for external expert guidance before any further official branch can start.
""",
    )
    write_text(
        bundle_dir / "QUESTIONS_FOR_REVIEW.txt",
        """Questions for review

1. Under the current public canon, is there a defensible positive public statement or formula equivalent to Z_P^EM = 1 and therefore e = g_P?
2. If yes, what is the minimal statement / literal / formula, and where is it located?
3. If no such public statement exists, should Trial-2 numeric alpha now close as structural pass / numeric open?
4. If a reconciliation is possible, what is the minimal bridge that makes the single-Z_P photon canon compatible with the proposed two-sector hierarchy?
""",
    )

    copied: list[str] = []
    for source in FILES_TO_COPY:
        require(source)
        copied.append(copy_file(bundle_dir, source))

    manifest_lines = [
        "Expert review bundle manifest",
        "",
        f"STAMP={stamp}",
        f"COPIED_COUNT={len(copied)}",
        "MISSING_COUNT=0",
        "",
        "[files]",
    ]
    manifest_lines.extend(sorted(copied))
    write_text(bundle_dir / "BUNDLE_MANIFEST.txt", "\n".join(manifest_lines) + "\n")

    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(bundle_dir.iterdir()):
            handle.write(path, arcname=path.name)

    return {
        "stamp": stamp,
        "bundle_dir": bundle_dir,
        "bundle_zip": bundle_zip,
        "copied_count": len(copied),
        "missing_count": 0,
        "staging_file_count": len(list(bundle_dir.iterdir())),
    }


# Function: run the bundle refresh from the CLI.

def main() -> None:
    """Create the current expert-review bundle and print the resulting paths."""
    bundle = build_bundle()
    print(f"[done] bundle_dir={bundle['bundle_dir']}")
    print(f"[done] bundle_zip={bundle['bundle_zip']}")
    print(f"[done] copied_count={bundle['copied_count']}")
    print(f"[done] missing_count={bundle['missing_count']}")
    print(f"[done] staging_file_count={bundle['staging_file_count']}")


if __name__ == "__main__":
    main()
