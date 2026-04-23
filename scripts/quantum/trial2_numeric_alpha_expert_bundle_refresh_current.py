#!/usr/bin/env python3
"""Refresh the current Trial-2 numeric-alpha expert bundle for external review.

This script packages the latest canonical docs, the retained expert notes, the
placeholder-compress / exact-coefficient / residual-scope classification
results, the retained Q-ball public artifacts, and the previously frozen
share-pack bundles into one zip so the current state can be reviewed externally
without rerunning any paper build.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRIVATE_OUT = ROOT / "output" / "private" / "quantum"
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

RETAINED_SHARE_PACK = PRIVATE_OUT / "expert_review_bundle_20260324_125609.zip"
RETAINED_LOOP_PACK = PRIVATE_OUT / "expert_review_bundle_20260324_205844.zip"

NOTE_ZP = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_zp_em_equals_one.md")
NOTE_ALPHA = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_alpha_is_prediction.md")
NOTE_DIMENSION = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_dimension_normalization_review.md")
NOTE_SI = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_si_dimension_tracking.md")
NOTE_PLACEHOLDER = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_placeholder_compress_and_attempt.md")
NOTE_QBALL = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_qball_noether_charge.md")

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
    NOTE_ZP,
    NOTE_ALPHA,
    NOTE_DIMENSION,
    NOTE_SI,
    NOTE_PLACEHOLDER,
    NOTE_QBALL,
    RETAINED_SHARE_PACK,
    RETAINED_LOOP_PACK,
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_current_canon_limit_future_canon_hold_source_inventory_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_dimensionless_alpha_closed_form_attempt_audit_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_dimensionless_alpha_exact_factor_tracking_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_dimensionless_alpha_numeric_evaluation_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_source_inventory_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_audit_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_declaration_gate_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_numeric_evaluation_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_source_inventory_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_audit_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_declaration_gate_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_numeric_evaluation_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_source_inventory_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_audit_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_declaration_gate_metrics.json",
    PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_numeric_evaluation_metrics.json",
    PUBLIC_OUT / "mass_origin_qball_charge_mapping_statement_freeze_metrics.json",
    PUBLIC_OUT / "mass_origin_qball_charge_operator_normalization_audit_metrics.json",
    PUBLIC_OUT / "mass_origin_qball_charge_discrete_frequency_inversion_metrics.json",
    ROOT / "scripts" / "quantum" / "t2a_1207.py",
    ROOT / "scripts" / "quantum" / "t2a_1211.py",
    ROOT / "scripts" / "quantum" / "t2a_1215.py",
    ROOT / "scripts" / "quantum" / "t2a_1219.py",
)


# Function: return a compact UTC timestamp when one is not provided.
def default_stamp() -> str:
    """Return a compact UTC timestamp suitable for bundle naming."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# Function: return a stable display path for manifest entries.

def display_path(path: Path) -> str:
    """Return a repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: fail early when a required input file is missing.

def require(path: Path) -> None:
    """Require one input path to exist before continuing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: write one UTF-8 text file into the staging directory.

def write_text(path: Path, text: str) -> None:
    """Write one UTF-8 text file."""
    path.write_text(text, encoding="utf-8")


# Function: copy one source file into the staging directory.

def copy_file(bundle_dir: Path, source: Path) -> str:
    """Copy one source file into the staging directory and return the filename."""
    destination = bundle_dir / source.name
    shutil.copy2(source, destination)
    return destination.name


# Function: build the README for the current expert-review bundle.

def readme_text(bundle_zip_name: str) -> str:
    """Return the canonical README text for the current expert-review bundle."""
    return (
        "Expert review bundle\n\n"
        "Purpose\n"
        "- Current route: Trial-2 numeric alpha adopted-U(1) external-import primary lane.\n"
        "- Latest completed official block: 8.7.56.1219-.1222.\n"
        "- The current public pack now fixes the residual carry order as adopted-U(1) external import primary, future-canon bridge secondary, and Q-ball Noether-charge reserve.\n"
        "- This bundle is for expert review on whether that carry order is the honest reading of the current pack and what kind of bridge would be needed to move beyond it.\n\n"
        "Current state\n"
        "- Current canon reopen: false.\n"
        "- Physical reject required: false.\n"
        "- Current canon explicit charge bridge: false.\n"
        "- Required residual coefficient: 0.30282212087175264.\n"
        "- Retained Q-ball ground-state charge: 0.9997806376467893.\n"
        "- Retained Q-ball alpha candidate: 0.07954256277236127.\n"
        "- Next official branch: 8.7.56.1223-.1226 adopted-U(1) external-import primary-lane contract.\n"
        "- Residual carry order: adopted-U(1) external import primary, future-canon bridge secondary, Q-ball Noether-charge reserve.\n\n"
        "Canonical bundle\n"
        f"- Zip: {bundle_zip_name}\n"
        "- Source markdown and JSON metrics are canonical; paper build was not rerun for this refresh.\n"
    )


# Function: build the bundle note for the current expert-review bundle.

def bundle_note_text() -> str:
    """Return the canonical note text for the current expert-review bundle."""
    return (
        "Current-state bundle note\n\n"
        "Frozen result\n"
        "- The placeholder chain was compressed at .1207-.1210 into one current-canon-limit future-canon hold state.\n"
        "- The exact action-level coefficient family was closed at .1211-.1214 with C_total = 1.\n"
        "- The .1215-.1218 review fixed that current canon still does not derive the required coefficient 0.30282212087175264.\n"
        "- The .1219-.1222 review fixed that the new Q-ball Noether-charge note does not open an independent normalization lane; it only strengthens the residual carry order.\n"
        "- Physical reject is still not selected.\n\n"
        "Why this bundle exists\n"
        "- This pack is meant to expose the post-compression state honestly.\n"
        "- It does not claim a reopen, a numeric alpha closeout, or a physical reject.\n"
        "- It asks whether the current residual carry order is correct and whether any explicit bridge exists beyond adopted-U(1) external import.\n"
    )


# Function: build the review questions for the current expert-review bundle.

def questions_text() -> str:
    """Return the canonical question text for the current expert-review bundle."""
    return (
        "Questions for review\n\n"
        "1. Does the current public pack contain any explicit bridge that turns the structural route e = g_P/sqrt(Z_P) into the public elementary charge with coefficient 0.30282212087175264?\n"
        "2. If not, is the honest primary reading now adopted-U(1) external import with future-canon bridge secondary and Q-ball Noether-charge reserve?\n"
        "3. Does the adopted U(1) stance merely preserve the external/public QED sector, or is there already enough current wording to elevate it into a numeric coefficient bridge?\n"
        "4. Given that the retained Q-ball ground-state charge is 0.9997806376467893, should the Q-ball Noether-charge route remain reserve evidence only?\n"
    )


# Function: build the manifest text for the current expert-review bundle.

def manifest_text(stamp: str, copied_sources: list[Path]) -> str:
    """Return the manifest text for the current expert-review bundle."""
    lines = [
        "Current expert review bundle manifest",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"STAMP={stamp}",
        f"COPIED_COUNT={len(copied_sources)}",
        "",
    ]
    lines.extend(display_path(path) for path in copied_sources)
    return "\n".join(lines) + "\n"


# Function: create the current expert-review bundle on disk.

def build_bundle(stamp: str) -> dict[str, object]:
    """Create the current expert-review bundle and return the resulting paths."""
    bundle_dir = PRIVATE_OUT / f"expert_review_bundle_{stamp}"
    bundle_zip = PRIVATE_OUT / f"expert_review_bundle_{stamp}.zip"

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    if bundle_zip.exists():
        bundle_zip.unlink()

    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied_sources: list[Path] = []
    for source in FILES_TO_COPY:
        require(source)
        copy_file(bundle_dir, source)
        copied_sources.append(source)

    write_text(bundle_dir / "README.txt", readme_text(bundle_zip.name))
    write_text(bundle_dir / "BUNDLE_NOTE.txt", bundle_note_text())
    write_text(bundle_dir / "QUESTIONS_FOR_REVIEW.txt", questions_text())
    write_text(bundle_dir / "BUNDLE_MANIFEST.txt", manifest_text(stamp, copied_sources))

    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(bundle_dir.iterdir()):
            if path.is_file():
                handle.write(path, arcname=path.name)

    with zipfile.ZipFile(bundle_zip, "r") as handle:
        zip_file_count = len(handle.namelist())

    return {
        "bundle_dir": bundle_dir,
        "bundle_zip": bundle_zip,
        "copied_count": len(copied_sources),
        "staging_file_count": len(list(bundle_dir.iterdir())),
        "zip_file_count": zip_file_count,
    }


# Function: parse CLI arguments for the bundle-refresh script.

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", help="UTC timestamp suffix to force into the bundle name.")
    return parser.parse_args()


# Function: run the current expert-bundle refresh from the CLI.

def main() -> None:
    """Build the current expert-review bundle and print the resulting paths."""
    args = parse_args()
    bundle = build_bundle(args.stamp or default_stamp())
    print(f"[done] bundle_dir={bundle['bundle_dir']}")
    print(f"[done] bundle_zip={bundle['bundle_zip']}")
    print(f"[done] copied_count={bundle['copied_count']}")
    print(f"[done] staging_file_count={bundle['staging_file_count']}")
    print(f"[done] zip_file_count={bundle['zip_file_count']}")


if __name__ == "__main__":
    main()
