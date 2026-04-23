#!/usr/bin/env python3
"""Generate 8.7.56.827-.830 Trial-2 numeric alpha same-sector-proxy-equivalence-symbol-fragment artifacts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = (
    ROOT
    / "scripts"
    / "quantum"
    / "mass_origin_v2_t2_alpha_same_sector_proxy_equivalence_terminal_glyph_branch.py"
)


# Function: stop execution when a required path is missing.
def require(path: Path) -> None:
    """Require an input path to exist before execution continues."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required template: {path}")


# Function: apply ordered text replacements.

def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    """Apply ordered text replacements to the template source."""
    for old, new in replacements:
        text = text.replace(old, new)

    return text


# Function: advance the terminal-glyph branch template by one residual layer.

def build_transformed_source(template_text: str) -> str:
    """Shift the prior/current/next branch labels from terminal-glyph to symbol-fragment."""
    replacements = [
        ("TERMINAL_ATOM", "__UPPER_PREV__"),
        ("TERMINAL_GLYPH", "__UPPER_CURR__"),
        ("SYMBOL_FRAGMENT", "__UPPER_NEXT__"),
        ("terminal_atom", "__UNDER_PREV__"),
        ("terminal_glyph", "__UNDER_CURR__"),
        ("symbol_fragment", "__UNDER_NEXT__"),
        ("terminal-atom", "__HYP_PREV__"),
        ("terminal-glyph", "__HYP_CURR__"),
        ("symbol-fragment", "__HYP_NEXT__"),
        ("terminal atom", "__PLAIN_PREV__"),
        ("terminal glyph", "__PLAIN_CURR__"),
        ("symbol fragment", "__PLAIN_NEXT__"),
        ("8.7.56.827-.830", "__OLD_NEXT_BRANCH__"),
        ("8.7.56.827", "__OLD_NEXT_STEP__"),
        ("8.7.56.823-.826", "8.7.56.827-.830"),
        ("8.7.56.823", "8.7.56.827"),
        ("8.7.56.824", "8.7.56.828"),
        ("8.7.56.825", "8.7.56.829"),
        ("8.7.56.826", "8.7.56.830"),
        ("advance_to_8_7_56_824", "advance_to_8_7_56_828"),
        ("advance_to_8_7_56_825", "advance_to_8_7_56_829"),
        ("advance_to_8_7_56_826", "advance_to_8_7_56_830"),
        ("one_hundred_third_refresh", "__OLD_NEXT_REFRESH_FILE__"),
        ("one-hundred-third refresh", "__OLD_NEXT_REFRESH_TEXT__"),
        ("one-hundred-third", "__OLD_NEXT_ORDINAL__"),
        ("one_hundred_second_refresh", "one_hundred_third_refresh"),
        ("one-hundred-second refresh", "one-hundred-third refresh"),
        ("one-hundred-second", "one-hundred-third"),
        ("__UPPER_PREV__", "TERMINAL_GLYPH"),
        ("__UPPER_CURR__", "SYMBOL_FRAGMENT"),
        ("__UPPER_NEXT__", "TERMINAL_ATOM"),
        ("__UNDER_PREV__", "terminal_glyph"),
        ("__UNDER_CURR__", "symbol_fragment"),
        ("__UNDER_NEXT__", "terminal_atom"),
        ("__HYP_PREV__", "terminal-glyph"),
        ("__HYP_CURR__", "symbol-fragment"),
        ("__HYP_NEXT__", "terminal-atom"),
        ("__PLAIN_PREV__", "terminal glyph"),
        ("__PLAIN_CURR__", "symbol fragment"),
        ("__PLAIN_NEXT__", "terminal atom"),
        ("__OLD_NEXT_BRANCH__", "8.7.56.831-.834"),
        ("__OLD_NEXT_STEP__", "8.7.56.831"),
        ("__OLD_NEXT_REFRESH_FILE__", "one_hundred_fourth_refresh"),
        ("__OLD_NEXT_REFRESH_TEXT__", "one-hundred-fourth refresh"),
        ("__OLD_NEXT_ORDINAL__", "one-hundred-fourth"),
    ]
    return apply_replacements(template_text, replacements)


# Function: execute the symbol-fragment branch by shifting the terminal-glyph template.

def main() -> None:
    """Execute the Trial-2 numeric alpha same-sector-proxy-equivalence-symbol-fragment residual branch."""
    require(TEMPLATE)
    transformed_source = build_transformed_source(TEMPLATE.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {
        "__file__": str(Path(__file__).resolve()),
        "__name__": "wavep_generated_same_sector_proxy_symbol_fragment_branch",
    }
    exec(compile(transformed_source, str(TEMPLATE), "exec"), namespace)
    run_cli = namespace.get("run_cli")
    if not callable(run_cli):
        raise SystemExit("[fail] transformed branch did not expose run_cli()")

    run_cli()


# Function: run the same-sector-proxy-equivalence-symbol-fragment residual branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the same-sector-proxy-equivalence-symbol-fragment residual branch."""
    main()


if __name__ == "__main__":
    run_cli()
