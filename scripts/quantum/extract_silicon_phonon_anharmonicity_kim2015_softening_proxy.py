from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(s: str) -> str:
    # Unify common unicode variants for robust regex parsing.
    out = str(s)
    out = out.replace("\u2212", "-")  # minus sign
    out = out.replace("\u00b1", "±")  # plus-minus
    out = out.replace("\ufb01", "fi")  # ligature artifacts
    out = out.replace("\ufb00", "ff")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for p in reader.pages:
        chunks.append(p.extract_text() or "")

    return _normalize_text("\n".join(chunks))


def _extract_first_float(text: str, *, pattern: str, label: str) -> float:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"missing {label}")

    return float(m.group(1))


def _extract_first_two_floats(text: str, *, pattern: str, label: str) -> tuple[float, float]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"missing {label}")

    return (float(m.group(1)), float(m.group(2)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract a simple temperature-dependent phonon softening proxy from the accepted manuscript PDF\n"
            "Kim et al., Phys. Rev. B 91, 014307 (2015) as cached via OSTI.\n"
            "This is used as a primary constraint candidate for Phase 7 / Step 7.14.20."
        )
    )
    ap.add_argument(
        "--source-dirname",
        default="osti_kim2015_prb91_014307_si_phonon_anharmonicity",
        help="Source directory name under data/quantum/sources/.",
    )
    ap.add_argument(
        "--pdf-name",
        default="osti_purl_1185765.pdf",
        help="Cached PDF filename under the source directory.",
    )
    args = ap.parse_args()

    root = _repo_root()
    src_dir = root / "data" / "quantum" / "sources" / str(args.source_dirname)
    extracted_json = src_dir / "extracted_values.json"
    pdf_path = src_dir / str(args.pdf_name)

    # 条件分岐: `not (extracted_json.exists() and extracted_json.stat().st_size > 0)` を満たす経路を評価する。
    if not (extracted_json.exists() and extracted_json.stat().st_size > 0):
        raise SystemExit(f"[fail] missing: {extracted_json}")

    # 条件分岐: `not (pdf_path.exists() and pdf_path.stat().st_size > 0)` を満たす経路を評価する。

    if not (pdf_path.exists() and pdf_path.stat().st_size > 0):
        raise SystemExit(f"[fail] missing: {pdf_path}")

    meta: dict[str, Any] = json.loads(extracted_json.read_text(encoding="utf-8"))
    text = _extract_pdf_text(pdf_path)

    # Temperature range: "from 100 to 1500 K"
    t_min_k, t_max_k = _extract_first_two_floats(
        text,
        pattern=r"from\s+([0-9.]+)\s+to\s+([0-9.]+)\s*K",
        label="temperature range (from ... to ... K)",
    )

    # Isobaric mean fractional energy shift: "<Δε_i/ε_i>=-0.07"
    frac_isobaric = _extract_first_float(
        text,
        pattern=r"mean fractional energy shifts.*?=\s*([-+]?\d+\.\d+)",
        label="mean fractional energy shift (isobaric)",
    )

    # Isothermal comparator in the abstract: "isothermal parameter of <Δε_i/ε_i>=-0.01"
    frac_isothermal = _extract_first_float(
        text,
        pattern=r"isothermal parameter of.*?=\s*([-+]?\d+\.\d+)",
        label="mean fractional energy shift (isothermal)",
    )

    # Mean isobaric Grüneisen parameter: "mean isobaric ... parameter of 6.95 ±0.67"
    gamma_isobaric, gamma_isobaric_pm = _extract_first_two_floats(
        text,
        pattern=r"mean isobaric.*?parameter of\s*([0-9.]+)\s*±\s*([0-9.]+)",
        label="mean isobaric Gruneisen parameter ±",
    )

    # The table lists an isothermal mean ~0.98 (no uncertainty reported in the accepted manuscript cache).
    gamma_isothermal: float | None = None
    m_gi = re.search(r"\s\u03b3\u0304P\s*([0-9.]+)\s*-", text)
    # 条件分岐: `m_gi` を満たす経路を評価する。
    if m_gi:
        gamma_isothermal = float(m_gi.group(1))

    # Build a minimal proxy: assume linear-in-T fractional shift anchored at T_min.
    # This matches the abstract's (100K→1500K) statement and is used only as a frozen constraint candidate.

    proxy = {
        "kind": "linear_fractional_energy_shift",
        "t_ref_K": float(t_min_k),
        "t_max_K": float(t_max_k),
        "fractional_energy_shift_at_t_max_isobaric": float(frac_isobaric),
        "fractional_energy_shift_at_t_max_isothermal": float(frac_isothermal),
        "mean_gruneisen_isobaric": {"value": float(gamma_isobaric), "pm": float(gamma_isobaric_pm)},
        "mean_gruneisen_isothermal": (None if gamma_isothermal is None else float(gamma_isothermal)),
        "definition": (
            "scale(T) = 1 + s*(T - t_ref)/(t_max - t_ref), "
            "with s = fractional_energy_shift_at_t_max_isobaric (relative to t_ref)."
        ),
        "notes": [
            "Values are parsed from the accepted manuscript text (PDF) cached under data/quantum/sources/.",
            "This proxy is intentionally low-parameter: it freezes a single global softening scale for ω (all modes).",
            "It is a candidate constraint for Step 7.14.20; success requires strict + holdout to pass in α(T) tests.",
        ],
    }

    meta.setdefault("parsed_from_pdf", {})
    meta["parsed_from_pdf"] = {
        "updated_utc": _iso_utc_now(),
        "pdf_name": str(pdf_path.name),
        "temperature_range_K": {"min": float(t_min_k), "max": float(t_max_k)},
        "mean_fractional_energy_shift": {"isobaric": float(frac_isobaric), "isothermal": float(frac_isothermal)},
        "mean_gruneisen": {"isobaric": {"value": float(gamma_isobaric), "pm": float(gamma_isobaric_pm)}, "isothermal": gamma_isothermal},
        "softening_proxy": proxy,
    }

    extracted_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] updated: {extracted_json}")
    # Avoid non-ASCII glyphs (e.g., en-dash/±) to keep Windows console output stable.
    print(
        f"[ok] parsed: T={t_min_k:g}-{t_max_k:g} K, "
        f"frac_shift_isobaric={frac_isobaric:+.3f}, "
        f"gamma_isobaric={gamma_isobaric:.3f}+/-{gamma_isobaric_pm:.3f}"
    )


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
