from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ConstantSpec:
    code: str
    url: str
    relpath: str


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0` を満たす経路を評価する。
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return

    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req, timeout=30) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    # 条件分岐: `out_path.stat().st_size == 0` を満たす経路を評価する。

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s)


def _parse_value_cell(html_text: str, *, label: str) -> tuple[Decimal, str, str]:
    """
    Extract "Numerical value" / "Standard uncertainty" rows from NIST Cuu pages.

    Returns (value, unit, raw_text). If the value is "(exact)", returns value=0 and empty unit.
    """
    m = re.search(rf"{re.escape(label)}\s*</TD>\s*<TD[^>]*>(.*?)</TD>", html_text, flags=re.I | re.S)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"missing row: {label}")

    cell_html = m.group(1)
    cell_txt = _strip_tags(html_lib.unescape(cell_html))
    cell_txt = " ".join(cell_txt.split())

    # 条件分岐: `"exact" in cell_txt.lower()` を満たす経路を評価する。
    if "exact" in cell_txt.lower():
        return Decimal(0), "", cell_txt

    # 条件分岐: `"x 10" in cell_txt` を満たす経路を評価する。

    if "x 10" in cell_txt:
        left, right = cell_txt.split("x 10", 1)
        # Some SI-defined constants have infinite decimal expansions and are shown with "..." in NIST Cuu.
        left_clean = left.replace("...", "").strip()
        num = Decimal(re.sub(r"\s+", "", left_clean))
        right = right.strip()
        parts = right.split(" ", 1)
        exp = int(parts[0])
        unit = parts[1].strip() if len(parts) > 1 else ""
        value = num * (Decimal(10) ** exp)
        return value, unit, cell_txt

    # Example: speed of light: "299 792 458 m s -1"

    tokens = cell_txt.split()
    num_tokens: list[str] = []
    rest_tokens: list[str] = []
    for tok in tokens:
        # 条件分岐: `re.fullmatch(r"[0-9]+(\.[0-9]+)?", tok)` を満たす経路を評価する。
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?", tok):
            num_tokens.append(tok)
        else:
            rest_tokens.append(tok)

    # 条件分岐: `not num_tokens` を満たす経路を評価する。

    if not num_tokens:
        raise ValueError(f"cannot parse numeric value: {cell_txt!r}")

    value = Decimal("".join(num_tokens))
    unit = " ".join(rest_tokens).strip()
    return value, unit, cell_txt


def _extract_constant(html_text: str, *, expected_codata_year: int | None) -> dict[str, object]:
    mt = re.search(r"<title>\s*CODATA Value:\s*(.*?)</title>", html_text, flags=re.I | re.S)
    # 条件分岐: `not mt` を満たす経路を評価する。
    if not mt:
        raise ValueError("missing <title> CODATA Value")

    name = " ".join(_strip_tags(html_lib.unescape(mt.group(1))).split()).strip()

    ms = re.search(r"Source:\s*(\d{4})\s*CODATA", html_text, flags=re.I)
    codata_year = int(ms.group(1)) if ms else None
    # 条件分岐: `expected_codata_year is not None and codata_year is not None and codata_year...` を満たす経路を評価する。
    if expected_codata_year is not None and codata_year is not None and codata_year != expected_codata_year:
        raise ValueError(f"CODATA year mismatch: got {codata_year}, expected {expected_codata_year}")

    # Most SI-redefined constants use "Numerical value" (not "Value"), and some are "(exact)".

    value, unit_value, raw_value = _parse_value_cell(html_text, label="Numerical value")
    sigma, unit_sigma, raw_sigma = _parse_value_cell(html_text, label="Standard uncertainty")

    unit = unit_value.strip() if unit_value.strip() else unit_sigma.strip()
    # 条件分岐: `unit_sigma.strip() and unit_value.strip() and unit_sigma.strip() != unit_valu...` を満たす経路を評価する。
    if unit_sigma.strip() and unit_value.strip() and unit_sigma.strip() != unit_value.strip():
        raise ValueError(f"unit mismatch: value unit={unit_value!r} vs sigma unit={unit_sigma!r}")

    sigma_si = float(sigma)
    # 条件分岐: `"exact" in str(raw_sigma).lower()` を満たす経路を評価する。
    if "exact" in str(raw_sigma).lower():
        sigma_si = 0.0

    return {
        "name": name,
        "codata_year": codata_year,
        "value_si": float(value),
        "sigma_si": sigma_si,
        "unit_si": unit,
        "raw": {"value_cell_text": raw_value, "sigma_cell_text": raw_sigma},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch NIST Cuu CODATA constants for Phase 7 / Step 7.15 (stat mech / thermo baseline): "
            "blackbody radiation constants (c, h, ħ, k_B, σ). "
            "Writes manifest.json and extracted_values.json under data/quantum/sources/."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--out-dirname",
        default="nist_codata_2022_blackbody_constants",
        help="Output directory name under data/quantum/sources/.",
    )
    ap.add_argument(
        "--expect-codata-year",
        type=int,
        default=2022,
        help="Abort if fetched pages are not from this CODATA year (default: 2022). Use 0 to disable.",
    )
    args = ap.parse_args()

    expected_year = int(args.expect_codata_year)
    expected_year_opt = None if expected_year <= 0 else expected_year

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / str(args.out_dirname)
    src_dir.mkdir(parents=True, exist_ok=True)

    base = "https://physics.nist.gov/cgi-bin/cuu/Value"
    specs: list[ConstantSpec] = [
        ConstantSpec(code="c", url=f"{base}?c", relpath="nist_cuu_Value_c.html"),
        ConstantSpec(code="h", url=f"{base}?h", relpath="nist_cuu_Value_h.html"),
        ConstantSpec(code="hbar", url=f"{base}?hbar", relpath="nist_cuu_Value_hbar.html"),
        ConstantSpec(code="k", url=f"{base}?k", relpath="nist_cuu_Value_k.html"),
        ConstantSpec(code="sigma", url=f"{base}?sigma", relpath="nist_cuu_Value_sigma.html"),
    ]

    # 条件分岐: `not args.offline` を満たす経路を評価する。
    if not args.offline:
        for spec in specs:
            _download(spec.url, src_dir / spec.relpath)

    missing: list[Path] = []
    extracted: dict[str, object] = {}
    for spec in specs:
        path = src_dir / spec.relpath
        # 条件分岐: `not path.exists() or path.stat().st_size == 0` を満たす経路を評価する。
        if not path.exists() or path.stat().st_size == 0:
            missing.append(path)
            continue

        html_text = path.read_text(encoding="utf-8", errors="replace")
        extracted[spec.code] = {
            "url": spec.url,
            "local_path": str(path),
            "local_sha256": _sha256(path),
            **_extract_constant(html_text, expected_codata_year=expected_year_opt),
        }

    # 条件分岐: `missing` を満たす経路を評価する。

    if missing:
        raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

    out_extracted = src_dir / "extracted_values.json"
    out_extracted.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "dataset": "Phase 7 / Step 7.15 blackbody constants (NIST Cuu CODATA)",
                "expected_codata_year": expected_year_opt,
                "constants": extracted,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Phase 7 / Step 7.15 blackbody constants (NIST Cuu CODATA)",
        "notes": [
            "These are CODATA pages from NIST 'Constants, Units and Uncertainty' (Cuu).",
            "The year shown on-page is checked unless --expect-codata-year=0.",
            "The extracted_values.json is derived from the HTML files and is intended for offline reproducible analysis.",
            "Some constants (c,h,k_B,ħ) are exact in the SI; their standard uncertainty appears as '(exact)'.",
        ],
        "constants": [spec.code for spec in specs],
        "files": [],
    }
    for spec in specs:
        path = src_dir / spec.relpath
        manifest["files"].append(
            {"url": spec.url, "path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path).upper()}
        )

    out_manifest = src_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_extracted}")
    print(f"[ok] wrote: {out_manifest}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
