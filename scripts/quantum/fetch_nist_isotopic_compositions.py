from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sanitize_token(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "unknown"


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return
    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req) as resp, out_path.open("wb") as f:
        f.write(resp.read())
    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._in_tr = False
        self._in_cell = False
        self._cur_row: list[str] = []
        self._cur_cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "tr":
            self._in_tr = True
            self._cur_row = []
            return
        if t in ("td", "th") and self._in_tr:
            self._in_cell = True
            self._cur_cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._in_tr and self._in_cell:
            txt = "".join(self._cur_cell_parts)
            txt = html_module.unescape(txt).replace("\u00a0", " ")
            txt = re.sub(r"\s+", " ", txt).strip()
            self._cur_row.append(txt)
            self._cur_cell_parts = []
            self._in_cell = False
            return
        if t == "tr" and self._in_tr:
            if self._cur_row:
                self.rows.append(self._cur_row)
            self._cur_row = []
            self._in_tr = False

    def handle_data(self, data: str) -> None:
        if self._in_tr and self._in_cell:
            self._cur_cell_parts.append(data)


def _parse_value_with_paren_unc(s: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse e.g.:
      '1.007 825 032 23(9)' -> (1.00782503223, 9e-11)
      '0.999 885(70)' -> (0.999885, 7e-5)
      '' -> (None, None)
    """
    ss = str(s).strip()
    if not ss:
        return None, None
    ss = ss.replace(" ", "")
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?:\((\d+)\))?", ss)
    if not m:
        return None, None
    val_str = m.group(1)
    unc_str = m.group(2)
    try:
        val = float(val_str)
    except Exception:
        return None, None
    if not unc_str:
        return val, None
    decimals = 0
    if "." in val_str:
        decimals = len(val_str.split(".", 1)[1])
    try:
        unc = int(unc_str) * (10.0 ** (-decimals))
    except Exception:
        unc = None
    return val, unc


def _parse_range(s: str) -> Optional[list[float]]:
    """
    Parse e.g. '[1.007 84, 1.008 11]' -> [1.00784, 1.00811]
    """
    ss = str(s).strip()
    if not ss:
        return None
    m = re.fullmatch(r"\[(.+),(.+)\]", ss)
    if not m:
        return None
    left = m.group(1).strip().replace(" ", "")
    right = m.group(2).strip().replace(" ", "")
    try:
        return [float(left), float(right)]
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch NIST isotopic compositions (relative atomic masses) for an element.")
    ap.add_argument("--element", default="H", help="Element symbol (e.g., H). Default: H.")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: do not download; only validate local cache files exist.",
    )
    args = ap.parse_args(argv)

    element = str(args.element).strip()
    if not element:
        raise SystemExit("[fail] --element is empty")

    root = _repo_root()
    elem_token = _sanitize_token(element)
    src_dir = root / "data" / "quantum" / "sources" / f"nist_isotopic_compositions_{elem_token}"
    src_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele={element}"
    raw_path = src_dir / f"nist_isotopic_compositions__{elem_token}.html"
    extracted_path = src_dir / "extracted_values.json"
    manifest_path = src_dir / "manifest.json"

    if args.offline:
        missing: list[Path] = []
        for p in (raw_path, extracted_path, manifest_path):
            if not p.exists():
                missing.append(p)
        if missing:
            raise SystemExit("[fail] missing cache files:\n" + "\n".join(f"- {p}" for p in missing))
        print("[ok] offline check passed")
        return 0

    _download(url, raw_path)

    html = raw_path.read_text(encoding="utf-8", errors="replace")
    parser = _HTMLTableParser()
    parser.feed(html)

    atomic_number: Optional[int] = None
    standard_atomic_weight_range_u: Optional[list[float]] = None
    isotopes: list[dict[str, Any]] = []

    for row in parser.rows:
        # Example (H): ['1','H','1','1.007 825 032 23(9)','0.999 885(70)','[1.007 84, 1.008 11]','m']
        # Example (D): ['D','2','2.014 101 778 12(12)','0.000 115(70)']
        if not row:
            continue
        if "Isotope" in row and "Relative Atomic Mass" in row:
            continue
        if row[0].strip() == "Quantity" and "Value" in row:
            continue

        rec: dict[str, Any] = {}
        if len(row) >= 7 and re.fullmatch(r"\d+", row[0].strip() or ""):
            # First isotope row includes atomic number and standard atomic weight range.
            atomic_number = int(row[0].strip())
            sym = row[1].strip()
            mass_number = int(row[2].strip())
            mass_text = row[3]
            comp_text = row[4]
            std_weight_text = row[5]
            notes = (row[6].strip() or None)
            if standard_atomic_weight_range_u is None:
                standard_atomic_weight_range_u = _parse_range(std_weight_text)
        elif len(row) >= 3 and re.fullmatch(r"\d+", row[1].strip() or ""):
            sym = row[0].strip()
            mass_number = int(row[1].strip())
            mass_text = row[2] if len(row) >= 3 else ""
            comp_text = row[3] if len(row) >= 4 else ""
            notes = None
        else:
            continue

        mass_u, mass_unc_u = _parse_value_with_paren_unc(mass_text)
        comp, comp_unc = _parse_value_with_paren_unc(comp_text)
        rec = {
            "symbol": sym,
            "mass_number": mass_number,
            "relative_atomic_mass_u": mass_u,
            "relative_atomic_mass_unc_u": mass_unc_u,
            "isotopic_composition": comp,
            "isotopic_composition_unc": comp_unc,
            "notes": notes,
        }
        if mass_u is not None:
            isotopes.append(rec)

    extracted: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "NIST Atomic Weights and Isotopic Compositions (stand_alone.pl)",
        "element": element,
        "query_url": url,
        "raw_file": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "atomic_number": atomic_number,
        "standard_atomic_weight_range_u": standard_atomic_weight_range_u,
        "isotopes": isotopes,
        "notes": [
            "This cache is used to fix primary-source-backed isotope masses for Phase 7 / Step 7.12 reduced-mass scaling.",
            "Uncertainties are parsed from the parenthesis notation as absolute u (atomic mass unit) values.",
        ],
    }
    extracted_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {
        "generated_utc": extracted["generated_utc"],
        "dataset": extracted["dataset"],
        "element": element,
        "query_url": url,
        "raw_file": raw_path.name,
        "raw_sha256": extracted["raw_sha256"],
        "extracted_values": extracted_path.name,
        "notes": [
            "Primary-source cache for isotopic compositions / relative atomic masses (NIST PML).",
            "Used by scripts/quantum/molecular_isotopic_scaling.py for H2/HD/D2 reduced mass precision.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {extracted_path}")
    print(f"[ok] wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

