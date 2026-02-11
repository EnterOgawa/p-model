from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Optional
from urllib.request import Request, urlopen

from pypdf import PdfReader


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


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return
    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req, timeout=60) as resp, out_path.open("wb") as f:
        f.write(resp.read())
    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")
    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


_TEMP_TRANSLATE = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "f": "1",
        "S": "5",
        "&": "6",
        "b": "5",
        "u": "0",
        "U": "0",
        "c": "2",
        "?": "2",
    }
)


_NUM_TRANSLATE = str.maketrans(
    {
        ",": ".",
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "?": "2",
        # Empirically, the PDF text extraction often maps "6" to "b" inside mantissas (e.g., 7.9b7-04 ~ 7.967e-04).
        "b": "6",
    }
)


def _clean_ascii(s: str) -> str:
    ss = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    ss = ss.replace("\u00b0", "°")
    ss = re.sub(r"\s+", " ", ss)
    return ss.strip()


def _parse_temp_c(token: str) -> Optional[float]:
    t = str(token).strip().translate(_TEMP_TRANSLATE)
    t = re.sub(r"[^0-9.+-]", "", t)
    if t.startswith("."):
        t = "0" + t
    if not t:
        return None
    try:
        return float(t)
    except Exception:
        return None


def _parse_nbs_float(token: str) -> Optional[float]:
    s = str(token).strip().translate(_NUM_TRANSLATE)
    s = re.sub(r"[^0-9.+-Ee]", "", s)
    if not s:
        return None
    # Scientific notation like 7.439-04 meaning 7.439e-04.
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)([+-])(\d+)$", s)
    if m:
        try:
            mant = float(m.group(1))
            exp = int(m.group(3))
            if m.group(2) == "-":
                exp = -exp
            return mant * (10.0**exp)
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


_SPLIT_NBS_NUMBER_REPAIRS: list[tuple[re.Pattern[str], str]] = [
    # Join a mantissa split like "1. 910-01" -> "1.910-01" (avoid temperatures like "30. 1.98...").
    (re.compile(r"(\d+)\.\s+(\d{2,}[+-]\d+)\b"), r"\1.\2"),
    # Join a dot split like "1 .468+03" -> "1.468+03".
    (re.compile(r"(\d+)\s+\.(\d{2,}[+-]\d+)\b"), r"\1.\2"),
    # Join exponent split like "1.905 -01" -> "1.905-01".
    (re.compile(r"(\d+\.\d+)\s+([+-]\d{2,})\b"), r"\1\2"),
    # Join exponent digits split like "1.914- 01" -> "1.914-01".
    (re.compile(r"(\d+\.\d+[+-])\s+(\d{2,})\b"), r"\1\2"),
]


def _repair_split_nbs_numbers(clean_line: str) -> str:
    out = clean_line
    for _ in range(3):
        before = out
        for pat, repl in _SPLIT_NBS_NUMBER_REPAIRS:
            out = pat.sub(repl, out)
        if out == before:
            break
    return out


def _find_appendix_e_start(reader: PdfReader) -> Optional[int]:
    for i, p in enumerate(reader.pages):
        t = p.extract_text() or ""
        tc = _compact(t)
        # Use a distinctive phrase from the Appendix E header to avoid false positives earlier in the report.
        if "APPENDIXE" in tc and "0TO50" in tc and "VANDERPAUW" in tc:
            return i
    return None


def _split_pos_for_page(lines: list[str]) -> Optional[int]:
    # In layout extraction, many intra-column gaps exist. We estimate the inter-column split
    # by taking the *largest* whitespace run per line and using its midpoint.
    candidates: list[float] = []
    for ln in lines:
        runs = [(m.start(), m.end() - m.start()) for m in re.finditer(r" {10,}", ln)]
        if not runs:
            continue
        s, l = max(runs, key=lambda x: x[1])
        # Require a sufficiently wide gap to represent the column separator.
        if l >= 20:
            candidates.append(float(s) + float(l) / 2.0)
    if not candidates:
        return None
    return int(round(median(candidates)))


def _split_columns(line: str, split_pos: Optional[int]) -> tuple[str, str]:
    if split_pos is None or split_pos <= 0:
        return line.rstrip(), ""
    left = line[:split_pos].rstrip()
    right = line[split_pos:].strip()
    return left, right


def _is_sample_header(line: str) -> bool:
    u = line.upper()
    if "TYPE" not in u:
        # Some extracted headers degrade "TYPE" (e.g., "TYPF"); still detect via "TYP".
        if "TYP" not in u:
            return False
    # Accept degraded spellings like "P-TYPF", "N-TYPIT", etc.
    return bool(re.search(r"\bP\s*-?\s*TYP", u) or re.search(r"\bN\s*-?\s*TYP", u))


def _infer_type(line: str) -> Optional[str]:
    u = line.upper()
    if re.search(r"\bP\s*-?\s*TYP", u):
        return "p"
    if re.search(r"\bN\s*-?\s*TYP", u):
        return "n"
    return None


def _infer_doping(line: str, *, sample_type: Optional[str]) -> Optional[str]:
    u = line.upper()
    if sample_type == "p":
        if "(AL" in u or "AL)" in u:
            return "Al"
        return "B"
    return None


_RANGE_HEADER_RE = re.compile(r"\brange\b\s*\bof\b", flags=re.IGNORECASE)


@dataclass
class RangeRow:
    t_c: float
    rho: float
    rho_lo: Optional[float]
    rho_hi: Optional[float]
    raw: str


def _parse_range_row(line: str) -> Optional[RangeRow]:
    raw_clean = _clean_ascii(line)
    repaired = _repair_split_nbs_numbers(raw_clean)
    tokens = repaired.split()
    if len(tokens) < 4:
        return None

    t_c = _parse_temp_c(tokens[0])
    rho = _parse_nbs_float(tokens[1])
    rho_lo = _parse_nbs_float(tokens[2])

    rho_hi: Optional[float] = None
    # Look for the "TO" separator token in various degraded forms.
    for i in range(3, len(tokens) - 1):
        sep = tokens[i].upper()
        if sep in {"TO", "T0", "RO", "R0"}:
            rho_hi = _parse_nbs_float(tokens[i + 1])
            break
        if sep == "T" and tokens[i + 1] in {"0", "O", "o"} and i + 2 < len(tokens):
            rho_hi = _parse_nbs_float(tokens[i + 2])
            break
    if rho_hi is None:
        rho_hi = _parse_nbs_float(tokens[3])

    if t_c is None or rho is None:
        return None
    return RangeRow(t_c=float(t_c), rho=float(rho), rho_lo=rho_lo, rho_hi=rho_hi, raw=raw_clean)


def _extract_samples_from_stream(lines: list[str], *, source_pages: list[int]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    reading = False
    rows: list[RangeRow] = []

    def flush() -> None:
        nonlocal cur, reading, rows
        if cur is None:
            rows = []
            reading = False
            return
        if rows:
            # Keep the last parsed range table for the sample.
            cur["rho_range_table"] = [
                {
                    "T_C": r.t_c,
                    "rho_ohm_cm": r.rho,
                    "rho_lo_ohm_cm": r.rho_lo,
                    "rho_hi_ohm_cm": r.rho_hi,
                    "raw": r.raw,
                }
                for r in sorted(rows, key=lambda x: x.t_c)
            ]
        if cur.get("rho_range_table"):
            samples.append(cur)
        cur = None
        rows = []
        reading = False

    for raw in lines:
        line = _clean_ascii(raw)
        if not line:
            continue

        if _is_sample_header(line):
            flush()
            st = _infer_type(line)
            cur = {
                "sample_id": f"sample_{len(samples)+1:03d}",
                "raw_header": line,
                "type": st,
                "doping": _infer_doping(line, sample_type=st),
                "source_pages_pdf_1based": source_pages,
                "rho_range_table": [],
            }
            continue

        if cur is None:
            continue

        if _RANGE_HEADER_RE.search(line) and "TO" not in line.upper():
            reading = True
            rows = []
            continue

        if reading:
            # Stop if another header-like section begins.
            if line.lower().startswith("ranges given"):
                reading = False
                continue
            rr = _parse_range_row(line)
            if rr is None:
                # Some extracted lines merge "logp=..." etc; ignore.
                continue
            rows.append(rr)
            # Typical table has temperatures {0,10,20,23,30,40,50}.
            if len({round(r.t_c, 2) for r in rows}) >= 7:
                reading = False
                # Keep parsing until next sample; if another range table appears, it will overwrite on flush.
                continue

    flush()
    return samples


def _extract_samples(pdf_path: Path) -> dict[str, Any]:
    reader = PdfReader(str(pdf_path))
    start = _find_appendix_e_start(reader)
    if start is None:
        raise ValueError("APPENDIX E not found in PDF text")

    # Scan a bounded window after Appendix E start (covers the data tables in this report).
    pg0 = start
    pg1 = min(len(reader.pages), start + 15)
    pages = list(range(pg0, pg1))

    left_lines: list[str] = []
    right_lines: list[str] = []
    source_pages_1based: list[int] = [i + 1 for i in pages]

    for i in pages:
        t = reader.pages[i].extract_text(extraction_mode="layout") or ""
        ls = t.splitlines()
        split_pos = _split_pos_for_page(ls)
        for ln in ls:
            left, right = _split_columns(ln, split_pos)
            if left.strip():
                left_lines.append(left)
            if right.strip():
                right_lines.append(right)

    left_samples = _extract_samples_from_stream(left_lines, source_pages=source_pages_1based)
    right_samples = _extract_samples_from_stream(right_lines, source_pages=source_pages_1based)

    samples = left_samples + right_samples
    # Assign stable ids in the merged list.
    for i, s in enumerate(samples, start=1):
        s["sample_id"] = f"sample_{i:03d}"

    return {
        "appendix_e_scan_pages_pdf_1based": source_pages_1based,
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch NBS IR 74-496 (NIST/NVL pubs) and extract Appendix E: "
            "silicon resistivity vs temperature ranges (0–50°C) for Step 7.14."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Offline mode: do not download; only validate cache and re-extract.")
    args = ap.parse_args(argv)

    root = _repo_root()
    out_dir = root / "data" / "quantum" / "sources" / "nist_nbsir74_496_silicon_resistivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    url_pdf = "https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nbsir74-496.pdf"
    out_pdf = out_dir / "nbsir74-496.pdf"

    if not args.offline:
        _download(url_pdf, out_pdf)

    if not out_pdf.exists() or out_pdf.stat().st_size == 0:
        raise SystemExit(f"[fail] missing pdf: {out_pdf}")

    extracted = _extract_samples(out_pdf)
    out_extracted = out_dir / "extracted_values.json"
    out_extracted.write_text(
        json.dumps(
            {
                "generated_utc": _iso_utc_now(),
                "dataset": "NBS IR 74-496 Appendix E: silicon resistivity vs temperature (0–50°C) ranges",
                "source_url": url_pdf,
                "inputs": {"pdf_path": str(out_pdf), "pdf_sha256": _sha256(out_pdf)},
                **extracted,
                "notes": [
                    "Values are extracted from the report PDF using pypdf text extraction in layout mode.",
                    "Some confidence-bound entries may be missing if parsing fails; mean values are prioritized.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "generated_utc": _iso_utc_now(),
        "dataset": "NBS IR 74-496 (Legacy IR) PDF",
        "files": [
            {
                "url": url_pdf,
                "path": str(out_pdf),
                "bytes": int(out_pdf.stat().st_size),
                "sha256": _sha256(out_pdf).upper(),
            }
        ],
    }
    out_manifest = out_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_extracted}")
    print(f"[ok] wrote: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
