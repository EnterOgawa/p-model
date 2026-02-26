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


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_sha256` の入出力契約と処理意図を定義する。

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


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0` を満たす経路を評価する。
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return

    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req, timeout=60) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    # 条件分岐: `out_path.stat().st_size == 0` を満たす経路を評価する。

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


# 関数: `_clean_ascii` の入出力契約と処理意図を定義する。
def _clean_ascii(s: str) -> str:
    ss = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    ss = ss.replace("\u00b0", "°")
    ss = re.sub(r"\s+", " ", ss)
    return ss.strip()


# 関数: `_parse_temp_c` の入出力契約と処理意図を定義する。

def _parse_temp_c(token: str) -> Optional[float]:
    t = str(token).strip().translate(_TEMP_TRANSLATE)
    t = re.sub(r"[^0-9.+-]", "", t)
    # 条件分岐: `t.startswith(".")` を満たす経路を評価する。
    if t.startswith("."):
        t = "0" + t

    # 条件分岐: `not t` を満たす経路を評価する。

    if not t:
        return None

    try:
        return float(t)
    except Exception:
        return None


# 関数: `_parse_nbs_float` の入出力契約と処理意図を定義する。

def _parse_nbs_float(token: str) -> Optional[float]:
    s = str(token).strip().translate(_NUM_TRANSLATE)
    s = re.sub(r"[^0-9.+-Ee]", "", s)
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None
    # Scientific notation like 7.439-04 meaning 7.439e-04.

    m = re.match(r"^([+-]?\d+(?:\.\d+)?)([+-])(\d+)$", s)
    # 条件分岐: `m` を満たす経路を評価する。
    if m:
        try:
            mant = float(m.group(1))
            exp = int(m.group(3))
            # 条件分岐: `m.group(2) == "-"` を満たす経路を評価する。
            if m.group(2) == "-":
                exp = -exp

            return mant * (10.0**exp)
        except Exception:
            return None

    try:
        return float(s)
    except Exception:
        return None


# 関数: `_compact` の入出力契約と処理意図を定義する。

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


# 関数: `_repair_split_nbs_numbers` の入出力契約と処理意図を定義する。
def _repair_split_nbs_numbers(clean_line: str) -> str:
    out = clean_line
    for _ in range(3):
        before = out
        for pat, repl in _SPLIT_NBS_NUMBER_REPAIRS:
            out = pat.sub(repl, out)

        # 条件分岐: `out == before` を満たす経路を評価する。

        if out == before:
            break

    return out


# 関数: `_find_appendix_e_start` の入出力契約と処理意図を定義する。

def _find_appendix_e_start(reader: PdfReader) -> Optional[int]:
    for i, p in enumerate(reader.pages):
        t = p.extract_text() or ""
        tc = _compact(t)
        # Use a distinctive phrase from the Appendix E header to avoid false positives earlier in the report.
        if "APPENDIXE" in tc and "0TO50" in tc and "VANDERPAUW" in tc:
            return i

    return None


# 関数: `_split_pos_for_page` の入出力契約と処理意図を定義する。

def _split_pos_for_page(lines: list[str]) -> Optional[int]:
    # In layout extraction, many intra-column gaps exist. We estimate the inter-column split
    # by taking the *largest* whitespace run per line and using its midpoint.
    candidates: list[float] = []
    for ln in lines:
        runs = [(m.start(), m.end() - m.start()) for m in re.finditer(r" {10,}", ln)]
        # 条件分岐: `not runs` を満たす経路を評価する。
        if not runs:
            continue

        s, l = max(runs, key=lambda x: x[1])
        # Require a sufficiently wide gap to represent the column separator.
        if l >= 20:
            candidates.append(float(s) + float(l) / 2.0)

    # 条件分岐: `not candidates` を満たす経路を評価する。

    if not candidates:
        return None

    return int(round(median(candidates)))


# 関数: `_split_columns` の入出力契約と処理意図を定義する。

def _split_columns(line: str, split_pos: Optional[int]) -> tuple[str, str]:
    # 条件分岐: `split_pos is None or split_pos <= 0` を満たす経路を評価する。
    if split_pos is None or split_pos <= 0:
        return line.rstrip(), ""

    left = line[:split_pos].rstrip()
    right = line[split_pos:].strip()
    return left, right


# 関数: `_is_sample_header` の入出力契約と処理意図を定義する。

def _is_sample_header(line: str) -> bool:
    u = line.upper()
    # 条件分岐: `"TYPE" not in u` を満たす経路を評価する。
    if "TYPE" not in u:
        # Some extracted headers degrade "TYPE" (e.g., "TYPF"); still detect via "TYP".
        if "TYP" not in u:
            return False
    # Accept degraded spellings like "P-TYPF", "N-TYPIT", etc.

    return bool(re.search(r"\bP\s*-?\s*TYP", u) or re.search(r"\bN\s*-?\s*TYP", u))


# 関数: `_infer_type` の入出力契約と処理意図を定義する。

def _infer_type(line: str) -> Optional[str]:
    u = line.upper()
    # 条件分岐: `re.search(r"\bP\s*-?\s*TYP", u)` を満たす経路を評価する。
    if re.search(r"\bP\s*-?\s*TYP", u):
        return "p"

    # 条件分岐: `re.search(r"\bN\s*-?\s*TYP", u)` を満たす経路を評価する。

    if re.search(r"\bN\s*-?\s*TYP", u):
        return "n"

    return None


# 関数: `_infer_doping` の入出力契約と処理意図を定義する。

def _infer_doping(line: str, *, sample_type: Optional[str]) -> Optional[str]:
    u = line.upper()
    # 条件分岐: `sample_type == "p"` を満たす経路を評価する。
    if sample_type == "p":
        # 条件分岐: `"(AL" in u or "AL)" in u` を満たす経路を評価する。
        if "(AL" in u or "AL)" in u:
            return "Al"

        return "B"

    return None


_RANGE_HEADER_RE = re.compile(r"\brange\b\s*\bof\b", flags=re.IGNORECASE)


# クラス: `RangeRow` の責務と境界条件を定義する。
@dataclass
class RangeRow:
    t_c: float
    rho: float
    rho_lo: Optional[float]
    rho_hi: Optional[float]
    raw: str


# 関数: `_parse_range_row` の入出力契約と処理意図を定義する。

def _parse_range_row(line: str) -> Optional[RangeRow]:
    raw_clean = _clean_ascii(line)
    repaired = _repair_split_nbs_numbers(raw_clean)
    tokens = repaired.split()
    # 条件分岐: `len(tokens) < 4` を満たす経路を評価する。
    if len(tokens) < 4:
        return None

    t_c = _parse_temp_c(tokens[0])
    rho = _parse_nbs_float(tokens[1])
    rho_lo = _parse_nbs_float(tokens[2])

    rho_hi: Optional[float] = None
    # Look for the "TO" separator token in various degraded forms.
    for i in range(3, len(tokens) - 1):
        sep = tokens[i].upper()
        # 条件分岐: `sep in {"TO", "T0", "RO", "R0"}` を満たす経路を評価する。
        if sep in {"TO", "T0", "RO", "R0"}:
            rho_hi = _parse_nbs_float(tokens[i + 1])
            break

        # 条件分岐: `sep == "T" and tokens[i + 1] in {"0", "O", "o"} and i + 2 < len(tokens)` を満たす経路を評価する。

        if sep == "T" and tokens[i + 1] in {"0", "O", "o"} and i + 2 < len(tokens):
            rho_hi = _parse_nbs_float(tokens[i + 2])
            break

    # 条件分岐: `rho_hi is None` を満たす経路を評価する。

    if rho_hi is None:
        rho_hi = _parse_nbs_float(tokens[3])

    # 条件分岐: `t_c is None or rho is None` を満たす経路を評価する。

    if t_c is None or rho is None:
        return None

    return RangeRow(t_c=float(t_c), rho=float(rho), rho_lo=rho_lo, rho_hi=rho_hi, raw=raw_clean)


# 関数: `_extract_samples_from_stream` の入出力契約と処理意図を定義する。

def _extract_samples_from_stream(lines: list[str], *, source_pages: list[int]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    reading = False
    rows: list[RangeRow] = []

    # 関数: `flush` の入出力契約と処理意図を定義する。
    def flush() -> None:
        nonlocal cur, reading, rows
        # 条件分岐: `cur is None` を満たす経路を評価する。
        if cur is None:
            rows = []
            reading = False
            return

        # 条件分岐: `rows` を満たす経路を評価する。

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

        # 条件分岐: `cur.get("rho_range_table")` を満たす経路を評価する。

        if cur.get("rho_range_table"):
            samples.append(cur)

        cur = None
        rows = []
        reading = False

    for raw in lines:
        line = _clean_ascii(raw)
        # 条件分岐: `not line` を満たす経路を評価する。
        if not line:
            continue

        # 条件分岐: `_is_sample_header(line)` を満たす経路を評価する。

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

        # 条件分岐: `cur is None` を満たす経路を評価する。

        if cur is None:
            continue

        # 条件分岐: `_RANGE_HEADER_RE.search(line) and "TO" not in line.upper()` を満たす経路を評価する。

        if _RANGE_HEADER_RE.search(line) and "TO" not in line.upper():
            reading = True
            rows = []
            continue

        # 条件分岐: `reading` を満たす経路を評価する。

        if reading:
            # Stop if another header-like section begins.
            if line.lower().startswith("ranges given"):
                reading = False
                continue

            rr = _parse_range_row(line)
            # 条件分岐: `rr is None` を満たす経路を評価する。
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


# 関数: `_extract_samples` の入出力契約と処理意図を定義する。

def _extract_samples(pdf_path: Path) -> dict[str, Any]:
    reader = PdfReader(str(pdf_path))
    start = _find_appendix_e_start(reader)
    # 条件分岐: `start is None` を満たす経路を評価する。
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
            # 条件分岐: `left.strip()` を満たす経路を評価する。
            if left.strip():
                left_lines.append(left)

            # 条件分岐: `right.strip()` を満たす経路を評価する。

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


# 関数: `main` の入出力契約と処理意図を定義する。

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

    # 条件分岐: `not args.offline` を満たす経路を評価する。
    if not args.offline:
        _download(url_pdf, out_pdf)

    # 条件分岐: `not out_pdf.exists() or out_pdf.stat().st_size == 0` を満たす経路を評価する。

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


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
