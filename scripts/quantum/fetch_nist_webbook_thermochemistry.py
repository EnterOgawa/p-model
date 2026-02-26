from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
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
            # 条件分岐: `not b` を満たす経路を評価する。
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
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0` を満たす経路を評価する。
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
        # 条件分岐: `t == "tr"` を満たす経路を評価する。
        if t == "tr":
            self._in_tr = True
            self._cur_row = []
            return

        # 条件分岐: `t in ("td", "th") and self._in_tr` を満たす経路を評価する。

        if t in ("td", "th") and self._in_tr:
            self._in_cell = True
            self._cur_cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        # 条件分岐: `t in ("td", "th") and self._in_tr and self._in_cell` を満たす経路を評価する。
        if t in ("td", "th") and self._in_tr and self._in_cell:
            txt = "".join(self._cur_cell_parts)
            txt = txt.replace("\u00a0", " ")  # &nbsp;
            txt = re.sub(r"\s+", " ", txt).strip()
            self._cur_row.append(txt)
            self._cur_cell_parts = []
            self._in_cell = False
            return

        # 条件分岐: `t == "tr" and self._in_tr` を満たす経路を評価する。

        if t == "tr" and self._in_tr:
            # 条件分岐: `self._cur_row` を満たす経路を評価する。
            if self._cur_row:
                self.rows.append(self._cur_row)

            self._cur_row = []
            self._in_tr = False

    def handle_data(self, data: str) -> None:
        # 条件分岐: `self._in_tr and self._in_cell` を満たす経路を評価する。
        if self._in_tr and self._in_cell:
            self._cur_cell_parts.append(data)


def _as_float(s: str) -> Optional[float]:
    ss = str(s).strip()
    # 条件分岐: `not ss` を満たす経路を評価する。
    if not ss:
        return None

    m = re.search(r"[-+]?\d+(?:\.\d+)?", ss)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return None

    try:
        return float(m.group(0))
    except Exception:
        return None


def _parse_value_plusminus(s: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse strings like:
      '217.998 ± 0.006' -> (217.998, 0.006)
      '0.32' -> (0.32, None)
    """
    ss = str(s).strip()
    # 条件分岐: `not ss` を満たす経路を評価する。
    if not ss:
        return None, None

    # 条件分岐: `"±" in ss` を満たす経路を評価する。

    if "±" in ss:
        left, right = ss.split("±", 1)
        return _as_float(left), _as_float(right)

    return _as_float(ss), None


@dataclass(frozen=True)
class ThermoRecord:
    kind: str
    value_kj_per_mol: float
    unc_kj_per_mol: Optional[float]
    evidence: str


def _extract_thermo_gas_section(html: str) -> str:
    m = re.search(r'<h2 id="Thermo-Gas">', html, re.IGNORECASE)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError('Thermo-Gas section not found (expected <h2 id="Thermo-Gas">)')

    start = m.start()
    m2 = re.search(r"<h2 id=", html[start + 10 :], re.IGNORECASE)
    end = (start + 10 + m2.start()) if m2 else len(html)
    return html[start:end]


def _extract_tables(section_html: str) -> list[str]:
    return re.findall(r'<table class="data"[^>]*>.*?</table>', section_html, flags=re.IGNORECASE | re.DOTALL)


def _select_dhf_kj_per_mol(tables_rows: list[list[list[str]]]) -> tuple[Optional[ThermoRecord], list[dict[str, Any]]]:
    """
    Select a single best ΔfH°(gas) (kJ/mol) value.
    Strategy:
      1) Prefer explicit ΔfH°gas rows from 'Quantity/Value/Units' tables (CODATA/Review if available).
      2) Fallback: Shomate table row 'H' (kJ/mol) when present.
    Returns: (selected, candidates_debug)
    """
    candidates: list[tuple[int, ThermoRecord]] = []
    debug: list[dict[str, Any]] = []

    for ti, rows in enumerate(tables_rows):
        # 条件分岐: `not rows` を満たす経路を評価する。
        if not rows:
            continue

        header = rows[0]
        header0 = (header[0] if header else "").strip().lower()

        # Case A: Quantity/Value/Units table
        if len(header) >= 3 and header0 == "quantity":
            for ri, r in enumerate(rows[1:], start=1):
                # 条件分岐: `len(r) < 3` を満たす経路を評価する。
                if len(r) < 3:
                    continue

                q = r[0].strip()
                v = r[1].strip()
                u = r[2].strip()
                method = r[3].strip() if len(r) > 3 else ""
                comment = r[5].strip() if len(r) > 5 else ""

                # 条件分岐: `not q.startswith("ΔfH")` を満たす経路を評価する。
                if not q.startswith("ΔfH"):
                    continue

                # 条件分岐: `"kj/mol" not in u.lower()` を満たす経路を評価する。

                if "kj/mol" not in u.lower():
                    continue

                val, unc = _parse_value_plusminus(v)
                # 条件分岐: `val is None` を満たす経路を評価する。
                if val is None:
                    continue

                score = 10_000
                # 条件分岐: `"codata" in comment.lower()` を満たす経路を評価する。
                if "codata" in comment.lower():
                    score += 200

                # 条件分岐: `"review" in method.lower()` を満たす経路を評価する。

                if "review" in method.lower():
                    score += 100
                # Prefer rows with uncertainty reported.

                if unc is not None:
                    score += 10

                score -= ri  # stable tie-break

                rec = ThermoRecord(
                    kind="delta_f_h_gas",
                    value_kj_per_mol=float(val),
                    unc_kj_per_mol=(None if unc is None else float(unc)),
                    evidence=f"table={ti}, row={ri}, q={q}, units={u}",
                )
                candidates.append((score, rec))
                debug.append(
                    {
                        "table_index": ti,
                        "row_index": ri,
                        "quantity": q,
                        "value_raw": v,
                        "units": u,
                        "method": method,
                        "comment": comment,
                        "parsed_value_kj_per_mol": val,
                        "parsed_unc_kj_per_mol": unc,
                        "score": score,
                    }
                )

        # Case B: Shomate coefficients table

        if header0.startswith("temperature"):
            for ri, r in enumerate(rows[1:], start=1):
                # 条件分岐: `len(r) < 2` を満たす経路を評価する。
                if len(r) < 2:
                    continue

                # 条件分岐: `str(r[0]).strip() != "H"` を満たす経路を評価する。

                if str(r[0]).strip() != "H":
                    continue
                # Shomate H is ΔfH° (kJ/mol) at 298.15 K, constant across ranges.

                v = str(r[1]).strip()
                val = _as_float(v)
                # 条件分岐: `val is None` を満たす経路を評価する。
                if val is None:
                    continue

                score = 1_000 - ri
                rec = ThermoRecord(
                    kind="shomate_H",
                    value_kj_per_mol=float(val),
                    unc_kj_per_mol=None,
                    evidence=f"table={ti}, row={ri}, key=H (Shomate)",
                )
                candidates.append((score, rec))
                debug.append(
                    {
                        "table_index": ti,
                        "row_index": ri,
                        "quantity": "H (Shomate)",
                        "value_raw": v,
                        "units": "kJ/mol (Shomate)",
                        "parsed_value_kj_per_mol": val,
                        "parsed_unc_kj_per_mol": None,
                        "score": score,
                    }
                )

    # 条件分岐: `not candidates` を満たす経路を評価する。

    if not candidates:
        return None, debug

    best = max(candidates, key=lambda x: x[0])[1]
    return best, debug


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch NIST Chemistry WebBook thermochemistry section and cache ΔfH° values.")
    ap.add_argument("--id", required=True, help="NIST Chemistry WebBook ID (e.g., C12385136 for H atom).")
    ap.add_argument("--slug", required=True, help="Cache folder token (saved under data/quantum/sources/nist_webbook_thermo_<slug>/).")
    ap.add_argument("--offline", action="store_true", help="Offline mode: do not download; only validate local cache files exist.")
    args = ap.parse_args(argv)

    root = _repo_root()
    slug = _sanitize_token(args.slug)
    webbook_id = str(args.id).strip()

    out_dir = root / "data" / "quantum" / "sources" / f"nist_webbook_thermo_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID={webbook_id}&Units=SI&Mask=1"
    raw_path = out_dir / f"nist_webbook_thermo__{slug}__{_sanitize_token(webbook_id)}.html"
    extracted_path = out_dir / "extracted_values.json"
    manifest_path = out_dir / "manifest.json"

    # 条件分岐: `args.offline` を満たす経路を評価する。
    if args.offline:
        missing: list[Path] = []
        for p in (raw_path, extracted_path, manifest_path):
            # 条件分岐: `not p.exists() or p.stat().st_size <= 0` を満たす経路を評価する。
            if not p.exists() or p.stat().st_size <= 0:
                missing.append(p)

        # 条件分岐: `missing` を満たす経路を評価する。

        if missing:
            raise SystemExit("[fail] missing cache files:\n" + "\n".join(f"- {p}" for p in missing))

        print("[ok] offline check passed")
        return 0

    _download(url, raw_path)
    html = raw_path.read_text(encoding="utf-8", errors="replace")

    section = _extract_thermo_gas_section(html)
    tables = _extract_tables(section)

    tables_rows: list[list[list[str]]] = []
    for t in tables:
        p = _HTMLTableParser()
        p.feed(t)
        # 条件分岐: `p.rows` を満たす経路を評価する。
        if p.rows:
            tables_rows.append(p.rows)

    selected, debug = _select_dhf_kj_per_mol(tables_rows)

    extracted: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "NIST Chemistry WebBook: Gas phase thermochemistry data",
        "webbook_id": webbook_id,
        "query_url": url,
        "raw_file": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "selected": (
            None
            if selected is None
            else {
                "kind": selected.kind,
                "dhf_kj_per_mol": selected.value_kj_per_mol,
                "dhf_unc_kj_per_mol": selected.unc_kj_per_mol,
                "evidence": selected.evidence,
            }
        ),
        "notes": [
            "This cache is used to fix an offline-stable thermochemistry baseline for Phase 7 / Step 7.12 (molecular binding energy).",
            "Selection priority: explicit ΔfH°gas row (Quantity/Value table) > Shomate table H coefficient fallback.",
            "If ΔfH° is not present for an elemental standard state, it may appear only as Shomate H=0.0 or be absent; handle with care downstream.",
        ],
        "tables_parsed": len(tables_rows),
        "selection_debug": debug,
    }
    extracted_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] extracted: {extracted_path}")

    manifest = {
        "generated_utc": extracted["generated_utc"],
        "dataset": f"NIST WebBook thermochemistry: {webbook_id}",
        "notes": [
            "Cache for offline-repro analysis in Phase 7 / Step 7.12 (thermochemistry baseline).",
            "If you change selection rules, update the manuscript and regenerate paper outputs.",
        ],
        "files": [
            {
                "url": url,
                "path": str(raw_path),
                "bytes": int(raw_path.stat().st_size),
                "sha256": _sha256(raw_path),
            },
            {
                "url": None,
                "path": str(extracted_path),
                "bytes": int(extracted_path.stat().st_size),
                "sha256": _sha256(extracted_path),
                "derived_from": [raw_path.name],
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest: {manifest_path}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

