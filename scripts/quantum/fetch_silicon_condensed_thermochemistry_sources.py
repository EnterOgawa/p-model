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


def _extract_section(html: str, *, section_id: str) -> str:
    m = re.search(rf'<h2 id="{re.escape(section_id)}">', html, flags=re.IGNORECASE)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"section not found: {section_id}")

    start = m.start()
    m2 = re.search(r"<h2 id=", html[start + 10 :], flags=re.IGNORECASE)
    end = (start + 10 + m2.start()) if m2 else len(html)
    return html[start:end]


def _extract_tables(section_html: str) -> list[str]:
    return re.findall(r'<table class="data"[^>]*>.*?</table>', section_html, flags=re.IGNORECASE | re.DOTALL)


def _parse_table_rows(table_html: str) -> list[list[str]]:
    p = _HTMLTableParser()
    p.feed(table_html)
    return p.rows


def _as_float(s: str) -> Optional[float]:
    ss = str(s).strip()
    # 条件分岐: `not ss` を満たす経路を評価する。
    if not ss:
        return None
    # Shomate coefficients sometimes appear as: -1.198306×10 -10

    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*×\s*10\s*([-+]?\d+)", ss)
    # 条件分岐: `m` を満たす経路を評価する。
    if m:
        try:
            return float(m.group(1)) * (10.0 ** int(m.group(2)))
        except Exception:
            return None
    # plain float

    m2 = re.search(r"[-+]?\d+(?:\.\d+)?", ss)
    # 条件分岐: `not m2` を満たす経路を評価する。
    if not m2:
        return None

    try:
        return float(m2.group(0))
    except Exception:
        return None


@dataclass(frozen=True)
class ShomateBlock:
    phase: str
    t_min_k: float
    t_max_k: float
    coeffs: dict[str, float]
    reference: str | None
    comment: str | None


def _parse_shomate_table(*, rows: list[list[str]], phase: str) -> ShomateBlock:
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        raise ValueError("empty table")

    kv: dict[str, str] = {}
    for r in rows:
        # 条件分岐: `len(r) < 2` を満たす経路を評価する。
        if len(r) < 2:
            continue

        k = str(r[0]).strip()
        v = str(r[1]).strip()
        # 条件分岐: `not k` を満たす経路を評価する。
        if not k:
            continue

        kv[k] = v

    tr = kv.get("Temperature (K)")
    # 条件分岐: `not tr` を満たす経路を評価する。
    if not tr:
        raise ValueError("missing Temperature (K) row")

    m = re.search(r"([-+]?\d+(?:\.\d*)?)\s*to\s*([-+]?\d+(?:\.\d*)?)", tr)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"cannot parse temperature range: {tr!r}")

    t_min = float(m.group(1))
    t_max = float(m.group(2))

    coeffs: dict[str, float] = {}
    for key in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        # 条件分岐: `key not in kv` を満たす経路を評価する。
        if key not in kv:
            raise ValueError(f"missing coefficient: {key}")

        val = _as_float(kv[key])
        # 条件分岐: `val is None` を満たす経路を評価する。
        if val is None:
            raise ValueError(f"cannot parse coefficient {key}: {kv[key]!r}")

        coeffs[key] = float(val)

    return ShomateBlock(
        phase=phase,
        t_min_k=t_min,
        t_max_k=t_max,
        coeffs=coeffs,
        reference=(kv.get("Reference") or None),
        comment=(kv.get("Comment") or None),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch NIST Chemistry WebBook condensed phase thermochemistry for silicon (Si) and extract "
            "Shomate heat-capacity coefficients (solid/liquid) for Step 7.14."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Offline mode: do not download; only validate cache files.")
    args = ap.parse_args(argv)

    root = _repo_root()
    out_dir = root / "data" / "quantum" / "sources" / "nist_webbook_condensed_silicon_si"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Silicon entry in NIST WebBook.
    webbook_id = "C7440213"
    url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID={webbook_id}&Units=SI&Mask=2#Thermo-Condensed"
    out_html = out_dir / "nist_webbook_condensed_si_mask2.html"

    # 条件分岐: `not args.offline` を満たす経路を評価する。
    if not args.offline:
        _download(url, out_html)

    # 条件分岐: `not out_html.exists() or out_html.stat().st_size == 0` を満たす経路を評価する。

    if not out_html.exists() or out_html.stat().st_size == 0:
        raise SystemExit(f"[fail] missing html: {out_html}")

    html = out_html.read_text(encoding="utf-8", errors="replace")
    section = _extract_section(html, section_id="Thermo-Condensed")
    tables = _extract_tables(section)

    # Identify Shomate tables by nearby <h3> headers.
    shomate_blocks: list[ShomateBlock] = []
    for ti, table_html in enumerate(tables):
        idx = section.find(table_html)
        h3s = list(re.finditer(r"<h3>(.*?)</h3>", section[:idx], flags=re.IGNORECASE | re.DOTALL))
        title = h3s[-1].group(1) if h3s else ""
        title = re.sub(r"<[^>]+>", " ", title)
        title = re.sub(r"\s+", " ", title).strip()

        # 条件分岐: `"Heat Capacity" not in title or "Shomate" not in title` を満たす経路を評価する。
        if "Heat Capacity" not in title or "Shomate" not in title:
            continue

        rows = _parse_table_rows(table_html)
        phase = "solid" if title.lower().startswith("solid") else ("liquid" if title.lower().startswith("liquid") else title)
        shomate_blocks.append(_parse_shomate_table(rows=rows, phase=phase))

    # 条件分岐: `not shomate_blocks` を満たす経路を評価する。

    if not shomate_blocks:
        raise SystemExit("[fail] no Shomate blocks found in Thermo-Condensed section")

    extracted = {
        "generated_utc": _iso_utc_now(),
        "dataset": "NIST Chemistry WebBook: condensed phase thermochemistry (Si; Shomate Cp coefficients)",
        "species": {"name": "silicon", "formula": "Si", "webbook_id": webbook_id},
        "source_url": url,
        "inputs": {"html_path": str(out_html), "html_sha256": _sha256(out_html)},
        "shomate": [
            {
                "phase": b.phase,
                "t_min_k": b.t_min_k,
                "t_max_k": b.t_max_k,
                "coeffs": b.coeffs,
                "reference": b.reference,
                "comment": b.comment,
            }
            for b in shomate_blocks
        ],
        "notes": [
            "Shomate equation uses t=T/1000 and yields Cp° in J/(mol*K).",
            "This cache is used as a primary baseline for Step 7.14 (condensed matter) and intended for offline reproducibility.",
        ],
    }

    out_extracted = out_dir / "extracted_values.json"
    out_extracted.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "generated_utc": _iso_utc_now(),
        "dataset": extracted["dataset"],
        "files": [
            {
                "url": url,
                "path": str(out_html),
                "bytes": int(out_html.stat().st_size),
                "sha256": _sha256(out_html).upper(),
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
