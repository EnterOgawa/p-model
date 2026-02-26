from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


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
    with urlopen(req, timeout=30) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    # 条件分岐: `out_path.stat().st_size == 0` を満たす経路を評価する。

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


# 関数: `_strip_tags` の入出力契約と処理意図を定義する。

def _strip_tags(s: str) -> str:
    # Avoid stripping comparison operators like "< 50K" that appear in plain text.
    return re.sub(r"</?[A-Za-z][^>]*>", " ", s)


# 関数: `_extract_tables` の入出力契約と処理意図を定義する。

def _extract_tables(html: str, *, class_name: str) -> list[str]:
    return re.findall(
        rf"<table[^>]*class=['\"]{re.escape(class_name)}['\"][^>]*>.*?</table>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )


# 関数: `_cell_text` の入出力契約と処理意図を定義する。

def _cell_text(cell_html: str) -> str:
    txt = _strip_tags(unescape(cell_html)).replace("\u00a0", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# 関数: `_parse_thermal_conductivity_table` の入出力契約と処理意図を定義する。

def _parse_thermal_conductivity_table(*, html: str) -> dict[str, Any]:
    """
    Parse the NIST TRC cryogenics OFHC Copper page for the Thermal Conductivity table.

    The page provides k(T) fits for several RRR values via:
      log10 k = (a + c*T^0.5 + e*T + g*T^1.5 + i*T^2) / (1 + b*T^0.5 + d*T + f*T^1.5 + h*T^2)
    """
    tables = _extract_tables(html, class_name="properties")
    # 条件分岐: `not tables` を満たす経路を評価する。
    if not tables:
        raise ValueError("missing table class='properties'")

    tbl = None
    for t in tables:
        # 条件分岐: `"Thermal Conductivity" in t and "RRR" in t` を満たす経路を評価する。
        if "Thermal Conductivity" in t and "RRR" in t:
            tbl = t
            break

    # 条件分岐: `tbl is None` を満たす経路を評価する。

    if tbl is None:
        raise ValueError("missing Thermal Conductivity table")

    rows = re.findall(r"<tr[^>]*>.*?</tr>", tbl, flags=re.IGNORECASE | re.DOTALL)
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        raise ValueError("no <tr> rows found in Thermal Conductivity table")

    rrrs: list[int] = []
    units: str | None = None
    coeffs_by_rrr: dict[int, dict[str, float]] = {}
    ranges_by_rrr: dict[int, dict[str, float]] = {}
    fit_error_by_rrr: dict[int, float] = {}

    # 関数: `_ensure_rrrs` の入出力契約と処理意図を定義する。
    def _ensure_rrrs(ncols: int) -> None:
        nonlocal rrrs
        # 条件分岐: `rrrs` を満たす経路を評価する。
        if rrrs:
            return

        raise ValueError(f"RRR header not parsed (ncols={ncols})")

    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
        # 条件分岐: `not cells` を満たす経路を評価する。
        if not cells:
            continue

        t0 = _cell_text(cells[0])
        # 条件分岐: `not t0` を満たす経路を評価する。
        if not t0:
            continue

        # Header row: contains "RRR = <n>" cells.

        if ("Thermal Conductivity" in t0) or ("Thermal Conductivity" in _cell_text(" ".join(cells[:2]))):
            # Parse RRR values from the full row cells.
            for c in cells[1:]:
                m = re.search(r"RRR\s*=\s*([0-9]+)", _cell_text(c), flags=re.IGNORECASE)
                # 条件分岐: `m` を満たす経路を評価する。
                if m:
                    rrrs.append(int(m.group(1)))

            # 条件分岐: `not rrrs` を満たす経路を評価する。

            if not rrrs:
                raise ValueError("could not parse RRR values from header row")

            for r in rrrs:
                coeffs_by_rrr[r] = {}
                ranges_by_rrr[r] = {}

            continue

        # 条件分岐: `t0.upper() == "UNITS"` を満たす経路を評価する。

        if t0.upper() == "UNITS":
            _ensure_rrrs(len(cells) - 1)
            col_units: list[str] = []
            for c in cells[1 : 1 + len(rrrs)]:
                col_units.append(_cell_text(c))

            col_units = [u for u in col_units if u]
            # 条件分岐: `col_units and len(set(col_units)) == 1` を満たす経路を評価する。
            if col_units and len(set(col_units)) == 1:
                units = col_units[0]
            else:
                # Keep the first non-empty unit, but record the raw list for inspection.
                units = col_units[0] if col_units else ""

            continue

        key = t0.strip().lower()

        # Coefficients a..i
        if re.fullmatch(r"[a-i]", key):
            _ensure_rrrs(len(cells) - 1)
            for r, c in zip(rrrs, cells[1 : 1 + len(rrrs)]):
                s = _cell_text(c)
                # 条件分岐: `not re.fullmatch(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", s)` を満たす経路を評価する。
                if not re.fullmatch(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", s):
                    raise ValueError(f"invalid numeric coefficient {key} for RRR={r}: {s!r}")

                coeffs_by_rrr[int(r)][key] = float(s)

            continue

        # Ranges: "low range" / "high range" rows.

        if "low range" in key:
            _ensure_rrrs(len(cells) - 1)
            for r, c in zip(rrrs, cells[1 : 1 + len(rrrs)]):
                s = _cell_text(c)
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*K", s, flags=re.IGNORECASE)
                # 条件分岐: `not m` を満たす経路を評価する。
                if not m:
                    raise ValueError(f"invalid low-range cell for RRR={r}: {s!r}")

                ranges_by_rrr[int(r)]["t_min_k"] = float(m.group(1))

            continue

        # 条件分岐: `"high" in key and "range" in key` を満たす経路を評価する。

        if "high" in key and "range" in key:
            _ensure_rrrs(len(cells) - 1)
            for r, c in zip(rrrs, cells[1 : 1 + len(rrrs)]):
                s = _cell_text(c)
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*K", s, flags=re.IGNORECASE)
                # 条件分岐: `not m` を満たす経路を評価する。
                if not m:
                    raise ValueError(f"invalid high-range cell for RRR={r}: {s!r}")

                ranges_by_rrr[int(r)]["t_max_k"] = float(m.group(1))

            continue

        # 条件分岐: `"curve fit" in key and "error" in key` を満たす経路を評価する。

        if "curve fit" in key and "error" in key:
            _ensure_rrrs(len(cells) - 1)
            for r, c in zip(rrrs, cells[1 : 1 + len(rrrs)]):
                s = _cell_text(c)
                # 条件分岐: `not re.fullmatch(r"[-+]?\d+(?:\.\d*)?", s)` を満たす経路を評価する。
                if not re.fullmatch(r"[-+]?\d+(?:\.\d*)?", s):
                    raise ValueError(f"invalid fit error cell for RRR={r}: {s!r}")

                fit_error_by_rrr[int(r)] = float(s)

            continue

    # 条件分岐: `not rrrs` を満たす経路を評価する。

    if not rrrs:
        raise ValueError("missing RRR values")

    # 条件分岐: `units is None` を満たす経路を評価する。

    if units is None:
        raise ValueError("missing UNITS row")

    for r in rrrs:
        missing = [k for k in "abcdefghi" if k not in coeffs_by_rrr[r]]
        # 条件分岐: `missing` を満たす経路を評価する。
        if missing:
            raise ValueError(f"missing coefficients for RRR={r}: {missing}")

        # 条件分岐: `"t_min_k" not in ranges_by_rrr[r] or "t_max_k" not in ranges_by_rrr[r]` を満たす経路を評価する。

        if "t_min_k" not in ranges_by_rrr[r] or "t_max_k" not in ranges_by_rrr[r]:
            raise ValueError(f"missing range for RRR={r}: {ranges_by_rrr[r]}")

        # 条件分岐: `r not in fit_error_by_rrr` を満たす経路を評価する。

        if r not in fit_error_by_rrr:
            raise ValueError(f"missing fit error % for RRR={r}")

    out_by_rrr: dict[str, Any] = {}
    for r in rrrs:
        out_by_rrr[str(r)] = {
            "coefficients": coeffs_by_rrr[r],
            "data_range": ranges_by_rrr[r],
            "curve_fit_error_percent_relative_to_data": fit_error_by_rrr[r],
        }

    return {
        "material": "OFHC Copper (UNS C10100/C10200)",
        "property": "thermal_conductivity",
        "units": units,
        "rrr": out_by_rrr,
        "model": {
            "form": "log10(k)=N(T)/D(T); k=10^(N/D)",
            "numerator": "a + c*T^0.5 + e*T + g*T^1.5 + i*T^2",
            "denominator": "1 + b*T^0.5 + d*T + f*T^1.5 + h*T^2",
        },
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch NIST TRC cryogenics 'Material Properties: OFHC Copper' page and extract the thermal "
            "conductivity fit coefficients (RRR=50/100/150/300/500). Writes manifest.json and "
            "extracted_values.json under data/quantum/sources/."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--out-dirname",
        default="nist_trc_ofhc_copper_thermal_conductivity",
        help="Output directory name under data/quantum/sources/.",
    )
    args = ap.parse_args()

    root = _repo_root()
    src_dir = root / "data" / "quantum" / "sources" / str(args.out_dirname)
    src_dir.mkdir(parents=True, exist_ok=True)

    url = "https://trc.nist.gov/cryogenics/materials/OFHC%20Copper/OFHC_Copper_rev1.htm"
    html_path = src_dir / "OFHC_Copper_rev1.htm"

    # 条件分岐: `not args.offline` を満たす経路を評価する。
    if not args.offline:
        _download(url, html_path)

    # 条件分岐: `not html_path.exists() or html_path.stat().st_size == 0` を満たす経路を評価する。

    if not html_path.exists() or html_path.stat().st_size == 0:
        raise SystemExit(f"[fail] missing file: {html_path}")

    html = html_path.read_text(encoding="utf-8", errors="replace")
    extracted = _parse_thermal_conductivity_table(html=html)

    out_extracted = src_dir / "extracted_values.json"
    out_extracted.write_text(
        json.dumps(
            {
                "generated_utc": _iso_utc_now(),
                "dataset": "Phase 7 / Step 7.14.6 OFHC copper thermal conductivity (NIST TRC cryogenics)",
                "source": {"url": url, "local_path": str(html_path), "local_sha256": _sha256(html_path)},
                **extracted,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "Phase 7 / Step 7.14.6 OFHC copper thermal conductivity (NIST TRC cryogenics)",
        "notes": [
            "NIST TRC Cryogenics 'Material Properties: OFHC Copper' provides fitted equations for k(T) with coefficients a–i for several RRR values.",
            "This project caches the HTML for offline reproducibility.",
            "extracted_values.json is derived from the cached HTML and is intended for offline analysis.",
        ],
        "files": [
            {"name": html_path.name, "url": url, "path": str(html_path), "bytes": int(html_path.stat().st_size), "sha256": _sha256(html_path).upper()}
        ],
    }
    out_manifest = src_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_extracted}")
    print(f"[ok] wrote: {out_manifest}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

