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


def _strip_tags(s: str) -> str:
    # Avoid stripping comparison operators like "< 50K" that appear in plain text.
    return re.sub(r"</?[A-Za-z][^>]*>", " ", s)


def _extract_first_table(html: str, *, class_name: str) -> str:
    m = re.search(
        rf"<table[^>]*class=['\"]{re.escape(class_name)}['\"][^>]*>.*?</table>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"missing table class={class_name!r}")

    return m.group(0)


def _parse_coeff_table(*, html: str) -> dict[str, Any]:
    tbl = _extract_first_table(html, class_name="properties")
    rows = re.findall(r"<tr[^>]*>.*?</tr>", tbl, flags=re.IGNORECASE | re.DOTALL)

    coeffs: dict[str, float] = {}
    units_raw: str | None = None
    data_range: dict[str, float] | None = None
    fit_error_raw: str | None = None
    ref_1: str | None = None

    for row in rows:
        txt = _strip_tags(unescape(row)).replace("\u00a0", " ")
        txt = re.sub(r"\s+", " ", txt).strip()
        # 条件分岐: `not txt` を満たす経路を評価する。
        if not txt:
            continue

        parts = txt.split()
        # 条件分岐: `len(parts) >= 2 and re.fullmatch(r"[a-lA-L]", parts[0])` を満たす経路を評価する。
        if len(parts) >= 2 and re.fullmatch(r"[a-lA-L]", parts[0]):
            key = parts[0].lower()
            val_s = parts[1]
            # 条件分岐: `re.fullmatch(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", val_s)` を満たす経路を評価する。
            if re.fullmatch(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", val_s):
                coeffs[key] = float(val_s)

            continue

        # 条件分岐: `"UNITS" in txt.upper() and units_raw is None` を満たす経路を評価する。

        if "UNITS" in txt.upper() and units_raw is None:
            units_raw = txt

        # 条件分岐: `"data range" in txt.lower() and data_range is None` を満たす経路を評価する。

        if "data range" in txt.lower() and data_range is None:
            m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*K", txt, flags=re.IGNORECASE)
            # 条件分岐: `m` を満たす経路を評価する。
            if m:
                data_range = {"t_min_k": float(m.group(1)), "t_max_k": float(m.group(2))}

        # 条件分岐: `"curve fit" in txt.lower() and "error" in txt.lower()` を満たす経路を評価する。

        if "curve fit" in txt.lower() and "error" in txt.lower():
            fit_error_raw = txt

    mref = re.search(r"\[1\]\s*(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
    # 条件分岐: `mref` を満たす経路を評価する。
    if mref:
        ref_1 = re.sub(r"\s+", " ", _strip_tags(unescape(mref.group(1)))).strip()

    missing = [k for k in "abcdefghijkl" if k not in coeffs]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        raise ValueError(f"missing coefficients: {missing}")

    # 条件分岐: `data_range is None` を満たす経路を評価する。

    if data_range is None:
        raise ValueError("missing data range row")

    fit_obj: dict[str, Any] = {"raw": fit_error_raw or ""}
    # 条件分岐: `fit_error_raw` を満たす経路を評価する。
    if fit_error_raw:
        # Example (after tag stripping): "0.03x10-8 (T < 50K); 0.5x10-8 (T ≥ 50K)"
        m_lt = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*x\s*10\s*-8\s*\(\s*T\s*<\s*([0-9]+)\s*K", fit_error_raw)
        m_ge = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*x\s*10\s*-8\s*\(\s*T\s*(?:>=|≥)\s*([0-9]+)\s*K", fit_error_raw)
        # 条件分岐: `m_lt` を満たす経路を評価する。
        if m_lt:
            fit_obj["lt"] = {"t_k": float(m_lt.group(2)), "sigma_1e_8_per_k": float(m_lt.group(1))}

        # 条件分岐: `m_ge` を満たす経路を評価する。

        if m_ge:
            fit_obj["ge"] = {"t_k": float(m_ge.group(2)), "sigma_1e_8_per_k": float(m_ge.group(1))}

    return {
        "coefficients": coeffs,
        "units_raw": units_raw or "",
        "data_range": data_range,
        "fit_error_relative_to_data": fit_obj,
        "reference_1": ref_1 or "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch NIST TRC cryogenics 'Material Properties: Silicon' page (thermal expansion coefficient) "
            "for Phase 7 / Step 7.14.5. Caches HTML and equation/plot images, and extracts coefficients a–l."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--out-dirname",
        default="nist_trc_silicon_thermal_expansion",
        help="Output directory name under data/quantum/sources/.",
    )
    args = ap.parse_args()

    root = _repo_root()
    src_dir = root / "data" / "quantum" / "sources" / str(args.out_dirname)
    src_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://trc.nist.gov/cryogenics/materials/Silicon/"
    targets = [
        {"name": "Silicon.htm", "url": base_url + "Silicon.htm", "path": src_dir / "Silicon.htm"},
        {"name": "te.png", "url": base_url + "te.png", "path": src_dir / "te.png"},
        {"name": "Siliconplot.png", "url": base_url + "Siliconplot.png", "path": src_dir / "Siliconplot.png"},
        {"name": "temp.gif", "url": base_url + "temp.gif", "path": src_dir / "temp.gif"},
    ]

    # 条件分岐: `not args.offline` を満たす経路を評価する。
    if not args.offline:
        for t in targets:
            _download(str(t["url"]), Path(t["path"]))

    missing: list[Path] = []
    for t in targets:
        p = Path(t["path"])
        # 条件分岐: `not p.exists() or p.stat().st_size == 0` を満たす経路を評価する。
        if not p.exists() or p.stat().st_size == 0:
            missing.append(p)

    # 条件分岐: `missing` を満たす経路を評価する。

    if missing:
        raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

    html_path = src_dir / "Silicon.htm"
    html = html_path.read_text(encoding="utf-8", errors="replace")
    extracted = _parse_coeff_table(html=html)

    out_extracted = src_dir / "extracted_values.json"
    out_extracted.write_text(
        json.dumps(
            {
                "generated_utc": _iso_utc_now(),
                "dataset": "Phase 7 / Step 7.14.5 silicon thermal expansion coefficient (NIST TRC cryogenics)",
                "source": {
                    "url": base_url + "Silicon.htm",
                    "local_path": str(html_path),
                    "local_sha256": _sha256(html_path),
                },
                **extracted,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "Phase 7 / Step 7.14.5 silicon thermal expansion coefficient (NIST TRC cryogenics)",
        "notes": [
            "NIST TRC Cryogenics 'Material Properties: Silicon' provides a fitted equation and coefficients a–l.",
            "This project caches the HTML and images for offline reproducibility.",
            "extracted_values.json is derived from the cached HTML and is intended for offline analysis.",
        ],
        "files": [],
    }
    for t in targets:
        p = Path(t["path"])
        manifest["files"].append(
            {
                "name": str(t["name"]),
                "url": str(t["url"]),
                "path": str(p),
                "bytes": int(p.stat().st_size),
                "sha256": _sha256(p).upper(),
            }
        )

    out_manifest = src_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_extracted}")
    print(f"[ok] wrote: {out_manifest}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
