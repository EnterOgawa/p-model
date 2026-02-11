from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
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


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return
    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req, timeout=30) as resp, out_path.open("wb") as f:
        f.write(resp.read())
    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")
    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _as_float(s: str) -> Optional[float]:
    ss = str(s).strip()
    if not ss or ss.upper() == "N/A":
        return None
    try:
        return float(ss)
    except Exception:
        return None


def _parse_janaf_txt(txt: str) -> dict[str, Any]:
    lines = [ln.rstrip("\n") for ln in txt.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty JANAF txt")

    title = lines[0].strip()

    # The downloadable format is a single tab-delimited table. It does not label phases per row,
    # but contains a transition marker line "CRYSTAL <--> LIQUID".
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("T(") or ln.startswith("T(K)"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("header line not found (expected T(K) ...)")

    header = lines[header_idx].strip()
    data_lines = lines[header_idx + 1 :]

    t_melt_k: float | None = None
    for ln in data_lines:
        if "CRYSTAL" in ln.upper() and "LIQUID" in ln.upper():
            f0 = ln.split("\t", 1)[0]
            t_melt_k = _as_float(f0)
            break

    points: list[dict[str, Any]] = []
    for raw in data_lines:
        fields = [f.strip() for f in raw.split("\t")]
        if not fields:
            continue

        t_k = _as_float(fields[0])
        if t_k is None:
            continue

        cp = _as_float(fields[1]) if len(fields) > 1 else None
        s = _as_float(fields[2]) if len(fields) > 2 else None
        g = _as_float(fields[3]) if len(fields) > 3 else None
        hh = _as_float(fields[4]) if len(fields) > 4 else None
        marker = "\t".join(fields[5:]).strip() if len(fields) > 5 else ""
        marker = marker or None

        if cp is None:
            continue

        phase = "unknown"
        phase_code = "?"
        if marker and "TRANSITION" in marker.upper():
            phase = "transition"
            phase_code = "trans"
        elif t_melt_k is not None:
            if marker and ("CRYSTAL" in marker.upper() and "LIQUID" in marker.upper()):
                phase = "solid"
                phase_code = "cr"
            elif float(t_k) > float(t_melt_k) + 1e-9:
                phase = "liquid"
                phase_code = "l"
            else:
                phase = "solid"
                phase_code = "cr"

        points.append(
            {
                "phase_code": phase_code,
                "phase": phase,
                "T_K": float(t_k),
                "Cp_J_per_molK": float(cp),
                "S_J_per_molK": None if s is None else float(s),
                "minus_G_minus_H_over_T_J_per_molK": None if g is None else float(g),
                "H_minus_H_Tr_kJ_per_mol": None if hh is None else float(hh),
                "marker": marker,
            }
        )

    if not points:
        raise ValueError("no data points parsed from JANAF txt")

    points_sorted = sorted(points, key=lambda r: (str(r["phase"]), float(r["T_K"])))
    phases = sorted({str(p["phase"]) for p in points_sorted})

    return {
        "title": title,
        "header": header,
        "t_melt_k": t_melt_k,
        "phases": phases,
        "points": points_sorted,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch NIST-JANAF thermochemical table for Silicon (Si) and extract Cp(T) for Step 7.14 "
            "(low-temperature heat-capacity baseline / Debye-fit target)."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Offline mode: do not download; only validate cache files.")
    args = ap.parse_args(argv)

    root = _repo_root()
    out_dir = root / "data" / "quantum" / "sources" / "nist_janaf_silicon_si"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = "https://janaf.nist.gov/tables/Si-004"
    url_html = f"{base}.html"
    url_txt = f"{base}.txt"

    out_html = out_dir / "nist_janaf_Si-004.html"
    out_txt = out_dir / "nist_janaf_Si-004.txt"

    if not args.offline:
        _download(url_html, out_html)
        _download(url_txt, out_txt)

    if not out_txt.exists() or out_txt.stat().st_size == 0:
        raise SystemExit(f"[fail] missing txt: {out_txt}")

    parsed = _parse_janaf_txt(out_txt.read_text(encoding="utf-8", errors="replace"))

    extracted = {
        "generated_utc": _iso_utc_now(),
        "dataset": "NIST-JANAF thermochemical table: Silicon (Si-004)",
        "species": {"name": "Silicon", "formula": "Si", "janaf_table": "Si-004"},
        "source_urls": {"html": url_html, "txt": url_txt},
        "inputs": {
            "html_path": str(out_html) if out_html.exists() else None,
            "html_sha256": _sha256(out_html) if out_html.exists() else None,
            "txt_path": str(out_txt),
            "txt_sha256": _sha256(out_txt),
        },
        "title": parsed.get("title"),
        "table_header": parsed.get("header"),
        "t_melt_k": parsed.get("t_melt_k"),
        "phases": parsed["phases"],
        "points": parsed["points"],
        "notes": [
            "Cp values are taken from the NIST-JANAF tab-delimited table (Si-004).",
            "The table provides Cp° at discrete temperatures and does not include explicit uncertainties.",
            "Phase labels in this cache are assigned using the JANAF transition marker line (CRYSTAL <--> LIQUID).",
            "This cache is used as a primary baseline for Step 7.14 (condensed matter) and intended for offline reproducibility.",
        ],
    }

    out_extracted = out_dir / "extracted_values.json"
    out_extracted.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    files = []
    if out_html.exists():
        files.append(
            {
                "url": url_html,
                "path": str(out_html),
                "bytes": int(out_html.stat().st_size),
                "sha256": _sha256(out_html).upper(),
            }
        )
    files.append(
        {
            "url": url_txt,
            "path": str(out_txt),
            "bytes": int(out_txt.stat().st_size),
            "sha256": _sha256(out_txt).upper(),
        }
    )

    manifest = {
        "generated_utc": _iso_utc_now(),
        "dataset": extracted["dataset"],
        "files": files,
    }

    out_manifest = out_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_extracted}")
    print(f"[ok] wrote: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
