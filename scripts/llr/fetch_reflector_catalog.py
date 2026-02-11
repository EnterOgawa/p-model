#!/usr/bin/env python3
"""
fetch_reflector_catalog.py

LLR（EDC実測）のモデル検証で使用する月面反射器（Apollo 11/14/15, Lunokhod 1/2）の
Moon DE421 Principal-Axis（= SPICE: MOON_PA_DE421）座標を、一次ソースから抽出して
data/llr/reflectors_de421_pa.json を生成する。

一次ソース:
  - Murphy et al., "Laser Ranging to the Lost Lunokhod 1 Reflector"
    arXiv:1009.5720（e-print の TeX から抽出）
    Icarus DOI: 10.1016/j.icarus.2010.11.010
  - ILRS mission page: NGLR-1 (NGL-1)
    https://ilrs.gsfc.nasa.gov/missions/satellite_missions/current_missions/ngl1_general.html
    （Principal Axis-PA の座標を使用）

抽出対象:
  - Table 4 (tab:all-5): 反射器座標（DE421 principal-axis）。静的潮汐成分は既に反映済み。
  - Table 5 (tab:tides): 静的潮汐オフセット（参考。未変形Moonへ戻すときは差し引く）。

オフライン再現:
  - 取得した arXiv e-print は data/llr/sources/ にキャッシュする。
  - 2回目以降は --offline でネットワーク無しでも生成できる（キャッシュがある場合）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


ARXIV_ID = "1009.5720"
ARXIV_EPRINT_URL = f"https://arxiv.org/e-print/{ARXIV_ID}"
ARXIV_ABS_URL = f"https://arxiv.org/abs/{ARXIV_ID}"
JOURNAL_DOI = "10.1016/j.icarus.2010.11.010"

# ILRS: NGLR-1 (NGL-1) reflector coordinates (Principal Axis-PA)
ILRS_NGLR1_URL = "https://ilrs.gsfc.nasa.gov/missions/satellite_missions/current_missions/ngl1_general.html"


@dataclass(frozen=True)
class ParsedRow:
    name: str
    r_m: float
    phi_deg: float
    lam_deg: float
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class TideRow:
    name: str
    dr_m: float
    r_dphi_m: float
    r_dlambda_cosphi_m: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[dl] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-fetch-reflectors/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")


def _extract_tar(tar_path: Path, out_dir: Path, *, force: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"[skip] extracted: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for p in out_dir.glob("*"):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)
    print(f"[ok] extracted: {out_dir}")


def _find_tex_file(extract_dir: Path) -> Optional[Path]:
    candidates = sorted(extract_dir.glob("*.tex"), key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


def _table_slice(tex: str, label: str) -> str:
    """
    Return the full \\begin{table}...\\end{table} block that contains \\label{<label>}.
    """
    marker = f"\\label{{{label}}}"
    idx = tex.find(marker)
    if idx < 0:
        raise ValueError(f"label not found: {label}")
    start = tex.rfind("\\begin{table}", 0, idx)
    if start < 0:
        raise ValueError(f"begin{{table}} not found for: {label}")
    end = tex.find("\\end{table}", idx)
    if end < 0:
        raise ValueError(f"end{{table}} not found for: {label}")
    return tex[start : end + len("\\end{table}")]


def _strip_html_lines(html: str) -> list[str]:
    # Naive but robust enough for the ILRS mission pages we use:
    # replace tags with newlines and collapse whitespace.
    text = re.sub(r"<[^>]+>", "\n", html)
    text = re.sub(r"\n+", "\n", text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines


def _parse_ilrs_nglr1_pa(html: str) -> dict[str, Any]:
    """
    Parse ILRS NGLR-1 page and return PA coordinates.

    Expected (tag-stripped) lines include:
      Retroreflector Coordinates (Principal Axis-PA):
      lon=...
      lat=...
      rad=...    (km)
      x=...
      y=...
      z=...      (km)
    """
    lines = _strip_html_lines(html)
    # Find the PA section.
    start = None
    for i, line in enumerate(lines):
        if "Retroreflector Coordinates (Principal Axis-PA)" in line:
            start = i
            break
    if start is None:
        raise ValueError("ILRS page parse failed: PA section header not found.")

    def _pick(prefix: str) -> float:
        for line in lines[start : start + 80]:
            if line.startswith(prefix):
                _, v = line.split("=", 1)
                return float(v.strip())
        raise ValueError(f"ILRS page parse failed: missing {prefix} in PA section.")

    lon_deg = _pick("lon")
    lat_deg = _pick("lat")
    rad_km = _pick("rad")
    x_km = _pick("x")
    y_km = _pick("y")
    z_km = _pick("z")

    km_to_m = 1000.0
    return {
        "name": "NGLR-1 (NGL-1)",
        "r_m": rad_km * km_to_m,
        "phi_deg": lat_deg,
        "lambda_deg": lon_deg,
        "x_m": x_km * km_to_m,
        "y_m": y_km * km_to_m,
        "z_m": z_km * km_to_m,
    }


_RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*])?(?:\{[^}]*})?")


def _clean_tex_cell(s: str) -> str:
    """
    Minimal TeX stripping for numeric cells in Murphy et al. tables.

    Note:
      - \\phtneg / \\phtdig / \\phtast are alignment phantoms (NOT signs).
    """
    s = s.strip()
    s = s.replace("\\phtneg", "")
    s = s.replace("\\phtdig", "")
    s = s.replace("\\phtast", "")
    s = s.replace("$", "")
    s = s.replace("{", "").replace("}", "")
    s = _RE_LATEX_CMD.sub("", s)
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s


def _clean_tex_name(s: str) -> str:
    s = s.strip()
    s = s.replace("\\phtneg", "")
    s = s.replace("\\phtdig", "")
    s = s.replace("\\phtast", "")
    s = s.replace("$", "")
    s = s.replace("{", "").replace("}", "")
    s = _RE_LATEX_CMD.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_float(cell: str) -> float:
    v = _clean_tex_cell(cell)
    if not v:
        raise ValueError(f"empty numeric cell: {cell!r}")
    return float(v)


def _split_rows(table_block: str) -> list[str]:
    m = re.search(r"\\begin\{tabular\}\{[^}]+\}", table_block)
    if not m:
        raise ValueError("missing tabular environment")
    body_start = m.end()
    body_end = table_block.find("\\end{tabular}", body_start)
    if body_end < 0:
        raise ValueError("missing end{tabular}")
    body = table_block[body_start:body_end]

    parts = body.split("\\tabularnewline")
    rows: list[str] = []
    for p in parts:
        r = p.replace("\n", " ").strip()
        if not r:
            continue
        r = r.replace("\\hline", "").strip()
        if not r:
            continue
        if r.startswith("Reflector&") or r.startswith("Reflector &"):
            continue
        rows.append(r)
    return rows


def _parse_all5(table_block: str) -> list[ParsedRow]:
    rows = _split_rows(table_block)
    out: list[ParsedRow] = []
    for r in rows:
        cells = [c.strip() for c in r.split("&")]
        if len(cells) < 7:
            continue
        name = _clean_tex_name(cells[0])
        if name.lower() == "reflector":
            continue
        out.append(
            ParsedRow(
                name=name,
                r_m=_to_float(cells[1]),
                phi_deg=_to_float(cells[2]),
                lam_deg=_to_float(cells[3]),
                x_m=_to_float(cells[4]),
                y_m=_to_float(cells[5]),
                z_m=_to_float(cells[6]),
            )
        )
    if not out:
        raise ValueError("no rows parsed from tab:all-5")
    return out


def _parse_tides(table_block: str) -> list[TideRow]:
    rows = _split_rows(table_block)
    out: list[TideRow] = []
    for r in rows:
        cells = [c.strip() for c in r.split("&")]
        if len(cells) < 4:
            continue
        name = _clean_tex_name(cells[0])
        if name.lower() == "reflector":
            continue
        out.append(
            TideRow(
                name=name,
                dr_m=_to_float(cells[1]),
                r_dphi_m=_to_float(cells[2]),
                r_dlambda_cosphi_m=_to_float(cells[3]),
            )
        )
    if not out:
        raise ValueError("no rows parsed from tab:tides")
    return out


def _canon_key(reflector_name: str) -> str:
    n = reflector_name.strip().lower()
    mapping = {
        "apollo 11": "apollo11",
        "apollo 14": "apollo14",
        "apollo 15": "apollo15",
        "lunokhod 1": "luna17",
        "lunokhod 2": "luna21",
    }
    if n in mapping:
        return mapping[n]
    raise ValueError(f"unknown reflector name in source table: {reflector_name!r}")


def main() -> int:
    root = _repo_root()
    sources_dir = root / "data" / "llr" / "sources"
    tar_path = sources_dir / f"arxiv_{ARXIV_ID}_e-print.tar.gz"
    extract_dir = sources_dir / f"arxiv_{ARXIV_ID}"
    ilrs_nglr1_path = sources_dir / "ilrs_ngl1_general.html"
    out_json = root / "data" / "llr" / "reflectors_de421_pa.json"

    ap = argparse.ArgumentParser(description="Fetch/parse LLR reflector coordinates from Murphy et al. (arXiv:1009.5720).")
    ap.add_argument("--offline", action="store_true", help="Do not use network; only succeed if cached source exists.")
    ap.add_argument("--force", action="store_true", help="Re-download and overwrite cached source / output JSON.")
    args = ap.parse_args()

    if args.offline:
        if not tar_path.exists():
            print(f"[err] offline and missing: {tar_path}")
            return 2
        if not ilrs_nglr1_path.exists():
            print(f"[err] offline and missing: {ilrs_nglr1_path}")
            return 2
    else:
        _download(ARXIV_EPRINT_URL, tar_path, force=bool(args.force))
        _download(ILRS_NGLR1_URL, ilrs_nglr1_path, force=bool(args.force))

    _extract_tar(tar_path, extract_dir, force=bool(args.force))

    tex_path = _find_tex_file(extract_dir)
    if tex_path is None or not tex_path.exists():
        print(f"[err] missing .tex in: {extract_dir}")
        return 3

    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    all5 = _table_slice(tex, "tab:all-5")
    tides = _table_slice(tex, "tab:tides")

    coords = _parse_all5(all5)
    tides_rows = _parse_tides(tides)

    tides_by_key = {_canon_key(r.name): r for r in tides_rows}

    display_name = {
        "apollo11": "Apollo 11 LRRR",
        "apollo14": "Apollo 14 LRRR",
        "apollo15": "Apollo 15 LRRR",
        "luna17": "Luna 17 / Lunokhod 1 LRRR",
        "luna21": "Luna 21 / Lunokhod 2 LRRR",
    }

    reflectors: dict[str, Any] = {}
    for r in coords:
        key = _canon_key(r.name)
        tr = tides_by_key.get(key)
        reflectors[key] = {
            "name": display_name.get(key) or r.name,
            "r_m": r.r_m,
            "phi_deg": r.phi_deg,
            "lambda_deg": r.lam_deg,
            "x_m": r.x_m,
            "y_m": r.y_m,
            "z_m": r.z_m,
            "static_tide_offsets": (
                {
                    "dr_m": tr.dr_m,
                    "r_dphi_m": tr.r_dphi_m,
                    "r_dlambda_cosphi_m": tr.r_dlambda_cosphi_m,
                }
                if tr is not None
                else None
            ),
        }

    # NGLR-1 (new reflector): source is ILRS mission page, PA coordinates (km → m).
    ilrs_html = ilrs_nglr1_path.read_text(encoding="utf-8", errors="replace")
    nglr1 = _parse_ilrs_nglr1_pa(ilrs_html)
    reflectors["nglr1"] = {
        **nglr1,
        "static_tide_offsets": None,
        "source": {
            "title": "ILRS mission page: NGLR-1 (NGL-1)",
            "url": ILRS_NGLR1_URL,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "cached_source": {
                "html_path": str(ilrs_nglr1_path),
                "html_sha256": _sha256(ilrs_nglr1_path) if ilrs_nglr1_path.exists() else None,
            },
            "notes": "Principal Axis-PA coordinates (km) parsed from ILRS page; converted to meters.",
        },
    }

    meta: dict[str, Any] = {
        "catalog": "waveP.llr.reflectors.de421_principal_axis.v3",
        "frame": "MOON_PA_DE421",
        "units": {"distance": "m", "angles": "deg"},
        "source": {
            "title": "Laser Ranging to the Lost Lunokhod 1 Reflector",
            "authors": [
                "T. W. Murphy Jr.",
                "E. G. Adelberger",
                "J. B. R. Battat",
                "C. D. Hoyle",
                "N. H. Johnson",
                "R. J. McMillan",
                "E. L. Michelsen",
                "C. W. Stubbs",
                "H. E. Swanson",
            ],
            "arxiv": ARXIV_ID,
            "arxiv_url": ARXIV_ABS_URL,
            "doi": JOURNAL_DOI,
            "tables": {"coordinates": "Table 4 (tab:all-5)", "static_tides": "Table 5 (tab:tides)"},
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "cached_source": {
                "tar_path": str(tar_path),
                "tar_sha256": _sha256(tar_path) if tar_path.exists() else None,
                "tex_path": str(tex_path),
                "tex_sha256": _sha256(tex_path) if tex_path.exists() else None,
            },
        },
        "additional_sources": [
            {
                "title": "ILRS mission page: NGLR-1 (NGL-1)",
                "url": ILRS_NGLR1_URL,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "cached_source": {
                    "html_path": str(ilrs_nglr1_path),
                    "html_sha256": _sha256(ilrs_nglr1_path) if ilrs_nglr1_path.exists() else None,
                },
                "notes": "Used for NGLR-1 coordinates (Principal Axis-PA).",
            }
        ],
        "notes": [
            "Moon-centered Cartesian coordinates in the DE421 principal-axis system (Murphy et al.).",
            "Static (permanent) tidal deformation is already incorporated in Table 4; to get the un-deformed Moon, subtract Table 5 offsets.",
            "NGLR-1 (nglr1) coordinates are taken from the ILRS mission page (Principal Axis-PA).",
            "This file is intended as an authoritative, reproducible catalog for waveP LLR checks.",
        ],
        "reflectors": reflectors,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_json}")
    for k in sorted(reflectors.keys()):
        v = reflectors[k]
        print(f"  - {k}: X,Y,Z = {v['x_m']:.3f}, {v['y_m']:.3f}, {v['z_m']:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
