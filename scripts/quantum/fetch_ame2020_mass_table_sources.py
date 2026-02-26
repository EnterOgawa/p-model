from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


# クラス: `FileSpec` の責務と境界条件を定義する。
@dataclass(frozen=True)
class FileSpec:
    url: str
    relpath: str


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


# 関数: `_parse_int` の入出力契約と処理意図を定義する。

def _parse_int(s: str) -> int | None:
    try:
        return int(s.strip())
    except Exception:
        return None


# 関数: `_parse_float` の入出力契約と処理意図を定義する。

def _parse_float(s: str) -> float | None:
    t = s.strip()
    # 条件分岐: `not t or t == "*" or "*" in t` を満たす経路を評価する。
    if not t or t == "*" or "*" in t:
        return None
    # AME uses '#' as "estimated" marker in place of decimal point (or near values).

    t = t.replace("#", "")
    try:
        return float(t)
    except Exception:
        return None


# 関数: `_iter_rows_mass_1_mas20` の入出力契約と処理意図を定義する。

def _iter_rows_mass_1_mas20(text: str) -> list[dict[str, object]]:
    """
    Parse AME2020 mass_1.mas20 fixed-width lines.

    Source format (from header):
      a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,f13.5,1x,f10.5,1x,a2,f13.5,f11.5,1x,i3,1x,f13.6,f12.6
    """
    rows: list[dict[str, object]] = []
    for ln in text.splitlines():
        # 条件分岐: `not ln` を満たす経路を評価する。
        if not ln:
            continue
        # Some lines in the distributed text file may omit trailing spaces.
        # Pad to the expected fixed-width length so slice-based parsing is stable.

        if len(ln) < 80:
            continue

        # 条件分岐: `len(ln) < 135` を満たす経路を評価する。

        if len(ln) < 135:
            ln = ln.ljust(135)

        # Data lines should have parseable N/Z/A fields in fixed positions.

        n = _parse_int(ln[4:9])
        z = _parse_int(ln[9:14])
        a = _parse_int(ln[14:19])
        # 条件分岐: `n is None or z is None or a is None` を満たす経路を評価する。
        if n is None or z is None or a is None:
            continue

        el = ln[20:23].strip()
        origin = ln[23:27].strip()
        mass_excess_kev = _parse_float(ln[28:42])
        mass_excess_sigma_kev = _parse_float(ln[42:54])
        binding_kev_per_a = _parse_float(ln[54:67])
        binding_sigma_kev_per_a = _parse_float(ln[68:78])

        rows.append(
            {
                "Z": int(z),
                "N": int(n),
                "A": int(a),
                "symbol": el,
                "origin": origin,
                "mass_excess_keV": mass_excess_kev,
                "mass_excess_sigma_keV": mass_excess_sigma_kev,
                "binding_keV_per_A": binding_kev_per_a,
                "binding_sigma_keV_per_A": binding_sigma_kev_per_a,
                "raw_line": ln,
            }
        )

    return rows


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch AME2020 mass table (IAEA AMDC 'mass_1.mas20' text) and extract a compact JSON table "
            "for offline reproducible nuclear binding-energy baselines."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--out-dirname",
        default="iaea_amdc_ame2020_mass_1_mas20",
        help="Output directory name under data/quantum/sources/ (default: iaea_amdc_ame2020_mass_1_mas20).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / str(args.out_dirname)
    src_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt"
    files = [FileSpec(url=url, relpath="mass_1.mas20.txt")]

    # 条件分岐: `not args.offline` を満たす経路を評価する。
    if not args.offline:
        for spec in files:
            _download(spec.url, src_dir / spec.relpath)

    missing: list[Path] = []
    for spec in files:
        p = src_dir / spec.relpath
        # 条件分岐: `not p.exists() or p.stat().st_size == 0` を満たす経路を評価する。
        if not p.exists() or p.stat().st_size == 0:
            missing.append(p)

    # 条件分岐: `missing` を満たす経路を評価する。

    if missing:
        raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

    mass_path = src_dir / "mass_1.mas20.txt"
    text = mass_path.read_text(encoding="utf-8", errors="replace")
    rows = _iter_rows_mass_1_mas20(text)
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        raise SystemExit(f"[fail] parsed 0 rows from: {mass_path}")

    # Representative nuclei used in Step 7.13.11+ (A-dependence baseline).

    selected = [
        {"key": "d", "Z": 1, "A": 2, "label": "deuteron (H-2)"},
        {"key": "t", "Z": 1, "A": 3, "label": "triton (H-3)"},
        {"key": "h", "Z": 2, "A": 3, "label": "helion (He-3)"},
        {"key": "alpha", "Z": 2, "A": 4, "label": "alpha (He-4)"},
        {"key": "li6", "Z": 3, "A": 6, "label": "Li-6"},
        {"key": "c12", "Z": 6, "A": 12, "label": "C-12"},
        {"key": "o16", "Z": 8, "A": 16, "label": "O-16"},
        {"key": "ca40", "Z": 20, "A": 40, "label": "Ca-40"},
        {"key": "fe56", "Z": 26, "A": 56, "label": "Fe-56"},
        {"key": "ni62", "Z": 28, "A": 62, "label": "Ni-62"},
        {"key": "pb208", "Z": 82, "A": 208, "label": "Pb-208"},
    ]

    extracted = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "AME2020 mass table (mass_1.mas20) parsed table (binding energy per nucleon, mass excess)",
        "source": {
            "provider": "IAEA AMDC",
            "url": url,
            "reference_papers": [
                {
                    "title": "The Ame2020 atomic mass evaluation (I)",
                    "journal": "Chinese Physics C",
                    "volume": "45",
                    "article": "030002",
                    "year": 2021,
                    "authors": ["W.J. Huang", "M. Wang", "F.G. Kondev", "G. Audi", "S. Naimi"],
                },
                {
                    "title": "The Ame2020 atomic mass evaluation (II)",
                    "journal": "Chinese Physics C",
                    "volume": "45",
                    "article": "030003",
                    "year": 2021,
                    "authors": ["M. Wang", "W.J. Huang", "F.G. Kondev", "G. Audi", "S. Naimi"],
                },
            ],
        },
        "format": {
            "type": "fixed_width",
            "line_length": 135,
            "fields": {
                "N": {"slice": [4, 9], "type": "int"},
                "Z": {"slice": [9, 14], "type": "int"},
                "A": {"slice": [14, 19], "type": "int"},
                "symbol": {"slice": [20, 23], "type": "str"},
                "origin": {"slice": [23, 27], "type": "str"},
                "mass_excess_keV": {"slice": [28, 42], "type": "float"},
                "mass_excess_sigma_keV": {"slice": [42, 54], "type": "float"},
                "binding_keV_per_A": {"slice": [54, 67], "type": "float"},
                "binding_sigma_keV_per_A": {"slice": [68, 78], "type": "float"},
            },
            "notes": [
                "Some lines may omit trailing spaces; the parser right-pads them to the nominal fixed width.",
                "AME uses '#' markers for estimated values; this parser strips '#' and keeps the raw_line.",
                "Some rows may contain '*' for non-calculable quantities; such numeric fields are set to null.",
            ],
        },
        "selected": selected,
        "rows": rows,
    }

    out_extracted = src_dir / "extracted_values.json"
    out_extracted.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest: dict[str, object] = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Phase 7 nuclear baseline primary sources (IAEA AMDC AME2020 mass table)",
        "notes": [
            f"Primary text file: {url}",
            "The extracted_values.json is derived from the text file and is intended for offline reproducible analysis.",
        ],
        "files": [],
    }

    # 関数: `add_file` の入出力契約と処理意図を定義する。
    def add_file(*, url: str | None, path: Path, extra: dict[str, object] | None = None) -> None:
        item = {"url": url, "path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path)}
        # 条件分岐: `extra` を満たす経路を評価する。
        if extra:
            item.update(extra)

        assert isinstance(manifest["files"], list)
        manifest["files"].append(item)

    for spec in files:
        add_file(url=spec.url, path=src_dir / spec.relpath)

    add_file(url=None, path=out_extracted, extra={"derived_from": str(mass_path)})

    out_manifest = src_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] extracted: {out_extracted}")
    print(f"[ok] manifest : {out_manifest}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
