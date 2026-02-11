from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class FileSpec:
    url: str
    relpath: str


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


def _find_unique_row(rows: list[dict[str, str]], *, z: int, n: int, a: int) -> dict[str, str]:
    matches = [r for r in rows if int(r["z"]) == z and int(r["n"]) == n and int(r["a"]) == a]
    if not matches:
        raise ValueError(f"row not found: z={z} n={n} a={a}")
    if len(matches) != 1:
        raise ValueError(f"row not unique: z={z} n={n} a={a} (n_matches={len(matches)})")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch a compiled nuclear charge-radii table (IAEA radii database CSV) and extract light-nuclei A=3 "
            "charge rms radii (tritium/helion) for offline reproducible analysis."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--out-dirname",
        default="iaea_charge_radii",
        help="Output directory name under data/quantum/sources/ (default: iaea_charge_radii).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / str(args.out_dirname)
    src_dir.mkdir(parents=True, exist_ok=True)

    base_page = "https://www-nds.iaea.org/radii/"
    csv_url = "https://www-nds.iaea.org/radii/charge_radii.csv"
    doi = "10.1016/j.adt.2011.12.006"

    files = [FileSpec(url=csv_url, relpath="charge_radii.csv")]
    if not args.offline:
        for spec in files:
            _download(spec.url, src_dir / spec.relpath)

    missing: list[Path] = []
    for spec in files:
        p = src_dir / spec.relpath
        if not p.exists() or p.stat().st_size == 0:
            missing.append(p)
    if missing:
        raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

    csv_path = src_dir / "charge_radii.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]

    # Extract A=3 radii:
    #   tritium (H-3): Z=1, N=2, A=3
    #   helion  (He-3): Z=2, N=1, A=3
    row_t = _find_unique_row(rows, z=1, n=2, a=3)
    row_h = _find_unique_row(rows, z=2, n=1, a=3)

    def pack(row: dict[str, str], *, key: str) -> dict[str, object]:
        radius_val = row.get("radius_val", "").strip()
        radius_unc = row.get("radius_unc", "").strip()
        if not radius_val or not radius_unc:
            raise ValueError(f"missing radius fields for key={key}: {row}")
        return {
            "key": key,
            "Z": int(row["z"]),
            "symbol": str(row.get("symbol", "")).strip(),
            "N": int(row["n"]),
            "A": int(row["a"]),
            "radius_fm": float(radius_val),
            "radius_sigma_fm": float(radius_unc),
            "raw_row": row,
        }

    extracted = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "IAEA nuclear charge radii compilation (charge_radii.csv)",
        "source": {"page": base_page, "csv_url": csv_url, "doi": doi},
        "extraction": {
            "selected": [
                {"key": "t", "Z": 1, "N": 2, "A": 3, "label": "tritium (H-3)"},
                {"key": "h", "Z": 2, "N": 1, "A": 3, "label": "helion (He-3)"},
            ],
            "notes": [
                "This dataset is a compilation; it is used here as a dedicated source for A=3 charge radii.",
                "The full CSV is cached for offline reproducibility; extracted_values.json is derived from it.",
            ],
        },
        "radii": {"t": pack(row_t, key="t"), "h": pack(row_h, key="h")},
    }

    out_extracted = src_dir / "extracted_values.json"
    out_extracted.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Phase 7 nuclear baseline primary sources (IAEA charge radii compilation)",
        "notes": [
            f"Primary CSV: {csv_url}",
            f"Reference page: {base_page}",
            f"DOI: {doi}",
            "The extracted_values.json is derived from the CSV and is intended for offline reproducible analysis.",
        ],
        "files": [],
    }

    def add_file(*, url: str | None, path: Path, extra: dict[str, object] | None = None) -> None:
        item = {"url": url, "path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path)}
        if extra:
            item.update(extra)
        manifest["files"].append(item)

    for spec in files:
        add_file(url=spec.url, path=src_dir / spec.relpath)
    add_file(url=None, path=out_extracted, extra={"derived_from": str(csv_path)})

    out_manifest = src_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] extracted: {out_extracted}")
    print(f"[ok] manifest : {out_manifest}")


if __name__ == "__main__":
    main()

