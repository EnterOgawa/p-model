from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class FileSpec:
    url: str
    relpath: str


def _parse_energy_to_keV(obj: object) -> tuple[float, float | None] | None:
    if not isinstance(obj, dict):
        return None
    val_raw = obj.get("value")
    if val_raw is None:
        return None
    try:
        e_val = float(val_raw)
    except Exception:
        return None
    unit = str(obj.get("unit", "")).strip().lower()
    if unit in {"kev", ""}:
        e_keV = float(e_val)
    elif unit == "mev":
        e_keV = float(e_val) * 1000.0
    else:
        return None

    sigma_raw = obj.get("uncertainty")
    try:
        e_sigma = float(sigma_raw) if sigma_raw is not None else None
    except Exception:
        e_sigma = None
    return float(e_keV), (float(e_sigma) if e_sigma is not None else None)


def _parse_dimensionless(obj: object) -> tuple[float, float | None] | None:
    if not isinstance(obj, dict):
        return None
    val_raw = obj.get("value")
    if val_raw is None:
        return None
    try:
        v = float(val_raw)
    except Exception:
        return None
    sigma_raw = obj.get("uncertainty")
    try:
        s = float(sigma_raw) if sigma_raw is not None else None
    except Exception:
        s = None
    return float(v), (float(s) if s is not None else None)


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
    with urlopen(req, timeout=120) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")
    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch NNDC NuDat 3.0 chart JSON (primary.json + secondary.json) and extract low-lying spectroscopy "
            "(E(2+_1), E(4+_1), E(3-_1), R4/2) per nuclide from secondary.json: excitedStateEnergies.*. "
            "Writes manifest.json plus extracted_e2plus.json and extracted_spectroscopy.json under "
            "data/quantum/sources/ for offline reproducibility."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--out-dirname",
        default="nndc_nudat3_primary_secondary",
        help="Output directory name under data/quantum/sources/ (default: nndc_nudat3_primary_secondary).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / str(args.out_dirname)
    src_dir.mkdir(parents=True, exist_ok=True)

    primary_url = "https://www.nndc.bnl.gov/nudat3/data/primary.json"
    secondary_url = "https://www.nndc.bnl.gov/nudat3/data/secondary.json"
    files = [
        FileSpec(url=primary_url, relpath="primary.json"),
        FileSpec(url=secondary_url, relpath="secondary.json"),
    ]

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

    primary_path = src_dir / "primary.json"
    secondary_path = src_dir / "secondary.json"

    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    secondary = json.loads(secondary_path.read_text(encoding="utf-8"))
    if not isinstance(primary, dict) or not primary:
        raise SystemExit(f"[fail] invalid primary.json (expected non-empty object): {primary_path}")
    if not isinstance(secondary, dict) or not secondary:
        raise SystemExit(f"[fail] invalid secondary.json (expected non-empty object): {secondary_path}")

    # key -> (Z, N, A, name)
    meta_by_key: dict[str, tuple[int, int, int, str]] = {}
    n_skipped_primary = 0
    for key, v in primary.items():
        if not isinstance(v, dict):
            n_skipped_primary += 1
            continue
        try:
            Z = int(v.get("z", -1))
            N = int(v.get("n", -1))
            A = int(v.get("a", -1))
        except Exception:
            n_skipped_primary += 1
            continue
        if Z < 1 or N < 0 or A < 2:
            n_skipped_primary += 1
            continue
        name = str(v.get("name") or v.get("label") or key)
        meta_by_key[str(key)] = (int(Z), int(N), int(A), name)

    rows: list[dict[str, object]] = []
    n_missing_meta = 0
    n_missing_e2 = 0
    n_bad_unit = 0
    n_bad_value = 0

    for key, v in secondary.items():
        if not isinstance(v, dict):
            continue
        ese = v.get("excitedStateEnergies") if isinstance(v.get("excitedStateEnergies"), dict) else {}

        e2 = _parse_energy_to_keV(ese.get("firstTwoPlusEnergy"))
        e4 = _parse_energy_to_keV(ese.get("firstFourPlusEnergy"))
        e3m = _parse_energy_to_keV(ese.get("firstThreeMinusEnergy"))
        r42 = _parse_dimensionless(ese.get("firstFourPlusOverFirstTwoPlusEnergy"))

        if e2 is None:
            n_missing_e2 += 1
        if e2 is None and e4 is None and e3m is None and r42 is None:
            continue

        meta = meta_by_key.get(str(key))
        if meta is None:
            n_missing_meta += 1
            continue
        Z, N, A, name = meta
        row: dict[str, object] = {
            "key": str(key),
            "name": str(name),
            "Z": int(Z),
            "N": int(N),
            "A": int(A),
        }
        if e2 is not None:
            e2_keV, e2_sig = e2
            row["e2plus_keV"] = float(e2_keV)
            row["e2plus_sigma_keV"] = float(e2_sig) if e2_sig is not None else None
        if e4 is not None:
            e4_keV, e4_sig = e4
            row["e4plus_keV"] = float(e4_keV)
            row["e4plus_sigma_keV"] = float(e4_sig) if e4_sig is not None else None
        if e3m is not None:
            e3_keV, e3_sig = e3m
            row["e3minus_keV"] = float(e3_keV)
            row["e3minus_sigma_keV"] = float(e3_sig) if e3_sig is not None else None
        if r42 is not None:
            r_val, r_sig = r42
            row["r42"] = float(r_val)
            row["r42_sigma"] = float(r_sig) if r_sig is not None else None
        rows.append(
            row
        )

    rows = sorted(rows, key=lambda r: (int(r.get("Z", 9999)), int(r.get("N", 9999))))
    rows_e2 = [
        {
            "key": r["key"],
            "name": r["name"],
            "Z": r["Z"],
            "N": r["N"],
            "A": r["A"],
            "e2plus_keV": r.get("e2plus_keV"),
            "e2plus_sigma_keV": r.get("e2plus_sigma_keV"),
        }
        for r in rows
        if isinstance(r, dict) and r.get("e2plus_keV") is not None
    ]

    out_e2 = src_dir / "extracted_e2plus.json"
    out_e2.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "dataset": "NNDC NuDat 3.0 chart data (secondary.json: excitedStateEnergies.firstTwoPlusEnergy)",
                "units": {"e2plus_keV": "keV", "e2plus_sigma_keV": "keV"},
                "source": {
                    "nudat3": "https://www.nndc.bnl.gov/nudat3/",
                    "primary_url": primary_url,
                    "secondary_url": secondary_url,
                    "notes": [
                        "This dataset is derived from NuDat 3.0 static JSON used by the chart of nuclides UI.",
                        "E(2+_1) is taken from excitedStateEnergies.firstTwoPlusEnergy (keV).",
                        "This script caches primary.json/secondary.json for offline reproducibility; extracted_e2plus.json is derived from them.",
                    ],
                },
                "diag": {
                    "n_primary_keys": int(len(primary)),
                    "n_secondary_keys": int(len(secondary)),
                    "n_primary_skipped": int(n_skipped_primary),
                    "n_rows": int(len(rows_e2)),
                    "n_missing_meta": int(n_missing_meta),
                    "n_missing_e2": int(n_missing_e2),
                    "n_bad_unit": int(n_bad_unit),
                    "n_bad_value": int(n_bad_value),
                },
                "rows": rows_e2,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_multi = src_dir / "extracted_spectroscopy.json"
    out_multi.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "dataset": "NNDC NuDat 3.0 chart data (secondary.json: excitedStateEnergies.*)",
                "units": {
                    "e2plus_keV": "keV",
                    "e2plus_sigma_keV": "keV",
                    "e4plus_keV": "keV",
                    "e4plus_sigma_keV": "keV",
                    "e3minus_keV": "keV",
                    "e3minus_sigma_keV": "keV",
                    "r42": "dimensionless",
                    "r42_sigma": "dimensionless",
                },
                "source": {
                    "nudat3": "https://www.nndc.bnl.gov/nudat3/",
                    "primary_url": primary_url,
                    "secondary_url": secondary_url,
                    "notes": [
                        "This dataset is derived from NuDat 3.0 static JSON used by the chart of nuclides UI.",
                        "Extracted fields (when present): firstTwoPlusEnergy, firstFourPlusEnergy, firstThreeMinusEnergy, firstFourPlusOverFirstTwoPlusEnergy.",
                        "This script caches primary.json/secondary.json for offline reproducibility; extracted_spectroscopy.json is derived from them.",
                    ],
                },
                "diag": {
                    "n_primary_keys": int(len(primary)),
                    "n_secondary_keys": int(len(secondary)),
                    "n_primary_skipped": int(n_skipped_primary),
                    "n_rows": int(len(rows)),
                    "n_missing_meta": int(n_missing_meta),
                    "n_missing_e2": int(n_missing_e2),
                    "n_bad_unit": int(n_bad_unit),
                    "n_bad_value": int(n_bad_value),
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest: dict[str, object] = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Phase 7 spectroscopy input (NuDat 3.0 chart JSON; low-lying energies + R4/2)",
        "notes": [
            "Primary source is NNDC NuDat 3.0 static chart data (primary.json + secondary.json).",
            f"Primary JSON: {primary_url}",
            f"Secondary JSON: {secondary_url}",
            "The extracted_*.json files are derived from the JSON and are intended for offline reproducible analysis.",
        ],
        "files": [],
    }

    def add_file(*, url: str | None, path: Path, extra: dict[str, object] | None = None) -> None:
        item: dict[str, object] = {"url": url, "path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path)}
        if extra:
            item.update(extra)
        manifest["files"].append(item)

    for spec in files:
        add_file(url=spec.url, path=src_dir / spec.relpath)
    add_file(url=None, path=out_e2, extra={"derived_from": [str(primary_path), str(secondary_path)]})
    add_file(url=None, path=out_multi, extra={"derived_from": [str(primary_path), str(secondary_path)]})

    out_manifest = src_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] extracted: {out_e2}")
    print(f"[ok] extracted: {out_multi}")
    print(f"[ok] manifest : {out_manifest}")


if __name__ == "__main__":
    main()
