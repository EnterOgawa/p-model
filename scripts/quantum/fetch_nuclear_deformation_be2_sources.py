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
    with urlopen(req, timeout=60) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")
    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _rank_reference(ref: str) -> int:
    # Prefer the 2016 evaluation over the 2001 one when both are present.
    order = {
        "ADNDT_107_1_2016_2016PR01": 2,
        "ADNDT_78_1_2001_2001RA27": 1,
    }
    return int(order.get(str(ref), 0))


def _rank_entry_type(entry_type: str) -> int:
    # Prefer model-independent when duplicates exist.
    order = {
        "MODEL_INDEPENDENT": 3,
        "LOW_MODEL_DEPENDENT": 2,
        "MODEL_DEPENDENT": 1,
    }
    return int(order.get(str(entry_type), 0))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch NNDC B(E2) adopted-entries.json and extract deformationParameter β (dimensionless) per nuclide. "
            "Writes manifest.json and extracted_beta2.json under data/quantum/sources/ for offline reproducibility."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--out-dirname",
        default="nndc_be2_adopted_entries",
        help="Output directory name under data/quantum/sources/ (default: nndc_be2_adopted_entries).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / str(args.out_dirname)
    src_dir.mkdir(parents=True, exist_ok=True)

    base_page = "https://www.nndc.bnl.gov/be2/"
    json_url = "https://www.nndc.bnl.gov/be2/data/adopted-entries.json"
    files = [FileSpec(url=json_url, relpath="adopted-entries.json")]

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

    raw_path = src_dir / "adopted-entries.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"[fail] invalid adopted-entries.json (expected non-empty list): {raw_path}")

    # Extract E(2+1) proxy: the transitionEnergy in the adopted entries (typically 2+->0+ in keV),
    # which equals E(2+1) for even-even nuclei with a 0+ ground state.
    best_e2: dict[tuple[int, int], dict[str, object]] = {}
    n_skipped_e2 = 0
    for entry in raw:
        if not isinstance(entry, dict):
            n_skipped_e2 += 1
            continue
        nuclide = entry.get("nuclide") if isinstance(entry.get("nuclide"), dict) else {}
        Z = int(nuclide.get("z", -1))
        N = int(nuclide.get("n", -1))
        A = int(nuclide.get("a", -1))
        if Z < 1 or N < 0 or A < 2:
            n_skipped_e2 += 1
            continue

        te = entry.get("transitionEnergy") if isinstance(entry.get("transitionEnergy"), dict) else None
        if not isinstance(te, dict) or "value" not in te:
            n_skipped_e2 += 1
            continue
        try:
            e_val = float(te.get("value"))
        except Exception:
            n_skipped_e2 += 1
            continue
        te_err = te.get("error")
        try:
            e_sigma = float(te_err) if te_err is not None else float("nan")
        except Exception:
            e_sigma = float("nan")

        ref = str(entry.get("reference", ""))
        etype = str(entry.get("adoptedEntryType", ""))
        key = (int(Z), int(N))
        cand_e2 = {
            "Z": int(Z),
            "N": int(N),
            "A": int(A),
            "e2plus_keV": float(e_val),
            "e2plus_sigma_keV": float(e_sigma) if (e_sigma == e_sigma) else None,  # NaN -> None
            "reference": ref,
            "adoptedEntryType": etype,
        }

        prev = best_e2.get(key)
        if prev is None:
            best_e2[key] = cand_e2
            continue

        r_prev = _rank_reference(str(prev.get("reference", "")))
        r_cand = _rank_reference(ref)
        if r_cand != r_prev:
            if r_cand > r_prev:
                best_e2[key] = cand_e2
            continue

        t_prev = _rank_entry_type(str(prev.get("adoptedEntryType", "")))
        t_cand = _rank_entry_type(etype)
        if t_cand != t_prev:
            if t_cand > t_prev:
                best_e2[key] = cand_e2
            continue

        s_prev = prev.get("e2plus_sigma_keV")
        s_cand = cand_e2.get("e2plus_sigma_keV")
        if isinstance(s_prev, (int, float)) and isinstance(s_cand, (int, float)):
            if (s_cand == s_cand) and (s_prev == s_prev) and float(s_cand) < float(s_prev):
                best_e2[key] = cand_e2
                continue

    # Keep one best entry per (Z,N) based on:
    #   - higher reference rank (2016 > 2001)
    #   - higher adoptedEntryType rank (MODEL_INDEPENDENT > ...)
    #   - lower reported uncertainty (if tied)
    best: dict[tuple[int, int], dict[str, object]] = {}
    n_skipped = 0
    for entry in raw:
        if not isinstance(entry, dict):
            n_skipped += 1
            continue
        nuclide = entry.get("nuclide") if isinstance(entry.get("nuclide"), dict) else {}
        Z = int(nuclide.get("z", -1))
        N = int(nuclide.get("n", -1))
        A = int(nuclide.get("a", -1))
        if Z < 1 or N < 0 or A < 2:
            n_skipped += 1
            continue

        beta = entry.get("deformationParameter") if isinstance(entry.get("deformationParameter"), dict) else None
        if not isinstance(beta, dict) or "value" not in beta:
            n_skipped += 1
            continue
        try:
            beta_val = float(beta.get("value"))
        except Exception:
            n_skipped += 1
            continue
        beta_err = beta.get("error")
        try:
            beta_sigma = float(beta_err) if beta_err is not None else float("nan")
        except Exception:
            beta_sigma = float("nan")

        ref = str(entry.get("reference", ""))
        etype = str(entry.get("adoptedEntryType", ""))
        key = (int(Z), int(N))

        cand = {
            "Z": int(Z),
            "N": int(N),
            "A": int(A),
            "beta2": float(beta_val),
            "beta2_sigma": float(beta_sigma) if (beta_sigma == beta_sigma) else None,  # NaN -> None
            "reference": ref,
            "adoptedEntryType": etype,
        }

        prev = best.get(key)
        if prev is None:
            best[key] = cand
            continue

        r_prev = _rank_reference(str(prev.get("reference", "")))
        r_cand = _rank_reference(ref)
        if r_cand != r_prev:
            if r_cand > r_prev:
                best[key] = cand
            continue

        t_prev = _rank_entry_type(str(prev.get("adoptedEntryType", "")))
        t_cand = _rank_entry_type(etype)
        if t_cand != t_prev:
            if t_cand > t_prev:
                best[key] = cand
            continue

        # Tie-breaker: smaller uncertainty wins (if both finite).
        s_prev = prev.get("beta2_sigma")
        s_cand = cand.get("beta2_sigma")
        if isinstance(s_prev, (int, float)) and isinstance(s_cand, (int, float)):
            if (s_cand == s_cand) and (s_prev == s_prev) and float(s_cand) < float(s_prev):
                best[key] = cand
                continue

        # Otherwise keep first.

    rows = [best[k] for k in sorted(best.keys())]
    out_extracted = src_dir / "extracted_beta2.json"
    out_extracted.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "dataset": "NNDC B(E2) adopted entries (includes deformationParameter β derived from B(E2;0+→2+))",
                "source": {
                    "page": base_page,
                    "json_url": json_url,
                    "notes": [
                        "The NNDC page cites: B. Pritychenko et al., At. Data Nucl. Data Tables 107, 1 (2016) and the earlier evaluation ADNDT 78, 1 (2001).",
                        "This script caches the NNDC-served JSON for offline reproducibility; extracted_beta2.json is derived from it.",
                    ],
                },
                "selection_policy": {
                    "per_nuclide_best_entry": [
                        "Prefer reference ADNDT 107, 1 (2016) over ADNDT 78, 1 (2001) if both exist.",
                        "Prefer adoptedEntryType MODEL_INDEPENDENT > LOW_MODEL_DEPENDENT > MODEL_DEPENDENT.",
                        "If tied, prefer smaller beta2_sigma (when finite).",
                    ]
                },
                "diag": {"n_raw_entries": int(len(raw)), "n_unique_nuclides": int(len(rows)), "n_skipped": int(n_skipped)},
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    e2_rows = [best_e2[k] for k in sorted(best_e2.keys())]
    out_e2 = src_dir / "extracted_e2plus.json"
    out_e2.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "dataset": "NNDC B(E2) adopted entries (transitionEnergy as E(2+1) proxy in keV)",
                "units": {"e2plus_keV": "keV", "e2plus_sigma_keV": "keV"},
                "source": {
                    "page": base_page,
                    "json_url": json_url,
                    "notes": [
                        "transitionEnergy in the adopted entries is typically the 2+->0+ gamma energy (keV).",
                        "For even-even nuclei with a 0+ ground state, this equals E(2+_1).",
                        "This script caches the NNDC-served JSON for offline reproducibility; extracted_e2plus.json is derived from it.",
                    ],
                },
                "selection_policy": {
                    "per_nuclide_best_entry": [
                        "Prefer reference ADNDT 107, 1 (2016) over ADNDT 78, 1 (2001) if both exist.",
                        "Prefer adoptedEntryType MODEL_INDEPENDENT > LOW_MODEL_DEPENDENT > MODEL_DEPENDENT.",
                        "If tied, prefer smaller e2plus_sigma_keV (when finite).",
                    ]
                },
                "diag": {"n_raw_entries": int(len(raw)), "n_unique_nuclides": int(len(e2_rows)), "n_skipped": int(n_skipped_e2)},
                "rows": e2_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest: dict[str, object] = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Phase 7 nuclear structure input (β2 / E(2+1) proxy from NNDC B(E2) adopted entries)",
        "notes": [
            f"Base page: {base_page}",
            f"Primary JSON: {json_url}",
            "The extracted_beta2.json is derived from the JSON and is intended for offline reproducible analysis.",
            "The extracted_e2plus.json is derived from the same JSON (transitionEnergy as E(2+1) proxy).",
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
    add_file(url=None, path=out_extracted, extra={"derived_from": str(raw_path)})
    add_file(url=None, path=out_e2, extra={"derived_from": str(raw_path)})

    out_manifest = src_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] extracted: {out_extracted}")
    print(f"[ok] extracted: {out_e2}")
    print(f"[ok] manifest : {out_manifest}")


if __name__ == "__main__":
    main()
