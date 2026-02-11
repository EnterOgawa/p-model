from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_constant(extracted: dict[str, Any], *, code: str) -> dict[str, Any]:
    constants = extracted.get("constants")
    if not isinstance(constants, dict):
        raise SystemExit("[fail] extracted_values.json missing 'constants' dict")
    rec = constants.get(code)
    if not isinstance(rec, dict):
        raise SystemExit(f"[fail] missing constant '{code}' in extracted_values.json")
    return rec


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_silicon_lattice"
    extracted_path = src_dir / "extracted_values.json"
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_silicon_lattice_sources.py"
        )

    extracted = _read_json(extracted_path)
    asil = _require_constant(extracted, code="asil")
    d220sil = _require_constant(extracted, code="d220sil")

    records: list[dict[str, Any]] = []
    for code, rec in [("asil", asil), ("d220sil", d220sil)]:
        name = str(rec.get("name") or "")
        value_m = float(rec.get("value_si"))
        sigma_m = float(rec.get("sigma_si"))
        unit = str(rec.get("unit_si") or "")

        value_A = value_m / 1e-10
        sigma_A = sigma_m / 1e-10
        rel = sigma_m / value_m if value_m != 0 else float("nan")

        records.append(
            {
                "code": code,
                "name": name,
                "value_m": value_m,
                "sigma_m": sigma_m,
                "unit_si": unit,
                "value_A": value_A,
                "sigma_A": sigma_A,
                "rel_sigma": rel,
                "source_url": rec.get("url"),
                "local_path": rec.get("local_path"),
                "local_sha256": rec.get("local_sha256"),
                "codata_year": rec.get("codata_year"),
            }
        )

    out_csv = out_dir / "condensed_silicon_lattice_baseline.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "code",
                "name",
                "value_m",
                "sigma_m",
                "unit_si",
                "value_A",
                "sigma_A",
                "rel_sigma",
                "codata_year",
                "source_url",
                "local_path",
                "local_sha256",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # Plot (Å scale).
    xs = list(range(len(records)))
    ys = [float(r["value_A"]) for r in records]
    yerr = [float(r["sigma_A"]) for r in records]
    labels = [str(r["code"]) for r in records]

    plt.figure(figsize=(7, 3.6))
    plt.errorbar(xs, ys, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(xs, labels)
    plt.ylabel("Value (Å)")
    plt.title("Silicon lattice constants (NIST CODATA)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_png = out_dir / "condensed_silicon_lattice_baseline.png"
    plt.savefig(out_png, dpi=180)
    plt.close()

    falsification_targets = []
    sigma_multiplier = 3.0
    for r in records:
        sigma_m = float(r["sigma_m"])
        falsification_targets.append(
            {
                "code": str(r["code"]),
                "target_value_m": float(r["value_m"]),
                "sigma_m": sigma_m,
                "reject_if_abs_minus_target_gt_m": float(sigma_multiplier * sigma_m),
            }
        )

    out_metrics = out_dir / "condensed_silicon_lattice_baseline_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.14.1",
                "inputs": {
                    "extracted_values": {
                        "path": str(extracted_path),
                        "sha256": _sha256(extracted_path),
                    }
                },
                "falsification": {
                    "sigma_multiplier": sigma_multiplier,
                    "targets": falsification_targets,
                    "notes": [
                        "CODATA provides explicit uncertainties; reject thresholds use a strict ±3σ envelope.",
                        "This does not claim P-model predicts these material-specific values today; it freezes targets for future derivations.",
                    ],
                },
                "results": records,
                "outputs": {
                    "csv": str(out_csv),
                    "png": str(out_png),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_metrics}")


if __name__ == "__main__":
    main()
