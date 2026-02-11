#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
condensed_falsification_pack.py

Phase 7 / Step 7.18.2（Part III closeout）:
物性/熱（凝縮系 + 熱統計の基準量）について、各 baseline_metrics.json に含まれる
falsification（棄却閾値・適用条件）を束ねた “反証条件パック（JSON）” を固定出力する。

出力（固定）:
  - output/public/quantum/condensed_falsification_pack.json

方針:
- 再計算は行わない（既に生成済みの metrics JSON を入力として束ねる）。
- 具体的な棄却閾値は、各 metrics JSON の `falsification` を正とする。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pack(*, root: Path, metrics_relpaths: List[str]) -> Dict[str, Any]:
    tests: List[Dict[str, Any]] = []
    missing: List[str] = []

    for rel in metrics_relpaths:
        p = (root / rel).resolve()
        if not p.exists():
            missing.append(rel)
            continue
        j = _read_json(p)
        tests.append(
            {
                "metrics_json": _relpath(p),
                "metrics_sha256": _sha256(p),
                "phase": j.get("phase"),
                "step": j.get("step"),
                "inputs": j.get("inputs"),
                "falsification": j.get("falsification"),
                "outputs": j.get("outputs"),
                "notes": j.get("notes"),
            }
        )

    if missing:
        raise SystemExit(
            "[fail] missing metrics JSON(s):\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\n"
            "Run the corresponding baseline scripts under scripts/quantum/ to generate them."
        )

    return {
        "generated_utc": _iso_utc_now(),
        "phase": 7,
        "step": "7.18.2",
        "name": "Condensed/Thermo falsification pack (baseline targets + thresholds)",
        "policy": {
            "note": "Reject thresholds are defined per-test in each metrics JSON under `falsification`.",
            "audit": [
                "Do not re-fit targets; treat baseline targets as frozen inputs.",
                "If a test has no numeric threshold (falsification is null/notes-only), treat it as a baseline/audit constraint and define a separate quantitative test before claiming a 3σ rejection.",
            ],
        },
        "tests": tests,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate condensed/thermo falsification pack JSON (collect baseline metrics JSON).")
    ap.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "output" / "public" / "quantum" / "condensed_falsification_pack.json"),
        help="Output JSON path (default: output/public/quantum/condensed_falsification_pack.json).",
    )
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_relpaths = [
        "output/public/quantum/condensed_silicon_lattice_baseline_metrics.json",
        "output/public/quantum/condensed_silicon_heat_capacity_baseline_metrics.json",
        "output/public/quantum/condensed_silicon_heat_capacity_debye_baseline_metrics.json",
        "output/public/quantum/condensed_silicon_resistivity_temperature_coefficient_baseline_metrics.json",
        "output/public/quantum/condensed_silicon_thermal_expansion_baseline_metrics.json",
        "output/public/quantum/condensed_ofhc_copper_thermal_conductivity_baseline_metrics.json",
        "output/public/quantum/thermo_blackbody_entropy_baseline_metrics.json",
        "output/public/quantum/thermo_blackbody_radiation_baseline_metrics.json",
    ]

    payload = build_pack(root=_ROOT, metrics_relpaths=metrics_relpaths)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {_relpath(out_path)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_condensed_falsification_pack",
                "phase": "7.18.2",
                "inputs": {"metrics_jsons": metrics_relpaths},
                "outputs": {"pack_json": _relpath(out_path)},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

