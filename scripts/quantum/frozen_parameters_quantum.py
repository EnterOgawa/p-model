#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frozen_parameters_quantum.py

Phase 7 / Step 7.18.3:
Part III（量子）で凍結した値（核/Bell/物性/熱）を 1つのJSONへ集約し、
本文（doc/paper/12_part3_quantum.md）から参照できる形に固定する。

出力（固定）:
  - output/quantum/frozen_parameters_quantum.json

方針:
- 既存の確定出力（metrics / pack JSON）から収集し、再計算はしない。
- 値の “正” はこのJSONと、その参照先（inputs.*.sha256）に置く。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _require_path(path: Path, *, hint: str) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing: {path}\n{hint}")


def _walk_collect_key_values(obj: Any, *, key: str, out: List[Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            _walk_collect_key_values(v, key=key, out=out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_collect_key_values(v, key=key, out=out)


def _unique_float(values: List[Any], *, name: str) -> float:
    uniq: Set[float] = set()
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        uniq.add(float(fv))
    if not uniq:
        raise SystemExit(f"[fail] could not extract {name} as a finite float")
    if len(uniq) != 1:
        raise SystemExit(f"[fail] expected a single unique {name} value, got {sorted(uniq)}")
    return next(iter(uniq))


def build_quantum_frozen_parameters(*, root: Path) -> Dict[str, Any]:
    theory_frozen = root / "output" / "theory" / "frozen_parameters.json"
    bell_pack = root / "output" / "quantum" / "bell" / "falsification_pack.json"
    bell_freeze_policy = root / "output" / "quantum" / "bell" / "freeze_policy.json"
    nuclear_pack = root / "output" / "quantum" / "nuclear_binding_energy_frequency_mapping_falsification_pack.json"
    nuclear_theory_diff_metrics = root / "output" / "quantum" / "nuclear_binding_energy_frequency_mapping_theory_diff_metrics.json"
    nuclear_effective_potential_canonical = root / "output" / "quantum" / "nuclear_effective_potential_canonical_metrics.json"
    pairing_metrics = root / "output" / "quantum" / "nuclear_pairing_effect_systematics_metrics.json"
    condensed_pack = root / "output" / "quantum" / "condensed_falsification_pack.json"

    _require_path(theory_frozen, hint="Run: python -B scripts/theory/freeze_parameters.py")
    _require_path(bell_pack, hint="Run: python -B scripts/quantum/bell_primary_products.py --overwrite")
    _require_path(bell_freeze_policy, hint="Run: python -B scripts/quantum/bell_primary_products.py")
    _require_path(nuclear_pack, hint="Run: python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_falsification_pack.py")
    _require_path(nuclear_theory_diff_metrics, hint="Run: python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_theory_diff.py")
    _require_path(nuclear_effective_potential_canonical, hint="Run: python -B scripts/quantum/nuclear_effective_potential_canonical.py (or upstream selector step)")
    _require_path(pairing_metrics, hint="Run: python -B scripts/quantum/nuclear_pairing_effect_systematics.py")
    _require_path(condensed_pack, hint="Run: python -B scripts/quantum/condensed_falsification_pack.py")

    j_theory = _read_json(theory_frozen)
    j_bell = _read_json(bell_pack)
    j_bell_freeze = _read_json(bell_freeze_policy)
    j_nuclear_pack = _read_json(nuclear_pack)
    j_theory_diff = _read_json(nuclear_theory_diff_metrics)
    j_eff = _read_json(nuclear_effective_potential_canonical)
    j_pair = _read_json(pairing_metrics)
    j_cond = _read_json(condensed_pack)

    # Nuclear ν_sat: collect from the (small) theory-diff metrics payload.
    nu_sat_vals: List[Any] = []
    _walk_collect_key_values(j_theory_diff, key="nu_sat_frozen", out=nu_sat_vals)
    nu_sat = _unique_float(nu_sat_vals, name="nu_sat_frozen")

    # Nuclear effective potential canonical selection.
    sel_root = j_eff.get("barrier_tail_channel_split_kq_triplet_barrier_fraction_scan")
    if not isinstance(sel_root, dict):
        raise SystemExit("[fail] nuclear_effective_potential_canonical_metrics.json missing selection root")
    kq = sel_root.get("selected_channel_split_kq_from_7_13_8_4")
    geom = sel_root.get("frozen_singlet_geometry_from_7_13_8_4")
    triplet = sel_root.get("selected")
    if not isinstance(kq, dict) or not isinstance(geom, dict) or not isinstance(triplet, dict):
        raise SystemExit("[fail] canonical selection fields missing in nuclear_effective_potential_canonical_metrics.json")

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "phase": 7,
        "step": "7.18.3",
        "name": "Quantum frozen parameters (Part III): centralized frozen values + audit hashes",
        "policy": {
            "note": "This JSON is the canonical index for Part III frozen values. The primary truth remains the referenced input files pinned by sha256.",
        },
        "inputs": {
            "theory_frozen_parameters_json": {"path": _relpath(theory_frozen), "sha256": _sha256(theory_frozen)},
            "bell_falsification_pack_json": {"path": _relpath(bell_pack), "sha256": _sha256(bell_pack)},
            "bell_freeze_policy_json": {"path": _relpath(bell_freeze_policy), "sha256": _sha256(bell_freeze_policy)},
            "nuclear_binding_energy_falsification_pack_json": {
                "path": _relpath(nuclear_pack),
                "sha256": _sha256(nuclear_pack),
            },
            "nuclear_binding_energy_theory_diff_metrics_json": {
                "path": _relpath(nuclear_theory_diff_metrics),
                "sha256": _sha256(nuclear_theory_diff_metrics),
            },
            "nuclear_effective_potential_canonical_metrics_json": {
                "path": _relpath(nuclear_effective_potential_canonical),
                "sha256": _sha256(nuclear_effective_potential_canonical),
            },
            "nuclear_pairing_effect_systematics_metrics_json": {"path": _relpath(pairing_metrics), "sha256": _sha256(pairing_metrics)},
            "condensed_falsification_pack_json": {"path": _relpath(condensed_pack), "sha256": _sha256(condensed_pack)},
        },
        "frozen": {
            "beta": {
                "beta": j_theory.get("beta"),
                "beta_sigma": j_theory.get("beta_sigma"),
                "gamma_pmodel": j_theory.get("gamma_pmodel"),
                "gamma_pmodel_sigma": j_theory.get("gamma_pmodel_sigma"),
                "beta_source": (j_theory.get("policy") or {}).get("beta_source") if isinstance(j_theory.get("policy"), dict) else None,
                "source": _relpath(theory_frozen),
            },
            "bell": {
                "natural_window": j_bell.get("natural_window"),
                "thresholds": j_bell.get("thresholds"),
                "conditions": j_bell.get("conditions"),
                "blind_freeze_policy": {
                    "policy_id": j_bell_freeze.get("policy_id"),
                    "source": _relpath(bell_freeze_policy),
                },
                "source": _relpath(bell_pack),
            },
            "nuclear": {
                "binding_energy": {
                    "nu_sat_frozen": nu_sat,
                    "thresholds": j_nuclear_pack.get("thresholds"),
                    "conditions": j_nuclear_pack.get("conditions"),
                    "source_pack": _relpath(nuclear_pack),
                    "source_metrics_for_nu_sat": _relpath(nuclear_theory_diff_metrics),
                },
                "effective_potential": {
                    "selected_channel_split_kq_from_7_13_8_4": kq,
                    "frozen_singlet_geometry_from_7_13_8_4": geom,
                    "selected_triplet": triplet,
                    "source": _relpath(nuclear_effective_potential_canonical),
                },
                "pairing_effect_systematics": {
                    "fit": j_pair.get("fit"),
                    "counts": j_pair.get("counts"),
                    "source": _relpath(pairing_metrics),
                },
            },
            "condensed_thermo": {
                "pack_json": _relpath(condensed_pack),
                "tests_n": len(j_cond.get("tests") or []) if isinstance(j_cond.get("tests"), list) else None,
            },
        },
    }
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build Part III (quantum) frozen parameter index JSON.")
    ap.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "output" / "quantum" / "frozen_parameters_quantum.json"),
        help="Output JSON path (default: output/quantum/frozen_parameters_quantum.json).",
    )
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_quantum_frozen_parameters(root=_ROOT)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {_relpath(out_path)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_frozen_parameters",
                "phase": "7.18.3",
                "inputs": payload.get("inputs"),
                "outputs": {"frozen_parameters_quantum_json": _relpath(out_path)},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
