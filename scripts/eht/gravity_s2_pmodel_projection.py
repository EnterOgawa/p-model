#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _maybe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _z_rel_gr(*, epsilon: float, beta: float) -> float:
    # A compact "clock-rate" representation consistent with the 1PN term:
    # z ≈ (1/√(1-2ε))(1/√(1-β²)) - 1 ≈ ε + β²/2 + O(ε², β⁴, εβ²).
    return (1.0 / (math.sqrt(1.0 - 2.0 * epsilon) * math.sqrt(1.0 - beta * beta))) - 1.0


def _z_rel_pmodel(*, epsilon: float, beta: float, delta: float) -> float:
    # Part I core (clock model):
    #   dτ/dt = exp(-ε) * sqrt(1-β²)  =>  z = dt/dτ - 1.
    # Optional saturation (extension; δ0):
    #   dτ/dt = exp(-ε) * sqrt((1-β²+δ0)/(1+δ0)).
    return (math.exp(epsilon) * math.sqrt((1.0 + delta) / (1.0 - beta * beta + delta))) - 1.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_in = root / "output" / "eht" / "gravity_s2_constraints.json"
    default_outdir = root / "output" / "eht"

    ap = argparse.ArgumentParser(
        description="Project P-model vs GR differences onto GRAVITY S2 definitions (f, f_SP) as an order-of-magnitude test."
    )
    ap.add_argument("--constraints-json", type=str, default=str(default_in))
    ap.add_argument("--delta", type=float, default=0.0, help="Clock saturation δ0 (extension; default: 0.0=disabled)")
    ap.add_argument("--outdir", type=str, default=str(default_outdir))
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_path = Path(args.constraints_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "gravity_s2_pmodel_projection.json"

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "constraints_json": str(in_path),
            "delta": float(args.delta),
            "model_note": "Uses P-model clock formula exp(-ε) and compares to GR sqrt(1-2ε); precession part is order-estimate only.",
        },
        "ok": True,
        "derived": {},
        "outputs": {"json": str(out_json)},
    }

    if not in_path.exists():
        payload["ok"] = False
        payload["reason"] = "missing_constraints_json"
        _write_json(out_json, payload)
        print(f"[warn] missing input; wrote: {out_json}")
        return 0

    j = _read_json(in_path)
    row_redshift = (j.get("rows") or {}).get("redshift_2018") or {}
    row_prec = (j.get("rows") or {}).get("precession_2020") or {}

    orbit = (row_redshift.get("orbit_at_pericenter") or {}) if isinstance(row_redshift, dict) else {}
    r_peri_rs = _maybe_float(orbit.get("r_peri_schwarzschild_radii"))
    v_peri_kms = _maybe_float(orbit.get("v_peri_kms"))
    if r_peri_rs is None or r_peri_rs <= 0 or v_peri_kms is None or v_peri_kms <= 0:
        payload["ok"] = False
        payload["reason"] = "missing_orbit_pericenter_inputs"
        payload["derived"]["orbit"] = {"r_peri_schwarzschild_radii": r_peri_rs, "v_peri_kms": v_peri_kms}
        _write_json(out_json, payload)
        print(f"[warn] missing orbit inputs; wrote: {out_json}")
        return 0

    c_kms = 299_792.458
    beta = float(v_peri_kms) / c_kms
    epsilon = 1.0 / (2.0 * float(r_peri_rs))  # ε = GM/(c^2 r) = R_S/(2r)

    z_gr = _z_rel_gr(epsilon=epsilon, beta=beta)
    z_p = _z_rel_pmodel(epsilon=epsilon, beta=beta, delta=float(args.delta))
    dz = z_p - z_gr
    f_eff = (z_p / z_gr) if (z_gr != 0.0) else float("nan")
    delta_f = f_eff - 1.0

    # Observed constraints (from extracted abstracts)
    f_obs = _maybe_float(((row_redshift.get("constraint") or {}).get("f")) if isinstance(row_redshift, dict) else None)
    f_sigma_stat = _maybe_float(
        ((row_redshift.get("constraint") or {}).get("sigma_stat")) if isinstance(row_redshift, dict) else None
    )
    f_sigma_sys = _maybe_float(
        ((row_redshift.get("constraint") or {}).get("sigma_sys")) if isinstance(row_redshift, dict) else None
    )
    f_sigma_total = (
        math.sqrt(float(f_sigma_stat) ** 2 + float(f_sigma_sys) ** 2)
        if (f_sigma_stat is not None and f_sigma_sys is not None)
        else None
    )

    f_req_3sigma = abs(delta_f) / 3.0 if math.isfinite(delta_f) and delta_f != 0.0 else None
    gap_f = (
        (float(f_sigma_total) / float(f_req_3sigma))
        if (f_sigma_total is not None and f_req_3sigma is not None and f_req_3sigma > 0)
        else None
    )
    zscore_p_vs_obs = (
        ((float(f_eff) - float(f_obs)) / float(f_sigma_total))
        if (f_obs is not None and f_sigma_total is not None and f_sigma_total > 0 and math.isfinite(f_eff))
        else None
    )

    # Precession: we do not implement a full dynamical prediction here.
    # Provide a 2PN/1PN scale estimate (order-of-magnitude) at pericenter.
    beta2 = beta * beta
    f_sp_obs = _maybe_float(((row_prec.get("constraint") or {}).get("f_sp")) if isinstance(row_prec, dict) else None)
    f_sp_sigma = _maybe_float(((row_prec.get("constraint") or {}).get("sigma")) if isinstance(row_prec, dict) else None)

    # If P-model matches GR at 1PN (as required by Solar-System precession), the first discriminating differences
    # are expected at O(β²) relative to the 1PN precession term.
    delta_f_sp_order = beta2
    f_sp_req_3sigma_order = abs(delta_f_sp_order) / 3.0 if delta_f_sp_order > 0 else None
    gap_f_sp = (
        (float(f_sp_sigma) / float(f_sp_req_3sigma_order))
        if (f_sp_sigma is not None and f_sp_req_3sigma_order is not None and f_sp_req_3sigma_order > 0)
        else None
    )

    payload["derived"] = {
        "pericenter": {
            "r_peri_schwarzschild_radii": float(r_peri_rs),
            "v_peri_kms": float(v_peri_kms),
            "beta": beta,
            "beta2": beta2,
            "epsilon": epsilon,
            "epsilon2": epsilon * epsilon,
        },
        "redshift_f": {
            "z_rel_gr": float(z_gr),
            "z_rel_pmodel": float(z_p),
            "delta_z": float(dz),
            "f_effective_pmodel_over_gr": float(f_eff),
            "delta_f": float(delta_f),
            "sigma_f_required_3sigma": f_req_3sigma,
            "obs": {
                "f": f_obs,
                "sigma_stat": f_sigma_stat,
                "sigma_sys": f_sigma_sys,
                "sigma_total_quadrature": f_sigma_total,
            },
            "gap_sigma_now_over_required": gap_f,
            "zscore_pmodel_vs_obs": zscore_p_vs_obs,
        },
        "precession_f_sp": {
            "f_sp_pmodel_assumed_1pn": 1.0,
            "delta_f_sp_order_estimate_2pn_over_1pn": float(delta_f_sp_order),
            "sigma_f_sp_required_3sigma_order": f_sp_req_3sigma_order,
            "obs": {"f_sp": f_sp_obs, "sigma": f_sp_sigma},
            "gap_sigma_now_over_required_order": gap_f_sp,
        },
        "notes": [
            "redshift_f uses a pericenter-only scale estimate based on the P-model clock formula exp(-ε) and the GR factor √(1-2ε). It is not a full forward model of the observed time series.",
            "precession_f_sp is an order-of-magnitude estimate only (assumes P-model matches GR at 1PN and differences enter at ~2PN scale).",
        ],
    }

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "gravity_s2_pmodel_projection",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {"ok": bool(payload.get("ok")), "delta_f": float(delta_f)},
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
