#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_brace_block_from(text: str, start_index: int, start_token: str) -> Optional[str]:
    if start_index < 0:
        return None
    j = start_index + len(start_token)
    depth = 1
    out = []
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
        j += 1
    return None


def _extract_first_nonempty_brace_block(text: str, start_token: str, *, min_len: int = 40) -> Optional[str]:
    search_from = 0
    while True:
        i = text.find(start_token, search_from)
        if i < 0:
            return None
        block = _extract_brace_block_from(text, i, start_token)
        if block is not None and len(block.strip()) >= min_len:
            return block
        search_from = i + len(start_token)


def _maybe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _norm_tex(s: str) -> str:
    # Keep the TeX control sequences but drop whitespace and \! to make regex parsing robust.
    s = s.replace("\\!", "")
    s = re.sub(r"\s+", "", s)
    return s


@dataclass(frozen=True)
class RedshiftConstraint:
    f: float
    sigma_stat: float
    sigma_sys: float
    pericenter_au: Optional[float]
    pericenter_schwarzschild_radii: Optional[float]
    v_peri_kms: Optional[float]


@dataclass(frozen=True)
class PrecessionConstraint:
    f_sp: float
    sigma: float
    delta_phi_arcmin_per_orbit: Optional[float]
    eccentricity: Optional[float]


def _parse_redshift_2018(tex: str) -> Dict[str, Any]:
    abstract = _extract_first_nonempty_brace_block(tex, "\\abstract{")
    if abstract is None:
        return {"ok": False, "reason": "abstract_not_found"}

    norm = _norm_tex(abstract)

    m = re.search(r"f=([0-9.]+)\\pm([0-9.]+)\|_\\mathrm\{stat\}\\pm([0-9.]+)\|_\\mathrm\{sys\}", norm)
    if not m:
        return {"ok": False, "reason": "f_stat_sys_not_found", "abstract_snippet": abstract[:600]}

    f = _maybe_float(m.group(1))
    sigma_stat = _maybe_float(m.group(2))
    sigma_sys = _maybe_float(m.group(3))
    if f is None or sigma_stat is None or sigma_sys is None:
        return {"ok": False, "reason": "f_parse_failed"}

    pericenter_au = None
    pericenter_rs = None
    v_kms = None

    m_au_rs = re.search(r"pericentreat\$([0-9.]+)\\,\\mathrm\{AU\}.*?\{\\approx\}\\,?([0-9.]+)\$Schwarzschildradii", norm)  # noqa: E501
    if m_au_rs:
        pericenter_au = _maybe_float(m_au_rs.group(1))
        pericenter_rs = _maybe_float(m_au_rs.group(2))

    m_v = re.search(r"orbitalspeedof.*?([0-9.]+)\\,\\mathrm\{km/s\}", norm)
    if m_v:
        v_kms = _maybe_float(m_v.group(1))

    c = RedshiftConstraint(
        f=float(f),
        sigma_stat=float(sigma_stat),
        sigma_sys=float(sigma_sys),
        pericenter_au=None if pericenter_au is None else float(pericenter_au),
        pericenter_schwarzschild_radii=None if pericenter_rs is None else float(pericenter_rs),
        v_peri_kms=None if v_kms is None else float(v_kms),
    )
    return {
        "ok": True,
        "abstract_anchor": {"token": "\\abstract{...}", "abstract_snippet": abstract[:800]},
        "constraint": {
            "definition": {"f_newton": 0.0, "f_gr": 1.0},
            "f": c.f,
            "sigma_stat": c.sigma_stat,
            "sigma_sys": c.sigma_sys,
            "sigma_total_quadrature": float((c.sigma_stat**2 + c.sigma_sys**2) ** 0.5),
        },
        "orbit_at_pericenter": {
            "r_peri_au": c.pericenter_au,
            "r_peri_schwarzschild_radii": c.pericenter_schwarzschild_radii,
            "v_peri_kms": c.v_peri_kms,
        },
    }


def _parse_precession_2020(tex: str) -> Dict[str, Any]:
    abstract = _extract_first_nonempty_brace_block(tex, "\\abstract{")
    if abstract is None:
        return {"ok": False, "reason": "abstract_not_found"}

    norm = _norm_tex(abstract)
    m = re.search(r"f_\\mathrm\{SP\}=([0-9.]+)\\pm([0-9.]+)", norm)
    if not m:
        return {"ok": False, "reason": "f_sp_not_found", "abstract_snippet": abstract[:600]}

    f_sp = _maybe_float(m.group(1))
    sigma = _maybe_float(m.group(2))
    if f_sp is None or sigma is None:
        return {"ok": False, "reason": "f_sp_parse_failed"}

    delta_phi_arcmin = None
    m_phi = re.search(r"\\delta\\phi\\approx([0-9.]+)'", norm)
    if m_phi:
        delta_phi_arcmin = _maybe_float(m_phi.group(1))

    e = None
    m_e = re.search(r"e=([0-9.]+)", norm)
    if m_e:
        e = _maybe_float(m_e.group(1))

    c = PrecessionConstraint(
        f_sp=float(f_sp),
        sigma=float(sigma),
        delta_phi_arcmin_per_orbit=None if delta_phi_arcmin is None else float(delta_phi_arcmin),
        eccentricity=None if e is None else float(e),
    )
    return {
        "ok": True,
        "abstract_anchor": {"token": "\\abstract{...}", "abstract_snippet": abstract[:800]},
        "constraint": {
            "definition": {"f_sp_newton": 0.0, "f_sp_gr": 1.0},
            "f_sp": c.f_sp,
            "sigma": c.sigma,
        },
        "orbit": {
            "eccentricity": c.eccentricity,
            "delta_phi_arcmin_per_orbit": c.delta_phi_arcmin_per_orbit,
        },
    }


def _estimate_discrimination_scale(*, r_peri_rs: Optional[float], v_peri_kms: Optional[float]) -> Dict[str, Any]:
    if r_peri_rs is None or v_peri_kms is None:
        return {"ok": False, "reason": "missing_inputs"}

    c_kms = 299_792.458
    beta = float(v_peri_kms) / c_kms
    beta2 = beta * beta
    beta4 = beta2 * beta2

    # ε ≡ GM/(c^2 r) = R_S/(2r)
    eps = 1.0 / (2.0 * float(r_peri_rs))
    eps2 = eps * eps

    # Order-of-magnitude: 1PN redshift amplitude ~ v^2/(2c^2) + GM/(c^2 r)
    amp_1pn = beta2 / 2.0 + eps
    # Order-of-magnitude: 2PN (from exp(-ε) and sqrt(1-v^2/c^2) Taylor terms).
    amp_2pn = eps2 / 2.0 + beta4 / 8.0
    ratio = amp_2pn / amp_1pn if amp_1pn > 0 else None

    # Schwarzschild precession: 1PN ~ O(beta^2), 2PN ~ O(beta^4).
    ratio_sp = beta2  # beta4/beta2

    return {
        "ok": True,
        "inputs": {"r_peri_schwarzschild_radii": float(r_peri_rs), "v_peri_kms": float(v_peri_kms)},
        "dimensionless": {
            "beta": beta,
            "beta2": beta2,
            "beta4": beta4,
            "epsilon": eps,
            "epsilon2": eps2,
        },
        "order_estimates": {
            "redshift": {
                "amp_1pn": amp_1pn,
                "amp_2pn": amp_2pn,
                "ratio_2pn_over_1pn": ratio,
                "sigma_f_required_order": ratio,
            },
            "precession": {
                "ratio_2pn_over_1pn": ratio_sp,
                "sigma_f_sp_required_order": ratio_sp,
            },
        },
        "notes": [
            "This block is an order-of-magnitude estimate. It maps 2PN/1PN scales to an effective |Δf| scale, not a full forward model fit.",
            "If P-model == GR at 1PN, S2 needs σ(f) ~ O(1e-4) to see 2PN-level differences (current is ~O(1e-1)).",
        ],
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()

    default_tex_2018 = root / "data" / "eht" / "sources" / "arxiv_1807.09409" / "GravitationalRedshift_arXiv_20180717.tex"
    default_tex_2020 = root / "data" / "eht" / "sources" / "arxiv_2004.07187" / "s2_precession_afterproof.tex"
    default_outdir = root / "output" / "eht"

    ap = argparse.ArgumentParser(description="Extract GRAVITY S2 constraints (f, f_SP) from arXiv TeX sources.")
    ap.add_argument("--tex-2018", type=str, default=str(default_tex_2018), help="2018 TeX path.")
    ap.add_argument("--tex-2020", type=str, default=str(default_tex_2020), help="2020 TeX path.")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/eht).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_2018_path = Path(args.tex_2018)
    tex_2020_path = Path(args.tex_2020)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "tex_2018": str(tex_2018_path),
            "tex_2020": str(tex_2020_path),
            "sources_manifest": str(root / "output" / "eht" / "gravity_s2_sources_manifest.json"),
        },
        "ok": True,
        "rows": {},
    }

    missing = [str(p) for p in (tex_2018_path, tex_2020_path) if not p.exists()]
    if missing:
        payload["ok"] = False
        payload["reason"] = "missing_tex_sources"
        payload["missing"] = missing
        out_json = outdir / "gravity_s2_constraints.json"
        _write_json(out_json, payload)
        print(f"[warn] missing sources; wrote: {out_json}")
        return 0

    tex_2018 = _read_text(tex_2018_path)
    tex_2020 = _read_text(tex_2020_path)

    row_2018 = _parse_redshift_2018(tex_2018)
    row_2020 = _parse_precession_2020(tex_2020)
    payload["rows"]["redshift_2018"] = row_2018
    payload["rows"]["precession_2020"] = row_2020

    r_peri_rs = None
    v_peri_kms = None
    if row_2018.get("ok") is True:
        orbit = (row_2018.get("orbit_at_pericenter") or {})
        r_peri_rs = orbit.get("r_peri_schwarzschild_radii")
        v_peri_kms = orbit.get("v_peri_kms")

    payload["derived"] = {"discrimination_scale": _estimate_discrimination_scale(r_peri_rs=r_peri_rs, v_peri_kms=v_peri_kms)}
    payload["outputs"] = {"json": str(outdir / "gravity_s2_constraints.json")}

    out_json = outdir / "gravity_s2_constraints.json"
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "gravity_s2_constraints",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {
                    "ok": bool(payload["ok"]),
                    "redshift_ok": bool(row_2018.get("ok") is True),
                    "precession_ok": bool(row_2020.get("ok") is True),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
