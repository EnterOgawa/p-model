from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_nist_codata_constants(*, root: Path) -> dict[str, dict[str, object]]:
    src_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    extracted = src_dir / "extracted_values.json"
    if not extracted.exists():
        raise SystemExit(
            "[fail] missing extracted CODATA constants.\n"
            "Run:\n"
            "  python -B scripts/quantum/fetch_nuclear_binding_sources.py\n"
            f"Expected: {extracted}"
        )
    payload = _load_json(extracted)
    consts = payload.get("constants")
    if not isinstance(consts, dict):
        raise SystemExit(f"[fail] invalid extracted_values.json: constants is not a dict: {extracted}")
    return {k: v for k, v in consts.items() if isinstance(v, dict)}


def _load_np_scattering_sets(*, root: Path) -> dict[int, dict[str, object]]:
    extracted = root / "data" / "quantum" / "sources" / "np_scattering_low_energy_arxiv_0704_1024v1_extracted.json"
    if not extracted.exists():
        raise SystemExit(
            "[fail] missing extracted np scattering values.\n"
            "Run:\n"
            "  python -B scripts/quantum/fetch_nuclear_np_scattering_sources.py\n"
            f"Expected: {extracted}"
        )
    j = _load_json(extracted)
    sets = j.get("parameter_sets")
    if not isinstance(sets, list) or not sets:
        raise SystemExit(f"[fail] invalid extracted file: parameter_sets missing/empty: {extracted}")
    out: dict[int, dict[str, object]] = {}
    for s in sets:
        if not isinstance(s, dict):
            continue
        try:
            eq = int(s.get("eq_label"))
        except Exception:
            continue
        params = s.get("params")
        if isinstance(params, dict):
            out[eq] = params
    return out


def _load_pion_range_scale(*, root: Path) -> dict[str, object]:
    """
    Load λπ (Compton wavelength) from the cached PDG-based hadron baseline metrics
    (Phase 7 / Step 7.13.1 output).
    """
    metrics = root / "output" / "quantum" / "qcd_hadron_masses_baseline_metrics.json"
    if not metrics.exists():
        raise SystemExit(
            "[fail] missing hadron baseline metrics (needed for λπ range constraint).\n"
            "Run:\n"
            "  python -B scripts/quantum/qcd_hadron_masses_baseline.py\n"
            f"Expected: {metrics}"
        )
    j = _load_json(metrics)
    rows = j.get("rows")
    if not isinstance(rows, list):
        raise SystemExit(f"[fail] invalid hadron baseline metrics: rows missing: {metrics}")

    lam_pm = None
    lam_0 = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("label") == "π±":
            lam_pm = r.get("compton_lambda_fm")
        if r.get("label") == "π0":
            lam_0 = r.get("compton_lambda_fm")
    if not (isinstance(lam_pm, (int, float)) and math.isfinite(float(lam_pm))):
        raise SystemExit(f"[fail] π± compton_lambda_fm missing/invalid in: {metrics}")
    if not (isinstance(lam_0, (int, float)) and math.isfinite(float(lam_0))):
        raise SystemExit(f"[fail] π0 compton_lambda_fm missing/invalid in: {metrics}")
    pdg_src = j.get("pdg_source") if isinstance(j.get("pdg_source"), dict) else {}
    return {
        "lambda_pi_pm_fm": float(lam_pm),
        "lambda_pi0_fm": float(lam_0),
        "metrics_path": str(metrics),
        "metrics_sha256": _sha256(metrics),
        "pdg_source": pdg_src,
    }


def _get_value(params: dict[str, object], key: str) -> float:
    obj = params.get(key)
    if isinstance(obj, dict) and "value" in obj:
        return float(obj["value"])
    raise KeyError(key)


def _cot(x: float) -> float:
    if abs(x) < 1e-10:
        return (1.0 / x) - (x / 3.0)
    return math.cos(x) / math.sin(x)


def _coth(x: float) -> float:
    if abs(x) < 1e-10:
        return (1.0 / x) + (x / 3.0)
    return math.cosh(x) / math.sinh(x)


def _region_q(*, e_mev: float, vdepth_mev: float, mu_mev: float, hbarc_mev_fm: float) -> tuple[str, float]:
    """
    For potential V(r) = -Vdepth (attractive), the radial equation is:
      u'' + q^2 u = 0,  q^2 = 2μ(E + Vdepth)/(ħc)^2.

    Returns:
      ("osc", q)   for E+Vdepth > 0 (sin/cos),
      ("evan", a)  for E+Vdepth < 0 (sinh/cosh with a = sqrt(-q^2)),
      ("zero", 0)  for E+Vdepth = 0.
    """
    if not (
        mu_mev > 0
        and hbarc_mev_fm > 0
        and math.isfinite(e_mev)
        and math.isfinite(vdepth_mev)
        and vdepth_mev >= 0
    ):
        return ("zero", float("nan"))
    s = e_mev + vdepth_mev
    if s > 0:
        return ("osc", math.sqrt(2.0 * mu_mev * s) / hbarc_mev_fm)
    if s < 0:
        return ("evan", math.sqrt(2.0 * mu_mev * (-s)) / hbarc_mev_fm)
    return ("zero", 0.0)


def _region_q_potential(
    *, e_mev: float, potential_mev: float, mu_mev: float, hbarc_mev_fm: float
) -> tuple[str, float]:
    """
    For signed potential energy V(r)=potential_mev (repulsive positive, attractive negative),
    the radial equation is:
      u'' + q^2 u = 0,  q^2 = 2μ(E - V)/(ħc)^2.

    Returns:
      ("osc", q)   for E - V > 0 (sin/cos),
      ("evan", a)  for E - V < 0 (sinh/cosh with a = sqrt(-q^2)),
      ("zero", 0)  for E - V = 0.
    """
    if not (
        mu_mev > 0
        and hbarc_mev_fm > 0
        and math.isfinite(e_mev)
        and math.isfinite(potential_mev)
    ):
        return ("zero", float("nan"))
    s = e_mev - potential_mev
    if s > 0:
        return ("osc", math.sqrt(2.0 * mu_mev * s) / hbarc_mev_fm)
    if s < 0:
        return ("evan", math.sqrt(2.0 * mu_mev * (-s)) / hbarc_mev_fm)
    return ("zero", 0.0)


def _propagate_y(*, y_in: float, mode: str, q: float, l_fm: float) -> float:
    """
    Propagate log-derivative y=u'/u across a constant-potential segment of length L.
    """
    if not (math.isfinite(l_fm) and l_fm >= 0):
        return float("nan")
    if l_fm == 0.0:
        return float(y_in)

    if mode == "zero" or q == 0.0:
        if math.isinf(y_in):
            return float(1.0 / l_fm)
        denom = 1.0 + (y_in * l_fm)
        if denom == 0.0:
            return float("nan")
        return float(y_in / denom)

    if mode == "osc":
        ql = q * l_fm
        if math.isinf(y_in):
            return float(q * _cot(ql))
        s = math.sin(ql)
        c = math.cos(ql)
        denom = c + (y_in / q) * s
        if abs(denom) < 1e-18:
            return float("nan")
        num = (-q * s) + (y_in * c)
        return float(num / denom)

    if mode == "evan":
        al = q * l_fm
        if math.isinf(y_in):
            return float(q * _coth(al))
        s = math.sinh(al)
        c = math.cosh(al)
        denom = c + (y_in / q) * s
        if abs(denom) < 1e-18:
            return float("nan")
        num = (q * s) + (y_in * c)
        return float(num / denom)

    return float("nan")


def _y_at_r2(
    *,
    e_mev: float,
    r1_fm: float,
    r2_fm: float,
    v1_mev: float,
    v2_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> float:
    if not (0 < r1_fm < r2_fm and v1_mev >= 0 and v2_mev >= 0):
        return float("nan")
    y: float = float("inf")  # regular at r=0

    m1, q1 = _region_q(e_mev=e_mev, vdepth_mev=v1_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    y = _propagate_y(y_in=y, mode=m1, q=q1, l_fm=r1_fm)

    m2, q2 = _region_q(e_mev=e_mev, vdepth_mev=v2_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    y = _propagate_y(y_in=y, mode=m2, q=q2, l_fm=(r2_fm - r1_fm))
    return float(y)


def _y_at_r2_signed_two_range(
    *,
    e_mev: float,
    r1_fm: float,
    r2_fm: float,
    v1_mev: float,
    v2_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> float:
    """
    y=u'/u at r=R2 for a signed 2-step potential, keeping the same convention:
      V=-V1 (0<=r<R1),
      V=-V2 (R1<=r<R2),
      V=0   (r>R2),
    where V2 is allowed to be negative (outer repulsive barrier when V2<0).
    """
    if not (
        0 < r1_fm < r2_fm
        and math.isfinite(e_mev)
        and math.isfinite(v1_mev)
        and math.isfinite(v2_mev)
        and mu_mev > 0
        and hbarc_mev_fm > 0
    ):
        return float("nan")
    y: float = float("inf")  # regular at r=0

    m1, q1 = _region_q_potential(e_mev=e_mev, potential_mev=-v1_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    y = _propagate_y(y_in=y, mode=m1, q=q1, l_fm=r1_fm)

    m2, q2 = _region_q_potential(e_mev=e_mev, potential_mev=-v2_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    y = _propagate_y(y_in=y, mode=m2, q=q2, l_fm=(r2_fm - r1_fm))
    return float(y)


def _y_at_r_end_segments(
    *, e_mev: float, segments: list[tuple[float, float]], mu_mev: float, hbarc_mev_fm: float
) -> float:
    """
    Generic log-derivative propagation for a piecewise-constant signed potential.

    segments: list of (length_fm, potential_mev), where potential_mev is signed
      (repulsive positive, attractive negative). The exterior region beyond the last segment
      is assumed to be free (V=0).
    """
    if not (math.isfinite(e_mev) and mu_mev > 0 and hbarc_mev_fm > 0):
        return float("nan")
    y: float = float("inf")
    for l_fm, pot_mev in segments:
        if not (math.isfinite(l_fm) and l_fm >= 0 and math.isfinite(pot_mev)):
            return float("nan")
        if l_fm == 0.0:
            continue
        mode, q = _region_q_potential(e_mev=e_mev, potential_mev=float(pot_mev), mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        y = _propagate_y(y_in=y, mode=mode, q=q, l_fm=float(l_fm))
        if not math.isfinite(y):
            return float("nan")
    return float(y)


def _y_at_r2_repulsive_core_two_range(
    *,
    e_mev: float,
    rc_fm: float,
    r1_fm: float,
    r2_fm: float,
    vc_mev: float,
    v1_mev: float,
    v2_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> float:
    """
    y=u'/u at r=R2 for the repulsive-core + 2-range attractive well:
      V=+Vc (0<=r<Rc),
      V=-V1 (Rc<=r<R1),
      V=-V2 (R1<=r<R2),
      V=0   (r>R2).
    """
    if not (0 <= rc_fm < r1_fm < r2_fm and vc_mev >= 0 and v1_mev >= 0 and v2_mev >= 0):
        return float("nan")
    y: float = float("inf")  # regular at r=0

    # Repulsive core: V=+Vc.
    if rc_fm > 0.0:
        m0, q0 = _region_q_potential(e_mev=e_mev, potential_mev=vc_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        y = _propagate_y(y_in=y, mode=m0, q=q0, l_fm=rc_fm)

    # Inner attractive well: V=-V1.
    m1, q1 = _region_q_potential(e_mev=e_mev, potential_mev=-v1_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    y = _propagate_y(y_in=y, mode=m1, q=q1, l_fm=(r1_fm - rc_fm))

    # Outer attractive well: V=-V2.
    m2, q2 = _region_q_potential(e_mev=e_mev, potential_mev=-v2_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    y = _propagate_y(y_in=y, mode=m2, q=q2, l_fm=(r2_fm - r1_fm))
    return float(y)


def _solve_v1_from_b(
    *,
    b_mev: float,
    r1_fm: float,
    r2_fm: float,
    v2_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
    v1_max_mev: float = 500.0,
) -> dict[str, float]:
    """
    Solve V1 (inner depth) from the deuteron bound-state condition E=-B:
      y(R2; E=-B) = -kappa, where kappa = sqrt(2 μ B)/(ħc).

    We scan V1>=max(V2,B) and take the first sign-change bracket (minimal V1),
    rejecting very deep roots with q1*R1 > pi to avoid excited-branch jumps.
    """
    if not (b_mev > 0 and 0 < r1_fm < r2_fm and v2_mev >= 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid inputs")

    kappa = math.sqrt(2.0 * mu_mev * b_mev) / hbarc_mev_fm
    v1_min = max(v2_mev, b_mev + 1e-6)

    def f(v1: float) -> float:
        y = _y_at_r2(
            e_mev=-b_mev,
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            v1_mev=v1,
            v2_mev=v2_mev,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        return y + kappa

    n_scan = 120
    v_grid = [v1_min + (v1_max_mev - v1_min) * i / (n_scan - 1) for i in range(n_scan)]
    vals: list[tuple[float, float]] = []
    for v in v_grid:
        fv = f(v)
        if not math.isfinite(fv):
            continue
        vals.append((v, fv))
    if len(vals) < 6:
        raise ValueError("no valid bound-state samples for V1 scan")

    bracket: tuple[float, float] | None = None
    for (v0, f0), (v1, f1) in zip(vals[:-1], vals[1:]):
        if f0 == 0:
            bracket = (v0, v0)
            break
        if (f0 > 0) != (f1 > 0):
            mid = 0.5 * (v0 + v1)
            m1, q1 = _region_q(e_mev=-b_mev, vdepth_mev=mid, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
            x1 = q1 * r1_fm if (m1 == "osc" and math.isfinite(q1)) else 0.0
            if m1 == "osc" and x1 > (math.pi * 0.999):
                continue
            bracket = (v0, v1)
            break
    if bracket is None:
        raise ValueError("no sign-change bracket for bound-state V1 within scan range")

    lo, hi = bracket
    if lo == hi:
        v1_star = lo
    else:
        f_lo = f(lo)
        f_hi = f(hi)
        if not ((f_lo > 0) != (f_hi > 0)):
            raise ValueError("invalid V1 bracket")
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if f_mid == 0 or (hi - lo) < 1e-10:
                lo = mid
                break
            if (f_mid > 0) == (f_lo > 0):
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
                f_hi = f_mid
        v1_star = 0.5 * (lo + hi)

    return {"V1_MeV": float(v1_star), "kappa_fm1": float(kappa)}


def _solve_v1_from_b_segments(
    *,
    b_mev: float,
    first_len_fm: float,
    segments_after_first: list[tuple[float, float]],
    mu_mev: float,
    hbarc_mev_fm: float,
    v1_max_mev: float = 800.0,
) -> dict[str, float]:
    """
    Solve V1 (inner depth) from the deuteron bound-state condition E=-B for a
    piecewise-constant signed potential:

      segments = [(L1, -V1), (L2, V2_pot), (L3, V3_pot), ...],  with L1=first_len_fm,
      and the exterior region beyond the last segment is free (V=0).

    Condition:
      y(r_end; E=-B) = -kappa,  kappa = sqrt(2 μ B)/(ħc).

    We scan V1 and take the first bracket (minimal V1), rejecting deep roots with
    q1*L1 > pi to avoid excited-branch jumps.
    """
    if not (b_mev > 0 and math.isfinite(first_len_fm) and first_len_fm > 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid inputs")
    if not segments_after_first:
        raise ValueError("need segments_after_first")
    for l_fm, pot_mev in segments_after_first:
        if not (math.isfinite(l_fm) and l_fm >= 0 and math.isfinite(pot_mev)):
            raise ValueError("invalid segments_after_first")

    kappa = math.sqrt(2.0 * mu_mev * b_mev) / hbarc_mev_fm

    # Minimal bound: V1 must exceed B so that the inner region can support an oscillatory component.
    v1_min = b_mev + 1e-6

    def f(v1_mev: float) -> float:
        if not (math.isfinite(v1_mev) and v1_mev > b_mev):
            return float("nan")
        # Reject excited-branch roots: ensure the first segment stays on the ground branch.
        mode1, q1 = _region_q_potential(
            e_mev=-b_mev, potential_mev=-float(v1_mev), mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )
        if not (mode1 == "osc" and math.isfinite(q1) and (q1 * first_len_fm) < (math.pi * 0.999)):
            return float("nan")

        segs = [(float(first_len_fm), -float(v1_mev))] + [(float(l), float(p)) for l, p in segments_after_first]
        y = _y_at_r_end_segments(e_mev=-b_mev, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        if not math.isfinite(y):
            return float("nan")
        return float(y + kappa)

    n_scan = 160
    v_grid = [v1_min + (v1_max_mev - v1_min) * i / (n_scan - 1) for i in range(n_scan)]
    vals: list[tuple[float, float]] = []
    for v in v_grid:
        fv = f(v)
        if not math.isfinite(fv):
            continue
        vals.append((v, fv))
    if len(vals) < 6:
        raise ValueError("no valid bound-state samples for V1 scan (segments)")

    bracket: tuple[float, float] | None = None
    for (v0, f0), (v1, f1) in zip(vals[:-1], vals[1:]):
        if f0 == 0.0:
            bracket = (v0, v0)
            break
        if (f0 > 0) != (f1 > 0):
            bracket = (v0, v1)
            break
    if bracket is None:
        raise ValueError("no sign-change bracket for bound-state V1 within scan range (segments)")

    lo, hi = bracket
    if lo == hi:
        v1_star = lo
    else:
        f_lo = f(lo)
        f_hi = f(hi)
        if not ((f_lo > 0) != (f_hi > 0)):
            raise ValueError("invalid V1 bracket (segments)")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if not math.isfinite(f_mid):
                mid_a = (2.0 * lo + hi) / 3.0
                mid_b = (lo + 2.0 * hi) / 3.0
                fa = f(mid_a)
                fb = f(mid_b)
                if math.isfinite(fa):
                    mid, f_mid = mid_a, fa
                elif math.isfinite(fb):
                    mid, f_mid = mid_b, fb
                else:
                    break
            if f_mid == 0.0 or (hi - lo) < 1e-10:
                lo = mid
                break
            if (f_mid > 0) == (f_lo > 0):
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
                f_hi = f_mid
        v1_star = 0.5 * (lo + hi)

    return {"V1_MeV": float(v1_star), "kappa_fm1": float(kappa)}


def _solve_v1_from_b_repulsive_core_two_range(
    *,
    b_mev: float,
    rc_fm: float,
    r1_fm: float,
    r2_fm: float,
    vc_mev: float,
    v2_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
    v1_max_mev: float = 1200.0,
) -> dict[str, float]:
    """
    Solve V1 (inner depth) from the deuteron bound-state condition E=-B:
      y(R2; E=-B) = -kappa, where kappa = sqrt(2 μ B)/(ħc),
    for the repulsive-core + two-range ansatz.
    """
    if not (
        b_mev > 0
        and 0.0 <= rc_fm < r1_fm < r2_fm
        and vc_mev >= 0
        and v2_mev >= 0
        and mu_mev > 0
        and hbarc_mev_fm > 0
    ):
        raise ValueError("invalid inputs")

    kappa = math.sqrt(2.0 * mu_mev * b_mev) / hbarc_mev_fm
    v1_min = max(v2_mev, b_mev + 1e-6)

    def f(v1: float) -> float:
        y = _y_at_r2_repulsive_core_two_range(
            e_mev=-b_mev,
            rc_fm=rc_fm,
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            vc_mev=vc_mev,
            v1_mev=v1,
            v2_mev=v2_mev,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        return y + kappa

    n_scan = 180
    v_grid = [v1_min + (v1_max_mev - v1_min) * i / (n_scan - 1) for i in range(n_scan)]
    vals: list[tuple[float, float]] = []
    for v in v_grid:
        # Avoid excited-branch jumps by staying in the ground-like interval.
        if v > b_mev:
            q = math.sqrt(2.0 * mu_mev * (v - b_mev)) / hbarc_mev_fm
            if math.isfinite(q) and q * max(1e-6, (r1_fm - rc_fm)) > (math.pi * 0.999):
                continue
        fv = f(v)
        if not math.isfinite(fv):
            continue
        vals.append((v, fv))
    if len(vals) < 8:
        raise ValueError("no valid bound-state samples for V1 scan")

    bracket: tuple[float, float] | None = None
    for (v0, f0), (v1, f1) in zip(vals[:-1], vals[1:]):
        if f0 == 0.0:
            bracket = (v0, v0)
            break
        if (f0 > 0) != (f1 > 0):
            bracket = (v0, v1)
            break
    if bracket is None:
        raise ValueError("no sign-change bracket for bound-state V1 within scan range")

    lo, hi = bracket
    if lo == hi:
        v1_star = lo
    else:
        f_lo = f(lo)
        f_hi = f(hi)
        if not ((f_lo > 0) != (f_hi > 0)):
            raise ValueError("invalid V1 bracket")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if f_mid == 0.0 or (hi - lo) < 1e-10:
                lo = mid
                break
            if (f_mid > 0) == (f_lo > 0):
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
                f_hi = f_mid
        v1_star = 0.5 * (lo + hi)

    return {"V1_MeV": float(v1_star), "kappa_fm1": float(kappa)}


def _scattering_length_exact(
    *, r1_fm: float, r2_fm: float, v1_mev: float, v2_mev: float, mu_mev: float, hbarc_mev_fm: float
) -> float:
    y = _y_at_r2(
        e_mev=0.0,
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        v1_mev=v1_mev,
        v2_mev=v2_mev,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    if not math.isfinite(y) or abs(y) < 1e-18:
        return float("nan")
    return float(r2_fm - (1.0 / y))


def _scattering_length_exact_signed_two_range(
    *, r1_fm: float, r2_fm: float, v1_mev: float, v2_mev: float, mu_mev: float, hbarc_mev_fm: float
) -> float:
    y = _y_at_r2_signed_two_range(
        e_mev=0.0,
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        v1_mev=v1_mev,
        v2_mev=v2_mev,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    if not math.isfinite(y) or abs(y) < 1e-18:
        return float("nan")
    return float(r2_fm - (1.0 / y))


def _scattering_length_exact_segments(
    *, r_end_fm: float, segments: list[tuple[float, float]], mu_mev: float, hbarc_mev_fm: float
) -> float:
    y = _y_at_r_end_segments(e_mev=0.0, segments=segments, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    if not math.isfinite(y) or abs(y) < 1e-18:
        return float("nan")
    return float(r_end_fm - (1.0 / y))


def _scattering_length_exact_repulsive_core_two_range(
    *,
    rc_fm: float,
    r1_fm: float,
    r2_fm: float,
    vc_mev: float,
    v1_mev: float,
    v2_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> float:
    y = _y_at_r2_repulsive_core_two_range(
        e_mev=0.0,
        rc_fm=rc_fm,
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        vc_mev=vc_mev,
        v1_mev=v1_mev,
        v2_mev=v2_mev,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    if not math.isfinite(y) or abs(y) < 1e-18:
        return float("nan")
    return float(r2_fm - (1.0 / y))


def _phase_shift(
    *, k_fm1: float, r1_fm: float, r2_fm: float, v1_mev: float, v2_mev: float, mu_mev: float, hbarc_mev_fm: float
) -> float:
    if k_fm1 == 0.0:
        return 0.0
    if not (k_fm1 > 0 and 0 < r1_fm < r2_fm and v1_mev >= 0 and v2_mev >= 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        return float("nan")
    e_mev = (hbarc_mev_fm**2) * (k_fm1**2) / (2.0 * mu_mev)
    y = _y_at_r2(
        e_mev=e_mev,
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        v1_mev=v1_mev,
        v2_mev=v2_mev,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    if not math.isfinite(y):
        return float("nan")
    if abs(y) < 1e-18:
        delta = (math.pi / 2.0) - (k_fm1 * r2_fm)
    else:
        delta = math.atan(k_fm1 / y) - (k_fm1 * r2_fm)
    while delta > math.pi / 2.0:
        delta -= math.pi
    while delta < -math.pi / 2.0:
        delta += math.pi
    return float(delta)


def _phase_shift_signed_two_range(
    *, k_fm1: float, r1_fm: float, r2_fm: float, v1_mev: float, v2_mev: float, mu_mev: float, hbarc_mev_fm: float
) -> float:
    if k_fm1 == 0.0:
        return 0.0
    if not (
        k_fm1 > 0
        and 0 < r1_fm < r2_fm
        and math.isfinite(v1_mev)
        and math.isfinite(v2_mev)
        and mu_mev > 0
        and hbarc_mev_fm > 0
    ):
        return float("nan")
    e_mev = (hbarc_mev_fm**2) * (k_fm1**2) / (2.0 * mu_mev)
    y = _y_at_r2_signed_two_range(
        e_mev=e_mev,
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        v1_mev=v1_mev,
        v2_mev=v2_mev,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    if not math.isfinite(y):
        return float("nan")
    if abs(y) < 1e-18:
        delta = (math.pi / 2.0) - (k_fm1 * r2_fm)
    else:
        delta = math.atan(k_fm1 / y) - (k_fm1 * r2_fm)
    while delta > math.pi / 2.0:
        delta -= math.pi
    while delta < -math.pi / 2.0:
        delta += math.pi
    return float(delta)


def _phase_shift_segments(
    *,
    k_fm1: float,
    r_end_fm: float,
    segments: list[tuple[float, float]],
    mu_mev: float,
    hbarc_mev_fm: float,
) -> float:
    if k_fm1 == 0.0:
        return 0.0
    if not (k_fm1 > 0 and math.isfinite(r_end_fm) and r_end_fm > 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        return float("nan")
    e_mev = (hbarc_mev_fm**2) * (k_fm1**2) / (2.0 * mu_mev)
    y = _y_at_r_end_segments(e_mev=e_mev, segments=segments, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    if not math.isfinite(y):
        return float("nan")
    if abs(y) < 1e-18:
        delta = (math.pi / 2.0) - (k_fm1 * r_end_fm)
    else:
        delta = math.atan(k_fm1 / y) - (k_fm1 * r_end_fm)
    while delta > math.pi / 2.0:
        delta -= math.pi
    while delta < -math.pi / 2.0:
        delta += math.pi
    return float(delta)


def _phase_shift_repulsive_core_two_range(
    *,
    k_fm1: float,
    rc_fm: float,
    r1_fm: float,
    r2_fm: float,
    vc_mev: float,
    v1_mev: float,
    v2_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> float:
    if k_fm1 == 0.0:
        return 0.0
    if not (
        k_fm1 > 0
        and 0 <= rc_fm < r1_fm < r2_fm
        and vc_mev >= 0
        and v1_mev >= 0
        and v2_mev >= 0
        and mu_mev > 0
        and hbarc_mev_fm > 0
    ):
        return float("nan")
    e_mev = (hbarc_mev_fm**2) * (k_fm1**2) / (2.0 * mu_mev)
    y = _y_at_r2_repulsive_core_two_range(
        e_mev=e_mev,
        rc_fm=rc_fm,
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        vc_mev=vc_mev,
        v1_mev=v1_mev,
        v2_mev=v2_mev,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    if not math.isfinite(y):
        return float("nan")
    if abs(y) < 1e-18:
        delta = (math.pi / 2.0) - (k_fm1 * r2_fm)
    else:
        delta = math.atan(k_fm1 / y) - (k_fm1 * r2_fm)
    while delta > math.pi / 2.0:
        delta -= math.pi
    while delta < -math.pi / 2.0:
        delta += math.pi
    return float(delta)


def _solve_3x3(a: list[list[float]], b: list[float]) -> list[float]:
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-18:
            raise ValueError("singular matrix")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        for j in range(col, 4):
            m[col][j] /= piv
        for r in range(col + 1, 3):
            fac = m[r][col]
            for j in range(col, 4):
                m[r][j] -= fac * m[col][j]
    x = [0.0, 0.0, 0.0]
    for r in reversed(range(3)):
        x[r] = m[r][3] - sum(m[r][c] * x[c] for c in range(r + 1, 3))
    return x


def _fit_kcot_ere(
    *, r1_fm: float, r2_fm: float, v1_mev: float, v2_mev: float, mu_mev: float, hbarc_mev_fm: float
) -> dict[str, object]:
    k_grid = [0.002 * i for i in range(1, 31)]  # 0.002..0.060 fm^-1
    k2s: list[float] = []
    k4s: list[float] = []
    ys: list[float] = []
    points: list[dict[str, float]] = []
    for k in k_grid:
        delta = _phase_shift(
            k_fm1=k,
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            v1_mev=v1_mev,
            v2_mev=v2_mev,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        if not math.isfinite(delta):
            continue
        t = math.tan(delta)
        if abs(t) < 1e-15:
            continue
        y = k / t
        k2 = k * k
        if not (math.isfinite(y) and math.isfinite(k2)):
            continue
        k2s.append(k2)
        k4s.append(k2 * k2)
        ys.append(y)
        points.append({"k_fm1": float(k), "kcot_fm1": float(y), "delta_rad": float(delta)})
    if len(ys) < 10:
        raise ValueError("insufficient points for ERE fit")
    s00 = float(len(ys))
    s01 = float(sum(k2s))
    s02 = float(sum(k4s))
    s11 = float(sum(x * x for x in k2s))
    s12 = float(sum(x * z for x, z in zip(k2s, k4s)))
    s22 = float(sum(z * z for z in k4s))
    b0 = float(sum(ys))
    b1 = float(sum(y * x for y, x in zip(ys, k2s)))
    b2 = float(sum(y * z for y, z in zip(ys, k4s)))
    c0, c2, c4 = _solve_3x3([[s00, s01, s02], [s01, s11, s12], [s02, s12, s22]], [b0, b1, b2])
    a_out = -1.0 / c0 if c0 != 0 else float("nan")
    r_eff = 2.0 * c2
    v2_out = c4
    rms = math.sqrt(
        sum((y - (c0 + c2 * x + c4 * (x * x))) ** 2 for y, x in zip(ys, k2s)) / float(len(ys))
    )
    return {
        "a_fm": float(a_out),
        "r_eff_fm": float(r_eff),
        "v2_fm3": float(v2_out),
        "coeffs": {"c0_fm1": float(c0), "c2_fm": float(c2), "c4_fm3": float(c4)},
        "fit_rms_fm1": float(rms),
        "points": points,
    }


def _fit_kcot_ere_signed_two_range(
    *, r1_fm: float, r2_fm: float, v1_mev: float, v2_mev: float, mu_mev: float, hbarc_mev_fm: float
) -> dict[str, object]:
    k_grid = [0.002 * i for i in range(1, 31)]  # 0.002..0.060 fm^-1
    k2s: list[float] = []
    k4s: list[float] = []
    ys: list[float] = []
    points: list[dict[str, float]] = []
    for k in k_grid:
        delta = _phase_shift_signed_two_range(
            k_fm1=k,
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            v1_mev=v1_mev,
            v2_mev=v2_mev,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        if not math.isfinite(delta):
            continue
        t = math.tan(delta)
        if abs(t) < 1e-15:
            continue
        y = k / t
        k2 = k * k
        if not (math.isfinite(y) and math.isfinite(k2)):
            continue
        k2s.append(k2)
        k4s.append(k2 * k2)
        ys.append(y)
        points.append({"k_fm1": float(k), "kcot_fm1": float(y), "delta_rad": float(delta)})
    if len(ys) < 10:
        raise ValueError("insufficient points for ERE fit")
    s00 = float(len(ys))
    s01 = float(sum(k2s))
    s02 = float(sum(k4s))
    s11 = float(sum(x * x for x in k2s))
    s12 = float(sum(x * z for x, z in zip(k2s, k4s)))
    s22 = float(sum(z * z for z in k4s))
    b0 = float(sum(ys))
    b1 = float(sum(y * x for y, x in zip(ys, k2s)))
    b2 = float(sum(y * z for y, z in zip(ys, k4s)))
    c0, c2, c4 = _solve_3x3([[s00, s01, s02], [s01, s11, s12], [s02, s12, s22]], [b0, b1, b2])
    a_out = -1.0 / c0 if c0 != 0 else float("nan")
    r_eff = 2.0 * c2
    v2_out = c4
    rms = math.sqrt(
        sum((y - (c0 + c2 * x + c4 * (x * x))) ** 2 for y, x in zip(ys, k2s)) / float(len(ys))
    )
    return {
        "a_fm": float(a_out),
        "r_eff_fm": float(r_eff),
        "v2_fm3": float(v2_out),
        "coeffs": {"c0_fm1": float(c0), "c2_fm": float(c2), "c4_fm3": float(c4)},
        "fit_rms_fm1": float(rms),
        "points": points,
    }


def _fit_kcot_ere_segments(
    *,
    r_end_fm: float,
    segments: list[tuple[float, float]],
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    k_grid = [0.002 * i for i in range(1, 31)]  # 0.002..0.060 fm^-1
    k2s: list[float] = []
    k4s: list[float] = []
    ys: list[float] = []
    points: list[dict[str, float]] = []
    for k in k_grid:
        delta = _phase_shift_segments(
            k_fm1=k,
            r_end_fm=r_end_fm,
            segments=segments,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        if not math.isfinite(delta):
            continue
        t = math.tan(delta)
        if abs(t) < 1e-15:
            continue
        y = k / t
        k2 = k * k
        if not (math.isfinite(y) and math.isfinite(k2)):
            continue
        k2s.append(k2)
        k4s.append(k2 * k2)
        ys.append(y)
        points.append({"k_fm1": float(k), "kcot_fm1": float(y), "delta_rad": float(delta)})
    if len(ys) < 10:
        raise ValueError("insufficient points for ERE fit")
    s00 = float(len(ys))
    s01 = float(sum(k2s))
    s02 = float(sum(k4s))
    s11 = float(sum(x * x for x in k2s))
    s12 = float(sum(x * z for x, z in zip(k2s, k4s)))
    s22 = float(sum(z * z for z in k4s))
    b0 = float(sum(ys))
    b1 = float(sum(y * x for y, x in zip(ys, k2s)))
    b2 = float(sum(y * z for y, z in zip(ys, k4s)))
    c0, c2, c4 = _solve_3x3([[s00, s01, s02], [s01, s11, s12], [s02, s12, s22]], [b0, b1, b2])
    a_out = -1.0 / c0 if c0 != 0 else float("nan")
    r_eff = 2.0 * c2
    v2_out = c4
    rms = math.sqrt(
        sum((y - (c0 + c2 * x + c4 * (x * x))) ** 2 for y, x in zip(ys, k2s)) / float(len(ys))
    )
    return {
        "a_fm": float(a_out),
        "r_eff_fm": float(r_eff),
        "v2_fm3": float(v2_out),
        "coeffs": {"c0_fm1": float(c0), "c2_fm": float(c2), "c4_fm3": float(c4)},
        "fit_rms_fm1": float(rms),
        "points": points,
    }


def _fit_kcot_ere_repulsive_core_two_range(
    *,
    rc_fm: float,
    r1_fm: float,
    r2_fm: float,
    vc_mev: float,
    v1_mev: float,
    v2_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    k_grid = [0.002 * i for i in range(1, 31)]  # 0.002..0.060 fm^-1
    k2s: list[float] = []
    k4s: list[float] = []
    ys: list[float] = []
    points: list[dict[str, float]] = []
    for k in k_grid:
        delta = _phase_shift_repulsive_core_two_range(
            k_fm1=k,
            rc_fm=rc_fm,
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            vc_mev=vc_mev,
            v1_mev=v1_mev,
            v2_mev=v2_mev,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        if not math.isfinite(delta):
            continue
        t = math.tan(delta)
        if abs(t) < 1e-15:
            continue
        y = k / t
        k2 = k * k
        if not (math.isfinite(y) and math.isfinite(k2)):
            continue
        k2s.append(k2)
        k4s.append(k2 * k2)
        ys.append(y)
        points.append({"k_fm1": float(k), "kcot_fm1": float(y), "delta_rad": float(delta)})
    if len(ys) < 10:
        raise ValueError("insufficient points for ERE fit")
    s00 = float(len(ys))
    s01 = float(sum(k2s))
    s02 = float(sum(k4s))
    s11 = float(sum(x * x for x in k2s))
    s12 = float(sum(x * z for x, z in zip(k2s, k4s)))
    s22 = float(sum(z * z for z in k4s))
    b0 = float(sum(ys))
    b1 = float(sum(y * x for y, x in zip(ys, k2s)))
    b2 = float(sum(y * z for y, z in zip(ys, k4s)))
    c0, c2, c4 = _solve_3x3([[s00, s01, s02], [s01, s11, s12], [s02, s12, s22]], [b0, b1, b2])
    a_out = -1.0 / c0 if c0 != 0 else float("nan")
    r_eff = 2.0 * c2
    v2_out = c4
    rms = math.sqrt(
        sum((y - (c0 + c2 * x + c4 * (x * x))) ** 2 for y, x in zip(ys, k2s)) / float(len(ys))
    )
    return {
        "a_fm": float(a_out),
        "r_eff_fm": float(r_eff),
        "v2_fm3": float(v2_out),
        "coeffs": {"c0_fm1": float(c0), "c2_fm": float(c2), "c4_fm3": float(c4)},
        "fit_rms_fm1": float(rms),
        "points": points,
    }


def _eval_triplet_candidate(
    *,
    b_mev: float,
    r1_fm: float,
    r2_fm: float,
    v2_mev: float,
    targets: dict[str, float],
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object] | None:
    if not (0.05 < r1_fm < r2_fm < 6.0 and v2_mev >= 0):
        return None
    try:
        v1 = _solve_v1_from_b(
            b_mev=b_mev, r1_fm=r1_fm, r2_fm=r2_fm, v2_mev=v2_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )["V1_MeV"]
    except Exception:
        return None

    if not (math.isfinite(v1) and v1 >= v2_mev):
        return None

    a_exact = _scattering_length_exact(
        r1_fm=r1_fm, r2_fm=r2_fm, v1_mev=v1, v2_mev=v2_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
    )
    if not math.isfinite(a_exact):
        return None

    try:
        ere = _fit_kcot_ere(r1_fm=r1_fm, r2_fm=r2_fm, v1_mev=v1, v2_mev=v2_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    except Exception:
        return None

    r_eff = float(ere["r_eff_fm"])
    v2 = float(ere["v2_fm3"])
    if not (math.isfinite(r_eff) and math.isfinite(v2)):
        return None

    da = a_exact - float(targets["a_t_fm"])
    dr = r_eff - float(targets["r_t_fm"])
    dv2 = v2 - float(targets["v2t_fm3"])
    score = abs(da) / 0.2 + abs(dr) / 0.05 + abs(dv2) / 0.05

    return {
        "r1_fm": float(r1_fm),
        "r2_fm": float(r2_fm),
        "v2_mev": float(v2_mev),
        "v1_mev": float(v1),
        "a_exact_fm": float(a_exact),
        "ere": ere,
        "deltas": {"da_fm": float(da), "dr_fm": float(dr), "dv2_fm3": float(dv2)},
        "score": float(score),
    }


def _yukawa_tail_avg_factor(*, length_fm: float, lambda_pi_fm: float) -> float:
    """
    Average factor for an exponential tail relative to its boundary value:

      V(r) = V(R2) * exp(-(r-R2)/λπ),  for r in [R2, R2+L]

    Approximating this tail by a constant segment with the same mean value gives:
      <V>/V(R2) = (λπ/L) * (1 - exp(-L/λπ)).

    Returns a number in (0,1] for L>0.
    """
    if not (math.isfinite(length_fm) and length_fm > 0 and math.isfinite(lambda_pi_fm) and lambda_pi_fm > 0):
        return float("nan")
    x = float(length_fm) / float(lambda_pi_fm)
    if x <= 0:
        return float("nan")
    return float((1.0 / x) * (1.0 - math.exp(-x)))


def _barrier_tail_config(
    *,
    lambda_pi_fm: float,
    tail_len_over_lambda: float,
    barrier_len_fraction: float,
    barrier_height_factor: float,
) -> dict[str, float]:
    """
    Helper for Step 7.13.5: split the Yukawa-tail coarse graining region into
    a repulsive barrier + attractive tail, while preserving the *mean* tail value.

    Total tail coarse-graining length:
      L3_total = tail_len_over_lambda * λπ

    Split:
      Lb = barrier_len_fraction * L3_total
      Lt = L3_total - Lb

    For a given outer-well depth V2>=0, we define the mean tail depth:
      V3_mean = <exp tail mean> * V2

    Then we choose the barrier height and the compensating tail depth:
      Vb = barrier_height_factor * V3_mean   (repulsive: +Vb)
      Vt = (V3_mean*L3_total + Vb*Lb) / Lt   (attractive: -Vt)

    This makes (Lb*(+Vb) + Lt*(-Vt))/L3_total = -V3_mean.
    """
    if not (math.isfinite(lambda_pi_fm) and lambda_pi_fm > 0):
        raise ValueError("invalid lambda_pi_fm")
    if not (math.isfinite(tail_len_over_lambda) and tail_len_over_lambda > 0):
        raise ValueError("invalid tail_len_over_lambda")
    if not (math.isfinite(barrier_len_fraction) and 0.0 < barrier_len_fraction < 1.0):
        raise ValueError("invalid barrier_len_fraction")
    if not (math.isfinite(barrier_height_factor) and barrier_height_factor >= 0.0):
        raise ValueError("invalid barrier_height_factor")

    l3_total_fm = float(tail_len_over_lambda) * float(lambda_pi_fm)
    if not (math.isfinite(l3_total_fm) and l3_total_fm > 0):
        raise ValueError("invalid L3_total")
    lb_fm = float(barrier_len_fraction) * float(l3_total_fm)
    lt_fm = float(l3_total_fm) - float(lb_fm)
    if not (math.isfinite(lb_fm) and math.isfinite(lt_fm) and lb_fm > 0 and lt_fm > 0):
        raise ValueError("invalid barrier/tail split")

    tail_factor = _yukawa_tail_avg_factor(length_fm=l3_total_fm, lambda_pi_fm=lambda_pi_fm)
    if not (math.isfinite(tail_factor) and 0 < tail_factor <= 1.0):
        raise ValueError("invalid tail_factor")

    # Linear coefficients in V2: V3_mean=tail_factor*V2, Vb=barrier_coeff*V2, Vt=tail_depth_coeff*V2.
    barrier_coeff = float(tail_factor) * float(barrier_height_factor)
    tail_depth_coeff = float(tail_factor) * float((l3_total_fm + lb_fm * float(barrier_height_factor)) / lt_fm)

    return {
        "lambda_pi_fm": float(lambda_pi_fm),
        "tail_len_over_lambda": float(tail_len_over_lambda),
        "L3_total_fm": float(l3_total_fm),
        "barrier_len_fraction": float(barrier_len_fraction),
        "Lb_fm": float(lb_fm),
        "Lt_fm": float(lt_fm),
        "tail_factor": float(tail_factor),
        "barrier_height_factor": float(barrier_height_factor),
        "barrier_coeff": float(barrier_coeff),
        "tail_depth_coeff": float(tail_depth_coeff),
    }


def _barrier_tail_config_free_depth(
    *,
    lambda_pi_fm: float,
    tail_len_over_lambda: float,
    barrier_len_fraction: float,
    barrier_height_factor: float,
    tail_depth_factor: float,
) -> dict[str, float]:
    """
    Helper for Step 7.13.7: split the Yukawa-tail coarse graining region into a repulsive barrier
    + attractive tail, but allow the tail depth to vary independently (no mean-preserving rule).

    Total tail coarse-graining length:
      L3_total = tail_len_over_lambda * λπ

    Split:
      Lb = barrier_len_fraction * L3_total
      Lt = L3_total - Lb

    For a given outer-well depth V2>=0, we define:
      V3_mean = <exp tail mean> * V2
      Vb      = barrier_height_factor * V3_mean   (repulsive: +Vb)
      Vt      = tail_depth_factor   * V3_mean     (attractive: -Vt)

    Note: mean-preserving corresponds to:
      tail_depth_factor = (L3_total + Lb*barrier_height_factor) / Lt.
    """
    if not (math.isfinite(lambda_pi_fm) and lambda_pi_fm > 0):
        raise ValueError("invalid lambda_pi_fm")
    if not (math.isfinite(tail_len_over_lambda) and tail_len_over_lambda > 0):
        raise ValueError("invalid tail_len_over_lambda")
    if not (math.isfinite(barrier_len_fraction) and 0.0 < barrier_len_fraction < 1.0):
        raise ValueError("invalid barrier_len_fraction")
    if not (math.isfinite(barrier_height_factor) and barrier_height_factor >= 0.0):
        raise ValueError("invalid barrier_height_factor")
    if not (math.isfinite(tail_depth_factor) and tail_depth_factor >= 0.0):
        raise ValueError("invalid tail_depth_factor")

    l3_total_fm = float(tail_len_over_lambda) * float(lambda_pi_fm)
    if not (math.isfinite(l3_total_fm) and l3_total_fm > 0):
        raise ValueError("invalid L3_total")
    lb_fm = float(barrier_len_fraction) * float(l3_total_fm)
    lt_fm = float(l3_total_fm) - float(lb_fm)
    if not (math.isfinite(lb_fm) and math.isfinite(lt_fm) and lb_fm > 0 and lt_fm > 0):
        raise ValueError("invalid barrier/tail split")

    tail_factor = _yukawa_tail_avg_factor(length_fm=l3_total_fm, lambda_pi_fm=lambda_pi_fm)
    if not (math.isfinite(tail_factor) and 0 < tail_factor <= 1.0):
        raise ValueError("invalid tail_factor")

    barrier_coeff = float(tail_factor) * float(barrier_height_factor)
    tail_depth_coeff = float(tail_factor) * float(tail_depth_factor)

    tail_depth_factor_mean_preserving = float((l3_total_fm + lb_fm * float(barrier_height_factor)) / lt_fm)

    return {
        "lambda_pi_fm": float(lambda_pi_fm),
        "tail_len_over_lambda": float(tail_len_over_lambda),
        "L3_total_fm": float(l3_total_fm),
        "barrier_len_fraction": float(barrier_len_fraction),
        "Lb_fm": float(lb_fm),
        "Lt_fm": float(lt_fm),
        "tail_factor": float(tail_factor),
        "barrier_height_factor": float(barrier_height_factor),
        "tail_depth_factor": float(tail_depth_factor),
        "tail_depth_factor_mean_preserving": float(tail_depth_factor_mean_preserving),
        "barrier_coeff": float(barrier_coeff),
        "tail_depth_coeff": float(tail_depth_coeff),
    }


def _eval_triplet_candidate_three_range_tail(
    *,
    b_mev: float,
    r1_fm: float,
    r2_fm: float,
    v2_mev: float,
    lambda_pi_fm: float,
    tail_len_over_lambda: float,
    targets: dict[str, float],
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object] | None:
    """
    3-range ansatz class (minimal extension of the two-range well):

      V(r) = -V1   (0<=r<R1),
             -V2   (R1<=r<R2),
             -V3   (R2<=r<R3),  with V3 = <exp tail> * V2,  R3 = R2 + L3, L3 = tail_len_over_lambda * λπ,
             0      (r>R3).
    """
    if not (0.05 < r1_fm < r2_fm < 6.5 and v2_mev >= 0 and math.isfinite(lambda_pi_fm) and lambda_pi_fm > 0):
        return None
    if not (math.isfinite(tail_len_over_lambda) and tail_len_over_lambda > 0):
        return None

    l3_fm = float(tail_len_over_lambda) * float(lambda_pi_fm)
    if not (math.isfinite(l3_fm) and l3_fm > 0):
        return None
    r3_fm = float(r2_fm) + float(l3_fm)
    if not (r3_fm > r2_fm and r3_fm < 10.0):
        return None

    tail_factor = _yukawa_tail_avg_factor(length_fm=l3_fm, lambda_pi_fm=lambda_pi_fm)
    if not (math.isfinite(tail_factor) and 0 < tail_factor <= 1.0):
        return None
    v3_mev = float(v2_mev) * float(tail_factor)
    if not (math.isfinite(v3_mev) and v3_mev >= 0):
        return None

    seg_after = [
        (float(r2_fm - r1_fm), -float(v2_mev)),
        (float(l3_fm), -float(v3_mev)),
    ]
    try:
        v1 = _solve_v1_from_b_segments(
            b_mev=b_mev,
            first_len_fm=float(r1_fm),
            segments_after_first=seg_after,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )["V1_MeV"]
    except Exception:
        return None
    if not (math.isfinite(v1) and v1 > b_mev and v1 >= v2_mev):
        return None

    segs = [(float(r1_fm), -float(v1))] + seg_after

    a_exact = _scattering_length_exact_segments(r_end_fm=r3_fm, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    if not math.isfinite(a_exact):
        return None
    try:
        ere = _fit_kcot_ere_segments(r_end_fm=r3_fm, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    except Exception:
        return None
    r_eff = float(ere["r_eff_fm"])
    v2_shape = float(ere["v2_fm3"])
    if not (math.isfinite(r_eff) and math.isfinite(v2_shape)):
        return None

    da = a_exact - float(targets["a_t_fm"])
    dr = r_eff - float(targets["r_t_fm"])
    dv2 = v2_shape - float(targets["v2t_fm3"])
    score = abs(da) / 0.2 + abs(dr) / 0.05 + abs(dv2) / 0.05

    return {
        "r1_fm": float(r1_fm),
        "r2_fm": float(r2_fm),
        "r3_fm": float(r3_fm),
        "tail_len_fm": float(l3_fm),
        "tail_factor": float(tail_factor),
        "v2_mev": float(v2_mev),
        "v3_mev": float(v3_mev),
        "v1_mev": float(v1),
        "a_exact_fm": float(a_exact),
        "ere": ere,
        "deltas": {"da_fm": float(da), "dr_fm": float(dr), "dv2_fm3": float(dv2)},
        "score": float(score),
    }


def _eval_triplet_candidate_three_range_barrier_tail(
    *,
    b_mev: float,
    r1_fm: float,
    r2_fm: float,
    v2_mev: float,
    lambda_pi_fm: float,
    tail_len_over_lambda: float,
    barrier_len_fraction: float,
    barrier_height_factor: float,
    targets: dict[str, float],
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object] | None:
    """
    Step 7.13.5: 3-range tail, but split into barrier+tail (preserving mean tail).

      V(r) = -V1                (0<=r<R1),
             -V2                (R1<=r<R2),
             +Vb                (R2<=r<Rb),
             -Vt                (Rb<=r<R3),
              0                 (r>R3),

    with:
      R3 = R2 + L3_total,  Rb = R2 + Lb,
      V3_mean = <exp tail mean> * V2,
      Vb = barrier_height_factor * V3_mean,
      Vt chosen so that the length-weighted mean over [R2,R3] equals -V3_mean.
    """
    if not (0.05 < r1_fm < r2_fm < 6.5 and v2_mev >= 0 and math.isfinite(lambda_pi_fm) and lambda_pi_fm > 0):
        return None

    try:
        cfg = _barrier_tail_config(
            lambda_pi_fm=lambda_pi_fm,
            tail_len_over_lambda=tail_len_over_lambda,
            barrier_len_fraction=barrier_len_fraction,
            barrier_height_factor=barrier_height_factor,
        )
    except Exception:
        return None

    l3_total_fm = float(cfg["L3_total_fm"])
    lb_fm = float(cfg["Lb_fm"])
    lt_fm = float(cfg["Lt_fm"])
    tail_factor = float(cfg["tail_factor"])
    barrier_coeff = float(cfg["barrier_coeff"])
    tail_depth_coeff = float(cfg["tail_depth_coeff"])

    r3_fm = float(r2_fm) + float(l3_total_fm)
    rb_fm = float(r2_fm) + float(lb_fm)
    if not (r3_fm > rb_fm > r2_fm and r3_fm < 10.0):
        return None

    v3_mean_mev = float(v2_mev) * float(tail_factor)
    v3_barrier_mev = float(v2_mev) * float(barrier_coeff)
    v3_tail_mev = float(v2_mev) * float(tail_depth_coeff)
    if not (math.isfinite(v3_mean_mev) and math.isfinite(v3_barrier_mev) and math.isfinite(v3_tail_mev)):
        return None
    if not (v3_mean_mev >= 0 and v3_barrier_mev >= 0 and v3_tail_mev >= 0):
        return None

    seg_after = [
        (float(r2_fm - r1_fm), -float(v2_mev)),
        (float(lb_fm), +float(v3_barrier_mev)),
        (float(lt_fm), -float(v3_tail_mev)),
    ]
    try:
        v1 = _solve_v1_from_b_segments(
            b_mev=b_mev,
            first_len_fm=float(r1_fm),
            segments_after_first=seg_after,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )["V1_MeV"]
    except Exception:
        return None
    if not (math.isfinite(v1) and v1 > b_mev and v1 >= v2_mev):
        return None

    segs = [(float(r1_fm), -float(v1))] + seg_after

    a_exact = _scattering_length_exact_segments(r_end_fm=r3_fm, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    if not math.isfinite(a_exact):
        return None
    try:
        ere = _fit_kcot_ere_segments(r_end_fm=r3_fm, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    except Exception:
        return None
    r_eff = float(ere["r_eff_fm"])
    v2_shape = float(ere["v2_fm3"])
    if not (math.isfinite(r_eff) and math.isfinite(v2_shape)):
        return None

    da = a_exact - float(targets["a_t_fm"])
    dr = r_eff - float(targets["r_t_fm"])
    dv2 = v2_shape - float(targets["v2t_fm3"])
    score = abs(da) / 0.2 + abs(dr) / 0.05 + abs(dv2) / 0.05

    return {
        "r1_fm": float(r1_fm),
        "r2_fm": float(r2_fm),
        "r3_fm": float(r3_fm),
        "rb_fm": float(rb_fm),
        "tail_total_len_fm": float(l3_total_fm),
        "tail_factor": float(tail_factor),
        "barrier_len_fraction": float(barrier_len_fraction),
        "barrier_height_factor": float(barrier_height_factor),
        "lb_fm": float(lb_fm),
        "lt_fm": float(lt_fm),
        "v2_mev": float(v2_mev),
        "v3_mean_mev": float(v3_mean_mev),
        "v3_barrier_mev": float(v3_barrier_mev),
        "v3_tail_mev": float(v3_tail_mev),
        "v1_mev": float(v1),
        "a_exact_fm": float(a_exact),
        "ere": ere,
        "deltas": {"da_fm": float(da), "dr_fm": float(dr), "dv2_fm3": float(dv2)},
        "score": float(score),
        "barrier_tail_config": cfg,
    }


def _eval_triplet_candidate_three_range_barrier_tail_free_depth(
    *,
    b_mev: float,
    r1_fm: float,
    r2_fm: float,
    v2_mev: float,
    lambda_pi_fm: float,
    tail_len_over_lambda: float,
    barrier_len_fraction: float,
    barrier_height_factor: float,
    tail_depth_factor: float,
    targets: dict[str, float],
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object] | None:
    """
    Step 7.13.7: barrier+tail split with an independently tunable tail depth factor (no mean-preserving rule).
    """
    if not (
        0.05 < r1_fm < r2_fm < 6.5
        and v2_mev >= 0
        and math.isfinite(lambda_pi_fm)
        and lambda_pi_fm > 0
        and math.isfinite(tail_depth_factor)
        and tail_depth_factor >= 0
    ):
        return None

    try:
        cfg = _barrier_tail_config_free_depth(
            lambda_pi_fm=lambda_pi_fm,
            tail_len_over_lambda=tail_len_over_lambda,
            barrier_len_fraction=barrier_len_fraction,
            barrier_height_factor=barrier_height_factor,
            tail_depth_factor=tail_depth_factor,
        )
    except Exception:
        return None

    l3_total_fm = float(cfg["L3_total_fm"])
    lb_fm = float(cfg["Lb_fm"])
    lt_fm = float(cfg["Lt_fm"])
    tail_factor = float(cfg["tail_factor"])
    barrier_coeff = float(cfg["barrier_coeff"])
    tail_depth_coeff = float(cfg["tail_depth_coeff"])

    r3_fm = float(r2_fm) + float(l3_total_fm)
    rb_fm = float(r2_fm) + float(lb_fm)
    if not (r3_fm > rb_fm > r2_fm and r3_fm < 10.0):
        return None

    v3_mean_mev = float(v2_mev) * float(tail_factor)
    v3_barrier_mev = float(v2_mev) * float(barrier_coeff)
    v3_tail_mev = float(v2_mev) * float(tail_depth_coeff)
    if not (math.isfinite(v3_mean_mev) and math.isfinite(v3_barrier_mev) and math.isfinite(v3_tail_mev)):
        return None
    if not (v3_mean_mev >= 0 and v3_barrier_mev >= 0 and v3_tail_mev >= 0):
        return None

    seg_after = [
        (float(r2_fm - r1_fm), -float(v2_mev)),
        (float(lb_fm), +float(v3_barrier_mev)),
        (float(lt_fm), -float(v3_tail_mev)),
    ]
    try:
        v1 = _solve_v1_from_b_segments(
            b_mev=b_mev,
            first_len_fm=float(r1_fm),
            segments_after_first=seg_after,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )["V1_MeV"]
    except Exception:
        return None
    if not (math.isfinite(v1) and v1 > b_mev and v1 >= v2_mev):
        return None

    segs = [(float(r1_fm), -float(v1))] + seg_after

    a_exact = _scattering_length_exact_segments(r_end_fm=r3_fm, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    if not math.isfinite(a_exact):
        return None
    try:
        ere = _fit_kcot_ere_segments(r_end_fm=r3_fm, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    except Exception:
        return None
    r_eff = float(ere["r_eff_fm"])
    v2_shape = float(ere["v2_fm3"])
    if not (math.isfinite(r_eff) and math.isfinite(v2_shape)):
        return None

    da = a_exact - float(targets["a_t_fm"])
    dr = r_eff - float(targets["r_t_fm"])
    dv2 = v2_shape - float(targets["v2t_fm3"])
    score = abs(da) / 0.2 + abs(dr) / 0.05 + abs(dv2) / 0.05

    return {
        "r1_fm": float(r1_fm),
        "r2_fm": float(r2_fm),
        "r3_fm": float(r3_fm),
        "rb_fm": float(rb_fm),
        "tail_total_len_fm": float(l3_total_fm),
        "tail_factor": float(tail_factor),
        "barrier_len_fraction": float(barrier_len_fraction),
        "barrier_height_factor": float(barrier_height_factor),
        "tail_depth_factor": float(tail_depth_factor),
        "lb_fm": float(lb_fm),
        "lt_fm": float(lt_fm),
        "v2_mev": float(v2_mev),
        "v3_mean_mev": float(v3_mean_mev),
        "v3_barrier_mev": float(v3_barrier_mev),
        "v3_tail_mev": float(v3_tail_mev),
        "v1_mev": float(v1),
        "a_exact_fm": float(a_exact),
        "ere": ere,
        "deltas": {"da_fm": float(da), "dr_fm": float(dr), "dv2_fm3": float(dv2)},
        "score": float(score),
        "barrier_tail_config": cfg,
    }


def _eval_triplet_candidate_repulsive_core_fixed_geometry(
    *,
    b_mev: float,
    rc_fm: float,
    vc_mev: float,
    r1_fm: float,
    r2_fm: float,
    v2_mev: float,
    targets: dict[str, float],
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object] | None:
    # Keep a non-zero core so that step 7.9.8 is not degenerate with step 7.9.6/7.9.7.
    if not (0.05 <= rc_fm < (r1_fm - 0.02) and 0.05 < r1_fm < r2_fm < 6.0 and vc_mev >= 0 and v2_mev >= 0):
        return None
    try:
        v1 = _solve_v1_from_b_repulsive_core_two_range(
            b_mev=b_mev,
            rc_fm=rc_fm,
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            vc_mev=vc_mev,
            v2_mev=v2_mev,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )["V1_MeV"]
    except Exception:
        return None

    if not (math.isfinite(v1) and v1 >= v2_mev):
        return None

    a_exact = _scattering_length_exact_repulsive_core_two_range(
        rc_fm=rc_fm,
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        vc_mev=vc_mev,
        v1_mev=v1,
        v2_mev=v2_mev,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    if not math.isfinite(a_exact):
        return None

    try:
        ere = _fit_kcot_ere_repulsive_core_two_range(
            rc_fm=rc_fm,
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            vc_mev=vc_mev,
            v1_mev=v1,
            v2_mev=v2_mev,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
    except Exception:
        return None

    r_eff = float(ere["r_eff_fm"])
    v2 = float(ere["v2_fm3"])
    if not (math.isfinite(r_eff) and math.isfinite(v2)):
        return None

    da = a_exact - float(targets["a_t_fm"])
    dr = r_eff - float(targets["r_t_fm"])
    dv2 = v2 - float(targets["v2t_fm3"])
    score = abs(da) / 0.2 + abs(dr) / 0.05 + abs(dv2) / 0.05

    return {
        "rc_fm": float(rc_fm),
        "vc_mev": float(vc_mev),
        "r1_fm": float(r1_fm),
        "r2_fm": float(r2_fm),
        "v2_mev": float(v2_mev),
        "v1_mev": float(v1),
        "a_exact_fm": float(a_exact),
        "ere": ere,
        "deltas": {"da_fm": float(da), "dr_fm": float(dr), "dv2_fm3": float(dv2)},
        "score": float(score),
    }


def _fit_triplet_repulsive_core_fixed_geometry(
    *,
    b_mev: float,
    targets: dict[str, float],
    r1_fm: float,
    r2_fm: float,
    v2_hint_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    """
    Step 7.9.8 triplet fit:
      - Fix (R1,R2) to the Step 7.9.6 best-fit values to avoid adding extra geometric degrees.
      - Add a repulsive core (Rc,Vc) and re-fit V2 such that (a_t,r_t,v2t) remain matched, with V1 solved by B.
    """
    rc_min = 0.05
    rc_max = min(max(rc_min + 0.02, r1_fm - 0.05), 0.9 * r1_fm)
    if not (rc_max > rc_min and r1_fm > 0.1 and r2_fm > r1_fm):
        raise ValueError("invalid geometry for core fit")

    v2_lo = 0.0
    v2_hi = min(150.0, max(60.0, float(v2_hint_mev) + 60.0))
    n_rc = 10
    rc_grid = [rc_min + (rc_max - rc_min) * i / (n_rc - 1) for i in range(n_rc)]
    vc_grid = [50.0, 100.0, 200.0, 300.0, 400.0, 600.0, 800.0]
    n_v2 = 24
    v2_grid = [v2_lo + (v2_hi - v2_lo) * i / (n_v2 - 1) for i in range(n_v2)]

    best: dict[str, object] | None = None
    for rc in rc_grid:
        for vc in vc_grid:
            for v2 in v2_grid:
                cand = _eval_triplet_candidate_repulsive_core_fixed_geometry(
                    b_mev=b_mev,
                    rc_fm=rc,
                    vc_mev=vc,
                    r1_fm=r1_fm,
                    r2_fm=r2_fm,
                    v2_mev=v2,
                    targets=targets,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
                if cand is None:
                    continue
                if best is None or float(cand["score"]) < float(best["score"]):
                    best = cand

    if best is None:
        raise ValueError("no valid candidates in grid scan")

    # Local refinement (coordinate descent).
    rc = float(best["rc_fm"])
    vc = float(best["vc_mev"])
    v2 = float(best["v2_mev"])
    step_rc = 0.04
    step_vc = 80.0
    step_v2 = 4.0
    for _ in range(40):
        improved = False
        for drc, dvc, dv2 in [
            (+step_rc, 0.0, 0.0),
            (-step_rc, 0.0, 0.0),
            (0.0, +step_vc, 0.0),
            (0.0, -step_vc, 0.0),
            (0.0, 0.0, +step_v2),
            (0.0, 0.0, -step_v2),
        ]:
            rc_c = min(max(rc + drc, rc_min), rc_max)
            vc_c = min(max(vc + dvc, 0.0), 900.0)
            v2_c = min(max(v2 + dv2, 0.0), 200.0)
            cand = _eval_triplet_candidate_repulsive_core_fixed_geometry(
                b_mev=b_mev,
                rc_fm=rc_c,
                vc_mev=vc_c,
                r1_fm=r1_fm,
                r2_fm=r2_fm,
                v2_mev=v2_c,
                targets=targets,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            if cand is None:
                continue
            if float(cand["score"]) + 1e-12 < float(best["score"]):
                best = cand
                rc = float(best["rc_fm"])
                vc = float(best["vc_mev"])
                v2 = float(best["v2_mev"])
                improved = True
        if not improved:
            step_rc *= 0.6
            step_vc *= 0.6
            step_v2 *= 0.6
            if step_rc < 0.003 and step_vc < 5.0 and step_v2 < 0.2:
                break

    best = dict(best)
    best["note"] = "Fit triplet by scanning (Rc,Vc,V2) with fixed (R1,R2); V1 is solved by B."
    best["fixed_geometry"] = {"R1_fm": float(r1_fm), "R2_fm": float(r2_fm)}
    return best


def _fit_triplet_two_range(
    *,
    b_mev: float,
    targets: dict[str, float],
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    """
    Fit a two-range (two-step) attractive potential:
      V(r) = -V1 for 0<=r<R1,
             -V2 for R1<=r<R2,
              0  for r>=R2,
    with constraints:
      - V1 is solved from B (bound state),
      - (R1,R2,V2) are searched to match (a_t, r_t, v2t) as closely as possible.
    """
    r1_grid = [0.25 + 0.05 * i for i in range(1, 19)]  # 0.30..1.15 fm
    r2_grid_all = [1.3 + 0.05 * i for i in range(1, 33)]  # 1.35..2.90 fm
    v2_grid = [2.0 + 2.0 * i for i in range(0, 30)]  # 2..60 MeV

    best: dict[str, object] | None = None
    for r1 in r1_grid:
        for r2 in r2_grid_all:
            if r2 <= r1 + 0.25:
                continue
            for v2 in v2_grid:
                cand = _eval_triplet_candidate(
                    b_mev=b_mev,
                    r1_fm=r1,
                    r2_fm=r2,
                    v2_mev=v2,
                    targets=targets,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
                if cand is None:
                    continue
                if best is None or float(cand["score"]) < float(best["score"]):
                    best = cand

    if best is None:
        raise ValueError("no valid two-range candidate found in coarse grid")

    r1 = float(best["r1_fm"])
    r2 = float(best["r2_fm"])
    v2 = float(best["v2_mev"])
    step_r1 = 0.03
    step_r2 = 0.03
    step_v2 = 3.0
    for _ in range(24):
        improved = False
        for dr1, dr2, dv in [
            (+step_r1, 0.0, 0.0),
            (-step_r1, 0.0, 0.0),
            (0.0, +step_r2, 0.0),
            (0.0, -step_r2, 0.0),
            (0.0, 0.0, +step_v2),
            (0.0, 0.0, -step_v2),
        ]:
            cand = _eval_triplet_candidate(
                b_mev=b_mev,
                r1_fm=max(0.05, r1 + dr1),
                r2_fm=max((r1 + dr1) + 0.25, r2 + dr2),
                v2_mev=max(0.1, v2 + dv),
                targets=targets,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            if cand is None:
                continue
            if float(cand["score"]) + 1e-12 < float(best["score"]):
                best = cand
                r1 = float(best["r1_fm"])
                r2 = float(best["r2_fm"])
                v2 = float(best["v2_mev"])
                improved = True
        if not improved:
            step_r1 *= 0.6
            step_r2 *= 0.6
            step_v2 *= 0.6
            if step_r1 < 0.004 and step_r2 < 0.004 and step_v2 < 0.35:
                break

    return best


def _fit_triplet_two_range_pion_constrained(
    *,
    b_mev: float,
    targets: dict[str, float],
    mu_mev: float,
    hbarc_mev_fm: float,
    lambda_pi_fm: float,
    r1_over_lambda_min: float = 0.25,
    r1_over_lambda_max: float = 0.75,
    r2_over_lambda_min: float = 1.00,
    r2_over_lambda_max: float = 2.05,
) -> dict[str, object]:
    """
    Same as _fit_triplet_two_range(), but constrains geometry by λπ (operational range scale):
      R1 = O(λπ),  R2 = O(λπ),
    by searching only within specified ratio ranges.
    """
    if not (math.isfinite(lambda_pi_fm) and lambda_pi_fm > 0):
        raise ValueError("invalid lambda_pi_fm")
    if not (r1_over_lambda_min > 0 and r1_over_lambda_max > r1_over_lambda_min):
        raise ValueError("invalid R1/λπ bounds")
    if not (r2_over_lambda_min > 0 and r2_over_lambda_max > r2_over_lambda_min):
        raise ValueError("invalid R2/λπ bounds")

    r1_min = float(r1_over_lambda_min) * float(lambda_pi_fm)
    r1_max = float(r1_over_lambda_max) * float(lambda_pi_fm)
    r2_min = float(r2_over_lambda_min) * float(lambda_pi_fm)
    r2_max = float(r2_over_lambda_max) * float(lambda_pi_fm)

    r1_grid = [r1_min + (r1_max - r1_min) * i / 16.0 for i in range(17)]
    r2_grid_all = [r2_min + (r2_max - r2_min) * i / 28.0 for i in range(29)]
    v2_grid = [2.0 + 2.0 * i for i in range(0, 30)]  # 2..60 MeV

    best: dict[str, object] | None = None
    for r1 in r1_grid:
        for r2 in r2_grid_all:
            if r2 <= r1 + 0.25:
                continue
            for v2 in v2_grid:
                cand = _eval_triplet_candidate(
                    b_mev=b_mev,
                    r1_fm=r1,
                    r2_fm=r2,
                    v2_mev=v2,
                    targets=targets,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
                if cand is None:
                    continue
                if best is None or float(cand["score"]) < float(best["score"]):
                    best = cand

    if best is None:
        raise ValueError("no valid two-range candidate found under λπ constraints")

    r1 = float(best["r1_fm"])
    r2 = float(best["r2_fm"])
    v2 = float(best["v2_mev"])

    step_r1 = 0.03
    step_r2 = 0.03
    step_v2 = 3.0
    for _ in range(24):
        improved = False
        for dr1, dr2, dv in [
            (+step_r1, 0.0, 0.0),
            (-step_r1, 0.0, 0.0),
            (0.0, +step_r2, 0.0),
            (0.0, -step_r2, 0.0),
            (0.0, 0.0, +step_v2),
            (0.0, 0.0, -step_v2),
        ]:
            r1_c = min(max(r1 + dr1, r1_min), r1_max)
            r2_c = min(max(r2 + dr2, max(r1_c + 0.25, r2_min)), r2_max)
            v2_c = max(0.1, v2 + dv)
            cand = _eval_triplet_candidate(
                b_mev=b_mev,
                r1_fm=r1_c,
                r2_fm=r2_c,
                v2_mev=v2_c,
                targets=targets,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            if cand is None:
                continue
            if float(cand["score"]) + 1e-12 < float(best["score"]):
                best = cand
                r1 = float(best["r1_fm"])
                r2 = float(best["r2_fm"])
                v2 = float(best["v2_mev"])
                improved = True
        if not improved:
            step_r1 *= 0.6
            step_r2 *= 0.6
            step_v2 *= 0.6
            if step_r1 < 0.004 and step_r2 < 0.004 and step_v2 < 0.35:
                break

    best = dict(best)
    best["pion_constraint"] = {
        "lambda_pi_fm": float(lambda_pi_fm),
        "R1_over_lambda_pi_bounds": [float(r1_over_lambda_min), float(r1_over_lambda_max)],
        "R2_over_lambda_pi_bounds": [float(r2_over_lambda_min), float(r2_over_lambda_max)],
    }
    return best


def _fit_triplet_three_range_tail_pion_constrained(
    *,
    b_mev: float,
    targets: dict[str, float],
    mu_mev: float,
    hbarc_mev_fm: float,
    lambda_pi_fm: float,
    tail_len_over_lambda: float = 1.0,
    r1_over_lambda_min: float = 0.25,
    r1_over_lambda_max: float = 0.75,
    r2_over_lambda_min: float = 1.00,
    r2_over_lambda_max: float = 2.05,
) -> dict[str, object]:
    """
    Step 7.13.4 triplet fit:

    Same search structure as _fit_triplet_two_range_pion_constrained(), but the ansatz class is
    extended by a minimal Yukawa-like tail coarse-graining (a third segment beyond R2) with
    no additional free parameters:

      R3 = R2 + (tail_len_over_lambda * λπ),
      V3 = <exp tail mean> * V2.
    """
    if not (math.isfinite(lambda_pi_fm) and lambda_pi_fm > 0):
        raise ValueError("invalid lambda_pi_fm")
    if not (math.isfinite(tail_len_over_lambda) and tail_len_over_lambda > 0):
        raise ValueError("invalid tail_len_over_lambda")
    if not (r1_over_lambda_min > 0 and r1_over_lambda_max > r1_over_lambda_min):
        raise ValueError("invalid R1/λπ bounds")
    if not (r2_over_lambda_min > 0 and r2_over_lambda_max > r2_over_lambda_min):
        raise ValueError("invalid R2/λπ bounds")

    r1_min = float(r1_over_lambda_min) * float(lambda_pi_fm)
    r1_max = float(r1_over_lambda_max) * float(lambda_pi_fm)
    r2_min = float(r2_over_lambda_min) * float(lambda_pi_fm)
    r2_max = float(r2_over_lambda_max) * float(lambda_pi_fm)

    r1_grid = [r1_min + (r1_max - r1_min) * i / 16.0 for i in range(17)]
    r2_grid_all = [r2_min + (r2_max - r2_min) * i / 28.0 for i in range(29)]
    v2_grid = [2.0 + 2.0 * i for i in range(0, 30)]  # 2..60 MeV

    best: dict[str, object] | None = None
    for r1 in r1_grid:
        for r2 in r2_grid_all:
            if r2 <= r1 + 0.25:
                continue
            for v2 in v2_grid:
                cand = _eval_triplet_candidate_three_range_tail(
                    b_mev=b_mev,
                    r1_fm=r1,
                    r2_fm=r2,
                    v2_mev=v2,
                    lambda_pi_fm=float(lambda_pi_fm),
                    tail_len_over_lambda=float(tail_len_over_lambda),
                    targets=targets,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
                if cand is None:
                    continue
                if best is None or float(cand["score"]) < float(best["score"]):
                    best = cand

    if best is None:
        raise ValueError("no valid three-range candidate found under λπ constraints")

    r1 = float(best["r1_fm"])
    r2 = float(best["r2_fm"])
    v2 = float(best["v2_mev"])

    step_r1 = 0.03
    step_r2 = 0.03
    step_v2 = 3.0
    for _ in range(24):
        improved = False
        for dr1, dr2, dv in [
            (+step_r1, 0.0, 0.0),
            (-step_r1, 0.0, 0.0),
            (0.0, +step_r2, 0.0),
            (0.0, -step_r2, 0.0),
            (0.0, 0.0, +step_v2),
            (0.0, 0.0, -step_v2),
        ]:
            r1_c = min(max(r1 + dr1, r1_min), r1_max)
            r2_c = min(max(r2 + dr2, max(r1_c + 0.25, r2_min)), r2_max)
            v2_c = max(0.1, v2 + dv)
            cand = _eval_triplet_candidate_three_range_tail(
                b_mev=b_mev,
                r1_fm=r1_c,
                r2_fm=r2_c,
                v2_mev=v2_c,
                lambda_pi_fm=float(lambda_pi_fm),
                tail_len_over_lambda=float(tail_len_over_lambda),
                targets=targets,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            if cand is None:
                continue
            if float(cand["score"]) + 1e-12 < float(best["score"]):
                best = cand
                r1 = float(best["r1_fm"])
                r2 = float(best["r2_fm"])
                v2 = float(best["v2_mev"])
                improved = True
        if not improved:
            step_r1 *= 0.6
            step_r2 *= 0.6
            step_v2 *= 0.6
            if step_r1 < 0.004 and step_r2 < 0.004 and step_v2 < 0.35:
                break

    best = dict(best)
    best["pion_constraint"] = {
        "lambda_pi_fm": float(lambda_pi_fm),
        "R1_over_lambda_pi_bounds": [float(r1_over_lambda_min), float(r1_over_lambda_max)],
        "R2_over_lambda_pi_bounds": [float(r2_over_lambda_min), float(r2_over_lambda_max)],
    }
    best["tail_constraint"] = {
        "tail_len_over_lambda": float(tail_len_over_lambda),
        "tail_mean_factor_definition": "<V>/V(R2) = (λπ/L3) * (1 - exp(-L3/λπ))",
    }
    return best


def _fit_triplet_three_range_barrier_tail_pion_constrained(
    *,
    b_mev: float,
    targets: dict[str, float],
    mu_mev: float,
    hbarc_mev_fm: float,
    lambda_pi_fm: float,
    tail_len_over_lambda: float = 1.0,
    barrier_len_fraction: float = 0.5,
    barrier_height_factor: float = 1.0,
    r1_over_lambda_min: float = 0.25,
    r1_over_lambda_max: float = 0.75,
    r2_over_lambda_min: float = 1.00,
    r2_over_lambda_max: float = 2.05,
) -> dict[str, object]:
    """
    Step 7.13.5 triplet fit:

    Same search structure as _fit_triplet_three_range_tail_pion_constrained(), but the coarse-grained
    Yukawa tail region (R2..R3) is split into a repulsive barrier + compensating attractive tail, while
    preserving the original mean tail value (so we do not introduce an independent "tail strength" parameter).
    """
    if not (math.isfinite(lambda_pi_fm) and lambda_pi_fm > 0):
        raise ValueError("invalid lambda_pi_fm")
    if not (math.isfinite(tail_len_over_lambda) and tail_len_over_lambda > 0):
        raise ValueError("invalid tail_len_over_lambda")
    if not (math.isfinite(barrier_len_fraction) and 0.0 < barrier_len_fraction < 1.0):
        raise ValueError("invalid barrier_len_fraction")
    if not (math.isfinite(barrier_height_factor) and barrier_height_factor >= 0.0):
        raise ValueError("invalid barrier_height_factor")
    if not (r1_over_lambda_min > 0 and r1_over_lambda_max > r1_over_lambda_min):
        raise ValueError("invalid R1/λπ bounds")
    if not (r2_over_lambda_min > 0 and r2_over_lambda_max > r2_over_lambda_min):
        raise ValueError("invalid R2/λπ bounds")

    r1_min = float(r1_over_lambda_min) * float(lambda_pi_fm)
    r1_max = float(r1_over_lambda_max) * float(lambda_pi_fm)
    r2_min = float(r2_over_lambda_min) * float(lambda_pi_fm)
    r2_max = float(r2_over_lambda_max) * float(lambda_pi_fm)

    r1_grid = [r1_min + (r1_max - r1_min) * i / 16.0 for i in range(17)]
    r2_grid_all = [r2_min + (r2_max - r2_min) * i / 28.0 for i in range(29)]
    v2_grid = [2.0 + 2.0 * i for i in range(0, 30)]  # 2..60 MeV

    best: dict[str, object] | None = None
    for r1 in r1_grid:
        for r2 in r2_grid_all:
            if r2 <= r1 + 0.25:
                continue
            for v2 in v2_grid:
                cand = _eval_triplet_candidate_three_range_barrier_tail(
                    b_mev=b_mev,
                    r1_fm=r1,
                    r2_fm=r2,
                    v2_mev=v2,
                    lambda_pi_fm=float(lambda_pi_fm),
                    tail_len_over_lambda=float(tail_len_over_lambda),
                    barrier_len_fraction=float(barrier_len_fraction),
                    barrier_height_factor=float(barrier_height_factor),
                    targets=targets,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
                if cand is None:
                    continue
                if best is None or float(cand["score"]) < float(best["score"]):
                    best = cand

    if best is None:
        raise ValueError("no valid barrier+tail three-range candidate found under λπ constraints")

    r1 = float(best["r1_fm"])
    r2 = float(best["r2_fm"])
    v2 = float(best["v2_mev"])

    step_r1 = 0.03
    step_r2 = 0.03
    step_v2 = 3.0
    for _ in range(24):
        improved = False
        for dr1, dr2, dv in [
            (+step_r1, 0.0, 0.0),
            (-step_r1, 0.0, 0.0),
            (0.0, +step_r2, 0.0),
            (0.0, -step_r2, 0.0),
            (0.0, 0.0, +step_v2),
            (0.0, 0.0, -step_v2),
        ]:
            r1_c = min(max(r1 + dr1, r1_min), r1_max)
            r2_c = min(max(r2 + dr2, max(r1_c + 0.25, r2_min)), r2_max)
            v2_c = max(0.1, v2 + dv)
            cand = _eval_triplet_candidate_three_range_barrier_tail(
                b_mev=b_mev,
                r1_fm=r1_c,
                r2_fm=r2_c,
                v2_mev=v2_c,
                lambda_pi_fm=float(lambda_pi_fm),
                tail_len_over_lambda=float(tail_len_over_lambda),
                barrier_len_fraction=float(barrier_len_fraction),
                barrier_height_factor=float(barrier_height_factor),
                targets=targets,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            if cand is None:
                continue
            if float(cand["score"]) + 1e-12 < float(best["score"]):
                best = cand
                r1 = float(best["r1_fm"])
                r2 = float(best["r2_fm"])
                v2 = float(best["v2_mev"])
                improved = True
        if not improved:
            step_r1 *= 0.6
            step_r2 *= 0.6
            step_v2 *= 0.6
            if step_r1 < 0.004 and step_r2 < 0.004 and step_v2 < 0.35:
                break

    best = dict(best)
    best["pion_constraint"] = {
        "lambda_pi_fm": float(lambda_pi_fm),
        "R1_over_lambda_pi_bounds": [float(r1_over_lambda_min), float(r1_over_lambda_max)],
        "R2_over_lambda_pi_bounds": [float(r2_over_lambda_min), float(r2_over_lambda_max)],
    }
    best["tail_constraint"] = {
        "tail_len_over_lambda": float(tail_len_over_lambda),
        "barrier_len_fraction": float(barrier_len_fraction),
        "barrier_height_factor": float(barrier_height_factor),
        "tail_mean_factor_definition": "<V>/V(R2) = (λπ/L3) * (1 - exp(-L3/λπ))",
        "barrier_tail_mean_rule": "mean over [R2,R3]: (Lb*(+Vb) + Lt*(-Vt))/L3 = -V3_mean",
    }
    return best


def _fit_triplet_three_range_barrier_tail_pion_constrained_free_depth(
    *,
    b_mev: float,
    targets: dict[str, float],
    mu_mev: float,
    hbarc_mev_fm: float,
    lambda_pi_fm: float,
    tail_len_over_lambda: float = 1.0,
    barrier_len_fraction: float = 0.5,
    barrier_height_factor: float = 1.0,
    tail_depth_factor: float = 3.0,
    r1_over_lambda_min: float = 0.25,
    r1_over_lambda_max: float = 0.75,
    r2_over_lambda_min: float = 1.00,
    r2_over_lambda_max: float = 2.05,
) -> dict[str, object]:
    """
    Step 7.13.7 triplet fit:

    Same search structure as _fit_triplet_three_range_barrier_tail_pion_constrained(), but the barrier+tail split
    does *not* enforce the mean-preserving rule. Instead, the tail depth factor is specified independently.
    """
    if not (math.isfinite(lambda_pi_fm) and lambda_pi_fm > 0):
        raise ValueError("invalid lambda_pi_fm")
    if not (math.isfinite(tail_len_over_lambda) and tail_len_over_lambda > 0):
        raise ValueError("invalid tail_len_over_lambda")
    if not (math.isfinite(barrier_len_fraction) and 0.0 < barrier_len_fraction < 1.0):
        raise ValueError("invalid barrier_len_fraction")
    if not (math.isfinite(barrier_height_factor) and barrier_height_factor >= 0.0):
        raise ValueError("invalid barrier_height_factor")
    if not (math.isfinite(tail_depth_factor) and tail_depth_factor >= 0.0):
        raise ValueError("invalid tail_depth_factor")
    if not (r1_over_lambda_min > 0 and r1_over_lambda_max > r1_over_lambda_min):
        raise ValueError("invalid R1/λπ bounds")
    if not (r2_over_lambda_min > 0 and r2_over_lambda_max > r2_over_lambda_min):
        raise ValueError("invalid R2/λπ bounds")

    r1_min = float(r1_over_lambda_min) * float(lambda_pi_fm)
    r1_max = float(r1_over_lambda_max) * float(lambda_pi_fm)
    r2_min = float(r2_over_lambda_min) * float(lambda_pi_fm)
    r2_max = float(r2_over_lambda_max) * float(lambda_pi_fm)

    r1_grid = [r1_min + (r1_max - r1_min) * i / 16.0 for i in range(17)]
    r2_grid_all = [r2_min + (r2_max - r2_min) * i / 28.0 for i in range(29)]
    v2_grid = [2.0 + 2.0 * i for i in range(0, 30)]  # 2..60 MeV

    best: dict[str, object] | None = None
    for r1 in r1_grid:
        for r2 in r2_grid_all:
            if r2 <= r1 + 0.25:
                continue
            for v2 in v2_grid:
                cand = _eval_triplet_candidate_three_range_barrier_tail_free_depth(
                    b_mev=b_mev,
                    r1_fm=r1,
                    r2_fm=r2,
                    v2_mev=v2,
                    lambda_pi_fm=float(lambda_pi_fm),
                    tail_len_over_lambda=float(tail_len_over_lambda),
                    barrier_len_fraction=float(barrier_len_fraction),
                    barrier_height_factor=float(barrier_height_factor),
                    tail_depth_factor=float(tail_depth_factor),
                    targets=targets,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
                if cand is None:
                    continue
                if best is None or float(cand["score"]) < float(best["score"]):
                    best = cand

    if best is None:
        raise ValueError("no valid barrier+tail (free tail depth) candidate found under λπ constraints")

    r1 = float(best["r1_fm"])
    r2 = float(best["r2_fm"])
    v2 = float(best["v2_mev"])

    step_r1 = 0.03
    step_r2 = 0.03
    step_v2 = 3.0
    for _ in range(24):
        improved = False
        for dr1, dr2, dv in [
            (+step_r1, 0.0, 0.0),
            (-step_r1, 0.0, 0.0),
            (0.0, +step_r2, 0.0),
            (0.0, -step_r2, 0.0),
            (0.0, 0.0, +step_v2),
            (0.0, 0.0, -step_v2),
        ]:
            r1_c = min(max(r1 + dr1, r1_min), r1_max)
            r2_c = min(max(r2 + dr2, max(r1_c + 0.25, r2_min)), r2_max)
            v2_c = max(0.1, v2 + dv)
            cand = _eval_triplet_candidate_three_range_barrier_tail_free_depth(
                b_mev=b_mev,
                r1_fm=r1_c,
                r2_fm=r2_c,
                v2_mev=v2_c,
                lambda_pi_fm=float(lambda_pi_fm),
                tail_len_over_lambda=float(tail_len_over_lambda),
                barrier_len_fraction=float(barrier_len_fraction),
                barrier_height_factor=float(barrier_height_factor),
                tail_depth_factor=float(tail_depth_factor),
                targets=targets,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            if cand is None:
                continue
            if float(cand["score"]) + 1e-12 < float(best["score"]):
                best = cand
                r1 = float(best["r1_fm"])
                r2 = float(best["r2_fm"])
                v2 = float(best["v2_mev"])
                improved = True
        if not improved:
            step_r1 *= 0.6
            step_r2 *= 0.6
            step_v2 *= 0.6
            if step_r1 < 0.004 and step_r2 < 0.004 and step_v2 < 0.35:
                break

    best = dict(best)
    best["pion_constraint"] = {
        "lambda_pi_fm": float(lambda_pi_fm),
        "R1_over_lambda_pi_bounds": [float(r1_over_lambda_min), float(r1_over_lambda_max)],
        "R2_over_lambda_pi_bounds": [float(r2_over_lambda_min), float(r2_over_lambda_max)],
    }
    best["tail_constraint"] = {
        "tail_len_over_lambda": float(tail_len_over_lambda),
        "barrier_len_fraction": float(barrier_len_fraction),
        "barrier_height_factor": float(barrier_height_factor),
        "tail_depth_factor": float(tail_depth_factor),
        "tail_mean_factor_definition": "<V>/V(R2) = (λπ/L3) * (1 - exp(-L3/λπ))",
        "barrier_tail_rule": "Vb = barrier_height_factor*V3_mean, Vt = tail_depth_factor*V3_mean (no mean-preserving constraint)",
    }
    return best


def _fit_lambda_for_singlet(
    *,
    a_s_target_fm: float,
    r1_fm: float,
    r2_fm: float,
    v1_t_mev: float,
    v2_t_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, float]:
    """
    Fit a single scaling λ so that:
      V1_s = λ V1_t,  V2_s = λ V2_t
    matches the singlet scattering length a_s at k->0.
    """
    if not (math.isfinite(a_s_target_fm) and a_s_target_fm != 0):
        raise ValueError("invalid a_s target")

    def f(lam: float) -> float:
        a = _scattering_length_exact(
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            v1_mev=lam * v1_t_mev,
            v2_mev=lam * v2_t_mev,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        return a - a_s_target_fm

    lam_min = 0.05
    lam_max = 3.0
    n_scan = 500
    grid = [lam_min + (lam_max - lam_min) * i / (n_scan - 1) for i in range(n_scan)]
    best_lam = None
    best_abs = None
    prev_lam: float | None = None
    prev_f: float | None = None
    brackets: list[tuple[float, float, float]] = []
    for lam in grid:
        fl = f(lam)
        if not math.isfinite(fl):
            prev_lam = None
            prev_f = None
            continue
        if best_abs is None or abs(fl) < best_abs:
            best_abs = abs(fl)
            best_lam = lam
        if prev_lam is not None and prev_f is not None:
            if fl == 0 or (fl > 0) != (prev_f > 0):
                mid = 0.5 * (prev_lam + lam)
                brackets.append((prev_lam, lam, abs(mid - 0.9)))
        prev_lam = lam
        prev_f = fl

    if not brackets:
        if best_lam is None:
            raise ValueError("no usable λ samples")
        return {"lambda": float(best_lam), "note": "no sign-change; using best |Δa_s| grid point"}

    brackets.sort(key=lambda t: t[2])
    lo, hi, _ = brackets[0]
    f_lo = f(lo)
    f_hi = f(hi)
    if not ((f_lo > 0) != (f_hi > 0)):
        if best_lam is None:
            raise ValueError("invalid λ bracket and no fallback")
        return {"lambda": float(best_lam), "note": "invalid bracket; using best |Δa_s| grid point"}

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_mid == 0 or (hi - lo) < 1e-10:
            return {"lambda": float(mid), "note": "bisection solve on λ"}
        if (f_mid > 0) == (f_lo > 0):
            lo = mid
            f_lo = f_mid
        else:
            hi = mid
            f_hi = f_mid
    return {"lambda": float(0.5 * (lo + hi)), "note": "bisection solve on λ (max iter)"}


def _fit_v2s_for_singlet(
    *,
    a_s_target_fm: float,
    r1_fm: float,
    r2_fm: float,
    v1_s_mev: float,
    v2_hint_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    """
    Fit the outer depth V2_s so that the singlet scattering length matches a_s at k->0,
    keeping the same geometry (R1,R2) and fixing V1_s.

    To avoid numerical instability around poles (a -> ±∞), we solve for y(R2; E=0):
        a = R2 - 1/y  =>  y_target = 1/(R2 - a_target)
    and find V2_s such that y(V2_s) = y_target.
    """
    if not (math.isfinite(a_s_target_fm) and a_s_target_fm != 0.0):
        raise ValueError("invalid a_s target")
    if not (0 < r1_fm < r2_fm and v1_s_mev >= 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid geometry/params for singlet fit")

    denom = r2_fm - a_s_target_fm
    if not (math.isfinite(denom) and abs(denom) > 1e-12):
        raise ValueError("invalid a_s target for y_target")
    y_target = 1.0 / denom

    a_abs_max = 1e6

    def f(v2_s: float) -> float:
        y = _y_at_r2(
            e_mev=0.0,
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            v1_mev=v1_s_mev,
            v2_mev=v2_s,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        if not math.isfinite(y) or abs(y) < 1e-18:
            return float("nan")
        a = r2_fm - (1.0 / y)
        if not math.isfinite(a) or abs(a) > a_abs_max:
            return float("nan")
        return float(y - y_target)

    def scan_and_solve(*, v2_min: float, v2_max: float, tag: str) -> dict[str, object]:
        n_scan = 600
        grid = [v2_min + (v2_max - v2_min) * i / (n_scan - 1) for i in range(n_scan)]
        best_v2: float | None = None
        best_abs: float | None = None
        prev_v2: float | None = None
        prev_f: float | None = None
        brackets: list[tuple[float, float, float]] = []

        for v2 in grid:
            fv = f(v2)
            if not math.isfinite(fv):
                prev_v2 = None
                prev_f = None
                continue
            if best_abs is None or abs(fv) < best_abs:
                best_abs = abs(fv)
                best_v2 = v2
            if prev_v2 is not None and prev_f is not None:
                if fv == 0.0 or (fv > 0) != (prev_f > 0):
                    mid = 0.5 * (prev_v2 + v2)
                    brackets.append((prev_v2, v2, abs(mid - v2_hint_mev)))
            prev_v2 = v2
            prev_f = fv

        if not brackets:
            if best_v2 is None:
                raise ValueError(f"no usable V2_s samples ({tag})")
            return {
                "V2_s_MeV": float(best_v2),
                "method": "grid_best",
                "note": f"no sign-change; using best |Δy| grid point ({tag})",
                "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                "y_target_fm1": float(y_target),
            }

        brackets.sort(key=lambda t: t[2])
        lo, hi, _ = brackets[0]
        f_lo = f(lo)
        f_hi = f(hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi) and ((f_lo > 0) != (f_hi > 0))):
            if best_v2 is None:
                raise ValueError(f"invalid V2_s bracket and no fallback ({tag})")
            return {
                "V2_s_MeV": float(best_v2),
                "method": "grid_best",
                "note": f"invalid bracket; using best |Δy| grid point ({tag})",
                "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                "y_target_fm1": float(y_target),
            }

        for _ in range(90):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if not math.isfinite(f_mid):
                mid_a = (2.0 * lo + hi) / 3.0
                mid_b = (lo + 2.0 * hi) / 3.0
                fa = f(mid_a)
                fb = f(mid_b)
                if math.isfinite(fa):
                    mid, f_mid = mid_a, fa
                elif math.isfinite(fb):
                    mid, f_mid = mid_b, fb
                else:
                    break
            if f_mid == 0.0 or (hi - lo) < 1e-9:
                return {
                    "V2_s_MeV": float(mid),
                    "method": "bisection",
                    "note": f"bisection solve on V2_s ({tag})",
                    "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                    "y_target_fm1": float(y_target),
                }
            if (f_mid > 0) == (f_lo > 0):
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
                f_hi = f_mid

        if best_v2 is None:
            raise ValueError(f"bisection failed and no fallback ({tag})")
        return {
            "V2_s_MeV": float(best_v2),
            "method": "grid_best",
            "note": f"bisection failed; using best |Δy| grid point ({tag})",
            "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
            "y_target_fm1": float(y_target),
        }

    # First pass: enforce V2_s <= V1_s (keeps the intended two-step profile).
    strict_max = min(float(v1_s_mev), 500.0)
    fit = scan_and_solve(v2_min=0.0, v2_max=strict_max, tag="strict (V2<=V1)")
    v2_s = float(fit["V2_s_MeV"])
    a_check = _scattering_length_exact(
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        v1_mev=v1_s_mev,
        v2_mev=v2_s,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    if math.isfinite(a_check) and abs(a_check - a_s_target_fm) < 0.05 and str(fit.get("method")) == "bisection":
        fit["a_s_exact_fm"] = float(a_check)
        return fit

    # Second pass: allow V2_s > V1_s if needed to hit the target a_s.
    relaxed = scan_and_solve(v2_min=0.0, v2_max=500.0, tag="relaxed (allow V2>V1)")
    v2_s2 = float(relaxed["V2_s_MeV"])
    a_check2 = _scattering_length_exact(
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        v1_mev=v1_s_mev,
        v2_mev=v2_s2,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    relaxed["a_s_exact_fm"] = float(a_check2) if math.isfinite(a_check2) else float("nan")
    if v2_s2 > float(v1_s_mev) + 1e-9:
        relaxed["note_profile"] = "V2_s > V1_s (outer well deeper than inner); indicates this ansatz may be unnatural for singlet."
    return relaxed


def _fit_v2s_for_singlet_signed_two_range(
    *,
    a_s_target_fm: float,
    r1_fm: float,
    r2_fm: float,
    v1_s_mev: float,
    v2_hint_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    """
    Same as _fit_v2s_for_singlet(), but allows signed V2_s (outer barrier when V2_s<0).
    """
    if not (math.isfinite(a_s_target_fm) and a_s_target_fm != 0.0):
        raise ValueError("invalid a_s target")
    if not (0 < r1_fm < r2_fm and math.isfinite(v1_s_mev) and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid geometry/params for singlet fit")

    denom = r2_fm - a_s_target_fm
    if not (math.isfinite(denom) and abs(denom) > 1e-12):
        raise ValueError("invalid a_s target for y_target")
    y_target = 1.0 / denom

    a_abs_max = 1e6

    def f(v2_s: float) -> float:
        y = _y_at_r2_signed_two_range(
            e_mev=0.0,
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            v1_mev=v1_s_mev,
            v2_mev=v2_s,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        if not math.isfinite(y) or abs(y) < 1e-18:
            return float("nan")
        a = r2_fm - (1.0 / y)
        if not math.isfinite(a) or abs(a) > a_abs_max:
            return float("nan")
        return float(y - y_target)

    def scan_and_solve(*, v2_min: float, v2_max: float, tag: str) -> dict[str, object]:
        n_scan = 700
        grid = [v2_min + (v2_max - v2_min) * i / (n_scan - 1) for i in range(n_scan)]
        best_v2: float | None = None
        best_abs: float | None = None
        prev_v2: float | None = None
        prev_f: float | None = None
        brackets: list[tuple[float, float, float, float]] = []

        for v2 in grid:
            fv = f(v2)
            if not math.isfinite(fv):
                prev_v2 = None
                prev_f = None
                continue
            if best_abs is None or abs(fv) < best_abs:
                best_abs = abs(fv)
                best_v2 = v2
            if prev_v2 is not None and prev_f is not None:
                if fv == 0.0 or (fv > 0) != (prev_f > 0):
                    mid = 0.5 * (prev_v2 + v2)
                    brackets.append((prev_v2, v2, abs(mid), abs(mid - v2_hint_mev)))
            prev_v2 = v2
            prev_f = fv

        if not brackets:
            if best_v2 is None:
                raise ValueError(f"no usable V2_s samples ({tag})")
            return {
                "V2_s_MeV": float(best_v2),
                "method": "grid_best",
                "note": f"no sign-change; using best |Δy| grid point ({tag})",
                "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                "y_target_fm1": float(y_target),
            }

        brackets.sort(key=lambda t: (t[2], t[3]))
        lo, hi, _, _ = brackets[0]
        f_lo = f(lo)
        f_hi = f(hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi) and ((f_lo > 0) != (f_hi > 0))):
            if best_v2 is None:
                raise ValueError(f"invalid V2_s bracket and no fallback ({tag})")
            return {
                "V2_s_MeV": float(best_v2),
                "method": "grid_best",
                "note": f"invalid bracket; using best |Δy| grid point ({tag})",
                "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                "y_target_fm1": float(y_target),
            }

        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if not math.isfinite(f_mid):
                mid_a = (2.0 * lo + hi) / 3.0
                mid_b = (lo + 2.0 * hi) / 3.0
                fa = f(mid_a)
                fb = f(mid_b)
                if math.isfinite(fa):
                    mid, f_mid = mid_a, fa
                elif math.isfinite(fb):
                    mid, f_mid = mid_b, fb
                else:
                    break
            if f_mid == 0.0 or (hi - lo) < 1e-9:
                return {
                    "V2_s_MeV": float(mid),
                    "method": "bisection",
                    "note": f"bisection solve on V2_s ({tag})",
                    "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                    "y_target_fm1": float(y_target),
                }
            if (f_mid > 0) == (f_lo > 0):
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
                f_hi = f_mid

        if best_v2 is None:
            raise ValueError(f"bisection failed and no fallback ({tag})")
        return {
            "V2_s_MeV": float(best_v2),
            "method": "grid_best",
            "note": f"bisection failed; using best |Δy| grid point ({tag})",
            "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
            "y_target_fm1": float(y_target),
        }

    # First pass: keep |V2_s| <= V1_s as a minimal "shape" bound.
    strict_max = min(500.0, abs(float(v1_s_mev)))
    fit = scan_and_solve(v2_min=-strict_max, v2_max=+strict_max, tag="bounded (|V2|<=V1)")
    v2_s = float(fit["V2_s_MeV"])
    a_check = _scattering_length_exact_signed_two_range(
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        v1_mev=v1_s_mev,
        v2_mev=v2_s,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    if math.isfinite(a_check) and abs(a_check - a_s_target_fm) < 0.05 and str(fit.get("method")) == "bisection":
        fit["a_s_exact_fm"] = float(a_check)
        return fit

    # Second pass: widen if needed.
    relaxed = scan_and_solve(v2_min=-700.0, v2_max=700.0, tag="wide (allow barrier)")
    v2_s2 = float(relaxed["V2_s_MeV"])
    a_check2 = _scattering_length_exact_signed_two_range(
        r1_fm=r1_fm,
        r2_fm=r2_fm,
        v1_mev=v1_s_mev,
        v2_mev=v2_s2,
        mu_mev=mu_mev,
        hbarc_mev_fm=hbarc_mev_fm,
    )
    relaxed["a_s_exact_fm"] = float(a_check2) if math.isfinite(a_check2) else float("nan")
    if v2_s2 < -1e-12:
        relaxed["note_profile"] = "V2_s < 0 (outer repulsive barrier)."
    elif v2_s2 > float(v1_s_mev) + 1e-9:
        relaxed["note_profile"] = "V2_s > V1_s (outer well deeper than inner); indicates this ansatz may be unnatural for singlet."
    return relaxed


def _fit_v2s_for_singlet_three_range_tail(
    *,
    a_s_target_fm: float,
    r1_fm: float,
    r2_fm: float,
    tail_len_fm: float,
    v1_s_mev: float,
    v3_tail_mev: float,
    v2_hint_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
    allow_signed_v2: bool = True,
) -> dict[str, object]:
    """
    Solve V2_s from a_s (k->0) for a 3-range ansatz with a fixed tail segment:

      V(r) = -V1_s (0<=r<R1),
             -V2_s (R1<=r<R2),  (signed allowed if allow_signed_v2=True),
             -V3_tail (R2<=r<R3),  R3=R2+L3,  with fixed (V3_tail,L3),
             0 (r>R3).

    This keeps the singlet fit degrees at 2: (V1_s,V2_s) for (a_s,r_s), leaving v2s as a prediction.
    """
    if not (math.isfinite(a_s_target_fm) and a_s_target_fm != 0.0):
        raise ValueError("invalid a_s target")
    if not (
        0 < r1_fm < r2_fm
        and math.isfinite(tail_len_fm)
        and tail_len_fm > 0
        and math.isfinite(v1_s_mev)
        and mu_mev > 0
        and hbarc_mev_fm > 0
    ):
        raise ValueError("invalid geometry/params for 3-range singlet V2 fit")

    r3_fm = float(r2_fm) + float(tail_len_fm)
    denom = r3_fm - a_s_target_fm
    if not (math.isfinite(denom) and abs(denom) > 1e-12):
        raise ValueError("invalid a_s target for y_target")
    y_target = 1.0 / denom

    a_abs_max = 1e6

    def f(v2_s: float) -> float:
        segs = [
            (float(r1_fm), -float(v1_s_mev)),
            (float(r2_fm - r1_fm), -float(v2_s)),
            (float(tail_len_fm), -float(v3_tail_mev)),
        ]
        y = _y_at_r_end_segments(e_mev=0.0, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        if not math.isfinite(y) or abs(y) < 1e-18:
            return float("nan")
        a = r3_fm - (1.0 / y)
        if not math.isfinite(a) or abs(a) > a_abs_max:
            return float("nan")
        return float(y - y_target)

    def scan_and_solve(*, v2_min: float, v2_max: float, tag: str) -> dict[str, object]:
        n_scan = 800
        grid = [v2_min + (v2_max - v2_min) * i / (n_scan - 1) for i in range(n_scan)]
        best_v2: float | None = None
        best_abs: float | None = None
        prev_v2: float | None = None
        prev_f: float | None = None
        brackets: list[tuple[float, float, float, float]] = []

        for v2 in grid:
            fv = f(v2)
            if not math.isfinite(fv):
                prev_v2 = None
                prev_f = None
                continue
            if best_abs is None or abs(fv) < best_abs:
                best_abs = abs(fv)
                best_v2 = v2
            if prev_v2 is not None and prev_f is not None:
                if fv == 0.0 or (fv > 0) != (prev_f > 0):
                    mid = 0.5 * (prev_v2 + v2)
                    brackets.append((prev_v2, v2, abs(mid), abs(mid - v2_hint_mev)))
            prev_v2 = v2
            prev_f = fv

        if not brackets:
            if best_v2 is None:
                raise ValueError(f"no usable V2_s samples ({tag})")
            return {
                "V2_s_MeV": float(best_v2),
                "method": "grid_best",
                "note": f"no sign-change; using best |Δy| grid point ({tag})",
                "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                "y_target_fm1": float(y_target),
            }

        brackets.sort(key=lambda t: (t[2], t[3]))
        lo, hi, _, _ = brackets[0]
        f_lo = f(lo)
        f_hi = f(hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi) and ((f_lo > 0) != (f_hi > 0))):
            if best_v2 is None:
                raise ValueError(f"invalid V2_s bracket and no fallback ({tag})")
            return {
                "V2_s_MeV": float(best_v2),
                "method": "grid_best",
                "note": f"invalid bracket; using best |Δy| grid point ({tag})",
                "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                "y_target_fm1": float(y_target),
            }

        for _ in range(120):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if not math.isfinite(f_mid):
                mid_a = (2.0 * lo + hi) / 3.0
                mid_b = (lo + 2.0 * hi) / 3.0
                fa = f(mid_a)
                fb = f(mid_b)
                if math.isfinite(fa):
                    mid, f_mid = mid_a, fa
                elif math.isfinite(fb):
                    mid, f_mid = mid_b, fb
                else:
                    break
            if f_mid == 0.0 or (hi - lo) < 1e-9:
                return {
                    "V2_s_MeV": float(mid),
                    "method": "bisection",
                    "note": f"bisection solve on V2_s ({tag})",
                    "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                    "y_target_fm1": float(y_target),
                }
            if (f_mid > 0) == (f_lo > 0):
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
                f_hi = f_mid

        if best_v2 is None:
            raise ValueError(f"bisection failed and no fallback ({tag})")
        return {
            "V2_s_MeV": float(best_v2),
            "method": "grid_best",
            "note": f"bisection failed; using best |Δy| grid point ({tag})",
            "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
            "y_target_fm1": float(y_target),
        }

    if allow_signed_v2:
        strict_max = min(700.0, max(50.0, abs(float(v1_s_mev))))
        fit = scan_and_solve(v2_min=-strict_max, v2_max=+strict_max, tag="bounded (|V2|<=~V1)")
    else:
        strict_max = min(float(v1_s_mev), 700.0)
        fit = scan_and_solve(v2_min=0.0, v2_max=strict_max, tag="strict (V2>=0)")

    v2_s = float(fit["V2_s_MeV"])
    segs_chk = [
        (float(r1_fm), -float(v1_s_mev)),
        (float(r2_fm - r1_fm), -float(v2_s)),
        (float(tail_len_fm), -float(v3_tail_mev)),
    ]
    a_check = _scattering_length_exact_segments(
        r_end_fm=r3_fm, segments=segs_chk, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
    )
    if math.isfinite(a_check) and abs(a_check - a_s_target_fm) < 0.05 and str(fit.get("method")) == "bisection":
        fit["a_s_exact_fm"] = float(a_check)
        return fit

    relaxed = (
        scan_and_solve(v2_min=-900.0, v2_max=900.0, tag="wide (allow barrier)")
        if allow_signed_v2
        else scan_and_solve(v2_min=0.0, v2_max=900.0, tag="wide (allow V2>V1)")
    )
    v2_s2 = float(relaxed["V2_s_MeV"])
    segs_chk2 = [
        (float(r1_fm), -float(v1_s_mev)),
        (float(r2_fm - r1_fm), -float(v2_s2)),
        (float(tail_len_fm), -float(v3_tail_mev)),
    ]
    a_check2 = _scattering_length_exact_segments(
        r_end_fm=r3_fm, segments=segs_chk2, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
    )
    relaxed["a_s_exact_fm"] = float(a_check2) if math.isfinite(a_check2) else float("nan")
    if allow_signed_v2 and v2_s2 < -1e-12:
        relaxed["note_profile"] = "V2_s < 0 (outer repulsive barrier)."
    elif (not allow_signed_v2) and v2_s2 > float(v1_s_mev) + 1e-9:
        relaxed["note_profile"] = "V2_s > V1_s (outer well deeper than inner); indicates this ansatz may be unnatural for singlet."
    return relaxed


def _fit_v2s_for_singlet_three_range_barrier_tail(
    *,
    a_s_target_fm: float,
    r1_fm: float,
    r2_fm: float,
    cfg: dict[str, float],
    v1_s_mev: float,
    v2_hint_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    """
    Step 7.13.5: solve V2_s from a_s (k->0) for a split barrier+tail region beyond R2.

      V(r) = -V1_s (0<=r<R1),
             -V2_s (R1<=r<R2),  (V2_s>=0),
             +Vb   (R2<=r<Rb),
             -Vt   (Rb<=r<R3),  R3=R2+L3_total,
              0    (r>R3),

    with barrier/tail parameters determined from V2_s via cfg (mean-preserving rule).
    This keeps the singlet fit degrees at 2: (V1_s,V2_s) for (a_s,r_s), leaving v2s as a prediction.
    """
    if not (math.isfinite(a_s_target_fm) and a_s_target_fm != 0.0):
        raise ValueError("invalid a_s target")
    if not (0 < r1_fm < r2_fm and math.isfinite(v1_s_mev) and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid geometry/params for barrier+tail singlet V2 fit")
    for k in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
        if k not in cfg or not math.isfinite(float(cfg[k])):
            raise ValueError(f"invalid cfg missing {k}")

    l3_total_fm = float(cfg["L3_total_fm"])
    lb_fm = float(cfg["Lb_fm"])
    lt_fm = float(cfg["Lt_fm"])
    barrier_coeff = float(cfg["barrier_coeff"])
    tail_depth_coeff = float(cfg["tail_depth_coeff"])

    r3_fm = float(r2_fm) + float(l3_total_fm)
    denom = r3_fm - a_s_target_fm
    if not (math.isfinite(denom) and abs(denom) > 1e-12):
        raise ValueError("invalid a_s target for y_target")
    y_target = 1.0 / denom

    a_abs_max = 1e6

    def f(v2_s: float) -> float:
        if not (math.isfinite(v2_s) and v2_s >= 0):
            return float("nan")
        vb = float(v2_s) * float(barrier_coeff)
        vt = float(v2_s) * float(tail_depth_coeff)
        if not (math.isfinite(vb) and math.isfinite(vt) and vb >= 0 and vt >= 0):
            return float("nan")
        segs = [
            (float(r1_fm), -float(v1_s_mev)),
            (float(r2_fm - r1_fm), -float(v2_s)),
            (float(lb_fm), +float(vb)),
            (float(lt_fm), -float(vt)),
        ]
        y = _y_at_r_end_segments(e_mev=0.0, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        if not math.isfinite(y) or abs(y) < 1e-18:
            return float("nan")
        a = r3_fm - (1.0 / y)
        if not math.isfinite(a) or abs(a) > a_abs_max:
            return float("nan")
        return float(y - y_target)

    def scan_and_solve(*, v2_min: float, v2_max: float, tag: str) -> dict[str, object]:
        n_scan = 900
        grid = [v2_min + (v2_max - v2_min) * i / (n_scan - 1) for i in range(n_scan)]
        best_v2: float | None = None
        best_abs: float | None = None
        prev_v2: float | None = None
        prev_f: float | None = None
        brackets: list[tuple[float, float, float, float]] = []

        for v2 in grid:
            fv = f(v2)
            if not math.isfinite(fv):
                prev_v2 = None
                prev_f = None
                continue
            if best_abs is None or abs(fv) < best_abs:
                best_abs = abs(fv)
                best_v2 = v2
            if prev_v2 is not None and prev_f is not None:
                if fv == 0.0 or (fv > 0) != (prev_f > 0):
                    mid = 0.5 * (prev_v2 + v2)
                    brackets.append((prev_v2, v2, abs(mid), abs(mid - v2_hint_mev)))
            prev_v2 = v2
            prev_f = fv

        if not brackets:
            if best_v2 is None:
                raise ValueError(f"no usable V2_s samples ({tag})")
            return {
                "V2_s_MeV": float(best_v2),
                "method": "grid_best",
                "note": f"no sign-change; using best |Δy| grid point ({tag})",
                "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                "y_target_fm1": float(y_target),
            }

        brackets.sort(key=lambda t: (t[2], t[3]))
        lo, hi, _, _ = brackets[0]
        f_lo = f(lo)
        f_hi = f(hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi) and ((f_lo > 0) != (f_hi > 0))):
            if best_v2 is None:
                raise ValueError(f"invalid V2_s bracket and no fallback ({tag})")
            return {
                "V2_s_MeV": float(best_v2),
                "method": "grid_best",
                "note": f"invalid bracket; using best |Δy| grid point ({tag})",
                "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                "y_target_fm1": float(y_target),
            }

        for _ in range(90):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if not math.isfinite(f_mid):
                mid_a = (2.0 * lo + hi) / 3.0
                mid_b = (lo + 2.0 * hi) / 3.0
                fa = f(mid_a)
                fb = f(mid_b)
                if math.isfinite(fa):
                    mid, f_mid = mid_a, fa
                elif math.isfinite(fb):
                    mid, f_mid = mid_b, fb
                else:
                    break
            if f_mid == 0.0 or (hi - lo) < 1e-9:
                return {
                    "V2_s_MeV": float(mid),
                    "method": "bisection",
                    "note": f"bisection solve on V2_s ({tag})",
                    "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                    "y_target_fm1": float(y_target),
                }
            if (f_mid > 0) == (f_lo > 0):
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
                f_hi = f_mid

        if best_v2 is None:
            raise ValueError(f"bisection failed and no fallback ({tag})")
        return {
            "V2_s_MeV": float(best_v2),
            "method": "grid_best",
            "note": f"bisection failed; using best |Δy| grid point ({tag})",
            "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
            "y_target_fm1": float(y_target),
        }

    strict_max = min(float(v1_s_mev), 700.0)
    fit = scan_and_solve(v2_min=0.0, v2_max=strict_max, tag="strict (V2>=0, V2<=V1)")

    v2_s = float(fit["V2_s_MeV"])
    vb_chk = v2_s * barrier_coeff
    vt_chk = v2_s * tail_depth_coeff
    segs_chk = [
        (float(r1_fm), -float(v1_s_mev)),
        (float(r2_fm - r1_fm), -float(v2_s)),
        (float(lb_fm), +float(vb_chk)),
        (float(lt_fm), -float(vt_chk)),
    ]
    a_check = _scattering_length_exact_segments(
        r_end_fm=r3_fm, segments=segs_chk, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
    )
    if math.isfinite(a_check) and abs(a_check - a_s_target_fm) < 0.05 and str(fit.get("method")) == "bisection":
        fit["a_s_exact_fm"] = float(a_check)
        return fit

    relaxed = scan_and_solve(v2_min=0.0, v2_max=900.0, tag="wide (allow V2>V1)")
    v2_s2 = float(relaxed["V2_s_MeV"])
    vb_chk2 = v2_s2 * barrier_coeff
    vt_chk2 = v2_s2 * tail_depth_coeff
    segs_chk2 = [
        (float(r1_fm), -float(v1_s_mev)),
        (float(r2_fm - r1_fm), -float(v2_s2)),
        (float(lb_fm), +float(vb_chk2)),
        (float(lt_fm), -float(vt_chk2)),
    ]
    a_check2 = _scattering_length_exact_segments(
        r_end_fm=r3_fm, segments=segs_chk2, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
    )
    relaxed["a_s_exact_fm"] = float(a_check2) if math.isfinite(a_check2) else float("nan")
    if v2_s2 > float(v1_s_mev) + 1e-9:
        relaxed["note_profile"] = "V2_s > V1_s (outer well deeper than inner); indicates this ansatz may be unnatural for singlet."
    return relaxed


def _fit_v1v2_for_singlet_by_a_and_r(
    *,
    a_s_target_fm: float,
    r_s_target_fm: float,
    r1_fm: float,
    r2_fm: float,
    v1_hint_mev: float,
    v2_hint_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
    allow_signed_v2: bool = False,
) -> dict[str, object]:
    """
    Fit (V1_s, V2_s) to match singlet (a_s, r_s) on the same 2-range geometry (R1,R2).

    Implementation:
      - For each candidate V1_s, solve V2_s from a_s(k->0) using the stable y-target approach.
      - Compute r_eff from a small-k ERE fit, and solve V1_s by bisection on (r_eff - r_target).

    Root selection (if multiple brackets exist):
      - Prefer solutions with V2_s <= V1_s (keeps the intended 2-step profile).
      - Among them, choose the shallowest solution (minimal V1_s), with triplet-proximity only as a weak tie-breaker.
    """
    if not (math.isfinite(a_s_target_fm) and math.isfinite(r_s_target_fm)):
        raise ValueError("invalid singlet targets")
    if not (0 < r1_fm < r2_fm and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid geometry/params for singlet fit")

    v1_lo = 0.1
    v1_hi = max(500.0, 1.6 * float(v1_hint_mev))
    v1_hi = min(v1_hi, 900.0)

    tol_a = 0.05  # fm
    tol_r = 2e-3  # fm
    max_iter = 70

    def eval_v1(v1_s: float) -> dict[str, object] | None:
        if not (math.isfinite(v1_s) and v1_s >= 0):
            return None
        try:
            v2_fit = (
                _fit_v2s_for_singlet_signed_two_range(
                    a_s_target_fm=a_s_target_fm,
                    r1_fm=r1_fm,
                    r2_fm=r2_fm,
                    v1_s_mev=v1_s,
                    v2_hint_mev=v2_hint_mev,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
                if allow_signed_v2
                else _fit_v2s_for_singlet(
                    a_s_target_fm=a_s_target_fm,
                    r1_fm=r1_fm,
                    r2_fm=r2_fm,
                    v1_s_mev=v1_s,
                    v2_hint_mev=v2_hint_mev,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
            )
        except Exception:
            return None

        v2_s = float(v2_fit.get("V2_s_MeV", float("nan")))
        if not math.isfinite(v2_s):
            return None
        if not allow_signed_v2 and v2_s < 0:
            return None

        try:
            a_exact = (
                _scattering_length_exact_signed_two_range(
                    r1_fm=r1_fm,
                    r2_fm=r2_fm,
                    v1_mev=v1_s,
                    v2_mev=v2_s,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
                if allow_signed_v2
                else _scattering_length_exact(
                    r1_fm=r1_fm,
                    r2_fm=r2_fm,
                    v1_mev=v1_s,
                    v2_mev=v2_s,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
            )
        except Exception:
            return None
        if not (math.isfinite(a_exact) and abs(a_exact - a_s_target_fm) <= tol_a):
            # Discard candidates that do not actually match the a_s target.
            return None
        try:
            ere = (
                _fit_kcot_ere_signed_two_range(
                    r1_fm=r1_fm,
                    r2_fm=r2_fm,
                    v1_mev=v1_s,
                    v2_mev=v2_s,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
                if allow_signed_v2
                else _fit_kcot_ere(
                    r1_fm=r1_fm,
                    r2_fm=r2_fm,
                    v1_mev=v1_s,
                    v2_mev=v2_s,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
            )
        except Exception:
            return None

        r_eff = float(ere.get("r_eff_fm", float("nan")))
        if not math.isfinite(r_eff):
            return None

        return {
            "V1_s_MeV": float(v1_s),
            "V2_s_MeV": float(v2_s),
            "a_s_exact_fm": float(a_exact) if math.isfinite(a_exact) else float("nan"),
            "ere": ere,
            "g_r_fm": float(r_eff - r_s_target_fm),
            "fit_v2": v2_fit,
        }

    # Scan V1_s to find sign-change brackets for g(V1)=r_eff(V1)-r_target.
    n_scan = 260
    v1_grid = [v1_lo + (v1_hi - v1_lo) * i / (n_scan - 1) for i in range(n_scan)]
    pts: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_abs = None

    for v1 in v1_grid:
        e = eval_v1(v1)
        if e is None:
            continue
        g = float(e["g_r_fm"])
        if not math.isfinite(g):
            continue
        pts.append(e)
        if best_abs is None or abs(g) < best_abs:
            best_abs = abs(g)
            best = e

    if len(pts) < 2:
        if best is None:
            raise ValueError("no usable V1_s samples")
        best = dict(best)
        best.update({"fit_method": "grid_best", "note_fit": "insufficient valid samples for bracketing"})
        best["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
        return best

    brackets: list[tuple[dict[str, object], dict[str, object], float]] = []
    for a, b in zip(pts[:-1], pts[1:]):
        ga = float(a["g_r_fm"])
        gb = float(b["g_r_fm"])
        if ga == 0.0 or (ga > 0) != (gb > 0):
            mid = 0.5 * (float(a["V1_s_MeV"]) + float(b["V1_s_MeV"]))
            brackets.append((a, b, float(mid)))

    if not brackets:
        if best is None:
            raise ValueError("no V1_s sign-change bracket and no fallback")
        best = dict(best)
        best.update({"fit_method": "grid_best", "note_fit": "no sign-change bracket; using best |Δr_s| grid point"})
        best["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
        return best

    # Solve each bracket and pick the most "natural" one.
    def penalty(sol: dict[str, object]) -> float:
        v1 = float(sol["V1_s_MeV"])
        v2 = float(sol["V2_s_MeV"])
        p = v1
        if (not allow_signed_v2) and v2 > v1 + 1e-9:
            p += 1e6
        if allow_signed_v2:
            p += 0.05 * abs(v2)
        p += 1e-3 * abs(v1 - float(v1_hint_mev))
        if allow_signed_v2:
            p += 1e-3 * abs(v2 - float(v2_hint_mev))
        return float(p)

    best_sol: dict[str, object] | None = None
    best_pen = None

    brackets.sort(key=lambda t: t[2])
    for lo_e, hi_e, _ in brackets[:8]:
        lo = float(lo_e["V1_s_MeV"])
        hi = float(hi_e["V1_s_MeV"])
        g_lo = float(lo_e["g_r_fm"])
        g_hi = float(hi_e["g_r_fm"])
        if not ((g_lo > 0) != (g_hi > 0)):
            continue

        sol_lo: dict[str, object] = lo_e
        sol_hi: dict[str, object] = hi_e
        cand_sol: dict[str, object] | None = None
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            sol_mid = eval_v1(mid)
            if sol_mid is None:
                mid_a = (2.0 * lo + hi) / 3.0
                mid_b = (lo + 2.0 * hi) / 3.0
                sol_a = eval_v1(mid_a)
                sol_b = eval_v1(mid_b)
                sol_mid = sol_a if sol_a is not None else sol_b
                if sol_mid is None:
                    break
                mid = float(sol_mid["V1_s_MeV"])

            g_mid = float(sol_mid["g_r_fm"])
            if not math.isfinite(g_mid):
                break
            if abs(g_mid) < tol_r or (hi - lo) < 1e-6:
                cand_sol = dict(sol_mid)
                cand_sol.update(
                    {"fit_method": "bisection", "note_fit": "solve V1_s by r_s (bisection) with V2_s(a_s) inner solve"}
                )
                cand_sol["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
                cand_sol["bisection"] = {"tol_a_fm": float(tol_a), "tol_r_fm": float(tol_r), "max_iter": int(max_iter)}
                break
            if (g_mid > 0) == (g_lo > 0):
                lo = mid
                g_lo = g_mid
                sol_lo = sol_mid
            else:
                hi = mid
                g_hi = g_mid
                sol_hi = sol_mid

        if cand_sol is None:
            # Fall back to whichever endpoint is closer in |g|.
            cand = sol_lo if abs(float(sol_lo["g_r_fm"])) < abs(float(sol_hi["g_r_fm"])) else sol_hi
            cand_sol = dict(cand)
            cand_sol.update({"fit_method": "bracket_best", "note_fit": "bisection did not converge; using best endpoint"})
            cand_sol["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
            cand_sol["bisection"] = {"tol_a_fm": float(tol_a), "tol_r_fm": float(tol_r), "max_iter": int(max_iter)}

        pen = penalty(cand_sol)
        if best_pen is None or pen < best_pen:
            best_pen = pen
            best_sol = cand_sol

    if best_sol is None:
        if best is None:
            raise ValueError("failed to solve V1_s root and no fallback")
        best_sol = dict(best)
        best_sol.update({"fit_method": "grid_best", "note_fit": "root solve failed; using best |Δr_s| grid point"})
        best_sol["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}

    v1_final = float(best_sol["V1_s_MeV"])
    v2_final = float(best_sol["V2_s_MeV"])
    if allow_signed_v2 and v2_final < -1e-12:
        best_sol["note_profile"] = "V2_s < 0 (outer repulsive barrier)."
    elif v2_final > v1_final + 1e-9:
        best_sol["note_profile"] = "V2_s > V1_s (outer well deeper than inner); indicates this ansatz may be unnatural for singlet."
    return best_sol


def _fit_v1v2_for_singlet_by_a_and_r_three_range_tail(
    *,
    a_s_target_fm: float,
    r_s_target_fm: float,
    r1_fm: float,
    r2_fm: float,
    tail_len_fm: float,
    v3_tail_mev: float,
    v1_hint_mev: float,
    v2_hint_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
    allow_signed_v2: bool = True,
) -> dict[str, object]:
    """
    Fit (V1_s,V2_s) to match singlet (a_s,r_s) on the 3-range tail ansatz, with a fixed tail segment
    (V3_tail,L3) shared from the triplet fit.
    """
    if not (math.isfinite(a_s_target_fm) and math.isfinite(r_s_target_fm)):
        raise ValueError("invalid singlet targets")
    if not (0 < r1_fm < r2_fm and math.isfinite(tail_len_fm) and tail_len_fm > 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid geometry/params for 3-range singlet fit")

    v1_lo = 0.1
    v1_hi = max(500.0, 1.6 * float(v1_hint_mev))
    v1_hi = min(v1_hi, 900.0)

    tol_a = 0.05  # fm
    tol_r = 2e-3  # fm
    max_iter = 70

    r3_fm = float(r2_fm) + float(tail_len_fm)

    def eval_v1(v1_s: float) -> dict[str, object] | None:
        if not (math.isfinite(v1_s) and v1_s >= 0):
            return None
        try:
            v2_fit = _fit_v2s_for_singlet_three_range_tail(
                a_s_target_fm=a_s_target_fm,
                r1_fm=r1_fm,
                r2_fm=r2_fm,
                tail_len_fm=tail_len_fm,
                v1_s_mev=v1_s,
                v3_tail_mev=v3_tail_mev,
                v2_hint_mev=v2_hint_mev,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
                allow_signed_v2=allow_signed_v2,
            )
        except Exception:
            return None

        v2_s = float(v2_fit.get("V2_s_MeV", float("nan")))
        if not math.isfinite(v2_s):
            return None
        if not allow_signed_v2 and v2_s < 0:
            return None

        segs = [
            (float(r1_fm), -float(v1_s)),
            (float(r2_fm - r1_fm), -float(v2_s)),
            (float(tail_len_fm), -float(v3_tail_mev)),
        ]
        try:
            a_exact = _scattering_length_exact_segments(
                r_end_fm=r3_fm, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
            )
        except Exception:
            return None
        if not (math.isfinite(a_exact) and abs(a_exact - a_s_target_fm) <= tol_a):
            return None
        try:
            ere = _fit_kcot_ere_segments(r_end_fm=r3_fm, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        except Exception:
            return None
        r_eff = float(ere.get("r_eff_fm", float("nan")))
        if not math.isfinite(r_eff):
            return None

        return {
            "V1_s_MeV": float(v1_s),
            "V2_s_MeV": float(v2_s),
            "V3_tail_MeV": float(v3_tail_mev),
            "a_s_exact_fm": float(a_exact),
            "ere": ere,
            "g_r_fm": float(r_eff - r_s_target_fm),
            "fit_v2": v2_fit,
        }

    # Scan V1_s to find sign-change brackets for g(V1)=r_eff(V1)-r_target.
    n_scan = 260
    v1_grid = [v1_lo + (v1_hi - v1_lo) * i / (n_scan - 1) for i in range(n_scan)]
    pts: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_abs = None

    for v1 in v1_grid:
        e = eval_v1(v1)
        if e is None:
            continue
        g = float(e["g_r_fm"])
        if not math.isfinite(g):
            continue
        pts.append(e)
        if best_abs is None or abs(g) < best_abs:
            best_abs = abs(g)
            best = e

    if len(pts) < 2:
        if best is None:
            raise ValueError("no usable V1_s samples (3-range tail)")
        best = dict(best)
        best.update({"fit_method": "grid_best", "note_fit": "insufficient valid samples for bracketing (3-range tail)"})
        best["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
        best["tail"] = {"R3_fm": float(r3_fm), "L3_fm": float(tail_len_fm), "V3_tail_MeV": float(v3_tail_mev)}
        return best

    brackets: list[tuple[dict[str, object], dict[str, object], float]] = []
    for a, b in zip(pts[:-1], pts[1:]):
        ga = float(a["g_r_fm"])
        gb = float(b["g_r_fm"])
        if ga == 0.0 or (ga > 0) != (gb > 0):
            mid = 0.5 * (float(a["V1_s_MeV"]) + float(b["V1_s_MeV"]))
            brackets.append((a, b, float(mid)))

    if not brackets:
        if best is None:
            raise ValueError("no V1_s sign-change bracket and no fallback (3-range tail)")
        best = dict(best)
        best.update({"fit_method": "grid_best", "note_fit": "no sign-change bracket; using best |Δr_s| (3-range tail)"})
        best["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
        best["tail"] = {"R3_fm": float(r3_fm), "L3_fm": float(tail_len_fm), "V3_tail_MeV": float(v3_tail_mev)}
        return best

    def penalty(sol: dict[str, object]) -> float:
        v1 = float(sol["V1_s_MeV"])
        v2 = float(sol["V2_s_MeV"])
        p = v1
        if allow_signed_v2:
            p += 0.05 * abs(v2)
        else:
            if v2 > v1 + 1e-9:
                p += 1e6
        p += 1e-3 * abs(v1 - float(v1_hint_mev))
        p += 1e-3 * abs(v2 - float(v2_hint_mev))
        return float(p)

    best_sol: dict[str, object] | None = None
    best_pen = None

    brackets.sort(key=lambda t: t[2])
    for lo_e, hi_e, _ in brackets[:8]:
        lo = float(lo_e["V1_s_MeV"])
        hi = float(hi_e["V1_s_MeV"])
        g_lo = float(lo_e["g_r_fm"])
        g_hi = float(hi_e["g_r_fm"])
        if not ((g_lo > 0) != (g_hi > 0)):
            continue

        sol_lo: dict[str, object] = lo_e
        sol_hi: dict[str, object] = hi_e
        cand_sol: dict[str, object] | None = None
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            sol_mid = eval_v1(mid)
            if sol_mid is None:
                mid_a = (2.0 * lo + hi) / 3.0
                mid_b = (lo + 2.0 * hi) / 3.0
                sol_a = eval_v1(mid_a)
                sol_b = eval_v1(mid_b)
                sol_mid = sol_a if sol_a is not None else sol_b
                if sol_mid is None:
                    break
                mid = float(sol_mid["V1_s_MeV"])

            g_mid = float(sol_mid["g_r_fm"])
            if not math.isfinite(g_mid):
                break
            if abs(g_mid) < tol_r or (hi - lo) < 1e-6:
                cand_sol = dict(sol_mid)
                cand_sol.update({"fit_method": "bisection", "note_fit": "solve V1_s by r_s (3-range tail); V2_s(a_s) inner solve"})
                cand_sol["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
                cand_sol["bisection"] = {"tol_a_fm": float(tol_a), "tol_r_fm": float(tol_r), "max_iter": int(max_iter)}
                break
            if (g_mid > 0) == (g_lo > 0):
                lo = mid
                g_lo = g_mid
                sol_lo = sol_mid
            else:
                hi = mid
                g_hi = g_mid
                sol_hi = sol_mid

        if cand_sol is None:
            cand = sol_lo if abs(float(sol_lo["g_r_fm"])) < abs(float(sol_hi["g_r_fm"])) else sol_hi
            cand_sol = dict(cand)
            cand_sol.update({"fit_method": "bracket_best", "note_fit": "bisection did not converge; using best endpoint (3-range tail)"})
            cand_sol["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
            cand_sol["bisection"] = {"tol_a_fm": float(tol_a), "tol_r_fm": float(tol_r), "max_iter": int(max_iter)}

        pen = penalty(cand_sol)
        if best_pen is None or pen < best_pen:
            best_pen = pen
            best_sol = cand_sol

    if best_sol is None:
        if best is None:
            raise ValueError("failed to solve V1_s root and no fallback (3-range tail)")
        best_sol = dict(best)
        best_sol.update({"fit_method": "grid_best", "note_fit": "root solve failed; using best |Δr_s| grid point (3-range tail)"})
        best_sol["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}

    best_sol["tail"] = {"R3_fm": float(r3_fm), "L3_fm": float(tail_len_fm), "V3_tail_MeV": float(v3_tail_mev)}
    v2_final = float(best_sol["V2_s_MeV"])
    if allow_signed_v2 and v2_final < -1e-12:
        best_sol["note_profile"] = "V2_s < 0 (outer repulsive barrier)."
    return best_sol


def _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
    *,
    a_s_target_fm: float,
    r_s_target_fm: float,
    r1_fm: float,
    r2_fm: float,
    cfg: dict[str, float],
    v1_hint_mev: float,
    v2_hint_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    """
    Step 7.13.5: Fit (V1_s,V2_s) to match singlet (a_s,r_s) under the barrier+tail split of the
    Yukawa tail coarse-graining region beyond R2.
    """
    if not (math.isfinite(a_s_target_fm) and math.isfinite(r_s_target_fm)):
        raise ValueError("invalid singlet targets")
    if not (0 < r1_fm < r2_fm and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid geometry/params for barrier+tail singlet fit")
    for k in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
        if k not in cfg or not math.isfinite(float(cfg[k])):
            raise ValueError(f"invalid cfg missing {k}")

    l3_total_fm = float(cfg["L3_total_fm"])
    lb_fm = float(cfg["Lb_fm"])
    lt_fm = float(cfg["Lt_fm"])
    barrier_coeff = float(cfg["barrier_coeff"])
    tail_depth_coeff = float(cfg["tail_depth_coeff"])

    r3_fm = float(r2_fm) + float(l3_total_fm)

    v1_lo = 0.1
    v1_hi = max(500.0, 1.6 * float(v1_hint_mev))
    v1_hi = min(v1_hi, 900.0)

    tol_a = 0.05  # fm
    tol_r = 2e-3  # fm

    def eval_v1(v1_s: float) -> dict[str, object] | None:
        if not (math.isfinite(v1_s) and v1_s >= 0):
            return None
        try:
            v2_fit = _fit_v2s_for_singlet_three_range_barrier_tail(
                a_s_target_fm=a_s_target_fm,
                r1_fm=r1_fm,
                r2_fm=r2_fm,
                cfg=cfg,
                v1_s_mev=v1_s,
                v2_hint_mev=v2_hint_mev,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
        except Exception:
            return None

        v2_s = float(v2_fit.get("V2_s_MeV", float("nan")))
        if not (math.isfinite(v2_s) and v2_s >= 0):
            return None

        vb = float(v2_s) * float(barrier_coeff)
        vt = float(v2_s) * float(tail_depth_coeff)
        if not (math.isfinite(vb) and math.isfinite(vt) and vb >= 0 and vt >= 0):
            return None

        segs = [
            (float(r1_fm), -float(v1_s)),
            (float(r2_fm - r1_fm), -float(v2_s)),
            (float(lb_fm), +float(vb)),
            (float(lt_fm), -float(vt)),
        ]
        try:
            a_exact = _scattering_length_exact_segments(
                r_end_fm=r3_fm, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
            )
        except Exception:
            return None
        if not (math.isfinite(a_exact) and abs(a_exact - a_s_target_fm) <= tol_a):
            return None

        try:
            ere = _fit_kcot_ere_segments(r_end_fm=r3_fm, segments=segs, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        except Exception:
            return None
        r_eff = float(ere.get("r_eff_fm", float("nan")))
        if not math.isfinite(r_eff):
            return None

        return {
            "V1_s_MeV": float(v1_s),
            "V2_s_MeV": float(v2_s),
            "Vb_s_MeV": float(vb),
            "Vt_s_MeV": float(vt),
            "a_s_exact_fm": float(a_exact),
            "ere": ere,
            "g_r_fm": float(r_eff - r_s_target_fm),
            "fit_v2": v2_fit,
        }

    # Scan V1_s to find sign-change brackets for g(V1)=r_eff(V1)-r_target.
    n_scan = 260
    v1_grid = [v1_lo + (v1_hi - v1_lo) * i / (n_scan - 1) for i in range(n_scan)]
    pts: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_abs: float | None = None

    for v1 in v1_grid:
        e = eval_v1(v1)
        if e is None:
            continue
        g = float(e["g_r_fm"])
        if not math.isfinite(g):
            continue
        pts.append(e)
        if best_abs is None or abs(g) < best_abs:
            best_abs = abs(g)
            best = e

    if len(pts) < 3:
        if best is None:
            raise ValueError("no usable V1_s samples (barrier+tail)")
        best_sol = dict(best)
        best_sol.update({"fit_method": "grid_best", "note_fit": "insufficient points; using best |Δr_s| grid point"})
        best_sol["barrier_tail"] = dict(cfg)
        return best_sol

    brackets: list[tuple[float, float]] = []
    for a, b in zip(pts[:-1], pts[1:]):
        g0 = float(a["g_r_fm"])
        g1 = float(b["g_r_fm"])
        if (g0 > 0) != (g1 > 0):
            brackets.append((float(a["V1_s_MeV"]), float(b["V1_s_MeV"])))

    if not brackets:
        if best is None:
            raise ValueError("no sign-change bracket and no fallback (barrier+tail)")
        best_sol = dict(best)
        best_sol.update({"fit_method": "grid_best", "note_fit": "no sign-change; using best |Δr_s| grid point"})
        best_sol["barrier_tail"] = dict(cfg)
        return best_sol

    # Prefer bracket closest to v1_hint (ties to the shallow branch).
    brackets.sort(key=lambda t: abs(0.5 * (t[0] + t[1]) - float(v1_hint_mev)))
    lo, hi = brackets[0]
    f_lo_e = eval_v1(lo)
    f_hi_e = eval_v1(hi)
    if f_lo_e is None or f_hi_e is None:
        if best is None:
            raise ValueError("invalid bracket and no fallback (barrier+tail)")
        best_sol = dict(best)
        best_sol.update({"fit_method": "grid_best", "note_fit": "invalid bracket; using best |Δr_s| grid point"})
        best_sol["barrier_tail"] = dict(cfg)
        return best_sol

    g_lo = float(f_lo_e["g_r_fm"])
    g_hi = float(f_hi_e["g_r_fm"])
    if not ((g_lo > 0) != (g_hi > 0)):
        if best is None:
            raise ValueError("degenerate bracket and no fallback (barrier+tail)")
        best_sol = dict(best)
        best_sol.update({"fit_method": "grid_best", "note_fit": "degenerate bracket; using best |Δr_s| grid point"})
        best_sol["barrier_tail"] = dict(cfg)
        return best_sol

    sol_lo = f_lo_e
    sol_hi = f_hi_e
    max_iter = 70
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        sol_mid = eval_v1(mid)
        if sol_mid is None:
            mid_a = (2.0 * lo + hi) / 3.0
            mid_b = (lo + 2.0 * hi) / 3.0
            sol_a = eval_v1(mid_a)
            sol_b = eval_v1(mid_b)
            sol_mid = sol_a if sol_a is not None else sol_b
            if sol_mid is None:
                break
            mid = float(sol_mid["V1_s_MeV"])

        g_mid = float(sol_mid["g_r_fm"])
        if not math.isfinite(g_mid):
            break
        if abs(g_mid) < tol_r or (hi - lo) < 1e-6:
            best_sol = dict(sol_mid)
            best_sol.update({"fit_method": "bisection", "note_fit": "solve V1_s by r_s (bisection) with barrier+tail"})
            best_sol["barrier_tail"] = dict(cfg)
            v1_final = float(best_sol["V1_s_MeV"])
            v2_final = float(best_sol["V2_s_MeV"])
            if v2_final > v1_final + 1e-9:
                best_sol["note_profile"] = "V2_s > V1_s (outer well deeper than inner); indicates this ansatz may be unnatural for singlet."
            return best_sol
        if (g_mid > 0) == (g_lo > 0):
            lo = mid
            g_lo = g_mid
            sol_lo = sol_mid
        else:
            hi = mid
            g_hi = g_mid
            sol_hi = sol_mid

    # Fallback: choose better endpoint.
    cand = sol_lo if abs(float(sol_lo["g_r_fm"])) < abs(float(sol_hi["g_r_fm"])) else sol_hi
    best_sol = dict(cand)
    best_sol.update({"fit_method": "bracket_best", "note_fit": "bisection did not converge; using best endpoint (barrier+tail)"})
    best_sol["barrier_tail"] = dict(cfg)
    v1_final = float(best_sol["V1_s_MeV"])
    v2_final = float(best_sol["V2_s_MeV"])
    if v2_final > v1_final + 1e-9:
        best_sol["note_profile"] = "V2_s > V1_s (outer well deeper than inner); indicates this ansatz may be unnatural for singlet."
    return best_sol


def _fit_v2s_for_singlet_repulsive_core_two_range(
    *,
    a_s_target_fm: float,
    rc_fm: float,
    r1_fm: float,
    r2_fm: float,
    vc_mev: float,
    v1_s_mev: float,
    v2_hint_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    """
    Fit V2_s so that the singlet scattering length matches a_s at k->0,
    on the repulsive-core + two-range geometry, with fixed (Rc,Vc,R1,R2,V1_s).
    """
    if not (math.isfinite(a_s_target_fm) and a_s_target_fm != 0.0):
        raise ValueError("invalid a_s target")
    if not (0.0 <= rc_fm < r1_fm < r2_fm and vc_mev >= 0 and v1_s_mev >= 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid geometry/params for singlet fit")

    denom = r2_fm - a_s_target_fm
    if not (math.isfinite(denom) and abs(denom) > 1e-12):
        raise ValueError("invalid a_s target for y_target")
    y_target = 1.0 / denom

    a_abs_max = 1e6

    def f(v2_s: float) -> float:
        y = _y_at_r2_repulsive_core_two_range(
            e_mev=0.0,
            rc_fm=rc_fm,
            r1_fm=r1_fm,
            r2_fm=r2_fm,
            vc_mev=vc_mev,
            v1_mev=v1_s_mev,
            v2_mev=v2_s,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        if not math.isfinite(y) or abs(y) < 1e-18:
            return float("nan")
        a = r2_fm - (1.0 / y)
        if not math.isfinite(a) or abs(a) > a_abs_max:
            return float("nan")
        return float(y - y_target)

    def scan_and_solve(*, v2_min: float, v2_max: float, tag: str) -> dict[str, object]:
        n_scan = 700
        grid = [v2_min + (v2_max - v2_min) * i / (n_scan - 1) for i in range(n_scan)]
        best_v2: float | None = None
        best_abs: float | None = None
        prev_v2: float | None = None
        prev_f: float | None = None
        brackets: list[tuple[float, float, float]] = []

        for v2 in grid:
            fv = f(v2)
            if not math.isfinite(fv):
                prev_v2 = None
                prev_f = None
                continue
            if best_abs is None or abs(fv) < best_abs:
                best_abs = abs(fv)
                best_v2 = v2
            if prev_v2 is not None and prev_f is not None:
                if fv == 0.0 or (fv > 0) != (prev_f > 0):
                    mid = 0.5 * (prev_v2 + v2)
                    brackets.append((prev_v2, v2, abs(mid - v2_hint_mev)))
            prev_v2 = v2
            prev_f = fv

        if not brackets:
            if best_v2 is None:
                raise ValueError(f"no usable V2_s samples ({tag})")
            return {
                "V2_s_MeV": float(best_v2),
                "fit_method": "grid_best",
                "note_fit": f"no sign-change; using best |Δy| grid point ({tag})",
                "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                "y_target_fm1": float(y_target),
            }

        brackets.sort(key=lambda t: t[2])
        lo, hi, _ = brackets[0]
        f_lo = f(lo)
        f_hi = f(hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi) and ((f_lo > 0) != (f_hi > 0))):
            if best_v2 is None:
                raise ValueError(f"invalid V2_s bracket and no fallback ({tag})")
            return {
                "V2_s_MeV": float(best_v2),
                "fit_method": "grid_best",
                "note_fit": f"invalid bracket; using best |Δy| grid point ({tag})",
                "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                "y_target_fm1": float(y_target),
            }

        for _ in range(90):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if not math.isfinite(f_mid):
                mid_a = (2.0 * lo + hi) / 3.0
                mid_b = (lo + 2.0 * hi) / 3.0
                fa = f(mid_a)
                fb = f(mid_b)
                if math.isfinite(fa):
                    mid, f_mid = mid_a, fa
                elif math.isfinite(fb):
                    mid, f_mid = mid_b, fb
                else:
                    break
            if f_mid == 0.0 or (hi - lo) < 1e-9:
                return {
                    "V2_s_MeV": float(mid),
                    "fit_method": "bisection",
                    "note_fit": f"bisection solve on V2_s ({tag})",
                    "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
                    "y_target_fm1": float(y_target),
                }
            if (f_mid > 0) == (f_lo > 0):
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
                f_hi = f_mid

        if best_v2 is None:
            raise ValueError(f"bisection failed and no fallback ({tag})")
        return {
            "V2_s_MeV": float(best_v2),
            "fit_method": "grid_best",
            "note_fit": f"bisection failed; using best |Δy| grid point ({tag})",
            "scan": {"v2_min": float(v2_min), "v2_max": float(v2_max), "n_scan": int(n_scan)},
            "y_target_fm1": float(y_target),
        }

    strict_max = min(float(v1_s_mev), 600.0)
    fit = scan_and_solve(v2_min=0.0, v2_max=strict_max, tag="strict (V2<=V1)")
    return fit


def _fit_v1v2_for_singlet_repulsive_core_two_range_by_a_and_r(
    *,
    a_s_target_fm: float,
    r_s_target_fm: float,
    rc_fm: float,
    r1_fm: float,
    r2_fm: float,
    vc_mev: float,
    v1_hint_mev: float,
    v2_hint_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    """
    Fit (V1_s, V2_s) to match singlet (a_s, r_s) under the repulsive-core + two-range geometry.
    """
    if not (math.isfinite(a_s_target_fm) and math.isfinite(r_s_target_fm)):
        raise ValueError("invalid singlet targets")
    if not (0.0 <= rc_fm < r1_fm < r2_fm and vc_mev >= 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid geometry/params for singlet fit")

    v1_lo = 0.1
    v1_hi = min(max(700.0, 1.6 * float(v1_hint_mev)), 1200.0)
    tol_a = 0.05
    tol_r = 2e-3
    max_iter = 70

    def eval_v1(v1_s: float) -> dict[str, object] | None:
        if not (math.isfinite(v1_s) and v1_s >= 0):
            return None
        try:
            v2_fit = _fit_v2s_for_singlet_repulsive_core_two_range(
                a_s_target_fm=a_s_target_fm,
                rc_fm=rc_fm,
                r1_fm=r1_fm,
                r2_fm=r2_fm,
                vc_mev=vc_mev,
                v1_s_mev=v1_s,
                v2_hint_mev=v2_hint_mev,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
        except Exception:
            return None
        v2_s = float(v2_fit.get("V2_s_MeV", float("nan")))
        if not (math.isfinite(v2_s) and v2_s >= 0):
            return None
        try:
            a_exact = _scattering_length_exact_repulsive_core_two_range(
                rc_fm=rc_fm,
                r1_fm=r1_fm,
                r2_fm=r2_fm,
                vc_mev=vc_mev,
                v1_mev=v1_s,
                v2_mev=v2_s,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
        except Exception:
            return None
        if not (math.isfinite(a_exact) and abs(a_exact - a_s_target_fm) <= tol_a):
            return None
        try:
            ere = _fit_kcot_ere_repulsive_core_two_range(
                rc_fm=rc_fm,
                r1_fm=r1_fm,
                r2_fm=r2_fm,
                vc_mev=vc_mev,
                v1_mev=v1_s,
                v2_mev=v2_s,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
        except Exception:
            return None
        r_eff = float(ere.get("r_eff_fm", float("nan")))
        if not math.isfinite(r_eff):
            return None
        return {
            "V1_s_MeV": float(v1_s),
            "V2_s_MeV": float(v2_s),
            "a_s_exact_fm": float(a_exact),
            "ere": ere,
            "g_r_fm": float(r_eff - r_s_target_fm),
            "fit_v2": v2_fit,
        }

    n_scan = 260
    v1_grid = [v1_lo + (v1_hi - v1_lo) * i / (n_scan - 1) for i in range(n_scan)]
    pts: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_abs = None
    for v1 in v1_grid:
        e = eval_v1(v1)
        if e is None:
            continue
        g = float(e["g_r_fm"])
        if not math.isfinite(g):
            continue
        pts.append(e)
        if best_abs is None or abs(g) < best_abs:
            best_abs = abs(g)
            best = e

    if len(pts) < 2:
        if best is None:
            raise ValueError("no usable V1_s samples")
        best = dict(best)
        best.update({"fit_method": "grid_best", "note_fit": "insufficient valid samples for bracketing"})
        best["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
        return best

    brackets: list[tuple[dict[str, object], dict[str, object], float]] = []
    for a, b in zip(pts[:-1], pts[1:]):
        ga = float(a["g_r_fm"])
        gb = float(b["g_r_fm"])
        if ga == 0.0 or (ga > 0) != (gb > 0):
            mid = 0.5 * (float(a["V1_s_MeV"]) + float(b["V1_s_MeV"]))
            brackets.append((a, b, float(mid)))

    if not brackets:
        if best is None:
            raise ValueError("no V1_s sign-change bracket and no fallback")
        best = dict(best)
        best.update({"fit_method": "grid_best", "note_fit": "no sign-change bracket; using best |Δr_s| grid point"})
        best["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
        return best

    def penalty(sol: dict[str, object]) -> float:
        v1 = float(sol["V1_s_MeV"])
        v2 = float(sol["V2_s_MeV"])
        p = v1
        if v2 > v1 + 1e-9:
            p += 1e6
        p += 1e-3 * abs(v1 - float(v1_hint_mev))
        return float(p)

    best_sol: dict[str, object] | None = None
    best_pen = None

    brackets.sort(key=lambda t: t[2])
    for lo_e, hi_e, _ in brackets[:8]:
        lo = float(lo_e["V1_s_MeV"])
        hi = float(hi_e["V1_s_MeV"])
        g_lo = float(lo_e["g_r_fm"])
        g_hi = float(hi_e["g_r_fm"])
        if not ((g_lo > 0) != (g_hi > 0)):
            continue

        sol_lo: dict[str, object] = lo_e
        sol_hi: dict[str, object] = hi_e
        cand_sol: dict[str, object] | None = None
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            sol_mid = eval_v1(mid)
            if sol_mid is None:
                mid_a = (2.0 * lo + hi) / 3.0
                mid_b = (lo + 2.0 * hi) / 3.0
                sol_a = eval_v1(mid_a)
                sol_b = eval_v1(mid_b)
                sol_mid = sol_a if sol_a is not None else sol_b
                if sol_mid is None:
                    break
                mid = float(sol_mid["V1_s_MeV"])

            g_mid = float(sol_mid["g_r_fm"])
            if not math.isfinite(g_mid):
                break
            if abs(g_mid) < tol_r or (hi - lo) < 1e-6:
                cand_sol = dict(sol_mid)
                cand_sol.update(
                    {"fit_method": "bisection", "note_fit": "solve V1_s by r_s (bisection) with V2_s(a_s) inner solve"}
                )
                cand_sol["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
                cand_sol["bisection"] = {"tol_a_fm": float(tol_a), "tol_r_fm": float(tol_r), "max_iter": int(max_iter)}
                break
            if (g_mid > 0) == (g_lo > 0):
                lo = mid
                g_lo = g_mid
                sol_lo = sol_mid
            else:
                hi = mid
                g_hi = g_mid
                sol_hi = sol_mid

        if cand_sol is None:
            cand = sol_lo if abs(float(sol_lo["g_r_fm"])) < abs(float(sol_hi["g_r_fm"])) else sol_hi
            cand_sol = dict(cand)
            cand_sol.update({"fit_method": "bracket_best", "note_fit": "bisection did not converge; using best endpoint"})
            cand_sol["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}
            cand_sol["bisection"] = {"tol_a_fm": float(tol_a), "tol_r_fm": float(tol_r), "max_iter": int(max_iter)}

        pen = penalty(cand_sol)
        if best_pen is None or pen < best_pen:
            best_pen = pen
            best_sol = cand_sol

    if best_sol is None:
        if best is None:
            raise ValueError("failed to solve V1_s root and no fallback")
        best_sol = dict(best)
        best_sol.update({"fit_method": "grid_best", "note_fit": "root solve failed; using best |Δr_s| grid point"})
        best_sol["scan_v1"] = {"v1_min": float(v1_lo), "v1_max": float(v1_hi), "n_scan": int(n_scan)}

    v1_final = float(best_sol["V1_s_MeV"])
    v2_final = float(best_sol["V2_s_MeV"])
    if v2_final > v1_final + 1e-9:
        best_sol["note_profile"] = "V2_s > V1_s (outer well deeper than inner); indicates this ansatz may be unnatural for singlet."
    return best_sol


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Phase 7 / Step 7.9+ nuclear two-range ansatz experiments")
    p.add_argument(
        "--step",
        choices=[
            "7.9.6",
            "7.9.7",
            "7.9.8",
            "7.13.3",
            "7.13.4",
            "7.13.5",
            "7.13.6",
            "7.13.7",
            "7.13.8",
            "7.13.8.1",
            "7.13.8.2",
            "7.13.8.3",
            "7.13.8.4",
            "7.13.8.5",
        ],
        default="7.9.8",
        help="Select which roadmap step output to generate (default: 7.9.8).",
    )
    args = p.parse_args(argv)
    step = str(args.step)

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exact SI constants (for conversion only).
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    hbarc_mev_fm = 197.326_980_4

    consts = _load_nist_codata_constants(root=root)
    for k in ("mp", "mn", "md"):
        if k not in consts:
            raise SystemExit(f"[fail] missing constant {k!r} in extracted_values.json")
    mp = float(consts["mp"]["value_si"])
    mn = float(consts["mn"]["value_si"])
    md = float(consts["md"]["value_si"])

    dm = (mp + mn - md)
    b_j = dm * (c**2)
    b_mev = b_j / (1e6 * e_charge)

    mu_kg = (mp * mn) / (mp + mn)
    mu_mev = (mu_kg * c**2) / (1e6 * e_charge)

    np_sets = _load_np_scattering_sets(root=root)
    eq18 = np_sets.get(18)
    eq19 = np_sets.get(19)
    if not (isinstance(eq18, dict) and isinstance(eq19, dict)):
        raise SystemExit("[fail] missing eq18/eq19 in extracted np scattering JSON")

    datasets = [
        {
            "label": "eq18 (GWU/SAID)",
            "eq_label": 18,
            "triplet": {
                "a_t_fm": _get_value(eq18, "a_t_fm"),
                "r_t_fm": _get_value(eq18, "r_t_fm"),
                "v2t_fm3": _get_value(eq18, "v2t_fm3"),
            },
            "singlet": {
                "a_s_fm": _get_value(eq18, "a_s_fm"),
                "r_s_fm": _get_value(eq18, "r_s_fm"),
                "v2s_fm3": _get_value(eq18, "v2s_fm3"),
            },
        },
        {
            "label": "eq19 (Nijmegen)",
            "eq_label": 19,
            "triplet": {
                "a_t_fm": _get_value(eq19, "a_t_fm"),
                "r_t_fm": _get_value(eq19, "r_t_fm"),
                "v2t_fm3": _get_value(eq19, "v2t_fm3"),
            },
            "singlet": {
                "a_s_fm": _get_value(eq19, "a_s_fm"),
                "r_s_fm": _get_value(eq19, "r_s_fm"),
                "v2s_fm3": _get_value(eq19, "v2s_fm3"),
            },
        },
    ]

    pion_scale = (
        _load_pion_range_scale(root=root)
        if step
        in (
            "7.13.3",
            "7.13.4",
            "7.13.5",
            "7.13.6",
            "7.13.7",
            "7.13.8",
            "7.13.8.1",
            "7.13.8.2",
            "7.13.8.3",
            "7.13.8.4",
            "7.13.8.5",
        )
        else None
    )
    lambda_pi_pm_fm = float(pion_scale["lambda_pi_pm_fm"]) if pion_scale is not None else None

    barrier_height_factor_best: float | None = None
    barrier_height_factor_scan: dict[str, object] | None = None
    tail_depth_factor_best: float | None = None
    tail_depth_factor_scan: dict[str, object] | None = None
    barrier_height_factor_kq_best: float | None = None
    tail_depth_factor_kq_best: float | None = None
    barrier_tail_kq_scan: dict[str, object] | None = None
    barrier_height_factor_kq_t_best: float | None = None
    tail_depth_factor_kq_t_best: float | None = None
    barrier_height_factor_kq_s_best: float | None = None
    tail_depth_factor_kq_s_best: float | None = None
    barrier_tail_channel_split_kq_scan: dict[str, object] | None = None
    singlet_r1_over_lambda_pi_best: float | None = None
    barrier_tail_channel_split_kq_singlet_r1_scan: dict[str, object] | None = None
    singlet_r2_over_lambda_pi_best: float | None = None
    barrier_tail_channel_split_kq_singlet_r2_scan: dict[str, object] | None = None
    triplet_barrier_len_fraction_best: float | None = None
    barrier_tail_channel_split_kq_triplet_barrier_fraction_scan: dict[str, object] | None = None
    if step == "7.13.6":
        if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
            raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.6")

        v2s_obs_list_scan = [float(d["singlet"]["v2s_fm3"]) for d in datasets]
        v2s_env_scan = {"min": float(min(v2s_obs_list_scan)), "max": float(max(v2s_obs_list_scan))}

        # Relax a single DOF: barrier height factor k (= barrier_height_factor).
        # Keep the rest (tail_len_over_lambda, barrier_len_fraction) fixed for reproducibility.
        k_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
        scan_rows: list[dict[str, object]] = []
        best_rank: tuple[int, float, float] | None = None
        best_k: float | None = None

        for k in k_grid:
            per_ds: list[dict[str, object]] = []
            preds: list[float] = []
            for d in datasets:
                label = str(d["label"])
                eq_label = int(d["eq_label"])
                trip = d["triplet"]
                sing = d["singlet"]
                try:
                    base_fit_t = _fit_triplet_three_range_barrier_tail_pion_constrained(
                        b_mev=b_mev,
                        targets=trip,
                        mu_mev=mu_mev,
                        hbarc_mev_fm=hbarc_mev_fm,
                        lambda_pi_fm=float(lambda_pi_pm_fm),
                        tail_len_over_lambda=1.0,
                        barrier_len_fraction=0.5,
                        barrier_height_factor=float(k),
                    )
                    r1 = float(base_fit_t["r1_fm"])
                    r2 = float(base_fit_t["r2_fm"])
                    v1t = float(base_fit_t["v1_mev"])
                    v2t = float(base_fit_t["v2_mev"])

                    cfg_any = base_fit_t.get("barrier_tail_config", {})
                    if not isinstance(cfg_any, dict):
                        raise ValueError("missing barrier_tail_config")
                    cfg = {str(kk): float(vv) for kk, vv in cfg_any.items() if math.isfinite(float(vv))}

                    sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                        a_s_target_fm=float(sing["a_s_fm"]),
                        r_s_target_fm=float(sing["r_s_fm"]),
                        r1_fm=r1,
                        r2_fm=r2,
                        cfg=cfg,
                        v1_hint_mev=v1t,
                        v2_hint_mev=v2t,
                        mu_mev=mu_mev,
                        hbarc_mev_fm=hbarc_mev_fm,
                    )
                    v1s = float(sing_fit["V1_s_MeV"])
                    v2s = float(sing_fit["V2_s_MeV"])
                    ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
                    if ere_s is None:
                        for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                            if req not in cfg:
                                raise ValueError(f"missing {req} in barrier_tail_config")
                        vb = float(v2s) * float(cfg["barrier_coeff"])
                        vt = float(v2s) * float(cfg["tail_depth_coeff"])
                        segs_s = [
                            (float(r1), -float(v1s)),
                            (float(r2 - r1), -float(v2s)),
                            (float(cfg["Lb_fm"]), +float(vb)),
                            (float(cfg["Lt_fm"]), -float(vt)),
                        ]
                        r3_end = float(r2) + float(cfg["L3_total_fm"])
                        ere_s = _fit_kcot_ere_segments(
                            r_end_fm=r3_end,
                            segments=segs_s,
                            mu_mev=mu_mev,
                            hbarc_mev_fm=hbarc_mev_fm,
                        )
                    v2s_pred_fm3 = float(ere_s["v2_fm3"])
                    within = bool(v2s_env_scan["min"] <= v2s_pred_fm3 <= v2s_env_scan["max"]) if math.isfinite(v2s_pred_fm3) else False
                    preds.append(v2s_pred_fm3)
                    per_ds.append(
                        {
                            "label": label,
                            "eq_label": eq_label,
                            "v2s_pred_fm3": float(v2s_pred_fm3),
                            "within_envelope": bool(within),
                        }
                    )
                except Exception as e:
                    preds.append(float("nan"))
                    per_ds.append({"label": label, "eq_label": eq_label, "v2s_pred_fm3": float("nan"), "within_envelope": False, "error": str(e)})

            dists: list[float] = []
            for v in preds:
                if not math.isfinite(v):
                    dists.append(float("inf"))
                elif v < float(v2s_env_scan["min"]):
                    dists.append(float(v2s_env_scan["min"]) - float(v))
                elif v > float(v2s_env_scan["max"]):
                    dists.append(float(v) - float(v2s_env_scan["max"]))
                else:
                    dists.append(0.0)
            within_all = bool(all(dd == 0.0 for dd in dists)) if all(math.isfinite(dd) for dd in dists) else False
            max_dist = float(max(dists)) if dists else float("nan")

            scan_rows.append(
                {
                    "barrier_height_factor": float(k),
                    "within_all": bool(within_all),
                    "max_dist_to_env_fm3": (float(max_dist) if math.isfinite(max_dist) else None),
                    "per_dataset": per_ds,
                }
            )

            rank = (0 if within_all else 1, max_dist if math.isfinite(max_dist) else float("inf"), abs(float(k) - 1.0))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_k = float(k)

        if best_k is None:
            raise SystemExit("[fail] barrier_height_factor scan produced no candidates")
        barrier_height_factor_best = float(best_k)
        barrier_height_factor_scan = {
            "k_grid": [float(k) for k in k_grid],
            "selected_k": float(best_k),
            "observed_envelope_v2s_fm3": v2s_env_scan,
            "rows": scan_rows,
            "policy": {
                "barrier_len_fraction_fixed": 0.5,
                "tail_len_over_lambda_fixed": 1.0,
                "selection_rule": "Prefer any k that makes all predicted v2s within envelope; tie-break by minimal max_dist_to_env, then |k-1|.",
            },
        }

    if step == "7.13.7":
        if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
            raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.7")

        v2s_obs_list_scan = [float(d["singlet"]["v2s_fm3"]) for d in datasets]
        v2s_env_scan = {"min": float(min(v2s_obs_list_scan)), "max": float(max(v2s_obs_list_scan))}

        # Relax a single DOF relative to Step 7.13.6: allow tail depth to vary independently (q),
        # while keeping the barrier height factor fixed to reduce overfitting.
        barrier_height_factor_fixed = 2.0
        tail_depth_factor_mean_preserving = 4.0  # for barrier_len_fraction=0.5 and k=2.0
        q_grid = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            2.0,
            3.0,
            4.0,
        ]
        scan_rows_q: list[dict[str, object]] = []
        best_rank_q: tuple[int, float, float] | None = None
        best_q: float | None = None

        for q in q_grid:
            per_ds: list[dict[str, object]] = []
            preds: list[float] = []
            for d in datasets:
                label = str(d["label"])
                eq_label = int(d["eq_label"])
                trip = d["triplet"]
                sing = d["singlet"]
                try:
                    base_fit_t = _fit_triplet_three_range_barrier_tail_pion_constrained_free_depth(
                        b_mev=b_mev,
                        targets=trip,
                        mu_mev=mu_mev,
                        hbarc_mev_fm=hbarc_mev_fm,
                        lambda_pi_fm=float(lambda_pi_pm_fm),
                        tail_len_over_lambda=1.0,
                        barrier_len_fraction=0.5,
                        barrier_height_factor=float(barrier_height_factor_fixed),
                        tail_depth_factor=float(q),
                    )
                    r1 = float(base_fit_t["r1_fm"])
                    r2 = float(base_fit_t["r2_fm"])
                    v1t = float(base_fit_t["v1_mev"])
                    v2t = float(base_fit_t["v2_mev"])

                    cfg_any = base_fit_t.get("barrier_tail_config", {})
                    if not isinstance(cfg_any, dict):
                        raise ValueError("missing barrier_tail_config")
                    cfg = {str(kk): float(vv) for kk, vv in cfg_any.items() if math.isfinite(float(vv))}

                    sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                        a_s_target_fm=float(sing["a_s_fm"]),
                        r_s_target_fm=float(sing["r_s_fm"]),
                        r1_fm=r1,
                        r2_fm=r2,
                        cfg=cfg,
                        v1_hint_mev=v1t,
                        v2_hint_mev=v2t,
                        mu_mev=mu_mev,
                        hbarc_mev_fm=hbarc_mev_fm,
                    )
                    v1s = float(sing_fit["V1_s_MeV"])
                    v2s = float(sing_fit["V2_s_MeV"])
                    ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
                    if ere_s is None:
                        for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                            if req not in cfg:
                                raise ValueError(f"missing {req} in barrier_tail_config")
                        vb = float(v2s) * float(cfg["barrier_coeff"])
                        vt = float(v2s) * float(cfg["tail_depth_coeff"])
                        segs_s = [
                            (float(r1), -float(v1s)),
                            (float(r2 - r1), -float(v2s)),
                            (float(cfg["Lb_fm"]), +float(vb)),
                            (float(cfg["Lt_fm"]), -float(vt)),
                        ]
                        r3_end = float(r2) + float(cfg["L3_total_fm"])
                        ere_s = _fit_kcot_ere_segments(
                            r_end_fm=r3_end,
                            segments=segs_s,
                            mu_mev=mu_mev,
                            hbarc_mev_fm=hbarc_mev_fm,
                        )
                    v2s_pred_fm3 = float(ere_s["v2_fm3"])
                    within = bool(v2s_env_scan["min"] <= v2s_pred_fm3 <= v2s_env_scan["max"]) if math.isfinite(v2s_pred_fm3) else False
                    preds.append(v2s_pred_fm3)
                    per_ds.append(
                        {
                            "label": label,
                            "eq_label": eq_label,
                            "v2s_pred_fm3": float(v2s_pred_fm3),
                            "within_envelope": bool(within),
                        }
                    )
                except Exception as e:
                    preds.append(float("nan"))
                    per_ds.append({"label": label, "eq_label": eq_label, "v2s_pred_fm3": float("nan"), "within_envelope": False, "error": str(e)})

            dists: list[float] = []
            for v in preds:
                if not math.isfinite(v):
                    dists.append(float("inf"))
                elif v < float(v2s_env_scan["min"]):
                    dists.append(float(v2s_env_scan["min"]) - float(v))
                elif v > float(v2s_env_scan["max"]):
                    dists.append(float(v) - float(v2s_env_scan["max"]))
                else:
                    dists.append(0.0)
            within_all = bool(all(dd == 0.0 for dd in dists)) if all(math.isfinite(dd) for dd in dists) else False
            max_dist = float(max(dists)) if dists else float("nan")

            scan_rows_q.append(
                {
                    "tail_depth_factor": float(q),
                    "within_all": bool(within_all),
                    "max_dist_to_env_fm3": (float(max_dist) if math.isfinite(max_dist) else None),
                    "per_dataset": per_ds,
                }
            )

            rank = (0 if within_all else 1, max_dist if math.isfinite(max_dist) else float("inf"), abs(float(q) - float(tail_depth_factor_mean_preserving)))
            if best_rank_q is None or rank < best_rank_q:
                best_rank_q = rank
                best_q = float(q)

        if best_q is None:
            raise SystemExit("[fail] tail_depth_factor scan produced no candidates")
        tail_depth_factor_best = float(best_q)
        tail_depth_factor_scan = {
            "q_grid": [float(q) for q in q_grid],
            "selected_q": float(best_q),
            "observed_envelope_v2s_fm3": v2s_env_scan,
            "rows": scan_rows_q,
            "policy": {
                "tail_len_over_lambda_fixed": 1.0,
                "barrier_len_fraction_fixed": 0.5,
                "barrier_height_factor_fixed": float(barrier_height_factor_fixed),
                "tail_depth_factor_mean_preserving": float(tail_depth_factor_mean_preserving),
                "selection_rule": "Prefer any q that makes all predicted v2s within envelope; tie-break by minimal max_dist_to_env, then |q-q_mean_preserving|.",
            },
        }

    if step == "7.13.8":
        if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
            raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.8")

        v2s_obs_list_scan = [float(d["singlet"]["v2s_fm3"]) for d in datasets]
        v2s_env_scan = {"min": float(min(v2s_obs_list_scan)), "max": float(max(v2s_obs_list_scan))}

        # Global 2-DOF scan: (k,q) under fixed (tail_len_over_lambda, barrier_len_fraction).
        # This is the minimal extension beyond 7.13.7 to test whether the envelope can be matched
        # without per-dataset tuning.
        k_grid = [1.0, 1.5, 2.0, 2.5, 3.0]
        q_grid = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

        scan_rows_kq: list[dict[str, object]] = []
        best_rank_kq: tuple[int, float, float, float] | None = None
        best_k_kq: float | None = None
        best_q_kq: float | None = None

        for k in k_grid:
            for q in q_grid:
                per_ds: list[dict[str, object]] = []
                preds: list[float] = []
                for d in datasets:
                    label = str(d["label"])
                    eq_label = int(d["eq_label"])
                    trip = d["triplet"]
                    sing = d["singlet"]
                    try:
                        base_fit_t = _fit_triplet_three_range_barrier_tail_pion_constrained_free_depth(
                            b_mev=b_mev,
                            targets=trip,
                            mu_mev=mu_mev,
                            hbarc_mev_fm=hbarc_mev_fm,
                            lambda_pi_fm=float(lambda_pi_pm_fm),
                            tail_len_over_lambda=1.0,
                            barrier_len_fraction=0.5,
                            barrier_height_factor=float(k),
                            tail_depth_factor=float(q),
                        )
                        r1 = float(base_fit_t["r1_fm"])
                        r2 = float(base_fit_t["r2_fm"])
                        v1t = float(base_fit_t["v1_mev"])
                        v2t = float(base_fit_t["v2_mev"])

                        cfg_any = base_fit_t.get("barrier_tail_config", {})
                        if not isinstance(cfg_any, dict):
                            raise ValueError("missing barrier_tail_config")
                        cfg = {str(kk): float(vv) for kk, vv in cfg_any.items() if math.isfinite(float(vv))}

                        sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                            a_s_target_fm=float(sing["a_s_fm"]),
                            r_s_target_fm=float(sing["r_s_fm"]),
                            r1_fm=r1,
                            r2_fm=r2,
                            cfg=cfg,
                            v1_hint_mev=v1t,
                            v2_hint_mev=v2t,
                            mu_mev=mu_mev,
                            hbarc_mev_fm=hbarc_mev_fm,
                        )
                        ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
                        if ere_s is None:
                            v1s = float(sing_fit["V1_s_MeV"])
                            v2s = float(sing_fit["V2_s_MeV"])
                            for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                                if req not in cfg:
                                    raise ValueError(f"missing {req} in barrier_tail_config")
                            vb = float(v2s) * float(cfg["barrier_coeff"])
                            vt = float(v2s) * float(cfg["tail_depth_coeff"])
                            segs_s = [
                                (float(r1), -float(v1s)),
                                (float(r2 - r1), -float(v2s)),
                                (float(cfg["Lb_fm"]), +float(vb)),
                                (float(cfg["Lt_fm"]), -float(vt)),
                            ]
                            r3_end = float(r2) + float(cfg["L3_total_fm"])
                            ere_s = _fit_kcot_ere_segments(
                                r_end_fm=r3_end,
                                segments=segs_s,
                                mu_mev=mu_mev,
                                hbarc_mev_fm=hbarc_mev_fm,
                            )
                        v2s_pred_fm3 = float(ere_s["v2_fm3"])
                        within = bool(v2s_env_scan["min"] <= v2s_pred_fm3 <= v2s_env_scan["max"]) if math.isfinite(v2s_pred_fm3) else False
                        preds.append(v2s_pred_fm3)
                        per_ds.append(
                            {
                                "label": label,
                                "eq_label": eq_label,
                                "v2s_pred_fm3": float(v2s_pred_fm3),
                                "within_envelope": bool(within),
                            }
                        )
                    except Exception as e:
                        preds.append(float("nan"))
                        per_ds.append(
                            {"label": label, "eq_label": eq_label, "v2s_pred_fm3": float("nan"), "within_envelope": False, "error": str(e)}
                        )

                dists: list[float] = []
                for v in preds:
                    if not math.isfinite(v):
                        dists.append(float("inf"))
                    elif v < float(v2s_env_scan["min"]):
                        dists.append(float(v2s_env_scan["min"]) - float(v))
                    elif v > float(v2s_env_scan["max"]):
                        dists.append(float(v) - float(v2s_env_scan["max"]))
                    else:
                        dists.append(0.0)
                within_all = bool(all(dd == 0.0 for dd in dists)) if all(math.isfinite(dd) for dd in dists) else False
                max_dist = float(max(dists)) if dists else float("nan")

                scan_rows_kq.append(
                    {
                        "barrier_height_factor": float(k),
                        "tail_depth_factor": float(q),
                        "within_all": bool(within_all),
                        "max_dist_to_env_fm3": (float(max_dist) if math.isfinite(max_dist) else None),
                        "per_dataset": per_ds,
                    }
                )

                # Mean-preserving baseline under barrier_len_fraction=0.5: q_mp = 2 + k.
                q_mp = 2.0 + float(k)
                rank = (
                    0 if within_all else 1,
                    max_dist if math.isfinite(max_dist) else float("inf"),
                    abs(float(q) - float(q_mp)),
                    abs(float(k) - 1.0),
                )
                if best_rank_kq is None or rank < best_rank_kq:
                    best_rank_kq = rank
                    best_k_kq = float(k)
                    best_q_kq = float(q)

        if best_k_kq is None or best_q_kq is None:
            raise SystemExit("[fail] (k,q) scan produced no candidates")
        barrier_height_factor_kq_best = float(best_k_kq)
        tail_depth_factor_kq_best = float(best_q_kq)
        barrier_tail_kq_scan = {
            "k_grid": [float(x) for x in k_grid],
            "q_grid": [float(x) for x in q_grid],
            "selected": {"k": float(best_k_kq), "q": float(best_q_kq)},
            "observed_envelope_v2s_fm3": v2s_env_scan,
            "rows": scan_rows_kq,
            "policy": {
                "tail_len_over_lambda_fixed": 1.0,
                "barrier_len_fraction_fixed": 0.5,
                "selection_rule": "Prefer any (k,q) that makes all predicted v2s within envelope; tie-break by minimal max_dist_to_env, then |q-q_mean_preserving(k)|, then |k-1|.",
                "q_mean_preserving_formula": "q_mp = (L3_total + Lb*k)/Lt = 2 + k (for barrier_len_fraction=0.5)",
            },
        }

    if step == "7.13.8.1":
        if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
            raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.8.1")

        v2s_obs_list_scan = [float(d["singlet"]["v2s_fm3"]) for d in datasets]
        v2t_obs_list_scan = [float(d["triplet"]["v2t_fm3"]) for d in datasets]
        rs_obs_list_scan = [float(d["singlet"]["r_s_fm"]) for d in datasets]
        v2s_env_scan = {"min": float(min(v2s_obs_list_scan)), "max": float(max(v2s_obs_list_scan))}
        v2t_env_scan = {"min": float(min(v2t_obs_list_scan)), "max": float(max(v2t_obs_list_scan))}
        rs_env_scan = {"min": float(min(rs_obs_list_scan)), "max": float(max(rs_obs_list_scan))}

        def dist_to_env(x: float, env: dict[str, float]) -> float:
            if not math.isfinite(x):
                return float("inf")
            lo = float(env["min"])
            hi = float(env["max"])
            if lo <= x <= hi:
                return 0.0
            return float(min(abs(x - lo), abs(x - hi)))

        # Extended global 2-DOF scan: (k,q) under fixed (tail_len_over_lambda, barrier_len_fraction),
        # now requiring *both* triplet v2t and singlet v2s to be compatible with the analysis-dependent envelope.
        k_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        q_grid = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

        scan_rows_kq: list[dict[str, object]] = []
        best_rank_kq: tuple[int, int, float, float, float, float, float, float] | None = None
        best_k_kq: float | None = None
        best_q_kq: float | None = None

        for k in k_grid:
            for q in q_grid:
                per_ds: list[dict[str, object]] = []
                max_dist_v2t = 0.0
                max_dist_v2s = 0.0
                max_dist_rs = 0.0
                outside_count = 0

                for d in datasets:
                    label = str(d["label"])
                    eq_label = int(d["eq_label"])
                    trip = d["triplet"]
                    sing = d["singlet"]
                    try:
                        base_fit_t = _fit_triplet_three_range_barrier_tail_pion_constrained_free_depth(
                            b_mev=b_mev,
                            targets=trip,
                            mu_mev=mu_mev,
                            hbarc_mev_fm=hbarc_mev_fm,
                            lambda_pi_fm=float(lambda_pi_pm_fm),
                            tail_len_over_lambda=1.0,
                            barrier_len_fraction=0.5,
                            barrier_height_factor=float(k),
                            tail_depth_factor=float(q),
                        )
                        v2t_fit_fm3 = float(base_fit_t["ere"]["v2_fm3"])

                        r1 = float(base_fit_t["r1_fm"])
                        r2 = float(base_fit_t["r2_fm"])
                        v1t = float(base_fit_t["v1_mev"])
                        v2t = float(base_fit_t["v2_mev"])

                        cfg_any = base_fit_t.get("barrier_tail_config", {})
                        if not isinstance(cfg_any, dict):
                            raise ValueError("missing barrier_tail_config")
                        cfg = {str(kk): float(vv) for kk, vv in cfg_any.items() if math.isfinite(float(vv))}

                        sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                            a_s_target_fm=float(sing["a_s_fm"]),
                            r_s_target_fm=float(sing["r_s_fm"]),
                            r1_fm=r1,
                            r2_fm=r2,
                            cfg=cfg,
                            v1_hint_mev=v1t,
                            v2_hint_mev=v2t,
                            mu_mev=mu_mev,
                            hbarc_mev_fm=hbarc_mev_fm,
                        )
                        ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
                        if ere_s is None:
                            v1s = float(sing_fit["V1_s_MeV"])
                            v2s = float(sing_fit["V2_s_MeV"])
                            for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                                if req not in cfg:
                                    raise ValueError(f"missing {req} in barrier_tail_config")
                            vb = float(v2s) * float(cfg["barrier_coeff"])
                            vt = float(v2s) * float(cfg["tail_depth_coeff"])
                            segs_s = [
                                (float(r1), -float(v1s)),
                                (float(r2 - r1), -float(v2s)),
                                (float(cfg["Lb_fm"]), +float(vb)),
                                (float(cfg["Lt_fm"]), -float(vt)),
                            ]
                            r3_end = float(r2) + float(cfg["L3_total_fm"])
                            ere_s = _fit_kcot_ere_segments(
                                r_end_fm=r3_end,
                                segments=segs_s,
                                mu_mev=mu_mev,
                                hbarc_mev_fm=hbarc_mev_fm,
                            )
                        v2s_pred_fm3 = float(ere_s["v2_fm3"])
                        r_s_fit_fm = float(ere_s["r_eff_fm"])
                    except Exception as e:
                        per_ds.append(
                            {
                                "label": label,
                                "eq_label": eq_label,
                                "v2t_fit_fm3": None,
                                "v2s_pred_fm3": None,
                                "r_s_fit_fm": None,
                                "within_v2t_envelope": False,
                                "within_v2s_envelope": False,
                                "within_r_s_envelope": False,
                                "error": str(e),
                            }
                        )
                        outside_count += 3
                        max_dist_v2t = float("inf")
                        max_dist_v2s = float("inf")
                        max_dist_rs = float("inf")
                        continue

                    within_v2t = bool(v2t_env_scan["min"] <= v2t_fit_fm3 <= v2t_env_scan["max"]) if math.isfinite(v2t_fit_fm3) else False
                    within_v2s = bool(v2s_env_scan["min"] <= v2s_pred_fm3 <= v2s_env_scan["max"]) if math.isfinite(v2s_pred_fm3) else False
                    within_rs = bool(rs_env_scan["min"] <= r_s_fit_fm <= rs_env_scan["max"]) if math.isfinite(r_s_fit_fm) else False
                    outside_count += int(not within_v2t) + int(not within_v2s) + int(not within_rs)

                    max_dist_v2t = max(float(max_dist_v2t), dist_to_env(float(v2t_fit_fm3), v2t_env_scan))
                    max_dist_v2s = max(float(max_dist_v2s), dist_to_env(float(v2s_pred_fm3), v2s_env_scan))
                    max_dist_rs = max(float(max_dist_rs), dist_to_env(float(r_s_fit_fm), rs_env_scan))

                    per_ds.append(
                        {
                            "label": label,
                            "eq_label": eq_label,
                            "v2t_fit_fm3": float(v2t_fit_fm3),
                            "v2s_pred_fm3": float(v2s_pred_fm3),
                            "r_s_fit_fm": float(r_s_fit_fm),
                            "within_v2t_envelope": bool(within_v2t),
                            "within_v2s_envelope": bool(within_v2s),
                            "within_r_s_envelope": bool(within_rs),
                        }
                    )

                within_all = bool(outside_count == 0)
                max_dist_total = float(max(max_dist_v2t, max_dist_v2s))
                scan_rows_kq.append(
                    {
                        "barrier_height_factor": float(k),
                        "tail_depth_factor": float(q),
                        "within_all": bool(within_all),
                        "outside_count": int(outside_count),
                        "max_dist_v2t_fm3": (float(max_dist_v2t) if math.isfinite(max_dist_v2t) else None),
                        "max_dist_v2s_fm3": (float(max_dist_v2s) if math.isfinite(max_dist_v2s) else None),
                        "max_dist_r_s_fm": (float(max_dist_rs) if math.isfinite(max_dist_rs) else None),
                        "max_dist_total_fm3": (float(max_dist_total) if math.isfinite(max_dist_total) else None),
                        "per_dataset": per_ds,
                    }
                )

                rank = (
                    0 if within_all else 1,
                    int(outside_count),
                    float(max_dist_total) if math.isfinite(max_dist_total) else float("inf"),
                    float(max_dist_rs) if math.isfinite(max_dist_rs) else float("inf"),
                    float(max_dist_v2t) if math.isfinite(max_dist_v2t) else float("inf"),
                    float(max_dist_v2s) if math.isfinite(max_dist_v2s) else float("inf"),
                    float(k),
                    float(q),
                )
                if best_rank_kq is None or rank < best_rank_kq:
                    best_rank_kq = rank
                    best_k_kq = float(k)
                    best_q_kq = float(q)

        if best_k_kq is None or best_q_kq is None:
            raise SystemExit("[fail] extended (k,q) scan produced no candidates")
        barrier_height_factor_kq_best = float(best_k_kq)
        tail_depth_factor_kq_best = float(best_q_kq)
        barrier_tail_kq_scan = {
            "k_grid": [float(x) for x in k_grid],
            "q_grid": [float(x) for x in q_grid],
            "selected": {
                "k": float(best_k_kq),
                "q": float(best_q_kq),
                "rank": list(best_rank_kq) if best_rank_kq is not None else None,
            },
            "observed_envelope_v2s_fm3": v2s_env_scan,
            "observed_envelope_v2t_fm3": v2t_env_scan,
            "observed_envelope_r_s_fm": rs_env_scan,
            "rows": scan_rows_kq,
            "policy": {
                "tail_len_over_lambda_fixed": 1.0,
                "barrier_len_fraction_fixed": 0.5,
                "selection_rule": "Prefer any (k,q) that makes triplet v2t and singlet (r_s,v2s) all within envelope; tie-break by minimal max_dist_total(v2), then max_dist(r_s), then k, then q.",
            },
        }

    if step == "7.13.8.2":
        if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
            raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.8.2")

        v2s_obs_list_scan = [float(d["singlet"]["v2s_fm3"]) for d in datasets]
        v2t_obs_list_scan = [float(d["triplet"]["v2t_fm3"]) for d in datasets]
        rs_obs_list_scan = [float(d["singlet"]["r_s_fm"]) for d in datasets]
        v2s_env_scan = {"min": float(min(v2s_obs_list_scan)), "max": float(max(v2s_obs_list_scan))}
        v2t_env_scan = {"min": float(min(v2t_obs_list_scan)), "max": float(max(v2t_obs_list_scan))}
        rs_env_scan = {"min": float(min(rs_obs_list_scan)), "max": float(max(rs_obs_list_scan))}

        def dist_to_env(x: float, env: dict[str, float]) -> float:
            if not math.isfinite(x):
                return float("inf")
            lo = float(env["min"])
            hi = float(env["max"])
            if lo <= x <= hi:
                return 0.0
            return float(min(abs(x - lo), abs(x - hi)))

        # Channel-split scan: allow separate (k,q) for triplet and singlet outer structure.
        # Keep (tail_len_over_lambda, barrier_len_fraction) fixed to avoid hidden DOF.
        # Keep the scan grid intentionally modest: we want a reproducible "next-minimal" cut
        # that runs in a reasonable time on Windows while still covering the known tension points.
        k_t_grid = [3.0, 3.5, 4.0]
        q_t_grid = [1.2, 1.3, 1.4]
        k_s_grid = [0.5, 1.0, 1.5, 2.0]
        q_s_grid = [0.6, 0.8, 1.0]

        tol_a_fm = 0.2
        tol_r_fm = 0.05
        scale_v2_fm3 = 0.05

        scan_rows: list[dict[str, object]] = []
        best_rank: tuple[int, int, float, float, float, float, float, float, float, float] | None = None
        best_k_t: float | None = None
        best_q_t: float | None = None
        best_k_s: float | None = None
        best_q_s: float | None = None

        triplet_cache: dict[tuple[float, float, int], dict[str, object]] = {}

        for k_t in k_t_grid:
            for q_t in q_t_grid:
                for k_s in k_s_grid:
                    for q_s in q_s_grid:
                        per_ds: list[dict[str, object]] = []
                        outside_count = 0
                        max_dist_v2t = 0.0
                        max_dist_v2s = 0.0
                        max_abs_dr_s = 0.0
                        max_triplet_pen = 0.0

                        for d in datasets:
                            label = str(d["label"])
                            eq_label = int(d["eq_label"])
                            trip = d["triplet"]
                            sing = d["singlet"]

                            tkey = (float(k_t), float(q_t), eq_label)
                            tfit = triplet_cache.get(tkey)
                            if tfit is None:
                                try:
                                    base_fit_t = _fit_triplet_three_range_barrier_tail_pion_constrained_free_depth(
                                        b_mev=b_mev,
                                        targets=trip,
                                        mu_mev=mu_mev,
                                        hbarc_mev_fm=hbarc_mev_fm,
                                        lambda_pi_fm=float(lambda_pi_pm_fm),
                                        tail_len_over_lambda=1.0,
                                        barrier_len_fraction=0.5,
                                        barrier_height_factor=float(k_t),
                                        tail_depth_factor=float(q_t),
                                    )
                                    triplet_cache[tkey] = base_fit_t
                                    tfit = base_fit_t
                                except Exception as e:
                                    triplet_cache[tkey] = {"error": str(e)}
                                    tfit = triplet_cache[tkey]

                            t_err = (
                                str(tfit.get("error"))
                                if isinstance(tfit, dict) and isinstance(tfit.get("error"), str) and str(tfit.get("error"))
                                else None
                            )
                            if t_err is not None:
                                per_ds.append(
                                    {
                                        "label": label,
                                        "eq_label": eq_label,
                                        "within_triplet_ar_tolerance": False,
                                        "within_v2t_envelope": False,
                                        "within_r_s_envelope": False,
                                        "within_v2s_envelope": False,
                                        "error": t_err,
                                    }
                                )
                                outside_count += 4
                                max_dist_v2t = float("inf")
                                max_dist_v2s = float("inf")
                                max_abs_dr_s = float("inf")
                                max_triplet_pen = float("inf")
                                continue

                            a_t_fit = float(tfit.get("a_exact_fm", float("nan")))
                            ere_t = tfit.get("ere") if isinstance(tfit.get("ere"), dict) else {}
                            r_t_fit = float(ere_t.get("r_eff_fm", float("nan"))) if isinstance(ere_t, dict) else float("nan")
                            v2t_fit = float(ere_t.get("v2_fm3", float("nan"))) if isinstance(ere_t, dict) else float("nan")
                            da_t = a_t_fit - float(trip["a_t_fm"])
                            dr_t = r_t_fit - float(trip["r_t_fm"])

                            ok_ar = bool(
                                math.isfinite(da_t)
                                and math.isfinite(dr_t)
                                and abs(float(da_t)) <= float(tol_a_fm)
                                and abs(float(dr_t)) <= float(tol_r_fm)
                            )
                            ok_v2t = bool(
                                math.isfinite(v2t_fit)
                                and float(v2t_env_scan["min"]) <= float(v2t_fit) <= float(v2t_env_scan["max"])
                            )
                            outside_count += int(not ok_ar) + int(not ok_v2t)

                            if math.isfinite(da_t) and math.isfinite(dr_t):
                                triplet_pen = max(abs(float(da_t)) / float(tol_a_fm), abs(float(dr_t)) / float(tol_r_fm))
                                max_triplet_pen = max(float(max_triplet_pen), float(triplet_pen))
                            else:
                                max_triplet_pen = float("inf")

                            max_dist_v2t = max(float(max_dist_v2t), float(dist_to_env(float(v2t_fit), v2t_env_scan)))

                            sing_error = None
                            v2s_pred_fm3: float | None = None
                            r_s_fit_fm: float | None = None
                            dr_s = float("nan")
                            ok_rs = False
                            ok_v2s = False

                            try:
                                r1 = float(tfit["r1_fm"])
                                r2 = float(tfit["r2_fm"])
                                v1_hint = float(tfit["v1_mev"])
                                v2_hint = float(tfit["v2_mev"])

                                cfg_any = _barrier_tail_config_free_depth(
                                    lambda_pi_fm=float(lambda_pi_pm_fm),
                                    tail_len_over_lambda=1.0,
                                    barrier_len_fraction=0.5,
                                    barrier_height_factor=float(k_s),
                                    tail_depth_factor=float(q_s),
                                )
                                if not isinstance(cfg_any, dict):
                                    raise ValueError("invalid barrier_tail_config (singlet)")
                                cfg = {str(kk): float(vv) for kk, vv in cfg_any.items() if math.isfinite(float(vv))}

                                sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                                    a_s_target_fm=float(sing["a_s_fm"]),
                                    r_s_target_fm=float(sing["r_s_fm"]),
                                    r1_fm=r1,
                                    r2_fm=r2,
                                    cfg=cfg,
                                    v1_hint_mev=v1_hint,
                                    v2_hint_mev=v2_hint,
                                    mu_mev=mu_mev,
                                    hbarc_mev_fm=hbarc_mev_fm,
                                )
                                v1s = float(sing_fit["V1_s_MeV"])
                                v2s = float(sing_fit["V2_s_MeV"])
                                ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
                                if ere_s is None:
                                    for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                                        if req not in cfg:
                                            raise ValueError(f"missing {req} in barrier_tail_config")
                                    vb = float(v2s) * float(cfg["barrier_coeff"])
                                    vt = float(v2s) * float(cfg["tail_depth_coeff"])
                                    segs_s = [
                                        (float(r1), -float(v1s)),
                                        (float(r2 - r1), -float(v2s)),
                                        (float(cfg["Lb_fm"]), +float(vb)),
                                        (float(cfg["Lt_fm"]), -float(vt)),
                                    ]
                                    r3_end = float(r2) + float(cfg["L3_total_fm"])
                                    ere_s = _fit_kcot_ere_segments(
                                        r_end_fm=r3_end,
                                        segments=segs_s,
                                        mu_mev=mu_mev,
                                        hbarc_mev_fm=hbarc_mev_fm,
                                    )

                                v2s_pred_fm3 = float(ere_s["v2_fm3"])
                                r_s_fit_fm = float(ere_s["r_eff_fm"])
                                dr_s = float(r_s_fit_fm) - float(sing["r_s_fm"])
                                max_abs_dr_s = max(float(max_abs_dr_s), abs(float(dr_s)))

                                ok_rs = bool(
                                    math.isfinite(r_s_fit_fm)
                                    and float(rs_env_scan["min"]) <= float(r_s_fit_fm) <= float(rs_env_scan["max"])
                                )
                                ok_v2s = bool(
                                    math.isfinite(v2s_pred_fm3)
                                    and float(v2s_env_scan["min"]) <= float(v2s_pred_fm3) <= float(v2s_env_scan["max"])
                                )
                                outside_count += int(not ok_rs) + int(not ok_v2s)
                                max_dist_v2s = max(float(max_dist_v2s), float(dist_to_env(float(v2s_pred_fm3), v2s_env_scan)))
                            except Exception as e:
                                sing_error = str(e)
                                outside_count += 2
                                max_dist_v2s = float("inf")
                                max_abs_dr_s = float("inf")

                            per_ds.append(
                                {
                                    "label": label,
                                    "eq_label": eq_label,
                                    "a_t_fit_fm": float(a_t_fit) if math.isfinite(a_t_fit) else None,
                                    "r_t_fit_fm": float(r_t_fit) if math.isfinite(r_t_fit) else None,
                                    "v2t_fit_fm3": float(v2t_fit) if math.isfinite(v2t_fit) else None,
                                    "a_t_fit_minus_obs_fm": float(da_t) if math.isfinite(da_t) else None,
                                    "r_t_fit_minus_obs_fm": float(dr_t) if math.isfinite(dr_t) else None,
                                    "within_triplet_ar_tolerance": bool(ok_ar),
                                    "within_v2t_envelope": bool(ok_v2t),
                                    "v2s_pred_fm3": float(v2s_pred_fm3) if v2s_pred_fm3 is not None and math.isfinite(v2s_pred_fm3) else None,
                                    "r_s_fit_fm": float(r_s_fit_fm) if r_s_fit_fm is not None and math.isfinite(r_s_fit_fm) else None,
                                    "r_s_fit_minus_obs_fm": float(dr_s) if math.isfinite(dr_s) else None,
                                    "within_r_s_envelope": bool(ok_rs),
                                    "within_v2s_envelope": bool(ok_v2s),
                                    "error": sing_error,
                                }
                            )

                        within_all = bool(outside_count == 0)
                        max_v2_total = float(max(float(max_dist_v2t), float(max_dist_v2s)))
                        delta_kq = abs(float(k_t) - float(k_s)) + abs(float(q_t) - float(q_s))
                        rank = (
                            0 if within_all else 1,
                            int(outside_count),
                            float(max_v2_total) / float(scale_v2_fm3) if math.isfinite(max_v2_total) else float("inf"),
                            float(max_abs_dr_s) / float(tol_r_fm) if math.isfinite(max_abs_dr_s) else float("inf"),
                            float(max_triplet_pen) if math.isfinite(max_triplet_pen) else float("inf"),
                            float(delta_kq),
                            float(k_t),
                            float(q_t),
                            float(k_s),
                            float(q_s),
                        )
                        scan_rows.append(
                            {
                                "k_t": float(k_t),
                                "q_t": float(q_t),
                                "k_s": float(k_s),
                                "q_s": float(q_s),
                                "within_all": bool(within_all),
                                "outside_count": int(outside_count),
                                "max_dist_v2t_fm3": (float(max_dist_v2t) if math.isfinite(max_dist_v2t) else None),
                                "max_dist_v2s_fm3": (float(max_dist_v2s) if math.isfinite(max_dist_v2s) else None),
                                "max_abs_r_s_fit_minus_obs_fm": (float(max_abs_dr_s) if math.isfinite(max_abs_dr_s) else None),
                                "max_triplet_ar_penalty": (float(max_triplet_pen) if math.isfinite(max_triplet_pen) else None),
                                "per_dataset": per_ds,
                            }
                        )
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_k_t = float(k_t)
                            best_q_t = float(q_t)
                            best_k_s = float(k_s)
                            best_q_s = float(q_s)

        if best_k_t is None or best_q_t is None or best_k_s is None or best_q_s is None:
            raise SystemExit("[fail] channel-split (k,q) scan produced no candidates")
        barrier_height_factor_kq_t_best = float(best_k_t)
        tail_depth_factor_kq_t_best = float(best_q_t)
        barrier_height_factor_kq_s_best = float(best_k_s)
        tail_depth_factor_kq_s_best = float(best_q_s)
        barrier_tail_channel_split_kq_scan = {
            "k_t_grid": [float(x) for x in k_t_grid],
            "q_t_grid": [float(x) for x in q_t_grid],
            "k_s_grid": [float(x) for x in k_s_grid],
            "q_s_grid": [float(x) for x in q_s_grid],
            "selected": {
                "k_t": float(best_k_t),
                "q_t": float(best_q_t),
                "k_s": float(best_k_s),
                "q_s": float(best_q_s),
                "rank": list(best_rank) if best_rank is not None else None,
            },
            "observed_envelope_v2s_fm3": v2s_env_scan,
            "observed_envelope_v2t_fm3": v2t_env_scan,
            "observed_envelope_r_s_fm": rs_env_scan,
            "rows": scan_rows,
            "policy": {
                "tail_len_over_lambda_fixed": 1.0,
                "barrier_len_fraction_fixed": 0.5,
                "triplet_ar_tolerance": {"tol_a_fm": float(tol_a_fm), "tol_r_fm": float(tol_r_fm)},
                "selection_rule": "Prefer any (k_t,q_t,k_s,q_s) that makes triplet (a_t,r_t) stable and v2t within envelope, and singlet (r_s,v2s) within envelope; tie-break by minimal max_dist(v2), then |Δr_s|, then triplet AR penalty, then minimal channel split |k_t-k_s|+|q_t-q_s|.",
            },
        }

    if step == "7.13.8.3":
        if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
            raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.8.3")

        # This step builds on the fixed output of Step 7.13.8.2 (channel-split kq selection),
        # then scans a single additional DOF: the singlet inner boundary R1_s / λπ.
        prev_metrics = out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_metrics.json"
        if not prev_metrics.exists():
            raise SystemExit(
                "[fail] missing Step 7.13.8.2 fixed output needed for Step 7.13.8.3.\n"
                "Run first:\n"
                "  python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.13.8.2\n"
                f"Expected: {prev_metrics}"
            )
        prev = _load_json(prev_metrics)
        prev_scan = prev.get("barrier_tail_channel_split_kq_scan")
        if not isinstance(prev_scan, dict):
            raise SystemExit(f"[fail] invalid 7.13.8.2 metrics: missing barrier_tail_channel_split_kq_scan: {prev_metrics}")
        prev_sel = prev_scan.get("selected")
        if not isinstance(prev_sel, dict):
            raise SystemExit(f"[fail] invalid 7.13.8.2 metrics: missing selected: {prev_metrics}")

        k_t_sel = float(prev_sel.get("k_t", float("nan")))
        q_t_sel = float(prev_sel.get("q_t", float("nan")))
        k_s_sel = float(prev_sel.get("k_s", float("nan")))
        q_s_sel = float(prev_sel.get("q_s", float("nan")))
        if not (
            math.isfinite(k_t_sel)
            and math.isfinite(q_t_sel)
            and math.isfinite(k_s_sel)
            and math.isfinite(q_s_sel)
            and k_t_sel >= 0.0
            and q_t_sel >= 0.0
            and k_s_sel >= 0.0
            and q_s_sel >= 0.0
        ):
            raise SystemExit(f"[fail] invalid 7.13.8.2 selected (k_t,q_t,k_s,q_s) in: {prev_metrics}")

        # Make the selected channel-split parameters available downstream (plot/CSV/metrics).
        barrier_height_factor_kq_t_best = float(k_t_sel)
        tail_depth_factor_kq_t_best = float(q_t_sel)
        barrier_height_factor_kq_s_best = float(k_s_sel)
        tail_depth_factor_kq_s_best = float(q_s_sel)

        v2s_obs_list_scan = [float(d["singlet"]["v2s_fm3"]) for d in datasets]
        v2t_obs_list_scan = [float(d["triplet"]["v2t_fm3"]) for d in datasets]
        rs_obs_list_scan = [float(d["singlet"]["r_s_fm"]) for d in datasets]
        v2s_env_scan = {"min": float(min(v2s_obs_list_scan)), "max": float(max(v2s_obs_list_scan))}
        v2t_env_scan = {"min": float(min(v2t_obs_list_scan)), "max": float(max(v2t_obs_list_scan))}
        rs_env_scan = {"min": float(min(rs_obs_list_scan)), "max": float(max(rs_obs_list_scan))}

        def dist_to_env(x: float, env: dict[str, float]) -> float:
            if not math.isfinite(x):
                return float("inf")
            lo = float(env["min"])
            hi = float(env["max"])
            if lo <= x <= hi:
                return 0.0
            return float(min(abs(x - lo), abs(x - hi)))

        tol_a_fm = 0.2
        tol_r_fm = 0.05
        scale_v2_fm3 = 0.05

        # Precompute triplet fits once (depends only on the fixed (k_t,q_t) selection).
        triplet_fit_by_eq: dict[int, dict[str, object]] = {}
        for d in datasets:
            eq_label = int(d["eq_label"])
            trip = d["triplet"]
            base_fit_t = _fit_triplet_three_range_barrier_tail_pion_constrained_free_depth(
                b_mev=b_mev,
                targets=trip,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
                lambda_pi_fm=float(lambda_pi_pm_fm),
                tail_len_over_lambda=1.0,
                barrier_len_fraction=0.5,
                barrier_height_factor=float(k_t_sel),
                tail_depth_factor=float(q_t_sel),
            )
            triplet_fit_by_eq[eq_label] = base_fit_t

        # Precompute singlet barrier+tail config once (depends only on fixed (k_s,q_s) selection).
        cfg_any = _barrier_tail_config_free_depth(
            lambda_pi_fm=float(lambda_pi_pm_fm),
            tail_len_over_lambda=1.0,
            barrier_len_fraction=0.5,
            barrier_height_factor=float(k_s_sel),
            tail_depth_factor=float(q_s_sel),
        )
        if not isinstance(cfg_any, dict):
            raise SystemExit("[fail] invalid barrier_tail_config for step 7.13.8.3")
        cfg = {str(k): float(v) for k, v in cfg_any.items() if math.isfinite(float(v))}

        # Scan only the singlet inner boundary ratio (global): R1_s = (R1_s/λπ)·λπ.
        r1_over_lambda_grid = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

        scan_rows: list[dict[str, object]] = []
        best_rank: tuple[int, int, float, float, float, float, float] | None = None
        best_r1_over: float | None = None

        for r1_over in r1_over_lambda_grid:
            r1_s_fm = float(r1_over) * float(lambda_pi_pm_fm)
            per_ds: list[dict[str, object]] = []
            outside_count = 0
            max_dist_v2t = 0.0
            max_dist_v2s = 0.0
            max_abs_dr_s = 0.0
            max_triplet_pen = 0.0
            max_abs_delta_r1_over_lambda = 0.0

            for d in datasets:
                label = str(d["label"])
                eq_label = int(d["eq_label"])
                trip = d["triplet"]
                sing = d["singlet"]

                tfit = triplet_fit_by_eq[eq_label]
                a_t_fit = float(tfit.get("a_exact_fm", float("nan")))
                ere_t = tfit.get("ere") if isinstance(tfit.get("ere"), dict) else {}
                r_t_fit = float(ere_t.get("r_eff_fm", float("nan"))) if isinstance(ere_t, dict) else float("nan")
                v2t_fit = float(ere_t.get("v2_fm3", float("nan"))) if isinstance(ere_t, dict) else float("nan")
                da_t = a_t_fit - float(trip["a_t_fm"])
                dr_t = r_t_fit - float(trip["r_t_fm"])

                ok_ar = bool(
                    math.isfinite(da_t)
                    and math.isfinite(dr_t)
                    and abs(float(da_t)) <= float(tol_a_fm)
                    and abs(float(dr_t)) <= float(tol_r_fm)
                )
                ok_v2t = bool(
                    math.isfinite(v2t_fit) and float(v2t_env_scan["min"]) <= float(v2t_fit) <= float(v2t_env_scan["max"])
                )
                outside_count += int(not ok_ar) + int(not ok_v2t)

                if math.isfinite(da_t) and math.isfinite(dr_t):
                    triplet_pen = max(abs(float(da_t)) / float(tol_a_fm), abs(float(dr_t)) / float(tol_r_fm))
                    max_triplet_pen = max(float(max_triplet_pen), float(triplet_pen))
                else:
                    max_triplet_pen = float("inf")

                max_dist_v2t = max(float(max_dist_v2t), float(dist_to_env(float(v2t_fit), v2t_env_scan)))

                sing_error = None
                v2s_pred_fm3: float | None = None
                r_s_fit_fm: float | None = None
                dr_s = float("nan")
                ok_rs = False
                ok_v2s = False

                try:
                    r1_t_fm = float(tfit["r1_fm"])
                    r2_t_fm = float(tfit["r2_fm"])
                    v1_hint = float(tfit["v1_mev"])
                    v2_hint = float(tfit["v2_mev"])

                    r2_s_fm = float(r2_t_fm)
                    if not (0.0 < r1_s_fm < r2_s_fm - 0.25):
                        raise ValueError("invalid singlet geometry (R1_s,R2_s)")

                    max_abs_delta_r1_over_lambda = max(
                        float(max_abs_delta_r1_over_lambda),
                        abs(float(r1_s_fm - r1_t_fm)) / float(lambda_pi_pm_fm),
                    )

                    sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                        a_s_target_fm=float(sing["a_s_fm"]),
                        r_s_target_fm=float(sing["r_s_fm"]),
                        r1_fm=float(r1_s_fm),
                        r2_fm=float(r2_s_fm),
                        cfg=cfg,
                        v1_hint_mev=float(v1_hint),
                        v2_hint_mev=float(v2_hint),
                        mu_mev=mu_mev,
                        hbarc_mev_fm=hbarc_mev_fm,
                    )
                    v1s = float(sing_fit["V1_s_MeV"])
                    v2s = float(sing_fit["V2_s_MeV"])
                    ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
                    if ere_s is None:
                        for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                            if req not in cfg:
                                raise ValueError(f"missing cfg[{req}] for ERE recompute")
                        r3_end = float(r2_s_fm) + float(cfg["L3_total_fm"])
                        vb_s = float(v2s) * float(cfg["barrier_coeff"])
                        vt_s = float(v2s) * float(cfg["tail_depth_coeff"])
                        segs_s = [
                            (float(r1_s_fm), -float(v1s)),
                            (float(r2_s_fm - r1_s_fm), -float(v2s)),
                            (float(cfg["Lb_fm"]), +float(vb_s)),
                            (float(cfg["Lt_fm"]), -float(vt_s)),
                        ]
                        ere_s = _fit_kcot_ere_segments(
                            r_end_fm=r3_end,
                            segments=segs_s,
                            mu_mev=mu_mev,
                            hbarc_mev_fm=hbarc_mev_fm,
                        )

                    v2s_pred_fm3 = float(ere_s["v2_fm3"])
                    r_s_fit_fm = float(ere_s["r_eff_fm"])
                    dr_s = float(r_s_fit_fm) - float(sing["r_s_fm"])
                    max_abs_dr_s = max(float(max_abs_dr_s), abs(float(dr_s)))

                    ok_rs = bool(
                        math.isfinite(r_s_fit_fm) and float(rs_env_scan["min"]) <= float(r_s_fit_fm) <= float(rs_env_scan["max"])
                    )
                    ok_v2s = bool(
                        math.isfinite(v2s_pred_fm3)
                        and float(v2s_env_scan["min"]) <= float(v2s_pred_fm3) <= float(v2s_env_scan["max"])
                    )
                    outside_count += int(not ok_rs) + int(not ok_v2s)
                    max_dist_v2s = max(float(max_dist_v2s), float(dist_to_env(float(v2s_pred_fm3), v2s_env_scan)))
                except Exception as e:
                    sing_error = str(e)
                    outside_count += 2
                    max_dist_v2s = float("inf")
                    max_abs_dr_s = float("inf")

                per_ds.append(
                    {
                        "label": label,
                        "eq_label": eq_label,
                        "a_t_fit_fm": float(a_t_fit) if math.isfinite(a_t_fit) else None,
                        "r_t_fit_fm": float(r_t_fit) if math.isfinite(r_t_fit) else None,
                        "v2t_fit_fm3": float(v2t_fit) if math.isfinite(v2t_fit) else None,
                        "a_t_fit_minus_obs_fm": float(da_t) if math.isfinite(da_t) else None,
                        "r_t_fit_minus_obs_fm": float(dr_t) if math.isfinite(dr_t) else None,
                        "within_triplet_ar_tolerance": bool(ok_ar),
                        "within_v2t_envelope": bool(ok_v2t),
                        "r1_s_fm": float(r1_s_fm),
                        "v2s_pred_fm3": float(v2s_pred_fm3) if v2s_pred_fm3 is not None and math.isfinite(v2s_pred_fm3) else None,
                        "r_s_fit_fm": float(r_s_fit_fm) if r_s_fit_fm is not None and math.isfinite(r_s_fit_fm) else None,
                        "r_s_fit_minus_obs_fm": float(dr_s) if math.isfinite(dr_s) else None,
                        "within_r_s_envelope": bool(ok_rs),
                        "within_v2s_envelope": bool(ok_v2s),
                        "error": sing_error,
                    }
                )

            within_all = bool(outside_count == 0)
            max_v2_total = float(max(float(max_dist_v2t), float(max_dist_v2s)))
            rank = (
                0 if within_all else 1,
                int(outside_count),
                float(max_v2_total) / float(scale_v2_fm3) if math.isfinite(max_v2_total) else float("inf"),
                float(max_abs_dr_s) / float(tol_r_fm) if math.isfinite(max_abs_dr_s) else float("inf"),
                float(max_triplet_pen) if math.isfinite(max_triplet_pen) else float("inf"),
                float(max_abs_delta_r1_over_lambda) if math.isfinite(max_abs_delta_r1_over_lambda) else float("inf"),
                float(r1_over),
            )
            scan_rows.append(
                {
                    "r1_s_over_lambda_pi_pm": float(r1_over),
                    "r1_s_fm": float(r1_s_fm),
                    "within_all": bool(within_all),
                    "outside_count": int(outside_count),
                    "max_dist_v2t_fm3": (float(max_dist_v2t) if math.isfinite(max_dist_v2t) else None),
                    "max_dist_v2s_fm3": (float(max_dist_v2s) if math.isfinite(max_dist_v2s) else None),
                    "max_abs_r_s_fit_minus_obs_fm": (float(max_abs_dr_s) if math.isfinite(max_abs_dr_s) else None),
                    "max_triplet_ar_penalty": (float(max_triplet_pen) if math.isfinite(max_triplet_pen) else None),
                    "max_abs_delta_r1_over_lambda_pi_pm": (
                        float(max_abs_delta_r1_over_lambda) if math.isfinite(max_abs_delta_r1_over_lambda) else None
                    ),
                    "per_dataset": per_ds,
                }
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_r1_over = float(r1_over)

        if best_r1_over is None:
            raise SystemExit("[fail] singlet R1/λπ scan produced no candidates")
        singlet_r1_over_lambda_pi_best = float(best_r1_over)
        barrier_tail_channel_split_kq_singlet_r1_scan = {
            "source_step": "7.13.8.2",
            "source_metrics_json": str(prev_metrics),
            "selected_channel_split_kq": {"k_t": float(k_t_sel), "q_t": float(q_t_sel), "k_s": float(k_s_sel), "q_s": float(q_s_sel)},
            "r1_over_lambda_grid": [float(x) for x in r1_over_lambda_grid],
            "selected": {"r1_s_over_lambda_pi_pm": float(best_r1_over), "rank": list(best_rank) if best_rank is not None else None},
            "observed_envelope_v2s_fm3": v2s_env_scan,
            "observed_envelope_v2t_fm3": v2t_env_scan,
            "observed_envelope_r_s_fm": rs_env_scan,
            "rows": scan_rows,
            "policy": {
                "r1_s_rule": "R1_s = (R1_s/λπ)·λπ is global across datasets; R2_s follows the triplet-fitted R2 per dataset.",
                "tail_len_over_lambda_fixed": 1.0,
                "barrier_len_fraction_fixed": 0.5,
                "triplet_ar_tolerance": {"tol_a_fm": float(tol_a_fm), "tol_r_fm": float(tol_r_fm)},
                "selection_rule": "Prefer any R1_s/λπ that makes singlet (r_s,v2s) within envelope given the frozen (k_t,q_t,k_s,q_s); tie-break by minimal max_dist(v2), then |Δr_s|, then triplet AR penalty, then minimal |R1_s-R1_t|/λπ.",
            },
        }

    if step == "7.13.8.4":
        if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
            raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.8.4")

        # This step builds on two fixed outputs:
        # - Step 7.13.8.2 (frozen channel-split (k_t,q_t)/(k_s,q_s))
        # - Step 7.13.8.3 (singlet R1_s/λπ scan); here we freeze a deterministic R1_s/λπ choice
        #   that best matches r_s across datasets, then scan one additional DOF: singlet R2_s/λπ.
        prev_kq_metrics = out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_metrics.json"
        if not prev_kq_metrics.exists():
            raise SystemExit(
                "[fail] missing Step 7.13.8.2 fixed output needed for Step 7.13.8.4.\n"
                "Run first:\n"
                "  python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.13.8.2\n"
                f"Expected: {prev_kq_metrics}"
            )
        prev_kq = _load_json(prev_kq_metrics)
        prev_kq_scan = prev_kq.get("barrier_tail_channel_split_kq_scan")
        if not isinstance(prev_kq_scan, dict):
            raise SystemExit(f"[fail] invalid 7.13.8.2 metrics: missing barrier_tail_channel_split_kq_scan: {prev_kq_metrics}")
        prev_kq_sel = prev_kq_scan.get("selected")
        if not isinstance(prev_kq_sel, dict):
            raise SystemExit(f"[fail] invalid 7.13.8.2 metrics: missing selected: {prev_kq_metrics}")

        k_t_sel = float(prev_kq_sel.get("k_t", float("nan")))
        q_t_sel = float(prev_kq_sel.get("q_t", float("nan")))
        k_s_sel = float(prev_kq_sel.get("k_s", float("nan")))
        q_s_sel = float(prev_kq_sel.get("q_s", float("nan")))
        if not (
            math.isfinite(k_t_sel)
            and math.isfinite(q_t_sel)
            and math.isfinite(k_s_sel)
            and math.isfinite(q_s_sel)
            and k_t_sel >= 0.0
            and q_t_sel >= 0.0
            and k_s_sel >= 0.0
            and q_s_sel >= 0.0
        ):
            raise SystemExit(f"[fail] invalid 7.13.8.2 selected (k_t,q_t,k_s,q_s) in: {prev_kq_metrics}")

        # Make the selected channel-split parameters available downstream (plot/CSV/metrics).
        barrier_height_factor_kq_t_best = float(k_t_sel)
        tail_depth_factor_kq_t_best = float(q_t_sel)
        barrier_height_factor_kq_s_best = float(k_s_sel)
        tail_depth_factor_kq_s_best = float(q_s_sel)

        prev_r1_metrics = out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_singlet_r1_scan_metrics.json"
        if not prev_r1_metrics.exists():
            raise SystemExit(
                "[fail] missing Step 7.13.8.3 fixed output needed for Step 7.13.8.4.\n"
                "Run first:\n"
                "  python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.13.8.3\n"
                f"Expected: {prev_r1_metrics}"
            )
        prev_r1 = _load_json(prev_r1_metrics)
        prev_r1_scan = prev_r1.get("barrier_tail_channel_split_kq_singlet_r1_scan")
        if not isinstance(prev_r1_scan, dict):
            raise SystemExit(f"[fail] invalid 7.13.8.3 metrics: missing barrier_tail_channel_split_kq_singlet_r1_scan: {prev_r1_metrics}")
        prev_r1_rows = prev_r1_scan.get("rows")
        if not isinstance(prev_r1_rows, list):
            raise SystemExit(f"[fail] invalid 7.13.8.3 metrics: missing rows: {prev_r1_metrics}")

        # Freeze R1_s/λπ by a deterministic rule:
        # Prefer any R1_s/λπ that makes r_s(fit) within the observed envelope for all datasets.
        # If multiple exist, tie-break by minimal max_dist(v2s), then minimal max |Δr_s|.
        def _row_ok_rs_all(row: dict[str, object]) -> bool:
            per_ds = row.get("per_dataset")
            if not isinstance(per_ds, list) or len(per_ds) == 0:
                return False
            for pd in per_ds:
                if not isinstance(pd, dict):
                    return False
                if pd.get("error") is not None:
                    return False
                if not bool(pd.get("within_r_s_envelope")):
                    return False
            return True

        def _r1_row_key(row: dict[str, object]) -> tuple[float, float, float]:
            max_dist_v2s = float(row.get("max_dist_v2s_fm3", float("inf")))
            max_abs_dr_s = float(row.get("max_abs_r_s_fit_minus_obs_fm", float("inf")))
            r1_over = float(row.get("r1_s_over_lambda_pi_pm", float("inf")))
            return (max_dist_v2s, max_abs_dr_s, r1_over)

        r1_candidates: list[dict[str, object]] = [
            row for row in prev_r1_rows if isinstance(row, dict) and "r1_s_over_lambda_pi_pm" in row and _row_ok_rs_all(row)
        ]
        if r1_candidates:
            best_r1_row = min(r1_candidates, key=_r1_row_key)
        else:
            best_r1_row = min(
                (row for row in prev_r1_rows if isinstance(row, dict) and "r1_s_over_lambda_pi_pm" in row),
                key=lambda row: (
                    float(row.get("max_abs_r_s_fit_minus_obs_fm", float("inf"))),
                    float(row.get("max_dist_v2s_fm3", float("inf"))),
                    float(row.get("r1_s_over_lambda_pi_pm", float("inf"))),
                ),
            )

        r1_over_fixed = float(best_r1_row.get("r1_s_over_lambda_pi_pm", float("nan")))
        if not math.isfinite(r1_over_fixed):
            raise SystemExit("[fail] could not determine fixed R1_s/λπ from 7.13.8.3 scan")
        singlet_r1_over_lambda_pi_best = float(r1_over_fixed)
        r1_s_fm_fixed = float(r1_over_fixed) * float(lambda_pi_pm_fm)

        v2s_obs_list_scan = [float(d["singlet"]["v2s_fm3"]) for d in datasets]
        v2t_obs_list_scan = [float(d["triplet"]["v2t_fm3"]) for d in datasets]
        rs_obs_list_scan = [float(d["singlet"]["r_s_fm"]) for d in datasets]
        v2s_env_scan = {"min": float(min(v2s_obs_list_scan)), "max": float(max(v2s_obs_list_scan))}
        v2t_env_scan = {"min": float(min(v2t_obs_list_scan)), "max": float(max(v2t_obs_list_scan))}
        rs_env_scan = {"min": float(min(rs_obs_list_scan)), "max": float(max(rs_obs_list_scan))}

        def dist_to_env(x: float, env: dict[str, float]) -> float:
            if not math.isfinite(x):
                return float("inf")
            lo = float(env["min"])
            hi = float(env["max"])
            if lo <= x <= hi:
                return 0.0
            return float(min(abs(x - lo), abs(x - hi)))

        tol_a_fm = 0.2
        tol_r_fm = 0.05
        scale_v2_fm3 = 0.05

        # Precompute triplet fits once (depends only on the fixed (k_t,q_t) selection).
        triplet_fit_by_eq: dict[int, dict[str, object]] = {}
        for d in datasets:
            eq_label = int(d["eq_label"])
            trip = d["triplet"]
            base_fit_t = _fit_triplet_three_range_barrier_tail_pion_constrained_free_depth(
                b_mev=b_mev,
                targets=trip,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
                lambda_pi_fm=float(lambda_pi_pm_fm),
                tail_len_over_lambda=1.0,
                barrier_len_fraction=0.5,
                barrier_height_factor=float(k_t_sel),
                tail_depth_factor=float(q_t_sel),
            )
            triplet_fit_by_eq[eq_label] = base_fit_t

        # Precompute singlet barrier+tail config once (depends only on fixed (k_s,q_s) selection).
        cfg_any = _barrier_tail_config_free_depth(
            lambda_pi_fm=float(lambda_pi_pm_fm),
            tail_len_over_lambda=1.0,
            barrier_len_fraction=0.5,
            barrier_height_factor=float(k_s_sel),
            tail_depth_factor=float(q_s_sel),
        )
        if not isinstance(cfg_any, dict):
            raise SystemExit("[fail] invalid barrier_tail_config for step 7.13.8.4")
        cfg = {str(k): float(v) for k, v in cfg_any.items() if math.isfinite(float(v))}

        # Scan only the singlet outer boundary ratio (global): R2_s = (R2_s/λπ)·λπ.
        r2_over_lambda_grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.05]

        scan_rows: list[dict[str, object]] = []
        best_rank: tuple[int, int, float, float, float, float, float] | None = None
        best_r2_over: float | None = None

        for r2_over in r2_over_lambda_grid:
            r2_s_fm = float(r2_over) * float(lambda_pi_pm_fm)
            per_ds: list[dict[str, object]] = []
            outside_count = 0
            max_dist_v2t = 0.0
            max_dist_v2s = 0.0
            max_abs_dr_s = 0.0
            max_triplet_pen = 0.0
            max_abs_delta_r2_over_lambda = 0.0

            for d in datasets:
                label = str(d["label"])
                eq_label = int(d["eq_label"])
                trip = d["triplet"]
                sing = d["singlet"]

                tfit = triplet_fit_by_eq[eq_label]
                a_t_fit = float(tfit.get("a_exact_fm", float("nan")))
                ere_t = tfit.get("ere") if isinstance(tfit.get("ere"), dict) else {}
                r_t_fit = float(ere_t.get("r_eff_fm", float("nan"))) if isinstance(ere_t, dict) else float("nan")
                v2t_fit = float(ere_t.get("v2_fm3", float("nan"))) if isinstance(ere_t, dict) else float("nan")
                da_t = a_t_fit - float(trip["a_t_fm"])
                dr_t = r_t_fit - float(trip["r_t_fm"])

                ok_ar = bool(
                    math.isfinite(da_t)
                    and math.isfinite(dr_t)
                    and abs(float(da_t)) <= float(tol_a_fm)
                    and abs(float(dr_t)) <= float(tol_r_fm)
                )
                ok_v2t = bool(
                    math.isfinite(v2t_fit) and float(v2t_env_scan["min"]) <= float(v2t_fit) <= float(v2t_env_scan["max"])
                )
                outside_count += int(not ok_ar) + int(not ok_v2t)

                if math.isfinite(da_t) and math.isfinite(dr_t):
                    triplet_pen = max(abs(float(da_t)) / float(tol_a_fm), abs(float(dr_t)) / float(tol_r_fm))
                    max_triplet_pen = max(float(max_triplet_pen), float(triplet_pen))
                else:
                    max_triplet_pen = float("inf")

                max_dist_v2t = max(float(max_dist_v2t), float(dist_to_env(float(v2t_fit), v2t_env_scan)))

                sing_error = None
                v2s_pred_fm3: float | None = None
                r_s_fit_fm: float | None = None
                dr_s = float("nan")
                ok_rs = False
                ok_v2s = False

                try:
                    r2_t_fm = float(tfit["r2_fm"])
                    v1_hint = float(tfit["v1_mev"])
                    v2_hint = float(tfit["v2_mev"])

                    if not (0.0 < r1_s_fm_fixed < r2_s_fm - 0.25):
                        raise ValueError("invalid singlet geometry (R1_s,R2_s)")

                    max_abs_delta_r2_over_lambda = max(
                        float(max_abs_delta_r2_over_lambda),
                        abs(float(r2_s_fm - r2_t_fm)) / float(lambda_pi_pm_fm),
                    )

                    sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                        a_s_target_fm=float(sing["a_s_fm"]),
                        r_s_target_fm=float(sing["r_s_fm"]),
                        r1_fm=float(r1_s_fm_fixed),
                        r2_fm=float(r2_s_fm),
                        cfg=cfg,
                        v1_hint_mev=float(v1_hint),
                        v2_hint_mev=float(v2_hint),
                        mu_mev=mu_mev,
                        hbarc_mev_fm=hbarc_mev_fm,
                    )
                    v1s = float(sing_fit["V1_s_MeV"])
                    v2s = float(sing_fit["V2_s_MeV"])
                    ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
                    if ere_s is None:
                        for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                            if req not in cfg:
                                raise ValueError(f"missing cfg[{req}] for ERE recompute")
                        r3_end = float(r2_s_fm) + float(cfg["L3_total_fm"])
                        vb_s = float(v2s) * float(cfg["barrier_coeff"])
                        vt_s = float(v2s) * float(cfg["tail_depth_coeff"])
                        segs_s = [
                            (float(r1_s_fm_fixed), -float(v1s)),
                            (float(r2_s_fm - r1_s_fm_fixed), -float(v2s)),
                            (float(cfg["Lb_fm"]), +float(vb_s)),
                            (float(cfg["Lt_fm"]), -float(vt_s)),
                        ]
                        ere_s = _fit_kcot_ere_segments(
                            r_end_fm=r3_end, segments=segs_s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
                        )

                    if not isinstance(ere_s, dict):
                        raise ValueError("missing ERE for singlet fit")
                    v2s_pred_fm3 = float(ere_s.get("v2_fm3", float("nan")))
                    r_s_fit_fm = float(ere_s.get("r_eff_fm", float("nan")))
                    dr_s = float(r_s_fit_fm) - float(sing["r_s_fm"])
                    ok_rs = bool(
                        math.isfinite(r_s_fit_fm) and float(rs_env_scan["min"]) <= float(r_s_fit_fm) <= float(rs_env_scan["max"])
                    )
                    ok_v2s = bool(
                        math.isfinite(v2s_pred_fm3)
                        and float(v2s_env_scan["min"]) <= float(v2s_pred_fm3) <= float(v2s_env_scan["max"])
                    )
                    outside_count += int(not ok_rs) + int(not ok_v2s)
                    max_dist_v2s = max(float(max_dist_v2s), float(dist_to_env(float(v2s_pred_fm3), v2s_env_scan)))
                    max_abs_dr_s = max(float(max_abs_dr_s), abs(float(dr_s)))
                except Exception as e:
                    sing_error = str(e)
                    outside_count += 2
                    max_dist_v2s = float("inf")
                    max_abs_dr_s = float("inf")

                per_ds.append(
                    {
                        "label": label,
                        "eq_label": eq_label,
                        "a_t_fit_fm": float(a_t_fit) if math.isfinite(a_t_fit) else None,
                        "r_t_fit_fm": float(r_t_fit) if math.isfinite(r_t_fit) else None,
                        "v2t_fit_fm3": float(v2t_fit) if math.isfinite(v2t_fit) else None,
                        "a_t_fit_minus_obs_fm": float(da_t) if math.isfinite(da_t) else None,
                        "r_t_fit_minus_obs_fm": float(dr_t) if math.isfinite(dr_t) else None,
                        "within_triplet_ar_tolerance": bool(ok_ar),
                        "within_v2t_envelope": bool(ok_v2t),
                        "r1_s_fm": float(r1_s_fm_fixed),
                        "r2_s_fm": float(r2_s_fm),
                        "v2s_pred_fm3": float(v2s_pred_fm3) if v2s_pred_fm3 is not None and math.isfinite(v2s_pred_fm3) else None,
                        "r_s_fit_fm": float(r_s_fit_fm) if r_s_fit_fm is not None and math.isfinite(r_s_fit_fm) else None,
                        "r_s_fit_minus_obs_fm": float(dr_s) if math.isfinite(dr_s) else None,
                        "within_r_s_envelope": bool(ok_rs),
                        "within_v2s_envelope": bool(ok_v2s),
                        "error": sing_error,
                    }
                )

            within_all = bool(outside_count == 0)
            max_v2_total = float(max(float(max_dist_v2t), float(max_dist_v2s)))
            rank = (
                0 if within_all else 1,
                int(outside_count),
                float(max_v2_total) / float(scale_v2_fm3) if math.isfinite(max_v2_total) else float("inf"),
                float(max_abs_dr_s) / float(tol_r_fm) if math.isfinite(max_abs_dr_s) else float("inf"),
                float(max_triplet_pen) if math.isfinite(max_triplet_pen) else float("inf"),
                float(max_abs_delta_r2_over_lambda) if math.isfinite(max_abs_delta_r2_over_lambda) else float("inf"),
                float(r2_over),
            )
            scan_rows.append(
                {
                    "r2_s_over_lambda_pi_pm": float(r2_over),
                    "r2_s_fm": float(r2_s_fm),
                    "within_all": bool(within_all),
                    "outside_count": int(outside_count),
                    "max_dist_v2t_fm3": (float(max_dist_v2t) if math.isfinite(max_dist_v2t) else None),
                    "max_dist_v2s_fm3": (float(max_dist_v2s) if math.isfinite(max_dist_v2s) else None),
                    "max_abs_r_s_fit_minus_obs_fm": (float(max_abs_dr_s) if math.isfinite(max_abs_dr_s) else None),
                    "max_triplet_ar_penalty": (float(max_triplet_pen) if math.isfinite(max_triplet_pen) else None),
                    "max_abs_delta_r2_over_lambda_pi_pm": (
                        float(max_abs_delta_r2_over_lambda) if math.isfinite(max_abs_delta_r2_over_lambda) else None
                    ),
                    "per_dataset": per_ds,
                }
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_r2_over = float(r2_over)

        if best_r2_over is None:
            raise SystemExit("[fail] singlet R2/λπ scan produced no candidates")
        singlet_r2_over_lambda_pi_best = float(best_r2_over)
        barrier_tail_channel_split_kq_singlet_r2_scan = {
            "source_step": "7.13.8.2",
            "source_metrics_json": str(prev_kq_metrics),
            "source_r1_scan_step": "7.13.8.3",
            "source_r1_scan_metrics_json": str(prev_r1_metrics),
            "selected_channel_split_kq": {"k_t": float(k_t_sel), "q_t": float(q_t_sel), "k_s": float(k_s_sel), "q_s": float(q_s_sel)},
            "r1_s_over_lambda_pi_pm_fixed": float(r1_over_fixed),
            "r2_over_lambda_grid": [float(x) for x in r2_over_lambda_grid],
            "selected": {"r2_s_over_lambda_pi_pm": float(best_r2_over), "rank": list(best_rank) if best_rank is not None else None},
            "observed_envelope_v2s_fm3": v2s_env_scan,
            "observed_envelope_v2t_fm3": v2t_env_scan,
            "observed_envelope_r_s_fm": rs_env_scan,
            "rows": scan_rows,
            "policy": {
                "r1_freeze_rule": (
                    "Freeze R1_s/λπ using the Step 7.13.8.3 scan: pick any row where r_s(fit) is within the observed envelope for all datasets; "
                    "tie-break by minimal max_dist(v2s), then minimal max |Δr_s|."
                ),
                "r2_s_rule": "R2_s = (R2_s/λπ)·λπ is global across datasets; barrier+tail lengths beyond R2_s remain fixed in units of λπ.",
                "tail_len_over_lambda_fixed": 1.0,
                "barrier_len_fraction_fixed": 0.5,
                "triplet_ar_tolerance": {"tol_a_fm": float(tol_a_fm), "tol_r_fm": float(tol_r_fm)},
                "selection_rule": "Prefer any R2_s/λπ that makes singlet (r_s,v2s) within envelope given the frozen (k_t,q_t,k_s,q_s) and fixed R1_s; tie-break by minimal max_dist(v2), then |Δr_s|, then triplet AR penalty, then minimal |R2_s-R2_t|/λπ.",
            },
        }

    if step == "7.13.8.5":
        if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
            raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.8.5")

        # Build on Step 7.13.8.4 (singlet geometry split succeeded). Here we freeze the singlet
        # geometry (R1_s/λπ, R2_s/λπ) and the singlet (k_s,q_s), and scan one additional geometric DOF
        # for the triplet: the barrier/tail split fraction beyond R2, plus a small re-scan of k_t.
        prev_r2_metrics = out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_singlet_r2_scan_metrics.json"
        if not prev_r2_metrics.exists():
            raise SystemExit(
                "[fail] missing Step 7.13.8.4 fixed output needed for Step 7.13.8.5.\n"
                "Run first:\n"
                "  python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.13.8.4\n"
                f"Expected: {prev_r2_metrics}"
            )
        prev_r2 = _load_json(prev_r2_metrics)
        prev_r2_scan = prev_r2.get("barrier_tail_channel_split_kq_singlet_r2_scan")
        if not isinstance(prev_r2_scan, dict):
            raise SystemExit(
                f"[fail] invalid 7.13.8.4 metrics: missing barrier_tail_channel_split_kq_singlet_r2_scan: {prev_r2_metrics}"
            )
        prev_sel_kq = prev_r2_scan.get("selected_channel_split_kq")
        if not isinstance(prev_sel_kq, dict):
            raise SystemExit(f"[fail] invalid 7.13.8.4 metrics: missing selected_channel_split_kq: {prev_r2_metrics}")

        k_t_prev = float(prev_sel_kq.get("k_t", float("nan")))
        q_t_fixed = float(prev_sel_kq.get("q_t", float("nan")))
        k_s_sel = float(prev_sel_kq.get("k_s", float("nan")))
        q_s_sel = float(prev_sel_kq.get("q_s", float("nan")))
        if not (
            math.isfinite(k_t_prev) and math.isfinite(q_t_fixed) and math.isfinite(k_s_sel) and math.isfinite(q_s_sel)
        ):
            raise SystemExit(f"[fail] invalid 7.13.8.4 selected_channel_split_kq in: {prev_r2_metrics}")

        r1_over_fixed = float(prev_r2_scan.get("r1_s_over_lambda_pi_pm_fixed", float("nan")))
        prev_r2_sel = prev_r2_scan.get("selected")
        if not isinstance(prev_r2_sel, dict):
            raise SystemExit(f"[fail] invalid 7.13.8.4 metrics: missing selected: {prev_r2_metrics}")
        r2_over_fixed = float(prev_r2_sel.get("r2_s_over_lambda_pi_pm", float("nan")))
        if not (math.isfinite(r1_over_fixed) and math.isfinite(r2_over_fixed)):
            raise SystemExit(f"[fail] invalid 7.13.8.4 singlet geometry in: {prev_r2_metrics}")

        # Freeze singlet geometry and (k_s,q_s) from 7.13.8.4.
        singlet_r1_over_lambda_pi_best = float(r1_over_fixed)
        singlet_r2_over_lambda_pi_best = float(r2_over_fixed)
        barrier_height_factor_kq_s_best = float(k_s_sel)
        tail_depth_factor_kq_s_best = float(q_s_sel)

        # Freeze triplet tail depth factor from 7.13.8.4, but allow a small re-scan on k_t.
        tail_depth_factor_kq_t_best = float(q_t_fixed)

        v2s_obs_list_scan = [float(d["singlet"]["v2s_fm3"]) for d in datasets]
        v2t_obs_list_scan = [float(d["triplet"]["v2t_fm3"]) for d in datasets]
        rs_obs_list_scan = [float(d["singlet"]["r_s_fm"]) for d in datasets]
        v2s_env_scan = {"min": float(min(v2s_obs_list_scan)), "max": float(max(v2s_obs_list_scan))}
        v2t_env_scan = {"min": float(min(v2t_obs_list_scan)), "max": float(max(v2t_obs_list_scan))}
        rs_env_scan = {"min": float(min(rs_obs_list_scan)), "max": float(max(rs_obs_list_scan))}

        def dist_to_env(x: float, env: dict[str, float]) -> float:
            if not math.isfinite(x):
                return float("inf")
            lo = float(env["min"])
            hi = float(env["max"])
            if lo <= x <= hi:
                return 0.0
            return float(min(abs(x - lo), abs(x - hi)))

        def margin_to_env(x: float, env: dict[str, float]) -> float:
            # Positive only when inside; otherwise negative (by distance).
            if not math.isfinite(x):
                return float("-inf")
            lo = float(env["min"])
            hi = float(env["max"])
            if lo <= x <= hi:
                return float(min(x - lo, hi - x))
            return -dist_to_env(x, env)

        tol_a_fm = 0.2
        tol_r_fm = 0.05

        # Precompute the frozen singlet fit once (does not depend on the triplet scan params).
        r1_s_fm_fixed = float(singlet_r1_over_lambda_pi_best) * float(lambda_pi_pm_fm)
        r2_s_fm_fixed = float(singlet_r2_over_lambda_pi_best) * float(lambda_pi_pm_fm)
        cfg_s_any = _barrier_tail_config_free_depth(
            lambda_pi_fm=float(lambda_pi_pm_fm),
            tail_len_over_lambda=1.0,
            barrier_len_fraction=0.5,
            barrier_height_factor=float(barrier_height_factor_kq_s_best),
            tail_depth_factor=float(tail_depth_factor_kq_s_best),
        )
        if not isinstance(cfg_s_any, dict):
            raise SystemExit("[fail] invalid barrier_tail_config for step 7.13.8.5")
        cfg_s = {str(k): float(v) for k, v in cfg_s_any.items() if math.isfinite(float(v))}

        singlet_fixed_by_eq: dict[int, dict[str, float]] = {}
        for d in datasets:
            eq_label = int(d["eq_label"])
            sing = d["singlet"]
            sfit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                a_s_target_fm=float(sing["a_s_fm"]),
                r_s_target_fm=float(sing["r_s_fm"]),
                r1_fm=float(r1_s_fm_fixed),
                r2_fm=float(r2_s_fm_fixed),
                cfg=cfg_s,
                v1_hint_mev=120.0,
                v2_hint_mev=10.0,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            ere_s = sfit.get("ere") if isinstance(sfit.get("ere"), dict) else None
            if ere_s is None:
                vb = float(sfit["V2_s_MeV"]) * float(cfg_s["barrier_coeff"])
                vt = float(sfit["V2_s_MeV"]) * float(cfg_s["tail_depth_coeff"])
                segs_s = [
                    (float(r1_s_fm_fixed), -float(sfit["V1_s_MeV"])),
                    (float(r2_s_fm_fixed - r1_s_fm_fixed), -float(sfit["V2_s_MeV"])),
                    (float(cfg_s["Lb_fm"]), +float(vb)),
                    (float(cfg_s["Lt_fm"]), -float(vt)),
                ]
                r3_end = float(r2_s_fm_fixed) + float(cfg_s["L3_total_fm"])
                ere_s = _fit_kcot_ere_segments(r_end_fm=r3_end, segments=segs_s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
            singlet_fixed_by_eq[eq_label] = {
                "r_s_fit_fm": float(ere_s["r_eff_fm"]),
                "v2s_pred_fm3": float(ere_s["v2_fm3"]),
            }

        # Scan: triplet barrier/tail split fraction and a small grid on k_t (q_t fixed from 7.13.8.4).
        barrier_len_fraction_grid = [0.1, 0.15, 0.2, 0.25, 0.3]
        k_t_grid = [0.0, 0.5, 1.0, 2.0, float(k_t_prev)]

        scan_rows: list[dict[str, object]] = []
        best_rank: tuple[int, int, float, float, float, float, float, float] | None = None
        best_frac: float | None = None
        best_k: float | None = None

        for frac in barrier_len_fraction_grid:
            for k_t in k_t_grid:
                per_ds: list[dict[str, object]] = []
                outside_count = 0
                max_dist_v2t = 0.0
                min_margin_v2t = float("inf")
                max_triplet_pen = 0.0

                for d in datasets:
                    label = str(d["label"])
                    eq_label = int(d["eq_label"])
                    trip = d["triplet"]
                    sing = d["singlet"]

                    tfit = _fit_triplet_three_range_barrier_tail_pion_constrained_free_depth(
                        b_mev=b_mev,
                        targets=trip,
                        mu_mev=mu_mev,
                        hbarc_mev_fm=hbarc_mev_fm,
                        lambda_pi_fm=float(lambda_pi_pm_fm),
                        tail_len_over_lambda=1.0,
                        barrier_len_fraction=float(frac),
                        barrier_height_factor=float(k_t),
                        tail_depth_factor=float(q_t_fixed),
                    )
                    a_t_fit = float(tfit.get("a_exact_fm", float("nan")))
                    ere_t = tfit.get("ere") if isinstance(tfit.get("ere"), dict) else {}
                    r_t_fit = float(ere_t.get("r_eff_fm", float("nan"))) if isinstance(ere_t, dict) else float("nan")
                    v2t_fit = float(ere_t.get("v2_fm3", float("nan"))) if isinstance(ere_t, dict) else float("nan")

                    da_t = a_t_fit - float(trip["a_t_fm"])
                    dr_t = r_t_fit - float(trip["r_t_fm"])

                    ok_ar = bool(
                        math.isfinite(da_t)
                        and math.isfinite(dr_t)
                        and abs(float(da_t)) <= float(tol_a_fm)
                        and abs(float(dr_t)) <= float(tol_r_fm)
                    )
                    ok_v2t = bool(
                        math.isfinite(v2t_fit) and float(v2t_env_scan["min"]) <= float(v2t_fit) <= float(v2t_env_scan["max"])
                    )

                    dist_v2t = dist_to_env(float(v2t_fit), v2t_env_scan)
                    max_dist_v2t = max(float(max_dist_v2t), float(dist_v2t))
                    min_margin_v2t = min(float(min_margin_v2t), float(margin_to_env(float(v2t_fit), v2t_env_scan)))

                    if math.isfinite(da_t) and math.isfinite(dr_t):
                        triplet_pen = max(abs(float(da_t)) / float(tol_a_fm), abs(float(dr_t)) / float(tol_r_fm))
                        max_triplet_pen = max(float(max_triplet_pen), float(triplet_pen))
                    else:
                        max_triplet_pen = float("inf")

                    # Frozen singlet check (from 7.13.8.4).
                    sf = singlet_fixed_by_eq.get(eq_label, {})
                    r_s_fit = float(sf.get("r_s_fit_fm", float("nan")))
                    v2s_pred = float(sf.get("v2s_pred_fm3", float("nan")))
                    ok_rs = bool(math.isfinite(r_s_fit) and float(rs_env_scan["min"]) <= float(r_s_fit) <= float(rs_env_scan["max"]))
                    ok_v2s = bool(
                        math.isfinite(v2s_pred) and float(v2s_env_scan["min"]) <= float(v2s_pred) <= float(v2s_env_scan["max"])
                    )

                    outside_count += int(not ok_ar) + int(not ok_v2t) + int(not ok_rs) + int(not ok_v2s)

                    per_ds.append(
                        {
                            "label": label,
                            "eq_label": eq_label,
                            "a_t_fit_fm": (float(a_t_fit) if math.isfinite(a_t_fit) else None),
                            "r_t_fit_fm": (float(r_t_fit) if math.isfinite(r_t_fit) else None),
                            "v2t_fit_fm3": (float(v2t_fit) if math.isfinite(v2t_fit) else None),
                            "a_t_fit_minus_obs_fm": (float(da_t) if math.isfinite(da_t) else None),
                            "r_t_fit_minus_obs_fm": (float(dr_t) if math.isfinite(dr_t) else None),
                            "within_triplet_ar_tolerance": bool(ok_ar),
                            "within_v2t_envelope": bool(ok_v2t),
                            "r1_s_fm": float(r1_s_fm_fixed),
                            "r2_s_fm": float(r2_s_fm_fixed),
                            "v2s_pred_fm3": (float(v2s_pred) if math.isfinite(v2s_pred) else None),
                            "r_s_fit_fm": (float(r_s_fit) if math.isfinite(r_s_fit) else None),
                            "r_s_fit_minus_obs_fm": (
                                float(r_s_fit - float(sing["r_s_fm"])) if math.isfinite(r_s_fit) else None
                            ),
                            "within_r_s_envelope": bool(ok_rs),
                            "within_v2s_envelope": bool(ok_v2s),
                            "error": None,
                        }
                    )

                within_all = bool(outside_count == 0)

                # Selection: prefer strict pass; then fewer violations; then keep v2t away from the envelope edge.
                v2_margin_rank = -float(min_margin_v2t) if math.isfinite(min_margin_v2t) else float("inf")
                rank = (
                    0 if within_all else 1,
                    int(outside_count),
                    float(v2_margin_rank),
                    float(max_triplet_pen) if math.isfinite(max_triplet_pen) else float("inf"),
                    abs(float(frac) - 0.5),
                    abs(float(k_t) - float(k_t_prev)),
                    float(frac),
                    float(k_t),
                )
                scan_rows.append(
                    {
                        "barrier_len_fraction_t": float(frac),
                        "k_t": float(k_t),
                        "q_t_fixed": float(q_t_fixed),
                        "within_all": bool(within_all),
                        "outside_count": int(outside_count),
                        "max_dist_v2t_fm3": (float(max_dist_v2t) if math.isfinite(max_dist_v2t) else None),
                        "min_margin_v2t_fm3": (float(min_margin_v2t) if math.isfinite(min_margin_v2t) else None),
                        "max_triplet_ar_penalty": (float(max_triplet_pen) if math.isfinite(max_triplet_pen) else None),
                        "per_dataset": per_ds,
                    }
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_frac = float(frac)
                    best_k = float(k_t)

        if best_frac is None or best_k is None:
            raise SystemExit("[fail] triplet barrier fraction scan produced no candidates")

        triplet_barrier_len_fraction_best = float(best_frac)
        barrier_height_factor_kq_t_best = float(best_k)
        barrier_tail_channel_split_kq_triplet_barrier_fraction_scan = {
            "source_step": "7.13.8.4",
            "source_metrics_json": str(prev_r2_metrics),
            "selected_channel_split_kq_from_7_13_8_4": {
                "k_t_prev": float(k_t_prev),
                "q_t_fixed": float(q_t_fixed),
                "k_s": float(k_s_sel),
                "q_s": float(q_s_sel),
            },
            "frozen_singlet_geometry_from_7_13_8_4": {
                "r1_s_over_lambda_pi_pm_fixed": float(r1_over_fixed),
                "r2_s_over_lambda_pi_pm_fixed": float(r2_over_fixed),
            },
            "scan_grids": {
                "barrier_len_fraction_t_grid": [float(x) for x in barrier_len_fraction_grid],
                "k_t_grid": [float(x) for x in k_t_grid],
            },
            "selected": {
                "barrier_len_fraction_t": float(best_frac),
                "k_t": float(best_k),
                "q_t_fixed": float(q_t_fixed),
                "rank": list(best_rank) if best_rank is not None else None,
            },
            "observed_envelope_v2s_fm3": v2s_env_scan,
            "observed_envelope_v2t_fm3": v2t_env_scan,
            "observed_envelope_r_s_fm": rs_env_scan,
            "policy": {
                "frozen_from_step": "Freeze singlet geometry (R1_s/λπ, R2_s/λπ) and (k_s,q_s) from Step 7.13.8.4.",
                "triplet_scan": (
                    "Scan triplet barrier_len_fraction_t (within L3=λπ) and a small grid on k_t, with q_t frozen; "
                    "require triplet (a_t,r_t) within tolerance and v2t within envelope, while keeping the frozen singlet (r_s,v2s) within envelope."
                ),
                "triplet_ar_tolerance": {"tol_a_fm": float(tol_a_fm), "tol_r_fm": float(tol_r_fm)},
                "selection_rule": (
                    "Prefer strict pass (within_all), then fewer violations, then maximize the minimum margin of v2t to the envelope edge, "
                    "then minimize triplet AR penalty, then prefer barrier_len_fraction_t closer to 0.5, then prefer k_t closer to the previous selection."
                ),
            },
            "rows": scan_rows,
        }

    results: list[dict[str, object]] = []
    for d in datasets:
        label = str(d["label"])
        trip = d["triplet"]
        sing = d["singlet"]

        if step in ("7.13.5", "7.13.6"):
            base_fit_t = _fit_triplet_three_range_barrier_tail_pion_constrained(
                b_mev=b_mev,
                targets=trip,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
                lambda_pi_fm=float(lambda_pi_pm_fm),
                tail_len_over_lambda=1.0,
                barrier_len_fraction=0.5,
                barrier_height_factor=(float(barrier_height_factor_best) if step == "7.13.6" else 1.0),
            )
        elif step == "7.13.7":
            base_fit_t = _fit_triplet_three_range_barrier_tail_pion_constrained_free_depth(
                b_mev=b_mev,
                targets=trip,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
                lambda_pi_fm=float(lambda_pi_pm_fm),
                tail_len_over_lambda=1.0,
                barrier_len_fraction=0.5,
                barrier_height_factor=2.0,
                tail_depth_factor=float(tail_depth_factor_best) if tail_depth_factor_best is not None else 4.0,
            )
        elif step in ("7.13.8", "7.13.8.1", "7.13.8.2", "7.13.8.3", "7.13.8.4", "7.13.8.5"):
            if step in ("7.13.8.2", "7.13.8.3", "7.13.8.4", "7.13.8.5"):
                if barrier_height_factor_kq_t_best is None or tail_depth_factor_kq_t_best is None:
                    raise SystemExit("[fail] missing selected (k_t,q_t) for step 7.13.8.2/7.13.8.4/7.13.8.5")
                k_use = float(barrier_height_factor_kq_t_best)
                q_use = float(tail_depth_factor_kq_t_best)
            else:
                k_use = float(barrier_height_factor_kq_best) if barrier_height_factor_kq_best is not None else 2.0
                q_use = float(tail_depth_factor_kq_best) if tail_depth_factor_kq_best is not None else 0.7
            if step == "7.13.8.5":
                if triplet_barrier_len_fraction_best is None or not math.isfinite(triplet_barrier_len_fraction_best):
                    raise SystemExit("[fail] missing selected barrier_len_fraction_t for step 7.13.8.5")
                barrier_len_fraction_use = float(triplet_barrier_len_fraction_best)
            else:
                barrier_len_fraction_use = 0.5
            base_fit_t = _fit_triplet_three_range_barrier_tail_pion_constrained_free_depth(
                b_mev=b_mev,
                targets=trip,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
                lambda_pi_fm=float(lambda_pi_pm_fm),
                tail_len_over_lambda=1.0,
                barrier_len_fraction=float(barrier_len_fraction_use),
                barrier_height_factor=float(k_use),
                tail_depth_factor=float(q_use),
            )
        elif step == "7.13.4":
            base_fit_t = _fit_triplet_three_range_tail_pion_constrained(
                b_mev=b_mev,
                targets=trip,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
                lambda_pi_fm=float(lambda_pi_pm_fm),
                tail_len_over_lambda=1.0,
            )
        elif step == "7.13.3":
            base_fit_t = _fit_triplet_two_range_pion_constrained(
                b_mev=b_mev,
                targets=trip,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
                lambda_pi_fm=float(lambda_pi_pm_fm),
            )
        else:
            base_fit_t = _fit_triplet_two_range(b_mev=b_mev, targets=trip, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        r1 = float(base_fit_t["r1_fm"])
        r2 = float(base_fit_t["r2_fm"])
        r3 = (
            float(base_fit_t["r3_fm"])
            if step
            in (
                "7.13.4",
                "7.13.5",
                "7.13.6",
                "7.13.7",
                "7.13.8",
                "7.13.8.1",
                "7.13.8.2",
                "7.13.8.3",
                "7.13.8.4",
                "7.13.8.5",
            )
            else r2
        )

        if step == "7.9.8":
            fit_t = _fit_triplet_repulsive_core_fixed_geometry(
                b_mev=b_mev,
                targets=trip,
                r1_fm=r1,
                r2_fm=r2,
                v2_hint_mev=float(base_fit_t["v2_mev"]),
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            rc = float(fit_t["rc_fm"])
            vc = float(fit_t["vc_mev"])
            v2 = float(fit_t["v2_mev"])
            v1 = float(fit_t["v1_mev"])
            a_t_exact = float(fit_t["a_exact_fm"])
            ere_t = fit_t["ere"]
            v3 = float(fit_t.get("v3_mev", 0.0)) if step == "7.13.4" else 0.0
        else:
            fit_t = base_fit_t
            rc = 0.0
            vc = 0.0
            v2 = float(fit_t["v2_mev"])
            v1 = float(fit_t["v1_mev"])
            a_t_exact = float(fit_t["a_exact_fm"])
            ere_t = fit_t["ere"]
            v3 = float(fit_t.get("v3_mev", 0.0)) if step == "7.13.4" else 0.0

        if step == "7.9.6":
            sing_fit: dict[str, object] = _fit_v2s_for_singlet(
                a_s_target_fm=float(sing["a_s_fm"]),
                r1_fm=r1,
                r2_fm=r2,
                v1_s_mev=v1,  # share V1 with triplet
                v2_hint_mev=v2,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            v1s = float(v1)  # shared
            v2s = float(sing_fit["V2_s_MeV"])
            fit_mode = "Shared geometry and shared V1; fit V2_s to match a_s (k->0)."
            note = "Singlet: uses shared geometry and V1; fits V2_s by a_s. (r_s, v2s) are predictions."
            fit_targets = ["a_s"]
            pred_targets = ["r_s", "v2s"]
            a_s_exact = _scattering_length_exact(
                r1_fm=r1, r2_fm=r2, v1_mev=v1s, v2_mev=v2s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
            )
            ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
            if ere_s is None:
                ere_s = _fit_kcot_ere(
                    r1_fm=r1, r2_fm=r2, v1_mev=v1s, v2_mev=v2s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
                )
            v3s = 0.0
        elif step == "7.9.7":
            sing_fit = _fit_v1v2_for_singlet_by_a_and_r(
                a_s_target_fm=float(sing["a_s_fm"]),
                r_s_target_fm=float(sing["r_s_fm"]),
                r1_fm=r1,
                r2_fm=r2,
                v1_hint_mev=v1,
                v2_hint_mev=v2,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            v1s = float(sing_fit["V1_s_MeV"])
            v2s = float(sing_fit["V2_s_MeV"])
            fit_mode = "Shared geometry; fit (V1_s,V2_s) to match (a_s,r_s)."
            note = "Singlet: uses shared geometry; fits (V1_s,V2_s) by (a_s,r_s). v2s is a prediction."
            fit_targets = ["a_s", "r_s"]
            pred_targets = ["v2s"]
            a_s_exact = _scattering_length_exact(
                r1_fm=r1, r2_fm=r2, v1_mev=v1s, v2_mev=v2s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
            )
            ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
            if ere_s is None:
                ere_s = _fit_kcot_ere(
                    r1_fm=r1, r2_fm=r2, v1_mev=v1s, v2_mev=v2s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
                )
            v3s = 0.0
        elif step == "7.13.3":
            sing_fit = _fit_v1v2_for_singlet_by_a_and_r(
                a_s_target_fm=float(sing["a_s_fm"]),
                r_s_target_fm=float(sing["r_s_fm"]),
                r1_fm=r1,
                r2_fm=r2,
                v1_hint_mev=v1,
                v2_hint_mev=v2,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
                allow_signed_v2=True,
            )
            v1s = float(sing_fit["V1_s_MeV"])
            v2s = float(sing_fit["V2_s_MeV"])
            fit_mode = "Shared geometry; fit (V1_s,V2_s) with signed V2_s to match (a_s,r_s)."
            note = "Singlet: uses shared geometry; fits (V1_s,V2_s) by (a_s,r_s) allowing outer barrier (V2_s<0). v2s is a prediction."
            fit_targets = ["a_s", "r_s"]
            pred_targets = ["v2s"]
            a_s_exact = _scattering_length_exact_signed_two_range(
                r1_fm=r1, r2_fm=r2, v1_mev=v1s, v2_mev=v2s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
            )
            ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
            if ere_s is None:
                ere_s = _fit_kcot_ere_signed_two_range(
                    r1_fm=r1, r2_fm=r2, v1_mev=v1s, v2_mev=v2s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
                )
            v3s = 0.0
        elif step == "7.13.8.2":
            if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
                raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.8.2")
            if barrier_height_factor_kq_s_best is None or tail_depth_factor_kq_s_best is None:
                raise SystemExit("[fail] missing selected (k_s,q_s) for step 7.13.8.2")
            cfg_any = _barrier_tail_config_free_depth(
                lambda_pi_fm=float(lambda_pi_pm_fm),
                tail_len_over_lambda=1.0,
                barrier_len_fraction=0.5,
                barrier_height_factor=float(barrier_height_factor_kq_s_best),
                tail_depth_factor=float(tail_depth_factor_kq_s_best),
            )
            if not isinstance(cfg_any, dict):
                raise SystemExit("[fail] invalid barrier_tail_config for step 7.13.8.2")
            cfg = {str(k): float(v) for k, v in cfg_any.items() if math.isfinite(float(v))}
            sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                a_s_target_fm=float(sing["a_s_fm"]),
                r_s_target_fm=float(sing["r_s_fm"]),
                r1_fm=r1,
                r2_fm=r2,
                cfg=cfg,
                v1_hint_mev=v1,
                v2_hint_mev=v2,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            v1s = float(sing_fit["V1_s_MeV"])
            v2s = float(sing_fit["V2_s_MeV"])
            fit_mode = "Shared geometry (R1,R2,R3=R2+λπ) + barrier+tail split; singlet uses its own (k_s,q_s) while triplet uses (k_t,q_t); fit (V1_s,V2_s>=0) to match (a_s,r_s)."
            note = "Singlet: uses shared (R1,R2) but allows channel-dependent barrier/tail scaling; fits (V1_s,V2_s) by (a_s,r_s). v2s is a prediction."
            fit_targets = ["a_s", "r_s"]
            pred_targets = ["v2s"]
            a_s_exact = float(sing_fit.get("a_s_exact_fm", float("nan")))
            ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
            if ere_s is None:
                for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                    if req not in cfg:
                        raise SystemExit(f"[fail] missing {req} in barrier_tail_config")
                vb = float(v2s) * float(cfg["barrier_coeff"])
                vt = float(v2s) * float(cfg["tail_depth_coeff"])
                segs_s = [
                    (float(r1), -float(v1s)),
                    (float(r2 - r1), -float(v2s)),
                    (float(cfg["Lb_fm"]), +float(vb)),
                    (float(cfg["Lt_fm"]), -float(vt)),
                ]
                r3_end = float(r2) + float(cfg["L3_total_fm"])
                ere_s = _fit_kcot_ere_segments(r_end_fm=r3_end, segments=segs_s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
            # Store explicit barrier/tail heights for plotting/output.
            if isinstance(ere_s, dict):
                try:
                    if "Vb_s_MeV" not in sing_fit or "Vt_s_MeV" not in sing_fit:
                        vb = float(v2s) * float(cfg["barrier_coeff"])
                        vt = float(v2s) * float(cfg["tail_depth_coeff"])
                        sing_fit["Vb_s_MeV"] = float(vb)
                        sing_fit["Vt_s_MeV"] = float(vt)
                except Exception:
                    pass
            v3s = 0.0
        elif step == "7.13.8.3":
            if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
                raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.8.3")
            if barrier_height_factor_kq_s_best is None or tail_depth_factor_kq_s_best is None:
                raise SystemExit("[fail] missing selected (k_s,q_s) for step 7.13.8.3")
            if singlet_r1_over_lambda_pi_best is None or not math.isfinite(singlet_r1_over_lambda_pi_best):
                raise SystemExit("[fail] missing selected R1_s/λπ for step 7.13.8.3")

            r1_s = float(singlet_r1_over_lambda_pi_best) * float(lambda_pi_pm_fm)
            r2_s = float(r2)
            if not (0.0 < r1_s < r2_s - 0.25):
                raise SystemExit("[fail] invalid singlet geometry for step 7.13.8.3 (R1_s >= R2-0.25)")

            cfg_any = _barrier_tail_config_free_depth(
                lambda_pi_fm=float(lambda_pi_pm_fm),
                tail_len_over_lambda=1.0,
                barrier_len_fraction=0.5,
                barrier_height_factor=float(barrier_height_factor_kq_s_best),
                tail_depth_factor=float(tail_depth_factor_kq_s_best),
            )
            if not isinstance(cfg_any, dict):
                raise SystemExit("[fail] invalid barrier_tail_config for step 7.13.8.3")
            cfg = {str(k): float(v) for k, v in cfg_any.items() if math.isfinite(float(v))}

            sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                a_s_target_fm=float(sing["a_s_fm"]),
                r_s_target_fm=float(sing["r_s_fm"]),
                r1_fm=r1_s,
                r2_fm=r2_s,
                cfg=cfg,
                v1_hint_mev=v1,
                v2_hint_mev=v2,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            v1s = float(sing_fit["V1_s_MeV"])
            v2s = float(sing_fit["V2_s_MeV"])
            fit_mode = "Channel-split barrier+tail (k_t,q_t)/(k_s,q_s) + singlet R1_s override; fit (V1_s,V2_s>=0) to match (a_s,r_s)."
            note = "Singlet: uses R1_s=(R1_s/λπ)·λπ (global) and R2_s from triplet; fits (V1_s,V2_s) by (a_s,r_s). v2s is a prediction."
            fit_targets = ["a_s", "r_s"]
            pred_targets = ["v2s"]
            a_s_exact = float(sing_fit.get("a_s_exact_fm", float("nan")))
            ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
            if ere_s is None:
                for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                    if req not in cfg:
                        raise SystemExit(f"[fail] missing {req} in barrier_tail_config")
                vb = float(v2s) * float(cfg["barrier_coeff"])
                vt = float(v2s) * float(cfg["tail_depth_coeff"])
                segs_s = [
                    (float(r1_s), -float(v1s)),
                    (float(r2_s - r1_s), -float(v2s)),
                    (float(cfg["Lb_fm"]), +float(vb)),
                    (float(cfg["Lt_fm"]), -float(vt)),
                ]
                r3_end = float(r2_s) + float(cfg["L3_total_fm"])
                ere_s = _fit_kcot_ere_segments(r_end_fm=r3_end, segments=segs_s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)

            # Store explicit barrier/tail heights and the singlet geometry for plotting/output.
            try:
                vb = float(v2s) * float(cfg["barrier_coeff"])
                vt = float(v2s) * float(cfg["tail_depth_coeff"])
                sing_fit["Vb_s_MeV"] = float(vb)
                sing_fit["Vt_s_MeV"] = float(vt)
                rb_s = float(r2_s) + float(cfg.get("Lb_fm", 0.0))
                r3_s = float(r2_s) + float(cfg.get("L3_total_fm", 0.0))
                sing_fit["geometry"] = {
                    "R1_fm": float(r1_s),
                    "R2_fm": float(r2_s),
                    "Rb_fm": float(rb_s),
                    "R3_fm": float(r3_s),
                    "R1_s_over_lambda_pi_pm": float(singlet_r1_over_lambda_pi_best),
                }
            except Exception:
                pass
            v3s = 0.0
        elif step in ("7.13.8.4", "7.13.8.5"):
            if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
                raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.8.4/7.13.8.5")
            if barrier_height_factor_kq_s_best is None or tail_depth_factor_kq_s_best is None:
                raise SystemExit("[fail] missing selected (k_s,q_s) for step 7.13.8.4/7.13.8.5")
            if singlet_r1_over_lambda_pi_best is None or not math.isfinite(singlet_r1_over_lambda_pi_best):
                raise SystemExit("[fail] missing fixed R1_s/λπ for step 7.13.8.4/7.13.8.5")
            if singlet_r2_over_lambda_pi_best is None or not math.isfinite(singlet_r2_over_lambda_pi_best):
                raise SystemExit("[fail] missing selected R2_s/λπ for step 7.13.8.4/7.13.8.5")

            r1_s = float(singlet_r1_over_lambda_pi_best) * float(lambda_pi_pm_fm)
            r2_s = float(singlet_r2_over_lambda_pi_best) * float(lambda_pi_pm_fm)
            if not (0.0 < r1_s < r2_s - 0.25):
                raise SystemExit("[fail] invalid singlet geometry for step 7.13.8.4 (R1_s >= R2_s-0.25)")

            cfg_any = _barrier_tail_config_free_depth(
                lambda_pi_fm=float(lambda_pi_pm_fm),
                tail_len_over_lambda=1.0,
                barrier_len_fraction=0.5,
                barrier_height_factor=float(barrier_height_factor_kq_s_best),
                tail_depth_factor=float(tail_depth_factor_kq_s_best),
            )
            if not isinstance(cfg_any, dict):
                raise SystemExit("[fail] invalid barrier_tail_config for step 7.13.8.4")
            cfg = {str(k): float(v) for k, v in cfg_any.items() if math.isfinite(float(v))}

            sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                a_s_target_fm=float(sing["a_s_fm"]),
                r_s_target_fm=float(sing["r_s_fm"]),
                r1_fm=r1_s,
                r2_fm=r2_s,
                cfg=cfg,
                v1_hint_mev=v1,
                v2_hint_mev=v2,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            v1s = float(sing_fit["V1_s_MeV"])
            v2s = float(sing_fit["V2_s_MeV"])
            fit_mode = "Channel-split barrier+tail (k_t,q_t)/(k_s,q_s) + singlet geometry split (R1_s,R2_s); fit (V1_s,V2_s>=0) to match (a_s,r_s)."
            note = "Singlet: uses R1_s=(R1_s/λπ)·λπ (frozen from Step 7.13.8.3 by r_s envelope) and a global R2_s=(R2_s/λπ)·λπ; fits (V1_s,V2_s) by (a_s,r_s). v2s is a prediction."
            fit_targets = ["a_s", "r_s"]
            pred_targets = ["v2s"]
            a_s_exact = float(sing_fit.get("a_s_exact_fm", float("nan")))
            ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
            if ere_s is None:
                for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                    if req not in cfg:
                        raise SystemExit(f"[fail] missing {req} in barrier_tail_config")
                vb = float(v2s) * float(cfg["barrier_coeff"])
                vt = float(v2s) * float(cfg["tail_depth_coeff"])
                segs_s = [
                    (float(r1_s), -float(v1s)),
                    (float(r2_s - r1_s), -float(v2s)),
                    (float(cfg["Lb_fm"]), +float(vb)),
                    (float(cfg["Lt_fm"]), -float(vt)),
                ]
                r3_end = float(r2_s) + float(cfg["L3_total_fm"])
                ere_s = _fit_kcot_ere_segments(r_end_fm=r3_end, segments=segs_s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)

            # Store explicit barrier/tail heights and the singlet geometry for plotting/output.
            try:
                vb = float(v2s) * float(cfg["barrier_coeff"])
                vt = float(v2s) * float(cfg["tail_depth_coeff"])
                sing_fit["Vb_s_MeV"] = float(vb)
                sing_fit["Vt_s_MeV"] = float(vt)
                rb_s = float(r2_s) + float(cfg.get("Lb_fm", 0.0))
                r3_s = float(r2_s) + float(cfg.get("L3_total_fm", 0.0))
                sing_fit["geometry"] = {
                    "R1_fm": float(r1_s),
                    "R2_fm": float(r2_s),
                    "Rb_fm": float(rb_s),
                    "R3_fm": float(r3_s),
                    "R1_s_over_lambda_pi_pm": float(singlet_r1_over_lambda_pi_best),
                    "R2_s_over_lambda_pi_pm": float(singlet_r2_over_lambda_pi_best),
                }
            except Exception:
                pass
            v3s = 0.0
        elif step in ("7.13.5", "7.13.6", "7.13.7", "7.13.8", "7.13.8.1"):
            if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
                raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.5/7.13.6/7.13.7/7.13.8")
            cfg_any = base_fit_t.get("barrier_tail_config", {})
            if not isinstance(cfg_any, dict):
                raise SystemExit("[fail] missing barrier_tail_config for step 7.13.5/7.13.6/7.13.7/7.13.8")
            cfg = {str(k): float(v) for k, v in cfg_any.items() if math.isfinite(float(v))}
            sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_barrier_tail(
                a_s_target_fm=float(sing["a_s_fm"]),
                r_s_target_fm=float(sing["r_s_fm"]),
                r1_fm=r1,
                r2_fm=r2,
                cfg=cfg,
                v1_hint_mev=v1,
                v2_hint_mev=v2,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            v1s = float(sing_fit["V1_s_MeV"])
            v2s = float(sing_fit["V2_s_MeV"])
            fit_mode = "Shared geometry (R1,R2,R3=R2+λπ) + barrier+tail split; fit (V1_s,V2_s>=0) to match (a_s,r_s)."
            note = "Singlet: uses shared (R1,R2) and a fixed barrier+tail split beyond R2; fits (V1_s,V2_s) by (a_s,r_s). v2s is a prediction."
            fit_targets = ["a_s", "r_s"]
            pred_targets = ["v2s"]
            a_s_exact = float(sing_fit.get("a_s_exact_fm", float("nan")))
            ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
            if ere_s is None:
                for req in ("L3_total_fm", "Lb_fm", "Lt_fm", "barrier_coeff", "tail_depth_coeff"):
                    if req not in cfg:
                        raise SystemExit(f"[fail] missing {req} in barrier_tail_config")
                vb = float(v2s) * float(cfg["barrier_coeff"])
                vt = float(v2s) * float(cfg["tail_depth_coeff"])
                segs_s = [
                    (float(r1), -float(v1s)),
                    (float(r2 - r1), -float(v2s)),
                    (float(cfg["Lb_fm"]), +float(vb)),
                    (float(cfg["Lt_fm"]), -float(vt)),
                ]
                r3_end = float(r2) + float(cfg["L3_total_fm"])
                ere_s = _fit_kcot_ere_segments(r_end_fm=r3_end, segments=segs_s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
            v3s = 0.0
        elif step == "7.13.4":
            if lambda_pi_pm_fm is None or not math.isfinite(lambda_pi_pm_fm):
                raise SystemExit("[fail] missing lambda_pi_pm_fm for step 7.13.4")
            tail_len_fm = float(lambda_pi_pm_fm)  # minimal operational choice: L3 = λπ
            sing_fit = _fit_v1v2_for_singlet_by_a_and_r_three_range_tail(
                a_s_target_fm=float(sing["a_s_fm"]),
                r_s_target_fm=float(sing["r_s_fm"]),
                r1_fm=r1,
                r2_fm=r2,
                tail_len_fm=tail_len_fm,
                v3_tail_mev=float(v3),
                v1_hint_mev=v1,
                v2_hint_mev=v2,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
                allow_signed_v2=True,
            )
            v1s = float(sing_fit["V1_s_MeV"])
            v2s = float(sing_fit["V2_s_MeV"])
            v3s = float(v3)
            fit_mode = "Shared geometry (R1,R2,R3=R2+λπ) + shared tail V3; fit (V1_s,V2_s) (signed V2 allowed) to match (a_s,r_s)."
            note = "Singlet: uses shared (R1,R2) from triplet and a fixed tail segment (V3,L3=λπ); fits (V1_s,V2_s) by (a_s,r_s). v2s is a prediction."
            fit_targets = ["a_s", "r_s"]
            pred_targets = ["v2s"]
            a_s_exact = float(sing_fit.get("a_s_exact_fm", float("nan")))
            ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
            if ere_s is None:
                segs_s = [
                    (float(r1), -float(v1s)),
                    (float(r2 - r1), -float(v2s)),
                    (float(tail_len_fm), -float(v3s)),
                ]
                ere_s = _fit_kcot_ere_segments(r_end_fm=float(r3), segments=segs_s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        else:
            sing_fit = _fit_v1v2_for_singlet_repulsive_core_two_range_by_a_and_r(
                a_s_target_fm=float(sing["a_s_fm"]),
                r_s_target_fm=float(sing["r_s_fm"]),
                rc_fm=rc,
                r1_fm=r1,
                r2_fm=r2,
                vc_mev=vc,
                v1_hint_mev=v1,
                v2_hint_mev=v2,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            v1s = float(sing_fit["V1_s_MeV"])
            v2s = float(sing_fit["V2_s_MeV"])
            fit_mode = "Shared geometry (including repulsive core); fit (V1_s,V2_s) to match (a_s,r_s)."
            note = "Singlet: uses shared geometry (Rc,Vc,R1,R2); fits (V1_s,V2_s) by (a_s,r_s). v2s is a prediction."
            fit_targets = ["a_s", "r_s"]
            pred_targets = ["v2s"]
            a_s_exact = _scattering_length_exact_repulsive_core_two_range(
                rc_fm=rc,
                r1_fm=r1,
                r2_fm=r2,
                vc_mev=vc,
                v1_mev=v1s,
                v2_mev=v2s,
                mu_mev=mu_mev,
                hbarc_mev_fm=hbarc_mev_fm,
            )
            ere_s = sing_fit.get("ere") if isinstance(sing_fit.get("ere"), dict) else None
            if ere_s is None:
                ere_s = _fit_kcot_ere_repulsive_core_two_range(
                    rc_fm=rc,
                    r1_fm=r1,
                    r2_fm=r2,
                    vc_mev=vc,
                    v1_mev=v1s,
                    v2_mev=v2s,
                    mu_mev=mu_mev,
                    hbarc_mev_fm=hbarc_mev_fm,
                )
            v3s = 0.0

        cmp_triplet = {
            "a_t_fit_minus_obs_fm": float(a_t_exact - float(trip["a_t_fm"])),
            "r_t_fit_minus_obs_fm": float(float(ere_t["r_eff_fm"]) - float(trip["r_t_fm"])),
            "v2t_fit_minus_obs_fm3": float(float(ere_t["v2_fm3"]) - float(trip["v2t_fm3"])),
        }
        if step == "7.9.6":
            cmp_singlet = {
                "a_s_fit_minus_obs_fm": float(a_s_exact - float(sing["a_s_fm"])),
                "r_s_pred_minus_obs_fm": float(float(ere_s["r_eff_fm"]) - float(sing["r_s_fm"])),
                "v2s_pred_minus_obs_fm3": float(float(ere_s["v2_fm3"]) - float(sing["v2s_fm3"])),
            }
        else:
            cmp_singlet = {
                "a_s_fit_minus_obs_fm": float(a_s_exact - float(sing["a_s_fm"])),
                "r_s_fit_minus_obs_fm": float(float(ere_s["r_eff_fm"]) - float(sing["r_s_fm"])),
                "v2s_pred_minus_obs_fm3": float(float(ere_s["v2_fm3"]) - float(sing["v2s_fm3"])),
            }

        results.append(
            {
                "label": label,
                "eq_label": int(d["eq_label"]),
                "inputs": {"B_MeV": float(b_mev), "triplet": trip, "singlet": sing},
                "fit_triplet": {
                    "geometry": (
                        (
                            {
                                "R1_fm": float(r1),
                                "R2_fm": float(r2),
                                "lambda_pi_pm_fm": float(lambda_pi_pm_fm),
                                "R1_over_lambda_pi_pm": float(r1 / float(lambda_pi_pm_fm)),
                                "R2_over_lambda_pi_pm": float(r2 / float(lambda_pi_pm_fm)),
                                "pion_constraint": base_fit_t.get("pion_constraint", {}),
                            }
                            if step == "7.13.3"
                            else (
                                {
                                    "R1_fm": float(r1),
                                    "R2_fm": float(r2),
                                    "R3_fm": float(r3),
                                    "L3_fm": float(r3 - r2),
                                    "lambda_pi_pm_fm": float(lambda_pi_pm_fm),
                                    "R1_over_lambda_pi_pm": float(r1 / float(lambda_pi_pm_fm)),
                                    "R2_over_lambda_pi_pm": float(r2 / float(lambda_pi_pm_fm)),
                                    "R3_over_lambda_pi_pm": float(r3 / float(lambda_pi_pm_fm)),
                                    "pion_constraint": base_fit_t.get("pion_constraint", {}),
                                    "tail_constraint": base_fit_t.get("tail_constraint", {}),
                                    "tail_factor": float(base_fit_t.get("tail_factor", float("nan"))),
                                }
                                if step == "7.13.4"
                                else (
                                    {
                                        "R1_fm": float(r1),
                                        "R2_fm": float(r2),
                                        "Rb_fm": float(base_fit_t.get("rb_fm", float("nan"))),
                                        "R3_fm": float(r3),
                                        "L3_total_fm": float(r3 - r2),
                                        "Lb_fm": float(base_fit_t.get("lb_fm", float("nan"))),
                                        "Lt_fm": float(base_fit_t.get("lt_fm", float("nan"))),
                                        "lambda_pi_pm_fm": float(lambda_pi_pm_fm),
                                        "R1_over_lambda_pi_pm": float(r1 / float(lambda_pi_pm_fm)),
                                        "R2_over_lambda_pi_pm": float(r2 / float(lambda_pi_pm_fm)),
                                        "Rb_over_lambda_pi_pm": float(float(base_fit_t.get("rb_fm", float("nan"))) / float(lambda_pi_pm_fm)),
                                        "R3_over_lambda_pi_pm": float(r3 / float(lambda_pi_pm_fm)),
                                        "pion_constraint": base_fit_t.get("pion_constraint", {}),
                                        "tail_constraint": base_fit_t.get("tail_constraint", {}),
                                        "tail_factor": float(base_fit_t.get("tail_factor", float("nan"))),
                                        "barrier_tail_config": base_fit_t.get("barrier_tail_config", {}),
                                    }
                                    if step
                                    in (
                                        "7.13.5",
                                        "7.13.6",
                                        "7.13.7",
                                        "7.13.8",
                                        "7.13.8.1",
                                        "7.13.8.2",
                                        "7.13.8.3",
                                        "7.13.8.4",
                                        "7.13.8.5",
                                    )
                                    else {"R1_fm": float(r1), "R2_fm": float(r2)}
                                )
                            )
                        )
                        if step != "7.9.8"
                        else {"Rc_fm": float(rc), "Vc_MeV": float(vc), "R1_fm": float(r1), "R2_fm": float(r2)}
                    ),
                    "V1_t_MeV": float(v1),
                    "V2_t_MeV": float(v2),
                    "V3_t_MeV": float(v3) if step == "7.13.4" else 0.0,
                    "V3_mean_t_MeV": (
                        float(fit_t.get("v3_mean_mev", float("nan")))
                        if step
                        in (
                            "7.13.5",
                            "7.13.6",
                            "7.13.7",
                            "7.13.8",
                            "7.13.8.1",
                            "7.13.8.2",
                            "7.13.8.3",
                            "7.13.8.4",
                            "7.13.8.5",
                        )
                        else 0.0
                    ),
                    "V3_barrier_t_MeV": (
                        float(fit_t.get("v3_barrier_mev", float("nan")))
                        if step
                        in (
                            "7.13.5",
                            "7.13.6",
                            "7.13.7",
                            "7.13.8",
                            "7.13.8.1",
                            "7.13.8.2",
                            "7.13.8.3",
                            "7.13.8.4",
                            "7.13.8.5",
                        )
                        else 0.0
                    ),
                    "V3_tail_t_MeV": (
                        float(fit_t.get("v3_tail_mev", float("nan")))
                        if step
                        in (
                            "7.13.5",
                            "7.13.6",
                            "7.13.7",
                            "7.13.8",
                            "7.13.8.1",
                            "7.13.8.2",
                            "7.13.8.3",
                            "7.13.8.4",
                            "7.13.8.5",
                        )
                        else 0.0
                    ),
                    "a_t_exact_fm": float(a_t_exact),
                    "ere": ere_t,
                    "fit_score": float(fit_t.get("score", base_fit_t.get("score", float("nan")))),
                    "fit_deltas": fit_t.get("deltas", base_fit_t.get("deltas", {})),
                    "note": (
                        (
                            "Triplet: λπ-constrained geometry; solve V1 by B, then search (R1,R2,V2) to match (a_t,r_t,v2t) on this ansatz class."
                            if step == "7.13.3"
                            else (
                                 "Triplet: λπ-constrained geometry + minimal tail (R3=R2+λπ, V3=<tail>*V2); solve V1 by B, then search (R1,R2,V2) to match (a_t,r_t,v2t)."
                                 if step == "7.13.4"
                                 else (
                                     "Triplet: λπ-constrained geometry + barrier+tail split beyond R2 (mean-preserving); solve V1 by B, then search (R1,R2,V2) to match (a_t,r_t,v2t)."
                                     if step == "7.13.5"
                                     else (
                                        "Triplet: λπ-constrained geometry + barrier+tail split beyond R2 (mean-preserving); select a single barrier-height factor k by cross-systematics, then fit per dataset."
                                        if step == "7.13.6"
                                        else (
                                            "Triplet: λπ-constrained geometry + barrier+tail split beyond R2 (free tail depth); select a single tail-depth factor q by cross-systematics, then fit per dataset."
                                            if step == "7.13.7"
                                            else (
                                                "Triplet: λπ-constrained geometry + barrier+tail split beyond R2 (free tail depth); select a single global (k,q) by cross-systematics, then fit per dataset."
                                                if step == "7.13.8"
                                                else (
                                                    "Triplet: λπ-constrained geometry + barrier+tail split beyond R2 (free tail depth); extended (k,q)-scan with dual-envelope criteria (triplet v2t + singlet v2s), then fit per dataset."
                                                    if step == "7.13.8.1"
                                                    else (
                                                        "Triplet: λπ-constrained geometry + barrier+tail split beyond R2 (free tail depth); channel-split (k_t,q_t)/(k_s,q_s) scan with envelope/tolerance criteria, then fit per dataset."
                                                        if step == "7.13.8.2"
                                                        else (
                                                            "Triplet: λπ-constrained geometry + barrier+tail split beyond R2 (free tail depth); reuse frozen (k_t,q_t)/(k_s,q_s) from Step 7.13.8.2 and scan singlet R1_s/λπ, then fit per dataset."
                                                            if step == "7.13.8.3"
                                                            else (
                                                                "Triplet: λπ-constrained geometry + barrier+tail split beyond R2 (free tail depth); reuse frozen (k_t,q_t)/(k_s,q_s) from Step 7.13.8.2, freeze R1_s from Step 7.13.8.3, then scan singlet R2_s/λπ and fit per dataset."
                                                                if step == "7.13.8.4"
                                                                else (
                                                                    "Triplet: λπ-constrained geometry + barrier+tail split beyond R2 (free tail depth); freeze singlet split from Step 7.13.8.4 and scan triplet barrier_len_fraction_t (and small k_t grid) to recover (a_t,r_t) within tolerance, then fit per dataset."
                                                                    if step == "7.13.8.5"
                                                                    else "Triplet: solve V1 by B, then search (R1,R2,V2) to match (a_t,r_t,v2t) on this ansatz class."
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        if step != "7.9.8"
                        else str(fit_t.get("note", "")) or "Triplet: fixed (R1,R2) and search (Rc,Vc,V2); V1 is solved by B."
                    ),
                },
                "fit_singlet": {
                    "V1_s_MeV": float(v1s),
                    "V2_s_MeV": float(v2s),
                    "V3_s_MeV": float(v3s) if step == "7.13.4" else 0.0,
                    "Vb_s_MeV": (
                        float(sing_fit.get("Vb_s_MeV", float("nan")))
                        if step
                        in (
                            "7.13.5",
                            "7.13.6",
                            "7.13.7",
                            "7.13.8",
                            "7.13.8.1",
                            "7.13.8.2",
                            "7.13.8.3",
                            "7.13.8.4",
                            "7.13.8.5",
                        )
                        else 0.0
                    ),
                    "Vt_s_MeV": (
                        float(sing_fit.get("Vt_s_MeV", float("nan")))
                        if step
                        in (
                            "7.13.5",
                            "7.13.6",
                            "7.13.7",
                            "7.13.8",
                            "7.13.8.1",
                            "7.13.8.2",
                            "7.13.8.3",
                            "7.13.8.4",
                            "7.13.8.5",
                        )
                        else 0.0
                    ),
                    "geometry": (sing_fit.get("geometry", {}) if step in ("7.13.8.3", "7.13.8.4", "7.13.8.5") else {}),
                    "a_s_exact_fm": float(a_s_exact),
                    "ere": ere_s,
                    "fit_mode": fit_mode,
                    "fit_targets": fit_targets,
                    "prediction_targets": pred_targets,
                    "fit_method": str(sing_fit.get("fit_method", sing_fit.get("method", ""))),
                    "fit_scan": (
                        sing_fit.get("scan", {})
                        if step == "7.9.6"
                        else {
                            "v1": sing_fit.get("scan_v1", {}),
                            "v2": (
                                sing_fit.get("fit_v2", {}).get("scan", {})
                                if isinstance(sing_fit.get("fit_v2"), dict)
                                else {}
                            ),
                        }
                    ),
                    "note_fit": str(sing_fit.get("note_fit", sing_fit.get("note", ""))),
                    "note_profile": str(sing_fit.get("note_profile", "")),
                    "note": note,
                },
                "comparison": {"triplet": cmp_triplet, "singlet": cmp_singlet},
            }
        )

    v2t_obs_list = [float(d["triplet"]["v2t_fm3"]) for d in datasets]
    rs_obs_list = [float(d["singlet"]["r_s_fm"]) for d in datasets]
    v2s_obs_list = [float(d["singlet"]["v2s_fm3"]) for d in datasets]
    v2t_env = {"min": float(min(v2t_obs_list)), "max": float(max(v2t_obs_list))}
    rs_env = {"min": float(min(rs_obs_list)), "max": float(max(rs_obs_list))}
    v2s_env = {"min": float(min(v2s_obs_list)), "max": float(max(v2s_obs_list))}

    v2t_fit = [float(r["fit_triplet"]["ere"]["v2_fm3"]) for r in results]
    v2t_within = all(math.isfinite(v) and v2t_env["min"] <= v <= v2t_env["max"] for v in v2t_fit)
    v2s_pred = [float(r["fit_singlet"]["ere"]["v2_fm3"]) for r in results]
    v2s_within = all(math.isfinite(v) and v2s_env["min"] <= v <= v2s_env["max"] for v in v2s_pred)
    rs_pred = [float(r["fit_singlet"]["ere"]["r_eff_fm"]) for r in results]
    rs_within = all(math.isfinite(v) and rs_env["min"] <= v <= rs_env["max"] for v in rs_pred)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.6, 8.6), dpi=160, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, wspace=0.25, hspace=0.25)

    for row, r in enumerate(results):
        label = str(r["label"])
        geo = r["fit_triplet"]["geometry"]
        if step == "7.9.8":
            rc = float(geo["Rc_fm"])
            vc = float(geo["Vc_MeV"])
        else:
            rc = 0.0
            vc = 0.0
        r1 = float(geo["R1_fm"])
        r2 = float(geo["R2_fm"])
        r3 = float(geo.get("R3_fm", r2))
        rb = (
            float(geo.get("Rb_fm", r2))
            if step
            in (
                "7.13.5",
                "7.13.6",
                "7.13.7",
                "7.13.8",
                "7.13.8.1",
                "7.13.8.2",
                "7.13.8.3",
                "7.13.8.4",
                "7.13.8.5",
            )
            else r2
        )
        v1t = float(r["fit_triplet"]["V1_t_MeV"])
        v2t = float(r["fit_triplet"]["V2_t_MeV"])
        v3t = float(r["fit_triplet"].get("V3_t_MeV", 0.0))
        vb_t = (
            float(r["fit_triplet"].get("V3_barrier_t_MeV", 0.0))
            if step
            in (
                "7.13.5",
                "7.13.6",
                "7.13.7",
                "7.13.8",
                "7.13.8.1",
                "7.13.8.2",
                "7.13.8.3",
                "7.13.8.4",
                "7.13.8.5",
            )
            else 0.0
        )
        vt_t = (
            float(r["fit_triplet"].get("V3_tail_t_MeV", 0.0))
            if step
            in (
                "7.13.5",
                "7.13.6",
                "7.13.7",
                "7.13.8",
                "7.13.8.1",
                "7.13.8.2",
                "7.13.8.3",
                "7.13.8.4",
                "7.13.8.5",
            )
            else 0.0
        )
        v1s = float(r["fit_singlet"]["V1_s_MeV"])
        v2s = float(r["fit_singlet"]["V2_s_MeV"])
        v3s = float(r["fit_singlet"].get("V3_s_MeV", 0.0))
        vb_s = (
            float(r["fit_singlet"].get("Vb_s_MeV", 0.0))
            if step
            in (
                "7.13.5",
                "7.13.6",
                "7.13.7",
                "7.13.8",
                "7.13.8.1",
                "7.13.8.2",
                "7.13.8.3",
                "7.13.8.4",
                "7.13.8.5",
            )
            else 0.0
        )
        vt_s = (
            float(r["fit_singlet"].get("Vt_s_MeV", 0.0))
            if step
            in (
                "7.13.5",
                "7.13.6",
                "7.13.7",
                "7.13.8",
                "7.13.8.1",
                "7.13.8.2",
                "7.13.8.3",
                "7.13.8.4",
                "7.13.8.5",
            )
            else 0.0
        )

        ax0 = fig.add_subplot(gs[row, 0])
        r_plot = [i * 0.02 for i in range(0, 501)]  # 0..10 fm

        sing_geo = r.get("fit_singlet", {}).get("geometry", {})
        if not isinstance(sing_geo, dict):
            sing_geo = {}
        r1_s = float(sing_geo.get("R1_fm", r1))
        r2_s = float(sing_geo.get("R2_fm", r2))
        rb_s_geo = float(sing_geo.get("Rb_fm", rb))
        r3_s_geo = float(sing_geo.get("R3_fm", r3))

        def v_profile(
            rr: float, *, vc_mev: float, rc_fm: float, r1_fm: float, r2_fm: float, r3_fm: float, v1: float, v2: float, v3: float
        ) -> float:
            if rr < rc_fm:
                return vc_mev
            if rr < r1_fm:
                return -v1
            if rr < r2_fm:
                return -v2
            if rr < r3_fm:
                return -v3
            return 0.0

        def v_profile_barrier_tail(
            rr: float,
            *,
            vc_mev: float,
            rc_fm: float,
            r1_fm: float,
            r2_fm: float,
            rb_fm: float,
            r3_fm: float,
            v1: float,
            v2: float,
            vb: float,
            vt: float,
        ) -> float:
            if rr < rc_fm:
                return vc_mev
            if rr < r1_fm:
                return -v1
            if rr < r2_fm:
                return -v2
            if rr < rb_fm:
                return +vb
            if rr < r3_fm:
                return -vt
            return 0.0

        if step in (
            "7.13.5",
            "7.13.6",
            "7.13.7",
            "7.13.8",
            "7.13.8.1",
            "7.13.8.2",
            "7.13.8.3",
            "7.13.8.4",
            "7.13.8.5",
        ):
            vt = [
                v_profile_barrier_tail(
                    rr, vc_mev=vc, rc_fm=rc, r1_fm=r1, r2_fm=r2, rb_fm=rb, r3_fm=r3, v1=v1t, v2=v2t, vb=vb_t, vt=vt_t
                )
                for rr in r_plot
            ]
            vs = [
                v_profile_barrier_tail(
                    rr,
                    vc_mev=vc,
                    rc_fm=rc,
                    r1_fm=r1_s,
                    r2_fm=r2_s,
                    rb_fm=rb_s_geo,
                    r3_fm=r3_s_geo,
                    v1=v1s,
                    v2=v2s,
                    vb=vb_s,
                    vt=vt_s,
                )
                for rr in r_plot
            ]
        else:
            vt = [v_profile(rr, vc_mev=vc, rc_fm=rc, r1_fm=r1, r2_fm=r2, r3_fm=r3, v1=v1t, v2=v2t, v3=v3t) for rr in r_plot]
            vs = [v_profile(rr, vc_mev=vc, rc_fm=rc, r1_fm=r1_s, r2_fm=r2_s, r3_fm=r3, v1=v1s, v2=v2s, v3=v3s) for rr in r_plot]
        ax0.plot(r_plot, vt, lw=2.0, color="tab:blue", label="triplet (fit B,a_t,r_t,v2t)")
        if step == "7.9.6":
            ax0.plot(r_plot, vs, lw=2.0, color="tab:orange", label="singlet (share V1; fit V2 by a_s)")
        elif step == "7.13.3":
            ax0.plot(r_plot, vs, lw=2.0, color="tab:orange", label="singlet (fit V1,V2(signed) by a_s,r_s)")
        elif step in (
            "7.13.5",
            "7.13.6",
            "7.13.7",
            "7.13.8",
            "7.13.8.1",
            "7.13.8.2",
            "7.13.8.3",
            "7.13.8.4",
            "7.13.8.5",
        ):
            ax0.plot(r_plot, vs, lw=2.0, color="tab:orange", label="singlet (fit V1,V2>=0 by a_s,r_s; barrier+tail)")
        elif step == "7.13.4":
            ax0.plot(r_plot, vs, lw=2.0, color="tab:orange", label="singlet (fit V1,V2(signed) by a_s,r_s; shared tail)")
        else:
            ax0.plot(r_plot, vs, lw=2.0, color="tab:orange", label="singlet (fit V1,V2 by a_s,r_s)")
        if step == "7.9.8" and rc > 0.0:
            ax0.axvline(rc, color="0.35", lw=1.0, ls="--")
        ax0.axvline(r1, color="0.35", lw=1.0, ls=":")
        if step in ("7.13.8.3", "7.13.8.4", "7.13.8.5") and math.isfinite(r1_s) and abs(r1_s - r1) > 1e-6:
            ax0.axvline(r1_s, color="tab:orange", lw=1.0, ls="--")
        ax0.axvline(r2, color="0.35", lw=1.0, ls=":")
        if step in ("7.13.8.4", "7.13.8.5") and math.isfinite(r2_s) and abs(r2_s - r2) > 1e-6:
            ax0.axvline(r2_s, color="tab:orange", lw=1.0, ls="--")
        if (
            step
            in (
                "7.13.5",
                "7.13.6",
                "7.13.7",
                "7.13.8",
                "7.13.8.1",
                "7.13.8.2",
                "7.13.8.3",
                "7.13.8.4",
                "7.13.8.5",
            )
            and rb > r2 + 1e-9
        ):
            ax0.axvline(rb, color="0.35", lw=1.0, ls=":")
        if step in ("7.13.8.4", "7.13.8.5") and math.isfinite(rb_s_geo) and abs(rb_s_geo - rb) > 1e-6:
            ax0.axvline(rb_s_geo, color="tab:orange", lw=1.0, ls="--")
        if step in ("7.13.8.4", "7.13.8.5") and math.isfinite(r3_s_geo) and abs(r3_s_geo - r3) > 1e-6:
            ax0.axvline(r3_s_geo, color="tab:orange", lw=1.0, ls="--")
        if (
            step
            in (
                "7.13.4",
                "7.13.5",
                "7.13.6",
                "7.13.7",
                "7.13.8",
                "7.13.8.1",
                "7.13.8.2",
                "7.13.8.3",
                "7.13.8.4",
                "7.13.8.5",
            )
            and r3 > r2 + 1e-9
        ):
            ax0.axvline(r3, color="0.35", lw=1.0, ls=":")
        if (
            step in ("7.13.3", "7.13.4", "7.13.5", "7.13.6", "7.13.7", "7.13.8", "7.13.8.1", "7.13.8.2", "7.13.8.3", "7.13.8.4", "7.13.8.5")
            and lambda_pi_pm_fm is not None
            and math.isfinite(lambda_pi_pm_fm)
        ):
            ax0.axvline(float(lambda_pi_pm_fm), color="tab:green", lw=1.1, ls="--")
        ax0.set_xlabel("r (fm)")
        ax0.set_ylabel("V(r) (MeV)")
        if step == "7.13.3":
            title = f"{label}: 2-range (λπ constrained; signed V2_s)"
        elif step in ("7.13.5", "7.13.6"):
            title = f"{label}: 3-range (λπ constrained; barrier+tail split)"
        elif step in ("7.13.7", "7.13.8", "7.13.8.1"):
            title = f"{label}: 3-range (λπ constrained; barrier+tail split; free tail depth)"
        elif step == "7.13.8.2":
            title = f"{label}: 3-range (λπ constrained; barrier+tail split; channel-split)"
        elif step == "7.13.8.3":
            title = f"{label}: 3-range (λπ constrained; barrier+tail split; channel-split + singlet R1_s scan)"
        elif step == "7.13.8.4":
            title = f"{label}: 3-range (λπ constrained; barrier+tail split; channel-split + singlet R2_s scan)"
        elif step == "7.13.8.5":
            title = f"{label}: 3-range (λπ constrained; barrier+tail split; triplet barrier-fraction scan)"
        elif step == "7.13.4":
            title = f"{label}: 3-range (λπ constrained; shared tail)"
        elif step == "7.9.8":
            title = f"{label}: repulsive core + two-range well"
        else:
            title = f"{label}: two-range well (shared geometry)"
        ax0.set_title(title)
        ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
        ax0.legend(frameon=True, fontsize=8, loc="lower right")
        ax0.text(
            0.02,
            0.98,
            (
                (
                    (
                        f"R1≈{r1:.3f} fm, R2≈{r2:.3f} fm\nV1t≈{v1t:.2f}, V2t≈{v2t:.2f} MeV"
                        if step != "7.9.8"
                        else f"Rc≈{rc:.3f} fm, Vc≈{vc:.1f} MeV\nR1≈{r1:.3f} fm, R2≈{r2:.3f} fm\nV1t≈{v1t:.2f}, V2t≈{v2t:.2f} MeV"
                    )
                    if step
                    not in ("7.13.3", "7.13.4", "7.13.5", "7.13.6", "7.13.7", "7.13.8", "7.13.8.1", "7.13.8.2", "7.13.8.3", "7.13.8.4", "7.13.8.5")
                    else (
                        (
                            f"R1≈{r1:.3f} fm, R2≈{r2:.3f} fm\n"
                            f"R1/λπ≈{(r1 / float(lambda_pi_pm_fm)):.3f}, R2/λπ≈{(r2 / float(lambda_pi_pm_fm)):.3f}\n"
                            f"V1t≈{v1t:.2f}, V2t≈{v2t:.2f} MeV\nV2s≈{v2s:.2f} MeV"
                        )
                        if step == "7.13.3"
                        else (
                            (
                                f"R1≈{r1:.3f} fm, R2≈{r2:.3f} fm, Rb≈{rb:.3f} fm, R3≈{r3:.3f} fm\n"
                                f"R1/λπ≈{(r1 / float(lambda_pi_pm_fm)):.3f}, R2/λπ≈{(r2 / float(lambda_pi_pm_fm)):.3f}, "
                                f"Rb/λπ≈{(rb / float(lambda_pi_pm_fm)):.3f}, R3/λπ≈{(r3 / float(lambda_pi_pm_fm)):.3f}\n"
                                f"V1t≈{v1t:.2f}, V2t≈{v2t:.2f}, Vb≈{vb_t:.2f}, Vt≈{vt_t:.2f} MeV\nV2s≈{v2s:.2f} MeV"
                            )
                            if step in ("7.13.5", "7.13.6", "7.13.7", "7.13.8", "7.13.8.1", "7.13.8.2", "7.13.8.3", "7.13.8.4", "7.13.8.5")
                            else (
                                f"R1≈{r1:.3f} fm, R2≈{r2:.3f} fm, R3≈{r3:.3f} fm\n"
                                f"R1/λπ≈{(r1 / float(lambda_pi_pm_fm)):.3f}, R2/λπ≈{(r2 / float(lambda_pi_pm_fm)):.3f}, R3/λπ≈{(r3 / float(lambda_pi_pm_fm)):.3f}\n"
                                f"V1t≈{v1t:.2f}, V2t≈{v2t:.2f}, V3t≈{v3t:.2f} MeV\nV2s≈{v2s:.2f} MeV"
                            )
                        )
                    )
                )
            ),
            transform=ax0.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
        )

        ax1 = fig.add_subplot(gs[row, 1])
        ere_t = r["fit_triplet"]["ere"]
        pts = ere_t["points"]
        xs = [(p["k_fm1"] ** 2) for p in pts]
        ys = [p["kcot_fm1"] for p in pts]
        ax1.plot(xs, ys, "o", ms=3.2, alpha=0.8, label="k-grid")
        coeffs = ere_t["coeffs"]
        c0 = float(coeffs["c0_fm1"])
        c2 = float(coeffs["c2_fm"])
        c4 = float(coeffs["c4_fm3"])
        x_line = [0.0, max(xs) if xs else 0.002**2]
        y_line = [c0 + c2 * x + c4 * (x * x) for x in x_line]
        ax1.plot(x_line, y_line, "-", lw=2.0, color="0.35", label="ERE fit")
        ax1.set_xlabel("k² (fm⁻²)")
        ax1.set_ylabel("k cot δ (fm⁻¹)")
        ax1.set_title("Triplet: ERE fit (target v2)")
        ax1.grid(True, ls=":", lw=0.6, alpha=0.6)
        ax1.legend(frameon=True, fontsize=9, loc="best")
        ax1.text(
            0.02,
            0.02,
            f"a≈{float(ere_t['a_fm']):.4f} fm\nr_t≈{float(ere_t['r_eff_fm']):.4f} fm\nv2≈{float(ere_t['v2_fm3']):.3f} fm³",
            transform=ax1.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
        )

        ax2 = fig.add_subplot(gs[row, 2])
        dt = r["comparison"]["triplet"]
        ds = r["comparison"]["singlet"]
        if step == "7.9.6":
            names = ["v2t(fit)", "r_s(pred)", "v2s(pred)"]
            deltas = [
                float(dt["v2t_fit_minus_obs_fm3"]),
                float(ds["r_s_pred_minus_obs_fm"]),
                float(ds["v2s_pred_minus_obs_fm3"]),
            ]
        else:
            names = ["v2t(fit)", "r_s(fit)", "v2s(pred)"]
            deltas = [
                float(dt["v2t_fit_minus_obs_fm3"]),
                float(ds["r_s_fit_minus_obs_fm"]),
                float(ds["v2s_pred_minus_obs_fm3"]),
            ]
        ax2.axhline(0.0, color="0.3", lw=1.0)
        ax2.bar(range(len(names)), deltas, color=["tab:blue", "tab:orange", "tab:orange"], alpha=0.85)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=10, ha="right")
        ax2.set_ylabel("fit/pred − obs (units: fm³, fm, fm³)")
        ax2.set_title("Cross-check vs observed (eq source)")
        ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
        ax2.text(
            0.02,
            0.98,
            "Triplet: v2 is targeted\nSinglet: a_s (+ r_s) are targeted; v2 is predicted",
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
        )

    suptitle_by_step = {
        "7.9.6": "Phase 7 / Step 7.9.6: two-range ansatz — fit triplet (B,a_t,r_t,v2t), predict singlet (r_s,v2s)",
        "7.9.7": "Phase 7 / Step 7.9.7: two-range ansatz — fit triplet (B,a_t,r_t,v2t), fit singlet (a_s,r_s), predict v2s",
        "7.9.8": "Phase 7 / Step 7.9.8: repulsive core + two-range — fit triplet (B,a_t,r_t,v2t), fit singlet (a_s,r_s), predict v2s",
        "7.13.3": "Phase 7 / Step 7.13.3: λπ-constrained 2-range — fit triplet (B,a_t,r_t,v2t), fit singlet (a_s,r_s) with signed V2, predict v2s",
        "7.13.4": "Phase 7 / Step 7.13.4: λπ-constrained 3-range (shared tail) — fit triplet (B,a_t,r_t,v2t), fit singlet (a_s,r_s) with signed V2, predict v2s",
        "7.13.5": "Phase 7 / Step 7.13.5: λπ-constrained 3-range (barrier+tail split; mean-preserving) — fit triplet (B,a_t,r_t,v2t), fit singlet (a_s,r_s) with V2>=0, predict v2s",
        "7.13.6": "Phase 7 / Step 7.13.6: λπ-constrained 3-range (barrier+tail split; mean-preserving) + k-scan — select k by cross-systematics, then fit triplet/singlet and predict v2s",
        "7.13.7": "Phase 7 / Step 7.13.7: λπ-constrained 3-range (barrier+tail split; free tail depth) + q-scan — select q by cross-systematics, then fit triplet/singlet and predict v2s",
        "7.13.8": "Phase 7 / Step 7.13.8: λπ-constrained 3-range (barrier+tail split; free tail depth) + (k,q)-scan — select (k,q) by cross-systematics, then fit triplet/singlet and predict v2s",
        "7.13.8.1": "Phase 7 / Step 7.13.8.1: λπ-constrained 3-range (barrier+tail split; free tail depth) + extended (k,q)-scan — require triplet v2t and singlet v2s within envelope",
        "7.13.8.2": "Phase 7 / Step 7.13.8.2: λπ-constrained 3-range (barrier+tail split; free tail depth) + channel-split (k,q) — scan (k_t,q_t,k_s,q_s) and fit triplet/singlet",
        "7.13.8.3": "Phase 7 / Step 7.13.8.3: λπ-constrained 3-range (barrier+tail split; free tail depth) + channel-split (k,q) + singlet R1_s/λπ scan — reuse 7.13.8.2 selection and scan R1_s/λπ",
        "7.13.8.4": "Phase 7 / Step 7.13.8.4: λπ-constrained 3-range (barrier+tail split; free tail depth) + channel-split (k,q) + singlet R2_s/λπ scan — freeze R1_s from 7.13.8.3 and scan R2_s/λπ",
        "7.13.8.5": "Phase 7 / Step 7.13.8.5: λπ-constrained 3-range (barrier+tail split; free tail depth) + channel-split (k,q) + triplet barrier split scan — freeze singlet split from 7.13.8.4 and scan triplet barrier_len_fraction_t",
    }
    fig.suptitle(suptitle_by_step[step], y=1.02)

    out_png_by_step = {
        "7.9.6": out_dir / "nuclear_effective_potential_two_range.png",
        "7.9.7": out_dir / "nuclear_effective_potential_two_range_fit_as_rs.png",
        "7.9.8": out_dir / "nuclear_effective_potential_repulsive_core_two_range.png",
        "7.13.3": out_dir / "nuclear_effective_potential_pion_constrained_signed_v2.png",
        "7.13.4": out_dir / "nuclear_effective_potential_pion_constrained_three_range_tail.png",
        "7.13.5": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail.png",
        "7.13.6": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_k_scan.png",
        "7.13.7": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_q_scan.png",
        "7.13.8": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan.png",
        "7.13.8.1": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_v2t.png",
        "7.13.8.2": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan.png",
        "7.13.8.3": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_singlet_r1_scan.png",
        "7.13.8.4": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_singlet_r2_scan.png",
        "7.13.8.5": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan.png",
    }
    out_png = out_png_by_step[step]
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_csv: Path | None = None
    if step == "7.13.3":
        out_csv = out_dir / "nuclear_effective_potential_pion_constrained_signed_v2.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "dataset",
                    "eq_label",
                    "lambda_pi_pm_fm",
                    "R1_fm",
                    "R2_fm",
                    "R1_over_lambda_pi_pm",
                    "R2_over_lambda_pi_pm",
                    "V1_t_MeV",
                    "V2_t_MeV",
                    "V1_s_MeV",
                    "V2_s_MeV",
                    "v2s_pred_fm3",
                    "v2s_obs_fm3",
                    "v2s_within_envelope",
                ]
            )
            for r in results:
                geo = r["fit_triplet"]["geometry"]
                r1 = float(geo["R1_fm"])
                r2 = float(geo["R2_fm"])
                lam = float(geo["lambda_pi_pm_fm"])
                v2s_pred_fm3 = float(r["fit_singlet"]["ere"]["v2_fm3"])
                v2s_obs_fm3 = float(r["inputs"]["singlet"]["v2s_fm3"])
                within = bool(v2s_env["min"] <= v2s_pred_fm3 <= v2s_env["max"]) if math.isfinite(v2s_pred_fm3) else False
                w.writerow(
                    [
                        str(r["label"]),
                        int(r["eq_label"]),
                        f"{lam:.12g}",
                        f"{r1:.12g}",
                        f"{r2:.12g}",
                        f"{(r1 / lam):.12g}",
                        f"{(r2 / lam):.12g}",
                        f"{float(r['fit_triplet']['V1_t_MeV']):.12g}",
                        f"{float(r['fit_triplet']['V2_t_MeV']):.12g}",
                        f"{float(r['fit_singlet']['V1_s_MeV']):.12g}",
                        f"{float(r['fit_singlet']['V2_s_MeV']):.12g}",
                        f"{v2s_pred_fm3:.12g}",
                        f"{v2s_obs_fm3:.12g}",
                        int(within),
                    ]
                )
    elif step == "7.13.4":
        out_csv = out_dir / "nuclear_effective_potential_pion_constrained_three_range_tail.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "dataset",
                    "eq_label",
                    "lambda_pi_pm_fm",
                    "R1_fm",
                    "R2_fm",
                    "R3_fm",
                    "L3_fm",
                    "tail_factor",
                    "R1_over_lambda_pi_pm",
                    "R2_over_lambda_pi_pm",
                    "R3_over_lambda_pi_pm",
                    "V1_t_MeV",
                    "V2_t_MeV",
                    "V3_t_MeV",
                    "V1_s_MeV",
                    "V2_s_MeV",
                    "V3_s_MeV",
                    "v2s_pred_fm3",
                    "v2s_obs_fm3",
                    "v2s_within_envelope",
                ]
            )
            for r in results:
                geo = r["fit_triplet"]["geometry"]
                r1 = float(geo["R1_fm"])
                r2 = float(geo["R2_fm"])
                r3 = float(geo["R3_fm"])
                l3 = float(geo["L3_fm"])
                lam = float(geo["lambda_pi_pm_fm"])
                tail_factor = float(geo.get("tail_factor", float("nan")))
                v2s_pred_fm3 = float(r["fit_singlet"]["ere"]["v2_fm3"])
                v2s_obs_fm3 = float(r["inputs"]["singlet"]["v2s_fm3"])
                within = bool(v2s_env["min"] <= v2s_pred_fm3 <= v2s_env["max"]) if math.isfinite(v2s_pred_fm3) else False
                w.writerow(
                    [
                        str(r["label"]),
                        int(r["eq_label"]),
                        f"{lam:.12g}",
                        f"{r1:.12g}",
                        f"{r2:.12g}",
                        f"{r3:.12g}",
                        f"{l3:.12g}",
                        f"{tail_factor:.12g}" if math.isfinite(tail_factor) else "",
                        f"{(r1 / lam):.12g}",
                        f"{(r2 / lam):.12g}",
                        f"{(r3 / lam):.12g}",
                        f"{float(r['fit_triplet']['V1_t_MeV']):.12g}",
                        f"{float(r['fit_triplet']['V2_t_MeV']):.12g}",
                        f"{float(r['fit_triplet'].get('V3_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_singlet']['V1_s_MeV']):.12g}",
                        f"{float(r['fit_singlet']['V2_s_MeV']):.12g}",
                        f"{float(r['fit_singlet'].get('V3_s_MeV', 0.0)):.12g}",
                        f"{v2s_pred_fm3:.12g}",
                        f"{v2s_obs_fm3:.12g}",
                        int(within),
                    ]
                )
    elif step == "7.13.8.2":
        if (
            barrier_height_factor_kq_t_best is None
            or tail_depth_factor_kq_t_best is None
            or barrier_height_factor_kq_s_best is None
            or tail_depth_factor_kq_s_best is None
        ):
            raise SystemExit("[fail] missing selected channel-split (k_t,q_t,k_s,q_s) for step 7.13.8.2")

        out_csv = out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "dataset",
                    "eq_label",
                    "lambda_pi_pm_fm",
                    "R1_fm",
                    "R2_fm",
                    "Rb_fm",
                    "R3_fm",
                    "L3_total_fm",
                    "Lb_fm",
                    "Lt_fm",
                    "tail_factor",
                    "barrier_len_fraction",
                    "k_t",
                    "q_t",
                    "k_s",
                    "q_s",
                    "V1_t_MeV",
                    "V2_t_MeV",
                    "V3_mean_t_MeV",
                    "V3_barrier_t_MeV",
                    "V3_tail_t_MeV",
                    "V1_s_MeV",
                    "V2_s_MeV",
                    "Vb_s_MeV",
                    "Vt_s_MeV",
                    "v2s_pred_fm3",
                    "v2s_obs_fm3",
                    "v2s_within_envelope",
                    "v2t_fit_fm3",
                    "v2t_obs_fm3",
                    "v2t_within_envelope",
                    "r_s_fit_fm",
                    "r_s_obs_fm",
                    "r_s_within_envelope",
                ]
            )
            for r in results:
                geo = r["fit_triplet"]["geometry"]
                r1 = float(geo["R1_fm"])
                r2 = float(geo["R2_fm"])
                rb = float(geo.get("Rb_fm", float("nan")))
                r3 = float(geo.get("R3_fm", float("nan")))
                l3_total = float(geo.get("L3_total_fm", float("nan")))
                lb = float(geo.get("Lb_fm", float("nan")))
                lt = float(geo.get("Lt_fm", float("nan")))
                lam = float(geo["lambda_pi_pm_fm"])
                tail_factor = float(geo.get("tail_factor", float("nan")))
                cfg = geo.get("barrier_tail_config", {})
                barrier_len_fraction = float(cfg.get("barrier_len_fraction", float("nan"))) if isinstance(cfg, dict) else float("nan")

                v2s_pred_fm3 = float(r["fit_singlet"]["ere"]["v2_fm3"])
                v2s_obs_fm3 = float(r["inputs"]["singlet"]["v2s_fm3"])
                within_v2s = bool(v2s_env["min"] <= v2s_pred_fm3 <= v2s_env["max"]) if math.isfinite(v2s_pred_fm3) else False

                v2t_fit_fm3 = float(r["fit_triplet"]["ere"]["v2_fm3"])
                v2t_obs_fm3 = float(r["inputs"]["triplet"]["v2t_fm3"])
                within_v2t = bool(v2t_env["min"] <= v2t_fit_fm3 <= v2t_env["max"]) if math.isfinite(v2t_fit_fm3) else False

                rs_fit_fm = float(r["fit_singlet"]["ere"]["r_eff_fm"])
                rs_obs_fm = float(r["inputs"]["singlet"]["r_s_fm"])
                within_rs = bool(rs_env["min"] <= rs_fit_fm <= rs_env["max"]) if math.isfinite(rs_fit_fm) else False

                w.writerow(
                    [
                        str(r["label"]),
                        int(r["eq_label"]),
                        f"{lam:.12g}",
                        f"{r1:.12g}",
                        f"{r2:.12g}",
                        f"{rb:.12g}" if math.isfinite(rb) else "",
                        f"{r3:.12g}" if math.isfinite(r3) else "",
                        f"{l3_total:.12g}" if math.isfinite(l3_total) else "",
                        f"{lb:.12g}" if math.isfinite(lb) else "",
                        f"{lt:.12g}" if math.isfinite(lt) else "",
                        f"{tail_factor:.12g}" if math.isfinite(tail_factor) else "",
                        f"{barrier_len_fraction:.12g}" if math.isfinite(barrier_len_fraction) else "",
                        f"{float(barrier_height_factor_kq_t_best):.12g}",
                        f"{float(tail_depth_factor_kq_t_best):.12g}",
                        f"{float(barrier_height_factor_kq_s_best):.12g}",
                        f"{float(tail_depth_factor_kq_s_best):.12g}",
                        f"{float(r['fit_triplet']['V1_t_MeV']):.12g}",
                        f"{float(r['fit_triplet']['V2_t_MeV']):.12g}",
                        f"{float(r['fit_triplet'].get('V3_mean_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_triplet'].get('V3_barrier_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_triplet'].get('V3_tail_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_singlet']['V1_s_MeV']):.12g}",
                        f"{float(r['fit_singlet']['V2_s_MeV']):.12g}",
                        f"{float(r['fit_singlet'].get('Vb_s_MeV', 0.0)):.12g}",
                        f"{float(r['fit_singlet'].get('Vt_s_MeV', 0.0)):.12g}",
                        f"{v2s_pred_fm3:.12g}",
                        f"{v2s_obs_fm3:.12g}",
                        int(within_v2s),
                        f"{v2t_fit_fm3:.12g}",
                        f"{v2t_obs_fm3:.12g}",
                        int(within_v2t),
                        f"{rs_fit_fm:.12g}",
                        f"{rs_obs_fm:.12g}",
                        int(within_rs),
                    ]
                )
    elif step == "7.13.8.3":
        if (
            barrier_height_factor_kq_t_best is None
            or tail_depth_factor_kq_t_best is None
            or barrier_height_factor_kq_s_best is None
            or tail_depth_factor_kq_s_best is None
            or singlet_r1_over_lambda_pi_best is None
        ):
            raise SystemExit("[fail] missing selected params for step 7.13.8.3 (k_t,q_t,k_s,q_s,R1_s/λπ)")

        out_csv = out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_singlet_r1_scan.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "dataset",
                    "eq_label",
                    "lambda_pi_pm_fm",
                    "R1_t_fm",
                    "R2_t_fm",
                    "Rb_fm",
                    "R3_fm",
                    "R1_s_fm",
                    "R1_s_over_lambda_pi_pm",
                    "L3_total_fm",
                    "Lb_fm",
                    "Lt_fm",
                    "tail_factor",
                    "barrier_len_fraction",
                    "k_t",
                    "q_t",
                    "k_s",
                    "q_s",
                    "V1_t_MeV",
                    "V2_t_MeV",
                    "V3_mean_t_MeV",
                    "V3_barrier_t_MeV",
                    "V3_tail_t_MeV",
                    "V1_s_MeV",
                    "V2_s_MeV",
                    "Vb_s_MeV",
                    "Vt_s_MeV",
                    "v2s_pred_fm3",
                    "v2s_obs_fm3",
                    "v2s_within_envelope",
                    "v2t_fit_fm3",
                    "v2t_obs_fm3",
                    "v2t_within_envelope",
                    "r_s_fit_fm",
                    "r_s_obs_fm",
                    "r_s_within_envelope",
                ]
            )
            for r in results:
                geo_t = r["fit_triplet"]["geometry"]
                geo_s = r.get("fit_singlet", {}).get("geometry", {})
                if not isinstance(geo_s, dict):
                    geo_s = {}

                r1_t = float(geo_t["R1_fm"])
                r2_t = float(geo_t["R2_fm"])
                rb = float(geo_t.get("Rb_fm", float("nan")))
                r3 = float(geo_t.get("R3_fm", float("nan")))
                l3_total = float(geo_t.get("L3_total_fm", float("nan")))
                lb = float(geo_t.get("Lb_fm", float("nan")))
                lt = float(geo_t.get("Lt_fm", float("nan")))
                lam = float(geo_t["lambda_pi_pm_fm"])
                tail_factor = float(geo_t.get("tail_factor", float("nan")))
                cfg = geo_t.get("barrier_tail_config", {})
                barrier_len_fraction = float(cfg.get("barrier_len_fraction", float("nan"))) if isinstance(cfg, dict) else float("nan")

                r1_s = float(geo_s.get("R1_fm", float("nan")))
                r1_s_over = float(geo_s.get("R1_s_over_lambda_pi_pm", float("nan")))

                v2s_pred_fm3 = float(r["fit_singlet"]["ere"]["v2_fm3"])
                v2s_obs_fm3 = float(r["inputs"]["singlet"]["v2s_fm3"])
                within_v2s = bool(v2s_env["min"] <= v2s_pred_fm3 <= v2s_env["max"]) if math.isfinite(v2s_pred_fm3) else False

                v2t_fit_fm3 = float(r["fit_triplet"]["ere"]["v2_fm3"])
                v2t_obs_fm3 = float(r["inputs"]["triplet"]["v2t_fm3"])
                within_v2t = bool(v2t_env["min"] <= v2t_fit_fm3 <= v2t_env["max"]) if math.isfinite(v2t_fit_fm3) else False

                rs_fit_fm = float(r["fit_singlet"]["ere"]["r_eff_fm"])
                rs_obs_fm = float(r["inputs"]["singlet"]["r_s_fm"])
                within_rs = bool(rs_env["min"] <= rs_fit_fm <= rs_env["max"]) if math.isfinite(rs_fit_fm) else False

                w.writerow(
                    [
                        str(r["label"]),
                        int(r["eq_label"]),
                        f"{lam:.12g}",
                        f"{r1_t:.12g}",
                        f"{r2_t:.12g}",
                        f"{rb:.12g}" if math.isfinite(rb) else "",
                        f"{r3:.12g}" if math.isfinite(r3) else "",
                        f"{r1_s:.12g}" if math.isfinite(r1_s) else "",
                        f"{r1_s_over:.12g}" if math.isfinite(r1_s_over) else "",
                        f"{l3_total:.12g}" if math.isfinite(l3_total) else "",
                        f"{lb:.12g}" if math.isfinite(lb) else "",
                        f"{lt:.12g}" if math.isfinite(lt) else "",
                        f"{tail_factor:.12g}" if math.isfinite(tail_factor) else "",
                        f"{barrier_len_fraction:.12g}" if math.isfinite(barrier_len_fraction) else "",
                        f"{float(barrier_height_factor_kq_t_best):.12g}",
                        f"{float(tail_depth_factor_kq_t_best):.12g}",
                        f"{float(barrier_height_factor_kq_s_best):.12g}",
                        f"{float(tail_depth_factor_kq_s_best):.12g}",
                        f"{float(r['fit_triplet']['V1_t_MeV']):.12g}",
                        f"{float(r['fit_triplet']['V2_t_MeV']):.12g}",
                        f"{float(r['fit_triplet'].get('V3_mean_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_triplet'].get('V3_barrier_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_triplet'].get('V3_tail_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_singlet']['V1_s_MeV']):.12g}",
                        f"{float(r['fit_singlet']['V2_s_MeV']):.12g}",
                        f"{float(r['fit_singlet'].get('Vb_s_MeV', 0.0)):.12g}",
                        f"{float(r['fit_singlet'].get('Vt_s_MeV', 0.0)):.12g}",
                        f"{v2s_pred_fm3:.12g}",
                        f"{v2s_obs_fm3:.12g}",
                        int(within_v2s),
                        f"{v2t_fit_fm3:.12g}",
                        f"{v2t_obs_fm3:.12g}",
                        int(within_v2t),
                        f"{rs_fit_fm:.12g}",
                        f"{rs_obs_fm:.12g}",
                        int(within_rs),
                    ]
                )
    elif step in ("7.13.8.4", "7.13.8.5"):
        if (
            barrier_height_factor_kq_t_best is None
            or tail_depth_factor_kq_t_best is None
            or barrier_height_factor_kq_s_best is None
            or tail_depth_factor_kq_s_best is None
            or singlet_r1_over_lambda_pi_best is None
            or singlet_r2_over_lambda_pi_best is None
        ):
            raise SystemExit("[fail] missing selected params for step 7.13.8.4/7.13.8.5 (k_t,q_t,k_s,q_s,R1_s/λπ,R2_s/λπ)")
        if step == "7.13.8.5" and (triplet_barrier_len_fraction_best is None or not math.isfinite(triplet_barrier_len_fraction_best)):
            raise SystemExit("[fail] missing selected barrier_len_fraction_t for step 7.13.8.5")

        out_csv = (
            out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_singlet_r2_scan.csv"
            if step == "7.13.8.4"
            else out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan.csv"
        )
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "dataset",
                    "eq_label",
                    "lambda_pi_pm_fm",
                    "R1_t_fm",
                    "R2_t_fm",
                    "Rb_t_fm",
                    "R3_t_fm",
                    "R1_s_fm",
                    "R2_s_fm",
                    "R1_s_over_lambda_pi_pm",
                    "R2_s_over_lambda_pi_pm",
                    "L3_total_fm",
                    "Lb_fm",
                    "Lt_fm",
                    "tail_factor",
                    "barrier_len_fraction",
                    "k_t",
                    "q_t",
                    "k_s",
                    "q_s",
                    "V1_t_MeV",
                    "V2_t_MeV",
                    "V3_mean_t_MeV",
                    "V3_barrier_t_MeV",
                    "V3_tail_t_MeV",
                    "V1_s_MeV",
                    "V2_s_MeV",
                    "Vb_s_MeV",
                    "Vt_s_MeV",
                    "v2s_pred_fm3",
                    "v2s_obs_fm3",
                    "v2s_within_envelope",
                    "v2t_fit_fm3",
                    "v2t_obs_fm3",
                    "v2t_within_envelope",
                    "r_s_fit_fm",
                    "r_s_obs_fm",
                    "r_s_within_envelope",
                ]
            )
            for r in results:
                geo_t = r["fit_triplet"]["geometry"]
                geo_s = r.get("fit_singlet", {}).get("geometry", {})
                if not isinstance(geo_s, dict):
                    geo_s = {}

                r1_t = float(geo_t["R1_fm"])
                r2_t = float(geo_t["R2_fm"])
                rb_t = float(geo_t.get("Rb_fm", float("nan")))
                r3_t = float(geo_t.get("R3_fm", float("nan")))
                l3_total = float(geo_t.get("L3_total_fm", float("nan")))
                lb = float(geo_t.get("Lb_fm", float("nan")))
                lt = float(geo_t.get("Lt_fm", float("nan")))
                lam = float(geo_t["lambda_pi_pm_fm"])
                tail_factor = float(geo_t.get("tail_factor", float("nan")))
                cfg = geo_t.get("barrier_tail_config", {})
                barrier_len_fraction = float(cfg.get("barrier_len_fraction", float("nan"))) if isinstance(cfg, dict) else float("nan")

                r1_s = float(geo_s.get("R1_fm", float("nan")))
                r2_s = float(geo_s.get("R2_fm", float("nan")))
                r1_s_over = float(geo_s.get("R1_s_over_lambda_pi_pm", float("nan")))
                r2_s_over = float(geo_s.get("R2_s_over_lambda_pi_pm", float("nan")))

                v2s_pred_fm3 = float(r["fit_singlet"]["ere"]["v2_fm3"])
                v2s_obs_fm3 = float(r["inputs"]["singlet"]["v2s_fm3"])
                within_v2s = bool(v2s_env["min"] <= v2s_pred_fm3 <= v2s_env["max"]) if math.isfinite(v2s_pred_fm3) else False

                v2t_fit_fm3 = float(r["fit_triplet"]["ere"]["v2_fm3"])
                v2t_obs_fm3 = float(r["inputs"]["triplet"]["v2t_fm3"])
                within_v2t = bool(v2t_env["min"] <= v2t_fit_fm3 <= v2t_env["max"]) if math.isfinite(v2t_fit_fm3) else False

                rs_fit_fm = float(r["fit_singlet"]["ere"]["r_eff_fm"])
                rs_obs_fm = float(r["inputs"]["singlet"]["r_s_fm"])
                within_rs = bool(rs_env["min"] <= rs_fit_fm <= rs_env["max"]) if math.isfinite(rs_fit_fm) else False

                w.writerow(
                    [
                        str(r["label"]),
                        int(r["eq_label"]),
                        f"{lam:.12g}",
                        f"{r1_t:.12g}",
                        f"{r2_t:.12g}",
                        f"{rb_t:.12g}" if math.isfinite(rb_t) else "",
                        f"{r3_t:.12g}" if math.isfinite(r3_t) else "",
                        f"{r1_s:.12g}" if math.isfinite(r1_s) else "",
                        f"{r2_s:.12g}" if math.isfinite(r2_s) else "",
                        f"{r1_s_over:.12g}" if math.isfinite(r1_s_over) else "",
                        f"{r2_s_over:.12g}" if math.isfinite(r2_s_over) else "",
                        f"{l3_total:.12g}" if math.isfinite(l3_total) else "",
                        f"{lb:.12g}" if math.isfinite(lb) else "",
                        f"{lt:.12g}" if math.isfinite(lt) else "",
                        f"{tail_factor:.12g}" if math.isfinite(tail_factor) else "",
                        f"{barrier_len_fraction:.12g}" if math.isfinite(barrier_len_fraction) else "",
                        f"{float(barrier_height_factor_kq_t_best):.12g}",
                        f"{float(tail_depth_factor_kq_t_best):.12g}",
                        f"{float(barrier_height_factor_kq_s_best):.12g}",
                        f"{float(tail_depth_factor_kq_s_best):.12g}",
                        f"{float(r['fit_triplet']['V1_t_MeV']):.12g}",
                        f"{float(r['fit_triplet']['V2_t_MeV']):.12g}",
                        f"{float(r['fit_triplet'].get('V3_mean_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_triplet'].get('V3_barrier_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_triplet'].get('V3_tail_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_singlet']['V1_s_MeV']):.12g}",
                        f"{float(r['fit_singlet']['V2_s_MeV']):.12g}",
                        f"{float(r['fit_singlet'].get('Vb_s_MeV', 0.0)):.12g}",
                        f"{float(r['fit_singlet'].get('Vt_s_MeV', 0.0)):.12g}",
                        f"{v2s_pred_fm3:.12g}",
                        f"{v2s_obs_fm3:.12g}",
                        int(within_v2s),
                        f"{v2t_fit_fm3:.12g}",
                        f"{v2t_obs_fm3:.12g}",
                        int(within_v2t),
                        f"{rs_fit_fm:.12g}",
                        f"{rs_obs_fm:.12g}",
                        int(within_rs),
                    ]
                )
    elif step in ("7.13.5", "7.13.6", "7.13.7", "7.13.8", "7.13.8.1"):
        out_csv_by_step = {
            "7.13.5": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail.csv",
            "7.13.6": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_k_scan.csv",
            "7.13.7": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_q_scan.csv",
            "7.13.8": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan.csv",
            "7.13.8.1": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_v2t.csv",
        }
        out_csv = out_csv_by_step[step]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "dataset",
                    "eq_label",
                    "lambda_pi_pm_fm",
                    "R1_fm",
                    "R2_fm",
                    "Rb_fm",
                    "R3_fm",
                    "L3_total_fm",
                    "Lb_fm",
                    "Lt_fm",
                    "tail_factor",
                    "barrier_len_fraction",
                    "barrier_height_factor",
                    "V1_t_MeV",
                    "V2_t_MeV",
                    "V3_mean_t_MeV",
                    "V3_barrier_t_MeV",
                    "V3_tail_t_MeV",
                    "V1_s_MeV",
                    "V2_s_MeV",
                    "Vb_s_MeV",
                    "Vt_s_MeV",
                    "v2s_pred_fm3",
                    "v2s_obs_fm3",
                    "v2s_within_envelope",
                    "v2t_fit_fm3",
                    "v2t_obs_fm3",
                    "v2t_within_envelope",
                ]
            )
            for r in results:
                geo = r["fit_triplet"]["geometry"]
                r1 = float(geo["R1_fm"])
                r2 = float(geo["R2_fm"])
                rb = float(geo.get("Rb_fm", float("nan")))
                r3 = float(geo.get("R3_fm", float("nan")))
                l3_total = float(geo.get("L3_total_fm", float("nan")))
                lb = float(geo.get("Lb_fm", float("nan")))
                lt = float(geo.get("Lt_fm", float("nan")))
                lam = float(geo["lambda_pi_pm_fm"])
                tail_factor = float(geo.get("tail_factor", float("nan")))
                cfg = geo.get("barrier_tail_config", {})
                barrier_len_fraction = float(cfg.get("barrier_len_fraction", float("nan"))) if isinstance(cfg, dict) else float("nan")
                barrier_height_factor = float(cfg.get("barrier_height_factor", float("nan"))) if isinstance(cfg, dict) else float("nan")
                v2s_pred_fm3 = float(r["fit_singlet"]["ere"]["v2_fm3"])
                v2s_obs_fm3 = float(r["inputs"]["singlet"]["v2s_fm3"])
                within = bool(v2s_env["min"] <= v2s_pred_fm3 <= v2s_env["max"]) if math.isfinite(v2s_pred_fm3) else False
                v2t_fit_fm3 = float(r["fit_triplet"]["ere"]["v2_fm3"])
                v2t_obs_fm3 = float(r["inputs"]["triplet"]["v2t_fm3"])
                within_v2t = bool(v2t_env["min"] <= v2t_fit_fm3 <= v2t_env["max"]) if math.isfinite(v2t_fit_fm3) else False
                w.writerow(
                    [
                        str(r["label"]),
                        int(r["eq_label"]),
                        f"{lam:.12g}",
                        f"{r1:.12g}",
                        f"{r2:.12g}",
                        f"{rb:.12g}" if math.isfinite(rb) else "",
                        f"{r3:.12g}" if math.isfinite(r3) else "",
                        f"{l3_total:.12g}" if math.isfinite(l3_total) else "",
                        f"{lb:.12g}" if math.isfinite(lb) else "",
                        f"{lt:.12g}" if math.isfinite(lt) else "",
                        f"{tail_factor:.12g}" if math.isfinite(tail_factor) else "",
                        f"{barrier_len_fraction:.12g}" if math.isfinite(barrier_len_fraction) else "",
                        f"{barrier_height_factor:.12g}" if math.isfinite(barrier_height_factor) else "",
                        f"{float(r['fit_triplet']['V1_t_MeV']):.12g}",
                        f"{float(r['fit_triplet']['V2_t_MeV']):.12g}",
                        f"{float(r['fit_triplet'].get('V3_mean_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_triplet'].get('V3_barrier_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_triplet'].get('V3_tail_t_MeV', 0.0)):.12g}",
                        f"{float(r['fit_singlet']['V1_s_MeV']):.12g}",
                        f"{float(r['fit_singlet']['V2_s_MeV']):.12g}",
                        f"{float(r['fit_singlet'].get('Vb_s_MeV', 0.0)):.12g}",
                        f"{float(r['fit_singlet'].get('Vt_s_MeV', 0.0)):.12g}",
                        f"{v2s_pred_fm3:.12g}",
                        f"{v2s_obs_fm3:.12g}",
                        int(within),
                        f"{v2t_fit_fm3:.12g}",
                        f"{v2t_obs_fm3:.12g}",
                        int(within_v2t),
                    ]
                )

    codata_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    codata_manifest = codata_dir / "manifest.json"
    codata_extracted = codata_dir / "extracted_values.json"
    np_manifest = root / "data" / "quantum" / "sources" / "np_scattering_low_energy_arxiv_0704_1024v1_manifest.json"
    np_extracted = root / "data" / "quantum" / "sources" / "np_scattering_low_energy_arxiv_0704_1024v1_extracted.json"

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": step,
        "model": {
            "effective_equation": "Nonrelativistic s-wave Schrödinger equation for relative motion with an effective potential V(r)=μ φ(r)=-μ c^2 u(r).",
            "ansatz": (
                "λπ-constrained two-range: V=-V1 (r<R1), V=-V2 (R1<r<R2), 0 (r>R2), with signed V2_s allowed for singlet."
                if step == "7.13.3"
                else (
                    (
                        "λπ-constrained three-range (shared tail): V=-V1 (r<R1), V=-V2 (R1<r<R2), V=-V3 (R2<r<R3), 0 (r>R3), "
                        "with R3=R2+λπ and V3=<exp tail mean>*V2. Signed V2_s is allowed for singlet."
                        if step == "7.13.4"
                        else (
                            "λπ-constrained three-range (barrier+tail split; mean-preserving): "
                            "V=-V1 (r<R1), V=-V2 (R1<r<R2), V=+Vb (R2<r<Rb), V=-Vt (Rb<r<R3), 0 (r>R3), "
                            "with R3=R2+λπ, Rb=R2+f·λπ, V3_mean=<exp tail mean>*V2, Vb=k·V3_mean, and Vt chosen so that "
                            "(Lb*(+Vb)+Lt*(-Vt))/L3 = -V3_mean. Singlet restricts V2_s>=0."
                            if step == "7.13.5"
                            else (
                                (
                                    "λπ-constrained three-range (barrier+tail split; mean-preserving) with k-scan: "
                                    "same ansatz as step 7.13.5 but with a single global barrier-height factor k selected by cross-systematics (eq18–eq19 envelope). "
                                    "Singlet restricts V2_s>=0."
                                )
                                if step == "7.13.6"
                                else (
                                    (
                                        "λπ-constrained three-range (barrier+tail split; free tail depth) with q-scan: "
                                        "same geometry and barrier as step 7.13.5, but with tail depth Vt=q·V3_mean (no mean-preserving constraint). "
                                        "Singlet restricts V2_s>=0."
                                    )
                                    if step == "7.13.7"
                                    else (
                                        (
                                            "λπ-constrained three-range (barrier+tail split; free tail depth) with (k,q)-scan: "
                                            "same geometry as step 7.13.5, but allow both barrier height k and tail depth Vt=q·V3_mean (no mean-preserving constraint), "
                                            "selected jointly by cross-systematics (eq18–eq19 envelope). Singlet restricts V2_s>=0."
                                        )
                                        if step == "7.13.8"
                                        else (
                                            (
                                                "λπ-constrained three-range (barrier+tail split; free tail depth) with extended (k,q)-scan: "
                                                "same ansatz as step 7.13.8 but select (k,q) with a dual-envelope criterion (triplet v2t + singlet v2s) before fitting per dataset."
                                            )
                                            if step == "7.13.8.1"
                                            else (
                                                (
                                                    "λπ-constrained three-range (barrier+tail split; channel-split (k,q)): "
                                                    "same geometry as step 7.13.8 but allow separate (k_t,q_t) for triplet and (k_s,q_s) for singlet, "
                                                    "selected jointly by envelope/tolerance criteria before fitting per dataset. Singlet restricts V2_s>=0."
                                                )
                                                if step == "7.13.8.2"
                                                else (
                                                    (
                                                        "λπ-constrained three-range (barrier+tail split; channel-split (k,q) + singlet R1_s/λπ scan): "
                                                        "reuse the frozen (k_t,q_t)/(k_s,q_s) from step 7.13.8.2 and scan a single global singlet R1_s/λπ; "
                                                        "R2_s follows the triplet-fitted R2 per dataset. Singlet restricts V2_s>=0."
                                                    )
                                                    if step == "7.13.8.3"
                                                    else (
                                                        "Two-range (two-step) attractive well: V=-V1 (r<R1), V=-V2 (R1<r<R2), 0 (r>R2)."
                                                        if step != "7.9.8"
                                                        else "Repulsive core + two-range: V=+Vc (r<Rc), V=-V1 (Rc<r<R1), V=-V2 (R1<r<R2), 0 (r>R2)."
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "fit_constraints": {
                "triplet": ["B (CODATA mass defect; solves V1)", "a_t (np low-energy)", "r_t (np low-energy)", "v2t (shape parameter; target)"],
                "singlet": (
                    ["a_s (np low-energy; fits V2_s with shared geometry and shared V1)"]
                    if step == "7.9.6"
                    else (
                        (
                            ["a_s (np low-energy; fits V1_s,V2_s; allow signed V2_s)", "r_s (np low-energy; fits V1_s,V2_s)"]
                            if step == "7.13.3"
                            else (
                                [
                                    "a_s (np low-energy; fits V1_s,V2_s; allow signed V2_s; shared tail fixed from triplet)",
                                    "r_s (np low-energy; fits V1_s,V2_s; shared tail fixed from triplet)",
                                ]
                                if step == "7.13.4"
                                else (
                                     [
                                         "a_s (np low-energy; fits V1_s,V2_s>=0; barrier+tail split fixed by λπ)",
                                         "r_s (np low-energy; fits V1_s,V2_s>=0; barrier+tail split fixed by λπ)",
                                     ]
                                    if step in ("7.13.5", "7.13.6", "7.13.7", "7.13.8", "7.13.8.1", "7.13.8.2", "7.13.8.3", "7.13.8.4")
                                    else ["a_s (np low-energy; fits V1_s,V2_s)", "r_s (np low-energy; fits V1_s,V2_s)"]
                                )
                            )
                        )
                    )
                ),
            },
            "prediction_targets": {"singlet": (["r_s", "v2s"] if step == "7.9.6" else ["v2s"])},
            "positioning": [
                "Phenomenological constraint (effective model), not a first-principles derivation of nuclear forces.",
                "The ansatz class is intentionally low-parameter to avoid 'arbitrary function fit'.",
                (
                    "To limit arbitrariness between channels, singlet uses shared geometry and shares V1; only V2_s is adjusted to match a_s."
                    if step == "7.9.6"
                    else (
                        "To limit arbitrariness between channels, singlet uses shared geometry; (V1_s,V2_s) are the minimal extra freedom to match (a_s,r_s)."
                        if step == "7.9.7"
                        else (
                            "To limit arbitrariness, constrain geometry by λπ (hadron-scale range) and allow only a signed V2_s (outer barrier) as the minimal sign-structure; v2s is the differential prediction."
                            if step == "7.13.3"
                            else (
                                "To limit arbitrariness, constrain geometry by λπ and add only a minimal Yukawa-like tail (R3=R2+λπ, V3=<tail>*V2) with no new free parameters; singlet fits (V1_s,V2_s) by (a_s,r_s) and v2s is the differential prediction."
                                if step == "7.13.4"
                                else (
                                    "To limit arbitrariness, constrain geometry by λπ and split the tail into barrier+tail via a fixed mean-preserving rule (no independent tail-strength parameter); singlet fits (V1_s,V2_s>=0) by (a_s,r_s) and v2s is the differential prediction."
                                    if step == "7.13.5"
                                    else (
                                        "To limit arbitrariness, constrain geometry by λπ and split the tail into barrier+tail via a fixed mean-preserving rule; then select a single global k by cross-systematics (eq18–eq19 envelope) before fitting per dataset."
                                        if step == "7.13.6"
                                        else (
                                            "To limit arbitrariness, constrain geometry by λπ and split the tail into barrier+tail, but allow a single global tail-depth factor q (no mean-preserving) selected by cross-systematics before fitting per dataset."
                                            if step == "7.13.7"
                                            else (
                                                "To limit arbitrariness, constrain geometry by λπ and split the tail into barrier+tail, then select a single global (k,q) (no mean-preserving) by cross-systematics before fitting per dataset."
                                                if step == "7.13.8"
                                                else (
                                                    "To limit arbitrariness, constrain geometry by λπ and split the tail into barrier+tail, then select (k,q) with a dual-envelope criterion (triplet v2t + singlet v2s) before fitting per dataset."
                                                    if step == "7.13.8.1"
                                                    else (
                                                        "To limit arbitrariness, constrain geometry by λπ and split the tail into barrier+tail, but allow channel-dependent (k_t,q_t)/(k_s,q_s); select them jointly by envelope/tolerance criteria before fitting per dataset."
                                                        if step == "7.13.8.2"
                                                        else (
                                                            "To limit arbitrariness, reuse the frozen channel-split (k_t,q_t)/(k_s,q_s) from Step 7.13.8.2 and scan only a single additional DOF (global R1_s/λπ) for the singlet before fitting per dataset."
                                                            if step == "7.13.8.3"
                                                            else "To limit arbitrariness, keep (R1,R2) fixed from step 7.9.6 and add only (Rc,Vc) as sign-structure; singlet fits (V1_s,V2_s) by (a_s,r_s) and v2s is the differential prediction."
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
            ],
        },
        "sources": [
            {
                "dataset": "NIST Cuu CODATA constants (mp,mn,md) for deuteron binding baseline",
                "local_manifest": str(codata_manifest),
                "local_manifest_sha256": _sha256(codata_manifest),
                "local_extracted": str(codata_extracted),
                "local_extracted_sha256": _sha256(codata_extracted),
            },
            {
                "dataset": "np scattering low-energy parameters (arXiv:0704.1024v1; eq18–eq19)",
                "local_manifest": str(np_manifest),
                "local_manifest_sha256": _sha256(np_manifest),
                "local_extracted": str(np_extracted),
                "local_extracted_sha256": _sha256(np_extracted),
            },
        ],
        "constants": {"hbarc_MeV_fm": float(hbarc_mev_fm), "mu_c2_MeV": float(mu_mev), "B_MeV": float(b_mev)},
        "results_by_dataset": results,
        "systematics_proxy": {
            "observed_envelope_v2t_fm3": v2t_env,
            "observed_envelope_r_s_fm": rs_env,
            "observed_envelope_v2s_fm3": v2s_env,
            "predicted_r_s_within_envelope": bool(rs_within),
            "predicted_v2s_within_envelope": bool(v2s_within),
            "notes": [
                "Eq.(18) and Eq.(19) are two phase-shift analyses in the same primary source; their difference is treated as an analysis-dependent systematics proxy.",
                "Triplet fit is performed per dataset to expose sensitivity of the inferred u-profile to that proxy.",
                (
                    "In step 7.9.6, singlet (r_s,v2s) are predictions after matching only a_s."
                    if step == "7.9.6"
                    else "In step 7.9.7/7.9.8/7.13.3/7.13.4/7.13.5/7.13.6/7.13.7/7.13.8/7.13.8.1/7.13.8.2/7.13.8.3/7.13.8.4/7.13.8.5, singlet r_s is fitted per dataset; v2s is the remaining differential prediction."
                ),
            ],
        },
        "falsification": {
            "acceptance_criteria": [
                (
                    "Under the two-range ansatz class, after fitting triplet targets and fitting singlet a_s by adjusting only V2_s (shared geometry and shared V1), the predicted singlet (r_s,v2s) should be compatible with the observed range (eq18–eq19)."
                    if step == "7.9.6"
                    else (
                        "Under the two-range ansatz class, after fitting triplet targets and fitting singlet (a_s,r_s) by adjusting (V1_s,V2_s) (shared geometry), the predicted singlet v2s should be compatible with the observed range (eq18–eq19)."
                        if step == "7.9.7"
                        else (
                            "Under the λπ-constrained two-range ansatz class (allow signed V2_s), after fitting triplet targets and fitting singlet (a_s,r_s), the predicted singlet v2s should be compatible with the observed range (eq18–eq19)."
                            if step == "7.13.3"
                            else (
                                "Under the λπ-constrained three-range (shared tail) ansatz class, after fitting triplet targets and fitting singlet (a_s,r_s) allowing signed V2_s, the predicted singlet v2s should be compatible with the observed range (eq18–eq19)."
                                if step == "7.13.4"
                                else (
                                    "Under the λπ-constrained three-range (barrier+tail split; mean-preserving) ansatz class, after fitting triplet targets and fitting singlet (a_s,r_s) with V2_s>=0, the predicted singlet v2s should be compatible with the observed range (eq18–eq19)."
                                    if step == "7.13.5"
                                    else (
                                        "Under the λπ-constrained three-range (barrier+tail split; mean-preserving) ansatz class with a single global k selected by cross-systematics, after fitting triplet targets and fitting singlet (a_s,r_s) with V2_s>=0, the predicted singlet v2s should be compatible with the observed range (eq18–eq19)."
                                        if step == "7.13.6"
                                        else (
                                            "Under the λπ-constrained three-range (barrier+tail split; free tail depth) ansatz class with a single global q selected by cross-systematics, after fitting triplet targets and fitting singlet (a_s,r_s) with V2_s>=0, the predicted singlet v2s should be compatible with the observed range (eq18–eq19)."
                                            if step == "7.13.7"
                                            else (
                                                "Under the λπ-constrained three-range (barrier+tail split; free tail depth) ansatz class with a single global (k,q) selected by cross-systematics, after fitting triplet targets and fitting singlet (a_s,r_s) with V2_s>=0, the predicted singlet v2s should be compatible with the observed range (eq18–eq19)."
                                                if step == "7.13.8"
                                                else (
                                                    "Under the λπ-constrained three-range (barrier+tail split; free tail depth) ansatz class with a single global (k,q) selected by cross-systematics, after fitting triplet targets and fitting singlet (a_s,r_s) with V2_s>=0, both the resulting triplet v2t and the predicted singlet v2s should be compatible with the observed ranges (eq18–eq19)."
                                                    if step == "7.13.8.1"
                                                    else (
                                                        "Under the λπ-constrained three-range (barrier+tail split; free tail depth) ansatz class with channel-split (k_t,q_t)/(k_s,q_s) selected by cross-systematics, after fitting triplet targets and fitting singlet (a_s,r_s) with V2_s>=0, both triplet v2t and singlet (r_s,v2s) should be compatible with the observed envelopes (eq18–eq19)."
                                                        if step == "7.13.8.2"
                                                        else (
                                                            "Under the λπ-constrained three-range (barrier+tail split; free tail depth) ansatz class, reuse the frozen channel-split (k_t,q_t)/(k_s,q_s) from Step 7.13.8.2 and scan a single additional DOF (global R1_s/λπ) for the singlet; the resulting triplet v2t and singlet (r_s,v2s) should be compatible with the observed envelopes (eq18–eq19)."
                                                            if step == "7.13.8.3"
                                                            else "Under the repulsive-core + two-range ansatz class, after fitting triplet targets and fitting singlet (a_s,r_s), the predicted singlet v2s should be compatible with the observed range (eq18–eq19)."
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                (
                    "If singlet (r_s,v2s) falls outside the analysis-dependent envelope, the ansatz class is rejected as a representation of the nuclear-scale u-profile."
                    if step == "7.9.6"
                    else (
                        "If any of (triplet v2t, singlet r_s, singlet v2s) falls outside the analysis-dependent envelope, the ansatz class is rejected as a representation of the nuclear-scale u-profile."
                        if step in ("7.13.8.1", "7.13.8.2", "7.13.8.3", "7.13.8.4")
                        else "If predicted singlet v2s falls outside the analysis-dependent envelope, the ansatz class is rejected as a representation of the nuclear-scale u-profile."
                    )
                ),
            ],
            "within_envelope": (
                {"r_s": bool(rs_within), "v2s": bool(v2s_within), "v2t": bool(v2t_within)}
                if step in ("7.13.8.1", "7.13.8.2", "7.13.8.3", "7.13.8.4")
                else {"r_s": bool(rs_within), "v2s": bool(v2s_within)}
            ),
        },
        "outputs": {"png": str(out_png)},
    }

    if out_csv is not None:
        metrics["outputs"]["csv"] = str(out_csv)
    if pion_scale is not None:
        metrics["pion_range_scale"] = pion_scale
    if barrier_height_factor_scan is not None:
        metrics["barrier_height_factor_scan"] = barrier_height_factor_scan
    if tail_depth_factor_scan is not None:
        metrics["tail_depth_factor_scan"] = tail_depth_factor_scan
    if barrier_tail_kq_scan is not None:
        metrics["barrier_tail_kq_scan"] = barrier_tail_kq_scan
    if barrier_tail_channel_split_kq_scan is not None:
        metrics["barrier_tail_channel_split_kq_scan"] = barrier_tail_channel_split_kq_scan
    if barrier_tail_channel_split_kq_singlet_r1_scan is not None:
        metrics["barrier_tail_channel_split_kq_singlet_r1_scan"] = barrier_tail_channel_split_kq_singlet_r1_scan
    if barrier_tail_channel_split_kq_singlet_r2_scan is not None:
        metrics["barrier_tail_channel_split_kq_singlet_r2_scan"] = barrier_tail_channel_split_kq_singlet_r2_scan
    if barrier_tail_channel_split_kq_triplet_barrier_fraction_scan is not None:
        metrics["barrier_tail_channel_split_kq_triplet_barrier_fraction_scan"] = (
            barrier_tail_channel_split_kq_triplet_barrier_fraction_scan
        )

    out_json_by_step = {
        "7.9.6": out_dir / "nuclear_effective_potential_two_range_metrics.json",
        "7.9.7": out_dir / "nuclear_effective_potential_two_range_fit_as_rs_metrics.json",
        "7.9.8": out_dir / "nuclear_effective_potential_repulsive_core_two_range_metrics.json",
        "7.13.3": out_dir / "nuclear_effective_potential_pion_constrained_signed_v2_metrics.json",
        "7.13.4": out_dir / "nuclear_effective_potential_pion_constrained_three_range_tail_metrics.json",
        "7.13.5": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_metrics.json",
        "7.13.6": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_k_scan_metrics.json",
        "7.13.7": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_q_scan_metrics.json",
        "7.13.8": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_metrics.json",
        "7.13.8.1": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_v2t_metrics.json",
        "7.13.8.2": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_metrics.json",
        "7.13.8.3": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_singlet_r1_scan_metrics.json",
        "7.13.8.4": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_singlet_r2_scan_metrics.json",
        "7.13.8.5": out_dir / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan_metrics.json",
    }
    out_json = out_json_by_step[step]
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Canonical alias (stable path) for downstream A-dependence / paper tables.
    # This avoids hard-coding long, step-specific filenames in other scripts.
    canonical_json: Path | None = None
    if step == "7.13.8.5":
        canonical_json = out_dir / "nuclear_effective_potential_canonical_metrics.json"
        canonical_metrics = dict(metrics)
        canonical_metrics["canonical"] = {
            "kind": "nuclear_effective_potential",
            "selected_step": step,
            "selected_metrics_json": str(out_json),
            "note": "This file is an alias copy of the current best-surviving candidate used downstream.",
        }
        canonical_json.write_text(json.dumps(canonical_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] wrote:")
    print(f"  {out_png}")
    if out_csv is not None:
        print(f"  {out_csv}")
    print(f"  {out_json}")
    if canonical_json is not None:
        print(f"  {canonical_json}")


if __name__ == "__main__":
    main()
