from __future__ import annotations

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


def _y_core(
    *, e_mev: float, vc_mev: float, rc_fm: float, mu_mev: float, hbarc_mev_fm: float
) -> float:
    """
    Log-derivative y = u'/u at r=Rc for the inner repulsive core (s-wave, regular at r=0).

    Region I potential: V(r)=+Vc for 0<=r<Rc.
    """
    if rc_fm <= 0.0:
        return float("inf")
    if not (mu_mev > 0 and hbarc_mev_fm > 0 and math.isfinite(vc_mev) and vc_mev >= 0):
        return float("nan")

    if vc_mev == e_mev:
        return 1.0 / rc_fm

    if e_mev < vc_mev:
        alpha = math.sqrt(max(0.0, 2.0 * mu_mev * (vc_mev - e_mev))) / hbarc_mev_fm
        if alpha == 0.0:
            return 1.0 / rc_fm
        return float(alpha * _coth(alpha * rc_fm))

    p = math.sqrt(max(0.0, 2.0 * mu_mev * (e_mev - vc_mev))) / hbarc_mev_fm
    if p == 0.0:
        return 1.0 / rc_fm
    return float(p * _cot(p * rc_fm))


def _y_after_well(*, y_in: float, q_fm1: float, l_fm: float) -> float:
    """
    Propagate log-derivative y=u'/u across a constant potential region of length L (s-wave).
    """
    if not (math.isfinite(q_fm1) and q_fm1 >= 0 and math.isfinite(l_fm) and l_fm >= 0):
        return float("nan")

    if l_fm == 0.0:
        return float(y_in)

    # Hard boundary u(0)=0 corresponds to y_in -> +∞.
    if math.isinf(y_in):
        if q_fm1 == 0.0:
            return float(1.0 / l_fm)
        return float(q_fm1 * _cot(q_fm1 * l_fm))

    if q_fm1 == 0.0:
        denom = 1.0 + (y_in * l_fm)
        if denom == 0.0:
            return float("nan")
        return float(y_in / denom)

    ql = q_fm1 * l_fm
    s = math.sin(ql)
    c = math.cos(ql)

    denom = c + (y_in / q_fm1) * s
    if abs(denom) < 1e-18:
        return float("nan")
    num = (-q_fm1 * s) + (y_in * c)
    return float(num / denom)


def _solve_bound_x(
    *, kappa_fm1: float, l_fm: float, y_c_fm1: float, prefer_x0: float = 2.1, x_max: float = 2.5
) -> float:
    """
    Solve for x = q L in the bound-state condition at energy E=-B.
    """
    if not (math.isfinite(kappa_fm1) and kappa_fm1 > 0 and math.isfinite(l_fm) and l_fm > 0):
        raise ValueError("invalid kappa or L")

    def f(x: float) -> float:
        k2 = x / l_fm
        if not math.isfinite(y_c_fm1):
            y_r = k2 / math.tan(x)
            return y_r + kappa_fm1
        num = (y_c_fm1 * k2 * math.cos(x)) - ((k2 * k2) * math.sin(x))
        den = (y_c_fm1 * math.sin(x)) + (k2 * math.cos(x))
        if abs(den) < 1e-18:
            return float("nan")
        return (num / den) + kappa_fm1

    lo = (math.pi / 2.0) + 1e-7
    hi = math.pi - 1e-7
    # Stay within the ground-state interval (pi/2, pi) to avoid branch jumps.
    # NOTE: For finite y_c, multiple roots can appear within this interval; we select the
    # smallest-x root (minimal depth), which is the continuous ground-branch analogue.
    x_grid = [lo + (hi - lo) * i / 1199 for i in range(1200)]
    prev_x: float | None = None
    prev_f: float | None = None
    brackets: list[tuple[float, float, float]] = []
    for x in x_grid:
        fx = f(x)
        if not math.isfinite(fx):
            prev_x = None
            prev_f = None
            continue
        if prev_x is not None and prev_f is not None:
            if fx == 0 or (fx > 0) != (prev_f > 0):
                mid = 0.5 * (prev_x + x)
                brackets.append((prev_x, x, float(mid)))
        prev_x = x
        prev_f = fx

    if not brackets:
        raise ValueError("no bound-state bracket found for x=qL")

    # Pick the smallest-x bracket (ground branch) and reject deep branches that
    # would correspond to additional nodes (excited-like solutions) in this ansatz.
    brackets.sort(key=lambda t: t[2])
    lo, hi, mid = brackets[0]
    if mid > x_max:
        raise ValueError(f"bound-state root too deep in x=qL (mid={mid:.3f} > {x_max:.3f}); reject branch")
    flo = f(lo)
    fhi = f(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi) and (flo == 0 or (flo > 0) != (fhi > 0))):
        raise ValueError("invalid bound-state bracket after selection")

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if not math.isfinite(fmid):
            mid = math.nextafter(mid, lo)
            fmid = f(mid)
            if not math.isfinite(fmid):
                break
        if fmid == 0 or (hi - lo) < 1e-14:
            return float(mid)
        if (fmid > 0) == (flo > 0):
            lo = mid
            flo = fmid
        else:
            hi = mid
            fhi = fmid
    return float(0.5 * (lo + hi))


def _triplet_depth_from_b(
    *,
    b_mev: float,
    rc_fm: float,
    l_fm: float,
    vc_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, float]:
    """
    Determine V0_t from the bound-state condition at energy -B.
    """
    if not (mu_mev > 0 and b_mev > 0 and l_fm > 0 and vc_mev >= 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid inputs")

    kappa = math.sqrt(2.0 * mu_mev * b_mev) / hbarc_mev_fm
    y_c = _y_core(e_mev=-b_mev, vc_mev=vc_mev, rc_fm=rc_fm, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    x = _solve_bound_x(kappa_fm1=kappa, l_fm=l_fm, y_c_fm1=y_c)
    q = x / l_fm
    v0 = b_mev + (hbarc_mev_fm**2) * (q**2) / (2.0 * mu_mev)
    return {"V0_mev": float(v0), "kappa_fm1": float(kappa), "x": float(x), "q_fm1": float(q), "y_c_fm1": float(y_c)}


def _scattering_length_zero_energy(
    *,
    rc_fm: float,
    l_fm: float,
    vc_mev: float,
    v0_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> float:
    """
    Exact k->0 s-wave scattering length via log-derivative matching:
      a = R - 1/y_R.
    """
    if not (l_fm >= 0 and rc_fm >= 0 and vc_mev >= 0 and v0_mev >= 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        return float("nan")

    r_out = rc_fm + l_fm
    y_c = _y_core(e_mev=0.0, vc_mev=vc_mev, rc_fm=rc_fm, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    q0 = math.sqrt(2.0 * mu_mev * v0_mev) / hbarc_mev_fm if v0_mev > 0 else 0.0
    y_r = _y_after_well(y_in=y_c, q_fm1=q0, l_fm=l_fm)
    if not math.isfinite(y_r) or abs(y_r) < 1e-18:
        return float("nan")
    return float(r_out - (1.0 / y_r))


def _phase_shift(
    *,
    k_fm1: float,
    rc_fm: float,
    l_fm: float,
    vc_mev: float,
    v0_mev: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> float:
    """
    s-wave phase shift δ for the finite repulsive core + attractive well.
    """
    if k_fm1 == 0.0:
        return 0.0
    if not (k_fm1 > 0 and l_fm >= 0 and rc_fm >= 0 and vc_mev >= 0 and v0_mev >= 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        return float("nan")

    r_out = rc_fm + l_fm
    e_mev = (hbarc_mev_fm**2) * (k_fm1**2) / (2.0 * mu_mev)

    y_c = _y_core(e_mev=e_mev, vc_mev=vc_mev, rc_fm=rc_fm, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
    q = math.sqrt(2.0 * mu_mev * (e_mev + v0_mev)) / hbarc_mev_fm if (e_mev + v0_mev) > 0 else 0.0
    y_r = _y_after_well(y_in=y_c, q_fm1=q, l_fm=l_fm)
    if not math.isfinite(y_r):
        return float("nan")

    if abs(y_r) < 1e-18:
        delta = (math.pi / 2.0) - (k_fm1 * r_out)
    else:
        delta = math.atan(k_fm1 / y_r) - (k_fm1 * r_out)

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
    *, rc_fm: float, l_fm: float, vc_mev: float, v0_mev: float, mu_mev: float, hbarc_mev_fm: float
) -> dict[str, object]:
    """
    Fit effective-range expansion:
      k cot δ = c0 + c2 k^2 + c4 k^4,
    where:
      c0=-1/a, c2=r/2, c4=v2.
    """
    k_grid = [0.002 * i for i in range(1, 31)]  # 0.002..0.060 fm^-1

    k2s: list[float] = []
    k4s: list[float] = []
    ys: list[float] = []
    points: list[dict[str, float]] = []

    for k in k_grid:
        delta = _phase_shift(
            k_fm1=k,
            rc_fm=rc_fm,
            l_fm=l_fm,
            vc_mev=vc_mev,
            v0_mev=v0_mev,
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

    a = -1.0 / c0 if c0 != 0 else float("nan")
    r_eff = 2.0 * c2
    v2 = c4

    rms = math.sqrt(
        sum((y - (c0 + c2 * x + c4 * (x * x))) ** 2 for y, x in zip(ys, k2s)) / float(len(ys))
    )

    return {
        "a_fm": float(a),
        "r_eff_fm": float(r_eff),
        "v2_fm3": float(v2),
        "coeffs": {"c0_fm1": float(c0), "c2_fm": float(c2), "c4_fm3": float(c4)},
        "fit_rms_fm1": float(rms),
        "points": points,
    }


def _solve_l_for_triplet_a(
    *,
    rc_fm: float,
    vc_mev: float,
    b_mev: float,
    a_target_fm: float,
    mu_mev: float,
    hbarc_mev_fm: float,
    l_min_fm: float = 0.4,
    l_max_fm: float = 6.0,
) -> tuple[float, float]:
    """
    For fixed (Rc, Vc) and fixed B (thus V0 determined by the bound condition),
    solve L such that a(L)=a_target.
    """
    if not (l_min_fm > 0 and l_max_fm > l_min_fm):
        raise ValueError("invalid L range")

    def a_of_l(l: float) -> float:
        v0 = _triplet_depth_from_b(
            b_mev=b_mev, rc_fm=rc_fm, l_fm=l, vc_mev=vc_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )["V0_mev"]
        return _scattering_length_zero_energy(
            rc_fm=rc_fm, l_fm=l, vc_mev=vc_mev, v0_mev=v0, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )

    cache: dict[float, tuple[float, float]] = {}

    def f_of_l(l: float) -> float:
        if l in cache:
            return cache[l][0]
        try:
            v0 = _triplet_depth_from_b(
                b_mev=b_mev, rc_fm=rc_fm, l_fm=l, vc_mev=vc_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
            )["V0_mev"]
            a_pred = _scattering_length_zero_energy(
                rc_fm=rc_fm, l_fm=l, vc_mev=vc_mev, v0_mev=v0, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
            )
            f = a_pred - a_target_fm
        except Exception:
            cache[l] = (float("nan"), float("nan"))
            return float("nan")
        cache[l] = (float(f), float(v0))
        return float(f)

    def v0_of_l(l: float) -> float:
        if l in cache:
            return cache[l][1]
        _ = f_of_l(l)
        return cache[l][1]

    # Try to bracket around a nuclear-ish scale near L~2 fm, expanding outward.
    l0 = min(max(2.0, l_min_fm), l_max_fm)
    f0 = f_of_l(l0)
    if math.isfinite(f0) and abs(f0) < 1e-12:
        return float(l0), float(v0_of_l(l0))

    bracket: tuple[float, float] | None = None
    step = 0.05
    for _ in range(28):
        lo = max(l_min_fm, l0 - step)
        hi = min(l_max_fm, l0 + step)
        f_lo = f_of_l(lo)
        f_hi = f_of_l(hi)
        if math.isfinite(f_lo) and math.isfinite(f_hi) and ((f_lo > 0) != (f_hi > 0)):
            bracket = (lo, hi)
            break
        if lo == l_min_fm and hi == l_max_fm:
            break
        step *= 1.6

    if bracket is None:
        # Fallback: coarse scan for a sign change, then take the bracket closest to L~2 fm.
        n_scan = 90
        l_grid = [l_min_fm + (l_max_fm - l_min_fm) * i / (n_scan - 1) for i in range(n_scan)]
        prev_l: float | None = None
        prev_f: float | None = None
        brackets: list[tuple[float, float, float]] = []
        for l in l_grid:
            fl = f_of_l(l)
            if not math.isfinite(fl) or abs(fl) > 1e6:
                prev_l = None
                prev_f = None
                continue
            if prev_l is not None and prev_f is not None:
                if (fl > 0) != (prev_f > 0):
                    mid = 0.5 * (prev_l + l)
                    brackets.append((prev_l, l, abs(mid - 2.0)))
            prev_l = l
            prev_f = fl
        if not brackets:
            raise ValueError("no sign-change bracket found for L (a_target)")
        brackets.sort(key=lambda t: t[2])
        bracket = (brackets[0][0], brackets[0][1])

    lo, hi = bracket
    f_lo = f_of_l(lo)
    f_hi = f_of_l(hi)
    if not ((f_lo > 0) != (f_hi > 0)):
        raise ValueError("invalid L bracket")

    for _ in range(55):
        mid = 0.5 * (lo + hi)
        f_mid = f_of_l(mid)
        if f_mid == 0 or (hi - lo) < 1e-12:
            return float(mid), float(v0_of_l(mid))
        if (f_mid > 0) == (f_lo > 0):
            lo = mid
            f_lo = f_mid
        else:
            hi = mid
            f_hi = f_mid

    mid = 0.5 * (lo + hi)
    return float(mid), float(v0_of_l(mid))


def _solve_triplet_geometry_for_vc(
    *,
    vc_mev: float,
    b_mev: float,
    a_t_fm: float,
    r_t_fm: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    """
    For fixed Vc, fit (Rc, L) such that:
      - B is satisfied by V0=V0(B,Rc,L,Vc),
      - a_t matches (by solving L),
      - r_t matches (by solving Rc).
    """
    rc_min = 0.0
    rc_max = 1.2
    cache: dict[float, dict[str, object]] = {}

    def eval_rc(rc: float) -> dict[str, object]:
        if rc in cache:
            return cache[rc]
        l, v0 = _solve_l_for_triplet_a(
            rc_fm=rc, vc_mev=vc_mev, b_mev=b_mev, a_target_fm=a_t_fm, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )
        ere = _fit_kcot_ere(rc_fm=rc, l_fm=l, vc_mev=vc_mev, v0_mev=v0, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        out = {"Rc_fm": float(rc), "L_fm": float(l), "R_fm": float(rc + l), "V0_t_MeV": float(v0), "ere": ere}
        cache[rc] = out
        return out

    def g_of_rc(rc: float) -> float:
        s = eval_rc(rc)
        return float(s["ere"]["r_eff_fm"]) - r_t_fm

    # Adaptive bracket search around Rc~0.4 fm.
    rc0 = min(max(0.4, rc_min), rc_max)
    g0 = g_of_rc(rc0)
    if math.isfinite(g0) and abs(g0) < 1e-10:
        s = eval_rc(rc0)
        s["r_t_fit_minus_target_fm"] = float(g0)
        s["note"] = "near-exact root at rc0"
        return s

    bracket: tuple[float, float] | None = None
    step = 0.05
    evaluated: list[tuple[float, float]] = [(rc0, float(g0))]
    for _ in range(22):
        lo = max(rc_min, rc0 - step)
        hi = min(rc_max, rc0 + step)
        try:
            g_lo = g_of_rc(lo)
            evaluated.append((lo, float(g_lo)))
        except Exception:
            g_lo = float("nan")
        try:
            g_hi = g_of_rc(hi)
            evaluated.append((hi, float(g_hi)))
        except Exception:
            g_hi = float("nan")
        if math.isfinite(g_lo) and math.isfinite(g_hi) and ((g_lo > 0) != (g_hi > 0)):
            bracket = (lo, hi)
            break
        if lo == rc_min and hi == rc_max:
            break
        step *= 1.6

    if bracket is None:
        # Fallback: coarse scan for a bracket.
        rc_grid = [rc_min + (rc_max - rc_min) * i / 60 for i in range(61)]
        prev_rc: float | None = None
        prev_g: float | None = None
        brackets: list[tuple[float, float, float]] = []
        for rc in rc_grid:
            try:
                gg = g_of_rc(rc)
            except Exception:
                prev_rc = None
                prev_g = None
                continue
            evaluated.append((rc, float(gg)))
            if not math.isfinite(gg) or abs(gg) > 1e3:
                prev_rc = None
                prev_g = None
                continue
            if prev_rc is not None and prev_g is not None and ((gg > 0) != (prev_g > 0)):
                mid = 0.5 * (prev_rc + rc)
                brackets.append((prev_rc, rc, abs(mid - 0.4)))
            prev_rc = rc
            prev_g = float(gg)
        if brackets:
            brackets.sort(key=lambda t: t[2])
            bracket = (brackets[0][0], brackets[0][1])

    if bracket is None:
        best_rc, best_g = min(evaluated, key=lambda t: abs(t[1]) if math.isfinite(t[1]) else float("inf"))
        s = eval_rc(best_rc)
        s["r_t_fit_minus_target_fm"] = float(best_g)
        s["note"] = "no bracket for Rc; using best |Δr_t| evaluated"
        return s

    rc_lo, rc_hi = bracket
    g_lo = g_of_rc(rc_lo)
    g_hi = g_of_rc(rc_hi)
    if not ((g_lo > 0) != (g_hi > 0)):
        best_rc, best_g = min(evaluated, key=lambda t: abs(t[1]) if math.isfinite(t[1]) else float("inf"))
        s = eval_rc(best_rc)
        s["r_t_fit_minus_target_fm"] = float(best_g)
        s["note"] = "invalid rc bracket; using best |Δr_t| evaluated"
        return s

    for _ in range(38):
        rc_mid = 0.5 * (rc_lo + rc_hi)
        g_mid = g_of_rc(rc_mid)
        if g_mid == 0 or (rc_hi - rc_lo) < 1e-10:
            rc_lo = rc_mid
            g_lo = g_mid
            break
        if (g_mid > 0) == (g_lo > 0):
            rc_lo = rc_mid
            g_lo = g_mid
        else:
            rc_hi = rc_mid
            g_hi = g_mid

    s = eval_rc(rc_lo)
    s["r_t_fit_minus_target_fm"] = float(g_lo)
    s["note"] = "bisection solve on Rc"
    return s


def _solve_vc_for_triplet(
    *,
    b_mev: float,
    a_t_fm: float,
    r_t_fm: float,
    v2t_target_fm3: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> dict[str, object]:
    """
    Fit Vc such that after fitting (Rc,L) to (a_t,r_t) with B-bound V0,
    the triplet shape parameter v2 matches the target.
    """
    if not math.isfinite(v2t_target_fm3):
        raise ValueError("invalid v2 target")

    cache: dict[float, dict[str, object]] = {}

    def eval_vc(vc: float) -> dict[str, object]:
        if vc in cache:
            return cache[vc]
        geo = _solve_triplet_geometry_for_vc(
            vc_mev=vc, b_mev=b_mev, a_t_fm=a_t_fm, r_t_fm=r_t_fm, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )
        cache[vc] = geo
        return geo

    def h(vc: float) -> float:
        geo = eval_vc(vc)
        v2 = float(geo["ere"]["v2_fm3"])
        return v2 - v2t_target_fm3

    vc_candidates = [0.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 2000.0]
    vals: list[tuple[float, float]] = []
    for vc in vc_candidates:
        try:
            hv = h(vc)
        except Exception:
            continue
        if not math.isfinite(hv) or abs(hv) > 10.0:
            continue
        vals.append((float(vc), float(hv)))

    if len(vals) < 2:
        geo = eval_vc(400.0)
        geo["Vc_MeV"] = float(400.0)
        geo["v2_fit_minus_target_fm3"] = float(float(geo["ere"]["v2_fm3"]) - v2t_target_fm3)
        geo["note_vc"] = "vc candidates failed; using fallback vc=400 MeV"
        return geo

    vals.sort(key=lambda t: t[0])
    brackets: list[tuple[float, float]] = []
    for (v0, f0), (v1, f1) in zip(vals[:-1], vals[1:]):
        if (f0 > 0) != (f1 > 0):
            brackets.append((v0, v1))

    if not brackets:
        best_vc, best_h = min(vals, key=lambda t: abs(t[1]))
        geo = eval_vc(best_vc)
        geo["Vc_MeV"] = float(best_vc)
        geo["v2_fit_minus_target_fm3"] = float(best_h)
        geo["note_vc"] = "no sign-change in vc candidates; using best |Δv2| on candidates"
        return geo

    lo, hi = brackets[0]  # prefer smallest-mid by candidate ordering
    f_lo = h(lo)
    f_hi = h(hi)
    if not ((f_lo > 0) != (f_hi > 0)):
        best_vc, best_h = min(vals, key=lambda t: abs(t[1]))
        geo = eval_vc(best_vc)
        geo["Vc_MeV"] = float(best_vc)
        geo["v2_fit_minus_target_fm3"] = float(best_h)
        geo["note_vc"] = "invalid vc bracket; using best |Δv2| on candidates"
        return geo

    for _ in range(26):
        mid = 0.5 * (lo + hi)
        f_mid = h(mid)
        if f_mid == 0 or (hi - lo) < 1e-4:
            lo = mid
            break
        if (f_mid > 0) == (f_lo > 0):
            lo = mid
            f_lo = f_mid
        else:
            hi = mid
            f_hi = f_mid

    vc_star = float(lo)
    geo = eval_vc(vc_star)
    geo["Vc_MeV"] = vc_star
    geo["v2_fit_minus_target_fm3"] = float(float(geo["ere"]["v2_fm3"]) - v2t_target_fm3)
    geo["note_vc"] = "bisection solve on Vc (fit v2t)"
    return geo


def _solve_singlet_depth_from_a(
    *,
    rc_fm: float,
    l_fm: float,
    vc_mev: float,
    a_s_target_fm: float,
    mu_mev: float,
    hbarc_mev_fm: float,
) -> float:
    """
    With geometry (Rc,L,Vc) fixed, solve V0_s such that the singlet scattering length a_s matches.
    """
    if not math.isfinite(a_s_target_fm) or a_s_target_fm == 0:
        raise ValueError("invalid a_s target")

    def f(v0: float) -> float:
        a_pred = _scattering_length_zero_energy(
            rc_fm=rc_fm, l_fm=l_fm, vc_mev=vc_mev, v0_mev=v0, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )
        return a_pred - a_s_target_fm

    v0_min = 0.0
    v0_max = 500.0
    n_scan = 2001
    v_grid = [v0_min + (v0_max - v0_min) * i / (n_scan - 1) for i in range(n_scan)]

    prev_v: float | None = None
    prev_f: float | None = None
    brackets: list[tuple[float, float, float]] = []
    best_v = None
    best_abs = None

    for v in v_grid:
        fv = f(v)
        if not math.isfinite(fv):
            prev_v = None
            prev_f = None
            continue
        if abs(fv) < 1e6:
            if best_abs is None or abs(fv) < best_abs:
                best_abs = abs(fv)
                best_v = v
        if prev_v is not None and prev_f is not None:
            if fv == 0 or (fv > 0) != (prev_f > 0):
                mid = 0.5 * (prev_v + v)
                brackets.append((prev_v, v, mid))
        prev_v = v
        prev_f = fv

    if brackets:
        brackets.sort(key=lambda t: t[2])
        lo, hi, _ = brackets[0]
        f_lo = f(lo)
        f_hi = f(hi)
        if not ((f_lo > 0) != (f_hi > 0)):
            if best_v is None:
                raise ValueError("no usable bracket for singlet V0")
            return float(best_v)
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if f_mid == 0 or (hi - lo) < 1e-9:
                return float(mid)
            if (f_mid > 0) == (f_lo > 0):
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
                f_hi = f_mid
        return float(0.5 * (lo + hi))

    if best_v is None:
        raise ValueError("no solution for singlet V0_s (a_s)")
    return float(best_v)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    results: list[dict[str, object]] = []
    for d in datasets:
        label = str(d["label"])
        trip = d["triplet"]
        sing = d["singlet"]
        a_t = float(trip["a_t_fm"])
        r_t = float(trip["r_t_fm"])
        v2t_obs = float(trip["v2t_fm3"])

        fit_triplet = _solve_vc_for_triplet(
            b_mev=b_mev,
            a_t_fm=a_t,
            r_t_fm=r_t,
            v2t_target_fm3=v2t_obs,
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        rc = float(fit_triplet["Rc_fm"])
        l = float(fit_triplet["L_fm"])
        rout = float(fit_triplet["R_fm"])
        vc = float(fit_triplet.get("Vc_MeV", float("nan")))
        v0t = float(fit_triplet["V0_t_MeV"])

        v0s = _solve_singlet_depth_from_a(
            rc_fm=rc,
            l_fm=l,
            vc_mev=vc,
            a_s_target_fm=float(sing["a_s_fm"]),
            mu_mev=mu_mev,
            hbarc_mev_fm=hbarc_mev_fm,
        )
        ere_s = _fit_kcot_ere(rc_fm=rc, l_fm=l, vc_mev=vc, v0_mev=v0s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)

        ere_t = fit_triplet["ere"]
        cmp_triplet = {
            "a_t_fit_minus_obs_fm": float(float(ere_t["a_fm"]) - a_t),
            "r_t_fit_minus_obs_fm": float(float(ere_t["r_eff_fm"]) - r_t),
            "v2t_fit_minus_obs_fm3": float(float(ere_t["v2_fm3"]) - v2t_obs),
        }
        cmp_singlet = {
            "a_s_fit_minus_obs_fm": float(float(ere_s["a_fm"]) - float(sing["a_s_fm"])),
            "r_s_pred_minus_obs_fm": float(float(ere_s["r_eff_fm"]) - float(sing["r_s_fm"])),
            "v2s_pred_minus_obs_fm3": float(float(ere_s["v2_fm3"]) - float(sing["v2s_fm3"])),
        }

        results.append(
            {
                "label": label,
                "eq_label": int(d["eq_label"]),
                "inputs": {"B_MeV": float(b_mev), "triplet": trip, "singlet": sing},
                "fit_triplet": {
                    "geometry": {"Rc_fm": rc, "L_fm": l, "R_fm": rout},
                    "Vc_MeV": vc,
                    "V0_t_MeV": v0t,
                    "ere": ere_t,
                    "notes": [
                        "Triplet fit uses B to determine V0_t via the bound-state condition.",
                        "Then (Rc,L) are solved to match (a_t,r_t) and Vc is solved to match v2t.",
                    ],
                    "fit_diagnostics": {
                        "r_t_fit_minus_target_fm": float(fit_triplet.get("r_t_fit_minus_target_fm", 0.0)),
                        "v2_fit_minus_target_fm3": float(fit_triplet.get("v2_fit_minus_target_fm3", float("nan"))),
                        "note_rc": str(fit_triplet.get("note", "")),
                        "note_vc": str(fit_triplet.get("note_vc", "")),
                    },
                },
                "fit_singlet": {
                    "V0_s_MeV": float(v0s),
                    "ere": ere_s,
                    "note": "Singlet channel uses the same geometry (Rc,L,Vc) but fits only a_s; (r_s, v2s) are predictions for this ansatz class.",
                },
                "comparison": {"triplet": cmp_triplet, "singlet": cmp_singlet},
            }
        )

    v2t_obs_list = [float(d["triplet"]["v2t_fm3"]) for d in datasets]
    v2s_obs_list = [float(d["singlet"]["v2s_fm3"]) for d in datasets]
    rs_obs_list = [float(d["singlet"]["r_s_fm"]) for d in datasets]
    v2t_env = {"min": float(min(v2t_obs_list)), "max": float(max(v2t_obs_list))}
    v2s_env = {"min": float(min(v2s_obs_list)), "max": float(max(v2s_obs_list))}
    rs_env = {"min": float(min(rs_obs_list)), "max": float(max(rs_obs_list))}

    rs_pred = [float(r["fit_singlet"]["ere"]["r_eff_fm"]) for r in results]
    v2s_pred = [float(r["fit_singlet"]["ere"]["v2_fm3"]) for r in results]
    rs_within = all(rs_env["min"] <= v <= rs_env["max"] for v in rs_pred if math.isfinite(v))
    v2s_within = all(v2s_env["min"] <= v <= v2s_env["max"] for v in v2s_pred if math.isfinite(v))

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.6, 8.6), dpi=160, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, wspace=0.25, hspace=0.25)

    for row, r in enumerate(results):
        label = str(r["label"])
        geo = r["fit_triplet"]["geometry"]
        rc = float(geo["Rc_fm"])
        l = float(geo["L_fm"])
        rout = float(geo["R_fm"])
        vc = float(r["fit_triplet"]["Vc_MeV"])
        v0t = float(r["fit_triplet"]["V0_t_MeV"])
        v0s = float(r["fit_singlet"]["V0_s_MeV"])

        ax0 = fig.add_subplot(gs[row, 0])
        r_plot = [i * 0.02 for i in range(0, 501)]  # 0..10 fm

        def v_profile(rr: float, *, v0: float) -> float:
            if rr < rc:
                return min(vc, 200.0)
            if rr < rout:
                return -v0
            return 0.0

        vt = [v_profile(rr, v0=v0t) for rr in r_plot]
        vs = [v_profile(rr, v0=v0s) for rr in r_plot]
        ax0.plot(r_plot, vt, lw=2.0, color="tab:blue", label="triplet (fit B,a_t,r_t,v2t)")
        ax0.plot(r_plot, vs, lw=2.0, color="tab:orange", label="singlet (fit a_s only)")
        ax0.axvline(rc, color="0.35", lw=1.0, ls=":")
        ax0.axvline(rout, color="0.35", lw=1.0, ls=":")
        ax0.set_xlabel("r (fm)")
        ax0.set_ylabel("V(r) (MeV)")
        ax0.set_title(f"{label}: finite core + well (geometry shared)")
        ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
        ax0.legend(frameon=True, fontsize=8, loc="lower right")
        ax0.text(
            0.02,
            0.98,
            f"Rc≈{rc:.3f} fm, R≈{rout:.3f} fm\nVc≈{vc:.1f} MeV\nV0_t≈{v0t:.2f} MeV, V0_s≈{v0s:.2f} MeV",
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
        ax1.set_title("Triplet: ERE fit (v2 fitted)")
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
        names = ["v2t(fit)", "r_s(pred)", "v2s(pred)"]
        deltas = [float(dt["v2t_fit_minus_obs_fm3"]), float(ds["r_s_pred_minus_obs_fm"]), float(ds["v2s_pred_minus_obs_fm3"])]
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
            "Triplet: v2 is fitted\nSinglet: r_s,v2 are predicted",
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
        )

    fig.suptitle(
        "Phase 7 / Step 7.9.5: finite repulsive core + well — fit triplet (B,a_t,r_t,v2t), predict singlet (r_s,v2s)",
        y=1.02,
    )

    out_png = out_dir / "nuclear_effective_potential_finite_core_well.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    codata_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    codata_manifest = codata_dir / "manifest.json"
    codata_extracted = codata_dir / "extracted_values.json"
    np_manifest = root / "data" / "quantum" / "sources" / "np_scattering_low_energy_arxiv_0704_1024v1_manifest.json"
    np_extracted = root / "data" / "quantum" / "sources" / "np_scattering_low_energy_arxiv_0704_1024v1_extracted.json"

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.9.5",
        "model": {
            "effective_equation": "Nonrelativistic s-wave Schrödinger equation for relative motion with an effective potential V(r)=μ φ(r)=-μ c^2 u(r).",
            "ansatz": "Finite repulsive core + attractive square well (4 params: Rc, L, Vc, V0; with V0_t fixed by B, and Vc fit to v2t).",
            "fit_constraints": {
                "triplet": ["B (CODATA mass defect)", "a_t (np low-energy)", "r_t (np low-energy)", "v2t (shape parameter; fit)"],
                "singlet": ["a_s (np low-energy)"],
            },
            "prediction_targets": {"singlet": ["r_s", "v2s"]},
            "positioning": [
                "Phenomenological constraint (effective model), not a first-principles derivation of nuclear forces.",
                "The ansatz class is intentionally low-parameter to avoid 'arbitrary function fit'.",
                "Compared to Step 7.9.4 (hard core), Step 7.9.5 adds one degree of freedom (finite Vc) and uses v2t as an explicit constraint to remove residual arbitrariness.",
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
            "observed_envelope_r_s_fm": rs_env,
            "observed_envelope_v2t_fm3": v2t_env,
            "observed_envelope_v2s_fm3": v2s_env,
            "predicted_r_s_within_envelope": bool(rs_within),
            "predicted_v2s_within_envelope": bool(v2s_within),
            "notes": [
                "Eq.(18) and Eq.(19) are two phase-shift analyses in the same primary source; their difference is treated as an analysis-dependent systematics proxy.",
                "In this step, v2t is used as a fit constraint to remove the residual degree of freedom introduced by finite Vc.",
            ],
        },
        "falsification": {
            "acceptance_criteria": [
                "Under the finite-core + well ansatz class, after fitting triplet (B,a_t,r_t,v2t) and fitting singlet a_s with shared geometry, the predicted singlet (r_s,v2s) should be compatible with the observed range (eq18–eq19).",
                "If singlet (r_s,v2s) falls outside the analysis-dependent envelope, the ansatz class is rejected as a representation of the nuclear-scale u-profile.",
            ],
            "within_envelope": {"r_s": bool(rs_within), "v2s": bool(v2s_within)},
        },
        "outputs": {"png": str(out_png)},
    }

    out_json = out_dir / "nuclear_effective_potential_finite_core_well_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] wrote:")
    print(f"  {out_png}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
