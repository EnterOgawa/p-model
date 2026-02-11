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


def _solve_bound_x(*, kappa_fm1: float, l_fm: float) -> float:
    """
    Solve x in (pi/2, pi) for the s-wave bound-state condition (hard-core at r=R_c):

      k cot(kL) = -kappa,  with L = R - R_c, k = x/L.

    i.e.
      x cot x + kappa L = 0.
    """
    if not (math.isfinite(kappa_fm1) and kappa_fm1 > 0 and math.isfinite(l_fm) and l_fm > 0):
        raise ValueError("invalid kappa or L")

    lo = (math.pi / 2.0) + 1e-10
    hi = math.pi - 1e-10

    def f(x: float) -> float:
        return (x / math.tan(x)) + (kappa_fm1 * l_fm)

    flo = f(lo)
    fhi = f(hi)
    if not (flo > 0 and fhi < 0):
        raise ValueError(f"no bound-state bracket: f(lo)={flo}, f(hi)={fhi}, kappaL={kappa_fm1*l_fm}")

    for _ in range(90):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0 or (hi - lo) < 1e-14:
            return mid
        if fmid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _well_depth_from_l(*, mu_mev: float, b_mev: float, l_fm: float, hbarc_mev_fm: float) -> dict[str, float]:
    """
    Hard-core + attractive square well:
      V(r)=+∞ for r<Rc
      V(r)=-V0 for Rc<=r<R
      V(r)=0 for r>=R

    For a bound state at energy -B, the condition depends only on L=R-Rc:
      k cot(kL) = -kappa.
    """
    if not (mu_mev > 0 and b_mev > 0 and l_fm > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid inputs")
    kappa = math.sqrt(2.0 * mu_mev * b_mev) / hbarc_mev_fm
    x = _solve_bound_x(kappa_fm1=kappa, l_fm=l_fm)
    k = x / l_fm
    v0 = b_mev + (hbarc_mev_fm**2) * (k**2) / (2.0 * mu_mev)
    return {"V0_mev": float(v0), "kappa_fm1": float(kappa), "k_fm1": float(k), "x": float(x)}


def _scattering_length_hard_core_well(*, rc_fm: float, l_fm: float, v0_mev: float, mu_mev: float, hbarc_mev_fm: float) -> float:
    """
    k->0 exact s-wave scattering length for hard-core + attractive well:

      a = R - tan(qL)/q,  with R=Rc+L, q = sqrt(2μV0)/(ħc).
    """
    if not (rc_fm >= 0 and l_fm > 0 and v0_mev > 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        return float("nan")
    r_fm = rc_fm + l_fm
    q = math.sqrt(2.0 * mu_mev * v0_mev) / hbarc_mev_fm
    if not (math.isfinite(q) and q > 0):
        return float("nan")
    return float(r_fm - (math.tan(q * l_fm) / q))


def _phase_shift_hard_core_well(*, k_fm1: float, rc_fm: float, l_fm: float, v0_mev: float, mu_mev: float, hbarc_mev_fm: float) -> float:
    """
    s-wave phase shift for hard-core + attractive well.

    Outside: u ~ sin(kr + δ). Inside well (Rc<=r<=R): u ~ sin(q(r-Rc)), enforcing u(Rc)=0.
    Matching at r=R gives:
      tan(kR + δ) = (k/q) tan(qL)
    with R=Rc+L, q=sqrt(k^2 + k0^2), k0^2=2μV0/(ħc)^2.
    """
    if k_fm1 == 0.0:
        return 0.0
    if not (k_fm1 > 0 and l_fm > 0 and v0_mev > 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        return float("nan")
    r_fm = rc_fm + l_fm
    k0 = math.sqrt(2.0 * mu_mev * v0_mev) / hbarc_mev_fm
    q = math.sqrt(k_fm1**2 + k0**2)
    t = (k_fm1 / q) * math.tan(q * l_fm)
    delta = math.atan(t) - (k_fm1 * r_fm)

    # Wrap to a principal interval near 0 (k is small in our usage).
    while delta > math.pi / 2.0:
        delta -= math.pi
    while delta < -math.pi / 2.0:
        delta += math.pi
    return float(delta)


def _solve_3x3(a: list[list[float]], b: list[float]) -> list[float]:
    """
    Solve A x = b for 3x3 A using Gaussian elimination with partial pivot.
    """
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
    *, rc_fm: float, l_fm: float, v0_mev: float, mu_mev: float, hbarc_mev_fm: float
) -> dict[str, object]:
    """
    Fit effective-range expansion:
      k cot δ = c0 + c2 k^2 + c4 k^4,
    where:
      c0 = -1/a,  c2 = r/2,  c4 = v2.
    """
    k_grid = [0.002 * i for i in range(1, 31)]  # 0.002..0.060 fm^-1

    k2s: list[float] = []
    k4s: list[float] = []
    ys: list[float] = []
    points: list[dict[str, float]] = []

    for k in k_grid:
        delta = _phase_shift_hard_core_well(
            k_fm1=k, rc_fm=rc_fm, l_fm=l_fm, v0_mev=v0_mev, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )
        if not math.isfinite(delta):
            continue
        t = math.tan(delta)
        if abs(t) < 1e-15:
            continue
        k2 = k * k
        y = k / t  # k cot δ
        if not (math.isfinite(y) and math.isfinite(k2)):
            continue
        k2s.append(k2)
        k4s.append(k2 * k2)
        ys.append(y)
        points.append({"k_fm1": float(k), "kcot_fm1": float(y), "delta_rad": float(delta)})

    if len(ys) < 8:
        raise ValueError("insufficient points for ERE fit")

    s00 = float(len(ys))
    s01 = float(sum(k2s))
    s02 = float(sum(k4s))
    s11 = float(sum(x * x for x in k2s))  # sum(k^4)
    s12 = float(sum(x * z for x, z in zip(k2s, k4s)))  # sum(k^6)
    s22 = float(sum(z * z for z in k4s))  # sum(k^8)

    b0 = float(sum(ys))
    b1 = float(sum(y * x for y, x in zip(ys, k2s)))
    b2 = float(sum(y * z for y, z in zip(ys, k4s)))

    c0, c2, c4 = _solve_3x3([[s00, s01, s02], [s01, s11, s12], [s02, s12, s22]], [b0, b1, b2])

    a = -1.0 / c0 if c0 != 0 else float("nan")
    r_eff = 2.0 * c2
    v2 = c4

    rss = 0.0
    for y, k2, k4 in zip(ys, k2s, k4s):
        yhat = c0 + c2 * k2 + c4 * k4
        rss += (y - yhat) ** 2
    rms = math.sqrt(rss / len(ys))

    return {
        "a_fm": float(a),
        "r_eff_fm": float(r_eff),
        "v2_fm3": float(v2),
        "coeffs": {"c0_fm1": float(c0), "c2_fm": float(c2), "c4_fm3": float(c4)},
        "fit_rms_fm1": float(rms),
        "points": points,
    }


def _solve_l_for_triplet_a(
    *, rc_fm: float, a_target_fm: float, mu_mev: float, b_mev: float, hbarc_mev_fm: float
) -> float:
    """
    For fixed Rc and fixed bound state energy B, solve L such that a(L; Rc, V0(L)) = a_target.
    """
    l_min = 0.5
    l_max = 6.0
    n_scan = 2000
    l_grid = [l_min + (l_max - l_min) * i / (n_scan - 1) for i in range(n_scan)]

    vals: list[tuple[float, float]] = []
    for l in l_grid:
        try:
            v = _well_depth_from_l(mu_mev=mu_mev, b_mev=b_mev, l_fm=l, hbarc_mev_fm=hbarc_mev_fm)["V0_mev"]
            a_pred = _scattering_length_hard_core_well(
                rc_fm=rc_fm, l_fm=l, v0_mev=v, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
            )
        except Exception:
            continue
        if not math.isfinite(a_pred) or abs(a_pred) > 1e4:
            continue
        f = a_pred - a_target_fm
        if not math.isfinite(f) or abs(f) > 1e4:
            continue
        vals.append((l, f))

    if len(vals) < 3:
        raise ValueError("no valid L grid points for a(L)")

    brackets: list[tuple[float, float, float]] = []
    for (l0, f0), (l1, f1) in zip(vals[:-1], vals[1:]):
        if f0 == 0:
            return l0
        if (f0 > 0) != (f1 > 0):
            mid = 0.5 * (l0 + l1)
            score = abs(mid - 2.0) + 0.2 * abs(rc_fm - 0.5)
            brackets.append((l0, l1, score))

    if not brackets:
        raise ValueError("no sign-change bracket found for L (a_target)")

    brackets.sort(key=lambda t: t[2])
    lo, hi, _ = brackets[0]

    def f_of_l(l: float) -> float:
        v = _well_depth_from_l(mu_mev=mu_mev, b_mev=b_mev, l_fm=l, hbarc_mev_fm=hbarc_mev_fm)["V0_mev"]
        a_pred = _scattering_length_hard_core_well(
            rc_fm=rc_fm, l_fm=l, v0_mev=v, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )
        return a_pred - a_target_fm

    f_lo = f_of_l(lo)
    f_hi = f_of_l(hi)
    if not ((f_lo > 0) != (f_hi > 0)):
        raise ValueError("invalid L bracket")

    for _ in range(90):
        mid = 0.5 * (lo + hi)
        f_mid = f_of_l(mid)
        if f_mid == 0 or (hi - lo) < 1e-12:
            return mid
        if (f_mid > 0) == (f_lo > 0):
            lo = mid
            f_lo = f_mid
        else:
            hi = mid
            f_hi = f_mid
    return 0.5 * (lo + hi)


def _solve_triplet_core_well_geometry(
    *, a_t_fm: float, r_t_fm: float, mu_mev: float, b_mev: float, hbarc_mev_fm: float
) -> dict[str, float]:
    """
    Fit hard-core + well geometry (Rc, L) such that, with V0 fixed by B:
      a_t matches and r_t matches.
    Then V0 follows from (B,L).
    """
    rc_min = 0.0
    rc_max = 1.2
    n_scan = 241
    rc_grid = [rc_min + (rc_max - rc_min) * i / (n_scan - 1) for i in range(n_scan)]

    samples: list[dict[str, float]] = []
    for rc in rc_grid:
        try:
            l = _solve_l_for_triplet_a(
                rc_fm=rc, a_target_fm=a_t_fm, mu_mev=mu_mev, b_mev=b_mev, hbarc_mev_fm=hbarc_mev_fm
            )
            v0 = _well_depth_from_l(mu_mev=mu_mev, b_mev=b_mev, l_fm=l, hbarc_mev_fm=hbarc_mev_fm)["V0_mev"]
            ere = _fit_kcot_ere(rc_fm=rc, l_fm=l, v0_mev=v0, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
            r_pred = float(ere["r_eff_fm"])
            if not math.isfinite(r_pred):
                continue
            samples.append({"rc": float(rc), "l": float(l), "v0": float(v0), "g": float(r_pred - r_t_fm)})
        except Exception:
            continue

    if len(samples) < 5:
        raise ValueError("insufficient samples to solve Rc")

    bracket: tuple[float, float] | None = None
    samples_sorted = sorted(samples, key=lambda d: d["rc"])
    for s0, s1 in zip(samples_sorted[:-1], samples_sorted[1:]):
        g0 = s0["g"]
        g1 = s1["g"]
        if g0 == 0:
            bracket = (s0["rc"], s0["rc"])
            break
        if (g0 > 0) != (g1 > 0):
            bracket = (s0["rc"], s1["rc"])
            break

    def g_of_rc(rc: float) -> tuple[float, float, float]:
        l = _solve_l_for_triplet_a(
            rc_fm=rc, a_target_fm=a_t_fm, mu_mev=mu_mev, b_mev=b_mev, hbarc_mev_fm=hbarc_mev_fm
        )
        v0 = _well_depth_from_l(mu_mev=mu_mev, b_mev=b_mev, l_fm=l, hbarc_mev_fm=hbarc_mev_fm)["V0_mev"]
        ere = _fit_kcot_ere(rc_fm=rc, l_fm=l, v0_mev=v0, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)
        r_pred = float(ere["r_eff_fm"])
        return float(r_pred - r_t_fm), float(l), float(v0)

    if bracket is None:
        best = min(samples_sorted, key=lambda d: abs(d["g"]))
        return {
            "Rc_fm": float(best["rc"]),
            "L_fm": float(best["l"]),
            "R_fm": float(best["rc"] + best["l"]),
            "V0_t_MeV": float(best["v0"]),
            "r_t_fit_minus_target_fm": float(best["g"]),
            "note": "no sign-change in g(rc); using best |Δr_t| sample",
        }

    rc_lo, rc_hi = bracket
    if rc_lo == rc_hi:
        g0, l0, v0 = g_of_rc(rc_lo)
        return {
            "Rc_fm": float(rc_lo),
            "L_fm": float(l0),
            "R_fm": float(rc_lo + l0),
            "V0_t_MeV": float(v0),
            "r_t_fit_minus_target_fm": float(g0),
            "note": "exact root found on scan grid",
        }

    g_lo, l_lo, v0_lo = g_of_rc(rc_lo)
    g_hi, l_hi, v0_hi = g_of_rc(rc_hi)
    if not ((g_lo > 0) != (g_hi > 0)):
        raise ValueError("invalid rc bracket after evaluation")

    for _ in range(70):
        rc_mid = 0.5 * (rc_lo + rc_hi)
        g_mid, l_mid, v0_mid = g_of_rc(rc_mid)
        if abs(g_mid) < 1e-9 or (rc_hi - rc_lo) < 1e-9:
            return {
                "Rc_fm": float(rc_mid),
                "L_fm": float(l_mid),
                "R_fm": float(rc_mid + l_mid),
                "V0_t_MeV": float(v0_mid),
                "r_t_fit_minus_target_fm": float(g_mid),
                "note": "bisection solved",
            }
        if (g_mid > 0) == (g_lo > 0):
            rc_lo = rc_mid
            g_lo = g_mid
            l_lo = l_mid
            v0_lo = v0_mid
        else:
            rc_hi = rc_mid
            g_hi = g_mid
            l_hi = l_mid
            v0_hi = v0_mid

    rc_mid = 0.5 * (rc_lo + rc_hi)
    g_mid, l_mid, v0_mid = g_of_rc(rc_mid)
    return {
        "Rc_fm": float(rc_mid),
        "L_fm": float(l_mid),
        "R_fm": float(rc_mid + l_mid),
        "V0_t_MeV": float(v0_mid),
        "r_t_fit_minus_target_fm": float(g_mid),
        "note": "bisection max-iter",
    }


def _solve_depth_for_singlet_a(
    *, a_target_fm: float, rc_fm: float, l_fm: float, mu_mev: float, hbarc_mev_fm: float
) -> float:
    """
    For fixed geometry (Rc,L), solve V0 so that scattering length a matches a_target.

    For large negative a (singlet), choose the branch with qL in (0, pi/2) (no bound state).
    """
    r_fm = rc_fm + l_fm
    if not (l_fm > 0 and mu_mev > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid inputs")

    q_lo = 1e-9
    q_hi = (math.pi / (2.0 * l_fm)) - 1e-9

    def a_of_q(q: float) -> float:
        return float(r_fm - (math.tan(q * l_fm) / q))

    f_lo = a_of_q(q_lo) - a_target_fm
    f_hi = a_of_q(q_hi) - a_target_fm
    if not (math.isfinite(f_lo) and math.isfinite(f_hi) and (f_lo > 0) != (f_hi > 0)):
        raise ValueError("no q bracket for a_target in (0,pi/2L)")

    for _ in range(120):
        q_mid = 0.5 * (q_lo + q_hi)
        f_mid = a_of_q(q_mid) - a_target_fm
        if abs(f_mid) < 1e-12 or (q_hi - q_lo) < 1e-12:
            q_lo = q_mid
            q_hi = q_mid
            break
        if (f_mid > 0) == (f_lo > 0):
            q_lo = q_mid
            f_lo = f_mid
        else:
            q_hi = q_mid
            f_hi = f_mid

    q = 0.5 * (q_lo + q_hi)
    v0 = (hbarc_mev_fm**2) * (q**2) / (2.0 * mu_mev)
    return float(v0)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exact SI constants (conversion only).
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    hbarc_mev_fm = 197.326_980_4  # treat as exact here

    # Baseline constants for B and reduced mass μ.
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

    # np scattering parameter sets (eq18/eq19)
    np_sets = _load_np_scattering_sets(root=root)
    eq18 = np_sets.get(18)
    eq19 = np_sets.get(19)
    if not (isinstance(eq18, dict) and isinstance(eq19, dict)):
        raise SystemExit("[fail] missing eq18/eq19 in extracted np scattering JSON")

    datasets = [
        {
            "label": "eq18 (GWU/SAID)",
            "eq_label": 18,
            "a_t_fm": _get_value(eq18, "a_t_fm"),
            "r_t_fm": _get_value(eq18, "r_t_fm"),
            "v2t_fm3": _get_value(eq18, "v2t_fm3"),
            "a_s_fm": _get_value(eq18, "a_s_fm"),
            "r_s_fm": _get_value(eq18, "r_s_fm"),
            "v2s_fm3": _get_value(eq18, "v2s_fm3"),
        },
        {
            "label": "eq19 (Nijmegen)",
            "eq_label": 19,
            "a_t_fm": _get_value(eq19, "a_t_fm"),
            "r_t_fm": _get_value(eq19, "r_t_fm"),
            "v2t_fm3": _get_value(eq19, "v2t_fm3"),
            "a_s_fm": _get_value(eq19, "a_s_fm"),
            "r_s_fm": _get_value(eq19, "r_s_fm"),
            "v2s_fm3": _get_value(eq19, "v2s_fm3"),
        },
    ]

    results: list[dict[str, object]] = []
    for d in datasets:
        geom = _solve_triplet_core_well_geometry(
            a_t_fm=float(d["a_t_fm"]),
            r_t_fm=float(d["r_t_fm"]),
            mu_mev=float(mu_mev),
            b_mev=float(b_mev),
            hbarc_mev_fm=hbarc_mev_fm,
        )
        rc = float(geom["Rc_fm"])
        l = float(geom["L_fm"])
        r_outer = float(geom["R_fm"])
        v0_t = float(geom["V0_t_MeV"])

        a_t_exact = _scattering_length_hard_core_well(
            rc_fm=rc, l_fm=l, v0_mev=v0_t, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )
        ere_t = _fit_kcot_ere(rc_fm=rc, l_fm=l, v0_mev=v0_t, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)

        v0_s = _solve_depth_for_singlet_a(
            a_target_fm=float(d["a_s_fm"]),
            rc_fm=rc,
            l_fm=l,
            mu_mev=float(mu_mev),
            hbarc_mev_fm=hbarc_mev_fm,
        )
        a_s_exact = _scattering_length_hard_core_well(
            rc_fm=rc, l_fm=l, v0_mev=v0_s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm
        )
        ere_s = _fit_kcot_ere(rc_fm=rc, l_fm=l, v0_mev=v0_s, mu_mev=mu_mev, hbarc_mev_fm=hbarc_mev_fm)

        out = {
            "label": str(d["label"]),
            "eq_label": int(d["eq_label"]),
            "inputs": {
                "B_MeV": float(b_mev),
                "triplet": {"a_t_fm": float(d["a_t_fm"]), "r_t_fm": float(d["r_t_fm"]), "v2t_fm3": float(d["v2t_fm3"])},
                "singlet": {"a_s_fm": float(d["a_s_fm"]), "r_s_fm": float(d["r_s_fm"]), "v2s_fm3": float(d["v2s_fm3"])},
            },
            "fit_triplet": {
                "geometry": {"Rc_fm": rc, "L_fm": l, "R_fm": r_outer},
                "V0_t_MeV": v0_t,
                "a_t_exact_fm": float(a_t_exact),
                "ere": ere_t,
                "note": "Triplet channel is fitted to (B, a_t, r_t) by construction (via Rc,L with V0_t(B,L)).",
            },
            "fit_singlet": {
                "V0_s_MeV": float(v0_s),
                "depth_ratio_V0s_over_V0t": float(v0_s / v0_t) if v0_t > 0 else None,
                "a_s_exact_fm": float(a_s_exact),
                "ere": ere_s,
                "note": "Singlet channel uses the same geometry (Rc,L) but fixes only a_s; (r_s, v2s) are predictions for this ansatz class.",
            },
            "comparison": {
                "triplet": {
                    "a_t_fit_minus_obs_fm": float(a_t_exact) - float(d["a_t_fm"]),
                    "r_t_fit_minus_obs_fm": float(ere_t["r_eff_fm"]) - float(d["r_t_fm"]),
                    "v2t_pred_minus_obs_fm3": float(ere_t["v2_fm3"]) - float(d["v2t_fm3"]),
                },
                "singlet": {
                    "a_s_fit_minus_obs_fm": float(a_s_exact) - float(d["a_s_fm"]),
                    "r_s_pred_minus_obs_fm": float(ere_s["r_eff_fm"]) - float(d["r_s_fm"]),
                    "v2s_pred_minus_obs_fm3": float(ere_s["v2_fm3"]) - float(d["v2s_fm3"]),
                },
            },
        }
        results.append(out)

    v2t_obs = [float(d["v2t_fm3"]) for d in datasets]
    v2s_obs = [float(d["v2s_fm3"]) for d in datasets]
    v2t_env = {"min": float(min(v2t_obs)), "max": float(max(v2t_obs))}
    v2s_env = {"min": float(min(v2s_obs)), "max": float(max(v2s_obs))}

    v2t_pred = [float(r["fit_triplet"]["ere"]["v2_fm3"]) for r in results]
    v2s_pred = [float(r["fit_singlet"]["ere"]["v2_fm3"]) for r in results]
    v2t_within = all(v2t_env["min"] <= v <= v2t_env["max"] for v in v2t_pred if math.isfinite(v))
    v2s_within = all(v2s_env["min"] <= v <= v2s_env["max"] for v in v2s_pred if math.isfinite(v))

    # Plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.6, 8.6), dpi=160, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, wspace=0.25, hspace=0.25)

    for row, r in enumerate(results):
        label = str(r["label"])
        rc = float(r["fit_triplet"]["geometry"]["Rc_fm"])
        l = float(r["fit_triplet"]["geometry"]["L_fm"])
        rout = float(r["fit_triplet"]["geometry"]["R_fm"])
        v0t = float(r["fit_triplet"]["V0_t_MeV"])
        v0s = float(r["fit_singlet"]["V0_s_MeV"])

        # (A) potential profiles (cap hard core as +200 MeV for visualization)
        ax0 = fig.add_subplot(gs[row, 0])
        r_plot = [i * 0.02 for i in range(0, 501)]  # 0..10 fm

        def v_profile(rr: float, *, v0: float) -> float:
            if rr < rc:
                return 200.0
            if rr < rout:
                return -v0
            return 0.0

        vt = [v_profile(rr, v0=v0t) for rr in r_plot]
        vs = [v_profile(rr, v0=v0s) for rr in r_plot]
        ax0.plot(r_plot, vt, lw=2.0, color="tab:blue", label="triplet (fit B,a_t,r_t)")
        ax0.plot(r_plot, vs, lw=2.0, color="tab:orange", label="singlet (fit a_s only)")
        ax0.axvline(rc, color="0.35", lw=1.0, ls=":")
        ax0.axvline(rout, color="0.35", lw=1.0, ls=":")
        ax0.set_xlabel("r (fm)")
        ax0.set_ylabel("V(r) (MeV)")
        ax0.set_title(f"{label}: hard-core + well (geometry shared)")
        ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
        ax0.legend(frameon=True, fontsize=8, loc="lower right")
        ax0.text(
            0.02,
            0.98,
            f"Rc≈{rc:.3f} fm, R≈{rout:.3f} fm\nV0_t≈{v0t:.2f} MeV, V0_s≈{v0s:.2f} MeV",
            transform=ax0.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
        )

        # (B) triplet ERE fit: k cot δ vs k^2
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
        x_line = [0.0, max(xs)]
        y_line = [c0 + c2 * x + c4 * (x * x) for x in x_line]
        ax1.plot(x_line, y_line, "-", lw=2.0, color="0.35", label="ERE fit")
        ax1.set_xlabel("k² (fm⁻²)")
        ax1.set_ylabel("k cot δ (fm⁻¹)")
        ax1.set_title("Triplet: ERE fit (predict v2)")
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

        # (C) prediction deltas
        ax2 = fig.add_subplot(gs[row, 2])
        dt = r["comparison"]["triplet"]
        ds = r["comparison"]["singlet"]
        names = ["v2t", "r_s", "v2s"]
        deltas = [
            float(dt["v2t_pred_minus_obs_fm3"]),
            float(ds["r_s_pred_minus_obs_fm"]),
            float(ds["v2s_pred_minus_obs_fm3"]),
        ]
        ax2.axhline(0.0, color="0.3", lw=1.0)
        ax2.bar(range(len(names)), deltas, color=["tab:blue", "tab:orange", "tab:orange"], alpha=0.85)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names)
        ax2.set_ylabel("pred − obs (units: fm³, fm, fm³)")
        ax2.set_title("Predictions vs observed (eq source)")
        ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
        ax2.text(
            0.02,
            0.98,
            "Triplet: v2 is predicted\nSinglet: r_s,v2 are predicted",
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
        )

    fig.suptitle(
        "Phase 7 / Step 7.9.4: hard-core + well ansatz — fit triplet (B,a_t,r_t), predict v2 and singlet (r_s,v2)",
        y=1.02,
    )

    out_png = out_dir / "nuclear_effective_potential_core_well.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    codata_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    codata_manifest = codata_dir / "manifest.json"
    np_manifest = root / "data" / "quantum" / "sources" / "np_scattering_low_energy_arxiv_0704_1024v1_manifest.json"
    np_extracted = root / "data" / "quantum" / "sources" / "np_scattering_low_energy_arxiv_0704_1024v1_extracted.json"

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.9.4",
        "model": {
            "effective_equation": "Nonrelativistic s-wave Schrödinger equation for relative motion with an effective potential V(r)=μ φ(r)=-μ c^2 u(r).",
            "ansatz": "Hard-core + attractive square well (3 parameters for triplet: Rc, L, V0; with V0 fixed by (B,L)).",
            "fit_constraints": {"triplet": ["B (CODATA mass defect)", "a_t (np low-energy)", "r_t (np low-energy)"], "singlet": ["a_s (np low-energy)"]},
            "prediction_targets": {"triplet": ["v2t (shape parameter)"], "singlet": ["r_s", "v2s"]},
            "positioning": [
                "Phenomenological constraint (effective model), not a first-principles derivation of nuclear forces.",
                "The ansatz class is intentionally low-parameter to avoid 'arbitrary function fit'.",
            ],
        },
        "sources": [
            {
                "dataset": "NIST Cuu CODATA constants (mp,mn,md) for deuteron binding baseline",
                "local_manifest": str(codata_manifest),
                "local_manifest_sha256": _sha256(codata_manifest) if codata_manifest.exists() else None,
                "local_extracted": str(codata_dir / "extracted_values.json"),
                "local_extracted_sha256": _sha256(codata_dir / "extracted_values.json")
                if (codata_dir / "extracted_values.json").exists()
                else None,
            },
            {
                "dataset": "np scattering low-energy parameters (arXiv:0704.1024v1; eq18–eq19)",
                "local_manifest": str(np_manifest),
                "local_manifest_sha256": _sha256(np_manifest) if np_manifest.exists() else None,
                "local_extracted": str(np_extracted),
                "local_extracted_sha256": _sha256(np_extracted) if np_extracted.exists() else None,
            },
        ],
        "constants": {"hbarc_MeV_fm": hbarc_mev_fm, "mu_c2_MeV": float(mu_mev), "B_MeV": float(b_mev)},
        "results_by_dataset": results,
        "systematics_proxy": {
            "observed_envelope_v2t_fm3": v2t_env,
            "observed_envelope_v2s_fm3": v2s_env,
            "predicted_v2t_within_envelope": bool(v2t_within),
            "predicted_v2s_within_envelope": bool(v2s_within),
            "notes": [
                "Eq.(18) and Eq.(19) are two phase-shift analyses in the same primary source; their difference is treated as an analysis-dependent systematics proxy.",
                "This script fits each dataset separately to expose sensitivity of the inferred u-profile to that proxy.",
            ],
        },
        "falsification": {
            "acceptance_criteria": [
                "Under the hard-core + well ansatz class, after fitting triplet (B,a_t,r_t), the predicted v2t should be compatible with the observed range (eq18–eq19) unless a specific phase-shift analysis is justified.",
                "With geometry fixed by the triplet fit, after fitting singlet a_s, the predicted (r_s,v2s) should be compatible with observed values (within analysis-dependent envelope) or the ansatz class is rejected for representing the nuclear-scale u-profile.",
            ],
            "within_envelope": {"v2t": bool(v2t_within), "v2s": bool(v2s_within)},
        },
        "outputs": {"png": str(out_png)},
    }

    out_json = out_dir / "nuclear_effective_potential_core_well_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()

