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


def _solve_bound_x(*, kappa_fm1: float, r_fm: float) -> float:
    """
    Solve x in (pi/2, pi) for the s-wave bound-state condition:

      k cot(kR) = -kappa,  with k = x/R

    which becomes:

      x cot x + kappa R = 0.
    """
    if not (math.isfinite(kappa_fm1) and kappa_fm1 > 0 and math.isfinite(r_fm) and r_fm > 0):
        raise ValueError("invalid kappa or R")

    lo = (math.pi / 2.0) + 1e-7
    hi = math.pi - 1e-7

    def f(x: float) -> float:
        return (x / math.tan(x)) + (kappa_fm1 * r_fm)

    flo = f(lo)
    fhi = f(hi)
    if not (flo > 0 and fhi < 0):
        raise ValueError(f"no bracket for bound x: f(lo)={flo}, f(hi)={fhi}, kappaR={kappa_fm1*r_fm}")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0 or (hi - lo) < 1e-14:
            return mid
        if fmid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _square_well_from_r(*, mu_mev: float, b_mev: float, r_fm: float, hbarc_mev_fm: float) -> dict[str, float]:
    """
    Given B (fixed) and R, solve the well depth V0 by the bound-state condition.

    Returns: V0 (MeV), x (dimensionless), k (fm^-1), kappa (fm^-1).
    """
    if not (mu_mev > 0 and b_mev > 0 and r_fm > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid inputs")

    kappa = math.sqrt(2.0 * mu_mev * b_mev) / hbarc_mev_fm
    x = _solve_bound_x(kappa_fm1=kappa, r_fm=r_fm)
    k = x / r_fm
    v0 = b_mev + (hbarc_mev_fm**2) * (k**2) / (2.0 * mu_mev)
    return {"V0_mev": float(v0), "x": float(x), "k_fm1": float(k), "kappa_fm1": float(kappa)}


def _square_well_scattering_length(*, mu_mev: float, v0_mev: float, r_fm: float, hbarc_mev_fm: float) -> float:
    # Exact k->0 expression for an attractive square well.
    q0 = math.sqrt(2.0 * mu_mev * v0_mev) / hbarc_mev_fm
    if not (math.isfinite(q0) and q0 > 0):
        return float("nan")
    return float(r_fm - (math.tan(q0 * r_fm) / q0))


def _fit_square_well_to_b_and_a(
    *,
    mu_mev: float,
    b_mev: float,
    a_target_fm: float,
    hbarc_mev_fm: float,
    r_min_fm: float = 0.5,
    r_max_fm: float = 5.0,
) -> dict[str, float]:
    """
    Fit (R, V0) such that:
      - the well supports a bound state at energy -B (deuteron),
      - the triplet scattering length a_t matches a_target.

    Strategy:
      - For each R, V0 is determined by the bound-state condition.
      - Then solve a(R) = a_target by bracketing + bisection.
    """
    if not (math.isfinite(a_target_fm) and a_target_fm != 0):
        raise ValueError("invalid a_target")
    if not (r_min_fm > 0 and r_max_fm > r_min_fm):
        raise ValueError("invalid R range")

    n_scan = 2000
    r_grid = [r_min_fm + (r_max_fm - r_min_fm) * i / (n_scan - 1) for i in range(n_scan)]

    candidates: list[tuple[float, float, float]] = []
    prev_r: float | None = None
    prev_g: float | None = None

    for r in r_grid:
        try:
            bound = _square_well_from_r(mu_mev=mu_mev, b_mev=b_mev, r_fm=r, hbarc_mev_fm=hbarc_mev_fm)
            a_pred = _square_well_scattering_length(
                mu_mev=mu_mev, v0_mev=bound["V0_mev"], r_fm=r, hbarc_mev_fm=hbarc_mev_fm
            )
        except Exception:
            prev_r = None
            prev_g = None
            continue

        if not (math.isfinite(a_pred) and abs(a_pred) < 1e6):
            prev_r = None
            prev_g = None
            continue

        g = a_pred - a_target_fm
        if prev_r is not None and prev_g is not None:
            if math.isfinite(g) and math.isfinite(prev_g) and (g == 0 or (g > 0) != (prev_g > 0)):
                # Avoid brackets that include a near-divergence (huge |g|).
                if abs(g) < 1e3 and abs(prev_g) < 1e3:
                    candidates.append((prev_r, r, abs(prev_r - 2.0) + abs(r - 2.0)))
        prev_r = r
        prev_g = g

    if not candidates:
        raise ValueError("no bracket found for a(R)=a_target within scan range")

    # Prefer a "nuclear-ish" range near 2 fm to avoid jumping to high-n/near-resonant solutions.
    candidates.sort(key=lambda t: t[2])
    lo_r, hi_r, _ = candidates[0]

    def g_of_r(r: float) -> float:
        bound = _square_well_from_r(mu_mev=mu_mev, b_mev=b_mev, r_fm=r, hbarc_mev_fm=hbarc_mev_fm)
        a_pred = _square_well_scattering_length(mu_mev=mu_mev, v0_mev=bound["V0_mev"], r_fm=r, hbarc_mev_fm=hbarc_mev_fm)
        return a_pred - a_target_fm

    g_lo = g_of_r(lo_r)
    g_hi = g_of_r(hi_r)
    if not (math.isfinite(g_lo) and math.isfinite(g_hi) and (g_lo == 0 or (g_lo > 0) != (g_hi > 0))):
        raise ValueError("invalid bracket after selection")

    for _ in range(90):
        mid = 0.5 * (lo_r + hi_r)
        g_mid = g_of_r(mid)
        if g_mid == 0 or (hi_r - lo_r) < 1e-12:
            lo_r = mid
            hi_r = mid
            break
        if (g_mid > 0) == (g_lo > 0):
            lo_r = mid
            g_lo = g_mid
        else:
            hi_r = mid
            g_hi = g_mid

    r_fit = 0.5 * (lo_r + hi_r)
    bound_fit = _square_well_from_r(mu_mev=mu_mev, b_mev=b_mev, r_fm=r_fit, hbarc_mev_fm=hbarc_mev_fm)
    a_fit = _square_well_scattering_length(
        mu_mev=mu_mev, v0_mev=bound_fit["V0_mev"], r_fm=r_fit, hbarc_mev_fm=hbarc_mev_fm
    )

    return {
        "R_fm": float(r_fit),
        "V0_mev": float(bound_fit["V0_mev"]),
        "a_t_fm_target": float(a_target_fm),
        "a_t_fm_fit": float(a_fit),
        "bound_x": float(bound_fit["x"]),
        "bound_k_fm1": float(bound_fit["k_fm1"]),
        "bound_kappa_fm1": float(bound_fit["kappa_fm1"]),
    }


def _phase_shift_square_well(*, k_fm1: float, mu_mev: float, v0_mev: float, r_fm: float, hbarc_mev_fm: float) -> float:
    """
    s-wave phase shift for an attractive square well.

    Matching gives:
      tan(kR + δ) = (k/q) tan(qR),  q = sqrt(k^2 + k0^2),  k0^2 = 2μV0/(ħc)^2
    and we take the k→0 continuous branch so δ→0.
    """
    if k_fm1 == 0.0:
        return 0.0
    k0 = math.sqrt(2.0 * mu_mev * v0_mev) / hbarc_mev_fm
    q = math.sqrt(k_fm1**2 + k0**2)
    t = (k_fm1 / q) * math.tan(q * r_fm)
    delta = math.atan(t) - (k_fm1 * r_fm)
    # Wrap to a principal interval near 0 (k is small in our usage).
    while delta > math.pi / 2.0:
        delta -= math.pi
    while delta < -math.pi / 2.0:
        delta += math.pi
    return float(delta)


def _fit_effective_range_expansion(
    *, mu_mev: float, v0_mev: float, r_fm: float, hbarc_mev_fm: float
) -> dict[str, object]:
    """
    Fit k cot δ = -1/a + (r_e/2) k^2 over a low-k grid, returning (a, r_e).
    """
    # Use a genuinely low-k grid so the fitted r_eff approximates the k→0 definition.
    # (If k is too large, higher-order shape parameters bias the slope.)
    k_grid = [0.002 * i for i in range(1, 11)]  # 0.002..0.020 fm^-1
    xs: list[float] = []
    ys: list[float] = []
    points: list[dict[str, float]] = []
    for k in k_grid:
        delta = _phase_shift_square_well(k_fm1=k, mu_mev=mu_mev, v0_mev=v0_mev, r_fm=r_fm, hbarc_mev_fm=hbarc_mev_fm)
        if not math.isfinite(delta) or abs(delta) < 1e-12:
            continue
        y = k / math.tan(delta)  # k cot δ
        x = k * k
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
        points.append({"k_fm1": float(k), "k2_fm2": float(x), "kcot_fm1": float(y), "delta_rad": float(delta)})

    if len(xs) < 3:
        raise ValueError("insufficient points for ERE fit")

    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("degenerate x grid for ERE fit")
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar  # k cot δ at k->0  == -1/a

    a = -1.0 / intercept if intercept != 0 else float("nan")
    r_eff = 2.0 * slope

    # Residual RMS as a fit-quality diagnostic (not a statistical uncertainty).
    rms = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / len(xs))

    return {
        "a_fm": float(a),
        "r_eff_fm": float(r_eff),
        "intercept_kcot_fm1": float(intercept),
        "slope_half_r_fm": float(slope),
        "fit_rms_fm1": float(rms),
        "points": points,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exact SI constants (for conversion only).
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    hbarc_mev_fm = 197.326_980_4  # CODATA 2018/2022 consistent; treat as exact here.

    # Load baseline constants
    consts = _load_nist_codata_constants(root=root)
    for k in ("mp", "mn", "md"):
        if k not in consts:
            raise SystemExit(f"[fail] missing constant {k!r} in extracted_values.json")

    mp = float(consts["mp"]["value_si"])
    mn = float(consts["mn"]["value_si"])
    md = float(consts["md"]["value_si"])

    # Deuteron binding energy from mass defect (same baseline as Step 7.9.1)
    dm = (mp + mn - md)
    b_j = dm * (c**2)
    b_mev = b_j / (1e6 * e_charge)

    # Reduced mass (energy equivalent in MeV): mu_mev = μ c^2
    mu_kg = (mp * mn) / (mp + mn)
    mu_mev = (mu_kg * c**2) / (1e6 * e_charge)

    # Load np scattering parameters (eq18 / eq19)
    np_sets = _load_np_scattering_sets(root=root)
    eq18 = np_sets.get(18)
    eq19 = np_sets.get(19)
    if not (isinstance(eq18, dict) and isinstance(eq19, dict)):
        raise SystemExit("[fail] missing eq18/eq19 in extracted np scattering JSON")

    triplet_targets = [
        {
            "label": "eq18 (GWU/SAID)",
            "eq_label": 18,
            "a_t_fm": _get_value(eq18, "a_t_fm"),
            "r_t_fm": _get_value(eq18, "r_t_fm"),
        },
        {
            "label": "eq19 (Nijmegen)",
            "eq_label": 19,
            "a_t_fm": _get_value(eq19, "a_t_fm"),
            "r_t_fm": _get_value(eq19, "r_t_fm"),
        },
    ]

    fits: list[dict[str, object]] = []
    for t in triplet_targets:
        fit = _fit_square_well_to_b_and_a(
            mu_mev=mu_mev, b_mev=b_mev, a_target_fm=float(t["a_t_fm"]), hbarc_mev_fm=hbarc_mev_fm
        )
        ere = _fit_effective_range_expansion(
            mu_mev=mu_mev, v0_mev=float(fit["V0_mev"]), r_fm=float(fit["R_fm"]), hbarc_mev_fm=hbarc_mev_fm
        )
        u0 = float(fit["V0_mev"]) / float(mu_mev) if mu_mev > 0 else float("nan")
        t_out = {
            "label": str(t["label"]),
            "eq_label": int(t["eq_label"]),
            "inputs": {"B_MeV": float(b_mev), "a_t_fm": float(t["a_t_fm"]), "r_t_fm": float(t["r_t_fm"])},
            "fit_square_well": fit,
            "ere_from_phase_shift": ere,
            "p_profile_mapping": {
                "u0": u0,
                "P_over_P0_attractive_core": math.exp(u0) if math.isfinite(u0) else None,
                "notes": [
                    "Square well uses V(r)=-V0 for r<R, 0 otherwise.",
                    "With the minimal Part III mapping V=μ φ and u=-φ/c^2, the well corresponds to u≈+V0/(μc^2) and thus P/P0=exp(u)>1 inside the attractive region.",
                    "This does not explain *why* such a u-profile arises (that is the open first-principles problem); it only freezes an effective constraint from low-energy data.",
                ],
            },
            "comparison": {
                "a_t_fit_minus_target_fm": float(fit["a_t_fm_fit"]) - float(t["a_t_fm"]),
                "r_eff_minus_observed_fm": float(ere["r_eff_fm"]) - float(t["r_t_fm"]),
            },
        }
        fits.append(t_out)

    # Envelope check: r_t should be within [min(eq18,eq19), max(eq18,eq19)] for this simplest ansatz.
    r_obs = [float(t["r_t_fm"]) for t in triplet_targets]
    r_min = min(r_obs)
    r_max = max(r_obs)
    r_pred = [float(f["ere_from_phase_shift"]["r_eff_fm"]) for f in fits]
    within_envelope = all((r_min <= r <= r_max) for r in r_pred if math.isfinite(r))

    # Plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.6, 7.4), dpi=160, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.28)

    # (A) Potential profiles
    ax0 = fig.add_subplot(gs[0, 0])
    r_plot = [i * 0.02 for i in range(0, 501)]  # 0..10 fm
    colors = ["tab:blue", "tab:orange"]
    for f, col in zip(fits, colors):
        r0 = float(f["fit_square_well"]["R_fm"])
        v0 = float(f["fit_square_well"]["V0_mev"])
        v = [(-v0 if rr <= r0 else 0.0) for rr in r_plot]
        ax0.plot(r_plot, v, lw=2.2, color=col, label=f["label"])
        ax0.axvline(r0, color=col, lw=1.0, ls=":", alpha=0.7)
    ax0.set_xlabel("r (fm)")
    ax0.set_ylabel("V(r) (MeV)")
    ax0.set_title("Phenomenological u-profile as a square-well potential (triplet)")
    ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax0.legend(frameon=True, fontsize=9, loc="lower right")
    ax0.text(
        0.02,
        0.98,
        f"Fixed B≈{b_mev:.6f} MeV (CODATA)\nFit: (R,V0) from B + a_t",
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )

    # (B) ERE fit (use first fit as representative)
    ax1 = fig.add_subplot(gs[0, 1])
    ere0 = fits[0]["ere_from_phase_shift"]
    pts = ere0["points"]
    xs = [p["k2_fm2"] for p in pts]
    ys = [p["kcot_fm1"] for p in pts]
    ax1.plot(xs, ys, "o", ms=4.5, label="k-grid")
    x_line = [0.0, max(xs)]
    a0 = float(ere0["intercept_kcot_fm1"])
    a1 = float(ere0["slope_half_r_fm"])
    y_line = [a0 + a1 * x for x in x_line]
    ax1.plot(x_line, y_line, "-", lw=2.0, color="0.35", label="linear fit")
    ax1.set_xlabel("k² (fm⁻²)")
    ax1.set_ylabel("k cot δ (fm⁻¹)")
    ax1.set_title(f"Effective-range expansion fit ({fits[0]['label']})")
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax1.legend(frameon=True, fontsize=9, loc="best")
    ax1.text(
        0.02,
        0.02,
        f"a≈{float(ere0['a_fm']):.4f} fm\nr_t(pred)≈{float(ere0['r_eff_fm']):.4f} fm\nfit rms≈{float(ere0['fit_rms_fm1']):.2e} fm⁻¹",
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )

    # (C) r_t predicted vs observed
    ax2 = fig.add_subplot(gs[1, 0])
    labels = [f["label"] for f in fits]
    y_obs = [f["inputs"]["r_t_fm"] for f in fits]
    y_pred = [float(f["ere_from_phase_shift"]["r_eff_fm"]) for f in fits]
    x = list(range(len(labels)))
    ax2.plot(x, y_obs, "o", ms=7, label="observed r_t (source)")
    ax2.plot(x, y_pred, "s", ms=7, label="predicted r_t (square well)")
    ax2.fill_between([-0.5, len(labels) - 0.5], [r_min, r_min], [r_max, r_max], color="0.9", alpha=0.8, label="obs envelope (eq18–eq19)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("r_t (fm)")
    ax2.set_title("Triplet effective range: observed vs predicted")
    ax2.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax2.legend(frameon=True, fontsize=9, loc="best")

    # (D) P/P0 mapping summary
    ax3 = fig.add_subplot(gs[1, 1])
    u0s = [float(f["p_profile_mapping"]["u0"]) for f in fits]
    p_ratios = [float(f["p_profile_mapping"]["P_over_P0_attractive_core"]) for f in fits]
    ax3.bar(x, p_ratios, color=colors, alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha="right")
    ax3.set_ylabel("P/P0 at r< R (from V0/μc²)")
    ax3.set_title("Mapping back to P-model variable u=ln(P/P0)")
    ax3.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    for i, (u0, pr) in enumerate(zip(u0s, p_ratios)):
        ax3.text(i, pr, f"u0≈{u0:.3e}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Phase 7 / Step 7.9.3: nuclear effective equation (square-well ansatz) — constraints frozen", y=1.02)

    out_png = out_dir / "nuclear_effective_potential_square_well.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Sources / traceability
    codata_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    codata_manifest = codata_dir / "manifest.json"
    np_manifest = root / "data" / "quantum" / "sources" / "np_scattering_low_energy_arxiv_0704_1024v1_manifest.json"
    np_extracted = root / "data" / "quantum" / "sources" / "np_scattering_low_energy_arxiv_0704_1024v1_extracted.json"

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.9.3",
        "model": {
            "effective_equation": "Nonrelativistic s-wave Schrödinger equation for relative motion with an effective potential V(r)=μ φ(r)=-μ c^2 u(r).",
            "ansatz": "Short-range u-profile represented as a square-well potential (two parameters: range R, depth V0).",
            "fit_constraints": ["Deuteron binding energy B (CODATA mass defect)", "Triplet scattering length a_t (np low-energy parameters)"],
            "prediction_targets": ["Triplet effective range r_t (effective-range expansion)"],
            "positioning": [
                "This is a phenomenological constraint (effective model), not a first-principles derivation of nuclear forces.",
                "The goal is to freeze a minimal, reproducible bridge from the P-variable u to low-energy nuclear observables and expose falsification conditions for simple ansatz classes.",
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
        "constants": {
            "hbarc_MeV_fm": hbarc_mev_fm,
            "mu_c2_MeV": float(mu_mev),
            "B_MeV": float(b_mev),
        },
        "fits": fits,
        "falsification": {
            "acceptance_criteria": [
                "Under the simplest short-range ansatz class (square well u-profile; 2 parameters), fitting (B, a_t) should predict r_t within the observed envelope (eq18–eq19). If not, this ansatz class is rejected as a candidate for the nuclear-scale u-profile.",
                "Future step (7.9.4+): extend to include singlet channel and/or shape parameters (v2) without losing predictive power (avoid 'arbitrary function fit').",
            ],
            "observed_envelope_triplet_r_t_fm": {"min": r_min, "max": r_max},
            "predicted_within_envelope": within_envelope,
        },
        "outputs": {"png": str(out_png)},
        "notes": [
            "This script intentionally does not introduce a new fundamental coupling constant; it freezes an effective constraint on u via a minimal ansatz.",
            "If the square well passes, it does not prove P-model; it only shows that a very low-parameter u-profile can satisfy the deuteron+np low-energy constraints.",
            "Next (7.9.3+): decide and fix what 'first-principles' claim is attempted (e.g., source term/nonlinearity in the u field equation that yields such u-profiles).",
        ],
    }

    out_json = out_dir / "nuclear_effective_potential_square_well_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()
