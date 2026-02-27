from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。
def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `chosen` を満たす経路を評価する。
        if chosen:
            mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]

        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["font.size"] = 13.0
        mpl.rcParams["axes.titlesize"] = 17.0
        mpl.rcParams["axes.labelsize"] = 14.0
        mpl.rcParams["xtick.labelsize"] = 12.0
        mpl.rcParams["ytick.labelsize"] = 12.0
        mpl.rcParams["legend.fontsize"] = 12.0
    except Exception:
        return


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_to_float` の入出力契約と処理意図を定義する。

def _to_float(v: Any) -> Optional[float]:
    try:
        val = float(v)
    except Exception:
        return None

    # 条件分岐: `math.isnan(val) or math.isinf(val)` を満たす経路を評価する。

    if math.isnan(val) or math.isinf(val):
        return None

    return val


# 関数: `_fmt_float` の入出力契約と処理意図を定義する。

def _fmt_float(x: float, digits: int = 6) -> str:
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# 関数: `_pretty_pulsar_name` の入出力契約と処理意図を定義する。
def _pretty_pulsar_name(psr_id: str) -> str:
    alias = {
        "psr_b1913p16": "B1913+16\n(Hulse-Taylor)",
        "psr_j0737m3039ab": "J0737-3039A/B\n(Double Pulsar)",
        "psr_j1738p0333": "J1738+0333",
        "psr_j0348p0432": "J0348+0432",
    }
    key = (psr_id or "").strip().lower()
    return alias.get(key, psr_id)


# 関数: `_extract_rows` の入出力契約と処理意図を定義する。

def _extract_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in payload.get("metrics") or []:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        derived = row.get("derived") if isinstance(row.get("derived"), dict) else {}
        m1 = _to_float(derived.get("m1_msun"))
        m2 = _to_float(derived.get("m2_msun"))
        # 条件分岐: `m1 is None or m2 is None or m1 <= 0.0 or m2 <= 0.0` を満たす経路を評価する。
        if m1 is None or m2 is None or m1 <= 0.0 or m2 <= 0.0:
            continue

        delta = _to_float(row.get("delta"))
        sigma_1 = _to_float(row.get("sigma_1"))
        sigma_95 = _to_float(row.get("sigma_95"))
        upper_95 = _to_float(row.get("non_gr_fraction_upper_95"))
        # 条件分岐: `upper_95 is None and delta is not None and sigma_95 is not None` を満たす経路を評価する。
        if upper_95 is None and delta is not None and sigma_95 is not None:
            upper_95 = abs(delta) + sigma_95

        # 条件分岐: `upper_95 is None and delta is not None and sigma_1 is not None` を満たす経路を評価する。

        if upper_95 is None and delta is not None and sigma_1 is not None:
            upper_95 = abs(delta) + 1.96 * sigma_1

        # 条件分岐: `upper_95 is None` を満たす経路を評価する。

        if upper_95 is None:
            upper_95 = float("nan")

        z = None
        # 条件分岐: `delta is not None and sigma_1 is not None and sigma_1 > 0.0` を満たす経路を評価する。
        if delta is not None and sigma_1 is not None and sigma_1 > 0.0:
            z = delta / sigma_1

        out.append(
            {
                "id": str(row.get("id") or ""),
                "name": str(row.get("name") or ""),
                "m1_msun": m1,
                "m2_msun": m2,
                "delta_R": delta,
                "sigma_1_R": sigma_1,
                "sigma_95_R": sigma_95,
                "z_R": z,
                "non_gr_fraction_upper_95": upper_95,
            }
        )

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise SystemExit("no usable binary rows found in input metrics")

    return out


# 関数: `_dipole_norm` の入出力契約と処理意図を定義する。

def _dipole_norm(m1: float, m2: float, eta1: float, eta2: float) -> float:
    total = m1 + m2
    r1 = m2 / total
    r2 = -m1 / total
    d = eta1 * m1 * r1 + eta2 * m2 * r2
    denom = max(1e-30, abs(total) * max(abs(eta1), abs(eta2)))
    return abs(d) / denom


# 関数: `_build_audit_rows` の入出力契約と処理意図を定義する。

def _build_audit_rows(
    rows: Sequence[Dict[str, Any]],
    eta0: float,
    epsilon_counterfactual: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        m1 = float(row["m1_msun"])
        m2 = float(row["m2_msun"])
        dipole_universal = _dipole_norm(m1, m2, eta0, eta0)
        dipole_counterfactual = _dipole_norm(m1, m2, eta0, eta0 * (1.0 + epsilon_counterfactual))
        upper_95 = _to_float(row.get("non_gr_fraction_upper_95"))
        eps_lim = None
        # 条件分岐: `upper_95 is not None and upper_95 > 0.0` を満たす経路を評価する。
        if upper_95 is not None and upper_95 > 0.0:
            eps_lim = math.sqrt(upper_95)

        out.append(
            {
                **row,
                "dipole_norm_universal": dipole_universal,
                "dipole_norm_counterfactual": dipole_counterfactual,
                "epsilon_counterfactual": epsilon_counterfactual,
                "epsilon_limit_95": eps_lim,
            }
        )

    return out


# 関数: `_gate_payload` の入出力契約と処理意図を定義する。

def _gate_payload(rows: Sequence[Dict[str, Any]], z_reject: float, dipole_tol: float, epsilon_gate: float) -> Dict[str, Any]:
    max_dipole_norm = max(float(r.get("dipole_norm_universal") or 0.0) for r in rows)
    z_values = [abs(float(z)) for z in (r.get("z_R") for r in rows) if z is not None]
    max_abs_z = max(z_values) if z_values else float("nan")
    eps_limits = [float(e) for e in (r.get("epsilon_limit_95") for r in rows) if e is not None and np.isfinite(float(e))]
    eps_limit_min = min(eps_limits) if eps_limits else float("nan")

    gate1 = {
        "id": "dipole::equivalence_principle_cancellation",
        "metric": "max_dipole_norm_universal",
        "value": max_dipole_norm,
        "threshold": dipole_tol,
        "comparator": "<=",
        "status": "pass" if max_dipole_norm <= dipole_tol else "reject",
        "hardness": "hard",
        "note": "eta_i/m_i の普遍性を課すと COM 系で双極子モーメントが相殺されること。",
    }
    gate2 = {
        "id": "dipole::quadrupole_leading_consistency",
        "metric": "max_abs_z_R_assuming_Req1",
        "value": max_abs_z,
        "threshold": z_reject,
        "comparator": "<=",
        "status": "pass" if np.isfinite(max_abs_z) and max_abs_z <= z_reject else "reject",
        "hardness": "hard",
        "note": "双極子ゼロ（R=1）仮定で連星パルサー観測が 3σ 以内に収まること。",
    }
    gate3 = {
        "id": "dipole::universality_margin_95",
        "metric": "min_epsilon_limit_95",
        "value": eps_limit_min,
        "threshold": epsilon_gate,
        "comparator": "<=",
        "status": "pass" if np.isfinite(eps_limit_min) and eps_limit_min <= epsilon_gate else "watch",
        "hardness": "watch",
        "note": "95%上限から逆算した eta 非普遍性許容幅 sqrt(non_gr_fraction_upper_95)。",
    }
    gates = [gate1, gate2, gate3]
    n_reject = sum(1 for g in gates if g["status"] == "reject")
    n_watch = sum(1 for g in gates if g["status"] == "watch")
    overall = "pass" if n_reject == 0 else "reject"
    return {
        "overall_status": overall,
        "decision": "dipole_cancelled_quadrupole_leading" if overall == "pass" else "dipole_cancellation_not_confirmed",
        "hard_reject_n": n_reject,
        "watch_n": n_watch,
        "gates": gates,
        "max_dipole_norm_universal": max_dipole_norm,
        "max_abs_z_R_assuming_Req1": max_abs_z,
        "min_epsilon_limit_95": eps_limit_min,
    }


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(rows: Sequence[Dict[str, Any]], z_reject: float, out_png: Path) -> None:
    _set_japanese_font()
    labels_raw = [str(r.get("id") or f"row{i+1}") for i, r in enumerate(rows)]
    labels = [_pretty_pulsar_name(label) for label in labels_raw]
    x = np.arange(len(rows), dtype=float)

    z = np.array([float(r["z_R"]) if r.get("z_R") is not None else np.nan for r in rows], dtype=float)
    dip_u = np.array([float(r.get("dipole_norm_universal") or 0.0) for r in rows], dtype=float)
    dip_c = np.array([float(r.get("dipole_norm_counterfactual") or 0.0) for r in rows], dtype=float)
    eps_lim = np.array([float(r["epsilon_limit_95"]) if r.get("epsilon_limit_95") is not None else np.nan for r in rows], dtype=float)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14.8, 13.9), sharex=True)
    fig.suptitle("P_μ minimal extension: dipole-cancellation audit (equivalence-principle condition)", fontsize=19.0)

    ax0.bar(x, np.nan_to_num(z, nan=0.0), color="#1f77b4", alpha=0.9)
    ax0.axhline(z_reject, color="#333333", linestyle="--", linewidth=1.0)
    ax0.axhline(-z_reject, color="#333333", linestyle="--", linewidth=1.0)
    ax0.axhline(0.0, color="#666666", linestyle="-", linewidth=0.9)
    x_label = (x[-1] + 0.45) if len(x) > 0 else 0.45
    ax0.text(
        x_label,
        z_reject,
        "3σ (Reject)",
        ha="left",
        va="bottom",
        fontsize=11.0,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
        clip_on=False,
    )
    ax0.text(
        x_label,
        -z_reject,
        "-3σ (Reject)",
        ha="left",
        va="top",
        fontsize=11.0,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
        clip_on=False,
    )
    ax0.set_ylabel("z = (R-1)/σ", fontsize=15.0)
    ax0.set_title("Binary-pulsar consistency under dipole=0 (quadrupole-leading)", fontsize=17.8, pad=10.0)
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.tick_params(axis="both", labelsize=12.5)
    max_abs_z = float(np.nanmax(np.abs(np.nan_to_num(z, nan=0.0)))) if len(z) else 1.0
    for xi, zi, label in zip(x, np.nan_to_num(z, nan=0.0), labels):
        y_offset = 0.12 + 0.03 * max_abs_z
        y_text = zi + y_offset if zi >= 0.0 else zi - y_offset
        va_text = "bottom" if zi >= 0.0 else "top"
        ax0.text(xi, y_text, label.split("\n")[0], ha="center", va=va_text, fontsize=10.5, color="0.22")

    width = 0.36
    ax1.bar(x - width / 2.0, dip_u, width=width, color="#2ca02c", alpha=0.9, label="universal η (expected ~0)")
    ax1.bar(x + width / 2.0, dip_c, width=width, color="#ff7f0e", alpha=0.9, label="counterfactual η2=η1(1+ε)")
    ax1.set_yscale("log")
    ax1.set_ylabel("|d|/(η M r)", fontsize=15.0)
    ax1.set_title("Dipole moment normalization", fontsize=17.8, pad=10.0)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.tick_params(axis="both", labelsize=12.5)
    ax1.legend(loc="upper right")
    inset = ax1.inset_axes([0.64, 0.12, 0.33, 0.40])
    inset.bar(x, dip_u, width=0.42, color="#2ca02c", alpha=0.95)
    inset.set_yscale("linear")
    inset.set_xticks([])
    inset.set_title("zoom: universal η", fontsize=10.0)
    max_u = float(np.max(dip_u)) if len(dip_u) else 0.0
    inset.set_ylim(0.0, max(1e-18, 1.15 * max_u))
    inset.grid(True, axis="y", alpha=0.20)

    ax2.bar(x, np.nan_to_num(eps_lim, nan=0.0), color="#9467bd", alpha=0.9)
    ax2.set_ylabel("sqrt(non-GR upper95)", fontsize=15.0)
    ax2.set_title("Inferred universality tolerance (95%)", fontsize=17.8, pad=10.0)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.tick_params(axis="both", labelsize=12.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=8, ha="right", fontsize=12.0)
    ax2.set_xlabel("binary pulsar systems", fontsize=14.0)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "name",
                "m1_msun",
                "m2_msun",
                "delta_R",
                "sigma_1_R",
                "sigma_95_R",
                "z_R",
                "non_gr_fraction_upper_95",
                "dipole_norm_universal",
                "dipole_norm_counterfactual",
                "epsilon_counterfactual",
                "epsilon_limit_95",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.get("id", ""),
                    r.get("name", ""),
                    _fmt_float(float(r["m1_msun"]), 7),
                    _fmt_float(float(r["m2_msun"]), 7),
                    "" if r.get("delta_R") is None else _fmt_float(float(r["delta_R"]), 8),
                    "" if r.get("sigma_1_R") is None else _fmt_float(float(r["sigma_1_R"]), 8),
                    "" if r.get("sigma_95_R") is None else _fmt_float(float(r["sigma_95_R"]), 8),
                    "" if r.get("z_R") is None else _fmt_float(float(r["z_R"]), 8),
                    "" if r.get("non_gr_fraction_upper_95") is None else _fmt_float(float(r["non_gr_fraction_upper_95"]), 8),
                    _fmt_float(float(r["dipole_norm_universal"]), 10),
                    _fmt_float(float(r["dipole_norm_counterfactual"]), 10),
                    _fmt_float(float(r["epsilon_counterfactual"]), 8),
                    "" if r.get("epsilon_limit_95") is None else _fmt_float(float(r["epsilon_limit_95"]), 8),
                ]
            )


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _ROOT
    default_in_public = root / "output" / "public" / "pulsar" / "binary_pulsar_orbital_decay_metrics.json"
    default_in_private = root / "output" / "private" / "pulsar" / "binary_pulsar_orbital_decay_metrics.json"
    default_outdir = root / "output" / "private" / "theory"
    default_public_outdir = root / "output" / "public" / "theory"
    default_canon_outdir = root / "output" / "theory"

    ap = argparse.ArgumentParser(description="Dipole-radiation cancellation audit under equivalence-principle condition.")
    ap.add_argument("--in-json", type=str, default=str(default_in_public), help="Binary pulsar metrics JSON path.")
    ap.add_argument(
        "--in-json-fallback",
        type=str,
        default=str(default_in_private),
        help="Fallback binary pulsar metrics JSON path.",
    )
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output directory.")
    ap.add_argument("--public-outdir", type=str, default=str(default_public_outdir), help="Public output directory.")
    ap.add_argument("--canon-outdir", type=str, default=str(default_canon_outdir), help="Canonical output directory.")
    ap.add_argument("--eta0", type=float, default=1.0, help="Universal coupling ratio baseline.")
    ap.add_argument(
        "--epsilon-counterfactual",
        type=float,
        default=1.0e-3,
        help="Counterfactual non-universality applied to companion 2: eta2=eta1*(1+epsilon).",
    )
    ap.add_argument("--z-reject", type=float, default=3.0, help="Reject gate on |z| for R consistency.")
    ap.add_argument(
        "--dipole-tol",
        type=float,
        default=1.0e-12,
        help="Tolerance for max normalized dipole moment in universal branch.",
    )
    ap.add_argument(
        "--epsilon-gate",
        type=float,
        default=2.0e-2,
        help="Watch gate for inferred universality tolerance sqrt(non_gr_fraction_upper_95).",
    )
    ap.add_argument("--no-public-copy", action="store_true", help="Do not copy outputs to public directory.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_json = Path(args.in_json)
    # 条件分岐: `not in_json.exists()` を満たす経路を評価する。
    if not in_json.exists():
        in_json = Path(args.in_json_fallback)

    # 条件分岐: `not in_json.exists()` を満たす経路を評価する。

    if not in_json.exists():
        raise SystemExit(f"input not found: {args.in_json} (fallback: {args.in_json_fallback})")

    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    canon_outdir = Path(args.canon_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)
    canon_outdir.mkdir(parents=True, exist_ok=True)

    payload = _read_json(in_json)
    rows = _extract_rows(payload)
    audit_rows = _build_audit_rows(rows, eta0=float(args.eta0), epsilon_counterfactual=float(args.epsilon_counterfactual))
    summary = _gate_payload(
        audit_rows,
        z_reject=float(args.z_reject),
        dipole_tol=float(args.dipole_tol),
        epsilon_gate=float(args.epsilon_gate),
    )

    out_json = outdir / "pmodel_vector_dipole_cancellation_audit.json"
    out_csv = outdir / "pmodel_vector_dipole_cancellation_audit.csv"
    out_png = outdir / "pmodel_vector_dipole_cancellation_audit.png"

    _plot(audit_rows, z_reject=float(args.z_reject), out_png=out_png)
    _write_csv(out_csv, audit_rows)

    payload_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "wavep.theory.pmodel_vector_dipole_cancellation_audit.v1",
        "title": "P_μ minimal extension audit: dipole cancellation under equivalence principle",
        "intent": (
            "Show that universal time-wave source-to-inertial-mass ratio cancels dipole radiation in COM frame, "
            "leaving quadrupole as the leading binary-radiation term."
        ),
        "equations": {
            "interaction": "L_int = g * P_mu * J^mu",
            "dipole_moment": "d = sum_a(q_a r_a) = sum_a(eta m_a r_a), eta=q/m",
            "cancellation_condition": "eta_a=eta (universal) => d = eta * M * R_cm = 0 (COM frame)",
            "radiation_scaling": "P_dipole ∝ |ddot(d)|^2 => 0 under universality; leading term is quadrupole",
        },
        "inputs": {
            "binary_pulsar_metrics_json": str(in_json).replace("\\", "/"),
            "eta0": float(args.eta0),
            "epsilon_counterfactual": float(args.epsilon_counterfactual),
        },
        "gate": {
            "z_reject": float(args.z_reject),
            "dipole_tol": float(args.dipole_tol),
            "epsilon_gate": float(args.epsilon_gate),
        },
        "rows": audit_rows,
        "summary": summary,
        "outputs": {
            "rows_json": str(out_json).replace("\\", "/"),
            "rows_csv": str(out_csv).replace("\\", "/"),
            "plot_png": str(out_png).replace("\\", "/"),
        },
    }
    _write_json(out_json, payload_out)

    copied: List[Path] = []
    canon_copied: List[Path] = []
    for src in (out_json, out_csv, out_png):
        dst = canon_outdir / src.name
        # 条件分岐: `src.resolve() == dst.resolve()` を満たす経路を評価する。
        if src.resolve() == dst.resolve():
            continue

        shutil.copy2(src, dst)
        canon_copied.append(dst)

    # 条件分岐: `not args.no_public_copy` を満たす経路を評価する。
    if not args.no_public_copy:
        for src in (out_json, out_csv, out_png):
            dst = public_outdir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)

    try:
        worklog.append_event(
            {
                "event_type": "theory_pmodel_vector_dipole_cancellation_audit",
                "argv": sys.argv,
                "inputs": {"binary_pulsar_metrics_json": in_json},
                "outputs": {
                    "rows_json": out_json,
                    "rows_csv": out_csv,
                    "plot_png": out_png,
                    "canon_copies": canon_copied,
                    "public_copies": copied,
                },
                "metrics": {
                    "overall_status": summary.get("overall_status"),
                    "decision": summary.get("decision"),
                    "hard_reject_n": summary.get("hard_reject_n"),
                    "watch_n": summary.get("watch_n"),
                    "max_dipole_norm_universal": summary.get("max_dipole_norm_universal"),
                    "max_abs_z_R_assuming_Req1": summary.get("max_abs_z_R_assuming_Req1"),
                    "min_epsilon_limit_95": summary.get("min_epsilon_limit_95"),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json : {out_json}")
    print(f"[ok] csv  : {out_csv}")
    print(f"[ok] png  : {out_png}")
    # 条件分岐: `canon_copied` を満たす経路を評価する。
    if canon_copied:
        print(f"[ok] canon copies : {len(canon_copied)} files -> {canon_outdir}")

    # 条件分岐: `copied` を満たす経路を評価する。
    if copied:
        print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")

    print(f"[ok] overall_status={summary.get('overall_status')} decision={summary.get('decision')}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
