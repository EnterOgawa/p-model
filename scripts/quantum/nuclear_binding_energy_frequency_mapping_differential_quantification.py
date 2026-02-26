from __future__ import annotations

import csv
import json
import math
from pathlib import Path


# 関数: `_sha256` の入出力契約と処理意図を定義する。
def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


# 関数: `_percentile` の入出力契約と処理意図を定義する。

def _percentile(sorted_vals: list[float], p: float) -> float:
    # 条件分岐: `not sorted_vals` を満たす経路を評価する。
    if not sorted_vals:
        raise ValueError("empty")

    # 条件分岐: `p <= 0` を満たす経路を評価する。

    if p <= 0:
        return float(sorted_vals[0])

    # 条件分岐: `p >= 100` を満たす経路を評価する。

    if p >= 100:
        return float(sorted_vals[-1])

    x = (len(sorted_vals) - 1) * (p / 100.0)
    i0 = int(math.floor(x))
    i1 = int(math.ceil(x))
    # 条件分岐: `i0 == i1` を満たす経路を評価する。
    if i0 == i1:
        return float(sorted_vals[i0])

    w = x - i0
    return float((1.0 - w) * sorted_vals[i0] + w * sorted_vals[i1])


# 関数: `_stats` の入出力契約と処理意図を定義する。

def _stats(vals: list[float]) -> dict[str, float]:
    # 条件分岐: `not vals` を満たす経路を評価する。
    if not vals:
        return {"n": 0.0, "median": float("nan"), "p16": float("nan"), "p84": float("nan"), "max": float("nan")}

    vs = sorted(vals)
    return {
        "n": float(len(vs)),
        "median": _percentile(vs, 50),
        "p16": _percentile(vs, 16),
        "p84": _percentile(vs, 84),
        "max": float(vs[-1]),
    }


# 関数: `_a_group` の入出力契約と処理意図を定義する。

def _a_group(a: int) -> str:
    # 条件分岐: `a <= 40` を満たす経路を評価する。
    if a <= 40:
        return "light_A_le_40"

    # 条件分岐: `a <= 120` を満たす経路を評価する。

    if a <= 120:
        return "mid_A_41_120"

    return "heavy_A_ge_121"


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv = out_dir / "nuclear_binding_energy_frequency_mapping_theory_diff.csv"
    # 条件分岐: `not in_csv.exists()` を満たす経路を評価する。
    if not in_csv.exists():
        raise SystemExit(
            "[fail] missing theory-diff CSV.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_theory_diff.py\n"
            f"Expected: {in_csv}"
        )

    rows_out: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []

    abs_delta_semf_all: list[float] = []
    abs_delta_yuk_all: list[float] = []
    req_rel_semf_all: list[float] = []
    req_rel_yuk_all: list[float] = []

    by_group: dict[str, dict[str, list[float]]] = {
        "light_A_le_40": {"abs_delta_semf": [], "abs_delta_yukawa": [], "req_rel_semf": [], "req_rel_yukawa": []},
        "mid_A_41_120": {"abs_delta_semf": [], "abs_delta_yukawa": [], "req_rel_semf": [], "req_rel_yukawa": []},
        "heavy_A_ge_121": {"abs_delta_semf": [], "abs_delta_yukawa": [], "req_rel_semf": [], "req_rel_yukawa": []},
    }

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                z = int(row["Z"])
                n = int(row["N"])
                a = int(row["A"])
                symbol = str(row.get("symbol", "")).strip()
                b_obs = float(row["B_obs_MeV"])
                sigma_obs = float(row.get("sigma_B_obs_MeV", "nan"))
                delta_semf = float(row["Delta_B_P_minus_SEMF_MeV"])
                delta_yuk = float(row["Delta_B_P_minus_Yukawa_MeV"])
            except Exception:
                continue

            # 条件分岐: `a <= 1 or not (math.isfinite(b_obs) and b_obs > 0)` を満たす経路を評価する。

            if a <= 1 or not (math.isfinite(b_obs) and b_obs > 0):
                continue

            abs_semf = abs(delta_semf) if math.isfinite(delta_semf) else float("nan")
            abs_yuk = abs(delta_yuk) if math.isfinite(delta_yuk) else float("nan")
            req_semf = (abs_semf / 3.0) if math.isfinite(abs_semf) else float("nan")
            req_yuk = (abs_yuk / 3.0) if math.isfinite(abs_yuk) else float("nan")
            req_rel_semf = (req_semf / b_obs) if math.isfinite(req_semf) else float("nan")
            req_rel_yuk = (req_yuk / b_obs) if math.isfinite(req_yuk) else float("nan")

            resolvable_semf = bool(math.isfinite(sigma_obs) and math.isfinite(req_semf) and sigma_obs <= req_semf)
            resolvable_yuk = bool(math.isfinite(sigma_obs) and math.isfinite(req_yuk) and sigma_obs <= req_yuk)
            improve_semf = (sigma_obs / req_semf) if (math.isfinite(sigma_obs) and math.isfinite(req_semf) and req_semf > 0) else float("nan")
            improve_yuk = (sigma_obs / req_yuk) if (math.isfinite(sigma_obs) and math.isfinite(req_yuk) and req_yuk > 0) else float("nan")
            a_grp = _a_group(a)

            # 条件分岐: `math.isfinite(abs_semf)` を満たす経路を評価する。
            if math.isfinite(abs_semf):
                abs_delta_semf_all.append(abs_semf)
                by_group[a_grp]["abs_delta_semf"].append(abs_semf)

            # 条件分岐: `math.isfinite(abs_yuk)` を満たす経路を評価する。

            if math.isfinite(abs_yuk):
                abs_delta_yuk_all.append(abs_yuk)
                by_group[a_grp]["abs_delta_yukawa"].append(abs_yuk)

            # 条件分岐: `math.isfinite(req_rel_semf)` を満たす経路を評価する。

            if math.isfinite(req_rel_semf):
                req_rel_semf_all.append(req_rel_semf)
                by_group[a_grp]["req_rel_semf"].append(req_rel_semf)

            # 条件分岐: `math.isfinite(req_rel_yuk)` を満たす経路を評価する。

            if math.isfinite(req_rel_yuk):
                req_rel_yuk_all.append(req_rel_yuk)
                by_group[a_grp]["req_rel_yukawa"].append(req_rel_yuk)

            rec = {
                "Z": z,
                "N": n,
                "A": a,
                "symbol": symbol,
                "A_group": a_grp,
                "B_obs_MeV": f"{b_obs:.6f}",
                "sigma_B_obs_MeV": f"{sigma_obs:.8g}" if math.isfinite(sigma_obs) else "",
                "Delta_B_P_minus_SEMF_MeV": f"{delta_semf:.6f}" if math.isfinite(delta_semf) else "",
                "Delta_B_P_minus_Yukawa_MeV": f"{delta_yuk:.6f}" if math.isfinite(delta_yuk) else "",
                "abs_Delta_B_P_minus_SEMF_MeV": f"{abs_semf:.6f}" if math.isfinite(abs_semf) else "",
                "abs_Delta_B_P_minus_Yukawa_MeV": f"{abs_yuk:.6f}" if math.isfinite(abs_yuk) else "",
                "required_sigma_3sigma_SEMF_MeV": f"{req_semf:.8g}" if math.isfinite(req_semf) else "",
                "required_sigma_3sigma_Yukawa_MeV": f"{req_yuk:.8g}" if math.isfinite(req_yuk) else "",
                "required_relative_precision_SEMF": f"{req_rel_semf:.8g}" if math.isfinite(req_rel_semf) else "",
                "required_relative_precision_Yukawa": f"{req_rel_yuk:.8g}" if math.isfinite(req_rel_yuk) else "",
                "resolvable_now_SEMF_sigma_le_req": "1" if resolvable_semf else "0",
                "resolvable_now_Yukawa_sigma_le_req": "1" if resolvable_yuk else "0",
                "precision_ratio_obs_over_req_SEMF": f"{improve_semf:.8g}" if math.isfinite(improve_semf) else "",
                "precision_ratio_obs_over_req_Yukawa": f"{improve_yuk:.8g}" if math.isfinite(improve_yuk) else "",
            }
            rows_out.append(rec)

    # 条件分岐: `not rows_out` を満たす経路を評価する。

    if not rows_out:
        raise SystemExit(f"[fail] parsed 0 usable rows from: {in_csv}")

    # Top list for reviewer-facing priorities.

    sorted_semf = sorted(
        [r for r in rows_out if r["abs_Delta_B_P_minus_SEMF_MeV"] != ""],
        key=lambda r: float(r["abs_Delta_B_P_minus_SEMF_MeV"]),
        reverse=True,
    )
    sorted_yuk = sorted(
        [r for r in rows_out if r["abs_Delta_B_P_minus_Yukawa_MeV"] != ""],
        key=lambda r: float(r["abs_Delta_B_P_minus_Yukawa_MeV"]),
        reverse=True,
    )
    for rank, row in enumerate(sorted_semf[:20], start=1):
        top_rows.append({"rank": rank, "channel": "P_minus_SEMF", **row})

    for rank, row in enumerate(sorted_yuk[:20], start=1):
        top_rows.append({"rank": rank, "channel": "P_minus_Yukawa", **row})

    out_csv = out_dir / "nuclear_binding_energy_frequency_mapping_differential_quantification.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    out_top = out_dir / "nuclear_binding_energy_frequency_mapping_differential_quantification_top20.csv"
    # 条件分岐: `top_rows` を満たす経路を評価する。
    if top_rows:
        with out_top.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
            writer.writeheader()
            writer.writerows(top_rows)

    # Plot

    import matplotlib.pyplot as plt

    a_vals = [int(r["A"]) for r in rows_out]
    d_semf = [float(r["abs_Delta_B_P_minus_SEMF_MeV"]) if r["abs_Delta_B_P_minus_SEMF_MeV"] else float("nan") for r in rows_out]
    d_yuk = [float(r["abs_Delta_B_P_minus_Yukawa_MeV"]) if r["abs_Delta_B_P_minus_Yukawa_MeV"] else float("nan") for r in rows_out]
    rel_semf = [float(r["required_relative_precision_SEMF"]) if r["required_relative_precision_SEMF"] else float("nan") for r in rows_out]
    rel_yuk = [float(r["required_relative_precision_Yukawa"]) if r["required_relative_precision_Yukawa"] else float("nan") for r in rows_out]

    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(a_vals, d_semf, s=8, alpha=0.20, color="tab:red", label="abs(P-SEMF)")
    ax0.scatter(a_vals, d_yuk, s=8, alpha=0.20, color="tab:blue", label="abs(P-Yukawa)")
    ax0.set_xlabel("A")
    ax0.set_ylabel("abs(Delta B) [MeV]")
    ax0.set_title("Differential magnitude across all nuclei")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="upper left", fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    top_semf_for_plot = sorted_semf[:10]
    lbls = [f"{r['symbol']}-{r['A']}" if r["symbol"] else f"Z{r['Z']}A{r['A']}" for r in top_semf_for_plot]
    vals = [float(r["abs_Delta_B_P_minus_SEMF_MeV"]) for r in top_semf_for_plot]
    ax1.bar(range(len(vals)), vals, color="tab:red", alpha=0.85)
    ax1.set_xticks(range(len(vals)))
    ax1.set_xticklabels(lbls, rotation=30, ha="right")
    ax1.set_ylabel("abs(Delta B_P-SEMF) [MeV]")
    ax1.set_title("Top-10 nuclei by differential signal (P vs SEMF)")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    ax2 = fig.add_subplot(gs[1, 0])
    semf_log = [math.log10(v) for v in rel_semf if math.isfinite(v) and v > 0]
    yuk_log = [math.log10(v) for v in rel_yuk if math.isfinite(v) and v > 0]
    ax2.hist(semf_log, bins=60, alpha=0.55, color="tab:red", label="required rel sigma (P-SEMF)")
    ax2.hist(yuk_log, bins=60, alpha=0.55, color="tab:blue", label="required rel sigma (P-Yukawa)")
    ax2.set_xlabel("log10(required relative sigma at 3sigma)")
    ax2.set_ylabel("count")
    ax2.set_title("Precision requirement distribution")
    ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    g_labels = ["light<=40", "41<=mid<=120", "heavy>=121"]
    g_keys = ["light_A_le_40", "mid_A_41_120", "heavy_A_ge_121"]
    g_vals_semf = [_stats(by_group[g]["abs_delta_semf"])["median"] for g in g_keys]
    g_vals_yuk = [_stats(by_group[g]["abs_delta_yukawa"])["median"] for g in g_keys]
    x = list(range(len(g_labels)))
    w = 0.38
    ax3.bar([i - w / 2 for i in x], g_vals_semf, width=w, color="tab:red", alpha=0.85, label="median abs(P-SEMF)")
    ax3.bar([i + w / 2 for i in x], g_vals_yuk, width=w, color="tab:blue", alpha=0.85, label="median abs(P-Yukawa)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(g_labels)
    ax3.set_ylabel("median abs(Delta B) [MeV]")
    ax3.set_title("Differential scale by A category")
    ax3.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax3.legend(loc="upper left", fontsize=8)

    fig.suptitle("Phase 7 / Step 7.13.17.12: quantitative differential predictions and precision targets", y=1.01)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.09, wspace=0.22, hspace=0.30)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_differential_quantification.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    group_stats = {}
    for key, payload in by_group.items():
        group_stats[key] = {
            "abs_delta_semf_mev": _stats(payload["abs_delta_semf"]),
            "abs_delta_yukawa_mev": _stats(payload["abs_delta_yukawa"]),
            "required_relative_sigma_semf": _stats(payload["req_rel_semf"]),
            "required_relative_sigma_yukawa": _stats(payload["req_rel_yukawa"]),
        }

    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_differential_quantification_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.13.17.12",
                "inputs": {"theory_diff_csv": {"path": str(in_csv), "sha256": _sha256(in_csv), "rows": len(rows_out)}},
                "definitions": {
                    "delta_be_mev": "B_pred_pmodel - B_pred_standard_proxy",
                    "required_sigma_3sigma_mev": "abs(delta_be_mev)/3",
                    "required_relative_sigma": "required_sigma_3sigma_mev / B_obs",
                },
                "global_stats": {
                    "abs_delta_semf_mev": _stats(abs_delta_semf_all),
                    "abs_delta_yukawa_mev": _stats(abs_delta_yuk_all),
                    "required_relative_sigma_semf": _stats(req_rel_semf_all),
                    "required_relative_sigma_yukawa": _stats(req_rel_yuk_all),
                },
                "group_stats": group_stats,
                "top20_csv": str(out_top) if top_rows else None,
                "outputs": {"png": str(out_png), "csv": str(out_csv)},
                "notes": [
                    "Current sigma_B_obs comes from AME-side tabulated values in the frozen CSV and is used only as an operational precision indicator.",
                    "This step freezes candidate nuclei and precision scale before introducing any new model freedom.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_png}")
    print(f"  {out_csv}")
    # 条件分岐: `top_rows` を満たす経路を評価する。
    if top_rows:
        print(f"  {out_top}")

    print(f"  {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
