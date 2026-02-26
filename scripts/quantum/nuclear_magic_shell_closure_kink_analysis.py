from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


MAGIC_LIST = [2, 8, 20, 28, 50, 82, 126]
REPRESENTATIVE_CLOSURES = [
    {"label": "O-16", "Z": 8, "N": 8},
    {"label": "Ca-40", "Z": 20, "N": 20},
    {"label": "Sn-132", "Z": 50, "N": 82},
    {"label": "Pb-208", "Z": 82, "N": 126},
]


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


# 関数: `_rms` の入出力契約と処理意図を定義する。

def _rms(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    # 条件分岐: `not finite` を満たす経路を評価する。
    if not finite:
        return float("nan")

    return math.sqrt(sum(v * v for v in finite) / float(len(finite)))


# 関数: `_safe_median` の入出力契約と処理意図を定義する。

def _safe_median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    # 条件分岐: `not finite` を満たす経路を評価する。
    if not finite:
        return float("nan")

    return float(median(finite))


# 関数: `_load_all_nuclei_csv` の入出力契約と処理意図を定義する。

def _load_all_nuclei_csv(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = int(row["Z"])
            n = int(row["N"])
            a = int(row["A"])
            out[(z, n)] = {
                "Z": z,
                "N": n,
                "A": a,
                "symbol": str(row.get("symbol", "")),
                "B_obs_MeV": float(row["B_obs_MeV"]),
                "B_pred_collective_MeV": float(row["B_pred_collective_MeV"]),
            }

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise SystemExit(f"[fail] no rows loaded: {path}")

    return out


# 関数: `_load_semf_proxy_csv` の入出力契約と処理意図を定義する。

def _load_semf_proxy_csv(path: Path) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = int(row["Z"])
            n = int(row["N"])
            b_obs = float(row["B_obs_MeV"])
            delta_p_minus_semf = float(row["Delta_B_P_minus_SEMF_MeV"])
            # Delta_B_P_minus_SEMF = B_P - B_SEMF.
            b_semf = b_obs - delta_p_minus_semf
            out[(z, n)] = b_semf

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise SystemExit(f"[fail] no rows loaded: {path}")

    return out


# 関数: `_build_s2n_map` の入出力契約と処理意図を定義する。

def _build_s2n_map(binding_map: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for (z, n), b_parent in binding_map.items():
        child = (z, n - 2)
        # 条件分岐: `child in binding_map` を満たす経路を評価する。
        if child in binding_map:
            out[(z, n)] = float(b_parent - binding_map[child])

    return out


# 関数: `_build_s2p_map` の入出力契約と処理意図を定義する。

def _build_s2p_map(binding_map: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for (z, n), b_parent in binding_map.items():
        child = (z - 2, n)
        # 条件分岐: `child in binding_map` を満たす経路を評価する。
        if child in binding_map:
            out[(z, n)] = float(b_parent - binding_map[child])

    return out


# 関数: `_build_gap2n_rows` の入出力契約と処理意図を定義する。

def _build_gap2n_rows(
    s2n_obs: dict[tuple[int, int], float],
    s2n_p: dict[tuple[int, int], float],
    s2n_semf: dict[tuple[int, int], float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n0 in MAGIC_LIST:
        for z in sorted({key[0] for key in s2n_obs.keys()}):
            k0 = (z, n0)
            k1 = (z, n0 + 2)
            # 条件分岐: `k0 not in s2n_obs or k1 not in s2n_obs` を満たす経路を評価する。
            if k0 not in s2n_obs or k1 not in s2n_obs:
                continue

            # 条件分岐: `k0 not in s2n_p or k1 not in s2n_p` を満たす経路を評価する。

            if k0 not in s2n_p or k1 not in s2n_p:
                continue

            # 条件分岐: `k0 not in s2n_semf or k1 not in s2n_semf` を満たす経路を評価する。

            if k0 not in s2n_semf or k1 not in s2n_semf:
                continue

            gap_obs = float(s2n_obs[k0] - s2n_obs[k1])
            gap_p = float(s2n_p[k0] - s2n_p[k1])
            gap_semf = float(s2n_semf[k0] - s2n_semf[k1])
            rows.append(
                {
                    "axis": "N",
                    "magic": n0,
                    "Z": z,
                    "gap_kind": "gap_S2n",
                    "gap_obs_MeV": gap_obs,
                    "gap_pred_P_MeV": gap_p,
                    "gap_pred_shell_proxy_MeV": gap_semf,
                    "resid_P_MeV": float(gap_p - gap_obs),
                    "resid_shell_proxy_MeV": float(gap_semf - gap_obs),
                    "delta_P_minus_shell_proxy_MeV": float(gap_p - gap_semf),
                }
            )

    return rows


# 関数: `_build_gap2p_rows` の入出力契約と処理意図を定義する。

def _build_gap2p_rows(
    s2p_obs: dict[tuple[int, int], float],
    s2p_p: dict[tuple[int, int], float],
    s2p_semf: dict[tuple[int, int], float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for z0 in MAGIC_LIST:
        for n in sorted({key[1] for key in s2p_obs.keys()}):
            k0 = (z0, n)
            k1 = (z0 + 2, n)
            # 条件分岐: `k0 not in s2p_obs or k1 not in s2p_obs` を満たす経路を評価する。
            if k0 not in s2p_obs or k1 not in s2p_obs:
                continue

            # 条件分岐: `k0 not in s2p_p or k1 not in s2p_p` を満たす経路を評価する。

            if k0 not in s2p_p or k1 not in s2p_p:
                continue

            # 条件分岐: `k0 not in s2p_semf or k1 not in s2p_semf` を満たす経路を評価する。

            if k0 not in s2p_semf or k1 not in s2p_semf:
                continue

            gap_obs = float(s2p_obs[k0] - s2p_obs[k1])
            gap_p = float(s2p_p[k0] - s2p_p[k1])
            gap_semf = float(s2p_semf[k0] - s2p_semf[k1])
            rows.append(
                {
                    "axis": "Z",
                    "magic": z0,
                    "N": n,
                    "gap_kind": "gap_S2p",
                    "gap_obs_MeV": gap_obs,
                    "gap_pred_P_MeV": gap_p,
                    "gap_pred_shell_proxy_MeV": gap_semf,
                    "resid_P_MeV": float(gap_p - gap_obs),
                    "resid_shell_proxy_MeV": float(gap_semf - gap_obs),
                    "delta_P_minus_shell_proxy_MeV": float(gap_p - gap_semf),
                }
            )

    return rows


# 関数: `_summarize_by_magic` の入出力契約と処理意図を定義する。

def _summarize_by_magic(rows: list[dict[str, Any]], *, axis: str, gap_kind: str) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["magic"])].append(row)

    out: list[dict[str, Any]] = []
    for m in MAGIC_LIST:
        items = grouped.get(m, [])
        out.append(
            {
                "axis": axis,
                "magic": m,
                "gap_kind": gap_kind,
                "n": len(items),
                "median_gap_obs_MeV": _safe_median([float(x["gap_obs_MeV"]) for x in items]),
                "median_gap_pred_P_MeV": _safe_median([float(x["gap_pred_P_MeV"]) for x in items]),
                "median_gap_pred_shell_proxy_MeV": _safe_median(
                    [float(x["gap_pred_shell_proxy_MeV"]) for x in items]
                ),
                "rms_resid_P_MeV": _rms([float(x["resid_P_MeV"]) for x in items]),
                "rms_resid_shell_proxy_MeV": _rms([float(x["resid_shell_proxy_MeV"]) for x in items]),
                "median_delta_P_minus_shell_proxy_MeV": _safe_median(
                    [float(x["delta_P_minus_shell_proxy_MeV"]) for x in items]
                ),
            }
        )

    return out


# 関数: `_representative_rows` の入出力契約と処理意図を定義する。

def _representative_rows(
    closures: list[dict[str, Any]],
    gap2n_rows: list[dict[str, Any]],
    gap2p_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_zn_magic: dict[tuple[int, int], dict[str, Any]] = {}
    by_nz_magic: dict[tuple[int, int], dict[str, Any]] = {}
    for row in gap2n_rows:
        by_zn_magic[(int(row["Z"]), int(row["magic"]))] = row

    for row in gap2p_rows:
        by_nz_magic[(int(row["magic"]), int(row["N"]))] = row

    out: list[dict[str, Any]] = []
    for c in closures:
        z = int(c["Z"])
        n = int(c["N"])
        r_n = by_zn_magic.get((z, n))
        r_p = by_nz_magic.get((z, n))
        out.append(
            {
                "closure": c["label"],
                "Z": z,
                "N": n,
                "gap_S2n_obs_MeV": float(r_n["gap_obs_MeV"]) if r_n else float("nan"),
                "gap_S2n_pred_P_MeV": float(r_n["gap_pred_P_MeV"]) if r_n else float("nan"),
                "gap_S2n_pred_shell_proxy_MeV": float(r_n["gap_pred_shell_proxy_MeV"]) if r_n else float("nan"),
                "gap_S2p_obs_MeV": float(r_p["gap_obs_MeV"]) if r_p else float("nan"),
                "gap_S2p_pred_P_MeV": float(r_p["gap_pred_P_MeV"]) if r_p else float("nan"),
                "gap_S2p_pred_shell_proxy_MeV": float(r_p["gap_pred_shell_proxy_MeV"]) if r_p else float("nan"),
            }
        )

    return out


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        # 条件分岐: `not rows` を満たす経路を評価する。
        if not rows:
            f.write("")
            return

        headers = list(rows[0].keys())
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow([row.get(h) for h in headers])


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_nuclei_csv = out_dir / "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.csv"
    diff_csv = out_dir / "nuclear_binding_energy_frequency_mapping_differential_quantification.csv"
    shell_proxy_metrics = (
        out_dir
        / "nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_shellS_g2_metrics.json"
    )

    # 条件分岐: `not all_nuclei_csv.exists()` を満たす経路を評価する。
    if not all_nuclei_csv.exists():
        raise SystemExit(f"[fail] missing required input: {all_nuclei_csv}")

    # 条件分岐: `not diff_csv.exists()` を満たす経路を評価する。

    if not diff_csv.exists():
        raise SystemExit(f"[fail] missing required input: {diff_csv}")

    # 条件分岐: `not shell_proxy_metrics.exists()` を満たす経路を評価する。

    if not shell_proxy_metrics.exists():
        raise SystemExit(f"[fail] missing required input: {shell_proxy_metrics}")

    all_rows = _load_all_nuclei_csv(all_nuclei_csv)
    semf_proxy_map = _load_semf_proxy_csv(diff_csv)

    b_obs: dict[tuple[int, int], float] = {k: float(v["B_obs_MeV"]) for k, v in all_rows.items()}
    b_p: dict[tuple[int, int], float] = {k: float(v["B_pred_collective_MeV"]) for k, v in all_rows.items()}
    b_semf: dict[tuple[int, int], float] = {k: float(semf_proxy_map[k]) for k in all_rows.keys() if k in semf_proxy_map}

    s2n_obs = _build_s2n_map(b_obs)
    s2n_p = _build_s2n_map(b_p)
    s2n_semf = _build_s2n_map(b_semf)
    s2p_obs = _build_s2p_map(b_obs)
    s2p_p = _build_s2p_map(b_p)
    s2p_semf = _build_s2p_map(b_semf)

    gap2n_rows = _build_gap2n_rows(s2n_obs, s2n_p, s2n_semf)
    gap2p_rows = _build_gap2p_rows(s2p_obs, s2p_p, s2p_semf)

    summary_n = _summarize_by_magic(gap2n_rows, axis="N", gap_kind="gap_S2n")
    summary_z = _summarize_by_magic(gap2p_rows, axis="Z", gap_kind="gap_S2p")
    summary_rows = summary_n + summary_z
    representative_rows = _representative_rows(REPRESENTATIVE_CLOSURES, gap2n_rows, gap2p_rows)

    # Load shellS_g2 summary as "standard shell-model proxy" diagnostic.
    shell_payload = json.loads(shell_proxy_metrics.read_text(encoding="utf-8"))
    shell_results = shell_payload.get("results", [])
    best_shell_row: dict[str, Any] | None = None
    best_delta = float("inf")
    # 条件分岐: `isinstance(shell_results, list)` を満たす経路を評価する。
    if isinstance(shell_results, list):
        for item in shell_results:
            # 条件分岐: `not isinstance(item, dict)` を満たす経路を評価する。
            if not isinstance(item, dict):
                continue

            gap_sn = item.get("gap_Sn", {}) if isinstance(item.get("gap_Sn"), dict) else {}
            gap_sp = item.get("gap_Sp", {}) if isinstance(item.get("gap_Sp"), dict) else {}
            other_sn = gap_sn.get("other_magic", {}) if isinstance(gap_sn.get("other_magic"), dict) else {}
            other_sp = gap_sp.get("other_magic", {}) if isinstance(gap_sp.get("other_magic"), dict) else {}
            shell_sn = float(other_sn.get("rms_resid_pairing_shell_MeV", float("nan")))
            pair_sn = float(other_sn.get("rms_resid_pairing_only_MeV", float("nan")))
            shell_sp = float(other_sp.get("rms_resid_pairing_shell_MeV", float("nan")))
            pair_sp = float(other_sp.get("rms_resid_pairing_only_MeV", float("nan")))
            # 条件分岐: `not (math.isfinite(shell_sn) and math.isfinite(pair_sn))` を満たす経路を評価する。
            if not (math.isfinite(shell_sn) and math.isfinite(pair_sn)):
                continue

            # 条件分岐: `not math.isfinite(shell_sp)` を満たす経路を評価する。

            if not math.isfinite(shell_sp):
                shell_sp = 0.0

            # 条件分岐: `not math.isfinite(pair_sp)` を満たす経路を評価する。

            if not math.isfinite(pair_sp):
                pair_sp = 0.0

            delta = abs(shell_sn - pair_sn) + abs(shell_sp - pair_sp)
            # 条件分岐: `delta < best_delta` を満たす経路を評価する。
            if delta < best_delta:
                best_delta = delta
                best_shell_row = item

    # Plot.

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    x = list(range(len(MAGIC_LIST)))
    lbl = [str(m) for m in MAGIC_LIST]

    obs_n = [float(r["median_gap_obs_MeV"]) for r in summary_n]
    p_n = [float(r["median_gap_pred_P_MeV"]) for r in summary_n]
    s_n = [float(r["median_gap_pred_shell_proxy_MeV"]) for r in summary_n]
    ax00.plot(x, obs_n, marker="o", lw=1.6, label="obs")
    ax00.plot(x, p_n, marker="s", lw=1.6, label="P-model")
    ax00.plot(x, s_n, marker="^", lw=1.6, label="shell proxy (SEMF)")
    ax00.set_xticks(x)
    ax00.set_xticklabels(lbl)
    ax00.set_xlabel("magic N")
    ax00.set_ylabel("median gap_S2n [MeV]")
    ax00.set_title("S2n kink medians by magic N")
    ax00.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax00.legend(loc="best", fontsize=8)

    obs_z = [float(r["median_gap_obs_MeV"]) for r in summary_z]
    p_z = [float(r["median_gap_pred_P_MeV"]) for r in summary_z]
    s_z = [float(r["median_gap_pred_shell_proxy_MeV"]) for r in summary_z]
    ax01.plot(x, obs_z, marker="o", lw=1.6, label="obs")
    ax01.plot(x, p_z, marker="s", lw=1.6, label="P-model")
    ax01.plot(x, s_z, marker="^", lw=1.6, label="shell proxy (SEMF)")
    ax01.set_xticks(x)
    ax01.set_xticklabels(lbl)
    ax01.set_xlabel("magic Z")
    ax01.set_ylabel("median gap_S2p [MeV]")
    ax01.set_title("S2p kink medians by magic Z")
    ax01.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax01.legend(loc="best", fontsize=8)

    rms_p_n = [float(r["rms_resid_P_MeV"]) for r in summary_n]
    rms_s_n = [float(r["rms_resid_shell_proxy_MeV"]) for r in summary_n]
    ax10.bar([i - 0.18 for i in x], rms_p_n, width=0.36, label="P-model", alpha=0.85)
    ax10.bar([i + 0.18 for i in x], rms_s_n, width=0.36, label="shell proxy (SEMF)", alpha=0.85)
    ax10.set_xticks(x)
    ax10.set_xticklabels(lbl)
    ax10.set_xlabel("magic N")
    ax10.set_ylabel("RMS residual gap_S2n [MeV]")
    ax10.set_title("S2n kink residual RMS")
    ax10.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax10.legend(loc="best", fontsize=8)

    rms_p_z = [float(r["rms_resid_P_MeV"]) for r in summary_z]
    rms_s_z = [float(r["rms_resid_shell_proxy_MeV"]) for r in summary_z]
    ax11.bar([i - 0.18 for i in x], rms_p_z, width=0.36, label="P-model", alpha=0.85)
    ax11.bar([i + 0.18 for i in x], rms_s_z, width=0.36, label="shell proxy (SEMF)", alpha=0.85)
    ax11.set_xticks(x)
    ax11.set_xticklabels(lbl)
    ax11.set_xlabel("magic Z")
    ax11.set_ylabel("RMS residual gap_S2p [MeV]")
    ax11.set_title("S2p kink residual RMS")
    ax11.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax11.legend(loc="best", fontsize=8)

    fig.suptitle("Phase 7 / Step 7.16.2: magic-number shell-closure kink quantification (S2n/S2p)", y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))

    out_png = out_dir / "nuclear_magic_shell_closure_kink_quantification.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_summary_csv = out_dir / "nuclear_magic_shell_closure_kink_summary.csv"
    out_rep_csv = out_dir / "nuclear_magic_shell_closure_kink_representative4.csv"
    _write_csv(out_summary_csv, summary_rows)
    _write_csv(out_rep_csv, representative_rows)

    out_json = out_dir / "nuclear_magic_shell_closure_kink_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.16.2",
                "inputs": {
                    "all_nuclei_csv": {"path": str(all_nuclei_csv), "sha256": _sha256(all_nuclei_csv)},
                    "differential_quantification_csv": {"path": str(diff_csv), "sha256": _sha256(diff_csv)},
                    "shell_model_proxy_metrics": {"path": str(shell_proxy_metrics), "sha256": _sha256(shell_proxy_metrics)},
                },
                "definitions": {
                    "S2n": "S_2n(Z,N)=B(Z,N)-B(Z,N-2)",
                    "S2p": "S_2p(Z,N)=B(Z,N)-B(Z-2,N)",
                    "gap_S2n": "gap_2n(Z;N0)=S_2n(Z,N0)-S_2n(Z,N0+2)",
                    "gap_S2p": "gap_2p(N;Z0)=S_2p(Z0,N)-S_2p(Z0+2,N)",
                    "shell_proxy": "SEMF proxy reconstructed from Delta_B_P_minus_SEMF",
                },
                "magic_list": MAGIC_LIST,
                "counts": {
                    "n_gap_S2n_rows": len(gap2n_rows),
                    "n_gap_S2p_rows": len(gap2p_rows),
                    "n_representative": len(representative_rows),
                },
                "summary_by_magic": {
                    "gap_S2n": summary_n,
                    "gap_S2p": summary_z,
                },
                "representative_4_closures": representative_rows,
                "shell_model_proxy_reference": {
                    "source_step": shell_payload.get("step"),
                    "best_row_by_other_magic_delta": best_shell_row,
                },
                "outputs": {
                    "summary_csv": str(out_summary_csv),
                    "representative_csv": str(out_rep_csv),
                    "figure_png": str(out_png),
                },
                "notes": [
                    "Step 7.16.2 quantifies shell-closure kinks using S2n/S2p and compares P-model with a smooth shell-proxy baseline.",
                    "The shell-model proxy reference is attached from Step 7.13.15.33 (shellS_g2) to keep continuity with previous frozen diagnostics.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_summary_csv}")
    print(f"  {out_rep_csv}")
    print(f"  {out_png}")
    print(f"  {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
