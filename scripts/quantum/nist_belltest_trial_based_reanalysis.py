from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrialCounts:
    n_trials: np.ndarray  # (2,2) by (a_setting, b_setting)
    n_coinc: np.ndarray  # (2,2) by (a_setting, b_setting)
    n_click_a: np.ndarray  # (2,) by a_setting
    n_click_b: np.ndarray  # (2,) by b_setting
    n_trials_a: np.ndarray  # (2,) by a_setting
    n_trials_b: np.ndarray  # (2,) by b_setting
    n_total: int
    n_bad_sync: int
    n_invalid_settings: int


def _map_setting_bits(x: np.ndarray) -> np.ndarray:
    # In processed_compressed/hdf5 builds, setting is stored as a bitmask:
    #   1 => setting 0
    #   2 => setting 1
    # Other values are rare anomalies (both or neither fired).
    return np.where(x == 1, 0, np.where(x == 2, 1, -1)).astype(np.int8, copy=False)


def _compute_trial_counts(
    *,
    a_setting_raw: np.ndarray,
    b_setting_raw: np.ndarray,
    a_clicks: np.ndarray,
    b_clicks: np.ndarray,
    bad_sync_idx: np.ndarray,
) -> TrialCounts:
    n = int(a_clicks.shape[0])
    # 条件分岐: `b_clicks.shape[0] != n or a_setting_raw.shape[0] != n or b_setting_raw.shape[...` を満たす経路を評価する。
    if b_clicks.shape[0] != n or a_setting_raw.shape[0] != n or b_setting_raw.shape[0] != n:
        raise ValueError("array length mismatch in hdf5 build file")

    mask = np.ones((n,), dtype=bool)
    for idx in bad_sync_idx.tolist():
        # 条件分岐: `0 <= idx < n` を満たす経路を評価する。
        if 0 <= idx < n:
            mask[int(idx)] = False

    a_setting = _map_setting_bits(a_setting_raw)
    b_setting = _map_setting_bits(b_setting_raw)
    invalid = mask & ((a_setting < 0) | (b_setting < 0))
    mask_valid = mask & ~invalid

    a_has = mask_valid & (a_clicks != 0)
    b_has = mask_valid & (b_clicks != 0)
    coinc = mask_valid & ((a_clicks & b_clicks) != 0)

    n_trials = np.zeros((2, 2), dtype=np.int64)
    n_coinc = np.zeros((2, 2), dtype=np.int64)
    for a in (0, 1):
        for b in (0, 1):
            sel = mask_valid & (a_setting == a) & (b_setting == b)
            n_trials[a, b] = int(sel.sum())
            n_coinc[a, b] = int((coinc & sel).sum())

    n_trials_a = np.array(
        [int((mask_valid & (a_setting == 0)).sum()), int((mask_valid & (a_setting == 1)).sum())], dtype=np.int64
    )
    n_trials_b = np.array(
        [int((mask_valid & (b_setting == 0)).sum()), int((mask_valid & (b_setting == 1)).sum())], dtype=np.int64
    )
    n_click_a = np.array(
        [int((a_has & (a_setting == 0)).sum()), int((a_has & (a_setting == 1)).sum())], dtype=np.int64
    )
    n_click_b = np.array(
        [int((b_has & (b_setting == 0)).sum()), int((b_has & (b_setting == 1)).sum())], dtype=np.int64
    )

    return TrialCounts(
        n_trials=n_trials,
        n_coinc=n_coinc,
        n_click_a=n_click_a,
        n_click_b=n_click_b,
        n_trials_a=n_trials_a,
        n_trials_b=n_trials_b,
        n_total=int(mask_valid.sum()),
        n_bad_sync=int(np.unique(bad_sync_idx).size),
        n_invalid_settings=int(invalid.sum()),
    )


def _ch_j_variants(counts: TrialCounts) -> dict[str, dict[str, float | int]]:
    # Clauser–Horne (CH) form for binary outcomes A,B ∈ {0,1}:
    #   J = P11(A1,B1)+P11(A1,B2)+P11(A2,B1)-P11(A2,B2) - P1(A1) - P1(B1) <= 0 (local)
    # We compute the four possible choices of (A1,B1) ∈ {(0,0),(0,1),(1,0),(1,1)}.
    # (A2,B2) are the opposite settings.
    p_coinc = counts.n_coinc / np.maximum(1, counts.n_trials)
    p_a = counts.n_click_a / np.maximum(1, counts.n_trials_a)
    p_b = counts.n_click_b / np.maximum(1, counts.n_trials_b)

    out: dict[str, dict[str, float | int]] = {}
    best_key = None
    best_j = None
    for a1 in (0, 1):
        a2 = 1 - a1
        for b1 in (0, 1):
            b2 = 1 - b1
            j = (
                float(p_coinc[a1, b1])
                + float(p_coinc[a1, b2])
                + float(p_coinc[a2, b1])
                - float(p_coinc[a2, b2])
                - float(p_a[a1])
                - float(p_b[b1])
            )
            key = f"A1={a1},B1={b1}"
            out[key] = {
                "A1": int(a1),
                "A2": int(a2),
                "B1": int(b1),
                "B2": int(b2),
                "J_prob": float(j),
            }
            # 条件分岐: `best_j is None or j > best_j` を満たす経路を評価する。
            if best_j is None or j > best_j:
                best_j = j
                best_key = key

    assert best_key is not None and best_j is not None
    out["_best"] = {"key": best_key, "J_prob": float(best_j)}
    return out


def _load_coincidence_sweep(csv_path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        raise ValueError(f"empty csv: {csv_path}")

    def col(name: str, t=float) -> np.ndarray:
        return np.asarray([t(r[name]) for r in rows])

    return {
        "window_ns": col("window_ns", float),
        "pairs_total": col("pairs_total", int).astype(np.int64),
        "c00": col("c00", int).astype(np.int64),
        "c01": col("c01", int).astype(np.int64),
        "c10": col("c10", int).astype(np.int64),
        "c11": col("c11", int).astype(np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trial-based reanalysis of NIST belltestdata using processed_compressed/hdf5 build files."
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=None,
        help="Path to *.dat.compressed.build.hdf5 (default: 03_43 afterfixingModeLocking cached path).",
    )
    parser.add_argument(
        "--out-tag",
        default="03_43_afterfixingModeLocking_s3600",
        help="Output tag suffix (default: 03_43_afterfixingModeLocking_s3600).",
    )
    parser.add_argument(
        "--max-syncs",
        type=int,
        default=None,
        help="Optional: only analyze first N sync pulses (for quick checks).",
    )
    parser.add_argument(
        "--coincidence-sweep-csv",
        type=Path,
        default=None,
        help=(
            "Optional: path to coincidence-based window sweep CSV produced by "
            "nist_belltest_time_tag_reanalysis.py; used to compare trial-based vs coincidence-based."
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    default_hdf5 = (
        root
        / "data"
        / "quantum"
        / "sources"
        / "nist_belltestdata"
        / "processed_compressed"
        / "hdf5"
        / "2015_09_18"
        / "03_43_CH_pockel_100kHz.run4.afterTimingfix2_afterfixingModeLocking.dat.compressed.build.hdf5"
    )
    hdf5_path = args.hdf5 or default_hdf5
    # 条件分岐: `not hdf5_path.exists()` を満たす経路を評価する。
    if not hdf5_path.exists():
        raise SystemExit(f"[fail] missing hdf5 build: {hdf5_path} (run fetch_nist_belltestdata.py --hdf5)")

    # Load trial-based (sync-indexed) arrays from the build file.

    try:
        import h5py
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "[fail] missing dependency: h5py\n"
            "Install with: python -m pip install -U h5py\n"
            f"Import error: {e}"
        )

    with h5py.File(hdf5_path, "r") as f:
        # Config
        cfg = {}
        for side in ("alice", "bob"):
            g = f[f"config/{side}"]
            cfg[side] = {k: int(g[k][()]) for k in g.keys()}

        a_setting = f["alice/settings"]
        b_setting = f["bob/settings"]
        a_clicks = f["alice/clicks"]
        b_clicks = f["bob/clicks"]

        n = int(a_clicks.shape[0])
        n_use = min(n, int(args.max_syncs)) if args.max_syncs is not None else n

        bad: set[int] = set()
        for side in ("alice", "bob"):
            g = f[side]
            # 条件分岐: `"badSyncInfo" not in g` を満たす経路を評価する。
            if "badSyncInfo" not in g:
                continue

            arr = g["badSyncInfo"][()]
            # Observed shapes: (1, N) in some runs; missing in others.
            flat = np.asarray(arr).reshape(-1)
            bad |= set(map(int, flat.tolist()))

        bad_idx = np.asarray(sorted(i for i in bad if 0 <= i < n_use), dtype=np.int64)

        counts = _compute_trial_counts(
            a_setting_raw=a_setting[:n_use],
            b_setting_raw=b_setting[:n_use],
            a_clicks=a_clicks[:n_use],
            b_clicks=b_clicks[:n_use],
            bad_sync_idx=bad_idx,
        )

    ch = _ch_j_variants(counts)

    # Optional: compare against coincidence-based (greedy pairing) window sweep.
    sweep = None
    # 条件分岐: `args.coincidence_sweep_csv is not None` を満たす経路を評価する。
    if args.coincidence_sweep_csv is not None:
        sweep_csv = args.coincidence_sweep_csv
    else:
        sweep_csv = out_dir / f"nist_belltest_coincidence_sweep__{args.out_tag}.csv"
        sweep_csv = sweep_csv if sweep_csv.exists() else None

    # 条件分岐: `sweep_csv is not None and sweep_csv.exists()` を満たす経路を評価する。

    if sweep_csv is not None and sweep_csv.exists():
        sweep = _load_coincidence_sweep(sweep_csv)
        # Compute CH J_prob for A1=0,B1=0 using sweep coincidences + trial-based denominators.
        a1 = 0
        b1 = 0
        a2 = 1
        b2 = 1
        p_a1 = float(counts.n_click_a[a1] / max(1, counts.n_trials_a[a1]))
        p_b1 = float(counts.n_click_b[b1] / max(1, counts.n_trials_b[b1]))
        j_sweep = (
            sweep["c00"] / np.maximum(1, counts.n_trials[0, 0])
            + sweep["c01"] / np.maximum(1, counts.n_trials[0, 1])
            + sweep["c10"] / np.maximum(1, counts.n_trials[1, 0])
            - sweep["c11"] / np.maximum(1, counts.n_trials[1, 1])
            - p_a1
            - p_b1
        ).astype(np.float64)
        sweep["J_prob_A1=0_B1=0"] = j_sweep

    # Outputs

    tag = args.out_tag
    out_png = out_dir / f"nist_belltest_trial_based__{tag}.png"
    out_json = out_dir / f"nist_belltest_trial_based_metrics__{tag}.json"
    out_csv = out_dir / f"nist_belltest_trial_based_counts__{tag}.csv"

    # CSV summary
    lines = [
        ["a_setting", "b_setting", "n_trials", "n_coinc"],
        ["0", "0", str(int(counts.n_trials[0, 0])), str(int(counts.n_coinc[0, 0]))],
        ["0", "1", str(int(counts.n_trials[0, 1])), str(int(counts.n_coinc[0, 1]))],
        ["1", "0", str(int(counts.n_trials[1, 0])), str(int(counts.n_coinc[1, 0]))],
        ["1", "1", str(int(counts.n_trials[1, 1])), str(int(counts.n_coinc[1, 1]))],
    ]
    out_csv.write_text("\n".join(",".join(r) for r in lines) + "\n", encoding="utf-8")

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6), dpi=160)

    ax0 = axes[0]
    ax0.set_title("Coincidences: coincidence-based (window) vs trial-based (sync/slot)")

    # 条件分岐: `sweep is not None` を満たす経路を評価する。
    if sweep is not None:
        ax0.plot(
            sweep["window_ns"],
            sweep["pairs_total"],
            marker="o",
            lw=1.4,
            label="coincidence-based pairs_total (greedy, PPS-aligned)",
        )

    trial_total = int(counts.n_coinc.sum())
    ax0.axhline(trial_total, color="0.2", ls="--", lw=1.0, label=f"trial-based coincidences={trial_total}")
    ax0.set_xscale("log")
    ax0.set_xlabel("coincidence window (ns)")
    ax0.set_ylabel("coincidences (count)")
    ax0.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(frameon=True, fontsize=9)

    ax1 = axes[1]
    ax1.set_title("CH J_prob (A1=0,B1=0): window dependence")
    # Trial-based J for A1=0,B1=0.
    j_trial = float(ch["A1=0,B1=0"]["J_prob"])
    # 条件分岐: `sweep is not None and "J_prob_A1=0_B1=0" in sweep` を満たす経路を評価する。
    if sweep is not None and "J_prob_A1=0_B1=0" in sweep:
        ax1.plot(
            sweep["window_ns"],
            sweep["J_prob_A1=0_B1=0"],
            marker="o",
            lw=1.4,
            label="coincidence-based J_prob",
        )
        # closest match point
        idx = int(np.argmin(np.abs(sweep["J_prob_A1=0_B1=0"] - j_trial)))
        ax1.axvline(float(sweep["window_ns"][idx]), color="0.5", ls=":", lw=1.0)
        ax1.text(
            float(sweep["window_ns"][idx]) * 1.06,
            float(sweep["J_prob_A1=0_B1=0"][idx]),
            f"closest\\n{float(sweep['window_ns'][idx]):g} ns",
            fontsize=8,
            va="bottom",
        )

    ax1.axhline(j_trial, color="0.2", ls="--", lw=1.0, label=f"trial-based J_prob={j_trial:.3g}")
    ax1.axhline(0.0, color="0.4", ls="-", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel("coincidence window (ns)")
    ax1.set_ylabel("J_prob")
    ax1.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)

    note = (
        f"hdf5={hdf5_path.name} | syncs_used={counts.n_total} (bad_sync={counts.n_bad_sync}, invalid_setting={counts.n_invalid_settings})\\n"
        f"cfg(alice)={cfg['alice']} cfg(bob)={cfg['bob']} | best_J={ch['_best']['J_prob']:.3g} ({ch['_best']['key']})"
    )
    fig.text(0.01, -0.02, note, fontsize=8)
    fig.tight_layout()

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": {
            "source": "NIST belltestdata (processed_compressed/hdf5 build; Shalm et al. 2015 time-tag repository)",
            "hdf5": str(hdf5_path),
        },
        "config": cfg,
        "analysis": {
            "max_syncs": args.max_syncs,
            "exclude_bad_sync": True,
            "coincidence_definition": "same-slot within sync: (alice_clicks & bob_clicks) != 0",
            "setting_mapping": "1->0, 2->1 (other values treated as invalid and excluded)",
            "ch_definition": "CH J_prob = P11(A1,B1)+P11(A1,B2)+P11(A2,B1)-P11(A2,B2) - P1(A1) - P1(B1)",
        },
        "counts": {
            "syncs_used": counts.n_total,
            "bad_sync_unique": counts.n_bad_sync,
            "invalid_settings": counts.n_invalid_settings,
            "trials_by_setting_pair": counts.n_trials.tolist(),
            "coinc_by_setting_pair": counts.n_coinc.tolist(),
            "alice_trials_by_setting": counts.n_trials_a.tolist(),
            "bob_trials_by_setting": counts.n_trials_b.tolist(),
            "alice_clicks_by_setting": counts.n_click_a.tolist(),
            "bob_clicks_by_setting": counts.n_click_b.tolist(),
        },
        "ch_j_variants": ch,
        "comparison": {},
        "outputs": {"png": str(out_png), "json": str(out_json), "csv": str(out_csv)},
        "notes": [
            "This script focuses on trial-based counting using the NIST 'build' (processed) file.",
            "It is intended to quantify algorithm dependence between trial-based and coincidence-based pipelines.",
        ],
    }

    # 条件分岐: `sweep is not None` を満たす経路を評価する。
    if sweep is not None:
        metrics["comparison"] = {
            "coincidence_sweep_csv": str(sweep_csv),
            "windows_ns": sweep["window_ns"].astype(float).tolist(),
            "pairs_total": sweep["pairs_total"].astype(int).tolist(),
            "j_prob_a1_0_b1_0": sweep.get("J_prob_A1=0_B1=0", np.asarray([], dtype=float)).astype(float).tolist(),
        }

    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
