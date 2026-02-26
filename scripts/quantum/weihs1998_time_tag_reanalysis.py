from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ChshBest:
    s_value: float
    abs_s_value: float
    swap_a: bool
    swap_b: bool
    sign_matrix: list[list[int]]


def _chsh_sign_patterns() -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    for mask in range(16):
        s = np.array(
            [
                [1 if (mask >> 0) & 1 else -1, 1 if (mask >> 1) & 1 else -1],
                [1 if (mask >> 2) & 1 else -1, 1 if (mask >> 3) & 1 else -1],
            ],
            dtype=np.int8,
        )
        # 条件分岐: `int(np.prod(s)) == -1` を満たす経路を評価する。
        if int(np.prod(s)) == -1:
            patterns.append(s)

    return patterns


_CHSH_SIGNS = _chsh_sign_patterns()


def _best_chsh(E: np.ndarray) -> ChshBest:
    # 条件分岐: `E.shape != (2, 2)` を満たす経路を評価する。
    if E.shape != (2, 2):
        raise ValueError(f"E must be 2x2, got {E.shape}")

    best: ChshBest | None = None
    for swap_a in (False, True):
        for swap_b in (False, True):
            E2 = E.copy()
            # 条件分岐: `swap_a` を満たす経路を評価する。
            if swap_a:
                E2 = E2[[1, 0], :]

            # 条件分岐: `swap_b` を満たす経路を評価する。

            if swap_b:
                E2 = E2[:, [1, 0]]

            for s in _CHSH_SIGNS:
                v = float(np.sum(s * E2))
                a = abs(v)
                # 条件分岐: `best is None or a > best.abs_s_value` を満たす経路を評価する。
                if best is None or a > best.abs_s_value:
                    best = ChshBest(
                        s_value=v,
                        abs_s_value=a,
                        swap_a=swap_a,
                        swap_b=swap_b,
                        sign_matrix=s.astype(int).tolist(),
                    )

    assert best is not None
    return best


def _canonical_chsh(E: np.ndarray) -> float:
    # Canonical (one common convention): S = E00 + E01 + E10 - E11
    return float(E[0, 0] + E[0, 1] + E[1, 0] - E[1, 1])


def _apply_chsh_variant(E: np.ndarray, variant: ChshBest) -> float:
    E2 = E.copy()
    # 条件分岐: `variant.swap_a` を満たす経路を評価する。
    if variant.swap_a:
        E2 = E2[[1, 0], :]

    # 条件分岐: `variant.swap_b` を満たす経路を評価する。

    if variant.swap_b:
        E2 = E2[:, [1, 0]]

    s = np.array(variant.sign_matrix, dtype=np.int8)
    return float(np.sum(s * E2))


def _read_zip_bytes(zip_path: Path, member: str) -> bytes:
    with zipfile.ZipFile(zip_path) as z:
        return z.read(member)


def _load_run_arrays(*, src_dir: Path, subdir: str, run: str) -> dict[str, np.ndarray]:
    alice_zip = src_dir / "Alice.zip"
    bob_zip = src_dir / "Bob.zip"
    # 条件分岐: `not alice_zip.exists() or not bob_zip.exists()` を満たす経路を評価する。
    if not alice_zip.exists() or not bob_zip.exists():
        raise FileNotFoundError(
            "Missing Weihs1998 zips. Fetch first:\n"
            "  python -B scripts/quantum/fetch_weihs1998_zenodo_7185335.py"
        )

    a_prefix = f"Alice/Timetags/{subdir}/{run}"
    b_prefix = f"Bob/Timetags/{subdir}/{run}"

    a_v = np.frombuffer(_read_zip_bytes(alice_zip, f"{a_prefix}_V.dat"), dtype=">f8")
    a_c = np.frombuffer(_read_zip_bytes(alice_zip, f"{a_prefix}_C.dat"), dtype=">u2")
    b_v = np.frombuffer(_read_zip_bytes(bob_zip, f"{b_prefix}_V.dat"), dtype=">f8")
    b_c = np.frombuffer(_read_zip_bytes(bob_zip, f"{b_prefix}_C.dat"), dtype=">u2")

    # 条件分岐: `len(a_v) != len(a_c) or len(b_v) != len(b_c)` を満たす経路を評価する。
    if len(a_v) != len(a_c) or len(b_v) != len(b_c):
        raise RuntimeError(
            f"length mismatch: A(V)={len(a_v)} A(C)={len(a_c)}; B(V)={len(b_v)} B(C)={len(b_c)}"
        )

    return {"a_t": a_v, "a_c": a_c, "b_t": b_v, "b_c": b_c}


def _estimate_offset_s(t_a: np.ndarray, t_b: np.ndarray, *, sample_max: int = 200_000) -> dict[str, float]:
    # Estimate constant offset by nearest-neighbor dt peak (coarse but robust enough for plotting sweeps).
    if t_a.size == 0 or t_b.size == 0:
        raise ValueError("empty t_a/t_b")

    # 条件分岐: `t_a.size > sample_max` を満たす経路を評価する。

    if t_a.size > sample_max:
        idx = np.linspace(0, t_a.size - 1, sample_max, dtype=np.int64)
        a = t_a[idx]
    else:
        a = t_a

    j = np.searchsorted(t_b, a)
    j2 = np.clip(j, 0, t_b.size - 1)
    j1 = np.clip(j - 1, 0, t_b.size - 1)
    choose_j2 = np.abs(t_b[j2] - a) < np.abs(t_b[j1] - a)
    jn = np.where(choose_j2, j2, j1)
    dt = t_b[jn] - a

    med = float(np.median(dt))
    half = 200e-9  # +/- 200 ns around median
    mask = (dt > (med - half)) & (dt < (med + half))
    sel = dt[mask]
    # 条件分岐: `sel.size < 100` を満たす経路を評価する。
    if sel.size < 100:
        return {"offset_s": med, "offset_method": "median_dt", "offset_peak_count": float(sel.size)}

    # Bin width: 0.5 ns. Enough to locate the peak without overfitting noise.

    bin_w = 0.5e-9
    n_bins = int(math.ceil((2 * half) / bin_w))
    edges = np.linspace(med - half, med + half, n_bins + 1)
    counts, _ = np.histogram(sel, bins=edges)
    k = int(np.argmax(counts))
    offset = 0.5 * (float(edges[k]) + float(edges[k + 1]))
    return {
        "offset_s": offset,
        "offset_method": "nearest_neighbor_dt_peak",
        "offset_median_s": med,
        "offset_peak_count": float(int(counts[k])),
        "offset_hist_bin_width_s": bin_w,
        "offset_hist_half_range_s": half,
    }


def _extract_setting_and_outcome(c: int, *, encoding: str) -> tuple[int, int]:
    # c in {0,1,2,3}.
    if encoding == "bit0-setting":
        setting = c & 1
        detector = (c >> 1) & 1
    # 条件分岐: 前段条件が不成立で、`encoding == "bit0-detector"` を追加評価する。
    elif encoding == "bit0-detector":
        detector = c & 1
        setting = (c >> 1) & 1
    else:
        raise ValueError(f"unknown encoding: {encoding}")

    outcome = 1 if detector == 0 else -1
    return int(setting), int(outcome)


def _pair_and_accumulate(
    t_a: np.ndarray,
    c_a: np.ndarray,
    t_b: np.ndarray,
    c_b: np.ndarray,
    *,
    offset_s: float,
    window_s: float,
    encoding: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    # Greedy 1:1 pairing by time order within |dt - offset| <= window_s.
    n = np.zeros((2, 2), dtype=np.int64)
    sum_prod = np.zeros((2, 2), dtype=np.int64)
    pairs_total = 0

    i = 0
    j = 0
    na = int(t_a.size)
    nb = int(t_b.size)
    while i < na and j < nb:
        dt = float(t_b[j] - t_a[i] - offset_s)
        # 条件分岐: `dt < -window_s` を満たす経路を評価する。
        if dt < -window_s:
            j += 1
            continue

        # 条件分岐: `dt > window_s` を満たす経路を評価する。

        if dt > window_s:
            i += 1
            continue

        a_set, a_out = _extract_setting_and_outcome(int(c_a[i]), encoding=encoding)
        b_set, b_out = _extract_setting_and_outcome(int(c_b[j]), encoding=encoding)
        prod = a_out * b_out
        n[a_set, b_set] += 1
        sum_prod[a_set, b_set] += prod
        pairs_total += 1
        i += 1
        j += 1

    return n, sum_prod, pairs_total


def _safe_div(sum_prod: np.ndarray, n: np.ndarray) -> np.ndarray:
    E = np.full((2, 2), np.nan, dtype=float)
    for a in (0, 1):
        for b in (0, 1):
            # 条件分岐: `n[a, b] > 0` を満たす経路を評価する。
            if n[a, b] > 0:
                E[a, b] = float(sum_prod[a, b]) / float(n[a, b])

    return E


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reanalyze Weihs et al. 1998 time-tag data (Zenodo 7185335) with coincidence-window sweeps."
    )
    parser.add_argument("--subdir", default="longdist", help="Timetags subdirectory (default: longdist)")
    parser.add_argument("--run", default="longdist1", help="Run prefix inside the subdir (default: longdist1)")
    parser.add_argument(
        "--encoding",
        default="bit0-setting",
        choices=["bit0-setting", "bit0-detector"],
        help="Bit assignment in *_C.dat (default: bit0-setting).",
    )
    parser.add_argument(
        "--offset-ns",
        type=float,
        default=None,
        help="Optional fixed time offset in ns (B - A - offset). If omitted, estimate from data.",
    )
    parser.add_argument(
        "--windows-ns",
        default="0.25,0.5,0.75,1,1.5,2,3,4,6,8,10",
        help="Comma-separated coincidence half-widths in ns (|dt| <= window).",
    )
    parser.add_argument(
        "--ref-window-ns",
        type=float,
        default=1.0,
        help="Reference window (ns) used to pick a fixed CHSH sign/swap variant (default: 1.0).",
    )
    parser.add_argument(
        "--out-tag",
        default=None,
        help="Output tag (default: weihs1998_<subdir>_<run>)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / "zenodo_7185335"
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    subdir = str(args.subdir)
    run = str(args.run)
    out_tag = str(args.out_tag) if args.out_tag else f"weihs1998_{subdir}_{run}"

    arrays = _load_run_arrays(src_dir=src_dir, subdir=subdir, run=run)
    t_a = arrays["a_t"]
    c_a = arrays["a_c"]
    t_b = arrays["b_t"]
    c_b = arrays["b_c"]

    offset_info = _estimate_offset_s(t_a, t_b)
    offset_s = float(args.offset_ns) * 1e-9 if args.offset_ns is not None else float(offset_info["offset_s"])

    windows_ns = [float(x.strip()) for x in str(args.windows_ns).split(",") if x.strip()]
    windows_ns = sorted(set(windows_ns))

    # Pick a fixed sign/swap variant from a reference window to avoid "per-window best" overfitting.
    ref_w = float(args.ref_window_ns)
    # 条件分岐: `ref_w not in windows_ns` を満たす経路を評価する。
    if ref_w not in windows_ns:
        # Use the closest value in the list.
        ref_w = min(windows_ns, key=lambda x: abs(x - ref_w))

    variant_fixed: ChshBest | None = None
    for w_ns in windows_ns:
        # 条件分岐: `w_ns != ref_w` を満たす経路を評価する。
        if w_ns != ref_w:
            continue

        n, sum_prod, _ = _pair_and_accumulate(
            t_a,
            c_a,
            t_b,
            c_b,
            offset_s=offset_s,
            window_s=float(w_ns) * 1e-9,
            encoding=str(args.encoding),
        )
        E_ref = _safe_div(sum_prod, n)
        variant_fixed = _best_chsh(E_ref)
        break

    # 条件分岐: `variant_fixed is None` を満たす経路を評価する。

    if variant_fixed is None:
        raise RuntimeError("failed to determine fixed CHSH variant")

    rows: list[dict[str, object]] = []
    best_overall: dict[str, object] | None = None
    for w_ns in windows_ns:
        window_s = float(w_ns) * 1e-9
        n, sum_prod, pairs_total = _pair_and_accumulate(
            t_a,
            c_a,
            t_b,
            c_b,
            offset_s=offset_s,
            window_s=window_s,
            encoding=str(args.encoding),
        )
        E = _safe_div(sum_prod, n)
        best = _best_chsh(E)
        fixed = _apply_chsh_variant(E, variant_fixed) if np.isfinite(E).all() else float("nan")
        canonical = _canonical_chsh(E) if np.isfinite(E).all() else float("nan")

        rec: dict[str, object] = {
            "window_ns": float(w_ns),
            "offset_ns": float(offset_s * 1e9),
            "pairs_total": int(pairs_total),
            "N00": int(n[0, 0]),
            "N01": int(n[0, 1]),
            "N10": int(n[1, 0]),
            "N11": int(n[1, 1]),
            "E00": float(E[0, 0]) if np.isfinite(E[0, 0]) else None,
            "E01": float(E[0, 1]) if np.isfinite(E[0, 1]) else None,
            "E10": float(E[1, 0]) if np.isfinite(E[1, 0]) else None,
            "E11": float(E[1, 1]) if np.isfinite(E[1, 1]) else None,
            "S_canonical": canonical,
            "S_fixed": fixed,
            "S_fixed_abs": abs(float(fixed)) if np.isfinite(fixed) else None,
            "S_best": float(best.s_value),
            "S_best_abs": float(best.abs_s_value),
            "S_best_swap_a": bool(best.swap_a),
            "S_best_swap_b": bool(best.swap_b),
            "S_best_sign00": int(best.sign_matrix[0][0]),
            "S_best_sign01": int(best.sign_matrix[0][1]),
            "S_best_sign10": int(best.sign_matrix[1][0]),
            "S_best_sign11": int(best.sign_matrix[1][1]),
        }
        rows.append(rec)

        # 条件分岐: `best_overall is None or float(best.abs_s_value) > float(best_overall["S_best_...` を満たす経路を評価する。
        if best_overall is None or float(best.abs_s_value) > float(best_overall["S_best_abs"]):
            best_overall = rec

    assert best_overall is not None

    csv_path = out_dir / f"weihs1998_chsh_sweep__{out_tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[ok] csv: {csv_path}")

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Weihs et al. 1998 time-tag (Zenodo 7185335)",
        "source_dir": str(src_dir),
        "subdir": subdir,
        "run": run,
        "encoding": str(args.encoding),
        "events": {"alice": int(t_a.size), "bob": int(t_b.size)},
        "offset": {**offset_info, "offset_used_s": offset_s},
        "windows_ns": windows_ns,
        "fixed_variant": {
            "ref_window_ns": ref_w,
            "swap_a": bool(variant_fixed.swap_a),
            "swap_b": bool(variant_fixed.swap_b),
            "sign_matrix": variant_fixed.sign_matrix,
        },
        "best_overall": best_overall,
        "repro": {
            "fetch": "python -B scripts/quantum/fetch_weihs1998_zenodo_7185335.py",
            "run": (
                "python -B scripts/quantum/weihs1998_time_tag_reanalysis.py "
                f"--subdir {subdir} --run {run} --encoding {args.encoding} --out-tag {out_tag}"
            ),
        },
        "outputs": {
            "csv": str(csv_path),
        },
    }

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib not available: {e}")
        png_path = None
    else:
        xs = [float(r["window_ns"]) for r in rows]
        ss = [float(r["S_fixed_abs"]) if r["S_fixed_abs"] is not None else float("nan") for r in rows]
        ps = [float(r["pairs_total"]) for r in rows]

        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        ax[0].plot(xs, ss, marker="o", lw=2, label="|S| (fixed CHSH variant)")
        ax[0].axhline(2.0, color="k", ls="--", lw=1, label="local bound |S|=2")
        ax[0].set_ylabel("|S|")
        ax[0].set_title(f"Weihs 1998 (Zenodo 7185335): {subdir}/{run}  offset={offset_s*1e9:.3f} ns")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(loc="best")

        ax[1].plot(xs, ps, marker="o", lw=2, color="tab:blue")
        ax[1].set_ylabel("pairs")
        ax[1].set_xlabel("coincidence window half-width (ns)")
        ax[1].grid(True, alpha=0.3)

        png_path = out_dir / f"weihs1998_chsh_sweep__{out_tag}.png"
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"[ok] png: {png_path}")
        metrics["outputs"]["png"] = str(png_path)

    metrics_path = out_dir / f"weihs1998_chsh_sweep_metrics__{out_tag}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] metrics: {metrics_path}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
