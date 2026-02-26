from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class Params:
    # Event-ready validity (Supplementary: Section E/K in the original release notes).
    event_ready_window_start_channel0_ps: int = 5_426_350
    event_ready_window_start_channel1_ps: int = 5_425_700
    event_ready_window_length_ps: int = 55_000 - 2_550
    event_ready_window_separation_ps: int = 250_000

    # Local readout window (nanoseconds after sync).
    readout_window_start_ns: int = 10_620
    readout_window_length_ns: int = 3_700

    # Invalid-marker veto window (number of sync pulses).
    check_for_invalid_marker_in_past: int = 250

    # Sweep (Figure S3 in the example script)
    start_offset_min_ps: int = -3_000
    start_offset_max_ps: int = 4_000
    start_offset_points: int = 100


def _load_table_from_zip(zip_path: Path, *, member: str) -> tuple[np.ndarray, np.ndarray]:
    # 条件分岐: `not zip_path.exists()` を満たす経路を評価する。
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member, "r") as f:
            ts = np.loadtxt(f, delimiter=",", skiprows=0, usecols=[0], dtype=np.datetime64)

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member, "r") as f:
            data = np.loadtxt(f, delimiter=",", skiprows=0, usecols=np.arange(1, 17), dtype=np.int64)

    return ts, data


@dataclass(frozen=True)
class ChshResult:
    n_trials: int
    k: int
    p_value: float
    correlation_matrix: np.ndarray  # shape (4,4)
    e: np.ndarray  # E00,E01,E10,E11
    e_err: np.ndarray
    s: float
    s_err: float


def _analyze(data: np.ndarray, *, p: Params, start_offset_ps: int = 0) -> ChshResult:
    # Columns in bell_open_data.txt (see release notes / example script).
    event_ready_click1_time = data[:, 2]
    event_ready_click1_channel = data[:, 3]
    event_ready_click2_time = data[:, 4]
    event_ready_click2_channel = data[:, 5]

    random_number_a = data[:, 6]
    random_number_b = data[:, 7]
    readout_click_a_time = data[:, 10]
    readout_click_b_time = data[:, 11]

    click_after_excite_a_time = data[:, 12]
    click_after_excite_b_time = data[:, 13]
    last_invalid_marker_a = data[:, 14]
    last_invalid_marker_b = data[:, 15]

    start0 = p.event_ready_window_start_channel0_ps + int(start_offset_ps)
    start1 = p.event_ready_window_start_channel1_ps + int(start_offset_ps)
    length_ps = p.event_ready_window_length_ps
    sep_ps = p.event_ready_window_separation_ps

    w1_c0 = (start0 <= event_ready_click1_time) & (event_ready_click1_time < (start0 + length_ps)) & (
        event_ready_click1_channel == 0
    )
    w1_c1 = (start1 <= event_ready_click1_time) & (event_ready_click1_time < (start1 + length_ps)) & (
        event_ready_click1_channel == 1
    )
    w1 = w1_c0 | w1_c1

    w2_c0 = (start0 + sep_ps <= event_ready_click2_time) & (
        event_ready_click2_time < (start0 + sep_ps + length_ps)
    ) & (event_ready_click2_channel == 0)
    w2_c1 = (start1 + sep_ps <= event_ready_click2_time) & (
        event_ready_click2_time < (start1 + sep_ps + length_ps)
    ) & (event_ready_click2_channel == 1)
    w2 = w2_c0 | w2_c1

    psi_min = event_ready_click1_channel != event_ready_click2_channel
    event_ready_ok = w1 & w2 & psi_min

    no_invalid_a = (last_invalid_marker_a == 0) | (last_invalid_marker_a > p.check_for_invalid_marker_in_past)
    no_invalid_b = (last_invalid_marker_b == 0) | (last_invalid_marker_b > p.check_for_invalid_marker_in_past)
    no_invalid = no_invalid_a & no_invalid_b

    no_excitation = (click_after_excite_a_time == 0) & (click_after_excite_b_time == 0)

    trial = event_ready_ok & no_invalid & no_excitation

    det_a = (readout_click_a_time > p.readout_window_start_ns) & (
        readout_click_a_time <= p.readout_window_start_ns + p.readout_window_length_ns
    )
    det_b = (readout_click_b_time > p.readout_window_start_ns) & (
        readout_click_b_time <= p.readout_window_start_ns + p.readout_window_length_ns
    )

    a_i = random_number_a
    b_i = random_number_b
    x_i = det_a.astype(np.int8) * 2 - 1
    y_i = det_b.astype(np.int8) * 2 - 1
    t_i = trial.astype(np.int64)

    n = int(np.sum(t_i))
    c_i = t_i * ((-1) ** (a_i * b_i) * (x_i * y_i) + 1) // 2
    k = int(np.sum(c_i))
    tau = 5.4e-6 * 2.0
    ksi = 3.0 / 4.0 + 3.0 * (tau + tau**2)
    p_value = float(1.0 - binom.cdf(k - 1, n, ksi))

    inputs_ab = [(0, 0), (0, 1), (1, 0), (1, 1)]
    outputs_xy = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]

    corr = np.zeros((4, 4), dtype=np.int64)
    e = np.zeros((4,), dtype=float)
    e_err = np.zeros((4,), dtype=float)
    for ii, (a, b) in enumerate(inputs_ab):
        for jj, (x, y) in enumerate(outputs_xy):
            corr[ii, jj] = int(np.sum(t_i & (a_i == a) & (b_i == b) & (x_i == x) & (y_i == y)))

        tot = float(np.sum(corr[ii, :]))
        # 条件分岐: `tot <= 0.0` を満たす経路を評価する。
        if tot <= 0.0:
            e[ii] = float("nan")
            e_err[ii] = float("nan")
            continue

        e[ii] = float((corr[ii, 0] - corr[ii, 1] - corr[ii, 2] + corr[ii, 3]) / tot)
        e_err[ii] = float(np.sqrt((1.0 - e[ii] ** 2) / tot))

    s = float(e[0] + e[1] + e[2] - e[3])
    s_err = float(np.sqrt(np.sum(e_err**2)))

    return ChshResult(
        n_trials=n,
        k=k,
        p_value=p_value,
        correlation_matrix=corr,
        e=e,
        e_err=e_err,
        s=s,
        s_err=s_err,
    )


@dataclass(frozen=True)
class Hensen2016Params:
    # event_ready_window_start_channel0 differs between old/new detector runs.
    event_ready_window_start_channel0_old_detector_ps: int = 5_426_000
    event_ready_window_start_channel0_new_detector_ps: int = 5_426_000 - 700

    event_ready_window_start_channel1_ps: int = 5_425_100
    event_ready_window_length_ps: int = 50_000
    event_ready_window_separation_ps: int = 250_000

    # psi-plus uses a shorter second window to limit after-pulsing.
    event_ready_second_window_length_psi_plus_channel0_ps: int = 4_000
    event_ready_second_window_length_psi_plus_channel1_ps: int = 2_500

    # Local readout window (nanoseconds after sync).
    readout_window_start_ns: int = 10_620
    readout_window_length_ns: int = 3_700

    # Invalid-marker veto window (number of sync pulses).
    check_for_invalid_marker_in_past: int = 250

    # Sweep (use the same range as the 2015 example for comparability).
    start_offset_min_ps: int = -3_000
    start_offset_max_ps: int = 4_000
    start_offset_points: int = 100


@dataclass(frozen=True)
class Hensen2016Result:
    n_trials_total: int
    k_total: int
    p_value: float
    s_psi_plus: float
    s_err_psi_plus: float
    n_trials_psi_plus: int
    s_psi_min: float
    s_err_psi_min: float
    n_trials_psi_min: int
    s_combined: float
    s_err_combined: float


def _analyze_hensen2016(
    data_old: np.ndarray, data_new: np.ndarray, *, p: Hensen2016Params, start_offset_ps: int = 0
) -> Hensen2016Result:
    a_i: list[np.ndarray] = []
    b_i: list[np.ndarray] = []
    x_i: list[np.ndarray] = []
    y_i: list[np.ndarray] = []
    t_i: list[np.ndarray] = []

    for start0, data in [
        (p.event_ready_window_start_channel0_old_detector_ps, data_old),
        (p.event_ready_window_start_channel0_new_detector_ps, data_new),
    ]:
        start0 = int(start0) + int(start_offset_ps)
        start1 = int(p.event_ready_window_start_channel1_ps) + int(start_offset_ps)
        length_ps = int(p.event_ready_window_length_ps)
        sep_ps = int(p.event_ready_window_separation_ps)

        second_len0 = int(p.event_ready_second_window_length_psi_plus_channel0_ps)
        second_len1 = int(p.event_ready_second_window_length_psi_plus_channel1_ps)

        # Columns in bell_open_data_2_*.txt (see the original example script).
        event_ready_click1_time = data[:, 2]
        event_ready_click1_channel = data[:, 3]
        event_ready_click2_time = data[:, 4]
        event_ready_click2_channel = data[:, 5]

        random_number_a = data[:, 6]
        random_number_b = data[:, 7]
        readout_click_a_time = data[:, 10]
        readout_click_b_time = data[:, 11]

        click_after_excite_a_time = data[:, 12]
        click_after_excite_b_time = data[:, 13]
        last_invalid_marker_a = data[:, 14]
        last_invalid_marker_b = data[:, 15]

        w1_c0 = (start0 <= event_ready_click1_time) & (event_ready_click1_time < (start0 + length_ps)) & (
            event_ready_click1_channel == 0
        )
        w1_c1 = (start1 <= event_ready_click1_time) & (event_ready_click1_time < (start1 + length_ps)) & (
            event_ready_click1_channel == 1
        )
        w1 = w1_c0 | w1_c1

        w2_c0_psi_min = (start0 + sep_ps <= event_ready_click2_time) & (
            event_ready_click2_time < (start0 + sep_ps + length_ps)
        ) & (event_ready_click2_channel == 0)
        w2_c1_psi_min = (start1 + sep_ps <= event_ready_click2_time) & (
            event_ready_click2_time < (start1 + sep_ps + length_ps)
        ) & (event_ready_click2_channel == 1)
        w2_psi_min = w2_c0_psi_min | w2_c1_psi_min

        w2_c0_psi_plus = (start0 + sep_ps <= event_ready_click2_time) & (
            event_ready_click2_time < (start0 + sep_ps + second_len0)
        ) & (event_ready_click2_channel == 0)
        w2_c1_psi_plus = (start1 + sep_ps <= event_ready_click2_time) & (
            event_ready_click2_time < (start1 + sep_ps + second_len1)
        ) & (event_ready_click2_channel == 1)
        w2_psi_plus = w2_c0_psi_plus | w2_c1_psi_plus

        psi_min = event_ready_click1_channel != event_ready_click2_channel
        psi_plus = np.logical_not(psi_min)
        event_ready_ok = (w1 & w2_psi_min & psi_min) | (w1 & w2_psi_plus & psi_plus)

        no_invalid_a = (last_invalid_marker_a == 0) | (last_invalid_marker_a > p.check_for_invalid_marker_in_past)
        no_invalid_b = (last_invalid_marker_b == 0) | (last_invalid_marker_b > p.check_for_invalid_marker_in_past)
        no_invalid = no_invalid_a & no_invalid_b

        no_excitation = (click_after_excite_a_time == 0) & (click_after_excite_b_time == 0)

        bell_trial = event_ready_ok & no_invalid & no_excitation

        det_a = (readout_click_a_time > p.readout_window_start_ns) & (
            readout_click_a_time <= p.readout_window_start_ns + p.readout_window_length_ns
        )
        det_b = (readout_click_b_time > p.readout_window_start_ns) & (
            readout_click_b_time <= p.readout_window_start_ns + p.readout_window_length_ns
        )

        a_i.append(random_number_a.astype(np.int64))
        b_i.append(random_number_b.astype(np.int64))
        x_i.append(det_a.astype(np.int8) * 2 - 1)
        y_i.append(det_b.astype(np.int8) * 2 - 1)
        # t_i is {-1,+1,0}: sign indicates psi, 0 means invalid.
        t_i.append(bell_trial.astype(np.int64) * (psi_plus.astype(np.int64) * 2 - 1))

    a = np.concatenate(a_i)
    b = np.concatenate(b_i)
    x = np.concatenate(x_i)
    y = np.concatenate(y_i)
    t = np.concatenate(t_i)

    n = int(np.sum(np.abs(t)))
    # Python2 reference implementation uses integer division for (t+1)/2.
    psi_bit = (t + 1) // 2  # {-1,0,+1} -> {0,0,1}
    c_i = np.abs(t) * ((-1) ** (a * (b + psi_bit)) * (x * y) + 1) // 2
    k = int(np.sum(c_i))
    tau = 5.4e-6 * 2.0
    ksi = 3.0 / 4.0 + 3.0 * (tau + tau**2)
    p_value = float(1.0 - binom.cdf(k - 1, n, ksi))

    inputs_ab = [(0, 0), (0, 1), (1, 0), (1, 1)]
    outputs_xy = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]

    def _corr_and_s(psi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
        corr = np.zeros((4, 4), dtype=np.int64)
        e = np.zeros((4,), dtype=float)
        e_err = np.zeros((4,), dtype=float)
        for ii, (aa, bb) in enumerate(inputs_ab):
            for jj, (xx, yy) in enumerate(outputs_xy):
                corr[ii, jj] = int(np.sum((t == psi) & (a == aa) & (b == bb) & (x == xx) & (y == yy)))

            tot = float(np.sum(corr[ii, :]))
            # 条件分岐: `tot <= 0.0` を満たす経路を評価する。
            if tot <= 0.0:
                e[ii] = float("nan")
                e_err[ii] = float("nan")
                continue

            e[ii] = float((corr[ii, 0] - corr[ii, 1] - corr[ii, 2] + corr[ii, 3]) / tot)
            e_err[ii] = float(np.sqrt((1.0 - e[ii] ** 2) / tot))

        # CHSH S differs by psi (see example script).

        if psi == +1:
            s = float(e[0] + e[1] - e[2] + e[3])
        else:
            s = float(e[0] + e[1] + e[2] - e[3])

        s_err = float(np.sqrt(np.sum(e_err**2)))
        n_trials = int(np.sum(corr))
        return corr, e, e_err, s, s_err, n_trials

    _, _, _, s_plus, s_err_plus, n_plus = _corr_and_s(+1)
    _, _, _, s_min, s_err_min, n_min = _corr_and_s(-1)

    n_tot = int(n_plus + n_min)
    # 条件分岐: `n_tot <= 0` を満たす経路を評価する。
    if n_tot <= 0:
        raise RuntimeError("no valid trials in combined dataset")

    w_plus = float(n_plus) / float(n_tot)
    w_min = float(n_min) / float(n_tot)
    s_combined = float(w_plus * s_plus + w_min * s_min)
    s_err_combined = float(np.sqrt((w_plus**2) * (s_err_plus**2) + (w_min**2) * (s_err_min**2)))

    return Hensen2016Result(
        n_trials_total=n,
        k_total=k,
        p_value=p_value,
        s_psi_plus=s_plus,
        s_err_psi_plus=s_err_plus,
        n_trials_psi_plus=n_plus,
        s_psi_min=s_min,
        s_err_psi_min=s_err_min,
        n_trials_psi_min=n_min,
        s_combined=s_combined,
        s_err_combined=s_err_combined,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reanalyze Delft (Hensen et al.) open data and reproduce CHSH S (2015 Nature or 2016 Sci Rep)."
    )
    parser.add_argument(
        "--profile",
        choices=["hensen2015", "hensen2016_srep30289"],
        default="hensen2015",
        help="Dataset profile (default: hensen2015).",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Path to data.zip (default: data/quantum/sources/<profile>/data.zip).",
    )
    parser.add_argument(
        "--out-tag",
        type=str,
        default=None,
        help="Output tag (default: derived from profile).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = str(args.profile)
    # 条件分岐: `profile == "hensen2015"` を満たす経路を評価する。
    if profile == "hensen2015":
        src_zip = args.zip or (root / "data" / "quantum" / "sources" / "delft_hensen2015" / "data.zip")
        out_tag = args.out_tag or "delft_hensen2015"

        p = Params()
        _, data = _load_table_from_zip(src_zip, member="bell_open_data.txt")

        baseline = _analyze(data, p=p, start_offset_ps=0)

        offsets = np.linspace(p.start_offset_min_ps, p.start_offset_max_ps, p.start_offset_points)
        s_vals = np.zeros_like(offsets, dtype=float)
        s_errs = np.zeros_like(offsets, dtype=float)
        n_vals = np.zeros_like(offsets, dtype=int)
        for i, off in enumerate(offsets.astype(int).tolist()):
            r = _analyze(data, p=p, start_offset_ps=int(off))
            s_vals[i] = r.s
            s_errs[i] = r.s_err
            n_vals[i] = r.n_trials

        out_png = out_dir / f"{out_tag}_chsh.png"
        out_json = out_dir / f"{out_tag}_chsh_metrics.json"
        out_csv = out_dir / f"{out_tag}_chsh_sweep_start_offset.csv"

        lines = ["start_offset_ps,n_trials,S,S_err"]
        for off, n, s, se in zip(offsets.tolist(), n_vals.tolist(), s_vals.tolist(), s_errs.tolist()):
            lines.append(f"{int(round(off))},{int(n)},{s:.6f},{se:.6f}")

        out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 7.2), dpi=150, sharex=True)

        ax1.fill_between(offsets, s_vals - s_errs, s_vals + s_errs, color="#1f77b4", alpha=0.2, lw=0)
        ax1.plot(offsets, s_vals, color="#1f77b4", lw=1.8, label="S(start_offset)")
        ax1.axhline(2.0, color="0.25", ls="--", lw=1.0, label="Bell bound (2)")
        ax1.axhline(2.0 * np.sqrt(2.0), color="0.25", ls=":", lw=1.0, label="2√2 (QM reference)")
        ax1.axhline(baseline.s, color="#ff7f0e", ls="-.", lw=1.0, label=f"baseline S={baseline.s:.3f}")
        ax1.set_ylabel("CHSH S")
        ax1.set_title("Delft (Hensen 2015) open data: CHSH S and event-ready window start sensitivity")
        ax1.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)
        ax1.legend(frameon=True, fontsize=9, loc="upper right")

        ax2.plot(offsets, n_vals, color="#2ca02c", lw=1.8)
        ax2.set_xlabel("event-ready window start offset (ps)")
        ax2.set_ylabel("valid Bell trials (n)")
        ax2.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)

        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)

        metrics = {
            "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "dataset": {
                "profile": profile,
                "source": "Delft loophole-free Bell test (Hensen et al. 2015) open data",
                "zip": str(src_zip),
                "member": "bell_open_data.txt",
            },
            "params": {
                **{k: int(v) for k, v in p.__dict__.items() if k.endswith(("_ps", "_ns")) or k.endswith("_past")},
                "sweep": {
                    "start_offset_min_ps": p.start_offset_min_ps,
                    "start_offset_max_ps": p.start_offset_max_ps,
                    "start_offset_points": p.start_offset_points,
                },
            },
            "baseline": {
                "n_trials": baseline.n_trials,
                "k": baseline.k,
                "p_value": baseline.p_value,
                "correlation_matrix": baseline.correlation_matrix.tolist(),
                "E": baseline.e.tolist(),
                "E_err": baseline.e_err.tolist(),
                "S": baseline.s,
                "S_err": baseline.s_err,
            },
            "sweep_start_offset": {
                "start_offset_ps": offsets.astype(int).tolist(),
                "n_trials": list(map(int, n_vals.tolist())),
                "S": s_vals.tolist(),
                "S_err": s_errs.tolist(),
                "csv": str(out_csv),
            },
            "outputs": {"png": str(out_png), "csv": str(out_csv)},
            "notes": [
                "This script follows the logic in the published bell_open_data_analysis_example.py (converted to reproducible waveP style).",
                "The sweep reproduces the documented sensitivity plot idea (Supplementary Fig. S3).",
            ],
        }
        out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[ok] png : {out_png}")
        print(f"[ok] csv : {out_csv}")
        print(f"[ok] json: {out_json}")
        return

    # 条件分岐: `profile == "hensen2016_srep30289"` を満たす経路を評価する。

    if profile == "hensen2016_srep30289":
        src_zip = args.zip or (root / "data" / "quantum" / "sources" / "delft_hensen2016_srep30289" / "data.zip")
        out_tag = args.out_tag or "delft_hensen2016_srep30289"

        p2 = Hensen2016Params()
        _, data_old = _load_table_from_zip(src_zip, member="bell_open_data_2_old_detector.txt")
        _, data_new = _load_table_from_zip(src_zip, member="bell_open_data_2_new_detector.txt")

        baseline2 = _analyze_hensen2016(data_old, data_new, p=p2, start_offset_ps=0)

        offsets = np.linspace(p2.start_offset_min_ps, p2.start_offset_max_ps, p2.start_offset_points)
        s_vals = np.zeros_like(offsets, dtype=float)
        s_errs = np.zeros_like(offsets, dtype=float)
        n_vals = np.zeros_like(offsets, dtype=int)
        for i, off in enumerate(offsets.astype(int).tolist()):
            r = _analyze_hensen2016(data_old, data_new, p=p2, start_offset_ps=int(off))
            s_vals[i] = r.s_combined
            s_errs[i] = r.s_err_combined
            n_vals[i] = r.n_trials_total

        out_png = out_dir / f"{out_tag}_chsh.png"
        out_json = out_dir / f"{out_tag}_chsh_metrics.json"
        out_csv = out_dir / f"{out_tag}_chsh_sweep_start_offset.csv"

        lines = ["start_offset_ps,n_trials_total,S_combined,S_err_combined"]
        for off, n, s, se in zip(offsets.tolist(), n_vals.tolist(), s_vals.tolist(), s_errs.tolist()):
            lines.append(f"{int(round(off))},{int(n)},{s:.6f},{se:.6f}")

        out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 7.2), dpi=150, sharex=True)

        ax1.fill_between(offsets, s_vals - s_errs, s_vals + s_errs, color="#1f77b4", alpha=0.2, lw=0)
        ax1.plot(offsets, s_vals, color="#1f77b4", lw=1.8, label="S_combined(start_offset)")
        ax1.axhline(2.0, color="0.25", ls="--", lw=1.0, label="Bell bound (2)")
        ax1.axhline(2.0 * np.sqrt(2.0), color="0.25", ls=":", lw=1.0, label="2√2 (QM reference)")
        ax1.axhline(
            baseline2.s_combined, color="#ff7f0e", ls="-.", lw=1.0, label=f"baseline S={baseline2.s_combined:.3f}"
        )
        ax1.set_ylabel("CHSH S (combined)")
        ax1.set_title("Delft (Hensen 2016; Sci Rep 30289) open data: CHSH S and window-start sensitivity")
        ax1.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)
        ax1.legend(frameon=True, fontsize=9, loc="upper right")

        ax2.plot(offsets, n_vals, color="#2ca02c", lw=1.8)
        ax2.set_xlabel("event-ready window start offset (ps)")
        ax2.set_ylabel("valid Bell trials (n)")
        ax2.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)

        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)

        metrics = {
            "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "dataset": {
                "profile": profile,
                "source": "Delft second loophole-free Bell test (Hensen et al. 2016; Sci Rep 6, 30289) open data",
                "zip": str(src_zip),
                "members": ["bell_open_data_2_old_detector.txt", "bell_open_data_2_new_detector.txt"],
            },
            "params": {
                **{k: int(v) for k, v in p2.__dict__.items() if k.endswith(("_ps", "_ns")) or k.endswith("_past")},
                "sweep": {
                    "start_offset_min_ps": p2.start_offset_min_ps,
                    "start_offset_max_ps": p2.start_offset_max_ps,
                    "start_offset_points": p2.start_offset_points,
                },
            },
            "baseline": {
                "n_trials_total": baseline2.n_trials_total,
                "k_total": baseline2.k_total,
                "p_value": baseline2.p_value,
                "psi_plus": {
                    "n_trials": baseline2.n_trials_psi_plus,
                    "S": baseline2.s_psi_plus,
                    "S_err": baseline2.s_err_psi_plus,
                },
                "psi_min": {
                    "n_trials": baseline2.n_trials_psi_min,
                    "S": baseline2.s_psi_min,
                    "S_err": baseline2.s_err_psi_min,
                },
                "combined": {"S": baseline2.s_combined, "S_err": baseline2.s_err_combined},
            },
            "sweep_start_offset": {
                "start_offset_ps": offsets.astype(int).tolist(),
                "n_trials_total": list(map(int, n_vals.tolist())),
                "S_combined": s_vals.tolist(),
                "S_err_combined": s_errs.tolist(),
                "csv": str(out_csv),
            },
            "outputs": {"png": str(out_png), "csv": str(out_csv)},
            "notes": [
                "This script follows bell_open_data_2_analysis_example.py (Sci Rep 30289 dataset) and adapts it to waveP style.",
                "CHSH S is computed for psi-plus and psi-minus separately, then combined (weighted by trial counts).",
                "The sweep applies a common start_offset to both detector subsets and both channels for sensitivity inspection.",
            ],
        }
        out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[ok] png : {out_png}")
        print(f"[ok] csv : {out_csv}")
        print(f"[ok] json: {out_json}")
        return

    raise SystemExit(f"[fail] unknown profile: {profile}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
