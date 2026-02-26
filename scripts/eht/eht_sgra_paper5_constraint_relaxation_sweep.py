#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _get_atomic_fail_unknown(row: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    pass_fail = row.get("pass_fail") or {}
    failed: Set[str] = set()
    unknown: Set[str] = set()
    for k, v in pass_fail.items():
        # 条件分岐: `k in {"All", "EHT", "non_EHT"}` を満たす経路を評価する。
        if k in {"All", "EHT", "non_EHT"}:
            continue

        # 条件分岐: `v in {"Fail", "FAIL", "fail"}` を満たす経路を評価する。

        if v in {"Fail", "FAIL", "fail"}:
            failed.add(str(k))
            continue

        # 条件分岐: `v in {"Pass", "PASS", "pass"}` を満たす経路を評価する。

        if v in {"Pass", "PASS", "pass"}:
            continue

        # 条件分岐: `v in {"--", None, ""}` を満たす経路を評価する。

        if v in {"--", None, ""}:
            unknown.add(str(k))
            continue

        unknown.add(str(k))

    return failed, unknown


def _iter_rows(metrics: Dict[str, Any]) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    tables = (metrics.get("extracted") or {}).get("pass_fail_tables") or {}
    for table_key, table in tables.items():
        rows = table.get("rows") or []
        for idx, row in enumerate(rows):
            yield (str(table_key), int(idx), row)


def _compute_pass_stats(
    rows: List[Dict[str, Any]],
    *,
    relax: Set[str],
) -> Dict[str, Any]:
    rows_n = len(rows)
    pass_n = 0
    unknown_rows_n = 0
    for row in rows:
        failed, unknown = _get_atomic_fail_unknown(row)
        failed = failed - relax
        unknown = unknown - relax
        # 条件分岐: `unknown` を満たす経路を評価する。
        if unknown:
            unknown_rows_n += 1

        # 条件分岐: `not failed and not unknown` を満たす経路を評価する。

        if not failed and not unknown:
            pass_n += 1

    return {
        "rows_n": rows_n,
        "pass_n": pass_n,
        "pass_fraction": (pass_n / rows_n) if rows_n else None,
        "unknown_rows_n": unknown_rows_n,
    }


def _collect_rows_by_table(metrics: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for table_key, _, row in _iter_rows(metrics):
        out.setdefault(table_key, []).append(row)

    return out


def _atomic_keys_in_rows(rows: List[Dict[str, Any]]) -> List[str]:
    keys: Set[str] = set()
    for row in rows:
        pf = row.get("pass_fail") or {}
        for k in pf.keys():
            # 条件分岐: `k in {"All", "EHT", "non_EHT"}` を満たす経路を評価する。
            if k in {"All", "EHT", "non_EHT"}:
                continue

            keys.add(str(k))

    return sorted(keys)


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_in = root / "output" / "private" / "eht" / "eht_sgra_paper5_pass_fraction_tables_metrics.json"
    default_out = root / "output" / "private" / "eht" / "eht_sgra_paper5_constraint_relaxation_sweep_metrics.json"
    default_png = root / "output" / "private" / "eht" / "eht_sgra_paper5_constraint_relaxation_sweep.png"

    ap = argparse.ArgumentParser(description="Compute pass-fraction under constraint-relaxation scenarios (Paper V Pass/Fail tables).")
    ap.add_argument("--in-metrics", type=str, default=str(default_in))
    ap.add_argument("--out", type=str, default=str(default_out))
    ap.add_argument("--out-png", type=str, default=str(default_png))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_metrics = Path(args.in_metrics)
    out_json = Path(args.out)
    out_png = Path(args.out_png)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "inputs": {"metrics_json": str(in_metrics)},
        "extracted": {},
        "derived": {},
        "outputs": {"json": str(out_json), "png": str(out_png)},
    }

    # 条件分岐: `not in_metrics.exists()` を満たす経路を評価する。
    if not in_metrics.exists():
        payload["ok"] = False
        payload["reason"] = "missing_input_metrics_json"
        _write_json(out_json, payload)
        print(f"[warn] missing input; wrote: {out_json}")
        return 0

    metrics = _read_json(in_metrics)
    rows_by_table = _collect_rows_by_table(metrics)
    all_rows: List[Dict[str, Any]] = []
    for rows in rows_by_table.values():
        all_rows.extend(rows)

    atomic_all = _atomic_keys_in_rows(all_rows)
    payload["extracted"] = {
        "tables_n": len(rows_by_table),
        "rows_total_n": len(all_rows),
        "atomic_constraints": atomic_all,
    }

    scenarios: Dict[str, Set[str]] = {
        "baseline": set(),
        "relax_M3": {"M3"},
        "relax_F_2um": {"F_2um"},
        "relax_M3_plus_F_2um": {"M3", "F_2um"},
    }

    derived: Dict[str, Any] = {"global": {}, "by_table": {}}
    for name, relax in scenarios.items():
        derived["global"][name] = _compute_pass_stats(all_rows, relax=relax)
        derived["global"][name]["relax"] = sorted(relax)

    relax_single: List[Dict[str, Any]] = []
    for k in atomic_all:
        st = _compute_pass_stats(all_rows, relax={k})
        st["relax"] = [k]
        relax_single.append(st)

    relax_single_sorted = sorted(relax_single, key=lambda x: (x.get("pass_n") or 0), reverse=True)
    derived["global"]["relax_single_ranked"] = relax_single_sorted[:15]

    for table_key, rows in rows_by_table.items():
        dtab: Dict[str, Any] = {"rows_n": len(rows), "atomic_constraints": _atomic_keys_in_rows(rows), "scenarios": {}}
        for name, relax in scenarios.items():
            dtab["scenarios"][name] = _compute_pass_stats(rows, relax=relax)
            dtab["scenarios"][name]["relax"] = sorted(relax)

        derived["by_table"][table_key] = dtab

    payload["derived"] = derived

    # 条件分岐: `not bool(args.no_plot)` を満たす経路を評価する。
    if not bool(args.no_plot):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_png.parent.mkdir(parents=True, exist_ok=True)

            labels = ["baseline", "relax_M3", "relax_F_2um", "relax_M3_plus_F_2um"]
            vals = [(derived["global"][k]["pass_fraction"] or 0.0) for k in labels]

            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.bar(labels, vals, color=["#777777", "#1f77b4", "#ff7f0e", "#2ca02c"])
            ax.set_ylim(0.0, min(1.0, max(vals) * 1.25 + 1e-6))
            ax.set_ylabel("pass fraction (atomic constraints)")
            ax.set_title("Paper V Pass/Fail tables: relaxation sweep (global)")
            ax.tick_params(axis="x", labelrotation=15)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

            fig.tight_layout()
            fig.savefig(out_png, dpi=160)
            plt.close(fig)
        except Exception:
            pass

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper5_constraint_relaxation_sweep",
                "inputs": {"metrics_json": str(in_metrics.relative_to(root)).replace("\\", "/")},
                "outputs": [
                    str(out_json.relative_to(root)).replace("\\", "/"),
                    str(out_png.relative_to(root)).replace("\\", "/"),
                ],
                "metrics": {
                    "ok": bool(payload.get("ok")),
                    "rows_total_n": int(payload.get("extracted", {}).get("rows_total_n") or 0),
                    "baseline_pass_n": int(payload.get("derived", {}).get("global", {}).get("baseline", {}).get("pass_n") or 0),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    # 条件分岐: `out_png.exists()` を満たす経路を評価する。
    if out_png.exists():
        print(f"[ok] png : {out_png}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
