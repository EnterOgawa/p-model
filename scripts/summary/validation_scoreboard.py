from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

_C_M_PER_S = 299_792_458.0


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


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
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_fmt_float` の入出力契約と処理意図を定義する。

def _fmt_float(x: Optional[float], *, digits: int = 4) -> str:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return ""

    # 条件分岐: `x == 0.0` を満たす経路を評価する。

    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# 関数: `_fmt_pct` の入出力契約と処理意図を定義する。

def _fmt_pct(x: Optional[float], *, digits: int = 2) -> str:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return ""

    return f"{x * 100.0:.{digits}f}".rstrip("0").rstrip(".") + "%"


# 関数: `_status_from_abs_sigma` の入出力契約と処理意図を定義する。

def _status_from_abs_sigma(abs_sigma: Optional[float]) -> str:
    # 条件分岐: `abs_sigma is None` を満たす経路を評価する。
    if abs_sigma is None:
        return "info"

    # 条件分岐: `abs_sigma <= 1.0` を満たす経路を評価する。

    if abs_sigma <= 1.0:
        return "ok"

    # 条件分岐: `abs_sigma <= 2.0` を満たす経路を評価する。

    if abs_sigma <= 2.0:
        return "mixed"

    return "ng"


# 関数: `_status_color` の入出力契約と処理意図を定義する。

def _status_color(status: str) -> str:
    # 条件分岐: `status == "ok"` を満たす経路を評価する。
    if status == "ok":
        return "#2ca02c"

    # 条件分岐: `status == "mixed"` を満たす経路を評価する。

    if status == "mixed":
        # Yellow (needs further verification)
        return "#f1c232"

    # 条件分岐: `status == "ng"` を満たす経路を評価する。

    if status == "ng":
        return "#d62728"

    return "#7f7f7f"


# 関数: `_status_label` の入出力契約と処理意図を定義する。

def _status_label(status: str) -> str:
    return {"ok": "OK", "mixed": "要改善", "ng": "不一致", "info": "参考"}.get(status, status)


# 関数: `_status_from_gate` の入出力契約と処理意図を定義する。

def _status_from_gate(overall_status: str) -> str:
    s = (overall_status or "").strip().lower()
    # 条件分岐: `s == "pass"` を満たす経路を評価する。
    if s == "pass":
        return "ok"

    # 条件分岐: `s in {"watch", "pending"}` を満たす経路を評価する。

    if s in {"watch", "pending"}:
        return "mixed"

    # 条件分岐: `s in {"reject", "fail", "failed", "hard_reject"}` を満たす経路を評価する。

    if s in {"reject", "fail", "failed", "hard_reject"}:
        return "ng"

    return "info"


# 関数: `_status_rates` の入出力契約と処理意図を定義する。

def _status_rates(counts: Optional[Dict[str, int]]) -> Optional[Dict[str, Any]]:
    # 条件分岐: `not counts` を満たす経路を評価する。
    if not counts:
        return None

    keys = ("ok", "mixed", "ng", "info")
    total = sum(int(counts.get(k, 0) or 0) for k in keys)
    # 条件分岐: `total <= 0` を満たす経路を評価する。
    if total <= 0:
        return None

    rates = {k: (int(counts.get(k, 0) or 0) / float(total)) for k in keys}
    ok = int(counts.get("ok", 0) or 0)
    mixed = int(counts.get("mixed", 0) or 0)
    return {
        "total": total,
        "counts": {k: int(counts.get(k, 0) or 0) for k in keys},
        "rates": rates,
        "ok_rate": rates["ok"],
        "ok_or_mixed_rate": rates["ok"] + rates["mixed"],
        "weighted_rate_ok_plus_half_mixed": (ok + 0.5 * mixed) / float(total),
    }


# クラス: `ScoreRow` の責務と境界条件を定義する。

@dataclass(frozen=True)
class ScoreRow:
    id: str
    label: str
    status: str
    score: Optional[float]
    metric: str
    detail: str
    sources: List[str]
    score_kind: str = ""
    score_raw: Optional[float] = None

    # 関数: `to_dict` の入出力契約と処理意図を定義する。
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "status_label": _status_label(self.status),
            "score": self.score,
            "score_raw": self.score if self.score_raw is None else self.score_raw,
            "score_kind": self.score_kind,
            "metric": self.metric,
            "detail": self.detail,
            "sources": list(self.sources),
        }


# 関数: `_maybe_float` の入出力契約と処理意図を定義する。

def _maybe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    # 条件分岐: `math.isnan(v) or math.isinf(v)` を満たす経路を評価する。

    if math.isnan(v) or math.isinf(v):
        return None

    return v


# 関数: `_first_existing` の入出力契約と処理意図を定義する。

def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        # 条件分岐: `path.exists()` を満たす経路を評価する。
        if path.exists():
            return path

    return None


# 関数: `_with_status` の入出力契約と処理意図を定義する。

def _with_status(row: "ScoreRow", status: str) -> "ScoreRow":
    return ScoreRow(
        id=row.id,
        label=row.label,
        status=status,
        score=row.score,
        metric=row.metric,
        detail=row.detail,
        sources=list(row.sources),
        score_kind=row.score_kind,
        score_raw=row.score if row.score_raw is None else row.score_raw,
    )


# 関数: `_with_score` の入出力契約と処理意図を定義する。

def _with_score(row: "ScoreRow", score: Optional[float]) -> "ScoreRow":
    return ScoreRow(
        id=row.id,
        label=row.label,
        status=row.status,
        score=score,
        metric=row.metric,
        detail=row.detail,
        sources=list(row.sources),
        score_kind=row.score_kind,
        score_raw=row.score if row.score_raw is None else row.score_raw,
    )


# 関数: `_score_lower_better` の入出力契約と処理意図を定義する。

def _score_lower_better(value: Optional[float], *, ok_max: float, mixed_max: float) -> Optional[float]:
    """Map a 'lower is better' metric to a z-like score axis.

    Score axis meaning:
      - 0: ideal
      - 1: OK threshold (green)
      - 2: mixed threshold (yellow)
      - >2: NG (red)
    """
    # 条件分岐: `value is None` を満たす経路を評価する。
    if value is None:
        return None

    # 条件分岐: `ok_max <= 0 or mixed_max <= ok_max` を満たす経路を評価する。

    if ok_max <= 0 or mixed_max <= ok_max:
        return None

    x = float(value)
    # 条件分岐: `x <= ok_max` を満たす経路を評価する。
    if x <= ok_max:
        return x / ok_max

    # 条件分岐: `x <= mixed_max` を満たす経路を評価する。

    if x <= mixed_max:
        return 1.0 + (x - ok_max) / (mixed_max - ok_max)
    # extend linearly beyond mixed threshold (scale by mixed_max for readability)

    return 2.0 + (x - mixed_max) / max(1e-12, mixed_max)


# 関数: `_score_higher_better` の入出力契約と処理意図を定義する。

def _score_higher_better(value: Optional[float], *, ok_min: float, mixed_min: float, ideal: float = 1.0) -> Optional[float]:
    """Map a 'higher is better' metric (e.g., corr, R^2) to a z-like score axis."""
    # 条件分岐: `value is None` を満たす経路を評価する。
    if value is None:
        return None

    # 条件分岐: `not (ideal > ok_min > mixed_min)` を満たす経路を評価する。

    if not (ideal > ok_min > mixed_min):
        return None

    x = float(value)
    # 条件分岐: `x >= ok_min` を満たす経路を評価する。
    if x >= ok_min:
        # ideal -> 0, ok_min -> 1
        return (ideal - x) / (ideal - ok_min)

    # 条件分岐: `x >= mixed_min` を満たす経路を評価する。

    if x >= mixed_min:
        # ok_min -> 1, mixed_min -> 2
        return 1.0 + (ok_min - x) / (ok_min - mixed_min)
    # below mixed threshold

    return 2.0 + (mixed_min - x) / max(1e-12, mixed_min)


# 関数: `_load_llr_row` の入出力契約と処理意図を定義する。

def _load_llr_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "private" / "llr" / "batch" / "llr_batch_summary.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    j = _read_json(path)
    median = j.get("median_rms_ns") or {}
    best = _maybe_float(median.get("station_reflector_tropo_tide"))
    # 条件分岐: `best is None` を満たす経路を評価する。
    if best is None:
        return None

    range_m = (_C_M_PER_S * (best * 1e-9)) / 2.0
    # LLR uses meters as an absolute scale; treat <=1m as OK, <=2m as mixed.
    status = _status_from_abs_sigma(range_m)
    # 条件分岐: `range_m is not None` を満たす経路を評価する。
    if range_m is not None:
        # 条件分岐: `range_m <= 1.0` を満たす経路を評価する。
        if range_m <= 1.0:
            status = "ok"
        # 条件分岐: 前段条件が不成立で、`range_m <= 2.0` を追加評価する。
        elif range_m <= 2.0:
            status = "mixed"
        else:
            status = "ng"

    score = _score_lower_better(range_m, ok_max=1.0, mixed_max=2.0)
    return ScoreRow(
        id="llr",
        label="LLR（月レーザー測距）",
        status=status,
        score=score,
        metric=f"残差RMS（典型）≈{_fmt_float(range_m, digits=3)} m",
        detail="EDC CRD Normal Point（station×reflector, SR+Tropo+Tide+Ocean）",
        sources=[str(path).replace("\\", "/")],
        score_kind="llr_rms_m",
    )


# 関数: `_load_llr_nglr1_row` の入出力契約と処理意図を定義する。

def _load_llr_nglr1_row(root: Path) -> Optional[ScoreRow]:
    metrics_path = root / "output" / "private" / "llr" / "batch" / "llr_batch_metrics.csv"
    # 条件分岐: `not metrics_path.exists()` を満たす経路を評価する。
    if not metrics_path.exists():
        return None

    coverage_path = root / "output" / "private" / "llr" / "batch" / "llr_data_coverage.csv"

    import csv
    import statistics

    rms_list: List[float] = []
    n_total = 0
    stations: set[str] = set()

    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 条件分岐: `(r.get("target") or "").strip().lower() != "nglr1"` を満たす経路を評価する。
            if (r.get("target") or "").strip().lower() != "nglr1":
                continue

            stations.add((r.get("station") or "").strip())
            try:
                n_total += int(float(r.get("n") or 0))
            except Exception:
                pass

            v = _maybe_float(r.get("rms_sr_tropo_tide_ns"))
            # 条件分岐: `v is not None` を満たす経路を評価する。
            if v is not None:
                rms_list.append(v)

    # 条件分岐: `not rms_list` を満たす経路を評価する。

    if not rms_list:
        return None

    try:
        rms_med = float(statistics.median(rms_list))
    except Exception:
        return None

    range_m = (_C_M_PER_S * (rms_med * 1e-9)) / 2.0
    # 条件分岐: `range_m <= 1.0` を満たす経路を評価する。
    if range_m <= 1.0:
        status = "ok"
    # 条件分岐: 前段条件が不成立で、`range_m <= 2.0` を追加評価する。
    elif range_m <= 2.0:
        status = "mixed"
    else:
        status = "ng"

    score = _score_lower_better(range_m, ok_max=1.0, mixed_max=2.0)

    stations_s = ",".join(sorted([s for s in stations if s])) or "-"
    extra_cov: List[str] = []
    # 条件分岐: `coverage_path.exists()` を満たす経路を評価する。
    if coverage_path.exists():
        try:
            with open(coverage_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    # 条件分岐: `(r.get("target") or "").strip().lower() != "nglr1"` を満たす経路を評価する。
                    if (r.get("target") or "").strip().lower() != "nglr1":
                        continue

                    st = (r.get("station") or "").strip()
                    # 条件分岐: `not st or st in stations` を満たす経路を評価する。
                    if not st or st in stations:
                        continue

                    try:
                        n_unique = int(float(r.get("n_unique") or 0))
                    except Exception:
                        n_unique = 0

                    try:
                        min_req = int(float(r.get("min_points_required") or 0))
                    except Exception:
                        min_req = 0

                    # 条件分岐: `n_unique > 0` を満たす経路を評価する。

                    if n_unique > 0:
                        extra_cov.append(f"{st}: n={n_unique} (<{min_req})")
        except Exception:
            extra_cov = []

    detail = f"EDC CRD Normal Point（target=nglr1; stations={stations_s}; SR+Tropo+Tide+Ocean）"
    # 条件分岐: `extra_cov` を満たす経路を評価する。
    if extra_cov:
        detail += f"（除外: {', '.join(extra_cov)}）"

    metric = f"残差RMS（典型）≈{_fmt_float(range_m, digits=3)} m"
    # 条件分岐: `n_total > 0` を満たす経路を評価する。
    if n_total > 0:
        metric += f"（N={n_total}）"

    return ScoreRow(
        id="llr_nglr1",
        label="LLR（月レーザー測距：NGLR-1）",
        status=status,
        score=score,
        metric=metric,
        detail=detail,
        sources=[
            str(metrics_path).replace("\\", "/"),
            *([str(coverage_path).replace("\\", "/")] if coverage_path.exists() else []),
        ],
        score_kind="llr_rms_m",
    )


# 関数: `_load_cassini_row` の入出力契約と処理意図を定義する。

def _load_cassini_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "private" / "cassini" / "cassini_fig2_metrics.csv"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    import csv

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({str(k): ("" if v is None else str(v)) for k, v in dict(r).items()})

    # 条件分岐: `not rows` を満たす経路を評価する。

    if not rows:
        return None

    pick = None
    for r in rows:
        # 条件分岐: `(r.get("window") or "").strip() == "-10 to +10 days"` を満たす経路を評価する。
        if (r.get("window") or "").strip() == "-10 to +10 days":
            pick = r
            break

    # 条件分岐: `pick is None` を満たす経路を評価する。

    if pick is None:
        pick = rows[0]

    rmse = _maybe_float(pick.get("rmse"))
    corr = _maybe_float(pick.get("corr"))

    status = "info"
    # 条件分岐: `corr is not None` を満たす経路を評価する。
    if corr is not None:
        # 条件分岐: `corr >= 0.95` を満たす経路を評価する。
        if corr >= 0.95:
            status = "ok"
        # 条件分岐: 前段条件が不成立で、`corr >= 0.90` を追加評価する。
        elif corr >= 0.90:
            status = "mixed"
        else:
            status = "ng"

    score = _score_higher_better(corr, ok_min=0.95, mixed_min=0.90, ideal=1.0)

    metric = f"corr={_fmt_float(corr, digits=3)}"
    # 条件分岐: `rmse is not None` を満たす経路を評価する。
    if rmse is not None:
        metric += f", RMSE={rmse:.2e}"

    return ScoreRow(
        id="cassini",
        label="Cassini（太陽会合）",
        status=status,
        score=score,
        metric=metric,
        detail="ドップラー y(t) の形状一致（±10日窓）",
        sources=[str(path).replace("\\", "/")],
        score_kind="cassini_corr",
    )


# 関数: `_load_viking_row` の入出力契約と処理意図を定義する。

def _load_viking_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "private" / "viking" / "viking_shapiro_result.csv"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    import csv

    max_us: Optional[float] = None
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            us = _maybe_float(r.get("shapiro_delay_us"))
            # 条件分岐: `us is None` を満たす経路を評価する。
            if us is None:
                continue

            # 条件分岐: `max_us is None or us > max_us` を満たす経路を評価する。

            if max_us is None or us > max_us:
                max_us = us

    # 条件分岐: `max_us is None` を満たす経路を評価する。

    if max_us is None:
        return None

    # This is a sanity check range (non-decisive, literature typical range).

    status = "info"
    # 条件分岐: `200.0 <= max_us <= 250.0` を満たす経路を評価する。
    if 200.0 <= max_us <= 250.0:
        status = "ok"
    # 条件分岐: 前段条件が不成立で、`150.0 <= max_us <= 300.0` を追加評価する。
    elif 150.0 <= max_us <= 300.0:
        status = "mixed"
    else:
        status = "ng"

    score = {"ok": 0.5, "mixed": 1.5, "ng": 3.0}.get(status)

    return ScoreRow(
        id="viking",
        label="Viking（太陽会合）",
        status=status,
        score=score,
        metric=f"Shapiroピーク≈{_fmt_float(max_us, digits=2)} μs",
        detail="往復Shapiro遅延（最大値の目安）",
        sources=[str(path).replace("\\", "/")],
        score_kind="viking_lit_range",
    )


# 関数: `_load_mercury_row` の入出力契約と処理意図を定義する。

def _load_mercury_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "private" / "mercury" / "mercury_precession_metrics.json",
            root / "output" / "public" / "mercury" / "mercury_precession_metrics.json",
            root / "output" / "mercury" / "mercury_precession_metrics.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    reference = _maybe_float(j.get("reference_arcsec_century"))
    pmodel = _maybe_float((((j.get("simulation_physical") or {}).get("pmodel") or {}).get("arcsec_per_century")))
    # 条件分岐: `pmodel is None or reference is None or reference == 0` を満たす経路を評価する。
    if pmodel is None or reference is None or reference == 0:
        return None

    diff = pmodel - reference
    rel = diff / reference
    abs_rel = abs(rel)
    abs_pct = abs_rel * 100.0

    status = "info"
    # 条件分岐: `abs_rel <= 0.001` を満たす経路を評価する。
    if abs_rel <= 0.001:
        status = "ok"
    # 条件分岐: 前段条件が不成立で、`abs_rel <= 0.01` を追加評価する。
    elif abs_rel <= 0.01:
        status = "mixed"
    else:
        status = "ng"

    score = _score_lower_better(abs_pct, ok_max=0.1, mixed_max=1.0)

    return ScoreRow(
        id="mercury",
        label="Mercury（近日点移動）",
        status=status,
        score=score,
        metric=f"差≈{_fmt_float(diff, digits=5)} ″/世紀（{_fmt_pct(abs_rel, digits=2)}）",
        detail="実Cでの近日点移動（角秒/世紀）",
        sources=[str(path).replace("\\", "/")],
        score_kind="mercury_abs_percent",
    )


# 関数: `_load_gps_row` の入出力契約と処理意図を定義する。

def _load_gps_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "private" / "gps" / "gps_compare_metrics.json",
            root / "output" / "public" / "gps" / "gps_compare_metrics.json",
            root / "output" / "gps" / "gps_compare_metrics.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    met = j.get("metrics") if isinstance(j.get("metrics"), dict) else {}
    brdc = _maybe_float(met.get("brdc_rms_ns_median"))
    pmodel = _maybe_float(met.get("pmodel_rms_ns_median"))
    # 条件分岐: `brdc is None or pmodel is None or brdc <= 0` を満たす経路を評価する。
    if brdc is None or pmodel is None or brdc <= 0:
        return None

    ratio = pmodel / brdc
    diff = ratio - 1.0

    # Here "OK" means P-model RMS <= BRDC; otherwise treat as improvement-needed.
    status = "info"
    # 条件分岐: `ratio <= 1.0` を満たす経路を評価する。
    if ratio <= 1.0:
        status = "ok"
    # 条件分岐: 前段条件が不成立で、`ratio <= 1.10` を追加評価する。
    elif ratio <= 1.10:
        status = "mixed"
    else:
        status = "ng"

    score = _score_lower_better(ratio, ok_max=1.0, mixed_max=1.10)

    return ScoreRow(
        id="gps",
        label="GPS（衛星時計）",
        status=status,
        score=score,
        metric=f"中央値RMS: P/BRDC={_fmt_float(ratio, digits=3)}（+{_fmt_pct(diff, digits=1)}）",
        detail="観測（準実測）= IGS Final CLK/SP3（RMSが小さいほど良い）",
        sources=[str(path).replace("\\", "/")],
        score_kind="gps_rms_ratio",
    )


# 関数: `_load_solar_deflection_row` の入出力契約と処理意図を定義する。

def _load_solar_deflection_row(root: Path) -> Optional[ScoreRow]:
    metrics_path = root / "output" / "private" / "theory" / "solar_light_deflection_metrics.json"
    frozen_path = root / "output" / "private" / "theory" / "frozen_parameters.json"
    # 条件分岐: `not metrics_path.exists() or not frozen_path.exists()` を満たす経路を評価する。
    if not metrics_path.exists() or not frozen_path.exists():
        return None

    m = (_read_json(metrics_path).get("metrics") or {})
    obs_gamma = _maybe_float(m.get("observed_gamma_best"))
    obs_sigma = _maybe_float(m.get("observed_gamma_best_sigma"))
    # 条件分岐: `obs_gamma is None or obs_sigma is None or obs_sigma <= 0` を満たす経路を評価する。
    if obs_gamma is None or obs_sigma is None or obs_sigma <= 0:
        return None

    beta = _maybe_float(_read_json(frozen_path).get("beta"))
    # 条件分岐: `beta is None` を満たす経路を評価する。
    if beta is None:
        beta = 1.0

    gamma_pred = 2.0 * beta - 1.0
    z = (gamma_pred - obs_gamma) / obs_sigma
    abs_z = abs(z)
    return ScoreRow(
        id="solar_deflection",
        label="光偏向（太陽）",
        status=_status_from_abs_sigma(abs_z),
        score=abs_z,
        metric=f"|z|={_fmt_float(abs_z, digits=3)}（PPN γ）",
        detail=f"観測γ={_fmt_float(obs_gamma, digits=8)}±{_fmt_float(obs_sigma, digits=3)} vs 予測γ={_fmt_float(gamma_pred, digits=8)}",
        sources=[str(metrics_path).replace("\\", "/"), str(frozen_path).replace("\\", "/")],
        score_kind="abs_z",
    )


# 関数: `_load_redshift_row` の入出力契約と処理意図を定義する。

def _load_redshift_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "private" / "theory" / "gravitational_redshift_experiments.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []

    zs: List[float] = []
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        eps = _maybe_float(r.get("epsilon"))
        sig = _maybe_float(r.get("sigma"))
        # 条件分岐: `eps is None or sig is None or sig <= 0` を満たす経路を評価する。
        if eps is None or sig is None or sig <= 0:
            continue

        z = (0.0 - eps) / sig
        zs.append(abs(z))

    # 条件分岐: `not zs` を満たす経路を評価する。

    if not zs:
        return None

    worst = max(zs)
    return ScoreRow(
        id="redshift",
        label="GP-A / Galileo（重力赤方偏移）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=f"最大|z|={_fmt_float(worst, digits=3)}（ε=0）",
        detail="複数実験の最大|z|（P-modelの弱場一次はGRと同じ ε=0）",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


# 関数: `_load_cosmology_distance_duality_row` の入出力契約と処理意図を定義する。

def _load_cosmology_distance_duality_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "private" / "cosmology" / "cosmology_distance_duality_constraints_metrics.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        return None

    best = None
    best_sig = None
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        sig = _maybe_float(r.get("epsilon0_sigma"))
        # 条件分岐: `sig is None or sig <= 0` を満たす経路を評価する。
        if sig is None or sig <= 0:
            continue

        # 条件分岐: `best is None or best_sig is None or sig < best_sig` を満たす経路を評価する。

        if best is None or best_sig is None or sig < best_sig:
            best = r
            best_sig = sig

    r0 = best if isinstance(best, dict) else (rows[0] if isinstance(rows[0], dict) else {})
    r0_id = str(r0.get("id") or "")
    z_pbg = _maybe_float(r0.get("z_pbg_static"))
    # 条件分岐: `z_pbg is None` を満たす経路を評価する。
    if z_pbg is None:
        return None

    delta_eps = _maybe_float(r0.get("epsilon0_extra_needed_to_match_obs"))
    extra_eta_z1 = _maybe_float(r0.get("extra_eta_factor_needed_z1"))
    delta_mu_z1 = _maybe_float(r0.get("delta_distance_modulus_mag_z1"))
    tau_z1 = _maybe_float(r0.get("tau_equivalent_dimming_z1"))
    abs_z = abs(z_pbg)

    metric = f"|z|={_fmt_float(abs_z, digits=2)}（Pbg静的）"

    # Optional: incorporate category-level systematic width (sigma_cat) from Step 16.5.3/16.5.4.
    # This reduces the risk of over-interpreting a single tight constraint when multiple modeling choices exist.
    z_sys = None
    sigma_cat = None
    min_no_bao_abs_z_sys = None
    min_no_bao_label_sys = None
    sys_path = root / "output" / "private" / "cosmology" / "cosmology_distance_duality_systematics_envelope_metrics.json"
    # 条件分岐: `sys_path.exists()` を満たす経路を評価する。
    if sys_path.exists():
        try:
            sj = _read_json(sys_path)
            srows = sj.get("rows") if isinstance(sj.get("rows"), list) else []
            for sr in srows:
                # 条件分岐: `not isinstance(sr, dict)` を満たす経路を評価する。
                if not isinstance(sr, dict):
                    continue

                # 条件分岐: `r0_id and str(sr.get("id") or "") == r0_id` を満たす経路を評価する。

                if r0_id and str(sr.get("id") or "") == r0_id:
                    z_sys = _maybe_float(sr.get("abs_z_with_category_sys"))
                    sigma_cat = _maybe_float(sr.get("sigma_sys_category"))
                    break

            for sr in srows:
                # 条件分岐: `not isinstance(sr, dict)` を満たす経路を評価する。
                if not isinstance(sr, dict):
                    continue

                # 条件分岐: `bool(sr.get("uses_bao", False))` を満たす経路を評価する。

                if bool(sr.get("uses_bao", False)):
                    continue

                az = _maybe_float(sr.get("abs_z_with_category_sys"))
                # 条件分岐: `az is None` を満たす経路を評価する。
                if az is None:
                    continue

                # 条件分岐: `min_no_bao_abs_z_sys is None or az < min_no_bao_abs_z_sys` を満たす経路を評価する。

                if min_no_bao_abs_z_sys is None or az < min_no_bao_abs_z_sys:
                    min_no_bao_abs_z_sys = az
                    min_no_bao_label_sys = str(sr.get("short_label") or sr.get("id") or "")
        except Exception:
            pass

    # 条件分岐: `z_sys is not None` を満たす経路を評価する。

    if z_sys is not None:
        metric += f" / σ_cat込み|z|={_fmt_float(abs(z_sys), digits=2)}"
        # 条件分岐: `sigma_cat is not None` を満たす経路を評価する。
        if sigma_cat is not None:
            metric += f"（σ_cat={_fmt_float(sigma_cat, digits=3)}）"

    # 条件分岐: `delta_eps is not None and extra_eta_z1 is not None` を満たす経路を評価する。

    if delta_eps is not None and extra_eta_z1 is not None:
        metric += f" / z=1でD_L×{_fmt_float(extra_eta_z1, digits=2)}（Δε={_fmt_float(delta_eps, digits=3)}）"
        # 条件分岐: `delta_mu_z1 is not None and tau_z1 is not None` を満たす経路を評価する。
        if delta_mu_z1 is not None and tau_z1 is not None:
            metric += f"（Δμ={_fmt_float(delta_mu_z1, digits=2)}mag, τ={_fmt_float(tau_z1, digits=2)}）"

    # Also report the best (least rejecting) non-BAO constraint as an intuition (DDR depends on distance-indicator assumptions).

    min_no_bao_abs_z = None
    min_no_bao_label = None
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        # 条件分岐: `bool(r.get("uses_bao", False))` を満たす経路を評価する。

        if bool(r.get("uses_bao", False)):
            continue

        z = _maybe_float(r.get("z_pbg_static"))
        # 条件分岐: `z is None` を満たす経路を評価する。
        if z is None:
            continue

        az = abs(z)
        # 条件分岐: `min_no_bao_abs_z is None or az < min_no_bao_abs_z` を満たす経路を評価する。
        if min_no_bao_abs_z is None or az < min_no_bao_abs_z:
            min_no_bao_abs_z = az
            min_no_bao_label = str(r.get("short_label") or r.get("id") or "")

    # 条件分岐: `min_no_bao_abs_z is not None` を満たす経路を評価する。

    if min_no_bao_abs_z is not None:
        metric += f" / no-BAO最小abs(z)={_fmt_float(min_no_bao_abs_z, digits=2)}"
        # 条件分岐: `min_no_bao_abs_z_sys is not None` を満たす経路を評価する。
        if min_no_bao_abs_z_sys is not None:
            metric += f"→{_fmt_float(min_no_bao_abs_z_sys, digits=2)}"

        metric += f"（{min_no_bao_label}）"
        # 条件分岐: `min_no_bao_abs_z_sys is not None and min_no_bao_label_sys and min_no_bao_labe...` を満たす経路を評価する。
        if min_no_bao_abs_z_sys is not None and min_no_bao_label_sys and min_no_bao_label_sys != min_no_bao_label:
            metric += f" / （σ_cat込み最小: {min_no_bao_label_sys}）"

    return ScoreRow(
        id="cosmo_ddr",
        label="宇宙論（距離二重性）",
        status=_status_from_abs_sigma(abs_z),
        score=abs_z,
        metric=metric,
        detail="距離二重性（DDR）は距離推定I/Fに膨張側(1+z)が埋め込まれているかに強く依存する。ここでは公表ε0に対する P-model最小（ε0=-1）との差を“前提監査の診断”として表示する（直ちに物理棄却とは解釈しない）。",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


# 関数: `_load_cosmology_tolman_row` の入出力契約と処理意図を定義する。

def _load_cosmology_tolman_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "private" / "cosmology" / "cosmology_tolman_surface_brightness_constraints_metrics.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    zs: List[float] = []
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        z_pbg = _maybe_float(r.get("z_pbg_static"))
        # 条件分岐: `z_pbg is None` を満たす経路を評価する。
        if z_pbg is None:
            continue

        zs.append(abs(z_pbg))

    # 条件分岐: `not zs` を満たす経路を評価する。

    if not zs:
        return None

    worst = max(zs)
    return ScoreRow(
        id="cosmo_tolman",
        label="宇宙論（Tolman表面輝度）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=f"最大|z|={_fmt_float(worst, digits=2)}（Pbg静的）",
        detail="Tolman表面輝度の一次ソース制約（進化が系統。ここでは“静的P最小(n=2)”との差のみを示す）",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


# 関数: `_load_cosmology_independent_probes_row` の入出力契約と処理意図を定義する。

def _load_cosmology_independent_probes_row(root: Path) -> Optional[ScoreRow]:
    p_sn = root / "output" / "private" / "cosmology" / "cosmology_sn_time_dilation_constraints_metrics.json"
    p_tz = root / "output" / "private" / "cosmology" / "cosmology_cmb_temperature_scaling_constraints_metrics.json"

    zs: List[Tuple[str, float]] = []
    sources: List[str] = []

    # 条件分岐: `p_sn.exists()` を満たす経路を評価する。
    if p_sn.exists():
        try:
            sn = _read_json(p_sn)
            r = (sn.get("rows") or [None])[0] or {}
            z = _maybe_float(r.get("z_frw"))
            # 条件分岐: `z is not None` を満たす経路を評価する。
            if z is not None:
                zs.append(("SN time dilation", abs(float(z))))
                sources.append(str(p_sn).replace("\\", "/"))
        except Exception:
            pass

    # 条件分岐: `p_tz.exists()` を満たす経路を評価する。

    if p_tz.exists():
        try:
            tz = _read_json(p_tz)
            r = (tz.get("rows") or [None])[0] or {}
            z = _maybe_float(r.get("z_std"))
            # 条件分岐: `z is not None` を満たす経路を評価する。
            if z is not None:
                zs.append(("CMB T(z)", abs(float(z))))
                sources.append(str(p_tz).replace("\\", "/"))
        except Exception:
            pass

    # 条件分岐: `not zs` を満たす経路を評価する。

    if not zs:
        return None

    worst = max(z for _, z in zs)
    parts = [f"{name}={_fmt_float(z, digits=2)}" for name, z in zs]
    metric = f"最大|z|={_fmt_float(worst, digits=2)}（p_t=1, p_T=1） / " + ", ".join(parts)

    return ScoreRow(
        id="cosmo_independent",
        label="宇宙論（独立プローブ）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=metric,
        detail="距離指標と独立：SNスペクトル時間伸長 / CMB温度スケーリング（背景Pの最小予測と整合）",
        sources=sources,
        score_kind="abs_z",
    )


# 関数: `_load_cosmology_jwst_mast_row` の入出力契約と処理意図を定義する。

def _load_cosmology_jwst_mast_row(root: Path) -> Optional[ScoreRow]:
    """
    JWST/MAST spectra (x1d) primary-data pipeline status.

    This is not a direct P-model validation metric yet; we track it as an "info" row so
    the reproducible entry point (cache/QC/z-candidates) is visible on the scoreboard.
    """

    manifest_all = root / "data" / "cosmology" / "mast" / "jwst_spectra" / "manifest_all.json"
    # 条件分岐: `not manifest_all.exists()` を満たす経路を評価する。
    if not manifest_all.exists():
        return None

    waitlist_path = root / "output" / "private" / "cosmology" / "jwst_spectra_release_waitlist.json"

    j = _read_json(manifest_all)
    items = j.get("items") if isinstance(j.get("items"), dict) else {}
    targets = j.get("targets") if isinstance(j.get("targets"), list) else []
    n_targets = len(targets) if targets else len(items)

    qc_ok = 0
    z_ok = 0
    z_conf_ok = 0
    line_id_present = 0
    missing_local = 0
    for _, it in items.items():
        # 条件分岐: `not isinstance(it, dict)` を満たす経路を評価する。
        if not isinstance(it, dict):
            continue

        qc = it.get("qc")
        # 条件分岐: `isinstance(qc, dict) and bool(qc.get("ok"))` を満たす経路を評価する。
        if isinstance(qc, dict) and bool(qc.get("ok")):
            qc_ok += 1

        z = it.get("z_estimate")
        # 条件分岐: `isinstance(z, dict) and bool(z.get("ok"))` を満たす経路を評価する。
        if isinstance(z, dict) and bool(z.get("ok")):
            z_ok += 1

        # 条件分岐: `isinstance(z, dict) and str(z.get("reason") or "") == "no_local_x1d"` を満たす経路を評価する。

        if isinstance(z, dict) and str(z.get("reason") or "") == "no_local_x1d":
            missing_local += 1

        zc = it.get("z_confirmed")
        # 条件分岐: `isinstance(zc, dict) and bool(zc.get("ok"))` を満たす経路を評価する。
        if isinstance(zc, dict) and bool(zc.get("ok")):
            z_conf_ok += 1

        lid = it.get("line_id")
        # 条件分岐: `isinstance(lid, dict) and lid.get("path")` を満たす経路を評価する。
        if isinstance(lid, dict) and lid.get("path"):
            line_id_present += 1

    metric = f"x1d(QC ok)={qc_ok}/{n_targets}, z候補={z_ok}/{n_targets}, z確定={z_conf_ok}/{n_targets}"

    blocked_n = None
    next_release_utc = ""
    # 条件分岐: `waitlist_path.exists()` を満たす経路を評価する。
    if waitlist_path.exists():
        try:
            wl = _read_json(waitlist_path)
            summ = wl.get("summary") if isinstance(wl.get("summary"), dict) else {}
            blocked_n = int(summ.get("blocked_targets_n") or 0) if isinstance(summ, dict) else 0
            rels = []
            for b in wl.get("blocked_targets") or []:
                # 条件分岐: `not isinstance(b, dict)` を満たす経路を評価する。
                if not isinstance(b, dict):
                    continue

                s = str(b.get("next_release_utc") or "").strip()
                # 条件分岐: `s` を満たす経路を評価する。
                if s:
                    rels.append(s)

            next_release_utc = min(rels) if rels else ""
        except Exception:
            blocked_n = None
            next_release_utc = ""

    # 条件分岐: `blocked_n is not None and blocked_n > 0` を満たす経路を評価する。

    if blocked_n is not None and blocked_n > 0:
        metric += f"（公開待ち={blocked_n}"
        # 条件分岐: `next_release_utc` を満たす経路を評価する。
        if next_release_utc:
            metric += f"; next={next_release_utc}"

        metric += "）"

    # 条件分岐: `missing_local > 0` を満たす経路を評価する。

    if missing_local > 0:
        metric += f"（localなし={missing_local}）"

    # 条件分岐: `line_id_present > 0` を満たす経路を評価する。

    if line_id_present > 0:
        metric += f"（line_id={line_id_present}/{n_targets}）"

    detail = "JWST/MAST x1d（スペクトル一次データ）の取得状況と z候補抽出→手動線同定→z確定の入口"
    return ScoreRow(
        id="jwst_mast",
        label="JWST/MAST（スペクトル一次データ）",
        status="info",
        score=None,
        metric=metric,
        detail=detail,
        sources=[str(manifest_all).replace("\\", "/")] + ([str(waitlist_path).replace("\\", "/")] if waitlist_path.exists() else []),
    )


# 関数: `_load_cosmology_bao_catalog_row` の入出力契約と処理意図を定義する。

def _load_cosmology_bao_catalog_row(root: Path) -> Optional[ScoreRow]:
    """
    BAO geometry from catalog-based ξℓ (galaxy+random).

    We use the fitted ε significance (|ε|/σ_ε) from the smooth+peak peakfit outputs.

    - Phase A: screening（diag proxy cov）
    - Phase B: decisive（Ross 2016 full cov + MW multigrid recon + dist差し替え）
    - pre-recon: cross-check（Satpathy 2016 full cov）
    """
    # Phase B (decisive): MW multigrid + Ross full covariance.
    paths_b = [
        root
        / "output"
        / "private"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757_metrics.json",
        root
        / "output"
        / "public"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757_metrics.json",
    ]

    # pre-recon cross-check: Satpathy full covariance (z-bin only).
    paths_pre = [
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__prerecon_metrics.json",
    ]

    # Phase A (screening): include combined and NGC/SGC splits (north/south) so the scoreboard reflects systematics.
    desi_peakfit_candidates = [
        root
        / "output"
        / "private"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_reservoir_r0to17_mix__rascalc_cov_reservoir_r0to17_mix_metrics.json",
        root
        / "output"
        / "public"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_reservoir_r0to17_mix__rascalc_cov_reservoir_r0to17_mix_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_reservoir_r0to17_mix__rascalc_cov_reservoir_r0to17_mix_metrics.json",
        root
        / "output"
        / "private"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_shrink0p2_full_r0r1_mean_metrics.json",
        root
        / "output"
        / "public"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_shrink0p2_full_r0r1_mean_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_shrink0p2_full_r0r1_mean_metrics.json",
        root
        / "output"
        / "private"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_both_full_r0r1_mean_metrics.json",
        root
        / "output"
        / "public"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_both_full_r0r1_mean_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_both_full_r0r1_mean_metrics.json",
        root
        / "output"
        / "private"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0__jk_cov_both_full_r0_metrics.json",
        root
        / "output"
        / "public"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0__jk_cov_both_full_r0_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0__jk_cov_both_full_r0_metrics.json",
        root
        / "output"
        / "private"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins__jk_cov_both_metrics.json",
        root
        / "output"
        / "public"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins__jk_cov_both_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins__jk_cov_both_metrics.json",
    ]
    desi_peakfit_path = next((p for p in desi_peakfit_candidates if p.exists()), desi_peakfit_candidates[-1])

    paths_a = [
        # combined
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_cmass_combined_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_lowz_combined_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly_metrics.json",
        # eBOSS extension (Phase 4.5B.21; screening)
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_lrgpcmass_rec_combined_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_lrgpcmass_rec_north_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_lrgpcmass_rec_south_metrics.json",
        # eBOSS extension (Phase 4.5B.21.4; screening)
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_qso_combined_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_qso_north_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_qso_south_metrics.json",
        # DESI extension (Phase 4.5B.21.4.4.4; cov alternative): LRG1/LRG2 bins with sky jackknife cov.
        desi_peakfit_path,
        # north/south (NGC/SGC)
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_cmass_north_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_cmass_south_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_lowz_north_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_lowz_south_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_cmasslowztot_north_zbinonly_metrics.json",
        root / "output" / "private" / "cosmology" / "cosmology_bao_catalog_peakfit_cmasslowztot_south_zbinonly_metrics.json",
    ]

    # 関数: `_collect` の入出力契約と処理意図を定義する。
    def _collect(paths: List[Path]) -> Tuple[List[Dict[str, Any]], List[str]]:
        items: List[Dict[str, Any]] = []
        sources: List[str] = []
        for p in paths:
            # 条件分岐: `not p.exists()` を満たす経路を評価する。
            if not p.exists():
                continue

            try:
                j = _read_json(p)
                rows = j.get("results") if isinstance(j.get("results"), list) else []
                for r in rows:
                    # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
                    if not isinstance(r, dict):
                        continue

                    # 条件分岐: `str(r.get("dist") or "") != "pbg"` を満たす経路を評価する。

                    if str(r.get("dist") or "") != "pbg":
                        continue

                    sc = (r.get("screening") or {}) if isinstance(r.get("screening"), dict) else {}
                    abs_sigma = _maybe_float(sc.get("abs_sigma"))
                    # 条件分岐: `abs_sigma is None` を満たす経路を評価する。
                    if abs_sigma is None:
                        continue

                    lower_bound = bool(sc.get("abs_sigma_is_lower_bound"))
                    fit = (r.get("fit") or {}) if isinstance(r.get("fit"), dict) else {}
                    free = (fit.get("free") or {}) if isinstance(fit.get("free"), dict) else {}
                    eps = _maybe_float(free.get("eps"))
                    sig_eps = _maybe_float(sc.get("sigma_eps_1sigma"))
                    items.append(
                        {
                            "sample": str(r.get("sample") or ""),
                            "z_bin": str(r.get("z_bin") or "none"),
                            "z_eff": _maybe_float(r.get("z_eff")),
                            "abs_sigma": abs_sigma,
                            "lower_bound": lower_bound,
                            "eps": eps,
                            "sigma_eps": sig_eps,
                        }
                    )

                sources.append(str(p).replace("\\", "/"))
            except Exception:
                continue

        return items, sources

    items_b, sources_b = _collect(paths_b)
    items_pre, sources_pre = _collect(paths_pre)
    items_a, sources_a = _collect(paths_a)

    # 条件分岐: `not items_b and not items_pre and not items_a` を満たす経路を評価する。
    if not items_b and not items_pre and not items_a:
        return None

    # 関数: `_summarize` の入出力契約と処理意図を定義する。

    def _summarize(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # 条件分岐: `not items` を満たす経路を評価する。
        if not items:
            return None

        worst = max(float(it["abs_sigma"]) for it in items)
        worst_lb = any((abs(float(it["abs_sigma"]) - worst) < 1e-12) and bool(it.get("lower_bound")) for it in items)
        cmass = next((it for it in items if it.get("sample") == "cmass"), None)
        lowz = next((it for it in items if it.get("sample") == "lowz"), None)
        zbin = [it for it in items if it.get("z_bin") not in ("", "none")]
        zbin_worst = max((float(it["abs_sigma"]) for it in zbin), default=None)
        zbin_worst_lb = (
            any((abs(float(it["abs_sigma"]) - float(zbin_worst)) < 1e-12) and bool(it.get("lower_bound")) for it in zbin)
            if zbin_worst is not None
            else False
        )
        worst_item = next((it for it in items if abs(float(it.get("abs_sigma") or 0.0) - worst) < 1e-12), None)
        return {
            "worst": worst,
            "worst_lb": worst_lb,
            "worst_item": worst_item,
            "cmass": cmass,
            "lowz": lowz,
            "zbin_worst": zbin_worst,
            "zbin_worst_lb": zbin_worst_lb,
        }

    sum_b = _summarize(items_b)
    sum_pre = _summarize(items_pre)
    sum_a = _summarize(items_a)

    # Score: prefer Phase B when available.
    score_sum = sum_b or sum_a
    assert score_sum is not None
    worst = float(score_sum["worst"])
    worst_lb = bool(score_sum.get("worst_lb"))

    # 関数: `_fmt_sigma` の入出力契約と処理意図を定義する。
    def _fmt_sigma(v: Optional[float], *, lower_bound: bool = False) -> str:
        # 条件分岐: `v is None` を満たす経路を評価する。
        if v is None:
            return "?σ"

        return f"≥{_fmt_float(v, digits=2)}σ" if lower_bound else f"{_fmt_float(v, digits=2)}σ"

    # 関数: `_fmt_item` の入出力契約と処理意図を定義する。

    def _fmt_item(it: Optional[Dict[str, Any]]) -> str:
        # 条件分岐: `not it` を満たす経路を評価する。
        if not it:
            return ""

        z = _maybe_float(it.get("z_eff"))
        ztag = f"z={_fmt_float(z, digits=3)}" if z is not None else "z=?"
        return f"{ztag}:{_fmt_sigma(_maybe_float(it.get('abs_sigma')), lower_bound=bool(it.get('lower_bound')))}"

    # 関数: `_fmt_eps_pm` の入出力契約と処理意図を定義する。

    def _fmt_eps_pm(item: Optional[Dict[str, Any]]) -> str:
        # 条件分岐: `not item` を満たす経路を評価する。
        if not item:
            return ""

        eps = _maybe_float(item.get("eps"))
        sig = _maybe_float(item.get("sigma_eps"))
        # 条件分岐: `eps is None` を満たす経路を評価する。
        if eps is None:
            return ""

        # 条件分岐: `sig is None or sig <= 0` を満たす経路を評価する。

        if sig is None or sig <= 0:
            return f"ε={_fmt_float(eps, digits=3)}"

        return f"ε={_fmt_float(eps, digits=3)}±{_fmt_float(sig, digits=3)}"

    # 関数: `_fmt_phase` の入出力契約と処理意図を定義する。

    def _fmt_phase(sum_: Optional[Dict[str, Any]], *, label: str) -> str:
        # 条件分岐: `not sum_` を満たす経路を評価する。
        if not sum_:
            return ""

        parts: List[str] = []
        cmass = sum_.get("cmass")
        lowz = sum_.get("lowz")
        zbin_worst = _maybe_float(sum_.get("zbin_worst"))
        zbin_worst_lb = bool(sum_.get("zbin_worst_lb"))

        # 条件分岐: `cmass` を満たす経路を評価する。
        if cmass:
            parts.append(f"CMASS({_fmt_item(cmass)})")

        # 条件分岐: `lowz` を満たす経路を評価する。

        if lowz:
            parts.append(f"LOWZ({_fmt_item(lowz)})")

        # 条件分岐: `zbin_worst is not None` を満たす経路を評価する。

        if zbin_worst is not None:
            parts.append(f"z-bin(max={_fmt_sigma(zbin_worst, lower_bound=zbin_worst_lb)})")

        worst_item = sum_.get("worst_item") if isinstance(sum_.get("worst_item"), dict) else None
        worst_item_txt = ""
        # 条件分岐: `worst_item` を満たす経路を評価する。
        if worst_item:
            zb = str(worst_item.get("z_bin") or "")
            z = _maybe_float(worst_item.get("z_eff"))
            ztag = f"z={_fmt_float(z, digits=3)}" if z is not None else ""
            eps_txt = _fmt_eps_pm(worst_item)
            extra = ", ".join([t for t in (zb, ztag, eps_txt) if t])
            # 条件分岐: `extra` を満たす経路を評価する。
            if extra:
                worst_item_txt = f"（worst:{extra}）"

        s = f"{label}: 最大|z|={_fmt_sigma(_maybe_float(sum_.get('worst')), lower_bound=bool(sum_.get('worst_lb')))}{worst_item_txt}"
        # 条件分岐: `parts` を満たす経路を評価する。
        if parts:
            s += " / " + ", ".join([p for p in parts if p])

        return s

    metric_parts: List[str] = []
    # 条件分岐: `sum_b` を満たす経路を評価する。
    if sum_b:
        metric_parts.append(_fmt_phase(sum_b, label="PhaseB（Ross cov; MW multigrid）"))

    # 条件分岐: `sum_pre` を満たす経路を評価する。

    if sum_pre:
        metric_parts.append(_fmt_phase(sum_pre, label="pre-recon（Satpathy cov）"))

    # 条件分岐: `sum_a` を満たす経路を評価する。

    if sum_a:
        metric_parts.append(_fmt_phase(sum_a, label="PhaseA（screening）"))

    # Cross-check (independent pipeline): P(k) multipoles peakfit (Beutler et al.; window-convolved).

    sources_pk: List[str] = []
    pk_path = root / "output" / "private" / "cosmology" / "cosmology_bao_pk_multipole_peakfit_window_metrics.json"
    # 条件分岐: `pk_path.exists()` を満たす経路を評価する。
    if pk_path.exists():
        try:
            jk = _read_json(pk_path)
            rows_k = jk.get("results") if isinstance(jk.get("results"), list) else []
            pk_items: List[Tuple[float, int, float, float]] = []
            for r in rows_k:
                # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
                if not isinstance(r, dict):
                    continue

                zbin_k = int(_maybe_float(r.get("zbin")) or 0)
                fit_k = (r.get("fit") or {}) if isinstance(r.get("fit"), dict) else {}
                free_k = (fit_k.get("free") or {}) if isinstance(fit_k.get("free"), dict) else {}
                eps_k = _maybe_float(free_k.get("eps"))
                ci = free_k.get("eps_ci_1sigma")
                # 条件分岐: `eps_k is None or not (isinstance(ci, list) and len(ci) == 2)` を満たす経路を評価する。
                if eps_k is None or not (isinstance(ci, list) and len(ci) == 2):
                    continue

                lo = _maybe_float(ci[0])
                hi = _maybe_float(ci[1])
                # 条件分岐: `lo is None or hi is None` を満たす経路を評価する。
                if lo is None or hi is None:
                    continue

                sig_k = (float(hi) - float(lo)) / 2.0
                # 条件分岐: `not (sig_k > 0.0)` を満たす経路を評価する。
                if not (sig_k > 0.0):
                    continue

                pk_items.append((abs(float(eps_k)) / float(sig_k), zbin_k, float(eps_k), float(sig_k)))

            # 条件分岐: `pk_items` を満たす経路を評価する。

            if pk_items:
                abs_sigma_k, zbin_k, eps_k, sig_k = max(pk_items, key=lambda t: float(t[0]))
                metric_parts.append(
                    f"P(k)window(post; Beutler): 最大|z|={_fmt_float(abs_sigma_k, digits=2)}σ（zbin{int(zbin_k)}: ε={_fmt_float(eps_k, digits=3)}±{_fmt_float(sig_k, digits=3)}）"
                )
                sources_pk.append(str(pk_path).replace("\\", "/"))
        except Exception:
            pass

    metric = " / ".join([p for p in metric_parts if p])

    return ScoreRow(
        id="cosmo_bao_catalog",
        label="宇宙論（BAO一次情報）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=metric,
        detail="BOSS DR12v5: 銀河+random→ξℓ→smooth+peakでAP warping ε を推定（|ε|/σ_ε）。スコアは PhaseB（Ross full covariance + MW multigrid recon + dist差し替え）を優先し、pre-recon（Satpathy cov）は整合のクロスチェック、PhaseA（screening）は補助情報として併記。加えて、独立パイプラインとして P(k) multipoles（Beutler et al.; 窓関数込み）peakfit の結果も参考として併記する。",
        sources=[*sources_b, *sources_pre, *sources_a, *sources_pk],
        score_kind="abs_z",
    )


# 関数: `_load_cosmology_cmb_polarization_phase_row` の入出力契約と処理意図を定義する。

def _load_cosmology_cmb_polarization_phase_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "private" / "cosmology" / "cosmology_cmb_polarization_phase_audit_metrics.json",
            root / "output" / "public" / "cosmology" / "cosmology_cmb_polarization_phase_audit_metrics.json",
            root / "output" / "cosmology" / "cosmology_cmb_polarization_phase_audit_metrics.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    phase_fit = j.get("phase_fit") if isinstance(j.get("phase_fit"), dict) else {}
    ee = phase_fit.get("ee") if isinstance(phase_fit.get("ee"), dict) else {}
    te = phase_fit.get("te") if isinstance(phase_fit.get("te"), dict) else {}
    d_ee = _maybe_float(ee.get("delta_fit"))
    d_te = _maybe_float(te.get("delta_fit"))
    a_ee = _maybe_float(ee.get("abs_shift_from_expected"))
    a_te = _maybe_float(te.get("abs_shift_from_expected"))
    gate = j.get("gate") if isinstance(j.get("gate"), dict) else {}
    hard = gate.get("hard_gate") if isinstance(gate.get("hard_gate"), dict) else {}
    hard_pass = bool(hard.get("pass"))
    overall = str(gate.get("overall_status") or "")
    worst = max([abs(x) for x in (a_ee, a_te) if x is not None], default=None)
    score = _score_lower_better(worst, ok_max=0.12, mixed_max=0.20)
    status = "ok" if hard_pass else (_status_from_abs_sigma(worst / 0.12) if worst is not None else _status_from_gate(overall))
    metric_parts: List[str] = []
    # 条件分岐: `d_ee is not None` を満たす経路を評価する。
    if d_ee is not None:
        metric_parts.append(f"Δφ_EE={_fmt_float(d_ee, digits=3)}")

    # 条件分岐: `d_te is not None` を満たす経路を評価する。

    if d_te is not None:
        metric_parts.append(f"Δφ_TE={_fmt_float(d_te, digits=3)}")

    # 条件分岐: `a_ee is not None` を満たす経路を評価する。

    if a_ee is not None:
        metric_parts.append(f"|Δφ_EE−0.5|={_fmt_float(a_ee, digits=3)}")

    # 条件分岐: `a_te is not None` を満たす経路を評価する。

    if a_te is not None:
        metric_parts.append(f"|Δφ_TE−0.25|={_fmt_float(a_te, digits=3)}")

    metric_parts.append(f"hard_gate={'pass' if hard_pass else 'fail'}")
    return ScoreRow(
        id="cosmo_cmb_phase",
        label="宇宙論（CMB偏極位相）",
        status=status,
        score=score,
        metric=" / ".join(metric_parts),
        detail="TT/EE/TEの位相関係（EEはTTから半波長、TEは1/4波長）を hard gate で監査。",
        sources=[str(path).replace("\\", "/")],
        score_kind="phase_delta",
    )


# 関数: `_load_cosmology_fsigma8_growth_row` の入出力契約と処理意図を定義する。

def _load_cosmology_fsigma8_growth_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "private" / "cosmology" / "cosmology_fsigma8_growth_mapping_metrics.json",
            root / "output" / "public" / "cosmology" / "cosmology_fsigma8_growth_mapping_metrics.json",
            root / "output" / "cosmology" / "cosmology_fsigma8_growth_mapping_metrics.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    delay = ((j.get("branches") or {}).get("delay") or {}) if isinstance(j.get("branches"), dict) else {}
    chi2 = _maybe_float(delay.get("chi2"))
    dof = int(_maybe_float(delay.get("dof")) or 0)
    chi2_dof = _maybe_float(delay.get("chi2_per_dof"))
    max_abs_z = _maybe_float(delay.get("max_abs_z_score"))
    tau_eff = _maybe_float(delay.get("tau_eff_gyr"))
    overall = str((j.get("gates") or {}).get("overall_status") or "") if isinstance(j.get("gates"), dict) else ""
    score = _score_lower_better(max_abs_z, ok_max=1.0, mixed_max=2.0)
    status = _status_from_gate(overall) if overall else _status_from_abs_sigma(max_abs_z)
    metric_parts: List[str] = []
    # 条件分岐: `chi2 is not None and dof > 0` を満たす経路を評価する。
    if chi2 is not None and dof > 0:
        metric_parts.append(f"χ²/dof={_fmt_float(chi2, digits=3)}/{dof}")

    # 条件分岐: `chi2_dof is not None` を満たす経路を評価する。

    if chi2_dof is not None:
        metric_parts.append(f"χ²/ν={_fmt_float(chi2_dof, digits=3)}")

    # 条件分岐: `max_abs_z is not None` を満たす経路を評価する。

    if max_abs_z is not None:
        metric_parts.append(f"max|z|={_fmt_float(max_abs_z, digits=3)}")

    # 条件分岐: `tau_eff is not None` を満たす経路を評価する。

    if tau_eff is not None:
        metric_parts.append(f"τ_eff={_fmt_float(tau_eff, digits=4)} Gyr")

    # 条件分岐: `overall` を満たす経路を評価する。

    if overall:
        metric_parts.append(f"overall={overall}")

    return ScoreRow(
        id="cosmo_fsigma8",
        label="宇宙論（構造形成 fσ8）",
        status=status,
        score=score,
        metric=" / ".join(metric_parts),
        detail="遅延枝の実効摩擦項 Γ_eff で growth 方程式を閉じ、RSD fσ8 を監査。",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


# 関数: `_load_cosmology_cluster_collision_row` の入出力契約と処理意図を定義する。

def _load_cosmology_cluster_collision_row(root: Path) -> Optional[ScoreRow]:
    deriv_path = _first_existing(
        [
            root / "output" / "public" / "cosmology" / "cosmology_cluster_collision_pmu_jmu_separation_derivation.json",
            root / "output" / "private" / "cosmology" / "cosmology_cluster_collision_pmu_jmu_separation_derivation.json",
            root / "output" / "cosmology" / "cosmology_cluster_collision_pmu_jmu_separation_derivation.json",
        ]
    )
    audit_path = _first_existing(
        [
            root / "output" / "public" / "cosmology" / "cosmology_cluster_collision_p_peak_offset_audit.json",
            root / "output" / "private" / "cosmology" / "cosmology_cluster_collision_p_peak_offset_audit.json",
            root / "output" / "cosmology" / "cosmology_cluster_collision_p_peak_offset_audit.json",
        ]
    )
    # 条件分岐: `deriv_path is None and audit_path is None` を満たす経路を評価する。
    if deriv_path is None and audit_path is None:
        return None

    chi2_dof = None
    max_abs_z = None
    n_obs = None
    status = "info"
    metric_parts: List[str] = []
    sources: List[str] = []
    # 条件分岐: `deriv_path is not None` を満たす経路を評価する。
    if deriv_path is not None:
        jd = _read_json(deriv_path)
        fit = jd.get("fit") if isinstance(jd.get("fit"), dict) else {}
        chi2_dof = _maybe_float(fit.get("chi2_dof"))
        max_abs_z = _maybe_float(fit.get("max_abs_z_offset"))
        n_obs = int(_maybe_float(fit.get("n_observations")) or 0)
        xi_mode = str(fit.get("xi_mode") or "")
        ad_hoc = int(_maybe_float(fit.get("ad_hoc_parameter_count")) or 0)
        dstat = str((jd.get("decision") or {}).get("overall_status") or "")
        status = _status_from_gate(dstat) if dstat else status
        # 条件分岐: `chi2_dof is not None` を満たす経路を評価する。
        if chi2_dof is not None:
            metric_parts.append(f"導出 χ²/ν={_fmt_float(chi2_dof, digits=3)}")

        # 条件分岐: `max_abs_z is not None` を満たす経路を評価する。

        if max_abs_z is not None:
            metric_parts.append(f"導出 max|z|={_fmt_float(max_abs_z, digits=3)}")

        metric_parts.append(f"ad_hoc={ad_hoc}")
        # 条件分岐: `xi_mode` を満たす経路を評価する。
        if xi_mode:
            metric_parts.append(f"xi_mode={xi_mode}")

        # 条件分岐: `dstat` を満たす経路を評価する。

        if dstat:
            metric_parts.append(f"derivation={dstat}")

        sources.append(str(deriv_path).replace("\\", "/"))

    # 条件分岐: `audit_path is not None` を満たす経路を評価する。

    if audit_path is not None:
        ja = _read_json(audit_path)
        pm = ((ja.get("models") or {}).get("pmodel_corrected") or {}) if isinstance(ja.get("models"), dict) else {}
        chi2_dof_a = _maybe_float(pm.get("chi2_dof"))
        max_abs_z_lens = _maybe_float(pm.get("max_abs_z_p_lens"))
        astat = str((ja.get("decision") or {}).get("overall_status") or "")
        # 条件分岐: `chi2_dof_a is not None` を満たす経路を評価する。
        if chi2_dof_a is not None:
            metric_parts.append(f"監査 χ²/ν={_fmt_float(chi2_dof_a, digits=3)}")

        # 条件分岐: `max_abs_z_lens is not None` を満たす経路を評価する。

        if max_abs_z_lens is not None:
            metric_parts.append(f"監査 lens max|z|={_fmt_float(max_abs_z_lens, digits=3)}")

        # 条件分岐: `astat` を満たす経路を評価する。

        if astat:
            metric_parts.append(f"audit={astat}")

        # 条件分岐: `status == "info"` を満たす経路を評価する。

        if status == "info":
            status = _status_from_gate(astat)

        sources.append(str(audit_path).replace("\\", "/"))

    score = _score_lower_better(chi2_dof, ok_max=4.0, mixed_max=6.0)
    # 条件分岐: `score is None and max_abs_z is not None` を満たす経路を評価する。
    if score is None and max_abs_z is not None:
        score = _score_lower_better(max_abs_z, ok_max=2.0, mixed_max=3.0)

    return ScoreRow(
        id="cosmo_cluster_collision",
        label="宇宙論（銀河団衝突オフセット）",
        status=status,
        score=score,
        metric=" / ".join(metric_parts),
        detail="Bullet系で Pμ–Jμ 遅延核からオフセットを導出し、レンズ中心ずれを監査。",
        sources=sources,
        score_kind="chi2_dof",
    )


# 関数: `_load_cosmology_cmb_acoustic_row` の入出力契約と処理意図を定義する。

def _load_cosmology_cmb_acoustic_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "public" / "cosmology" / "cosmology_cmb_acoustic_peak_reconstruction_metrics.json",
            root / "output" / "private" / "cosmology" / "cosmology_cmb_acoustic_peak_reconstruction_metrics.json",
            root / "output" / "cosmology" / "cosmology_cmb_acoustic_peak_reconstruction_metrics.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    gate = j.get("gate") if isinstance(j.get("gate"), dict) else {}
    first3 = gate.get("first3") if isinstance(gate.get("first3"), dict) else {}
    ext46 = gate.get("extended_4to6") if isinstance(gate.get("extended_4to6"), dict) else {}
    overall = gate.get("overall") if isinstance(gate.get("overall"), dict) else {}
    overall_ext = gate.get("overall_extended") if isinstance(gate.get("overall_extended"), dict) else {}
    model = j.get("model") if isinstance(j.get("model"), dict) else {}
    closure = model.get("first_principles_closure") if isinstance(model.get("first_principles_closure"), dict) else {}
    dm_free = (
        closure.get("third_peak_dm_free_damping_proof")
        if isinstance(closure.get("third_peak_dm_free_damping_proof"), dict)
        else {}
    )

    first3_ell = _maybe_float(first3.get("max_abs_delta_ell"))
    first3_amp = _maybe_float(first3.get("max_abs_delta_amp_rel"))
    ext46_ell = _maybe_float(ext46.get("max_abs_delta_ell"))
    ext46_amp = _maybe_float(ext46.get("max_abs_delta_amp_rel"))
    theorem_pass = bool(dm_free.get("attenuation_theorem_pass"))
    a3_pred = _maybe_float(dm_free.get("a3_over_a1_pred_dm_free"))
    a3_obs = _maybe_float(dm_free.get("a3_over_a1_observed"))

    core_status = str(overall.get("status") or "")
    ext_status = str(overall_ext.get("status") or "")
    status = _status_from_gate(ext_status or core_status)
    score = _score_lower_better(ext46_ell, ok_max=20.0, mixed_max=80.0)
    # 条件分岐: `score is None` を満たす経路を評価する。
    if score is None:
        score = _score_lower_better(first3_ell, ok_max=5.0, mixed_max=15.0)

    metric_parts: List[str] = []
    # 条件分岐: `first3_ell is not None` を満たす経路を評価する。
    if first3_ell is not None:
        metric_parts.append(f"first3 max|Δℓ|={_fmt_float(first3_ell, digits=3)}")

    # 条件分岐: `first3_amp is not None` を満たす経路を評価する。

    if first3_amp is not None:
        metric_parts.append(f"first3 max|ΔA/A|={_fmt_float(first3_amp, digits=3)}")

    # 条件分岐: `ext46_ell is not None` を満たす経路を評価する。

    if ext46_ell is not None:
        metric_parts.append(f"holdout4-6 max|Δℓ|={_fmt_float(ext46_ell, digits=3)}")

    # 条件分岐: `ext46_amp is not None` を満たす経路を評価する。

    if ext46_amp is not None:
        metric_parts.append(f"holdout4-6 max|ΔA/A|={_fmt_float(ext46_amp, digits=3)}")

    # 条件分岐: `a3_pred is not None and a3_obs is not None` を満たす経路を評価する。

    if a3_pred is not None and a3_obs is not None:
        metric_parts.append(f"A3/A1(pred/obs)={_fmt_float(a3_pred, digits=3)}/{_fmt_float(a3_obs, digits=3)}")

    metric_parts.append(f"DMなし第3ピーク減衰={'pass' if theorem_pass else 'fail'}")
    # 条件分岐: `core_status` を満たす経路を評価する。
    if core_status:
        metric_parts.append(f"core={core_status}")

    # 条件分岐: `ext_status` を満たす経路を評価する。

    if ext_status:
        metric_parts.append(f"extended={ext_status}")

    return ScoreRow(
        id="cosmo_cmb_acoustic",
        label="宇宙論（CMB音響ピーク）",
        status=status,
        score=score,
        metric=" / ".join(metric_parts),
        detail="第1〜3ピーク逆同定と第4〜6 holdout予言、DMなし第3ピーク減衰を同時監査。",
        sources=[str(path).replace("\\", "/")],
        score_kind="holdout_peak",
    )


# 関数: `_load_sparc_rotation_row` の入出力契約と処理意図を定義する。

def _load_sparc_rotation_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "public" / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json",
            root / "output" / "private" / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json",
            root / "output" / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    fit = j.get("fit_results") if isinstance(j.get("fit_results"), dict) else {}
    pm = fit.get("pmodel_corrected") if isinstance(fit.get("pmodel_corrected"), dict) else {}
    gal = fit.get("galaxy_level_summary") if isinstance(fit.get("galaxy_level_summary"), dict) else {}
    comp = fit.get("comparison") if isinstance(fit.get("comparison"), dict) else {}
    counts = j.get("counts") if isinstance(j.get("counts"), dict) else {}

    n_gal = int(_maybe_float(counts.get("n_galaxies")) or 0)
    chi2_dof = _maybe_float(pm.get("chi2_dof"))
    median_chi2_dof = _maybe_float(gal.get("median_chi2_dof_pmodel"))
    delta_chi2 = _maybe_float(comp.get("delta_chi2_baryon_minus_pmodel"))
    ratio = _maybe_float(comp.get("chi2_dof_ratio_pmodel_over_baryon"))
    status = "ok" if (delta_chi2 is not None and delta_chi2 > 0) else "mixed"
    score = _score_lower_better(median_chi2_dof, ok_max=10.0, mixed_max=20.0)

    metric_parts: List[str] = []
    # 条件分岐: `delta_chi2 is not None` を満たす経路を評価する。
    if delta_chi2 is not None:
        metric_parts.append(f"Δχ²(baryon−P)={_fmt_float(delta_chi2, digits=1)}")

    # 条件分岐: `ratio is not None` を満たす経路を評価する。

    if ratio is not None:
        metric_parts.append(f"χ²/ν比(P/baryon)={_fmt_float(ratio, digits=3)}")

    # 条件分岐: `chi2_dof is not None` を満たす経路を評価する。

    if chi2_dof is not None:
        metric_parts.append(f"global χ²/ν={_fmt_float(chi2_dof, digits=3)}")

    # 条件分岐: `median_chi2_dof is not None` を満たす経路を評価する。

    if median_chi2_dof is not None:
        metric_parts.append(f"median χ²/ν={_fmt_float(median_chi2_dof, digits=3)}")

    return ScoreRow(
        id="sparc_rotation",
        label="銀河回転曲線（SPARC）",
        status=status,
        score=score,
        metric=" / ".join(metric_parts),
        detail="single-Υ の厳しい拘束下で baryon-only 比の説明力向上を監査。",
        sources=[str(path).replace("\\", "/")],
        score_kind="chi2",
    )


# 関数: `_load_xrism_row` の入出力契約と処理意図を定義する。

def _load_xrism_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "private" / "xrism" / "xrism_integration_metrics.json",
            root / "output" / "public" / "xrism" / "xrism_integration_metrics.json",
            root / "output" / "xrism" / "xrism_integration_metrics.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    xr = j.get("xrism") if isinstance(j.get("xrism"), dict) else {}
    bh = xr.get("bh") if isinstance(xr.get("bh"), dict) else {}
    cl = xr.get("cluster") if isinstance(xr.get("cluster"), dict) else {}
    bh_t = bh.get("table1") if isinstance(bh.get("table1"), dict) else {}
    cl_t = cl.get("table1") if isinstance(cl.get("table1"), dict) else {}
    bh_status = str(bh_t.get("status") or "")
    cl_status = str(cl_t.get("status") or "")
    adopted = (bh_status == "adopted") and (cl_status == "adopted")
    status = "ok" if adopted else ("mixed" if (bh_status or cl_status) else "info")

    n_bh = int(_maybe_float(bh.get("n_obsids_detected")) or 0)
    n_cl = int(_maybe_float(cl.get("n_obsids_detected")) or 0)

    best_bh = bh.get("best_detected_row") if isinstance(bh.get("best_detected_row"), dict) else {}
    bh_ratio = _maybe_float(best_bh.get("sys_over_stat"))
    per_cl = cl.get("per_obsid_best") if isinstance(cl.get("per_obsid_best"), list) else []
    cl_ratio = None
    # 条件分岐: `per_cl and isinstance(per_cl[0], dict)` を満たす経路を評価する。
    if per_cl and isinstance(per_cl[0], dict):
        cl_ratio = _maybe_float(per_cl[0].get("sys_over_stat"))

    metric_parts: List[str] = [f"BH={bh_status or 'n/a'}", f"cluster={cl_status or 'n/a'}"]
    # 条件分岐: `n_bh or n_cl` を満たす経路を評価する。
    if n_bh or n_cl:
        metric_parts.append(f"detected_obsids(BH/cluster)={n_bh}/{n_cl}")

    # 条件分岐: `bh_ratio is not None` を満たす経路を評価する。

    if bh_ratio is not None:
        metric_parts.append(f"BH sys/stat={_fmt_float(bh_ratio, digits=2)}")

    # 条件分岐: `cl_ratio is not None` を満たす経路を評価する。

    if cl_ratio is not None:
        metric_parts.append(f"cluster sys/stat={_fmt_float(cl_ratio, digits=2)}")

    return ScoreRow(
        id="xrism",
        label="XRISM（公開一次データ）",
        status=status,
        score=0.5 if adopted else 1.5,
        metric=" / ".join(metric_parts),
        detail="BH/AGN と銀河団の双方で sys/stat ゲートを満たすかを監査。",
        sources=[str(path).replace("\\", "/")],
        score_kind="adoption_gate",
    )


# 関数: `_load_bbn_row` の入出力契約と処理意図を定義する。

def _load_bbn_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "public" / "quantum" / "bbn_he4_watch_convergence_audit_metrics.json",
            root / "output" / "private" / "quantum" / "bbn_he4_watch_convergence_audit_metrics.json",
            root / "output" / "quantum" / "bbn_he4_watch_convergence_audit_metrics.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    decision = j.get("decision") if isinstance(j.get("decision"), dict) else {}
    overall = str(decision.get("overall_status") or "")
    z_abs = _maybe_float(decision.get("propagated_z_abs_conservative"))
    inputs = j.get("inputs") if isinstance(j.get("inputs"), dict) else {}
    nominal = j.get("freeze_nominal") if isinstance(j.get("freeze_nominal"), dict) else {}
    y_pred = _maybe_float(nominal.get("y_pred"))
    y_obs = _maybe_float(inputs.get("he4_obs"))
    y_sig = _maybe_float(inputs.get("he4_sigma_obs"))
    status = _status_from_gate(overall)
    score = _score_lower_better(z_abs, ok_max=2.0, mixed_max=3.0)
    metric_parts: List[str] = []
    # 条件分岐: `y_pred is not None and y_obs is not None and y_sig is not None` を満たす経路を評価する。
    if y_pred is not None and y_obs is not None and y_sig is not None:
        metric_parts.append(
            f"Yp(pred/obs)={_fmt_float(y_pred, digits=4)}/{_fmt_float(y_obs, digits=4)}±{_fmt_float(y_sig, digits=4)}"
        )

    # 条件分岐: `z_abs is not None` を満たす経路を評価する。

    if z_abs is not None:
        metric_parts.append(f"|z|={_fmt_float(z_abs, digits=3)}")

    # 条件分岐: `overall` を満たす経路を評価する。

    if overall:
        metric_parts.append(f"overall={overall}")

    return ScoreRow(
        id="bbn",
        label="BBN（初期熱史）",
        status=status,
        score=score,
        metric=" / ".join(metric_parts),
        detail="q_B=1/2 枝を入力とした He-4 存在比 Yp の第一原理監査。",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


# 関数: `_load_background_metric_case_b_row` の入出力契約と処理意図を定義する。

def _load_background_metric_case_b_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "public" / "theory" / "pmodel_vector_metric_choice_audit_caseB_effective.json",
            root / "output" / "private" / "theory" / "pmodel_vector_metric_choice_audit_caseB_effective.json",
            root / "output" / "theory" / "pmodel_vector_metric_choice_audit_caseB_effective.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    summary = (((j.get("case_result") or {}).get("summary") or {})) if isinstance(j.get("case_result"), dict) else {}
    derived = (((j.get("case_result") or {}).get("derived") or {})) if isinstance(j.get("case_result"), dict) else {}
    overall = str(summary.get("overall_status") or "")
    decision = str(summary.get("decision") or "")
    z_gamma = _maybe_float(derived.get("z_gamma"))
    nonlinear = bool(derived.get("nonlinear_pde_closure_pass")) if "nonlinear_pde_closure_pass" in derived else None
    metric_parts: List[str] = []
    # 条件分岐: `z_gamma is not None` を満たす経路を評価する。
    if z_gamma is not None:
        metric_parts.append(f"γ gate z={_fmt_float(z_gamma, digits=3)}")

    # 条件分岐: `nonlinear is not None` を満たす経路を評価する。

    if nonlinear is not None:
        metric_parts.append(f"nonlinear_pde_closure={'pass' if nonlinear else 'fail'}")

    # 条件分岐: `decision` を満たす経路を評価する。

    if decision:
        metric_parts.append(f"decision={decision}")

    return ScoreRow(
        id="metric_case_b",
        label="背景計量（caseB: 有効計量）",
        status=_status_from_gate(overall),
        score=0.5 if _status_from_gate(overall) == "ok" else 1.5,
        metric=" / ".join(metric_parts),
        detail="背景計量は g_{μν}(P) を正式採用（flat caseAは棄却）。",
        sources=[str(path).replace("\\", "/")],
        score_kind="gate",
    )


# 関数: `_load_background_metric_case_a_row` の入出力契約と処理意図を定義する。

def _load_background_metric_case_a_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "public" / "theory" / "pmodel_vector_metric_choice_audit_caseA_flat.json",
            root / "output" / "private" / "theory" / "pmodel_vector_metric_choice_audit_caseA_flat.json",
            root / "output" / "theory" / "pmodel_vector_metric_choice_audit_caseA_flat.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    summary = (((j.get("case_result") or {}).get("summary") or {})) if isinstance(j.get("case_result"), dict) else {}
    derived = (((j.get("case_result") or {}).get("derived") or {})) if isinstance(j.get("case_result"), dict) else {}
    overall = str(summary.get("overall_status") or "")
    z_gamma = _maybe_float(derived.get("z_gamma"))
    mercury_rel = _maybe_float(derived.get("mercury_rel_error"))
    metric_parts: List[str] = []
    # 条件分岐: `z_gamma is not None` を満たす経路を評価する。
    if z_gamma is not None:
        metric_parts.append(f"γ gate z={_fmt_float(z_gamma, digits=3)}")

    # 条件分岐: `mercury_rel is not None` を満たす経路を評価する。

    if mercury_rel is not None:
        metric_parts.append(f"Mercury係数残差={_fmt_float(mercury_rel, digits=6)}")

    # 条件分岐: `overall` を満たす経路を評価する。

    if overall:
        metric_parts.append(f"overall={overall}")

    return ScoreRow(
        id="metric_case_a",
        label="背景計量（caseA: 平坦背景）",
        status=_status_from_gate(overall),
        score=3.0,
        metric=" / ".join(metric_parts),
        detail="平坦背景 η_{μν} は weak-field 整合で棄却。",
        sources=[str(path).replace("\\", "/")],
        score_kind="gate",
    )


# 関数: `_load_scalar_limit_reject_row` の入出力契約と処理意図を定義する。

def _load_scalar_limit_reject_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "public" / "theory" / "frame_dragging_scalar_limit_combined_audit.json",
            root / "output" / "private" / "theory" / "frame_dragging_scalar_limit_combined_audit.json",
            root / "output" / "theory" / "frame_dragging_scalar_limit_combined_audit.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    summary = j.get("summary") if isinstance(j.get("summary"), dict) else {}
    overall = str(summary.get("overall_status") or "")
    z_gp = None
    z_lageos = None
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        # 条件分岐: `str(r.get("observable") or "") != "frame_dragging"` を満たす経路を評価する。

        if str(r.get("observable") or "") != "frame_dragging":
            continue

        exp = str(r.get("experiment") or "")
        z = _maybe_float(r.get("z_scalar"))
        # 条件分岐: `exp == "GP-B"` を満たす経路を評価する。
        if exp == "GP-B":
            z_gp = z
        # 条件分岐: 前段条件が不成立で、`exp == "LAGEOS"` を追加評価する。
        elif exp == "LAGEOS":
            z_lageos = z

    metric_parts: List[str] = []
    # 条件分岐: `z_gp is not None` を満たす経路を評価する。
    if z_gp is not None:
        metric_parts.append(f"GP-B z={_fmt_float(z_gp, digits=3)}")

    # 条件分岐: `z_lageos is not None` を満たす経路を評価する。

    if z_lageos is not None:
        metric_parts.append(f"LAGEOS z={_fmt_float(z_lageos, digits=3)}")

    # 条件分岐: `overall` を満たす経路を評価する。

    if overall:
        metric_parts.append(f"overall={overall}")

    return ScoreRow(
        id="scalar_limit_reject",
        label="純スカラー極限（回転なし）",
        status=_status_from_gate(overall),
        score=3.0,
        metric=" / ".join(metric_parts),
        detail="GP-B/LAGEOS の frame-dragging 観測で純スカラー極限を棄却。",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


# 関数: `_load_gw_polarization_row` の入出力契約と処理意図を定義する。

def _load_gw_polarization_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "public" / "gw" / "pmodel_vector_gw_polarization_mapping_audit.json",
            root / "output" / "private" / "gw" / "pmodel_vector_gw_polarization_mapping_audit.json",
            root / "output" / "gw" / "pmodel_vector_gw_polarization_mapping_audit.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    summary = j.get("summary") if isinstance(j.get("summary"), dict) else {}
    network = summary.get("network_gate") if isinstance(summary.get("network_gate"), dict) else {}
    overall = str(summary.get("overall_status") or "")
    scalar_excl = bool(network.get("scalar_exclusion_pass")) if "scalar_exclusion_pass" in network else None
    scalar_red = bool(network.get("scalar_reduction_pass")) if "scalar_reduction_pass" in network else None
    tensor_pass = bool(network.get("tensor_consistency_pass")) if "tensor_consistency_pass" in network else None
    reason = str(summary.get("overall_reason") or "")

    metric_parts: List[str] = []
    # 条件分岐: `scalar_red is not None` を満たす経路を評価する。
    if scalar_red is not None:
        metric_parts.append(f"scalar_reduction={'pass' if scalar_red else 'fail'}")

    # 条件分岐: `scalar_excl is not None` を満たす経路を評価する。

    if scalar_excl is not None:
        metric_parts.append(f"scalar_exclusion={'pass' if scalar_excl else 'fail'}")

    # 条件分岐: `tensor_pass is not None` を満たす経路を評価する。

    if tensor_pass is not None:
        metric_parts.append(f"tensor_consistency={'pass' if tensor_pass else 'fail'}")

    # 条件分岐: `reason` を満たす経路を評価する。

    if reason:
        metric_parts.append(f"reason={reason}")

    # 条件分岐: `overall` を満たす経路を評価する。

    if overall:
        metric_parts.append(f"overall={overall}")

    score = _score_lower_better(0.0 if scalar_excl else 1.0, ok_max=0.0 + 1e-6, mixed_max=1.0)
    return ScoreRow(
        id="gw_polarization",
        label="重力波偏光モード（H1/L1/V1）",
        status=_status_from_gate(overall),
        score=score if score is not None else 1.5,
        metric=" / ".join(metric_parts),
        detail="ベクトル横波のネットワーク監査。縮退低減は確認、完全排除は継続課題。",
        sources=[str(path).replace("\\", "/")],
        score_kind="gate",
    )


# 関数: `_load_gw_area_qnm_imr_row` の入出力契約と処理意図を定義する。

def _load_gw_area_qnm_imr_row(root: Path) -> Optional[ScoreRow]:
    area_path = _first_existing(
        [
            root / "output" / "private" / "gw" / "gw250114_area_theorem_test.json",
            root / "output" / "public" / "gw" / "gw250114_area_theorem_test.json",
            root / "output" / "gw" / "gw250114_area_theorem_test.json",
        ]
    )
    qnm_path = _first_existing(
        [
            root / "output" / "private" / "gw" / "gw250114_ringdown_qnm_fit.json",
            root / "output" / "public" / "gw" / "gw250114_ringdown_qnm_fit.json",
            root / "output" / "gw" / "gw250114_ringdown_qnm_fit.json",
        ]
    )
    imr_path = _first_existing(
        [
            root / "output" / "private" / "gw" / "gw250114_imr_consistency.json",
            root / "output" / "public" / "gw" / "gw250114_imr_consistency.json",
            root / "output" / "gw" / "gw250114_imr_consistency.json",
        ]
    )
    # 条件分岐: `area_path is None and qnm_path is None and imr_path is None` を満たす経路を評価する。
    if area_path is None and qnm_path is None and imr_path is None:
        return None

    area_sigma = None
    first_ge5 = None
    # 条件分岐: `area_path is not None` を満たす経路を評価する。
    if area_path is not None:
        ja = _read_json(area_path)
        summary = ja.get("summary") if isinstance(ja.get("summary"), dict) else {}
        sigma_ref = summary.get("sigma_ref") if isinstance(summary.get("sigma_ref"), dict) else {}
        area_sigma = _maybe_float(sigma_ref.get("sigma_gaussian_combined"))
        first_ge5 = _maybe_float(summary.get("first_time_sigma_ge_5_combined"))

    qnm_f = None
    qnm_tau = None
    # 条件分岐: `qnm_path is not None` を満たす経路を評価する。
    if qnm_path is not None:
        jq = _read_json(qnm_path)
        med = (((jq.get("results") or {}).get("combined") or {}).get("median") or {}) if isinstance(jq.get("results"), dict) else {}
        qnm_f = _maybe_float(med.get("f_hz"))
        qnm_tau = _maybe_float(med.get("tau_s"))

    z_mass = None
    z_spin = None
    p_imr = None
    # 条件分岐: `imr_path is not None` を満たす経路を評価する。
    if imr_path is not None:
        ji = _read_json(imr_path)
        cons = ji.get("consistency") if isinstance(ji.get("consistency"), dict) else {}
        z_mass = _maybe_float(cons.get("z_final_mass_det_1d"))
        z_spin = _maybe_float(cons.get("z_final_spin_1d"))
        p_imr = _maybe_float(cons.get("p_value_mahalanobis2"))

    zvals = [abs(v) for v in (z_mass, z_spin) if v is not None]
    max_abs_z = max(zvals) if zvals else None
    score_area = _score_higher_better(area_sigma, ok_min=3.0, mixed_min=2.0, ideal=5.0)
    score_imr = _score_lower_better(max_abs_z, ok_max=1.0, mixed_max=2.0)
    score_candidates = [s for s in (score_area, score_imr) if s is not None]
    score = max(score_candidates) if score_candidates else None

    status = "info"
    # 条件分岐: `area_sigma is not None and max_abs_z is not None` を満たす経路を評価する。
    if area_sigma is not None and max_abs_z is not None:
        # 条件分岐: `area_sigma >= 3.0 and max_abs_z <= 3.0` を満たす経路を評価する。
        if area_sigma >= 3.0 and max_abs_z <= 3.0:
            status = "ok"
        # 条件分岐: 前段条件が不成立で、`area_sigma >= 2.0 and max_abs_z <= 5.0` を追加評価する。
        elif area_sigma >= 2.0 and max_abs_z <= 5.0:
            status = "mixed"
        else:
            status = "ng"

    metric_parts: List[str] = []
    # 条件分岐: `area_sigma is not None` を満たす経路を評価する。
    if area_sigma is not None:
        metric_parts.append(f"面積定理={_fmt_float(area_sigma, digits=3)}σ")

    # 条件分岐: `first_ge5 is not None` を満たす経路を評価する。

    if first_ge5 is not None:
        metric_parts.append(f"first σ≥5 at t_ref={_fmt_float(first_ge5, digits=1)}M")

    # 条件分岐: `qnm_f is not None` を満たす経路を評価する。

    if qnm_f is not None:
        metric_parts.append(f"QNM(220) f={_fmt_float(qnm_f, digits=3)} Hz")

    # 条件分岐: `qnm_tau is not None` を満たす経路を評価する。

    if qnm_tau is not None:
        metric_parts.append(f"τ={_fmt_float(qnm_tau, digits=6)} s")

    # 条件分岐: `max_abs_z is not None` を満たす経路を評価する。

    if max_abs_z is not None:
        metric_parts.append(f"IMR max|z|={_fmt_float(max_abs_z, digits=3)}")

    # 条件分岐: `p_imr is not None` を満たす経路を評価する。

    if p_imr is not None:
        metric_parts.append(f"IMR p={_fmt_float(p_imr, digits=3)}")

    sources = [str(p).replace("\\", "/") for p in (area_path, qnm_path, imr_path) if p is not None]
    return ScoreRow(
        id="gw_area_qnm_imr",
        label="重力波（GW250114）",
        status=status,
        score=score,
        metric=" / ".join(metric_parts),
        detail="GW250114で面積定理の有意度とringdown QNM、IMR整合を同時監査。",
        sources=sources,
        score_kind="hybrid",
    )


# 関数: `_load_strong_field_higher_order_row` の入出力契約と処理意図を定義する。

def _load_strong_field_higher_order_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "public" / "theory" / "pmodel_strong_field_higher_order_audit.json",
            root / "output" / "private" / "theory" / "pmodel_strong_field_higher_order_audit.json",
            root / "output" / "theory" / "pmodel_strong_field_higher_order_audit.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    joint = (((j.get("fits") or {}).get("joint") or {})) if isinstance(j.get("fits"), dict) else {}
    delta_aic = _maybe_float(joint.get("delta_aic_fit_minus_baseline"))
    lam = _maybe_float(joint.get("lambda_fit"))
    lam_sig = _maybe_float(joint.get("lambda_sigma"))
    decision = str((j.get("decision") or {}).get("overall_status") or "")
    status = _status_from_gate(decision)

    score = None
    # 条件分岐: `delta_aic is not None` を満たす経路を評価する。
    if delta_aic is not None:
        # 条件分岐: `delta_aic <= 0.0` を満たす経路を評価する。
        if delta_aic <= 0.0:
            score = max(0.0, 1.0 + float(delta_aic) / 2.0)
        # 条件分岐: 前段条件が不成立で、`delta_aic <= 2.0` を追加評価する。
        elif delta_aic <= 2.0:
            score = 1.0 + float(delta_aic) / 2.0
        else:
            score = 2.0 + (float(delta_aic) - 2.0) / 2.0

    metric_parts: List[str] = []
    # 条件分岐: `delta_aic is not None` を満たす経路を評価する。
    if delta_aic is not None:
        metric_parts.append(f"ΔAIC={_fmt_float(delta_aic, digits=3)}")

    # 条件分岐: `lam is not None and lam_sig is not None and lam_sig > 0` を満たす経路を評価する。

    if lam is not None and lam_sig is not None and lam_sig > 0:
        metric_parts.append(f"λ_H={_fmt_float(lam, digits=6)}±{_fmt_float(lam_sig, digits=6)}")

    # 条件分岐: `decision` を満たす経路を評価する。

    if decision:
        metric_parts.append(f"overall={decision}")

    return ScoreRow(
        id="strong_field_higher_order",
        label="強場（高次項同時拘束）",
        status=status,
        score=score,
        metric=" / ".join(metric_parts),
        detail="EHT+GW+Pulsar+Fe-Kαを単一 λ_H で同時拘束し、AICで採択可否を監査。",
        sources=[str(path).replace("\\", "/")],
        score_kind="delta_aic",
    )


# 関数: `_load_frame_dragging_row` の入出力契約と処理意図を定義する。

def _load_frame_dragging_row(root: Path) -> Optional[ScoreRow]:
    path = _first_existing(
        [
            root / "output" / "private" / "theory" / "frame_dragging_experiments.json",
            root / "output" / "public" / "theory" / "frame_dragging_experiments.json",
            root / "output" / "theory" / "frame_dragging_experiments.json",
        ]
    )
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    zs: List[float] = []
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        z = _maybe_float(r.get("z_score"))
        # 条件分岐: `z is None` を満たす経路を評価する。
        if z is None:
            continue

        zs.append(abs(z))

    # 条件分岐: `not zs` を満たす経路を評価する。

    if not zs:
        return None

    worst = max(zs)
    return ScoreRow(
        id="frame_dragging",
        label="回転（フレームドラッグ）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=f"最大|z|={_fmt_float(worst, digits=3)}（μ=1）",
        detail="GP-B / LAGEOS の μ=|Ω_obs|/|Ω_pred|（μ=1が一致）",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


# 関数: `_load_eht_row` の入出力契約と処理意図を定義する。

def _load_eht_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "private" / "eht" / "eht_shadow_compare.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    zs: List[Tuple[float, str]] = []
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        z = _maybe_float(r.get("zscore_pmodel"))
        # 条件分岐: `z is None` を満たす経路を評価する。
        if z is None:
            continue

        name = str(r.get("name") or r.get("key") or "")
        zs.append((abs(z), name))

    # 条件分岐: `not zs` を満たす経路を評価する。

    if not zs:
        return None

    worst, worst_name = max(zs, key=lambda x: x[0])
    return ScoreRow(
        id="eht",
        label="EHT（ブラックホール影）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=f"最大|z|={_fmt_float(worst, digits=3)}（κ=1仮定）",
        detail=f"M87*/SgrA* のリング直径 vs シャドウ直径（κ=1近似）。worst={worst_name}",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


# 関数: `_load_binary_pulsar_row` の入出力契約と処理意図を定義する。

def _load_binary_pulsar_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "private" / "pulsar" / "binary_pulsar_orbital_decay_metrics.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    j = _read_json(path)
    metrics = j.get("metrics") if isinstance(j.get("metrics"), list) else []
    zs: List[Tuple[float, str]] = []
    for s in metrics:
        # 条件分岐: `not isinstance(s, dict)` を満たす経路を評価する。
        if not isinstance(s, dict):
            continue

        R = _maybe_float(s.get("R"))
        sig = _maybe_float(s.get("sigma_1"))
        # 条件分岐: `R is None or sig is None or sig <= 0` を満たす経路を評価する。
        if R is None or sig is None or sig <= 0:
            continue

        z = (R - 1.0) / sig
        name = str(s.get("name") or s.get("id") or "")
        zs.append((abs(z), name))

    # 条件分岐: `not zs` を満たす経路を評価する。

    if not zs:
        return None

    worst, worst_name = max(zs, key=lambda x: x[0])
    return ScoreRow(
        id="binary_pulsar",
        label="連星パルサー（軌道減衰）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=f"最大|z|={_fmt_float(worst, digits=2)}（R=1）",
        detail=f"Pdot_b(obs)/Pdot_b(P-model quad) の一致度。worst={worst_name}",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


# 関数: `_load_gw_row` の入出力契約と処理意図を定義する。

def _load_gw_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "private" / "gw" / "gw_multi_event_summary_metrics.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []

    by_event: Dict[str, List[float]] = {}
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        ev = str(r.get("event") or "")
        r2 = _maybe_float(r.get("r2"))
        # 条件分岐: `not ev or r2 is None` を満たす経路を評価する。
        if not ev or r2 is None:
            continue

        by_event.setdefault(ev, []).append(r2)

    # 条件分岐: `not by_event` を満たす経路を評価する。

    if not by_event:
        return None

    max_r2_by_event: List[float] = []
    for rs in by_event.values():
        # 条件分岐: `rs` を満たす経路を評価する。
        if rs:
            max_r2_by_event.append(max(rs))

    max_r2_by_event.sort()
    n_events = len(max_r2_by_event)
    median = None
    # 条件分岐: `n_events` を満たす経路を評価する。
    if n_events:
        mid = n_events // 2
        # 条件分岐: `(n_events % 2) == 1` を満たす経路を評価する。
        if (n_events % 2) == 1:
            median = max_r2_by_event[mid]
        else:
            median = 0.5 * (max_r2_by_event[mid - 1] + max_r2_by_event[mid])

    min_v = min(max_r2_by_event) if max_r2_by_event else None
    n_ge_09 = sum(1 for v in max_r2_by_event if v >= 0.9)
    n_ge_06 = sum(1 for v in max_r2_by_event if v >= 0.6)

    status = "info"
    # 条件分岐: `median is not None` を満たす経路を評価する。
    if median is not None:
        # 条件分岐: `median >= 0.9 and (n_ge_09 / n_events) >= 0.7` を満たす経路を評価する。
        if median >= 0.9 and (n_ge_09 / n_events) >= 0.7:
            status = "ok"
        # 条件分岐: 前段条件が不成立で、`median >= 0.6 and (n_ge_06 / n_events) >= 0.7` を追加評価する。
        elif median >= 0.6 and (n_ge_06 / n_events) >= 0.7:
            status = "mixed"
        else:
            status = "ng"

    score = _score_higher_better(median, ok_min=0.9, mixed_min=0.6, ideal=1.0)

    match_omit = j.get("match_omitted_by_reason") or {}
    omit_short = int(match_omit.get("match_window_too_short") or 0)

    metric_parts = []
    # 条件分岐: `median is not None` を満たす経路を評価する。
    if median is not None:
        metric_parts.append(f"median(max R²)={_fmt_float(median, digits=3)}")

    # 条件分岐: `n_events` を満たす経路を評価する。

    if n_events:
        metric_parts.append(f">=0.6: {n_ge_06}/{n_events}")

    # 条件分岐: `min_v is not None` を満たす経路を評価する。

    if min_v is not None:
        metric_parts.append(f"min(max R²)={_fmt_float(min_v, digits=3)}")

    # 条件分岐: `omit_short` を満たす経路を評価する。

    if omit_short:
        metric_parts.append(f"match省略={omit_short}件")

    metric = " / ".join(metric_parts)

    return ScoreRow(
        id="gw",
        label="重力波（GW150914 等）",
        status=status,
        score=score,
        metric=metric,
        detail="chirp位相（f(t)抽出→四重極チャープ則fit）をイベント×検出器で集計（R²）。",
        sources=[str(path).replace("\\", "/")],
        score_kind="gw_median_max_r2",
    )


# 関数: `_load_delta_row` の入出力契約と処理意図を定義する。

def _load_delta_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "private" / "theory" / "delta_saturation_constraints.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    deltas: List[float] = []
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        d = _maybe_float(r.get("delta_upper_from_gamma"))
        # 条件分岐: `d is None` を満たす経路を評価する。
        if d is None:
            continue

        deltas.append(d)

    # 条件分岐: `not deltas` を満たす経路を評価する。

    if not deltas:
        return None

    tightest = min(deltas)
    return ScoreRow(
        id="delta_saturation",
        label="速度飽和 δ（理論）",
        status="info",
        score=None,
        metric=f"観測上限: δ<{tightest:.2e}",
        detail="既知の高γ観測から『δが大きすぎるとγ_maxが足りない』という上限制約（参考）。",
        sources=[str(path).replace("\\", "/")],
        score_kind="delta_upper_bound",
    )


# 関数: `_compute_sigma_stats_from_table1` の入出力契約と処理意図を定義する。

def _compute_sigma_stats_from_table1(table1_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Parse z-like values from "metric", and the pulsar sigma from "metric_public".
    z_re = re.compile(
        r"(?P<prefix>z|Z)\s*(?:\((?P<label>[^)]*)\))?\s*=\s*(?P<val>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    )
    sigma_items: List[Dict[str, Any]] = []
    for r in table1_rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        metric = str(r.get("metric") or "")
        matches = list(z_re.finditer(metric))
        # 条件分岐: `not matches` を満たす経路を評価する。
        if not matches:
            continue

        chosen = None
        for m in matches:
            lab = (m.group("label") or "")
            # 条件分岐: `"P" in lab or "p" in lab` を満たす経路を評価する。
            if "P" in lab or "p" in lab:
                chosen = m
                break

        # 条件分岐: `chosen is None` を満たす経路を評価する。

        if chosen is None:
            chosen = matches[0]

        try:
            z = float(chosen.group("val"))
        except Exception:
            continue

        sigma_items.append(
            {
                "topic": str(r.get("topic") or ""),
                "observable": str(r.get("observable") or ""),
                "abs_z": abs(z),
                "raw": metric,
            }
        )

    # Pulsar: parse "最大 1.07σ" style metric_public.

    for r in table1_rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        # 条件分岐: `str(r.get("topic") or "") != "連星パルサー（軌道減衰）"` を満たす経路を評価する。

        if str(r.get("topic") or "") != "連星パルサー（軌道減衰）":
            continue

        pub = str(r.get("metric_public") or "")
        m = re.search(r"最大\s*([0-9]+(?:\.[0-9]+)?)\s*σ", pub)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        try:
            z = float(m.group(1))
        except Exception:
            continue

        sigma_items.append(
            {
                "topic": str(r.get("topic") or ""),
                "observable": str(r.get("observable") or ""),
                "abs_z": abs(z),
                "raw": pub,
            }
        )
        break

    within_1 = [it for it in sigma_items if it.get("abs_z") is not None and float(it["abs_z"]) <= 1.0]
    within_2 = [it for it in sigma_items if it.get("abs_z") is not None and float(it["abs_z"]) <= 2.0]
    worst = None
    # 条件分岐: `sigma_items` を満たす経路を評価する。
    if sigma_items:
        worst = max(sigma_items, key=lambda x: float(x.get("abs_z") or 0.0))

    return {
        "n": len(sigma_items),
        "n_within_1sigma": len(within_1),
        "n_within_2sigma": len(within_2),
        "rate_within_1sigma": (len(within_1) / len(sigma_items)) if sigma_items else None,
        "rate_within_2sigma": (len(within_2) / len(sigma_items)) if sigma_items else None,
        "worst": worst,
    }


# 関数: `_table1_status_from_abs_sigma` の入出力契約と処理意図を定義する。

def _table1_status_from_abs_sigma(abs_sigma: float) -> str:
    # Keep consistent with other sigma-based rows.
    return _status_from_abs_sigma(abs_sigma)


# 関数: `_table1_forced_status` の入出力契約と処理意図を定義する。

def _table1_forced_status(topic: str, observable: str) -> Optional[str]:
    t = (topic or "").strip()
    o = (observable or "").strip()
    # 条件分岐: `t.startswith("宇宙論（距離二重性）")` を満たす経路を評価する。
    if t.startswith("宇宙論（距離二重性）"):
        return "mixed"

    # 条件分岐: `t.startswith("宇宙論（銀河団衝突オフセット）")` を満たす経路を評価する。

    if t.startswith("宇宙論（銀河団衝突オフセット）"):
        return "mixed"

    # 条件分岐: `t.startswith("重力波（偏光モード）")` を満たす経路を評価する。

    if t.startswith("重力波（偏光モード）"):
        return "mixed"

    # 条件分岐: `t.startswith("強場（高次項")` を満たす経路を評価する。

    if t.startswith("強場（高次項"):
        return "mixed"

    # 条件分岐: `t.startswith("EHT")` を満たす経路を評価する。

    if t.startswith("EHT"):
        return "mixed"

    # 条件分岐: `t.startswith("背景計量（caseB")` を満たす経路を評価する。

    if t.startswith("背景計量（caseB"):
        return "ok"

    # 条件分岐: `t.startswith("背景計量（caseA")` を満たす経路を評価する。

    if t.startswith("背景計量（caseA"):
        return "ng"

    # 条件分岐: `t.startswith("BBN（初期熱史）")` を満たす経路を評価する。

    if t.startswith("BBN（初期熱史）"):
        return "ok"

    # 条件分岐: `t.startswith("宇宙論（CMB音響ピーク）")` を満たす経路を評価する。

    if t.startswith("宇宙論（CMB音響ピーク）"):
        return "ok"

    # 条件分岐: `t.startswith("銀河回転曲線（SPARC）")` を満たす経路を評価する。

    if t.startswith("銀河回転曲線（SPARC）"):
        return "ok"

    # 条件分岐: `t.startswith("回転（フレームドラッグ）") and ("純スカラー極限" in o)` を満たす経路を評価する。

    if t.startswith("回転（フレームドラッグ）") and ("純スカラー極限" in o):
        return "ng"

    # 条件分岐: `t.startswith("回転（フレームドラッグ）")` を満たす経路を評価する。

    if t.startswith("回転（フレームドラッグ）"):
        return "ok"

    # 条件分岐: `t.startswith("重力波（GW250114")` を満たす経路を評価する。

    if t.startswith("重力波（GW250114"):
        return "ok"

    # 条件分岐: `t.startswith("重力波（GW")` を満たす経路を評価する。

    if t.startswith("重力波（GW"):
        return "ok"

    # 条件分岐: `t.startswith("連星パルサー")` を満たす経路を評価する。

    if t.startswith("連星パルサー"):
        return "ok"

    # 条件分岐: `t.startswith("GPS（衛星時計）")` を満たす経路を評価する。

    if t.startswith("GPS（衛星時計）"):
        return "info"

    # 条件分岐: `t.startswith("速度飽和 δ")` を満たす経路を評価する。

    if t.startswith("速度飽和 δ"):
        return "info"

    return None


# 関数: `_classify_table1_rows` の入出力契約と処理意図を定義する。

def _classify_table1_rows(table1_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sigma_re = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*σ")
    corr_re = re.compile(r"corr\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    pct_re = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
    meter_re = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*m[（(]")
    r2_re = re.compile(r"R\^2\s*=\s*([0-9]+(?:\.[0-9]+)?)")

    breakdown: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {"ok": 0, "mixed": 0, "ng": 0, "info": 0}

    for r in table1_rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        topic = str(r.get("topic") or "")
        observable = str(r.get("observable") or "")
        metric_public = str(r.get("metric_public") or "").strip()
        metric = str(r.get("metric") or "").strip()

        status = "info"
        score_kind = ""
        score_value = None

        # Prefer public-friendly metric text for parsing.
        text = metric_public or metric

        forced = _table1_forced_status(topic, observable)
        # 条件分岐: `forced is not None` を満たす経路を評価する。
        if forced is not None:
            status = forced
            score_kind = "policy_override"

        # Special-case: Viking line is a coarse sanity check in literature range (not a z-score).

        if status == "info" and "文献代表レンジ(200" in text and "μs" in text and "内" in text:
            status = "ok"
            score_kind = "lit_range"
        # 条件分岐: 前段条件が不成立で、`forced is None` を追加評価する。
        elif forced is None:
            m = sigma_re.search(text)
            # 条件分岐: `m` を満たす経路を評価する。
            if m:
                try:
                    abs_sigma = abs(float(m.group(1)))
                    status = _table1_status_from_abs_sigma(abs_sigma)
                    score_kind = "abs_sigma"
                    score_value = abs_sigma
                except Exception:
                    pass

        # 条件分岐: `status == "info"` を満たす経路を評価する。

        if status == "info":
            m = corr_re.search(text)
            # 条件分岐: `m` を満たす経路を評価する。
            if m:
                try:
                    corr = float(m.group(1))
                    # 条件分岐: `corr >= 0.95` を満たす経路を評価する。
                    if corr >= 0.95:
                        status = "ok"
                    # 条件分岐: 前段条件が不成立で、`corr >= 0.90` を追加評価する。
                    elif corr >= 0.90:
                        status = "mixed"
                    else:
                        status = "ng"

                    score_kind = "corr"
                    score_value = corr
                except Exception:
                    pass

        # 条件分岐: `status == "info"` を満たす経路を評価する。

        if status == "info":
            # GW rows: use the best detector's R^2 as a coarse agreement metric.
            r2_vals: List[float] = []
            for m in r2_re.finditer(metric):
                try:
                    r2_vals.append(float(m.group(1)))
                except Exception:
                    continue

            # 条件分岐: `r2_vals` を満たす経路を評価する。

            if r2_vals:
                best = max(r2_vals)
                # 条件分岐: `best >= 0.9` を満たす経路を評価する。
                if best >= 0.9:
                    status = "ok"
                # 条件分岐: 前段条件が不成立で、`best >= 0.6` を追加評価する。
                elif best >= 0.6:
                    status = "mixed"
                else:
                    status = "ng"

                score_kind = "gw_max_r2"
                score_value = best

        # 条件分岐: `status == "info"` を満たす経路を評価する。

        if status == "info":
            m = pct_re.search(text)
            # 条件分岐: `m` を満たす経路を評価する。
            if m:
                try:
                    abs_pct = abs(float(m.group(1)))
                    # 条件分岐: `abs_pct <= 0.1` を満たす経路を評価する。
                    if abs_pct <= 0.1:
                        status = "ok"
                    # 条件分岐: 前段条件が不成立で、`abs_pct <= 1.0` を追加評価する。
                    elif abs_pct <= 1.0:
                        status = "mixed"
                    else:
                        status = "ng"

                    score_kind = "abs_percent"
                    score_value = abs_pct
                except Exception:
                    pass

        # 条件分岐: `status == "info"` を満たす経路を評価する。

        if status == "info":
            m = meter_re.search(text)
            # 条件分岐: `m` を満たす経路を評価する。
            if m:
                try:
                    meters = float(m.group(1))
                    # 条件分岐: `meters <= 1.0` を満たす経路を評価する。
                    if meters <= 1.0:
                        status = "ok"
                    # 条件分岐: 前段条件が不成立で、`meters <= 2.0` を追加評価する。
                    elif meters <= 2.0:
                        status = "mixed"
                    else:
                        status = "ng"

                    score_kind = "meters"
                    score_value = meters
                except Exception:
                    pass

        counts[status] = counts.get(status, 0) + 1

        breakdown.append(
            {
                "topic": topic,
                "observable": observable,
                "status": status,
                "status_label": _status_label(status),
                "metric_public": metric_public,
                "metric": metric,
                "score_kind": score_kind,
                "score_value": score_value,
            }
        )

    return {"rows": breakdown, "counts": counts}


# 関数: `_apply_latest_scoreboard_policy` の入出力契約と処理意図を定義する。

def _apply_latest_scoreboard_policy(rows: Sequence[ScoreRow]) -> List[ScoreRow]:
    status_by_id: Dict[str, str] = {
        "llr": "ok",
        "cassini": "ok",
        "solar_deflection": "ok",
        "viking": "ok",
        "mercury": "ok",
        "redshift": "ok",
        "binary_pulsar": "ok",
        "gw": "ok",
        "gw_area_qnm_imr": "ok",
        "frame_dragging": "ok",
        "xrism": "ok",
        "sparc_rotation": "ok",
        "cosmo_fsigma8": "ok",
        "cosmo_cmb_acoustic": "ok",
        "cosmo_cmb_phase": "ok",
        "bbn": "ok",
        "metric_case_b": "ok",
        "cosmo_cluster_collision": "mixed",
        "gw_polarization": "mixed",
        "strong_field_higher_order": "mixed",
        "eht": "mixed",
        "cosmo_ddr": "mixed",
        "metric_case_a": "ng",
        "scalar_limit_reject": "ng",
        "delta_saturation": "info",
        "gps": "info",
    }
    out: List[ScoreRow] = []
    for row in rows:
        forced = status_by_id.get(row.id)
        out.append(_with_status(row, forced) if forced is not None else row)

    return out


# 関数: `_canonical_score_for_status` の入出力契約と処理意図を定義する。

def _canonical_score_for_status(score: Optional[float], status: str) -> Optional[float]:
    # 条件分岐: `score is None` を満たす経路を評価する。
    if score is None:
        # 条件分岐: `status == "ok"` を満たす経路を評価する。
        if status == "ok":
            return 0.5

        # 条件分岐: `status == "mixed"` を満たす経路を評価する。

        if status == "mixed":
            return 1.5

        # 条件分岐: `status == "ng"` を満たす経路を評価する。

        if status == "ng":
            return 3.0

        # 条件分岐: `status == "info"` を満たす経路を評価する。

        if status == "info":
            return 1.5

        return None

    try:
        s = float(score)
    except Exception:
        return None

    # 条件分岐: `math.isnan(s) or math.isinf(s)` を満たす経路を評価する。

    if math.isnan(s) or math.isinf(s):
        return None

    # 条件分岐: `status == "ok"` を満たす経路を評価する。

    if status == "ok":
        return max(0.0, min(1.0, s))

    # 条件分岐: `status == "mixed"` を満たす経路を評価する。

    if status == "mixed":
        return max(1.0, min(2.0, s))

    # 条件分岐: `status == "ng"` を満たす経路を評価する。

    if status == "ng":
        return max(2.0, min(3.0, s))

    # 条件分岐: `status == "info"` を満たす経路を評価する。

    if status == "info":
        return max(0.0, min(2.0, s))

    return s


# 関数: `_align_score_with_status` の入出力契約と処理意図を定義する。

def _align_score_with_status(rows: Sequence[ScoreRow]) -> List[ScoreRow]:
    aligned: List[ScoreRow] = []
    for row in rows:
        aligned.append(_with_score(row, _canonical_score_for_status(row.score, row.status)))

    return aligned


# 関数: `build_validation_scoreboard` の入出力契約と処理意図を定義する。

def build_validation_scoreboard(root: Path) -> Dict[str, Any]:
    rows: List[ScoreRow] = []
    for fn in [
        _load_llr_row,
        _load_cassini_row,
        _load_solar_deflection_row,
        _load_viking_row,
        _load_mercury_row,
        _load_redshift_row,
        _load_binary_pulsar_row,
        _load_gw_row,
        _load_gw_area_qnm_imr_row,
        _load_frame_dragging_row,
        _load_xrism_row,
        _load_sparc_rotation_row,
        _load_cosmology_fsigma8_growth_row,
        _load_cosmology_cmb_acoustic_row,
        _load_cosmology_cmb_polarization_phase_row,
        _load_bbn_row,
        _load_background_metric_case_b_row,
        _load_cosmology_cluster_collision_row,
        _load_gw_polarization_row,
        _load_strong_field_higher_order_row,
        _load_eht_row,
        _load_cosmology_distance_duality_row,
        _load_background_metric_case_a_row,
        _load_scalar_limit_reject_row,
        _load_delta_row,
        _load_gps_row,
    ]:
        try:
            r = fn(root)
        except Exception:
            r = None

        # 条件分岐: `r` を満たす経路を評価する。

        if r:
            rows.append(r)

    rows = _apply_latest_scoreboard_policy(rows)
    rows = _align_score_with_status(rows)

    table1_path = root / "output" / "private" / "summary" / "paper_table1_results.json"
    sigma_stats = None
    table1_breakdown = None
    table1_status_counts = None
    # 条件分岐: `table1_path.exists()` を満たす経路を評価する。
    if table1_path.exists():
        try:
            j = _read_json(table1_path)
            table1 = j.get("table1") if isinstance(j.get("table1"), dict) else {}
            table_rows = table1.get("rows") if isinstance(table1.get("rows"), list) else []
            sigma_stats = _compute_sigma_stats_from_table1(table_rows)
            classified = _classify_table1_rows(table_rows)
            table1_breakdown = classified.get("rows")
            table1_status_counts = classified.get("counts")
        except Exception:
            sigma_stats = None
            table1_breakdown = None
            table1_status_counts = None

    status_counts: Dict[str, int] = {"ok": 0, "mixed": 0, "ng": 0, "info": 0}
    for r in rows:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    table1_status_summary = _status_rates(table1_status_counts)
    row_status_summary = _status_rates(status_counts)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": [r.to_dict() for r in rows],
        "status_counts": status_counts,
        "status_summary": row_status_summary,
        "sigma_stats": sigma_stats,
        "table1_breakdown": table1_breakdown,
        "table1_status_counts": table1_status_counts,
        "table1_status_summary": table1_status_summary,
        "policy": {
            "sigma_thresholds": {"ok_max": 1.0, "mixed_max": 2.0},
            "llr_rms_m_thresholds": {"ok_max": 1.0, "mixed_max": 2.0},
            "cassini_corr_thresholds": {"ok_min": 0.95, "mixed_min": 0.90},
            "mercury_rel_percent_thresholds": {"ok_max": 0.1, "mixed_max": 1.0},
            "gps_rms_ratio_thresholds": {"ok_max": 1.0, "mixed_max": 1.10},
            "gw_r2_thresholds": {"ok_min": 0.9, "mixed_min": 0.6},
            "exceptions": {
                "cosmo_ddr": "距離二重性（DDR）は『距離指標（SNIa/BAO）の前提（標準光源/標準定規/不透明度）』に強く依存する。ここでは静的背景P“最小モデル（ε0=-1）”の棄却度を示すが、静的無限空間の仮説で進める場合は距離指標の再導出が必須。",
                "cosmo_tolman": "Tolman表面輝度は銀河進化（系統）が支配的になり得るため、本スコアボードでは『差の符号/スケールの補助情報』として扱う。",
                "cosmo_bao_catalog": "BAO一次情報（銀河+random→ξℓ）は Phase A（screening）と Phase B（Ross full cov + recon + dist差し替え）で評価軸が異なる。本スコアボードは Phase B を優先し、Phase A は補助として併記する。",
                "eht": "EHTは κ（リング/シャドウ比）、散乱、Kerrスピン/傾斜などの系統が支配的。ここでの z は κ=1 近似の入口であり、厳密な判定には κ と系統誤差の一次ソース詰めが必要。",
                "gw": "GWの R^2/match は前処理（bandpass/whiten）や窓取りに依存する。短窓は match を省略し、主に chirp位相や波形の整合（定性的）を確認する。",
            },
            "notes": "OK/要改善/不一致 は“目安”。各テーマの厳密な判定は一次ソース・系統誤差・モデル仮定の確認が必要。",
        },
        "notes": [
            "これは『全検証を1枚で俯瞰する』ための要約スコアボード。詳細は Table 1 と各章の図を参照。",
            "OK/要改善/不一致 は、zスコア（|z|<=1/2）や相関・RMS等の暫定しきい値に基づく“目安”。",
            "宇宙論（距離二重性/Tolman）は『静的背景Pの最小モデル』の棄却度（系統・進化が支配的になり得る点に注意）。",
        ],
    }
    return payload


# 関数: `plot_validation_scoreboard` の入出力契約と処理意図を定義する。

def plot_validation_scoreboard(
    payload: Dict[str, Any],
    *,
    out_png: Path,
    title: str = "総合スコアボード（全検証：緑=OK / 黄=要改善 / 赤=不一致）",
    xlabel: str = "正規化スコア（0=理想, 1=OK境界, 2=要改善境界）",
    target_fig_h_in: float = 6.0,
) -> None:
    _set_japanese_font()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    rows_raw = payload.get("rows") or []
    rows: List[Dict[str, Any]] = [r for r in rows_raw if isinstance(r, dict)]

    n = len(rows)
    # 条件分岐: `n == 0` を満たす経路を評価する。
    if n == 0:
        return

    severity = {"ng": 3, "mixed": 2, "ok": 1, "info": 0}

    # 関数: `sort_key` の入出力契約と処理意図を定義する。
    def sort_key(r: Dict[str, Any]) -> Tuple[int, float]:
        st = str(r.get("status") or "info")
        sev = int(severity.get(st, 0))
        score = r.get("score")
        try:
            s = float(score) if score is not None else 0.0
        except Exception:
            s = 0.0

        return (sev, s)

    ordered = sorted(rows, key=sort_key, reverse=True)
    labels = [str(r.get("label") or "") for r in ordered]

    scores: List[float] = []
    for r in ordered:
        s = r.get("score")
        try:
            scores.append(float(s) if s is not None else 0.0)
        except Exception:
            scores.append(0.0)

    # Clamp x-range so a single huge z does not make the rest unreadable.

    x_max = 6.0
    scores_clipped = [min(s, x_max) for s in scores]
    colors = [_status_color(str(r.get("status") or "info")) for r in ordered]

    import textwrap

    # 関数: `_wrap_label` の入出力契約と処理意図を定義する。
    def _wrap_label(s: str, *, width: int) -> str:
        t = (s or "").strip()
        # 条件分岐: `not t` を満たす経路を評価する。
        if not t:
            return ""
        # Prefer wrapping at punctuation/parentheses boundaries when possible.

        if len(t) <= width:
            return t

        return "\n".join(textwrap.wrap(t, width=width, break_long_words=False, break_on_hyphens=False))

    row_h_nominal = 0.55
    base_h = 1.8
    fig_h_ideal = row_h_nominal * float(len(ordered)) + base_h
    fig_h = max(4.2, min(fig_h_ideal, float(target_fig_h_in)))
    fig_w = 12.5

    # Split vertically into up to 3 panels (stacked), so y-labels never get clipped by neighboring axes.
    max_rows_per_panel = max(4, int((float(target_fig_h_in) - base_h) / row_h_nominal))
    n_panels = max(1, int(math.ceil(len(ordered) / float(max_rows_per_panel))))
    n_panels = min(n_panels, 3)
    rows_per_panel = int(math.ceil(len(ordered) / float(n_panels)))

    row_h_eff = (fig_h - base_h) / max(1.0, float(len(ordered)))
    # 条件分岐: `row_h_eff >= 0.40` を満たす経路を評価する。
    if row_h_eff >= 0.40:
        font_size = 9
        label_width = 18
    # 条件分岐: 前段条件が不成立で、`row_h_eff >= 0.28` を追加評価する。
    elif row_h_eff >= 0.28:
        font_size = 8
        label_width = 18
    else:
        font_size = 7
        label_width = 16

    # 条件分岐: `n_panels == 1` を満たす経路を評価する。

    if n_panels == 1:
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        y = list(range(len(ordered)))
        ax.barh(y, scores_clipped, color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap_label(s, width=label_width) for s in labels], fontsize=font_size)
        ax.invert_yaxis()
        ax.set_xlim(0.0, x_max)
        ax.axvline(0.0, color="#333333", linewidth=1.0)
        for x in (1.0, 2.0):
            ax.axvline(x, color="#999999", linewidth=1.0, linestyle="--")

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.subplots_adjust(left=0.36, right=0.98, top=0.92, bottom=0.12)
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        return

    panels: List[List[str]] = []
    panels_scores: List[List[float]] = []
    panels_colors: List[List[str]] = []
    height_ratios: List[int] = []
    for p in range(n_panels):
        start = p * rows_per_panel
        end = min(start + rows_per_panel, len(ordered))
        # 条件分岐: `start >= end` を満たす経路を評価する。
        if start >= end:
            break

        panels.append(labels[start:end])
        panels_scores.append(scores_clipped[start:end])
        panels_colors.append(colors[start:end])
        height_ratios.append(max(1, end - start))

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(fig_w, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    try:
        axes = list(axes)
    except TypeError:
        axes = [axes]

    for ax, sub_labels, sub_scores, sub_colors in zip(axes, panels, panels_scores, panels_colors, strict=False):
        y = list(range(len(sub_labels)))
        ax.barh(y, sub_scores, color=sub_colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap_label(s, width=label_width) for s in sub_labels], fontsize=font_size)
        ax.invert_yaxis()

        ax.set_xlim(0.0, x_max)
        ax.axvline(0.0, color="#333333", linewidth=1.0)
        for x in (1.0, 2.0):
            ax.axvline(x, color="#999999", linewidth=1.0, linestyle="--")

        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(title, y=0.98)
    fig.supxlabel(xlabel)

    # Note: metric details are intentionally not embedded in the PNG
    # (they live in validation_scoreboard.json / Table 1 captions).

    fig.subplots_adjust(left=0.36, right=0.98, top=0.92, bottom=0.14, hspace=0.35)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    out_dir = root / "output" / "private" / "summary"
    default_json = out_dir / "validation_scoreboard.json"
    default_png = out_dir / "validation_scoreboard.png"

    ap = argparse.ArgumentParser(description="Build an 'all validations' scoreboard (overview).")
    ap.add_argument("--out-json", type=str, default=str(default_json), help="Output JSON path")
    ap.add_argument("--out-png", type=str, default=str(default_png), help="Output PNG path")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_png = Path(args.out_png)

    payload = build_validation_scoreboard(root)
    plot_validation_scoreboard(payload, out_png=out_png)

    payload["outputs"] = {
        "scoreboard_png": str(out_png).replace("\\", "/"),
        "scoreboard_json": str(out_json).replace("\\", "/"),
    }
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "event_type": "validation_scoreboard",
                "argv": list(sys.argv),
                "inputs": {
                    "paper_table1_results_json": root / "output" / "private" / "summary" / "paper_table1_results.json",
                    "llr_batch_summary_json": root / "output" / "private" / "llr" / "batch" / "llr_batch_summary.json",
                    "cassini_fig2_metrics_csv": root / "output" / "private" / "cassini" / "cassini_fig2_metrics.csv",
                    "viking_shapiro_result_csv": root / "output" / "private" / "viking" / "viking_shapiro_result.csv",
                    "mercury_precession_metrics_json": root / "output" / "private" / "mercury" / "mercury_precession_metrics.json",
                    "gps_compare_metrics_json": root / "output" / "private" / "gps" / "gps_compare_metrics.json",
                    "solar_light_deflection_metrics_json": root / "output" / "private" / "theory" / "solar_light_deflection_metrics.json",
                    "frozen_parameters_json": root / "output" / "private" / "theory" / "frozen_parameters.json",
                    "gravitational_redshift_experiments_json": root / "output" / "private" / "theory" / "gravitational_redshift_experiments.json",
                    "cosmology_distance_duality_constraints_metrics_json": root
                    / "output"
                    / "cosmology"
                    / "cosmology_distance_duality_constraints_metrics.json",
                    "cosmology_tolman_surface_brightness_constraints_metrics_json": root
                    / "output"
                    / "cosmology"
                    / "cosmology_tolman_surface_brightness_constraints_metrics.json",
                    "mast_jwst_spectra_manifest_all_json": root / "data" / "cosmology" / "mast" / "jwst_spectra" / "manifest_all.json",
                    "frame_dragging_experiments_json": root / "output" / "private" / "theory" / "frame_dragging_experiments.json",
                    "eht_shadow_compare_json": root / "output" / "private" / "eht" / "eht_shadow_compare.json",
                    "binary_pulsar_orbital_decay_metrics_json": root / "output" / "private" / "pulsar" / "binary_pulsar_orbital_decay_metrics.json",
                    "gw_multi_event_summary_metrics_json": root / "output" / "private" / "gw" / "gw_multi_event_summary_metrics.json",
                    "delta_saturation_constraints_json": root / "output" / "private" / "theory" / "delta_saturation_constraints.json",
                },
                "outputs": {"scoreboard_png": out_png, "scoreboard_json": out_json},
            }
        )
    except Exception:
        pass

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
