from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.summary.validation_scoreboard import plot_validation_scoreboard  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score_norm_quantum(metric_public: str, metric_fallback: str, pmodel: str) -> Optional[float]:
    """
    Return a normalized discrepancy score in [0,3] (smaller is better) for the quantum Table 1.

    Rules (coarse, for an overview scoreboard):
    - Use |σ| / |z| when present.
    - Use max(Δ...) when a sweep metric is provided (e.g., Bell selection sensitivity).
    - Fall back to a categorical score inferred from the P-model column:
        - "same/weak-field mapping" => ~green
        - "entrance/constraint/target" => ~yellow
    """
    text = (metric_public or "").strip() or (metric_fallback or "").strip()

    sigma_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*σ", text)
    if sigma_m:
        try:
            return _clamp(abs(float(sigma_m.group(1))), 0.0, 3.0)
        except Exception:
            pass

    z_m = re.search(r"\bz\s*=\s*([+-]?[0-9]+(?:\.[0-9]+)?)\b", text)
    if z_m:
        try:
            return _clamp(abs(float(z_m.group(1))), 0.0, 3.0)
        except Exception:
            pass

    delta_vals: List[float] = []
    for m in re.finditer(r"Δ[^0-9+\-]*([+-]?[0-9]+(?:\.[0-9]+)?)", text):
        try:
            delta_vals.append(abs(float(m.group(1))))
        except Exception:
            continue
    if delta_vals:
        return _clamp(max(delta_vals) / 0.33, 0.0, 3.0)

    pm = (pmodel or "").strip()
    if pm:
        if any(k in pm for k in ("同", "同スケール", "弱場写像", "整合", "ε=0")):
            return 0.5
        if any(k in pm for k in ("入口", "基準値", "ターゲット", "固定", "再導出", "必要条件", "制約")):
            return 1.5

    return 1.5


def _status_from_score(score: Optional[float]) -> str:
    if score is None:
        return "info"
    if score <= 1.0:
        return "ok"
    if score <= 2.0:
        return "mixed"
    return "ng"


def build_quantum_scoreboard(root: Path) -> Dict[str, Any]:
    table1_json = root / "output" / "private" / "summary" / "paper_table1_quantum_results.json"
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "part3_quantum",
        "inputs": {"paper_table1_quantum_results_json": str(table1_json).replace("\\", "/")},
        "rows": [],
        "notes": [
            "これは Part III（量子）Table 1 を『1枚で俯瞰』するための要約スコアボード。",
            "OK/要改善/不一致 は、z/σ/手続き感度（Δ）の粗い proxy に基づく“目安”。厳密な判定は各節の棄却条件を正とする。",
        ],
    }

    if not table1_json.exists():
        return payload

    j = _read_json(table1_json)
    table1 = j.get("table1") if isinstance(j.get("table1"), dict) else {}
    rows = table1.get("rows") if isinstance(table1.get("rows"), list) else []

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        label = str(r.get("topic") or "").strip()
        metric = str(r.get("metric") or "")
        metric_public = str(r.get("metric_public") or "")
        pmodel = str(r.get("pmodel") or "")

        score = _score_norm_quantum(metric_public, metric, pmodel)
        status = _status_from_score(score)

        out_rows.append(
            {
                "id": re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower() or f"row_{len(out_rows)+1}",
                "label": label,
                "status": status,
                "score": score if score is not None and math.isfinite(float(score)) else None,
                "metric": (metric_public or metric).strip(),
                "detail": "",
                "sources": [str(table1_json).replace("\\", "/")],
                "score_kind": "table1_quantum_proxy",
            }
        )

    payload["rows"] = out_rows
    return payload


def main() -> int:
    root = _repo_root()
    out_dir = root / "output" / "private" / "summary"
    default_json = out_dir / "quantum_scoreboard.json"
    default_png = out_dir / "quantum_scoreboard.png"

    ap = argparse.ArgumentParser(description="Build a quantum-only (Part III) scoreboard (overview).")
    ap.add_argument("--out-json", type=str, default=str(default_json), help="Output JSON path")
    ap.add_argument("--out-png", type=str, default=str(default_png), help="Output PNG path")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_png = Path(args.out_png)

    payload = build_quantum_scoreboard(root)
    plot_validation_scoreboard(
        payload,
        out_png=out_png,
        title="総合スコアボード（量子：緑=OK / 黄=要改善 / 赤=不一致）",
        xlabel="正規化スコア（0=理想, 1=OK境界, 2=要改善境界）",
    )

    payload["outputs"] = {
        "scoreboard_png": str(out_png).replace("\\", "/"),
        "scoreboard_json": str(out_json).replace("\\", "/"),
    }
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "event_type": "quantum_scoreboard",
                "argv": list(sys.argv),
                "inputs": {
                    "paper_table1_quantum_results_json": root / "output" / "private" / "summary" / "paper_table1_quantum_results.json"
                },
                "outputs": {"scoreboard_png": out_png, "scoreboard_json": out_json},
            }
        )
    except Exception:
        pass

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
