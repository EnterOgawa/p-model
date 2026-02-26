#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    # 条件分岐: `not xs` を満たす経路を評価する。
    if not xs:
        return {"n": 0}

    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mean = sum(xs_sorted) / n
    # 条件分岐: `n >= 2` を満たす経路を評価する。
    if n >= 2:
        var = sum((x - mean) ** 2 for x in xs_sorted) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0

    median = xs_sorted[n // 2] if (n % 2 == 1) else 0.5 * (xs_sorted[n // 2 - 1] + xs_sorted[n // 2])
    return {
        "n": int(n),
        "mean": float(mean),
        "std": float(std),
        "median": float(median),
        "min": float(xs_sorted[0]),
        "max": float(xs_sorted[-1]),
    }


def _clean_test_name(s: str) -> str:
    t = s
    t = t.replace(r"\,$-$\,", "-")
    t = t.replace(r"\,$-$", "-")
    t = t.replace(r"$-$", "-")
    t = t.replace(r"\,", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _parse_float(s: str) -> Optional[float]:
    t = s.strip()
    # 条件分岐: `not t` を満たす経路を評価する。
    if not t:
        return None

    try:
        v = float(t)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _parse_s_token(token: str) -> Optional[Dict[str, Any]]:
    t = token.strip()
    # 条件分岐: `not t` を満たす経路を評価する。
    if not t:
        return None

    t = t.replace("$", "")
    # 条件分岐: `r"\%" in t` を満たす経路を評価する。
    if r"\%" in t:
        unit = "percent"
        v = _parse_float(t.replace(r"\%", "").strip())
        # 条件分岐: `v is None` を満たす経路を評価する。
        if v is None:
            return None

        return {"value": float(v), "unit": unit}

    # 条件分岐: `r"\degr" in t` を満たす経路を評価する。

    if r"\degr" in t:
        unit = "deg"
        v = _parse_float(t.replace(r"\degr", "").strip())
        # 条件分岐: `v is None` を満たす経路を評価する。
        if v is None:
            return None

        return {"value": float(v), "unit": unit}

    v = _parse_float(t)
    # 条件分岐: `v is None` を満たす経路を評価する。
    if v is None:
        return None

    return {"value": float(v), "unit": "unknown"}


@dataclass(frozen=True)
class SysErrRow:
    test_tex: str
    test: str
    casa_s: Dict[str, Any]
    casa_s_over_sigma_th: Optional[float]
    casa_n: Optional[int]
    hops_s: Dict[str, Any]
    hops_s_over_sigma_th: Optional[float]
    hops_n: Optional[int]
    source_anchor: Dict[str, Any]


def _parse_int(s: str) -> Optional[int]:
    t = s.strip()
    # 条件分岐: `not t` を満たす経路を評価する。
    if not t:
        return None

    try:
        return int(t)
    except Exception:
        return None


def _extract_sgra_syserr_table_rows(tex_lines: Sequence[str], *, source_path: Path) -> List[SysErrRow]:
    rows: List[SysErrRow] = []
    in_sgra = False
    for lineno, raw in enumerate(tex_lines, start=1):
        line = raw.strip()
        # 条件分岐: `not in_sgra` を満たす経路を評価する。
        if not in_sgra:
            # 条件分岐: `line.startswith(r"\sgra&")` を満たす経路を評価する。
            if line.startswith(r"\sgra&"):
                in_sgra = True
            else:
                continue

        # 条件分岐: `line.startswith(r"\hline")` を満たす経路を評価する。

        if line.startswith(r"\hline"):
            break

        # 条件分岐: `"&" not in line or r"\\" not in line` を満たす経路を評価する。

        if "&" not in line or r"\\" not in line:
            continue

        parts = [p.strip() for p in line.split("&")]
        # 条件分岐: `len(parts) < 8` を満たす経路を評価する。
        if len(parts) < 8:
            continue

        test_tex = parts[1]
        test = _clean_test_name(test_tex)

        casa_s = _parse_s_token(parts[2])
        hops_s = _parse_s_token(parts[5])
        # 条件分岐: `casa_s is None or hops_s is None` を満たす経路を評価する。
        if casa_s is None or hops_s is None:
            continue

        casa_ratio = _parse_float(parts[3])
        casa_n = _parse_int(parts[4])
        hops_ratio = _parse_float(parts[6])
        hops_n = _parse_int(parts[7].replace(r"\\", "").strip())

        rows.append(
            SysErrRow(
                test_tex=test_tex,
                test=test,
                casa_s=casa_s,
                casa_s_over_sigma_th=casa_ratio,
                casa_n=casa_n,
                hops_s=hops_s,
                hops_s_over_sigma_th=hops_ratio,
                hops_n=hops_n,
                source_anchor={"path": str(source_path), "line": int(lineno), "label": "tab:syserr"},
            )
        )

    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.08679" / "main.tex"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(
        description="Extract Sgr A* non-closing systematic error budget table (tab:syserr) from Paper II (arXiv:2311.08679) TeX."
    )
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper II main.tex)")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/private/eht)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "eht_sgra_paper2_syserr_table_metrics.json"

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"tex": str(tex_path)},
        "ok": True,
        "extracted": {},
        "derived": {},
        "outputs": {"json": str(out_json)},
    }

    # 条件分岐: `not tex_path.exists()` を満たす経路を評価する。
    if not tex_path.exists():
        payload["ok"] = False
        payload["reason"] = "missing_input_tex"
        _write_json(out_json, payload)
        print(f"[warn] missing input; wrote: {out_json}")
        return 0

    lines = _read_text(tex_path).splitlines()
    rows = _extract_sgra_syserr_table_rows(lines, source_path=tex_path)
    payload["extracted"]["rows"] = [r.__dict__ for r in rows]
    payload["extracted"]["rows_n"] = int(len(rows))
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        payload["ok"] = False
        payload["reason"] = "sgra_rows_not_found"
        _write_json(out_json, payload)
        print(f"[warn] sgra rows not found; wrote: {out_json}")
        return 0

    # Derived summaries (use only percent-valued rows as amplitude-like scales).

    casa_amp_pct = [r.casa_s["value"] for r in rows if r.casa_s.get("unit") == "percent"]
    hops_amp_pct = [r.hops_s["value"] for r in rows if r.hops_s.get("unit") == "percent"]
    both_amp_pct = casa_amp_pct + hops_amp_pct

    casa_phase_deg = [r.casa_s["value"] for r in rows if r.casa_s.get("unit") == "deg"]
    hops_phase_deg = [r.hops_s["value"] for r in rows if r.hops_s.get("unit") == "deg"]
    both_phase_deg = casa_phase_deg + hops_phase_deg

    amp_pct_max = max(both_amp_pct) if both_amp_pct else None
    payload["derived"] = {
        "sgra_amp_percent_summary_casa": _summary(casa_amp_pct),
        "sgra_amp_percent_summary_hops": _summary(hops_amp_pct),
        "sgra_amp_percent_summary_both": _summary(both_amp_pct),
        "sgra_phase_deg_summary_casa": _summary(casa_phase_deg),
        "sgra_phase_deg_summary_hops": _summary(hops_phase_deg),
        "sgra_phase_deg_summary_both": _summary(both_phase_deg),
        "sgra_amp_fraction_max_over_pipelines": float(amp_pct_max) / 100.0 if amp_pct_max is not None else None,
        "notes": [
            "tab:syserr reports s (systematic) for closure quantities; percent-valued rows are used here as amplitude-like systematic scales.",
            "These values are not a direct mapping to the published ring diameter uncertainty; treat as a calibration/systematics scale indicator.",
        ],
    }

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper2_syserr_table_metrics",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {
                    "ok": bool(payload.get("ok")),
                    "rows_n": int(payload["extracted"].get("rows_n") or 0),
                    "sgra_amp_fraction_max_over_pipelines": payload["derived"].get("sgra_amp_fraction_max_over_pipelines"),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
