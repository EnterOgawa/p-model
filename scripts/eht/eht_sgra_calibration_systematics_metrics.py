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
from typing import Any, Dict, Optional, Sequence

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


def _find_sentence_block(text: str, needle: str, *, window: int = 700) -> Optional[str]:
    i = text.find(needle)
    # 条件分岐: `i < 0` を満たす経路を評価する。
    if i < 0:
        return None

    a = max(0, i - 120)
    b = min(len(text), i + window)
    return text[a:b]


def _sigma_uniform_range(lo: float, hi: float) -> float:
    return abs(float(hi) - float(lo)) / math.sqrt(12.0)


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


def _parse_float_or_none(s: str) -> Optional[float]:
    t = s.strip()
    # 条件分岐: `not t or t in {"---", "--"}` を満たす経路を評価する。
    if not t or t in {"---", "--"}:
        return None

    try:
        v = float(t)
        return v if math.isfinite(v) else None
    except Exception:
        return None


@dataclass(frozen=True)
class SyntheticGainRow:
    station: str
    g_offset_mean: Optional[float]
    g_p_mean: Optional[float]
    g_offset_apr6: Optional[float]
    g_p_apr6: Optional[float]
    g_offset_apr7: Optional[float]
    g_p_apr7: Optional[float]
    source: Dict[str, Any]


def _parse_synthetic_gain_table(tex: str, *, source_path: Path) -> Dict[str, Any]:
    rows = []
    for lineno, raw in enumerate(tex.splitlines(), start=1):
        line = raw.strip()
        # 条件分岐: `not line or "&" not in line` を満たす経路を評価する。
        if not line or "&" not in line:
            continue

        # 条件分岐: `line.startswith("Station") or line.startswith("\\hline")` を満たす経路を評価する。

        if line.startswith("Station") or line.startswith("\\hline"):
            continue

        parts = [p.strip().rstrip("\\").strip() for p in line.split("&")]
        # 条件分岐: `len(parts) < 7` を満たす経路を評価する。
        if len(parts) < 7:
            continue

        station = parts[0]
        # 条件分岐: `station in {"ALMA", "APEX", "SMT", "JCMT", "LMT", "IRAM", "SMA", "SPT"}` を満たす経路を評価する。
        if station in {"ALMA", "APEX", "SMT", "JCMT", "LMT", "IRAM", "SMA", "SPT"}:
            g_offset_mean = _parse_float_or_none(parts[1])
            g_p_mean = _parse_float_or_none(parts[2])
            g_offset_apr6 = _parse_float_or_none(parts[3])
            g_p_apr6 = _parse_float_or_none(parts[4])
            g_offset_apr7 = _parse_float_or_none(parts[5])
            g_p_apr7 = _parse_float_or_none(parts[6])
            rows.append(
                SyntheticGainRow(
                    station=station,
                    g_offset_mean=g_offset_mean,
                    g_p_mean=g_p_mean,
                    g_offset_apr6=g_offset_apr6,
                    g_p_apr6=g_p_apr6,
                    g_offset_apr7=g_offset_apr7,
                    g_p_apr7=g_p_apr7,
                    source={"path": str(source_path), "line": int(lineno), "note": "appendix_synthetic.tex"},
                )
            )

    def _combine(a: Optional[float], b: Optional[float]) -> Optional[float]:
        # 条件分岐: `a is None or b is None` を満たす経路を評価する。
        if a is None or b is None:
            return None

        return math.sqrt(float(a) ** 2 + float(b) ** 2)

    combined_mean = [_combine(r.g_offset_mean, r.g_p_mean) for r in rows]
    combined_apr6 = [_combine(r.g_offset_apr6, r.g_p_apr6) for r in rows]
    combined_apr7 = [_combine(r.g_offset_apr7, r.g_p_apr7) for r in rows]

    return {
        "rows": [r.__dict__ for r in rows],
        "rows_n": int(len(rows)),
        "derived": {
            "g_offset_mean_summary": _summary([r.g_offset_mean for r in rows if r.g_offset_mean is not None]),
            "g_p_mean_summary": _summary([r.g_p_mean for r in rows if r.g_p_mean is not None]),
            "combined_mean_summary": _summary([v for v in combined_mean if v is not None]),
            "combined_apr6_summary": _summary([v for v in combined_apr6 if v is not None]),
            "combined_apr7_summary": _summary([v for v in combined_apr7 if v is not None]),
            "combined_mean_max": max((v for v in combined_mean if v is not None), default=None),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.09479" / "observations.tex"
    default_synth = root / "data" / "eht" / "sources" / "arxiv_2311.09479" / "appendix_synthetic.tex"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(description="Extract calibration/systematics scalars from Sgr A* Paper III TeX (observations).")
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper III observations.tex)")
    ap.add_argument(
        "--synthetic-tex",
        type=str,
        default=str(default_synth),
        help="Input TeX (default: Paper III appendix_synthetic.tex; gain table)",
    )
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/private/eht)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    synth_path = Path(args.synthetic_tex)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "eht_sgra_calibration_systematics_metrics.json"
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"tex": str(tex_path), "synthetic_tex": str(synth_path)},
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

    tex = _read_text(tex_path)

    # Gain uncertainty sentence.
    gain_snip = _find_sentence_block(tex, "a priori")
    payload["extracted"]["gain_uncertainty_anchor_snippet"] = gain_snip

    m_range = re.search(r"gains are\s+(\d+)\s*-\s*(\d+)\\%", tex)
    m_intra = re.search(r"intrasite baselines\s*\(\$\s*\\sim\s*(\d+)\\%\s*\$\)", tex)
    m_lmt = re.search(r"LMT\s*\(\$\s*\\sim\s*(\d+)\\%\s*\$\)", tex)

    # 条件分岐: `m_range` を満たす経路を評価する。
    if m_range:
        lo_pct = float(m_range.group(1))
        hi_pct = float(m_range.group(2))
        payload["extracted"]["gain_uncertainty_percent_range_typical"] = [lo_pct, hi_pct]
        payload["derived"]["gain_uncertainty_fraction_range_typical"] = [lo_pct / 100.0, hi_pct / 100.0]
        payload["derived"]["gain_uncertainty_fraction_sigma_uniform_typical"] = _sigma_uniform_range(
            lo_pct / 100.0, hi_pct / 100.0
        )

    # 条件分岐: `m_intra` を満たす経路を評価する。

    if m_intra:
        intra_pct = float(m_intra.group(1))
        payload["extracted"]["gain_uncertainty_percent_intrasite"] = intra_pct
        payload["derived"]["gain_uncertainty_fraction_intrasite"] = intra_pct / 100.0

    # 条件分岐: `m_lmt` を満たす経路を評価する。

    if m_lmt:
        lmt_pct = float(m_lmt.group(1))
        payload["extracted"]["gain_uncertainty_percent_lmt"] = lmt_pct
        payload["derived"]["gain_uncertainty_fraction_lmt"] = lmt_pct / 100.0

    # Non-closing errors sentence.

    nc_snip = _find_sentence_block(tex, "non-closing errors are estimated")
    payload["extracted"]["non_closing_anchor_snippet"] = nc_snip

    m_nc = re.search(r"estimated to be\s+(\d+)\\degree.*?and\s+(\d+)\\%\s+in log closure amplitude", tex, flags=re.S)
    m_tr = re.search(
        r"translate to\s+(\d+)\\degree.*?and\s+(\d+)\\%\s+systematic non-closing uncertainties in visibility amplitudes",
        tex,
        flags=re.S,
    )
    # 条件分岐: `m_nc` を満たす経路を評価する。
    if m_nc:
        cphase_deg = float(m_nc.group(1))
        lca_pct = float(m_nc.group(2))
        payload["extracted"]["non_closing_closure_phase_deg"] = cphase_deg
        payload["extracted"]["non_closing_log_closure_amp_percent"] = lca_pct
        payload["derived"]["non_closing_log_closure_amp_fraction"] = lca_pct / 100.0

    # 条件分岐: `m_tr` を満たす経路を評価する。

    if m_tr:
        vis_phase_deg = float(m_tr.group(1))
        vis_amp_pct = float(m_tr.group(2))
        payload["extracted"]["non_closing_vis_phase_deg"] = vis_phase_deg
        payload["extracted"]["non_closing_vis_amp_percent"] = vis_amp_pct
        payload["derived"]["non_closing_vis_amp_fraction"] = vis_amp_pct / 100.0

    # Synthetic gain table (appendix_synthetic).

    if synth_path.exists():
        synth_tex = _read_text(synth_path)
        payload["extracted"]["synthetic_gain_table"] = _parse_synthetic_gain_table(synth_tex, source_path=synth_path)
        combined_max = ((payload["extracted"]["synthetic_gain_table"] or {}).get("derived") or {}).get("combined_mean_max")
        # 条件分岐: `combined_max is not None` を満たす経路を評価する。
        if combined_max is not None:
            payload["derived"]["gain_synthetic_combined_mean_max"] = float(combined_max)
    else:
        payload["extracted"]["synthetic_gain_table"] = {"ok": False, "reason": "missing_synthetic_tex", "path": str(synth_path)}

    # Sanity: require at least the key scalars.

    required_keys = [
        "gain_uncertainty_percent_range_typical",
        "gain_uncertainty_percent_lmt",
        "gain_uncertainty_percent_intrasite",
        "non_closing_vis_amp_percent",
        "non_closing_closure_phase_deg",
    ]
    missing = [k for k in required_keys if k not in payload["extracted"]]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        payload["ok"] = False
        payload["reason"] = "missing_expected_fields"
        payload["missing"] = missing

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_calibration_systematics_metrics",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {"ok": bool(payload.get("ok")), "missing": len(missing)},
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
