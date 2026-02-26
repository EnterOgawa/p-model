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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_read_text` の入出力契約と処理意図を定義する。

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: `_find_block` の入出力契約と処理意図を定義する。

def _find_block(text: str, needle: str, *, window: int = 1100) -> Optional[str]:
    i = text.find(needle)
    # 条件分岐: `i < 0` を満たす経路を評価する。
    if i < 0:
        return None

    a = max(0, i - 160)
    b = min(len(text), i + window)
    return text[a:b]


# 関数: `_unwrap_multirow_cell` の入出力契約と処理意図を定義する。

def _unwrap_multirow_cell(s: str) -> str:
    s = str(s).strip()
    m = re.match(r"^\\multirow\{[^}]+\}\{[^}]+\}\{(.+)\}$", s)
    # 条件分岐: `m` を満たす経路を評価する。
    if m:
        return m.group(1).strip()

    return s


# 関数: `_tex_to_plain` の入出力契約と処理意図を定義する。

def _tex_to_plain(s: str) -> str:
    s = str(s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# 関数: `_parse_pm_tuple` の入出力契約と処理意図を定義する。

def _parse_pm_tuple(s: str) -> Optional[Tuple[float, float]]:
    """
    Parse (+x, -y) into (x, y) with both positive floats.
    """
    raw = str(s).strip()
    # 条件分岐: `not raw` を満たす経路を評価する。
    if not raw:
        return None

    # 条件分岐: `raw.startswith("(") and raw.endswith(")")` を満たす経路を評価する。

    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()

    parts = [p.strip() for p in raw.split(",")]
    # 条件分岐: `len(parts) != 2` を満たす経路を評価する。
    if len(parts) != 2:
        return None

    try:
        plus = float(parts[0].replace("+", "").strip())
        minus = float(parts[1].replace("-", "").replace("+", "").strip())
    except Exception:
        return None

    # 条件分岐: `not (math.isfinite(plus) and math.isfinite(minus))` を満たす経路を評価する。

    if not (math.isfinite(plus) and math.isfinite(minus)):
        return None

    return (abs(plus), abs(minus))


# 関数: `_sym_sigma` の入出力契約と処理意図を定義する。

def _sym_sigma(pm: Tuple[float, float]) -> float:
    return 0.5 * (float(pm[0]) + float(pm[1]))


# クラス: `AlphaCalRow` の責務と境界条件を定義する。

@dataclass(frozen=True)
class AlphaCalRow:
    analysis_class: str
    software_plain: str
    software_tex: str
    day: str
    alpha: float
    sigma_alpha_stat_pm: Tuple[float, float]
    sigma_alpha_tot_pm: Tuple[float, float]
    source_anchor: Dict[str, Any]


# 関数: `_parse_alphacal_table` の入出力契約と処理意図を定義する。

def _parse_alphacal_table(tex: str, *, source_path: Path) -> List[AlphaCalRow]:
    label = "\\label{tab:alphacal}"
    lines = tex.splitlines()

    label_idx = None
    for i, line in enumerate(lines):
        # 条件分岐: `label in line` を満たす経路を評価する。
        if label in line:
            label_idx = i
            break

    # 条件分岐: `label_idx is None` を満たす経路を評価する。

    if label_idx is None:
        return []

    startdata_idx = None
    for j in range(label_idx, len(lines)):
        # 条件分岐: `"\\startdata" in lines[j]` を満たす経路を評価する。
        if "\\startdata" in lines[j]:
            startdata_idx = j
            break

    # 条件分岐: `startdata_idx is None` を満たす経路を評価する。

    if startdata_idx is None:
        return []

    enddata_idx = None
    for j in range(startdata_idx, len(lines)):
        # 条件分岐: `"\\enddata" in lines[j]` を満たす経路を評価する。
        if "\\enddata" in lines[j]:
            enddata_idx = j
            break

    # 条件分岐: `enddata_idx is None` を満たす経路を評価する。

    if enddata_idx is None:
        enddata_idx = len(lines)

    rows: List[AlphaCalRow] = []
    cur_class: Optional[str] = None

    buf = ""
    buf_start_lineno: Optional[int] = None

    # 関数: `_flush_row` の入出力契約と処理意図を定義する。
    def _flush_row(row_text: str, *, lineno: int) -> None:
        nonlocal cur_class, rows
        t = str(row_text).strip()
        # 条件分岐: `not t` を満たす経路を評価する。
        if not t:
            return

        t = t.replace("\\\\", "").strip()
        # 条件分岐: `not t or t.startswith("\\hline") or t.startswith("\\cline")` を満たす経路を評価する。
        if not t or t.startswith("\\hline") or t.startswith("\\cline"):
            return

        parts = [p.strip() for p in t.split("&")]
        # 条件分岐: `len(parts) < 6` を満たす経路を評価する。
        if len(parts) < 6:
            return

        cell0 = _unwrap_multirow_cell(parts[0])
        # 条件分岐: `cell0` を満たす経路を評価する。
        if cell0:
            cur_class = _tex_to_plain(cell0)

        # 条件分岐: `not cur_class` を満たす経路を評価する。

        if not cur_class:
            return

        cell1_tex = _unwrap_multirow_cell(parts[1])
        cell1_plain = _tex_to_plain(cell1_tex)
        day = _tex_to_plain(parts[2])

        try:
            alpha = float(_tex_to_plain(parts[3]))
        except Exception:
            return

        stat_pm = _parse_pm_tuple(parts[4])
        tot_pm = _parse_pm_tuple(parts[5])
        # 条件分岐: `stat_pm is None or tot_pm is None` を満たす経路を評価する。
        if stat_pm is None or tot_pm is None:
            return

        rows.append(
            AlphaCalRow(
                analysis_class=str(cur_class),
                software_plain=str(cell1_plain),
                software_tex=str(cell1_tex).strip(),
                day=str(day),
                alpha=float(alpha),
                sigma_alpha_stat_pm=stat_pm,
                sigma_alpha_tot_pm=tot_pm,
                source_anchor={"path": str(source_path), "line": int(lineno), "label": "tab:alphacal"},
            )
        )

    for off, raw in enumerate(lines[startdata_idx + 1 : enddata_idx], start=0):
        lineno = (startdata_idx + 2) + off  # 1-based lineno for this raw line
        s = raw.strip()
        # 条件分岐: `not s` を満たす経路を評価する。
        if not s:
            continue

        # 条件分岐: `s.startswith("\\hline") or s.startswith("\\cline")` を満たす経路を評価する。

        if s.startswith("\\hline") or s.startswith("\\cline"):
            continue

        # 条件分岐: `buf_start_lineno is None` を満たす経路を評価する。

        if buf_start_lineno is None:
            buf_start_lineno = int(lineno)

        buf = (buf + " " + s).strip()
        # 条件分岐: `"\\\\" not in s` を満たす経路を評価する。
        if "\\\\" not in s:
            continue

        _flush_row(buf, lineno=int(buf_start_lineno))
        buf = ""
        buf_start_lineno = None

    return rows


# 関数: `_summary` の入出力契約と処理意図を定義する。

def _summary(values: Sequence[float]) -> Dict[str, Any]:
    x = np.array(list(values), dtype=float)
    x = x[np.isfinite(x)]
    # 条件分岐: `x.size == 0` を満たす経路を評価する。
    if x.size == 0:
        return {"n": 0}

    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size >= 2 else 0.0,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.08697" / "results.tex"
    default_shadow = root / "output" / "private" / "eht" / "eht_shadow_compare.json"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(description="Parse Sgr A* Paper IV alpha calibration table (tab:alphacal).")
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper IV results.tex)")
    ap.add_argument(
        "--shadow-compare-json",
        type=str,
        default=str(default_shadow),
        help="eht_shadow_compare.json for shadow coeffs (default: output/private/eht/eht_shadow_compare.json)",
    )
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/private/eht)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    shadow_path = Path(args.shadow_compare_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "eht_sgra_paper4_alpha_calibration_metrics.json"
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"tex": str(tex_path), "shadow_compare_json": str(shadow_path)},
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
    payload["extracted"]["alphacal_anchor_snippet"] = _find_block(tex, "\\label{tab:alphacal}")

    rows = _parse_alphacal_table(tex, source_path=tex_path)
    payload["extracted"]["rows_n"] = int(len(rows))
    payload["extracted"]["rows"] = [
        {
            "analysis_class": r.analysis_class,
            "software": r.software_plain,
            "software_tex": r.software_tex,
            "day": r.day,
            "alpha": r.alpha,
            "sigma_alpha_stat_pm": {"plus": r.sigma_alpha_stat_pm[0], "minus": r.sigma_alpha_stat_pm[1]},
            "sigma_alpha_tot_pm": {"plus": r.sigma_alpha_tot_pm[0], "minus": r.sigma_alpha_tot_pm[1]},
            "source_anchor": r.source_anchor,
        }
        for r in rows
    ]

    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        payload["ok"] = False
        payload["reason"] = "no_rows_parsed"
        _write_json(out_json, payload)
        print(f"[warn] no rows; wrote: {out_json}")
        return 0

    coeff_gr = None
    coeff_p = None
    # 条件分岐: `shadow_path.exists()` を満たす経路を評価する。
    if shadow_path.exists():
        try:
            shadow = _read_json(shadow_path)
            coeff_gr = float(((shadow.get("reference_gr") or {}).get("shadow_diameter_coeff_rg")))
            coeff_p = float(((shadow.get("pmodel") or {}).get("shadow_diameter_coeff_rg")))
        except Exception:
            coeff_gr = None
            coeff_p = None

    sig_tot_sym = [_sym_sigma(r.sigma_alpha_tot_pm) for r in rows]
    sig_stat_sym = [_sym_sigma(r.sigma_alpha_stat_pm) for r in rows]

    derived: Dict[str, Any] = {
        "alpha_summary": _summary([r.alpha for r in rows]),
        "sigma_alpha_stat_sym_summary": _summary(sig_stat_sym),
        "sigma_alpha_tot_sym_summary": _summary(sig_tot_sym),
        "notes": [
            "Paper IV defines d = α θ_g and calibrates α using a GRMHD synthetic suite (Paper V).",
            "The α uncertainty includes statistical and (dominant) theory/morphology components; this file reports the quoted 68% intervals.",
        ],
    }

    # 条件分岐: `coeff_gr is not None and math.isfinite(coeff_gr) and coeff_gr > 0` を満たす経路を評価する。
    if coeff_gr is not None and math.isfinite(coeff_gr) and coeff_gr > 0:
        sig_kappa_tot_sym = [float(s) / float(coeff_gr) for s in sig_tot_sym]
        derived["kappa_sigma_proxy_gr_from_alpha_tot_sym_min"] = float(np.min(sig_kappa_tot_sym))
        derived["kappa_sigma_proxy_gr_from_alpha_tot_sym_median"] = float(np.median(sig_kappa_tot_sym))
        derived["kappa_sigma_proxy_gr_from_alpha_tot_sym_max"] = float(np.max(sig_kappa_tot_sym))
        derived["shadow_coeff_gr"] = float(coeff_gr)

    # 条件分岐: `coeff_p is not None and math.isfinite(coeff_p) and coeff_p > 0` を満たす経路を評価する。

    if coeff_p is not None and math.isfinite(coeff_p) and coeff_p > 0:
        sig_kappa_tot_sym_p = [float(s) / float(coeff_p) for s in sig_tot_sym]
        derived["kappa_sigma_proxy_p_from_alpha_tot_sym_min"] = float(np.min(sig_kappa_tot_sym_p))
        derived["kappa_sigma_proxy_p_from_alpha_tot_sym_median"] = float(np.median(sig_kappa_tot_sym_p))
        derived["kappa_sigma_proxy_p_from_alpha_tot_sym_max"] = float(np.max(sig_kappa_tot_sym_p))
        derived["shadow_coeff_pmodel"] = float(coeff_p)

    payload["derived"] = derived

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper4_alpha_calibration_metrics",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {"ok": bool(payload.get("ok")), "rows_n": int(len(rows))},
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
