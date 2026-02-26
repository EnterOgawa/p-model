#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _parse_float(token: str) -> Optional[float]:
    t = token.strip()
    # 条件分岐: `not t` を満たす経路を評価する。
    if not t:
        return None

    # 条件分岐: `t in {"--", "---"}` を満たす経路を評価する。

    if t in {"--", "---"}:
        return None

    try:
        v = float(t)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _parse_cell_values(cell: str) -> List[float]:
    t = cell.strip()
    # 条件分岐: `not t or t in {"--", "---"}` を満たす経路を評価する。
    if not t or t in {"--", "---"}:
        return []

    parts = [p.strip() for p in t.split(",")]
    out: List[float] = []
    for p in parts:
        v = _parse_float(p)
        # 条件分岐: `v is None` を満たす経路を評価する。
        if v is None:
            continue

        out.append(float(v))

    return out


def _parse_latex_sci(token: str) -> Optional[float]:
    t = token.strip()
    # 条件分岐: `not t` を満たす経路を評価する。
    if not t:
        return None

    t = t.replace("$", "")
    t = re.sub(r"\s+", "", t)
    m = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\\times10\^\{([+-]?\d+)\}$", t)
    # 条件分岐: `m` を満たす経路を評価する。
    if m:
        try:
            return float(m.group(1)) * (10.0 ** int(m.group(2)))
        except Exception:
            return None

    try:
        return float(t)
    except Exception:
        return None


def _strip_unescaped_tex_comment(line: str) -> str:
    # TeX comment starts at first unescaped '%'.
    return re.split(r"(?<!\\)%", line, maxsplit=1)[0]


def _extract_table_rows(tex_lines: Sequence[str]) -> List[Tuple[int, str]]:
    start_idx = None
    for i, line in enumerate(tex_lines):
        # 条件分岐: `"\\startdata" in line` を満たす経路を評価する。
        if "\\startdata" in line:
            start_idx = i
            break

    # 条件分岐: `start_idx is None` を満たす経路を評価する。

    if start_idx is None:
        return []

    end_idx = None
    for j in range(start_idx, len(tex_lines)):
        # 条件分岐: `"\\enddata" in tex_lines[j]` を満たす経路を評価する。
        if "\\enddata" in tex_lines[j]:
            end_idx = j
            break

    # 条件分岐: `end_idx is None` を満たす経路を評価する。

    if end_idx is None:
        end_idx = len(tex_lines)

    rows: List[Tuple[int, str]] = []
    for off, raw in enumerate(tex_lines[start_idx + 1 : end_idx], start=0):
        lineno = (start_idx + 2) + off
        s = raw.strip()
        # 条件分岐: `s.startswith("%")` を満たす経路を評価する。
        if s.startswith("%"):
            continue

        s = _strip_unescaped_tex_comment(s).strip()
        # 条件分岐: `not s` を満たす経路を評価する。
        if not s:
            continue

        # 条件分岐: `s.startswith("\\hline")` を満たす経路を評価する。

        if s.startswith("\\hline"):
            continue

        # 条件分岐: `"&" not in s` を満たす経路を評価する。

        if "&" not in s:
            continue

        s = s.replace("\\\\", "").strip()
        rows.append((int(lineno), s))

    return rows


def _rank_constraints(by_constraint: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for name, rec in by_constraint.items():
        values = rec.get("values_all") if isinstance(rec.get("values_all"), list) else []
        vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
        # 条件分岐: `not vals` を満たす経路を評価する。
        if not vals:
            continue

        items.append(
            {
                "constraint": name,
                "n": int(len(vals)),
                "min": float(min(vals)),
                "max": float(max(vals)),
                "mean": float(sum(vals) / len(vals)),
            }
        )

    items.sort(key=lambda x: (x["mean"], x["min"]))
    return items


def _summarize_pass_fail_rows(
    rows: Sequence[Dict[str, Any]],
    pass_fail_columns: Sequence[str],
    *,
    exclude_from_fail_modes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    n = int(len(rows))
    summary: Dict[str, Any] = {
        "rows_n": n,
        "by_constraint": {},
        "constraints_ranked_by_fail_fraction": [],
        "fail_modes_top": [],
        "fail_count_histogram": {},
    }
    # 条件分岐: `n == 0` を満たす経路を評価する。
    if n == 0:
        return summary

    exclude = set(exclude_from_fail_modes or [])
    counts: Dict[str, Dict[str, int]] = {c: {"pass": 0, "fail": 0, "unknown": 0} for c in pass_fail_columns}
    fail_modes: Dict[str, int] = {}
    fail_hist: Dict[int, int] = {}

    for row in rows:
        pf = row.get("pass_fail")
        # 条件分岐: `not isinstance(pf, dict)` を満たす経路を評価する。
        if not isinstance(pf, dict):
            continue

        failed_cols = []
        fail_count = 0
        for c in pass_fail_columns:
            v = pf.get(c)
            # 条件分岐: `v == "Pass"` を満たす経路を評価する。
            if v == "Pass":
                counts[c]["pass"] += 1
            # 条件分岐: 前段条件が不成立で、`v == "Fail"` を追加評価する。
            elif v == "Fail":
                counts[c]["fail"] += 1
                fail_count += 1
                # 条件分岐: `c not in exclude` を満たす経路を評価する。
                if c not in exclude:
                    failed_cols.append(c)
            else:
                counts[c]["unknown"] += 1

        fail_hist[fail_count] = int(fail_hist.get(fail_count, 0) + 1)

        # 条件分岐: `failed_cols` を満たす経路を評価する。
        if failed_cols:
            key = "|".join(sorted(failed_cols))
            fail_modes[key] = int(fail_modes.get(key, 0) + 1)

    by_constraint = {}
    ranked = []
    for c in pass_fail_columns:
        rec = counts.get(c, {})
        pass_n = int(rec.get("pass", 0))
        fail_n = int(rec.get("fail", 0))
        unknown_n = int(rec.get("unknown", 0))
        pass_frac = float(pass_n / n)
        fail_frac = float(fail_n / n)
        by_constraint[c] = {
            "pass_n": pass_n,
            "fail_n": fail_n,
            "unknown_n": unknown_n,
            "pass_fraction": pass_frac,
            "fail_fraction": fail_frac,
        }
        ranked.append({"constraint": c, "fail_fraction": fail_frac, "pass_fraction": pass_frac, "n": n})

    ranked.sort(key=lambda x: (-x["fail_fraction"], x["constraint"]))

    ranked_excl = [x for x in ranked if x.get("constraint") not in exclude]

    fail_modes_top = [{"failed_constraints_key": k, "count": int(v)} for k, v in fail_modes.items()]
    fail_modes_top.sort(key=lambda x: (-x["count"], x["failed_constraints_key"]))
    fail_modes_top = fail_modes_top[:20]

    summary["by_constraint"] = by_constraint
    summary["constraints_ranked_by_fail_fraction"] = ranked
    summary["constraints_ranked_by_fail_fraction_excluding_aggregates"] = ranked_excl
    summary["excluded_from_fail_modes"] = sorted(exclude)
    summary["fail_modes_top"] = fail_modes_top
    summary["fail_count_histogram"] = {str(k): int(v) for k, v in sorted(fail_hist.items(), key=lambda kv: int(kv[0]))}
    return summary


def _normalize_constraint_token_paper5(token: str) -> Optional[str]:
    t = token.strip()
    # 条件分岐: `not t` を満たす経路を評価する。
    if not t:
        return None

    # 条件分岐: `t in {"??", "--", "---"}` を満たす経路を評価する。

    if t in {"??", "--", "---"}:
        return None

    # Keep some TeX macros for matching.

    if re.search(r"\\mi\{\s*3\s*\}", t):
        return "M3"

    # 条件分岐: `"Mring width" in t or "\\Mring width" in t` を満たす経路を評価する。

    if "Mring width" in t or "\\Mring width" in t:
        return "Ring_W"

    # 条件分岐: `"Mring diameter" in t or "\\Mring diameter" in t` を満たす経路を評価する。

    if "Mring diameter" in t or "\\Mring diameter" in t:
        return "Ring_D"

    # 条件分岐: `"Mring asymmetry" in t or "\\Mring asymmetry" in t` を満たす経路を評価する。

    if "Mring asymmetry" in t or "\\Mring asymmetry" in t:
        return "Ring_A"

    # 条件分岐: `"86" in t and "GHz" in t` を満たす経路を評価する。

    if "86" in t and "GHz" in t:
        # 条件分岐: `"flux" in t` を満たす経路を評価する。
        if "flux" in t:
            return "F86"

        # 条件分岐: `"size" in t` を満たす経路を評価する。

        if "size" in t:
            return "lambda_maj86"

    # 条件分岐: `"2.2" in t and ("\\mu" in t or "μ" in t or "mu" in t) and "flux" in t` を満たす経路を評価する。

    if "2.2" in t and ("\\mu" in t or "μ" in t or "mu" in t) and "flux" in t:
        return "F_2um"

    # Fallback to a compact plain-text token to avoid exploding variants.

    t = t.replace("$", "")
    t = re.sub(r"\\[, ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def _summarize_near_passing_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows_n": int(len(rows)), "constraints_ranked_by_count": [], "fail_combos_top": []}
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        return out

    counts: Dict[str, int] = {}
    combos: Dict[str, int] = {}
    unknown_n = 0
    for row in rows:
        failed = row.get("failed_constraints_norm")
        # 条件分岐: `not isinstance(failed, list)` を満たす経路を評価する。
        if not isinstance(failed, list):
            continue

        failed_clean = [c for c in failed if isinstance(c, str) and c]
        # 条件分岐: `not failed_clean` を満たす経路を評価する。
        if not failed_clean:
            unknown_n += 1
            continue

        for c in failed_clean:
            counts[c] = int(counts.get(c, 0) + 1)

        key = "|".join(sorted(set(failed_clean)))
        combos[key] = int(combos.get(key, 0) + 1)

    ranked = [{"constraint": k, "count": int(v), "fraction": float(v / len(rows))} for k, v in counts.items()]
    ranked.sort(key=lambda x: (-x["count"], x["constraint"]))

    combos_ranked = [{"failed_constraints_key": k, "count": int(v)} for k, v in combos.items()]
    combos_ranked.sort(key=lambda x: (-x["count"], x["failed_constraints_key"]))

    out["constraints_ranked_by_count"] = ranked
    out["fail_combos_top"] = combos_ranked[:20]
    out["unknown_rows_n"] = int(unknown_n)
    return out


def _parse_paper5_near_passing_table(tex_path: Path, *, label: str) -> Dict[str, Any]:
    lines = _read_text(tex_path).splitlines()
    rows = _extract_table_rows(lines)

    parsed_rows: List[Dict[str, Any]] = []
    for lineno, row in rows:
        cols = [c.strip() for c in row.split("&")]
        # 条件分岐: `len(cols) < 6` を満たす経路を評価する。
        if len(cols) < 6:
            continue

        cols = cols[:6]
        code_setup = cols[0]
        mad_sane = cols[1]
        spin = _parse_float(cols[2])
        inc = _parse_float(cols[3])
        rh = _parse_float(cols[4])
        failed_raw = cols[5]

        failed_parts = [p.strip() for p in re.split(r"(?<!\\),", failed_raw) if p.strip()]
        failed_norm = []
        for p in failed_parts:
            norm = _normalize_constraint_token_paper5(p)
            # 条件分岐: `norm is None` を満たす経路を評価する。
            if norm is None:
                continue

            failed_norm.append(norm)

        parsed_rows.append(
            {
                "code_setup": code_setup,
                "mad_sane": mad_sane,
                "spin": spin if spin is not None else cols[2],
                "inc": inc if inc is not None else cols[3],
                "Rh": rh if rh is not None else cols[4],
                "failed_constraints_raw": failed_raw,
                "failed_constraints_norm": failed_norm,
                "source_anchor": {"path": str(tex_path), "line": int(lineno), "label": label},
            }
        )

    return {"rows_n": int(len(parsed_rows)), "rows": parsed_rows, "summary": _summarize_near_passing_rows(parsed_rows)}


def _parse_paper5_pass_fail_table(
    tex_path: Path,
    *,
    label: str,
    params: Sequence[Tuple[str, str]],
    pass_fail_columns: Sequence[str],
    numeric: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Parse Paper V "Pass/Fail Table" (deluxetable) rows.

    params: list of (name, kind) where kind in {"str","float"}.
    numeric: list of (name, kind) where kind in {"float","latex_sci","raw"}.
    """
    lines = _read_text(tex_path).splitlines()
    rows = _extract_table_rows(lines)

    expected_cols = int(len(params) + len(pass_fail_columns) + len(numeric))
    parsed_rows: List[Dict[str, Any]] = []
    for lineno, row in rows:
        cols = [c.strip() for c in row.split("&")]
        # 条件分岐: `len(cols) < expected_cols` を満たす経路を評価する。
        if len(cols) < expected_cols:
            continue

        cols = cols[:expected_cols]
        pos = 0
        params_out: Dict[str, Any] = {}
        for name, kind in params:
            tok = cols[pos]
            pos += 1
            # 条件分岐: `kind == "float"` を満たす経路を評価する。
            if kind == "float":
                v = _parse_float(tok)
                params_out[name] = v if v is not None else tok
            else:
                params_out[name] = tok

        pass_fail_out: Dict[str, Any] = {}
        for name in pass_fail_columns:
            tok = cols[pos]
            pos += 1
            pass_fail_out[name] = tok

        numeric_out: Dict[str, Any] = {}
        for name, kind in numeric:
            tok = cols[pos]
            pos += 1
            # 条件分岐: `kind == "latex_sci"` を満たす経路を評価する。
            if kind == "latex_sci":
                v = _parse_latex_sci(tok)
                numeric_out[name] = v if v is not None else tok
            # 条件分岐: 前段条件が不成立で、`kind == "float"` を追加評価する。
            elif kind == "float":
                v = _parse_float(tok)
                numeric_out[name] = v if v is not None else tok
            else:
                numeric_out[name] = tok

        parsed_rows.append(
            {
                "params": params_out,
                "pass_fail": pass_fail_out,
                "numeric": numeric_out,
                "source_anchor": {"path": str(tex_path), "line": int(lineno), "label": label},
            }
        )

    summary = _summarize_pass_fail_rows(
        parsed_rows,
        pass_fail_columns,
        exclude_from_fail_modes=["non_EHT", "EHT", "All"],
    )
    return {
        "rows_n": int(len(parsed_rows)),
        "rows": parsed_rows,
        "summary": summary,
    }


def _aggregate_pass_fail_tables(
    pass_fail_tables: Dict[str, Any],
    *,
    exclude_constraints: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    exclude = set(exclude_constraints or [])
    tables_summary: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, int]] = {}

    for table_key, tab in pass_fail_tables.items():
        # 条件分岐: `not isinstance(tab, dict)` を満たす経路を評価する。
        if not isinstance(tab, dict):
            continue

        summary = tab.get("summary", {}) if isinstance(tab.get("summary"), dict) else {}
        by_constraint = summary.get("by_constraint", {}) if isinstance(summary.get("by_constraint"), dict) else {}
        rows_n = tab.get("rows_n")
        # 条件分岐: `not isinstance(rows_n, int)` を満たす経路を評価する。
        if not isinstance(rows_n, int):
            rows_n = summary.get("rows_n") if isinstance(summary.get("rows_n"), int) else None

        all_pass_fraction = None
        rec_all = by_constraint.get("All") if isinstance(by_constraint.get("All"), dict) else None
        # 条件分岐: `isinstance(rec_all, dict) and isinstance(rec_all.get("pass_fraction"), (int,...` を満たす経路を評価する。
        if isinstance(rec_all, dict) and isinstance(rec_all.get("pass_fraction"), (int, float)):
            all_pass_fraction = float(rec_all["pass_fraction"])

        tables_summary.append(
            {
                "table_key": str(table_key),
                "rows_n": int(rows_n) if isinstance(rows_n, int) else None,
                "all_pass_fraction": all_pass_fraction,
            }
        )

        for constraint, rec in by_constraint.items():
            # 条件分岐: `constraint in exclude` を満たす経路を評価する。
            if constraint in exclude:
                continue

            # 条件分岐: `not isinstance(rec, dict)` を満たす経路を評価する。

            if not isinstance(rec, dict):
                continue

            fail_n = int(rec.get("fail_n", 0))
            pass_n = int(rec.get("pass_n", 0))
            unknown_n = int(rec.get("unknown_n", 0))
            denom = fail_n + pass_n + unknown_n
            # 条件分岐: `denom <= 0` を満たす経路を評価する。
            if denom <= 0:
                continue

            tot = totals.setdefault(constraint, {"fail_n": 0, "pass_n": 0, "unknown_n": 0, "denom": 0, "tables_n": 0})
            tot["fail_n"] += fail_n
            tot["pass_n"] += pass_n
            tot["unknown_n"] += unknown_n
            tot["denom"] += denom
            tot["tables_n"] += 1

    ranked: List[Dict[str, Any]] = []
    for constraint, tot in totals.items():
        denom = int(tot.get("denom", 0))
        # 条件分岐: `denom <= 0` を満たす経路を評価する。
        if denom <= 0:
            continue

        fail_n = int(tot.get("fail_n", 0))
        pass_n = int(tot.get("pass_n", 0))
        unknown_n = int(tot.get("unknown_n", 0))
        ranked.append(
            {
                "constraint": str(constraint),
                "fail_n": fail_n,
                "pass_n": pass_n,
                "unknown_n": unknown_n,
                "n_total": denom,
                "tables_n": int(tot.get("tables_n", 0)),
                "fail_fraction": float(fail_n / denom),
                "pass_fraction": float(pass_n / denom),
            }
        )

    ranked.sort(key=lambda x: (-x["fail_fraction"], x["constraint"]))
    tables_summary.sort(key=lambda x: (x.get("table_key") or ""))

    return {
        "tables_n": int(len(tables_summary)),
        "tables": tables_summary,
        "excluded_constraints": sorted(exclude),
        "constraints_ranked_by_global_fail_fraction": ranked,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_thermal = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "thermal_model_pass_fraction_table.tex"
    default_nonthermal = (
        root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "nonthermal_model_pass_fraction_table.tex"
    )
    default_critical_beta = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Critical_Beta_Table.tex"
    default_bhac_varkappa = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "BHAC_varkappa_Table.tex"
    default_frankfurt_thermal = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Frankfurt_Table.tex"
    default_frankfurt_kappa5 = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Frankfurt_k5_Table.tex"
    default_frankfurt_fixed_kappa_var_eff = (
        root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Frankfurt_All_Epsilon_Table.tex"
    )
    default_illinois_thermal = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Illinois_Table.tex"
    default_koral_thermal = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Koral_Table.tex"
    default_hamr_thermal = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Hamr_Thermal_Table.tex"
    default_hamr_variable_kappa = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Hamr_varkappa_Table.tex"
    default_hamr_tilted = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Hamr_All_Tilts.tex"
    default_hamr_nonthermal = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Hamr_Nonthermal_Table.tex"
    default_ressler_wind_fed = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "Ressler_Table.tex"
    default_fail_one_thermal = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "fail_one_thermal.tex"
    default_fail_one_nonthermal = (
        root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "fail_one_nonthermal.tex"
    )
    default_fail_none = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "Tables" / "fail_none.tex"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(description="Parse Sgr A* Paper V pass fraction tables (tab:passfraction_thermal/tab:passfraction).")
    ap.add_argument("--thermal-tex", type=str, default=str(default_thermal))
    ap.add_argument("--exploratory-tex", type=str, default=str(default_nonthermal))
    ap.add_argument("--critical-beta-tex", type=str, default=str(default_critical_beta))
    ap.add_argument("--bhac-varkappa-tex", type=str, default=str(default_bhac_varkappa))
    ap.add_argument("--frankfurt-thermal-tex", type=str, default=str(default_frankfurt_thermal))
    ap.add_argument("--frankfurt-kappa5-tex", type=str, default=str(default_frankfurt_kappa5))
    ap.add_argument("--frankfurt-fixed-kappa-var-eff-tex", type=str, default=str(default_frankfurt_fixed_kappa_var_eff))
    ap.add_argument("--illinois-thermal-tex", type=str, default=str(default_illinois_thermal))
    ap.add_argument("--koral-thermal-tex", type=str, default=str(default_koral_thermal))
    ap.add_argument("--hamr-thermal-tex", type=str, default=str(default_hamr_thermal))
    ap.add_argument("--hamr-variable-kappa-tex", type=str, default=str(default_hamr_variable_kappa))
    ap.add_argument("--hamr-tilted-tex", type=str, default=str(default_hamr_tilted))
    ap.add_argument("--hamr-nonthermal-tex", type=str, default=str(default_hamr_nonthermal))
    ap.add_argument("--ressler-wind-fed-tex", type=str, default=str(default_ressler_wind_fed))
    ap.add_argument("--fail-one-thermal-tex", type=str, default=str(default_fail_one_thermal))
    ap.add_argument("--fail-one-nonthermal-tex", type=str, default=str(default_fail_one_nonthermal))
    ap.add_argument("--fail-none-tex", type=str, default=str(default_fail_none))
    ap.add_argument("--outdir", type=str, default=str(default_outdir))
    args = ap.parse_args(list(argv) if argv is not None else None)

    thermal_path = Path(args.thermal_tex)
    expl_path = Path(args.exploratory_tex)
    critical_beta_path = Path(args.critical_beta_tex)
    bhac_varkappa_path = Path(args.bhac_varkappa_tex)
    frankfurt_thermal_path = Path(args.frankfurt_thermal_tex)
    frankfurt_kappa5_path = Path(args.frankfurt_kappa5_tex)
    frankfurt_fixed_kappa_var_eff_path = Path(args.frankfurt_fixed_kappa_var_eff_tex)
    illinois_thermal_path = Path(args.illinois_thermal_tex)
    koral_thermal_path = Path(args.koral_thermal_tex)
    hamr_thermal_path = Path(args.hamr_thermal_tex)
    hamr_variable_kappa_path = Path(args.hamr_variable_kappa_tex)
    hamr_tilted_path = Path(args.hamr_tilted_tex)
    hamr_nonthermal_path = Path(args.hamr_nonthermal_tex)
    ressler_wind_fed_path = Path(args.ressler_wind_fed_tex)
    fail_one_thermal_path = Path(args.fail_one_thermal_tex)
    fail_one_nonthermal_path = Path(args.fail_one_nonthermal_tex)
    fail_none_path = Path(args.fail_none_tex)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "eht_sgra_paper5_pass_fraction_tables_metrics.json"
    out_png = outdir / "eht_sgra_paper5_pass_fail_tables_fail_fractions.png"
    out_png_global = outdir / "eht_sgra_paper5_pass_fail_global_constraint_fail_fractions.png"
    out_png_near = outdir / "eht_sgra_paper5_near_passing_failed_constraints.png"

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "thermal_tex": str(thermal_path),
            "exploratory_tex": str(expl_path),
            "critical_beta_tex": str(critical_beta_path),
            "bhac_varkappa_tex": str(bhac_varkappa_path),
            "frankfurt_thermal_tex": str(frankfurt_thermal_path),
            "frankfurt_kappa5_tex": str(frankfurt_kappa5_path),
            "frankfurt_fixed_kappa_var_eff_tex": str(frankfurt_fixed_kappa_var_eff_path),
            "illinois_thermal_tex": str(illinois_thermal_path),
            "koral_thermal_tex": str(koral_thermal_path),
            "hamr_thermal_tex": str(hamr_thermal_path),
            "hamr_variable_kappa_tex": str(hamr_variable_kappa_path),
            "hamr_tilted_tex": str(hamr_tilted_path),
            "hamr_nonthermal_tex": str(hamr_nonthermal_path),
            "ressler_wind_fed_tex": str(ressler_wind_fed_path),
            "fail_one_thermal_tex": str(fail_one_thermal_path),
            "fail_one_nonthermal_tex": str(fail_one_nonthermal_path),
            "fail_none_tex": str(fail_none_path),
        },
        "ok": True,
        "extracted": {},
        "derived": {},
        "outputs": {"json": str(out_json), "png": str(out_png), "png_global": str(out_png_global), "png_near": str(out_png_near)},
    }

    # 条件分岐: `not thermal_path.exists() or not expl_path.exists()` を満たす経路を評価する。
    if not thermal_path.exists() or not expl_path.exists():
        payload["ok"] = False
        payload["reason"] = "missing_input_tex"
        payload["missing"] = {
            "thermal_tex": (not thermal_path.exists()),
            "exploratory_tex": (not expl_path.exists()),
        }
        _write_json(out_json, payload)
        print(f"[warn] missing input; wrote: {out_json}")
        return 0

    # Thermal table: constraint | kharma | bhac | hamr

    thermal_lines = _read_text(thermal_path).splitlines()
    thermal_rows = _extract_table_rows(thermal_lines)
    thermal_by_constraint: Dict[str, Dict[str, Any]] = {}
    for lineno, row in thermal_rows:
        cols = [c.strip() for c in row.split("&")]
        # 条件分岐: `len(cols) < 4` を満たす経路を評価する。
        if len(cols) < 4:
            continue

        constraint = cols[0]
        kharma = _parse_float(cols[1])
        bhac = _parse_float(cols[2])
        hamr = _parse_float(cols[3])
        values_all = [v for v in (kharma, bhac, hamr) if v is not None]
        thermal_by_constraint[constraint] = {
            "kharma": kharma,
            "bhac": bhac,
            "hamr": hamr,
            "values_all": values_all,
            "source_anchor": {"path": str(thermal_path), "line": int(lineno), "label": "tab:passfraction_thermal"},
        }

    # Exploratory table (tab:passfraction): constraint/model | BHAC (4 cols) | HAMR (3 cols)

    expl_lines = _read_text(expl_path).splitlines()
    expl_rows = _extract_table_rows(expl_lines)
    expl_by_constraint: Dict[str, Dict[str, Any]] = {}
    for lineno, row in expl_rows:
        cols = [c.strip() for c in row.split("&")]
        # 条件分岐: `len(cols) < 8` を満たす経路を評価する。
        if len(cols) < 8:
            continue

        constraint = cols[0]
        # 条件分岐: `not constraint` を満たす経路を評価する。
        if not constraint:
            continue

        bhac_thermal = _parse_cell_values(cols[1])
        bhac_kappa_var = _parse_cell_values(cols[2])
        bhac_kappa35_eps = _parse_cell_values(cols[3])
        bhac_kappa5 = _parse_cell_values(cols[4])
        hamr_thermal = _parse_cell_values(cols[5])
        hamr_kappa_var = _parse_cell_values(cols[6])
        hamr_p4 = _parse_cell_values(cols[7])
        values_all = bhac_thermal + bhac_kappa_var + bhac_kappa35_eps + bhac_kappa5 + hamr_thermal + hamr_kappa_var + hamr_p4
        expl_by_constraint[constraint] = {
            "bhac": {
                "thermal": bhac_thermal,
                "kappa_sigma_beta": bhac_kappa_var,
                "kappa35_eps": bhac_kappa35_eps,
                "kappa5": bhac_kappa5,
            },
            "hamr": {"thermal": hamr_thermal, "kappa_sigma_beta": hamr_kappa_var, "p4": hamr_p4},
            "values_all": values_all,
            "source_anchor": {"path": str(expl_path), "line": int(lineno), "label": "tab:passfraction"},
        }

    # 条件分岐: `not thermal_by_constraint or not expl_by_constraint` を満たす経路を評価する。

    if not thermal_by_constraint or not expl_by_constraint:
        payload["ok"] = False
        payload["reason"] = "no_rows_parsed"
        payload["extracted"] = {
            "thermal_rows_n": int(len(thermal_rows)),
            "exploratory_rows_n": int(len(expl_rows)),
        }
        _write_json(out_json, payload)
        print(f"[warn] no rows parsed; wrote: {out_json}")
        return 0

    payload["extracted"] = {
        "thermal": {"constraints_n": int(len(thermal_by_constraint)), "by_constraint": thermal_by_constraint},
        "exploratory": {"constraints_n": int(len(expl_by_constraint)), "by_constraint": expl_by_constraint},
    }

    # Optional: detailed Pass/Fail tables (tab:betacritPF / tab:VKbhacPF)
    pass_fail_tables: Dict[str, Any] = {}
    warnings: List[str] = []
    # 条件分岐: `critical_beta_path.exists()` を満たす経路を評価する。
    if critical_beta_path.exists():
        pass_fail_tables["critical_beta_models"] = _parse_paper5_pass_fail_table(
            critical_beta_path,
            label="tab:betacritPF",
            params=[("M_or_S", "str"), ("spin", "float"), ("i_deg", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "non_EHT",
                "lambda_230",
                "Nulls",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[],
        )
    else:
        warnings.append(f"missing_critical_beta_tex: {critical_beta_path}")

    # 条件分岐: `bhac_varkappa_path.exists()` を満たす経路を評価する。

    if bhac_varkappa_path.exists():
        pass_fail_tables["frankfurt_variable_kappa_models_bhac"] = _parse_paper5_pass_fail_table(
            bhac_varkappa_path,
            label="tab:VKbhacPF",
            params=[("M_or_S", "str"), ("spin", "float"), ("i_deg", "float"), ("Rh", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "non_EHT",
                "lambda_230",
                "Nulls",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[
                ("mdot_over_mdotedd", "latex_sci"),
                ("P_out_cgs", "latex_sci"),
                ("P_out_over_mdot_c2", "latex_sci"),
            ],
        )
    else:
        warnings.append(f"missing_bhac_varkappa_tex: {bhac_varkappa_path}")

    # 条件分岐: `frankfurt_thermal_path.exists()` を満たす経路を評価する。

    if frankfurt_thermal_path.exists():
        pass_fail_tables["frankfurt_thermal_models"] = _parse_paper5_pass_fail_table(
            frankfurt_thermal_path,
            label="tab:frankfurtPF",
            params=[("M_or_S", "str"), ("spin", "float"), ("i_deg", "float"), ("Rh", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "L_X",
                "non_EHT",
                "lambda_230",
                "Nulls",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[
                ("mdot_over_mdotedd", "latex_sci"),
                ("P_out_cgs", "latex_sci"),
                ("P_out_over_mdot_c2", "latex_sci"),
            ],
        )
    else:
        warnings.append(f"missing_frankfurt_thermal_tex: {frankfurt_thermal_path}")

    # 条件分岐: `frankfurt_kappa5_path.exists()` を満たす経路を評価する。

    if frankfurt_kappa5_path.exists():
        pass_fail_tables["frankfurt_kappa5_models"] = _parse_paper5_pass_fail_table(
            frankfurt_kappa5_path,
            label="tab:frankfurtk5PF",
            params=[("M_or_S", "str"), ("spin", "float"), ("i_deg", "float"), ("Rh", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "non_EHT",
                "lambda_230",
                "Nulls",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[
                ("mdot_over_mdotedd", "latex_sci"),
                ("P_out_cgs", "latex_sci"),
                ("P_out_over_mdot_c2", "latex_sci"),
            ],
        )
    else:
        warnings.append(f"missing_frankfurt_kappa5_tex: {frankfurt_kappa5_path}")

    # 条件分岐: `frankfurt_fixed_kappa_var_eff_path.exists()` を満たす経路を評価する。

    if frankfurt_fixed_kappa_var_eff_path.exists():
        pass_fail_tables["frankfurt_fixed_kappa_variable_efficiency_models"] = _parse_paper5_pass_fail_table(
            frankfurt_fixed_kappa_var_eff_path,
            label="tab:frankfurtfkPF",
            params=[("epsilon", "float"), ("M_or_S", "str"), ("spin", "float"), ("i_deg", "float"), ("Rh", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "non_EHT",
                "lambda_230",
                "Nulls",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[
                ("mdot_over_mdotedd", "latex_sci"),
                ("P_out_cgs", "latex_sci"),
                ("P_out_over_mdot_c2", "latex_sci"),
            ],
        )
    else:
        warnings.append(f"missing_frankfurt_fixed_kappa_var_eff_tex: {frankfurt_fixed_kappa_var_eff_path}")

    # 条件分岐: `illinois_thermal_path.exists()` を満たす経路を評価する。

    if illinois_thermal_path.exists():
        pass_fail_tables["illinois_thermal_models"] = _parse_paper5_pass_fail_table(
            illinois_thermal_path,
            label="tab:illinoisPF",
            params=[("M_or_S", "str"), ("spin", "float"), ("i_deg", "float"), ("Rh", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "L_X",
                "non_EHT",
                "lambda_230",
                "Nulls",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[
                ("mdot_over_mdotedd", "latex_sci"),
                ("Lbol_over_mdot_c2", "latex_sci"),
                ("P_out_cgs", "latex_sci"),
                ("P_out_over_mdot_c2", "latex_sci"),
            ],
        )
    else:
        warnings.append(f"missing_illinois_thermal_tex: {illinois_thermal_path}")

    # 条件分岐: `koral_thermal_path.exists()` を満たす経路を評価する。

    if koral_thermal_path.exists():
        pass_fail_tables["koral_thermal_models"] = _parse_paper5_pass_fail_table(
            koral_thermal_path,
            label="tab:koralPF",
            params=[("M_or_S", "str"), ("spin", "float"), ("i_deg", "float"), ("Rh", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "non_EHT",
                "lambda_230",
                "Nulls",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[
                ("mdot_over_mdotedd", "latex_sci"),
                ("P_out_cgs", "latex_sci"),
                ("P_out_over_mdot_c2", "latex_sci"),
            ],
        )
    else:
        warnings.append(f"missing_koral_thermal_tex: {koral_thermal_path}")

    # 条件分岐: `hamr_thermal_path.exists()` を満たす経路を評価する。

    if hamr_thermal_path.exists():
        pass_fail_tables["hamr_thermal_models"] = _parse_paper5_pass_fail_table(
            hamr_thermal_path,
            label="tab:ThamrPF",
            params=[("M_or_S", "str"), ("spin", "float"), ("i_deg", "float"), ("Rh", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "L_X",
                "non_EHT",
                "lambda_230",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[
                ("mdot_over_mdotedd", "latex_sci"),
                ("Lbol_over_mdot_c2", "latex_sci"),
                ("P_out_cgs", "latex_sci"),
                ("P_out_over_mdot_c2", "latex_sci"),
            ],
        )
    else:
        warnings.append(f"missing_hamr_thermal_tex: {hamr_thermal_path}")

    # 条件分岐: `hamr_variable_kappa_path.exists()` を満たす経路を評価する。

    if hamr_variable_kappa_path.exists():
        pass_fail_tables["hamr_variable_kappa_models"] = _parse_paper5_pass_fail_table(
            hamr_variable_kappa_path,
            label="tab:tab:vkhamrPF",
            params=[("M_or_S", "str"), ("spin", "float"), ("i_deg", "float"), ("Rh", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "L_X",
                "non_EHT",
                "lambda_230",
                "Nulls",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[
                ("mdot_over_mdotedd", "latex_sci"),
                ("Lbol_over_mdot_c2", "latex_sci"),
                ("P_out_cgs", "latex_sci"),
                ("P_out_over_mdot_c2", "latex_sci"),
            ],
        )
    else:
        warnings.append(f"missing_hamr_variable_kappa_tex: {hamr_variable_kappa_path}")

    # 条件分岐: `hamr_tilted_path.exists()` を満たす経路を評価する。

    if hamr_tilted_path.exists():
        pass_fail_tables["hamr_tilted_models"] = _parse_paper5_pass_fail_table(
            hamr_tilted_path,
            label="tab:TiltedhamrPF",
            params=[("tilt_deg", "float"), ("spin", "float"), ("i_deg", "float"), ("Rh", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "non_EHT",
                "lambda_230",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[],
        )
    else:
        warnings.append(f"missing_hamr_tilted_tex: {hamr_tilted_path}")

    # 条件分岐: `hamr_nonthermal_path.exists()` を満たす経路を評価する。

    if hamr_nonthermal_path.exists():
        pass_fail_tables["hamr_nonthermal_powerlaw_models"] = _parse_paper5_pass_fail_table(
            hamr_nonthermal_path,
            label="tab:hamr_nth",
            params=[("M_or_S", "str"), ("spin", "float"), ("i_deg", "float"), ("Rh", "float")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "non_EHT",
                "lambda_230",
                "Nulls",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[],
        )
    else:
        warnings.append(f"missing_hamr_nonthermal_tex: {hamr_nonthermal_path}")

    # 条件分岐: `ressler_wind_fed_path.exists()` を満たす経路を評価する。

    if ressler_wind_fed_path.exists():
        pass_fail_tables["wind_fed_models"] = _parse_paper5_pass_fail_table(
            ressler_wind_fed_path,
            label="tab:resslerPF",
            params=[("beta", "str")],
            pass_fail_columns=[
                "F86",
                "lambda_maj86",
                "F_2um",
                "non_EHT",
                "lambda_230",
                "Nulls",
                "Ring_D",
                "Ring_W",
                "Ring_A",
                "var_4Glambda",
                "EHT",
                "All",
                "M3",
            ],
            numeric=[],
        )
    else:
        warnings.append(f"missing_ressler_wind_fed_tex: {ressler_wind_fed_path}")

    near_tables: Dict[str, Any] = {}
    # 条件分岐: `fail_one_thermal_path.exists()` を満たす経路を評価する。
    if fail_one_thermal_path.exists():
        near_tables["fail_one_thermal"] = _parse_paper5_near_passing_table(fail_one_thermal_path, label="tab:fail_one_thermal")
    else:
        warnings.append(f"missing_fail_one_thermal_tex: {fail_one_thermal_path}")

    # 条件分岐: `fail_one_nonthermal_path.exists()` を満たす経路を評価する。

    if fail_one_nonthermal_path.exists():
        near_tables["fail_one_nonthermal"] = _parse_paper5_near_passing_table(
            fail_one_nonthermal_path, label="tab:fail_one_nonthermal"
        )
    else:
        warnings.append(f"missing_fail_one_nonthermal_tex: {fail_one_nonthermal_path}")

    # 条件分岐: `fail_none_path.exists()` を満たす経路を評価する。

    if fail_none_path.exists():
        near_tables["fail_none"] = _parse_paper5_near_passing_table(fail_none_path, label="tab:fail_none")
    else:
        warnings.append(f"missing_fail_none_tex: {fail_none_path}")

    # 条件分岐: `pass_fail_tables` を満たす経路を評価する。

    if pass_fail_tables:
        payload["extracted"]["pass_fail_tables"] = pass_fail_tables

    # 条件分岐: `near_tables` を満たす経路を評価する。

    if near_tables:
        payload["extracted"]["near_passing"] = near_tables

    # 条件分岐: `warnings` を満たす経路を評価する。

    if warnings:
        payload["warnings"] = warnings

    payload["derived"] = {
        "thermal": {"constraints_ranked_by_mean_pass_fraction": _rank_constraints(thermal_by_constraint)},
        "exploratory": {"constraints_ranked_by_mean_pass_fraction": _rank_constraints(expl_by_constraint)},
        "notes": [
            "Pass fractions are model-selection diagnostics from Paper V tables; they are not σ(κ) and should not be mixed into κσ(method-scatter).",
            "Use these rankings to identify which constraints (e.g., M-ring width, variability) dominate model exclusion and thus where emission-model systematics likely matter most.",
        ],
    }

    # 条件分岐: `pass_fail_tables` を満たす経路を評価する。
    if pass_fail_tables:
        payload["derived"]["pass_fail_tables_global"] = _aggregate_pass_fail_tables(
            pass_fail_tables,
            exclude_constraints=["non_EHT", "EHT", "All"],
        )

    near_passing = payload.get("extracted", {}).get("near_passing")
    # 条件分岐: `isinstance(near_passing, dict) and near_passing` を満たす経路を評価する。
    if isinstance(near_passing, dict) and near_passing:
        combined_rows: List[Dict[str, Any]] = []
        for key, tab in near_passing.items():
            # 条件分岐: `not isinstance(tab, dict)` を満たす経路を評価する。
            if not isinstance(tab, dict):
                continue

            rows = tab.get("rows")
            # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
            if not isinstance(rows, list):
                continue

            for r in rows:
                # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
                if not isinstance(r, dict):
                    continue

                r2 = dict(r)
                r2["table_key"] = key
                combined_rows.append(r2)

        payload["derived"]["near_passing"] = {"combined_summary": _summarize_near_passing_rows(combined_rows)}

    try:
        import matplotlib.pyplot as plt  # noqa: F401

        _set_japanese_font()

        tables = payload.get("extracted", {}).get("pass_fail_tables", {})
        # 条件分岐: `isinstance(tables, dict) and tables` を満たす経路を評価する。
        if isinstance(tables, dict) and tables:
            plot_defs = [
                ("critical_beta_models", "Paper V: Critical Beta models (tab:betacritPF)"),
                ("frankfurt_variable_kappa_models_bhac", "Paper V: Frankfurt variable κ models (tab:VKbhacPF; BHAC)"),
            ]
            keys = [k for k, _ in plot_defs if k in tables]
            # 条件分岐: `keys` を満たす経路を評価する。
            if keys:
                nrows = int(len(keys))
                fig = plt.figure(figsize=(10.5, 4.3 * nrows))
                for idx, (key, label) in enumerate(plot_defs, start=1):
                    # 条件分岐: `key not in tables` を満たす経路を評価する。
                    if key not in tables:
                        continue

                    tab = tables[key]
                    summary = tab.get("summary", {}) if isinstance(tab, dict) else {}
                    ranked = summary.get("constraints_ranked_by_fail_fraction_excluding_aggregates")
                    # 条件分岐: `not isinstance(ranked, list) or not ranked` を満たす経路を評価する。
                    if not isinstance(ranked, list) or not ranked:
                        ranked = summary.get("constraints_ranked_by_fail_fraction") if isinstance(summary.get("constraints_ranked_by_fail_fraction"), list) else []

                    ranked = [r for r in ranked if isinstance(r, dict) and isinstance(r.get("constraint"), str)]
                    constraints = [r["constraint"] for r in ranked]
                    fail_fracs = [float(r.get("fail_fraction", 0.0)) for r in ranked]

                    ax = fig.add_subplot(nrows, 1, idx)
                    y = list(range(len(constraints)))
                    ax.barh(y, fail_fracs, color="#d62728", alpha=0.85)
                    ax.set_yticks(y)
                    ax.set_yticklabels(constraints)
                    ax.invert_yaxis()
                    ax.set_xlim(0.0, 1.0)
                    ax.set_xlabel("Fail fraction")

                    n_total = tab.get("rows_n")
                    all_rec = summary.get("by_constraint", {}).get("All") if isinstance(summary.get("by_constraint"), dict) else None
                    all_pass = None
                    # 条件分岐: `isinstance(all_rec, dict)` を満たす経路を評価する。
                    if isinstance(all_rec, dict):
                        all_pass = all_rec.get("pass_fraction")

                    subtitle = f"N={n_total}" if isinstance(n_total, int) else "N=?"
                    # 条件分岐: `isinstance(all_pass, (int, float))` を満たす経路を評価する。
                    if isinstance(all_pass, (int, float)):
                        subtitle += f", All-pass fraction={float(all_pass):.3f}"

                    ax.set_title(f"{label} ({subtitle})")
                    ax.grid(True, axis="x", alpha=0.25)

                    for yi, val in enumerate(fail_fracs):
                        # 条件分岐: `not (0.0 <= val <= 1.0)` を満たす経路を評価する。
                        if not (0.0 <= val <= 1.0):
                            continue

                        ax.text(min(0.98, val + 0.02), yi, f"{val:.2f}", va="center", ha="left", fontsize=9)

                fig.tight_layout()
                out_png.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_png, dpi=200)
                plt.close(fig)

        global_ranked = payload.get("derived", {}).get("pass_fail_tables_global", {}).get("constraints_ranked_by_global_fail_fraction")
        # 条件分岐: `isinstance(global_ranked, list) and global_ranked` を満たす経路を評価する。
        if isinstance(global_ranked, list) and global_ranked:
            top = [r for r in global_ranked if isinstance(r, dict) and isinstance(r.get("constraint"), str)]
            top = top[:12]
            # 条件分岐: `top` を満たす経路を評価する。
            if top:
                constraints = [r["constraint"] for r in top]
                fail_fracs = [float(r.get("fail_fraction", 0.0)) for r in top]
                totals = [int(r.get("n_total", 0)) for r in top]
                tables_n = [int(r.get("tables_n", 0)) for r in top]

                fig = plt.figure(figsize=(10.5, 0.6 * len(constraints) + 1.6))
                ax = fig.add_subplot(1, 1, 1)
                y = list(range(len(constraints)))
                ax.barh(y, fail_fracs, color="#d62728", alpha=0.85)
                ax.set_yticks(y)
                ax.set_yticklabels(constraints)
                ax.invert_yaxis()
                ax.set_xlim(0.0, 1.0)
                ax.set_xlabel("Fail fraction (aggregated over Paper V pass/fail tables)")
                ax.set_title("Paper V pass/fail: dominant constraints (global fail fraction)")
                ax.grid(True, axis="x", alpha=0.25)
                for yi, (val, n_tot, n_tab) in enumerate(zip(fail_fracs, totals, tables_n)):
                    # 条件分岐: `not (0.0 <= val <= 1.0)` を満たす経路を評価する。
                    if not (0.0 <= val <= 1.0):
                        continue

                    ax.text(
                        min(0.98, val + 0.02),
                        yi,
                        f"{val:.2f} (N={n_tot}, tables={n_tab})",
                        va="center",
                        ha="left",
                        fontsize=9,
                    )

                fig.tight_layout()
                out_png_global.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_png_global, dpi=200)
                plt.close(fig)

        near_ranked = (
            payload.get("derived", {})
            .get("near_passing", {})
            .get("combined_summary", {})
            .get("constraints_ranked_by_count")
        )
        # 条件分岐: `isinstance(near_ranked, list) and near_ranked` を満たす経路を評価する。
        if isinstance(near_ranked, list) and near_ranked:
            items = [r for r in near_ranked if isinstance(r, dict) and isinstance(r.get("constraint"), str)]
            constraints = [r["constraint"] for r in items]
            counts = [int(r.get("count", 0)) for r in items]
            total = int(sum(counts)) if counts else 0
            # 条件分岐: `constraints and total > 0` を満たす経路を評価する。
            if constraints and total > 0:
                fig = plt.figure(figsize=(10.5, 0.55 * len(constraints) + 1.6))
                ax = fig.add_subplot(1, 1, 1)
                y = list(range(len(constraints)))
                ax.barh(y, counts, color="#1f77b4", alpha=0.85)
                ax.set_yticks(y)
                ax.set_yticklabels(constraints)
                ax.invert_yaxis()
                ax.set_xlabel("Count (near-passing models)")
                ax.set_title("Paper V near-passing: failed constraint counts (fail_one + fail_none)")
                ax.grid(True, axis="x", alpha=0.25)
                for yi, val in enumerate(counts):
                    # 条件分岐: `val <= 0` を満たす経路を評価する。
                    if val <= 0:
                        continue

                    ax.text(val + 0.05, yi, f"{val} ({val/total:.2f})", va="center", ha="left", fontsize=9)

                fig.tight_layout()
                out_png_near.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_png_near, dpi=200)
                plt.close(fig)
    except Exception as e:
        payload["plot_error"] = str(e)

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper5_pass_fraction_tables_metrics",
                "outputs": [
                    str(out_json.relative_to(root)).replace("\\", "/"),
                    str(out_png.relative_to(root)).replace("\\", "/"),
                    str(out_png_global.relative_to(root)).replace("\\", "/"),
                    str(out_png_near.relative_to(root)).replace("\\", "/"),
                ],
                "metrics": {
                    "ok": bool(payload.get("ok")),
                    "thermal_constraints_n": int(payload["extracted"]["thermal"]["constraints_n"]),
                    "exploratory_constraints_n": int(payload["extracted"]["exploratory"]["constraints_n"]),
                    "critical_beta_rows_n": int(
                        payload.get("extracted", {}).get("pass_fail_tables", {}).get("critical_beta_models", {}).get("rows_n", 0)
                    ),
                    "bhac_varkappa_rows_n": int(
                        payload.get("extracted", {})
                        .get("pass_fail_tables", {})
                        .get("frankfurt_variable_kappa_models_bhac", {})
                        .get("rows_n", 0)
                    ),
                    "pass_fail_tables_n": int(len(payload.get("extracted", {}).get("pass_fail_tables", {}) or {})),
                    "near_passing_tables_n": int(len(payload.get("extracted", {}).get("near_passing", {}) or {})),
                    "near_passing_rows_n": int(
                        payload.get("derived", {}).get("near_passing", {}).get("combined_summary", {}).get("rows_n", 0)
                    ),
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
