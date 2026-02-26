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


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_clean_latex_token` の入出力契約と処理意図を定義する。

def _clean_latex_token(s: str) -> str:
    x = s.strip()
    x = re.sub(r"\\texttt\{([^}]*)\}", r"\1", x)
    x = x.replace("\\smili", "smili")
    x = x.replace("\\difmap", "difmap")
    x = x.replace("\\ehtim", "ehtim")
    x = re.sub(r"\\[a-zA-Z]+\s*", "", x)
    x = x.replace("{", "").replace("}", "")
    x = re.sub(r"\s+", " ", x).strip()
    return x


_RE_VAL = re.compile(
    r"\$?\s*(?P<mid>[-+]?\d+(?:\.\d+)?)\s*_\{\s*(?P<lo>[-+]?\d+(?:\.\d+)?)\s*\}\s*\^\{\s*(?P<hi>[-+]?\d+(?:\.\d+)?)\s*\}\s*\$?"
)


# 関数: `_parse_pm_val` の入出力契約と処理意図を定義する。
def _parse_pm_val(token: str) -> Optional[Dict[str, float]]:
    m = _RE_VAL.search(token)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return None

    mid = float(m.group("mid"))
    lo = float(m.group("lo"))
    hi = float(m.group("hi"))
    # 条件分岐: `not (math.isfinite(mid) and math.isfinite(lo) and math.isfinite(hi))` を満たす経路を評価する。
    if not (math.isfinite(mid) and math.isfinite(lo) and math.isfinite(hi)):
        return None

    return {
        "mid": mid,
        "err_minus": abs(lo),
        "err_plus": abs(hi),
        "lo": mid + lo,
        "hi": mid + hi,
    }


# クラス: `ParsedRow` の責務と境界条件を定義する。

@dataclass(frozen=True)
class ParsedRow:
    method: str
    prior: Optional[str]
    grmhd: Dict[str, float]
    analytic_kerr: Dict[str, float]
    analytic_non_kerr: Dict[str, float]
    source_anchor: Dict[str, Any]


# 関数: `_find_table_block` の入出力契約と処理意図を定義する。

def _find_table_block(lines: List[str], caption_substring: str) -> Optional[Tuple[int, int]]:
    for i, line in enumerate(lines):
        # 条件分岐: `caption_substring in line` を満たす経路を評価する。
        if caption_substring in line:
            start = i
            while start >= 0 and "\\begin{table" not in lines[start]:
                start -= 1

            # 条件分岐: `start < 0` を満たす経路を評価する。

            if start < 0:
                start = i

            end = i
            while end < len(lines) and "\\end{table" not in lines[end]:
                end += 1

            # 条件分岐: `end >= len(lines)` を満たす経路を評価する。

            if end >= len(lines):
                end = len(lines) - 1

            return (start, end)

    return None


# 関数: `_extract_tabular_lines` の入出力契約と処理意図を定義する。

def _extract_tabular_lines(block_lines: List[str]) -> Optional[Tuple[int, int]]:
    a = None
    b = None
    for i, line in enumerate(block_lines):
        # 条件分岐: `"\\begin{tabular" in line` を満たす経路を評価する。
        if "\\begin{tabular" in line:
            a = i
            break

    # 条件分岐: `a is None` を満たす経路を評価する。

    if a is None:
        return None

    for j in range(a, len(block_lines)):
        # 条件分岐: `"\\end{tabular" in block_lines[j]` を満たす経路を評価する。
        if "\\end{tabular" in block_lines[j]:
            b = j
            break

    # 条件分岐: `b is None` を満たす経路を評価する。

    if b is None:
        return None

    return (a, b)


# 関数: `_iter_rows_with_anchors` の入出力契約と処理意図を定義する。

def _iter_rows_with_anchors(
    lines: List[str], *, start_line_no: int, allow_blank_method: bool
) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    buf = ""
    buf_start = None
    for off, raw in enumerate(lines):
        lineno = start_line_no + off
        s = raw.strip()
        # 条件分岐: `not s` を満たす経路を評価する。
        if not s:
            continue

        if any(
            k in s
            for k in (
                "\\textrm",
                "\\colrule",
                "\\hline",
                "\\begin{",
                "\\end{",
            )
        ):
            continue

        # 条件分岐: `buf_start is None` を満たす経路を評価する。

        if buf_start is None:
            buf_start = lineno

        # 条件分岐: `buf` を満たす経路を評価する。

        if buf:
            buf += " " + s
        else:
            buf = s

        # 条件分岐: `"\\\\" in s` を満たす経路を評価する。

        if "\\\\" in s:
            rows.append((buf_start, buf))
            buf = ""
            buf_start = None

    # 条件分岐: `buf and buf_start is not None and allow_blank_method` を満たす経路を評価する。

    if buf and buf_start is not None and allow_blank_method:
        rows.append((buf_start, buf))

    return rows


# 関数: `_parse_dsh_table` の入出力契約と処理意図を定義する。

def _parse_dsh_table(
    *,
    lines: List[str],
    tex_path: Path,
    block_start_line_no: int,
    tabular_lines: List[str],
    tabular_start_line_no: int,
) -> List[ParsedRow]:
    rows: List[ParsedRow] = []
    raw_rows = _iter_rows_with_anchors(tabular_lines, start_line_no=tabular_start_line_no, allow_blank_method=False)
    for lineno, row in raw_rows:
        before, _, after = row.partition("\\\\")
        cols = [c.strip() for c in before.split("&")]
        # 条件分岐: `len(cols) < 4` を満たす経路を評価する。
        if len(cols) < 4:
            continue

        method = _clean_latex_token(cols[0])
        # 条件分岐: `not method` を満たす経路を評価する。
        if not method:
            continue

        grmhd = _parse_pm_val(cols[1] or "")
        ak = _parse_pm_val(cols[2] or "")
        ank = _parse_pm_val(cols[3] or "")
        # 条件分岐: `not (grmhd and ak and ank)` を満たす経路を評価する。
        if not (grmhd and ak and ank):
            continue

        rows.append(
            ParsedRow(
                method=method,
                prior=None,
                grmhd=grmhd,
                analytic_kerr=ak,
                analytic_non_kerr=ank,
                source_anchor={"path": str(tex_path), "line": int(lineno), "table": "shadow_diameter"},
            )
        )

    return rows


# 関数: `_parse_delta_table` の入出力契約と処理意図を定義する。

def _parse_delta_table(
    *,
    lines: List[str],
    tex_path: Path,
    tabular_lines: List[str],
    tabular_start_line_no: int,
) -> List[ParsedRow]:
    rows: List[ParsedRow] = []
    raw_rows = _iter_rows_with_anchors(tabular_lines, start_line_no=tabular_start_line_no, allow_blank_method=False)

    current_method: Optional[str] = None
    for lineno, row in raw_rows:
        before, _, after = row.partition("\\\\")
        after_clean = _clean_latex_token(after)

        cols = [c.strip() for c in before.split("&")]
        # 条件分岐: `len(cols) < 5` を満たす経路を評価する。
        if len(cols) < 5:
            continue

        method = _clean_latex_token(cols[0])
        prior = _clean_latex_token(cols[1])

        # 条件分岐: `method.endswith("-") and after_clean` を満たす経路を評価する。
        if method.endswith("-") and after_clean:
            method = f"{method}{after_clean}"

        # 条件分岐: `not method` を満たす経路を評価する。

        if not method:
            method = current_method or ""

        # 条件分岐: `not method` を満たす経路を評価する。

        if not method:
            continue

        current_method = method

        grmhd = _parse_pm_val(cols[2] or "")
        ak = _parse_pm_val(cols[3] or "")
        ank = _parse_pm_val(cols[4] or "")
        # 条件分岐: `not (grmhd and ak and ank)` を満たす経路を評価する。
        if not (grmhd and ak and ank):
            continue

        rows.append(
            ParsedRow(
                method=method,
                prior=prior or None,
                grmhd=grmhd,
                analytic_kerr=ak,
                analytic_non_kerr=ank,
                source_anchor={"path": str(tex_path), "line": int(lineno), "table": "delta"},
            )
        )

    return rows


# 関数: `_envelope` の入出力契約と処理意図を定義する。

def _envelope(rows: List[ParsedRow]) -> Dict[str, Any]:
    mids = []
    lo_edges = []
    hi_edges = []
    for r in rows:
        for v in (r.grmhd, r.analytic_kerr, r.analytic_non_kerr):
            mids.append(float(v["mid"]))
            lo_edges.append(float(v["lo"]))
            hi_edges.append(float(v["hi"]))

    # 条件分岐: `not mids` を満たす経路を評価する。

    if not mids:
        return {"n": 0}

    return {
        "n": int(len(mids)),
        "mid_min": float(min(mids)),
        "mid_max": float(max(mids)),
        "ci68_lo_min": float(min(lo_edges)),
        "ci68_hi_max": float(max(hi_edges)),
    }


# 関数: `_summary` の入出力契約と処理意図を定義する。

def _summary(values: List[float]) -> Dict[str, Any]:
    xs: List[float] = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue

        # 条件分岐: `math.isfinite(x)` を満たす経路を評価する。

        if math.isfinite(x):
            xs.append(x)

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

    # 条件分岐: `n % 2 == 1` を満たす経路を評価する。

    if n % 2 == 1:
        median = xs_sorted[n // 2]
    else:
        median = 0.5 * (xs_sorted[n // 2 - 1] + xs_sorted[n // 2])

    return {
        "n": int(n),
        "mean": float(mean),
        "std": float(std),
        "median": float(median),
        "min": float(xs_sorted[0]),
        "max": float(xs_sorted[-1]),
        "rel_std": float(std / mean) if mean != 0 else float("nan"),
    }


# 関数: `_get_sgra_ring_diameter_uas` の入出力契約と処理意図を定義する。

def _get_sgra_ring_diameter_uas(shadow_compare: Dict[str, Any]) -> Optional[float]:
    for r in shadow_compare.get("rows", []):
        # 条件分岐: `isinstance(r, dict) and r.get("key") == "sgra"` を満たす経路を評価する。
        if isinstance(r, dict) and r.get("key") == "sgra":
            try:
                v = float(r.get("ring_diameter_obs_uas"))
                return v if math.isfinite(v) and v > 0 else None
            except Exception:
                return None

    return None


# 関数: `_kappa_from_dsh` の入出力契約と処理意図を定義する。

def _kappa_from_dsh(*, ring_uas: float, dsh: Dict[str, float]) -> Dict[str, float]:
    mid = ring_uas / float(dsh["mid"])
    lo = ring_uas / float(dsh["hi"])  # larger shadow -> smaller kappa
    hi = ring_uas / float(dsh["lo"])  # smaller shadow -> larger kappa
    sigma_minus = mid - lo
    sigma_plus = hi - mid
    sigma = 0.5 * (sigma_minus + sigma_plus)
    return {
        "mid": float(mid),
        "ci68_lo": float(lo),
        "ci68_hi": float(hi),
        "sigma_minus": float(sigma_minus),
        "sigma_plus": float(sigma_plus),
        "sigma_avg": float(sigma),
        "rel_sigma_avg": float(sigma / mid) if mid != 0 else float("nan"),
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.09484" / "main.tex"
    default_shadow_compare = root / "output" / "private" / "eht" / "eht_shadow_compare.json"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(description="Extract Sgr A* Paper VI (arXiv:2311.09484) metric-test constraints tables.")
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper VI main.tex)")
    ap.add_argument(
        "--shadow-compare-json",
        type=str,
        default=str(default_shadow_compare),
        help="Optional ring diameter source (default: output/private/eht/eht_shadow_compare.json)",
    )
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/private/eht)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    shadow_compare_path = Path(args.shadow_compare_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "eht_sgra_paper6_metric_constraints.json"

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"tex": str(tex_path), "shadow_compare_json": str(shadow_compare_path)},
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

    text = _read_text(tex_path)
    lines = text.splitlines()

    dsh_block = _find_table_block(lines, "The Inferred Shadow Diameter")
    delta_block = _find_table_block(lines, "Schwarzschild Deviation Parameter")
    # 条件分岐: `dsh_block is None or delta_block is None` を満たす経路を評価する。
    if dsh_block is None or delta_block is None:
        payload["ok"] = False
        payload["reason"] = "table_block_not_found"
        payload["extracted"]["found_blocks"] = {"shadow_diameter": bool(dsh_block), "delta": bool(delta_block)}
        _write_json(out_json, payload)
        print(f"[warn] table not found; wrote: {out_json}")
        return 0

    dsh_start, dsh_end = dsh_block
    delta_start, delta_end = delta_block

    payload["extracted"]["shadow_diameter_block"] = {
        "start_line": int(dsh_start + 1),
        "end_line": int(dsh_end + 1),
    }
    payload["extracted"]["delta_block"] = {"start_line": int(delta_start + 1), "end_line": int(delta_end + 1)}

    dsh_block_lines = lines[dsh_start : dsh_end + 1]
    delta_block_lines = lines[delta_start : delta_end + 1]

    dsh_tabular = _extract_tabular_lines(dsh_block_lines)
    delta_tabular = _extract_tabular_lines(delta_block_lines)
    # 条件分岐: `dsh_tabular is None or delta_tabular is None` を満たす経路を評価する。
    if dsh_tabular is None or delta_tabular is None:
        payload["ok"] = False
        payload["reason"] = "tabular_not_found"
        _write_json(out_json, payload)
        print(f"[warn] tabular not found; wrote: {out_json}")
        return 0

    dsh_a, dsh_b = dsh_tabular
    delta_a, delta_b = delta_tabular
    dsh_tab_lines = dsh_block_lines[dsh_a : dsh_b + 1]
    delta_tab_lines = delta_block_lines[delta_a : delta_b + 1]

    dsh_rows = _parse_dsh_table(
        lines=lines,
        tex_path=tex_path,
        block_start_line_no=dsh_start + 1,
        tabular_lines=dsh_tab_lines,
        tabular_start_line_no=(dsh_start + 1) + dsh_a,
    )
    delta_rows = _parse_delta_table(
        lines=lines,
        tex_path=tex_path,
        tabular_lines=delta_tab_lines,
        tabular_start_line_no=(delta_start + 1) + delta_a,
    )

    payload["extracted"]["shadow_diameter_rows_n"] = int(len(dsh_rows))
    payload["extracted"]["delta_rows_n"] = int(len(delta_rows))
    payload["extracted"]["shadow_diameter_rows"] = [
        {
            "method": r.method,
            "grmhd": r.grmhd,
            "analytic_kerr": r.analytic_kerr,
            "analytic_non_kerr": r.analytic_non_kerr,
            "source_anchor": r.source_anchor,
        }
        for r in dsh_rows
    ]
    payload["extracted"]["delta_rows"] = [
        {
            "method": r.method,
            "theta_g_prior": r.prior,
            "grmhd": r.grmhd,
            "analytic_kerr": r.analytic_kerr,
            "analytic_non_kerr": r.analytic_non_kerr,
            "source_anchor": r.source_anchor,
        }
        for r in delta_rows
    ]

    payload["derived"]["shadow_diameter_envelope_uas"] = _envelope(dsh_rows)
    payload["derived"]["delta_envelope"] = _envelope(delta_rows)

    ring_uas = None
    # 条件分岐: `shadow_compare_path.exists()` を満たす経路を評価する。
    if shadow_compare_path.exists():
        try:
            ring_uas = _get_sgra_ring_diameter_uas(_read_json(shadow_compare_path))
        except Exception:
            ring_uas = None

    # 条件分岐: `ring_uas is not None` を満たす経路を評価する。

    if ring_uas is not None:
        kappas = []
        for r in dsh_rows:
            for model_key, v in (
                ("grmhd", r.grmhd),
                ("analytic_kerr", r.analytic_kerr),
                ("analytic_non_kerr", r.analytic_non_kerr),
            ):
                k = _kappa_from_dsh(ring_uas=ring_uas, dsh=v)
                kappas.append(
                    {
                        "method": r.method,
                        "model": model_key,
                        "kappa_ring_over_shadow": k,
                        "shadow_diameter_uas": v,
                        "source_anchor": r.source_anchor,
                    }
                )

        kappa_mids = [float(e["kappa_ring_over_shadow"]["mid"]) for e in kappas if isinstance(e, dict)]
        kappa_mid_by_method: List[Dict[str, Any]] = []
        for method in sorted({str(e.get("method") or "") for e in kappas if isinstance(e, dict)}):
            # 条件分岐: `not method` を満たす経路を評価する。
            if not method:
                continue

            mids = [
                float(e["kappa_ring_over_shadow"]["mid"])
                for e in kappas
                if isinstance(e, dict) and str(e.get("method") or "") == method
            ]
            kappa_mid_by_method.append({"method": method, "kappa_mid_summary": _summary(mids)})

        payload["derived"]["kappa_from_paper6_shadow_diameter_table"] = {
            "ring_diameter_obs_uas": float(ring_uas),
            "entries_n": int(len(kappas)),
            "entries": kappas,
            "kappa_mid_summary": _summary(kappa_mids),
            "kappa_mid_by_method": kappa_mid_by_method,
            "sigma_avg_max": float(max(e["kappa_ring_over_shadow"]["sigma_avg"] for e in kappas)) if kappas else None,
            "sigma_avg_median": (
                float(sorted(e["kappa_ring_over_shadow"]["sigma_avg"] for e in kappas)[len(kappas) // 2]) if kappas else None
            ),
        }

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper6_metric_constraints",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {
                    "ok": bool(payload.get("ok")),
                    "shadow_diameter_rows_n": int(len(dsh_rows)),
                    "delta_rows_n": int(len(delta_rows)),
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
