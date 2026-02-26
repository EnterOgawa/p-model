#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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


_RE_ALMA_SMT = re.compile(
    r"\\Delta\s*g_\{\\rm\{ALMA\}\}.*?\\Delta\s*g_\{\\rm\{SMT\}\}.*?estimated\s+to\s+be\s+([0-9]*\.?[0-9]+)\s+and\s+([0-9]*\.?[0-9]+)",
    flags=re.IGNORECASE,
)
_RE_LMT_TOT = re.compile(
    r"\\Delta\s*g_\{\\rm\{LMT\}\}.*?estimated\s+to\s+be\s+([0-9]*\.?[0-9]+).*?\\Delta\s*g_\{\\rm\{tot\}\}.*?estimated\s+to\s+be\s+at\s+most\s+([0-9]*\.?[0-9]+)",
    flags=re.IGNORECASE,
)


def _find_two_floats(
    lines: Sequence[str], pattern: re.Pattern[str]
) -> Optional[Tuple[int, float, float, str]]:
    for i, raw in enumerate(lines, start=1):
        m = pattern.search(raw)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        a = float(m.group(1))
        b = float(m.group(2))
        # 条件分岐: `not (math.isfinite(a) and math.isfinite(b))` を満たす経路を評価する。
        if not (math.isfinite(a) and math.isfinite(b)):
            continue

        return (i, a, b, raw.strip())

    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.08679" / "main.tex"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(
        description="Extract gain residual uncertainty scalars (Δg) from Sgr A* Paper II (arXiv:2311.08679) TeX."
    )
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper II main.tex)")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/private/eht)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "eht_sgra_paper2_gain_uncertainties_metrics.json"

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

    found_alma_smt = _find_two_floats(lines, _RE_ALMA_SMT)
    found_lmt_tot = _find_two_floats(lines, _RE_LMT_TOT)
    payload["extracted"]["found_blocks"] = {
        "delta_g_alma_smt": bool(found_alma_smt),
        "delta_g_lmt_tot": bool(found_lmt_tot),
    }
    # 条件分岐: `not (found_alma_smt and found_lmt_tot)` を満たす経路を評価する。
    if not (found_alma_smt and found_lmt_tot):
        payload["ok"] = False
        payload["reason"] = "required_value_not_found"
        _write_json(out_json, payload)
        print(f"[warn] values not found; wrote: {out_json}")
        return 0

    line_alma_smt, dg_alma, dg_smt, snippet_alma_smt = found_alma_smt
    line_lmt_tot, dg_lmt, dg_tot, snippet_lmt_tot = found_lmt_tot

    payload["extracted"]["delta_g_alma_smt"] = {
        "delta_g_alma": float(dg_alma),
        "delta_g_smt": float(dg_smt),
        "source_anchor": {"path": str(tex_path), "line": int(line_alma_smt)},
        "source_snippet": snippet_alma_smt,
    }
    payload["extracted"]["delta_g_lmt_tot"] = {
        "delta_g_lmt": float(dg_lmt),
        "delta_g_tot": float(dg_tot),
        "source_anchor": {"path": str(tex_path), "line": int(line_lmt_tot)},
        "source_snippet": snippet_lmt_tot,
    }

    payload["derived"] = {
        "delta_g_alma_smt_quadrature": float(math.sqrt(dg_alma * dg_alma + dg_smt * dg_smt)),
        "delta_g_lmt_smt_quadrature": float(math.sqrt(dg_lmt * dg_lmt + dg_smt * dg_smt)),
        "delta_g_lmt_smt_tot_quadrature": float(math.sqrt(dg_lmt * dg_lmt + dg_smt * dg_smt + dg_tot * dg_tot)),
        "notes": [
            "These Δg values are residual fractional station-gain uncertainties after calibration using calibrators (Paper II).",
            "We treat them as amplitude-calibration systematic scales; they are not a direct uncertainty of the published ring diameter.",
        ],
    }

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper2_gain_uncertainties_metrics",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {
                    "ok": bool(payload.get("ok")),
                    "delta_g_alma": float(dg_alma),
                    "delta_g_smt": float(dg_smt),
                    "delta_g_lmt": float(dg_lmt),
                    "delta_g_tot": float(dg_tot),
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
