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


def _find_block(text: str, needle: str, *, window: int = 1200) -> Optional[str]:
    i = text.find(needle)
    # 条件分岐: `i < 0` を満たす経路を評価する。
    if i < 0:
        return None

    a = max(0, i - 160)
    b = min(len(text), i + window)
    return text[a:b]


_RE_BASELINE = re.compile(r"\\multicolumn\{1\}\{c\}\{(?P<u>\d+(?:\.\d+)?)\}")
_RE_PM = re.compile(r"(?P<mid>\d+(?:\.\d+)?)\s*\\pm\s*(?P<sig>\d+(?:\.\d+)?)")
_RE_TENPOW = re.compile(r"\\times\s*10\^\{\s*(?P<exp>[-+]?\d+)\s*\}")


def _parse_norm_variance(cell: str) -> Optional[Dict[str, Any]]:
    s = cell.strip().strip("$")
    s = s.replace(" ", "")
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None
    # Example: (17.2\pm14.3)\times10^{-4}

    m_pm = _RE_PM.search(s)
    m_exp = _RE_TENPOW.search(s)
    # 条件分岐: `not (m_pm and m_exp)` を満たす経路を評価する。
    if not (m_pm and m_exp):
        return None

    mid = float(m_pm.group("mid"))
    sig = float(m_pm.group("sig"))
    exp = int(m_exp.group("exp"))
    # 条件分岐: `not (math.isfinite(mid) and math.isfinite(sig))` を満たす経路を評価する。
    if not (math.isfinite(mid) and math.isfinite(sig)):
        return None

    scale = 10.0 ** float(exp)
    mid_scaled = mid * scale
    sig_scaled = sig * scale
    rms = math.sqrt(mid_scaled) if mid_scaled >= 0 else float("nan")
    return {
        "mid": float(mid_scaled),
        "sigma": float(sig_scaled),
        "exp10": int(exp),
        "rms_fraction": float(rms) if math.isfinite(rms) else None,
    }


def _parse_percent_pm(cell: str) -> Optional[Tuple[float, float]]:
    s = cell.strip().strip("$")
    s = s.replace(" ", "")
    m = _RE_PM.search(s)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return None

    mid = float(m.group("mid"))
    sig = float(m.group("sig"))
    # 条件分岐: `not (math.isfinite(mid) and math.isfinite(sig))` を満たす経路を評価する。
    if not (math.isfinite(mid) and math.isfinite(sig)):
        return None

    return (mid, sig)


def _parse_float_pm(cell: str) -> Optional[Tuple[float, float]]:
    s = cell.strip().strip("$")
    s = s.replace(" ", "")
    m = _RE_PM.search(s)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return None

    mid = float(m.group("mid"))
    sig = float(m.group("sig"))
    # 条件分岐: `not (math.isfinite(mid) and math.isfinite(sig))` を満たす経路を評価する。
    if not (math.isfinite(mid) and math.isfinite(sig)):
        return None

    return (mid, sig)


def _parse_upper_limit(cell: str) -> Optional[float]:
    s = cell.strip().strip("$")
    s = s.replace(" ", "")
    # 条件分岐: `not s.startswith("<")` を満たす経路を評価する。
    if not s.startswith("<"):
        return None

    try:
        v = float(s[1:].split("~", 1)[0])
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _extract_debiased_table(lines: Sequence[str], *, source_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    label = r"\label{tab:debiased_noise}"

    label_idx = None
    for i, line in enumerate(lines):
        # 条件分岐: `label in line` を満たす経路を評価する。
        if label in line:
            label_idx = i
            break

    # 条件分岐: `label_idx is None` を満たす経路を評価する。

    if label_idx is None:
        return ([], {})

    # Find start/end data block.

    startdata_idx = None
    for j in range(label_idx, len(lines)):
        # 条件分岐: `r"\startdata" in lines[j]` を満たす経路を評価する。
        if r"\startdata" in lines[j]:
            startdata_idx = j
            break

    # 条件分岐: `startdata_idx is None` を満たす経路を評価する。

    if startdata_idx is None:
        return ([], {})

    enddata_idx = None
    for j in range(startdata_idx, len(lines)):
        # 条件分岐: `r"\enddata" in lines[j]` を満たす経路を評価する。
        if r"\enddata" in lines[j]:
            enddata_idx = j
            break

    # 条件分岐: `enddata_idx is None` を満たす経路を評価する。

    if enddata_idx is None:
        enddata_idx = len(lines)

    baseline_rows: List[Dict[str, Any]] = []
    params: Dict[str, Any] = {}
    section = "baseline"

    for off, raw in enumerate(lines[startdata_idx + 1 : enddata_idx], start=0):
        lineno = (startdata_idx + 2) + off
        s = raw.strip()
        # 条件分岐: `not s` を満たす経路を評価する。
        if not s:
            continue

        # 条件分岐: `s.startswith(r"\hline")` を満たす経路を評価する。

        if s.startswith(r"\hline"):
            section = "params"
            continue

        # 条件分岐: `"&" not in s or r"\\" not in s` を満たす経路を評価する。

        if "&" not in s or r"\\" not in s:
            continue

        cols = [c.strip().rstrip("\\").strip() for c in s.split("&")]
        # 条件分岐: `section == "baseline"` を満たす経路を評価する。
        if section == "baseline":
            # 条件分岐: `len(cols) < 3` を満たす経路を評価する。
            if len(cols) < 3:
                continue

            m_u = _RE_BASELINE.search(cols[0])
            # 条件分岐: `not m_u` を満たす経路を評価する。
            if not m_u:
                continue

            u = float(m_u.group("u"))
            nv = _parse_norm_variance(cols[1])
            fit = cols[2].strip().upper()
            # 条件分岐: `nv is None or fit not in {"Y", "N"}` を満たす経路を評価する。
            if nv is None or fit not in {"Y", "N"}:
                continue

            baseline_rows.append(
                {
                    "u_gly": float(u),
                    "normalized_variance": nv,
                    "fit": fit,
                    "source_anchor": {"path": str(source_path), "line": int(lineno), "label": "tab:debiased_noise"},
                }
            )
        else:
            # 条件分岐: `len(cols) < 3` を満たす経路を評価する。
            if len(cols) < 3:
                continue

            q = cols[0]
            sym = cols[1].strip().strip("$")
            est = cols[2]
            # 条件分岐: `sym == "a_4"` を満たす経路を評価する。
            if sym == "a_4":
                pm = _parse_percent_pm(est)
                # 条件分岐: `pm is None` を満たす経路を評価する。
                if pm is None:
                    continue

                params["a4_percent"] = {
                    "mid": float(pm[0]),
                    "sigma": float(pm[1]),
                    "source_anchor": {"path": str(source_path), "line": int(lineno), "label": "tab:debiased_noise"},
                }
            # 条件分岐: 前段条件が不成立で、`sym == "b"` を追加評価する。
            elif sym == "b":
                pm = _parse_float_pm(est)
                # 条件分岐: `pm is None` を満たす経路を評価する。
                if pm is None:
                    continue

                params["b"] = {
                    "mid": float(pm[0]),
                    "sigma": float(pm[1]),
                    "source_anchor": {"path": str(source_path), "line": int(lineno), "label": "tab:debiased_noise"},
                }
            # 条件分岐: 前段条件が不成立で、`sym == "u_0"` を追加評価する。
            elif sym == "u_0":
                ul = _parse_upper_limit(est)
                # 条件分岐: `ul is None` を満たす経路を評価する。
                if ul is None:
                    continue

                params["u0_gly_upper_1sigma"] = {
                    "upper": float(ul),
                    "source_anchor": {"path": str(source_path), "line": int(lineno), "label": "tab:debiased_noise"},
                }
            else:
                # Keep unknown rows for debugging.
                params.setdefault("other", []).append(
                    {
                        "quantity": q,
                        "symbol": sym,
                        "estimate_tex": est,
                        "source_anchor": {"path": str(source_path), "line": int(lineno), "label": "tab:debiased_noise"},
                    }
                )

    return (baseline_rows, params)


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.08697" / "results.tex"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(description="Parse Sgr A* Paper IV debiased variability table (tab:debiased_noise).")
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper IV results.tex)")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/private/eht)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "eht_sgra_paper4_debiased_noise_table_metrics.json"
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

    tex = _read_text(tex_path)
    payload["extracted"]["debiased_noise_anchor_snippet"] = _find_block(tex, r"\label{tab:debiased_noise}")
    lines = tex.splitlines()

    baseline_rows, params = _extract_debiased_table(lines, source_path=tex_path)
    payload["extracted"]["baseline_rows_n"] = int(len(baseline_rows))
    payload["extracted"]["baseline_rows"] = baseline_rows
    payload["extracted"]["params"] = params

    a4 = params.get("a4_percent") if isinstance(params, dict) else None
    a4_mid = a4.get("mid") if isinstance(a4, dict) else None
    a4_sig = a4.get("sigma") if isinstance(a4, dict) else None

    # 条件分岐: `a4_mid is None or a4_sig is None` を満たす経路を評価する。
    if a4_mid is None or a4_sig is None:
        payload["ok"] = False
        payload["reason"] = "a4_not_found"
        _write_json(out_json, payload)
        print(f"[warn] a4 not found; wrote: {out_json}")
        return 0

    # Derived summary: convert a4 (%) to fraction.

    payload["derived"]["a4_fraction_at_4gly"] = {
        "mid": float(a4_mid) / 100.0,
        "sigma": float(a4_sig) / 100.0,
        "units": "fraction",
    }
    payload["derived"]["notes"] = [
        "Paper IV tab:debiased_noise reports debiased intrinsic variability estimates and broken-power-law fit parameters.",
        "We treat a4 (excess noise at |u|=4 Gλ) as an amplitude-variability systematic scale indicator (not a direct mapping to ring diameter uncertainty).",
    ]

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper4_debiased_noise_table_metrics",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {
                    "ok": bool(payload.get("ok")),
                    "baseline_rows_n": int(payload["extracted"].get("baseline_rows_n") or 0),
                    "a4_fraction_mid": float(payload["derived"]["a4_fraction_at_4gly"]["mid"]),
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
