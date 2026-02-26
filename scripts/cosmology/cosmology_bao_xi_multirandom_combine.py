#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_xi_multirandom_combine.py

Phase 4（宇宙論）/ Step 4.5B.21.4.4.6.7（multi-random 頑健性）:
既に生成済みの catalog-based ξℓ（xi_from_catalogs）と jackknife covariance を、
random realization（random_index）の違いに対して平均化して「fit可能な dv+cov」を作る。

意図：
- Corrfunc 再計算を避けて、まずは random 依存（ノイズ）を下げた dv+cov を作り、
  peakfit（ε）と cross-check を軽く回せるようにする。
- ここでの平均化は “暫定の頑健性チェック” であり、最終的には
  raw からの multi-random 合成（DR/RR 合算）や結合random（reservoir）へ接続する。

入力（例）:
  - output/private/cosmology/cosmology_bao_xi_from_catalogs_*__<out_tag>_metrics.json
  - output/private/cosmology/cosmology_bao_xi_from_catalogs_*__<out_tag>__jk_cov.npz

出力（固定規約に合わせて新規 out_tag を付与）:
  - output/private/cosmology/cosmology_bao_xi_from_catalogs_*__<out_tag_new>.npz
  - output/private/cosmology/cosmology_bao_xi_from_catalogs_*__<out_tag_new>_metrics.json
  - output/private/cosmology/cosmology_bao_xi_from_catalogs_*__<out_tag_new>__jk_cov.npz
  - output/private/cosmology/cosmology_bao_xi_from_catalogs_*__<out_tag_new>__jk_cov_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# 関数: `_now_utc_iso` の入出力契約と処理意図を定義する。

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_as_relpath` の入出力契約と処理意図を定義する。

def _as_relpath(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_strip_metrics_suffix` の入出力契約と処理意図を定義する。

def _strip_metrics_suffix(metrics_path: Path) -> str:
    name = metrics_path.name
    # 条件分岐: `not name.endswith("_metrics.json")` を満たす経路を評価する。
    if not name.endswith("_metrics.json"):
        raise ValueError(f"expected *_metrics.json, got: {metrics_path}")

    return name[: -len("_metrics.json")]


# 関数: `_split_case_and_out_tag` の入出力契約と処理意図を定義する。

def _split_case_and_out_tag(stem: str) -> Tuple[str, str]:
    # 条件分岐: `"__" not in stem` を満たす経路を評価する。
    if "__" not in stem:
        raise ValueError(f"expected '__<out_tag>' in stem, got: {stem}")

    case_prefix, out_tag = stem.rsplit("__", 1)
    # 条件分岐: `not out_tag` を満たす経路を評価する。
    if not out_tag:
        raise ValueError(f"empty out_tag in stem: {stem}")

    return case_prefix, out_tag


# 関数: `_load_metrics` の入出力契約と処理意図を定義する。

def _load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_assert_same_case` の入出力契約と処理意図を定義する。

def _assert_same_case(metrics_list: List[Dict[str, Any]], paths: List[Path]) -> None:
    # 関数: `key` の入出力契約と処理意図を定義する。
    def key(m: Dict[str, Any]) -> Dict[str, Any]:
        params = m.get("params", {}) or {}
        # Compare the case identity (exclude out_tag).
        return {
            "sample": params.get("sample", None),
            "caps": params.get("caps", None),
            "distance_model": params.get("distance_model", None),
            "z_source": params.get("z_source", None),
            "los": params.get("los", None),
            "weight_scheme": params.get("weight_scheme", None),
            "z_cut": (params.get("z_cut", {}) or {}).get("bin", None),
            "recon_mode": ((params.get("recon", {}) or {}).get("mode", None)),
            "estimator_spec_hash": params.get("estimator_spec_hash", None),
            # Binning/config that must match for dv to be aligned.
            "bins": params.get("bins", None),
            "mu_bins": params.get("mu_bins", None),
        }

    ref = key(metrics_list[0])
    for m, p in zip(metrics_list[1:], paths[1:]):
        k = key(m)
        # 条件分岐: `k != ref` を満たす経路を評価する。
        if k != ref:
            raise ValueError(
                "metrics mismatch (different case config).\n"
                f"ref={ref}\n"
                f"cur={k}\n"
                f"path={p}"
            )


# 関数: `_mean_npz_arrays` の入出力契約と処理意図を定義する。

def _mean_npz_arrays(npz_paths: List[Path]) -> Dict[str, np.ndarray]:
    # 条件分岐: `not npz_paths` を満たす経路を評価する。
    if not npz_paths:
        raise ValueError("empty npz_paths")

    keys = None
    arrays_sum: Dict[str, np.ndarray] = {}
    arrays_first: Dict[str, np.ndarray] = {}

    for i, p in enumerate(npz_paths):
        with np.load(p) as z:
            # 条件分岐: `keys is None` を満たす経路を評価する。
            if keys is None:
                keys = list(z.files)
                for k in keys:
                    arrays_first[k] = np.asarray(z[k])
                    arrays_sum[k] = np.asarray(z[k], dtype=np.float64)
            else:
                # 条件分岐: `list(z.files) != list(keys)` を満たす経路を評価する。
                if list(z.files) != list(keys):
                    raise ValueError(f"npz keys mismatch: {p} vs {npz_paths[0]}")

                for k in keys:
                    arrays_sum[k] = arrays_sum[k] + np.asarray(z[k], dtype=np.float64)

    assert keys is not None
    n = float(len(npz_paths))
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        v0 = arrays_first[k]
        # 条件分岐: `v0.ndim == 0` を満たす経路を評価する。
        if v0.ndim == 0:
            out[k] = np.asarray(float(arrays_sum[k]) / n, dtype=np.float64)
        else:
            out[k] = (np.asarray(arrays_sum[k], dtype=np.float64) / n).astype(np.float64, copy=False)

    return out


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Combine xi_from_catalogs outputs across multiple random realizations.")
    ap.add_argument(
        "--xi-metrics-json",
        nargs="+",
        required=True,
        help="Input xi metrics json files (same case, different out_tag=random).",
    )
    ap.add_argument("--out-tag", required=True, help="New out_tag for the combined outputs.")
    ap.add_argument(
        "--out-dir",
        default="output/private/cosmology",
        help="Output directory (default: output/private/cosmology).",
    )
    ap.add_argument(
        "--cov-mode",
        choices=["mean", "first", "provided"],
        default="mean",
        help=(
            "How to populate the output __jk_cov.npz. "
            "'mean' averages input cov matrices (requires each input to have __jk_cov.npz). "
            "'first' copies the first input cov file. "
            "'provided' copies --jk-cov-npz and does not require input cov files. "
            "(default: mean)"
        ),
    )
    ap.add_argument(
        "--jk-cov-npz",
        default="",
        help="When --cov-mode provided: path to an existing __jk_cov.npz to use for the combined output.",
    )

    args = ap.parse_args(argv)
    out_dir = (_ROOT / str(args.out_dir)).resolve() if not Path(str(args.out_dir)).is_absolute() else Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_paths = [(_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p) for p in list(args.xi_metrics_json)]
    for p in metrics_paths:
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            raise SystemExit(f"missing metrics json: {p}")

    metrics_list = [_load_metrics(p) for p in metrics_paths]
    _assert_same_case(metrics_list, metrics_paths)

    stem0 = _strip_metrics_suffix(metrics_paths[0])
    case_prefix, _ = _split_case_and_out_tag(stem0)
    out_tag_new = str(args.out_tag).strip()
    # 条件分岐: `not out_tag_new` を満たす経路を評価する。
    if not out_tag_new:
        raise SystemExit("--out-tag must be non-empty")

    out_stem = f"{case_prefix}__{out_tag_new}"

    in_npz_paths = []
    in_jk_paths = []
    cov_mode = str(args.cov_mode)
    for mp in metrics_paths:
        in_stem = _strip_metrics_suffix(mp)
        in_npz = mp.with_name(f"{in_stem}.npz")
        # 条件分岐: `not in_npz.exists()` を満たす経路を評価する。
        if not in_npz.exists():
            raise SystemExit(f"missing xi npz for {mp}: {in_npz}")

        in_npz_paths.append(in_npz)
        # 条件分岐: `cov_mode in ("mean", "first")` を満たす経路を評価する。
        if cov_mode in ("mean", "first"):
            in_jk = mp.with_name(f"{in_stem}__jk_cov.npz")
            # 条件分岐: `not in_jk.exists()` を満たす経路を評価する。
            if not in_jk.exists():
                raise SystemExit(f"missing jk cov npz for {mp}: {in_jk}")

            in_jk_paths.append(in_jk)

    out_npz = out_dir / f"{out_stem}.npz"
    out_metrics = out_dir / f"{out_stem}_metrics.json"
    out_jk_cov = out_dir / f"{out_stem}__jk_cov.npz"
    out_jk_metrics = out_dir / f"{out_stem}__jk_cov_metrics.json"

    # Combine xi npz (mean by key).
    mean_npz = _mean_npz_arrays(in_npz_paths)
    np.savez_compressed(out_npz, **mean_npz)

    # Covariance output.
    if cov_mode == "provided":
        jk_cov_npz_arg = str(args.jk_cov_npz).strip()
        # 条件分岐: `not jk_cov_npz_arg` を満たす経路を評価する。
        if not jk_cov_npz_arg:
            raise SystemExit("--cov-mode provided requires --jk-cov-npz")

        cov_src = (_ROOT / jk_cov_npz_arg).resolve() if not Path(jk_cov_npz_arg).is_absolute() else Path(jk_cov_npz_arg)
        # 条件分岐: `not cov_src.exists()` を満たす経路を評価する。
        if not cov_src.exists():
            raise SystemExit(f"missing --jk-cov-npz: {cov_src}")

        with np.load(cov_src) as zc:
            s0 = np.asarray(zc["s"], dtype=np.float64).reshape(-1)
            cov_full = np.asarray(zc["cov"], dtype=np.float64)
            yjk = np.asarray(zc["y_jk"], dtype=np.float64) if "y_jk" in zc.files else np.zeros((0, 0), dtype=np.float64)
            ra0 = np.asarray(zc["ra_edges_deg"], dtype=np.float64) if "ra_edges_deg" in zc.files else np.zeros(0, dtype=np.float64)

        cov_full = np.asarray(cov_full, dtype=np.float64).reshape(2 * int(s0.size), 2 * int(s0.size))
        # Sanity check: xi npz s bins must match.
        s_xi = np.asarray(mean_npz.get("s", []), dtype=np.float64).reshape(-1)
        # 条件分岐: `s_xi.size and (s_xi.shape != s0.shape or not np.allclose(s_xi, s0, rtol=0.0,...` を満たす経路を評価する。
        if s_xi.size and (s_xi.shape != s0.shape or not np.allclose(s_xi, s0, rtol=0.0, atol=1e-12)):
            raise SystemExit(f"provided jk cov uses different s bins vs combined xi npz: {cov_src}")

        np.savez_compressed(out_jk_cov, s=s0, cov=cov_full, y_jk=yjk, ra_edges_deg=ra0)
        cov_source_note: Dict[str, Any] = {"mode": "provided", "jk_cov_npz": _as_relpath(cov_src)}
    # 条件分岐: 前段条件が不成立で、`cov_mode == "first"` を追加評価する。
    elif cov_mode == "first":
        src = in_jk_paths[0]
        with np.load(src) as zc:
            s0 = np.asarray(zc["s"], dtype=np.float64).reshape(-1)
            cov_full = np.asarray(zc["cov"], dtype=np.float64)
            yjk = np.asarray(zc["y_jk"], dtype=np.float64) if "y_jk" in zc.files else np.zeros((0, 0), dtype=np.float64)
            ra0 = np.asarray(zc["ra_edges_deg"], dtype=np.float64) if "ra_edges_deg" in zc.files else np.zeros(0, dtype=np.float64)

        cov_full = np.asarray(cov_full, dtype=np.float64).reshape(2 * int(s0.size), 2 * int(s0.size))
        np.savez_compressed(out_jk_cov, s=s0, cov=cov_full, y_jk=yjk, ra_edges_deg=ra0)
        cov_source_note = {"mode": "first", "jk_cov_npz": _as_relpath(src)}
    else:
        # mean
        yjk_list: List[np.ndarray] = []
        cov_list: List[np.ndarray] = []
        s_list: List[np.ndarray] = []
        ra_edges_list: List[np.ndarray] = []
        for p in in_jk_paths:
            with np.load(p) as z:
                s_list.append(np.asarray(z["s"], dtype=np.float64))
                cov_list.append(np.asarray(z["cov"], dtype=np.float64))
                # 条件分岐: `"y_jk" in z.files` を満たす経路を評価する。
                if "y_jk" in z.files:
                    yjk_list.append(np.asarray(z["y_jk"], dtype=np.float64))

                # 条件分岐: `"ra_edges_deg" in z.files` を満たす経路を評価する。

                if "ra_edges_deg" in z.files:
                    ra_edges_list.append(np.asarray(z["ra_edges_deg"], dtype=np.float64))

        s0 = np.asarray(s_list[0], dtype=np.float64).reshape(-1)
        for i, s in enumerate(s_list[1:], start=1):
            s = np.asarray(s, dtype=np.float64).reshape(-1)
            # 条件分岐: `s.shape != s0.shape or np.max(np.abs(s - s0)) > 0.0` を満たす経路を評価する。
            if s.shape != s0.shape or np.max(np.abs(s - s0)) > 0.0:
                raise SystemExit(f"jackknife s mismatch: {in_jk_paths[i]} vs {in_jk_paths[0]}")

        cov_mean = np.mean(np.stack(cov_list, axis=0), axis=0)
        cov_mean = 0.5 * (cov_mean + cov_mean.T)

        yjk_mean = np.mean(np.stack(yjk_list, axis=0), axis=0) if yjk_list else np.zeros((0, 0), dtype=np.float64)
        ra0 = ra_edges_list[0] if ra_edges_list else np.zeros(0, dtype=np.float64)
        # 条件分岐: `ra_edges_list` を満たす経路を評価する。
        if ra_edges_list:
            for i, ra in enumerate(ra_edges_list[1:], start=1):
                # 条件分岐: `ra.shape != ra0.shape or np.max(np.abs(ra - ra0)) > 0.0` を満たす経路を評価する。
                if ra.shape != ra0.shape or np.max(np.abs(ra - ra0)) > 0.0:
                    raise SystemExit(f"jackknife ra_edges mismatch: {in_jk_paths[i]} vs {in_jk_paths[0]}")

        np.savez_compressed(out_jk_cov, s=s0, cov=cov_mean, y_jk=yjk_mean, ra_edges_deg=ra0)
        cov_source_note = {"mode": "mean", "jk_cov_npz": [_as_relpath(p) for p in in_jk_paths]}

    # Metrics json (xi)

    m_out = dict(metrics_list[0])
    m_out["generated_utc"] = _now_utc_iso()
    m_out.setdefault("inputs", {})
    # 条件分岐: `isinstance(m_out["inputs"], dict)` を満たす経路を評価する。
    if isinstance(m_out["inputs"], dict):
        m_out["inputs"]["multirandom_sources"] = [_as_relpath(p) for p in metrics_paths]

    m_out.setdefault("params", {})
    # 条件分岐: `isinstance(m_out["params"], dict)` を満たす経路を評価する。
    if isinstance(m_out["params"], dict):
        m_out["params"]["out_tag"] = out_tag_new

    m_out.setdefault("derived", {})
    # 条件分岐: `isinstance(m_out["derived"], dict)` を満たす経路を評価する。
    if isinstance(m_out["derived"], dict):
        m_out["derived"]["multirandom_combine"] = {
            "n_inputs": int(len(metrics_paths)),
            "xi_method": "mean_on_saved_npz",
            "jackknife_cov_mode": cov_source_note,
        }

    m_out["outputs"] = {"npz": _as_relpath(out_npz)}
    out_metrics.write_text(json.dumps(m_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Metrics json (jk)
    jk_meta: Dict[str, Any] = {
        "generated_utc": _now_utc_iso(),
        "domain": "cosmology",
        "step": "4.5B.21.4.4.6.7 (multirandom combine; jackknife cov)",
        "inputs": {"jk_cov_npz": cov_source_note},
        "outputs": {"jk_cov_npz": _as_relpath(out_jk_cov)},
        "params": {"n_inputs": int(len(metrics_paths))},
    }
    out_jk_metrics.write_text(json.dumps(jk_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[ok] wrote:")
    print(f"  {out_npz}")
    print(f"  {out_metrics}")
    print(f"  {out_jk_cov}")
    print(f"  {out_jk_metrics}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
