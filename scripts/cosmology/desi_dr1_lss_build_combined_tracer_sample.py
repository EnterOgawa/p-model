#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
desi_dr1_lss_build_combined_tracer_sample.py

Phase 4 / Step 4.5B.21.4.4.7（DESI DR1: multi-tracer）補助:
既に抽出済みの DESI DR1 LSS（galaxy/random）NPZ を材料に、
複数 tracer（例：LRG + ELG）を単一サンプルとして結合した data_dir を作る。

目的：
- DESI BAO VI の LRG3+ELG1（0.8<z<1.1）など、複数 tracer の合算ケースを
  「一次統計：銀河+random→ξ(s,μ)→ξℓ→dv+cov→ε fit」へ接続する。

方針：
- 入力 data_dir の manifest.json（例：data/cosmology/desi_dr1_lss_reservoir_r0/manifest.json）にある
  item（{sample}:{cap}）を参照し、同一 cap（NGC/SGC）内で tracer を連結する。
- random は連結後に任意で downsample（target_rows）できる（計算コスト抑制）。

出力：
  <out-data-dir>/
    extracted/
      - <OUT_SAMPLE>_NGC_clustering.dat.fits.npz
      - <OUT_SAMPLE>_SGC_clustering.dat.fits.npz
      - <OUT_SAMPLE>_NGC_0toX_clustering.ran.fits.combined_reservoir_<N>_seed<S>.npz
      - <OUT_SAMPLE>_SGC_0toX_clustering.ran.fits.combined_reservoir_<N>_seed<S>.npz
    manifest.json
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

from scripts.summary import worklog  # noqa: E402


def _relpath(p: Path) -> str:
    try:
        return p.relative_to(_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _load_manifest(path: Path) -> Dict[str, Any]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")

    obj = json.loads(path.read_text(encoding="utf-8"))
    # 条件分岐: `not isinstance(obj, dict)` を満たす経路を評価する。
    if not isinstance(obj, dict):
        raise ValueError("manifest must be a JSON object")

    return obj


def _require_item(manifest: Dict[str, Any], *, sample: str, cap: str) -> Dict[str, Any]:
    key = f"{str(sample).strip()}:{str(cap).strip()}"
    it = (manifest.get("items") or {}).get(key)
    # 条件分岐: `not isinstance(it, dict) or "galaxy" not in it or "random" not in it` を満たす経路を評価する。
    if not isinstance(it, dict) or "galaxy" not in it or "random" not in it:
        raise KeyError(f"missing item {key} (galaxy/random) in manifest")

    return it


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _concat_npz_dicts(dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    # 条件分岐: `not dicts` を満たす経路を評価する。
    if not dicts:
        return {}

    keys = list(dicts[0].keys())
    # 条件分岐: `not keys` を満たす経路を評価する。
    if not keys:
        return {}

    for d in dicts[1:]:
        # 条件分岐: `list(d.keys()) != keys` を満たす経路を評価する。
        if list(d.keys()) != keys:
            raise ValueError("npz keys mismatch across inputs (ensure same extracted columns)")

    out: Dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.concatenate([np.asarray(d[k]) for d in dicts], axis=0)

    return out


def _downsample_rows(
    arrs: Dict[str, np.ndarray], *, target_rows: int | None, seed: int, shuffle: bool
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    # 条件分岐: `not arrs` を満たす経路を評価する。
    if not arrs:
        return arrs, {"target_rows": target_rows, "selected_rows": 0}

    n = int(next(iter(arrs.values())).shape[0])
    for k, v in arrs.items():
        # 条件分岐: `int(np.asarray(v).shape[0]) != n` を満たす経路を評価する。
        if int(np.asarray(v).shape[0]) != n:
            raise ValueError(f"array length mismatch in npz: key={k} n={np.asarray(v).shape[0]} vs {n}")

    # 条件分岐: `target_rows is None or int(target_rows) >= n` を満たす経路を評価する。

    if target_rows is None or int(target_rows) >= n:
        sel = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        sel = rng.choice(n, size=int(target_rows), replace=False)
        sel.sort()

    out = {k: np.asarray(v)[sel] for k, v in arrs.items()}
    # 条件分岐: `shuffle and sel.size > 0` を満たす経路を評価する。
    if shuffle and sel.size > 0:
        rng2 = np.random.default_rng(int(seed) + 17_000_000)
        perm = np.arange(int(sel.size), dtype=np.int64)
        rng2.shuffle(perm)
        out = {k: np.asarray(v)[perm] for k, v in out.items()}

    return out, {"rows_in": n, "rows_out": int(sel.size), "seed": int(seed), "shuffle_rows": bool(shuffle)}


def _desi_default_weight(cols: Dict[str, np.ndarray]) -> np.ndarray:
    w_fkp = np.asarray(cols["WEIGHT_FKP"], dtype=np.float64)
    w_base = np.asarray(cols["WEIGHT"], dtype=np.float64)
    return w_fkp * w_base


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a combined-tracer DESI DR1 LSS sample from extracted NPZ.")
    ap.add_argument(
        "--in-data-dir",
        type=str,
        required=True,
        help=(
            "Default input DESI data dir with manifest.json + extracted NPZ "
            "(used for any sample not overridden by --in-data-dir-map). "
            "Example: data/cosmology/desi_dr1_lss_reservoir_r0"
        ),
    )
    ap.add_argument(
        "--in-data-dir-map",
        type=str,
        default="",
        help=(
            "Optional per-sample input dir overrides as comma-separated 'sample=dir' pairs. "
            "Example: lrg=data/cosmology/desi_dr1_lss_reservoir_r0to17_mix,elg_lopnotqso=data/cosmology/desi_dr1_lss_elg_lopnotqso_reservoir_r0to17_mix"
        ),
    )
    ap.add_argument(
        "--samples",
        type=str,
        required=True,
        help="Comma-separated tracer sample ids to combine (e.g., lrg,elg_lopnotqso)",
    )
    ap.add_argument(
        "--out-sample",
        type=str,
        required=True,
        help="Output combined sample id to write into manifest (e.g., lrg_elg_lopnotqso)",
    )
    ap.add_argument(
        "--out-data-dir",
        type=str,
        required=True,
        help="Output data dir (e.g., data/cosmology/desi_dr1_lss_lrg_elg_reservoir_r0)",
    )
    ap.add_argument(
        "--random-kind",
        type=str,
        default="random",
        help="Random kind to expect in input manifest (default: random)",
    )
    ap.add_argument(
        "--random-rescale",
        choices=["none", "match_combined_total"],
        default="none",
        help=(
            "Optional: rescale per-sample random weights before concatenation to make the random-to-galaxy "
            "normalization consistent across samples within each cap. "
            "match_combined_total preserves the total random weight sum per cap."
        ),
    )
    ap.add_argument(
        "--target-random-rows",
        type=int,
        default=2_000_000,
        help="Downsample concatenated random to this many rows per cap (default: 2,000,000). Use 0 to keep all.",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    ap.add_argument("--no-shuffle", action="store_true", help="Do not shuffle rows after (optional) downsampling.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_dir = (_ROOT / Path(str(args.in_data_dir))).resolve() if not Path(str(args.in_data_dir)).is_absolute() else Path(str(args.in_data_dir))
    out_dir = (_ROOT / Path(str(args.out_data_dir))).resolve() if not Path(str(args.out_data_dir)).is_absolute() else Path(str(args.out_data_dir))
    out_ext = out_dir / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ext.mkdir(parents=True, exist_ok=True)

    in_manifest_path = in_dir / "manifest.json"
    base_manifest = _load_manifest(in_manifest_path)

    in_dir_map: Dict[str, Path] = {}
    # 条件分岐: `str(args.in_data_dir_map).strip()` を満たす経路を評価する。
    if str(args.in_data_dir_map).strip():
        for part in str(args.in_data_dir_map).split(","):
            p = part.strip()
            # 条件分岐: `not p` を満たす経路を評価する。
            if not p:
                continue

            # 条件分岐: `"=" not in p` を満たす経路を評価する。

            if "=" not in p:
                raise SystemExit(f"invalid --in-data-dir-map entry (expected sample=dir): {p!r}")

            s_key, s_dir = p.split("=", 1)
            s_key = str(s_key).strip().lower()
            # 条件分岐: `not s_key` を満たす経路を評価する。
            if not s_key:
                raise SystemExit(f"invalid --in-data-dir-map entry (empty sample): {p!r}")

            d = (_ROOT / Path(str(s_dir).strip())).resolve() if not Path(str(s_dir).strip()).is_absolute() else Path(str(s_dir).strip())
            in_dir_map[s_key] = d

    manifest_by_dir: Dict[Path, Dict[str, Any]] = {in_dir: base_manifest}
    for d in in_dir_map.values():
        # 条件分岐: `d not in manifest_by_dir` を満たす経路を評価する。
        if d not in manifest_by_dir:
            manifest_by_dir[d] = _load_manifest(d / "manifest.json")

    samples = [s.strip().lower() for s in str(args.samples).split(",") if s.strip()]
    # 条件分岐: `not samples` を満たす経路を評価する。
    if not samples:
        raise SystemExit("--samples must not be empty")

    out_sample = str(args.out_sample).strip().lower()
    # 条件分岐: `not out_sample` を満たす経路を評価する。
    if not out_sample:
        raise SystemExit("--out-sample must not be empty")

    random_kind = str(args.random_kind).strip()
    random_rescale = str(args.random_rescale).strip()
    target_random_rows = int(args.target_random_rows)
    # 条件分岐: `target_random_rows <= 0` を満たす経路を評価する。
    if target_random_rows <= 0:
        target_random_rows = 0

    shuffle_rows = not bool(args.no_shuffle)

    items_out: Dict[str, Any] = {}
    per_cap_meta: Dict[str, Any] = {}

    for cap in ("north", "south"):
        gal_by_sample: Dict[str, Dict[str, np.ndarray]] = {}
        rnd_by_sample: Dict[str, Dict[str, np.ndarray]] = {}
        in_paths: Dict[str, Any] = {"by_sample": {}}
        for s in samples:
            s_l = str(s).strip().lower()
            d_eff = in_dir_map.get(s_l, in_dir)
            m_eff = manifest_by_dir[d_eff]
            it = _require_item(m_eff, sample=s, cap=cap)
            # 条件分岐: `str((it.get("random") or {}).get("kind", "")).strip() != random_kind` を満たす経路を評価する。
            if str((it.get("random") or {}).get("kind", "")).strip() != random_kind:
                got = str((it.get("random") or {}).get("kind", "")).strip()
                raise SystemExit(f"random kind mismatch for {s}:{cap}: expected {random_kind!r} but got {got!r}")

            gal_npz = (_ROOT / Path(str(it["galaxy"]["npz_path"]))).resolve()
            rnd_npz = (_ROOT / Path(str(it["random"]["npz_path"]))).resolve()
            # 条件分岐: `not gal_npz.exists()` を満たす経路を評価する。
            if not gal_npz.exists():
                raise SystemExit(f"missing galaxy npz: {gal_npz}")

            # 条件分岐: `not rnd_npz.exists()` を満たす経路を評価する。

            if not rnd_npz.exists():
                raise SystemExit(f"missing random npz: {rnd_npz}")

            gal_by_sample[s_l] = _load_npz(gal_npz)
            rnd_by_sample[s_l] = _load_npz(rnd_npz)
            in_paths["by_sample"][s_l] = {
                "in_data_dir": _relpath(d_eff),
                "in_manifest": _relpath(d_eff / "manifest.json"),
                "galaxy_npz": _relpath(gal_npz),
                "random_npz": _relpath(rnd_npz),
            }

        # Optional: rescale per-sample random weights (constant per sample+cap).

        rescale_meta: Dict[str, Any] | None = None
        # 条件分岐: `random_rescale == "match_combined_total" and len(samples) >= 2` を満たす経路を評価する。
        if random_rescale == "match_combined_total" and len(samples) >= 2:
            sum_wg: Dict[str, float] = {}
            sum_wr: Dict[str, float] = {}
            for s in samples:
                s_l = str(s).strip().lower()
                wg = _desi_default_weight(gal_by_sample[s_l])
                wr = _desi_default_weight(rnd_by_sample[s_l])
                sum_wg[s_l] = float(np.sum(wg))
                sum_wr[s_l] = float(np.sum(wr))

            denom = float(sum(sum_wg.values()))
            # 条件分岐: `not (denom > 0.0)` を満たす経路を評価する。
            if not (denom > 0.0):
                raise SystemExit(f"invalid sum_wg total for cap={cap}")

            c_cap = float(sum(sum_wr.values())) / denom
            factors: Dict[str, float] = {}
            for s in samples:
                s_l = str(s).strip().lower()
                # 条件分岐: `not (sum_wr[s_l] > 0.0)` を満たす経路を評価する。
                if not (sum_wr[s_l] > 0.0):
                    raise SystemExit(f"invalid sum_wr for cap={cap} sample={s_l}")

                a = float(c_cap) * float(sum_wg[s_l]) / float(sum_wr[s_l])
                factors[s_l] = a
                rnd_by_sample[s_l]["WEIGHT"] = np.asarray(rnd_by_sample[s_l]["WEIGHT"], dtype=np.float64) * float(a)

            rescale_meta = {
                "mode": random_rescale,
                "weight_definition": "desi_default (WEIGHT_FKP*WEIGHT)",
                "c_cap": float(c_cap),
                "sum_wg": sum_wg,
                "sum_wr_before": sum_wr,
                "random_weight_multipliers": factors,
            }

        gal_cat = _concat_npz_dicts([gal_by_sample[str(s).strip().lower()] for s in samples])
        rnd_cat = _concat_npz_dicts([rnd_by_sample[str(s).strip().lower()] for s in samples])

        # Downsample random for cost control (preserve selection by uniform sampling).
        target = None if target_random_rows <= 0 else int(target_random_rows)
        rnd_cat2, rnd_sampling = _downsample_rows(rnd_cat, target_rows=target, seed=int(args.seed) + (0 if cap == "north" else 1), shuffle=shuffle_rows)

        # Write NPZ
        cap_tag = "NGC" if cap == "north" else "SGC"
        gal_name = f"{out_sample.upper()}_{cap_tag}_clustering.dat.fits.npz"
        rnd_name = (
            f"{out_sample.upper()}_{cap_tag}_combined_clustering.ran.fits"
            f".combined_reservoir_{(rnd_sampling.get('rows_out') or 0)}_seed{int(args.seed)}.npz"
        )
        gal_out = out_ext / gal_name
        rnd_out = out_ext / rnd_name

        np.savez_compressed(gal_out, **gal_cat)
        np.savez_compressed(rnd_out, **rnd_cat2)

        key = f"{out_sample}:{cap}"
        items_out[key] = {
            "galaxy": {"npz_path": _relpath(gal_out), "extract": {"sampling": {"method": "concat"}}},
            "random": {
                "kind": random_kind,
                "npz_path": _relpath(rnd_out),
                "extract": {
                    "sampling": {
                        "method": "concat_then_downsample" if target is not None else "concat",
                        "target_rows": target,
                        "seed": int(args.seed),
                        "shuffle_rows": bool(shuffle_rows),
                    },
                    "details": rnd_sampling,
                },
            },
        }
        per_cap_meta[cap] = {
            "inputs": in_paths,
            "galaxy": {"rows_out": int(next(iter(gal_cat.values())).shape[0]) if gal_cat else 0},
            "random": rnd_sampling,
            "random_rescale": rescale_meta,
            "outputs": {"galaxy_npz": _relpath(gal_out), "random_npz": _relpath(rnd_out)},
        }

    out_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "dataset": "desi_dr1_lss",
        "note": "Combined-tracer sample built from already extracted DESI DR1 LSS NPZ catalogs.",
        "inputs": {
            "in_data_dir": _relpath(in_dir),
            "in_manifest": _relpath(in_manifest_path),
            "in_data_dir_map": ({k: _relpath(v) for k, v in in_dir_map.items()} if in_dir_map else None),
            "samples": samples,
            "out_sample": out_sample,
            "random_kind": random_kind,
            "random_rescale": random_rescale,
            "target_random_rows": (None if target_random_rows <= 0 else int(target_random_rows)),
            "seed": int(args.seed),
            "shuffle_rows": bool(shuffle_rows),
        },
        "per_cap": per_cap_meta,
        "items": items_out,
    }

    out_manifest_path = out_dir / "manifest.json"
    out_manifest_path.write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        worklog.append_event(
            {
                "domain": "cosmology",
                "action": "desi_dr1_lss_build_combined_tracer_sample",
                "inputs": {"in_manifest": in_manifest_path},
                "outputs": {"out_manifest": out_manifest_path},
                "params": out_manifest.get("inputs", {}),
            }
        )
    except Exception:
        pass

    print(f"[ok] wrote: {out_manifest_path}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
