#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
desi_dr1_lss_build_combined_random_reservoir.py

Phase 4 / Step 4.5B.21.4.4.6.7（DESI multi-random 頑健性）補助:
既に抽出済みの DESI DR1 LSS（galaxy/random）NPZ を材料に、
複数 random_index の「混合 random（reservoir）」を 1つにまとめた data_dir を作る。

狙い:
  - random_index 依存のノイズ/揺らぎを抑えた 1回の Corrfunc 実行を可能にする（Strategy B）。
  - dv=[xi0,xi2] と sky-jackknife covariance を “同じ random” から生成して整合を保つ。

入出力:
  - 入力: data/cosmology/desi_dr1_lss_reservoir_r{idx}/manifest.json（idx=0..17 など）
  - 出力: data/cosmology/<out-data-dir>/
      - extracted/ (galaxy は r0 をコピー、random は混合して新規作成)
      - manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
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


def _parse_indices(spec: str) -> List[int]:
    s = str(spec).strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        raise ValueError("empty --random-indices")

    out: List[int] = []
    for part in s.split(","):
        p = part.strip()
        # 条件分岐: `not p` を満たす経路を評価する。
        if not p:
            continue

        # 条件分岐: `"-" in p` を満たす経路を評価する。

        if "-" in p:
            a_s, b_s = p.split("-", 1)
            a = int(a_s)
            b = int(b_s)
            # 条件分岐: `b < a` を満たす経路を評価する。
            if b < a:
                raise ValueError(f"invalid range: {p}")

            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(p))
    # stable unique

    seen: set[int] = set()
    uniq: List[int] = []
    for i in out:
        # 条件分岐: `i in seen` を満たす経路を評価する。
        if i in seen:
            continue

        seen.add(i)
        uniq.append(i)

    # 条件分岐: `not uniq` を満たす経路を評価する。

    if not uniq:
        raise ValueError("no indices parsed from --random-indices")

    # 条件分岐: `any(i < 0 for i in uniq)` を満たす経路を評価する。

    if any(i < 0 for i in uniq):
        raise ValueError("random indices must be >=0")

    return uniq


def _relpath(p: Path) -> str:
    try:
        return p.relative_to(_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _load_manifest(data_dir: Path) -> Dict[str, Any]:
    mp = data_dir / "manifest.json"
    # 条件分岐: `not mp.exists()` を満たす経路を評価する。
    if not mp.exists():
        raise FileNotFoundError(f"manifest not found: {mp}")

    return json.loads(mp.read_text(encoding="utf-8"))


def _manifest_item(manifest: Dict[str, Any], *, sample: str, cap: str) -> Dict[str, Any]:
    key = f"{sample}:{cap}"
    it = (manifest.get("items") or {}).get(key)
    # 条件分岐: `not isinstance(it, dict)` を満たす経路を評価する。
    if not isinstance(it, dict):
        raise KeyError(f"missing item {key} in manifest")

    # 条件分岐: `"galaxy" not in it or "random" not in it` を満たす経路を評価する。

    if "galaxy" not in it or "random" not in it:
        raise KeyError(f"missing galaxy/random in manifest item {key}")

    return it


def _infer_combined_random_name(raw_or_npz_name: str, *, idx_tag: str, target_rows: int, seed: int) -> str:
    s = str(raw_or_npz_name)
    # Prefer a stable replacement of the trailing _<idx>_clustering.ran.fits...
    m = re.search(r"_(\d+)(_clustering\.ran\.fits.*)$", s)
    # 条件分岐: `m` を満たす経路を評価する。
    if m:
        base = s[: m.start(1)]
        suffix = m.group(2)
        out = f"{base}{idx_tag}{suffix}.mix_reservoir_{int(target_rows)}_seed{int(seed)}.npz"
        return out
    # Fallback: append tag.

    stem = Path(s).name
    return f"{stem}.mix_reservoir_{int(target_rows)}_seed{int(seed)}_{idx_tag}.npz"


def _combine_random_npz(
    *,
    src_npz_paths: List[Path],
    out_npz_path: Path,
    target_rows: int,
    seed: int,
    take_mode: str,
    shuffle_rows: bool,
) -> Dict[str, Any]:
    # 条件分岐: `not src_npz_paths` を満たす経路を評価する。
    if not src_npz_paths:
        raise ValueError("empty src_npz_paths")

    # 条件分岐: `not (int(target_rows) > 0)` を満たす経路を評価する。

    if not (int(target_rows) > 0):
        raise ValueError("--target-rows must be >0")

    # Determine allocation per index (deterministic; remainder goes to earlier indices).

    n_src = int(len(src_npz_paths))
    n_each = int(target_rows) // n_src
    rem = int(target_rows) - n_each * n_src
    takes = [n_each + (1 if i < rem else 0) for i in range(n_src)]
    assert sum(takes) == int(target_rows)

    # Read first file to learn keys/dtypes/shapes.
    with np.load(src_npz_paths[0]) as z0:
        keys = list(z0.files)
        # 条件分岐: `not keys` を満たす経路を評価する。
        if not keys:
            raise ValueError(f"empty npz: {src_npz_paths[0]}")

        for k in keys:
            a0 = np.asarray(z0[k])
            # 条件分岐: `a0.ndim != 1` を満たす経路を評価する。
            if a0.ndim != 1:
                raise ValueError(f"expected 1D arrays in npz: key={k} shape={a0.shape} ({src_npz_paths[0]})")

        dtypes = {k: np.asarray(z0[k]).dtype for k in keys}

    take_mode = str(take_mode).strip().lower()
    # 条件分岐: `take_mode not in ("random", "prefix")` を満たす経路を評価する。
    if take_mode not in ("random", "prefix"):
        raise ValueError("--take-mode must be random/prefix")

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    out_arrays: Dict[str, np.ndarray] = {k: np.empty(int(target_rows), dtype=dtypes[k]) for k in keys}
    offset = 0
    per_src: List[Dict[str, Any]] = []
    for i, (src_path, take_n) in enumerate(zip(src_npz_paths, takes, strict=True)):
        with np.load(src_path) as z:
            # 条件分岐: `list(z.files) != list(keys)` を満たす経路を評価する。
            if list(z.files) != list(keys):
                raise ValueError(f"npz keys mismatch: {src_path} vs {src_npz_paths[0]}")

            n_avail = int(np.asarray(z[keys[0]]).shape[0])
            # 条件分岐: `n_avail < int(take_n)` を満たす経路を評価する。
            if n_avail < int(take_n):
                raise ValueError(f"npz too small: need {take_n} but have {n_avail}: {src_path}")

            sl = slice(offset, offset + int(take_n))
            # 条件分岐: `take_mode == "prefix"` を満たす経路を評価する。
            if take_mode == "prefix":
                sel = slice(0, int(take_n))
                sel_meta: Dict[str, Any] = {"mode": "prefix", "start": 0, "stop": int(take_n)}
            else:
                rng = np.random.default_rng(int(seed) + int(i))
                sel = rng.choice(int(n_avail), size=int(take_n), replace=False)
                # Sort for slightly more cache-friendly reads; order is irrelevant because we can shuffle later.
                sel.sort()
                sel_meta = {"mode": "random_choice_no_replace", "seed": int(seed) + int(i)}

            for k in keys:
                out_arrays[k][sl] = np.asarray(z[k])[sel]

        per_src.append({"npz_path": _relpath(src_path), "take_rows": int(take_n), "select": sel_meta})
        offset += int(take_n)

    assert offset == int(target_rows)

    # 条件分岐: `shuffle_rows` を満たす経路を評価する。
    if shuffle_rows:
        rng = np.random.default_rng(int(seed))
        perm = np.arange(int(target_rows), dtype=np.int64)
        rng.shuffle(perm)
        for k in keys:
            out_arrays[k] = out_arrays[k][perm]

    np.savez_compressed(out_npz_path, **out_arrays)
    return {
        "target_rows": int(target_rows),
        "seed": int(seed),
        "take_mode": str(take_mode),
        "shuffle_rows": bool(shuffle_rows),
        "per_source": per_src,
        "keys": keys,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-data-dir", type=str, required=True, help="output data dir (e.g., data/cosmology/desi_dr1_lss_reservoir_r0to17_mix)")
    ap.add_argument(
        "--in-dir-template",
        type=str,
        default="data/cosmology/desi_dr1_lss_reservoir_r{idx}",
        help="input data dir template with {idx} (default: data/cosmology/desi_dr1_lss_reservoir_r{idx})",
    )
    ap.add_argument("--random-indices", type=str, default="0-17", help="comma/range list (default: 0-17)")
    ap.add_argument("--sample", type=str, default="lrg", help="tracer (default: lrg)")
    ap.add_argument("--target-rows", type=int, default=2_000_000, help="rows per cap in the combined random (default: 2,000,000)")
    ap.add_argument("--seed", type=int, default=0, help="seed used for optional shuffle (default: 0)")
    ap.add_argument("--take-mode", choices=["random", "prefix"], default="random", help="how to sample rows from each source NPZ (default: random)")
    ap.add_argument("--no-shuffle", action="store_true", help="do not shuffle rows after concatenation")
    args = ap.parse_args(list(argv) if argv is not None else None)

    indices = _parse_indices(str(args.random_indices))
    sample = str(args.sample).strip().lower()
    out_data_dir = _ROOT / Path(str(args.out_data_dir))
    in_template = str(args.in_dir_template)
    target_rows = int(args.target_rows)
    seed = int(args.seed)
    take_mode = str(args.take_mode)
    shuffle_rows = not bool(args.no_shuffle)

    # Load manifests
    in_dirs: List[Tuple[int, Path]] = []
    manifests: Dict[int, Dict[str, Any]] = {}
    for idx in indices:
        d = _ROOT / Path(in_template.format(idx=int(idx)))
        in_dirs.append((int(idx), d))
        manifests[int(idx)] = _load_manifest(d)

    # Use the first index as canonical for galaxy.

    idx0 = int(indices[0])
    m0 = manifests[idx0]

    out_extracted = out_data_dir / "extracted"
    out_extracted.mkdir(parents=True, exist_ok=True)

    # Copy galaxy NPZ from idx0 (north/south).
    galaxy_out: Dict[str, Path] = {}
    for cap in ("north", "south"):
        it0 = _manifest_item(m0, sample=sample, cap=cap)
        gal_npz_src = _ROOT / Path(str(it0["galaxy"]["npz_path"]))
        # 条件分岐: `not gal_npz_src.exists()` を満たす経路を評価する。
        if not gal_npz_src.exists():
            raise FileNotFoundError(f"missing galaxy npz: {gal_npz_src}")

        gal_name = gal_npz_src.name
        gal_npz_dst = out_extracted / gal_name
        shutil.copy2(gal_npz_src, gal_npz_dst)
        galaxy_out[cap] = gal_npz_dst

    # Build combined random NPZ for each cap.

    idx_tag = f"{min(indices)}to{max(indices)}"
    random_out: Dict[str, Path] = {}
    random_meta: Dict[str, Any] = {}
    for cap in ("north", "south"):
        # Gather source random NPZs (one per random index)
        src_npz_paths: List[Path] = []
        raw_name_hint: str | None = None
        for idx in indices:
            mi = manifests[int(idx)]
            iti = _manifest_item(mi, sample=sample, cap=cap)
            rnd_npz_src = _ROOT / Path(str(iti["random"]["npz_path"]))
            # 条件分岐: `not rnd_npz_src.exists()` を満たす経路を評価する。
            if not rnd_npz_src.exists():
                raise FileNotFoundError(f"missing random npz: {rnd_npz_src}")

            src_npz_paths.append(rnd_npz_src)
            # 条件分岐: `raw_name_hint is None` を満たす経路を評価する。
            if raw_name_hint is None:
                raw_path = iti.get("random", {}).get("raw_path")
                raw_name_hint = Path(str(raw_path)).name if raw_path else rnd_npz_src.name

        out_name = _infer_combined_random_name(
            raw_name_hint or src_npz_paths[0].name,
            idx_tag=idx_tag,
            target_rows=target_rows,
            seed=seed,
        )
        out_npz_path = out_extracted / out_name
        meta = _combine_random_npz(
            src_npz_paths=src_npz_paths,
            out_npz_path=out_npz_path,
            target_rows=target_rows,
            seed=seed,
            take_mode=take_mode,
            shuffle_rows=shuffle_rows,
        )
        random_out[cap] = out_npz_path
        random_meta[cap] = meta

    manifest_out: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "dataset": "desi_dr1_lss",
        "note": "Built a combined random(reservoir) from multiple extracted random_index reservoirs.",
        "inputs": {
            "in_dir_template": in_template,
            "random_indices": [int(i) for i in indices],
            "target_rows_per_cap": int(target_rows),
            "seed": int(seed),
            "take_mode": str(take_mode),
            "shuffle_rows": bool(shuffle_rows),
        },
        "items": {},
    }

    # Populate items in the schema expected by cosmology_bao_xi_from_catalogs.py
    for cap in ("north", "south"):
        key = f"{sample}:{cap}"
        manifest_out["items"][key] = {
            "galaxy": {"npz_path": _relpath(galaxy_out[cap])},
            "random": {
                "kind": "random",
                "random_index": -1,
                "random_indices": [int(i) for i in indices],
                "npz_path": _relpath(random_out[cap]),
                "extract": {
                    "sampling": {
                        "method": "mix_reservoir_from_extracted",
                        "target_rows": int(target_rows),
                        "seed": int(seed),
                        "shuffle_rows": bool(shuffle_rows),
                        "source": random_meta[cap].get("per_source", []),
                    }
                },
            },
        }

    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_path = out_data_dir / "manifest.json"
    out_manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "domain": "cosmology",
            "action": "desi_dr1_lss_build_combined_random_reservoir",
            "inputs": [str((_ROOT / Path(in_template.format(idx=int(i))) / 'manifest.json')) for i in indices],
            "outputs": [str(out_manifest_path), str(galaxy_out["north"]), str(galaxy_out["south"]), str(random_out["north"]), str(random_out["south"])],
            "params": manifest_out["inputs"],
        }
    )

    print(f"[ok] wrote: {out_manifest_path}")
    print(f"[ok] random north: {random_out['north']}")
    print(f"[ok] random south: {random_out['south']}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
