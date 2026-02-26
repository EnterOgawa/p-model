from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


# クラス: `FileSpec` の責務と境界条件を定義する。
@dataclass(frozen=True)
class FileSpec:
    url: str
    relpath: str


# 関数: `_utc_now` の入出力契約と処理意図を定義する。

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            # 条件分岐: `not chunk` を満たす経路を評価する。
            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0` を満たす経路を評価する。
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return

    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req, timeout=60) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    # 条件分岐: `out_path.stat().st_size == 0` を満たす経路を評価する。

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch Particle Data Group (PDG) Monte Carlo particle masses/widths table "
            "(RPP 2024 edition) and write a sha256 manifest."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / "pdg_rpp_2024_mass_width"
    src_dir.mkdir(parents=True, exist_ok=True)

    files = [
        FileSpec(
            url="https://pdg.lbl.gov/2024/mcdata/mass_width_2024.txt",
            relpath="mass_width_2024.txt",
        )
    ]

    missing: list[Path] = []
    for spec in files:
        path = src_dir / spec.relpath
        # 条件分岐: `not args.offline` を満たす経路を評価する。
        if not args.offline:
            _download(spec.url, path)

        # 条件分岐: `not path.exists() or path.stat().st_size == 0` を満たす経路を評価する。

        if not path.exists() or path.stat().st_size == 0:
            missing.append(path)

    # 条件分岐: `missing` を満たす経路を評価する。

    if missing:
        raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

    manifest = {
        "generated_utc": _utc_now(),
        "dataset": "PDG RPP 2024 Monte Carlo table: masses/widths/PDG IDs",
        "source": {
            "publisher": "Particle Data Group (Berkeley)",
            "edition": "Review of Particle Physics (RPP) 2024",
            "notes": [
                "This file is intended for Monte Carlo usage and contains PDG-style particle ID numbers and masses/widths.",
                "We cache it to make hadron-scale baselines reproducible offline (Phase 7 / Step 7.13).",
            ],
        },
        "files": [],
    }

    for spec in files:
        path = src_dir / spec.relpath
        manifest["files"].append(
            {"url": spec.url, "path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path)}
        )

    out = src_dir / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest: {out}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

