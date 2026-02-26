from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


# クラス: `FileSpec` の責務と境界条件を定義する。
@dataclass(frozen=True)
class FileSpec:
    url: str
    relpath: str
    expected_bytes: int | None = None


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, out_path: Path, *, expected_bytes: int | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0` を満たす経路を評価する。
    if out_path.exists() and out_path.stat().st_size > 0:
        # 条件分岐: `expected_bytes is None or out_path.stat().st_size == expected_bytes` を満たす経路を評価する。
        if expected_bytes is None or out_path.stat().st_size == expected_bytes:
            print(f"[skip] exists: {out_path}")
            return

        print(f"[redo] size mismatch: {out_path} ({out_path.stat().st_size} != {expected_bytes})")
        out_path.unlink()

    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req) as resp, out_path.open("wb") as f:
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total is not None else None
        done = 0
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            # 条件分岐: `not chunk` を満たす経路を評価する。
            if not chunk:
                break

            f.write(chunk)
            done += len(chunk)
            # 条件分岐: `total_i and done and done % (32 * 1024 * 1024) < len(chunk)` を満たす経路を評価する。
            if total_i and done and done % (32 * 1024 * 1024) < len(chunk):
                print(f"  ... {done/total_i:.1%} ({done}/{total_i} bytes)")

    # 条件分岐: `expected_bytes is not None and out_path.stat().st_size != expected_bytes` を満たす経路を評価する。

    if expected_bytes is not None and out_path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"downloaded size mismatch: {out_path} ({out_path.stat().st_size} != {expected_bytes})"
        )

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Delft (Hensen et al. 2015) loophole-free Bell test open data (trial table)."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download; only verify expected files exist.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / "delft_hensen2015"
    src_dir.mkdir(parents=True, exist_ok=True)

    # 4TU.ResearchData open file (data.zip). Landing page:
    # https://data.4tu.nl/datasets/e8cf2991-3153-48ad-b67d-dfd7d7d97fd3
    files: list[FileSpec] = [
        FileSpec(
            url="https://data.4tu.nl/file/e8cf2991-3153-48ad-b67d-dfd7d7d97fd3/289c4850-6ed5-45a3-8b19-de8671f873a8",
            relpath="data.zip",
            expected_bytes=152_913,
        )
    ]

    missing: list[Path] = []
    for spec in files:
        path = src_dir / spec.relpath
        # 条件分岐: `args.offline` を満たす経路を評価する。
        if args.offline:
            # 条件分岐: `not path.exists()` を満たす経路を評価する。
            if not path.exists():
                missing.append(path)

            continue

        _download(spec.url, path, expected_bytes=spec.expected_bytes)

    # 条件分岐: `args.offline` を満たす経路を評価する。

    if args.offline:
        # 条件分岐: `missing` を満たす経路を評価する。
        if missing:
            raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

        print("[ok] offline check passed")
        return

    manifest = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Delft loophole-free Bell test (Hensen et al. 2015) open data (trial table)",
        "files": [],
    }
    for spec in files:
        path = src_dir / spec.relpath
        manifest["files"].append(
            {
                "url": spec.url,
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )

    out = src_dir / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest: {out}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

