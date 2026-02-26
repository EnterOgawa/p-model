from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class FileSpec:
    url: str
    relpath: str
    expected_bytes: int | None = None


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch primary sources for Phase 7 / Step 7.8 (vacuum/QED precision: Casimir, Lamb shift, H 1S–2S, alpha from recoil/g-2)."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download; only verify expected files exist.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)

    files: list[FileSpec] = [
        # Casimir force measurement (AFM; sphere-plane).
        FileSpec(
            url="https://arxiv.org/pdf/quant-ph/9906062v3.pdf",
            relpath="arxiv_quant-ph_9906062v3.pdf",
        ),
        # Lamb shift (light hydrogen-like ions; definitions + scaling).
        FileSpec(
            url="https://arxiv.org/pdf/physics/0009069v1.pdf",
            relpath="arxiv_physics_0009069v1.pdf",
        ),
        # Hydrogen 1S–2S transition frequency (precision benchmark; uncertainty budget).
        FileSpec(
            url="https://arxiv.org/pdf/1107.3101v1.pdf",
            relpath="arxiv_1107.3101v1.pdf",
        ),
        # Fine-structure constant from recoil (atom interferometry).
        FileSpec(
            url="https://arxiv.org/pdf/0812.3139v1.pdf",
            relpath="arxiv_0812.3139v1.pdf",
        ),
        # Fine-structure constant from electron g-2 (QED).
        FileSpec(
            url="https://arxiv.org/pdf/0801.1134v2.pdf",
            relpath="arxiv_0801.1134v2.pdf",
        ),
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
        "dataset": "Phase 7 / Step 7.8 QED/vacuum primary sources (Casimir, Lamb shift, H 1S–2S, alpha from recoil/g-2)",
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

    out = src_dir / "qed_vacuum_precision_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest: {out}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
