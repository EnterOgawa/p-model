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
        description="Fetch primary sources for Phase 7 / Step 7.7 (single-photon interference, HOM, squeezed light)."
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

    zenodo_dir = src_dir / "zenodo_6371310"
    zenodo_dir.mkdir(parents=True, exist_ok=True)

    files: list[FileSpec] = [
        # Single-photon interference over 150 km (Mach–Zehnder).
        FileSpec(
            url="https://arxiv.org/pdf/quant-ph/0403104v2.pdf",
            relpath="arxiv_quant-ph_0403104v2.pdf",
        ),
        # Quantum interference of identical photons from remote GaAs quantum dots (HOM-type).
        FileSpec(
            url="https://arxiv.org/pdf/2106.03871v2.pdf",
            relpath="arxiv_2106.03871v2.pdf",
        ),
        # 10 dB squeezed light (benchmark for squeezing level).
        FileSpec(
            url="https://arxiv.org/pdf/0706.1431v1.pdf",
            relpath="arxiv_0706.1431v1.pdf",
        ),
        # Zenodo: Quantum interference of identical photons from remote GaAs quantum dots (raw data by figure).
        FileSpec(
            url="https://zenodo.org/api/records/6371310/files/DataFig1.zip/content",
            relpath=str(Path("zenodo_6371310") / "DataFig1.zip"),
            expected_bytes=460_129,
        ),
        FileSpec(
            url="https://zenodo.org/api/records/6371310/files/DataFig2.zip/content",
            relpath=str(Path("zenodo_6371310") / "DataFig2.zip"),
            expected_bytes=501_721,
        ),
        FileSpec(
            url="https://zenodo.org/api/records/6371310/files/DataFig3.zip/content",
            relpath=str(Path("zenodo_6371310") / "DataFig3.zip"),
            expected_bytes=13_839,
        ),
        FileSpec(
            url="https://zenodo.org/api/records/6371310/files/DataExFig1.zip/content",
            relpath=str(Path("zenodo_6371310") / "DataExFig1.zip"),
            expected_bytes=3_479_458,
        ),
        FileSpec(
            url="https://zenodo.org/api/records/6371310/files/DataExfig2.zip/content",
            relpath=str(Path("zenodo_6371310") / "DataExfig2.zip"),
            expected_bytes=33_035,
        ),
        FileSpec(
            url="https://zenodo.org/api/records/6371310/files/DataExfig3.zip/content",
            relpath=str(Path("zenodo_6371310") / "DataExfig3.zip"),
            expected_bytes=27_163_866,
        ),
        FileSpec(
            url="https://zenodo.org/api/records/6371310/files/DataExFig4.zip/content",
            relpath=str(Path("zenodo_6371310") / "DataExFig4.zip"),
            expected_bytes=27_021,
        ),
        FileSpec(
            url="https://zenodo.org/api/records/6371310/files/DataExFig5.zip/content",
            relpath=str(Path("zenodo_6371310") / "DataExFig5.zip"),
            expected_bytes=84_001,
        ),
        FileSpec(
            url="https://zenodo.org/api/records/6371310/files/DataExFig6.zip/content",
            relpath=str(Path("zenodo_6371310") / "DataExFig6.zip"),
            expected_bytes=7_244,
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
        "dataset": "Phase 7 / Step 7.7 photon interference primary sources (arXiv PDFs + Zenodo raw data)",
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

    out = zenodo_dir / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest: {out}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
