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
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, out_path: Path, *, expected_bytes: int | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
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
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total_i and done and done % (32 * 1024 * 1024) < len(chunk):
                print(f"  ... {done/total_i:.1%} ({done}/{total_i} bytes)")

    if expected_bytes is not None and out_path.stat().st_size != expected_bytes:
        raise RuntimeError(f"downloaded size mismatch: {out_path} ({out_path.stat().st_size} != {expected_bytes})")
    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Big Bell Test 2018 (Nature) supplementary + source data (human-choice inputs; Figure 2)."
    )
    parser.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / "big_bell_test_2018"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Nature article: https://www.nature.com/articles/s41586-018-0085-3
    files: list[FileSpec] = [
        FileSpec(
            url="https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0085-3/MediaObjects/41586_2018_85_MOESM1_ESM.pdf",
            relpath="supplementary_info.pdf",
            expected_bytes=8_618_782,
        ),
        FileSpec(
            url="https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0085-3/MediaObjects/41586_2018_85_MOESM2_ESM.xlsx",
            relpath="source_data.xlsx",
            expected_bytes=9_497_057,
        ),
    ]

    missing: list[Path] = []
    for spec in files:
        path = src_dir / spec.relpath
        if args.offline:
            if not path.exists():
                missing.append(path)
            continue
        _download(spec.url, path, expected_bytes=spec.expected_bytes)

    if args.offline:
        if missing:
            raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))
        print("[ok] offline check passed")
        return

    manifest = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Big Bell Test 2018 (Nature): supplementary + source data (human-choice inputs; Figure 2)",
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


if __name__ == "__main__":
    main()

