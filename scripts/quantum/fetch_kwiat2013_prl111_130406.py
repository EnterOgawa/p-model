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


def _download(url: str, out_path: Path, *, expected_bytes: int | None, max_bytes: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        if expected_bytes is None or out_path.stat().st_size == expected_bytes:
            print(f"[skip] exists: {out_path}")
            return
        print(f"[redo] size mismatch: {out_path} ({out_path.stat().st_size} != {expected_bytes})")
        out_path.unlink()

    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req) as resp:
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total is not None else None
        if total_i is not None and total_i > max_bytes:
            raise SystemExit(
                f"[fail] refusing to download large file ({total_i} bytes > max={max_bytes} bytes):\n"
                f"  {url}\n"
                "Re-run with a larger --max-gib if you really want this download."
            )
        with out_path.open("wb") as f:
            done = 0
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total_i and done and done % (128 * 1024 * 1024) < len(chunk):
                    print(f"  ... {done/total_i:.1%} ({done}/{total_i} bytes)")

    if expected_bytes is not None and out_path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"downloaded size mismatch: {out_path} ({out_path.stat().st_size} != {expected_bytes})"
        )
    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")
    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _format_gib(x: int) -> str:
    return f"{x/(1024**3):.3f} GiB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch Christensen et al. 2013 (PRL 111, 130406; Kwiat group) CH Bell-test time-tag data "
            "from the Illinois QI page (CH_Bell_Data.zip + data_organization.txt)."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--max-gib",
        type=float,
        default=1.0,
        help="Refuse single-file downloads larger than this (GiB). Default: 1.0",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / "kwiat2013_prl111_130406"
    src_dir.mkdir(parents=True, exist_ok=True)

    files: list[FileSpec] = [
        FileSpec(
            url="https://research.physics.illinois.edu/QI/BellTest/data_organization.txt",
            relpath="data_organization.txt",
        ),
        FileSpec(
            url="https://research.physics.illinois.edu/QI/BellTest/CH_Bell_Data.zip",
            relpath="CH_Bell_Data.zip",
        ),
    ]

    max_bytes = int(float(args.max_gib) * (1024**3))
    if args.offline:
        missing: list[Path] = []
        for spec in files:
            path = src_dir / spec.relpath
            if not path.exists() or path.stat().st_size == 0:
                missing.append(path)
        if missing:
            raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))
        print("[ok] offline check passed")
        return

    for spec in files:
        if spec.expected_bytes is not None and spec.expected_bytes > max_bytes:
            raise SystemExit(
                f"[fail] refusing to download large file ({_format_gib(spec.expected_bytes)} > max={args.max_gib} GiB):\n"
                f"  {spec.relpath}\n"
                "Re-run with a larger --max-gib if you really want this download."
            )
        _download(spec.url, src_dir / spec.relpath, expected_bytes=spec.expected_bytes, max_bytes=max_bytes)

    manifest = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Christensen et al. 2013 (PRL 111, 130406) CH Bell-test time-tag data (Illinois QI page)",
        "paper": {
            "journal": "Phys. Rev. Lett.",
            "volume": 111,
            "page": 130406,
            "year": 2013,
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


if __name__ == "__main__":
    main()

