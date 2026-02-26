from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class FileSpec:
    url: str
    relpath: str
    expected_bytes: int | None = None


_S3_BUCKET = "nist-belltestdata"
_S3_BASE = f"https://s3.amazonaws.com/{_S3_BUCKET}"
_DEFAULT_DATE = "2015_09_18"
_DEFAULT_RUN_BASE = "03_31_CH_pockel_100kHz.run4.afterTimingfix2_training"
_ALICE_SUFFIX = ".alice.dat.compressed.zip"
_BOB_SUFFIX = ".bob.dat.compressed.zip"
_HDF5_BUILD_SUFFIX = ".dat.compressed.build.hdf5"
_PART_RE = re.compile(r"^(?P<base>.+)\.(?P<side>alice|bob)\.dat\.compressed\.(?P<part>z\d\d)$")


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
            # 条件分岐: `total_i and done and done % (64 * 1024 * 1024) < len(chunk)` を満たす経路を評価する。
            if total_i and done and done % (64 * 1024 * 1024) < len(chunk):
                print(f"  ... {done/total_i:.1%} ({done}/{total_i} bytes)")

    # 条件分岐: `expected_bytes is not None and out_path.stat().st_size != expected_bytes` を満たす経路を評価する。

    if expected_bytes is not None and out_path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"downloaded size mismatch: {out_path} ({out_path.stat().st_size} != {expected_bytes})"
        )

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _s3_list_objects_v2(*, prefix: str) -> list[dict[str, str | int]]:
    out: list[dict[str, str | int]] = []
    token: str | None = None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        # 条件分岐: `token is not None` を満たす経路を評価する。
        if token is not None:
            q["continuation-token"] = token

        url = f"https://{_S3_BUCKET}.s3.amazonaws.com/?" + urlencode(q)
        req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
        with urlopen(req) as resp:
            xml = resp.read()

        root = ET.fromstring(xml)
        for c in root.findall("s3:Contents", ns):
            key = c.findtext("s3:Key", default="", namespaces=ns)
            size_s = c.findtext("s3:Size", default="0", namespaces=ns)
            # 条件分岐: `not key` を満たす経路を評価する。
            if not key:
                continue

            out.append({"key": key, "bytes": int(size_s)})

        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=ns).lower() == "true"
        token = root.findtext("s3:NextContinuationToken", default=None, namespaces=ns)
        # 条件分岐: `not is_truncated` を満たす経路を評価する。
        if not is_truncated:
            break

    return out


def _list_run_bases(
    *, date: str, contains: str | None
) -> tuple[dict[str, int], dict[str, int], set[str], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    # Returns (alice_zip_sizes, bob_zip_sizes, multipart_bases, alice_parts, bob_parts)
    prefix_a = f"belldata/compressed/alice/{date}/"
    prefix_b = f"belldata/compressed/bob/{date}/"

    alice: dict[str, int] = {}
    bob: dict[str, int] = {}
    multipart: set[str] = set()
    alice_parts: dict[str, dict[str, int]] = {}
    bob_parts: dict[str, dict[str, int]] = {}

    for rec in _s3_list_objects_v2(prefix=prefix_a):
        key = str(rec["key"])
        # 条件分岐: `not key.startswith(prefix_a)` を満たす経路を評価する。
        if not key.startswith(prefix_a):
            continue

        rel = key[len(prefix_a) :]
        # 条件分岐: `contains and contains not in rel` を満たす経路を評価する。
        if contains and contains not in rel:
            continue

        # 条件分岐: `rel.endswith(_ALICE_SUFFIX)` を満たす経路を評価する。

        if rel.endswith(_ALICE_SUFFIX):
            base = rel[: -len(_ALICE_SUFFIX)]
            alice[base] = int(rec["bytes"])
            continue

        m = _PART_RE.match(rel)
        # 条件分岐: `m and m.group("side") == "alice"` を満たす経路を評価する。
        if m and m.group("side") == "alice":
            base = m.group("base")
            part = m.group("part")
            multipart.add(base)
            alice_parts.setdefault(base, {})[part] = int(rec["bytes"])

    for rec in _s3_list_objects_v2(prefix=prefix_b):
        key = str(rec["key"])
        # 条件分岐: `not key.startswith(prefix_b)` を満たす経路を評価する。
        if not key.startswith(prefix_b):
            continue

        rel = key[len(prefix_b) :]
        # 条件分岐: `contains and contains not in rel` を満たす経路を評価する。
        if contains and contains not in rel:
            continue

        # 条件分岐: `rel.endswith(_BOB_SUFFIX)` を満たす経路を評価する。

        if rel.endswith(_BOB_SUFFIX):
            base = rel[: -len(_BOB_SUFFIX)]
            bob[base] = int(rec["bytes"])
            continue

        m = _PART_RE.match(rel)
        # 条件分岐: `m and m.group("side") == "bob"` を満たす経路を評価する。
        if m and m.group("side") == "bob":
            base = m.group("base")
            part = m.group("part")
            multipart.add(base)
            bob_parts.setdefault(base, {})[part] = int(rec["bytes"])

    return alice, bob, multipart, alice_parts, bob_parts


def _list_hdf5_builds(*, date: str, contains: str | None) -> dict[str, int]:
    prefix = f"belldata/processed_compressed/hdf5/{date}/"
    sizes: dict[str, int] = {}
    for rec in _s3_list_objects_v2(prefix=prefix):
        key = str(rec["key"])
        # 条件分岐: `not key.startswith(prefix)` を満たす経路を評価する。
        if not key.startswith(prefix):
            continue

        rel = key[len(prefix) :]
        # 条件分岐: `contains and contains not in rel` を満たす経路を評価する。
        if contains and contains not in rel:
            continue

        # 条件分岐: `not rel.endswith(_HDF5_BUILD_SUFFIX)` を満たす経路を評価する。

        if not rel.endswith(_HDF5_BUILD_SUFFIX):
            continue

        base = rel[: -len(_HDF5_BUILD_SUFFIX)]
        sizes[base] = int(rec["bytes"])

    return sizes


def _format_gib(n_bytes: int) -> str:
    return f"{n_bytes / (1024**3):.2f} GiB"


def _spec_for_run(*, side: str, date: str, run_base: str, expected_bytes: int | None) -> FileSpec:
    filename = f"{run_base}.{side}.dat.compressed.zip"
    relpath = f"compressed/{side}/{date}/{filename}"
    url = f"{_S3_BASE}/belldata/{relpath}"
    return FileSpec(url=url, relpath=relpath, expected_bytes=expected_bytes)


def _spec_for_run_part(*, side: str, date: str, run_base: str, part: str, expected_bytes: int | None) -> FileSpec:
    filename = f"{run_base}.{side}.dat.compressed.{part}"
    relpath = f"compressed/{side}/{date}/{filename}"
    url = f"{_S3_BASE}/belldata/{relpath}"
    return FileSpec(url=url, relpath=relpath, expected_bytes=expected_bytes)


def _spec_for_hdf5_build(*, date: str, run_base: str, expected_bytes: int | None) -> FileSpec:
    filename = f"{run_base}{_HDF5_BUILD_SUFFIX}"
    relpath = f"processed_compressed/hdf5/{date}/{filename}"
    url = f"{_S3_BASE}/belldata/{relpath}"
    return FileSpec(url=url, relpath=relpath, expected_bytes=expected_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch NIST Bell test time-tag data (cached under data/quantum/sources/)."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download; only verify expected files exist.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write manifest.json with url/path/size/sha256 (default: on when downloading).",
    )
    parser.add_argument(
        "--date",
        default=_DEFAULT_DATE,
        help=f'Data date under belldata/compressed/<side>/ (default: "{_DEFAULT_DATE}")',
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help=(
            "Run base name to fetch (repeatable). "
            f'Default: "{_DEFAULT_RUN_BASE}". Use --list to discover candidates.'
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available run base names for the given --date and exit.",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Optional substring filter when listing runs (matches the filename).",
    )
    parser.add_argument(
        "--list-hdf5",
        action="store_true",
        help="List available processed_compressed/hdf5 build files for the given --date and exit.",
    )
    parser.add_argument(
        "--hdf5",
        action="store_true",
        help="Also fetch processed_compressed/hdf5 build file for each selected --run.",
    )
    parser.add_argument(
        "--max-gib",
        type=float,
        default=0.6,
        help="Safety limit: refuse to download any single file larger than this size (default: 0.6 GiB).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / "nist_belltestdata"
    src_dir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `args.list` を満たす経路を評価する。
    if args.list:
        alice, bob, multipart, alice_parts, bob_parts = _list_run_bases(date=args.date, contains=args.filter)
        common = sorted(set(alice).intersection(bob))
        # 条件分岐: `not common` を満たす経路を評価する。
        if not common:
            raise SystemExit("[fail] no matching alice/bob runs found (check --date/--filter)")

        print(f"[info] date={args.date} common_runs={len(common)}")
        for base in common:
            a = alice.get(base)
            b = bob.get(base)
            a_total = int(a or 0) + int(sum((alice_parts.get(base) or {}).values()))
            b_total = int(b or 0) + int(sum((bob_parts.get(base) or {}).values()))
            note = " [multipart]" if base in multipart else ""
            # 条件分岐: `a is None or b is None` を満たす経路を評価する。
            if a is None or b is None:
                continue

            print(
                f"- {base}{note} | alice={_format_gib(a_total)} bob={_format_gib(b_total)} total={_format_gib(a_total+b_total)}"
            )

        return

    # 条件分岐: `args.list_hdf5` を満たす経路を評価する。

    if args.list_hdf5:
        hdf5 = _list_hdf5_builds(date=args.date, contains=args.filter)
        bases = sorted(hdf5)
        # 条件分岐: `not bases` を満たす経路を評価する。
        if not bases:
            raise SystemExit("[fail] no matching hdf5 build files found (check --date/--filter)")

        print(f"[info] date={args.date} hdf5_builds={len(bases)}")
        for base in bases:
            print(f"- {base} | hdf5={_format_gib(int(hdf5[base]))}")

        return

    run_bases = args.run or [_DEFAULT_RUN_BASE]

    # Default run: 2015-09-18 training set is small enough to handle on Windows.
    # Note: This is explicitly documented as "bad" (no mode-lock) by NIST, but is
    # suitable to validate the reanalysis pipeline and check time-tag dependencies.
    files: list[FileSpec] = [
        FileSpec(
            url=f"{_S3_BASE}/belldata/File_Folder_Descriptions.pdf",
            relpath="File_Folder_Descriptions.pdf",
            expected_bytes=397_922,
        ),
        FileSpec(
            url=f"{_S3_BASE}/belldata/File_Folder_Descriptions_Addendum_2017_02.pdf",
            relpath="File_Folder_Descriptions_Addendum_2017_02.pdf",
            expected_bytes=49_394,
        ),
    ]

    # For non-default runs, resolve sizes via S3 listing (so we can store expected_bytes in the manifest).
    size_map_a: dict[str, int] = {}
    size_map_b: dict[str, int] = {}
    size_map_hdf5: dict[str, int] = {}
    multipart: set[str] = set()
    parts_a: dict[str, dict[str, int]] = {}
    parts_b: dict[str, dict[str, int]] = {}
    # 条件分岐: `any(rb != _DEFAULT_RUN_BASE or args.date != _DEFAULT_DATE for rb in run_bases)` を満たす経路を評価する。
    if any(rb != _DEFAULT_RUN_BASE or args.date != _DEFAULT_DATE for rb in run_bases):
        size_map_a, size_map_b, multipart, parts_a, parts_b = _list_run_bases(date=args.date, contains=None)
        # 条件分岐: `args.hdf5` を満たす経路を評価する。
        if args.hdf5:
            size_map_hdf5 = _list_hdf5_builds(date=args.date, contains=None)

    # For the default run we keep the historical expected_bytes constants, even if --date changes.

    default_bytes_alice = 215_879_052
    default_bytes_bob = 226_961_414

    max_bytes = int(args.max_gib * (1024**3))

    for run_base in run_bases:
        parts_for_a = parts_a.get(run_base) or {}
        parts_for_b = parts_b.get(run_base) or {}
        # 条件分岐: `run_base in multipart and (not parts_for_a or not parts_for_b)` を満たす経路を評価する。
        if run_base in multipart and (not parts_for_a or not parts_for_b):
            raise SystemExit(
                f"[fail] run appears to be multipart, but part listing is incomplete: {run_base}\n"
                "Try re-running with --list to confirm available parts for both alice/bob."
            )

        # 条件分岐: `run_base == _DEFAULT_RUN_BASE and args.date == _DEFAULT_DATE` を満たす経路を評価する。

        if run_base == _DEFAULT_RUN_BASE and args.date == _DEFAULT_DATE:
            exp_a = default_bytes_alice
            exp_b = default_bytes_bob
        else:
            exp_a = size_map_a.get(run_base)
            exp_b = size_map_b.get(run_base)
            # 条件分岐: `exp_a is None or exp_b is None` を満たす経路を評価する。
            if exp_a is None or exp_b is None:
                hint = ' (tip: run with --list to discover valid "run base" names)'
                raise SystemExit(f"[fail] run not found for both sides: {run_base}{hint}")

        # 条件分岐: `not args.offline` を満たす経路を評価する。

        if not args.offline:
            for side, exp in (("alice", exp_a), ("bob", exp_b)):
                # 条件分岐: `exp is not None and exp > max_bytes` を満たす経路を評価する。
                if exp is not None and exp > max_bytes:
                    raise SystemExit(
                        f"[fail] refusing to download large file ({_format_gib(exp)} > max={args.max_gib} GiB):\n"
                        f"  run={run_base} side={side}\n"
                        "Re-run with a larger --max-gib if you really want this download."
                    )

        files.append(_spec_for_run(side="alice", date=args.date, run_base=run_base, expected_bytes=exp_a))
        files.append(_spec_for_run(side="bob", date=args.date, run_base=run_base, expected_bytes=exp_b))

        # Multipart runs have additional ".z01/.z02..." parts per side.
        # We fetch them alongside the final ".zip" so downstream scripts can reconstruct the full archive.
        for part, exp in sorted(parts_for_a.items()):
            # 条件分岐: `not args.offline and exp > max_bytes` を満たす経路を評価する。
            if not args.offline and exp > max_bytes:
                raise SystemExit(
                    f"[fail] refusing to download large file ({_format_gib(exp)} > max={args.max_gib} GiB):\n"
                    f"  run={run_base} side=alice part={part}\n"
                    "Re-run with a larger --max-gib if you really want this download."
                )

            files.append(_spec_for_run_part(side="alice", date=args.date, run_base=run_base, part=part, expected_bytes=exp))

        for part, exp in sorted(parts_for_b.items()):
            # 条件分岐: `not args.offline and exp > max_bytes` を満たす経路を評価する。
            if not args.offline and exp > max_bytes:
                raise SystemExit(
                    f"[fail] refusing to download large file ({_format_gib(exp)} > max={args.max_gib} GiB):\n"
                    f"  run={run_base} side=bob part={part}\n"
                    "Re-run with a larger --max-gib if you really want this download."
                )

            files.append(_spec_for_run_part(side="bob", date=args.date, run_base=run_base, part=part, expected_bytes=exp))

        # 条件分岐: `args.hdf5` を満たす経路を評価する。

        if args.hdf5:
            exp_h = size_map_hdf5.get(run_base)
            # 条件分岐: `exp_h is None` を満たす経路を評価する。
            if exp_h is None:
                hint = ' (tip: run with --list-hdf5 to discover available build files)'
                raise SystemExit(f"[fail] hdf5 build not found: {run_base}{hint}")

            # 条件分岐: `not args.offline and exp_h > max_bytes` を満たす経路を評価する。

            if not args.offline and exp_h > max_bytes:
                raise SystemExit(
                    f"[fail] refusing to download large file ({_format_gib(exp_h)} > max={args.max_gib} GiB):\n"
                    f"  hdf5_build run={run_base}\n"
                    "Re-run with a larger --max-gib if you really want this download."
                )

            files.append(_spec_for_hdf5_build(date=args.date, run_base=run_base, expected_bytes=exp_h))

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

    # Default behavior: write/refresh manifest after download (or when files were already cached).

    write_manifest = bool(args.write_manifest) or True
    # 条件分岐: `write_manifest` を満たす経路を評価する。
    if write_manifest:
        out = src_dir / "manifest.json"
        existing = None
        # 条件分岐: `out.exists()` を満たす経路を評価する。
        if out.exists():
            try:
                existing = json.loads(out.read_text(encoding="utf-8"))
            except Exception:
                existing = None

        manifest = {
            "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "dataset": "nist-belltestdata (Shalm et al. 2015 time-tag repository)",
            "date": str(args.date),
            "runs": [],
            "files": [],
        }

        # 条件分岐: `isinstance(existing, dict)` を満たす経路を評価する。
        if isinstance(existing, dict):
            # 条件分岐: `str(existing.get("date", "")) == str(args.date)` を満たす経路を評価する。
            if str(existing.get("date", "")) == str(args.date):
                # Merge runs/files across multiple invocations to keep a stable cache record.
                manifest["runs"] = list(existing.get("runs", []))
                manifest["files"] = list(existing.get("files", []))

        # Merge runs.

        run_set = {str(r) for r in manifest["runs"]} | {str(r) for r in run_bases}
        manifest["runs"] = sorted(run_set)

        # Merge files by URL (most stable key); newest record wins.
        files_by_url: dict[str, dict[str, object]] = {}
        for rec in manifest["files"]:
            # 条件分岐: `not isinstance(rec, dict)` を満たす経路を評価する。
            if not isinstance(rec, dict):
                continue

            url = rec.get("url")
            # 条件分岐: `isinstance(url, str) and url` を満たす経路を評価する。
            if isinstance(url, str) and url:
                files_by_url[url] = rec

        for spec in files:
            path = src_dir / spec.relpath
            entry = {
                "url": spec.url,
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            files_by_url[spec.url] = entry

        manifest["files"] = [files_by_url[k] for k in sorted(files_by_url.keys())]

        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] manifest: {out}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
