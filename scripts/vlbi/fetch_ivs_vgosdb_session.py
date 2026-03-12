#!/usr/bin/env python3
"""
fetch_ivs_vgosdb_session.py

IVS vgosDb セッションの一次アーカイブを取得し、再利用可能な形で固定する。

既定の配布元:
- CDDIS: https://cddis.nasa.gov/archive/vlbi/ivsdata/vgosdb/

注記:
- CDDIS 側の構成や認証要件は時期で変わる可能性があるため、
  実運用では --source-url でセッションURLを明示指定する。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_BASE_URL = "https://cddis.nasa.gov/archive/vlbi/ivsdata/vgosdb/"


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# 関数: `_detect_archive_type` の入出力契約と処理意図を定義する。

def _detect_archive_type(path: Path) -> str:
    if tarfile.is_tarfile(str(path)):
        return "tar"

    if zipfile.is_zipfile(str(path)):
        return "zip"

    head = b""
    try:
        head = path.read_bytes()[:256]
    except Exception:
        pass

    preview = head.decode("utf-8", errors="ignore").strip().lower()
    if preview.startswith("<!doctype html") or preview.startswith("<html"):
        raise RuntimeError(
            "downloaded file looks like HTML (likely auth/login page); "
            "check --cookie-jar and ~/.netrc for CDDIS credentials."
        )

    raise RuntimeError(f"downloaded file is not a supported archive: {path}")


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, dst: Path, *, force: bool, timeout_s: int = 180) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "waveP-vlbi-fetch/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    tmp.replace(dst)


# 関数: `_download_with_curl` の入出力契約と処理意図を定義する。

def _download_with_curl(
    url: str,
    dst: Path,
    *,
    force: bool,
    cookie_jar: Path,
    curl_bin: str = "curl",
    use_netrc: bool = True,
    timeout_s: int = 300,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cookie_jar.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    cmd = [str(curl_bin)]
    if use_netrc:
        cmd.append("-n")

    cmd.extend(
        [
            "-L",
            "-b",
            str(cookie_jar),
            "-c",
            str(cookie_jar),
            "-o",
            str(tmp),
            str(url),
        ]
    )
    cp = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_s)
    if cp.returncode != 0:
        stderr = (cp.stderr or "").strip()
        stdout = (cp.stdout or "").strip()
        detail = stderr if stderr else stdout
        raise RuntimeError(f"curl download failed (rc={cp.returncode}): {detail}")

    tmp.replace(dst)


# 関数: `_fetch_listing_html` の入出力契約と処理意図を定義する。

def _fetch_listing_html(url: str, timeout_s: int = 120) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "waveP-vlbi-fetch/1.0",
            "Accept": "text/html,application/xhtml+xml,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.read().decode("utf-8", errors="ignore")


# 関数: `_extract_links` の入出力契約と処理意図を定義する。

def _extract_links(base_url: str, html: str) -> List[str]:
    links: List[str] = []
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
        href = str(m.group(1)).strip()
        if not href:
            continue

        if href.startswith("#"):
            continue

        if href in {"./", "../"}:
            continue

        full = urllib.parse.urljoin(base_url, href)
        full = full.split("#", 1)[0].split("?", 1)[0]
        links.append(full)

    seen = set()
    uniq: List[str] = []
    for u in links:
        if u in seen:
            continue

        seen.add(u)
        uniq.append(u)

    return uniq


# 関数: `_is_archive_url` の入出力契約と処理意図を定義する。

def _is_archive_url(url: str) -> bool:
    low = url.lower()
    return low.endswith(".tgz") or low.endswith(".tar.gz") or low.endswith(".zip")


# 関数: `_discover_session_url` の入出力契約と処理意図を定義する。

def _discover_session_url(base_url: str, session: str, max_depth: int) -> Optional[str]:
    session_u = session.upper()
    root = str(base_url).rstrip("/") + "/"
    queue: List[tuple[str, int]] = [(root, 0)]
    visited = set()
    while queue:
        current, depth = queue.pop(0)
        if current in visited:
            continue

        visited.add(current)
        try:
            html = _fetch_listing_html(current)
        except Exception:
            continue

        links = _extract_links(current, html)
        for link in links:
            if _is_archive_url(link) and session_u in link.upper():
                return link

        if depth >= max_depth:
            continue

        for link in links:
            if not link.endswith("/"):
                continue

            tail = link.rstrip("/").split("/")[-1]
            cond_year = bool(re.fullmatch(r"\d{4}", tail))
            cond_session = session_u in link.upper()
            if cond_year or cond_session:
                queue.append((link, depth + 1))

    return None


# 関数: `_extract_archive` の入出力契約と処理意図を定義する。

def _extract_archive(archive: Path, dst_dir: Path, *, force: bool) -> Dict[str, object]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[str] = []
    if force:
        for p in dst_dir.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    archive_type = _detect_archive_type(archive)
    if archive_type == "tar":
        with tarfile.open(str(archive), mode="r:*") as tf:
            members = tf.getmembers()
            tf.extractall(str(dst_dir))
            extracted = [m.name for m in members]

        return {"archive_type": "tar", "member_count": int(len(extracted)), "members_sample": extracted[:20]}

    if archive_type == "zip":
        with zipfile.ZipFile(str(archive), mode="r") as zf:
            names = zf.namelist()
            zf.extractall(str(dst_dir))
            extracted = list(names)

        return {"archive_type": "zip", "member_count": int(len(extracted)), "members_sample": extracted[:20]}

    raise RuntimeError(f"unsupported archive format: {archive}")


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, payload: Dict[str, object]) -> None:
    rows = [
        ("session", str(payload.get("session") or "")),
        ("source_url", str(payload.get("source_url") or "")),
        ("archive_path", str(payload.get("archive_path") or "")),
        ("archive_size_bytes", str(int(payload.get("archive_size_bytes") or 0))),
        ("archive_sha256", str(payload.get("archive_sha256") or "")),
        ("extracted_root", str(payload.get("extracted_root") or "")),
        ("archive_type", str(payload.get("archive_type") or "")),
        ("member_count", str(int(payload.get("member_count") or 0))),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in rows:
            w.writerow([k, v])


# 関数: `_sync_public` の入出力契約と処理意図を定義する。

def _sync_public(root: Path, outputs: List[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for p in outputs:
        if p.exists():
            shutil.copy2(p, dst / p.name)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Fetch IVS vgosDb session archive and extract to data/vlbi/sources.")
    ap.add_argument("--session", type=str, default="AUA020", help="Session code, e.g. AUA020")
    ap.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL for vgosDb archive root.")
    ap.add_argument("--source-url", type=str, default="", help="Direct archive URL (recommended).")
    ap.add_argument("--archive-name", type=str, default="", help="Archive filename when source-url is not provided.")
    ap.add_argument(
        "--input-archive",
        type=Path,
        default=None,
        help="Use already downloaded local archive instead of network.",
    )
    ap.add_argument("--force", action="store_true", help="Re-download/re-extract even if cache exists.")
    ap.add_argument("--no-extract", action="store_true", help="Download only, skip extraction.")
    ap.add_argument(
        "--max-discovery-depth",
        type=int,
        default=3,
        help="Directory traversal depth for auto discovery under base-url.",
    )
    ap.add_argument(
        "--download-method",
        type=str,
        default="auto",
        choices=["auto", "curl", "urllib"],
        help="Archive download backend. auto=curl for cddis.nasa.gov, else urllib.",
    )
    ap.add_argument(
        "--cookie-jar",
        type=Path,
        default=root / "data" / "vlbi" / "sources" / "cddis" / "cookies.txt",
        help="Cookie jar path used by curl -b/-c when download-method includes curl.",
    )
    ap.add_argument(
        "--curl-bin",
        type=str,
        default="curl",
        help="curl executable path/name when using curl download.",
    )
    ap.add_argument(
        "--no-curl-netrc",
        action="store_true",
        help="Disable curl -n (netrc auth). By default curl uses ~/.netrc.",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="Network timeout seconds for archive download.",
    )
    ap.add_argument(
        "--cddis-year",
        type=str,
        default="",
        help="Year folder under base-url (e.g. 2017). If empty and session starts with YY, infer 20YY.",
    )
    args = ap.parse_args()

    session = str(args.session).strip()
    if not session:
        raise ValueError("session must not be empty")

    session_slug = "".join(ch if ch.isalnum() else "_" for ch in session).upper()
    data_root = root / "data" / "vlbi" / "sources" / "vgosdb" / session_slug
    raw_dir = data_root / "raw"
    extracted_dir = data_root / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    source_url = str(args.source_url).strip()
    archive_name = str(args.archive_name).strip()

    if args.input_archive is not None:
        src = args.input_archive.resolve()
        if not src.exists():
            raise FileNotFoundError(f"input archive not found: {src}")

        local_archive = raw_dir / src.name
        if local_archive.exists() and bool(args.force):
            local_archive.unlink()

        if not local_archive.exists():
            shutil.copy2(src, local_archive)

        if not source_url:
            source_url = f"file://{src.as_posix()}"
    else:
        if not source_url:
            if archive_name:
                base = str(args.base_url).rstrip("/") + "/"
                source_url = urllib.parse.urljoin(base, archive_name)
            elif re.fullmatch(r"\d{2}[A-Z]{3}\d{2}[A-Z0-9]{2}", session_slug):
                year_hint = str(args.cddis_year).strip()
                if not year_hint:
                    year_hint = "20" + session_slug[:2]

                base = str(args.base_url).rstrip("/") + "/"
                source_url = urllib.parse.urljoin(base, f"{year_hint}/{session_slug}.tgz")
            else:
                found = _discover_session_url(
                    base_url=str(args.base_url),
                    session=session_slug,
                    max_depth=max(1, int(args.max_discovery_depth)),
                )
                if found is None:
                    raise RuntimeError(
                        "session archive URL not discovered automatically. "
                        "Specify --source-url or --archive-name explicitly."
                    )

                source_url = found

        local_archive = raw_dir / Path(urllib.parse.urlparse(source_url).path).name
        method = str(args.download_method).strip().lower()
        if method == "auto":
            parsed_host = (urllib.parse.urlparse(source_url).hostname or "").lower()
            method = "curl" if "cddis.nasa.gov" in parsed_host else "urllib"

        if method == "curl":
            _download_with_curl(
                source_url,
                local_archive,
                force=bool(args.force),
                cookie_jar=Path(args.cookie_jar).expanduser().resolve(),
                curl_bin=str(args.curl_bin),
                use_netrc=not bool(args.no_curl_netrc),
                timeout_s=max(30, int(args.timeout_sec)),
            )
        else:
            _download(
                source_url,
                local_archive,
                force=bool(args.force),
                timeout_s=max(30, int(args.timeout_sec)),
            )

    archive_size = int(local_archive.stat().st_size)
    archive_hash = _sha256(local_archive)
    archive_type_detected = _detect_archive_type(local_archive)
    extract_info: Dict[str, object] = {}
    if not bool(args.no_extract):
        extract_info = _extract_archive(local_archive, extracted_dir, force=bool(args.force))

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_data = data_root / "manifest.json"
    manifest_json = out_dir / f"vlbi_{session_slug.lower()}_fetch_manifest.json"
    manifest_csv = out_dir / f"vlbi_{session_slug.lower()}_fetch_manifest.csv"
    payload: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session": session_slug,
        "source_url": source_url,
        "archive_path": str(local_archive),
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_hash,
        "archive_type_detected": archive_type_detected,
        "extracted_root": str(extracted_dir),
        "no_extract": bool(args.no_extract),
        "download_method": str(args.download_method),
        "cookie_jar": str(Path(args.cookie_jar).expanduser().resolve()),
        "how_to_fit_command": "python -B scripts/vlbi/vlbi_beta_direct_fit_from_vgosdb.py "
        + f"--session {session_slug} --input-root \"{extracted_dir}\"",
    }
    payload.update(extract_info)
    manifest_data.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(manifest_csv, payload)
    _sync_public(root, [manifest_json, manifest_csv])
    print("Wrote:", manifest_data)
    print("Wrote:", manifest_json)
    print("Wrote:", manifest_csv)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
