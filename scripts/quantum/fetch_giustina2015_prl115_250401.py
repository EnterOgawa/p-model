from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class FileSpec:
    url: str
    relpath: str
    headers: dict[str, str] | None = None


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, out_path: Path, *, headers: dict[str, str] | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return

    req_headers = {"User-Agent": "waveP/quantum-fetch"}
    if headers:
        req_headers.update(headers)

    req = Request(url, headers=req_headers)
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
            if total_i and done and done % (128 * 1024 * 1024) < len(chunk):
                print(f"  ... {done/total_i:.1%} ({done}/{total_i} bytes)")

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")
    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            member_name = member.name.replace("\\", "/")
            if member_name.startswith("/") or ".." in member_name.split("/"):
                raise RuntimeError(f"unsafe tar member path: {member.name}")
        tf.extractall(out_dir)


def _safe_filename(name: str) -> str:
    # Keep it stable and filesystem-friendly (Windows).
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return s[:180] if s else "download"


def _maybe_fetch_aps_supplemental(*, base_dir: Path, url: str) -> dict[str, object]:
    """
    Try to fetch APS supplemental landing page and any linked files.

    Notes:
    - In some environments APS blocks automated requests (HTTP 403).
    - This function is best-effort and should not fail the whole script.
    """
    out_dir = base_dir / "aps_supplemental"
    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {"landing_url": url, "ok": False, "downloaded": 0, "skipped": 0, "errors": []}

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (waveP/quantum-fetch)"})
        with urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        if isinstance(e, HTTPError):
            status["http_status"] = int(getattr(e, "code", 0) or 0)
            try:
                status["response_headers"] = dict(getattr(e, "headers", {}) or {})
            except Exception:
                pass

            # APS supplemental is protected by Cloudflare in some environments.
            # Typical symptom: HTTP 403 + response header `cf-mitigated: challenge`.
            try:
                hdr = getattr(e, "headers", None)
                cf = hdr.get("cf-mitigated") if hdr is not None else None
                status["blocked_by_cloudflare"] = bool(int(status["http_status"]) == 403 and cf == "challenge")
            except Exception:
                pass
        status["errors"].append(f"{type(e).__name__}: {e}")
        return status

    # Naive href scraping is enough here; do not pull an HTML dependency.
    hrefs = re.findall(r'href=[\"\\\']([^\"\\\']+)[\"\\\']', html, flags=re.IGNORECASE)
    # Keep only likely downloadable assets.
    keep_ext = (".zip", ".pdf", ".csv", ".txt", ".dat", ".json", ".tar.gz", ".tgz")
    links: list[str] = []
    for href in hrefs:
        h = href.strip()
        if not h:
            continue
        if not h.lower().endswith(keep_ext):
            continue
        if h.startswith("//"):
            h = "https:" + h
        elif h.startswith("/"):
            h = "https://link.aps.org" + h
        elif not (h.startswith("http://") or h.startswith("https://")):
            # Relative to the landing URL.
            base = url.rsplit("/", 1)[0]
            h = base + "/" + h
        links.append(h)

    # De-dup while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for u in links:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)

    for file_url in uniq:
        name = file_url.split("/")[-1].split("?")[0]
        name = _safe_filename(name)
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            status["skipped"] = int(status["skipped"]) + 1
            continue
        try:
            _download(file_url, dst, headers={"User-Agent": "Mozilla/5.0 (waveP/quantum-fetch)"})
            status["downloaded"] = int(status["downloaded"]) + 1
        except Exception as e:
            status["errors"].append(f"{type(e).__name__}: {e} ({file_url})")

    status["ok"] = True
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Giustina et al. 2015 (PhysRevLett.115.250401) paper PDFs + arXiv source (incl. supplement) "
            "and write a sha256 manifest. Note: time-tag click logs are not included here."
        )
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download; only verify expected files exist and (re)write manifest.json.",
    )
    parser.add_argument(
        "--include-aps-supplemental",
        action="store_true",
        help="Best-effort: try to fetch APS supplemental materials (may be blocked by HTTP 403).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / "giustina2015_prl115_250401"
    src_dir.mkdir(parents=True, exist_ok=True)

    arxiv_id = "1511.03190v2"
    doi = "10.1103/PhysRevLett.115.250401"

    # arXiv: paper + source tarball (includes "anc/supplemental_material_Vienna_20151220.pdf")
    arxiv_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    arxiv_src_url = f"https://arxiv.org/e-print/{arxiv_id}"

    # APS Harvest API (works even when journals.aps.org supplemental is blocked).
    aps_json_url = f"http://harvest.aps.org/v2/journals/articles/{doi}"
    aps_fulltext_url = f"http://harvest.aps.org/v2/journals/articles/{doi}/fulltext"
    aps_supp_url = f"https://link.aps.org/supplemental/{doi}"

    files: list[FileSpec] = [
        FileSpec(url=arxiv_pdf_url, relpath=f"arxiv_{arxiv_id}.pdf"),
        FileSpec(url=arxiv_src_url, relpath=f"arxiv_{arxiv_id}_src.tar.gz"),
        FileSpec(url=aps_json_url, relpath="aps_article.json", headers={"Accept": "application/json"}),
        FileSpec(url=aps_fulltext_url, relpath="aps_fulltext.pdf", headers={"Accept": "application/pdf"}),
        FileSpec(url=aps_fulltext_url, relpath="aps_article_bag.zip", headers={"Accept": "application/zip"}),
    ]

    expected_supp_rel = Path("arxiv_src") / "anc" / "supplemental_material_Vienna_20151220.pdf"
    expected_supp = src_dir / expected_supp_rel

    missing: list[Path] = []
    for spec in files:
        path = src_dir / spec.relpath
        if not args.offline:
            _download(spec.url, path, headers=spec.headers)
        if not path.exists():
            missing.append(path)

    if missing:
        raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

    # Extract arXiv source if needed (to obtain supplemental PDF offline).
    arxiv_src_tar = src_dir / f"arxiv_{arxiv_id}_src.tar.gz"
    extract_dir = src_dir / "arxiv_src"
    if not expected_supp.exists():
        if not arxiv_src_tar.exists():
            raise SystemExit(f"[fail] missing arXiv source tarball: {arxiv_src_tar}")
        print(f"[info] extracting: {arxiv_src_tar} -> {extract_dir}")
        _safe_extract_tar(arxiv_src_tar, extract_dir)
        if not expected_supp.exists():
            raise SystemExit(f"[fail] expected supplemental PDF not found after extract: {expected_supp}")
        print(f"[ok] extracted: {expected_supp}")
    else:
        print(f"[skip] extracted exists: {expected_supp}")

    # Basic type sanity checks.
    pdf_path = src_dir / "aps_fulltext.pdf"
    if pdf_path.exists():
        with pdf_path.open("rb") as f:
            if f.read(5) != b"%PDF-":
                raise RuntimeError(f"APS fulltext is not a PDF: {pdf_path}")
    bag_path = src_dir / "aps_article_bag.zip"
    if bag_path.exists():
        with bag_path.open("rb") as f:
            if f.read(2) != b"PK":
                raise RuntimeError(f"APS bag is not a zip: {bag_path}")

    manifest = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Giustina et al. 2015 (PhysRevLett.115.250401) paper PDFs + arXiv source (incl. supplement)",
        "doi": doi,
        "arxiv_id": arxiv_id,
        "notes": [
            "This manifest covers paper PDFs + supplemental PDF from arXiv source.",
            "Phonon/photon time-tag click logs (raw event data) are not included here; public availability is unresolved.",
            "APS supplemental landing pages may be blocked by Cloudflare (HTTP 403 + cf-mitigated: challenge) in some environments; Harvest API is used for fulltext.",
        ],
        "files": [],
    }

    def add_file(*, url: str | None, path: Path, extra: dict[str, object] | None = None) -> None:
        item = {
            "url": url,
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        if extra:
            item.update(extra)
        manifest["files"].append(item)

    for spec in files:
        add_file(url=spec.url, path=src_dir / spec.relpath, extra={"headers": spec.headers} if spec.headers else None)

    add_file(
        url=None,
        path=expected_supp,
        extra={"derived_from": str(arxiv_src_tar), "tar_member": "anc/supplemental_material_Vienna_20151220.pdf"},
    )

    aps_supp_status: dict[str, object] | None = None
    if args.include_aps_supplemental and (not args.offline):
        print(f"[info] trying APS supplemental: {aps_supp_url}")
        aps_supp_status = _maybe_fetch_aps_supplemental(base_dir=src_dir, url=aps_supp_url)
        if not aps_supp_status.get("ok"):
            print(f"[warn] APS supplemental fetch failed: {aps_supp_status.get('errors')}")
            if aps_supp_status.get("blocked_by_cloudflare"):
                print(
                    "[hint] APS supplemental is protected by Cloudflare in this environment.\n"
                    "       Download manually in a browser and place files under:\n"
                    f"       {src_dir / 'aps_supplemental'}\n"
                    "       Then run:\n"
                    "       python -B scripts/quantum/fetch_giustina2015_prl115_250401.py --offline\n"
                    "       to (re)write manifest.json including those files."
                )
    if (src_dir / "aps_supplemental").exists():
        for p in sorted((src_dir / "aps_supplemental").glob("*")):
            if p.is_file() and p.stat().st_size > 0:
                add_file(url=None, path=p, extra={"group": "aps_supplemental"})
    if aps_supp_status is not None:
        manifest["aps_supplemental_attempt"] = aps_supp_status

    out = src_dir / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest: {out}")


if __name__ == "__main__":
    main()
