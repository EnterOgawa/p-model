#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _read_head(path: Path, n: int = 8) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _download(url: str, dst: Path, *, force: bool, user_agent: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[dl] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=240) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")


def _safe_extractall(tf: tarfile.TarFile, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in tf.getmembers():
        if not isinstance(m.name, str):
            continue
        p = Path(m.name)
        if p.is_absolute():
            raise RuntimeError(f"unsafe tar member (absolute path): {m.name}")
        if any(part in ("..", "") for part in p.parts):
            raise RuntimeError(f"unsafe tar member (path traversal): {m.name}")
    tf.extractall(out_dir)


def _extract_src(tar_path: Path, out_dir: Path, *, force: bool) -> bool:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"[skip] extracted: {out_dir}")
        return True

    if force and out_dir.exists():
        for p in out_dir.glob("*"):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)

    head = _read_head(tar_path, 5)
    if head.startswith(b"%PDF"):
        print(f"[warn] e-print looks like PDF (not tar.gz): {tar_path.name} (skipping extract)")
        return False
    if not head.startswith(b"\x1f\x8b"):
        print(f"[warn] e-print is not gzip (skipping extract): {tar_path.name}")
        return False

    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            _safe_extractall(tf, out_dir)
        print(f"[ok] extracted: {out_dir}")
        return True
    except tarfile.ReadError as e:
        print(f"[warn] tar read failed: {tar_path.name}: {e} (skipping extract)")
        return False


def _count_files(dir_path: Path, suffix: str) -> Optional[int]:
    if not (dir_path.exists() and dir_path.is_dir()):
        return None
    return sum(1 for _ in dir_path.rglob(f"*{suffix}"))


def _safe_arxiv_id_for_path(arxiv_id: str) -> str:
    # Convert old-style ids like "astro-ph/0607432" to a path-safe token that matches our cache naming.
    # Example: "astro-ph/0607432" -> "astroph_0607432"
    return arxiv_id.replace("-", "").replace("/", "_")


def _paper_list() -> List[Tuple[str, str, str]]:
    # Sources that contribute to Wielgus+2022 tab:detections_other_papers (Paper V "historical distribution").
    # Marrone (2006) thesis is handled separately; it is not an arXiv e-print.
    return [
        ("marrone2006_mm_variability", "astro-ph/0607432", "Marrone et al. (2006) arXiv:astro-ph/0607432 (SMA 2005)"),
        ("marrone2008_flare", "0712.2877", "Marrone et al. (2008) An X-Ray, Infrared, and Submillimeter Flare of Sagittarius A*"),
        ("yusef2009_mwl_2007", "0907.3786", "Yusef-Zadeh et al. (2009) Simultaneous Multi-Wavelength Observations of Sgr A* During 2007 April 1-11"),
        ("dexter2014_submm_timescale", "1308.5968", "Dexter et al. (2014) An 8 h characteristic time-scale in submillimetre light curves of Sagittarius A*"),
        ("fazio2018_mwl_flares", "1807.07599", "Fazio et al. (2018) Multiwavelength Light Curves of Two Remarkable Sagittarius A* Flares"),
        ("bower2018_alma_polarimetry", "1810.07317", "Bower et al. (2018) ALMA Polarimetry of Sgr A*"),
        ("witzel2021_rapid_variability", "2011.09582", "Witzel et al. (2021) Rapid Variability of Sgr A* across the Electromagnetic Spectrum"),
    ]


def _add_row_for_arxiv(
    root: Path,
    sources_dir: Path,
    *,
    key: str,
    arxiv_id: str,
    label: str,
    offline: bool,
    force: bool,
    user_agent: str,
) -> Dict[str, Any]:
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    src_url = f"https://arxiv.org/e-print/{arxiv_id}"

    safe_id = _safe_arxiv_id_for_path(arxiv_id)
    pdf_path = sources_dir / f"arxiv_{safe_id}.pdf"
    src_path = sources_dir / f"arxiv_{safe_id}_src.tar.gz"
    extract_dir = sources_dir / f"arxiv_{safe_id}"

    notes: List[str] = []
    if not offline:
        try:
            _download(pdf_url, pdf_path, force=force, user_agent=user_agent)
        except Exception as e:
            notes.append(f"pdf download failed: {e}")
        try:
            _download(src_url, src_path, force=force, user_agent=user_agent)
        except Exception as e:
            notes.append(f"src download failed: {e}")

    extracted_ok = False
    try:
        if src_path.exists():
            extracted_ok = _extract_src(src_path, extract_dir, force=force)
    except Exception as e:
        notes.append(f"extract failed: {e}")
        extracted_ok = False

    return {
        "key": key,
        "arxiv": arxiv_id,
        "arxiv_safe_id": safe_id,
        "label": label,
        "url_abs": f"https://arxiv.org/abs/{arxiv_id}",
        "local_pdf": str(pdf_path.relative_to(root)).replace("\\", "/") if pdf_path.exists() else None,
        "local_pdf_sha256": _sha256(pdf_path) if pdf_path.exists() else None,
        "local_src": str(src_path.relative_to(root)).replace("\\", "/") if src_path.exists() else None,
        "local_src_sha256": _sha256(src_path) if src_path.exists() else None,
        "extract_dir": str(extract_dir.relative_to(root)).replace("\\", "/") if extract_dir.exists() else None,
        "extract_ok": extracted_ok,
        "tex_files": _count_files(extract_dir, ".tex"),
        "data_like_files": _count_files(extract_dir, ".dat"),
        "notes": notes,
    }


def _add_row_for_marrone2006_thesis(
    root: Path,
    sources_dir: Path,
    *,
    offline: bool,
    force: bool,
    user_agent: str,
) -> Dict[str, Any]:
    url_pdf = "https://lweb.cfa.harvard.edu/~dmarrone/dpm_thesis.pdf"
    url_ps_gz = "https://lweb.cfa.harvard.edu/~dmarrone/dpm_thesis.ps.gz"

    pdf_path = sources_dir / "marrone2006_thesis_dpm_thesis.pdf"
    ps_gz_path = sources_dir / "marrone2006_thesis_dpm_thesis.ps.gz"

    notes: List[str] = []
    if not offline:
        try:
            _download(url_pdf, pdf_path, force=force, user_agent=user_agent)
        except Exception as e:
            notes.append(f"pdf download failed: {e}")
        try:
            _download(url_ps_gz, ps_gz_path, force=force, user_agent=user_agent)
        except Exception as e:
            notes.append(f"ps.gz download failed: {e}")

    return {
        "key": "marrone2006_thesis",
        "kind": "direct",
        "label": "Marrone (2006) thesis (CfA) dpm_thesis.{pdf,ps.gz}",
        "urls": {"pdf": url_pdf, "ps_gz": url_ps_gz},
        "local_pdf": str(pdf_path.relative_to(root)).replace("\\", "/") if pdf_path.exists() else None,
        "local_pdf_sha256": _sha256(pdf_path) if pdf_path.exists() else None,
        "local_ps_gz": str(ps_gz_path.relative_to(root)).replace("\\", "/") if ps_gz_path.exists() else None,
        "local_ps_gz_sha256": _sha256(ps_gz_path) if ps_gz_path.exists() else None,
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch historical (pre-EHT) Sgr A* mm/submm light curve papers (PDF + arXiv source) for offline reproducibility."
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify existing files and write manifest.")
    ap.add_argument("--force", action="store_true", help="Redownload/re-extract even if files already exist.")
    args = ap.parse_args()

    root = _repo_root()
    sources_dir = root / "data" / "eht" / "sources"
    out_dir = root / "output" / "private" / "eht"
    out_dir.mkdir(parents=True, exist_ok=True)

    user_agent = "waveP-eht-historical-fetch/1.0"

    rows: List[Dict[str, Any]] = []
    rows.append(
        _add_row_for_marrone2006_thesis(
            root,
            sources_dir,
            offline=bool(args.offline),
            force=bool(args.force),
            user_agent=user_agent,
        )
    )
    for key, arxiv_id, label in _paper_list():
        rows.append(
            _add_row_for_arxiv(
                root,
                sources_dir,
                key=key,
                arxiv_id=arxiv_id,
                label=label,
                offline=bool(args.offline),
                force=bool(args.force),
                user_agent=user_agent,
            )
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": {"offline": bool(args.offline), "force": bool(args.force)},
        "output": "output/private/eht/eht_sgra_historical_mm_lightcurves_papers_sources_manifest.json",
        "rows": rows,
    }
    out_json = out_dir / "eht_sgra_historical_mm_lightcurves_papers_sources_manifest.json"
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        files_with_sha256 = 0
        for r in rows:
            if not isinstance(r, dict):
                continue
            for k, v in r.items():
                if isinstance(k, str) and k.endswith("_sha256") and isinstance(v, str) and v:
                    files_with_sha256 += 1
        worklog.append_event(
            {
                "ts_utc": manifest["generated_utc"],
                "topic": "eht",
                "action": "fetch_sgra_historical_mm_lightcurves_papers",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {
                    "rows": len(rows),
                    "papers": len(rows),
                    "arxiv_rows": sum(1 for r in rows if isinstance(r, dict) and r.get("arxiv")),
                    "direct_rows": sum(1 for r in rows if isinstance(r, dict) and r.get("kind") == "direct"),
                    "pdf_ok": sum(1 for r in rows if r.get("local_pdf_sha256")),
                    "src_ok": sum(1 for r in rows if r.get("local_src_sha256")),
                    "files_with_sha256": int(files_with_sha256),
                    "extract_ok": sum(1 for r in rows if r.get("extract_ok") is True),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
