from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _download(url: str, out_path: Path, *, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0 and not force` を満たす経路を評価する。
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"[skip] exists: {out_path}")
        return

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    print(f"[dl] {url}")
    with urlopen(req, timeout=180) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)

    tmp.replace(out_path)
    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _strip_tags(s: str) -> str:
    ss = re.sub(r"<[^>]+>", "", s)
    ss = html_module.unescape(ss).replace("\u00a0", " ")
    ss = re.sub(r"\s+", " ", ss).strip()
    return ss


def _extract_arxiv_abstract(html: str) -> str:
    m = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', html, flags=re.S | re.M)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError("abstract blockquote not found in arXiv HTML")

    txt = _strip_tags(m.group(1))
    return re.sub(r"^Abstract:\s*", "", txt)


def _parse_value_with_paren_unc(s: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse e.g.:
      '36\\,748.362\\,282(26)' -> (36748.362282, 0.000026)
      '35999.582834(11)' -> (35999.582834, 0.000011)
      '36405.78253(7)' -> (36405.78253, 0.00007)
    """
    ss = str(s).strip()
    # 条件分岐: `not ss` を満たす経路を評価する。
    if not ss:
        return None, None

    ss = ss.replace(" ", "")
    ss = ss.replace("\\,", "")  # arXiv thin-space thousands separators
    ss = ss.replace(",", "")  # just in case
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?:\((\d+)\))?", ss)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return None, None

    val_str = m.group(1)
    unc_str = m.group(2)
    try:
        val = float(val_str)
    except Exception:
        return None, None

    # 条件分岐: `not unc_str` を満たす経路を評価する。

    if not unc_str:
        return val, None

    decimals = 0
    # 条件分岐: `"." in val_str` を満たす経路を評価する。
    if "." in val_str:
        decimals = len(val_str.split(".", 1)[1])

    try:
        unc = int(unc_str) * (10.0 ** (-decimals))
    except Exception:
        unc = None

    return val, unc


def _extract_d0_from_arxiv_abstract(abstract: str, *, molecule: str) -> dict[str, Any]:
    """
    Extract spectroscopic dissociation energy D0 from an arXiv abstract.
    Returns a dict with at least:
      - d0_token
      - d0_cm^-1
      - d0_unc_cm^-1
    """
    # 条件分岐: `molecule == "D2"` を満たす経路を評価する。
    if molecule == "D2":
        # Pattern: ... dissociation energy ... D_0(D_2) = 36\,748.362\,282(26) \wn ...
        m = re.search(r"D_0[^{=]{0,120}=\s*([0-9][0-9\\,\.]+(?:\(\d+\))?)", abstract)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            raise ValueError("D0 token not found in D2 arXiv abstract")

        token = m.group(1)
        val, unc = _parse_value_with_paren_unc(token)
        return {"d0_token": token, "d0_cm^-1": val, "d0_unc_cm^-1": unc, "rotational_N": 0}

    # 条件分岐: `molecule == "H2"` を満たす経路を評価する。

    if molecule == "H2":
        # Prefer the experimental value: "The new result of 35999.582834(11) cm^{-1} ..."
        m = re.search(r"new result of\s+([0-9]+(?:\.[0-9]+)?\(\d+\))", abstract, flags=re.I)
        # 条件分岐: `m` を満たす経路を評価する。
        if m:
            token = m.group(1)
            val, unc = _parse_value_with_paren_unc(token)
            return {"d0_token": token, "d0_cm^-1": val, "d0_unc_cm^-1": unc, "rotational_N": 1}

        # Fallback: first wavenumber with uncertainty after mentioning dissociation energy.

        m2 = re.search(r"dissociation energy.*?([0-9]+(?:\.[0-9]+)?\(\d+\))\s*cm", abstract, flags=re.I)
        # 条件分岐: `not m2` を満たす経路を評価する。
        if not m2:
            raise ValueError("D0 token not found in H2 arXiv abstract")

        token = m2.group(1)
        val, unc = _parse_value_with_paren_unc(token)
        return {"d0_token": token, "d0_cm^-1": val, "d0_unc_cm^-1": unc, "rotational_N": 1}

    raise ValueError(f"unsupported molecule: {molecule}")


def _extract_d0_from_vu_html(html: str) -> dict[str, Any]:
    # Pattern: ... D0(HD)=36405.78253(7)cm-1 ...
    m = re.search(r"D0\(HD\)\s*=\s*([0-9]+(?:\.[0-9]+)?\(\d+\))\s*cm-1", html)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        # Sometimes there is no space between token and unit.
        m = re.search(r"D0\(HD\)\s*=\s*([0-9]+(?:\.[0-9]+)?\(\d+\))cm-1", html)

    # 条件分岐: `not m` を満たす経路を評価する。

    if not m:
        raise ValueError("D0(HD) token not found in VU HTML")

    token = m.group(1)
    val, unc = _parse_value_with_paren_unc(token)
    return {"d0_token": token, "d0_cm^-1": val, "d0_unc_cm^-1": unc, "rotational_N": 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch spectroscopic dissociation energies D0 (0 K; wavenumber) for H2/HD/D2 and cache as primary sources."
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: do not download; only validate local cache files exist.",
    )
    ap.add_argument("--force", action="store_true", help="Redownload even if cache files already exist.")
    args = ap.parse_args(argv)

    root = _repo_root()
    src_dir = root / "data" / "quantum" / "sources" / "molecular_dissociation_d0_spectroscopic"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Sources
    arxiv_sources = [
        {
            "molecule": "H2",
            "arxiv_id": "1902.09471",
            "note": "ortho-H2 dissociation energy D0^{N=1} (spectroscopic; wavenumber)",
        },
        {
            "molecule": "D2",
            "arxiv_id": "2202.00532",
            "note": "D2 dissociation energy D0 (spectroscopic; wavenumber)",
        },
    ]

    vu_hd_url = (
        "https://research.vu.nl/en/publications/ionization-and-dissociation-energies-of-hd-and-dipole-induced-gu-/"
    )

    raw_files: list[Path] = []
    extracted_records: list[dict[str, Any]] = []

    # arXiv: cache abs HTML + PDF.
    for s in arxiv_sources:
        arxiv_id = s["arxiv_id"]
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        abs_path = src_dir / f"arxiv_abs_{arxiv_id}.html"
        pdf_path = src_dir / f"arxiv_pdf_{arxiv_id}.pdf"

        # 条件分岐: `args.offline` を満たす経路を評価する。
        if args.offline:
            raw_files.extend([abs_path, pdf_path])
            continue

        _download(abs_url, abs_path, force=bool(args.force))
        _download(pdf_url, pdf_path, force=bool(args.force))
        raw_files.extend([abs_path, pdf_path])

        html = abs_path.read_text(encoding="utf-8", errors="replace")
        abstract = _extract_arxiv_abstract(html)
        extracted = _extract_d0_from_arxiv_abstract(abstract, molecule=str(s["molecule"]))
        extracted_records.append(
            {
                "molecule": s["molecule"],
                "rotational_N": extracted["rotational_N"],
                "d0_cm^-1": extracted["d0_cm^-1"],
                "d0_unc_cm^-1": extracted["d0_unc_cm^-1"],
                "source": {
                    "type": "arxiv",
                    "arxiv_id": arxiv_id,
                    "url_abs": abs_url,
                    "url_pdf": pdf_url,
                    "raw_abs_file": abs_path.name,
                    "raw_abs_sha256": _sha256(abs_path),
                    "raw_pdf_file": pdf_path.name,
                    "raw_pdf_sha256": _sha256(pdf_path),
                    "note": s["note"],
                    "abstract_text": abstract,
                    "d0_token": extracted["d0_token"],
                },
            }
        )

    # VU Amsterdam: cache HTML page for HD.

    vu_path = src_dir / "vu_publication_hd_d0.html"
    # 条件分岐: `args.offline` を満たす経路を評価する。
    if args.offline:
        raw_files.append(vu_path)
    else:
        _download(vu_hd_url, vu_path, force=bool(args.force))
        raw_files.append(vu_path)

        vu_html = vu_path.read_text(encoding="utf-8", errors="replace")
        extracted = _extract_d0_from_vu_html(vu_html)
        extracted_records.append(
            {
                "molecule": "HD",
                "rotational_N": extracted["rotational_N"],
                "d0_cm^-1": extracted["d0_cm^-1"],
                "d0_unc_cm^-1": extracted["d0_unc_cm^-1"],
                "source": {
                    "type": "vu_publication_page",
                    "url": vu_hd_url,
                    "raw_html_file": vu_path.name,
                    "raw_html_sha256": _sha256(vu_path),
                    "note": "VU Amsterdam publication page (abstract text contains D0(HD) in cm^-1)",
                    "d0_token": extracted["d0_token"],
                },
            }
        )

    extracted_path = src_dir / "extracted_values.json"
    manifest_path = src_dir / "manifest.json"

    # 条件分岐: `args.offline` を満たす経路を評価する。
    if args.offline:
        missing = [p for p in raw_files + [extracted_path, manifest_path] if not p.exists()]
        # 条件分岐: `missing` を満たす経路を評価する。
        if missing:
            raise SystemExit("[fail] missing cache files:\n" + "\n".join(f"- {p}" for p in missing))

        print("[ok] offline check passed")
        return 0

    extracted_json: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "Spectroscopic dissociation energy D0 (0 K; wavenumber) for H2/HD/D2",
        "units": {"d0": "cm^-1"},
        "sources": extracted_records,
        "notes": [
            "This cache fixes D0 (0 K; spectroscopic dissociation energy) as an independent baseline for Phase 7 / Step 7.12.",
            "H2 uses an ortho-H2 (N=1) D0 value as reported in the referenced primary source; rotational state must be tracked.",
            "Do not conflate D0 (0 K) with thermochemistry-based dissociation enthalpy at 298 K.",
        ],
    }
    extracted_path.write_text(json.dumps(extracted_json, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {
        "generated_utc": extracted_json["generated_utc"],
        "dataset": extracted_json["dataset"],
        "dir": src_dir.name,
        "raw_files": [
            {
                "file": p.name,
                "sha256": _sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in raw_files
            if p.exists()
        ],
        "extracted_values": extracted_path.name,
        "notes": [
            "Primary-source cache for spectroscopic D0 baselines (H2/HD/D2).",
            "Used by scripts/quantum/molecular_dissociation_d0_spectroscopic.py.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {extracted_path}")
    print(f"[ok] wrote: {manifest_path}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

