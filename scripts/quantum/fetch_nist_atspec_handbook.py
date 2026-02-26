from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0` を満たす経路を評価する。
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return

    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")

    with out_path.open("rb") as f:
        head = f.read(5)

    # 条件分岐: `head != b"%PDF-"` を満たす経路を評価する。

    if head != b"%PDF-":
        raise RuntimeError(f"[fail] downloaded file does not look like a PDF: {out_path} (head={head!r})")


# 関数: `_parse_value_with_paren_unc` の入出力契約と処理意図を定義する。

def _parse_value_with_paren_unc(s: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse e.g.:
      '1420.405 751 766 7(10)' -> (1420.4057517667, 1e-9)
    """
    ss = str(s).strip()
    # 条件分岐: `not ss` を満たす経路を評価する。
    if not ss:
        return None, None

    ss = ss.replace(" ", "")
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


# 関数: `_extract_pdf_text` の入出力契約と処理意図を定義する。

def _extract_pdf_text(pdf_path: Path) -> str:
    # 条件分岐: `not pdf_path.exists()` を満たす経路を評価する。
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        text += "\n" + (page.extract_text() or "")

    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch NIST AtSpec handbook PDF and extract Hydrogen 21 cm (hyperfine) benchmark (Phase 7 / Step 7.12)."
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: do not download; only validate local cache files exist.",
    )
    args = ap.parse_args(argv)

    root = _repo_root()
    out_dir = root / "data" / "quantum" / "sources" / "nist_atspec_handbook"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.physics.nist.gov/Pubs/AtSpec/AtSpec.PDF"
    raw_path = out_dir / "AtSpec.PDF"
    extracted_path = out_dir / "extracted_values.json"
    manifest_path = out_dir / "manifest.json"

    # 条件分岐: `args.offline` を満たす経路を評価する。
    if args.offline:
        missing: list[Path] = []
        for p in (raw_path, extracted_path, manifest_path):
            # 条件分岐: `not p.exists() or p.stat().st_size <= 0` を満たす経路を評価する。
            if not p.exists() or p.stat().st_size <= 0:
                missing.append(p)

        # 条件分岐: `missing` を満たす経路を評価する。

        if missing:
            raise SystemExit("[fail] missing cache files:\n" + "\n".join(f"- {p}" for p in missing))

        print("[ok] offline check passed")
        return 0

    _download(url, raw_path)

    text = _extract_pdf_text(raw_path)
    # Example found in AtSpec.PDF text extraction:
    #   [1420.405 751 766 7(10) MHz]
    pat = re.compile(r"(1420\.\s*[0-9][0-9\s]{6,})\(\s*(\d+)\s*\)\s*MHz", re.I)
    m = pat.search(text)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise RuntimeError(f"[fail] could not find Hydrogen hyperfine (21 cm) frequency token in: {raw_path}")

    val_part = m.group(1).strip()
    unc_digits = m.group(2).strip()
    token = f"{val_part}({unc_digits})"
    f_mhz, sigma_mhz = _parse_value_with_paren_unc(token)
    # 条件分岐: `f_mhz is None or sigma_mhz is None` を満たす経路を評価する。
    if f_mhz is None or sigma_mhz is None:
        raise RuntimeError(f"[fail] could not parse value/uncertainty token: {token!r}")

    extracted: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "NIST AtSpec handbook (PDF): Hydrogen ground-state hyperfine (21 cm line)",
        "url": url,
        "raw_file": str(raw_path),
        "hydrogen_hyperfine_21cm": {
            "token": token,
            "match": m.group(0),
            "f_mhz": float(f_mhz),
            "sigma_mhz": float(sigma_mhz),
            "f_hz": float(f_mhz * 1e6),
            "sigma_hz": float(sigma_mhz * 1e6),
        },
    }
    extracted_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "NIST AtSpec handbook PDF cache (Phase 7 / Step 7.12; H 21 cm hyperfine benchmark).",
        "files": [
            {
                "url": url,
                "path": str(raw_path),
                "bytes": int(raw_path.stat().st_size),
                "sha256": _sha256(raw_path),
            },
            {
                "url": None,
                "path": str(extracted_path),
                "bytes": int(extracted_path.stat().st_size),
                "sha256": _sha256(extracted_path),
                "derived_from": [raw_path.name],
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {raw_path}")
    print(f"[ok] wrote: {extracted_path}")
    print(f"[ok] wrote: {manifest_path}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

