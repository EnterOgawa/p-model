from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen


# クラス: `MolatD2FuvFileSpec` の責務と境界条件を定義する。
@dataclass(frozen=True)
class MolatD2FuvFileSpec:
    file_id: str
    upper_state_label: str

    # 関数: `url` の入出力契約と処理意図を定義する。
    @property
    def url(self) -> str:
        return f"https://molat.obspm.fr/index.php?page=pages/Molecules/D2/view_data.php&fichier={self.file_id}"


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

    return h.hexdigest().upper()


# 関数: `_download` の入出力契約と処理意図を定義する。

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


# 関数: `_extract_pre_block` の入出力契約と処理意図を定義する。

def _extract_pre_block(html: str) -> str:
    m = re.search(r"<pre>\s*(.*?)\s*</pre>", html, flags=re.IGNORECASE | re.DOTALL)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError("missing <pre>...</pre> block")
    # Normalize line endings and strip trailing spaces to keep cache stable.

    text = m.group(1)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Drop leading empty lines.
    while lines and not lines[0].strip():
        lines.pop(0)

    return "\n".join(lines).strip() + "\n"


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# 関数: `_parse_pre_stats` の入出力契約と処理意図を定義する。

def _parse_pre_stats(pre_text: str) -> dict[str, Any]:
    """
    MOLAT D2 view_data: whitespace-delimited rows like:
      VU JU VL JL A(s-1) TR.E.(cm-1) note
    Header is 2-3 lines, then a count, then numeric rows.
    """
    n_rows = 0
    a_min = None
    a_max = None
    nu_min = None
    nu_max = None
    for line in pre_text.splitlines():
        parts = line.split()
        # 条件分岐: `len(parts) < 7` を満たす経路を評価する。
        if len(parts) < 7:
            continue

        vu = _safe_float(parts[0])
        ju = _safe_float(parts[1])
        vl = _safe_float(parts[2])
        jl = _safe_float(parts[3])
        a = _safe_float(parts[4].replace("D", "E"))
        nu = _safe_float(parts[5])
        # 条件分岐: `None in (vu, ju, vl, jl, a, nu)` を満たす経路を評価する。
        if None in (vu, ju, vl, jl, a, nu):
            continue

        n_rows += 1
        a_min = a if a_min is None else min(a_min, a)
        a_max = a if a_max is None else max(a_max, a)
        nu_min = nu if nu_min is None else min(nu_min, nu)
        nu_max = nu if nu_max is None else max(nu_max, nu)

    return {
        "n_rows": n_rows,
        "A_min_s^-1": (None if a_min is None else float(a_min)),
        "A_max_s^-1": (None if a_max is None else float(a_max)),
        "wavenumber_min_cm^-1": (None if nu_min is None else float(nu_min)),
        "wavenumber_max_cm^-1": (None if nu_max is None else float(nu_max)),
    }


# 関数: `_write_manifest` の入出力契約と処理意図を定義する。

def _write_manifest(
    *,
    out_dir: Path,
    generated_utc: str,
    base_page_url: str,
    raw_files: list[Path],
    extracted_file: Path,
) -> Path:
    manifest: dict[str, Any] = {
        "generated_utc": generated_utc,
        "dataset": "MOLAT (OBSPM): D2 FUV emission line lists (transition energies + A; Abgrall et al. 1999)",
        "base_page_url": base_page_url,
        "raw_files": [
            {
                "file": p.name,
                "bytes": p.stat().st_size,
                "sha256": _sha256(p),
            }
            for p in raw_files
            if p.exists()
        ],
        "extracted_values": extracted_file.name,
        "notes": [
            "Source site: MOLAT (Paris Observatory / LERMA).",
            "This cache stores the raw HTML of view_data pages and the extracted <pre> blocks as plain text.",
            "Each numeric row is: VU JU VL JL A[s^-1] TR.E.[cm^-1] note.",
        ],
    }
    out_path = out_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch MOLAT D2 FUV emission line lists (view_data) and cache for offline use.")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: do not download; only validate local cache files exist.",
    )
    ap.add_argument("--force", action="store_true", help="Redownload even if cache files already exist.")
    args = ap.parse_args(argv)

    root = _repo_root()
    out_dir = root / "data" / "quantum" / "sources" / "molat_d2_fuv_emission_arlsj1999"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_page_url = "https://molat.obspm.fr/index.php?page=pages/Molecules/D2/D2.php"
    base_page_path = out_dir / "molat_d2_dataset_page__D2.php.html"

    specs = [
        MolatD2FuvFileSpec(file_id="BphysB99", upper_state_label="B"),
        MolatD2FuvFileSpec(file_id="BpphysB99", upper_state_label="B′"),
        MolatD2FuvFileSpec(file_id="CmphysB99", upper_state_label="C−"),
        MolatD2FuvFileSpec(file_id="CpphysB99", upper_state_label="C+"),
        MolatD2FuvFileSpec(file_id="DmphysB99", upper_state_label="D−"),
        MolatD2FuvFileSpec(file_id="DpphysB99", upper_state_label="D+"),
    ]

    extracted_path = out_dir / "extracted_values.json"
    manifest_path = out_dir / "manifest.json"

    raw_html_paths: list[Path] = [base_page_path]
    pre_txt_paths: list[Path] = []
    for spec in specs:
        raw_html_paths.append(out_dir / f"molat_d2_view_data__{spec.file_id}.html")
        pre_txt_paths.append(out_dir / f"molat_d2_lines__{spec.file_id}.txt")

    # 条件分岐: `args.offline` を満たす経路を評価する。

    if args.offline:
        missing = [p for p in [*raw_html_paths, *pre_txt_paths, extracted_path, manifest_path] if not p.exists() or p.stat().st_size <= 0]
        # 条件分岐: `missing` を満たす経路を評価する。
        if missing:
            raise SystemExit("[fail] missing cache files:\n" + "\n".join(f"- {p}" for p in missing))

        print("[ok] offline check passed")
        return 0

    generated_utc = _iso_utc_now()

    _download(base_page_url, base_page_path, force=bool(args.force))

    files_out: list[dict[str, Any]] = []
    for spec, raw_html_path, pre_txt_path in zip(specs, raw_html_paths[1:], pre_txt_paths, strict=False):
        _download(spec.url, raw_html_path, force=bool(args.force))
        html = raw_html_path.read_text(encoding="utf-8", errors="replace")
        pre_text = _extract_pre_block(html)
        pre_txt_path.write_text(pre_text, encoding="utf-8")
        stats = _parse_pre_stats(pre_text)
        files_out.append(
            {
                "file_id": spec.file_id,
                "upper_state_label": spec.upper_state_label,
                "url": spec.url,
                "raw_html_file": raw_html_path.name,
                "raw_html_sha256": _sha256(raw_html_path),
                "pre_text_file": pre_txt_path.name,
                "pre_text_sha256": _sha256(pre_txt_path),
                "stats": stats,
            }
        )

    extracted: dict[str, Any] = {
        "generated_utc": generated_utc,
        "dataset": "MOLAT (OBSPM): D2 FUV emission line lists (transition energies + A; Abgrall et al. 1999)",
        "base_page_url": base_page_url,
        "files": files_out,
        "notes": [
            "Upper-state labels follow MOLAT page naming: B, B′, C−, C+, D−, D+.",
            "Lower state is the electronic ground state X (as described on MOLAT D2 page).",
            "This dataset is used as an objective 'representative transitions' target set (top-N by A).",
        ],
    }
    extracted_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_file = _write_manifest(
        out_dir=out_dir,
        generated_utc=generated_utc,
        base_page_url=base_page_url,
        raw_files=[*raw_html_paths, *pre_txt_paths],
        extracted_file=extracted_path,
    )

    print(f"[ok] wrote: {extracted_path}")
    print(f"[ok] wrote: {manifest_file}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
