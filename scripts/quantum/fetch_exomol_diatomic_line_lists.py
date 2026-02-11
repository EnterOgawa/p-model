from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ExoMolLineListSpec:
    molecule: str  # ExoMol molecule folder (e.g., "H2")
    isotopologue: str  # ExoMol isotopologue tag (e.g., "1H2", "1H-2H")
    dataset: str  # ExoMol dataset tag (e.g., "RACPPK")

    @property
    def page_url(self) -> str:
        return f"https://exomol.com/data/molecules/{self.molecule}/{self.isotopologue}/{self.dataset}"

    @property
    def db_base_url(self) -> str:
        return f"https://exomol.com/db/{self.molecule}/{self.isotopologue}/{self.dataset}/"

    @property
    def file_prefix(self) -> str:
        return f"{self.isotopologue}__{self.dataset}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest().upper()


def _sanitize_token(s: str) -> str:
    ss = str(s).strip().lower()
    ss = re.sub(r"\s+", "_", ss)
    ss = re.sub(r"[^a-z0-9_]+", "_", ss)
    ss = re.sub(r"_+", "_", ss).strip("_")
    return ss or "unknown"


def _download(url: str, out_path: Path, *, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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


def _read_bz2_text(path: Path) -> str:
    raw = path.read_bytes()
    return bz2.decompress(raw).decode("utf-8", errors="replace")


def _parse_states_stats(text: str) -> dict[str, Any]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            state_id = int(parts[0])
            energy_cm_inv = float(parts[1])
        except Exception:
            continue
        rows.append((state_id, energy_cm_inv, len(parts)))
    if not rows:
        return {"n_states": 0}
    energies = [e for _, e, _ in rows]
    col_counts = sorted({c for _, _, c in rows})
    return {
        "n_states": len(rows),
        "energy_min_cm^-1": float(min(energies)),
        "energy_max_cm^-1": float(max(energies)),
        "state_id_min": int(min(i for i, _, _ in rows)),
        "state_id_max": int(max(i for i, _, _ in rows)),
        "columns_observed": col_counts,
    }


def _parse_trans_stats(text: str) -> dict[str, Any]:
    n = 0
    nu_min = None
    nu_max = None
    a_max = None
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            A = float(parts[2])
            nu = float(parts[3])
        except Exception:
            continue
        n += 1
        nu_min = nu if nu_min is None else min(nu_min, nu)
        nu_max = nu if nu_max is None else max(nu_max, nu)
        a_max = A if a_max is None else max(a_max, A)
    return {
        "n_trans": n,
        "wavenumber_min_cm^-1": (None if nu_min is None else float(nu_min)),
        "wavenumber_max_cm^-1": (None if nu_max is None else float(nu_max)),
        "A_max_s^-1": (None if a_max is None else float(a_max)),
    }


def _write_manifest(
    *,
    out_dir: Path,
    spec: ExoMolLineListSpec,
    generated_utc: str,
    raw_files: list[Path],
    extracted_file: Path,
) -> Path:
    manifest = {
        "generated_utc": generated_utc,
        "dataset": "ExoMol line list (states/trans/pf)",
        "molecule": spec.molecule,
        "isotopologue": spec.isotopologue,
        "dataset_tag": spec.dataset,
        "page_url": spec.page_url,
        "db_base_url": spec.db_base_url,
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
            "Primary-source cache for diatomic transition line lists (ExoMol).",
            "Used to fix representative transitions as targets in Phase 7 / Step 7.12.",
        ],
    }
    out_path = out_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _write_extracted(
    *,
    out_dir: Path,
    spec: ExoMolLineListSpec,
    generated_utc: str,
    page_html_path: Optional[Path],
    states_path: Path,
    trans_path: Path,
    pf_path: Path,
) -> Path:
    states_text = _read_bz2_text(states_path)
    trans_text = _read_bz2_text(trans_path)
    extracted = {
        "generated_utc": generated_utc,
        "dataset": "ExoMol line list (states/trans/pf)",
        "molecule": spec.molecule,
        "isotopologue": spec.isotopologue,
        "dataset_tag": spec.dataset,
        "page_url": spec.page_url,
        "db_base_url": spec.db_base_url,
        "raw": {
            "page_html_file": (None if page_html_path is None else page_html_path.name),
            "page_html_sha256": (None if page_html_path is None else _sha256(page_html_path)),
            "states_bz2_file": states_path.name,
            "states_bz2_sha256": _sha256(states_path),
            "trans_bz2_file": trans_path.name,
            "trans_bz2_sha256": _sha256(trans_path),
            "pf_file": pf_path.name,
            "pf_sha256": _sha256(pf_path),
        },
        "stats": {
            "states": _parse_states_stats(states_text),
            "trans": _parse_trans_stats(trans_text),
        },
        "notes": [
            "The .states/.trans files are text tables compressed with bzip2.",
            "Transitions contain (upper_id, lower_id, A[s^-1], wavenumber[cm^-1]) in this dataset.",
        ],
    }
    out_path = out_dir / "extracted_values.json"
    out_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _offline_check(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("[fail] missing cache files:\n" + "\n".join(f"- {p}" for p in missing))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch ExoMol diatomic line lists (H2/HD) and cache for offline use.")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: do not download; only validate local cache files exist.",
    )
    ap.add_argument("--force", action="store_true", help="Redownload even if cache files already exist.")
    args = ap.parse_args(argv)

    root = _repo_root()
    base_dir = root / "data" / "quantum" / "sources"
    base_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ExoMolLineListSpec(molecule="H2", isotopologue="1H2", dataset="RACPPK"),
        ExoMolLineListSpec(molecule="H2", isotopologue="1H-2H", dataset="ADJSAAM"),
    ]

    for spec in specs:
        slug = f"exomol_{_sanitize_token(spec.molecule)}_{_sanitize_token(spec.isotopologue)}_{_sanitize_token(spec.dataset)}"
        out_dir = base_dir / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        page_html_path = out_dir / f"exomol_dataset_page__{spec.file_prefix}.html"
        states_path = out_dir / f"{spec.file_prefix}.states.bz2"
        trans_path = out_dir / f"{spec.file_prefix}.trans.bz2"
        pf_path = out_dir / f"{spec.file_prefix}.pf"
        extracted_path = out_dir / "extracted_values.json"
        manifest_path = out_dir / "manifest.json"

        if args.offline:
            _offline_check([states_path, trans_path, pf_path, extracted_path, manifest_path])
            continue

        generated_utc = _iso_utc_now()

        # Cache the dataset page (small; useful for provenance).
        try:
            _download(spec.page_url, page_html_path, force=bool(args.force))
        except Exception as e:
            print(f"[warn] page fetch failed: {spec.page_url}: {e}")
            page_html_path = None

        _download(spec.db_base_url + states_path.name, states_path, force=bool(args.force))
        _download(spec.db_base_url + trans_path.name, trans_path, force=bool(args.force))
        _download(spec.db_base_url + pf_path.name, pf_path, force=bool(args.force))

        extracted_file = _write_extracted(
            out_dir=out_dir,
            spec=spec,
            generated_utc=generated_utc,
            page_html_path=page_html_path,
            states_path=states_path,
            trans_path=trans_path,
            pf_path=pf_path,
        )
        manifest_file = _write_manifest(
            out_dir=out_dir,
            spec=spec,
            generated_utc=generated_utc,
            raw_files=[p for p in [page_html_path, states_path, trans_path, pf_path] if p is not None],
            extracted_file=extracted_file,
        )

        print(f"[ok] wrote: {extracted_file}")
        print(f"[ok] wrote: {manifest_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

