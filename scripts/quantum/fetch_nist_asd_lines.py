from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class TargetLine:
    id: str
    approx_lambda_vac_A: float
    window_A: float
    prefer_max_Aki: bool = True


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
    return h.hexdigest()


def _sanitize_token(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("+", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "unknown"


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return
    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req) as resp, out_path.open("wb") as f:
        f.write(resp.read())
    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    rows_raw = list(csv.reader(txt.splitlines(), delimiter="\t", quotechar='"'))
    if not rows_raw:
        raise ValueError(f"empty TSV: {path}")
    header = [h.strip() for h in rows_raw[0]]
    # NIST often leaves a trailing empty column due to final '\t'
    while header and header[-1] == "":
        header.pop()

    rows: list[dict[str, str]] = []
    for r in rows_raw[1:]:
        row = list(r)
        while row and row[-1] == "":
            row.pop()
        if not row:
            continue
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        rec = {header[i]: (row[i].strip() if i < len(row) else "") for i in range(len(header))}
        rows.append(rec)
    return header, rows


def _as_float(s: str | None) -> float | None:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    try:
        return float(ss)
    except Exception:
        return None


def _derive_lambda_from_obs_nu_A(nu_invA: float) -> float | None:
    if nu_invA == 0.0:
        return None
    # NIST "obs_nu(A)" values are negative in this output; wavelength is positive.
    return -1.0 / nu_invA


def _pick_line_for_target(rows: list[dict[str, str]], target: TargetLine) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for r in rows:
        nu = _as_float(r.get("obs_nu(A)"))
        if nu is None:
            continue
        lam = _derive_lambda_from_obs_nu_A(nu)
        if lam is None:
            continue
        if abs(lam - target.approx_lambda_vac_A) > target.window_A:
            continue
        aki = _as_float(r.get("Aki(s^-1)"))
        unc_nu = _as_float(r.get("unc_obs_nu"))
        candidates.append(
            {
                "lambda_vac_A": float(lam),
                "delta_A": float(lam - target.approx_lambda_vac_A),
                "Aki_s^-1": aki,
                "nu_obs_invA": float(nu),
                "nu_obs_unc_invA": (abs(float(unc_nu)) if unc_nu is not None else None),
                "Acc": (r.get("Acc") or None),
                "Type": (r.get("Type") or None),
                "ritz_nu_invA": _as_float(r.get("ritz_wn(A)")),
                "ritz_unc_invA": _as_float(r.get("unc_ritz_wn")),
                "raw": r,
            }
        )

    if not candidates:
        return None

    # Prefer the most intense/standard E1 line when available (max Aki).
    if target.prefer_max_Aki:
        with_aki = [c for c in candidates if c.get("Aki_s^-1") is not None]
        if with_aki:
            best = max(with_aki, key=lambda c: float(c["Aki_s^-1"]))  # type: ignore[arg-type]
            return best

    # Fallback: closest wavelength.
    return min(candidates, key=lambda c: abs(float(c["delta_A"])))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch NIST ASD line output (Atomic Spectra Database) and derive a small fixed baseline."
    )
    ap.add_argument(
        "--spectra",
        default="H I",
        help='Spectrum string for NIST ASD (e.g., "H I", "He I"). Default: "H I".',
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: do not download; only validate local cache files exist.",
    )
    ap.add_argument(
        "--format",
        type=int,
        choices=[0, 2, 3],
        default=3,
        help="NIST ASD output format (0=HTML, 2=CSV, 3=tab-delimited). Default: 3.",
    )
    ap.add_argument(
        "--no-ritz",
        action="store_true",
        help="Do not request Ritz values in output (show_calc_wl=0).",
    )
    ap.add_argument(
        "--no-unc",
        action="store_true",
        help="Do not request uncertainties in output (unc_out=0).",
    )
    args = ap.parse_args(argv)

    root = _repo_root()
    spectra = str(args.spectra).strip()
    spectra_token = _sanitize_token(spectra)
    src_dir = root / "data" / "quantum" / "sources" / f"nist_asd_{spectra_token}_lines"
    src_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: Avoid passing output_type/unit to lines1.pl. As of 2026-01-25, some combinations
    # trigger a server-side 500 ("undefined value as an ARRAY reference").
    q: dict[str, str] = {
        "spectra": spectra,
        "allowed_out": "1",
        "forbid_out": "1",
        "format": str(int(args.format)),
        "show_obs_wl": "1",
    }
    if not args.no_ritz:
        q["show_calc_wl"] = "1"
    if not args.no_unc:
        q["unc_out"] = "1"

    url = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?" + urlencode(q)
    ext = {0: "html", 2: "csv", 3: "tsv"}[int(args.format)]
    file_name = f"nist_asd_lines__{spectra_token}__format{int(args.format)}__obs"
    if not args.no_ritz:
        file_name += "_ritz"
    if not args.no_unc:
        file_name += "_unc"
    file_name += f".{ext}"

    raw_path = src_dir / file_name
    extracted_path = src_dir / "extracted_values.json"
    manifest_path = src_dir / "manifest.json"

    if args.offline:
        missing: list[Path] = []
        for p in (raw_path, extracted_path, manifest_path):
            if not p.exists():
                missing.append(p)
        if missing:
            raise SystemExit("[fail] missing cache files:\n" + "\n".join(f"- {p}" for p in missing))
        print("[ok] offline check passed")
        return 0

    _download(url, raw_path)

    extracted: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "NIST Atomic Spectra Database (ASD) line output",
        "spectra": spectra,
        "query": q,
        "query_url": url,
        "raw_file": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "notes": [
            "The NIST ASD 'lines1.pl' output uses obs_nu(A) (negative) in this configuration. "
            "We convert via lambda_vac_A = -1/obs_nu(A).",
            "This extracted_values.json is meant as an offline-stable baseline (fixed targets), not a full theory derivation.",
        ],
    }

    # Only parse TSV/tab-delimited in this first implementation.
    if int(args.format) != 3:
        extracted["warning"] = "format!=3: extracted_values.json only records the download; no row parsing performed."
        extracted_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        header, rows = _read_tsv(raw_path)
        extracted["columns"] = header
        extracted["row_count"] = len(rows)

    # Default baseline targets (Hydrogen; vacuum wavelengths in Å).
    # For other spectra, this selection may be empty until extended.
    targets: list[TargetLine] = []
    if spectra.strip().lower() in ("h i", "h 1", "h1", "hydrogen i", "hydrogen 1"):
        targets = [
            TargetLine(id="H_I_Lyα", approx_lambda_vac_A=1215.67, window_A=1.0),
            TargetLine(id="H_I_Hα", approx_lambda_vac_A=6564.61, window_A=3.0),
            TargetLine(id="H_I_Hβ", approx_lambda_vac_A=4862.68, window_A=3.0),
            TargetLine(id="H_I_Hγ", approx_lambda_vac_A=4341.69, window_A=3.0),
        ]
    elif spectra.strip().lower() in ("he i", "he 1", "he1", "helium i", "helium 1"):
        # Neutral helium (multi-electron minimal baseline; visible lines; vacuum wavelengths in Å).
        targets = [
            TargetLine(id="He_I_447.27nm", approx_lambda_vac_A=4472.735, window_A=5.0),
            TargetLine(id="He_I_501.71nm", approx_lambda_vac_A=5017.077, window_A=5.0),
            TargetLine(id="He_I_587.73nm", approx_lambda_vac_A=5877.250, window_A=5.0),
            TargetLine(id="He_I_668.00nm", approx_lambda_vac_A=6679.995, window_A=5.0),
        ]

        selected: list[dict[str, Any]] = []
        for t in targets:
            best = _pick_line_for_target(rows, t)
            if best is None:
                continue
            lam_A = float(best["lambda_vac_A"])
            nu = float(best["nu_obs_invA"])
            unc_nu = best.get("nu_obs_unc_invA")
            lam_unc_A = None
            if unc_nu is not None and nu != 0.0:
                lam_unc_A = abs(float(unc_nu)) / (nu * nu)

            best_out = {
                "id": t.id,
                "approx_lambda_vac_A": float(t.approx_lambda_vac_A),
                "selected": {
                    "lambda_vac_A": lam_A,
                    "lambda_vac_nm": lam_A / 10.0,
                    "lambda_vac_unc_A": lam_unc_A,
                    "nu_obs_invA": nu,
                    "nu_obs_unc_invA": unc_nu,
                    "Aki_s^-1": best.get("Aki_s^-1"),
                    "Acc": best.get("Acc"),
                    "Type": best.get("Type"),
                    "ritz_nu_invA": best.get("ritz_nu_invA"),
                    "ritz_unc_invA": best.get("ritz_unc_invA"),
                },
            }
            selected.append(best_out)

        extracted["selected_lines"] = selected
        extracted_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] extracted: {extracted_path}")

    manifest = {
        "generated_utc": extracted["generated_utc"],
        "dataset": f"NIST ASD line output: {spectra}",
        "notes": [
            "Cache for offline-repro analysis in Phase 7 / Step 7.12 (atomic baseline).",
            "If you add more spectra or change query parameters, update the manuscript and regenerate paper outputs.",
        ],
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
                "bytes": int(extracted_path.stat().st_size) if extracted_path.exists() else 0,
                "sha256": _sha256(extracted_path) if extracted_path.exists() else None,
                "derived_from": [raw_path.name],
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
