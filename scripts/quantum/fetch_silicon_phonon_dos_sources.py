from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0` を満たす経路を評価する。
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return

    # Some repositories (e.g., CaltechAUTHORS file storage redirects) may reject
    # non-browser user agents with 403; use a conservative browser-like UA.

    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=30) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    # 条件分岐: `out_path.stat().st_size == 0` を満たす経路を評価する。

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _parse_float(s: str) -> float:
    x = float(str(s).strip())
    # 条件分岐: `not math.isfinite(x)` を満たす経路を評価する。
    if not math.isfinite(x):
        raise ValueError(f"non-finite float: {s!r}")

    return float(x)


def _extract_textarea_table(html: str) -> list[dict[str, float]]:
    m = re.search(r"<textarea[^>]*>(.*?)</textarea>", html, flags=re.IGNORECASE | re.DOTALL)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError("missing <textarea> table with DOS data")

    rows: list[dict[str, float]] = []
    for line in m.group(1).strip().splitlines():
        parts = str(line).strip().split()
        # 条件分岐: `len(parts) < 2` を満たす経路を評価する。
        if len(parts) < 2:
            continue

        omega = _parse_float(parts[0])
        dos = _parse_float(parts[1])
        rows.append({"omega_rad_s": float(omega), "dos_per_m3_per_rad_s": float(dos)})

    # 条件分岐: `len(rows) < 10` を満たす経路を評価する。

    if len(rows) < 10:
        raise ValueError(f"too few DOS rows parsed: n={len(rows)}")

    # Ensure monotonic omega.

    rows.sort(key=lambda r: float(r["omega_rad_s"]))
    for i in range(1, len(rows)):
        # 条件分岐: `rows[i]["omega_rad_s"] <= rows[i - 1]["omega_rad_s"]` を満たす経路を評価する。
        if rows[i]["omega_rad_s"] <= rows[i - 1]["omega_rad_s"]:
            raise ValueError("omega is not strictly increasing after sort")

    return rows


def _trapz_xy(xs: list[float], ys: list[float]) -> float:
    # 条件分岐: `len(xs) != len(ys) or len(xs) < 2` を満たす経路を評価する。
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("invalid arrays for trapz")

    s = 0.0
    for i in range(1, len(xs)):
        dx = float(xs[i]) - float(xs[i - 1])
        # 条件分岐: `dx <= 0.0` を満たす経路を評価する。
        if dx <= 0.0:
            raise ValueError("xs must be strictly increasing")

        s += 0.5 * (float(ys[i - 1]) + float(ys[i])) * dx

    return float(s)


def _find_x_at_cum_fraction(xs: list[float], ys: list[float], *, frac: float) -> dict[str, float]:
    """
    Return the x value where cumulative trapezoidal integral reaches frac of the total.
    """
    # 条件分岐: `not (0.0 < float(frac) < 1.0)` を満たす経路を評価する。
    if not (0.0 < float(frac) < 1.0):
        raise ValueError("frac must be in (0,1)")

    total = _trapz_xy(xs, ys)
    target = float(frac) * float(total)
    cum = 0.0
    for i in range(1, len(xs)):
        x0 = float(xs[i - 1])
        x1 = float(xs[i])
        y0 = float(ys[i - 1])
        y1 = float(ys[i])
        dx = x1 - x0
        area = 0.5 * (y0 + y1) * dx
        # 条件分岐: `cum + area >= target and area > 0.0` を満たす経路を評価する。
        if cum + area >= target and area > 0.0:
            # Linear interpolation in x within this trapezoid using cumulative area fraction.
            # For robustness (and because the grid is fine), approximate the DOS linearly.
            # Solve for t in [0,1] such that area from x0 to x0+t*dx equals (target-cum).
            remaining = target - cum
            # area(t) = ∫_0^t (y0 + (y1-y0)u) du * dx
            #        = dx * (y0*t + 0.5*(y1-y0)*t^2)
            # Solve: 0.5*dx*(y1-y0)*t^2 + dx*y0*t - remaining = 0
            a = 0.5 * dx * (y1 - y0)
            b = dx * y0
            c = -remaining
            t: float
            # 条件分岐: `abs(a) < 1e-30` を満たす経路を評価する。
            if abs(a) < 1e-30:
                t = 0.0 if abs(b) < 1e-30 else float(-c / b)
            else:
                disc = b * b - 4.0 * a * c
                disc = max(0.0, float(disc))
                t = float((-b + math.sqrt(disc)) / (2.0 * a))

            t = min(1.0, max(0.0, float(t)))
            return {"x": float(x0 + t * dx), "cum": float(cum + remaining), "total": float(total)}

        cum += area

    return {"x": float(xs[-1]), "cum": float(total), "total": float(total)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch a public silicon phonon density-of-states (DOS) table (HTML) and extract the numeric "
            "ω–D(ω) table for offline analysis. Writes manifest.json and extracted_values.json under "
            "data/quantum/sources/."
        )
    )
    ap.add_argument(
        "--source",
        default="hadley_html",
        choices=["hadley_html", "osti_kim2015_prb91_014307", "caltechauthors_kim2015_prb91_014307"],
        help=(
            "Data source to cache. "
            "hadley_html=public numeric table proxy; "
            "osti_kim2015_prb91_014307=primary INS-based anharmonic phonon DOS study (accepted manuscript via OSTI); "
            "caltechauthors_kim2015_prb91_014307=open-access mirror of the published paper PDF (CaltechAUTHORS)."
        ),
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--out-dirname",
        default=None,
        help=(
            "Output directory name under data/quantum/sources/. "
            "If omitted, a source-specific default is used."
        ),
    )
    args = ap.parse_args()

    root = _repo_root()
    source = str(args.source)
    # 条件分岐: `source == "hadley_html"` を満たす経路を評価する。
    if source == "hadley_html":
        default_dir = "hadley_si_phonon_dos"
    # 条件分岐: 前段条件が不成立で、`source == "osti_kim2015_prb91_014307"` を追加評価する。
    elif source == "osti_kim2015_prb91_014307":
        default_dir = "osti_kim2015_prb91_014307_si_phonon_anharmonicity"
    else:
        default_dir = "caltechauthors_kim2015_prb91_014307_si_phonon_anharmonicity"

    src_dir = root / "data" / "quantum" / "sources" / str(args.out_dirname or default_dir)
    src_dir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `source == "hadley_html"` を満たす経路を評価する。
    if source == "hadley_html":
        url = "https://lampz.tugraz.at/~hadley/ss1/phonons/dos/si_phonon_dos.html"
        html_path = src_dir / "si_phonon_dos.html"

        # 条件分岐: `not args.offline` を満たす経路を評価する。
        if not args.offline:
            _download(url, html_path)

        # 条件分岐: `not html_path.exists() or html_path.stat().st_size == 0` を満たす経路を評価する。

        if not html_path.exists() or html_path.stat().st_size == 0:
            raise SystemExit(f"[fail] missing: {html_path}")

        html = html_path.read_text(encoding="utf-8", errors="replace")
        rows = _extract_textarea_table(html)
        omegas = [float(r["omega_rad_s"]) for r in rows]
        dos = [float(r["dos_per_m3_per_rad_s"]) for r in rows]

        integral = _trapz_xy(omegas, dos)
        # The page states that the integral corresponds to the phonon mode density
        # (≈3× atom density). Use the extracted integral to estimate the implied atom density.
        n_atoms_m3 = float(integral / 3.0)

        half = _find_x_at_cum_fraction(omegas, dos, frac=0.5)
        omega_half = float(half["x"])
        nu_half_thz = float(omega_half / (2.0 * math.pi) / 1e12)

        # Convert ω to Einstein temperature θ=ħω/k_B (K). Use exact SI definitions.
        h_J_s = 6.626_070_15e-34
        k_B_J_K = 1.380_649e-23
        hbar_over_kb_K_s = (h_J_s / (2.0 * math.pi)) / k_B_J_K
        theta_half_k = float(hbar_over_kb_K_s * omega_half)

        out_extracted = src_dir / "extracted_values.json"
        out_extracted.write_text(
            json.dumps(
                {
                    "generated_utc": _iso_utc_now(),
                    "dataset": "Phase 7 / Step 7.14 silicon phonon DOS (public HTML table cache)",
                    "source": {"url": url, "local_path": str(html_path), "local_sha256": _sha256(html_path)},
                    "units": {
                        "omega": "rad/s",
                        "dos": "1/(m^3·(rad/s))",
                        "note": "D(ω) is provided as an absolute density of states per unit volume on the source page (no bibliographic reference given).",
                    },
                    "derived": {
                        "integral_dos_domega_per_m3": float(integral),
                        "implied_atom_number_density_per_m3": float(n_atoms_m3),
                        "split_half_integral": {
                            "omega_rad_s": float(omega_half),
                            "nu_THz": float(nu_half_thz),
                            "theta_K": float(theta_half_k),
                            "notes": [
                                "The split is defined by ω such that ∫_0^ω D(ω') dω' equals half of the total integral.",
                                "For Si (diamond structure, 2 atoms per primitive cell), acoustic and optical branches have equal mode counts (3 and 3).",
                                "Therefore half-integral provides a robust acoustic/optical proxy split for two-group Grüneisen tests.",
                            ],
                        },
                    },
                    "rows": rows,
                    "notes": [
                        "This cache is intended as an initial numeric phonon DOS proxy to constrain mode-weighting in α(T) tests.",
                        "The source page does not provide a formal citation; treat it as a public proxy dataset and replace with an INS/IXS primary dataset when available.",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        manifest: dict[str, Any] = {
            "generated_utc": _iso_utc_now(),
            "dataset": "Phase 7 / Step 7.14 silicon phonon DOS (public HTML table cache)",
            "source": source,
            "notes": [
                "Cache the HTML containing the raw numeric ω–D(ω) table for offline reproducibility.",
                "extracted_values.json is derived from the cached HTML and contains parsed numeric arrays and basic integrals.",
            ],
            "files": [
                {
                    "name": html_path.name,
                    "url": url,
                    "path": str(html_path),
                    "bytes": int(html_path.stat().st_size),
                    "sha256": _sha256(html_path).upper(),
                }
            ],
        }
        out_manifest = src_dir / "manifest.json"
        out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"[ok] wrote: {out_extracted}")
        print(f"[ok] wrote: {out_manifest}")
        return

    # Primary (experimental) anharmonic phonon DOS reference (accepted manuscript via OSTI).
    # Kim et al. Phys. Rev. B 91, 014307 (2015). DOI: 10.1103/PhysRevB.91.014307

    if source == "osti_kim2015_prb91_014307":
        osti_id = "1185765"
        # NOTE: OSTI has two entrypoint URL styles; the /pages/ form is stable for public access.
        url_landing = f"https://www.osti.gov/pages/biblio/{osti_id}-phonon-anharmonicity-silicon-from"
        url_pdf = f"https://www.osti.gov/servlets/purl/{osti_id}"
        html_path = src_dir / f"osti_biblio_{osti_id}.html"
        pdf_path = src_dir / f"osti_purl_{osti_id}.pdf"

        # 条件分岐: `not args.offline` を満たす経路を評価する。
        if not args.offline:
            _download(url_landing, html_path)
            _download(url_pdf, pdf_path)

        missing: list[Path] = []
        for p in [html_path, pdf_path]:
            # 条件分岐: `not p.exists() or p.stat().st_size == 0` を満たす経路を評価する。
            if not p.exists() or p.stat().st_size == 0:
                missing.append(p)

        # 条件分岐: `missing` を満たす経路を評価する。

        if missing:
            raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

        extracted = {
            "generated_utc": _iso_utc_now(),
            "dataset": "Phase 7 / Step 7.14 silicon phonon anharmonicity (INS-derived phonon DOS; accepted manuscript cache)",
            "source": {
                "landing_url": url_landing,
                "pdf_url": url_pdf,
                "local_landing_html": str(html_path),
                "local_landing_html_sha256": _sha256(html_path),
                "local_pdf": str(pdf_path),
                "local_pdf_sha256": _sha256(pdf_path),
                "doi": "10.1103/PhysRevB.91.014307",
                "osti_id": osti_id,
            },
            "notes": [
                "This is a primary experimental reference (INS-based phonon DOS vs temperature) used to constrain anharmonic softening beyond static DOS proxies.",
                "Numeric extraction (e.g., ΔE/E vs T, DOS rescaling) is performed in a separate step to keep caching reproducible and robust.",
            ],
        }
        out_extracted = src_dir / "extracted_values.json"
        out_extracted.write_text(json.dumps(extracted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        manifest: dict[str, Any] = {
            "generated_utc": _iso_utc_now(),
            "dataset": extracted["dataset"],
            "source": source,
            "files": [
                {
                    "name": html_path.name,
                    "url": url_landing,
                    "path": str(html_path),
                    "bytes": int(html_path.stat().st_size),
                    "sha256": _sha256(html_path).upper(),
                },
                {
                    "name": pdf_path.name,
                    "url": url_pdf,
                    "path": str(pdf_path),
                    "bytes": int(pdf_path.stat().st_size),
                    "sha256": _sha256(pdf_path).upper(),
                },
            ],
        }
        out_manifest = src_dir / "manifest.json"
        out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"[ok] wrote: {out_extracted}")
        print(f"[ok] wrote: {out_manifest}")
        return

    # Open-access mirror of the published paper PDF via CaltechAUTHORS.
    # Kim et al. Phys. Rev. B 91, 014307 (2015). DOI: 10.1103/PhysRevB.91.014307

    if source == "caltechauthors_kim2015_prb91_014307":
        record_id = "67hvq-njk27"
        url_landing = f"https://authors.library.caltech.edu/records/{record_id}/latest"
        url_pdf_base = f"https://authors.library.caltech.edu/records/{record_id}/files/PhysRevB.91.014307.pdf?download=1"
        # The file download endpoint may be cached with a short-lived S3 presigned URL.
        # Add a cache-buster to force a fresh redirect.
        cache_buster_ts = int(datetime.now(timezone.utc).timestamp())
        url_pdf = f"{url_pdf_base}&ts={cache_buster_ts}"
        html_path = src_dir / f"caltechauthors_record_{record_id}.html"
        pdf_path = src_dir / "PhysRevB.91.014307.pdf"

        # 条件分岐: `not args.offline` を満たす経路を評価する。
        if not args.offline:
            _download(url_landing, html_path)
            _download(url_pdf, pdf_path)

        missing: list[Path] = []
        for p in [html_path, pdf_path]:
            # 条件分岐: `not p.exists() or p.stat().st_size == 0` を満たす経路を評価する。
            if not p.exists() or p.stat().st_size == 0:
                missing.append(p)

        # 条件分岐: `missing` を満たす経路を評価する。

        if missing:
            raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

        extracted = {
            "generated_utc": _iso_utc_now(),
            "dataset": "Phase 7 / Step 7.14 silicon phonon anharmonicity (INS-derived phonon DOS vs temperature; published PDF mirror cache)",
            "source": {
                "landing_url": url_landing,
                "pdf_url_base": url_pdf_base,
                "pdf_url": url_pdf,
                "cache_buster_ts": cache_buster_ts,
                "local_landing_html": str(html_path),
                "local_landing_html_sha256": _sha256(html_path),
                "local_pdf": str(pdf_path),
                "local_pdf_sha256": _sha256(pdf_path),
                "doi": "10.1103/PhysRevB.91.014307",
                "record_id": record_id,
            },
            "notes": [
                "This is an open-access mirror of the published paper PDF (CaltechAUTHORS) and complements the OSTI accepted manuscript cache.",
                "It may contain higher-quality figures (vector or higher resolution) useful for extracting temperature-dependent phonon DOS/softening constraints.",
            ],
        }
        out_extracted = src_dir / "extracted_values.json"
        out_extracted.write_text(json.dumps(extracted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        manifest: dict[str, Any] = {
            "generated_utc": _iso_utc_now(),
            "dataset": extracted["dataset"],
            "source": source,
            "files": [
                {
                    "name": html_path.name,
                    "url": url_landing,
                    "path": str(html_path),
                    "bytes": int(html_path.stat().st_size),
                    "sha256": _sha256(html_path).upper(),
                },
                {
                    "name": pdf_path.name,
                    "url": url_pdf,
                    "path": str(pdf_path),
                    "bytes": int(pdf_path.stat().st_size),
                    "sha256": _sha256(pdf_path).upper(),
                },
            ],
        }
        out_manifest = src_dir / "manifest.json"
        out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"[ok] wrote: {out_extracted}")
        print(f"[ok] wrote: {out_manifest}")
        return

    raise SystemExit(f"[fail] unsupported source: {source!r}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
