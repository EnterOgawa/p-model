from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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
    with urlopen(req, timeout=30) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    # 条件分岐: `out_path.stat().st_size == 0` を満たす経路を評価する。

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


# 関数: `_extract_first_float` の入出力契約と処理意図を定義する。

def _extract_first_float(html: str, *, pattern: str, label: str) -> float:
    m = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"missing {label}")

    return float(m.group(1))


# 関数: `_extract_linear_cij` の入出力契約と処理意図を定義する。

def _extract_linear_cij(html: str, *, ij: str) -> dict[str, float]:
    """
    Parse lines like:
      C<sub>11</sub> &asymp;16.38 - 1.28&middot;10<sup>-3</sup>T
    """
    ij = str(ij)
    m = re.search(
        rf"C<sub>{re.escape(ij)}</sub>\s*&asymp;\s*([0-9.]+)\s*-\s*([0-9.]+)\s*&middot;\s*10<sup>\s*([-+]?\d+)\s*</sup>\s*T",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"missing linear C{ij}(T) formula")

    intercept = float(m.group(1))
    coef = float(m.group(2))
    exp = int(m.group(3))
    slope = -coef * (10.0**exp)
    return {"intercept_1e11_dyn_cm2": float(intercept), "slope_1e11_dyn_cm2_per_K": float(slope)}


# 関数: `_decode_phonon_label` の入出力契約と処理意図を定義する。

def _decode_phonon_label(label_html: str) -> str:
    s = str(label_html)
    # Entities used in the Ioffe table.
    s = s.replace("&nu;", "ν").replace("&Gamma;", "Γ").replace("&nbsp;", "")
    # Preserve subscripts as an underscore-delimited label.
    s = s.replace("<sub>", "_").replace("</sub>", "")
    # Drop remaining tags/spaces.
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", "", s)
    return s


# 関数: `_extract_phonon_frequencies` の入出力契約と処理意図を定義する。

def _extract_phonon_frequencies(html: str) -> dict[str, Any]:
    """
    Extract the small phonon-frequency table from Ioffe mechanic.html.
    The table lists a few representative phonon frequencies in 10^12 Hz.
    """
    m = re.search(
        r"<h3>\s*Phonon\s+frequencies.*?</h3>\s*<table>(.*?)</table>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError("missing phonon frequencies table")

    table = m.group(1)

    rows: list[dict[str, Any]] = []

    # Exact SI definitions (2019+), consistent with CODATA 2022.
    h_J_s = 6.626_070_15e-34
    k_B_J_K = 1.380_649e-23
    h_over_kb_K_s = h_J_s / k_B_J_K

    for mr in re.finditer(
        r"<tr>\s*<td>.*?</td>\s*<td>\s*(.*?)\s*</td>\s*<td>\s*([0-9.]+)\s*10<sup>\s*12\s*</sup>\s*Hz\s*</td>\s*<td[^>]*>\s*(.*?)\s*</td>",
        table,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        label_html = mr.group(1)
        nu_thz = float(mr.group(2))
        remarks_html = mr.group(3)

        label = _decode_phonon_label(label_html)
        remarks = re.sub(r"<[^>]+>", "", str(remarks_html)).replace("&nbsp;", "").strip()

        mode = None
        point = None
        mm = re.match(r"ν_([^()]+)\(([^)]+)\)", label)
        # 条件分岐: `mm` を満たす経路を評価する。
        if mm:
            mode = mm.group(1)
            point = mm.group(2)

        nu_hz = nu_thz * 1e12
        theta_k = h_over_kb_K_s * nu_hz
        rows.append(
            {
                "label": label,
                "mode": mode,
                "point": point,
                "nu_1e12_Hz": float(nu_thz),
                "theta_K": float(theta_k),
                "remarks": remarks if remarks else None,
            }
        )

    # 条件分岐: `not rows` を満たす経路を評価する。

    if not rows:
        raise ValueError("phonon frequencies table parsed but no rows found")

    return {
        "units": {"nu": "1e12 Hz", "theta": "K"},
        "notes": [
            "Phonon frequencies are listed on the Ioffe page under 'Phonon frequencies (in units of 10^12 Hz)'.",
            "Einstein temperature is computed as θ=hν/k_B using exact SI constants (h, k_B).",
        ],
        "rows": rows,
    }


# 関数: `_parse_mechanic_html` の入出力契約と処理意図を定義する。

def _parse_mechanic_html(html: str) -> dict[str, Any]:
    # Keep HTML entities (e.g., &middot;, &asymp;, &lt;) intact for robust regex parsing.
    txt = html

    bulk_modulus = _extract_first_float(
        txt,
        pattern=r"Bulk\s+moduluss</td>\s*<td>\s*([0-9.]+)\s*&middot;\s*10<sup>\s*11\s*</sup>\s*dyn/cm<sup>2</sup>",
        label="bulk modulus row",
    )
    density = _extract_first_float(
        txt,
        pattern=r"Density</td>\s*<td>\s*([0-9.]+)\s*g/cm<sup>3</sup>",
        label="density row",
    )

    c11 = _extract_first_float(
        txt,
        pattern=r"C<sub>\s*11\s*</sub>\s*=\s*([0-9.]+)\s*&middot;\s*10<sup>\s*11\s*</sup>\s*dyn/cm<sup>2</sup>",
        label="C11 scalar",
    )
    c12 = _extract_first_float(
        txt,
        pattern=r"C<sub>\s*12\s*</sub>\s*=\s*([0-9.]+)\s*&middot;\s*10<sup>\s*11\s*</sup>\s*dyn/cm<sup>2</sup>",
        label="C12 scalar",
    )
    c44 = _extract_first_float(
        txt,
        pattern=r"C<sub>\s*44\s*</sub>\s*=\s*([0-9.]+)\s*&middot;\s*10<sup>\s*11\s*</sup>\s*dyn/cm<sup>2</sup>",
        label="C44 scalar",
    )

    m_range = re.search(r"For\s*([0-9.]+)\s*K\s*&lt;\s*T\s*&lt;\s*([0-9.]+)\s*K", txt, flags=re.IGNORECASE)
    # 条件分岐: `not m_range` を満たす経路を評価する。
    if not m_range:
        raise ValueError("missing temperature range header for linear Cij(T)")

    t_min = float(m_range.group(1))
    t_max = float(m_range.group(2))

    linear = {
        "T_range_K": {"min": float(t_min), "max": float(t_max)},
        "C11": _extract_linear_cij(txt, ij="11"),
        "C12": _extract_linear_cij(txt, ij="12"),
        "C44": _extract_linear_cij(txt, ij="44"),
    }

    # Derived (assuming cubic crystal): B = (C11 + 2*C12)/3 in the same units.
    bulk_from_cij = (float(c11) + 2.0 * float(c12)) / 3.0

    return {
        "material": "Silicon (Si)",
        "units": {
            "Cij": "1e11 dyn/cm^2",
            "bulk_modulus": "1e11 dyn/cm^2",
            "density": "g/cm^3",
            "note": "1 dyn/cm^2 = 0.1 Pa, so 1e11 dyn/cm^2 = 10 GPa.",
        },
        "values": {
            "bulk_modulus_1e11_dyn_cm2": float(bulk_modulus),
            "bulk_modulus_from_C11_C12_1e11_dyn_cm2": float(bulk_from_cij),
            "density_g_cm3": float(density),
            "C11_1e11_dyn_cm2": float(c11),
            "C12_1e11_dyn_cm2": float(c12),
            "C44_1e11_dyn_cm2": float(c44),
        },
        "temperature_dependence_linear": linear,
        "phonon_frequencies": _extract_phonon_frequencies(txt),
        "notes": [
            "This cache is used as an independent reference for bulk modulus B(T) via B=(C11+2*C12)/3.",
            "The page provides a linear Cij(T) approximation for 400K<T<873K; outside that range, this project treats B as a constant (see analysis script).",
            "The underlying sources are referenced on the page (McSkimin 1953; Nikanorov et al. 1971) via the local cached reference.html.",
        ],
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch Ioffe semiconductor database page for silicon mechanical properties and extract "
            "elastic constants (C11,C12,C44) and a linear temperature dependence approximation. "
            "Writes manifest.json and extracted_values.json under data/quantum/sources/."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--out-dirname",
        default="ioffe_silicon_mechanical_properties",
        help="Output directory name under data/quantum/sources/.",
    )
    args = ap.parse_args()

    root = _repo_root()
    src_dir = root / "data" / "quantum" / "sources" / str(args.out_dirname)
    src_dir.mkdir(parents=True, exist_ok=True)

    url_page = "https://www.ioffe.ru/SVA/NSM/Semicond/Si/mechanic.html"
    url_ref = "https://www.ioffe.ru/SVA/NSM/Semicond/Si/reference.html"
    page_path = src_dir / "ioffe_mechanic.html"
    ref_path = src_dir / "ioffe_reference.html"

    # 条件分岐: `not args.offline` を満たす経路を評価する。
    if not args.offline:
        _download(url_page, page_path)
        _download(url_ref, ref_path)

    missing: list[Path] = []
    for p in [page_path, ref_path]:
        # 条件分岐: `not p.exists() or p.stat().st_size == 0` を満たす経路を評価する。
        if not p.exists() or p.stat().st_size == 0:
            missing.append(p)

    # 条件分岐: `missing` を満たす経路を評価する。

    if missing:
        raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

    html = page_path.read_text(encoding="utf-8", errors="replace")
    extracted = _parse_mechanic_html(html)

    out_extracted = src_dir / "extracted_values.json"
    out_extracted.write_text(
        json.dumps(
            {
                "generated_utc": _iso_utc_now(),
                "dataset": "Phase 7 / Step 7.14 silicon elastic constants (Ioffe semiconductor database)",
                "source": {"url": url_page, "local_path": str(page_path), "local_sha256": _sha256(page_path)},
                "reference_page": {"url": url_ref, "local_path": str(ref_path), "local_sha256": _sha256(ref_path)},
                **extracted,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "Phase 7 / Step 7.14 silicon elastic constants (Ioffe semiconductor database)",
        "notes": [
            "The Ioffe semiconductor database is used here as an external reference for elastic constants and a linear Cij(T) approximation.",
            "Both the main page and the local reference.html are cached for offline reproducibility.",
            "extracted_values.json is derived from the cached HTML files and is intended for offline analysis.",
        ],
        "files": [
            {
                "name": page_path.name,
                "url": url_page,
                "path": str(page_path),
                "bytes": int(page_path.stat().st_size),
                "sha256": _sha256(page_path).upper(),
            },
            {
                "name": ref_path.name,
                "url": url_ref,
                "path": str(ref_path),
                "bytes": int(ref_path.stat().st_size),
                "sha256": _sha256(ref_path).upper(),
            },
        ],
    }
    out_manifest = src_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_extracted}")
    print(f"[ok] wrote: {out_manifest}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
