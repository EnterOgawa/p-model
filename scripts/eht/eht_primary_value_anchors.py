#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fmt_float_tokens(x: float) -> List[str]:
    # 条件分岐: `not math.isfinite(float(x))` を満たす経路を評価する。
    if not math.isfinite(float(x)):
        return []

    g = f"{float(x):g}"
    out = {g}
    # also allow one-decimal form for integers (e.g. 7 -> 7.0) and vice versa
    if abs(float(x) - round(float(x))) < 1e-12:
        out.add(f"{float(x):.1f}")
    else:
        out.add(str(int(round(float(x)))))  # coarse fallback

    return [s for s in out if s and s != "nan"]


def _fmt_decimal_tokens(x: float, *, decimals: int = 2) -> List[str]:
    """Format float tokens without a coarse integer fallback.

    Intended for small signed values (e.g., δ=-0.08) where rounding to an integer would be too broad.
    """

    # 条件分岐: `not math.isfinite(float(x))` を満たす経路を評価する。
    if not math.isfinite(float(x)):
        return []

    s0 = f"{float(x):.{int(decimals)}f}"
    out = {s0}
    # 条件分岐: `"." in s0 and s0.endswith("0")` を満たす経路を評価する。
    if "." in s0 and s0.endswith("0"):
        out.add(s0.rstrip("0").rstrip("."))

    return sorted([s for s in out if s and s != "nan"], key=len, reverse=True)


def _derive_unpacked_dir(local_src: str) -> Optional[str]:
    s = str(local_src).strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None

    # 条件分岐: `s.endswith("_src.tar.gz")` を満たす経路を評価する。

    if s.endswith("_src.tar.gz"):
        return s[: -len("_src.tar.gz")]

    return None


def _iter_tex_files(root_dir: Path) -> Iterable[Path]:
    for p in root_dir.rglob("*.tex"):
        # avoid extremely deep / build directories (none expected, but keep safe)
        yield p


def _find_regex_in_files(
    *,
    root: Path,
    tex_dir: Path,
    pattern: re.Pattern[str],
    max_hits: int = 5,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for p in _iter_tex_files(tex_dir):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            # 条件分岐: `pattern.search(line)` を満たす経路を評価する。
            if pattern.search(line):
                snippet = line.strip()
                # 条件分岐: `len(snippet) > 220` を満たす経路を評価する。
                if len(snippet) > 220:
                    snippet = snippet[:217] + "..."

                hits.append(
                    {
                        "path": str(p.relative_to(root)).replace("\\", "/"),
                        "lineno": lineno,
                        "snippet": snippet,
                    }
                )
                # 条件分岐: `len(hits) >= max_hits` を満たす経路を評価する。
                if len(hits) >= max_hits:
                    return hits

    return hits


@dataclass(frozen=True)
class AnchorSpec:
    key: str
    field: str
    value: float
    sigma: Optional[float]
    units: str
    source_key: str
    regex: str
    note: str
    derived_check: Optional[Dict[str, Any]] = None


def _build_anchor_specs(eht: Dict[str, Any]) -> List[AnchorSpec]:
    objects = eht.get("objects") if isinstance(eht.get("objects"), list) else []
    # 条件分岐: `not objects` を満たす経路を評価する。
    if not objects:
        return []

    specs: List[AnchorSpec] = []
    for obj in objects:
        # 条件分岐: `not isinstance(obj, dict)` を満たす経路を評価する。
        if not isinstance(obj, dict):
            continue

        key = str(obj.get("key") or "").strip()
        # 条件分岐: `not key` を満たす経路を評価する。
        if not key:
            continue

        srcs = obj.get("source_keys") if isinstance(obj.get("source_keys"), list) else []
        srcs = [str(s).strip() for s in srcs if str(s).strip()]

        # 条件分岐: `key == "sgra"` を満たす経路を評価する。
        if key == "sgra":
            # EHT Sgr A* Paper I (TeX is available via arXiv source).
            if "eht_sgra_paper1_2022" in srcs:
                ring = float(obj.get("ring_diameter_uas", float("nan")))
                ring_sig = float(obj.get("ring_diameter_uas_sigma", float("nan")))
                # 条件分岐: `math.isfinite(ring) and math.isfinite(ring_sig) and ring_sig >= 0` を満たす経路を評価する。
                if math.isfinite(ring) and math.isfinite(ring_sig) and ring_sig >= 0:
                    # matches patterns like: 51.8 \pm 2.3\,\uas  or  51.8 \pm 2.3\,\mu{\rm as}
                    v_tokens = _fmt_float_tokens(ring)
                    s_tokens = _fmt_float_tokens(ring_sig)
                    alts = [rf"{v}\s*\\pm\s*{s}" for v in v_tokens for s in s_tokens]
                    specs.append(
                        AnchorSpec(
                            key=key,
                            field="ring_diameter_uas",
                            value=ring,
                            sigma=ring_sig,
                            units="µas",
                            source_key="eht_sgra_paper1_2022",
                            regex=r"(" + "|".join(alts) + r")",
                            note="ring diameter (EHT Paper I, credible interval)",
                        )
                    )

                sh = float(obj.get("shadow_diameter_uas", float("nan")))
                sh_sig = float(obj.get("shadow_diameter_uas_sigma", float("nan")))
                # 条件分岐: `math.isfinite(sh) and math.isfinite(sh_sig) and sh_sig >= 0` を満たす経路を評価する。
                if math.isfinite(sh) and math.isfinite(sh_sig) and sh_sig >= 0:
                    v_tokens = _fmt_float_tokens(sh)
                    s_tokens = _fmt_float_tokens(sh_sig)
                    alts = [rf"{v}\s*\\pm\s*{s}" for v in v_tokens for s in s_tokens]
                    specs.append(
                        AnchorSpec(
                            key=key,
                            field="shadow_diameter_uas",
                            value=sh,
                            sigma=sh_sig,
                            units="µas",
                            source_key="eht_sgra_paper1_2022",
                            regex=r"(" + "|".join(alts) + r")",
                            note="shadow diameter (EHT-only inference; model-dependent)",
                        )
                    )

                # δ (fractional shadow deviation; Paper I text)

                dv = float(obj.get("delta_schwarzschild_vlti", float("nan")))
                dv_p = float(obj.get("delta_schwarzschild_vlti_sigma_plus", float("nan")))
                dv_m = float(obj.get("delta_schwarzschild_vlti_sigma_minus", float("nan")))
                # 条件分岐: `math.isfinite(dv) and math.isfinite(dv_p) and math.isfinite(dv_m)` を満たす経路を評価する。
                if math.isfinite(dv) and math.isfinite(dv_p) and math.isfinite(dv_m):
                    dv_t = _fmt_decimal_tokens(dv, decimals=2)
                    dvp_t = _fmt_decimal_tokens(dv_p, decimals=2)
                    dvm_t = _fmt_decimal_tokens(-abs(dv_m), decimals=2)
                    specs.append(
                        AnchorSpec(
                            key=key,
                            field="delta_schwarzschild_vlti",
                            value=dv,
                            sigma=max(abs(dv_p), abs(dv_m)),
                            units="dimensionless",
                            source_key="eht_sgra_paper1_2022",
                            regex=(
                                r"("
                                + "|".join(re.escape(t) for t in dv_t)
                                + r")\^\{\+("
                                + "|".join(re.escape(t) for t in dvp_t)
                                + r")\}_\{("
                                + "|".join(re.escape(t) for t in dvm_t)
                                + r")\}"
                            ),
                            note="shadow deviation δ (VLTI prior; Paper I text)",
                        )
                    )

                dk = float(obj.get("delta_schwarzschild_keck", float("nan")))
                dk_p = float(obj.get("delta_schwarzschild_keck_sigma_plus", float("nan")))
                dk_m = float(obj.get("delta_schwarzschild_keck_sigma_minus", float("nan")))
                # 条件分岐: `math.isfinite(dk) and math.isfinite(dk_p) and math.isfinite(dk_m)` を満たす経路を評価する。
                if math.isfinite(dk) and math.isfinite(dk_p) and math.isfinite(dk_m):
                    dk_t = _fmt_decimal_tokens(dk, decimals=2)
                    dkp_t = _fmt_decimal_tokens(dk_p, decimals=2)
                    dkm_t = _fmt_decimal_tokens(-abs(dk_m), decimals=2)
                    specs.append(
                        AnchorSpec(
                            key=key,
                            field="delta_schwarzschild_keck",
                            value=dk,
                            sigma=max(abs(dk_p), abs(dk_m)),
                            units="dimensionless",
                            source_key="eht_sgra_paper1_2022",
                            regex=(
                                r"("
                                + "|".join(re.escape(t) for t in dk_t)
                                + r")\^\{\+("
                                + "|".join(re.escape(t) for t in dkp_t)
                                + r")\}_\{("
                                + "|".join(re.escape(t) for t in dkm_t)
                                + r")\}"
                            ),
                            note="shadow deviation δ (Keck prior; Paper I text)",
                        )
                    )

                dmin = float(obj.get("delta_kerr_min", float("nan")))
                dmax = float(obj.get("delta_kerr_max", float("nan")))
                # 条件分岐: `math.isfinite(dmin) and math.isfinite(dmax)` を満たす経路を評価する。
                if math.isfinite(dmin) and math.isfinite(dmax):
                    dmin_t = _fmt_decimal_tokens(dmin, decimals=2)
                    dmax_t = _fmt_decimal_tokens(dmax, decimals=2)
                    specs.append(
                        AnchorSpec(
                            key=key,
                            field="delta_kerr_range",
                            value=dmin,
                            sigma=dmax,
                            units="dimensionless",
                            source_key="eht_sgra_paper1_2022",
                            regex=(
                                r"("
                                + "|".join(re.escape(t) for t in dmin_t)
                                + r")\s*\\leq\s*\\delta\s*\\leq\s*("
                                + "|".join(re.escape(t) for t in dmax_t)
                                + r")"
                            ),
                            note="Kerr range for δ (Paper I text)",
                        )
                    )

                # Table ranges: fractional width W/d and brightness asymmetry A.

                w_lo = float(obj.get("ring_fractional_width_min", float("nan")))
                w_hi = float(obj.get("ring_fractional_width_max", float("nan")))
                # 条件分岐: `math.isfinite(w_lo) and math.isfinite(w_hi) and 0 < w_lo < w_hi < 1.0` を満たす経路を評価する。
                if math.isfinite(w_lo) and math.isfinite(w_hi) and 0 < w_lo < w_hi < 1.0:
                    plo = int(round(100.0 * w_lo))
                    phi = int(round(100.0 * w_hi))
                    specs.append(
                        AnchorSpec(
                            key=key,
                            field="ring_fractional_width_range",
                            value=w_lo,
                            sigma=w_hi,
                            units="fraction",
                            source_key="eht_sgra_paper1_2022",
                            regex=rf"{plo}\s*[-–]\s*{phi}\s*\\%",
                            note="fractional width W/d range (~30–50%) in Table (approx)",
                        )
                    )

                a_lo = float(obj.get("ring_brightness_asymmetry_min", float("nan")))
                a_hi = float(obj.get("ring_brightness_asymmetry_max", float("nan")))
                # 条件分岐: `math.isfinite(a_lo) and math.isfinite(a_hi) and 0 <= a_lo < a_hi` を満たす経路を評価する。
                if math.isfinite(a_lo) and math.isfinite(a_hi) and 0 <= a_lo < a_hi:
                    specs.append(
                        AnchorSpec(
                            key=key,
                            field="ring_brightness_asymmetry_range",
                            value=a_lo,
                            sigma=a_hi,
                            units="dimensionless",
                            source_key="eht_sgra_paper1_2022",
                            regex=rf"{a_lo:g}\s*[-–]\s*{a_hi:g}",
                            note="brightness asymmetry A range (~0.04–0.3) in Table (approx)",
                        )
                    )

                scat = obj.get("scattering_kernel_fwhm_uas")
                # 条件分岐: `scat is not None` を満たす経路を評価する。
                if scat is not None:
                    try:
                        scat_v = float(scat)
                    except Exception:
                        scat_v = float("nan")

                    # 条件分岐: `math.isfinite(scat_v) and scat_v > 0` を満たす経路を評価する。

                    if math.isfinite(scat_v) and scat_v > 0:
                        v_tokens = _fmt_float_tokens(scat_v)
                        # prefer contexts with FWHM
                        alts = [rf"FWHM.*{v}" for v in v_tokens]
                        specs.append(
                            AnchorSpec(
                                key=key,
                                field="scattering_kernel_fwhm_uas",
                                value=scat_v,
                                sigma=None,
                                units="µas",
                                source_key="eht_sgra_paper1_2022",
                                regex=r"(" + "|".join(alts) + r")",
                                note="representative blurred thin-ring comparison (not a fitted kernel)",
                            )
                        )

            # Johnson+2018 (TeX): anchor the *coefficients* and compare against the 230 GHz (λ=0.13 cm) derived µas values.

            if "johnson_scattering_2018" in srcs:
                lam_cm = 0.13  # 230 GHz ≈ 1.3 mm
                # Paper values: theta_maj = (1.380 ± 0.013) λ_cm^2 mas, theta_min = (0.703 ± 0.013) λ_cm^2 mas.
                for which, coeff, coeff_sig, target_field, target_sigma_field in (
                    ("maj", 1.380, 0.013, "scattering_kernel_fwhm_major_uas", "scattering_kernel_fwhm_major_uas_sigma"),
                    ("min", 0.703, 0.013, "scattering_kernel_fwhm_minor_uas", "scattering_kernel_fwhm_minor_uas_sigma"),
                ):
                    expected_uas = float(coeff * lam_cm * lam_cm * 1000.0)
                    expected_sig_uas = float(coeff_sig * lam_cm * lam_cm * 1000.0)
                    stored_uas = float(obj.get(target_field, float("nan")))
                    stored_sig_uas = float(obj.get(target_sigma_field, float("nan")))
                    # Keep the TeX matching strict enough: coefficients are often written with 3 decimals (e.g. 1.380),
                    # while float:g drops trailing zeros (1.38). Accept both.
                    v_tokens = sorted({f"{float(coeff):g}", f"{float(coeff):.3f}"})
                    s_tokens = sorted({f"{float(coeff_sig):g}", f"{float(coeff_sig):.3f}"})
                    alts = [rf"{v}\s*\\pm\s*{s}" for v in v_tokens for s in s_tokens]
                    specs.append(
                        AnchorSpec(
                            key=key,
                            field=f"scattering_kernel_coeff_{which}_lambda_cm2_mas",
                            value=float(coeff),
                            sigma=float(coeff_sig),
                            units="mas × λ_cm^2",
                            source_key="johnson_scattering_2018",
                            regex=r"(" + "|".join(alts) + r")",
                            note="anisotropic Gaussian scattering kernel coefficient (Johnson+2018)",
                            derived_check={
                                "derived_to_230ghz_lambda_cm": lam_cm,
                                "expected_uas": expected_uas,
                                "expected_uas_sigma": expected_sig_uas,
                                "stored_uas": (None if not math.isfinite(stored_uas) else stored_uas),
                                "stored_uas_sigma": (None if not math.isfinite(stored_sig_uas) else stored_sig_uas),
                                "target_field": target_field,
                                "target_sigma_field": target_sigma_field,
                            },
                        )
                    )

                pa = float(obj.get("scattering_kernel_pa_deg", float("nan")))
                pa_sig = float(obj.get("scattering_kernel_pa_deg_sigma", float("nan")))
                # 条件分岐: `math.isfinite(pa) and math.isfinite(pa_sig) and pa_sig >= 0` を満たす経路を評価する。
                if math.isfinite(pa) and math.isfinite(pa_sig) and pa_sig >= 0:
                    v_tokens = _fmt_float_tokens(pa)
                    s_tokens = _fmt_float_tokens(pa_sig)
                    alts = [rf"{v}\s*\\pm\s*{s}" for v in v_tokens for s in s_tokens]
                    specs.append(
                        AnchorSpec(
                            key=key,
                            field="scattering_kernel_pa_deg",
                            value=pa,
                            sigma=pa_sig,
                            units="deg",
                            source_key="johnson_scattering_2018",
                            regex=r"(" + "|".join(alts) + r")",
                            note="scattering kernel position angle (Johnson+2018)",
                        )
                    )

            # Zhu+2018 (TeX): anchor refractive scattering wander/distortion/asymmetry ranges at 230 GHz.

            if "zhu_scattering_limitations_2018" in srcs:
                for field, lo_key, hi_key in (
                    ("refractive_wander_uas_range", "refractive_wander_uas_min", "refractive_wander_uas_max"),
                    ("refractive_distortion_uas_range", "refractive_distortion_uas_min", "refractive_distortion_uas_max"),
                    ("refractive_asymmetry_uas_range", "refractive_asymmetry_uas_min", "refractive_asymmetry_uas_max"),
                ):
                    lo = float(obj.get(lo_key, float("nan")))
                    hi = float(obj.get(hi_key, float("nan")))
                    # 条件分岐: `not (math.isfinite(lo) and math.isfinite(hi) and lo >= 0 and hi >= lo)` を満たす経路を評価する。
                    if not (math.isfinite(lo) and math.isfinite(hi) and lo >= 0 and hi >= lo):
                        continue

                    lo_s = f"{lo:.2f}".rstrip("0").rstrip(".")
                    hi_s = f"{hi:.2f}".rstrip("0").rstrip(".")
                    lo_re = re.escape(lo_s)
                    hi_re = re.escape(hi_s)
                    specs.append(
                        AnchorSpec(
                            key=key,
                            field=field,
                            value=lo,
                            sigma=hi,
                            units="µas",
                            source_key="zhu_scattering_limitations_2018",
                            regex=rf"{lo_re}.*{hi_re}",
                            note="refractive scattering (mean vs alternative model max) at 230 GHz (Zhu+2018)",
                        )
                    )

    return specs


def main() -> int:
    root = _repo_root()
    inp = root / "data" / "eht" / "eht_black_holes.json"
    out_dir = root / "output" / "private" / "eht"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "eht_primary_value_anchors.json"

    eht = _read_json(inp)
    sources = eht.get("sources") if isinstance(eht.get("sources"), dict) else {}

    tex_sources: Dict[str, Path] = {}
    for sk, sv in sources.items():
        # 条件分岐: `not (isinstance(sk, str) and isinstance(sv, dict))` を満たす経路を評価する。
        if not (isinstance(sk, str) and isinstance(sv, dict)):
            continue

        local_src = sv.get("local_src")
        # 条件分岐: `not isinstance(local_src, str) or not local_src.strip()` を満たす経路を評価する。
        if not isinstance(local_src, str) or not local_src.strip():
            continue

        unpacked = _derive_unpacked_dir(local_src)
        # 条件分岐: `not unpacked` を満たす経路を評価する。
        if not unpacked:
            continue

        tex_dir = root / Path(unpacked)
        # 条件分岐: `tex_dir.exists() and tex_dir.is_dir()` を満たす経路を評価する。
        if tex_dir.exists() and tex_dir.is_dir():
            tex_sources[sk] = tex_dir

    specs = _build_anchor_specs(eht)
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        tex_dir = tex_sources.get(spec.source_key)
        # 条件分岐: `not isinstance(tex_dir, Path)` を満たす経路を評価する。
        if not isinstance(tex_dir, Path):
            rows.append(
                {
                    "object_key": spec.key,
                    "field": spec.field,
                    "value": spec.value,
                    "sigma": spec.sigma,
                    "units": spec.units,
                    "source_key": spec.source_key,
                    "regex": spec.regex,
                    "note": spec.note,
                    "found": False,
                    "hits": [],
                    "reason": "no local TeX source directory available",
                }
            )
            continue

        pat = re.compile(spec.regex)
        hits = _find_regex_in_files(root=root, tex_dir=tex_dir, pattern=pat, max_hits=5)
        row_out: Dict[str, Any] = {
            "object_key": spec.key,
            "field": spec.field,
            "value": spec.value,
            "sigma": spec.sigma,
            "units": spec.units,
            "source_key": spec.source_key,
            "tex_dir": str(tex_dir.relative_to(root)).replace("\\", "/"),
            "regex": spec.regex,
            "note": spec.note,
            "found": len(hits) > 0,
            "hits": hits,
        }
        # 条件分岐: `spec.derived_check is not None` を満たす経路を評価する。
        if spec.derived_check is not None:
            dc = dict(spec.derived_check)
            stored = dc.get("stored_uas")
            expected = dc.get("expected_uas")
            try:
                # 条件分岐: `stored is not None and expected is not None` を満たす経路を評価する。
                if stored is not None and expected is not None:
                    dc["delta_uas__stored_minus_expected"] = float(stored) - float(expected)
            except Exception:
                pass

            row_out["derived_check"] = dc

        rows.append(row_out)

    totals = {
        "anchors_total": len(rows),
        "anchors_found": sum(1 for r in rows if r.get("found") is True),
        "anchors_missing": sum(1 for r in rows if r.get("found") is False),
        "tex_sources_available": len(tex_sources),
    }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(inp.relative_to(root)).replace("\\", "/"),
        "output": str(out_json.relative_to(root)).replace("\\", "/"),
        "tex_sources": {k: str(v.relative_to(root)).replace("\\", "/") for k, v in tex_sources.items()},
        "totals": totals,
        "rows": rows,
    }

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_primary_value_anchors",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": totals,
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] totals: {totals}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
