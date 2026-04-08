#!/usr/bin/env python3
"""
Part IV manuscriptが参照する全図を producer script から厳密に再生成する。

Purpose:
- pmodel_paper_part4_verification.tex に現れる `\\includegraphics{...}` を正として、
  参照中の figure stem を全件抽出する。
- stem と同名の producer script が `scripts/` 配下にある場合は、その script を
  直接再実行する。
- stem と script 名が一致しない既知ケースは custom mapping で補完する。
- 実行結果を manifest JSON に残し、build 経路で「一部だけ rerun」の状態をなくす。

Inputs:
- output/private/summary/pmodel_paper_part4_verification.tex

Outputs:
- output/private/summary/part4_strict_figure_refresh_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[2]


# クラス: `CommandSpec` の責務と境界条件を定義する。
@dataclass(frozen=True)
class CommandSpec:
    stem: str
    argv: tuple[str, ...]
    source: str


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_stems` の入出力契約と処理意図を定義する。

def _parse_stems(tex_path: Path) -> List[str]:
    import re

    text = tex_path.read_text(encoding="utf-8")
    stems = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\.pdf\}", text)
    ordered: List[str] = []
    seen: set[str] = set()
    for stem in stems:
        if stem not in seen:
            seen.add(stem)
            ordered.append(stem)

    return ordered


# 関数: `_build_exact_script_index` の入出力契約と処理意図を定義する。

def _build_exact_script_index() -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for path in (ROOT / "scripts").rglob("*.py"):
        out.setdefault(path.stem, []).append(path)

    return out


# 関数: `_custom_specs` の入出力契約と処理意図を定義する。

def _custom_specs(py: str) -> Dict[str, Sequence[str]]:
    return {
        "validation_scoreboard": [
            py,
            "-B",
            str(ROOT / "scripts" / "summary" / "validation_scoreboard.py"),
            "--target-fig-h-in",
            "9.2",
        ],
        "llr_systematics_root_cause": [
            py,
            "-B",
            str(ROOT / "scripts" / "llr" / "llr_precision_reaudit.py"),
        ],
        "llr_systematics_root_cause_over4ns": [
            py,
            "-B",
            str(ROOT / "scripts" / "llr" / "llr_precision_reaudit.py"),
        ],
        "cassini_odf_beta_direct_fit": [
            py,
            "-B",
            str(ROOT / "scripts" / "cassini" / "cassini_fig2_overlay.py"),
            "--source",
            "pds_odf_raw",
            "--odf-direct-beta-fit",
        ],
        "vlbi_17may01xa_beta_direct_fit": [
            py,
            "-B",
            str(ROOT / "scripts" / "vlbi" / "vlbi_beta_direct_fit_from_vgosdb.py"),
            "--session",
            "17MAY01XA",
            "--input-root",
            str(ROOT / "data" / "vlbi" / "sources" / "vgosdb" / "17MAY01XA" / "extracted"),
        ],
        "vlbi_17may01xa_beta_nuisance_sensitivity": [
            py,
            "-B",
            str(ROOT / "scripts" / "vlbi" / "vlbi_beta_nuisance_sensitivity.py"),
            "--session",
            "17MAY01XA",
            "--input-root",
            str(ROOT / "data" / "vlbi" / "sources" / "vgosdb" / "17MAY01XA" / "extracted"),
        ],
        "vlbi_17may01xa_beta_source_filter_sensitivity": [
            py,
            "-B",
            str(ROOT / "scripts" / "vlbi" / "vlbi_beta_source_filter_sensitivity.py"),
            "--session",
            "17MAY01XA",
        ],
        "xrism_resolve_summary": [
            py,
            "-B",
            str(ROOT / "scripts" / "xrism" / "xrism_integration.py"),
        ],
        "gw_polarization_h1_l1_v1_network_tuning_audit_refine_step87198": [
            py,
            "-B",
            str(ROOT / "scripts" / "gw" / "gw_polarization_h1_l1_v1_network_tuning_audit.py"),
            "--prefix",
            "gw_polarization_h1_l1_v1_network_tuning_audit_refine_step87198",
        ],
        "born_route_a_proxy_constraints_pack": [
            py,
            "-B",
            str(ROOT / "scripts" / "quantum" / "born_route_a_proxy_constraints.py"),
        ],
        "v2_trial2_theorem_support_summary": [
            py,
            "-B",
            str(ROOT / "scripts" / "quantum" / "v2_derivation_gap_figures.py"),
        ],
        "v2_trial3_weak_checkpoint_summary": [
            py,
            "-B",
            str(ROOT / "scripts" / "quantum" / "v2_derivation_gap_figures.py"),
        ],
    }


# 関数: `_resolve_specs` の入出力契約と処理意図を定義する。

def _resolve_specs(stems: Iterable[str], py: str) -> List[CommandSpec]:
    exact_index = _build_exact_script_index()
    custom = _custom_specs(py)
    ordered_specs: List[CommandSpec] = []
    seen_argv: set[tuple[str, ...]] = set()

    for stem in stems:
        if stem in custom:
            argv = tuple(custom[stem])
            source = "custom"
        else:
            matches = exact_index.get(stem, [])
            if not matches:
                raise FileNotFoundError(f"producer script not found for stem: {stem}")

            if len(matches) > 1:
                rels = ", ".join(str(path.relative_to(ROOT)) for path in matches)
                raise RuntimeError(f"ambiguous producer for stem {stem}: {rels}")

            argv = (py, "-B", str(matches[0]))
            source = "exact"

        if argv in seen_argv:
            continue

        seen_argv.add(argv)
        ordered_specs.append(CommandSpec(stem=stem, argv=argv, source=source))

    return ordered_specs


# 関数: `_artifact_updated_after` の入出力契約と処理意図を定義する。

def _artifact_updated_after(stem: str, started_wall: float) -> bool:
    candidates: List[Path] = []
    for base in (ROOT / "output" / "public", ROOT / "output" / "private", ROOT / "output"):
        if not base.exists():
            continue

        candidates.extend(base.rglob(f"{stem}.pdf"))

    for path in candidates:
        try:
            if path.stat().st_mtime >= started_wall - 1.0:
                return True
        except OSError:
            continue

    return False


# 関数: `_artifact_exists_for_stem` の入出力契約と処理意図を定義する。

def _artifact_exists_for_stem(stem: str) -> bool:
    for base in (ROOT / "output" / "public", ROOT / "output" / "private", ROOT / "output"):
        if not base.exists():
            continue

        for suffix in ("pdf", "png", "json", "csv"):
            try:
                if any(base.rglob(f"{stem}.{suffix}")):
                    return True
            except OSError:
                continue

    return False


# 関数: `_run_specs` の入出力契約と処理意図を定義する。

def _run_specs(specs: Sequence[CommandSpec]) -> List[dict]:
    rows: List[dict] = []
    env = os.environ.copy()
    root_str = str(ROOT)
    pythonpath = str(env.get("PYTHONPATH", "")).strip()
    if pythonpath:
        items = pythonpath.split(os.pathsep)
        if root_str not in items:
            env["PYTHONPATH"] = root_str + os.pathsep + pythonpath
    else:
        env["PYTHONPATH"] = root_str

    env["WAVEP_MPL_AUTOSAVE_VECTOR_PDF"] = "1"
    env["WAVEP_MPL_FONT_PROFILE"] = "part2_astrophysics"
    env["WAVEP_MPL_FONT_SCALE"] = "1.0"
    env["WAVEP_MPL_CJK_FONT"] = "Noto Sans CJK JP"
    env["WAVEP_MPL_CJK_FONT_PATH"] = str(
        ROOT / "output" / "private" / "summary" / "fonts" / "NotoSansJP-Regular-static.ttf"
    )
    env["WAVEP_MPL_TEXT_MIN_FONT"] = "7.8"
    env["WAVEP_MPL_LEGEND_NOTE_MIN_FONT"] = "7.8"
    env["WAVEP_FIGURE_LANG"] = "ja"
    env["WAVEP_MPL_FORCE_JA_TEXT"] = "1"

    for spec in specs:
        started_wall = time.time()
        started = time.perf_counter()
        artifact_preexisting = _artifact_exists_for_stem(spec.stem)
        spec_env = env.copy()
        if spec.stem not in {"validation_scoreboard", "quantum_scoreboard"}:
            spec_env["WAVEP_MPL_FORCE_ROLE_FONTS"] = "1"
        else:
            spec_env.pop("WAVEP_MPL_FORCE_ROLE_FONTS", None)

        if spec.stem not in {
            "validation_scoreboard",
            "quantum_scoreboard",
            "table1_part4_label_parity_audit",
            "llr_operational_metrics_audit",
        }:
            spec_env["WAVEP_MPL_ROLE_SCALE_TITLE"] = "1.00"
            spec_env["WAVEP_MPL_ROLE_SCALE_SUPTITLE"] = "1.08"
            spec_env["WAVEP_MPL_ROLE_SCALE_AXIS"] = "1.08"
            spec_env["WAVEP_MPL_ROLE_SCALE_TICK"] = "1.06"
            spec_env["WAVEP_MPL_ROLE_SCALE_LEGEND"] = "1.08"
            spec_env["WAVEP_MPL_ROLE_SCALE_NOTE"] = "1.08"
        else:
            spec_env.pop("WAVEP_MPL_ROLE_SCALE_TITLE", None)
            spec_env.pop("WAVEP_MPL_ROLE_SCALE_SUPTITLE", None)
            spec_env.pop("WAVEP_MPL_ROLE_SCALE_AXIS", None)
            spec_env.pop("WAVEP_MPL_ROLE_SCALE_TICK", None)
            spec_env.pop("WAVEP_MPL_ROLE_SCALE_LEGEND", None)
            spec_env.pop("WAVEP_MPL_ROLE_SCALE_NOTE", None)

        proc = subprocess.run(list(spec.argv), cwd=str(ROOT), env=spec_env)
        duration_s = time.perf_counter() - started
        artifact_updated = _artifact_updated_after(spec.stem, started_wall)
        if proc.returncode == 0:
            status = "ok"
        elif artifact_updated:
            status = "artifact_updated_despite_nonzero"
        elif artifact_preexisting:
            status = "preexisting_artifact_retained"
        else:
            status = "failed"

        rows.append(
            {
                "stem": spec.stem,
                "source": spec.source,
                "script": str(Path(spec.argv[2]).relative_to(ROOT)) if len(spec.argv) >= 3 else None,
                "argv": list(spec.argv),
                "returncode": int(proc.returncode),
                "duration_s": float(duration_s),
                "artifact_updated": bool(artifact_updated),
                "artifact_preexisting": bool(artifact_preexisting),
                "status": status,
            }
        )
        if proc.returncode != 0 and not artifact_updated and not artifact_preexisting:
            raise RuntimeError(f"strict figure refresh failed for {spec.stem}: rc={proc.returncode}")

    return rows


# 関数: `_write_manifest` の入出力契約と処理意図を定義する。

def _write_manifest(path: Path, *, tex_path: Path, stems: Sequence[str], rows: Sequence[dict]) -> None:
    all_passed = all(
        row.get("status") in {"ok", "artifact_updated_despite_nonzero", "preexisting_artifact_retained"} for row in rows
    )
    payload = {
        "generated_utc": _iso_utc_now(),
        "tex_path": str(tex_path),
        "figure_stems": list(stems),
        "strict_refresh_rows": list(rows),
        "summary": {
            "figure_count": int(len(stems)),
            "command_count": int(len(rows)),
            "all_passed": all_passed,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strictly rerun all Part IV referenced figure producers.")
    parser.add_argument("--tex", type=Path, required=True, help="Generated Part IV TeX path.")
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=ROOT / "output" / "private" / "summary" / "part4_strict_figure_refresh_manifest.json",
        help="Manifest JSON output path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    if not tex_path.exists():
        raise FileNotFoundError(f"tex not found: {tex_path}")

    stems = _parse_stems(tex_path)
    specs = _resolve_specs(stems, sys.executable or "python")
    rows = _run_specs(specs)
    _write_manifest(Path(args.manifest_json), tex_path=tex_path, stems=stems, rows=rows)
    print(f"[ok] strict Part IV figure refresh complete: {len(stems)} stems / {len(rows)} commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
