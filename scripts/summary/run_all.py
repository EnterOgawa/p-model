from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog


# クラス: `Task` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Task:
    key: str
    argv: List[str]
    cwd: Path
    requires_network: bool = False
    requires_cache_globs: Optional[List[str]] = None
    optional: bool = False


# 関数: `_repo_root` の入出力契約と処理意図を定義する。

def _repo_root() -> Path:
    return _ROOT


# 関数: `_has_cache` の入出力契約と処理意図を定義する。

def _has_cache(root: Path, pattern: str) -> bool:
    return any((root / pattern).parent.glob(Path(pattern).name))


# 関数: `_load_gw_event_list` の入出力契約と処理意図を定義する。

def _load_gw_event_list(root: Path) -> List[Dict[str, object]]:
    path = root / "data" / "gw" / "event_list.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return []

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    events = obj.get("events")
    # 条件分岐: `not isinstance(events, list)` を満たす経路を評価する。
    if not isinstance(events, list):
        return []

    profiles = obj.get("profiles")
    # 条件分岐: `not isinstance(profiles, dict)` を満たす経路を評価する。
    if not isinstance(profiles, dict):
        profiles = {}

    out: List[Dict[str, object]] = []
    for e in events:
        # 条件分岐: `not isinstance(e, dict)` を満たす経路を評価する。
        if not isinstance(e, dict):
            continue

        name = str(e.get("name") or "").strip()
        # 条件分岐: `not name` を満たす経路を評価する。
        if not name:
            continue

        slug = str(e.get("slug") or name.lower()).strip() or name.lower()
        argv: List[str] = []

        # v1: explicit argv list on the event.
        ev_argv = e.get("argv")
        # 条件分岐: `isinstance(ev_argv, list) and any(str(x).strip() for x in ev_argv)` を満たす経路を評価する。
        if isinstance(ev_argv, list) and any(str(x).strip() for x in ev_argv):
            argv = [str(x) for x in ev_argv if str(x).strip()]
        else:
            # v2: profile + structured fields + extra_argv.
            prof_name = str(e.get("profile") or "").strip()
            prof = profiles.get(prof_name) if prof_name else None
            prof_argv = prof.get("argv") if isinstance(prof, dict) else None
            # 条件分岐: `isinstance(prof_argv, list)` を満たす経路を評価する。
            if isinstance(prof_argv, list):
                argv.extend([str(x) for x in prof_argv if str(x).strip()])

            catalog = str(e.get("catalog") or "").strip()
            # 条件分岐: `catalog` を満たす経路を評価する。
            if catalog:
                argv += ["--catalog", catalog]

            detectors = str(e.get("detectors") or "").strip()
            # 条件分岐: `detectors` を満たす経路を評価する。
            if detectors:
                argv += ["--detectors", detectors]

            extra = e.get("extra_argv")
            # 条件分岐: `isinstance(extra, list)` を満たす経路を評価する。
            if isinstance(extra, list):
                argv.extend([str(x) for x in extra if str(x).strip()])

        out.append(
            {
                "name": name,
                "slug": slug,
                "argv": argv,
                "optional": bool(e.get("optional", True)),
            }
        )

    return out


# 関数: `_build_gw_tasks` の入出力契約と処理意図を定義する。

def _build_gw_tasks(*, root: Path, py: str, offline: bool, cwd: Path) -> List[Task]:
    script = root / "scripts" / "gw" / "gw150914_chirp_phase.py"
    events = _load_gw_event_list(root)
    # 条件分岐: `not events` を満たす経路を評価する。
    if not events:
        # Fallback (kept minimal; primary source is data/gw/event_list.json).
        events = [
            {
                "name": "GW150914",
                "slug": "gw150914",
                "argv": ["--event", "GW150914", "--wave-frange=70,300"],
                "optional": False,
            }
        ]

    tasks: List[Task] = []
    for e in events:
        name = str(e.get("name") or "").strip()
        slug = str(e.get("slug") or name.lower()).strip() or name.lower()
        argv = list(e.get("argv") or [])
        # Ensure --event is present (for reproducible cache/output naming).
        if "--event" not in argv:
            argv = ["--event", name] + argv

        key = f"gw_{slug}_chirp_phase"
        cache_globs = [
            f"data/gw/{slug}/*txt.gz",
            f"data/gw/{slug}/{name}_event.json",
        ]
        tasks.append(
            Task(
                key=key,
                argv=[py, "-B", str(script)] + [str(x) for x in argv] + (["--offline"] if offline else []),
                cwd=cwd,
                requires_network=True,
                requires_cache_globs=cache_globs,
                optional=bool(e.get("optional", True)),
            )
        )

    tasks.append(
        Task(
            key="gw150914_h1_l1_amplitude_ratio",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "gw" / "gw150914_h1_l1_amplitude_ratio.py"),
                "--event",
                "GW150914",
            ]
            + (["--offline"] if offline else []),
            cwd=cwd,
            requires_network=True,
            requires_cache_globs=[
                "data/gw/gw150914/*txt.gz",
                "data/gw/gw150914/GW150914_event.json",
            ],
            optional=False,
        )
    )

    tasks.append(
        Task(
            key="gw_multi_event_summary",
            argv=[py, "-B", str(root / "scripts" / "gw" / "gw_multi_event_summary.py")],
            cwd=cwd,
        )
    )
    tasks.append(
        Task(
            key="gw_event_list_diagnostics",
            argv=[py, "-B", str(root / "scripts" / "gw" / "gw_event_list_diagnostics.py")],
            cwd=cwd,
        )
    )
    return tasks


# 関数: `_run_task` の入出力契約と処理意図を定義する。

def _run_task(task: Task, env: Dict[str, str], log_dir: Path) -> Dict[str, object]:
    log_path = log_dir / f"{task.key}.log"
    started = time.perf_counter()

    p = subprocess.Popen(
        task.argv,
        cwd=str(task.cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    lines: List[str] = []
    assert p.stdout is not None
    with open(log_path, "w", encoding="utf-8") as f:
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
            lines.append(line)

    rc = p.wait()
    elapsed_s = time.perf_counter() - started

    return {
        "key": task.key,
        "argv": task.argv,
        "cwd": str(task.cwd),
        "returncode": rc,
        "elapsed_s": elapsed_s,
        "log": str(log_path),
        "ok": rc == 0,
        "tail": "".join(lines[-40:]),
    }


# 関数: `_run_task_quiet` の入出力契約と処理意図を定義する。

def _run_task_quiet(task: Task, env: Dict[str, str], log_dir: Path) -> Dict[str, object]:
    """
    Run one task and write stdout to log file only.
    (Used for parallel execution to avoid interleaved console output.)
    """
    log_path = log_dir / f"{task.key}.log"
    started = time.perf_counter()

    p = subprocess.Popen(
        task.argv,
        cwd=str(task.cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    lines: List[str] = []
    assert p.stdout is not None
    with open(log_path, "w", encoding="utf-8") as f:
        for line in p.stdout:
            f.write(line)
            lines.append(line)

    rc = p.wait()
    elapsed_s = time.perf_counter() - started

    return {
        "key": task.key,
        "argv": task.argv,
        "cwd": str(task.cwd),
        "returncode": rc,
        "elapsed_s": elapsed_s,
        "log": str(log_path),
        "ok": rc == 0,
        "tail": "".join(lines[-40:]),
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    out_summary = root / "output" / "private" / "summary"
    out_summary.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Run all P-model verification pipelines and refresh the summary report.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Do not use network; rely on cached data if available.")
    mode.add_argument("--online", action="store_true", help="Allow network (HORIZONS fetch) if caches are missing.")
    ap.add_argument(
        "--include-blocked",
        action="store_true",
        help="Also run blocked/standby topics (see doc/BLOCKED.md). Off by default.",
    )
    ap.add_argument(
        "--horizons-insecure",
        action="store_true",
        help="Disable SSL certificate verification for Horizons (not recommended; use only for debugging/proxy).",
    )
    ap.add_argument(
        "--horizons-ca-bundle",
        type=str,
        default="",
        help="Path to a custom CA bundle (PEM) to trust for Horizons SSL.",
    )
    ap.add_argument(
        "--llr-update-edc-batch",
        action="store_true",
        help="Update data/llr/llr_edc_batch_manifest.json by fetching EDC batch data (may be large; off by default).",
    )
    ap.add_argument(
        "--llr-years",
        type=str,
        default="2022-2025",
        help="Year range passed to fetch_llr_edc_batch.py when --llr-update-edc-batch is enabled (YYYY or YYYY-YYYY).",
    )
    ap.add_argument(
        "--llr-include-daily",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include daily NP2 (YYYYMMDD) when fetching EDC batch via --llr-update-edc-batch (default: enabled).",
    )
    ap.add_argument("--open", action="store_true", help="Open the HTML report at the end (Windows only).")
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Max concurrent tasks (default: 1). Uses safe group-level parallelism; DOCX export remains serial.",
    )
    ap.add_argument(
        "--docx-orientation",
        choices=["portrait", "landscape"],
        default="landscape",
        help="Page orientation for DOCX exports (default: landscape).",
    )
    ap.add_argument(
        "--docx-margin-mm",
        type=float,
        default=7.0,
        help="Page margins in mm for DOCX exports (default: 7; slightly safer than 5 for Word UI/print).",
    )
    args = ap.parse_args()

    offline = bool(args.offline) or (not args.online)
    include_blocked = bool(args.include_blocked)
    docx_orientation = str(args.docx_orientation)
    docx_margin_mm = float(args.docx_margin_mm)
    jobs = max(1, int(getattr(args, "jobs", 1) or 1))

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["HORIZONS_OFFLINE"] = "1" if offline else "0"
    # 条件分岐: `args.horizons_insecure` を満たす経路を評価する。
    if args.horizons_insecure:
        env["HORIZONS_INSECURE"] = "1"

    # 条件分岐: `args.horizons_ca_bundle` を満たす経路を評価する。

    if args.horizons_ca_bundle:
        env["HORIZONS_CA_BUNDLE"] = args.horizons_ca_bundle

    # When running multiple tasks concurrently, avoid CPU oversubscription from BLAS/OpenMP defaults.

    if jobs > 1:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.setdefault(k, "1")

    py = sys.executable
    cwd = root

    llr_primary = root / "data" / "llr" / "llr_primary.np2"
    llr_demo = root / "data" / "llr" / "demo_llr_like.crd"
    llr_manifest = root / "data" / "llr" / "llr_edc_batch_manifest.json"

    cassini_pds_root = root / "data" / "cassini" / "pds_sce1"
    cassini_manifest_odf = cassini_pds_root / "manifest_odf.json"
    cassini_manifest_tdf = cassini_pds_root / "manifest_tdf.json"

    cassini_tdf_band = ""
    # 条件分岐: `cassini_manifest_tdf.exists()` を満たす経路を評価する。
    if cassini_manifest_tdf.exists():
        try:
            cassini_tdf_band = str(json.loads(cassini_manifest_tdf.read_text(encoding="utf-8")).get("band") or "")
        except Exception:
            cassini_tdf_band = ""

    cassini_need_tdf_both = (not cassini_manifest_tdf.exists()) or (cassini_tdf_band.lower() != "both")

    report_task = Task(
        key="public_dashboard_final",
        argv=[py, "-B", str(root / "scripts" / "summary" / "public_dashboard.py")],
        cwd=cwd,
    )

    gw_tasks = _build_gw_tasks(root=root, py=py, offline=offline, cwd=cwd)

    blocked_tasks: List[Task] = [
        Task(
            key="bepicolombo_more_psa_status",
            argv=[py, "-B", str(root / "scripts" / "bepicolombo" / "more_psa_status.py")]
            + (["--offline"] if offline else []),
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="bepicolombo_more_fetch_collections",
            argv=[py, "-B", str(root / "scripts" / "bepicolombo" / "more_fetch_collections.py")]
            + (["--offline"] if offline else []),
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="bepicolombo_more_document_catalog",
            argv=[py, "-B", str(root / "scripts" / "bepicolombo" / "more_document_catalog.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="bepicolombo_spice_psa_status",
            argv=[py, "-B", str(root / "scripts" / "bepicolombo" / "spice_psa_status.py")]
            + (["--offline"] if offline else []),
            cwd=cwd,
            requires_network=True,
            requires_cache_globs=[
                "data/bepicolombo/psa_spice/base_index.html",
            ],
            optional=True,
        ),
        Task(
            key="bepicolombo_fetch_spice_kernels_psa",
            argv=[py, "-B", str(root / "scripts" / "bepicolombo" / "fetch_spice_kernels_psa.py")]
            + (["--offline"] if offline else []),
            cwd=cwd,
            requires_network=True,
            requires_cache_globs=[
                "data/bepicolombo/kernels/psa/kernels_meta.json",
            ],
            optional=True,
        ),
        Task(
            key="bepicolombo_shapiro_predict",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "bepicolombo" / "bepicolombo_shapiro_predict.py"),
                "--min-b-rsun",
                "1.0",
            ],
            cwd=cwd,
            requires_cache_globs=[
                "data/bepicolombo/kernels/psa/kernels_meta.json",
            ],
            optional=True,
        ),
        Task(
            key="bepicolombo_conjunction_catalog",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "bepicolombo" / "bepicolombo_conjunction_catalog.py"),
                "--min-b-rsun",
                "1.0",
                "--max-b-rsun",
                "10.0",
            ],
            cwd=cwd,
            requires_cache_globs=[
                "data/bepicolombo/kernels/psa/kernels_meta.json",
            ],
            optional=True,
        ),
    ]

    tasks: List[Task] = [
        Task(
            key="gps_fetch_igs_bkg",
            argv=[py, "-B", str(root / "scripts" / "gps" / "fetch_igs_bkg.py")],
            cwd=cwd,
            requires_network=True,
            optional=True,
        ),
        Task(
            key="llr_fetch_reflector_catalog",
            argv=[py, "-B", str(root / "scripts" / "llr" / "fetch_reflector_catalog.py")],
            cwd=cwd,
            requires_network=True,
            optional=True,
        ),
        Task(
            key="llr_fetch_moon_kernels_naif",
            argv=[py, "-B", str(root / "scripts" / "llr" / "fetch_moon_kernels_naif.py")],
            cwd=cwd,
            requires_network=True,
            optional=True,
        ),
        Task(
            key="llr_fetch_edc",
            argv=[py, "-B", str(root / "scripts" / "llr" / "fetch_llr_edc.py"), "--latest"],
            cwd=cwd,
            requires_network=True,
            optional=True,
        ),
        Task(
            key="llr_fetch_station_edc",
            argv=[py, "-B", str(root / "scripts" / "llr" / "fetch_station_edc.py")],
            cwd=cwd,
            requires_network=True,
            optional=True,
        ),
        Task(
            key="llr_fetch_pos_eop",
            argv=[py, "-B", str(root / "scripts" / "llr" / "fetch_pos_eop_edc.py")],
            cwd=cwd,
            requires_network=True,
            optional=True,
        ),
        Task(
            key="theory_solar_light_deflection",
            argv=[py, "-B", str(root / "scripts" / "theory" / "solar_light_deflection.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_gps_time_dilation",
            argv=[py, "-B", str(root / "scripts" / "theory" / "gps_time_dilation.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_delta_saturation_constraints",
            argv=[py, "-B", str(root / "scripts" / "theory" / "delta_saturation_constraints.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_gravitational_redshift_experiments",
            argv=[py, "-B", str(root / "scripts" / "theory" / "gravitational_redshift_experiments.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_frame_dragging_experiments",
            argv=[py, "-B", str(root / "scripts" / "theory" / "frame_dragging_experiments.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_gpb_scalar_limit_audit",
            argv=[py, "-B", str(root / "scripts" / "theory" / "gpb_scalar_limit_audit.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_frame_dragging_scalar_limit_combined_audit",
            argv=[py, "-B", str(root / "scripts" / "theory" / "frame_dragging_scalar_limit_combined_audit.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_pmodel_rotating_sphere_p_distribution_audit",
            argv=[py, "-B", str(root / "scripts" / "theory" / "pmodel_rotating_sphere_p_distribution_audit.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_dynamic_p_quadrupole_scalings",
            argv=[py, "-B", str(root / "scripts" / "theory" / "dynamic_p_quadrupole_scalings.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_pmodel_core_mapping_overview",
            argv=[py, "-B", str(root / "scripts" / "theory" / "pmodel_core_mapping_overview.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_pmodel_core_concept_comparison",
            argv=[py, "-B", str(root / "scripts" / "theory" / "pmodel_core_concept_comparison.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_pmodel_rejection_protocol_flowchart",
            argv=[py, "-B", str(root / "scripts" / "theory" / "pmodel_rejection_protocol_flowchart.py")],
            cwd=cwd,
        ),
        Task(
            key="theory_pmodel_beta_freeze_rationale",
            argv=[py, "-B", str(root / "scripts" / "theory" / "pmodel_beta_freeze_rationale.py")],
            cwd=cwd,
        ),
        Task(
            key="gps_compare_clocks",
            argv=[py, "-B", str(root / "scripts" / "gps" / "compare_clocks.py")],
            cwd=cwd,
        ),
        Task(
            key="gps_plot",
            argv=[py, "-B", str(root / "scripts" / "gps" / "plot.py")],
            cwd=cwd,
        ),
        Task(
            key="cassini_fig2_overlay",
            argv=[py, "-B", str(root / "scripts" / "cassini" / "cassini_fig2_overlay.py"), "--no-sweep"],
            cwd=cwd,
        ),
        Task(
            key="viking_shapiro",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "viking" / "viking_shapiro_check.py"),
                "--beta",
                "1.0",
            ]
            + (["--offline"] if offline else []),
            cwd=cwd,
            requires_network=True,
            requires_cache_globs=[
                "output/private/viking/horizons_cache/horizons_vectors_399_*.txt",
                "output/private/viking/horizons_cache/horizons_vectors_499_*.txt",
            ],
            optional=True,
        ),
        Task(
            key="viking_slides",
            argv=[py, "-B", str(root / "scripts" / "viking" / "update_slides.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="llr_quicklook",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "llr" / "llr_crd_quicklook.py"),
                str(llr_demo),
                "--assume-two-way",
            ],
            cwd=cwd,
        ),
        Task(
            key="llr_overlay",
            argv=[py, "-B", str(root / "scripts" / "llr" / "llr_pmodel_overlay_horizons_noargs.py")],
            cwd=cwd,
            requires_network=True,
            requires_cache_globs=[
                "output/private/llr/horizons_cache/horizons_vectors_301_*.csv",
                "output/private/llr/horizons_cache/horizons_vectors_10_*.csv",
            ],
            optional=True,
        ),
        Task(
            key="llr_batch_eval",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "llr" / "llr_batch_eval.py"),
                "--time-tag-mode",
                "auto",
                "--min-points",
                "30",
                "--chunk",
                "50",
            ]
            + (["--offline"] if offline else []),
            cwd=cwd,
            requires_network=True,
            requires_cache_globs=[
                "output/private/llr/horizons_cache/horizons_vectors_301_*.csv",
                "output/private/llr/horizons_cache/horizons_vectors_10_*.csv",
            ],
            optional=True,
        ),
        Task(
            key="mercury_precession",
            argv=[py, "-B", str(root / "scripts" / "mercury" / "mercury_precession_v3.py")],
            cwd=cwd,
        ),
        Task(
            key="eht_shadow_compare",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_shadow_compare.py")],
            cwd=cwd,
        ),
        Task(
            key="eht_kerr_shadow_coeff_grid",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_kerr_shadow_coeff_grid.py")],
            cwd=cwd,
        ),
        Task(
            key="eht_kerr_shadow_coeff_definition_sensitivity",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "eht" / "eht_kerr_shadow_coeff_definition_sensitivity.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="eht_sources_integrity",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sources_integrity.py")],
            cwd=cwd,
        ),
        Task(
            key="eht_primary_value_anchors",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_primary_value_anchors.py")],
            cwd=cwd,
        ),
        Task(
            key="eht_gravity_s2_constraints",
            argv=[py, "-B", str(root / "scripts" / "eht" / "gravity_s2_constraints.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_gravity_s2_pmodel_projection",
            argv=[py, "-B", str(root / "scripts" / "eht" / "gravity_s2_pmodel_projection.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_ringfit_table_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_ringfit_table_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_calibration_systematics_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_calibration_systematics_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper2_gain_uncertainties_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper2_gain_uncertainties_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper2_syserr_table_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper2_syserr_table_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_variability_noise_model_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_variability_noise_model_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_mringfits_table_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_mringfits_table_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper4_alpha_calibration_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper4_alpha_calibration_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper4_thetag_table_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper4_thetag_table_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper4_morphology_table_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper4_morphology_table_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper4_debiased_noise_table_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper4_debiased_noise_table_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper5_pass_fraction_tables_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper5_pass_fraction_tables_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper5_key_constraints_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper5_key_constraints_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_gravity_sgra_flux_distribution_metrics",
            argv=[py, "-B", str(root / "scripts" / "eht" / "gravity_sgra_flux_distribution_metrics.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper5_constraint_relaxation_sweep",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper5_constraint_relaxation_sweep.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper5_m3_nir_reconnection_conditions",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper5_m3_nir_reconnection_conditions.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_sgra_paper6_metric_constraints",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_sgra_paper6_metric_constraints.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="eht_kappa_error_budget",
            argv=[py, "-B", str(root / "scripts" / "eht" / "eht_kappa_error_budget.py")],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="pulsar_binary_pulsar_orbital_decay",
            argv=[py, "-B", str(root / "scripts" / "pulsar" / "binary_pulsar_orbital_decay.py")],
            cwd=cwd,
        ),
        *gw_tasks,
        Task(
            key="cosmology_fetch_boss_dr12_satpathy2016_corrfunc_multipoles",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "fetch_boss_dr12_satpathy2016_corrfunc_multipoles.py"),
            ],
            cwd=cwd,
            requires_network=True,
            optional=True,
        ),
        Task(
            key="cosmology_fetch_boss_dr12_beutler2016_bao_powspec",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "fetch_boss_dr12_beutler2016_bao_powspec.py"),
            ],
            cwd=cwd,
            requires_network=True,
            optional=True,
        ),
        Task(
            key="cosmology_fetch_mast_jwst_spectra",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "fetch_mast_jwst_spectra.py"),
                "--offline",
                "--estimate-z",
                "--confirm-z",
                "--init-line-id",
            ],
            cwd=cwd,
            requires_cache_globs=[
                "data/cosmology/mast/jwst_spectra/manifest_all.json",
            ],
            optional=True,
        ),
        Task(
            key="cosmology_redshift_pbg",
            argv=[py, "-B", str(root / "scripts" / "cosmology" / "cosmology_redshift_pbg.py")],
            cwd=cwd,
        ),
        Task(
            key="cosmology_observable_scalings",
            argv=[py, "-B", str(root / "scripts" / "cosmology" / "cosmology_observable_scalings.py")],
            cwd=cwd,
        ),
        Task(
            key="cosmology_sn_time_dilation_constraints",
            argv=[py, "-B", str(root / "scripts" / "cosmology" / "cosmology_sn_time_dilation_constraints.py")],
            cwd=cwd,
        ),
        Task(
            key="cosmology_cmb_temperature_scaling_constraints",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_cmb_temperature_scaling_constraints.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_cmb_peak_uplift_audit",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_cmb_peak_uplift_audit.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_alcock_paczynski_constraints",
            argv=[py, "-B", str(root / "scripts" / "cosmology" / "cosmology_alcock_paczynski_constraints.py")],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_duality_constraints",
            argv=[py, "-B", str(root / "scripts" / "cosmology" / "cosmology_distance_duality_constraints.py")],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_duality_source_sensitivity",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_distance_duality_source_sensitivity.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_duality_internal_consistency",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_distance_duality_internal_consistency.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_duality_systematics_envelope",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_distance_duality_systematics_envelope.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_reconnection_parameter_space",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_reconnection_parameter_space.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_reconnection_plausibility",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_reconnection_plausibility.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_reconnection_required_ruler_evolution",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_reconnection_required_ruler_evolution.py"),
                "--use-independent-probes",
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_bao_distance_ratio_fit",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_distance_ratio_fit.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_bao_xi_multipole_peakfit",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_xi_multipole_peakfit.py"),
            ],
            cwd=cwd,
            requires_cache_globs=[
                "data/cosmology/ross_2016_combineddr12_corrfunc/Ross_2016_COMBINEDDR12_zbin1_correlation_function_monopole_post_recon_bincent0.dat",
                "data/cosmology/ross_2016_combineddr12_corrfunc/Ross_2016_COMBINEDDR12_zbin1_covariance_monoquad_post_recon_bincent0.dat",
            ],
        ),
        Task(
            key="cosmology_bao_xi_multipole_peakfit_pre_recon",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_xi_multipole_peakfit.py"),
                "--dataset",
                "satpathy_pre",
            ],
            cwd=cwd,
            requires_cache_globs=[
                "data/cosmology/satpathy_2016_combineddr12_fs_corrfunc_multipoles/Satpathy_2016_COMBINEDDR12_Bin1_Monopole_pre_recon.dat",
                "data/cosmology/satpathy_2016_combineddr12_fs_corrfunc_multipoles/Satpathy_2016_COMBINEDDR12_Bin1_Covariance_pre_recon.txt",
            ],
            optional=True,
        ),
        Task(
            key="cosmology_bao_pk_multipole_peakfit",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_pk_multipole_peakfit.py"),
            ],
            cwd=cwd,
            requires_cache_globs=[
                "data/cosmology/beutler_2016_combineddr12_bao_powspec/Beutleretal_pk_monopole_DR12_NGC_z1_postrecon_120.dat",
                "data/cosmology/beutler_2016_combineddr12_bao_powspec/Beutleretal_cov_patchy_z1_NGC_postrecon_1_30_1_30_1_1_996_60.dat",
            ],
        ),
        Task(
            key="cosmology_bao_pk_multipole_peakfit_pre_recon",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_pk_multipole_peakfit.py"),
                "--recon",
                "prerecon",
            ],
            cwd=cwd,
            requires_cache_globs=[
                "data/cosmology/beutler_2016_combineddr12_bao_powspec/Beutleretal_pk_monopole_DR12_NGC_z1_prerecon_120.dat",
                "data/cosmology/beutler_2016_combineddr12_bao_powspec/Beutleretal_cov_patchy_z1_NGC_prerecon_1_30_1_30_1_1_2045_60.dat",
            ],
            optional=True,
        ),
        Task(
            key="cosmology_bao_pk_multipole_peakfit_window",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_pk_multipole_peakfit.py"),
                "--window",
            ],
            cwd=cwd,
            requires_cache_globs=[
                "data/cosmology/beutler_2016_combineddr12_bao_powspec/Beutleretal_pk_monopole_DR12_NGC_z1_postrecon_120.dat",
                "data/cosmology/beutler_2016_combineddr12_bao_powspec/Beutleretal_cov_patchy_z1_NGC_postrecon_1_30_1_30_1_1_996_60.dat",
                "data/cosmology/beutler_2016_combineddr12_bao_powspec/Beutleretal_window_z1_NGC.dat",
            ],
        ),
        Task(
            key="cosmology_bao_pk_multipole_peakfit_pre_recon_window",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_pk_multipole_peakfit.py"),
                "--recon",
                "prerecon",
                "--window",
            ],
            cwd=cwd,
            requires_cache_globs=[
                "data/cosmology/beutler_2016_combineddr12_bao_powspec/Beutleretal_pk_monopole_DR12_NGC_z1_prerecon_120.dat",
                "data/cosmology/beutler_2016_combineddr12_bao_powspec/Beutleretal_cov_patchy_z1_NGC_prerecon_1_30_1_30_1_1_2045_60.dat",
                "data/cosmology/beutler_2016_combineddr12_bao_powspec/Beutleretal_window_z1_NGC.dat",
            ],
            optional=True,
        ),
        Task(
            key="cosmology_distance_indicator_rederivation_bao_survey_sensitivity",
            argv=[
                py,
                "-B",
                str(
                    root
                    / "scripts"
                    / "cosmology"
                    / "cosmology_distance_indicator_rederivation_bao_survey_sensitivity.py"
                ),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_tolman_surface_brightness_constraints",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_tolman_surface_brightness_constraints.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_static_infinite_hypothesis_pack",
            argv=[py, "-B", str(root / "scripts" / "cosmology" / "cosmology_static_infinite_hypothesis_pack.py")],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_indicator_error_budget",
            argv=[py, "-B", str(root / "scripts" / "cosmology" / "cosmology_distance_indicator_error_budget.py")],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_indicator_error_budget_sensitivity",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_distance_indicator_error_budget_sensitivity.py"),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_indicator_rederivation_requirements",
            argv=[
                py,
                "-B",
                str(
                    root
                    / "scripts"
                    / "cosmology"
                    / "cosmology_distance_indicator_rederivation_requirements.py"
                ),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_indicator_rederivation_candidate_search",
            argv=[
                py,
                "-B",
                str(
                    root
                    / "scripts"
                    / "cosmology"
                    / "cosmology_distance_indicator_rederivation_candidate_search.py"
                ),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_indicator_rederivation_limiting_summary",
            argv=[
                py,
                "-B",
                str(
                    root
                    / "scripts"
                    / "cosmology"
                    / "cosmology_distance_indicator_rederivation_limiting_summary.py"
                ),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_indicator_rederivation_selected_sources",
            argv=[
                py,
                "-B",
                str(
                    root
                    / "scripts"
                    / "cosmology"
                    / "cosmology_distance_indicator_rederivation_selected_sources.py"
                ),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_indicator_rederivation_policy_sensitivity",
            argv=[
                py,
                "-B",
                str(
                    root
                    / "scripts"
                    / "cosmology"
                    / "cosmology_distance_indicator_rederivation_policy_sensitivity.py"
                ),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_indicator_rederivation_global_prior_search",
            argv=[
                py,
                "-B",
                str(
                    root
                    / "scripts"
                    / "cosmology"
                    / "cosmology_distance_indicator_rederivation_global_prior_search.py"
                ),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_indicator_rederivation_bao_relax_thresholds",
            argv=[
                py,
                "-B",
                str(
                    root
                    / "scripts"
                    / "cosmology"
                    / "cosmology_distance_indicator_rederivation_bao_relax_thresholds.py"
                ),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_tension_attribution",
            argv=[py, "-B", str(root / "scripts" / "cosmology" / "cosmology_tension_attribution.py")],
            cwd=cwd,
        ),
        Task(
            key="cosmology_ddr_reconnection_conditions",
            argv=[py, "-B", str(root / "scripts" / "cosmology" / "cosmology_ddr_reconnection_conditions.py")],
            cwd=cwd,
        ),
        Task(
            key="cosmology_distance_indicator_rederivation_candidate_matrix",
            argv=[
                py,
                "-B",
                str(
                    root
                    / "scripts"
                    / "cosmology"
                    / "cosmology_distance_indicator_rederivation_candidate_matrix.py"
                ),
            ],
            cwd=cwd,
        ),
        Task(
            key="cosmology_bao_scaled_distance_fit_sensitivity",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_scaled_distance_fit_sensitivity.py"),
            ],
            cwd=cwd,
        ),
        # BAO (catalog-based ξℓ): Derived-only summaries from precomputed WSL/Corrfunc outputs.
        # These tasks are optional because ξ(s,μ) computation requires Corrfunc under WSL/Linux.
        Task(
            key="cosmology_bao_catalog_wedge_anisotropy",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_wedge_anisotropy.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_wedge_anisotropy_recon_grid_iso",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_wedge_anisotropy.py"),
                "--out-tag",
                "recon_grid_iso",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmass_combined",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmass",
                "--caps",
                "combined",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmass_combined_recon_grid_iso",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmass",
                "--caps",
                "combined",
                "--out-tag",
                "recon_grid_iso",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmass_north",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmass",
                "--caps",
                "north",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmass_north_recon_grid_iso",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmass",
                "--caps",
                "north",
                "--out-tag",
                "recon_grid_iso",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmass_south",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmass",
                "--caps",
                "south",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmass_south_recon_grid_iso",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmass",
                "--caps",
                "south",
                "--out-tag",
                "recon_grid_iso",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_lowz_combined",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "lowz",
                "--caps",
                "combined",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_lowz_combined_recon_grid_iso",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "lowz",
                "--caps",
                "combined",
                "--out-tag",
                "recon_grid_iso",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_lowz_north",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "lowz",
                "--caps",
                "north",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_lowz_north_recon_grid_iso",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "lowz",
                "--caps",
                "north",
                "--out-tag",
                "recon_grid_iso",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_lowz_south",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "lowz",
                "--caps",
                "south",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_lowz_south_recon_grid_iso",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "lowz",
                "--caps",
                "south",
                "--out-tag",
                "recon_grid_iso",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_caps_summary_recon_grid_iso",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit_caps_summary.py"),
                "--out-tag",
                "recon_grid_iso",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmasslowztot",
                "--caps",
                "combined",
                "--require-zbin",
                "--out-tag",
                "prerecon",
                "--cov-source",
                "satpathy",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly_recon_grid_iso",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmasslowztot",
                "--caps",
                "combined",
                "--require-zbin",
                "--out-tag",
                "recon_grid_iso",
                "--dists",
                "lcdm",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly_recon_mw_multigrid",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmasslowztot",
                "--caps",
                "combined",
                "--require-zbin",
                "--out-tag",
                "recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757",
                "--dists",
                "lcdm,pbg",
                "--cov-source",
                "ross",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmasslowztot_north_zbinonly",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmasslowztot",
                "--caps",
                "north",
                "--require-zbin",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_cmasslowztot_south_zbinonly",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit.py"),
                "--sample",
                "cmasslowztot",
                "--caps",
                "south",
                "--require-zbin",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peak_summary",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peak_summary.py"),
                "--sample",
                "cmasslowztot",
                "--caps",
                "combined",
                "--require-zbin",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peak_summary_recon_grid_iso",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peak_summary.py"),
                "--sample",
                "cmasslowztot",
                "--caps",
                "combined",
                "--require-zbin",
                "--out-tag",
                "recon_grid_iso",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_vs_published_crosscheck",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_vs_published_crosscheck.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_vs_published_multipoles_overlay",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_vs_published_multipoles_overlay.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_recon_param_scan",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_recon_param_scan.py"),
                "--include-baseline",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_recon_param_scan_ani",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_recon_param_scan.py"),
                "--out-tag-prefix",
                "recon_grid_ani",
                "--include-baseline",
                "--out-png",
                "output/private/cosmology/cosmology_bao_recon_param_scan_ani.png",
                "--out-json",
                "output/private/cosmology/cosmology_bao_recon_param_scan_ani_metrics.json",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_recon_gap_summary",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_recon_gap_summary.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_recon_gap_broadband_fit",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_recon_gap_broadband_fit.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_recon_gap_broadband_sensitivity",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_recon_gap_broadband_sensitivity.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_recon_gap_ross_eval_alignment",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_recon_gap_ross_eval_alignment.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_window_multipoles",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_window_multipoles.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_window_mixing",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_window_mixing.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_weight_scheme_sensitivity",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_weight_scheme_sensitivity.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_random_max_rows_sensitivity",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_random_max_rows_sensitivity.py"),
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_bao_catalog_peakfit_settings_sensitivity",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "cosmology" / "cosmology_bao_catalog_peakfit_settings_sensitivity.py"),
                "--sample",
                "lrg",
                "--caps",
                "combined",
                "--out-tag",
                "w_desi_default_ms_off_y1bins",
            ],
            cwd=cwd,
            optional=True,
        ),
        Task(
            key="cosmology_distance_indicator_rederivation_bao_mode_sensitivity",
            argv=[
                py,
                "-B",
                str(
                    root
                    / "scripts"
                    / "cosmology"
                    / "cosmology_distance_indicator_rederivation_bao_mode_sensitivity.py"
                ),
            ],
            cwd=cwd,
        ),
        Task(
            key="theory_freeze_parameters",
            argv=[py, "-B", str(root / "scripts" / "theory" / "freeze_parameters.py")],
            cwd=cwd,
        ),
        Task(
            key="decisive_falsification",
            argv=[py, "-B", str(root / "scripts" / "summary" / "decisive_falsification.py")],
            cwd=cwd,
        ),
        Task(
            key="decisive_candidates",
            argv=[py, "-B", str(root / "scripts" / "summary" / "decisive_candidates.py")],
            cwd=cwd,
        ),
        Task(
            key="quantum_falsification",
            argv=[py, "-B", str(root / "scripts" / "summary" / "quantum_falsification.py")],
            cwd=cwd,
        ),
        Task(
            key="paper_table1",
            argv=[py, "-B", str(root / "scripts" / "summary" / "paper_tables.py")],
            cwd=cwd,
        ),
        Task(
            key="validation_scoreboard",
            argv=[py, "-B", str(root / "scripts" / "summary" / "validation_scoreboard.py")],
            cwd=cwd,
        ),
        Task(
            key="paper_lint",
            argv=[py, "-B", str(root / "scripts" / "summary" / "paper_lint.py")],
            cwd=cwd,
        ),
        Task(
            key="part4_verification_lint",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "summary" / "paper_lint.py"),
                "--manuscript",
                "doc/paper/13_part4_verification.md",
                "--strict",
            ],
            cwd=cwd,
        ),
        Task(
            key="paper_html",
            argv=[py, "-B", str(root / "scripts" / "summary" / "paper_html.py")],
            cwd=cwd,
        ),
        Task(
            key="part4_verification_html",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "summary" / "paper_html.py"),
                "--profile",
                "part4_verification",
            ],
            cwd=cwd,
        ),
    ]

    # 条件分岐: `include_blocked` を満たす経路を評価する。
    if include_blocked:
        tasks.extend(blocked_tasks)

    # 条件分岐: `(not offline) and (bool(args.llr_update_edc_batch) or (not llr_manifest.exist...` を満たす経路を評価する。

    if (not offline) and (bool(args.llr_update_edc_batch) or (not llr_manifest.exists())):
        llr_fetch_args = [py, "-B", str(root / "scripts" / "llr" / "fetch_llr_edc_batch.py")]
        # 条件分岐: `str(args.llr_years).strip()` を満たす経路を評価する。
        if str(args.llr_years).strip():
            llr_fetch_args += ["--years", str(args.llr_years).strip()]

        # 条件分岐: `bool(args.llr_include_daily)` を満たす経路を評価する。

        if bool(args.llr_include_daily):
            llr_fetch_args += ["--include-daily"]

        tasks.insert(
            3,
            Task(
                key="llr_fetch_edc_batch",
                argv=llr_fetch_args,
                cwd=cwd,
                requires_network=True,
                optional=True,
            ),
        )

    # Cassini (PDS primary cache): fetch once when caches are missing or incomplete (large files; avoid re-hashing every run).
    # Note: Cassini plasma correction uses Ka+X; therefore we prefer TDF band=both.

    if not offline and (not cassini_manifest_odf.exists() or cassini_need_tdf_both):
        try:
            cassini_overlay_idx = next(i for i, t in enumerate(tasks) if t.key == "cassini_fig2_overlay")
        except StopIteration:
            cassini_overlay_idx = len(tasks)

        # 条件分岐: `not cassini_manifest_odf.exists()` を満たす経路を評価する。

        if not cassini_manifest_odf.exists():
            tasks.insert(
                cassini_overlay_idx,
                Task(
                    key="cassini_fetch_pds_odf",
                    argv=[
                        py,
                        "-B",
                        str(root / "scripts" / "cassini" / "fetch_cassini_pds_sce1_odf.py"),
                    ],
                    cwd=cwd,
                    requires_network=True,
                    optional=True,
                ),
            )
            cassini_overlay_idx += 1

        # 条件分岐: `cassini_need_tdf_both` を満たす経路を評価する。

        if cassini_need_tdf_both:
            tasks.insert(
                cassini_overlay_idx,
                Task(
                    key="cassini_fetch_pds_tdf",
                    argv=[
                        py,
                        "-B",
                        str(root / "scripts" / "cassini" / "fetch_cassini_pds_sce1_tdf.py"),
                        "--band",
                        "both",
                    ],
                    cwd=cwd,
                    requires_network=True,
                    optional=True,
                ),
            )
            cassini_overlay_idx += 1

    log_dir = out_summary / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline" if offline else "online",
        "include_blocked": include_blocked,
        "jobs": jobs,
        "docx": {"orientation": docx_orientation, "margin_mm": docx_margin_mm},
        "python": sys.version,
        "tasks": [],
    }

    failures = 0

    # 関数: `_skip_record` の入出力契約と処理意図を定義する。
    def _skip_record(t: Task) -> Optional[Dict[str, object]]:
        # 条件分岐: `offline and t.requires_cache_globs` を満たす経路を評価する。
        if offline and t.requires_cache_globs:
            missing = [p for p in t.requires_cache_globs if not _has_cache(root, p)]
            # 条件分岐: `missing` を満たす経路を評価する。
            if missing:
                return {
                    "key": t.key,
                    "skipped": True,
                    "reason": f"offline mode and cache missing: {', '.join(missing)}",
                }

        # 条件分岐: `offline and t.requires_network and not t.requires_cache_globs` を満たす経路を評価する。

        if offline and t.requires_network and not t.requires_cache_globs:
            return {"key": t.key, "skipped": True, "reason": "offline mode"}

        return None

    # Final aggregation tasks must run after all topic tasks finish.

    final_task_keys = {
        "theory_freeze_parameters",
        "decisive_falsification",
        "decisive_candidates",
        "quantum_falsification",
        "paper_table1",
        "validation_scoreboard",
        "paper_lint",
        "paper_html",
    }

    # 条件分岐: `jobs <= 1` を満たす経路を評価する。
    if jobs <= 1:
        for task in tasks:
            # 条件分岐: `task.key == "llr_quicklook"` を満たす経路を評価する。
            if task.key == "llr_quicklook":
                llr_input = llr_primary if llr_primary.exists() else llr_demo
                # 条件分岐: `len(task.argv) >= 4 and task.argv[3] != str(llr_input)` を満たす経路を評価する。
                if len(task.argv) >= 4 and task.argv[3] != str(llr_input):
                    task = Task(
                        key=task.key,
                        argv=[task.argv[0], task.argv[1], task.argv[2], str(llr_input)] + task.argv[4:],
                        cwd=task.cwd,
                        requires_network=task.requires_network,
                        requires_cache_globs=task.requires_cache_globs,
                        optional=task.optional,
                    )

            rec0 = _skip_record(task)
            # 条件分岐: `rec0 is not None` を満たす経路を評価する。
            if rec0 is not None:
                run_info["tasks"].append(rec0)
                print(f"[skip] {task.key}: {rec0['reason']}")
                continue

            print(f"\n=== Running: {task.key} ===")
            rec = _run_task(task, env=env, log_dir=log_dir)
            run_info["tasks"].append(rec)
            # 条件分岐: `not rec.get("ok", False)` を満たす経路を評価する。
            if not rec.get("ok", False):
                # 条件分岐: `task.optional` を満たす経路を評価する。
                if task.optional:
                    print(f"[warn] {task.key} failed (optional). See log: {rec['log']}")
                else:
                    print(f"[err] {task.key} failed. See log: {rec['log']}")
                    failures += 1
    else:
        print_lock = Lock()
        net_lock = Lock()
        task_records: List[Optional[Dict[str, object]]] = [None] * len(tasks)

        # 関数: `_group_name` の入出力契約と処理意図を定義する。
        def _group_name(key: str) -> str:
            return key.split("_", 1)[0]

        groups: Dict[str, List[int]] = {}
        final_indices: List[int] = []
        for i, t in enumerate(tasks):
            # 条件分岐: `t.key in final_task_keys` を満たす経路を評価する。
            if t.key in final_task_keys:
                final_indices.append(i)
                continue

            groups.setdefault(_group_name(t.key), []).append(i)

        # 関数: `_run_group` の入出力契約と処理意図を定義する。

        def _run_group(group: str, indices: List[int]) -> None:
            for i in indices:
                t = tasks[i]
                # 条件分岐: `t.key == "llr_quicklook"` を満たす経路を評価する。
                if t.key == "llr_quicklook":
                    llr_input = llr_primary if llr_primary.exists() else llr_demo
                    # 条件分岐: `len(t.argv) >= 4 and t.argv[3] != str(llr_input)` を満たす経路を評価する。
                    if len(t.argv) >= 4 and t.argv[3] != str(llr_input):
                        t = Task(
                            key=t.key,
                            argv=[t.argv[0], t.argv[1], t.argv[2], str(llr_input)] + t.argv[4:],
                            cwd=t.cwd,
                            requires_network=t.requires_network,
                            requires_cache_globs=t.requires_cache_globs,
                            optional=t.optional,
                        )

                rec0 = _skip_record(t)
                # 条件分岐: `rec0 is not None` を満たす経路を評価する。
                if rec0 is not None:
                    task_records[i] = rec0
                    with print_lock:
                        print(f"[skip] {t.key}: {rec0['reason']}")

                    continue

                with print_lock:
                    print(f"\n=== Running: {t.key} (group={group}) ===")

                try:
                    # 条件分岐: `t.requires_network` を満たす経路を評価する。
                    if t.requires_network:
                        with net_lock:
                            rec = _run_task_quiet(t, env=env, log_dir=log_dir)
                    else:
                        rec = _run_task_quiet(t, env=env, log_dir=log_dir)
                except Exception as e:
                    rec = {
                        "key": t.key,
                        "argv": t.argv,
                        "cwd": str(t.cwd),
                        "returncode": None,
                        "elapsed_s": 0.0,
                        "log": str(log_dir / f"{t.key}.log"),
                        "ok": False,
                        "tail": f"internal error: {e}",
                    }

                task_records[i] = rec
                # 条件分岐: `not rec.get("ok", False)` を満たす経路を評価する。
                if not rec.get("ok", False):
                    with print_lock:
                        # 条件分岐: `t.optional` を満たす経路を評価する。
                        if t.optional:
                            print(f"[warn] {t.key} failed (optional). See log: {rec['log']}")
                        else:
                            print(f"[err] {t.key} failed. See log: {rec['log']}")

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_run_group, g, idxs) for g, idxs in sorted(groups.items())]
            for fut in futs:
                fut.result()

        # Run final aggregation tasks serially (needs outputs from all groups).

        for i in final_indices:
            t = tasks[i]
            rec0 = _skip_record(t)
            # 条件分岐: `rec0 is not None` を満たす経路を評価する。
            if rec0 is not None:
                task_records[i] = rec0
                print(f"[skip] {t.key}: {rec0['reason']}")
                continue

            print(f"\n=== Running: {t.key} ===")
            rec = _run_task(t, env=env, log_dir=log_dir)
            task_records[i] = rec

        # Compact into run_info (preserve original order)

        ordered: List[Dict[str, object]] = []
        for i, t in enumerate(tasks):
            rec = task_records[i]
            # 条件分岐: `rec is None` を満たす経路を評価する。
            if rec is None:
                rec = {"key": t.key, "skipped": True, "reason": "not executed (internal scheduling error)"}

            ordered.append(rec)

        run_info["tasks"] = ordered
        failures = sum(
            1
            for i, t in enumerate(tasks)
            if (not bool(ordered[i].get("skipped", False)))
            and (not bool(ordered[i].get("ok", True)))
            and (not bool(t.optional))
        )

    status_path = out_summary / "run_all_status.json"
    status_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] status: {status_path}")

    # Generate the report AFTER writing status, so the HTML can embed the latest run_all_status.json.
    print(f"\n=== Running: {report_task.key} ===")
    rec = _run_task(report_task, env=env, log_dir=log_dir)
    run_info["tasks"].append(rec)
    status_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    # 条件分岐: `not rec.get("ok", False)` を満たす経路を評価する。
    if not rec.get("ok", False):
        print(f"[err] {report_task.key} failed. See log: {rec['log']}")
        failures += 1

    # Refresh once more so the report can also include the summary_report task result.

    print(f"\n=== Refreshing: {report_task.key} ===")
    _run_task(report_task, env=env, log_dir=log_dir)

    # Export DOCX versions for easy sharing/editing (best effort; depends on local Word availability).
    docx_tasks: List[Task] = [
        Task(
            key="docx_public_report",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "summary" / "html_to_docx.py"),
                "--in",
                str(out_summary / "pmodel_public_report.html"),
                "--out",
                str(out_summary / "pmodel_public_report.docx"),
                "--pagebreak-validations",
                "--orientation",
                docx_orientation,
                "--margin-mm",
                str(docx_margin_mm),
            ],
            cwd=cwd,
        ),
        Task(
            key="docx_paper",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "summary" / "html_to_docx.py"),
                "--in",
                str(out_summary / "pmodel_paper.html"),
                "--out",
                str(out_summary / "pmodel_paper.docx"),
                "--paper-equations",
                "--orientation",
                docx_orientation,
                "--margin-mm",
                str(docx_margin_mm),
            ],
            cwd=cwd,
        ),
        Task(
            key="docx_part4_verification",
            argv=[
                py,
                "-B",
                str(root / "scripts" / "summary" / "html_to_docx.py"),
                "--in",
                str(out_summary / "pmodel_paper_part4_verification.html"),
                "--out",
                str(out_summary / "pmodel_paper_part4_verification.docx"),
                "--paper-equations",
                "--orientation",
                docx_orientation,
                "--margin-mm",
                str(docx_margin_mm),
            ],
            cwd=cwd,
        ),
    ]
    for task in docx_tasks:
        print(f"\n=== Running: {task.key} ===")
        rec = _run_task(task, env=env, log_dir=log_dir)
        # html_to_docx.py returns:
        #   0: ok, 3: skipped (no Word backend), otherwise: error.
        rc = int(rec.get("returncode") or 0)
        # 条件分岐: `rc == 3` を満たす経路を評価する。
        if rc == 3:
            rec["skipped"] = True
            rec["reason"] = "no supported Word backend found"

        run_info["tasks"].append(rec)
        # 条件分岐: `not rec.get("ok", False)` を満たす経路を評価する。
        if not rec.get("ok", False):
            # 条件分岐: `rc == 3` を満たす経路を評価する。
            if rc == 3:
                print(f"[warn] {task.key} skipped (no Word). See log: {rec['log']}")
            else:
                print(f"[err] {task.key} failed. See log: {rec['log']}")
                failures += 1

    # Persist once more so the report (and readers) can see DOCX task results.

    status_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # Append machine-readable work history (JSONL)
    try:
        public_html = out_summary / "pmodel_public_report.html"
        public_docx = out_summary / "pmodel_public_report.docx"
        paper_html = out_summary / "pmodel_paper.html"
        paper_docx = out_summary / "pmodel_paper.docx"
        part4_html = out_summary / "pmodel_paper_part4_verification.html"
        part4_docx = out_summary / "pmodel_paper_part4_verification.docx"
        worklog.append_event(
            {
                "event_type": "run_all",
                "mode": "offline" if offline else "online",
                "include_blocked": include_blocked,
                "argv": sys.argv,
                "failures": int(failures),
                "status_path": status_path,
                "outputs": {
                    "public_report_html": public_html,
                    "public_report_docx": public_docx if public_docx.exists() else None,
                    "paper_html": paper_html if paper_html.exists() else None,
                    "paper_docx": paper_docx if paper_docx.exists() else None,
                    "part4_verification_html": part4_html if part4_html.exists() else None,
                    "part4_verification_docx": part4_docx if part4_docx.exists() else None,
                    "run_all_logs_dir": log_dir,
                },
            }
        )
    except Exception:
        pass

    # 条件分岐: `args.open` を満たす経路を評価する。

    if args.open:
        # Canonical output is the public-friendly report (single entry point).
        public_html = out_summary / "pmodel_public_report.html"
        legacy_html = out_summary / "pmodel_report.html"
        # 条件分岐: `os.name == "nt"` を満たす経路を評価する。
        if os.name == "nt":
            try:
                os.startfile(str(public_html if public_html.exists() else legacy_html))  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[warn] failed to open report: {e}", file=sys.stderr)
        else:
            print("[warn] --open is supported on Windows only.", file=sys.stderr)

    return 1 if failures else 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
