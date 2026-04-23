#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part2_rebuild_source_figures.py

目的:
- Part II（宇宙物理）の source figure を locale 別に再生成する。
- 日本語 canonical を保ったまま、英語などの localized figure を
  `output/.../locales/<locale>/...` へ出す。

入力:
- 既存の canonical data / metrics / cached public artifacts
- `WAVEP_PAPER_LOCALE` / `WAVEP_FIGURE_LOCALE` / `WAVEP_FIGURE_LANG`

出力:
- Part II で参照する source figure 群の localized artifact

前提:
- 各 script は canonical な data / metrics を正として読み、図を再生成する。
- locale ごとの path 分離と text localization は sitecustomize 側で担う。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


_ROOT = Path(__file__).resolve().parents[2]


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_build_commands` の入出力契約と処理意図を定義する。
def _build_commands(root: Path) -> List[List[str]]:
    py = sys.executable or "python"
    commands: List[List[str]] = [
        [py, "-B", str(root / "scripts" / "summary" / "validation_scoreboard.py")],
        [py, "-B", str(root / "scripts" / "llr" / "llr_pmodel_overlay_horizons_noargs.py")],
        [py, "-B", str(root / "scripts" / "llr" / "llr_batch_eval.py")],
        [py, "-B", str(root / "scripts" / "cassini" / "cassini_fig2_overlay.py")],
        [py, "-B", str(root / "scripts" / "theory" / "solar_light_deflection.py")],
        [py, "-B", str(root / "scripts" / "viking" / "update_slides.py")],
        [py, "-B", str(root / "scripts" / "gps" / "plot.py")],
        [py, "-B", str(root / "scripts" / "eht" / "eht_kappa_first_principles_transfer.py")],
        [py, "-B", str(root / "scripts" / "eht" / "eht_shadow_compare.py")],
        [py, "-B", str(root / "scripts" / "eht" / "eht_m87_persistent_shadow_metrics.py")],
        [py, "-B", str(root / "scripts" / "eht" / "eht_kappa_error_budget.py")],
        [py, "-B", str(root / "scripts" / "theory" / "gravitational_redshift_experiments.py")],
        [py, "-B", str(root / "scripts" / "pulsar" / "binary_pulsar_orbital_decay.py")],
        [py, "-B", str(root / "scripts" / "gw" / "gw150914_chirp_phase.py"), "--offline"],
        [py, "-B", str(root / "scripts" / "gw" / "gw_ringdown_qnm.py"), "--offline"],
        [py, "-B", str(root / "scripts" / "gw" / "gw_area_theorem.py"), "--offline"],
        [py, "-B", str(root / "scripts" / "gw" / "gw_imr_consistency.py"), "--offline"],
        [py, "-B", str(root / "scripts" / "gw" / "gw_multi_event_summary.py")],
        [py, "-B", str(root / "scripts" / "xrism" / "xrism_integration.py")],
        [py, "-B", str(root / "scripts" / "xrism" / "fek_relativistic_broadening_isco_constraints.py"), "--plot-only"],
        [py, "-B", str(root / "scripts" / "cosmology" / "sparc_rotation_curve_pmodel_audit.py")],
        [py, "-B", str(root / "scripts" / "theory" / "delta_saturation_constraints.py")],
        [py, "-B", str(root / "scripts" / "summary" / "decisive_falsification.py")],
    ]
    return commands


# 関数: `main` の入出力契約と処理意図を定義する。
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild Part II source figures for the active locale.")
    ap.add_argument("--strict", action="store_true", help="Stop on first failing source figure rebuild.")
    args = ap.parse_args(argv)

    root = _repo_root()
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(root))
    commands = _build_commands(root)
    failures: List[str] = []

    for command in commands:
        try:
            subprocess.run(command, cwd=str(root), check=True, env=env)
        except Exception as exc:
            cmd_text = " ".join(str(item) for item in command)
            failures.append(f"{cmd_text}: {exc}")
            print(f"[warn] part2 localized figure rebuild failed: {cmd_text}\n  {exc}")
            if args.strict:
                return 2

    if failures:
        print(f"[warn] part2 localized figure rebuild finished with failures={len(failures)}")
        return 1

    print(f"[ok] part2 localized figure rebuild finished: commands={len(commands)}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。
if __name__ == "__main__":
    raise SystemExit(main())
