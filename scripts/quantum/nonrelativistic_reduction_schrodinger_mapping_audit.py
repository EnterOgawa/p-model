#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nonrelativistic_reduction_schrodinger_mapping_audit.py

Step 8.7.10:
作用由来の方程式から非相対論極限（Schr写像）へ落とす際の近似順序
（ε_v^2, ε_phi, ε_env）を固定し、運用ゲートを JSON/CSV/PNG で監査する。

出力:
  - output/public/quantum/nonrelativistic_reduction_schrodinger_mapping_audit.json
  - output/public/quantum/nonrelativistic_reduction_schrodinger_mapping_audit.csv
  - output/public/quantum/nonrelativistic_reduction_schrodinger_mapping_audit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_as_float` の入出力契約と処理意図を定義する。

def _as_float(v: Any) -> Optional[float]:
    # 条件分岐: `isinstance(v, (int, float))` を満たす経路を評価する。
    if isinstance(v, (int, float)):
        f = float(v)
        # 条件分岐: `math.isfinite(f)` を満たす経路を評価する。
        if math.isfinite(f):
            return f

    return None


# クラス: `ChannelInput` の責務と境界条件を定義する。

@dataclass
class ChannelInput:
    channel: str
    mass_kg: float
    velocity_m_per_s: float
    spatial_scale_m: float
    envelope_time_s: float
    note: str


# 関数: `_channel_metrics` の入出力契約と処理意図を定義する。

def _channel_metrics(ch: ChannelInput, *, c_m_per_s: float, hbar_j_s: float) -> Dict[str, Any]:
    omega0 = (ch.mass_kg * (c_m_per_s**2)) / hbar_j_s
    eps_v2 = (ch.velocity_m_per_s / c_m_per_s) ** 2
    eps_phi = (9.80665 * ch.spatial_scale_m) / (c_m_per_s**2)
    eps_env = 1.0 / max(omega0 * ch.envelope_time_s, 1e-300)
    eps_max = max(eps_v2, eps_phi, eps_env)
    eps_sum = eps_v2 + eps_phi + eps_env
    return {
        "channel": ch.channel,
        "epsilon_v2": float(eps_v2),
        "epsilon_phi": float(eps_phi),
        "epsilon_env": float(eps_env),
        "epsilon_max": float(eps_max),
        "epsilon_sum": float(eps_sum),
        "mass_kg": float(ch.mass_kg),
        "velocity_m_per_s": float(ch.velocity_m_per_s),
        "spatial_scale_m": float(ch.spatial_scale_m),
        "envelope_time_s": float(ch.envelope_time_s),
        "note": ch.note,
    }


# 関数: `build_pack` の入出力契約と処理意図を定義する。

def build_pack() -> Dict[str, Any]:
    cow_path = ROOT / "output" / "public" / "quantum" / "cow_phase_shift_metrics.json"
    atom_path = ROOT / "output" / "public" / "quantum" / "atom_interferometer_gravimeter_phase_metrics.json"
    grav_path = ROOT / "output" / "public" / "quantum" / "gravity_quantum_interference_delta_predictions.json"

    cow = _read_json(cow_path)
    atom = _read_json(atom_path)
    grav = _read_json(grav_path)

    c_m_per_s = 299_792_458.0
    hbar_j_s = 1.054_571_817e-34

    mass_neutron = _as_float(((cow.get("constants") or {}).get("m_neutron_kg"))) or 1.67492749804e-27
    v0_cow = _as_float(((cow.get("config") or {}).get("v0_m_per_s")) ) or 2_000.0
    h_cow = _as_float(((cow.get("config") or {}).get("H_m")) ) or 0.03
    t_cow = h_cow / v0_cow if v0_cow > 0 else 1.0

    baselines = grav.get("baselines") if isinstance(grav.get("baselines"), dict) else {}
    atom_base = baselines.get("atom_interferometer") if isinstance(baselines.get("atom_interferometer"), dict) else {}
    clock_base = baselines.get("optical_clock_leveling") if isinstance(baselines.get("optical_clock_leveling"), dict) else {}
    constants = grav.get("constants") if isinstance(grav.get("constants"), dict) else {}

    mass_cs = _as_float(constants.get("m_cs133_kg")) or 2.2069469514370953e-25
    v_rec = _as_float(atom_base.get("v_rec_m_per_s")) or 0.007047815654124096
    arm_scale = _as_float(atom_base.get("arm_separation_scale_m")) or 0.0028191262616496385
    t_atom = _as_float(atom_base.get("T_s")) or _as_float((atom.get("config") or {}).get("T_s")) or 0.4

    delta_u = _as_float(clock_base.get("delta_u_geodetic_m2_s2")) or 3915.88
    h_clock = delta_u / 9.80665
    t_clock = 1.0
    v_clock = 0.0

    channels = [
        ChannelInput(
            channel="cow_neutron",
            mass_kg=mass_neutron,
            velocity_m_per_s=v0_cow,
            spatial_scale_m=h_cow,
            envelope_time_s=t_cow,
            note="COW 位相差（H/v0）で包絡時間を定義。",
        ),
        ChannelInput(
            channel="atom_gravimeter",
            mass_kg=mass_cs,
            velocity_m_per_s=v_rec,
            spatial_scale_m=arm_scale,
            envelope_time_s=t_atom,
            note="原子干渉計（recoil速度・アーム分離）を proxy として採用。",
        ),
        ChannelInput(
            channel="optical_clock_leveling_proxy",
            mass_kg=mass_cs,
            velocity_m_per_s=v_clock,
            spatial_scale_m=h_clock,
            envelope_time_s=t_clock,
            note="時計は運動項を 0 とし、重力差の proxy を監査。",
        ),
    ]

    rows = [_channel_metrics(ch, c_m_per_s=c_m_per_s, hbar_j_s=hbar_j_s) for ch in channels]

    threshold = 1.0e-6
    criteria = []
    for row in rows:
        criteria.append(
            {
                "channel": row["channel"],
                "metric": "epsilon_max",
                "value": float(row["epsilon_max"]),
                "threshold": threshold,
                "operator": "<=",
                "pass": bool(float(row["epsilon_max"]) <= threshold),
                "note": "max(ε_v^2, ε_phi, ε_env) <= 1e-6 を非相対論近似の運用ゲートとして固定。",
            }
        )

    fail = [c["channel"] for c in criteria if not bool(c.get("pass"))]
    status = "pass" if not fail else "watch"

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.10", "name": "Nonrelativistic reduction (2.6 -> 2.5) audit"},
        "equations": {
            "starting_point": "From EL (minimal action): D_mu D^mu P + dV/dP* = 0",
            "phase_split": "P = exp(-i m c^2 t / hbar) * psi",
            "reduction": (
                "i hbar d_t psi = [(-i hbar nabla - q A)^2/(2m) + q A0 + m phi] psi "
                "- (hbar^2/(2m c^2)) (d_t + i(qA0+mphi)/hbar)^2 psi + O(c^-4)"
            ),
            "schrodinger_limit": "Drop the c^-2 correction term under epsilon ordering, then recover Eq.(2.5.1).",
        },
        "approximation_order": {
            "epsilon_v2": "(v/c)^2",
            "epsilon_phi": "|phi|/c^2 (operational proxy: g*L/c^2)",
            "epsilon_env": "|d_t^2 psi| / (omega0 |d_t psi|) ~ 1/(omega0*T_env)",
            "gate": "max(epsilon_v2, epsilon_phi, epsilon_env) <= 1e-6",
            "dropped_terms": [
                "O(epsilon_v2^2)",
                "O(epsilon_phi^2)",
                "O(epsilon_v2 * epsilon_phi)",
                "O(epsilon_env)",
            ],
        },
        "inputs": {
            "cow_phase_shift_metrics_json": _rel(cow_path),
            "atom_interferometer_gravimeter_phase_metrics_json": _rel(atom_path),
            "gravity_quantum_interference_delta_predictions_json": _rel(grav_path),
        },
        "channels": rows,
        "criteria": criteria,
        "decision": {
            "nonrelativistic_reduction_gate": status,
            "fail_channels": fail,
            "rule": "All channel epsilon_max <= threshold => pass; otherwise watch.",
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]], criteria: List[Dict[str, Any]]) -> None:
    merged: Dict[str, Dict[str, Any]] = {str(r["channel"]): dict(r) for r in rows if isinstance(r, dict)}
    for c in criteria:
        # 条件分岐: `not isinstance(c, dict)` を満たす経路を評価する。
        if not isinstance(c, dict):
            continue

        ch = str(c.get("channel") or "")
        # 条件分岐: `ch in merged` を満たす経路を評価する。
        if ch in merged:
            merged[ch]["gate_metric"] = c.get("metric")
            merged[ch]["gate_threshold"] = c.get("threshold")
            merged[ch]["gate_pass"] = c.get("pass")

    fieldnames = [
        "channel",
        "epsilon_v2",
        "epsilon_phi",
        "epsilon_env",
        "epsilon_max",
        "epsilon_sum",
        "velocity_m_per_s",
        "spatial_scale_m",
        "envelope_time_s",
        "gate_metric",
        "gate_threshold",
        "gate_pass",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(merged.keys()):
            writer.writerow(merged[key])


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(path: Path, rows: List[Dict[str, Any]], threshold: float) -> None:
    labels = [str(r.get("channel") or "") for r in rows]
    eps_v2 = [float(r.get("epsilon_v2", math.nan)) for r in rows]
    eps_phi = [float(r.get("epsilon_phi", math.nan)) for r in rows]
    eps_env = [float(r.get("epsilon_env", math.nan)) for r in rows]
    eps_max = [float(r.get("epsilon_max", math.nan)) for r in rows]

    y = np.arange(len(labels))
    h = 0.22

    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=180)
    ax.barh(y - h, eps_v2, height=h, color="#1d4ed8", label="epsilon_v2")
    ax.barh(y, eps_phi, height=h, color="#f59e0b", label="epsilon_phi")
    ax.barh(y + h, eps_env, height=h, color="#2f9e44", label="epsilon_env")
    ax.scatter(eps_max, y, marker="D", color="#111827", s=24, label="epsilon_max")
    ax.axvline(threshold, linestyle="--", color="#6b7280", linewidth=1.2, label="gate threshold")
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.set_xlabel("dimensionless scale (log)")
    ax.set_title("Nonrelativistic reduction audit (2.6 -> 2.5 Schr mapping)")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate nonrelativistic-reduction (Schr mapping) audit pack.")
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "nonrelativistic_reduction_schrodinger_mapping_audit.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "nonrelativistic_reduction_schrodinger_mapping_audit.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "nonrelativistic_reduction_schrodinger_mapping_audit.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。
    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # 条件分岐: `not out_csv.is_absolute()` を満たす経路を評価する。

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    # 条件分岐: `not out_png.is_absolute()` を満たす経路を評価する。

    if not out_png.is_absolute():
        out_png = (ROOT / out_png).resolve()

    payload = build_pack()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = payload.get("channels") if isinstance(payload.get("channels"), list) else []
    criteria = payload.get("criteria") if isinstance(payload.get("criteria"), list) else []
    _write_csv(out_csv, rows if isinstance(rows, list) else [], criteria if isinstance(criteria, list) else [])
    _plot(out_png, rows if isinstance(rows, list) else [], threshold=1.0e-6)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_nonrelativistic_reduction_schrodinger_mapping_audit",
                "phase": "8.7.10",
                "inputs": payload.get("inputs"),
                "outputs": {
                    "nonrelativistic_reduction_schrodinger_mapping_audit_json": _rel(out_json),
                    "nonrelativistic_reduction_schrodinger_mapping_audit_csv": _rel(out_csv),
                    "nonrelativistic_reduction_schrodinger_mapping_audit_png": _rel(out_png),
                },
                "decision": payload.get("decision"),
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
