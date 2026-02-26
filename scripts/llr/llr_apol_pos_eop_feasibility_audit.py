#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
APOL_META_PATH = ROOT / "data" / "llr" / "stations" / "apol.json"
SNX_ROOT = ROOT / "data" / "llr" / "pos_eop" / "snx"
OUT_DIR = ROOT / "output" / "private" / "llr"
OUT_JSON = OUT_DIR / "llr_apol_pos_eop_feasibility_audit.json"
OUT_CSV = OUT_DIR / "llr_apol_pos_eop_feasibility_audit.csv"
OUT_PNG = OUT_DIR / "llr_apol_pos_eop_feasibility_audit.png"


def _parse_station_xyz(path: Path) -> Dict[str, Tuple[float, float, float]]:
    stations: Dict[str, Dict[str, float]] = {}
    in_est = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            # 条件分岐: `line.startswith("+SOLUTION/ESTIMATE")` を満たす経路を評価する。
            if line.startswith("+SOLUTION/ESTIMATE"):
                in_est = True
                continue

            # 条件分岐: `in_est and line.startswith("-SOLUTION/ESTIMATE")` を満たす経路を評価する。

            if in_est and line.startswith("-SOLUTION/ESTIMATE"):
                break

            # 条件分岐: `not in_est` を満たす経路を評価する。

            if not in_est:
                continue

            # 条件分岐: `not line.strip() or line.startswith("*")` を満たす経路を評価する。

            if not line.strip() or line.startswith("*"):
                continue

            parts = line.split()
            # 条件分岐: `len(parts) < 10` を満たす経路を評価する。
            if len(parts) < 10:
                continue

            typ = str(parts[1]).strip().upper()
            # 条件分岐: `typ not in ("STAX", "STAY", "STAZ")` を満たす経路を評価する。
            if typ not in ("STAX", "STAY", "STAZ"):
                continue

            code = str(parts[2]).strip()
            try:
                value = float(parts[-2])
            except Exception:
                continue

            rec = stations.setdefault(code, {})
            rec[typ] = float(value)

    out: Dict[str, Tuple[float, float, float]] = {}
    for code, rec in stations.items():
        # 条件分岐: `all(k in rec for k in ("STAX", "STAY", "STAZ"))` を満たす経路を評価する。
        if all(k in rec for k in ("STAX", "STAY", "STAZ")):
            out[code] = (float(rec["STAX"]), float(rec["STAY"]), float(rec["STAZ"]))

    return out


def _ecef_from_geodetic(lat_deg: float, lon_deg: float, h_m: float) -> Tuple[float, float, float]:
    a = 6_378_137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    n = a / math.sqrt(1.0 - e2 * (math.sin(lat) ** 2))
    x = (n + float(h_m)) * math.cos(lat) * math.cos(lon)
    y = (n + float(h_m)) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - e2) + float(h_m)) * math.sin(lat)
    return (x, y, z)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `not APOL_META_PATH.exists()` を満たす経路を評価する。
    if not APOL_META_PATH.exists():
        raise FileNotFoundError(f"APOL metadata missing: {APOL_META_PATH}")

    apol_meta = json.loads(APOL_META_PATH.read_text(encoding="utf-8"))
    apol_pad_id = int(apol_meta.get("cdp_pad_id"))
    apol_code = str(apol_pad_id)

    lat = float(apol_meta.get("lat_deg"))
    lon = float(apol_meta.get("lon_deg"))
    h_m = float(apol_meta.get("height_m"))
    apol_xyz = _ecef_from_geodetic(lat, lon, h_m)

    snx_paths = sorted(SNX_ROOT.glob("*/*/pos_eop_*.snx.gz"))
    code_counts: Dict[str, int] = {}
    code_xyz_sample: Dict[str, Tuple[float, float, float]] = {}
    code_file_sample: Dict[str, str] = {}
    apol_hits: list[str] = []

    for p in snx_paths:
        station_xyz = _parse_station_xyz(p)
        # 条件分岐: `apol_code in station_xyz` を満たす経路を評価する。
        if apol_code in station_xyz:
            apol_hits.append(str(p.relative_to(ROOT)))

        for code, xyz in station_xyz.items():
            code_counts[code] = int(code_counts.get(code, 0)) + 1
            # 条件分岐: `code not in code_xyz_sample` を満たす経路を評価する。
            if code not in code_xyz_sample:
                code_xyz_sample[code] = xyz
                code_file_sample[code] = str(p.relative_to(ROOT))

    rows = []
    for code, xyz in code_xyz_sample.items():
        dx = float(xyz[0] - apol_xyz[0])
        dy = float(xyz[1] - apol_xyz[1])
        dz = float(xyz[2] - apol_xyz[2])
        dist_m = float(math.sqrt(dx * dx + dy * dy + dz * dz))
        rows.append(
            {
                "station_code": str(code),
                "distance_from_apol_geodetic_km": dist_m / 1000.0,
                "n_files_with_code": int(code_counts.get(code, 0)),
                "sample_snx_path": str(code_file_sample.get(code, "")),
            }
        )

    df = pd.DataFrame(rows)
    # 条件分岐: `len(df) > 0` を満たす経路を評価する。
    if len(df) > 0:
        df = df.sort_values(["distance_from_apol_geodetic_km", "station_code"]).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    nearest_code = None
    nearest_km = float("nan")
    # 条件分岐: `len(df) > 0` を満たす経路を評価する。
    if len(df) > 0:
        nearest_code = str(df.iloc[0]["station_code"])
        nearest_km = float(df.iloc[0]["distance_from_apol_geodetic_km"])

    alias_safe_threshold_km = 5.0
    alias_safe = bool(np.isfinite(nearest_km) and nearest_km <= alias_safe_threshold_km)
    apol_present = len(apol_hits) > 0

    # 条件分岐: `apol_present` を満たす経路を評価する。
    if apol_present:
        decision = "pass"
        reasons = ["APOL pad code is present in cached pos+eop SINEX files."]
    # 条件分岐: 前段条件が不成立で、`alias_safe` を追加評価する。
    elif alias_safe:
        decision = "watch"
        reasons = [
            "APOL pad code is absent in cached pos+eop SINEX files.",
            "A near-code candidate exists, but alias mapping requires primary-source confirmation.",
        ]
    else:
        decision = "watch"
        reasons = [
            "APOL pad code is absent in cached pos+eop SINEX files.",
            "Nearest available station code is too far for safe alias mapping.",
            "APOL remains slrlog-only until primary-source pos+eop coordinates are available.",
        ]

    topn = min(10, len(df))
    # 条件分岐: `topn > 0` を満たす経路を評価する。
    if topn > 0:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        top = df.head(topn).copy()
        labels = [str(x) for x in top["station_code"].tolist()]
        vals = [float(x) for x in top["distance_from_apol_geodetic_km"].tolist()]
        colors = ["#2ca02c" if i == 0 else "#4c78a8" for i in range(topn)]
        ax.bar(labels, vals, color=colors)
        ax.axhline(alias_safe_threshold_km, color="#ff7f0e", linestyle="--", linewidth=1.6, label="alias-safe threshold")
        ax.set_ylabel("distance from APOL geodetic [km]")
        ax.set_xlabel("pos+eop station code")
        ax.set_title("APOL pos+eop feasibility audit (nearest station-code distances)")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
        plt.close(fig)

    summary: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "inputs": {
            "apol_metadata": str(APOL_META_PATH.relative_to(ROOT)),
            "snx_root": str(SNX_ROOT.relative_to(ROOT)),
        },
        "metrics": {
            "n_snx_files_scanned": int(len(snx_paths)),
            "n_unique_station_codes": int(len(code_counts)),
            "apol_pad_id": int(apol_pad_id),
            "apol_code_present_in_cache": bool(apol_present),
            "n_files_with_apol_code": int(len(apol_hits)),
            "nearest_station_code": nearest_code,
            "nearest_distance_km": nearest_km,
            "alias_safe_threshold_km": alias_safe_threshold_km,
            "alias_safe_candidate": bool(alias_safe),
        },
        "apol_reference": {
            "site_name": apol_meta.get("site_name"),
            "lat_deg": lat,
            "lon_deg": lon,
            "height_m": h_m,
            "ecef_from_geodetic_m": {
                "x_m": float(apol_xyz[0]),
                "y_m": float(apol_xyz[1]),
                "z_m": float(apol_xyz[2]),
            },
        },
        "sample_apol_hits": apol_hits[:10],
        "reasons": reasons,
        "recommended_next": [
            "APOLの一次座標I/F（pos+eop対応）の取得手段を確認する。",
            "取得できるまで iers_station_coord_unified は watch を維持する。",
        ],
        "artifacts": {
            "csv": str(OUT_CSV.relative_to(ROOT)),
            "png": str(OUT_PNG.relative_to(ROOT)) if OUT_PNG.exists() else None,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "n_snx_files_scanned": len(snx_paths), "n_files_with_apol_code": len(apol_hits), "nearest_station_code": nearest_code, "nearest_distance_km": nearest_km}, ensure_ascii=False))
    print(f"[ok] {OUT_JSON.relative_to(ROOT)}")
    print(f"[ok] {OUT_CSV.relative_to(ROOT)}")
    # 条件分岐: `OUT_PNG.exists()` を満たす経路を評価する。
    if OUT_PNG.exists():
        print(f"[ok] {OUT_PNG.relative_to(ROOT)}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
