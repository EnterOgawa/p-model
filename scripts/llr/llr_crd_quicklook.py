#!/usr/bin/env python3
"""
llr_crd_quicklook.py

CRD (ILRS Consolidated Laser Ranging Data) の "11 Normal Point" を抽出して、
CSV と簡易グラフ（往復時間・距離の時系列）を作る“検証用ワンショット”スクリプト。

主に LLR を想定していますが、CRDのNormal Point(11)が入っていれば SLR でも動きます。

使い方:
  python llr_crd_quicklook.py path/to/file.crd
  python llr_crd_quicklook.py path/to/dir --recursive
  python llr_crd_quicklook.py path/to/file.crd.gz

出力:
  out/<inputname>_npt11.csv
  out/<inputname>_summary.json
  out/<inputname>_range_timeseries.png
  out/<inputname>_tof_timeseries.png
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

C = 299_792_458.0  # m/s


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix.lower() == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _rtype(line: str) -> str:
    if not line:
        return ""
    head2 = line[:2].strip().upper()
    if head2:
        return head2
    return line.split()[0].strip().upper()


def _is_na(tok: str) -> bool:
    return tok.lower() in ("na", "nan")


def _to_int(tok: str) -> Optional[int]:
    if tok is None or _is_na(tok):
        return None
    try:
        return int(tok)
    except ValueError:
        return None


def _to_float(tok: str) -> Optional[float]:
    if tok is None or _is_na(tok):
        return None
    try:
        return float(tok)
    except ValueError:
        return None


@dataclass
class Context:
    station_name: Optional[str] = None
    target_name: Optional[str] = None
    session_day_utc: Optional[datetime] = None
    range_type: Optional[int] = None


def _parse_h2(tokens: List[str], ctx: Context) -> None:
    if len(tokens) >= 2 and not _is_na(tokens[1]):
        ctx.station_name = tokens[1]


def _parse_h3(tokens: List[str], ctx: Context) -> None:
    if len(tokens) >= 2 and not _is_na(tokens[1]):
        ctx.target_name = tokens[1]


def _parse_h4(tokens: List[str], ctx: Context) -> None:
    def _dt_at(i: int) -> Optional[datetime]:
        try:
            y = _to_int(tokens[i]); mo = _to_int(tokens[i+1]); d = _to_int(tokens[i+2])
            hh = _to_int(tokens[i+3]); mm = _to_int(tokens[i+4]); ss = _to_int(tokens[i+5])
        except Exception:
            return None
        if None in (y, mo, d, hh, mm, ss):
            return None
        return datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc)

    start = _dt_at(2)
    if start:
        ctx.session_day_utc = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)

    rt = None
    if len(tokens) >= 21:
        rt = _to_int(tokens[20])
    if rt is None and len(tokens) >= 2:
        rt = _to_int(tokens[-2]) if len(tokens) >= 2 else None
    ctx.range_type = rt


def _epoch_from_sod(ctx: Context, sod: Optional[float]) -> Optional[datetime]:
    if ctx.session_day_utc is None or sod is None:
        return None
    return ctx.session_day_utc + timedelta(seconds=float(sod))


def _one_way_range_m(range_type: Optional[int], tof_s: Optional[float], default_two_way: bool = True) -> Optional[float]:
    if tof_s is None:
        return None
    if range_type == 2:
        return (tof_s * C) / 2.0
    if range_type == 1:
        return tof_s * C
    if default_two_way:
        return (tof_s * C) / 2.0
    return None


def parse_npt11(path: Path, default_two_way: bool = True) -> pd.DataFrame:
    ctx = Context()
    rows: List[Dict[str, Any]] = []

    with _open_text(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            rt = _rtype(line)
            tokens = line.split()

            if rt == "H2":
                _parse_h2(tokens, ctx)
            elif rt == "H3":
                _parse_h3(tokens, ctx)
            elif rt == "H4":
                _parse_h4(tokens, ctx)
            elif rt == "11":
                sod = _to_float(tokens[1]) if len(tokens) > 1 else None
                tof = _to_float(tokens[2]) if len(tokens) > 2 else None
                epoch = _epoch_from_sod(ctx, sod)
                rng = _one_way_range_m(ctx.range_type, tof, default_two_way=default_two_way)
                rows.append({
                    "file": path.name,
                    "lineno": lineno,
                    "station": ctx.station_name,
                    "target": ctx.target_name,
                    "session_day_utc": ctx.session_day_utc.isoformat() if ctx.session_day_utc else None,
                    "range_type": ctx.range_type,
                    "seconds_of_day": sod,
                    "tof_s": tof,
                    "epoch_utc": epoch.isoformat() if epoch else None,
                    "one_way_range_m": rng,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Pandas may infer a single datetime format and coerce valid ISO8601 strings
        # (with microseconds / tz offsets) into NaT. Force ISO8601 parsing.
        df["epoch_utc"] = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce", format="ISO8601")
        df = df.sort_values("epoch_utc")
    return df


def quicklook(df: pd.DataFrame, out_prefix: Path) -> Dict[str, Any]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df_valid = df.dropna(subset=["epoch_utc"]).copy() if (not df.empty and "epoch_utc" in df.columns) else df

    # If the file contains multiple stations/targets, pick the most common pair for a clean public plot.
    df_sel = df_valid
    selected_station = None
    selected_target = None
    if not df_valid.empty and "station" in df_valid.columns and "target" in df_valid.columns:
        st = df_valid["station"].astype(str).str.strip().str.upper()
        tg = df_valid["target"].astype(str).str.strip().str.lower()
        if st.notna().any():
            selected_station = str(st.value_counts().index[0])
        if tg.notna().any():
            selected_target = str(tg.value_counts().index[0])
        if selected_station and selected_target:
            mask = (st == selected_station) & (tg == selected_target)
            df_sel = df_valid.loc[mask].copy()

    summary: Dict[str, Any] = {
        "n_npt11_total": int(len(df)),
        "n_valid_epoch_total": int(len(df_valid)),
        "n_npt11": int(len(df_sel)),
        "n_valid_epoch": int(len(df_sel)),
        "selected_station": selected_station,
        "selected_target": selected_target,
        "station": None,
        "target": None,
        "range_type": None,
        "epoch_min_utc": None,
        "epoch_max_utc": None,
        "tof_s_min": None,
        "tof_s_max": None,
        "one_way_range_m_min": None,
        "one_way_range_m_max": None,
    }

    if df.empty:
        return summary

    summary["station"] = df_sel["station"].dropna().iloc[0] if (not df_sel.empty and df_sel["station"].notna().any()) else None
    summary["target"] = df_sel["target"].dropna().iloc[0] if (not df_sel.empty and df_sel["target"].notna().any()) else None
    summary["range_type"] = int(df_sel["range_type"].dropna().iloc[0]) if (not df_sel.empty and df_sel["range_type"].notna().any()) else None
    if not df_sel.empty and df_sel["epoch_utc"].notna().any():
        summary["epoch_min_utc"] = df_sel["epoch_utc"].min().isoformat()
        summary["epoch_max_utc"] = df_sel["epoch_utc"].max().isoformat()
    if not df_sel.empty and df_sel["tof_s"].notna().any():
        summary["tof_s_min"] = float(df_sel["tof_s"].min())
        summary["tof_s_max"] = float(df_sel["tof_s"].max())
    if not df_sel.empty and df_sel["one_way_range_m"].notna().any():
        summary["one_way_range_m_min"] = float(df_sel["one_way_range_m"].min())
        summary["one_way_range_m_max"] = float(df_sel["one_way_range_m"].max())

    # CSV
    df.to_csv(out_prefix.with_suffix("").as_posix() + "_npt11.csv", index=False)

    # Plot: range time series
    if not df_sel.empty and df_sel["one_way_range_m"].notna().any():
        _set_japanese_font()
        plt.figure(figsize=(10, 4.5))
        plt.plot(df_sel["epoch_utc"], df_sel["one_way_range_m"])
        plt.xlabel("UTC時刻")
        plt.ylabel("片道距離 [m]（TOFから換算）")
        plt.title("CRD Normal Point（record 11）：片道距離 時系列")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_prefix.with_suffix("").as_posix() + "_range_timeseries.png", dpi=200)
        plt.close()

        # 過去版で生成していた「距離ヒストグラム」は不要（レポート自動追加の混乱回避）
        old_hist = Path(out_prefix.with_suffix("").as_posix() + "_range_hist.png")
        if old_hist.exists():
            try:
                old_hist.unlink()
            except Exception:
                pass

    # Plot: tof time series
    if not df_sel.empty and df_sel["tof_s"].notna().any():
        _set_japanese_font()
        plt.figure(figsize=(10, 4.5))
        plt.plot(df_sel["epoch_utc"], df_sel["tof_s"])
        plt.xlabel("UTC時刻")
        plt.ylabel("飛行時間 TOF [s]")
        plt.title("CRD Normal Point（record 11）：TOF 時系列")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_prefix.with_suffix("").as_posix() + "_tof_timeseries.png", dpi=200)
        plt.close()

    # Summary JSON
    with open(out_prefix.with_suffix("").as_posix() + "_summary.json", "w", encoding="utf-8") as w:
        json.dump(summary, w, ensure_ascii=False, indent=2)

    return summary


def iter_inputs(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        pats = [
            "*.crd", "*.CRD",
            "*.npt", "*.NPT",
            "*.np2", "*.NP2",
            "*.crd.gz", "*.CRD.gz",
            "*.npt.gz", "*.NPT.gz",
            "*.np2.gz", "*.NP2.gz",
        ]
        files: List[Path] = []
        for pat in pats:
            files.extend(path.rglob(pat) if recursive else path.glob(pat))
        return sorted(set(files))
    raise FileNotFoundError(path)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_outdir = repo / "output" / "private" / "llr"

    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="CRD file or directory")
    ap.add_argument("--recursive", action="store_true", help="If input is a directory, search recursively")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output directory")
    ap.add_argument("--assume-two-way", action="store_true",
                    help="If H4 range_type is missing/unknown, assume two-way (recommended for LLR)")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    default_two_way = bool(args.assume_two_way)

    inputs = iter_inputs(in_path, args.recursive)
    if not inputs:
        print("No input files found.")
        return

    all_rows = []
    for p in inputs:
        df = parse_npt11(p, default_two_way=default_two_way)
        if df.empty:
            print(f"[skip] {p} : no record 11 found")
            continue
        prefix = outdir / p.stem
        summary = quicklook(df, prefix)
        print(f"[ok] {p.name}  n11={summary['n_npt11']}  station={summary['station']}  target={summary['target']}")
        all_rows.append(df)

    if len(all_rows) >= 2:
        merged = pd.concat(all_rows, ignore_index=True).sort_values("epoch_utc")
        merged_path = outdir / "merged_npt11.csv"
        merged.to_csv(merged_path, index=False)
        print(f"[merged] {merged_path}  rows={len(merged)}")


if __name__ == "__main__":
    main()
