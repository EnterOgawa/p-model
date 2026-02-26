import argparse
import math
import re  # 正規表現モジュールを追加
import sys
from datetime import datetime, timezone
from pathlib import Path

# =============================
# Constants & Settings
# =============================
C = 299792458.0
MU_E = 3.986005e14
F_GPS = -4.442807633e-10
OMEGA_E = 7.292115e-5  # rad/s (Earth rotation, approximate)

# Repo paths
ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402

DATA_DIR = ROOT / "data" / "gps"
OUT_DIR = ROOT / "output" / "private" / "gps"

# 入力ファイル（data/gps）
FILE_RINEX_NAV = DATA_DIR / "BRDC00IGS_R_20252740000_01D_MN.rnx"
FILE_SP3 = DATA_DIR / "IGS0OPSFIN_20252740000_01D_15M_ORB.SP3"
FILE_CLK = DATA_DIR / "IGS0OPSFIN_20252740000_01D_05M_CLK.CLK"

# 出力（output/gps）
OUTPUT_SUMMARY = OUT_DIR / "summary_batch.csv"

# =============================
# Helpers
# =============================
def parse_float(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))

def time_to_utc(year, month, day, hour, minute, second):
    return datetime(year, month, day, hour, minute, int(second), 
                    microsecond=int((second - int(second))*1e6), tzinfo=timezone.utc)

def rms(data):
    if not data: return 0.0
    s = sum(x*x for x in data)
    return math.sqrt(s / len(data))

def linreg_affine(x, y):
    n = len(x)
    if n < 2: return 0.0, 0.0
    sx = sum(x); sy = sum(y)
    sxx = sum(xi*xi for xi in x)
    sxy = sum(xi*yi for xi, yi in zip(x, y))
    det = n*sxx - sx*sx
    if abs(det) < 1e-12: return sum(y)/n, 0.0
    a = (sxx*sy - sx*sxy)/det
    b = (n*sxy - sx*sy)/det
    return a, b


def pmodel_rate(mu: float, r_m: float, v_m_s: float, delta: float) -> float:
    # P-model clock rate (dimensionless):
    #   core: dτ/dt = exp(-mu/(c^2 r)) * sqrt(1 - v^2/c^2)
    #   optional saturation (extension): sqrt((1 - v^2/c^2 + δ0)/(1+δ0))
    g = math.exp(-mu / (C * C * r_m))
    vfac = math.sqrt((1.0 - (v_m_s * v_m_s) / (C * C) + delta) / (1.0 + delta))
    return g * vfac


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s: float):
    return (a[0] * s, a[1] * s, a[2] * s)


def _norm(a) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def inertial_speed_from_sp3_ecef(
    times, positions_xyz_m, omega_e_rad_s: float
) -> list[float]:
    # SP3 positions are ECEF-like. Differentiate in the rotating frame then add ω×r to approximate inertial velocity.
    n = len(times)
    # 条件分岐: `n < 2` を満たす経路を評価する。
    if n < 2:
        return [0.0 for _ in range(n)]

    v_rot = []
    for i in range(n):
        # 条件分岐: `i == 0` を満たす経路を評価する。
        if i == 0:
            dt = (times[i + 1] - times[i]).total_seconds()
            dr = _sub(positions_xyz_m[i + 1], positions_xyz_m[i])
        # 条件分岐: 前段条件が不成立で、`i == n - 1` を追加評価する。
        elif i == n - 1:
            dt = (times[i] - times[i - 1]).total_seconds()
            dr = _sub(positions_xyz_m[i], positions_xyz_m[i - 1])
        else:
            dt = (times[i + 1] - times[i - 1]).total_seconds()
            dr = _sub(positions_xyz_m[i + 1], positions_xyz_m[i - 1])

        # 条件分岐: `dt <= 0` を満たす経路を評価する。

        if dt <= 0:
            v_rot.append((0.0, 0.0, 0.0))
        else:
            v_rot.append(_mul(dr, 1.0 / dt))

    speeds = []
    for i in range(n):
        x, y, z = positions_xyz_m[i]
        wxr = (-omega_e_rad_s * y, omega_e_rad_s * x, 0.0)
        v_inert = _add(v_rot[i], wxr)
        speeds.append(_norm(v_inert))

    return speeds


def dt_rel_from_radius_series(tsec: list[float], r_m: list[float]) -> list[float]:
    # Standard GNSS relativistic correction term (eccentricity effect):
    #   dt_rel(t) = -2 (r·v) / c^2 = -2 r (dr/dt) / c^2
    n = len(r_m)
    # 条件分岐: `n < 2` を満たす経路を評価する。
    if n < 2:
        return [0.0 for _ in range(n)]

    out: list[float] = []
    for i in range(n):
        # 条件分岐: `i == 0` を満たす経路を評価する。
        if i == 0:
            dt = tsec[i + 1] - tsec[i]
            dr = r_m[i + 1] - r_m[i]
        # 条件分岐: 前段条件が不成立で、`i == n - 1` を追加評価する。
        elif i == n - 1:
            dt = tsec[i] - tsec[i - 1]
            dr = r_m[i] - r_m[i - 1]
        else:
            dt = tsec[i + 1] - tsec[i - 1]
            dr = r_m[i + 1] - r_m[i - 1]

        drdt = (dr / dt) if dt != 0.0 else 0.0
        out.append((-2.0 * r_m[i] * drdt) / (C * C))

    return out

# =============================
# Parsers (Revised)
# =============================

def load_igs_clk(filepath):
    data = {}
    print(f"Loading CLK: {filepath} ...")
    with open(filepath, "r") as f:
        for line in f:
            if not line.startswith("AS "): continue
            parts = line.split()
            if len(parts) < 10: continue
            prn = parts[1]
            if not prn.startswith("G"): continue
            
            y, m, d, hh, mm = map(int, parts[2:7])
            ss = float(parts[7])
            clk_bias = float(parts[9])
            t = time_to_utc(y, m, d, hh, mm, ss)
            
            if prn not in data: data[prn] = {}
            data[prn][t] = clk_bias

    return data

def load_igs_sp3(filepath):
    data = {}
    print(f"Loading SP3: {filepath} ...")
    with open(filepath, "r") as f:
        current_time = None
        for line in f:
            # 条件分岐: `line.startswith("* ")` を満たす経路を評価する。
            if line.startswith("* "):
                parts = line.split()
                y, m, d, hh, mm = map(int, parts[1:6])
                ss = float(parts[6])
                current_time = time_to_utc(y, m, d, hh, mm, ss)
            # 条件分岐: 前段条件が不成立で、`line.startswith("PG") and current_time` を追加評価する。
            elif line.startswith("PG") and current_time:
                prn = line[1:4]
                try:
                    x = float(line[4:18]) * 1000.0
                    y = float(line[18:32]) * 1000.0
                    z = float(line[32:46]) * 1000.0
                    if prn not in data: data[prn] = {}
                    data[prn][current_time] = (x, y, z)
                except:
                    continue

    return data

def load_brdc(filepath):
    """ 
    [修正版] Mixed RINEXに対応 + 連結数値対策
    """
    nav_data = {}
    print(f"Loading NAV: {filepath} ...")
    
    with open(filepath, "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        # データ行の先頭パターンをチェック (例: "G01 2025 ...")
        try:
            # 簡易チェック: 1文字目がGか
            sys_id = line[0]
            # 条件分岐: `sys_id != 'G'` を満たす経路を評価する。
            if sys_id != 'G':
                continue
            
            # ★ここが最重要修正★
            # "00-6.123" のように数字とマイナスがくっついている箇所にスペースを入れる
            # ただし "E-05" のような指数表記は壊さない（数字の直後のマイナスのみ対象）

            line_fixed = re.sub(r'(\d)-', r'\1 -', line)
            
            parts = line_fixed.replace("D", "E").split()
            
            # 条件分岐: `len(parts) < 8` を満たす経路を評価する。
            if len(parts) < 8:
                continue
                
            # PRN取得

            prn = parts[0] # "G01"
            
            # 時刻パース
            y = int(parts[1])
            if y < 100: y += 2000
            m, d, hh, mm = map(int, parts[2:6])
            ss = float(parts[6])
            toc = time_to_utc(y, m, d, hh, mm, ss)
            
            # クロックパラメータ (a0, a1, a2)
            a0 = parse_float(parts[7])
            a1 = parse_float(parts[8])
            a2 = parse_float(parts[9])
            
            if prn not in nav_data: nav_data[prn] = []
            nav_data[prn].append({
                "toc": toc,
                "a0": a0, "a1": a1, "a2": a2
            })
            
        except Exception:
            # パースできない行は無視
            continue
            
    return nav_data

def get_brdc_clk(prn, t, nav_db):
    if prn not in nav_db: return None
    best_nav = None
    min_dt = 1e9
    for nav in nav_db[prn]:
        dt = abs((t - nav["toc"]).total_seconds())
        # 条件分岐: `dt < min_dt` を満たす経路を評価する。
        if dt < min_dt:
            min_dt = dt
            best_nav = nav

    if best_nav is None: return None
    
    dt_t = (t - best_nav["toc"]).total_seconds()
    dts = best_nav["a0"] + best_nav["a1"]*dt_t + best_nav["a2"]*(dt_t**2)
    return dts

# =============================
# Main Process
# =============================

def main():
    ap = argparse.ArgumentParser(description="GPS: IGS clock vs BRDC vs P-model（観測比較）")
    ap.add_argument("--delta", type=float, default=0.0, help="速度飽和 δ0（拡張仮説; default: 0.0=disabled）")
    ap.add_argument("--r-ref-m", type=float, default=6_378_137.0, help="基準半径 r_ref [m]（default: 地球半径）")
    ap.add_argument(
        "--ref-include-earth-rotation",
        action="store_true",
        help="基準点（地上）の速度に地球自転を含める（default: 含めない）",
    )
    ap.add_argument(
        "--no-sat-earth-rotation",
        action="store_true",
        help="衛星速度で ω×r 補正を無効化する（default: 補正あり）",
    )
    args = ap.parse_args()

    delta = float(args.delta)
    r_ref = float(args.r_ref_m)
    omega = float(OMEGA_E)

    v_ref = (omega * r_ref) if bool(args.ref_include_earth_rotation) else 0.0
    rate_ref = pmodel_rate(MU_E, r_ref, v_ref, delta=delta)

    print("--- Starting GPS Clock Comparison Batch (All Satellites) ---")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    clk_all = load_igs_clk(FILE_CLK)
    sp3_all = load_igs_sp3(FILE_SP3)
    nav_all = load_brdc(FILE_RINEX_NAV)
    
    # 積集合をとる
    sats = sorted(set(clk_all.keys()) & set(sp3_all.keys()) & set(nav_all.keys()))
    sats = [s for s in sats if s.startswith("G")]
    print(f"Target Satellites ({len(sats)}): {sats}")
    
    results_summary = []
    
    for sat in sats:
        try:
            times = sorted(list(set(clk_all[sat].keys()) & set(sp3_all[sat].keys())))
            # 条件分岐: `len(times) < 10` を満たす経路を評価する。
            if len(times) < 10:
                print(f"Skipping {sat}: Not enough data")
                continue
                
            t_clk = []
            x = []
            igs = []
            brdc = []
            pos_list = []
            t0 = times[0]

            for t in times:
                val_brdc = get_brdc_clk(sat, t, nav_all)
                if val_brdc is None: continue
                
                val_igs = clk_all[sat][t]
                pos = sp3_all[sat][t]
                r_mag = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                
                t_clk.append(t)
                x.append((t - t0).total_seconds())
                igs.append(val_igs)
                brdc.append(val_brdc)
                pos_list.append(pos)

            if not t_clk: continue

            # Residuals
            y_b = [bc - ic for bc, ic in zip(brdc, igs)]
            a_b, b_b = linreg_affine(x, y_b)
            res_b = [y - (a_b + b_b*xi) for y, xi in zip(y_b, x)]

            # P-model: build clock offset from orbit (SP3) and compare to IGS by removing bias+drift.
            v_list = inertial_speed_from_sp3_ecef(
                t_clk, pos_list, omega_e_rad_s=(0.0 if bool(args.no_sat_earth_rotation) else omega)
            )

            r_list = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in pos_list]
            rate_sat = [pmodel_rate(MU_E, r, v, delta=delta) for r, v in zip(r_list, v_list)]
            rate_rel = [rs / rate_ref for rs in rate_sat]

            pmodel_clk = [0.0]
            for i in range(1, len(t_clk)):
                dt = (t_clk[i] - t_clk[i - 1]).total_seconds()
                # 条件分岐: `dt <= 0` を満たす経路を評価する。
                if dt <= 0:
                    pmodel_clk.append(pmodel_clk[-1])
                    continue

                u0 = rate_rel[i - 1] - 1.0
                u1 = rate_rel[i] - 1.0
                pmodel_clk.append(pmodel_clk[-1] + 0.5 * (u0 + u1) * dt)

            # IGS CLK is aligned using broadcast ephemerides (where the eccentricity-related relativistic correction
            # is treated as a separate term). Therefore compare the "clock term" with dt_rel removed.

            dt_rel = dt_rel_from_radius_series(x, r_list)
            pmodel_clk_no_rel = [pc - dtr for pc, dtr in zip(pmodel_clk, dt_rel)]

            y_p = [pc - ic for pc, ic in zip(pmodel_clk_no_rel, igs)]
            a_p, b_p = linreg_affine(x, y_p)
            res_p = [y - (a_p + b_p * xi) for y, xi in zip(y_p, x)]

            rms_b = rms(res_b)
            rms_p = rms(res_p)

            # Save Detail CSV
            filename_detail = OUT_DIR / f"residual_precise_{sat}.csv"
            with open(filename_detail, "wt", encoding="utf-8") as f_det:
                f_det.write(
                    "time_utc,tsec,igs_clk_s,brdc_clk_s,pmodel_clk_s,dt_rel_s,pmodel_clk_no_rel_s,"
                    "r_m,v_m_s,rate_sat,rate_rel,"
                    "res_brdc_s,res_pmodel_s\n"
                )
                for i in range(len(t_clk)):
                    f_det.write(f"{t_clk[i].isoformat()},{x[i]:.3f},"
                                f"{igs[i]:.12e},{brdc[i]:.12e},{pmodel_clk[i]:.12e},{dt_rel[i]:.12e},{pmodel_clk_no_rel[i]:.12e},"
                                f"{r_list[i]:.3f},{v_list[i]:.6f},{rate_sat[i]:.12e},{rate_rel[i]:.12e},"
                                f"{res_b[i]:.12e},{res_p[i]:.12e}\n")
            
            print(
                f"Processed {sat}: Epochs={len(t_clk)}, "
                f"RMS(BRDC-IGS)={rms_b*1e9:.3f}ns, RMS(P-model(dt_rel除去)-IGS)={rms_p*1e9:.3f}ns"
            )
            
            results_summary.append({
                "PRN": sat,
                "RMS_BRDC_m": rms_b * C,
                "RMS_PMODEL_m": rms_p * C,
                "Mean_BRDC_s": sum(y_b)/len(y_b),
                "Max_Res_s": max([abs(v) for v in res_b])
            })
            
        except Exception as e:
            print(f"Error processing {sat}: {e}")
            continue

    # 条件分岐: `results_summary` を満たす経路を評価する。

    if results_summary:
        import csv
        with open(OUTPUT_SUMMARY, "wt", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "PRN",
                    "RMS_BRDC_m",
                    "RMS_PMODEL_m",
                    "Mean_BRDC_s",
                    "Max_Res_s",
                ],
            )
            writer.writeheader()
            writer.writerows(results_summary)

        print("Batch Completed.")

        try:
            worklog.append_event(
                {
                    "event_type": "gps_compare_clocks",
                    "argv": sys.argv,
                    "metrics": {
                        "n_sats": int(len(results_summary)),
                        "date_utc": "2025-10-01",
                    },
                    "outputs": {
                        "summary_batch_csv": OUTPUT_SUMMARY,
                        "out_dir": OUT_DIR,
                    },
                }
            )
        except Exception:
            pass

# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
