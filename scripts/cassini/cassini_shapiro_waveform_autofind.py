# -*- coding: utf-8 -*-
import argparse
import csv
import math
import urllib.parse
import urllib.request
import ssl
from datetime import datetime, timezone
from pathlib import Path

C = 299792458.0
MU_SUN = 1.3271244e20  # m^3/s^2
R_SUN = 6.957e8        # m
AU = 1.495978707e11    # m

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# 関数: `fetch_horizons` の入出力契約と処理意図を定義する。
def fetch_horizons(command: str, start: str, stop: str, step: str, center="500@10") -> str:
    params = {
        "format": "text",
        "COMMAND": f"'{command}'",
        "CENTER": f"'{center}'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "REF_SYSTEM": "'ICRF'",
        "OUT_UNITS": "'KM-S'",
        "CSV_FORMAT": "'YES'",
        "VEC_CORR": "'NONE'",
        "VEC_DELTA_T": "'NO'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
    }
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=180, context=SSL_CTX) as resp:
        return resp.read().decode("utf-8", errors="ignore")

# 関数: `parse_vectors_csv` の入出力契約と処理意図を定義する。

def parse_vectors_csv(txt: str):
    # 条件分岐: `"$$SOE" not in txt or "$$EOE" not in txt` を満たす経路を評価する。
    if "$$SOE" not in txt or "$$EOE" not in txt:
        return None

    block = txt.split("$$SOE")[1].split("$$EOE")[0].strip()

    rows = []
    for line in block.splitlines():
        line = line.strip()
        # 条件分岐: `not line` を満たす経路を評価する。
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        # Expect: JD, Calendar, X,Y,Z,VX,VY,VZ
        if len(parts) < 8:
            continue

        cal = parts[1].replace("A.D.", "").strip()
        t = datetime.strptime(cal, "%Y-%b-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
        vx = float(parts[5]); vy = float(parts[6]); vz = float(parts[7])
        rows.append((t, x, y, z, vx, vy, vz))

    return rows

# 関数: `dot` の入出力契約と処理意図を定義する。

def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
# 関数: `cross` の入出力契約と処理意図を定義する。
def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

# 関数: `norm` の入出力契約と処理意図を定義する。

def norm(a): return math.sqrt(dot(a,a))
# 関数: `sub` の入出力契約と処理意図を定義する。
def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
# 関数: `add` の入出力契約と処理意図を定義する。
def add(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
# 関数: `mul` の入出力契約と処理意図を定義する。
def mul(a, k): return (a[0]*k, a[1]*k, a[2]*k)

# 関数: `impact_b_and_dbdt` の入出力契約と処理意図を定義する。
def impact_b_and_dbdt(rE, vE, rS, vS):
    """
    b = |rE x rS| / |rS - rE|
    bdot computed analytically using u = rE x rS.
    """
    dr = sub(rS, rE)
    dv = sub(vS, vE)

    u = cross(rE, rS)
    up = add(cross(vE, rS), cross(rE, vS))

    U = norm(u)
    D = norm(dr)

    # Safety
    if U == 0.0 or D == 0.0:
        return None

    # dU/dt = (u · u') / |u|

    dU = dot(u, up) / U
    # dD/dt = (dr · dv) / |dr|
    dD = dot(dr, dv) / D

    b = U / D
    # bdot = (D*dU - U*dD) / D^2
    bdot = (D*dU - U*dD) / (D*D)
    return b, bdot, D

# 関数: `shapiro_roundtrip` の入出力契約と処理意図を定義する。

def shapiro_roundtrip(r1, r2, b, gamma=1.0):
    # Cassini Nature 2003 Eq.(1): Δt = 2(1+γ) GM/c^3 ln(4 r1 r2 / b^2)
    return 2.0*(1.0+gamma)*MU_SUN/(C**3) * math.log((4.0*r1*r2)/(b*b))

# 関数: `y_gr_roundtrip` の入出力契約と処理意図を定義する。

def y_gr_roundtrip(b, bdot, gamma=1.0):
    # Cassini Nature 2003 Eq.(2): y = d(Δt)/dt = 4(1+γ) GM/c^3 (1/b) db/dt
    return 4.0*(1.0+gamma)*MU_SUN/(C**3) * (bdot/b)

# 関数: `main` の入出力契約と処理意図を定義する。

def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Cassini Shapiro waveform (Eq1/Eq2; HORIZONS required).")
    parser.add_argument("--start", default="2002-06-06 00:00")
    parser.add_argument("--stop", default="2002-07-07 00:00")
    parser.add_argument("--step", default="1 m")
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="P-model beta (sets gamma = 2*beta - 1). If omitted, --gamma is used.",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="PPN gamma (default: 1.0)")
    args = parser.parse_args()

    start = args.start
    stop = args.stop
    step = args.step  # Use 1-minute for sharper peak; can change to \"5 m\" if needed

    earth_txt = fetch_horizons("399", start, stop, step, center="500@10")
    earth = parse_vectors_csv(earth_txt)
    # 条件分岐: `earth is None` を満たす経路を評価する。
    if earth is None:
        raise RuntimeError("Earth vectors parse failed")

    # Cassini candidates; -82 worked for you

    cass_txt = fetch_horizons("-82", start, stop, step, center="500@10")
    cass = parse_vectors_csv(cass_txt)
    # 条件分岐: `cass is None` を満たす経路を評価する。
    if cass is None:
        raise RuntimeError("Cassini vectors parse failed")

    # maps: time -> (r[m], v[m/s])

    E = {t: ((x*1000.0,y*1000.0,z*1000.0),(vx*1000.0,vy*1000.0,vz*1000.0))
         for (t,x,y,z,vx,vy,vz) in earth}
    S = {t: ((x*1000.0,y*1000.0,z*1000.0),(vx*1000.0,vy*1000.0,vz*1000.0))
         for (t,x,y,z,vx,vy,vz) in cass}

    times = sorted(set(E.keys()) & set(S.keys()))
    # 条件分岐: `len(times) < 1000` を満たす経路を評価する。
    if len(times) < 1000:
        raise RuntimeError(f"Too few common epochs: {len(times)}")

    gamma = args.gamma if args.beta is None else (2.0 * args.beta - 1.0)

    r1_list=[]; r2_list=[]; b_list=[]; bdot_list=[]; dt_list=[]; y_list=[]
    for t in times:
        rE, vE = E[t]
        rS, vS = S[t]
        r1 = norm(rE)
        r2 = norm(rS)

        out = impact_b_and_dbdt(rE, vE, rS, vS)
        # 条件分岐: `out is None` を満たす経路を評価する。
        if out is None:
            continue

        b, bdot, R = out

        dt_rt = shapiro_roundtrip(r1, r2, b, gamma=gamma)
        y = y_gr_roundtrip(b, bdot, gamma=gamma)

        r1_list.append(r1); r2_list.append(r2)
        b_list.append(b); bdot_list.append(bdot)
        dt_list.append(dt_rt); y_list.append(y)

    # find extrema

    idx_bmin = min(range(len(b_list)), key=lambda i: b_list[i])
    idx_ypeak = max(range(len(y_list)), key=lambda i: abs(y_list[i]))

    bmin = b_list[idx_bmin]
    t_bmin = times[idx_bmin]
    ypeak = abs(y_list[idx_ypeak])
    t_ypeak = times[idx_ypeak]

    r2_med = sorted([v/AU for v in r2_list])[len(r2_list)//2]

    print("Samples:", len(b_list))
    print("b_min / R_sun:", bmin / R_SUN, "at", t_bmin.isoformat())
    print("y_peak:", ypeak, "at", t_ypeak.isoformat())
    print("r2 median (AU):", r2_med)
    print("Delta_t range (us):", min(dt_list)*1e6, "to", max(dt_list)*1e6)

    out = out_dir / "cassini_shapiro_waveform_analytic.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_utc","r1_m","r2_m","b_m","bdot_mps","delta_t_s","y_frac"])
        for t,r1,r2,b,bd,dtv,yv in zip(times, r1_list, r2_list, b_list, bdot_list, dt_list, y_list):
            w.writerow([t.isoformat(), f"{r1:.6e}", f"{r2:.6e}", f"{b:.6e}",
                        f"{bd:.6e}", f"{dtv:.12e}", f"{yv:.12e}"])

    print("Wrote:", out)

# 条件分岐: `__name__=="__main__"` を満たす経路を評価する。

if __name__=="__main__":
    main()
