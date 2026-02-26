# -*- coding: utf-8 -*-
import argparse
import csv
import math
import ssl
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

C = 299792458.0
MU_SUN = 1.3271244e20
R_SUN = 6.957e8
AU = 1.495978707e11

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# 関数: `fetch_horizons` の入出力契約と処理意図を定義する。
def fetch_horizons(command: str, start: str, stop: str, step: str, center="500@0") -> str:
    params = {
        "format": "text",
        "COMMAND": f"'{command}'",
        "CENTER": f"'{center}'",          # SSB
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
        raise RuntimeError("Missing $$SOE/$$EOE in HORIZONS response")

    block = txt.split("$$SOE")[1].split("$$EOE")[0].strip()
    rows=[]
    for line in block.splitlines():
        parts=[p.strip() for p in line.strip().split(",")]
        # 条件分岐: `len(parts) < 8` を満たす経路を評価する。
        if len(parts) < 8:
            continue

        cal = parts[1].replace("A.D.", "").strip()
        t = datetime.strptime(cal, "%Y-%b-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        x=float(parts[2])*1000.0; y=float(parts[3])*1000.0; z=float(parts[4])*1000.0
        vx=float(parts[5])*1000.0; vy=float(parts[6])*1000.0; vz=float(parts[7])*1000.0
        rows.append((t, (x,y,z), (vx,vy,vz)))

    return rows

# 関数: `dot` の入出力契約と処理意図を定義する。

def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
# 関数: `cross` の入出力契約と処理意図を定義する。
def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

# 関数: `norm` の入出力契約と処理意図を定義する。

def norm(a): return math.sqrt(dot(a,a))
# 関数: `sub` の入出力契約と処理意図を定義する。
def sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
# 関数: `add` の入出力契約と処理意図を定義する。
def add(a,b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

# 関数: `impact_b_and_bdot` の入出力契約と処理意図を定義する。
def impact_b_and_bdot(rE, vE, rS, vS):
    dr = sub(rS, rE)
    dv = sub(vS, vE)

    u = cross(rE, rS)
    up = add(cross(vE, rS), cross(rE, vS))

    U = norm(u)
    D = norm(dr)
    # 条件分岐: `U == 0.0 or D == 0.0` を満たす経路を評価する。
    if U == 0.0 or D == 0.0:
        return None

    dU = dot(u, up) / U
    dD = dot(dr, dv) / D

    b = U / D
    bdot = (D*dU - U*dD) / (D*D)
    return b, bdot

# 関数: `shapiro_roundtrip` の入出力契約と処理意図を定義する。

def shapiro_roundtrip(r1, r2, b, gamma=1.0):
    # Cassini Eq(1) (b-approx form)
    return 2.0*(1.0+gamma)*MU_SUN/(C**3) * math.log((4.0*r1*r2)/(b*b))

# 関数: `y_gr_roundtrip` の入出力契約と処理意図を定義する。

def y_gr_roundtrip(b, bdot, gamma=1.0):
    # Cassini Eq(2)
    return 4.0*(1.0+gamma)*MU_SUN/(C**3) * (bdot/b)

# 関数: `main` の入出力契約と処理意図を定義する。

def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Cassini Shapiro (barycentric; Eq1/Eq2; HORIZONS required).")
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
    step = args.step

    # All relative to SSB
    earth = parse_vectors_csv(fetch_horizons("399", start, stop, step, center="500@0"))
    cass  = parse_vectors_csv(fetch_horizons("-82", start, stop, step, center="500@0"))
    sun   = parse_vectors_csv(fetch_horizons("10",  start, stop, step, center="500@0"))

    E = {t:(r,v) for (t,r,v) in earth}
    S = {t:(r,v) for (t,r,v) in cass}
    U = {t:(r,v) for (t,r,v) in sun}

    times = sorted(set(E.keys()) & set(S.keys()) & set(U.keys()))
    # 条件分岐: `len(times) < 1000` を満たす経路を評価する。
    if len(times) < 1000:
        raise RuntimeError(f"Too few common epochs: {len(times)}")

    gamma = args.gamma if args.beta is None else (2.0 * args.beta - 1.0)
    b_list=[]; y_list=[]; dt_list=[]
    r2_list=[]

    for t in times:
        (rE,vE) = E[t]
        (rS,vS) = S[t]
        (rU,vU) = U[t]

        # Convert to Sun-centered with Sun motion removed
        rE0 = sub(rE, rU); vE0 = sub(vE, vU)
        rS0 = sub(rS, rU); vS0 = sub(vS, vU)

        out = impact_b_and_bdot(rE0, vE0, rS0, vS0)
        # 条件分岐: `out is None` を満たす経路を評価する。
        if out is None:
            continue

        b, bdot = out

        r1 = norm(rE0)
        r2 = norm(rS0)

        dt_rt = shapiro_roundtrip(r1, r2, b, gamma=gamma)
        y = y_gr_roundtrip(b, bdot, gamma=gamma)

        b_list.append(b)
        y_list.append(y)
        dt_list.append(dt_rt)
        r2_list.append(r2)

    idx_bmin = min(range(len(b_list)), key=lambda i: b_list[i])
    idx_ypeak = max(range(len(y_list)), key=lambda i: abs(y_list[i]))

    bmin=b_list[idx_bmin]
    ypeak=abs(y_list[idx_ypeak])
    t_bmin=times[idx_bmin]
    t_ypeak=times[idx_ypeak]
    r2_med = sorted([r/AU for r in r2_list])[len(r2_list)//2]

    print("Samples:", len(b_list))
    print("b_min / R_sun:", bmin/R_SUN, "at", t_bmin.isoformat())
    print("y_peak:", ypeak, "at", t_ypeak.isoformat())
    print("r2 median (AU):", r2_med)
    print("Delta_t range (us):", min(dt_list)*1e6, "to", max(dt_list)*1e6)

    out = out_dir / "cassini_shapiro_barycentric.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["time_utc","b_m","delta_t_s","y_frac"])
        for t,b,dtv,yv in zip(times, b_list, dt_list, y_list):
            w.writerow([t.isoformat(), f"{b:.6e}", f"{dtv:.12e}", f"{yv:.12e}"])

    print("Wrote:", out)

# 条件分岐: `__name__=="__main__"` を満たす経路を評価する。

if __name__=="__main__":
    main()
