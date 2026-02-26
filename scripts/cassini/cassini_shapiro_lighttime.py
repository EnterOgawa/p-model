# -*- coding: utf-8 -*-
import argparse
import csv
import math
import urllib.parse
import urllib.request
import ssl
from datetime import datetime, timezone, timedelta
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
# 関数: `sub` の入出力契約と処理意図を定義する。
def sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
# 関数: `add` の入出力契約と処理意図を定義する。
def add(a,b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
# 関数: `mul` の入出力契約と処理意図を定義する。
def mul(a,k): return (a[0]*k, a[1]*k, a[2]*k)
# 関数: `norm` の入出力契約と処理意図を定義する。
def norm(a): return math.sqrt(dot(a,a))

# Hermite interpolation for position using (r0,v0) and (r1,v1)
def hermite(r0,v0,r1,v1, u, h):
    # u in [0,1], h = dt seconds
    u2=u*u; u3=u2*u
    h00 =  2*u3 - 3*u2 + 1
    h10 =      u3 - 2*u2 + u
    h01 = -2*u3 + 3*u2
    h11 =      u3 -   u2
    return add(
        add(mul(r0,h00), mul(v0, h10*h)),
        add(mul(r1,h01), mul(v1, h11*h))
    )

# クラス: `StateTable` の責務と境界条件を定義する。

class StateTable:
    # 関数: `__init__` の入出力契約と処理意図を定義する。
    def __init__(self, rows):
        self.t=[r[0] for r in rows]
        self.r=[r[1] for r in rows]
        self.v=[r[2] for r in rows]
        self.dt=(self.t[1]-self.t[0]).total_seconds()

    # 関数: `state` の入出力契約と処理意図を定義する。

    def state(self, tq: datetime):
        # assume tq within range; find i such that t[i] <= tq <= t[i+1]
        # uniform grid, so compute index directly
        t0=self.t[0]
        s=(tq-t0).total_seconds()
        i=int(s//self.dt)
        if i < 0: i=0
        if i >= len(self.t)-1: i=len(self.t)-2
        t_i=self.t[i]
        u=(tq-t_i).total_seconds()/self.dt
        r=hermite(self.r[i], self.v[i], self.r[i+1], self.v[i+1], u, self.dt)
        # velocity not needed for delay, but could be obtained by derivative; skip for now
        return r

# 関数: `shapiro_oneway` の入出力契約と処理意図を定義する。

def shapiro_oneway(r1, r2, R, gamma=1.0):
    # One-way Shapiro delay (weak field): (1+gamma) GM/c^3 ln((r1+r2+R)/(r1+r2-R))
    num = (r1 + r2 + R)
    den = (r1 + r2 - R)
    return (1.0+gamma)*MU_SUN/(C**3) * math.log(num/den)

# 関数: `solve_downlink_time` の入出力契約と処理意図を定義する。

def solve_downlink_time(tR, earth_tab, cass_tab, gamma=1.0, iters=8):
    # Solve tR = tB + (R/c + dt_shapiro) for tB
    tB = tR - timedelta(seconds=3600)  # initial guess ~1h
    rR = earth_tab.state(tR)
    for _ in range(iters):
        rB = cass_tab.state(tB)
        R = norm(sub(rR, rB))
        dtS = shapiro_oneway(norm(rR), norm(rB), R, gamma=gamma)
        tB_new = tR - timedelta(seconds=(R/C + dtS))
        # 条件分岐: `abs((tB_new - tB).total_seconds()) < 1e-6` を満たす経路を評価する。
        if abs((tB_new - tB).total_seconds()) < 1e-6:
            break

        tB = tB_new

    return tB

# 関数: `solve_uplink_time` の入出力契約と処理意図を定義する。

def solve_uplink_time(tB, earth_tab, cass_tab, gamma=1.0, iters=8):
    # Solve tB = tT + (R/c + dt_shapiro) for tT
    tT = tB - timedelta(seconds=3600)  # initial guess
    rB = cass_tab.state(tB)
    for _ in range(iters):
        rT = earth_tab.state(tT)
        R = norm(sub(rB, rT))
        dtS = shapiro_oneway(norm(rT), norm(rB), R, gamma=gamma)
        tT_new = tB - timedelta(seconds=(R/C + dtS))
        # 条件分岐: `abs((tT_new - tT).total_seconds()) < 1e-6` を満たす経路を評価する。
        if abs((tT_new - tT).total_seconds()) < 1e-6:
            break

        tT = tT_new

    return tT

# 関数: `main` の入出力契約と処理意図を定義する。

def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Cassini Shapiro light-time (iterative; HORIZONS required).")
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
    step = args.step  # use 1 minute grid

    earth_rows = parse_vectors_csv(fetch_horizons("399", start, stop, step, center="500@10"))
    cass_rows  = parse_vectors_csv(fetch_horizons("-82", start, stop, step, center="500@10"))

    earth = StateTable(earth_rows)
    cass  = StateTable(cass_rows)

    gamma = args.gamma if args.beta is None else (2.0 * args.beta - 1.0)

    # Choose receive times (use the same grid as input)
    times = earth.t[:]  # 1-minute grid

    dt2_list=[]
    b_list=[]

    for tR in times:
        # solve downlink bounce time
        tB = solve_downlink_time(tR, earth, cass, gamma=gamma)
        # solve uplink transmit time
        tT = solve_uplink_time(tB, earth, cass, gamma=gamma)

        rR = earth.state(tR)
        rB = cass.state(tB)
        rT = earth.state(tT)

        # downlink one-way
        R_down = norm(sub(rR, rB))
        dtS_down = shapiro_oneway(norm(rR), norm(rB), R_down, gamma=gamma)

        # uplink one-way
        R_up = norm(sub(rB, rT))
        dtS_up = shapiro_oneway(norm(rT), norm(rB), R_up, gamma=gamma)

        dt2 = dtS_up + dtS_down
        dt2_list.append(dt2)

        # define a "b" proxy for diagnostics: take min of uplink/downlink line b
        # line b = |r1 x r2| / |r2 - r1| using endpoints for each leg
        def b_line(a,b):
            u = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
            U = math.sqrt(u[0]**2+u[1]**2+u[2]**2)
            D = norm(sub(b,a))
            return U/D

        b_up = b_line(rT, rB)
        b_dn = b_line(rB, rR)
        b_list.append(min(b_up, b_dn))

    # y(t) ≈ d(dt2)/dtR using central diff (1-minute)

    y=[]
    for i in range(len(times)):
        # 条件分岐: `1 <= i <= len(times)-2` を満たす経路を評価する。
        if 1 <= i <= len(times)-2:
            y.append(-(dt2_list[i+1]-dt2_list[i-1])/(120.0))
        # 条件分岐: 前段条件が不成立で、`i==0` を追加評価する。
        elif i==0:
            y.append(-(dt2_list[1]-dt2_list[0])/(60.0))
        else:
            y.append(-(dt2_list[-1]-dt2_list[-2])/(60.0))

    # summarize

    idx_bmin = min(range(len(b_list)), key=lambda i: b_list[i])
    idx_ypeak = max(range(len(y)), key=lambda i: abs(y[i]))

    print("Samples:", len(times))
    print("b_min / R_sun:", b_list[idx_bmin]/R_SUN, "at", times[idx_bmin].isoformat())
    print("y_peak:", abs(y[idx_ypeak]), "at", times[idx_ypeak].isoformat())
    print("dt2 range (us):", min(dt2_list)*1e6, "to", max(dt2_list)*1e6)

    out = out_dir / "cassini_shapiro_lighttime.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["time_recv_utc","b_minleg_m","dt2_s","y_frac"])
        for tR, bb, dt2, yy in zip(times, b_list, dt2_list, y):
            w.writerow([tR.isoformat(), f"{bb:.6e}", f"{dt2:.12e}", f"{yy:.12e}"])

    print("Wrote:", out)

# 条件分岐: `__name__=="__main__"` を満たす経路を評価する。

if __name__=="__main__":
    main()
