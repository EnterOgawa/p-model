import argparse
import csv
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone
import ssl
from pathlib import Path

C = 299792458.0
MU_SUN = 1.3271244e20  # m^3/s^2
R_SUN = 6.957e8        # m

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

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
    try:
        with urllib.request.urlopen(url, timeout=180, context=SSL_CTX) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print("HORIZONS URL (for debugging):")
        print(url)
        raise

def parse_vectors_csv(txt: str):
    # 条件分岐: `"$$SOE" not in txt or "$$EOE" not in txt` を満たす経路を評価する。
    if "$$SOE" not in txt or "$$EOE" not in txt:
        raise RuntimeError("Missing $$SOE/$$EOE.\nFirst 1200 chars:\n" + txt[:1200])

    block = txt.split("$$SOE")[1].split("$$EOE")[0].strip()

    rows=[]
    for line in block.splitlines():
        line=line.strip()
        # 条件分岐: `not line` を満たす経路を評価する。
        if not line:
            continue

        parts=[p.strip() for p in line.split(",")]
        # Typical vectors CSV has at least: JD, Calendar, X, Y, Z, VX, VY, VZ
        if len(parts) < 8:
            continue

        cal = parts[1].replace("A.D.", "").strip()
        # Example: 2002-Jun-06 00:00:00.0000
        t = datetime.strptime(cal, "%Y-%b-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)

        x=float(parts[2]); y=float(parts[3]); z=float(parts[4])
        vx=float(parts[5]); vy=float(parts[6]); vz=float(parts[7])
        rows.append((t,x,y,z,vx,vy,vz))

    return rows

def norm3(x,y,z):
    return math.sqrt(x*x+y*y+z*z)

def impact_parameter_b(rE, rS):
    # b = |rE x rS| / |rS - rE|
    x1,y1,z1=rE; x2,y2,z2=rS
    cx=y1*z2 - z1*y2
    cy=z1*x2 - x1*z2
    cz=x1*y2 - y1*x2
    num=math.sqrt(cx*cx+cy*cy+cz*cz)
    dx=x2-x1; dy=y2-y1; dz=z2-z1
    den=math.sqrt(dx*dx+dy*dy+dz*dz)
    return num/den

def central_diff(vals, dt):
    n=len(vals)
    out=[0.0]*n
    for i in range(n):
        # 条件分岐: `1 <= i <= n-2` を満たす経路を評価する。
        if 1 <= i <= n-2:
            out[i]=(vals[i+1]-vals[i-1])/(2*dt)
        # 条件分岐: 前段条件が不成立で、`i==0` を追加評価する。
        elif i==0:
            out[i]=(vals[1]-vals[0])/dt
        else:
            out[i]=(vals[n-1]-vals[n-2])/dt

    return out

def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Cassini Shapiro waveform (Eq1/Eq2; HORIZONS required).")
    parser.add_argument("--start", default="2002-06-06 00:00")
    parser.add_argument("--stop", default="2002-07-07 00:00")
    parser.add_argument("--step", default="5 m")
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

    # Earth = 399
    earth_txt = fetch_horizons("399", start, stop, step, center="500@10")
    earth = parse_vectors_csv(earth_txt)

    # Cassini: try 82 then -82
    try:
        cass_txt = fetch_horizons("82", start, stop, step, center="500@10")
        cass = parse_vectors_csv(cass_txt)
    except Exception:
        cass_txt = fetch_horizons("-82", start, stop, step, center="500@10")
        cass = parse_vectors_csv(cass_txt)

    E = {t:(x,y,z) for (t,x,y,z,_,_,_) in earth}
    S = {t:(x,y,z) for (t,x,y,z,_,_,_) in cass}
    times = sorted(set(E.keys()) & set(S.keys()))
    # 条件分岐: `len(times) < 100` を満たす経路を評価する。
    if len(times) < 100:
        raise RuntimeError(f"Too few common epochs: {len(times)}")

    r1=[]; r2=[]; b=[]
    for t in times:
        xE,yE,zE = E[t]
        xS,yS,zS = S[t]
        rE=(xE*1000.0, yE*1000.0, zE*1000.0)
        rC=(xS*1000.0, yS*1000.0, zS*1000.0)
        r1.append(norm3(*rE))
        r2.append(norm3(*rC))
        b.append(impact_parameter_b(rE, rC))

    dt=300.0
    dbdt=central_diff(b, dt)

    gamma = args.gamma if args.beta is None else (2.0 * args.beta - 1.0)
    one_plus_gamma=1.0+gamma
    coef_t=2.0*one_plus_gamma*MU_SUN/(C**3)
    coef_y=4.0*one_plus_gamma*MU_SUN/(C**3)

    delta_t=[]; y=[]
    for i in range(len(times)):
        arg=(4.0*r1[i]*r2[i])/(b[i]**2)
        delta_t.append(coef_t*math.log(arg))
        y.append(coef_y*(dbdt[i]/b[i]))

    b_min=min(b)
    y_peak=max(abs(v) for v in y)

    print("Samples:", len(times))
    print("b_min / R_sun:", b_min / R_SUN)
    print("y_peak:", y_peak)

    out = out_dir / "cassini_shapiro_waveform.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["time_utc","r1_m","r2_m","b_m","dbdt_mps","delta_t_s","y_frac"])
        for t,a,b2,bb,db,dtv,yv in zip(times, r1, r2, b, dbdt, delta_t, y):
            w.writerow([t.isoformat(), f"{a:.6e}", f"{b2:.6e}", f"{bb:.6e}",
                        f"{db:.6e}", f"{dtv:.12e}", f"{yv:.12e}"])

    print("Wrote:", out)

# 条件分岐: `__name__=="__main__"` を満たす経路を評価する。

if __name__=="__main__":
    main()
