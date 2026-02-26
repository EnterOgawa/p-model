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

def parse_vectors_csv(txt: str):
    # 条件分岐: `"$$SOE" not in txt or "$$EOE" not in txt` を満たす経路を評価する。
    if "$$SOE" not in txt or "$$EOE" not in txt:
        raise RuntimeError("Missing $$SOE/$$EOE")

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

def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def norm(a): return math.sqrt(dot(a,a))
def sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def add(a,b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

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

def shapiro_dt(r1, r2, b, gamma=1.0):
    # Cassini Eq(1): round-trip delay (b approximation)
    return 2.0*(1.0+gamma)*MU_SUN/(C**3) * math.log((4.0*r1*r2)/(b*b))

def y_eq2(b, bdot, gamma=1.0):
    # Cassini Eq(2) approximation
    return 4.0*(1.0+gamma)*MU_SUN/(C**3) * (bdot/b)

def y_full(r1, r1dot, r2, r2dot, b, bdot, gamma=1.0):
    # Cassini Doppler observable (round-trip) uses: y = - d(Delta_t)/dt
    coef = 2.0*(1.0+gamma)*MU_SUN/(C**3)
    return -coef * ((r1dot/r1) + (r2dot/r2) - 2.0*(bdot/b))

def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "private" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Cassini Shapiro y(t) (cached geometry via HORIZONS).")
    parser.add_argument("--start", default="2002-06-06 00:00")
    parser.add_argument("--stop", default="2002-07-07 00:00")
    parser.add_argument("--step", default="1 m")
    parser.add_argument(
        "--in-csv",
        default=None,
        help="Optional input CSV with geometry columns to recompute dt/y without HORIZONS.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="P-model beta (sets gamma = 2*beta - 1). If omitted, --gamma is used.",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="PPN gamma (default: 1.0)")
    parser.add_argument(
        "--out",
        default="output/private/cassini/cassini_shapiro_y_full.csv",
        help="Output CSV path (default: output/private/cassini/cassini_shapiro_y_full.csv)",
    )
    args = parser.parse_args()

    start = args.start
    stop = args.stop
    step = args.step
    gamma = args.gamma if args.beta is None else (2.0 * args.beta - 1.0)
    out = Path(args.out)
    # 条件分岐: `not out.is_absolute()` を満たす経路を評価する。
    if not out.is_absolute():
        out = (root / out).resolve()

    # 条件分岐: `args.in_csv` を満たす経路を評価する。

    if args.in_csv:
        in_path = Path(args.in_csv)
        # 条件分岐: `not in_path.is_absolute()` を満たす経路を評価する。
        if not in_path.is_absolute():
            in_path = (root / in_path).resolve()

        out_rows = []
        b_list = []
        y2_list = []
        yf_list = []
        dt_list = []

        with in_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = datetime.fromisoformat(row["time_utc"])
                # 条件分岐: `t.tzinfo is None` を満たす経路を評価する。
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                else:
                    t = t.astimezone(timezone.utc)

                r1 = float(row["r1_m"])
                r2 = float(row["r2_m"])
                b = float(row["b_m"])
                bdot = float(row["bdot_mps"])
                r1dot = float(row["r1dot_mps"])
                r2dot = float(row["r2dot_mps"])

                dtv = shapiro_dt(r1, r2, b, gamma=gamma)
                y2 = y_eq2(b, bdot, gamma=gamma)
                yf = y_full(r1, r1dot, r2, r2dot, b, bdot, gamma=gamma)

                out_rows.append((t, r1, r2, b, bdot, r1dot, r2dot, dtv, y2, yf))
                b_list.append(b)
                y2_list.append(y2)
                yf_list.append(yf)
                dt_list.append(dtv)

        # 条件分岐: `not out_rows` を満たす経路を評価する。

        if not out_rows:
            raise RuntimeError(f"No rows read from --in-csv: {in_path}")

        idx_bmin = min(range(len(b_list)), key=lambda i: b_list[i])
        idx_y2 = max(range(len(y2_list)), key=lambda i: abs(y2_list[i]))
        idx_yf = max(range(len(yf_list)), key=lambda i: abs(yf_list[i]))

        print("Samples:", len(out_rows))
        print("b_min / R_sun:", b_list[idx_bmin] / R_SUN, "at", out_rows[idx_bmin][0].isoformat())
        print("y_peak Eq(2):", abs(y2_list[idx_y2]), "at", out_rows[idx_y2][0].isoformat())
        print("y_peak FULL :", abs(yf_list[idx_yf]), "at", out_rows[idx_yf][0].isoformat())
        print("Delta_t range (us):", min(dt_list) * 1e6, "to", max(dt_list) * 1e6)

        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "time_utc",
                    "r1_m",
                    "r2_m",
                    "b_m",
                    "bdot_mps",
                    "r1dot_mps",
                    "r2dot_mps",
                    "delta_t_s",
                    "y_eq2",
                    "y_full",
                ]
            )
            for (t, r1, r2, b, bd, rd1, rd2, dtv, y2, yf) in out_rows:
                w.writerow(
                    [
                        t.isoformat(),
                        f"{r1:.6e}",
                        f"{r2:.6e}",
                        f"{b:.6e}",
                        f"{bd:.6e}",
                        f"{rd1:.6e}",
                        f"{rd2:.6e}",
                        f"{dtv:.12e}",
                        f"{y2:.12e}",
                        f"{yf:.12e}",
                    ]
                )

        print("Wrote:", out)
        return

    E = {}
    S = {}
    times = []

    # 条件分岐: `not args.in_csv` を満たす経路を評価する。
    if not args.in_csv:
        earth = parse_vectors_csv(fetch_horizons("399", start, stop, step, center="500@10"))
        cass  = parse_vectors_csv(fetch_horizons("-82", start, stop, step, center="500@10"))

        E = {t:(r,v) for (t,r,v) in earth}
        S = {t:(r,v) for (t,r,v) in cass}
        times = sorted(set(E.keys()) & set(S.keys()))
        # 条件分岐: `len(times) < 1000` を満たす経路を評価する。
        if len(times) < 1000:
            raise RuntimeError("Too few common epochs")

    out_rows=[]
    b_list=[]
    y2_list=[]
    yf_list=[]
    dt_list=[]

    for t in times:
        rE,vE = E[t]
        rS,vS = S[t]

        r1 = norm(rE)
        r2 = norm(rS)

        impact = impact_b_and_bdot(rE, vE, rS, vS)
        # 条件分岐: `impact is None` を満たす経路を評価する。
        if impact is None:
            continue

        b, bdot = impact

        # rdot = (r·v)/|r|
        r1dot = dot(rE, vE)/r1
        r2dot = dot(rS, vS)/r2

        dtv = shapiro_dt(r1, r2, b, gamma=gamma)
        y2  = y_eq2(b, bdot, gamma=gamma)
        yf  = y_full(r1, r1dot, r2, r2dot, b, bdot, gamma=gamma)

        out_rows.append((t, r1, r2, b, bdot, r1dot, r2dot, dtv, y2, yf))
        b_list.append(b); y2_list.append(y2); yf_list.append(yf); dt_list.append(dtv)

    # summaries

    idx_bmin = min(range(len(b_list)), key=lambda i: b_list[i])
    idx_y2 = max(range(len(y2_list)), key=lambda i: abs(y2_list[i]))
    idx_yf = max(range(len(yf_list)), key=lambda i: abs(yf_list[i]))

    print("Samples:", len(out_rows))
    print("b_min / R_sun:", b_list[idx_bmin]/R_SUN, "at", out_rows[idx_bmin][0].isoformat())
    print("y_peak Eq(2):", abs(y2_list[idx_y2]), "at", out_rows[idx_y2][0].isoformat())
    print("y_peak FULL :", abs(yf_list[idx_yf]), "at", out_rows[idx_yf][0].isoformat())
    print("Delta_t range (us):", min(dt_list)*1e6, "to", max(dt_list)*1e6)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["time_utc","r1_m","r2_m","b_m","bdot_mps","r1dot_mps","r2dot_mps",
                    "delta_t_s","y_eq2","y_full"])
        for (t,r1,r2,b,bd,rd1,rd2,dtv,y2,yf) in out_rows:
            w.writerow([t.isoformat(), f"{r1:.6e}", f"{r2:.6e}", f"{b:.6e}", f"{bd:.6e}",
                        f"{rd1:.6e}", f"{rd2:.6e}", f"{dtv:.12e}", f"{y2:.12e}", f"{yf:.12e}"])

    print("Wrote:", out)

# 条件分岐: `__name__=="__main__"` を満たす経路を評価する。

if __name__=="__main__":
    main()
