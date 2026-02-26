# -*- coding: utf-8 -*-
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import urllib.parse
import urllib.request
import ssl
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog

# --- 定数設定 ---
C = 299792458.0
MU_SUN = 1.3271244e20  # m^3/s^2 (太陽の重力定数)

def _make_ssl_context() -> ssl.SSLContext:
    """
    - HORIZONS_INSECURE=1 なら検証無効（最終手段）
    - HORIZONS_CA_BUNDLE / SSL_CERT_FILE / REQUESTS_CA_BUNDLE があればそれをcafileとして利用
    - それ以外はデフォルトのCAを利用
    """
    # 条件分岐: `os.environ.get("HORIZONS_INSECURE", "").strip() == "1"` を満たす経路を評価する。
    if os.environ.get("HORIZONS_INSECURE", "").strip() == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    cafile = (
        os.environ.get("HORIZONS_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
    )

    # 条件分岐: `cafile` を満たす経路を評価する。
    if cafile:
        return ssl.create_default_context(cafile=cafile)

    return ssl.create_default_context()


# SSL設定（Horizonsへの接続用）

SSL_CTX = _make_ssl_context()

def _cache_key(command: str, start: str, stop: str, step: str, center: str) -> str:
    s = f"cmd={command}|center={center}|start={start}|stop={stop}|step={step}|type=VECTORS|ref=ICRF|units=KM-S|csv=YES"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def fetch_horizons(command: str, start: str, stop: str, step: str, center="500@10") -> str:
    """NASA JPL Horizons APIからエフェメリス(Vector)を取得"""
    params = {
        "format": "text",
        "COMMAND": f"'{command}'",
        "CENTER": f"'{center}'", # 500@10 = Sun center
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
    print(f"Fetching {command} data from Horizons...")
    with urllib.request.urlopen(url, timeout=180, context=SSL_CTX) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def fetch_horizons_cached(
    command: str,
    start: str,
    stop: str,
    step: str,
    *,
    center: str,
    cache_dir: Path,
    offline: bool,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(command, start, stop, step, center)
    txt_path = cache_dir / f"horizons_vectors_{command}_{key}.txt"
    meta_path = cache_dir / f"horizons_vectors_{command}_{key}.json"

    # 条件分岐: `txt_path.exists()` を満たす経路を評価する。
    if txt_path.exists():
        print(f"[cache] Using {txt_path}")
        return txt_path.read_text(encoding="utf-8", errors="ignore")

    # 条件分岐: `offline` を満たす経路を評価する。

    if offline:
        raise RuntimeError(f"Offline mode: cache not found: {txt_path}")

    txt = fetch_horizons(command, start, stop, step, center=center)
    txt_path.write_text(txt, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "command": command,
                "center": center,
                "start": start,
                "stop": stop,
                "step": step,
                "saved_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[cache] Saved {txt_path}")
    return txt

def parse_vectors_csv(txt: str):
    """HorizonsのCSV出力をパースして (time, r, v) のリストを返す"""
    # 条件分岐: `"$$SOE" not in txt or "$$EOE" not in txt` を満たす経路を評価する。
    if "$$SOE" not in txt or "$$EOE" not in txt:
        raise RuntimeError(f"Horizons API Error: Missing $$SOE/$$EOE markers.")
    
    block = txt.split("$$SOE")[1].split("$$EOE")[0].strip()
    rows=[]
    for line in block.splitlines():
        parts=[p.strip() for p in line.strip().split(",")]
        # 条件分岐: `len(parts) < 8` を満たす経路を評価する。
        if len(parts) < 8:
            continue
        # 日付パース (例: A.D. 1976-Nov-25 00:00:00.0000)

        cal = parts[1].replace("A.D.", "").strip()
        t = datetime.strptime(cal, "%Y-%b-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        
        # km -> m 変換
        x=float(parts[2])*1000.0; y=float(parts[3])*1000.0; z=float(parts[4])*1000.0
        vx=float(parts[5])*1000.0; vy=float(parts[6])*1000.0; vz=float(parts[7])*1000.0
        rows.append((t, (x,y,z), (vx,vy,vz)))

    return rows

# --- ベクトル演算関数 ---

def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def norm(a): return math.sqrt(dot(a,a))

# --- Shapiro遅延計算 (片道) ---
def shapiro_oneway(r1, r2, R, gamma=1.0):
    """
    (1+gamma) * (GM/c^3) * ln( (r1 + r2 + R) / (r1 + r2 - R) )
    """
    val = (r1 + r2 + R) / (r1 + r2 - R)
    return (1.0+gamma)*MU_SUN/(C**3) * math.log(val)

def main():
    out_dir = _ROOT / "output" / "private" / "viking"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Viking Shapiro delay check (HORIZONS required).")
    parser.add_argument("--start", default="1976-11-01")
    parser.add_argument("--stop", default="1976-12-30")
    parser.add_argument("--step", default="4 h")
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="P-model beta (sets gamma = 2*beta - 1). If omitted, --gamma is used.",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="PPN gamma (default: 1.0)")
    parser.add_argument("--offline", action="store_true", help="Use cached Horizons data only (no network).")
    args = parser.parse_args()

    start_str = args.start
    stop_str = args.stop
    step_str = args.step

    print(f"--- Viking Shapiro Delay Check ({start_str} to {stop_str}) ---")

    # 399: Earth, 499: Mars
    offline = args.offline or os.environ.get("HORIZONS_OFFLINE", "").strip() == "1"
    cache_dir = out_dir / "horizons_cache"

    earth_txt = fetch_horizons_cached(
        "399",
        start_str,
        stop_str,
        step_str,
        center="500@10",
        cache_dir=cache_dir,
        offline=offline,
    )
    mars_txt = fetch_horizons_cached(
        "499",
        start_str,
        stop_str,
        step_str,
        center="500@10",
        cache_dir=cache_dir,
        offline=offline,
    )

    earth_data = parse_vectors_csv(earth_txt)
    mars_data = parse_vectors_csv(mars_txt)

    # 辞書化して時刻マッチング
    E = {r[0]: r[1] for r in earth_data} # (time: position_tuple)
    M = {r[0]: r[1] for r in mars_data}
    
    times = sorted(set(E.keys()) & set(M.keys()))
    print(f"Calculating {len(times)} points...")

    results = []
    max_delay_us = 0.0
    max_delay_time = None
    gamma = args.gamma if args.beta is None else (2.0 * args.beta - 1.0)

    for t in times:
        rE = E[t] # 地球位置
        rM = M[t] # 火星位置
        
        # 簡易計算: 光の往復時間は無視して同時刻の位置で計算（誤差は微小）
        # 正確には反復計算が必要ですが、Shapiro遅延のオーダー確認にはこれで十分です
        
        r1 = norm(rE)           # 太陽-地球
        r2 = norm(rM)           # 太陽-火星
        R  = norm(sub(rE, rM))  # 地球-火星距離
        
        # 往復分のShapiro遅延 (片道 x 2)
        # バイキングはランダーからの応答なので往復で見ます
        dt_shapiro_oneway = shapiro_oneway(r1, r2, R, gamma)
        total_delay_us = (dt_shapiro_oneway * 2.0) * 1e6 # マイクロ秒換算
        
        results.append((t, total_delay_us))
        
        # 条件分岐: `total_delay_us > max_delay_us` を満たす経路を評価する。
        if total_delay_us > max_delay_us:
            max_delay_us = total_delay_us
            max_delay_time = t

    # 結果表示

    print("-" * 40)
    print(f"Peak Shapiro Delay (Round Trip): {max_delay_us:.2f} microseconds")
    print(f"At Time (UTC)                  : {max_delay_time}")
    print("-" * 40)
    print("Expected value: approx 200 - 250 microseconds")

    # CSV保存
    out_file = out_dir / "viking_shapiro_result.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_utc", "shapiro_delay_us"])
        for r in results:
            w.writerow([r[0].isoformat(), f"{r[1]:.4f}"])

    print(f"Saved: {out_file}")

    try:
        worklog.append_event(
            {
                "event_type": "viking_shapiro",
                "argv": sys.argv,
                "metrics": {
                    "peak_delay_us_round_trip": float(max_delay_us),
                    "peak_time_utc": max_delay_time.isoformat() if max_delay_time else None,
                    "gamma": float(gamma),
                    "beta": float(args.beta) if args.beta is not None else None,
                    "n_points": int(len(results)),
                },
                "outputs": {
                    "result_csv": out_file,
                    "horizons_cache_dir": cache_dir,
                },
            }
        )
    except Exception:
        pass

# 条件分岐: `__name__=="__main__"` を満たす経路を評価する。

if __name__=="__main__":
    main()
