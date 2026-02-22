#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import http.cookiejar
import json
import netrc
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STATIONS_DIR = ROOT / "data" / "llr" / "stations"
APOL_META_PATH = STATIONS_DIR / "apol.json"
LLR_EDC_MANIFEST_PATH = ROOT / "data" / "llr" / "llr_edc_batch_manifest.json"
EDC_SLRLOG_LIST = "https://edc.dgfi.tum.de/pub/slr/slrlog/"
EDC_SLRLOG_OLDLOG_LIST = "https://edc.dgfi.tum.de/pub/slr/slrlog/oldlog/"
EDC_SLRHST_LIST = "https://edc.dgfi.tum.de/pub/slr/slrhst/"
EDC_SLRMAIL_INDEX = "https://edc.dgfi.tum.de/pub/slr/slrmail/slrmail.index"
EDC_BASE = "https://edc.dgfi.tum.de"
CDDIS_SLRLOG_LIST = "https://cddis.nasa.gov/archive/slr/slrlog/"
CDDIS_SLRLOG_OLDLOG_LIST = "https://cddis.nasa.gov/archive/slr/slrlog/oldlog/"
CDDIS_SLRHST_LIST = "https://cddis.nasa.gov/archive/slr/slrhst/"
CDDIS_SLRECC_LIST = "https://cddis.nasa.gov/archive/slr/slrecc/"
CDDIS_LLR_NPT_CRD_ROOT = "https://cddis.nasa.gov/archive/slr/data/npt_crd/"
CDDIS_LLR_FR_CRD_ROOT = "https://cddis.nasa.gov/archive/slr/data/fr_crd/"
CDDIS_LLR_TARGETS = ("apollo", "apollo11", "apollo14", "apollo15")
CDDIS_PRODUCTS_POS_EOP_ROOT = "https://cddis.nasa.gov/archive/slr/products/pos+eop/"
CDDIS_PRODUCTS_RESOURCE_ROOT = "https://cddis.nasa.gov/archive/slr/products/resource/"
CDDIS_PRODUCTS_AC_ROOT = "https://cddis.nasa.gov/archive/slr/products/ac/"
CDDIS_PRODUCTS_AC_ARCHIVE_ROOT = "https://cddis.nasa.gov/archive/slr/products/ac/archive/"
CDDIS_ILRSAC_PRODUCTS_OPS_POS_EOP_ROOT = "https://cddis.nasa.gov/archive/slr/products/ilrsac/products/ops/pos+eop/"
CDDIS_ILRSAC_PRODUCTS_OPS_ROOT = "https://cddis.nasa.gov/archive/slr/products/ilrsac/products/ops/"
CDDIS_SLROCC_ROOT = "https://cddis.nasa.gov/archive/slr/slrocc/"
CDDIS_SLROCC_SLRCOOR_TXT = f"{CDDIS_SLROCC_ROOT}slrcoor.txt"
CDDIS_SLROCC_SLRCOOR_SPL = f"{CDDIS_SLROCC_ROOT}slrcoor.spl"
CDDIS_SLROCC_OCC_TXT = f"{CDDIS_SLROCC_ROOT}slrocc.txt"
CDDIS_SLROCC_SLRCAL_TXT = f"{CDDIS_SLROCC_ROOT}slrcal.txt"
CDDIS_SLROCC_SLRCAL_SPL = f"{CDDIS_SLROCC_ROOT}slrcal.spl"
REPRO_ROOT = "https://edc.dgfi.tum.de/pub/slr/products/REPRO_SERIES/REPRO2020/ilrs2020_individual_ac"
REPRO_CENTERS = ("ilrsa", "nsgf", "bkg", "dgfi", "esa", "gfz", "jcet", "asi")
EARTHDATA_HOST = "urs.earthdata.nasa.gov"
EARTHDATA_USER_ENV_KEYS = ("EARTHDATA_USERNAME", "EARTHDATA_USER", "EDL_USERNAME", "EDL_USER")
EARTHDATA_PASS_ENV_KEYS = ("EARTHDATA_PASSWORD", "EARTHDATA_PASS", "EDL_PASSWORD", "EDL_PASS")
ILRS_PRIMARY_PRODUCT_FALLBACK_URLS = (
    "https://ilrs.gsfc.nasa.gov/docs/2025/ILRS_Data_Handling_File_20250513.snx",
    "https://ilrs.gsfc.nasa.gov/docs/2025/SLRF2020_POS+VEL_2025.05.13.snx",
    "https://cddis.nasa.gov/archive/slr/slrecc/slrecc.250513.ILRS.xyz.snx.gz",
    "https://cddis.nasa.gov/archive/slr/slrecc/slrecc.250513.ILRS.une.snx.gz",
)
OUT_DIR = ROOT / "output" / "private" / "llr"
OUT_JSON = OUT_DIR / "llr_apol_primary_coord_route_audit.json"
OUT_CSV = OUT_DIR / "llr_apol_primary_coord_route_audit.csv"
OUT_PNG = OUT_DIR / "llr_apol_primary_coord_route_audit.png"
OUT_MERGE_JSON = OUT_DIR / "llr_apol_primary_coord_merge_route.json"
_EARTHDATA_AUTH_CONTEXT: Dict[str, Any] = {"enabled": False}


def _first_non_empty_env(keys: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
    for key in keys:
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value, key
    return None, None


def _mask_user(user: Optional[str]) -> Optional[str]:
    if user is None:
        return None
    s = str(user).strip()
    if not s:
        return None
    if len(s) <= 2:
        return "*" * len(s)
    return f"{s[:2]}***"


def _read_earthdata_netrc(netrc_path: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    candidates: List[Path] = []
    if netrc_path:
        candidates.append(Path(netrc_path).expanduser())
    else:
        candidates.append(Path.home() / ".netrc")
        candidates.append(Path.home() / "_netrc")

    for p in candidates:
        if not p.exists():
            continue
        try:
            auth = netrc.netrc(str(p)).authenticators(EARTHDATA_HOST)
        except Exception as e:
            return None, None, None, f"netrc_parse_error({p}): {e}"
        if not auth:
            continue
        login, _account, password = auth
        user = str(login or "").strip() if login is not None else ""
        pw = str(password or "").strip() if password is not None else ""
        if user and pw:
            return user, pw, str(p), None
    return None, None, None, None


def _configure_earthdata_auth(
    *,
    mode: str,
    username: Optional[str],
    password: Optional[str],
    netrc_path: Optional[str],
) -> Dict[str, Any]:
    global _EARTHDATA_AUTH_CONTEXT
    diag: Dict[str, Any] = {
        "mode": str(mode),
        "status": "disabled",
        "credential_source": None,
        "username_hint": None,
        "error": None,
        "netrc_path": str(Path(netrc_path).expanduser()) if netrc_path else None,
    }
    mode_l = str(mode or "auto").strip().lower()
    if mode_l == "off":
        diag["status"] = "disabled"
        diag["error"] = "earthdata_auth_disabled_by_option"
        _EARTHDATA_AUTH_CONTEXT = {"enabled": False, "reason": "auth_mode_off"}
        return diag

    user = str(username or "").strip()
    pw = str(password or "").strip()
    source = None
    chosen_netrc_path: Optional[str] = str(Path(netrc_path).expanduser()) if netrc_path else None
    if user and pw:
        source = "cli"
    else:
        env_user, env_user_key = _first_non_empty_env(EARTHDATA_USER_ENV_KEYS)
        env_pass, env_pass_key = _first_non_empty_env(EARTHDATA_PASS_ENV_KEYS)
        if env_user and env_pass:
            user = env_user
            pw = env_pass
            source = f"env:{env_user_key}+{env_pass_key}"

    if not source:
        n_user, n_pw, n_path, n_err = _read_earthdata_netrc(netrc_path=netrc_path)
        if n_err:
            diag["error"] = n_err
        if n_user and n_pw:
            user = n_user
            pw = n_pw
            chosen_netrc_path = n_path
            source = f"netrc:{n_path}"

    if not source:
        diag["status"] = "error" if mode_l == "force" else "disabled"
        if not diag.get("error"):
            diag["error"] = "earthdata_credentials_not_found"
        _EARTHDATA_AUTH_CONTEXT = {"enabled": False, "reason": str(diag.get("error") or "credentials_not_found")}
        return diag

    try:
        cookie_jar = http.cookiejar.CookieJar()
        pw_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        pw_mgr.add_password(None, f"https://{EARTHDATA_HOST}", user, pw)
        pw_mgr.add_password(None, EARTHDATA_HOST, user, pw)
        opener = urllib.request.build_opener(
            urllib.request.HTTPBasicAuthHandler(pw_mgr),
            urllib.request.HTTPCookieProcessor(cookie_jar),
        )
        opener.addheaders = [("User-Agent", "waveP-llr-apol-route-audit/1.0")]
        urllib.request.install_opener(opener)
    except Exception as e:
        diag["status"] = "error"
        diag["credential_source"] = source
        diag["username_hint"] = _mask_user(user)
        diag["error"] = f"earthdata_opener_install_error: {e}"
        return diag

    diag["status"] = "enabled"
    diag["credential_source"] = source
    diag["username_hint"] = _mask_user(user)
    _EARTHDATA_AUTH_CONTEXT = {
        "enabled": True,
        "user": user,
        "password": pw,
        "credential_source": source,
        "netrc_path": chosen_netrc_path,
        "curl_path": None,
        "curl_cookie_path": None,
        "curl_netrc_path": None,
        "curl_fetch_used": False,
        "curl_last_error": None,
    }
    return diag


def _fetch_text(url: str, timeout_s: int) -> str:
    data, _meta = _fetch_bytes(url, timeout_s=timeout_s)
    return data.decode("utf-8", "replace")


def _fetch_text_direct(url: str, timeout_s: int) -> str:
    opener = urllib.request.build_opener()
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-llr-apol-route-audit/1.0"})
    with opener.open(req, timeout=timeout_s) as response:
        data = response.read()
    return data.decode("utf-8", "replace")


def _looks_like_earthdata_login_html(text: str) -> bool:
    text_head = str(text or "")[:4000]
    return "Earthdata Login" in text_head or "urs.earthdata.nasa.gov" in text_head


def _is_cddis_slrecc_url(url: str) -> bool:
    url_l = str(url or "").lower()
    return "cddis.nasa.gov/archive/slr/slrecc/" in url_l and "slrecc" in url_l


def _is_cddis_archive_url(url: str) -> bool:
    url_l = str(url or "").lower()
    return "cddis.nasa.gov/archive/" in url_l


def _prepare_earthdata_curl_context() -> Dict[str, Any]:
    ctx = _EARTHDATA_AUTH_CONTEXT
    if not bool(ctx.get("enabled")):
        raise RuntimeError("earthdata_auth_not_enabled")

    curl_path = str(ctx.get("curl_path") or "").strip()
    if not curl_path:
        curl_path = str(shutil.which("curl.exe") or shutil.which("curl") or "").strip()
        if not curl_path:
            raise RuntimeError("curl_not_found")
        ctx["curl_path"] = curl_path

    netrc_path = str(ctx.get("curl_netrc_path") or "").strip()
    if not netrc_path:
        configured_netrc = str(ctx.get("netrc_path") or "").strip()
        if configured_netrc and Path(configured_netrc).exists():
            netrc_path = configured_netrc
        else:
            user = str(ctx.get("user") or "").strip()
            pw = str(ctx.get("password") or "").strip()
            if not user or not pw:
                raise RuntimeError("earthdata_credentials_missing_for_curl")
            tmp_netrc = Path(tempfile.gettempdir()) / f"wavep_earthdata_{os.getpid()}.netrc"
            tmp_netrc.write_text(
                f"machine {EARTHDATA_HOST}\nlogin {user}\npassword {pw}\n",
                encoding="utf-8",
            )
            netrc_path = str(tmp_netrc)
        ctx["curl_netrc_path"] = netrc_path

    cookie_path = str(ctx.get("curl_cookie_path") or "").strip()
    if not cookie_path:
        cookie_path = str(Path(tempfile.gettempdir()) / f"wavep_earthdata_{os.getpid()}.cookies")
        ctx["curl_cookie_path"] = cookie_path

    return ctx


def _fetch_bytes_with_earthdata_curl(url: str, timeout_s: int) -> Tuple[bytes, Dict[str, Any]]:
    ctx = _prepare_earthdata_curl_context()
    cmd = [
        str(ctx.get("curl_path")),
        "-sS",
        "-L",
        "--max-time",
        str(max(1, int(timeout_s))),
        "-n",
        "--netrc-file",
        str(ctx.get("curl_netrc_path")),
        "-c",
        str(ctx.get("curl_cookie_path")),
        "-b",
        str(ctx.get("curl_cookie_path")),
        str(url),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace") if proc.stderr else ""
        raise RuntimeError(f"curl_exit_{proc.returncode}: {stderr[:240]}")
    data = bytes(proc.stdout or b"")
    if not data:
        raise RuntimeError("curl_empty_body")
    ctx["curl_fetch_used"] = True
    return data, {
        "content_type": "",
        "content_encoding": "",
        "fetch_method": "curl_netrc_cookie",
    }


def _fetch_bytes(url: str, timeout_s: int) -> Tuple[bytes, Dict[str, Any]]:
    if _is_cddis_archive_url(url) and bool(_EARTHDATA_AUTH_CONTEXT.get("enabled")):
        try:
            return _fetch_bytes_with_earthdata_curl(url=url, timeout_s=timeout_s)
        except Exception as e:
            _EARTHDATA_AUTH_CONTEXT["curl_last_error"] = str(e)
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-llr-apol-route-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = r.read()
        meta = {
            "content_type": str(r.headers.get("Content-Type") or ""),
            "content_encoding": str(r.headers.get("Content-Encoding") or ""),
        }
        return data, meta


def _iter_hrefs(html: str) -> List[str]:
    return re.findall(r'href=[\'"]([^\'"]+)[\'"]', html)


def _yymmdd_to_yyyymmdd(yymmdd: str) -> Optional[str]:
    s = str(yymmdd or "").strip()
    if not re.fullmatch(r"\d{6}", s):
        return None
    yy = int(s[:2])
    yyyy = 2000 + yy if yy < 80 else 1900 + yy
    return f"{yyyy:04d}{s[2:]}"


def _date_age_days_from_yyyymmdd(yyyymmdd: Optional[str]) -> Optional[int]:
    s = str(yyyymmdd or "").strip()
    if not re.fullmatch(r"\d{8}", s):
        return None
    try:
        dt = datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    delta = datetime.now(timezone.utc) - dt
    return int(delta.days)


def _extract_float(text: str) -> Optional[float]:
    if text is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", str(text))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _extract_signed_angle(text: str, pos_letter: str, neg_letter: str) -> Optional[float]:
    if text is None:
        return None
    value = _extract_float(text)
    if value is None:
        return None
    t = str(text).upper()
    if neg_letter.upper() in t:
        return -abs(value)
    if pos_letter.upper() in t:
        return abs(value)
    return value


def _parse_apol_site_log(text: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "date_prepared": None,
        "x_m": None,
        "y_m": None,
        "z_m": None,
        "lat_deg": None,
        "lon_deg": None,
        "height_m": None,
    }

    for line in text.splitlines():
        line_s = line.strip()
        if not line_s:
            continue

        if "Date Prepared" in line_s and rec.get("date_prepared") is None:
            rec["date_prepared"] = line_s.split(":", 1)[1].strip() if ":" in line_s else None
            continue
        if "X coordinate" in line_s:
            rec["x_m"] = _extract_float(line_s.split(":", 1)[1] if ":" in line_s else line_s)
            continue
        if "Y coordinate" in line_s:
            rec["y_m"] = _extract_float(line_s.split(":", 1)[1] if ":" in line_s else line_s)
            continue
        if "Z coordinate" in line_s:
            rec["z_m"] = _extract_float(line_s.split(":", 1)[1] if ":" in line_s else line_s)
            continue
        if "Latitude" in line_s and "[deg]" in line_s:
            rec["lat_deg"] = _extract_signed_angle(line_s.split(":", 1)[1] if ":" in line_s else line_s, "N", "S")
            continue
        if "Longitude" in line_s and "[deg]" in line_s:
            rec["lon_deg"] = _extract_signed_angle(line_s.split(":", 1)[1] if ":" in line_s else line_s, "E", "W")
            continue
        if "Elevation" in line_s and "[m]" in line_s:
            rec["height_m"] = _extract_float(line_s.split(":", 1)[1] if ":" in line_s else line_s)
            continue

    rec["has_xyz"] = bool(
        rec.get("x_m") is not None and rec.get("y_m") is not None and rec.get("z_m") is not None
    )
    rec["has_geodetic"] = bool(
        rec.get("lat_deg") is not None and rec.get("lon_deg") is not None and rec.get("height_m") is not None
    )
    return rec


def _scan_local_logs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(STATIONS_DIR.glob("apol_*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        parsed = _parse_apol_site_log(text)
        m = re.search(r"apol_(\d{8})\.log$", path.name, flags=re.IGNORECASE)
        log_date = m.group(1) if m else None
        rows.append(
            {
                "source": "local_cache",
                "source_group": "local_cache",
                "log_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "log_date": log_date,
                **parsed,
            }
        )
    return rows


def _list_remote_apol_logs(timeout_s: int) -> List[Tuple[str, str]]:
    html = _fetch_text(EDC_SLRLOG_LIST, timeout_s=timeout_s)
    rows: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for href in _iter_hrefs(html):
        m = re.search(r"/pub/slr/slrlog/apol_(\d{8})\.log$", href, flags=re.IGNORECASE)
        if not m:
            continue
        ymd = m.group(1)
        if ymd in seen:
            continue
        seen.add(ymd)
        url = f"{EDC_BASE}{href}" if href.startswith("/") else href
        rows.append((ymd, url))
    rows.sort(reverse=True)
    return rows


def _scan_remote_logs(max_logs: int, timeout_s: int) -> Tuple[List[Dict[str, Any]], Optional[str], int]:
    remote_rows: List[Dict[str, Any]] = []
    try:
        remote_list = _list_remote_apol_logs(timeout_s=timeout_s)
    except Exception as e:
        return remote_rows, str(e), 0

    n_total = len(remote_list)
    for ymd, url in remote_list[: max(0, int(max_logs))]:
        try:
            text = _fetch_text(url, timeout_s=timeout_s)
            parsed = _parse_apol_site_log(text)
            remote_rows.append(
                {
                    "source": "edc_remote",
                    "source_group": "edc_remote",
                    "log_file": url,
                    "log_date": ymd,
                    **parsed,
                }
            )
        except Exception as e:
            remote_rows.append(
                {
                    "source": "edc_remote",
                    "source_group": "edc_remote",
                    "log_file": url,
                    "log_date": ymd,
                    "date_prepared": None,
                    "x_m": None,
                    "y_m": None,
                    "z_m": None,
                    "lat_deg": None,
                    "lon_deg": None,
                    "height_m": None,
                    "has_xyz": False,
                    "has_geodetic": False,
                    "fetch_error": str(e),
                }
            )
    return remote_rows, None, n_total


def _list_remote_apol_oldlog_logs(timeout_s: int) -> List[Tuple[str, str]]:
    html = _fetch_text(EDC_SLRLOG_OLDLOG_LIST, timeout_s=timeout_s)
    rows: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for href in _iter_hrefs(html):
        m = re.search(r"/pub/slr/slrlog/oldlog/apol_(\d{8})\.log$", href, flags=re.IGNORECASE)
        if not m:
            continue
        ymd = m.group(1)
        if ymd in seen:
            continue
        seen.add(ymd)
        url = f"{EDC_BASE}{href}" if href.startswith("/") else href
        rows.append((ymd, url))
    rows.sort(reverse=True)
    return rows


def _scan_remote_oldlog_logs(max_logs: int, timeout_s: int) -> Tuple[List[Dict[str, Any]], Optional[str], int]:
    remote_rows: List[Dict[str, Any]] = []
    try:
        remote_list = _list_remote_apol_oldlog_logs(timeout_s=timeout_s)
    except Exception as e:
        return remote_rows, str(e), 0

    n_total = len(remote_list)
    for ymd, url in remote_list[: max(0, int(max_logs))]:
        try:
            text = _fetch_text(url, timeout_s=timeout_s)
            parsed = _parse_apol_site_log(text)
            remote_rows.append(
                {
                    "source": "slrlog_oldlog_remote",
                    "source_group": "slrlog_oldlog",
                    "log_file": url,
                    "log_date": ymd,
                    **parsed,
                }
            )
        except Exception as e:
            remote_rows.append(
                {
                    "source": "slrlog_oldlog_remote",
                    "source_group": "slrlog_oldlog",
                    "log_file": url,
                    "log_date": ymd,
                    "date_prepared": None,
                    "x_m": None,
                    "y_m": None,
                    "z_m": None,
                    "lat_deg": None,
                    "lon_deg": None,
                    "height_m": None,
                    "has_xyz": False,
                    "has_geodetic": False,
                    "fetch_error": str(e),
                }
            )
    return remote_rows, None, n_total


def _list_remote_apol_hst_logs(timeout_s: int) -> List[Tuple[str, str]]:
    html = _fetch_text(EDC_SLRHST_LIST, timeout_s=timeout_s)
    rows: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for href in _iter_hrefs(html):
        m = re.search(r"/pub/slr/slrhst/apol_hst_(\d{8})\.log$", href, flags=re.IGNORECASE)
        if not m:
            continue
        ymd = m.group(1)
        if ymd in seen:
            continue
        seen.add(ymd)
        url = f"{EDC_BASE}{href}" if href.startswith("/") else href
        rows.append((ymd, url))
    rows.sort(reverse=True)
    return rows


def _parse_apol_hst_log(text: str) -> Dict[str, Any]:
    t = str(text or "")
    has_xyz = bool(re.search(r"\bX coordinate\b|\bY coordinate\b|\bZ coordinate\b", t, flags=re.IGNORECASE))
    has_geodetic = bool(
        re.search(r"\bLatitude\b", t, flags=re.IGNORECASE)
        and re.search(r"\bLongitude\b", t, flags=re.IGNORECASE)
        and re.search(r"\bElevation\b|\bHeight\b", t, flags=re.IGNORECASE)
    )
    return {"has_xyz": has_xyz, "has_geodetic": has_geodetic}


def _scan_slrhst_logs(max_logs: int, timeout_s: int) -> Tuple[List[Dict[str, Any]], Optional[str], int]:
    rows: List[Dict[str, Any]] = []
    try:
        candidates = _list_remote_apol_hst_logs(timeout_s=timeout_s)
    except Exception as e:
        return rows, str(e), 0

    n_total = len(candidates)
    for ymd, url in candidates[: max(0, int(max_logs))]:
        try:
            text = _fetch_text(url, timeout_s=timeout_s)
            parsed = _parse_apol_hst_log(text)
            rows.append(
                {
                    "source": "slrhst_remote",
                    "source_group": "slrhst_history",
                    "log_file": url,
                    "log_date": ymd,
                    "date_prepared": None,
                    "x_m": None,
                    "y_m": None,
                    "z_m": None,
                    "lat_deg": None,
                    "lon_deg": None,
                    "height_m": None,
                    "has_xyz": bool(parsed.get("has_xyz")),
                    "has_geodetic": bool(parsed.get("has_geodetic")),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "source": "slrhst_remote",
                    "source_group": "slrhst_history",
                    "log_file": url,
                    "log_date": ymd,
                    "date_prepared": None,
                    "x_m": None,
                    "y_m": None,
                    "z_m": None,
                    "lat_deg": None,
                    "lon_deg": None,
                    "height_m": None,
                    "has_xyz": False,
                    "has_geodetic": False,
                    "fetch_error": str(e),
                }
            )
    return rows, None, n_total


def _list_cddis_apol_logs(index_url: str, timeout_s: int, kind: str) -> List[Tuple[str, str]]:
    html = _fetch_text(index_url, timeout_s=timeout_s)
    rows: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for href in _iter_hrefs(html):
        h = str(href or "").strip()
        if not h:
            continue
        file_name = Path(h).name.strip()
        if kind == "slrlog":
            m = re.fullmatch(r"(apol_(\d{8})\.log)", file_name, flags=re.IGNORECASE)
        elif kind == "oldlog":
            m = re.fullmatch(r"(apol_(\d{8})\.log)", file_name, flags=re.IGNORECASE)
        elif kind == "slrhst":
            m = re.fullmatch(r"(apol_hst_(\d{8})\.log)", file_name, flags=re.IGNORECASE)
        else:
            m = None
        if not m:
            continue
        fname = str(m.group(1))
        ymd = str(m.group(2))
        key = f"{kind}:{fname.lower()}"
        if key in seen:
            continue
        seen.add(key)
        rows.append((ymd, f"{index_url.rstrip('/')}/{fname}"))
    rows.sort(reverse=True)
    return rows


def _scan_cddis_site_logs(
    *,
    max_remote_logs: int,
    max_oldlog_logs: int,
    max_slrhst_logs: int,
    timeout_s: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "n_current_total": 0,
        "n_oldlog_total": 0,
        "n_slrhst_total": 0,
        "n_current_scanned": 0,
        "n_oldlog_scanned": 0,
        "n_slrhst_scanned": 0,
        "n_logs_with_xyz": 0,
        "n_logs_with_geodetic": 0,
        "errors": [],
    }

    try:
        current_logs = _list_cddis_apol_logs(CDDIS_SLRLOG_LIST, timeout_s=timeout_s, kind="slrlog")
        diag["n_current_total"] = int(len(current_logs))
    except Exception as e:
        current_logs = []
        diag["errors"].append(f"cddis_slrlog_list_error: {e}")
    try:
        old_logs = _list_cddis_apol_logs(CDDIS_SLRLOG_OLDLOG_LIST, timeout_s=timeout_s, kind="oldlog")
        diag["n_oldlog_total"] = int(len(old_logs))
    except Exception as e:
        old_logs = []
        diag["errors"].append(f"cddis_oldlog_list_error: {e}")
    try:
        hst_logs = _list_cddis_apol_logs(CDDIS_SLRHST_LIST, timeout_s=timeout_s, kind="slrhst")
        diag["n_slrhst_total"] = int(len(hst_logs))
    except Exception as e:
        hst_logs = []
        diag["errors"].append(f"cddis_slrhst_list_error: {e}")

    for ymd, url in current_logs[: max(0, int(max_remote_logs))]:
        rec: Dict[str, Any] = {
            "source": "cddis_slrlog_remote",
            "source_group": "cddis_slrlog",
            "log_file": url,
            "log_date": ymd,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": False,
            "has_geodetic": False,
        }
        try:
            text = _fetch_text(url, timeout_s=timeout_s)
            parsed = _parse_apol_site_log(text)
            rec.update(parsed)
        except Exception as e:
            rec["fetch_error"] = str(e)
        rows.append(rec)
        diag["n_current_scanned"] = int(diag["n_current_scanned"]) + 1
        if bool(rec.get("has_xyz")):
            diag["n_logs_with_xyz"] = int(diag["n_logs_with_xyz"]) + 1
        if bool(rec.get("has_geodetic")):
            diag["n_logs_with_geodetic"] = int(diag["n_logs_with_geodetic"]) + 1

    for ymd, url in old_logs[: max(0, int(max_oldlog_logs))]:
        rec = {
            "source": "cddis_slrlog_oldlog_remote",
            "source_group": "cddis_slrlog_oldlog",
            "log_file": url,
            "log_date": ymd,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": False,
            "has_geodetic": False,
        }
        try:
            text = _fetch_text(url, timeout_s=timeout_s)
            parsed = _parse_apol_site_log(text)
            rec.update(parsed)
        except Exception as e:
            rec["fetch_error"] = str(e)
        rows.append(rec)
        diag["n_oldlog_scanned"] = int(diag["n_oldlog_scanned"]) + 1
        if bool(rec.get("has_xyz")):
            diag["n_logs_with_xyz"] = int(diag["n_logs_with_xyz"]) + 1
        if bool(rec.get("has_geodetic")):
            diag["n_logs_with_geodetic"] = int(diag["n_logs_with_geodetic"]) + 1

    for ymd, url in hst_logs[: max(0, int(max_slrhst_logs))]:
        rec = {
            "source": "cddis_slrhst_remote",
            "source_group": "cddis_slrhst_history",
            "log_file": url,
            "log_date": ymd,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": False,
            "has_geodetic": False,
        }
        try:
            text = _fetch_text(url, timeout_s=timeout_s)
            parsed_hst = _parse_apol_hst_log(text)
            rec["has_xyz"] = bool(parsed_hst.get("has_xyz"))
            rec["has_geodetic"] = bool(parsed_hst.get("has_geodetic"))
        except Exception as e:
            rec["fetch_error"] = str(e)
        rows.append(rec)
        diag["n_slrhst_scanned"] = int(diag["n_slrhst_scanned"]) + 1
        if bool(rec.get("has_xyz")):
            diag["n_logs_with_xyz"] = int(diag["n_logs_with_xyz"]) + 1
        if bool(rec.get("has_geodetic")):
            diag["n_logs_with_geodetic"] = int(diag["n_logs_with_geodetic"]) + 1

    return rows, diag


def _list_cddis_dir_entries(index_url: str, timeout_s: int) -> List[str]:
    def _extract_entries(html_text: str) -> List[str]:
        entries_local: List[str] = []
        seen_local: set[str] = set()
        for href in _iter_hrefs(html_text):
            name = Path(str(href or "").strip()).name.strip()
            if not name or name in ("..", ".", "index.html"):
                continue
            key = name.lower()
            if key in seen_local:
                continue
            seen_local.add(key)
            entries_local.append(name)
        return entries_local

    html = _fetch_text(index_url, timeout_s=timeout_s)
    entries = _extract_entries(html)
    if entries:
        return entries

    if _is_cddis_archive_url(index_url):
        try:
            fallback_html = _fetch_text_direct(index_url, timeout_s=timeout_s)
            if not _looks_like_earthdata_login_html(fallback_html):
                fallback_entries = _extract_entries(fallback_html)
                if fallback_entries:
                    return fallback_entries
        except Exception:
            pass
    return entries


def _scan_cddis_llr_crd_headers(
    *,
    apol_code: str,
    timeout_s: int,
    max_files: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_with_h2_apol": 0,
        "n_with_pad_code": 0,
        "n_with_xyz_marker": 0,
        "n_with_geodetic_marker": 0,
        "source_counts": {},
        "errors": [],
        "sample_urls": [],
    }
    candidate_files: List[Tuple[str, str, str, str]] = []
    roots = [
        ("npt", CDDIS_LLR_NPT_CRD_ROOT),
        ("fr", CDDIS_LLR_FR_CRD_ROOT),
    ]

    def _date_key(fname: str) -> str:
        m = re.search(r"_(\d{8})\.", str(fname), flags=re.IGNORECASE)
        if m:
            return str(m.group(1))
        m = re.search(r"_(\d{6})\.", str(fname), flags=re.IGNORECASE)
        if m:
            return f"{str(m.group(1))}01"
        return "00000000"

    for source_name, root_url in roots:
        for target in CDDIS_LLR_TARGETS:
            target_url = f"{root_url.rstrip('/')}/{target}/"
            try:
                years = _list_cddis_dir_entries(target_url, timeout_s=timeout_s)
            except Exception as e:
                diag["errors"].append(f"{source_name}:{target}:list_year_error={e}")
                continue
            year_names = sorted(
                [v for v in years if re.fullmatch(r"\d{4}", str(v))],
                reverse=True,
            )
            for year in year_names:
                year_url = f"{target_url}{year}/"
                try:
                    names = _list_cddis_dir_entries(year_url, timeout_s=timeout_s)
                except Exception as e:
                    diag["errors"].append(f"{source_name}:{target}:{year}:list_file_error={e}")
                    continue
                for fname in names:
                    fname_l = str(fname).lower()
                    if source_name == "npt":
                        if not fname_l.endswith(".npt"):
                            continue
                    else:
                        if not (fname_l.endswith(".frd") or fname_l.endswith(".frd.gz")):
                            continue
                    candidate_files.append((source_name, target, str(year), f"{year_url}{fname}"))

    candidate_files = sorted(candidate_files, key=lambda v: (_date_key(v[3]), v[0], v[1], v[3]), reverse=True)
    diag["n_candidate_files"] = int(len(candidate_files))
    if int(max_files) > 0:
        candidate_files = candidate_files[: int(max_files)]

    for source_name, target, year, url in candidate_files:
        rec: Dict[str, Any] = {
            "source": f"cddis_llr_crd_{source_name}",
            "source_group": "cddis_llr_crd",
            "log_file": url,
            "log_date": re.search(r"_(\d{8})\.", url).group(1) if re.search(r"_(\d{8})\.", url) else None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": False,
            "has_geodetic": False,
            "cddis_llr_stream": source_name,
            "cddis_llr_target": target,
            "cddis_llr_year": year,
        }
        try:
            data, _meta = _fetch_bytes(url, timeout_s=timeout_s)
            text = _decode_possible_snx(data, url=url)
            if re.search(r"(?mi)^\s*h2\s+APOL\b", text):
                diag["n_with_h2_apol"] = int(diag["n_with_h2_apol"]) + 1
            if re.search(rf"(?mi)^\s*h2\s+APOL\s+{re.escape(str(apol_code))}\b", text):
                diag["n_with_pad_code"] = int(diag["n_with_pad_code"]) + 1
            if re.search(r"(?mi)\bX coordinate\b|\bY coordinate\b|\bZ coordinate\b|\bITRF\b", text):
                rec["has_xyz"] = True
                diag["n_with_xyz_marker"] = int(diag["n_with_xyz_marker"]) + 1
            if re.search(r"(?mi)\bLatitude\b|\bLongitude\b|\bElevation\b", text):
                rec["has_geodetic"] = True
                diag["n_with_geodetic_marker"] = int(diag["n_with_geodetic_marker"]) + 1
        except Exception as e:
            rec["fetch_error"] = str(e)
            diag["n_fetch_error"] = int(diag["n_fetch_error"]) + 1
        rows.append(rec)
        diag["n_files_scanned"] = int(diag["n_files_scanned"]) + 1
        src_count_key = f"{source_name}:{target}"
        diag["source_counts"][src_count_key] = int(diag["source_counts"].get(src_count_key) or 0) + 1
        samples = diag.get("sample_urls")
        if isinstance(samples, list) and len(samples) < 20:
            samples.append(url)

    rows.append(
        {
            "source": "cddis_llr_crd_summary",
            "source_group": "cddis_llr_crd",
            "log_file": CDDIS_LLR_NPT_CRD_ROOT,
            "log_date": None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_with_xyz_marker") or 0) > 0),
            "has_geodetic": bool(int(diag.get("n_with_geodetic_marker") or 0) > 0),
            "cddis_llr_candidate_files": int(diag.get("n_candidate_files") or 0),
            "cddis_llr_scanned_files": int(diag.get("n_files_scanned") or 0),
            "cddis_llr_apol_hits": int(diag.get("n_with_h2_apol") or 0),
            "cddis_llr_pad_hits": int(diag.get("n_with_pad_code") or 0),
            "cddis_llr_xyz_hits": int(diag.get("n_with_xyz_marker") or 0),
        }
    )
    return rows, diag


def _pick_latest_cddis_pos_eop_files(file_names: List[str]) -> List[Tuple[str, str, int, str]]:
    latest_by_center: Dict[str, Tuple[str, str, int, str]] = {}
    for name in file_names:
        m = re.fullmatch(r"([a-z0-9]+)\.pos\+eop\.(\d{6})\.v(\d+)\.snx\.gz", str(name).strip(), flags=re.IGNORECASE)
        if not m:
            continue
        center = str(m.group(1)).lower()
        yymmdd = str(m.group(2))
        ver = int(m.group(3))
        rec = (center, yymmdd, ver, str(name).strip())
        prev = latest_by_center.get(center)
        if prev is None or ver > int(prev[2]):
            latest_by_center[center] = rec
    out = list(latest_by_center.values())
    out.sort(key=lambda v: (str(v[1]), int(v[2]), str(v[0])), reverse=True)
    return out


def _scan_cddis_products_pos_eop_daily(
    *,
    apol_code: str,
    apol_domes: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
    max_days: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_url": CDDIS_PRODUCTS_POS_EOP_ROOT,
        "n_year_dirs": 0,
        "n_day_dirs_total": 0,
        "n_day_dirs_scanned": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "errors": [],
    }

    candidate_days: List[Tuple[str, str, str]] = []
    try:
        year_names = _list_cddis_dir_entries(CDDIS_PRODUCTS_POS_EOP_ROOT, timeout_s=timeout_s)
    except Exception as e:
        diag["errors"] = [f"list_year_error: {e}"]
        rows.append(
            {
                "source": "cddis_pos_eop_daily",
                "source_group": "cddis_pos_eop_daily",
                "log_file": CDDIS_PRODUCTS_POS_EOP_ROOT,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "fetch_error": str(e),
            }
        )
        return rows, diag

    years = sorted([str(v) for v in year_names if re.fullmatch(r"\d{4}", str(v))], reverse=True)
    diag["n_year_dirs"] = int(len(years))
    for year in years:
        year_url = f"{CDDIS_PRODUCTS_POS_EOP_ROOT.rstrip('/')}/{year}/"
        try:
            day_names = _list_cddis_dir_entries(year_url, timeout_s=timeout_s)
        except Exception as e:
            diag["errors"].append(f"{year}:list_day_error={e}")
            continue
        days = sorted([str(v) for v in day_names if re.fullmatch(r"\d{6}", str(v))], reverse=True)
        for day in days:
            candidate_days.append((year, day, f"{year_url}{day}/"))

    candidate_days.sort(key=lambda v: f"{v[0]}{v[1]}", reverse=True)
    diag["n_day_dirs_total"] = int(len(candidate_days))
    if int(max_days) > 0:
        candidate_days = candidate_days[: int(max_days)]
    diag["n_day_dirs_scanned"] = int(len(candidate_days))

    for year, day, day_url in candidate_days:
        try:
            file_names = _list_cddis_dir_entries(day_url, timeout_s=timeout_s)
        except Exception as e:
            diag["errors"].append(f"{year}/{day}:list_file_error={e}")
            continue
        latest_files = _pick_latest_cddis_pos_eop_files(file_names)
        diag["n_candidate_files"] = int(diag["n_candidate_files"]) + int(len(latest_files))

        for center, yymmdd, ver, fname in latest_files:
            url = f"{day_url}{fname}"
            log_date = _yymmdd_to_yyyymmdd(yymmdd)
            rec: Dict[str, Any] = {
                "source": f"cddis_pos_eop_{center}",
                "source_group": "cddis_pos_eop_daily",
                "log_file": url,
                "log_date": log_date,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "cddis_pos_eop_center": center,
                "cddis_pos_eop_yymmdd": yymmdd,
                "cddis_pos_eop_version": int(ver),
            }
            try:
                data, _meta = _fetch_bytes(url, timeout_s=timeout_s)
                text = _decode_possible_snx(data, url=url)
                parsed = _parse_sinex_apol_match(
                    text,
                    apol_code=apol_code,
                    apol_domes=apol_domes,
                    alias_tokens=apol_alias_tokens,
                )
                site_match_n = int(parsed.get("site_match_count") or 0)
                xyz_match_n = int(parsed.get("xyz_match_count") or 0)
                rec["cddis_pos_eop_site_match_n"] = site_match_n
                rec["cddis_pos_eop_xyz_match_n"] = xyz_match_n
                rec["cddis_pos_eop_xyz_keys"] = ",".join([str(v) for v in (parsed.get("xyz_keys") or [])])
                rec["has_xyz"] = bool(xyz_match_n > 0)
                diag["n_site_match"] = int(diag["n_site_match"]) + site_match_n
                diag["n_xyz_match"] = int(diag["n_xyz_match"]) + xyz_match_n
                xyz_candidates = parsed.get("xyz_candidates")
                cand0 = xyz_candidates[0] if isinstance(xyz_candidates, list) and xyz_candidates else None
                if isinstance(cand0, dict):
                    rec["x_m"] = float(cand0.get("stax_m")) if cand0.get("stax_m") is not None else None
                    rec["y_m"] = float(cand0.get("stay_m")) if cand0.get("stay_m") is not None else None
                    rec["z_m"] = float(cand0.get("staz_m")) if cand0.get("staz_m") is not None else None
                    rec["cddis_pos_eop_xyz_key_selected"] = str(cand0.get("key") or "")
            except Exception as e:
                rec["fetch_error"] = str(e)
                diag["n_fetch_error"] = int(diag["n_fetch_error"]) + 1
            rows.append(rec)
            diag["n_files_scanned"] = int(diag["n_files_scanned"]) + 1

    rows.append(
        {
            "source": "cddis_pos_eop_daily_summary",
            "source_group": "cddis_pos_eop_daily",
            "log_file": CDDIS_PRODUCTS_POS_EOP_ROOT,
            "log_date": None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_xyz_match") or 0) > 0),
            "has_geodetic": False,
            "cddis_pos_eop_days_scanned": int(diag.get("n_day_dirs_scanned") or 0),
            "cddis_pos_eop_files_scanned": int(diag.get("n_files_scanned") or 0),
            "cddis_pos_eop_site_match_n": int(diag.get("n_site_match") or 0),
            "cddis_pos_eop_xyz_match_n": int(diag.get("n_xyz_match") or 0),
        }
    )
    return rows, diag


def _pick_cddis_ilrsac_ops_files(
    file_names: List[str],
    *,
    center_hint: Optional[str] = None,
) -> List[Tuple[str, str, int, str]]:
    latest_by_center_day: Dict[Tuple[str, str], Tuple[str, str, int, str]] = {}
    center_hint_u = str(center_hint or "").strip().lower()
    for raw_name in file_names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        m = re.fullmatch(r"([a-z0-9]+)\.pos\+eop\.(\d{6})(?:\.v(\d+))?\.snx(?:\.(?:gz|z))?", name, flags=re.IGNORECASE)
        if not m:
            m = re.fullmatch(r"([a-z0-9]+)\.(\d{6})(?:\.v(\d+))?\.snx(?:\.(?:gz|z))?", name, flags=re.IGNORECASE)
        if not m:
            continue
        center = str(m.group(1)).lower()
        if center_hint_u and center != center_hint_u:
            continue
        yymmdd = str(m.group(2))
        ver = int(m.group(3) or 1)
        key = (center, yymmdd)
        rec = (center, yymmdd, ver, name)
        prev = latest_by_center_day.get(key)
        if prev is None or int(prev[2]) < ver:
            latest_by_center_day[key] = rec

    out = list(latest_by_center_day.values())
    out.sort(key=lambda v: (str(v[1]), int(v[2]), str(v[0])), reverse=True)
    return out


def _pick_cddis_ilrsac_pos_eop_files(file_names: List[str], max_files: int) -> List[Tuple[str, str, int, str]]:
    out = _pick_cddis_ilrsac_ops_files(file_names, center_hint=None)
    if int(max_files) > 0:
        out = out[: int(max_files)]
    return out


def _scan_cddis_ilrsac_ops_pos_eop(
    *,
    apol_code: str,
    apol_domes: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
    max_files: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_url": CDDIS_ILRSAC_PRODUCTS_OPS_POS_EOP_ROOT,
        "ops_root_url": CDDIS_ILRSAC_PRODUCTS_OPS_ROOT,
        "n_dir_entries": 0,
        "n_ops_dir_entries": 0,
        "n_center_dirs_detected": 0,
        "n_center_dirs_scanned": 0,
        "n_candidate_files_flat": 0,
        "n_candidate_files_center_route": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "flat_route_latest_yymmdd": None,
        "center_route_latest_yymmdd": None,
        "latest_candidate_yymmdd": None,
        "latest_candidate_yyyymmdd": None,
        "latest_candidate_age_days": None,
        "stale_archive_route": False,
        "errors": [],
        "candidate_files": [],
    }

    block_tokens = {
        "favicon",
        "jquery",
        "ol",
        "cddis",
        "tophat",
        "background",
        "data",
        "links",
        "faq",
        "staff",
        "contactus",
        "gnss",
        "slr",
        "vlbi",
        "doris",
        "reports",
        "archive",
        "about",
        "programs",
        "publications",
        "home",
        "foia",
        "index",
        "www",
        "icsu",
        "mailto",
        "forum",
        "hp_privacy",
    }

    try:
        entries = _list_cddis_dir_entries(CDDIS_ILRSAC_PRODUCTS_OPS_POS_EOP_ROOT, timeout_s=timeout_s)
    except Exception as e:
        diag["errors"] = [f"list_error: {e}"]
        rows.append(
            {
                "source": "cddis_ilrsac_pos_eop",
                "source_group": "cddis_ilrsac_pos_eop",
                "log_file": CDDIS_ILRSAC_PRODUCTS_OPS_POS_EOP_ROOT,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "fetch_error": str(e),
            }
        )
        return rows, diag

    diag["n_dir_entries"] = int(len(entries))
    flat_candidates = _pick_cddis_ilrsac_ops_files(entries, center_hint=None)
    diag["n_candidate_files_flat"] = int(len(flat_candidates))
    if flat_candidates:
        diag["flat_route_latest_yymmdd"] = str(flat_candidates[0][1])

    center_candidates: List[Tuple[str, str, int, str, str, str]] = []
    try:
        ops_entries = _list_cddis_dir_entries(CDDIS_ILRSAC_PRODUCTS_OPS_ROOT, timeout_s=timeout_s)
        diag["n_ops_dir_entries"] = int(len(ops_entries))
        center_dirs = sorted(
            {
                str(v).strip().lower()
                for v in ops_entries
                if re.fullmatch(r"[a-z0-9]{3,12}", str(v).strip(), flags=re.IGNORECASE)
                and str(v).strip().lower() not in block_tokens
            }
        )
        diag["n_center_dirs_detected"] = int(len(center_dirs))
        center_candidates_flat: List[Tuple[str, str, int, str]] = []
        for center in center_dirs:
            center_url = f"{CDDIS_ILRSAC_PRODUCTS_OPS_ROOT.rstrip('/')}/{center}/"
            try:
                center_entries = _list_cddis_dir_entries(center_url, timeout_s=timeout_s)
            except Exception as e:
                diag["errors"].append(f"center_list_error:{center}={e}")
                continue
            diag["n_center_dirs_scanned"] = int(diag["n_center_dirs_scanned"]) + 1
            center_pick = _pick_cddis_ilrsac_ops_files(center_entries, center_hint=center)
            center_candidates_flat.extend(center_pick)
        diag["n_candidate_files_center_route"] = int(len(center_candidates_flat))
        if center_candidates_flat:
            center_candidates_flat.sort(key=lambda v: (str(v[1]), int(v[2]), str(v[0])), reverse=True)
            diag["center_route_latest_yymmdd"] = str(center_candidates_flat[0][1])
        for center, yymmdd, ver, fname in center_candidates_flat:
            center_url = f"{CDDIS_ILRSAC_PRODUCTS_OPS_ROOT.rstrip('/')}/{center}/"
            file_url = f"{center_url}{fname}"
            center_candidates.append((center, yymmdd, int(ver), fname, file_url, "ops_center"))
    except Exception as e:
        diag["errors"].append(f"ops_root_list_error: {e}")

    merged: Dict[Tuple[str, str], Tuple[str, str, int, str, str, str]] = {}
    for center, yymmdd, ver, fname in flat_candidates:
        url = f"{CDDIS_ILRSAC_PRODUCTS_OPS_POS_EOP_ROOT.rstrip('/')}/{fname}"
        key = (str(center), str(yymmdd))
        merged[key] = (str(center), str(yymmdd), int(ver), str(fname), str(url), "flat_pos+eop")
    for center, yymmdd, ver, fname, url, route_name in center_candidates:
        key = (str(center), str(yymmdd))
        prev = merged.get(key)
        if prev is None or int(prev[2]) < int(ver):
            merged[key] = (str(center), str(yymmdd), int(ver), str(fname), str(url), str(route_name))

    candidates = list(merged.values())
    candidates.sort(key=lambda v: (str(v[1]), int(v[2]), str(v[0])), reverse=True)
    if int(max_files) > 0:
        candidates = candidates[: int(max_files)]
    diag["n_candidate_files"] = int(len(candidates))
    diag["candidate_files"] = [f"{str(v[5])}:{str(v[3])}" for v in candidates]
    if candidates:
        latest_yymmdd = str(candidates[0][1])
        latest_yyyymmdd = _yymmdd_to_yyyymmdd(latest_yymmdd)
        latest_age = _date_age_days_from_yyyymmdd(latest_yyyymmdd)
        diag["latest_candidate_yymmdd"] = latest_yymmdd
        diag["latest_candidate_yyyymmdd"] = latest_yyyymmdd
        diag["latest_candidate_age_days"] = latest_age
        diag["stale_archive_route"] = bool(latest_age is not None and latest_age > 3650)

    for center, yymmdd, ver, fname, url, route_name in candidates:
        rec: Dict[str, Any] = {
            "source": f"cddis_ilrsac_pos_eop_{center}",
            "source_group": "cddis_ilrsac_pos_eop",
            "log_file": url,
            "log_date": _yymmdd_to_yyyymmdd(yymmdd),
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": False,
            "has_geodetic": False,
            "cddis_ilrsac_center": center,
            "cddis_ilrsac_yymmdd": yymmdd,
            "cddis_ilrsac_version": int(ver),
            "cddis_ilrsac_filename": fname,
            "cddis_ilrsac_route": route_name,
        }
        try:
            data, _meta = _fetch_bytes(url, timeout_s=timeout_s)
            text = _decode_possible_snx(data, url=url)
            parsed = _parse_sinex_apol_match(
                text,
                apol_code=apol_code,
                apol_domes=apol_domes,
                alias_tokens=apol_alias_tokens,
            )
            site_match_n = int(parsed.get("site_match_count") or 0)
            xyz_match_n = int(parsed.get("xyz_match_count") or 0)
            rec["cddis_ilrsac_site_match_n"] = site_match_n
            rec["cddis_ilrsac_xyz_match_n"] = xyz_match_n
            rec["cddis_ilrsac_xyz_keys"] = ",".join([str(v) for v in (parsed.get("xyz_keys") or [])])
            rec["has_xyz"] = bool(xyz_match_n > 0)
            diag["n_site_match"] = int(diag.get("n_site_match") or 0) + site_match_n
            diag["n_xyz_match"] = int(diag.get("n_xyz_match") or 0) + xyz_match_n
            xyz_candidates = parsed.get("xyz_candidates")
            cand0 = xyz_candidates[0] if isinstance(xyz_candidates, list) and xyz_candidates else None
            if isinstance(cand0, dict):
                rec["x_m"] = float(cand0.get("stax_m")) if cand0.get("stax_m") is not None else None
                rec["y_m"] = float(cand0.get("stay_m")) if cand0.get("stay_m") is not None else None
                rec["z_m"] = float(cand0.get("staz_m")) if cand0.get("staz_m") is not None else None
                rec["cddis_ilrsac_xyz_key_selected"] = str(cand0.get("key") or "")
        except Exception as e:
            rec["fetch_error"] = str(e)
            diag["n_fetch_error"] = int(diag.get("n_fetch_error") or 0) + 1
            diag["errors"].append(f"{fname}: {e}")
        rows.append(rec)
        diag["n_files_scanned"] = int(diag.get("n_files_scanned") or 0) + 1

    rows.append(
        {
            "source": "cddis_ilrsac_pos_eop_summary",
            "source_group": "cddis_ilrsac_pos_eop",
            "log_file": CDDIS_ILRSAC_PRODUCTS_OPS_POS_EOP_ROOT,
            "log_date": None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_xyz_match") or 0) > 0),
            "has_geodetic": False,
            "cddis_ilrsac_files_scanned": int(diag.get("n_files_scanned") or 0),
            "cddis_ilrsac_site_match_n": int(diag.get("n_site_match") or 0),
            "cddis_ilrsac_xyz_match_n": int(diag.get("n_xyz_match") or 0),
        }
    )
    return rows, diag


def _extract_date_key_from_resource_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return "00000000"
    m = re.search(r"(20\d{2})[._-]?(\d{2})[._-]?(\d{2})", text)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    m = re.search(r"\.(\d{6})\.", text)
    if m:
        ymd = _yymmdd_to_yyyymmdd(str(m.group(1)))
        if ymd:
            return ymd
    m = re.search(r"_(\d{6})_", text)
    if m:
        ymd = _yymmdd_to_yyyymmdd(str(m.group(1)))
        if ymd:
            return ymd
    return "00000000"


def _pick_cddis_resource_sinex_files(file_names: List[str], max_files: int) -> List[Tuple[str, str, str]]:
    candidates: List[Tuple[str, str, str]] = []
    for raw_name in file_names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        name_u = name.upper()
        if not (name_u.endswith(".SNX") or name_u.endswith(".SNX.GZ")):
            continue
        kind = None
        if "ILRS_DATA_HANDLING_FILE" in name_u:
            kind = "ilrs_dhf"
        elif re.search(r"SLRF\d{4}_POS\+VEL", name_u):
            kind = "slrf_posvel"
        if not kind:
            continue
        date_key = _extract_date_key_from_resource_name(name)
        candidates.append((kind, date_key, name))

    candidates.sort(key=lambda v: (str(v[1]), str(v[0]), str(v[2])), reverse=True)
    if int(max_files) > 0:
        candidates = candidates[: int(max_files)]
    return candidates


def _scan_cddis_resource_coordinate_sinex(
    *,
    apol_code: str,
    apol_domes: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
    max_files: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_url": CDDIS_PRODUCTS_RESOURCE_ROOT,
        "n_dir_entries": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "n_site_rows_scanned": 0,
        "n_alias_candidate": 0,
        "n_alias_candidate_with_xyz": 0,
        "errors": [],
        "candidate_files": [],
    }

    try:
        entries = _list_cddis_dir_entries(CDDIS_PRODUCTS_RESOURCE_ROOT, timeout_s=timeout_s)
    except Exception as e:
        diag["errors"] = [f"list_error: {e}"]
        rows.append(
            {
                "source": "cddis_resource_sinex",
                "source_group": "cddis_resource_sinex",
                "log_file": CDDIS_PRODUCTS_RESOURCE_ROOT,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "fetch_error": str(e),
            }
        )
        return rows, diag

    diag["n_dir_entries"] = int(len(entries))
    candidates = _pick_cddis_resource_sinex_files(entries, max_files=max_files)
    diag["n_candidate_files"] = int(len(candidates))
    diag["candidate_files"] = [str(v[2]) for v in candidates]

    for kind, date_key, fname in candidates:
        url = f"{CDDIS_PRODUCTS_RESOURCE_ROOT.rstrip('/')}/{fname}"
        rec: Dict[str, Any] = {
            "source": f"cddis_resource_{kind}",
            "source_group": "cddis_resource_sinex",
            "log_file": url,
            "log_date": date_key if re.fullmatch(r"\d{8}", str(date_key)) else None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": False,
            "has_geodetic": False,
            "cddis_resource_kind": kind,
            "cddis_resource_filename": fname,
        }
        try:
            data, _meta = _fetch_bytes(url, timeout_s=timeout_s)
            text = _decode_possible_snx(data, url=url)
            parsed = _parse_sinex_apol_match(
                text,
                apol_code=apol_code,
                apol_domes=apol_domes,
                alias_tokens=apol_alias_tokens,
            )
            site_match_n = int(parsed.get("site_match_count") or 0)
            xyz_match_n = int(parsed.get("xyz_match_count") or 0)
            site_rows_n = int(parsed.get("site_rows_total") or 0)
            alias_candidate_n = int(parsed.get("alias_candidate_count") or 0)
            alias_candidate_xyz_n = int(parsed.get("alias_candidate_xyz_count") or 0)
            xyz_candidates = parsed.get("xyz_candidates")
            cand0 = xyz_candidates[0] if isinstance(xyz_candidates, list) and xyz_candidates else None

            rec["has_xyz"] = bool(xyz_match_n > 0)
            rec["cddis_resource_site_match_n"] = site_match_n
            rec["cddis_resource_xyz_match_n"] = xyz_match_n
            rec["cddis_resource_site_rows_n"] = site_rows_n
            rec["cddis_resource_alias_candidate_n"] = alias_candidate_n
            rec["cddis_resource_alias_candidate_xyz_n"] = alias_candidate_xyz_n
            rec["cddis_resource_xyz_keys"] = ",".join([str(v) for v in (parsed.get("xyz_keys") or [])])

            if isinstance(cand0, dict):
                rec["x_m"] = float(cand0.get("stax_m")) if cand0.get("stax_m") is not None else None
                rec["y_m"] = float(cand0.get("stay_m")) if cand0.get("stay_m") is not None else None
                rec["z_m"] = float(cand0.get("staz_m")) if cand0.get("staz_m") is not None else None
                rec["cddis_resource_xyz_key_selected"] = str(cand0.get("key") or "")

            diag["n_site_match"] = int(diag["n_site_match"]) + site_match_n
            diag["n_xyz_match"] = int(diag["n_xyz_match"]) + xyz_match_n
            diag["n_site_rows_scanned"] = int(diag["n_site_rows_scanned"]) + site_rows_n
            diag["n_alias_candidate"] = int(diag["n_alias_candidate"]) + alias_candidate_n
            diag["n_alias_candidate_with_xyz"] = int(diag["n_alias_candidate_with_xyz"]) + alias_candidate_xyz_n
        except Exception as e:
            rec["fetch_error"] = str(e)
            diag["n_fetch_error"] = int(diag["n_fetch_error"]) + 1
            diag["errors"].append(f"{fname}: {e}")

        rows.append(rec)
        diag["n_files_scanned"] = int(diag["n_files_scanned"]) + 1

    rows.append(
        {
            "source": "cddis_resource_sinex_summary",
            "source_group": "cddis_resource_sinex",
            "log_file": CDDIS_PRODUCTS_RESOURCE_ROOT,
            "log_date": None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_xyz_match") or 0) > 0),
            "has_geodetic": False,
            "cddis_resource_files_scanned": int(diag.get("n_files_scanned") or 0),
            "cddis_resource_site_match_n": int(diag.get("n_site_match") or 0),
            "cddis_resource_xyz_match_n": int(diag.get("n_xyz_match") or 0),
        }
    )
    return rows, diag


def _pick_cddis_ac_descriptor_files(file_names: List[str], max_files: int) -> List[Tuple[str, str, str]]:
    candidates: List[Tuple[str, str, str]] = []
    seen: set[str] = set()
    for raw_name in file_names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        name_l = name.lower()
        if name_l in seen:
            continue
        if name_l in ("md5sums", "sha512sums", "archive"):
            continue
        if name_l.endswith(".dsc"):
            kind = "dsc"
        elif name_l.endswith(".txt") and "qc" in name_l:
            kind = "qc_txt"
        else:
            continue
        date_key = "00000000"
        m_date8 = re.search(r"(\d{8})", name_l)
        if m_date8:
            date_key = str(m_date8.group(1))
        else:
            m_date6 = re.search(r"(\d{6})", name_l)
            if m_date6:
                date_key = _yymmdd_to_yyyymmdd(str(m_date6.group(1))) or "00000000"
        seen.add(name_l)
        candidates.append((kind, date_key, name))

    candidates.sort(key=lambda v: (str(v[1]), str(v[0]), str(v[2])), reverse=True)
    if int(max_files) > 0:
        candidates = candidates[: int(max_files)]
    return candidates


def _scan_cddis_ac_coordinate_descriptors(
    *,
    apol_code: str,
    apol_domes: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
    max_files: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_urls": [
            CDDIS_PRODUCTS_AC_ROOT,
            CDDIS_PRODUCTS_AC_ARCHIVE_ROOT,
        ],
        "n_dir_entries": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_code_hit": 0,
        "n_domes_hit": 0,
        "n_alias_hit": 0,
        "n_sinex_like_files": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "errors": [],
        "candidate_files": [],
    }
    alias_tokens_norm = _normalize_alias_tokens(apol_alias_tokens)
    apol_code_u = str(apol_code or "").upper().strip()
    apol_domes_u = str(apol_domes or "").upper().strip()

    candidates_all: List[Tuple[str, str, str, str, str]] = []
    roots = [
        ("current", CDDIS_PRODUCTS_AC_ROOT),
        ("archive", CDDIS_PRODUCTS_AC_ARCHIVE_ROOT),
    ]
    for root_kind, root_url in roots:
        try:
            entries = _list_cddis_dir_entries(root_url, timeout_s=timeout_s)
        except Exception as e:
            diag["errors"].append(f"{root_kind}:list_error={e}")
            continue
        diag["n_dir_entries"] = int(diag.get("n_dir_entries") or 0) + int(len(entries))
        picked = _pick_cddis_ac_descriptor_files(entries, max_files=0)
        for kind, date_key, fname in picked:
            candidates_all.append((root_kind, root_url, kind, date_key, fname))

    candidates_all.sort(key=lambda v: (str(v[3]), str(v[2]), str(v[4])), reverse=True)
    dedup_url: set[str] = set()
    candidates: List[Tuple[str, str, str, str, str]] = []
    for root_kind, root_url, kind, date_key, fname in candidates_all:
        url = f"{root_url.rstrip('/')}/{fname}"
        key = str(url).lower()
        if key in dedup_url:
            continue
        dedup_url.add(key)
        candidates.append((root_kind, root_url, kind, date_key, fname))

    if int(max_files) > 0:
        candidates = candidates[: int(max_files)]
    diag["n_candidate_files"] = int(len(candidates))
    diag["candidate_files"] = [f"{v[0]}:{v[4]}" for v in candidates]

    for root_kind, root_url, kind, date_key, fname in candidates:
        url = f"{root_url.rstrip('/')}/{fname}"
        rec: Dict[str, Any] = {
            "source": f"cddis_ac_{kind}",
            "source_group": "cddis_ac_catalog",
            "log_file": url,
            "log_date": date_key if re.fullmatch(r"\d{8}", str(date_key)) else None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": False,
            "has_geodetic": False,
            "cddis_ac_root_kind": root_kind,
            "cddis_ac_kind": kind,
            "cddis_ac_filename": fname,
        }
        try:
            data, _meta = _fetch_bytes(url, timeout_s=timeout_s)
            text = data.decode("utf-8", "replace")
            text_u = str(text or "").upper()
            code_hit = bool(apol_code_u and apol_code_u in text_u)
            domes_hit = bool(apol_domes_u and apol_domes_u in text_u)
            alias_hit = _match_alias_in_desc(desc_u=text_u, alias_tokens=alias_tokens_norm)
            rec["cddis_ac_code_hit"] = code_hit
            rec["cddis_ac_domes_hit"] = domes_hit
            rec["cddis_ac_alias_hit"] = str(alias_hit or "")
            if code_hit:
                diag["n_code_hit"] = int(diag.get("n_code_hit") or 0) + 1
            if domes_hit:
                diag["n_domes_hit"] = int(diag.get("n_domes_hit") or 0) + 1
            if alias_hit:
                diag["n_alias_hit"] = int(diag.get("n_alias_hit") or 0) + 1

            if "+SITE/ID" in text and "+SOLUTION/ESTIMATE" in text:
                diag["n_sinex_like_files"] = int(diag.get("n_sinex_like_files") or 0) + 1
                parsed = _parse_sinex_apol_match(
                    text,
                    apol_code=apol_code,
                    apol_domes=apol_domes,
                    alias_tokens=apol_alias_tokens,
                )
                site_match_n = int(parsed.get("site_match_count") or 0)
                xyz_match_n = int(parsed.get("xyz_match_count") or 0)
                rec["cddis_ac_site_match_n"] = site_match_n
                rec["cddis_ac_xyz_match_n"] = xyz_match_n
                rec["cddis_ac_xyz_keys"] = ",".join([str(v) for v in (parsed.get("xyz_keys") or [])])
                rec["has_xyz"] = bool(xyz_match_n > 0)
                diag["n_site_match"] = int(diag.get("n_site_match") or 0) + site_match_n
                diag["n_xyz_match"] = int(diag.get("n_xyz_match") or 0) + xyz_match_n
                xyz_candidates = parsed.get("xyz_candidates")
                cand0 = xyz_candidates[0] if isinstance(xyz_candidates, list) and xyz_candidates else None
                if isinstance(cand0, dict):
                    rec["x_m"] = float(cand0.get("stax_m")) if cand0.get("stax_m") is not None else None
                    rec["y_m"] = float(cand0.get("stay_m")) if cand0.get("stay_m") is not None else None
                    rec["z_m"] = float(cand0.get("staz_m")) if cand0.get("staz_m") is not None else None
                    rec["cddis_ac_xyz_key_selected"] = str(cand0.get("key") or "")
        except Exception as e:
            rec["fetch_error"] = str(e)
            diag["n_fetch_error"] = int(diag.get("n_fetch_error") or 0) + 1
            diag["errors"].append(f"{root_kind}:{fname}: {e}")
        rows.append(rec)
        diag["n_files_scanned"] = int(diag.get("n_files_scanned") or 0) + 1

    rows.append(
        {
            "source": "cddis_ac_catalog_summary",
            "source_group": "cddis_ac_catalog",
            "log_file": CDDIS_PRODUCTS_AC_ROOT,
            "log_date": None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_xyz_match") or 0) > 0),
            "has_geodetic": False,
            "cddis_ac_files_scanned": int(diag.get("n_files_scanned") or 0),
            "cddis_ac_code_hit_n": int(diag.get("n_code_hit") or 0),
            "cddis_ac_domes_hit_n": int(diag.get("n_domes_hit") or 0),
            "cddis_ac_alias_hit_n": int(diag.get("n_alias_hit") or 0),
            "cddis_ac_site_match_n": int(diag.get("n_site_match") or 0),
            "cddis_ac_xyz_match_n": int(diag.get("n_xyz_match") or 0),
        }
    )
    return rows, diag


def _parse_slrocc_asof_yyyymmdd(text: str) -> Optional[str]:
    m = re.search(r"\bas of\s+(\d{1,2}-[A-Za-z]{3}-\d{4})\b", str(text or ""), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        dt = datetime.strptime(str(m.group(1)), "%d-%b-%Y")
        return dt.strftime("%Y%m%d")
    except Exception:
        return None


def _parse_slrocc_pipe_columns(line: str) -> List[str]:
    raw = str(line or "").strip()
    if "|" not in raw:
        return []
    cols = [v.strip() for v in raw.strip("|").split("|")]
    return cols


def _scan_cddis_slrocc_coordinate_catalog(
    *,
    apol_code: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_SLRCOOR_TXT,
        "asof_yyyymmdd": None,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "errors": [],
    }

    try:
        text = _fetch_text(CDDIS_SLROCC_SLRCOOR_TXT, timeout_s=timeout_s)
    except Exception as e:
        diag["errors"] = [f"fetch_error: {e}"]
        rows.append(
            {
                "source": "cddis_slrocc_slrcoor",
                "source_group": "cddis_slrocc_coordinate_catalog",
                "log_file": CDDIS_SLROCC_SLRCOOR_TXT,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "fetch_error": str(e),
            }
        )
        return rows, diag

    if _looks_like_earthdata_login_html(text):
        try:
            fallback_text = _fetch_text_direct(CDDIS_SLROCC_SLRCOOR_TXT, timeout_s=timeout_s)
            if not _looks_like_earthdata_login_html(fallback_text):
                text = fallback_text
            else:
                diag["errors"] = list(diag.get("errors") or []) + ["fetch_warning: earthdata_login_html_fallback"]
        except Exception as e:
            diag["errors"] = list(diag.get("errors") or []) + [f"fallback_fetch_error: {e}"]

    log_date = _parse_slrocc_asof_yyyymmdd(text)
    diag["asof_yyyymmdd"] = log_date
    alias_tokens = _normalize_alias_tokens(apol_alias_tokens)
    apol_code_u = str(apol_code or "").strip().upper()

    for line in str(text).splitlines():
        cols = _parse_slrocc_pipe_columns(line)
        if len(cols) < 15:
            continue
        sta = str(cols[0] or "").strip()
        if not re.fullmatch(r"\d{4}", sta):
            continue
        diag["n_table_rows"] = int(diag["n_table_rows"]) + 1
        site_name = str(cols[1] or "").strip()
        occ_sys = str(cols[2] or "").strip()
        designtr = str(cols[5] or "").strip()
        desc_u = f"{site_name} {occ_sys}".upper()
        alias_hit = _match_alias_in_desc(desc_u=desc_u, alias_tokens=alias_tokens)
        if not (
            str(sta).upper() == apol_code_u
            or str(designtr).upper().startswith(apol_code_u)
            or bool(alias_hit)
        ):
            continue

        lat_text = str(cols[7] or "").strip()
        lon_text = str(cols[8] or "").strip()
        elev_text = str(cols[9] or "").strip()
        ellip_text = str(cols[10] or "").strip()
        x_text = str(cols[11] or "").strip()
        y_text = str(cols[12] or "").strip()
        z_text = str(cols[13] or "").strip()

        x_m = _extract_float(x_text)
        y_m = _extract_float(y_text)
        z_m = _extract_float(z_text)
        lat_deg = _extract_signed_angle(lat_text, "N", "S")
        lon_deg = _extract_signed_angle(lon_text, "E", "W")
        height_m = _extract_float(ellip_text) if _extract_float(ellip_text) is not None else _extract_float(elev_text)

        has_xyz = bool(x_m is not None and y_m is not None and z_m is not None)
        has_geodetic = bool(lat_deg is not None and lon_deg is not None and height_m is not None)
        diag["n_apol_rows"] = int(diag["n_apol_rows"]) + 1
        if has_xyz:
            diag["n_apol_rows_with_xyz"] = int(diag["n_apol_rows_with_xyz"]) + 1
        else:
            diag["n_apol_rows_xyz_null"] = int(diag["n_apol_rows_xyz_null"]) + 1
        if has_geodetic:
            diag["n_apol_rows_with_geodetic"] = int(diag["n_apol_rows_with_geodetic"]) + 1

        rows.append(
            {
                "source": "cddis_slrocc_slrcoor",
                "source_group": "cddis_slrocc_coordinate_catalog",
                "log_file": CDDIS_SLROCC_SLRCOOR_TXT,
                "log_date": log_date,
                "date_prepared": None,
                "x_m": x_m,
                "y_m": y_m,
                "z_m": z_m,
                "lat_deg": lat_deg,
                "lon_deg": lon_deg,
                "height_m": height_m,
                "has_xyz": has_xyz,
                "has_geodetic": has_geodetic,
                "slrocc_station_code": sta,
                "slrocc_site_name": site_name,
                "slrocc_occ_system": occ_sys,
                "slrocc_designator": designtr,
                "slrocc_alias_hit": alias_hit,
            }
        )

    rows.append(
        {
            "source": "cddis_slrocc_slrcoor_summary",
            "source_group": "cddis_slrocc_coordinate_catalog",
            "log_file": CDDIS_SLROCC_SLRCOOR_TXT,
            "log_date": log_date,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_apol_rows_with_xyz") or 0) > 0),
            "has_geodetic": bool(int(diag.get("n_apol_rows_with_geodetic") or 0) > 0),
            "slrocc_apol_rows": int(diag.get("n_apol_rows") or 0),
            "slrocc_apol_xyz_rows": int(diag.get("n_apol_rows_with_xyz") or 0),
            "slrocc_apol_geod_rows": int(diag.get("n_apol_rows_with_geodetic") or 0),
        }
    )
    return rows, diag


def _scan_cddis_slrocc_occupation_catalog(
    *,
    apol_code: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_OCC_TXT,
        "asof_yyyymmdd": None,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "n_apol_rows_with_domes": 0,
        "errors": [],
    }

    try:
        text = _fetch_text(CDDIS_SLROCC_OCC_TXT, timeout_s=timeout_s)
    except Exception as e:
        diag["errors"] = [f"fetch_error: {e}"]
        rows.append(
            {
                "source": "cddis_slrocc_occupation",
                "source_group": "cddis_slrocc_occupation_catalog",
                "log_file": CDDIS_SLROCC_OCC_TXT,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "fetch_error": str(e),
            }
        )
        return rows, diag

    if _looks_like_earthdata_login_html(text):
        try:
            fallback_text = _fetch_text_direct(CDDIS_SLROCC_OCC_TXT, timeout_s=timeout_s)
            if not _looks_like_earthdata_login_html(fallback_text):
                text = fallback_text
            else:
                diag["errors"] = list(diag.get("errors") or []) + ["fetch_warning: earthdata_login_html_fallback"]
        except Exception as e:
            diag["errors"] = list(diag.get("errors") or []) + [f"fallback_fetch_error: {e}"]

    log_date = _parse_slrocc_asof_yyyymmdd(text)
    diag["asof_yyyymmdd"] = log_date
    alias_tokens = _normalize_alias_tokens(apol_alias_tokens)
    apol_code_u = str(apol_code or "").strip().upper()

    for line in str(text).splitlines():
        cols = _parse_slrocc_pipe_columns(line)
        if len(cols) < 11:
            continue
        sta = str(cols[0] or "").strip()
        if not re.fullmatch(r"\d{4}", sta):
            continue
        diag["n_table_rows"] = int(diag["n_table_rows"]) + 1
        site_name = str(cols[1] or "").strip()
        occ_sys = str(cols[2] or "").strip()
        occ_name = str(cols[3] or "").strip()
        start_date = str(cols[4] or "").strip()
        end_date = str(cols[5] or "").strip()
        designtr = str(cols[6] or "").strip()
        domes = str(cols[7] or "").strip()
        comments = str(cols[10] or "").strip()
        desc_u = f"{site_name} {occ_sys} {occ_name}".upper()
        alias_hit = _match_alias_in_desc(desc_u=desc_u, alias_tokens=alias_tokens)
        if not (
            str(sta).upper() == apol_code_u
            or str(designtr).upper().startswith(apol_code_u)
            or bool(alias_hit)
        ):
            continue

        diag["n_apol_rows"] = int(diag["n_apol_rows"]) + 1
        diag["n_apol_rows_xyz_null"] = int(diag["n_apol_rows_xyz_null"]) + 1
        if domes:
            diag["n_apol_rows_with_domes"] = int(diag["n_apol_rows_with_domes"]) + 1

        rows.append(
            {
                "source": "cddis_slrocc_occupation",
                "source_group": "cddis_slrocc_occupation_catalog",
                "log_file": CDDIS_SLROCC_OCC_TXT,
                "log_date": log_date,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "slrocc_occ_station_code": sta,
                "slrocc_occ_site_name": site_name,
                "slrocc_occ_system": occ_sys,
                "slrocc_occ_name": occ_name,
                "slrocc_occ_start_date": start_date,
                "slrocc_occ_end_date": end_date,
                "slrocc_occ_designator": designtr,
                "slrocc_occ_domes": domes,
                "slrocc_occ_alias_hit": alias_hit,
                "slrocc_occ_comments": comments,
            }
        )

    rows.append(
        {
            "source": "cddis_slrocc_occupation_summary",
            "source_group": "cddis_slrocc_occupation_catalog",
            "log_file": CDDIS_SLROCC_OCC_TXT,
            "log_date": log_date,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_apol_rows_with_xyz") or 0) > 0),
            "has_geodetic": bool(int(diag.get("n_apol_rows_with_geodetic") or 0) > 0),
            "slrocc_occ_apol_rows": int(diag.get("n_apol_rows") or 0),
            "slrocc_occ_apol_xyz_rows": int(diag.get("n_apol_rows_with_xyz") or 0),
            "slrocc_occ_apol_geod_rows": int(diag.get("n_apol_rows_with_geodetic") or 0),
            "slrocc_occ_apol_rows_with_domes": int(diag.get("n_apol_rows_with_domes") or 0),
        }
    )
    return rows, diag


def _scan_cddis_slrocc_calibration_catalog(
    *,
    apol_code: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_SLRCAL_TXT,
        "asof_yyyymmdd": None,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "n_apol_rows_with_cal_target": 0,
        "errors": [],
    }

    try:
        text = _fetch_text(CDDIS_SLROCC_SLRCAL_TXT, timeout_s=timeout_s)
    except Exception as e:
        diag["errors"] = [f"fetch_error: {e}"]
        rows.append(
            {
                "source": "cddis_slrocc_calibration",
                "source_group": "cddis_slrocc_calibration_catalog",
                "log_file": CDDIS_SLROCC_SLRCAL_TXT,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "fetch_error": str(e),
            }
        )
        return rows, diag

    if _looks_like_earthdata_login_html(text):
        try:
            fallback_text = _fetch_text_direct(CDDIS_SLROCC_SLRCAL_TXT, timeout_s=timeout_s)
            if not _looks_like_earthdata_login_html(fallback_text):
                text = fallback_text
            else:
                diag["errors"] = list(diag.get("errors") or []) + ["fetch_warning: earthdata_login_html_fallback"]
        except Exception as e:
            diag["errors"] = list(diag.get("errors") or []) + [f"fallback_fetch_error: {e}"]

    log_date = _parse_slrocc_asof_yyyymmdd(text)
    diag["asof_yyyymmdd"] = log_date
    alias_tokens = _normalize_alias_tokens(apol_alias_tokens)
    apol_code_u = str(apol_code or "").strip().upper()

    for line in str(text).splitlines():
        cols = _parse_slrocc_pipe_columns(line)
        if len(cols) < 12:
            continue
        sta = str(cols[0] or "").strip()
        if not re.fullmatch(r"\d{4}", sta):
            continue
        diag["n_table_rows"] = int(diag["n_table_rows"]) + 1

        site_name = str(cols[1] or "").strip()
        occ_sys = str(cols[2] or "").strip()
        start_date = str(cols[3] or "").strip()
        end_date = str(cols[4] or "").strip()
        designtr = str(cols[5] or "").strip()
        survey_date = str(cols[6] or "").strip()
        cal_target = str(cols[7] or "").strip()
        range_mm = str(cols[8] or "").strip()
        range_unc_mm = str(cols[9] or "").strip()
        azimuth_deg = str(cols[10] or "").strip()
        elev_deg = str(cols[11] or "").strip()

        desc_u = f"{site_name} {occ_sys}".upper()
        alias_hit = _match_alias_in_desc(desc_u=desc_u, alias_tokens=alias_tokens)
        if not (
            str(sta).upper() == apol_code_u
            or str(designtr).upper().startswith(apol_code_u)
            or bool(alias_hit)
        ):
            continue

        diag["n_apol_rows"] = int(diag["n_apol_rows"]) + 1
        diag["n_apol_rows_xyz_null"] = int(diag["n_apol_rows_xyz_null"]) + 1
        cal_target_u = str(cal_target).upper()
        if cal_target and cal_target_u not in ("NULL", "NONE", "N/A", "-"):
            diag["n_apol_rows_with_cal_target"] = int(diag["n_apol_rows_with_cal_target"]) + 1

        rows.append(
            {
                "source": "cddis_slrocc_calibration",
                "source_group": "cddis_slrocc_calibration_catalog",
                "log_file": CDDIS_SLROCC_SLRCAL_TXT,
                "log_date": log_date,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "slrocc_cal_station_code": sta,
                "slrocc_cal_site_name": site_name,
                "slrocc_cal_system": occ_sys,
                "slrocc_cal_start_date": start_date,
                "slrocc_cal_end_date": end_date,
                "slrocc_cal_designator": designtr,
                "slrocc_cal_survey_date": survey_date,
                "slrocc_cal_target": cal_target,
                "slrocc_cal_range_mm": range_mm,
                "slrocc_cal_range_unc_mm": range_unc_mm,
                "slrocc_cal_azimuth_deg": azimuth_deg,
                "slrocc_cal_elev_deg": elev_deg,
                "slrocc_cal_alias_hit": alias_hit,
            }
        )

    rows.append(
        {
            "source": "cddis_slrocc_calibration_summary",
            "source_group": "cddis_slrocc_calibration_catalog",
            "log_file": CDDIS_SLROCC_SLRCAL_TXT,
            "log_date": log_date,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_apol_rows_with_xyz") or 0) > 0),
            "has_geodetic": bool(int(diag.get("n_apol_rows_with_geodetic") or 0) > 0),
            "slrocc_cal_apol_rows": int(diag.get("n_apol_rows") or 0),
            "slrocc_cal_apol_xyz_rows": int(diag.get("n_apol_rows_with_xyz") or 0),
            "slrocc_cal_apol_geod_rows": int(diag.get("n_apol_rows_with_geodetic") or 0),
            "slrocc_cal_apol_rows_with_cal_target": int(diag.get("n_apol_rows_with_cal_target") or 0),
        }
    )
    return rows, diag


def _looks_like_ecef_xyz(x_m: Optional[float], y_m: Optional[float], z_m: Optional[float]) -> bool:
    if x_m is None or y_m is None or z_m is None:
        return False
    try:
        return bool(abs(float(x_m)) > 1.0e5 and abs(float(y_m)) > 1.0e5 and abs(float(z_m)) > 1.0e5)
    except Exception:
        return False


def _scan_cddis_slrocc_spl_coordinate_catalog(
    *,
    apol_code: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_SLRCOOR_SPL,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "errors": [],
    }

    try:
        text = _fetch_text(CDDIS_SLROCC_SLRCOOR_SPL, timeout_s=timeout_s)
    except Exception as e:
        diag["errors"] = [f"fetch_error: {e}"]
        rows.append(
            {
                "source": "cddis_slrocc_slrcoor_spl",
                "source_group": "cddis_slrocc_spl_catalog",
                "log_file": CDDIS_SLROCC_SLRCOOR_SPL,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "fetch_error": str(e),
            }
        )
        return rows, diag

    if _looks_like_earthdata_login_html(text):
        try:
            fallback_text = _fetch_text_direct(CDDIS_SLROCC_SLRCOOR_SPL, timeout_s=timeout_s)
            if not _looks_like_earthdata_login_html(fallback_text):
                text = fallback_text
            else:
                diag["errors"] = list(diag.get("errors") or []) + ["fetch_warning: earthdata_login_html_fallback"]
        except Exception as e:
            diag["errors"] = list(diag.get("errors") or []) + [f"fallback_fetch_error: {e}"]

    alias_tokens = _normalize_alias_tokens(apol_alias_tokens)
    apol_code_u = str(apol_code or "").strip().upper()

    for line in str(text).splitlines():
        m = re.match(r"^\s*(\d{4})\s+([A-Za-z].*)$", str(line))
        if not m:
            continue
        sta = str(m.group(1) or "").strip()
        body = str(m.group(2) or "").strip()
        if not sta:
            continue
        diag["n_table_rows"] = int(diag["n_table_rows"]) + 1

        desc_u = f"{sta} {body}".upper()
        alias_hit = _match_alias_in_desc(desc_u=desc_u, alias_tokens=alias_tokens)
        if not (str(sta).upper() == apol_code_u or bool(alias_hit)):
            continue

        x_m: Optional[float] = None
        y_m: Optional[float] = None
        z_m: Optional[float] = None
        float_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", body)
        if len(float_tokens) >= 3:
            try:
                cand_x = float(float_tokens[-3])
                cand_y = float(float_tokens[-2])
                cand_z = float(float_tokens[-1])
                if _looks_like_ecef_xyz(cand_x, cand_y, cand_z):
                    x_m = cand_x
                    y_m = cand_y
                    z_m = cand_z
            except Exception:
                pass

        has_xyz = bool(_looks_like_ecef_xyz(x_m, y_m, z_m))
        has_geodetic = False
        diag["n_apol_rows"] = int(diag["n_apol_rows"]) + 1
        if has_xyz:
            diag["n_apol_rows_with_xyz"] = int(diag["n_apol_rows_with_xyz"]) + 1
        else:
            diag["n_apol_rows_xyz_null"] = int(diag["n_apol_rows_xyz_null"]) + 1
        if has_geodetic:
            diag["n_apol_rows_with_geodetic"] = int(diag["n_apol_rows_with_geodetic"]) + 1

        rows.append(
            {
                "source": "cddis_slrocc_slrcoor_spl",
                "source_group": "cddis_slrocc_spl_catalog",
                "log_file": CDDIS_SLROCC_SLRCOOR_SPL,
                "log_date": None,
                "date_prepared": None,
                "x_m": x_m,
                "y_m": y_m,
                "z_m": z_m,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": has_xyz,
                "has_geodetic": has_geodetic,
                "slrocc_spl_station_code": sta,
                "slrocc_spl_alias_hit": alias_hit,
                "slrocc_spl_line_excerpt": body[:240],
            }
        )

    rows.append(
        {
            "source": "cddis_slrocc_slrcoor_spl_summary",
            "source_group": "cddis_slrocc_spl_catalog",
            "log_file": CDDIS_SLROCC_SLRCOOR_SPL,
            "log_date": None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_apol_rows_with_xyz") or 0) > 0),
            "has_geodetic": bool(int(diag.get("n_apol_rows_with_geodetic") or 0) > 0),
            "slrocc_spl_apol_rows": int(diag.get("n_apol_rows") or 0),
            "slrocc_spl_apol_xyz_rows": int(diag.get("n_apol_rows_with_xyz") or 0),
            "slrocc_spl_apol_geod_rows": int(diag.get("n_apol_rows_with_geodetic") or 0),
        }
    )
    return rows, diag


def _scan_cddis_slrocc_calibration_spl_catalog(
    *,
    apol_code: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_SLRCAL_SPL,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "errors": [],
    }

    try:
        text = _fetch_text(CDDIS_SLROCC_SLRCAL_SPL, timeout_s=timeout_s)
    except Exception as e:
        diag["errors"] = [f"fetch_error: {e}"]
        rows.append(
            {
                "source": "cddis_slrocc_slrcal_spl",
                "source_group": "cddis_slrocc_calibration_spl_catalog",
                "log_file": CDDIS_SLROCC_SLRCAL_SPL,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "fetch_error": str(e),
            }
        )
        return rows, diag

    if _looks_like_earthdata_login_html(text):
        try:
            fallback_text = _fetch_text_direct(CDDIS_SLROCC_SLRCAL_SPL, timeout_s=timeout_s)
            if not _looks_like_earthdata_login_html(fallback_text):
                text = fallback_text
            else:
                diag["errors"] = list(diag.get("errors") or []) + ["fetch_warning: earthdata_login_html_fallback"]
        except Exception as e:
            diag["errors"] = list(diag.get("errors") or []) + [f"fallback_fetch_error: {e}"]

    alias_tokens = _normalize_alias_tokens(apol_alias_tokens)
    apol_code_u = str(apol_code or "").strip().upper()

    for line in str(text).splitlines():
        m = re.match(r"^\s*(\d{4})\s+([A-Za-z].*)$", str(line))
        if not m:
            continue
        sta = str(m.group(1) or "").strip()
        body = str(m.group(2) or "").strip()
        if not sta:
            continue
        diag["n_table_rows"] = int(diag["n_table_rows"]) + 1

        desc_u = f"{sta} {body}".upper()
        alias_hit = _match_alias_in_desc(desc_u=desc_u, alias_tokens=alias_tokens)
        if not (str(sta).upper() == apol_code_u or bool(alias_hit)):
            continue

        x_m: Optional[float] = None
        y_m: Optional[float] = None
        z_m: Optional[float] = None
        float_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", body)
        if len(float_tokens) >= 3:
            try:
                cand_x = float(float_tokens[-3])
                cand_y = float(float_tokens[-2])
                cand_z = float(float_tokens[-1])
                if _looks_like_ecef_xyz(cand_x, cand_y, cand_z):
                    x_m = cand_x
                    y_m = cand_y
                    z_m = cand_z
            except Exception:
                pass

        has_xyz = bool(_looks_like_ecef_xyz(x_m, y_m, z_m))
        has_geodetic = False
        diag["n_apol_rows"] = int(diag["n_apol_rows"]) + 1
        if has_xyz:
            diag["n_apol_rows_with_xyz"] = int(diag["n_apol_rows_with_xyz"]) + 1
        else:
            diag["n_apol_rows_xyz_null"] = int(diag["n_apol_rows_xyz_null"]) + 1
        if has_geodetic:
            diag["n_apol_rows_with_geodetic"] = int(diag["n_apol_rows_with_geodetic"]) + 1

        rows.append(
            {
                "source": "cddis_slrocc_slrcal_spl",
                "source_group": "cddis_slrocc_calibration_spl_catalog",
                "log_file": CDDIS_SLROCC_SLRCAL_SPL,
                "log_date": None,
                "date_prepared": None,
                "x_m": x_m,
                "y_m": y_m,
                "z_m": z_m,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": has_xyz,
                "has_geodetic": has_geodetic,
                "slrocc_cal_spl_station_code": sta,
                "slrocc_cal_spl_alias_hit": alias_hit,
                "slrocc_cal_spl_line_excerpt": body[:240],
            }
        )

    rows.append(
        {
            "source": "cddis_slrocc_slrcal_spl_summary",
            "source_group": "cddis_slrocc_calibration_spl_catalog",
            "log_file": CDDIS_SLROCC_SLRCAL_SPL,
            "log_date": None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_apol_rows_with_xyz") or 0) > 0),
            "has_geodetic": bool(int(diag.get("n_apol_rows_with_geodetic") or 0) > 0),
            "slrocc_cal_spl_apol_rows": int(diag.get("n_apol_rows") or 0),
            "slrocc_cal_spl_apol_xyz_rows": int(diag.get("n_apol_rows_with_xyz") or 0),
            "slrocc_cal_spl_apol_geod_rows": int(diag.get("n_apol_rows_with_geodetic") or 0),
        }
    )
    return rows, diag


def _pick_cddis_slrocc_daily_coordinate_files(
    *,
    timeout_s: int,
    max_years: int,
    max_files: int,
) -> Tuple[List[Tuple[str, str, str, str, str]], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "n_year_dirs": 0,
        "n_year_dirs_scanned": 0,
        "n_candidate_files_total": 0,
        "n_candidate_files_selected": 0,
        "errors": [],
        "candidate_files": [],
    }
    candidates: List[Tuple[str, str, str, str, str]] = []
    try:
        root_entries = _list_cddis_dir_entries(CDDIS_SLROCC_ROOT, timeout_s=timeout_s)
    except Exception as e:
        diag["errors"] = [f"list_year_error: {e}"]
        return [], diag

    years = sorted([str(v) for v in root_entries if re.fullmatch(r"\d{4}", str(v))], reverse=True)
    diag["n_year_dirs"] = int(len(years))
    if int(max_years) > 0:
        years = years[: int(max_years)]
    diag["n_year_dirs_scanned"] = int(len(years))

    for year in years:
        year_url = f"{CDDIS_SLROCC_ROOT.rstrip('/')}/{year}/"
        try:
            file_names = _list_cddis_dir_entries(year_url, timeout_s=timeout_s)
        except Exception as e:
            diag["errors"].append(f"{year}:list_file_error={e}")
            continue
        for fname in file_names:
            name = str(fname).strip()
            m = re.fullmatch(r"slrcoor\.(\d{6})", name, flags=re.IGNORECASE)
            if not m:
                continue
            yymmdd = str(m.group(1))
            log_date = _yymmdd_to_yyyymmdd(yymmdd) or "00000000"
            url = f"{year_url}{name}"
            candidates.append((log_date, url, name, str(year), yymmdd))

    candidates.sort(key=lambda v: (str(v[0]), str(v[2])), reverse=True)
    diag["n_candidate_files_total"] = int(len(candidates))
    if int(max_files) > 0:
        candidates = candidates[: int(max_files)]
    diag["n_candidate_files_selected"] = int(len(candidates))
    diag["candidate_files"] = [f"{v[3]}/{v[2]}" for v in candidates[:64]]
    return candidates, diag


def _scan_cddis_slrocc_daily_coordinate_catalog(
    *,
    apol_code: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
    max_years: int,
    max_files: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_ROOT,
        "n_year_dirs": 0,
        "n_year_dirs_scanned": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "errors": [],
        "candidate_files": [],
    }
    candidates, pick_diag = _pick_cddis_slrocc_daily_coordinate_files(
        timeout_s=timeout_s,
        max_years=max_years,
        max_files=max_files,
    )
    diag["n_year_dirs"] = int(pick_diag.get("n_year_dirs") or 0)
    diag["n_year_dirs_scanned"] = int(pick_diag.get("n_year_dirs_scanned") or 0)
    diag["n_candidate_files"] = int(pick_diag.get("n_candidate_files_selected") or 0)
    diag["candidate_files"] = list(pick_diag.get("candidate_files") or [])
    if pick_diag.get("errors"):
        diag["errors"].extend([str(v) for v in (pick_diag.get("errors") or []) if str(v)])
    if not candidates:
        rows.append(
            {
                "source": "cddis_slrocc_daily_slrcoor",
                "source_group": "cddis_slrocc_daily_catalog",
                "log_file": CDDIS_SLROCC_ROOT,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "fetch_error": "no_daily_slrcoor_candidates",
            }
        )
        return rows, diag

    alias_tokens = _normalize_alias_tokens(apol_alias_tokens)
    apol_code_u = str(apol_code or "").strip().upper()
    for log_date, url, fname, year, yymmdd in candidates:
        rec: Dict[str, Any] = {
            "source": "cddis_slrocc_daily_slrcoor",
            "source_group": "cddis_slrocc_daily_catalog",
            "log_file": url,
            "log_date": log_date,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": False,
            "has_geodetic": False,
            "cddis_slrocc_daily_filename": fname,
            "cddis_slrocc_daily_year": year,
            "cddis_slrocc_daily_yymmdd": yymmdd,
            "cddis_slrocc_daily_apol_rows": 0,
            "cddis_slrocc_daily_apol_rows_with_xyz": 0,
        }
        try:
            text = _fetch_text(url, timeout_s=timeout_s)
        except Exception as e:
            rec["fetch_error"] = str(e)
            diag["n_fetch_error"] = int(diag["n_fetch_error"]) + 1
            rows.append(rec)
            diag["n_files_scanned"] = int(diag["n_files_scanned"]) + 1
            continue

        file_table_rows = 0
        file_apol_rows = 0
        file_apol_xyz_rows = 0
        file_apol_geod_rows = 0
        file_apol_xyz_null_rows = 0
        for line in str(text).splitlines():
            cols = _parse_slrocc_pipe_columns(line)
            if len(cols) < 15:
                continue
            sta = str(cols[0] or "").strip()
            if not re.fullmatch(r"\d{4}", sta):
                continue
            file_table_rows += 1
            site_name = str(cols[1] or "").strip()
            occ_sys = str(cols[2] or "").strip()
            designtr = str(cols[5] or "").strip()
            desc_u = f"{site_name} {occ_sys}".upper()
            alias_hit = _match_alias_in_desc(desc_u=desc_u, alias_tokens=alias_tokens)
            if not (
                str(sta).upper() == apol_code_u
                or str(designtr).upper().startswith(apol_code_u)
                or bool(alias_hit)
            ):
                continue

            file_apol_rows += 1
            lat_text = str(cols[7] or "").strip()
            lon_text = str(cols[8] or "").strip()
            elev_text = str(cols[9] or "").strip()
            ellip_text = str(cols[10] or "").strip()
            x_text = str(cols[11] or "").strip()
            y_text = str(cols[12] or "").strip()
            z_text = str(cols[13] or "").strip()

            x_m = _extract_float(x_text)
            y_m = _extract_float(y_text)
            z_m = _extract_float(z_text)
            lat_deg = _extract_signed_angle(lat_text, "N", "S")
            lon_deg = _extract_signed_angle(lon_text, "E", "W")
            height_m = _extract_float(ellip_text) if _extract_float(ellip_text) is not None else _extract_float(elev_text)

            has_xyz = bool(x_m is not None and y_m is not None and z_m is not None)
            has_geodetic = bool(lat_deg is not None and lon_deg is not None and height_m is not None)
            if has_xyz:
                file_apol_xyz_rows += 1
            else:
                file_apol_xyz_null_rows += 1
            if has_geodetic:
                file_apol_geod_rows += 1

            rec["x_m"] = x_m
            rec["y_m"] = y_m
            rec["z_m"] = z_m
            rec["lat_deg"] = lat_deg
            rec["lon_deg"] = lon_deg
            rec["height_m"] = height_m
            rec["has_xyz"] = has_xyz
            rec["has_geodetic"] = has_geodetic
            rec["slrocc_station_code"] = sta
            rec["slrocc_site_name"] = site_name
            rec["slrocc_occ_system"] = occ_sys
            rec["slrocc_designator"] = designtr
            rec["slrocc_alias_hit"] = alias_hit
            break

        rec["cddis_slrocc_daily_apol_rows"] = int(file_apol_rows)
        rec["cddis_slrocc_daily_apol_rows_with_xyz"] = int(file_apol_xyz_rows)
        rec["cddis_slrocc_daily_apol_rows_with_geodetic"] = int(file_apol_geod_rows)
        rec["cddis_slrocc_daily_apol_rows_xyz_null"] = int(file_apol_xyz_null_rows)

        diag["n_table_rows"] = int(diag["n_table_rows"]) + int(file_table_rows)
        diag["n_apol_rows"] = int(diag["n_apol_rows"]) + int(file_apol_rows)
        diag["n_apol_rows_with_xyz"] = int(diag["n_apol_rows_with_xyz"]) + int(file_apol_xyz_rows)
        diag["n_apol_rows_with_geodetic"] = int(diag["n_apol_rows_with_geodetic"]) + int(file_apol_geod_rows)
        diag["n_apol_rows_xyz_null"] = int(diag["n_apol_rows_xyz_null"]) + int(file_apol_xyz_null_rows)

        rows.append(rec)
        diag["n_files_scanned"] = int(diag["n_files_scanned"]) + 1

    rows.append(
        {
            "source": "cddis_slrocc_daily_slrcoor_summary",
            "source_group": "cddis_slrocc_daily_catalog",
            "log_file": CDDIS_SLROCC_ROOT,
            "log_date": None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_apol_rows_with_xyz") or 0) > 0),
            "has_geodetic": bool(int(diag.get("n_apol_rows_with_geodetic") or 0) > 0),
            "cddis_slrocc_daily_files_scanned": int(diag.get("n_files_scanned") or 0),
            "cddis_slrocc_daily_apol_rows": int(diag.get("n_apol_rows") or 0),
            "cddis_slrocc_daily_apol_xyz_rows": int(diag.get("n_apol_rows_with_xyz") or 0),
            "cddis_slrocc_daily_apol_geod_rows": int(diag.get("n_apol_rows_with_geodetic") or 0),
        }
    )
    return rows, diag


def _extract_urls(text: str) -> List[str]:
    urls: List[str] = []
    for raw in re.findall(r"https?://\S+", str(text or "")):
        url = raw.strip().rstrip(").,;\"'`>")
        if url and url not in urls:
            urls.append(url)
    return urls


def _collect_ilrs_primary_product_urls(timeout_s: int, max_notices: int) -> Tuple[List[str], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "index_url": EDC_SLRMAIL_INDEX,
        "index_read_error": None,
        "notice_ids_found": [],
        "notice_ids_scanned": [],
        "notice_fetch_errors": [],
        "urls_from_notice": [],
    }
    urls: List[str] = []
    try:
        index_text = _fetch_text(EDC_SLRMAIL_INDEX, timeout_s=timeout_s)
    except Exception as e:
        diag["index_read_error"] = str(e)
        return list(ILRS_PRIMARY_PRODUCT_FALLBACK_URLS), diag

    candidates: List[Tuple[int, int, str]] = []
    for line in str(index_text).splitlines():
        if not any(k in line.lower() for k in ("data handling file", "slrf2020", "eccentricity file")):
            continue
        m = re.match(r"\s*(\d+)\s*\|\s*(\d{4})-\d{2}-\d{2}\s*\|", line)
        if not m:
            continue
        notice_id = int(m.group(1))
        year = int(m.group(2))
        candidates.append((notice_id, year, line))

    candidates.sort(reverse=True)
    diag["notice_ids_found"] = [int(v[0]) for v in candidates]

    for notice_id, year, _line in candidates[: max(0, int(max_notices))]:
        diag["notice_ids_scanned"].append(int(notice_id))
        notice_url = f"{EDC_BASE}/pub/slr/slrmail/{year:04d}/slrmail.{notice_id:05d}"
        try:
            text = _fetch_text(notice_url, timeout_s=timeout_s)
        except Exception as e:
            diag["notice_fetch_errors"].append(f"{notice_id}:{e}")
            continue
        for u in _extract_urls(text):
            ul = u.lower()
            if ("ilrs.gsfc.nasa.gov/docs/" in ul and ("ilrs_data_handling_file" in ul or "slrf2020_pos+vel" in ul)) or (
                "cddis.nasa.gov/archive/slr/slrecc/" in ul and "slrecc" in ul
            ):
                if u not in urls:
                    urls.append(u)

        cddis_slrecc_base = None
        for u in _extract_urls(text):
            ul = u.lower()
            if "cddis.nasa.gov/archive/slr/slrecc/" in ul:
                cddis_slrecc_base = u.rstrip(" /")
                break
        if cddis_slrecc_base:
            if not cddis_slrecc_base.endswith("/"):
                cddis_slrecc_base = f"{cddis_slrecc_base}/"
            for fname in re.findall(r"\b(slrecc\.\d{6}\.ILRS\.(?:xyz|une)\.snx(?:\.gz)?)\b", text, flags=re.IGNORECASE):
                cand = f"{cddis_slrecc_base}{fname}"
                if cand not in urls:
                    urls.append(cand)

    if not urls:
        urls = list(ILRS_PRIMARY_PRODUCT_FALLBACK_URLS)
    diag["urls_from_notice"] = urls
    return urls, diag


def _decode_unix_compress_z(data: bytes) -> bytes:
    try:
        import unlzw3  # type: ignore

        return bytes(unlzw3.unlzw(data))
    except Exception:
        pass

    commands: List[List[str]] = []
    uncompress_path = shutil.which("uncompress")
    if uncompress_path:
        commands.append([uncompress_path, "-c"])
    wsl_path = shutil.which("wsl.exe") or shutil.which("wsl")
    if wsl_path:
        commands.append([wsl_path, "--", "bash", "-lc", "uncompress -c"])

    last_error = "no_decoder_available"
    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=90,
            )
        except Exception as e:
            last_error = f"spawn_error({cmd[0]}): {e}"
            continue
        if int(proc.returncode) == 0 and bool(proc.stdout):
            return bytes(proc.stdout)
        stderr_text = proc.stderr.decode("utf-8", "replace").strip()
        last_error = f"decoder_rc={proc.returncode}; stderr={stderr_text[:240]}"
    raise RuntimeError(f"unix_compress_decode_failed: {last_error}")


def _decode_possible_snx(data: bytes, url: str) -> str:
    blob = data
    if str(url).lower().endswith(".gz"):
        try:
            blob = gzip.decompress(blob)
        except Exception:
            blob = data
    elif str(url).lower().endswith(".z"):
        try:
            blob = _decode_unix_compress_z(blob)
        except Exception:
            blob = data
    try:
        return blob.decode("utf-8", "replace")
    except Exception:
        return blob.decode("latin-1", "replace")


def _scan_ilrs_primary_products(
    *,
    apol_code: str,
    apol_domes: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
    max_notices: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    urls, gather_diag = _collect_ilrs_primary_product_urls(timeout_s=timeout_s, max_notices=max_notices)
    diag: Dict[str, Any] = {
        "n_urls_scanned": 0,
        "n_fetch_ok": 0,
        "n_login_blocked": 0,
        "n_slrecc_urls": 0,
        "n_slrecc_xyz_urls": 0,
        "n_slrecc_une_urls": 0,
        "n_slrecc_fetch_ok": 0,
        "n_slrecc_login_blocked": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "n_site_rows_scanned": 0,
        "n_alias_candidate": 0,
        "n_alias_candidate_with_xyz": 0,
        "alias_tokens": _normalize_alias_tokens(apol_alias_tokens),
        "alias_candidate_examples": [],
        "errors": [],
        "url_summary": [],
        "url_gather": gather_diag,
    }

    for url in urls:
        rec: Dict[str, Any] = {
            "source": "ilrs_primary_products",
            "source_group": "ilrs_primary_products",
            "log_file": url,
            "log_date": None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": False,
            "has_geodetic": False,
        }
        is_slrecc = "cddis.nasa.gov/archive/slr/slrecc/" in str(url).lower() and "slrecc" in str(url).lower()
        if is_slrecc:
            diag["n_slrecc_urls"] = int(diag["n_slrecc_urls"]) + 1
            if ".ilrs.xyz.snx" in str(url).lower():
                diag["n_slrecc_xyz_urls"] = int(diag["n_slrecc_xyz_urls"]) + 1
            if ".ilrs.une.snx" in str(url).lower():
                diag["n_slrecc_une_urls"] = int(diag["n_slrecc_une_urls"]) + 1
        diag["n_urls_scanned"] = int(diag["n_urls_scanned"]) + 1
        try:
            data, meta = _fetch_bytes(url, timeout_s=timeout_s)
        except Exception as e:
            rec["fetch_error"] = str(e)
            diag["errors"].append(f"{url}: {e}")
            rows.append(rec)
            continue

        diag["n_fetch_ok"] = int(diag["n_fetch_ok"]) + 1
        if is_slrecc:
            diag["n_slrecc_fetch_ok"] = int(diag["n_slrecc_fetch_ok"]) + 1
        text_head = data[:1024].decode("utf-8", "replace")
        login_blocked = "Earthdata Login" in text_head or "urs.earthdata.nasa.gov" in text_head
        if login_blocked:
            diag["n_login_blocked"] = int(diag["n_login_blocked"]) + 1
            if is_slrecc:
                diag["n_slrecc_login_blocked"] = int(diag["n_slrecc_login_blocked"]) + 1
            rec["fetch_error"] = "earthdata_login_required"
            rec["content_type"] = meta.get("content_type")
            rows.append(rec)
            diag["url_summary"].append(
                {
                    "url": url,
                    "status": "login_blocked",
                    "content_type": str(meta.get("content_type") or ""),
                    "is_slrecc": bool(is_slrecc),
                }
            )
            continue

        text = _decode_possible_snx(data, url=url)
        parsed = _parse_sinex_apol_match(
            text,
            apol_code=apol_code,
            apol_domes=apol_domes,
            alias_tokens=apol_alias_tokens,
        )
        site_match_n = int(parsed.get("site_match_count") or 0)
        xyz_match_n = int(parsed.get("xyz_match_count") or 0)
        site_rows_n = int(parsed.get("site_rows_total") or 0)
        alias_candidate_n = int(parsed.get("alias_candidate_count") or 0)
        alias_candidate_xyz_n = int(parsed.get("alias_candidate_xyz_count") or 0)
        xyz_candidates = parsed.get("xyz_candidates") if isinstance(parsed.get("xyz_candidates"), list) else []
        alias_candidates = parsed.get("alias_candidates") if isinstance(parsed.get("alias_candidates"), list) else []
        cand0 = xyz_candidates[0] if xyz_candidates else None
        diag["n_site_match"] = int(diag["n_site_match"]) + site_match_n
        diag["n_xyz_match"] = int(diag["n_xyz_match"]) + xyz_match_n
        diag["n_site_rows_scanned"] = int(diag["n_site_rows_scanned"]) + site_rows_n
        diag["n_alias_candidate"] = int(diag["n_alias_candidate"]) + alias_candidate_n
        diag["n_alias_candidate_with_xyz"] = int(diag["n_alias_candidate_with_xyz"]) + alias_candidate_xyz_n
        rec["has_xyz"] = bool(xyz_match_n > 0)
        rec["ilrs_site_match_n"] = site_match_n
        rec["ilrs_xyz_match_n"] = xyz_match_n
        rec["ilrs_site_rows_n"] = site_rows_n
        rec["ilrs_alias_candidate_n"] = alias_candidate_n
        rec["ilrs_alias_candidate_xyz_n"] = alias_candidate_xyz_n
        rec["ilrs_xyz_keys"] = ",".join([str(v.get("key") or "") for v in xyz_candidates if isinstance(v, dict)])
        rec["ilrs_match_by"] = ",".join(sorted({str(v.get("match_by") or "") for v in (parsed.get("site_matches") or []) if v}))
        rec["ilrs_alias_candidate_keys"] = ",".join(
            [str(v.get("key") or "") for v in alias_candidates if isinstance(v, dict) and str(v.get("key") or "")]
        )
        rec["ilrs_content_type"] = str(meta.get("content_type") or "")
        if isinstance(cand0, dict):
            rec["x_m"] = float(cand0.get("stax_m"))
            rec["y_m"] = float(cand0.get("stay_m"))
            rec["z_m"] = float(cand0.get("staz_m"))
            rec["ilrs_xyz_key_selected"] = str(cand0.get("key") or "")
            rec["ilrs_xyz_match_by_selected"] = str(cand0.get("match_by") or "")
        if alias_candidate_n > 0:
            examples = diag.get("alias_candidate_examples")
            if isinstance(examples, list):
                for candidate in alias_candidates[:5]:
                    if len(examples) >= 15:
                        break
                    if not isinstance(candidate, dict):
                        continue
                    examples.append(
                        {
                            "url": url,
                            "key": str(candidate.get("key") or ""),
                            "code": str(candidate.get("code") or ""),
                            "pt": str(candidate.get("pt") or ""),
                            "domes": str(candidate.get("domes") or ""),
                            "cdp_pad": str(candidate.get("cdp_pad") or ""),
                            "candidate_by": str(candidate.get("candidate_by") or ""),
                        }
                    )
        rows.append(rec)
        diag["url_summary"].append(
            {
                "url": url,
                "status": "ok",
                "site_match_n": site_match_n,
                "xyz_match_n": xyz_match_n,
                "site_rows_n": site_rows_n,
                "alias_candidate_n": alias_candidate_n,
                "alias_candidate_xyz_n": alias_candidate_xyz_n,
                "content_type": str(meta.get("content_type") or ""),
                "is_slrecc": bool(is_slrecc),
            }
        )

    return rows, diag


def _pick_cddis_slrecc_catalog_files(file_names: List[str], max_files: int) -> List[Tuple[str, str, str]]:
    candidates: List[Tuple[str, str, str]] = []
    seen: set[str] = set()
    for raw_name in file_names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        name_l = name.lower()
        if name_l in seen:
            continue
        kind = None
        date_key = "00000000"

        m_slrecc = re.fullmatch(r"slrecc\.(\d{6})\.ilrs\.(xyz|une)\.snx(?:\.gz)?", name_l)
        if m_slrecc:
            ymd = _yymmdd_to_yyyymmdd(str(m_slrecc.group(1))) or "00000000"
            kind = f"slrecc_{str(m_slrecc.group(2))}"
            date_key = ymd
        else:
            m_ecc = re.fullmatch(r"ecc_(xyz|une)(?:_(\d{6}))?\.snx(?:\.gz)?", name_l)
            if m_ecc:
                kind = f"ecc_{str(m_ecc.group(1))}"
                y6 = str(m_ecc.group(2) or "").strip()
                if y6:
                    date_key = _yymmdd_to_yyyymmdd(y6) or "00000000"
                else:
                    date_key = "99999999"

        if not kind:
            continue
        seen.add(name_l)
        candidates.append((kind, date_key, name))

    candidates.sort(key=lambda v: (str(v[1]), str(v[0]), str(v[2])), reverse=True)
    if int(max_files) > 0:
        candidates = candidates[: int(max_files)]
    return candidates


def _scan_cddis_slrecc_catalog(
    *,
    apol_code: str,
    apol_domes: str,
    apol_alias_tokens: Optional[List[str]],
    timeout_s: int,
    max_files: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "source_url": CDDIS_SLRECC_LIST,
        "n_dir_entries": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "n_site_rows_scanned": 0,
        "n_alias_candidate": 0,
        "n_alias_candidate_with_xyz": 0,
        "errors": [],
        "candidate_files": [],
    }
    try:
        entries = _list_cddis_dir_entries(CDDIS_SLRECC_LIST, timeout_s=timeout_s)
    except Exception as e:
        diag["errors"] = [f"list_error: {e}"]
        rows.append(
            {
                "source": "cddis_slrecc_catalog",
                "source_group": "cddis_slrecc_catalog",
                "log_file": CDDIS_SLRECC_LIST,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
                "fetch_error": str(e),
            }
        )
        return rows, diag

    diag["n_dir_entries"] = int(len(entries))
    candidates = _pick_cddis_slrecc_catalog_files(entries, max_files=max_files)
    diag["n_candidate_files"] = int(len(candidates))
    diag["candidate_files"] = [str(v[2]) for v in candidates]

    for kind, date_key, fname in candidates:
        url = f"{CDDIS_SLRECC_LIST.rstrip('/')}/{fname}"
        rec: Dict[str, Any] = {
            "source": f"cddis_slrecc_{kind}",
            "source_group": "cddis_slrecc_catalog",
            "log_file": url,
            "log_date": date_key if re.fullmatch(r"\d{8}", str(date_key)) else None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": False,
            "has_geodetic": False,
            "cddis_slrecc_kind": kind,
            "cddis_slrecc_filename": fname,
        }
        try:
            data, _meta = _fetch_bytes(url, timeout_s=timeout_s)
            text = _decode_possible_snx(data, url=url)
            parsed = _parse_sinex_apol_match(
                text,
                apol_code=apol_code,
                apol_domes=apol_domes,
                alias_tokens=apol_alias_tokens,
            )
            site_match_n = int(parsed.get("site_match_count") or 0)
            xyz_match_n = int(parsed.get("xyz_match_count") or 0)
            site_rows_n = int(parsed.get("site_rows_total") or 0)
            alias_candidate_n = int(parsed.get("alias_candidate_count") or 0)
            alias_candidate_xyz_n = int(parsed.get("alias_candidate_xyz_count") or 0)
            xyz_candidates = parsed.get("xyz_candidates")
            cand0 = xyz_candidates[0] if isinstance(xyz_candidates, list) and xyz_candidates else None

            rec["has_xyz"] = bool(xyz_match_n > 0)
            rec["cddis_slrecc_site_match_n"] = site_match_n
            rec["cddis_slrecc_xyz_match_n"] = xyz_match_n
            rec["cddis_slrecc_site_rows_n"] = site_rows_n
            rec["cddis_slrecc_alias_candidate_n"] = alias_candidate_n
            rec["cddis_slrecc_alias_candidate_xyz_n"] = alias_candidate_xyz_n
            rec["cddis_slrecc_xyz_keys"] = ",".join([str(v) for v in (parsed.get("xyz_keys") or [])])

            if isinstance(cand0, dict):
                rec["x_m"] = float(cand0.get("stax_m")) if cand0.get("stax_m") is not None else None
                rec["y_m"] = float(cand0.get("stay_m")) if cand0.get("stay_m") is not None else None
                rec["z_m"] = float(cand0.get("staz_m")) if cand0.get("staz_m") is not None else None
                rec["cddis_slrecc_xyz_key_selected"] = str(cand0.get("key") or "")

            diag["n_site_match"] = int(diag["n_site_match"]) + site_match_n
            diag["n_xyz_match"] = int(diag["n_xyz_match"]) + xyz_match_n
            diag["n_site_rows_scanned"] = int(diag["n_site_rows_scanned"]) + site_rows_n
            diag["n_alias_candidate"] = int(diag["n_alias_candidate"]) + alias_candidate_n
            diag["n_alias_candidate_with_xyz"] = int(diag["n_alias_candidate_with_xyz"]) + alias_candidate_xyz_n
        except Exception as e:
            rec["fetch_error"] = str(e)
            diag["n_fetch_error"] = int(diag["n_fetch_error"]) + 1
            diag["errors"].append(f"{fname}: {e}")
        rows.append(rec)
        diag["n_files_scanned"] = int(diag["n_files_scanned"]) + 1

    rows.append(
        {
            "source": "cddis_slrecc_catalog_summary",
            "source_group": "cddis_slrecc_catalog",
            "log_file": CDDIS_SLRECC_LIST,
            "log_date": None,
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_xyz_match") or 0) > 0),
            "has_geodetic": False,
            "cddis_slrecc_files_scanned": int(diag.get("n_files_scanned") or 0),
            "cddis_slrecc_site_match_n": int(diag.get("n_site_match") or 0),
            "cddis_slrecc_xyz_match_n": int(diag.get("n_xyz_match") or 0),
        }
    )
    return rows, diag


def _list_repro_center_files(center: str, timeout_s: int) -> List[Tuple[str, int, str]]:
    center_l = str(center).strip().lower()
    url = f"{REPRO_ROOT}/{center_l}/"
    html = _fetch_text(url, timeout_s=timeout_s)
    rows: List[Tuple[str, int, str]] = []
    pat = re.compile(
        rf"/pub/slr/products/REPRO_SERIES/REPRO2020/ilrs2020_individual_ac/{re.escape(center_l)}/{re.escape(center_l)}\.pos\+eop\.(\d{{6}})\.v(\d+)\.snx\.gz$",
        flags=re.IGNORECASE,
    )
    for href in _iter_hrefs(html):
        m = pat.search(href)
        if not m:
            continue
        yymmdd = str(m.group(1))
        ver = int(m.group(2))
        u = f"{EDC_BASE}{href}" if href.startswith("/") else href
        rows.append((yymmdd, ver, u))
    return rows


def _pick_repro_latest(rows: List[Tuple[str, int, str]]) -> Optional[Tuple[str, int, str]]:
    if not rows:
        return None

    def _k(v: Tuple[str, int, str]) -> Tuple[int, int]:
        y = _yymmdd_to_yyyymmdd(v[0]) or "00000000"
        return (int(y), int(v[1]))

    return max(rows, key=_k)


def _normalize_alias_tokens(alias_tokens: Optional[List[str]]) -> List[str]:
    tokens: List[str] = []
    seen: set[str] = set()
    for raw in (alias_tokens or []) + ["APOL", "APOLLO", "APACHE POINT"]:
        token = re.sub(r"\s+", " ", str(raw or "").strip().upper())
        if len(token) < 4:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    tokens.sort(key=lambda v: (-len(v), v))
    return tokens


def _build_apol_alias_tokens(apol_meta: Dict[str, Any]) -> List[str]:
    seeds: List[str] = []
    for key in ("station", "site_name", "name"):
        value = str(apol_meta.get(key) or "").strip()
        if value:
            seeds.append(value)
    seeds.extend(
        [
            "APOL",
            "APOLLO",
            "APACHE POINT",
            "APACHE-POINT",
            "APACHEPOINT",
        ]
    )
    return _normalize_alias_tokens(seeds)


def _match_alias_in_desc(desc_u: str, alias_tokens: List[str]) -> Optional[str]:
    compact_desc = re.sub(r"[^A-Z0-9]+", "", str(desc_u or ""))
    for token in alias_tokens:
        token_u = str(token or "").upper().strip()
        if not token_u:
            continue
        if token_u in desc_u:
            return token_u
        compact_tok = re.sub(r"[^A-Z0-9]+", "", token_u)
        if compact_tok and compact_tok in compact_desc:
            return token_u
    return None


def _parse_sinex_apol_match(
    text: str,
    *,
    apol_code: str,
    apol_domes: str,
    alias_tokens: Optional[List[str]] = None,
) -> Dict[str, Any]:
    site_rows: List[Dict[str, str]] = []
    est_types: Dict[Tuple[str, str], set[str]] = {}
    est_values: Dict[Tuple[str, str], Dict[str, float]] = {}
    alias_tokens_norm = _normalize_alias_tokens(alias_tokens)
    apol_code_u = str(apol_code or "").upper().strip()
    apol_domes_u = str(apol_domes or "").upper().strip()
    domes_prefix = ""
    m_domes = re.match(r"^(\d{5})", apol_domes_u)
    if m_domes:
        domes_prefix = str(m_domes.group(1))

    def _triplet(vals: Dict[str, float]) -> Optional[Tuple[float, float, float]]:
        try:
            return (float(vals["STAX"]), float(vals["STAY"]), float(vals["STAZ"]))
        except Exception:
            return None

    in_site = False
    in_est = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("+SITE/ID"):
            in_site = True
            continue
        if in_site and line.startswith("-SITE/ID"):
            in_site = False
            continue
        if line.startswith("+SOLUTION/ESTIMATE"):
            in_est = True
            continue
        if in_est and line.startswith("-SOLUTION/ESTIMATE"):
            in_est = False
            continue

        if in_site:
            s = line.strip()
            if not s or s.startswith("*"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            code = str(parts[0]).strip()
            pt = str(parts[1]).strip()
            domes = str(parts[2]).strip()
            cdp_sod = ""
            cdp_pad = ""
            if parts:
                tail = str(parts[-1]).strip().upper()
                if re.fullmatch(r"[0-9A-Z\-]{4,16}", tail):
                    cdp_sod = tail
                    m_pad = re.match(r"^(\d{4})", cdp_sod)
                    if m_pad:
                        cdp_pad = str(m_pad.group(1))
            desc_tokens: List[str] = []
            if len(parts) >= 13:
                desc_tokens = parts[4:-8]
            if not desc_tokens and len(parts) >= 5:
                desc_tokens = parts[4:]
                if cdp_sod and desc_tokens and str(desc_tokens[-1]).upper() == cdp_sod:
                    desc_tokens = desc_tokens[:-1]
            desc = " ".join(desc_tokens).strip()
            site_rows.append(
                {
                    "code": code,
                    "pt": pt,
                    "domes": domes,
                    "desc": desc,
                    "cdp_sod": cdp_sod,
                    "cdp_pad": cdp_pad,
                }
            )
            continue

        if in_est:
            s = line.strip()
            if not s or s.startswith("*"):
                continue
            parts = s.split()
            if len(parts) < 10:
                continue
            typ = str(parts[1]).strip().upper()
            if typ not in ("STAX", "STAY", "STAZ"):
                continue
            code = str(parts[2]).strip()
            pt = str(parts[3]).strip()
            est_types.setdefault((code, pt), set()).add(typ)
            val: Optional[float] = None
            for idx in (8, 9, len(parts) - 2, len(parts) - 1):
                if idx < 0 or idx >= len(parts):
                    continue
                try:
                    val = float(str(parts[idx]))
                    break
                except Exception:
                    continue
            if val is not None:
                est_values.setdefault((code, pt), {})[typ] = float(val)

    apol_site_matches: List[Dict[str, str]] = []
    alias_candidates: List[Dict[str, str]] = []
    for rec in site_rows:
        by: List[str] = []
        alias_by: List[str] = []
        code_u = str(rec.get("code") or "").upper()
        domes_u = str(rec.get("domes") or "").upper()
        desc_u = str(rec.get("desc") or "").upper()
        cdp_pad_u = str(rec.get("cdp_pad") or "").upper()
        cdp_sod_u = str(rec.get("cdp_sod") or "").upper()
        alias_hit = _match_alias_in_desc(desc_u=desc_u, alias_tokens=alias_tokens_norm)

        if code_u == apol_code_u:
            by.append("code")
            alias_by.append("code_exact")
        if domes_u == apol_domes_u:
            by.append("domes")
            alias_by.append("domes_exact")
        if cdp_pad_u == apol_code_u:
            by.append("cdp_pad")
            alias_by.append("cdp_pad_exact")
        if alias_hit:
            by.append(f"name:{alias_hit}")
            alias_by.append(f"name:{alias_hit}")
        if domes_prefix and domes_u.startswith(domes_prefix):
            alias_by.append(f"domes_prefix:{domes_prefix}")
        if cdp_sod_u.startswith(apol_code_u):
            alias_by.append(f"cdp_sod_prefix:{apol_code_u}")

        if by:
            apol_site_matches.append(
                {
                    "code": str(rec.get("code") or ""),
                    "pt": str(rec.get("pt") or ""),
                    "domes": str(rec.get("domes") or ""),
                    "cdp_pad": str(rec.get("cdp_pad") or ""),
                    "cdp_sod": str(rec.get("cdp_sod") or ""),
                    "match_by": ",".join(by),
                }
            )
        if alias_by:
            alias_candidates.append(
                {
                    "key": f"{str(rec.get('code') or '')}:{str(rec.get('pt') or '')}",
                    "code": str(rec.get("code") or ""),
                    "pt": str(rec.get("pt") or ""),
                    "domes": str(rec.get("domes") or ""),
                    "cdp_pad": str(rec.get("cdp_pad") or ""),
                    "cdp_sod": str(rec.get("cdp_sod") or ""),
                    "desc": str(rec.get("desc") or ""),
                    "candidate_by": ",".join(sorted(set(alias_by))),
                }
            )

    xyz_candidates: List[Dict[str, Any]] = []
    for rec in apol_site_matches:
        code = str(rec.get("code") or "")
        pt = str(rec.get("pt") or "")
        if {"STAX", "STAY", "STAZ"}.issubset(est_types.get((code, pt), set())) and (code, pt) in est_values:
            vals = est_values.get((code, pt), {})
            tri = _triplet(vals)
            if tri is None:
                continue
            xyz_candidates.append(
                {
                    "key": f"{code}:{pt}",
                    "code": code,
                    "pt": pt,
                    "domes": str(rec.get("domes") or ""),
                    "cdp_pad": str(rec.get("cdp_pad") or ""),
                    "cdp_sod": str(rec.get("cdp_sod") or ""),
                    "match_by": str(rec.get("match_by") or ""),
                    "stax_m": tri[0],
                    "stay_m": tri[1],
                    "staz_m": tri[2],
                }
            )
        else:
            for (code2, pt2), types in est_types.items():
                if code2 == code and {"STAX", "STAY", "STAZ"}.issubset(types) and (code2, pt2) in est_values:
                    vals = est_values.get((code2, pt2), {})
                    tri = _triplet(vals)
                    if tri is None:
                        continue
                    xyz_candidates.append(
                        {
                            "key": f"{code2}:{pt2}",
                            "code": code2,
                            "pt": pt2,
                            "domes": str(rec.get("domes") or ""),
                            "cdp_pad": str(rec.get("cdp_pad") or ""),
                            "cdp_sod": str(rec.get("cdp_sod") or ""),
                            "match_by": str(rec.get("match_by") or ""),
                            "stax_m": tri[0],
                            "stay_m": tri[1],
                            "staz_m": tri[2],
                        }
                    )
                    break

    if not xyz_candidates:
        for (code2, pt2), types in est_types.items():
            if (
                str(code2).upper() == apol_code_u
                and {"STAX", "STAY", "STAZ"}.issubset(types)
                and (code2, pt2) in est_values
            ):
                vals = est_values.get((code2, pt2), {})
                tri = _triplet(vals)
                if tri is None:
                    continue
                xyz_candidates.append(
                    {
                        "key": f"{code2}:{pt2}",
                        "code": code2,
                        "pt": pt2,
                        "domes": "",
                        "cdp_pad": "",
                        "cdp_sod": "",
                        "match_by": "code_only_fallback",
                        "stax_m": tri[0],
                        "stay_m": tri[1],
                        "staz_m": tri[2],
                    }
                )

    xyz_by_key: Dict[str, Dict[str, Any]] = {}
    for rec in xyz_candidates:
        key = str(rec.get("key") or "")
        if key and key not in xyz_by_key:
            xyz_by_key[key] = rec
    xyz_candidates = [xyz_by_key[k] for k in sorted(xyz_by_key.keys())]
    xyz_keys = [str(v.get("key") or "") for v in xyz_candidates if str(v.get("key") or "")]
    alias_by_key: Dict[str, Dict[str, str]] = {}
    for rec in alias_candidates:
        key = str(rec.get("key") or "")
        if key and key not in alias_by_key:
            alias_by_key[key] = rec
    alias_candidates = [alias_by_key[k] for k in sorted(alias_by_key.keys())]
    xyz_keys_set = set(xyz_keys)
    alias_xyz_count = sum(1 for rec in alias_candidates if str(rec.get("key") or "") in xyz_keys_set)
    return {
        "alias_tokens": alias_tokens_norm,
        "site_rows_total": int(len(site_rows)),
        "estimate_keys_total": int(len(est_types)),
        "site_matches": apol_site_matches,
        "site_match_count": int(len(apol_site_matches)),
        "alias_candidates": alias_candidates,
        "alias_candidate_count": int(len(alias_candidates)),
        "alias_candidate_xyz_count": int(alias_xyz_count),
        "xyz_match_count": int(len(xyz_candidates)),
        "xyz_keys": xyz_keys,
        "xyz_candidates": xyz_candidates,
    }


def _build_deterministic_merge_route(
    *,
    df: pd.DataFrame,
    apol_code: str,
    apol_domes: str,
    apol_alias_tokens: Optional[List[str]],
    ilrs_diag: Dict[str, Any],
    earthdata_auth_diag: Dict[str, Any],
) -> Dict[str, Any]:
    priorities = [
        "ilrs_primary_products",
        "cddis_slrecc_catalog",
        "cddis_resource_sinex",
        "cddis_ac_catalog",
        "cddis_slrocc_daily_catalog",
        "cddis_slrocc_spl_catalog",
        "cddis_slrocc_occupation_catalog",
        "cddis_slrocc_calibration_catalog",
        "cddis_slrocc_calibration_spl_catalog",
        "cddis_pos_eop_daily",
        "cddis_ilrsac_pos_eop",
        "cddis_slrlog",
        "cddis_slrlog_oldlog",
        "cddis_slrhst_history",
        "cddis_llr_crd",
        "cddis_slrocc_coordinate_catalog",
        "repro2020",
        "edc_remote",
        "slrlog_oldlog",
        "local_cache",
        "slrhst_history",
        "llr_crd_header",
    ]
    labels = {
        "ilrs_primary_products": "ILRS primary products",
        "cddis_slrecc_catalog": "CDDIS slrecc catalog",
        "cddis_resource_sinex": "CDDIS resource SINEX",
        "cddis_ac_catalog": "CDDIS AC descriptor catalog",
        "cddis_slrocc_daily_catalog": "CDDIS slrocc daily catalog",
        "cddis_slrocc_spl_catalog": "CDDIS slrocc SPL catalog",
        "cddis_slrocc_occupation_catalog": "CDDIS slrocc occupation catalog",
        "cddis_slrocc_calibration_catalog": "CDDIS slrocc calibration catalog",
        "cddis_slrocc_calibration_spl_catalog": "CDDIS slrocc calibration SPL catalog",
        "cddis_pos_eop_daily": "CDDIS pos+eop daily",
        "cddis_ilrsac_pos_eop": "CDDIS ILRSAC ops pos+eop",
        "cddis_slrlog": "CDDIS site-log",
        "cddis_slrlog_oldlog": "CDDIS oldlog",
        "cddis_slrhst_history": "CDDIS station-history",
        "cddis_llr_crd": "CDDIS LLR CRD header",
        "cddis_slrocc_coordinate_catalog": "CDDIS slrocc coordinate catalog",
        "repro2020": "REPRO2020 (8 AC)",
        "edc_remote": "EDC site-log",
        "slrlog_oldlog": "EDC oldlog",
        "local_cache": "local station cache",
        "slrhst_history": "ILRS station-history (slrhst)",
        "llr_crd_header": "LLR CRD header",
    }
    grp_counts = df.groupby("source_group").size().to_dict() if len(df) > 0 else {}
    grp_xyz = df[df["has_xyz"] == True].groupby("source_group").size().to_dict() if len(df) > 0 else {}
    grp_geod = df[df["has_geodetic"] == True].groupby("source_group").size().to_dict() if len(df) > 0 else {}
    alias_tokens_norm = _normalize_alias_tokens(apol_alias_tokens)
    alias_rule_text = ", ".join(alias_tokens_norm[:3]) if alias_tokens_norm else "APOL/APOLLO"

    resolved_source_group = None
    xyz_rows = df[df["has_xyz"] == True].copy()
    if len(xyz_rows) > 0:
        for group in priorities:
            if int(grp_xyz.get(group, 0)) > 0:
                resolved_source_group = group
                break
        if not resolved_source_group:
            cand = sorted({str(v) for v in xyz_rows.get("source_group", pd.Series(dtype=object)).tolist() if str(v)})
            resolved_source_group = cand[0] if cand else None

    selected_record: Optional[Dict[str, Any]] = None
    selected_xyz: Optional[Dict[str, Any]] = None
    if resolved_source_group:
        rows = xyz_rows[xyz_rows["source_group"] == resolved_source_group].copy()
        if len(rows) > 0:
            rows["log_date_sort"] = rows["log_date"].apply(lambda v: str(v) if re.fullmatch(r"\d{8}", str(v or "")) else "00000000")
            rows = rows.sort_values(["log_date_sort", "source"], ascending=[False, True])
            top = rows.iloc[0].to_dict()
            selected_record = {
                "source_group": resolved_source_group,
                "source": str(top.get("source") or ""),
                "log_file": str(top.get("log_file") or ""),
                "log_date": str(top.get("log_date") or "") or None,
            }
            try:
                x = float(top.get("x_m"))
                y = float(top.get("y_m"))
                z = float(top.get("z_m"))
                selected_xyz = {
                    "station": "APOL",
                    "x_m": x,
                    "y_m": y,
                    "z_m": z,
                    "coord_source": f"{labels.get(resolved_source_group, resolved_source_group)}",
                    "source_group": resolved_source_group,
                    "log_file": str(top.get("log_file") or ""),
                    "log_date": str(top.get("log_date") or "") or None,
                }
                if str(top.get("pos_eop_yymmdd") or ""):
                    selected_xyz["pos_eop_yymmdd"] = str(top.get("pos_eop_yymmdd"))
                if str(top.get("pos_eop_ref_epoch_utc") or ""):
                    selected_xyz["pos_eop_ref_epoch_utc"] = str(top.get("pos_eop_ref_epoch_utc"))
            except Exception:
                selected_xyz = None

    route_steps: List[Dict[str, Any]] = []
    for idx, group in enumerate(priorities, start=1):
        count_all = int(grp_counts.get(group, 0))
        count_xyz = int(grp_xyz.get(group, 0))
        count_geod = int(grp_geod.get(group, 0))
        status = "not_scanned"
        if resolved_source_group == group:
            status = "selected"
        elif count_xyz > 0:
            status = "candidate_xyz"
        elif count_all > 0:
            status = "scanned_no_xyz"
        detail = {
            "order": int(idx),
            "source_group": group,
            "label": labels.get(group, group),
            "status": status,
            "n_rows": count_all,
            "n_xyz_rows": count_xyz,
            "n_geodetic_rows": count_geod,
            "match_rule": (
                f"station code={apol_code} or DOMES={apol_domes} "
                f"or CDP-SOD prefix={apol_code} or station name contains [{alias_rule_text}]"
            ),
            "xyz_rule": "STAX/STAY/STAZ triplet available for same (code,pt)",
        }
        if group == "ilrs_primary_products":
            detail["earthdata_auth_status"] = str(earthdata_auth_diag.get("status") or "unknown")
            detail["n_slrecc_urls"] = int(ilrs_diag.get("n_slrecc_urls") or 0)
            detail["n_slrecc_login_blocked"] = int(ilrs_diag.get("n_slrecc_login_blocked") or 0)
            detail["n_site_rows_scanned"] = int(ilrs_diag.get("n_site_rows_scanned") or 0)
            detail["n_alias_candidate"] = int(ilrs_diag.get("n_alias_candidate") or 0)
            detail["n_alias_candidate_with_xyz"] = int(ilrs_diag.get("n_alias_candidate_with_xyz") or 0)
        route_steps.append(detail)

    blocked_reasons: List[str] = []
    if not resolved_source_group:
        if int(ilrs_diag.get("n_slrecc_login_blocked") or 0) > 0:
            blocked_reasons.append("CDDIS slrecc の一部が Earthdata 認証未通過で取得不可。")
        if int(ilrs_diag.get("n_site_rows_scanned") or 0) > 0 and int(ilrs_diag.get("n_alias_candidate") or 0) == 0:
            blocked_reasons.append(
                "ILRS primary SITE/ID で APOL alias（code/DOMES/CDP-SOD/name）の候補が 0 件。"
            )
        if int(ilrs_diag.get("n_alias_candidate") or 0) > 0 and int(ilrs_diag.get("n_xyz_match") or 0) == 0:
            blocked_reasons.append(
                "ILRS primary SITE/ID で APOL alias 候補は検出したが、STAX/STAY/STAZ triplet は未確認。"
            )
        if int(ilrs_diag.get("n_urls_scanned") or 0) > 0 and int(ilrs_diag.get("n_xyz_match") or 0) == 0:
            blocked_reasons.append("ILRS primary products（取得済み分）で APOL 7045 の XYZ triplet が未確認。")
        if int(grp_counts.get("cddis_slrecc_catalog", 0)) > 0 and int(grp_xyz.get("cddis_slrecc_catalog", 0)) == 0:
            blocked_reasons.append("CDDIS slrecc catalog（ecc_xyz/ecc_une/slrecc ILRS xyz/une）でも APOL 一次XYZは未確認。")
        if int(grp_counts.get("cddis_resource_sinex", 0)) > 0 and int(grp_xyz.get("cddis_resource_sinex", 0)) == 0:
            blocked_reasons.append("CDDIS resource（ILRS Data Handling File / SLRF POS+VEL）でも APOL 一次XYZは未確認。")
        if int(grp_counts.get("cddis_ac_catalog", 0)) > 0 and int(grp_xyz.get("cddis_ac_catalog", 0)) == 0:
            blocked_reasons.append("CDDIS products/ac（AC descriptor/QC）でも APOL 一次XYZは未確認。")
        if int(grp_counts.get("cddis_slrocc_daily_catalog", 0)) > 0 and int(grp_xyz.get("cddis_slrocc_daily_catalog", 0)) == 0:
            blocked_reasons.append("CDDIS slrocc 日次slrcoor（YYYY/slrcoor.YYMMDD）でも APOL 7045 行の X/Y/Z 列は NULL。")
        if int(grp_counts.get("cddis_slrocc_spl_catalog", 0)) > 0 and int(grp_xyz.get("cddis_slrocc_spl_catalog", 0)) == 0:
            blocked_reasons.append("CDDIS slrocc（slrcoor.spl）でも APOL 7045 の一次XYZは未確認。")
        if int(grp_counts.get("cddis_slrocc_occupation_catalog", 0)) > 0 and int(grp_xyz.get("cddis_slrocc_occupation_catalog", 0)) == 0:
            blocked_reasons.append("CDDIS slrocc（slrocc.txt）でも APOL 7045 occupation 行はあるが XYZ 列は未提供。")
        if int(grp_counts.get("cddis_slrocc_calibration_catalog", 0)) > 0 and int(grp_xyz.get("cddis_slrocc_calibration_catalog", 0)) == 0:
            blocked_reasons.append("CDDIS slrocc（slrcal.txt）でも APOL 7045 calibration 行はあるが XYZ 列は未提供。")
        if int(grp_counts.get("cddis_slrocc_calibration_spl_catalog", 0)) > 0 and int(grp_xyz.get("cddis_slrocc_calibration_spl_catalog", 0)) == 0:
            blocked_reasons.append("CDDIS slrocc（slrcal.spl）でも APOL 7045 calibration 行は未掲載または XYZ 未提供。")
        if int(grp_counts.get("cddis_slrlog", 0)) > 0 and int(grp_xyz.get("cddis_slrlog", 0)) == 0:
            blocked_reasons.append("CDDIS site-log（apol_*.log）でも APOL の XYZ は空欄。")
        if int(grp_counts.get("cddis_slrlog_oldlog", 0)) > 0 and int(grp_xyz.get("cddis_slrlog_oldlog", 0)) == 0:
            blocked_reasons.append("CDDIS oldlog（apol_*.log）でも APOL の XYZ は空欄。")
        if int(grp_counts.get("cddis_slrhst_history", 0)) > 0 and int(grp_xyz.get("cddis_slrhst_history", 0)) == 0:
            blocked_reasons.append("CDDIS station-history（apol_hst_*.log）でも APOL 一次XYZは未確認。")
        if int(grp_counts.get("cddis_llr_crd", 0)) > 0 and int(grp_xyz.get("cddis_llr_crd", 0)) == 0:
            blocked_reasons.append("CDDIS LLR CRD（npt/fr）ヘッダにも APOL 一次XYZ記載は未確認。")
        if int(grp_counts.get("cddis_pos_eop_daily", 0)) > 0 and int(grp_xyz.get("cddis_pos_eop_daily", 0)) == 0:
            blocked_reasons.append("CDDIS products/pos+eop（日次）でも APOL 一次XYZは未確認。")
        if int(grp_counts.get("cddis_ilrsac_pos_eop", 0)) > 0 and int(grp_xyz.get("cddis_ilrsac_pos_eop", 0)) == 0:
            blocked_reasons.append("CDDIS ILRSAC ops pos+eop（.snx.Z）でも APOL 一次XYZは未確認。")
        if int(grp_counts.get("cddis_slrocc_coordinate_catalog", 0)) > 0 and int(grp_xyz.get("cddis_slrocc_coordinate_catalog", 0)) == 0:
            blocked_reasons.append("CDDIS slrocc（slrcoor.txt）の APOL 7045 行でも XYZ 列は NULL。")
        if int(grp_xyz.get("repro2020", 0)) == 0 and int(grp_counts.get("repro2020", 0)) > 0:
            blocked_reasons.append("REPRO2020（8 AC）でも APOL 一次XYZが未確認。")
        if not blocked_reasons:
            blocked_reasons.append("優先ソースで APOL 一次XYZが未確認。")

    return {
        "version": "apol_primary_xyz_merge_v1",
        "status": "resolved" if bool(resolved_source_group) else "watch",
        "target": {"station_code": str(apol_code), "domes": str(apol_domes)},
        "priority_order": priorities,
        "resolved_source_group": resolved_source_group,
        "selected_record": selected_record,
        "selected_xyz": selected_xyz,
        "earthdata_auth": {
            "status": str(earthdata_auth_diag.get("status") or "unknown"),
            "credential_source": earthdata_auth_diag.get("credential_source"),
            "username_hint": earthdata_auth_diag.get("username_hint"),
            "error": earthdata_auth_diag.get("error"),
        },
        "route_steps": route_steps,
        "blocked_reasons": blocked_reasons,
    }


def _scan_repro_sources(
    *,
    apol_code: str,
    apol_domes: str,
    timeout_s: int,
    max_centers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    centers = list(REPRO_CENTERS)[: max(0, int(max_centers))]
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    n_files_scanned = 0
    n_site_match = 0
    n_xyz_match = 0

    for center in centers:
        try:
            cand = _list_repro_center_files(center=center, timeout_s=timeout_s)
        except Exception as e:
            errors.append(f"{center}: list_error={e}")
            rows.append(
                {
                    "source": f"repro2020_{center}",
                    "source_group": "repro2020",
                    "log_file": f"{REPRO_ROOT}/{center}/",
                    "log_date": None,
                    "has_xyz": False,
                    "has_geodetic": False,
                    "repro_center": center,
                    "repro_fetch_error": f"list_error: {e}",
                }
            )
            continue

        latest = _pick_repro_latest(cand)
        if latest is None:
            rows.append(
                {
                    "source": f"repro2020_{center}",
                    "source_group": "repro2020",
                    "log_file": f"{REPRO_ROOT}/{center}/",
                    "log_date": None,
                    "has_xyz": False,
                    "has_geodetic": False,
                    "repro_center": center,
                    "repro_fetch_error": "no_snx_files",
                }
            )
            continue

        yymmdd, ver, url = latest
        yyyymmdd = _yymmdd_to_yyyymmdd(yymmdd)
        n_files_scanned += 1

        try:
            raw = urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "waveP-llr-apol-route-audit/1.0"}),
                timeout=timeout_s,
            ).read()
            if raw.startswith(b"\x1f\x8b"):
                text = gzip.decompress(raw).decode("utf-8", "replace")
            else:
                text = raw.decode("utf-8", "replace")
        except Exception as e:
            errors.append(f"{center}@{yymmdd}: fetch_error={e}")
            rows.append(
                {
                    "source": f"repro2020_{center}",
                    "source_group": "repro2020",
                    "log_file": url,
                    "log_date": yyyymmdd,
                    "has_xyz": False,
                    "has_geodetic": False,
                    "repro_center": center,
                    "repro_yymmdd": yymmdd,
                    "repro_version": int(ver),
                    "repro_fetch_error": str(e),
                }
            )
            continue

        parsed = _parse_sinex_apol_match(text, apol_code=apol_code, apol_domes=apol_domes)
        site_match_n = int(parsed.get("site_match_count") or 0)
        xyz_match_n = int(parsed.get("xyz_match_count") or 0)
        xyz_candidates = parsed.get("xyz_candidates") if isinstance(parsed.get("xyz_candidates"), list) else []
        cand0 = xyz_candidates[0] if xyz_candidates else None
        n_site_match += site_match_n
        n_xyz_match += xyz_match_n

        rows.append(
            {
                "source": f"repro2020_{center}",
                "source_group": "repro2020",
                "log_file": url,
                "log_date": yyyymmdd,
                "has_xyz": bool(xyz_match_n > 0),
                "has_geodetic": False,
                "repro_center": center,
                "repro_yymmdd": yymmdd,
                "repro_version": int(ver),
                "repro_site_match_n": site_match_n,
                "repro_xyz_match_n": xyz_match_n,
                "repro_xyz_keys": ",".join(parsed.get("xyz_keys") or []),
                "repro_xyz_key_selected": (str(cand0.get("key") or "") if isinstance(cand0, dict) else ""),
                "repro_match_by": ",".join(sorted({str(v.get("match_by") or "") for v in (parsed.get("site_matches") or []) if v})),
                "x_m": (float(cand0.get("stax_m")) if isinstance(cand0, dict) and cand0.get("stax_m") is not None else None),
                "y_m": (float(cand0.get("stay_m")) if isinstance(cand0, dict) and cand0.get("stay_m") is not None else None),
                "z_m": (float(cand0.get("staz_m")) if isinstance(cand0, dict) and cand0.get("staz_m") is not None else None),
            }
        )

    diag = {
        "n_centers_requested": int(len(centers)),
        "n_files_scanned": int(n_files_scanned),
        "n_site_match": int(n_site_match),
        "n_xyz_match": int(n_xyz_match),
        "errors": errors,
    }
    return rows, diag


def _scan_llr_crd_header_sources(
    *,
    apol_code: str,
    max_files: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "manifest_exists": bool(LLR_EDC_MANIFEST_PATH.exists()),
        "n_manifest_files_total": 0,
        "n_apol_entries": 0,
        "n_files_scanned": 0,
        "n_missing_cached_files": 0,
        "n_with_h2_apol": 0,
        "n_with_pad_code": 0,
        "n_with_xyz_marker": 0,
        "n_with_geodetic_marker": 0,
        "latest_entry_date": None,
        "errors": [],
    }
    if not LLR_EDC_MANIFEST_PATH.exists():
        diag["errors"] = ["manifest_missing"]
        return rows, diag

    try:
        manifest = json.loads(LLR_EDC_MANIFEST_PATH.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        diag["errors"] = [f"manifest_read_error: {e}"]
        return rows, diag

    files = manifest.get("files")
    if not isinstance(files, list):
        diag["errors"] = ["manifest_files_not_list"]
        return rows, diag

    def _entry_date(v: Dict[str, Any]) -> str:
        name = str(v.get("filename") or "")
        m = re.search(r"_(\d{8})\.", name)
        if m:
            return m.group(1)
        m = re.search(r"_(\d{6})\.", name)
        if m:
            return f"{m.group(1)}01"
        return "00000000"

    diag["n_manifest_files_total"] = int(len(files))
    apol_entries: List[Dict[str, Any]] = []
    for rec in files:
        if not isinstance(rec, dict):
            continue
        st = rec.get("stations")
        if not isinstance(st, list):
            continue
        if any(str(s).upper() == "APOL" for s in st):
            apol_entries.append(rec)

    apol_entries = sorted(apol_entries, key=_entry_date, reverse=True)
    diag["n_apol_entries"] = int(len(apol_entries))
    if apol_entries:
        latest = _entry_date(apol_entries[0])
        diag["latest_entry_date"] = latest if latest != "00000000" else None

    scan_entries = apol_entries[: max(0, int(max_files))] if int(max_files) > 0 else apol_entries
    for rec in scan_entries:
        rel = str(rec.get("cached_path") or "").strip()
        if not rel:
            continue
        path = ROOT / rel
        if not path.exists():
            diag["n_missing_cached_files"] = int(diag["n_missing_cached_files"]) + 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            diag["n_missing_cached_files"] = int(diag["n_missing_cached_files"]) + 1
            continue

        diag["n_files_scanned"] = int(diag["n_files_scanned"]) + 1
        if re.search(r"(?mi)^\s*h2\s+APOL\b", text):
            diag["n_with_h2_apol"] = int(diag["n_with_h2_apol"]) + 1
        if re.search(rf"(?mi)^\s*h2\s+APOL\s+{re.escape(str(apol_code))}\b", text):
            diag["n_with_pad_code"] = int(diag["n_with_pad_code"]) + 1
        if re.search(r"(?mi)\bX coordinate\b|\bY coordinate\b|\bZ coordinate\b|\bITRF\b", text):
            diag["n_with_xyz_marker"] = int(diag["n_with_xyz_marker"]) + 1
        if re.search(r"(?mi)\bLatitude\b|\bLongitude\b|\bElevation\b", text):
            diag["n_with_geodetic_marker"] = int(diag["n_with_geodetic_marker"]) + 1

    rows.append(
        {
            "source": "llr_crd_header_apol",
            "source_group": "llr_crd_header",
            "log_file": str(LLR_EDC_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "log_date": diag.get("latest_entry_date"),
            "date_prepared": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat_deg": None,
            "lon_deg": None,
            "height_m": None,
            "has_xyz": bool(int(diag.get("n_with_xyz_marker") or 0) > 0),
            "has_geodetic": bool(int(diag.get("n_with_geodetic_marker") or 0) > 0),
            "crd_apol_entries": int(diag.get("n_apol_entries") or 0),
            "crd_files_scanned": int(diag.get("n_files_scanned") or 0),
            "crd_missing_cached_files": int(diag.get("n_missing_cached_files") or 0),
            "crd_h2_apol_hits": int(diag.get("n_with_h2_apol") or 0),
            "crd_pad_code_hits": int(diag.get("n_with_pad_code") or 0),
        }
    )
    return rows, diag


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit APOL primary-coordinate route (site-log + REPRO2020 xyz availability).")
    ap.add_argument("--max-remote-logs", type=int, default=120, help="Maximum number of remote APOL logs to scan.")
    ap.add_argument(
        "--max-oldlog-logs",
        type=int,
        default=32,
        help="Maximum number of APOL oldlog site files to scan from slrlog/oldlog.",
    )
    ap.add_argument(
        "--max-repro-centers",
        type=int,
        default=len(REPRO_CENTERS),
        help="Maximum number of REPRO2020 AC centers to scan.",
    )
    ap.add_argument(
        "--max-slrhst-logs",
        type=int,
        default=24,
        help="Maximum number of APOL station-history (slrhst) logs to scan.",
    )
    ap.add_argument(
        "--max-ilrs-slrmail-notices",
        type=int,
        default=3,
        help="Maximum number of slrmail notices to parse when collecting ILRS primary product URLs.",
    )
    ap.add_argument(
        "--earthdata-auth-mode",
        choices=("auto", "off", "force"),
        default="auto",
        help="Earthdata auth mode for CDDIS slrecc access (auto/off/force).",
    )
    ap.add_argument(
        "--earthdata-username",
        type=str,
        default=None,
        help="Earthdata username (optional; if omitted, env/.netrc are used).",
    )
    ap.add_argument(
        "--earthdata-password",
        type=str,
        default=None,
        help="Earthdata password (optional; if omitted, env/.netrc are used).",
    )
    ap.add_argument(
        "--earthdata-netrc",
        type=str,
        default=None,
        help="Optional netrc path for Earthdata credentials (default: ~/.netrc or ~/_netrc).",
    )
    ap.add_argument(
        "--max-crd-header-files",
        type=int,
        default=600,
        help="Maximum number of APOL-tagged local CRD files to scan for coordinate markers.",
    )
    ap.add_argument(
        "--max-cddis-crd-files",
        type=int,
        default=120,
        help="Maximum number of CDDIS LLR CRD files (npt/fr) to scan for APOL coordinate markers.",
    )
    ap.add_argument(
        "--max-cddis-pos-eop-days",
        type=int,
        default=14,
        help="Maximum number of latest CDDIS products/pos+eop day-directories to scan (daily latest per AC).",
    )
    ap.add_argument(
        "--max-cddis-ilrsac-files",
        type=int,
        default=48,
        help="Maximum number of CDDIS ILRSAC ops pos+eop SINEX(.Z/.gz) files to scan.",
    )
    ap.add_argument(
        "--max-cddis-slrecc-files",
        type=int,
        default=16,
        help="Maximum number of CDDIS slrecc catalog files (ecc/slrecc sinex) to scan.",
    )
    ap.add_argument(
        "--max-cddis-resource-files",
        type=int,
        default=10,
        help="Maximum number of CDDIS products/resource SINEX files (ILRS_DHF/SLRF_POS+VEL) to scan.",
    )
    ap.add_argument(
        "--max-cddis-ac-files",
        type=int,
        default=24,
        help="Maximum number of CDDIS products/ac descriptor files (.dsc/.txt) to scan.",
    )
    ap.add_argument(
        "--max-cddis-slrocc-years",
        type=int,
        default=4,
        help="Maximum number of latest CDDIS slrocc year-directories to scan for daily slrcoor files.",
    )
    ap.add_argument(
        "--max-cddis-slrocc-files",
        type=int,
        default=24,
        help="Maximum number of CDDIS slrocc daily slrcoor.YYMMDD files to scan.",
    )
    ap.add_argument("--timeout-s", type=int, default=45, help="HTTP timeout seconds for remote fetch.")
    ap.add_argument("--offline", action="store_true", help="Scan local cached APOL logs only.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not APOL_META_PATH.exists():
        raise FileNotFoundError(f"missing APOL metadata: {APOL_META_PATH}")

    apol_meta = json.loads(APOL_META_PATH.read_text(encoding="utf-8"))
    apol_code = str(apol_meta.get("cdp_pad_id") or "7045")
    apol_domes = str(apol_meta.get("domes_id") or apol_meta.get("domes") or "49447S001")
    apol_alias_tokens = _build_apol_alias_tokens(apol_meta)
    earthdata_auth_diag: Dict[str, Any]
    if args.offline:
        earthdata_auth_diag = {
            "mode": str(args.earthdata_auth_mode),
            "status": "offline_mode_skip",
            "credential_source": None,
            "username_hint": None,
            "error": "offline_mode_skip",
            "netrc_path": str(Path(args.earthdata_netrc).expanduser()) if args.earthdata_netrc else None,
        }
    else:
        earthdata_auth_diag = _configure_earthdata_auth(
            mode=str(args.earthdata_auth_mode),
            username=args.earthdata_username,
            password=args.earthdata_password,
            netrc_path=args.earthdata_netrc,
        )

    local_rows = _scan_local_logs()
    remote_rows: List[Dict[str, Any]] = []
    remote_error: Optional[str] = None
    remote_total = 0
    if not args.offline:
        remote_rows, remote_error, remote_total = _scan_remote_logs(
            max_logs=int(args.max_remote_logs),
            timeout_s=int(args.timeout_s),
        )

    oldlog_rows: List[Dict[str, Any]] = []
    oldlog_error: Optional[str] = None
    oldlog_total = 0
    if not args.offline and int(args.max_oldlog_logs) > 0:
        oldlog_rows, oldlog_error, oldlog_total = _scan_remote_oldlog_logs(
            max_logs=int(args.max_oldlog_logs),
            timeout_s=int(args.timeout_s),
        )

    slrhst_rows: List[Dict[str, Any]] = []
    slrhst_error: Optional[str] = None
    slrhst_total = 0
    if not args.offline and int(args.max_slrhst_logs) > 0:
        slrhst_rows, slrhst_error, slrhst_total = _scan_slrhst_logs(
            max_logs=int(args.max_slrhst_logs),
            timeout_s=int(args.timeout_s),
        )

    cddis_rows: List[Dict[str, Any]] = []
    cddis_diag: Dict[str, Any] = {
        "n_current_total": 0,
        "n_oldlog_total": 0,
        "n_slrhst_total": 0,
        "n_current_scanned": 0,
        "n_oldlog_scanned": 0,
        "n_slrhst_scanned": 0,
        "n_logs_with_xyz": 0,
        "n_logs_with_geodetic": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
    }
    if not args.offline:
        cddis_rows, cddis_diag = _scan_cddis_site_logs(
            max_remote_logs=int(args.max_remote_logs),
            max_oldlog_logs=int(args.max_oldlog_logs),
            max_slrhst_logs=int(args.max_slrhst_logs),
            timeout_s=int(args.timeout_s),
        )

    cddis_llr_crd_rows: List[Dict[str, Any]] = []
    cddis_llr_crd_diag: Dict[str, Any] = {
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_with_h2_apol": 0,
        "n_with_pad_code": 0,
        "n_with_xyz_marker": 0,
        "n_with_geodetic_marker": 0,
        "source_counts": {},
        "errors": ["offline_mode_skip"] if args.offline else [],
        "sample_urls": [],
    }
    if not args.offline and int(args.max_cddis_crd_files) > 0:
        cddis_llr_crd_rows, cddis_llr_crd_diag = _scan_cddis_llr_crd_headers(
            apol_code=apol_code,
            timeout_s=int(args.timeout_s),
            max_files=int(args.max_cddis_crd_files),
        )

    cddis_pos_eop_rows: List[Dict[str, Any]] = []
    cddis_pos_eop_diag: Dict[str, Any] = {
        "source_url": CDDIS_PRODUCTS_POS_EOP_ROOT if not args.offline else None,
        "n_year_dirs": 0,
        "n_day_dirs_total": 0,
        "n_day_dirs_scanned": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
    }
    if not args.offline and int(args.max_cddis_pos_eop_days) > 0:
        cddis_pos_eop_rows, cddis_pos_eop_diag = _scan_cddis_products_pos_eop_daily(
            apol_code=apol_code,
            apol_domes=apol_domes,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
            max_days=int(args.max_cddis_pos_eop_days),
        )

    cddis_ilrsac_rows: List[Dict[str, Any]] = []
    cddis_ilrsac_diag: Dict[str, Any] = {
        "source_url": CDDIS_ILRSAC_PRODUCTS_OPS_POS_EOP_ROOT if not args.offline else None,
        "n_dir_entries": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
        "candidate_files": [],
    }
    if not args.offline and int(args.max_cddis_ilrsac_files) > 0:
        cddis_ilrsac_rows, cddis_ilrsac_diag = _scan_cddis_ilrsac_ops_pos_eop(
            apol_code=apol_code,
            apol_domes=apol_domes,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
            max_files=int(args.max_cddis_ilrsac_files),
        )

    cddis_slrecc_rows: List[Dict[str, Any]] = []
    cddis_slrecc_diag: Dict[str, Any] = {
        "source_url": CDDIS_SLRECC_LIST if not args.offline else None,
        "n_dir_entries": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "n_site_rows_scanned": 0,
        "n_alias_candidate": 0,
        "n_alias_candidate_with_xyz": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
        "candidate_files": [],
    }
    if not args.offline and int(args.max_cddis_slrecc_files) > 0:
        cddis_slrecc_rows, cddis_slrecc_diag = _scan_cddis_slrecc_catalog(
            apol_code=apol_code,
            apol_domes=apol_domes,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
            max_files=int(args.max_cddis_slrecc_files),
        )

    cddis_resource_rows: List[Dict[str, Any]] = []
    cddis_resource_diag: Dict[str, Any] = {
        "source_url": CDDIS_PRODUCTS_RESOURCE_ROOT if not args.offline else None,
        "n_dir_entries": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "n_site_rows_scanned": 0,
        "n_alias_candidate": 0,
        "n_alias_candidate_with_xyz": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
        "candidate_files": [],
    }
    if not args.offline and int(args.max_cddis_resource_files) > 0:
        cddis_resource_rows, cddis_resource_diag = _scan_cddis_resource_coordinate_sinex(
            apol_code=apol_code,
            apol_domes=apol_domes,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
            max_files=int(args.max_cddis_resource_files),
        )

    cddis_ac_rows: List[Dict[str, Any]] = []
    cddis_ac_diag: Dict[str, Any] = {
        "source_urls": (
            [CDDIS_PRODUCTS_AC_ROOT, CDDIS_PRODUCTS_AC_ARCHIVE_ROOT] if not args.offline else []
        ),
        "n_dir_entries": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_code_hit": 0,
        "n_domes_hit": 0,
        "n_alias_hit": 0,
        "n_sinex_like_files": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
        "candidate_files": [],
    }
    if not args.offline and int(args.max_cddis_ac_files) > 0:
        cddis_ac_rows, cddis_ac_diag = _scan_cddis_ac_coordinate_descriptors(
            apol_code=apol_code,
            apol_domes=apol_domes,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
            max_files=int(args.max_cddis_ac_files),
        )

    cddis_slrocc_rows: List[Dict[str, Any]] = []
    cddis_slrocc_diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_SLRCOOR_TXT if not args.offline else None,
        "asof_yyyymmdd": None,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
    }
    if not args.offline:
        cddis_slrocc_rows, cddis_slrocc_diag = _scan_cddis_slrocc_coordinate_catalog(
            apol_code=apol_code,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
        )

    cddis_slrocc_spl_rows: List[Dict[str, Any]] = []
    cddis_slrocc_spl_diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_SLRCOOR_SPL if not args.offline else None,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
    }
    if not args.offline:
        cddis_slrocc_spl_rows, cddis_slrocc_spl_diag = _scan_cddis_slrocc_spl_coordinate_catalog(
            apol_code=apol_code,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
        )

    cddis_slrocc_occ_rows: List[Dict[str, Any]] = []
    cddis_slrocc_occ_diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_OCC_TXT if not args.offline else None,
        "asof_yyyymmdd": None,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "n_apol_rows_with_domes": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
    }
    if not args.offline:
        cddis_slrocc_occ_rows, cddis_slrocc_occ_diag = _scan_cddis_slrocc_occupation_catalog(
            apol_code=apol_code,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
        )

    cddis_slrocc_cal_rows: List[Dict[str, Any]] = []
    cddis_slrocc_cal_diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_SLRCAL_TXT if not args.offline else None,
        "asof_yyyymmdd": None,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "n_apol_rows_with_cal_target": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
    }
    if not args.offline:
        cddis_slrocc_cal_rows, cddis_slrocc_cal_diag = _scan_cddis_slrocc_calibration_catalog(
            apol_code=apol_code,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
        )

    cddis_slrocc_cal_spl_rows: List[Dict[str, Any]] = []
    cddis_slrocc_cal_spl_diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_SLRCAL_SPL if not args.offline else None,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
    }
    if not args.offline:
        cddis_slrocc_cal_spl_rows, cddis_slrocc_cal_spl_diag = _scan_cddis_slrocc_calibration_spl_catalog(
            apol_code=apol_code,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
        )

    cddis_slrocc_daily_rows: List[Dict[str, Any]] = []
    cddis_slrocc_daily_diag: Dict[str, Any] = {
        "source_url": CDDIS_SLROCC_ROOT if not args.offline else None,
        "n_year_dirs": 0,
        "n_year_dirs_scanned": 0,
        "n_candidate_files": 0,
        "n_files_scanned": 0,
        "n_fetch_error": 0,
        "n_table_rows": 0,
        "n_apol_rows": 0,
        "n_apol_rows_with_xyz": 0,
        "n_apol_rows_with_geodetic": 0,
        "n_apol_rows_xyz_null": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
        "candidate_files": [],
    }
    if not args.offline and int(args.max_cddis_slrocc_files) > 0:
        cddis_slrocc_daily_rows, cddis_slrocc_daily_diag = _scan_cddis_slrocc_daily_coordinate_catalog(
            apol_code=apol_code,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
            max_years=int(args.max_cddis_slrocc_years),
            max_files=int(args.max_cddis_slrocc_files),
        )

    ilrs_rows: List[Dict[str, Any]] = []
    ilrs_diag: Dict[str, Any] = {
        "n_urls_scanned": 0,
        "n_fetch_ok": 0,
        "n_login_blocked": 0,
        "n_slrecc_urls": 0,
        "n_slrecc_xyz_urls": 0,
        "n_slrecc_une_urls": 0,
        "n_slrecc_fetch_ok": 0,
        "n_slrecc_login_blocked": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
        "url_summary": [],
        "url_gather": {
            "index_url": EDC_SLRMAIL_INDEX,
            "index_read_error": "offline_mode_skip" if args.offline else None,
            "notice_ids_found": [],
            "notice_ids_scanned": [],
            "notice_fetch_errors": [],
            "urls_from_notice": [],
        },
    }
    if not args.offline and int(args.max_ilrs_slrmail_notices) > 0:
        ilrs_rows, ilrs_diag = _scan_ilrs_primary_products(
            apol_code=apol_code,
            apol_domes=apol_domes,
            apol_alias_tokens=apol_alias_tokens,
            timeout_s=int(args.timeout_s),
            max_notices=int(args.max_ilrs_slrmail_notices),
        )

    repro_rows: List[Dict[str, Any]] = []
    repro_diag: Dict[str, Any] = {
        "n_centers_requested": 0,
        "n_files_scanned": 0,
        "n_site_match": 0,
        "n_xyz_match": 0,
        "errors": ["offline_mode_skip"] if args.offline else [],
    }
    if not args.offline and int(args.max_repro_centers) > 0:
        repro_rows, repro_diag = _scan_repro_sources(
            apol_code=apol_code,
            apol_domes=apol_domes,
            timeout_s=int(args.timeout_s),
            max_centers=int(args.max_repro_centers),
        )

    crd_rows, crd_diag = _scan_llr_crd_header_sources(
        apol_code=apol_code,
        max_files=int(args.max_crd_header_files),
    )

    all_rows = (
        local_rows
        + remote_rows
        + oldlog_rows
        + slrhst_rows
        + cddis_rows
        + cddis_llr_crd_rows
        + cddis_pos_eop_rows
        + cddis_ilrsac_rows
        + cddis_slrecc_rows
        + cddis_resource_rows
        + cddis_ac_rows
        + cddis_slrocc_rows
        + cddis_slrocc_spl_rows
        + cddis_slrocc_occ_rows
        + cddis_slrocc_cal_rows
        + cddis_slrocc_cal_spl_rows
        + cddis_slrocc_daily_rows
        + repro_rows
        + ilrs_rows
        + crd_rows
    )
    if not all_rows:
        all_rows = [
            {
                "source": "none",
                "source_group": "none",
                "log_file": None,
                "log_date": None,
                "date_prepared": None,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "lat_deg": None,
                "lon_deg": None,
                "height_m": None,
                "has_xyz": False,
                "has_geodetic": False,
            }
        ]

    df = pd.DataFrame(all_rows)
    if "log_date" not in df.columns:
        df["log_date"] = None
    if "source_group" not in df.columns:
        df["source_group"] = df["source"]
    if "has_xyz" not in df.columns:
        df["has_xyz"] = False
    if "has_geodetic" not in df.columns:
        df["has_geodetic"] = False

    def _log_sort_key(v: Any) -> str:
        s = str(v) if v is not None else ""
        return s if re.fullmatch(r"\d{8}", s) else "00000000"

    df["log_date_sort"] = df["log_date"].apply(_log_sort_key)
    df = df.sort_values(["source_group", "source", "log_date_sort"], ascending=[True, True, False]).reset_index(drop=True)
    df_out = df.drop(columns=["log_date_sort"], errors="ignore")
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    xyz_rows = df[df["has_xyz"] == True].copy()
    geod_rows = df[df["has_geodetic"] == True].copy()

    def _latest_log_date(rows: pd.DataFrame) -> Optional[str]:
        vals: List[str] = []
        for v in rows.get("log_date", pd.Series(dtype=object)).tolist():
            s = str(v) if v is not None else ""
            if re.fullmatch(r"\d{8}", s):
                vals.append(s)
        return max(vals) if vals else None

    latest_xyz = _latest_log_date(xyz_rows)
    latest_geod = _latest_log_date(geod_rows)

    has_primary_xyz = len(xyz_rows) > 0
    has_geodetic_only = len(geod_rows) > 0 and not has_primary_xyz
    repro_site_match_n = int(repro_diag.get("n_site_match") or 0)
    repro_xyz_match_n = int(repro_diag.get("n_xyz_match") or 0)

    if has_primary_xyz:
        xyz_sources = sorted({str(v) for v in xyz_rows.get("source_group", pd.Series(dtype=object)).tolist() if str(v)})
        decision = "pass"
        reasons = [
            f"APOL一次座標（X/Y/Z）が監査ソースで確認できた（source_group={','.join(xyz_sources)}）。",
        ]
        next_actions = [
            "APOL X/Y/Z を station override に実装し、pos+eop運用との差分監査を実行する。",
            "適用epoch/velocityの整合を確認して iers_station_coord_unified の解除判定を実施する。",
        ]
    elif has_geodetic_only:
        decision = "watch"
        reasons = [
            "APOL site-log は緯度/経度/標高のみで、X/Y/Z（ITRF）が未記載。",
            f"pos+eopキャッシュ側にも pad code {apol_code} が無く、一次座標の統一が未達。",
        ]
        if repro_site_match_n > 0 and repro_xyz_match_n == 0:
            reasons.append(
                "REPRO2020 では APOL 識別子一致は見つかるが、STAX/STAY/STAZ の三成分が揃う一次XYZは確認できない。"
            )
        if oldlog_total > 0:
            reasons.append("APOL oldlog（slrlog/oldlog）でも一次XYZの記載は確認できない。")
        if slrhst_total > 0:
            reasons.append("APOL station-history（slrhst）でも一次XYZの記載は確認できない。")
        if int(cddis_diag.get("n_current_scanned") or 0) > 0 and int(cddis_diag.get("n_logs_with_xyz") or 0) == 0:
            reasons.append("CDDIS site-log（apol_*.log）でも一次XYZは未記載（X/Y/Z空欄）。")
        if int(cddis_diag.get("n_oldlog_scanned") or 0) > 0 and int(cddis_diag.get("n_logs_with_xyz") or 0) == 0:
            reasons.append("CDDIS oldlog（apol_*.log）でも一次XYZは未記載（X/Y/Z空欄）。")
        if int(cddis_diag.get("n_slrhst_scanned") or 0) > 0 and int(cddis_diag.get("n_logs_with_xyz") or 0) == 0:
            reasons.append("CDDIS station-history（apol_hst_*.log）でも一次XYZは未確認。")
        if int(cddis_llr_crd_diag.get("n_files_scanned") or 0) > 0 and int(cddis_llr_crd_diag.get("n_with_xyz_marker") or 0) == 0:
            reasons.append("CDDIS LLR CRD（npt/fr）ヘッダにも APOL 一次XYZ記載は確認できない。")
        if int(cddis_pos_eop_diag.get("n_files_scanned") or 0) > 0 and int(cddis_pos_eop_diag.get("n_xyz_match") or 0) == 0:
            reasons.append("CDDIS products/pos+eop（日次）でも APOL 一次XYZは確認できない。")
        if int(cddis_ilrsac_diag.get("n_files_scanned") or 0) > 0 and int(cddis_ilrsac_diag.get("n_xyz_match") or 0) == 0:
            reasons.append("CDDIS ILRSAC ops pos+eop（.snx.Z）でも APOL 一次XYZは確認できない。")
        ilrsac_latest_yyyymmdd = str(cddis_ilrsac_diag.get("latest_candidate_yyyymmdd") or "").strip()
        ilrsac_latest_age_days = cddis_ilrsac_diag.get("latest_candidate_age_days")
        if bool(cddis_ilrsac_diag.get("stale_archive_route")) and ilrsac_latest_yyyymmdd:
            reasons.append(
                f"CDDIS ILRSAC ops archive の最新候補は {ilrsac_latest_yyyymmdd}（約{int(ilrsac_latest_age_days or 0)}日前）で、"
                "現代座標統一の主経路には使えないため低優先監視へ降格。"
            )
        if int(cddis_slrecc_diag.get("n_files_scanned") or 0) > 0 and int(cddis_slrecc_diag.get("n_xyz_match") or 0) == 0:
            reasons.append("CDDIS slrecc catalog（ecc_xyz/ecc_une/slrecc ILRS xyz/une）でも APOL 一次XYZは確認できない。")
        if int(cddis_resource_diag.get("n_files_scanned") or 0) > 0 and int(cddis_resource_diag.get("n_xyz_match") or 0) == 0:
            reasons.append("CDDIS products/resource（ILRS_DHF/SLRF_POS+VEL）でも APOL 一次XYZは確認できない。")
        if int(cddis_ac_diag.get("n_files_scanned") or 0) > 0 and int(cddis_ac_diag.get("n_xyz_match") or 0) == 0:
            reasons.append("CDDIS products/ac（AC descriptor/QC）でも APOL 一次XYZは確認できない。")
        if int(cddis_slrocc_diag.get("n_apol_rows") or 0) > 0 and int(cddis_slrocc_diag.get("n_apol_rows_with_xyz") or 0) == 0:
            reasons.append("CDDIS slrocc（slrcoor.txt）でも APOL 7045 行の X/Y/Z 列は NULL。")
        if int(cddis_slrocc_spl_diag.get("n_apol_rows") or 0) > 0 and int(cddis_slrocc_spl_diag.get("n_apol_rows_with_xyz") or 0) == 0:
            reasons.append("CDDIS slrocc（slrcoor.spl）でも APOL 7045 行の X/Y/Z 列は NULL。")
        if int(cddis_slrocc_spl_diag.get("n_apol_rows") or 0) == 0 and int(cddis_slrocc_spl_diag.get("n_table_rows") or 0) > 0:
            reasons.append("CDDIS slrocc（slrcoor.spl）でも APOL 7045 行自体が未掲載。")
        if int(cddis_slrocc_occ_diag.get("n_apol_rows") or 0) > 0 and int(cddis_slrocc_occ_diag.get("n_apol_rows_with_xyz") or 0) == 0:
            reasons.append("CDDIS slrocc（slrocc.txt）でも APOL 7045 occupation 行はあるが XYZ 列は未提供。")
        if int(cddis_slrocc_cal_diag.get("n_apol_rows") or 0) > 0 and int(cddis_slrocc_cal_diag.get("n_apol_rows_with_xyz") or 0) == 0:
            reasons.append("CDDIS slrocc（slrcal.txt）でも APOL 7045 calibration 行はあるが XYZ 列は未提供。")
        if int(cddis_slrocc_cal_spl_diag.get("n_apol_rows") or 0) > 0 and int(cddis_slrocc_cal_spl_diag.get("n_apol_rows_with_xyz") or 0) == 0:
            reasons.append("CDDIS slrocc（slrcal.spl）でも APOL 7045 calibration 行はあるが XYZ 列は未提供。")
        if int(cddis_slrocc_cal_spl_diag.get("n_apol_rows") or 0) == 0 and int(cddis_slrocc_cal_spl_diag.get("n_table_rows") or 0) > 0:
            reasons.append("CDDIS slrocc（slrcal.spl）でも APOL 7045 calibration 行自体が未掲載。")
        if int(cddis_slrocc_daily_diag.get("n_apol_rows") or 0) > 0 and int(cddis_slrocc_daily_diag.get("n_apol_rows_with_xyz") or 0) == 0:
            reasons.append("CDDIS slrocc 日次slrcoor（YYYY/slrcoor.YYMMDD）でも APOL 7045 行の X/Y/Z 列は NULL。")
        if int(ilrs_diag.get("n_urls_scanned") or 0) > 0 and int(ilrs_diag.get("n_xyz_match") or 0) == 0:
            reasons.append("ILRS primary products（DHF/SLRF/ecc notice経由）でも APOL 一次XYZは確認できない。")
        if int(ilrs_diag.get("n_site_rows_scanned") or 0) > 0 and int(ilrs_diag.get("n_alias_candidate") or 0) == 0:
            reasons.append(
                "ILRS primary SITE/ID の alias 照合（code/DOMES/CDP-SOD/name）で APOL 候補が 0 件。"
            )
        if int(ilrs_diag.get("n_alias_candidate") or 0) > 0 and int(ilrs_diag.get("n_alias_candidate_with_xyz") or 0) == 0:
            reasons.append("ILRS primary SITE/ID に APOL alias 候補はあるが、対応する STAX/STAY/STAZ が未確認。")
        if int(ilrs_diag.get("n_slrecc_login_blocked") or 0) > 0:
            reasons.append(
                f"CDDIS slrecc は Earthdata 認証未通過（login_blocked={int(ilrs_diag.get('n_slrecc_login_blocked') or 0)}）で、一次XYZの最終確認に未到達。"
            )
        if int(crd_diag.get("n_files_scanned") or 0) > 0 and int(crd_diag.get("n_with_xyz_marker") or 0) == 0:
            reasons.append("APOL-tag付きLLR CRDヘッダ（h2 APOL 7045）にも一次XYZ記載は確認できない。")
        next_actions = [
            "APOLの一次座標（ITRF XYZ）を含む公開ソースを追加取得する。",
            "ILRSAC ops archive は低優先の監視対象とし、CDDIS products/pos+eop（日次）と一次ソース更新を優先する。",
            "取得まで iers_station_coord_unified は watch を維持する。",
        ]
    else:
        decision = "watch"
        reasons = [
            "APOLの一次座標情報がローカル/リモートの監査範囲で取得できなかった。",
        ]
        next_actions = [
            "APOL site-log の再取得経路を点検する。",
            "取得まで iers_station_coord_unified は watch を維持する。",
        ]

    src_counts = df.groupby("source").size().to_dict()
    grp_counts = df.groupby("source_group").size().to_dict()
    n_xyz_by_source = df[df["has_xyz"] == True].groupby("source").size().to_dict()
    n_geod_by_source = df[df["has_geodetic"] == True].groupby("source").size().to_dict()
    n_xyz_by_group = df[df["has_xyz"] == True].groupby("source_group").size().to_dict()
    n_geod_by_group = df[df["has_geodetic"] == True].groupby("source_group").size().to_dict()
    merge_route = _build_deterministic_merge_route(
        df=df,
        apol_code=apol_code,
        apol_domes=apol_domes,
        apol_alias_tokens=apol_alias_tokens,
        ilrs_diag=ilrs_diag,
        earthdata_auth_diag=earthdata_auth_diag,
    )
    ilrsac_candidate_n = int(cddis_ilrsac_diag.get("n_candidate_files") or 0)
    ilrsac_scanned_n = int(cddis_ilrsac_diag.get("n_files_scanned") or 0)
    ilrsac_fetch_error_n = int(cddis_ilrsac_diag.get("n_fetch_error") or 0)
    ilrsac_coverage_complete = bool(
        ilrsac_candidate_n > 0 and ilrsac_scanned_n >= ilrsac_candidate_n and ilrsac_fetch_error_n == 0
    )
    if str(decision) == "pass":
        watch_lock_status = "not_applicable_pass"
    elif bool(ilrsac_coverage_complete) and str(merge_route.get("status") or "") == "watch":
        watch_lock_status = "fixed_watch"
    else:
        watch_lock_status = "pending_reaudit"
    watch_lock_basis = [
        f"decision={decision}",
        f"merge_route_status={str(merge_route.get('status') or '')}",
        f"ilrsac_scanned={ilrsac_scanned_n}/{ilrsac_candidate_n}",
        f"ilrsac_fetch_error={ilrsac_fetch_error_n}",
    ]
    watch_lock_release_conditions = [
        "いずれかの優先source_groupで APOL 一次XYZ（STAX/STAY/STAZ）が検出される。",
        "deterministic_merge_route.status が resolved に遷移する。",
        "resolved遷移後に llr_batch_eval --station-override-json で iers_station_coord_unified が pass になる。",
    ]
    resolved_transition_detected = bool(
        str(merge_route.get("status") or "") == "resolved" and isinstance(merge_route.get("selected_xyz"), dict)
    )
    resolved_transition_source_group = str(merge_route.get("resolved_source_group") or "").strip() or None
    resolved_transition_basis = [
        f"merge_route_status={str(merge_route.get('status') or '')}",
        f"selected_xyz_present={bool(isinstance(merge_route.get('selected_xyz'), dict))}",
        f"resolved_source_group={resolved_transition_source_group or ''}",
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    labels = [
        "local_cache",
        "edc_remote",
        "slrlog_oldlog",
        "cddis_slrecc_catalog",
        "cddis_resource_sinex",
        "cddis_ac_catalog",
        "cddis_slrocc_daily_catalog",
        "cddis_slrocc_spl_catalog",
        "cddis_slrocc_occupation_catalog",
        "cddis_slrocc_calibration_catalog",
        "cddis_slrocc_calibration_spl_catalog",
        "cddis_pos_eop_daily",
        "cddis_ilrsac_pos_eop",
        "cddis_slrlog",
        "cddis_slrlog_oldlog",
        "cddis_slrhst_history",
        "cddis_llr_crd",
        "cddis_slrocc_coordinate_catalog",
        "repro2020",
        "slrhst_history",
        "ilrs_primary_products",
        "llr_crd_header",
    ]
    xyz_vals = [int(n_xyz_by_group.get(v, 0)) for v in labels]
    geod_vals = [int(n_geod_by_group.get(v, 0)) for v in labels]
    xs = np.arange(len(labels))
    w = 0.35
    ax.bar(xs - w / 2, xyz_vals, width=w, color="#4c78a8", label="logs with XYZ")
    ax.bar(xs + w / 2, geod_vals, width=w, color="#f58518", label="logs with geodetic")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count")
    ax.set_title("APOL primary-coordinate route audit (site-log + resource + ac-descriptor + pos+eop + ilrsac-pos+eop + slrocc + slrocc-spl + slrocc-occ + slrocc-cal + slrocc-cal-spl + slrocc-daily + REPRO2020 + ILRS products)")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "watch_lock": {
            "status": watch_lock_status,
            "basis": watch_lock_basis,
            "ilrsac_coverage_complete": bool(ilrsac_coverage_complete),
            "release_conditions": watch_lock_release_conditions,
        },
        "resolved_transition_monitor": {
            "detected": bool(resolved_transition_detected),
            "source_group": resolved_transition_source_group,
            "basis": resolved_transition_basis,
        },
        "inputs": {
            "apol_metadata": str(APOL_META_PATH.relative_to(ROOT)).replace("\\", "/"),
            "local_station_dir": str(STATIONS_DIR.relative_to(ROOT)).replace("\\", "/"),
            "edc_slrlog_list": EDC_SLRLOG_LIST if not args.offline else None,
            "edc_slrlog_oldlog_list": EDC_SLRLOG_OLDLOG_LIST if not args.offline and int(args.max_oldlog_logs) > 0 else None,
            "edc_slrhst_list": EDC_SLRHST_LIST if not args.offline and int(args.max_slrhst_logs) > 0 else None,
            "cddis_slrlog_list": CDDIS_SLRLOG_LIST if not args.offline else None,
            "cddis_slrlog_oldlog_list": CDDIS_SLRLOG_OLDLOG_LIST if not args.offline and int(args.max_oldlog_logs) > 0 else None,
            "cddis_slrhst_list": CDDIS_SLRHST_LIST if not args.offline and int(args.max_slrhst_logs) > 0 else None,
            "cddis_slrecc_list": CDDIS_SLRECC_LIST if not args.offline and int(args.max_cddis_slrecc_files) > 0 else None,
            "cddis_llr_npt_crd_root": CDDIS_LLR_NPT_CRD_ROOT if not args.offline and int(args.max_cddis_crd_files) > 0 else None,
            "cddis_llr_fr_crd_root": CDDIS_LLR_FR_CRD_ROOT if not args.offline and int(args.max_cddis_crd_files) > 0 else None,
            "cddis_products_pos_eop_root": CDDIS_PRODUCTS_POS_EOP_ROOT if not args.offline and int(args.max_cddis_pos_eop_days) > 0 else None,
            "cddis_ilrsac_products_ops_pos_eop_root": CDDIS_ILRSAC_PRODUCTS_OPS_POS_EOP_ROOT if not args.offline and int(args.max_cddis_ilrsac_files) > 0 else None,
            "cddis_products_resource_root": CDDIS_PRODUCTS_RESOURCE_ROOT if not args.offline and int(args.max_cddis_resource_files) > 0 else None,
            "cddis_products_ac_root": CDDIS_PRODUCTS_AC_ROOT if not args.offline and int(args.max_cddis_ac_files) > 0 else None,
            "cddis_products_ac_archive_root": CDDIS_PRODUCTS_AC_ARCHIVE_ROOT if not args.offline and int(args.max_cddis_ac_files) > 0 else None,
            "cddis_slrocc_slrcoor_txt": CDDIS_SLROCC_SLRCOOR_TXT if not args.offline else None,
            "cddis_slrocc_slrcoor_spl": CDDIS_SLROCC_SLRCOOR_SPL if not args.offline else None,
            "cddis_slrocc_occ_txt": CDDIS_SLROCC_OCC_TXT if not args.offline else None,
            "cddis_slrocc_slrcal_txt": CDDIS_SLROCC_SLRCAL_TXT if not args.offline else None,
            "cddis_slrocc_slrcal_spl": CDDIS_SLROCC_SLRCAL_SPL if not args.offline else None,
            "cddis_slrocc_root": CDDIS_SLROCC_ROOT if not args.offline and int(args.max_cddis_slrocc_files) > 0 else None,
            "edc_slrmail_index": EDC_SLRMAIL_INDEX if not args.offline and int(args.max_ilrs_slrmail_notices) > 0 else None,
            "repro2020_root": REPRO_ROOT if not args.offline and int(args.max_repro_centers) > 0 else None,
            "offline": bool(args.offline),
            "max_remote_logs": int(args.max_remote_logs),
            "max_oldlog_logs": int(args.max_oldlog_logs),
            "max_repro_centers": int(args.max_repro_centers),
            "max_slrhst_logs": int(args.max_slrhst_logs),
            "max_ilrs_slrmail_notices": int(args.max_ilrs_slrmail_notices),
            "max_crd_header_files": int(args.max_crd_header_files),
            "max_cddis_crd_files": int(args.max_cddis_crd_files),
            "max_cddis_pos_eop_days": int(args.max_cddis_pos_eop_days),
            "max_cddis_ilrsac_files": int(args.max_cddis_ilrsac_files),
            "max_cddis_slrecc_files": int(args.max_cddis_slrecc_files),
            "max_cddis_resource_files": int(args.max_cddis_resource_files),
            "max_cddis_ac_files": int(args.max_cddis_ac_files),
            "max_cddis_slrocc_years": int(args.max_cddis_slrocc_years),
            "max_cddis_slrocc_files": int(args.max_cddis_slrocc_files),
            "apol_alias_tokens": apol_alias_tokens,
            "earthdata_auth_mode": str(args.earthdata_auth_mode),
            "earthdata_auth": earthdata_auth_diag,
        },
        "metrics": {
            "n_rows_total": int(len(df)),
            "n_logs_with_xyz": int(len(xyz_rows)),
            "n_logs_with_xyz_any_source": int(len(xyz_rows)),
            "n_logs_with_geodetic": int(len(geod_rows)),
            "latest_log_with_xyz": latest_xyz,
            "latest_log_with_geodetic": latest_geod,
            "source_counts": {str(k): int(v) for k, v in src_counts.items()},
            "source_xyz_counts": {str(k): int(v) for k, v in n_xyz_by_source.items()},
            "source_geodetic_counts": {str(k): int(v) for k, v in n_geod_by_source.items()},
            "source_group_counts": {str(k): int(v) for k, v in grp_counts.items()},
            "source_group_xyz_counts": {str(k): int(v) for k, v in n_xyz_by_group.items()},
            "source_group_geodetic_counts": {str(k): int(v) for k, v in n_geod_by_group.items()},
            "remote_logs_available_total": int(remote_total),
            "remote_scan_error": remote_error,
            "oldlog_logs_available_total": int(oldlog_total),
            "oldlog_scan_error": oldlog_error,
            "slrhst_logs_available_total": int(slrhst_total),
            "slrhst_scan_error": slrhst_error,
            "cddis_site_logs": cddis_diag,
            "cddis_llr_crd_headers": cddis_llr_crd_diag,
            "cddis_pos_eop_daily": cddis_pos_eop_diag,
            "cddis_ilrsac_pos_eop": cddis_ilrsac_diag,
            "cddis_slrecc_catalog": cddis_slrecc_diag,
            "cddis_resource_sinex": cddis_resource_diag,
            "cddis_ac_catalog": cddis_ac_diag,
            "cddis_slrocc_coordinate_catalog": cddis_slrocc_diag,
            "cddis_slrocc_spl_catalog": cddis_slrocc_spl_diag,
            "cddis_slrocc_occupation_catalog": cddis_slrocc_occ_diag,
            "cddis_slrocc_calibration_catalog": cddis_slrocc_cal_diag,
            "cddis_slrocc_calibration_spl_catalog": cddis_slrocc_cal_spl_diag,
            "cddis_slrocc_daily_catalog": cddis_slrocc_daily_diag,
            "apol_code": apol_code,
            "apol_domes": apol_domes,
            "apol_alias_tokens": apol_alias_tokens,
            "apol_xyz_available_any_source": bool(has_primary_xyz),
            "apol_meta_has_xyz": bool(
                apol_meta.get("x_m") is not None and apol_meta.get("y_m") is not None and apol_meta.get("z_m") is not None
            ),
            "apol_meta_has_geodetic": bool(
                apol_meta.get("lat_deg") is not None
                and apol_meta.get("lon_deg") is not None
                and apol_meta.get("height_m") is not None
            ),
            "apol_deterministic_merge_route_status": merge_route.get("status"),
            "apol_deterministic_merge_route_resolved_source_group": merge_route.get("resolved_source_group"),
            "resolved_transition_detected": bool(resolved_transition_detected),
            "watch_lock_status": watch_lock_status,
            "ilrsac_coverage_complete": bool(ilrsac_coverage_complete),
            "repro2020": repro_diag,
            "cddis_site_logs": cddis_diag,
            "cddis_llr_crd_headers": cddis_llr_crd_diag,
            "cddis_pos_eop_daily": cddis_pos_eop_diag,
            "cddis_ilrsac_pos_eop": cddis_ilrsac_diag,
            "cddis_slrecc_catalog": cddis_slrecc_diag,
            "cddis_resource_sinex": cddis_resource_diag,
            "cddis_ac_catalog": cddis_ac_diag,
            "cddis_slrocc_coordinate_catalog": cddis_slrocc_diag,
            "cddis_slrocc_spl_catalog": cddis_slrocc_spl_diag,
            "cddis_slrocc_occupation_catalog": cddis_slrocc_occ_diag,
            "cddis_slrocc_calibration_catalog": cddis_slrocc_cal_diag,
            "cddis_slrocc_calibration_spl_catalog": cddis_slrocc_cal_spl_diag,
            "cddis_slrocc_daily_catalog": cddis_slrocc_daily_diag,
            "ilrs_primary_products": ilrs_diag,
            "llr_crd_header": crd_diag,
        },
        "reasons": reasons,
        "recommended_next": next_actions,
        "deterministic_merge_route": merge_route,
        "artifacts": {
            "csv": str(OUT_CSV.relative_to(ROOT)).replace("\\", "/"),
            "png": str(OUT_PNG.relative_to(ROOT)).replace("\\", "/"),
            "merge_route_json": str(OUT_MERGE_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MERGE_JSON.write_text(json.dumps(merge_route, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "decision": decision,
                "n_rows_total": int(len(df)),
                "n_logs_with_xyz": int(len(xyz_rows)),
                "n_logs_with_xyz_any_source": int(len(xyz_rows)),
                "n_logs_with_geodetic": int(len(geod_rows)),
                "latest_log_with_xyz": latest_xyz,
                "latest_log_with_geodetic": latest_geod,
                "oldlog_logs_available_total": int(oldlog_total),
                "slrhst_logs_available_total": int(slrhst_total),
                "cddis_current_scanned": int(cddis_diag.get("n_current_scanned") or 0),
                "cddis_oldlog_scanned": int(cddis_diag.get("n_oldlog_scanned") or 0),
                "cddis_slrhst_scanned": int(cddis_diag.get("n_slrhst_scanned") or 0),
                "cddis_logs_with_xyz": int(cddis_diag.get("n_logs_with_xyz") or 0),
                "cddis_llr_crd_scanned": int(cddis_llr_crd_diag.get("n_files_scanned") or 0),
                "cddis_llr_crd_xyz_hits": int(cddis_llr_crd_diag.get("n_with_xyz_marker") or 0),
                "cddis_pos_eop_scanned": int(cddis_pos_eop_diag.get("n_files_scanned") or 0),
                "cddis_pos_eop_xyz_match": int(cddis_pos_eop_diag.get("n_xyz_match") or 0),
                "cddis_ilrsac_scanned": int(cddis_ilrsac_diag.get("n_files_scanned") or 0),
                "cddis_ilrsac_xyz_match": int(cddis_ilrsac_diag.get("n_xyz_match") or 0),
                "cddis_slrecc_scanned": int(cddis_slrecc_diag.get("n_files_scanned") or 0),
                "cddis_slrecc_xyz_match": int(cddis_slrecc_diag.get("n_xyz_match") or 0),
                "cddis_resource_scanned": int(cddis_resource_diag.get("n_files_scanned") or 0),
                "cddis_resource_xyz_match": int(cddis_resource_diag.get("n_xyz_match") or 0),
                "cddis_ac_scanned": int(cddis_ac_diag.get("n_files_scanned") or 0),
                "cddis_ac_xyz_match": int(cddis_ac_diag.get("n_xyz_match") or 0),
                "cddis_ac_code_hit": int(cddis_ac_diag.get("n_code_hit") or 0),
                "cddis_slrocc_apol_rows": int(cddis_slrocc_diag.get("n_apol_rows") or 0),
                "cddis_slrocc_apol_xyz_rows": int(cddis_slrocc_diag.get("n_apol_rows_with_xyz") or 0),
                "cddis_slrocc_spl_apol_rows": int(cddis_slrocc_spl_diag.get("n_apol_rows") or 0),
                "cddis_slrocc_spl_apol_xyz_rows": int(cddis_slrocc_spl_diag.get("n_apol_rows_with_xyz") or 0),
                "cddis_slrocc_occ_apol_rows": int(cddis_slrocc_occ_diag.get("n_apol_rows") or 0),
                "cddis_slrocc_occ_apol_xyz_rows": int(cddis_slrocc_occ_diag.get("n_apol_rows_with_xyz") or 0),
                "cddis_slrocc_cal_apol_rows": int(cddis_slrocc_cal_diag.get("n_apol_rows") or 0),
                "cddis_slrocc_cal_apol_xyz_rows": int(cddis_slrocc_cal_diag.get("n_apol_rows_with_xyz") or 0),
                "cddis_slrocc_cal_spl_apol_rows": int(cddis_slrocc_cal_spl_diag.get("n_apol_rows") or 0),
                "cddis_slrocc_cal_spl_apol_xyz_rows": int(cddis_slrocc_cal_spl_diag.get("n_apol_rows_with_xyz") or 0),
                "cddis_slrocc_daily_scanned": int(cddis_slrocc_daily_diag.get("n_files_scanned") or 0),
                "cddis_slrocc_daily_apol_rows": int(cddis_slrocc_daily_diag.get("n_apol_rows") or 0),
                "cddis_slrocc_daily_apol_xyz_rows": int(cddis_slrocc_daily_diag.get("n_apol_rows_with_xyz") or 0),
                "ilrs_primary_urls_scanned": int(ilrs_diag.get("n_urls_scanned") or 0),
                "ilrs_primary_xyz_match": int(ilrs_diag.get("n_xyz_match") or 0),
                "crd_header_files_scanned": int(crd_diag.get("n_files_scanned") or 0),
                "crd_header_pad_code_hits": int(crd_diag.get("n_with_pad_code") or 0),
                "crd_header_xyz_marker_hits": int(crd_diag.get("n_with_xyz_marker") or 0),
                "repro2020_n_files_scanned": int(repro_diag.get("n_files_scanned") or 0),
                "repro2020_n_xyz_match": int(repro_diag.get("n_xyz_match") or 0),
                "earthdata_auth_status": str(earthdata_auth_diag.get("status") or "unknown"),
                "ilrs_slrecc_login_blocked": int(ilrs_diag.get("n_slrecc_login_blocked") or 0),
                "merge_route_status": merge_route.get("status"),
                "merge_route_resolved_source_group": merge_route.get("resolved_source_group"),
                "resolved_transition_detected": bool(resolved_transition_detected),
                "watch_lock_status": watch_lock_status,
                "ilrsac_coverage_complete": bool(ilrsac_coverage_complete),
                "cddis_ilrsac_candidates": ilrsac_candidate_n,
                "cddis_ilrsac_files_scanned": ilrsac_scanned_n,
            },
            ensure_ascii=False,
        )
    )
    print(f"[ok] {OUT_JSON.relative_to(ROOT)}")
    print(f"[ok] {OUT_CSV.relative_to(ROOT)}")
    print(f"[ok] {OUT_PNG.relative_to(ROOT)}")
    print(f"[ok] {OUT_MERGE_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
