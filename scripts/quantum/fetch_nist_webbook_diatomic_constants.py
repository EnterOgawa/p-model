from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


# 関数: `_sanitize_token` の入出力契約と処理意図を定義する。

def _sanitize_token(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "unknown"


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0` を満たす経路を評価する。
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return

    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


# クラス: `_HTMLTableParser` の責務と境界条件を定義する。

class _HTMLTableParser(HTMLParser):
    # 関数: `__init__` の入出力契約と処理意図を定義する。
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._in_tr = False
        self._in_cell = False
        self._cur_row: list[str] = []
        self._cur_cell_parts: list[str] = []

    # 関数: `handle_starttag` の入出力契約と処理意図を定義する。

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # 条件分岐: `tag.lower() == "tr"` を満たす経路を評価する。
        if tag.lower() == "tr":
            self._in_tr = True
            self._cur_row = []
            return

        # 条件分岐: `tag.lower() in ("td", "th") and self._in_tr` を満たす経路を評価する。

        if tag.lower() in ("td", "th") and self._in_tr:
            self._in_cell = True
            self._cur_cell_parts = []

    # 関数: `handle_endtag` の入出力契約と処理意図を定義する。

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        # 条件分岐: `t in ("td", "th") and self._in_tr and self._in_cell` を満たす経路を評価する。
        if t in ("td", "th") and self._in_tr and self._in_cell:
            txt = "".join(self._cur_cell_parts)
            txt = txt.replace("\u00a0", " ")  # &nbsp;
            txt = re.sub(r"\s+", " ", txt).strip()
            self._cur_row.append(txt)
            self._cur_cell_parts = []
            self._in_cell = False
            return

        # 条件分岐: `t == "tr" and self._in_tr` を満たす経路を評価する。

        if t == "tr" and self._in_tr:
            # 条件分岐: `self._cur_row` を満たす経路を評価する。
            if self._cur_row:
                self.rows.append(self._cur_row)

            self._cur_row = []
            self._in_tr = False

    # 関数: `handle_data` の入出力契約と処理意図を定義する。

    def handle_data(self, data: str) -> None:
        # 条件分岐: `self._in_tr and self._in_cell` を満たす経路を評価する。
        if self._in_tr and self._in_cell:
            self._cur_cell_parts.append(data)


# 関数: `_safe_float_from_cell` の入出力契約と処理意図を定義する。

def _safe_float_from_cell(s: str) -> Optional[float]:
    ss = str(s).strip()
    # 条件分岐: `not ss` を満たす経路を評価する。
    if not ss:
        return None

    m = re.search(r"[-+]?\d+(?:\.\d+)?", ss)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return None

    try:
        return float(m.group(0))
    except Exception:
        return None


# クラス: `DiatomicConstants` の責務と境界条件を定義する。

@dataclass(frozen=True)
class DiatomicConstants:
    state: str
    Te_cm_inv: Optional[float]
    omega_e_cm_inv: Optional[float]
    omega_exe_cm_inv: Optional[float]
    omega_eye_cm_inv: Optional[float]
    Be_cm_inv: Optional[float]
    alpha_e_cm_inv: Optional[float]
    gamma_e_cm_inv: Optional[float]
    De_cm_inv: Optional[float]
    beta_e_cm_inv: Optional[float]
    re_A: Optional[float]
    trans: str
    nu00: str

    # 関数: `from_row` の入出力契約と処理意図を定義する。
    @classmethod
    def from_row(cls, row: list[str]) -> "DiatomicConstants":
        # 条件分岐: `len(row) < 13` を満たす経路を評価する。
        if len(row) < 13:
            raise ValueError(f"expected 13 columns; got {len(row)}")

        return cls(
            state=row[0].strip(),
            Te_cm_inv=_safe_float_from_cell(row[1]),
            omega_e_cm_inv=_safe_float_from_cell(row[2]),
            omega_exe_cm_inv=_safe_float_from_cell(row[3]),
            omega_eye_cm_inv=_safe_float_from_cell(row[4]),
            Be_cm_inv=_safe_float_from_cell(row[5]),
            alpha_e_cm_inv=_safe_float_from_cell(row[6]),
            gamma_e_cm_inv=_safe_float_from_cell(row[7]),
            De_cm_inv=_safe_float_from_cell(row[8]),
            beta_e_cm_inv=_safe_float_from_cell(row[9]),
            re_A=_safe_float_from_cell(row[10]),
            trans=row[11].strip(),
            nu00=row[12].strip(),
        )


# 関数: `_extract_first_diatomic_table` の入出力契約と処理意図を定義する。

def _extract_first_diatomic_table(html: str) -> str:
    m = re.search(r'<table class="small data"><caption[^>]*>\s*Diatomic constants', html, re.IGNORECASE)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError("Diatomic constants table not found in HTML")

    start = m.start()
    end = html.find("</table>", start)
    # 条件分岐: `end < 0` を満たす経路を評価する。
    if end < 0:
        raise ValueError("Diatomic constants table end </table> not found")

    return html[start : end + len("</table>")]


# 関数: `_pick_ground_state` の入出力契約と処理意図を定義する。

def _pick_ground_state(rows: list[list[str]]) -> DiatomicConstants:
    # Prefer: State starts with "X" and Te==0 (ground electronic state).
    for r in rows:
        # 条件分岐: `len(r) < 13` を満たす経路を評価する。
        if len(r) < 13:
            continue

        state = (r[0] or "").strip()
        # 条件分岐: `not state.startswith("X")` を満たす経路を評価する。
        if not state.startswith("X"):
            continue

        te = _safe_float_from_cell(r[1] or "")
        # 条件分岐: `te is None or abs(te) > 1e-12` を満たす経路を評価する。
        if te is None or abs(te) > 1e-12:
            continue

        return DiatomicConstants.from_row(r)

    raise ValueError("ground state row not found (expected State='X ...' and Te=0)")


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch NIST Chemistry WebBook diatomic constants table and cache a baseline.")
    ap.add_argument(
        "--id",
        default="C1333740",
        help="NIST Chemistry WebBook ID (e.g., C1333740 for H2).",
    )
    ap.add_argument(
        "--slug",
        default="h2",
        help="Cache folder token (saved under data/quantum/sources/nist_webbook_diatomic_<slug>/).",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: do not download; only validate local cache files exist.",
    )
    args = ap.parse_args(argv)

    root = _repo_root()
    slug = _sanitize_token(args.slug)
    webbook_id = str(args.id).strip()

    out_dir = root / "data" / "quantum" / "sources" / f"nist_webbook_diatomic_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID={webbook_id}&Units=SI&Mask=1000"
    raw_path = out_dir / f"nist_webbook_diatomic_constants__{slug}__{_sanitize_token(webbook_id)}.html"
    extracted_path = out_dir / "extracted_values.json"
    manifest_path = out_dir / "manifest.json"

    # 条件分岐: `args.offline` を満たす経路を評価する。
    if args.offline:
        missing: list[Path] = []
        for p in (raw_path, extracted_path, manifest_path):
            # 条件分岐: `not p.exists() or p.stat().st_size <= 0` を満たす経路を評価する。
            if not p.exists() or p.stat().st_size <= 0:
                missing.append(p)

        # 条件分岐: `missing` を満たす経路を評価する。

        if missing:
            raise SystemExit("[fail] missing cache files:\n" + "\n".join(f"- {p}" for p in missing))

        print("[ok] offline check passed")
        return 0

    _download(url, raw_path)

    html = raw_path.read_text(encoding="utf-8", errors="replace")
    table_html = _extract_first_diatomic_table(html)

    parser = _HTMLTableParser()
    parser.feed(table_html)

    constants = _pick_ground_state(parser.rows)

    extracted: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "NIST Chemistry WebBook: Constants of diatomic molecules (Huber & Herzberg compilation)",
        "webbook_id": webbook_id,
        "query_url": url,
        "raw_file": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "selected": {
            "slug": slug,
            "state": constants.state,
            "Te_cm^-1": constants.Te_cm_inv,
            "omega_e_cm^-1": constants.omega_e_cm_inv,
            "omega_e_x_e_cm^-1": constants.omega_exe_cm_inv,
            "omega_e_y_e_cm^-1": constants.omega_eye_cm_inv,
            "B_e_cm^-1": constants.Be_cm_inv,
            "alpha_e_cm^-1": constants.alpha_e_cm_inv,
            "gamma_e_cm^-1": constants.gamma_e_cm_inv,
            "D_e_cm^-1": constants.De_cm_inv,
            "beta_e_cm^-1": constants.beta_e_cm_inv,
            "r_e_A": constants.re_A,
            "trans": constants.trans,
            "nu00": constants.nu00,
        },
        "notes": [
            "This cache is used to fix a small, offline-stable baseline for Phase 7 / Step 7.12 (molecular constants).",
            "The WebBook table contains many electronic states; we pick the ground state by (State starts with 'X' and Te=0).",
            "Do not treat this as a derivation; it fixes target values from a primary source for future tests.",
        ],
        "table_columns": [
            "State",
            "Te",
            "omega_e",
            "omega_e x_e",
            "omega_e y_e",
            "B_e",
            "alpha_e",
            "gamma_e",
            "D_e",
            "beta_e",
            "r_e",
            "Trans",
            "nu_00",
        ],
        "units": {
            "Te": "cm^-1",
            "omega_e": "cm^-1",
            "omega_e x_e": "cm^-1",
            "omega_e y_e": "cm^-1",
            "B_e": "cm^-1",
            "alpha_e": "cm^-1",
            "gamma_e": "cm^-1",
            "D_e": "cm^-1",
            "beta_e": "cm^-1",
            "r_e": "Å",
        },
    }
    extracted_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] extracted: {extracted_path}")

    manifest = {
        "generated_utc": extracted["generated_utc"],
        "dataset": f"NIST WebBook diatomic constants: {webbook_id}",
        "notes": [
            "Cache for offline-repro analysis in Phase 7 / Step 7.12 (molecular baseline).",
            "If you change the selection rule or pick a different state, update the manuscript and regenerate paper outputs.",
        ],
        "files": [
            {
                "url": url,
                "path": str(raw_path),
                "bytes": int(raw_path.stat().st_size),
                "sha256": _sha256(raw_path),
            },
            {
                "url": None,
                "path": str(extracted_path),
                "bytes": int(extracted_path.stat().st_size),
                "sha256": _sha256(extracted_path),
                "derived_from": [raw_path.name],
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest: {manifest_path}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

