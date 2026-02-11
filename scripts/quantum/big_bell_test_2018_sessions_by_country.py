from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET


_NS_MAIN = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class CountrySessions:
    country: str
    sessions: int


def _cellref_to_rowcol(cellref: str) -> tuple[int, int]:
    col = ""
    row = ""
    for ch in cellref:
        if ch.isalpha():
            col += ch
        else:
            row += ch
    c = 0
    for ch in col.upper():
        c = c * 26 + (ord(ch) - 64)
    return int(row), c


def _load_shared_strings(z: zipfile.ZipFile) -> List[str]:
    ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
    out: List[str] = []
    for si in ss.findall("m:si", _NS_MAIN):
        # shared strings can contain multiple <t> (with formatting). Concatenate.
        parts = []
        for t in si.findall(".//m:t", _NS_MAIN):
            if t.text:
                parts.append(t.text)
        out.append("".join(parts))
    return out


def _sheet_cells(z: zipfile.ZipFile, *, sheet_xml: str, shared_strings: List[str]) -> Dict[tuple[int, int], str]:
    root = ET.fromstring(z.read(sheet_xml))
    cells: Dict[tuple[int, int], str] = {}
    for c in root.findall(".//m:c", _NS_MAIN):
        cellref = c.attrib.get("r")
        if not cellref:
            continue
        v = c.find("m:v", _NS_MAIN)
        if v is None or v.text is None:
            continue
        raw = v.text
        t = c.attrib.get("t")
        if t == "s":
            try:
                val = shared_strings[int(raw)]
            except Exception:
                val = ""
        else:
            val = raw
        r, col = _cellref_to_rowcol(cellref)
        cells[(r, col)] = val
    return cells


def _load_sessions_by_country(xlsx_path: Path) -> List[CountrySessions]:
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)
    with zipfile.ZipFile(xlsx_path, "r") as z:
        shared = _load_shared_strings(z)
        # In the published file, Panel(a)SessionsByCountry is sheet2.xml.
        cells = _sheet_cells(z, sheet_xml="xl/worksheets/sheet2.xml", shared_strings=shared)

    out: List[CountrySessions] = []
    row = 2
    while True:
        country = (cells.get((row, 1)) or "").strip()
        sessions_s = (cells.get((row, 2)) or "").strip()
        if not country and not sessions_s:
            break
        if country and sessions_s:
            try:
                sessions = int(float(sessions_s))
            except Exception:
                sessions = 0
            out.append(CountrySessions(country=country, sessions=sessions))
        row += 1
        if row > 100_000:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Big Bell Test 2018 source data: sessions by country (Fig.2a).")
    ap.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="Path to source_data.xlsx (default: data/quantum/sources/big_bell_test_2018/source_data.xlsx).",
    )
    ap.add_argument("--top-n", type=int, default=20, help="How many top countries to show in the plot (default: 20).")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    xlsx = args.xlsx or (root / "data" / "quantum" / "sources" / "big_bell_test_2018" / "source_data.xlsx")
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_sessions_by_country(xlsx)
    rows_sorted = sorted(rows, key=lambda r: int(r.sessions), reverse=True)
    total_sessions = int(sum(int(r.sessions) for r in rows_sorted))

    out_csv = out_dir / "big_bell_test_2018_sessions_by_country.csv"
    out_png = out_dir / "big_bell_test_2018_sessions_by_country.png"
    out_json = out_dir / "big_bell_test_2018_sessions_by_country_metrics.json"

    # CSV
    lines = ["country,sessions"]
    for r in rows_sorted:
        country = r.country.replace('"', '""')
        lines.append(f"\"{country}\",{int(r.sessions)}")
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Plot (top N)
    import matplotlib.pyplot as plt

    top_n = max(1, int(args.top_n))
    top = rows_sorted[:top_n]
    labels = [r.country for r in reversed(top)]
    vals = [int(r.sessions) for r in reversed(top)]

    fig, ax = plt.subplots(figsize=(10.8, 6.8), dpi=150)
    ax.barh(labels, vals, color="#1f77b4", alpha=0.85)
    ax.set_xlabel("sessions (Google Analytics)")
    ax.set_title(f"Big Bell Test 2018 (Nature): sessions by country (Fig.2a) — total={total_sessions:,}")
    ax.grid(True, axis="x", ls=":", lw=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": {
            "source": "Big Bell Test 2018 (Nature) source data (Fig.2a sessions by country)",
            "xlsx": str(xlsx),
        },
        "summary": {"n_countries": int(len(rows_sorted)), "total_sessions": total_sessions, "top_n": int(top_n)},
        "top": [{"country": r.country, "sessions": int(r.sessions)} for r in top],
        "outputs": {"png": str(out_png), "csv": str(out_csv)},
        "notes": [
            "This is not a reanalysis of Bell inequality statistics; it summarizes the published Source Data for Fig.2(a)."
        ],
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()

