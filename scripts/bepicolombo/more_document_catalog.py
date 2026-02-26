#!/usr/bin/env python3
"""
BepiColombo / MORE (Mercury Orbiter Radio-science Experiment)

PSA の `bc_mpo_more/document/` にある PDS4 Document（*.lblx / *.pdf）を
オフラインで索引化し、一般向けレポートで参照できる JSON/CSV を生成する。

入力（キャッシュ）:
- data/bepicolombo/psa_more/document/*.lblx
- data/bepicolombo/psa_more/document/*.pdf（任意, --download-pdfs で取得済み想定）

出力:
- output/private/bepicolombo/more_document_catalog.json
- output/private/bepicolombo/more_document_catalog.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_as_int` の入出力契約と処理意図を定義する。

def _as_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


# 関数: `_parse_lblx` の入出力契約と処理意図を定義する。

def _parse_lblx(lblx_path: Path) -> Dict[str, Any]:
    ns = {"p": "http://pds.nasa.gov/pds4/pds/v1"}
    tree = ET.parse(lblx_path)
    root = tree.getroot()

    # 関数: `t` の入出力契約と処理意図を定義する。
    def t(path: str, default: str = "") -> str:
        v = root.findtext(path, default=default, namespaces=ns)
        return str(v).strip()

    keywords: List[str] = []
    for kw in root.findall("p:Identification_Area/p:Citation_Information/p:keyword", namespaces=ns):
        # 条件分岐: `kw.text` を満たす経路を評価する。
        if kw.text:
            s = str(kw.text).strip()
            # 条件分岐: `s` を満たす経路を評価する。
            if s:
                keywords.append(s)

    file_name = t("p:Document/p:Document_Edition/p:Document_File/p:file_name")
    file_size_b = t("p:Document/p:Document_Edition/p:Document_File/p:file_size")
    md5 = t("p:Document/p:Document_Edition/p:Document_File/p:md5_checksum")
    pdf_path = (lblx_path.parent / file_name) if file_name else None

    return {
        "logical_identifier": t("p:Identification_Area/p:logical_identifier"),
        "version_id": t("p:Identification_Area/p:version_id"),
        "title": t("p:Identification_Area/p:title"),
        "publication_year": _as_int(t("p:Identification_Area/p:Citation_Information/p:publication_year")) or None,
        "keywords": keywords,
        "citation_description": t("p:Identification_Area/p:Citation_Information/p:description"),
        "modification_date": t(
            "p:Identification_Area/p:Modification_History/p:Modification_Detail/p:modification_date"
        ),
        "document_name": t("p:Document/p:document_name"),
        "revision_id": t("p:Document/p:revision_id"),
        "publication_date": t("p:Document/p:publication_date"),
        "file_name": file_name,
        "file_size_bytes": _as_int(file_size_b) if file_size_b else None,
        "md5": md5 or None,
        "lblx_path": str(lblx_path.resolve()),
        "pdf_path": str(pdf_path.resolve()) if (pdf_path and pdf_path.exists()) else None,
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    parser = argparse.ArgumentParser(description="Index PSA bc_mpo_more document/*.lblx into JSON/CSV (offline)")
    parser.add_argument(
        "--doc-dir",
        type=str,
        default="data/bepicolombo/psa_more/document",
        help="Local cache directory containing *.lblx/*.pdf. Default: data/bepicolombo/psa_more/document",
    )
    args = parser.parse_args()

    root = _ROOT
    doc_dir = (root / Path(args.doc_dir)).resolve()
    out_dir = root / "output" / "private" / "bepicolombo"
    out_dir.mkdir(parents=True, exist_ok=True)

    lblx_files = sorted(doc_dir.glob("*.lblx"))
    docs: List[Dict[str, Any]] = []
    for p in lblx_files:
        try:
            docs.append(_parse_lblx(p))
        except Exception as e:
            docs.append(
                {
                    "lblx_path": str(p.resolve()),
                    "error": str(e),
                }
            )

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now_iso(),
        "doc_dir": str(doc_dir),
        "document_count": int(len(docs)),
        "documents": docs,
    }

    json_path = out_dir / "more_document_catalog.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = out_dir / "more_document_catalog.csv"
    headers = [
        "logical_identifier",
        "version_id",
        "title",
        "publication_year",
        "document_name",
        "revision_id",
        "publication_date",
        "file_name",
        "file_size_bytes",
        "md5",
        "pdf_path",
        "lblx_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for d in docs:
            row = {k: d.get(k) for k in headers}
            w.writerow(row)

    try:
        worklog.append_event(
            {
                "event_type": "bepicolombo_more_document_catalog",
                "argv": sys.argv,
                "inputs": {"doc_dir": doc_dir},
                "outputs": {"json": json_path, "csv": csv_path},
            }
        )
    except Exception:
        pass

    print("Wrote:", json_path)
    print("Wrote:", csv_path)


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
