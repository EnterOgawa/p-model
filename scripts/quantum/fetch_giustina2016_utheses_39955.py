from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url: str
    out_name: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _http_get_bytes(url: str, *, timeout_s: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _http_head_status(url: str, *, timeout_s: int = 30) -> int | None:
    try:
        req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"}, method="HEAD")
        with urlopen(req, timeout=timeout_s) as resp:
            return int(getattr(resp, "status", 200))
    except HTTPError as e:
        try:
            return int(e.code)
        except Exception:
            return None
    except (URLError, TimeoutError):
        return None


def _fetch_with_retries(url: str, *, timeout_s: int = 60, attempts: int = 4, sleep_s: float = 1.0) -> bytes:
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return _http_get_bytes(url, timeout_s=timeout_s)
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            if i >= attempts:
                break
            time.sleep(sleep_s * i)
    raise RuntimeError(f"fetch failed after {attempts} attempts: {url} ({type(last_err).__name__}: {last_err})")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_utheses_doc(solr: dict[str, Any]) -> dict[str, Any]:
    resp = solr.get("response") if isinstance(solr.get("response"), dict) else {}
    docs = resp.get("docs") if isinstance(resp.get("docs"), list) else []
    doc = docs[0] if docs and isinstance(docs[0], dict) else {}
    return doc


def _extract_phaidra_info(obj: dict[str, Any]) -> dict[str, Any]:
    info = obj.get("info") if isinstance(obj.get("info"), dict) else {}
    return info


def _extract_phaidra_jsonld_summary(obj: dict[str, Any]) -> dict[str, Any]:
    md = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    jsonld = md.get("JSON-LD") if isinstance(md.get("JSON-LD"), dict) else {}
    if not jsonld:
        return {}
    return {
        "ebucore:filename": jsonld.get("ebucore:filename"),
        "ebucore:hasMimeType": jsonld.get("ebucore:hasMimeType"),
        "schema:numberOfPages": jsonld.get("schema:numberOfPages"),
        "edm:rights": jsonld.get("edm:rights"),
        "dce:title": jsonld.get("dce:title"),
        "dcterms:type": jsonld.get("dcterms:type"),
    }


def _safe_pid_token(pid: str) -> str:
    # e.g. "o:1331601" -> "o1331601" for filenames
    return pid.replace(":", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="Do not fetch; use cached snapshots.")
    ap.add_argument("--refresh", action="store_true", help="Re-fetch snapshots even if cached files exist.")
    args = ap.parse_args()

    src_dir = ROOT / "data" / "quantum" / "sources" / "giustina2016_utheses_39955"
    out_dir = ROOT / "output" / "quantum" / "bell"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        SourceSpec(
            name="phaidra_object_info_o1331600",
            url="https://phaidra.univie.ac.at/api/object/o:1331600/info",
            out_name="phaidra_o1331600_info.json",
        ),
        SourceSpec(
            name="utheses_solr_id39955",
            # uTheses UI is a SPA; this Solr proxy is the stable machine endpoint.
            url="https://utheses-gateway.univie.ac.at/api/proxy/solrSelect/q=id:39955&rows=10&start=0&wt=json",
            out_name="utheses_id39955_solr.json",
        ),
    ]

    retrieved: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {}
    for spec in specs:
        out_path = src_dir / spec.out_name
        if args.offline:
            if not out_path.exists():
                raise SystemExit(f"[fail] missing cached snapshot (offline): {out_path}")
        else:
            if args.refresh or (not out_path.exists()) or out_path.stat().st_size == 0:
                b = _fetch_with_retries(spec.url, timeout_s=90, attempts=4)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b)
        obj = _read_json(out_path)
        snapshots[spec.name] = obj
        retrieved.append(
            {
                "name": spec.name,
                "url": spec.url,
                "path": str(out_path),
                "bytes": int(out_path.stat().st_size),
                "sha256": _sha256(out_path),
            }
        )

    phaidra_info = _extract_phaidra_info(snapshots["phaidra_object_info_o1331600"])
    utheses_doc = _extract_utheses_doc(snapshots["utheses_solr_id39955"])

    thesis_doc_pid_raw = utheses_doc.get("thesis_doc_pid")
    thesis_doc_pid = thesis_doc_pid_raw.strip() if isinstance(thesis_doc_pid_raw, str) else None
    if thesis_doc_pid:
        pid_tok = _safe_pid_token(thesis_doc_pid)
        thesis_spec = SourceSpec(
            name=f"phaidra_object_metadata_{pid_tok}",
            url=f"https://phaidra.univie.ac.at/api/object/{thesis_doc_pid}/metadata",
            out_name=f"phaidra_{pid_tok}_metadata.json",
        )
        out_path = src_dir / thesis_spec.out_name
        if args.offline:
            if not out_path.exists():
                raise SystemExit(f"[fail] missing cached snapshot (offline): {out_path}")
        else:
            if args.refresh or (not out_path.exists()) or out_path.stat().st_size == 0:
                b = _fetch_with_retries(thesis_spec.url, timeout_s=90, attempts=4)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b)
        obj = _read_json(out_path)
        snapshots[thesis_spec.name] = obj
        retrieved.append(
            {
                "name": thesis_spec.name,
                "url": thesis_spec.url,
                "path": str(out_path),
                "bytes": int(out_path.stat().st_size),
                "sha256": _sha256(out_path),
            }
        )

    fulltext_locked = utheses_doc.get("fulltext_locked")
    try:
        fulltext_locked_i = int(fulltext_locked) if fulltext_locked is not None else None
    except Exception:
        fulltext_locked_i = None

    probe = {
        "generated_utc": _utc_now(),
        "dataset": "Giustina 2016 thesis (uTheses id=39955; Phaidra o:1331600)",
        "utheses": {
            "id": 39955,
            "solr_doc": {
                "publication_date": utheses_doc.get("publication_date"),
                "fulltext_locked": fulltext_locked_i,
                "pid": utheses_doc.get("pid"),
                "thesis_doc_pid": thesis_doc_pid,
                "urn": utheses_doc.get("urn"),
            },
            "urls": {
                "detail": "https://utheses.univie.ac.at/detail/39955",
                "solr_proxy": specs[1].url,
            },
        },
        "phaidra": {
            "pid": "o:1331600",
            "info": {
                "cmodel": phaidra_info.get("cmodel"),
                "dc_title": phaidra_info.get("dc_title"),
                "dc_creator": phaidra_info.get("dc_creator"),
                "dc_identifier": phaidra_info.get("dc_identifier"),
                "datastreams": phaidra_info.get("datastreams"),
                "created": phaidra_info.get("created"),
                "modified": phaidra_info.get("modified"),
            },
            "urls": {
                "detail": "https://phaidra.univie.ac.at/o:1331600",
                "api_info": specs[0].url,
            },
        },
        "thesis_doc": None,
        "status": {
            "is_public_fulltext": bool(fulltext_locked_i == 0),
            "note": (
                "If fulltext_locked becomes 0, fetch thesis_doc_pid metadata and attempt to download the PDF"
                " (verify sha256 and cache in data/quantum/sources)."
            ),
        },
        "source_snapshots": retrieved,
    }

    if thesis_doc_pid:
        pid_tok = _safe_pid_token(thesis_doc_pid)
        api_metadata = f"https://phaidra.univie.ac.at/api/object/{thesis_doc_pid}/metadata"
        api_download = f"https://phaidra.univie.ac.at/api/object/{thesis_doc_pid}/download"
        detail = f"https://phaidra.univie.ac.at/{thesis_doc_pid}"

        meta_summary: dict[str, Any] = {}
        meta_obj = snapshots.get(f"phaidra_object_metadata_{pid_tok}")
        if isinstance(meta_obj, dict):
            meta_summary = _extract_phaidra_jsonld_summary(meta_obj)

        probe["thesis_doc"] = {
            "pid": thesis_doc_pid,
            "jsonld_summary": meta_summary,
            "urls": {
                "detail": detail,
                "api_metadata": api_metadata,
                "api_download": api_download,
            },
            "http_head_status": {
                "detail": _http_head_status(detail),
                "api_metadata": _http_head_status(api_metadata),
                "api_download": _http_head_status(api_download),
            },
        }

    _write_json(out_dir / "giustina2016_utheses_probe.json", probe)

    manifest = {
        "generated_utc": _utc_now(),
        "dataset": probe["dataset"],
        "notes": [
            "This dataset is used as a watch/probe for potential public release of thesis fulltext or attachments.",
            "uTheses UI is a SPA; the stable machine endpoint is the uTheses gateway Solr proxy.",
            "At the time of this snapshot, uTheses may report fulltext_locked=1 (no public full text).",
            "Phaidra thesis_doc_pid metadata may be visible while the detail/download endpoints remain restricted.",
        ],
        "files": retrieved,
    }
    _write_json(src_dir / "manifest.json", manifest)

    print("[ok] wrote:")
    print(f"  - {out_dir / 'giustina2016_utheses_probe.json'}")
    print(f"  - {src_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
