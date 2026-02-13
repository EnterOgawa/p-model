from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class EventSpec:
    name: str
    slug: str
    type: str
    profile: str
    optional: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "slug": self.slug,
            "type": self.type,
            "profile": self.profile,
            "optional": self.optional,
        }


def _load_event_specs(root: Path) -> List[EventSpec]:
    path = root / "data" / "gw" / "event_list.json"
    if not path.exists():
        return []
    try:
        obj = _read_json(path)
    except Exception:
        return []
    evs = obj.get("events")
    if not isinstance(evs, list):
        return []
    out: List[EventSpec] = []
    for e in evs:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()
        if not name:
            continue
        slug = str(e.get("slug") or name.lower()).strip() or name.lower()
        typ = str(e.get("type") or "").strip().upper() or "UNKNOWN"
        prof = str(e.get("profile") or "").strip()
        optional = bool(e.get("optional", True))
        out.append(EventSpec(name=name, slug=slug, type=typ, profile=prof, optional=optional))
    return out


def _load_run_all_status(root: Path) -> Dict[str, Any]:
    path = root / "output" / "private" / "summary" / "run_all_status.json"
    if not path.exists():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _task_map_from_status(status: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tasks = status.get("tasks")
    if not isinstance(tasks, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for t in tasks:
        if not isinstance(t, dict):
            continue
        key = str(t.get("key") or "").strip()
        if not key:
            continue
        out[key] = t
    return out


_REASON_RULES: List[Tuple[str, re.Pattern[str]]] = [
    ("offline_missing_cache", re.compile(r"offline mode", flags=re.IGNORECASE)),
    ("http_404", re.compile(r"HTTPError\\s*:\\s*404", flags=re.IGNORECASE)),
    ("fetch_inputs_failed", re.compile(r"\\[err\\]\\s*fetch inputs failed", flags=re.IGNORECASE)),
    ("no_valid_detector_tracks", re.compile(r"no valid detector tracks", flags=re.IGNORECASE)),
    ("fit_failed", re.compile(r"\\[warn\\]\\s*fit failed", flags=re.IGNORECASE)),
    ("ssl_error", re.compile(r"SSL|certificate|CERTIFICATE_VERIFY_FAILED", flags=re.IGNORECASE)),
    ("timeout", re.compile(r"timed? out|timeout", flags=re.IGNORECASE)),
]


def _classify_reason(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    for name, pat in _REASON_RULES:
        if pat.search(s):
            return name
    return "unknown"


def _event_result(
    *, root: Path, spec: EventSpec, task_rec: Optional[Dict[str, Any]]
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """
    Returns (ok, reason, log_path, metrics_path).
    reason is set only when ok is False (best-effort classification).
    """
    metrics = root / "output" / "private" / "gw" / f"{spec.slug}_chirp_phase_metrics.json"
    metrics_path = str(metrics) if metrics.exists() else None

    if task_rec is None:
        # If no run_all record exists, fall back to presence/absence of metrics.
        if metrics_path is not None:
            return True, "", None, metrics_path
        return False, "not_run", None, None

    if bool(task_rec.get("skipped")):
        reason = str(task_rec.get("reason") or "") or "skipped"
        return False, reason, str(task_rec.get("log") or "") or None, metrics_path

    ok = bool(task_rec.get("ok"))
    log_path = str(task_rec.get("log") or "") or None
    if ok:
        # Success should normally imply metrics exist, but keep this robust.
        if metrics_path is None:
            return True, "", log_path, None
        return True, "", log_path, metrics_path

    tail = str(task_rec.get("tail") or "")
    reason = _classify_reason(tail) or "failed"
    return False, reason, log_path, metrics_path


def _summarize_by_type(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        typ = str(r.get("type") or "UNKNOWN")
        ok = bool(r.get("ok"))
        reason = str(r.get("reason") or "")

        s = by_type.setdefault(
            typ,
            {
                "n_events": 0,
                "n_ok": 0,
                "n_failed": 0,
                "reasons": {},
            },
        )
        s["n_events"] += 1
        if ok:
            s["n_ok"] += 1
        else:
            s["n_failed"] += 1
            if reason:
                s["reasons"][reason] = int(s["reasons"].get(reason, 0)) + 1

    return by_type


def _load_detector_stats_from_metrics(metrics_path: Optional[str]) -> Dict[str, Any]:
    if not metrics_path:
        return {}
    try:
        p = Path(metrics_path)
        if not p.exists():
            return {}
        j = _read_json(p)
    except Exception:
        return {}

    ok_dets: List[str] = []
    for d in (j.get("detectors") or []):
        if not isinstance(d, dict):
            continue
        det = str(d.get("detector") or "").strip()
        if det:
            ok_dets.append(det)

    skipped = j.get("skipped_detectors") or []
    if not isinstance(skipped, list):
        skipped = []

    skipped_simple: List[Dict[str, Any]] = []
    skipped_by_reason: Dict[str, int] = {}
    skipped_by_subreason: Dict[str, int] = {}
    skipped_by_detector: Dict[str, Dict[str, int]] = {}
    skipped_by_detector_subreason: Dict[str, Dict[str, int]] = {}
    for s in skipped:
        if not isinstance(s, dict):
            continue
        det = str(s.get("detector") or "").strip() or "?"
        reason = str(s.get("reason") or "").strip() or "unknown"
        subreason = str(s.get("subreason") or "").strip()
        skipped_simple.append({"detector": det, "reason": reason, **({"subreason": subreason} if subreason else {})})
        skipped_by_reason[reason] = int(skipped_by_reason.get(reason, 0)) + 1
        per_det = skipped_by_detector.setdefault(det, {})
        per_det[reason] = int(per_det.get(reason, 0)) + 1
        if subreason:
            skipped_by_subreason[subreason] = int(skipped_by_subreason.get(subreason, 0)) + 1
            per_det2 = skipped_by_detector_subreason.setdefault(det, {})
            per_det2[subreason] = int(per_det2.get(subreason, 0)) + 1

    return {
        "ok_detectors": ok_dets,
        "n_ok_detectors": int(len(ok_dets)),
        "skipped_detectors": skipped_simple,
        "n_skipped_detectors": int(len(skipped_simple)),
        "skipped_by_reason": skipped_by_reason,
        **({"skipped_by_subreason": skipped_by_subreason} if skipped_by_subreason else {}),
        "skipped_by_detector": skipped_by_detector,
        **({"skipped_by_detector_subreason": skipped_by_detector_subreason} if skipped_by_detector_subreason else {}),
    }


def _summarize_detector_skips(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_reason: Dict[str, int] = {}
    by_subreason: Dict[str, int] = {}
    by_detector: Dict[str, Dict[str, int]] = {}
    by_detector_subreason: Dict[str, Dict[str, int]] = {}
    for r in rows:
        s = r.get("detector_stats") or {}
        if not isinstance(s, dict):
            continue
        per_reason = s.get("skipped_by_reason") or {}
        if isinstance(per_reason, dict):
            for k, v in per_reason.items():
                by_reason[str(k)] = int(by_reason.get(str(k), 0)) + int(v or 0)
        per_sub = s.get("skipped_by_subreason") or {}
        if isinstance(per_sub, dict):
            for k, v in per_sub.items():
                by_subreason[str(k)] = int(by_subreason.get(str(k), 0)) + int(v or 0)
        per_det = s.get("skipped_by_detector") or {}
        if isinstance(per_det, dict):
            for det, rr in per_det.items():
                if not isinstance(rr, dict):
                    continue
                out = by_detector.setdefault(str(det), {})
                for reason, v in rr.items():
                    out[str(reason)] = int(out.get(str(reason), 0)) + int(v or 0)
        per_det2 = s.get("skipped_by_detector_subreason") or {}
        if isinstance(per_det2, dict):
            for det, rr in per_det2.items():
                if not isinstance(rr, dict):
                    continue
                out2 = by_detector_subreason.setdefault(str(det), {})
                for reason, v in rr.items():
                    out2[str(reason)] = int(out2.get(str(reason), 0)) + int(v or 0)

    return {
        "skipped_by_reason": by_reason,
        **({"skipped_by_subreason": by_subreason} if by_subreason else {}),
        "skipped_by_detector": by_detector,
        **({"skipped_by_detector_subreason": by_detector_subreason} if by_detector_subreason else {}),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose GW event_list run status (success/failure reasons).")
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: output/private/gw/gw_event_list_diagnostics.json).",
    )
    args = ap.parse_args(argv)

    root = _repo_root()
    out_path = Path(args.out) if args.out else (root / "output" / "private" / "gw" / "gw_event_list_diagnostics.json")
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()

    specs = _load_event_specs(root)
    status = _load_run_all_status(root)
    task_map = _task_map_from_status(status)

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        key = f"gw_{spec.slug}_chirp_phase"
        rec = task_map.get(key)
        ok, reason, log_path, metrics_path = _event_result(root=root, spec=spec, task_rec=rec)
        det_stats = _load_detector_stats_from_metrics(metrics_path)
        rows.append(
            {
                **spec.to_dict(),
                "task_key": key,
                "ok": ok,
                **({"reason": reason} if (not ok and reason) else {}),
                **({"log": log_path} if log_path else {}),
                **({"metrics": metrics_path} if metrics_path else {}),
                **({"detector_stats": det_stats} if det_stats else {}),
            }
        )

    summary = {
        "generated_utc": _iso_utc_now(),
        "inputs": {
            "event_list": str(root / "data" / "gw" / "event_list.json"),
            "run_all_status": str(root / "output" / "private" / "summary" / "run_all_status.json"),
        },
        "events": rows,
        "by_type": _summarize_by_type(rows),
        "detectors": _summarize_detector_skips(rows),
    }

    _write_json(out_path, summary)
    print(f"[ok] json: {out_path}")

    try:
        worklog.append_event(
            {
                "event_type": "gw_event_list_diagnostics",
                "inputs": summary.get("inputs"),
                "outputs": {"json": out_path},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
