from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog


# 関数: `_read_json` の入出力契約と処理意図を定義する。
def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_sha256_file` の入出力契約と処理意図を定義する。

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_ROOT).as_posix()
    except Exception:
        return path.resolve().as_posix()


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "input_id",
                "path",
                "exists",
                "expected_sha256",
                "current_sha256",
                "hash_changed",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Route-B watch monitor: rerun only when input hashes changed.")
    ap.add_argument(
        "--audit-json",
        type=str,
        default=str(_ROOT / "output" / "public" / "quantum" / "weak_interaction_beta_decay_route_b_standalone_audit.json"),
        help="Route-B standalone audit JSON to monitor.",
    )
    ap.add_argument("--step-tag", type=str, default="8.7.31.34", help="Step tag for monitor output.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(_ROOT / "output" / "private" / "quantum"),
        help="Private output directory for monitor artifacts.",
    )
    ap.add_argument(
        "--public-outdir",
        type=str,
        default=str(_ROOT / "output" / "public" / "quantum"),
        help="Public output directory for monitor artifacts.",
    )
    ap.add_argument("--no-public-copy", action="store_true", help="Do not copy outputs to public directory.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    audit_json = Path(args.audit_json)
    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    out_json = outdir / "weak_interaction_beta_decay_route_b_watch_monitor.json"
    out_csv = outdir / "weak_interaction_beta_decay_route_b_watch_monitor.csv"

    # 条件分岐: `not audit_json.exists()` を満たす経路を評価する。
    if not audit_json.exists():
        raise SystemExit(f"[fail] audit json not found: {audit_json}")

    payload_audit = _read_json(audit_json)
    inputs = payload_audit.get("inputs") or {}
    # 条件分岐: `not isinstance(inputs, dict) or not inputs` を満たす経路を評価する。
    if not isinstance(inputs, dict) or not inputs:
        raise SystemExit("[fail] audit json has no usable inputs block")

    rows: List[Dict[str, Any]] = []
    changed_inputs: List[str] = []
    for input_id, meta_any in inputs.items():
        meta = meta_any if isinstance(meta_any, dict) else {}
        input_path = _ROOT / str(meta.get("path") or "")
        expected = str(meta.get("sha256") or "").lower()
        exists = input_path.exists()
        current = _sha256_file(input_path) if exists else ""
        changed = (not exists) or (current != expected)
        # 条件分岐: `changed` を満たす経路を評価する。
        if changed:
            changed_inputs.append(str(input_id))

        rows.append(
            {
                "input_id": str(input_id),
                "path": _rel(input_path),
                "exists": bool(exists),
                "expected_sha256": expected,
                "current_sha256": current,
                "hash_changed": bool(changed),
            }
        )

    changed_inputs_sorted = sorted(changed_inputs)
    input_hash_changed = bool(changed_inputs_sorted)
    route_decision = payload_audit.get("decision") or {}
    monitor_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "wavep.quantum.route_b_watch_monitor.v1",
        "phase": 8,
        "step": str(args.step_tag),
        "title": "Weak-interaction Route-B watch monitor",
        "audit_source": {
            "path": _rel(audit_json),
            "step": payload_audit.get("step"),
            "overall_status": route_decision.get("overall_status"),
            "transition": route_decision.get("transition"),
        },
        "monitor": {
            "input_hash_changed": input_hash_changed,
            "changed_inputs_n": len(changed_inputs_sorted),
            "changed_inputs": changed_inputs_sorted,
            "rerun_required": input_hash_changed,
            "rerun_policy": "rerun_route_b_only_if_input_hash_changed",
            "action_taken": "rerun_required_pending" if input_hash_changed else "skip_rerun_keep_fixed",
        },
        "inputs": rows,
        "outputs": {
            "monitor_json": _rel(out_json),
            "monitor_csv": _rel(out_csv),
        },
    }

    _write_json(out_json, monitor_payload)
    _write_csv(out_csv, rows)

    copied: List[Path] = []
    # 条件分岐: `not args.no_public_copy` を満たす経路を評価する。
    if not args.no_public_copy:
        public_outdir.mkdir(parents=True, exist_ok=True)
        for src in (out_json, out_csv):
            dst = public_outdir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)

    try:
        worklog.append_event(
            {
                "event_type": "quantum_route_b_watch_monitor",
                "argv": sys.argv,
                "inputs": {"audit_json": audit_json},
                "outputs": {
                    "monitor_json": out_json,
                    "monitor_csv": out_csv,
                    "public_copies": copied,
                },
                "metrics": {
                    "step": str(args.step_tag),
                    "audit_step": payload_audit.get("step"),
                    "audit_status": route_decision.get("overall_status"),
                    "input_hash_changed": input_hash_changed,
                    "changed_inputs_n": len(changed_inputs_sorted),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] monitor json: {out_json}")
    print(f"[ok] monitor csv : {out_csv}")
    # 条件分岐: `copied` を満たす経路を評価する。
    if copied:
        print(f"[ok] public copies: {len(copied)} -> {public_outdir}")

    print(f"[monitor] input_hash_changed={input_hash_changed} changed_inputs_n={len(changed_inputs_sorted)}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
