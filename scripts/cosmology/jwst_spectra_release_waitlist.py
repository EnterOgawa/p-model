"""
jwst_spectra_release_waitlist.py

Phase 4 / Step 4.6:
Summarize JWST/MAST per-target manifests and highlight unreleased (proprietary) observations,
so the "公開待ち" state is machine-verifiable and reproducible offline.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_utc_iso(s: str) -> datetime:
    # Expect "+00:00" style. Guard for robustness.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _fmt_utc(dt: datetime | None) -> str | None:
    # 条件分岐: `dt is None` を満たす経路を評価する。
    if dt is None:
        return None

    return dt.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class TargetReleaseRow:
    target_slug: str
    target_name: str
    manifest_path: str
    manifest_generated_utc: str | None
    n_obs: int
    n_released: int
    n_unreleased: int
    next_release_utc: str | None
    latest_release_utc: str | None


def _summarize_manifest(manifest_path: Path) -> TargetReleaseRow:
    obj = _load_json(manifest_path)
    target_slug = str(obj.get("target_slug") or manifest_path.parent.name)
    target_name = str(obj.get("target_name") or target_slug)
    generated_utc = obj.get("generated_utc")

    obs = obj.get("obs", []) or []
    # 条件分岐: `not isinstance(obs, list)` を満たす経路を評価する。
    if not isinstance(obs, list):
        obs = []

    n_obs = len(obs)
    flags = [o.get("is_released") for o in obs if isinstance(o, dict)]
    n_released = sum(f is True for f in flags)
    n_unreleased = sum(f is False for f in flags)

    release_times_all: list[datetime] = []
    release_times_unreleased: list[datetime] = []
    for o in obs:
        # 条件分岐: `not isinstance(o, dict)` を満たす経路を評価する。
        if not isinstance(o, dict):
            continue

        t = o.get("t_obs_release_utc")
        # 条件分岐: `not t` を満たす経路を評価する。
        if not t:
            continue

        try:
            dt = _parse_utc_iso(str(t))
        except Exception:
            continue

        release_times_all.append(dt)
        # 条件分岐: `o.get("is_released") is False` を満たす経路を評価する。
        if o.get("is_released") is False:
            release_times_unreleased.append(dt)

    next_release = min(release_times_unreleased) if release_times_unreleased else None
    latest_release = max(release_times_all) if release_times_all else None

    return TargetReleaseRow(
        target_slug=target_slug,
        target_name=target_name,
        manifest_path=str(manifest_path.as_posix()),
        manifest_generated_utc=str(generated_utc) if generated_utc is not None else None,
        n_obs=n_obs,
        n_released=n_released,
        n_unreleased=n_unreleased,
        next_release_utc=_fmt_utc(next_release),
        latest_release_utc=_fmt_utc(latest_release),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize JWST spectra manifest release status (Phase 4 / Step 4.6).")
    ap.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "cosmology" / "mast" / "jwst_spectra"),
        help="Input JWST spectra cache directory (default: data/cosmology/mast/jwst_spectra)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "output" / "private" / "cosmology"),
        help="Output directory (default: output/private/cosmology)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_dir = Path(args.data_dir)
    # 条件分岐: `not data_dir.is_absolute()` を満たす経路を評価する。
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    out_dir = Path(args.out_dir)
    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: list[Path] = []
    for child in sorted(data_dir.iterdir()):
        # 条件分岐: `not child.is_dir()` を満たす経路を評価する。
        if not child.is_dir():
            continue

        # 条件分岐: `child.name.startswith("_") or child.name.startswith(".")` を満たす経路を評価する。

        if child.name.startswith("_") or child.name.startswith("."):
            continue

        mp = child / "manifest.json"
        # 条件分岐: `mp.exists()` を満たす経路を評価する。
        if mp.exists():
            manifest_paths.append(mp)

    rows = [_summarize_manifest(p) for p in manifest_paths]
    rows_sorted = sorted(rows, key=lambda r: (r.n_unreleased == 0, r.next_release_utc or "", r.target_slug))

    now_utc = datetime.now(timezone.utc)
    blocked = [r for r in rows_sorted if r.n_unreleased > 0]
    blocked_pretty = []
    for r in blocked:
        days_left = None
        # 条件分岐: `r.next_release_utc` を満たす経路を評価する。
        if r.next_release_utc:
            try:
                dt = _parse_utc_iso(r.next_release_utc)
                days_left = (dt - now_utc).total_seconds() / 86400.0
            except Exception:
                days_left = None

        blocked_pretty.append(
            {
                "target": r.target_name,
                "target_slug": r.target_slug,
                "next_release_utc": r.next_release_utc,
                "days_until_release": days_left,
            }
        )

    out_json = out_dir / "jwst_spectra_release_waitlist.json"
    out_csv = out_dir / "jwst_spectra_release_waitlist.csv"

    payload: dict[str, Any] = {
        "generated_utc": _utc_now_iso(),
        "domain": "cosmology",
        "step": "4.6 (JWST/MAST spectra release waitlist)",
        "inputs": {"data_dir": str(data_dir.as_posix())},
        "outputs": {"json": str(out_json.as_posix()), "csv": str(out_csv.as_posix())},
        "summary": {
            "targets_n": len(rows_sorted),
            "blocked_targets_n": len(blocked),
        },
        "blocked_targets": blocked_pretty,
        "rows": [r.__dict__ for r in rows_sorted],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "target_slug",
                "target_name",
                "n_obs",
                "n_released",
                "n_unreleased",
                "next_release_utc",
                "latest_release_utc",
                "manifest_generated_utc",
                "manifest_path",
            ]
        )
        for r in rows_sorted:
            w.writerow(
                [
                    r.target_slug,
                    r.target_name,
                    r.n_obs,
                    r.n_released,
                    r.n_unreleased,
                    r.next_release_utc or "",
                    r.latest_release_utc or "",
                    r.manifest_generated_utc or "",
                    r.manifest_path,
                ]
            )

    print(f"[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")
    # 条件分岐: `blocked` を満たす経路を評価する。
    if blocked:
        print("[info] blocked targets (unreleased observations):")
        for b in blocked_pretty:
            # 条件分岐: `b["days_until_release"] is None` を満たす経路を評価する。
            if b["days_until_release"] is None:
                print(f"  - {b['target']} ({b['next_release_utc']})")
            else:
                print(f"  - {b['target']} ({b['next_release_utc']}; in {b['days_until_release']:.1f} days)")
    else:
        print("[info] no blocked targets found.")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
