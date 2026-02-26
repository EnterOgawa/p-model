from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog


# 関数: `_iso_now` の入出力契約と処理意図を定義する。
def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            # 条件分岐: `not block` を満たす経路を評価する。
            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 条件分岐: `not rows` を満たす経路を評価する。
        if not rows:
            return

        headers = list(rows[0].keys())
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(h) for h in headers])


# 関数: `_ensure_pdf` の入出力契約と処理意図を定義する。

def _ensure_pdf(*, url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `path.exists()` を満たす経路を評価する。
    if path.exists():
        return

    urllib.request.urlretrieve(url, path)


# 関数: `_normalize_text` の入出力契約と処理意図を定義する。

def _normalize_text(text: str) -> str:
    out = text.replace("－", "-").replace("–", "-").replace("—", "-").replace("−", "-")
    out = out.replace("×", "x")
    out = re.sub(r"\s+", " ", out)
    return out


# 関数: `_extract_pdf_text` の入出力契約と処理意図を定義する。

def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    joined = "\n".join((page.extract_text() or "") for page in reader.pages)
    return _normalize_text(joined)


# 関数: `_match_required` の入出力契約と処理意図を定義する。

def _match_required(pattern: str, text: str, *, label: str) -> tuple[float, float]:
    hit = re.search(pattern, text, flags=re.IGNORECASE)
    # 条件分岐: `not hit` を満たす経路を評価する。
    if not hit:
        raise SystemExit(f"[fail] cannot extract {label} from CKM source text")

    value = float(hit.group(1))
    sigma = float(hit.group(2))
    return value, sigma


# 関数: `_build_plot` の入出力契約と処理意図を定義する。

def _build_plot(
    *,
    out_png: Path,
    sum_reported: float,
    sigma_reported: float,
    sum_from_elements: float,
    delta_ckm: float,
    sigma_uncorrelated: float,
    hard_z_threshold: float,
    watch_z_threshold: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=170)
    ax0, ax1 = axes

    ax0.errorbar(
        [0, 1],
        [sum_reported, sum_from_elements],
        yerr=[sigma_reported, 0.0],
        fmt="o",
        capsize=5.0,
        color="#4c78a8",
    )
    ax0.axhline(1.0, color="#444444", ls="--", lw=1.0, label="unitarity = 1")
    ax0.set_xticks([0, 1], ["reported (PDG)", "from Vud/Vus/Vub"])
    ax0.set_ylabel("first-row sum")
    ax0.set_title("CKM first-row closure")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="upper right", fontsize=8)

    z_reported = abs(delta_ckm / sigma_reported) if sigma_reported > 0.0 else float("nan")
    z_uncorrelated = abs(delta_ckm / sigma_uncorrelated) if sigma_uncorrelated > 0.0 else float("nan")
    ax1.bar(
        ["|z| reported", "|z| uncorrelated", "watch gate", "hard gate"],
        [z_reported, z_uncorrelated, watch_z_threshold, hard_z_threshold],
        color=["#f58518", "#4c78a8", "#72b7b2", "#54a24b"],
    )
    ax1.set_ylabel("closure z-score")
    ax1.set_title(f"Δ_CKM={delta_ckm:.4e}")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Weak-interaction CKM first-row audit (Step 8.7.22)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# 関数: `_correlation_reassessment` の入出力契約と処理意図を定義する。

def _correlation_reassessment(
    *,
    delta_ckm_reported: float,
    first_row_sigma_reported: float,
    first_row_sigma_uncorrelated: float,
    hard_z_threshold: float,
    watch_z_threshold: float,
) -> dict[str, Any]:
    abs_delta = abs(delta_ckm_reported)
    z_reported = abs_delta / first_row_sigma_reported if first_row_sigma_reported > 0.0 else float("nan")
    z_uncorrelated = (
        abs_delta / first_row_sigma_uncorrelated if first_row_sigma_uncorrelated > 0.0 else float("nan")
    )
    sigma_required_watch = abs_delta / watch_z_threshold if watch_z_threshold > 0.0 else float("nan")
    sigma_required_hard = abs_delta / hard_z_threshold if hard_z_threshold > 0.0 else float("nan")
    sigma_gap_watch = (
        sigma_required_watch - first_row_sigma_reported
        if math.isfinite(sigma_required_watch) and first_row_sigma_reported > 0.0
        else float("nan")
    )
    sigma_ratio_reported_to_uncorrelated = (
        first_row_sigma_reported / first_row_sigma_uncorrelated
        if first_row_sigma_uncorrelated > 0.0
        else float("nan")
    )
    watch_to_pass_with_reported_sigma = bool(math.isfinite(z_reported) and z_reported <= watch_z_threshold)
    watch_to_pass_with_uncorrelated_sigma = bool(math.isfinite(z_uncorrelated) and z_uncorrelated <= watch_z_threshold)
    hard_to_pass_with_reported_sigma = bool(math.isfinite(z_reported) and z_reported <= hard_z_threshold)

    # 条件分岐: `watch_to_pass_with_reported_sigma` を満たす経路を評価する。
    if watch_to_pass_with_reported_sigma:
        watch_resolution_status = "watch_pass_already_satisfied"
        watch_lock_reason = "none"
    else:
        watch_resolution_status = "watch_locked_by_current_primary_source_precision"
        # 条件分岐: `math.isfinite(sigma_gap_watch) and sigma_gap_watch > 0.0` を満たす経路を評価する。
        if math.isfinite(sigma_gap_watch) and sigma_gap_watch > 0.0:
            watch_lock_reason = (
                "reported first-row sigma is below watch-pass requirement; input precision update is required."
            )
        else:
            watch_lock_reason = "watch gate not satisfied under current primary-source first-row closure."

    return {
        "z_reported": z_reported,
        "z_uncorrelated_elements": z_uncorrelated,
        "sigma_required_for_watch_pass": sigma_required_watch,
        "sigma_required_for_hard_pass": sigma_required_hard,
        "sigma_gap_watch_minus_reported": sigma_gap_watch,
        "sigma_ratio_reported_to_uncorrelated": sigma_ratio_reported_to_uncorrelated,
        "hard_to_pass_with_reported_sigma": hard_to_pass_with_reported_sigma,
        "watch_to_pass_with_reported_sigma": watch_to_pass_with_reported_sigma,
        "watch_to_pass_with_uncorrelated_sigma": watch_to_pass_with_uncorrelated_sigma,
        "watch_resolution_status": watch_resolution_status,
        "watch_lock_reason": watch_lock_reason,
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8.7.22: CKM first-row quantitative audit")
    parser.add_argument(
        "--source-url",
        type=str,
        default="https://pdg.lbl.gov/2024/reviews/rpp2024-rev-ckm-matrix.pdf",
        help="Primary source URL for CKM review PDF.",
    )
    parser.add_argument(
        "--in-pdf",
        type=Path,
        default=ROOT / "data" / "quantum" / "sources" / "pdg_rpp2024_rev_ckm_matrix.pdf",
        help="Local cache path for CKM review PDF.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "output" / "public" / "quantum",
        help="Output directory.",
    )
    parser.add_argument(
        "--hard-z-threshold",
        type=float,
        default=3.0,
        help="Hard gate threshold for abs(z).",
    )
    parser.add_argument(
        "--watch-z-threshold",
        type=float,
        default=2.0,
        help="Watch threshold for abs(z).",
    )
    args = parser.parse_args()

    in_pdf = args.in_pdf
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_pdf(url=str(args.source_url), path=in_pdf)
    text = _extract_pdf_text(in_pdf)

    vud, vud_sigma = _match_required(
        r"\|Vud\|\s*=\s*([0-9]+\.[0-9]+)\s*[±\+\-]\s*([0-9]+\.[0-9]+)\s*\.?\s*\(12\.7\)",
        text,
        label="|Vud| (Eq.12.7)",
    )
    vus, vus_sigma = _match_required(
        r"\|Vus\|\s*=\s*([0-9]+\.[0-9]+)\s*[±\+\-]\s*([0-9]+\.[0-9]+)\s*\.?\s*\(12\.8\)",
        text,
        label="|Vus| (Eq.12.8)",
    )
    vub_milli, vub_milli_sigma = _match_required(
        r"\|Vub\|\s*=\s*\(?\s*([0-9]+\.[0-9]+)\s*[±\+\-]\s*([0-9]+\.[0-9]+)\s*\)?\s*x?\s*10\s*-?\s*3\s*\.?\s*\(12\.12\)",
        text,
        label="|Vub| (Eq.12.12)",
    )
    first_row_sum_reported, first_row_sigma_reported = _match_required(
        r"\|Vud\|2\+\|Vus\|2\+\|Vub\|2\s*=\s*([0-9]+\.[0-9]+)\s*[±\+\-]\s*([0-9]+\.[0-9]+)\s*\(1st\s*row\)",
        text,
        label="CKM first-row sum",
    )

    vub = float(vub_milli * 1.0e-3)
    vub_sigma = float(vub_milli_sigma * 1.0e-3)

    first_row_sum_from_elements = float(vud**2 + vus**2 + vub**2)
    first_row_sigma_uncorrelated = float(
        math.sqrt((2.0 * vud * vud_sigma) ** 2 + (2.0 * vus * vus_sigma) ** 2 + (2.0 * vub * vub_sigma) ** 2)
    )
    delta_ckm_reported = float(first_row_sum_reported - 1.0)
    z_abs_reported = float(abs(delta_ckm_reported) / first_row_sigma_reported) if first_row_sigma_reported > 0 else float("nan")
    correlation_reassessment = _correlation_reassessment(
        delta_ckm_reported=delta_ckm_reported,
        first_row_sigma_reported=first_row_sigma_reported,
        first_row_sigma_uncorrelated=first_row_sigma_uncorrelated,
        hard_z_threshold=float(args.hard_z_threshold),
        watch_z_threshold=float(args.watch_z_threshold),
    )

    hard_pass = bool(math.isfinite(z_abs_reported) and z_abs_reported <= float(args.hard_z_threshold))
    watch_pass = bool(math.isfinite(z_abs_reported) and z_abs_reported <= float(args.watch_z_threshold))
    # 条件分岐: `hard_pass and watch_pass` を満たす経路を評価する。
    if hard_pass and watch_pass:
        status = "pass"
    # 条件分岐: 前段条件が不成立で、`hard_pass` を追加評価する。
    elif hard_pass:
        status = "watch"
    else:
        status = "reject"

    out_json = out_dir / "weak_interaction_ckm_first_row_audit.json"
    out_csv = out_dir / "weak_interaction_ckm_first_row_audit_summary.csv"
    out_png = out_dir / "weak_interaction_ckm_first_row_audit.png"

    summary_rows = [
        {
            "source": "PDG 2024 CKM review (Eq.12.7/12.8/12.12; first-row line)",
            "vud": vud,
            "vud_sigma": vud_sigma,
            "vus": vus,
            "vus_sigma": vus_sigma,
            "vub": vub,
            "vub_sigma": vub_sigma,
            "first_row_sum_reported": first_row_sum_reported,
            "first_row_sigma_reported": first_row_sigma_reported,
            "first_row_sum_from_elements": first_row_sum_from_elements,
            "first_row_sigma_uncorrelated": first_row_sigma_uncorrelated,
            "delta_ckm_reported": delta_ckm_reported,
            "abs_z_reported": z_abs_reported,
            "abs_z_uncorrelated_elements": correlation_reassessment["z_uncorrelated_elements"],
            "sigma_required_for_watch_pass": correlation_reassessment["sigma_required_for_watch_pass"],
            "sigma_gap_watch_minus_reported": correlation_reassessment["sigma_gap_watch_minus_reported"],
            "watch_resolution_status": correlation_reassessment["watch_resolution_status"],
            "hard_z_threshold": float(args.hard_z_threshold),
            "watch_z_threshold": float(args.watch_z_threshold),
            "hard_pass": hard_pass,
            "watch_pass": watch_pass,
            "status": status,
        }
    ]
    _write_csv(out_csv, summary_rows)
    _build_plot(
        out_png=out_png,
        sum_reported=first_row_sum_reported,
        sigma_reported=first_row_sigma_reported,
        sum_from_elements=first_row_sum_from_elements,
        delta_ckm=delta_ckm_reported,
        sigma_uncorrelated=first_row_sigma_uncorrelated,
        hard_z_threshold=float(args.hard_z_threshold),
        watch_z_threshold=float(args.watch_z_threshold),
    )

    payload = {
        "generated_utc": _iso_now(),
        "phase": 8,
        "step": "8.7.22",
        "title": "Weak-interaction CKM first-row audit",
        "source": {
            "url": str(args.source_url),
            "local_pdf": _rel(in_pdf),
            "sha256": _sha256(in_pdf),
        },
        "extraction": {
            "vud_eq_12_7": {"value": vud, "sigma": vud_sigma},
            "vus_eq_12_8": {"value": vus, "sigma": vus_sigma},
            "vub_eq_12_12": {"value": vub, "sigma": vub_sigma},
            "first_row_reported": {"value": first_row_sum_reported, "sigma": first_row_sigma_reported},
        },
        "derived": {
            "first_row_sum_from_elements": first_row_sum_from_elements,
            "first_row_sigma_uncorrelated": first_row_sigma_uncorrelated,
            "delta_ckm_reported": delta_ckm_reported,
            "abs_z_reported": z_abs_reported,
        },
        "correlation_reassessment": correlation_reassessment,
        "gate": {
            "hard_z_threshold": float(args.hard_z_threshold),
            "watch_z_threshold": float(args.watch_z_threshold),
            "hard_pass": hard_pass,
            "watch_pass": watch_pass,
            "status": status,
            "rule": "hard if abs(z)<=3, watch if abs(z)<=2 (abs(z)>2 and <=3 => watch)",
        },
        "pmns_gate": {
            "status": "not_evaluated",
            "reason": "PMNS closure is out of scope in this CKM first-row pack.",
        },
        "outputs": {
            "summary_csv": _rel(out_csv),
            "audit_json": _rel(out_json),
            "audit_png": _rel(out_png),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    worklog.append_event(
        {
            "type": "step_update",
            "step": "8.7.22",
            "script": "scripts/quantum/weak_interaction_ckm_first_row_audit.py",
            "inputs": [_rel(in_pdf)],
            "outputs": [_rel(out_csv), _rel(out_json), _rel(out_png)],
            "gate_status": status,
            "abs_z_reported": z_abs_reported,
        }
    )

    print("[ok] wrote:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print(f"  {out_png}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
