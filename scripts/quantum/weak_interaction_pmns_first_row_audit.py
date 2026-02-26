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

ROW_LABELS = ("e", "mu", "tau")
COL_LABELS = ("1", "2", "3")


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


# 関数: `_extract_pdf_text` の入出力契約と処理意図を定義する。

def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# 関数: `_normalize_text` の入出力契約と処理意図を定義する。

def _normalize_text(text: str) -> str:
    out = text.replace("−", "-").replace("–", "-").replace("—", "-").replace("→", "->")
    out = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", out)
    out = re.sub(r"\s+", " ", out)
    return out


# 関数: `_extract_matrix_ranges` の入出力契約と処理意図を定義する。

def _extract_matrix_ranges(text: str, *, marker: str, end_marker: str | None) -> list[dict[str, Any]]:
    start = text.find(marker)
    # 条件分岐: `start < 0` を満たす経路を評価する。
    if start < 0:
        raise SystemExit(f"[fail] cannot locate marker in PMNS source: {marker}")

    end = text.find(end_marker, start + len(marker)) if end_marker else -1
    # 条件分岐: `end < 0` を満たす経路を評価する。
    if end < 0:
        end = len(text)

    section = text[start:end]
    pairs = re.findall(r"([0-9]\.[0-9]+)\s*->\s*([0-9]\.[0-9]+)", section)
    # 条件分岐: `len(pairs) != 9` を満たす経路を評価する。
    if len(pairs) != 9:
        raise SystemExit(f"[fail] expected 9 PMNS range pairs at marker {marker}, found {len(pairs)}")

    out: list[dict[str, Any]] = []
    idx = 0
    for row in ROW_LABELS:
        for col in COL_LABELS:
            low = float(pairs[idx][0])
            high = float(pairs[idx][1])
            center = 0.5 * (low + high)
            sigma_proxy = (high - low) / 6.0
            out.append(
                {
                    "row": row,
                    "col": col,
                    "low_3sigma": low,
                    "high_3sigma": high,
                    "center_proxy": center,
                    "sigma_proxy": sigma_proxy,
                }
            )
            idx += 1

    return out


# 関数: `_dataset_metrics` の入出力契約と処理意図を定義する。

def _dataset_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(str(e["row"]), str(e["col"])): e for e in entries}
    row_e = [by_key[("e", "1")], by_key[("e", "2")], by_key[("e", "3")]]
    center_vals = [float(v["center_proxy"]) for v in row_e]
    sigma_vals = [float(v["sigma_proxy"]) for v in row_e]
    low_vals = [float(v["low_3sigma"]) for v in row_e]
    high_vals = [float(v["high_3sigma"]) for v in row_e]

    sum_center = float(sum(v * v for v in center_vals))
    sigma_uncorrelated_proxy = float(math.sqrt(sum((2.0 * center_vals[i] * sigma_vals[i]) ** 2 for i in range(3))))
    delta = float(sum_center - 1.0)
    abs_z = float(abs(delta) / sigma_uncorrelated_proxy) if sigma_uncorrelated_proxy > 0.0 else float("nan")

    sum_low = float(sum(v * v for v in low_vals))
    sum_high = float(sum(v * v for v in high_vals))

    return {
        "first_row_elements": {
            "Ue1": row_e[0],
            "Ue2": row_e[1],
            "Ue3": row_e[2],
        },
        "derived": {
            "first_row_sum_center_proxy": sum_center,
            "first_row_sigma_uncorrelated_proxy": sigma_uncorrelated_proxy,
            "delta_pmns_center_proxy": delta,
            "abs_z_center_proxy": abs_z,
            "first_row_sum_low_3sigma_corner": sum_low,
            "first_row_sum_high_3sigma_corner": sum_high,
        },
    }


# 関数: `_build_plot` の入出力契約と処理意図を定義する。

def _build_plot(
    *,
    out_png: Path,
    selected_label: str,
    selected_metrics: dict[str, Any],
    hard_z_threshold: float,
    watch_z_threshold: float,
) -> None:
    row = selected_metrics["first_row_elements"]
    labels = ["|Ue1|", "|Ue2|", "|Ue3|"]
    centers = [float(row["Ue1"]["center_proxy"]), float(row["Ue2"]["center_proxy"]), float(row["Ue3"]["center_proxy"])]
    sigma = [float(row["Ue1"]["sigma_proxy"]), float(row["Ue2"]["sigma_proxy"]), float(row["Ue3"]["sigma_proxy"])]
    lows = [float(row["Ue1"]["low_3sigma"]), float(row["Ue2"]["low_3sigma"]), float(row["Ue3"]["low_3sigma"])]
    highs = [float(row["Ue1"]["high_3sigma"]), float(row["Ue2"]["high_3sigma"]), float(row["Ue3"]["high_3sigma"])]

    derived = selected_metrics["derived"]
    abs_delta = abs(float(derived["delta_pmns_center_proxy"]))
    sigma_sum = float(derived["first_row_sigma_uncorrelated_proxy"])
    hard_level = float(hard_z_threshold * sigma_sum)
    watch_level = float(watch_z_threshold * sigma_sum)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=170)
    ax0, ax1 = axes

    x = [0, 1, 2]
    ax0.errorbar(x, centers, yerr=[3.0 * v for v in sigma], fmt="o", capsize=5.0, color="#4c78a8", label="center ±3σ proxy")
    for idx in range(3):
        ax0.hlines(y=[lows[idx], highs[idx]], xmin=idx - 0.22, xmax=idx + 0.22, color="#f58518", lw=2.0, alpha=0.9)

    ax0.set_xticks(x, labels)
    ax0.set_ylabel("PMNS first-row element magnitude")
    ax0.set_title(f"NuFIT v5.2 3σ ranges ({selected_label})")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="best", fontsize=8)

    ax1.bar(["abs(Δ_PMNS)", "watch (2σ)", "hard (3σ)"], [abs_delta, watch_level, hard_level], color=["#f58518", "#72b7b2", "#54a24b"])
    ax1.set_ylabel("first-row closure gap")
    ax1.set_title(
        "PMNS first-row closure\n"
        f"Δ={float(derived['delta_pmns_center_proxy']):.4e}, |z|={float(derived['abs_z_center_proxy']):.3f}"
    )
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Weak-interaction PMNS first-row audit (Step 8.7.22)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8.7.22: PMNS first-row quantitative audit")
    parser.add_argument(
        "--source-url",
        type=str,
        default="http://www.nu-fit.org/sites/default/files/v52.tbl-mixing.pdf",
        help="Primary source URL for NuFIT v5.2 leptonic mixing matrix table PDF.",
    )
    parser.add_argument(
        "--in-pdf",
        type=Path,
        default=ROOT / "data" / "quantum" / "sources" / "nufit_v52_tbl_mixing.pdf",
        help="Local cache path for NuFIT v5.2 PMNS table PDF.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "output" / "public" / "quantum",
        help="Output directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("with_sk_atm", "without_sk_atm"),
        default="with_sk_atm",
        help="Dataset used for PMNS gate evaluation.",
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
    text = _normalize_text(_extract_pdf_text(in_pdf))

    ranges_without = _extract_matrix_ranges(
        text,
        marker="|U|w/o SK-atm",
        end_marker="|U|with SK-atm",
    )
    ranges_with = _extract_matrix_ranges(
        text,
        marker="|U|with SK-atm",
        end_marker=None,
    )

    metrics_without = _dataset_metrics(ranges_without)
    metrics_with = _dataset_metrics(ranges_with)
    metrics_by_dataset = {
        "without_sk_atm": metrics_without,
        "with_sk_atm": metrics_with,
    }
    selected = metrics_by_dataset[str(args.dataset)]
    selected_derived = selected["derived"]

    abs_z = float(selected_derived["abs_z_center_proxy"])
    hard_pass = bool(math.isfinite(abs_z) and abs_z <= float(args.hard_z_threshold))
    watch_pass = bool(math.isfinite(abs_z) and abs_z <= float(args.watch_z_threshold))
    # 条件分岐: `hard_pass and watch_pass` を満たす経路を評価する。
    if hard_pass and watch_pass:
        status = "pass"
    # 条件分岐: 前段条件が不成立で、`hard_pass` を追加評価する。
    elif hard_pass:
        status = "watch"
    else:
        status = "reject"

    out_json = out_dir / "weak_interaction_pmns_first_row_audit.json"
    out_csv = out_dir / "weak_interaction_pmns_first_row_audit_summary.csv"
    out_png = out_dir / "weak_interaction_pmns_first_row_audit.png"

    summary_rows = []
    for label, metrics in (("without_sk_atm", metrics_without), ("with_sk_atm", metrics_with)):
        row = metrics["first_row_elements"]
        derived = metrics["derived"]
        row_entry = {
            "dataset": label,
            "Ue1_low_3sigma": float(row["Ue1"]["low_3sigma"]),
            "Ue1_high_3sigma": float(row["Ue1"]["high_3sigma"]),
            "Ue2_low_3sigma": float(row["Ue2"]["low_3sigma"]),
            "Ue2_high_3sigma": float(row["Ue2"]["high_3sigma"]),
            "Ue3_low_3sigma": float(row["Ue3"]["low_3sigma"]),
            "Ue3_high_3sigma": float(row["Ue3"]["high_3sigma"]),
            "first_row_sum_center_proxy": float(derived["first_row_sum_center_proxy"]),
            "first_row_sigma_uncorrelated_proxy": float(derived["first_row_sigma_uncorrelated_proxy"]),
            "delta_pmns_center_proxy": float(derived["delta_pmns_center_proxy"]),
            "abs_z_center_proxy": float(derived["abs_z_center_proxy"]),
            "first_row_sum_low_3sigma_corner": float(derived["first_row_sum_low_3sigma_corner"]),
            "first_row_sum_high_3sigma_corner": float(derived["first_row_sum_high_3sigma_corner"]),
            "hard_z_threshold": float(args.hard_z_threshold),
            "watch_z_threshold": float(args.watch_z_threshold),
            "hard_pass": bool(label == str(args.dataset) and hard_pass),
            "watch_pass": bool(label == str(args.dataset) and watch_pass),
            "status": status if label == str(args.dataset) else "reference_only",
        }
        summary_rows.append(row_entry)

    _write_csv(out_csv, summary_rows)

    _build_plot(
        out_png=out_png,
        selected_label=str(args.dataset),
        selected_metrics=selected,
        hard_z_threshold=float(args.hard_z_threshold),
        watch_z_threshold=float(args.watch_z_threshold),
    )

    payload = {
        "generated_utc": _iso_now(),
        "phase": 8,
        "step": "8.7.22",
        "title": "Weak-interaction PMNS first-row audit",
        "source": {
            "url": str(args.source_url),
            "local_pdf": _rel(in_pdf),
            "sha256": _sha256(in_pdf),
        },
        "extraction": {
            "dataset_without_sk_atm_3sigma_ranges": ranges_without,
            "dataset_with_sk_atm_3sigma_ranges": ranges_with,
        },
        "derived": {
            "without_sk_atm": metrics_without["derived"],
            "with_sk_atm": metrics_with["derived"],
        },
        "gate": {
            "dataset": str(args.dataset),
            "hard_z_threshold": float(args.hard_z_threshold),
            "watch_z_threshold": float(args.watch_z_threshold),
            "hard_pass": hard_pass,
            "watch_pass": watch_pass,
            "status": status,
            "rule": "NuFIT v5.2 row-1 3σ range proxies; sigma_proxy=(high-low)/6; hard if abs(z)<=3, watch if abs(z)<=2.",
        },
        "outputs": {
            "summary_csv": _rel(out_csv),
            "audit_json": _rel(out_json),
            "audit_png": _rel(out_png),
        },
        "notes": [
            "This is an operational PMNS closure gate using NuFIT v5.2 row-1 3σ ranges.",
            "Row-1 center values are midpoint proxies and the uncertainty is propagated as uncorrelated sigma proxies.",
            "CKM+PMNS combined closure is evaluated in weak_interaction_beta_decay_route_ab_audit.py.",
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    worklog.append_event(
        {
            "type": "step_update",
            "step": "8.7.22",
            "script": "scripts/quantum/weak_interaction_pmns_first_row_audit.py",
            "inputs": [_rel(in_pdf)],
            "outputs": [_rel(out_csv), _rel(out_json), _rel(out_png)],
            "gate_status": status,
            "abs_z_center_proxy": abs_z,
            "dataset": str(args.dataset),
        }
    )

    print("[ok] wrote:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print(f"  {out_png}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

