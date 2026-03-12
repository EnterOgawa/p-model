from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_timeline_rows` の入出力契約と処理意図を定義する。
def _timeline_rows() -> List[Dict[str, Any]]:
    return [
        {"label": "重力波偏光\n(LIGO O5)", "year": 2027.5, "span": "2027-2030"},
        {"label": "距離指標前提検証\n(標準サイレン/JWST)", "year": 2028.5, "span": "2027-2030"},
        {"label": "宇宙論 ln(1+z)\n(DESI/Euclid/Roman)", "year": 2030.0, "span": "2027-2035"},
        {"label": "ブラックホール影\n(ngEHT/BHEX)", "year": 2030.5, "span": "2030-2031"},
        {"label": "マクロ量子干渉\n(MAQRO型)", "year": 2035.5, "span": "2035+"},
    ]


# 関数: `_render_timeline` の入出力契約と処理意図を定義する。
def _render_timeline(*, rows: List[Dict[str, Any]], out_pdf: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.2), dpi=180)
    ax.set_xlim(2026.5, 2036.5)
    y_step = 1.9
    ys = [idx * y_step for idx in range(len(rows))]
    ax.set_ylim(-0.9, ys[-1] + 1.1)
    ax.hlines(y=-0.1, xmin=2027, xmax=2036, color="#666666", linewidth=1.4, zorder=1)

    years = [float(row["year"]) for row in rows]
    ax.scatter(years, ys, s=85, color="#0B5FA5", zorder=3)

    for idx, row in enumerate(rows):
        year = float(row["year"])
        y = ys[idx]
        label = str(row["label"])
        span = str(row["span"])
        ax.text(year + 0.08, y + 0.28, label, fontsize=12.0, va="center", ha="left")
        ax.text(year + 0.08, y - 0.28, f"決着時期: {span}", fontsize=10.6, va="center", ha="left", color="#333333")

    ax.set_title("Part V: 未来観測による決着タイムライン（P-model 差分予測）", fontsize=15.0, pad=10.0)
    ax.set_xlabel("年（UTC）", fontsize=12.5)
    ax.set_yticks([])
    ax.set_xticks([2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036])
    ax.tick_params(axis="x", labelsize=11.2)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。
def main() -> int:
    root = _repo_root()
    out_dir = root / "output" / "private" / "summary" / "figures"
    out_pdf = out_dir / "part5_future_predictions_timeline.pdf"
    out_png = out_dir / "part5_future_predictions_timeline.png"
    out_json = root / "output" / "private" / "summary" / "part5_future_predictions_timeline_metrics.json"

    rows = _timeline_rows()
    _render_timeline(rows=rows, out_pdf=out_pdf, out_png=out_png)

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "outputs": {
            "timeline_pdf": str(out_pdf).replace("\\", "/"),
            "timeline_png": str(out_png).replace("\\", "/"),
        },
        "rows": rows,
        "notes": [
            "Part V の未来予測タイムライン図（PDFベクター正本）。",
            "数値は Part II/Part IV の差分予測方針を一般向けに再整理した目安。"
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] pdf : {out_pdf}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。
if __name__ == "__main__":
    raise SystemExit(main())
