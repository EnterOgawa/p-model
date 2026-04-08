"""
目的: 重力波 topic の gw multi event summary に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.utils.plot_style import (  # noqa: E402
    apply_paper_style,
    apply_wavep_figure_layout,
    get_wavep_font_size,
    resolve_wavep_cjk_font_family,
)


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        chosen = resolve_wavep_cjk_font_family()
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = [chosen, "DejaVu Sans"]
        mpl.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# クラス: `Point` の責務と境界条件を定義する。

@dataclass(frozen=True)
class Point:
    event: str
    slug: str
    detector: str
    preprocess: str
    method: str
    r2: Optional[float]
    match: Optional[float]

    # 関数: `to_dict` の入出力契約と処理意図を定義する。
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "slug": self.slug,
            "detector": self.detector,
            "preprocess": self.preprocess,
            "method": self.method,
            "r2": self.r2,
            "match": self.match,
        }


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    # 条件分岐: `not (v == v)` を満たす経路を評価する。

    if not (v == v):  # NaN
        return None

    return v


# 関数: `_load_event_list` の入出力契約と処理意図を定義する。

def _load_event_list(root: Path) -> List[Dict[str, Any]]:
    path = root / "data" / "gw" / "event_list.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return []

    try:
        obj = _read_json(path)
    except Exception:
        return []

    events = obj.get("events") if isinstance(obj, dict) else None
    # 条件分岐: `not isinstance(events, list)` を満たす経路を評価する。
    if not isinstance(events, list):
        return []

    out: List[Dict[str, Any]] = []
    for e in events:
        # 条件分岐: `isinstance(e, dict)` を満たす経路を評価する。
        if isinstance(e, dict):
            out.append(e)

    return out


# 関数: `_load_default_event_pairs` の入出力契約と処理意図を定義する。

def _load_default_event_pairs(root: Path) -> List[Tuple[str, str]]:
    events = _load_event_list(root)
    out: List[Tuple[str, str]] = []
    for e in events:
        name = str(e.get("name") or "").strip()
        # 条件分岐: `not name` を満たす経路を評価する。
        if not name:
            continue

        slug = str(e.get("slug") or name.lower()).strip() or name.lower()
        out.append((name, slug))

    return out


# 関数: `_load_event_meta_by_slug` の入出力契約と処理意図を定義する。

def _load_event_meta_by_slug(root: Path) -> Dict[str, Dict[str, float]]:
    events = _load_event_list(root)
    out: Dict[str, Dict[str, float]] = {}
    for e in events:
        name = str(e.get("name") or "").strip()
        # 条件分岐: `not name` を満たす経路を評価する。
        if not name:
            continue

        slug = str(e.get("slug") or name.lower()).strip() or name.lower()
        meta = e.get("meta")
        # 条件分岐: `not isinstance(meta, dict)` を満たす経路を評価する。
        if not isinstance(meta, dict):
            continue

        snr = _safe_float(meta.get("network_snr"))
        far = _safe_float(meta.get("far_yr"))
        p_astro = _safe_float(meta.get("p_astro"))
        m: Dict[str, float] = {}
        # 条件分岐: `snr is not None` を満たす経路を評価する。
        if snr is not None:
            m["network_snr"] = float(snr)

        # 条件分岐: `far is not None` を満たす経路を評価する。

        if far is not None:
            m["far_yr"] = float(far)

        # 条件分岐: `p_astro is not None` を満たす経路を評価する。

        if p_astro is not None:
            m["p_astro"] = float(p_astro)

        # 条件分岐: `m` を満たす経路を評価する。

        if m:
            out[slug] = m

    return out


# 関数: `_fmt_g` の入出力契約と処理意図を定義する。

def _fmt_g(x: Optional[float], *, digits: int = 3) -> str:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return ""

    return f"{float(x):.{int(digits)}g}"


# 関数: `_collect_points` の入出力契約と処理意図を定義する。

def _collect_points(
    root: Path, events: Sequence[Tuple[str, str]]
) -> Tuple[List[Point], List[str], List[Optional[Tuple[float, float]]], Dict[str, int]]:
    points: List[Point] = []
    used_paths: List[str] = []
    wave_franges: List[Optional[Tuple[float, float]]] = []
    match_omitted_by_reason: Dict[str, int] = {}
    for name, slug in events:
        path = root / "output" / "private" / "gw" / f"{slug}_chirp_phase_metrics.json"
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            continue

        j = _read_json(path)
        fr: Optional[Tuple[float, float]] = None
        params = j.get("params") or {}
        # 条件分岐: `isinstance(params, dict)` を満たす経路を評価する。
        if isinstance(params, dict):
            wf = params.get("wave_frange_hz")
            # 条件分岐: `isinstance(wf, list) and len(wf) == 2` を満たす経路を評価する。
            if isinstance(wf, list) and len(wf) == 2:
                flo = _safe_float(wf[0])
                fhi = _safe_float(wf[1])
                # 条件分岐: `flo is not None and fhi is not None and flo > 0 and fhi > 0` を満たす経路を評価する。
                if flo is not None and fhi is not None and flo > 0 and fhi > 0:
                    fr = (float(flo), float(fhi)) if flo <= fhi else (float(fhi), float(flo))

        dets = j.get("detectors") or []
        # 条件分岐: `not isinstance(dets, list)` を満たす経路を評価する。
        if not isinstance(dets, list):
            continue

        used_paths.append(str(path).replace("\\", "/"))
        wave_franges.append(fr)
        for d in dets:
            # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
            if not isinstance(d, dict):
                continue

            det = str(d.get("detector") or "")
            preprocess = str(d.get("preprocess") or "")
            method = str(d.get("method_used") or "")
            fit = d.get("fit") if isinstance(d.get("fit"), dict) else {}
            wf = d.get("waveform_fit") if isinstance(d.get("waveform_fit"), dict) else {}
            match = _safe_float(wf.get("overlap"))
            # 条件分岐: `match is None and isinstance(wf, dict)` を満たす経路を評価する。
            if match is None and isinstance(wf, dict):
                reason = str(wf.get("reason") or "").strip()
                # 条件分岐: `reason` を満たす経路を評価する。
                if reason:
                    match_omitted_by_reason[reason] = int(match_omitted_by_reason.get(reason, 0)) + 1

            points.append(
                Point(
                    event=name,
                    slug=slug,
                    detector=det,
                    preprocess=preprocess,
                    method=method,
                    r2=_safe_float(fit.get("r2")),
                    match=match,
                )
            )

    return points, used_paths, wave_franges, match_omitted_by_reason


# 関数: `_detector_order` の入出力契約と処理意図を定義する。

def _detector_order(dets: List[str]) -> List[str]:
    preferred = ["H1", "L1", "V1", "K1"]
    seen = []
    for d in preferred:
        # 条件分岐: `d in dets` を満たす経路を評価する。
        if d in dets:
            seen.append(d)

    for d in dets:
        # 条件分岐: `d not in seen` を満たす経路を評価する。
        if d not in seen:
            seen.append(d)

    return seen


# 関数: `_plot_summary` の入出力契約と処理意図を定義する。

def _plot_summary(
    *,
    points: List[Point],
    events: List[str],
    tick_labels: Optional[List[str]],
    out_png: Path,
    title: str,
    public: bool,
    wave_frange_hz: Optional[Tuple[float, float]] = None,
    match_omitted_by_reason: Optional[Dict[str, int]] = None,
) -> None:
    apply_paper_style()
    _set_japanese_font()

    dets = _detector_order(sorted({p.detector for p in points if p.detector}))
    ev_index = {ev: i for i, ev in enumerate(events)}

    fig, axes = plt.subplots(2, 1, sharex=True)
    apply_wavep_figure_layout(fig, template="part2_two_panel_dense_x")
    ax_r2, ax_m = axes[0], axes[1]

    x_base = list(range(len(events)))
    n_det = max(1, len(dets))
    # Spread points slightly to avoid overlap (detector-wise offsets).
    offsets = [0.0] if n_det == 1 else [(-0.25 + 0.5 * i / (n_det - 1)) for i in range(n_det)]

    colors = {
        "H1": "#1f77b4",
        "L1": "#ff7f0e",
        "V1": "#2ca02c",
        "K1": "#d62728",
    }

    for det_i, det in enumerate(dets):
        col = colors.get(det, None)
        xs_r2: List[float] = []
        ys_r2: List[float] = []
        xs_m: List[float] = []
        ys_m: List[float] = []
        for p in points:
            # 条件分岐: `p.detector != det` を満たす経路を評価する。
            if p.detector != det:
                continue

            # 条件分岐: `p.event not in ev_index` を満たす経路を評価する。

            if p.event not in ev_index:
                continue

            x = float(ev_index[p.event]) + float(offsets[det_i])
            # 条件分岐: `p.r2 is not None` を満たす経路を評価する。
            if p.r2 is not None:
                xs_r2.append(x)
                ys_r2.append(float(p.r2))

            # 条件分岐: `p.match is not None` を満たす経路を評価する。

            if p.match is not None:
                xs_m.append(x)
                ys_m.append(float(p.match))

        # 条件分岐: `xs_r2` を満たす経路を評価する。

        if xs_r2:
            ax_r2.plot(
                xs_r2,
                ys_r2,
                marker="o",
                linestyle="None",
                color=col,
                label=det,
                markersize=4.8,
            )

        # 条件分岐: `xs_m` を満たす経路を評価する。

        if xs_m:
            ax_m.plot(
                xs_m,
                ys_m,
                marker="o",
                linestyle="None",
                color=col,
                label=det,
                markersize=4.8,
            )

    for ax in (ax_r2, ax_m):
        ax.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

    ax_r2.set_ylabel("R^2")
    ax_r2.set_ylim(-0.05, 1.05)
    # 条件分岐: `wave_frange_hz is not None` を満たす経路を評価する。
    if wave_frange_hz is not None:
        ax_m.set_ylabel(
            f"match ({wave_frange_hz[0]:g}..{wave_frange_hz[1]:g} Hz)",
        )
    else:
        ax_m.set_ylabel("match")

    ax_m.set_ylim(-0.05, 1.05)
    ax_r2.margins(x=0.03)
    ax_m.margins(x=0.03)

    ax_m.set_xticks(x_base)
    xlabels = tick_labels if (isinstance(tick_labels, list) and len(tick_labels) == len(events)) else events
    ax_m.set_xticklabels(xlabels, rotation=0, ha="center")
    for lab in ax_m.get_xticklabels():
        lab.set_fontsize(get_wavep_font_size("tick"))
        lab.set_linespacing(1.0)

    ax_m.tick_params(axis="x", pad=6)

    fig.suptitle(title, y=0.975, fontsize=get_wavep_font_size("suptitle"))

    # Legend: show once (top panel)
    if dets:
        ax_r2.legend(loc="upper right", frameon=True, fontsize=get_wavep_font_size("legend"))

    # 図下注記は論文本文側へ移し、図中の重なりを回避する。

    bottom = 0.135
    # 条件分岐: `isinstance(xlabels, list)` を満たす経路を評価する。
    if isinstance(xlabels, list):
        try:
            max_lines = max(1 + str(s).count("\n") for s in xlabels)
        except Exception:
            max_lines = 1

        bottom = min(0.225, 0.135 + 0.034 * max(0, max_lines - 1))

    fig.subplots_adjust(bottom=bottom)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(out_png, dpi=220)
        fig.savefig(out_png.with_suffix(".pdf"))

    plt.close(fig)


# 関数: `_plot_placeholder` の入出力契約と処理意図を定義する。

def _plot_placeholder(out_png: Path, *, title: str) -> None:
    apply_paper_style()
    _set_japanese_font()
    fig = plt.figure()
    apply_wavep_figure_layout(fig, template="part2_single_panel")
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=get_wavep_font_size("title"))
    ax.text(
        0.5,
        0.4,
        "出力未生成: output/private/gw/*_chirp_phase_metrics.json が見つかりません。\n"
        "先に scripts/gw/gw150914_chirp_phase.py を実行してください。",
        ha="center",
        va="center",
        fontsize=get_wavep_font_size("note"),
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(out_png, dpi=220)
        fig.savefig(out_png.with_suffix(".pdf"))

    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    defaults = _load_default_event_pairs(root)
    # 条件分岐: `not defaults` を満たす経路を評価する。
    if not defaults:
        defaults = [
            ("GW150914", "gw150914"),
            ("GW151226", "gw151226"),
            ("GW170104", "gw170104"),
            ("GW170817", "gw170817"),
            ("GW190425", "gw190425"),
        ]

    ap = argparse.ArgumentParser(description="Summarize GW multi-event chirp consistency metrics (R^2/match).")
    ap.add_argument(
        "--events",
        type=str,
        default=",".join([n for n, _ in defaults]),
        help="Comma-separated event names (default: GW150914,GW151226,GW170104,GW170817,GW190425).",
    )
    ap.add_argument(
        "--slugs",
        type=str,
        default=",".join([s for _, s in defaults]),
        help="Comma-separated slugs for output/private/gw/*_chirp_phase_metrics.json (must match --events order).",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="output/private/gw",
        help="Output directory (default: output/private/gw).",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="gw_multi_event_summary",
        help="Output file prefix (default: gw_multi_event_summary).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    events = [s.strip() for s in str(args.events).split(",") if s.strip()]
    slugs = [s.strip() for s in str(args.slugs).split(",") if s.strip()]
    # 条件分岐: `len(events) != len(slugs)` を満たす経路を評価する。
    if len(events) != len(slugs):
        print("[err] --events and --slugs must have the same length.", file=sys.stderr)
        return 2

    ev_pairs = list(zip(events, slugs))
    meta_by_slug = _load_event_meta_by_slug(root)
    tick_labels = []
    tick_labels_public = []
    event_meta_rows: List[Dict[str, Any]] = []
    for name, slug in ev_pairs:
        meta = meta_by_slug.get(slug) or {}
        snr = _safe_float(meta.get("network_snr"))
        far = _safe_float(meta.get("far_yr"))

        if "_" in name:
            head, tail = name.split("_", 1)
            lines_compact = [head, tail]
        else:
            lines_compact = [name]

        tick_labels.append("\n".join(lines_compact))
        tick_labels_public.append("\n".join(lines_compact))

        # 条件分岐: `meta` を満たす経路を評価する。
        if meta:
            event_meta_rows.append({"event": name, "slug": slug, **meta})

    out_dir = Path(args.outdir)
    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    out_png = out_dir / f"{args.prefix}.png"
    out_png_public = out_dir / f"{args.prefix}_public.png"
    out_json = out_dir / f"{args.prefix}_metrics.json"
    public_dir = root / "output" / "public" / "gw"
    summary_dir = root / "output" / "private" / "summary" / "figures"

    points, used_paths, wave_franges, match_omitted_by_reason = _collect_points(root, ev_pairs)
    fr_unique = {fr for fr in wave_franges if fr is not None}
    wave_frange_hz: Optional[Tuple[float, float]] = None
    # 条件分岐: `fr_unique and len(fr_unique) == 1` を満たす経路を評価する。
    if fr_unique and len(fr_unique) == 1:
        wave_frange_hz = next(iter(fr_unique))

    # 条件分岐: `points` を満たす経路を評価する。

    if points:
        title = "重力波（複数イベント）：R^2 と match の要約"
        # 条件分岐: `wave_frange_hz is not None` を満たす経路を評価する。
        if wave_frange_hz is not None:
            title = f"重力波（複数イベント）：R^2 と match ({wave_frange_hz[0]:g}..{wave_frange_hz[1]:g} Hz)"

        _plot_summary(
            points=points,
            events=events,
            tick_labels=tick_labels,
            out_png=out_png,
            title=title,
            public=False,
            wave_frange_hz=wave_frange_hz,
            match_omitted_by_reason=match_omitted_by_reason,
        )
        _plot_summary(
            points=points,
            events=events,
            tick_labels=tick_labels_public,
            out_png=out_png_public,
            title="重力波（複数イベント）：観測と単純モデルの一致度",
            public=True,
            wave_frange_hz=wave_frange_hz,
            match_omitted_by_reason=match_omitted_by_reason,
        )
    else:
        _plot_placeholder(out_png, title="重力波（複数イベント）要約")
        _plot_placeholder(out_png_public, title="重力波（複数イベント）要約")

    out_pdf = out_png.with_suffix(".pdf")
    out_pdf_public = out_png_public.with_suffix(".pdf")
    canonical_public_png = public_dir / f"{args.prefix}.png"
    canonical_public_pdf = public_dir / f"{args.prefix}.pdf"
    canonical_summary_png = summary_dir / f"{args.prefix}.png"
    canonical_summary_pdf = summary_dir / f"{args.prefix}.pdf"
    public_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_png_public, canonical_public_png)
    shutil.copy2(out_pdf_public, canonical_public_pdf)
    shutil.copy2(out_png_public, canonical_summary_png)
    shutil.copy2(out_pdf_public, canonical_summary_pdf)

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "inputs": {
            "events": events,
            "slugs": slugs,
            "source_metrics": used_paths,
            **({"wave_frange_hz": list(wave_frange_hz)} if wave_frange_hz is not None else {}),
        },
        "outputs": {
            "png": str(out_png).replace("\\", "/"),
            "pdf": str(out_pdf).replace("\\", "/"),
            "public_png": str(out_png_public).replace("\\", "/"),
            "public_pdf": str(out_pdf_public).replace("\\", "/"),
            "canonical_public_png": str(canonical_public_png).replace("\\", "/"),
            "canonical_public_pdf": str(canonical_public_pdf).replace("\\", "/"),
            "summary_png": str(canonical_summary_png).replace("\\", "/"),
            "summary_pdf": str(canonical_summary_pdf).replace("\\", "/"),
            "metrics_json": str(out_json).replace("\\", "/"),
        },
        **({"event_meta": event_meta_rows} if event_meta_rows else {}),
        "match_omitted_by_reason": match_omitted_by_reason,
        "rows": [p.to_dict() for p in points],
    }
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "event_type": "gw_multi_event_summary",
                "argv": list(sys.argv),
                "inputs": {"source_metrics": used_paths},
                "outputs": {
                    "png": out_png,
                    "pdf": out_pdf,
                    "public_png": out_png_public,
                    "public_pdf": out_pdf_public,
                    "canonical_public_png": canonical_public_png,
                    "canonical_public_pdf": canonical_public_pdf,
                    "summary_png": canonical_summary_png,
                    "summary_pdf": canonical_summary_pdf,
                    "metrics_json": out_json,
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] pub : {out_png_public}")
    print(f"[ok] public canonical : {canonical_public_png}")
    print(f"[ok] summary canonical: {canonical_summary_png}")
    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
