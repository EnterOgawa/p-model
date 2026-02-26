from __future__ import annotations

import argparse
import json
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


def _repo_root() -> Path:
    return _ROOT


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class Point:
    event: str
    slug: str
    detector: str
    preprocess: str
    method: str
    r2: Optional[float]
    match: Optional[float]

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


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    # 条件分岐: `not (v == v)` を満たす経路を評価する。

    if not (v == v):  # NaN
        return None

    return v


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


def _fmt_g(x: Optional[float], *, digits: int = 3) -> str:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return ""

    return f"{float(x):.{int(digits)}g}"


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
    _set_japanese_font()

    dets = _detector_order(sorted({p.detector for p in points if p.detector}))
    ev_index = {ev: i for i, ev in enumerate(events)}

    # Layout: two panels (R^2 and match)
    figsize = (13.5, 7.2) if public else (12.5, 6.2)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
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
            ax_r2.plot(xs_r2, ys_r2, marker="o", linestyle="None", color=col, label=det)

        # 条件分岐: `xs_m` を満たす経路を評価する。

        if xs_m:
            ax_m.plot(xs_m, ys_m, marker="o", linestyle="None", color=col, label=det)

    for ax in (ax_r2, ax_m):
        ax.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

    ax_r2.set_ylabel("R^2（1に近いほど良い）")
    ax_r2.set_ylim(-0.05, 1.05)
    # 条件分岐: `wave_frange_hz is not None` を満たす経路を評価する。
    if wave_frange_hz is not None:
        ax_m.set_ylabel(f"match（{wave_frange_hz[0]:g}..{wave_frange_hz[1]:g} Hz, 1に近いほど良い）")
    else:
        ax_m.set_ylabel("match（窓内, 1に近いほど良い）")

    ax_m.set_ylim(-0.05, 1.05)

    ax_m.set_xticks(x_base)
    xlabels = tick_labels if (isinstance(tick_labels, list) and len(tick_labels) == len(events)) else events
    ax_m.set_xticklabels(xlabels, rotation=0)
    for lab in ax_m.get_xticklabels():
        lab.set_fontsize(9.0 if public else 8.5)

    fig.suptitle(title, y=0.98)

    # Legend: show once (top panel)
    if dets:
        ax_r2.legend(loc="upper right", frameon=True, fontsize=10 if public else 9)

    foot_parts = [
        "R^2: t=t_c−A f^(−8/3)（四重極チャープ則, Newton近似）への当てはめの決定係数。",
        "match: 観測波形と単純テンプレートの一致度（正規化内積, 参考）。",
    ]
    # 条件分岐: `wave_frange_hz is not None` を満たす経路を評価する。
    if wave_frange_hz is not None:
        foot_parts.append(f"match窓: 周波数帯 {wave_frange_hz[0]:g}..{wave_frange_hz[1]:g} Hz に対応する時刻範囲。")
    else:
        foot_parts.append("※イベント/検出器により前処理（bandpass/whiten）や窓が異なる。")

    # 条件分岐: `public` を満たす経路を評価する。

    if public:
        foot_parts.append("SNR: GWOSC公開メタ情報（network matched-filter SNR）。")
    else:
        foot_parts.append("SNR/FAR: GWOSC公開メタ情報（SNR=network matched-filter, FAR=誤警報率[1/yr]）。")

    # 条件分岐: `match_omitted_by_reason` を満たす経路を評価する。

    if match_omitted_by_reason:
        n_omit = int(sum(int(v) for v in match_omitted_by_reason.values()))
        # 条件分岐: `n_omit > 0` を満たす経路を評価する。
        if n_omit > 0:
            foot_parts.append(f"※短窓などでmatchを省略: {n_omit}件（過大評価防止）。")

    foot_parts.append("※公式テンプレート解析の代替ではない。")
    foot = " ".join(foot_parts)
    fig.text(0.5, 0.02, foot, ha="center", va="bottom", fontsize=10 if public else 9)

    bottom = 0.05
    # 条件分岐: `isinstance(xlabels, list)` を満たす経路を評価する。
    if isinstance(xlabels, list):
        try:
            max_lines = max(1 + str(s).count("\n") for s in xlabels)
        except Exception:
            max_lines = 1

        bottom = min(0.2, 0.05 + 0.03 * max(0, max_lines - 1))

    fig.tight_layout(rect=(0.02, bottom, 0.98, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _plot_placeholder(out_png: Path, *, title: str) -> None:
    _set_japanese_font()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=16)
    ax.text(
        0.5,
        0.4,
        "出力未生成: output/private/gw/*_chirp_phase_metrics.json が見つかりません。\n"
        "先に scripts/gw/gw150914_chirp_phase.py を実行してください。",
        ha="center",
        va="center",
        fontsize=12,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


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

        lines = [name]
        # 条件分岐: `snr is not None` を満たす経路を評価する。
        if snr is not None:
            lines.append(f"SNR={_fmt_g(snr, digits=3)}")

        # 条件分岐: `far is not None` を満たす経路を評価する。

        if far is not None:
            lines.append(f"FAR={_fmt_g(far, digits=2)}/yr")

        tick_labels.append("\n".join(lines))

        lines_pub = [name]
        # 条件分岐: `snr is not None` を満たす経路を評価する。
        if snr is not None:
            lines_pub.append(f"SNR={_fmt_g(snr, digits=3)}")

        tick_labels_public.append("\n".join(lines_pub))

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

    points, used_paths, wave_franges, match_omitted_by_reason = _collect_points(root, ev_pairs)
    fr_unique = {fr for fr in wave_franges if fr is not None}
    wave_frange_hz: Optional[Tuple[float, float]] = None
    # 条件分岐: `fr_unique and len(fr_unique) == 1` を満たす経路を評価する。
    if fr_unique and len(fr_unique) == 1:
        wave_frange_hz = next(iter(fr_unique))

    # 条件分岐: `points` を満たす経路を評価する。

    if points:
        title = "重力波（複数イベント）：chirp整合（R^2）と波形match（窓内）"
        # 条件分岐: `wave_frange_hz is not None` を満たす経路を評価する。
        if wave_frange_hz is not None:
            title = f"重力波（複数イベント）：chirp整合（R^2）と波形match（{wave_frange_hz[0]:g}..{wave_frange_hz[1]:g} Hz）"

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
            title="重力波（複数イベント）：観測と単純モデルの一致度（要約）",
            public=True,
            wave_frange_hz=wave_frange_hz,
            match_omitted_by_reason=match_omitted_by_reason,
        )
    else:
        _plot_placeholder(out_png, title="重力波（複数イベント）要約")
        _plot_placeholder(out_png_public, title="重力波（複数イベント）要約")

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
            "public_png": str(out_png_public).replace("\\", "/"),
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
                "outputs": {"png": out_png, "public_png": out_png_public, "metrics_json": out_json},
            }
        )
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] pub : {out_png_public}")
    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
