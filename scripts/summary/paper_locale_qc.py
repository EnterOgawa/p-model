#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_locale_qc.py

locale 別論文 build の「二重チェック」を行う。

目的:
- source / manifest 側で、locale ごとの source 切り替えが壊れていないかを確認する。
- 生成済み TeX / PDF 側で、図参照や出力名が locale 間で混線していないかを確認する。

モード:
- manifest: manifest と source path だけを監査する
- surface: 生成済み TeX / PDF と図参照だけを監査する
- all: 上の両方を監査する
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import paper_locale_registry as locale_registry  # noqa: E402
from scripts.summary import paper_profile_content as profile_content  # noqa: E402
from scripts.summary import worklog  # noqa: E402


_GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_GRAPHICSPATH_RE = re.compile(r"\\graphicspath\{((?:\{[^}]+\})+)\}")
_GRAPHICSPATH_ENTRY_RE = re.compile(r"\{([^}]+)\}")
_JAPANESE_CHAR_RE = re.compile(r"[ぁ-ゖァ-ヺ一-龯々〆〤]")
_IMAGE_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg")
_DEFAULT_OUTDIR = _ROOT / "output" / "private" / "summary"
_DEFAULT_PUBLIC_ROOT = _ROOT / "output" / "public"
_DEFAULT_PRIVATE_ROOT = _ROOT / "output" / "private"
_KNOWN_PDFTOTEXT_PATHS = (
    Path(r"C:\texlive\2024\bin\windows\pdftotext.exe"),
    Path(r"C:\texlive\2023\bin\windows\pdftotext.exe"),
)


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            # 条件分岐: `not block` を満たす経路を評価する。
            if not block:
                break

            h.update(block)

    return h.hexdigest()


# 関数: `_collect_language_hits` の入出力契約と処理意図を定義する。

def _collect_language_hits(text: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not _JAPANESE_CHAR_RE.search(line):
            continue

        hits.append({"line": lineno, "text": line.strip()[:180]})
        if len(hits) >= limit:
            break

    return hits


# 関数: `_find_pdftotext` の入出力契約と処理意図を定義する。

def _find_pdftotext() -> str | None:
    if _find_pdftotext._checked:
        return _find_pdftotext._cached

    candidate = shutil.which("pdftotext") or shutil.which("pdftotext.exe")
    if candidate:
        _find_pdftotext._cached = candidate
        _find_pdftotext._checked = True
        return candidate

    for known_path in _KNOWN_PDFTOTEXT_PATHS:
        if known_path.exists():
            _find_pdftotext._cached = str(known_path)
            _find_pdftotext._checked = True
            return _find_pdftotext._cached

    _find_pdftotext._checked = True
    _find_pdftotext._cached = None
    return None


_find_pdftotext._checked = False
_find_pdftotext._cached = None


# 関数: `_extract_pdf_text` の入出力契約と処理意図を定義する。
def _extract_pdf_text(path: Path) -> str | None:
    resolved = path.resolve()
    if resolved in _extract_pdf_text._cache:
        return _extract_pdf_text._cache[resolved]

    tool_path = _find_pdftotext()
    if tool_path is None:
        _extract_pdf_text._cache[resolved] = None
        return None

    try:
        completed = subprocess.run(
            [tool_path, str(resolved), "-"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except Exception:
        _extract_pdf_text._cache[resolved] = None
        return None

    text = completed.stdout if completed.returncode == 0 else None
    _extract_pdf_text._cache[resolved] = text
    return text


_extract_pdf_text._cache = {}


# クラス: `_Issue` の責務と境界条件を定義する。
@dataclass(frozen=True)
class _Issue:
    level: str
    code: str
    detail: Dict[str, Any]


# 関数: `_emit_issue` の入出力契約と処理意図を定義する。

def _emit_issue(bucket: List[_Issue], *, level: str, code: str, **detail: Any) -> None:
    bucket.append(_Issue(level=level, code=code, detail=detail))


# 関数: `_normalize_mode` の入出力契約と処理意図を定義する。

def _normalize_mode(raw: str) -> str:
    candidate = str(raw).strip().lower()
    return candidate if candidate in {"manifest", "surface", "all"} else "all"


# 関数: `_resolve_profiles` の入出力契約と処理意図を定義する。

def _resolve_profiles(raw_profile: str) -> List[str]:
    if raw_profile == "all":
        return list(profile_content.PAPER_PROFILES)

    return [raw_profile]


# 関数: `_is_optional_compat_profile` の入出力契約と処理意図を定義する。

def _is_optional_compat_profile(profile: str) -> bool:
    return profile == profile_content.PART3_COMPAT_PROFILE


# 関数: `_required_manifest_keys` の入出力契約と処理意図を定義する。

def _required_manifest_keys(root: Path) -> List[str]:
    payload = locale_registry.load_manifest(root, locale=locale_registry.DEFAULT_PAPER_LOCALE)
    return sorted(payload.keys())


# 関数: `_load_manifest_payload` の入出力契約と処理意図を定義する。

def _load_manifest_payload(root: Path, locale: str) -> Dict[str, str]:
    return locale_registry.load_manifest(root, locale=locale)


# 関数: `_audit_manifest_source_language` の入出力契約と処理意図を定義する。

def _audit_manifest_source_language(
    *,
    root: Path,
    locale: str,
    profiles: Sequence[str],
    issues: List[_Issue],
) -> List[Dict[str, Any]]:
    if locale == locale_registry.DEFAULT_PAPER_LOCALE:
        return []

    checks: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    for profile in profiles:
        manuscript_path = profile_content.resolve_manuscript_path(root, profile, locale=locale)
        if manuscript_path in seen or not manuscript_path.exists():
            continue

        seen.add(manuscript_path)
        hits = _collect_language_hits(manuscript_path.read_text(encoding="utf-8", errors="replace"))
        checks.append({"profile": profile, "manuscript": _rel(manuscript_path), "hit_count": len(hits)})
        if hits:
            _emit_issue(
                issues,
                level="error",
                code="manifest_source_language_residue",
                locale=locale,
                profile=profile,
                manuscript=_rel(manuscript_path),
                hits=hits,
            )

    return checks


# 関数: `_audit_manifest` の入出力契約と処理意図を定義する。

def _audit_manifest(*, root: Path, locale: str, profiles: Sequence[str], issues: List[_Issue]) -> Dict[str, Any]:
    manifest_path = locale_registry.resolve_manifest_path(root, locale=locale)
    # 条件分岐: `not manifest_path.exists()` を満たす経路を評価する。
    if not manifest_path.exists():
        _emit_issue(
            issues,
            level="error",
            code="manifest_missing",
            locale=locale,
            manifest=_rel(manifest_path),
        )
        return {"manifest": _rel(manifest_path), "ok": False}

    payload = _load_manifest_payload(root, locale)
    reference_keys = _required_manifest_keys(root)
    missing_keys = [key for key in reference_keys if key not in payload]
    for key in missing_keys:
        _emit_issue(
            issues,
            level="error",
            code="manifest_key_missing",
            locale=locale,
            key=key,
            manifest=_rel(manifest_path),
        )

    extra_keys = sorted(set(payload.keys()) - set(reference_keys))
    for key in extra_keys:
        _emit_issue(
            issues,
            level="warning",
            code="manifest_extra_key",
            locale=locale,
            key=key,
            manifest=_rel(manifest_path),
        )

    ja_payload = _load_manifest_payload(root, locale_registry.DEFAULT_PAPER_LOCALE)
    checked_keys: List[Dict[str, Any]] = []
    for key in sorted(payload.keys()):
        path = root / payload[key]
        exists = path.exists()
        checked_keys.append({"key": key, "path": payload[key], "exists": exists})
        # 条件分岐: `not exists` を満たす経路を評価する。
        if not exists:
            _emit_issue(
                issues,
                level="error",
                code="manifest_path_missing",
                locale=locale,
                key=key,
                path=payload[key],
            )
            continue

        if locale != locale_registry.DEFAULT_PAPER_LOCALE and key in ja_payload and payload[key] == ja_payload[key]:
            _emit_issue(
                issues,
                level="warning",
                code="manifest_uses_ja_source",
                locale=locale,
                key=key,
                path=payload[key],
            )

    for profile in profiles:
        ja_tex_name = profile_content.resolve_tex_name(profile, locale=locale_registry.DEFAULT_PAPER_LOCALE)
        ja_pdf_name = profile_content.resolve_pdf_name(profile, locale=locale_registry.DEFAULT_PAPER_LOCALE)
        locale_tex_name = profile_content.resolve_tex_name(profile, locale=locale)
        locale_pdf_name = profile_content.resolve_pdf_name(profile, locale=locale)
        if locale != locale_registry.DEFAULT_PAPER_LOCALE and locale_tex_name == ja_tex_name:
            _emit_issue(
                issues,
                level="error",
                code="localized_tex_name_collision",
                locale=locale,
                profile=profile,
                tex_name=locale_tex_name,
            )

        if locale != locale_registry.DEFAULT_PAPER_LOCALE and locale_pdf_name == ja_pdf_name:
            _emit_issue(
                issues,
                level="error",
                code="localized_pdf_name_collision",
                locale=locale,
                profile=profile,
                pdf_name=locale_pdf_name,
            )

    source_language_checks = _audit_manifest_source_language(root=root, locale=locale, profiles=profiles, issues=issues)
    return {
        "manifest": _rel(manifest_path),
        "required_keys": reference_keys,
        "checked_keys": checked_keys,
        "source_language_checks": source_language_checks,
        "ok": not any(issue.level == "error" for issue in issues if issue.code.startswith("manifest_") or issue.code.endswith("_collision")),
    }


# 関数: `_iter_graphic_refs` の入出力契約と処理意図を定義する。

def _iter_graphic_refs(tex_text: str) -> Iterable[str]:
    for match in _GRAPHICS_RE.finditer(tex_text):
        yield match.group(1).strip()


# 関数: `_resolve_graphic_ref` の入出力契約と処理意図を定義する。

def _resolve_graphic_ref(tex_path: Path, ref: str) -> Path | None:
    ref_path = Path(ref)
    candidates: List[Path] = []
    tex_text = tex_path.read_text(encoding="utf-8", errors="replace")
    graphic_dirs: List[Path] = [tex_path.parent, tex_path.parent / "figures"]
    for match in _GRAPHICSPATH_RE.finditer(tex_text):
        for entry in _GRAPHICSPATH_ENTRY_RE.findall(match.group(1)):
            graphic_dirs.append((tex_path.parent / Path(entry)).resolve())

    if ref_path.suffix:
        candidates.extend(base / ref_path for base in graphic_dirs)
    else:
        for suffix in _IMAGE_SUFFIXES:
            candidates.extend((base / f"{ref}{suffix}") for base in graphic_dirs)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


# 関数: `_gather_same_name_candidates` の入出力契約と処理意図を定義する。

def _gather_same_name_candidates(name: str, *, locale: str) -> Dict[str, List[Path]]:
    cache_key = (locale, name)
    # 条件分岐: `cache_key in _gather_same_name_candidates._cache` を満たす経路を評価する。
    if cache_key in _gather_same_name_candidates._cache:
        return _gather_same_name_candidates._cache[cache_key]

    all_hits: List[Path] = []
    for root in (_DEFAULT_PUBLIC_ROOT, _DEFAULT_PRIVATE_ROOT):
        if root.exists():
            all_hits.extend(path.resolve() for path in root.rglob(name))

    canonical = [path for path in all_hits if f"{Path('locales')}{Path().anchor}" not in str(path)]
    canonical = [path for path in all_hits if f"{Path('locales')}" not in path.parts]
    localized = [
        path
        for path in all_hits
        if "locales" in path.parts and locale in path.parts and path.parts[path.parts.index("locales") + 1] == locale
    ]
    by_other_locale = [
        path
        for path in all_hits
        if "locales" in path.parts and (locale not in path.parts or path.parts[path.parts.index("locales") + 1] != locale)
    ]
    result = {"canonical": canonical, "localized": localized, "other_locale": by_other_locale}
    _gather_same_name_candidates._cache[cache_key] = result
    return result


_gather_same_name_candidates._cache = {}


# 関数: `_hash_matches_any` の入出力契約と処理意図を定義する。
def _hash_matches_any(path: Path, candidates: Sequence[Path]) -> bool:
    if not candidates:
        return False

    base_hash = _sha256(path)
    return any(candidate.exists() and _sha256(candidate) == base_hash for candidate in candidates)


# 関数: `_audit_pdf_language` の入出力契約と処理意図を定義する。

def _audit_pdf_language(
    *,
    path: Path,
    locale: str,
    issues: List[_Issue],
    code: str,
    **detail: Any,
) -> None:
    if locale == locale_registry.DEFAULT_PAPER_LOCALE or path.suffix.lower() != ".pdf":
        return

    text = _extract_pdf_text(path)
    if text is None:
        if not _audit_pdf_language._skip_warning_emitted:
            _emit_issue(
                issues,
                level="warning",
                code="pdf_language_scan_skipped",
                locale=locale,
                reason="pdftotext_unavailable_or_failed",
            )
            _audit_pdf_language._skip_warning_emitted = True

        return

    hits = _collect_language_hits(text)
    if hits:
        _emit_issue(
            issues,
            level="error",
            code=code,
            locale=locale,
            hits=hits,
            **detail,
        )


_audit_pdf_language._skip_warning_emitted = False


# 関数: `_audit_generated_tex_language` の入出力契約と処理意図を定義する。
def _audit_generated_tex_language(*, tex_path: Path, locale: str, issues: List[_Issue]) -> None:
    if locale == locale_registry.DEFAULT_PAPER_LOCALE:
        return

    hits = _collect_language_hits(tex_path.read_text(encoding="utf-8", errors="replace"))
    if hits:
        _emit_issue(
            issues,
            level="error",
            code="generated_tex_language_residue",
            locale=locale,
            tex=_rel(tex_path),
            hits=hits,
        )


# 関数: `_audit_surface_for_profile` の入出力契約と処理意図を定義する。

def _audit_surface_for_profile(
    *,
    profile: str,
    outdir: Path,
    locale: str,
    issues: List[_Issue],
    strict_localized_figures: bool,
) -> Dict[str, Any]:
    tex_path = outdir / profile_content.resolve_tex_name(profile, locale=locale)
    pdf_path = outdir / profile_content.resolve_pdf_name(profile, locale=locale)
    record: Dict[str, Any] = {
        "profile": profile,
        "tex": _rel(tex_path),
        "pdf": _rel(pdf_path),
        "graphics": [],
    }

    if not tex_path.exists():
        level = "warning" if _is_optional_compat_profile(profile) else "error"
        code = "surface_tex_missing_compat" if _is_optional_compat_profile(profile) else "surface_tex_missing"
        _emit_issue(issues, level=level, code=code, locale=locale, profile=profile, tex=_rel(tex_path))
        return record

    if not pdf_path.exists():
        level = "warning" if _is_optional_compat_profile(profile) else "error"
        code = "surface_pdf_missing_compat" if _is_optional_compat_profile(profile) else "surface_pdf_missing"
        _emit_issue(issues, level=level, code=code, locale=locale, profile=profile, pdf=_rel(pdf_path))

    _audit_generated_tex_language(tex_path=tex_path, locale=locale, issues=issues)
    if pdf_path.exists():
        _audit_pdf_language(
            path=pdf_path,
            locale=locale,
            issues=issues,
            code="generated_pdf_language_residue",
            profile=profile,
            pdf=_rel(pdf_path),
        )

    tex_text = tex_path.read_text(encoding="utf-8", errors="replace")
    for ref in _iter_graphic_refs(tex_text):
        resolved = _resolve_graphic_ref(tex_path, ref)
        graphic_record: Dict[str, Any] = {"ref": ref, "resolved": _rel(resolved) if resolved else None}
        record["graphics"].append(graphic_record)
        if resolved is None:
            _emit_issue(
                issues,
                level="error",
                code="figure_reference_missing",
                locale=locale,
                profile=profile,
                tex=_rel(tex_path),
                ref=ref,
            )
            continue

        _audit_pdf_language(
            path=resolved,
            locale=locale,
            issues=issues,
            code="figure_pdf_language_residue",
            profile=profile,
            ref=ref,
            figure_pdf=_rel(resolved),
        )

        if locale == locale_registry.DEFAULT_PAPER_LOCALE and "locales" in resolved.parts:
            _emit_issue(
                issues,
                level="error",
                code="ja_surface_uses_localized_figure",
                locale=locale,
                profile=profile,
                ref=ref,
                resolved=_rel(resolved),
            )

        if locale != locale_registry.DEFAULT_PAPER_LOCALE and "locales" in resolved.parts:
            locale_index = resolved.parts.index("locales")
            actual_locale = resolved.parts[locale_index + 1] if locale_index + 1 < len(resolved.parts) else ""
            if actual_locale != locale:
                _emit_issue(
                    issues,
                    level="error",
                    code="surface_uses_other_locale_figure",
                    locale=locale,
                    profile=profile,
                    ref=ref,
                    resolved=_rel(resolved),
                    actual_locale=actual_locale,
                )

        candidates = _gather_same_name_candidates(Path(ref).name, locale=locale)
        graphic_record["candidate_counts"] = {key: len(value) for key, value in candidates.items()}
        if locale == locale_registry.DEFAULT_PAPER_LOCALE:
            if candidates["canonical"] and (not _hash_matches_any(resolved, candidates["canonical"])):
                _emit_issue(
                    issues,
                    level="warning",
                    code="ja_figure_hash_not_canonical",
                    locale=locale,
                    profile=profile,
                    ref=ref,
                    resolved=_rel(resolved),
                )

            continue

        localized_exists = len(candidates["localized"]) > 0
        matches_localized = _hash_matches_any(resolved, candidates["localized"])
        matches_canonical = _hash_matches_any(resolved, candidates["canonical"])

        if localized_exists and not matches_localized:
            level = "error" if strict_localized_figures else "warning"
            code = "localized_figure_unused" if matches_canonical else "localized_figure_hash_mismatch"
            _emit_issue(
                issues,
                level=level,
                code=code,
                locale=locale,
                profile=profile,
                ref=ref,
                resolved=_rel(resolved),
                localized_candidates=[_rel(path) for path in candidates["localized"][:5]],
            )
        elif (not localized_exists) and matches_canonical:
            _emit_issue(
                issues,
                level="warning",
                code="localized_figure_missing_using_canonical",
                locale=locale,
                profile=profile,
                ref=ref,
                resolved=_rel(resolved),
            )

    return record


# 関数: `_build_output_path` の入出力契約と処理意図を定義する。

def _build_output_path(*, outdir: Path, locale: str, profile_label: str) -> Path:
    base_name = f"paper_locale_qc_{profile_label}.json"
    return outdir / locale_registry.localized_output_name(base_name, locale=locale)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Locale-aware paper double-check (manifest + generated surface).")
    parser.add_argument(
        "--profile",
        choices=("all",) + profile_content.PAPER_PROFILES,
        default="all",
        help="audit target profile or all profiles (default: all).",
    )
    parser.add_argument(
        "--locale",
        default=None,
        help="paper locale key (default: env WAVEP_PAPER_LOCALE or ja).",
    )
    parser.add_argument(
        "--mode",
        choices=("manifest", "surface", "all"),
        default="all",
        help="manifest only / generated surface only / both (default: all).",
    )
    parser.add_argument(
        "--outdir",
        default=str(_DEFAULT_OUTDIR),
        help="summary output directory where generated TeX/PDF live.",
    )
    parser.add_argument(
        "--strict-localized-figures",
        action="store_true",
        help="treat localized figure leakage as error instead of warning.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    locale = locale_registry.resolve_active_locale(args.locale)
    mode = _normalize_mode(args.mode)
    outdir = Path(str(args.outdir)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    profiles = _resolve_profiles(str(args.profile))
    issues: List[_Issue] = []
    result: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "locale": locale,
        "mode": mode,
        "profiles": profiles,
    }

    if mode in {"manifest", "all"}:
        result["manifest_audit"] = _audit_manifest(root=_ROOT, locale=locale, profiles=profiles, issues=issues)

    if mode in {"surface", "all"}:
        result["surface_audit"] = [
            _audit_surface_for_profile(
                profile=profile,
                outdir=outdir,
                locale=locale,
                issues=issues,
                strict_localized_figures=bool(args.strict_localized_figures),
            )
            for profile in profiles
        ]

    error_count = sum(1 for issue in issues if issue.level == "error")
    warning_count = sum(1 for issue in issues if issue.level == "warning")
    payload = {
        **result,
        "ok": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": [asdict(issue) for issue in issues],
    }

    out_json = _build_output_path(outdir=outdir, locale=locale, profile_label=str(args.profile))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_json}")
    print(f"paper_locale_qc: ok={payload['ok']} errors={error_count} warnings={warning_count}")

    try:
        worklog.append_event(
            {
                "event_type": "paper_locale_qc",
                "locale": locale,
                "mode": mode,
                "profiles": profiles,
                "ok": payload["ok"],
                "error_count": error_count,
                "warning_count": warning_count,
                "out_json": out_json,
            }
        )
    except Exception:
        pass

    return 0 if payload["ok"] else 1


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
