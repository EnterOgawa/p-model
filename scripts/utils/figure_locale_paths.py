"""
figure_locale_paths.py

図 artifact の locale 別出力 path を管理する。

目的:
- 現行の日本語図 (`ja`) は canonical path のまま維持し、比較基準として残す。
- 将来の英語版など (`en` ほか) は `locales/<locale>/...` へ退避し、
  既存の `output/public` / `output/private` 図を上書きしないようにする。

前提:
- locale を変えても物理量や数値結果は同じであり、差分は主に図中テキストと
  参照用 artifact の置き場にある。
- `html_to_docx.py` などの補助系は今回の scope 外である。
"""

from __future__ import annotations

import os
from pathlib import Path


_DEFAULT_LOCALE = "ja"
_LOCALE_MARKER = "locales"


# 関数: `_normalize_locale` の入出力契約と処理意図を定義する。
def _normalize_locale(raw: str | None) -> str:
    candidate = str(raw or "").strip().lower()
    if candidate in {"ja", "jp", "japanese", "ja-jp", "ja_jp"}:
        return "ja"

    if candidate in {"en", "english", "en-us", "en_us", "en-gb", "en_gb"}:
        return "en"

    return candidate or _DEFAULT_LOCALE


# 関数: `resolve_figure_output_locale` の入出力契約と処理意図を定義する。
def resolve_figure_output_locale(requested: str | None = None) -> str:
    """
    図 artifact の出力 locale を返す。

    優先順:
    1. 呼び出し側の明示指定
    2. `WAVEP_FIGURE_LOCALE`
    3. `WAVEP_PAPER_LOCALE`
    4. `WAVEP_FIGURE_LANG`（ja/en のみ locale として流用）
    5. `ja`
    """
    if requested and str(requested).strip():
        return _normalize_locale(requested)

    env_locale = _normalize_locale(os.environ.get("WAVEP_FIGURE_LOCALE", ""))
    if env_locale != _DEFAULT_LOCALE or str(os.environ.get("WAVEP_FIGURE_LOCALE", "")).strip():
        return env_locale

    paper_locale = _normalize_locale(os.environ.get("WAVEP_PAPER_LOCALE", ""))
    if paper_locale != _DEFAULT_LOCALE or str(os.environ.get("WAVEP_PAPER_LOCALE", "")).strip():
        return paper_locale

    figure_lang = _normalize_locale(os.environ.get("WAVEP_FIGURE_LANG", ""))
    if figure_lang in {"ja", "en"}:
        return figure_lang

    return _DEFAULT_LOCALE


# 関数: `is_default_figure_locale` の入出力契約と処理意図を定義する。
def is_default_figure_locale(locale: str | None = None) -> bool:
    return resolve_figure_output_locale(locale) == _DEFAULT_LOCALE


# 関数: `_already_localized` の入出力契約と処理意図を定義する。
def _already_localized(parts: tuple[str, ...], *, locale: str) -> bool:
    for index, value in enumerate(parts[:-1]):
        if value != _LOCALE_MARKER:
            continue

        if index + 1 < len(parts) and parts[index + 1] == locale:
            return True

    return False


# 関数: `_localize_under_output_root` の入出力契約と処理意図を定義する。
def _localize_under_output_root(path: Path, *, output_root: Path, locale: str) -> Path:
    rel = path.resolve().relative_to(output_root.resolve())
    rel_parts = tuple(rel.parts)
    if _already_localized(rel_parts, locale=locale):
        return path

    localized_rel = Path(*rel_parts[:-1]) / _LOCALE_MARKER / locale / rel_parts[-1]
    return output_root / localized_rel


# 関数: `localize_figure_output_path` の入出力契約と処理意図を定義する。
def localize_figure_output_path(path: Path | str, *, root: Path | None = None, locale: str | None = None) -> Path:
    """
    `output/public` / `output/private` 配下の図 artifact path を locale 別 path へ写像する。

    ルール:
    - `ja` は現行 path をそのまま返す。
    - 非 `ja` は `.../<topic>/locales/<locale>/<filename>` へ出す。
    - `output/` 外の path は変更しない。
    """
    path_obj = Path(path)
    original_is_absolute = path_obj.is_absolute()
    repo_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    active_locale = resolve_figure_output_locale(locale)
    if active_locale == _DEFAULT_LOCALE:
        return path_obj

    resolved_path = path_obj if path_obj.is_absolute() else (repo_root / path_obj)
    public_root = repo_root / "output" / "public"
    private_root = repo_root / "output" / "private"

    try:
        resolved_path.resolve().relative_to(public_root.resolve())
        return _localize_under_output_root(resolved_path, output_root=public_root, locale=active_locale)
    except Exception:
        pass

    try:
        resolved_path.resolve().relative_to(private_root.resolve())
        return _localize_under_output_root(resolved_path, output_root=private_root, locale=active_locale)
    except Exception:
        pass

    if original_is_absolute:
        return resolved_path

    return path_obj
