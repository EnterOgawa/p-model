#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_locale_registry.py

論文 build に使う locale manifest と locale 別出力名を管理する。

目的:
- 現行の日本語 build を壊さず、将来の多言語版が同じ source / output 名を
  取り合わない構造を先に固定する。
- `doc/paper/locales/<locale>/manifest.json` を build chain の唯一の参照点にする。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_PAPER_LOCALE = "ja"
_MANIFEST_FILENAME = "manifest.json"


# 関数: `resolve_active_locale` の入出力契約と処理意図を定義する。
def resolve_active_locale(requested: str | None = None) -> str:
    candidate = (requested or os.environ.get("WAVEP_PAPER_LOCALE", "") or DEFAULT_PAPER_LOCALE).strip().lower()
    return candidate or DEFAULT_PAPER_LOCALE


# 関数: `resolve_manifest_path` の入出力契約と処理意図を定義する。
def resolve_manifest_path(root: Path, locale: str | None = None) -> Path:
    active_locale = resolve_active_locale(locale)
    return root / "doc" / "paper" / "locales" / active_locale / _MANIFEST_FILENAME


# 関数: `_normalize_manifest` の入出力契約と処理意図を定義する。
def _normalize_manifest(payload: Dict[str, Any]) -> Dict[str, str]:
    raw_paths = payload.get("paths")
    if isinstance(raw_paths, dict):
        source = raw_paths
    else:
        source = payload

    normalized: Dict[str, str] = {}
    for key, value in source.items():
        if isinstance(value, str) and value.strip():
            normalized[str(key)] = value.strip()

    return normalized


# 関数: `load_manifest` の入出力契約と処理意図を定義する。
def load_manifest(root: Path, locale: str | None = None) -> Dict[str, str]:
    manifest_path = resolve_manifest_path(root, locale=locale)
    if not manifest_path.exists():
        raise FileNotFoundError(f"paper locale manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = _normalize_manifest(payload if isinstance(payload, dict) else {})
    if not normalized:
        raise ValueError(f"paper locale manifest has no usable paths: {manifest_path}")

    return normalized


# 関数: `resolve_source_path` の入出力契約と処理意図を定義する。
def resolve_source_path(root: Path, key: str, locale: str | None = None) -> Path:
    manifest = load_manifest(root, locale=locale)
    if key not in manifest:
        active_locale = resolve_active_locale(locale)
        raise KeyError(f"paper locale manifest entry missing: locale={active_locale} key={key}")

    return (root / Path(manifest[key])).resolve()


# 関数: `localized_output_name` の入出力契約と処理意図を定義する。
def localized_output_name(base_name: str, locale: str | None = None) -> str:
    active_locale = resolve_active_locale(locale)
    if active_locale == DEFAULT_PAPER_LOCALE:
        return base_name

    path = Path(base_name)
    if path.suffix:
        return f"{path.stem}_{active_locale}{path.suffix}"

    return f"{base_name}_{active_locale}"
