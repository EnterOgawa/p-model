#!/usr/bin/env python3
"""Generate 8.7.56.3763-.3766 corrected pack-refresh return gate artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_3203 as base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


base.PRIOR_AUDIT = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3759-3762",
        "updated_pack_corrected_pack_refresh_return_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
base.STEP_TAG = "8.7.56.3763-3766"
base.STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "pack-refresh gate / corrected hybrid-reserve return refresh"
)
base.STEM = build_compact_artifact_stem(
    base.STEP_TAG,
    "updated_pack_corrected_pack_refresh_gate_hybrid_reserve_return",
    prefix="q",
)
base.PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_pack_refresh_return_audited_hybrid_reserve_secondary_gate"
)
base.BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_pack_refresh_return_audited_hybrid_reserve_next"
)

_ORIGINAL_PAYLOAD = base.sign_base.payload


# 関数: payload の step と route を current branch 向けに補正する。
def _rewrite_outputs() -> None:
    """Patch route metadata after the base script writes the artifacts."""
    for kind in ("declaration_gate", "route_sync"):
        path = build_metrics_paths(base.PUBLIC_OUT, base.STEM, kind)["json"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["inputs"]["routes"]["next_route"] = "8.7.56.3767"
        payload["inputs"]["routes"]["followup_route"] = "8.7.56.3771"
        payload["summary"]["recommended_next_route_or_none"] = "8.7.56.3767"
        payload["summary"]["selected_followup_route_or_none"] = "8.7.56.3771"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


base.sign_base.payload = lambda step, *args, **kwargs: _ORIGINAL_PAYLOAD(
    "8.7.56.3765" if step == "8.7.56.3205" else step,
    *args,
    **kwargs,
)


if __name__ == "__main__":
    base.main()
    _rewrite_outputs()
