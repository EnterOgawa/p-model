#!/usr/bin/env python3
"""Generate 8.7.56.3303-.3306 corrected vacuum-subtraction return audit artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_3191 as base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


base.PRIOR_GATE = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3299-3302",
        "updated_pack_corrected_mixed_kernel_gate_vacuum_subtraction_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
base.PRIOR_SPLIT_AUDIT = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3287-3290",
        "updated_pack_corrected_probe_split_return_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
base.PRIOR_MIXED_AUDIT = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3295-3298",
        "updated_pack_corrected_mixed_kernel_return_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
base.OLDER_VACUUM_AUDIT = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3247-3250",
        "updated_pack_corrected_vacuum_subtraction_return_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
base.STEP_TAG = "8.7.56.3303-3306"
base.STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "vacuum-subtraction return refresh audit"
)
base.STEM = build_compact_artifact_stem(
    base.STEP_TAG,
    "updated_pack_corrected_vacuum_subtraction_return_refresh_audit",
    prefix="q",
)
base.PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_mixed_kernel_return_audited_vacuum_subtraction_primary_"
    "pack_refresh_secondary_hybrid_reserve_next"
)
base.BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_vacuum_subtraction_return_audited_pack_refresh_primary_"
    "hybrid_reserve_gate"
)

_ORIGINAL_PAYLOAD = base.sign_base.payload


# 関数: payload の step と route を current branch 向けに補正する。
def _rewrite_outputs() -> None:
    """Patch route metadata after the base script writes the artifacts."""
    for kind in ("declaration_gate", "route_sync"):
        path = build_metrics_paths(base.PUBLIC_OUT, base.STEM, kind)["json"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["inputs"]["routes"]["next_route"] = "8.7.56.3307"
        payload["inputs"]["routes"]["followup_route"] = "8.7.56.3311"
        payload["summary"]["recommended_next_route_or_none"] = "8.7.56.3307"
        payload["summary"]["selected_followup_route_or_none"] = "8.7.56.3311"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


base.sign_base.payload = lambda step, *args, **kwargs: _ORIGINAL_PAYLOAD(
    "8.7.56.3305" if step == "8.7.56.3193" else step,
    *args,
    **kwargs,
)


if __name__ == "__main__":
    base.main()
    _rewrite_outputs()
