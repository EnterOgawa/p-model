#!/usr/bin/env python3
"""Generate 8.7.56.3839-.3842 corrected pack-refresh return repeat artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_3559 as base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


base.PRIOR_GATE = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3835-3838",
        "updated_pack_corrected_reserve_registry_gate_pack_refresh_return",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
base.PRIOR_AUDIT = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3831-3834",
        "updated_pack_corrected_reserve_registry_return_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
base.OLDER_RETURN_AUDIT = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3815-3818",
        "updated_pack_corrected_pack_refresh_return_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
base.STEP_TAG = "8.7.56.3839-3842"
base.STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "pack-refresh return repeat audit"
)
base.STEM = build_compact_artifact_stem(
    base.STEP_TAG,
    "updated_pack_corrected_pack_refresh_return_repeat_audit",
    prefix="q",
)
base.PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_reserve_registry_return_audited_pack_refresh_return_next"
)
base.BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_pack_refresh_return_cycle_repeat_detected_probe_split_primary_"
    "mixed_kernel_secondary_gate"
)

_ORIGINAL_PAYLOAD = base.sign_base.payload


# 関数: payload の step と route を current branch 向けに補正する。
def _rewrite_outputs() -> None:
    """Patch route metadata after the base script writes the artifacts."""
    for kind in ("declaration_gate", "route_sync"):
        path = build_metrics_paths(base.PUBLIC_OUT, base.STEM, kind)["json"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["inputs"]["routes"]["next_route"] = "8.7.56.3843"
        payload["inputs"]["routes"]["followup_route"] = "8.7.56.3847"
        payload["summary"]["recommended_next_route_or_none"] = "8.7.56.3843"
        payload["summary"]["selected_followup_route_or_none"] = "8.7.56.3847"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


base.sign_base.payload = lambda step, *args, **kwargs: _ORIGINAL_PAYLOAD(
    "8.7.56.3841" if step == "8.7.56.3561" else step,
    *args,
    **kwargs,
)


if __name__ == "__main__":
    base.main()
    _rewrite_outputs()
