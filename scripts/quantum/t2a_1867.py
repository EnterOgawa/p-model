#!/usr/bin/env python3
"""Generate 8.7.56.1867-.1870 second post-dormant reactivation audit artifacts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1859 as base


base.POST_CLOSEOUT_DORMANT_GATE = (
    base.PUBLIC_OUT
    / "q_8_7_56_1863_1866_post_dormant_wait_restore_registry_refresh_declaration_gate_metrics.json"
)
base.STEP_TAG = "8.7.56.1867-1870"
base.STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional second post-dormant "
    "pack-update or external-input reactivation"
)
base.STEM = base.build_compact_artifact_stem(
    base.STEP_TAG,
    "second_post_dormant_reactivation_audit",
    prefix="q",
)
base.PRIOR_CLASS = (
    "vector_qball_form_factor_global_exact_alpha_signed_form_factor_dormant_"
    "registry_refreshed_wait_restored"
)
base.BRANCH_CLASS = (
    "vector_qball_form_factor_second_post_dormant_reactivation_audit_no_new_surface_"
    "registry_refresh_next"
)
base.NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_second_post_dormant_wait_restore_"
    "registry_refresh"
)
base.NEXT_ROUTE = "8.7.56.1871"
base.FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_third_post_dormant_"
    "pack_update_or_external_input_reactivation"
)
base.FOLLOWUP_ROUTE = "8.7.56.1875"
_ORIGINAL_READ_JSON = base.read_json
_ORIGINAL_PAYLOAD = base.payload


# 関数: 直前 branch の summary key を互換化して旧 main を再利用する。
def _compat_read_json(path: Path) -> dict:
    """Normalize the prior registry-refresh summary into the legacy dormant keys."""
    data = _ORIGINAL_READ_JSON(path)
    if path != base.POST_CLOSEOUT_DORMANT_GATE:
        return data

    summary = data.get("summary", {})
    summary.setdefault(
        "post_closeout_dormant_registry_retained",
        bool(summary.get("post_dormant_registry_refreshed", False)),
    )
    summary.setdefault("exact_alpha_promotion_retained", True)
    summary.setdefault("exact_signed_form_factor_promotion_retained", True)
    data["summary"] = summary
    return data


# 関数: payload 内の step 番号と status 名を新 branch に合わせて補正する。

def _compat_payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Rewrite legacy payload metadata into the second post-dormant branch IDs."""
    step_map = {
        "8.7.56.1861": "8.7.56.1869",
        "8.7.56.1862": "8.7.56.1870",
    }
    status_map = {
        "vector_qball_form_factor_post_dormant_reactivation_audit_declared": (
            "vector_qball_form_factor_second_post_dormant_reactivation_audit_declared"
        ),
        "vector_qball_form_factor_post_dormant_reactivation_route_synced": (
            "vector_qball_form_factor_second_post_dormant_reactivation_route_synced"
        ),
    }
    decision = dict(decision)
    if "overall_status" in decision:
        decision["overall_status"] = status_map.get(
            decision["overall_status"],
            decision["overall_status"],
        )

    return _ORIGINAL_PAYLOAD(
        step_map.get(step, step),
        name,
        inputs,
        rows,
        summary,
        decision,
        evidence,
    )


base.read_json = _compat_read_json
base.payload = _compat_payload


if __name__ == "__main__":
    base.main()
