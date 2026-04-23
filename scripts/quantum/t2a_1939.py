#!/usr/bin/env python3
"""Generate 8.7.56.1939-.1942 eleventh post-dormant reactivation audit artifacts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1907 as parent


base = parent.base
base.POST_CLOSEOUT_DORMANT_GATE = (
    base.PUBLIC_OUT
    / "q_8_7_56_1935_1938_tenth_post_dormant_wait_restore_registry_refresh_declaration_gate_metrics.json"
)
base.STEP_TAG = "8.7.56.1939-1942"
base.STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional eleventh post-dormant "
    "pack-update or external-input reactivation"
)
base.STEM = base.build_compact_artifact_stem(
    base.STEP_TAG,
    "eleventh_post_dormant_reactivation_audit",
    prefix="q",
)
base.PRIOR_CLASS = (
    "vector_qball_form_factor_global_exact_alpha_signed_form_factor_tenth_post_dormant_"
    "registry_refreshed_wait_restored"
)
base.BRANCH_CLASS = (
    "vector_qball_form_factor_eleventh_post_dormant_reactivation_audit_no_new_surface_"
    "registry_refresh_next"
)
base.NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_eleventh_post_dormant_wait_restore_"
    "registry_refresh"
)
base.NEXT_ROUTE = "8.7.56.1943"
base.FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_twelfth_post_dormant_"
    "pack_update_or_external_input_reactivation"
)
base.FOLLOWUP_ROUTE = "8.7.56.1947"
_ORIGINAL_PAYLOAD = base.payload


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
    """Rewrite legacy payload metadata into the eleventh post-dormant branch IDs."""
    step_map = {
        "8.7.56.1869": "8.7.56.1941",
        "8.7.56.1870": "8.7.56.1942",
        "8.7.56.1877": "8.7.56.1941",
        "8.7.56.1878": "8.7.56.1942",
        "8.7.56.1885": "8.7.56.1941",
        "8.7.56.1886": "8.7.56.1942",
        "8.7.56.1893": "8.7.56.1941",
        "8.7.56.1894": "8.7.56.1942",
        "8.7.56.1901": "8.7.56.1941",
        "8.7.56.1902": "8.7.56.1942",
        "8.7.56.1909": "8.7.56.1941",
        "8.7.56.1910": "8.7.56.1942",
        "8.7.56.1917": "8.7.56.1941",
        "8.7.56.1918": "8.7.56.1942",
        "8.7.56.1925": "8.7.56.1941",
        "8.7.56.1926": "8.7.56.1942",
        "8.7.56.1933": "8.7.56.1941",
        "8.7.56.1934": "8.7.56.1942",
    }
    status_map = {
        "vector_qball_form_factor_second_post_dormant_reactivation_audit_declared": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_audit_declared"
        ),
        "vector_qball_form_factor_second_post_dormant_reactivation_route_synced": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_route_synced"
        ),
        "vector_qball_form_factor_third_post_dormant_reactivation_audit_declared": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_audit_declared"
        ),
        "vector_qball_form_factor_third_post_dormant_reactivation_route_synced": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_route_synced"
        ),
        "vector_qball_form_factor_fourth_post_dormant_reactivation_audit_declared": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_audit_declared"
        ),
        "vector_qball_form_factor_fourth_post_dormant_reactivation_route_synced": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_route_synced"
        ),
        "vector_qball_form_factor_fifth_post_dormant_reactivation_audit_declared": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_audit_declared"
        ),
        "vector_qball_form_factor_fifth_post_dormant_reactivation_route_synced": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_route_synced"
        ),
        "vector_qball_form_factor_sixth_post_dormant_reactivation_audit_declared": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_audit_declared"
        ),
        "vector_qball_form_factor_sixth_post_dormant_reactivation_route_synced": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_route_synced"
        ),
        "vector_qball_form_factor_seventh_post_dormant_reactivation_audit_declared": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_audit_declared"
        ),
        "vector_qball_form_factor_seventh_post_dormant_reactivation_route_synced": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_route_synced"
        ),
        "vector_qball_form_factor_eighth_post_dormant_reactivation_audit_declared": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_audit_declared"
        ),
        "vector_qball_form_factor_eighth_post_dormant_reactivation_route_synced": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_route_synced"
        ),
        "vector_qball_form_factor_ninth_post_dormant_reactivation_audit_declared": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_audit_declared"
        ),
        "vector_qball_form_factor_ninth_post_dormant_reactivation_route_synced": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_route_synced"
        ),
        "vector_qball_form_factor_tenth_post_dormant_reactivation_audit_declared": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_audit_declared"
        ),
        "vector_qball_form_factor_tenth_post_dormant_reactivation_route_synced": (
            "vector_qball_form_factor_eleventh_post_dormant_reactivation_route_synced"
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


base.payload = _compat_payload


if __name__ == "__main__":
    base.main()
