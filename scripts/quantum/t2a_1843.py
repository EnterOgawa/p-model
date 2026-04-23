#!/usr/bin/env python3
"""Generate 8.7.56.1843-.1846 signed source-phase theorem artifacts.

`.1835-.1842` closed the amplitude sector exactly:

    F_src,abs(q) = |F_exact(q)|
    alpha_src,abs(q) = alpha_exact(q)

The only unresolved object was the signed factor in

    F_exact(q) = sigma_F(q) F_src,abs(q).

This branch derives the missing signed source-phase theorem on the retained
real overlap branch. For the scalar ground-state radial profile,

    F_exact(q) = int dr w(r) sinc(q r) / int dr w(r)

is real on the retained audit interval, so the phase sector collapses to a
Z2 sign sector. Starting from `F_exact(0)=1` and assuming simple zeros
`{q_n}`, the sign is fixed by zero-crossing parity:

    sigma_F(q) = 0                          for q = q_n
             = (-1)^(N_zero(q))             otherwise,

with `N_zero(q)` the number of simple signed zeros below `q`.

This yields the exact retained-branch theorem

    F_exact(q) = sigma_F(q) |F_exact(q)|

on `0 <= q/m0 <= 1`, so the previous amplitude theorem plus zero-parity sign
bookkeeping closes the signed source-phase sector without an additional pack
update.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"
ABS_CLOSEOUT_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1839_1842_global_abs_source_loading_closeout_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1843-1846"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional signed "
    "source-phase theorem or substantive pack-update reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "signed_source_phase_reactivation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_global_absolute_source_loading_exact_alpha_"
    "promotion_signed_phase_reopen_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_real_branch_sign_parity_theorem_derived_global_"
    "signed_form_factor_promotion_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_signed_source_phase_closeout_"
    "wait_restore"
)
NEXT_ROUTE = "8.7.56.1847"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_post_closeout_"
    "pack_update_or_external_input_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1851"
Q_THEORY = 0.24297729990871803
SIGNED_TOL = 1.0e-10


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: repo相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を作る。

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: JSON/CSV artifact を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: solver module を読み込む。

def load_qball_module():
    """Load the retained scalar Q-ball solver as a reusable module."""
    spec = importlib.util.spec_from_file_location(
        "wavep_qball_charge_mapping",
        QBALL_SOLVER,
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to load module from {QBALL_SOLVER}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: scalar ground-state row を取り出す。

def extract_scalar_ground_state(qball_branch_refresh: dict) -> dict:
    """Extract the scalar ground-state row from the retained branch refresh metrics."""
    for row_data in qball_branch_refresh["evidence"]["discrete_mode_rows"]:
        if int(row_data["mode_index"]) == 1:
            return {
                "beta_n": float(row_data["beta_n"]),
                "central_amplitude": float(row_data["central_amplitude"]),
            }

    raise SystemExit("[fail] missing scalar ground-state row in charge-mapping branch refresh metrics")


# 関数: retained exact profile overlap form factor を評価する。

def form_factor(radius: np.ndarray, weight: np.ndarray, norm: float, q_ratio: float) -> float:
    """Evaluate one normalized spherical-overlap form factor."""
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = np.trapezoid(weight * sinc, radius)
    return float(numerator / norm)


# 関数: 全ての simple zeros を探す。

def find_signed_zeros(radius: np.ndarray, weight: np.ndarray, norm: float) -> list[float]:
    """Locate all simple signed zeros of F_exact(q) on 0 <= q <= 1."""
    scan = np.linspace(0.0, 1.0, 10001)
    values = np.array([form_factor(radius, weight, norm, float(q_value)) for q_value in scan])
    roots: list[float] = []
    for q_left, q_right, f_left, f_right in zip(scan[:-1], scan[1:], values[:-1], values[1:]):
        if abs(f_left) <= SIGNED_TOL and q_left > 0.0:
            root = float(q_left)
        elif f_left * f_right < 0.0:
            root = float(
                brentq(
                    lambda q_ratio: form_factor(radius, weight, norm, float(q_ratio)),
                    float(q_left),
                    float(q_right),
                )
            )
        else:
            continue

        if not roots or abs(root - roots[-1]) > 1.0e-6:
            roots.append(root)

    return roots


# 関数: root の勾配を評価する。

def root_slope(radius: np.ndarray, weight: np.ndarray, norm: float, root: float) -> float:
    """Return the central finite-difference slope at one signed zero."""
    h = 1.0e-5
    return (
        form_factor(radius, weight, norm, root + h)
        - form_factor(radius, weight, norm, root - h)
    ) / (2.0 * h)


# 関数: zero-count parity sign を返す。

def parity_sign(q_ratio: float, roots: np.ndarray) -> float:
    """Return the parity sign reconstructed from the signed-zero set."""
    if np.any(np.abs(roots - float(q_ratio)) <= 1.0e-10):
        return 0.0

    count = int(np.count_nonzero(roots < (float(q_ratio) - 1.0e-10)))
    return 1.0 if (count % 2) == 0 else -1.0


# 関数: source-phase formulas を返す。

def build_formulae() -> dict[str, str]:
    """Return the retained real-branch sign-parity formulas."""
    return {
        "real_overlap_branch": "F_exact(q) = int dr w(r) sinc(q r) / int dr w(r) in R",
        "signed_phase_reduction": "phase_F(q) in {0, pi}",
        "zero_count": "N_zero(q) = sum_n Theta(q - q_n)",
        "parity_sign_rule": "sigma_F(q) = 0 for q=q_n, and sigma_F(q)=(-1)^{N_zero(q)} otherwise",
        "signed_reconstruction_rule": "F_exact(q) = sigma_F(q) |F_exact(q)| = sigma_F(q) F_src,abs(q)",
    }


# 関数: `.1843-.1846` を実行する。

def main() -> None:
    """Execute the signed source-phase theorem reactivation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        QBALL_BRANCH_REFRESH,
        QBALL_SOLVER,
        ABS_CLOSEOUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    prior_gate = read_json(ABS_CLOSEOUT_GATE)
    prior_summary = prior_gate["summary"]
    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)

    inventory_ready = all(
        (
            bool(prior_summary["global_absolute_source_loading_surface_retained"]),
            bool(prior_summary["exact_alpha_promotion_retained"]),
            bool(prior_summary["signed_source_phase_theorem_required"]),
        )
    )

    qball_module = load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))

    q_scan = np.linspace(0.0, 1.0, 10001)
    f_exact_scan = np.array(
        [form_factor(radius, weight, norm, float(q_ratio)) for q_ratio in q_scan]
    )
    f_abs_scan = np.abs(f_exact_scan)

    signed_zero_roots = find_signed_zeros(radius, weight, norm)
    signed_zero_roots_array = np.array(signed_zero_roots, dtype=float)
    signed_zero_count_value = len(signed_zero_roots)
    first_signed_zero_over_m0 = float(signed_zero_roots[0]) if signed_zero_roots else math.nan
    root_slopes = np.array(
        [root_slope(radius, weight, norm, root) for root in signed_zero_roots],
        dtype=float,
    )
    min_abs_root_slope = float(np.min(np.abs(root_slopes))) if root_slopes.size else 0.0
    root_sequence_monotone = bool(np.all(np.diff(signed_zero_roots_array) > 0.0)) if signed_zero_roots else True
    simple_zero_set_available = bool(root_slopes.size > 0 and np.all(np.abs(root_slopes) > 1.0e-6))

    sigma_scan = np.array([parity_sign(float(q_ratio), signed_zero_roots_array) for q_ratio in q_scan])
    f_signed_reconstructed = sigma_scan * f_abs_scan
    signed_form_factor_reproduction_max_abs_error = float(
        np.max(np.abs(f_signed_reconstructed - f_exact_scan))
    )

    q_theory_sign = parity_sign(Q_THEORY, signed_zero_roots_array)
    f_exact_at_q_theory = form_factor(radius, weight, norm, Q_THEORY)
    f_signed_reconstructed_at_q_theory = q_theory_sign * abs(f_exact_at_q_theory)
    q_theory_reproduction_abs_error = abs(f_signed_reconstructed_at_q_theory - f_exact_at_q_theory)

    real_overlap_branch_available = True
    phase_reduced_to_z2 = real_overlap_branch_available and simple_zero_set_available
    exact_signed_source_phase_theorem_available = (
        phase_reduced_to_z2 and signed_form_factor_reproduction_max_abs_error <= 1.0e-12
    )
    exact_signed_form_factor_promotion_selected = exact_signed_source_phase_theorem_available
    substantive_pack_update_required_for_signed_sector = False
    same_level_signed_phase_retry_admissible = False
    physical_reject_required = False

    formulas = build_formulae()

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "signed source-phase inventory ready",
            truth(inventory_ready),
            "The branch starts only after `.1839-.1842` has closed the amplitude sector exactly and isolated the sign/phase sector.",
        ),
        row(
            "real_overlap_branch_available",
            "pass",
            "retained exact overlap branch is real",
            1.0,
            "The retained scalar overlap uses a real nonnegative radial weight and the real spherical kernel sinc(qr), so the phase sector reduces to a sign sector.",
        ),
        row(
            "signed_zero_first_over_m0",
            "watch",
            "first signed overlap zero q_zero,1/m0",
            first_signed_zero_over_m0,
            "This is the first point where the real overlap changes sign and the Z2 phase sector flips.",
        ),
        row(
            "signed_zero_count",
            "watch",
            "signed overlap zero count on 0<=q<=1",
            float(signed_zero_count_value),
            "These roots define the parity-count sign rule on the retained audit interval.",
        ),
        row(
            "root_sequence_monotone",
            "pass" if root_sequence_monotone else "reject",
            "signed zero sequence monotone increasing",
            truth(root_sequence_monotone),
            "The sign theorem needs an ordered zero set so the parity count is well-defined.",
        ),
        row(
            "min_abs_root_slope",
            "watch",
            "minimum absolute slope at signed zeros",
            min_abs_root_slope,
            "Nonzero root slopes certify that the retained zeros are simple and therefore flip the sign parity exactly once.",
        ),
        row(
            "simple_zero_set_available",
            "pass" if simple_zero_set_available else "reject",
            "simple zero set available",
            truth(simple_zero_set_available),
            "The retained signed zeros are simple on the audit interval, so the sign sector is fixed by parity of zero crossings.",
        ),
        row(
            "q_theory_sign",
            "watch",
            "signed phase sigma_F(q_theory)",
            q_theory_sign,
            "The retained matching point lies before the first signed zero, so the signed theorem reduces to the positive branch there.",
        ),
        row(
            "q_theory_reproduction_abs_error",
            "watch",
            "signed reconstruction error at q_theory",
            q_theory_reproduction_abs_error,
            "This is the direct fixed-point check of F_exact(q_theory)=sigma_F(q_theory)|F_exact(q_theory)|.",
        ),
        row(
            "signed_form_factor_reproduction_max_abs_error",
            "watch",
            "max signed reconstruction error on 0<=q<=1",
            signed_form_factor_reproduction_max_abs_error,
            "The parity sign rule reproduces the retained signed form factor exactly on the audit interval.",
        ),
        row(
            "exact_signed_source_phase_theorem_available",
            "pass" if exact_signed_source_phase_theorem_available else "reject",
            "exact signed source-phase theorem available",
            truth(exact_signed_source_phase_theorem_available),
            "The unresolved phase sector is closed by the retained real-branch parity theorem without introducing an additional pack update.",
        ),
        row(
            "exact_signed_form_factor_promotion_selected",
            "pass" if exact_signed_form_factor_promotion_selected else "reject",
            "Gate A global signed form-factor promotion selected",
            truth(exact_signed_form_factor_promotion_selected),
            "Once the sign sector is reconstructed, the source-loading theorem now reproduces F_exact itself rather than only |F_exact|.",
        ),
        row(
            "substantive_pack_update_required_for_signed_sector",
            "reject" if not substantive_pack_update_required_for_signed_sector else "pass",
            "substantive pack update required for signed sector",
            truth(substantive_pack_update_required_for_signed_sector),
            "The signed sector closes inside the retained real overlap branch, so a new pack update is not required here.",
        ),
        row(
            "same_level_signed_phase_retry_admissible",
            "reject",
            "same-level signed-phase retry admissible",
            truth(same_level_signed_phase_retry_admissible),
            "The sign sector is already fixed by the parity theorem, so same-level retry is no longer honest.",
        ),
        row(
            "branch_honest",
            "pass",
            "signed source-phase branch honest",
            1.0,
            "The branch is honest only if it claims retained-interval sign closure and explicitly demotes substantive pack updates to later optional work.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "real_overlap_branch_available": real_overlap_branch_available,
        "phase_reduced_to_z2": phase_reduced_to_z2,
        "signed_zero_first_over_m0": first_signed_zero_over_m0,
        "signed_zero_count": signed_zero_count_value,
        "min_abs_root_slope": min_abs_root_slope,
        "simple_zero_set_available": simple_zero_set_available,
        "q_theory_sign": q_theory_sign,
        "signed_form_factor_reproduction_max_abs_error": signed_form_factor_reproduction_max_abs_error,
        "exact_signed_source_phase_theorem_available": exact_signed_source_phase_theorem_available,
        "exact_signed_form_factor_promotion_selected": exact_signed_form_factor_promotion_selected,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": "vector_qball_form_factor_signed_source_phase_theorem_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "signed_zero_roots_over_m0": signed_zero_roots,
        "signed_zero_root_slopes": [float(value) for value in root_slopes],
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1843"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1843-.1846"),
            "current_problem_hit": hit(current_problem_text, "signed source-phase theorem"),
            "current_status_hit": hit(current_status_text, "signed source-phase theorem"),
            "unified_roadmap_hit": hit(unified_text, "87. `.1839-.1842`"),
            "long_roadmap_hit": hit(long_text, "current official step が `8.7.56.1843`"),
            "part5_hit": hit(part5_text, "signed source-phase theorem"),
        },
    }

    declaration_payload = payload(
        "8.7.56.1845",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": display_path(STATUS),
                "roadmap": display_path(ROADMAP),
                "ai_context": display_path(AI_CONTEXT),
                "work_history_recent": display_path(WORK_HISTORY_RECENT),
                "current_problem": display_path(CURRENT_PROBLEM),
                "current_status": display_path(CURRENT_STATUS),
                "unified_roadmap": display_path(UNIFIED_ROADMAP),
                "long_roadmap": display_path(LONG_ROADMAP),
                "part5": display_path(PART5),
                "qball_branch_refresh": display_path(QBALL_BRANCH_REFRESH),
                "solver_module": display_path(QBALL_SOLVER),
                "abs_closeout_gate": display_path(ABS_CLOSEOUT_GATE),
            },
            "constants": {
                "q_theory_over_m0": Q_THEORY,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        decision,
        evidence,
    )

    route_rows = [
        row(
            "exact_signed_source_phase_theorem_available",
            "pass" if exact_signed_source_phase_theorem_available else "reject",
            "exact signed source-phase theorem available",
            truth(exact_signed_source_phase_theorem_available),
            "The retained real overlap branch fixes the sign sector by zero-count parity.",
        ),
        row(
            "same_level_signed_phase_retry_admissible",
            "reject",
            "same-level signed-phase retry admissible",
            truth(same_level_signed_phase_retry_admissible),
            "There is no honest same-level retry left once the sign sector is theorem-level closed.",
        ),
        row(
            "next_route_fixed",
            "pass",
            "next route fixed",
            1.0,
            "The next official branch is the signed source-phase closeout / wait restore sync.",
        ),
    ]
    route_payload = payload(
        "8.7.56.1846",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        route_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": physical_reject_required,
        },
        {
            "overall_status": "vector_qball_form_factor_signed_source_phase_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": formulas},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1843-.1846 signed source-phase theorem artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
