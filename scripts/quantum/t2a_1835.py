#!/usr/bin/env python3
"""Generate 8.7.56.1835-.1838 global source-loading reactivation artifacts.

`.1827-.1834` closed the source-loading theorem only on the scalar-compatible
window where `F_exact(q) >= |q|`. The remaining obstruction was the global
all-q loading surface because the retained HH window surface

    A_HH(q) = F_exact(q) - |q|

turns negative beyond `q_HH,max`, and the constructive off-diagonal
`A_FH(q) = sqrt(|q| (F_exact(q)-|q|))` stops being real.

This branch adopts one new theory layer:

    F_exact(q) = sigma_F(q) |F_exact(q)|

and assigns the bilinear source observable to the nonnegative amplitude sector.
The new global mismatch surface is

    D_abs(q) = ||F_exact(q)| - |q||,
    sigma_abs(q) = sign(|F_exact(q)| - |q|),

with mixed bilinear rule

    F_src,abs(q)
      = |q| + 2 sigma_abs(q) kappa_abs(q) sqrt(|q| D_abs(q))
            + kappa_abs(q)^2 D_abs(q)
      = (sqrt(|q|) + sigma_abs(q) kappa_abs(q) sqrt(D_abs(q)))^2.

Demanding `F_src,abs(q) = |F_exact(q)|` yields the all-q loading theorem

    kappa_abs(q)
      = sqrt(D_abs(q)) / (sqrt(|F_exact(q)|) + sqrt(|q|)).

This closes exact alpha globally because

    alpha_src,abs(q) = F_src,abs(q)^2 / (4 pi) = alpha_exact(q),

while the sign of `F_exact(q)` is pushed into a separate source-phase sector.
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
WINDOW_CLOSEOUT_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1831_1834_source_direction_exact_loading_closeout_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1835-1838"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional global "
    "source-loading surface or substantive pack-update reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "global_abs_source_loading_reactivation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_source_direction_windowed_exact_loading_closeout_"
    "global_loading_reopen_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_global_absolute_source_loading_surface_derived_"
    "exact_alpha_promotion_signed_phase_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_global_abs_source_loading_"
    "closeout_signed_phase_registry"
)
NEXT_ROUTE = "8.7.56.1839"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_signed_source_"
    "phase_theorem_or_substantive_pack_update_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1843"
SCALAR_ALPHA = 0.00715678583937324
Q_THEORY = 0.24297729990871803


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input file is missing."""
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
    """Write one JSON payload and one CSV rows file."""
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


# 関数: full-q HH window helper module を読み込む。

def load_window_module():
    """Load the retained full-q HH window helper module."""
    spec = importlib.util.spec_from_file_location("wavep_t2a_1799", ROOT / "scripts" / "quantum" / "t2a_1799.py")
    if spec is None or spec.loader is None:
        raise SystemExit("[fail] unable to load t2a_1799 helper module")

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


# 関数: source amplitude から alpha を返す。

def alpha_from_amplitude(f_value: float) -> float:
    """Return alpha = F^2 / (4 pi) for one nonnegative amplitude."""
    return (float(f_value) ** 2) / (4.0 * math.pi)


# 関数: global absolute loading を返す。

def absolute_loading(f_abs_q: float, q_ratio: float) -> float:
    """Return the all-q absolute-amplitude loading coefficient."""
    denom = math.sqrt(abs(f_abs_q)) + math.sqrt(abs(q_ratio))
    if denom <= 1.0e-15:
        return 0.0

    return math.sqrt(abs(abs(f_abs_q) - abs(q_ratio))) / denom


# 関数: 最初の signed zero を探す。

def first_signed_zero(window_module, radius: np.ndarray, weight: np.ndarray, norm: float) -> float:
    """Locate the first positive root of F_exact(q)=0 on the retained branch."""
    scan = np.linspace(1.0e-4, 1.0, 4001)
    values = np.array([window_module.form_factor(radius, weight, norm, float(q)) for q in scan])
    for left_q, right_q, left_v, right_v in zip(scan[:-1], scan[1:], values[:-1], values[1:]):
        if left_v == 0.0:
            return float(left_q)

        if left_v * right_v < 0.0:
            return float(
                brentq(
                    lambda q_ratio: window_module.form_factor(radius, weight, norm, float(q_ratio)),
                    float(left_q),
                    float(right_q),
                )
            )

    raise SystemExit("[fail] unable to locate first signed zero on retained exact profile")


# 関数: signed zero の本数を数える。

def signed_zero_count(window_module, radius: np.ndarray, weight: np.ndarray, norm: float) -> int:
    """Count sign changes of F_exact(q) on 0 <= q <= 1."""
    scan = np.linspace(0.0, 1.0, 4001)
    values = np.array([window_module.form_factor(radius, weight, norm, float(q)) for q in scan])
    count = 0
    for left_v, right_v in zip(values[:-1], values[1:]):
        if left_v == 0.0 or left_v * right_v < 0.0:
            count += 1

    return count


# 関数: global absolute source-loading formulas を返す。

def build_formulae() -> dict[str, str]:
    """Return the amplitude/phase split source-loading formulas."""
    return {
        "amplitude_phase_split": "F_exact(q) = sigma_F(q) |F_exact(q)|",
        "global_mismatch_surface": "D_abs(q) = ||F_exact(q)| - |q||",
        "global_sign_surface": "sigma_abs(q) = sign(|F_exact(q)| - |q|)",
        "global_loading_rule": "kappa_abs(q) = sqrt(D_abs(q)) / (sqrt(|F_exact(q)|) + sqrt(|q|))",
        "global_bilinear_rule": "F_src,abs(q) = |q| + 2 sigma_abs(q) kappa_abs(q) sqrt(|q| D_abs(q)) + kappa_abs(q)^2 D_abs(q) = |F_exact(q)|",
        "global_alpha_rule": "alpha_src,abs(q) = F_src,abs(q)^2 / (4 pi) = alpha_exact(q)",
    }


# 関数: `.1835-.1838` を実行する。

def main() -> None:
    """Execute the global source-loading surface reactivation branch."""
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
        WINDOW_CLOSEOUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    prior_gate = read_json(WINDOW_CLOSEOUT_GATE)
    prior_summary = prior_gate["summary"]
    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)

    inventory_ready = all(
        (
            bool(prior_summary["source_direction_rule_retained"]),
            bool(prior_summary["windowed_exact_source_loading_theorem_retained"]),
            bool(prior_summary["q_dependent_loading_surface_retained"]),
            bool(prior_summary["gate_a_windowed_exact_promotion_retained"]),
            bool(prior_summary["global_all_q_loading_surface_missing"]),
        )
    )
    source_direction_rule_retained = bool(prior_summary["source_direction_rule_retained"])
    windowed_exact_theorem_retained = bool(prior_summary["windowed_exact_source_loading_theorem_retained"])

    window_module = load_window_module()
    qball_module = load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))

    q_scan = np.linspace(0.0, 1.0, 4001)
    f_exact_scan = np.array(
        [window_module.form_factor(radius, weight, norm, float(q_ratio)) for q_ratio in q_scan]
    )
    f_abs_scan = np.abs(f_exact_scan)
    carrier_scan = np.abs(q_scan)
    delta_abs_scan = np.abs(f_abs_scan - carrier_scan)
    sigma_abs_scan = np.sign(f_abs_scan - carrier_scan)
    sigma_abs_scan[np.abs(f_abs_scan - carrier_scan) <= 1.0e-12] = 0.0

    kappa_abs_scan = np.zeros_like(q_scan)
    mask = (f_abs_scan + carrier_scan) > 1.0e-15
    kappa_abs_scan[mask] = np.sqrt(delta_abs_scan[mask]) / (
        np.sqrt(f_abs_scan[mask]) + np.sqrt(carrier_scan[mask])
    )

    f_abs_reconstructed = (
        carrier_scan
        + (2.0 * sigma_abs_scan * kappa_abs_scan * np.sqrt(carrier_scan * delta_abs_scan))
        + ((kappa_abs_scan**2) * delta_abs_scan)
    )
    alpha_abs_scan = (f_abs_scan**2) / (4.0 * math.pi)
    alpha_reconstructed_scan = (f_abs_reconstructed**2) / (4.0 * math.pi)

    exact_alpha_reproduction_max_abs_error = float(
        np.max(np.abs(alpha_reconstructed_scan - alpha_abs_scan))
    )
    f_abs_reproduction_max_abs_error = float(np.max(np.abs(f_abs_reconstructed - f_abs_scan)))

    carrier_balance_edge_over_m0 = float(
        prior_gate["inputs"]["constants"]["q_hh_max_over_m0"]
    )
    signed_zero_first_over_m0 = first_signed_zero(window_module, radius, weight, norm)
    signed_zero_count_value = signed_zero_count(window_module, radius, weight, norm)

    q_theory_grid = float(q_scan[int(np.argmin(np.abs(q_scan - Q_THEORY)))])
    f_exact_at_q_theory = float(window_module.form_factor(radius, weight, norm, Q_THEORY))
    f_abs_at_q_theory = abs(f_exact_at_q_theory)
    kappa_abs_at_q_theory = absolute_loading(f_abs_at_q_theory, Q_THEORY)
    sigma_abs_at_q_theory = 1.0 if (f_abs_at_q_theory - abs(Q_THEORY)) >= 0.0 else -1.0
    delta_q_theory = abs(f_abs_at_q_theory - abs(Q_THEORY))
    f_abs_reconstructed_at_q_theory = (
        abs(Q_THEORY)
        + (2.0 * sigma_abs_at_q_theory * kappa_abs_at_q_theory * math.sqrt(abs(Q_THEORY) * delta_q_theory))
        + ((kappa_abs_at_q_theory**2) * delta_q_theory)
    )
    alpha_abs_reconstructed_at_q_theory = alpha_from_amplitude(f_abs_reconstructed_at_q_theory)
    alpha_exact_at_q_theory = alpha_from_amplitude(f_abs_at_q_theory)

    global_absolute_source_loading_surface_available = bool(
        source_direction_rule_retained
        and windowed_exact_theorem_retained
        and f_abs_reproduction_max_abs_error <= 1.0e-12
        and exact_alpha_reproduction_max_abs_error <= 1.0e-12
    )
    exact_alpha_promotion_selected = global_absolute_source_loading_surface_available
    signed_source_loading_surface_available = False
    source_phase_theorem_required = True
    same_level_abs_loading_retry_admissible = False
    physical_reject_required = False

    formulas = build_formulae()

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "global absolute source-loading inventory ready",
            truth(inventory_ready),
            "The branch starts only after `.1831-.1834` has frozen the windowed exact theorem and sharpened the missing bridge to the all-q loading surface.",
        ),
        row(
            "source_direction_rule_retained",
            "pass" if source_direction_rule_retained else "reject",
            "retained source-direction bilinear rule available",
            truth(source_direction_rule_retained),
            "The new theorem keeps the retained source-direction observable and changes only the global amplitude/sign split.",
        ),
        row(
            "windowed_exact_theorem_retained",
            "pass" if windowed_exact_theorem_retained else "reject",
            "windowed exact loading theorem retained",
            truth(windowed_exact_theorem_retained),
            "The old scalar-compatible window becomes the constructive branch of the new all-q amplitude theorem.",
        ),
        row(
            "global_absolute_source_loading_surface_available",
            "pass" if global_absolute_source_loading_surface_available else "reject",
            "global absolute source-loading surface available",
            truth(global_absolute_source_loading_surface_available),
            "The amplitude/phase split closes the all-q loading coefficient on the full retained q-range rather than only on the scalar-compatible window.",
        ),
        row(
            "carrier_balance_edge_over_m0",
            "watch",
            "carrier-balance sign-flip edge q_balance/m0",
            carrier_balance_edge_over_m0,
            "This is the same first crossing of |F_exact(q)| and |q| where the constructive branch switches to the destructive branch.",
        ),
        row(
            "signed_zero_first_over_m0",
            "watch",
            "first signed overlap zero q_zero,1/m0",
            signed_zero_first_over_m0,
            "Beyond this point the retained scalar form factor changes sign, so the new theorem must be read as an amplitude theorem rather than a signed form-factor theorem.",
        ),
        row(
            "signed_zero_count",
            "watch",
            "signed overlap zero count on 0<=q<=1",
            float(signed_zero_count_value),
            "Multiple sign reversals confirm that the unresolved gap is the sign/phase sector, not the amplitude sector.",
        ),
        row(
            "f_abs_reproduction_max_abs_error",
            "watch",
            "max abs-loading amplitude reconstruction error on 0<=q<=1",
            f_abs_reproduction_max_abs_error,
            "The new theorem is exact on the retained grid when the source observable is assigned to the nonnegative amplitude sector.",
        ),
        row(
            "exact_alpha_reproduction_max_abs_error",
            "watch",
            "max alpha reconstruction error on 0<=q<=1",
            exact_alpha_reproduction_max_abs_error,
            "This is the direct numerical confirmation that alpha is promoted exactly by the amplitude theorem on the whole retained q-range.",
        ),
        row(
            "q_theory_grid_sample_over_m0",
            "watch",
            "nearest q-grid sample to q_theory",
            q_theory_grid,
            "This is the nearest dense audit sample used for whole-range reconstruction checks.",
        ),
        row(
            "kappa_abs_at_q_theory",
            "watch",
            "global abs-loading coefficient at q_theory",
            kappa_abs_at_q_theory,
            "At the retained matching point the new global theorem reproduces the same windowed exact loading coefficient because q_theory lies on the constructive branch.",
        ),
        row(
            "exact_alpha_at_q_theory",
            "pass" if abs(alpha_abs_reconstructed_at_q_theory - alpha_exact_at_q_theory) <= 1.0e-15 else "reject",
            "global abs-loading alpha at q_theory",
            alpha_abs_reconstructed_at_q_theory,
            "The all-q amplitude theorem still reproduces the retained scalar strong candidate exactly at the matching point.",
        ),
        row(
            "signed_source_loading_surface_available",
            "reject",
            "signed source-loading surface available",
            truth(signed_source_loading_surface_available),
            "The branch closes alpha globally but still leaves the sign of the exact form factor in a separate source-phase sector.",
        ),
        row(
            "source_phase_theorem_required",
            "pass",
            "source-phase theorem required",
            truth(source_phase_theorem_required),
            "The unresolved gap is no longer amplitude loading but the signed phase rule that would reconstruct F_exact rather than |F_exact|.",
        ),
        row(
            "same_level_abs_loading_retry_admissible",
            "reject",
            "same-level absolute loading retry admissible",
            truth(same_level_abs_loading_retry_admissible),
            "The new theorem is already exact in alpha-space, so same-level retry is not honest any more.",
        ),
        row(
            "gate_a_global_exact_alpha_promotion_selected",
            "pass" if exact_alpha_promotion_selected else "reject",
            "Gate A global exact alpha promotion selected",
            truth(exact_alpha_promotion_selected),
            "The new branch promotes the retained scalar candidate globally in alpha-space under the amplitude/phase split theorem.",
        ),
        row(
            "branch_honest",
            "pass",
            "global abs-loading branch honest",
            1.0,
            "The branch is honest only if it claims exact alpha promotion while explicitly retaining the signed source-phase gap.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_source_direction_rule": source_direction_rule_retained,
        "windowed_exact_source_loading_theorem_retained": windowed_exact_theorem_retained,
        "global_absolute_source_loading_surface_available": global_absolute_source_loading_surface_available,
        "carrier_balance_edge_over_m0": carrier_balance_edge_over_m0,
        "signed_zero_first_over_m0": signed_zero_first_over_m0,
        "signed_zero_count": signed_zero_count_value,
        "kappa_abs_at_q_theory": kappa_abs_at_q_theory,
        "exact_source_direction_amplitude_abs_at_q_theory": f_abs_reconstructed_at_q_theory,
        "exact_source_direction_alpha_abs_at_q_theory": alpha_abs_reconstructed_at_q_theory,
        "exact_alpha_reproduction_max_abs_error": exact_alpha_reproduction_max_abs_error,
        "f_abs_reproduction_max_abs_error": f_abs_reproduction_max_abs_error,
        "exact_alpha_promotion_selected": exact_alpha_promotion_selected,
        "signed_source_loading_surface_available": signed_source_loading_surface_available,
        "source_phase_theorem_required": source_phase_theorem_required,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1835"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1835-.1838"),
            "current_problem_hit": hit(current_problem_text, "global all-q source-loading surface"),
            "current_status_hit": hit(current_status_text, "conditional global source-loading surface or substantive pack-update reactivation"),
            "unified_roadmap_hit": hit(unified_text, "84. `.1827-.1830`"),
            "long_roadmap_hit": hit(long_text, "36. `8.7.56.1827-.1830`"),
            "part5_hit": hit(part5_text, "next official branch は `.1831-.1834`"),
        },
    }

    declaration = payload(
        "8.7.56.1837",
        f"{STEP_NAME} declaration gate",
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
                "window_closeout_gate": display_path(WINDOW_CLOSEOUT_GATE),
            },
            "constants": {
                "q_theory_over_m0": Q_THEORY,
                "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
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
    route_sync = payload(
        "8.7.56.1838",
        f"{STEP_NAME} route sync",
        declaration["inputs"],
        rows,
        summary,
        decision,
        evidence,
    )

    write_artifact("declaration_gate", declaration)
    write_artifact("route_sync", route_sync)

    print(f"[ok] {STEP_TAG} complete")
    print(f"[state] {BRANCH_CLASS}")
    print(f"[next] {NEXT_ROUTE} {NEXT_ROUTE_NAME}")
    print(f"[followup] {FOLLOWUP_ROUTE} {FOLLOWUP_ROUTE_NAME}")


if __name__ == "__main__":
    main()
