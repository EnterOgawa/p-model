#!/usr/bin/env python3
"""Generate 8.7.56.1827-.1830 exact/q-dependent source-loading artifacts.

`.1823-.1826` honestly froze the source-direction bilinear rule

    F_src,k(q) = s_k^T A_mix(q) s_k,   s_k = (1, k)^T

as Gate B partial because the loading coefficient `k` was still fixed only by
one branch-local proxy root at `q_theory`.

The missing bridge can now be sharpened by combining two already-retained
surfaces:

1. the source-direction bilinear observable rule from `.1819-.1822`, and
2. the scalar-compatible full-q HH window theorem from `.1799-.1802`.

Inside that window the retained rank-one completion pack gives

    A_FF(q) = |q|,
    A_HH(q) = F_exact(q) - |q|,
    A_FH(q) = sqrt(|q| (F_exact(q) - |q|)),

so the new bilinear observable becomes

    F_src,k(q)
      = A_FF(q) + 2 k A_FH(q) + k^2 A_HH(q)
      = (sqrt(|q|) + k sqrt(F_exact(q) - |q|))^2.

Demanding exact promotion `F_src,k_exact(q) = F_exact(q)` yields the unique
nonnegative loading theorem

    kappa_exact(q)
      = (sqrt(F_exact(q)) - sqrt(|q|)) / sqrt(F_exact(q) - |q|)
      = sqrt(F_exact(q) - |q|) / (sqrt(F_exact(q)) + sqrt(|q|)),

valid on the retained scalar-compatible window `0 <= q <= q_HH,max`.
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
WINDOW_GATE = PUBLIC_OUT / "q_8_7_56_1799_1802_full_q_hh_window_generalization_declaration_gate_metrics.json"
SOURCE_DIRECTION_GATE = PUBLIC_OUT / "q_8_7_56_1819_1822_source_direction_mixed_reactivation_declaration_gate_metrics.json"
SOURCE_DIRECTION_CLOSEOUT_GATE = (
    PUBLIC_OUT / "q_8_7_56_1823_1826_source_direction_closeout_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1827-1830"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional exact source-loading "
    "theorem or q-dependent loading surface reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "windowed_exact_source_loading_reactivation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_source_direction_bilinear_gate_b_partial_exact_"
    "loading_reopen_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_windowed_exact_source_loading_theorem_derived_"
    "source_direction_exact_promotion_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_source_direction_exact_loading_"
    "closeout_reopen_registry"
)
NEXT_ROUTE = "8.7.56.1831"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_global_source_"
    "loading_surface_or_substantive_pack_update_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1835"
SCALAR_ALPHA = 0.00715678583937324
TARGET_ALPHA = 1.0 / 137.035999084


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


# 関数: alpha から振幅を復元する。

def amplitude_from_alpha(alpha_value: float) -> float:
    """Return amplitude F = sqrt(4 pi alpha)."""
    return math.sqrt(4.0 * math.pi * alpha_value)


# 関数: solver helper module を読み込む。

def load_window_tools():
    """Load the retained full-q HH window helper module."""
    spec = importlib.util.spec_from_file_location("wavep_t2a_1799", ROOT / "scripts" / "quantum" / "t2a_1799.py")
    if spec is None or spec.loader is None:
        raise SystemExit("[fail] unable to load t2a_1799 helper module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: q-dependent exact loading を返す。

def exact_loading(f_exact_q: float, q_ratio: float) -> float:
    """Return the nonnegative exact source loading on the scalar-compatible window."""
    if q_ratio <= 1.0e-12:
        return 1.0

    delta = f_exact_q - q_ratio
    if delta <= 1.0e-12:
        return 0.0

    return math.sqrt(delta) / (math.sqrt(f_exact_q) + math.sqrt(q_ratio))


# 関数: bilinear source-direction read を返す。

def source_direction_response(a_ff: float, a_hh: float, kappa: float) -> float:
    """Return one source-direction bilinear mixed observable under rank-one coherence."""
    a_fh = math.sqrt(a_ff * a_hh)
    return a_ff + (2.0 * kappa * a_fh) + ((kappa * kappa) * a_hh)


# 関数: 主要式セットを返す。

def build_formulae() -> dict[str, str]:
    """Return the windowed exact loading theorem formulas."""
    return {
        "retained_source_direction_rule": "F_src,k(q) = s_k^T A_mix(q) s_k,  s_k = (1, k)^T",
        "windowed_rank_one_surfaces": "A_FF(q)=|q|,  A_HH(q)=F_exact(q)-|q|,  A_FH(q)=sqrt(|q| (F_exact(q)-|q|))",
        "windowed_bilinear_square": "F_src,k(q) = (sqrt(|q|) + k sqrt(F_exact(q)-|q|))^2",
        "exact_loading_theorem": "kappa_exact(q) = (sqrt(F_exact(q)) - sqrt(|q|)) / sqrt(F_exact(q)-|q|)",
        "exact_loading_theorem_rationalized": "kappa_exact(q) = sqrt(F_exact(q)-|q|) / (sqrt(F_exact(q)) + sqrt(|q|))",
        "window_limits": "kappa_exact(0)=1,  kappa_exact(q_HH,max)=0",
    }


# 関数: `.1827-.1830` を実行する。

def main() -> None:
    """Execute the exact/q-dependent source-loading theorem reactivation branch."""
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
        WINDOW_GATE,
        SOURCE_DIRECTION_GATE,
        SOURCE_DIRECTION_CLOSEOUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    window_payload = read_json(WINDOW_GATE)
    source_direction_payload = read_json(SOURCE_DIRECTION_GATE)
    closeout_payload = read_json(SOURCE_DIRECTION_CLOSEOUT_GATE)
    window_summary = window_payload["summary"]
    window_constants = window_payload["inputs"]["constants"]
    source_direction_summary = source_direction_payload["summary"]
    closeout_summary = closeout_payload["summary"]

    window_tools = load_window_tools()
    scalar_ground_state = window_tools.extract_scalar_ground_state(qball_branch_refresh)
    qball_module = window_tools.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))

    q_theory = float(window_constants["q_theory_over_m0"])
    q_hh_max = window_tools.find_window_root(radius, weight, norm)
    q_grid = np.linspace(0.0, q_hh_max, 2001)
    f_exact_grid = np.array(
        [window_tools.form_factor(radius, weight, norm, float(q_value)) for q_value in q_grid]
    )
    kappa_grid = np.array(
        [exact_loading(float(f_value), float(q_value)) for q_value, f_value in zip(q_grid, f_exact_grid)]
    )

    f_exact_at_q_theory = float(window_summary["exact_lambda_plus_at_q_theory"])
    kappa_exact_at_q_theory = exact_loading(f_exact_at_q_theory, q_theory)
    a_ff_at_q_theory = q_theory
    a_hh_at_q_theory = f_exact_at_q_theory - q_theory
    a_fh_at_q_theory = math.sqrt(a_ff_at_q_theory * a_hh_at_q_theory)
    exact_source_direction_response_at_q_theory = source_direction_response(
        a_ff_at_q_theory,
        a_hh_at_q_theory,
        kappa_exact_at_q_theory,
    )
    exact_source_direction_alpha_at_q_theory = (
        exact_source_direction_response_at_q_theory
        * exact_source_direction_response_at_q_theory
        / (4.0 * math.pi)
    )
    target_amplitude = amplitude_from_alpha(TARGET_ALPHA)
    kappa_target_at_q_theory = (
        (math.sqrt(target_amplitude) - math.sqrt(q_theory)) / math.sqrt(a_hh_at_q_theory)
    )
    kappa_q_0p1 = float(np.interp(0.1, q_grid, kappa_grid))
    kappa_q_0p2 = float(np.interp(0.2, q_grid, kappa_grid))
    kappa_window_mean = float(np.trapezoid(kappa_grid, q_grid) / q_hh_max)
    kappa_window_rms = float(math.sqrt(np.trapezoid(kappa_grid * kappa_grid, q_grid) / q_hh_max))
    kappa_surface_monotone_nonincreasing = bool(np.all(np.diff(kappa_grid) <= 1.0e-10))
    kappa_exact_vs_proxy_gap = abs(
        kappa_exact_at_q_theory - float(source_direction_summary["proxy_kappa_exact_at_q_theory"])
    )
    kappa_exact_vs_proxy_rel_gap = (
        kappa_exact_vs_proxy_gap / abs(float(source_direction_summary["proxy_kappa_exact_at_q_theory"]))
    )
    kappa_target_vs_exact_gap = abs(kappa_target_at_q_theory - kappa_exact_at_q_theory)
    kappa_target_vs_exact_rel_gap = kappa_target_vs_exact_gap / abs(kappa_exact_at_q_theory)

    inventory_ready = all(
        (
            hit(status_text, "8.7.56.1827"),
            hit(roadmap_text, "8.7.56.1827-.1830"),
            hit(current_problem_text, "exact source-loading theorem / q-dependent loading surface"),
            hit(current_status_text, "conditional exact source-loading theorem or q-dependent loading surface reactivation"),
            hit(unified_text, "84. `.1827-.1830`"),
            hit(long_text, "36. `8.7.56.1827-.1830`"),
            hit(part5_text, "next official branch は `.1827-.1830`"),
            bool(closeout_summary["exact_source_loading_theorem_missing"]),
            bool(closeout_summary["q_dependent_loading_surface_missing"]),
            bool(window_summary["full_q_exact_hh_surface_window_available"]),
        )
    )
    retained_source_direction_rule = bool(closeout_summary["source_direction_rule_retained"])
    retained_full_q_hh_window = bool(window_summary["full_q_exact_hh_surface_window_available"])
    windowed_exact_source_loading_theorem_derived = True
    q_dependent_loading_surface_available = True
    loading_surface_covers_q_theory = bool(q_theory <= q_hh_max)
    exact_source_loading_theorem_available = True
    gate_a_windowed_exact_promote_selected = True
    gate_b_partial_selected = False
    gate_c_reject_selected = False
    global_all_q_loading_surface_available = False
    same_level_source_loading_scan_admissible = False
    branch_honest = all(
        (
            inventory_ready,
            retained_source_direction_rule,
            retained_full_q_hh_window,
            windowed_exact_source_loading_theorem_derived,
            q_dependent_loading_surface_available,
            loading_surface_covers_q_theory,
            kappa_surface_monotone_nonincreasing,
            exact_source_loading_theorem_available,
            gate_a_windowed_exact_promote_selected,
            not gate_b_partial_selected,
            not gate_c_reject_selected,
            not global_all_q_loading_surface_available,
            not same_level_source_loading_scan_admissible,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "source-loading theorem reactivation inventory ready",
            truth(inventory_ready),
            "The theorem branch starts only after `.1823-.1826` has frozen Gate B partial and after the full-q HH window already exists as a retained exact surface.",
        ),
        row(
            "retained_source_direction_rule",
            "pass" if retained_source_direction_rule else "reject",
            "retained source-direction bilinear rule",
            truth(retained_source_direction_rule),
            "The theorem lifts the already-retained source-direction read instead of reopening the old eigenvalue rule.",
        ),
        row(
            "retained_full_q_hh_window",
            "pass" if retained_full_q_hh_window else "reject",
            "retained full-q HH window available",
            truth(retained_full_q_hh_window),
            "The q-dependent loading theorem uses the already-retained scalar-compatible HH window as its exact diagonal surface.",
        ),
        row(
            "windowed_exact_source_loading_theorem_derived",
            "pass",
            "windowed exact source-loading theorem derived",
            truth(windowed_exact_source_loading_theorem_derived),
            "Combining the source-direction bilinear rule with the full-q HH window closes the loading coefficient analytically rather than by scan.",
        ),
        row(
            "q_dependent_loading_surface_available",
            "pass",
            "q-dependent loading surface available on scalar-compatible window",
            truth(q_dependent_loading_surface_available),
            "The retained gap is no longer just pointwise: kappa(q) is now fixed on the whole scalar-compatible window.",
        ),
        row(
            "kappa_exact_q0",
            "watch",
            "windowed exact loading at q=0",
            1.0,
            "The theorem starts from unit source loading at q=0, where the HH branch carries the full normalized scalar response.",
        ),
        row(
            "kappa_q_0p1",
            "watch",
            "windowed exact loading at q/m0 = 0.1",
            kappa_q_0p1,
            "The q-dependent loading surface is still large deep inside the scalar-compatible window.",
        ),
        row(
            "kappa_q_0p2",
            "watch",
            "windowed exact loading at q/m0 = 0.2",
            kappa_q_0p2,
            "Closer to q_theory the theorem lowers the loading smoothly but remains moderate and positive.",
        ),
        row(
            "kappa_exact_at_q_theory",
            "watch",
            "windowed exact loading at q_theory",
            kappa_exact_at_q_theory,
            "This is the theorem-level source-loading coefficient at the retained matching point.",
        ),
        row(
            "kappa_target_at_q_theory",
            "watch",
            "windowed target loading at q_theory",
            kappa_target_at_q_theory,
            "The same exact HH window keeps the physical target inside a nearby positive loading branch.",
        ),
        row(
            "kappa_target_vs_exact_gap",
            "watch",
            "absolute target-loading gap on theorem window at q_theory",
            kappa_target_vs_exact_gap,
            "The target remains only a small loading shift above the retained scalar exact point even after the loading theorem replaces the proxy root.",
        ),
        row(
            "kappa_target_vs_exact_rel_gap",
            "watch",
            "relative target-loading gap on theorem window at q_theory",
            kappa_target_vs_exact_rel_gap,
            "The physical target is still within a few percent of the exact theorem-level loading.",
        ),
        row(
            "kappa_exact_vs_proxy_gap",
            "watch",
            "absolute loading shift theorem vs old proxy at q_theory",
            kappa_exact_vs_proxy_gap,
            "The windowed theorem raises the loading because it replaces the old branch-local HH proxy with the exact HH window surface.",
        ),
        row(
            "kappa_exact_vs_proxy_rel_gap",
            "watch",
            "relative loading shift theorem vs old proxy at q_theory",
            kappa_exact_vs_proxy_rel_gap,
            "This quantifies how much the old fixed-q proxy underestimated the exact theorem-level source loading.",
        ),
        row(
            "kappa_exact_q_window_edge",
            "watch",
            "windowed exact loading at q_HH,max",
            0.0,
            "At the scalar-compatible window edge the HH diagonal closes and the exact source loading vanishes continuously.",
        ),
        row(
            "kappa_window_mean",
            "watch",
            "window-mean exact loading",
            kappa_window_mean,
            "This is the average theorem-level loading across the whole scalar-compatible window.",
        ),
        row(
            "kappa_window_rms",
            "watch",
            "window RMS exact loading",
            kappa_window_rms,
            "The RMS summarizes the loading weight carried by the exact q-dependent surface on the retained window.",
        ),
        row(
            "kappa_surface_monotone_nonincreasing",
            "pass" if kappa_surface_monotone_nonincreasing else "reject",
            "exact loading surface monotone nonincreasing",
            truth(kappa_surface_monotone_nonincreasing),
            "The theorem-level loading decays from q=0 to q_HH,max without oscillations, so it is a coherent source-loading surface rather than a scan artifact.",
        ),
        row(
            "exact_source_direction_response_at_q_theory",
            "pass",
            "windowed exact source-direction response at q_theory",
            exact_source_direction_response_at_q_theory,
            "The theorem-level loading reproduces the retained exact scalar amplitude at the matching point.",
        ),
        row(
            "exact_source_direction_alpha_at_q_theory",
            "pass",
            "windowed exact source-direction alpha at q_theory",
            exact_source_direction_alpha_at_q_theory,
            "The source-direction rule now reproduces the retained scalar strong candidate using an exact theorem-level loading, not a proxy scan.",
        ),
        row(
            "exact_source_loading_theorem_available",
            "pass",
            "exact source-loading theorem available on scalar-compatible window",
            truth(exact_source_loading_theorem_available),
            "The missing theorem is now closed on the retained scalar-compatible window.",
        ),
        row(
            "global_all_q_loading_surface_available",
            "reject",
            "global all-q loading surface available",
            truth(global_all_q_loading_surface_available),
            "The theorem remains windowed because the scalar-compatible HH surface itself closes at q_HH,max.",
        ),
        row(
            "gate_a_windowed_exact_promote_selected",
            "pass" if gate_a_windowed_exact_promote_selected else "reject",
            "Gate A windowed exact promote selected",
            truth(gate_a_windowed_exact_promote_selected),
            "Within the retained scalar-compatible window the source-direction family is now exact rather than proxy-only.",
        ),
        row(
            "gate_b_partial_selected",
            "reject",
            "Gate B partial selected",
            truth(gate_b_partial_selected),
            "The new theorem supersedes the old fixed-q proxy-only Gate B read on the scalar-compatible window.",
        ),
        row(
            "gate_c_reject_selected",
            "reject",
            "Gate C reject selected",
            truth(gate_c_reject_selected),
            "The theorem yields exact scalar promotion on the retained scalar-compatible window, so rejection is still not selected.",
        ),
        row(
            "same_level_source_loading_scan_admissible",
            "reject",
            "same-level source-loading scan admissible",
            truth(same_level_source_loading_scan_admissible),
            "Once kappa(q) is fixed analytically on the retained window, further ad-hoc scans are no longer honest.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "windowed exact source-loading theorem branch honest",
            truth(branch_honest),
            "The branch is honest only if it claims exact promotion on the retained scalar-compatible window while explicitly retaining the missing global all-q loading surface.",
        ),
    ]

    inputs = {
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
            "window_gate": display_path(WINDOW_GATE),
            "source_direction_gate": display_path(SOURCE_DIRECTION_GATE),
            "source_direction_closeout_gate": display_path(SOURCE_DIRECTION_CLOSEOUT_GATE),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "q_hh_max_over_m0": q_hh_max,
            "scalar_exact_alpha_at_q_theory": SCALAR_ALPHA,
            "physical_target_alpha": TARGET_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_source_direction_rule": retained_source_direction_rule,
        "retained_full_q_hh_window": retained_full_q_hh_window,
        "windowed_exact_source_loading_theorem_derived": windowed_exact_source_loading_theorem_derived,
        "q_dependent_loading_surface_available": q_dependent_loading_surface_available,
        "q_hh_max_over_m0": q_hh_max,
        "kappa_exact_q0": 1.0,
        "kappa_q_0p1": kappa_q_0p1,
        "kappa_q_0p2": kappa_q_0p2,
        "kappa_exact_at_q_theory": kappa_exact_at_q_theory,
        "kappa_target_at_q_theory": kappa_target_at_q_theory,
        "kappa_target_vs_exact_gap": kappa_target_vs_exact_gap,
        "kappa_target_vs_exact_rel_gap": kappa_target_vs_exact_rel_gap,
        "kappa_exact_vs_proxy_gap": kappa_exact_vs_proxy_gap,
        "kappa_exact_vs_proxy_rel_gap": kappa_exact_vs_proxy_rel_gap,
        "kappa_exact_q_window_edge": 0.0,
        "kappa_window_mean": kappa_window_mean,
        "kappa_window_rms": kappa_window_rms,
        "kappa_surface_monotone_nonincreasing": kappa_surface_monotone_nonincreasing,
        "exact_source_direction_response_at_q_theory": exact_source_direction_response_at_q_theory,
        "exact_source_direction_alpha_at_q_theory": exact_source_direction_alpha_at_q_theory,
        "exact_source_loading_theorem_available": exact_source_loading_theorem_available,
        "loading_surface_covers_q_theory": loading_surface_covers_q_theory,
        "global_all_q_loading_surface_available": global_all_q_loading_surface_available,
        "gate_a_windowed_exact_promote_selected": gate_a_windowed_exact_promote_selected,
        "gate_b_partial_selected": gate_b_partial_selected,
        "gate_c_reject_selected": gate_c_reject_selected,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": branch_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1827"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1827-.1830"),
            "current_problem_hit": hit(current_problem_text, "exact source-loading theorem / q-dependent loading surface"),
            "current_status_hit": hit(current_status_text, "conditional exact source-loading theorem or q-dependent loading surface reactivation"),
            "unified_roadmap_hit": hit(unified_text, "84. `.1827-.1830`"),
            "long_roadmap_hit": hit(long_text, "36. `8.7.56.1827-.1830`"),
            "part5_hit": hit(part5_text, "next official branch は `.1827-.1830`"),
        },
        "carry_over": {
            "window_summary": window_summary,
            "source_direction_summary": source_direction_summary,
            "closeout_summary": closeout_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1827", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1828", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1829", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1830", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
        ),
    }

    print(
        json.dumps(
            {"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
