from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class GammaConstraint:
    id: str
    label: str
    gamma: float
    sigma: float
    kind: str
    source: Dict[str, Any]

    def to_dict(self, *, beta: float, beta_sigma: float) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "gamma": float(self.gamma),
            "sigma": float(self.sigma),
            "beta": float(beta),
            "beta_sigma": float(beta_sigma),
            "source": dict(self.source),
        }


def beta_from_gamma(gamma: float, sigma: float) -> Tuple[float, float]:
    # PPN mapping: (1+gamma)=2β  ->  β=(1+gamma)/2
    beta = 0.5 * (1.0 + float(gamma))
    beta_sigma = 0.5 * abs(float(sigma))
    return beta, beta_sigma


def gamma_from_beta(beta: float, beta_sigma: float) -> Tuple[float, float]:
    gamma = float(2.0 * beta - 1.0)
    gamma_sigma = float(2.0 * abs(beta_sigma))
    return gamma, gamma_sigma


def _constraints_from_known_sources() -> List[GammaConstraint]:
    # Note: This is a fit/predict *policy* input. We keep it small and stable and cite the primary sources.
    retrieved_utc = "2026-01-04T00:00:00Z"
    return [
        GammaConstraint(
            id="cassini_2003_bertotti_nature425",
            label="Cassini（Bertotti 2003）",
            gamma=1.000021,
            sigma=0.000023,
            kind="primary_paper_ppn_gamma",
            source={
                "title": "A test of general relativity using radio links with the Cassini spacecraft",
                "authors": "Bertotti, Iess, Tortora",
                "year": 2003,
                "journal": "Nature",
                "volume": "425",
                "pages": "374–376",
                "doi": "10.1038/nature01997",
                "url": "https://doi.org/10.1038/nature01997",
                "retrieved_utc": retrieved_utc,
                "note": "論文の代表結果：γ−1=(2.1±2.3)×10^-5（統計）。ここでは γ=1.000021±0.000023 を拘束として用いる。",
            },
        ),
    ]


def _load_vlbi_best_from_solar_deflection_metrics(root: Path) -> Optional[GammaConstraint]:
    path = root / "output" / "theory" / "solar_light_deflection_metrics.json"
    if not path.exists():
        return None
    try:
        j = _read_json(path)
        m = dict(j.get("metrics") or {})
        gamma = float(m["observed_gamma_best"])
        sigma = float(m["observed_gamma_best_sigma"])
        label = str(m.get("observed_best_label") or "VLBI（best）")
        source = {
            "kind": "derived_from_output",
            "from": "output/theory/solar_light_deflection_metrics.json",
            "observed_best_id": m.get("observed_best_id"),
            "observed_best_label": label,
            "retrieved_utc": str(j.get("generated_utc") or ""),
        }
        return GammaConstraint(
            id="vlbi_best_from_solar_deflection_metrics",
            label=f"太陽光偏向（{label}）",
            gamma=gamma,
            sigma=sigma,
            kind="derived_ppn_gamma",
            source=source,
        )
    except Exception:
        return None


def _weighted_average(betas: List[Tuple[float, float]]) -> Tuple[float, float]:
    # inverse-variance weighting
    items = [(b, s) for (b, s) in betas if s > 0 and math.isfinite(b) and math.isfinite(s)]
    if not items:
        return 1.0, float("nan")
    wsum = sum(1.0 / (s * s) for (_, s) in items)
    mean = sum(b / (s * s) for (b, s) in items) / wsum
    sigma = math.sqrt(1.0 / wsum)
    return float(mean), float(sigma)


def main() -> int:
    root = _repo_root()
    default_outdir = root / "output" / "theory"
    default_out = default_outdir / "frozen_parameters.json"

    ap = argparse.ArgumentParser(description="Freeze global parameter beta for Phase 7 decisive pack.")
    ap.add_argument(
        "--beta-source",
        choices=["cassini2003", "vlbi_best", "weighted", "fixed1"],
        default="cassini2003",
        help="Which constraint to use for beta (default: cassini2003).",
    )
    ap.add_argument("--beta", type=float, default=None, help="Override beta (takes precedence).")
    ap.add_argument("--beta-sigma", type=float, default=None, help="Override beta sigma (takes precedence).")
    ap.add_argument("--out", type=str, default=str(default_out), help="Output JSON path (default: output/theory/frozen_parameters.json)")
    args = ap.parse_args()

    # Constraints
    constraints: List[GammaConstraint] = []
    constraints.extend(_constraints_from_known_sources())
    vlbi_best = _load_vlbi_best_from_solar_deflection_metrics(root)
    if vlbi_best:
        constraints.append(vlbi_best)

    # Choose beta
    beta: float
    beta_sigma: float
    beta_source: str

    if args.beta is not None:
        beta = float(args.beta)
        beta_sigma = float(args.beta_sigma) if args.beta_sigma is not None else float("nan")
        beta_source = "override_cli"
    elif args.beta_source == "fixed1":
        beta = 1.0
        beta_sigma = 0.0
        beta_source = "fixed_beta_1"
    elif args.beta_source == "vlbi_best":
        if not vlbi_best:
            raise SystemExit("VLBI best constraint not found (solar_light_deflection_metrics.json missing).")
        beta, beta_sigma = beta_from_gamma(vlbi_best.gamma, vlbi_best.sigma)
        beta_source = "vlbi_best"
    elif args.beta_source == "weighted":
        beta_candidates: List[Tuple[float, float]] = []
        for c in constraints:
            b, s = beta_from_gamma(c.gamma, c.sigma)
            beta_candidates.append((b, s))
        beta, beta_sigma = _weighted_average(beta_candidates)
        beta_source = "weighted_constraints"
    else:
        # cassini2003 default
        cassini = next((c for c in constraints if c.id.startswith("cassini_2003")), None)
        if not cassini:
            raise SystemExit("Cassini constraint not found (internal definition missing).")
        beta, beta_sigma = beta_from_gamma(cassini.gamma, cassini.sigma)
        beta_source = "cassini2003"

    gamma, gamma_sigma = gamma_from_beta(beta, beta_sigma if math.isfinite(beta_sigma) else 0.0)

    out_path = Path(args.out)
    payload_constraints: List[Dict[str, Any]] = []
    for c in constraints:
        b, s = beta_from_gamma(c.gamma, c.sigma)
        payload_constraints.append(c.to_dict(beta=b, beta_sigma=s))

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "fit_predict_separation": True,
            "beta_source": beta_source,
            "note": "Phase 7（決定的検証パック）では、ここで凍結したβを以後固定して外挿（predict）する。速度飽和δ0は拡張仮説として別途扱う。",
        },
        "beta": float(beta),
        "beta_sigma": float(beta_sigma) if math.isfinite(beta_sigma) else None,
        "gamma_pmodel": float(gamma),
        "gamma_pmodel_sigma": float(gamma_sigma) if math.isfinite(gamma_sigma) else None,
        "constraints": payload_constraints,
        "outputs": {
            "frozen_parameters_json": str(out_path).replace("\\", "/"),
        },
    }

    _write_json(out_path, payload)

    # Work log (machine readable)
    try:
        from scripts.summary.worklog import append_event

        append_event(
            {
                "event_type": "freeze_parameters",
                "argv": list(sys.argv),
                "inputs": {
                    "solar_light_deflection_metrics_json": (root / "output" / "theory" / "solar_light_deflection_metrics.json"),
                },
                "params": {
                    "beta_source": beta_source,
                    "beta": beta,
                    "beta_sigma": beta_sigma if math.isfinite(beta_sigma) else None,
                    "gamma_pmodel": gamma,
                },
                "outputs": {"frozen_parameters_json": out_path},
            }
        )
    except Exception:
        pass

    print(f"Wrote: {out_path}")
    print(f"beta={beta} (sigma={beta_sigma}) -> gamma={gamma}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
