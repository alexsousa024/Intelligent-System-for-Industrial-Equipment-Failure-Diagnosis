#!/usr/bin/env python3
"""
CLI demo: run BN inference + KG procedure lookup and print an explanation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from demo_inference import BayesianNetworkInference
from decision_common import (
    CAUSES,
    load_knowledge_graph,
    make_decision_threshold_based,
    query_kg_for_procedures,
)


CAUSE_LABELS = {
    "BearingWearHigh": "BearingWear",
    "FanFault": "FanFault",
    "CloggedFilter": "CloggedFilter",
    "LowCoolingEfficiency": "LowCoolingEfficiency",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_model_path() -> Path:
    return _project_root() / "models" / "bn_model_em_B.bif"


def _default_ontology_path() -> Path:
    return _project_root() / "ontology.ttl"


def _format_evidence(evidence: Dict[str, str]) -> str:
    labels = {
        "spindle_temp": "SpindleTemp",
        "vibration_rms": "Vibration",
        "coolant_flow": "CoolantFlow",
    }
    parts = [f"{labels[k]}={evidence[k]}" for k in ["spindle_temp", "vibration_rms", "coolant_flow"]]
    return " and ".join(parts)


def _choose_procedure(kg, cause_name: str) -> Optional[Dict]:
    procs = query_kg_for_procedures(kg, [cause_name])
    if procs is None or len(procs) == 0:
        return None
    return procs.iloc[0].to_dict()


def build_explanation(
    evidence: Dict[str, str],
    overheat_prob: float,
    causes_prob: Dict[str, float],
    action: str,
    procedure: Optional[Dict],
) -> str:
    likely_cause_id = max(causes_prob, key=causes_prob.get)
    likely_cause_name = CAUSES.get(likely_cause_id, likely_cause_id)
    cause_label = CAUSE_LABELS.get(likely_cause_name, likely_cause_name)

    because = _format_evidence(evidence)
    base = (
        f"Because {because}, the system estimates Overheat={overheat_prob:.2f}. "
        f"Likely cause is {cause_label}."
    )

    if action != "APPLY_PROCEDURES":
        return f"{base} Recommended action: {action}."

    if procedure is None:
        return f"{base} Recommended action: APPLY_PROCEDURES (no procedure found in KG)."

    effort_h = procedure.get("effort_h")
    cost_eur = procedure.get("spare_parts_cost_eur")
    proc_name = procedure.get("name")

    return (
        f"{base} Recommended action: {proc_name} "
        f"({effort_h:g}h, {int(cost_eur)} EUR)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explain a BN+KG diagnosis in a single sentence."
    )
    parser.add_argument("--spindle-temp", type=float, default=73.0)
    parser.add_argument("--vibration-rms", type=float, default=1.20)
    parser.add_argument("--coolant-flow", type=float, default=0.40)
    parser.add_argument("--overheat-threshold", type=float, default=0.6)
    parser.add_argument("--cause-threshold", type=float, default=0.5)
    parser.add_argument("--model", type=Path, default=_default_model_path())
    parser.add_argument("--ontology", type=Path, default=_default_ontology_path())
    args = parser.parse_args()

    kg, _ = load_knowledge_graph(str(args.ontology))
    bn = BayesianNetworkInference(str(args.model))

    instance = {
        "spindle_temp": args.spindle_temp,
        "vibration_rms": args.vibration_rms,
        "coolant_flow": args.coolant_flow,
    }

    evidence = {
        "spindle_temp": bn.discretize_value(args.spindle_temp, "spindle_temp"),
        "vibration_rms": bn.discretize_value(args.vibration_rms, "vibration_rms"),
        "coolant_flow": bn.discretize_value(args.coolant_flow, "coolant_flow"),
    }

    overheat_prob, causes_prob = bn.predict(instance)
    decision = make_decision_threshold_based(
        kg,
        overheat_prob,
        causes_prob,
        args.overheat_threshold,
        args.cause_threshold,
    )

    procedure = None
    if decision.get("action") == "APPLY_PROCEDURES":
        likely_cause_id = max(causes_prob, key=causes_prob.get)
        likely_cause_name = CAUSES.get(likely_cause_id, likely_cause_id)
        procedure = _choose_procedure(kg, likely_cause_name)

    explanation = build_explanation(
        evidence, overheat_prob, causes_prob, decision["action"], procedure
    )
    print(explanation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
