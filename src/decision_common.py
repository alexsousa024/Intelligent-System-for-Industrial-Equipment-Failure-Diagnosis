from typing import Dict, List, Tuple
import pandas as pd
from rdflib import Graph
from torch import argmin

CAUSES = {
    "K1": "BearingWearHigh",
    "K2": "FanFault",
    "K3": "CloggedFilter",
    "K4": "LowCoolingEfficiency"
}

ONTOLOGY_PATH = "ontology.ttl"
BN_MODEL_PATH = "../models/bn_model_em_A.bif"


def load_knowledge_graph(path: str) -> Tuple[Graph, int]:
    g = Graph()
    g.parse(path, format="turtle")
    return g, len(g)


def query_kg_for_procedures(kg: Graph, causes: List[str]) -> pd.DataFrame:
    if kg is None:
        return pd.DataFrame()

    causes_filter = " ".join([f":{cause}" for cause in causes])

    query = f"""
    PREFIX : <http://www.semanticweb.org/admin/ontologies/2025/11/untitled-ontology-6/>

    SELECT ?proc ?cause ?component ?hours ?cost ?risk WHERE {{
         VALUES ?cause {{ {causes_filter} }}

         ?proc :mitigatesCause ?cause ;
               :targetsComponent ?component ;
               :hasEffortHours ?hours ;
               :hasSparePartsEUR ?cost ;
               :hasRiskRating ?risk .
    }}
    ORDER BY ?cost
    """

    results = kg.query(query)

    procedures_data = []
    for row in results:
        proc_name = str(row.proc).split("/")[-1]
        cause_name = str(row.cause).split("/")[-1]
        component_name = str(row.component).split("/")[-1]

        procedures_data.append({
            "name": proc_name,
            "mitigates_cause": cause_name,
            "targets_component": component_name,
            "effort_h": float(row.hours),
            "spare_parts_cost_eur": int(row.cost),
            "risk_rating": int(row.risk),
        })

    df = pd.DataFrame(procedures_data)
    if len(df) > 0:
        df.insert(0, "proc_id", [f"P{i+1}" for i in range(len(df))])
    return df


def make_decision_threshold_based(kg: Graph, overheat_prob: float,
                                  causes_prob: Dict[str, float],
                                  overheat_threshold: float,
                                  cause_threshold: float) -> Dict:
    decision = {
        "action": None,
        "overheat_prob": overheat_prob,
        "causes_above_threshold": [],
        "procedures": None,
        "strategy_info": {
            "overheat_threshold": overheat_threshold,
            "cause_threshold": cause_threshold,
        }
    }

    causes_above = [cid for cid, p in causes_prob.items() if p >= cause_threshold]
    decision["causes_above_threshold"] = causes_above

    if overheat_prob < overheat_threshold:
        decision["action"] = "CONTINUE"
    elif overheat_prob >= overheat_threshold and len(causes_above) == 0:
        decision["action"] = "SLOW_DOWN"
    else:
        decision["action"] = "APPLY_PROCEDURES"
        #Choose the most likely cause
        most_likely_cause = max(causes_above, key=lambda cid: causes_prob[cid])
        #most likely cause name
        most_likely_cause_name = CAUSES.get(most_likely_cause, most_likely_cause)

        decision["procedures"] = query_kg_for_procedures(kg, [most_likely_cause_name])

    return decision


def make_decision_cost_only(kg: Graph, overheat_prob: float,
                            causes_prob: Dict[str, float],
                            overheat_threshold: float,
                            cause_threshold: float) -> Dict:
    """
    Cost-only (Threshold + Cheapest Procedure by J).

    Usa a mesma lógica de ação da threshold_based:
      - Se P(overheat) < overheat_threshold  -> CONTINUE
      - Se P(overheat) >= overheat_threshold e não há causas acima do threshold -> SLOW_DOWN
      - Caso contrário -> APPLY_PROCEDURES

    Quando APPLY_PROCEDURES, escolhe o procedimento com menor score J entre as causas acima do threshold:
        J = m0 * (m/m0)^(1 - P(C))

    Ignora tempos/downtime; usa apenas custo de peças e P(C).
    """
    decision = {
        "action": None,
        "overheat_prob": float(overheat_prob),
        "causes_above_threshold": [],
        "procedures": None,
        "strategy_info": {
            "overheat_threshold": float(overheat_threshold),
            "cause_threshold": float(cause_threshold),
        }
    }

    causes_above = [cid for cid, p in causes_prob.items() if float(p) >= float(cause_threshold)]
    decision["causes_above_threshold"] = causes_above

    if float(overheat_prob) < float(overheat_threshold):
        decision["action"] = "CONTINUE"
        decision["strategy_info"]["note"] = "Threshold logic: low overheat risk -> CONTINUE."
        return decision

    if float(overheat_prob) >= float(overheat_threshold) and len(causes_above) == 0:
        decision["action"] = "SLOW_DOWN"
        decision["strategy_info"]["note"] = "Threshold logic: overheat risk but no cause above threshold -> SLOW_DOWN."
        return decision

    decision["action"] = "APPLY_PROCEDURES"

    cause_names = [CAUSES[cid] for cid in causes_above]
    procs_df = query_kg_for_procedures(kg, cause_names)

    if procs_df is None or len(procs_df) == 0:
        decision["procedures"] = pd.DataFrame()
        decision["strategy_info"]["note"] = "No procedures found for causes above threshold."
        return decision

 

    proc_scores = []
    rows = []

    for _, proc in procs_df.iterrows():
        cause_id = None
        for cid, cname in CAUSES.items():
            if cname == proc["mitigates_cause"]:
                cause_id = cid
                break
        if cause_id is None:
            continue

        p_cause = float(causes_prob.get(cause_id, 0.0))
        m_parts = float(proc["spare_parts_cost_eur"])
        if m_parts <= 0:
            continue

        J = m_parts ** (1.0 - p_cause)
        proc_scores.append((J, cause_id, proc, p_cause, m_parts))
        rows.append({
            "cause_id": cause_id,
            "cause_name": CAUSES.get(cause_id, str(cause_id)),
            "procedure": proc["name"],
            "mitigates_cause": proc["mitigates_cause"],
            "m_parts_eur": float(m_parts),
            "P_cause": float(p_cause),
            "J_maint": float(J),
        })

    if not proc_scores:
        decision["procedures"] = procs_df
        decision["strategy_info"]["note"] = "No valid procedure scores computed."
        return decision

    proc_scores.sort(key=lambda x: x[0])
    best_J, best_cause_id, best_proc, best_p, best_m = proc_scores[0]

    maintenance_scores_df = pd.DataFrame(rows).sort_values("J_maint", ascending=True).reset_index(drop=True)

    decision["causes_above_threshold"] = [best_cause_id]
    decision["procedures"] = query_kg_for_procedures(kg, [CAUSES[best_cause_id]])

    decision["strategy_info"].update({
        "selected_cause": best_cause_id,
        "selected_procedure": best_proc["name"],
        "P_cause": float(best_p),
        "m_parts_eur": float(best_m),
        "best_J_maint": float(best_J),
        "maintenance_scores_top10": maintenance_scores_df.head(10),
        "note": "Threshold logic + cheapest (min J) procedure among causes above threshold."
    })

    return decision




def make_decision(kg: Graph, overheat_prob: float, causes_prob: Dict[str, float],
                  strategy: str, **params) -> Dict:
    if strategy == "threshold_based":
        return make_decision_threshold_based(
            kg, overheat_prob, causes_prob,
            params["overheat_threshold"], params["cause_threshold"]
        )
    if strategy == "cost_only":
        return make_decision_cost_only(
            kg, overheat_prob, causes_prob,
            params["overheat_threshold"], params["cause_threshold"]
        )
        
    raise ValueError(f"Unknown strategy: {strategy}")
