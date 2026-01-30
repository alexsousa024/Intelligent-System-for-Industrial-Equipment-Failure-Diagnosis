'''
Bayesian Network Inference, including:
    - Load model from BIF file
    - Build evidence from input data
    - Infer target variable probability given evidence
    - Label risk level based on target probability
    - Infer probable latent causes given evidence

'''

from itertools import product
import pandas as pd

from pgmpy.readwrite import BIFReader

from pgmpy.factors.discrete import TabularCPD
from graphviz import Digraph

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination


def load_model(filename):
    if not filename.endswith(".bif"):
        raise ValueError("File must be a .bif file")

    reader = BIFReader(filename)
    model = reader.get_model()
    return model


def build_evidence(row, features):
    return {f: row[f] for f in features}

def infer_target(model, infer, target, evidence):
    q = infer.query([target], evidence=evidence)
    states = model.get_cpds(target).state_names[target]

    df = pd.DataFrame({
        target: states,
        "Probability": q.values
    })

    p_yes = q.values[states.index("Yes")]
    return df, p_yes


def risk_label(p_yes, low_threshold=0.3, high_threshold=0.6):
    if p_yes < low_threshold:
        return "LOW"
    elif p_yes < high_threshold:
        return "MEDIUM"
    else:
        return "HIGH"
    

def infer_causes(model, infer, causes, evidence, top_k=3):
    cause_probs = {}

    for c in causes:
        q = infer.query([c], evidence=evidence)
        states = model.get_cpds(c).state_names[c]

        idx = states.index("Yes") if "Yes" in states else 1
        cause_probs[c] = q.values[idx]

    df = (
        pd.DataFrame.from_dict(cause_probs, orient="index", columns=["Probability"])
          .sort_values("Probability", ascending=False)
          .head(top_k)
    )

    return df


def display_inference_results(evidence, target_df, risk, causes_df):

    
    print("\n================ CASE ANALYSIS ================")

    print("\nEvidence:")
    display(pd.DataFrame.from_dict(evidence, orient="index", columns=["Value"]))

    print("\nTarget probability:")
    display(target_df)

    print(f"\nRisk level: {risk}")

    if not causes_df.empty:
        print("\nProbable causes:")
        display(causes_df)

    print("==============================================\n")



def infer_case(model, row, features, target, causes, threshold=0.5, top_k=3):

    infer = VariableElimination(model)
    evidence = build_evidence(row, features)
    target_df, p_yes = infer_target(model, infer, target, evidence)
    risk = risk_label(p_yes)
    causes_df = pd.DataFrame()
    causes_df = infer_causes(model, infer, causes, evidence, top_k)

    display_inference_results(evidence, target_df, risk, causes_df)

    return p_yes
