'''
Bayesian Network Design and Training Functions, including:

Model structure definition:
    - Latent causes definition
    - Edges definition
    - Model columns definition

Model Training:
    - Extract compatible CPDs between models --> for MLe to EM transfer learning
    - Train BN model with MLE or EM
    - Save and load model to/from BIF file
    - Print CPDs table formatting and plotting Bayesian Networks with graphviz
'''

from itertools import product

from pgmpy.readwrite import BIFWriter, BIFReader

from pgmpy.factors.discrete import TabularCPD
from graphviz import Digraph

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization

### -------------------------- Bayesian Network Structure Definition -------------------------- ###

def get_latent_causes(supervised, printing):
    latent_causes = {
        "BearingWearHigh",
        "FanFault",
        "CloggedFilter",
        "LowCoolingEfficiency",
    }
    if not supervised:
        latent_causes.add("spindle_overheat")
    if printing:
        print("Latent_causes = {")
        for cause in latent_causes:
            print("       ", cause)
        print("}")
    return latent_causes

def get_edges(with_latent_causes, with_maintenance, printing):
    edges = [
        #Sensors --> Overheat
        ("spindle_temp","spindle_overheat"),
        ("coolant_flow","spindle_overheat"),
        ("vibration_rms","spindle_overheat"),
        ]
    if with_latent_causes:
        edges += [
            ("BearingWearHigh", "vibration_rms"),
            ("FanFault", "spindle_temp"),
            ("CloggedFilter", "coolant_flow"), 
            ("LowCoolingEfficiency", "spindle_temp"),
        ]
    if with_maintenance:
        edges.append(("action_type","spindle_overheat"))
        edges.append(("duration_h","spindle_overheat"))
        edges.append(("success","spindle_overheat"))
    
    if printing:
        print("Edges =[")
        for edge in edges:
            print("       ", edge)
        print("]")
    return edges

def get_cols_model(supervised, with_maintenance, printing):
    cols_model = [
        "spindle_temp",
        "coolant_flow",
        "vibration_rms",
        "spindle_speed",
    ]
    if supervised:
        cols_model.append("spindle_overheat")

    if with_maintenance:
        cols_model.append("duration_h")
        cols_model.append("action_type")
        cols_model.append("success")
        
    if printing:    
        print("Columns to the model = [")
        for col in cols_model:
            print("       ", col)
        print("]")
    return cols_model

def build_bayesian_network(edges, latent_causes=None):
    assert len(edges) > 0, "Edge list cannot be empty"
    model = DiscreteBayesianNetwork(edges)

    if latent_causes:
        for node in latent_causes:
            if node not in model.nodes():
                model.add_node(node)
        model.latents = set(latent_causes)
    return model


### -------------------------- Bayesian Network Model Training -------------------------- ###

def extract_compatible_cpds(source_model, target_model):

    compatible = []
    for cpd in source_model.get_cpds():
        var = cpd.variable
        if var not in target_model.nodes():
            continue

        # Parents must match exactly
        src_parents = set(cpd.variables[1:])
        tgt_parents = set(target_model.get_parents(var))

        if src_parents != tgt_parents:
            continue
        compatible.append(cpd)

    return compatible

def train_bn(model, data, cols_model, estimator_name, max_iter=100, init_model=None):
    df_bn = data[cols_model].copy()
    if estimator_name == "mle":
        model.fit(df_bn, estimator=MaximumLikelihoodEstimator)

    elif estimator_name == "em":
        if init_model is not None:
            cpds_init = extract_compatible_cpds(
                    init_model,
                    model
                )
            if cpds_init:
                model.add_cpds(*cpds_init)

        em = ExpectationMaximization(model, df_bn)
        cpds = em.get_parameters(max_iter=max_iter)
        model.add_cpds(*cpds)

    model.check_model()
    return model

def save_model(model, filename):
    if not filename.endswith(".bif"):
        filename += ".bif"

    writer = BIFWriter(model)
    writer.write_bif(filename)

def load_model(filename):
    if not filename.endswith(".bif"):
        raise ValueError("File must be a .bif file")

    reader = BIFReader(filename)
    model = reader.get_model()
    return model

def print_cpds(model):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, s: s

    try:
        for cpd in model.get_cpds():
            print("\n===================================")
            print(f"Variable: {cpd.variable}")
            print(cpd)
    finally:
        TabularCPD._truncate_strtable = backup

def format_cpd_table(cpd, decimals=3):

    var = cpd.variable
    parents = cpd.variables[1:]
    states = cpd.state_names
    vals = cpd.values

    # Root node
    if not parents:
        text = f"{var}\n"
        for s, p in zip(states[var], vals.flatten()):
            text += f"{s}: {round(float(p),decimals)}\n"
        return text

    parent_states = [states[p] for p in parents]
    flat = vals.reshape(len(states[var]), -1)

    # Header
    text = f"{var}\n"
    text += " | ".join(map(str, parents)) + " | " \
          + " | ".join(map(str, states[var])) + "\n"
    text += "-" * 50 + "\n"

    for i, comb in enumerate(product(*parent_states)):
        row = " | ".join(map(str, comb))

        probs = flat[:, i]
        row += " | " + " | ".join(
            str(round(float(p),decimals)) for p in probs
        )
        text += row + "\n"

    return text

def plot_bn_with_table_cpds(model):
    dot = Digraph(
        "Bayesian Network",
        format="png",
        graph_attr={"rankdir": "LR"}
    )

    # Nodes
    for node in model.nodes():
        dot.node(node, node, shape="ellipse")

    # Edges
    for a,b in model.edges():
        dot.edge(a,b)

    # CPTs
    for cpd in model.get_cpds():
        label = format_cpd_table(cpd)

        dot.node(
            f"cpd_{cpd.variable}",
            label=label,
            shape="box",
            style="rounded,filled",
            fillcolor="lightyellow",
            fontname="Courier"
        )

        dot.edge(
            cpd.variable,
            f"cpd_{cpd.variable}",
            style="dotted"
        )

    return dot
