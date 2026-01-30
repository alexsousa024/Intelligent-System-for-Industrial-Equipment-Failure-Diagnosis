# Final - Intelligent System for Industrial Equipment Failure Diagnosis

This folder contains the end-to-end pipeline for industrial equipment failure diagnosis.
The system combines:
- Bayesian Networks (BN) for probabilistic diagnosis from sensor telemetry.
- A Knowledge Graph (KG) for maintenance procedures.
- Decision strategies to choose actions and maintenance procedures.

The main entry point is the demo app, which showcases the full pipeline.

## Demo (full pipeline)

The demo is a Streamlit app that loads the BN model and the KG, accepts CSV or manual input,
estimates overheat and causes, and recommends actions and procedures.

Run:

```bash
python -m pip install streamlit pandas rdflib pgmpy matplotlib graphviz
streamlit run Final/src/demo.py
```

If you only need inference and not visualization, the `graphviz` system package is optional.

## Project structure (this folder)

```
Final/
  data/                  # CSV data used for training, evaluation, and ontology
  models/                # Pretrained BN models in .bif format
  ontology.ttl           # Knowledge Graph ontology for procedures and components
  src/                   # Pipeline scripts (see below)
  *.ipynb                # Notebooks for data analysis and model development
```

### Key data files
- `Final/data/telemetry.csv`: raw sensor telemetry
- `Final/data/labels.csv`: labels for target variables
- `Final/data/causes.csv`, `components.csv`, `procedures.csv`, `relations.csv`: KG inputs
- `Final/data/test_data.csv`: example dataset used by the demo

### BN models
- `Final/models/bn_model_em_A.bif`: default BN used by the demo
- `Final/models/bn_model_em_B.bif`, `bn_model_em_from_mle.bif`, `bn_model_mle.bif`: alternatives

## Code guide (src/)

- `Final/src/demo.py`: Streamlit demo of the full pipeline (BN inference + KG + decision).
- `Final/src/demo_inference.py`: BN loader and inference helpers used by the demo.
- `Final/src/decision_common.py`: Decision strategies and KG query logic.
- `Final/src/compare_strategies.py`: Simulates many random cases and compares strategies.
- `Final/src/explain_example.py`: Example of inference explanation and outputs.
- `Final/src/bn_data_prepaparation.py`: Prepares data for BN training.
- `Final/src/bn_desing_and_training.py`: Builds and trains BN models.
- `Final/src/bn_evaluation.py`: Evaluates BN performance.
- `Final/src/bn_inference.py`: Core inference functions for BN models.

## How to use the pipeline

### 1) Run the demo (recommended)
This demonstrates the entire pipeline in one place.

```bash
streamlit run Final/src/demo.py
```

### 2) Compare decision strategies
Generates comparison tables and plots for different strategies.

```bash
python -u Final/src/compare_strategies.py
```

Outputs:
- `strategy_comparison_results.csv`
- `scheduled_procedures.csv`
- `probability_distributions_grid.png`
- `probability_distributions_overlay.png`
- `total_cost_by_strategy.png`
- `action_distribution_by_strategy.png`
- `procedure_distribution_by_strategy.png`

### 3) Train or evaluate BN models
Use the scripts in `Final/src/` to prepare data, train, and evaluate models.

```bash
python -u Final/src/bn_data_prepaparation.py
python -u Final/src/bn_desing_and_training.py
python -u Final/src/bn_evaluation.py
```

## What the system does

1. Reads telemetry (e.g., temperature, vibration, coolant flow).
2. Runs BN inference to estimate overheat risk and probable causes.
3. Queries the KG to find maintenance procedures for those causes.
4. Applies a decision strategy:
   - `threshold_based`: uses probability thresholds.
   - `cost_only`: chooses the cheapest maintenance action based on cause probability.

## Notes

- Update model and ontology paths in `Final/src/decision_common.py` if you want to switch models or KG.
- If you see `ModuleNotFoundError: graphviz`, install the Python package and (optionally) the system Graphviz binaries.
