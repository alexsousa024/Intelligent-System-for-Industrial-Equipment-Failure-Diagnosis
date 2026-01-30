# compare_strategies.py
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from demo_inference import load_bayesian_network, predict_with_bayesian_network
from decision_common import (
    BN_MODEL_PATH,
    CAUSES,
    ONTOLOGY_PATH,
    load_knowledge_graph,
    make_decision,
)



# =========================
# Random instance generator
# =========================

def generate_random_instances(n: int, seed: Optional[int] = 42) -> List[Dict]:
    """
    Gera instâncias aleatórias com ranges plausíveis.
    Ajusta os ranges se quiseres aproximar mais ao teu dataset real.
    """
    rng = random.Random() if seed is None else random.Random(seed)
    out = []

    for i in range(n):
        spindle_temp = rng.uniform(40.0, 120.0)      # °C
        vibration_rms = rng.uniform(0.0, 3.0)        # arbitrary
        coolant_flow = rng.uniform(0.0, 2.0)         # arbitrary

        out.append({
            "timestamp": datetime.utcnow().isoformat(),
            "machine_id": f"M-{rng.choice(['A','B','C','D'])}",
            "spindle_temp": spindle_temp,
            "vibration_rms": vibration_rms,
            "coolant_flow": coolant_flow,
            "ambient_temp": rng.uniform(10.0, 40.0),
            "feed_rate": rng.uniform(0.5, 1.5),
            "spindle_speed": rng.uniform(1000, 8000),
            "load_pct": rng.uniform(0.0, 1.0),
            "power_kw": rng.uniform(0.0, 10.0),
            "tool_wear": rng.uniform(0.0, 1.0),
        })

    return out


# =========================
# Evaluation runner
# =========================

def plot_probability_distributions(probs_df: pd.DataFrame, out_dir: str = ".") -> Dict[str, str]:
    cols = list(probs_df.columns)
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows), constrained_layout=True)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        ax.hist(probs_df[col], bins=20, alpha=0.8, color="#2A6F97", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")
        ax.set_xlim(0.0, 1.0)

    for ax in axes[len(cols):]:
        ax.axis("off")

    grid_path = f"{out_dir}/probability_distributions_grid.png"
    fig.suptitle("Probability Distributions (Overheat + Causes)", y=1.02)
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for col in cols:
        ax2.hist(probs_df[col], bins=20, alpha=0.35, label=col, edgecolor="white")
    ax2.set_title("Overlayed Probability Distributions")
    ax2.set_xlabel("Probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0.0, 1.0)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    overlay_path = f"{out_dir}/probability_distributions_overlay.png"
    fig2.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return {"grid": grid_path, "overlay": overlay_path}


def plot_total_costs(total_costs: Dict[str, float], out_dir: str = ".") -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(total_costs.keys())
    values = [total_costs[k] for k in labels]
    ax.bar(labels, values, color="#52796F", edgecolor="white")
    ax.set_title("Total Spare Parts Cost by Strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Total Cost (EUR)")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    path = f"{out_dir}/total_cost_by_strategy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_action_distribution(df: pd.DataFrame, strategies: List[str], out_dir: str = ".") -> str:
    action_counts = []
    actions = set()
    for strat in strategies:
        counts = df[f"{strat}_action"].value_counts(dropna=False)
        actions.update(counts.index.tolist())
        action_counts.append(counts)

    actions = sorted(actions)
    data = {strat: [action_counts[i].get(a, 0) for a in actions] for i, strat in enumerate(strategies)}
    plot_df = pd.DataFrame(data, index=actions)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax, width=0.85, edgecolor="white")
    ax.set_title("Action Distribution by Strategy")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.legend(title="Strategy", fontsize=9)
    path = f"{out_dir}/action_distribution_by_strategy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_procedure_distribution(scheduled_df: pd.DataFrame, out_dir: str = ".", top_k: int = 10) -> str:
    if scheduled_df.empty:
        return ""

    counts = (
        scheduled_df.groupby(["strategy", "procedure"])
        .size()
        .reset_index(name="count")
    )
    counts["procedure"] = counts["procedure"].fillna("")

    top_counts = (
        counts.sort_values(["strategy", "count"], ascending=[True, False])
        .groupby("strategy")
        .head(top_k)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for strat in top_counts["strategy"].unique():
        subset = top_counts[top_counts["strategy"] == strat]
        ax.bar(
            [f"{strat}: {p}" for p in subset["procedure"]],
            subset["count"],
            label=strat,
            edgecolor="white",
        )
    ax.set_title(f"Top 4 Procedures Scheduled by Strategy")
    ax.set_xlabel("Procedure")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(title="Strategy", fontsize=9)
    path = f"{out_dir}/procedure_distribution_by_strategy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_comparison(
    n: int = 100,
    seed: Optional[int] = 42,
    save_csv: bool = True,
    plot: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    results_dir = os.path.join("Final", "results")
    os.makedirs(results_dir, exist_ok=True)
    kg, triples = load_knowledge_graph(ONTOLOGY_PATH)
    print(f"✅ KG loaded: {triples} triples")

    bn = load_bayesian_network(BN_MODEL_PATH)
    if bn is None:
        raise RuntimeError("❌ Failed to load BN model.")
    print("✅ BN loaded")

    instances = generate_random_instances(n, seed=None)
    print(f"✅ Generated {len(instances)} random instances")

    strategies = {
        "threshold_based": dict(overheat_threshold=0.3, cause_threshold=0.3),
        "cost_only": dict(overheat_threshold=0.3, cause_threshold=0.3, m0_ref_eur=100.0),
    }

    scheduled_rows = []
    rows = []
    probs_rows = []
    for idx, inst in enumerate(instances):
        overheat_prob, causes_prob = predict_with_bayesian_network(bn, inst)

        row = {
            "idx": idx,
            "spindle_temp": inst["spindle_temp"],
            "vibration_rms": inst["vibration_rms"],
            "coolant_flow": inst["coolant_flow"],
            "overheat_prob": float(overheat_prob),
        }
        for k, v in causes_prob.items():
            row[f"P_{k}"] = float(v)

        probs_row = {"P_overheat": float(overheat_prob)}
        for k, v in causes_prob.items():
            probs_row[f"P_{k}"] = float(v)
        probs_rows.append(probs_row)

        # decisions
        for strat, params in strategies.items():
            d = make_decision(kg, overheat_prob, causes_prob, strat, **params)
            row[f"{strat}_action"] = d["action"]
            row[f"{strat}_cause"] = d["causes_above_threshold"][0] if d["causes_above_threshold"] else ""

            # Record scheduled procedures (when APPLY_PROCEDURES)
            proc_names = []
            proc_ids = []
            total_cost_eur = 0.0
            if d.get("action") == "APPLY_PROCEDURES":
                procs = d.get("procedures")
                if isinstance(procs, pd.DataFrame) and len(procs) > 0:
                    total_cost_eur = float(procs["spare_parts_cost_eur"].sum())
                    for _, pr in procs.iterrows():
                        proc_names.append(str(pr.get("name", "")))
                        proc_ids.append(str(pr.get("proc_id", "")))
                        scheduled_rows.append({
                            "idx": idx,
                            "strategy": strat,
                            "overheat_prob": float(overheat_prob),
                            "cause_id": row.get(f"{strat}_cause", ""),
                            "proc_id": str(pr.get("proc_id", "")),
                            "procedure": str(pr.get("name", "")),
                            "mitigates_cause": str(pr.get("mitigates_cause", "")),
                            "targets_component": str(pr.get("targets_component", "")),
                            "effort_h": float(pr.get("effort_h", 0.0)),
                            "spare_parts_cost_eur": float(pr.get("spare_parts_cost_eur", 0.0)),
                            "risk_rating": float(pr.get("risk_rating", 0.0)),
                        })

            row[f"{strat}_procedure_names"] = ";".join([p for p in proc_names if p])
            row[f"{strat}_procedure_ids"] = ";".join([p for p in proc_ids if p])
            row[f"{strat}_total_cost_eur"] = float(total_cost_eur)

            if strat == "cost_only":
                row["cost_only_best_J_maint"] = d.get("strategy_info", {}).get("best_J_maint", None)
        rows.append(row)

    df = pd.DataFrame(rows)
    probs_df = pd.DataFrame(probs_rows)

    scheduled_df = pd.DataFrame(scheduled_rows)

    print("\n=== Scheduled procedures (APPLY_PROCEDURES) ===")
    if len(scheduled_df) == 0:
        print("No procedures were scheduled.")
    else:
        # Counts per strategy
        print("\nCounts by strategy:")
        print(scheduled_df["strategy"].value_counts())

        # Top procedures per strategy
        print("\nTop procedures per strategy:")
        top = (scheduled_df
               .groupby(["strategy", "procedure"])\
               .size()\
               .reset_index(name="count")\
               .sort_values(["strategy", "count"], ascending=[True, False]))
        for strat in scheduled_df["strategy"].unique():
            print(f"\n[{strat}] top scheduled procedures")
            print(top[top["strategy"] == strat].head(10).to_string(index=False))

        print("\nSample scheduled rows:")
        print(scheduled_df.head(10).to_string(index=False))

    # Summary
    print("\n=== Action counts ===")
    for strat in strategies.keys():
        print(f"\n[{strat}]")
        print(df[f"{strat}_action"].value_counts(dropna=False))

    # Agreement
    def agreement(a: str, b: str) -> float:
        return (df[a] == df[b]).mean()

    print("\n=== Pairwise agreement (actions) ===")
    pairs = [
        ("threshold_based_action", "cost_only_action"),
        ("threshold_based_action", "cost_only_action"),
    ]
    for a, b in pairs:
        print(f"{a} vs {b}: {agreement(a, b):.2%}")

    total_costs = {
        strat: float(df[f"{strat}_total_cost_eur"].sum())
        for strat in strategies.keys()
    }
    print("\n=== Total spare parts cost by strategy ===")
    for strat, cost in total_costs.items():
        print(f"{strat}: €{cost:.2f}")

    if plot:
        plot_paths = plot_probability_distributions(probs_df, out_dir=results_dir)
        print("\n=== Probability distribution plots ===")
        print(f"Grid: {plot_paths['grid']}")
        print(f"Overlay: {plot_paths['overlay']}")
        cost_path = plot_total_costs(total_costs, out_dir=results_dir)
        action_path = plot_action_distribution(df, list(strategies.keys()), out_dir=results_dir)
        proc_path = plot_procedure_distribution(scheduled_df, out_dir=results_dir)
        print("\n=== Strategy plots ===")
        print(f"Total cost: {cost_path}")
        print(f"Action distribution: {action_path}")
        if proc_path:
            print(f"Procedure distribution: {proc_path}")
        else:
            print("Procedure distribution: (no procedures scheduled)")

    if save_csv:
        out_path = os.path.join(results_dir, "strategy_comparison_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\n💾 Saved: {out_path}")
        sched_path = os.path.join(results_dir, "scheduled_procedures.csv")
        scheduled_df.to_csv(sched_path, index=False)
        print(f"💾 Saved: {sched_path}")

    return df, total_costs


if __name__ == "__main__":
    run_comparison(n=1000, seed=0, save_csv=True)
