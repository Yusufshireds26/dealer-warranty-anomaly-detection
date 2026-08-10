from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt




PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "dealer_anomaly_results.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)




def load_results():
    """Load anomaly detection pipeline results."""

    return pd.read_csv(RESULTS_FILE)



def plot_consensus_distribution(results):
    """
    Plot the number of dealers receiving each
    multi-model consensus score.
    """

    counts = (
        results["ConsensusScore"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(
        counts.index.astype(str),
        counts.values
    )

    # Add count labels above bars
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    ax.set_xlabel("Number of Models Flagging Dealer")
    ax.set_ylabel("Number of Dealers")

    ax.set_title(
        "Multi-Model Anomaly Detection Consensus",
        fontweight="bold",
        pad=22
    )

    ax.text(
        0.5,
        1.01,
        "Higher scores indicate stronger agreement across four anomaly detection models",
        transform=ax.transAxes,
        ha="center",
        fontsize=10
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.set_axisbelow(True)

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "consensus_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Created: {output_file}")




def plot_model_flags(results):
    """
    Compare the number of dealers flagged
    by each anomaly detection model.
    """

    model_counts = {
        "Isolation Forest": int(results["IF_Flag"].sum()),
        "LOF": int(results["LOF_Flag"].sum()),
        "DBSCAN": int(results["DBSCAN_Flag"].sum()),
        "One-Class SVM": int(results["SVM_Flag"].sum())
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(
        model_counts.keys(),
        model_counts.values()
    )

    # Add values above bars
    for bar, value in zip(bars, model_counts.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    ax.set_ylabel("Dealers Flagged")

    ax.set_title(
        "Anomaly Detection Model Comparison",
        fontweight="bold",
        pad=22
    )

    ax.text(
        0.5,
        1.01,
        "Number of synthetic dealers independently flagged by each model",
        transform=ax.transAxes,
        ha="center",
        fontsize=10
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.set_axisbelow(True)

    plt.xticks(rotation=10)

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "model_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Created: {output_file}")




def plot_top_risk_dealers(results, top_n=15):
    """
    Display the dealers receiving the highest
    multi-model anomaly consensus scores.
    """

    # Rank primarily by model agreement
    # and secondarily by Isolation Forest score.
    top_dealers = (
        results
        .sort_values(
            ["ConsensusScore", "IF_Score"],
            ascending=[False, True]
        )
        .head(top_n)
        .copy()
    )

    # Reverse order so highest-ranked dealer appears at the top
    top_dealers = top_dealers.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    bars = ax.barh(
        top_dealers["DealerID"],
        top_dealers["ConsensusScore"]
    )

    # Add consensus score labels
    for bar, value in zip(
        bars,
        top_dealers["ConsensusScore"]
    ):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(int(value)),
            va="center",
            fontweight="bold"
        )

    ax.set_xlabel("Consensus Score")
    ax.set_ylabel("Synthetic Dealer")

    ax.set_title(
        "Highest-Ranked Dealer Anomalies",
        fontweight="bold",
        pad=22
    )

    ax.text(
        0.5,
        1.01,
        "Dealers prioritized by agreement across multiple anomaly detection models",
        transform=ax.transAxes,
        ha="center",
        fontsize=10
    )

    ax.set_xlim(
        0,
        max(4.5, top_dealers["ConsensusScore"].max() + 0.5)
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.set_axisbelow(True)

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "top_risk_dealers.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Created: {output_file}")



if __name__ == "__main__":

    print("Loading anomaly detection results...")

    results = load_results()

    print(f"Dealers loaded: {len(results)}")
    print()

    plot_consensus_distribution(results)
    plot_model_flags(results)
    plot_top_risk_dealers(results)

    print()
    print("All visualizations created successfully.")