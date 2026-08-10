from pathlib import Path
import sys

import pandas as pd

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_data, prepare_period_data
from src.models import (
    run_isolation_forest,
    run_lof,
    run_dbscan,
    run_one_class_svm,
)
from src.consensus import add_consensus_score


def run_pipeline(filepath, period="Q1"):
    """
    Run the complete dealer anomaly detection pipeline.
    """

    # 1. Load and prepare data
    data = load_data(filepath)

    results, X_scaled, features = prepare_period_data(
        data,
        period
    )

    # 2. Isolation Forest
    results["IF_Flag"], results["IF_Score"] = (
        run_isolation_forest(X_scaled)
    )

    # 3. Local Outlier Factor
    results["LOF_Flag"], results["LOF_Score"] = (
        run_lof(X_scaled)
    )

    # 4. DBSCAN
    results["DBSCAN_Flag"], results["DBSCAN_Cluster"] = (
        run_dbscan(X_scaled)
    )

    # 5. One-Class SVM
    results["SVM_Flag"], results["SVM_Score"] = (
        run_one_class_svm(X_scaled)
    )

    # 6. Multi-model consensus scoring
    results = add_consensus_score(results)

    return results


if __name__ == "__main__":

    input_file = PROJECT_ROOT / "data" / "sample_dataset.xlsx"

    results = run_pipeline(
        input_file,
        period="Q1"
    )

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "dealer_anomaly_results.csv"

    results.to_csv(
        output_file,
        index=False
    )

    print("Pipeline completed successfully.")
    print(f"Results saved to: {output_file}")
    print()
    print("Consensus Score Distribution:")
    print(
        results["ConsensusScore"]
        .value_counts()
        .sort_index()
    )
