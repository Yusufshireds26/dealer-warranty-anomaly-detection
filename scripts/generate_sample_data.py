from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "sample_dataset.xlsx"


def generate_sample_data(n_dealers=200, random_state=42):
    """
    Generate a synthetic dealer-level warranty dataset.

    The data is fully synthetic and contains no proprietary,
    confidential, or real dealer information.
    """

    rng = np.random.default_rng(random_state)

    dealer_ids = [f"DEALER_{i:03d}" for i in range(1, n_dealers + 1)]

    data = pd.DataFrame({
        "DealerID": dealer_ids,

        "Q1_RecordCount": rng.integers(20, 250, n_dealers),
        "Q2_RecordCount": rng.integers(20, 250, n_dealers),

        "Q1_Feature1": rng.normal(1000, 180, n_dealers),
        "Q1_Feature2": rng.normal(5.0, 1.2, n_dealers),
        "Q1_Feature3": rng.normal(0.15, 0.04, n_dealers),
        "Q1_Feature4": rng.normal(35, 8, n_dealers),
        "Q1_Feature5": rng.normal(0.08, 0.025, n_dealers),

        "Q2_Feature1": rng.normal(1050, 190, n_dealers),
        "Q2_Feature2": rng.normal(5.2, 1.3, n_dealers),
        "Q2_Feature3": rng.normal(0.16, 0.045, n_dealers),
        "Q2_Feature4": rng.normal(34, 8, n_dealers),
        "Q2_Feature5": rng.normal(0.085, 0.025, n_dealers),
    })

    # Inject a small set of intentionally unusual dealers
    # so the anomaly-detection models have meaningful patterns to detect.
    anomaly_indices = rng.choice(
        data.index,
        size=max(8, int(n_dealers * 0.05)),
        replace=False
    )

    data.loc[anomaly_indices, "Q1_Feature1"] *= rng.uniform(
        1.8, 2.8, len(anomaly_indices)
    )
    data.loc[anomaly_indices, "Q1_Feature2"] *= rng.uniform(
        1.5, 2.2, len(anomaly_indices)
    )
    data.loc[anomaly_indices, "Q1_Feature3"] *= rng.uniform(
        2.0, 3.5, len(anomaly_indices)
    )

    data.loc[anomaly_indices, "Q2_Feature1"] *= rng.uniform(
        1.7, 2.7, len(anomaly_indices)
    )
    data.loc[anomaly_indices, "Q2_Feature2"] *= rng.uniform(
        1.4, 2.1, len(anomaly_indices)
    )
    data.loc[anomaly_indices, "Q2_Feature3"] *= rng.uniform(
        2.0, 3.2, len(anomaly_indices)
    )

    # Keep ratio-like features within reasonable bounds
    ratio_columns = [
        "Q1_Feature3",
        "Q1_Feature5",
        "Q2_Feature3",
        "Q2_Feature5",
    ]

    for column in ratio_columns:
        data[column] = data[column].clip(lower=0)

    return data


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(exist_ok=True)

    sample_data = generate_sample_data()

    sample_data.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print(f"Synthetic dataset created: {OUTPUT_FILE}")
    print(f"Dealers generated: {len(sample_data)}")
    print(f"Columns generated: {len(sample_data.columns)}")
