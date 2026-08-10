import pandas as pd


def add_consensus_score(results):
    """
    Combine anomaly flags from all four models into a consensus score.

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame containing anomaly flags from each model.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ConsensusScore and RiskTier added.
    """

    flag_columns = [
        "IF_Flag",
        "LOF_Flag",
        "DBSCAN_Flag",
        "SVM_Flag"
    ]

    results["ConsensusScore"] = results[flag_columns].sum(axis=1)

    results["RiskTier"] = pd.cut(
        results["ConsensusScore"],
        bins=[-1, 0, 1, 2, 4],
        labels=[
            "Normal",
            "Low",
            "Medium",
            "High"
        ]
    )

    results = results.sort_values(
        "ConsensusScore",
        ascending=False
    )

    return results
