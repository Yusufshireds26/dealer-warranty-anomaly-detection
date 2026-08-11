import pandas as pd

from src.consensus import add_consensus_score


def create_results():
    return pd.DataFrame({
        "DealerID": [
            "DEALER_A",
            "DEALER_B",
            "DEALER_C",
            "DEALER_D",
            "DEALER_E"
        ],
        "IF_Flag": [0, 1, 1, 1, 1],
        "LOF_Flag": [0, 0, 1, 1, 1],
        "DBSCAN_Flag": [0, 0, 0, 1, 1],
        "SVM_Flag": [0, 0, 0, 0, 1],
    })


def test_consensus_scores():
    results = create_results()

    output = add_consensus_score(results)

    scores = sorted(
        output["ConsensusScore"].tolist()
    )

    assert scores == [0, 1, 2, 3, 4]


def test_consensus_score_range():
    results = create_results()

    output = add_consensus_score(results)

    assert output["ConsensusScore"].min() >= 0
    assert output["ConsensusScore"].max() <= 4


def test_risk_tier_created():
    results = create_results()

    output = add_consensus_score(results)

    assert "RiskTier" in output.columns


def test_high_consensus_is_high_risk():
    results = create_results()

    output = add_consensus_score(results)

    highest = output.iloc[0]

    assert highest["ConsensusScore"] == 4
    assert highest["RiskTier"] == "High"
