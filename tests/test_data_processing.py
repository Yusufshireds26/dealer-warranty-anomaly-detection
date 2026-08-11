import pandas as pd

from src.data_processing import prepare_period_data


def create_sample_data():
    return pd.DataFrame({
        "DealerID": [
            "A",
            "B",
            "C"
        ],

        "Q1_RecordCount": [
            10,
            20,
            30
        ],

        "Q2_RecordCount": [
            15,
            25,
            35
        ],

        "Q1_Feature1": [1.0, 2.0, 3.0],
        "Q1_Feature2": [2.0, 3.0, 4.0],
        "Q1_Feature3": [3.0, 4.0, 5.0],
        "Q1_Feature4": [4.0, 5.0, 6.0],
        "Q1_Feature5": [5.0, 6.0, 7.0],

        "Q2_Feature1": [1.5, 2.5, 3.5],
        "Q2_Feature2": [2.5, 3.5, 4.5],
        "Q2_Feature3": [3.5, 4.5, 5.5],
        "Q2_Feature4": [4.5, 5.5, 6.5],
        "Q2_Feature5": [5.5, 6.5, 7.5],
    })


def test_prepare_q1_data():
    data = create_sample_data()

    subset, X_scaled, features = prepare_period_data(
        data,
        "Q1"
    )

    assert len(subset) == 3
    assert X_scaled.shape == (3, 5)
    assert len(features) == 5


def test_prepare_q2_data():
    data = create_sample_data()

    subset, X_scaled, features = prepare_period_data(
        data,
        "Q2"
    )

    assert len(subset) == 3
    assert X_scaled.shape == (3, 5)
    assert len(features) == 5


def test_scaled_features_centered():
    data = create_sample_data()

    _, X_scaled, _ = prepare_period_data(
        data,
        "Q1"
    )

    means = X_scaled.mean(axis=0)

    assert all(
        abs(mean) < 1e-10
        for mean in means
    )
