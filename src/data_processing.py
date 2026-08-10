import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {
    "Q1": [
        "Q1_Feature1",
        "Q1_Feature2",
        "Q1_Feature3",
        "Q1_Feature4",
        "Q1_Feature5",
    ],
    "Q2": [
        "Q2_Feature1",
        "Q2_Feature2",
        "Q2_Feature3",
        "Q2_Feature4",
        "Q2_Feature5",
    ],
}


def load_data(filepath):
    """
    Load the dealer-level warranty dataset.

    Parameters
    ----------
    filepath : str
        Path to the Excel dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded and cleaned dataset.
    """

    data = pd.read_excel(filepath)

    data = data.fillna(0)

    return data


def prepare_period_data(data, period):
    """
    Filter and scale features for a specific analysis period.

    Parameters
    ----------
    data : pandas.DataFrame
        Dealer-level dataset.

    period : str
        Analysis period, such as "Q1" or "Q2".

    Returns
    -------
    subset : pandas.DataFrame
        Filtered dealer data.

    X_scaled : numpy.ndarray
        Standardized feature matrix.

    features : list
        Features used for the selected period.
    """

    features = FEATURE_SETS[period]

    subset = data[
        data[f"{period}_RecordCount"] > 0
    ].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        subset[features]
    )

    return subset, X_scaled, features
