import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# ----------------------------------
# LOAD DATA
# ----------------------------------

data = pd.read_excel(
    "sample_dataset.xlsx"
)

data = data.fillna(0)

print("Dataset Shape:", data.shape)

# ----------------------------------
# FEATURE CONFIGURATION
# ----------------------------------

feature_sets = {
    "Q1": [
        "Q1_Feature1",
        "Q1_Feature2",
        "Q1_Feature3",
        "Q1_Feature4",
        "Q1_Feature5"
    ],
    "Q2": [
        "Q2_Feature1",
        "Q2_Feature2",
        "Q2_Feature3",
        "Q2_Feature4",
        "Q2_Feature5"
    ]
}

results = {}
feature_importance = {}

# ----------------------------------
# MODELING LOOP
# ----------------------------------

for period, features in feature_sets.items():

    print(f"\nProcessing {period}")

    subset = data[
        data[f"{period}_RecordCount"] > 0
    ].copy()

    scaler = StandardScaler()

    X = scaler.fit_transform(
        subset[features]
    )

    lof = LocalOutlierFactor(
        n_neighbors=10,
        contamination=0.10
    )

    predictions = lof.fit_predict(X)

    subset[f"{period}_Anomaly"] = (
        pd.Series(predictions)
        .replace({
            1: 0,
            -1: 1
        })
        .values
    )

    subset[f"{period}_Score"] = (
        -lof.negative_outlier_factor_
    )

    subset[f"{period}_Rank"] = (
        subset[f"{period}_Score"]
        .rank(
            ascending=False,
            method="first"
        )
        .astype(int)
    )

    # ----------------------------------
    # FEATURE IMPORTANCE
    # ----------------------------------

    normal_avg = (
        subset[
            subset[f"{period}_Anomaly"] == 0
        ][features]
        .mean()
    )

    anomaly_avg = (
        subset[
            subset[f"{period}_Anomaly"] == 1
        ][features]
        .mean()
    )

    importance = pd.DataFrame(
        {
            "Feature": features,
            "NormalAvg": normal_avg.values,
            "AnomalyAvg": anomaly_avg.values
        }
    )

    importance["Difference"] = (
        importance["AnomalyAvg"]
        - importance["NormalAvg"]
    )

    importance["AbsDifference"] = (
        importance["Difference"]
        .abs()
    )

    importance = importance.sort_values(
        "AbsDifference",
        ascending=False
    )

    importance["Rank"] = range(
        1,
        len(importance) + 1
    )

    feature_importance[period] = importance

    results[period] = subset

# ----------------------------------
# CREATE RESULTS TABLES
# ----------------------------------

q1_results = (
    results["Q1"][
        [
            "EntityID",
            "Q1_RecordCount",
            "Q1_Anomaly",
            "Q1_Score",
            "Q1_Rank"
        ]
    ]
    .sort_values("Q1_Rank")
)

q2_results = (
    results["Q2"][
        [
            "EntityID",
            "Q2_RecordCount",
            "Q2_Anomaly",
            "Q2_Score",
            "Q2_Rank"
        ]
    ]
    .sort_values("Q2_Rank")
)

# ----------------------------------
# COMPARISON
# ----------------------------------

comparison = (
    q1_results.merge(
        q2_results,
        on="EntityID",
        how="outer"
    )
)

comparison["EntityStatus"] = "Normal"

comparison.loc[
    (comparison["Q1_Anomaly"] == 1)
    &
    (comparison["Q2_Anomaly"] == 1),
    "EntityStatus"
] = "Persistent Anomaly"

comparison.loc[
    (comparison["Q1_Anomaly"] == 1)
    &
    (comparison["Q2_Anomaly"] == 0),
    "EntityStatus"
] = "Improved"

comparison.loc[
    (comparison["Q1_Anomaly"] == 0)
    &
    (comparison["Q2_Anomaly"] == 1),
    "EntityStatus"
] = "Emerging Anomaly"

# ----------------------------------
# EXPORT RESULTS
# ----------------------------------

output_file = (
    "lof_results.xlsx"
)

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    q1_results.to_excel(
        writer,
        sheet_name="Q1_Results",
        index=False
    )

    q2_results.to_excel(
        writer,
        sheet_name="Q2_Results",
        index=False
    )

    comparison.to_excel(
        writer,
        sheet_name="Comparison",
        index=False
    )

    feature_importance["Q1"].to_excel(
        writer,
        sheet_name="Q1_Feature_Importance",
        index=False
    )

    feature_importance["Q2"].to_excel(
        writer,
        sheet_name="Q2_Feature_Importance",
        index=False
    )

print("\nAnalysis Complete")
print(output_file)