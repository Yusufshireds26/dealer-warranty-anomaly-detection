from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM

def run_isolation_forest(X, contamination=0.10):
    """
    Run Isolation Forest anomaly detection.

    Parameters
    ----------
    X : array-like
        Scaled feature matrix.

    contamination : float
        Expected proportion of anomalies.

    Returns
    -------
    anomaly_flags : array
        1 = anomaly
        0 = normal

    anomaly_scores : array
        Isolation Forest anomaly scores.
    """

    model = IsolationForest(
        n_estimators=500,
        contamination=contamination,
        random_state=42
    )

    model.fit(X)

    predictions = model.predict(X)

    # sklearn:
    #  1 = normal
    # -1 = anomaly
    #
    # Convert to:
    #  0 = normal
    #  1 = anomaly
    anomaly_flags = (predictions == -1).astype(int)

    anomaly_scores = model.decision_function(X)

    return anomaly_flags, anomaly_scores

def run_lof(X, contamination=0.10, n_neighbors=20):
    """
    Run Local Outlier Factor anomaly detection.

    Returns
    -------
    anomaly_flags : array
        1 = anomaly
        0 = normal

    anomaly_scores : array
        LOF anomaly scores.
    """

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    predictions = model.fit_predict(X)

    anomaly_flags = (predictions == -1).astype(int)

    # More negative = more anomalous
    anomaly_scores = model.negative_outlier_factor_

    return anomaly_flags, anomaly_scores

def run_dbscan(X, eps=0.5, min_samples=5):
    """
    Run DBSCAN clustering for anomaly detection.

    DBSCAN labels noise points as -1.
    """

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X)

    anomaly_flags = (labels == -1).astype(int)

    return anomaly_flags, labels

def run_one_class_svm(X, nu=0.10):
    """
    Run One-Class SVM anomaly detection.
    """

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=nu
    )

    model.fit(X)

    predictions = model.predict(X)

    anomaly_flags = (predictions == -1).astype(int)

    anomaly_scores = model.decision_function(X)

    return anomaly_flags, anomaly_scores

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM


def run_isolation_forest(X, contamination=0.10):

    model = IsolationForest(
        n_estimators=500,
        contamination=contamination,
        random_state=42
    )

    model.fit(X)

    predictions = model.predict(X)

    anomaly_flags = (predictions == -1).astype(int)

    anomaly_scores = model.decision_function(X)

    return anomaly_flags, anomaly_scores


def run_lof(X, contamination=0.10, n_neighbors=20):

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    predictions = model.fit_predict(X)

    anomaly_flags = (predictions == -1).astype(int)

    anomaly_scores = model.negative_outlier_factor_

    return anomaly_flags, anomaly_scores


def run_dbscan(X, eps=0.5, min_samples=5):

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X)

    anomaly_flags = (labels == -1).astype(int)

    return anomaly_flags, labels


def run_one_class_svm(X, nu=0.10):

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=nu
    )

    model.fit(X)

    predictions = model.predict(X)

    anomaly_flags = (predictions == -1).astype(int)

    anomaly_scores = model.decision_function(X)

    return anomaly_flags, anomaly_scores

