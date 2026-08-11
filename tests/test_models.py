import numpy as np

from src.models import (
    run_isolation_forest,
    run_lof,
    run_dbscan,
    run_one_class_svm,
)


def create_test_data():
    """Create a small reproducible feature matrix."""

    rng = np.random.default_rng(42)

    normal_data = rng.normal(
        loc=0,
        scale=1,
        size=(50, 5)
    )

    anomalies = rng.normal(
        loc=6,
        scale=1,
        size=(5, 5)
    )

    return np.vstack([
        normal_data,
        anomalies
    ])


def test_isolation_forest_output_length():
    X = create_test_data()

    flags, scores = run_isolation_forest(X)

    assert len(flags) == len(X)
    assert len(scores) == len(X)


def test_isolation_forest_binary_flags():
    X = create_test_data()

    flags, _ = run_isolation_forest(X)

    assert set(np.unique(flags)).issubset({0, 1})


def test_lof_output_length():
    X = create_test_data()

    flags, scores = run_lof(X)

    assert len(flags) == len(X)
    assert len(scores) == len(X)


def test_lof_binary_flags():
    X = create_test_data()

    flags, _ = run_lof(X)

    assert set(np.unique(flags)).issubset({0, 1})


def test_dbscan_output_length():
    X = create_test_data()

    flags, labels = run_dbscan(X)

    assert len(flags) == len(X)
    assert len(labels) == len(X)


def test_dbscan_binary_flags():
    X = create_test_data()

    flags, _ = run_dbscan(X)

    assert set(np.unique(flags)).issubset({0, 1})


def test_one_class_svm_output_length():
    X = create_test_data()

    flags, scores = run_one_class_svm(X)

    assert len(flags) == len(X)
    assert len(scores) == len(X)


def test_one_class_svm_binary_flags():
    X = create_test_data()

    flags, _ = run_one_class_svm(X)

    assert set(np.unique(flags)).issubset({0, 1})
