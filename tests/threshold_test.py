"""Unit tests for the Threshold class and its Shapley value computations.

This module contains tests for:
- Shapley value calculation with various neighbor scenarios,
- Multiclass support,
- Comparison against expected results.
"""

from __future__ import annotations

from _pytest import unittest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tests_grading.conftest import *  # noqa: F403

from shapiq_student.threshold import Threshold


# Test shapley values
def test_shapley_threshold(data_train, x_explain, knn_basic):
    """Test Shapley value calculation for the Threshold class with a basic KNN and sample data.

    Ensures the result is a non-zero numpy array of the correct shape.
    """
    x_train, y_train = data_train
    threshold = 10.0
    class_index = int(knn_basic.predict(x_explain)[0])

    explainer = Threshold(knn_basic, x_train, y_train, class_index, threshold)
    result = explainer.threshold_knn_shapley(x_explain[0])

    assert isinstance(result, np.ndarray)
    assert result.shape == (x_train.shape[0],)
    assert np.any(result != 0)


# Test no training points are in radius
def test_shapley_no_neighbors(data_train, knn_basic):
    """Test Shapley value calculation when no training points are within the threshold radius.

    Ensures all Shapley values are zero when the query point is far from all training data.
    """
    x_train, y_train = data_train
    far_point = np.array([1000.0] * x_train.shape[1])
    threshold = 0.01  # winzig
    class_index = 0

    explainer = Threshold(knn_basic, x_train, y_train, class_index, threshold)
    result = explainer.threshold_knn_shapley(far_point)
    # All values should be zero since no neighbors are found
    assert np.all(result == 0.0)


# Test one training point in radius
def test_shapley_with_only_one_neighbor(data_train, knn_basic):
    """Test Shapley value calculation when only one training point is within the threshold radius.

    Ensures the result has the correct shape and contains non-zero Shapley values.
    """
    x_train, y_train = data_train
    x_query = x_train + 1e-8
    threshold = 0.01
    class_index = y_train

    explainer = Threshold(knn_basic, x_train, y_train, class_index, threshold)
    result = explainer.threshold_knn_shapley(x_query)
    # Result should have valid shape and contain non-zero Shapley values
    assert result.shape == (x_train.shape[0],)
    assert np.any(result != 0)


# Test Shapley values for a multiclass
def test_shapley_multiclass(data_train_multiclass, x_explain_multiclass, knn_basic_multiclass):
    """Test Shapley value calculation for the Threshold class with multiclass data.

    Ensures the result has the correct shape and contains non-zero Shapley values for multiclass scenarios.
    """
    x_train, y_train = data_train_multiclass
    threshold = 10.0
    class_index = int(knn_basic_multiclass.predict(x_explain_multiclass)[0])

    explainer = Threshold(knn_basic_multiclass, x_train, y_train, class_index, threshold)
    result = explainer.threshold_knn_shapley(x_explain_multiclass[0])

    assert result.shape == (x_train.shape[0],)
    assert np.any(result != 0)


# Test the threshold Shapley result against calculated expected result
def test_shapley_result_against_expected():
    """Test the threshold Shapley result against a manually calculated expected result.

    Ensures the computed Shapley values match the expected values within a tolerance.
    """
    X = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([0, 1, 0])
    threshold = 1.5
    class_index = 1

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)

    explainer = Threshold(knn, X, y, class_index, threshold)
    result = explainer.threshold_knn_shapley(np.array([1.0, 1.0]))
    expected = np.array([-0.5, 0.666667, -0.5])

    np.testing.assert_allclose(result, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
