"""Unit tests for the weighted_knn_shapley function."""

from __future__ import annotations
import numpy as np
from sep_alml_groupd.shapley import weighted_knn_shapley


def test_weighted_knn_shapley_output_shape():
    x_train = np.random.Generator(10, 2)
    y_train = np.array([0, 1] * 5)
    x_test = np.array([0.5, 0.5])
    y_test = 1
    gamma = 1.0
    K = 3

    result = weighted_knn_shapley(x_train, y_train, x_test, y_test, gamma, K)

    assert result.shape == (10,)
    assert np.all(np.isfinite(result))

def test_weighted_knn_shapley_all_zero_distance():
    x_train = np.ones((10, 2))
    y_train = np.array([1]*10)
    x_test = np.ones(2)
    y_test = 1
    gamma = 1.0
    K = 2

    result = weighted_knn_shapley(x_train, y_train, x_test, y_test, gamma, K)
    assert np.all(result >= 0)
