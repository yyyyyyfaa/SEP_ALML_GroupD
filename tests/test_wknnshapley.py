"""Unit tests for the weighted_knn_shapley function."""

from __future__ import annotations

import unittest

import numpy as np
from sep_alml_groupd.knn_explainer2 import weighted_knn_shapley
import pytest


class TestWeightedKNN(unittest.TestCase):
    def setUp(self):
        self.model = weighted_knn_shapley()
        self.model.dataset = np.random.Generator(10, 2)
        self.model.labels = np.array([0, 1] * 5)

    def test_weighted_knn_output_shape(self):
        x_test = np.array([0.5, 0.5])
        y_test = 1
        gamma = 1.0
        K = 3

        result = self.model.weighted_knn_shapley(x_test, y_test, gamma, K)
        assert result.shape == (9,)
        assert np.all(np.isfinite(result))

    def test_weighted_knn_zero_distance(self):
        self.model.dataset = np.ones((10, 2))
        self.model.labels = np.ones(10, dtype = int)

        x_test = np.ones(2)
        y_test = 1
        gamma = 1.0
        K = 2

        result = self.model.weighted_knn_shapley(x_test, y_test, gamma, K)
        assert result.shape == (9,)
        assert np.all(result >= 0)

    def test_weighted_knn_invalid_K(self):
        x_test = np.array([0.5, 0.5])
        y_test = 1
        gamma = 1.0
        K = 0

        with pytest.raises(Exception):
            self.model.weighted_knn_shapley(x_test, y_test, gamma, K)

    def test_weighted_mismatched_dimension(self):
        x_test = np.array([0.5])
        y_test = 1
        gamma = 1.0
        K = 2

        with pytest.raises(Exception):
            self.model.weighted_knn_shapley(x_test, y_test, gamma, K)

if __name__ == "__main__":
    unittest.main()
