"""Unit tests for the weighted_knn_shapley function."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_student.wknn_explainer2 import Weighted


def model():
    model = Weighted()
    model.dataset = np.ones((10, 2))
    model.labels = np.ones(10, dtype = int)
    return model

def test_weighted_knn_output_shape(model):
    """Test that the output shape and values of weighted_knn_shapley are as expected."""
    x_val = np.array([0.5, 0.5])
    y_val = 1
    gamma = 1.0
    K = 3

    result = model.weighted_knn_shapley(x_val, y_val, gamma, K)
    assert result.shape == (9,)
    assert np.all(np.isfinite(result))

def test_weighted_knn_zero_distance(model):
    """Test the behavior when all distances are zero (identical points)."""
    model.dataset = np.ones((10, 2))
    model.labels = np.ones(10, dtype = int)

    x_val = np.ones(2)
    y_val = 1
    gamma = 1.0
    K = 2

    result = model.weighted_knn_shapley(x_val, y_val, gamma, K)
    assert result.shape == (9,)
    assert np.all(result >= 0)

def test_weighted_knn_invalid_K(model):
    """Test that weighted_knn_shapley raises an exception when K is invalid (e.g., zero)."""
    x_val = np.array([0.5, 0.5])
    y_val = 1
    gamma = 1.0
    K = 0

    with pytest.raises(ValueError, match="K must be greater than 0"):
        model.weighted_knn_shapley(x_val, y_val, gamma, K)

def test_weighted_knn_K_larger(model):
    """Test the behavior when K is larger than the dataset size."""
    x_val = np.array([0.2, 0.2])
    y_val = 1
    gamma = 1.0
    K = 20

    result = model.weighted_knn_shapley(x_val, y_val, gamma, K)

    assert np.all(result <= 0)

def test_weighted_mismatched_dimension(model):
    """Test that weighted_knn_shapley raises an exception for mismatched input dimensions."""
    x_val = np.array([0.5])
    y_val = 1
    gamma = 1.0
    K = 2

    with pytest.raises(ValueError, match="dimension|shape|size|mismatch"):
        model.weighted_knn_shapley(x_val, y_val, gamma, K)
