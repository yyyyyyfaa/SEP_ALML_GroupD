"""Unit tests for the weighted_knn_shapley function."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_student.wknn_explainer2 import Weighted


@pytest.fixture
def model():
    """Pytest fixture that creates and returns an instance of the Weighted class.

    Initialized with a dataset of shape (10, 2) filled with ones and corresponding
    labels as an array of ones (integer type). This fixture can be used in tests
    that require a pre-defined Weighted model.
    """
    dataset = np.ones((10, 2))
    labels = np.ones(10, dtype = int)
    return Weighted(dataset, labels)

def test_weighted_knn_output_shape(model):
    """Test that the output shape and values of weighted_knn_shapley are as expected."""
    x_val = np.array([0.5, 0.5])
    y_val = 1
    gamma = 1.0
    K = 3

    result = model.weighted_knn_shapley(x_val, y_val, gamma, K)
    assert result.shape == (10,)
    assert np.all(np.isfinite(result))

def test_weighted_knn_zero_distance(model):
    """Test the behavior when all distances are zero (identical points)."""
    model.dataset = np.ones((10, 2))
    model.labels = np.ones(10, dtype = int)

    x_val = np.ones(2)
    y_val = 1
    gamma = 1.0
    K = 2

    with pytest.raises(ValueError, match="No samples remaining after filtering. Cannot compute Shapley values."):
        model.weighted_knn_shapley(x_val, y_val, gamma, K)

def test_weighted_knn_invalid_K(model):
    """Test that weighted_knn_shapley raises an exception when K is invalid (e.g., zero)."""
    x_val = np.array([0.5, 0.5])
    y_val = 1
    gamma = 1.0
    K = 0

    with pytest.raises(ValueError, match="K must be greater than 0."):
        model.weighted_knn_shapley(x_val, y_val, gamma, K)

def test_weighted_knn_K_larger(model):
    """Test the behavior when K is larger than the dataset size."""
    x_val = np.array([0.2, 0.2])
    y_val = 1
    gamma = 1.0
    K = 20 #larger than the dataset

    with pytest.raises(ValueError, match="K cannot be greater than number of samples N."):
        model.weighted_knn_shapley(x_val, y_val, gamma, K)

def test_weighted_mismatched_dimension(model):
    """Test that weighted_knn_shapley raises an exception for mismatched input dimensions."""
    x_val = np.array([0.5])
    y_val = 1
    gamma = 1.0
    K = 2

    with pytest.raises(ValueError, match="Feature dimension mismatch between x_val and dataset."):
        model.weighted_knn_shapley(x_val, y_val, gamma, K)

def test_prepare_data_removal_and_sorting():
    """Test that prepare_data correctly removes(x_val, y_val) and sorts by distance."""
    dataset = np.array([[0, 0], [1, 1], [2, 2]])
    labels = np.array([1, 0, 1])
    model = Weighted(dataset, labels)

    x_val = np.array([1, 1])
    y_val = 0

    X_filtered, Y_filterd, sorted_distances = model.prepare_data(x_val, y_val)

    #point [1, 1] with label 0 should be removed
    EXPECTED_FILTERED_SHAPE = 2
    assert X_filtered.shape[0] == EXPECTED_FILTERED_SHAPE
    assert not np.any((X_filtered == [1, 1]).all(axis = 1))

    #test distancesort
    assert np.allclose(sorted_distances, sorted(sorted_distances))

    #labels have to fit to the filtered X
    assert Y_filterd.shape == (2,)
    assert np.array_equal(Y_filterd, [1, 1])
