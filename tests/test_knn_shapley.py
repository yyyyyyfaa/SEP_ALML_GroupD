"""Unit tests for KNN Shapley and KNN Explainer functionality.

This module contains tests for the KNNShapley and KNNExplainer classes,
verifying correctness of Shapley value computations and API consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.knn_explainer import KNNExplainer
from shapiq_student.knn_shapley import KNNShapley


def test_unweighted_knn_shapley_single_manual():
    """Test knn_shapley_single : expected return [ -1/6, 1/3, 1/3 ]."""
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 1, 1])
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)

    x_test = np.array([1.0])
    pred = 1
    explainer = KNNShapley(model, X_train, y_train, pred)

    # Use single-point API
    sv = explainer.knn_shapley_single(x_test, pred)
    expected = np.array([-1/6, 1/3, 1/3])

    assert isinstance(sv, np.ndarray)
    assert sv.shape == (len(X_train),)
    assert np.allclose(sv, expected, atol=1e-6)


def test_unweighted_knn_shapley_k1_nearest_only():
    """When K=1, only the nearest point has Shapley value as 1, others be 0."""
    X_train = np.array([[0], [1], [2]])
    y_train = np.array([0, 1, 1])
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)

    x_test = np.array([0.5])
    pred = model.predict(x_test.reshape(1, -1))[0]
    explainer = KNNShapley(model, X_train, y_train, pred)

    shap = explainer.knn_shapley(x_test)
    nearest_idx = model.kneighbors(x_test.reshape(1, -1), return_distance=False)[0][0]

    # Only one non-zero
    nz = np.nonzero(shap)[0]
    assert list(nz) == [nearest_idx]
    assert pytest.approx(1.0, rel=1e-6) == shap[nearest_idx]


def test_list_input_consistency():
    """Lists input vs ndarray."""
    X_train_list = [[0.0], [1.0], [2.0]]
    y_train_list = [0, 1, 1]
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train_list, y_train_list)

    x_test = [1.5]
    pred = model.predict(np.atleast_2d(x_test))[0]
    explainer_list = KNNShapley(model, X_train_list, y_train_list, pred)
    explainer_array = KNNShapley(model, np.array(X_train_list), np.array(y_train_list), pred)

    shap_list = explainer_list.knn_shapley(x_test)
    shap_array = explainer_array.knn_shapley(np.array(x_test))
    np.testing.assert_allclose(shap_list, shap_array, atol=1e-6)

def test_single_training_sample_shapley():
    """When there is only 1 training sample, no matter what k is,.

    the shapley values should be 1.
    """
    # prepare a single training sample
    X_train = np.array([[0.0, 0.0]])
    y_train = np.array([1])
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)

    x_test = np.array([0.0, 0.0])
    y_pred = model.predict(x_test.reshape(1, -1))[0]
    explainer = KNNShapley(model, X_train, y_train, y_pred)

    sv = explainer.knn_shapley(x_test)
    assert sv.shape == (1,)
    # contribution shoule be 1
    assert pytest.approx(1.0, rel=1e-6) == sv[0]

def test_equidistant_samples_symmetry():
    """When 2 points have same dists and same labels,.

    their shapley values should be same.
    """
    # 3 points：[-1], [1] have same dists with point 0，third points is different
    X_train = np.array([[-1.0], [1.0], [10.0]])
    y_train = np.array([0, 0, 1])
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)

    x_test = np.array([0.0])
    y_pred = model.predict(x_test.reshape(1, -1))[0]
    explainer = KNNShapley(model, X_train, y_train, y_pred)

    sv = explainer.knn_shapley(x_test)
    # the points with same dists should have same shapley values
    assert pytest.approx(sv[0], rel=1e-6) == sv[1]

def test_invalid_input_type_raises_type_error():
    """Test that TypeError is raised when X contains invalid (non-numeric) input types."""
    #  string in X，y is normal
    X_train = np.array([[0.0, 1.0],
                        [1.0, 2.0],
                        ["a", "b"]], dtype=object)
    y_train = np.array([0, 1, 1])
    x_test  = np.array([0.5, 1.5])
    K = 2

    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train[:2].astype(float), y_train[:2])
    # create an instance for class KNNShapley
    explainer = KNNShapley(model, X_train, y_train, class_index=1)

    with pytest.raises(TypeError):
        _ = explainer.knn_shapley(x_test)

def test_unweighted_init_sets_mode_and_attributes():
    """Test that KNNExplainer initializes mode and attributes correctly for unweighted KNN."""
    X = np.array([[0.], [1.], [2.]])
    y = np.array([0, 1, 2])
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)

    explainer = KNNExplainer(model=model, data=X, labels=y, model_name="test")


    assert explainer.dataset is X
    assert explainer.labels is y
    assert explainer.model_name == "test"
    assert (explainer.N, explainer.M) == X.shape
    assert explainer.mode == "normal"

def test_explain_wraps_values_and_metadata():
    """Test that KNNExplainer.explain returns correct values and metadata."""
    X = np.array([[0.], [1.], [2.]])
    y = np.array([0, 1, 2])
    model = KNeighborsClassifier(n_neighbors=1)  # uniform weights → mode='normal'
    model.fit(X, y)

    explainer = KNNExplainer(model=model, data=X, labels=y)

    iv = explainer.explain(np.array([1.0]))
    # values
    expected = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(iv.values, expected, rtol=1e-6)
    N_PLAYERS = 3
    assert iv.n_players       == N_PLAYERS
    assert iv.min_order       == 1
    assert iv.max_order       == 1
    assert iv.index           == "SV"
    assert pytest.approx(iv.baseline_value) == 0.0










