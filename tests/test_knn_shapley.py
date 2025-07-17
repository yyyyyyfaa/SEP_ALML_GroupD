import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.knn_shapley import KNNShapley

def simple_dataset():
    """
    1D toy dataset: two 0 and two 1
    """
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])
    return X_train, y_train

def test_single_sample_k2_1d(simple_dataset):
    X_train, y_train = simple_dataset
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)

    x_test = np.array([0.1])
    pred = model.predict(x_test.reshape(1, -1))[0]
    explainer = KNNShapley(model, X_train, y_train, pred)
    shap = explainer.knn_shapley(x_test)

    # dtype and shape
    assert isinstance(shap, np.ndarray)
    assert shap.shape == (len(X_train),)

    # sum should be 1.0
    assert pytest.approx(1.0, rel=1e-6) == shap.sum()

    # only 2 not zero
    nz = np.nonzero(shap)[0]
    expected = set(model.kneighbors(x_test.reshape(1, -1),
                                    return_distance=False)[0])
    assert set(nz) == expected

def test_multiple_samples_average_fixed_class_index(simple_dataset):
    X_train, y_train = simple_dataset
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)

    X_test = np.array([[0.1], [2.9]])
    # class_index stable as first point's predic
    pred0 = model.predict(X_test[0].reshape(1, -1))[0]
    explainer = KNNShapley(model, X_train, y_train, pred0)
    shap = explainer.knn_shapley(X_test)

    assert shap.shape == (len(X_train),)
    # first contribution 1.0，second 0.0，after all sum should be 0.5
    assert pytest.approx(0.5, rel=1e-6) == shap.sum()

def test_k1_only_nearest(simple_dataset):
    X_train, y_train = simple_dataset
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)

    x_test = np.array([1.1])
    pred = model.predict(x_test.reshape(1, -1))[0]
    explainer = KNNShapley(model, X_train, y_train, pred)
    shap = explainer.knn_shapley(x_test)

    # sum 1.0
    assert pytest.approx(1.0, rel=1e-6) == shap.sum()

    # biggest Shapley value -> nearest neighbour
    nearest_idx = model.kneighbors(x_test.reshape(1, -1),
                                   return_distance=False)[0][0]
    assert nearest_idx == int(np.argmax(shap))

def test_none_class_index_all_zero(simple_dataset):
    X_train, y_train = simple_dataset
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)

    x_test = np.array([0.1])
    explainer = KNNShapley(model, X_train, y_train, None)
    shap = explainer.knn_shapley(x_test)
    assert shap.shape == (len(X_train),)
    assert np.all(shap == 0.0)

def test_input_types_consistency(simple_dataset):
    X_train, y_train = simple_dataset
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)

    x_list = [1.5]
    pred = model.predict(np.atleast_2d(x_list))[0]
    explainer_list  = KNNShapley(model, X_train.tolist(), y_train.tolist(), pred)
    explainer_array = KNNShapley(model, X_train,           y_train,           pred)

    out_list  = explainer_list.knn_shapley(x_list)
    out_array = explainer_array.knn_shapley(np.array(x_list))
    np.testing.assert_allclose(out_list, out_array)

def test_batch_and_single_equal(simple_dataset):
    X_train, y_train = simple_dataset
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)

    x_val = np.array([1.5])
    pred = model.predict(x_val.reshape(1, -1))[0]
    explainer = KNNShapley(model, X_train, y_train, pred)

    single = explainer.knn_shapley(x_val)
    batch  = explainer.knn_shapley(np.vstack([x_val, x_val]))
    np.testing.assert_allclose(single, batch)

def test_invalid_input_dimension_raises():
    X_train = np.array([[0, 1], [1, 0]])
    y_train = np.array([0, 1])
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    explainer = KNNShapley(model, X_train, y_train, 0)

    # input dim is false，sklearn should raise ValueError
    with pytest.raises(ValueError):
        explainer.knn_shapley(np.array([[0, 1, 2]]))


