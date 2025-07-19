from scipy.stats import norm

from shapiq_student import GaussianCopulaImputer
import pytest
import numpy as np
import pandas as pd


def dummy_model(x: np.ndarray) -> np.ndarray:
    """A dummy model that returns the sum of the features.

    Note:
        This callable is just here that we satisfy the Imputer's model parameter and tha we can
    check if the Imputer can be called with coalitions and returns a vector of "predictions".

    Args:
    x   : Input data as a 2D numpy array with shape (n_samples, n_features).

    Returns:
        A 1D numpy array with the sum of the features for each sample.
    """
    return np.sum(x, axis=1)

def simple_data():
    np.random.seed(0)
    data = np.random.rand(100, 5)
    x = np.random.rand(1, 5)
    return data, x


class TestGaussianCopulaImputer:
    """Tests for calling the GaussianCopulaImputer class."""

    def test_fit_sets_correct_mean_covariance(self):
        data, X = simple_data()
        imp = GaussianCopulaImputer(model=dummy_model, data=data)
        imp.fit(X)

        assert imp.mean.shape == (data.shape[1],)
        assert imp.CovMatrix.shape == (data.shape[1], data.shape[1])

    def test_fit_correct_mean_covariance(self):
        data, x = simple_data()
        imp = GaussianCopulaImputer(model=dummy_model, data=data)
        imp.fit(x)

        # Berechne erwartete Mittelwerte auf den ECDF-transformierten Daten
        V = np.zeros_like(data)
        for j in range(data.shape[1]):
            V[:, j] = imp.ecdf_transform(data[:, j])
        expected_mean = V.mean(axis=0)
        expected_cov = np.cov(V, rowvar=False, bias=False)

        np.testing.assert_allclose(imp.mean, expected_mean, rtol=1e-5)
        np.testing.assert_allclose(imp.CovMatrix, expected_cov, rtol=1e-5)

    def test_ecdf_forward_backward(self):
        data,_ = simple_data()
        col = data[:, 0]
        imp = GaussianCopulaImputer(model=dummy_model, data = data)
        v = imp.ecdf_transform(col)
        quantile = norm.cdf(v)
        back = imp.inverse_ecdf(col, quantile)
        np.testing.assert_allclose(back, col)

    def test_fit_with_simple_data(self):
        data, mask_data, x = simple_data()
        imputer = GaussianCopulaImputer(dummy_model, data)
        imputer.fit(x)
        assert imputer.CovMatrix.shape == (2, 2)
        assert np.all(np.isfinite(imputer.CovMatrix))

    def test_transform_no_missing(self):
        data, x = simple_data()
        imputer = GaussianCopulaImputer(model=dummy_model, data= data)
        imputer.fit(x)
        x_imp = imputer.transform(x)
        assert np.allclose(x_imp, x)

    def test_call_with_simple_data(self):
        data, mask_data, x = simple_data()
        imputer = GaussianCopulaImputer(model=dummy_model,data=  data)
        imputer.fit(x)
        coalitions = np.array([[True, False], [False, True]])
        preds = imputer(coalitions)
        assert preds.shape == (2,)
        assert np.all(np.isfinite(preds))