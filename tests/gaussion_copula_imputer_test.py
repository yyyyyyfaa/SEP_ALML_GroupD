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
        np.testing.assert_allclose(back, col,  rtol=1e-6, atol=1 - 1e-6)

    def test_transform_no_missing(self):
        data, x = simple_data()
        imputer = GaussianCopulaImputer(model=dummy_model, data= data)
        imputer.fit(x)
        x_imp = imputer.transform(x)
        assert np.allclose(x_imp, x)

    def test_transform_with_nan(self):
        data, x = simple_data()
        x_missing = x.copy()
        x_missing[0, 2] = np.nan
        imputer = GaussianCopulaImputer(model=dummy_model, data= data)
        imputer.fit(x)
        X_imp = imputer.transform(x_missing)

        assert not np.isnan(X_imp).any()
        assert X_imp.shape == x.shape

    def test_call_returns_prediction(self):
        data, x = simple_data()
        imputer = GaussianCopulaImputer(model=dummy_model, data= data)
        imputer.fit(x)
        coalitions = np.array([[True, False, True, True, False]])
        preds = imputer(coalitions)

        assert preds.shape == (1,)
        assert np.isfinite(preds).all()

    def test_transform_all_missing(self):
        data, x = simple_data()
        x_missing = np.full_like(x, np.nan)
        imputer = GaussianCopulaImputer(model=dummy_model, data=data)
        imputer.fit(x)
        x_imp = imputer.transform(x_missing)
        assert not np.isnan(x_imp).any()
        assert x_imp.shape == x.shape

    def test_transform_without_mask_defaults_to_nan(self):

        data, x = simple_data()
        imputer = GaussianCopulaImputer(model=dummy_model, data=data)
        imputer.fit(x)

        transformed = imputer.transform(x, mask=None)

        assert np.allclose(transformed, x)

    def test_call_with_multiple_coalitions(self):

        data, x = simple_data()
        imputer = GaussianCopulaImputer(model=dummy_model, data=data)
        imputer.fit(x)

        coalitions = [
            [True, True, False, False, True],
            [False, False, False, False, False],
            [True, True, True, True, True]
        ]
        preds = imputer(coalitions)

        assert preds.shape == (3,)
        assert np.isfinite(preds).all()

    def test_fit_with_mask_data_filters_rows(self):
        data, x = simple_data()
        mask_data = np.zeros_like(data, dtype=bool)
        mask_data[0, 0] = True

        imp = GaussianCopulaImputer(model=dummy_model, data=data)
        imp.fit(x, mask_data=mask_data)

        assert imp.mean.shape[0] == data.shape[1]

    def test_call_invokes_model_with_imputed_values(self):
        data, x = simple_data()
        imp = GaussianCopulaImputer(model=dummy_model, data=data)
        imp.fit(x)
        coalitions = np.array([[False, True]])
        preds = imp(coalitions)
        transformed = imp.transform(x, mask=np.array([[True, False]]))
        expected = np.sum(transformed, axis=1)
        assert preds[0] == pytest.approx(expected[0])
