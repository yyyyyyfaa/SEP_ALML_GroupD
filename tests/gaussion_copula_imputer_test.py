from __future__ import annotations

"""Test class for Gaussian Copula Imputer."""

import numpy as np
from scipy.stats import norm

from shapiq_student import GaussianCopulaImputer

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
    """Generates data for testing purposes.

    Returns: tuple[np.ndarray, np.ndarray]: training data.
    """
    rng = np.random.default_rng(seed=0)
    data = rng.random((100, 5))
    x = rng.random((1, 5))
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
        """Tests if transform does not change data."""
        data, x = simple_data()
        imputer = GaussianCopulaImputer(model=dummy_model, data= data)
        imputer.fit(x)
        # Data should remain the same
        x_imp = imputer.transform(x)
        assert np.allclose(x_imp, x)

    def test_call_returns_prediction(self):
        """Tests if the __call__ method returns valid prediction."""
        data, x = simple_data()
        imputer = GaussianCopulaImputer(model=dummy_model, data= data)
        imputer.fit(x)
        coalitions = np.array([[True, False, True, True, False]])
        preds = imputer(coalitions)

        assert preds.shape == (1,)
        assert np.isfinite(preds).all()

    def test_transform_all_missing(self):
        """Tests imputation if all values are missing in a row."""
        data, x = simple_data()
        x_missing = np.full_like(x, np.nan)
        imputer = GaussianCopulaImputer(model=dummy_model, data=data)
        imputer.fit(x)
        x_imp = imputer.transform(x_missing)
        assert not np.isnan(x_imp).any()
        assert x_imp.shape == x.shape


    def test_fit_with_mask_data_filters_rows(self):
        """Tests if rows with missing values are excluded while fitting."""
        data, x = simple_data()
        mask_data = np.zeros_like(data, dtype=bool)
        mask_data[0, 0] = True

        imp = GaussianCopulaImputer(model=dummy_model, data=data)
        imp.fit(x, mask_data=mask_data)

        assert imp.mean.shape[0] == data.shape[1]

    def test_coalitions_type_conversion(self):
        """Tests that integer-type coalitions are converted to boolean."""
        data, x = simple_data()
        imputer = GaussianCopulaImputer(model=dummy_model, data=data)
        imputer.fit(x)

        coalitions = np.array([[1, 0, 1, 1, 0]])  # dtype = int

        preds = imputer(coalitions)

        assert preds.shape == (1,)
        assert np.isfinite(preds).all()

    def test_cond_mean_fallback_if_covariance_singular(self):
        """Tests if fallback is used when covariance is singular."""
        data, x = simple_data()
        imputer = GaussianCopulaImputer(model=dummy_model, data=data)
        imputer.fit(x)

        data[:, 1] = data[:, 0]
        x[0, 1] = x[0, 0]

        coalition = [False, False, True, True, True]
        coalitions = [coalition]

        preds = imputer(coalitions)

        assert preds.shape == (1,)
        assert np.isfinite(preds).all()
