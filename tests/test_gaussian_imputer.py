"""
Test suite for GaussianImputer implementation.

This module contains comprehensive tests for the GaussianImputer class,
verifying its functionality for conditional mean imputation based on
multivariate Gaussian distribution assumptions.

The tests cover:
- Parameter estimation (mean and covariance)
- Missing value imputation accuracy
- Edge cases (no missing values, degenerate covariance)
- Integration with shapiq framework via __call__ method
"""
import numpy as np
import pytest
from shapiq.games.imputer.base import Imputer
from shapiq_student.imputer import GaussianImputer

class DummyModel:
    """
    Simple test model that returns sum of features.
    Used for testing imputer functionality without complex model dependencies.
    """
    def __call__(self, X):
        """
        Return sum of features for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Sum of features for each sample.
        """
        # Sum of features for testing
        return np.sum(X, axis=1)

def generate_simple_data():
    """
    Generate simple 2D test data with known statistics.

    Creates a small dataset with predictable mean and covariance
    for testing parameter estimation accuracy.

    Returns
    -------
    data : np.ndarray of shape (3, 2)
        Training data with known statistics.
    mask_data : None
        No missing values in training data.
    x : np.ndarray of shape (1, 2)
        Sample to be explained/imputed.
    """
    # 2D data with known mean and covariance
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # No missing in training
    mask_data = None
    # Single explain instance
    x = np.array([[7.0, 8.0]])
    return data, mask_data, x

def test_fit_sets_mean_and_covariance():
    """
    Test that fit() correctly estimates mean and covariance from data.
    Verifies that the imputer learns the correct distributional parameters
    from complete training data.
    """
    data, mask_data, x = generate_simple_data()
    imp = GaussianImputer(model=DummyModel(), data=data)
    imp.fit(x, mask_data)
    # mean of data
    expected_mean = np.array([3.0, 4.0])
    # covariance of data rows
    expected_cov = np.cov(data, rowvar=False, bias=False)
    np.testing.assert_allclose(imp.mean, expected_mean)
    np.testing.assert_allclose(imp.CovMatrix, expected_cov)

def test_transform_no_missing_leaves_data_unchanged():
    """
    Test that transform() doesn't modify data without missing values.
    Ensures that complete data passes through unchanged, preserving
    the original values exactly.
    """
    data, _, x = generate_simple_data()
    imp = GaussianImputer(model=DummyModel(), data=data).fit(x)
    X = np.array([[10.0, 20.0]])
    X_imp = imp.transform(X.copy())
    np.testing.assert_array_equal(X_imp, X)

def test_transform_single_missing_value():
    """
    Test conditional mean imputation for single missing feature.
    Verifies that the imputed value matches the theoretical conditional
    expectation from multivariate Gaussian distribution.
    """
    data, _, x = generate_simple_data()
    imp = GaussianImputer(model=DummyModel(), data=data).fit(x)
    # Mask first feature
    X = np.array([[np.nan, 10.0]])
    mask = np.array([[True, False]])
    X_imp = imp.transform(X, mask=mask)
    # Conditional expectation: E[X0 | X1=10] = mu0 + Sigma01 / Sigma11 * (10 - mu1)
    mu = imp.mean
    Sigma = imp.CovMatrix
    cond = mu[0] + Sigma[0,1]/Sigma[1,1]*(10.0 - mu[1])
    assert X_imp[0,0] == pytest.approx(cond)
    assert X_imp[0,1] == 10.0

def test_transform_degenerate_covariance_fallback():
    """
    Test fallback behavior when covariance matrix is singular.
    When features are perfectly correlated (degenerate covariance),
    the imputer should fall back to unconditional mean.
    """
    # Create degenerate covariance: identical features
    data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    x = np.array([[0.0, 0.0]])
    imp = GaussianImputer(model=DummyModel(), data=data).fit(x)
    # Mask both features: should fallback to unconditional mean
    X = np.array([[np.nan, np.nan]])
    mask = np.array([[True, True]])
    X_imp = imp.transform(X, mask=mask)
    expected_mean = imp.mean
    np.testing.assert_allclose(X_imp[0], expected_mean)

def test_call_invokes_model_with_imputed_values():
    """
    Test __call__ method integration with shapiq framework.
    Verifies that coalitions are correctly interpreted, missing values
    are imputed, and the model is called with imputed data.
    """
    data, _, x = generate_simple_data()
    imp = GaussianImputer(model=DummyModel(), data=data).fit(x)
    # Coalition to keep only second feature
    coalitions = np.array([[False, True]])
    # After imputation, X_cond = [cond0, x1]
    preds = imp(coalitions)
    # Model returns sum
    # cond0 = E[X0|X1=x1] computed by transform
    transformed = imp.transform(x, mask=np.array([[True, False]]))
    expected = np.sum(transformed, axis=1)
    assert preds[0] == pytest.approx(expected[0])

if __name__ == "__main__":
    pytest.main()





