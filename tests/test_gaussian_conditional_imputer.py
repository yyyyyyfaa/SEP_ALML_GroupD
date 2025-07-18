import numpy as np
import pytest

from shapiq_student.gaussian_conditional_imputer import GaussianConditionalImputer

# dummy callable model and background data for initialization
dummy_model = lambda X: X
dummy_data = dummy_data = np.zeros((2, 2))


def test_init_requires_callable_model():
    # model must be callable or have prediction function
    with pytest.raises(ValueError):
        GaussianConditionalImputer(model=5, data=dummy_data)


def test_fit_computes_mean_and_covariance():
    # Fit on complete data
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    imputer = GaussianConditionalImputer(dummy_model, dummy_data).fit(X)
    assert np.allclose(imputer.mean, [3.0, 4.0])
    expected_cov = np.cov(X, rowvar=False, bias=False)
    assert np.allclose(imputer.CovMatrix, expected_cov)


def test_transform_without_missing_leaves_data_unchanged():
    # No missing values returns identical array
    X = np.array([[0.1, -0.2],
                  [1.2,  0.3]])
    mask = np.zeros_like(X, dtype=bool)
    imputer = GaussianConditionalImputer(dummy_model, dummy_data).fit(X)
    X_imp = imputer.transform(X.copy(), mask=mask)
    assert np.array_equal(X_imp, X)


def test_transform_with_default_mask_and_nan():
    # mask=None should treat np.nan as missing
    X_full = np.array([[10.0, 20.0],
                       [30.0, 40.0],
                       [50.0, 60.0]])
    imputer = GaussianConditionalImputer(dummy_model, dummy_data).fit(X_full)
    X = np.array([[np.nan, 20.0]])
    X_imp = imputer.transform(X)
    assert not np.isnan(X_imp[0, 0])
    assert X_imp[0, 1] == 20.0


def test_regularization_prevents_singularity():
    # Zero covariance leads to mean fill
    imputer = GaussianConditionalImputer(dummy_model, dummy_data)
    imputer.mean = np.array([1.0, 2.0])
    imputer.CovMatrix = np.zeros((2, 2))
    X = np.array([[np.nan, 5.0]])
    mask = np.array([[True, False]])
    X_imp = imputer.transform(X.copy(), mask=mask)
    assert X_imp[0, 0] == pytest.approx(1.0)


def test_transform_multiple_missing_indices_reproducibility():
    # sampling should be reproducible given same seed
    imputer = GaussianConditionalImputer(dummy_model, dummy_data)
    imputer.mean = np.zeros(3)
    imputer.CovMatrix = np.eye(3)
    X = np.array([[np.nan, 1.0, np.nan]])
    mask = np.array([[True, False, True]])
    np.random.seed(42)
    first = imputer.transform(X.copy(), mask=mask)
    np.random.seed(42)
    second = imputer.transform(X.copy(), mask=mask)
    assert np.array_equal(first, second)


def test_transform_preserves_input_array():
    # transform should not modify original input
    X_full = np.array([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]])
    imputer = GaussianConditionalImputer(dummy_model, dummy_data).fit(X_full)
    X = np.array([[np.nan, 2.0]])
    mask = np.array([[True, False]])
    X_copy = X.copy()
    _ = imputer.transform(X, mask=mask)
    np.testing.assert_array_equal(X, X_copy)


def test_shape_preservation_and_dtype():
    # output shape and dtype
    X_full = np.random.rand(5, 4)
    imputer = GaussianConditionalImputer(dummy_model, dummy_data).fit(X_full)
    X = np.vstack([X_full, np.full((1, 4), np.nan)])
    X_imp = imputer.transform(X)
    assert X_imp.shape == X.shape
    assert X_imp.dtype == float



