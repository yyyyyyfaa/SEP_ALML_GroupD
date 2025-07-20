"""Gaussian  Imputer class."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import inv
from shapiq.games.imputer.base import Imputer

if TYPE_CHECKING:
    from shapiq.utils import Model


class GaussianImputer(Imputer):
    """Gaussian Conditional Imputer for missing value imputation.

    This imputer implements conditional mean imputation based on multivariate
    Gaussian distribution assumptions. It estimates the conditional expectation
    of missing features given observed features using maximum likelihood estimation.

    The imputer learns the mean vector and covariance matrix from complete training
    data, then uses the conditional distribution properties of multivariate Gaussian
    to fill missing values with their expected values given observed features.

    """

    def __init__(self, model: Model, data: np.ndarray, x: np.ndarray | None = None):
        """Initialize Gaussian Copula Imputer.

        Args:
            model (Model):
            data (np.ndarray):
            x (np.ndarray):

        """
        self.model = model
        self.data = data
        self._x = x
        self.n_players = data.shape[1]
        self.verbose = False

    def fit(self, x: np.ndarray, mask_data: np.ndarray | None = None):
        """Fit the Gaussian imputer by estimating distribution parameters.

        Estimates the mean vector and covariance matrix from the training data.
        Only uses complete samples (without missing values) for parameter estimation.

        Args:
            x :( np.ndarray of shape (n_features,)): The sample to be explained/imputed. Stored for later use.
            mask_data : (np.ndarray of shape (n_samples, n_features), optional): Boolean mask indicating missing values in training data.
            If None, assumes all training data is complete.

        Returns:
            self (GaussianImputer): Returns self for method chaining.
        """
        # save sample to explain
        self._x = x
        # only use the samples without any missing values to estimate
        if mask_data is not None:
            full_rows = ~mask_data.any(axis=1)
            X_full = self.data[full_rows]
        else:
            X_full = self.data
        self.mean = X_full.mean(axis=0)
        # every row only 1 sample and use no bias estimate
        self.CovMatrix = np.cov(X_full, rowvar=False, bias=False)
        return self

    def transform(self, X: np.ndarray, mask: np.ndarray| None = None) -> np.ndarray:
        """Transform data by imputing missing values with conditional means.

        For each sample, missing features are filled with their conditional
        expectation given observed features, based on the fitted Gaussian distribution.

        Args:
            X (np.ndarray of shape (n_samples, n_features)): Input data with potentially missing values (NaN).
            mask : (np.ndarray of shape (n_samples, n_features), optional): Boolean mask where True indicates missing values.
            If None, missing values are detected as NaN.

        Returns:
            X_imputed (np.ndarray of shape (n_samples, n_features)): Data with missing values filled by conditional means.
        """
        X_imp = X.copy()
        # Handle default mask
        if mask is None:
            mask = np.isnan(X_imp)
        n_samples, n_features = X.shape

        for i in range(n_samples):
            # find out the missing value
            miss_idx = np.where(mask[i])[0]
            if miss_idx.size == 0:
                continue
            obs_idx = np.where(~mask[i])[0]

            # 2.  submatrix accroding to "Multivariate Gaussian distribution"
            mu = self.mean
            Sigma = self.CovMatrix
            mu_M = mu[miss_idx]
            # if no observed features, fallback to unconditional mean
            if obs_idx.size == 0:
                X_imp[i, miss_idx] = mu_M
                continue

            mu_O = mu[obs_idx]
            Sigma_OO = Sigma[np.ix_(obs_idx, obs_idx)]
            Sigma_MO = Sigma[np.ix_(miss_idx, obs_idx)]

            # 3. compute mean value
            if np.linalg.matrix_rank(Sigma_OO) < Sigma_OO.shape[0]:
                cond_mean = mu_M
            else:
                inv_OO = inv(Sigma_OO)
                x_O = X_imp[i, obs_idx]
                cond_mean = mu_M + Sigma_MO.dot(inv_OO.dot(x_O - mu_O))
            X_imp[i, miss_idx] = cond_mean

            # fill with conditional mean only
            X_imp[i, miss_idx] = cond_mean
        return X_imp

    # Using imputer object as a method
    def __call__(self, coalitions: np.ndarray, verbose: bool = False) -> np.ndarray:  # noqa: ARG002, FBT001, FBT002
        """Make the imputer callable for use with shapiq framework.

        This method allows the imputer to be used as a game function in the
        shapiq framework. It takes coalitions (indicating which features to keep)
        and returns model predictions after imputing missing features.

        Args:
            coalitions (np.ndarray of shape (n_coalitions, n_features)): Boolean array where True indicates features to keep (observe),
            and False indicates features to set as missing (impute).
        verbose (bool, default=False): If True, print verbose output. Currently not used.

        Returns:
            predictions (np.ndarray of shape (n_coalitions,)): Model predictions for each coalition after imputation.
        """
        if isinstance(coalitions, list):
            coalitions = np.array(coalitions)
        if coalitions.dtype != bool:
            coalitions = coalitions.astype(bool)

        n_coalitions, n_features = coalitions.shape
        predictions = np.zeros(n_coalitions)

        for i, coalition in enumerate(coalitions):
            # coalition=True keep，False as missing
            mask = (~coalition).reshape(1, -1)
            X_cond = self.transform(self._x, mask=mask)
            predictions[i] = self.model(X_cond)[0]

        return predictions
