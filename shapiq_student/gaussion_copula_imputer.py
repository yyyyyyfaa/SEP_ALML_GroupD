from __future__ import annotations

"""Gaussian Copula Imputer class"""

from typing_extensions import TYPE_CHECKING

import numpy as np
from numpy.linalg import inv
from shapiq.games.imputer.base import Imputer
from scipy.stats import norm, rankdata

class GaussianImputer(Imputer):

    def __init__(self, model, data: np.ndarray, x: np.ndarray | None = None):
        self.model = model
        self.data = data
        self._x = x
        self.n_players = data.shape[1]
        self.randomState = np.random.RandomState(42)
        self.verbose = False

    def fit(self, x: np.ndarray):
        self._x = x
        return self

    # Using imputer object as a method
    def __call__(self, coalitions: np.ndarray, verbose: bool = False) -> np.ndarray:
        if isinstance(coalitions, list):
            coalitions = np.array(coalitions)
        if coalitions.dtype != bool:
            coalitions = coalitions.astype(bool)

        n_coalitions, n_features = coalitions.shape
        predictions = np.zeros(n_coalitions)

        for i, coalition in enumerate(coalitions):
            x_masked = self._x.copy()
            x_masked[0, ~coalition] = 0  # set inactive features to 0
            predictions[i] = self.model(x_masked)[0]

        return predictions


class GaussianCopulaImputer(Imputer):

    def __init__(self, model, data: np.ndarray, x: np.ndarray | None = None):
        self.model = model
        self.data = data
        self._x = x
        self.n_players = data.shape[1]
        self.randomState = np.random.RandomState(42)
        self.verbose = False
        self.n_features = data.shape[1]

    def ecdf_transform(self, x: np.ndarray) -> np.ndarray:
        """Compute empirical CDF transform tp normal margins.

        Args:
            x (np.ndarray): Input data vector.

        Returns:
             np.ndarray: Transformed data to standard normal space.
        """
        x = np.asarray(x).flatten()
        ranks = rankdata(x, method='average') # Ranking the values
        uniform = np.clip(ranks / len(x), 1e-6, 1 - 1e-6)
        return norm.ppf(uniform)

    def inverse_ecdf(self, x: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Back transforms with ecdf using quantiles.

        Args:
            x (np.ndarray): Input data vector.
            values (np.ndarray): Quantile values.

        Returns:
            np.ndarray: Inversed-transformed values.
        """
        # Approximate inverse ECDF using quantiles
        sorted_x = np.sort(x)
        quantiles = np.clip(values, 1e-6, 1 - 1e-6)  # avoid infs
        return np.quantile(sorted_x, quantiles)

    def fit(self, x: np.ndarray, mask_data: np.ndarray = None) -> GaussianCopulaImputer:
        """Fits the imputer to data using Gaussian copula imputer.

        Args:
            x (np.ndarray): Data matrix.
            mask_data (np.ndarray): Masking missing values.

        Return:
            self
        """
        self._x = x

        if mask_data is not None:
            full_rows = ~mask_data.any(axis=1)
            X_full = self.data[full_rows]
        else:
            X_full = self.data

        # Transform each marginal to normal via ECDF
        V = np.zeros_like(X_full)
        self.ecdf_data = []
        for j in range(self.n_features):
            v_j = self.ecdf_transform(X_full[:, j])
            V[:, j] = v_j
            self.ecdf_data.append(X_full[:, j])  # for inverse transform

        self.mean = V.mean(axis=0)
        self.CovMatrix = np.cov(V, rowvar=False, bias=False)
        self.CovMatrix += np.eye(self.n_features) * 1e-6
        return self

    def transform(self, X: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        X_imp = X.copy()
        if mask is None:
            mask = np.isnan(X_imp)
        n_samples = X.shape[0]

        for i in range(n_samples):
            miss_idx = np.where(mask[i])[0]
            if miss_idx.size == 0:
                continue
            obs_idx = np.where(~mask[i])[0]

            # If there are no miss_idx transfrom back
            if obs_idx.size == 0:

                for j in miss_idx:
                    p = norm.cdf(self.mean[j])
                    X_imp[i, j] = self.inverse_ecdf(self.ecdf_data[j], p)
                continue

            mu_O = self.mean[obs_idx]
            mu_M = self.mean[miss_idx]
            Sigma_OO = self.CovMatrix[np.ix_(obs_idx, obs_idx)]
            Sigma_MO = self.CovMatrix[np.ix_(miss_idx, obs_idx)]


            # Looks if Matrix Sigma_00 has full rank(invertibal)
            if np.linalg.matrix_rank(Sigma_OO) < Sigma_OO.shape[0]:
                cond_mean = mu_M
            else:
                inv_OO = inv(Sigma_OO)
                x_O = self.ecdf_transform(X_imp[i, obs_idx])
                cond_mean = mu_M + Sigma_MO.dot(inv_OO.dot(x_O - mu_O))

            for idx, j in enumerate(miss_idx):
                p = norm.cdf(cond_mean[idx])
                X_imp[i, j] = self.inverse_ecdf(self.ecdf_data[j], p)

        return X_imp

    def __call__(self, coalitions: np.ndarray, verbose: bool = False) -> np.ndarray:
        if isinstance(coalitions, list):
            coalitions = np.array(coalitions)
        if coalitions.dtype != bool:
            coalitions = coalitions.astype(bool)

        n_coalitions = coalitions.shape[0]
        predictions = np.zeros(n_coalitions)

        for i, coalition in enumerate(coalitions):
            mask = (~coalition).reshape(1, -1)
            X_cond = self.transform(self._x, mask=mask)
            predictions[i] = self.model(X_cond)[0]

        return predictions


