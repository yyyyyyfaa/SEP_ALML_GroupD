import numpy as np
from numpy.linalg import inv
from shapiq.games.imputer.base import Imputer

class GaussianImputer(Imputer):

    def __init__(self, model, data: np.ndarray, x: np.ndarray | None = None):
        self.model = model
        self.data = data
        self._x = x
        self.n_players = data.shape[1]
        self.verbose = False

    def fit(self, x: np.ndarray, mask_data: np.ndarray = None):
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

    def transform(self, X: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
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
    def __call__(self, coalitions: np.ndarray, verbose: bool = False) -> np.ndarray:
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


class GaussianCopulaImputer(Imputer):

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