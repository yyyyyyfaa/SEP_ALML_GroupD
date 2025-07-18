import numpy as np
from numpy.linalg import inv
from shapiq.games.imputer.base import Imputer

class GaussianConditionalImputer(Imputer):
    def fit(self, X: np.ndarray, mask: np.ndarray = None):
        # Handle default mask
        if mask is None:
            mask = np.isnan(X)
        # 1. only use the samples without any missing values to estimate
        full_rows = ~mask.any(axis=1)
        X_full = X[full_rows]
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
        eps = 1e-6

        for i in range(n_samples):
            # find out the missing value
            missing_idx = np.where(mask[i])[0]
            if missing_idx.size == 0:
                continue
            observed_idx = np.where(~mask[i])[0]

            # 2.  submatrix accroding to "Multivariate Gaussian distribution"
            mu = self.mean
            Sigma = self.CovMatrix
            Sigma_CC = Sigma[np.ix_(observed_idx, observed_idx)]
            Sigma_MC = Sigma[np.ix_(missing_idx,   observed_idx)]
            Sigma_MM = Sigma[np.ix_(missing_idx,   missing_idx)]

            # 3. compute mean value
            x_C = X_imp[i, observed_idx]
            mu_C = mu[observed_idx]
            mu_M = mu[missing_idx]
            Sigma_CC_inv = inv(Sigma_CC + eps * np.eye(Sigma_CC.shape[0]))  # 确保 Σ_CC 可逆
            cond_mean = mu_M + Sigma_MC.dot(Sigma_CC_inv.dot(x_C - mu_C))

            # 4. compute covariance
            cond_cov = Sigma_MM - Sigma_MC.dot(Sigma_CC_inv.dot(Sigma_MC.T))

            # 5. use mean value to fill missing value
            sampled = np.random.multivariate_normal(cond_mean, cond_cov)
            X_imp[i, missing_idx] = sampled

        return X_imp






