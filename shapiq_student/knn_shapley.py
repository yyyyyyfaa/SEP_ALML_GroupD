"""Module for computing KNN Shapley values for data points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shapiq.utils import Model



class KNNShapley:
    """Class to compute KNN Shapley values for data points.

    This class estimates the contribution of each training data point to the prediction
    of a K-Nearest Neighbors model using Shapley values.

    """

    def __init__(self, model: Model, data: np.ndarray, labels: np.ndarray, class_index: np.ndarray) -> None:
        """Initialize the KNNShapley instance.

        Args:
            model (Model): The KNN model instance.
            data (np.ndarray): The training data.
            labels (np.ndarray): The labels for the training data.
            class_index (np.ndarray): y_Test data.
        """
        self.model = model
        self.dataset = data
        self.labels = labels
        self.class_index = class_index


    def knn_shapley_single(self, X_test: np.ndarray, y_test : int ) -> np.ndarray:
        """Compute KNN Shapley values for a single test point.

        Args:
            X_test (np.ndarray): The test data points.
            y_test (int): The test label.

        Returns:
            np.ndarray: The computed Shapley values for one training data point.
        """
        y_train = np.asarray(self.labels)
        X_train = np.asarray(self.dataset)
        N = X_train.shape[0]
        K = getattr(self.model, "n_neighbors", 10)

        # storage shapley values
        shap_values = np.zeros(N)
        # for j in range (n_test):
        # process each test point
        dists = np.linalg.norm(X_train - X_test, axis=1)
        idx_sorted = np.argsort(dists)
        s = np.zeros(N)
        # init the shapley value of furthest point
        last_idx = idx_sorted[-1]
        s[last_idx] = int(y_train[last_idx] == y_test) / N

        # add contribution from the second far to the nearest point
        for i in range(N - 2, -1, -1):
            cur_idx = idx_sorted[i]
            nxt_idx = idx_sorted[i + 1]
            # diff of contribution between test point i and i+1
            delta = (int(y_train[cur_idx] == y_test) -
                int(y_train[nxt_idx]== y_test)) / K
            prob = min(K, i + 1) / (i + 1)
            s[cur_idx] = s[nxt_idx] + delta * prob

        shap_values[:] = s
        return shap_values




    def knn_shapley(self, X_test: np.ndarray) -> np.ndarray:
        """Compute the average KNN Shapley values for one or more test points.

        Args:
            X_test (np.ndarray): Array with test data points.

        Returns:
            np.ndarray: The average Shapley values for each test point.
        """
        y_test = [self.class_index] * len(X_test)
        # Make sure it is a scalar
        y_test = np.asarray(y_test).flatten()
        X_train = np.asarray(self.dataset)
        N = X_train.shape[0]
        n_test = X_test.shape[0]
        sv = np.zeros(N)
        for i in range(n_test):
            sv += self.knn_shapley_single(X_test[i], y_test[i])
        return sv / n_test

