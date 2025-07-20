"""This module provides the Threshold class for calculating threshold-based KNN Shapley values."""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shapiq.utils import Model


class Threshold:
    """A class for calculating threshold-based KNN Shapley values."""

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        labels: np.ndarray,
        class_index: np.ndarray,
        threshold: float,
    ) -> None:
        """Initialize the Threshold class with model, data, labels, class index, and threshold.

        Args:
            model (Model): The machine learning model used for predictions.
            data (np.ndarray): The training dataset.
            labels (np.ndarray): The labels corresponding to the training data.
            class_index (np.ndarray): The index of the class for which Shapley values are calculated.
            threshold (float): The distance threshold for considering neighbors.
        """
        self.model = model
        self.dataset = data
        self.labels = labels
        self.class_index = class_index
        self.threshold = threshold

    def threshold_knn_shapley_single(self, x: np.ndarray, y_test: int) -> np.ndarray:
        """Calculates the Shaply Value for one test point.

        Args:
            x (np.ndarray): Radius from Neighborhood.
            y_test (int): Class index.

        Returns:
            np.ndarry: Array of shapley values for one test point.
        """
        x_val = x
        y_val = y_test
        X = self.dataset
        y = self.labels
        N = self.dataset.shape[0]  # Amount of training points
        num_classes = len(np.unique(y))

        # Initialising
        phi = np.zeros(N)

        for i in range(N):
            x_i, y_i = X[i], y[i]

            # Remove training points i from the data
            x_minus_i = np.delete(X, i, axis=0)
            y_minus_i = np.delete(y, i, axis=0)

            distance = np.linalg.norm(x_i - x_val)

            if distance <= self.threshold:
                # Amount of the reduced training data set
                c = N - 1

                distance_d_minus_zi = np.linalg.norm(x_minus_i - x_val, axis=1)
                neighbor_indices_minus_zi = np.where(distance_d_minus_zi <= self.threshold)[0]

                # Amount of neighbors from x_val in D-zi within the threshold
                c_x = len(neighbor_indices_minus_zi) + 1

                term2 = (int(y_i == y_val) - (1 / num_classes)) / c_x
                k = 2

                if c_x >= k:
                    # Amount of neighbors from x_val in D-zi with the same labels
                    y_neighbors = y_minus_i[neighbor_indices_minus_zi]
                    c_plus = np.sum(y_neighbors == y_val)

                    # Calculating A1
                    A1 = (int(y_i == y_val) / c_x) - c_plus / (c_x * (c_x - 1))

                    # Calculating A2
                    A2 = 0.0
                    for k in range(c):
                        if c - k >= 0 and c_x >= 0:
                            bin1 = comb(c - k, c_x)
                            bin2 = comb(c + 1, c_x)
                            term1 = 1 / (k + 1) - 1 / (k + 1) * (bin1 / bin2)
                            A2 += term1
                    A2 -= 1.0
                    # Calculating shapley value for z_i
                    phi[i] = int(c_x >= k) * A1 * A2 + term2

                else:
                    phi[i] = term2

        return phi

    def threshold_knn_shapley(self, x: np.ndarray) -> np.ndarray:
        """Calculates the threshold-KNN Shapley values for multiple validation points.

        Args:
            x (np.ndarray): Array of validation points for which to calculate Shapley values.

        Returns:
            np.ndarray: Array containing the aggregated Shapley values for the training data.
        """
        y_test = [self.class_index] * len(x)
        # Make sure it is a scalar
        y_test = np.asarray(y_test).flatten()
        X_train = np.asarray(self.dataset)
        N = X_train.shape[0]
        n_test = x.shape[0]
        phi = np.zeros(N)
        for i in range(n_test):
            phi += self.threshold_knn_shapley_single(x[i], y_test[i])
        return phi
