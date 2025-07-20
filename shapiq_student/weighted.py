"""This module provides the Weighted class for computing weighted k-nearest neighbor."""

from __future__ import annotations

from math import comb

import numpy as np


class Weighted:
    """A class for computing weighted k-nearest neighbor Shapley values using various methods.

    for a given dataset and labels.
    """

    def __init__(self, dataset: np.ndarray, labels: np.ndarray, method: str = "weighted") -> None:
        """Initialize the Weighted class with a dataset, labels, and computation method.

        Args:
            dataset (np.ndarray): The dataset to be used for k-nearest neighbor calculations.
            labels (np.ndarray): The labels corresponding to the dataset.
            method (str, optional): The method used for Shapley value computation (default is "weighted").
            class_index (np.ndarray): The index of the class to be used for Shapley value computation.
            gamma (int): The gamma parameter of the Shapley value computation.
        """
        self.dataset = dataset
        self.method = method
        self.labels = labels

    def prepare_data(self,  x_val: np.ndarray, y_val: any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepares the data by filtering out the input sample and sorting the remaining samples by distance.

        Args:
            x_val (np.ndarray): The input data point to exclude from the dataset.
            y_val (any): The label of the input data point to exclude.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Filtered and sorted dataset (X),
                - Corresponding labels (Y),
                - Sorted distances from x_val.

        Raises:
            ValueError: If the feature dimension of x_val does not match the dataset.
        """
        self.x = x_val
        mask = ~((self.dataset == x_val).all(axis=1) & (self.labels == y_val))
        X = self.dataset[mask]
        Y = self.labels[mask]
         # Calculating the distance
        distance = np.linalg.norm(X - x_val, axis=1)
        x_val = np.array(x_val)
        if x_val.shape[0] != self.dataset.shape[1]:
            msg = "Feature dimension mismatch between x_val and dataset."
            raise ValueError(msg)

        # Sorting for distance
        sorted_index = np.argsort(distance)
        return X[sorted_index], Y[sorted_index], distance[sorted_index]

    def compute_weights(
        self,
        sorted_distance: np.ndarray,
        Y_sorted: np.ndarray,
        y_val: int,
        gamma: int,
        K: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the weighted values and discretized weight intervals for the k-nearest neighbor Shapley value calculation.

        Args:
            sorted_distance (np.ndarray): Array of distances sorted in ascending order.
            Y_sorted (np.ndarray): Sorted labels corresponding to the sorted distances.
            y_val (int): Target label.
            gamma (int): The gamma parameter for the RBF kernel.
            K (int): The number of nearest neighbors to consider.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing the weighted values (w_j) and the discretized
            weight intervals (w_k).
        """
        b = 3 #Seite 8: Baselines & Settings & Hyperparameters/Seite 6 Remark 3
        intervalls = 2**b  # Anzahl der Intervalle
        w_i = np.exp(-sorted_distance / gamma)  # RBF Kernel weight
        w_k = np.linspace(0, K, (intervalls) * K)
        w_i_discret = np.array([w_k[np.argmin(np.abs(w_k - w_i_discret))] for w_i_discret in w_i])
        w_j = (2 * (Y_sorted == y_val).astype(int) - 1) * w_i_discret
        return w_j, w_k

    def compute_ranks(self, w_j: np.ndarray) -> np.ndarray:
        """Computes the ranks of the weighted values.

        Args:
        w_j (np.ndarray): Weighted values for each data point.

        Returns:
            np.ndarray: Array containing the ranks of the weighted values.
        """
        sorted_indices = np.argsort(w_j)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(w_j))
        return ranks

    def compute_f_i(self, N: int, K: int, w_k: np.ndarray, w_j: np.ndarray, i: int) -> dict:
        """Computes the F_i values used in the weighted k-nearest neighbor Shapley value calculation.

        Args:
            N (int): Total number of data points.
            K (int): Number of nearest neighbors.
            w_k (np.ndarray): Array of discretized weight intervals.
            w_j (np.ndarray): Weighted values for each data point.
            i (int): Index of the current data point.

        Returns:
            dict: Dictionary containing computed F_i values.
        """
        # Initialisierung von F als Dictionary
        F_i = {}
        # F als 0 setzen
        for m in range(1, N + 1):
            for length in range(1, K):
                for s in w_k:
                    F_i[(m, length, s)] = 0

        for m in range(1, N + 1):
            if m == i:
                continue
            F_i[(m, 1, w_j[m - 1])] = 1

        for length in range(2, K):
            for m in range (length, N + 1):
                for s in w_k:
                    w_m = w_j[m - 1]
                    F_i[(m, length, s)] = sum(F_i.get((t, length - 1, s - w_m), 0) for t in range(1, m))
        return F_i

    def compute_r_im(self, i: int, N: int, K: int, Y_sorted: np.ndarray, y_val: int, w_j: np.ndarray, helper_array: np.ndarray, F_i: dict) -> dict:
        """Computes the R_im values used in the weighted k-nearest neighbor Shapley value calculation.

        Args:
        i (int): Index of the current data point.
        N (int): Total number of data points.
        K (int): Number of nearest neighbors.
        Y_sorted (np.ndarray): Sorted labels.
        y_val (int): Target label.
        w_j (np.ndarray): Weighted values for each data point.
        helper_array (np.ndarray): Helper array for weight lookups.
        F_i (dict): Precomputed F_i values.

        Returns:
            dict: Dictionary containing computed R_im values.
        """
        # Berechnung von R_0
        R_im = {}

        upper = max(i + 1, K + 1)

        #Berechnung von R_im
        for m in range(upper, N + 1):
            R_im[i ,m] = 0
            for t in range(1, m - 1):
                if Y_sorted[i - 1] == y_val:
                    count_target_end = np.sum(w_j > - w_j[i - 1])
                    count_target_start = np.sum(w_j > - w_j[m - 1])

                    for s in range(- count_target_start, - count_target_end):
                        R_im[i, m] += F_i.get((t, K - 1, helper_array[s]), 0)

                else:
                    count_target_end = np.sum(w_j > - w_j[m - 1])
                    count_target_start = np.sum(w_j > - w_j[i - 1])

                    for s in range(- count_target_start, - count_target_end):
                        R_im[i, m] += F_i.get((t, K - 1, helper_array[s]), 0)
        return R_im

    def compute_g_il(self, i: int, N: int, K: int, Y_sorted: np.ndarray, y_val: int, w_j: np.ndarray, helper_array: np.ndarray, F_i: dict, count_zero: int) -> dict:
        """Computes the G_il values used in the weighted k-nearest neighbor Shapley value calculation.

        Args:
            i (int): Index of the current data point.
            N (int): Total number of data points.
            K (int): Number of nearest neighbors.
            Y_sorted (np.ndarray): Sorted labels.
            y_val (int): Target label.
            w_j (np.ndarray): Weighted values for each data point.
            helper_array (np.ndarray): Helper array for weight lookups.
            F_i (dict): Precomputed F_i values.
            count_zero (int): Count of positive weights.

        Returns:
            dict: Dictionary containing computed G_il values.
        """
        G_il = {}

        for count in range(1, len(w_j)):
            if w_j[count] < 0:
                G_il[count] = -1
            else:
                for length in range(1, K):
                    G_il[i, length] = 0
                    for m in range(N + 1):
                        if m != i:
                            if Y_sorted[i - 1] == y_val:
                                count_target = np.sum(w_j > - w_j[i - 1])
                                for s in range(- count_target, - count_zero):
                                    G_il[i, length] += F_i.get((m, length, helper_array[s]), 0)

                            else:
                                count_target = np.sum(w_j > - w_j[i - 1])
                                for s in range(- count_zero, - count_target):
                                    G_il[i, length] += F_i.get((m, length, helper_array[s]), 0)
        return G_il

    def weighted_knn_shapley(self, x_val: np.ndarray, y_val: int, gamma: int, K: int) -> np.ndarray:
        """Computes the weighted k-nearest neighbor Shapley values for a given input.

        Args:
            x_val (np.ndarray): The input data point for which to compute Shapley values.
            y_val (int): The label of the input data point.
            gamma (int): The gamma parameter for the RBF kernel.
            K (int): The number of nearest neighbors to consider.

        Returns:
            np.ndarray: The computed Shapley values for the input data point.
        """
        if K <= 0:
            msg = "K must be greater than 0."
            raise ValueError(msg)
        X, Y, sorted_distance = self.prepare_data(x_val, y_val)

        N = len(X)
        if N == 0:
            msg = "No samples remaining after filtering. Cannot compute Shapley values."
            raise ValueError(msg)

        if K > N:
            msg = "K cannot be greater than number of samples N."
            raise ValueError(msg)
        phi = np.zeros(N)

        w_j, w_k = self.compute_weights(sorted_distance, Y, y_val, gamma, K)
        ranks = self.compute_ranks(w_j)

        helper_array = np.zeros(len(ranks))

        for ind in range(len(ranks)):
            helper_array[ranks[ind]] = w_j[ind]

        count_zero = np.sum(w_j > 0)

        for i in range(1, N+ 1):
            F_i = self.compute_f_i(N, K, w_k, w_j, i)
            R_im = self.compute_r_im(i, N, K, Y, y_val, w_j, helper_array, F_i)
            G_il = self.compute_g_il(i, N, K, Y, y_val, w_j, helper_array, F_i, count_zero)

            sign = []
            sign = np.sign(w_j[i - 1])

            first_term = 0

            for length in range(K):

                first_term += G_il.get(i, length) / comb(N - 1, length)

            first_term = (1 / N) * first_term

            second_term = 0
            for m in range(max(i + 1, K + 1), N + 1):
               second_term += R_im.get(i, m) / (m * comb(m - 1, K))

            phi[i - 1] = sign * (first_term + second_term)

        return phi
