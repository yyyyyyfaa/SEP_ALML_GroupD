"""This module provides the Weighted class for computing weighted k-nearest neighbor Shapley values.

using various methods for a given dataset and labels.
"""

from __future__ import annotations

from math import comb

import numpy as np
from shapiq import Explainer


class Weighted:
    """A class for computing weighted k-nearest neighbor Shapley values using various methods for a given dataset and labels.

    Attributes:
    ----------
    dataset : np.ndarray
        The dataset to be used for k-nearest neighbor calculations.
    labels : np.ndarray
        The labels corresponding to the dataset.
    method : str
        The method used for Shapley value computation.

    Methods:
    -------
    weighted_knn_shapley(x_val: list, y_val: int, gamma: int, K: int)
        Computes the weighted k-nearest neighbor Shapley values for a given input.
    """
    def __init__(self, dataset: np.ndarray, labels: np.ndarray, method: str = "standard_shapley") -> None:
        self.dataset = dataset
        self.method = method
        self.labels = labels

    def weighted_knn_shapley(self, x_val: list, y_val: int, gamma: int, K: int):

        mask = ~((self.dataset == x_val).all(axis=1) & (self.labels == y_val))
        X = self.dataset[mask]
        Y = self.labels[mask]

        N = len(X)  # Menge der Daten im Datensatz
        phi = np.zeros(N)

        # Berechnung der distanz
        distance = np.linalg.norm(X - x_val, axis=1)

        # Sortieren nach Distanz
        sorted_index = np.argsort(distance)  # Indizes für Sortierung
        sorted_distance = distance[sorted_index]  # sortierung nach Distanz
        Y_sorted = Y[sorted_index]  # sortierung nach labels

        # Berechnung der Gewichtung
        b = 3 #Seite 8: Baselines & Settings & Hyperparameters/Seite 6 Remark 3
        intervalls = 2**b  # Anzahl der Intervalle
        w_i = np.exp(-sorted_distance / gamma)  # RBF Kernel weight
        w_k = np.linspace(0, K, (intervalls) * K)
        w_i_discret = np.array([w_k[np.argmin(np.abs(w_k - w_i_discret))] for w_i_discret in w_i])
        w_j = (2 * (Y_sorted == y_val).astype(int) - 1) * w_i_discret

        sorted_indices = np.argsort(w_j)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(w_j))

        helper_array = np.zeros(len(ranks))

        for ind in range(len(ranks)):
            helper_array[ranks[ind]] = w_j[ind]

        count_zero = np.sum(w_j > 0)

        for i in range(1, N+ 1):
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

            # Berechnung von G
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

            # Berechnung des Shapleys von shapley Values
            sign = []
            sign = np.sign(w_j[i - 1])

            first_term = 0
            for length in range(K):

                first_term += G_il.get(i, length) / comb(N-1, length - 1)

            first_term = (1 / N) * first_term

            second_term = 0
            for m in range(max(i + 1, K + 1), N + 1):
               second_term += R_im.get(i, m) / (m * comb(m - 1, K))

            phi[i - 1] = sign * (first_term + second_term)

        return phi
