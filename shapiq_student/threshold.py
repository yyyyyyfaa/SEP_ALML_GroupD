from math import comb

import numpy as np

class Threshold:
    def __init__(self, model, data, labels, class_index, threshold):
        self.model = model
        self.dataset = data
        self.labels = labels
        self.class_index = class_index
        self.threshold = threshold

    def threshold_knn_shapley_single(
            self, x: tuple[float, float], y_test
    ) -> np.ndarray:
        """Berechnet die Threshold-KNN-Shapey Werte für einen Validierungspunkt.

        Args:
            x_query : Die Validierungspunkte (x_val, y_val).
            threshold (float): Der Radius der Nachbarschaft.
            num_classes (int): Die Anzahl der Klassen von der Klassifizierungsaufgabe.

        Returns:
            np.ndarry: Ein Array mit den berechneten Spaley-Werten.
        """
        x_val = x
        y_val = y_test
        X = self.dataset
        y = self.labels
        N = self.dataset.shape[0]  # Menge der Trainingspunkte
        num_classes = len(np.unique(y))
        print(num_classes)

        # Initialisierung
        phi = np.zeros(N)

        for i in range(N):
            x_i, y_i = X[i], y[i]


            # Trainingspunkte i aus den Daten entfernen
            X_minus_i = np.delete(X, i, axis=0)
            y_minus_i = np.delete(y, i, axis=0)

            distance = np.linalg.norm(x_i - x_val)

            if distance <= self.threshold:
                # Grösse des Reduzierten Trainingsdatensatz
                c = N - 1

                distance_d_minus_zi = np.linalg.norm(X_minus_i - x_val, axis=1)
                neighbor_indices_minus_zi = np.where(distance_d_minus_zi <= self.threshold)[0]

                # Anzahl der Nachbarn von x_val in D-zi innerhalb des treshholds
                c_x = len(neighbor_indices_minus_zi) + 1

                term2 = (int(y_i == y_val) - (1 / num_classes)) / c_x

                if c_x >= 2:
                    # Anzahl der Nachbarn von x_val in D-zi, mit gleiches Label haben
                    y_neighbors = y_minus_i[neighbor_indices_minus_zi]
                    c_plus = np.sum(y_neighbors == y_val)

                    # A1
                    A1 = (int(y_i == y_val) / c_x) - c_plus / (c_x * (c_x - 1))

                    # A2
                    A2 = 0.0
                    for k in range(c):
                        if c - k >= 0 and c_x >= 0:
                            bin1 = comb(c - k, c_x)
                            bin2 = comb(c + 1, c_x)
                            term1 = 1 / (k + 1) - 1 / (k + 1) * (bin1 / bin2)
                            A2 += term1
                    A2 -= 1.0
                    # Berechnung des Shapley_wertes für zi
                    phi[i] = int(c_x >= 2) * A1 * A2 + term2

                else:
                    phi[i] = term2

        return phi

    def threshold_knn_shapley(self, x: tuple[float, float]) -> np.ndarray:
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