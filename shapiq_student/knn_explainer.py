import numpy as np
from networkx import neighbors
from shapiq import Explainer, InteractionValues
from scipy.special import comb


class KNNExplainer(Explainer):
    def __init__(self,
        model,
        data: np.ndarray,
        labels: np.ndarray,
        class_index = int | None,
        model_name : str = None,
        max_order: int = 1,
        index = "SV"):
            super().__init__(model, data, class_index, max_order=max_order, index = index)
            self.model = model
            self.dataset = data
            self.labels = labels
            self.class_index = class_index
            self.N, self.M = data.shape
            self.model_name = model_name
            self.max_order = max_order
            self.index = "SV"


    def explain(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        if self.model_name == "knn_basic":
            shapley_values = self.knn_shapley(x)
        elif self.model_name == "knn_radius":
            threshold = kwargs["threshold"]
            x_query = kwargs["x_query"]
            num_classes = kwargs["num_classes"]
            shapley_values = self.threshold_knn_shapley(x_query, threshold, num_classes)
        elif self.model_name == "knn_weighted":
            gamma = kwargs["gamma"]
            shapley_values = self.weighted_knn_shapley(x, gamma)
        else:
            raise ValueError("Unknown mode in KNNExplainer.")


        return np.array(shapley_values)

    def knn_shapley(self, x_query):
        # TODO Implement knn shapley
        pass

    def threshold_knn_shapley(
        self, x_query: tuple[float, float], threshold: float, num_classes: int
    ) -> np.ndarray:
        """Berechnet die Threshold-KNN-Shapey Werte für einen Validierungspunkt.

        Args:
            x_query : Die Validierungspunkte (x_val, y_val).
            threshold (float): Der Radius der Nachbarschaft.
            num_classes (int): Die Anzahl der Klassen von der Klassifizierungsaufgabe.

        Returns:
            np.ndarry: Ein Array mit den berechneten Spaley-Werten.
        """
        x_val, y_val = x_query
        X, y = self.dataset, self.labels
        N = self.dataset.shape[0]  # Menge der Trainingspunkte

        # Initialisierung
        phi = np.zeros(N)

        for i in range(N):
            x_i, y_i = X[i], y[i]

            # Trainingspunkte i aus den Daten entfernen
            X_minus_i = np.delete(X, i, axis=0)
            y_minus_i = np.delete(y, i, axis=0)

            distance = np.linalg.norm(x_i - x_val)
            if distance <= threshold:
                # Grösse des Reduzierten Trainingsdatensatz
                c = N - 1

                distance_d_minus_zi = np.linalg.norm(X_minus_i - x_val, axis=1)
                neighbor_indices_minus_zi = np.where(distance_d_minus_zi <= threshold)[0]

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

    def weighted_knn_shapley(self, x_query, gamma):
        # TODO Implement weighted
        pass
