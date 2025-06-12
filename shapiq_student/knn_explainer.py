import numpy as np
from shapiq import Explainer, InteractionValues


class KNNExplainer(Explainer):
    def __init__(
        self, model, dataset: np.ndarray, labels: np.ndarray, method: str = "standard_shapley"
    ):  # labels hinzugefügt für WKNN
        super(KNNExplainer, self).__init__(model, dataset)
        self.dataset = dataset
        self.method = method
        self.labels = labels  # labels hinzugefügt für WKNN

    def explain_function(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        if self.method == "standard_shapley":
            interaction_values = self.knn_shapley(x)
        elif self.method == "threshold":
            threshold = kwargs["threshold"]
            interaction_values = self.threshold_knn_shapley(x, threshold)
        elif self.method == "weighted":
            gamma = kwargs["gamma"]
            interaction_values = self.weighted_knn_shapley(x, gamma)
        else:
            print('Method must be one of "standard_shapley", "threshold", "weighted"')

        return interaction_values

    def knn_shapley(self, x_query):
        # TODO Implement knn shapley
        pass

    def threshold_knn_shapley(self, x_query, threshold):
        # TODO Implement theshold
        pass

    def weighted_knn_shapley(self, x_query, gamma, K):
        # TODO Implement weighted
        # if K = null ??
        x_val, y_val = x_query
        X = self.dataset
        Y = self.labels
        N = len(X)  # Menge der Daten im Datensatz

        # Berechnung der distanz
        distance = np.linalg.norm(X - x_val, axis=1)

        # Sortieren nach Distanz
        sorted_index = np.argsort(distance)  # Indizes für Sortierung
        sorted_distance = distance[sorted_index]  # sortierung nach Distanz
        X_sorted = X[sorted_index]  # sortierung nach X
        Y_sorted = Y[sorted_index]  # sortierung nach labels
        D = list(
            zip(X_sorted, Y_sorted, strict=False)
        )  # sortierter wieder zusammengefügter Datensatz

        # Berechnung der Gewichtung
        w_i = np.exp(-sorted_distance / gamma)  # RBF Kernel weight
        w_j = (2 * (Y_sorted == y_val).astype(int) - 1) * w_i

        for i in range(1, N):
            w_k = w_i * K
            # Initialisierung von F als Dictionary
            F_i = {}
            # F als 0 setzen
            for m in range(1, N):
                for l in range(1, K - 1):
                    for s in w_k:
                        F_i[(m, l, s)] = 0

            for m in range(1, N):
                if m == i:
                    continue
                F_i = 1

            # Berechnung von F
            for l in range(2, K-1):
                for t in range(1, l):
                    F_0 = sum(F_i[(t, l - 1, s)])
                for m in range(l, N):
                    if m == i:
                        continue
                    for s in w_k:
                        w_m = w_j[m]
                        F_i[(m, l, s)] = F_0[(s - w_m)]

            # Berechnung von R_0
            R_0 = {}
            upper = max(i + 1, K + 1)
            for s in w_k:
                R_0[s] = sum(F_i[(t, K - 1, s)] for t in range(1, upper - 1) if t != i)
            #Berechnung von R_im
            for m in range(upper, N):
                if Y_sorted == y_val:
                    R_im = sum(R_0[s] for s in range(- w_i, - w_m))
                else:
                    R_im = sum(R_0[s] for s in range(- w_m, - w_i))
                R_0 = R_0 + F_i[(m, K - 1, s)]

            # TODO Berechnung von G

            # TODO Berechnung des Shapleys von z
