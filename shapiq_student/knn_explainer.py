import numpy as np
from networkx import neighbors
from shapiq import Explainer, InteractionValues


class KNNExplainer(Explainer):
    def __init__(self, model, dataset: np.ndarray, labels, method: str):
        super(KNNExplainer, self).__init__(model, dataset)
        self.dataset = dataset
        self.method = method
        self.labels = labels

    def explain_function(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        if self.method == 'standard_shapley':
            intercation_values = self.knn_shapley(x)
        elif self.method == 'threshold':
            threshold = kwargs['threshold']
            intercation_values = self.threshold_knn_shapley(x, threshold)
        elif self.method == 'weighted':
            gamma = kwargs['gamma']
            interaction_values = self.weighted_knn_shapley(x,gamma)
        else:
            print('Method must be one of "standard_shapley", "threshold", "weighted"')

        return intercation_values


    def knn_shapley(self, x_query):
        # TODO Implement knn shapley
        pass


    def threshold_knn_shapley(self, x_query, threshold, num_classes):
        # TODO Implement threshold
        # Berechnet Treshold knn shapley werte fuer einen Validierungspunkt

        x_val, y_val = x_query
        X = self.dataset
        Y = self.labels

        N = self.dataset.shape[0] # Menge der Trainingspunkte

        distance = np.linalg.norm(x_query - self.dataset, axis=1) # Distanz berechnen
        print(distance) # Testen

        neighbors_mask = distance <= threshold # if true
        neighbor_indices = np.where(neighbors_mask)[0] # Indizes der Punkte innerhalb treshold / Labelvergleich

        c = len(neighbor_indices) # Menge der Nachbarn innerhalb des treshold

        if c == 0:
            return np.zeros(N)

        # Initialisierung aller Shapley werte mit 0
        phi = np.zeros(N)

        y_neighbors = self.dataset[neighbor_indices] # Wieviele Nachbarn das gleich label wie y_validierung haben
        c_plus = np.sum(y_neighbors == self.dataset)

        return phi
    pass


    def weighted_knn_shapley(self, x_query, gamma):
        # TODO Implement weighted
        pass