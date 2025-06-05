import numpy as np
from shapiq import Explainer, InteractionValues


class KNNExplainer(Explainer):
    def __init__(self, model, dataset: np.ndarray, labels: np.ndarray, method: str = 'standard_shapley'): #labels hinzugefügt für WKNN
        super(KNNExplainer, self).__init__(model, dataset)
        self.dataset = dataset
        self.method = method
        self.labels = labels #labels hinzugefügt für WKNN

    def explain_function(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        if self.method == 'standard_shapley':
            interaction_values = self.knn_shapley(x)
        elif self.method == 'threshold':
            threshold = kwargs['threshold']
            interaction_values = self.threshold_knn_shapley(x, threshold)
        elif self.method == 'weighted':
            gamma = kwargs['gamma']
            interaction_values = self.weighted_knn_shapley(x,gamma)
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
        N = len(X) #Menge der Daten im Datensatz

        #Berechnung der distanz
        distance = np.linalg.norm(X - x_val, axis = 1)

        #Sortieren nach Distanz
        sorted_index = np.argsort(distance) #Indizes für Sortierung
        sorted_distance = distance[sorted_index] #sortierung nach Distanz
        X_sorted = X[sorted_index] #sortierung nach X
        Y_sorted = Y[sorted_index] #sortierung nach labels
        D = list(zip(X_sorted, Y_sorted)) #sortierter wieder zusammengefügter Datensatz

        # Berechnung der Gewichtung
        w_i = np.exp(-sorted_distance / gamma) #RBF Kernel weight
        w_j = (2 * (Y_sorted == y_val).astype(int) - 1) * w_i


        for i in range (1, N):

            # TODO Initialisierung von F
            W_K = w_i * K
            for m in range(1, N):
                for l in range(1, K - 1):
                    for s in W_K:
                        F[(m, l, s)] = 0
            

            # TODO Berechnung von F

            # TODO Berechnung von R

            # TODO Berechnung von G

            # TODO Berechnung des Shapleys von z
            pass
    pass