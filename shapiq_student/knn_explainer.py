import numpy as np
from networkx import neighbors
from shapiq import Explainer, InteractionValues
from scipy.special import comb



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


    def threshold_knn_shapley(self, x_query: tuple[float,float], threshold: float, num_classes: int) -> np.ndarray:
        """Berechnet die Threshold-KNN-Shapey Werte.

        Args:
            x_query : Die Validierungspunkte (x_val, y_val).
            threshold (float): Der Radius der Nachbarschaft.
            num_classes (int): Die Anzahl der Klassen von der Klassifizierungsaufgabe.

        Returns:
            np.ndarry: Ein Array mit den berechneten Spaley-Werten.
        """
        # TODO Implement threshold
        # Berechnet Treshold knn shapley werte fuer einen Validierungspunkt

        x_val, y_val = x_query
        X, y = self.dataset, self.labels

        N = self.dataset.shape[0] # Menge der Trainingspunkte

        # Initialisierung aller Shapley werte mit 0
        phi = np.zeros(N)


        for i in range(N):
            x_i, y_i = X[i], y[i]


            #Trainingspunkte i aus den Daten entfernen
            X_minus_i = np.delete(X, i, axis=0)
            y_minus_i = np.delete(y, i ,axis=0)

            distance = np.linalg.norm(x_i - x_val)
            neighbors_mask = distance <= threshold # if true
            neighbor_indices = np.where(neighbors_mask)[0] # Indizes der Punkte innerhalb treshold / Labelvergleich

            if len(neighbor_indices) > 0:
                #Grösse des Reduzierten Trainingsdatensatz
                c = N - 1

                distance_d_minus_zi = np.linalg.norm(X_minus_i - x_val, axis=1)
                neighbor_indices_minus_zi = np.where(distance_d_minus_zi <= threshold)[0]

                #Anzahl der Nachbarn von x_val in D-zi innerhalb des treshholds
                c_x = 1 + np.abs(neighbor_indices_minus_zi)

                # Anzahl der Nachbarn von x_val in D-zi, mit gleiches Label haben
                y_neighbors = self.labels[neighbor_indices_minus_zi]
                c_plus = np.sum(y_neighbors == y_val)

                #A1
                A1 = (int(y_i == y_val) / c_x) - c_plus / (c_x * (c_x - 1))

                #A2
                A2 = 0.0
                for k in range(c):
                    if c-k >= 0 and c_x >= 0:
                        bin1 = comb(c - k, c_x)
                        bin2 = comb(c + 1, c_x)
                        term = 1/(k+1) - 1/(k+1) * (bin1 / bin2)
                        A2 += term
                A2 -= 1.0

                #Berechnung des Shapley_wertes für zi
                phi[i] = int(c_x >= 2) * A1 * A2 + (int(y_i == y_val) - ( 1 / num_classes)) / c_x
        return phi



    def weighted_knn_shapley(self, x_query, gamma):
        # TODO Implement weighted
        pass

