import numpy as np
from networkx import neighbors
from scipy.stats import triang
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
            interaction_values = self.knn_shapley(x)
        elif self.method == 'threshold':
            ### new version
            y_test = kwargs['y_test']
            threshold = kwargs['threshold']
            K0 = kwargs['K0']
            dis_metric = kwargs['dis_metric']
            interaction_values = self.threshold_knn_shapley(x, y_test, threshold, K0, dis_metric)

            ### old version
            # threshold = kwargs['threshold']
            # x_query = kwargs['x_query']
            # num_classes = kwargs['num_classes']
            # interaction_values = self.threshold_knn_shapley(x_query, threshold, num_classes)
        elif self.method == 'weighted':
            gamma = kwargs['gamma']
            interaction_values = self.weighted_knn_shapley(x,gamma)
        else:
            print('Method must be one of "standard_shapley", "threshold", "weighted"')

        return interaction_values


    def knn_shapley(self, x_query):
        # TODO Implement knn shapley
        pass

    ### new version
    def tnn_shapley_single(self, x_train, y_train, x_test, y_test, tau, K0, dis_metric):
        N = len(y_train)
        sv = np.zeros(N)

        C = max(y_train) + 1
        if dis_metric == 'cosine':
            distance = -np.dot(x_train, x_test) / np.linalg.norm(x_train, axis=1)
        else:
            distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
        Itau = (distance < tau).nonzero()[0]

        Ct = len(Itau)
        Ca = np.sum(y_train[Itau] == y_test)

        reusable_sum = 0
        stable_ratio = 1
        for j in range(N):
            stable_ratio *= (N - j - Ct) / (N - j)
            reusable_sum += (1 / (j + 1)) * (1 - stable_ratio)
            # reusable_sum += (1/(j+1)) * (1 - comb(N-1-j, Ct) / comb(N, Ct))

        for i in Itau:
            sv[i] = (int(y_test == y_train[i]) - 1 / C) / Ct
            if Ct >= 2:
                ca = Ca - int(y_test == y_train[i])
                sv[i] += (int(y_test == y_train[i]) / Ct - ca / (Ct * (Ct - 1))) * (reusable_sum - 1)

        return sv

    def threshold_knn_shapley(self, x_test, y_test, tau=0, K0=10, dis_metric='cosine'):
        train_samples = len(self.labels)
        shapley_values = np.zeros(train_samples)
        test_samples = len(y_test)

        for i in range(test_samples):
            x_t, y_t = x_test[i], y_test[i]
            shapley_values += self.tnn_shapley_single(self.dataset, self.labels, x_t, y_t, tau, K0, dis_metric)

        return shapley_values

    ### old version
    # def threshold_knn_shapley(self, x_query: tuple[float,float], threshold: float, num_classes: int) -> np.ndarray:
    #     """Berechnet die Threshold-KNN-Shapey Werte.
    #
    #     Args:
    #         x_query : Die Validierungspunkte (x_val, y_val).
    #         threshold (float): Der Radius der Nachbarschaft.
    #         num_classes (int): Die Anzahl der Klassen von der Klassifizierungsaufgabe.
    #
    #     Returns:
    #         np.ndarry: Ein Array mit den berechneten Spaley-Werten.
    #     """
    #     # TODO Implement threshold
    #     # Berechnet Treshold knn shapley werte fuer einen Validierungspunkt
    #
    #     x_val, y_val = x_query
    #     X, y = self.dataset, self.labels
    #
    #     N = self.dataset.shape[0] # Menge der Trainingspunkte
    #
    #     # Initialisierung aller Shapley werte mit 0
    #     phi = np.zeros(N)
    #
    #
    #     for i in range(N):
    #         x_i, y_i = X[i], y[i]
    #
    #
    #         #Trainingspunkte i aus den Daten entfernen
    #         X_minus_i = np.delete(X, i, axis=0)
    #         y_minus_i = np.delete(y, i ,axis=0)
    #
    #         distance = np.linalg.norm(x_i - x_val)
    #         if distance <= threshold:
    #             #Grösse des Reduzierten Trainingsdatensatz
    #             c = N - 1
    #
    #             distance_d_minus_zi = np.linalg.norm(X_minus_i - x_val, axis=1)
    #             neighbor_indices_minus_zi = np.where(distance_d_minus_zi <= threshold)[0]
    #
    #             #Anzahl der Nachbarn von x_val in D-zi innerhalb des treshholds
    #             c_x = 1 + np.abs(neighbor_indices_minus_zi)
    #
    #             # Anzahl der Nachbarn von x_val in D-zi, mit gleiches Label haben
    #             y_neighbors = self.labels[neighbor_indices_minus_zi]
    #             c_plus = np.sum(y_neighbors == y_val)
    #
    #             #A1
    #             A1 = (int(y_i == y_val) / c_x) - c_plus / (c_x * (c_x - 1))
    #
    #             #A2
    #             A2 = 0.0
    #             for k in range(c):
    #                 if c-k >= 0 and c_x >= 0:
    #                     bin1 = comb(c - k, c_x)
    #                     bin2 = comb(c + 1, c_x)
    #                     term = 1/(k+1) - 1/(k+1) * (bin1 / bin2)
    #                     A2 += term
    #             A2 -= 1.0
    #
    #             #Berechnung des Shapley_wertes für zi
    #             phi[i] = int(c_x >= 2) * A1 * A2 + (int(y_i == y_val) - ( 1 / num_classes)) / c_x
    #    return phi



    def weighted_knn_shapley(self, x_query, gamma):
        # TODO Implement weighted
        pass

