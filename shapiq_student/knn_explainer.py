import numpy as np
from shapiq import Explainer, InteractionValues


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
        sv = np.zeros(N) # Initialisierung

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

    def threshold_knn_shapley(self, x_test, y_test, tau=0, K0=0, dis_metric='cosine'):
        train_samples = len(self.labels)
        shapley_values = np.zeros(train_samples)
        test_samples = len(y_test)

        for i in range(test_samples):
            x_t, y_t = x_test[i], y_test[i]
            shapley_values += self.tnn_shapley_single(self.dataset, self.labels, x_t, y_t, tau, K0, dis_metric)

        return shapley_values

    def weighted_knn_shapley(self, x_query, gamma):
        # TODO Implement weighted
        pass

