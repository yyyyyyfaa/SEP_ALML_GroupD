import numpy as np
from shapiq import Explainer, InteractionValues
from scipy.special import comb

class KNNExplainer(Explainer):
    def __init__(self, model, dataset: np.ndarray, labels, method: str):
        super(KNNExplainer, self).__init__(model, dataset)
        self.dataset = dataset
        self.method = method
        self.labels = labels

    def explain_function(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        if self.method == "standard_shapley":
            X_test = kwargs['X_test']
            y_test = kwargs['y_test']
            X_train = kwargs['X_train']
            y_train = kwargs['y_train']
            K = kwargs["K"]
            interaction_values = self.knn_shapley(X_test, y_test, X_train, y_train, K)
        elif self.method == "threshold":
            threshold = kwargs["threshold"]
            x_query = kwargs["x_query"]
            num_classes = kwargs["num_classes"]
            interaction_values = self.threshold_knn_shapley(x_query, threshold, num_classes)
        elif self.method == "weighted":
            gamma = kwargs["gamma"]
            interaction_values = self.weighted_knn_shapley(x, gamma)
        else:
            print('Method must be one of "standard_shapley", "threshold", "weighted"')

        return interaction_values

    def knn_shapley(self, X_test, y_test, X_train, y_train, K):
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        n_test = X_test.shape[0]
        N = self.dataset.shape[0]
        # storage shapley values
        shap_values = np.zeros(N)
        # process each test point
        for x_t, y_t in zip(X_test, y_test):
            dists = np.linalg.norm(X_train - x_t, axis=1)
            # sort by dist from small to large
            idx_sorted = np.argsort(dists)
            s = np.zeros(len(X_train))
            # init the shapley value of furthest point
            last_idx = idx_sorted[-1]
            s[last_idx] = int(np.array_equal(y_train[last_idx], y_t)) / N

            # add contribution from the second far to the nearest point
            for i in range(N - 2, -1, -1):
                if i + 1 >= len(idx_sorted):
                    continue
                cur_idx = idx_sorted[i]
                nxt_idx = idx_sorted[i + 1]
                # diff of contribution between test point i and i+1
                delta = (int(np.array_equal(y_train[cur_idx], y_t)) -
                         int(np.array_equal(y_train[nxt_idx], y_t))) / K
                prob = min(K, i + 1) / (i + 1)
                s[cur_idx] = s[nxt_idx] + delta * prob

            shap_values += s
        shap_values /= n_test # divided by the amount of tests values
        return shap_values


    def threshold_knn_shapley(self, x_query, threshold):
        # TODO Implement theshold
        pass


    def weighted_knn_shapley(self, x_query, gamma):
        # TODO Implement weighted
        pass