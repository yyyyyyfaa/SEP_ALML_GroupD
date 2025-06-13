import numpy as np
from shapiq import Explainer, InteractionValues
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class KNNExplainer(Explainer):
    def __init__(self, knn_model):
        self.model = knn_model
        # extract training data
        try:
            self.X_train = knn_model._fit_X
            self.y_train = knn_model._y
        except AttributeError:
            raise ValueError("call fit(X_train, y_train) first!")
        self.K = knn_model.n_neighbors
        self.N = self.X_train.shape[0]

        if self.X_train.ndim !=2:
            raise ValueError("X_train should be 2D array")
        if self.y_train.ndim != 1 or self.y_train.shape[0] != self.N:
            raise ValueError("length of y_train should be same with samples of X_train")

    def explain(self, X_test, y_test):
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        n_test = X_test.shape[0]
        # storage shapley values
        shap_values = np.zeros(self.N, dtype=float)
        # process each test point
        for x_t, y_t in zip(X_test, y_test):
            dists = np.linalg.norm(self.X_train - x_t, axis=1)
            # sort by dist from small to large
            idx_sorted = np.argsort(dists)
            s = np.zeros(self.N, dtype=float)
            # init the shapley value of furthest point
            last_idx = idx_sorted[-1]
            s[last_idx] = int(self.y_train[last_idx] == y_t) / self.N

            # add contribution from the second far to the nearest point
            for i in range(self.N -2 , -1, -1):
                cur_idx = idx_sorted[i]
                nxt_idx = idx_sorted[i+1]
                # diff of contribution between test point i and i+1
                delta = (int(self.y_train[cur_idx] == y_t) -
                         int(self.y_train[nxt_idx] == y_t)) / self.K
                prob = min(self.K, i + 1) / (i+1)
                s[cur_idx] = s[nxt_idx] + delta * prob

            shap_values[idx_sorted] += s
        shap_values /= n_test
        return shap_values




    def threshold_knn_shapley(self, x_query, threshold):
        # TODO Implement theshold
        pass


    def weighted_knn_shapley(self, x_query, gamma):
        # TODO Implement weighted
        pass