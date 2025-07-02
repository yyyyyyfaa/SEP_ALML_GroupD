import numpy as np


class KNNShapley:

    def __init__(self, model, data, labels, class_index):
        self.model = model
        self.dataset = data
        self.labels = labels
        self.class_index = class_index

    # def knn_shapley(self, x: np.ndarray, K: int = 10) -> np.ndarray:
    #     x = np.asarray(x)
    #     N = self.dataset.shape[0]
    #     y_train = self.labels
    #     X_train = self.dataset
    #     class_index = self.class_index
    #
    #     shap_values = np.zeros(N)
    #
    #     # Wenn x ein 2D-Array ist, nimm nur erstes Beispiel
    #     if len(x.shape) == 2:
    #         x = x[0]
    #
    #     # Berechne Distanzen
    #     dists = np.linalg.norm(X_train - x, axis=1)
    #     idx_sorted = np.argsort(dists)
    #
    #     s = np.zeros(N)
    #     last_idx = idx_sorted[-1]
    #     s[last_idx] = int(y_train[last_idx] == class_index) / N
    #
    #     for i in range(N - 2, -1, -1):
    #         cur_idx = idx_sorted[i]
    #         nxt_idx = idx_sorted[i + 1]
    #
    #         delta = (int(y_train[cur_idx] == class_index) - int(y_train[nxt_idx] == class_index)) / K
    #         prob = min(K, i + 1) / (i + 1)
    #         s[cur_idx] = s[nxt_idx] + delta * prob
    #
    #     shap_values = s
    #     return shap_values

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