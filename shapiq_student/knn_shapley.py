import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNShapley:
    def __init__(self, model, data, labels, class_index):
        self.model = model
        self.dataset = data
        self.labels = labels
        self.class_index = class_index

    def knn_shapley(self, x):
        X_test = np.atleast_2d(x)
        y_train = np.asarray(self.labels)
        X_train = np.asarray(self.dataset)
        N = X_train.shape[0]
        n_test = X_test.shape[0]
        K = self.model.n_neighbors

        shap_values = np.zeros(N, dtype=float)

        for x_t in X_test:
            # 1) give order to every test point
            dists, idx_all = self.model.kneighbors(
                x_t.reshape(1, -1), n_neighbors=N, return_distance=True
            )
            idx_all = idx_all[0]

            # 2)init the furthest point's Shapley
            s = np.zeros(N, dtype=float)
            far_idx = idx_all[-1]
            s[far_idx] = int(y_train[far_idx] == self.class_index) / N

            # 3) # add contribution from the second far to the nearest point
            for rank in range(N - 2, -1, -1):
                cur_idx = idx_all[rank]
                nxt_idx = idx_all[rank + 1]
                same_cur = int(y_train[cur_idx] == self.class_index)
                same_nxt = int(y_train[nxt_idx] == self.class_index)
                delta = (same_cur - same_nxt) / K
                factor = min(K, rank + 1) / (rank + 1)
                s[cur_idx] = s[nxt_idx] + delta * factor

            shap_values += s

        if n_test > 1:
            shap_values /= n_test

        return shap_values

    '''
    shapley/n_test
    def knn_shapley(self, x):

        X_test = x
        X_train = np.asarray(self.dataset)
        N = X_train.shape[0]
        n_test = X_test.shape[0]
        sv = np.zeros(N)
        for i in range(n_test):
            sv += self.knn_shapley_singel(x[i])

        return sv/n_test
    '''