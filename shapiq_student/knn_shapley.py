import numpy as np



class KNNShapley:

    def __init__(self, model, data, labels, class_index):
        self.model = model
        self.dataset = data
        self.labels = labels
        self.class_index = class_index


    def knn_shapley_single(self, X_test, y_test):
        y_train = np.asarray(self.labels)
        X_train = np.asarray(self.dataset)
        N = X_train.shape[0]
        n_test = X_test.shape[0]
        K = getattr(self.model, 'n_neighbors', 10)

        # storage shapley values
        shap_values = np.zeros(N)
        #for j in range (n_test):
            # process each test point
        dists = np.linalg.norm(X_train - X_test, axis=1)
        idx_sorted = np.argsort(dists)
        s = np.zeros(N)
        # init the shapley value of furthest point
        last_idx = idx_sorted[-1]
        s[last_idx] = int(y_train[last_idx] == y_test) / N

        # add contribution from the second far to the nearest point
        for i in range(N - 2, -1, -1):
            cur_idx = idx_sorted[i]
            nxt_idx = idx_sorted[i + 1]
            # diff of contribution between test point i and i+1
            delta = (int(y_train[cur_idx] == y_test) -
                int(y_train[nxt_idx]== y_test)) / K
            prob = min(K, i + 1) / (i + 1)
            s[cur_idx] = s[nxt_idx] + delta * prob

        shap_values[:] = s
        return shap_values




    def knn_shapley(self, X_test):
        X = np.atleast_2d(X_test)
        n = X.shape[0]
        sv = np.zeros(np.asarray(self.dataset).shape[0])
        # Use provided class_index for all test points
        for x in X:
            sv += self.knn_shapley_single(x, self.class_index)
        return sv / n


