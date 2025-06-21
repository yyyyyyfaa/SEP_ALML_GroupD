import numpy as np
from shapiq import Explainer, InteractionValues


def factorial(n):
    if n in {0, 1} or n < 0:
        return 1
    return n * factorial(n - 1)

def comb(n, k):
    if k < 0 or k > n:
        return 1
    return factorial(n) // (factorial(k) * factorial(n - k))

class KNNExplainer(Explainer):
    def __init__(
        self, model, dataset: np.ndarray, labels: np.ndarray, method: str = "standard_shapley"
        ):# labels hinzugefügt für WKNN
        super(KNNExplainer, self).__init__(model, dataset)
        self.dataset = dataset
        self.method = method
        self.labels = labels  # labels hinzugefügt für WKNN

    def knn_shapley(self, x_query):
        # TODO Implement knn shapley
        pass

    def threshold_knn_shapley(self, x_query, threshold):
        # TODO Implement theshold
        pass
    def weighted_knn_shapley(self, x_train, y_train, x_test, y_test, gamma, K):
        x_val, y_val = x_test, y_test
        X = x_train
        Y = y_train
        N = len(X)
        phi = np.zeros(N)  # Proper zero initialization

        distance = np.linalg.norm(X - x_val, axis=1)
        sorted_index = np.argsort(distance)
        sorted_distance = distance[sorted_index]
        X_sorted = X[sorted_index]
        Y_sorted = Y[sorted_index]
        w_i = np.exp(-sorted_distance / gamma)
        w_j = (2 * (Y_sorted == y_val).astype(int) - 1) * w_i

        b = 3
        w_k = np.linspace(0, K, (2**b) * K)

        for i in range(N):
            # Initialize F_i
            F_i = {}
            for m in range(N):
                for length in range(1, K):
                    for s in w_k:
                        F_i[(m, length, s)] = 0

            for m in range(N):
                if m == i:
                    continue
                for s in w_k:
                    F_i[(m, 1, s)] = 1

            # Compute F
            for length in range(2, K):
                for m in range(length, N):
                    if m == i:
                        continue
                    for s in w_k:
                        w_m = w_j[m]
                        F_i[(m, length, s)] = sum(F_i.get((t, length - 1, s - w_m), 0) for t in range(1, length))

            # Compute G_il for all l
            G_il = {}
            for l in range(1, K):
                G_il[l] = 0
                for m in range(N):
                    if m == i:
                        continue
                    if Y_sorted[m] == y_val:
                        for s in w_k:
                            if -w_j[m] <= s <= 0:
                                G_il[l] += F_i.get((m, l, s), 0)
                    else:
                        for s in w_k:
                            if 0 <= s <= -w_j[m]:
                                G_il[l] += F_i.get((m, l, s), 0)

            # Compute R_im for all m
            R_im = {}
            upper = max(i + 1, K + 1)
            R_0 = {}
            for s in w_k:
                R_0[s] = sum(F_i.get((t, K - 1, s), 0) for t in range(1, upper) if t != i)
            for m in range(upper, N):
                if Y_sorted[m] == y_val:
                    R_im[m] = sum(R_0.get(s, 0) for s in w_k if -w_j[m] <= s <= -w_j[i])
                else:
                    R_im[m] = sum(R_0.get(s, 0) for s in w_k if -w_j[m] <= s <= -w_j[i])
                for s in w_k:
                    R_0[s] += F_i.get((m, K - 1, s), 0)

            # Compute Shapley value for i
            sign = np.sign(w_i[i])
            first_term = 0
            for l in range(1, K):
                first_term += G_il.get(l, 0) / (N - l)
            first_term = (1 / N) * first_term

            second_term = 0
            for m in range(upper, N):
                denom = m * (m - K) if (m - K) != 0 else 1
                second_term += R_im.get(m, 0) / denom

            phi[i] = sign * (first_term + second_term)
        return phi

