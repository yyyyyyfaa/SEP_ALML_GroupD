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
        # Implement weighted
        # if K = null ??
        for i in range(len(x_test)):
            x_val, y_val = x_test[i], y_test[i]
            #print(x_test[i]) # Debugging-Ausgabe
            #print(y_test[i]) # Debugging-Ausgabe
            #print(f"Berechnung des Shapley-Wertes für den Testpunkt {i + 1} von {len(x_test)}: {x_val}, Label: {y_val}") # Debugging-Ausgabe
        X = x_train
        Y = y_train
        N = len(X)  # Menge der Daten im Datensatz 
        #print(f"Anzahl der Datenpunkte im Datensatz: {N}") # Debugging-Ausgabe
        phi = np.zeros(N+1)

        # Berechnung der distanz
        distance = np.linalg.norm(X - x_val, axis=1)

        # Sortieren nach Distanz
        sorted_index = np.argsort(distance)  # Indizes für Sortierung
        sorted_distance = distance[sorted_index]  # sortierung nach Distanz
        X_sorted = X[sorted_index]  # sortierung nach X
        Y_sorted = Y[sorted_index]  # sortierung nach labels
        D = list(
            zip(X_sorted, Y_sorted, strict=False)
        )  # sortierter wieder zusammengefügter Datensatz

        # Berechnung der Gewichtung
        b = 3 #Seite 8: Baselines & Settings & Hyperparameters/Seite 6 Remark 3
        intervalls = 2**b  # Anzahl der Intervalle
        w_i = np.exp(-sorted_distance / gamma)  # RBF Kernel weight
        w_k = np.linspace(0, K, (intervalls) * K)
        w_i_discret = np.array([w_k[np.argmin(np.abs(w_k - w_i_discret))] for w_i_discret in w_i])
        w_j = (2 * (Y_sorted == y_val).astype(int) - 1) * w_i_discret
        #print(w_k)  # Debugging-Ausgabe
        #print(w_i)  # Debugging-Ausgabe
        #print(w_i_discret)  # Debugging-Ausgabe
        #print(w_j)  # Debugging-Ausgabe
        #print(len(w_j))  # Debugging-Ausgabe

        sorted_indices = np.argsort(w_j)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(w_j))
        #print(f"Ranks: {ranks}")  # Debugging-Ausgabe

        for i in range(1, N+ 1):
            # Initialisierung von F als Dictionary
            F_i = {}
            # F als 0 setzen
            for m in range(1, N + 1):
                for length in range(1, K):
                    for s in w_k:
                        F_i[(m, length, s)] = 0

            for m in range(1, N + 1):
                if m == i:
                    continue
                F_i[(m, 1, w_j[m - 1])] = 1
                #print(f"F_i[{m}, {1}, {w_j[m]}] = {F_i[(m, 1, w_j[m])]}")  # Debugging-Ausgabe
                #print(f"F_i[{m}, 1, {s}] = {F_i[(m, 1, s)]}")  # Debugging-Ausgabe

            for length in range(2, K):
                for m in range (length, N + 1):
                    for s in w_k:
                        w_m = w_j[m - 1]
                        F_i[(m, length, s)] = sum(F_i.get((t, length - 1, s - w_m), 0) for t in range(1, m))
                        #print(f"F_i[{m}, {length}, {s}] = {F_i[(m, length, s)]}")  # Debugging-Ausgabe
                        #print(f"Berechnung von F_i für m={m}, length={length}, s={s}")  # Debugging-Ausgabe
                        #print(f"F_i[t, length - 1, s - w_m] = {F_i.get(((t, length - 1, s - w_m), 0) for t in range(1, m))}")  # Debugging-Ausgabe

            # Berechnung von R_0
            R_im = {}

            upper = max(i + 1, K + 1)
            #print(f"upper = {upper}")  # Debugging-Ausgabe
            #print(f"N = {N}")  # Debugging-Ausgabe
            #print(F_i)  # Debugging-Ausgabe
            #Berechnung von R_im
            for m in range(upper, N + 1):
                R_im[i ,m] = 0
                #print(f"Berechnung von R_im für m={m}, i={i}")  # Debugging-Ausgabe
                for t in range(1, m - 1):
                    #print(f"Berechnung von R_im für m={m}, i={i}, t={t}")  # Debugging-Ausgabe
                    #print(f"Y_sorted[i] = {Y_sorted[i]}, y_val = {y_val}")  # Debugging-Ausgabe#
                    #print(f"w_i_discret[i] = {w_i_discret[i]}, w_j[m] = {w_j[m]}")  # Debugging-Ausgabe
                    #print(f"F_i[t, K - 1, s] = {F_i.get((t, K - 1, s), 0)}")  # Debugging-Ausgabe
                    if Y_sorted[i - 1] == y_val:
                        for s in range(- ranks[i - 1], - ranks[m - 1]):
                            R_im[i, m] += F_i.get((t, K - 1, w_k[s]), 0)
                            #print(-ranks[i], -ranks[m])  # Debugging-Ausgabe
                            #if F_i.get((t, K -1, w_k[s]), 0) != 0:
                                #print(f"F_i[t, K - 1, s] = {F_i.get((t, K - 1, w_k[s]), 0)}")  # Debugging-Ausgabe
                            #print(t, K - 1, s)  # Debugging-Ausgabe
                            #print(f"s = {s}")  # Debugging-Ausgabe
                            #if R_im[i, m] != 0:
                                #print(f"R_im[{i, m}] = {R_im[i, m]}")  # Debugging-Ausgabe
                    else:
                        for s in range(- ranks[m - 1], - ranks[i - 1]):
                            R_im[i, m] += F_i.get((t, K - 1, w_k[s]), 0)
                            #if F_i.get((t, K - 1, w_k[s]), 0) != 0:
                                #print(f"F_i[t, K - 1, s] = {F_i.get((t, K - 1, w_k[s]), 0)}")  # Debugging-Ausgabe
                            #print(t, K - 1, s)  # Debugging-Ausgabe
                            #print(f"R_im[{i, m}] = {R_im[i, m]}")  # Debugging-Ausgabe

            # Berechnung von G
            G_il = {}
            for count in range(1, len(w_i)):
                #print(f"w_i[{count}] = {w_i[count]}")  # Debugging-Ausgabe
                if w_i_discret[count] < 0:
                    G_il[count] = -1
                else:
                    for length in range(1, K):
                        G_il[i, length] = 0
                        for m in range(N + 1):
                            if m != i:
                                if Y_sorted[i - 1] == y_val:
                                    for s in range(- ranks[i- 1], 0):
                                        G_il[i, length] += F_i.get((m, length, w_k[s]), 0)
                                        #print(f"F_i.get((m, length, s), 0) {F_i.get((m, length, w_k[s]), 0)}")  # Debugging-Ausgabe
                                        #print(f"G_il[{i ,length}] = {G_il[i ,length]}")  # Debugging-Ausgabe
                                else:
                                    for s in range(- ranks[i- 1]):
                                        G_il[i, length] += F_i.get((m, length, w_k[s]), 0)
                                        #print(f"G_il[{i ,length}] = {G_il[i ,length]}")  # Debugging-Ausgabe

            # Berechnung des Shapleys von shapley Values
            sign = []
            sign = np.sign(w_i_discret[i - 1])

            first_term = 0
            for length in range(K):
                #print(f"G_il.get(i, {length}) = {G_il.get(i, length)}")  # Debugging-Ausgabe
                first_term += G_il.get(i, length) / comb(N-1, length)

            first_term = (1 / N) * first_term
            #print(f"Erster Term: {first_term}") # Debugging-Ausgabe

            second_term = 0
            for m in range(max(i + 1, K + 1), N + 1):
               second_term += R_im.get(i, m) / m * comb(m - 1, K)
               #print(f"R_im.get(i, {m}) = {R_im.get(i, m)}")  # Debugging-Ausgabe
            #print(second_term) # Debugging-Ausgabe

            phi[i] = sign * (first_term + second_term)

            #print(f"Shapley-Wert für den Testpunkt {i + 1} von {len(x_test)}: {phi[i]}") # Debugging-Ausgabe
        return phi
