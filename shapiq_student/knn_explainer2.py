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
        #print(f"Anzahl der Datenpunkte im Datensatz: {N}")
        phi = np.zeros(N)

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
        w_i = np.exp(-sorted_distance / gamma)  # RBF Kernel weight
        w_k = np.linspace(0, K, (2**b) * K)
        w_i_discret = np.array([w_k[np.argmin(np.abs(w_k - w_i_discret))] for w_i_discret in w_i])
        w_j = (2 * (Y_sorted == y_val).astype(int) - 1) * w_i_discret

        for i in range(1, N):
            # Initialisierung von F als Dictionary
            F_i = {}
            F_0 = {}
            # F als 0 setzen
            for m in range(1, N):
                for length in range(1, K - 1):
                    for s in w_k:
                        F_i[(m, length, s)] = 0
                        #print(f"F_i[{m}, {length}, {s}] = {F_i[(m, length, s)]}")  # Debugging-Ausgabe

            for m in range(1, N):
                if m == i:
                    continue
                for s in w_k:
                    F_i[(m, 1, s)] = 1
                    #print(f"F_i[{m}, 1, {s}] = {F_i[(m, 1, s)]}")  # Debugging-Ausgabe

            # Berechnung von F
            for length in range(2, K-1):
                for s in w_k:
                    F_0[length] = sum(F_i.get((t, length - 1, s), 0) for t in range(1, length))
                    #print(f"F_0[{length}] = {F_0[length]}")  # Debugging-Ausgabe
                for m in range(length, N):
                    if m == i:
                        continue
                    for s in w_k:
                        w_m = w_j[m]
                        F_i[(m, length, s)] = F_0.get((s - w_m), 0)
                        #print(f"length={length}, m={m}, s={s}, F_0 key={s - w_m}, F_0 val={F_0.get(s - w_m, 'not set')}") # Debugging-Ausgabe

            # Berechnung von R_0
            R_0 = {}
            R_im = {}
            upper = max(i + 1, K + 1)
            for s in w_k:
                R_0[s] = sum(F_i.get((t, K - 1, s), 0) for t in range(1, upper - 1) if t != i)
                #print(f"F_i[{i}, {K - 1}, {s}] = {F_i.get((i, K - 1, s), 0)}")  # Debugging-Ausgabe
                #print(i) # Debugging-Ausgabe
                #print(f"R_0[{s}] = {R_0[s]}")  # Debugging-Ausgabe

            #Berechnung von R_im
            for m in range(upper, N):
                if Y_sorted[i] == y_val:
                    R_im = sum(R_0[s] for s in range(- w_i, - w_m))
                else:
                    R_im = sum(R_0[s] for s in range(- w_m, - w_i))
                for s in w_k:
                    R_0 = R_0 + F_i.get((m, K - 1, s), 0)


            # Berechnung von G
            G_i0 = {}
            for count in range(1, len(w_i)):
                #print(f"w_i[{count}] = {w_i[count]}")  # Debugging-Ausgabe
                if w_i_discret[count] < 0:
                    G_i0[count] = -1
                else:
                    for length in range(1, K - 1):
                        G_il = {}
                        if Y_sorted[i] == y_val:
                                G_il[i] = sum(F_i.get((m, length, s), 0) for m in range(N) if m != i) * sum(F_i.get((m, length, s), 0) for s in range(len(-w_i), 0))
                                #print(f"G_il[{i}] = {G_il[i]}")  # Debugging-Ausgabe
                        else:
                                G_il[i] = sum(F_i.get((m, length, s), 0) for m in range(N) if m != i) * sum(F_i.get((m, length, s), 0) for s in range(0, len(-w_i)))
                                #print(f"G_il[{i}] = {G_il[i]}")  # Debugging-Ausgabe

            # Berechnung des Shapleys von shapley Values
            sign = []
            sign = np.sign(w_i_discret[i])

            first_term = 0
            for length in range(K):
                first_term += G_il[i] / comb(N-1, length)
            first_term = (1 / N) * first_term
            #print(f"Erster Term: {first_term}") # Debugging-Ausgabe

            second_term = 0
            for m in range(max(i + 1, K + 1), N + 1):
               second_term += R_im / m * comb(m - 1, K)
            #print(second_term) # Debugging-Ausgabe

            phi[i] = sign * (first_term + second_term)
            
            print(f"Shapley-Wert für den Testpunkt {i + 1} von {len(x_test)}: {phi[i]}") # Debugging-Ausgabe
            print(w_k)  # Debugging-Ausgabe
            print(w_i_discret)  # Debugging-Ausgabe
        return phi
