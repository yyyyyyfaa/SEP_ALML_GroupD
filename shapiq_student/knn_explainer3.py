import numpy as np
from networkx import neighbors
from shapiq import Explainer, InteractionValues


class KNNExplainer3(Explainer):
    def __init__(self,
        model,
        data: np.ndarray,
        labels: np.ndarray,
        class_index = int | None,
        model_name : str = None,
        max_order: int = 1,
        index = "SV"):
            super().__init__(model, data, class_index, max_order=max_order, index = index)
            self.model = model
            self.dataset = data
            self.labels = labels
            self.class_index = class_index
            self.model_name = model_name
            self.N, self.M = data.shape
            self.max_order = max_order
            self.index = "SV"
            self.threshold = Threshold(model, data, labels, class_index)

            if hasattr(model, 'weights') and model.weights == 'distance':
                self.mode = 'weighted'
            elif hasattr(model, 'radius') and model.radius is not None:
                self.mode = 'threshold'
            else:
                self.mode = 'normal'


    def explain(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        radius = kwargs.get("radius")
        gamma = kwargs.get("gamma")
        if radius is not None and radius > 0:
            threshold = radius
            x_query = kwargs["x_query"]
            num_classes = kwargs["num_classes"]
            shapley_values = self.threshold.threshold_knn_shapley(x_query, threshold, num_classes)
            print("shapley_values:", shapley_values)
            print(threshold, radius)
            print(x_query, num_classes)
        elif gamma is not None:
            shapley_values = self.weighted_knn_shapley(x, gamma)
        else:
            shapley_values = self.knn_shapley(x)

        n_features = x.shape[1] if len(x.shape) > 1 else x.shape[0]
        print(n_features)
        interaction_values = InteractionValues(
            values=np.array(shapley_values),
            n_players=n_features,
            min_order=1,
            max_order=1,
            index="SV",
            baseline_value=0.0,
        )

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
