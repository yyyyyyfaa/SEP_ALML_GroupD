import numpy as np
from sklearn.metrics.pairwise import distance_metrics

from shapiq import Explainer, InteractionValues

class KNNExplainer2(Explainer):
    def __init__(self,
        model,
        data: np.ndarray,
        labels: np.ndarray,
        class_index : int | None = None,
        model_name : str = None,
        max_order: int = 1,
        index = "SV",
        random_state = 42):
            super().__init__(model, data, class_index, max_order=max_order, index = index)
            self.model = model
            self.dataset = data
            self.labels = labels
            self.class_index = class_index
            self.model_name = model_name
            self.N, self.M = data.shape
            self.random_state = np.random.RandomState(random_state)

            if hasattr(model, 'weights') and model.weights == 'distance':
                self.mode = 'weighted'
            elif hasattr(model, 'radius') and model.radius is not None:
                self.mode = 'threshold'
            else:
                self.mode = 'normal'


    def explain(self, x: np.ndarray | None = None, *args, **kwargs) -> InteractionValues:
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
            x_train_few = kwargs["x_train_few"]
            y_train_few = kwargs["y_train_few"]
            x_val_few = kwargs["x_val_few"]
            y_val_few = kwargs["y_val_few"]
            K = kwargs["K"]
            dis_metric = kwargs["dis_metric"]
            shapley_values = self.knn_shapley_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric=dis_metric)
        n_samples = self.dataset.shape[0]
        print(n_samples)
        interaction_values = InteractionValues(
            values=np.array(shapley_values),
            n_players=n_samples,
            min_order=1,
            max_order=1,
            index="SV",
            baseline_value=0.0,
        )
        print(shapley_values)
        return interaction_values

    def rank_neighbor(self, x_test, x_train, dis_metric):
        if dis_metric == 'cosine':
            distance = -np.dot(x_train, x_test) / np.linalg.norm(x_train, axis=1)
        else:
            distance = np.array([np.linalg.norm(x - x_test) for x in x_train])

        return np.argsort(distance)

    # x_test, y_test are single data point
    def knn_shapley_RJ_single(self, x_train_few, y_train_few, x_test, y_test, K, dis_metric):
        N = len(y_train_few)
        sv = np.zeros(N)
        rank = self.rank_neighbor(x_test, x_train_few, dis_metric=dis_metric)
        sv[int(rank[-1])] += int(y_test == y_train_few[int(rank[-1])]) / N

        for j in range(2, N + 1):
            i = N + 1 - j
            sv[int(rank[-j])] = sv[int(rank[-(j - 1)])] + ((int(y_test == y_train_few[int(rank[-j])]) - int(
                y_test == y_train_few[int(rank[-(j - 1)])])) / K) * min(K, i) / i

        return sv

    # Original KNN-Shapley proposed in http://www.vldb.org/pvldb/vol12/p1610-jia.pdf
    def knn_shapley_RJ(self, x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric):

        N = len(y_train_few)
        sv = np.zeros(N)
        print(y_val_few)

        n_test = len(y_val_few)
        for i in range(n_test):
            x_test, y_test = x_val_few[i], y_val_few[i]
            sv += self.knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric=dis_metric)
            print(dis_metric)

        return sv