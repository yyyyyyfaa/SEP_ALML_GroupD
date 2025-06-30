import numpy as np
from networkx import neighbors
from shapiq import Explainer, InteractionValues
from shapiq_student.threshold import Threshold


class KNNExplainer(Explainer):
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


    def weighted_knn_shapley(self, x_query, gamma):
        # TODO Implement weighted
        pass
