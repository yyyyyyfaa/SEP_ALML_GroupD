import numpy as np
from shapiq import Explainer, InteractionValues

from shapiq_student.knn_shapley import KNNShapley


class KNNExplainer(Explainer):
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
            self.dataset = data
            self.labels = labels
            self.model_name = model_name
            self.N, self.M = data.shape
            self.knn_shapley = KNNShapley(model, data, labels, class_index)
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
            shapley_values = self.threshold.threshold_knn_shapley(x)
        elif gamma is not None:
            shapley_values = self.weighted_knn_shapley(x, gamma)
        else:
            print("Shapley")
            shapley_values = self.knn_shapley.knn_shapley(x)
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

        return interaction_values
