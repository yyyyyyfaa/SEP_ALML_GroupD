from typing import Any

import numpy as np
from networkx import neighbors
from shapiq import Explainer, InteractionValues

#from shapiq_student.threshold import Threshold
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
            self.max_order = max_order
            self.index = index
            self.random_state = np.random.RandomState(random_state)

            if hasattr(model, 'weights') and model.weights == 'distance':
                self.mode = 'weighted'
            elif hasattr(model, 'radius') and model.radius is not None:
                self.mode = 'threshold'
                #self.threshold = Threshold(model, data, labels, class_index, model.radius)
            else:
                self.mode = 'normal'
                self.knn_shapley = KNNShapley(model, data, labels, class_index)

    def explain(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        gamma = kwargs.get("gamma")
        if self.mode == "threshold":
            shapley_values = self.threshold_knn_shapley(x)
        elif self.mode == 'weighted':
            shapley_values = self.weighted_knn_shapley(x, gamma)
        else:
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


    def threshold_knn_shapley(self, x: np.ndarray) -> np.ndarray:
        pass
    def weighted_knn_shapley(self, x: np.ndarray, gamma: float) -> np.ndarray:
        pass