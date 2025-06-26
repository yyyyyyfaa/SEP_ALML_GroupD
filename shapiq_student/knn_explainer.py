import numpy as np
from shapiq import Explainer, InteractionValues

class KNNExplainer(Explainer):
    def __init__(self, model, dataset: np.ndarray, labels, method: str):
        super(KNNExplainer, self).__init__(model, dataset)
        self.dataset = dataset
        self.method = method
        self.labels = labels

    def explain_function(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        if self.method == "standard_shapley":
            interaction_values = self.knn_shapley(x)
        elif self.method == "threshold":
            threshold = kwargs["threshold"]
            x_query = kwargs["x_query"]
            num_classes = kwargs["num_classes"]
            interaction_values = self.threshold_knn_shapley(x_query, threshold, num_classes)
        elif self.method == "weighted":
            gamma = kwargs["gamma"]
            interaction_values = self.weighted_knn_shapley(x, gamma)
        else:
            print('Method must be one of "standard_shapley", "threshold", "weighted"')

        return interaction_values



    def knn_shapley(self, x_query):
        # TODO Implement knn
        pass


    def threshold_knn_shapley(self, x_query, threshold):
        # TODO Implement theshold
        pass


    def weighted_knn_shapley(self, x_query, gamma):
        # TODO Implement weighted
        pass