import numpy as np
from shapiq import Explainer, InteractionValues


class KNNExplainer(Explainer):
    def __init__(self, model, dataset: np.ndarray, method: str = 'standard_shapley'):
        super(KNNExplainer, self).__init__(model, dataset)
        self.dataset = dataset
        self.method = method

    def explain_function(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        if self.method == 'standard_shapley':
            intercation_values = self.knn_shapley(x)
        elif self.method == 'threshold':
            threshold = kwargs['threshold']
            intercation_values = self.threshold_knn_shapley(x, threshold)
        elif self.method == 'weighted':
            gamma = kwargs['gamma']
            interaction_values = self.weighted_knn_shapley(x,gamma)
        else:
            print('Method must be one of "standard_shapley", "threshold", "weighted"')

        return intercation_values


    def knn_shapley(self, x_query):
        # TODO Implement knn shapley
        pass


    def threshold_knn_shapley(self, x_query, threshold):
        # TODO Implement theshold
        pass


    def weighted_knn_shapley(self, x_query, gamma):
        # TODO Implement weighted
        pass