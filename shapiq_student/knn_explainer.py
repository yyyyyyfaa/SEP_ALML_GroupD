"""KNNExplainer module for explaining KNN model predictions using Shapley values.

This module provides the KNNExplainer class, which supports normal, weighted, and threshold-based KNN explanations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapiq import Explainer, InteractionValues

from .knn_shapley import KNNShapley
from .threshold import Threshold
from .weighted import Weighted

if TYPE_CHECKING:
    from typing import Any

    from shapiq.explainer.custom_types import ExplainerIndices

    from shapiq.explainer.custom_types import ExplainerIndices

class KNNExplainer(Explainer):
    def __init__(self,
        model,
        data: np.ndarray,
        labels: np.ndarray,
        class_index : int | None = None,
        model_name : str | None = None,
        max_order: int = 1,
        index: ExplainerIndices = "SV",
        random_state: int = 42) -> None:
        """Initialize the KNNExplainer.

        Args:
            model: The KNN model to be explained.
            data (np.ndarray): The dataset used for explanation.
            labels (np.ndarray): The labels corresponding to the data.
            class_index (int | None): The class index for classification tasks.
            model_name (str): Optional name of the model.
            max_order (int): The maximum order of interactions to consider.
            index (str): The index type for explanation.
            random_state (int): Seed for random number generation.
        """
        super().__init__(model, data, class_index, max_order=max_order, index = index)
        self.dataset = data
        self.labels = labels
        self.max_order = max_order
        self.index = index
        self.model_name = model_name
        self.N, self.M = data.shape
        self.random_state = np.random.RandomState(random_state)

        if hasattr(model, "weights") and model.weights == "distance":
            self.mode = "weighted"
            self.weights = Weighted(model, data, labels, class_index, model.weights)
        elif hasattr(model, "radius") and model.radius is not None:
            self.mode = "threshold"
            self.threshold = Threshold(model, data, labels, class_index, model.radius)
        else:
            self.mode = "normal"
            self.knn= KNNShapley(model, data, labels, class_index)

    def explain(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:  # Pylance bei args, kwargs?
        """Explain the prediction for a given input sample using KNN-based Shapley values.

        Args
            x(np.ndarray): The input sample to explain.
            *args : Additional positional arguments.
            **kwargs : Additional keyword arguments. May include 'gamma' for weighted KNN.

        Returns:
            InteractionValues: The computed Shapley or interaction values for the input sample.
        """
        if self.mode == "threshold":
            shapley_values = self.threshold.threshold_knn_shapley(x)
        elif self.mode == "weighted":
            shapley_values = self.weights.weighted_knn_shapley(x)
        else:
            shapley_values = self.knn.knn_shapley(x)

        n_samples = self.dataset.shape[0]
        interaction_values = InteractionValues(
            values=np.array(shapley_values),
            n_players=n_samples,
            min_order=1,
            max_order=1,
            index="SV",
            baseline_value=0.0,
        )

        return interaction_values
