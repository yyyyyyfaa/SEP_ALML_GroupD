import pytest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from shapiq_student.knn_explainer import KNNExplainer


class TestKNNExplainerInit:
    """Test cases for KNNExplainer initialization"""

    def test_init_with_valid_classifier(self):
        """Test successful initialization with a fitted KNN classifier"""
        # Create a simple dataset
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(X, y)

        # Initialize explainer
        explainer = KNNExplainer(knn)

        # Assertions
        assert explainer.K == 2
        assert explainer.N == 3
        assert np.array_equal(explainer.X_train, X)
        assert np.array_equal(explainer.y_train, y)
        assert explainer.model is knn

    def test_init_with_unfitted_model(self):
        """Test that unfitted model raises ValueError"""
        knn = KNeighborsClassifier(n_neighbors=2)

        with pytest.raises(ValueError) as exc_info:
            KNNExplainer(knn)

        assert "call fit(X_train, y_train) first!" in str(exc_info.value)

    def test_init_with_invalid_x_train_dimension(self):
        """Test error when X_train is not 2D"""
        # Create a mock KNN object with 1D X_train




