import pytest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from shapiq_student.knn_explainer import KNNExplainer


class TestKNNExplainerInit:
    """Test cases for KNNExplainer initialization"""

    def test_init_with_valid_classifier(self):
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

    def test_init_with_mismatched_lengths(self):
        """Test error when X_train and y_train have different lengths"""

        class MockKNN:
            def __init__(self):
                self._fit_X = np.array([[1, 2], [3, 4], [5, 6]])
                self._y = np.array([0, 1])  # Length 2 instead of 3
                self.n_neighbors = 2

        mock_knn = MockKNN()

        with pytest.raises(ValueError) as exc_info:
            KNNExplainer(mock_knn)

        assert "length of y_train should be same with samples of X_train" in str(exc_info.value)

    def test_init_with_invalid_x_train_dimension(self):
        """Test error when X_train is not 2D"""

        # Create a mock KNN object with 1D X_train
        class MockKNN:
            def __init__(self):
                self._fit_X = np.array([1, 2, 3])  # 1D array
                self._y = np.array([0, 1, 0])
                self.n_neighbors = 2

        mock_knn = MockKNN()

        with pytest.raises(ValueError) as exc_info:
            KNNExplainer(mock_knn)

        assert "X_train should be 2D array" in str(exc_info.value)

    class TestKNNExplainerExplain:
        """Test cases for the explain method"""

        def setup_method(self):
            """Setup test data for each test method"""
            # Create a simple dataset
            self.X_train = np.array([
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0]
            ])
            self.y_train = np.array([0, 0, 1, 1, 1])

            # Create and fit KNN
            self.knn = KNeighborsClassifier(n_neighbors=3)
            self.knn.fit(self.X_train, self.y_train)

            # Create explainer
            self.explainer = KNNExplainer(self.knn)

        def test_explain_single_test_point(self):
            """Test explain with single test point"""
            X_test = np.array([[1.5, 1.5]])
            y_test = np.array([0])

            shap_values = self.explainer.explain(X_test, y_test)

            # Check output shape
            assert shap_values.shape == (5,)  # Same as number of training points

            # Check that values are finite
            assert np.all(np.isfinite(shap_values))

            # Check sum property (should be between -1 and 1)
            assert -1 <= np.sum(shap_values) <= 1




