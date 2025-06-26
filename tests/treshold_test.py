import unittest
import numpy as np
from shapiq_student.knn_explainer import KNNExplainer


class TestTreshold(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        self.y_train = np.array([0, 1, 0])

        self.explainer = KNNExplainer(
            self,
            dataset=self.X_train,
            labels=self.y_train,
            method="threshold"
        )

    def test_against_manual_knn_threshold(self):
        x_query = (np.array([1.0, 1.0]), 1)
        threshold = 1.5
        num_classes = 2

        result = self.explainer.threshold_knn_shapley(x_query, threshold, num_classes)

        expected_result = np.array([-0.25, 0.333333, -0.25])

        self.assertEqual(result.shape, expected_result.shape)
        np.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_shapley_all_zero_if_outside_threshold(self):
        x_query = (np.array([1.0, 1.0]), 1)
        threshold = 0.2
        num_classes = 2

        result = self.explainer.threshold_knn_shapley(x_query, threshold, num_classes)
        self.assertTrue(np.all(result == 0))


if __name__ == "__main__":
    unittest.main()
