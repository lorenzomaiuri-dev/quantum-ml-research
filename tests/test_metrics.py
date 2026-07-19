import unittest

from quantum_framework.evaluation import compute_classification_metrics


class MetricTests(unittest.TestCase):
    def test_binary_auc_uses_positive_class_probability(self):
        result = compute_classification_metrics(
            labels=[0, 0, 1, 1],
            preds=[0, 0, 1, 1],
            probs=[[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
        )
        self.assertEqual(result["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
