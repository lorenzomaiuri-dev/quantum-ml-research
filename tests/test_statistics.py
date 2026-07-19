import unittest

from quantum_framework.evaluation import paired_comparison, summarize


class StatisticsTests(unittest.TestCase):
    def test_single_pair_is_descriptive_only(self):
        result = paired_comparison([0.8], [0.7])
        self.assertAlmostEqual(result["differences"]["mean"], 0.1)
        self.assertIsNone(result["paired_t_test"])
        self.assertIsNone(result["mean_difference_ci"])

    def test_multiple_pairs_include_inference(self):
        result = paired_comparison(
            [0.80, 0.72, 0.91],
            [0.70, 0.65, 0.82],
            bootstrap_samples=100,
        )
        self.assertEqual(result["direction"], "quantum_minus_classical")
        self.assertEqual(result["differences"]["n"], 3)
        self.assertIsNotNone(result["paired_t_test"])
        self.assertEqual(len(result["bootstrap_mean_difference_ci"]), 2)

    def test_summary_uses_sample_standard_deviation(self):
        result = summarize([1.0, 2.0, 3.0])
        self.assertEqual(result, {"n": 3, "mean": 2.0, "std": 1.0})

    def test_mismatched_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_comparison([0.8], [0.7, 0.6])


if __name__ == "__main__":
    unittest.main()
