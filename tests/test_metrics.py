import unittest

from lifelong_vla.metrics import summarize, task_forgetting


class MetricsTest(unittest.TestCase):
    def test_forgetting_uses_best_historical_score(self):
        matrix = [[0.8], [0.7, 0.75], [0.6, 0.70, 0.9]]
        actual = task_forgetting(matrix)
        for observed, expected in zip(actual, [0.2, 0.05, 0.0]):
            self.assertAlmostEqual(observed, expected)

    def test_empty_summary(self):
        summary = summarize([])
        self.assertEqual(summary["final_average_accuracy"], 0.0)
        self.assertEqual(summary["average_forgetting"], 0.0)


if __name__ == "__main__":
    unittest.main()
