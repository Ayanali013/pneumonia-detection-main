import unittest

import numpy as np

from evaluate_model import classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_calculates_binary_metrics_and_confusion_matrix(self) -> None:
        metrics = classification_metrics(
            labels=np.array([0, 0, 1, 1]),
            probabilities=np.array([0.1, 0.8, 0.7, 0.2]),
            threshold=0.5,
        )

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["sensitivity_recall"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(
            metrics["confusion_matrix"],
            {
                "true_positive": 1,
                "false_positive": 1,
                "true_negative": 1,
                "false_negative": 1,
            },
        )

    def test_marks_undefined_class_metrics_as_none(self) -> None:
        metrics = classification_metrics(
            labels=np.array([1, 1]),
            probabilities=np.array([0.1, 0.2]),
            threshold=0.5,
        )

        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["specificity"])
        self.assertIsNone(metrics["f1"])

    def test_rejects_invalid_metrics_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            classification_metrics(np.array([1]), np.array([]), 0.5)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            classification_metrics(np.array([0]), np.array([0.2]), 1.1)


if __name__ == "__main__":
    unittest.main()
