import unittest

import numpy as np

from evaluate_model import (
    _numeric_vector,
    _validate_image_array,
    classification_metrics,
)


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
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertEqual(
            metrics["class_counts"], {"Normal": 2, "Pneumonia": 2}
        )
        self.assertEqual(
            metrics["metric_denominators"],
            {
                "accuracy": 4,
                "precision": 2,
                "sensitivity_recall": 2,
                "specificity": 2,
                "f1": 4,
                "roc_auc_positive_negative_pairs": 4,
            },
        )
        self.assertEqual(
            metrics["confusion_matrix"],
            {
                "label_order": ["Normal", "Pneumonia"],
                "rows": "actual",
                "columns": "predicted",
                "matrix": [[1, 1], [1, 1]],
            },
        )
        self.assertEqual(
            metrics["confusion_matrix_counts"],
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
        self.assertEqual(metrics["f1"], 0.0)
        self.assertIsNone(metrics["roc_auc"])

    def test_roc_auc_uses_raw_probabilities_and_handles_ties(self) -> None:
        metrics = classification_metrics(
            labels=np.array([0, 1, 0, 1]),
            probabilities=np.array([0.1, 0.5, 0.5, 0.9]),
            threshold=0.5,
        )

        self.assertEqual(metrics["roc_auc"], 0.875)

    def test_roc_auc_is_one_for_perfect_predictions(self) -> None:
        metrics = classification_metrics(
            labels=np.array([0, 0, 1, 1]),
            probabilities=np.array([0.1, 0.2, 0.8, 0.9]),
            threshold=0.5,
        )

        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_roc_auc_is_zero_for_reversed_predictions(self) -> None:
        metrics = classification_metrics(
            labels=np.array([0, 0, 1, 1]),
            probabilities=np.array([0.8, 0.9, 0.1, 0.2]),
            threshold=0.5,
        )

        self.assertEqual(metrics["roc_auc"], 0.0)

    def test_roc_auc_is_half_for_tied_positive_and_negative_scores(self) -> None:
        metrics = classification_metrics(
            labels=np.array([0, 1]),
            probabilities=np.array([0.5, 0.5]),
            threshold=0.5,
        )

        self.assertEqual(metrics["roc_auc"], 0.5)

    def test_rejects_invalid_metrics_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            classification_metrics(np.array([1]), np.array([]), 0.5)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            classification_metrics(np.array([0]), np.array([0.2]), 1.1)
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            classification_metrics(np.array([2]), np.array([0.2]), 0.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            classification_metrics(np.array([np.nan]), np.array([0.2]), 0.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            classification_metrics(np.array([0]), np.array([np.inf]), 0.5)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            classification_metrics(np.array([0]), np.array([-0.1]), 0.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            classification_metrics(np.array([0]), np.array([0.2]), np.nan)

    def test_rejects_complex_labels_and_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "complex"):
            _numeric_vector(np.array([1 + 2j]), "Labels")
        with self.assertRaisesRegex(ValueError, "complex"):
            _numeric_vector(np.array([0.5 + 1j]), "Probabilities")

    def test_accepts_uint8_raw_images_including_zero_and_one_values(self) -> None:
        images = np.zeros((1, 180, 180), dtype=np.uint8)
        images[0, 0, :2] = [0, 1]

        np.testing.assert_array_equal(_validate_image_array(images), images)

    def test_rejects_non_uint8_image_arrays(self) -> None:
        invalid_images = (
            np.array([[[0, 1]]], dtype=np.int64),
            np.array([[[0.0, 1.0]]], dtype=np.float32),
            np.array([[[0.0, 255.0]]], dtype=np.float32),
            np.array([[[0, 1]]], dtype=np.int8),
            np.ones((1, 180, 180), dtype=np.complex64),
            np.full((1, 180, 180), 128, dtype=object),
            np.ones((1, 180, 180), dtype=np.bool_),
        )

        for images in invalid_images:
            with self.subTest(dtype=images.dtype, shape=images.shape):
                with self.assertRaisesRegex(ValueError, "uint8"):
                    _validate_image_array(images)

    def test_rejects_empty_and_invalidly_shaped_image_arrays(self) -> None:
        invalid_images = (
            np.empty((0, 180, 180), dtype=np.uint8),
            np.zeros((180, 180), dtype=np.uint8),
            np.zeros((1, 179, 180), dtype=np.uint8),
        )

        for images in invalid_images:
            with self.subTest(shape=images.shape):
                with self.assertRaises(ValueError):
                    _validate_image_array(images)


if __name__ == "__main__":
    unittest.main()
