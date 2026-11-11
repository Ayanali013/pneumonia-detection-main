import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DATA_PATHS = {
    (PROJECT_ROOT / "x_ray_full_dataset.h5").resolve(),
    (PROJECT_ROOT / "report and presentation" / "x_ray_full_dataset.h5").resolve(),
}
DEFAULT_MODEL = PROJECT_ROOT / "pneumonia_detection_model.keras"


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    raw_labels = np.asarray(labels).reshape(-1)
    raw_probabilities = np.asarray(probabilities).reshape(-1)
    if raw_labels.shape != raw_probabilities.shape:
        raise ValueError("Labels and probabilities must be non-empty arrays of equal length.")
    labels = _numeric_vector(labels, "Labels")
    probabilities = _numeric_vector(probabilities, "Probabilities")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must be between 0 and 1.")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("Threshold must be a finite value between 0 and 1.") from error
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be a finite value between 0 and 1.")

    predictions = probabilities >= threshold
    positive = labels == 1
    negative = labels == 0
    true_positive = int(np.count_nonzero(predictions & positive))
    false_positive = int(np.count_nonzero(predictions & negative))
    true_negative = int(np.count_nonzero(~predictions & negative))
    false_negative = int(np.count_nonzero(~predictions & positive))

    positive_count = int(np.count_nonzero(positive))
    negative_count = int(np.count_nonzero(negative))
    predicted_positive_count = true_positive + false_positive
    accuracy_denominator = int(labels.size)
    precision_denominator = predicted_positive_count
    sensitivity_denominator = positive_count
    specificity_denominator = negative_count
    f1_denominator = 2 * true_positive + false_positive + false_negative
    roc_auc_denominator = positive_count * negative_count

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "sample_count": accuracy_denominator,
        "class_counts": {
            "Normal": negative_count,
            "Pneumonia": positive_count,
        },
        "threshold": float(threshold),
        "accuracy": ratio(true_positive + true_negative, accuracy_denominator),
        "precision": ratio(true_positive, precision_denominator),
        "sensitivity_recall": ratio(true_positive, sensitivity_denominator),
        "specificity": ratio(true_negative, specificity_denominator),
        "f1": ratio(2 * true_positive, f1_denominator),
        "roc_auc": _roc_auc(labels, probabilities, roc_auc_denominator),
        "metric_denominators": {
            "accuracy": accuracy_denominator,
            "precision": precision_denominator,
            "sensitivity_recall": sensitivity_denominator,
            "specificity": specificity_denominator,
            "f1": f1_denominator,
            "roc_auc_positive_negative_pairs": roc_auc_denominator,
        },
        "confusion_matrix": {
            "label_order": ["Normal", "Pneumonia"],
            "rows": "actual",
            "columns": "predicted",
            "matrix": [
                [true_negative, false_positive],
                [false_negative, true_positive],
            ],
        },
        "confusion_matrix_counts": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
    }


def _numeric_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must not contain complex values.")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must contain numeric values.")
    array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(
            "Labels and probabilities must be non-empty arrays of equal length."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array.astype(np.float64, copy=False)


def _validate_image_array(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images)
    if array.dtype != np.dtype(np.uint8):
        raise ValueError("Images must use uint8 raw pixel values in the range [0, 255].")
    if array.ndim != 3 or array.shape[1:] != (180, 180):
        raise ValueError(f"Expected images shaped (count, 180, 180); got {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError("The evaluation dataset must contain at least one image.")
    return array


def _roc_auc(
    labels: np.ndarray, probabilities: np.ndarray, positive_negative_pairs: int
) -> float | None:
    if positive_negative_pairs == 0:
        return None

    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    ranks = np.arange(1, labels.size + 1, dtype=np.float64)
    _, group_ids = np.unique(sorted_probabilities, return_inverse=True)
    group_counts = np.bincount(group_ids)
    average_ranks = (
        np.bincount(group_ids, weights=ranks) / group_counts
    )[group_ids]
    positive_rank_sum = float(average_ranks[sorted_labels == 1].sum())
    positive_count = int(np.count_nonzero(labels == 1))
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / positive_negative_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the CNN on a separate HDF5 dataset with images and labels."
        ),
        epilog=(
            "Examples:\n"
            "  python evaluate_model.py --dataset .\\external_eval.h5\n"
            "  python evaluate_model.py --dataset .\\external_eval.h5 "
            "--model .\\pneumonia_detection_model.keras --threshold 0.5\n\n"
            "The default threshold is 0.5. A supplied threshold is applied as-is; "
            "select thresholds using validation data only, never the test set. "
            "The evaluator cannot verify dataset or patient independence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "HDF5 file with uint8 raw grayscale images shaped (N, 180, 180) "
            "and binary labels shaped (N,). Convert external images with a "
            "separately documented preprocessing adapter; do not normalize "
            "them, because the evaluator divides by 255 exactly once. "
            "Use data independent of training."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Saved CNN model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed probability threshold in [0, 1] (default: 0.5); never tune on test data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {args.dataset}")
    if args.dataset.resolve() in TRAINING_DATA_PATHS:
        raise ValueError(
            "The supplied file is part of the training dataset. "
            "Provide an independent held-out dataset instead."
        )
    if not args.model.is_file():
        raise FileNotFoundError(f"Saved CNN not found: {args.model}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not np.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be a finite value between 0 and 1.")

    import h5py
    import tensorflow as tf

    with h5py.File(args.dataset, "r") as dataset_file:
        if "images" not in dataset_file or "labels" not in dataset_file:
            raise ValueError("The HDF5 dataset must contain 'images' and 'labels' arrays.")
        images = dataset_file["images"][:]
        labels = dataset_file["labels"][:]

    images = _validate_image_array(images)
    if labels.shape != (images.shape[0],):
        raise ValueError(f"Expected one label per image; got labels shaped {labels.shape}.")
    labels = _numeric_vector(labels, "Labels")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must contain only 0 and 1.")

    model = tf.keras.models.load_model(args.model)
    if model.input_shape[1:] != (180, 180, 1) or model.output_shape[-1] != 1:
        raise ValueError(
            f"Dataset is incompatible with the model: "
            f"{model.input_shape} -> {model.output_shape}."
        )
    evaluation_data = tf.data.Dataset.from_tensor_slices(images).batch(args.batch_size)
    evaluation_data = evaluation_data.map(
        lambda batch_images: tf.cast(
            batch_images[..., tf.newaxis], tf.float32
        ) / 255.0,
        num_parallel_calls=2,
    )
    probabilities = model.predict(evaluation_data, verbose=0).reshape(-1)
    metrics = classification_metrics(labels, probabilities, args.threshold)
    metrics["dataset"] = str(args.dataset.resolve())
    metrics["independence_note"] = (
        "Metrics are meaningful as a generalization estimate only if this file "
        "contains samples independent of training, ideally separated by patient."
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
