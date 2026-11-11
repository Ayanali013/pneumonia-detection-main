import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DATA_PATHS = {
    (PROJECT_ROOT / "x_ray_full_dataset.h5").resolve(),
    (PROJECT_ROOT / "report and presentation" / "x_ray_full_dataset.h5").resolve(),
}
DEFAULT_MODEL = PROJECT_ROOT / "pneumonia_detection_model.keras"


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)
    if labels.size == 0 or labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must be non-empty arrays of equal length.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    predictions = probabilities >= threshold
    positive = labels == 1
    negative = labels == 0
    true_positive = int(np.count_nonzero(predictions & positive))
    false_positive = int(np.count_nonzero(predictions & negative))
    true_negative = int(np.count_nonzero(~predictions & negative))
    false_negative = int(np.count_nonzero(~predictions & positive))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision = ratio(true_positive, true_positive + false_positive)
    sensitivity = ratio(true_positive, true_positive + false_negative)
    specificity = ratio(true_negative, true_negative + false_positive)
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision is not None
        and sensitivity is not None
        and precision + sensitivity
        else None
    )
    return {
        "sample_count": int(labels.size),
        "threshold": threshold,
        "accuracy": (true_positive + true_negative) / labels.size,
        "precision": precision,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the CNN on a separate HDF5 dataset with images and labels."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Independent held-out HDF5 file; do not use the training dataset.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
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
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    with h5py.File(args.dataset, "r") as dataset_file:
        if "images" not in dataset_file or "labels" not in dataset_file:
            raise ValueError("The HDF5 dataset must contain 'images' and 'labels' arrays.")
        images = dataset_file["images"][:]
        labels = dataset_file["labels"][:]

    if images.ndim != 3 or images.shape[1:] != (180, 180):
        raise ValueError(f"Expected images shaped (count, 180, 180); got {images.shape}.")
    if labels.shape != (images.shape[0],):
        raise ValueError(f"Expected one label per image; got labels shaped {labels.shape}.")

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
