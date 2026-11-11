import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "x_ray_full_dataset.h5"
DEFAULT_MODEL = PROJECT_ROOT / "pneumonia_detection_model.keras"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue training the saved CNN on every image in an HDF5 dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Saved CNN not found: {args.model}")

    with h5py.File(args.dataset, "r") as dataset_file:
        if "images" not in dataset_file or "labels" not in dataset_file:
            raise ValueError("The HDF5 dataset must contain 'images' and 'labels' arrays.")
        images = dataset_file["images"][:]
        labels = dataset_file["labels"][:]

    if images.ndim != 3 or images.shape[1:] != (180, 180):
        raise ValueError(f"Expected images shaped (count, 180, 180); got {images.shape}.")
    if labels.shape != (images.shape[0],):
        raise ValueError(f"Expected one label per image; got labels shaped {labels.shape}.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must contain only 0 (Normal) and 1 (Pneumonia).")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("The training dataset must contain both classes.")

    tf.keras.utils.set_random_seed(args.seed)
    model = tf.keras.models.load_model(args.model)
    if model.input_shape[1:] != (180, 180, 1) or model.output_shape[-1] != 1:
        raise ValueError(
            f"Dataset is incompatible with the model: "
            f"{model.input_shape} -> {model.output_shape}."
        )
    if model.optimizer is None:
        raise ValueError("The saved model is not compiled with an optimizer.")

    starting_iteration = int(model.optimizer.iterations.numpy())
    class_counts = {str(label): int(np.count_nonzero(labels == label)) for label in (0, 1)}
    train_data = tf.data.Dataset.from_tensor_slices(
        (images, labels.reshape(-1, 1).astype(np.float32))
    )
    train_data = train_data.shuffle(
        len(labels), seed=args.seed, reshuffle_each_iteration=True
    )
    train_data = train_data.batch(args.batch_size)
    train_data = train_data.map(
        lambda batch_images, batch_labels: (
            tf.cast(batch_images[..., tf.newaxis], tf.float32) / 255.0,
            batch_labels,
        ),
        num_parallel_calls=2,
    )
    train_data = train_data.prefetch(1)

    print(
        "All dataset images will be used for training; "
        "no independent validation set is used.",
        flush=True,
    )
    history = model.fit(train_data, epochs=args.epochs, verbose=2)
    final_iteration = int(model.optimizer.iterations.numpy())
    temporary_model = args.model.with_name(f".{args.model.stem}.training.keras")
    if temporary_model.exists():
        raise FileExistsError(f"Temporary checkpoint already exists: {temporary_model}")

    try:
        model.save(temporary_model)
        verified_model = tf.keras.models.load_model(temporary_model)
        if int(verified_model.optimizer.iterations.numpy()) != final_iteration:
            raise ValueError("The saved checkpoint optimizer state did not verify.")
        os.replace(temporary_model, args.model)
    finally:
        if temporary_model.exists():
            temporary_model.unlink()

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "image_count": int(images.shape[0]),
        "class_counts": class_counts,
        "all_images_used_for_training": True,
        "validation_performed": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "starting_optimizer_iteration": starting_iteration,
        "final_optimizer_iteration": final_iteration,
        "final_training_loss": float(history.history["loss"][-1]),
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "model": str(args.model.resolve()),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
    }
    history_path = PROJECT_ROOT / "training_history.jsonl"
    with history_path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
