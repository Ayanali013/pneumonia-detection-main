import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from evaluate_model import classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "x_ray_full_dataset.h5"
EXISTING_MODEL = PROJECT_ROOT / "pneumonia_detection_model.keras"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
REQUIRED_PACKAGES = ("tensorflow", "keras", "numpy", "h5py")
PYTHON_VERSION = (3, 11)
IMAGE_SHAPE = (180, 180)
MODEL_INPUT_SHAPE = (None, 180, 180, 1)
MODEL_OUTPUT_SHAPE = (None, 1)
EXPECTED_PARAMETER_COUNT = 6_646_529
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
THRESHOLD = 0.5
LOSS = "binary_crossentropy"
LEARNING_RATE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh CNN with a duplicate-aware image-level split, then "
            "evaluate its selected checkpoint once on the held-out test set."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify dependencies and fresh-model save/load compatibility; do not train.",
    )
    return parser.parse_args()


def _normalized_package_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def read_required_versions(requirements_path: Path = REQUIREMENTS_FILE) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if "==" not in line:
            continue
        name, version = line.split("==", maxsplit=1)
        pins[_normalized_package_name(name.strip())] = version.strip()

    missing_pins = [
        _normalized_package_name(name)
        for name in REQUIRED_PACKAGES
        if _normalized_package_name(name) not in pins
    ]
    if missing_pins:
        raise RuntimeError(
            f"Missing exact dependency pins in {requirements_path}: "
            f"{', '.join(missing_pins)}."
        )
    return pins


def validate_environment_versions(
    python_version: tuple[int, int],
    required_versions: dict[str, str],
    installed_versions: dict[str, str],
) -> None:
    if python_version != PYTHON_VERSION:
        raise RuntimeError(
            f"Python 3.11 is required for this experiment; found "
            f"{python_version[0]}.{python_version[1]}."
        )

    mismatches = []
    for package in REQUIRED_PACKAGES:
        name = _normalized_package_name(package)
        expected = required_versions.get(name)
        installed = installed_versions.get(name)
        if installed != expected:
            mismatches.append(
                f"{package}=={expected} required, found {installed or 'not installed'}"
            )
    if mismatches:
        raise RuntimeError(
            "Incompatible ML environment; install the versions pinned in "
            f"{REQUIREMENTS_FILE.name}: " + "; ".join(mismatches)
        )


def verify_environment() -> dict[str, str]:
    if sys.version_info[:2] != PYTHON_VERSION:
        raise RuntimeError(
            f"Python 3.11 is required for this experiment; found "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )

    required = read_required_versions()
    installed: dict[str, str] = {}
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            installed[_normalized_package_name(package)] = (
                importlib.metadata.version(package)
            )
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            "Required packages are not installed: " + ", ".join(missing)
        )

    validate_environment_versions(sys.version_info[:2], required, installed)
    return {
        package: installed[_normalized_package_name(package)]
        for package in REQUIRED_PACKAGES
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_artifacts_unchanged(
    dataset_path: Path,
    model_path: Path,
    expected_dataset_sha256: str,
    expected_model_sha256: str,
) -> None:
    if sha256_file(dataset_path) != expected_dataset_sha256:
        raise RuntimeError(f"The source dataset changed during the experiment: {dataset_path}")
    if sha256_file(model_path) != expected_model_sha256:
        raise RuntimeError(f"The original checkpoint changed during the experiment: {model_path}")


def validate_dataset_arrays(images: np.ndarray, labels: np.ndarray) -> None:
    if images.dtype != np.uint8:
        raise ValueError(f"Expected raw uint8 images; got {images.dtype}.")
    if images.ndim != 3 or images.shape[1:] != IMAGE_SHAPE:
        raise ValueError(
            f"Expected images shaped (count, 180, 180); got {images.shape}."
        )
    if images.shape[0] == 0:
        raise ValueError("The dataset must contain at least one image.")
    if labels.shape != (images.shape[0],):
        raise ValueError(
            f"Expected one label per image; got labels shaped {labels.shape}."
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must contain only 0 (Normal) and 1 (Pneumonia).")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("The dataset must contain both classes.")


def image_sha256_hashes(images: np.ndarray) -> list[str]:
    return [
        hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        for image in images
    ]


def _largest_remainder_counts(total: int) -> np.ndarray:
    exact = np.asarray(SPLIT_RATIOS, dtype=np.float64) * total
    counts = np.floor(exact).astype(np.int64)
    remaining = total - int(counts.sum())
    order = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def create_group_stratified_split(
    labels: np.ndarray, image_hashes: list[str], seed: int = 42
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Labels must be a non-empty one-dimensional array.")
    if labels.size != len(image_hashes):
        raise ValueError("There must be one image hash for every label.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must contain only 0 and 1.")

    groups: dict[str, list[int]] = {}
    group_labels: dict[str, int] = {}
    for index, image_hash in enumerate(image_hashes):
        if not image_hash:
            raise ValueError(f"Image hash is empty at row {index}.")
        label = int(labels[index])
        if image_hash in group_labels and group_labels[image_hash] != label:
            raise ValueError(
                f"Identical image hash has conflicting labels: {image_hash}."
            )
        group_labels[image_hash] = label
        groups.setdefault(image_hash, []).append(index)

    rng = np.random.default_rng(seed)
    indices_by_split: dict[str, list[int]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }
    split_index = {name: index for index, name in enumerate(SPLIT_NAMES)}

    for label in (0, 1):
        class_groups = [
            (image_hash, indices)
            for image_hash, indices in groups.items()
            if group_labels[image_hash] == label
        ]
        targets = _largest_remainder_counts(
            int(np.count_nonzero(labels == label))
        )
        random_tie_breaks = {
            image_hash: float(rng.random()) for image_hash, _ in class_groups
        }
        class_groups.sort(
            key=lambda item: (
                -len(item[1]),
                random_tie_breaks[item[0]],
                item[0],
            )
        )

        current_counts = np.zeros(len(SPLIT_NAMES), dtype=np.int64)
        for image_hash, indices in class_groups:
            group_size = len(indices)
            candidate_costs = []
            for candidate in range(len(SPLIT_NAMES)):
                projected = current_counts.copy()
                projected[candidate] += group_size
                normalized_error = (
                    (projected - targets) / np.maximum(targets, 1)
                )
                candidate_costs.append(float(np.square(normalized_error).sum()))

            best_cost = min(candidate_costs)
            best_splits = [
                index
                for index, cost in enumerate(candidate_costs)
                if np.isclose(cost, best_cost, rtol=0.0, atol=1e-12)
            ]
            chosen = int(rng.choice(best_splits))
            current_counts[chosen] += group_size
            indices_by_split[SPLIT_NAMES[chosen]].extend(indices)

    result = {
        name: np.asarray(
            rng.permutation(indices_by_split[name]), dtype=np.int64
        )
        for name in SPLIT_NAMES
    }
    validate_split_indices(labels, image_hashes, result)
    return result


def validate_split_indices(
    labels: np.ndarray,
    image_hashes: list[str],
    splits: dict[str, np.ndarray],
) -> dict[str, dict[str, int]]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size != len(image_hashes):
        raise ValueError("Labels and image hashes must be matching one-dimensional arrays.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"Splits must contain exactly: {', '.join(SPLIT_NAMES)}.")

    all_indices = np.concatenate(
        [np.asarray(splits[name], dtype=np.int64) for name in SPLIT_NAMES]
    )
    expected_indices = np.arange(len(labels), dtype=np.int64)
    if all_indices.size != expected_indices.size or not np.array_equal(
        np.sort(all_indices), expected_indices
    ):
        raise ValueError("Split indices must be disjoint and cover every dataset row.")

    hash_owners: dict[str, str] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        indices = np.asarray(splits[split_name], dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"The {split_name} split must not be empty.")
        counts = {
            str(label): int(np.count_nonzero(np.asarray(labels)[indices] == label))
            for label in (0, 1)
        }
        class_counts[split_name] = counts
        for index in indices:
            image_hash = image_hashes[int(index)]
            owner = hash_owners.setdefault(image_hash, split_name)
            if owner != split_name:
                raise ValueError(
                    f"Duplicate image hash crosses splits: {image_hash}."
                )
    return class_counts


def build_fresh_model(seed: int = 42) -> Any:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(180, 180, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=["accuracy"],
    )
    verify_model_contract(model, tf)
    return model


def verify_model_contract(model: Any, tf: Any) -> dict[str, object]:
    if tuple(model.input_shape) != MODEL_INPUT_SHAPE:
        raise RuntimeError(f"Unexpected fresh-model input shape: {model.input_shape}.")
    if tuple(model.output_shape) != MODEL_OUTPUT_SHAPE:
        raise RuntimeError(f"Unexpected fresh-model output shape: {model.output_shape}.")
    parameter_count = int(model.count_params())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_COUNT} parameters; found {parameter_count}."
        )
    if model.loss != LOSS or model.optimizer.__class__.__name__ != "Adam":
        raise RuntimeError("Fresh model has an unexpected compile configuration.")
    get_compile_config = getattr(model, "get_compile_config", None)
    if not callable(get_compile_config):
        raise RuntimeError("Keras does not expose the compile configuration.")
    compile_config = get_compile_config()
    optimizer_config = compile_config.get("optimizer")
    optimizer_parameters = (
        optimizer_config.get("config")
        if isinstance(optimizer_config, dict)
        else None
    )
    if (
        compile_config.get("loss") != LOSS
        or compile_config.get("metrics") != ["accuracy"]
        or not isinstance(optimizer_config, dict)
        or optimizer_config.get("class_name") != "Adam"
        or not isinstance(optimizer_parameters, dict)
    ):
        raise RuntimeError("Fresh model has an unexpected serialized compile configuration.")
    configured_learning_rate = float(optimizer_parameters["learning_rate"])
    learning_rate = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    if not np.isclose(
        configured_learning_rate, LEARNING_RATE, rtol=0.0, atol=1e-10
    ) or not np.isclose(learning_rate, LEARNING_RATE, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"Unexpected Adam learning rate: {learning_rate}.")
    return {
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "parameter_count": parameter_count,
        "compile": {
            "loss": LOSS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "metrics": ["accuracy"],
        },
    }


def verify_model_save_load(tf: Any, seed: int) -> dict[str, object]:
    model = build_fresh_model(seed)
    contract = verify_model_contract(model, tf)
    with tempfile.TemporaryDirectory(prefix="pneumocatcher-model-preflight-") as temp_dir:
        path = Path(temp_dir) / "fresh_model.keras"
        model.save(path)
        reloaded = tf.keras.models.load_model(path)
        reloaded_contract = verify_model_contract(reloaded, tf)
    if reloaded_contract != contract:
        raise RuntimeError("Fresh model contract changed after save/reload.")
    return contract


def create_experiment_directory(output_root: Path, seed: int) -> Path:
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_seed{seed}_"
        + uuid4().hex[:8]
    )
    experiment_dir = output_root / f"fresh_cnn_{run_id}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def assert_new_checkpoint_path(
    checkpoint_path: Path, protected_paths: tuple[Path, ...]
) -> None:
    resolved_checkpoint = checkpoint_path.resolve()
    protected = {path.resolve() for path in protected_paths}
    if resolved_checkpoint in protected:
        raise ValueError(
            f"Experiment checkpoint path is protected and cannot be overwritten: "
            f"{resolved_checkpoint}"
        )


def fit_training_data(
    model: Any,
    training_data: Any,
    validation_data: Any,
    epochs: int,
    callbacks: list[Any],
) -> Any:
    return model.fit(
        x=training_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )


def _make_dataset(
    tf: Any,
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    dataset = tf.data.Dataset.from_tensor_slices(
        (images[indices], labels[indices].reshape(-1, 1).astype(np.float32))
    )
    if shuffle:
        dataset = dataset.shuffle(
            int(indices.size), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda batch_images, batch_labels: (
            tf.cast(batch_images[..., tf.newaxis], tf.float32) / 255.0,
            batch_labels,
        ),
        num_parallel_calls=2,
    )
    return dataset.prefetch(1)


def _make_test_dataset(
    tf: Any, images: np.ndarray, indices: np.ndarray, batch_size: int
) -> Any:
    dataset = tf.data.Dataset.from_tensor_slices(images[indices]).batch(batch_size)
    dataset = dataset.map(
        lambda batch_images: (
            tf.cast(batch_images[..., tf.newaxis], tf.float32) / 255.0
        ),
        num_parallel_calls=2,
    )
    return dataset.prefetch(1)


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
    os.replace(temporary_path, path)


def _class_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    split_labels = labels[indices]
    return {
        "Normal": int(np.count_nonzero(split_labels == 0)),
        "Pneumonia": int(np.count_nonzero(split_labels == 1)),
    }


def run_experiment(args: argparse.Namespace) -> Path:
    environment = verify_environment()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if args.dataset.resolve() != DEFAULT_DATASET.resolve():
        raise ValueError(
            f"This workflow is limited to the project dataset: {DEFAULT_DATASET}."
        )
    if not EXISTING_MODEL.is_file():
        raise FileNotFoundError(f"Original checkpoint not found: {EXISTING_MODEL}")
    if args.dataset.resolve() == EXISTING_MODEL.resolve():
        raise ValueError("Dataset path cannot be the original model checkpoint.")

    dataset_path = args.dataset.resolve()
    original_model_path = EXISTING_MODEL.resolve()
    dataset_sha256 = sha256_file(dataset_path)
    original_model_sha256 = sha256_file(original_model_path)

    import h5py
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    model_preflight = verify_model_save_load(tf, args.seed)
    with h5py.File(dataset_path, "r") as dataset_file:
        if "images" not in dataset_file or "labels" not in dataset_file:
            raise ValueError("The HDF5 dataset must contain 'images' and 'labels'.")
        images = dataset_file["images"][:]
        labels = dataset_file["labels"][:]

    validate_dataset_arrays(images, labels)
    image_hashes = image_sha256_hashes(images)
    splits = create_group_stratified_split(labels, image_hashes, args.seed)
    split_class_counts = validate_split_indices(labels, image_hashes, splits)
    split_sizes = {name: int(splits[name].size) for name in SPLIT_NAMES}

    experiment_dir = create_experiment_directory(args.output_root, args.seed)
    new_model_path = experiment_dir / "best_model.keras"
    assert_new_checkpoint_path(
        new_model_path, (dataset_path, original_model_path)
    )

    duplicate_groups: dict[str, list[int]] = {}
    for index, image_hash in enumerate(image_hashes):
        duplicate_groups.setdefault(image_hash, []).append(index)
    duplicate_groups = {
        image_hash: indices
        for image_hash, indices in duplicate_groups.items()
        if len(indices) > 1
    }

    split_manifest = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "image_shape": list(images.shape),
        "image_dtype": str(images.dtype),
        "labels_shape": list(labels.shape),
        "label_mapping": {"0": "Normal", "1": "Pneumonia"},
        "split_unit": "image-level, exact-duplicate-group-aware; not patient-level",
        "split_ratios_target": dict(zip(SPLIT_NAMES, SPLIT_RATIOS)),
        "seed": args.seed,
        "split_sizes": split_sizes,
        "class_counts": split_class_counts,
        "indices_by_split": {
            name: splits[name].tolist() for name in SPLIT_NAMES
        },
        "image_sha256_by_row": image_hashes,
        "duplicate_groups": duplicate_groups,
    }
    experiment_config = {
        "seed": args.seed,
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.patience,
        "threshold": THRESHOLD,
        "model_architecture": [
            "Conv2D(32, 3x3, relu)",
            "MaxPooling2D(2x2)",
            "Conv2D(64, 3x3, relu)",
            "MaxPooling2D(2x2)",
            "Conv2D(128, 3x3, relu)",
            "MaxPooling2D(2x2)",
            "Flatten",
            "Dense(128, relu)",
            "Dropout(0.5)",
            "Dense(1, sigmoid)",
        ],
        "model_input_shape": list(MODEL_INPUT_SHAPE),
        "model_output_shape": list(MODEL_OUTPUT_SHAPE),
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "preflight_model": model_preflight,
        "compile": {
            "loss": LOSS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "metrics": ["accuracy"],
        },
        "environment": {
            "python": sys.version.split()[0],
            **environment,
        },
        "original_checkpoint_path": str(original_model_path),
        "original_checkpoint_sha256": original_model_sha256,
    }
    _write_json(experiment_dir / "split_manifest.json", split_manifest)
    _write_json(experiment_dir / "experiment_config.json", experiment_config)
    print(
        json.dumps(
            {"split_sizes": split_sizes, "class_counts": split_class_counts},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    model = build_fresh_model(args.seed)
    model_contract = verify_model_contract(model, tf)
    experiment_config["verified_model"] = model_contract
    _write_json(experiment_dir / "experiment_config.json", experiment_config)

    training_data = _make_dataset(
        tf,
        images,
        labels,
        splits["train"],
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_data = _make_dataset(
        tf,
        images,
        labels,
        splits["validation"],
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=new_model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ]
    history = fit_training_data(
        model,
        training_data,
        validation_data,
        args.epochs,
        callbacks,
    )
    _write_json(
        experiment_dir / "training_history.json",
        {name: [float(value) for value in values] for name, values in history.history.items()},
    )
    if not new_model_path.is_file():
        raise RuntimeError("Training completed without saving a best-model checkpoint.")

    selected_model = tf.keras.models.load_model(new_model_path)
    verify_model_contract(selected_model, tf)
    test_data = _make_test_dataset(
        tf, images, splits["test"], args.batch_size
    )
    test_probabilities = selected_model.predict(test_data, verbose=0).reshape(-1)
    test_metrics = classification_metrics(
        labels[splits["test"]], test_probabilities, THRESHOLD
    )
    test_metrics.update(
        {
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_sha256,
            "model_path": str(new_model_path.resolve()),
            "model_sha256": sha256_file(new_model_path),
            "seed": args.seed,
            "threshold": THRESHOLD,
            "test_sample_count": split_sizes["test"],
            "test_class_counts": _class_counts(labels, splits["test"]),
            "split_manifest": str((experiment_dir / "split_manifest.json").resolve()),
            "independence_note": (
                "This is an image-level, duplicate-group-aware holdout. "
                "Patient-level independence is not established."
            ),
        }
    )
    _write_json(experiment_dir / "test_metrics.json", test_metrics)
    verify_source_artifacts_unchanged(
        dataset_path,
        original_model_path,
        dataset_sha256,
        original_model_sha256,
    )
    print(json.dumps(test_metrics, indent=2, sort_keys=True))
    print(f"Experiment artifacts: {experiment_dir.resolve()}")
    return experiment_dir


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        environment = verify_environment()
        import tensorflow as tf

        contract = verify_model_save_load(tf, args.seed)
        print(
            json.dumps(
                {"environment": environment, "verified_model": contract},
                indent=2,
                sort_keys=True,
            )
        )
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
