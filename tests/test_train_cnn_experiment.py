import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_cnn_experiment import (
    EXPECTED_PARAMETER_COUNT,
    MODEL_INPUT_SHAPE,
    MODEL_OUTPUT_SHAPE,
    assert_new_checkpoint_path,
    build_fresh_model,
    create_group_stratified_split,
    sha256_file,
    validate_environment_versions,
    validate_split_indices,
    verify_model_contract,
    verify_source_artifacts_unchanged,
)


def make_grouped_labels_and_hashes() -> tuple[np.ndarray, list[str]]:
    labels = np.repeat(np.array([0, 1], dtype=np.uint8), 100)
    hashes = []
    for label in (0, 1):
        for class_index in range(100):
            if class_index < 20:
                image_hash = f"class-{label}-duplicate-{class_index // 2}"
            else:
                image_hash = f"class-{label}-single-{class_index}"
            hashes.append(image_hash)
    return labels, hashes


class GroupStratifiedSplitTests(unittest.TestCase):
    def test_split_is_reproducible_for_a_fixed_seed(self) -> None:
        labels, hashes = make_grouped_labels_and_hashes()

        first = create_group_stratified_split(labels, hashes, seed=42)
        second = create_group_stratified_split(labels, hashes, seed=42)

        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(first[name], second[name])

    def test_splits_are_disjoint_and_cover_all_rows(self) -> None:
        labels, hashes = make_grouped_labels_and_hashes()
        splits = create_group_stratified_split(labels, hashes, seed=42)
        all_indices = np.concatenate(list(splits.values()))

        self.assertEqual(len(np.unique(all_indices)), len(labels))
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(len(labels)))

    def test_duplicate_hashes_never_cross_splits(self) -> None:
        labels, hashes = make_grouped_labels_and_hashes()
        splits = create_group_stratified_split(labels, hashes, seed=42)
        hash_owner = {}

        for split_name, indices in splits.items():
            for index in indices:
                image_hash = hashes[int(index)]
                if image_hash in hash_owner:
                    self.assertEqual(hash_owner[image_hash], split_name)
                hash_owner[image_hash] = split_name

    def test_split_preserves_approximate_class_stratification(self) -> None:
        labels, hashes = make_grouped_labels_and_hashes()
        splits = create_group_stratified_split(labels, hashes, seed=42)
        overall_positive_fraction = float(np.mean(labels == 1))

        for indices in splits.values():
            split_positive_fraction = float(np.mean(labels[indices] == 1))
            self.assertLessEqual(
                abs(split_positive_fraction - overall_positive_fraction), 0.05
            )

    def test_conflicting_labels_for_identical_hashes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            create_group_stratified_split(
                np.array([0, 1], dtype=np.uint8),
                ["same-image", "same-image"],
                seed=42,
            )

    def test_validator_rejects_hash_leakage_across_splits(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.uint8)
        hashes = ["duplicate", "duplicate", "positive-a", "positive-b"]
        splits = {
            "train": np.array([0, 2]),
            "validation": np.array([1]),
            "test": np.array([3]),
        }

        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_split_indices(labels, hashes, splits)


class TrainingSafetyTests(unittest.TestCase):
    def test_test_data_is_not_an_argument_to_model_fit(self) -> None:
        class FitSpy:
            def fit(self, **kwargs):
                self.kwargs = kwargs
                return "history"

        from train_cnn_experiment import fit_training_data

        train_data = object()
        validation_data = object()
        test_data = object()
        spy = FitSpy()
        fit_signature = inspect.signature(fit_training_data)

        fit_training_data(
            spy,
            training_data=train_data,
            validation_data=validation_data,
            epochs=3,
            callbacks=[],
        )

        self.assertNotIn("test_data", fit_signature.parameters)
        self.assertIs(spy.kwargs["x"], train_data)
        self.assertIs(spy.kwargs["validation_data"], validation_data)
        self.assertNotIn(test_data, spy.kwargs.values())

    def test_experiment_checkpoint_path_cannot_replace_source_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.h5"
            original_model = root / "existing_model.keras"
            new_model = root / "experiment" / "best_model.keras"
            dataset.touch()
            original_model.touch()

            assert_new_checkpoint_path(new_model, (dataset, original_model))
            with self.assertRaisesRegex(ValueError, "protected"):
                assert_new_checkpoint_path(original_model, (dataset, original_model))

    def test_source_artifact_hashes_are_checked_for_unchanged_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "source.h5"
            model = root / "existing.keras"
            dataset.write_bytes(b"synthetic hdf5 fixture")
            model.write_bytes(b"synthetic checkpoint fixture")
            dataset_hash = sha256_file(dataset)
            model_hash = sha256_file(model)

            verify_source_artifacts_unchanged(
                dataset, model, dataset_hash, model_hash
            )
            self.assertEqual(sha256_file(dataset), dataset_hash)
            self.assertEqual(sha256_file(model), model_hash)

    def test_environment_guard_rejects_incompatible_versions(self) -> None:
        required = {
            "tensorflow": "2.16.2",
            "keras": "3.15.1",
            "numpy": "1.26.4",
            "h5py": "3.16.0",
        }
        installed = dict(required, tensorflow="2.15.1")

        with self.assertRaisesRegex(RuntimeError, "tensorflow==2.16.2 required"):
            validate_environment_versions((3, 11), required, installed)


@unittest.skipIf(
    importlib.util.find_spec("tensorflow") is None,
    "TensorFlow is not installed in this interpreter.",
)
class FreshModelTests(unittest.TestCase):
    def test_fresh_model_matches_verified_architecture_and_compile_contract(self) -> None:
        import tensorflow as tf

        model = build_fresh_model(seed=42)
        contract = verify_model_contract(model, tf)

        self.assertEqual(tuple(model.input_shape), MODEL_INPUT_SHAPE)
        self.assertEqual(tuple(model.output_shape), MODEL_OUTPUT_SHAPE)
        self.assertEqual(model.count_params(), EXPECTED_PARAMETER_COUNT)
        self.assertEqual(contract["compile"]["loss"], "binary_crossentropy")
        self.assertEqual(contract["compile"]["optimizer"], "Adam")
        self.assertEqual(contract["compile"]["metrics"], ["accuracy"])
