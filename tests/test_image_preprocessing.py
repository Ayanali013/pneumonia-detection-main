import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from image_preprocessing import MAX_UPLOAD_BYTES, prepare_image


def make_image_bytes(image_format: str = "PNG", mode: str = "L") -> bytes:
    image = Image.new(mode, (16, 12))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


class PrepareImageTests(unittest.TestCase):
    def test_prepares_normalized_grayscale_model_input(self) -> None:
        _, model_input = prepare_image(make_image_bytes(), channels=1)

        self.assertEqual(model_input.shape, (1, 180, 180, 1))
        self.assertEqual(model_input.dtype, np.float32)
        self.assertGreaterEqual(float(model_input.min()), 0.0)
        self.assertLessEqual(float(model_input.max()), 1.0)

    def test_repeats_grayscale_values_for_three_channel_models(self) -> None:
        _, model_input = prepare_image(
            make_image_bytes(mode="RGB"), channels=3
        )

        self.assertEqual(model_input.shape, (1, 180, 180, 3))
        np.testing.assert_array_equal(model_input[..., 0], model_input[..., 1])
        np.testing.assert_array_equal(model_input[..., 1], model_input[..., 2])

    def test_rejects_empty_or_oversized_uploads(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            prepare_image(b"", channels=1)
        with self.assertRaisesRegex(ValueError, "10 MB"):
            prepare_image(b"x" * (MAX_UPLOAD_BYTES + 1), channels=1)

    def test_rejects_unreadable_or_unsupported_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "readable"):
            prepare_image(b"not an image", channels=1)
        with self.assertRaisesRegex(ValueError, "JPEG and PNG"):
            prepare_image(make_image_bytes("GIF"), channels=1)

    def test_rejects_unsupported_channel_count_and_pixel_dimensions(self) -> None:
        image_bytes = make_image_bytes()
        with self.assertRaisesRegex(ValueError, "channels"):
            prepare_image(image_bytes, channels=2)
        with patch("image_preprocessing.MAX_IMAGE_PIXELS", 10):
            with self.assertRaisesRegex(ValueError, "dimensions"):
                prepare_image(image_bytes, channels=1)


if __name__ == "__main__":
    unittest.main()
