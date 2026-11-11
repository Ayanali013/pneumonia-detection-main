from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_SIZE = (180, 180)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG"}


def prepare_image(image_data: bytes, channels: int) -> tuple[Image.Image, np.ndarray]:
    if not image_data:
        raise ValueError("The uploaded file is empty.")
    if len(image_data) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded file exceeds the 10 MB limit.")
    if channels not in (1, 3):
        raise ValueError(f"Unsupported model input channels: {channels}")

    try:
        with Image.open(BytesIO(image_data)) as source:
            if source.format not in SUPPORTED_FORMATS:
                raise ValueError("Only JPEG and PNG images are supported.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("The image dimensions exceed the supported limit.")
            image = source.convert("L").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError) as exc:
        raise ValueError("The uploaded file is not a readable JPEG or PNG image.") from exc

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array[np.newaxis, ..., np.newaxis]
    if channels == 3:
        array = np.concatenate((array, array, array), axis=3)
    return image, array
