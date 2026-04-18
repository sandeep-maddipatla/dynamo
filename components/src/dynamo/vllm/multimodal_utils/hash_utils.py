# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from typing import Any, Mapping, Sequence

import blake3
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_URL_KEY = "image_url"


def image_to_bytes(img: Any) -> bytes:
    """Convert a supported image object to PNG bytes for hashing."""
    from PIL import Image

    if isinstance(img, bytes):
        return img

    if isinstance(img, Image.Image | np.ndarray):
        return img.tobytes()

    raise TypeError(f"Unsupported image type for hashing: {type(img)}")


def compute_mm_uuids_from_images(images: Sequence[Any]) -> list[str]:
    """
    Compute blake3 hex UUIDs for image inputs.
    """
    uuids: list[str] = []
    for img in images:
        raw_bytes = image_to_bytes(img)
        uuids.append(blake3.blake3(raw_bytes).hexdigest())
    return uuids


def _image_url_entry_to_bytes(entry: Any) -> bytes | None:
    """Return stable bytes for a single image_url entry, or None if unsupported.

    Handles: {"Url": str}, {"Decoded": b64str|bytes}, plain str, plain bytes.
    """
    if isinstance(entry, bytes):
        return entry
    if isinstance(entry, str):
        return entry.encode("utf-8")
    if isinstance(entry, Mapping):
        if "Url" in entry and isinstance(entry["Url"], str):
            return entry["Url"].encode("utf-8")
        if "Decoded" in entry:
            decoded = entry["Decoded"]
            if isinstance(decoded, bytes):
                return decoded
            if isinstance(decoded, str):
                try:
                    return base64.b64decode(decoded)
                except Exception:
                    return decoded.encode("utf-8")
    return None


def compute_mm_uuids_from_request(
    request: Mapping[str, Any] | None,
) -> dict[str, list[str]] | None:
    """Compute stable multi_modal_uuids from image_url entries on the request.

    Hashes the original image_url entries so that P and D workers produce the
    same mm_feature.identifier regardless of whether either side holds real
    pixels, pre-computed embeddings, or synthetic placeholders.
    Returns None if no image_url entries are present.
    """
    if not request:
        return None
    mm_map = request.get("multi_modal_data")
    if not mm_map:
        return None
    entries = mm_map.get(IMAGE_URL_KEY)
    if not entries:
        return None
    if not isinstance(entries, (list, tuple)):
        entries = [entries]

    uuids: list[str] = []
    for entry in entries:
        raw = _image_url_entry_to_bytes(entry)
        if raw is None:
            logger.debug(
                "compute_mm_uuids_from_request: unsupported image_url entry type %s",
                type(entry),
            )
            return None
        uuids.append(blake3.blake3(raw).hexdigest())
    return {"image": uuids} if uuids else None
