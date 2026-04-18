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
    """Return stable, content-identifying bytes for a single image_url entry.

    The PreprocessedRequest carries image_url entries as one of:
      1. {"Url": "<str>"}          - hash the URL string (stable per endpoint)
      2. {"Decoded": "<b64 str>"}  - hash the decoded image bytes
      3. plain str                 - treated as a URL
      4. plain bytes               - treated as already-decoded bytes
    Returns None for unsupported shapes.
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
    """Compute stable multi_modal_uuids from a PreprocessedRequest.

    Hashes the original image_url entries so that prefill and decode workers
    produce the same identifier for the same request, regardless of whether
    either side holds real pixels, pre-computed embeddings, or synthetic
    placeholders. This keeps mm_feature.identifier (and therefore the KV
    block hash) consistent across P and D in a disaggregated deployment.
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
    if not uuids:
        return None
    return {"image": uuids}
