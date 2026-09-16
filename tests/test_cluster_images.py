"""Tests for cluster_images, focused on mirrored-duplicate detection.

Fixtures are generated procedurally rather than committed as binaries: ORB needs
corners to latch onto, so each "photo" is a deterministic arrangement of shapes
and speckle, and each variant is a transform of that same array.
"""

import cv2
import numpy as np
import pytest

from image_deduplication import cluster_images


def _make_image(seed, size=480):
    """A deterministic, feature-rich synthetic photo."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 30, dtype=np.uint8)
    for _ in range(14):
        x, y = rng.integers(20, size - 80, 2)
        w, h = rng.integers(40, 110, 2)
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        else:
            cv2.circle(img, (x, y), int(w / 2), color, -1)
    for _ in range(9):
        p1 = tuple(int(v) for v in rng.integers(0, size, 2))
        p2 = tuple(int(v) for v in rng.integers(0, size, 2))
        cv2.line(img, p1, p2, (255, 255, 255), 2)
    # Speckle gives ORB plenty of additional corners to key on.
    noise = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
    return cv2.add(img, noise)


def _write(tmp_path, name, img):
    path = tmp_path / f"{name}.png"
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def base_image():
    return _make_image(seed=11)


def _clustered_together(clusters, a, b):
    return any(a in group and b in group for group in clusters)


def test_mirrored_duplicate_is_missed_by_default(tmp_path, base_image):
    """Documents the default: ORB descriptors are not mirror invariant."""
    a = _write(tmp_path, "original", base_image)
    b = _write(tmp_path, "mirrored", cv2.flip(base_image, 1))

    clusters = cluster_images([a, b])

    assert not _clustered_together(clusters, a, b)


def test_mirrored_duplicate_is_found_when_enabled(tmp_path, base_image):
    a = _write(tmp_path, "original", base_image)
    b = _write(tmp_path, "mirrored", cv2.flip(base_image, 1))

    clusters = cluster_images([a, b], detect_mirrored=True)

    assert _clustered_together(clusters, a, b)


def test_unrelated_images_stay_apart_when_enabled(tmp_path):
    """The mirrored pass must not buy recall with false positives."""
    paths = [
        _write(tmp_path, f"photo{i}", _make_image(seed=100 + i))
        for i in range(4)
    ]

    clusters = cluster_images(paths, detect_mirrored=True)

    assert clusters == []


def test_cropped_duplicate_still_found(tmp_path, base_image):
    """Regression: the existing (non-mirrored) behaviour is unchanged."""
    h, w = base_image.shape[:2]
    cropped = base_image[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]
    a = _write(tmp_path, "original", base_image)
    b = _write(tmp_path, "cropped", cropped)

    for enabled in (False, True):
        clusters = cluster_images([a, b], detect_mirrored=enabled)
        assert _clustered_together(clusters, a, b), f"detect_mirrored={enabled}"


def test_featureless_image_does_not_crash(tmp_path, base_image):
    """A blank frame yields no ORB descriptors; it must be skipped, not raise."""
    blank = np.zeros_like(base_image)
    a = _write(tmp_path, "original", base_image)
    b = _write(tmp_path, "blank", blank)

    for enabled in (False, True):
        assert cluster_images([a, b], detect_mirrored=enabled) == []
