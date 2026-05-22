"""utils.io_safe 测试。"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pytest

from utils import imread_unchanged, imread_with_flag, imwrite_safe, gdal_open


def test_imread_imwrite_roundtrip_uint8(tmp_path):
    """中文路径 + uint8 图像往返。"""
    p = tmp_path / "测试图.png"
    img = (np.random.rand(8, 8) * 255).astype(np.uint8)
    assert imwrite_safe(str(p), img) is True
    loaded = imread_unchanged(str(p))
    assert loaded is not None
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, img)


def test_imread_unchanged_preserves_uint16(tmp_path):
    p = tmp_path / "label_uint16.png"
    img = np.array([[0, 1000, 60000], [1, 2, 65535]], dtype=np.uint16)
    assert imwrite_safe(str(p), img) is True
    loaded = imread_unchanged(str(p))
    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, img)


def test_imread_with_flag_grayscale(tmp_path):
    p = tmp_path / "rgb.png"
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 0] = 255  # blue 通道
    assert imwrite_safe(str(p), img) is True
    gray = imread_with_flag(str(p), cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    assert gray.ndim == 2


def test_imread_missing_returns_none(tmp_path):
    p = tmp_path / "does_not_exist.png"
    assert imread_unchanged(str(p)) is None


def test_imread_empty_returns_none(tmp_path):
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    assert imread_unchanged(str(p)) is None


def test_imread_corrupt_returns_none(tmp_path):
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"this is not a valid png")
    assert imread_unchanged(str(p)) is None


def test_imwrite_no_extension_fails(tmp_path):
    p = tmp_path / "noext"
    img = np.zeros((4, 4), dtype=np.uint8)
    assert imwrite_safe(str(p), img) is False


def test_gdal_open_invalid_mode():
    osgeo = pytest.importorskip("osgeo")  # noqa: F841
    with pytest.raises(ValueError):
        gdal_open("anything", mode="rw")


def test_gdal_open_missing_returns_none(tmp_path):
    pytest.importorskip("osgeo")
    p = tmp_path / "no_such.tif"
    assert gdal_open(str(p)) is None
