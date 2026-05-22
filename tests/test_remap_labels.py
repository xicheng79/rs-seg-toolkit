"""remap_labels 测试。"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pytest

from remap_labels import process_labels, remap_image, _build_lut


def _write_png(path, arr):
    cv2.imencode('.png', arr)[1].tofile(str(path))


def test_remap_basic_uint8(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_png(src / "a.png",
               np.array([[0, 30, 30], [50, 30, 0]], dtype=np.uint8))

    process_labels(str(src), str(dst), {30: 1}, ext='.png',
                   unmapped='keep', unmapped_value=0)

    out = cv2.imdecode(np.fromfile(str(dst / "a.png"), dtype=np.uint8),
                       cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(
        out, np.array([[0, 1, 1], [50, 1, 0]], dtype=np.uint8))


def test_remap_unmapped_set(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_png(src / "a.png",
               np.array([[10, 30, 99]], dtype=np.uint8))

    process_labels(str(src), str(dst), {30: 1}, ext='.png',
                   unmapped='set', unmapped_value=0)

    out = cv2.imdecode(np.fromfile(str(dst / "a.png"), dtype=np.uint8),
                       cv2.IMREAD_UNCHANGED)
    # 10 和 99 都被设为 0，30 被映射为 1
    np.testing.assert_array_equal(out, np.array([[0, 1, 0]], dtype=np.uint8))


def test_remap_uint16_dtype_preserved(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    arr = np.array([[0, 1000, 60000]], dtype=np.uint16)
    _write_png(src / "a.png", arr)

    process_labels(str(src), str(dst), {1000: 1, 60000: 2},
                   ext='.png', unmapped='keep', unmapped_value=0)

    out = cv2.imdecode(np.fromfile(str(dst / "a.png"), dtype=np.uint8),
                       cv2.IMREAD_UNCHANGED)
    assert out.dtype == np.uint16, "uint16 标签必须保留 dtype"
    np.testing.assert_array_equal(out, np.array([[0, 1, 2]], dtype=np.uint16))


def test_remap_ext_case_insensitive(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_png(src / "a.PNG",
               np.array([[30]], dtype=np.uint8))

    process_labels(str(src), str(dst), {30: 1}, ext='.png')

    # 大小写不敏感：.PNG 也应被处理
    assert (dst / "a.PNG").exists() or (dst / "a.png").exists()


def test_build_lut_uint8():
    lut = _build_lut({30: 1, 50: 2}, np.uint8, unmapped='keep', unmapped_value=0)
    assert lut.dtype == np.uint8
    assert lut.shape == (256,)
    assert lut[30] == 1
    assert lut[50] == 2
    assert lut[10] == 10  # keep
    assert lut[0] == 0


def test_build_lut_unmapped_set():
    lut = _build_lut({30: 1}, np.uint8, unmapped='set', unmapped_value=0)
    assert lut[30] == 1
    assert lut[10] == 0  # set
    assert lut[0] == 0


def test_remap_image_uses_lut():
    arr = np.array([[0, 30, 50]], dtype=np.uint8)
    out = remap_image(arr, {30: 1, 50: 2}, unmapped='keep', unmapped_value=0)
    np.testing.assert_array_equal(out, np.array([[0, 1, 2]], dtype=np.uint8))
    assert out.dtype == np.uint8


def test_remap_value_out_of_uint8_range():
    """超出 uint8 范围的映射键应该报错。"""
    with pytest.raises(ValueError):
        _build_lut({500: 1}, np.uint8, unmapped='keep', unmapped_value=0)
