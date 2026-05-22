"""clip_image 测试。

clip_image.py 严重依赖 GDAL（gdal_open + gdal.GetDriverByName），
没有 GDAL 时整个模块跳过。
"""
from __future__ import annotations

import os
import numpy as np
import pytest

pytest.importorskip("osgeo")

from osgeo import gdal  # noqa: E402
from clip_image import gdal_to_opencv  # noqa: E402


def test_gdal_to_opencv_default_no_bgr_swap():
    """默认 bgr_swap=False：3 波段输入波段顺序保持原样。"""
    # GDAL 多波段返回 (C, H, W)
    arr = np.zeros((3, 4, 4), dtype=np.uint8)
    arr[0] = 10  # band 1
    arr[1] = 20  # band 2
    arr[2] = 30  # band 3

    out = gdal_to_opencv(arr)
    assert out.shape == (4, 4, 3)
    # 默认不翻转：通道顺序应为 (band1, band2, band3) -> (10, 20, 30)
    np.testing.assert_array_equal(out[0, 0], np.array([10, 20, 30]))


def test_gdal_to_opencv_explicit_bgr_swap():
    """bgr_swap=True：通道顺序翻转（band3, band2, band1）。"""
    arr = np.zeros((3, 4, 4), dtype=np.uint8)
    arr[0] = 10
    arr[1] = 20
    arr[2] = 30

    out = gdal_to_opencv(arr, bgr_swap=True)
    np.testing.assert_array_equal(out[0, 0], np.array([30, 20, 10]))


def test_gdal_to_opencv_single_band():
    """单波段：GDAL 返回 (H, W)，应保持原状。"""
    arr = (np.arange(16, dtype=np.uint8)).reshape(4, 4)
    out = gdal_to_opencv(arr)
    np.testing.assert_array_equal(out, arr)


def test_gdal_to_opencv_4_band_no_swap():
    """4 波段（如 RGBA 或 RGBN）：bgr_swap 不应触发翻转，原样保留。"""
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    for i in range(4):
        arr[i] = (i + 1) * 10
    out = gdal_to_opencv(arr, bgr_swap=True)
    # 仅 3 波段才做 BGR 翻转，4 波段保持原顺序
    assert out.shape == (4, 4, 4)
    np.testing.assert_array_equal(out[0, 0], np.array([10, 20, 30, 40]))


def test_process_folder_callable():
    """烟雾测试：process_folder 与 clip_image_gdal 可被导入且 bgr_swap 默认为 False。"""
    import inspect
    from clip_image import process_folder, clip_image_gdal

    sig = inspect.signature(clip_image_gdal)
    assert sig.parameters['bgr_swap'].default is False, \
        "clip_image_gdal 默认 bgr_swap 必须为 False，避免静默破坏波段语义"

    sig2 = inspect.signature(process_folder)
    # process_folder 也应支持透传 bgr_swap
    assert 'bgr_swap' in sig2.parameters
