"""stitch_images 测试。"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pytest

from stitch_images import stitch_images


def _write_png(path, arr):
    cv2.imencode('.png', arr)[1].tofile(str(path))


def test_stitch_2x2_no_overlap_grayscale(tmp_path):
    """4 张 4x4 切片拼成 8x8 整图。"""
    src = tmp_path / "src"
    src.mkdir()
    base = "scene"

    # 序号 = (row * grid_w) + col + 1，row-major
    # 4 个 patch 各填不同灰度值
    patches = [10, 20, 30, 40]
    for idx, val in enumerate(patches, start=1):
        arr = np.full((4, 4), val, dtype=np.uint8)
        _write_png(src / f"{base}_{idx}.png", arr)

    stitch_images(src_dir=str(src), grid_w=2, grid_h=2,
                  patch_size=4, overlap=0, img_ext='.png')

    out = src / "merged_result" / f"{base}.png"
    assert out.exists()
    canvas = cv2.imdecode(np.fromfile(str(out), dtype=np.uint8),
                          cv2.IMREAD_UNCHANGED)
    assert canvas.shape == (8, 8)
    # 检查 4 个象限
    np.testing.assert_array_equal(canvas[0:4, 0:4], 10)  # 左上
    np.testing.assert_array_equal(canvas[0:4, 4:8], 20)  # 右上
    np.testing.assert_array_equal(canvas[4:8, 0:4], 30)  # 左下
    np.testing.assert_array_equal(canvas[4:8, 4:8], 40)  # 右下


def test_stitch_preserves_uint16(tmp_path):
    """16 位标签切片必须保留 dtype。"""
    src = tmp_path / "src"
    src.mkdir()
    base = "label"
    patches = [1000, 2000, 30000, 60000]
    for idx, val in enumerate(patches, start=1):
        arr = np.full((4, 4), val, dtype=np.uint16)
        _write_png(src / f"{base}_{idx}.png", arr)

    stitch_images(src_dir=str(src), grid_w=2, grid_h=2,
                  patch_size=4, overlap=0, img_ext='.png')

    out = src / "merged_result" / f"{base}.png"
    canvas = cv2.imdecode(np.fromfile(str(out), dtype=np.uint8),
                          cv2.IMREAD_UNCHANGED)
    assert canvas.dtype == np.uint16
    assert canvas[0, 0] == 1000
    assert canvas[7, 7] == 60000


def test_stitch_3channel_color(tmp_path):
    """3 通道切片正确拼接。"""
    src = tmp_path / "src"
    src.mkdir()
    base = "rgb"
    for idx in range(1, 5):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[..., (idx - 1) % 3] = 200  # 各 patch 高亮不同通道
        _write_png(src / f"{base}_{idx}.png", arr)

    stitch_images(src_dir=str(src), grid_w=2, grid_h=2,
                  patch_size=4, overlap=0, img_ext='.png')

    out = src / "merged_result" / f"{base}.png"
    canvas = cv2.imdecode(np.fromfile(str(out), dtype=np.uint8),
                          cv2.IMREAD_UNCHANGED)
    assert canvas.shape == (8, 8, 3)


def test_stitch_missing_patch_warns_continues(tmp_path, capsys):
    """缺失某个切片时仅警告，不抛异常。"""
    src = tmp_path / "src"
    src.mkdir()
    base = "scene"
    # 只放 3 张，缺第 4 块
    for idx in [1, 2, 3]:
        _write_png(src / f"{base}_{idx}.png",
                   np.full((4, 4), idx * 10, dtype=np.uint8))

    # 不应抛异常
    stitch_images(src_dir=str(src), grid_w=2, grid_h=2,
                  patch_size=4, overlap=0, img_ext='.png')

    out = src / "merged_result" / f"{base}.png"
    assert out.exists()
    captured = capsys.readouterr()
    # 应有警告（数量不符）
    assert "警告" in captured.out or "warning" in captured.out.lower()
