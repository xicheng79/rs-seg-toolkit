"""compute_label_distribution 测试。"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pytest

from compute_label_distribution import LabelAnalyzer


def _write_label(path, arr):
    cv2.imencode('.png', arr)[1].tofile(str(path))


def test_auto_detect_classes(tmp_path):
    """class_values=None 时自动检测所有类别。"""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _write_label(label_dir / "a.png",
                 np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint8))
    _write_label(label_dir / "b.png",
                 np.array([[0, 0, 1]], dtype=np.uint8))

    analyzer = LabelAnalyzer(folder_path=str(label_dir), ext='.png',
                             class_values=None, ignore_value=None,
                             use_gdal=False)
    report = analyzer.run()

    classes = sorted(r['class'] for r in report['rows'])
    assert classes == [0, 1, 2, 3]
    assert report['total_images'] == 2


def test_explicit_class_values_binary_compat(tmp_path):
    """显式 class_values=[0, 255] 兼容旧版二分类。"""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _write_label(label_dir / "a.png",
                 np.array([[0, 0, 255], [255, 0, 0]], dtype=np.uint8))

    analyzer = LabelAnalyzer(folder_path=str(label_dir), ext='.png',
                             class_values=[0, 255], use_gdal=False)
    report = analyzer.run()
    classes = sorted(r['class'] for r in report['rows'])
    assert classes == [0, 255]


def test_ignore_value_excluded(tmp_path):
    """ignore_value=255 时 255 不计入任何类别也不计入 total_pixels。"""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _write_label(label_dir / "a.png",
                 np.array([[0, 1, 255], [1, 1, 255]], dtype=np.uint8))

    analyzer = LabelAnalyzer(folder_path=str(label_dir), ext='.png',
                             class_values=[0, 1], ignore_value=255,
                             use_gdal=False)
    report = analyzer.run()
    # 总像素 6 - ignore 2 = 4
    assert report['total_pixels'] == 4
    assert report['ignore_pixels'] == 2

    rows = {r['class']: r for r in report['rows']}
    assert rows[0]['pixels'] == 1
    assert rows[1]['pixels'] == 3


def test_inverse_freq_weight_in_report(tmp_path):
    """报告里包含 inverse-frequency 权重。"""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    # 创建不平衡数据：类别 0 占多数
    _write_label(label_dir / "a.png",
                 np.array([[0] * 9 + [1]] * 10, dtype=np.uint8))  # 90 个 0, 10 个 1

    analyzer = LabelAnalyzer(folder_path=str(label_dir), ext='.png',
                             class_values=[0, 1], use_gdal=False)
    report = analyzer.run()
    rows = {r['class']: r for r in report['rows']}
    # 类别 1 的频率低，应有更高的 inv_freq_weight
    assert 'inv_freq_weight' in rows[0]
    assert 'inv_freq_weight' in rows[1]
    assert rows[1]['inv_freq_weight'] > rows[0]['inv_freq_weight']


def test_empty_folder_does_not_crash(tmp_path):
    """空目录场景：不应抛异常。"""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    analyzer = LabelAnalyzer(folder_path=str(label_dir), ext='.png',
                             class_values=None, use_gdal=False)
    # 不应抛异常；返回 None 或空报告均可
    try:
        analyzer.run()
    except Exception as e:
        pytest.fail(f"空目录不应抛异常: {e}")


def test_multi_class_report_rows(tmp_path):
    """多类（5 类）正确输出每类统计。"""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _write_label(label_dir / "a.png",
                 np.array([[0, 1, 2, 3, 4]], dtype=np.uint8))
    _write_label(label_dir / "b.png",
                 np.array([[0, 0, 1, 2, 4]], dtype=np.uint8))

    analyzer = LabelAnalyzer(folder_path=str(label_dir), ext='.png',
                             class_values=None, use_gdal=False)
    report = analyzer.run()
    classes = sorted(r['class'] for r in report['rows'])
    assert classes == [0, 1, 2, 3, 4]
    rows = {r['class']: r for r in report['rows']}
    # 类别 3 只在 a.png 出现
    assert rows[3]['images'] == 1
    # 类别 0 在 a 和 b 都出现
    assert rows[0]['images'] == 2
