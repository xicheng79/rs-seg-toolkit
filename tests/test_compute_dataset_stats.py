"""compute_dataset_stats 测试（OpenCV 模式；GDAL 模式 importorskip）。"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pytest

from compute_dataset_stats import DatasetStats


def _write_uint8_png(path, arr):
    cv2.imencode('.png', arr)[1].tofile(str(path))


def test_single_band_uint8_normalized(tmp_path):
    """单波段 uint8 归一化。"""
    src = tmp_path / "imgs"
    src.mkdir()
    _write_uint8_png(src / "a.png",
                     np.array([[100, 200], [50, 0]], dtype=np.uint8))
    _write_uint8_png(src / "b.png",
                     np.array([[100, 100], [100, 100]], dtype=np.uint8))

    stats = DatasetStats(path=str(src), img_ext='.png', is_norm=True,
                         nodata=None, use_gdal=False)
    result = stats.calculate()
    assert result is not None
    # mean/std 应在 [0, 1]，因 uint8 归一到 /255
    assert 0.0 <= result['mean'][0] <= 1.0
    assert 0.0 <= result['std'][0] <= 1.0
    assert result['bands'] == 1
    assert result['dtype'] == 'uint8'
    assert result['pixel_count'] == 8


def test_no_normalize_returns_raw(tmp_path):
    """is_norm=False 时输出原始像素值。"""
    src = tmp_path / "imgs"
    src.mkdir()
    _write_uint8_png(src / "a.png",
                     np.array([[100, 100], [100, 100]], dtype=np.uint8))

    stats = DatasetStats(path=str(src), img_ext='.png', is_norm=False,
                         nodata=None, use_gdal=False)
    result = stats.calculate()
    # 单值数据，mean=100, std=0
    assert result['mean'][0] == pytest.approx(100.0)
    assert result['std'][0] == pytest.approx(0.0)


def test_three_band_uint8(tmp_path):
    """3 波段 uint8。注意：OpenCV imread 默认 BGR，但 DatasetStats 内部转换为 RGB
    后统计，所以输出 mean 是 RGB 顺序。"""
    src = tmp_path / "imgs"
    src.mkdir()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 0] = 50    # OpenCV BGR 中的 B
    img[..., 1] = 100   # G
    img[..., 2] = 200   # R
    _write_uint8_png(src / "a.png", img)

    stats = DatasetStats(path=str(src), img_ext='.png', is_norm=False,
                         nodata=None, use_gdal=False)
    result = stats.calculate()
    assert result['bands'] == 3
    # 内部已转为 RGB，所以输出顺序 [R, G, B] = [200, 100, 50]
    assert result['mean'][0] == pytest.approx(200.0)
    assert result['mean'][1] == pytest.approx(100.0)
    assert result['mean'][2] == pytest.approx(50.0)


def test_nodata_excludes_pixels(tmp_path):
    """NoData=0：所有波段都为 0 的像素不计入。"""
    src = tmp_path / "imgs"
    src.mkdir()
    arr = np.array([[0, 100], [100, 100]], dtype=np.uint8)
    _write_uint8_png(src / "a.png", arr)

    stats_with_nodata = DatasetStats(path=str(src), img_ext='.png',
                                     is_norm=False, nodata=0, use_gdal=False)
    r1 = stats_with_nodata.calculate()
    # 排除 1 个 NoData 像素后只剩 3 个 100 -> mean=100
    assert r1['mean'][0] == pytest.approx(100.0)
    assert r1['pixel_count'] == 3

    stats_without = DatasetStats(path=str(src), img_ext='.png',
                                 is_norm=False, nodata=None, use_gdal=False)
    r2 = stats_without.calculate()
    # 不排除：mean = (0+100+100+100)/4 = 75
    assert r2['mean'][0] == pytest.approx(75.0)
    assert r2['pixel_count'] == 4


def test_empty_folder_returns_none(tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    stats = DatasetStats(path=str(src), img_ext='.png', use_gdal=False)
    assert stats.calculate() is None


def test_inconsistent_bands_skipped(tmp_path, capsys):
    """波段数不一致的图像被跳过，不破坏统计。"""
    src = tmp_path / "imgs"
    src.mkdir()
    # 第一张：3 波段
    _write_uint8_png(src / "a.png",
                     np.full((4, 4, 3), 100, dtype=np.uint8))
    # 第二张：单波段（OpenCV 灰度图保存后再读为单波段）
    _write_uint8_png(src / "b.png",
                     np.full((4, 4), 50, dtype=np.uint8))

    stats = DatasetStats(path=str(src), img_ext='.png', is_norm=False,
                         use_gdal=False)
    result = stats.calculate()
    # 第一张定型 3 波段，第二张不一致被跳过
    assert result['bands'] == 3
    assert result['skipped'] >= 1


def test_gdal_mode_multi_band_uint16(tmp_path):
    """GDAL 模式下读取多波段 uint16 数据。"""
    pytest.importorskip("osgeo")
    from osgeo import gdal

    src = tmp_path / "imgs"
    src.mkdir()
    big_path = str(src / "a.tif")

    H = W = 4
    bands = 4
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(big_path, W, H, bands, gdal.GDT_UInt16)
    for b in range(bands):
        ds.GetRasterBand(b + 1).WriteArray(
            np.full((H, W), (b + 1) * 1000, dtype=np.uint16))
    ds.FlushCache()
    ds = None

    stats = DatasetStats(path=str(src), img_ext='.tif', is_norm=False,
                         nodata=None, use_gdal=True)
    result = stats.calculate()
    assert result['bands'] == 4
    assert result['dtype'] == 'uint16'
    for b in range(bands):
        assert result['mean'][b] == pytest.approx((b + 1) * 1000.0)
