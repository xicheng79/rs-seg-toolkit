"""convert_png_to_geotiff 测试。

P1 关键修复：尺寸不一致时默认拒绝写入，避免错误坐标。
完全依赖 GDAL，无 GDAL 时跳过整个文件。
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pytest

pytest.importorskip("osgeo")

from osgeo import gdal  # noqa: E402

from convert_png_to_geotiff import process_georeference  # noqa: E402


def _make_ref_tif(path, w, h, *, gt=(100.0, 1.0, 0.0, 200.0, 0.0, -1.0)):
    """创建一个最小参考 GeoTIFF。"""
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, w, h, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(gt)
    # 用 EPSG:4326 (WGS84) 投影
    srs = gdal.osr.SpatialReference() if hasattr(gdal, 'osr') else None
    if srs is None:
        from osgeo import osr
        srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.zeros((h, w), dtype=np.uint8))
    ds.FlushCache()
    ds = None


def _write_png(path, arr):
    cv2.imencode('.png', arr)[1].tofile(str(path))


def test_size_match_writes_geotiff(tmp_path):
    """尺寸一致时正常写入。"""
    ref = tmp_path / "ref.tif"
    mask = tmp_path / "mask.png"
    out = tmp_path / "out.tif"

    _make_ref_tif(str(ref), w=8, h=8)
    _write_png(mask, (np.random.rand(8, 8) * 255).astype(np.uint8))

    process_georeference(str(ref), str(mask), str(out), force=False)
    assert out.exists()

    # 验证写入文件具有有效的 GeoTransform
    ds = gdal.Open(str(out))
    assert ds is not None
    gt = ds.GetGeoTransform()
    assert gt[0] == pytest.approx(100.0)
    assert gt[3] == pytest.approx(200.0)
    ds = None


def test_size_mismatch_default_refuses(tmp_path, capsys):
    """尺寸不一致时默认（force=False）拒绝写入。"""
    ref = tmp_path / "ref.tif"
    mask = tmp_path / "mask.png"
    out = tmp_path / "out.tif"

    _make_ref_tif(str(ref), w=8, h=8)
    # PNG 尺寸故意不同
    _write_png(mask, (np.random.rand(4, 4) * 255).astype(np.uint8))

    process_georeference(str(ref), str(mask), str(out))  # 默认 force=False
    assert not out.exists(), "尺寸不匹配时默认必须拒绝写入"

    captured = capsys.readouterr()
    out_text = captured.out + captured.err
    assert "尺寸" in out_text or "size" in out_text.lower() or "拒绝" in out_text


def test_size_mismatch_force_true_writes(tmp_path, capsys):
    """force=True 时即使尺寸不匹配也强制写入（用户明确知情）。"""
    ref = tmp_path / "ref.tif"
    mask = tmp_path / "mask.png"
    out = tmp_path / "out.tif"

    _make_ref_tif(str(ref), w=8, h=8)
    _write_png(mask, (np.random.rand(4, 4) * 255).astype(np.uint8))

    process_georeference(str(ref), str(mask), str(out), force=True)
    assert out.exists(), "force=True 必须强制写入"


def test_missing_reference_returns_gracefully(tmp_path):
    """参考影像不存在时函数不应抛异常，安全 return。"""
    mask = tmp_path / "mask.png"
    out = tmp_path / "out.tif"
    _write_png(mask, np.zeros((4, 4), dtype=np.uint8))

    # 不应抛异常
    process_georeference(str(tmp_path / "no_such.tif"),
                         str(mask), str(out), force=False)
    assert not out.exists()


def test_missing_mask_returns_gracefully(tmp_path):
    """PNG 不存在时函数不应抛异常。"""
    ref = tmp_path / "ref.tif"
    out = tmp_path / "out.tif"
    _make_ref_tif(str(ref), w=8, h=8)

    process_georeference(str(ref),
                         str(tmp_path / "no_such.png"),
                         str(out), force=False)
    assert not out.exists()
