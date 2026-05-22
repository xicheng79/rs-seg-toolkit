"""clip_image 测试。

clip_image.py 严重依赖 GDAL（gdal_open + gdal.GetDriverByName），
没有 GDAL 时整个模块跳过。
"""
from __future__ import annotations

import os
import numpy as np
import pytest

pytest.importorskip("osgeo")

import cv2  # noqa: E402
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


# ---------- 以下为最小可行改造（坐标制命名 / auto 扩展名 / 小图策略 / 参数校验）的回归测试 ----------

def _make_geotiff(path, width, height, bands=3, dtype=gdal.GDT_Byte,
                  fill_pattern='ramp', color_interp=None, nodata=None,
                  arrays=None):
    """生成一张带坐标的合成 GeoTIFF 用于测试。

    :param color_interp: list[int]，每个波段的 GDAL ColorInterpretation
    :param nodata: 设置 band1 的 nodata 元数据
    :param arrays: list[ndarray]，显式指定每个波段的数据（覆盖 fill_pattern）
    """
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(path, width, height, bands, dtype)
    ds.SetGeoTransform((100.0, 1.0, 0.0, 200.0, 0.0, -1.0))
    np_dtype = {gdal.GDT_Byte: np.uint8, gdal.GDT_UInt16: np.uint16,
                gdal.GDT_Float32: np.float32}[dtype]
    for b in range(bands):
        if arrays is not None:
            arr = arrays[b].astype(np_dtype)
        elif fill_pattern == 'ramp':
            row = np.arange(width, dtype=np_dtype)
            arr = np.tile(row, (height, 1)) + b * 10
        else:
            arr = np.full((height, width), (b + 1) * 5, dtype=np_dtype)
        band = ds.GetRasterBand(b + 1)
        band.WriteArray(arr.astype(np_dtype))
        if color_interp is not None:
            band.SetColorInterpretation(color_interp[b])
        if nodata is not None and b == 0:
            band.SetNoDataValue(float(nodata))
    ds.FlushCache()
    ds = None


def test_clip_validates_args(tmp_path):
    from clip_image import clip_image_gdal
    src = str(tmp_path / "x.tif")
    _make_geotiff(src, 100, 100, bands=3)
    dst = str(tmp_path / "out")

    with pytest.raises(ValueError):
        clip_image_gdal(src, dst, crop_size=0)
    with pytest.raises(ValueError):
        clip_image_gdal(src, dst, crop_size=64, overlap_ratio=1.0)
    with pytest.raises(ValueError):
        clip_image_gdal(src, dst, crop_size=64, overlap_ratio=-0.1)
    with pytest.raises(ValueError):
        clip_image_gdal(src, dst, crop_size=64, dst_ext='.bmp')
    with pytest.raises(ValueError):
        clip_image_gdal(src, dst, crop_size=64, small_image='reflect')
    with pytest.raises(ValueError):
        clip_image_gdal(src, dst, crop_size=64, edge_policy='bad')


def test_clip_uses_coordinate_naming_and_auto_png(tmp_path):
    """3 波段 uint8 -> auto 应选 .png，且文件名为坐标制 _x000000_y000000.png。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "img.tif")
    _make_geotiff(src, 200, 200, bands=3)  # 200x200，crop=128，重叠 0 -> 出 4 块
    dst = str(tmp_path / "out")

    stat = clip_image_gdal(src, dst, crop_size=128, overlap_ratio=0.0, dst_ext='auto')
    assert stat['ext'] == '.png'
    assert stat['failed'] == 0
    files = sorted(os.listdir(dst))
    # 期望左上角坐标：(0,0), (72,0), (0,72), (72,72)（128 对齐 + 末尾补块到 200-128=72）
    expected = {
        'img_x000000_y000000.png',
        'img_x000072_y000000.png',
        'img_x000000_y000072.png',
        'img_x000072_y000072.png',
    }
    assert set(files) == expected


def test_clip_small_image_skip(tmp_path):
    from clip_image import clip_image_gdal
    src = str(tmp_path / "tiny.tif")
    _make_geotiff(src, 64, 64, bands=3)
    dst = str(tmp_path / "out")

    stat = clip_image_gdal(src, dst, crop_size=128, small_image='skip')
    assert stat['skipped'] == 1
    assert stat['total'] == 0
    assert not os.path.isdir(dst) or os.listdir(dst) == []


def test_clip_small_image_pad(tmp_path):
    """small_image='pad' 时输出尺寸严格 = crop_size。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "tiny.tif")
    _make_geotiff(src, 64, 50, bands=3)
    dst = str(tmp_path / "out")

    stat = clip_image_gdal(src, dst, crop_size=128, small_image='pad', dst_ext='.png')
    assert stat['success'] == 1
    out_files = os.listdir(dst)
    assert len(out_files) == 1
    img = cv2.imdecode(
        np.fromfile(os.path.join(dst, out_files[0]), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert img.shape[:2] == (128, 128), f"pad 后尺寸应为 crop_size，实际 {img.shape}"


def test_clip_float32_auto_falls_back_to_tif(tmp_path):
    """float32 多波段无法存 PNG，auto 应自动选 .tif，并保留 dtype/坐标。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "f32.tif")
    _make_geotiff(src, 200, 200, bands=3, dtype=gdal.GDT_Float32)
    dst = str(tmp_path / "out")

    stat = clip_image_gdal(src, dst, crop_size=128, overlap_ratio=0.0, dst_ext='auto')
    assert stat['ext'] == '.tif'
    assert stat['failed'] == 0

    # 验证一个 patch 的 dtype 是 float32 且坐标已平移
    patch_path = os.path.join(dst, 'f32_x000000_y000000.tif')
    pds = gdal.Open(patch_path)
    assert pds is not None
    assert pds.GetRasterBand(1).DataType == gdal.GDT_Float32
    gt = pds.GetGeoTransform()
    # 源 GT 左上 (100, 200), 像素 (1, -1)；x_off=0,y_off=0 -> 与源一致
    assert abs(gt[0] - 100.0) < 1e-6 and abs(gt[3] - 200.0) < 1e-6
    pds = None


def test_clip_explicit_png_on_float_raises(tmp_path):
    """显式指定 .png 但 dtype 不兼容时应抛 ValueError，而非静默丢失。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "f32.tif")
    _make_geotiff(src, 100, 100, bands=3, dtype=gdal.GDT_Float32)
    dst = str(tmp_path / "out")

    with pytest.raises(ValueError):
        clip_image_gdal(src, dst, crop_size=64, dst_ext='.png')


def test_clip_creates_dst_dir(tmp_path):
    """clip_image_gdal 应自动创建不存在的 dst_dir。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "img.tif")
    _make_geotiff(src, 200, 200, bands=3)
    dst = str(tmp_path / "deep" / "nested" / "out")
    assert not os.path.exists(dst)
    stat = clip_image_gdal(src, dst, crop_size=128, overlap_ratio=0.0)
    assert os.path.isdir(dst) and stat['success'] > 0


# ---------- band_order 自动检测 ----------

def test_band_order_auto_detects_rgb(tmp_path):
    """有 ColorInterp = R,G,B 时，band_order=auto 应判定为 rgb -> 写 PNG 时翻转。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "rgb.tif")
    # 三个波段填不同常量，便于检查写入结果通道顺序
    arrays = [np.full((64, 64), 200, np.uint8),  # band1=R=200
              np.full((64, 64), 100, np.uint8),  # band2=G=100
              np.full((64, 64),  50, np.uint8)]  # band3=B=50
    _make_geotiff(src, 64, 64, bands=3,
                  color_interp=[gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand],
                  arrays=arrays)
    dst = str(tmp_path / "out")

    stat = clip_image_gdal(src, dst, crop_size=64, overlap_ratio=0.0,
                           dst_ext='.png', band_order='auto', small_image='pad')
    assert stat['success'] == 1
    out_file = os.path.join(dst, os.listdir(dst)[0])
    img = cv2.imdecode(np.fromfile(out_file, np.uint8), cv2.IMREAD_UNCHANGED)
    # band_order=auto 检测到 rgb -> 写 PNG 前翻转 -> cv2 默认按 BGR 读 -> 读回应该是 (200,100,50)（R,G,B）
    # cv2.imread 默认 BGR，所以 img[0,0] = (B=50, G=100, R=200)
    assert tuple(img[0, 0]) == (50, 100, 200), \
        f"band_order=auto + RGB 输入：cv2 读回应 BGR=(50,100,200)，实际 {tuple(img[0,0])}"


def test_band_order_auto_unknown_falls_back(tmp_path):
    """ColorInterp 全为 Undefined 时，band_order=auto 应退回 legacy bgr_swap=False（不翻转）。

    注：GTiff 驱动会默认给 3 波段设 [Red,Green,Blue]，要测"未知"场景必须显式
    把每个波段的 ColorInterp 设为 GCI_Undefined（=0）。
    """
    from clip_image import clip_image_gdal
    src = str(tmp_path / "unk.tif")
    arrays = [np.full((64, 64), 200, np.uint8),
              np.full((64, 64), 100, np.uint8),
              np.full((64, 64),  50, np.uint8)]
    _make_geotiff(src, 64, 64, bands=3, arrays=arrays,
                  color_interp=[gdal.GCI_Undefined] * 3)
    dst = str(tmp_path / "out")

    stat = clip_image_gdal(src, dst, crop_size=64, overlap_ratio=0.0,
                           dst_ext='.png', band_order='auto', small_image='pad')
    assert stat['success'] == 1
    out_file = os.path.join(dst, os.listdir(dst)[0])
    img = cv2.imdecode(np.fromfile(out_file, np.uint8), cv2.IMREAD_UNCHANGED)
    # 未翻转：内存 (200,100,50) 当 BGR 写出，cv2 读回直接 (200,100,50)
    assert tuple(img[0, 0]) == (200, 100, 50)


def test_band_order_explicit_bgr_no_swap(tmp_path):
    """band_order='bgr' 显式声明：不翻转，与不设 band_order 一致。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "bgr.tif")
    arrays = [np.full((64, 64), 50, np.uint8),   # band1=B
              np.full((64, 64), 100, np.uint8),  # band2=G
              np.full((64, 64), 200, np.uint8)]  # band3=R
    _make_geotiff(src, 64, 64, bands=3, arrays=arrays)
    dst = str(tmp_path / "out")
    clip_image_gdal(src, dst, crop_size=64, overlap_ratio=0.0,
                    dst_ext='.png', band_order='bgr', small_image='pad')
    out_file = os.path.join(dst, os.listdir(dst)[0])
    img = cv2.imdecode(np.fromfile(out_file, np.uint8), cv2.IMREAD_UNCHANGED)
    assert tuple(img[0, 0]) == (50, 100, 200)


# ---------- nodata / 有效率过滤 ----------

def test_min_valid_ratio_filters_nodata_patches(tmp_path):
    """min_valid_ratio=0.5：左半全 nodata 的 patch 应被过滤。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "nd.tif")
    # 200x100 单波段，左 100 列 = 0（nodata），右 100 列 = 100
    arr = np.zeros((100, 200), dtype=np.uint8)
    arr[:, 100:] = 100
    _make_geotiff(src, 200, 100, bands=1, arrays=[arr], nodata=0)
    dst = str(tmp_path / "out")

    # crop=100, overlap=0 -> 左块 (x=0) 全是 nodata；右块 (x=100) 全有效
    stat = clip_image_gdal(src, dst, crop_size=100, overlap_ratio=0.0,
                           dst_ext='.png', min_valid_ratio=0.5)
    assert stat['filtered'] == 1
    assert stat['success'] == 1
    files = os.listdir(dst)
    # 只应留下 x=100 的 patch
    assert len(files) == 1 and 'x000100' in files[0]


def test_min_valid_ratio_no_nodata_metadata(tmp_path):
    """nodata 元数据缺失但用户显式指定 nodata_value 时也能过滤。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "nd2.tif")
    arr = np.zeros((100, 200), dtype=np.uint8)
    arr[:, 100:] = 100
    _make_geotiff(src, 200, 100, bands=1, arrays=[arr])  # 不设 nodata 元数据
    dst = str(tmp_path / "out")

    stat = clip_image_gdal(src, dst, crop_size=100, overlap_ratio=0.0,
                           dst_ext='.png', nodata_value=0, min_valid_ratio=0.5)
    assert stat['filtered'] == 1 and stat['success'] == 1


def test_validates_band_order_and_min_valid_ratio(tmp_path):
    from clip_image import clip_image_gdal
    src = str(tmp_path / "x.tif")
    _make_geotiff(src, 100, 100, bands=3)
    dst = str(tmp_path / "out")
    with pytest.raises(ValueError):
        clip_image_gdal(src, dst, crop_size=64, band_order='unknown')
    with pytest.raises(ValueError):
        clip_image_gdal(src, dst, crop_size=64, min_valid_ratio=1.5)


# ---------- clip_pair 成对裁剪 ----------

def test_clip_pair_basic_alignment(tmp_path):
    """成对裁剪：image/label 同坐标输出，文件名一一对应。"""
    from clip_image import clip_pair
    img_path = str(tmp_path / "scene.tif")
    lbl_path = str(tmp_path / "scene_label.tif")
    _make_geotiff(img_path, 200, 200, bands=3)
    _make_geotiff(lbl_path, 200, 200, bands=1, fill_pattern='const')

    img_dst = str(tmp_path / "img_out")
    lbl_dst = str(tmp_path / "lbl_out")
    stat = clip_pair(img_path, lbl_path, img_dst, lbl_dst,
                     crop_size=128, overlap_ratio=0.0)
    # 200/128 -> [0, 72]，4 块
    assert stat['success'] == 4 and stat['failed'] == 0
    img_files = sorted(os.listdir(img_dst))
    lbl_files = sorted(os.listdir(lbl_dst))
    # 坐标段必须严格对齐
    img_coords = ['_'.join(f.split('_')[-2:]).split('.')[0] for f in img_files]
    lbl_coords = ['_'.join(f.split('_')[-2:]).split('.')[0] for f in lbl_files]
    assert img_coords == lbl_coords


def test_clip_pair_size_mismatch_raises(tmp_path):
    from clip_image import clip_pair
    img_path = str(tmp_path / "img.tif")
    lbl_path = str(tmp_path / "lbl.tif")
    _make_geotiff(img_path, 200, 200, bands=3)
    _make_geotiff(lbl_path, 200, 199, bands=1)  # 故意差一行
    with pytest.raises(ValueError, match="栅格尺寸不一致"):
        clip_pair(img_path, lbl_path,
                  str(tmp_path / "io"), str(tmp_path / "lo"),
                  crop_size=128)


def test_clip_pair_foreground_filter_keeps_pair_in_sync(tmp_path):
    """min_foreground_ratio：label 前景不足时，image 和 label 都不写。"""
    from clip_image import clip_pair
    img_path = str(tmp_path / "img.tif")
    lbl_path = str(tmp_path / "lbl.tif")
    _make_geotiff(img_path, 200, 100, bands=3)
    # label：左半全 0（背景），右半全 1（前景）
    lbl_arr = np.zeros((100, 200), dtype=np.uint8)
    lbl_arr[:, 100:] = 1
    _make_geotiff(lbl_path, 200, 100, bands=1, arrays=[lbl_arr])

    img_dst = str(tmp_path / "img_out")
    lbl_dst = str(tmp_path / "lbl_out")
    stat = clip_pair(img_path, lbl_path, img_dst, lbl_dst,
                     crop_size=100, overlap_ratio=0.0,
                     min_foreground_ratio=0.5)
    # 左块前景比 = 0 < 0.5 被过滤；右块前景比 = 1.0 >= 0.5 保留
    assert stat['filtered'] == 1 and stat['success'] == 1
    img_files = os.listdir(img_dst)
    lbl_files = os.listdir(lbl_dst)
    # 关键：两边数量必须严格相同
    assert len(img_files) == len(lbl_files) == 1
    # 且都是 x=100 的 patch
    assert 'x000100' in img_files[0] and 'x000100' in lbl_files[0]


def test_process_pair_folder_matches_by_stem(tmp_path):
    """process_pair_folder：按 stem 匹配；缺对应的应被跳过且不报错。"""
    from clip_image import process_pair_folder
    img_dir = tmp_path / "images"; img_dir.mkdir()
    lbl_dir = tmp_path / "labels"; lbl_dir.mkdir()
    # 两张 image，其中一张有对应 label，另一张没有
    _make_geotiff(str(img_dir / "a.tif"), 200, 200, bands=3)
    _make_geotiff(str(img_dir / "b.tif"), 200, 200, bands=3)
    _make_geotiff(str(lbl_dir / "a.tif"), 200, 200, bands=1)
    # lbl_dir 还多一个孤儿
    _make_geotiff(str(lbl_dir / "c.tif"), 200, 200, bands=1)

    summary = process_pair_folder(
        str(img_dir), str(lbl_dir),
        dst_image_dir=str(tmp_path / "img_out"),
        dst_label_dir=str(tmp_path / "lbl_out"),
        crop_size=128, overlap_ratio=0.0,
    )
    assert summary['matched'] == 1   # 仅 stem='a' 匹配上
    # a 共 4 块 (200/128 -> [0,72] x [0,72])
    assert summary['success'] == 4
    img_outs = os.listdir(str(tmp_path / "img_out"))
    lbl_outs = os.listdir(str(tmp_path / "lbl_out"))
    assert len(img_outs) == len(lbl_outs) == 4


# ---------- edge_policy 末尾补块策略 ----------

def _x_coords_in_dir(dir_path):
    """从目录里所有文件名解析出 x 坐标段（_xNNNNNN_yNNNNNN.ext）。"""
    out = []
    for f in os.listdir(dir_path):
        # 去扩展名后按 _ 分割，找 xNNNNNN
        stem = os.path.splitext(f)[0]
        parts = stem.split('_')
        x_part = [p for p in parts if p.startswith('x') and p[1:].isdigit()]
        if x_part:
            out.append(int(x_part[-1][1:]))
    return sorted(out)


def test_edge_policy_append_default_matches_old_behavior(tmp_path):
    """默认 edge_policy='append'：width=200,crop=128,overlap=0 -> x_steps=[0,72]。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "img.tif")
    _make_geotiff(src, 200, 128, bands=3)
    dst = str(tmp_path / "out")

    stat = clip_image_gdal(src, dst, crop_size=128, overlap_ratio=0.0,
                           dst_ext='.png')  # 默认 edge_policy='append'
    assert stat['success'] == 2
    xs = _x_coords_in_dir(dst)
    assert xs == [0, 72], f"append 应输出 x=[0,72]，实际 {xs}"


def test_edge_policy_drop_drops_tail(tmp_path):
    """edge_policy='drop'：丢掉末尾不齐的边缘条带，仅留 x=0。"""
    from clip_image import clip_image_gdal
    src = str(tmp_path / "img.tif")
    _make_geotiff(src, 200, 128, bands=3)
    dst = str(tmp_path / "out")

    stat = clip_image_gdal(src, dst, crop_size=128, overlap_ratio=0.0,
                           dst_ext='.png', edge_policy='drop')
    assert stat['success'] == 1
    xs = _x_coords_in_dir(dst)
    assert xs == [0], f"drop 应只保留 x=[0]，实际 {xs}"


def test_edge_policy_pad_keeps_tail_and_pads_zero(tmp_path):
    """edge_policy='pad'：边缘块坐标为 last+stride(=128)，读出 72 列后右侧补 0。

    输出 PNG 尺寸严格 = crop_size；padding 区域应为 0。
    """
    from clip_image import clip_image_gdal
    src = str(tmp_path / "img.tif")
    # 全 100：方便检查 pad 区域确为 0、未 pad 区域确为 100
    arr = np.full((128, 200), 100, dtype=np.uint8)
    _make_geotiff(src, 200, 128, bands=3,
                  arrays=[arr, arr, arr])
    dst = str(tmp_path / "out")

    stat = clip_image_gdal(src, dst, crop_size=128, overlap_ratio=0.0,
                           dst_ext='.png', edge_policy='pad')
    assert stat['success'] == 2
    xs = _x_coords_in_dir(dst)
    assert xs == [0, 128], f"pad 应输出 x=[0,128]，实际 {xs}"

    # 检查 x=128 的边缘块：尺寸 128x128，左 72 列 = 100，右 56 列 = 0
    edge_path = os.path.join(dst, 'img_x000128_y000000.png')
    img = cv2.imdecode(np.fromfile(edge_path, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img.shape[:2] == (128, 128), f"pad 后尺寸必须为 crop_size，实际 {img.shape}"
    # 三波段值都一样，无需关心 BGR 顺序
    left = img[:, :72]
    right = img[:, 72:]
    assert (left == 100).all(), "pad 边缘块左侧 72 列应保留原值 100"
    assert (right == 0).all(), "pad 边缘块右侧 56 列应被补 0"


def test_clip_pair_edge_policy_drop_keeps_pair_sync(tmp_path):
    """成对模式 edge_policy='drop'：image/label 同步只保留 x=0。"""
    from clip_image import clip_pair
    img_path = str(tmp_path / "img.tif")
    lbl_path = str(tmp_path / "lbl.tif")
    _make_geotiff(img_path, 200, 128, bands=3)
    _make_geotiff(lbl_path, 200, 128, bands=1, fill_pattern='const')
    img_dst = str(tmp_path / "img_out")
    lbl_dst = str(tmp_path / "lbl_out")

    stat = clip_pair(img_path, lbl_path, img_dst, lbl_dst,
                     crop_size=128, overlap_ratio=0.0,
                     edge_policy='drop')
    assert stat['success'] == 1
    img_xs = _x_coords_in_dir(img_dst)
    lbl_xs = _x_coords_in_dir(lbl_dst)
    assert img_xs == lbl_xs == [0]


def test_clip_pair_edge_policy_pad_keeps_pair_sync_and_size(tmp_path):
    """成对模式 edge_policy='pad'：image/label 都在 x=128 出边缘块，两边尺寸都为 crop_size。"""
    from clip_image import clip_pair
    img_path = str(tmp_path / "img.tif")
    lbl_path = str(tmp_path / "lbl.tif")
    _make_geotiff(img_path, 200, 128, bands=3)
    _make_geotiff(lbl_path, 200, 128, bands=1, fill_pattern='const')
    img_dst = str(tmp_path / "img_out")
    lbl_dst = str(tmp_path / "lbl_out")

    stat = clip_pair(img_path, lbl_path, img_dst, lbl_dst,
                     crop_size=128, overlap_ratio=0.0,
                     image_dst_ext='.png', label_dst_ext='.png',
                     edge_policy='pad')
    assert stat['success'] == 2
    img_xs = _x_coords_in_dir(img_dst)
    lbl_xs = _x_coords_in_dir(lbl_dst)
    assert img_xs == lbl_xs == [0, 128]

    # 抽查边缘块尺寸：image / label 都应该是 128x128
    img_edge = cv2.imdecode(
        np.fromfile(os.path.join(img_dst, 'img_x000128_y000000.png'), np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    lbl_edge = cv2.imdecode(
        np.fromfile(os.path.join(lbl_dst, 'lbl_x000128_y000000.png'), np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert img_edge.shape[:2] == (128, 128)
    assert lbl_edge.shape[:2] == (128, 128)


