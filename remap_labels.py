"""
将标签图按用户指定的 {old_value: new_value} 映射做像素值重映射。

特性：
- 参数化的 mapping 字典，不再硬编码
- unmapped='keep' 保留未映射值（向后兼容旧行为）
       'set'  把所有未映射值统一为 unmapped_value（清洗杂色）
- 用 numpy LUT（查找表）实现，对 uint8/uint16 都可向量化
- 使用 IMREAD_UNCHANGED 保留 16 位标签的 dtype
- 后缀大小写不敏感，支持 .png/.tif 等
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional


# OpenCV imencode 支持的常见标签后缀
_SUPPORTED_EXTS = {'.png', '.tif', '.tiff', '.bmp'}


def cv2_imread_unchanged(file_path: str):
    """支持中文路径的 OpenCV 读取，IMREAD_UNCHANGED 保留原 dtype。"""
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
                            cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"读取错误: {file_path} - {e}")
        return None


def cv2_imwrite_safe(save_path: str, img: np.ndarray) -> bool:
    """支持中文路径的 OpenCV 写入。返回是否成功。"""
    try:
        ext = os.path.splitext(save_path)[1]
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            print(f"写入失败（imencode 返回 False）: {save_path}")
            return False
        buf.tofile(save_path)
        return True
    except Exception as e:
        print(f"写入错误: {save_path} - {e}")
        return False


def _build_lut(mapping: Dict[int, int], dtype: np.dtype,
               unmapped: str, unmapped_value: int) -> Optional[np.ndarray]:
    """
    为整数 dtype 构造查找表 LUT[v] = mapping.get(v, default)。
    仅在 dtype 是较小的整数（uint8/uint16/int8）时有意义；更大的整数返回 None
    让上层走逐值替换。
    """
    if dtype == np.uint8:
        lut_size = 256
        lut_dtype = np.uint8
    elif dtype == np.uint16:
        lut_size = 65536
        lut_dtype = np.uint16
    elif dtype == np.int8:
        lut_size = 256
        lut_dtype = np.int8
    else:
        return None

    if unmapped == 'keep':
        lut = np.arange(lut_size, dtype=np.int64)
    elif unmapped == 'set':
        lut = np.full(lut_size, unmapped_value, dtype=np.int64)
    else:
        raise ValueError(f"unmapped 必须是 'keep' 或 'set'，得到: {unmapped!r}")

    for old, new in mapping.items():
        if not (0 <= int(old) < lut_size):
            raise ValueError(
                f"mapping 中的旧值 {old} 超出 dtype {dtype} 的范围 [0,{lut_size})")
        lut[int(old)] = int(new)

    # 范围检查：new 值是否能装进 lut_dtype
    info = np.iinfo(lut_dtype)
    if lut.min() < info.min or lut.max() > info.max:
        raise ValueError(
            f"mapping 的新值超出 dtype {lut_dtype} 范围 [{info.min},{info.max}]")

    return lut.astype(lut_dtype)


def _remap_fallback(img: np.ndarray, mapping: Dict[int, int],
                    unmapped: str, unmapped_value: int) -> np.ndarray:
    """LUT 不适用时（如 int32 标签）的逐值替换。"""
    if unmapped == 'set':
        out = np.full_like(img, unmapped_value)
    else:
        out = img.copy()
    for old, new in mapping.items():
        out[img == old] = new
    return out


def remap_image(img: np.ndarray, mapping: Dict[int, int],
                unmapped: str = 'keep', unmapped_value: int = 0) -> np.ndarray:
    """对单张标签图做重映射，返回与输入同 dtype 的新数组。"""
    lut = _build_lut(mapping, img.dtype, unmapped, unmapped_value)
    if lut is not None:
        return lut[img]
    return _remap_fallback(img, mapping, unmapped, unmapped_value)


def process_labels(src_dir: str, dst_dir: str,
                   mapping: Dict[int, int],
                   ext: str = '.png',
                   unmapped: str = 'keep',
                   unmapped_value: int = 0) -> None:
    """
    批量重映射目录下所有标签。

    :param src_dir: 输入目录
    :param dst_dir: 输出目录（自动创建）
    :param mapping: {旧值: 新值} 字典，例如 {30: 1} 或 {0: 0, 255: 1}
    :param ext: 处理的标签后缀（不区分大小写）
    :param unmapped: 'keep' 保留未映射值（默认）；'set' 改为 unmapped_value
    :param unmapped_value: unmapped='set' 时使用的填充值
    """
    if ext.lower() not in _SUPPORTED_EXTS:
        print(f"[警告] 后缀 {ext} 不在常见标签格式 {_SUPPORTED_EXTS} 中，仍会尝试处理。")

    if not os.path.isdir(src_dir):
        print(f"错误: 输入目录不存在: {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith(ext.lower())]
    print(f"开始处理 {len(files)} 个 {ext} 文件，"
          f"映射 {len(mapping)} 条，未映射策略={unmapped}")

    success, fail = 0, 0
    for fn in tqdm(files):
        src_path = os.path.join(src_dir, fn)
        dst_path = os.path.join(dst_dir, fn)
        img = cv2_imread_unchanged(src_path)
        if img is None:
            fail += 1
            continue
        try:
            out = remap_image(img, mapping, unmapped, unmapped_value)
        except ValueError as e:
            print(f"\n[错误] {fn}: {e}")
            fail += 1
            continue
        if cv2_imwrite_safe(dst_path, out):
            success += 1
        else:
            fail += 1

    print(f"完成。成功: {success}, 失败: {fail}")


if __name__ == '__main__':
    # ====== 配置区 ======
    SRC_DIR = r"E:\Samples-Water\chengdu\label-png"
    DST_DIR = r"E:\Samples-Water\chengdu\label-png-new"

    # 映射字典：{旧像素值: 新像素值}
    # 示例 1：旧版兼容（前景 30 -> 1）
    MAPPING = {30: 1}
    # 示例 2：多类重排（注释掉示例 1 启用）
    # MAPPING = {0: 0, 50: 1, 100: 2, 150: 3, 200: 4}
    # 示例 3：二值化（255 -> 1，其他 -> 0）
    # MAPPING = {255: 1}; UNMAPPED = 'set'; UNMAPPED_VALUE = 0

    EXT = '.png'
    UNMAPPED = 'keep'    # 'keep' 保留未映射值；'set' 将其改为 UNMAPPED_VALUE
    UNMAPPED_VALUE = 0

    process_labels(SRC_DIR, DST_DIR, MAPPING,
                   ext=EXT, unmapped=UNMAPPED, unmapped_value=UNMAPPED_VALUE)