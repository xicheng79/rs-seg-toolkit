import os
import numpy as np
import cv2
from osgeo import gdal
from tqdm import tqdm

# --- 辅助函数：解决中文路径问题的 OpenCV 读写 ---
def cv2_imwrite_safe(save_path, img):
    try:
        # 获取文件扩展名
        ext = os.path.splitext(save_path)[1]
        cv2.imencode(ext, img)[1].tofile(save_path)
    except Exception as e:
        print(f"写入失败: {save_path} - {e}")

# --- 核心转换函数 ---
def gdal_to_opencv(gdal_data, bgr_swap=False):
    """
    将 GDAL ReadAsArray 的 (Bands, Height, Width) 转为 OpenCV 的 (Height, Width, Bands)。

    :param gdal_data: GDAL 读出的 ndarray，(C,H,W) 或单波段 (H,W)
    :param bgr_swap: 仅当输入恰好是 3 波段，且你确定波段顺序是 RGB、需要保存为 OpenCV BGR
                     才设为 True。默认 False（保持原波段顺序，不做颜色翻转）。
                     注意：遥感影像第 1/2/3 波段不一定就是 R/G/B，盲目翻转会破坏数据语义。
    """
    # 维度转换：(C, H, W) -> (H, W, C)
    if len(gdal_data.shape) == 3:
        opencv_data = np.transpose(gdal_data, (1, 2, 0))

        # 仅在用户明确要求时做 RGB <-> BGR 翻转
        if bgr_swap and opencv_data.shape[2] == 3:
            opencv_data = cv2.cvtColor(opencv_data, cv2.COLOR_RGB2BGR)
    else:
        # 单波段
        opencv_data = gdal_data

    return opencv_data

def clip_image_gdal(src_path, dst_dir, crop_size=1024, overlap_ratio=0.1,
                   dst_ext='.png', bgr_swap=False):
    """
    使用 GDAL 分块读取并裁剪
    :param overlap_ratio: 重叠率，0.1 表示重叠 10%
    :param bgr_swap: 见 gdal_to_opencv 的说明；默认 False
    """
    # 1. 打开图像
    dataset = gdal.Open(src_path)
    if not dataset:
        print(f"无法打开文件: {src_path}")
        return

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    
    # 获取文件名
    filename = os.path.splitext(os.path.basename(src_path))[0]
    
    # 2. 计算步长 (Stride)
    # 步长 = 裁剪尺寸 * (1 - 重叠率)
    stride = int(crop_size * (1 - overlap_ratio))
    
    # 确保步长至少为1
    stride = max(1, stride)

    # 3. 动态计算切片位置
    # 使用 range 生成左上角坐标列表
    x_steps = list(range(0, width - crop_size + 1, stride))
    y_steps = list(range(0, height - crop_size + 1, stride))
    
    # 处理边缘情况：如果最后一块没覆盖到边缘，强制添加一个从边缘往回数的块
    if (width - crop_size) % stride != 0:
        x_steps.append(width - crop_size)
    if (height - crop_size) % stride != 0:
        y_steps.append(height - crop_size)

    # 如果图比裁剪尺寸还小，直接取整个图（或跳过，看需求）
    if width < crop_size: x_steps = [0]
    if height < crop_size: y_steps = [0]
    
    total_patches = len(x_steps) * len(y_steps)
    
    # print(f"正在处理: {filename}, 尺寸: {width}x{height}, 预计裁剪: {total_patches} 张")

    count = 0
    for y in y_steps:
        for x in x_steps:
            count += 1
            
            # 计算实际读取的宽高（处理小图情况）
            curr_w = min(crop_size, width - x)
            curr_h = min(crop_size, height - y)
            
            # GDAL 读取: ReadAsArray(x_off, y_off, x_size, y_size)
            # 无论多少波段，ReadAsArray 都会自动处理
            data = dataset.ReadAsArray(x, y, curr_w, curr_h)
            
            # 转换为 OpenCV 格式
            img_cv = gdal_to_opencv(data, bgr_swap=bgr_swap)

            # 构建保存路径
            save_name = f"{filename}_{count}{dst_ext}"
            save_path = os.path.join(dst_dir, save_name)

            # 保存
            cv2_imwrite_safe(save_path, img_cv)

    # 释放 GDAL 数据集
    dataset = None

def process_folder(src_dir, crop_size=1024, bgr_swap=False):
    """
    :param bgr_swap: 见 gdal_to_opencv 的说明；默认 False，保持原波段顺序
    """
    # 自动创建输出目录
    dst_dir = os.path.join(src_dir, f"crop_{crop_size}")
    os.makedirs(dst_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.tif', '.tiff', '.img'))]

    print(f"找到 {len(files)} 个影像文件，准备开始裁剪...")

    for f in tqdm(files):
        src_path = os.path.join(src_dir, f)
        # 调用裁剪函数
        clip_image_gdal(src_path, dst_dir, crop_size=crop_size, bgr_swap=bgr_swap)

if __name__ == "__main__":
    # 配置区域
    INPUT_DIR = r"E:\Samples-Water\chengdu\image"  # 你的输入路径
    CROP_SIZE = 1024
    # 是否做 RGB<->BGR 翻转。
    # 仅当输入恰为 3 波段、波段顺序确为 RGB、且后续工具按 BGR 解读时才设为 True。
    # 遥感多波段 / 单波段 / 非 RGB 波段顺序的场景请保持 False。
    BGR_SWAP = False

    process_folder(INPUT_DIR, CROP_SIZE, bgr_swap=BGR_SWAP)
    print("所有处理完成。")