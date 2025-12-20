import os
import cv2
import numpy as np
from tqdm import tqdm

def cv2_imread_safe(file_path):
    """安全读取图像，支持中文路径"""
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"读取失败: {file_path} - {e}")
        return None

def cv2_imwrite_safe(save_path, img):
    """安全写入图像，支持中文路径"""
    try:
        ext = os.path.splitext(save_path)[1]
        cv2.imencode(ext, img)[1].tofile(save_path)
    except Exception as e:
        print(f"写入失败: {save_path} - {e}")

def stitch_images(src_dir, grid_w, grid_h, patch_size, overlap=0, img_ext='.png'):
    """
    :param src_dir: 小图所在文件夹
    :param grid_w: 水平方向切片数量 (Column number)
    :param grid_h: 垂直方向切片数量 (Row number)
    :param patch_size: 小图尺寸
    :param overlap: 重叠大小
    """
    # 1. 计算大图理论尺寸
    # 公式：(N * size) - (overlap * (N - 1))
    total_w = grid_w * patch_size - (overlap * (grid_w - 1))
    total_h = grid_h * patch_size - (overlap * (grid_h - 1))

    # 输出目录
    merge_dir = os.path.join(src_dir, 'merged_result')
    os.makedirs(merge_dir, exist_ok=True)

    # 2. 扫描并分组文件
    # 假设文件名格式: {base_name}_{index}.png
    # 我们需要提取 base_name
    files = [f for f in os.listdir(src_dir) if f.endswith(img_ext)]
    
    # 使用字典归类： {'base_name': [file1, file2...]}
    groups = {}
    for f in files:
        # 假设文件名最后一段是序号，前面是基准名
        # 例如: "chengdu_image_1.png" -> base="chengdu_image", id=1
        name_part = os.path.splitext(f)[0]
        if '_' in name_part:
            base_name = "_".join(name_part.split('_')[:-1])
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(f)
    
    print(f"检测到 {len(groups)} 组图像需拼接。目标尺寸: {total_w}x{total_h}")

    # 3. 开始拼接
    for base_name, patch_files in tqdm(groups.items()):
        # 检查切片数量是否足够 (可选)
        expected_count = grid_w * grid_h
        if len(patch_files) != expected_count:
            print(f"\n[警告] {base_name} 切片数量 {len(patch_files)} 与设定网格 {expected_count} 不符，可能拼接失败。")

        # 读取第一张图以确定通道数
        first_img_path = os.path.join(src_dir, f"{base_name}_1{img_ext}")
        if not os.path.exists(first_img_path):
             # 尝试寻找列表中存在的第一个
             first_img_path = os.path.join(src_dir, patch_files[0])
        
        sample_img = cv2_imread_safe(first_img_path)
        if sample_img is None: continue

        # 初始化大图画布
        if len(sample_img.shape) == 2:
            # 单通道 (H, W)
            canvas = np.zeros((total_h, total_w), dtype=sample_img.dtype)
            channels = 1
        else:
            # 多通道 (H, W, C)
            h, w, c = sample_img.shape
            canvas = np.zeros((total_h, total_w, c), dtype=sample_img.dtype)
            channels = c

        # 核心拼接循环
        # 假设切片序号是 行优先 (Row-major) 还是 列优先? 
        # 通常 Cut 工具生成的顺序是：先第一行(左到右)，再第二行。
        # Index = (row * grid_w) + col + 1
        
        for row in range(grid_h):
            for col in range(grid_w):
                # 计算当前切片的序号 (从1开始)
                patch_idx = (row * grid_w) + col + 1
                patch_name = f"{base_name}_{patch_idx}{img_ext}"
                patch_path = os.path.join(src_dir, patch_name)

                if not os.path.exists(patch_path):
                    # print(f"缺失切片: {patch_name}")
                    continue
                
                patch = cv2_imread_safe(patch_path)
                if patch is None: continue

                # 计算在画布上的坐标
                # 注意：这里需要处理 Overlap。通常拼接时只取中心有效区域，或者简单的直接覆盖。
                # 简单策略：直接按步长覆盖（后一张盖前一张的边）
                
                start_x = col * (patch_size - overlap)
                start_y = row * (patch_size - overlap)
                
                end_x = start_x + patch_size
                end_y = start_y + patch_size
                
                # 边界保护
                if end_x > total_w: end_x = total_w
                if end_y > total_h: end_y = total_h
                
                # 截取 patch 的有效部分 (防止最后一块溢出)
                p_h = end_y - start_y
                p_w = end_x - start_x
                
                # 填入画布
                if channels == 1:
                    canvas[start_y:end_y, start_x:end_x] = patch[0:p_h, 0:p_w]
                else:
                    canvas[start_y:end_y, start_x:end_x, :] = patch[0:p_h, 0:p_w, :]

        # 保存结果
        save_path = os.path.join(merge_dir, f"{base_name}{img_ext}")
        cv2_imwrite_safe(save_path, canvas)

if __name__ == "__main__":
    # 配置参数
    # 必须与 CutPicture.py 中的切片参数完全一致
    PARAMS = {
        'src_dir': r"E:\nets-dataset\water\samples-water(chengdu)\changelabel\split",
        'grid_w': 2,       # 水平横着切了几刀 (X_NUM)
        'grid_h': 2,       # 垂直竖着切了几刀 (Y_NUM)
        'patch_size': 512, # 切片大小
        'overlap': 0,      # 重叠大小 (Common)
        'img_ext': '.png'
    }

    stitch_images(**PARAMS)
    print("所有拼接完成。")