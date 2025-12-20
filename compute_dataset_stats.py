import cv2
import os
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

class DatasetStats:
    def __init__(self, path, img_ext='.png', is_norm=True):
        self.path = path
        self.img_ext = img_ext
        self.is_norm = is_norm
        
    def cv2_imread_safe(self, file_path):
        """安全读取图像，支持中文路径"""
        try:
            return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"读取错误: {file_path} - {e}")
            return None

    def calculate(self):
        # 初始化累加器 (使用 float64 防止溢出)
        # channel_sum: [B_sum, G_sum, R_sum] (OpenCV 默认读入是 BGR，我们内部处理，最后转 RGB)
        cumulative_sum = np.zeros(3, dtype=np.float64)
        cumulative_sq_sum = np.zeros(3, dtype=np.float64)
        total_pixel_count = 0
        
        file_list = [f for f in os.listdir(self.path) if f.endswith(self.img_ext)]
        print(f"开始处理 {len(file_list)} 张图像...")

        for file_name in tqdm(file_list):
            file_path = os.path.join(self.path, file_name)
            
            img = self.cv2_imread_safe(file_path)
            if img is None:
                continue

            # 图像转换：BGR -> RGB (深度学习通常使用 RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 归一化策略优化：
            # 不在循环里除以 255，直接累加原始值，速度更快，精度损失可忽略。
            # 最后结果再除以 255。
            img = img.astype(np.float64)
            
            # reshape 成 (N, 3) 方便计算
            img = img.reshape(-1, 3)
            
            # 累加像素值
            cumulative_sum += np.sum(img, axis=0)
            
            # 累加像素值的平方
            cumulative_sq_sum += np.sum(img ** 2, axis=0)
            
            # 累加像素总数
            total_pixel_count += img.shape[0]

        if total_pixel_count == 0:
            print("未找到有效像素，请检查路径或文件。")
            return

        # --- 计算最终统计量 ---
        
        # Mean = Sum / N
        mean = cumulative_sum / total_pixel_count
        
        # Std = sqrt( E[x^2] - (E[x])^2 )
        # Variance = (Sum_sq / N) - Mean^2
        variance = (cumulative_sq_sum / total_pixel_count) - (mean ** 2)
        std = np.sqrt(variance)

        # 如果需要归一化 (0-1)，则除以 255
        if self.is_norm:
            mean /= 255.0
            std /= 255.0

        self.print_table(mean, std)

    def print_table(self, mean, std):
        table = PrettyTable()
        table.field_names = ["Type", "R", "G", "B"]
        
        # 保留4位小数
        table.add_row(["Mean", f"{mean[0]:.4f}", f"{mean[1]:.4f}", f"{mean[2]:.4f}"])
        table.add_row(["Std",  f"{std[0]:.4f}",  f"{std[1]:.4f}",  f"{std[2]:.4f}"])
        
        print("\n计算结果 (Order: RGB):")
        print(table)
        
        # 方便复制的代码格式
        print(f"\n[Copy for Config]\nMean: {mean.tolist()}\nStd:  {std.tolist()}")

if __name__ == '__main__':
    # 配置区
    DATA_PATH = r'E:\Samples-Water\chengdu\image'  # 修改为你的实际路径
    FILE_EXT = '.tif'  # 通常遥感影像或数据集可能是 png, jpg, tif
    NORMALIZE = True   # 是否输出 0-1 范围的数值

    stats_tool = DatasetStats(DATA_PATH, FILE_EXT, NORMALIZE)
    stats_tool.calculate()