import cv2
import os
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

class LabelAnalyzer:
    def __init__(self, folder_path, ext='.png', target_value=255, background_value=0):
        """
        :param folder_path: 标签文件夹路径
        :param ext: 文件后缀
        :param target_value: 正样本像素值 (通常为 255 或 1)
        :param background_value: 负样本/背景像素值 (通常为 0)
        """
        self.folder_path = folder_path
        self.ext = ext
        self.target_val = target_value
        self.bg_val = background_value

    def cv2_imread_chinese(self, file_path):
        """安全读取中文路径，强制以灰度模式读取"""
        try:
            # 标签文件一定要用 GRAYSCALE 读取，避免 BGR/RGB 混乱
            return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"读取错误: {file_path} - {e}")
            return None

    def run(self):
        # 统计计数器
        stats = {
            'pos_imgs': 0,      # 包含正样本的图片数
            'neg_imgs': 0,      # 纯背景图片数
            'pos_pixels': 0,    # 正样本像素总数
            'neg_pixels': 0,    # 负样本像素总数
            'ignore_pixels': 0, # 既不是正也不是负的杂色像素
            'total_pixels': 0
        }

        files = [f for f in os.listdir(self.folder_path) if f.endswith(self.ext)]
        print(f"开始分析 {len(files)} 张标签图像...")

        for file_name in tqdm(files):
            file_path = os.path.join(self.folder_path, file_name)
            img = self.cv2_imread_chinese(file_path)

            if img is None:
                continue

            # --- Numpy 高速统计核心 ---
            # 统计当前图中正样本像素个数
            # 使用 np.count_nonzero 或 sum 都可以，速度极快
            curr_pos = np.count_nonzero(img == self.target_val)
            curr_neg = np.count_nonzero(img == self.bg_val)
            
            # 图像总像素
            h, w = img.shape
            curr_total = h * w
            
            # 计算杂色（异常值）
            curr_ignore = curr_total - (curr_pos + curr_neg)

            # --- 更新全局统计 ---
            stats['pos_pixels'] += curr_pos
            stats['neg_pixels'] += curr_neg
            stats['ignore_pixels'] += curr_ignore
            stats['total_pixels'] += curr_total

            # 统计图片层级 (只要包含1个正样本像素，就算正样本图)
            if curr_pos > 0:
                stats['pos_imgs'] += 1
            else:
                stats['neg_imgs'] += 1

        self.print_report(stats)

    def print_report(self, stats):
        total_imgs = stats['pos_imgs'] + stats['neg_imgs']
        total_px = stats['total_pixels']
        
        # 防止除以零
        if total_imgs == 0 or total_px == 0:
            print("未检测到有效数据。")
            return

        table = PrettyTable()
        table.field_names = ["Category", "Count (Images)", "Img Ratio", "Count (Pixels)", "Pixel Ratio"]

        # 正样本行
        table.add_row([
            "Positive (Target)", 
            stats['pos_imgs'], 
            f"{stats['pos_imgs']/total_imgs:.2%}",
            stats['pos_pixels'],
            f"{stats['pos_pixels']/total_px:.2%}"
        ])

        # 负样本行
        table.add_row([
            "Negative (Background)", 
            stats['neg_imgs'], 
            f"{stats['neg_imgs']/total_imgs:.2%}",
            stats['neg_pixels'],
            f"{stats['neg_pixels']/total_px:.2%}"
        ])

        # 异常值行 (如果有)
        if stats['ignore_pixels'] > 0:
            table.add_row([
                "Ignored/Noise", 
                "-", 
                "-",
                stats['ignore_pixels'],
                f"{stats['ignore_pixels']/total_px:.2%}"
            ])

        print("\n=== 数据集分布统计报告 ===")
        print(table)
        
        # 给出训练建议
        pos_ratio = stats['pos_pixels'] / total_px
        if pos_ratio < 0.05:
            print("\n[分析建议] 正样本极其稀疏 (<5%)。")
            print("建议措施: 1. 使用 Weighted Cross Entropy Loss (增大正样本权重)")
            print("          2. 使用 Dice Loss 或 Focal Loss")
            print("          3. 在数据增强中增加包含正样本图片的采样概率 (Oversampling)")

if __name__ == "__main__":
    # 配置
    # 注意：Label通常是单通道灰度图。
    # 0 代表黑色背景，255 (或1) 代表白色前景
    PATH = r'E:\nets-dataset\water\train_samples\label'
    
    analyzer = LabelAnalyzer(
        folder_path=PATH, 
        ext='.png', 
        target_value=255,  # 如果你的label是 0/1 mask，这里改成 1
        background_value=0
    )
    analyzer.run()