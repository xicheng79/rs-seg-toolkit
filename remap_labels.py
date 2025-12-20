import os
import cv2
import numpy as np
from tqdm import tqdm

def process_labels():
    # 使用 os.path 规范化路径，解决斜杠混用问题
    root = r"E:\Samples-Water\chengdu\label-png"
    new_root = r"E:\Samples-Water\chengdu\label-png-new"
    
    # 自动创建输出目录
    os.makedirs(new_root, exist_ok=True)
    
    # 获取文件列表
    files = os.listdir(root)
    
    # 定义读取和写入中文路径的辅助函数
    def cv2_imread_chinese(file_path):
        try:
            # np.fromfile 读取文件内容到内存，cv2.imdecode 解码
            return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"读取错误: {file_path} - {e}")
            return None

    def cv2_imwrite_chinese(save_path, img):
        try:
            # cv2.imencode 编码图像，tofile 写入文件
            cv2.imencode('.png', img)[1].tofile(save_path)
        except Exception as e:
            print(f"写入错误: {save_path} - {e}")

    print(f"开始处理，共 {len(files)} 个文件...")

    for img_file in tqdm(files):
        # 忽略大小写检查后缀
        if img_file.lower().endswith('.png'):
            file_path = os.path.join(root, img_file)
            save_path = os.path.join(new_root, img_file)
            
            # 读取图像
            img = cv2_imread_chinese(file_path)
            
            # 空值检查（防止读取失败导致 Crash）
            if img is None:
                print(f"\n警告: 无法读取文件 {img_file}，已跳过。")
                continue
            
            # -------------------------------------------------
            # 核心逻辑：像素值映射
            # -------------------------------------------------
            # 仅修改需要变动的值。假设背景是0，前景是30，需要转为1
            # 原地操作，节省内存
            
            # 也可以使用 np.where，或者掩码赋值
            # 逻辑：将像素值为 30 的改为 1
            img[img == 30] = 1
            
            # 如果需要确保其他杂乱值被清洗为0，可以使用 np.where
            # img = np.where(img == 30, 1, 0).astype(np.uint8) 
            # 上面这行会把除30以外的所有值都变成0。如果你确定图中只有0和30，用 img[img==30]=1 即可。
            # -------------------------------------------------

            # 保存图像
            cv2_imwrite_chinese(save_path, img)

if __name__ == '__main__':
    process_labels()