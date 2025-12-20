import os
import glob

def batch_rename_gis_files():
    # --- 配置区域 ---
    # 建议：路径使用 raw string (r"...") 或正斜杠
    list_txt_path = r'E:\Samples-水体\chengdu-1024\new.txt'
    target_folder = r'E:\Samples-水体\chengdu-1024\label-shp'
    
    # 检查路径是否存在
    if not os.path.exists(list_txt_path) or not os.path.exists(target_folder):
        print("错误：路径不存在，请检查配置。")
        return

    # 1. 读取新文件名列表
    with open(list_txt_path, 'r', encoding='utf-8') as f:
        # 过滤空行并去除换行符
        new_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"加载了 {len(new_names)} 个新文件名。")

    # 2. 获取目标文件夹中的所有文件
    all_files = os.listdir(target_folder)
    
    # 筛选并排序 (排序至关重要，保证与txt列表对应)
    tif_files = sorted([f for f in all_files if f.lower().endswith('.tif')])
    shp_files = sorted([f for f in all_files if f.lower().endswith('.shp')])

    # 3. 检查数量匹配
    if len(tif_files) > 0 and len(tif_files) != len(new_names):
        print(f"[警告] TIF文件数量({len(tif_files)}) 与 名称列表数量({len(new_names)}) 不一致！停止操作以防错误。")
        return
    
    if len(shp_files) > 0 and len(shp_files) != len(new_names):
        print(f"[警告] SHP文件数量({len(shp_files)}) 与 名称列表数量({len(new_names)}) 不一致！停止操作以防错误。")
        return

    # 4. 执行重命名 - TIF
    if tif_files:
        print("\n--- 开始重命名 TIF 文件 ---")
        for old_filename, new_base_name in zip(tif_files, new_names):
            old_path = os.path.join(target_folder, old_filename)
            new_path = os.path.join(target_folder, new_base_name + ".tif")
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_filename} -> {new_base_name}.tif")
            except OSError as e:
                print(f"Error renaming {old_filename}: {e}")

    # 5. 执行重命名 - SHP (包含副作用文件 .dbf, .shx 等)
    if shp_files:
        print("\n--- 开始重命名 SHP 及关联文件 ---")
        for old_filename, new_base_name in zip(shp_files, new_names):
            # 获取旧文件的基础名称（不含后缀），例如 "data_123.shp" -> "data_123"
            old_basename = os.path.splitext(old_filename)[0]
            
            # 查找所有同名文件 (data_123.shp, data_123.dbf, data_123.prj ...)
            # 使用 glob 匹配所有扩展名
            related_files = glob.glob(os.path.join(target_folder, f"{old_basename}.*"))
            
            for old_file_path in related_files:
                # 获取该文件的后缀 (.dbf)
                ext = os.path.splitext(old_file_path)[1]
                
                # 构建新路径
                new_file_name = new_base_name + ext
                new_file_path = os.path.join(target_folder, new_file_name)
                
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {os.path.basename(old_file_path)} -> {new_file_name}")
                except OSError as e:
                    print(f"Error renaming {os.path.basename(old_file_path)}: {e}")

if __name__ == '__main__':
    batch_rename_gis_files()