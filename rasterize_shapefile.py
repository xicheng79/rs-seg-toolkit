import os
import sys
from osgeo import gdal, ogr
from tqdm import tqdm

def rasterize_layer(shapefile_path, reference_raster_path, save_path, attribute_field="ID", target_value=None):
    """
    将 Shapefile 栅格化为与参考影像一致的图像
    
    :param shapefile_path: 输入矢量文件路径
    :param reference_raster_path: 参考栅格影像（用于提供尺寸和投影）
    :param save_path: 输出保存路径
    :param attribute_field: 矢量属性表中用于决定像素值的字段名
    :param target_value: (可选) 如果设定此值，忽略属性表字段，将所有多边形统一烧录为该值 (如 255)
    """
    
    # 1. 打开参考影像获取地理信息
    ref_ds = gdal.Open(reference_raster_path, gdal.GA_ReadOnly)
    if ref_ds is None:
        print(f"无法打开参考影像: {reference_raster_path}")
        return
    
    x_res = ref_ds.RasterXSize
    y_res = ref_ds.RasterYSize
    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    
    # 2. 打开矢量文件
    shp_ds = ogr.Open(shapefile_path)
    if shp_ds is None:
        print(f"无法打开矢量文件: {shapefile_path}")
        return
    layer = shp_ds.GetLayer()

    # 3. 确定输出格式驱动
    # 根据后缀判断驱动，防止 "名为png实为tif" 的错误
    ext = os.path.splitext(save_path)[1].lower()
    if ext == '.png':
        driver_name = 'PNG'
    elif ext in ['.tif', '.tiff']:
        driver_name = 'GTiff'
    else:
        driver_name = 'GTiff' # 默认兜底
    
    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        print(f"GDAL不支持该驱动: {driver_name}")
        return

    # 4. 创建目标栅格
    # PNG 驱动通常不支持 Create 方法直接创建复杂数据集，需要用 CreateCopy 或者 MemDriver 中转
    # 为了通用性，我们先在内存中创建，然后 CreateCopy 到目标文件
    mem_driver = gdal.GetDriverByName('MEM')
    target_ds = mem_driver.Create('', x_res, y_res, 1, gdal.GDT_Byte)
    
    # 设置地理信息 (注意：PNG格式往往不支持写入完整的地理坐标，伴随生成的 .aux.xml 或 .pgw 包含坐标)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    
    # 初始化背景色 (0 为黑色背景)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0) 

    # 5. 执行栅格化 (Rasterize)
    options = []
    
    if target_value is not None:
        # 模式A：强制指定值（例如把所有多边形都变成 255）
        # burn_values=[target_value]
        gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    else:
        # 模式B：使用属性表中的字段值
        options = [f"ATTRIBUTE={attribute_field}"]
        gdal.RasterizeLayer(target_ds, [1], layer, options=options)

    # 6. 保存到硬盘
    driver.CreateCopy(save_path, target_ds)
    
    # 清理资源
    target_ds = None
    ref_ds = None
    shp_ds = None

def main():
    # 配置 GDAL 中文支持 (现代环境通常不需要设为 NO，如有乱码可尝试置空或 YES)
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8") # 或 'CP936' 取决于你的shp编码
    ogr.RegisterAll()

    # --- 配置路径 ---
    shp_dir = r'E:\Samples-Water\chengdu-1024\label-shp'
    ref_img_dir = r'E:\Samples-Water\chengdu-1024\img-tif'
    save_dir = r'E:\Samples-Water\chengdu-1024\label-png'

    # 自动创建输出目录
    os.makedirs(save_dir, exist_ok=True)

    file_list = [f for f in os.listdir(shp_dir) if f.endswith('.shp')]

    print(f"发现 {len(file_list)} 个矢量文件，开始转换...")

    for file_name in tqdm(file_list):
        name_no_ext = os.path.splitext(file_name)[0]
        
        shp_path = os.path.join(shp_dir, file_name)
        
        # 假设参考影像名为同名 .tif
        ref_path = os.path.join(ref_img_dir, name_no_ext + '.tif')
        
        # 输出路径
        save_path = os.path.join(save_dir, name_no_ext + '.png')

        if not os.path.exists(ref_path):
            print(f"跳过: 找不到对应的参考影像 {ref_path}")
            continue

        # 执行转换
        # 如果你的 Shapefile ID=30，但你想在 png 里表现为 255 (白色)，
        # 可以设置 target_value=255。如果想保留 ID 原值，删掉 target_value 参数。
        rasterize_layer(
            shapefile_path=shp_path,
            reference_raster_path=ref_path,
            save_path=save_path,
            attribute_field="ID",
            # target_value=255 # <--- 如果需要强制转换颜色，请取消注释这行
        )

    print("转换完成。")

if __name__ == "__main__":
    main()