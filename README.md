# rs-seg-toolkit

遥感语义分割数据预处理工具集。把大幅 GeoTIFF 切成训练 patch、做标签栅格化/重映射/统计、推理后拼回带坐标的大图——一条命令一件事，不强迫你接受全套流程。

## 这个仓库能解决什么

- 大 GeoTIFF 一张几个 GB，直接训练会爆内存：用 `clip_image.py` 切成固定尺寸的小 patch
- image 和 label 必须严格对齐：用 `clip_image.py` 的成对模式，同一套网格切两边
- patch 里有大片 nodata / 全黑 / 无前景：训练前过滤掉，省 epoch 时间
- 标签是 shapefile：用 `rasterize_shapefile.py` 转成跟参考影像同尺寸的 mask
- 推理输出是无坐标 PNG：用 `convert_png_to_geotiff.py` 挂上参考影像的坐标，QGIS/ArcGIS 能直接打开
- Windows 中文路径 / 多波段 / 16 位 / float32：常见坑都已经踩过

## 安装

GDAL 在 Windows 上推荐用 conda 装，pip 经常折腾不顺。

```bash
conda create -n rsseg python=3.9 -y
conda activate rsseg
conda install -c conda-forge gdal -y
pip install -r requirements.txt
```

`requirements.txt` 里也列了 `GDAL>=3.4`，conda 装好后 pip 会跳过。如果只有 pip，自己确认系统 GDAL 二进制版本后 `pip install GDAL==<version>`。

确认环境 OK：

```bash
python -c "from osgeo import gdal; import cv2, numpy; print('GDAL', gdal.__version__, '| cv2', cv2.__version__)"
```

## 60 秒试一下：把一张大图切成训练 patch

```bash
python clip_image.py --src data/images --dst output/patches --crop-size 1024
```

输出文件名是坐标制，比如 `scene_x000000_y000000.png`、`scene_x001024_y000000.png`，方便后面拼回去或者排查样本位置。

直接跑 `python clip_image.py` 不带参数会打印 DEMO 提示并使用脚本里的默认路径，方便你先看看脚本能不能跑通。

## 完整工作流（按需取用）

下面 6 步对应大多数遥感语义分割项目的数据准备流程。每一步都是独立命令，跳过任何一步都可以。

### 1. 矢量标签 → 栅格 mask（如果标签是 .shp）

```bash
python rasterize_shapefile.py ^
  --shp-dir data/label-shp ^
  --ref-img-dir data/images ^
  --save-dir data/label-png ^
  --attribute-field ID ^
  --target-value 1
```

`--target-value` 不传则保留属性原值；传 `1` 会把所有多边形烧成像素值 1（适合训练），传 `255` 适合可视化。要求 shp 与参考影像同名（`scene_a.shp` ↔ `scene_a.tif`）。

### 2. image / label 成对裁剪（核心步骤）

```bash
python clip_image.py ^
  --src-image data/images ^
  --src-label data/label-png ^
  --dst-image output/img_patches ^
  --dst-label output/lbl_patches ^
  --crop-size 1024 ^
  --overlap-ratio 0 ^
  --edge-policy drop ^
  --band-order auto ^
  --min-valid-ratio 0.5 ^
  --min-foreground-ratio 0.01
```

成对模式按文件名 stem 匹配 image/label，用同一套网格切，结果一一对应。常用过滤：

- `--min-valid-ratio 0.5`：patch 中有效像素（非 nodata）占比 < 50% 就丢掉
- `--min-foreground-ratio 0.01`：label patch 中前景占比 < 1% 就丢掉，避免大量纯背景样本

只裁单边（如只切影像、不切标签）用 `--src` / `--dst` 即可。

### 3. 标签像素值重映射（如果你的标签不是从 0 开始的连续整数）

```bash
python remap_labels.py ^
  --src output/lbl_patches ^
  --dst output/lbl_patches_remap ^
  --mapping "30:1,50:2,100:3" ^
  --unmapped set ^
  --unmapped-value 0
```

`--mapping` 格式 `old:new,old:new,...`。`--unmapped` 决定不在映射里的像素怎么处理：`keep` 保留原值，`set` 改为 `--unmapped-value`（典型是 0 当背景）。

### 4. 生成 train.txt / val.txt 列表

`split_dataset.py` 当前是配置区脚本，**需要打开文件改顶部参数后运行**：

```python
# split_dataset.py 底部
IMG_DIR = r'D:\path\to\images'
TXT_OUTPUT = r'D:\path\to\train_list.txt'
EXT = '.png'
generate_file_list(IMG_DIR, TXT_OUTPUT, EXT, shuffle=True)
```

也可以在自己的脚本里 `from split_dataset import generate_file_list, copy_files_from_list` 直接调用。

### 5. 数据集统计（归一化参数 + 类别分布）

均值/方差（支持任意波段、dtype，自动跳过 NoData）：

```bash
python compute_dataset_stats.py --path output/img_patches --ext .tif --nodata 0
```

类别分布（自动检测所有类别 ID，输出像素数 / 占比 / inverse-frequency 权重）：

```bash
python compute_label_distribution.py --path output/lbl_patches_remap --ext .png --classes auto
```

### 6. 推理结果挂坐标（GIS 软件可直接打开）

给一张无坐标的 PNG 结果挂上参考影像的地理坐标，输出 GeoTIFF：

```bash
python convert_png_to_geotiff.py ^
  --ref data/images/scene_a.tif ^
  --mask result/scene_a.png ^
  --out result/scene_a.tif
```

输出 GeoTIFF 在 QGIS / ArcGIS 中能直接对位叠加。尺寸不一致会拒绝写入，加 `--force` 才会强写（不推荐）。

> **关于 patch 拼回大图**：仓库里的 `stitch_images.py` 假设的命名约定是 `{base}_{index}.png`（连续数字序号），与 `clip_image.py` 当前的坐标制命名 `{base}_x{X}_y{Y}.png` **不兼容**。如果你的 patch 是用本仓库 `clip_image.py` 切的，建议自己按文件名里的 x/y 坐标做无重叠拼接（坐标制反推位置最直接），或在循环外把名字改回数字序号。`stitch_images.py` 适合处理外部已经按数字序号编号的切片。

## 关键参数怎么选（clip_image.py）

裁剪是整个流程最容易踩坑的一步，下面这几个参数值得花 1 分钟看懂。

### `--edge-policy` 大图末尾不齐怎么办

`width=200, crop=128, overlap=0` 时无法整齐切：

| 策略 | 输出位置 | 适用 |
| :--- | :--- | :--- |
| `append`（默认） | x=[0, 72]，最后一块贴右边沿，与上一块重叠 56 列 | 推理拼接、需要全图覆盖 |
| `drop` | x=[0]，丢掉末尾不齐条带 | **训练集推荐**，避免重复样本 |
| `pad` | x=[0, 128]，最后一块右侧补 0 到 crop_size | 训练集，保留边缘真实位置 |

### `--small-image` 整张图比 crop_size 还小

| 策略 | 行为 |
| :--- | :--- |
| `skip`（默认） | 跳过，不输出小 patch |
| `pad` | 右下补 0 到 crop_size 后输出 |

训练时 batch 要求所有样本同尺寸，`skip` 最安全；想保留每张图至少出一块用 `pad`。

### `--band-order` 三波段影像 RGB 还是 BGR

| 策略 | 行为 |
| :--- | :--- |
| `keep`（默认） | 完全不动通道顺序 |
| `auto` | 从 GDAL ColorInterpretation 自动检测 RGB/BGR；元数据缺失时回退 keep |
| `rgb` | 强制按"输入是 RGB"处理（写 PNG 时翻成 BGR） |
| `bgr` | 强制按"输入是 BGR"处理 |

`auto` 是大部分情况下最省心的选项。如果发现保存的 PNG 在 OpenCV 下颜色看着对、在 PIL/QGIS 下颜色反了，多半是这里的问题。

### 输出格式 `--dst-ext`

默认 `auto`：uint8 + 1/3 波段 → `.png`；其他（多波段、float32、16 位）自动落 GeoTIFF 保留 dtype 和坐标。也可以显式传 `.png` / `.tif` 强制格式（不兼容会 ValueError）。

## 脚本速查表

| 脚本 | 一句话作用 |
| :--- | :--- |
| `rasterize_shapefile.py` | shp 标签 → 与参考影像同尺寸的 PNG mask |
| `clip_image.py` | 大图滑窗裁剪成训练 patch；支持成对、过滤、边缘策略 |
| `remap_labels.py` | 标签像素值按 `{old:new}` 字典重映射 |
| `split_dataset.py` | 生成 train.txt / val.txt，按列表复制子集（配置区脚本） |
| `compute_dataset_stats.py` | 计算多波段均值/方差，自动跳过 NoData |
| `compute_label_distribution.py` | 多类标签像素分布、inverse-frequency 权重 |
| `stitch_images.py` | 按网格把切片拼回大图（命名约定 `{base}_{index}.png`，与 clip_image.py 当前坐标制命名不直接兼容） |
| `convert_png_to_geotiff.py` | 给无坐标 PNG 挂上参考影像的 GeoTransform |
| `visualize_training_metrics.py` | 从权重文件名解析训练指标，pyecharts 出交互式曲线 |
| `batch_rename_gis_files.py` | 按映射文件批量改名 .shp/.tif（带 sidecar 白名单、dry-run） |
| `batch_change_extension.py` | 批量改后缀名（仅重命名，不做格式转换） |

每个脚本都支持 `--help` 看完整参数；不带任何参数运行会打印 DEMO 提示。

## 常见问题

**Q: GDAL 装不上 / `from osgeo import gdal` 报 DLL load failed？**
Windows 上 99% 是 GDAL 二进制 + Python 版本不匹配。推荐 `conda install -c conda-forge gdal`，别用 pip。

**Q: 中文路径下 cv2 / GDAL 读图失败？**
仓库里的 `utils.imread_unchanged` / `utils.imwrite_safe` / `utils.gdal_open` 都做过中文路径包装，所有脚本默认走这套。但 GDAL 对极端路径（emoji、超长路径）仍可能挂；生产建议英文路径。

**Q: 为什么 patch 文件名是 `_x000128_y000256` 不是 `_1`、`_2`？**
坐标制命名能反推 patch 在原图的位置，方便排查样本、拼回大图、做可视化；也避免了 IO 失败时序号跳号导致的 image/label 错配。

**Q: `--overlap-ratio 0` 为什么 `append` 还是会出现重叠 patch？**
图宽不能被 crop_size 整除时，最后一块从右边沿往回数，会跟倒数第二块重叠几十像素。这是 `append` 保证全图覆盖的代价。训练数据集请用 `--edge-policy drop`。

**Q: 输出是 PNG 还是 GeoTIFF？**
`--dst-ext auto`（默认）下，uint8 + 1/3 波段输出 PNG；多波段或 float32 / 16 位自动落 GeoTIFF 保留坐标和数值精度。强制要 PNG 但 dtype 不兼容时会直接报错，不会静默丢数据。

**Q: 我能只用其中一个脚本吗？**
能。每个脚本都是独立 CLI，没有互相 import 依赖（`utils/` 里的安全 IO 工具除外）。按需取用。

## 测试

```bash
python -m pytest tests -q
```

当前 89 个测试，覆盖 clip / stitch / remap / 统计 / 重命名 / 安全 IO 等核心路径。

## License

MIT License（详见 LICENSE 文件）。
