# rs-seg-preprocessing

**rs-seg-preprocessing** is a lightweight collection of Python utilities designed to streamline the data processing workflow for **Remote Sensing Semantic Segmentation**.

Developed for research teams and small-scale deep learning projects, this toolkit addresses common pain points encountered when working with high-resolution satellite/aerial imagery (e.g., GeoTIFFs), including handling large image cropping, geospatial metadata preservation, and dataset statistical analysis.

## 🌟 Key Features

* **Geospatial Awareness**: seamless handling of `.tif` files using GDAL, preserving projection and transform information.
* **Large Image Support**: Efficient sliding-window cropping for massive remote sensing images to prevent memory overflows.
* **Workflow Oriented**: Covers the entire lifecycle from vector labeling (`.shp`) to model inference result reconstruction.
* **Robustness**: Optimized for Windows environments (supports non-ASCII/Chinese file paths).

## 📦 Dependencies

Ensure your environment has the following core libraries installed:

* `gdal` (Essential for .tif/.shp handling)
* `opencv-python`
* `numpy`
* `tqdm`
* `prettytable`
* `pyecharts` (For log visualization)

## 🗂️ Script Reference

The toolkit is organized by functionality. All scripts use snake_case naming.

### 1. Data Preprocessing (Before Training)

| Script Name | Function | Input | Output |
| :--- | :--- | :--- | :--- |
| **`rasterize_shapefile.py`** | Converts vector labels to raster masks. | `.shp`, Ref Image | `.png`/`.tif` Mask |
| **`clip_image.py`** | Crops large satellite images into small patches (e.g., 512x512) using a sliding window. Support overlap. | Large `.tif` | Image Patches |
| **`remap_labels.py`** | Remaps pixel values (e.g., converting 255 to 1 for training). | Label Images | Label Images |
| **`batch_change_extension.py`** | Batch renames file extensions (e.g., `.png` to `.tif`). | Folder | Folder |

### 2. Dataset Construction & Analysis

| Script Name | Function | Description |
| :--- | :--- | :--- |
| **`dataset_file_utils.py`** | **Split & Indexing** | Generates `train.txt`/`val.txt` lists and supports shuffling/copying subsets. |
| **`compute_dataset_stats.py`** | **Normalization** | Calculates the Mean and Std (RGB) of the dataset for input normalization. |
| **`compute_label_distribution.py`** | **Class Balance** | Analyzes the ratio of positive/negative samples to guide Loss function selection. |

### 3. Post-processing (After Inference)

| Script Name | Function | Input | Output |
| :--- | :--- | :--- | :--- |
| **`stitch_images.py`** | Merges small inference patches back into the original large image size. | Patch Folder | Large Image |
| **`convert_png_to_geotiff.py`** | Restores geospatial information (Projection/GeoTransform) to the prediction result. | Result `.png`, Ref `.tif` | GeoTIFF |

### 4. Utilities

| Script Name | Function | Description |
| :--- | :--- | :--- |
| **`visualize_training_metrics.py`** | **Log Visualization** | Parses training logs (from filenames) to plot interactive Loss/Accuracy curves. |
| **`batch_rename_gis_files.py`** | **Safe Renaming** | Safely renames `.shp` (and associated .dbf/.shx) or `.tif` files based on a list. |

---

## 🚀 Recommended Workflow

### Step 1: Data Preparation
If you start with Vector data (`.shp`):
1.  Run `rasterize_shapefile.py` to generate the Ground Truth masks.
2.  Run `clip_image.py` to crop both original images and masks into patches (e.g., 1024x1024).

### Step 2: Dataset Organization
1.  Run `remap_labels.py` if your label values need adjustment (e.g., 0/255 -> 0/1).
2.  Run `dataset_file_utils.py` to generate `train.txt` and `val.txt`.
3.  Run `compute_dataset_stats.py` to get normalization parameters for your config file.

### Step 3: Training & Monitoring
* Train your model using your preferred framework (PyTorch/TensorFlow).
* Use `visualize_training_metrics.py` to check training progress if your checkpoints save metrics in filenames.

### Step 4: Inference & Delivery
1.  Run inference on test patches.
2.  Run `stitch_images.py` to merge the results.
3.  Run `convert_png_to_geotiff.py` to give the results coordinates for display in GIS software (ArcGIS/QGIS).

## 📝 Notes for Team Members

* **Path Encoding**: The scripts have been patched to support Chinese characters in paths on Windows, but using English paths is still recommended to avoid edge cases with GDAL.
* **Memory**: `clip_image.py` uses block reading and is memory safe. However, `stitch_images.py` requires loading the full result into memory; ensure sufficient RAM for very large areas.
* **GeoTransform**: Always keep the original raw TIFF files. The `convert_png_to_geotiff.py` script strictly requires the original image to copy the coordinate system.

## License

Internal Research Use / MIT License