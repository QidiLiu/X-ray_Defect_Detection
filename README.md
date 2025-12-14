# X-ray Defect Detection - train-electrode-keypointdiff

Diffusion solution, focusing on the electrode detection stage.

## 1. Prerequisite

### 1-1. Environment

Create a venv environment and activate it with the following command:
```bash
conda env create -f environment.yml
conda activate xray-defect-detection
```

### 1-2. Dataset

The trainset folder structure is as follows:
```bash
train_crop_data
├─img_crop               # 900 PNG images with 3 channels
├─neg_line_mask_crop     # 900 PNG masks with 1 channel
├─neg_point_mask_crop    # 900 PNG masks with 1 channel
├─pos_line_mask_crop     # 900 PNG masks with 1 channel
└─pos_point_mask_crop    # 900 PNG masks with 1 channel
```

Please see [GitHub - Xiaoqi-Zhao-DLUT/X-ray-PBD](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD) for more detail.

## 2. Train

```bash
python script/train.py
```

## 3. Test

````bash
python script/test.py
````

## 4. Export

````bash
python script/export.py
````
