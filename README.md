# X-ray Defect Detection - train-electrode-mdcnet

Reproduction of original (baseline) solution, focusing on the electrode detection stage.

## 1. Prerequisite

### 1-1. Environment

Create a conda environment and activate it with the following command:
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
python train.py --data_root ../_data/train_crop_data --anchor_path ../_data/train_crop_data/img_crop/10-20-11__NULL_1_2_sangdun-battery_separator_shadow_interference.png --epochs 80
```

## 3. Infer

Download the "Region_seg.pth" from [GitHub - Xiaoqi-Zhao-DLUT/X-ray-PBD](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD) and save it in saved_model, then modify the `utils/config.py` for testset dir and run:
````bash
python infer.py
````
