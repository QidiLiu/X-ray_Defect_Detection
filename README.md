# X-ray Defect Detection - infer-electrode-mdcnet-trt

Inference speed test for the `MDCNet` model using TensorRT with RTX4060Ti.

## 1. Prerequisite

### 1-1. Environment

- CMake 4.0.0
- TensorRT 10.9.0.34
- OpenCV 4.11.0 (with CUDA support)

### 1-2. Dataset and Model

- Input test data: random image from [GitHub - Xiaoqi-Zhao-DLUT/X-ray-PBD](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD)
- Pretrained weights: you can train and export MDCNet model yourself (please see branch "train-electrode-mdcnet" for more detail) or download the pretrained weights from [GitHub - Xiaoqi-Zhao-DLUT/X-ray-PBD](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD) and convert it to TensorRT engine using `trtexec`.

## 2. Infer

Build the project using CMake, and then run the executable:
```bash
cd build/Release
.\infer-electrode-mdcnet-trt.exe --trt_model C:/Users/ben/Dev/X-ray_Defect_Detection/train-electrode-mdcnet/saved_model/MDCNet_seg.engine
```

## 3. Result

```bash
 ======================
 Avg time per sample:
 6.391 ms
 Batch size:
 1
 Avg FPS:
 156 fps
 ======================
```

**Acknowledgement**:

- The code in directory "third_party/tensorrt-cpp-api-6.0" is sourced from [GitHub - cyrusbehr/tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api), developed by Cyrus Behroozi. This component is used under its original MIT license.