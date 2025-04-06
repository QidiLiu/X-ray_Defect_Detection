# X-ray Defect Detection

> Power Battery Detection (PBD) aims to judge whether the battery cell is OK or NG based on the number and overhang. Therefore, object counting and localization are necessary processing for PBD, which can provide accurate coordinate information for all anode and cathode endpoints[^1].

Dr. Zhao (赵骁骐) and his team proposed a solution for power battery detection (PBD) and released an associated dataset[^2], which is [open-sourced on GitHub](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD). I forked this repository to reproduce their algorithm and explore potential improvements to this solution.

The task takes an X-ray image as input and outputs the locations of all anodes and cathodes. The original (baseline) solution first crops the ROI (region of interest) from the input image, and then uses the ROI and a prompt image as inputs to MDCNet to obtain the anode and cathode output information. My reproduction focuses on the second stage, since the first stage is essentially an object detection task, which is relatively simple nowadays.

I organized different solutions into separate Git branches for better readability:

- **main**: Overview of this repository.
- **train-electrode-mdcnet**: Reproduction of original (baseline) solution, focusing on the electrode detection stage.
- **infer-electrode-mdcnet-trt**: Inference of electrode detection stage using TensorRT.

**Reference**:

[^1]: [GitHub - Xiaoqi-Zhao-DLUT/X-ray-PBD](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD)
[^2]: [Zhao, X., Pang, Y., Chen, Z., et al. (2024). Towards Automatic Power Battery Detection: New Challenge, Benchmark Dataset and Baseline. In CVPR.](https://arxiv.org/pdf/2312.02528v2.pdf)
