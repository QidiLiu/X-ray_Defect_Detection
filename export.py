import os

import torch
import torch.onnx

from model.MDCNet import MDCNet_fcn

# Change the paths according to your environment
trtexec_path = 'C:/Users/ben/Lib/TensorRT-10.9.0.34/bin/trtexec.exe' 
trtlib_dir = 'C:/Users/ben/Lib/TensorRT-10.9.0.34/lib'
pth_path = 'saved_model/MDCNet_seg.pth'
onnx_path = 'saved_model/MDCNet_seg.onnx'
engine_path = 'saved_model/MDCNet_seg.engine'

img_size = 352
dummy_input = torch.randn(1, 3, img_size, img_size)
dummy_anchor = torch.randn(1, 3, img_size, img_size)
model = MDCNet_fcn()
model.load_state_dict(torch.load(pth_path))
model.eval()

torch.onnx.export(
    model,
    (dummy_input, dummy_anchor),
    onnx_path,
    input_names=['in_img', 'in_anchor'],
    output_names=['out_mask'],
    opset_version=18 # Choose an opset version that is compatible with your TensorRT version
)

os.environ['PATH'] = trtlib_dir + os.pathsep + os.environ['PATH']
os.system(f"{trtexec_path} --onnx={onnx_path} --saveEngine={engine_path}")