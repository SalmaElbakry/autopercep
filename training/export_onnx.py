import torch
from create_demo_video import load_model_from_ckpt
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader, QuantFormat, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process

import cv2
from pathlib import Path
import numpy as np
import onnxruntime as ort

def preprocess_to_nchw_float32(img_path, img_size):
    img_bgr = cv2.imread(str(img_path))
    img = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Example ImageNet normalization (adapt to your training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img[None, ...]   

class MyCalibData(CalibrationDataReader):
    def __init__(self, paths, img_size):
        self._it = iter(paths)
        self.img_size = img_size
        
    def get_next(self):
        try:
            path = next(self._it)
        except StopIteration:
            return None
        # load -> resize -> normalize -> NCHW float32
        x = preprocess_to_nchw_float32(path, self.img_size)  # implement this for your data
        return {"input": x}                   # key must match export input
    
def export_onnx(backbone, img_size=256, out_agents=4, in_path="/home/lab/Documents/picar/picar_ros2/training/checkpoints/best.ckpt", out_path="onnx_models/pi_vision_resnet18.onnx"):
    model = load_model_from_ckpt(in_path, backbone=backbone, out_agents=out_agents).cpu()
    model.eval()

    # Use the real input size you’ll use on the Pi (e.g., 224x224)
    dummy = torch.randn(1, 3, img_size, img_size)

    # Good defaults for CPU inference
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        opset_version=12,                 # 12–17 are commonly safe; 12 is very compatible
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}}  # allow variable batch
    )
    
def dynamic_quantization(input_path, out_path):
    # quantize_dynamic(
    #     model_input=input_path,
    #     model_output=out_path,
    #     weight_type=QuantType.QInt8
    # )
    quantize_dynamic(
        model_input=input_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"]   # <- NO Conv here
    )
        
def static_quantization(input_path, out_path, calib_data, img_size):
    calib = MyCalibData(calib_data, img_size)

    # quantize_static(
    #     model_input=input_path,
    #     model_output=out_path,
    #     calibration_data_reader=calib,
    #     weight_type=QuantType.QInt8,     # weights INT8
    #     activation_type=QuantType.QInt8  # activations INT8
    # )
    
    # quantize_static(
    #     model_input=input_path,
    #     model_output=out_path,
    #     calibration_data_reader=calib,
    #     activation_type=QuantType.QUInt8,   # activations = uint8
    #     weight_type=QuantType.QInt8,        # weights = int8 (per-channel OK)
    #     quant_format=QuantFormat.QOperator, # <- produces QLinearConv, QLinearMatMul, ...
    #     per_channel=True
    # )

    quantize_static(
        model_input=input_path,
        model_output=out_path,
        calibration_data_reader=calib,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        quant_format=QuantFormat.QOperator,   # -> QLinearConv, QLinearMatMul
        per_channel=True,
        calibrate_method=CalibrationMethod.Percentile,  # robust scales
    )
        
if __name__ == '__main__':
    img_size = 256
    backbones = ['resnet18', 'mobilenet_v3_large', 'mobilenet_v3_small']
    out_folder = "onnx_models_256"
    
    for backbone in backbones:
        in_path = f"/home/lab/Documents/picar/picar_ros2/training/checkpoints_256/best_{backbone}.ckpt"
        out_pre_path = f"{out_folder}/pi_vision_{backbone}_{img_size}.onnx"
        out_path = f"{out_folder}/pi_vision_{backbone}_{img_size}_preprocessed.onnx"
        dyn_path = f"{out_folder}/pi_vision_{backbone}_{img_size}_int8_dynamicQ.onnx"
        sta_path = f"{out_folder}/pi_vision_{backbone}_{img_size}_int8_staticQ.onnx"
        
        print(f"creating backbone: {backbone} onnx model")
        
        export_onnx(backbone, img_size, in_path=in_path, out_path=out_pre_path)
        
        print(f"pre-processing backbone: {backbone} onnx model")
        
        quant_pre_process(
            input_model_path=out_pre_path,
            output_model_path=out_path,
            auto_merge=True  # enables helpful graph merges/fusions
        )
        
        calib_folder = Path("/home/lab/Documents/picar/2025_train_data/quantization_data")
        calib_images = list(calib_folder.rglob("*.png"))
        
        print(f"creating backbone: {backbone} dynamic quantization model")
        dynamic_quantization(out_path, dyn_path)
        
        print(f"creating backbone: {backbone} static quantization model")
        static_quantization(out_path, sta_path, calib_images, img_size)