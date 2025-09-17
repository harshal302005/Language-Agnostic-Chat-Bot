# export.py
import torch
from transformers import MBartForConditionalGeneration
from onnx2tflite import onnx_converter  # From MPolaris/onnx2tflite

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
input = torch.randn(1, 512)  # Dummy input

# Export to ONNX
torch.onnx.export(model, input, "mbart.onnx", opset_version=11)

# Convert to TFLite (quantized for CPU)
onnx_converter(
    onnx_model_path="mbart.onnx",
    need_simplify=True,
    output_path="./",
    target_formats=['tflite'],
    int8_model=True,  # Quantize for efficiency
    int8_mean=[0.0], int8_std=[1.0],  # Adjust for your data
    image_root=None  # Or path to calibration images
)
