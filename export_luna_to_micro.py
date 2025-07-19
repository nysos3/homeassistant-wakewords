import torch
import ai_edge_torch
from heybuddy.wakeword import WakeWordMLPModel

# Load state dict from training
state_dict = torch.load("luna.pt", map_location="cpu")

# Reconstruct model and load weights
model = WakeWordMLPModel()
model.load_state_dict(state_dict)
model.eval()

# Create sample input (based on ONNX: [1, 16, 96])
sample_input = (torch.randn(1, 16, 96),)

# Convert to EdgeModel
edge_model = ai_edge_torch.convert(model, sample_input)

# Export as quantized TFLite-Micro model
edge_model.export("luna_micro/model.tflite")

