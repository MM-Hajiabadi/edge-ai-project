import json
from pathlib import Path
import numpy as np
import torch
from src.edge_ai_anomaly_detection.components.model_trainer import Autoencoder
from src.edge_ai_anomaly_detection.utils.common import save_json


def quantize_tensor(tensor, bits=8, percentile=99.9):
    
    t_flat = tensor.flatten()
    lo = torch.quantile(t_flat, (100-percentile)/100).item()
    hi = torch.quantile(t_flat, percentile/100).item()
    qmin = -(2**(bits-1)); qmax = 2**(bits-1)-1
    scale = (hi-lo)/(qmax-qmin)
    if scale == 0: scale = 1e-8
    zp = int(qmin - round(lo/scale))
    zp = max(qmin, min(qmax, zp))
    q = torch.clamp(torch.round(tensor/scale)+zp, qmin, qmax).to(torch.int8)
    return q, scale, zp


class Quantization:

    def __init__(self, config, model_path: Path, processed_dir: Path):
        
        self.config = config
        self.model_path = model_path
        self.processed_dir = processed_dir

    def run(self):

        model = Autoencoder()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        weights = {}
        for name, layer in model.named_children():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                w = layer.weight.detach()
                q_w, scale, zp = quantize_tensor(w, self.config.bits, self.config.percentile)
                weights[name] = {
                    "shape": list(w.shape),
                    "int8_weights": q_w.flatten().tolist(),
                    "scale": float(scale),
                    "zero_point": int(zp),
                    "bias": layer.bias.detach().tolist() if layer.bias is not None else None,
                }
                print(f"  {name}: shape={list(w.shape)}, scale={scale:.6f}, zp={zp}")

        self.config.golden_weights.parent.mkdir(parents=True, exist_ok=True)
        save_json(weights, self.config.golden_weights)
        print(f"✅ golden_weights.json → {self.config.golden_weights}")