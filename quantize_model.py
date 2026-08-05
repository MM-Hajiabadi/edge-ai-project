"""
کوانتیزاسیون دستی INT8 — بدون وابستگی به API خودکار PyTorch
خروجی: golden_weights.json با وزن‌ها + scale/zero_point برای Verilog
"""
import os
import numpy as np
import torch
import torch.nn as nn
from train_autoencoder import Autoencoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder_best.pth")

def quantize_tensor(tensor, bits=8):
    """کوانتیزاسیون دستی یک تنسور به INT8"""
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    
    # محاسبه scale و zero_point
    qmin = -(2 ** (bits - 1))  # -128
    qmax = 2 ** (bits - 1) - 1  # 127
    
    scale = (t_max - t_min) / (qmax - qmin)
    if scale == 0:
        scale = 1e-8
    zero_point = qmin - round(t_min / scale)
    zero_point = max(qmin, min(qmax, zero_point))
    
    # کوانتیزه کردن
    q_tensor = torch.round(tensor / scale) + zero_point
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    
    return q_tensor.to(torch.int8), scale, zero_point

def quantize():
    print("=" * 60)
    print("کوانتیزاسیون دستی INT8 (بدون API خودکار)")
    print("=" * 60)
    
    # 1. بارگذاری مدل Float32
    model = Autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print(f"✅ مدل Float32 بارگذاری شد")
    
    # 2. کوانتیزه کردن وزن‌های هر لایه
    weights = {}
    print(f"\nکوانتیزه‌سازی وزن‌ها:")
    for name, layer in model.named_children():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            w = layer.weight.detach()
            q_w, scale, zp = quantize_tensor(w)
            weights[name] = {
                "shape": list(w.shape),
                "int8_weights": q_w.flatten().tolist(),
                "scale": float(scale),
                "zero_point": int(zp),
                "bias": layer.bias.detach().tolist() if layer.bias is not None else None
            }
            print(f"  {name}: shape={list(w.shape)}, scale={scale:.6f}, zp={zp}")
    
    # 3. ذخیره به JSON
    import json
    with open(os.path.join(MODEL_DIR, "golden_weights.json"), "w") as f:
        json.dump(weights, f)
    print(f"\n✅ golden_weights.json ذخیره شد (برای پیاده‌سازی سخت‌افزاری)")
    
    # 4. ذخیره مدل کوانتیزه (به صورت dict)
    torch.save({k: v["int8_weights"] for k, v in weights.items()}, 
               os.path.join(MODEL_DIR, "autoencoder_int8.pth"))
    print(f"✅ مدل INT8 (وزن‌ها) ذخیره شد")
    print("=" * 60)

if __name__ == "__main__":
    quantize()
